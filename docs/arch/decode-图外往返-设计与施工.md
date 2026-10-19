# decode 图外往返优化 —— 设计与施工(GPU server 大改面)

> 立项 2026-10-19。**本刀主战场是 `crates/backends/cuda`(GPU server)与
> `owl-iface::contract`(命令语义),不是模型层。**
> 背景实测见 `docs/arch/gguf-格式支持-设计与施工.md` §22.7/§22.8。

---

## 一、立案:数字先钉死

同一模型(Qwen3.8-27B-Q4_K_M)、同一张 3090 Ti、同一 `llm_speedtest std`:

| 臂 | decode | 每步 | GPU busy | 图外 |
|---|---:|---:|---:|---:|
| **图回放**(现役) | 36.5 t/s | 27.4 ms | **24.57 ms**(1477 发) | **2.8 ms** |
| eager(`no_graph`) | 13.1 t/s | 88.5 ms | ≈24.6 ms | **≈64 ms** |
| llama.cpp(参照,eager) | 41.2 t/s | 24.3 ms | ≈24.2 ms | ≈0 |

**目标(验收线)**

| 项 | 现状 | 目标 | 判据 |
|---|---:|---:|---|
| 图回放 decode | 36.5 t/s | **≥40.0 t/s** | std 五点三轮 @512/@1024 |
| eager decode | 13.1 t/s | **≥35 t/s** | 同上(`no_graph = true`) |
| `srv.wait.dtoh` | 2.64 ms/步 | **≤0.3 ms/步** | `/debug/metrics`(perf 构建) |
| `srv.wait.sync` | 3.90 ms/次 | **≤0.2 ms/次** | 同上 |
| 逐命令往返 | launch 19.4 / alloc 14.9 / free 15.8 µs | **≤5 µs** | 同上 |

---

## 二、问题建模:一步 decode 的三段时间

```
[客户端] --命令--> [server 线程] --CUDA API--> [GPU]
   (A) 往返            (B) 段内发射              (C) 执行
```

实测(32 步平均,perf 构建):

| 段 | 内容 | 现状/步 | 性质 |
|---|---|---:|---|
| (A) | `launch` 128 段 × 19.4 µs + `free`/`alloc` 各 128 × 15 µs | **6.4 ms** | **纯往返** |
| (A) | `sync` 1.2 次 × 3.90 ms | **4.6 ms** | 真排空 |
| (A) | `dtoh` 1 次 × 2.64 ms | **2.6 ms** | 真排空 + 往返 |
| (B) | `srv.op.*` 128 段 × 7-9 µs | 4.5 ms | 段内发射(应保留) |
| (B) | `srv.queue.*` 384 × 7 µs | 2.7 ms | 出队开销 |
| (C) | GPU | 24.6 ms | 只有这里该占时间 |

⇒ **三个病根**:

1. **P1 回执走了唤醒路径**:`launch`/`alloc`/`free` 文档写 “issue 即 resolve”,
   实测 15-19 µs/条(应该是 ~1-2 µs 的入队成本)。128 段/步 ⇒ 6.4 ms。
2. **P2 每步两次 GPU 排空**:`sync` 3.90 ms + `dtoh` 2.64 ms。它们串在
   “采样 → 下一步输入” 这条链上,**本该与下一步的图回放重叠**。
3. **P3 图外的输入槽写**:采样结果回来后要写下一步的 ids/pos/slots/lens/q_cu,
   再触发图回放 —— 目前是**串行、逐项**。

---

## 三、设计

### 3.1 总原则

> **把“上一 token 的结果回读”与“下一 token 的图回放”做成一条流水线**;
> **把“逐命令回执”降级为“批量 drain 一次唤醒”**。
> GPU server 只多一条语义(“延迟回执”),不引入第二套命令通道。

### 3.2 P2/P3:回读异步化 + 输入槽双缓冲(收益主项)

**现状**:`sample → dtoh(2.64 ms) → 写输入槽 → graph_launch` 全串行。

**目标流水线**(双缓冲 A/B):

```
step n:   graph_launch(n)  →  [异步] dtoh(n)  ─┐
step n+1:  写槽(n+1, 用 dtoh(n) 的结果) ←──────┘ 与 graph(n+1) 的发射并行
```

关键观察:**dtoh(n) 的结果只被 step n+1 的输入槽使用**,而 `graph_launch(n+1)`
的节点内容与它无关(槽内容在发射*前*写入即有效)。所以可以把“等 dtoh”推迟到
**写槽那一刻**,而不是 `dtoh` 调用处。

**接口改动**(`owl-iface::contract`,加 3 个方法,**不改现有语义**):

| 新方法 | 语义 | 备注 |
|---|---|---|
| `dtoh_async(b, out_buf) -> Ticket` | 发起到 host 缓冲的异步回读(流序,不排空) | `out_buf` = **pinned 双缓冲**(池化,2 个) |
| `ticket_ready(t) -> bool` | 非阻塞查询(设备侧事件查询) | 禁在捕获期 |
| `ticket_wait(t)` | 阻塞取(仅在真要读数据时) | 现在唯一的排空点 |

引擎侧:`exec` 把 `sample` 后的 `dtoh` 改成 `dtoh_async`,在**下一步写槽前**
`ticket_wait`。这一步把 2.64 ms 的排空从关键路径移到“与 GPU 并行”的位置。

⚠️ **不同步的边界**:采样 kernel 在图内 ⇒ dtoh 必须等图**该节点**完成
(用事件,不是整图 sync)。这就是要求 `graph_launch` 返回“图内可查询事件”的原因
(见 §3.4)。

### 3.3 P1:回执批量(fire-and-forget 真语义)

**现状**:每条 `launch`/`alloc`/`free` 都 `wait` 到 ack(15-19 µs)。而这三类的
文档语义就是“issue 即 resolve” ⇒ **回执不需要逐条等**。

**设计(两档,先做 a)**

- **(a) 客户端侧合并**:`GpuClient` 内部维护 `pending_acks: Vec<Ack>`,对
  `launch/alloc/free` 只**登记**不 await;在遇到**需要序的命令**
  (`sync`/`dtoh`/`memset`/`graph_launch`/捕获开始)或 `pending` 超 N 条时
  一次 drain:`for ack in drain { ack.send(Ok) }` —— 但 ack 由**server** 发,
  所以真正要做的是把“回执”变成**序号区间**:`Command::AckBarrier(upto)`。
- **(b) server 侧批量回执**:server 在段循环里把 `launch/alloc/free` 的 ack
  **攒到段末一次性 send**(每段 11 发 ⇒ 1 次唤醒),配合 (a) 的 barrier 语义。

落地后:128 段/步 ⇒ 128 次唤醒(而非 384+),单次成本回到 ~2 µs ⇒ **6.4 → ~1.0 ms/步**。

⚠️ **不许静默吞错**:`launch` 的真实失败(CUDA 错误)现在靠 `cuLaunchKernel`
返回值在 server 侧就返回 Err;批量档必须保留**逐条 Err 上报**(区间内任一失败 ⇒
该区间整体 Err 且带首错),不能只报最后一条。

### 3.4 图内事件(`graph_launch` 的可见性)

要让 `dtoh_async` 精确等到“图内采样节点完成”而不是整图,需要 `graph_launch`
支持在 replay 后取一个**图内事件**。两条路线:

| 路线 | 做法 | 代价 |
|---|---|---|
| **A(推荐)** | 采样节点留在**图外**(图只到 logits),dtoh 等 logits 那块的完成事件 | 少图内节点;采样开销 ~0.1 ms/步 → 可接受 |
| B | 图内插入 `cuGraphAddEventRecordNode`,replay 后 `cuEventQuery` 该节点 | 需要 graph 侧改 + 事件节点在每图一个;事件必须**每次 replay 重录**,CUDA 允许但语义要审 |

首版走 **A**:采样与 argmax 本来就在图外(`owl_argmax_f32idx_f16` 出现在 decode
步的图外),只是**回读**要异步化。

### 3.5 顺带(低风险、同批做)

- `srv.wait.memset_zero`:19 次/步 × 33 µs。清零本可并进 alloc 或延后到**首次写前**
  (块池已有 `note_zeroing` 事件账),目标是每步 0 次同步等待。
- `srv.queue.*` 7 µs × 384:server 侧出队可以**一次 drain 多命令**再逐条处理
  (现有循环已是 drain,但每次 `recv` 唤醒有成本 ⇒ 用 `try_recv` 循环)。

---

## 四、不变量(改动不许破)

| # | 不变量 | 判据 |
|---|---|---|
| I-1 | **命令序**:同一流的命令按入队序执行 | 现有 `srv.queue/op` 账保持;新增批量档不得跨序 |
| I-2 | **错误必达**:任何 CUDA 失败必须 Err 到调用方 | 批量档逐条 Err + 区间首错;负例测试(注入坏指针) |
| I-3 | **捕获期禁回执延迟**:捕获窗口内不得插入事件/查询 | 现有 `capture_stream()` 守卫扩展;门:捕获期调用即拒 |
| I-4 | **块账与 ack 同序**:`free` 的 ack 不得早于块真正可复用 | 现有 `graph_destroy` 排空口径不变 |
| I-5 | **pin 缓冲生命周期**:异步回读的目标缓冲由宿主持有到 `ticket_wait` | 双缓冲由 `exec` 池化;drop 时 `ticket_wait` 兜底 |

---

## 五、刀序(每刀独立可测)

| 刀 | 内容 | 交付判据 |
|---|---|---|
| **M0** | 观测面沉淀:`srv.wait.*` 默认编进 release(或 `[cuda] srv_timing` 时动态开) + `/debug/metrics` 加 `decode.step` 分相(dtoh/sync/槽写) | 账本可复现(本文件 §二 表) |
| **M1** | `dtoh_async` + `Ticket` + 双缓冲(**只改 contract + cuda server + exec 的采样回读**) | 图回放 **27.4 → ≤25.5 ms/步**;e2e 首 token 不变 |
| **M2** | 回执批量(客户端 barrier + server 段末合并 ack) | eager **88.5 → ≤35 ms/步**;`srv.wait.launch` ≤5 µs |
| **M3** | memset 清零延后 + 出队批量 | 每步 `srv.wait.memset_zero` = 0 |
| **M4** | 收口:graph 与 eager 两臂 std 五点三轮 + 账本入档 | 图 ≥40.0 t/s / eager ≥35 t/s |

---

## 六、影响面清单(**为什么说这刀大**)

| 层 | 文件/面 | 改动 |
|---|---|---|
| 契约 | `crates/backends/iface/src/contract.rs` | +3 方法(`dtoh_async`/`ticket_ready`/`ticket_wait`)、+1 不变量注释 |
| 命令 | `crates/backends/cuda/src/command.rs` | +`DtohAsync`/`AckBarrier`;`kind()` 词表扩 |
| server | `crates/backends/cuda/src/server/mod.rs`(出队/回执)、`server/harvest.rs`(dtoh 路径) | 回执合并 + drain 批量 |
| 状态 | `crates/backends/cuda/src/state.rs` | 事件池扩展(异步回读要独立事件,**不能复用 tail_event**) |
| 客户端 | `crates/backends/cuda/src/gpu_client.rs` | pending 回执队列 + barrier |
| 引擎 | `crates/engine/src/exec/*`(decode 步的采样回读与槽写) | 双缓冲 + 延迟 wait |
| 门 | `crates/backends/cuda/tests/*`、`engine/tests*` | 新门:回执批量序/错误必达/捕获期拒/双缓冲无撕裂 |

**风险最高的三处**(必须写负例门):
1. 批量回执把 **Err 吞掉**(I-2);
2. 异步回读与**捕获期**相撞(I-3);
3. 双缓冲下**上一 token 的 token id 被写进下一步槽**时的一步错位(静默错值)—— 用
   “同一 prompt 图/eager 两臂逐 token 文本相同”当门。

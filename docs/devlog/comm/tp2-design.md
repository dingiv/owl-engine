# TP2 施工设计 —— 从落点到提交切分(自研集合库 v1)

> 2026-10-16 · 状态:**设计**(不动代码)· 上游:《tp2-prestudy.md》§7 裁决 / §8 序列 / §9 最小库草案;
> 《tp-overview.md》行业对照。
> 本卷回答"具体怎么实现":**改哪些文件、加什么命令/核/账目、协议不变量怎么证、门怎么挂、分几刀提交**。
> **结构(2026-10-19 收口)**:§0–§9 = 设计/契约/门表/不变量(本文);**§10 = 当前形态(收口,先读它)**;
> **逐轮实测账与已作废的中间结论 → `tp2-log.md`**(§10–§57 原文,只作取证与复现依据)。
> ⚠️ **全仓里所有 `tp2-design.md §N`(N ≥ 10)的引用,编号一律到 `tp2-log.md` 查**
> (实录卷保留了原编号与标题,可直接按号检索)。

---

## 0. 三条实况(设计的地基;全部代码级核实)

| # | 实况 | 出处 | 对设计的影响 |
|---|---|---|---|
| 1 | `Command` 枚举 = 设备演员的**唯一命令面**("新增能力 = 新变体 + actor 一臂");**捕获期只放行 `Launch`/`Alloc`** | `cuda/src/command.rs` 头注 + `state.rs graph_begin` | peer 授予/导入必须在**建图之前**完成 ⇒ staging **常驻**、per-AR 零分配(与预研 §5.3 一致) |
| 2 | iface 的 peer 三件套(`malloc_peer_shared`/`enable_peer_access`/`map_remote`)**全仓零实现**(无 `impl Device for`/`impl Pool for`);真实分配面 = `owl-cuda::state.rs` 的 `GpuCtx{blocks, ledger, capture, …}` | `grep -rn "impl Device for\|impl Pool for"` = 空 | v1 **不走 iface 池 trait**,直接扩 `Command` + `GpuCtx`;iface 面留作长期正名(§7 有裁决项) |
| 3 | cudarc 0.19.10 `driver/sys` 已含全套 VMM 符号(`cuMemAddressReserve:13158` / `cuMemCreate:13321` / `cuMemExportToShareableHandle:13371` / `cuMemImportFromShareableHandle:13663` / `cuMemMap:13679` / `cuMemSetAccess:14002`) | 本地 registry 源码 | 只扩 `ffi.rs` 白名单(7 条),**零新依赖**;A4 纪律要求该扩项单独登记 |

---

## 1. 落点总览(文件级)

```
【S1 能力探测 + PeerShared 分配/授予】
crates/shared/src/device.rs        +DeviceRow.bar1_bytes;+peer_caps(a,b)->PeerMechanism;+nvidia-smi 查询加 bar1.total
crates/backends/cuda/src/ffi.rs    +7 条 VMM 白名单(逐条注释:用途/准入)
crates/backends/cuda/src/command.rs +3 命令:PeerAlloc / PeerImport / PeerRelease
crates/backends/cuda/src/state.rs  +GpuCtx::peer_alloc/peer_import/peer_release + BlockEntry.peer(共享标记/远端 uuid)
                                     + 单笔 ≤192MB 硬门(超限响亮拒)+ peer_mapped_bytes 账
crates/backends/cuda/src/gpu_client.rs +3 门面方法(DeviceClient 契约面)
crates/backends/comm/src/peer.rs            (新)PeerPlan:探测→分档→对称握手→授予→账;唯一持有 PeerCaps/机制判定
crates/backends/comm/src/lib.rs             Topology 实化(UUID 序 + p2p_matrix + mechanism 矩阵)
crates/backends/iface/src/lib.rs   peer 死声明退役/降级(D1 派生动作;见 §7.0)

【S2 allreduce】
crates/kernels/cu/comm/ar_stage.cu   (新)本地→slot 拷贝 + __threadfence_system + counter++(带 credit 等待)
crates/kernels/cu/comm/ar_reduce.cu  (新)等 peer counter>seen → 累加 peer slot → seen++
crates/kernels/src/ops/comm.rs       (新)两张声明表(f16/bf16/f32 + stage/reduce)L3 事实面
crates/kernels/src/server/comm.rs    (新)CommRuntime 家族(assemble + run)
crates/kernels/src/server/mod.rs     +注册两行
crates/kernels/src/client/comm.rs    (新)PARSE_SLOTS + parse face(与声明表同票)
crates/models/src/ops.rs             +SemanticKernel::CommArStage/CommArReduce + driver 映射
crates/backends/comm/src/ar.rs                (新)AllReduce(plan, tp, bufs):chunk 计划 + 调用序列

【S3 allgather(留位)】crates/kernels/cu/comm/ag_*.cu + 同套声明/家族 + comm/ag.rs

【S4 引擎接线】
crates/models/src/plan/tp.rs(新)TpPlan{rank,world,split 面:哪些 Linear 行并行/头数/词表切}
crates/models/src/layers/linear.rs  行并行输出的 AR 调用点(显式传 &TpOpts)
crates/models/src/loader 侧(权重)          shard 感知装载(按行/列切后再 pack)
crates/engine/src/{engine,turn,scheduler,graph_fleet,state}.rs  双 actor:Vec<GpuClient> + 每 rank 图/KV/池
crates/engine/tests + gpu_tp2_equals_tp1_gate(TP2 ≡ TP1 逐 token)

【S5 prefill 重叠】ar_stage/ar_reduce 合并为单核流水版(pipelined ring)
【S6 3080 对装车】20G 预算账 + 长跑门 + 三线表
```

---

## 2. S1:能力探测与 PeerShared 分配(风险最高,先做)

### 2.1 分档(唯一判据 = 单笔上限)

```rust
// shared/src/device.rs(扩)
pub struct DeviceRow { pub ordinal: usize, pub name: String, pub bdf: String,
                       pub uuid: String, pub bar1_bytes: u64, /* 新 */ }

pub enum PeerMechanism { Static, Window, None }   // 3090类 / 3080类 / 禁用

/// 判据(设计裁决,替代一切经验规则):
///   bar1_a ≥ 2GiB && bar1_b ≥ 2GiB           → Static(enable 可用,单笔无墙)
///   否则两卡都能 P2P(ACCESS_SUPPORTED)     → Window(禁 enable;单笔 ≤ bar1 − 64MiB 且 ≤192MiB)
///   任一不可 P2P                             → None(拒启,响亮)
pub fn peer_mechanism(a: &DeviceRow, b: &DeviceRow) -> PeerMechanism;
```

- BAR1 来源:`nvidia-smi --query-gpu=index,uuid,bar1.total`(本仓已有 `parse_nvidia_smi_mem` 同族解析;礼仪启动已 shell out 枚举设备)⇒ **只加一列**;
- `Window` 档的单笔上限:`min(bar1_bytes − 64MiB 预留, 192MiB)`;**硬门写在 `GpuCtx::peer_alloc` 里**(不是调用方自觉);
- `Static` 档当前不启用 enable(A2.6 修正后:两条路都走 VMM 显式授予,机制一致 ⇒ **只有一档代码路径**;enable 只作为将来大 BAR1 卡的优化,留 trait 位)。

### 2.2 分配与授予(单进程双 actor 的对称握手)

```
rank A                                            rank B
────────                                          ────────
PeerAlloc{bytes=staging, name}                    PeerAlloc{bytes=staging, name}
  ├ 守卫: bytes ≤ cap(Window 档)                    ├ 同
  ├ cuMemAddressReserve(对齐 2MiB)                  │
  ├ n×cuMemCreate(粒度=设备最小,2MiB) + cuMemMap     │
  ├ cuMemSetAccess(self RW)                        │
  ├ cuMemExportToShareableHandle(POSIX_FD)         │
  └ → fd                                           │
        ────────────── fd 经 comm::PeerPlan 交换 ──────────────→
PeerImport{fd, bytes, peer_uuid}                  PeerImport{fd, bytes, peer_uuid}
  ├ cuMemImportFromShareableHandle                 │
  ├ cuMemAddressReserve + cuMemMap(对端物理页)      │
  ├ cuMemSetAccess(self R, peer R)                 │
  └ → alias 块句柄(Bytes)                          │
```

- **两侧都走同一条代码**(对称);`PeerPlan` 只负责:flutter 把 A 的 fd 交给 B、B 的 fd 交给 A,然后等两边的 `PeerImport` 回执;
- 授予粒度 = `cuMemCreate` 的块(2MiB)⇒ 天然满足 "≤192MB 分块" 纪律;单笔总量 = staging 大小(设计 2×32MiB = 64MiB/rank,预算账见 §6);
- **捕获期拒绝**:`PeerAlloc/PeerImport/PeerRelease` 三个命令在 `CaptureState.is_some()` 时返回结构化错误(与 `GraphBegin` 护栏同律)。

### 2.3 账本

| 账 | 落点 | 语义 |
|---|---|---|
| 本端分配 | `GpuCtx.blocks` 块条目(+ `peer_shared: true`)+ 现有 `bytes_alive` | 与普通块同账,只是带共享标记 |
| 对端映射 | `MemStats.peer_mapped_bytes` / `peer_mappings`(iface 已有声明) | 在 `GpuCtx` 侧新增计数并接到账报 |
| 上限门 | `PeerAlloc` 内 | 超 `cap` ⇒ `ModelError`(响亮,不静默降级) |

---

## 3. S2:allreduce(协议 + 核 + 调用序列)

### 3.1 staging 布局(每 rank 一块;peer 可读整个块)

```
偏移          内容
0x00..0x40   header: { u64 produced; u64 consumed; u32 slot_bytes; u32 n_slots; u32 magic; u32 rsv }
0x40..       slot[0]  (slot_bytes)
...          slot[1]  (slot_bytes)
```
- `produced` = 本端已写入的调用数(由 stage 核递增);
- `consumed` = 本端已消费的**对端**调用数(由 reduce 核递增)——**它是对端 stage 核读的 credit**(经 alias);
- v1:`n_slots = 2`;slot 大小 = min(32MiB, 预估最大 chunk);`magic` 防串块。

### 3.2 两个核(签名为设计值)

```c
// ar_stage:本地 P 的 chunk → 自己 staging 的 slot[produced % 2]
// 语义:credit 等待(防覆盖)→ 拷贝 → system fence → 发布 produced++
__global__ void owl_comm_ar_stage_f16(
    const __half* __restrict__ local,   // 本端 payload(部分和)
    unsigned char* __restrict__ own,    // 本端 staging(含 header)
    unsigned char* __restrict__ peer,   // 对端 staging(只读,取 consumed)
    unsigned long long bytes, unsigned int slot_bytes, unsigned int slab_off);

// ar_reduce:等 peer 的 produced > 本端 consumed → 累加 peer slot[consumed % 2] → 发布 consumed++
__global__ void owl_comm_ar_reduce_f16(
    __half* __restrict__ out,           // 本端 payload(就地累加:out += peer)
    unsigned char* __restrict__ own,    // 本端 staging(= consumed 的宿主)
    const unsigned char* __restrict__ peer,
    unsigned long long bytes, unsigned int slot_bytes, unsigned int slab_off);
```

**协议不变量(设计时必须能证)**:
- **I1 不覆盖未读**:stage 等待 `peer.consumed + n_slots > own.produced` 才写 slot;
- **I2 不读未写**:reduce 等待 `peer.produced > own.consumed`;
- **I3 相对量**:两侧只做**相对比较**(`>`)+ 设备内自增游标 ⇒ **图回放安全**(不烘绝对值,
  与 vLLM custom AR 同思路,但省掉每轮 host 复位);
- **I4 内存序**:生产者发布前 `__threadfence_system()`(写数据 → 可见于对端 → 再写 produced);
  消费者观察到 `produced` 后同 fence 再读数据(对 PCIe 语义,系统级 fence 是必要条件);
- **I5 死锁界**:两 rank 调用序列严格同构(计划期保证)+ credit 上限 2 ⇒ 自旋有界;
  自旋附**超时计数**(如 2^32 次)后置错误码并**不静默出结果**(宁可整轮失败,不可错值)。

### 3.3 chunk 计划(AR 的对外门面)

```rust
// comm/src/ar.rs
pub struct ArPlan { pub bytes: u64, pub dt: Dtype, pub chunk: u32, pub calls: u32 }
pub fn plan_ar(bytes: u64, dt: Dtype, slot_bytes: u32) -> ArPlan; // calls = ceil(bytes/slot)

pub fn allreduce(tp: &TpCtx, buf: &BytesView, plan: &ArPlan, out_kernels: &mut KernelSink) -> Result<()>;
```
- **v1**:`calls` 个 `(stage_i, reduce_i)` 核按序入流(简单、正确;重叠留给 S5);
- 每核参数含 `slab_off = i * chunk`(偏移量烘进图节点 ✓);
- decode(T=1,10KB)⇒ `calls = 1`,两次发射;prefill(80MB)⇒ `calls = 3`(32MiB slot)。

### 3.4 为什么不用 `cuMemcpyPeerAsync`

同流序不可捕获 + 无信号语义(要另配事件);自研核把"搬运+信号+归约"合到一处,
才拿到**单程穿越 + 零额外事件对象**(A2.2 InGraph 的前提)。

---

## 4. S4:引擎接线(TP2 的第一版形态)

| 面 | 现状 | TP2 改法 |
|---|---|---|
| 设备持有 | `Engine<GpuClient>` 单面(`DeviceSelector::Ordinal`) | `faces: Vec<GpuClient>`(world=2)+ `TpPlan`;每 face 自己的 actor/线程/图/KV 池(A2.6) |
| 装载 | 全量键 | **shard 感知**:列切(行块连 scales/zp 一起切)、行切(按 K 切,group 边界对齐 128)⇒ pack 前切,不落全量 |
| 层 | `Linear::forward(x, ctx)` | `forward(x, ctx, tp)`;行并行 GEMM 后接 `ar::allreduce` |
| 词表面 | lm_head 全量 | **v1 复制**(两 rank 各算全词表 ⇒ 无需 AG;探针只在 rank0 出) |
| 采样 | 单卡采样 | v1:两 rank 同 logits + 同种子 **greedy** ⇒ 无 token 广播(裁决:投机档再议) |
| KV | 单池 | 每 rank 池 = 自己的 KV 头(**2 头/rank** 全注意力;GDN 状态按 V 头 24/rank) |
| 图 | 单 rank 捕获 | 两 rank 各自捕获;**调用序列同构**由 TpPlan 保证(I5) |

**门**:`gpu_tp2_equals_tp1_gate`(同权重同 prompt,greedy 逐 token 一致;参照 `gpu_spec_identity_gate` 的 token 级硬判)。

---

## 5. 门(与预研 §9.5 同表;此处定文件与判据)

| # | 门 | 文件 | 判据 |
|---|---|---|---|
| G1 | host 参考对拍 | `crates/kernels/tests/comm_ar_parity.rs` | 2 rank 真机,随机数据,边界尺寸 0/1B/10KB/1MB/32MiB±1/100MB,三 dtype |
| G2 | 确定性 | 同上 | 同输入两跑**逐位**相同 |
| G3 | 图捕获 | 同上 | 捕获 replay ≡ eager |
| G4 | 192MB 墙 | `crates/backends/cuda/tests/peer_shared_gate.rs` | ≤cap 成功;cap+1 响亮拒(错误码断言) |
| G5 | 账本 | 同上 | `peer_mapped_bytes` 生命周期闭合、`bytes_alive` 无漂移 |
| G6 | 长跑 | 同上(或 S6 装车门) | ≥1000 次 AR 零 Xid |
| G7 | 性能 | 同上 | 64MiB 载荷 ≥12 GB/s;10KB 单次 ≤10µs |
| G8 | TP2 ≡ TP1 | `crates/engine/tests/`(e2e 档) | 逐 token 一致(见 §4) |

---

## 6. 显存/预算账(A5.1;设计值)

| 项 | 单 rank | 说明 |
|---|---|---|
| staging | 64 MiB | 2 slot × 32 MiB + header;VMM 常驻 |
| 对端映射 | 64 MiB | 只记账,不占本卡显存(对端 FB) |
| **Window 档总映射** | **128 MiB** | ≤ 192 MiB 墙 ✓(留 64MiB 裕量) |
| 27B 权重/rank | ~7.6 GB | int4 行/列切后 |
| lm_head(复制) | ~0.64 GB | v1 裁决 |
| KV/GDN 池 | 减半(按头切) | 27B:全注意力 2 KV 头/rank |
| 图/激活 | **不减半** | A5.1 分解表逐项重算(20G 卡的紧点) |

---

## 7. 裁决与待定(施工前必须关掉)

### 7.0 已裁决

| # | 裁决(2026-10-16,用户) | 含义 |
|---|---|---|
| **D1** | **直扩 `Command`**(不走 iface 池 trait) | v1 唯一规范 = `Command::PeerAlloc/PeerImport/PeerRelease` + `GpuCtx` 账房;派生动作见 §7.0.1。**不允许"声明面写一套、实现面写另一套"。** |

#### 7.0.1 C1 刀的退役清单(D1 派生;**已逐条核实**)

| 残留 | 实况 | 建议处置 |
|---|---|---|
| `crates/backends/iface/src/lib.rs`(312 行)的 A6 声明面 | `Backend`/`Pool`/`Device` traits + `PoolId/PoolKind/PoolConfig/PoolUsage/MemStats/MemPhase/DeviceDesc` + peer 三件套;**全仓零实现零调用**:`PoolConfig`/`PoolUsage`/`MemStats`/`malloc_peer_shared`/`enable_peer_access`/`map_remote` 全部 **0 引用**,无 `impl Device for`/`impl Pool for`,iface 自带 0 个测试 | **加头注降级为"历史设计记录(非规范)"**(其注释是 A6 的设计出处);头注写清 **当前规范 = `owl-cuda::command` + `GpuCtx` 账房** |
| `crates/backends/utils`(`owl-backend-utils`) | **不在 workspace members 里**(根本不参与编译),且引用**已退役**的 `CudaDevice`(`shared/src/testkit/mod.rs:12` 自述:"旧治理世界 CudaDevice/CudaPool …随声明式重构退役") | **删或归档 本地归档区** —— 已是坏引用死码 |
| `crates/backends/rocm` | 仅 README 占位 | 留(charter:rocm = 未来厂商栈占位) | |

### 7.1 已关闭(2026-10-16,用户"关掉 D2-D6";默认值即裁决)

| # | 项 | 裁决(生效) |
|---|---|---|
| D2 | `PeerRelease` 语义 | 显式释放 + `Drop` 兜底(与块账房一致);`ids` 同时收分配块与别名块 |
| D3 | staging 大小 | **32MiB × 2 slot**(总映射 128MiB ≤ 192MiB 墙);C1 实装后先测 3080 档授予成功率 |
| D4 | 自旋超时后行为 | 置错误码 + 下次 host 同步**响亮失败**(A2.7 全有或全无) |
| D5 | 混插对(Static×Window) | v1 **拒**(`classify` 三态已实现;混插 ⇒ `Rejected`) |
| D6 | 退役清单处置力度 | 按 §7.0.1 三行建议(iface 降级头注 / `backends/utils` 删或归档 / rocm 留) |

---

## 8. 提交切分(每刀独立可验、可回退)

| 刀 | 内容 | 门 |
|---|---|---|
| C1 | S1:探测/分档 + 三个命令 + VMM 白名单 + 账本 + **iface 死声明退役(D1 派生)** | G4/G5 + 单测(分档表) |
| C2 | S2:两个核 + 声明表 + 家族 + parse 面 | op_tables / L3 门 / marlin_ffi_sig 同套纪律 |
| C3 | S2:AR 门面 + 三 dtype + chunk 计划 | G1/G2/G3 |
| C4 | S2:性能门 + 长跑 | G6/G7 |
| C5 | S4:双 actor + shard 装载 + 层接线 | G8(TP2≡TP1) |
| C6 | S5:prefill 流水单核 | prefill 增益 |
| C7 | S6:3080 对装车(三线表 + 总对比) | 三线表 |

---

## 9. 风险与回退

| 风险 | 概率 | 回退 |
|---|---|---|
| Window 档授予在 27B 规模下不稳(驱动边界) | 中 | C1 就是**先行探针**;失败则退回 SHM(SegmentBoundary)先跑通 |
| 图回放下的 credit 协议死锁 | 中 | 自旋超时 + 结构化失败;protocol 单测含"乱序/慢对端"注入 |
| 双 actor 重构面过大 | 高 | C5 单列一刀;先 TP2 只跑 **一条** 请求(不碰 scheduler 并发面) |
| 20G 卡装不下(权重+池+图) | 中 | §6 预算表逐项实测;必要时 262k→64k 或 v1 只在 3090 对验数值、3080 对验容量 |
| 驱动/内核升级掀桌 | 已知 | 外部存档:驱动补丁仓弹药链(四次教训);门里带驱动指纹记录 |

---
## 10. ★ 当前形态(收口;2026-10-19)

> ⚠️ **最新状态(2026-10-19 深夜,§59/§60)**:
> · **已落地并实测**:量化核向量化(decode 退化 −12.4%→−6.6%)· **AR fp8 载荷**(+6.8%,分档)·
>   P0.1/P0.2 逐核账 · P0.0 合并入主干(`1d0ed15`);
> · **1-stage(§35)的现状**:竞态**已修**(独立输出块,门 5/5 绿),但**在 27B 生产档从不触发**
>   —— `tp_role` 标在最外层 `SliceView`、而"落对称槽"的判据在 `Op::Call` 臂(§60.2 三证据)
>   ⇒ **本卷 §35 的"GEMM 直写对称槽"接线是死路**,不要在它上面继续投入;
> · **下一刀 = v1 两核融合**(不用对称槽):设计/改动清单/复现命令见
>   `tp2-roadmap.md §11`(会话交接面)。

> **本节是唯一的"现在是什么样、什么结论算数"入口。**
> 逐轮实测账、已被推翻的中间结论、每一轮的立案与自我更正 → **`tp2-log.md`**(§10–§57 原文)。

### 10.1 现役配置(生产默认)

| 项 | 值 | 开关 |
|---|---|---|
| AR 形态 | **v1 两段**(`ar_stage` 核 + `ar_reduce` 核;读对端槽)| 缺省 |
| 写腿变体(push)| 写对端槽;**prefill +4.0%**(§37)| `runtime.tp_push = true` |
| 融合核(1-stage)| ⚠️ **不可用**:L1 逐位门**间歇性失败**(§57.4)| `runtime.tp_one_stage = true`(需 `tp_ordinals`,与 `tp_push` 互斥;2026-10-21 起取代 `OWL_TP_1STAGE`) |
| 卡对 | 3090 Ti 对 = **ord 0/1**(`Static`,BAR1 32G)· 3080 对 = **ord 2/3**(`Window`,BAR1 256M)| flavor 的 `device_ordinal` |
| 链路 | **PCIe Gen3 x16**(D2H 13.19 / H2D 13.07 / P2P 单向 12.7 GB/s = 理论 15.75 的 81%)| — |

⚠️ **CUDA ordinal ≠ nvidia-smi 序**(CUDA 走 `FASTEST_FIRST`);把 CVD 传给 vLLM 会被它按
`PCI_BUS_ID` 重解释 ⇒ **判据 = 双 worker 启动日志的显存容量一致**。

### 10.2 门表(2026-10-19 复跑,11 个全绿;3080 对 ord 2/3)

`raw_sync_primitives_gate` · `peer_shared_gate` · `ar_transport_gate`(G1 11 case 三 dtype /
G2 两跑逐位同 / **G3 捕获 replay ≡ eager**)· `graph_multistream_gate`(**须 `--test-threads=1`**,
并发跑会 `STREAM_CAPTURE_INVALIDATED` = 测试卫生)· `graph_memcpy_gate` · `ar_stress_gate` ·
`tp_assemble_gate` · `tp_determinism_gate` · `tp_g8_gate`(L1 = **0 项不同**)· `tp_transport_bits_gate` ·
`ar_channel_bench`。

### 10.3 AR 的实测账(两个战场,别混)

| | **prefill**(512 token ⇒ 载荷 5.24 MB)| **decode**(1 token ⇒ **10.24 KB**)|
|---|---|---|
| 主导 | **β**(带宽)| **α**(固定成本;搬运只占 0.8µs)|
| AR 时延 | push **493.5 µs** / v1 590.6 µs(§53)| **20.58 µs/次**(图内真设备时延,§50.7)|
| 硬件地板 | 411 µs(= Gen3 x16 的 81%,**128B TLP 天花板**)| — |
| 份额 | AR = TP2 的 **19%**(123.4 µs/token)| **128 次/token ⇒ 2.63 ms = decode 的 11.2%** |
| 固定项 | 每档 ~50 µs 固定开销 | 栅栏 2.0 µs · **第二个核 9.3 µs** · 协议 **≈0**(旁路实测)|
| 映射税 | 读腿 VMM vs runtime = **3.8–5.9%**(写腿仅 1.5%)| — |

#### 10.3b ★★ decode 单步的**核账**(2026-10-19 §58;nsys 图模式逐核,3080 对,中位)

```
TP1 Σ=28413µs(836 核/步) → TP2 Σ=21034µs(1092 核/步;两卡差 <0.05ms)
TP2 步 = 11.47(减半的 marlin) + **8.8ms 不随 rank 缩的项** + 2.86(AR) + 2.38(host) = 23.4ms ✓
```

| 项 | TP1 | TP2 | 判语 |
|---|---:|---:|---|
| marlin(量化权重)| 21515 | **11473** | ✓ 唯一腰斩项 = TP 的全部收益 |
| **`gemvx` = lm_head(BF16)** | 3506 | **3506** | ✗ **复制**(每卡读全词表,726 GB/s 已顶带宽)| **刀①未落** |
| **AR**(stage+reduce)| — | **2860** | ✗ TP 的税(22.3µs/AR 对)| **刀②未落** |
| norm 两族(128 核)| 1561 | 1547 | ✓ **刀③已落**:1547 → **364**(2.99/2.69µs/核;旧 13.6/10.6)|
| **`owl_embed_f16`** | 288 | 288 | ✓ **刀④已落**:288 → **2.1**(几何 block 1→512 + 16B 向量拷贝)|
| attention / GDN / gemv_dual / 小核 | ~1543 | ~1519 | ✗ 不动 |
| **host 空隙** | 1.61ms | **2.38ms** | fill+launch 仅 0.2ms ⇒ **尾巴 ~2.1ms 未定谳** |

**⇒ decode 只买到 1.315× 的原因已闭合**:能切的切了,不缩的有 8.8ms(31%),再加 AR 2.86ms。
**五把刀(每把都有实测分母)**:**③④已落**(−1.41ms 实测;decode TP1 **33.57** / TP2 **45.07**,
TP2/TP1 **1.315→1.342**;§58.10);剩①lm_head 按词表切(−1.76ms / 数值中性)
②AR 瘦身(−1.4ms)⑤argmax 并行(0.02ms,受静态 smem=256 限)⑥host 尾巴 2.1ms(未定谳)。

### 10.4 比值与 1.7× 的路线

```
prefill:TP2/TP1 = **1.530×**(fresh A/B 2026-10-19 22:00 · 专属端口 8141;vLLM 同机 = 1.52×)
decode :TP2/TP1 = **1.342×**(33.57 → 45.07 t/s;旧 32.02/42.11 = 1.315×)
AR 全串行的算术上限 = 1.64×(⇒ prefill 已吃到上限的 ~93%)
```

| 路线 | 收益 | 代价 | 状态 |
|---|---|---|---|
| **PCIe Gen4**(若 BIOS 可开)| **1.71×** | 需人工(iKVM/BIOS 查 PCIe 档 + `sudo lspci` 看 MPS)| **待人工** |
| **NVLink 桥**(仅 3090 Ti 对)| ~**1.80×** | 采购 + 装车 | 待人工 |
| **MPS 128→256**(root 改 Device Control)| +1.6% | 需 root;与 Gen4 可叠加 | 待人工 |
| **fp8 交换**(e4m3,载荷减半 = AR **−46%**,实测)| 1.61 → **1.74×** | **f16 尾数 10→3(64× 粗化)**;需新数值判据 | 数值等价件已过门(§47),真传输未落 |

### 10.5 ★ 已封死的路(不要重开;每条都有实测出处)

> ⚠️ **本表是"当前判据",不是"永久判决"**:多数条款是**单轴 A/B × 单一形状 × 单一机器状态**下的读数。
> 逐条的范围与翻案条件(12 条)→ `tp2-roadmap.md` **§9 待翻案清单**;纪律见 §10.7 ㊱。

| 路 | 判决 | 出处 |
|---|---|---|
| CE 换通道 | 三段串行,净负 ⇒ 回退 | §32 |
| 图内跨流重叠写腿 | PCIe **写**与本地访存**完全串行**;读能重叠 92% | §41 |
| 读腿 97% 重叠可移植 | ❌ **形状特有**(长同质链);微批扫描下吞吐比卡 1.21–1.28× | §48/§49 |
| TBO / 微批 | 与 N 无关,总时间 = GEMM + PCIe(零重叠)⇒ 无地基 | §49.3 |
| vLLM 的 D 结构照搬 | ❌ 它的 barrier 把 flag **写向对端**;本引擎规则是 **header 留本端**,且 1-stage 载荷由**上游 GEMM 多线程**写,单线程 fence 背不住可见性(它的载荷是紧邻核的 CE 拷贝) | §57.3 |
| 1-stage 融合核 | ⚠️ **L1 间歇性失败(0 / 243755)** ⇒ §51 的"−4.7%"是在**不可用路径**上量的 | §57.4 |
| "VMM 读腿税 17–20%" | ❌ 跨程序相减的伪差;**同程序 A/B = 3.8–5.9%** | §50.8 |
| "decode 换结构可得 5.8%" | ❌ 修正为 **+1.2%**;差额在**核的同步面**(§56.5) | §56.4/56.5 |
| fp8 用在 decode | 载荷才 10KB ⇒ 减半只省 0.4µs | §53 |
| **"绕开 VMM(改用 runtime 映射)"** | ❌ **现役是写腿**(push):**写腿税 0.4–1.2% 全档免疫**;**读腿**才有 14–24% 的税(且 10KB 档≈0)⇒ 绕开 VMM 在 decode ≈ **0.01%**、prefill ≈ **0.2%** ⇒ 不值得(代价 = enable 语义 + BAR1 窗口纪律 + 两套代码路径)| §60 |
| **"API 成本贵"** | ❌ VMM 一次性 **1.47–1.89ms**(装配期;`cuMemSetAccess(peer)` 一发占 58%)与 runtime 1.73ms **同量级**;稳态零调用 | §60.3 |
| **"TP 对 decode 零收益 / 只买到 +1.6%"** | ❌ **错标签**:`42.14` 是 TP2-**v1** 臂(§35.3),真 TP1 = 32.02 ⇒ **1.315×**;缺口不在 AR(2.86ms)而在 **8.8ms 不缩项** | §58.2/§58.6 |

### 10.6 工具手册(全部在树,复现/取证用)

| 工具 | 用途 |
|---|---|
| `crates/backends/cuda/tests/ar_channel_bench.rs` | AR 微基准 + **图内 AR 回放**(`OWL_ARB_GRAPH=1`,量真设备时延)+ 旁路协议/单核臂 |
| `tools/ar-struct-probe.cu` | **四协议结构对比台**(现役 v1 形状 / CE+融合核 / release-acquire / 末次 relaxed;三尺寸 × 四几何;自带数值校验)|
| `tools/p2p-wr-tune.cu` | 写腿:存储策略 / 栅栏档 / 几何 / 尺寸扫描 |
| `tools/p2p-overlap-probe.cu` | 重叠归属:X(核级)/Y(错相)/V(微批扫描)/R(读腿) |
| `tools/vmm-peer-probe.cu` | 映射路径 A/B(runtime vs VMM) |
| `tools/tp-ab-std.sh` | 生产 std A/B 跑手(起服 → `llm_speedtest --flavor std` → 停服)|
| **`tools/nsys-decode-account.sh` + `nsys_decode_account.py`** | **图模式逐核分账**(nsys `--cuda-graph-trace=node`;不改代码/不改档位/可分卡 ⇒ decode 步的核表 + 两 rank 时间线)|
| `[probes] step_profile = true` | host 面的分相(每步 `fill+launch` / `token-dtoh` / 尾巴)|
| **`tools/vmm-mapping-tax-probe.cu`** | **映射路径成本账**(2026-10-19 §60):① API 逐步成本(VMM vs runtime);② **访问税尺寸阶梯**(10KB→32MiB)× 读/写 × 单边/双边;③ **导入访问位 A/B**(READ vs RW);④ 两条 AR 协议形状 |
| **`crates/kernels/tests/ln_elems_bench.rs`** | **小核延迟台**(norm/fused_add/embed/argmax:**数值对拍 + µs;M=1/M=32 双档**分辨 host/设备受限;几何扫描;`*` = 生产几何)|

⚠️ **同机多会话的端口纪律**(§58.10.7 ㉜):`tools/tp-ab-std.sh` 默认 8137,**与本机别的
agent 会话会互相击杀**(症状 = 客户端 `ConnectError` + 服务端零错 + 进程消失)。
并行跑时:`PORT=8141` **且**运行档里 `[runtime] bind = "127.0.0.1:8141"`;
脚本已加 preflight(端口被**别的 worktree**占用 ⇒ 响亮拒,不代为清理)。

### 10.7 纪律速查(全卷累积;详条见实录卷各节)

⑭ 单边探针不能预测双边生产 · ⑮ 回退也要复测 · ⑯a fork 在链起点 · ⑯b 判据与被测共享瓶颈时无效 ·
⑰ 结构更对≠更快 · ⑱ **门绿证明不了新路径被执行**(要计数器)· ⑲ 只换指针要查块间私有状态 ·
⑳ 通道类改动先过隔离基准再上生产 · ㉑ 负面结论也要有分母 · ㉒ 裸环境成立≠引擎里成立 ·
㉓ 拓宽"不可能出错"的路前先问谁会踩空 · ㉔ 静默 no-op 是延迟爆炸的定时器 · ㉕ 能力落地以"真生效"为判据 ·
㉖ 慢变量走"攒→集中 dump" · ㉗ **测速台不校验数值 = 永远发现不了"快但错"**(§57.4 现场)·
㉘ 读数必须标注机器状态(别的卡跑满会让 P2P 腰斩) · **㉜ 同机多会话共用端口 ⇒ 互相击杀**(症状 = 客户端 ConnectError +
服务端零错 + 进程消失;⇒ 测速用**专属端口**且配置面同步 `bind`,清理只收自己 worktree 的实例) ·
**㉝ 判据自己先自证**(期望值要从**被测输入**算;位型解码器必须跟 dtype 走 —— 两条都是本轮现场踩的假红)
· **逐核账要看「次数 × 单次」**(128 个 10µs 的核 = 1.3ms/步 = 6%;一 token 一线程的 embed = 288µs)
· **㉞ 映射路径的税只在「跳卡读」上,且随载荷**比例**放大**(10KB ~0% → 5MiB 14–24% → 32MiB 21–24%);
**写腿免疫(≤1.2%)** ⇒ **协议选型(写腿优先)比换映射路径大一个量级**(§60.5;push 的 −19% 里
有一部分就是"把跳卡读换成了跳卡写");
· **㉟ peer 导入占的是导入方的 BAR1 窗口,别名块必须显式 unmap**(不 unmap = 每轮泄载荷大小的窗口;
3080 的 256MiB 窗口三轮就能吃光 ⇒ `out of memory`;§60.6 现场自留)
· **㊱ 任何"封死/作废"结论必须带三个范围字段:①形状 ②机器状态 ③组合(其它变量当时的值)**。
缺任何一项,它只能当"当前配置下的读数",不能当判决 —— 否则换组合时后人会拿"范围外的旧读数"
当依据(错杀好路,或盲信已不成立的路)。**卷内现成反例**:§41 被写成通性 ⇒ §44 自翻;
"VMM 读腿税 3.8–5.9%" ⇒ §60 翻了 3 倍;"1.64× 上限"是模型(能抬上限的 S5 从未施工)。
⇒ 这类"范围内读数"的收容所 = `tp2-roadmap.md` **§9 待翻案清单(12 条)**,每条带"翻案条件 + 便宜重测"。

### 10.8 未决(按优先级)

> **执行面**(阶段划分 / 每刀的判据与量法 / 里程碑 / 进度勾选表)→
> **`tp2-roadmap.md`**(本目录,随库入库;2026-10-19 立;本地草稿区(本地台账)仍可用)。本节只列"还剩什么"。

1. **decode 剩下四把刀**(**§58.7/§58.10;每把都有实测分母**)——
   ①**lm_head 按词表切**(`TpSplit::ColParallel`;−1.76ms / +8%,**数值中性**,副产品每卡省 1.18 GiB)
   ②AR 同步面瘦身(−1.4ms / +6%)⑤`owl_argmax` 并行(需连静态 smem=256 一起改)
   ⑥host 尾巴 2.1ms(未定谳;fill+launch 仅 0.2ms);
   ✅ **③④已落**(§58.10:−1.41ms 实测,decode TP2 42.11→**45.07** / TP1 32.02→**33.57**);
2. **1-stage 的 L1 竞态**(§57.4)—— 它是"融合核"这条路的地基(②的前置);
3. **人工三条**:BIOS 查 PCIe 档 + `sudo lspci` 看 MPS(⇒ 1.71× / +1.6%);NVLink 桥(⇒ 1.80×);
4. **fp8 真传输**(数值等价件已过门;需先立**非饱和**的 AR 边界误差判据;prefill 主战场);
5. prefill 的核同步面瘦身(**20.58µs → ~9µs 量级**,§56.5)。

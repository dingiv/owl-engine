# TP2 施工设计 —— 从落点到提交切分(自研集合库 v1)

> 2026-10-16 · 状态:**设计**(不动代码)· 上游:《TP2-通信-预研.md》§7 裁决 / §8 序列 / §9 最小库草案;
> 《TP-实现方案综述.md》行业对照。
> 本卷回答"具体怎么实现":**改哪些文件、加什么命令/核/账目、协议不变量怎么证、门怎么挂、分几刀提交**。

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
| `crates/backends/utils`(`owl-backend-utils`) | **不在 workspace members 里**(根本不参与编译),且引用**已退役**的 `CudaDevice`(`shared/src/testkit/mod.rs:12` 自述:"旧治理世界 CudaDevice/CudaPool …随声明式重构退役") | **删或归档 `docs/archive/`** —— 已是坏引用死码 |
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
| 驱动/内核升级掀桌 | 已知 | p2p-build 弹药链(四次教训);门里带驱动指纹记录 |

---

## 10. 壳子落地记录(2026-10-16,本轮;**只落框架,不落执行臂**)

| 落点 | 内容 | 状态 |
|---|---|---|
| `crates/backends/comm/src/peer.rs`(新) | `PeerCaps` / `PeerMechanism` / `classify`(三态) / `per_alloc_limit` / `PeerPlan::plan` + 5 单测 | ✅ 纯逻辑完整;`PeerPlan::open` = 结构化桩 |
| `crates/backends/comm/src/ar.rs`(新) | `StagingLayout`(header 偏移 / slot 偏移 / 乒乓选择 / 总额度)+ `ArPlan` + `plan_ar` + `chunk()` + 5 单测 | ✅ 纯逻辑完整;`AllReduce::enqueue` = 结构化桩 |
| `crates/backends/comm/src/lib.rs` | `pub mod ar/peer`;`CommError` +3 变体(`PeerRejected` / `PeerAllocOverLimit` / `NotYetImplemented`);`Topology` + `mechanisms` 分档矩阵 + `from_caps` + 2 单测 | ✅ |
| `crates/backends/cuda/src/command.rs` | +3 命令(`PeerAlloc` / `PeerImport` / `PeerRelease`)+ `kind()` / `created_at()` 表项 | ✅ 线格式定型 |
| `crates/backends/cuda/src/server/peer.rs`(新) | 三个 handler = **结构化桩**(统一 loud `Msg`,注释写明 S1 要填的 VMM 序列与额度门) | ✅ 桩;执行臂留 S1 |
| `crates/backends/cuda/src/server/mod.rs` | `mod peer;` + 5 处匹配加臂(追踪 / 主派发 / Closing 拒 / 捕获拒名单 / 名字表) | ✅ |
| `crates/backends/cuda/src/gpu_client.rs` | +3 门面方法(`peer_alloc` / `peer_import` / `peer_release`),错误原样透传 | ✅ |

**本轮跑过的门**:
- `cargo test -p owl-comm` ⇒ **12/12**;
- `cargo check -p owl-cuda` ⇒ ✓;
- `cargo check --workspace --all-targets` ⇒ ✓(只余既存 warning);
- 顺带补:本 worktree 缺 `testdata/args/e2e-27b.toml`(gitignore 本地资产)⇒ 已从主树拷入(否则 server e2e 测试目标不编译 —— 与回归轮发现同一族坑)。

**纪律落地**:捕获期三个 `Peer*` 命令由 `Self::reject` catch-all 自动拒绝
(施工设计 §2.2 纪律 1:peer 授予必须在建图之前)。

**桩不会悄悄留成永久物**:`owl-comm` 里有两条**专门断言桩行为**的测试
(`peer::tests::open_is_loud_stub_not_silent` / `ar::tests::enqueue_is_loud_stub_not_silent`)
—— S1/S2 填真实现时,这两条必红 ⇒ 强制作者同步更新测试与文档(而不是把桩留在原地)。

**下一会话(C1 后半)的起手式**:
1. `handle_peer_alloc`:填 `cuMemAddressReserve → n×cuMemCreate+cuMemMap → cuMemSetAccess(self RW)
   → cuMemExportToShareableHandle`,并把**额度门**(BAR1−64MiB,硬上限 192MiB)写在这里 +
   块账房登记(`peer_shared` 标记)+ `peer_mapped_bytes`;
2. `handle_peer_import`:填 import → map → setAccess(对端 R)+ 别名块登记;
3. `handle_peer_release`:显式回收 + `Drop` 兜底;
4. 扩展 `ffi.rs` 白名单(7 条 VMM 符号,逐条注释);
5. 门:`G4`(墙)/`G5`(账本)+ 单测("超限响亮拒"的错误码断言)。

> ✅ **已完成(2026-10-16 晚)—— 见 §11**(白名单扩到 10 条:释放三件套
> `cuMemUnmap`/`cuMemRelease`/`cuMemAddressFree` 也必须在内)。

---

## 11. C1 后半落地记录(2026-10-16 晚;执行臂 + 门)

### 11.1 落地清单

| 落点 | 内容 | 门 |
|---|---|---|
| `crates/shared/src/device.rs` | **三个额度常量上移单源**(`PEER_ALLOC_HARD_CAP`/`WINDOW_BAR1_RESERVE`/`STATIC_BAR1_MIN`)—— comm 与 cuda 读同一份(避免 cuda→comm 反向依赖) | — |
| `crates/backends/comm/src/peer.rs` | 常量改 re-export(API 路径不变) | `owl-comm` 12/12 |
| `crates/backends/cuda/src/ffi.rs` | **VMM 白名单 10 条 + 类型 13 项**(A4 准入逐条登记:粒度查询/AddressReserve/Create/Map/SetAccess/Export/Import/Unmap/Release/AddressFree) | check |
| `crates/backends/cuda/src/vmm.rs`(新) | RAII 构建守卫(**失败逆序回滚**)+ `alloc_mapped`/`import_mapped`/`set_access_self`/`export_fd`/`release`/`fail_with_rollback`/`close_fd`(`File::from_raw_fd` 接管 ⇒ 零 libc 依赖) | 单测 2 + 真机 |
| `crates/backends/cuda/src/state.rs` | `Block::Peer(PeerBlock{Shared,Alias})`;`GpuCtx` +4 字段(`peer_alloc_cap`/`peer_shared_bytes`/`peer_alias_bytes`/`peer_mappings`)+ `peer_alloc`/`peer_import`/`peer_release`/`peer_json`/`peer_report`;**`GpuCtx::drop` 兜底**;`block_ptr` 支持 Peer 且**图租约登记扩到 Peer**;`free_blocks` 对 Peer = 响亮计数 no-op | G4/G5 |
| `crates/backends/cuda/src/server/peer.rs` | 三 handler 真实现 + **凭据线格式**(显式 28 B 头:magic `OWLP`/v1/mapped_bytes/reserved/fd) | cred 单测 2 |
| `crates/backends/cuda/src/server/mod.rs` | 账报追加 peer 段(文本行 + `LedgerJson."peer"` 对象) | `gpu_server` JSON 门仍绿 |
| `crates/backends/cuda/tests/peer_shared_gate.rs`(新) | **G4/G5 门**(真机双卡;env `OWL_PEER_TEST_DEV_A/B`,未设 = skip) | ✅ |
| `crates/backends/iface/src/lib.rs` | **A6 声明面降级头注(D1 派生)**——历史设计记录(非规范);peer 三件套行内注记 | check |

### 11.2 门结果(2026-10-16 晚;**3080 对 = CUDA ordinal 2/3**,均空闲)

```bash
OWL_PEER_TEST_DEV_A=2 OWL_PEER_TEST_DEV_B=3 \
  cargo test -p owl-cuda --test peer_shared_gate -- --nocapture
```

**全绿**(0.48s):G4a `PeerAlloc 64MiB` 成功(账 +64MiB;引擎 live +64MiB = 真实占 FB);
G4b `cap+1` 响亮拒(消息含"超单笔上限"+
数值)且**拒绝后账零漂移**;G4c 坏凭据(magic 篡改)响亮拒;**跨卡别名读回逐字节一致**
(A 写 f32×4 → D2D 到 peer 块 → `sync` → B 经别名 `dtoh`)—— **VMM 显式授予的跨卡通路成立**
(A2.6 "禁 enable" 姿势实测可用);alias 不占本卡 live;普通 `Free` 对 peer 块 = no-op(账不变);
G5 双方 `peer_release` 后三账归零 + live 归位;负例(重复释放 / 普通块走 peer_release)响亮拒。

**同轮回归**:`gpu_server` 契约 2/2(LedgerJson 加 `"peer"` 段后 JSON 合法性门/步骤 13 仍绿);
`owl-comm` 12/12;`owl-cuda --lib` 22/22;`cargo check --workspace --all-targets` ✓。

### 11.3 关键实现裁决(本轮定形)

1. **额度门在 server**(`GpuCtx::peer_alloc`):cap 默认 = `owl_shared::device::PEER_ALLOC_HARD_CAP`;
   诊断/门读 `LedgerJson."peer".cap_bytes`;
2. **凭据 = 28 B 显式头**;**同进程 fd 直传,所有权归分配侧**(导入侧绝不 `close` ——
   否则分配侧释放时双 close);跨进程形态(SCM_RIGHTS)不在 v1;
3. **`setAccess` 显式两步**(owner = self RW;alias = self R)且**不调 enable**;
4. **释放唯一口 = `PeerRelease`**:校验(全有或全无:DeadBlock / 非 Peer / **图租约钉住** ⇒ 拒)
   后执行(失败不短路、聚合上报);fd 由分配侧 close;
5. **图租约扩到 Peer**:`block_ptr` 在捕获窗登记 Peer 块;`peer_release` 拒收被钉块(否则图踩陈旧 VA);
6. **`GpuCtx::drop` 兜底**:遗留 peer 块按 `vmm::release` 收回 + WARN(不静默)。

### 11.4 遗留(明确不在本轮)

| 项 | 说明 | 归属 |
|---|---|---|
| `PeerPlan::open` 实现 | comm 侧编排入口;需先定"两侧 actor 引用"的接口形状(owl-comm 不依赖 owl-cuda)—— 真正的调用方是 S4,建议与双 actor 接线同批 | C1 尾 / S4 |
| `crates/backends/utils` 退役 | 不在 workspace members + 引用退役 `CudaDevice`(坏引用死码);处置 = 删或归档(D6 建议),涉及删目录 ⇒ 待用户点头 | C1 尾 |
| staging 尺寸实测 | 设计值 2×32 MiB;本轮门用 64 MiB 单笔(等价额度)验证授予可行性 | S2 |
| 累计额度硬门 | 现为**单笔**硬门 + 累计观测(`peer_json`);预算墙本就是"单笔"口径(§10.5 约束 1),累计门待 S2/S6 实测再定 | S2/S6 |

---

## 12. C2 落地记录(2026-10-16 深夜;S2 核 + 声明 + 家族 + parse)

### 12.1 落地清单

| 落点 | 内容 | 门 |
|---|---|---|
| `crates/kernels/cu/comm/ar_stage.cu`(新) | 三 dtype stage 核(f32/f16/bf16):单 block;credit 自旋(I1);**每线程 `__threadfence_system` → `__syncthreads` → thread0 发布**的发布律;超时置 `err=1`;失败路径用 shared 标志**不提前 return**(防 `__syncthreads` 死锁) | 签名互证 + 冒烟 |
| `crates/kernels/cu/comm/ar_reduce.cu`(新) | 三 dtype reduce 核:等 `peer.produced > own.consumed`(I2)+ acquire fence;**f32 中间精度累加**(与 elems/add 同律);超时 `err=2` | 同上 |
| `crates/kernels/src/family/comm.rs`(新) | 族身份 `comm_ar_stage` / `comm_ar_reduce`;资产 id + 六符号 + `BLOCK` + `AR_HDR_BYTES`(=64;三处同值注记) | build.rs 扫描 |
| `crates/kernels/src/ops/comm.rs`(新) | **六张声明表**(stage×3 + reduce×3);payload 槽 rank 0(阶数无关),staging 两槽 = U8 字节块 | op_tables 互证 |
| `crates/kernels/src/client/comm.rs`(新) | `PARSE_SLOTS` / `PARSE_SLOTS_REDUCE` + 双向同型 call(`parse`/`to_launch`;out_elems = 字节口径) | parse 锁 |
| `crates/kernels/src/server/comm.rs`(新) | 两族 runtime(三 dtype 分派 + **F2 尺寸自证** + 单 block 发射) | 冒烟 |
| 注册四行 | `family/mod.rs` / `ops/mod.rs`(+TABLES)/ `server/mod.rs`(+factories)/ `client/mod.rs` | check |
| 两处锁表 | ops 槽序锁(名字 + 类型签名,6 项)+ client parse 锁(cases/want,6 项) | 全绿 |
| `crates/backends/comm/src/ar.rs` | `+ hdr::ERR`(偏移 28)/ `err_code`(1/2/3)—— 与 `.cu` 同值 | 12/12 |

### 12.2 实现裁决(本轮定形)

1. **单 block(单写者游标)**:`produced`/`consumed` 各只有一个写者 —— v1 的
   前提;grid 由 runtime 自持 `(1,1,1)`(LaunchMsg 的 `(0,0,0)` 哨兵即此义)。
2. **发布律**:数据写 → 每线程 `__threadfence_system()` → `__syncthreads()`
   → thread0 fence 后写游标(与 CUDA 编程指南 threadFenceReduction 同构;
   fence 只覆盖本线程 ⇒ 必须全线程各自 fence)。
3. **自旋超时 5 s**(`%globaltimer`;前 4096 次紧旋,之后 `__nanosleep(200ns)`
   退避)→ 置 err 并放弃;**host 判据面 = `hdr::ERR`**。
4. **自环 = 合法配置**(`own == peer`)—— 这使单卡冒烟可判定,也是 C3 双 rank
   对拍的前置工具。
5. dtype 选择 = **线格式 dtype**(声明表已由 L3 校验;不另设参数)。

### 12.3 门结果(2026-10-16 深夜)

- `op_tables` **10/10**:表 ⇔ `.cu` 签名逐参一致(含名字级互证)+ R1
  (编出符号 ⊆ 声明集,46 源);
- `owl-kernels --lib` **79/79**:`ops::tests::slot_order_lock_family_ops`
  (槽序 + 类型签名双锁)/ `client::tests::parse_slots_are_exactly_declared`
  (含覆盖率门)/ `cu_sig::tests::every_declared_op_has_asset_and_symbols`;
- **`comm_ar_smoke` 真机冒烟(3090 空闲卡,CUDA ordinal 0)**:
  自环 2 轮 = 游标 1→2、payload 2×→4×、`err=0`;
  **超时负例** = 恰 5.000 s 返回、`err=2`、`consumed` 不发布、payload 逐位不变;
- `no_runtime_compile` 2/2(AOT 资产链);`owl-comm` 12/12;
  `cargo check --workspace --all-targets` ✓。

### 12.4 C2 未含(→ C3)

| 项 | 说明 |
|---|---|
| 双 rank 真机对拍(G1/G2/G3) | 需两 actor/两卡 + 跨卡别名(B 侧 `PeerImport`)—— 与 AR 门面同批 |
| `AllReduce::enqueue`(comm 侧门面) | 桩 → 真实现(发射序列 + chunk 计划接线) |
| `PeerPlan::open`(S1 尾) | 仍是桩;真正调用方是 S4 |
| models 映射(`SemanticKernel::Comm*`) | 层侧接线(S4) |

---

## 13. C3 落地记录(2026-10-16 深夜;AR 门面 + G1/G2/G3 双 rank 真机)

### 13.1 落地清单

| 落点 | 内容 | 门 |
|---|---|---|
| `crates/backends/comm/src/ar.rs` | `ArPhase` / `ArChunk` / **`AllReduce::steps`**(纯数据展开:第 i 块 = `stage_i → reduce_i`);桩 `enqueue` 退役(两侧同 plan ⇒ 序列同构 I5 由它保证) | 单测(展开律) |
| `crates/kernels/src/client/comm.rs` | **发射名按 dtype 选声明 entry**;`AR_STAGE`/`AR_REDUCE` 降为"族名(路由 id)"注释面 | parse 锁 |
| `crates/kernels/src/ops/comm.rs` | staging 两槽 rank 0 → **1**(L3 事实面硬要求,见 13.3②) | op_tables |
| `crates/backends/cuda/src/server/peer.rs` | peer 块返回 `rank = 1`(一维字节块语义) | S1 门回归 |
| `crates/backends/cuda/tests/comm_ar_parity.rs`(新) | **G1/G2/G3 双 rank 真机门**(env `OWL_PEER_TEST_DEV_A/B`) | ✅ |
| `crates/backends/cuda/Cargo.toml` | dev-dep `owl-comm`(chunk 计划单源,测试不重写布局算术) | — |

### 13.2 门结果(3080 对 = CUDA ordinal 2/3;全绿 6.13 s)

- **G1(host 参考对拍)8 case,逐字节一致**:
  `4B`/`10KB` × {f32, f16, bf16} + `1MB`/`32MiB`/`32MiB+1`(f32)。
  `32MiB+1` 同时证两件事:**跨 chunk(calls=2)** 与**非整元素尾部字节不动**;
- **G2(确定性)**:同输入两跑逐位相同;
- **G3(图捕获)**:捕获 → replay **两次** ≡ host 参考(相对量游标 I3 的图回放实证);
  额外的正面断言:**图存活时 `peer_release` 被图引用租约响亮拒收**(租约机制在双 actor 真机上生效),`graph_destroy` 后可释放。

### 13.3 途中定谳(实现/测试四条)

1. **族名 ≠ 发射名**:`NamedSlots` 按 `kernel.name` 回声明表取址 ⇒ 发射名必须**逐字**
   是声明 entry(`owl_comm_ar_stage_f32`);族名(`comm_ar_stage`)只是路由 id 的一部分
   (`registry.names() = id ∪ ops()` 两者都注册)。
2. **L3 事实面:块参 `rank ≥ 1`**(rank 0 = 未定形 ⇒ 拒):staging 两槽声明 rank 1、
   `peer_alloc/import` 返回 rank 1;payload 槽声明 rank 0(显式阶数无关,实际 rank 由调用方给)。
3. **panic 时内核仍在自旋 ⇒ 进程退出 SIGSEGV**(测试失败时的次生现象,非驱动缺陷):
   排查顺序 = 先修测试主错;崩溃现场保留"内核在飞"这一点供将来参考。
4. **池化 pinned 租约容量 ≥ 请求**:`slice_bytes_mut()` 是整个池化区域,
   写前必须 `[..n]` 截断(否则 `copy_from_slice` 长度不符 panic)。

### 13.4 C3 未含

| 项 | 归属 |
|---|---|
| G6(长跑 ≥1000 次 AR)/ G7(性能:64MiB ≥12 GB/s;10KB ≤10µs) | C4 |
| `PeerPlan::open` 编排(分档→握手→账,唯一持有 PeerCaps) | C1 尾 / S4 同批 |
| models 映射(`SemanticKernel::Comm*`)+ 引擎双 actor 接线 | S4 |

---

## 14. S4 侦察与分期(2026-10-16 深夜;开工第一刀已落)

> 背景:用户裁决"直接 S4 接线";同时 `crates/backends/comm` 已迁 `crates/backends/comm`
> (目录治理:通信与后端层同区)。本段是 S4 的**侦察结论 + 分刀计划**,
> 以及已落地的第一刀。

### 14.1 侦察结论(规模现实)

| 面 | 现状 | 双 actor 改造量 |
|---|---|---|
| `crates/engine`(9,953 行) | `Engine<GpuClient>` 单 face 贯穿:`engine.rs`(构造/装配)/ `graph_fleet.rs`(1,198 行:图队)/ `state.rs`(1,181 行:KV 池/GDN 格/快照)/ `exec.rs`(执行器)/ `scheduler.rs` / `prefix_cache.rs` / `session.rs` | **大**:每处 `client` 使用点都要变成 `faces[rank]`;额外引入"跨 rank 同步点"(AR 是**设备侧**同步,host 侧只需保证两侧发射序同构) |
| `crates/models`(12,910 行 layers) | `Linear::forward(xs, ctx)` 走 `TensorOps::call(SemanticKernel::…)` 声明图;权重经 `Loadable`/`LoaderOps` 由声明驱动装载(pack 在装载面) | **中**:切分要在 pack **之前**(group 对齐);AR 接入点 = 行并行层输出后追加 `TensorOps::call(CommAr*)` |
| `crates/kernels` | 家族臂 lower 模板已就绪(`lower_gemm` 家族臂直构 LaunchMsg);`driver::resolve` 是"语义 op → 发射名"唯一路由面 | **小**:AR 两臂的 driver 注册 |
| `crates/backends/comm`(已迁) | `TpPlan` 面未立(本刀落);`PeerPlan::open` 仍是桩 | 中:握手编排 + 账 |

**结论**:S4 不是"一刀",而是 **五个子刀**(每刀独立可验)。

### 14.2 分刀计划

| 刀 | 内容 | 落点 | 验收 |
|---|---|---|---|
| **S4a ✅ 已落** | **TP 计划骨架**:`TpPlan`(rank/world)+ 三种切分(Col/Row/Replicate)+ 切分原语(`take_rows/cols_f32`、`slice_last_dim_f32`)+ 整除硬门 | `crates/models/src/plan/tp.rs`(新)+ `plan/mod.rs` + `lib.rs` | 单测 **9/9**:区间铺满/互斥、拒非整除、原语与手算参考;**两条数学硬证据**(RowParallel 部分和相加 == 全量 `W@x`;ColParallel 行块拼接 == 全量) |
| **S4b ✅ 已落** | **AR 语义算子 + driver 注册**:`SemanticKernel::CommArStage/CommArReduce` → `driver::resolve` 按 dtype 拾取 `owl_comm_ar_*` entry(FOREIGN 哨兵);发射名**单源** = `family::comm::{stage_entry,reduce_entry}(dt)`(client 与 driver 共读;非法 dtype 响亮 panic) | `crates/models/src/ops.rs` + `crates/kernels/src/driver.rs` + `crates/kernels/src/family/comm.rs` + `client/comm.rs`(SIG/SIG_REDUCE) | 单测:`ids_resolve_to_foreign_families` **四点同票**(词表↔driver↔家族 sig;dt=F16 ⇒ `owl_comm_ar_stage_f16`);`owl-models --lib` 180/180 |
| S4c | **常驻 staging 注入 + 装载 shard 感知**:引擎注入 own/peer 两块(建图前完成 VMM 握手);`Loadable` 面加 `TpSplit`(切分在 pack 前;量化 group 对齐硬门) | `models/src/module.rs`(ForwardCtx/LoaderOps)`+ layers` | 切分装载与全量装载的 **packed 字节**参考对拍 |
| S4d | **Linear 接线**:行并行层输出后追加 AR 调用(需 S4b/S4c) | `models/src/layers/linear.rs` | 单层 TP2 ≡ TP1(数值;可先在 cuda 测试里手工构造) |
| S4e | **引擎双 actor**:`Engine::on_tp(Vec<GpuClient>)` + 每 rank 图/KV/池;单请求(不碰 scheduler 并发面) | `engine/src/{engine,graph_fleet,state,exec}.rs` | **G8**:`gpu_tp2_equals_tp1_gate`(逐 token 一致) |

### 14.3 S4a 已落细节(本刀)

- `TpPlan::new(rank, world)`:非法即拒;`single()` = 单卡退化;
- **切分语义(矩阵视角 [out, in])**:`ColParallel` 切 out(权重行,下游消费分片)、
  `RowParallel` 切 in(权重列,输出 = 部分和 ⇒ 需 AR)、`Replicate` 不切(norm/embedding);
- **整除硬门**:`dim % world != 0` ⇒ 结构化拒(不许 pad/静默降级);
- **切分原语是纯函数**(零设备/零格式知识)—— 量化 packed 权重的切分由装载面
  按**同款区间规则**做(group 对齐在装载面另立硬门);
- 单测包含**切分数学的两条端到端硬证据**(host 数值,零设备):
  RowParallel 两 rank 部分和相加 == 全量积;ColParallel 行块拼接 == 全量积。

### 14.4 S4b 已落细节(同夜)

- `SemanticKernel::CommArStage`(op_id `comm.ar_stage`)/ `CommArReduce`(`comm.ar_reduce`);
- `driver::resolve` 两行(dtype ⇒ 三实例之一;`FOREIGN` 哨兵 —— 几何由家族 runtime 自持);
- **发射名单源**:`family::comm::stage_entry(dt)` / `reduce_entry(dt)`(client 与 driver 共读,
  非法 dtype 响亮 panic —— 不静默降级);
- `client::comm` 新增 `SIG`(`T,O,T,sz,sz,sz`)/ `SIG_REDUCE`(`O,T,T,sz,sz,sz`),
  挂进 `models::ops::foreign_sig`(家族 sig 表);
- 验收:`owl-models --lib` **180/180**、`owl-kernels --lib` 79/79、`op_tables` 10/10。

**S4c 起手式**(下一刀):`ForwardCtx`/`LoaderOps` 加 `TpSplit` 面 + 引擎注入
staging 常驻块(建图前完成 `peer_alloc`/`peer_import` 握手)—— 关键约束:
**AR 的两个 staging 块必须在 `graph_begin` 之前入账**(捕获期拒 `Peer*` 命令,
见 §2.2 纪律 1)。

---

## 15. S4 实施方案(设计;2026-10-16 深夜 —— 切分不进结构体)

> 用户裁决(2026-10-16):"**不要把结构体切开** —— 如果库里这么多结构体
> 全都要感知切分,那就没法写了"。本段把这句话落成设计:**四个窄口**,
> 模型声明层(12,910 行)与全部层结构体**零改动**。

### 15.1 总原则

| 谁 | 是否感知切分 | 怎么做到 |
|---|---|---|
| `Linear`/`Attention`/`GDN`/`MLP`/`DecoderLayer` 等**层结构体** | **不感知**(零字段/零分支) | 它们本来就"从 spec 取维度";spec 切一次,层自动拿到分片尺寸 |
| `ModelSpec`(形状单源) | **一个纯函数** `sharded(plan)` | 见 §15.2 |
| 权重装载面 | **一张规则表**(数据) | 见 §15.3 |
| 图装配面(AR 注入) | **只有 `Linear::forward` 读 ctx 里的可选 TP 上下文**(3 行) | 见 §15.4 |
| 引擎状态池 | `ModelDims` 构造时套 plan | 见 §15.5 |

**唯一"结构体级"的改动 = `Linear` 里读一次 `ctx.tp`(不新增字段)**。
其余全是"在既有单源处变换一次"。

### 15.2 ① 形状:`ModelSpec::sharded(plan)`(✅ 已落)

- `hidden` **不切**(层间通道全宽;切它要在每层入口 allgather,不划算);
- `inter` 切;`full_heads=(hq,hkv,hd)` 切 `(hq,hkv)`(GQA 比例保持);`gdn_heads` 切 `(nk,nv)`;
- `vocab` 不切(v1 复制 lm_head;AG 词表切分是将来项);`hd`/`hk`/`hv` 不切;
- **整除硬门**:任一切分维不被 world 整除 ⇒ 拒(不许 pad)。

**27B × world=2 实测值**:`inter 17408→8704`、`hq 24→12`、`hkv 4→2`、
`nk 16→8`、`nv 48→24`;单测 4/4(含"分片拼接回全量"的几何闭合)。

### 15.3 ② 权重切分:规则表 + 源格式的**矩形切**

**源格式**(`formats/w4a16.rs` 头注,已是 mmap 懒物化):
`weight_packed` i32 `[out, in/8]`(LSB-first)、`weight_scale` bf16 `[out, in/128]`。

| 切法 | 源张量怎么切 | group 对齐 |
|---|---|---|
| **out 切**(行) | `packed` 行切、`scale` 行切 | **天然安全**(行与 group 无关) |
| **in 切**(列) | `packed` 列切(每列 = 8 个 in 元素)、`scale` 列切 | 必须切在 **128 边界**(group);`in/2` 是 128 的倍数时安全 |

⇒ **两种切都是"矩形切",且能在 mmap 读取时直接只读所需区间(零拷贝)**;
f16 直读权重(非量化层)同理按 `[out, in]` 矩形切。**pack 布局无需改动**。

**五类切分形态**(规则表的词汇):

| 形态 | 含义 | 需要 AR? |
|---|---|---|
| `Replicate` | 不切(norm/embed/lm_head) | 否 |
| `UniformCol(out)` | out 维等分(行切) | 否(下游消费分片) |
| `UniformRow(in)` | in 维等分(列切,group 对齐) | **是** |
| `SegmentedCol` | out 维**分段**切(融合权重) | 否 |
| `PerHead(out)` | 按头切的小向量(A_log/dt_bias) | 否 |

**27B 逐键清单(world=2;实施时按层代码逐条核实)**:

| 键模式 | 源 [out, in] | 形态 | 本 rank |
|---|---|---|---|
| `*.mlp.{gate,up}_proj` | [17408, 5120] | UniformCol | [8704, 5120] |
| `*.mlp.down_proj` | [5120, 17408] | **UniformRow** ⇒ AR | [5120, 8704] |
| `*.self_attn.q_proj` | [12288, 5120] | **SegmentedCol**(q 24×256 ⊕ gate 24×256 两段,各按头) | [6144, 5120](2×3072) |
| `*.self_attn.{k,v}_proj` | [1024, 5120] | UniformCol(4 头 → 2) | [512, 5120] |
| `*.self_attn.o_proj` | [5120, 6144] | **UniformRow**(in = 24×256) ⇒ AR | [5120, 3072] |
| `*.linear_attn.in_proj_qkv` | [10240, 5120] | **SegmentedCol**(q 2048 / k 2048 / v 6144 三段) | [5120, 5120](1024+1024+3072) |
| `*.linear_attn.in_proj_z` | [6144, 5120] | UniformCol(48 头 → 24) | [3072, 5120] |
| `*.linear_attn.{in_proj_a,in_proj_b}` | [16, 5120] | PerHead(16 → 8) | [8, 5120] |
| `*.linear_attn.out_proj` | [5120, 6144] | **UniformRow** ⇒ AR | [5120, 3072] |
| `*.linear_attn.conv1d` | [6144, 1, 4] | **SegmentedCol**(通道跟 qkv 分段) | 按通道切(实施核实) |
| `*.linear_attn.{A_log,dt_bias}` | [16] | PerHead | [8] |
| `*.linear_attn.norm` | [128] | Replicate(per-head-dim,不随头数变) | [128] |
| `*.self_attn.{q_norm,k_norm}` | [256] | Replicate(同上) | [256] |
| 各 `*layernorm*` / `model.norm` | [5120] | Replicate | [5120] |
| `embed_tokens` / `lm_head` | [248320, 5120] | Replicate(v1) | 全量 |

**规则表是数据**(✅ 已落 `plan::rules`)`fn rule_of(key: &str) -> WeightRule`(单点;
装载面与 AR 注入面**同读一张表**)+ `needs_ar(key)`(AR 注入判据)+
`SegKind::segments(spec)`(段长按 spec 展开 —— **表是静态数据,模型换规格不改表**)。

### 15.3b 分片协议与装载侧接线(下一刀;"一个函数"的形态)

**用户裁决(2026-10-16)**:"一个函数,传入当前 slice 的 id 和总 slice 数,自动算出
该切哪一块给你 —— 四块卡可以同时加载"。**正名**:这是**确定性分片协议**,
不是分布式共识 —— 规则是纯函数 ⇒ 每个 rank 独立算,**零协调、零通信、可并行**
(共识只在"有分歧要消解"时才需要,这里没有分歧)。

形态落地 = `TpPlan::new(slice_id, n_slices)` + `shard_range(dim)` / `segmented_ranges(&[段长])`
(✅ 已落;`TpPlan` 即通用分片协议,PP/EP 将来复用同一函数)。

**底层已就绪**(侦察结论,零新增):
- `LoaderCtx.shard`(并行度;前人留口"TP 随多卡立项");
- `WeightSource::{take_range, copy_native_into, convert_chunk_into_bytes}`
  —— **按元素区间读**,mmap 零拷贝;
- `Want{key, dtype, shape, sink}` + 装填循环按 CHUNK 推进(`off` = 源坐标)。

**缺的三件(下一刀)**:
1. `LoaderCtx.slice_id`(与 `shard` 配对);
2. 装填循环的**源区间替换**(把全量 `[0, n)` 换成本 slice 区间):
   - **行切**(out):连续区间 ⇒ `off` 起点 = `row_start × in`、`n` = 分片元素数(**一行改动**);
   - **列切**(in):stride 读(逐行取子区间)⇒ 加"stride 装填器"(或单键全量读后再切,单键 ≤ ~45MB);
   - **分段切**:多段行区间 ⇒ 段循环(区间清单来自 `segmented_ranges`);
3. 分片形状贯通:装载的目标 `shape` = 分片形状(层侧从 `ModelSpec::sharded` 派生,§15.2)。

**验收硬判据**:切分装载的**字节 == "先全量装载再切"**(纯数据搬运,零数值自由度)。

### 15.4 ③ AR 注入:`ForwardCtx.tp`(唯一的结构体级触点,3 行)

```rust
// module.rs(入口上下文;**不想用的层一行都不写**)
pub struct ForwardCtx<'a> { …, pub tp: Option<&'a TpCtx> }

// plan/tp.rs(或 module.rs)
pub struct TpCtx {
    pub plan: TpPlan,
    pub own: TensorOps,      // 本端 staging(引擎注入的常驻块叶子)
    pub peer: TensorOps,     // 对端别名(常驻块叶子)
    pub slot_bytes: u64,     // comm::ar::StagingLayout 的值
}

// layers/linear.rs(全部改动就这三行,不新增字段)
fn forward(&self, xs: &TensorOps, ctx: &ForwardCtx) -> TensorOps {
    let y = self.forward_inner(xs);
    match ctx.tp {
        Some(tp) if tp.needs_ar(self.w.key()) => tp.allreduce(y, ctx.tokens),  // RowParallel ⇒ 追加 AR
        _ => y,
    }
}
```

- `needs_ar(key)` = 规则表查询(§15.3 的 UniformRow 项);
- `allreduce(y, …)` = 按 `slot_bytes` 展开 chunk 计划(`comm::ar::AllReduce::steps` 同源),
  追加 `SemanticKernel::CommArStage/CommArReduce` 节点(S4b 已接线);
- **原地语义**:reduce 是 in-place(out += peer),与 GDN 的 state 读写同款先例;
  图的依赖序天然保证"stage → reduce → 消费者";
- **staging 常驻**:两块必须在 `graph_begin` 之前入账(捕获期拒 `Peer*` 命令 —— §2.2 纪律 1)。

### 15.5 ④ KV/状态池:`ModelDims` 套 plan

`engine/src/state.rs` 的 `ModelDims{hq,hkv,hd,nk,hk,nv,hv,hidden}` 已参数化 ——
构造时套同一 plan(与 §15.2 同源;KV 池按头切、GDN 格按 nv 切)。**零新概念**。

### 15.6 S4e 引擎双 actor(形态)

- `Engine::on_tp(Vec<GpuClient>)`:faces 列表 + 每 face 自己的 actor/线程;
- **每 rank 一份**图/KV/池(现有 `GraphFleet`/`StatePool` 按 rank 实例化);
- **host 侧只需保证两侧发射序同构**(AR 的同步在设备侧 —— 自旋协议);
- 单请求先行(不碰 scheduler 并发面);**G8 = TP2 ≡ TP1 逐 token**。

### 15.7 实施顺序与验收

| 序 | 内容 | 验收 |
|---|---|---|
| ① ✅ | `ModelSpec::sharded` | 单测 4/4(27B/0.8B、world=1/2/3、拼接回全量) |
| ② | `rule_of` 规则表 + 装载矩形切(mmapped 区间读) | 逐键切分与"先全量后切"**字节一致**(host 对拍) |
| ③ | `TpCtx` + `ForwardCtx.tp` + `Linear` 3 行 | 单层 RowParallel:TP2 ≡ TP1(数值) |
| ④ | `ModelDims` 套 plan | 池尺寸按头数变化、e2e 不炸 |
| ⑤ | 引擎双 actor | **G8** 逐 token 一致(整模) |

**②的"字节一致"验收是硬判据**(切分是纯数据搬运,零数值自由度)。

### 15.3c 装载面分片 —— 已落(2026-10-17)

| 落点 | 内容 |
|---|---|
| `LoaderCtx.slice_id` / `LoadOpts.shard+slice_id` | **确定性分片协议的入参**(`shard = 1` ⇒ 零开销退化) |
| `Want.segments` + `LoaderOps::with_segments` | **融合权重的段长事实**(q⊕gate / q⊕k⊕v)由构造者声明;怎么切仍归装载面 |
| `plan/span.rs`(新) | `SourceSpan{Rows, Cols, RowsSeg}` + `map`(纯算术:目标 `[off,len)` → 源读序列,连续段自动合并)+ `plan_span`(规则×shape×plan×段长 → 读计划 + 分片形状) |
| `formats/shard.rs`(新) | `build_shard_pass`(就地改写 `want.shape` 与 `DeviceRearrange{rows,cols}`)+ **`ShardedSource`**(把目标坐标系的读请求翻译成源坐标系的读序列;未进表的键直通) |
| `eval_load` 注入 | 一行 pass + 一层源包装 ⇒ **6 条装载路径零改动** |

**验收**:`formats::shard` 5/5 —— 行切/列切/**分段切**逐元素 **== "先全量后切"**
(内存源对拍;参考实现 = `plan::tp` 的切分原语)+ Replicate 直通 + `shard = 1` 零开销;
`owl-models` 199/199;`check --workspace --all-targets` 0 错。

**剩余(接线面)**:① 融合权重的构造者(GDN/Attention)在 `layout()` 里
`.with_segments(...)` 声明段长(3 行);② engine 用 `ModelSpec::sharded` 构造
分片 spec + 每 rank 各跑一次 `load_from_dir`(双 actor 装载)。

### 15.3d 装载接线 —— 已落(2026-10-17)

| 落点 | 内容 |
|---|---|
| `GDN/Attention::layout` | 融合权重的**段长事实**:`in_proj_qkvz` 四段(q\|k\|v\|z)、`conv1d` 三段(q\|k\|v)、`q_proj` 两段(value\|gate)—— 各 3 行 `.with_segments_for(...)` |
| `build_shard_pass` | **反推 + 硬门**:从层侧的分片形状反推源全量形状(等分切分的逆),`plan_span` 算读计划;若层侧(spec 切分)与装载侧(规则表切分)约定不一致 ⇒ **响亮拒**(胜过"装载出形状不对的块、forward 时才炸") |
| `LoaderOps::with_segments_for` | 定向段长标注(键尾匹配;唯一 `tail` 实现) |
| `engine::ModelLoader::slice(slice_id, shard)` | 生产口:`shard_spec`(形状单源)/`shard_opts`/`shard_ctx`;三处装载点接入 |
| `CpuFace::upload_pinned` | 按**声明 dtype** 解码(补 F16 分支)—— 旧实现一律按 f32 解码,会把 f16 权重块**只写一半元素**且值全错 |

**门**:`crates/engine/tests/tp_load_gate.rs`(真检查点 0.8B,CPU face **零 GPU**):
全量(`shard=1`)vs 分片(`0/2`)两个 manifest,**7 个抽样键逐元素一致** ——
覆盖 `UniformCol`(gate_proj)/`UniformRow`(down_proj、o_proj)/`SegmentedCol`
(q_proj QGate 两段、**in_proj_qkvz QKVZ 四段**)/`PerHead`(in_proj_a)/`Replicate`(layernorm);
**分片装载 41.7s vs 全量 68.2s**(1.6×:真只读了本片)。
运行:`OWL_0_8B_DIR=<模型目录> cargo test -p owl-engine --test tp_load_gate`(未设 = skip)。

**途中两个真缺陷(都已修)**:
1. **双重切分**:层侧 `ModelSpec::sharded` 已把 Want 建成分片形状,装载面又切一次 ⇒
   形状减半(门当场抓出:期望 [1792,1024] 实得 [896,1024])⇒ 改为"分片形状反推全量 +
   一致性硬门";
2. **CpuFace 的 f16 盲区**:它自己的注释写"f16 装载路径不走 CPU;若将来要,可加 F16
   解码分支" —— 本轮就是那个"将来"(分片装载门用 f16 权重 + CPU face)。

### 15.4b AR 注入 —— 已落(2026-10-17;形态修正:`Op::CommAllReduce`)

**设计原案是 `SemanticKernel::Call` 路线,实测否定**:解释器(`interpreters/eval.rs`)
对**每个 `Op::Call` 节点恒 `alloc_out_node`** —— 即装配器会给 AR 节点分配一个新输出块,
而 AR 的 `own`/`stage` 必须是**引擎注入的常驻 staging**。⇒ 改为**零分配多发射节点**:

| 件 | 内容 |
|---|---|
| `Op::CommAllReduce { own, peer, bytes, slot_bytes }` | 一节点**两发射**(stage → reduce,同流序);**输出 = 输入的别名**(payload 原地累加)—— 与 `Op::SlotWrite` 同族的"零分配"节点 |
| `TensorOps::comm_allreduce(..)` | 构造器(父 = payload;`Bytes` 全字段供线格式事实面) |
| `eval.rs` dispatch | 两次 `face.launch`(client::comm 的 `ArStageCall`/`ArReduceCall` 构造);`bytes > slot_bytes` ⇒ **响亮拒**(v1 单块;多块是 S5) |
| `reference.rs`(host 参考) | AR **响亮拒**(跨卡归约无 host 参考;TP 数值验证走真机门 G8) |
| `TpCtx { plan, own, peer, slot_bytes }` + `ForwardCtx.tp` | 引擎装配期注入;**唯一读者 = `Linear::forward` 3 行** |
| `Linear::forward` | `needs_ar(self.w.key())`(规则表单源)⇒ `tp.allreduce(&y)`;`ctx.tp = None` 时零分支代价;**零新增字段** |

**单测**(`linear::tp_injection_tests`):`down_proj`(行并行)⇒ 追加 AR(载荷字节 = 输出形状 × dtype);
`gate_proj`(列并行)⇒ 不加;`ctx.tp = None`(单卡)⇒ 不加;输出形状/dtype 保持(别名语义)。

**回归**:`owl-models` 200/200 · `owl-engine` 42/42 · `owl-comm` 12/12 · workspace check 0 错。

**剩余**:④ `ModelDims` 套 plan(KV/状态池按头切);⑤ 引擎双 actor(每 rank 图/KV/池)+ **G8**。

### 15.4c 感知下沉到解释器(2026-10-17 用户裁决;形态定稿)

> 用户:"这个东西能不能把它**下沉到这个解释器层**去做啊?我们上面这个 layer 层,
> 它最多就是**给一些标记**,在 new 函数里面去给一些标记之类的?"

**采纳**(且**净删机制**):

| | §15.4b(上一形态) | **§15.4c(定稿)** |
|---|---|---|
| layer | `forward` 读 `ctx.tp` + 构造 AR 节点 | `new` 算一次标记 + `forward` **一行纯透传** |
| 决策/插入 | layer(3 行) | **解释器**(求值尾部统一;一处) |
| 机制 | `Op::CommAllReduce` + `ForwardCtx.tp` | **两者皆删** |

落点:
- `TensorOps.tp_role: WeightRule`(**纯标注**;`with_tp_role`/`tp_role`);
- `Linear::new` 里 `tp_role: rule_of(key)`(**构造期算一次**—— 标记是权重键的
  函数,不需要手工给:规则表已能算);`forward` = `forward_inner(xs).with_tp_role(self.tp_role)`;
- **解释器**(`eval.rs`)求值尾部:节点的 `tp_role == UniformRow` 且
  `EvalCtx.tp = Some(..)` ⇒ 追加 `stage → reduce`(超 slot 响亮拒;输出 = 同块别名)。
  单卡(`tp = None`)⇒ 零分支代价;
- `TpCtx { plan, own, peer, slot_bytes }` 保留为**纯数据**,挂在 `EvalCtx.tp`
  (注入口与 ⑤ 引擎接线同批 —— 那时才有真实 staging 来源);
- 将来 PP/EP 或其它"层输出后插通信"的形态,**同一处**扩展(不再散到各 layer)。

单测:`new_marks_tp_role_from_rule_table`(down/o/out_proj ⇒ Row;gate/up ⇒ Col;
q_proj ⇒ SegQGate;in_proj_qkvz ⇒ SegQKVZ;in_proj_a ⇒ PerHead)+
`forward_passes_mark_through`(纯透传;形状不变)。

### 15.5c 形状判据收口到核的家(2026-10-17 用户裁决)

> 用户:"不要把它写成那个硬编码机制" —— 指 §15.5b 里 engine 侧的
> `matches!((nk,nv), (16,48) | (8,24))`(硬编码换皮)。

**正确形态**(项目已有先例:XQA 的 `paged_decode_ok(hd, page)` 谓词住在
`kernels::driver::attn`,engine 只消费):

| 落点 | 内容 |
|---|---|
| `kernels::family::gdn_chunked` | `HOST_SLOT_ARM_SHAPES: &[(nk, nv)]`(**验证清单,唯一产地**)+ `host_slot_arm_shape_ok(nk, nv)` 谓词 —— 与核同源、随核演进;新档(如 world=4 的 (4,12))必须**先过真机图档验证再入表** |
| `models::kernel` 垫子 | re-export(engine 无 `owl-kernels` 直接依赖;与 `gdn_table_bytes` 同款"核词汇从核的家来") |
| `engine::prefill_plan` | `gdn_host_slot_arm` 改为消费谓词 —— **engine 零数字** |

**概念澄清**(用户同段提到的"从 config 拿维度"):维度单源**已经**是
`ModelSpec`,而它由 `ModelSpec::from_ckpt(dir, …)` **从检查点 config.json
(优先)+ 形状兜底**探测 ⇒ "从 config 拿维度"这条本就成立;被收口的是
**实现知识**(哪些档位验证过),那不是 config 能给的。

单测:清单与谓词同源 + 未验证档 `(4,12)`/混插档不放行。

---

## 16. S4 ⑤-K1 落地记录(2026-10-17;eval 注入口 + rank 标记点)

> 用户裁决:①注入口 = **函数参数法**(AGENTS #10:依赖写在参数里);
> ③"每层带 rank 标记点" = **注入点打标**(layer 零改动)。

| 落点 | 内容 |
|---|---|
| `models/src/interpreters/eval.rs` | **7 个求值入口**带 `tp: Option<&TpCtx>`:`eval_ops_env` / `eval_ops_scoped_env` / `eval_ops_scoped_env_pool` / `eval_ops_multi` / `eval_ops_multi_env` / `eval_ops_multi_scoped_env` / `eval_segment`(piecewise **逐段**带上 —— 段内 AR 否则全丢)。EvalCtx 存 `tp.cloned()`(轻句柄) |
| `models/src/module.rs` | `TpCtx: Clone`(Bytes 小 POD;每图/每次求值各持一份) |
| **rank 标记点** | 注入处**决策点打标**(发射之前):静态 tag 计数 `tp.ar.r0`/`tp.ar.r1`(metrics;store 缺席 = 静默跳过,纯观测面)+ `log_debug!("tp", "AR 注入 rank=… node=… label=… bytes=… slot=…")`。层键不在节点上(layer 零改动的代价),故记 node/label/字节数 |
| `engine/src/graph_plan.rs` | `GraphPlan.tp: Option<TpCtx>` + `plan`/`plan_piecewise` 参数;eager/捕获/piecewise 三路求值全带 |
| `engine/src/graph_fleet.rs` / `prefill_plan.rs` | `FleetBoot.tp` / `BucketBoot.tp`(装配期烘进每张图/每个桶) |
| `engine/src/running.rs` + `exec/*` | `RunningEngine.tp`(执行面唯一注入口;K2/K3 填入);exec **全部**求值点透传 `self.tp.as_ref()` |
| 便利入口(`eval`/`eval_ops`/`eval_ops_scoped`/`eval_ops_tap`) | 签名不变(单卡/测试门面),内部传 `None` |

**门**(`owl-models --lib tp_injection_tests`,3 条):
`no_tp_means_no_injection`(tp = None 零发射)/ `tp_injects_ar_and_marks_rank`
(tp = Some ⇒ 真走到 comm 发射:CpuFace 在 **stage 核**上响亮拒,错误文本点名
`owl_comm_ar_stage`;且 `tp.ar.r1` 计数 ≥ 1)/ `payload_over_slot_is_loud`(超 slot 响亮拒)。

**回归**:`owl-models` 204/204 · `owl-engine` 45/45 · workspace check 0 错;
单卡 GPU e2e(`OWL_E2E_ONLY=two_turns`)**PASS**(boot_rank 重构后生产路无回归)。

---

## 17. S4 ⑤-K2 落地记录(2026-10-17;双 rank 装配面 + staging 握手)

| 落点 | 内容 |
|---|---|
| `engine/src/tp.rs`(新) | `TpStaging`(own/peer 双块 + `StagingLayout` 单源)/ `TpRank`(分片计划 + staging + `RankModel` + `RankDev`)/ `TpEngine`(装配 + 出口 + `shutdown`) |
| `TpEngine::assemble`(GpuClient 特化) | **探测** UUID(驱动枚举)→ BAR1(`nvidia-smi -q -d MEMORY`,BDF 键)→ P2P(`cuDeviceCanAccessPeer` **双向**)⇒ `PeerPlan::plan`(**混插/不支持 ⇒ 响亮拒**,D5)⇒ **显式能力事实**入通用装配 |
| 握手(建图之前) | `peer_alloc`(两侧)→ fd 交叉 → `peer_import`(两侧;所有权归分配侧)⇒ `TpCtx{plan, own, peer, slot_bytes}` |
| `assemble_with_staging`(通用) | 每 rank `boot_rank`(**唯实现**,与单卡 `Engine::run` 同一条);两侧 slot 布局一致性硬门;rank0 的 tokenizer 归 host(rank1 的**丢弃**) |
| `engine/src/engine.rs` | `Engine::run` 拆出 **`boot_rank(face, cfg, &RankModel, tp, t0) -> RankDev`**(`RankModel` = `LoadedModel` 去掉 tokenizer);`Engine::into_parts`/`device_ordinal`(TP 面) |
| `GraphFleet::close` / `PrefillBuckets::close` | **整族退休**(逐图 `GraphPlan::close`;失败仅告警 + 记首错不短路)—— 图租约钉住 staging ⇒ 必须先卸图再 `PeerRelease` |
| `cuda/ffi.rs` | +`peer_supported(a,b)`(`cuDeviceCanAccessPeer`;A4 登记第 11 条,证据 `cudarc 0.19.10 sys/mod.rs:9342`) |
| `shared/device.rs` | +`parse_nvidia_smi_bar1`(详查面 → BDF → bytes)+ `bdf_norm` 认 **8 位 domain**(`00000000:41:00.0`) |
| `comm` | **`PeerPlan::open` 桩退役**(+`CommError::NotYetImplemented` 变体退役)—— 策略归 comm,握手编排归 `owl-engine::tp`(只有那里有两侧 actor) |
| `engine/Cargo.toml` | +`owl-comm`(分档/布局单源) |

**门**:`crates/engine/tests/tp_assemble_gate.rs`(真机两卡 0.8B;未设 env = skip)

```bash
OWL_TP_TEST_DEV_A=0 OWL_TP_TEST_DEV_B=1 OWL_0_8B_DIR=<…/Qwen3.5-0.8B> \
  cargo test -p owl-engine --test tp_assemble_gate -- --nocapture
```

判据与实测(CUDA ordinal 0/1 = 3090 Ti 对;Static 档):

| # | 判据 | 实测 |
|---|---|---|
| 1 | 两侧分片 spec == 全量 spec 的一半(inter/hq/hkv/nk/nv);hidden/vocab 不切 | ✓ |
| 2 | staging 两侧同额;账 = 1 分配 + 1 别名,各自 alias == 对端 shared | ✓(账 66MiB = 粒度上取整) |
| 3 | **别名连通**:rank0 写 own → rank1 经 peer 读回逐字节等 | ✓(交叉接线未错位) |
| 4 | 双 rank 主会话图**真捕到**(staging 块烘进图) | ✓(slab 5 MiB ×2) |
| 5 | 收尾:卸图 + `PeerRelease` ⇒ peer 三账归零 | ✓ |
| 附 | AR 注入计数两 rank 相等且 > 0(模型对称 ⇒ 注入对称;标记面) | ✓(r0 = r1 = 96) |

**途中三个定谳**(都当场修):
1. `nvidia-smi --query-gpu=bar1.total` **不是合法字段**(实测 rc=2,615 驱动)⇒ 走
   `-q -d MEMORY` 详查面(按 BDF 对账;`--query-gpu` 无 BAR1 字段);
2. `bdf_norm` 对 **8 位 domain**(`00000000:41:00.0`,nvidia-smi 写法)误判 ⇒
   两源(驱动 `0000:41:00.0` / nvidia-smi `00000000:41:00.0`)归一修复;
3. VMM 账 = **粒度上取整**(64MiB+64B → 66MiB;min granularity 2MiB)⇒
   门按不变量断言(两侧对称 / ≥ 请求 / 超出 < 2MiB / alias == 对端 shared)。

**回归**:`owl-shared` 65/65 · `owl-comm` 11/11 · `owl-models` 204/204 ·
`owl-engine` 45/45 · workspace check 0 错。

**下一步(⑤-K3,驱动 + G8)**:双 rank 交替驱动(先入两队再同步 —— AR 设备侧自旋,
单边同步 = 死锁)⇒ `RunningEngine` 面的每 rank 束 + `EvalCtx.tp` 真注入口 ⇒
**G8:TP2 ≡ TP1 逐 token**(单请求;不碰 scheduler 并发面)。

---

## 18. S4 ⑤-K3 落地记录(2026-10-17;双 rank 驱动 —— 并发入队 / 收割分相)

> 本轮把 ⑤ 从"装配"推到"能跑":两 rank 在同一 turn 内**并发推进**(同一份设备侧
> 代码),AR 协议真跑通(游标配平、零 err);**G8(TP2 ≡ TP1 逐 token)仍红**,根因已
> 收敛到"embedding 输出 1 ULP 分歧"这一层(见 §18.5;下一刀)。

### 18.1 形态:rank 束 + 设备半 + 并发

| 件 | 内容 |
|---|---|
| `RankDev<D>`(K3 扩) | 一个 rank 的**全部**状态:模型件(`model`/`rope`/`spec`/`quant_plan`/`mtp_dir`)+ 运行态(`env`/`spec_rt`/`tp`/`dims`/`pool`/`graphs`)—— `Engine::run`(单卡)与 `TpEngine`(双 rank)共用 |
| `RunningEngine{ dev, peer }` | `dev` = rank0 束,`peer: Option<RankDev>` = rank1 束;**host 侧账本单一**(tokenizer/会话/turn/spec 决策) |
| **设备半**(移到 `impl RankDev`) | `begin_dev` / `decode_dev` / `snap_dev` / **`prefill_chunk_enqueue`**(+bucket 路/pf_* 探针/`prefill_harvest`)/ 草稿 `prefill_dflash_encode`·`prefill_extend`;`ChunkHost{ boot_seq, probes, knobs, tok, sid, bt_chain, gdn_slot, bank }` = **主机侧事实**(只读;`bank` 仅主 rank 给 ⇒ 草稿臂在 peer 路响亮拒) |
| **`join2`**(exec/mod.rs) | 手写两 future 并发器(引擎 lib 不依赖 async 运行时/组合器;`Box::pin` + `poll_fn`) |
| 驱动相 | `execute_begin` / `execute_prefill` / `execute_decode` = **两 rank 并发跑同一函数** + **主 rank 收割**(`harvest_prefill`);快照两 rank 并发 |

### 18.2 途中四个定谳(全部当场修)

1. **staging header 从未初始化**(★最重):`peer_alloc` 只给裸内存 ⇒ 核读 `magic=0`
   ⇒ `err=3` **静默空转**(AR 一次没生效,而 host 侧毫无报错——首跑两 rank logits
   差 20 就是这么来的)。修:`init_staging_header`(`upload_pinned` + `sync`)在
   **建图之前**落 64B header,magic/slot_bytes/n_slots 三件由 `owl_comm::ar` 单源;
   并加**装配后自证**(读 header:游标必须全 0、err = 0 —— 越界即响亮拒)。
2. **装配期 warmup 不得真跑 AR**:图 warmup 是"真跑一遍 eager",而双 rank 装配是
   顺序的 ⇒ 单边 AR 撞 5s 自旋超时 + 游标永久失步。修:`eval_current_dry()`(warmup/
   分段干跑**临时摘掉 `tp`**;真 AR 只在回放期)。
3. **入队半必须两 rank 并发**(★K3 的形态决定):host 在 eager prefill 里存在
   **设备等待点**(实测 `hidden_full` eval 单边 50s = 10×5s 超时;串行方案必死)⇒
   同一份设备侧代码在两 rank 束上并发跑(`join2`),"先入队后收割"的分相纪律由
   `prefill_harvest`/`decode` 的读点保证。
4. **`mfact` 改置值语义**(`counter_set`):双 rank 各写一次同名事实(AR 后两侧同值
   ⇒ 幂等);累加会让 63 位 checksum 当场溢出。

### 18.3 门结果(3080 对 = CUDA ordinal 2/3;3090 对当时被他会话占用)

| 门 | 结果 |
|---|---|
| `tp_assemble_gate`(K2) | ✅ 全绿(形状/账/别名连通/双图捕获/收尾三账归零) |
| **游标/err 自证**(装配后 + 每 turn 后) | ✅ 装配期 0/0/err0;**turn 后 produced==consumed=={48, 144}、err=0** |
| **两 rank 同值** | ✅ **逐位相同**(248320 项,0 项不同)—— AR 接线/协议**真通** |
| 单卡 GPU e2e(`two_turns`) | ✅ 无回归 |
| `owl-shared`/`owl-comm`/`owl-models`/`owl-engine` | ✅ 65/11/204/45 全绿 |
| **G8(TP2 ≡ TP1 逐 token)** | ❌ **红**:见 §18.5 |

### 18.4 TP v1 范围护栏(`TpEngine::assemble` 响亮拒)

投机草稿(`draft.depth != 0`)/ prefill 桶族(`prefill_bucket != Off`,桶出图尾自带
读数与分相纪律冲突)/ 读数类探针(`prefill_cksum`/`pf_fin_check`/`pf_fence`/
`dflash_probe`)。`pf_bisect`/`pf_stages` **放行**(只多一次同步;并发驱动下安全 ——
它们是逐层对拍的主工具)。

### 18.5 ★ G8 定谳(2026-10-17 补测):分歧 = **AR 的固有数值语义**,不是接线 bug

**方法**(干净隔离,别再用多根归约的 checksum — 见下"诊断假象"):

- `pf_bisect` 的**多根**求值只服务"一次性看全层";它的**读数不可信** —— 层输出的块会被
  后续 in-place 残差写**覆盖**(实测:同 run 里 `embed` 多根读到的值与孤立读到的值不同;
  且**同一节点两次孤立 eval 也不等**:`fin` 首值 5.332 vs 5.258,因为 eager 前向会
  **原地推进 GDN 状态格**)。
- 干净做法 = **逐层孤立 eval**:每次只求值一个层根,且**先 `reset_gdn`**(状态清零)⇒
  与"turn 首 prefill"同起点。判据 = 首 4 值 + 未完桶。

**结果(0.8B,TP1 单卡 vs TP2 双 rank,同 prompt/同 ids)**:

| 节点 | TP1 | TP2 | 差 |
|---|---|---|---|
| `embed` | `[0.014282227, 0.0087890625, -0.008178711, …]` | **逐位相同** | 0(查表 ✓ 无关) |
| **`l0`** | `[-0.3010254, 0.07946777, 0.01109314, …]` | `[-0.30126953, 0.07946777, 0.011077881, …]` | **恰 1 ULP**(第一次分歧) |
| `l1` | `[-0.32421875, 0.070739746, -0.008216858]` | `[-0.32421875, 0.070739746, -0.00819397]` | 1 ULP |
| `l2` | 相同 | 相同 | 0 |
| `l3` | `[-0.15332031, 0.105163574, -0.09173584]` | `[0.055664063, 0.09753418, -0.083984375]` | **已放大** |
| `l23` / `fin` | `[-0.25195313, …]` / `[-0.1583252, 1.4150391, 3.0507813]` | `[-0.57910156, …]` / 显著不同 | 混沌放大 |

**判读**:分歧**精确出现在第一个带 AR 的输出**(layer0 = GDN 的 `out_proj`,行并行),
量级恰 **1 ULP** ⇒ 差异源 = **AR 的部分和语义**:

- 单卡:整段 K 的 **f32 累加** → 一次落 f16;
- TP:两半各 f32 累加 → **各自 round 成 f16**(staging payload = 层输出 dtype)→ **f16 相加**。

⇒ 这是 TP 的**固有**浮点语义(任何"半和 + 相加"的实现都有),再经 24 层 **GDN 递推
(非线性)混沌放大** ⇒ 最终 logits 出现**实质性**(非近局)差异 —— 与"接线错误"无关
(接线侧已由"两 rank 逐位相同"证清白 ✓;权重/ids 也已逐字节/逐字排除 ✓)。

**因此"G8:逐 token 一致"这条硬门在当前 f16 AR 语义下不可达**;可选的处置(裁决待定):

| 方案 | 内容 | 代价/风险 |
|---|---|---|
| **A 同序参考臂**(唯一能"逐位"的路) | 单卡参考臂的 row-parallel 层也按"两半 GEMM + f16 相加"计算(`Linear` 加一个诊断档)⇒ TP2 ≡ TP1-emu **逐位**;而 TP1-emu vs 真 TP1 的差 = 1 ULP(有界,可预测) | 多一个诊断臂;要维护两个参考口径 |
| **B 容差/统计门**(诚实的最小门) | L1 两 rank 逐位(已 ✓);L2 前若干层逐层 \|Δ\| ≤ 1 ULP 界、且**恶化速率**单调可控;L3 最终 logits:top-k 重叠率/贪心一致率 + 任务级(少量 prompt 的答案一致)阈值 | 不是"逐位",但符合 TP 工程惯例(vLLM 等同样接受) |
| **C f32 AR 载荷** | staging payload 改 f32(2× 字节;slot 64MiB ⇒ 总 128MiB/rank,仍 < 192MiB 墙)⇒ 去掉中间 f16 舍入,**减小**差异 | **不解决**混沌放大(求和序仍不同)⇒ 不构成 gate 的解,只作数值优化 |

**次刀建议**:先落 **A**(它同时是"logits/数值口径特殊处理"的正解,也是唯一能让 G8
恢复"逐位"硬判的路径),再把 B 的三层判据写成门(防回归);C 挂 S6 备选。

#### 18.5.1 诊断假象两则(留档,防后人重踩)

1. **多根归约的读数会被覆盖**:`eval_ops_multi_env` 里 171 条根一次性求值,早期根的块可能
   被后续 in-place 操作(残差融写)覆盖 ⇒ 读回来的是"后来的值"。**层 checksum 只信孤立 eval。**
2. **孤立 eval 必须 reset_gdn**:eager 前向原地推进 GDN 状态 ⇒ 连续两次孤立 eval 的
   同节点结果不同(实测 `fin` 首值 5.332 vs 5.258)。凡"多次求值对拍"的实验,必须先回到
   同一起点(reset_gdn),否则比的是状态不是算子。

#### 18.5.2 已排除项(全部逐字节/逐字验证)

- **接线**:两 rank logits **逐位相同**(248320 项,0 项不同)⇒ AR 协议/别名/序都正确;
- **权重**:同卡 GPU 面 `load_from_dir(0,1)` vs `(0,2)` —— `embed_tokens.weight`
  **508559360 字节逐字节相等**(layernorm 同);
- **ids**:两臂 `tb.ids` 逐字相同(前 8/末 8 实测);
- **查表**:孤立 eval 的 `embed` 三臂逐位相同。

#### 18.5.3 可复用工具(本轮新建,全部 env 门控)

- 装配期:`staging` header 自证(游标必须 0/0、err = 0);
- 运行期:`tp.ar.r{0,1}` 注入计数(对称性)、`OWL_TP_TRACE` 分相时间戳、
  `OWL_G8_BISECT=1`(G8 门开 `pf_bisect` + 逐层 checksum 对拍;⚠️ 多根读数不可信,
  见 §18.5.1 —— 逐层判据请用"孤立 eval + reset_gdn"法);
- 权重面:同卡 `load_from_dir((0,1))` vs `((0,2))` 的**同键逐字节对拍**;
- 双 rank 面:`read_output_f32` / `read_peer_output_f32`(G8 门的"两 rank 同值"实证)。

---

## 19. G8 门表改写(2026-10-17;业界判例收口 —— **本段取代 §5 的 G8 行**)

> §18.5 已定谳:TP2 与单卡的分歧 = **AR 的固有数值语义**(部分和各自 round f16 再相加;
> 单卡是整段 f32 累加、一次落 f16),经 24 层 GDN 递推混沌放大。**"TP2 ≡ TP1 逐 token"
> 与业界实践及物理可达性不符** —— 本节把两家的判例与我们的新门表钉死。

### 19.1 业界判例(2026-10-17 现场核对源码/issue/文档)

**vLLM**(本题的答案 = "预期差异 + rank 级 CI + 跨 TP 免责"):

| 事实 | 出处 |
|---|---|
| 官方立场:TP 归约"必然"与单卡不同、小差异放大后换 token,**"this is expected"** | issue #2891(已 close;维护者 hmellor 原话) |
| 默认**不保证**可复现("does not guarantee the reproducibility of the results by default, for the sake of performance") | `docs/usage/reproducibility.md` |
| 确定性档(`VLLM_BATCH_INVARIANT=1`)第 3 条:**关掉自定义 all-reduce**("custom all-reduce operations in tensor parallel mode") | `docs/features/batch_invariance.md` |
| AR 核 **fp32 累加**(`packed_t::A = array_t<float, 16/sizeof(T)>`,注释 "the (A)ccumulator type for reduction") | `csrc/custom_all_reduce.cuh` + `csrc/custom_collective_common.cuh` |
| **TP 正确性 CI = rank 判据**(`check_logprobs_close`):采样 token 相同 ⇒ 继续;不同 ⇒ **互为对方 top-N logprobs 候选**,打 warning **并停止比对**;逐 token 硬判(`check_outputs_equal`)只用于无 TP 场景 | `tests/models/language/generation/test_hybrid.py::test_distributed_correctness`(TP=1 vs TP=2)/ `tests/models/utils.py` |
| 自定义 AR 有 **8MB 尺寸门**(超过走 NCCL)⇒ 同一请求的归约路径随 batch 形状变 ⇒ 破坏 batch 不变性(其确定性档因此整条换实现) | `vllm/distributed/device_communicators/custom_all_reduce.py::should_custom_ar` |
| 理论源(**跨卡归约明确排除在 batch-invariance 承诺之外**;"NVLink-SHARP in-switch 归约"只在 Hopper(CUDA 12.8+)/Blackwell 上确定) | Thinking Machines《Defeating Nondeterminism in LLM Inference》(vLLM 引) |

**SGLang**(本题的答案 = "把归约序钉死;但**不**承诺 ≡ 单卡/跨 TP 度"):

| 事实 | 出处 |
|---|---|
| `--enable-deterministic-inference`;验收 = `test_deterministic --test-mode single/mixed/prefix/radix_cache` 判 **"Unique samples: 1"**(**只变 batch 形状/前缀切分/缓存状态,从不改 TP 度**) | `docs/docs/advanced_features/deterministic_inference.mdx` |
| ★ 官方 blog 对 TP 轴的原文:**"TP1 and TP2 are deterministic due to consistent floating-point addition order; larger TP setups require modifications to reduce kernels for determinism."** | lmsys.org/blog/2025-09-22-sglang-deterministic(§Future Work) |
| 已做:确定性 all-reduce 核(NCCL #10930;AMD #15340);遗留 "Not deterministic on Blackwell for TP4" #11513 | issue #10278 的 "Communication (NCCL)" 清单 |
| 实测:TP2 三种模式全绿;**TP4 在 prefix 模式炸**(Unique 5~77),且**与 AR 实现无关**(换 NCCL/symm-mem 依旧)⇒ 根因在别处"切分随 shape 变" | issue #10785 |
| 钉序四把锁:注意力 **fixed_split_size / num_splits=1 / chunked prefill 截断对齐 split size**;非贪心采样换 `multinomial_with_seed`(种子哈希 Gumbel);确定性档**绕开自定义 collective**(`_deterministic_collectives_enabled`);RL 全链再叠 `NCCL_ALGO=Ring` | blog + `parallel_state.py` + `sampler.py` |
| AR 核同样是 **fp32 累加**(`packed_t::A = array_t<float,…>`,与 vLLM 同源) | `python/sglang/kernels/aot/csrc/allreduce/custom_all_reduce.cuh` |

### 19.2 新门表(G8 拆三段;§5 的 G8 行作废)

| 门 | 名称 | 判据 | 文件 | 状态 |
|---|---|---|---|---|
| **G8-a** | **配置内不变性**(SGLang 判据) | **同 world、同 prompt**,变队列形状/ prefill chunk 切分/前缀缓存状态/会话热账 ⇒ **会话 token 账逐位唯一**("Unique samples: 1") | `crates/engine/tests/tp_determinism_gate.rs` | ✅ **已落地**(T1;4 装配档 × 4 arm + 复用臂全绿 —— §21.1) |
| **G8-b** | **vs 单卡功能等价**(vLLM 判据) | 贪心 token 相同 **或** 互为对方 top-N(logits 自算);**分歧即止** + 首分歧步/分歧率/逐步 max\|Δ\| 记账(阈值不设,先攒数据) | `crates/engine/tests/tp_g8_gate.rs` | ✅ **已落地**(T2;首步近局翻转、互含 true —— §21.2) |
| **G8-c** | **位级 transport/驱动**(同配置内) | ① 两 rank logits 逐位相同 ② **跨卡(3080 对) ≡ 同卡双 rank**(方案 A)逐位 | ② `crates/engine/tests/tp_transport_bits_gate.rs` | ✅ ① 已绿 ② **已落地**(T3;12 tok + 248320 logits 逐位相同 —— §21.3) |

**为什么 owl v1 天生满足 G8-a(卖点,写进文档)**:v1 的 AR 是**固定核 + 相对游标 + 单 chunk**
(载荷 > slot ⇒ 响亮拒,不静默分块)⇒ **归约序与 shape 无关** —— 正是 SGLang 给 TP=2 的
那句 "consistent floating-point addition order"。§S5 一旦放开多块/流水,**这条资产会被破坏**
⇒ 见 19.3 纪律 ①。

### 19.3 纪律(下一刀起生效)

1. **AR 归约序与 shape 解耦**(S5 前置):
   - 禁止"归约块数/树形随 shape 变"(SGLang TP4-prefix 案 = 同类病);
   - S5 多块 AR 二选一:(a) 切法固定(块数与 shape 无关)或 (b) 显式"确定性档"固定单一路径
     (vLLM 的等价做法 = 确定性档直接换实现);
   - 验收:§19.2 的 **G8-a 在 S5 之后仍必须绿**。
2. **禁用依赖 batch 形状的采样器**:非贪心确定性要用**种子哈希 Gumbel**(SGLang
   `multinomial_with_seed` 同法),`torch.multinomial`/依赖 batch 的实现不许进生产档(backlog T5)。
3. **f32 只在累加内部**:payload 保持模型 dtype,归约核内部 f32 累加(vLLM/SGLang/owl 三方同律);
   **方案 C(f32 载荷)删除**——它改不了结合序,只增流量。
4. **跨 TP 度不承诺等价**(与两家同款免责):文档/requirements 只承诺 ①同配置 run-to-run
   与配置内不变性 ②vs 单卡的功能/统计等价;`world ≥ 4` 的归约树必须先过 G8-a 才算接线完成。
5. **复用路径不复位(热 ≠ 冷是预期)**:同会话 `guard` 命中**不受 `prefix_cache` 开关控制**
   (开关只管跨会话块链复用);复用 ⇒ GDN **分段推进** ≠ 一次全量推进 ⇒ 数值不同、生成可不同
   (实测四档一致:hot `[198,17,13,220]` vs cold `[198,17,15,15]`)。业内同款:vLLM/SGLang 的
   prefix caching 均**不承诺**与全量位级一致 ⇒ **G8-a 只判"配置内唯一",不判"热==冷"**。
6. **读数槽纪律(供门/诊断)**:凡"本步产物"的取数槽必须由**产出该步的那条路径**写 ——
   prefill 末块走 eager(不写会话图输出槽)⇒ `read_output_f32` 在首步会读到上图 step 残值
   (T2 实测定谳;修法 = 末块 logits 发布进槽,§21.4-1)。新增读数面先问一句:
   **写槽的路径与产 token 的路径是同一条吗?**

## 20. 交接单:下一刀(2026-10-17 结)

### 20.0 你的位置

- worktree:`packages/owl-worktrees/w4a8-p1`(分支 `feat/w4a8-p1`);**当前 28 文件改动未提交**
  (含 `crates/engine/src/tp.rs`、`crates/engine/tests/tp_{assemble,g8}_gate.rs` 三个新文件);
  提交须等用户点头(工作区纪律)。
- 真机:TP 门用 **3080 对 = CUDA ordinal 2/3**;`OWL_0_8B_DIR=<…/Qwen/Qwen3.5-0.8B>`
  (0.8B 是 tied 检查点:无 `lm_head.weight`)。
- 现在绿的门:`tp_assemble_gate`(K2)/ `comm_ar_parity`(G1/G2/G3)/ `peer_shared_gate`(G4/G5)/
  单卡 GPU e2e(`OWL_E2E_ONLY=two_turns`)/ 回归(shared 65 · comm 11 · models 204 · engine 45)。
- 现在红的门:`tp_g8_gate`(旧判据"逐 token 一致")—— **按 §19.2 改成三段门后可预期转绿**。

### 20.1 已完成、**不要重开**的结论

1. 接线无罪:两 rank logits **逐位相同**(AR 协议/别名/序都对);权重(同卡 full vs shard 逐字节)、
   ids(逐字)都已排除;
2. 分歧 = **AR 固有数值**;首分歧在 **layer0(第一个带 AR 的输出)恰 1 ULP**,再被 GDN 递推放大;
3. 诊断假象两条(§18.5.1):多根归约读会被覆盖(层 checksum 只信孤立 eval);孤立 eval 必须
   **先 reset_gdn**(否则比的是状态不是算子);
4. 业内不承诺"≡ 单卡"(§19.1);SGLang 视 "TP1/TP2 归约序固定" 为确定性能力本身。

### 20.2 待实现任务(按建议顺序)

| # | 任务 | 落点 | 做法 | 验收 |
|---|---|---|---|---|
| **T1** | **G8-a 配置内不变性门**(最高价值) | 新 `crates/engine/tests/tp_determinism_gate.rs` | 同 world 下跑同一 prompt 的多组形状:① 顺序单请求 ② 并发 1/2/4(引擎已有 submit 队列)③ `prefill_chunk` 32/64/128 ④ 前缀缓存开/关;逐组取 `session_token_ids` 比**逐位唯一** | "Unique samples: 1" 形态;门名/判据与 §19.2 一致 |
| **T2** | **G8-b 改比较器** | `crates/engine/tests/tp_g8_gate.rs` | 用 `read_output_f32("logits")` 自算 top-N;判据 = token 相同 或 互为 top-N;**首分歧步 + 分歧率**上屏;分歧即止 | 与 vLLM `check_logprobs_close` 同形;阈值待 20.3 拍板 |
| **T3** | **方案 A:位级 transport 门** | `crates/engine/src/tp.rs`(加"允许同卡双 rank"的**测试档**入口,生产路径保持拒)+ 新 `tp_transport_bits_gate.rs` | 同卡开两个 `Engine`(同 ordinal)→ `TpEngine::assemble` → 与跨卡 TP2 比 token 账/logits **逐位** | 逐位相等;注意:两臂须**同 GPU 型号**、显存够(0.8B 可) |
| **T4** | **AR 归约序纪律落地** | `crates/backends/comm` 头注 + 本节表 | 把 19.3-① 写成代码注释与门依赖(S5 后 G8-a 必须仍绿) | 文档 + 门依赖登记 |
| **T5** | (backlog)**非贪心确定性采样** | `crates/engine/src/sampler.rs` | 种子哈希 Gumbel(抄 SGLang `multinomial_with_seed`) | v1 不做,记 backlog |

### 20.3 需拍板(留给用户/下一刀开头)

1. **G8-b 阈值**:只要求"互为 top-N + 分歧即止",还是加"首分歧步 ≥ K / 分歧率 ≤ x%"硬线?
   (vLLM 只做前者;建议先只做前者 + 打印指标,攒数据后再定线。)
2. **T3 是否现在做**:它价值 = 位级隔离 transport(协议/序/驱动任何回归当场红),成本 = 一个测试档入口 + 一个门(不动业务面)。
3. **"确定性档"是否立项**(S6):world ≥ 4 / S5 多块 AR 需要固定树形 —— 若要"确定性"作为产品能力,应现在就把它写进 flavor 面(参考 vLLM `VLLM_BATCH_INVARIANT` 的整条换实现思路)。
4. **提交**:28 文件是否分刀提交(K1/K2/K3 + 本文档各一刀)?

**拍板结果(2026-10-17 结)**:① G8-b = **只做"互为 top-N + 分歧即止" + 打印指标**(先攒数据,
不设硬线)② **T3 现在做**(已落地,§21.3)③ 确定性档**不立项**,记 backlog(S5 真要放开多块/
流水时再立项;验收条件已由 §19.3-① 写死 = 届时 G8-a 仍绿)④ **提交分刀** —— 实施后按代码
依赖收敛为:"⑤ 功能面"一刀 + "门/取证面"一刀 + 文档一刀(§21.7 报告切分实情)。

### 20.4 工具手册(复现与取证,全部已入库)

| 目的 | 命令/要点 |
|---|---|
| 跑 TP 装配门 | `OWL_TP_TEST_DEV_A=2 OWL_TP_TEST_DEV_B=3 OWL_0_8B_DIR=… cargo test -p owl-engine --test tp_assemble_gate` |
| 看 AR 是否真跑 | 装配后/每 turn 后读 staging header(**必须** 游标 0/0、err = 0;turn 后 produced == consumed) |
| 逐层对拍(干净法) | **逐层孤立 eval + 每次先 `reset_gdn`**;禁用多根 checksum 当层判据(会被 in-place 残写覆盖) |
| 权重面取证 | 同卡 `load_from_dir(dir, face, (0,1))` vs `((0,2))` 的**同键逐字节对拍** |
| 数值面取证 | `read_output_f32` / `read_peer_output_f32`(两 rank 同值实证);`OWL_G8_BISECT=1` 开 `pf_bisect` |

---

## 21. ⑤ 收尾轮落地记录(2026-10-17;T1–T4 全绿 + 三个定谳)

> 用户拍板(§20.3 四问的答案):① G8-b = 只做"互为 top-N + 分歧即止" + 打印指标(先攒数据,
> 不设硬线)② T3 现在做 ③ 确定性档**不立项**,记 backlog(S5 真要放开多块/流水时再立项,
> 验收条件 = 届时 G8-a 仍绿)④ 提交分刀(见 §20.3 末)。

### 21.1 T1:G8-a 配置内不变性门(`crates/engine/tests/tp_determinism_gate.rs`)

**判据**(SGLang "Unique samples: 1"):同 world(world = 2)、同 prompt、同 greedy 档,
只变形状 ⇒ **生成段 token 逐位唯一**。形状轴与 arm:

| 轴 | arm | 机制 |
|---|---|---|
| ① 队列形状 | `base` / `queue3`(目标前压 3 个干扰 turn) | 队列 FIFO + GDN 格/块链/事件序扰动 |
| ② prefill 块长 | 装配档 chunk = 64 / 32 / 128 | 块式喂入切分 |
| ③ 前缀缓存开关 | 装配档 `prefix_cache` = on / off | 跨会话块链复用(blocks_m) |
| ④ 会话热账 | `session-replay`(同 sid 重发同 prompt) | guard 失配 + 旧账归还 |
| ⑤ **前缀复用(热)** | `prefix-reuse`(回填历史 ⇒ guard 命中 ⇒ 只增量 prefill) | 真复用路径(带 `cached_submit > 0` 自证,防 arm 空转) |

**实测(3080 对 = CUDA ordinal 2/3;4 装配档 × 4 arm + 复用臂;27.0s 全绿)**:
- 16 个 arm 生成段**逐位唯一**:`[198, 220, 220, 16, 13, 220, 220, 17, 15, 15, 23, 220, 95859, 220, 16, 16]`
  —— **跨档也唯一**(chunk 32/64/128 与 pc on/off 都不改变 token);
- 复用臂真命中(`cached(submit=34, active=34)`;prompt 38 tok 中 34 复用、4 增量),同形重复逐位唯一;
- 附:arm ⑤ 的 turn1 与 `base` 同形 ⇒ 也逐位一致(顺带覆盖"先跑别的 turn 再跑同形")。

**为什么 v1 天生该过**:AR = 固定核 + 相对游标 + **单 chunk**(载荷 > slot ⇒ 响亮拒,不静默
分块)⇒ 归约序与 shape 无关(§19.2 卖点)。**本门必须在 S5 之后仍绿**(§19.3-①)。

### 21.2 T2:G8-b 比较器(`crates/engine/tests/tp_g8_gate.rs` 重写)

改前 = "逐 token 一致"硬判(**红**,§18.5)。改后 = vLLM `check_logprobs_close` 形态:
**逐位置比较 token**(每步一条 logits,采样步 = `Token` 事件对齐);分歧 ⇒ **必须互为对方
top-N**(N 默认 5,`OWL_G8B_TOPN` 可调),否则硬失败;**分歧即止**;首分歧步/分歧率/
逐步 max|Δ| 记账。

**实测(3080 对;全绿)**:

| 项 | 值 |
|---|---|
| L1 接线(两 rank 末步 logits 逐位) | **0 项不同**(max\|Δ\| = 0.000000) |
| 首分歧步 | **0**(首个 token) |
| 分歧处互含(N=5) | **true / true** |
| TP1 top5 | `[109757, 96328, 97698, 110789, 2005]` |
| TP2 top5 | `[2005, 96328, 95822, 109757, 98116]` |
| 逐步 max\|Δ\|(16 步) | 首步 12.18 → 末步 20.04;全程最大 21.93;均值 17.86 |
| 记账 | 判据窗口 1/16 步;窗口内分歧 1 步;全序列 token 不同 16/16(**参考值** —— 分歧后上下文已分岔) |

**判读**:两臂 top-5 有 3 项重合(109757 / 96328 / 2005)、**top-1 互换** ⇒ **近局翻转**
(数值噪声),不是逻辑错 ⇒ 与 vLLM 的 TP CI 判据同形收口。**"全序列 16/16 不同"正是"分歧即止"
这条纪律的存在理由**:token 一分岔,后续比较就没有意义(上下文不同了)。

### 21.3 T3:位级 transport 门(`crates/engine/tests/tp_transport_bits_gate.rs` + `TpEngine::assemble_same_device`)

**新入口**(生产路径**不**走):`TpEngine::assemble_same_device` —— 两 rank 查同一个 ordinal,
**跳过 PeerPlan 分档**(同卡自探测 P2P 恒 false),staging 仍走 VMM `peer_alloc`/`peer_import`
(**同卡自导入是 VMM 合法用法**:同一物理页在第二个 context 建只读别名 ⇒ AR 协议一字不改)。
握手抽了唯一实现 `handshake_staging`(生产 `assemble` 与同卡档共用 —— 禁两套规范)。

**实测(3080 对 2/3 + 同卡档 2;全绿 10.7s)**:
- 卡面断言:两档同型号(`RTX 3080`);
- **12 tok 逐位相同**;
- **末步 logits 248320 项逐位相同**(不同 0 项,max\|Δ\| = 0.000000)。

**结论**:transport(PCIe/窗口/DMA)面**零位贡献** ⇒ TP2 vs TP1 的全部分歧只来自
"分片 + AR 求和序"本身(§18.5)。**同卡两 rank 并发(自旋核与计算核共享 SM)无死锁** ——
这条是实测结论而非设计假设。

### 21.4 三个定谳(本轮;全部当场修或入档)

1. **★ prefill 末块的 logits 槽缺口(取证面 bug)**:`read_output_f32` 读的是**图 step**
   的输出槽(`GraphPlan.last`),而 **prefill 末块走 eager 路**(临时块 + 设备采样)
   ⇒ 门在首步读到的是上一次图 step(装配/warmup)的**残值**。实测证据:首条 logits 的
   argmax 与 `gen[0]` 无关(TP1 = 7286、TP2 = 6518),而 i ≥ 1 步严丝合缝
   (`argmax(logits[i]) == gen[i]`)。**修法** = 末块把 logits 发布进会话图输出槽
   (`eval_ops_multi_scoped_env` 两根 [logits, token] 一次求值 + `copy_block_at` 进槽;
   每 turn 一次,D2D ≤ 500KB)。修后首步对齐 ✓(T2 的判据才真正覆盖"首 token")。
2. **热 ≠ 冷是预期,不是 bug**:同会话 `guard` 命中(回填历史)**不受 `prefix_cache` 开关
   控制**(开关只管跨会话块链复用);复用 ⇒ **GDN 分段推进**(旧账 34 tok + 增量 4 tok)与
   一次全量推进(38 tok)**数值不同** ⇒ 生成可不同(实测 hot `[198,17,13,220]` vs cold
   `[198,17,15,15]`,**四档一致**)。业内同款:vLLM/SGLang 的 prefix caching 均**不承诺**
   与全量位级一致;SGLang 的 prefix 测项也只判"同状态重复唯一"。
   ⇒ **G8-a 只判"配置内唯一",不判"热==冷"**(纪律 §19.3-⑤)。
3. **门的基础设施必须自证不空转**:arm ⑤ 首版把 `cached_submit > 0` 写成断言 —— 若不断言,
   一个"复用从未发生"的门也会绿(首版实测 `active=0` 全档)。**教训**:凡"某路径必须被走到"
   的门,必须把该路径的**计数器**写成硬断言。

### 21.5 顺手清理(⑤ 的脚手架残留;全部零行为变化)

| 项 | 内容 |
|---|---|
| `boot_rank` 死构造 | 删"会话账 + 双块账房"构造(K3 已移到 `into_running`;每 rank 白建一个 SessionBank) |
| `RankDev` 三死字段 | 删 `quant_plan` / `mtp_dir` / `dims`(构造时写、零读面;真值走 `spec` / `env` / 局部) |
| `capture_gdn_snap` | 删(K3 后由 `RankDev::snap_dev` + `phases.rs` 两处 join2 驱动;同功能不留两个入口) |
| `ChunkHost.tok` / `pf_cksum(host)` | 删(未读字段/参数) |
| `TpStaging::layout()` | 删(零调用) |
| `exec/e4_api.rs` | 加模块级 `#![allow(dead_code)]` + 理由(E4 声明面;接线后删豁免) |
| `sampler.rs` | 修 `#[test]` 重复属性 + **给 `topp_mass_concentrates_head` 补 `#[test]`**(此前从未运行!) |

### 21.6 回归与新增工具

- **回归**:owl-shared 65 · owl-comm 11 · owl-models 204 · owl-engine 45(全绿);
  workspace check 0 错。TP 门耗时:t1 27.0s / t2 10.0s / t3 10.7s(3080 对)。
- **新增工具**:`RunningEngine::shutdown_tp()`(**执行态收尾**:卸图 + PeerRelease;
  此前只有装配态 `TpEngine::shutdown` ⇒ 多次装配的门无法干净收场)。收尾逻辑抽 `retire_rank`
  (装配态/执行态共用;⚠️ `peer_release` 是 GpuClient 固有方法 ⇒ 收尾面是 GpuClient 特化)。
- **门命令**(3080 对):
  ```bash
  OWL_TP_TEST_DEV_A=2 OWL_TP_TEST_DEV_B=3 OWL_0_8B_DIR=<…/Qwen3.5-0.8B> \
    cargo test -p owl-engine --test tp_determinism_gate -- --nocapture   # G8-a
  # … --test tp_g8_gate(OWL_G8B_TOPN 调 N)· --test tp_transport_bits_gate(OWL_TP_SAME_DEV 指同卡档卡)
  ```

---

## 22. S6 前哨:27B TP 乱码定谳(2026-10-17;TP 性能测试被此阻断)

### 22.1 现象(可复现,确定性)

| 臂 | 产物 |
|---|---|
| TP1 单卡 27B AWQ | 连贯(长城介绍通顺;17×23 = 391) |
| TP2 27B(3080 对 2/3、3090 Ti 对 0/1;chunk 128/1024;w4a16/w4a8) | **纯乱码**,且跨卡对**逐字复现**(确定性) |
| TP2 0.8B(f16) | 连贯(17×23 = 391)—— 与 T1–T3 门一致 |

boot 面全绿:分档 Window / staging 握手 / 自证 `produced=0 consumed=0 err=0` /
两 rank 图捕获;每卡 12.5G(27B @3080)。

### 22.2 ★ 根因:`rule_of` 不认量化派生键 ⇒ 分片区间**没翻译**

`crates/models/src/plan/rules.rs::rule_of` 只剥 `.weight`/`.bias` 尾巴。而
**量化档的 want 键是派生键**(`{base}.qweight` / `.packed_raw` / `.scales` /
`.zeros` / `.marlin_ws` / `.marlin_ctmp`)⇒ 一律回落 `WeightRule::Replicate`
⇒ `build_shard_pass` **`continue` 跳过**(不进 span 表)⇒ `ShardedSource` 直通
内层源 ⇒ **两个 rank 读到同一段(全量前段)**。

**硬证据**(新门 `crates/engine/tests/tp_load_gate_27b.rs`,真 27B 检查点、
只截 4 层、GPU face、`device_repack=false`):

```
✗ …layers.0.mlp.gate_proj.qweight: slice0 变了(len 22282240 vs 44564480;== 全量前段? true)
✗ …layers.0.mlp.gate_proj.qweight: slice1 变了(len 22282240 vs 44564480;== 全量前段? true)
… 160 条全同型(复制键检查 252 次 / 分片键检查 30 次;失败 160)
```

两 rank 读同一段 ⇒ 两侧部分和**逐位相同** ⇒ AR 把同一份部分和加了两遍 ⇒
输出全错。这也解释了 `pf_bisect` 实测"rank0 == rank1 全层一致"。

**为什么 0.8B 门全绿**:0.8B 是 f16,键名以 `.weight` 结尾 ⇒ `rule_of` 命中
⇒ 区间正常翻译。**f16 分片路径无罪,量化派生键路径从未被门覆盖。**

### 22.3 附带定谳(取证面两条,防后人重踩)

1. **`pf_bisect` 逐层 checksum 不能用于"单卡 vs 双卡"对拍** —— 0.8B(输出
   正确)上 `c0.embed` 同样不一致(多根 eval 的块覆盖,§18.5.1 的"只信孤立
   eval"是同一条);本机实测 450/450 全异,零信息量。
2. **`mfact` 打点双 rank 同 tag**:`boot_seq` 在 `into_running` 只分配一次,
   两 rank 共用 ⇒ `counter_set` 后写覆盖、前值丢失(单卡/双卡对拍会拿到
   混合值)。已在 `exec/prefill_chunk.rs` 的 pf 系列打点加 **rank 前缀**
   (`pf.r{rank}.…`);`mfact` 的原注("AR 后两侧同值 ⇒ 幂等")只对**真的同值**
   成立 —— 本次就是"两侧同值但都错"的反例。

### 22.4 修法(设计;未施工)

> ⚠️ **本段是定位过程的存档**:§22.4 的 A/B 两案当时尚未施工,后续按 A 案落地,
> 并在 §23(前向 NaN 定位链)→ §24(**总根因 = `DeviceRearrange` 双切**定谳)收口。
> 读本节只为看"当时怎么判",**修法以 §24 为准**。

**核心矛盾:派生键的 want 形状坐标系不统一**(规则表按逻辑 `[out, k]` 写):

| 键(27B gate_proj,full) | want 形状 | out 在哪维 | k 在哪维 | 规则表要求 |
|---|---|---|---|---|
| `packed_raw`(生产档) | `[17408, 640]` = `[out, k/8]` | dim0 | dim1 | Col→dim0 ✓ |
| `scales` | `[160, 17408]` = `[k/32, out]` | **dim1** | dim0 | Col→dim0 ✗ |
| `zeros` | `[160, 2176]` = `[k/32, out/8]` | **dim1** | dim0 | Col→dim0 ✗ |
| `qweight`(CPU 物化档) | `[320, 34816]` = `[k/16, out×2]`(marlin 视图) | **dim1** | dim0 | Col→dim0 ✗ |

两条候选路(择一,须用户裁决):

- **A(推荐,单一约定)**:把**全部**量化派生 want 规范化到**逻辑序**
  (`n 侧 = dim0`,`k 侧 = dim1`),物理序(转置/marlin 平铺)由
  `Layout::DeviceRearrange{rows, cols}` 与源适配器承载。这样 `rule_of`
  只需补"剥派生后缀",span 数学与段切**原样复用**;代价 = 动 `Linear::layout`
  的声明形状 + `AwqSource` 的供数序(或加一个转置视图)。
- **B(局部补丁)**:`build_shard_pass` 里对派生键**换维**(Col↔Row),并给
  `SourceSpan` 加"列段切"能力。**不推荐** —— 段切(`q_proj`/`in_proj_qkvz`/
  `conv1d` 的量化变体)在转置序下是 dim1 段,现有词汇
  (`Rows`/`RowsSeg`/`Cols` 单窗)表达不了,要扩词汇;两套坐标系并存 = 本仓
  最忌的"接口不唯一"。

**门的归属**:`tp_load_gate_27b` 是本次新增的**真检查点**门(0.8B 门覆盖不到
派生键/合成键/段切/打包口径四件);修好后它必须转绿,并纳入 S6 回归。

### 22.5 本轮落地清单(worktree `feat/w4a8-p1`,未提交)

- **server TP 启动档**(S4e 引擎双 actor 的 v1 形态):
  `RuntimeCfg.tp_ordinals`(`shared/config.rs`)+ `Devices` 设备束 /
  `resolve_tp_devices`(`apps/server/src/config.rs`)+ `actor_main` TP 分支
  与双卡锁(`apps/server/src/main.rs`);**actor/HTTP 面零改动**。
- **flavor `tp27b`**(`assets/flavors/`;TP 形状 + f16 族白名单 + 通信族两项)
  与对照档 `testdata/args/bench-27b-tp{1,2}.toml`。
- **门 `tp_load_gate_27b`**(上文)+ **pf 打点 rank 前缀**(取证面修)。
- **量化档护栏(已落)**:`tp_v1_check_quant(QuantPlan)`(engine 唯一判据)
  在 `assemble_with_staging` 的第 ④ 条护栏上,非 `F16` ⇒ **响亮拒**并指到
  本节;实测 27B TP2 启动即拒(消息含修法指向),f16 TP2 照常起。
  ⚠️ 临时件 —— §22.4 的 A 刀落地后撤。
- 顺带查出(本基座存量,main 上可能已修):flavor `awq27b` 缺
  `max_concurrency = 1`(单卡也起不来)+ 白名单是 fp8 族而缺省 `kv_fp8=false`。

### 22.6 TP2 吞吐实测(2026-10-17;**权重错误档**,仅供吞吐参考)

护栏临时旁路(`TEMP-BENCH`)后跑 std 五点三轮(`--ignore-eos`;3080 对 =
CUDA ordinal 2/3;档 = `tp27b`,max_seq/pool 8704、chunk 1024、greedy;
与 TP1 基准档**逐项同参**,唯一差别 = rank 数)。数据
`/tmp/owl-bench/tp2-3080-std.{csv,json}`(TP1 = `tp1-3080-std.csv`)。

⚠️ **口径警告**:本档权重是错的(§22.2 的 bug),但**计算形状/AR 流量/访存
与修好后完全一致** ⇒ 吞吐数字**有参考价值**;输出内容不可用。

(**修后数字见 §25.6** —— 本节是 0.244× 的立案现场,保留原读数做对照。)

| ctx | TP2 prefill t/s | TP2 decode t/s | TP2 TTFT | TP1 prefill | TP1 decode | TP1 TTFT |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 234.3 | 38.72 | 2186.7ms | 961.8 | 32.43 | 534.1ms |
| 1024 | 236.4 | 38.59 | 4334.5ms | 978.3 | 32.20 | 1048.7ms |
| 2048 | 236.4 | 37.87 | 8663.7ms | 985.7 | 31.93 | 2079.9ms |
| 4096 | 236.5 | 38.47 | 17323.2ms | 982.1 | 31.98 | 4172.7ms |
| 8192 | 236.2 | 38.22 | 34690.1ms | 963.0 | 31.06 | 8509.2ms |

**比值:prefill 0.24×(!!)· decode 1.20× · TTFT@8192 4.07×**。

#### 定谳候选 1:AR 的**大载荷**代价失控(prefill 崩盘的直接账)

- decode 面:TP2 26.0ms/tok vs TP1 31.3ms/tok ⇒ 省 5.3ms;而"权重读减半"的
  理论收益 = 15.2G→7.6G/rank ⇒ 20ms→10ms(**应省 10ms**)⇒ AR 吃掉约
  4.7ms/tok;按 64 层 × 2 处行并行 ≈ 128 次 AR ⇒ **≈ 37µs/AR**(PCIe 往返
  + 自旋,量级合理)。
- prefill 面:TP2 512 tok 用 2.18s,TP1 0.53s ⇒ 多出 1.65s;同样 ~128 次 AR
  ⇒ **≈ 12.9ms/AR**。载荷 = 512×5120×2B = **5.24MB** ⇒ 有效 **~400MB/s**
  (对 P2P 13.2GB/s 差 **33×**)。
- **两侧差 350×**(37µs vs 12.9ms)⇒ 不是"AR 慢",是**大载荷路径**另有
  串行化(候选:分片传输逐段同步 / 入队半-收割半强制 host 往返 / staging
  槽 32MiB 下的分段回环)。**这是 TP 线的头号性能立案**。
- 另注:prefill 逐点**完全平坦**(234~236),说明代价与 ctx 无关而与
  **token 数**(= AR 载荷)成正比 ⇒ 与"每 AR 固定开销"一致。

#### 定谳候选 2:TP2 跑 8 点后**段错误**(新立案)

第 8 点(第 2 轮 @2048)返回 `out=0` / 耗时 14.2s,随后 server
**SIGSEGV(core dumped)**。前 7 点全正常。初判方向:池累积(8 请求 KV)+
AR/staging 交互(或 §22.2 的错误权重把某处索引带飞)。**未定位**,列 TP 线
第二个立案;复现配方 = 上述 std 臂(8 点即现)。

---

## 23. S6 前哨②:27B TP 前向 NaN 定位链(2026-10-17 深夜;未结)

### 23.1 现象(最小复现)

- TP1 27B 单卡:正常(logits min −10.4 / max 27.1);
- TP2 27B(3080 对 2/3 或 3090 Ti 对 0/1):**logits = ±f32::MAX + NaN**,
  输出与输入**无关**(两个不同 prompt 逐字同串),贪心 = `"!!!!…"`、sampling
  同串 ⇒ **前向数值爆炸**,不是采样/模板问题;
- TP2 **0.8B**:正常 ⇒ 与"双 rank 机制"无关。

**最小复现(单层)**:`tools/trunc_ckpt_layers.py <N>` 造截断检查点(改
safetensors 头 + 重排 data_offsets;**索引 JSON 无用** —— `open_raw_index` 是
扫文件头),`OWL_27B_DIR=<截断目录>` 指向它即可。**N=1 单层 GDN 已复现 NaN**
⇒ 病灶在**一层之内**,不需要跨层二分。

### 23.2 已洗清(每条都有门/实证,勿重开)

| 面 | 证据 |
|---|---|
| **装载(全部键)** | 新门 `tp_load_gate_27b`:源级对拍 **142 次全过**(含 `packed_raw` 这唯一真实权重数据面;绕开重排直拍源字节) |
| **AR 核** | `comm_ar_parity` 扩 3 档 = **27B 真实载荷**(1.31MB / 10.5MB f16)→ 11 case 全绿 |
| **AR 接线** | turn 后游标 r0=r1=(128,128,0) err=0;`tp.ar` 计数对称(256/256);两 rank 逐层 tap 一致 |
| **marlin AWQ GEMM** | `marlin_parity` 扩 **6 个 27B 分片形状**(out_proj/down_proj/gate/up/q/k/v/in_proj_qkvz)→ 噪声 5e-4 全绿 |
| **GDN recurrence** | `gpu_recurrence_varlen_matches_expansion` 扩 **27B 真维 (nk=8,nv=24,kd=128,vd=128)** → 三档全绿 |
| **RowParallel 切法** | 层侧 `in_dim` = 分片 k 宽 ⇒ 装载映射到全量 k 的**对应段** ⇒ 与各 rank 的 ColParallel 输出匹配(手核,无双重切) |

途中修掉的**门自身伪影**(都是"门覆盖不到 ⇒ 假绿/假红"):
① `pf_bisect` 逐层 checksum **不可**用于单卡/双卡对拍(0.8B 正常模型上 170/170 全异);
② 量级探针必须 **dtype 感知 + 用均值**(形状不同时和不可比);
③ GDN 门原用**未归一化 k**(|k|²=kd ⇒ kd=128 时两核同溢出)+ 容差 1e-4 对
f16 量级 26 是假红(state 1e-5 → 1e-4 同为伪影)。

### 23.3 已定位(单层 tap 量级,17 根 → 无覆盖,可信)

`OWL_PF_STAGES=0` + `pf_bisect` 在 **N=1 模型**上(TP1 vs TP2-r0/r1):

| 站点 | TP1 | TP2 | 比值 | 备注 |
|---|---:|---:|---:|---|
| embed / n1 | 0.00800 / 0.66040 | 同 | **1.00** | 层输入逐位相同 ✓ |
| gq / gk / gv | 1.91 / 1.94 / 2.10 | 1.25 / 1.35 / 1.24 | 0.65/0.70/0.59 | 分片头数统计差异(可接受) |
| **gvc** | 0.1815 | 0.0425 | **0.23** | ⚠️ conv 输出 4× 偏小 |
| gqn / gkn / gg / gbeta | — | — | 1.18/1.30/0.75/1.00 | 可接受 |
| **mixed** | 0.0178 | **1.4187** | **79.7** | ✗ **首个爆炸点**;r0==r1 ⇒ AR 无辜 |
| h / n2 | 0.0204 / 0.0743 | 1.4189 / 0.729 | 69.6 / 9.8 | 残差流被污染 |
| **mlp** | 0.0138 | **2308.8** | **167303** | ✗ f16 溢出 ⇒ inf |
| fin | 0.2211 | 0.9996 | 4.5 | 已 NaN/含非有限 |

**读法**:`mixed` = `out_proj(norm_act(rec) ⊙ gate)`(GDN mixer 输出,**AR 后**)
—— 而 `norm_act` 是 RMSNorm(有界)⇒ **79.7× 不可能来自 recurrence 的量级**;
两侧一致 ⇒ **AR 无辜**。⇒ 病灶在 **GDN mixer 的"norm_act → ⊙gate → out_proj"
段或 `gvc` 那条 v 链**。

### 23.4 下一步(留给下一刀)

1. **`gvc` = 0.23× 是唯一"输入侧"异常** ⇒ 先查 conv 链:
   `owl_gdn_conv_fwd_f16` / `owl_gdn_conv_upd_dual_f16` 在 **分片 value_dim
   (3072)与全量(6144)** 的覆盖 —— 现有 conv 门用 nk=2/kd=16(d=32)/dim=12
   **完全没覆盖真实维度**(与 AR 载荷、marlin 形状、GDN 真维三案同型:
   **门没覆盖的那一格就是 bug 藏身处**)。
2. 若 conv 无罪:给 mixer 内部加 tap(`y` = recurrence 输出、`gated` = norm_act
   输出)—— 注意 `ctx.gdn_tap` 是 **8 件定长契约**(DFlash fold 按序消费),
   加件必须同步 fold,或另开一条 `gdn_tap2` 诊断链。
3. 定位后:`tp_load_gate_27b` + 上述四门作为回归;再撤 §22 的量化护栏(若已撤,
   则补回一次冒烟)。

---

## 24. ★ S6 定谳:`DeviceRearrange` 双切 —— 27B TP 前向爆炸的总根因(2026-10-17 结案)

### 24.1 根因(一行代码)

`formats/shard.rs::build_shard_pass` 里,量化权重键(`packed_raw`)的**布局**
被**又切了一道**:

```rust
// 原(错):rows/cols 再 shard_len 一次
WeightRule::UniformRow => (rows, plan.shard_len(cols)?),
_ => (plan.shard_len(rows)?, cols),
```

而**层是按分片 spec 声明**这个布局的 —— `Linear::layout` 传的是
`self.out_dim`(**分片后的 out**)、`self.in_dim/8`(**分片后的 k/8**)⇒
`DeviceRearrange{rows,cols}` 与 `w.shape` **同尺**。⇒ 再切一次 = **双切**:

| | 正确 | 双切后 |
|---|---|---|
| `w.shape`(= `n` 来源) | `[8192, 640]` | 同(未动) |
| `DeviceRearrange.rows/cols` | `(8192, 640)` | **`(4096, 640)`** ✗ |
| staged 矩形(`rows×cols`) | 5,242,880 字 | **2,621,440 字(一半)** ✗ |
| 发射 grid `(rows/64, cols/2)` | `(128, 320)` | **`(64, 320)`(一半)** ✗ |

⇒ GPU 重排只写了 marlin B 的**前半**,**后半是未初始化显存** ⇒ 量化 Linear
的 B 矩阵半垃圾 ⇒ 前向逐层放大到 f16 溢出 ⇒ **NaN**。
（0.8B 是 f16、无 `packed_raw` 键 ⇒ 从未踩到;`shard=1` 不走本 pass ⇒
对照臂永远绿 —— 这正是"门覆盖不到的那一格"。）

### 24.2 证据链(为什么能一路查到这一格)

1. **`tp_load_gate_27b::tp_repack_27b`(本轮新建,重排面门)**:设备字节 vs
   **CPU 参考**(源矩形 → `pack_marlin_b_fused`)。
   - **对照臂 `shard=1`:22 键全绿**(⇒ 参考实现正确,不是假红);
   - `shard=2`:44 键全红 ⇒ **病灶在分片路径**;
2. **装载器自带 `OWL_LOAD_VERIFY=1` 现场账**一锤定音:
   `rows=4096 cols=640 k=5120 n=5242880 p32=2621440 idx=20971520 b_ref=2621440`
   —— `n` 对(5,242,880 = 8192×640)、`p32` 只有一半、`idx` 表 8× 过长;
3. **单层复现**(`tools/trunc_ckpt_layers.py 1`)把病灶锁进一层之内;
4. 此前已逐项洗清:装载源读取(142 键源级对拍)、AR(载荷尺寸 3 档 + 游标)、
   marlin GEMM(6 个分片形状)、GDN recurrence(真维)、conv(真维)、
   norm_act(真维)、silu_mul(逐元素)、GPU 重排核(9 个分片形状)——
   **每一格单独都对,错的是"分片 pass 与层声明不同尺"这一格**。

### 24.3 修法(已落)

```rust
// 不变量:rows/cols == w.shape(分片形状)—— 显式钉死,不再二次切分
if let Layout::DeviceRearrange { op, .. } = w.layout {
    if axis != WantAxis::Logical { /* 响亮拒 */ }
    if w.shape.len() != 2 { /* 响亮拒 */ }
    w.layout = Layout::DeviceRearrange { op, rows: w.shape[0], cols: w.shape[1] };
}
```

### 24.4 验收(全绿)

- **重排面门:66 键全绿**(原 44 红);
- **TP2 27B 冒烟连贯**:长城(完整中文推理)+ `17×23 = 391` ✓;
- **std 15/15 全绿**(含此前必崩的第 8 点 —— **段错误随垃圾权重一起消失**);
- 回归:`owl-models` 206/206 · workspace check 0 错 · 0.8B 装载门 8/8 ·
  源级门 142/142。

### 24.5 性能现状(同二进制 A/B,3080 单卡 vs 3080 对)

| 指标 | TP1 单卡 | TP2 双卡 | 比 |
|---|---:|---:|---:|
| prefill t/s | **968.66** | 236.20 | **0.244×** |
| decode t/s | 32.01 | **38.65** | **1.207×** |
| TTFT@512 | 534ms | 2179ms | 4.1× |
| TTFT@8192 | 8579ms | 34669ms | 4.0× |

⇒ **正确性已闭环;性能瓶颈 100% 在 AR 通道**(prefill ≈ **12.9ms/AR** vs
decode ≈ **37µs/AR**;AR 核 = **单 block 256 线程** `family/comm.rs::BLOCK`)。
这与"通道要单独拉出去测/提速"的判断一致 —— 通道是独立工作面(载荷 1.31MB
时有效带宽仅 ~400MB/s,对 P2P 13.2GB/s 差 33×)。

### 24.6 教训提炼(四条律 + 取证面五纪律;下次遇"门全绿但前向错"先照此表走)

**① 门覆盖律 —— "门没覆盖的那一格就是 bug 藏身处"**。本轮**四次同型**命中:
AR 核门只测 10KB/1MB **f32** ⇒ 27B 的 **1.31MB f16** 载荷零覆盖;marlin 对拍
只测**全量维** ⇒ **分片**形状零覆盖;GDN recurrence 门只有 **ratio 1** ⇒ 27B
是 **ratio 3** 且 `kd=128` 压线;GPU 重排门只有全量形状 ⇒ 分片形状零覆盖。
⇒ 新档位(TP / 新维 / 新 dtype)上线前,**先补该档"最极端格子"的门**(成本 ~10 行)。

**② 对照臂纪律(门必须自证)**。门要自带"**已知绿**"的对照臂,否则无法区分
"参考写错(假红)"与"被测错(真红)"。本轮:`shard=1` 22 键全绿 / `shard=2` 44 键
全红 ⇒ 一句话钉到分片路径;无对照臂时那 44 条红只会被当成"门不可靠"而放过。

**③ 声明面同尺不变量(双切/双算之根)**。两处声明同一件事时(层声明的
`DeviceRearrange{rows,cols}` vs pass 推的"分片形状"),**必须钉死二者相等** ——
各推一遍 = **双切**,且**两处单独看都对**,只有乘积错。⇒ 凡"声明 + 推导"并存,
**推导处必须 `断言 == 声明`**,声明为唯一真源,推导量不得被下游当"源"再用。

**④ 爆炸三件一起读:形状 → max → 非有限计数**。"均值 79.7×"曾把人引偏
(norm_act 有界 ⇒ 不可能);真凶是**"输入有界(max 7.03)却输出饱和(max 64928
≈ f16 上限)"**这条非线性矛盾。⇒ 先读**形状**(不同形状的和不可比)、再读
**max**(均值掩盖离群)、再读**非有限计数**(首个 inf 的位置 = 病灶段)。

**⑤ 取证探针自身五纪律**(本轮修掉 4 处伪影 + 1 处探针 bug):
哈希面**不可跨 rank 数比**(0.8B 正常模型上 170/170 全异;`mfact` 双 rank 同 tag
+ `counter_set` = 后写覆盖 ⇒ 打点必须带 **rank 前缀**);`dtoh` 缓冲**按 dtype 定宽**
(原硬编码 `n*2` ⇒ f32 拍半读 = 垃圾);**容差匹配 dtype 精度**(f16 量级 26 时
1 ulp ≈ 1e-3 ⇒ 1e-4 是假红);**合成输入满足算子数值前提**(delta 规则要 `|k|²≈1`,
否则 kd=128 时两核**同溢出** = 判据失效);**诊断路径 ≠ 生产路径**
(`forward_bisect_stages` 用分离 add/rmsnorm,真实前向用 `FusedAddRmsnorm` 原地副作用)。

**⑥ "三流无保序"已证伪为本 bug 成因**(`launch_sync = true` ⇒ 数值逐字不变),
但隐患面真实:`Htod`/`UploadPinned` → **只 H2D**、`Launch` → **只 COMPUTE**
(无隐式序;staging header 初始化已有显式 `sync`)⇒ 单列工作面。

---

## 25. ★ AR 通道提速(2026-10-17;TP2 prefill 0.244× → 1.40×)

§24.5 立案:"**性能瓶颈 100% 在 AR 通道**(prefill ≈12.9ms/AR vs decode ≈37µs/AR;
AR 核 = 单 block 256 线程;载荷 5.24MB 时有效带宽仅 ~400MB/s,对 P2P 13.2GB/s 差 33×)"。
本节 = 该立案的隔离测量 → 定谳 → 修法 → 门 → 生产复测。

### 25.1 量法:先把通道单独拎出来(`crates/backends/comm/tests/ar_channel_bench.rs`;
2026-10-17 由 `crates/backends/cuda/tests/comm_ar_bench.rs` **归口通信包**,见 §26.7)

§24.5 那条账是**从端到端 tok/s 反推**出来的(混杂 GEMM/胶水核)⇒ 提速前必须先有尺子。
新基准走**生产同一批核**、零新增 kernel,三段分解:

| 探针 | 做法(哨兵 header 旁路一侧不变量)| 量的是什么 |
|---|---|---|
| **S** stage-only | 对端 `consumed = 2^62` ⇒ credit 永真 ⇒ 单边循环 stage | 本端 payload → 本端 staging 的**本地**拷贝 |
| **R** peer-read | 对端 `produced = 2^62` ⇒ I2 永真 ⇒ 单边循环 reduce | **对端读 + 本地 RMW** —— 大载荷慢路的主角 |
| **AR** full | 两 rank 各入队 K 次 `stage_i → reduce_i`,`tokio::join` 后同 sync | **生产路径**(含 credit 耦合 + 两次 fence + 发射) |

口径两律(本基准自己踩出来的,已写进其模块注):
- **host / 设备必须分离**:每次 `launch` = 一次 client↔server **同步往返**
  (~35-37µs;`ack` 在**入队后即回**,不等核完成)⇒ `enq` = host 面,`dev = total − enq`
  = 设备尾排。**判据只取 `dev`**;`dev ≈ 0` 只给上界("≤ 发射往返")。
- **阶梯不用重编译**:`OWL_ARB_SIZES/O__DTYPE/K`(见文件头)。

### 25.2 定谳①:12.9ms/AR = reduce 核**零 MLP**(不是带宽,不是协议)

修前实测(3080 对 2/3,f16,K=200;`R` = 纯对端读):

| 载荷 | R µs | 有效带宽 | 同字节 **f32** | 结论 |
|---:|---:|---:|---|---|
| 1.31MB | 3257 | 403 MB/s | — | |
| 5.24MB | 14439 | 363 MB/s | **7217**(f32)| **2×** |
| 10.49MB | 28919 | 363 MB/s | 14461 | |
| 32MiB | 92244 | 364 MB/s | — | |

- **带宽与字节数无关、与元素数成正比**;同字节 f32(元素数减半)恰好快 **2×**
  ⇒ 每元素**一趟串行访存往返**:`5.24MB/2B/256 线程 = 10240 迭代 × 1.41µs = 14.4ms`
  —— 与实测**逐位吻合**。病根 = **在飞访存数 = 1(线程)**,不是链路。
- `AR ≈ R + S`(14786 vs 14439+406)⇒ **协议/耦合开销 ≈ 0**;`S` 也是逐条
  (12.9GB/s,本地往返 ~0.32µs)。
- decode 尺寸(4B/10KB)修前 ≈ **71µs/AR = 2 × 35µs 纯 host 往返** ⇒ 小载荷面
  **本来就不是设备瓶颈**(这条解开了"prefill 0.24× 而 decode 1.21×"的两侧差 350×)。

### 25.3 定谳②:"小载荷比大载荷慢"的怪相 = **双循环结构的余数慢路**

首轮修法(展开主循环 + 逐条余数循环)后扫尺寸阶梯,出现反物理的
**20KB(83µs)慢于 128KB(0.2µs)**,且**复现**:
- 先疑外部抢卡(确实有一次:另一 agent 的 `engine_e2e` 占 CUDA ordinal 3,
  `clocks.sm=1710MHz/P2`)⇒ 改到**完全空闲**的 3090 Ti 对(0/1)**怪相照旧**
  ⇒ 不是干扰;
- **K 扫(50/200/800 同值)⇒ 是 per-launch,不是一次性代价**;
- 根因:`n_vec < 8×blockDim` 时展开主循环**一次都不进**,全部走逐条余数循环
  ⇒ 该档退回"每元素一趟往返"。
⇒ 修法 = **单循环 + 逐通道谓词**(先发 8 个读,再带谓词算/写;所有尺寸同一条路)。

### 25.4 ★ 定谳③:acquire **分工不对称** —— 慢把可见性洞盖住了(对照臂抓到)

向量化后 `tp_transport_bits_gate`(T3 位级 transport 门)**在 3090 Ti 对(0/1,同 PHB)红、
在 3080 对(2/3,跨 root complex)绿**。按②的对照臂纪律做了 A/B:

| 核 | 0/1(3090 Ti 对)| 2/3(3080 对)|
|---|---|---|
| **旧**(逐元素)| 绿 | 绿 |
| **新**(向量化)| **红**(token 0 起分歧)| 绿 |

⇒ **是本次改动引入的**,且是"旧核慢 ⇒ 洞被盖住"的经典形态。
机理:生产者是**全线程** release(每线程 `__threadfence_system()` → `__syncthreads()`
→ thread0 `STG.STRONG.SYS` 发布 `produced`);而消费者**只有 thread0** acquire,
其余 255 线程仅由 `__syncthreads()`(**block 域**,不足 **system 域**)排序
⇒ 加速后的**突发 8×16B 读**会跑到对端写可见之前。
修法 = **全体 acquire**(与生产者的 release 对称)。
⇒ 顺带证伪一条常见假设:"block 域屏障可以替 system 域 acquire" —— **不行**。

### 25.5 修法(内核两处 + 对称化一处;全在 `crates/kernels/cu/comm/`)

1. `ar_reduce.cu`:归约循环 → **16B 向量 + 8 路 + 逐通道谓词**(单循环;数值语义
   逐位不变 —— 仍是"每元素恰好一次 add",无重排 ⇒ 与 `ar.rs` 的
   "归约序与 shape 解耦"确定性纪律**不冲突**,G8-a 判据原样成立);
   16B 未对齐 ⇒ 标量兜底(降级只降速不改语义)。
2. `ar_stage.cu`:`owl_ar_copy_bytes` 同款(纯搬移,逐字节语义不变)。
3. `ar_reduce.cu`:acquire 提到**全体线程**(§25.4)。
   尾部语义不变(不足一向量的**整元素**走标量尾巴;**亚元素字节保持原值** —— 门判据)。

### 25.6 验收

**门(全绿)**:

| 门 | 位置 | 结果 |
|---|---|---|
| G1/G2/G3 对拍 / 确定性 / 图捕获 | `owl-comm` 的 `ar_transport_gate` | 11 case(三 dtype + 32MiB 跨 chunk)逐位 + 两跑同 + replay ≡ eager(0/1 与 2/3 双对)|
| 位级 transport | `tp_transport_bits_gate` | 同卡双 rank ≡ 跨卡 TP2:12 tok 逐位 + 末步 logits 248320 项逐位(**双对全绿**)|
| G8-a 配置内不变性 | `tp_determinism_gate` | 16 arm token 唯一 |
| G8-b 比较器 | `tp_g8_gate` | 首分歧步 0 处互为 top-5(max\|Δ\| 21.93)|
| 自环 + 超时负例 | `comm_ar_smoke` | 2 轮 2×/4× + 5s 超时 err=2 |

**通道读数(3080 对 2/3;f16;K=200)**:

| 载荷 | AR µs 修前 | AR µs 修后 | 有效 MB/s 修后 |
|---:|---:|---:|---:|
| 10KB(decode 1 行)| 71 | 71(**host 地板**)| 137 |
| 1.31MB(128 行)| 3337 | **177** | 7067 |
| 5.24MB(chunk 512)| **14786** | **822** | 6084 |
| 10.49MB(chunk 1024)| 30106 | **1584** | 6315 |
| 32MiB | 106297 | 7439 | 4300 |

⇒ **大载荷面 18×**(0.34 → 6.3GB/s);小载荷面 ≤host 发射往返(设备 ≤37µs)⇒ 无回归。

**生产 A/B(std 五点三轮,`--ignore-eos`;同二进制;3080 单卡 vs 双卡;ordinal 2 vs 2/3)**:

| 指标 | TP1 | TP2 **修前**(§24.5)| TP2 **修后** | 修后 TP2/TP1 |
|---|---:|---:|---:|---:|
| prefill t/s | 969.58 | 236.20 | **1353.08** | **1.396×** |
| decode t/s | 31.94 | 38.65 | **43.55** | **1.363×** |
| TTFT@512 | 533.69ms | 2178.7ms | **392.48ms** | 0.74× |
| TTFT@8192 | 8589.74ms | 34690.1ms | **6062.39ms** | 0.71× |

- **prefill 236 → 1353 = 5.73×**;TP2 从 **0.244×** TP1 翻到 **1.396×**。
- **decode 38.65 → 43.55(+12.7%)**;TP2/TP1 = **1.363×**。
- TTFT 三点**全点反超 TP1**(312KB 级 AR 载荷落在 couldn't-be-better 区)。
- 数据:`docs/bench/data/owl-tp-2026-10-17/tp{1,2}-arfix-std.{csv,json}`(gitignored 本地档)。

### 25.7 剩余杠杆(与 2000 t/s 的距离;本节是下一刀的入口)

修后 TP2 prefill 每 token = 740µs = **TP2 自有算力 ~516µs + AR 税 ~224µs**
(128 AR/步 × 1.31MB/token ÷ 6.3GB/s ≈ 208µs,吻合)⇒ **AR 仍占 ~30%**。
到 2000 t/s(500µs/token)需要二选一或并用:

1. **通道再提速(pull → push 对偶)**。现协议是 **pull**(对端 staging 被**远程读**)。
   对偶 = **push**(把本端部分和**远程写**进对端 staging,再**本地**读自己 staging 累加)。
   - **零新核**:两核换指针对绑即可 ——
     `stage(local, own=对端 staging, peer=对端 staging)`(写对端 + 读对端 consumed 当 credit)
     + `reduce(out, own=本端 staging, peer=本端 staging)`(等**本端** produced、读**本端** slot)。
   - **前置条件(2026-10-17 实测挡住)**:`peer_import` 的别名是**只读**授予
     (`state.rs::peer_import`:"只授本端设备 R")⇒ 经它写对端 staging 当场
     `CUDA_ERROR_ILLEGAL_ADDRESS`(基准 `OWL_ARB_PUSH=1` 探针 0.9s 内响亮失败)。
     ⇒ 要评估 push **必须先落"可写对端映射"(RW import)**,而该面正是
     **3080 类 `cuMemSetAccess` 老雷区**(见 `docs/llama.cpp/3080-p2p驱动攻坚-setAccess205-定位实录.md`)。
   - 预期收益:远程**写**是 posted、远程**读**要往返 ⇒ 若写侧能到 13.2GB/s,
     AR 税 ≈ 100µs/token ⇒ prefill ~1600 t/s(仍需重叠才够 2000)。
2. **重叠(结构性)**:AR 在关键路径上(GEMM → AR → 下一层 GEMM)。链路的唯一
   断法是**按 token 微批**:把 prefill chunk 拆两半做软件流水(半 A 的 AR 与半 B 的
   GEMM 并行)。代价 = 权重访存翻倍 —— 而 prefill 是算力受限(27B/rank ≈7.6GB
   权重 = 10ms,对 512 token 的 ~264ms 算力 ⇒ 权重账 4%)⇒ **代价可忽略**。
   预期:AR 全藏 ⇒ TP2 ≈ TP1/2 ≈ **1900 t/s** 量级。

### 25.8 测量面纪律(与 §24.6 五纪律并列;本轮新增)

- **⑦ host/设备必须分离计时**:本引擎每次 `launch` = 一次同步往返(~35-37µs)
  ⇒ 小载荷档的"设备时间"会被 host 地板淹没;**`dev ≈ 0` 只是上界不是零**。
  (生产 decode 不付这笔 —— 整步走**图回放**(单发射),故 §22.6 的 37µs/AR 可信。)
- **⑧ 反物理读数先查自己**:若"更大载荷更快"/"更小载荷更慢",先怀疑**探针结构**
  (本轮两次命中:M-1 的 host 地板、M-2 的双循环余数慢路),不要先写"缓存/干扰"。
- **⑨ 对照臂要用**旧代码**跑同一张卡**:本轮若只看 3080 对的绿,会把
  §25.4 的可见性洞带进生产 —— 是"A/B(旧核 × 0/1)= 绿"这笔把责任钉到本次改动。
- **⑩ 别把"不同卡对"的绿当普适**:同一二进制 0/1(同 PHB)红 / 2/3(跨 root complex)绿
  ⇒ **门要在两对上都跑**(本机四卡两对,成本 ~20s/次)。

---

## 26. ★ 通道审计:数据面**未到**天花板(2026-10-17;参照实现侦察 + 独立分母)

**立案口径(用户 2026-10-17)**:"先确定通道的延迟和带宽**有没有**达到要求、还有没有
通信算法上的空间 —— 有空间就不许把账推给算子;通信是通信自己的目标。"
§25 只回答了"把明显的病(零 MLP)治好了没",**没有**回答"离极限还有多远"。
本节补上那个分母,并对照业界自研通信框架逐条比对。

### 26.1 为什么要外部参照:§25 的 `RR` 探针**不能**当天花板

§25 加了 `RR`(两 rank **同时**互读),想用它当"AR 数据面上限"。**它不成立** ——
`RR` 用的是 **owl 自己的核**(单 block × 256 线程 + 8 路展开)⇒ 它只证明
"**我们这套读协议**能跑到多少",不能证明"链路能跑到多少"。
⇒ 新工具 `tools/nccl-ar-probe.cu`:同一对卡、同载荷、同 dtype(f16)、
**NCCL 2-rank all-reduce** 的 `cudaEvent` 计时(fork + 管道传 uniqueId,无 MPI)。
**这是分母。**(方法论:**"上限"必须用与被测对象无关的实现量,否则量的是自己。**)

### 26.2 ★ 定谳:差 **1.8×**(数据面),小载荷差更多(延迟面)

| 载荷(f16)| **NCCL** µs | owl AR µs | **owl/NCCL** | NCCL MB/s | owl MB/s |
|---:|---:|---:|---:|---:|---:|
| 10KB(1 行,decode)| **10.4** | (host 地板,不可比)| — | 936 | — |
| 1.31MB(128 行)| **134.9** | 185.9 | 1.38× | 9264 | 6723 |
| 5.24MB(512 行)| **464.0** | 863.5 | **1.86×** | 10775 | 5791 |
| 10.49MB(1024 行)| **900.7** | 1626.8 | **1.81×** | **11103** | 6147 |

- 两对卡**逐点一致**(3090 Ti 对:10.4/134.7/460.9/904.2µs)⇒ 与卡对无关;
- owl 的 `RR`(双向互读)= 7.2GB/s **不是**链路极限,只是读协议的上限;
  单向 `R` = 8.2–8.7GB/s;**NCCL 11.1GB/s**。
- **小载荷面更悬殊**:NCCL 10KB all-reduce = **10.4µs**,而 owl 生产 decode
  实测 ≈ **37µs/AR**(§22.6,图回放口径,不含 host 往返)⇒ **延迟面差 ~3.5×**。
- ⇒ **§24.5 说的"通道是瓶颈"在本节得到量化:通信侧还有 1.8× (带宽) / 3.5×
  (小载荷延迟) 的**算法与结构**空间,不是"已经到极限"。**

### 26.2b ★★ 定谳升级:搬运该交给**拷贝引擎**,而我们现在用 SM 的 `ld.global`(用户口径)

**来源:用户 2026-10-17 提醒** —— "GPU server 有三条流,一条计算、两条单向搬运,
你都可以用;两条卡四条流**对班**,双向并行,不要双向串行。"

于是补了两个探针(协议面旁路,只量搬运):

| 探针 | 做法 | 量的是什么 |
|---|---|---|
| **C1** | `memcpy_dtod_async(对端 staging → 本端 payload)`,单边循环 | **拷贝引擎(CE)** 的单向对端读 |
| **C2** | 两 rank **同时**互拷(各走自己的流) | "两卡各一条搬运流"的四流并行数据面 |

实测(3080 对;f16;`memcpy_dtod_async` = 驱动 D2D,走 CE):

| 载荷 | owl AR(现状)| `RR`(SM 双向读)| **`C2`(CE 双向)** | **C2 有效** | NCCL |
|---:|---:|---:|---:|---:|---:|
| 1.31MB | 186.0 | 107.7 | **97.3** | **12854 MB/s** | 134.9 |
| 5.24MB | 868.6 | 656.8 | **461.1** | **10845 MB/s** | 464.0 |
| 10.49MB | 1625.7 | 1335.6 | **972.7** | **10281 MB/s** | 900.7 |

(3090 Ti 对逐点一致:12.5 / 11.0 / 10.7 GB/s ⇒ 与卡对无关)

**三条结论**:
1. **CE 搬运 = NCCL 量级(10.3–12.9GB/s),比我们 SM 读快 1.4–1.5×** ⇒ 对端搬运
   **不该用 `ld.global` 做**(占 SM、延迟受限、单队列);该交给**拷贝引擎**。
2. **拆账**:5.24MB 的完整 AR = **868µs**,而**纯搬运只要 461µs** ⇒ 多出的
   **407µs 全是协议结构**(staging 拷贝 + 两核串联 + 两次跨卡握手)。**两笔都归通信。**
3. **CE 不吃 SM 且能挂另一条流** ⇒ 搬运可与计算流**真并行** —— 即用户说的
   "两条卡四条流对班";这把 §25.7 里"靠按 token 微批才能重叠"的前提也松开了
   (搬运自己就能藏到别的流后面)。

⇒ **`ar_v2` 的地基因此改写**:不是"把 SM 核写得更快",而是**换通道** ——
`stage`/对端交换走 CE(另一条流 + 事件序),SM 只留必要的小核(累加/发布)。
这也与 §26.3 的参照一致(NCCL/SGLang 都不是"用 SM 跨 PCIe 读"这一条路)。

### 26.3 参照实现侦察:两条路的写法(源码在树,逐条核对)

**vLLM `csrc/custom_all_reduce.cuh` + `custom_collective_common.cuh`**(本工作区
`packages/vllm`;SGLang 有同源分支):

| 设计点 | vLLM 做法 | owl 现状 | 判定 |
|---|---|---|---|
| **world=2 的算法** | **恒 1-stage**(`if (world_size_==2) 1stage`)—— 每线程读**对端输入缓冲**同址累加,**不拷贝 staging** | staging 拷贝(本端 R+W)+ 读对端 staging + 本端 RMW ⇒ **内存流量 6B vs 3B** | **可抄** |
| **向量化** | `packed_t<T>::P = 16B`(`ld.128/st.128`) | 16B uint4 ✓ | **已对齐** |
| **grid/block** | `512` 线程 × `min(36, ceil(size/512))` block(`kMaxBlocks=36`;"太多 SM 会争总线") | **1 block × 256** | **可抄** |
| **同步原语** | `ld.acquire.sys` / `st.release.sys`(**硬件 acquire/release**) | `__threadfence_system()`(=`membar.sys`)**全 256 线程各来一遍** | **可抄** |
| **握手参与者** | `threadIdx.x < ngpus`(2 线程) | thread0 自旋 + 全体 fence | **可抄** |
| **多 block 的游标** | `start[kMaxBlocks][ranks]` / `end[...]` **每 block 独立 flag** ⇒ 无需跨 block 归约 | 单写者游标 ⇒ 强制单 block | **可抄(关键)** |
| **buffer 注册** | `register_buffer`/`register_graph_buffers` + IPC handle,设备侧指针对表 ⇒ **图可捕获** | 启动期 VMM 授予固定 staging | **可抄** |
| 累加序 | "不重排地址 ⇒ 各 rank 累加序相同 ⇒ 逐位相同" | 同律 ✓ | **已对齐** |

**SGLang `device_communicators/custom_all_reduce_v2.py`**(更新的一代):

- 拆成 **PushPlane / PullPlane / Communicator**,核是
  `(Communicator, input, algo, graph_params, use_multicast)` 的纯函数,
  **三种算法:`1shot_push` / `1shot_pull` / `2shot_pull`**;
- **PullPlane 的注释直说了**:pull buffer 存在**只是因为**要接"非对称内存"的输入,
  "**K3 fused collectives bring their own symmetric input and borrow the
  semaphores alone**" ⇒ **输入本身是 symmetric memory 时不 staging**;
- 派发阈值(非 SM90/100 的保守缺省):`one_shot_push ≤ 1MB` / `…pull ≤ 16MB`
  / 之上退 NCCL;**`num_push_blocks = num_pull_blocks = num_sm`(全 SM)**;
- 阈值出处 = 它们的 `bench_custom_all_reduce.py` 扫描表 ⇒ **算法按尺寸分派是常态**。

⇒ **两条参照都指向同一组结论**:①**不做 staging 拷贝**;②**写多读少**(push/NCCL
都是 write-based);③**多 block + 硬件 acquire/release**;④**对称内存是地基**。

### 26.4 诊断:owl 差在哪(四条,全部可归因到 §25/§26.2 的读数)

1. **拷贝**:`stage` 核把 payload 搬进 staging(本端 R+W),`reduce` 再读一次本端
   —— 参考实现**根本不搬**(直接读对端 payload,或对称内存里就地读写)。
2. **两核串联**:`stage_i → reduce_i` 是两次核 + 两次跨卡握手(credit/release)
   串在关键路径上;参考 = **1 个核 + 2 个 barrier**。
3. **读而非写**:PCIe **读是非 posted**(要往返),**写是 posted**。参考实现
   (NCCL ring、SGLang push)以写为主 ⇒ 11.1GB/s vs 我们 6.1GB/s。
4. **单 block + `membar.sys` 全线程**:MLP 靠核内展开(§25 已修到 6.1GB/s),
   但**没有 SM 级并行**;且 256 个线程各自 `membar.sys` 是纯开销。

### 26.5 重写方案(通信面;名字待定 = `ar_v2`)

**地基 = 对称内存**:每 rank 一块 VMM 授予的**可写**对端映射(现在 `peer_import`
是只读 —— 见 §25.7);AR 的 payload 就落在它上面(**GEMM 输出直接进对称缓冲**,
stage 拷贝这一刀整条消失)。
**算法 = 按尺寸分派**(照 SGLang 的形状,阈值自己扫):
- `1shot_push`(小载荷,decode 面):把本端部分和**写**进对端对称槽 → 本地归约;
- `1shot`(world=2 大载荷):读对端 + 就地累加 —— 但要在 write 上做(见下);
- **两支都要先把"每次 AR 的跨卡往返次数"从 2 降到 1**(barrier 只留一个:
  "对端已写/已读" 用 `st.release.sys` 发布,`ld.acquire.sys` 观察)。
**发射 = 多 block**(每 block 独立 flag 槽,照 vLLM 的 `start/end[kMaxBlocks][ranks]`),
同步原语换 `ld.acquire.sys`/`st.release.sys`,握手只让 `tid < 2` 参与。
**确定性**:world=2 是逐元素加 ⇒ 不重排地址即天然逐位一致(与 §25 同律);
多 block **不改变每元素的累加序**(只是分区),G8-a 判据不受影响。

### 26.6 预期收益(诚实账)与风险

**收益**(按 §26.2 的分母,线性外推;不是"数量级"):
- prefill(AR 税 208µs/token):6.1 → 11GB/s ⇒ 115µs/token ⇒ 531+115 = **646µs
  ⇒ ~1550 t/s**(现 1353, **+14%**);
- decode(**延迟面**才是这里的主战场):37µs/AR → NCCL 量级(~10-15µs/AR)⇒
  4.7ms → ~1.5ms ⇒ 步时 23.1 → 19.9ms ⇒ **~50 t/s**(现 43.3, **+16%**);
- 两者都**纯通信侧**,不向算子要东西。

**风险与前置**:
- **RW 授予**(读写对端映射):`peer_import` 现在是只读。**但 NCCL 在同一对卡上
  跑到 11.1GB/s 双向装配成功 ⇒ "这对卡支持可写对端映射"已被外部证据证明**
  (NCCL 走的是它自己的 IPC 路;owl 走 VMM fd 路,仍需一次探针把未知变已知)。
  探针 = §25.7 的 `OWL_ARB_PUSH=1` 臂 + 一次 `set_access_self(rw)` 变体。
- **对称内存 + 图捕获**:payload 落到对称缓冲 ⇒ 地址固定 ⇒ 对图友好(反而比现在
  更稳);但**每层的 payload 块必须来自对称池**(池容量与 §6 预算账要重算)。
- **回滚位**:`ar_v1` 原样保留(档位开关),新臂进门后再切。

### 26.7 通信测试的归属(用户口径:测试要在通信包里)

**已落(2026-10-17)**:两个真机套件 `git mv` 进通信包
(`crates/backends/cuda/tests/comm_ar_parity.rs` → `crates/backends/comm/tests/
ar_transport_gate.rs`;`…/comm_ar_bench.rs` → `…/ar_channel_bench.rs`),
`owl-comm` 加 **dev-dependency 回指 `owl-cuda` + `owl-kernels`**(Cargo 允许 dev 环,
Cargo.toml 里已写明理由)。⇒ **通信的尺子与门都住在通信包里**,
且**不需要加载模型**即可迭代(用户口径:通信调试要快)。

现状:纯协议面的单元测试在 `owl-comm` 的 `src/ar.rs`;真机面在新位置;
`crates/kernels/tests/comm_ar_smoke.rs`(单卡自环 + 超时负例)留在原处
(它验的是**核自环语义**,不依赖 peer 装配)。目标形态:

- **纯协议面**(布局/分块/调用序/不变量)留 `owl-comm` 的**单元测试**(已有);
- **真机面**新增 `crates/backends/comm/tests/`:①**带宽/延迟审计**
  (R/S/RR + NCCL 分母对照的读数快照)②**算法正确性**(逐位对拍/确定性/图回放)
  ③**跨卡往返审计**(每次 AR 的跨卡往返次数 —— 延迟面的直接判据)。
  ⚠️ 依赖方向:`owl-cuda` 依赖 `owl-comm` ⇒ 若要在 `owl-comm/tests` 里直接起
  2 rank,需要 `owl-comm` 的 **dev-dependency 回指 `owl-cuda`**(Cargo 允许
  dev 环),或把真机套件放在 `owl-cuda/tests/` 并在命名上归口(取一,别两套)。
  **本节的读数快照先落在 `docs/bench/data/owl-ar-audit-2026-10-17/`。**

---

## 27. `ar_v2` 施工设计:换通道(CE + 分派 + 捕获安全的事件序)

立项依据 = §26(审计:数据面差 1.8×、小载荷差 3.5×)与 §26.2b(CE 真机 10.3–12.9GB/s)。
**用户口径 2026-10-17**:①再参考 vLLM / llama.cpp / SGLang 的优化技巧;②复合命令可以;
③**★ 三流之间的同步要特别小心 —— "传完数据立刻去算那个算子"这一类错,本仓已经有
很多次了,算之前必须同步。**

### 27.1 三方参照对照(源码在树,逐条核过)

| 设计点 | vLLM | llama.cpp(`ggml-cuda/allreduce.cu`,**与我们硬件同类**:无 NVLink 的 PCIe 消费卡) | SGLang | owl 现状 |
|---|---|---|---|---|
| world=2 算法 | **恒 1-stage**(不拷贝) | **按尺寸分派两臂** | 三臂分派(push/pull/2shot) | 单一 pull + staging |
| 小载荷(decode) | 1-stage | **核内忙等 host flag**(省发射) | `1shot_push`(≤1MB) | SM 读 + 自旋 |
| 大载荷(prefill) | 1-stage | **CE 分块 D2H+H2D + 流间事件 + 小 add 核**;chunk≈n/4(4 块在飞) | `2shot_pull` | SM 读 |
| 对端交换媒介 | 直读对端输入缓冲 | **pinned host 中转**(兼容优先) | 对称内存 | 本端 staging + 读对端 staging |
| **跨卡 flag 位置** | 设备内存(每 block 独立槽) | ★ **pinned host 内存**(每 (slot,rank) 一个 int) | 设备内存 semaphore | 设备内存 header |
| 信号机制 | `st.release.sys`/`ld.acquire.sys` | **volatile store + `__threadfence_system()`**;★ 明说 `atomicAdd_system` 在无 NVLink 的 PCIe 消费卡上**不可用** | `__scoped_atomic_*` | volatile + `__threadfence_system()` ✓(同款) |
| 事件 | — | ★ **一律 `cudaEventCreateWithFlags(DISABLE_TIMING)`** + 按 (slot,类型) 事件池 | graph pointer table | `wait_stream_on_tail` **新建事件、不池化** |
| 多 block | 每 block 独立 flag 槽(36×512) | 核内多 block | `num_sm` block | **1 block × 256** |
| 捕获 | IPC 指针对表 | CE 路径靠事件 | CUDA-graph 指针表 | 单流捕获(跨流等待被禁) |

**三方共识(可直接采)**:①**算法按尺寸分派**;②**大载荷搬交换 CE、核只做加法**;
③**小载荷把往返压到最低**(-host flag 就近);④**多 block + 每 block 独立槽**。

### 27.2 ★ 跨流同步契约(本节是硬前置;用户点名的头号风险)

**已发生的同类事故(本仓,写死)**:
- **E3(2026-10-16)**:`alloc{zero:true}` 的清零在 **COMPUTE** 流,H2D 搬运在 **H2D** 流,
  两流无序 ⇒ 晚到的 memset 把刚落地的搬运抹成 0(302 键 ✗18)。命中面 = **搬运极快的小键**
  —— "大键安全"恰恰是它藏得久的原因。修法 = `note_zeroing`/`wait_zeroing` 按块挂设备端
  事件等待(零 host 阻塞)。见 `docs/arch/三流事件序-设计与迁移.md` §9。
- **塔零案(2026-10-16)**:事件**池化复用**是**错**的 —— CUDA 语义里"事件被 A 流等待、
  同时被 B 流重录"= **UB 且不报错**;池轮转 8 事件下 4B VLM 四图**确定性错值**
  (bird top-1 47 vs 280)。⇒ `wait_stream_on_tail` 改为**每次新建事件**。
- **`wait_stream_on_tail` 在捕获期响亮拒**("跨流等待不可捕获")⇒ 任何"跨流序"设计
  **必须同时回答"图捕获下怎么表达"**,否则到了图里就被拒。

**`ar_v2` 的契约(四条,违反任一条即拒)**:

1. **C1 数据依赖必须显式**:凡"搬运在流 A、算子在流 B"的相邻对,必须在
   **B 上 `StreamWaitEvent(ev)`**,`ev` 在 **A 上** record —— 且 record 点必须在
   **该搬运入队之后**。**禁止**依赖"回执时序"或"host 排队顺序"建立先后(那是 E3 的坑)。
2. **C2 事件必须捕获安全**:跨流事件一律 `CU_EVENT_DISABLE_TIMING`
   (llama.cpp 同款;`cudaEventRecord` 在捕获期只接受 disable-timing 事件)。
   **新建 vs 池化按用途分**:一次性的依赖边**每次新建**(塔零案);
   若要池化,必须证明"同一事件在该 slot 上一次等待已入队完成且本 slot 的序号不绕回"
   —— 默认**池化需要门**。
3. **C3 捕获面二选一(设计时就定,不留"到时候再说")**:
   - **(a) 单流档**:所有 AR 节点都留在 COMPUTE 流(CE 拷贝也发在 COMPUTE 流)⇒
     零跨流、**必然可捕获**、零同步风险。**代价**:失去"搬运与计算并行"。
   - **(b) 双流档**:搬运走单独流 ⇒ **必须**有捕获安全的事件边(上 C2)+
     两流在捕获期**同图汇合**(fork/join)⇒ 需要一次"多流捕获"的**可行性门**先证明。
   **本轮先落 (a),(b) 单独立门**。
4. **C4 flag 的可见性语义与位置同改**:若把 flag 搬到 pinned host(§27.3),
   则 release/acquire 的作用域从"device↔device"变成"device↔host↔device" ⇒
   **必须重跑 G1/G2/G3 与 §25.4 的 acquire 对称性门**(那次就是分工不对称踩的坑)。

### 27.3 分期(每期独立可验、可回滚;`ar_v1` 全程保留)

**A期(本轮;零跨流风险,吃 §26.2b 的 1.4–1.5×)**:
把**对端交换从 SM 核换成 CE 拷贝**,仍**走 COMPUTE 流**(= `copy_block_at` 现有口,
`memcpy_dtod_async`),SM 只保留一个小核做"本地加 + 发布"。
- 拷贝平面:本端 payload →(CE)本端 staging;对端 staging →(CE)本端临时槽;
- 加法则:小核(或直接并进下一算子的 epilogue,列为 B 期的融合口);
- 协议头照旧(§3 的 I1–I5 一字不改)⇒ **门可原样复用**。
- 判据:AR(5.24MB) **868 → ≤650µs**;每次 AR 的跨卡往返 **2 → 1**。
- 理由:这条**不需要任何跨流事件**,所以**不可能**引入新的 E3 式洞 ——
  先把"换通道"的收益拿到手,再谈并行。

**B期(下一刀;需先过 C3(b) 的可行性门)**:
搬运移到单独流 ⇒ 与计算流并行(用户"四流对班")。前置门:**多流捕获可行性门**
(证明 `DISABLE_TIMING` 事件 + 两流 fork/join 能在一条图里捕获并回放 ≡ eager)。
过了才动;不过就停在 A 期(accrue 的收益已足够)。

**C期(与 A/B 并列可做)**:
① **flag 搬 pinned host**(借 llama.cpp:短往返,直接打 decode 的延迟面);
② **多 block + 每 block 独立 flag 槽**(抄 vLLM,解掉"单写者游标 ⇒ 强制单 block");
③ **对称内存**:GEMM 输出直接进对称缓冲 ⇒ staging 拷贝整条消失(需要 RW/对称池,
与 §25.7 的 push 前置同一件事 —— 但 CE 路径**不需要** RW 授予,故 A 期不被它阻挡)。

### 27.4 接口(复合命令;用户已认可此形)

按 §3.2 的立场("跨流序不暴露给客户端;序由**意图复合**保证、策略由 **server 独占**"),
AR 收成**一条复合命令**(与 `UploadAndLaunch` / `LaunchAndHarvest` 同族):

```rust
/// **一次 all-reduce 意图**:server 侧独占"用哪个搬运面/哪条流/怎么定序"。
/// 客户端**不出现**流、事件、槽、staging —— 与 §3.2 的两条复合口同律。
Command::AllReduce { spec: ArSpec, ack: Ack<Result<Bytes, ModelError>> }
// ArSpec 闭集:payload 块 + 偏移/字节 + dtype + 相位(独立或融合进后继)+ 调用序号
```

收益:①每 AR 从"2 条 Launch 命令"降成 **1 条**(decode 的 host 侧发射税直接减半);
②**流的编排留在 server**(改造不动引擎);③与 §3.3 `StagingPool` 同属"槽/序下沉"那条线。
**闭集纪律**:`ArSpec` 不提供"任意步骤序列";新增变体 = 显式加一臂 + 门(照 §3.2)。

### 27.5 门表(新增;与 §5/§19 的门并列)

| 门 | 判据 | 期 |
|---|---|---|
| **A1 通道读数** | `owl-comm` 的 `ar_channel_bench`:AR(5.24MB)≤650µs、每次往返 ≤1;两对卡都跑 | A |
| **A2 逐位/确定性/图回放** | `ar_transport_gate` G1/G2/G3 原样绿(**ar_v1 与 ar_v2 双档都跑**) | A |
| **A3 跨卡往返审计** | 新增:`/perf` 计数"每 AR 的跨卡 flag 往返次数"(延迟面直接判据;目标 1) | A |
| **★ A4 搬运-计算定序门** | 专打用户点名的错类:**"CE 搬运后立刻发射消费核"** 的用例,断言消费核读到的是**搬运后**的数据(小载荷档尤其 —— E3 的命中面就是"搬运极快的小键")。做法 = 拷完立即核读回 + 与源逐位比,循环 200 次 | A |
| **B1 多流捕获可行性门** | `DISABLE_TIMING` 事件 + 两流 fork/join 在一条图内捕获 + 回放 ≡ eager(逐位) | B 前置 |
| **B2 并行收益读数** | 搬运流与计算流并行时 AR 的有效时间下降(且总 step 时间不升) | B |
| **C1 flag 位置门** | flag 搬 pinned host 后 G1/G2/G3 + §25.4 的 acquire 对称性复验 | C |

### 27.6 不变量(与 §3 的 I1–I5 关系)

I1–I5(不覆盖未读 / 不读未写 / 相对游标 / 可见性序 / 两侧序列同构)**一字不改** ——
A 期只换**搬运执行体**(SM 核 → CE),协议语义不变 ⇒ 这正是 A 期风险低的原因。
多 block(C 期)会动 I1 的"单写者游标"前提 ⇒ 那时按 vLLM 的**每 block 独立 flag 槽**
把游标**按 block 分片**,并在 §3 就地改写 I1 的表述(改前先加门)。

### 27.7 ★★ 开工第一测就把 §27.3 的 A 期否掉了:真瓶颈是**本地访存**,不是搬运

`ar_v2` **A 期**(对端交换换 CE + 本地加)动手前按纪律先在基准里做候选读数(`A2`
探针 = `CE 拷(对端 slot → 本端 inbox)` + `本地加(inbox 当 peer 喂给现有 reduce 核)`,
两笔都在 COMPUTE 流,协议旁路):

| 载荷 | v1 reduce(`R dev`)| CE 拷贝(`C1 dev`)| **A2(CE + 本地加)** | 判定 |
|---:|---:|---:|---:|:--|
| 1.31MB | 100.6 | 82.2 | **140.7** | 慢 1.40× |
| 5.24MB | 603.9 | 384.6 | **811.3** | 慢 1.34× |
| 10.49MB | 1281.0 | 881.2 | **1723.6** | 慢 1.34× |

**(含一次伪影排除:首版 A2 让 reduce 读了 inbox 的 slot 1(越界),读数是 845µs;
改成"CE 拷到 reduce 将要读的那个 slot"后仍慢 1.34× ⇒ 结论不是伪影带来的。)**

再把 `L` 探针从 push 门里挪出来单跑(它只需本端 staging,**不需要 RW 授予**),
拆出了真正的大头:

| 载荷 | `L dev`(**纯本地 RMW**,零 PCIe)| `R dev`(= 本地 RMW + PCIe 读)| 差(≈ PCIe 的净贡献)|
|---:|---:|---:|---:|
| 1.31MB | **55.5** | 98.9 | 43 |
| 5.24MB | **425.6**(15.7MB 访存 ⇒ **37GB/s**)| 603.9 | **178** |
| 10.49MB | **878.2** | 1281.0 | 403 |

**⇒ 三条定谳(全部实测)**:

1. **v1 的 reduce 不吃 PCIe**:5.24MB 档里 **本地 RMW 占 425.6µs(70%)**,
   跨 PCIe 那 5.24MB 只多付 **178µs(30%)** —— 因为两者在**同一个核内重叠**。
2. **同一颗 AR 的两个核,本地效率差 3 倍**:`stage` 核(1 load + 1 store / 16B)
   = **114GB/s**;`reduce` 核(2 load + 1 store / 16B + 8 路展开的寄存器压力)
   = **37GB/s**。而 3080 的本地带宽能力是 **500+GB/s** ⇒ **我们离自己的本地带宽
   还差一个数量级,这不是通信算法问题,是"单 block 打不满 MLP"**。
3. **换通道(A 期)因此是负优化**:CE 把"PCIe 读"从"与本地 RMW 重叠"变成"串行前置",
   省下的 219µs(603.9→384.6)抵不过新增的本地往返。**A 期作废,且是数据否掉的。**

**方向修正(取代 §27.3 的分期)**:

| 序 | 事项 | 依据 |
|---|---|---|
| **1(下一刀)** | **多 block + 每 block 独立 flag 槽**(抄 vLLM:36 block × 512 线程,kMaxBlocks=36;我们 1×256 = **72 倍线程差**)| 定谳 2:本地访存差一个数量级;单 block 先打满本地带宽 |
| **2** | **少一次本地访存**(对称内存:GEMM 输出直进对称缓冲 ⇒ stage 拷贝消失;或把 AR 的 add 并进后继算子)| vLLM world=2 恒 1-stage = 内存流量 3B vs 我们 6B;在**本地访存受限**时,少一次访存的价值 ≫ 搬运通道的选择 |
| 3 | **CE / 双流对班**(原 A/B 期)| 定谳 1:PCIe 净贡献只有 30%,换成 CE 的绝对收益(219µs)在多 block 之后才有意义;且它**必须**配双流(B1 门)才有净胜 |
| 4 | flag 搬 pinned host(llama.cpp)| 打 decode 的**延迟**面(与本地带宽无关,单列) |

**方法论沉淀(第 11 条,接 §25.8 的四条)**:
**⑪ 换通道之前先问"当前瓶颈是搬运还是访存"** —— 本轮若不先做 `A2` 与 `L` 两个探针,
就会照着 §26.2b 的"CE 比 SM 读快 1.4-1.5×"直接开工,然后把一次**负优化**当成功劳报上去。
`C1/C2` 量的是**纯搬运**,`R` 量的是**搬运+访存`,三者的差才是"该动哪一边"。

### 27.8 落地记录:多 block reduce(2026-10-17;§27.7 的第 1 条)

**改动**(三处,零协议语义变化):
1. **头布局扩两格**(`ar.rs::hdr` + `ar_reduce.cu` 双侧同改,仍在 64 B 内):
   `WAIT_DONE` u64 @32(**单调** = 本端 `consumed+1`,不是布尔 ⇒ 与游标同源、
   **图回放安全、无需重置**)、`BLOCKS_DONE` u32 @40(完成块计数)。
2. **reduce 核多 block**:
   - 数据循环改**全局线程索引 + 全局 stride**(`blockIdx.x*blockDim.x + threadIdx.x`,
     `+= gridDim.x*blockDim.x`)⇒ 块内仍 16 B 连续,合并访问不变;
   - **两级等待**:`block 0` 去 **PCIe** 轮询对端 `produced`(I2 不变);其余 block 读
     **本端 header** 的 `wait_done`(**本地 L2**)⇒ 不给对端同一行加 N 倍 PCIe 轮询
     (这是"多 block 化"最容易顺手引入的坏账);
   - **最后一块发布**:每 block 收尾 `atomicAdd(blocks_done)`;计数达 `gridDim.x`
     的那一块**回零**并发布 `consumed`(对端 stage 的 credit)。回零在同核内、
     且在全部块加过之后 ⇒ 下一次调用(同流保序)必见 0(图回放安全)。
3. **grid 选择**下沉到 `server/comm.rs::ar_reduce_grid`:"每线程约 8 个 16B 向量",
   夹到 `[1, MAX_BLOCKS=32]` ⇒ 小载荷(decode 10KB)仍是 1 块(不白付两级等待),
   大载荷吃满。stage 族**保持单块**(它已是 114GB/s 的纯拷贝,不是瓶颈)。

**坑(写死,已踩)**:宏体里**不能**用多行 `//` 注释 + 行尾 `\` —— `//` 注释遇 `\`
会**把下一行也吞掉**(cpp 的"多行注释"实现),表现为 nvcc 报
`linkage specification is not allowed` + 变量"declared but never referenced"。
本仓宏内一律 `/* */`,照做。

**读数(3080 对;`owl-comm` 的 `ar_channel_bench`;K=200)**:

| 载荷 | `L dev`(纯本地 RMW)前→后 | `R dev` 前→后 | **AR µs 前→后** | AR MB/s |
|---:|---:|---:|---:|---:|
| 10KB | 0.22 | 0.21 → 0.22 | 68.6 → 72.7 | 134(host 地板) |
| 1.31MB | 55.5 → **0.27** | 98.9 → 94.2 | 186.0 → 192.5 | 6813 |
| 5.24MB | 425.6 → **6.23**(**68×**) | 603.9 → **489.5** | **868.6 → 713.5**(**-18%**) | 7350 |
| 10.49MB | 878.2 → **30.1**(29×) | 1281.0 → **976.7** | **1625.7 → 1346.3**(**-17%**) | 7788 |

- **本地 RMW 37GB/s → 2.5TB/s**(32 块摊开)⇒ 本地加法**几乎免费**;
- `R` 从"本地+PCIe 混合"变成**纯 PCIe 读受限**(5.24MB/489.5µs = 10.7GB/s);
- 门 `ar_transport_gate` G1/G2/G3 **全绿**(逐位 / 两跑同 / 回放 ≡ eager)。

**生产 A/B(std 五点三轮;同二进制;3080 单卡 vs 双卡;`tools/tp-ab-std.sh ab`)**:

| 指标 | TP1 | TP2(§25.6)| **TP2(本轮)** | TP2/TP1 |
|---|---:|---:|---:|---:|
| prefill t/s | 970.62 | 1348.53 | **1431.05** | 1.388 → **1.474×** |
| decode t/s | 31.89 | 43.32 | 43.44 | 1.362× |

**对账**:AR 占 TP2 的 ~28%(§25.7),AR 降 18% ⇒ 总时间应降 ~5%:
`739 × (1 − 0.28×0.18) ≈ 702µs ⇒ 1424 t/s`,实测 **1431** ✓(口径自洽)。

**下一步(按 §27.7 表,已可用基准判定)**:
- **A2 结构(CE + 本地加)现已反超**:多 block 后 `A2 dev`(5.24MB)= **393.3µs**
  vs `R dev` = 489.5µs ⇒ **再省 ~20% 的 AR**。原因是本地加法被摊平后,
  "CE 快 1.4-1.5×"那笔账终于兑现(§26.2b 的结论此时才成立)。
- 再往后 = **AR 与计算重叠**(四流对班;前置 = B1 多流捕获可行性门),
  收益上限 = 把剩下的 ~171µs/token 全藏掉 ⇒ ~1890 t/s。

### 27.9 落地记录:stage 族也多 block(同日第二刀)—— 两半都摊平后 AR −31%

`stage` 族做**同一套变换**(§27.8 的三处里 1、2、3 全适用):两级等待(block 0 读对端
`consumed`,其余读本端 `wait_done`)、拷贝循环全局 tid/stride、**最后一块发布 `produced`**,
`ar_reduce_grid` 同档位。

**坑(踩了,已修)**:credit 门是 `prod >= n_slots` 才做的 —— 初版把"发布令牌"写在
**该分支内**,于是小档(`prod < n_slots`)里 block 0 **不发令牌**、其余 block 空转到超时。
修法 = block 0 **无条件**发布令牌(即使本档无需 credit)。

| 载荷 | `S dev`(stage 纯本地拷贝)前→后 | **AR µs 前→后** | AR MB/s |
|---:|---|---:|---:|
| 10KB | 0.21 → 0.21 | 72.7 → 73.3 | 133(host 地板) |
| 1.31MB | 5.19 → **0.23** | 192.5 → **164.4** | 7603 |
| 5.24MB | 101.77 → **0.36** | 713.5 → **600.5** | **8326** |
| 10.49MB | 215.34 → **9.89** | 1346.3 → **1146.0** | **8726** |

**累计(自 §25 起,同载荷)**:`AR(5.24MB)` 14786 → 868.6 → 713.5 → **600.5µs**(**−96%**,24.6×);
`AR(10.49MB)` 30106 → 1626 → 1346 → **1146µs**(**−96%**)。
**对 NCCL(10775/11103 MB/s)**:8326/8726 ⇒ 已达 **77% / 79%**;对 `C1`(纯 CE 单向读,
381.6µs)仍是 **1.57×** ⇒ **§27.7 表第 3 条(CE)现在是明确的下一刀**。

**生产 A/B(std 五点三轮;同二进制;3080 单卡 vs 双卡)**:

| 指标 | TP1 | TP2(§25.6)| §27.8 | **§27.9(本刀)** | TP2/TP1 |
|---|---:|---:|---:|---:|---:|
| prefill t/s | 970.71 | 1348.53 | 1431.05 | **1477.70** | 1.388 → **1.522×** |
| decode t/s | 31.99 | 43.32 | 43.44 | 43.25 | 1.352× |

⇒ **两刀合计:prefill 236.20 → 1477.70(6.26×);TP2/TP1 由 0.244× → 1.522×。**
门 `ar_transport_gate` G1/G2/G3 每刀后都跑,全绿。

### 28. ★ 定谳:`tp_transport_bits_gate` 的判据**过强**(2026-10-17;同 §21.4-2 的同类错)

**现象**:该门在 3090 Ti 对(0/1)上间歇性红,报 `G8-c② 违约:第 3 个 token 起分歧`。

**取证(决定性,两步)**:
1. **纯通信压力门**(`owl-comm` 的 `ar_stress_gate`;**不加载模型**,2 rank 直接起):
   10KB×300 轮 / 1.31MB×100 / 5.24MB×60 / 5.24MB-f32×30 = **490 轮**,
   **每轮 err==0 且 payload 逐位 == a+b** ⇒ **AR 协议干净,transport 零位贡献**。
2. **失败形态本身排除竞态**:两次失败运行的**两侧序列各自逐字相同** ——
   ```
   跨卡 = [109876, 95833, 104784, 120642, 3709, 147368, 104528, 332, 104528, 332, 96446, 9616]
   同卡 = [109876, 95833, 104784, 144977, 3709, 147368,  96838, 125388, 95969, 1710, 271, 109876]
   ```
   前 3 token 相同、**第 4 个分叉**;而两臂路径一致的那些轮 **`max|Δ| = 0.000000`
   (logits 0/248320 不同)** ⇒ **两条路径各自自洽、可复现**,是**两条配置的差异**,
   不是"随机错"(竞态给不出同一个错串)。

**根因(判据层面,不是代码层面)**:本门的判据是"**跨卡臂 ≡ 同卡臂 逐位相同**",而
**跨卡(2 卡)与同卡(2 rank 挤 1 卡)是两种配置** —— §19.3 纪律 4 明写:
> 跨 TP 度**不承诺等价**;只承诺 ①**同配置** run-to-run 与配置内不变性
> ②vs 单卡的功能/统计等价。

⇒ **"逐位相同"是比纪律允许的更强的判据。** 这与 §21.4-2 的"热 ≠ 冷是预期"**同型** ——
那次已为 `G8-a` 纠正过("**只判配置内唯一,不判热==冷**"),**但 T3 的判据没跟着改**,
于是同一个错在这道门上又犯了一遍。

**修法(照 §19.3-4 与 §21.4-2 的既有裁决,不新立标准)**:
| ∥ | 判据 | 性质 |
|---|---|---|
| ① | **配置内 run-to-run 逐位唯一**(每臂各跑两次,逐位相同) | **硬门**(承诺得住) |
| ② | **跨度 vs 单卡:功能/统计等价**(同 top-5、末步 logits 在**路径一致时**逐位相同) | **硬门** |
| ③ | 跨卡臂 ≡ 同卡臂 **逐位相同** | **降级为诊断**(信息,不做判据;红了打印两侧序列与 `max|Δ|`) |

**纪律(本轮定,接 §25.8 五条 + §27.7 第 ⑪ 条)**:
**⑫ 判据不得强于承诺** —— 写门之前先问"这条路径**承诺**与谁等价?"(§19.3-4 已经答过)。
**"更强的证据"可以记录,但不能升格成判据** —— 那会把"配置差异"判成"bug"。

### 29. CE 刀开工前的最后一道坎(2026-10-17 当场解开):`n_slots` 必须降到 1

要把对端交换换 CE,唯一真正的障碍**不是接口而是寻址**:`memcpy_dtod_async` 的
**偏移是 host 侧烘进去的**(与 `graph_launch` 一样,回放不会变),而 v1 的 ring 让
数据落在 `HDR + (produced % 2) * slot_bytes` —— **奇偶是设备侧游标,host 不知道**。

三条候选(逐条排掉两条):
1. **host 跟 parity** ⇒ **破坏图回放**(烘进图的偏移第 2 次 replay 就错)—— 违反
   §3 的相对游标纪律,✗;
2. **保持双槽、CE 两端固定地址** ⇒ **有竞态**:单缓冲 + `n_slots=2` 的 credit 只
   要求 `peer.consumed ≥ prod-1`,管不住"我的 stage_{i+1} 覆盖掉对端还没拷走的 call i"
   (对端的 CE 拷挂在他自己的 rendezvous 之后,与我的 stage_{i+1} 无同步边)✗;
3. **★ `n_slots = 1`**(其余不动)⇒
   - 槽位恒 0 ⇒ 数据落在 `HDR + 0` = **两端都是固定地址** ⇒ **不烘 parity、图回放安全** ✓;
   - credit 变成**严格一深环**(`need = prod - 1 + 1 = prod`):我的 stage_i 必须等
     "对端已开始它的 stage_i";而**代发 credit(§27.9)是在 credit 检查之前写的**
     ⇒ 两侧**先发布后检查** ⇒ **不死锁** ✓,且此刻对端的 CE 拷_{i-1} 必已完成 ✓;
   - **`slot_bytes` 不动**(仍 32MB,给 `plan_ar` 分块与核内自校验用)—— 于是
     §27.3 里担心的"slot_bytes 双重语义"**根本不需要出现** ✓;
   - 附带:**staging 只要 1 格**(32MB+64B 而非 64MB+64B)⇒ 每 rank 少占 32MB ✓。
   - 代价(要实测):**管道深度 2 → 1**,两 rank 更贴身;若掉速就退回 v1(回滚位在)。

**⇒ 落地清单(四步,全部定点,不再有选择)**:
| # | 改哪 | 改什么 |
|---|---|---|
| 1 | `engine/src/tp.rs` | staging 布局 `N_SLOTS: 2 → 1`;新增一块 **inbox**(VMM `peer_alloc`,头 = `slot_bytes=32MB, n_slots=1`),入 TP 上下文 |
| 2 | `models/.../eval.rs` 注入点 | **2 发 → 3 发**:`stage`(own=本端 staging)→ **CE 拷**(`copy_block_at(&peer, HDR, &inbox, HDR, n)`;两端固定)→ `add`(`reduce` 核绑 `own=peer=inbox`、`slot_bytes` 同 inbox 头;**核零改动**) |
| 3 | 门 | 按 §28 的新判据复跑 + `ar_stress_gate`(490 轮 + 20 轮回放,含多 block)+ `tp_g8_gate`/`tp_determinism_gate` |
| 4 | 读数 | 判据:`AR(5.24MB)` 应 590 → **≤ 450µs**(A2 已量 392);若 `n_slots=1` 的贴身耦合拖慢,则看 `RR`/`AR` 两列决定去留 |

**为什么现在才算"能写"**:前两刀把三块地基铺好了 ——
①**尾部 rendezvous**(CE 拷是主机侧发的,不能自旋,必须由核等)②**代发 credit**
在检查之前写(否则 `n_slots=1` 的严格环会**死锁**)③**两端地址固定**才不碰图回放。
缺任何一块,这刀都会写成"能跑但间歇错"或"回放第二次就错"。

### 30. ★★ 定谳:CE 刀**被捕获白名单挡死**(2026-10-17;跑出来的,不是推的)

§29 把 CE 的**协议/寻址**坎解掉了(`n_slots=1`),按§29 四步落码后**引擎级门当场红**:

```
TP rank0 装配失败:图捕获进行中:该操作被拒
       —— 捕获期仅允许 Launch/Alloc;同步/搬运/图操作会破坏图捕获
```

**实验(一行,决定性)**:把 `alloc_inbox` 里的**头写入(pinned/upload/sync)整段去掉、
只留 `alloc`** ⇒ **仍然同一条报错**。⇒ 被拒的**不是** inbox 的分配/写头,而是
**`copy_block_at`(`memcpy_dtod_async`)本身** —— 报错原文把"**搬运**"与"同步/图操作"
并列点名。

**⇒ 定谳(架构级,与协议无关)**:
**本引擎的图捕获白名单只允许 `Launch` 与 `Alloc`;`Command::CopyBlock` 在图里一律拒。**
而 `ar_v2` 的 CE 拷(§26.2b 量到 392µs vs 590µs 的那一刀)**必须在图里**
(生产的 AR 全在回放图内)⇒ **CE 路径与当前捕获策略不相容**。

**这不是"没做完",是"此路在我们自己的策略下不通"** —— 两条出路,各自要立自己的案:

| 出路 | 内容 | 代价/前置 |
|---|---|---|
| **A. 放开白名单**(让 memcpy 进图) | CUDA 本身支持 D2D memcpy 作为图节点(cudaMemcpyAsync 在捕获窗内合法,且我们的 `memcpy_dtod_async` 走的就是 driver 的 D2D);白名单是**本仓的策略**而非 CUDA 限制 | 要动捕获策略 —— 那是**跨切面**改动(需重新论证"捕获期零不可捕获节点"的不变量 §3.6),且要新立门(图内 memcpy 的 replay ≡ eager 逐位) |
| **B. 到此为止** | 现役 AR(SM 核单段:读对端 + 累加)在**图内路径**上已到本引擎架构的天花板 | 无 —— 已经拿到 590µs / 对 NCCL 77–79% / prefill 6.26× |

**判据依据(为什么可以停)**:本引擎"图内只能用 `Launch`"这条约束下,能做的已经做完:
①单段(SM 核内读+加,本来就比"拷+加两段"少一次访存)②16B 向量 + 8 路展开(MLP)③32 块 × 512 线程
④两级等待 + 最后一块发布 + 硬件语义词待办(**已验等价**)。CE 与 `n_slots=1` 的**全部前置**
(尾部 rendezvous / 代发 credit / 两端固定地址 / inbox 分配)都已落地**并被门验过**
——**只差"图里允许 memcpy"这一条策略**,将来放开白名单时按 §29 四步直接接即可。

**纪律(⑬)**:**"设计闭环"不等于"可落地"** —— 每刀的**前置清单**里必须有一条
"**它要在图里跑吗?图里允许这个命令吗?**"。这一刀我在 §29 把协议、寻址、credit 全
推完了,却漏问了这一条 ⇒ 一轮返工。写进 §27.4 的接口设计纪律:**新命令/新搬运面进图前,
先对一遍捕获白名单**。

### 31. CE 落地实录:白名单放开 ✓,但**"AR 未生效"**(2026-10-17;留立案)

**用户裁决(2026-10-17)**:"通讯集合就是 GPU server 做的事 —— 本来就在开发地基;
你被挡住了,那**你来改**。你已经给上层提供了 allreduce 能力,别动上层就行;
**下层怎么动是我们的事情**。" ⇒ 改 server 的捕获策略,而不是把 CE 判死。

**① 放开捕获白名单(已落,保留)**:`server/mod.rs` 的捕获窗白名单原只有
`Launch/Alloc/MemsetZero/Ledger*/GraphEnd/GraphAbort/Free`;`CopyBlock`/`CopyBatch`
落到 `other => reject`。现已放行,**判据逐条核过**(不是顺手放行):
- CUDA 允许 D2D 异步拷贝作内存搬移节点(`cudaGraphAddMemcpyNode`);
- 本命令**无同步、无回读**(只是 `memcpy_dtod_async` 入 COMPUTE 流);
- **图引用租约自动成立**:`GraphRefs` 的机制是"**捕获窗内每一次 `block_ptr` 解析即登记**",
  而 `handle_copy_block` 本就对 src/dst 各解析一次 ⇒ **无需新增代码**;
- 与相邻核同流保序。
⇒ 这条独立成立、门全绿,单独留存(**不需要 CE 也能用**:将来任何图内 D2D 都受益)。

**② CE 接线(n_slots=1 + inbox + 三段 AR)已跑通"能进图",但语义错**:
- `tp_transport_bits_gate` **绿**(①配置内唯一 / ②跨度 vs 单卡 token 逐位 + logits 0/248320);
- 但 `tp_g8_gate` 的 **L1 判据红**:`rank0 vs rank1 末步 logits **248292 项不同, max|Δ| = 20.04**`
  —— 与 §21.4 记录过的"**AR 一次都没生效**"**逐项同签名**。

**★ 门覆盖律(第 4 次实例,最贵的一次)**:**`tp_transport_bits_gate` 抓不到这个错** ——
它比的是**同一份代码的两次运行**(跨卡 vs 同卡),**对称失效**(两 rank 都不加对端)
在两次运行里都一样 ⇒ 照样逐位相同 ⇒ 绿。**"AR 到底生效没有"只有 L1(两 rank 互比)能判。**
⇒ 纪律:**凡"对称性失效"(两个 rank 同时错同一种),跨配置对拍一律无效,必须用"两个 rank 互比"这一类判据。**

**留立案(下一刀的第一件事)**:CE 三段里"AR 没生效"的根因。已知的排查入口:
①**CE 拷是否真的搬到了数据**(把 `add` 前的 inbox 内容 dtoh 出来与对端槽比 —— 最直接);
②**捕获窗内 memcpy 是否真的进了图**(放行 ≠ 生效;`GraphEnd` 时报的节点数/`cap_prof` 日志);
③`inbox.produced = u64::MAX` 那条**常量**在 `graph_end`/replay 后是否仍在(它不该被谁写);
④`add` 核的 `own=peer=inbox` 绑定下,`slot = inbox.consumed % n_slots` 与 CE 拷的
目的偏移是否为同一格(两端都写 HDR+0,按 §29 应恒等 —— 需实测确认)。
**回退现状**:CE 接线已回退(树绿);白名单放开保留。

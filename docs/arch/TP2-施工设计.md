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

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

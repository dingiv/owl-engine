# TP2 **实录卷** —— 全程施工/战役实测账(2026-10-16 → 2026-10-19)

> **本卷是流水账**:按时间序保留每一轮的立案 / 读数 / 定谳 / 自我更正。
> ⚠️ **读之前先看主卷的"§10 当前形态(收口)"** —— 本卷里**有已被后续轮次推翻的中间结论**
> (典型:`§35.3` 的"1-stage 中性"、`§41` 的"PCIe 写独占资源"、`§49.1` 的"VMM 税 17–20%"、
> `§51` 的"1-stage −4.7%"、`§54.4` 的"decode +5.8%")。**结论以主卷为准,本卷只作取证与复现依据。**
>
> · 设计/契约/门表/不变量/工具手册 → `TP2-施工设计.md`
> · 每个 §N 的标题都带日期与一句话判决,可当索引查。

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

### 32. ★ CE 刀结案:全链落地、门全绿、**实测不划算 ⇒ 回退**(2026-10-17/18)

§29–§31 把 CE(拷贝引擎做对端交换)推到"只差接线",§31 的四探针是最后一道门。
**本轮把四探针全做完、把接线做通、把门跑绿,然后拿到了它的真实价格。**

#### 32.1 §31 四探针:全部答完(**捕获层无罪**)

| # | 探针 | 答案 | 证据 |
|---|---|---|---|
| ② | 捕获窗内 `memcpy` 是否真进图 | **是** | 新门 `graph_memcpy_gate`:T1a 进图(回放后 `dst == src`)、T1b 回放生效(图外清零后仍能复原 ⇒ 不是"捕获期 eager 拷了一次")、T2 与核**保序**(`out = 2·src` 而非旧值)、**T3 跨卡 peer 块**(源物理在对端显存,VMM 别名)回放两次读到最新值 |
| ① | CE 拷是否真搬到数据 | **是**(由 T3 直接覆盖) | 同上 |
| ④ | 目的格是否同一格 | **是** | 同上(两端 `HDR + 0`) |
| ③ | `inbox.produced = u64::MAX` 常量 | **在位**(T3 未消耗它) | 同上 |

**T3 附带实证**:图引用租约(`GraphRefs`)**自动覆盖跨卡别名块** —— 收尾时
`peer_release` 被租约**响亮拒**("块被图引用租约钉住;先 graph_destroy"),
顺序改对即过。这正是 §31 说"租约自动成立"的现场证据。

#### 32.2 ★★ 真根因:`n_slots ≥ 2` 守卫(实证,不是推断)

`ar_stage.cu` 有一条 `*own_nslots < 2u ⇒ err=3` 的守卫(`76e26d3` 加,2026-10-19
历史归并后 = **`264e9b0`**;旧号仍可 `git show 76e26d3` 查),而 CE 段
**必须 `n_slots = 1`**(§29:槽位恒 0 ⇒ 两端固定地址 ⇒ 不烘设备游标 ⇒ 图回放安全)。
两下相撞 = stage 核**一进门就被打死**:从不发布 `produced` ⇒ 对端 reduce 5 s 超时
(err=2)⇒ payload 一个字节都没加 —— 与 §31 记录的"AR 一次都没生效"签名逐步吻合。

**最小复现**(新增旋钮 `OWL_STRESS_SLOTS=1` 跑 `ar_stress_gate`):
```
复现前:[graph-stress] ✗ 回放 1 A 侧 err=2(数据超时)| produced=0 consumed=0
```

**放宽(≥ 1)的安全性论证**(写进两份核的头注 ④,两核**必须同律**):
`n_slots == 1` = 严格一深环(`need = prod`),由两条**现役**机制兜住 ——
(a) **代发 credit** 写在 credit 检查**之前** ⇒ 两侧先发布后检查 ⇒ 不死锁;
(b) **尾部 rendezvous**(发布 `produced` 后等 `peer.produced`)⇒ 单格不会被两端同时用。
v1 语义下同样安全(credit = 对端 reduce 的**真消费**,等到 `prod` 即已读走 `prod` 次)。
**门**:`OWL_STRESS_SLOTS=1` 档 40 轮 × (err==0 且逐位) + 20 轮回放 = 全绿;`=2` 档 60 轮无回归。

#### 32.3 接线与门(全绿)

按 §29 四步落地:`staging` 单格 + **inbox**(`copy_block_at` 的目的 + add 核的输入
—— **本端私有 ⇒ 普通块**,不必 VMM;比 §29 原案省一次创建/fd 交叉/import/cap 额度),
`eval` 注入点两发 → **三发**:`stage → CopyBlock(对端 slot0 → 本端 inbox slot0) → add`
(`reduce` 核绑 `own = peer = inbox`;inbox header `produced = u64::MAX` 让 I2 恒真,
数据就绪由**流序**保证)。

| 门 | 结果 |
|---|---|
| `graph_memcpy_gate`(T1/T2/T3) | 全绿 |
| `ar_stress_gate`(`slots=1` / `=2` 双档) | 全绿 |
| `ar_transport_gate` G1/G2/G3 | 全绿(11 case 三 dtype) |
| `tp_assemble_gate` | 全绿(staging 32 MiB/rank) |
| `tp_transport_bits_gate` | 全绿,且末步 logits **248320 项零差** |
| **`tp_g8_gate` L1(rank0 vs rank1)** | **0 项不同,max\|Δ\| = 0.000000**(§31 那次是 248292 项 / 20.04) |
| `tp_determinism_gate` | 全绿(16 arm 唯一) |

#### 32.4 ★★ 读数:CE 三段**在生产形状下没有收益**

新探针 **`A3`**(`ar_channel_bench`;**CE 三段 = 生产同形**,两 rank 并发、含真实
credit/rendezvous;`A2` 只是"CE 拷 + 本地加"的**单边旁路协议**读数):

| 载荷 | **AR µs**(v1 两段) | **A3 µs**(CE 三段) | 差 | `A3 dev` vs `AR dev` |
|---:|---:|---:|---:|---:|
| 10 KB(decode 一行) | 69.71 | **94.17** | **+35%** ⚠️ | 0.38 / 0.38(**都在 host 地板**) |
| 1.31 MB | 164.14 | 173.01 | +5.4% | 78.88 / 95.78(**A3 快 18%**) |
| 5.24 MB(27B prefill 512) | 600.11 | **595.25** | **−0.8%** | 502.82 / 532.95(**A3 快 5.6%**) |

**生产 A/B(std 五点三轮;同二进制;3080 对)**:

| 指标 | TP1 | TP2(v1,§27.9 基线) | **TP2(CE 接线)** | **TP2(v1,回退后复测)** | CE vs v1 |
|---|---:|---:|---:|---:|---:|
| prefill t/s | 971.51 | 1477.70 | 1493.96 | **1488.80** | **+0.35%**(噪声级) |
| decode t/s | 31.99 | 43.25 | 40.14 | **42.44** | **−5.7%** ⚠️ |

⚠️ **回退后必须复测**(本轮的关键一步):若只看 CE 版对 §27.9 旧基线的 +1.1%,
会把**机器状态/运行间波动**当成 CE 的功劳。同会话回退复测后:prefill 差仅 **0.35%**
—— 即 **CE 在 prefill 上什么也没换来**,而 decode 的 −5.7% 是真实代价。

**两条独立证据互相印证**:
- decode 退化 ≈ `(94.17 − 69.71) × 71 次 AR/token ≈ 1.7 ms/token` vs 实测
  `1/40.14 − 1/42.44 = 1.35 ms/token` —— **同量级同向**(71 次 = 每 token 的 AR 次数,
  由 §25 的"AR 占比 24% / 每次 590 µs"反算一致;差的部分是 SM 读本身也有延迟)。
- prefill 无差 ≈ dev 口径的 5.6% 被"**多一个图节点**"的派发税(71 节点 × ~1.5 µs/token
  ≈ 0.15 ms ≈ 1.6% 总时间)吃掉 ⇒ 净 ≈ 0。

#### 32.5 定谳(为什么必然如此)

**CE 把"与本地 RMW 重叠的 PCIe 读"变成了"串行的前置搬运"。**
v1 的 reduce 核里,对端读与本地加是**同一个核内的重叠**;CE 版则必须
`stage(发布 produced)→ rendezvous → CE 拷 → add` **串行四段**,而且 `n_slots=1`
的严格一深环没有 credit 余量(§29 已知的代价)。**这正是 §27.7 的 `A2` 探针第一次
给出的负优化结论** —— 当时是"多 block 之前",多 block 之后单边 `A2` 反超,
但**双边 + 真实协议**(`A3`)把优势吃回去了。

**纪律 ⑭(接 §25.8 的⑦–⑩、§27.7 的⑪、§28 的⑫、§30 的⑬)**:
**单边探针不能预测双边生产。** `A2`/`C1` 都是**单边、旁路协议**的读数:
①双边时两个方向**共享同一条 PCIe**;②协议(credit/rendezvous)只在完整序列里出现;
③生产在图里跑,多一个节点 = 一次派发税。三者都不在单边探针里。
⇒ **"换通道"类改动的判据必须是"生产同形探针"(本例 = `A3`)或生产 A/B,不能是单边读数。**

**⑮ 回退也要复测**(本轮现埸):只看"新代码 vs 旧文档里的基线"会把**运行间波动**
算成新代码的功劳(本次若不复测,prefill 的 +1.1% 会被写进结论;复测后是 0.35%)。
⇒ **A/B 两臂必须同一会话、同一机器状态下体重跑** —— 这是 `tools/tp-ab-std.sh ab`
存在的理由,别用"上次的基线"代替。

#### 32.6 处置:回退 CE 接线,**保留全部基础设施与证据**

| 项 | 处置 | 理由 |
|---|---|---|
| `n_slots ≥ 1` 守卫放宽(两份核) | **保留** | 原守卫**过强**(论证见 32.2);压力门双档绿;v1 语义下同样合法 |
| 捕获白名单放开 `CopyBlock`/`CopyBatch` | **保留** | 独立成立(判据逐条核过);任何图内 D2D 受益 |
| `graph_memcpy_gate`(T1/T2/T3) | **保留** | 图内搬运的判据面;跨卡 alias 租约实证 |
| `ar_channel_bench` 的 `A3` 探针 | **保留** | "CE 试过且不划算"的**可复现证据**;将来若解决节点税/双流,先看它 |
| `examples/device_rows.rs` | **保留** | CUDA ordinal ↔ 名字/BDF/UUID 排障工具 |
| `OWL_STRESS_SLOTS` 旋钮 | **保留** | 单格环的判据面 |
| `N_SLOTS_CE` 常量 | **保留(标注当前未启用)** | 它承载"n_slots=1 为什么必需"的论证 |
| **CE 接线**(`TpCtx.inbox`/`ce_slot_off`、`TpStaging.inbox`、三段序列、inbox 分配) | **回退** | 实测净负;留着会腐化(无生产用户的两条路径) |

**下一步(真正的杠杆,§27.7 表第 2/3 条的真身)**:
AR 的**总时间**要降,只有两条路 ——
①**少一次本地访存**(对称内存:GEMM 输出直进对称缓冲 ⇒ `stage` 整段消失);
②**把 AR 藏进计算**(按 token 微批软件流水;上限 = 把 ~24% 的 AR 份额全摊掉 ⇒
prefill ≈ 1900 t/s)。CE/换通道**不在**这两条上 —— 这就是它没有收益的结构性原因。

### 33. B1 多流图:能力落地 ✓ / 首个重叠探针 ≈ 0(2026-10-18)

**动机**:§27.7 表第 2 条(把 AR 藏进计算)的上限是"摊掉 AR 那 ~24% 的 prefill 份额"
(⇒ ≈1900 t/s),而它的**结构前提**是"**两条独立链拴进同一张图**"——本引擎的图一直是
单流的(`graph_begin` 恒绑 COMPUTE,`wait_stream_on_tail` 在捕获期**响亮拒**)。

#### 33.1 落地:第四条流 + 捕获期路由(fork/join 分成两件事)

| 件 | 落点 | 说明 |
|---|---|---|
| `STREAM_COMPUTE_AUX`(id=3) | `state.rs`(+ `lib.rs` 导出) | 第二条计算流;平时不发任何东西 |
| **发射路由跟着捕获走** | `launch_stream()`(全新:`handle_launch`/`memset`/`copy_block`/`copy_batch` 四处) | §3.2 的立场(流不暴露给客户端)⇒ 命令面只加"切路由",不加"流字段" |
| `capture_route_to(sid)` | 新命令 | **首次**切到非 origin 流时在 **origin** 记 `DISABLE_TIMING` 事件 + 目标流 wait(= **fork 边**,把第二条流卷入同一张图);之后切换是**纯切换** |
| `capture_join()` | 新命令 | 当前流 → origin 建依赖边(主链等支链),路由回 origin |
| `CaptureState.origin` | 新增 | `cuStreamEndCapture` 必须用**发起捕获**的流(fork 后 `stream` 已变) |
| `sync` / 收摊栅栏 | 三处流表加 AUX | — |

**门 `graph_multistream_gate`(M1–M4 全绿)**:
M1 多流能捕获(两条独立链都在图里)· **M2 join 保序**(第三条核读 AUX 的结果 = 104.0/6.0 而非旧值)
· M3 回放真执行两条链(改输入后数值跟着变)· M4 窗外调用**响亮拒**。

#### 33.2 ★ 实测:图内跨流的"偷时间"在本探针下 = 0

新探针 `ar_overlap_probe`(`ar_stress_gate`;两 rank 并发,1 MiB AR×4 + AUX 上的 add 链×cc):

| cc(AUX add 次数) | 回放时间 | 对照:add 链单独跑 |
|---:|---:|---:|
| 0(只有 AR) | 0.539 ms | — |
| 128 | 0.815 ms | 0.277 ms |
| 256 | **1.081 ms** | 0.552 ms |

⇒ 串行预期 `0.539 + 0.552 = 1.091` vs 实测 **1.081** ⇒ **重叠 ≈ 1%**。

**加做同流对照臂**(`OWL_OVL_SAME=1` ⇒ 计算链也发在 COMPUTE 上,顺序执行):
**1.081 ms —— 与跨流档完全相同** ⇒ **跨流没有换来任何并行**(若真并行,跨流档应明显更快)。

单卡旁证 `M5`(两条 add 链,`single=1.090 / dual=2.165` ⇒ 比 1.99)——
但**该判据本身无效**:add 链是**访存/L2 带宽饱和**的(实测吞吐 1.4 TB/s 两档相同),
并行不会加速带宽饱和的链 ⇒ 它只说明"L2 打满了",说明不了并行度。

#### 33.3 定谳与下一步判据

**能力层**:**多流图正确、可用、有门**(这是本刀的净收获 —— 它把"能不能"变成了"值不值")。
**收益层**:本探针下"AR 与访存密集的核并行"**零收益**。最可能的解释:**AR 的 PCIe 数据
也要穿过 L2**,与本地访存密集的核**争用同一条 L2↔SM 通路** ⇒ 串行/并行时间相同。

⇒ **下一个判据必须换链**:真实 prefill 的计算是 **GEMM(算力受限)**,不是 add(带宽受限)。
**在量出"AR + GEMM 并行"之前,不启动引擎侧微批接线的工程** —— 否则又是"照着文档里的
预期开工"(§32 的 CE 刀教训)。

**纪律 ⑯(接 §32 的⑭⑮)**:
- **fork 边记在链尾 = 人为串行化**(M5 第一版踩的:先发完链再 `capture_route_to`,
  那条 wait 就成了"支链等主链跑完")—— fork 要在**链的起点**,join 只在**真有人消费支链**时用;
- **判据与被测共享同一瓶颈时,该判据无效**(拿带宽饱和的 add 链去测并行度 ⇒ 永远得 2.0);
  测并行度必须让两条链**不争同一条瓶颈**。

### 34. 结构归位:`owl-comm` 合并进 `owl-cuda`(2026-10-18)

**用户裁决**:"你本来就是在写 GPU server —— 通信集合就是 GPU server 做的事;需要什么
能力就在 server 里写出来、暴露接口给上层调。"

**动作**:

| 原 | 现 |
|---|---|
| `crates/backends/comm/src/{ar,peer}.rs` | `crates/backends/cuda/src/comm/{ar,peer}.rs`(`git mv`,历史保留) |
| `crates/backends/comm/tests/ar_{channel_bench,stress_gate,transport_gate}.rs` | `crates/backends/cuda/tests/`(包名 `-p owl-cuda`) |
| `crates/backends/comm/src/lib.rs`(策略骨架) | **删除** —— `CommCap`/`CommBackend`/`CommRouter`/`Topology` 全仓 **0 处使用**(M2 前的规划占位);保留 `CommError`(peer 层在用),写在 `cuda/src/comm/mod.rs` |
| `owl-comm` 依赖边 | 删除(cuda / engine 两侧);workspace members 去掉 |
| `owl_comm::…` 引用 | `owl_cuda::comm::…`(engine 的 `tp.rs` / 门) |

**收益(结构,不是性能)**:**没有跨包约束了**。协议(`comm/`)· VMM 授予(server 的
`Peer*` 三命令)· 两核(`owl-kernels`)· 编排(`engine::tp`)现在可以在同一 crate 内
一起改 —— 下一刀"对称缓冲 ⇒ 1-stage allreduce"正需要这一点:它要同时动
**分配/plan 层**(GEMM 输出落进 peer-shared 区)与 **协议层**(消掉 stage 段),
按老结构就得跨三个包对齐接口。

**验证**:全仓 `cargo check --workspace --tests` 零错;`ar_transport_gate` G1/G2/G3 ·
`ar_stress_gate`(300 轮 + 20 轮回放)· `graph_memcpy_gate` T1–T3 ·
`graph_multistream_gate` M1–M5 · `tp_assemble_gate` / `tp_g8_gate`(L1 = 0 项不同)/
`tp_transport_bits_gate`(末步 248320 项零差)—— 全绿。

**⚠️ 文档口径**:§1–§33 里出现的 `crates/backends/comm` 路径是**当时的落点**(施工记录
按史实保留);现役路径见本表。

### 35. 1-stage allreduce:对称槽 ⇒ 融合核(2026-10-18;落地 ✓ 数值 ✓ **性能中性**)

> ⚠️ **2026-10-19 终极勘误(§60.2)+ 竞态已修(§59.2/§60.1)**:本节设计的
> "GEMM 直写对称槽"在**真实图结构下不成立** —— `tp_role` 由层打在
> `forward_inner(...)` 的**最外层窄视图**(`SliceView`)上,而"落对称槽"的决策
> 发生在更早的 `Op::Call` 臂 ⇒ `symmetric_out_slot` **恒返回 None**
> ⇒ **27B 生产档从未走到本路径**(§60.2 三证据:nsys 零 1stage 核 / arena 已装配 /
> 768/768 AR 注入全 `op=SliceView`)。§35.3–35.5 的"中性/从未执行"由此得到**根因**。
> ⇒ **本节的接线若要复活,必须先解决"判据点与接线点不在同一节点"**(§60.3 的两条路线)。

**动机**:§34 去掉跨包约束之后,做"追 NCCL 协议结构"那一刀。实测账(§32):
`AR`(600.1µs)比 `RR`(531.2µs)多 **68.9µs**,那是 **stage 段**引入的协议;而
NCCL 464.0µs ≈ `C2`(纯搬运极限 465.7µs)⇒ **它的协议开销 ≈ 0**,原因就是
**world=2 恒 1-stage**。

**参照(源码核实)**:`packages/vllm/csrc/custom_all_reduce.cuh`
```cuda
if (world_size_ == 2) { KL(ngpus, cross_device_reduce_1stage); }
// 核体:result[idx] = packed_reduce((const P**)&dp.ptrs[0], idx);
```
⇒ 它**原地累加所有 rank 的缓冲**,因为 `dp.ptrs[r]` 指向的是**对称张量**(启动时 IPC 交换)。
我们缺的正是这一条:**输入缓冲本身对称**。

#### 35.1 落地(四件)

| 件 | 落点 | 说明 |
|---|---|---|
| **融合核** | `cu/comm/ar_reduce.cu` 尾部 `OWL_AR_1STAGE_KERNEL` | 一个核五件事:发布 `produced` → 等 `peer.produced`(I2)→ **原地加对端槽** → 发布 `consumed` → **credit**(等 `peer.consumed` 追上 ⇒ 保护**下一个 GEMM 覆盖本槽**) |
| **槽束** | `models/module.rs::TpArenaSlots` | 2 个 payload 槽 = **2 个独立对称块**(乒乓)+ 1 个 header 块。⚠️ 槽**必须独立块**:分配接口的最小单位是块,同块布局会把"块内偏移"这层新接口扯进来 |
| **握手** | `engine/tp.rs::handshake_arena` | 每侧 3 块(2 槽 + header)→ 交叉 import → header 落地(**捕获窗外**) |
| **换块点** | `eval.rs::symmetric_out_slot` | `UniformRow` 节点的输出**直接落进对称槽**(GEMM 写它 ⇒ 对端可读);注入点按 `out.id` **反查**是否 arena 槽 ⇒ 是走融合核,否则回退 v1 两段 |

装配族:`server::CommAr1StageRuntime`(id `comm_ar_1stage`)+ `family` / `client` / `ops` 三处声明。

#### 35.2 门(全绿)

| 门 | 结果 |
|---|---|
| `tp_assemble_gate` | **两档都绿**:缺省(v1)账 66MiB / `OWL_TP_1STAGE=1` 账 132MiB;收尾三账归零(首跑漏了 arena 六块的释放 ⇒ 门当场捉到 **66MB/rank 泄漏**) |
| **`tp_g8_gate` L1** | **0 项不同,max\|Δ\| = 0.000000**;逐步 max\|Δ\| 序列 `[12.18, 21.93, …]` 与 v1 **逐字相同** |
| `tp_transport_bits_gate` | 全绿(末步 248320 项零差) |
| `tp_determinism_gate` | 全绿(16 arm 唯一) |

⇒ **数值正确性**:1-stage 与 v1 **逐位一致**。

#### 35.3 ★ 读数:同会话三档,**中性**

| 臂 | prefill t/s | decode t/s |
|---|---:|---:|
| **v1**(`OWL_TP_1STAGE=0`) | **1477.66** | 42.14 |
| 1-stage(首版;credit 判据误写严一档) | 1483.19 | 42.25 |
| 1-stage(credit 修正为与 v1 **逐字同式**:`need = produced − n_slots`) | **1479.52** | 42.13 |

三档全在 **±0.4%** 内 = 噪声 ⇒ **省掉 stage 段没有换来收益**。
(同会话对照 = 纪律⑮;TP1 `971.94` vs 历史 `971.51` 一致 ⇒ 机器状态没漂。)

#### 35.4 定谳与待查

结构性收益**本应**存在:少一个图节点(~1.5µs × 71 次/token ≈ 107µs/token)+ 少一次跨卡
同步往返(`AR − RR` = 68.9µs/AR)。**实测净 0** ⇒ 说明 1-stage 引入了等量开销。**最大嫌疑**:

> **GEMM 现在直写 peer-shared 块**(VMM `cuMemCreate`/`cuMemMap` 出来的、对端可映射的内存)。
> 它的**写路径**未必与普通 `cudaMalloc` 内存同价 —— 若慢,GEMM 的代价正好吃掉省下的协议。

**⇒ 下一步验证点(明确、便宜)**:量"写 peer 块 vs 写普通块"的同尺寸 GEMM 吞吐;
或把 1-stage 的输出槽换成**普通块** + 一次 `copy_block_at` 搬进对称槽 —— 那时**更快**
就说明是写路径的账。

**处置(两档都保留)**:
- **缺省走 v1**;开档 `OWL_TP_1STAGE=1`。门里两档断言已写死:**缺省不许有 arena**
  (不许悄悄多占 64MiB/rank),开档必须有。
- 保留 1-stage **不因为它赚钱**,而因为它是**地基**:"对称缓冲"这条线唯一下了地的端到端
  实现(核 · 槽束 · 握手 · 换块 · 门),查"写代价"时它就是试验台。

**纪律 ⑰(接 ⑭–⑯)**:**"结构更对"不等于"更快"** —— 从 NCCL 抄来的协议结构在我们这里
被一条不在纸面上的账(写路径/映射属性)中和。⇒ 结构性改动必须同会话 A/B,而且**收益要
归因到某段具体流量**;不能因为"照最佳实践做对了"就默认它快。

### 35.5 ★★ 修正(2026-10-18 同日):§35.3/35.4 的"中性"结论**作废** —— 1-stage 当时**从未执行**

**现场**:用新加的**路径存在性探针**(`tp_perf_split_probe`,dump `/tmp/tp-perf-*.txt`)读 metrics:
`OWL_TP_1STAGE=1` 档里 `srv.op.owl_comm_ar_stage_f16` / `..._ar_reduce_f16` 各 **1344 次**,
而 `owl_comm_ar_1stage_*` **0 次** ⇒ **跑的一直是 v1 两段**。所谓"性能中性"是 **v1 vs v1**。

**根因(一行级)**:AR 的 payload 主要来自 **`Op::Matmul`**(`eval.rs:998` 的 alloc 点),
而我只把 **`Op::Call`**(`:1090`)接进了 `symmetric_out_slot` ⇒ 对称槽永远拿不到 `out`
⇒ 注入点的 `out.id` 反查永远失败 ⇒ 静默回退 v1。**门全绿**(门验的是数值,而 v1 本来就对)。

⇒ **纪律 ⑱(接 §31 的门覆盖律、§32 的⑭)**:新路径必须有**路径存在性判据** ——
"有门 + 门绿"证明不了"新路径被执行"。落实:`tp.ar.1stage` / `tp.ar.v1` 两个 counter
+ 探针断言(**开 1-stage 时前者 > 0 且后者 == 0**)。没有它,这一刀会被当成"中性"归档。

**顺带修掉的第二处**:`symmetric_out_slot` 返回的 `Bytes` 曾写死 rank=1
⇒ 撞 `cublas_gemm_f16: out 期望 rank = 2 实得 1`;改为 **rank 0 + 调用点 `.view(dtype, rank)`**
(与私有块路径逐参同款)。

**修好后的真实读数**:

| 场景 | v1 | 1-stage | 差 |
|---|---:|---:|---:|
| **eager**(0.8B;`tp_perf_split_probe`,412 tok prefill) | 0.837 s | **0.794 s** | **−5.1%** |
| **生产 std**(27B;同会话;3080 对) | 1475.94 / 42.32 | **1486.60 / 42.51** | **prefill +0.72%** / decode +0.45% |

**路径判据**:v1 档 `fusion=0 · v1=1344`;1-stage 档 **`fusion=1344 · v1=0`**。
`tp_g8_gate` L1 在 1-stage 档仍 **0 项不同**(两 rank 逐位一致;`max|Δ|` 对 TP1 是 24.30
vs v1 的 21.93 —— 归约路径不同,属预期)。

**为什么 eager 省 5% 而生产只有 0.7%**:eager 逐核发射时,"少一个核 + 少一轮跨卡同步"
是**直接省下的墙钟**;生产是**整图回放**,那点省在 512-token chunk 的 27B 计算里被摊薄
(且图的节点派发本就比 eager 便宜)。⇒ 结构性收益是真的,**只是量级由 §35.3/35.4 的
"0"修正为"生产 +0.7% / eager −5%"**。

**⚠️ 同步修正**:§35.3 的三档表、§35.4 的"待查(GEMM 写 peer 块代价)"**都不再是结论** ——
写路径的账从未被真正量过(当时根本没走那条路)。真要做那个实验,先看路径判据再说。

### 36. ★★ 通道审计第二刀:读/写 + 映射路径 —— "0.77× NCCL" 的完整归因(2026-10-18)

**立案**:§32 的 A3 表里,`C2`(CE 双向读)的 **`dev` 列 460.46µs ≈ NCCL 464.0µs** —— 当时
结论是"NCCL 协议开销≈0,它就是纯搬运算术"。**本刀把这个比先修正,再拆开。**

#### 36.1 ★ 先修尺子:`dev` 列不是设备时(它减掉了 host 发射税)—— 两条定谳

`dev = (total − enq)/K`;设备受限时 `total = K·D`(**入队与执行重叠**,`ack` 入队即回)
⇒ **`dev` = D − e**(e = 本档**每次** host 发射往返)。

- **代数自证**(同一档内逐行成立):AR 档 `AR µs − dev = 71.17` = **该档 enq/call**;
  2 发射/次 ⇒ e = **35.6µs**。1.31MB 档 71.89 / 10.49MB 档 70.21 同律。
- **反证(`dev` 不可能是 D)**:`S`(5.24MB 纯本地拷贝)的 `dev` = **0.78µs** ⇒ 若当设备时
  就是 **6.7 TB/s** 的本地拷贝,物理不可能。

⇒ **`AR µs` / `A3 µs` 列(= total/K)才是设备时;单发射臂的真设备时 = `dev` + 35.6µs。**
**作废两条旧结论**:
- §26.2b/§35 的"**C2(461) ≈ NCCL(464)⇒ NCCL 协议开销≈0**"—— C2 真值 **496.1**,
  NCCL 比我们**最快**的双向读臂还快 **7%**;
- §27.9 的"对 C1(381.6µs)仍是 1.57×"—— C1 真值 419.3,对 AR(601.7)= **1.43×**。
- (§32.4 的 AR vs A3 两列同为 `µs` 列,**比较本身有效**,不动。)

#### 36.2 新工具(独立分母,不走引擎)

| 工具 | 量什么 | 口径 |
|---|---|---|
| `tools/p2p-wr-probe.cu` | **写腿 vs 读腿**;SM/CE × 单/双向;runtime 映射 | `cudaEvent` 纯设备时;8 路 16B 展开;**写臂有落地校验**(posted ≠ 落地) |
| `tools/vmm-peer-probe.cu` | **映射路径 A/B**:`cudaMalloc+EnablePeerAccess` vs `cuMemCreate+fd 导入+SetAccess(PROT_READ)`(复刻 `vmm.rs` 同参) | 同程序、同核、同载荷、同偏移(HDR=64) |

#### 36.3 读数(3080 对 2/3;f16;**K=200**;纯设备时)

**★★ 10.49MB 档(生产代表档 —— 见 36.4 的 L2 注)**

| 臂 | 1 向 µs | 2 向 µs | 2 向 GB/s | 双向惩罚 |
|---|---:|---:|---:|---:|
| SM 读 | 835.1 | 949.2 | 11.05 | **+13.7%** |
| **SM 写** | 796.6 | **824.5** | **12.72** | **+3.5%** |
| CE 读 | 806.2 | 833.7 | 12.58 | +3.4% |
| CE 写 | 795.4 | **823.2** | 12.74 | +3.5% |
| *NCCL 2-rank AR* | — | *907.1* | *11.6* | — |
| **owl 现役 AR(§27.9)** | — | **1141.4** | 8.8 | — |

**5.24MB 档(与 §32 同载荷)**

| 臂 | 1 向 | 2 向 |
|---|---:|---:|
| SM 读(runtime) | 405.4 | 476.7(11.00GB/s) |
| SM 读(VMM 导入) | 426.9 | **550.2(9.53)** |
| SM 读(VMM)+`__threadfence_system()` | 442.8 | — |
| SM 写(runtime) | 399.4 | **413.5(12.68)** |
| CE 读(VMM,= owl `C2` 真值) | 419.3 | 496.1 |
| SM 读 2 向(VMM,= owl `RR` 真值) | — | 566.7 |
| *NCCL* | — | *463.2* |

**映射路径差(同程序同核)**:1 向 12.93(RT)vs 12.28(VMM);**2 向 10.63(RT)vs 9.53(VMM)**
⇒ **VMM 导入路径在双向档慢 ~11%,单向档慢 5%**;10.49MB 单向档 RT 12.60 vs VMM 10.45(+21%)。

#### 36.4 ★ 四条定谳

1. **★ 我们是"SM 跨 PCIe 读"这一条腿的**——而且**SM 读在双向档掉 13.7%,写和 CE 只掉 3.5%**
   (non-posted 读在双向时争完成通道)。**这不是"读 vs 写"的绝对差,是"SM 读在双向时的额外税"。**
2. **★ 写腿更快**:10.49MB 双向 **SM 写 824.5µs / CE 写 823.2µs** vs SM 读 949.2 ⇒ **快 13-15%**;
   **并且比 NCCL(907.1)快 ~9%**。5.24MB 双向 SM 写 413.5 vs SM 读 476.7 ⇒ 快 15%,比 NCCL 快 11%。
3. **NCCL 的 907.1µs 落在"我们的 CE/写向"区间(823-834)与"我们的 SM 读向"(949)之间**
   ⇒ **它走的不是 SM 读那条腿**;这解释了它为什么比 `RR`/`AR` 快。
4. **映射路径确实有账**(VMM 导入比 runtime peer 映射双向慢 11%)——但**不是主因**:
   5.24MB 的 VMM 双向读 550.2 vs owl `RR` 566.7 已**一致(3%)**,10.49MB 的 VMM 单向读 1003.9
   vs owl `R` 真值 1010.6 **一致(0.7%)**。
   ⇒ **核体、几何、映射都已对得上;那条 100µs 级的差不是"哪里写错了",是"腿选错了"。**

⚠️ **L2 余温伪影(本刀自己踩的,记档)**:5.24MB 的探针**重复读同一块** ⇒ 命中本端 L2
(3080 L2 ≈ 5MB)⇒ 1 向读数被虚高(405µs);owl 的 `R` 臂**交替 slot 0/1**(工作集 10.5MB)
反而吃不到余温(527.5µs)。**两边的 10.49MB 档(> L2)才互相对得上** ⇒
**这类读数一律取"工作集 > L2"的档,否则量的是 L2 不是 PCIe。**

#### 36.5 待办(一条,带风险)

**push 协议的前置 = "可写对端映射"**:`peer_import` 现在授 `PROT_READ`,要写必须授
`PROT_READWRITE`。`tools/vmm-peer-probe.cu` 已写好该臂(`VMMPROBE_RW=1`),**默认不开** ——
它是 3080 `cuMemSetAccess` 老雷区的邻居(`docs/llama.cpp/3080-p2p驱动攻坚-setAccess205...`),
**跑之前必须确认没有别的 agent 占卡**(硬重置会殃及)。
判据:①RW 授予成功与否(响亮读错误码);②VMM 写向双向带宽是否仍达 12GB/s 量级。
若成立 ⇒ push 协议(融合核换指针对绑 + 对称槽,§25.7 第 1 条)的期望收益:
**AR 从 1141.4 ⇒ 824 量级(−28%),并反超 NCCL 9%。**

#### 36.6 ★★ 补记(同日):**RW 导入跑通了** —— 写腿在 VMM 路径上的完整矩阵

**用户口径(2026-10-18)**:"不会有重置了,但你会得到一个 **OOM** 报错 —— 那个区域只有
**192MB**(BAR1 − 64MB 预留)。" ⇒ 探针改为**独立小块**(`payload + HDR` 向上对齐到 2MiB),
`VMMPROBE_RW=1` 臂实测:

```
[vmm] RW 导入成功: B@A=0x…  A@B=0x…(size 各 6291456;⇒ 可写对端映射是通的)
  VMM-write-1way      403.80 us    12.98 GB/s
  VMM-write-2way      418.61 us    12.52 GB/s
  落地校验     ✓ 全等(两侧各抽样 256 个 16B 向量)
```

**⇒ `cuMemSetAccess(self, PROT_READWRITE)` 在导入的对端映射上是通的**(3080/615.71.09),
**且 posted 写真落地**(逐字节校验)。**§25.7 第 1 条(push 协议)的前置就此解除。**

**★ 完整矩阵(VMM 路径 = 生产同款;纯设备时;K=200)**

| 臂(VMM 映射) | 5.24MB µs | GB/s | 10.49MB µs | GB/s |
|---|---:|---:|---:|---:|
| SM 读 1 向 | 429.6 | 12.20 | 1004.8 | 10.44 |
| SM 读 1 向 + `__threadfence_system()` | 445.7 | 11.76 | 1014.0 | 10.34 |
| **SM 读 2 向**(= owl `RR` 同款) | **550.5** | **9.52** | **1082.2** | **9.69** |
| **SM 写 1 向** | **403.8** | **12.98** | **805.4** | **13.02** |
| **★ SM 写 2 向** | **418.6** | **12.52** | **834.3** | **12.57** |
| *owl 现役 AR(读向 + 协议)* | *601.7* | *8.7* | *1141.4* | *8.8* |
| *NCCL 2-rank AR* | *463.2* | *11.3* | *907.1* | *11.6* |

**三条账(同映射、同核、同载荷)**:
1. **写 2 向 vs 读 2 向**:418.6 vs 550.5 = **−24%**(10.49MB:834.3 vs 1082.2 = −23%);
2. **写 2 向 vs 现役 AR**:418.6 vs 601.7 = **−30%**(10.49MB:−27%);
3. **写 2 向 vs NCCL**:**快 9.6% / 8.0%** ⇒ **NCCL 也没到这条腿的天花板**(13.2GB/s)。

**⇒ "我们只有 NCCL 的 0.77×" 的完整答案:不是协议(§36.4-①的 35µs,5.8%),
是"我们选了 SM 跨 PCIe **读**这条腿,而它在双向档掉 13.7%,写只掉 3.5%"。**

#### 36.7 push 协议的落地形状与诚实预期

**数据面**(照 NCCL/SGLang 的 push 形):对称缓冲分两半 —— `own`(本 rank 部分和)
与 `inbox`(对端写进来的那半)。每 rank:①把自己的 `own` **写**进对端的 `inbox`;
②发布 flag;③等对端 flag;④**本地** `own += inbox`(纯本地 RMW,~42µs @5.24MB)。

- **前置**:对称槽必须**双侧 RW 导入**(本刀已证明可行);`peer_alloc` 已有 RW 自授,
  `peer_import` 加一个 `rw` 形参即可(现役调用点保持 `false`,零行为变化)。
- **诚实账**(不是"抄结构就快"):
  - 数据面 = 写 2 向 **418.6** + 本地 RMW **~42**(可流水掉大半)+ 协议 ~35 ⇒
    **≈ 460-480µs @5.24MB**(vs 现役 601.7 ⇒ **−20~24%**);
  - 生产口径:128 AR/chunk × 147µs = **18.8ms/chunk**(344.4ms 总)⇒ prefill
    **1487 → ~1573 t/s(+5.8%)**;decode 面(延迟腿)另有账;
  - **上限 = 链路 13.2GB/s**,我们现在实际只用 **8.8** ⇒ 写腿把利用率从 67% 提到 95%。
- **与 §35 的关系**:§35 的 1-stage 融合核(**对称槽 + 消 staging**)正是 push 的落点 ——
  同一个核,**只换指针绑定**:`stage(local→对端槽)` 变 `st.global 写对端槽`,
  `reduce` 读**本端**槽。**零新核,但要多一组 RW 导入的对称槽。**
- **不做**:CE 三段(§32 已否,串行税吃掉了数据面优势;而写腿**不需要** CE ——
  SM 写自己就到 12.5GB/s)。

### 37. ★★★ push 落地:写腿进生产(2026-10-18)—— AR −19%,TP2/TP1 1.53× → 1.59×

#### 37.1 第一版(纯指针对绑)是**大负优化**,而且现场就把它判了

按 §36.7 的"零新核,两核换指针对绑"直接做:stage/reduce 的 `own`/`peer` 实参对调 + 别名改 RW。

| 口径 | v1 | 纯对调 push | 差 |
|---|---:|---:|---:|
| 隔离基准 AR(5.24MB,K=100)| 601.7µs | **1225.5µs** | **+104%** |
| 生产 std prefill(27B,3080 对)| 1483.04 | 1315.79 | **−11.3%** |
| 生产 std decode | 42.17 | 34.25 | **−18.8%** |

**根因(结构性,不是调参)**:**对调把 header 一起搬到了对端。** `wait_done`(两级等待令牌)与
`blocks_done`(最后一块发布用的块计数)**是本核内部的块间同步状态**,与对端无关 ——
一旦落在对端内存:

- `atomicAdd(blocks_done)` 变成**远端原子**(16 block/核 × 2 核/AR);
- **15 个 block 紧自旋轮询远端 `wait_done`**(`OWL_AR_SPIN_TIGHT = 4096` 次紧自旋 ⇒ 每 block
  数千次 PCIe 读)⇒ **non-posted 读把 PCIe 完成通道吃光**,正好把写腿的优势全抵消。

#### 37.2 修法:**把"数据块"与"header"分开**(第 7 个实参)

核签名从 6 参变 7 参:`(local, own, peer, data, bytes, slot_bytes, slab_off)`

| 件 | 语义 |
|---|---|
| `own` / `peer` | **header 块** —— 两档**都不变**(本端 header / 对端 header)⇒ 块间令牌恒在本地 |
| **`data`** | **slot 数据块**:stage 写哪里 / reduce 读哪里 |
| v1 绑定 | stage `data = own`(写本端槽);reduce `data = peer`(读对端槽) |
| **push 绑定** | stage `data = peer`(**写对端槽**);reduce `data = own`(**读本端槽**) |

**协议零改动**:credit / `produced` / `consumed` / rendezvous 的**相对比较**在两种绑法下都成立
(两个游标仍按 AR 序锁步,`slot = k % n_slots` 对得上 —— 推导见 §36.6)。

落点:两核宏(`ar_stage.cu`/`ar_reduce.cu` 各 3 实例)· `ops/comm.rs` 六张表加 `data` 槽 ·
`client::comm::{ArStageCall,ArReduceCall}` · `server::comm::launch_ar` · 调用点 `eval.rs` + 三个通信门。

#### 37.3 ★ 修好后的读数

**隔离基准(`ar_channel_bench`;K=100;`OWL_ARB_AR_PUSH=1` 开 push 臂,别名走 `peer_import_rw`)**

| 载荷 | v1 AR µs | **push AR µs** | 差 | v1 MB/s | push MB/s |
|---:|---:|---:|---:|---:|---:|
| 1.31MB | 165.2 | 167.9 | +1.7% | 7566 | 7443 |
| **5.24MB** | **606.1** | **491.5** | **−18.9%** | 8249 | **10172** |
| **10.49MB** | **1140.6** | **943.0** | **−17.3%** | 8767 | **10604** |

⇒ 对 NCCL(463.2/907.1):**0.94× / 0.96×**(从 §36 的 0.77× 提上来)。
剩余 73µs = 本地 RMW(~42µs)+ 协议往返 —— **下一刀在那两笔上**(流水掉 RMW / 压掉一轮跨卡往返)。

**生产 std(同会话;27B AWQ-INT4;3080 对 2/3;`tools/tp-ab-std.sh`;`--ignore-eos`)**

| 臂 | prefill avg | prefill **median** | decode avg |
|---|---:|---:|---:|
| TP1(单卡分母) | 934.45 ⚠️ | **966.06** | 31.98 |
| TP2 **v1**(跑两次) | 1483.04 / 1476.91 | **1486.79** | 42.17 / 42.22 |
| TP2 **push** | **1535.89** | **1545.54** | **42.67** |

⚠️ TP1 那轮有一个受扰点(4096 档 496 t/s,别的 agent 会话在 3090 Ti 上跑 e2e)⇒ 分母取**中位数**。

- **prefill:1476.9 → 1535.9 = +4.0%**(中位口径 1486.8 → 1545.5 = +4.0%);
- **TP2/TP1:1.53× → 1.59×**(median 口径);decode 42.2 → 42.7(+1.1%);
- **AR 税:193.7 → 167.7 µs/token**(理想 2× = 483.4µs/token)。

**数值面(全绿)**:`tp_g8_gate` L1 = **0 项不同,max|Δ| = 0.000000**(v1 与 push **逐位一致**);
**路径存在性判据**(纪律⑱):`push=96 · v1=0`(开档)/ `push=0 · v1=96`(缺省)。

#### 37.4 纪律沉淀

**⑲(接 ⑭–⑱)**:**"只换指针"的改动,要检查"块间私有状态"有没有被连带搬走。**
本轮把 `own`/`peer` 对调时,`wait_done`/`blocks_done` 这两个**纯本核内部**的令牌跟着去了对端 ——
它们在文档里没被当成"数据面"的一部分,却是**同步面**的一部分。⇒ 判据:
**改绑指针前,先把每个 header 字段按"谁是写者 / 谁是读者"列一遍;写者与读者都在本端的字段,
其载体必须留在本端。**

**⑳ 负优化也要先量再改**:第一版 +104% 是**隔离基准 3 秒量出来的**,而生产 std 一轮要 ~6 分钟 ——
**通道类改动一律先过隔离基准,再上生产**(与 ⑭"单边探针不能预测双边生产"配套:
这里反过来,**隔离基准能预测生产的方向**,只是量级要打折:−18.9% ⇒ 生产 +4.0%)。

**落地开关**:`runtime.tp_push = true`(flavor/运行档;缺省 false)。
⚠️ **push 与 1-stage(`OWL_TP_1STAGE`)互斥**(两者都改写数据腿;同时开会静默走 push)。

### 38. ★★★ 图内跨流 = **完全串行**(2026-10-18)—— 双流藏 RMW 不通,微批(TBO)也没有地基

**立案**:§37 的 push AR = 494µs,分解见下;唯一可挖的是本地 RMW 那 44.6µs。
两条候选:①**双流重叠**(写对端 ‖ 本地 RMW);②**远端 reduce-add**(`red.global.add`
直接累加进对端 ⇒ 本地 RMW 整条消失)。

#### 38.1 push AR 的完整分解(`ar_stress_gate` 的 `pS`/`pR` 探针;5.24MB,K=100)

| 笔 | 设备时 | 说明 |
|---|---:|---|
| 写腿 `pS`(双边 stage,push 绑定)| **444 µs** | 11.8GB/s;含读 payload + credit/rendezvous |
| 本地 RMW `pR`(reduce,push 绑定)| **44.6 µs** | `payload += 本端槽` |
| **两核串联 + 握手** | **5.3 µs** | ⇒ **协议税≈0,融合两核没意义** |
| = push AR | **494 µs** | |

#### 38.2 两条候选的判据(裸 CUDA 探针 `tools/p2p-rmw-probe.cu`)

| 臂 | µs | 判 |
|---|---:|---|
| W(写对端,单边)| 399.6 | 13.12 GB/s |
| RMW(本地)| 27.6 | 190 GB/s |
| 串行(W → RMW)| 426.7 | — |
| **★ 双流(W ‖ RMW)** | **400.5** | **重叠 97% ⇒ 路 A 在裸 CUDA 上成立** |
| **RED-ADD 对端(f16x2,落地校验 ✓)** | **472.6** | **比纯写慢 18% ⇒ 路 B 否掉** |

#### 38.3 ★★★ 但**图内**是另一回事:跨流完全串行

生产是**图内回放** ⇒ 必须量图内。新探针 `ar_stress_gate::push_graph_overlap_probe`
(**真形状**:PCIe 写核 ‖ 本地 RMW 核;两条链的 header 互不重叠 ⇒ 一笔块就够):

| 载荷 / 链长 | 写链(AUX)| RMW 链(COMPUTE)| **双流** | 串行预期 | **比** |
|---|---:|---:|---:|---:|---:|
| 5.24MB × 128 | 105.09 ms | 5.42 ms | **110.54** | 110.51 | **1.000** |
| 1MiB × 512 | 260.88 | 9.05 | **270.01** | 269.94 | **1.000** |
| **64KB × 512**(访存可忽略)| 17.77 | 4.49 | **22.08** | 22.26 | **0.992** |

**⇒ 与载荷无关、与 PCIe/L2 争用无关 ⇒ 是**图回放层**的串行化。**

**再排除"owl 的捕获有问题"**(裸 CUDA,不经 owl):

| 判据(`tools/p2p-rmw-probe.cu` ⑦)| 比 | 判 |
|---|---:|---|
| 裸 CUDA 图:双分支(写链 ‖ RMW 链)vs 单流 | **1.000** | 不并行 |
| 裸 CUDA 图:**同质**对照(两条**不饱和**的本地 RMW 链)vs 单链 | **1.999** | **确证串行**(同质 ⇒ 与资源争用无关) |

**⇒ 定谳(三条)**:
1. **图内跨流 = 完全串行**(不是"重叠率低",是"零重叠");
2. **不是 owl 的 bug** —— 裸 CUDA 图同样串行(本机 615.71.09 + 3080 对;默认 `cudaGraphInstantiate`);
3. ⇒ **双流藏 RMW 不通**(省不下那 44.6µs);
   ⇒ **微批流水(TBO)在 owl 的图回放上也没有地基** —— 它的全部前提就是"两条独立链拴进
   同一张图并行跑"。**§33.3 的悬案就此负面结案,省下整个引擎侧微批工程。**

**未验分支(留档)**:换 `cudaGraphInstantiate` 的 flags(node priority / device launch)是否
改变调度 —— 本档只量了默认实例化。若将来要重开这条路,先量这一格。

#### 38.4 顺手修掉的两个真 bug(探针过程暴露)

| # | 现象 | 根因 | 修 |
|---|---|---|---|
| ① | **纯搬运的捕获窗被 `graph_end` 响亮拒**("空捕获窗") | 空窗哨兵只数 `CaptureNodes.total`(内核发射),**不数 `CopyBlock`/`CopyBatch`** | 加 `copies` 计数 + `is_empty() = total==0 && copies==0` |
| ② | **图内 D2D 搬运永远落在 COMPUTE** | §33 的落地记录写"四处都跟着捕获走",实测**只有 `handle_launch`/`memset` 两处**用了 `launch_stream()`;`copy_block`/`copy_batch` 仍写死 `stream(STREAM_COMPUTE)` | 两处改 `launch_stream()` + `note_copy()` |

⚠️ **②的副作用面(2026-10-18 复核修正)**:支流上的 **CE 搬运节点必须被显式 join**,
否则 `cuStreamEndCapture` 报 `CUDA_ERROR_STREAM_CAPTURE_UNJOINED`(**整次捕获被拒**)。

**复核实验**(`graph_multistream_gate::m6_capture_join_semantics`,四形状):

| 形状 | AUX 链 | 收尾 | 结果 |
|---|---|---|---|
| A | **核** | 无 join | **✓ 捕获成功** |
| B | **CE 拷贝** | 无 join | **✗ UNJOINED** |
| C | **CE 拷贝** | **在 AUX 上 join**(切回 origin 之前) | **✓ 捕获成功** |
| D | CE 拷贝 + COMPUTE 核链 | 无 join | **✗ UNJOINED** |

⇒ **修正两条**:
1. **不是 join 时机的问题**(A 形状同样没 join 却成功 ⇒ 我原先"join 写错"的怀疑排除);
2. **不是"搬运只能留在 origin 流"**(C 证明支流搬运可行)——
   准确说法是:**支流上的 CE 搬运节点必须显式 join 回 origin**(核节点无此要求;
   机制疑为 `cudaMemcpyAsync` D2D 在捕获期落在一个"origin 够不着的内部流"上)。
   ⚠️ 但 join 只能**在支流上**调用(切回 origin 后 `capture_join` 是 no-op),而 join 边
   = "origin 等支流链尾" ⇒ **"CE 搬运 ‖ 后续 COMPUTE 链"这个形状做不出并行**
   (在 §38.3 的"图内本来就不并行"之下,这不构成额外损失)。

`graph_multistream_gate` 的旧 M6(CE ‖ SM 吞吐)因此**迁出**到
`ar_stress_gate::push_graph_overlap_probe`(纯核链,无搬运);**join 语义本身**留在
`graph_multistream_gate::m6_capture_join_semantics` 当口径门。

#### 38.5 纪律沉淀

**㉑ 负面结论也要有分母**:"双流不行"这句话在 §33 只靠一条**无效判据**(L2 饱和的 add 链)
支撑;本轮把它补成了三条独立证据(真形状 / 不饱和同质 / 裸 CUDA 图)。**否定一条技术路线,
和肯定一条一样,需要同等级的判据** —— 否则要么白做工程(如微批),要么误杀。

**㉒ "裸环境成立" ≠ "引擎里成立"**:裸 CUDA 双流重叠 97%,图内 0%。**差异在框架层**
(CUDA graph 回放),不在代码层。⇒ 凡是"流/调度/并行"类的判据,**必须在目标框架的
执行形态(图内回放)里量一遍**。

#### 38.6 副作用面的**上层影响**审计(2026-10-18,用户提问驱动)

**问**:②把 `copy_block` 的捕获路由改成 `launch_stream()` 之后,**会不会让上层的图捕获失败?**

**答:不会,而且理由是可查的结构性事实,不是"碰巧"**:

| 事实 | 证据 |
|---|---|
| **引擎侧零 fork** | `capture_route_to` / `capture_join` 的调用点**只有 `owl-cuda` 的测试**;`crates/`、`apps/` 下 grep **零命中** |
| 捕获窗初始路由 = origin | `graph_begin` 里 `CaptureState { origin: stream.clone(), stream, … }` —— **同一个 Arc** |
| ⇒ `launch_stream()` 恒等于旧行为 | 无 fork ⇒ `capture.stream == stream(STREAM_COMPUTE)` ⇒ 与改动前逐字相同 |
| 实证 | 改动后已复跑:生产 std(push **1538.01** / decode 42.80)· `tp_g8_gate` 两档 L1 = 0 项不同 · `tp_assemble_gate` / `tp_transport_bits_gate` / `graph_memcpy_gate` / `ar_transport_gate` 全绿 |

**但风险面确实被拓宽了**(诚实记录):改动前 `copy_block` **永远在 COMPUTE** ⇒ UNJOINED
**不可能发生**;改动后**若将来有人 fork 并在支流上放 CE 搬运**,就会撞上它。

**处置:把谜语级错误翻成可操作诊断**(零行为变化,只改失败路径)。`state.rs::graph_end`
的两处 `stream_end_capture` 现在走 `Self::end_capture_err`:

```
graph_end: **捕获窗里有 fork 出去的支流没 join 回来**(驱动原文:…UNJOINED…)。
语义:…卷入的支流必须沿依赖边回到 origin,否则**整窗作废**
(不是「图有个洞」,是这次捕获被驱动拒掉 —— 发生在**捕获期**,不是回放期)。
· 支流上的**核节点**:不需要 join(实测合法);
· 支流上的 **CE 搬运节点**(copy_block/copy_batch = cudaMemcpyAsync):**必须显式 join** ——
  且 capture_join() 只能在**支流上**调用(切回 origin 之后它是 no-op,边不会建);
· 最省事的形态:把 CE 搬运留在 origin 流(别 fork)。
口径门:graph_multistream_gate::m6_capture_join_semantics(四形状)
```

**⇒ 纪律 ㉓(接 ⑲)**:**拓宽一条"不可能出错"的路径之前,先问"谁会在新形状上踩空、他看到的错误可不可操作"。**
本轮 `copy_block` 跟路由是**方向正确**的(§33 的意图就是"发射路由跟着捕获走"),但它把
"永远不可能 UNJOINED"变成了"可能",而驱动原文读不出该改什么 ⇒ **翻译成本极低,不翻译的代价是后人几小时**。

#### 38.7 追问引出的两条修正(2026-10-18;用户提问驱动)

**追问①"流是不是临时拉起的?"** —— **不是**。四条流(`H2D`/`COMPUTE`/`D2H`/**`COMPUTE_AUX`**)
在 **GPU server boot 时一次性建好**(`state.rs` 的建流表);`capture_route_to` 只做
`stream(sid)` **查表 + 记事件**,**零创建**。
**这是必须的,不是优化**:捕获期**不允许建流**(建流不可捕获)⇒ "要用第二条流"只能
"boot 期先建好、捕获期只切路由"。

**追问②"fork 的能力在上层还是 server?"** —— **在 GPU server 里**(命令面
`CaptureRoute`/`CaptureJoin`)。上层虽持有 `GpuClient`,但**语义上这是 server 的能力**;
`crates/`、`apps/` 下 grep **零调用**。⇒ §38.6 的问法("上层图捕获会不会失败")摆偏了,
准确说法是"**GPU server 自己将来若在 fork 状态下发 CE 搬运**" —— 今天零触发者。

**★ 修正:`capture_join` 的静默 no-op 改成响亮拒。**

```rust
if Arc::ptr_eq(&cur, &origin) { return Ok(()); }   // 旧:静默 no-op
```

**这是静默失败**(调用方说"我要 join",实际边根本没建),而且它让 2026-10-18 现场那个
"先切回 origin 再 join"的错写法**一路飘到 `cuStreamEndCapture`** 才以 `UNJOINED` 暴露 ——
**报错点离现场太远**。按纪律(静默失败一律拒)改成:

```
capture_join: **当前已在 origin 流上,join 边不会建**(静默 no-op 已按纪律改成拒)。
`capture_join()` 必须在**支流上**调用 —— 正确顺序:
　`capture_route_to(AUX)` → 在 AUX 上发射 → `capture_join()` → 路由自动回 origin。
若你只是想「切回 origin 继续发射」,用 `capture_route_to(origin)`(纯切换、不建边)。
```

**连带修掉的两处调用点**(它们正是这个错写法):`ar_stress_gate::push_graph_overlap_probe`
的收尾 join(删掉 —— 两条链都是核节点,无需 join;删后复跑 0.999 不变)·
`graph_multistream_gate` 的 M6 收尾 join(删掉,让驱动自己判 UNJOINED)。

**门(全绿)**:`graph_multistream_gate` 的 **M6 五形状** ——
A(核/无 join)✓ · B(CE/无 join)✗ UNJOINED · **C(CE/在 AUX join)✓** · D(CE+COMP/无 join)✗ ·
**E(在 origin 上 join)✓ 响亮拒**。

**纪律 ㉔(接 ㉓)**:**"静默 no-op"是延迟爆炸的定时器。** 一个"我什么都不做"的分支,会让
错误在离现场很远的地方(这里是驱动的 `cuStreamEndCapture`)才现形。**接口上凡是"调用方
表达了一个意图而我什么都没做"的分支,必须拒 —— 且把正确用法写进消息。**

### 39. ★★★★ **捕获路由从未接上核发射** —— §33/§38 的"多流"实验全部作废(2026-10-18)

**触发**:用户追问"流是不是临时拉起的?为什么不一开始就在 server 里申请?" ⇒ 复核流生命周期时,
顺手核了 `capture_route_to` 的**生效面**,发现致命缺口。

#### 39.1 根因:`handle_launch_inner` 写死 COMPUTE

```rust
// server/mod.rs:1497(修前)
let stream = match ctx.stream(STREAM_COMPUTE) {   // ← 写死;从未用 launch_stream()
```

§33 的落地记录写的是"**发射路由跟着捕获走** `launch_stream()`(handle_launch / memset_zero /
copy_block / copy_batch **四处**)"。**实测只有 `memset_zero`/`carve-memset`/`copy_block` 三处**;
**真正的核发射那处从未接上** ⇒ `capture_route_to(AUX)` 对**核发射完全无效**。

**⇒ 后果(全部作废)**:

| 作废项 | 为什么 |
|---|---|
| §33.2 的 M5(add 链 dual/single = 1.99 ⇒ "零重叠")| 两条链其实**都在 COMPUTE** ⇒ 必然串行 |
| **§38.3 的"图内跨流完全串行 1.000/0.992"** | 同上 —— **它测的是同一条流** |
| §33 的 M1/M2/M3(join 保序等)| 仍在(它们验的是捕获/回放语义,不依赖路由),但**"多流"这三个字名不副实** |
| `graph_multistream_gate` 的 M6 形状 A(✓ 捕获成功)| 核没去 AUX ⇒ 支流没工作 ⇒ 不 UNJOINED |

**修**:`handle_launch_inner` 改 `ctx.launch_stream()`(图外行为不变 —— 无捕获时它 = COMPUTE)。
**修后立刻自证**:M6 形状 A(AUX 核链、无 join)从 ✓ 变 **✗ UNJOINED** ⇒ 核**真的**被路由到 AUX 了。

#### 39.2 `capture_join` 的语义修正(两条链都发完之后才 join)

**修好路由后暴露的第二个问题**:"**两条独立分支 + 末尾 join**"这个形状**表达不出来** ——
旧 `capture_join` 只在"当前流 ≠ origin"时建边,切回 origin 后调用是 no-op(已改响亮拒)。

**正确语义**(本轮落地):`capture_join()` = **"origin 等 `involved` 里最后一条非 origin 流的当前链尾"**,
与"当前路由在哪"无关。**关键**:`cuStreamWaitEvent` **只影响之后发射的工作** ⇒ 正确顺序是

```
capture_route_to(AUX) → AUX 链 → capture_route_to(origin)(纯切换) → COMPUTE 链 → capture_join()
```

⇒ 两条链**中段互不等待**,join 只给 origin 的**尾部**建边 ⇒ 既合法(不 UNJOINED)又真并行。

**驱动硬规则(裸 CUDA 四形状实测,与节点类型无关)**:`fork` 出去的支流**只要有工作就必须 join**,
否则 `cuStreamEndCapture` 报 UNJOINED、**整窗作废**。
(A 只有 AUX 链 ✗ · B 只有 origin ✓ · C 两条链无 join ✗ · **D 两条链有 join ✓**)

#### 39.3 修好之后的读数

| 判据 | 结果 | 读法 |
|---|---:|---|
| **裸 CUDA 图**:两条**不饱和**同质链(RMW ‖ RMW)vs 单链 | **1.001** | ⇒ **CUDA 图回放能并行无依赖分支** |
| 裸 CUDA 图:PCIe 写链 ‖ 本地 RMW 链 vs 单流 | **0.972** | 部分重叠(46%) |
| owl `M5`(两条 add 链)dual/single | **1.43×** | 部分并行 —— 但 add 链 L2 饱和,判据仍不干净 |
| **owl `push_graph_overlap_probe`**(PCIe 写 ‖ 本地 RMW)| **1.000** | **仍完全串行** |

**⇒ 结论(诚实)**:
1. **"图回放不能并行"是错的** —— 裸 CUDA 同质对照 **1.001** 铁证;
2. **owl 侧仍是 1.000** ⇒ **owl 与裸 CUDA 之间还有一处未找出的差别**(候选:图实例化 flags /
   事件池复用 / 流优先级 / slab 分配节点的额外依赖边)。**未解,立案。**
3. **§38.3 的定谳作废**;`tools/p2p-rmw-probe.cu` 的 ⑦/⑧ 两格(裸 CUDA 图)现在**有 join**、可复现。

#### 39.4 与"1.7× 目标"的关系(纯算术)

| | µs/token |
|---|---:|
| TP1(966.06 t/s)| **1035.1** |
| TP2 理想 2× | 517.6 |
| AR **双向**地板(5.24MB×128/512 @ 12.7GB/s —— 双向实测值,不是单向 13.2)| **103.3** |
| 非 AR 超额 | 9.7 |
| ⇒ **"AR 全串行"下的上限** | **630.6 ⇒ 1586 t/s ⇒ 1.64×** |
| 目标 1.7× | **608.9 ⇒ 1642 t/s** |

**⇒ 1.7× 必须**藏掉 ≥ 21.7µs/token 的 AR(≈87µs/AR)**,而"藏"的唯一机制就是图内并行。**
⇒ **39.3 的未解之谜 = 1.7× 的必经之路。**

**纪律 ㉕(接 ㉔)**:**"能力落地了"必须以"能力真的生效"为判据,不能以"门绿了"为判据。**
本轮 `capture_route_to` 的命令面、状态机、门(M1-M5)**全都齐全且绿**,
**而它对核发射一次都没生效过** —— 因为没人验过"发射之后核在哪条流上"。
⇒ 新能力上线必须带**"生效面"判据**(本例 = "AUX 上有工作、无 join ⇒ 必须 UNJOINED";
路由没接上时它恒 ✓ —— 一个**反向判据**就能当场抓住)。

### 40. ★ 统计设施:**图结构簿记 → metrics(攒内存,最后集中 dump)**(2026-10-18;用户口径)

**用户口径(2026-10-18)**:"**你先不要着急着打印** —— 做一些统计,写进内存或显存里存着,
**等跑完了再集中把数字打印出来**。这样它就不会消耗你太多性能了。多加一点日志,磨刀不误砍柴工。"

#### 40.1 为什么需要它(§39 现场的教训)

§39 的现场是"owl 图内跨流 1.000(串行)vs 裸 CUDA 0.972/1.001(并行)"。**回放计时只能告诉你
"串行",不能告诉你"图里到底有没有并行分支"** —— 后者要**图结构**,而它**在捕获期一次算完**,
**零热路径成本**。之前的做法(探针里 `eprintln!`)既污染热路径、又散在各处,不成体系。

#### 40.2 落地:两个真源,零驱动调用

| 件 | 落点 | 说明 |
|---|---|---|
| **分支簿记** | `state.rs::CaptureState { aux_nodes, forks, joins }` | 发射时判 `cap.stream != cap.origin` ⇒ `aux_nodes++`(一次整数自增);`capture_route_to` 建 fork 边 ⇒ `forks++`;`capture_join` ⇒ `joins++` |
| **结构统计** | `ffi::GraphShape { nodes, aux_nodes, forks, joins, copies }` + `has_branches()` | **从 owl 自己的簿记生成** |
| **入 store** | `state.rs::graph_end` ⇒ `metrics::counter_add("graph.{nodes,aux_nodes,forks,joins,copies}")` | **捕获期一次**;由 `metrics::dump` 在跑完后集中出表 |
| **出表** | 探针尾部一次性 `eprintln` + `metrics::dump(path)` | **热路径零打印** |

⚠️ **为什么不用驱动的 `cuGraphGetEdges`/`cuGraphNodeGetDependencies`**:两者在本 crate 的
cudarc feature 集下**被门控掉**(实测 `found an item that was configured out`)。
**而图的边本来就是 owl 自己建的**(同流顺序边 + fork 边 + join 边)⇒ **自己的簿记就是真源** ——
不但够用,而且**零驱动调用**。

⚠️ **坑(踩过)**:`metrics` store 是**进程级 `OnceLock`** ⇒ **必须在捕获之前 `init_metrics`**;
事后 init 只能读到 0。

#### 40.3 修好路由之后的读数(`push_graph_overlap_probe`;chain=128,5.24MB)

```
[ovlp]   S 写链(AUX)               105.015 ms
[ovlp]   R RMW 链(COMPUTE)          5.293 ms
[ovlp]   D 双流(两链同图)            110.324 ms
[ovlp]   串行预期 110.308 ms ⇒ 双流/串行 = 1.000
[ovlp] 图结构:nodes=512 aux_nodes=256 forks=2 joins=2 copies=0
```

**⇒ 三条定谳**:
1. **图里真的有 fork 出去的分支**(`aux_nodes=256`、`forks=2`)—— §39 的路由修复**已生效**;
2. **双流仍然 1.000(完全串行)** ⇒ **不是"路由没接上"这类假象**,是真的没重叠;
3. **裸 CUDA 同形图 = 0.972**(部分重叠 46%)⇒ **owl 与裸 CUDA 之间仍有一处未找出的差别**。

**未解之谜的候选(立案,按便宜排序)**:
① cudarc 建流的 flags/优先级(`CudaContext::new_stream()` 的语义 vs `cudaStreamCreateWithFlags`);
② `cuGraphInstantiateWithFlags` 的旗标(owl 用零旗标);
③ 捕获窗内的 **slab 分配节点**(`graph_begin` 会预建 slab)是否给两条链引入了额外依赖边;
④ **最硬的判据 = 设备端时间戳**:给两个核各记 `[globaltimer start, end]` 进显存,
回放后一次性拷回 —— **存在性证明**,比吞吐比干净(设计照 §40 的口径:**攒显存,最后 dump**)。

**纪律 ㉖(接 ㉕)**:**"能测"和"测得准"是两件事。** 本轮前,探针的输出是**散落的 `eprintln`**,
读者要自己拼;而**图结构**这种"只在捕获期可知、热路径不该碰"的量,根本没有落脚点。
⇒ **凡"慢变量"(捕获期/装载期/配置期的量)都该走"攒 → 集中 dump"这条管道**,
与"快变量"(逐 token 的计时)分开 —— 前者零热路径成本,后者才需要探针。

### 41. ★★★★ "为什么图内跨流不并行"——**定谳:是 PCIe 写独占资源,不是图回放串行**(2026-10-18)

**问题**(§38/§39 遗留):owl 的 `[PCIe 写对端 ‖ 本地 RMW]` 双流 = **1.000(完全串行)**,
而同质对照(两条本地链)= **1.001(完全并行)**。到底是"图回放把跨流串行化"还是"资源冲突"?

#### 41.1 判据一:**串行度扫描**(把 RMW 链拉长,看 `D − S` 是否随 `R` 线性)

`OWL_OVLP_RMW_MULT`(COMPUTE 链核数 = `chain × mult`):

| mult | R'(RMW 链单跑)| D'(写 ‖ RMW')| **D' − S** | 吸收率 |
|---:|---:|---:|---:|---:|
| 4 | 20.999 ms | 126.469 ms | 21.427 ms | **−2.0%** |
| 16 | 83.975 ms | 189.237 ms | 84.188 ms | **−0.3%** |

**⇒ `D − S` 与 `R` 严格 1:1** ⇒ **RMW 一点没被吸收** ⇒ **不是"并行但被拖慢",是真串行。**

#### 41.2 判据二:把"SM 写核"与"PCIe 写通道"分开(CE 写臂)

| 臂 | ms | 读法 |
|---|---:|---|
| S 写链(SM `st.global` 写对端,单链)| 105.176 | **6.4 GB/s**(128 核 × 5.24MB)|
| **G CE 写链**(`copy_block_at` ⇒ 拷贝引擎,**不吃 SM**)| **51.129** | **13.1 GB/s** |
| D 双流(SM 写 ‖ RMW)| 110.880 | 串行预期 110.9 ⇒ **1.000** |
| **F 双流(CE 写 ‖ RMW)** | **56.909** | 串行预期 56.9 ⇒ **1.001** |

**⇒ 用 CE 写(完全不吃 SM)也照样串行** ⇒ **与 SM 无关,是 PCIe 写这条腿本身。**

#### 41.3 判据三:**只有"写"有这个问题,"读"没有**

| 臂 | ms | 串行预期 | 比 |
|---|---:|---:|---:|
| **E 双流(PCIe 读对端 ‖ 本地 RMW)** | **106.117** | 110.6 | **0.96(92% 重叠)** |
| D 双流(PCIe 写对端 ‖ 本地 RMW)| 110.880 | 110.9 | **1.000** |
| 同质对照(本地 ‖ 本地,裸 CUDA)| — | — | **1.001(完全并行)** |

#### 41.4 判据四:**裸 CUDA 同形图也一样串行**(排除 owl 实现)

`tools/p2p-rmw-probe.cu` 的同形图 + 同判据扫描(`RMW_MULT`):

| mult | RMW 链单跑 | 单流(串行)| 双分支 | 读法 |
|---:|---:|---:|---:|---|
| 4 | 13.287 ms | 54.147 ms | **62.615 ms** | **比串行还慢 16%** |
| 16 | 53.144 ms | 54.146 ms | **102.449 ms** | **≈ 写链 + RMW 链** |

**⇒ 裸 CUDA 也不并行(甚至跨流有额外开销)** ⇒ **不是 owl 的实现问题。**

#### 41.5 定谳(四条)

1. **图回放**能**并行无依赖分支** —— 同质对照(两条**本地**链)= **1.001**;
2. **PCIe 写**这条腿会**独占某个共享资源**,把另一条流上的**本地访存完全挡住**
   (SM 写 / CE 写都一样;`D − S` 与 `R` 严格 1:1);
3. **PCIe 读不占那个资源** —— `读对端 ‖ 本地 RMW` = **92% 重叠**;
4. **裸 CUDA 同样如此** ⇒ 是**硬件/驱动行为**,不是 owl 的实现缺陷。

**⇒ 对"藏 RMW"的判决**:push 的 AR 是 `[写对端 ‖ 本地 RMW]` —— **天然落在冲突形状里**
⇒ **跨流藏 RMW 这条路死**。§38 的结论方向对(零重叠),但**根因不是"图内串行"**(那条被
41.1/41.4 的"同质对照 1.001"否掉),而是 **PCIe 写的资源独占**。

**⇒ 对 1.7× 目标的影响**:§39.4 的算术(AR 全串行上限 **1.64×**)不变;而"藏 AR"的唯一机制
(跨流)现在被证否 ⇒ **1.7× 需要换一条完全不同的路**(候选见 §42)。

**⚠️ 顺手发现的两个真问题(单列,不与上面的定谳混)**:
- ~~SM 写对端只有 6.4GB/s ⇒ owl 的 stage 核比纯写核慢一倍~~
  **⇒ 被 §41.6 自我推翻:6.4GB/s 是探针写法的伪影。真读数:生产绑定写链 = 12.11 GB/s;
- `capture_join` 的**方向**:本轮为表达"两条独立分支 + 末尾 join",把它的语义改成
  "origin 等 `involved` 里最后一条非 origin 流" —— 与"当前路由在哪"解耦(§39.2)。

#### 41.6 ★ 收尾判据:**生产绑定**的写链(header 留本端)+ 一条自我推翻

**立案**:§41.2 的 `S` 臂(SM 写单链)只有 **6.4GB/s**,而 CE 写 13.1、裸探针 SM 写 13.12
⇒ 怀疑"owl 的 stage 核慢一倍"。

**排查(两条,都跑出来了)**:

| 假设 | 判据 | 结果 |
|---|---|---|
| "读源(读 payload)拖慢了写" | 裸探针加「读本地→写对端」核 | **399.82µs = 13.11GB/s**(与写常量 13.12 一样)⇒ **否** |
| "`__threadfence_system()` 贵" | 裸探针已有 `SM-write+fence` 对照 | 399.41 → 406.03(**+1.7%**)⇒ **否** |

**⇒ 剩下的唯一差别**:`S` 臂是"单边自洽"写法 —— `own = peer = data = 对端块`,
把 **header 也放到了对端** ⇒ 核入口的三次校验读(`magic`/`slot_bytes`/`n_slots`)
+ credit 读 + `atomicAdd(blocks_done)` **全变远端**。**(§37 的教训第三次重演)**

**⇒ 判据:改用生产绑定重测**(`H` 臂:`own = 本端 staging`(header 留本端)、
`peer = 对端别名`、`data = 对端别名`;对端 header 顶满 ⇒ 单边可跑):

| 臂 | ms | GB/s |
|---|---:|---:|
| S 单边自洽写链(header 在对端) | 105.036 | **6.4** |
| **H 生产绑定写链(header 留本端)** | **55.406** | **12.11** |
| G CE 写链 | 51.119 | 13.1 |
| 裸探针 SM 写核 | — | 13.12 |

**⇒ 定谳:`6.4GB/s` 是探针写法的伪影;owl 的写核在生产绑定下 **12.11GB/s**(与 CE/裸探针同量级)。
"stage 核慢一倍"**不成立** —— 记档以免后人重开。**

**⇒ 但"写 ‖ 本地 RMW 串行"这条**不受影响**(用生产绑定重测):

| 臂 | ms | 读法 |
|---|---:|---|
| H 生产绑定写链(单) | 55.406 | — |
| **I 生产绑定写 ‖ RMW'(mult=8)** | **97.010** | `I − H = 41.6ms ≈ R'(41.718)` ⇒ **吸收率 0.3%** |

**⇒ 换成生产绑定、写链快一倍,结论一模一样:PCIe 写与本地访存完全不能并行。**

#### 41.7 §41 终局定谳(五条)

1. **图回放**能**并行无依赖分支** —— 同质对照(两条**本地**链)= **1.001**;
2. **PCIe 写(任何形式:SM 写 / CE 写 / 生产绑定)与本地访存完全不能并行**
   —— 吸收率 **0.3% / −0.8% / −2.0%**(三种写法、三种链长,R 拉长 16× 仍严格 1:1);
3. **PCIe 读能** —— `读对端 ‖ 本地 RMW` = **92% 重叠**;
4. **裸 CUDA 同形图同样串行** ⇒ 是**硬件/驱动行为**,不是 owl 的实现缺陷;
5. **stage 核 6.4GB/s 是探针伪影**(header 放到了对端);**生产绑定下 12.11GB/s**。

**⇒ 对目标的判决**:push 的 AR 是 `[写对端 ‖ 本地 RMW]` —— **天生落在冲突形状里**
⇒ **跨流藏 RMW 这条路死**。§39.4 的算术(AR 全串行上限 **1.64×**)不变,
而"藏 AR"的唯一机制被证否 ⇒ **1.7× 必须换一条完全不同的路**。

### 42. ★★★★ 换到 3090 Ti 对复测 + **1.7× 的现实路径:NVLink 桥**(2026-10-18)

**用户口径**:"你刚才是在双卡 3080 上做的吗?那你现在来双卡 3090 上试一试。"

#### 42.1 三张表(同尺寸 5MiB、同会话;3090 对 = CUDA ordinal **0/1**)

**① 纯传输(torch / NCCL)**

| 臂 | **3090 对** | 3080 对 |
|---|---:|---:|
| torch 单向拷贝 | 408.9 / 406.8 µs(**12.82 / 12.89 GB/s**)| 410.1 / 410.1(12.79 / 12.78)|
| torch 双向拷贝 | **423.7 µs(12.37)** | 424.3(12.36)|
| NCCL send/recv(单向写)| 410.9(12.76)| 417.1(12.57)|
| NCCL sendrecv(双向)| 443.2(11.83)| 444.7(11.79)|
| **NCCL all_reduce** | **455.7–459.3 µs(11.5)** | 461.6–463.2(11.36)|

**⇒ 两对卡的纯传输逐点相同(±1%)** ⇒ **3090 对现在也走 PCIe,没有 NVLink。**

**② 我们的通道(`ar_channel_bench`,5.24MB,K=100)**

| 臂 | **3090 对** | 3080 对 | 差 |
|---|---:|---:|---:|
| AR **v1** | **552.7 µs** | 606.1 | **−8.8%** |
| AR **push** | **481.1 µs** | 491.5 | −2.1% |
| `R`(SM 单边读)| 393.0 | 492.5 | **−20.2%** |
| `RR`(SM 双向读)| 481.4 | 532.3 | −9.6% |
| `C1`(CE 单向读)| 383.3 | 385.1 | 持平 |
| `C2`(CE 双向读)| 455.3 | 465.8 | −2.3% |

**⇒ 3090 对的短板不是 SM 读**(3080 的 SM 读比它慢 20%);写腿两对卡持平。
⇒ 对 NCCL allreduce:3090 **0.97×**、3080 **0.94×**。

**③ 重叠判据(`push_graph_overlap_probe`)—— §41 的定谳换卡后是否成立**

| 判据 | **3090 对** | 3080 对 |
|---|---:|---:|
| `D` 双流(写‖RMW)/串行 | **1.001** | 1.000 |
| **`I` 生产绑定写 ‖ RMW' 吸收率** | **−0.4%** | 0.3% |
| **串行度扫描(mult=8)** | **−2.0%** | −0.8% |
| `H` 生产绑定写链(单)| 54.2 ms = **12.38 GB/s** | 55.4 ms = 12.11 |

**⇒ §41 的定谳在两对卡上都成立** ⇒ **不是 3080 特有,"PCIe 写 ‖ 本地访存 = 完全串行"是共性。**

#### 42.2 ★ TP2/TP1 的比值:**换到 3090 反而更差**

| | 3080 对 | **3090 Ti 对** |
|---|---:|---:|
| TP1 单卡 prefill | 966.06 t/s(1035.1 µs/token)| **1197.40 t/s(835.2 µs/token)** |
| TP2 理想 2× | 517.6 | 417.6 |
| push AR/AR(实测)| 491.5 → **122.9 µs/token** | 481.1 → **120.3** |
| 非 AR 超额 | 9.7 | ~7.8(按比例) |
| ⇒ **TP2 推算** | 650.2 ⇒ 1538 ⇒ **1.59×** | **545.7 ⇒ 1832 ⇒ 1.53×** |

**⇒ 比值由「AR 时间 ÷ TP1 时间」决定 —— TP1 越快,比值越差。**
**⇒ 换到 3090 对不会改善 TP2/TP1,反而从 1.59× 掉到 1.53×。**

#### 42.3 ★★★★ **1.7× 的现实路径:NVLink 桥**

**硬件事实(实测)**:

```
GPU 1/2 (3090 Ti): "NVML: Unable to retrieve Nvlink information as all links are inActive"  ← 桥未装
GPU 0/3 (3080):    "Device does not have or support Nvlink"                                  ← 无此硬件
nvidia-smi topo -m: 3090 对 = PHB(同一 PCIe host bridge)· 3080 对 = NODE(跨 bridge)
```

**⇒ 3090 Ti 支持 NVLink(RTX 3090/3090 Ti 均有 1 个连接器,112.5 GB/s 双向 ≈ 56 GB/s 单向),
**当前未装桥**;3080 **根本没有** NVLink。**

**⇒ 装桥后的推算(按 56 GB/s 单向)**:

| | 现在(PCIe 12.4GB/s)| **装桥后(56GB/s)** |
|---|---:|---:|
| AR 的传输段 | 423.7 µs | **93.6 µs** |
| + 本地 RMW + 协议(实测 57.4)| 57.4 | 57.4 |
| ⇒ AR | 481.1 | **151.0** |
| ⇒ AR 份额(×128/512)| 120.3 µs/token | **37.8** |
| ⇒ TP2 | 545.7 ⇒ 1832 ⇒ 1.53× | **463.2 ⇒ 2159 ⇒ 1.80×** |

**⇒ 装 NVLink 桥 ⇒ TP2/TP1 ≈ 1.80×,超过 1.7× 目标** —— 而且**不依赖**"藏 AR"
(§41 已证跨流藏不了;装桥是**把 AR 本身变快**,绕过那个死结)。

**⚠️ 未验前提**:①桥的实到带宽(56 GB/s 是规格值,实测可能 ~40-50);②**NVLink 下
"写 ‖ 本地 RMW" 是否仍串行**(§41 的机制是"PCIe 写独占 L2 未完成缓冲",NVLink 是另一条通路
⇒ **有可能不再串行**,那样还能再省 44.6µs)。

**⇒ 结论:1.7× 的第一条现实路径 = 给 3090 Ti 对装 NVLink 桥。**

### 43. ★★★ 同机对表:vLLM 的 TP2/TP1 **也只有 1.52×**(2026-10-18)

**用户口径**:"你去看一下 vLLM,它在单卡的时候 prefill 是多少?双卡的时候又是多少?它是能达到两倍、
接近 1.7 的,就在我们的机器上。" + 随后给出双卡实测 **2060 t/s @8192 / TTFT 6000ms+**。

#### 43.1 同一点(8192)对表

| 引擎 | 单卡 @8192 | 双卡 @8192 | **比** | 出处 |
|---|---:|---:|---:|---|
| **vLLM** @3090 Ti | **1360.6 t/s** | **2063.6 t/s** | **1.517×** | 单卡:`docs/bench/qwen38-3090单卡-引擎会战.md`(TTFT@8192 = 6020.9ms 反算;avg 1382.9);双卡:本机今日实测(2067/2060/2064,与用户给的 2060 一致)|
| **owl** @3080 对 | 953.5 | **1535.1**(push)| **1.610×** | `/tmp/owl-ab-{tp1,push2}`(同会话)|
| owl @3080 对 | 953.5 | 1491.0(v1)| 1.564× | 同上 |
| owl @3090 对 | 1191.6 | (推算 ~1830)| ~1.54× | 单卡实测;双卡受占用阻塞 |

#### 43.2 三条定谳

1. **vLLM 的 TP2/TP1 只有 1.52×** —— **"vLLM 能到 1.7×"这个前提不成立**;
   在**同一台机器、同一个点**上,它的 TP 增幅**低于**我们的 1.61×(push,3080 对)。
2. **绝对值的差距(2063.6 / 1535.1 = 1.34×)来自单卡基线**(1360.6 / 953.5 = **1.43×**)——
   **TP 之后差距反而收窄**(1.43× → 1.34×)⇒ **我们的 TP 效率不输**;
   真正要追的是**单卡 prefill 的绝对值**(vLLM 单卡比 owl 单卡快 43%)。
3. **§39.4 的算术(AR 全串行上限 1.64×)与 vLLM 的 1.52× 互相印证** ——
   **TP2/TP1 的上限就是被"AR 落在关键路径上"这条结构约束住的**,不是实现问题;
   vLLM 走 NCCL/custom-AR 也逃不掉(它的 1.52× 甚至比我们低)。

**⇒ 对"1.7×"这个目标的最终判定**:在**当前结构**(每层 2 次 AR、AR 全在关键路径、
链路 PCIe 12.4GB/s)下,**1.7× 对任何引擎都不可达** —— vLLM 实测 1.52×、我们 1.61×,
两者都低于 §39.4 的算术上限 1.64×。要突破,只有两条:**① 换链路(NVLink 桥,§42.3);
② 把 AR 从关键路径上拿掉**(§41 已证"跨流藏"在 PCIe 写形状下不可行)。

#### 43.3 顺带:vLLM 单卡装不下 27B 的重配置档

TP=1 复现受挫(三次 OOM):`qwen38_3090` 档(DFlash2 draft + 视觉塔 + 262k ctx)
在单张 24G 上**装不下** —— 全量 AWQ-INT4 权重 ≈ 21.4 GiB(TP2 日志:每卡 10.71 GiB),
加激活后 `Available KV cache memory` 只剩 **0.87 GiB**。
要单卡复测需同时关 `--vision 0 --spec-tokens 0` 并把 `gpu_mem_util` 提到 0.97
(还会撞 `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` 与 scheduler block size 的整除校验,需 env 改 1568)。
**⇒ 这也侧面说明:单卡跑 27B 的余量本来就在"能不能装下"的边界上,而 TP2 才让 262k 上下文成为可能。**

---

### 44. ★★★★★ 复检 §41:「PCIe 写独占资源」只对重负载成立;轻本地负载**全免费**(2026-10-19)

**立案(§41 的裂缝)**:两条互斥的证据同时躺在卷里 ——

| 出处 | 形状 | 结论 |
|---|---|---|
| §38.2 | 裸 CUDA、**两条 non-blocking 流**、**非图** | W ‖ RMW = 400.5(W=399.6)⇒ **重叠 97%** |
| §41.4/§41.7 | 裸 CUDA、**图内两分支** | `D − S` 与 `R` 严格 1:1 ⇒ **完全串行** |

二者不可能同时为真。§41 由此下了**全局性判决**("PCIe 写独占共享资源,跨流藏 = 死"),
而这个判决**又被 §38 用来否掉 TBO/微批**("微批也没有地基")⇒ **整条路线图押在它上面**,
必须重测。

#### 44.1 新工具:`tools/p2p-overlap-probe.cu`

十条臂(同载荷 5.24MB、同几何、**同计时口径**),并**内置两条存在性判据**(§39 的教训):

| 臂 | 形状 |
|---|---|
| A | 写链单跑(读本端 ⇒ `st.global` 写对端 + `__threadfence_system`)|
| B | 本地 RMW 单跑(`dst += src`,读 2 写 1)|
| C | **非图**双流 A ‖ B,**末尾 join 事件**(计时口径必须含 s2)|
| D | **图内**两分支 A ‖ B(链首 fork、两链发完再 join)|
| D2 | **图内**同质对照 B ‖ B(图到底跑不跑并行分支)|
| E | **单核融合**:一次发射,前一半 CTA 干 A、后一半干 B |
| F | **单核交错**:每个 CTA 先干 A 再干 B |
| G/H/I | `k_stream`(256MB 权重流,948 GB/s)单跑 / 非图 ‖ A / 图内 ‖ A |
| J/K/L | **cuBLAS 真 GEMM**(m512 n5120 k5120 fp16,80 TFLOPS)单跑 / 非图 ‖ A / 图内 ‖ A |
| verify | 落地校验:对端首字 = `0x5a5a5a5a` ∧ 本端首字 = `0x3c3c3c3c` |

补充旋钮:`OVL_WGRID` / `OVL_RGRID`(网格扫描)、`RMW_K`。

#### 44.2 ★★★ 读数(CUDA **ord 0/3**;5.24MB;K=64;**机器全空**)

> ⚠️ **卡序订正(2026-10-19 收尾核对)**:本机 **CUDA ordinal ≠ nvidia-smi 序**
> (`torch`/CUDA 用 `FASTEST_FIRST`):
> **ord 0/1 = 3090 Ti 对**(pci 41:00 / 42:00;`PHB`;BAR1 **32768MiB** 宽 ⇒ owl 判 `Static`);
> **ord 2/3 = 3080 对**(pci 01:00 / C1:00;`NODE`;BAR1 **256MiB** 窄 ⇒ owl 判 `Window`)。
> ⇒ 本节的 `ord 0/3` 实为 **3090Ti + 3080 混插对**(CUDA 直连 P2P 能跑,owl 的分档守卫会**拒**这种组合)。
> **两种拓扑都得出的 12.75 GB/s 与 Gen3 x16 的上限一致,结论不受影响;但"哪一对"必须按上面这条读。**
> (owl 生产 TP2 的合法对 = **ord 0/1** 或 **ord 2/3**。§43 的"3080 对"基准用的是 **2/3**。)


**① 轻负载(本地 RMW 23.8µs = 220 GB/s)**

| 臂 | µs | 读数 |
|---|---:|---|
| A 写链(单)| **411.34** | 12.75 GB/s |
| B 本地 RMW(单)| 23.76 | 220.6 GB/s |
| 串行预测 A+B | 435.11 | — |
| **C 非图双流 A ‖ B** | **410.54** | 串行度 **−0.034 ⇒ 全重叠** |
| **D 图内两分支 A ‖ B** | **409.52** | 串行度 **−0.077 ⇒ 全重叠** |
| D2 图内两分支 B ‖ B | 25.32 | 同质对照 ✓(图确实跑并行分支)|
| **E 单核融合(半写半 RMW)** | **410.42** | 串行度 **−0.039 ⇒ RMW 100% 免费** |
| F 单核交错(每 CTA 写→RMW)| 430.59 | 串行度 **+0.810 ⇒ 同 CTA 顺序做要付钱** |

**② 网格扫描(A ‖ B 串行度;W 链网格 / RMW 链网格)**

| WGRID / RGRID | A(写单)| C(非图双流)| D(图内两分支)| E(单核融合)|
|---|---:|---:|---:|---:|
| 16 / 16 | 416.0 | 417.8 (**+0.08**) | 413.5 | 411.0 |
| 68 / 68 | 465.5 | 456.8 | 452.4 | 410.6 |
| 680 / 68 | 537.5 | 578.3 | 574.8 | 410.5 |
| **16 / 680** | 413.4 | **418.7 (+0.30)** | 414.2 | 411.2 |
| 680 / 680 | 542.9 | 552.3 | 553.0 | 410.6 |
| 272 / 272 | 478.6 | 493.1 | 491.1 | 411.0 |

⇒ **本地 RMW 在"填满设备"(680 块 = 10 块/SM)时照样免费**;
⇒ 顺带一项**写作纪律**:**写腿的网格越大越慢**(16 块 410µs → 680 块 543µs,`−20%`),
即"多开线程"对 P2P 写是**净损失**。

**③ 重负载(真竞争者)**

| 竞争者 | 单跑 | 串行预测 | 非图 ‖ A | 图内 ‖ A | **实际藏住** |
|---|---:|---:|---:|---:|---:|
| `k_stream` 256MB(948 GB/s,4096 块)| 283.0 | 694.3 | **631.4** | **624.5** | **22–25%** |
| cuBLAS GEMM(80.2 TFLOPS)| 334.6 | 746.0 | **679.1** | **678.9** | **24%** |

#### 44.3 定谳(五条)

1. **轻本地负载与 PCIe 写腿可以完全并行** —— 三种机制(非图双流 / 图内分支 / 单核融合)、
   六种几何全部免费。**§41.7 第 2 条的措辞("PCIe 写把另一条流上的本地访存完全挡住")
   在字面上被推翻**;"图会把分支串行化"也再次被否(D2 同质对照 25.3µs)。
2. **重负载不免费** —— DRAM 流与真 GEMM 都只能藏 **22–25%**。
   ⇒ §41 的**方向**对(确实有共享资源),**机理**错:不是"写独占一切",
   而是**重访存/满占用的竞争者与写腿争同一条路**(L2↔DRAM↔根复合体)。
3. **单核融合的形态定死**:必须用**分开的 CTA 组**(E:免费);
   **同一 CTA 先写后做本地事(F:+20µs,0.81 串行)**。将来做 chunked 融合照此写。
4. **存在性判据两次救命**(§39 的纪律是有效的):第一版 L 臂 cuBLAS 发错了流
   ⇒ 图节点数 **64/128**、读数作废(而且"L = max(A,·) 完美重叠"的假象**正是这样来的**);
   网格扫描第一版 verify 读出 `0x00000000`(memset 走 legacy 流、核走 non-blocking 流,
   中间没夹同步)⇒ 已修。**两次都是"先信读数、后验路径"会翻车。**
5. **⚠️ 测量学重大警告**:另一进程把 **GPU1** 拉满时,**3080 对的 P2P 写从 12.75 掉到
   6.2 GB/s(2×)**,而 GPU 显存占用只是 20GB(看起来"没动我们的卡")。
   ⇒ **本卷所有 P2P 读数必须标注"机器是否空闲"**;§41.2 那个 6.4 GB/s 的 `S` 臂
   (与本次受污染的 A 臂**同一数值**)很可能同因,**另案复核**。

#### 44.4 对 1.7× 的判决(算账)

```
AR(生产 push)= 411.3(纯写地板,实测)           ← 12.75 GB/s = Gen3 x16 的 81%
             + ~31  (fence / credit / rendezvous;§38.1 的 444 − 411)
             + 44.6 (本地 RMW)
             + 5.3  (两核串联)
             = 494
```

- **可藏的上限** = RMW 那条 = 44.6µs/次 = **11.2µs/token = TP2 的 1.7%**
  ⇒ 即便把 RMW 全藏掉也只到 **1.64×**。
- **AR 全藏(122.9µs/token)这条路已被 44.2③ 堵死**:真 GEMM 只能吸收 24%
  ⇒ 微批 / TBO / GEMM-epilogue 融合**都不值得做**(TBO 自身开销 > 24% × AR)。
- ⇒ **1.7× 只剩两条路:①换链路(§45 的 Gen4 / NVLink);②减字节(8-bit 交换,数值风险)。**

### 45. ★★★★★ 链路代次定谳:GPU 跑在 **PCIe Gen3 x16** 上(主板数据表写的是 Gen4)

**用户口径贯穿全会**:"1.7×"。而 AR 的地板 = 链路。**所以必须先把链路问清楚。**

#### 45.1 读数(全部本地、可复现)

| 证据 | 值 | 判 |
|---|---|---|
| `nvidia-smi --query-gpu=pcie.link.gen.max` | **3**(4 张卡全)| 协商上限 = Gen3 |
| root port `max_link_speed`(4 个 GPU 槽)| **8.0 GT/s = Gen3 x16** | 端口**能力**就是 Gen3 |
| GPU 自身 `max_link_speed` | **16.0 GT/s = Gen4** | 卡是 Gen4 卡 |
| **D2H / H2D(torch,256MB,GPU0)** | **13.19 / 13.07 GB/s** | **Gen3 x16 特征值**(Gen4 应 ~26)|
| H2D ‖ D2H 聚合 | 22.20 GB/s | 全双工,单侧 ~11 |
| P2P 写(单向,本卷) | 12.75 GB/s = **81% × 15.75** | 与 Gen3 x16 上限吻合 |

**⇒ 三重独立证据闭死:GPU↔GPU 与 GPU↔host 两条路都是 Gen3 x16。**

#### 45.2 与硬件文档的冲突(待人工裁决)

| 项 | 事实 |
|---|---|
| 主板 | **TYAN S8030GM4NE-2T**(BIOS V2.03,2024-01-09)|
| 官方数据表(MOA,Jul 2021 Rev.2)| **"(5) PCI-E Gen4 x16 slots"**、框图 5 个槽全标 PCIE 4.0 x16 |
| CPU | **EPYC 74F3 = Milan**(`lspci`:*Milan Data Fabric*),**Gen4 能力** |
| 反证 | **同一张表**又写 M.2 = "by PCI-E Gen3 interface";实测 M.2 root port = **8.0 GT/s x4** ⇒ **表文自相矛盾** |

**⇒ 两种可能,必须人工定:(a) BIOS 里有 PCIe 速度档被压在 Gen3;
(b) 板子实际按 Gen3 布线,数据表的 Gen4 行是沿用/笔误。**

#### 45.3 待办(需人操作;iKVM 可用)

1. 进 BIOS 找 **"PCIe Speed" / "PCIe Gen Speed" / "Link Speed"** 一类选项(逐槽),
   确认是否 = Gen3/Auto;若是,**改 Gen4 并复验** `lscpu/lspci` 的 LnkCap 是否变 16 GT/s。
2. 顺带确认 **4 张卡有没有走 Gen3 转接卡/延长线**(转接卡不改 LnkCap,但会锁训练速度)。
3. 复验口径:D2H 应回到 ~26 GB/s;`p2p-overlap-probe` 的 `A` 臂应 ~205µs。

#### 45.4 收益(算账;若 Gen4 成立)

| | 现在(Gen3,12.75 GB/s)| **Gen4 后(~25.5 GB/s)** |
|---|---:|---:|
| AR 写腿 | 444 | **~222** |
| + fence/credit/rendezvous | 31 | 31 |
| + 本地 RMW | 44.6 | 44.6 |
| + 串联 | 5.3 | 5.3 |
| = AR | 494 | **~303** |
| AR / token(×128/512)| 122.9 | **75.7** |
| ⇒ TP2(µs/token)| 651.1 ⇒ 1536 t/s ⇒ **1.61×** | **603.9 ⇒ 1656 t/s ⇒ 1.71×** |

**⇒ 零代码改动,直接越过 1.7×**(且此时 §44.4 的"藏 RMW"变成**还要再赚 1.9%** 的余量)。

---

### 46. 「小包优化」定谳 + 文献调研(2026-10-19;用户指定两个角度)

**用户口径**:"从两个角度再来尝试优化:一个是小包优化,另一个是搜搜论文/项目看别人怎么写的。"

#### 46.1 写腿调优台(`tools/p2p-wr-tune.cu`;CUDA ord 0/3(见 §44.2 的卡序订正);5.24MB;机器全空)

| 臂 | µs | GB/s | 判 |
|---|---:|---:|---|
| `st.global` + fence(**基准**,grid 16×512 = 8192 线程)| 410.70 | 12.77 | 与 §44 的 A 臂一致 |
| **`st.global` − fence** | **399.58** | **13.12** | 栅栏 = **+11.1µs(2.7%)** |
| `__stcs`(streaming)+ fence | 410.41 | 12.77 | **零影响** |
| `__stwt`(write-through)+ fence | 410.28 | 12.78 | **零影响** |
| `__stcg` + fence | 410.13 | 12.78 | **零影响** |
| **每线程连续 64B**(4×uint4)+ fence | **737.33** | **7.11** | **灾难:−44%** |
| grid **4**(2048 线程)| **401.86** | **13.05** | **最优** |
| grid 8(4096 线程)| 402.54 | 13.02 | 次优 |
| grid 16(8192 = **生产几何**)| 410.70 | 12.77 | 生产用的就是这个 |
| grid 32(16384)| 434.72 | 12.06 | |
| grid 64(32768)| 452.32 | 11.59 | |

**四条定谳**:

1. **存储策略完全无用**(四种 cache hint 全在 12.77±0.01)⇒ 不是 L2 策略受限。
2. **在飞线程数越少越快**:2048 → 13.05 GB/s,8192(生产)→ 12.77,32768 → 11.59。
   **⇒ 生产 `MAX_THREADS = 8192` 不是最优,2048 可省 ~9µs/次(0.3%)。**
   (上一位把"32 块 vs 64 块"扫成 32 块最好 —— 那是在 BLOCK=256 下,**同为 8192 vs 16384 线程**,
   与本表同向;但**没人往 2048 那头扫过**。)
3. **每线程连续 64B 是陷阱**(7.11 GB/s):把"每个线程的 4 次写"串成依赖链 ⇒ 在飞事务数塌掉。
   **跨步(stride)形态必须保留。**
4. **栅栏是真钱**:`__threadfence_system()` = 11µs/次 = 2.75µs/token。
   ⚠️ 但它**不能删**(发布律 I4:数据系统可见 → 再写 flag),只能想办法摊薄。

#### 46.2 ★★★★★ 小包定谳:**19% 的缺口是 TLP 包头,不是软件**

| 路径 | GB/s | 占 Gen3 x16 理论(15.75)| 占 128B-MPS 理论(13.26)|
|---|---:|---:|---:|
| GPU↔host D2H / H2D | 13.19 / 13.07 | 83.7% / 83.0% | **99% / 98.5%** |
| CE P2P 写(§36.3)| 12.74 | 80.9% | 96% |
| SM P2P 写(本台)| 12.77–13.05 | 81–83% | 96–98% |

**`MPS=128B` 的理论效率** = 128/(128+24) = **84.2%**(= 13.26 GB/s);`MPS=256B` = **91.4%**(14.40)。

**⇒ 三条独立路径(host / CE / SM)全部落在 13 附近 = 128B-MPS 天花板的 96–99%**
**⇒ 那 19% 不是我们的代码,是每个 TLP 的 24B 包头。**

**⇒ 唯一杠杆 = MPS 128 → 256**(需 root 写 Device Control 的 MPS 域,且路径上所有端口一致):
写腿 444 → 409(−35µs/次 = 8.75µs/token)⇒ **比值 +1.6%**。
**待办(需人):`sudo lspci -vvv -s 01:00.0`(和 root port `00:01.1`)看 `DevCap/DevCtl` 的 MaxPayload ——
`DevCap` 若为 256B 而 `DevCtl` 只有 128B,就说明 RTL 支持、只是被 BIOS 谈判成 128。**

#### 46.3 顺手否掉的两个假设

| 假设 | 判据 | 结果 |
|---|---|---|
| "链路降功耗 ⇒ 每次 AR 付唤醒税"(用来解释生产 444 vs 探针 411)| 单次写之间夹 host sleep:背靠背 596µs / 间隙 200µs→10ms = **403.8 / 405.0 / 405.4µs** | **否**:间隙反而更快(收敛到 404),且 200µs 就饱和 |
| "写本地源会拖慢写对端" | §41.6 已测(读本地+写对端 = 13.11)| **否**(复验一致)|

#### 46.4 文献调研(用户指定的第二个角度)

**① sglang「PCIe oneshot allreduce」**(`lukealonso/sglang@d39236a`;`voipmonitor/rtx6kpro` 有提炼文档)
—— **与我们的 1-stage 对称槽设计同构**(全体同时 P2P 写 + system-scope barrier + 本地规约),
还多两件:**双缓冲**与**融合 RMSNorm**;**启动期自动标定 crossover**。

| | 读数 |
|---|---|
| vs NCCL 延迟(1KB→512KB)| **1.4–6× 更快**;**1MB 起 NCCL 反超**(crossover 512KB)|
| e2e decode(4× RTX PRO 6000,TP=4)| c=1 **+11.3%**,c=2 +8.1%,c=64 +7.3% |
| 8 卡跨 socket | **变慢**(system-scope barrier 要过 Infinity Fabric)|

**对我们的判决**:它的蜜月区是 **≤512KB 的小消息**;我们的 prefill AR 是 **5.24MB**,
**远在它自己的 crossover 之外** ⇒ **结构上抄不了**(这一点我们用 §36 的 NCCL 对表也验证过:
5.24MB 处 NCCL 463 vs 我们 494,差距本来就只有 6%)。
**但它的 decode 读数(+11%)值得记**:decode 的 AR 载荷是 10KB 级,正是它的蜜月区 ——
**若将来要挖 decode,这是现成的地图。**

⚠️ **它文档里还有一条我们命中的坑**:NODE 拓扑(走 CPU root port、无 PCIe switch)下,
驱动**默认把 CUDA 核发起的 P2P 访问走 SysMem staging**(比 NCCL 慢 ~15×),
需要 `NVreg_RegistryDwords="ForceP2P=0x11;RMForceP2PType=1;...;GrdmaPciTopoCheckOverride=1"`。
**我们的 3080 对正是 NODE 拓扑**(`nvidia-smi topo -m`),而 `/proc/driver/nvidia/params` 里
`RegistryDwords = ""`(**没设**)。**但实测写腿 12.77 GB/s 与 3090 对(PHB)持平**
⇒ **aikitoria BAR1 窗口补丁已经把这条顶掉了**(它就是我们 kernel patch 里的窗口机制),
**不是我们受影响的项**,记档以免后人误开这条开关。

**② TensorRT-LLM `lowprecision-allreduce`**(官方文档 + `tensorrt_llm/_torch/distributed/ops.py`)
—— **就是 §45 里那条"8-bit 交换"路线的现成先例,而且是 NVIDIA 官方出货件**:

- **算法**:传输前把 fp16/bf16 量化成 **fp8 e4m3**,传输后反量化再规约 ⇒ **通信量减半**;
- **量化粒度**:**per-warp** —— 每个 warp 里 31 个线程各量化 16B(496 个元素),第 32 个线程传一个标量 scale;
- **适用面**:"Improves performance for **large message sizes** in PCIe-based topologies"
  —— **正是我们这种 5.24MB 大消息**;NVLink 或小消息下**自动回退**;
- **代价**:官方 caveat 原文是 "**may affect model accuracy … evaluate the precision impact for your specific workload**";
- 开关:`AllReduceStrategy.LOWPRECISION`(或 `allreduce_strategy="LOWPRECISION"`)。

**⇒ 对我们的意义**:① 我们的目的是"1.7×",而 fp8 交换是**软件侧唯一能到那的路**(§44.4);
② **别自己发明**——照它的 per-warp(496 元素/scale)粒度抄,风险面最小、有官方出货背书;
③ 我们的验收门是 `tp_g8_gate`(现在 **0 项不同**),fp8 会**打破位精确**,必须**先立新判据**
(建议:max|Δ| 逐步序列 + 下游 logits 相对误差上界),再动手。

**③ llama.cpp 的 internal allreduce**(`GGML_CUDA_ALLREDUCE=internal`;双 2080Ti 22G 跑 Qwen3.8-27B 的实测博客)
—— 社区在 PCIe 上的选择也是**自研 oneshot 系**而不是 NCCL;**顺带一条 TP 教训**:
"拆卡本身买不到任何东西"(单卡 37.1 → 双卡 `-sm tensor` 37.5),**+60% 全部来自 MTP 草稿**。
⇒ 与我们 §42 的账同构:**TP 的收益来自"能上更大批/更长上下文/投机",不是来自"多一张卡"**。

#### 46.5 ★ 更新后的总账(软件侧已见底)

| 杠杆 | 省(µs/AR 次)| 折算到 TP2 | 备注 |
|---|---:|---:|---|
| MPS 128→256 | 35 | **+1.6%** | **需 root**;三条路径都卡在 128B 天花板 |
| 写腿线程数 8192→2048 | 9 | +0.4% | 一行(改 `MAX_THREADS`)|
| 藏掉本地 RMW(§44 单核融合 E 形)| ≤44.6 | +1.7% | 需 chunked 发布 |
| 栅栏摊薄 | ≤11 | +0.4% | 受正确性约束 |
| **软件合计** | **~100** | **≈ +3.5% ⇒ 1.61 × 1.035 ≈ 1.67×** | 仍不足以到 1.7 |

**⇒ 结论未变,但地基更硬了:1.7× 只有三条现实路 ——
①(免费/待人工)**MPS 256 | PCIe Gen4**(§45 的 BIOS);②(软件)**fp8 交换**(§46.4②,有官方先例);
③(硬件)**NVLink 桥**(§42.3)。**

---

### 47. ★★★★★ fp8 AR 交换 —— **第一步:判据先行(数值)已过**(2026-10-19;用户拍板)

**用户口径**:"fp8 AR 交换(唯一过 1.7× 的软件路)。"纪律 ⇒ **判据先行,先只改数值、不动协议。**

#### 47.1 ★ 设计:对称性的那个坑(先说清,再落码)

AR 的语义是 `out = a + b`。若**只**把传来的那一份量化:

| | rank0 算出 | rank1 算出 | 一致? |
|---|---|---|---|
| 只量化传输面 | `a + D(q(b))` | `b + D(q(a))` | **✗ 不一致 ⇒ L1 破** |

**⇒ 必须"两边各自把自己那一份也过一遍"**:`out = D(q(a)) + D(q(b))`(两边算的是**同一对值的同一个加法**,IEEE 加法可交换 ⇒ 逐位一致)。
**⇒ 这与真实 fp8 传输数值等价**(送出前量化一次、收端反量化,共用同一组尺度)——
所以这一步既是判据仪器,又是将来那份实现的一半。

#### 47.2 落地(最小切口:**只改一个核、零协议改动、零配置面**)

| 件 | 落点 |
|---|---|
| e4m3 往返 + 组尺度 | `cu/comm/ar_reduce.cu` 新增 `owl_ar_e4m3_round` / `owl_ar_qdq_scale` |
| fp8 快路 | `owl_ar_reduce_vec_fp8<T,V>`(与 `owl_ar_reduce_vec` **同几何、同签名**)|
| 派发 | `OWL_AR_REDUCE_FAST(TY, D, S, N)` —— ⚠️ 类型必须**当宏参传**(核体是宏展开,写进宏定义里外层替换不到,踩过一次)|
| 开关 | `#define OWL_AR_QDQ 1`(**编译期**,本轮只为拿数值)|

**粒度**:**每组 8×PER_VEC = 64 个元素一个尺度**(bf16),比 TRT-LLM 的 per-warp 496 **更细**;
scale 开销 2B/64 元素 = 3.1% 载荷。e4m3(±448 饱和)。

#### 47.3 读数(`tp_g8_gate`,0.8B,A/B = **CUDA ord 2/3 = 3080 对**;v1 绑定)

| 量 | 基线(bf16)| **fp8 e4m3** | 判 |
|---|---|---|---|
| **L1:rank0 vs rank1 末步 logits** | 0 项不同 / max\|Δ\| = 0.000000 | **0 项不同 / 0.000000** | **对称性完好 ✓✓** |
| L3 全程 max\|Δ\|(vs 单卡)| 21.928 | **21.768** | 同一包络 |
| L3 均值 | 17.864 | 17.799 | 同一包络 |
| 首分歧步 | 0:`109757` vs `2005` | 0:**同一对** `109757` vs `2005` | 同一包络 |
| 互含判定 N=5 | true / true | **true / true** | 判据绿 |
| 全序列 token 不同 | 16/16 | 16/16 | 参考值 |

**⇒ 定谳:fp8 e4m3(64 元素/尺度、对称形式)在本门的**全部判据上与基线不可区分**,
且偏离开销落在既有的"TP1 vs TP2 固有分歧"(f16 部分和舍入经 24 层 GDN 混沌放大)包络之内。**

⚠️ **诚实标注两条**:
1. 本门的 `max|Δ| vs 单卡` 是**饱和指标**(两条都分歧的混沌轨迹都落在 ~22 的量级)
   ⇒ 它**证明不了** fp8 无害,只证明"没把包络撑破"。**真正有分辨力的量是 AR 边界上的逐元素误差** ——
   待补(见 47.5 待办②)。
2. 本门跑的是 **0.8B/24 层**;27B/64 层的累积不同 ⇒ **上产前必须在 27B 档复验**。

#### 47.4 顺手发现的既有门问题(与 fp8 无关,另案)

`OWL_TP_PUSH=1` 时门断言 `tp.ar.v1 == 0`("开了 push 却仍在走 v1 绑定")**失败**:
本模型实测 **push=192 / v1=2016**。**这不是回归** —— arena 槽只对部分节点生效
(`eval.rs::symmetric_out_slot` 按 `out.id` 反查),其余照旧走 v1 两段。
**⇒ 门的期望值比现实严**(§39 的同族教训:门的判据要跟"实际覆盖了多少"对齐,而不是"开了就该全走")。

#### 47.5 收益账(第二步还没做 —— 现在只赢了数值,没赢时间)

**第二步 = 真把字节砍半**(fp8 载荷 + fp16 尺度写进对端 data 块;reduce 反量化):

```
载荷 = n(1B/元素) + n/64×2B(尺度) = 1.031n   vs  2n(bf16)  ⇒ 51.6%
写腿 444 → ~229µs(−215)  ⇒ AR 494 → ~279µs
AR/token 122.9 → 69.4     ⇒ TP2 651.1 → 597.6µs ⇒ 1673 t/s ⇒ **1.73×**
```
⚠️ 前提:量化的本地计算能藏进 PCIe 写(§44 已证"轻本地负载在写腿下免费");**未验**。

**第二步的落地要点(照 47.1 的对称律)**:
- 组必须改成**按元素区间连续**`[g·G, (g+1)·G)`(第一步是按线程映射分组的 —— 数值同类、但
  收端换映射后**找不到尺度**,必须区间化);
- stage 写 `[fp8 载荷 n B][尺度 n/64×fp16]`;reduce 读尺度 → 反量化对端那份 +
  **原地量化自己那份**(对称律);
- `slot_bytes` 不动(变小,装得下);`bytes` 仍是 bf16 语义(核内换算)。

**待办**:
① 第二步(真传输)落地 + 隔离基准(3 秒)再上生产(std 6 分钟;纪律 ⑳);
② **补"AR 边界逐元素误差"判据**(饱和指标之外的那个真数):在 `tp_transport_bits_gate` 里加一臂,
   喂真实形状的两份部分和,量 `D(q(a))+D(q(b))` vs `a+b` 的相对 L2 / 最大相对误差;
③ 27B 档复验 (0.8B 的混沌性使它无法当精度判别器)。

---

### 48. ★★★★★ 读写腿不对称:**PCIe 读能藏进计算(97%),写不能(24%)** —— §37 的 push 把藏的路堵死了(2026-10-19)

**立案(用户提问驱动)**:"你用 fp8 做同步,那中间数据是 **FP16** 怎么办?同步得过去吗?**FP16 那条路也保留**。
你再想想**其他办法**,读读文献。"

#### 48.1 先答 FP16(事实核对 + 账)

**事实**:`ModelSpec.dtype = Dtype::F16`(`formats/qwen35.rs::qwen3_5_0_8b` 与 `qwen3_8_27b` **两档都是 F16**;
注释里的"激活/池全 bf16"指别的路径)⇒ **我们生产 AR 的载荷本体是 f16(10 位尾数)**。

| 格式 | 尾数位 | 相对精度 |
|---|---:|---:|
| **f16(载荷本体)** | 10 | **0.049%** |
| bf16 | 7 | 0.39% |
| **e4m3(fp8)** | 3 | **3.125%** |
| int8 + 每块 scale | 均匀 8 位 | 块内 ≈ **R/254**(R = 块 max/典型)|

**⇒ e4m3 对 f16 载荷是"尾数 10 → 3"的 64× 粗化**,而且会成为**全引擎最粗的一环**
(INT4 权重配 per-group scale 比它细;fp8 KV 只影响注意力分数)。
**⇒ §47 的门判据(`max|Δ| vs 单卡`)**是**饱和指标**,看不见这件事 ⇒ **它不能当"fp8 无害"的证据。**
**⇒ 定性正确、定量未证。f16 全精度路径必须保留(它本来就是独立 dtype 实例;第二步要挂 flavor 配置,不做全局开关)。**

#### 48.2 缺的那一格:测 [PCIe **读** ‖ 真 GEMM]

`tools/p2p-overlap-probe.cu` 新增 `k_rp`(读对端 slot + 本地累加 = **v1 的 reduce 数据面**)与 R/M/N 三臂。

**读数(同一支二进制、同一几何;`RMW_K=64`;5.24MB;GEMM = m512 n5120 k5120 fp16)**

| 臂 | 机器干净 | 机器被污染 |
|---|---:|---:|
| A 写链(单)| 416.9 µs | 846.0 µs |
| **R 读链(单)** | **413.5 µs** | 540.4 µs |
| J' GEMM 链(单)| 337.9 µs(79.5 TFLOPS)| 566.3 µs(47.4)|
| **K/L 写 ‖ GEMM** | **+63.3% / +62.8%** | +72.8% / +65.9% |
| **M/N 读 ‖ GEMM** | **+2.4% / +2.1%** | +49.1% / +55.7% |

#### 48.3 定谳(四条)

1. **写腿与读腿的**裸带宽同速**(416.9 vs 413.5µs,±1%)—— 与 §36.3 的"写快 13–15%"不符
   (那是在**双向**档下量的;本条是单边)。
2. **★ 读腿能藏进真 GEMM:97–98% 重叠**(串行预测 752µs → 实测 424µs)。
3. **★ 写腿不能:只有 24–37% 重叠**(实测 681µs)。
   ⇒ 与 §41.3([读 ‖ 本地] = 92%)、【§41]([写 ‖ 本地] = 0%)、§44(H/I/K/L)全部一致:
   **规则是"PCIe 读(non-posted,走完成通道)不挡本地工作;PCIe 写(posted,占 L2 写回/合并缓冲)挡"。**
4. **⚠️ 机器污染会把这条优势从 97% 压到 ~50%**(第 2 次读数)⇒ 结论只在静机成立;
   且再一次印证 §44.3⑤"读数必须标注机器状态"。

#### 48.4 ★★★★ 文献交叉:业内那套 **全是读腿**,而 §37 的 push 把我们挪走了

| 实现 | 数据腿 | 出处 |
|---|---|---|
| **vLLM** `cross_device_reduce_1stage` | **读**(`packed_reduce((const P**)&dp.ptrs[0], idx)`)—— §36.4 实测也确认:"它走的不是 SM 读那条腿"的反面 | `csrc/custom_all_reduce.cuh`(§35 已引)|
| **NCCL** world=2 1-stage | **读**(同上形状)| 同上 |
| **sglang PCIe oneshot** | **写**("all GPUs write their data to all peers simultaneously")| §46.4① |
| **我们 §37 的 push** | **写**(§36.7:数据块搬到对端)| 本卷 §37 |

**⇒ 我们做 §37 时,AR 是**纯串行**的 ⇒ 写腿"裸带宽快"就赢了(+4.0%)。
但那一步**正好把重叠的路堵死** —— §44 已经量出写腿只能藏 24%。**

#### 48.5 判决:**1.7× 有一条不吃精度的路 —— 读腿 AR + 微批(TBO)**

- **机制**:TBO 把 512-token 块拆成 2×256。微批 A 的 AR(读腿)在飞时,GPU 跑**微批 B 的 GEMM**。
  [读 ‖ GEMM] = **97%** ⇒ **AR 从关键路径上消失。**
- **算账**:TP2 = 651.1µs/token,其中 AR = 122.9(19%)。藏掉 ~90% ⇒ **TP2 ≈ 540µs ⇒ 1852 t/s ⇒ 1.92×**
  (留 30% 余量也还有 **1.75×**)。**零精度代价**,而且 fp8 变成"可选"而不是"必需"。
- **TBO 自身成本(要实测)**:GEMM 的 m 从 512 → 256(效率略降)、权重每层读两遍(32 → 64 GB/s,DRAM 余量足够)、
  激活显存 ×2(512 token 档很小)、流水填充/排空 ≈ 一个 layer 的时间。

**⇒ 待办(按性价比)**:
1. **把 AR 切回读腿(v1 形态)**,先量"纯串行下 push vs v1"的差(§37 的 +4% 会不会被读腿的裸带宽补回来);
2. **TBO 立项**:先做一个**两微批的调度骨架 + 门**(数值必须与单微批一致:分块不能改变归约序),
   再上 std 量 AR 是否真的被藏掉 —— **判据 = `AR 段消失`(用 §40 的 metrics 看每一段的墙钟)**;
3. fp8(§47)降级为**备用弹药**:只在"藏不掉"的档位(如 decode 小消息、或 TBO 装不下的形状)才考虑,
   且**必须补 48.1 说的那个非饱和判据**(AR 边界逐元素误差)。

---

### 49. 形态 A/B 正面对表 + TBO 首测(2026-10-19;用户拍板"两条形态都保留,先做实验")

#### 49.1 ★★ 两条形态同会话对表(`ar_channel_bench`,ord 2/3 = 3080 对;f16;K=200)

| 量 | **形态 A = v1(读腿)** | **形态 B = push(写腿)** | 差 |
|---|---:|---:|---:|
| **AR @5.24MB** | **601.73 µs**(8.31 GB/s)| **504.30 µs**(9.92 GB/s)| **写腿快 16.2%** |
| 纯读腿 `R` | 496.5 µs | 516.0 µs | — |
| 纯写腿 `A2` | 423.3 µs(11.81 GB/s)| 427.1 µs | — |
| AR @10240B(decode 一行)| 35.45 µs | 34.46 µs | 同(host 地板)|

**⇒ 纯串行下写腿 AR 快 16.2%(省 97µs/次 = 24µs/token = TP2 的 3.7%)**
—— 与 §37 生产的 +4.0% 自洽(0.19 × 16% ≈ 3%)。

**★ 顺带捞到一个没量过的洞**:

| 腿 | 我们生产(经 VMM 别名)| 裸硬件(探针,`cudaMalloc` 对端)| 差 |
|---|---:|---:|---:|
| **写腿** | 423.3 µs / 11.81 GB/s | 416.9 / 12.58 | **仅 1.5%** |
| **读腿** | 496.5 µs / **10.6 GB/s** | 413.5 / **12.68** | **17~20%** |

⇒ **VMM 别名上的 SM `ld.global` 有 17-20% 的税**(`tools/vmm-peer-probe.cu` 头部已记了这条:
同核同卡换映射路径,9.93 → 12.93 GB/s),而**写腿不受影响**。
⚠️ **但 §32 已经试过"换通道"(CE 拉取)并回退** —— 结构性原因(§32.5):
CE 版把"核内重叠的对端读 + 本地加"变成 `stage → rendezvous → CE 拷 → add` 四段串行;
⇒ **读腿的税要修,就得在 SM 读这一条路上修(VMM 映射参数/页粒度),不能绕道 CE。**

#### 49.2 TBO 首测(`tools/p2p-overlap-probe.cu` 新增 S/T/U 臂;链 = 8 层 × [GEMM m256 + AR 2.62MB])

| 臂 | 读数 | 判 |
|---|---:|---|
| **U 微批 AR 读**(交替 4 块对端源,10.5MB>L2)| 221–252 µs | 11.9 GB/s ⇒ **真走 PCIe**(可信)|
| **S 单链**(GEMM+AR 都在 s1)| 3.519 ms | 440µs/层 ≈ 221+219 ✓ 串行成立 |
| **T 双链并行**(2 流,SM 读腿)| **5.776 ms** | 串行预测 7.038 ⇒ **只提升 17.9% ⇒ 吞吐比 1.22×**(理想 1.93×)|
| **T2 双链并行**(2 流,**CE 拉取**)| **6.790 ms** | 提升 **1.5%** ⇒ **1.02×,完全不重叠** |

**⇒ 朴素"双流 TBO"只藏掉约 36% 的 AR(1.22× / 理想 1.93×);换 CE 更差(1.02×)。**

**⚠️ 本轮自己踩的两个探针雷(记档,§39 同族)**:
1. **`cublasSetStream` 只在双链臂里设** ⇒ "单链"偷偷成了双流(GEMM 走默认流、AR 走 s1),
   读数 245µs/层 —— 那其实是**AR 被藏掉 84% 的假象**。修法:建 handle 后**立刻**设流。
2. **读源 `cudaMalloc` 时忘了 `cudaSetDevice(ob)`** ⇒ 读的是**本端** L2,读数 18.9µs(139 GB/s)。
   修法:源必须显式分配到对端设备。**⇒ "本端/对端"在探针里必须是显式常量,不能靠上下文。**

**诊断(为什么只有 36%)**:两条链的 GEMM 都是**满占用核**,彼此抢 SM ⇒ 两个 GEMM 实际近串行,
⇒ 一个链的 AR 在飞时,另一条链**没有 GEMM 在跑**。§48 的 97%([读 ‖ GEMM])是**单链对单 GEMM** 的形状,
**多链调度下拿不到**。CE 版 1.02× 说明 CE 虽然零 SM 占用,但**拷贝引擎本身成了串行点**。

**候选修法(按成本)**:
1. **3–4 个微批**(而不是 2):保证任一时刻都有一条链的 GEMM 可跑;
2. **显式交错调度**(单流/单图里排 `G_A, G_B, AR_A, AR_B, G_A′, G_B′ …`)—— 但我们的 AR 核**是自旋核**,
   排在同一流里会**堵住后续节点** ⇒ 要么非阻塞化,要么放支流;
3. **部分占用核**:把 GEMM 的 grid 留出余量(SM 槽留给 AR 核)。

**⇒ 对目标的账(用实测的 36%)**:TP2 = 321ms(128×AR 884µs/层 + 计算)**− 36%×AR ⇒ ~300ms ⇒ 1706 t/s ⇒ 1.77×**;
若调度修到 90% ⇒ **1.93×**。**⇒ TBO 仍然是过 1.7× 的最优路,但"两条流一边一条链"不够,必须做调度。**

#### 49.3 ★★★ TBO 微批扫描(N=2/3/4):**吞吐比卡死 1.21–1.28×,微批数无用** ⇒ 资源死结

**立案(v1 的假设)**:§49.2 两条流各一条链只藏 36% ⇒ 猜"两条链的满占用 GEMM 互抢 SM,
一条链的 AR 在飞时另一条没有 GEMM 可跑"。**判据:N 变大时吞吐比是否往 N 走。**

(`tools/p2p-overlap-probe.cu` 的 `V` 臂 = 微批扫描;链 = `NL=6` 层 × [GEMM m=512/N + AR 5.24MB/N];
读源在对端 20MB 里轮流 ⇒ 断 L2;`cublasSetStream` 建 handle 后立刻设)

| N | m | 单链 | N 链并行 | **吞吐比** | 理想 |
|---:|---:|---:|---:|---:|---:|
| 2 | 256 | 2.623 ms | 4.349 ms | **1.21×** | 2 |
| 3 | 170 | 2.034 ms | 4.784 ms | **1.28×** | 3 |
| 4 | 128 | 1.214 ms | 4.025 ms | **1.21×** | 4 |

**⇒ 吞吐比与 N 无关,且 N 链并行的总时间是常数(~4.0–4.8ms)。**

#### 49.4 定谳:把总时间拆成资源时间,**它就是两者之和**

```
总 PCIe = NL × 5.24MB / 12.7GB/s            = 2.47 ms
总 GEMM = NL × 2·512·5120·5120 / 70TFLOPS   = 2.30 ms
理想(max)  = 2.47 ms       实测 = 4.0–4.8 ms ≈ 2.47 + 2.30
```

**⇒ 计算(GEMM)与 PCIe 读**在这条链的形状下**没有任何重叠**(串行度 ≈ 1.000)。
**⇒ §48 的 [读 ‖ 真 GEMM] = 97% 与这里不矛盾,但**不可移植** —— 差别在形状:**

| §48 的 M 臂(重叠 97%)| 本节 V 臂(重叠 0%)|
|---|---|
| s1:**先发完** 64 个读核(16 块小核)| 两流**交替**发 `G → AR → G → AR` |
| s2:**再发完** 64 个 GEMM(满占用)| 每条链内部自带 G↔AR 依赖 |
| ⇒ t=0 时 GPU 上同时有"小核 + 大核"⇒ 之后一路流水 | ⇒ t=0 时只有链 A 的 G(满占用),链 B 在等 |

**待验的两个机制**(下一个实验,各约 15 行):
1. **核级**:单独一对 `[A.G ‖ B.AR]`(不套链)是否能重叠?能 ⇒ 是**链间排队**;不能 ⇒ 是**核级资源**。
2. **发序**:把 AR 链**先** enqueue、GEMM 后 enqueue(复刻 §48 成功形状)是否恢复重叠?
   能 ⇒ 说明**发序/队列深度**是可控变量(生产里可以故意让 AR 早发)。

**⇒ 对目标的账(修正)**:TBO 的收益**上限就是 AR 份额(19%)**,而当前朴素实现拿到的是 **0%**。
**⇒ 在"计算与 PCIe 读能并存"这件事被独立证实之前,1.7× 不能押在 TBO 上**;
**已被证实可用的仍只有 §45(Gen4)、§42.3(NVLink)、§46.2(MPS 256)、§47(fp8,有精度代价)** 四条。

#### 49.5 ★★★★ 机制判据(核级 / 发序 / 错相):**全部否掉 ⇒ 更正 §48 的适用范围**

`tools/p2p-overlap-probe.cu` 新增 `X`(核级)与 `Y`(错相)臂。链条形状:全批 512 token、GEMM m=512/n5120/k5120、
AR 读 5.24MB(源在对端 20MB 里轮流 ⇒ 断 L2)。

| 臂 | µs | 判 |
|---|---:|---|
| X1 GEMM(单)| 339.34 | 分母 |
| X2 AR 读(单)| 414.72 | 分母 |
| **X3 [AR 先发 ‖ GEMM]** | **468.00** | max=415、和=754 ⇒ **重叠**(超出 max 13%)|
| **X4 [GEMM 先发 ‖ AR]** | **388.71** | ⇒ **重叠**(甚至快过单开)|
| **X5 [GEMM ‖ GEMM ‖ AR]** | **620.54** | 2 GEMM 单跑 679 ⇒ **AR 挂在两个 GEMM 上几乎免费** |

| 臂 | 吞吐比 | 判 |
|---|---:|---|
| V 微批扫描 N=2 / 3 / 4 | 1.21× / 1.29× / 1.25×(理想 2/3/4)| 与 N 无关 |
| **Y 错相启动(N=2,链 0 等链 1 首层 GEMM 完成)** | **1.25×**(同相 1.21×)| **相位锁定被否** |

**⇒ 定谳(三条)**:

1. **核级完全能重叠** —— `[G ‖ AR]` 87–100%,`[G ‖ G ‖ AR]` 里 AR 免费。**不是 SM/PCIe 资源死结。**
2. **错相没用** —— 两链交替的**依赖序**(`AR_i` 必须等 `G_i`、`G_{i+1}` 必须等 `AR_i`)才是关键:
   **AR 永远排在自己那条链的 GEMM 之后,硬件不给"后到的核"抢占**;
   而 §48 的 M 臂是**先发完 64 个读核、再发完 64 个 GEMM**(长同质独立链 + 长 GEMM 链)⇒ 一路流水。
3. **★★ 更正 §48 的适用范围**:「PCIe 读能藏进计算」是**形状特有**的,
   **它不能用来支持 TBO**。§32.6 那句"把 AR 藏进计算 ⇒ prefill ≈ 1900 t/s"**仍然没有地基** ——
   这次不是 §41 的"写独占资源"(那条今天已被 §49.2 的读/写对照拆开),
   而是**"链内依赖序 + 无抢占"这个调度事实**。

**⇒ 1.7× 的现实路线收敛到四条(全部已实测有据)**:
| 路线 | 收益 | 代价 |
|---|---|---|
| §45 **PCIe Gen4**(若 BIOS 可开)| 1.71× | **需人工(BIOS/iKVM)** |
| §46.2 **MPS 128→256** | +1.6% | **需 root**;两档可叠加 |
| §42.3 **NVLink 桥**(仅 3090 对)| ~1.80× | 采购 + 装车 |
| §47 **fp8 交换** | ~1.73× | **尾数 10→3(64× 粗化)**;判据待补 |

**TBO 要复活,必须先回答一个新问题**:**能不能把 AR 从"链内后到核"变成"独立流 + 事件回连"?**
(即 §33 的 B1 多流图,§39 已修好路由)—— A 的 AR 放支流、由事件连回 A 的下一层;
**但依赖不变,所以它自己不能填空,必须另有独立计算源。** 这是下一刀,不是本轮的结论。

---

### 50. 小包(α)优化清单对表:我们的战场是 **decode**,不是 prefill(2026-10-19;用户提供清单)

**用户口径**(贴来一份自定义集合通信库的小包优化清单,问"再看看这个")。清单主旨:
**小包场景 α ≫ β** —— 少跳数、少同步轮次、降每次同步开销、数据一到就能用;
手段:NCCL 的 **LL/LL128**(数据+flag 打包成原子写、接收方轮询 flag,**不用重 fence**)、
双缓冲、并行轮询、按大小自动切算法、GPU-initiated/持久核。

#### 50.1 两个战场要分清(我们的账)

| | **prefill** | **decode** |
|---|---|---|
| 载荷 | **5.24 MB**(512 token × 5120 × f16)| **10.24 KB**(1 token × 5120 × f16)|
| 主导 | **β**(写腿 411µs = Gen3 x16 的 81%,**已顶在 TLP/128B-MPS 天花板的 96%**)| **α**(传输只有几 µs)|
| 清单的适用度 | **几乎不适用** | **正中靶心** |

**★ 而且 decode 这笔账我们从来没算过,它可能很贵**:

| 事实 | 值 |
|---|---|
| ⚠️ **本行是错标签(2026-10-19 §58.2 更正)**:`42.14` 是 **TP2-v1 臂**(§35.3),不是 TP1 |
| ⇒ 真实 TP1 = **32.02 t/s** ⇒ 比值 **1.315×**;缺口 **不在 AR**(2.86ms),而在 **8.8ms「不随 rank 缩」的项**(§58.6) | |
| AR 次数 / token | 128(2/层 × 64 层)|
| 若每次 α ≈ 20µs | **2.56 ms / 23.4 ms = 11%** |

#### 50.2 ★ 栅栏成本按载荷的缩放(新读数;`tools/p2p-wr-tune.cu`,ord 0/3)

| 载荷 | +fence | −fence | **栅栏差** | 占写腿 |
|---|---:|---:|---:|---:|
| **10 KB(decode 档)** | **5.38 µs** | 3.33 µs | **2.05 µs** | **38%** |
| 128 KB | 16.57 | 12.49 | 4.08 | 25% |
| 5.24 MB | 408.75 | 399.55 | 9.20 | 2.3% |

**⇒ 栅栏 = "固定 ~1.5µs + 体积项"**:体积涨 **512×** 它才涨 **4.5×**。
**⇒ 在 decode 档它是写腿的 38%;在 prefill 档只占 2.3%。**

#### 50.3 清单逐条对表

| 清单条目 | 我们的状态 | 判 |
|---|---|---|
| 算法选择 Ring/Tree/One-shot/Two-shot | **N=2 ⇒ 四者退化成同一个 one-shot**;我们已是架构上的 one-shot(§35 抄 vLLM `cross_device_reduce_1stage` 的对称槽)| **无选项** |
| **LL/LL128:数据+flag 打包原子写**,别用重 fence | ❌ 没做。现役 = **每线程 `__threadfence_system()`**(= `fence.sc.sys`,**最重档**)+ 独立 flag | **可抄,有实测收益(50.2)** |
| 少同步轮次 | ❌ AR 有**三段**跨卡同步:`credit`(前置往返)→ 数据+flag 发布 → **尾部 `rendezvous`**。**rendezvous 是 `n_slots=1` 时代的遗产**,而我们有 2 槽乒乓 | **可抄** |
| 双缓冲 / 流水线 | ⚠️ 有 2 槽,**但没因此取消 rendezvous** | 半 |
| 并行轮询 + 归约 | ✅ 已做(§26:block0 轮询对端,其余读本端 `wait_done`)| — |
| GPU-initiated / 预注册 / CUDA Graph | ✅ 已做(图捕获 + 预注册对称槽)| — |
| 持久核 | ❌ 没做(图回放已摊掉 launch 税 ⇒ 优先级低)| 低 |
| **按大小自动切算法**(小包 LL / 大包 Simple)| ❌ 只有一条路(v1/push 两种**绑定**,不是两种**协议**)| **可抄** |
| 单机极小包走 SHM | ✗ 不适用(我们有 P2P 直连,SHM 更慢)| — |
| 跨节点 RDMA | ✗ 不适用(单机)| — |

#### 50.4 ★★ 缺口:我们**没有生产 decode 的 AR 设备时延**

- `ar_channel_bench` 的 10KB 档给的是 **`enq=35.4µs`(host 入队 —— 图回放里不存在)** 与 `dev=0.08µs`(不是 AR 全程,不含跨卡等待);
- ⇒ **decode 侧能拿多少,现在只能靠猜**。

**⇒ 下一步第一件(必须先做,否则全是盲打)**:用 `OWL_GPU_PROF=1`(逐发射同步 COMPUTE 流;**注意它本身 +4.9%**)
或 §40 的 metrics,量出**生产 decode 一次 AR 的设备时延三段账**(credit 等待 / 数据+flag / rendezvous)。

#### 50.5 decode 侧候选刀(按"每 AR 省多少"排;位置都指得出)

| # | 刀 | 落点 | 预期 |
|---|---|---|---|
| 1 | `fence.sc.sys` → `fence.acq_rel.sys`,flag 用 `st.release.sys` | `ar_stage.cu` / `ar_reduce.cu` 的发布点 | ~1.5µs/次(10KB 档)|
| 2 | **用 2 槽余量取消尾部 rendezvous** | `ar_stage.cu` 尾段 | 一次跨卡往返 |
| 3 | credit 改**非阻塞检查**(2 槽下上游必领先)| `ar_stage.cu` 头段 | 一次跨卡往返 |
| 4 | 小包专档核(更少 block、更早到 flag)| 新族臂 | 待测 |

⚠️ **1–3 都必须先过门**(`tp_g8_gate` L1 逐位一致 + `tp_transport_bits_gate` + 压力门):
**这三刀动的全是"同步面",而 §36.8 的教训就是同步面的块间私有令牌被搬走会静默出错。**

#### 50.6 ★ 第一刀实测(栅栏档扫描):**省在"单线程 release 发布",不在"放宽 fence"**

`tools/p2p-wr-tune.cu` 新增 `k_wr_fence<FM>` 四档(10 KB 档,K=300,ord 0/3):

| 档 | µs | 判 |
|---|---:|---|
| 无栅栏 | 3.33 | 下界 |
| **`fence.sc.sys`(= 现役 `__threadfence_system()`)** | **5.28** | 基准 |
| `fence.acq_rel.sys` | **5.29** | **零收益 ⇒ 猜想"SC 比 acq_rel 贵"被否** |
| **`st.release.sys`(LL 式:数据 relaxed 写 → `__syncthreads()` → 单线程 release 发布 flag)** | **3.89** | **−1.4µs / −26%** ✓ |

**⇒ 定谳**:**收益来自"把发布从**每线程 fence**收缩成**单线程 release-store**"(NCCL LL 的原始洞察),
不是来自"把 fence 语义放宽"。**⇒ 这刀值得做,但它**不是** decode 的 α 大头。**

**⇒ α 账重排(每 AR 次,10 KB 档;括号 = 依据)**:

| 项 | 量级 | 依据 |
|---|---:|---|
| 写/读腿搬运 | 3.9–5.3 µs | 本节实测(已含发布)|
| **跨卡 flag 往返 × 2–3 次** | **~4–6 µs?** | **未测**;credit 前置 + rendezvous 尾段各一次 |
| 对端 RMW(10 KB)| ~0.5 µs | 本地带宽 |
| 两次核发射/排空 | ~2–4 µs | 图回放内 |
| **合计(粗估)** | **~11–16 µs** | ⇒ ×128 = **1.4–2.0 ms / 23.4 ms = 6–9%** |

**⇒ 正确的下一刀是"砍跨卡往返"(候选刀 #2/#3),不是 #1**;而**要砍得准,必须先量出那两次往返**。

**⚠️ 记一条方法学教训**:`ar_channel_bench` 的 10 KB 档 `enq = 37.4µs` 是**与载荷无关的常数**
(5.24 MB 档也是 33.5µs)⇒ **它量的是 bench 自己的 host 命令面,不是 AR**。
**⇒ "decode 侧 AR 有多贵"这个问题,bench 从架构上回答不了**,必须在**图回放形态**里量:
下一件要做的是**把 stage+reduce 捕进图、回放 K 次**,量真设备时延(§50.4 的缺口)。

#### 50.7 ★★★★ 补上缺口:**图内 AR 真设备时延** —— decode 的 α = 19.8µs/次 = 11.2% 的 decode

**做法**(`ar_channel_bench` 新增臂;`OWL_ARB_GRAPH=1` + `OWL_ARB_GRAPH_N=64`):
把 **N 对 AR(stage+reduce)捕进一张图**(两卡各一张,**同时回放**)⇒ host 只出一次 launch;
**墙钟 / N = 每次 AR 的真设备时延**(含 credit 前置 + rendezvous 尾段两次跨卡往返)。K=8,`f16`,ord 2/3:

| 载荷 | **图内 AR/次** | 搬运下界 | **α = 差** | α 占该档 |
|---|---:|---:|---:|---:|
| **10 KB(decode 档)** | **20.58 µs** | ~0.8 µs | **~19.8 µs** | **96%** |
| 128 KB | 35.65 µs | ~10 µs | ~25.6 µs | 72% |
| **5.24 MB(prefill 档)** | **604.87 µs** | ~496 µs(v1 读腿 `R`)| **~109 µs** | 18% |

**三条定谳**:

1. **decode 的一次 AR = 20.58µs,96% 是 α**(同步/协议/发射),搬运只占 0.8µs。
   ⇒ ×128 次/token = **2.63 ms / 23.4 ms = 11.2% 的 decode** —— 与 §50.1 的粗估(11%)对得上,
   **这就是"TP 在 decode 只买到 +1.6%"的全部原因。**
2. **α 不是常数**:10KB **19.8µs** → 5.24MB **109µs**。
   而 prefill 档那 109µs 里 **~85µs 就是 §49.1 量到的 VMM 读腿税(17-20%)**
   ⇒ **prefill 的"α"是映射税,decode 的才是纯同步。两条腿要对症下药。**
3. **bench 的 10 KB 档 `AR=37.5µs` 是 host 面虚高**(图内真值 **20.58µs**)⇒ §50.6 那条
   "bench 从架构上回答不了 decode 问题"的判断被证实。

**α(10KB)≈ 19.8µs 的拆解(粗估,带待验项)**:

| 项 | 量级 | 依据 |
|---|---:|---|
| 搬运 | 0.8 µs | 10KB / 12.7GB/s |
| 发布栅栏 | ~2.0 µs | §50.6 隔离实测(LL 式可省 1.4)|
| **两次跨卡往返**(credit + rendezvous)| **~4–8 µs?** | **未单独测**(候选刀 #2/#3)|
| 两次核发射 + 排空(图内)| ~4–8 µs | 未单独测 |
| 对端 RMW | ~0.5 µs | 本地带宽 |

**⇒ decode 侧的可拿额度:砍掉 #2/#3 与部分发射税 ⇒ 乐观 ~8–10µs/次 ⇒ ×128 ≈ 1.0–1.3ms ⇒ decode +4–5%。**
(不是 11% 全拿 —— 搬运 0.8µs 拿不掉,发射税也有一部分是图回放的固有成本。)

**⇒ 与此同时,prefill 侧的对称结论更强**:α = 109µs/次 ≈ **19µs/token ≈ TP2 的 3.5%**,
而其中 **~85µs 是 VMM 读腿税** ⇒ **"把 SM 读腿从 VMM 别名换回 runtime 映射"这一刀,
在 prefill 上值 ~3%**(且不影响 fp8/写腿那些账)。**这是 §49.1 那个洞的直接兑现路径。**

#### 50.8 ★★★★ 更正:"VMM 读腿税 17~20%" 是**跨程序相减**的伪差;真值是 **4~6%**

**动机**:§50.7 结尾把 prefill 的 α(109µs)大部分归给"VMM 读腿税",并据此建议"换回 runtime 映射"。
**判据**:`tools/vmm-peer-probe.cu` —— **同进程、同对卡、同核、同载荷,只换"对端块是怎么来的"**(K=200,5.24MB,f16)。

| 臂 | **3090 Ti 对**(ord 0/1;`Static`)| **3080 对**(ord 2/3;`Window`)|
|---|---:|---:|
| **RT-read-1way**(`cudaMalloc` + `cudaDeviceEnablePeerAccess`)| 404.95 µs / **12.95 GB/s** | 405.39 / 12.93 |
| **VMM-read-1way**(owl `peer_import` 同款)| 420.25 / **12.48** | 429.21 / 12.22 |
| VMM-read + fence | 432.06 / 12.13 | 444.05 / 11.81 |
| RT-read-2way | 477.02 / 10.99 | 495.14 / 10.59 |
| VMM-read-2way | 510.04 / 10.28 | 549.91 / 9.53 |

**⇒ 定谳:映射路径的账是 **1-way 3.8–5.9% / 2-way 6.9–11.1%**,两对卡同量级。**

**★ 更正 §50.7 与 §49.1 的归因**:我此前说的"VMM 读腿税 **17–20%**"是**跨程序相减**得到的
(owl 带协议的 `R` 臂 496.5µs vs 本 probe 的裸读 413.5µs)—— **那个差里大部分不是映射路径**。
**现在同程序 A/B 把它的真值钉在 4–6%(1-way)。**

#### 50.9 ★ prefill 的 α = 109µs 重排(全部同程序、可复现)

| 项 | 量级 | 依据 |
|---|---:|---|
| 搬运(v1 读腿 `R`)| 496 µs | `ar_channel_bench`(§49.1)|
| **映射路径税(VMM vs runtime)** | **~15–25 µs** | 本节(1-way 3.8–5.9%)|
| **发布栅栏** | **~12–15 µs** | 本节 `VMM-read+fence` − `VMM-read` |
| **协议(credit 前置 + rendezvous 尾段 + 两级等待)** | **~50–70 µs** | 差额;§50.7 的图内 AR 605µs − 上述各项 |
| 两核串联 | 5.3 µs | §38.1 |
| **合计** | **~109 µs** | 与 §50.7 的图内读数自洽 |

**⇒ ★★ 两条腿收敛到同一刀**:
- **prefill 的 α 大头 = 协议那 ~50–70µs/次 ≈ 12–17µs/token ≈ TP2 的 2–2.7%**;
- **decode 的 α 大头 = 同一件事(两次跨卡往返)≈ 4–8µs/次 ≈ decode 的 3–4%**。
- ⇒ **"砍协议往返"(§50.5 的刀 #2/#3:用 2 槽余量取消尾部 rendezvous、credit 改非阻塞)
  是 prefill 与 decode 共同的第一刀;而"换映射"排到第二(值 ~0.5% of TP2,且 Window 档可能落不了地)。**

**⇒ 处置**:`tools/vmm-peer-probe.cu` 的读数存档;§49.1/§50.7 的"VMM 17–20%"表述**作废**,
以本节 3.8–5.9%(1-way)为准。

---

### 51. ★★★★ 融合两核(1-stage)首次真读数:**−4.7%** —— §35.3 的"中性"作废(2026-10-19)

**线索链**:§50.9 估的"协议 50–70µs"被 §50.10 的**图内旁路实测否掉(≈0)**;
⇒ 转而量**核数**:图内 **10 KB 档单核 ≈ 11.2µs、第二个核的边际成本 ≈ 9.3µs**(N 扫描 4/16/64 = 16.63/12.44/11.25,
**随 N 收敛 ⇒ 无图节点去重**)⇒ **decode 每次 AR 的 20.5µs 里,~9.3µs 纯粹是"第二个核"。**
⇒ **融合 stage+reduce 成 1 核 = 每次省 ~9.3µs**;而这正是 §35 已实现的 `OWL_TP_1STAGE` ——
**它在 §35.3 被判"中性",但 §35.5 已查明"当时从未执行" ⇒ 从未被真正计时过。**

#### 51.1 读数(0.8B,`tp_perf_split_probe`,ord 2/3;各跑 2 次)

| 指标 | **v1(两核)** | **1-stage(融合一核)** | 差 |
|---|---:|---:|---:|
| `prefill.chunk` 总 | 2059.7 ms | **1991.8 ms** | **−3.3%** |
| 每 chunk **p50** | 37.46 ms | **35.94 ms** | **−4.1%** |
| 总墙钟(#1/#2) | 0.553 / 0.552 s | **0.521 / 0.530 s** | **−4.7%** |
| **路径判据** | v1=2016 · fusion=0 | **fusion=2016 · v1=0** | **真在跑 ✓** |
| p99 | 285.18 ms | 278.61 ms | −2.3% |

**⇒ 定谳**:1-stage 的收益是**真的**,§35.3 的"中性结论"**作废**(§35.5 已判定那次是 v1-vs-v1 的假读数)。

#### 51.2 为什么收益随载荷变:与"核数账"自洽

| 档 | AR 载荷 | 每次 AR | 融合省 | 占比 |
|---|---:|---:|---:|---:|
| **0.8B prefill** | 128 KB | ~35 µs | ~9.3 µs | ~27% ⇒ 本表 −4.1% ✓ |
| **27B prefill** | 5.24 MB | 601 µs | ~9–10 µs | **~1.7%** |
| **27B decode** | **10 KB** | **20.5 µs** | **~9.3 µs** | **≈45%** ⇒ **×128/token ≈ 1.19ms ≈ decode 的 5%** |

**⇒ 推论(待验)**:**1-stage 在 27B decode 上值 ~5%,在 27B prefill 上只值 ~1.7%。**
⚠️ 而 `OWL_TP_1STAGE` 与 `runtime.tp_push` **互斥**(§37.4:两者都改写数据腿,同时开会静默走 push)
⇒ **两者不是"谁更好",而是"按档选":prefill 拿 push 的写腿(+4%),decode 拿 1-stage 的融合(待验 ~5%)。**
**⇒ 下一步:26B 档的 decode A/B(`qwen38` flavor + `llm_speedtest` 或 std)把这条推论钉死。**

**⚠️ 顺手记一条工具纪律(本轮踩的)**:`ar_channel_bench` 的探针臂**必须自带 `set_hdr`** ——
我新加的"单核臂"继承了旁路重放后的 HUGE 计数,撞上 reduce 的
`while (*peer_prod <= cons)` ⇒ **卡满 5s 超时 ×N**(读数 5000026µs)。
**⇒ 结论:用哨兵值(HUGE)旁路等待的臂,只能用于"不等数据"的核(stage);reduce 档不能靠 HUGE 混过。**

---

### 52. ★★★★★ 1-stage 在 **27B(量化档)从未生效** —— 定位到设计前提,不是标签问题(2026-10-19)

**动机**:§51 在 0.8B 上量出 1-stage(融合核)**−4.7%**,据此推测 27B decode 值 ~5%
⇒ 直接上生产 A/B(`tools/tp-ab-std.sh tp2`,ord 2/3,27B AWQ-INT4):

| 指标 | **v1** | **OWL_TP_1STAGE=1** | 差 |
|---|---:|---:|---:|
| prefill avg | 1487.31 t/s | 1482.97 t/s | **−0.3%** |
| decode avg | 42.18 t/s | 41.98 t/s | **−0.5%** |

**⇒ 与 0.8B 的 +4.7% 方向相反** ⇒ **先查路径**(§35.5 的教训),用 `--perf-level counters --perf-dump`:

```
tp.ar.r0 = 256 · tp.ar.r1 = 256 · tp.ar.v1 = 512   ← 全走 v1 两段
tp.ar.1stage                                        ← 一行都没有(= 0)
```

**⇒ ★ 刚才那个 A/B 是 v1 vs v1,−0.3%/−0.5% 是纯噪声;我又站在 §35.5 那个坑边上了,这次被计数器拦住。**

#### 52.1 逐层定位(临时探针,已拆净)

| 环节 | 读数 | 判 |
|---|---|---|
| 启动装配 | 日志 `1-stage arena 就绪:每侧 2 槽 × 8MiB + 64B header`,fd 80–85,peer import 权限 R | **arena 真建好了** ✓ |
| `Linear::new` | `key=out_proj ⇒ **UniformRow**` · `key=down_proj ⇒ **UniformRow**` | **规则表正确** ✓ |
| 注入点 `symmetric_out_slot`(首批 24 次)| `is_ar=false tp=false arena=false` | 那批是**装载期 warmup**(无 tp 上下文)|
| 注入点(只统计 tp 在场)| **`调用=6144 · tp在场=4472 · is_ar=0 · 可用槽=0`** | **TP 在場 4472 次,一次都没命中** |

#### 52.2 ★★★ 真因:`forward_inner` 的**窄视图**把角色挡在了外面

`crates/models/src/layers/linear.rs`:
- `forward()`(唯一给节点打标处)= `self.forward_inner(xs).with_tp_role(self.tp_role)`
  ⇒ **角色打在"返回的那个节点"上**;
- 而**量化路径** `forward_inner` 返回的是
  `narrow_strided(&gemm, m, n_pack, 0, self.out_dim, vec![m, self.out_dim])`
  —— **行 stride = `n_pack`(打包宽)的窄视图**,里面的 `gemm` 节点**没有角色**;
- 注入点在 `Op::Matmul` / `Op::Call` 的求值里读的是 **`t.tp_role`(= 那个 GEMM 节点)**
  ⇒ 永远 `Replicate` ⇒ `is_ar=false`。

**⇒ 而 0.8B 为什么生效?** 它是 **`QuantPlan::F16`** 直读档 ⇒ **没有 `narrow_strided` 包装**
⇒ GEMM 节点本身就是被打标的那个节点 ✓✓ —— **两条路径的差别正在这里。**

#### 52.3 定谳:**这不是"补个标签"能修的,是设计前提不成立**

1-stage 的核心假设是 **"GEMM 的输出**原样**就是 AR 载荷"(写进对称槽即成立)——
**这只在 f16 直读路径成立**;量化路径的 GEMM 输出是 **带 padding 的打包布局**,
与槽要求的连续 `[m, out_dim]` **不是同一块内存**:
- 想在量化档启用 ⇒ 要么槽按**打包布局**申请 + AR 核按 **strided 切片**规约(新能力,动核与协议),
- 要么中间再加一次拷贝(**等于把融合的好处还回去**)。
⇒ **按 §51.2 的账,这扩展的收益上界是 27B prefill ~1.7% / decode ~5% —— 值不值得做是一个取舍,不是 bug 修复。**

**⇒ §51 的结论仍然成立**(0.8B 那条路真的走了融合核),但**不可移植到量化档**;
**27B 生产档(v1)的 AR 成本结构与 §50 的账一致,没有"白捡的融合收益"。**

**⚠️ 两条纪律再记一次**:
1. **门的绿 / 装配的"就绪"都不能证明路径在跑** —— 唯一判据是 `tp.ar.1stage` 这类**计数器**
   (§35.5 → §39 → 本节,同一族坑第 3 次);
2. **临时探针必须当场拆净**(本轮 3 处探针 + 1 处假设性改动已全部撤掉,树里不留临时代码)。

---

### 53. ★★★★ fp8 的收益:**把载荷减半直接量** —— 实测 −46%,比值 1.61 → 1.74(2026-10-19)

**动机**:§47 的 fp8 第一步(数值等价件)已过门,但"真把字节砍半"的收益一直只有**推算**。
**判据**:不用实现 —— **直接测 AR 在"半载荷"下的时延**(`ar_channel_bench`,`OWL_ARB_SIZES`,K=200,ord 2/3)。

| 载荷 | **v1(读腿)** | **push(写腿)** |
|---:|---:|---:|
| 1.31 MB | 162.83 µs | 153.70 µs |
| **2.62 MB(= fp8 后的 5.24MB)** | **307.81 µs** | **265.49 µs** |
| **5.24 MB(现状)** | **590.64 µs** | **493.52 µs** |

**⇒ fp8(e4m3,1B/元素 + 尺度)= 载荷减半 ⇒ 每次 AR 省:**
- **push 档:493.52 → 265.49 = −228.03 µs(−46.2%)**
- v1 档:590.64 → 307.81 = **−282.83 µs(−47.9%)**

**换算到 TP2(push 档)**:

```
AR/token = 493.52 × 128/512 = 123.4 µs  →  265.49 × 128/512 = 66.4 µs   (省 57.0)
TP2 = 651.1 − 57.0 = 594.1 µs/token ⇒ 1683 t/s ⇒ **1.74×**(vs TP1 = 966.06)
```

**⇒ 与 §47.5 的推算(1.73×)吻合 ⇒ 这条路的收益已被实测确认(代价 = f16→e4m3 的尾数 10→3,§48.1)。**

**顺带一条结构观察**:1.31 MB 档 153.70 µs = 8.5 GB/s(按 12.7 GB/s 线性外推应 ~103 µs)
⇒ **每档有 ~50 µs 的固定开销** —— 与 §50.7 的 decode 20.5µs/AR 是**同一族现象**(固定成本主导,
搬运只占小头)。**⇒ 小包优化的靶始终是"固定成本",不是带宽。**

---

### 54. ★★★★★ 逐点对表:**vLLM 的 AR 与我们到底差在哪**(2026-10-19;用户要求读源码)

**用户口径**:"去看 vLLM 的源代码,特别是一些细节的地方,它做的很不一样的地方 —— 它到底差在什么地方?"
本节是**源码级**对表(`packages/vllm` @ `e20bccc642`),并给出每条的**我方实测代价**。

#### 54.1 一次 allreduce 的**结构**(这是最大的一条)

**vLLM**(`csrc/libtorch_stable/custom_all_reduce.cu:70-95` + `csrc/custom_all_reduce.cuh:7-23`):

```cpp
if (reg_buffer) {                       // ① 输入**不在**对称缓冲里
  cudaMemcpyAsync(reg_buffer, inp, n, D2D, stream);   //    拷贝引擎搬一次 —— **零同步语义**
} else { reg_buffer = inp; }                          //    已在对称缓冲 ⇒ 连搬都省
fa->allreduce(stream, reg_buffer, out, numel());      // ② **一个**融合核
//   核体:barrier_at_start → packed_reduce(ptrs, idx) → barrier_at_end
```

**⇒ vLLM = 2 个 GPU 操作(CE 拷贝 + 1 核)、**1 次同步**(同步全在那一个核内)。**

**owl(现役 v1)**= `ar_stage` 核(本地拷贝 + **`__threadfence_system()`** + 发布 `produced` + credit 检查)
+ `ar_reduce` 核(等 `peer.produced` + 读对端槽 + 加 + 发 `consumed` + 尾部 rendezvous)
**⇒ 2 个核、4 段跨卡同步。**

| | vLLM | owl | **我方实测代价** |
|---|---|---|---|
| 第一段 | **`cudaMemcpyAsync` CE(无同步语义)** | **`ar_stage` 核 + 系统级栅栏 + flag 协议** | 图内"第二个核"边际 **9.3µs**(10KB,§50.10)|
| 同步次数 | **1**(核内 barrier)| **4**(produced / consumed / credit / rendezvous)| — |
| flag 原语 | **`st.release.sys` + `ld.acquire.sys`**;⚠️ **末次 barrier 用 relaxed/volatile**(注释:*"最终同步不需要可见性保证"*)| `__threadfence_system()`(=`fence.sc.sys`,**最重档**)+ 两级等待广播 | LL 式 release 省 **1.4µs/AR**(§50.6)|
| flag 写入者 | **`if (threadIdx.x < ngpus)`** —— world=2 只用 **2 个线程** | 单一计数器 + block0 轮询 + 本地 `wait_done` 广播(§26)| — |
| 数据面 | **读所有 rank 的对称缓冲**(`packed_reduce(&dp.ptrs[0], idx)`,pull)| v1 = 读对端槽(pull);push = 写对端槽(write)| 读写腿不对称 §48 |
| 向量/累加器 | `array_t<T,16/sizeof(T)>`(**16B packed**)+ float 累加器 | `uint4` 8 路展开 + float 中间 | 同量级 |
| 发射几何 | `threads=512`,`blocks = min(36, ceil(size/512))`(36 = 网格搜索结果,注释:*"too many SMs will cause contention on the bus"*)| `MAX_THREADS = 8192` ⇒ **固定 16 块** | 我们自己的 sweep:**写腿块越少越快**(4 块 402µs vs 16 块 411µs)→ **这条不能照抄,要按自己的核重定** |
| 大消息 | **> 8MB 回落 NCCL**;world>2 按 512KB/256KB 阈值切 1stage/2stage | 无回落,全尺寸自研 | — |
| world 支持 | `[2,4,6,8,16]` | 2 | — |
| 图上 | 捕获期:**rank-data 池 + `graph_unreg_buffers_` 逐个登记** | 捕获期把参数烤进节点 | — |

#### 54.2 ★★★ 为什么 vLLM 这样能行 —— 一个可以直接抄的**关键洞察**

**vLLM 没有"数据就绪"的 flag 协议**(我们那套 `produced`/`consumed`/credit/rendezvous)。它的保证来自两层:

1. **`cudaMemcpyAsync` 与核在**同一条流**上 ⇒ 拷贝完成才发核**(stream 序);**
2. **核首 `barrier_at_start` 是"全体到场"语义** ⇒ 对端的**核**已经开跑,而对端的核排在它自己的拷贝之后
   ⇒ **对端的数据必然已就位。**

**⇒ 于是"数据就绪"不需要任何专门的 flag 协议 —— 用「流序 + 到场 barrier」取代了我们的「发布 + 轮询 + credit + rendezvous」。**

#### 54.3 我方 §52 的死结,在 vLLM 里就是"unregistered 分支"

§52 定谳:量化档的 GEMM 输出是**带 padding 的打包布局**,进不了对称槽 ⇒ 1-stage 死。
**⇒ vLLM 的答案是:不要求输入在对称缓冲里 —— 先 CE 拷进去,再跑同一个融合核**(`reg_buffer` 那支)。
**⇒ 这正是我们缺的那条支路**:§35 的 `owl_comm_ar_1stage_*` 已经存在于"输入已在槽里"的形态,
**只差"先 CE 拷进槽"这一步**(而 owl 早有 `copy_block_at` 这个 CE 原语,§32 用过)。
⚠️ 但 §32 把 CE 接线回退了,理由是"CE 三段(=stage→rendezvous→CE 拷→add)在生产形状下没有收益"
—— **那是四段串行;vLLM 是两段(CE 拷 + 融合核)**。**⇒ 两者不是同一件事,§32 的结论不适用于本结构。**

#### 54.4 三条可抄的清单(按我方实测代价排序)

| # | 抄什么 | 我方预期 | 依据 |
|---|---|---|---|
| **A** | **把 `ar_stage` 核换成 CE 拷贝 + 让融合核接管同步**(= §54.3 的支路)| **省一个带同步语义的核 ⇒ decode ~5%**(9.3µs/AR × 128 = 1.19ms / 23.7ms)| §50.10 图内单核台 |
| **B** | **flag 用 `st.release.sys`/`ld.acquire.sys`,末次 barrier 用 relaxed** | 1.4µs/AR(10KB)×128 = **179µs = decode 0.8%** | §50.6 隔离台 |
| **C** | 几何"按尺寸 cap 块数" | **待定**(我们的核与它不同向:我们量到块越少越快)| §46.1 |

**⇒ A 是唯一的大项,而且它有 vLLM 生产结构背书 + 我方实测代价;B 顺手;**C 需要按自己的核重扫。**

---

### 55. AR 模块**全量清点 + 全门复跑 + "差在哪"的最终形态**(2026-10-19;用户要求"全弄过来、都测一下")

#### 55.1 模块清点(全部读过一遍)

| 层 | 文件 | 行 |
|---|---|---:|
| **核** | `cu/comm/ar_stage.cu` | 283 |
| | `cu/comm/ar_reduce.cu`(2-stage + **1-stage 融合核**) | 532 |
| **声明表** | `kernels/src/ops/comm.rs` | 222 |
| **发射面** | `kernels/src/server/comm.rs`(`launch_ar` + `ar_reduce_grid` + `MAX_THREADS`)| 355 |
| **客户端** | `kernels/src/client/comm.rs` | 347 |
| 族 id | `kernels/src/family/comm.rs` | 95 |
| **设备侧** | `backends/cuda/src/comm/{ar,peer,mod}.rs` | 586 |
| | `backends/cuda/src/state.rs`(捕获/图/池,AR 是其中一段)| 2804 |
| **引擎侧** | `engine/src/tp.rs`(staging + arena + handshake)| 841 |
| 注入点 | `models/src/interpreters/eval.rs`(`symmetric_out_slot` + 三处 AR)| — |

#### 55.2 ★ 11 个门/探针**全绿**(复跑记录,ord 2/3)

| 门 | 结果 | 备注 |
|---|---|---|
| `raw_sync_primitives_gate` | **ok 1/1** | 无卡档 |
| `peer_shared_gate` | **ok 1/1** | cap=192MiB staging=64MiB |
| `ar_transport_gate` | **ok 1/1** | G1 11 case(三 dtype + 32MiB 跨 chunk)/ G2 两跑逐位同 / G3 捕获 replay ≡ eager |
| `graph_multistream_gate` | **ok 4/4** | ⚠️ **必须 `--test-threads=1`**;并发跑会 `CUDA_ERROR_STREAM_CAPTURE_INVALIDATED`(同进程共享设备/捕获态)⇒ **测试卫生问题,不是功能 bug**(m5 单跑 1.090ms/回放 ✓)|
| `graph_memcpy_gate` | **ok 3/3** | 含跨卡 peer 搬运回放 2 次读到最新值 |
| `ar_stress_gate` | **ok 3/3** | |
| `tp_assemble_gate` | **ok 1/1** | |
| `tp_determinism_gate` | **ok 1/1** | |
| `tp_g8_gate` | **ok 2/2** | L1 = 0 项不同 |
| `tp_transport_bits_gate` | **ok 1/1** | |
| `ar_channel_bench` | **ok 1/1** | 微基准 |

#### 55.3 ★★ 1-stage 融合核的真语义(读完核体)——它纠正了 §54.4-A 的落地形态

```cuda
// owl_comm_ar_1stage_*(out, own, peer, peer_data, bytes)
① 发布 produced / 两级等待 / credit 头段
② __threadfence_system()                     // acquire 全线程
③ dst = out(本端 payload); src = peer_data(对端 payload)
   OWL_AR_REDUCE_FAST(T, dst, src, n);       // ⇒ **out += peer_data(原地求和)**
④ __threadfence_system() + consumed + 尾部 credit(p − n_slots)
```

**⇒ 它要求「**载荷本体住在对称槽里**」(对端要能读到我的 payload)—— 这就是 §52 量化档死掉的根源。**
**⇒ 而 vLLM 的 `out = torch.empty_like(inp)` ⇒ 它的 AR 是 **out-of-place**(读注册副本、写新张量)⇒ **不需要把结果拷回**。**
**⇒ 所以"CE 拷进 + 1-stage"对我们的 in-place 语义**还差一次拷回 ⇒ 那是 3 个 GPU 操作,不比现在好。**

#### 55.4 ★★★ 正确的落地形态(修正 §54.4-A)

**关键观察:v1 的 `ar_reduce` 核**本来**就是"读对端的槽 + 加到我的 payload"** ⇒
**它不要求我的载荷住在槽里** —— 只有"为了让对端能读到我"这一件事需要槽。

```
vLLM : [CE 拷进对称缓冲(流序,零同步语义)] + [1 核:barrier_at_start → 读对端 → 加 → barrier_at_end]
owl  : [ar_stage 核:拷进我的槽 + **系统级栅栏** + 发布 produced + credit] + [ar_reduce 核:等 produced + 读对端槽 + 加 + 发 consumed + rendezvous]
```

**⇒ 改造(v1 → vLLM 同构),载体不变(仍是那对对称槽),因此 **§52 的量化档问题不存在**:**

| # | 动作 | 省 |
|---|---|---|
| 1 | `ar_stage` 核 → **`copy_block_at`(CE)**:把私有 payload 拷进我的槽 | **一个带同步语义的核**(图内边际 9.3µs/AR @10KB)|
| 2 | reduce 核的 `produced` 发布/轮询 → **核首"到场 barrier"**(用**流序**保证对端拷贝已完成,如 vLLM)| `produced` 协议 |
| 3 | 删掉 credit 与尾部 rendezvous(2 槽乒乓的余量本来就够)| 两段跨卡往返 |
| 4 | flag 用 `st.release.sys`/`ld.acquire.sys`,末次 barrier 用 relaxed | 1.4µs/AR(§50.6)|

**⇒ 预期合计 ≈ 9.3 + 1.4 ≈ 10.7µs/AR ⇒ ×128 = 1.37ms / 23.7ms = **decode ~5.8%**。**
⚠️ 风险点三条:①CE 拷贝与核的**流序**必须真成立(§32 的 CE 是"三段串行"另一种结构,不是本形态);
②删 credit/rendezvous 是**同步面**改动,必须过 `ar_stress_gate`(压力门)+ `tp_g8_gate`(L1 逐位)+ `ar_transport_gate`(捕获≡eager);
③`n_slots=2` 的乒乓语义要重新论证(现役 credit 是按 `n_slots` 写的)。

---

### 56. ★★★★★ AR **结构变体对比台**:四协议同数、结构收益的**下界**、以及一个几何新发现(2026-10-19)

**动机**:§54 从 vLLM 源码看出结构差别([CE 拷贝 + 1 融合核] vs [stage 核 + reduce 核]),
§55.4 据此提改造 —— **但没有分母**。用户口径:"多测几个变化。"
**工具**:`tools/ar-struct-probe.cu`(新;同载荷、同两卡、**图内回放**、host 只出一次 launch、**带落地校验**)。

#### 56.1 四种协议(10 KB;N=32/图;REPS=5;ord 2/3)

| 变体 | µs/AR | vs A |
|---|---:|---:|
| **A 现役 v1 形状**(stage 核 + reduce 核 + produced 协议)| 9.03 | — |
| B `CE 拷贝 + 融合核`(sc.sys)| 8.24 | **−0.79** |
| C 同 B + `st.release.sys`/`ld.acquire.sys` | 7.84 | −1.19 |
| **D 同 C + 末次 barrier relaxed**(vLLM 的两个洞察都用上)| **6.77** | **−2.26(−25%)** |

**★ 四个变体的 `out[0]` 完全一致(0x66666651 = 0x11 + 32×0x22 ⇒ 32 次 AR 全对)**
⇒ **四种协议数值等价** ✓(这是本台的落地校验,不是推测)。

#### 56.2 三档尺寸(每档都四协议同数)

| 载荷 | A | B | C | **D** | **D−A** |
|---:|---:|---:|---:|---:|---:|
| **10 KB(decode)** | 9.03 | 8.24 | 7.84 | **6.77** | **−2.26(−25%)** |
| 128 KB | 22.92 | 19.01 | 18.55 | **17.54** | **−5.38(−23%)** |
| **5.24 MB(prefill)** | 521.06 | 510.29 | 506.05 | **502.00** | **−19.06(−3.7%)** |

#### 56.3 ★★ 几何扫描 —— **一个没人量到过的新发现**

| blocks | A @10KB | D @10KB | A @5.24MB | **D @5.24MB** |
|---:|---:|---:|---:|---:|
| 4 | 9.06 | 6.75 | 556.95 | 507.52 |
| **16(我方生产几何)** | 8.49 | **6.27** | **517.91** | **501.91** |
| 36(vLLM 几何)| 8.57 | 6.30 | 557.17 | 501.88 |
| 68 | 8.49 | 6.26 | 602.92 | 503.51 |

**⇒ A(两核)对块数敏感(557 → 603,±8%);D(CE 起始)几乎不敏感(502 ~ 508,±1%)。**
**根因**:A 的第一段是 **SM 写**(块越多越慢,§46.1 已量);D 的第一段是 **CE**(与 SM 无关)。
**⇒ 这正是 vLLM 敢设 36 块、并注释"too many SMs will cause contention"的底气 —— 它的第一段不吃 SM。**
⚠️ **推论**:D 结构对"生产里 AR 与别的核并发"也更友好(不吃 SM 槽)。

#### 56.4 ★★★ 收益修正(诚实):结构本身只值 **decode +1.2% / prefill +0.6%**

```
decode:  −2.26 µs/AR × 128 = −289 µs / 23700 µs = **+1.2%**
prefill: −16.0 µs/AR × 128/512 = −4.0 µs/token ⇒ TP2 651.1 → 647.1 = **+0.6%**
(取 blocks=16 = 我方生产几何的 D−A;36/68 块下 A 更慢是 A 被几何拖累,不算 D 的功劳)
```

**⇒ 这与我在 §54.4/§55.4 说的 "decode ~5.8%" 差得远 —— 那个数字里混进了别的东西。修正如下。**

#### 56.5 ★★★ 那 5.8% 的真相:差额在"**核的同步面**",不在"结构"

**本台的 A 是"v1-lite"**(只有 1 次 `st_vol` 发布 + 1 次自旋 + 1 次 fence),它 = **9.03µs**;
**而生产 AR @10KB = 20.58µs**(`ar_channel_bench` 图内实测,§50.7)⇒ **差 11.5µs**。

**⇒ 这 11.5µs 才是 decode 的大头,内容是:**
| 项 | 生产有、本台 A 没有 |
|---|---|
| `__threadfence_system()` 次数 | 生产 4 处(全线程),本台 2 处 ⇒ §50.6 量过单次 ~1.4–2µs |
| credit 检查 | 生产有(每次跨卡往返的余量)|
| 尾部 rendezvous | 生产有 |
| 两级等待(`wait_done`/`blocks_done` 广播)| 生产有(多 block 协调)|
| 映射路径 | 生产走 **VMM 导入**(§50.8 量过读腿 +3.8~5.9%)|
| 核形 | 生产 8 路展开 + 逐通道谓词 |

**⇒ 结论(修正版)**:
- **"换结构"单独 = decode +1.2% / prefill +0.6%**;
- **"把生产的 20.58µs 瘦到本台的量级" = 另一件更大的事,上限 decode ~11.2% 的绝大部分**;
- **两者是同一次改动的两面**:D 的结构([CE 拷贝] + [一个核:到场 barrier → 加])
  **天然不需要 credit / rendezvous / 两级等待**(用"流序 + 到场 barrier"取代)⇒
  **做结构改造时,瘦身是顺带的,但它的收益要在生产上量**(本台量不到,因为 A 已经是瘦的)。

**⇒ 诚实的验收口径**:改完在生产 `ar_channel_bench` 上量**图内 AR 真设备时延**(§50.7 的那把尺),
判据 = **20.58µs 降到 ~9µs 量级**(不是"本台 A→D 的 2.26µs")。
**⇒ 预期:decode +4~5%、prefill +1~2%。**

---

### 57. ★★★★★ 动生产的 D 结构:失败 + **根因**;并挖出 1-stage 的**既有竞态**(2026-10-19)

#### 57.1 做了什么

按 §56.2 的 D 形状,把 **1-stage 融合核的协议**换掉(它生产缺省关闭 ⇒ 零生产风险):
`[fence + produced++ + 等 peer.produced]` → **[取本端票 → 发布到对端(RW header)→ `ld.acquire.sys` 等对端票]**,
并把 arena 的 **header 导入改成 RW**(与 push 档同款)。

#### 57.2 读数:**L1 门失败**

`tp_g8_gate`(严格 L1:两 rank 末步 logits 必须**逐位相同**):

| 档 | L1 结果 |
|---|---|
| 缺省(v1)| **0 项不同 / max\|Δ\| = 0.000000** ✓(复跑 2/2 稳定)|
| 1-stage + **D 协议** | **246147 项不同 / max\|Δ\| = 0.6** ✗ |
| 1-stage + **原协议(回退后)** | **两次运行分别 0 与 243755** ⇒ **间歇性!** |

#### 57.3 ★★★ 根因(三条,都已落到证据上)

1. **vLLM 的 barrier 语义是"把票**写进对端内存**(push 方向)**;
   而本引擎有一条血泪规则(§36.8/§37):**header 必须留在本端**(把 header 搬到对端 = 大负优化)。
2. **更根本的一条**:1-stage 的**载荷由上游 GEMM 写**(多 block × 多线程),
   `__threadfence_system()` **只排序调用线程自己的写** ⇒ **单线程的 fence 无法为整个 GEMM 的写的系统可见性背书**。
   vLLM 之所以能这么干,是因为它的载荷是**紧邻核的一条 CE `cudaMemcpyAsync`**(单一引擎操作 + 流序可背书)。
   **⇒ 这正是 §54.2 那条"流序 + 到场 barrier"洞察的适用边界:它对 CE 拷贝成立,对"上游是个 SM 核"不成立。**
3. **⇒ 推论(重要)**:本引擎现役的 `produced` 轮询协议 **其实已经是"到场 barrier 的 pull 形态"** ——
   "票"就是 `produced`,"等对端"就是 `while (*peer_prod <= cons)`。**D 结构在本引擎里没有可换的空间,
   除非连"flag 写向对端"一起改**(而那要重做 header 的归属与可见性规则)。

#### 57.4 ★★★ 顺带挖出的既有缺陷:**1-stage 的 L1 是间歇性的(0 / 243755 两次不同)**

- 严格 L1 判据(`nbit == 0`)在 **1-stage 档**上**时过时不过** ⇒ **该路径存在跨 rank 竞态**;
- ⇒ **§51 的"1-stage −4.7%"必须标注为"在数值不可靠的路径上测得的提速"**
  (`tp_perf_split_probe` 只查路径计数与耗时,**不查数值** —— 这正是"门绿≠对"的同族陷阱:
  **测速台不校验数值,就永远发现不了它**);
- **⇒ 1-stage 若要转正,必须先修这个竞态**(候选:①`produced` 的发布是否被上游写的可见性背住;
  ②`n_slots=2` 乒乓与 `cons` 的推进是否真互斥;③`wait_done` 两级等待与 peer 轮的交互)。

#### 57.5 处置(树已回绿)

| 项 | 处置 |
|---|---|
| 1-stage 核体的 D 协议 | **回退**(恢复 `produced` 语义;文件里留一行注释记档)|
| arena header 的 RW 导入 | **回退**(改回 `PROT_READ`)|
| `tp_g8_gate` 的 push/v1 路径断言 | **保留放宽**(1-stage 档跳过 —— 它归 `tp_perf_split_probe` 管)|
| 树状态 | **`tp_g8_gate` 缺省档 L1 = 0 项不同 ✓ 绿**;`cargo build` 干净 |

**⇒ 下一步(不是本轮)**:**先修 1-stage 的 L1 竞态**(它是"融合核"这条路的地基,也是 §51 结论的前提);
之后 D 结构才谈得上 —— 而按 §57.3,它需要的不是"换协议",是**把 header 归属与可见性规则重做一遍**。

---

### 58. ★★★★★ decode 账的**第一次全量分解**(nsys 图模式逐核 + 分卡;2026-10-19 深夜)

**用户口径**:"你先测一遍啊。这 decode 一点提升都没有,大问题啊!"

#### 58.1 ★★★★ 更正:那条"decode 零提升"是**错标签**(先纠分母,再谈刀)

§50.1 的表里写着 `TP2 decode vs TP1 decode | 42.80 vs 42.14 t/s ⇒ TP 只买到 +1.6%`,
并据此推出"差的全部是 AR 的 α"。**42.14 不是 TP1 —— 它是 §35.3 表里的 TP2-v1 臂**
(`v1 / 1-stage / 1-stage` 三臂:1477.66/42.14、1483.19/42.25、1479.52/42.13)。
真正的 TP1 单卡 decode 在 §37 表里是 **31.98 t/s**,本轮复测 **32.02**(同一对 3080,同会话)。

**⇒ 真实的 decode 比值 ≈ 1.315×(不是 1.00×),但"账不对"这件事成立**(见 §58.6):
TP2 每卡流量只有 TP1 的 **0.576×**(见 §58.3 的 lm_head),decode 却只快 1.315×。

#### 58.2 本轮 A/B 口径(`tools/tp-ab-std.sh ab`;同二进制、同卡对 2/3、同会话)

| 臂 | prefill avg | decode avg | TTFT@512 |
|---|---:|---:|---:|
| TP1(单卡 3080) | 970.51 | **32.02** | 535.6 ms |
| TP2(3080 对) | 1485.46 | **42.11** | 356.8 ms |
| **比值** | **1.531×** | **1.315×** | 0.666× |

#### 58.3 权重字节账(为什么"每卡该减半"只减到 0.576×)

`models/cyankiwi/Qwen3.8-27B-AWQ-INT4`(64 层;hidden 5120;inter 17408;vocab 248320;48 线性注意力 + 16 全注意力):

| 组 | 字节 | TP2 切法 |
|---|---:|---|
| layer.mlp_or_gdn | 9.215 GiB | 切 → 4.61 GiB/卡 |
| layer.attn | 3.970 GiB | 切 → 1.99 GiB/卡 |
| **lm_head**(BF16,未量化) | **2.368 GiB** | **复制(不切)**⇒ 每卡全量 |
| embed(BF16) | 2.368 GiB | 复制(**decode 只读 1 行** ⇒ 不算流量)|
| vision / mtp | 0.858 / 0.791 GiB | 文本 decode 不走 |

```
TP1 每 token 流量 = 9.215+3.970+2.368 = 15.55 GiB
TP2 每 token 每卡 = (9.215+3.970)/2 + 2.368 = 8.96 GiB   ⇒ 比值 0.576(不是 0.5)
```

#### 58.4 ★★★★ 仪器:`nsys --cuda-graph-trace=node`(图模式逐核 + 分卡;已入树)

```bash
tools/nsys-decode-account.sh tp2 tp27b testdata/args/bench-27b-tp2.toml   # 起服+打点+出报告
python3 tools/nsys_decode_account.py tp1 tp2                              # 两臂并排单步表
```

**量法**(判据口径):按内核开始时间排序,间隙 >200µs 切段 ⇒ 一段 = 一次图回放 = 一 token 步;
跳预热两段,取**每核中位数**;`设备时间` = 该步所有核 (end−start) 之和。
**为什么不能用 `OWL_GPU_PROF`**:那是逐发射同步,捕获期跳过 ⇒ **图模式归因不到核**;eager 档会注入 host 噪声。

**两条结构读数(重要)**

| 事实 | 值 |
|---|---|
| 每步核数 | TP1 **836** · TP2 **1092**(+256 = 128×AR 两核)|
| 单步设备 span | TP1 28.37ms · TP2 **21.00ms**;且 **busy = span(100% 满,零空隙)** |
| 两 rank 差 | dev2 21.00ms vs dev3 21.00ms(逐段差 **<0.05ms**)⇒ **无拖后腿 rank** |
| 段间空隙(host 面) | TP1 1.61ms · TP2 **2.38ms** |

#### 58.5 ★★★★★ 两臂逐核对表(单步设备 µs,中位)

| 核 | TP1 | TP2 | 判语 |
|---|---:|---:|---|
| `marlin::Marlin<…>`(量化权重;320+32 核/步) | **21515** | **11473** | ✓ **正好腰斩 —— TP 的全部收益就在这一项** |
| `internal::gemvx`(**lm_head BF16**;1 核/步)| 3506 | **3506** | ✗ **复制** ⇒ 每卡都读全词表 |
| `ar_stage` + `ar_reduce`(128+128 核) | — | **2860** | ✗ TP 的税(22.3µs/AR 对) |
| `fused_add_rmsnorm`(64) | 877 | 871 | ✗ 不动 |
| `rmsnorm`(64) | 684 | 676 | ✗ 不动 |
| `gdn_decode_step_v2`(64) | 496 | 349 | 小幅 |
| `paged_attention_v2_*`(16+16) | 422 | 419 | ✗ 不动 |
| `owl_embed_f16`(1 核) | 288 | 288 | ✗ **单线程核**(见 58.7 刀④)|
| `gemv_dual`(48) | 236 | 226 | |
| 其余小核(silu/gdn_conv/add/qknorm/argmax/narrow/sigmoid) | 275 | 272 | |
| **设备合计** | **28413** | **21034** | |

#### 58.6 ★★★★ 账闭合:decode 只买到 1.315× 的三条原因

```
TP2 步 = **11.47**(减半的 marlin) + **8.8ms 完全不缩的固定项**(31%) + **2.86**(AR,12%)
         + host 2.38  ⇒ 23.4ms  ✓ 与 std 台 ITL 23.75ms 闭合
不缩项 8.8ms 明细:lm_head 3.51 + norm 1.55 + embed 0.29 + attention/GDN/其它小核 3.45
```

**⇒ 定谳:TP 把"能切的那一半"切了(21.5→11.5),但步里另有 8.8ms 与 rank 数无关,
再加 2.86ms 的 AR 税 ⇒ 净收益只剩 1.315×。**

#### 58.7 ★★★★ 五把刀(每把都有实测分母;按值排序)

| # | 刀 | 现值 | 目标 | 省/步 | decode |
|---|---|---:|---:|---:|---:|
| ① | **lm_head 按词表切**(`TpSplit::ColParallel`;vocab 分半)| 3.51ms | 1.75ms | **−1.76ms** | **+8%** |
| ② | AR 同步面瘦身(§56.5 的账)| 2.86ms | ~1.4ms | −1.4ms | +6% |
| ③ | norm 核 `gridX=1` → 多块/向量化(128 核/步)| 1.55ms | ~0.5ms | −1.05ms | +4.4% |
| ④ | `owl_embed_f16` **gridX=1 blk=1** → 并行(1 核/步)| 288µs | ~5µs | −0.28ms | +1.2% |
| ⑤ | `owl_argmax` 并行(1 核/步)| 27µs | ~5µs | −0.02ms | +0.1% |
| ⑥ | host 尾巴(未定谳;fill+launch 仅 0.2ms)| 2.1ms | ? | ? | ? |

**⇒ ①–④ 合计 −4.5ms ⇒ 23.4 → 18.9ms ⇒ decode 42.1 → ~53 t/s(+25%)**;
⚠️ ③④ 对 **TP1 同样有效**(−1.33ms)⇒ TP2/TP1 比值主要由 ① 改善(1.315 → ~1.42),
而 ①②③④ 一起上之后 TP2/TP1 ≈ 1.65×。

**① 的数值性质(为什么它排第一)**:logits 按词表行独立计算 ⇒ **切词表是数值中性的**
(无归约、无求和);代价是**两半 logits 的合并** —— greedy 档只需 8B(argmax 值+序号)交换,
采样档需 allgather(1MB ≈ 0.08ms,仍远小于 1.76ms)。工程面已有现成形状:
`TpSplit::ColParallel` + 装载侧的行切规则(q/k/v/gate/up 同款)。
**副产品**:lm_head 每卡省 **1.18 GiB** —— 20G 卡的预算也松一口气。

#### 58.8 host 面的分相(step_profile;`[probes] step_profile = true`)

| 段 | 值 | 说明 |
|---|---:|---|
| `fill+launch` | 0.17–0.34 ms | host 侧的落参 + 图发射 |
| `token-dtoh` | **21.2–21.7 ms** | 设备 argmax 档只回读 4B —— **这段就是"等图跑完"**,不是拷贝 |
| 尾巴(到下一步 fill+launch) | **~2.1 ms** | 采样收口 + pump/HTTP 回合;`[sched] avg=0.00ms`(调度无罪)|

⇒ 与 nsys 的段间空隙 2.38ms 互相印证。**6 号刀的目标就是这 2.1ms**;
最省事的候选 = 把"整图回放 × N"收进图内(设备侧 token 回环)或把批尾处理从关键路径上摘掉。

#### 58.9 纪律沉淀(接 ⑭–㉘)

**㉙ 分母要回到原始 CSV。** 本轮"decode 零提升"是一个**跨节转抄的错标签**(把 TP2-v1 的 42.14
当成了 TP1)—— 它一路传进 §50.1 的推论("差的全部是 AR 的 α"),而那个推论是**错的**
(真实缺口 15.5ms,AR 只占 2.86ms)。**任何"TP 没收益/某结构无效"的结论,先回到原始 CSV 核分母。**
**㉚ 逐核账要按"是否随 rank 数缩小"分类。** 分片工程里,"权重减半"只是账的一半;
先问"这一步还有哪些项**不缩**"(复制项/单核小核/协议/固定发射)。
**㉛ 图模式的逐核归因用 nsys `--cuda-graph-trace=node`**(不改代码、不改档位、可分卡);
`OWL_GPU_PROF` 只在 eager 档有意义 —— 两者别混用当同一把尺。

#### 58.10 ★★★★★ 刀③④**落地实录**(norm / embed 小核延迟治理;2026-10-19)

##### 58.10.1 改了什么(三处核 + 两处几何 + 一个新判据台)

| 面 | 旧 | 新 | 改动要点 |
|---|---|---|---|
| `cu/ln/rmsnorm.cu` | 逐 `__half`(2B)步进 + 8 级 `__syncthreads` 树 + `__shared__ float smem[256]` | **16B 向量装载**(uint4;VEC = 16/sizeof(T)) + **warp shuffle-xor + 每 warp 一槽** + **动态 smem**(`extern __shared__`) | 块上限从静态数组解绑;归约序改变(见 58.10.3) |
| `cu/ln/fused_add_rmsnorm.cu` | 同上(f16/bf16 两实例各自展开) | 同上 + 向量化的原地写/写出 | 代数不变:**未舍入**的 `v = m+r` 进归约(与被替核逐式一致),舍入后落 residual |
| `cu/elems/embed_f32.cu` | 逐元素 2B 拷贝 | **16B 向量拷贝**(VEC = 16/sizeof(T)) | 纯查表 ⇒ **逐位不变** |
| `driver::ops::rmsnorm` / `driver::ln::fused_add_rmsnorm` | block 256、smem 1KB | **block 512**、`smem = row_reduce_smem(block)` = ceil(block/32)·4 | 新增 `ROW_REDUCE_BLOCK` / `row_reduce_smem()` = **契约单源**;几何单测同步加断言 |
| `driver::elems::embed` | **block (1,1,1)**(一 token **一线程**) | block 512 | 病根:blockIdx 维铺开了、threadIdx 维没铺 |
| **新判据台** `crates/kernels/tests/ln_elems_bench.rs` | — | — | 四核 × 多几何:**数值对拍(独立 host 参考)+ 时延(M=1/M=32 双档)**;走生产同一条发射路(`Exec::load_aot` + driver 几何) |

##### 58.10.2 读数一:单核(判据台;单卡 3080;µs/发)

| 核 | block | 旧 | 新 | 倍数 |
|---|---|---:|---:|---|
| `owl_rmsnorm_f16` | 256→**512** | 9.35 | **3.18** | **2.9×** |
| `owl_fused_add_rmsnorm_f16` | 256→**512** | 9.35 | **3.36** | **2.8×** |
| `owl_embed_f16` | **1**→512 | **264** | **2.62** | **101×** |
| `owl_argmax_f32idx_f16` | 256(未动) | 31.6 | 31.6 | — |

几何扫描(为什么取 512):norm 256/512/1024 = 3.44/3.18/2.87µs(块越大越好但边际递减;
512 与 1024 差 0.3µs,而 512 对"行数很多"的 prefill 更省块);embed 1/64/128/256/512/1024 =
67.6/3.55/2.71/2.61/2.62/2.61µs(64 线程起已近饱和)。

##### 58.10.3 数值面(**归约序变了,必须逐条交代**)

| 判据 | 结果 |
|---|---|
| `owl_rmsnorm_f16` / `fused_add` vs **独立 host 参考**(f64 累加) | **3.0e-4 / 2.8e-4**(容差 1e-2)|
| `owl_rmsnorm_bf16` / `fused_add_bf16` | **2.3e-3 / 2.9e-3**(bf16 尾数 8 位 ⇒ 比 f16 大一个量级,仍 ≪1e-2)|
| `fused_add` 的 **residual 原地写** | **逐位相等**(f16 与 bf16 两档)|
| `owl_embed_f16` | **逐位相等**(纯查表)|

⇒ 改动只动**归约序**(线程内序 + warp 树序),不动逐元素代数;f16 域差 ~3e-4(= 归约序的量级)。

##### 58.10.4 读数二:**生产图内**(nsys 逐核;TP2 27B;单步 µs)

| 核 | 旧 | 新 | Δ |
|---|---:|---:|---|
| `owl_fused_add_rmsnorm_f16`(64 核) | 871.2 | **191.3** | −78%(13.6 → **2.99µs/核**)|
| `owl_rmsnorm_f16`(64 核) | 675.7 | **172.3** | −74%(10.6 → **2.69µs/核**)|
| `owl_embed_f16`(1 核) | 287.9 | **2.1** | −99.3% |
| **设备合计** | **21034** | **19622** | **−1412µs = −6.7%** |
| 单步 span | 21.00 ms | **19.63 ms** | 两卡差仍 <0.05ms(无拖后腿 rank)|

⇒ **与预测的 −1.42ms 逐位吻合**;图内单核(~2.7–3.0µs)比裸台(~3.2–3.4µs)还快(L2 热)。

##### 58.10.5 读数三:生产 std A/B(**专属端口 8141**;同二进制、同卡对 2/3、同会话、两臂各 15/15)

| 臂 | prefill | decode | 旧 decode |
|---|---:|---:|---:|
| TP1(单卡 3080) | 973.67(旧 970.51) | **33.57** | 32.02 ⇒ **+4.8%** |
| TP2(3080 对) | 1489.53(旧 1485.46) | **45.07** | 42.11 ⇒ **+7.0%** |
| **比值** | 1.530×(旧 1.531×) | **1.342×** | 旧 1.315× |

⇒ decode 增量与账一致(TP2: −1.41ms/19.6ms…23.4ms);prefill 微动(+0.2%,在噪声内)。

##### 58.10.6 门(全绿;改动后复跑)

`tp_g8_gate`(**L1 = 0 项不同** / L2 互含 top-5 / L3 记账)· `tp_determinism_gate`(16 arm 唯一)·
`tp_transport_bits_gate`(末步 248320 项零差)· `tp_assemble_gate`(账 66MiB 闭合)·
`driver` 几何单测 8/8(**新增 smem 契约断言**:`smem ≥ ceil(block/32)·4`)· `ln_elems_bench` 数值面。

##### 58.10.7 现场教训(两条,已焊进工具/文档)

**㉜ 同机多会话**共绑**同一端口** ⇒ **互相击杀**(本轮立案)。症状极像"引擎静默崩":
客户端 `ConnectError: All connection attempts failed` + 服务端日志**零错误** + 进程已消失
+ 设备日志无 OOM/Xid/core。**根因**:两边都按"占用该端口者"清场(一边 `kill $(lsof -ti:8137)`,
另一边 `ss -ltnp` 按端口收)——而**双方默认都绑 8137**(flavor 里写死)。
⇒ 处置:`tools/tp-ab-std.sh` 加**端口 preflight**(只收自己 worktree 的实例;**别人的响亮拒**,
不代为清理)+ 文件头写明"并行跑必须换专属 PORT **且运行档里声明同一 bind**"。
**㉝ 判据自己的 dtype/期望值也要过一遍**:本轮两次假红——① argmax 的"期望索引"不能用
手填的(`lcg_f16` 会产出**多个恰好相等**的最大值,而核的平局律 = 取**最小索引**;
正解是**从未上传的输入算期望**);② bf16 臂用 f16 解码器读位型(误差 0.5~0.8 的假红)。
**⇒ 律㉗(测速台不校验数值 = 永远发现不了"快但错")要加一条:判据本身先自证。**

##### 58.10.8 顺带:全特性 lib 门**自上一轮起就是红的**(两处漏跟,已修)

跑门时发现 `cargo test -p owl-kernels --lib`(**全特性**跑法)红两条 —— **与本轮核改动无关**,
是树里两处"声明/锁表漏跟",正是那两张锁存在的理由:

| 漏跟 | 出处 | 修法 |
|---|---|---|
| push 档的 **`data` 槽**(4 处:stage 的 `PARSE_SLOTS` + reduce 的 `PARSE_SLOTS_REDUCE` + `ops` 的 EXPECT 锁表) | 上一轮 push 落地(§36.7)只改了核签名/声明表/`parse`/`to_launch`,`PARSE_SLOTS*` 常量与 EXPECT 表没跟 | 核签名为真源(7 参)⇒ 补齐三处表项 |
| **`flashinfer_attn_ragged_f16`** 未进 client 锁的 `cases`/`want` 两张清单 | `c5297d4 feat: add vision tower`(2026-10-15)| client 面本来就有(`ATTN_RAGGED_SLOTS`)⇒ 补进两张清单 |

⇒ 复跑:`owl-kernels --lib` 默认 **75 通过** / 全特性 **88 通过** / `owl-models --lib` **206 通过**;
`tp_g8_gate` L1 = 0 项不同 ✓(重建 release 后复验)。
**纪律(补 ㉝)**:改声明/槽序类改动时,**两张锁表(`client` 的 cases+want、`ops` 的 EXPECT)是第三、第四处真源** ——
跑门必须用**全特性**跑法(`--features device,cublas,marlin,flashinfer,xqa`),否则漏项会被 feature 门遮住。

---

### 59. 历史归并:本分支 44 刀 → 10 刀(2026-10-19)

**用户口径**:"这个分支上历史提交太多了,都是些零碎的,能 squash 的就把它 squash 掉。"

**做法**(可复现、可回退):
- 备份分支 `backup/pre-squash-2026-10-19-tp2`(归并前 HEAD = 9f95b80)—— 旧 44 刀全部保活,
  本文档里对旧号(`dd1cf45`/`cc0d35a`/`76e26d3`)的引用仍可 `git show` 查到;
- 归并按**阶段边界**切 10 组:每组取"组内首个提交的完整信息"+ 追加 `本刀含 N 次追加落地`
  的主题索引 ⇒ **信息不丢,顶层变干净**;
- 关键不变量:**归并前后树哈希逐字节相同**(`b92fb32f…`)+ `git diff` 归并前后 = 空。

**归并后的 10 刀**(自基线 `fc2737db` 起):

| # | 提交 | 主题 |
|---|---|---|
| 1 | `f1545b2` | docs(tp): TP 三卷(综述/预研/施工设计)|
| 2 | `b1e46dc` | feat(tp-s1): peer 执行臂(VMM 授予 + G4/G5 门;含 退役 owl-backend-utils)|
| 3 | `7693b06` | feat(tp-s2): allreduce 两核 + 声明/家族/parse + G1/G2/G3 门 |
| 4 | `d00c2e8` | feat(tp-s4): 切分计划 → 分片装载 → AR 注入 → 形状感知 → 双 rank 装配 → 启动档(14 刀)|
| 5 | `dcc2db4` | fix(models): 分片装载正确性(★DeviceRearrange 双切 —— 27B TP 乱码定谳)+ §22–§24 结案 |
| 6 | `264e9b0` | perf(comm): AR 通道提速(核 MLP 化 + 多 block + 线程数 + 通道审计;prefill 1353 → 1477,12 刀)|
| 7 | `52aa7ce` | test(comm)+fix(gate): 压力门 + 判据收敛"承诺得住" + CE 刀立案(D2D 白名单放开)(7 刀)|
| 8 | `591402d` | feat(tp): TP2 主干落地(§34–§37)|
| 9 | `b39d0b7` | perf(kernels): decode 小核延迟治理(§58 刀③④)|
| 10 | `e096282` | docs(tp): §10 收口(拆卷 + 当前形态)|

⚠️ **本卷已入库**(2026-10-19 起:`docs/arch/TP2-实录卷.md`,与施工设计/路线图同行)—— 逐刀细账、
被推翻的中间结论、纪律沉淀都在本文件;入库面 = 本卷 + `docs/arch/TP2-施工设计.md`(§0–§9 + §10)
+ `docs/arch/TP2-下一步路线图.md`(执行面);`roadmap.local/` 仍作本地草稿区。

---

### 60. ★★★★★ 映射路径的成本账:VMM 显式授予 vs runtime(2026-10-19)

**用户口径**:"可不可以绕开 VMM 这个 API,你看一下这个 API 它的一个性能如何。"

#### 60.1 先分清两个面(混起来会把"值多少"算错一个量级)

| 面 | 是什么 | 谁付 | 本节读数 |
|---|---|---|---|
| **API 成本** | `cuMemAddressReserve/Create/Map/SetAccess/Export/Import` 那几发 | **装配期一次性**(staging 常驻、图前授予 ⇒ 稳态零调用)| **1.47–1.89 ms**(两卡两块 64MiB)|
| **访问路税** | 映射出来的内存**被访问**时与普通 `cudaMalloc` 内存的差 | **每次 AR、每个字节** | **读腿 0% → 24%**(随载荷);**写腿 ~1%** |

⇒ **"绕开 VMM"若为省 API 成本 = 走错门**(那 1.8ms 在启动时,与 decode 无关);
它唯一可能值的地方是**访问路税**。

#### 60.2 仪器:`tools/vmm-mapping-tax-probe.cu`(新;与 `vmm-peer-probe.cu` 分工)

`vmm-peer-probe.cu`(2026-10-18)= 单点定性;本探针 = **记账面**:① API 逐步成本;
② **尺寸阶梯**(10KB/128KB/1.25MiB/5MiB/32MiB)判"税是固定还是比例";③ **写腿**
(push 档前置 —— 现役生产就是写腿);④ **导入访问位 A/B**(READ vs RW —— v1 档 vs push 档);
⑤ **生产 AR 形状**两条协议各跑一遍。
跑法:`nvcc -O3 -arch=sm_86 -o /tmp/vmm-tax tools/vmm-mapping-tax-probe.cu -lcuda && /tmp/vmm-tax 2 3`
(⚠️ 跑前查卡;它的 peer 导入**占 BAR1 窗口**,探针自己负责 unmap —— 见 60.5 的坑)

#### 60.3 ★★ 表 A:API 一次性成本(64MiB/两卡两块;3 次中位)

| 步 | µs | 注 |
|---|---:|---|
| `cuMemAddressReserve` | 7 | |
| `cuMemCreate` | 65–68 | **一整个块一发**(生产同款:粒度只用于**对齐**)|
| `cuMemMap` | 9 | |
| `cuMemSetAccess`(self RW) | 154–188 | |
| `cuMemExportToShareableHandle` | 219–227 | |
| `cuMemImportFromShareableHandle` | 150–158 | |
| `cuMemAddressReserve`(import) | 9–11 | |
| `cuMemMap`(import) | 9–12 | |
| **`cuMemSetAccess`(peer,RW)** | **692–1086** | **一发占 58%** —— 正是驱动攻坚案(`setAccess205`)里那一发 |
| 释放(unmap×4 + release×2 + free×2) | 143–458 | |
| **VMM 合计** | **1474–1886** | 装配期一次性 |
| **runtime 合计**(`cudaMalloc`×2 + `cudaDeviceEnablePeerAccess`×2) | **1727–1730** | 与 VMM 同量级 |

⇒ **两条路的一次性成本一样(~1.8ms)**,"VMM 更贵"不成立。

#### 60.4 ★★★★★ 表 B:访问税(3080 对 ord2/3;同会话、同核、同载荷)

| 载荷 | 读 1way | 读+fence | 读 2way | 写 1way | 写 2way | **push(写腿)** | VMM 读:READ 导入 | VMM 读:RW 导入 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 KB | +0.1% | −0.1% | −4.8% | −0.2% | −0.4% | **+0.1%** | −0.5% | +0.1% |
| 128 KB | +1.2% | +1.0% | +3.9% | +0.4% | +0.7% | **+0.5%** | +1.1% | +1.2% |
| 1.25 MiB | +3.2% | +3.7% | +5.6% | +1.0% | +1.1% | **+1.0%** | +2.8% | +3.2% |
| **5 MiB** | **+20.1%** | +19.9% | +11.2% | **+1.2%** | +1.2% | **+1.1%** | **+13.9%** | **+20.1%** |
| **32 MiB** | **+23.7%** | +21.5% | +11.0% | **+1.1%** | +1.2% | **+1.1%** | **+20.7%** | **+23.7%** |

**四条定谳:**
1. **税在"读腿",而且是比例项**:10KB ~0% → 1.25MiB 3% → 5MiB 14–20% → 32MiB 21–24%。
   (与"PNG 页表/TLB 条目按量放大"的形态一致 —— 小包无感、大包痛。)
2. **写腿免疫**:全档 **+0.4~1.2%**(与 §36.x/§50.8 的"写腿 1.5%"同量级 ✓)。
3. **fence 不改变税**(5MiB:+19.9% vs +20.1%;32MiB:+21.5% vs +23.7%)⇒ 税不是"栅栏交互",是**访问路径本身**。
4. **导入访问位有效**:同载荷下 **READ 导入比 RW 导入快 3–6 个百分点**
   (5MiB:462 vs 487µs;32MiB:3220 vs 3300µs)⇒ 只读路用 READ 授;RW 别名只用于写路时不吃这笔。

#### 60.5 ★★★★ 结论(回答问题本身)

```
绕开 VMM 值多少?(以现役生产形状为分母)
  decode(AR 10.24KB,写腿):税 0.1%  ⇒ AR 占 decode 12% ⇒ **≈0.01%** —— 零
  prefill(AR 5.24MB,写腿):税 1.1%  ⇒ AR 占 prefill 19% ⇒ **≈0.2%**
⇒ **不值得动**(代价:enable 语义 + BAR1 窗口纪律 + 两套代码路径)。
```

**而它的真正价值在于回答"为什么 v1 慢、push 快"**:§37 量的"push 比 v1(读腿)快 16.2%"
与本节量的"读腿映射税 14–20%"**是同一件事的两种量法** ⇒ **push 的收益里有相当一块
不是"协议更优",而是"把跨卡读换成了跨卡写"** ⇒ 推论(给将来):
1. **协议设计优先"写腿"**:凡是"跨卡读"的形态(allgather、1-stage 的 `peer_data` 读、
   任何 peer 别名读),先问能不能改成"写对端";收益比换映射路径大**一个量级**(20% vs 1%);
2. 必须保留跨卡读时,**先跑本探针量税**,再决定是否换 RT 映射(runtime 路在 3080 对
   可用:enable + `cudaMalloc`,探针里 32MiB 档跑到 12.9 GB/s ✓;约束是 peer 段必须
   装进 BAR1 窗口 = 256MiB−64MiB,legacy 大段(>=192MiB)装不进 —— llama.cpp 857MB 案同源);
3. **§50.8 的"VMM 读税 3.8–5.9%"标注待复核**:本探针同会话读 READ 导入 = 13.9–20.7%。
   差异**未定谳**(候选:块大小/顺序/机器状态);**以本节阶梯为准**(它同一会话内同时含
   RT / READ / RW / 写 四组对照,自洽)。

#### 60.6 探针开发中踩到的两个坑(都值得留档)

① **驱动调用与事件都绑"当前上下文"**:`cuMemCreate/Map/Import` 落在哪个卡的地址空间,
   取决于调用时**当前设备** —— 探针里漏了 `cudaSetDevice(ord)` ⇒ 块建到别的卡上、
   事件跨上下文 record(报 `invalid resource handle`)。**教训**:凡是"多卡 + 每卡一块"
   的探针,助手函数第一行就该钉设备。
② **peer 导入占的是导入方的 BAR1 窗口**:探针第一版只 `unmap` 了**属主块**,漏了
   **导入别名** ⇒ 每轮泄 64MiB,3080 的 BAR1 只有 256MiB,**第 3 轮 import 直接
   `out of memory`**。⇒ 与引擎纪律对上:`PeerRelease` 必须同时收"分配块 + 别名块"
   (施工设计 §7.1 D2 当初就是这么定的;本次是它的现场复现)。


---

### 59. ★★★★★ P1.2 的靶被实测改写 + 1-stage 竞态**根因定位**(2026-10-19)

#### 59.1 ★★★ 定谳:候选刀 #2/#3(砍跨卡往返)**无效** —— α 全在"核的固定成本"

`ar_channel_bench` 图内臂(`OWL_ARB_GRAPH=1 N=64`,`OWL_ARB_SIZES=10240`,ord 2/3):

| 读数 | 值 |
|---|---|
| 图内 AR(全协议,10KB) | **22.13 µs/次** |
| 图内 AR(**旁路协议**) | 22.67 µs/次 |
| ⇒ **协议钱**(credit + rendezvous) | **−0.53 µs(−2.4%)** |
| **单核(1 node)/次** | **11.13 µs** |
| **第二个核的边际成本** | **11.00 µs** |

**⇒ §50.6 的猜想"跨卡往返 4–6µs"被否**;α(19.8µs)的**全部** = 两个核各自的 ~11µs
"落地/排空"固定成本(搬运只 0.8µs)。
**⇒ 靶 = 少发核(融合核),不是砍同步往返。**

#### 59.2 ★★★★★ 1-stage 竞态**根因**(复现 + 五步协议对读)

**复现**(`tp_g8_gate`,1-stage 档,0.8B,3 跑):2 绿 1 红
(红:`L1 rank0 vs rank1 = 242374 项不同, max|Δ| = 0.324219`)⇒ 间歇可复现。

**根因**(读 `cu/comm/ar_reduce.cu` 的 1-stage 头注五步 + 核体):

| 步 | 动作 | 问题 |
|---|---|---|
| ① | 发布 `own.produced++`("本槽就绪",对端据此来读) | — |
| ② | 等 `peer.produced > own.consumed` | — |
| ③ | **`out[i] += peer[i]`(原地加自己的对称槽)** | ★ **对端在①之后就来读我的槽,而我正在原地改它** ⇒ 读到**半加状态** |
| ④ | 发布 `own.consumed++` | — |
| ⑤ | credit(保护下一个 GEMM 覆盖槽) | — |

**⇒ 这是"协议语义冲突"而非时序缝隙**:同一个块被两种语义共用 ——
"对端要读**只含本端**的值"与"下游要读**加完**的值"。
**⇒ 对照组(vLLM `cross_device_reduce_1stage`):`result` 是**独立输出块**
(`result[idx] = Σ dp.ptrs[r][idx]`),输入对称缓冲**只读** ⇒ 结构性无此竞态。**

**修法(待做,三步)**:
1. 核:`out` 改为**独立输出块**(`out[i] = own[i] + peer[i]`,不原地);
2. 接线:`symmetric_out_slot` 的语义改 —— GEMM 写**对称槽**(输入),融合核写**普通输出块**
   (下游读它);显存 +1 份 payload(与 v1 的私有 payload 同量级);
3. 门:`tp_g8_gate` 的 1-stage 档**连跑 ≥5 次全绿** + `ar_transport_gate` + 压力门。

**收益**:AR 22.13 → ~11µs/次 ⇒ ×128 = **1.4 ms/步** ⇒ **decode +7%**(= 路线图 P1.2 的目标)。


---

### 60. ★★★★★ §59 的修法做完了,但**它修的那条路在生产形态下从未生效**(2026-10-19)

#### 60.1 竞态修复(§59.2 的三步)—— 已完成且门全绿

- **核**(`cu/comm/ar_reduce.cu`):1-stage 融合核加 `src` 槽,**输出改独立块**
  (`out[i] = src[i] + peer[i]`,新写 `owl_ar_add_vec<T,V>` 三操作数向量加法);
  对称槽全程只读 ⇒ §59.2 的语义冲突消失;
- **声明面四处同步**:`ops/comm.rs`(4 个 spec 的 PARAMS)+`ops/mod.rs` 的 `EXPECT`
  槽序锁 + `client/mod.rs` 两张锁表 + `client/comm.rs` 的 `Ar1StageCall`
  (加 `src` 字段/`new`/`parse`/`to_launch`)+ `server/comm.rs` 的 `launch_ar_1stage`
  (参数表 + `LaunchVal::Ptr(src)`)—— **改漏一处即 SIGSEGV**(本轮真踩:发射面漏跟
  ⇒ 参数错位 ⇒ 段错误;教训 = 参数表是**五处**同改,不是四处);
- **接线**(`eval.rs`):arena 句柄**先快照**(纯克隆,借用不跨 `&mut ctx` 分配)⇒
  融合核写 `alloc_out` 的独立块,下游指向它;
- **门**(`tp_g8_gate`,1-stage 档,**连跑 5 次**):**5/5 全绿**
  (`v1=0` 全走融合核、`L1 max|Δ| = 0.000000`)—— 修复前的"3 跑 1 红"消失 ✓

#### 60.2 ★★★★★ 但**1-stage 在生产形态(27B)从未生效** —— 判据错位

**证据链**(三段独立):
1. **nsys**(27B TP2,1-stage 档):AR 核只有 `owl_comm_ar_stage_f16`/`reduce_f16`
   —— **零个 `owl_comm_ar_1stage_*`**;
2. **启动日志**:`[tp] 1-stage 启用判据:env=Some("1") · push=false ⇒ arena 装配`
   —— **arena 真装配了**;
3. **AR 注入点的 debug 日志**(768 次,全部):
   ```
   AR 注入 rank=0 node=5379 op=SliceView { offset_elems: 0 } tp_role=UniformRow bytes=10240
   ```
   ⇒ **768/768 都是 `op=SliceView`**,而 `symmetric_out_slot`(唯一能给"对称槽"的地方)
   只在 **`Op::Call`/`Op::Matmul`** 两臂被调用 ⇒ **它的 `is_ar` 判据永远看不到
   `UniformRow`** ⇒ 恒返回 `None` ⇒ 全走 v1 两段。

**根因**:`tp_role` 由层在 `self.forward_inner(xs).with_tp_role(self.tp_role)` 打在
**最外层节点**上 —— 而 marlin 的 GEMM 输出后面还挂着**窄视图**(`SliceView`,裁 padding)
⇒ 标记落在**视图**上,而"落进对称槽"的决策发生在**更早的 Call 臂**。
**两者差一层结构** ⇒ 1-stage 的接线点与判据点**不在同一个节点上**。

⚠️ **方法论**:§57.4 记的"push=192 / v1=2016"里那 192 次 1-stage **不是这条路径**
(它来自 0.8B 的特定形态);27B 生产档是 **0 次**。
⇒ **"核写好了"≠"路径被走到",而"门绿"只证明被判据覆盖的那部分** ——
本轮的门(0.8B)与生产(27B)是**两个形态**,门绿不代表生产生效(§57.5 同族教训的第二次)。

#### 60.3 处置与下一步

| 项 | 结论 |
|---|---|
| 竞态修复本身 | **保留**(代码正确性提升;将来接线通了就能用) |
| P1.2 的预期收益(decode +7%) | **落空** —— 因为 1-stage 根本不触发 |
| 新增诊断面 | 注入点的 `op`/`tp_role` 字段(debug 级)+ 启动期 1-stage 判据行(**保留**) |

**新的可行路线(待裁决)**:**v1 两核融合** —— 在 **AR 注入点**(载荷 = 窄视图,10KB)
用一个核做"写对端槽 → 等对端 → 读对端槽 + 加"(即把 stage+reduce 合成一核)。
- 收益账:省**一个核的启动**(§59.1 单核 11.13µs;融合核估计 ~13–15µs)⇒
  每次省 ~7µs ⇒ ×128 = **0.9ms/步 ⇒ decode +4%**;
- 与 1-stage 的区别:**不需要对称槽**(payload 仍是私有块)⇒ 绕开 60.2 的结构错位;
- 代价:新核 + 协议(核内两阶段对称)+ 门。

---

### 61. ★★★★★ P0.4 第二次对账 + 重基线;顺路挖穿 **fp8 载荷路的三个真 bug**(2026-10-19)

#### 61.1 对账(merge main → feat/w4a8-p1)

| 项 | 值 |
|---|---|
| 合并 | `main@c0e38ae` → `feat/w4a8-p1@2c5fa64` = **`20cd14a`**(0 冲突) |
| 重叠面 | `crates/shared/src/config.rs`(主线删 `prefill_bucket × fi_plan_bake` 互斥门 / 我方加 `tp_ar_f8`)—— 相隔 1100 行,自动合 |
| 带进来的公共收益 | 桶图解禁(桶路接窗外 `fi_plan_sync`)、chunk 1024 + 桶阶梯、**快照税门**、FI 表页链响亮拒、`GRAPH_SLAB_BUDGET` 320→640MiB、verify_v2 显式关 |
| 门 | `cargo check --workspace --all-targets` ✓ · kernels 92 / models 217 / engine 54 / store 3 ✓ · `tp_g8_gate`(L1 0 项不同,max|Δ|=21.92 同包络)· `tp_determinism_gate` · `tp_transport_bits_gate` · `tp_assemble_gate` · `peer_shared_gate` · `ar_transport_gate` · `ar_stress_gate` · `graph_memcpy_gate` · `graph_multistream_gate` · `raw_sync_primitives_gate` **全绿** · 1-stage 档 `tp_g8_gate` **连跑 5/5 绿** |
| 复现环境 | 本机 CUDA 序实测:**ord0=3090Ti@41:00 · ord1=3090Ti@42:00 · ord2=3080@01:00 · ord3=3080@C1:00**(TP2 对 = 2/3;`OWL_TEST_DEVICE` 单卡档 = `01:00.0`) |

#### 61.2 重基线(std 五点三轮,双 3080;同二进制同会话)

| 臂 | prefill 中位 | decode 中位 | 备注 |
|---|---:|---:|---|
| base(w4a16,合并前基线) | 1475.5 | 44.30 | 历史对照 |
| **W4A8 + fp8 AR(v1,合并后)** | **2351.2** | 41.67 | vs 合并前 2302.8(均值)⇒ 合并**持平略好** |
| W4A8 + fp8 AR(v1,**核修后**) | **2441.4** | 41.72 | **+3.8~4.1% prefill,五点一致** |
| W4A8 + push + f16 | 2283.6 | 41.88 | push 未进过 W4A8 基线,本轮补测 |
| W4A8 + push + fp8(核修后) | 2418.6 | 41.80 | **比 v1+f8 低 ~1%** ⇒ push 维持**缺省关** |

#### 61.3 ★★★ 挖穿的三个 bug(判据先行才抓到)

**立案线索**:用户令"先对账 + 重基线",顺手把 §37 的 push 档与 fp8 组合 A/B —— 结果
**push×fp8 的 prefill 塌到 711 t/s**(v1+f8 = 2321),而 decode 完全不变(41.5)。

**(1) fp8 载荷的写/读是"逐字节"粒度**(SASS 级证据)

| 档 | 载荷核 | SASS |
|---|---|---|
| f16 | `owl_ar_copy_bytes`(已知好) | **8 × `STG.E.128`**(16 B) |
| fp8 | `owl_ar_quant_copy_fp8` | **8 × `STG.E.U8`**(1 B!) |
| fp8 reduce | `owl_ar_reduce_f8` | **16 × `LDG.E.U8` + 16 × `STG.E.U16`** |

本地写时 L2 吸收无感;**push 档写对端**时每字节一个 PCIe 事务 ⇒
`ar_stage_f8_f16` 184 µs → **8418 µs(46×)**(nsys 逐核:`/tmp/diag/pf-pushf8`)。
**修**:16 元素/lane(4 lane/组 × 8 组/warp)+ 一次 `uint4` 落 16 B + 一次 `uint4` 读 16 个 q;
reduce 侧同样 3 次 `LDG.E.128` / 2 次 `STG.E.128`(SASS 复验)。**同数学,只换粒度**。

**(2) f8 的"对称律"从未成立**(§47.1 的 L1 判据是**假的**)

- 本端:`dec(q) / inv`(f32 尺度倒数);对端:`dec(q) * scale16`(**f16 落地**尺度)
  ⇒ 两侧是**两个不同的函数** ⇒ `A(a)+P(b) ≠ A(b)+P(a)` ⇒ **两 rank 结果低位不一致**;
- **为什么一直没被发现**:该路此前**零绝对判据** —— 0.8B 门(唯一的 L1 门)载荷 2 KB
  < 256 KB 阈值 ⇒ **f8 分支根本不触发**;27B 真机只有 std 读数,不比对 rank 间逐位;
- **修**:两侧统一到同一个 `Q(x) = float(enc(x × 1/(m/448))) × f16(m/448)`
  ⇒ `out = Q(a) + Q(b)`,f32 加法可交换 ⇒ 两 rank **逐位一致**。

**(3) 两个更狠的"潜伏雷"(新判据当场抓获)**

| 雷 | 现象 | 修 |
|---|---|---|
| **尺度区未对齐** | 载荷元素数 `n` 为**奇数**时尺度区起点 `src+n` 落奇地址 ⇒ `__half` 存取 `CUDA_ERROR_MISALIGNED_ADDRESS`(旧布局同型,只是从没测过奇数 n) | 新线格式宏 `OWL_AR_FP8_SC_OFF(n) = (n+1) & ~1`(**两 .cu 逐字一致**) |
| **发散 warp 的 shuffle 死等** | `ng` 不是"每 warp 组数"整倍时,lane 间 trip count 不同 ⇒ `__shfl_xor_sync` **永久阻塞**(旧映射同型) | 循环上界改走**整 warp 基准组** `g0`;空转 lane 补 0 参与归约、跳内存、shuffle 后 `continue` |

**新增判据(这才是能抓住它们的根因)**:`ar_transport_gate` **G1b f8 载荷腿** ——
精确 host 预言机(e4m3 表 + RNE + 对称律)+ **两 rank 逐位一致** + 两跑确定性 +
**尾组**(n=5121,含奇数 n)与真实尺寸(10 KB / 1.31 MB)。**该路此前零绝对判据**。

#### 61.4 处置与读数

| 项 | 结论 |
|---|---|
| fp8 核向量化 | **保留**(prefill **+3.9%**,五点一致;decode 中性) |
| 对称律修正 | **保留**(正确性;数值与旧路差 ~1 f16 ulp 量级) |
| 尺度区偶对齐 / shuffle 防发散 | **保留**(修雷;`ng` 整倍时逐字节行为不变) |
| push × fp8 | **修好但不用**:比 v1+f8 低 ~1% ⇒ push 维持缺省关(与 §48"写腿不能藏进算"一致) |
| 遗留 | ① 引擎级(27B)f8 的 L1 判据仍缺(0.8B 门不触发 f8)⇒ 需要"27B 小步 L1 档";② f8 reduce 的远端读在 v1 下仍是 1 B 粒度(**但读宽容**,实测已足够) |

**方法论沉淀(接 §57.5/§60.2 的第三次同族教训)**:
**"核写好了 / 门绿了"≠"这条路的语义成立"** —— 判据必须**覆盖被改动的分支**,
否则那条分支就是"没判据的代码"。本轮三个 bug 全部落在**同一个盲区**:f8 分支。

---

### 62. ★★★★★ **TP2 × 投机解码首跑跑通 + M4 达成**(2026-10-19)

#### 62.1 战绩(std 五点三轮;双 3080;depth=4;`max_seq = 8704`)

| 臂 | prefill 中位 | decode 中位 | TTFT | 说明 |
|---|---:|---:|---:|---|
| TP2 非投机(W4A16) | 1475.5 | 44.30 | ~5500ms | 基线 |
| **TP2 + DFlash2 投机** | **1471.4** | **91.53**(峰值 **116.7**)| 2149ms | **decode 2.07×**;prefill 零税;15/15 点全绿 |

逐档 decode:512 → 106.7 · 1024 → 114.9 · 2048 → 64.2 · 4096 → 94.4 · 8192 → **90.3**。

#### 62.1b ★★★ **语义恒等已立**(同轮定谳 + 修复)

**判据**:greedy 投机**必须与非投机逐字一致**(接受律只吃 target 自己的 argmax)。
实测:**单卡档 spec ≡ 非 spec(逐字一致 ✓)** ⇒ 这条判据在本引擎是**硬判据**。

**TP2 档首测:不一致(且输出退化)** ⇒ 定谳 + 修复:

| 步 | 证据 |
|---|---|
| 二分 | `dumb = true`(草稿恒 anchor ⇒ 必走 m=0 回滚重放)**同样退化** ⇒ 与草稿/接受无关,病灶在**回滚路** |
| 根因 | **轮首快照 `capture_spec_snap` 与回滚 `restore_spec_snap` 都只打 rank0** —— 快照缓冲是**每 rank 各自的池** ⇒ peer 恢复的是陈旧态 ⇒ 重放 logits 错 |
| 修复 | 两处都改 `join2` 双 rank(单卡 = 原路)|
| 复验 | `dumb` 档 **逐字一致 ✓**;真投机档 **逐字一致 ✓**;输出连贯 |

⇒ **TP2 × 投机:性能 2.07× 且语义与裸 decode 逐位同源**(最强判据通过)。

#### 62.1c 参数扫描(2026-10-19 同轮;std 五点三轮)

| 档 | decode 中位 | 峰值 | 判语 |
|---|---:|---:|---|
| depth=4 · 草稿池 BF16 | 91.53 | 116.7 | 起点 |
| **depth=7 · 草稿池 BF16** | **104.73** | **153.1** | ★ **最佳**(相对 depth=4 **+14.4%**) |
| depth=7 · 草稿池 fp8 | 93.61 | 140.9 | **更差 −10.6%** ⇒ 草稿池**维持 BF16** |

逐档(depth=7):512 → 126.3 · 1024 → **148.2** · 2048 → 61.3 · 4096 → 109.7 · 8192 → **103.8**。
**语义恒等**:depth=7 档同样 **逐字一致 ✓**。
⇒ **TP2×投机最终:decode 104.73 = 非投机 44.30 的 2.36×**(prefill 持平)。

#### 62.1d ★★ 顺带定谳:**治理器的 `live` 是"两卡之和"**(TP 档的账要按卡摊)

现场:`[vram-gov] … live=23.72G … ⇒ budget=-6.58G cap=0 tok` —— 而单卡总量只有
19.63G ⇒ `live` **不可能是单卡**。根因:`owl_shared::vram::LIVE` 是**进程级全局
计数器**,TP2 下**两卡的分配都加进同一个数**,而 `total`/`per_tok` 是**单卡**口径
⇒ 拿两卡之和比单卡总量 ⇒ 预算负数 ⇒ **池被压到一半**(136 槽 = 4352 tok,而
`max_seq` = 8704)。

**修复**:`PoolPlan += tp_ranks`(boot_rank 从 `TpCtx.plan.world()` 填),
治理器用 `live / tp_ranks`(两 rank 对称 ⇒ 摊分是正确口径)。
**实测**:`KV 池面 136 → 272 槽`(= 满 8704 ctx ✓)、`草稿池 42.5 → 85.0MiB` ✓。
⇒ 这也解释了用户问的"为什么吃紧":**不是真吃紧,是账算错了**。
(真实占用:每张 3080 **13.7G / 20.5G**,**还剩 6.4G**。)

#### 62.2 到达路径(本日五刀;每刀都有现场判据)

| # | 刀 | 现场证据 |
|---|---|---|
| 1 | **草稿双 rank 装载**(spec 分片 + `LoadOpts{shard,slice_id}` + 草稿 rope 同源)| `DFlash2 草稿装载 1.02s`;分片几何 `q_proj 2048 / down_proj 5120×1088 / gate,up 8704` |
| 2 | **草稿池几何声明化**(分片 hkv 8→4)| `草稿池 10240B/tok`(单卡一半;fp8 档 5120) |
| 3 | **k 切键逐段懒物化病态修复**(`ShardedSource` 全键缓存)| 草稿装载门 300s 挂住 → **6.24s 绿** |
| 4 | **草稿 propose 镜像 + peer 树按 peer 的块重建** | 两 rank **交替**注入草稿 AR(80KB/次);否则算子准入拒"块不在本账本" |
| 5 | **verify / fold / `verify_forward` / `dflash_graph_round` 四处镜像 + `write_bt` 双 rank** | `verify 完成:tok=5`、`fold:图入队完 ×2`、端到端出词 |

#### 62.3 教训(本轮第四次同族)

**`join2` 里任何一半的错误都必须立刻上屏**:`join2` 会等另一半到底,被吞的错误
表现为"**挂住**"(peer 的 `owl_embed_f16: 块不在本账本` 被掩了两轮)。
凡并发两半:**两半都校验、两半都出声**。

**"树按谁建"是 TP 的隐式契约**:ops 树挂的是**设备块句柄**(权重/池叶子/页表/
drafter 实例/rope)—— 跨 rank 复用一个树 = 引用别人的账本 ⇒ 必拒。镜像时必须
**每 rank 各建自己的树**(host 输入可共享)。

#### 62.4 遗留(下一步)

1. **质量面(值一半的收益)**:本轮 `m` 多为 0(草稿常全拒),TP1 同档 AL≈2.4
   ⇒ 2.0× 里还有约一倍没吃到。先建 **TP2-spec 恒等门**(TP1+spec vs TP2+spec
   的 token 序列 + 接受账**逐位一致**),再查草稿 KV/位置/掩码面在双 rank 下是否对齐;
2. **`depth` 扫描**(现 4;上限 7)与 `draft_kv_fp8` A/B;
3. 显存:草稿池 + 主池在 20G 卡上紧(`live 23.7~25.0G` 对 `total 19.63G`,
   靠治理器压池)—— 后续要么 fp8 草稿池、要么降 `pool_tokens`。

---

## §63 P1.1 第二步:**lm_head 词表切 —— 引擎接线**(2026-10-19 同轮)

> 承接 §P1.1 详案(路线图)的"待做五步";第一步(`ModelSpec.head_split` /
> `Embedding::new_untied_head` / `logits_vocab()` 唯一真源)已在前轮落地
> (缺省关 ⇒ 零行为变化)。

### 63.1 战绩(std 五点三轮;双 3080;`tp2` 档;w4a16;非投机)

| 臂 | prefill avg | decode avg | 显存 `live`(每卡) |
|---|---:|---:|---:|
| base(`tp_vocab_split = false`)| 1493.8 | **44.55** | 11.42 G |
| **切档(`true`)** | 1486.1 | **48.06** | **10.24 G** |

- **decode 五点一致 +7.9%**(逐点 +6.9% ~ +9.4%;详案预期 +9%);
- prefill **持平**(+0.7%,噪声内);
- **显存 −1.18 GiB/卡**(live 11.42 → 10.24),与详案账(半表 lm_head)逐字节吻合;
- 冒烟:48 tok greedy 输出 **与不切档逐字一致**。

> 数据:`/tmp/owl-vsplit-ab/{base,vsplit}/tp2-std.{csv,json}`。

**附带收益(实测)**:IP 池容量上限(`[vram-gov] cap`)随 live 下降上抬 ——
不切档 cap ≈ 187.5k tok → 切档 **226156 tok**(+20%,同一档 `per_tok = 32768B`)。
256k(ctx 262144)仍需 **fp8 KV 池**(bf16 档差 ~1.7 GB;与 P1.1 无关的另一条路)。

### 63.2 五步落地(每步的判据面)

| # | 步 | 落点(文件级) | 判据 |
|---|---|---|---|
| 1 | **装载面规则传递** | `module.rs`: `Want.rule: Option<WeightRule>` + `LoaderOps::with_rule`;`linear.rs`: `layout` 收口(仅 `tp_role ≠ rule_of(key)` 时打标);`shard.rs`: `build_shard_pass` 优先读 `want.rule` | 新单测 `declared_rule_overrides_keytable`(声明档切/无声明直通) |
| 2 | **配置位 + 装配** | `[runtime] tp_vocab_split`(`shared/config.rs`)/`ServerConfig`/`ModelLoader::vocab_split`(写入 `spec.head_split` 于 `sharded()` **之前**) | **单卡给值 = 响亮拒**;`sharded()` 拒 `tied × head_split` |
| 3 | **槽宽 + stride** | `graph_fleet.rs`:decode/verify 两图 logits 槽与两处设备 argmax stride 改 `logits_vocab()`;`phases.rs`/`prefill_chunk.rs` 同类三处 | 半表档不越界(全量 stride 会跨行读) |
| 4 | **拼接面(唯一出口)** | `running.rs`: `RunningEngine::read_full_logits`(未切档 = 单次读,逐字节等价)+ `concat_head_rows` 纯函数(r0 低半 ++ r1 高半);消费点:decode 采样/argmax/诊断、`harvest_prefill` 末块 token、`spec.rs` verify 行、`draft.rs` eager verify | 形状不符 ⇒ 响亮拒(不静默拼错) |
| 5 | **门** | 新门 `tp_vocab_split_gate.rs`(27B;臂 A 不切 vs 臂 B 切;L1 结构 / L2 数值 / L3 功能);`tp_g8_gate` 加 `OWL_TP_VOCAB_SPLIT` 档(0.8B = tied ⇒ 验**响亮拒**) | 全绿(见 63.3) |

**判据面 = 事实而非意图位**:引擎侧一律用 `head_split_active()`(`logits_vocab() !=
vocab_size()`),不读 `spec.head_split` —— 意图可能被模型结构否定(tied 档),
"事实在模型里"。

### 63.3 验证读数(全绿)

| 面 | 命令/门 | 读数 |
|---|---|---|
| 切档正判据(27B,untied) | `tp_vocab_split_gate`(2/3) | 半表 124160 ×2 = 248320;**逐步 max\|Δ\| = 0.000000**(逐位相同);token 序列逐位相同 |
| tied 拒 | `tp_g8_gate` + `OWL_TP_VOCAB_SPLIT=1`(0.8B) | 装载**响亮拒** ✓(tied 表切不了) |
| 常规回归 | `tp_g8_gate`(不切档) | 全绿(L1 逐位 / L2 top-N 互含) |
| 装载面 | `tp_load_gate_27b` | 4/4 绿 |
| 装配/确定性/位传输 | `tp_assemble_gate` / `tp_determinism_gate` / `tp_transport_bits_gate` | 全绿 |
| lib | kernels 92 / models 229 / engine 57 | 全绿 |

### 63.4 ★ 定谳:**切档 × 草稿档(DFlash2/MTP)= 投机失效**(实测回归 104→40 t/s)

**现场**:`tp2-spec-d7-vsplit.toml`(TP2 + depth=7 + 切档)std 五点 ⇒ decode
**40.20**(不切档同档 **104.05**)。日志:`spec-degrade` 31 条 —— **AL 2.4 → 1.4**
⇒ B4 降级态机(盈亏线 1.4-1.9)持续把 spec 拖到裸 decode。

**根因**(定谳):**草稿头共享 target lm_head** ——
- DFlash2:`dflash2.rs` propose 内 `embed.lm_head_matmul(&hidden_head)`
  ⇒ 切档后 logits = `[7, 124160]`;而 `CandidateSelector` 按 **`vocab = 248320`**
  索引(码本 `[vocab, rank]` + topk)⇒ **候选只在前半词表**(rank0 半表,id 恰好
  与全局前段一致 ⇒ AL **降而不零**:1.4 的形态与"半个词表可用"一致);
- MTP:`mtp.rs` 同病(`embed.lm_head_matmul` + `argmax_f32idx(…, vocab, …)`)。

**性质**:不是数值问题(拼接本身已验证逐位一致),是**草稿选择面的词表域**
与 target 头切分**不匹配**。

**处置(本轮)**:boot 期**响亮拒**(`engine.rs::boot_rank`:mode ∈ {DFlash2, Mtp}
且 `logits_vocab() != vocab_size()` ⇒ 拒启,提示"关 tp_vocab_split 或 depth = 0")。
**解**(未施工,记遗留):草稿选择面半表化 —— 两 rank 各出 top-k 候选 + host 合并
(DFlash2 另需码本行偏移)。估:半天,见路线图 §11.6。

### 63.5 教训(两条)

1. **"共享件"是切分的隐藏契约面**:P2.1 的注释写"草稿 embed/lm_head 共享 target
   不持有" —— 当时是**省显存的好事**,P1.1 一来就成了**词表切的爆炸点**。
   凡"共享"必问:它的**维度**会不会被别的档切?(头/表/码本/池几何同族)
2. **意图位 ≠ 实效**:`spec.head_split` 与"本 rank logits 真的半表"是两回事
   (tied 档前者 true 后者 false;写错就是拼接 2× 宽)。引擎侧判据统一用
   `logits_vocab() != vocab_size()`(事实),配置位只进装载面。

### 63.6 遗留(下一步)

1. **草稿选择面半表化**(上文 63.4 的解)—— 这是**生产 spec 档 × 词表切**的
   唯一缺口(非投机档已 +7.9% 可用);
2. `tp_load_gate`(0.8B;CpuFace)**既有失败**(与本轮无关,stash 复现同错):
   `dtoh` 按 F32 读,而块的 dtype = F16 ⇒ "声明 7340032B != 收割 14680064B"。
   修法 = 门按 `LoadEntry.dtype` 读;低频门,另案;
3. 半表档的 greedy 走 host argmax(每步 ~1MB D2H)—— 详案说的"8B 分布式
   argmax"仍是后续优化(值 ≈ 半表档 decode 再 +2% 量级)。

---

## §64 草稿选择面半表化 —— **spec 档 × 词表切解锁**(2026-10-19 深夜)

> 承接 §63.4 的定谳(切档 × 草稿档 = 投机失效)与路线图 §11.7 的立案。

### 64.1 战绩(std 五点三轮;双 3080;`tp2-spec-d7` 档;w4a16;depth=7)

| 臂 | prefill avg | decode avg | 显存 `live`(每卡) | `spec-degrade` |
|---|---:|---:|---:|---:|
| base(`tp_vocab_split = false`)| 1472.8 | **104.05** | 12.61 G | 0 |
| **切档(`true`)** | 1472.0 | **109.74** | **11.43 G** | **0** |

- **decode 五点全正 +5.1% ~ +7.2%(均值 +5.5%)**;
- **显存 −1.18 GiB/卡**(12.61 → 11.43);
- **0 降级**(AL 与不切档同档 —— 草稿候选不再塌);
- 冒烟:48 tok greedy 输出 **与不切档逐字一致**。

> 数据:`/tmp/owl-vsplit-ab/{spec-base,spec-vsplit2}/tp2-std.{csv,json}`。

### 64.2 方案(零新核、零跨 rank 设备设施)

```text
propose 图（图内,每 rank 半表）:
  7 层草稿 → lm_head(半表) → logits [7, half]
  → Topk16Split/Merge（**半表域**,stride = 张量形状）→ topk 单缓冲 224
  → 图输出槽:`topk`（值 112 ‖ 索引 112）+ `sel_proj` [7,256]（草稿 dt）
host（engine）:
  读两 rank → rank1 id += half → merge_topk_two_halves（值降序/平局小 id）
  → 回送设备 → **rank0** select_from_candidates（全表域;码本**不切**）→ drafts
```

**正确性依据**:全局 top-16 ⊆ 两半各自 top-16 的并集(截断性质)。

### 64.3 落地(六件)

| # | 件 | 落点 |
|---|---|---|
| 1 | selector 拆三件 + `forward_hidden` + `propose_block_split`(零回归:smoke 逐字一致)| `models/layers/dflash2.rs` |
| 2 | **图两形态**:切档出 `topk`/`sel_proj`,未切档原路 | `engine/graph_fleet.rs::plan_dflash` |
| 3 | **`read_output_f32` 加 BF16 分支**(见 64.4 坑②)| `engine/graph_plan.rs` |
| 4 | **候选合并纯函数** `merge_topk_two_halves` + 4 单测 | `engine/running.rs` |
| 5 | **收割面** `dflash_select_merged`(读槽→合并→rank0 select→drafts);`dflash_graph_round` 切档分支 | `engine/exec/draft.rs` |
| 6 | 护栏收窄:DFlash2 **图档放行**;MTP / eager 降级档仍拒 | `engine/engine.rs::boot_rank` |

### 64.4 现场两个坑(都会静默错,记录防再踩)

1. **图输出槽 = "整父块"契约**:`sc.output(name, &view)` 输出的是 view 的**整父块**
   (原 `drafts` 注释:"输出槽形状 = 整缓冲 —— dtoh 整父块契约")。首版按 `[7,16]`×2
   声明两槽 ⇒ 装配期**响亮拒**("输出 topk_vals 归约 224 元素 ≠ 声明 112")⇒
   收成单槽 `[2·7·16]`(值区 ‖ 索引区),host 按同序切片。
2. **`read_output_f32` 把 BF16 当 F32 读**(旧实现 `match { F16 => ..., _ => f32 }`)——
   此前无 BF16 输出槽被读 ⇒ 潜伏;**草稿 `sel_proj` 槽是草稿 dt(BF16)** 才暴露。
   补 BF16 分支(`half::bf16`),对既有路径零影响。

### 64.5 验证(全绿)

| 面 | 门/命令 | 读数 |
|---|---|---|
| 组合正确性 | 生产 spec 档 smoke(不切 vs 切)| 48 tok **逐字一致**;0 降级 |
| 非投机切档 | `tp_vocab_split_gate`(27B) | L1/L2/L3 全绿(max\|Δ\|=0) |
| 常规 TP | `tp_g8_gate`(0.8B)+ lib | 全绿(kernels 92 / models 229 / engine 61) |
| 性能 | std 五点三轮 | decode **+5.5%**;显存 **−1.18 GiB/卡** |

### 64.6 残留

1. **MTP 草稿档 × 切档**:仍**响亮拒**(MTP 的 argmax stride/选择面未半表化)——
   若将来要 MTP × 切档,照本案同法(候选合并)做;
2. **DFlash2 eager 降级档 × 切档**:仍拒(`[graph] no_graph = true` 或捕获失败的
   降级路未做合并;生产图档已覆盖);
3. 9B/35B 等其它档的切档收益未量(本案只测 27B)。

---

## §65 v1 两核融合 —— **AR 一核落地**(2026-10-19 深夜)

> 出处:路线图 §11.2 细案;靶来自 §59.1 定谳(两核 22.13µs/次 = 两个核各自的
> 落地/排空,协议钱 ≈ 0)。

### 65.1 战绩

| 档 | 基线(两核) | **融合一核** | 差 |
|---|---:|---:|---:|
| **非投机(std 五点;切档)** | 47.44 | **49.25** | **+3.8%** |
| 投机 spec-d7(切档;两次交替采样) | 108.19 / 109.74 | 103.43 / **109.87** | **方差 ±5% ⇒ 待更多样本** |

- 非投机档与预测(§11.2 的 ~7µs/次 × 128 = 0.9ms/步 ≈ +4%)吻合;
- prefill 持平(1456~1484,噪声内);
- **0.8B 门:融合档 5/5 连跑全绿**(`tp.ar.fused=96 · v1=0 · push=0`;L1 两 rank 逐位
  相同;L2/L3 同 vLLM 判据)。

### 65.2 落地(核 + 五处参数表 + 注入 + 配置)

| # | 件 | 落点 |
|---|---|---|
| 1 | **融合核**(f32/f16/bf16 三实例):一核内两阶段 = 写槽 → 发布 produced → 尾部 rendezvous → 全线程 acquire → 读对端加 → 发布 consumed | `cu/comm/ar_fused.cu`(新;helper 逐字复制自两核,遵"`.cu` 间不 include"纪律) |
| 2 | 两段 `wait_done` 令牌:第一段 `2·prod+1`(credit 放行)/第二段 `2·prod+2`(rendezvous 放行) | 同上(核内) |
| 3 | `TpOpts.ar_fused` + `TpCtx.ar_fused` + `TpStaging.ar_fused` | `engine/tp.rs` / `models/module.rs` |
| 4 | 注入点分派(fused 档走一核;**fp8 腿仍两核**) | `models/interpreters/eval.rs` |
| 5 | `[runtime] tp_ar_fused`(缺省 false) | `shared/config.rs` + `apps/server/src/{config,main}.rs` |
| 6 | flavor `allowed_ops` + 2 核名 | `flavors/tp27b.toml` |
| 7 | 门:`OWL_TP_AR_FUSED=1` 档 + 路径判据(`fused>0 ∧ v1=0 ∧ push=0`) | `tests/tp_g8_gate.rs` |

**与两核的两处结构差**(详见 `ar_fused.cu` 头注):
① `wait_done` 走**两段值**(stage 的 rendezvous 在核尾顺带完成,融合核之后还要全体读 ⇒
必须放行全体);② `produced` **每调用 +1**(两核版 stage+reduce 各 +1 = 2×,靠流序掩盖;
融合核用正确游标 ⇒ credit 判据真正生效)。

### 65.3 两个坑

1. **`tp.ar.v1/push` 计数口径**(已修):该计数原在"数据块选择"段(v1/push 两档共用),
   融合档下也会增 ⇒ 门的路径判据误红。修法 = 计数**移到两核分支内**
   (语义本来就是"走了两核的哪种绑法")。
2. **★ 图捕获 slab 定尺低估 → 已根治(保守版)**(2026-10-19 同夜):融合核把两个节点
   并一 ⇒ 图捕获的**峰值活跃集合法变大**(少一个中间回收点;实测差 **139264B =
   一个 `[8, 8704] f16` 草稿 MLP 块**),而 slab 容量来自 **warmup 计量**(hint =
   12.36MiB)⇒ **生产 spec 档启动被拒**("捕获 slab 耗尽")。
   - 首试 **+1MiB 余量**:**不够**(cap 涨到 14.0M 仍差 139264B —— 且 hint 与本图
     真实峰值差**一个量级**:固定档 32MiB 可跑)⇒ 疑为 **hint 按图分配错位**
     (`take_hint` 一次性消费;图数/顺序一变,某图就拿到别的图的里程计);
   - **现行修法**(`state.rs::graph_begin`):动态定尺取 **`hint × 2`**(保守;仍在
     "图slab 640MiB"治理预算内,图数 × 2·hint)。固定档(`[cuda] capture_slab_mb`)
     不受影响。**验证**:生产 spec+切档+融合档**无固定档启动成功**,std decode
     **110.01**(该档最佳;基线 108~110)✓
   - **遗留**:`slab_hint` 的**按图定量**逻辑要修(现在靠 2× 兜;根治 = 每图各自
     计量/或 carve 失败自动放大重试 —— `hint` 的 take/put 语义见 `shared/slab_hint.rs`
     与 `cuda/server/mod.rs` 的 `meter_add`)。

### 65.4 遗留

1. **slab 定尺根治**:hint(warmup 计量)要含"融合档峰值" —— 或 `carve` 失败时
   **自动放大重试**(现在是一次性定尺,失败即拒启);
2. **fp8 融合腿**(prefill 的 >256KB 载荷)仍两核 —— decode 收益面已全覆盖,prefill 收益小;
3. **spec 档的融合收益**需更多样本(±5% 方差;当前 2+2 样本 ⇒ 持平略好);
4. `tp_vocab_split × tp_ar_fused × spec` 三档组合的**恒等门**(两臂逐字)未固化。

---

## §66 P0.3 host 尾巴**定谳**（2026-10-19 深夜;单刀 ≤1h）

> 目标（路线图 §3 P0.3）:把"host 尾巴 ~2.1ms/步"拆开 —— 上限 = decode **+10%**。

### 66.1 账（非投机档 `tp2-nosplit-fused`;`[probes] step_profile = true`）

| 相位 | 读数 | 判语 |
|---|---:|---|
| `fill+launch`（图入队准备 + launch）| **0.20–0.37 ms** | 小 ✓ |
| `host-sample`（logits dtoh 1MB + host 采样;采样档）| **0.35 ms** | 小 ✓ |
| `token-dtoh`（含**等 GPU 图回放**的同步点）| 18.3–18.8 ms | **不是 host 成本**（≈ 设备时间）|
| **循环头**（`step_need`/`admission_gate`/`pool_housekeeping`/`schedule`）| **≈ 0.000 ms**（新打点;`[sched] loophead=`）| **已排除** ✓ |
| **步间隔（wall 差）** | 20.5 ms | — |
| ⇒ **循环尾**（`host-sample` 打印 → 下次 `fill+launch` 打印）| **≈ 1.4–1.7 ms** | ← **主缺口（未拆）** |
| ⇒ **host 暴露合计** | **≈ 2.0 ms/步（~10%）** | 与 §2.1 旧账吻合 ⇒ **上限 decode +10% 成立** |

### 66.2 已落打点（恒开;纳秒级,零行为影响）

- `running.rs::pump`:`t_loophead`（循环头四件）并入既有 `[sched]` 日志
  ⇒ `[sched] n=… avg=… last=… loophead=…ms` ✓ 当场证明**循环头 ≈ 0**。

### 66.3 结论与下一刀（P2.2 的靶）

- **靶 = 循环尾 ~1.5ms/步**:它在 `pump()` 返回 `TurnEvent` **之后**（actor 事件消费 →
  下一轮 pump 头 → decode 准备）—— 即**在 `apps/server/actor.rs` 的循环里**,
  而**不是**引擎内部（引擎侧的 sink 侧全加起来 < 0.6ms）。
- **下一刀（P2.2 host 尾巴治理）**:在 actor 循环 + `pump` 出口加打点,拆这 1.5ms
  （候选:事件通道/锁、`/debug` 统计、日志格式化、`turns` 账务、下一轮 `step_need` 之前
  的会话/KV 事、TTFT 记账…）。**预期上限 decode +7%**（扣掉已计相位）。

#### 66.4 二次细分（P2.2 的**精确靶**;本轮收尾）

加 `exec-main` 打点（`execute_decode` 321→637）后:

| 段 | 读数 | 说明 |
|---|---:|---|
| `321→470`（decode 准备 + fill/launch）| **0.32 ms** | ✓ 小 |
| `470→635`（图回放 + dtoh + 采样）| **18.58 ms** | ≈ 设备时间 |
| `exec-main`（321→637）| **18.90 ms** | ✓ |
| 步间隔 | 20.5 ms | — |
| ⇒ **函数外**（637 → 下一步 321）| **≈ 1.55 ms** | ← **靶** |

`pump` 的 dispatch 本身极薄（`set_phase` + timer + 调用 ⇒ 纳秒级）⇒ **1.55ms
在「`pump` 返回 → actor 事件消费 → 下一轮 pump」之间** —— actor 循环每 token 走
`deliver(sink)`（事件序列化 + 通道发送）+ 账务（`pending_cache`/统计）。**下一刀(P2.2)
的入口 = `apps/server/actor.rs:124` 的 `running.pump()` 与 `deliver` 两侧加打点**。

**打点留存**(恒开、纳秒级、零行为影响):`[sched] loophead=`(循环头 ✓ ≈0)、
`step-prof exec-main=`(函数本体 18.9ms)。

#### 66.5 ★★ 定谳修正：那 1.55ms 是**探针假象**（P2.2 结案）

分段打点（`pump-phases`）把 `head` 1.63ms 精确定位到 **`sched→exec` 段**
（`entry→lh` 300ns、`lh→sched` 400ns、**`sched→exec` 1.63ms**）——而 `schedule()`
本身 ≈0 ⇒ 读码：`pump` 开头

```rust
if prof { owl_shared::metrics::query_metrics(&MetricsFilter::new().tag_prefix("engine.pump.")); }
```

**每步执行** ⇒ **实测 1.63ms/步**（**探针观测即污染**）。

**修**：降频到每 64 步一次（静态计数，零结构改动）⇒ 复测：`head` **800ns**
（原 1.63ms）、`sched→exec` **110ns**、`pump` **18.7–19.0ms**（原 20.1–20.8ms）。

**⇒ 对 host 尾巴账的修正（推翻 66.1 的结论行）**：

| 项 | 修正前（探针档）| 修正后（真实）|
|---|---:|---:|
| host 暴露/步 | ~2.0ms（~10%）| **≈ 0.6ms（~3%）** = fill+launch 0.25 + host-sample 0.35 |
| 循环尾 | 1.55ms | **≈ 0**（探针假象）|
| §2.1 旧账 "host 2.3ms" | — | 同为**探针污染读数**（那轮也开了 `step_profile`）|

**⇒ P2.2 结案**：无大头可修（剩下 0.6ms 全是必要工作：图入队 + host 采样）。
"上限 decode +10%" 改写为 **"+3% 且已触底"**。

**方法论坑（记为纪律㊲）**：**`step_profile` 会改变被测对象**（每步 +1.6ms）——
凡用它做的长跑读数（decode t/s、相位比），要么**先扣探针开销**，要么看
`pump-phases` 的 `head`/`sched→exec` 是否为 ns 级自证。

---

## §67 slab 定尺**根治**（保守版;2026-10-19 深夜）

> 承接 §65.3-2（融合档启动被拒 → 临时 `[cuda] capture_slab_mb = 32` 解封 + `state.rs` 的 hint×2 兜）。

### 67.1 根因（精确定位）

- `slab_hint` 设计：`GraphPlan` warmup 的 eager dry 跑 → `probe_end()` 取**实测影子峰值**
  `peak` → `hint = base(+12.5%)+1MiB` → `graph_begin` 消费（**口径已对齐**：`reclaim_capture`
  用 `peak`、plain 用 `metered`，见 `graph_plan.rs` 的 M1 教训注释）。
- **但 `peak` 是 warmup（eager）期的峰值**，而**捕获期**的回收点更少（**节点合并** ⇒
  块活跃期变长）：v1 两核融合把 stage+reduce 并成一节点后，**实测需求/base ∈ [1.40, 2.45)**
  （base ≈ 10.1MiB 时：12.36MiB 的旧 hint 差 139264B 仍爆；24.7MiB 够）。

### 67.2 修法（源头给足；不二次放大）

1. `graph_plan.rs`：hint 余量 **`+12.5%+1MiB` → `2× + 2MiB`**（覆盖 2.0 以下的需求比）；
2. `state.rs`：**回退**先前的 `hint × 2`（双保险会 4× 吃显存）⇒ 恢复原逻辑。

### 67.3 验证

| 档（均**无** `capture_slab_mb`）| 结果 |
|---|---|
| spec + 切档 + **融合** | ✅ 启动;**std decode 109.96**（与解封前 110.01 一致）|
| spec + 切档（不融合）| ✅ 启动 |

### 67.4 遗留（真根治）

`peak` 的**捕获期 liveness 直接跟踪**（现在是 warmup peak + 2× 余量兜）——
或 carve 失败自动放大重试（**受 `FnOnce` 闭包限制不可行**，见 §65.3-2）。

---

## 68. ★★★ P1.4 GDN prefill 布局重写 —— 逐位相同 1.61×,端到端 prefill **+5.7%**(2026-10-19)

### 68.0 立案:为什么不是 S5

本轮开工时路线图的"下一刀"是 **P2.4/S5(AR 分块流水,把 AR 藏进算里)**,但账已翻过:

| 出处 | 形状 | 结论 |
|---|---|---|
| §48.3 | **单发**:先发 64 个小读核、再发 64 个 GEMM | [读 ‖ GEMM] 重叠 **97%** |
| §49.3 | **链式**(`G→AR→G→AR`,= 生产依赖形状),N=2/3/4 微批 | 吞吐比**卡死 1.21–1.28×**,N 变大无用 |
| §49.4 | 定量 | 总时间 ≈ `总 PCIe + 总 GEMM`(**串行度 ≈ 1.000**)|
| §41.3 | 写腿 ‖ 本地 | **0%** |

⇒ "把 AR 藏进本层计算"在**核级没有地基**,S5 不是"改协议"能解决的;
**P1.4(GDN prefill,19% 未调研)才是当下最大的一块**。

### 68.1 先校准形状(**本轮第一个坑**)

P0.1 账里 `owl_gdn_recurrence_varlen_gqa_f16` = 664 ms/卡(19%)、768 发。
我按 27B 的 `nv=48/nk=16` 建台,**量出 3.29 µs/token、对不上生产 1.73**——
回 nsys 查该核的 **grid 才发现是 `(16, 24, 1)`**:

> **TP 把 linear-attention 的 value 头也切了** ⇒ 每 rank **nv=24 / nk=8**(`kv_group=3`)。
> 按此重建台子后 **T=1024 → 1771 µs**,与生产 1730 µs 逐点吻合 ✓

**纪律(新增 @39)**:任何"从端到端反推的单核读数",上核台前**先用 nsys 的 grid/block 核对形状**;
TP 档的"每 rank 头数"是**配置声明`/world`**,不是 config.json 的字面值。

### 68.2 归因阶梯(`tools/gdn-rec-probe.cu`;3080 ord2,nv=24,nk=8,T=1024,8k 预热后 1.9 GHz)

| 臂 | µs/token | Δ | 病根 |
|---|---:|---:|---|
| **V0 现役** | **1.74** | — | 生产同形 |
| V1 warp 自治(去 `__syncthreads`) | 1.50 | −0.24 | 8 个 warp 之间**零数据依赖**,barrier 纯税 |
| V3 = V1 + 寄存器预取 4 | 1.40 | −0.10 | 装载延迟(只剩 6%,不是主因) |
| D0 = V1 去**两次 32-lane 归约** | 0.98 | **−0.42** | 一列一 warp ⇒ 归约按列重复 |
| B1 = D0 再删 **q/k 装载** | 0.53 | **−0.45** | 块内 8 warp 各读同一份 256B(8× 冗余)|
| B3 = B1 再删 **out 写** | 0.34 | **−0.18** | 每 token 一条**单 lane 2B** 写(32B sector) |
| B4 = 再删 state 口 | 0.33 | −0.01 | 纯算术骨架 = 19% |

**⇒ 81% 是布局税,不是算术**(序列长度 T 的串行链本身只值 0.33 µs/token)。

### 68.3 新形(`_cols2`):一个 warp 管 2 个相邻 v 列

| 改动 | 效果 |
|---|---|
| C=2 列/warp ⇒ 归约树 2 列共用 | 每列 shfl **10 → 2.5** |
| 同一份 q/k 摊给 C 列;`g`/`beta` 是**每 head 一个标量** ⇒ 每 token 只取 1 次 | 装载指令 ÷(2×) |
| **删 shared / 删 barrier**(warp 自治) | −0.24 µs |
| out:2 列合一 **4B** 写 | −0.10 µs |
| 寄存器预取深度 2(槽索引**恒编译期**) | −0.10 µs |

**★ 算术序逐位不变** ⇒ 判据是 `out`/`state` 与旧核**逐字节相同**(不是容差)。
⚠️ 中途自踩两次,都值得记:
1. **环形槽用 `t % D` 运行期索引 ⇒ 退化成 local memory**(4.8 µs/token,比不预取还慢 3×)
   ⇒ 必须把 t 循环**按 D 展开**让槽号成常量;
2. 预取实现里 `y[c]` 被**复用**成 delta(状态对、out 全错)⇒ 台子的"逐字节对拍"当场抓到;
   **没有它,这个 bug 会以"out 合理但偏 3e-4"的形态流进生产**。

C 的扫描:C=1 1.41 / **C=2 1.087** / C=4 1.20(C=4 让并行 warp 数掉到 1/4,
延迟藏不住)⇒ 取 **C=2,D=2**。

### 68.4 落地

- `crates/kernels/cu/gdn/gdn_recurrence.cu`:入口名/签名/槽序**不变**;新增 `gdn_rec_cols2`
  快路 + **原形体逐字保留**为回落(kd≠128 或 vd 奇);`gdn_rec_fast_shape` 判据;
- `crates/kernels/src/driver.rs`:`recurrence_varlen_gqa_fast`(与核内判据**同源**)+
  grid 步长 16(快路)/ 8(回落)+ 单测 **`gdn_rec_geometry_matches_route`**
  (判据不同源不是"慢一点"而是**列覆盖错位**:legacy 的列步长 8、快路 16);
- 台:`crates/kernels/tests/gdn_recurrence_bench.rs`(生产形状 µs/token + 逐位参考落盘/比对 +
  **时钟自证**:空载 210 MHz,不预热会系统性偏低 2×)。

### 68.5 验收(全部实测)

| 面 | 读数 |
|---|---|
| 核台(3080 单卡,T=1024,刷新前参考) | **1.75 → 1.08 µs/token(−38% / 1.61×)**;**out+state 逐字节相同** |
| `gdn_chunked_golden` · `gdn_scalar_golden` · `gdn_big_t_arms_parity`(0.8B 大 T 双臂) | ✅ 全绿 |
| `tp_g8_gate`(0.8B TP2≡TP1,**L1 逐位**)· `tp_determinism` · `tp_transport_bits` · `tp_assemble` | ✅ 全绿 |
| `driver::tests`(含新几何门) | ✅ 9/9 |
| **端到端 std(TP2 / 双 3080 / 五点三轮 / `tp2-nospec-vsplit-fused`)** | prefill **1486.1 → 1570.35(+5.7%)** · decode 49.25 → **49.41**(噪声内,GDN decode 走另一核,未动)· TTFT@8k 5255 ms |

**机器状态**:本次 std 期间 3090 对上有别的会话(`owl_models` 3.7GB + `llama-tokenize`),
**3080 对本轮干净**(启动日志 `[boot] 选卡` 自证 rank0/1 = 3080 两张)。

### 68.6 遗留

1. **q/k 装载仍是 2 列冗余**(每 block 8 warp 各读一遍):下一个杠杆是"**lane 持有连续 4 行** +
   `uint2` 8B 向量装载"(每 lane 每 token 2 条装载代替 8 条)—— 但**求和序变**,
   不再是逐位相同(要另立容差判据);
2. **FLA chunked(matmul)版**是另一条路:TP1 的 `spec-dflash` 已用(`gdn_chunked=true`),
   但 AOT cubin 按 `NV=48` 烘死(`crates/kernels/triton/gdn_chunked_aot.json`),
   TP2 的 (8,24) 需重编 + 重走 c1–c5 金标 ⇒ **未量过它 vs 串行核**,属"待翻案"清单新条目;
3. `op_tables` 两门**在 HEAD 上就是红的**(`owl_comm_ar_stage_f8_f16` 的 `.cu` 形参表
   解析不出 —— P1.3 fp8 AR 遗留),与本刀无关,已当场用 `git stash` 复现验证。

---

## 69. decode 逐核账(2026-10-19,刀后重量)+ **一个仪器陷阱**(nsys node 模式虚报 graph launch 17×)

### 69.1 现状(TP2 / 双 3080 / `tp2-nospec-vsplit-fused`;prompt 63 tok、64 token 解码)

`tools/nsys-decode-account.sh p14-dec`(node 模式)单步:核 964 个、设备 Σ **17.22 ms**。

| 项 | µs/步 | 占比 | 判语 |
|---|---:|---:|---|
| marlin(量化权重;含次档) | 11461 | **57%** | 617 GB/s = **81% 峰值**,已贴墙 |
| **`owl_comm_ar_fused_f16`(128 对)** | **2285** | **11%** | 17.9 µs/对;10 KB 载荷的**传输只占 ~1 µs** ⇒ 全是小包延迟 |
| lm_head(`internal::…` = gemv) | 1755 | 9% | 词表切后(原 3506);**715 GB/s** |
| paged attention | 396 | 2% | |
| `gdn_decode_step_v2` | 348 | 1.7% | |
| `gemv_dual` / norm 两族 / silu / add / gdn_conv … | ~750 | 4% | |
| **设备合计** | **17216** | 85% | |

### 69.2 ★ 陷阱:node 模式把 `cuGraphLaunch` 量成 1.47 ms(真值 0.17 ms)

node 模式的分账显示"段间空隙 3.26 ms/步",其中 `cuGraphLaunch` **2.84 ms/步**
(p50 **1469 µs**/发 × 2 发 = 两 rank 各一张 ~994 节点的图)。若当真,这是 decode 的
第三大项(16%)。

**同一份配置换 `--cuda-graph-trace=graph` 复测:`cuGraphLaunch` p50 = 171 µs**
(Σ 23.2 ms / 132 发)⇒ 1.47 ms 里 **~1.3 ms 是 CUPTI 逐节点插桩的开销**,不是驱动。

**再用引擎自己的钟交叉验证**(`[probes] step_profile = true`,无 nsys):
```
[pump-phases] head=1.5µs [entry→lh=0.7µs lh→sched=0.6µs sched→exec=0.23µs] exec=19.13ms
[step-prof]  fill+launch=0.30~0.44ms   sample-and-emit=23µs   token-dtoh=18.78ms
```
⇒ **真实 host 面 = fill+launch + 采样 ≈ 0.33 ms/步(1.6%)**,与 §66 的 0.6 ms 同量级 ✓;
node 模式的 3.26 ms **作废**。

**律(新增 @40)**:**任何"图内"读数,先问仪器**——`--cuda-graph-trace=node` 会**逐节点插桩**,
它的 `cuGraphLaunch`/空隙**不可用于 host 面归因**(虚报 17×);host 面一律用
`step_profile` 或 `--cuda-graph-trace=graph` 复核。
(反向也成立:node 模式的**核内时长**仍然可信 —— 那才是它的用途。)

### 69.3 于是 decode 20.2 ms/步的真实构成(8k ctx 口径)

| 项 | ms | 占比 | 杠杆 |
|---|---:|---:|---|
| marlin 权重遍 | 11.1 | **55%** | 已 81% 带宽 ⇒ **只能减字节**(投机=一次权重读算 k 个 token)|
| **AR 小包**(128 对 × 17.9 µs) | **2.3** | **11%** | 传输 ~1 µs ⇒ **94% 是延迟**;上限 +11% |
| lm_head | 1.75 | 8.6% | 715 GB/s;**int4/int8 化可到 ~0.45 ms ⇒ +6.4%** |
| attention(+8k 上下文的 Δ)| 0.4~0.9 | 2–4% | |
| GDN+norm+小核 | ~1.1 | 5% | |
| **图内节点税**(994 节点:host 0.3 + 设备侧依赖间隙 ~1.5) | **~1.8** | **9%** | **节点数 ∝ 成本** ⇒ 融合小核 |
| host 尾巴 | 0.33 | 1.6% | 已定谳(§66) |

**⇒ "小包延迟"是 decode 里**真实存在**的一块(11%),但它**不是最大的一块**:
lm_head 量化(+6.4%,一刀)与节点税(~1.8 ms,9%)都在同一量级,而 marlin 的 55%
只能靠结构(投机已给 110 t/s)。

---

## 70. AR 小包延迟分相(2026-10-19)—— 协议本体很便宜,**fence 几乎免费**,节点税 1.34 µs/节点

### 70.1 工具:`tools/ar-latency-probe.cu`(单进程双卡、两线程各驱动一 rank)

把生产融合核的协议逐相复制(单槽 ⇒ 省掉环,但 credit/发布/rendezvous/读加/收尾全在),
每相打 `clock64()`(单 block ⇒ 同一 SM,亚 ns;⚠️ `%globaltimer` 在 sm_86 上**量化 ~1 µs**,
首轮就是被它坑到"每相只有 1.02 µs 分辨率")。

| 臂 | 做什么 |
|---|---|
| `MODE=0` | 全协议(生产同形) |
| `MODE=1` | fence 换 `__threadfence()`(device 域,数值不安全,只测成本) |
| `MODE=2` | 跳过 rendezvous |
| `MODE=3/4` | 只读不加 / 完全不读对端 |
| `MODE=5` | 无协议(只留核体:entry + 10 KB 本地拷 + 发布) |
| `MODE=6` | **AR 紧跟一个满占用核**(68 block 流式读 8 MB ⇒ 复现"GEMM 之后"的位形) |
| `GRAPH=1` | 捕获 K 个节点成图再回放(**双边**回放 = 全协议;`MODE=5` 时测节点税) |

### 70.2 读数(3080 对 2/3;10 KB;K=120~200;`OWL_CLK_GHZ=1.9`)

**MODE=0 全协议(隔离发射,两 rank 自由跑)= 8.35 µs/次**:

| 相 | p50 µs | 占比 |
|---|---:|---:|
| ②写本端 slot(10 KB) | 1.54 | 19% |
| ④发布 produced(含 `__syncthreads`) | 1.25 | 15% |
| **⑤等对端 produced** | **2.49** | **30%** |
| ⑥读对端 + 累加(10 KB 跨卡) | 1.11 | 14% |
| ①credit 等 | 0.43 | 5% |
| entry + 读对端 header | 1.34 | 16% |
| **③`__threadfence_system()`** | **0.04** | **0.5%** |

**⇒ 三条结论**:
1. **fence 不是问题**(0.04 µs;`MODE=6` 前驱核把 L2 弄脏后也只 0.47 µs)—— 推翻"system 域
   fence 贵"这条直觉假设;
2. **协议本体 ~4 µs**(写 1.5 + 读 1.1 + 发布 1.25),10 KB 的真实搬运就这个量级;
3. **最大单项是"等对端"2.5 µs** —— 而它是**两 rank 的最小 skew**,不是协议缺陷。

### 70.3 逐层加码:把 17.9 µs 拆到只剩一格

| 位形 | µs/次 | Δ |
|---|---:|---:|
| 隔离发射(两 rank 自由跑,全协议) | **8.35** | — |
| 前驱满占用核 + 全协议(`MODE=6`) | **8.70** | +0.35 ⇒ **"GEMM 之后"不是原因** |
| **图内双边回放**(K 节点串联,全协议) | **10.18** | +1.5 ⇒ **图/节点依赖的税** |
| `MODE=5` 同图(无协议,只核体) | 4.59 | 核体 3.25 + **节点税 1.34** |
| **生产**(`p14-dec`,128 发/步) | **17.90** | **还差 7.7 µs(43%)未归因** |

**生产侧另外两条定谳(来自 `p14-dec` 的 8704 发)**:
- **没有 skew 累积**:步内 #1..#127 的 p50 恒为 **17.7–19.0 µs**,只有**每步第 0 发**偏高
  (rank1 44 µs)⇒ 两 rank 的发射错位在**第一个 AR 一次吸收**,之后不再计息;
- 分布极窄(p10 16.9 / p90 19.7),长尾(<1%)是首包/换页,不是机制。

⇒ **剩余 7.7 µs 的候选只剩三格**(下一刀的打点顺序):①**VMM 别名读腿**
(§60 阶梯在 10 KB 档判"≈0",但那是**单向 burn** 形状,AR 是"读+本地 RMW+双向 flag");
②生产核是 **512 线程**(probe 256);③生产每步**交替两个 slot**(probe 固定 slot 0)。
判别法 = **给生产核加一版临时分相打点**(`__device__` 全局数组 + 宿主侧名读取),
一次就能定到哪里。

### 70.4 ★ 顺带量到一条**独立于 AR** 的账:图节点税

`MODE=5` 图内回放(200 节点串联)= **4.59 µs/节点**,隔离发射同一核 = **3.25 µs**
⇒ **每节点 ~1.34 µs**。decode 单步 **994 节点** ⇒ **~1.33 ms/步(6.6%)**,
与 §69.3 的"节点税 ~1.8 ms 估算"吻合(实测略小)。

**⇒ 这是一条与"减字节"完全正交的杠杆**:融合掉 100 个小核 = 省 0.13 ms/步(+0.7%);
要拿 3% 就得融 500 个节点 —— 所以它的正确用法是**跟着别的融合刀顺手做**,
不值单独开战。反过来说:**再往上加核(比如把大核拆成多个)要先付 1.34 µs 的买路钱**。

---

## 71. 更正 decode 的"权重带宽墙"(§69.3 的 as-of 作废)+ marlin m=1 配置空间已穷尽

### 71.1 字节账更正:每卡每 token 读 **8.29 GB**(不是 9.62)

按 checkpoint(`cyankiwi/Qwen3.8-27B-AWQ-INT4`,21.0 GB 张量)逐张量清点:

| 类 | 总量 | decode 每卡读 |
|---|---:|---:|
| `weight_packed`(int4) | 12.15 GB | ÷2 = **6.08** |
| `weight_scale`(g32) | 1.52 GB | ÷2 = **0.76** |
| `weight_zero_point` | 0.38 GB | ÷2 = **0.19** |
| `lm_head`(bf16) | 2.54 GB | ÷2 = **1.27**(词表切后)|
| `embed_tokens` | 2.54 GB | **≈0**(查表 = 1 行 128 B,不是遍历)|
| `mtp.*` + 视觉塔 | 1.89 GB | **0**(`depth=0`;视觉塔文本档不用)|
| **合计** | | **≈ 8.29 GB** |

⇒ `8.29 GB ÷ 13.22 ms = 627 GB/s`(**3080 20G 峰值 760 ⇒ 82%**),不是 §69.3 写的
"96% / 已贴墙" —— 那笔账的 9.62 GB 把 embed 表/MTP/视觉塔算了进去。

**拆开看,两半的效率差很多**:

| 核 | 字节 | 时间 | 有效带宽 | 判语 |
|---|---:|---:|---:|---|
| lm_head gemv(手写,专为 m=1) | 1.27 GB | 1.76 ms | **724 GB/s** | **95%,真贴墙** |
| **marlin(352 个量化 GEMM)** | 7.02 GB | 11.46 ms | **613 GB/s** | **81%,有 ~1.7 ms(+8.5%)空间** |

### 71.2 m=1 的 marlin 是**核本体**慢,不是引擎损失(微基准复核)

`marlin_bw_bench`(隔离,生产形状):34816×5120 **661** / 5120×17408 **630** /
16384×5120 **632** / 5120×6144 **549** GB/s(**m=1 与 m=8 同速**)⇒ 与引擎里反推的
613 完全吻合 ⇒ 图内没有额外损失,m=1 也不是 batch 形态问题。

### 71.3 配置扫描:**实例化档里没有更肥的**(零风险路走到头)

`marlin_host.cu` 的小批档表 = `{128,128,256} / {64,128,128} / {128,64,128}`,
外加一条本地 override 判据;`m=1` 走 **`m_block_size_8`** 路径。

新增"配置可注"的台子口(`marlin_gemm_v2_awq_cfg_ffi` / `gemm_v2_awq_raw_cfg`,
**仅测试面;生产仍 auto**),扫出的结果:

| 旋钮 | 结果 |
|---|---|
| `MARLIN_NO_M8=1`(强制非 m8) | 638 / 613 / 612 / 527 ⇒ **比 auto 慢**(m8 路本来就对) |
| `MARLIN_NO_OVERRIDE=1` | 与 auto **逐点相同**(那条 override 在 m=1 本来不触发) |
| `thread_k/n` = 64/128 · 128/64 · 64/256 · 128/256 · 32/128 | **全部 err**(变体不在实例化集里 —— 构建期门"selector ⊆ 实例化"保证了这点) |
| `thread_k/n` = 128/128(= auto 档) | 661 / 630 / 632 / 550 ⇒ 与 auto 一致 |

**⇒ 结论:auto 选的就是 m=1 唯一可用的档,80% 是核本体效率。**
要再往上只有一条路:**自己写 m=1 的 int4 GEMV**(读**同一份** marlin 打包布局,
线性/瓦片化搬运 + 8 路展开;数值同式 ⇒ 只差 fp32 累加序,~1e-7)。

### 71.4 为什么 marlin 到 81% 而 lm_head 能到 95%(写新核时的靶)

marlin 的 B 是**为张量核重排过的 tile 布局**(`(k/16, n·16/8)` i32 tile-perm),
m=1 时张量核按 16 行 padding 工作、4-bit 解包(shift/or/cvt)也照付;
lm_head 的 gemv 是**线性布局 + `uint4` 直读 + 8 路展开**,所以能贴峰值。
⇒ 新核的靶就三件事:①线性化访问(同一份字节,换读法)②解包与 FMA 融合
③每 warp 多行(`n`)以摊薄归约。

### 71.5 下一刀的顺序(用户裁决 2026-10-19)

1. **lm_head → int8**(+4.3%:1.76 → ~0.88 ms;误差估算 ~0.6% 相对 logits,
   判据 = top-1 ≥99.9% + 素 256 token 逐字一致);
   **副产品**:它就是一个 m=1 的量化 GEMV 骨架 ⇒ 直接给第 2 条当模板;
2. **marlin → m=1 int4 GEMV**(+8.5% 上限;读同一份布局,数值同式);
3. 再往上是 AR 协议与合批/投机(见 §70/§69)。

---

## 72. P1.5-a 第一段:int8 权重 GEMV 落地 —— **718.8 GB/s / 0.885 ms**(靶 700+,已过)

### 72.1 核(`cu/elems/gemv_i8w_f16.cu`)

`y[token, r] = scale[r] · Σ_k w_i8[r,k] · x[token,k]`(per-row 对称量化;装载期现算)。
形态:**一 warp 一行 · 8 warp/块 · 每 lane 一次读 16 B(16 个 int8)· fp32 累加 +
5 步归约**;块内**先把 x 那一行(≤10 KB)搬进 shared**给 8 个 warp 共用 ——
否则每行都要从 L2 重读 10 KB(124160 行 × 10 KB = 1.24 GB 的 L2 流量,会把带宽吃掉)。

**不用 `dp4a`**:激活是 f16(W8A16),没有 int8×int8 算术;把激活也量化(W8A8)
会在本已不是瓶颈的算力上再加一层 per-token 误差。

### 72.2 读数(`gemv_i8w_bench`;3080 单卡;M=1 与 M=8 同速 ⇒ 带宽受限已坐实)

| rows × cols | 权重字节 | µs/发 | 有效带宽 | 数值(与 host f32 参考,*同一份量化权重*) |
|---|---:|---:|---:|---:|
| **124160 × 5120(生产:词表切后)** | 0.635 GB | **885.4** | **718.8 GB/s** | max\|Δ\|/max = **2.6e-4** |
| 32768 × 5120 | 0.168 GB | 238.5 | 704.4 | 2.6e-4 |
| 124160 × 2048 | 0.254 GB | 358.4 | 711.6 | 3.3e-4 |

- **靶 ≥700 GB/s 达成**(718.8,与 bf16 路的 724 同效率 ⇒ 本核没有把"减字节"的收益
  还回去);预期生产线:**1.76 ms → ~0.89 ms ⇒ decode +4.3%**;
- `max|Δ| = 2.6e-4` 是"同一份 int8 权重的 f32 点积"与核输出的差(fp32 累加序 + f16 输出
  舍入)⇒ **核没有做多余的事**;剩下的是 int8 量化本身(~1.6% 相对 logit)——
  那要过 §71.5-a 的**精度门**(真实 hidden 上 top-1 ≥99.9% + 贪心 256 token 逐字一致)。

### 72.3 已落 / 未落

| 面 | 状态 |
|---|---|
| 核 + 声明表(ops/elems)· driver 几何 · 三张池族清单 · 判据台 | ✅ 已落并实测 |
| **装载器**:`lm_head.weight`(bf16 → per-row int8 + f32 尺度,只做本 rank 那半)| ☐ 未落 |
| **接线**:`Embedding::lm_head_matmul` 分派 + 语义 op + flavor 白名单 | ☐ 未落 |
| **精度门**(真实 hidden / greedy 256 token)| ☐ 未落(接线后才有真实 hidden)|

**下一步(顺序已定,别改)**:装载器 → 接线 → **先跑精度门** → 过了才进 std 测速。
精度门不过的退路:per-row → **per-128 分组尺度**(误差 ÷2 量级,代价 = scale 表 40×)。

---

## 73. P1.5-b 的定性:**不能照搬 vLLM 的 AWQ GEMV** —— 卡在布局(perm 查表税)

### 73.1 外部资料的确认(三件都有用)

读了 vLLM `csrc/libtorch_stable/quantization/awq/{gemm_kernels.cu,dequantize.cuh}`
(与 sglang `sgl-kernel/.../awq_kernel.cu` 同源):

| 点 | 我们核过的账 |
|---|---|
| AWQ 原生 GEMV 专为 `N=1` 写(uint32 打包 + `LDG.128` + 寄存器反量化)| 与 §71 实测同向:marlin m=1 = **613 GB/s(81%)**,GEMV 形态 = 95% |
| **解包比特技巧** `dequantize_s4_to_fp16x2`:`MAGIC = 0x64006400` + `prmt` + 减魔数 ⇒ **8 个 int4 → 8 个 fp16 只要 ~5 条指令** | ★ 比我们原计划的逐元素 `cvt.f32.s8`(16 条)省 2/3 ⇒ **入库,后续任何 4-bit 核直接用** |
| 组尺度/零点按 `k/G` 取、`uint4` 一次 8 个 | 与 checkpoint 的 g32 语义一致 |

### 73.2 ★ 但我们的权重不是 AWQ 原生布局

| | 布局 | GEMV 的读法 |
|---|---|---|
| AWQ/vLLM | `qweight[K/8][N]` i32:8 个 nibble 沿 **k** 连续、沿 **n** 紧邻 ⇒ warp 读 32 个 n 的 word = **128 B 连续,零置换** | 线性流 |
| **本仓** | 装载期已重打成 **marlin tile-perm**(`pack_marlin_b`):`(k/16, n·2)` i32,且每 1024 nibble 再套一层 **1024 项 `perm`**(`src = perm[dst0+j]`)| 每 512 B 连续块 = `[16k]×[64n]`,**但块内每个 nibble 的 (k,n) 由 perm 决定** |

⇒ 按 marlin 布局做 GEMV,**每个 nibble 要查一次 1024 项置换表**(16 B 数据配 32 次查表)
⇒ 查表开销基本吃光那 15% 的带宽差 ⇒ **我原计划里"把 B 的读取换成 marlin 布局"这一步,
现在有了量化判据:不值得**(该结论有范围:单 nibble 查表;若将来把 perm 折进
`__constant__` 的 1024 B 表并以 warp 级广播重排,需重新评估)。

### 73.3 剩下的唯一可行路:**双布局**(需要用户裁决,涉及内存取舍)

decode 读 **AWQ 原生**(即磁盘 `weight_packed` 原样、零置换),prefill 继续 marlin:

```
每卡  marlin 7.02 + AWQ 原生 7.02 = 14.04 GB
    + embed 2.54(复制态)          ← 可按词表切分 → 1.27(decode 每 token 只需 1 行,代价 = 每 token 一次小 all-gather)
    + lm_head 1.27               ← int8 → 0.64(§72 的核已就绪)
    + 运行底(KV/GDN/图/预留)     ≈ 2.1
  现状:17.9 + 2.1 = 20.0 GB  ⇒ **超 19.63,装不下**
  省完:15.95 + 2.1 = 18.05 GB ⇒ **能装,但 262k 上下文弹性基本没了**
```

**收益 +8.5%(11.46 → 9.7 ms);代价 +7 GB/卡 + 装载时间翻倍 + 两套布局要同步(双真相风险)。**

### 73.4 结论与顺序(2026-10-19 收尾)

1. **先收尾 lm_head int8**(核已 718.8 GB/s;剩装载器 + 接线 + 精度门)⇒ **+4.3%,
   零布局代价、零显存代价** —— 这是当下性价比最高的一刀;
2. 双布局**待用户裁决**:要 8.5% 还是保 262k 上下文的弹性;决定前先量 1 落地后的 `live`;
3. 解包 magic-num 技巧已入档(§73.1);它也是将来"真要写 4-bit GEMV"的地基。

### 72.4 P1.5-a 剩余三段的施工面(2026-10-19 收尾时勘查;下一次接手直接照做)

核已验收(§72.2),剩下的是**装载器 + 接线 + 精度门**。勘查结论:

**两条可选路径,推荐 (B)**:

| | 路径 | 改动面 | 代价 |
|---|---|---|---|
| A | 加 `QuantPlan::I8W` 臂,走**既有 RTN 派生键**机制(`formats::awq_build` 的 `rtn_g32_*` 同族) | `Linear::{layout,forward_inner}` + `awq_build` 新增 per-row int8 派生 + `SemanticKernel::GemvI8w` + flavor 白名单 | 动 `Linear`(已有五条臂,最脆) |
| **B(推荐)** | **装载期"设备一次性量化"**:权重上传后跑一个 `quant_i8_row`(bf16 `[rows,cols]` → i8 + f32 `[rows]`),然后**释放 bf16 块** | 新核 `cu/load/quant_i8_row_w.cu` + `Embedding` 加 `head_i8: Option<(Weight, Weight)>` + `lm_head_matmul` 分派 + 一个装载后钩子 | 不碰 `Linear`;但要在装载序列里插一次设备核(与 `marlin_repack_ct` 同一姿势,已有先例) |

**为什么推荐 B**:`Linear` 现役已有 F16 / W4A16 / W4A16Awq / W4A8Awq 四条臂,再插一条 int8
会把它变成"万能类型"(违反 §Rust 要求:按字段/状态拆小结构体);而 int8 头**只有
lm_head 一处消费者**,单独一个槽 + 一个分派即可,且**回退 = 删槽(bf16 原路不动)**。

**顺序(每步自带判据)**:
1. `quant_i8_row_w` 核 + 台子:为**逐行**对称量化(`s_r = max|w_r|/127`,`q = round(w/s)`),
   判据 = 与 host RTN **逐元素相同**;
2. `Embedding` 加槽 + `layout` 声明两个派生 Want(i8 `[rows,cols]`、scale `[rows]`,**行数随
   `head_vocab` ⇒ 与 §63 的词表切同源**)+ 装载后钩子跑核并释放 bf16;
3. `lm_head_matmul` 分派 `SemanticKernel::GemvI8w`(`aux=[rows, cols, tokens]`);
4. **精度门**(必须有真实 hidden 才能跑):同一 prompt 下 bf16 头 vs int8 头的
   **top-1 一致率 ≥99.9%** + **贪心 256 token 逐字一致**;不过 ⇒ 退到 **per-128 分组尺度**
   (误差降 ~2×,scale 表 0.5 MB → 20 MB);
5. std 五点三轮(预期 1.76 → ~0.89 ms ⇒ decode **+4.3%**:49.4 → ~51.6 t/s)。

---

## 74. marlin m=1 的 81%:**不是 tiling、不是并行度,是 tile-perm 布局本身**(2026-10-20)

### 74.1 把"换档位"这条路扫到底

上一轮(§71.3)只扫了 `(thread_k, thread_n)`,而 FFI 把 `threads` 写死 256 ⇒ 实例化表里
**只在 `threads=128` 下存在**的两档(`thread_n=64` / `thread_k=64`)根本没被扫到。
补上 `threads`(台子侧 `MARLIN_THREADS`,**只作用于显式配置分支**,auto 生产路不经过),
并同时 `MARLIN_NO_OVERRIDE=1`(否则本地 override 会把显式档改回 `{128,64,128}`):

| 配置(thread_k, n, threads) | 34816×5120 | 5120×17408 | 16384×5120 | 5120×6144 |
|---|---:|---:|---:|---:|
| 128 / 128 / 256(= auto) | 660.7 | 629.0 | 632.2 | 549 |
| 128 / **64** / 128 | 660.3 | 629.7 | 632.4 | — |
| **64** / 128 / 128 | 660.3 | 629.5 | 632.7 | — |

**逐点相同(±0.5 GB/s)** ⇒ **tile 形状与小批并行度都不是那 15% 的原因**。
(顺带定谳一个测量坑:不加 `NO_OVERRIDE` 时,显式 `thread_k/n` 会被本地 override
**静默改回**默认档 ⇒ 三档读出一模一样的数就是证据。)

### 74.2 那 15% 从哪来:对照我们自己写的两个 GEMV

| 读法 | 布局 | 有效带宽 |
|---|---|---:|
| lm_head bf16 gemv(cublas) | 线性 `[n, k]` | **724 GB/s(95%)** |
| `owl_gemv_i8w_f16`(§72) | 线性 `[n, k]` i8 | **718.8 GB/s** |
| **marlin(所有 tile)** | **tile-perm**(`(k/16, n·2)` i32 + 1024-项 `perm`) | **550–661 GB/s(73–87%)** |

⇒ 机器能跑 720+,**marlin 的缺口来自布局的访问形状**,不是 DRAM、不是算力、不是并行度。

### 74.3 于是只剩一条有收益的路:**读 marlin 布局 + 在寄存器里做 perm**

§73 否掉的是"**按 nibble 查 1024 项表**"(16 B 数据配 32 次查表)。但 `perm` 是
**纯算术**构造的(`repack::marlin_perm`:32×2×4×… + interleave `[0,2,4,6,1,3,5,7]`),
不是随机表 ⇒ 有两条更便宜的等价实现:

1. **纯算术求逆**(公式内联,每 nibble ~6 条 ALU;而本核的 ALU 预算有 40× 余量);
2. **1 KB shared 常量表**(每 block 预载一次,之后按 lane/nibble 索引):
   带宽需求 = 每 2 字节 1 次查表 ≈ 2.8 次/周期/SM,shared 吞吐(128 B/周期)绰绰有余。

读法 = **一个 warp 读一整块 512 B 连续块**(= `[16k] × [64n]`,完全 coalesced),
再用上面任一法把 32 个 nibble 归到各自的 (k,n) 上累加。**目标 ≥700 GB/s**
(现在 613)⇒ 11.46 → ~10.0 ms ⇒ **decode +6~7%**;且**不需要第二份权重**
(这正是 §73 双布局方案要换掉的东西)。

判据与风险:①单位带宽 ≥700(`marlin_bw_bench` 同形状);②**数值同式**
(同一份 int4 + 同 scale/zp,只差 fp32 累加序 ~1e-7);③门 = `tp_g8_gate` L1 + std。

## 75. P1.5-a 落地到 TP2(2026-10-20)—— 两个真 bug + 生产 A/B **decode +9.5%**

> 上一轮(§72–§74)只把 int8 头做到"单卡能跑、核 718.8 GB/s"。本轮把它接进
> **双卡词表切**的生产档,顺带修掉一个把 TP2 起服全堵死的 trunk 回归。三件事的
> 共同点:**判据都是"同值"**——两半 logits 同值、dtype 同值、账本同值。

### 75.1 TL;DR

| 面 | 修前 | 修后 |
|---|---|---|
| TP2 起服(任何档) | **必拒**:`adopt_ledger: 已发过 2221 条事件` | 起服 + 双 rank 图捕获 + std 全跑(0.8B 门:312 条事件那档也绿) |
| int8 头 × f16 执行档 | **装配即拒**:`owl_gemv_i8w_bf16` 声明 BF16 实得 F16 | 双 op_id(`_f16`/`_bf16`),层侧按激活 dtype 选 |
| int8 头 × 词表切 | **静谧错值**:两 rank 读同一段行(高半 = 低半复制) | 每 rank 本行区间(CPU 门 + GPU 运行期宽度/vs 双门) |
| 精度(TP1 27B,N=48) | — | **344/344 = 100%**(3/3 prompt 逐 token 全等) |
| 精度(TP2 词表切,N=256) | — | **744/768 = 96.9%**;唯一分歧步 = near-tie(margin **1.56e-2**) |
| 生产 std(3080 对,@1024/256) | decode **45.09** t/s | decode **49.37** t/s(**+9.5%**);prefill/TTFT 不变 |

### 75.2 bug ①:语义算子只有一个 `op_id` ⇒ 单一 dtype 实例

现象:`QuantPlan::I8W` 在 **f16 执行档**(27B W4A16Awq 的本档)装配即拒 ——

```
[dtype 守门] kernel "owl_gemv_i8w_bf16" 声明输出为 BF16,实得 F16
```

根因:核有两颗(`GemvI8wF16`/`GemvI8wBf16`,同一 `SRC`、模板参数不同),
但语义枚举只有 **一个** 变体、`op_id` 写死 `elems.gemv_i8w_bf16` ⇒ f16 档必撞。
(先例在仓里就有:`elems.quant_int8_row_{f16,bf16}` = 两个 op_id。)

修法(三处,与先例同形):
1. `models/src/ops.rs`:枚举拆 `GemvI8wBf16` / `GemvI8wF16`(`elems.gemv_i8w_f16`);
2. `kernels/src/driver.rs`:补 `"elems.gemv_i8w_f16" => elems::gemv_i8w(rows, cols, tokens)`;
3. `layers/embedding.rs::lm_head_matmul`:**按 `hidden.dtype` 选**(bf16 ⇒ BF16 核,
   f16 ⇒ F16 核;其它响亮 panic —— 头只有这两个档)。

门:`models/src/ops.rs::gemv_i8w_selection_face_is_wired`(op_id → 核名 → 声明出 dtype,
两档各一颗)。

### 75.3 bug ②:派生键没进"切分链路"(**静默错值**,最险)

现象:TP2 + 词表切 + int8 头,**装载绿、装配绿、生成连贯**,但把两 rank 的
logits 拼起来做 top-2 诊断时:

```
top1=97237(27.1719)  top2=221397(27.1719)  margin=0.00e0    ← 221397 = 97237 + 124160
```

**高半 == 低半,逐位相同** ⇒ 后半词表全是错的(只是"看起来能生成")。

根因两条(必须同时踩,故此前单卡门看不出来):
1. `plan::rules::DERIVED_SUFFIXES` **漏 `.i8q` / `.i8s`** ⇒ `base_key` 不剥后缀
   ⇒ `rule_of` 回落 `Replicate`(2026-10-17 就为量化族踩过同一坑,注释还写着
   "不剥 ⇒ 分片区间不翻译 ⇒ 两 rank 读同一段");
2. `Embedding::layout_lm_head` 的 i8 臂**没挂层声明规则** —— 词表切的
   `UniformCol` 覆盖只有 `Linear::layout` 里那一处(`with_tp_rule` ⇒ `Want.rule`),
   i8 头不走 `Linear`(§72 的设计),于是覆盖面漏了(`Want.rule = None`)。

修法:`DERIVED_SUFFIXES` 补两件 + i8 臂按 `h.rule` 调 `ops.with_rule(...)`
(与 `Linear::layout` 逐字同律:只在**真覆盖**时打标)。

门:
- CPU(`formats/shard.rs::derived_i8_head_rows_split_per_rank`):两 rank 各读
  本行区间、且**不许相等**(直接锁死"高半=低半");同时锁 `base_key` 剥后缀;
- GPU(`tests/tp_head_i8_gate.rs`):装配期 `spec.head_vocab × 2 == vocab`,
  运行期 `logits 宽 == head_vocab`。

### 75.4 顺带修的 trunk 回归:**TP2 起服全堵死**(与 int8 无关)

现象:`TpEngine::assemble` 在 rank1 报
`adopt_ledger: 本引擎已发过 N 条事件,只允许 boot 期(零事件)接管`
(0.8B **312 条** / 27B **2221 条**)⇒ **所有** TP2 门(0.8B 装配门、确定性门、
27B 门)与 server 起服全红,与 head 档无关。

根因:`c012cc5`(boot 收编设备面)之后 `Engine::on` **在 boot 就挂账本**,
而 TP2 的装配顺序是"先装载两 rank → 再 assemble";装载**必发事件** ⇒ rank1 的
私账已有内容 ⇒ 事后接管 = 把事件丢进孤儿账 ⇒ 设计上就该拒。**设计没错,调用面错了。**

修法(把"合流"提前到**装载之前**,并让接管可判):
1. `AllocLedger::book_token()`(默认 0 = 不表态;store 实现 = **书本体 Arc 指针**,
   不是 cell 指针 —— `ledger_entry()` 每次都新建 cell 壳);
2. `DeviceHandle::adopt_ledger`:**同一本书 ⇒ 幂等放行**(零事件丢失风险),
   另一本书 + 已记账 ⇒ **仍响亮拒**(老判据不削弱;`Arc::ptr_eq` 先行防双锁自死锁);
3. `Engine::new_with_ledger(cfg, ledger, dev_rank)`:rank1 在**构造期**接账本 +
   落定 `dev_rank`(池复用键带 rank,晚设会把两 rank 临时块混池);
4. `assemble_with_staging` 的"rank1 单元必须为空"判据加**同一性豁免**
   (`Arc::ptr_eq(cell0, cell1)` ⇒ 那本来就是主账,在借记录是两侧装载欠的);
5. 调用面:server `engine_of_tp` + 6 个 TP2 测试目标各一行
   (`Engine::new_with_ledger(cfg, e0.ledger_cell(), 1)`)。

门:CPU `device_handle.rs::adopt_same_book_is_idempotent_after_events`
(同书 + 已记账 ⇒ 放行;另一本 ⇒ 拒;token=0 ⇒ 按另一本);
GPU:0.8B `tp_assemble_gate` / `tp_determinism_gate` 双绿,server 起服日志
`[tp] 事件账:两 rank 合流到同一本(rank 维区分卡)`。

> 教训:判据写成"零事件"这类**时序谓词**时,必须同时给出**允许调用它的时点**
> 的入口(否则调用面一挪就全红);而"同一实体重复声明"应当是幂等,不是拒绝。

### 75.5 精度门(两进程;同进程两臂被池守卫拒)

同进程连跑两臂 = 两次 27B 装载 ⇒ 第二次 `PoolExhausted`(驱动 free 897MiB /
池 reserved 18816MiB;设备内存从不 Free 是设计前提,不是漏 free)。改成
`OWL_I8_HEAD_ARM=f16|i8|compare` 三态(两进程各一臂 + 零 GPU 比对)。

| 档 | 数据 | 判据 |
|---|---|---|
| **TP1 27B**(N=48,3 prompt) | **344/344**,3/3 逐 token 全等 | ≥0.95 ✅(实=1.0) |
| **TP2 词表切**(N=48) | 144/144,3/3 全等 | ✅ |
| **TP2 词表切**(N=256) | **744/768 = 96.9%**,2/3 全等 | ✅ |

唯一分歧(Rust 中位数那条,第 232 token)是**并列翻转**,不是错值:

| 臂 | top-1 | top-2 | margin |
|---|---|---|---|
| f16 头 | 478 (21.6250) | 16 (21.6094) | **1.56e-2** |
| int8 头 | 16 (21.6094) | 478 (21.6094) | 0.00e0 |

⇒ 满足"首分歧必须互为 top-N"(vLLM `check_logprobs_close` 同形)判据;
1.6e-2 的 logit 差在 int8 per-row(`s = absmax/127`,q 误差 ≤0.5)量级之内。

### 75.6 生产 A/B(server 档,std 口径,3080 对 @1024/256 ×3 轮)

| 臂 | prefill | TTFT | decode |
|---|---:|---:|---:|
| `head_quant = off` | 1596.5 t/s | 643.2 ms | **45.09** t/s(22.18 ms/tok) |
| `head_quant = rtn-int8-row` | 1594.9 t/s | 643.7 ms | **49.37** t/s(20.26 ms/tok) |
| 差 | −0.1% | +0.1% | **+9.5%** |

复测(另一 boot):off 45.25;int8 @2048 48.91 ⇒ 稳定。
TTFT/prefill 不变**符合结构**:prefill 也在**末行窄切后**调头(`forward_last` /
`prefill_chunk` 都是 `[1,d]`),头永远只吃 m=1 ⇒ 省的是纯权重读:
每 rank 1.27 GB(bf16)→ **0.635 GB**(i8),核 718.8 GB/s(§72)。

> 口径注:门内计时(int8 前/后 22.43 → 21.47 ms/tok,−4.3%)含**逐步 D2H 读
> 124160 logits 的同步**(两臂同加),把差值压小;**生产 std 的 +9.5% 才是准数**。

两条与本刀无关的观察(记在这,免得下轮白查):
1. **3080 对 × `tp27b` 档,≥4096 prompt 撞池上限**:`池 used=18671MiB / reserved=18720MiB`、
   驱动 free 1011MiB ⇒ `vram-hard-budget` 硬拒(该 turn 被优雅丢弃,不崩)。
   档的 `max_seq = 8704` 是按 24G 卡算的;20G 对得上更小的池或更低的 `vram_target`。
2. **停服后进程不退**:SIGTERM 走完优雅关停(打印 + 释放卡锁 + `exit 0`)后
   RSS 4.1GB 仍存活,需 `kill -9`(卡上内存随之释放)。别用 `pkill -f`(自杀陷阱)。

### 75.7 下一步

1. 本刀已把 `flavors/tp27b.toml` 默认开到 `rtn-int8-row`(回退 = 删该行);
2. std 五点三轮全矩阵(待 3090 对被占解除;当前 ordinal 1 被 vLLM 占着);
3. P1.5-b(读 marlin 布局 + 寄存器 perm,§74.3)仍未开工 —— 那一刀的靶是
   marlin 的 613 GB/s → 700+,与本刀(头)**叠加**后 decode 还有 ~6%。

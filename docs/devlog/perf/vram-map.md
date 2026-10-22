# 显存地图与账本盲区(3090 Ti 单卡 · 27B AWQ + DFlash2)

> 2026-10-22 · `metrics-optimize` 工作树;实测档:本地实测档。
> 实例:**RTX 3090 Ti 单卡**(BDF 41:00.0,24G)+
> **Qwen3.8-27B-AWQ-INT4** + **DFlash2 W4A16**;
> 档:`profiles/qwen38_27b-3090-w4a16-dflash-longctx.toml`(自动池 cap 117,357;
> `max_seq=67,584`);构建 `server --features perf,cupti`。
> 本文是**下一轮显存优化的底图**:先给实测分布,再列"哪些不在账上、为什么"、
> 算子家族分配器审计、A/B 旋钮与待办。

---

## 1. 24,164 MiB 的实测瀑布(P0 时 f16 口径)

> P5 后的分相/P8 int8 新数见 §1.1/§1.3;本表保留作历史底图。

2k 请求后(首次请求后 Temp 已常驻),`nvidia-smi = 23,615 MiB`:

| 项 | MiB | 说明 |
|---|---:|---|
| **总显存** | **24,164** | 3090 Ti 24G |
| 常驻权重 | **17,940** | store `Weight` 16,652 + `Unclassified` 1,288 |
| ├ 目标 27B AWQ | ~16,700 | embed 2,425 + 64 层 W4A16 marlin B ~13.2G + lm_head int4 606 + scales/zeros/norms |
| └ DFlash2 草稿 | ~1,240 | fc + 5 层 + 码本 242 + norms(文件 1.28G) |
| 主 KV 池 | **3,667** | 自动 cap 117,344 tok × 32 KiB;`max_seq` 只占 67,584 tok |
| GDN live 格(`gdn.conv_*`/`recurrence`) | **148** | 48 个 GDN 层 × 1 格(`gdn_slots=1`) |
| GDN 前缀快照(`gdn.slot.*`,`snap_max=1`) | **148** | 块边界状态副本,前缀复用/跨 turn 恢复 |
| GDN 投机快照(`gdn.ring.*`,`spec_snap`) | **148** | 每轮 verify 前拷贝,accept<8 时回滚 |
| 图 slab / 表(实际) | **139** | store `Graph` 61.9 + Pool 内图/表 77;预算常量写 320 |
| Temp(prefill 工作区) | **209** | 首请求后常驻,74 块,可复用、不增长 |
| CUDA context | **285** | 最小 `cuCtxCreate` 进程实测 |
| 模块 + cuBLAS + 图 exec + 分配器 | **~931** | 见 §4(目前是**粗账**) |
| 空闲(reserve 512 + guard 128 + 余) | **549** | |

> 上表的 GDN 合计 **444 MiB**(不是 296):live + 前缀快照 + 投机快照是三份
> **同形状、不同生命周期**的拷贝。`snap_max=0` 只去掉前缀快照;spec 档的
> 投机快照仍在。

`store` 账 = 17,939.8 + 4,187.9(Pool)+ 61.9(Graph)+ 209(Temp)= **22,398.6**;
`smi − store = 1,216.4`;其中 context 285,余 **931.4** 是本文 §4 的盲区。

---

## 1.1 P5 分相实测(2026-10-22 晚;最终代码)

boot 里程碑打 `[vram-phase]`(驱动 `cuMemGetInfo`;`used = total − free`,
`delta` = 与上一里程碑之差)。一次 67k DFlash2 档 boot:

| 里程碑 | used MiB | Δ MiB | 判读 |
|---|---:|---:|---|
| `server:context` | 438.4 | +438.4 | **CUDA primary ctx + cudarc 底座**(不是旧估的 285) |
| `server:weights` | 17,110.4 | **+16,672** | 目标 27B AWQ 常驻(含 `Unclassified` repack) |
| `env+spec` | 18,422.4 | **+1,312** | DFlash2 草稿(1.92B BF16)+ 少量环境 |
| `pool` | 21,366.4 | +2,944 | 本 boot cap 79,469 tok(池 2,483M)+ GDN/Temp |
| `graph:session` | 21,436.4 | 0 | 图 slab 被池/复用吸收;store live +64M |
| `graph:verify(d7)` | 21,510.4 | **+74** | 1 张 verify(836 节点) |
| `graph:fold(d7)` | 21,574.4 | **+64** | **单张** fold(见 §1.2) |
| `graph:dflash` | 21,646.4 | **+72** | DFlash2 propose 单图 |
| `graphs` 合计 | — | **+280** | 全部是**驱动侧 graph exec/模块**,不在块账 |

> 这一次 boot 的 cap 是 79,469(自动池抖动,见 §6.1);phase 的 Δ 与 cap 无关。
> `/debug/vram` 2k 请求后:store `gpu_used` 21,176.5 MiB + 池缓存 35.2 +
> **驱动残差 788.6** = 物理 22,000.3 MiB,free 2,164.6 MiB。
> 残差构成 ≈ context 438 + graph exec 280 + 模块/cuBLAS/分配器 ~70。

### 1.2 fold 图:7 张 → 1 张(−398M 驱动内存,AL 恒等)

fold 族原先按 `m=0..depth-1` 捕 7 张图(每张约 66M 驱动 exec)。fold 的 `m`
只影响 record/输出视图与网格 tile 数(≤32 行恒 1 tile);conv/recurrence 核
按设备侧 `cu=[0,rows]` 截断 ⇒ **按最大 rows 捕一张,回放时写 `cu`**。

| 方案 | fold 图数 | graph phase Δ | 65k prose AL | 判决 |
|---|---:|---:|---:|---|
| 逐 m 桶图(基线) | 7 | +462 MiB | 2.955 | 正确但贵 |
| eager fold(对照) | 0 | 0 | 2.955 | 正确,慢 |
| 单图 `m1 = depth` | 1 | +64 MiB | **2.175 / 2.210** | ❌ 边界错值 |
| **单图 `m1 = depth + 1`** | **1** | **+64 MiB** | **2.955** | ✅ 转正 |

**纪律(实测焊死)**:单图必须比最大 `rows` 多捕一行(`m1 = depth + 1`,
最多 8 = `SPEC_VERIFY_ROWS_MAX`);`m1 = depth` 在 `rows = depth` 的边界上
状态错(65k AL 2.18 vs 2.96)。动这个数之前先跑 65k AL 恒等门。

收益:graph phase 678 → **280 MiB**(−398M);2k 后驱动残差 1,156.6 →
**788.6 MiB**(−368M);65k AL/轮时与 eager 逐位一致(2.955)。

### 1.3 embed 表 int8(P8 首刀;opt-in)

`embed_tokens` 2,425 MiB(f16)是最大单项。新增 `[model].embed_quant = "f16" | "i8"`
(默认 f16,行为不变):
- 装载期 `rtn_i8_row_sym` 产 `q[i8, vocab×d]` + `s[f32, vocab]`,走派生键
  `...embed_tokens.i8q/.i8s`;查表核 `owl_embed_i8_f16` 边查边反量化。
- 新核有**逐位 host 参考门**(`ln_elems_bench::embed_i8_parity_gate`,
  OWL_TEST_DEVICE):`bad=0, worst=0`。
- 实测(同档,65k first):weights 相位 **17,110 → 15,926 MiB(−1,184)**;
  embed 块 **2,425 → 1,212.5 MiB**;cap 118,381(对照 boot 114,285;
  仍受 §6.1 抖动影响)。2k AL 2.879 / 65k AL 2.955 与 f16 **一致**。
- ⚠️ **质量门未过**:固定 greedy prompt 的文本在 int8 表下与 f16 分叉
  (AL 只证明 draft/verify 一致性,不是输出质量)。故默认 **f16**;
  `i8` 只在"显存优先、可接受再评测"的档位显式开:
  `server -p <档> --set model.embed_quant=i8`。
- **96k 实测**(`max_seq=98,304` + i8,同一图形/单 fold 图):97,084 prompt
  通过,AL 2.706,128 token,free 2,068.6 MiB(池 3,699M / cap 118,381)。
  ⇒ 单卡 96k 在 i8 档已落地;f16 档仍受 cap 抖动限制(好的 boot 114k 够,
  抖到 79k 不够)。
- 下一步质量门:小语料 PPL / 任务集,或 group-wise int8(每 128 列
  一个 scale,误差更小,scale 仅 ~20M)。

### 1.4 池按 `max_seq × 并发` 封顶 + fresh free(2026-10-22 晚)

两刀一起落:

1. **池按缓存开关分档**:
   - `prefix_cache=false`:池 = `min(cap, (ceil(max_seq/page)+1 页) × concurrency)`;
   - `prefix_cache=true`:池 = `cap`(整池都可被前缀缓存持有,新请求逐出)。
   67k 无缓存档池从 3,635 → **2,113 MiB**;96k 无缓存档 3,699 → **3,073 MiB**;
   dflash 16k 缓存档仍为 **2,069 MiB**(66,208 tok),缓存可占整池。
2. **池前 settle + fresh free**:`pool_cap` 原读 `owl_shared::vram::free_bytes()`
   (VramGuard 缓存原子),停止时刻随装载分配模式漂移 ⇒ cap 79k–116k。
   现在 pool 前 `sync` → `drain_temp_pool` → `mem_pool_trim` → `cuMemGetInfo`
   → `free_set`,池看到结算后的真值。

实测(同一二进制,连续 boot):

| 档 | cap(3 次) | 池 | 2k/97k 后 used | free |
|---|---:|---:|---:|---:|
| f16 67k(缓存 off) | **80,493 / 80,493 / 80,493** | 67,616 tok / 2,113 MiB | 21,648 MiB(89.6%) | 2,516 MiB |
| i8 96k(缓存 off) | **118,381 / 118,381 / 118,381** | 98,304 tok / 3,072 MiB | 21,456 MiB(88.8%) | 2,708 MiB |
| dflash 16k(缓存 on) | 66,237 | 66,208 tok / 2,069 MiB | — | — |

关键定谳:
- `temp_drained=0 块 / driver_trimmed=0 MiB` ⇒ **store 临时池和 CUDA 池缓存都不是抖动源**;
  “materialize lazy weights”不是正确修法。
- 旧 f16 高 cap 116k 是 **free 缓存停在高位**的过估(会过配池);真实 f16 cap 80k。
  f16 96k 因此不可行;**i8 省 1.2G 后 cap 118k,96k 才成立**。
- `post_pool_extra=1792M` 仍是保守预算;后续可按实测下调,不影响本轮 67k/96k。

---

## 2. 磁盘 19.5G ≠ 常驻 19.5G

| 项 | 磁盘 | 常驻 | 差异原因 |
|---|---:|---:|---|
| 目标 27B AWQ | ~19.5G(19.8GB safetensors) | ~16.7G | lm_head bf16 源(2.5G)→ int4 0.6G;AWQ 源布局重排为 marlin B,源 `packed_raw`/暂存不保留 |
| DFlash2 草稿 | 1.28G | ~1.24G | 独立检查点;bf16 passthrough |
| 视觉塔 | — | 0 | `vision=false` |

⇒ 常驻权重 **17.94G**;磁盘目标+草稿 **20.8G**;差 ~2.9G = 未保留的源布局/头。

---

## 3. 三套账本:谁记什么

| 账本 | 来源 | 覆盖 | 不覆盖 |
|---|---|---|---|
| `nvidia-smi` free/used | 驱动 | **物理真值** | 无分类 |
| store 事件/tier 账(`/debug/vram`) | 分配钩子 + `AssetClass` | Weight/Pool/Graph/Temp/Unclassified | 驱动/库内部;未标类的块 |
| `owl_shared::vram::LIVE` | server `Alloc` 的 live_add/sub | 经 server 的块 | 同上;`Unclassified` 不在其中 |
| server 块池账(`pool_ledger_snapshot`) | 默认池 used/reserved/cached | 池内块 + 缓存 | 驱动/库内部 |

**`Unclassified` 1,288 MiB 不是"绕过"**:它是 store 账里的**分类缺口**
(`repack.lm_head.packed_raw` 606 + draft/repack 输出等没打 `Weight` 标),
块本身仍经过 server 分配。`[vram-gov] live=16.26G` 不含它,所以旧预算高估;
2026-10-22 起单卡改用 **`free` 口径**,已天然覆盖,不再依赖 `live`。

**`spec_snap` 148M** 是 server 分配(DraftBank),但不在 `fixed_after` 的
`gdn_snap` 里;目前由 `post_pool_extra` 覆盖。这是**预算项透明度**问题,
不是漏分配。

---

## 4. 盲区/绕过审计(下一轮重点)

### 4.1 已知不在块账上的

| 项 | 证据 | 量级 | 能否拦 |
|---|---|---:|---|
| CUDA context | P5 `[vram-phase] server:context` | **438.4M**(含 primary ctx + cudarc 底座;最小 `cuCtxCreate` 裸进程 285M) | ❌ 驱动 |
| `CudaModule`(cubins/FI/marlin/xqa/gdn) | `exec.load_cubin` / 静态 `.a` 链接 | 文件 0.45M cubins + FI 16.7M + marlin 21.5M;加载后代码更大 | ❌ 驱动;只能减少装载/换模块 |
| `CudaBlas` 句柄/内部工作区 | `OwlCublas::new`(cudarc) | ~0.3G(4MB 预绑后 residual) | ⚠️ 已预绑 4MB;内部仍可能分配 |
| `CudaGraph` exec/upload | `graph_fleet` instantiate;`graph_flags.upload` | 未单列(与图 slab 分开) | ❌ 驱动;flag 无效 |
| cudaMallocAsync 元数据/对齐 | 默认池 | cached ~32M;reserved−used 小 | ❌ 驱动 |
| `harvest.rs` `stream.alloc::<u8>` | 直接 cudarc 分配(探针路径) | 小 | ✅ 可改走 `DeviceRes::alloc` |
| `vmm.rs` `cuMemCreate` | TP staging/VMM | 单卡不触发 | — |

### 4.2 算子家族分配器审计

- `KernelFamily::assemble(res: &mut dyn DeviceRes, exec: &mut Exec)`:
  - **workspace/scratch 都走 `res.alloc`**(`fi.float_ws`/`fi.int_ws`、
    `xqa.consts`/scratch、`gdn.soff`/`o_f32`、`cublas.ws` 4MB、
    `marlin.ws` 等)⇒ 进 server 块账。
  - **`exec.load_cubin`** 加载 `CudaModule`(xqa 4 个、gdn_chunked 6 个、
    gdn_scalar 1 个 …)⇒ **驱动模块内存,不在块账**。
  - **`CublasRuntime` 的 `CudaBlas::new`** 创建句柄/内部工作区 ⇒ 不在块账;
    已用 4MB `res.alloc` + `set_workspace` 替换内部默认工作区
    (2026-10-22;free +0.31G,65k AL 2.955 不变,130 tests 绿)。
  - `marlin` 的 `alloc_block` 返回 `unexpected`(家族不能自己块分配)⇒ 好。
  - **家族代码里没有 `stream.alloc`/`CudaSlice` 直分配**(审计 `grep`;
    只有 `harvest`/tests/examples)。
- **结论**:`DeviceRes` 只拦得住"显式向 server 要块"的路径;
  `Exec::load_cubin`、`CudaBlas`、`CudaGraph`、CUDA context 是**库/驱动内部**,
  拦不住,只能预分配/裁剪/计入固定项。这就是"统一分配器仍漏账"的边界。

---

## 5. 旋钮 A/B(67k 档,2k 请求前后)

| case | smi boot | after 2k | cap | Temp | 结论 |
|---|---:|---:|---:|---:|---|
| base | 23 101 | 23 487 | 113 261 | 209.0 | — |
| `cuda.graph_flags.upload=false` | 23 101 | 23 487 | 113 261 | 209.0 | **无变化** |
| `pool.pool_cap_mb=64` | 23 005 | 23 359 | 110 189 | **59.4** | Temp −150M;AL/速度不变 |
| `pool.vram_reserve_mb=768` | 22 941 | 23 327 | 108 141 | 209.0 | **峰值 −160M** |
| `cuda.graph=false` | — | — | — | — | ❌ DFlash2 需要 verify 图 |
| `cuda.capture_slab_mb=256` | — | — | — | — | ❌ 池分配差 4B |
| `CUBLAS_WORKSPACE_CONFIG=:4096:8` | 22 069 | 22 455 | 79 725 | 209.0 | boot-order 抖动,非稳定收益 |
| cuBLAS 4MB 预绑(已落码) | 23 229 | 23 615 | 117 357 | 209.0 | free +0.31G;AL 不变 |

> **自动池会把省下的显存全部吃掉**:降 Temp/cuBLAS 只会让 cap 变大,`smi`
> 峰值基本不变。要峰值真降,只能加 `vram_reserve_mb` 或降 `max_seq`
> (池仍按 cap 预分配;`max_seq` 只影响准入)。`reserve=768` 峰值 −160M,
> 96k 仍可(cap 108k > 98k);`reserve=1024` cap 64k,67k 档会拒启。

---

## 6. 已知未决

1. ✅ **自动池 cap 抖动已结案(2026-10-22 晚)**:根因不是 lazy weights,而是
   `pool_cap` 读 **缓存的 `vram::FREE`**;池前 settle + `cuMemGetInfo` 刷新后,
   f16 cap 稳定 **80,493**、i8 稳定 **118,381**(各 3 次 boot)。旧 116k 高 cap
   是过估。
2. **0.9G 残差已拆到 ~788M**:context 438 + graph exec 280 + 模块/cuBLAS/
   分配器 ~70(2026-10-22 P5)。剩下的大头已不是“未知”,而是**不可压缩的驱动底座**。
3. **图预算 320M vs 实测 store 图 256M**:`GRAPH_SLAB_BUDGET` 有 ~64M 余量;
   自动池按 320 扣,少给 ~2k token。降到 256M 需先证明 MTP/adaptive 多档不超。
4. **`spec_snap` 未进 `fixed_after`**:148M 在 `post_pool_extra` 里,账目不透明。
5. **4K/8K 草稿窗口**:环容量/槽映射已数据驱动;NC 核 `OWL_NC_SLIDING_WINDOW`
   仍是 2048,`[spec].draft_window=4096` 会 boot 拒(等 propose 核加运行时
   window)。
6. **单卡 190k**:主 KV fp8 6.2G + 权重+开销 >24G;需 int4 主 KV 或 TP2。
   embed 表 int8 **已落地(opt-in,省 1.18G)**;质量门(PPL/任务集)未过,
   默认仍 f16。int4 主 KV 是 190k 的下一根杠杆(16 KiB/tok)。
7. **fold 单图边界**:`m1 = depth + 1` 是实测纪律(§1.2),不是可随意优化的
   常数;改 fold 图拓扑/行数必须跑 65k AL 恒等门。
8. **配置面(2026-10-22)**:`prefix_cache_share` 已退役 —— 前缀缓存可占**整池**,
   新请求按 `ensure_for_len` 逐出设备叶到宿主,宿主不够再走磁盘;
   `kv_fp8` 从 `[kernel]` 移入 `[pool]`(池存储 dtype/几何归池面)。

---

## 7. 下一会话计划(P5 已完成,P6 起)

1. **P5 ✅ 阶段显存采样**(2026-10-22):`[vram-phase]` 已落 `apps/server` +
   `boot_rank`/`GraphFleet`;`/debug/vram` 加 `device` 块
   (`free/total/pool_cached/residual`)。结论见 §1.1。
2. **P5.1 ✅ fold 单图**(2026-10-22):7 → 1,−398M;纪律见 §1.2。
3. **P6 家族分配门**:把 `Exec::load_cubin`/`CudaBlas`/`CudaGraph` 的
   分配统一登记(或至少在 boot 日志报字节);加一个 source lint:
   家族 `server/*.rs` 禁止 `stream.alloc`/`CudaSlice`(当前审计为 0 命中)。
4. **P7 账本收敛**:✅ 池按 `max_seq×并发` 封顶 + 池前 settle/fresh free
   (2026-10-22 晚,§1.4);余项 = `spec_snap` 进 `fixed_after`、图预算按实测校准、
   `harvest` 直分配改走 `DeviceRes::alloc`。
5. **P8 大杠杆**:embed 量化(2,425M;int8 首选,省 ~1.2G)、int4 主 KV
   (190k 路)、draft 码本 fp8、4K/8K 窗口(等 propose 核运行时 window)。
6. 每项按 `docs/devlog/dev/profiling.md` 的口径验收:同卡同时段、同二进制
   开关、两遍噪声界;峰值用 `nvidia-smi` + `/debug/vram` 双证。

---

## 8. 下一会话 handoff(2026-10-22)

- **当前策略**:`prefix_cache=false` → 池 = `min(cap, max_seq×并发+1页)`;
  `prefix_cache=true` → 池 = `cap`(整池给缓存,新请求逐出 host/disk)。
  池前 `sync/drain/trim` + fresh `cuMemGetInfo` 已消掉 cap 抖动。
- **下一步**:3090 单卡测 **kvmem 选择层**(`[kvmem]` overcommit/workspace/
  select_* + host/disk 腿)与 **GGUF 格式**的内存管理(GGUF 权重/KV 池/
  量化表);两者都先看 `/debug/vram` 的 `Pool/Temp/Graph/Unclassified`
  四类与 `kv.slots_free/cached_blocks`。
- 本卷数据:`§1.4` 池封顶 + fresh free;`§1.3` embed int8(opt-in);
  `§1.2` fold 单图。原始 run 在 `perf-runs/metrics-optimize/`(gitignored)。

## 附:产物

- 原始数据:本地实测档(`p5-*`/`p5b-*` P5 分相、`fold1-*`、`perm-*`、
  `eager-*`、`fold8-*` fold A/B、`mem2-*` A/B、`cublas-ws`、
  `vram-timeline-2037`;汇总 `P0-P3-2026-10-22.md`、`P5-fold-2026-10-22.md`)。
- 相关代码:`crates/engine/src/state.rs`(FixedAfter/post/guard)、
  `crates/kernels/src/server/cublas.rs`(4MB 预绑)、
  `apps/server/src/debug.rs`(`blocks_top`)。

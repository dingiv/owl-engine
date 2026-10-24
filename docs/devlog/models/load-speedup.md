# 装载提速 · 定谳轮(27B AWQ,2026-10-16)

> 用户令:「继续优化模型 loader 的性能」。
> 本卷 = 定谳过程 + 五项修复 + 门与证据 + 挂账;坑册对应
> `pitfalls-general.md` §40(判定口禁副作用)/ §41(逐键排空);
> 前序:§12/§13(转换上 GPU + 单面装载)、§14(装载器解释器化)、
> `stream-ordering.md`(E1-E4,复合口/事件序)。

## 1. 结果

| 口径(engine 27B AWQ 单卡整装,release) | 提速前 | 提速后 |
|---|---|---|
| **权重装载** | **12.8s**(5 次 13.14~13.29s,±0.07) | **5.6~5.7s**(3/3 次 ±0.07)= **2.25×** |
| 逐键相位账(键循环内) | 11.12s(convert 10.84s) | 5.42s |
| 装载期暂存(设备常驻) | 89MB | 156MB |
| 设备显存峰值 | 20927MiB | 20927MiB(不变) |

口径:engine e2e 27B 档(`OWL_E2E_ONLY=gpu_awq27b_marlin_e2e`),
`[e2e] 27B 权重装载` 行 = `loader.load_qwen38_27b_awq()`(引擎构造与装配各自单列)。

## 2. 定谳方法(两条探针,先于结论)

1. **逐键双账**(`OWL_LOAD_DEBUG=1`):`KeyRecord{wall, total, phases}` ——
   `total` = 各 span 之和,**wall** = 臂函数墙钟;两者之差 = 未开 span 的段
   (分配/复合往返/排空),由 `StderrTap` 逐键上屏。
   本轮四轮定位全靠它把"整载慢"逐层收窄到具体一段。
2. **微基准**(`OWL_AWQ_IOBENCH=1`,`awq.rs` 内两个 env 门控测试):
   `bench_raw_copy_paths`(mmap 直拷 / 合并键 / 派生键)+ `bench_scales_chain`
   (旧链 vs 索引表 vs 单遍融合),每段带 `/proc/self/io` 磁盘读增量
   (区分"页缓存命中"与"真磁盘读")。

## 3. 基线账(release,12.8s 里 11.12s 在键上)

| 类 | 键数 | 字节 | 耗时 | 均速 | 病因(实测) |
|---|---|---|---|---|---|
| `.scales`(派生) | 352 | 1.52GB | **3.42s** | 0.21GB/s | 链条三分配两拷贝:`bf16_par`(11MB f32)+ `pack_marlin_s`(5.6MB u16)+ `u16s_le_bytes`;且每元素两次运行时整数除法 |
| `.packed_raw` 合并(in_proj_qkvz) | 48 | 2.01GB | 3.90s | 0.52GB/s | **每键新建 41.9MB 中间 Vec**(anon 首触 75ms)+ 二次拷贝;两子基直拷实测 **2.6ms** |
| `.weight`(embed/lm_head BF16→F16) | 2 | 5.09GB | 1.28s | 1.5~3.3GB/s | CPU F16C 单线程;`AwqSource` 无 `native_dtype` ⇒ 设备 cast 臂不命中 |
| `.packed_raw` 普通 | 304 | 10.15GB | 2.01s | 5.06GB/s | 单线程 mmap→pinned |
| `.zeros`(派生) | 352 | 0.38GB | 0.90s | 0.42GB/s | `i32s_par` → `unpack_zp_ct` → `pack_marlin_z` 四遍 |

另有**两笔不在键账里的**:①声扫(`LoadStaging::plan`)被"判定口调物化"拖成
3.5s;②直接臂**逐键** `face.sync()`(1050 个小键)在 CPU 提速后变成 5.9s 的
"等设备追平"。

## 4. 五项修复(全在装载面,零模型面)

1. **合并虚拟键零中间缓冲**(`formats/awq.rs`):
   `Resolved::Merged(Vec<RawEntry>)` —— 两子基 `weight_packed` 视图按序交给
   消费侧,`convert_chunk_into_bytes` 按 dst 偏移**直写**(支持任意跨段窗口);
   3.90s → 0.21s。门:`merged_packed_raw_direct_write_parity`(含整键/头部/
   跨段/尾部/乱序窗口 + 越界响亮拒)。
2. **scales 单遍融合**(`formats/awq.rs::pack_scales_bf16_into`,
   `AwqSource::pack_scales_bytes`):索引表 gather + bf16→f16 + LE 字节直写
   `dst` —— 一分配一路过;微基准 7.4~21ms → **0.7~1.1ms/键(5.1~7.6GB/s)**。
   与旧链**逐位等价**(bf16→f32 零舍入展宽 + 同一 `f16::from_f32`),
   门:`scales_fused_bytes_parity`(单段/两段)。
3. **`pack_marlin_s` 索引表化**(`kernels/family/marlin/repack.rs`):
   `marlin_scales_index(out, groups)`(每形状一次除法分解)+
   `pack_marlin_s_idx` 纯 gather;`pack_marlin_s` 保留为壳(二者同实现,无第二套)。
   门:`marlin_scales_index_pack_parity`(对冻结参考式逐位)。
4. **`AwqSource::{native_dtype, copy_native_into}`**(原字节通道):
   embed/lm_head(2.54GB ×2,BF16→F16)落**设备 cast 臂** —— CPU 只 memcpy
   (公开口 `mmap::copy_par`,≥8MB 四线程),cast 在 GPU;1.28s → 0.57s。
   两条纪律(见坑册 §40):判定口**必须廉价**(声扫对每个 Want 问一次,内部
   禁调 `resolve` —— 调了会当场物化全部派生键,实测整载多花 3.5s),且与
   `copy_native_into` **同源判定**(`Some` ⟹ 字节直拷必成,否则装载臂硬错)。
5. **逐键排空 → 按在飞字节排空**(`formats/load.rs::LoadStaging::note_inflight`):
   判据从 `off.is_multiple_of(chunk_elems*4) || off >= n`(每键一次全设备
   sync)改为**跨键累计在飞字节** ≥ `DRAIN_BYTES`(=4×CHUNK_BYTES)才排;
   键循环 host 8.3s → 6.2s。正确性三层兜:server 侧 pinned 租约**流完成回调
   才还池** + 复合口 `upload_and_launch` 的设备端事件序 + `eval_load` 末尾总栅栏。

## 5. 暂存区律修订(本轮最贵的一课)

cast 臂的**声明峰值 = embed/lm_head 的 2.54GB**,若直接按"声明峰值→槽规"
把它算进环尺寸,整载期暂存会被撑到上限(实测 2×134MB ⇒ 27B 在 layer 38/49
处 OOM;cudaMemPool `RELEASE_THRESHOLD=UINT64_MAX` ⇒ **cached 块不还 OS,
暂存即净占用**)。改为:

- `StageArm` 收敛为 **`RawIn`**(cast 输入 + 重排输入 —— 同一语义"原始字节上卡,
  设备侧消费者读它",**共用一环**)与 **`CastSegOut`**(流式分段输出);
- 槽规按**两栏账**合成:`min(不可流式栏, 上限)`;**可流式栏不参与撑槽**;
- 27B 档 = `raw.in 44.6MB×2 + cast.seg.out 33.6MB×2 = 156MB`;
- **段规实测拐点**:16.8MB 段 host 拷贝只有 5.6GB/s 且整载抖动大(5.7~8.4s),
  33.6MB 段回到 10GB/s 且抖动归零(5.59/5.65/5.73s)—— 小段不是"更省更稳"。

## 6. 门(全绿)

- **engine e2e 全场景 18/18**(0.8B 六场景 + 调度/GDN/fuse/FI/w4a16-marlin/
  spec 恒等 + 27B 五场景 + DFlash2 恒等);
- **`OWL_LOAD_VERIFY=1` 硬门**:`checked=1408 skipped(rearrange)=803 mismatch=0`
  + 351 个 `.packed_raw` 键 **GPU 重排 vs CPU fused 逐字 0 坏字**;
- **models**(release,`--test-threads=1`,GPU 门 63/65):`gpu_real_weights_smoke`、
  `gpu_vram_manifest_checksum`(302 键回读)、`gpu_real_image_e2e_matches_hf`
  (**四图 top-1 280/5388/74035/21655 逐项不变** + 贪心逐 token 同 HF)、
  `gpu_dflash2_draft_load_bench`(fc CPU-vs-GPU **0 坏字/3276800**);
  CPU/合成 159/159。两条 pre-existing 红:`gpu_dflash2_27b_loads_and_smokes`
  (`1.8408203 vs 1.9783204 @atol 0.05` —— 前后 A/B 逐位相同)、
  `gpu_dflash2_golden_matches`(worktree 缺 `crates/models/testdata` 金标);
- 测试方法学(两条坑):models 的 GPU 套件**必须串行**(libtest 默认并发 ⇒
  多个 21GB 装载同卡抢显存,假红在 7~20MB 分配处);DFlash2 目录两语义
  (`z-lab/…-DFlash2` = BF16 golden 参考系,`syvai/…-DFlash2-W4A16` = 量化档)。

## 7. 挂账(下一刀)

1. `.zeros` 派生链同款融合(`i32s_par` → `unpack_zp_ct` → `pack_marlin_z` 四遍,
   照 scales 做"索引表 + 单遍直出字节",估 −0.5s);
2. `.scales` 每键仍余 ~6ms = zeros 构建 + 5.6MB 新堆页(scratch 复用可再省);
3. embed/lm_head 流式 76 段:段规受 `CastSegOut` 32MB 限,再抬收益小;
4. `pinned_alloc` 冷启动 30ms/键(layer0 首现键)⇒ 池预热/按尺寸类预分配另案;
5. `LoadStaging` 的 `RawIn` 共享环已就位,但**重排臂仍无流式路**(超槽 = 响亮拒),
   若将来出现 >128MB 的原始 packed 键需补流式(否则大键撞 `over_cap_msg`)。

---

## 8. 追加:a8 档(scales_a8)入同一律(2026-10-17)

W4A8 档(`[model] quant = "w4a8"`)的 `scales_a8` 物化是**首版 naive 链**
—— 正是本节 §3 那条 `.scales` 病历的**同型复发**(`bf16_par` 中转 + 标量
双循环转置 + 逐 32-chunk `to_vec` perm + 独立 normalize)。27B 单键 ~65ms ×
~150 键 ≈ 10s ⇒ a8 档整装 26.69s(a16 11.88s)。

**修法 = §4 第 2/3 条的同律复用**(新 perm 表的索引表 + 单遍直出字节 + bf16 直读
求 max),产物:`marlin_scales_index_a8`(kernels)、`bf16_parts_max` /
`pack_scales_a8_bf16_into`(models/awq.rs);旧链 `pack_marlin_s_a8` 留作
**冻结参考**(两条逐位门锁它)。

| 口径(3090 Ti,同档同卡) | a16 臂 | a8 臂修前 | a8 臂修后 |
|---|---|---|---|
| 权重装载 | 11.88s | 26.69s | **14.39s** |
| 物化(键账) | 1.96s | ~22.6s | **4.90s** |
| a8 单键均值 | — | ~70ms | **13.5ms** |
| Δ vs a16 | — | +14.5s | **+2.5s** |

⇒ **−12.3s**。细节、门与“为什么不做 a8 设备重排核”的判据见
`w4a8-family.md` §19。

**教训(第 3 条同型复发)**:新增量化档 = 新增一条**派生键链** ⇒ 必须当场
套用本轮的“索引表 + 单遍直出字节”口径,别让首版 naive 链上车(它会以
“装载慢”而不是“报错”的形态潜伏)。

---

## 9. 冷盘装载重叠 + BF16→int8 head 融合(2026-10-24)

> 触发:用户实测 27B 启动约 **20s**。复现方式 = 对模型目录逐文件
> `posix_fadvise(DONTNEED)` 后启动;单 3090 Ti / `qwen38_27b-3090-w4a16`
> (AWQ-INT4,无 spec)release 档,从进程启动到 `/health` 就绪计时。

### 9.1 定谳:20s 是冷 page cache,不是权重数学

| 口径 | 键循环 | 权重/tokenizer/rope | 启动到就绪 |
|---|---:|---:|---:|
| 冷盘(修复前) | **17.13s** | 18.23s | **18.96s** |
| 热缓存(修复前) | 7.60s | 8.67s | 9.23s |

冷盘比热缓存多 **9.5s**,全部是 mmap 缺页等盘;CPU 侧的数
(`lm_head.i8q` 2.87s、`scales` 2.27s、`zeros` 0.72s、`packed_raw` 1.32s)
在冷盘时被 I/O 串行放大。单流 buffered `read` 实测 ~2.1GB/s,而模型实际读
**19.25GB** ⇒ 仅靠页缓存按需缺页无法在 10s 内完成。

### 9.2 两刀

1. **后台预取源区间**(`WeightSource::prefetch` / `prefetch_finish`):
   - `AwqSource` 只读索引/`linear_form`,收集本档将读的源区间(不 `resolve`,
     避免提前物化),按文件偏移合并;
   - `Mmap` 保留 fd,新增 `readahead()` 用 `pread` 填页缓存;
   - 装载键循环开始前启动后台线程,循环结束 `join`;冷盘 I/O 与 CPU 物化重叠;
   - 线程数实测:**2 条最优**(文件 0/2/4 与 1/3 轮转)。1 条 I/O 不饱和
     (键循环 8.93s),3/5 条 NVMe 队列争用(10.31/10.44s);按字节贪心重排
     会打乱文件顺序(9.84s)。
2. **BF16→int8 head 直读融合**:
   - 旧链 `bf16_par` 先展开 5.1GB f32,再 `rtn_i8_row_sym` 逐元素除/round,
     `lm_head.i8q` 单键 **2.87s**;
   - 新 `rtn_i8_row_sym_bf16` 两遍直读 BF16(行 amax + 量化),不建 f32 中间态;
   - 与旧链**逐位同源**(`i8_head_bf16_fused_matches_f32_chain`);
   - 热载键循环 7.60s → **6.52s**。

### 9.3 结果(同机、同档、release)

| 口径 | 修复前 | 修复后 |
|---|---:|---:|
| 冷盘键循环 | 17.13s | **8.30–8.99s** |
| 冷盘权重装载 | 18.23s | **9.42–10.16s** |
| **冷盘启动到就绪** | **18.96s** | **10.2–10.8s**(3 次) |
| 热缓存启动到就绪 | 9.23s | **8.20s** |

真机冷启后 `只回答:OK` 生成正常;预取线程在装载返回前 join,不带进服务期。

### 9.4 边界与挂账

- 预取在 **单卡** 走全量;TP2 只让 **rank0** 预取全量一次(两 rank 切片
  的并集 = 全量),rank1 吃页缓存 —— 若两 rank 都预取会读 2× 数据。
- GGUF / SafeTensors 源尚未覆写 `prefetch`(默认 no-op);27B GGUF 冷载仍是
  按需缺页。
- DFlash2 档冷启实测 **13.84s**(draft 权重 1.80s + 图捕获/装配余量),已优于
  原 20s,但离 10s 还差图捕获/草稿装载两刀。
- 剩余 CPU 项:`.zeros` 四遍链融合(~0.3s)、`.scales` 随机 gather 分块
  (~1s 潜力)、`pinned_alloc` 首现键 30ms 级池预热。

### 9.5 TP2:双 rank 并行装载(2026-10-24)

**病根**:`actor_main` 的 TP 分支先 `load_shard(rank0)`、再构造 rank1 并
`load_shard(rank1)`,两 rank 的切片源互不相交却完全串行。冷盘实测:

```text
rank0 eval_load 19.64s(冷) + rank1 eval_load 9.23s(rank0 已预热)
TP2 分片装载 31.44s · 启动到就绪 37.86s
```

**修法**:

1. **`Weight.dtype` 从 `Cell<Dtype>` 改 `AtomicUsize`**(Relaxed)
   —— 原来它让 `Model: !Sync` ⇒ `LoadedModel` 不能跨线程返回;改成原子后
   `Model: Send + Sync`,`Weight` 手写 `Clone`(原子值搬运)。
2. **rank0/rank1 两条 OS 线程并行 `load_shard`**:
   先建 rank1 引擎壳(账本在装载前合流),再 `std::thread::scope` 各起一条
   `current_thread` Tokio runtime;GPU actor 在独立线程,不会被 `actor_main`
   阻塞拖死。两 rank 的 CPU 物化与页缓存 I/O 真正重叠。
3. **TP2 只让 rank0 预取全量源区间**(19.25GB 一次),rank1 不重复预取。

**实测(2×3080,同 profile;冷盘 = fadvise DONTNEED 两个模型目录)**:

| 口径 | 串行(修复前) | 并行(修复后) |
|---|---:|---:|
| TP2 分片装载(冷) | 31.44s | **15.42s** |
| 启动到就绪(冷) | **37.86s** | **24.58s** |
| TP2 分片装载(热) | 22.24s | **13.52s** |
| 启动到就绪(热) | **26.14s** | **21.52s** |

并行后两 rank 的 `eval_load` 热载各 ~11.5/11.8s(CPU 共享 48 核,单 rank
不再独占);装载墙钟从串行的 22.2s 降到 13.5s(热)/15.4s(冷)。启动墙钟
冷 −13.3s / 热 −4.6s。

**仍未闭环**:

- `TpEngine::assemble_with_staging` 的两次 `boot_rank` 仍串行(图捕获热载
  ~7s)。不能直接并行:`boot_rank` 的池容量读全局 `vram::FREE`(单值,非
  per-device),两 rank 同时 `StatePool::alloc` 会互相覆盖 free 读数。要并
  行装配需先把 FREE 改 per-rank 或把池分配与图捕获拆开。
- TP2 的 `eval_load` 仍被两 rank 共享 CPU 拖到 ~11s;进一步要靠每 rank
  限流 rayon / 分阶段 CPU-I/O 流水,不是简单加线程。
- TP2 未跑满速推理回归门;当前只做了 boot + `只回答:OK` 冒烟。

> **2026-10-24 续**:上述前两条已由 §10 处理(per-device 账 + `boot_rank`
> join;草稿 CPU 化仍待与主载重叠)。本节数据保留作 before 基线。

---

## 10. TP2 装配并行 + per-device 显存账(2026-10-24 续)

> 承接 §9.5。目标:解除 `boot_rank` 串行的 P0 阻塞,并继续压 TP2 启动。
> 结果:**热 21.52 → 19.17s,冷 24.58 → 21.36s**(同 profile、same-host)。

### 10.1 P0:全局 `owl_shared::vram` → 每设备 `VramAccount`

原 `TOTAL/LIVE/FREE` 是进程级三原子,TP2 两 rank 并行 `boot_rank` 会互相
覆盖 `FREE`、把两卡 `LIVE` 相加 ⇒ 池容量错值。落地:

- `owl-iface::contract::{VramAccount, VramSnapshot}`(类型权威下沉;原子账 +
  只读快照);
- `GpuServer` 与 `GpuClient` **共持同一个 `Arc<VramAccount>`**(`spawn_with`
  创建、构造期注入;手工 `GpuClient::new` 仍得独立空账);
- `GpuCtx` 持设备账:alloc/free/slab/peer/`mem_info` 全部落本卡账;
- `VramGuard` 的 live/free 判据改读本设备账(全局函数仅保留为**诊断聚合镜像**);
- `DeviceClient::vram_snapshot()` + `DeviceHandle` 显式转发;
- `BootVram` 改为 per-device 快照(start 时的 free/total/live);
- `PoolPlan.vram` 携带本 rank `total/free/live`,`pool_cap` / `[vram-gov]` 不再
  读全局;
- `TpEngine::assemble_with_staging` 两 rank 各取**自己 face 的**池前 live
  基线,`weights_per_card` 由两 rank 各自 load 增量求平均。

**结果**:两 rank 的 `[vram-gov]` 各报 `free≈7.8G`、`cap≈44.4 万 tok`,
不再出现"rank1 看到 rank0 池"的顺序相关。

### 10.2 双 rank `boot_rank` 并行

`assemble_with_staging` 的两次 `boot_rank` 从顺序 `await` 改为
`tokio::join!`。装配期 AR 仍被 warmup 摘掉,两 rank 零协调。

注意:当前是**同一 async task 的并发**,不是两条 OS 线程;CPU 侧图装配仍
共享执行线程。真正的 OS 并行被 `slab_hint`(METer/HINT 进程级单值)挡住,
见 §10.5 挂账。

### 10.3 DFlash2 草稿源预取 / W4A16 零拷贝修复

- `W4A16Source` 补 `prefetch`/`prefetch_finish`(按索引收集
  `weight_packed/weight_scale` 区间,2 读流后台 `pread`);
- W4A16 的逐键 `DONTNEED` 停用:草稿源在 TP2 两 rank 间共享页缓存,逐键
  丢弃会把 rank1 打回冷盘;0.9GB 级 clean page 交给内核按压力回收;
- `ShardedSource::full_bytes` 的 `Ok(None)` 原先被 `?` 静默丢弃后仍
  `expect("刚填过")`(潜在 panic);改为显式 `is_none() => Ok(None)`。

热载草稿 4.75s(与之前 5.05s 同量级,略降);冷载 4.99s。预取主要消除
冷盘抖动,CPU 物化仍是主项。

### 10.4 BF16→int8 head 量化 AVX2

`rtn_i8_row_sym_bf16` 两处提速:

1. 行 amax 改**位型整数 max**(BF16 正域编码单调;清符号位取 u16 最大),
   不再逐元素展开 f32 + abs;
2. 第二遍 `v/s → RN-even → clamp ±127` 走 AVX2(运行期检测,非 x86/尾部
   回落标量;与标量逐位同门,新增 ties-even/非 8 倍数 k 单测)。

单卡热载键循环 6.47 → **6.24s**;TP2 热载主键 11.5 → **10.9–11.1s**
(两 rank 共享 CPU,收益被摊薄)。逐位门:
`bf16_i8_row_avx2_matches_scalar_including_ties`。

### 10.5 当前实测

| 口径 | §9.5 | 本节 | Δ |
|---|---:|---:|---:|
| 单卡 27B 热启动 | 7.75s | **7.49s** | −0.26 |
| TP2 分片装载(热) | 13.52s | **12.82s** | −0.70 |
| TP2 启动(热) | 21.52s | **19.17s** | **−2.35** |
| TP2 分片装载(冷) | 15.42s | **14.90s** | −0.52 |
| TP2 启动(冷) | 24.58s | **21.36s** | **−3.22** |

> 此表是 §10.2–10.4 阶段的中间结果；§10.7 再修 `ShardedSource` 全键拷贝后
> 降到热 **15.57s** / 冷 **17.99s**。

冷盘仍比热多约 2.2s(主键 13.2 vs 11.0);草稿冷 5.0s、热 4.8s。

### 10.6 仍未闭环 / 下一步

1. 目标 15s 还差 ~4s。最大两块 = **主模型两 rank CPU 物化 ~11s** +
   **草稿装载 ~5s**(串行发生在主载之后)。
2. 最优先:把 DFlash2 草稿装载**与主模型装载重叠**(草稿不依赖主权重,
   只需 `ModelSpec`;需把 `load_spec_runtime` 从 `boot_rank` 前移到装载阶段,
   用独立 face/clone 并发跑)。理论可把 5s 藏进 11s,TP2 热启动落到 ~14–15s。
3. 次优先:把 `boot_rank` 的 `tokio::join!` 升级为两条 OS 线程。前置 =
   `slab_hint` 的 `METER/HINT` 从进程级单值改为按 rank/设备隔离(当前会串
   hint,不能直接开)。
4. `lm_head` CPU int8 量化仍是单 rank ~1.2–2.3s;设备核
   `owl_quant_i8_row_w_{scale,q}_bf16` 已有 bench,尚未接装载路径。接入可
   再省 1–2s,但要新增 `i8q` 的设备派生装载臂。
5. `into_running` 的 `assemble_budget` 仍用全局 live / world 做 host 预算
   (此时 boot 已结束,无并行写;非 P0)。要彻底去全局,后续把两 rank
   `VramSnapshot` 一并传入。

### 10.7 二次定谳:双 rank 不要再走 `ShardedSource` 的“全键缓存”拷贝

> 用户追问:“双 rank 为什么比单 rank 多那么多?参考单 rank 的做法。”
> 对拍单/双 rank 逐键账后定谳:**主因不是锁,是双 rank 给每个键都多做了一次
> `vec![0; 整键] + copy`。**

单 rank 路径:`ctx.shard == 1` ⇒ `build_shard_pass` 返回空 spans ⇒
`ShardedSource` 全直通,`packed_raw` 从 mmap 直接 `copy_par` 到 pinned。

双 rank 路径:每个键都有 span ⇒ `ShardedSource::convert_chunk_into_bytes`
旧判据:

```rust
if ent.src_elems * esz <= FULL_CACHE_CAP { // 256MB
    full_bytes(...)   // vec![0; 整键] + 整键拷贝
    // 再按 span 拷到 dst
}
```

这个判据本意只治 **k 切键**(5120 段,逐段调内层会反复物化),但它对
**行切/单段键也生效**。实测同一键:

```text
single layers.63.self_attn.q_proj.packed_raw  31.5MB  convert 1.2ms
dual   layers.63.self_attn.q_proj.packed_raw  15.7MB  convert 19.0ms
```

半量数据反而慢 15×。逐键中位数 dual/single = **2.10×**;双 rank 主键
11.5s 里很大一块是这笔中间 `Vec` 的分配/清零/二次拷贝。

**修法**:按**段数**分流,不再按大小:

```rust
let needs_full_cache = ent.src_elems * esz <= FULL_CACHE_CAP && seq.len() > 8;
```

- 1–8 段(绝大多数行切/直接键):逐段直接写 `dst`,与单 rank 同路;
- >8 段(k 切/多段键):才整键缓存一次,避免 5120 次内层物化。

**实测(2×3080,同 profile)**:

| 口径 | 修前 | 修后 |
|---|---:|---:|
| 主键热载 / rank | 10.9–11.1s | **8.11s** |
| TP2 分片装载(热) | 12.82s | **9.96s** |
| TP2 启动(热) | 19.17s | **15.57s** |
| TP2 分片装载(冷) | 14.90s | **12.10s** |
| TP2 启动(冷) | 21.36s | **17.99s** |
| 草稿热载 | 4.75s | **4.19s** |

逐键 dual/single 中位数 **2.10× → 1.33×**。剩下的多段 MLP down_proj
仍走整键缓存(约 12–18ms/键),是下一层可选题;当前不再阻塞。

**结论**:双 rank 不应该“因为分片就所有键都走全键缓存”。单 rank 的直通
路径是正确基线;只有真正多段的 k 切键才需要整键缓存。

### 10.8 草稿也走单卡主模型的零拷贝路:DFlash2 4.2s → 1.2s

> 用户点:“草稿也可以写算子来加载，仿照单卡主模型；不一定要并行。”

DFlash2 草稿其实**已经**在用 GPU 算子：`packed_raw` 走
`U32/DeviceRearrange { op: CtRepack }`，GPU 核是 `owl_ct_repack_u32`。
问题不在“没用 GPU”，而在**喂给 GPU 的输入是 host 先整键物化过的**：

```rust
// W4A16Source::resolve(.packed_raw)
let __b = self.linear_bytes_for(base, key)?; // 内部:
self.maps[..][pe.start..pe.start + pe.nbytes].to_vec() // 整键 44.6MB
// 然后 ShardedSource 再从这个 Vec 切本 rank 的 22.3MB
```

于是每个草稿 MLP `packed_raw` 都是：

```text
mmap 整键 44.6MB
  → to_vec 整键 44.6MB       (host 分配+拷贝)
  → 切半 22.3MB 到 pinned    (第二次拷贝)
  → H2D + GPU CtRepack
```

单卡主模型的 AWQ `packed_raw` 不是这样：它 `Resolved::View` 直接
`copy_par` 本 rank 行段，没有整键 `Vec`。草稿仿照即可。

**修法**：

1. `W4A16Source::resolve` 的普通 `packed_raw` 返回 `Resolved::View(weight_packed)`
   （fc 五拆/合并虚拟键仍走构建路径）；
2. `W4A16Source::convert_chunk_into_bytes` 的 `View` 臂补 **I32 → U32**
   `copy_par` 直拷（此前只支持 F16/BF16，所以 View 会返回 `None`）；
3. 行切键 `seq.len() == 1` 直接写 pinned；只有 >8 段 k 切键才走整键缓存。

**实测**：

| 口径 | 修前 | 修后 |
|---|---:|---:|
| DFlash2 草稿热载 | 4.19s | **1.19 / 1.64s** |
| TP2 主载(热) | 9.96s | **9.85s** |
| TP2 启动(热) | 15.57s | **12.92s** |
| TP2 启动(冷) | 17.99s | **14.86s** |

现在启动结构：

```text
主模型双 rank host 键循环  ~9.9s(热)/ ~11.8s(冷)
DFlash2 草稿              ~1.2–1.6s
TP2 staging + 池 + 图捕获  ~1.2–1.5s
```

草稿不再是主项；它已经走 GPU CtRepack，只是之前 host 多喂了一整份
`Vec`。这也验证了“先对齐单卡主模型的直通路径，再谈新算子”的判断。

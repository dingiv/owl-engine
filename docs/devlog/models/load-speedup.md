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

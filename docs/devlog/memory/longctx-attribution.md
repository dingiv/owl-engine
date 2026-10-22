# 长上下文性能归因 —— 首轮(2026-10-15;perf worktree 会话)

> 目标:回答"长上下文的 prefill/decode 变慢,到底是哪些算子/步骤在耗"。
> 方法:`llm_speedtest`(黑盒)+ 引擎自带仪器(host 分相 → CUPTI 逐核)。
> 实例:`工作树 perf(旧位置)` 自建二进制(`--features perf,cupti`,
> commit `c18c0be` + 本 worktree 配置),**GPU2 单卡 3090 Ti**,生产形状
> (flavor `spec-dflash`:fp8 池 + XQA + DFlash2 d7),几何放大到 40960/41216。
> 原始数据:本地实测档(未入库)(host 档)、本地实测档(未入库)(CUPTI 档)。

---

## 1. 黑盒曲线(单并发,输出 256 tok,`sampling` 档)

| ctx | TTFT | prefill t/s | decode t/s | ITL | prefill/chunk | spec 轮时 |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 373 ms | 1379 | 89.0 | 11.3 ms | 364.1 ms (n=1) | 44.25 ms |
| 2048 | 1513 | 1355 | 97.3 | 10.3 | 375.7 (n=4) | 48.72 |
| 8192 | 6296 | 1302 | 63.7 | 15.8 | 392.3 (n=16) | 52.06 |
| 16384 | 13175 | 1244 | 56.8 | 17.7 | 410.8 (n=32) | 57.77 |
| 32768 | 28719 | 1141 | 47.9 | 21.0 | 447.9 (n=64) | 70.31 |

现象复现:prefill −17%(1379→1141),decode −46%(89→47.9),ITL ×1.86。

---

## 2. prefill 归因:**100% 是 FI 注意力**

`engine.phase.prefill` 的逐 chunk 均值 2048 档 377.95 → 32768 档 447.13(+69.2 ms),
而 CUPTI 逐核:`flashinfer_prefill_paged_fp8kv`(FA2 `BatchPrefillWithPagedKVCacheKernel`)

| ctx | FI 设备时间/chunk | n | 每次 |
|---:|---:|---:|---:|
| 2048(均摊 ctx≈1k) | **5.24 ms** | 64(=4 chunk×16 层) | 0.328 ms |
| 32768(均摊 ctx≈16k) | **74.48 ms** | 1024(=64×16) | 4.655 ms |

⇒ 增量 **+69.2 ms/chunk**,与 prefill 相位增量 **逐位相等**(69.2 ms)⇒
**长 ctx 的 prefill 变慢 = FI 注意力一项**,其余(GEMM/GDN/norm/搬运)与 ctx 无关、基本不动。
末块(全 ctx)最贵:`engine.phase.prefill` max = **518.1 ms** vs 首块 364 ms。

**FI 的效率**(单形状换算):T=512/ctx=16384 → 4.655 ms/层 = 103 GFLOP ⇒ **21 TFLOPS**;
ctx=32768 满 ctx 时约 **44 TFLOPS**。3090 Ti fp16(FP32 累加)峰值 ≈ 40 TFLOPS
⇒ **FI 在 50–100% 屋顶**。⇒ 这部分是 **O(T·ctx) 的固有计算**,调核空间小;
要再快只能改算法(减计算)。真正的余量在下方 decode 侧。

---

## 3. decode 归因:**100% 是 verify 图的 XQA 注意力**(+ 一笔 ctx 无关的 host 税)

`spec.round` 分相(每轮,epoch 视图):

| ctx | spec.round | verify.read | 其中设备(verify 图) | 其中 XQA `kernel_mha` | propose | snap | fold |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 51.35 ms | 32.78 | 11.89 | 3.19 | 10.64 | 1.08 | 1.62 |
| 32768 | 73.57 | 54.77 | 33.50 | **24.66** | 10.76 | 1.07 | 1.69 |

⇒ 每轮 +22.2 ms,`verify.read` +22.0 ms、`verify.read` 内的设备增量 = XQA 增量 = **+21.5 ms**
⇒ **decode 的长 ctx 增量 100% 落在 verify 图的 XQA `kernel_mha`**(16 层全注意力,每轮 16 发)。

**已排除的嫌疑**(同批数据):
- **草稿图**(DFlash2 propose,gid=29):8.69 → 8.75 ms/轮 —— **与 ctx 无关**(草稿注意力
  `owl_naive_attn_nc_bf16_v2` 1.14 → 1.155 ms/发);草稿不是病灶。
- fold 桶(6 个 m 桶)、GDN 递推、marlin GEMM、snapshot(159 MB copy_batch/轮,~1 ms)
  —— 全部与 ctx 无关。

**`verify.read` 里有一笔 ctx 无关的 ~21 ms/轮 host 税**(54.77 − 33.5 = 21.3;2048 档
同样 32.78 − 11.9 = 20.9 ⇒ **两档同名**,是常数项;512 档 32.17 减同量级设备时间
(**推** ⇒ 设备约 11 ms)⇒ 约 21 ms,占短 ctx 一轮的 47%)。构成:3.97 MB logits D2H
(8 行 × 248320 词表 × f16)+ **host 侧"抽-比式"接受**(`exec/spec.rs`:8 × 248320 元素
f32 复制 + `sampler::sample` 逐行)+ 图回放/槽读。长 ctx 时被注意力淹没,但仍是常数税。

---

## 4. 逐核 roofline 对照(32768 档)

| 核 | 发射 | 设备时间 | 每次 | 换算 | 屋顶判定 |
|---|---:|---:|---:|---|---|
| FA2 `BatchPrefill…`(FI) | 1024 | 4766.9 ms | 4.655 ms | O(T·ctx) 计算 | **~21–44 TFLOPS ⇒ 50–100% 峰值(计算绑死)** |
| `kernel_mha`(XQA verify) | 976 | 1503.6 ms | 1.541 ms | 3.22 GFLOP/发 ⇒ **2.09 TFLOPS = 5.2% 峰值** | **~10% 带宽(87–174 GB/s vs 900)** ⇒ **延迟/并行绑死** |
| `owl_naive_attn_nc_bf16_v2`(草稿) | 305 | 352.1 ms | 1.155 ms | ctx 无关 | 无问题 |
| marlin 两族 | 8141 | 459.5 ms | 0.056 ms | ctx 无关 | 无问题 |
| GDN 族(chunked+conv+rec+norm) | ~15000 | ~730 ms | — | ctx 无关 | 无问题 |

**XQA verify 的发射几何**(源码级):`M_TILESIZE=32` ⇒ `rowsPerBlock=32`,
`nb_token_blocks = ceil(8×6/32) = 2` ⇒ `grid=(nbSubSeq=20, hkv(4)×2=8, 1) = 160 CTA`,
KV-split 开(verify 臂非 `single_block`)。KV 因 token-block 分裂被**重读 2×**;
即便按 2× 计,有效带宽也只有 ~90–170 GB/s ⇒ **不是屋顶,是核在该形状下的延迟/并行问题**。

---

## 5. 本轮顺带发现的**仪器缺陷**(已复核,可入册)

1. **counters 无时间戳 ⇒ epoch 视图对 counters 结构性失明**:
   `MetricsStore.counters` 是 `HashMap<String,(u64,..)>`, `collect_since_epoch` 只能过滤
   计时样本 ⇒ `spec.accepted` 在 ctx=512 那一点就显示 618(该点实际只有 65 轮、理论 ≤455)。
   **后果**:凡"AL / 计数类"判据都不能用 epoch 视图(`spec.rounds`/`spec.tokens`/`spec.accepted`/`spec.m*`
   全部如此)。修法可选:计数样本化(带时间戳)或提供 `counter_delta` 快照差分 API。
2. **家族(FFI)路的发射账没有几何**:`fam_ffi_launch` 条目 `grid/block/smem` 全 0
   (marlin/FI/XQA/cublas 都走这条路)⇒ 想核几何就得回源码或加探针。
3. `srv.wait.dtoh` 是**图态唯一可用的 GPU 时间代理**(p50 48.4 ms/轮 @32k),
   但它把"等排空 + 拷贝"混在一起,只能当趋势;正账必须 CUPTI。
4. CUPTI **必须在 boot 武装**(`--perf-level gpu`);在线 `POST /debug/perf/level/gpu`
   只切 metrics 级,不会启 shim ⇒ 切档后 `gpu.*` 全 0(本轮踩过一次)。

---

## 6. 下一步候选(按性价比;均带判据)

| # | 动作 | 判据 | 量级 |
|---|---|---|---|
| N1 | **verify 走"单 token-block"几何**(`rowsPerBlock≥48` 的实例化,或强制 `nb_token_blocks=1`) | XQA verify 每次 ≤1.0 ms(现 1.54)⇒ 轮时 −10 ms | 0.5–1d |
| N2 | **拆那笔 21 ms host 税**:把"抽-比"接受下沉到设备(已有 argmax/采样核先例)或 mmap 直读 logits 槽免 f32 复制 | `verify.read` −10 ms/轮(短 ctx 直接 +25% decode) | 1–2d |
| N3 | **verify 交回 FI 的可行性**(需给 verify 路造 FI 表:q_cu/indices/indptr/last_len;FI 核 T=8 时只有 4 CTA ⇒ 必须同时开 split-KV) | 单发 ≤0.5 ms 才值得 | 2–3d,先做单算子探针 |
| N4 | **XQA 核内为什么慢**(页表走法/流水/归约):nsys ⚠️ 本仓 FI 档不可靠 ⇒ 用 CUPTI + 自写微基准或读 `mha.cu` 内循环 | 定位到具体瓶颈 | 1–2d |
| N5 | prefill 侧:FI 已近屋顶 ⇒ 只有**算法面**(减少 attend 的 KV / 更激进的量化)才有意义 | 先算账再立项 | — |

---

## 7. 复现命令

```bash
# 实例(GPU2 = GPU-89dffb90-...;生产在 GPU1,不打扰)
cd <worktree>/perf
CUDA_VISIBLE_DEVICES=GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx \
  ./target/release/server --flavor spec-dflash testdata/args/perf-ctxladder-27b.toml \
  --perf-level host --log-level info

# 梯子(逐点跑 llm_speedtest + 抓 epoch perf/发射账)
python3 tools/ctxladder.py --out <out-dir> \
  --lengths 512,2048,8192,16384,32768 --rounds 1 --notes "<档位说明>"

# 逐核档:CUPTI 必须在 boot 武装 ⇒ 重启实例并加 --perf-level gpu
```

---

# 二轮(同日下半场):decode 真身 —— **图派发税**,不是采样

> 用户令"一口气全上,把 decode 速度优化上来"。本节记录:① 落了什么、② 量到了什么、
> ③ 两条**负结果**(免得下轮重走)。

## 8. 落地(本 worktree,未进 main)

| # | 改动 | 位置 | 实测收益 |
|---|---|---|---|
| W1 | **verify 臂 KV-split 度封顶 8**(自动档 `sm/hkv = 21` 偏大) | `family/xqa.rs::VERIFY_SPLIT_CAP` + `launch_geometry_capped` + `server/xqa.rs::XqaGeom::{Auto,Capped,Single}` | XQA `kernel_mha` 1.541 → **1.492 ms/发(CUPTI,−3.2%)** ⇒ 轮时 **−0.4…−1.0 ms(−1.4%)** |
| W2 | **verify 分相细化**:`spec.verify.tokread` / `spec.verify.lgread` | `exec/spec.rs` | 诊断面(见 §9) |
| W3 | **几何扫描探针** | `crates/kernels/tests/xqa_geom_probe.rs` | 见 §10 |

## 9. `verify.read` 的真构成(首轮归因**在此处修正**)

首轮把 `verify.read`(54.8 ms/轮)里那笔 ~21 ms 记成"logits D2H + host 采样"——
**错了**。二轮三重证据:

1. **greedy A/B 不成立**:`[sampling] mode=greedy` 后 `verify.read` 32.78 → 32.23(@2048),
   只差 0.75 ms ⇒ 采样链不是那笔钱;
   (根因:`sampler_on` 来自**请求**的 `temperature/top_p`(AGT-06),llm_speedtest 每次都发 ⇒
   配置档位改不动它;要真绕开得发不带采样字段的请求。)
2. **细粒度计时**:`tokread`(实测 = tok 读之后的下一段 = **logits 读**)= **5.87 ms/轮**;
   3.97 MB D2H + 2M 元素 f16→f32 解码,量级自洽(传输仅 ~0.2 ms)。
3. **`step_profile` 探针**给出图独占墙时:`[verify-prof] step=1.85ms read=53.8ms **pure=50.0ms**`
   ⇒ **verify 图墙时 50 ms**,而 CUPTI 逐核合计 **33.5 ms** ⇒ **图内派发/交接 ~16.5 ms/轮**。

⇒ 修正后的每轮账(32768):`round 69.8 = 图墙 50(核 33.5 + 派发 16.5) + propose 10.6
+ snap/fold/emit 2.8 + 接受/自肃 ~6.4`;2048 档同构:`round 51.4 = 图墙 ~30(核 11.9 + 派发 ~18)
+ propose 10.6 + 2.8 + ~6`。

**⇒ decode 的最大单项是「verify 图的派发/交接税」(常数 ~15–18 ms/轮)**,不是注意力、不是采样:
短 ctx 时它占一轮 **29%**,长 ctx 占 **22%**,且**与 ctx 无关**(节点数定型)。

## 10. 负结果(两条,别再试)

1. **`M_TILESIZE=64` 一档不可行**:`mha.cu:85` `static_assert(nbValidRows <= 32u)` ⇒
   verify 的 48 打包单元(q=8 × headGrp6)**结构上**必须分 2 个 token block
   ⇒ KV 重读 ×2 是内核约束,不是几何选择(已回退资产变体,结论留在 `xqa_geom_probe` 头注)。
2. **`graph_flags.device_launch` 被驱动拒**:`graph_end: 实例化(flags=6): CUDA_ERROR_INVALID_VALUE`
   —— 图内含 MEM_ALLOC/MEM_FREE 节点(设备侧启动图不允许)⇒ 该逃生阀**关死**。
   (`upload=true` 单开只影响首次回放,稳态无差别。)

## 11. verify 图节点构成(debug 级 `gl-prof` 直方图;1484 节点全是 KERNEL)

| 组 | 节点数 | 说明 |
|---|---:|---|
| marlin(awq/int4 两族) | 321+32 | 权重 GEMM(8 行) |
| **fold 记录拷贝**(`owl_narrow_strided_f16`) | **304** | **承重**:`plan_fold_family(vg)` 以 verify 输出块为烘焙叶 ⇒ 删不得(曾误判"死重",已撤) |
| `owl_gdn_conv_fwd` | 144 | 48 GDN 层 × **q/k/v 三次发射** |
| `owl_gdn_l2norm` | 96 | 48 × 2 |
| rmsnorm / add / silu / fused_add | 65/64/64/64 | — |
| gemv_dual / recurrence / norm_act / sigmoid / gating | 48×5 | — |
| norm_rope / reshape_and_cache / **kernel_mha** / sigmoid_gate | 32/16/16/16 | — |
| argmax / embed / concat_rows | 8/1/1 | — |

⇒ 节点削减只有**核融合**一条路(如 conv q/k/v 三合一:144→48,约 96 节点 ≈ 1 ms/轮);
派发税本体(16.5 ms)是结构性,除非融合或换更粗的核。

## 12. ⚠️ 方法论:**decode t/s 的 ±20% 是 AL 抖动,不是引擎抖动**

同一 ctx=32768、同一档、同一客户端,四次运行的 `spec.round` 轮数 = **76 / 61 / 72 / 58**
(AL = 接受草稿均值 = 256/轮数 − 1 = 2.37 / 3.20 / 2.56 / **3.41**)⇒
decode t/s 直接 **47.9 / 57.0 / 48.9 / 63.2**(±20%)。

⇒ 判读纪律(入册):**decode 比较必须以「轮时(ms/轮)」为主键,`t/s` 只在轮数同档时才可比**;
llm_speedtest 的 t/s 是 AL 的函数,而 AL 随 prompt 域/温度/随机性漂。
本 worktree 的 `tools/ctxladder.py` 已把 `rounds`/`AL` 一并落进 `summary.tsv`。

---

# 三、投机开/关对照(用户追问补测;2026-10-15 晚)

同池同几何(`perf-ctxladder-27b.toml` vs `perf-ctxladder-bare.toml`,唯一差异 `[spec] depth 7 → 0`)。

| ctx | **spec-on** ms/tok | t/s | **裸 decode** ms/tok | t/s | 投机净加速 |
|---:|---:|---:|---:|---:|---:|
| 2048 | 15.17 | 65.9 | 25.02 | 40.0 | 1.65× |
| 8192 | 14.93 | 67.0 | 26.21 | 38.2 | 1.76× |
| 32768 | 15.82 | 63.2 | 36.41 | 27.5 | **2.30×** |

- spec-on 的 `ms/tok` = 轮时 ÷ (AL+1);裸档 = `engine.phase.decode` 步时(255 步,±0.9 ms 抖动极小)。
- **两条结论**:
  1. **投机在所有 ctx 都是净赢,且越长越赢**(1.65× → 2.30×)——反直觉但那是因为
     它把**每轮固定成本**(派发税 ~16.5 ms + 权重带宽)摊到 AL+1 ≈ 3.2–4.4 个 token 上;
  2. **投机把 per-token 的 ctx 敏感性砍半**:裸 **+0.37 ms / 1k ctx / token**
     (25.0→36.4 over 30k),spec **+0.16**(轮 +0.69 ms/1k ÷ 4.4 tok)——
     因为 8 行 verify **共享一次 KV 读**。
- **裸 decode @32768 的逐核**(CUPTI,255 步):
  `kernel_mha`(单行 XQA)= **14.52 ms/步(=16 层 × 0.91)**、decode 图(gid=2)设备
  = **16.41 ms/步**、其余 host/派发 ≈ 6.7 ms ⇒ 步时 37.6 ms。
  ⇒ 裸档的两大单项 = **单行注意力 14.5** + **图设备 16.4**;
  而 spec 档同量级的一轮(verify 图设备 33.5 + 派发 16.5)产出 4.4 个 token。

⇒ **优化优先级不变,但含义更清楚**:派发税(~11 µs/节点)与递推核的节点数是
**每 token 都要付**的钱 —— spec 只是把它摊薄了;**裸 decode 路径的 820 节点图
每 token 付 ~9 ms**,这也解释了"27B INT4 单卡只有 40 t/s"(权重带宽地板 ≈15 ms/token)。

---

# 四、V1 落地:FI 的 split-KV 修通(2026-10-15 晚;用户批准执行 V1)

**目标**:把 verify/decode 的注意力从 XQA(2.09 TFLOPS)换成 FI(prefill 核,split-KV)。
**结论**:**核级已全绿且大赢**,但**进图**还差一步(见 §4.4)。

## 4.1 三个根因(全是"从没真正跑过"型)

| # | 根因 | 证据 | 修法 |
|---|---|---|---|
| ① | **plan 的 `kv_indptr` 要 page 级**,我们一直传 token 级 | split 档 plan 算 `num_chunks_kv = ceil_div(kv_len_tokens, chunk_pages)` = 34,核按设备里的 `chunk×page = 2912 token` 算 `ceil_div(kv_len, chunk)` = 2 ⇒ partial 行的 `o_stride_n` 缩放与 `merge_indptr` 口径打架 ⇒ **除第 0 行全零**(T=8/ctx3072:零值 43008/49152) | `server/flashinfer.rs`:`kv_indptr = [0, ceil(ctx/page)]`。split 关时该数组不参与决策 ⇒ 这个单位错一直隐身 |
| ② | **尾部 CTA 掩码没接** | plan 只在 split 档生成 `block_valid_mask`(`i < new_batch_size`);`padded_batch = max(2·SM/hkv, tiles)` 会大于真工作项数 ⇒ 尾部 CTA 读陈旧调度表 | 适配器把该区指针接进 `params.block_valid_mask`(核侧守卫本来就在) |
| ③ | `partition_kv` 那行硬编码 `false` | 实际由 dispatcher 按 `tmp_v` 是否为空**自行覆盖** ⇒ 不影响行为;留着与 plan 同值只为自明 | 改成 `= plan_info.split_kv` |

**上游 main 同款**:①以外的两处是上游形态;① 是**唯一真凶**,且上游 plan/核的口径差只有在 `page_size > 1` 且 split 开时才暴露。

## 4.2 核级实测(3090 Ti;host oracle 门 1e-2;**全档 ≤5.5e-4,零坏行**)

| 形状 | FI+split | FI 无 split | XQA(现役) | vs XQA |
|---|---:|---:|---:|---:|
| T=1/ctx=3072 | **0.030 ms** | 0.126 | — | — |
| T=8/ctx=3072 | **0.046** | 0.342 | 0.196 | 4.3× |
| T=8/ctx=8192 | **0.078** | 0.906 | 0.530 | 6.8× |
| **T=8/ctx=32768** | **0.249** | 3.604 | **1.541** | **6.2×** |
| T=16/32/ctx=32768 | 0.460 / 0.686 | 3.604(三者同值 ⇒ 纯延迟) | — | — |
| **T=1/ctx=32768**(decode) | **0.118** | 1.264 | **0.910** | **7.7×** |
| T=2 / T=4 / ctx=32768 | 0.119 / 0.246 | 1.260 / 3.607 | — | — |
| T=512/ctx=16384(生产 prefill) | 5.393 | 5.394 | — | 中性(大 T 本就并行) |

⇒ 判据「≤0.3 ms/发」**达标**(0.249);小 T 收益 6–14×,大 T 中性。

## 4.3 已落地的收益(生产现状 = `fi_min_tokens=1`)

- **尾块面**:长 prompt 的**小尾块**(T 很小、ctx 很长)以前走 FI 无 split = 3.6 ms/层 × 16 = **~57 ms/请求**;现在 ~4 ms ⇒ **单请求省 ~53 ms**。
- **缺省转正**:`FI_SPLIT_KV` 缺省改为 **开**(`=0` 关回历史)。
- **引擎回归**(spec-dflash,ctx 阶梯,split 开):prefill **1347.9 / 1294.6 / 1141.9 t/s**
  (对照 split 前 1349.4 / 1299.5 / 1141.0)⇒ **±0.4% 内无回归** ✓;
  decode 逐点差异全部落在 AL 抖动内(§12 口径)。

## 4.4 ⚠️ 还没拿到的:把 verify/decode 的注意力换成 FI(需进图/换臂)

- **verify(每轮 16 层)**:现役 XQA 24.66 ms/轮 → FI+split 折算 **≈4.0 ms/轮**(0.249×16)
  ⇒ 轮时 70 → **~49 ms** ⇒ **decode +42%**。
- **裸 decode(每步 16 层)**:XQA 14.52 ms/步 → **≈1.9 ms/步** ⇒ 步时 37.6 → 25 ms ⇒ **+50%**。
- **卡点**:verify/decode 都是**捕获图**(XQA 的几何与 ctx 无关 ⇒ 一张图吃所有 ctx);
  而 **FI 的 plan 在 split 档依赖 ctx**(`num_chunks_kv`、`o_indptr`/`merge_indptr`、`padded_batch`)
  ⇒ **一张图只对一个 ctx 有效**。两条出路:
  1. **piecewise(推荐)**:把注意力移出图,成为图段之间的 **eager 断点**(每层一次 FI 调用 +
     一次 `graph_launch`);FI 的 plan 缓存按 `(T, ctx)` 自动命中(每轮 1 次 plan,~0.1 ms,
     换回 20 ms)。⇒ 正是 `prefill-graph.md` 的 **M2c 断点原语**复用。
  2. **按 ctx 桶预捕 verify 图**(每桶一张):图数 × 桶数,boot 时间与显存都涨;
     且桶内 kv_len 仍会漂 ⇒ 要核侧把"行 stride"改成传入常量(**改上游核**,不推荐)。
- **对 prefill-graph 设计的连带影响(必须回写主会话的设计)**:M2b「整段单图」的 P-1
  「plan 与 ctx 无关」**只在 split 关时成立**(F-5 的前提就是 `disable_split_kv=true`);
  split 开 ⇒ plan 变 ctx 依赖 ⇒ 那条设计必须显式标注该前提。

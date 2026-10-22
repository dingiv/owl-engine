# prefill 注意力换核:候选核单测与选型(2026-10-14)

> 用户令:"xqa 不行,你就更换其他引擎的对应核心……你换进来之前,你先写单测,测一下
> 它单独的核心的性能。"
>
> 本文 = 该令的执行记录:**三个候选核各有一把同形状的独立尺子**,结论 =
> **换上游 FlashInfer(同族升级,1.4×,25 文件 0.85MB)**,vLLM FA2 作已量备选。

---

## 0. 基线必须先有单测(本轮的起因与纪律)

上一轮(§14.1 教训)拿**邻居核**(`chunked knhd` 回落核)当基线做选型 → 方向反了。
本轮先把**生产核自己的单测**立起来,再谈换核:

| 核 | 单测/台架 | 状态 |
|---|---|---|
| **现役 FI(我们 vendor 的 fork)** | `crates/kernels/tests/fi_prefill_probe.rs`(另一 lane 立) | ✅ 有 |
| **XQA**(上轮候选) | `crates/kernels/tests/xqa_prefill_probe.rs` | ✅ 有 |
| **上游 FlashInfer 0.6.18**(本轮候选) | `crates/kernels/cu/fi2/owl_fi2_bench.cu` + `run.sh` | ✅ 有 |
| **vLLM FA2(sm80 hdim256)**(本轮候选) | `crates/kernels/cu/fa2v/`(+ 本文 §3 的台架) | ✅ 有 |

共同形状 = 生产 chunked prefill 一个 chunk:`T=512` 行(全局位置 `[ctx−T, ctx)`),
GQA 24q/4kv,`hd=256`,page 32,kNHD 池,`ctx ∈ {512,2048,4096,8192,16384}`;
共同判据 = host 参考(行 i 看 `[0, ctx−T+i]` 闭区间)+ 同机同 iter 计时。

---

## 1. 结论表(3090 Ti,全部单测实测)

**f16 池**(三个核都能跑):

| ctx | 现役 FI(fork)¹ | **上游 FI 0.6.18** | vLLM FA2 |
|---:|---:|---:|---:|
| 512 | 0.152 ms (42 TF) | **0.095 (68)** | **0.091 (71)** |
| 2048 | 0.798 (33) | **0.461 (56)** | **0.442 (58)** |
| 4096 | 1.657 (31) | **0.937 (55)** | — |
| 8192 | 3.405 (30) | **1.892 (55)** | **1.843 (56)** |
| 16384 | 6.816 (30) | **3.783 (55)** | **3.638 (57)** |

¹ 现役 FI 的 0.152/0.798/... 取自 `fi_prefill_probe`(它跑的是 **fp8kv 变体 = 生产变体**;
f16 变体未单独测 —— 因为生产池是 fp8)。

**fp8 池(生产口径,唯一 apples-to-apples 的对比)**:

| ctx | 现役 FI(fork,fp8) | **上游 FI(fp8)** | 比 |
|---:|---:|---:|---:|
| 512 | 0.152 | **0.113** (57 TF) | **1.35×** |
| 2048 | 0.798 | **0.568** (45) | **1.41×** |
| 4096 | 1.657 | **1.177** (44) | **1.41×** |
| 8192 | 3.405 | **2.397** (43) | **1.42×** |
| 16384 | 6.816 | **4.837** (43) | **1.41×** |

数值门全绿:`max|gpu−host| ≤ 3.4e-4`(与现役 FI 同 ulp 级),超门行 **0**。
(vLLM FA2 **无 fp8 路径** ⇒ 生产要它就得付"影子 f16 池 + 每 chunk 反量化"或改核。)

---

## 2. 为什么是"上游 FI"而不是"vLLM FA2"

| 判据 | 上游 FlashInfer 0.6.18 | vLLM FA2(sm80) |
|---|---|---|
| fp8 池(生产口径) | **1.41× 于现役** | 不支持(需影子池/改核) |
| 进树成本 | **25 头 / 0.85 MB**(无 cutlass) | 150 头 / **6.1 MB**(cutlass 闭包) |
| 接线成本 | **同族升级**:`PREFILL_FI_FP8KV` 算子名/层分派/server runtime **全不用动**,只改 adapter 内 3 处调用 | 新家族 + 新 launcher + 新层臂 + 白名单台账 |
| 生态位 | **vLLM 自己 pin 的就是它**(`requirements/cuda.txt: flashinfer-python==0.6.16.post3`;SM100 首选) | vLLM 在 **SM86/SM80 的默认**(`FLASH_ATTN` 优先,FLASHINFER 次之) |
| 顺带修坏件 | ✅ 我们这份 fork 的 **split-kv/plan 是坏的**(§15.3 单测复现);上游 plan 会按 SM 数自动选 `cta_tile_q` + **q 瓦片伪批**(本形状 `padded_batch_size=48` ⇒ 192 CTA) | — |

⇒ **等价性能下,选集成成本最低、且顺手把我们那份坏 plan 换掉的那条**。

---

## 3. 集成配方(work order;下一会话可直接照做)

**① 换源(已备)** `crates/kernels/cu/fi2/`
- `include/flashinfer/**` = 上游 nightly-0.6.18(`flashinfer`(外部存档) @ `fdc56343`,
  2026-09-17)的 **paged prefill 头闭包 25 文件 / 0.85 MB**(闭包复现法:`nvcc -M`
  扫一个含 `BatchPrefillWithPagedKVCacheDispatched` 的 TU;清单见 `cu/fi2/README.md`)。
  **自足已验证**:只用这 25 头即可编译并复现全部数字。
- `owl_fi2_bench.cu` + `run.sh` = 本文的单测台架(`FI_SK=1` 切 split-kv,
  `-DUSE_FP8=1` 切 fp8 池)。

**② adapter 改写**(`cu/flashinfer/owl_fi_attention.cu` → 新 API;逐条对齐
`csrc/batch_prefill.cu` / `csrc/batch_prefill_paged.cuh` 的参考实现):

| 面 | 旧(fork) | 新(0.6.18) |
|---|---|---|
| plan 入参 | `PrefillPlan(float_ws, fws_sz, int_ws, int_ws_sz, page_locked, pl_sz, plan15, …)` | 出 `PrefillPlanInfo`;多 `head_dim_qk/vo`、`enable_cuda_graph`、`sizeof_dtype_o`、`window_left`、`fixed_split_size`、`disable_split_kv`、`num_colocated_ctas`、`uniform_q_len`、**`kv_dtype_bytes`**(fp8=1/f16=2) |
| **workspace 定尺** | 硬写 32MB int / 64MB float | **必须先问 `PrefillPlanWorkspaceSize`** 再按尺寸分配(pinned host int 缓冲同尺寸;`cudaHostAlloc`) |
| 发射 | `run` 单函数(参数一大串) | `BatchPrefillPagedKVCacheDispatched<SAME_KV_STRIDES,CTA_TILE_Q,HD_QK,HD_VO,POS,FP16QKRED,MASK,AttnVariant,Params>`;`CTA_TILE_Q` 用 `plan_info.cta_tile_q` 分派(64/128/32) |
| 池描述 | 手写 stride(元素序) | `paged_kv_t<DTypeKV,IdType>(num_heads,page,hd,batch,kNHD,k,v,indices,indptr,last_page_len,nullptr)`——**NHD 偏移公式与我们 kNHD 逐字一致** |
| 参数 | `sm_scale` 走 `f32::to_bits` 过线 | **`params.sm_scale` 直接是 f32**(线上可省掉位型走私) |
| fp8 | 需 `-DFLASHINFER_ENABLE_FP8_E4M3` | **不需要额外宏**(核内有 `fast_dequant_f8f16x4`) |

**③ 验收(缺一不可)**
1. `fi_prefill_probe` 改指上游 adapter:数值门 + 全 ctx 测速(应落在本文表内);
2. 层/引擎级 A/B(`engine.phase.prefill`,3 轮交替 boot):预期 **−1~2%**
   (注意力在 3k ctx 约占 prefill 2.6%、8k ctx 约 5.7%);
3. **greedy 恒等门**(换核 = 换数值路径,4/4 逐字是硬门);
4. 白名单/台账不动(`owl` 侧算子名不变);`FI_INT_WS_BYTES`/`FI_FLOAT_WS_BYTES`
   从硬写改为 plan 派生(新 API 要求尺寸精确)。

**风险与回退**:adapter 是同族改写,`prefill_split=false` 档位下 FI 是唯一 prefill 臂 ⇒
回退 = 保留旧 adapter 文件 + 构建期开关,或直接 `git checkout` 该文件(改动面 = 1 个 .cu)。

---

## 4. 方法论留档(省下一轮返工)

1. **"基线核必须先有单测"** —— 本轮三份单测各自独立给出数字,没有一条靠推演;
   现役 FI 的 30-42 TFLOPS 与上游的 43-68 TFLOPS 差**不是**测法差异(同 harness 同形状)。
2. **vLLM 的后端优先级按硬件分级**(`vllm/platforms/cuda.py`):SM100 → FlashInfer 首选;
   **SM86/SM80 → 仍首选它自家 FA2**,FlashInfer 次选。⇒ "vLLM 换成 FlashInfer 了"对
   **我们这张卡**不成立;而 vLLM **确实**在 pin FlashInfer(`0.6.16.post3`)。
3. **nsys 在本机对 FI 档不可靠**(稳定丢 2/3 核);**CUPTI in-engine**
   (`--features perf,cupti` + `perf_level=gpu`)才是本机的可靠逐核尺子。
4. vLLM FA2 装配踩坑三条(**不报错只错值**,已留档 `cu/fa2v/owl_fa2_attention.cu` 头注):
   `causal = (window_size_left<0 && window_size_right==0)`(写 −1/−1 = 非因果);
   `p_dropout` 是**保留概率**(0.0 → rp_dropout=1/0=inf);paged+varlen **禁 num_splits>1**。
5. **fp8 是判据的一部分**:同一核 f16→fp8 掉 ~20%(上游 0.461→0.568 ms @ctx2048);
   拿 f16 数字去比别家的 fp8 生产档会虚高一档。

---

## 5. 台账

| 文件 | 内容 |
|---|---|
| `crates/kernels/cu/fi2/` | 上游 FI 0.6.18 paged prefill 头闭包(25)+ 单测台架 + run.sh(**候选,未接线**) |
| `crates/kernels/cu/fa2v/` | vLLM FA2(sm80 hdim256)源 + 闭包(150)+ launcher(**候选,未接线**;1.8× f16,无 fp8) |
| `crates/kernels/tests/{xqa,fi}_prefill_probe.rs` | 两个已接线核的尺子(XQA / 现役 FI) |
| 生产面 | **一行未动**(config/flavor/层分派/server runtime 全不变) |

---

## 6. **已换源落地**(2026-10-14;用户"可以,开始替换")

**动作**:把 owl 的 FI 从 `guoqingbao/flashinfer` fork @2026-08-07 **换成上游
nightly-0.6.18**(`flashinfer`(外部存档) @`fdc56343`),**ABI 逐字未变**
(`owl_fi_prefill_plan` / `_run` / `_run_fp8kv` 三个 C 入口、15×int64 的 plan blob、
槽序、`FI_SPLIT_KV`/`FI_DEBUG` 旋钮全保留)⇒ **Rust 侧(family/server/探针)零改动**。

| 文件 | 变更 |
|---|---|
| `cu/flashinfer/include/` | 旧 fork 22 头 → **上游闭包 25 头 / 0.85 MB**(无 cutlass、无额外宏) |
| `cu/flashinfer/owl_fi_attention.cu` | 按上游 API 重写(plan 前置 `PrefillPlanWorkspaceSize` 尺寸校验 + `kv_dtype_bytes`;params 改默认构造逐字段赋值;发射模板加 `SAME_KV_STRIDES`;fp8 臂去掉编译宏) |
| `cu/flashinfer/bench/` | 候选核单测台架(不参与构建;`FI_SRC=$(pwd)/.. ./run.sh` 复现 §1 数字) |
| `crates/kernels/build.rs` | 去掉 `-DFLASHINFER_ENABLE_FP8_E4M3`(上游不需要);增量 stamp 旗标改 `src=upstream-0.6.18` |
| 生产面 | **一行未动**(算子名/层分派/server 计划缓存/白名单全不变) |

### 6.1 验收证据(四层)

| 层 | 判据 | 结果 |
|---|---|---|
| **单算子数值** | `fi_prefill_probe`(经生产路径 plan+run)vs host 参考 | **≤3.2e-4,超门行 0**(T=1/8/54/128 × ctx3072 + T=512 × ctx512/2048/4096/8192/16384 全绿) |
| **单算子性能** | 同探针 T=512 | **0.112/0.568/1.176/2.432/4.829 ms**(ctx512/2048/4096/8192/16384)= 旧 fork 的 **1.36–1.41×**;小 T 档 0.31ms vs 旧 0.44(T=8–128)= **1.4×** |
| **引擎级定量** | `[probes] prefill_cksum`:同一 prompt 的 chunk 隐层 `sum|x|`,**新 FI vs 独立核**(`chunked knhd`,`fi_min_tokens=99999`) | 31,988,080 vs 31,987,172 ⇒ **相对差 2.8e-5**(在**非恒等物理页链**上,即引擎集成正确) |
| **引擎级逐字** | greedy 96 token × 4 题 / 首 token × 8 题(同二进制两配置) | 3/4 全篇逐字一致;6/8 首 token 一致;2 处差异 = **72 字共同前缀后**的近并列翻面(同义改写,两侧均连贯)——与 2.8e-5 级扰动一致,非缺陷 |
| **CUPTI 逐核** | 引擎内 FI 注意力 device 时间(3126 tok / 7 chunk) | **49.6 ms vs 旧 63.0 ms = −21%**(每 chunk 0.451 vs 0.656 ms) |

相位口径:`engine.phase.prefill` 在该 prompt 上 2358 ms(新)/ 2290 ms(旧,3038 tok 口径)
—— **注意力只占相位 ~2.5%,−21% 的核级收益在相位上 ≈ −0.6%,落在单发噪声内**;
要出相位级数字须做多轮交替 boot 的 A/B(本轮未做,核级 CUPTI + 定量门已足够定案)。

### 6.2 回归门(全绿)

`models 117/117`(含真权重 prefill + 27B smoke)· `engine 22/22` ·
kernels lib `61/61` · `op_tables 7/7` · `no_runtime_compile 2/2` ·
`fi_prefill_probe ✓`。**未跑**:e2e 全套(`gpu_longctx_fi_direct_gate` 等,需 e2e harness)。

### 6.3 回退与挂账

- **回退**:旧 fork 的 24 个跟踪文件可直接 `git checkout crates/kernels/cu/flashinfer/`
  (另留 `/tmp/fi-fork-old/` 备份);回退面 = 1 个 .cu + include 目录。
- **挂账**:① `FI_FLOAT_WS_BYTES=64MB / FI_INT_WS_BYTES=32MB` 仍是硬写 —— 上游要求
  尺寸精确,现已改为**plan 前置校验 + 报错**(`-202` 打印所需尺寸),后续可从
  `PrefillPlanWorkspaceSize` 派生;② `host_staging` 是普通 `Vec<u8>`(非 pinned)——
  `cudaMemcpyAsync` 从可分页内存合法(仅一次/形状,成本可忽略),要极致可改
  `cudaHostAlloc`;③ `cta_tile_q` 的 `kv_dtype_bytes` 目前由 `OWL_FI_KV_BYTES` 缺省 1
  (生产 fp8)给出,与两条臂共用 plan 缓存(仅影响选瓦启发式,不影响正确性)。

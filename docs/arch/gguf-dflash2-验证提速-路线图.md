# GGUF 线 DFlash2 投机 · 下一步优化路线图(2026-10-21 交接档)

> 交接对象:下一个会话。**读完第 3/4/5 节即可开工**;第 1/2 节是口径与已完成项(不要重测)。
> 主线:把 `verify` 的**长 ctx 成本**压下来,让 spec 在 4096/8192 也转正。

---

## 0. 一句话现状

投机已经在中短 ctx 转正(512 **+78%** / 1024 +63% / 2048 +14%),但 4096/8192 仍 −14%/−15%。
根因两条:① `verify` 的注意力是 **v2 paged、8 个伪序列各读一遍 KV**(llama.cpp 用 `fattn-tile`
一次读);② **AL 随生成位置衰减**(≈4.0 → ≈1.7),跌破盈亏线。

---

## 1. 基线(全部 2026-10-21 实测,单 3090 Ti,勿重测)

**硬件/档位**:3090 Ti @ `41:00.0`(UUID `GPU-e565c505-4921-9979-2e0f-2b83490b7aed`),
27B `Qwen3.8-27B-UD-IQ4_XS.gguf`(13.3 GiB,64 层 = 16 full + 48 GDN,full_heads=(24,4,256),词表 248320)
+ GGUF 草稿 `Qwen3.8-27B-DFlash2-Q4_K_M.gguf`(1.08 GiB,5 层,bf16 执行域)。
档位 = `profiles/gguf27b-iq-dflash.toml`。**跑 std 矩阵必须加 `--set pool.prefix_cache=false`**
(否则第二轮起撞既有不变量 `块已在宿主层` → 空响应;见 §7)。

### 1.1 spec=off(标准口径,`llm_speedtest --flavor std`,greedy)

| 点 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---:|---:|---:|---:|---:|
| prefill t/s | 1687–1694 | 1822–1841 | 1817–1830 | 1804–1818 | 1771–1788 |
| decode t/s | 40.4–40.6 | 40.4–40.9 | 40.5–41.0 | 39.9–40.2 | 38.8–39.2 |
| TTFT ms | 303 | 560 | 1124 | 2264 | 4620 |

均值:**prefill 1783 / decode 40.05**。原始数据 `docs/bench/data/gguf27b-iq4xs-std-nospec-2026-10-21.json`。

### 1.2 spec=d7(同一工具/参数)

| 点 | 三轮 decode | 相对基线 |
|---|---:|---:|
| 512 | 72.4 / 73.8 / 66.2 | **+78%** |
| 1024 | 69.2 / 66.3 / 63.7 | **+63%** |
| 2048 | 48.7 / 45.9 / 44.5 | +14% |
| 4096 | 34.6 / 34.2 / 34.7 | −14% |
| 8192 | 32.3 / 33.0 / 33.3 | −15% |

> spec 侧 JSON 未落盘(导出被中止);表值见 `docs/arch/gguf-dflash2-投机-调研与设计.md` §9.4。

### 1.3 轮内账(引擎探针,非 profiling 值)

| 项 | 值 | 出处 |
|---|---:|---|
| `round`(EWMA) | **62 ms**(短 ctx)/ 70 ms(8k) | `[spec-degrade] … 轮 62ms/裸 28ms` |
| `verify` | **52.6 ms**(图 GPU,d7=8 行) | `[verify-prof] pure` |
| `propose` | **6.1 ms**(图 GPU) | `[dflash-prof] pure_gpu` |
| 盈亏线 = round/bare | **2.2**(短)/ 2.5(8k) | 引擎自动算 |
| 实测窗口 AL | 2.0–4.0(长位置掉到 **1.67–2.0**) | `[spec-degrade] 窗口 AL=…` |

---

## 2. 已完成(今天,`feat/gguf`,HEAD = `8df83cc`;不重做)

| 提交 | 内容 | 效果 |
|---|---|---|
| `fde1cba` | 复用臂(tile)补 10 型 IQ/K-quant + **MMQ2 小 m 下限 64→4** | verify 175→127 ms |
| `8df83cc` | **MMQ2 J=16 小 m 档** + **MMQ2 的 BF16 激活变体**(草稿线) | verify 71→**52.6** ms;propose 11.5→**6.1** ms;round 83→**62** ms |

归纳出的两条硬事实(下个会话直接引用):

1. **MMVQ 的权重流量 = m × 权重字节数**(网格 `(n/n_warps, m)`)—— m=1 是对的,m=8 就是 8×;
2. **小 m 的最优臂是 MMQ2**(J=16):29 MB 张量 m=8 ≈ 35–40 µs(≈750–800 GB/s),
   而 tile 153 µs、MMVQ 238 µs(同 m);m=1 仍 MMVQ(带宽地板)。

门(全绿,改内核前先跑):
`cargo test -p owl-models --lib formats::gguf::kernel_gate::m_scan`(含
`gpu_gguf_tile_matches_mmvq_iq_family` / `gpu_gguf_mmq2_small_m` / `gpu_gguf_gemm_m_scan`)。

---

## 3. 调研:llama.cpp(`packages/llama.cpp`,分支 `dflash-optimize`)用的是谁

### 3.1 量化 GEMM(按 token 数三段分派,`ggml/src/ggml-cuda/ggml-cuda.cu:1975-2005`)

| m | 核 | 判据 |
|---|---|---|
| **≤ 8** | **`mul_mat_vec_q`(MMVQ,`ncols_dst` 模板化)** | `MMVQ_MAX_BATCH_SIZE = 8`(`mmvq.cuh:3`);Ada 个别型更紧,Ampere 一律 8 |
| > 8 | **`mul_mat_q`(MMQ,int8 tensor-core)** | `should_use_mmq`(sm86 `turing_mma_available` → true) |
| 非量化 | `mul_mat_vec_f` / cuBLAS | — |

**关键:他们的 m≤8 MMVQ 是模板化的多 token 版**(`mmvq.cu:585-760`):
权重块在 k 循环里**只读一次**,内层 `#pragma unroll` 展开 `ncols_dst` 个 token 复用
(`tmp[j][i] += vec_dot(vx, &y[j*…], kbx, kqs)`);Ampere 参数表(`mmvq.cu:436/563`):
m∈[1,4] → `nwarps=4`,`m∈[5,8]` → `nwarps=2`,`rows_per_block=2`。

### 3.2 注意力(`fattn.cu:640-760`)

| 形态 | 核 |
|---|---|
| decode(q=1,无 GQA-opt) | `fattn-vec` |
| **verify/propose(q=2..8)** | **`fattn-tile`**,`ncols` 模板 ∈ {2,4,8,16,32};hd=256/128 均查表(`ncols=8` → 256 线程 / occ 2 / `nbatch_Ka=64`) |

语义 = **一个 CTA 一次吃 8 行 q,K/V 每个 tile 只读一遍**。
DFlash2 草稿 = 标准 `build_attn` + **非因果 mask**(`src/models/dflash.cpp:725`),KV 用
`llama_kv_cache_iswa`(滑窗)、**非 paged**;整个投机循环还被捕成**一张** CUDA graph(`G7-③`)。

### 3.3 对照表(结论)

| 面 | llama.cpp | owl(现状) |
|---|---|---|
| m=1 | MMVQ,权重 1× | MMVQ,权重 1× ✓ |
| m=2..8 | 模板化 MMVQ(权重 1×,算术 O(m)) | **MMQ2 J=16**(int8 MMA)✓ 同量级 |
| m>8 | MMQ(J 8..128 自适应) | MMQ2(16/64/128)+ MMQ(dp4a)+ tile |
| **verify 注意力** | **`fattn-tile` ncols=8,KV 1×** | **v2 paged,**8 伪序列 ⇒ KV 8×** |
| KV | 连续滑窗(非 paged) | paged(业务需要) |
| 投机循环 | 整环一张图 | 三张图 |

---

## 4. 差距分析与假设(开工前先确认这几条数学)

**KV 数学**(27B:16 full 层 × 2 × hkv 4 × hd 256 × 2 B = **64 KB/token**):

- @8192 ctx ⇒ 512 MB;v2 按 8 个伪序列读 ⇒ **4 GB**,≈ **4.5 ms** @900 GB/s;
  另有 8× 的 CTA/归约开销(实测 v2 单发 39.8 µs × 2432 发 / 全会话,见 §9.2)。
- ⇒ 假设 **H1**:verify attention 换"8 行共享一次 KV"可省 **5–10 ms/轮**。

**但要诚实**:verify 图 GPU 52.6 ms 里,权重读取 ≈ 15.8 GB(含 lm_head 874 MB),
按 MMQ2 实测 750–800 GB/s 只应 ≈ 20–22 ms ⇒ **约 30 ms 尚未归因**。
所以第一步不是写核,而是**把 verify 的 52.6 ms 分相拆开**(H1 的 5–10 ms 可能只是其中一部分)。

**AL 衰减**(H2):长位置窗口 AL 掉到 1.67–2.0 < 盈亏线 2.2–2.5 ⇒ 降级机制介入,
实测 4096/8192 的 −14%/−15% 主要是**"spec 在亏 + 降级探测税"**,不是 propose/verify 本身。
⇒ 即使 verify 降到 40 ms,盈亏线也只到 1.9–2.0,**8192 大概率仍不赚**;要如实写进验收线。

---

## 5. 施工分解(V0–V4;每步一个门,先小步验证再动大核)

| 步 | 内容 | 门(先跑最小用例) | 估 |
|---|---|---|---|
| **V0** 零代码 A/B | 开 `dispatch.verify_fi = true` + `fi_plan_bake`(FI prefill 形态本来就是"多行 q 一次读 paged KV"),原档对照 | ① 前 64 token 贪心与 v2 逐字节一致;② std 512/2048/8192 三点 decode;③ `[verify-prof] pure` | 0.5 h |
| **V1** verify 分相 | 用 nsys `--cuda-graph-trace=node` 按 **CorrId** 分图(锚:`owl_dflash_select_bf16`=propose,`paged_attention_v2`=verify),给出 verify 的 kernel 预算表(GEMM / attn / GDN / act-quant / norm 各多少 ms) | 预算表的各项之和 ≈ 52.6±10% | 1 h |
| **V2** 8 行共享 KV | 若 V0 的 FI 臂正确且更快 ⇒ 直接转正(改 dispatch 默认 + profile);否则照 `fattn-tile` 写 `ncols=8` 变体(hd=256:256 线程 / occ 2 / KV tile 64) | ① 32k ctx 与 v2 逐行 argmax 一致;② 8 行 KV 流量 = 1×(nsys 核内字节);③ verify ≤ **35 ms** @8k | 0.5–1 d |
| **V3** 长 ctx 策略 | AL 衰减的处置:按 ctx 分段自动 depth(7→3),或只用降级机制并调 `spec.degrade_after/probe_every` | std 五点曲线**逐点不劣化**,4096 ≥ +5%、8192 ≥ 0% | 0.5 d |
| **V4** 余项(可选) | ① `out_dt` 的成对移位 2 发/GEMM 折进 act-quant 的 scale 参数(省 ~0.3 ms/轮);② MMQ2 的 k 向 split(把 n=5120 时只有 40 CTA 的问题解决) | 各自 ≥5% 才收 | 挂账 |

**回退姿势**:V0/V2 都是**档位开关**(`dispatch.verify_fi`)或**新臂**,v2 原路保留;
`OWL_GGUF_MMQ2=0` / `OWL_GGUF_TILE=0` 可一键回退到今天之前的臂。

---

## 6. 验收线(下个会话结案时按这张表报)

1. **std 五点三轮(spec d7)**:512/1024 ≥ +60%;2048 ≥ +10%;**4096 ≥ +5%;8192 ≥ 0%**;
2. **verify 图 GPU**:52.6 → **≤ 35 ms** @8k;round ≤ **75 ms** @8k;
3. **一致性**:spec-on 与 spec-off 的贪心输出前 64 token 逐字节一致(既有口径);
4. **回归**:`cargo test -p owl-models --lib`(295 绿)、`-p owl-engine --lib`(76 绿)、
   `-p owl-kernels --lib`(111 绿);
5. 数据落盘 `docs/bench/data/gguf27b-iq4xs-std-spec-2026-10-2X.json`。

---

## 7. 纪律与坑(踩过的,别再踩)

1. **std 矩阵必须 `--set pool.prefix_cache=false`**:同一 prompt 跨轮会撞
   `store: 不变量违反:块 ChainHash(…) 已在宿主层` ⇒ turn 弃置、空响应(既有 bug,未修,与本线无关)。
2. **测量口径**:`llm_speedtest --temperature 0 --top-p 1 --presence-penalty 0 --frequency-penalty 0`
   (缺省 0.7 会走 host 采样器,decode 掉 30%)。
3. **引擎自己的探针会双回放**:`probes.step_profile=true` 时 `[spec-prof]` 的
   `verify`/`propose` **翻倍**;真实值看 `[verify-prof] pure` / `[dflash-prof] pure_gpu`。
4. **nsys** 必须 `--cuda-graph-trace=node`;CorrId 分组对 graph 不稳(多数组只有 1 核),
   只有 **propose 组**(median 384 核)可靠 —— V1 要按"锚核 + 时序窗口"重新切。
5. **`pkill -f` 会匹配自己的命令行**(把 shell 杀掉):用 `echo $! > /tmp/owl.pid` + `kill $(cat …)`。
6. GPU 钉卡用 **UUID**;3080 两张被 ComfyUI 占用,别动。
7. 内核改动流程:`cu/*.cu` 加 wrapper → `family/gguf.rs` 入口名 + 能力表 →
   `ops/gguf.rs` 声明 + `pub static … meta_of::<T>()` → `driver.rs` 的分派臂;
   漏一处 = 构建期 `ops::source(…) 无该算子的准入声明` 或 `算子总表漏项`。
8. 本工作树**不主动 push**;提交到 `feat/gguf`。

---

## 8. 文件地图(交接清单)

| 面 | 文件 |
|---|---|
| 档位 | `profiles/gguf27b-iq-dflash.toml`(spec d7;ctx 12288) |
| 草稿装载 | `crates/models/src/formats/gguf/{draft_meta.rs, draft.rs}` |
| 草稿层 | `crates/models/src/layers/dflash2.rs`(`DraftFc::G`、`from_dims_keyed`) |
| GEMM 臂 | `crates/models/src/layers/linear.rs`(`gguf_quant_gemm` 的分派顺序、`GGUF_MMQ2_MIN_M`、out_dt 移位) |
| 内核 | `crates/kernels/cu/gguf/{gguf_mmq2.cu, gguf_tile.cu, gguf_gemm.cu}`;`crates/kernels/src/family/gguf.rs`(`MMQ2_SMALL_M/J`、`tile_entry_for`、`mmq2_ok`) |
| 引擎 | `crates/engine/src/{spec_runtime.rs, exec/spec.rs, exec/draft.rs, graph_fleet.rs}` |
| 门 | `crates/models/src/formats/gguf/kernel_gate.rs`(`mod m_scan`) |
| 文档 | 本文件 + `docs/arch/gguf-dflash2-投机-调研与设计.md` §8/§9 |

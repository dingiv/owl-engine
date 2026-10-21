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

**硬件/档位**:3090 Ti @ `41:00.0`(UUID `GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`),
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
3. **臂只看 `[dispatch] verify_arm/decode_arm`**(2026-10-21 单键化):boot 自证行
   `verify=FI(单图;臂=fi)` 把「请求臂」与「节点事实」并排打;两者不一致时先看这一行。
4. **引擎自己的探针会双回放**:`probes.step_profile=true` 时 `[spec-prof]` 的
   `verify`/`propose` **翻倍**;真实值看 `[verify-prof] pure` / `[dflash-prof] pure_gpu`。
5. **nsys** 必须 `--cuda-graph-trace=node`;CorrId 分组对 graph 不稳(多数组只有 1 核),
   只有 **propose 组**(median 384 核)可靠 —— V1 要按"锚核 + 时序窗口"重新切。
6. **`pkill -f` 会匹配自己的命令行**(把 shell 杀掉):用 `echo $! > /tmp/owl.pid` + `kill $(cat …)`。
7. GPU 钉卡用 **UUID**;3080 两张被 ComfyUI 占用,别动。
8. 内核改动流程:`cu/*.cu` 加 wrapper → `family/gguf.rs` 入口名 + 能力表 →
   `ops/gguf.rs` 声明 + `pub static … meta_of::<T>()` → `driver.rs` 的分派臂;
   漏一处 = 构建期 `ops::source(…) 无该算子的准入声明` 或 `算子总表漏项`。
9. 本工作树**不主动 push**;提交到 `feat/gguf`。

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

---

## 9. 2026-10-21 第二轮实施记录(V0 结案 + 刀 B 落地)

### 9.1 V0(零代码 A/B)结案:**配方缺一步,且 FI 不是等价结果**

- **配方缺 `dispatch.verify_v2 = false`**。臂选择序(`layers/attention.rs:1234` vs `:1268`):
  `verify_v2` 缺省 `true` ⇒ **v2 臂先命中,FI 节点虽登记却永不执行**。
  实测截获:开 `verify_fi` 而忘关 `verify_v2` 时 boot 打印
  `verify 面事实: FI 节点登记=true | 臂事实(attn.arm.verify.*):v2=32`
  —— **看臂事实、别看节点登记**(这条路子以后所有 A/B 都照此核对)。
- ✅ **该坑已结构性消除**(同日 `3bff4ed`):三/两个布尔收成**单值枚举**
  `[dispatch] verify_arm = "v2"|"fi"|"xqa"|"prefill"` / `decode_arm = "v2"|"fi"|"xqa"`
  (**decode 侧没有 `prefill`**:它跑的是分页 v2 主核 + LSE 归并;
  旧日志标签"prefill族"是错名,已随单键化正名为 `v2`);
  图面槽位与层内核选择共用同一个 `uses_fi()`,旧布尔键 `deny_unknown_fields` 拒启。
  现在"开了 FI 却跑 v2"**无法表示**。
- 真开 FI 后(`verify=fi=32`),verify 图 GPU(`[verify-prof] pure`):

| ctx | v2(旧 MMQ2) | FI(旧 MMQ2) | 结论 |
|---|---:|---:|---|
| 884 | 54.5 ms | 52.4 ms | ≈ 打平 |
| 7734 | **67.3 ms** | **52.3 ms** | FI **−15 ms**,且**对 ctx 完全平坦** |

- 但 **FI 的输出与 greedy 不逐字节一致**:884 ctx 下 spec-on(FI)与 spec-off 在**第 38 个字符**
  分叉;v2 在同一 ctx 下与 spec-off 完全一致(旧门口径)。⇒
  **FI 不能直接转正**(它是"另一次数值",不是同一个 argmax 序列);
  且 v2 在 7734 ctx 下与 spec-off 也在第 10 个字符分叉 ⇒ 逐字节门本身只在小 ctx 成立
  (既有性质,与本次改动无关)。
  **待办**:FI 与 v2 的 KV 累加序对齐(或明确降级为"数值近似臂")。

### 9.2 刀 B(`db87eb2`):IQ2_S / IQ2_XS / Q3_K 接 MMQ2 J=16

见提交信息。轮内 nsys 账(tiny 请求,同一轮窗口,ts 对齐):

| 内核 | 改前 | 改后 |
|---|---:|---:|
| `tile_iq2s` ×7 | 4.83 ms | `mmq2_iq2s` **0.83** |
| `tile_q3k` ×8 | 3.46 ms | `mmq2_q3k` **0.79** |
| `tile_iq2xs` ×2 | 1.40 ms | `mmq2_iq2xs` ~0.15 |
| **GPU busy/轮** | 51.7 ms | **51.7→43.5**(round 67.3 → **58.5**) |

⚠️ 新增 `mmq2_sram_stride_for` / `mmq2_smem_for`:Q3_K 布局 X 面 stride = **84**(不是 76)、
Q2_K = 100。**Q2_K 不能挂**:激活布局是 `MMQ_Q8_1_DS_LAYOUT_D2S6`,本适配层只有 D4/DS4(硬接出 inf)。

### 9.3 verify 轮预算表(nsys,`--cuda-graph-trace=node`,tiny 请求,单轮 51.7 ms busy)

| 项 | ms/轮 | 占比 | 备注 |
|---|---:|---:|---|
| MMQ2 GEMM(IQ4_XS 16.6 / Q4_K 6.8 / Q5_K 6.2 / IQ3_S 5.6 / IQ3_XXS 2.1 / Q6_K 0.7 / 新三型 1.8) | ~40 | 77% | 有效带宽 **≈330 GB/s**(权重 14.24 GB) |
| 其中 `tile_q8_0`(ssm_alpha/beta 96 发 × 35 µs) | 3.3 | 6% | n=48 → 12 CTA,69 GB/s |
| `gdn_recurrence` + `gdn_conv` 等 | 4.4 | 9% | |
| 其余(norm/silu/cast/quant/argmax) | ~3 | 6% | |
| **host 间隙**(轮内 6.2 ms) | 6.2 | 12% | 见 9.4 |

**GEMM 有效带宽的根因已定谳 = CTA 数,不是算法**:
`grid=(n/128, ceil(m/J))`,m=8 时 J=16 ⇒ `ntx=1` ⇒ **CTA 数 = n/128**。

| n | CTA | 实测带宽 |
|---:|---:|---:|
| 5120(ffn_down / attn_qkv 的 k 向内维) | 40 | 435 GB/s |
| 6144 / 10240 | 48 / 80 | ~490 |
| 17408(ffn_up/gate) | 136 | 570 GB/s |
| 248320(lm_head Q5_K,1940 CTA) | 1940 | **712 GB/s(≈ DRAM 顶)** |

⇒ 理论顶 712,小 n 只有 61%。**唯一解 = k 向切分(stream-k)**;
实测把 grid 硬拉到 168 会走 `fixup=true` 分支 → 空 `tmp_fixup` 立刻 IMA(已复现,勿再试)。

### 9.4 新发现:每轮 ~4.2 ms 的 host 空洞(propose→verify)

nsys 运行时 API 账(单轮 58.5 ms):

- `cuGraphLaunch` ×3 = **3.2 ms 主机时间**(单次最长 2.36 ms)—— CUPTI 下有放大,但非零;
- `cuMemcpyDtoDAsync` **387 次/轮**(单轮内 verify 前 194 次)= **2.3 ms 主机时间**;
- `cuMemcpyHtoDAsync` 25 次 = 0.12 ms;`cuMemGetInfo` 2 次(热路径设备查询,I-4 违规面)。
- 时序上表现为"propose 采样核结束 → 下一个核"**固定 4.0–4.6 ms 真空**(两臂都有,与 FI 无关)。

⚠️ **但这条在无插桩口径下几乎为零**:同档实测 `[spec-degrade] 轮 53ms`
(= verify pure 46.2 + propose pure_gpu ~7),即 **round ≈ verify + propose**,空洞只剩 <1 ms
⇒ §9.4 的 4.2 ms **主要是 CUPTI 的 graph-node 追踪放大**(cuGraphLaunch/每次 memcpy 的
主机耗时被逐节点记事件),**不是真成本**。
⇒ 结论修正:**别去做"上传合并/去 cuMemGetInfo"**;唯一剩下的杠杆还是 §9.3 的 GEMM 带宽。
(`cuMemGetInfo` 属热路径设备查询,I-4 面仍应清掉,但只有 ~0.2 ms。)

### 9.5 std 五项三轮(spec d7,v2 verify,**含刀 B**)

`docs/bench/data/gguf27b-iq4xs-std-spec-2026-10-21b.json`(五点三轮)+
`…-std-spec-hi-2026-10-21b.json`(4096/8192 补测三轮):

| 点 | 旧(spec d7) | 新 | 相对无投机(40.05) |
|---|---:|---:|---:|
| 512 | 72.4 / 73.8 / 66.2 | **82.2 / 82.9 / 72.9** | **+105%** |
| 1024 | 69.2 / 66.3 / 63.7 | **77.1 / 72.7 / 69.8** | +81% |
| 2048 | 48.7 / 45.9 / 44.5 | **53.6 / 48.8 / 52.8** | +32% |
| 4096 | 34.6 / 34.2 / 34.7 | 33.4 / 31.8 / 31.9 ‖ 34.4 / 34.0 / 33.1 | −16% |
| 8192 | 32.3 / 33.0 / 33.3 | 33.5 / 33.5 / 33.1 ‖ 32.2 / 34.0 / 33.6 | −16% |

prefill 1753(不变,1.75k 档);`verify pure` 884→**46.2 ms**、7734→**59.0 ms**(v2 臂,−8.3 ms)。
**4096/8192 仍不赚**:round@8k ≈ 64 ms、裸步 29 ms ⇒ 盈亏线 2.2,而窗口 AL 只有 1.7–2.0
—— 与 §4 H2 的预判一致:**这两点要靠 AL 或 round 再砍一半**,不是本轮的范围。

### 9.6 刀 C(`8d3bfe6`):MMQ2 J=8 档(轮 58.5 → 54.4 ms)

verify 块恒 8 行 ⇒ 直接吃 `mul_mat_q` 的最小 J 档 8(十型各补一份实例):

| 内核 | J=16 | J=8 |
|---|---:|---:|
| iq4xs | 16.58 | **15.30** |
| q4k | 6.77 | **5.91** |
| q5k | 6.15 | **5.59** |
| iq3s / iq3xxs | 5.62 / 2.08 | 5.61 / 2.06 |

⇒ **定谳:小 n 的 MMQ2 不是 MMA 算力受限**(J 减半只换 8~13%);J=64→16 那次的 2.3×
主要来自 smem/占用率,**不能按"线性砍算力"外推**。真墙仍是 §9.3 的 CTA 数。

std(2 轮,out 256;`--lengths 512,1024,2048,8192`):

| 点 | 9.5(刀 B) | **9.6(刀 B+C)** | 相对无投机 |
|---|---:|---:|---:|
| 512 | 82.2 / 82.9 / 72.9 | **89.0 / 89.2** | **+122%** |
| 1024 | 77.1 / 72.7 / 69.8 | **82.9 / 80.0** | **+105%** |
| 2048 | 53.6 / 48.8 / 52.8 | **57.2 / 61.8** | **+48%** |
| 8192 | 33.5 / 33.5 / 33.1 | 33.1 / 33.4 | −16% |

### 9.7 下一步队列(按 ROI 排序)

| # | 项 | 预期 | 代价 |
|---|---|---|---|
| 1 | **MMQ2 stream-k**(k 向切分;详见 §9.8 —— 唯一能翻盘的杠杆) | round 54→**45**;8k 由 −16% 翻 +15% | 大,动 4 文件 + 工作区账 |
| 2 | ~~host 空洞~~(9.4 已证伪:是 CUPTI 放大)⇒ 只清热路径 `cuMemGetInfo` | ~0.2 ms | 低 |
| 3 | **ssm_alpha/beta 小 GEMM**(n=48,96 发 3.3 ms):专用 k-split 单发核 | −2.5 ms | 中 |
| 4 | FI verify 数值对齐(9.1)后转正 | @8k −15 ms,且 ctx 平坦 | 中(数值取证) |
| 5 | Q2_K 的 D2S6 量化手(1 张 19 MB) | −0.1 ms | 低但无收益,挂账 |

### 9.8 下一刀(stream-k)的**可执行**拆解 —— 三个已探明的硬障碍

目标:`gx = nsm(84)` 代替 `gx = n/128`(40/48/80/136),让 40 CTA 档从 435 GB/s
爬到 712 GB/s 的 DRAM 顶。上游 `mul_mat_q` 的 stream-k 分支**已经在** vendored 代码里
(`mmq.cuh:1060-1240`),我们只是从没进过它。

1. **`gx > ntiles` 必走 `fixup=true` 分支**(`mmq.cuh:1215`),该分支写 `tmp_fixup`;
   传 nullptr 立刻 IMA(已复现:grid 硬拉 168 → 启动即 700)。⇒ 必须给工作区。
2. **`mul_mat_q_stream_k_fixup` 的 dst 是 `float*`**(`mmq.cuh:1241`),而本适配层
   的 `dst` 是 `__half*`(fork 改过 `process_tile` 直接写 half)⇒ **fixup 核要改成
   half 累加**(两处 `dst[...] += sum` → `__float2half(__half2float(dst[...]) + sum)`)。
   不改 = 静默错值,不是崩。
3. **算子层是一 op 一发射**(`KernelPick { name, shape }`),没有链式;而 fixup 必须
   紧跟主核同流发射。两条可行路:
   - (a) `linear.rs` 连发两 op:主核带 `tmp` 槽 → `GgufMmq2Fixup` 带 `tmp`+`out` 槽。
     **阻塞点**:`tmp` 要有图内**稳定指针** ⇒ 必须在装载期为每个 `Linear` 备一块
     `nsm·J·I·4 B`(J=8 时 84·8·128·4 = 344 KiB)的常驻 scratch,并把句柄穿到
     `gguf_quant_gemm`(`Linear` 增字段)。
   - (b) 核内"末块归约":每块做完 `__threadfence()` + 原子加计数,最后一块广播槽,
     由它跑 fixup(688 KiB 读,单块 ~3 µs)。**不需要第二次发射**,但要在 vendored
     核里加同步(30–50 行),且计数槽仍要常驻 scratch(可复用上条的工作区)。
   **推荐 (a)**:改动都在适配层,fork 面最小;工作区是"每层一块",与现有
   `Weight` 账同族。
4. 判据(上游同款,`mmq.cuh:1443`):`ntiles` 的波效率 `eff = 100·ntiles/(nsm·ceil(ntiles/nsm))`,
   `eff ≥ 90` ⇒ 保 tiling(gx=ntiles,**无 fixup**);否则 `gx = nsm`。
   本档:n=5120(ntiles=40,eff 47%)/6144(48,57%)/10240(80,95%→保 tiling)/17408(136,81%)
   ⇒ 前两档进 stream-k,后两档大概率仍 tiling。**lm_head(1940)永不动**。
5. 门:数值相对差 ≤5e-2(同 `mmq2_small_m`)、`verify pure` @884 与 @8192 各一发、
   `OWL_GGUF_MMQ2=0` 一键回退已有。**先只挂 IQ4_XS 一型**(占权重 50.7%、时间 28%),
   拿到读数再铺开。

### 9.9 2026-10-21 下午:**速度三刀**定谳(queue 重排)

三件事同批落地,账与 A/B 表见 `docs/bench/qwen38-3090单卡-引擎会战.md` §K。

| 刀 | 结论 | 收益 |
|---|---|---|
| **P1 小 n 走 MMVQ** | `ssm_alpha/beta`(n=48)在 tile 臂只有 12 CTA、34.2 µs/发;MMVQ 5.1 µs ⇒ 判据 =「`m ≤ TILE_MT` 且张量 ≤ 4 MiB(装得进 L2)」 | verify **−2.8 ms/轮**(45.2→42.4) |
| **P3 注意力臂 = FI 双面** | v2 分页核把 8 行 verify 拖在长 ctx(512 档 1.9 → 8k 档 **~15 ms**);FI 当一组算,f16 池直接用 | 8k verify **56.5→41.1**(−27%),decode 28.1→**39.9 t/s** |
| **P2 k 向分段** | **证伪**:算子层连发 S 段是同流依赖链,不并发,每多发 ≈**+3.2 µs** ⇒ S=2/4/8 = +1.4/+5.4/+11.4 ms(**越切越慢**)。数值面(链 vs 单发)5.4e-4~1.2e-3 ⇒ 机制没错、调度模型不成立 | — |

**P2 的连带结论(重要,省掉将来的弯路)**:

1. k 向分块只能是**单次发射内**的 stream-k(`gx = nsm`),不能靠算子层连发;
2. §F 的 stream-k 探针只值 **1.22×**,而真做要 fixup 第二发(+459 发 ≈ +1.5 ms)
   ⇒ 净值 ≈ **−4 ms(−7%)**,已降到"模型侧两条"之后;
3. **MMQ2 小 n 的墙不是 CTA 数**:同 k 内维实测 40 CTA 435 / 80 490 / 136 570 /
   1940 712 GB/s,而 `load_tiles` 是**同步** smem 装载(无 cp.async 流水)——
   要上 Marlin 级 600+ 只能改核内流水,那已超出"改适配层"的范围。

**queue 现状**(取代 §9.7):

| # | 项 | 预期 | 状态 |
|---|---|---|---|
| 1 | **模型侧**:pre-FC tap norm(EAGLE-3.1 病因①;长 ctx AL 可达 2×) | AL 1.33→2.6 级 | 未做(用户本轮明示先不做接受率) |
| 2 | **模型侧**:换目标量化(Q4_K_M / AWQ;AWQ 档 AL 已实测 2.85) | 同上 | 未做 |
| 3 | 核内**流水化** MMQ2(双缓冲 smem + cp.async;10 个 load_tiles 家族) | verify 35→~25(GEMM 74%→) | 大工程,未立项 |

#### 9.9.1 追加:P6(J=16 档 I=64)已落地,+13~17% 全线

| 项 | 结论 |
|---|---|
| **P6 I=64 / 128 线程** | `ntx=1` 时 tile 数 = `n/I` ⇒ I 减半 = **SM 覆盖 ×2**。verify 41.2→**35.3**,round 52.5→45.7,std 五点中位 **+12~17%** |
| P5 寄存器预取 | **零收益**(verify 41.2→41.5):`#pragma unroll` 的 16 发早已全在飞,且 40 CTA 时每 SM 10.2 GB/s ≈ 份额上限 11.1 |
| I=32 | **下界 = 64**:I<64 时某型 `load_tiles` 越界(起机 CUDA 700) |

⇒ §9.9 表里的"#3 核内流水化"改写为:**流水化已被证伪**(不是没做,是做了没用);
剩下的 GEMM 空间只有"把 SM 全铺开"这一条,而 I=64 已经拿到 80/84 SM。
下一步若还要 GEMM,只剩 **stream-k(I=64 之上再 k 切)** —— 收益上限从 2× 降到 ~1.2×。

| 4 | ~~P2 k 向分段(算子层)~~ / ~~P5 寄存器预取~~ | **双证伪,勿再试** | 已关闭 |
| 5 | FI `fixed_split_size`(≥16k 才有 1.6×) | 长 ctx | 挂账 |
| 6 | gdn_recurrence(3.65 ms)+ ssm 小 GEMM 之外的碎核(≈2 ms) | ~1.5 ms | 挂账 |

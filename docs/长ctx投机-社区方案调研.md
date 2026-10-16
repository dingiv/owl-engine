# 长上下文投机解码:我们的衰减 vs 社区方案(2026-10-15;perf 会话)

> 触发:用户判断"对 ctx 长度的敏感性太大,而且我们的投机核心是仿 xinfer 的社区实现,不是正规军"
> ⇒ 要①把衰减归因钉死 ②去社区找方案。
> 本文 = 两边对账 + 候选清单(带判据)。数据全部来自本 worktree 的实测
> (`docs/长ctx归因-首轮.md` + `perf-runs/`)。

---

## 一、我们的衰减:只有一个真项,它是**核效率**问题

| 项(每轮) | 2048 | 32768 | Δ | 性质 |
|---|---:|---:|---:|---|
| **verify 图设备**(其中 XQA 注意力) | 11.9(3.19) | **33.5(24.66)** | **+21.6** | **ctx 线性** ← 唯一真项 |
| 图派发税(1484 节点 × ~11 µs) | ~18 | ~16.5 | ~0 | 固定/轮 |
| 草稿 propose | 8.69 | 8.75 | ~0 | 固定(草稿注意力本就有界 ✓) |
| snap/fold/emit | 2.8 | 2.8 | 0 | 固定 |
| host + logits 读 | ~6.4 | ~6.4 | 0 | 固定 |
| **轮时** | **49.1** | **69.8** | **+20.7** | |
| **ms/token**(÷(AL+1)) | 15.17 | 15.82 | **+0.6** | **近乎平坦** |
| 对照:裸 decode ms/token | 25.02 | 36.41 | +11.4(+46%) | 有投机摊薄 |

**两条由数据直接读出的修正**(与直觉相反,值得记住):

1. **我们的"衰减"不是草稿接受率衰减**:AL(接受长度)= 3.24 / 3.51 / **4.41**(2k / 8k / 32k)——
   **越长的 ctx 反而接受得更多**(草稿 cross-attn 吃 target hidden,上下文越多越好猜)。
   真正的衰减 = **verify 注意力对 ctx 线性**(+21.6 ms/轮,占一轮 29%→35%)。
2. **该核的效率只有 2.09 TFLOPS(峰值 5%)**;即便按"8 行重读一遍 KV"算,有效带宽也只有
   62–174 GB/s(峰值 7–19%)。**同样的 KV 量,一个正常的核应该在 ~1 ms/轮量级**
   ⇒ 我们的"ctx 敏感性"首先是**核问题**,不是算法问题。
3. 次要项:prefill 侧 FI 注意力随 ctx 线性(−17% prefill t/s),**但那个核已在计算屋顶**
   (21–44 TFLOPS,50–100% fp32-累加峰值)⇒ prefill 的衰减是 O(T·ctx) 固有计算。

> 与社区论文的**定量吻合**:Dustin(ICML'26)测 32k/batch16 下 **verify 占解码延迟 87.5%**;
> 我们 32k/batch1 实测 **verify.read 占一轮 71%**。同一现象,我们是单批版。

---

## 二、社区方案(四路;带出处)

### A. 把 verify 注意力本身做快(稠密路线)——最成熟

- **分块 KV + 归约(Flash-Decoding / split-KV)**:vLLM / SGLang 的 decode+verify 全走
  paged split-KV + LSE 归约;长 ctx 单批时并行度只能靠"沿 KV 维切"。
- **LongSpec 的 Hybrid Tree Attention**(ICLR'25,Sail/北大):
  *"把 KV 分成两组:cached tokens(无需 mask)与 speculative tokens(需 mask),再用
  log-sum-exp 合并 —— 从而让 prefix 计算直接吃优化过的快核(Flash-Decoding)"*。
  ⇒ **这正是我们 verify 的形状**(长无 mask 前缀 + 块内 8 行因果),也正是 XQA 在该形状
  跑不动的结构性原因(它把 8 行拆成 2 个 token block 重读 KV,且 `rowsPerBlock ≤ 32` 是硬门)。
  出处:https://sail-sg.github.io/LongSpec/
- **MagicDec**(2024):长 ctx/大 batch 下投机仍赢的关键 = *"把 full-KV 的目标验证摊薄到
  多个草稿 token 上"*,并用 **sparse-KV drafting** 压草稿开销。与我们的实测同构
  (我们 verify 8 行共享一次 KV 读 ⇒ per-token ctx 敏感性砍半:0.37 → 0.16 ms/1k ctx/token)。

### B. 稀疏 verify / KV 选择(长 ctx 专属,研究前沿)

- **Dustin**(ICML'26,NYCU/TU Darmstadt):*"验证阶段的关键 token 选择"*——
  融合 **草稿 lookahead 信号** + **目标模型历史注意力**;并只对**少量 head** 打分以压开销;
  32k 下 self-attention **27.85×**、端到端 **9.17×**(Qwen2.5-72B,batch16)。
  明确否掉两条老路:**静态驱逐**(显著度漂移伤精度)、**动态全量打分**(开销 > 收益,>4k 就该退)。
  出处:https://arxiv.org/html/2606.24957
- **Quest**(2024):page 级 min/max key 统计做选择(便宜但仍非免费)。
- **MagicDec**:sparse-KV drafting(草稿侧稀疏)。

### C. 草稿侧的长 ctx 适配(针对"训练短、推理长")

- **LongSpec**:① **常数大小草稿 KV**(滑窗自注意力 + cross-attention 复用 target KV);
  ② **Anchor-Offset 位置索引**(短数据训练也能在大位置号上工作,治分布漂移);
  ③ embedding / lm_head 与 target 共享。长文档/长推理上 mean accept ≈ 3.5–4,加速 2.5–3.3×。
  **我们的 DFlash2 已经天然满足 ①(草稿注意力与 ctx 无关,实测 8.69→8.75 ms/轮)** —
  这条我们已经"抄对了"。
- **KVShot**(2026,阿里 Qwen 团队):把"草稿随投机步号 k 接受率递减"命名为 **long-range decay**,
  并给出机制解释:隐藏态是**有偏上下文压缩**(按当前 query 聚合、压制对后续 step 重要的信息);
  提出 **KV-Reuse 假设**(让草稿直接吃 target KV)。结论要**冷静记**:
  *"KV 复用提升了长程接受率,但在现有训练管线下端到端收益**边际**"*,
  两个结构瓶颈:浅草稿难估 target query、草稿侧 KV 投影梯度稀疏 ⇒ 指向 **block-wise 训练**。
  出处:https://arxiv.org/html/2604.26412v2
- 工程侧信号:NVIDIA 已放出 **`gpt-oss-120b-Eagle3-long-context`** 长 ctx 专用草稿
  ⇒ "长 ctx 草稿要专门训"已是社区共识。

### D. 自适应投机长度(最成熟的工程手段)

- **SGLang `--speculative-adaptive`**(EAGLE/EAGLE3,topk=1):
  按 **batch size 分档**各自维护 EMA(观测接受长度);
  策略 = `target_steps ≈ clamp(round(ema_accept_len) + 1, min, max)`;
  带 warmup / update_interval / 上下行滞回 / ceiling_coeff(按草稿质量给上限,防高 BS 过度投机);
  **关键工程约束:每个档位预捕获自己的 CUDA graph + 后端状态,切档 = 换引用,不在线重捕**。
  出处:https://docs.sglang.io/docs/advanced_features/adaptive_speculative_decoding
- **llama.cpp**:`--spec-draft-n-min / n-max`(草稿长度上下界)、
  `--spec-draft-p-min P`(贪心下最小投机概率,低了就停)、
  以及 **DSpark 的 `--spec-draft-conf-min P`**:"按草稿置信头逐位置截断 block"
  —— 对 **DFlash/DSpark 家族(与我们同族)** 的块扩散草稿,这是官方推荐的自适应形态。
  出处:https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md
- **vLLM**:`num_speculative_tokens` 静态配置 + 动态 SD(按负载调 k);文档亦承认
  "verify 的 K 倍放大"(BS×K)是主要成本项。
  出处:https://docs.vllm.ai/en/latest/features/spec_decode.html

---

## 三、能力对账(社区 vs 我们)

| 能力 | 社区 | owl 现状 | 差距性质 |
|---|---|---|---|
| verify 注意力核 | paged split-KV + Flash-Decoding + Hybrid Tree Attention | XQA SPEC_DEC,**2.09 TFLOPS**;FI 只接 prefill | **大(10× 量级)** |
| verify 稀疏化 | Dustin / Quest / MagicDec | 无 | 研究级 |
| 草稿 KV 有界 | LongSpec 常数 KV | ✓ 天然(注意力与 ctx 无关) | 已对齐 |
| 草稿接受率 | EAGLE-3+TTT、长 ctx 专用草稿 | z-lab DFlash2(第三方) | 模型选择问题 |
| 自适应深度 | SGLang EMA 分档 + 预捕获图;llama.cpp n-min/max + conf-min | **无**:depth=7 静态 | 中(基础设施半成:fold 桶 m=0..6 已是"分档预捕获") |
| 块内截断 | DSpark conf-min / llama.cpp p-min | 无(整块 8 行都 verify) | 中 |
| KV 量化 | fp8 / int4 | ✓ fp8 生产 | 已对齐 |
| 树形 verify(提 AL) | EAGLE-2/3、Medusa | 无(线性块扩散草稿) | 大,但依赖草稿类型 |

---

## 四、候选清单(按性价比;每项带判据)

| # | 动作 | 依据 | 判据 | 量级 |
|---|---|---|---|---|
| **V1** | **verify 换快核:FI(prefill 核)接管 verify + 开 split-KV** | LongSpec Hybrid Tree Attention 同构;FI 在 T=512/32k 已达 **4.66 ms/发 = 44 TFLOPS**(XQA 同 ctx 1.54 ms/发 = 2.09 TFLOPS) | 单算子探针:T=8…32、ctx=32k 下 ≤0.3 ms/发;端到端轮时 −15 ms | 2–3d(表件已在 `ChunkTables`,需给 verify 路造 FI 四件套 + plan) |
| **V2** | **verify block 截断**(按观测 AL/置信截断草稿) | llama.cpp `p-min` / DSpark `conf-min` | 同 AL 下轮时 −X;AL 不掉 | 1–2d(需按 depth 预捕获 verify 图) |
| **V3** | **自适应深度分档**(EMA + ctx/AL 双信号) | SGLang `speculative_adaptive` 全套(含滞回/ceiling/预捕获图) | 长 ctx 轮时/接受长度 联合不劣于最优静态档 | 2–3d(我们有 fold 桶先例) |
| **V4** | **稀疏 verify**(Dustin 式:草稿 lookahead + 历史注意力选 KV) | Dustin 27.85× attn / 9.17× e2e @32k | 先离线复现选择策略精度,再做核 | 1–2w(研究级) |
| **V5** | **草稿侧**:换长 ctx 专用/更好草稿;或按 KVShot 结论评估"block-wise 训练" | KVShot 明确:KV-Reuse 端到端**边际**收益;NVIDIA 已出长 ctx Eagle3 | 分域 AL 表(MATH/AGENT/CODE/PROSE) | 模型侧立项 |

**建议执行序**:V1 →(并行)V2/V3 → 视 V1 结果决定要不要碰 V4。
理由:**V1 是唯一"零研究风险、预期 10× 核效率、且属于社区已证实形态"的一项**;
V2/V3 是纯工程(把已有桶机器用起来);V4 收益最大但要求我们自研选择策略 + 精度门。

---

## 五、附:本次调研的两条"别再自己发明"的结论

1. **我们的 ctx 敏感性主凶是核,不是投机算法** —— 判定依据见 §一(唯一真项 = verify 注意力线性,
   而它只有 5% 峰值效率)。社区对同一现象的第一反应也是"把 verify 做快"(Hybrid Tree Attention /
   split-KV / Flash-Decoding),而不是换算法。
2. **自适应深度必须配"按档预捕获的图"** —— SGLang 明确写了这条工程约束(shape-dependent
   graph runner ⇒ 每档自己的图,切档只换引用)。我们已有 fold 桶(m=0..6)与 verify 桶(T=depth+1)
   两套现成机制,照抄即可,不要走"在线重捕"的路(会被我们自己的"捕获静止点律 + slab 定量"反噬)。

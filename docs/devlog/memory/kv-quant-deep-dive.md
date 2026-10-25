# KV 量化方案深度调研：FP8 / TQ k8v4 / KVarN（2026-10-25）

> 触发：用户问「llama.cpp 生态的 TurboQuant 与 vLLM 的 KVarN 是不是一个东西？
> 质量优先应该选哪个？」。本卷只做**事实核对 + owl 集成成本**，不落实现。
> 相关参考：
> - `repos/vllm-ampere-optimized/.../turboquant/config.py`（TQ 预设）
> - `repos/HyperQwen/kvarn/`（KVarN 0.30 移植与 Triton 核）
> - vLLM 官方评测：<https://vllm-project.github.io/2026/05/11/turboquant.html>
> - llama.cpp 生态对照（第三方基准）：<https://www.banandre.com/blog/kv-cache-quantization-benchmarks-turboquant-overrated-kvarn>

## 0. 结论先行

1. **三者不是一个东西**：
   - llama.cpp 生态的 TurboQuant：llama.cpp 自己的 KV 量化实现/预设；
   - vLLM 的 TurboQuant：vLLM 后端的 storage-only 量化（反量化回 BF16 再做 attention）；
   - KVarN：华为 CSL 的另一套算法（Hadamard + 迭代方差归一 + 128-token tile，
     K4/V2 或 K4/V4），不是 K8V4。
2. **质量优先时，TQ `k8v4` 不是好主方案**：
   - vLLM 官方结论：`turboquant_k8v4` 精度与 BF16/FP8 同级（>98–99% recovery），
     但容量只有 **2.4× vs BF16**（FP8 是 **2×**），吞吐/延迟却要付
     **约 20–25%** 的税；官方原话是「相对 FP8 没有显著优势，不值得」。
   - 换成 owl 的 27B 几何，K8V4 只比 FP8 多 **1.32×** 容量。
3. **质量 + 容量都要，KVarN 更对**：
   - HyperQwen 实测 `kvarn_k4v2_g128`：PPL **+0.16%**，262k 可跑，
     单 3090 池 302–420k token（FP8 约 200k）。
   - 代价：decode 明显变慢（100k batch 约 +20%，112k 单用户 + spec 约 **2.13×**），
     且它是**新的 tile/backend**，不是换一个 dtype。
4. **owl 集成成本是决定性因素**：
   - K8V4 可以保持 **page=32 / store / 前缀缓存 / v2 核** 不变，只需新增
     写核 + decode 核 + chunked prefill 核；工程量中等。
   - KVarN 的 tile **必须等于 block**（g128，设计点），而 owl 现在是
     **32-token 页 + 32-token 前缀哈希 + 32-token store 槽**。移植 KVarN
     等于新增一套 128-token tile 后端，并处理 sink/tail pool、tile 级
     prefix cache、spec verify；工程量远大于 K8V4。
5. **建议**：
   - 生产默认继续 **FP8**（质量最稳、零 decode 税）。
   - 若要突破 FP8 容量且质量优先：**KVarN k4v4（质量档）/ k4v2（容量档）**
     作为独立专项，先做 128-token tile 的可行性/微基准，不要把它伪装成
     「K8V4 的另一个预设」。
   - K8V4 仅作为**低风险小步**/对照实验；若目标是质量，它不是最优解。

## 1. 名词与语义核对

### 1.1 llama.cpp 生态的 TurboQuant

llama.cpp 社区 fork（如 TheTom/beellama）把 TurboQuant 做成 llama.cpp 的
KV cache 量化预设（`turbo2` / `turbo3_tcq` / `turbo4` 等），实现、格式、
kernel 都在 llama.cpp/ggml 内。它和 vLLM 的 `turboquant_*` **同名同源思路，
但不是同一实现，也不是同一种 cache 格式**。

第三方基准（Qwen3.6-27B / 单 3090 / llama.cpp fork）的结论：
- 普通 q4/q5 已经吸收了 TurboQuant 的 rotation 思路，TurboQuant 在 4-bit
  附近没有明显质量优势；
- TurboQuant 更慢（turbo4 prefill ~705 vs q4_0 ~850 tok/s）；
- KVarN 才把「质量-内存曲线」整体下压一档（kvarn4-kvarn4 约 q5 级质量、
  27.9% BF16 内存）。

> 这些是 llama.cpp 生态的数据，不能直接当 owl 的实测；但方向与 vLLM 官方
> 结论一致：**TQ k8v4 的收益主要不是质量，而是比 FP8 多一点容量。**

### 1.2 vLLM 的 TurboQuant（`turboquant_k8v4`）

代码事实（`repos/vllm-ampere-optimized/.../turboquant/config.py`）：

```
turboquant_k8v4:
  key_quant_bits = 8      # K 直接 e4m3，不做 rotation/MSE
  value_quant_bits = 4    # V 每个 (token, head) 向量 min/max 非对称 4-bit
  key_packed_size = hd
  value_packed_size = hd/2 + 4   # + f16 scale + f16 zero
  slot = hd + hd/2 + 4
```

vLLM 官方评测（H100、4 个模型、长上下文检索 + 推理）：
- 精度：`k8v4` 与 BF16/FP8 同级，长上下文检索和 reasoning 都 >98–99% recovery；
- 容量：2.4× vs BF16（FP8 2×）；
- 性能：所有 TQ 变体都严格慢于 BF16；k8v4 吞吐约 BF16 的 75–80%，
  prefill/decoding 都有 dequant 税；
- 官方建议：**FP8 仍是最好的默认；k8v4 相对 FP8 没有显著优势。**

### 1.3 KVarN

代码/论文事实（`repos/HyperQwen/kvarn/README.md`、`config.py`、`sinkhorn.py`）：

```
per (block=128 tokens, head):
  1. Hadamard rotation along head_dim
  2. 迭代 log-domain variance normalization（Sinkhorn，8 iters）
  3. 非对称 per-row RTN，K=4-bit，V=2/4-bit
  4. 把 RTN scale/zero 吸收进 Sinkhorn 的 per-channel / per-token 轴
layout:
  K packed [D, group/2] + s_col_K[D] + zp_K[D] + s_row_K[group]
  V packed [group, D/8 or D/4] + s_col_V[D] + s_row_V[group] + zp_V[group]
  tail/sink: 前 128 token 保持 fp16；未满 128 的尾块在 fp16 tail pool
```

关键点：
- **tile size 必须等于 block size**（`group=128` 是设计点；g64 是更细粒度备选）。
- KVarN 不是「K8V4」：它是 K4/V2 或 K4/V4。
- 它需要自己的 attention backend / Triton kernels / 布局，不是 vLLM 的
  `turboquant_*` 预设。
- 质量：HyperQwen `k4v2_g128` 实测 PPL **+0.16%**；llama.cpp 第三方基准
  `kvarn4-kvarn4` 约 q5 级、`kvarn4-kvarn3` 在 q4 内存以下仍有更好 tail。
- 性能：100k batch decode 约慢 20%；112k 单用户 + spec 约 **2.13×** 慢；
  prefill 基本持平。短请求并发也会受 128/2048-token 块粒度影响。

## 2. 按 owl 27B 几何算容量

owl 生产模型：16 个 full-attention 层 × 4 KV heads × hd256（单卡/单 rank；
TP2 再除以 2）。每 token 主池字节：

| 方案 | 每 token/head | 每 token（16×4） | 相对 FP8 | 相对 F16 |
|---|---:|---:|---:|---:|
| F16 | 1024 B | 65,536 B | 0.5× | 1× |
| FP8 | 512 B | 32,768 B | 1× | 2× |
| TQ K8V4 | 388 B | 24,832 B | **1.32×** | 2.64× |
| KVarN k4v4 g128 | 274 B | 17,536 B | **1.87×** | 3.74× |
| KVarN k4v2 g128 | 210 B | 13,440 B | **2.44×** | 4.87× |

> KVarN 的字节含 tile 内 K/V scale/zero、sink/tail 的固定开销另计；
> 上表是稳态 int4 tile 的主池部分。KVarN 的 128-token tile 还会带来
> prefix cache/store 粒度与短请求开销，实际收益要打折。

结论：
- K8V4 只多 32% 容量，换 20%+ decode 税，**质量优先时性价比低**。
- KVarN k4v4 多 87% 容量，质量约 q5 级；k4v2 多 144% 容量，质量 +0.16% PPL。
  两者都比 K8V4 更符合「质量/容量」目标。

## 3. owl 集成成本

### 3.1 K8V4（page=32 兼容）

需要：
- `KvQuant::K8V4` / 池预算 / 非对称 V slot 分配；
- K0 写核（K→e4m3，V→int4 + f16 scale/zero）；
- v2 decode/verify 核（D3 lane→dim 通道读 int4）；
- chunked prefill 核（smem 装 V 时反量化）；
- 配置/启动门：禁 FI/XQA/fused insert/prefill split/image_evict；
- 前缀缓存/store 不变（仍按 32-token 页字节搬运）。

工程量：中等，且与现有 page32/store/v2 路径兼容。风险主要在 CUDA 模板
和 V 的非对称 stride，不在架构。

### 3.2 KVarN（tile=128 = block）

需要的不只是 dtype：
- **128-token 逻辑块**：store/前缀哈希/页表/快照/image_evict 的粒度都变；
  当前 `KvStore`/`PoolLayout`/`KvPagedPolicy` 都是 32-token 页。
- **KVarN tile 布局**：每 (block, head) 一个 packed record（K/V 数据 +
  4 组 scale/zero），不是 owl 现在的 K/V 两个平面。
- **sink/tail pool**：前 128 token 保持 fp16；未满 tile 的尾部在独立 fp16
  pool；CUDA graph 需要固定 size，连接 scheduler/并发上限。
- **attention backend**：decode/verify/prefill 都要读 packed tile + scale，
  Triton 核（或 CUDA 重写）；spec verify 另有 fused verify 路径。
- **prefix cache**：KVarN 的 tile 级共享、sink 复住、retired sink 生命周期
  与 owl 的 32-token 内容树不同；需要重新设计。
- **image_evict**：K 已被旋转/量化，无法做现有 f16 K 重转 ⇒ 不支持。

工程量：远大于 K8V4，接近「新增一个 KV backend + 存储粒度改造」；
不是一两天的 dtype 接线。

## 4. 质量/速度/容量的取舍

| 目标 | 首选 | 理由 |
|---|---|---|
| 质量最稳、零风险 | **FP8（现役）** | 精度接近 BF16，2× 容量，无 decode 税 |
| 比 FP8 多一点容量、低风险 | TQ K8V4 | 1.32× FP8，质量接近；但 decode 税，收益有限 |
| 质量 + 明显更大容量 | **KVarN k4v4** | 1.87× FP8，约 q5 级质量；需新 backend |
| 容量优先、可接受质量/速度 | **KVarN k4v2** | 2.44× FP8，+0.16% PPL；decode 税更大 |
| 极限压缩 | KVarN k4v2 / k3v3 | 质量/速度代价高，只作生存档 |

**质量优先的裁决**：
- 不建议把 K8V4 作为生产默认；它相对 FP8 的容量收益不足以抵消 decode 税。
- 若必须突破 FP8 容量，选 KVarN k4v4（质量）或 k4v2（容量），
  作为独立专项推进。
- K8V4 可以保留为「低风险小步/对照实验」，或者直接搁置。

## 5. 建议的下一步（调研阶段）

1. **先测 FP8 基线**：在 owl 27B 单 3090 / 3080 TP2 上记录
   PPL/任务质量、std decode/prefill、长上下文检索；没有基线就无法判断
   KVarN 的收益。
2. **KVarN 纸面/源码复核**：
   - 128-token tile 与 owl page32 的冲突是否有低成本折中（g32？质量未验证，
     不建议直接上）；
   - sink/tail pool 能否映射到 owl 的 store/快照；
   - DFlash2 verify 是否必须改核。
3. **微基准**：先用 toy/单层数据对拍 KVarN k4v4 的量化误差与 decode 核
   吞吐，再决定是否做 128-token backend。
4. **K8V4 WIP**：当前在 `perf/load-speed` worktree 的 stash 里
   （`wip(k8v4): layout/config/kernels partial`）。若最终选 KVarN，
   保留为 spike 或回退；若选「低风险小步」，再恢复。

## 6. 开放问题

1. owl 是否愿意接受 **128-token 前缀缓存粒度**？这是 KVarN 的硬前提。
2. 质量目标到底是 PPL、任务通过率，还是长上下文 tail（99.9% KLD）？
   KVarN 的优势主要在 tail；如果只看平均 PPL，FP8 可能已经够。
3. DFlash2 + KVarN 的 verify 是否值得做？HyperQwen 实测 spec 下 KVarN
   decode 税可达 2.13×；owl 的 spec 是核心能力，不能忽略。
4. 是否接受「KVarN 只做 batch/长上下文档，单用户 spec 继续 FP8」的
   双档策略？这可能是质量/速度最务实的组合。

---

**一句话**：TurboQuant、vLLM TQ、KVarN 不是同一个东西；质量优先时，
**FP8 是默认，KVarN k4v4 是值得立项的容量升级，K8V4 只是低风险小步**。

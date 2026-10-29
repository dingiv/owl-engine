# awq-forge 需求文档

> 状态：立项冻结（2026-10-29）
> 定位：独立的 **FP8/BF16 → AWQ INT4** 离线量化工具，不负责推理。

## 项目目的

owl、vLLM、sglang 都只负责“装载和跑”已经量化好的模型；本工具负责把源检查点
（BF16，或 FP8 blockwise/per-tensor）转成 **AWQ W4A16、group_size=32、asymmetric**
的 `compressed-tensors` `pack-quantized` 检查点，布局与
`models/cyankiwi/Qwen3.8-27B-AWQ-INT4` 一致。

目标不是“再写一个 AWQ 算法”，而是：

1. 用社区高性能实现（GPTQModel / AutoAWQ）做 AWQ 数学与逐层调度；
2. 解决它们对 Qwen3.5/3.8 混合 GDN + VLM + MTP 架构、FP8 源、低显存的适配；
3. 做可靠的 compressed-tensors 重打包、结构对拍和质量门；
4. 把 27B 的 FP8 源做成**逐层流式反量化**，不落 54GB dense 中间态。

## 目标用户

- 模型维护者：把官方 FP8 模型转成 24G 消费卡可跑的 AWQ INT4。
- 推理引擎使用者：需要 vLLM / sglang 直接装载、行为可复现的量化产物。
- owl 维护者：需要与现有 `AwqSource` / compressed-tensors 布局严格一致的权重。

## 核心场景

### S0：0.8B 冒烟（开发期）

- 触发：`awq-forge run --source models/Qwen/Qwen3.5-0.8B ...`
- 输入：BF16 dense Qwen3.5-0.8B，24 层，1.7GB。
- 输出：AWQ g32-asym compressed-tensors 检查点。
- 终止：产物可被 vLLM/sglang 装载，固定 prompt 生成连贯；记录耗时/显存。
- 作用：验证 Qwen3.5 混合层、GDN、VLM、MTP 的模块发现与 AWQ 接线；
  不承担质量结论。

### S1：4B BF16 → AWQ INT4（P0 验收）

- 触发：`awq-forge run --source models/empero-ai/Qwen3.8-4B-Distill ...`
- 输入：BF16 dense VLM，8.68GB。
- 校准：UltraChat 128 条 × 2048 token。
- 输出：compressed-tensors `pack-quantized`，g32-asym，忽略 vision/MTP/norm/embed/head。
- 终止：单卡 24G 完成，耗时 ≤ 5 分钟；质量门通过；vLLM/sglang 装载成功。

### S2：27B FP8 → AWQ INT4（P1）

- 触发：`awq-forge run --source models/Qwen/Qwen3.8-27B-FP8 ...`
- 输入：FP8 blockwise 128×128，30GB。
- 输出：同 S1 布局的 27B AWQ INT4。
- 终止：双卡 24G 上完成；**不写 54GB dense 中间目录**；产物可装载。
- 质量门：GSM8K 200 不低于 BF16 基线 -1pp；needle 16k 可检索；std 15/15。
  （2026-10-29 27B 实测：BF16 不可得，参考臂用 FP8 源 + vLLM Marlin W8A16；
  同预算 1536 下 AWQ 96.5% vs FP8 97.0%，Δ=-0.5pp、McNemar p=1.00。）

### S3：验收与回滚

- 触发：量化完成后自动执行，或 `awq-forge verify <out>`。
- 输入：产物目录 + 源目录。
- 输出：`manifest.json`、结构报告、抽样反量化对拍、引擎装载报告、质量报告。
- 终止：全部硬门通过才 `publish`；失败保留 `.work/`，不覆盖已有产物。

## 完成标准

### P0a：0.8B 冒烟

- [ ] 正确识别 24 层中 full-attention / linear-attention(GDN) / MLP / vision / MTP。
- [ ] 只量化目标 Linear；`in_proj_a`、`in_proj_b`、vision、MTP、norm、embed、lm_head 保持原精度。
- [ ] 输出 `weight_packed/weight_scale/weight_zero_point/weight_shape` 四件套。
- [ ] 产物可被 vLLM 或 sglang 装载，固定 prompt 生成连贯。
- [ ] 记录端到端耗时、峰值显存、每层耗时。

### P0b：4B 验收

- [ ] 端到端 ≤ 5 分钟（单卡 24G，128×2048，g32-asym）。
- [ ] 峰值显存 ≤ 24GB；不得依赖 CPU offload 才能完成。
- [ ] 输出布局与 cyankiwi 27B AWQ 逐键一致（键名、dtype、shape、config group）。
- [ ] vLLM / sglang 至少一个引擎装载成功，固定 prompt 正常。
- [ ] GSM8K 200 ≥ BF16 同口径 -1pp。
- [ ] needle 16k 通过。
- [ ] std 15/15 通过。

### P1：27B

- [x] FP8 blockwise 逐层解码，不产生 54GB dense 目录。
- [x] 双卡 24G 完成，显存峰值留 ≥2GB 余量（主卡 13.57 GiB）。
- [x] 输出可被 vLLM/sglang 装载（vLLM TP2 + owl 均通过）。
- [x] 质量门同 S2（参考臂 = FP8 源；GSM8K -0.5pp / p=1.00、needle 3/3、std 15/15，
      见 `docs/bench/2026-10-29-pi27b-fp8-awq.md`）。

## 约束条件

- 硬件：2×RTX 3090 Ti 24G + 2×RTX 3080 20G；P0 用单张 3090 Ti。
- Python：3.12，uv 独立 venv/lock；不碰系统 Python。
- 主后端：llm-compressor 0.15（AWQ + 原生 compressed-tensors 导出）。
- 参考后端：AutoAWQ fork `quivent/autoawq-qwen35`（只作交叉验证）。
- GPTQModel 7.7：0.8B AWQ 质量实测失败，仅保留其 FP8 module-local
  decoder 作为 27B 流式反量化候选。
- 导出：llm-compressor 原生 CT；如需 AutoAWQ 布局，用官方
  `AutoAWQConverter` / `pack_to_int32`，不手写 bit 序。
- 行为配置：CLI 参数 + TOML job 文件；**不用环境变量控制量化行为**。
- 校准：UltraChat 128×2048（P0b/P1）；0.8B 冒烟可用 64×1024。
- 输出：g32-asym 是唯一 P0 方案；g128-sym 只作后续 A/B，不混入首期。

## 范围外

- 不做推理运行时、不做 KV cache 量化、不做 W4A8/NVFP4/GGUF。
- 不做训练、不做 QAT、不做蒸馏。
- 不把量化器塞进 owl 启动路径。
- 不保证所有 HF 架构；首期锁 Qwen3.5/3.8 dense VLM。
- 不手写 AWQ 数学；先复用社区实现，只有 profiling 证明热点才写 C++/CUDA。

## 关键风险

| 风险 | 缓解 |
|---|---|
| AWQ 映射与 Qwen3.5 GDN 偏差 | 0.8B 冒烟先做键面/量化面清单对拍；llm-compressor 已实测通过 |
| FP8 blockwise 解码的 padding/边界 | 复用 GPTQModel `model_dequant`；抽样与 `compressed-tensors` 对拍 |
| AutoAWQ GEMM → CT 的 bit 序 | 只用官方 `AutoAWQConverter`，不手写；加 round-trip 对拍 |
| 4B 5 分钟目标 | 先测 GPTQModel `true_sequential` + chunked；必要时减少 grid/校准，质量门兜底 |
| 27B 双卡显存 | 逐层解码 + 单层驻留；不整模型上卡；先 0.8B/4B 验证再上 |
| 质量退化 | GSM8K/needle/std 三道门；失败不发布 |

## 待确认（不阻塞 P0a）

- 是否在 P0b 同时导出 AutoAWQ 经典布局，作为 vLLM 的 fallback。
- 27B 是否使用双卡分层，还是单卡逐层 + CPU 权重池。

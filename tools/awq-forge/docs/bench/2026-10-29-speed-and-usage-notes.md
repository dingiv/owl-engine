# awq-forge 速度与使用注意事项

> 2026-10-29 记录。分两类速度问题：
> **A. 转换慢**（`convert_*.py` / 量化本身）；**B. 评测慢**（`quality_gate.py`
> 跑 GSM8K）。两类的“变快”手段不同，别混用——转换的旋钮改不了评测，
> 评测的批处理也加速不了逐层量化。

---

## A. 转换：为什么慢，怎么变快

### A1. 为什么慢（结构性原因）

llm-compressor 的 `sequential` pipeline 是**严格串行**的：

- 一次只 onload 一个 decoder 子图，逐层做 AWQ 的 grid search；
- 每层都要重放父层输入（activation replay），再做一次 scale 搜索；
- 默认档很保守：`n_grid=20`、`batch_size=1`、`seq_offload=cpu`。

所以耗时 ≈ `层数 × (grid 次数 × 校准前向 + 反向统计)`，GPU 上全是
小块 GEMM，吃不满带宽；`seq_offload=cpu` 还会让中间激活来回走 PCIe。

### A2. 变快的四个旋钮（实测）

| 旋钮 | 默认 | 快档 | 效果 |
|---|---|---|---|
| `n_grid` | 20 | **10**（甚至 5） | 搜索点数减半；质量门内无差异 |
| `batch_size` | 1 | **8** | 校准前向批量化，摊薄 kernel 启动 |
| `seq_offload` | `cpu` | **`cuda`**，双卡时 **`cuda:1`** | 激活留在显存，省 PCIe 往返 |
| `targets_per_subgraph` | 1 个 target | **2 个 target**（27B，约 2 层） | 子图数 65→33，少一半调度 |

Qwen3.5-0.8B 实测（128×2048，g32-asym，同一张 3090 Ti）：

| 配置 | 耗时 |
|---|---:|
| n20 / b1 / cpu | 394s |
| n20 / b1 / cuda | 131s |
| n20 / b8 / cuda | 124s |
| n10 / b4 / cuda | 72s |
| **n10 / b8 / cuda** | **57s** |
| n5 / b4 / cuda | 58s |

**6.9×**，质量都是 `17*23 → 391`。

4B：`b4 + cuda` 265s → **`b8 + cuda:1` 229s**。

> ⚠️ `seq_offload=cuda` 解析成 `cuda:0`，等于没换卡。双卡必须写
> `cuda:1`：它卸的是**中间激活**，不是权重。

### A3. 27B FP8 的特殊账

- 源是 blockwise FP8（33G），不能整模型上 24G 卡；用
  `layerwise_decompression=True` 只在当前子图解 FP8 → BF16，不落 54GB
  dense 中间目录。
- 权重在 CPU、当前 2 层子图 + workspace 在 GPU：主卡峰值 13.57 GiB。
- 实测：load 32.5s / quantize 210.4s / save 28.6s，**总 271.5s**。
- 复现参数：`--seq-offload cuda:1 --batch-size 8 --n-grid 10
  --targets-per-subgraph 2 --cpu-memory 80GB`。

### A4. 转换侧注意

1. **不要开四卡 DDP 做单模型 AWQ**：llm-compressor 的 DDP AWQ 在
   finalize 阶段会 all-reduce，本机 4 卡实测 600s 超时（0.8B 也一样）。
   四卡的正确用法是**并行跑多组配置/多个模型**。
2. `pipeline=basic`（整模型上卡）在 0.8B 都会 OOM，27B 更不可能。
3. 双卡峰值看**主卡**：第二张卡只放激活（~1.6G），别按“两张卡平分
   33G”估算。
4. 27B 产物**没有 MTP**：源 MTP 是 FP8，llm-compressor 不允许原样拷。
   DFlash2 草稿是独立目录，不受影响。

---

## B. 评测：为什么慢，怎么变快（本次最大的提速）

### B1. 慢的根源：单流解码 + TP 的每步通信

- 27B AWQ 单卡 greedy 解码 ≈ **45 t/s**（19G 权重 / 22ms，已贴
  3090 Ti 带宽墙）。
- 同一模型 vLLM **TP=2、c=1** 只有 **14.5 t/s**：每 token 64 层
  allreduce 走 PCIe，**延迟主导**，加卡不加速单流。
- 按 14.5 t/s 跑 200 题 × ~370 token ≈ **85 分钟**——这就是“之前那么慢”。

### B2. 变快：把并发拉起来（批处理摊薄每步开销）

关键两步（缺一不可）：

1. **服务端**：`--max-num-seqs 8`（默认 1 会直接串行）；
2. **客户端**：`--concurrency 8`（`quality_gate.py` 用线程池同时发 8 题）。

实测 vLLM TP=2：

| 并发 | 生成吞吐 | GSM8K 200 耗时 |
|---|---:|---:|
| c=1 | 14.5 t/s | ~85 min（预估，未跑完） |
| **c=8** | **104–108 t/s** | **705–722s（≈12 min）** |

**≈7.2×**，且 greedy 逐题结果不变（同一 AWQ 产物在 owl 与 vLLM 上
200 题 0 翻转，见 `2026-10-29-pi27b-fp8-awq.md`）。

为什么批处理有效：decode 每步的时间主要花在**权重读取 + allreduce
延迟**，这两项与 batch 大小基本无关；8 条序列共享一步 → 吞吐近似
线性。它不是算力瓶颈，所以不存在“批大了算不动”。

### B3. 什么时候**不要**用批处理

- **测延迟/TTFT/ITL**（`llm_speedtest --flavor std`）：必须 c=1，
  否则指标没有意义。
- **部署路径的冒烟**：owl 的 27B profile 是 `runtime.max_concurrency=1`，
  想批处理得改 profile；质量门目前把 owl 当“单流部署复核”，批量对比
  交给 vLLM。
- **显存紧**：`--max-num-seqs 8` 会把 KV 池吃得更满；20G 卡上先算账。

### B4. 快速起一个批量评测端点（2×3090 Ti）

```bash
cd packages/vllm-test
CUDA_VISIBLE_DEVICES=<3090-uuid-a>,<3090-uuid-b> \
VLLM_SKIP_P2P_CHECK=1 NCCL_P2P_DISABLE=0 NCCL_CUMEM_ENABLE=0 PYTHONHASHSEED=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
VLLM_NCCL_SO_PATH=/home/div/Documents/codes/packages/mistral-test/lib/libnccl.so.2 \
.venv/bin/vllm serve <模型目录> \
  --served-model-name <名字> --tensor-parallel-size 2 \
  --max-model-len 20480 --max-num-seqs 8 \
  --gpu-memory-utilization 0.92 --enforce-eager \
  --disable-custom-all-reduce --port 8140 --host 127.0.0.1
```

- `--max-model-len 20480`：16k needle（实际 16233 tok）+ 输出余量；
- `--enforce-eager`：省 CUDA graph 显存（评测不缺那点速度）；
- `--gpu-memory-utilization 0.92`：27B FP8 在 2×24G 上约剩 5.5G KV；
- 务必 `--disable-custom-all-reduce`（P2P 约束）。

---

## C. 使用注意事项（踩过的坑）

### C1. 评测口径

1. **GSM8K 预算 ≥1536**。pi 是 agent 特调，均值 360–382 token；
   768 会截断 3–4% 的题，AWQ/FP8 截断数不同 → 伪造出 -1.5pp 差距。
   1536 下只剩 1 题截断，结论 -0.5pp / p=1.00。
2. **greedy + `enable_thinking=false`**，两臂用完全相同的 prompt 文本
   和模板参数；不要一边开 thinking 一边关。
3. **看配对显著性，不只看总分差**。n=200 时 1–2 题的差就是噪声；
   用 `compare_quality.py` 输出 McNemar exact p。
4. **参考臂说明**：本地没有 27B BF16，用 FP8 源 + vLLM Marlin
   W8A16（bf16 激活）当基线——对 AWQ 是保守口径。别把“FP8 基线”
   写成“BF16 基线”。
5. **跨引擎对比要同预算、同模板**；引擎采样路径不同（owl greedy vs
   temp>0）会改变逐题结果，配对分析能兜住。

### C2. needle

- `quality_gate.py` 先用 1000 词探针校准 tokens/word（本模型
  **1.012 tok/word**），再按目标 token 数生成，避免“词数≈token 数”的
  想当然。
- 16k 门要求 head/mid/tail **全中**；只测尾部会漏远端读错误。
- prompt 实际 16233–16234 tok，`--max-model-len` 必须留出输出空间。

### C3. 进程与卡

- **owl 与 vLLM TP=2 不能同时占 3090 对**。切换前先确认
  `nvidia-smi --query-compute-apps` 清空。
- 停 vLLM 用**显式 PID**（`EngineCore`、`Worker_TP0/1`、APIServer、
  bash 壳），**不要 `pkill -f`**：模式会匹配到自己的命令行（本次
  实际踩到，shell 被 SIGTERM）。
- owl 启动很快（本次 8s 就绪），不需要长 sleep；超过 30s 才有问题。

### C4. 速度数字的读法

- **std（temp=1 / top_p=0.1）decode 22.5 t/s 不是带宽上限**：
  ITL 恒 44.4–45ms、与 ctx 无关；greedy 路径 ~45 t/s。差异是采样
  路径每 token 多 ~22ms，不是量化/长上下文问题。
- 比较不同引擎的 decode 速度时，先确认采样参数一致；否则用
  `llm_speedtest` 的同一 flavor 口径。

### C5. 产物与部署

- 27B AWQ 产物：19G，400 个 Linear，visual/embed/lm_head 为 BF16，
  **无 MTP**。质量门跑 `spec.depth=0`；要投机需另接 DFlash2 草稿。
- owl 的 27B 档位（`profiles/qwen38_27b-3090.toml`）默认 `w4a8`；
  本次质量门用 `--set model.quant=w4a16` 跑原味 AWQ，两者不要混淆。
- 质量门结果与复现命令见
  `docs/bench/2026-10-29-pi27b-fp8-awq.md`；
  原始数据在**工作区** `docs/bench/data/awq-forge-27b-2026-10-29/`（含逐题
  `.items.jsonl`），**不要放进本工具源码树**——`docs/bench/data/` 已加入
  `.gitignore`，临时结果写 `out/`。

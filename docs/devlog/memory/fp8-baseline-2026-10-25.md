# FP8 KV 基线质量/速度评测（2026-10-25）

> 目的：在 KVarN / T8V4 等 KV 量化方案之前，先给现役 **FP8** 档一份可复用的
> 质量 + 速度基线。测试方法借自：
> - `repos/HyperQwen/bench/quality_battery.py`（PPL + GSM8K）
> - `repos/HyperQwen/bench/needle_test.py`（长上下文检索）
> - `repos/vllm-ampere-optimized/eval/gsm8k_api_eval.py` / `gsm8k_eval.py`
>   （API 侧 GSM8K、post-`</think>`、`\boxed{}` 抽取）
> - `repos/vllm-ampere-optimized/benchmarks/eval_oai.py`（MMLU-Pro）
>
> 代码基线：`main@b2432680`（worktree `perf/load-speed` 只放脚本/记录，未改引擎）。

## 0. 结论

| 项 | 3090 单卡 FP8 | 3080 TP2 FP8 |
|---|---:|---:|
| GSM8K n=200（thinking off, greedy） | **97.0%**（194/200） | **97.0%**（194/200） |
| GSM8K 错误 | 6（其中 5 题 `finish=length`） | 6（同 6 个 idx，5 题 length） |
| MMLU-Pro n=50（thinking off） | 60.0%（30/50，21 题截断） | 58.0%（29/50，20 题截断） |
| Needle 16k / 32k / 64k @50% | **3/3 RETRIEVED** | — |
| Needle 64k / 128k / 240k @50% | — | **3/3 RETRIEVED** |
| std decode（spec-off） | ~46.1 t/s | ~57.2 t/s |
| std prefill（spec-off） | ~1.31–1.35k t/s | ~1.62–1.73k t/s |

**判读**：
- FP8 的 GSM8K 质量稳定，两种硬件形状完全一致（同 194/200、同错误集合）。
- 长上下文检索在 3090 的 64k 和 3080 的 240k 全部命中，FP8 基线可作后续
  KVarN 的对照。
- MMLU-Pro 的 60%/58% **不可当质量结论**：20–21/50 被 768-token 上限截断，
  属协议问题，不是量化退化。
- 速度是 **spec-off** 口径；生产 spec-on 的绝对值更高，但本次因已知 DFlash2
  tail-cap P0 未跑 spec-on（见 §2）。

## 1. 环境与协议

### 1.1 服务

| 端口 | 形状 | 档 | KV | 备注 |
|---|---|---|---|---|
| 8134 | 单 3090 Ti 24G | `profiles/qwen38_27b-3090.toml` | FP8 | **spec-off**（`--set spec.depth=0 --set spec.lookup=false`） |
| 8135 | 双 3080 20G TP2 | `profiles/qwen38_27b-3080_tp2.toml` | FP8 | **spec-off** 同上 |

> 质量测试用 spec-off 是**协议隔离**：spec 本身精确，但已知 tail-cap P0
> 会在 `max_new` 接近 depth 时设备自旋；spec-off 让 200 题长跑可重复。
> 生产 spec-on 的服务已在测试后恢复（8134/8135 均 `health ok`）。

### 1.2 数据与抽取

- GSM8K：`repos/vllm-ampere-optimized/benchmarks/gsm8k.jsonl` 前 200 题；
  gold = `####` 后的数字；pred = 最后一个 `</think>` 之后的 `\boxed{}`，
  否则最后一个数字；数值容差 `1e-4`。
- MMLU-Pro：`benchmarks/mmlu_pro.jsonl` 前 50 题；prompt 为
  “Question / Options / The answer is (X)”；pred 取最后一个答案字母。
- Needle：`All work and no play...` 填充，passcode `ZXCVBNM12345` 放在
  50% 深度；答案只要求 passcode，`max_tokens=512`，`temperature=0`，
  thinking off。
- 所有请求走 owl `/v1/chat/completions`；思考开关走
  `chat_template_kwargs.enable_thinking`。

### 1.3 速度

`packages/llm_speedtest` 的 `--flavor std`：512/1024/2048/4096/8192 × 3 轮，
输出 256 tok，单并发，固定 seed/nonce。表内取每个长度 3 轮的**中位数**。
注意 CLI 未传 `chat_template_kwargs`，因此速度测试默认 thinking on；
这与质量测试的 thinking off 不同，但速度口径与历史 std 一致。

## 2. 已知事故：spec-on 3090 在基线中途触发 P0

第一次 3090 用**生产 spec-on** 跑 GSM8K 200：

- 第 150 题左右（日志 turn 168）后，客户端仍在等；
- GPU1 100%，`server` 全线程 `futex_do_wait`，无 client 进程；
- 日志最后停在 `[chat] 受理 ... turn 168 (... max_new=768)`，之后只有
  `[sched] n=... avg=0.00ms` 与 N6 压力行，没有 turn 完成；
- SIGTERM 不退出，最终 `kill -9`；GPU 显存恢复。

这与 `docs/devlog/serving/spec-tailcap-gpu-spin-立案.md` 的已知 P0 形状一致：
长输出跑到 `max_new` 尾部、verify 块被裁到 ≤ depth 时设备自旋。基线因此改为
**spec-off** 重跑；这不是 FP8 的质量问题。

## 3. GSM8K n=200（thinking off, greedy）

| 形状 | 正确 | 准确率 | 错误 | finish=length | 平均 completion tok | wall |
|---|---:|---:|---:|---:|---:|---:|
| 3090 FP8 | 194/200 | **97.0%** | 6 | 5 | 371.4 | 25.6 min |
| 3080 TP2 FP8 | 194/200 | **97.0%** | 6 | 5 | 369.9 | 21.3 min |

错误 idx（两者相同）：`45, 62, 85, 93, 119, 153`。

- 45/62/93/119/153：`finish=length`（768 tok 上限），是**协议截断**，不是错答；
- 85：`finish=stop`，gold=44，两卡都答 48，是真实错误（同一模型/同一题）。

⇒ 以“未截断题”计，3090/3080 都是 **194/195 ≈ 99.5%**。
这个基线足以捕捉 KV 量化造成的明显质量退化；若要更严格，需把
`max_tokens` 提到 2048/4096 或改用 thinking-on 协议。

原始 JSON：
- `perf-runs/quality-fp8/3090-fp8-specoff-gsm8k-off-n200.json`
- `perf-runs/quality-fp8/3080tp2-fp8-specoff-gsm8k-off-n200.json`

## 4. MMLU-Pro n=50（低置信，仅作 smoke）

| 形状 | 正确 | 准确率 | 截断 | 平均 completion tok |
|---|---:|---:|---:|---:|
| 3090 FP8 | 30/50 | 60.0% | 21 | 653 |
| 3080 TP2 FP8 | 29/50 | 58.0% | 20 | 644 |

**不要用这两个数做质量结论**：20–21/50 在 768 tok 处被截断，模型还没给出
最终选项。它只说明 API/抽取链路可用；MLU 质量门需要 `max_tokens ≥ 2048`
或 thinking-on 重跑。

原始 JSON：
- `perf-runs/quality-fp8/3090-fp8-specoff-mmlu50.json`
- `perf-runs/quality-fp8/3080tp2-fp8-specoff-mmlu50.json`

## 5. Needle-in-a-haystack（depth 50%）

| 形状 | 目标 token | 结果 | elapsed |
|---|---:|---|---:|
| 3090 | 16,000 | RETRIEVED | 13.5 s |
| 3090 | 32,000 | RETRIEVED | 24.2 s |
| 3090 | 64,000 | RETRIEVED | 56.4 s |
| 3080 TP2 | 64,000 | RETRIEVED | 50.3 s |
| 3080 TP2 | 128,000 | RETRIEVED | 126.7 s |
| 3080 TP2 | 240,000 | RETRIEVED | 322.9 s |

240k 已覆盖 262k 档的绝大部分位置墙；FP8 在 3080 TP2 上仍能命中。

## 6. std 速度（spec-off）

### 3090 单卡

| prompt | prefill t/s | decode t/s | TTFT ms |
|---:|---:|---:|---:|
| 512 | 1262.6 | 46.20 | 407.5 |
| 1024 | 1344.8 | 46.54 | 763.4 |
| 2048 | 1349.0 | 46.57 | 1520.1 |
| 4096 | 1323.9 | 46.10 | 3095.9 |
| 8192 | 1309.7 | 45.34 | 6257.1 |

### 3080 TP2

| prompt | prefill t/s | decode t/s | TTFT ms |
|---:|---:|---:|---:|
| 512 | 1648.6 | 57.50 | 312.6 |
| 1024 | 1728.3 | 57.47 | 594.5 |
| 2048 | 1696.0 | 57.36 | 1209.6 |
| 4096 | 1666.5 | 57.57 | 2459.8 |
| 8192 | 1622.9 | 56.21 | 5049.9 |

> 这是 **spec-off** 口径；不能直接与历史 spec-on 的 3090 48.8 / 3080 139.8
> 对比。它的用途 = 与后续 KVarN 的同协议 A/B（KVarN 也需先关 spec 规避 P0）。

## 7. 缺口与下一步

1. **PPL / KLD 未跑**：owl 目前不支持 `prompt_logprobs` / `/v1/completions`，
   HyperQwen 的 `quality_battery.py` 不能直接复用。质量优先的话，建议给
   owl 加 teacher-forced prompt logprobs（或一个内部 scoring 端点），
   然后补 PPL（wikitext-2 / fineweb-2 / code）与 99.9% KLD。
2. **MMLU-Pro 重跑**：`max_tokens=2048` 或 thinking-on，n=50 即可；
   当前 768 截断率 40%，结果不可用。
3. **spec-on 速度**：等 DFlash2 tail-cap P0 修复后再跑 std；当前生产
   spec-on 已恢复，但不在本基线内。
4. **后续 KVarN/T8V4**：用同一套 `tools/quality_eval.py` +
   `tools/needle_eval.py` + std 口径；质量门先看 GSM8K 不劣化 >1–2 点、
   needle 命中不降、PPL/KLD 不超预算，再谈容量收益。

## 8. 复现命令

```bash
# 3090 spec-off 服务
./target/release/server -p profiles/qwen38_27b-3090.toml \
  --set spec.depth=0 --set spec.lookup=false --log-level info

# GSM8K 200
python3 tools/quality_eval.py --task gsm8k --n 200 \
  --data /home/div/Documents/codes/repos/vllm-ampere-optimized/benchmarks/gsm8k.jsonl \
  --base http://127.0.0.1:8134 --tag 3090-fp8-specoff-gsm8k-off \
  --out perf-runs/quality-fp8/3090-fp8-specoff-gsm8k-off-n200.json

# needle 64k
python3 tools/needle_eval.py --base http://127.0.0.1:8134 \
  --tokens 64000 --depth 0.5 --tag 3090-fp8-needle-64000 \
  --out perf-runs/quality-fp8/3090-fp8-needle-64000.json

# std（spec-off）
cd ../../llm_speedtest && uv run cli \
  --api-url http://127.0.0.1:8134/v1/chat/completions \
  --model qwen3.8-27b --flavor std --prompt-nonce fp8-3090 \
  --json .../3090-fp8-specoff-std.json
```

原始数据目录：`perf-runs/quality-fp8/`。

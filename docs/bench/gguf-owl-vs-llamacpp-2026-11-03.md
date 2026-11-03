# GGUF 正面对抗：owl vs llama.cpp（2026-11-03，单 3090 Ti）

> 用户口径：**"GGUF 这一边，跟 llama.cpp 正面对抗一次，看我们的最佳状态下的
> prefill / decode / 上下文长度能力。"**
> 本卷 = 首次同硬件、同模型、同草稿、同测量工具的受控对拍。

---

## 〇、一句话结论

**同卡同模型：owl 全面领先 —— prefill 1.5×、decode 2.2×（受控点）、TTFT 0.6×；
llama.cpp 唯一赢的是上下文上限（单卡原生 262k vs owl 172k）。**

---

## 一、口径（先看这张，否则数字不可比）

| 项 | 值 |
|---|---|
| 硬件 | **单张 RTX 3090 Ti（24G）**，UUID 钉死 `GPU-e565c505-…`（另一张空着不用） |
| 主模型 | `models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ4_XS.gguf`（14.25 GB）**两边同一个文件** |
| 草稿 | DFlash2 d7（block 8）：owl 用 `syvai/…-DFlash2-W4A16`（1.19G）；llama.cpp 用 `z-lab/…-DFlash2-gguf/Qwen3.8-27B-DFlash2-Q4_K_M.gguf`（1.14G）——**同源不同量化** |
| KV | owl = **fp8 e4m3**（32 KB/token）；llama.cpp = **q4_0**（27 KB/token，其 flavor 实测的最佳档） |
| ctx | 单点对照一律 **172032**（owl 档上限）；能力项另测更高 |
| 采样 | 同客户端参数（speedtest 默认），`--ignore-eos` 补满 256 token |
| 工具 | `packages/llm_speedtest`（**同一内核**跑两边）：`--flavor std` = 512/1024/2048/4096/8192 × 3 轮 |
| 软件 | owl `target/release/server`（sm86，main@87e5e9f5）；llama.cpp `build-master3`（VMM 权重补丁，CUDA 13.2）经 `llama-test` 起 |

**owl 起机**（档面全字段门要求完整档，`--print-config` 生成后使用）：

```bash
cd packages/owl
./target/release/server -p profiles/qwen38_27b-gguf_3090.toml --print-config \
  > profiles/measurement/qwen38_27b-gguf_3090-bench.toml
./target/release/server -p profiles/measurement/qwen38_27b-gguf_3090-bench.toml \
  --set 'parallel.device=["GPU-e565c505-4921-9979-2e0f-2b83490b7aed"]'
# 权重 14153 MiB · KV 池 5376 MiB（5376 页 × 32 tok = 172032 tok）· 草稿池 40.6 MiB
```

**llama.cpp 起机**：

```bash
cd packages/llama-test
uv run llama start -f qwen --variant master3 \
  --model models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ4_XS.gguf \
  --draft models/z-lab/Qwen3.8-27B-Dflash2-gguf/Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  --cvd GPU-e565c505-4921-9979-2e0f-2b83490b7aed \
  --devices CUDA0 --draft-device CUDA0 \
  --ctx 172032 --parallel 1 --port 8130 --ubatch 256 --kv q4_0 q4_0
# 起机 7 s
```

---

## 二、标准矩阵（std 五点三轮，同一工具、同一口径）

| 引擎 | prefill avg | decode avg | TTFT avg |
|---|---:|---:|---:|
| **owl（GGUF IQ4_XS）** | **1638.2 t/s** | **147.7 t/s** | **1846 ms** |
| llama.cpp（同模型） | 938.6 t/s | 76.9 t/s | 3461 ms |
| **比** | **1.75×** | **1.92×** | **0.53×** |

逐点（prefill / decode）：owl 1354/139（512）→ 1760/133（8192）；
llama.cpp 602/108（512）→ 1172/62（8192）。

**受控点（同 prompt：`--prompt-seed 20261019 --prompt-nonce owl-vs-llamacpp`，8k，2 轮）**：

| 引擎 | prefill | decode | TTFT | **AL** |
|---|---:|---:|---:|---:|
| **owl** | **1762.9** | **137.0** | **4644 ms** | **6.66** |
| llama.cpp | 1170.3 | 61.5 | 7373 ms | 3.36 |
| 比 | 1.51× | **2.23×** | 0.63× | **1.98×** |

AL 来源：owl = `Δspec.tokens/Δspec.rounds`；llama.cpp =
`1 + spec_decode_num_accepted_tokens_total / spec_decode_num_drafts_total`
（154 次验证吃 363 个草稿 ⇒ 3.36；草稿 tokens/验证 = 6.9 ≈ n=7 已用满）。

---

## 三、上下文长度（能力 + 长 prompt 性能）

| 点 | owl prefill / decode / TTFT | llama.cpp prefill / decode / TTFT | 比（prefill/decode） |
|---|---|---|---|
| **32k（受控 prompt，nonce=`long32k`）** | **1626.9 / 119.8 / 20.1 s** | 1099.2 / 63.3 / 30.3 s | **1.48× / 1.89×** |
| 131k（随机词池，非受控 prompt） | 1089.9 / 83.7 / 120.3 s | 772.4 / 51.4 / 171.5 s | 1.41× / 1.63× |
| **最长实测 prompt** | **163840**（1040.8 / 109.8 / 157.4 s） | **196608**（646.2 / 57.6 / 305.0 s） | — |
| **单卡可配 ctx** | **172032**（fp8 KV 32 KB/tok，池 5.25 GiB） | **262144**（q4_0 KV 27 KB/tok，起机 23.1 GiB） | llama 胜 |

- **owl 在长 ctx 上更"平"**：decode 139（512）→ 126（32k）→ 83.7（131k）；
  llama.cpp 108（512）→ 60（32k）→ 51（131k）。owl 的长 ctx 优势主要来自
  **AL 高 + FI/XQA 的 fp8 decode 路径**，不是权重带宽。
- **llama.cpp 的强项是"能装"**：q4_0 KV 只有 27 KB/token，单卡可配满原生
  262144；owl 的 GGUF 档在 24G 上封顶 172032（要更长只能换卡或减 KV 精度）。

---

## 四、为什么 owl 快（归因，不是玄学）

1. **AL 2×**（6.66 vs 3.36）——同一个 prompt、同一份草稿权重族。
   主要差异：**owl 开了 DFlash2 的 lookup-augmented drafting（LABD）**，
   llama.cpp 的 `--spec-type draft-dflash` 没有这条；其次是草稿量化
   （W4A16 syvai vs GGUF Q4_K_M）。
2. **prefill 1.5×** —— owl 走 **MMQ2 + FI 图 + chunk 2048**（1760 t/s 平台）；
   llama.cpp `--ubatch 256/--batch 2048` 在 1.1–1.2k 平台。
3. **TTFT** 是 prefill 的直接函数（0.63×），无额外项。

**⚠️ 一个结构性事实（本轮撞出来的）**：**owl 的 GGUF 主模型不能 TP2**——
`blk.0.ssm_out.weight` 的 v-head 宽 128 不是原子 256 的整数倍 ⇒ 块粒度聚合
不成立（设计 §14.6 ②，boot 响亮拒）。所以 owl 的 GGUF 路径**只能单卡**；
双卡最佳态是 AWQ-INT4 TP2（8136），那是另一条线。

---

## 五、结论与下一步

- **GGUF 线的对抗结果：owl 完胜**（prefill 1.5–1.75×、decode 1.9–2.2×、TTFT 0.5–0.6×），
  唯一劣势是单卡上下文上限（172k vs 262k）。
- **llama.cpp 的 155 t/s（台账里的生产值）是双卡数**；单卡同模型只有 ~62–77 t/s。
  若要"双卡 vs 双卡"，owl 侧必须换 AWQ TP2 档（模型格式不同，不是同题对拍）。
- 可继续量（未做）：
  1. **同草稿量化下的 AL 对拍**（把 llama.cpp 也换成 W4A16 不可行；反向把 owl 换成
     GGUF 草稿 `z-lab/…Q4_K_M.gguf` 可以）——可分离"lookup vs 量化"两个因素；
  2. **llama.cpp 关/开 lookup 的对照**（它没有该实现，只能作上界参考）；
  3. 并发 2（两边都开 np=2）的吞吐曲线。

---

## 六、**同 KV 位宽复测（q8_0）** —— 这才是同题对拍（2026-11-03 晚追加）

> 用户裁决：llama.cpp 用 q4_0 KV 是"降精度换容量"，不能拿来比速度 ⇒ 统一到 **8 bit 档**
> （owl = fp8-e4m3，llama.cpp = q8_0）。ctx 两边都是 172032、模型/草稿/工具不变。

| 口径 | **owl（fp8-e4m3）** | **llama.cpp（q8_0）** | 比 |
|---|---:|---:|---:|
| **显存 @172k** | **21605 MiB** | 23785 MiB | **owl 省 2.2 GiB** |
| std 五点三轮 prefill avg | **1638.2** | 935.2 | **1.75×** |
| std decode avg | **147.7** | 81.6 | **1.81×** |
| std TTFT avg | **1846 ms** | 3497 ms | 0.53× |
| 8k 受控 prefill | **1762.9** | 1163.6 | 1.51× |
| 8k 受控 decode | **137.0** | 66.5 | **2.06×** |
| 8k 受控 **AL** | **6.66** | 3.69 | **1.80×** |
| 8k TTFT | **4644 ms** | 7463 ms | 0.62× |
| 32k 受控 prefill | **1627.9** | 1112.0 | 1.46× |
| 32k 受控 decode | **134.7** | 52.6 | **2.56×** |
| 32k **AL** | **6.84** | 3.12 | **2.19×** |
| 32k TTFT | **20.1 s** | 29.9 s | 0.67× |

**三条新结论**：

1. **同 KV 位宽下 llama.cpp 反而更吃显存**（+2.2 GiB）—— 它上一轮"262k 很省"的
   唯一来源就是 **q4_0（0.5625 B/元素）**；换回 8 bit 后固定开销比 owl 大
   （23.8 − 172032×34.8KB ≈ 17.9 GiB vs owl 的 16.2 GiB）。
2. **KV 位宽对 llama.cpp 的速度几乎无影响**：q4_0 → q8_0 在 8k 上 decode
   61.5 → 66.5（+8%，方向上 q8_0 还更好），prefill 1170 → 1164（不变）。
   ⇒ 它用 4 bit 只买到**容量**，没买到速度。
3. **差距随 ctx 拉大**：decode 从 8k 的 2.06× 涨到 32k 的 **2.56×** ——
   因为 llama.cpp 的 AL 随 ctx 掉（3.69 → 3.12，−15%），owl 的 AL 反而**平**
   （6.66 → 6.84）。owl 的长 ctx 优势是"接受长度不掉"，不是"核更快"。

**顺带推翻它自己台账的一条**：llama.cpp flavor 注里写"q4_0 是本 draft 的原生搭档，
q8_0 会让 acc 4.17→3.49"。本轮同 prompt 受控复测**反向**（q4_0 AL 3.36 / q8_0 AL 3.69）
—— 该结论依赖的具体 prompt 未记录，不能当通用律用。

### 复现（q8_0 臂）

```bash
uv run llama start -f qwen --variant master3 \
  --model models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ4_XS.gguf \
  --draft models/z-lab/Qwen3.8-27B-Dflash2-gguf/Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  --cvd GPU-e565c505-4921-9979-2e0f-2b83490b7aed \
  --devices CUDA0 --draft-device CUDA0 \
  --ctx 172032 --parallel 1 --port 8130 --ubatch 256 --kv q8_0 q8_0
```
AL 口径：`1 + spec_decode_num_accepted_tokens_total / spec_decode_num_drafts_total`
（按请求前后取差分）。

---

## 七、GGUF TP2 放开（K/I 超块）—— 2026-11-03 深夜追加

> 前六节的 GGUF 数字全部是**单卡**（K/I 量子档的 `ssm_out` 原子 256 > v-head 128，
> 旧的"逐头取原子"几何必拒）。本轮改几何 + 补 per-rank 置换表后，**同一文件可直接双卡 TP2**。
> 设计与门：`docs/devlog/models/gguf-tp2-kquant-设计.md`、`gguf-format.md` §15.8。

### 7.1 摆盘

- 双 3090 Ti（41:00.0 / 42:00.0），TP2，`tp_vocab_split=false`，FP8 KV，FI 图；
- 模型：`unsloth/…UD-IQ4_XS`（14.25 GB，`ssm_out` 逐层 Q5_K/Q4_K/IQ4_XS/Q3_K）；
- 工具：`llm_speedtest` 同口径（nonce 固定、输出 128、并发 1、1 轮）。

### 7.2 结果（spec off）

| 点 | 单卡 3090 Ti | **TP2 双 3090 Ti** | 比 |
|---|---:|---:|---:|
| 权重驻留 | 14.9 GiB（驱动账 14981 MiB） | **7.7 GiB/rank**（8521 MiB/卡） | — |
| prefill 512 | 1768.9 | **2186.3** | **1.24×** |
| prefill 8192 | 1855.1 | **2520.6** | **1.36×** |
| decode 512 | 42.06 | **62.00** | **1.47×** |
| decode 8192 | 41.52 | **61.43** | **1.48×** |

spec-on（DFlash2 d7）单卡同口径：decode **156.45 / 119.85**（512 / 8k）；
3090 对 TP2 spec-on 未测（卡被占），**改用双 3080 20G 补测**（见 7.2b）。

### 7.2b 双 3080 20G（便宜硬件形状；含 spec-on 与长上下文）

同一文件、同工具/同 nonce，`tp_vocab_split=true`：

| 点 | 单 3080 | **2×3080 TP2** | 比 |
|---|---:|---:|---:|
| 权重驻留 | 16.6 GiB | **7.4 GiB/rank** | — |
| prefill 512 / 8k（spec off） | 1409.7 / 1493.3 | **1889.6 / 2142.6** | 1.34× / 1.43× |
| decode 512 / 8k（spec off） | 33.91 / 34.62 | **55.67 / 56.92** | **1.64× / 1.64×** |
| prefill 512 / 8k（**d7**） | 1384.4 / 1466.4 | **1846.3 / 2081.3** | 1.33× / 1.42× |
| decode 512 / 8k（**d7**） | 109.03 / 145.93 | **181.06 / 221.20** | **1.66× / 1.52×** |
| AL（d7） | 6.82 | 6.32 | −7% |

**长上下文（本改动的真正收益）**：单卡 GGUF 的 ctx 被 24G/20G 显存压到 172k（3090）
/ 更少；TP2 后 **262144（模型 native 上限）** 实测可达：

| 档 | 结果 |
|---|---|
| 131k needle @50% | **RETRIEVED**，端到端 93.7 s（≈1400 t/s prefill） |
| 260k needle @50% | **RETRIEVED**，端到端 238.4 s（≈1115 t/s prefill；262k 池 4.0 GiB/rank，总 11.7 GiB/rank） |
| IQ3_S（12.0GB）TP2 boot | ✓ 7.0 GiB/rank、cap ≈ 819k、decode 46.9/47.7（spec off） |

注：needle 答案里模型先"拒答"再复述 passcode（安全话术），判定按
`NEEDLE in text`（脚本既有口径），与其它会话一致。

**判读（2026-11-04 更正 + GDN 修复后）**：GGUF 的 TP 扩展系数 ~1.58×，与 AWQ-INT4
同二进制复测的 **~1.57×** 相当。绝对 prefill 上 **8k 档 AWQ 更快（2934 vs 2353，
w4a8 的 int8 张量核 vs GGUF 的 dequant-in-kernel）**，512 档 GGUF 略快；
decode 两者持平（spec off 56~57 t/s）。GGUF 的定位是显存/格式（14.25GB、K/I 量子、
20G 卡 262k），不是速度。

### 7.3 数值门（单卡 vs TP2）

thinking off、greedy、32 条 prompt（数学/事实/代码/中文/推理）：

| 口径 | 一致率 |
|---|---|
| 首 token（`max_tokens=1`） | **31/32** |
| 前 16 token | **29/32** |
| 前 64 token | 22/32 |
| prompt_tokens（分词） | **32/32** |

长输出的分叉是 **TP 的固有浮点语义**（两半各 f32 累加 → 各自 round 到 f16 → 相加，
与单卡整段 f32 累加不同；1 ULP 经 GDN 递推混沌放大，`tp2-log.md` §18.5 定谳），
不是分片错置。首 token/前 16 token 的高一致率 + 输出连贯 = 分片与置换表正确的判据。

### 7.4 与 llama.cpp 的关系

llama.cpp 的 GGUF 双卡数字（台账 155 t/s）依然**不可比**：它的 TP 是
`--split-mode row` 的另一种切法，且 KV 用 q4_0 换容量。若要做"同文件双卡"对拍，
llama.cpp 侧需 `--cvd` 两张 3090 + row split + 同 ctx；本轮未做（owl 侧先拿到数）。

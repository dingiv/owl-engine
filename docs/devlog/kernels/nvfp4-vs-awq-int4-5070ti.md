# NVFP4 vs AWQ-INT4：5070 Ti 实测（4B，2026-11-08）

**一句话**：在 RTX 5070 Ti（sm_120）上，NVFP4 的**prefill 优势 2.1×** 是格式给的
（原生 FP4 张量核）；**decode 的差距是核给的** —— 补了 m≤8 的 GEMV 臂后
decode 从 8.47 → **6.89 ms**（对 CUTLASS 臂 1.23×，对 AWQ 只差 1.17×）。

## 1. 口径

- 机器：RTX 5070 Ti 16G（driver 595.91.07），单会话，`kv_dtype=fp8`，`spec.depth=0`，
  `prefill_chunk=2048`，`max_seq=32768`；两臂除 `[model]` 三键外**逐字同档**
  （`.local-state/4b-nvfp4-full.toml` / `.local-state/4b-awq-full.toml`）。
- 模型：`Qwen3.8-4B-Distill`（32 层 hybrid：8 full-attn + 24 GDN），流式 738 张量。
- NVFP4 臂：`format=safetensors` + `quant=nvfp4`，**装载期 RTN**（权重 e2m1 + 每 16 元素 e4m3）。
- AWQ 臂：`format=awq` + `quant=w4a16`，CT `pack-quantized` **g32-asym + zp**，
  由 `tools/awq-forge/scripts/convert_rtn_g32.py` 产出（`--mode rtn`，**同一算法**，
  故差异只在布局/内核；`--mode awq` 可换真 AWQ 的 scale 搜索，速度面不变）。
- 仪器：`.local-state/bench.sh`（每档重启 2~3 次取中位）读 `/debug/perf/epoch/json` 的
  `decode.step` / `prefill.chunk`；长提示用 `.local-state/ab-ttft.sh` 读 `[chat] turn` 自报。

## 2. 结果

| 面 | NVFP4（+decode GEMV） | NVFP4（纯 CUTLASS） | AWQ-INT4 g32 | 读法 |
|---|---:|---:|---:|---|
| 权重 live（`[vram-phase]`） | **3166.3 MiB** | 3166.3 MiB | 3228.1 MiB | NVFP4 少 62 MiB |
| 权重比特/参数 | 4.5 | 4.5 | 4.75 | 4.5 < 4.75 |
| **decode.step**（短 ctx） | **6.89 ms（145 tok/s）** | 8.47 ms（118 tok/s） | 5.90 ms（169 tok/s） | GEMV 臂对 CUTLASS **1.23×** |
| decode（9.7k ctx） | 7.65 ms/tok | 9.19 ms/tok | 6.65 ms/tok | 差 1.15× |
| **prefill（chunk 2048）** | **98.5 ms → 20.8k tok/s** | 98.7 ms → 20.8k | 208.1 ms → 9.8k tok/s | NVFP4 **2.11×** |
| prefill（chunk 512） | 32.19 ms → 15.9k tok/s | — | 60.66 ms → 8.44k tok/s | NVFP4 1.88× |
| prefill（chunk 1024） | — | — | 112.78 ms → 9.08k tok/s | — |
| **端到端 TTFT（9733 tok 提示）** | 602 ms（16.2k tok/s） | — | 1283 ms（7.6k tok/s） | NVFP4 **2.13×** |

自检：两臂都能正确答 `17*23 → 391`（NVFP4 3.79 B/tok、AWQ 3.81 B/tok，同一档质量），
装配期 oracle 5 档几何全过（含 m=1 的 decode 形状与新 GEMV 臂）。

## 3. decode 臂（2026-11-08 新增 `cu/nvfp4/nvfp4_gemv.cu`）

按 2026-10-28 的"统一臂"裁决，decode（m=1）原本走 CUTLASS 128×128×256 主循环 ——
那是 prefill 的机器，m=1 被塞进 128 行 tile 里空转，实测只有 **242 GB/s**
（每 token 权重 2.049 GB / 8.48 ms = 峰值的 27%）。

新核是**纯 SIMT 流式反量化**（无 MMA、无 TMA），按"一 warp 吃 8 行"分工，
关键是吃透了 SFB 的仿射布局：`s0=16, s1=4, s3=1, s4=512` ⇒
**同一 group-quad 的 32 行 × 4 组 = 512 字节连续**，于是一个 32 字节扇区
正好装 `{2 行}×{4 个 b}×{4 组}`，只需让"一个 warp 负责这 8 行"，
缩放就是**零浪费**读取（数据侧按 k 切、缩放侧按 n 读，两边都满扇区）。

| 口径 | 本核 | CUTLASS 臂 | 比 |
|---|---:|---:|---:|
| 85 MB 单层（纯 DRAM 流） | **685 GB/s** | 572 GB/s | 1.20× |
| 11.8 MB 单层（L2 命中） | 1169 GB/s | 419 GB/s | 2.8× |
| 引擎 decode.step（4B 全模型） | **6.89 ms** | 8.47 ms | **1.23×** |

分流纪律：`m ≤ 8` ∧ `n%128==0` ∧ `k%64==0` ∧ SF 步长命中上式才用本核，
否则自动回落 CUTLASS 臂（`OWL_NVFP4_NO_GEMV=1` 可强制回落做 A/B）。
**数值**：与 CUTLASS 臂**逐位相同**（同一条 `quant_act` 产物、组内 half2 FMA + 组间 f32 累加）。

## 4. 取舍规则（可直接引用的判据）

按每 token 时间成本（9.7k ctx）：
NVFP4 = 0.0482 ms/prompt-tok + 7.65 ms/out-tok；AWQ = 0.1316 + 6.65。

**盈亏平衡 = 1.00 / 0.0834 ≈ 12 个提示 token 换 1 个输出 token**（短 ctx 下 18.5）。

| 负载 | 比值 | 结论 |
|---|---:|---|
| 32k 长文 + 短答（读日志/审代码/检索） | ~500 | NVFP4 快 **1.8×** |
| 8k 上下文 + 100 tok 答 | ~80 | NVFP4 快 1.4× |
| 8k 上下文 + 500 tok 答 | ~16 | 基本打平 |
| 日常对话 100 in / 300 out | 0.33 | AWQ 快 **1.15×** |

## 5. 还剩的 decode 差距（1.17×）在哪

CUPTI 归因（`--perf-level gpu`，399 tok 的 epoch）：

| 项 | ms/token | 说明 |
|---|---:|---|
| NVFP4 GEMV 全家（200 次发射） | ~3.7 | 1.78 GB fp4 + 0.22 GB sf |
| **f16 LM head**（cublas `gemvx`，1.27 GB） | **1.58** | `head_quant=off`/`embed_quant=f16` ⇒ 与格式无关 |
| graph 发射等待 | 0.77 | 200+ 节点的图 |
| attention / GDN / topk 等 | ~0.5 | — |

⇒ 下一个杠杆不是 GEMM，是 **head/embed 的量化**（省 0.6~0.8 ms）；
GEMM 本体现在 76% 峰值带宽（685/896），再挤出 15% 也只值 0.4 ms。

## 6. 顺带捞到的两条副产物

1. **NVFP4 不需要离线转换**：装载期从 BF16 现量化 ⇒ 无 5.7 GB 的 torch/llmcompressor 环境、
   无额外 4 GB 量化产物落盘（AWQ 这条链每一环都要）。
2. **g128-sym 的 AWQ（4.125 bpw，最省字节的一档）今天装不上**：`format=safetensors` +
   `quant=w4a16` 仍答 `LoadArm::F16Base`（`engine::load_arm_of` 注释里的"G128 待接线"）。
   若接线，decode 字节还能再降 ~6%（2.11 → 1.87 GB），是本清单里**最便宜的一刀**。

## 7. 复现

```bash
cd /home/ubuntu/gui_agent/owl-engine && source .local-state/build-env.sh

# AWQ 臂产物（RTN，同算法对照；真 AWQ 加 --mode awq）
cd tools/awq-forge && uv sync --extra quant --group dev
uv run python -u scripts/convert_rtn_g32.py \
  --source /home/ubuntu/gui_agent/models/empero-ai/Qwen3.8-4B-Distill \
  --output /home/ubuntu/gui_agent/models/empero-ai/Qwen3.8-4B-Distill-RTN-g32 \
  --device cpu            # 4B 需 ~9.3 GB 宿主内存；GPU 档会与另一个常驻实例抢显存

# 两臂 A/B（每档 3 次取中位）
PROFILE=.local-state/4b-awq-full.toml   bash .local-state/bench.sh 3 awq-g32
PROFILE=.local-state/4b-nvfp4-full.toml bash .local-state/bench.sh 3 nvfp4
OWL_NVFP4_NO_GEMV=1 PROFILE=.local-state/4b-nvfp4-full.toml bash .local-state/bench.sh 3 nvfp4-cutlass   # 强制回落
bash .local-state/ab-ttft.sh .local-state/4b-awq-full.toml .local-state/4b-nvfp4-full.toml

# 单核微基准（decode GEMV vs CUTLASS；大形状才是 DRAM，小形状全是 L2）
g++ -O2 -std=c++17 -I$CUDA_HOME/include .local-state/bench_gemv.cpp \
  -o /tmp/bench_gemv $(find target/release/build -name libnvfp4.a -newermt '-1 hour'|head -1) \
  -L$CUDA_HOME/lib64 -Wl,-rpath,$CUDA_HOME/lib64 -lcudart -pthread
/tmp/bench_gemv 30; OWL_NVFP4_NO_GEMV=1 /tmp/bench_gemv 30
```

**坑**：
- `--check` 对 `format=awq` 会拿 27B 的档做几何对账（`model_by_format(Awq)` 只有 `qwen3.8-27b-awq`）
  ⇒ 4B 的 AWQ 档 `--check` 必 FAIL（hidden 2560 vs 5120）。`--check` 只做体检，**boot 不受影响**；
  要消掉它得给 4B-AWQ 加一条注册表档（未做）。
- `bench.sh` 打印的 prefill tok/s 恒按 512 tok 折算 ⇒ **只有 chunk=512 那一行的 tok/s 是对的**，
  其它档要自己用 `chunk_ms / prefill_chunk` 反算（本表已按真实 chunk 反算）。

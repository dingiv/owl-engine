# GGUF × KVMem 单卡 3090 Ti decode 定谳（NSYS + metrics，2026-10-21）

> ⚠️ **2026-10-21 存证说明**:本卷引用的 `profiles/legacy/*` 已随「配置档全字段显式」裁决删除;数字与命令保持原样(当时现场),复跑请用 `profiles/` 下同名生产档。

> 工作树：`工作树 perf-review`，基线 `main@23116c7`（rebase 后）。
> 模型：`Qwen3.8-27B-Q4_K_M.gguf`（18.96 GB，Q4_K 193 / Q8_0 288 / Q6_K 17 / F32 353）。
> 卡：单张 RTX 3090 Ti（42:00.0；基线 NSYS 在 41:00.0，同型号）。
> 档：`profiles/legacy/gguf27b-262k.toml`（KVMem on，f16 池 32768 tok，host 17408 MiB）。
> 采样：greedy，`--ignore-eos`，固定 prompt seed；客户端 = `llm_speedtest`(外部存档)。

## 一、结论（先看这五条）

1. 单 token decode 已接近带宽墙：每步权重 **18.24 GB**，3090 Ti 936 GB/s 的物理地板 **19.5 ms**；
   实测 GPU busy **24.20 ms**，约 **80.5%** 峰值。
2. 最大单核项是 q4_K（9.63 GB / 10.66 ms ≈ **903 GB/s**，已贴墙）与 q8_0
   （7.16 GB / 8.37 ms ≈ **855 GB/s**）。q6_K 1.46 GB / 1.76 ms ≈ 827 GB/s。
3. **q8_0 的 MMVQ VDR 2→4 是确定收益**：单步 `decode.step` p50 **25.64→25.05 ms**
   （两轮中位，−2.3%），客户端 decode **38.7→39.6 t/s**（+2.3%）；32k 点
   p50 **25.61→25.00 ms**（−2.4%）。VDR=8 无额外收益（25.08 ms），取 4。
4. 曾经尝试的「q8_0 34B 块按 4B 对齐 + 移位拼接」**失败并已回退**：虽通过逐值门，
   但单步 p50 25.95→28.47 ms（更慢）。q8_0 的瓶颈不是 16-bit load 指令数，
   逐 lane 的移位/分支反而伤 ILP；不要再走这条路。
5. 4-bit/8-bit 权重格式本身决定了单 token 天花板。要再上一个台阶，**必须减少每步权重字节
   或增加每步 token 数**：前者只有换更低位宽（质量风险），后者就是投机解码（DFlash2）。

## 二、A/B 数据（同卡、同档、同 prompt）

### 2.1 16k prompt + 256 输出（42:00.0）

| 臂 | `decode.step` p50 | 客户端 decode | prefill |
|---|---:|---:|---:|
| baseline（VDR=2）run A | 25.679 ms | 38.61 t/s | 1570.6 t/s |
| baseline run B | 25.640 ms | 38.70 t/s | 1578.2 t/s |
| **VDR=4 run A** | **25.007 ms** | **39.62 t/s** | 1578.8 t/s |
| **VDR=4 run B** | **25.083 ms** | **39.54 t/s** | 1581.4 t/s |

⇒ 两轮中位：**25.66 → 25.05 ms（−2.4%）**；客户端 **38.66 → 39.58 t/s（+2.4%）**。
同点噪声 ≤0.3%，收益大于噪声。

### 2.2 32k prompt + 128 输出（42:00.0）

| 臂 | `decode.step` p50 | 客户端 decode | prefill |
|---|---:|---:|---:|
| baseline | 25.607 ms | 38.73 t/s | 1342.6 t/s |
| **VDR=4** | **25.003 ms** | **39.37 t/s** | 1272.6 t/s* |

\* 32k 两次 prefill 不同时段，不作为收益判据；decode 同点可比。
32k 的 KVMem 首次换页仍有约 **1.5 s** 的尖峰（`srv.seg.swap` p99），
但发生在首步/边界，`decode.step` p50 不受影响；长输出会被摊薄。

### 2.3 VDR 扫描（16k，42:00.0）

| VDR | p50 | 客户端 decode |
|---:|---:|---:|
| 2（baseline） | 25.66 ms | 38.66 t/s |
| **4** | **25.05 ms** | **39.58 t/s** |
| 8 | 25.08 ms | 39.58 t/s |

VDR=8 与 4 持平（寄存器压力抵消），取 4。

## 三、NSYS 单步核账

协议：`nsys profile --cuda-graph-trace=node --trace=cuda,nvtx,osrt`；
16k prompt + 64 输出；按 `argmax` 序列切 64 个 decode 步，取核总和/64。

| 核 | baseline µs/步 | VDR=4 µs/步 | 变化 |
|---|---:|---:|---:|
| `owl_gguf_gemm_q4k` | 10 690 | 10 658 | −0.3% |
| `owl_gguf_gemm_q8_0` | 8 976 | **8 374** | **−6.7%** |
| `owl_gguf_gemm_q6k` | 1 800 | 1 762 | −2.1% |
| `vllm_paged_attention_v2_...` | 1 371 | 1 330 | −3.0% |
| `owl_gguf_quant_act_f16` | 731 | 707 | −3.3% |
| `owl_gdn_decode_step_v2_f16` | 461 | 452 | −2.0% |
| 其余（norm/silu/gather/argmax/…） | ~900 | ~890 | — |
| **GPU busy 合计** | **24 970** | **24 203** | **−3.1%** |

q8_0 分网格（VDR=4，64 步合计）：

| gridX | 用途 | n/步 | µs/步 |
|---:|---|---:|---:|
| 12 | `ssm_alpha`/`ssm_beta`（n=48） | 94.5 | 317 |
| 256 | 全注意力 k/v（n=1024） | 31.5 | 236 |
| 1280 | `ssm_out`（n=5120,k=6144） | 47.25 | 1 991 |
| 1536 | `attn_gate`（n=6144） | 47.25 | 1 792 |
| 2560 | `attn_qkv`（n=10240） | 47.25 | 2 891 |
| 3072 | `attn_q`（n=12288） | 15.75 | 1 148 |

有效带宽（按文件精确字节）：
q4_K（扣掉 token_embd 的 0.715 GB，decode 只 gather 一行）≈ **903 GB/s**；
q8_0 **855 GB/s**；q6_K **827 GB/s**。q8_0 已从 797 提升，但距 q4_K 仍有 ~5%。

## 四、失败/否决记录（省下一次重走）

### 4.1 q8_0 按 4B 对齐 + 移位拼接（否决）

思路：q8_0 块 = `half d; int8 qs[32]`（34 B），`qs` 在 0/2 (mod 4) 之间交替；
从 `qs-2`（块基址）按 4B 对齐读，再
`(base[j] >> 16) | (base[j+1] << 16)` 拼出 4 字节，减少 16-bit load。

实测：
- 首版右移用了有符号 `int`，高 16 位符号扩展污染结果，逐值门 `max|Δ|/max=1.77e-1`；
- 改成无符号后门绿（2.86e-4），但**单步 p50 25.95→28.47 ms**（42:00.0 同档），
  逐 lane 分支/移位/额外地址计算把 ILP 吃光。
- 结论：**回退**。q8_0 的 34 B 布局不是靠逐 lane 拼字能救的；若将来要做，
  只能是「整行分块 → smem 对齐 staging」，预期收益也只剩 q8_0 的
  ~5%（≈0.2–0.4 ms），优先级低于投机解码。

### 4.2 其他已排除

- `fp8 KV`：32k decode 21.3 t/s，显著差于 f16 31.3（旧账，本轮未重复）。
- `fp8 + XQA`：27.0 t/s，仍差于 f16。
- 双 3090 layer split：6k 1377/36.57，慢于单卡（旧账）。
- VDR=8：与 4 持平，不取。

## 五、下一刀排序（按收益/成本）

1. **投机解码（DFlash2 GGUF）**：唯一能突破单 token 带宽地板的杠杆。
   现有 `feat/gguf` 分支已有 GGUF 草稿装载/验证；但 Q4_K_M 18.96 GB + 1.1 GB 草稿
   + KV/图在 24 GB 卡上余量很紧，需先做显存账/换 IQ4_XS 靶位。
2. **quant_act 归并/复用**：489 发/步、0.71 ms；同输入的多条投影
   （qkv/z/alpha/beta、ffn gate/up）目前各量化一次。做到「一输入一量化」可省
   ~0.3–0.5 ms，并少 ~200 个图节点。
3. **KVMem 首步换页尖峰**：32k+128 输出档 `srv.seg.swap` 有 1.5 s 单次尖峰；
   属于首步/边界，不影响 p50，但影响短输出体感。可做 staging 预分配/按工作集截断。
4. **q8_0 smem staging**：上限约 0.2–0.4 ms，成本高，暂缓。
5. **host 图外 ~0.8–1.0 ms**：`cuMemGetInfo`、argmax D2H、页表更新；单项都 <0.1 ms，
   优先级最低。

## 六、变更与复现

代码变更（未提交）：

```text
crates/kernels/cu/gguf/gguf.cuh
  VDR_Q8_0_Q8_1_MMVQ: 2 -> 4（含实测注释）
```

门：

```bash
OWL_TEST_DEVICE=41:00.0 cargo test -p owl-models --release \
  gpu_gguf_gemm_matches_cpu_reference -- --nocapture --test-threads=1
```

门内已补 **q8_0 m=1**（decode 形状）与 m=4（批量形状）两例；VDR=4 两例均绿。

复现命令（42:00.0，端口 8141）：

```bash
./target/release/server -p profiles/legacy/gguf27b-262k.toml \
  --set model.dir=/path/to/models/ggml-org/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf \
  --set runtime.bind=127.0.0.1:8141 \
  --set runtime.device.bdf=42:00.0 --log-level info

cd <llm_speedtest> && uv run cli \
  --api-url http://127.0.0.1:8141/v1/chat/completions \
  --model Qwen3.8-27B-Q4_K_M --lengths 16384 --rounds 1 \
  --output-length 256 --ignore-eos --temperature 0 --top-p 1 \
  --prompt-seed 20261019 --prompt-nonce <cold-nonce>
```

原始产物：`/tmp/owl-nsys-16k.*`（baseline）、`/tmp/owl-nsys-vdr4-16k.*`（VDR4）、
`/tmp/owl-a42-*.json`（A/B 客户端记录）。

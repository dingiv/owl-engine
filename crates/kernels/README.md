# owl-kernels —— 内核之家:算子矩阵与目标

> 使命:GPU 内核的**唯一注册地**。注册表 = "名字 → 可发射体"的账本;
> 发射体来源走三条通道(§一),消费者是 models 声明层(按名组合)与
> owl-cuda server(执行)。量化导向的完整推演见
> `roadmap.local/operator-matrix.md`(local);本文 = **在树正本**,
> 两矩阵 + 目标在此立住。

## 一、通道账本(发射体来源;三者并存,注册表不关心来源)

| 通道 | 形态 | 现役/规划 |
|---|---|---|
| **Source** | .cu 源 → nvrtc 首调编译(源哈希缓存)| **现役全部**(cu/ 注册表)|
| **Precompiled** | 预编 .a/动态库 + 符号 | 🎯 cuBLAS、Marlin(xinfer P1 sm86 .a 已验)|
| **External** | 库自带 plan/run 协议 + JIT 缓存(管理归 server)| 🎯 FlashInfer(后置)|

判据:算子能用**一个 .cu 文件说完** → Source;巨型模板/预编生态 →
Precompiled;自带编译器与 plan 协议 → External。

## 二、矩阵一:家族 × 量化方案(★=逐列工程 ◎=激活域跟随 + =新增家族)

| 家族 \ 方案 | FP16/BF16 | W4A16 | W4A8 | FP8 | FP4(50系) | GGUF |
|---|---|---|---|---|---|---|
| **GEMM**(187 发/步) | 🎯 cuBLAS | 🎯 Marlin | 🎯 Marlin-int8/CUTLASS | 🎯 cuBLASLt-fp8 | 🎯 CUTLASS-NVFP4 | 🎯 dequant→f16 + cuBLAS(K-quant 核后置) |
| **Attention**(decode/prefill) | ✅ decode naive / 🎯 prefill | ◎ | ◎ | ◎(+fp8 KV 可选) | ◎ | ◎ |
| **GDN**(conv/gating/l2/delta/norm_act) | ✅ decode 五核 / 🎯 prefill 三核 | ◎ | ◎ | ◎ | ◎ | ◎ |
| **Norm/Rope/Elementwise** | ✅ | ◎ | ◎ | ◎ | ◎ | ◎ |
| **激活量化 ops**(GEMM 前置) | — | —(按护栏 opt-in) | + amax/scale/quant | + fp8 cast/scale | + nvfp4 quant | — |
| **Repack**(loader 域) | — | + marlin repack | + | + fp8 块尺度 | + NVFP4 重量化 | + gguf 解析/K-quant |
| **Sampling/Embedding/narrow** | ✅ host argmax + embed + narrow | ◎ | ◎ | ◎ | ◎ | ◎ |

读数:**GEMM 行逐列工程;其余行组 = Q0 激活域地基一次覆盖全部列;
量化列新家族随列生。** decode(m=1)= 权重带宽瓶颈(量化的收益机制);
prefill = 计算瓶颈(cuBLAS/TF32 的收益机制)。

## 三、矩阵二:GGUF 三型专列(IQ4 / IQ3 / Q4_K_M)

| 方案 | 块结构 | dequant 公式 | bpw |
|---|---|---|---|
| **Q4_K_M** | 256 超块 = 8×32 子块;4bit ql/qh + 6bit scale/min | `q·scale + min`(仿射) | ~4.85 |
| **IQ4_XS** | 256 超块;4bit 索引 + 6bit scale | `kvalues_iq4nl[idx] × scale`(16 项 codebook) | ~4.25 |
| **IQ3_XXS/S** | 256 块;3bit 索引 + scale | `kvalues_iq3nl[idx] × scale` | ~3.3 |

| 家族 | IQ4_XS | IQ3_XXS/S | Q4_K_M |
|---|---|---|---|
| GEMM decode(m=1) | 🎯 MMQ(移植) | 🎯 MMQ | 🎯 MMQ |
| GEMM prefill(大批) | 🎯 dequant 核 → cuBLAS | 同 | 同 |
| dequant 核(独立小核) | 🎯 查表+scale | 🎯 查表+scale | 🎯 仿射 |
| repack(loader) | **零重排**(GGUF 块布局 = 内核读序)| 同 | 同 |
| 其余算子 | ◎ f16 域(Q0) | 同 | 同 |

**格式即生态律**:vendor GEMM 只吃标准布局(dense + 均匀分组 scale);
K-quant 超块 / IQ codebook 越界 → GGUF 三型的 GEMM 只有 llama.cpp 家
MMQ 可用(MMQ 移植领地 = decode 小批;大批 llama.cpp 自己都 dequant 后
回落 cuBLAS —— mmq.cu `should_use_mmq` 证据)。

## 四、现役清单(cu/ 注册表;Source 通道)

| 文件 | 内核 | 状态 |
|---|---|---|
| `cu/ops.cu` | `owl_matmul_f32` / `owl_matmul_nt_f32`(**裸三重循环,🎯 cuBLAS 替换为目标**)| ✅ |
| | `owl_add/mul/silu/sigmoid_f32` | ✅ |
| | `owl_rmsnorm_f32`(w_off) | ✅ |
| `cu/text/embed_f32.cu` | `owl_embed_f32` | ✅ |
| `cu/text/rope_f32.cu` | `owl_rope_half_partial_f32` | ✅ |
| `cu/text/attention.cu` | `owl_naive_decode_attn_f32` | ✅ |
| `cu/text/gdn.cu` | GDN decode 五核(gating_g/l2norm/conv_upd/delta_dec/norm_act) | ✅ |

## 五、目标(按序立住)

| # | 目标 | 验收 | 依赖 |
|---|---|---|---|
| 1 | **Q0 激活域地基**:Dtype::F16 进契约;琐核 dtype 参数化(nvrtc `-DTYPE` 双编);cast 核 | f16 域跑通 0.8B 全链,对拍 f16 容差 | — |
| 2 | **matmul → cuBLAS**(Precompiled 首例) | 同树对拍 1e-4 + 单核 bench | — |
| 3 | **PF1b prefill 内核**:`owl_prefill_attn`(T q + causal)+ GDN 三核移植(`gdn_kernels.cu`:conv1d_fwd / prefill gqa / varlen) | T=64 chunk == 逐步对拍;耗时 << T×decode | Q0(f16 形态)|
| 4 | **Q1 Marlin W4A16**(sm86 .a + repack + QuantizedLinear)| W4A16 checkpoint 端到端 + 对拍 + 步时 | Q0 |
| 5 | **Q4 GGUF**(Q4_K_M → IQ4_XS → IQ3) | GGUF 端到端 == llama.cpp 数值锚 | Q0 |
| 6 | Q2 FP8/W4A8;Q3 NVFP4(50 系硬件位) | 同 Q1 形态 | Q0;硬件 |

## 六、纪律

- **对拍**:每核引入/移植即写数值对拍(建核与对拍同批,不欠账);
- **C1**:维度推导只读声明 shape;块账长仅容量;
- **流序律**:同块顺序写依赖靠流序表达(仅限 decode 原地五核;
  prefill 走显式 state 进出的 functional 内核);
- **命名**:`owl_<域>_<语义>[_形态]_<dtype>`;
- **捕获**:一切新发射路径必须可捕获(核 launch on stream;禁捕获期
  Htod/Dtoh —— iface 契约);
- 深推演文档(local,不入 git):`roadmap.local/{operator-matrix,
  phase3-models-foundation, pf1-prefill-research, quant-integration-notes}.md`。

## 变更记录

| 日期 | 变更 |
|---|---|
| 2026-09-26 | 立宪:两矩阵(家族×方案 / GGUF 三型)+ 三通道账本 + 目标六项立住 + 纪律汇编 |

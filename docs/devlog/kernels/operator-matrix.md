# 算子专项:以量化为导向的算子家族矩阵

> 2026-09-26 v2(用户定调:**以量化为导向**决定算子家族,继而影响上层
> 计算层;v1 的全景清单降为 §一基线附录)。
> 支持序列(用户拍板):**W4A16 首发 → W4A8 / FP8 → FP4(NVFP4,50 系);
> FP16/BF16 = 无量化基线;GGUF 生态后置**。

## 〇、核心结构判断:所有方案共享一块地基 = 激活域

| 方案 | GEMM 内核 | 激活域 | 配套算子(热路径新增) |
|---|---|---|---|
| FP16/BF16(基线)| cuBLAS HSHMM | **f16/bf16** | cast(入口)|
| **W4A16** | **Marlin**(sm86/89 .a;sm120 新产物)| f16 | repack(loader);cast 已含于 f16 域 |
| W4A8 | Marlin-W4A8 / CUTLASS int8 | **int8 激活 + scale** | **动态激活量化核**(amax→scale→量化,GEMM 前置,×187)|
| FP8(W8A8) | cuBLASLt fp8 / CUTLASS(40 系原生 TC)| e4m3 | 激活 cast/scale 核 |
| FP4(NVFP4 W4A4) | CUTLASS / FlashInfer(sm120 原生 TC)| fp4 + e4m3 block scale | NVFP4 重量化(loader)+ 激活量化核 |
| GGUF | 逐 token 路径同 f16 域 | f16 | loader:GGUF 解析 + K-quant 布局(两档:载入期 dequant→f16 先行 / dequant-in-kernel 后置)|

## 〇.5 完整矩阵(家族 × 方案;★=逐列工程,◎=激活域跟随,+ =新增家族)

| 家族 \ 方案 | FP16/BF16 | W4A16 | W4A8 | FP8 | FP4(50系) | GGUF |
|---|---|---|---|---|---|---|
| **GEMM**(187 发/步) | cuBLAS | ★Marlin | ★Marlin-int8/CUTLASS | ★cuBLASLt-fp8 | ★CUTLASS-NVFP4 | dequant→f16 + cuBLAS(K-quant 核后置) |
| **Attention**(decode/prefill) | ◎f16 | ◎ | ◎ | ◎(+fp8 KV 可选) | ◎ | ◎ |
| **GDN**(conv/gating/l2/delta/norm_act) | ◎f16 | ◎ | ◎ | ◎ | ◎ | ◎ |
| **Norm/Rope/Elementwise** | ◎f16 | ◎ | ◎ | ◎ | ◎ | ◎ |
| **激活量化 ops**(GEMM 前置) | — | — | + amax/scale/quant | + fp8 cast/scale | + nvfp4 quant | — |
| **Repack**(loader 域) | — | + marlin repack | + | + fp8 块尺度 | + NVFP4 重量化 | + gguf 解析/K-quant |
| **Sampling/Embedding/narrow** | ◎ | ◎ | ◎ | ◎ | ◎ | ◎ |

三个结构读数(覆盖策略的依据):
1. **只有 GEMM 行逐列工程**:每方案一个专属内核(marlin/cublasLt/cutlass)
   —— 逐列做的真功夫;
2. **Attention/GDN/Norm/Rope/Elementwise 全行 = 激活域跟随者**:内核源
   不变,dtype 参数化重编即可 —— **Q0 一次投入,整行组 × 全部列同时覆盖**;
3. **量化专属新家族**(激活量化 ops + repack)只在量化列存在,随列立项
   随列生。

→ 矩阵虽大,工程自由度集中在:**GEMM 行(逐列)+ Q0 行组(一次)+
量化列新家族(随列)**。

**推论 1(Q0 地基)**:W4A16/FP16/FP8/GGUF 全部要求激活域离开 f32 ——
**Q0 = 激活域地基**(全模型 dtype 参数化:琐核/rope/attn/GDN 的 f16 形态
+ cast 核 + Dtype::F16 进契约)。
**推论 2(JIT 红利)**:nvrtc 源注册通道天然支持同一 .cu 按 dtype 宏
多次编译 —— dtype 参数化近乎免费,这正是我们选 JIT 通道的自洽回报。
**推论 3(W4A8 降档的数据依据)**:xfinfer P4 实测 decode(m=1)W4A8 无
杠杆(权重带宽瓶颈,W4A8≈W4A16 而精度更差);其价值在 prefill 的 int8
TC(sm89)—— 后置有据。

## 一、基线算子全景(无量化;v1 附录,数字自 Qwen3.5-0.8B)

每 decode 步:**GEMM 187 发**(GDN 层 5×18 + full 层 4×6 + MLP 3×24 +
lm_head)为步时主项;琐核 ≈300 发(embed/norm×49/rope×6/gating/l2norm/
conv/delta/norm_act/elementwise)。decode(m=1)= 权重带宽瓶颈
(f32 = 3.2GB/步);prefill = 计算瓶颈。

## 二、量化导向的算子家族矩阵(逐方案)

### Q0 激活域地基(所有方案公共底座)

- Dtype::F16 进 iface 契约词汇;
- 琐核 dtype 参数化:nvrtc 同源双编译(`-DTYPE=__half`),f32 形态保留;
- cast 核(f32→f16/f16→f32);
- 层签名不变形状变化 —— 数据流 dtype 由 QuantSpec/激活域决定。
- **验收**:f16 域跑通 0.8B 全链(GPU),输出与 f32 域对拍(容差按 f16
  精度,实测定)。

### Q1 W4A16(首发)

- GEMM:marlin sm86 .a(xinfer P1 资产:对拍 30/30,零 nvcc)+ repack
  (P2:parity 对 vLLM 黄金字节)+ loader transform(Q0 的 f16 激活域上
  marlin 直读);
- 层:QuantizedLinear(Want 附 repack transform);其余层零改动(已在
  f16 域);
- 选型:`(W4A16, sm86|sm89) → MarlinSm86`;降级链 → cuBLAS f16 →
  owl_matmul;
- **验收**:0.8B W4A16 checkpoint 端到端(GPU),与 f16 基线对拍(容差
  实测定);decode 步时对比 f16 基线(带宽减半的收益可见)。

### Q2 FP8 / W4A8(40 系第二梯队)

- FP8:cuBLASLt fp8 GEMM + 激活 scale 核(sm89 原生);精度优于 W4;
- W4A8:Marlin-W4A8 / CUTLASS int8(P4 已有契约闭环资产);decode 无
  杠杆(推论 3),价值 = prefill int8 TC —— 排 FP8 后;
- **验收**:同 Q1 形态(端到端 + 对拍 + 步时对比)。

### Q3 NVFP4(50 系位)

- CUTLASS/FlashInfer NVFP4 GEMM + 重量化 loader(INT4/f16 checkpoint →
  NVFP4)+ 激活量化核;**依赖 sm120 硬件到位**。

### Q4 GGUF 生态(loader 先行)

- GGUF 解析 + K-quant 布局 → **载入期 dequant→f16**(Q1 域直跑,正确性
  先行);dequant-in-kernel(4bit 带宽收益)后置;
- **验收**:主流 Q4_K_M 端到端 = f16 基线对拍。

### Q4+ GGUF 格式深勘(2026-09-26,本地 llama.cpp 树核实)

**独到之处四条**:① 单文件自包含(权重 + 元数据 KV + **内嵌词表** +
chat template,分发 = 一文件);② mmap 优先(压缩驻留按页懒加载);
③ 逐 tensor 量化 + **block 量化动物园**(ggml.h:Q8_0/Q4_K/IQ4_NL/
IQ4_XS/MXFP4…;K-quant = 256 超块内 32 子块,4bit 量化 + 6bit scale/min;
IQ = 非线性 codebook 查表 —— **非 GPTQ/AWQ 的逐组 scale+zero 结构**);
④ dequant-on-the-fly 哲学(权重永驻压缩态,MMQ = dequant-in-kernel GEMM,
mmq.cu + 22 模板实例 + per-arch mmq-config)。

**对 owl 的四层影响**:
- Loader(必做):GGUF 解析(header KV + tensor 表 + mmap)+ 载入期
  dequant→f16 → **GGUF 列在算子矩阵塔缩进 FP16 列**(算子零新增);
- QuantSpec 域扩展:Scheme::Gguf{ty}(K-quant/IQ 结构表达不了进
  W4A16{group,sym});分域律不变 —— type 也是方案数据,层不感知;
- 内核域(可选二期):MMQ dequant-in-kernel(拿 4bit 带宽收益)——
  参照物 = 本地 mmq.cu;接入 = matmul 混合输入变体(a f16 × b quant
  block),BackendPolicy 加 GGUF 列;
- Tokenizer 域(避坑):**阶段一要求外置 tokenizer.json**(Qwen 系 HF
  有),GGUF 内嵌词表读取挂账(llama-vocab 复刻,工程不小)。

**对拍基建已有**:llama.cpp 源码树 + llama-test 工具链在工作区,GGUF
数值锚可直接用 llama.cpp 生成;docs/llama.cpp 坏文件结案实录 = 先例坑位
(type/对齐/版本)已知,本次是读不是写。

### Q4++ 格式即生态律(2026-09-26,mmq.cu 证据)

**律:权重格式定义可服务内核的集合。** vendor GEMM(cuBLAS/CUTLASS/
Marlin/FlashInfer)全部假设标准布局(dense + 均匀分组 scale ± zeros/perm);
K-quant 超块混合位宽 scale、IQ codebook 越界 → 只剩 llama.cpp 自家 MMQ。
因果摆正:非“为绑定实现而极端”,而是低比特质量目标逼出非标准结构,
非标准结构天然只有自家能跑(ggml 自带全套 GEMM,无所谓)。

**决定性证据**(mmq.cu `should_use_mmq`):硬件有 fp16 MMA 时,
`return ne11 < MMQ_DP4A_MAX_BATCH_SIZE` —— **llama.cpp 自己在大批量
(prefill)时放弃自家 MMQ,dequant 后回落 cuBLAS**(line 2006
`ggml_cuda_mul_mat_cublas`)。这正是我们 Q4-a(dequant→f16→标准内核)
路线的同款选择,且证明该路线在大形状下更优是 llama.cpp 官方结论。

**GGUF 列内部分层(修订)**:
| 类型群 | 结构 | 可服务内核 |
|---|---|---|
| Q8_0/Q4_0/Q4_1/Q5_x | 均匀块 scale(32 块一 scale)| 通用块尺度核即可(结构上近 NVFP4)|
| Q2_K–Q6_K | 超块混合位宽 scale | llama.cpp MMQ(移植)或 dequant |
| IQ1_S–IQ4_NL/XS | 非线性 codebook | llama.cpp MMQ(最低优先:最异构 + codebook 管理)|

→ Q4-a(dequant→f16)再获背书;Q4-b(MMQ 移植)领地仅限 decode 小批量。

## 三、感知边界(重申并细化)

- **层感知**:QuantSpec(方案/layout/GEMM 语义)→ 决定 Want 的 repack
  transform 与激活域;
- **层不感知**:vendor(marlin/cutlass/flashinfer)、内核名、通道(Source/
  Precompiled/External)、arch;
- **激活域归 QuantSpec 派生**:scheme → dtype 形态 → 琐核编译变体 ——
  上层(层/模型)只见"我的数据是 f16",不知道为什么。

## 四、BackendPolicy(机制不变,矩阵加行)

```rust,ignore
fn resolve_gemm(&self, q: &QuantSpec, caps: &DeviceCaps) -> KernelName {
    match (q.scheme, caps.arch) {
        (None, _)            => CublasF16,                    // Q-基线
        (W4A16, Sm86 | Sm89) => MarlinW4A16Sm86,              // Q1
        (W4A16, Sm120)       => MarlinW4A16Sm120,
        (W8A8Fp8, Sm89 | Sm120) => CublasLtFp8,               // Q2
        (W4A8, Sm89)         => MarlinW4A8,                    // Q2 后
        (Nvfp4, Sm120)       => CutlassNvfp4,                  // Q3
        _                    => OwlMatmulF32,                  // 回退永可用
    }
}
```

解析时机 = Session::plan 期(捕获前,图内形态固定);降级链逐级回退;
用户 override 钉死(vLLM `backend=auto|...` 词汇同款)。

## 五、与现有阶段的衔接(修订)

| 阶段 | 消费 |
|---|---|
| phase3 PF1b | prefill 内核(Q0 f16 域形态)|
| phase3 PF2 | config 解析:quantization_config → QuantSpec(本专项的输入口)|
| phase3 PF4 | 采样原语 |
| **量化专项(本文件,Q0→Q1→Q2→Q3→Q4)** | Q0 可与 phase3 并行(dtype 参数化是纯 models 侧);Q1 依赖 xinfer marlin 资产接线;Q3 依赖 50 系硬件 |
| 阶段四 | AgentSession 消费 Q1(量化版 0.8B = 带宽减半,agent 长会话直接受益)|

## 六、变更记录

| 日期 | 变更 |
|---|---|
| 2026-09-26 | v2:以量化为导向重组(用户定调);支持序列 W4A16→W4A8/FP8→FP4+GGUF;Q0 激活域地基判定;W4A8 降档(xinfer P4 数据);Q0–Q4 阶梯 |
| 2026-09-26 | v1:全景清单 + 社区实现映射 + BackendPolicy 机制 |

## 七、GGUF 三型专列矩阵(IQ4 / IQ3 / Q4_K_M;2026-09-26)

先解困惑:**cuBLAS 不消费量化格式**。分工是两个内核:
- **MMQ**(dequant-in-kernel):解量化融合进 GEMM,压缩块不落显密 —— 格式
  专属,只有 llama.cpp 自家(小批 decode 主场);
- **dequant 核 + cuBLAS**:先把量化块解成 dense f16(小核,读块→查表/仿射),
  再调 cuBLAS —— cuBLAS 只见 dense(大批 prefill 主场;llama.cpp
  `should_use_mmq` 的同款分界)。

三型结构(dequant 都是纯函数:查表或仿射,完美可核化):

| 方案 | 块结构 | dequant 公式 | bpw |
|---|---|---|---|
| **Q4_K_M** | 256 超块 = 8×32 子块;4bit ql/qh + 6bit scale/min | `q·scale + min`(仿射)| ~4.85 |
| **IQ4_XS** | 256 超块;4bit 索引 + 6bit scale | `kvalues_iq4nl[idx] × scale`(**codebook 查表**,16 项表)| ~4.25 |
| **IQ3_XXS/S** | 256 块;3bit 索引 + scale(+IQ3_S 带重要性矩阵位)| `kvalues_iq3nl[idx] × scale`(查表)| ~3.3–3.7 |

家族 × 三方案矩阵:

| 家族 | IQ4_XS | IQ3_XXS/S | Q4_K_M |
|---|---|---|---|
| GEMM decode(m=1 小批)| MMQ(移植,自家) | MMQ | MMQ |
| GEMM prefill(大批)| dequant 核 → cuBLAS | 同 | 同 |
| dequant 核(独立小核)| 查表 + scale | 查表 + scale | 仿射 |
| repack(loader) | **零重排**(GGUF 块布局 = 内核读序,独门优势)| 同 | 同 |
| 激活域其余算子(norm/rope/attn/GDN/EW)| f16 域不变(Q0) | 同 | 同 |
| Sampling/词表 | tokenizer 外置(阶段一裁决)| 同 | 同 |

**QuantSpec**:`Scheme::Gguf { ty: Q4KM | Iq4Xs | Iq3Xxs }`(BackendPolicy
GGUF 列:decode→MMQ 系,pre-fill→dequant+cuBLAS,按 m 分派 —— llama.cpp
同款判据 `ne11 < MMQ_DP4A_MAX_BATCH`)。

**实施序**:Q4_K_M(生态最主流)→ IQ4_XS → IQ3(最末:最小众 + codebook
管理)。dequant 核语义可直接抄 llama.cpp CPU 参考实现(C 函数逐字对拍),
CUDA 版参照其 dequant/MMQ 源;数值锚 = llama.cpp 现场生成。

### Q3+ Blackwell FP4 精确定谳(2026-09-26,llama.cpp mmq-vec-dot.cuh 证据)

**问题**:GGUF 在 50 系会不会"反量化到 FP4,再用 FP4 算子核"?
**答:分三类,且主流路径不是"转换"而是"存储即 FP4 直通"**:

1. **MXFP4/NVFP4 存储的模型**(gpt-oss 等):零转换。数据本来就是
   e2m1(FP4)+ 块 scale,Blackwell(sm120)上 llama.cpp 走
   `blackwell_mma_available` → **block-scale FP4 MMA 指令**
   (`ggml_cuda_mmq_vec_dot_fp4_fp4_mma`,m16n8k64,每调用一枚 uint32
   scale 寄存器;MXFP4=scale_vec::2X ue8m0,NVFP4=4X ue4m3)。
   —— 不是"反量化到 FP4",是**存储即 FP4,字节直进 tensor core**。
2. **Q4_K_M / IQ4_XS / IQ3**:**不转 FP4**。(a) 它们是 int4 仿射/
   codebook,转 NVFP4 = 重量化(解浮点再重编 e2m1),有损二次量化;
   (b) decode m=1 是权重带宽瓶颈,Q4_K_M(~4.85bpw)与 NVFP4 带宽近似,
   FP4 TC 零收益;(c) 50 系上照旧走常规 MMQ(blackwell tile 配置)+
   大批 dequant→f16→cuBLAS。
3. **FP16/BF16 存储**:不转,fp16 MMA/cuBLAS。

**术语修正**:"反量化(dequant)到 FP4"方向反了 —— dequant 是
quant→浮点;转 FP4 是 **重量化(requant)**,两次量化有损,llama.cpp
不做推理期自动重量化(那属于模型工件离线转换,vLLM/TensorRT 生态有人
做,显式用户决策)。

**对 owl FP4 列(矩阵 Q3)的定义修订**:FP4 列 = **模型工件以
MXFP4/NVFP4 存储**的路线;Blackwell 内核参照 = llama.cpp
`fp4_fp4_mma`(现成 .cu 可移植,Q1-Q3 期接入);Q4_K_M/IQ 存储在
50 系**不转 FP4**,照旧 MMQ —— 与 llama.cpp 同策略。

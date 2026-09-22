# xInfer 基座改造 —— 需求规格说明书

> 版本:v1.0(2026-09-20 需求阐述轮定稿)。上游设计:同目录
> `marlin基座-施工设计文档.md`。指标口径与瓶颈理论引用
> `docs/README.md`(三大基础指标 = Prefill/Decode/上下文;prefill=算力+通信、
> decode=带宽、显存三抢地盘;**并发是约束不是目标**)。
> 状态:待评审 → 评审通过后冻结为 v1.0,变更走 §九 裁决记录。

---

## 一、北极星与定量目标

所有需求服务于一个论断:**在 3080/3090 消费卡对上,Nex/Qwen3.8 家族的
三项指标全面超越现有四引擎**(基线 = vLLM W4A16+P2P:prefill 7656 /
decode 157.1 / ctx 262k@1020k 池)。

| 指标 | 及格线 | 达标线 | 冲刺线 | 口径 |
|---|---|---|---|---|
| Prefill(tok/s) | 2178(llama 线) | **7656(vLLM 线)** | 9k+(W4A8 杠杆) | std 五点三轮,随机词池 |
| Decode(tok/s) | 133(llama 线) | **157(vLLM 线)** | 165+(MTP/DFlash 增益) | 同上,usage 口径 |
| 上下文(tokens) | 131k | **262k** | 1M(W4 + 三级缓存) | 单请求 max_model_len |

TTFT 附带口径:512-token prompt ≤150ms(冷 prefix)。
测量纪律:usage.completion_tokens 计数、随机词池整段 prompt、±20% 方差
警觉(docs/README.md §五)。

---

## 二、范围宣言

**做**:RTX 30 系(Ampere sm_86)为一等公民,RTX 40 系(Ada)顺带兼顾;
Qwen3.5 dense 与 Qwen3.5 MoE(GDN+MoE 混合,Nex 同族)两类架构;
≤8 并发(默认 4);W4A16/W4A8/AWQ-INT4 量化;3090_dual(8133)/
3080_dual(8134) 硬件位。

**不做**(舍弃即设计):
- 高并发服务场景(>8 并发、continuous batching 深水区、排队调度调优)
  —— 本地单人部署,凡为高并发设计的手段(大 batch、请求级抢占)一律不引入;
- 模型广度(每架构都要支持)—— 仅 SOTA 自用架构,但**留扩展接口**
  (模型注册表 trait,见 REQ-CODE-01);
- 非 CUDA 平台(Metal/CPU 路径不维护,xinfer 原有 cfg 分支保留但不测试);
- Hopper/Blackwell 专属算子(Machete/NVFP4)。

---

## 三、Prefill 优化需求

计算受限:上限 ≈ 硬件算力,调优 = 多卡算力合并 + 通信不打折 + 消除额外开销。

| 编号 | 需求 | 优先级 | 验收口径 |
|---|---|---|---|
| REQ-PRE-01 | **社区最先进 GEMM 算子**:W4A16/W4A8 GEMM 走 Marlin 家族(vLLM csrc 移植),禁止自研 kernel 进热路径 | **P0** | P1 对拍 allclose;W4A16 prefill ≥7656 |
| REQ-PRE-02 | **张量并行,IPC 通道**:多进程模型(rank/进程),卡间通信第一轮走 CUDA IPC(cudaMalloc+handle;P2P 生产试运行姿态,SHM 为回退保险丝);卡表用 UUID 钉,禁数字序 | **P0** | TP2 prefill ≥ 单卡 1.6×;零 Xid 长跑 |
| REQ-PRE-03 | **三级 KV 缓存**:GPU 池 → host RAM 卸载(pinned)→ 磁盘卸载;prefix 命中时 TTFT 近零;容量可配(gb 级) | P1 | 262k×多轮重灌场景 TTFT 改善可测;RAM 预算可设 |
| REQ-PRE-04 | **FP16 KV cache**:默认 KV dtype=fp16(与 mistral/llama/vLLM 对照口径一致) | **P0** | 与 vLLM bf16 输出语义对齐(数值对拍含 KV 路径) |
| REQ-PRE-05 | **CUDA graph**:decode 全图捕获 + prefill piecewise(分段捕获解动态 shape);禁 graph 的逃生开关保留 | **P0** | 开 graph 后 decode 提升 ≥10%(对照 eager) |
| REQ-PRE-06 | **Chunked prefill**:长 prompt 分块预填,块长吸附 mamba/attention 网格(Nex=2096,经验值 4192=2×2096) | **P0** | 8k prompt prefill 无碎片化回退;TTFT 曲线单调 |

## 四、Decode 优化需求

带宽受限:上限 ≈ 显存带宽 / 每 token 读取字节。杠杆 = 权重字节最小化
(量化)+ 每步额外开销最小化(graph/投机)。

| 编号 | 需求 | 优先级 | 验收口径 |
|---|---|---|---|
| REQ-DEC-01 | **Marlin 算子**(W4A16;REQ-PRE-01 同源,GEMM 与 prefill 共用) | **P0** | decode ≥157 |
| REQ-DEC-02 | **投机解码**:MTP(模型含权重时)+ **DFlash2**(外部草稿,工作区 models/ 现成);后期 DSpark、DDTree(待调研);spec-aware graph(verify 形状) | P1 | DFlash2 接受率日志可视;净增益 >0 的草稿配置可开 |
| REQ-DEC-03 | **MoE 模型**:fused MoE 走 marlin_moe_wna16(vLLM 现成);router/shared_expert/lm_head 不量化 | **P0** | Nex 34.7B-A3B 3080 对 std 全跑通 |
| REQ-DEC-04 | **循环 metrics 内嵌框架**:每 scheduler 循环计时(prefill chunk / decode step / 通信 / 采样分段耗时)、投机解码逐步接受率与接受长度分布、KV 命中率;结构化日志 + 可关 | P1 | 指标能解释 ±10% 性能波动;默认关,开启开销 <2% |
| REQ-DEC-05 | **GDN 线性注意力 CUDA 算子**:port lmdeploy turbomind delta_rule legacy 路径(arch<900,282+478 行;kRecurrent/kChunked 双模式;Conv1d+SiLU/L2Norm/beta-gate 全套);FLA Triton 作正确性基准,xinfer candle 版作回退 | P1 | 与 FLA 基准 allclose;decode GDN 层耗时相对 candle 版下降可测 |

## 五、上下文调优需求

上下文 = KV 池容量的函数:池显存 = 显存总盘 - 权重 - 激活。INT4 权重是
池的最大杠杆(W4 权重仅为 BF16 的 1/4)。

| 编号 | 需求 | 优先级 | 验收口径 |
|---|---|---|---|
| REQ-CTX-01 | **量化格式**:W4A16(GPTQ/AWQ, group128, sym)+ **W4A8**(W4A16 checkpoint 原地跑 int8 激活,零转换)+ AWQ INT4(AWQ 格式 repack 进 Marlin) | **P0**(W4A16/W4A8)/ P1(AWQ) | Arahide INT4 直载;W4A8 一个 flag 切换;数值对拍 |
| REQ-CTX-02 | **流水线并行/层切分**(PP):层间切分,无 allreduce;与 TP 组成可选拓扑(TP2 / PP2 / TP1) | P1 | PP2 vs TP2 对拍数据落 bench(验证 avesed"allreduce 吃一半 prefill"论断) |
| REQ-CTX-03 | **特种 KV 量化**:int8(速度近无损)/ fp8(池×2,FlashInfer 路径有速度税)/ q4(turbo 系,精度损伤待评)——xinfer `--kvcache-dtype` 现成机制沿用 | P2 | 每档 KV 的 速度/精度 二维数据表 |
| REQ-CTX-04 | **草稿模型量化**(DFlash2 草稿的 ISQ/GGUF 量化,降草稿驻留) | P2 | 草稿显存占用记录;接受率不降 |
| REQ-CTX-05 | **EXL 类特种格式**(trellis/QTensor,EXL3 路线)——后期深入设计 | P2(预留) | 仅需求占位,不排期 |
| REQ-CTX-06 | **YaRN 上下文突破**:rope_scaling yarn 外推,262k → 1M 级(工作区已有官方配方调研 docs/qwen3.8-上下文扩展到1M-调研);用户明确可延后 | P2(延后) | 1M 外推可加载;ppl/任务抽检精度记录;长文吞吐数据 |

## 其他需求
原地在线量化 mistral.rs 的 --isq

格式转化器, 如果要求使用特定模型格式, 需要提供一个格式转化器, 用于将官方的 fp16 模型转化为特定模型;

双生态支持, gguf k-quant/i-quant, 还有 marlin 的 w4a16 w4a8 模型

## 六、非功能需求

### 6.1 硬件 REQ-HW-01(P0)
RTX 30 系(Ampere sm_86:3080/3090)一等公民;RTX 40 系(Ada sm_89)编入
目标 arch 顺带兼容;**算子层留 arch 分发表**(`kernel_dispatch(arch) -> Impl`),
新增架构 = 注册新实现,不改调用方。禁止在业务代码里写死 arch。

### 6.2 代码规范 REQ-CODE-01/02(P0,评审门禁)
- **trait 优先 = 静态分发**:pub 函数的参数与返回值用泛型约束/
  `impl Trait` 静态分发(T: QuantGemm 形态),编译期单态化零开销;
  **`dyn Trait` 动态分发仅限配置期/注册表期**(模型注册表、算子
  arch 分发表这类运行前已定的多态),热路径(内核循环/每 token 路径)
  零 dyn 零虚表
- **tokio 深度使用**:API server/调度编排/请求生命周期全异步;
  **不写阻塞函数**(std::thread::sleep/阻塞 IO 禁入 async 上下文)
- **同步点隔离**(矛盾裁决 C3 细化):CUDA kernel launch 本身异步下发,
  真阻塞点只有 D2H 同步(logits 回传)——隔离在 runner 专用线程,
  不污染 tokio runtime

### 6.3 设计指导 REQ-DESIGN(P0,工作方法论)
1. **优先 copy 社区最佳实践**:先搜先抄后改;vLLM/sglang 的 kernel 与
   调度语义、avesed 的 W4A8 路径、avesed flashampere 的 Ampere attention
   分腿,都是直接抄的对象
2. **Port 优先于重写**:工作区引擎矩阵(mistral.rs/llama.cpp/vLLM fork/
   exllamav3)的已验证逻辑,写 Rust 移植;移植时保留出处注释
   (`// ported from vllm csrc/... @rev`)
3. 自研是最后手段:仅当社区无可抄(如 GDN Rust 路径)且 profile 证明
   是瓶颈时立项

### 6.4 精度评估
需要采用科学手段测评模型的精度表现, 是否因为量化与满血模型存在较大的损伤;

## 七、需求 → 施工阶段映射

| 施工阶段(设计文档 §四) | 覆盖需求 |
|---|---|
| P0 fork 骨架 | REQ-CODE/DESIGN 落地;砍裁清单执行 |
| P1 Marlin FFI POC | REQ-PRE-01 / REQ-DEC-01(对拍地基) |
| P2 W4A16 全链 | REQ-PRE-04/05/06、REQ-CTX-01(W4A16)、REQ-DEC-03(单层) |
| P3 MoE | REQ-DEC-03 完整、REQ-DEC-01 全量 |
| P4 W4A8 | REQ-CTX-01(W4A8) |
| P5 投机解码 | REQ-DEC-02、REQ-CTX-04 |
| P6 验收 | 全表 + REQ-PRE-03(三级 KV)、REQ-CTX-02(PP 对拍)、03/05 不排期项确认 |

注:REQ-PRE-02(TP/IPC)贯穿 P1-P6(xinfer runner 即多进程,IPC 通道是其
既有机制,UUID 钉卡与 P2P 姿态对齐本工作区纪律);REQ-DEC-04(metrics)
骨架随 P1 建,字段随阶段增。

## 八、矛盾裁决记录(生效优先级:见效最快 > 架构洁癖)

| # | 冲突 | 裁决 |
|---|---|---|
| C1 | CUDA graph(定shape) vs chunked prefill(动shape) | piecewise graph 分段捕获,decode 全图(vLLM 成熟方案);不为了 graph 放弃 chunk |
| C2 | trait 静态分发 vs 配置期多态需求 | 泛型/impl Trait 静态分发为默认(编译期单态化);`dyn Trait` 仅允许出现在配置期/注册表期(模型注册、arch 分发表——运行前已定,不在每 token 路径上);热路径零 dyn 零虚表 |
| C3 | tokio 异步 vs CUDA 同步语义 | 计算隔离在 runner 线程;tokio 只管 API/编排;禁止为"纯异步"把 kernel launch 包 async |
| C4 | 三级 KV 的 host RAM vs IPC/NCCL 缓冲区抢内存 | host RAM 预算显式配置(pinned 池上限),默认关闭三级缓存,需要时按场景开 |
| C5 | fp8 KV 池×2 vs 速度税 | 默认 fp16 KV(对照口径);fp8/turbo 作为 P2 数据项,不进默认档 |
| C6 | W4A8"新格式"错觉 | **不是新格式**:W4A16 checkpoint 原地跑,A 侧运行时动态量化,无转换无校准 |
| C7 | 投机解码 vs graph/开销 | spec-aware graph + REQ-DEC-04 计时护栏;净增益 ≤0 的配置默认关(工作区 dflash 教训) |
| C8 | 模型广度 vs 扩展接口 | 支持列表锁死两类;扩展点 = ModelRegistry trait + add-model 流程文档化(xinfer 有 AI-assisted skill 可借) |

## 九、变更记录

| 日期 | 版本 | 变更 | 依据 |
|---|---|---|---|
| 2026-09-20 | v1.0-draft | 需求阐述轮:全量需求编号化 + 矛盾裁决 8 条 | 本会话需求阐述 + docs/README.md 口径 + 会战/调研数据 |
| 2026-09-20 | v1.1 | 订正 REQ-CODE-01:trait 优先 = 泛型静态分发,dyn 仅限配置期(C2 同步改写);新增 REQ-CTX-06 YaRN(P2 延后) | 用户订正 |
| 2026-09-20 | v1.2 | 新增 REQ-DEC-05 GDN CUDA 算子(port lmdeploy turbomind legacy 路径);克隆 lmdeploy 原材料 | GDN CUDA 实现存在性调研收口 |

## 十、开放问题(不阻塞开工)

1. PP2 vs TP2+P2P 的 prefill 对拍(REQ-CTX-02,avesed 论断本地验证)
2. q4 KV(turbo4)精度损伤量化:疑似可感,需 ppl/任务双口径
3. DSpark/DDTree 草稿的接入面(依赖上游 head 格式调研)
4. EXL3 trellis 格式是否值得 port(工作区 exllamav3 实测 2012 t/s prefill
   有诱惑力,但 kernel 移植量大,与 Marlin 路线重叠度待评估)

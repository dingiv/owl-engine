# Owl 立项总设 —— graph 与 TP 一等公民的推理引擎

> 版本:v0.1(2026-09-22 立项)。需求基线:继承 `packages/xinfer/docs/arch/requirements.md`
> v1.0 全部定量目标(prefill 7656 / decode 157 / ctx 262k,及格/达标/冲刺三线不变),
> 本文档不重复指标表,只写**架构层裁决**。
> 关系:生产 = vLLM(不动);对照与判例库 = xinfer(冻结存档);本项目 = 从架构层
> 重新设计的继承者。原材料借力清单见 §三。

---

## 一、立项动机(为什么不是继续改 xinfer)

1. **graph 内存安全不是 bug,是架构位**:xinfer 的 graph 是后补的(`utils/graph.rs`
   包在模型外面),池记账、预算、捕获时机全部晚于内存规划存在——456524b 泄露案
   证明这条路补到头也只是"纪律绕行"。新引擎从内存规划器第一天就画预算。
2. **TP 通信同理**:xinfer 的通信在 attention-rs/nccl 后端里隐式发生,消息尺寸
   路由、graph 可捕获性、融合机会全部没有被架构表达。新引擎把通信做成一等
   crate,策略显式化。
3. **借力条件已成熟**:candle fork(图安全焊缝)、attention-rs(kernel+通信)、
   marlin-ffi(自研 W4A16/W4A8/moe)、DFlash2、GDN port 规格全部是已验证的
   现成件——新引擎的增量工作可以收缩在**编排层与内存/通信治理层**,这是
   xinfer 用血换来的、社区没有的部分。

## 二、设计公理(违反即返工,评审门禁)

### A1. Graph 是内存规划的一等公民(不是功能开关)

- **公理 A1.1 预算前置**:内存规划器从第一天就包含 `graph_allowance` 字段
  (借 ninfer `DecodeGraphProfile`),任何池计算发生前,先登记 decode 图 /
  verify 图 / 未来 piecewise 段的预留。池 = 显存 - 权重 - 激活 - **图预算**。
- **公理 A1.2 生命周期律**:图实例化后到销毁前,**禁止 trim/empty_cache/
  池属性变更**。回收只发生在两个合法窗口:启动捕获前(净空)、全部图销毁后。
  456524b 案与 mistral cuMemPoolTrimTo 案的法律条文,写进 graph governor 的
  debug_assert。
- **公理 A1.3 单共享捕获池**:所有图(ordinary/verify/未来 piecewise 段)
  共享一个 capture pool(vLLM `graph_pool_handle` 语义),峰值 ≈ 最大一张图,
  不允许各图独立池互相博弈提交窗口。
- **公理 A1.4 捕获预检 + 优雅降级**:每次捕获前查 free 显存,不足则收窄
  bs 档位表,再不足则整级关图回 eager——**禁止撞死**(vLLM 行为)。
- **公理 A1.5 kernel 接口可捕获化**:自研/移植 kernel 的契约里写死:
  形状由档位固定、动态量(frontier/pos/KV 表)一律 device 张量读、热路径
  禁 D2H/同步/host 分支。attention 不做例外——paged KV + device 标量,
  目标是 decode FULL 图,不为 piecewise 留 attention 豁口。

### A2. TP 通信是一等 crate(不是后端细节)

- **公理 A2.1 消息尺寸路由显式化**:comm 层按消息尺寸选后端——小消息
  (KB 级,decode allreduce)走单 kernel one-shot AR(可融合、可捕获);
  大消息(prefill 激活)走 two-shot/分块 ring。路由策略是架构对象,不是
  运行时巧合。依据:exl3 卷——prefill nccl +38% / decode native +22%。
- **公理 A2.2 通信必须可捕获或显式出段**:comm 后端要么 graph-capture
  安全(自研 AR:裸 kernel 天生安全),要么声明"只能在段边界 eager 执行",
  类型系统表达(`enum CommCap { InGraph, SegmentBoundary }`),
  编排层据此排段。禁止"捕获期意外撞上不可捕获通信"这类运行时惊喜。
- **公理 A2.5 NCCL 除名(2026-09-22 裁决)**:NCCL 内部自建通信缓冲
  (每卡数百 MB,不透明不可登记),违反 A5.2"一切分配过账本";capture
  支持姿势苛刻,违反 A2.2。通信全栈自研:one-shot/two-shot AR
  (port vLLM custom_all_reduce)+ IPC/VMM P2P 通道 + SHM host 中转
  保险丝(llama/exl3 native 判例)。**一切通信缓冲经后端账本登记**。
  这是 REQ-DESIGN-03 例外条款的正式引用(社区无可抄的"可审计 allreduce")。
- **公理 A2.6 设备隔离律 + 单进程多实例拓扑(2026-09-22 二次裁决,
  推翻初版多进程倾向)**:一个后端实例 = 一张卡 = 一个独立账本
  (DeviceDesc 绑 UUID,数字序禁);后端实例禁止触碰第二张卡;
  多卡 = **单进程内 N 个后端实例**,每卡一个 runner 线程
  (A3 的 bind_to_thread 纪律保证 context↔线程绑定)。
  跨卡通信走 enablePeerAccess 直写(one-shot AR),无 IPC handle
  序列化协议;控制面 = 进程内 channel。故障语义见 A6。
- **公理 A2.3 融合口子内建**:AR+残差+norm 的融合接口在 comm trait 里预留
  (`fused AR add rmsnorm`),decode 带宽-bound 的最大通信杠杆。
- **公理 A2.4 卡管理纪律继承**:UUID 钉卡唯一合法(数字序禁,09-19 事故);
  IPC 主路 / SHM 保险丝;P2P 能力探测与黑名单(GPU2 类雷卡)在 comm 初始化
  时完成,不进热路径。

### A5. 严格显存预算(硬合同,2026-09-22 增补)

启动时声明预算 B(如 23G),整个程序生命周期的设备内存占用**恒 ≤ B**。
可证伪,不可协商:

- **A5.1 分解封闭**:B = 权重 + KV 池 + 图预算(A1.1)+ 激活峰值 +
  workspace + 通信缓冲 + 运行时底价 + 安全余量。任何一项没有实测数字,
  **拒绝启动**。"先跑起来再看显存"在本引擎不存在。
- **A5.2 一切分配过账本**:后端层是唯一分配出口,账本按域记账
  (persistent 存量/scratch 峰值/延迟释放滞留);池外原语
  (cuBLAS workspace、NCCL 缓冲)必须预分配并登记——mistral trim 案
  与 vLLM draft 捕获 OOM 都是这一条缺失的判例。
- **A5.3 三层强制**:①启动静态证明(分解封闭性校验,超 B 拒启);
  ②运行时账本断言(每次分配记账,超支 fail-fast,debug 构建硬断言);
  ③周期性 `mem_get_info` 对账(捕捉账本外漏网,driver 无硬配额,
  对账是最后防线)。
- **A5.4 超支即违约**:运行中超 B 是架构 bug,处理方式是 fail-fast
  (带账本快照),不是优雅降级。降级只属于捕获预检(A1.4)的
  **事前**行为。

- **公理 A2.7 故障语义 = 全有或全无(2026-09-22 裁决)**:本地单人
  引擎不做 rank 级故障隔离与带病运行。任一 runner 线程报错 →
  全引擎进入 `Paused/Unrecoverable` 状态:① 停止调度,冻结全部设备的
  `LedgerSnapshot`;② 错误链完整收集(不崩溃,错误要能被读到);
  ③ 对外健康状态 = Unrecoverable。恢复 = 整进程重启,无 rank 级
  热恢复。Xid 级硬故障掀翻进程是其极端形态,与本题无冲突。

- **公理 A2.8 VMM-only 远程显存(2026-09-22 裁决,魔改驱动约束)**:
  本引擎适配魔改驱动(BAR1 仅 256MiB,无 ReBAR)。跨卡共享/可被对端
  P2P 映射的缓冲,**必须**走新版虚拟内存管理 API(cuMemCreate/
  cuMemMap/cuMemSetAccess,物理 chunk 粒度,律 2MiB),**禁止**使用
  legacy cudaMalloc/池分配导出——大块连续段无法通过 256MiB BAR1 窗口
  (vmm-p2p 会战判例:857MB 段 × 192MB 预算墙)。推论:
  ① stream-ordered 分配(alloc_scratch)仅限本卡自用,永不导出;
  ② `PoolKind::PeerShared` 独立成池,粒度对齐 2MiB,BAR1 窗口占用
     显式入账(账本新增 BAR1 维度,§A5.3 对账扩展);
  ③ cuMemExportToShareableHandle 留作未来多进程 IPC 的升级路径,
     与单进程 peer 直映射不冲突。

### A3. 同步点隔离(继承 REQ-CODE-03 并升格)

全引擎真阻塞点只有 logits D2H。runner 线程持有全部 CUDA 上下文,tokio 侧
零 CUDA 调用;runner↔tokio 用有界 channel。这条从第一天硬约束,不让
"顺手在 async 里碰一下 device"的口子存在。

### A4. 所有权边界(vendor 纪律,吸取 xinfer 尾期裁决)

- 借力 crate **冻结为只读快照**,新引擎**不修它们的语义**;需要的新语义
  全部在自己的 crate 层实现(graph governor / comm 策略)。
- 若未来审计发现快照语义有洞需要动它:那是**有意识的所有权转移**——
  按认领纪律走(parity 硬门 + 差异清单 + 只 cherry-pick 不 merge),
  不允许顺手改。
- 每个快照登记来源 rev 与已知差异(见 §三表)。

## 三、借力清单(原材料 → 用法)

| 原材料 | 来源 | 用法 | 所有权 |
|---|---|---|---|
| candle fork(图安全焊缝) | repos/candle-gb @ 23a6f38 | 张量底座,只读快照 | 冻结 |
| attention-rs @ c0f19f2 | repos/attention.rs | paged attention/fused rope/moe kernel | 冻结 |
| mistral.rs   |  全库 | copy 源码 | 冻结 |
| marlin-ffi / moe-marlin-ffi | packages/xinfer(P3 产物) | W4A16/W4A8/moe GEMM 主力算子 | **我们自己写的,活跃** |
| cudarc fork @ 2e81793 | 经 candle 传递 | driver API(graph/池) | 冻结 |
| DFlash2 草稿 | models/syvai 等 | 投机解码草稿 | 数据 |
| GDN port 规格 | requirements REQ-DEC-05 | turbomind delta_rule 移植任务书 | 任务 |
| graph 治理判例 | xinfer 全案 + mistral 案 + ninfer 源码 | A1 公理的出处与测试用例来源 | 判例库 |
| vLLM | 生产现役 | 对照基准 + piecewise/降级语义参考 | 外部 |

## 四、crate 划分

```
packages/owl/
├── crates/graph    图运行时:预算登记/捕获治理/replay 调度/降级(公理 A1 全部落点)
├── crates/comm     TP 通信:尺寸路由/one-shot AR/NCCL 边界/融合口(公理 A2 全部落点)
├── crates/kernels  算子分发:arch 表(sm86 先行)/marlin 接线/可捕获 kernel 契约(A1.5)
├── crates/core     模型注册表/scheduler/runner(同步隔离 A3)/内存规划器(含 graph_allowance)
└── crates/server   tokio API 层(M2 才开)
```

依赖方向单向:`server → core → {graph, comm, kernels}`;graph/comm/kernels
互不依赖,共享类型下沉 core。数据面:candle 快照作为 core 的张量后端引入
(方式待 M1 定:直接依赖 or 自持薄层)。

## 五、里程碑(粗粒度,细节走 roadmap)

- **M0(本轮)**:立项文档 + 骨架编译绿 + 公理落成 trait 签名。
- **M1 内存规划器 + 图运行时最小闭环**:单卡、单模型(小 decoder)、decode
  全图捕获/replay,allowance 预算生效,A1.2/A1.3/A1.4 有 debug_assert 与
  单测。验收:eager↔graph 数值对拍,重复捕获/销毁循环无显存漂移。
- **M2 TP2 通信**:comm 尺寸路由 + one-shot AR + NCCL 边界;UUID 钉卡;
  TP2 数值对拍 TP1。验收:对照 vLLM 同机 TP2 prefill/decode 三线表。
- **M3 模型与投机**:Qwen3.5 dense 接入(借 candle/attention-rs)+ DFlash2
  spec-aware graph;对齐 xinfer 3090 对成绩(decode 157 线)。
- **M4 超越线**:marlin moe / W4A8 / 三级缓存 / Nex MoE,对 requirements.md
  冲刺线。

## 六、风险与既知难点

1. **candle 快照与自研 core 的贴合度未知**:candle 的 eager 分配语义与
   A1 治理层的交界(参数缓存 scope 何时开)是 M1 第一个要验证的接缝;
   若不贴,退路是 core 自持薄张量层(kernels 直连),candle 降级为参考实现。
2. **one-shot AR 自研是本项目少数"无现成 Rust 件"的点**:参考 vLLM
   custom_all_reduce csrc 移植,工作量可控但属自研热路径,REQ-DESIGN-03
   的例外条款适用(社区无可抄才自研,此处成立)。
3. **单人工程带宽**:milestone 严格串行,M2 之前不碰任何模型广度。

## 七、裁决记录

- 2026-09-22 立项:独立新包 `owl`;xinfer 冻结存档(456524b 不修,作为
  反面判例归档);vLLM 继续生产;graph/TP 升格为一等公民并形成 A1/A2 公理。
- 2026-09-22 增补:A5 严格显存预算(硬合同);A2.5 NCCL 除名(通信全自研);
  A2.6 设备隔离律 + 单进程多实例拓扑(二次裁决,推翻初版多进程倾向);
  A2.7 故障语义全有或全无(Paused/Unrecoverable);iface 增加显存池管理与
  DeviceDesc 抽象,backends 增加第三方 CUDA 准入铁律(账外分配零容忍)。

# TP(张量并行)实现方案综述 —— 从数学骨架到各引擎实况

> 2026-10-16 · 定位:**M2 的背景读物**(与《TP2-通信-预研.md》配套:本综述讲"行业怎么做、
> 为什么这么做",预研卷讲"我们这台机器上怎么做、要写什么")。
> 读者假设:读完能独立回答"为什么每层恰好两次 allreduce""为什么 NCCL 在我们这台机上不能用"
> "为什么我们的 decode 只有 1.2~1.4 倍"。
> 相关:charter §二 A2(六条通信公理)· `crates/backends/comm/src/lib.rs`(策略骨架)·
> 《TP2-通信-预研.md》(现实约束表 / 27B 切分面 / §9 最小集合库设计)。

---

## 1. TP 的本质(一句话 + 数学骨架)

**TP = 把每个线性层的权重按维度切开分到各卡,各卡算"部分结果",再用集合通信拼回全量。**
它买的只有两件事:**① 单卡装不下的权重能装下**(容量);**② 显存带宽 bound 的 decode 近似线性加速**(带宽)。

### 1.1 两种切法(整个 TP 的全部基础)

设 `Y = X·W`,`X [T, K]`,`W [K, N]`:

| 切法 | W 怎么切 | 各 rank 算出什么 | 要通信吗 |
|---|---|---|---|
| **列并行(ColumnParallel)** | 沿 N 切 `W = [W₁ \| W₂]` | `Y₁ = X·W₁` —— 输出**分片**(输入全量) | 不用(出分片) |
| **行并行(RowParallel)** | 沿 K 切 `W = [W₁; W₂]` | `Yᵢ = Xᵢ·Wᵢ` —— **部分和** | **allreduce** 求和 |

**关键洞察:层内成对使用,刚好把"分片"与"全量"两边接上** ——
列并行的分片输出可以当行并行的分片输入;行并行经 AR 变回全量,喂给下一层的列并行。
⇒ **每层正好 2 次 AR**,且 AR 一定落在**行并行的输出**上。

### 1.2 Megatron 配方(2019 起,所有引擎同源)

```
Attention:  QKV proj 列并行 → 各 rank 在自己头上算 attention → o_proj 行并行 → AR ─┐
                                                                                  ├→ 残差 + LN(全量,复制)
MLP:        gate/up 列并行 → silu·mul(本地) → down 行并行 → AR ────────────────────┘
```

即每层两次 `[T, hidden]` 的 AR。**这两个 AR 恰好紧挨着残差加法** ⇒
`AR + 残差 + LayerNorm` 可以融合成一条核(Megatron / vLLM / TRT-LLM 都做),
这正是 charter **A2.3**(`fused AR add rmsnorm`)的口子。

### 1.3 为什么"列/行"必须成对(不是优化选择,是结构节奏)

只做列并行 ⇒ 每卡只有部分头,attention 内部需要全量头 ⇒ 必须 o_proj 行并行 + AR 拼回全量,
下一层才能再列并行。**"分片 ↔ 全量"的交替是 TP 的结构性节奏。**

---

## 2. 一次前向的完整数据流(以本机 27B 为例)

27B:`hidden 5120 / 64 层 = 48 GDN + 16 全注意力 / inter 17408 / vocab 248320 / group 128`,TP2:

| 张量 | 切法 | 每 rank 形状 | 通信 |
|---|---|---|---|
| `embed_tokens` | 词表行切 | [124160, 5120] | 装权重 |
| attn `q_proj` [6144,5120] | 列(按 Q 头) | [3072,5120] | — |
| attn `k/v_proj` [1024,5120] | **列(按 KV 头)** | [512,5120] | KV 池按头分片 |
| attn 注意力 | 每 rank 只算自己的 2 个 KV 头 | — | — |
| attn `o_proj` [5120,6144] | 行 | [5120,3072] | **AR [T,5120]** |
| GDN `in_proj_qkvz` | 列(NK 8 / NV 24) | K 1024 / V 3072 | — |
| GDN conv / 递推 | 通道切 / **按 V 头切** | conv 5120 / S: NV 24 | 状态池 per-rank 减半 |
| GDN `out_proj` | 行 | [5120,3072] | **AR [T,5120]** |
| MLP `gate/up` | 列 | [8704,5120] × 2 | — |
| MLP `down` | 行 | [5120,8704] | **AR [T,5120]** |
| `lm_head` | 列(词表)或**复制** | [124160,5120] 或全量 | 词表并行需 gather logits |

**每层 2 次 AR × 64 层 = 128 次/步**,载荷 `[T,5120]×dtype`:

- **decode** T=1:f16 **10 KB/次 ⇒ 1.31 MB/步** ⇒ 13.2GB/s 下仅 99µs,
  但 **128 次发射的延迟(~0.4-0.6ms)才是主角**;
- **prefill** T=8192:**80 MB/次 ⇒ 10.7 GB/rank** ⇒ 0.81s,必须与 GEMM 重叠。

---

## 3. 通信原语谱与成本模型

| 原语 | 每 rank 字节(N=2) | 同步步数 | 甜区 |
|---|---|---|---|
| **ring allreduce** | 1× payload(reduce-scatter 半 + allgather 半) | 2(N−1) 步 | **大消息**(prefill 80MB 级) |
| **one-shot AR**(写对端 staging + 读回累加) | 2× payload(1 写 + 1 读) | **1 轮** | **小消息**(decode KB 级) |
| **two-shot** | 2× payload | 2 轮 | 中等 |
| **reduce-scatter + all-gather** | 1× payload | 各 1 轮 | 与 SP(序列并行)配套 |
| **SHM / host 中转** | 2× 过 PCIe + 主机内存 | 1 轮(+2 拷贝) | 保险丝(有效 6-12GB/s) |

**成本模型**(TP=N,层内两次 AR):

```
步时 ≈ 权重字节/rank ÷ 显存带宽                    (compute/带宽项)
     + 2 × T × hidden × dtype × 2(N−1)/N ÷ 互连带宽 (AR 带宽项,ring)
     + 128 × 单发延迟                               (AR 延迟项)
```

三项相对大小随 T 剧变 ⇒ **A2.1"尺寸路由"的物理依据**:
decode 走 one-shot(省同步步),prefill 走 ring/two-shot(省字节)+ **与 GEMM 重叠**。

**两个必备工程手段**:
1. **融合**:`AR + 残差 + RMSNorm` 一条核(vLLM / TRT-LLM / Megatron 皆有);
2. **重叠(chunk pipeline)**:`[T,hidden]` 切块,第 i 块 AR 与第 i+1 块 GEMM 双流并行 ——
   prefill 那 0.81s 才可能被藏掉大半。

---

## 4. 各引擎的实现实况(带本机实测读数)

| 引擎 | 进程/拓扑 | 通信后端 | 切分实现 | 本机实测 |
|---|---|---|---|---|
| **vLLM** | **每 rank 一进程**(Ray/多进程 worker),NCCL 通信域 | NCCL → **custom_all_reduce**(one-shot,可入图);本机必须 `--disable-custom-all-reduce` | Megatron 式 `Column/RowParallelLinear` 移植;**lm_head 词表并行 + logits all_gather**;KV 按头分片 | **27B 3090 对 TP2:prefill 2040.6 / decode 108.9**(TP1 46.86 ⇒ **2.32×**);3080 对开 P2P:prefill **+28.5%** / decode +2.4% |
| **SGLang** | 每 rank 一进程 | NCCL + 自研 | 同源;前缀缓存靠 **`all_reduce(min)` 同步各 rank 命中数** | 27B 3090 对 `tp=2`(生产姿势) |
| **Megatron-LM**(祖师爷) | 训练向 TP+PP+DP+**SP** 全谱 | NCCL | 列/行并行 + **fused AR+add+LN**;SP 把 AR 换成 reduce-scatter/all-gather | — |
| **TensorRT-LLM** | 每 rank 一进程 | NCCL + 自研 plugin | 同源;核级融合最深(AR 与后续 op 融合) | — |
| **llama.cpp** | **单进程多卡** | `allreduce.cu`:internal(**SHM**)/ nccl / (P2P 设计完成待施工) | `--split-mode row`(=列并行 + 部分和,**host 归约**)/ `layer`(=流水,非 TP) | SHM **2538/131.7** · internal 2308/133.6 · none 1904/98.8;**P2P 被 192MB 墙结构性封死** |
| **exllamav3** | 单进程多卡 | **native(host SHM)** vs **NCCL** 两条 | `-tp -gs` 切分 | **prefill 要 NCCL(1809 vs 1307,+38%)/ decode 要 native(~91 vs ~75,+22%)** |
| **mistral.rs** | 单进程(多 context) | NCCL | 自研 + NCCL | 3090 对 decode **173.8**(该轮全场王) |
| **xinfer(candle 系)** | 单进程 | NCCL | TP2 | 3090 对 ISQ q4k decode **84** |

**结论:全世界的 TP 都是 2019 Megatron 的变体**,真正的分野只有四处:
**① 进程模型**(每 rank 一进程 vs 单进程多 context)、**② 通信后端**(NCCL vs 自研 one-shot vs SHM)、
**③ 融合/重叠深度**、**④ 采样与缓存面**(logits 怎么收、前缀缓存怎么同步)。

---

## 5. 邻居与变体(别把 TP 当唯一答案)

| 方案 | 切什么 | 通信 | 何时用 |
|---|---|---|---|
| **TP** | 权重维度 | allreduce(高频,小-大) | 单机内、延迟敏感(**我们**) |
| **PP**(流水) | 按层 | p2p send/recv(低频,大) | 跨机/层多;推理有 bubble、加延迟 |
| **SP**(序列并行) | 序列维 | reduce-scatter/all-gather | 长 ctx 激活内存;可把 AR 变形省带宽 |
| **EP**(专家并行) | 专家 | all-to-all | MoE 主力 |
| **DP**(数据并行) | 请求 | 无(副本独立) | 吞吐;与 TP 正交 |
| **Ring/Ulysses Attention** | 序列×头 | p2p 环 / all-to-all | 100k+ ctx 注意力(替代 KV 按头切) |
| llama.cpp `layer` split | 按层 | 无(轮流) | 单并发省内存;**不是 TP** |

**与我们的相关性**:推理里 TP 与 DP 正交 —— 同一对 3090 可以
"TP2 跑一个 27B"(压单流延迟/decode 带宽)或"DP2 各跑各的"(压吞吐)。

---

## 6. 传输层与可捕获性(本机最要命的两件事)

**链路层次**(自下而上):NVLink → PCIe P2P(BAR1 窗口)→ IPC handle → SHM host 中转。
本机**无 NVLink**(3090 Ti 与 3080 皆无)⇒ 天花板 = **PCIe P2P 13.2 GB/s**(实测六对),
SHM 兜底 6-12GB/s。

**为什么我们的通信必须自研**(A2.5 三条理由 + 本机第四条):
1. NCCL 自建缓冲不透明、不过账本(违 A5.2);
2. NCCL capture 姿势苛刻(违 A2.2);
3. **NCCL 的 P2P transport 自行调用 runtime peer-enable** ⇒ 3080 类对上触发进程级急切映射
   ⇒ **192MB 墙逐笔撞死**(llama.cpp 的"三味全灭"即此);
4. 自研 one-shot 是裸 kernel ⇒ **天生 InGraph**,与 owl 的图优先架构同构。

**捕获**:vLLM 的 custom one-shot AR 用"flag + 自旋"协议,这正是它能进图的原因,
但它要跨进程 IPC + 每轮 flag 重置;我们单进程 + VMM 显式授予 + peer alias 直读,
协议可以更薄(预研卷 §9.3)。

---

## 7. 精度 / 确定性 / 工程陷阱清单

1. **浮点非结合性** ⇒ TP2 与 TP1 **不可能逐位相同**;"TP2 ≡ TP1"门必须按
   **greedy 逐 token** 判(如 `gpu_spec_identity_gate`),不能按张量逐位判。
2. **采样 RNG 分叉**:两 rank 各采样会因随机流不同而分叉。通行解:
   **rank0(末 rank)采样 + 广播 token**,或全 rank 同步 RNG 状态。
   我们 v1(lm_head 复制 ⇒ 同 logits ⇒ 全 greedy)天然无分叉,**但开 temperature 必须处理**。
3. **KV 头数 < TP**:全注意力层仅 4 KV 头,TP8 无法按头切 ⇒ **KV 复制**(vLLM 做法)。
   27B:4/TP2=2 ✓;GDN 16/48 ✓。
4. **量化 group 边界必须对齐分片**:行并行切 K 时 `K/TP % group_size == 0`。
   27B:hidden 5120/2=2560=128×20 ✓;inter 17408/2=8704=128×68 ✓。
   **换 group_size 或 TP4 要重验。**
5. **GLUE 复制 = TP 红利上限**:norm/rope/残差在 TP 下是复制(每卡算全量 hidden)。
   本机逐核归因:胶水 36% + 调度 20% 不随 TP 切分 ⇒ **TP2 decode 上限 ~1.2-1.4×**,
   而非 vLLM 的 2.32×。**A2.3 融合口因此是必需品,不是可选项。**
6. **前缀缓存跨 rank 同步**:每 rank 只持自己的 KV 分片,命中/驱逐必须一致
   (sglang 用 `all_reduce(min)`)⇒ 我们把 prefix cache 排在 TP 之后。
7. **图捕获要求所有 rank 都发射**:任一 rank 少发一次 AR ⇒ flag 自旋永久等 = 死锁
   ⇒ 编排层必须保证两 rank 图严格同构(A2.2 类型系统的价值)。
8. **显存账**:TP2 每 rank 装一半权重,但**图预算/激活/通信 staging 不减半**;
   20G 卡上要算清(预研卷 §7.1 #6)。
9. **加载路径**:量化 checkpoint 最好**预先按 shard 存/只读自己要的键**,
   而不是全读再切(浪费 PCIe 与内存)。

---

## 8. 对 owl 的推论(方案 → 我们的选择)

| 设计轴 | 主流 | owl 的选择 | 理由 |
|---|---|---|---|
| 进程模型 | 每 rank 一进程(vLLM) | **单进程 N 后端**(A2.6) | 账本/图/租约是进程内对象;免 IPC 协议层 |
| 通信后端 | NCCL | **自研 one-shot + ring**(A2.5) | 账本合规 + 可捕获 + 3080 档禁 enable |
| 传输机制 | cudaMalloc + IPC handle | **VMM 显式授予 + peer alias** | 唯一能过 192MB 墙的单进程姿势 |
| 切分 | Megatron 列/行 | 同 + GDN **head-parallel** | GDN 天然按头切(预研卷 §2) |
| lm_head(v1) | 词表并行 + gather | **复制**(省一套 AG) | "只满足基本需求";AG 留位 |
| 融合 | AR+add+LN | **二期**(口子在 A2.3) | v1 先把正确性/账本/图捕获钉死 |
| 最佳参照 | — | **vLLM custom_all_reduce**(给协议)+ **llama.cpp p2p 设计**(给 3080 档姿势) | 两个都已在工作区有卷宗 |

**收束**:TP 的算法三十年没变(列/行并行 + 2×AR/层);
真正的工程分水岭只有两条 —— **怎么把 AR 做进图、怎么在本机链路现实(13.2GB/s + 192MB 墙)里把它做便宜**。
这也是我们的集合库值得自写的理由:难点不在算法,而在**图、账本、驱动边界**这三件别人不做的事。

---

## 出处

- 本机 vLLM/llama/exl3/mistral 读数:`docs/bench/vllm.md`、`docs/bench/qwen38-3090单卡-引擎会战.md`、
  `docs/bench/nex-3080对-三引擎会战.md`、`docs/exllamav3/*`、`docs/llama.cpp/p2p-allreduce-后端-施工文档.md`
- 驱动/链接现实与 192MB 墙:`docs/llama.cpp/v3基线不通-立案排查.md` §十、`vmm-p2p-3080会战-测试报告.md`、
  `docs/llama.cpp/3080-p2p驱动攻坚-setAccess205-定位实录.md`、`p2p-build/README.md`
- 本机逐核归因(胶水 36%):`roadmap.local/代差分析与演进路线.md`(2026-10-01)
- GDN/27B 结构:`models/cyankiwi/Qwen3.8-27B-AWQ-INT4/config.json`
- 公理侧:`docs/arch/charter.md` §二 A2;骨架:`crates/backends/comm/src/lib.rs`

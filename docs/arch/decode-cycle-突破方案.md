# decode cycle 突破方案(vLLM/SGLang 对标)

> 2026-10-03。前置:decode cycle 全息分账(同日台账)——owl 27B 单卡 decode 步
> **28.5ms**(nsys lm-start-to-lm-start,python 客户端口径 20 t/s 系同污染相对值),
> vLLM 同模型同卡 **21.3ms**(46.9 t/s,同工具)。差 7.2ms。本文 = 对比、归因、方案。

## 一、对标:vLLM Qwen3-Next decode 步结构(repos/vllm 实读)

GDN 层(`vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`,1748 行):

| 项 | vLLM | owl 现状 | 差 |
|---|---|---|---|
| 输入投影 | **in_proj_qkvz(1 发)+ in_proj_ba(1 发)** —— 列合并 | in_proj_qkv / in_proj_z / in_proj_b / in_proj_a **4 发** | **+2 GEMM/层** |
| conv 更新 | **causal_conv1d_update(1 发)** —— Tri Dao 核,单 state 块全段 | conv_upd ×3(q/q/k/v 三段三 state 块) | **+2 发/层** |
| gating | **fused_gdn_gating(1 发)** g+β 合算 | gating_g + sigmoid 2 发(D1 融合核内已并入) | 0(D1 后) |
| delta 递推 | fused_recurrent_gated_delta_rule_packed_decode(1 发) | delta_dec 1 发(或 D1 融合核) | 0 |
| q/k 归一 | l2norm 核 1-2 发 | D1 核内(或 2 发) | 0(D1 后) |
| 张量切片 | **PyTorch view = 零开销零发射** | **narrow = 物化内核 561 发/步**(GPU 0.69ms + 节点开销) | **+561 发/步** |
| 采样 | GPU 侧(logits 不下卡),token id 回读 | S3 host 采样:1MB logits D2H + host 惩罚链 + 同步 | **~1-3ms/步** |
| 图 | 全步单图捕获 ✓ | 同 ✓ | 0 |

GEMM 发射数:owl ~365-496/步 vs vLLM ~304(投影合并后)。
每步内核总数:owl **1924** vs vLLM 估 ~700-900。

## 二、归因(28.5 - 21.3 = 7.2ms 的去向)

1. **投影碎片化**(~1.5-2ms):in_proj 4 发(其中 b/a n=96 小 GEMM BW 极低)+
   mlp gate/up 分发;vLLM 合并后大 GEMM 单发(字节守恒,省的是发射开销与小
   GEMM 低效,不是大 GEMM 时间)
2. **narrow 物化**(~1-1.7ms):0.69ms GPU + 561 节点调度
3. **采样 host 链**(~1-3ms):1MB D2H 同步 + host 惩罚/采样
4. **GDN 链散装**(~0.5-1ms):conv 三段、gating 两发、delta state 带宽仅
   210GB/s(理想 ~900)
5. 其余杂项(~0.5ms)

## 三、方案(按依赖序三刀 + 一立项)

### 刀1:BlockSlice 协议原语(结构性,优先)

`Arg::Block { id }` → 增 `Arg::BlockSlice { id, byte_offset, elems }`;
narrow 在 SSA 期记录 (parent, offset),发射面传 `parent_ptr + offset` ——**切片
变视图,narrow 内核归零**。这是 PyTorch「slice = view」的 owl 对应物,也是
刀2 的前置(合并投影必然增多切片)。

- 影响面:contract(Arg)/ops lower/pool 句柄/launch/server——中等手术
- 收益:-561 发 + 0.69ms GPU ≈ **-1.2ms**;并永久性消除一切「为喂核而物化」
- 风险:块生命周期(切片与父同生灭,池引用 +1);对拍面全量回归

### 刀2:投影合并(装载期列拼接,零内核改动)

- GDN:in_proj_qkv+z+b+a(4)→ **qkvz + ba(2)**;mlp:gate/up(2)→ **gate_up(1)**;
  attn:k/v(2)→ 1
- AWQ/compressed-tensors 列拼接 = packed/scales/zeros **行堆叠**(组沿 in 维,
  out 维拼接无交叉;out 维均 %8==0)——装载期一次完成,零数值风险
- GEMM:48×(4+1+3→2+1+2)= 384-96 = **-96 发**;attn -16 发;含小 GEMM 低 BW
  回收 ≈ **-1.5~2ms**
- 依赖刀1(qkvz 输出喂 conv/l2norm 需要 4-6 个切片)

### 刀3:GDN conv 三段合一 + 采样下沉

- conv state 三块 → 单块 [slots, conv_dim, 3],conv_upd 一次吃全段
  (causal_conv1d_update 同构;融合 decode 核的 v 段寻址同步改)——-2 发/层
- 采样下沉:S3 惩罚链改设备核(logits 不下卡,token 4B 回读)——需先解
  greedy 复读质量问题(S3 立案动机;OWL_TEMP=0 设备 argmax ≡ greedy 可 A/B)
- 收益:conv ≈ -0.3ms;采样 ≈ -1~3ms(质量验证后)

### 立项级:E5 投机解码(DFlash2/MTP)

GEMM 已贴带宽墙(marlin m=1 1021GB/s=100%,lm_head 94%),**单 token/步的
物理下限 ≈ 权重字节/带宽 ≈ 13ms**——28.5→24 后即触墙。**唯一 >30% 的超车
杠杆 = 多 token/步**(spec decode);素材在册(DFlash2 调研归档、draft 系
模型在库)。

## 四、预期与执行序

| 步 | 内容 | 预期 | 工作量 |
|---|---|---|---|
| 刀1 | BlockSlice | -1.2ms | 1 会话(协议+发射+回归)|
| 刀2 | 投影合并 | -1.5~2ms | 1 会话(装载拼接+层改造)|
| 刀3 | conv 合一+采样下沉 | -1.3~3.3ms | 1-2 会话 |
| 合计 | | **28.5 → ~22-24ms(≈ vLLM 打平)** | |
| E5 | 投机解码 | 超车(×1.5-2) | 立项 |

## 五、GDN in_proj qkvz 合并规格(刀2 细化;2026-10-03 检查点勘察)

检查点事实(cyankiwi):`linear_attn.in_proj_qkv.weight_packed [10240, 640] I32` +
`weight_scale [10240, 160] BF16`(g=32)+ `weight_zero_point [1280, 160] I32`;
`in_proj_z` 同构 [6144, 640];**in_proj_b/a 为 BF16 未量化 [48, 5120]**。

合并规格(qkv+z,行堆叠):
- packed [16384, 640]、scale [16384, 160]、zero [2048, 160] —— 三表行堆叠,
  零数值风险(组沿 in 维,out 维拼接无交叉)
- **n=16384 = 64×256:marlin tile 约束(n%256==0)完美满足**
- b/a 不并入(BF16 未量化异构;2 发小 GEMM ~20µs,后置)
- 消费面:decode 融合核直接读 merged 基址 + 段偏移(q/k/v/z 段式索引,
  参数 +offsets);conv_upd q/k 加 x_offset/x_stride;prefill conv_fwd 同;
  z 段喂 norm_act 需 narrow 或核内偏移(推荐后者)
- 工作量:1-2 会话(awq.rs 虚拟合并键 + Linear 声明 + 全消费面 + 对拍)
- 与 D1 立案的关系:merge 落地不依赖 D1 解案(decode 旧链同样受益
  strided 化);但消费面改造建议与 D1 解案同窗(避免两次全消费面翻动)

## 六、判据纪律(本回合教训)

- 时延判据一律 `OWL_SAMPLER=greedy`(S3 采样非确定,文本/步时 A/B 均需确定态)
- ns 级分账用 `nsys --cuda-graph-trace=node` + lm_head 步界切分
- python 客户端口径仅作相对比较

## 七、计时体系二次修正与执行序再改判(2026-10-03 深夜)

**图回放 = 节点派发瓶颈(OWL_D2H_PROF 服务端分相实测)**:`STREAM_COMPUTE.synchronize()`
48.2ms vs 内核和 24.4ms → ~1900 节点以 ~25µs/节点派发,GPU 回放期 50% 空转。
本文 §三 的「刀1 收益 -1.2ms(1.8µs/发)」系 nsys node-trace 模式失真下的错估,
**作废**;真节点经济学 ≈ 25µs/节点(系统级,vLLM ~950 节点 × ~22µs ≈ 21.3ms 墙时自洽
——两侧同派发瓶颈)。

**执行序 v3**:
| 刀 | 内容 | 节点账 | 墙时预期 |
|---|---|---|---|
| D1 | GDN 融合核(已落地,墙时反转立案中) | -240 | ±0(观察) |
| 刀2 | qkvz 合并(已落地,本回合) | -48 | -1.2ms |
| **刀1** | **BlockSlice(narrow→视图)** | **-561** | **≈ -14ms(首杠杆)** |
| 刀3 | conv 三段合一 + 采样已在图外 | -96 | -2.4ms |
| 合计 | 1900 → ~955 节点 | | **≈ 24ms ≈ vLLM 打平** |
| E5 | 投机解码 | — | 超车 |

判据层级纪律:GPU 内核账(nsys)→ sync 墙时(OWL_D2H_PROF)→ harness 墙时,
三层矛盾即立案,不得单层定谳。

## 八、刀1 落地与派发税精测(2026-10-03 深夜追加)

刀1 已落地:narrow 11905→528(-95.6%),墙时 -4.5ms/token,文本双态逐字一致。
节点账:2471 → 2087(纯内核;192 对 cublas MEM_ALLOC/FREE 已由
cublasSetWorkspace 16MB 全局驻留工作区归零)。

**派发税精测**:sync 46.9ms − 内核 24.4ms = 22.5ms ÷ 2087 节点 ≈ **10.8µs/节点**。
owl 平均内核 11.4µs ≈ 派发成本 → 间隙半裸露;vLLM 平均内核 ~21µs > 派发 →
间隙全隐藏。**追平 vLLM = 节点数 ~950 + 单内核 ≥ 派发阈值。**

刀3 扩容(按消节点收益排序):
| 刀 | 内容 | 节点账 |
|---|---|---|
| 刀3a | b/a 投影零填充 pad-256 走 marlin(48%256≠0 → 填充;摘除 cublas) | -144(gemv+splitK 4 小节点/层 → 1 marlin) |
| 刀3b | GDN conv q/k 两发合一 | -48 |
| 刀3c | attn norm_rope q/k 合一 | -48 |
| 之后 | 残余融合(embedding 前处理/lm_head 后处理/残余 glue) | 逼近 950 |

## 九、刀1.6:捕获图 2.64× 整模重复执行根治(2026-10-03 深夜Ⅲ)

**时间戳探针**(OWL_TS_PROBE,probe_ts_f16 层间 clock64)首战即中:decode 图里
没有 ts 节点 → 材料单(cuGraphKernelNodeGetParams_v2 + cuFuncGetName
histogram,入 SRV_TIMING)揭盅:图 2471 节点 = 单步应有的 2~2.64 倍
(probe_ts ×132、gdn ×96、conv_upd ×192、gemvx/splitK ×192、marlin ×640)。

**根因**:try_capture 对每个输出根各跑一次 eval_ops,CSE memo 不跨调用
——"token" 树(整模+argmax)与 "logits" 树(整模)把整模捕获两遍以上。
**语义后果**:重复执行推进 GDN 状态两遍 = 每 token 被线性注意力看两次
(复读吸引子部分源于此);**性能后果**:回放税 ~2×。

**修复**:eval.rs eval_ops_multi(多根共享 memo 单次归约)+ try_capture
改单次归约逐根收割;workspace 二修(全局单例 → server 私有 4MB,并行
server 共享工作区会 ILLEGAL_ADDRESS)。

**实测**:图 2471→1044,回放 46.9→23.6ms(砍半),token-dtoh 23.7ms
≈ 42 t/s;文本从「为为为为」复读 attractor 变连贯长句,双态稳定;
models 104/104 + engine 30/30。

**对 vLLM 账本**:decode 步 23.6ms vs 21.3ms,差距 2.3ms(11%);
剩余 ≈ 1044 节点派发(刀3 家族可选)+ 内核本身。今日累计:会话起点
46.9(含重复)→ 23.6ms。

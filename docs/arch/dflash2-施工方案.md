# E5-DFlash2 施工方案 —— 语义定谳与 owl 落地设计

> 2026-10-07。上游参考:packages/sglang `srt/models/dflash.py`(1175 行,
> sglang-dflash-patch 分支)+ `kernels/ops/speculative/dflash.py` +
> dflash_worker_v2;检查点:models/z-lab/Qwen3.8-27B-DFlash2(BF16 3.85GB,
> 81 张量,1.92B 参数;W4A16/GGUF 备选在库)。
> 前置:M5 管线(verify/accept/fold/restore/桶形图/extend)drafter 无关,
> 全部复用;本文只写 DFlash2 特有件。

## 一、结构定谳(检查点 + sglang 源码双证)

```
DFlash2DraftModel(1.92B bf16):
  fc.weight          [5120, 25600]   memory = hidden_norm(fc(concat 5×target hidden))
  hidden_norm        [5120]          (target_layer_ids = [5,19,33,47,61])
  layers.0..4:                       32H/8KV hd128 GQA + RoPE(theta 1e7)+ q/k_norm
    self_attn        q[4096,5120] k/v[1024,5120] o[5120,4096]
    attention_conv   base[2,2,5120] + proj[1280,5120]   ← 分组动态深度卷积(见 §二)
    mlp              gate/up[17408,5120] down[5120,17408]
    mlp_conv         同 attention_conv
  norm               [5120]          final
  candidate_selector                 ← 草稿头(非 248K lm_head!)
    hidden_projection [256, 5120]
    predecessor_codebook [248320, 256]
    successor_codebook   [248320, 256]
  dflash_config: block_size=8(1锚+7草稿) conv_group_size=16 conv_kernel_size=2
    mask_token_id=248070 selector_rank=256 selector_top_k=16
    target_layer_ids=[5,19,33,47,61]; sliding_window=2048; is_causal=false
```

**架构要点(回应「MTP 头有问题」)**:DFlash2 草稿头 = 256 维码本选择器
(hidden_projection → 码本距离格),**一次前向出 7 草稿、零 248K GEMV**;
对比 MTP 逐步 argmax × 3 = 8ms lm_head 税。propose 图 GPU 地板
14.4ms(MTP)→ 预期 ~5-6ms(DFlash2)。

## 二、特有算子语义(sglang 源码逐式抽取)

### 2.1 DFlashGroupedConv(每子层一对:attention_conv / mlp_conv)
动态分组深度 K-tap 时间卷积(T=2,taps=2,groups=320,group_size=16):

```
coeffs = kernel_projection(x)           # [T, 2, 2, 320](input/output 两侧)
conv(x, c_side):  out[t,g] = Σ_tap c_side[tap,g] · x[t-tap,g] · [pos%8 ≥ tap]
  pos = t & 7(block_size=8;**跨块 tap 被位置掩码清零 → 卷积无跨轮状态** ✓)
层内流:  x → prepare(x)=(conv_in(x), k_out) → 子层(conv_in) → finish(·, k_out)
```
owl 落地:单 nvrtc 核(逐通道组标量系数 × 移位行 + 位置掩码;T 小)或
owl ops 组合(mul + SliceView 移位 + mask);extend 大 T 同核。

### 2.2 层流(pre-norm 残差,Qwen3 式)
```
residual = x; x = input_ln(x)
x_in = attention_conv.prepare(x); attn = self_attn(x_in); attn = conv.finish(attn)
x, residual = post_ln(attn, residual)
x_in = mlp_conv.prepare(x); h = mlp(x_in); h = mlp_conv.finish(h)
return h, residual          # (final norm 后 = 草稿 hidden)
```
self_attn = 标准 GQA(32H/8KV hd128,RoPE,sliding_window 2048,radix KV)
—— **owl Attention 层原生支持(hd128 全家在册)**。

### 2.3 memory 与草稿 KV(draft KV 物化 = sglang「正解」)
- memory = hidden_norm(fc(concat(taps[5,19,33,47,61])))  # [T_ctx, 5120]
- encode(prefill extend 同位):memory 逐行过 5 层 k_proj/v_proj →
  5 × [T_ctx, 2048] 草稿 KV 写链(= 目标 hidden 的层特异性投影);
  conv 无跨块状态(§2.1)✓ 雙 extend(mtp 版/ dflash 版)同管线位。
- propose 的 8 行 noise = embed([anchor, MASK×7]),self-attn 读草稿 KV。

### 2.4 candidate_selector(草稿头)
```
proj = hidden_projection(h[8])                        # [8, 256]
score[b,e,p,c] = unary[b,e,c] + ⟨A[pred]⊙proj, B[c]⟩   # A/B = 前/后继码本
  pred(slot e 的 p 位)= cand[e-1][p];slot 0 = verified anchor
sample_path:K=16 候选/slot 的格 → 贪心 walk(greedy_mask 行取 argmax
  —— 单图服务 greedy/sampling)
```
candidates/unary 来源 = worker 侧待最后一读(dflash_worker_v2 的
build_lattice 调用点);**greedy 域可先实现 argmax-walk 简式**(unary =
最终 norm hidden × ??? 待定)——M2 对拍锚 = sglang 同输入同输出。

## 三、owl 落地设计(复用地图)

| 件 | 来源 | 工作量 |
|---|---|---|
| verify/accept/fold/restore/快照 | M5 管线(drafter 无关)| **零** |
| 桶形图族 | M5(verify T=8 一张;propose **定形 8 行单桶** —— DFlash2 提案形状与 m 无关,**免桶**)| 小 |
| extend(草稿 KV 物化)| M5 prefill_extend 管线位,内核换 dflash encode | 小 |
| hidden taps [5,19,33,47,61] | Model forward 变体(tap 收集;tag 机制同族)| 中 |
| 分组动态卷积核 | 新 nvrtc 核 ×1 | 中 |
| DFlash2 组件 + 装载 | 新 models 组件(Attention/Mlp/RmsNorm 全复用;+conv/selector)| **大头** |
| selector greedy 简式 | 新(host walk;格构造 = 小 GEMM)| 中 |
| 身份门/AL 分域 | M5 测试架 | 零 |

轮成本预估:verify(T=8)≈ 31ms + propose ≈ 5-7ms + fold 1ms ≈ 38ms;
**AL(Xinfer 实测域)MATH 4.49 / AGENT 2.17 / CODE 2.74 / PROSE 1.13** →
MATH 域 ~115 t/s(+166%),PROSE 域 ~29(-33%)—— 域定律照旧,**分域默认开关 = C7 终局**。

## 四、里程碑

- **DF-1**:组件 + BF16 装载 + 内存/卷积语义对拍(锚 = sglang dflash.py
  同输入同输出;conv/selector 单测)
- **DF-2**:target taps + memory encode + prefill extend(草稿 KV 物化)
- **DF-3**:引擎接线(StepAction 复用;propose_first/propose 换 DFlash2 版)
  + 恒等门(greedy 逐位)
- **DF-4**:propose 定形图(单桶)+ verify T=8 图
- **DF-5**:分域验收(MATH ≥ +60%,prose 回退零损失;域开关)

## 五、风险

1. selector candidates/unary 来源未最终定谳(需读 worker build_lattice
   调用点;若依赖 target lm_head top-k 则头部成本回升 —— 读码定谳)。
2. 分组卷积核数值(T=8 小窗 + 位置掩码;对拍锚 sglang)。
3. 3.85GB BF16 草稿显存(24G 贴顶:20.7 目标 + 3.85 + KV… → W4A16 版
   1.1GB 必选,z-lab/Qwen3.8-27B-DFlash2-W4A16 在库,marlin 直通)。
4. sliding_window 2048:草稿 attention 的窗语义(memory 窗 or draft 窗)待核。

---

## 六、DF-1 施工实录(2026-10-07 当日完工)

### 6.1 语义定谳(读码钉死,锚 = sglang dflash.py + dflash_worker_v2 + qwen3_next.py)

- **taps 语义**:capture = **层输出残差流**(post-attention+MLP 残差加、
  下一层 norm 前)。sglang `set_eagle3_layers_to_capture([id+1])` 的 +1 是
  其 capture 钩子挂在下一层 prepare_attn 的记账,净效果 = "层 id 输出后"
  —— owl 对应 = model 层循环里的 `xs`(零换算)。
- **encode 行集**:每轮写 verify 块行 [0..=m](锚 + 已接受草稿;bonus 留
  下轮作行 0)。锚行每轮重写(幂等,1 行税)。kv_proj_only 后 **k_norm +
  RoPE 都要过**(dflash_worker_v2 `_append_target_hidden_sequential`)。
- **propose 注意力**:ENCODER_ONLY(is_causal=false)+ sliding_window
  2048;8 行噪声块**互相全可见 + 全前缀**;自块 k/v 不入池(sglang 写
  overallocated 槽后由下轮 encode 覆写;owl 直读核输入,零写零竞态)。
- **conv 位置掩码**:t % block_size 用**批内行号**(propose 恒 8 行 →
  掩码恒 [0..7];extend 编码不走 conv(kv-only)—— conv 定形 T=8)。
- **unary 变换**:output_multiplier=1.0、无 softcap(config 缺省恒等)。
- **noise_embed_scale = 1.0**(target 无 get_dflash_noise_embedding_scale)。

### 6.2 落地面(与 §三 的差异)

- **fc 拆分 = 装载期转置**(`Weight::new_transposed("fc", hidden, fan)` →
  W^T [fan, hidden] 行主序,行块 i 连续 = W 列块 i)—— 逐 tap
  `slice_view` 直吃,Σ 5 GEMM,**免 concat 免虚拟键**。平声明 + reshape
  重释不可行:行主序 W 的列块跨步,零拷贝重释只给行块。
- **selector 核面**:`owl_topk16_f16`(逐行 top-16,(值降,索引升)全序,
  块内 16 轮归约)+ `owl_dflash_select_f16`(格打分 + 贪心 walk 单块融合,
  平局取小索引)。SSA 单输出契约 → 单缓冲双区(vals|idx、toks|scores)+
  slice_view 拆分(slice dtoh 回整父块 —— 测试收割须整块后 host 切)。
- **NC attention**:`owl_naive_attn_nc_f16` 自块 k/v **核输入直读** +
  前缀池 classic 寻址([0, prefix) 槽直排解算)。写-读同核有跨 block
  竞态(SSA 兄弟根不被依赖求值,依赖边仅 with_sig 外核可用)→ 直读是
  唯一正解。生产 FI 臂 = kNonCausal 变体(DF-4,K0-dual 写 + wr 依赖边)。
- **rope**:草稿自有 Rope 实例(θ 1e7,rotary 全维 128,262144)。
- **norm_rope plain 布局**:ATTN_NORM_ROPE 融合核对非门控布局天然支持
  (row_stride = 行全长,head_stride = hd —— k 链同款)。

### 6.3 验收(全绿)

- tiny 四金标(GPU,OWL_TEST_DEVICE):conv prepare/finish 逐式对拍、
  selector lattice+walk 对拍、fc 列块拆分对拍、NC 写-读往返(池 k/v 直读
  0/192 bad)。
- 27B 检查点(OWL_DFLASH2_DIR):81 键装载、project_memory 全行对拍
  (fc+hidden_norm)、selector 真几何冒烟(token ∈ [0, vocab))。
- 回归:engine 32/32、models 112/112、kernels 8、shared 16 全绿;
  cuda 仅剩 `marlin_graph_dispatch_probe` **存量失败**(干净树复现,
  "捕获内首次初始化需 warmup" —— 外核句柄预热问题,与本任务无关,另案)。
- 踩坑记录:LoaderCtx::default() dtype = F32(f16 链必须显式 dtype:F16);
  测试 host 参考的 pos 表切片偏移(自块行号 ≠ 位置号);logits 舍入网格
  (host 排序与 device 同格才能比对平局序)。

### 6.4 DF-2 待办(下一步)

- ForwardCtx hidden_tap(同 gdn_tap 机制)或 model `last_hidden_tapped`;
- verify 图加 5 个 tap 输出槽(T=depth+1 定形);
- 引擎:草稿池分配(5 层 × hkv8/hd128)、draft rope(θ1e7)、
  spec_mode DFlash2 分派(C7 扩展);
- prefill chunk 循环 harvest taps → encode(草稿 KV 物化);
- 轮末 encode(行 [0..=m])+ propose_block 接线(噪声块槽 = 前缀尾段
  + 8 scratch)。

### 6.5 DF-2/DF-3 接线实录(同日;管线全通,恒等门挂 W4A16)

**落地**(owl-engine):
- `model.tapped_hidden`(层输出残差流采集;tap = 树内别名,共享前向);
- verify 桶形图加 tap0..4 输出槽(T = depth+1 定形);
- spec_mode 四态(C7 扩展):`OWL_DFLASH2_DIR` + `OWL_SPEC_DEPTH`(≤7);
- StatePool `dflash_kvs`(5 层 × 草稿几何 8/128,独立页池);
- 草稿 rope(θ 1e7,rotary 128,262144)+ RunningEngine.draft_rope;
- scheduler kv_slots [u32;9](depth+1 填充;消费面自算物理槽);
- prefill encode:is_last/中间 chunk 双分支 concat 单根(hidden+taps 拼
  OPS_CONCAT 根 → scoped eval 竞技场回收照旧 → 块内切片出 taps)→
  `prefill_dflash_encode`(memory → 5 层 kv-only 写);
- 轮末:`dflash_encode`(verify taps 行 [0..=m] → 草稿 KV;bonus 留下轮
  作行 0)+ `dflash_propose`(噪声块 [bonus, MASK×7] @ fp..fp+7,eager;
  首轮 ≡ 常式,pos = fed);
- prefill_extend MTP 早退门(**必须先于 blocks_mtp 预留** —— mtp 账房
  非 mtp 模式 0 块,门后置 = 池耗尽假象,排查 1.5h 的教训);

**验证**:boot(draft 装载 + verify 图 T=8 捕获 + prefill encode 18 行
1.6-1.9ms)→ spec 轮调度(pos/spec/槽表全对)→ NC 核发射。回归
engine 33/33(models 112、kernels 8、shared 16 全绿)。

**阻塞(定谳)**:27B 恒等门 OOM —— **BF16 草稿 3.85G + 27B AWQ target
(embed/lm_head 未量化,+4.8G;总量 ~19.9G)> 24G 卡减图捕获/激活余量**。
MTP 门能过 = 草稿仅 0.4G。**W4A16 草稿(1.1G)是 DF-3 收口前置**
(§五.3 预告兑现),同时兑现后 propose 图化(DF-4)显存压力同步解除。

### 6.6 四源调研 + W4A16 收口(2026-10-07 续)

**四源调研**(用户纠偏:先调研再动手):
- **sglang**(语义母本,已锚定)/ **xinfer**(Rust,models/dflash.rs 1058 +
  speculative/dflash.rs 882):DraftLinear(Bf16/Quant-WNA16 双态)≈ owl
  plan 参数化;Quant = ct-GPTQ-marlin g128 = owl QuantPlan::W4A16 **同一条
  marlin 线**(路线互证);conv/selector 走 attention-rs 现成核
- **exllamav3-community**:conv 语义逐式一致(exl3 纯因果无块掩码,T=8
  定形下与 sglang t%8 掩码数学等价 —— 三源互证);residual 融合 finish
  = 可摘优化;EXL3 4bpw 草稿在库(r0b0tlab)但需 EXL3 通路,不如 W4A16 近
- **llama.cpp**(ik):DFlash2 不可用(AGENTS 在案),反面教材
- **代码级复用判定**:xinfer 冻结快照(A4 律)不可搬,owl w4a16.rs 虚拟
  派生 ≈ WNA16 同构,语义认领;fc 五拆 = owl 特有(消费面差异,正当)

**W4A16 接线**(syvai/Qwen3.8-27B-DFlash2-W4A16,ct g128 对称 1.28GB):
- 组件 `new_with_plan`(W4A16:attn/mlp/fc 量化,norms/conv 投影/
  codebooks 恒 F16 = 检查点 ignore 同构);fc 双形态 DraftFc::{F16T, Q(五拆)}
- w4a16.rs:`fc_{i}.qweight/.scales` 列块派生(packed 域列连续)+
  elem_len 三分支 + `dequant_linear`(对拍口);SafeTensorsSource.has_key
- specs Loadable 键尾规则按 want 后缀分派(qweight/scales/ws/marlin_ctmp
  不加 .weight,norm 加);load_27b_dflash2 源族自动探测(open_raw_index
  查 fc.weight_packed —— SafeTensorsSource 的 dtype 校验会拒 W4A16 文件)
- **验收**:27B W4A16 装载(205 键)+ fc 对拍(marlin vs 反量化参考)+
  selector 真码本冒烟全绿;**引擎 8 图全捕获**(显存解除!)
- 恒等门跑到:25 轮发射全通,AL=0.00 全拒 + 文本在 pos≈30(页界)后分歧
  —— **T=8 verify 图形状 bug 立案**(MTP T=4 门过;T=8 需查 GDN 记录/
  页界/8 行 prefill attention),未决
- 顺手修:drafts 截断 depth(SliceView dtoh 整父块 1799 f32 的坑二次踩)

### 6.7 恒等门分歧排查实录(二分链,2026-10-07)

**现象**:W4A16 门 AL=0.00 全拒 + 文本 pos≈28 后分歧(各 depth 1..7 同
分歧,43tok vs 42tok)。

**二分实验**(OWL_DFLASH_DUMB / OWL_DFLASH_NOENCODE 开关,exec.rs):
1. 真草稿 → 分歧
2. 哑草稿(跳过 draft 前向,同 host 表)→ **仍分歧** → 草稿值洗清
3. 哑草稿 + 跳过 dflash_encode → **仍分歧** → encode 洗清
4. 哑草稿零副作用(② 纯 host vec,不调 propose)→ **仍分歧** →
   propose 调用洗清

**嫌疑收敛**(与 MTP Dumb 门仅剩差异):① taps 版 verify 图
(tapped_hidden + 5 输出槽)② prefill concat 单根(hidden+taps)。
两者数学上应恒等(纯标注/纯拼接),分歧 = 实现 bug(嫌疑:CSE/tag 引
导的发射序变化、输出槽布局、concat 块偏移),**已立案未决**。
诊断开关保留(OWL_DFLASH_DUMB/NOENCODE);dflash_propose 签名收窄为
直接返回 Vec<u32>(观测面块弃用)。

### 6.8 concat 参数实锤 + baseline 非确定性(同日三)

- **concat 参数 bug 实锤修复**(用户提示「算子参数又传错」命中):prefill
  concat 根 r 参传了 t·dd(应 t)→ rd = t·dd² → in 恒 0 → taps = hidden
  越界复制(全部 AL=0 的直接原因);两处已修
- **新现象:baseline 非确定性** —— 同一无 spec 引擎三次运行三种输出
  (18tok 空 / 39tok / 42tok,均为合理文本)!此前所有「分歧」判定失效
  (比较基线自身不稳)。嫌疑:测试实例间显存/状态残留(上实例 server
  线程未退净)、异步发射竞态、或未初始化读 —— MTP 门当时过 = 当时稳定,
  dflash boot 路径(8 图 + 草稿池 + capture slab 扩容)改变了实例残留形貌
- **下一步**:① baseline 稳定性先行(引擎 drop 显存栅栏 / gate 测试进程
  隔离)② baseline 稳定后重跑四象限(±DUMB × ±NOENCODE)③ selector/
  propose 数值对拍(already 绿于组件层)④ 恒等门收口

### 6.9 稳定基线 + 轮数累积型 bug 定谳(同日四)

- **基建修复实锤**:gate 测试实例收尾屏障(drop + sleep 3s;server 线程
  退出 + CUDA ctx 析构异步,下一实例 17G 分配与释放赛跑)→ baseline
  稳定(两次 42tok 逐位一致)
- **四象限重定谳**(稳定基线下):真草稿 / DUMB / DUMB+NOENCODE 全部
  43tok 同一分歧文本 → 草稿值/encode/propose 全洗清,**spec 轮结构本身**
  (与 MTP 门差异:taps 图 / concat prefill / Dumb 轮 27B 未验)
- **新定性**:修复 concat 后分歧呈**轮数累积型**——前 ~10 轮(≈28 tok)
  bonus 逐位正确,之后错;depth 1..7 无关(T=2 也错);depth=2 出重复
  垃圾文本(58tok)。嫌疑收敛:①全拒轮的 KV 残留×GDN fold 交互(重放
  行数/快照点)②verify 图 taps 收集的 GDN 记录槽污染 ③decode 回退臂
  的 kv_slots [u32;9] 尾零槽
- 下次首刀:原生 MTP 真草稿门跑 depth=7(隔离 taps/concat,Dumb 27B
  顺带补测)→ 四象限矩阵定位到单一变量

### 6.10 恒等门 42tok 一致 + L0 爆炸定位(同日五)

- **恒等门正确性收口** ✅:修 taps 版 verify 图 final-norm 缺失
  (tapped_hidden 曾返回 pre-norm hidden → tok = lm_head(pre-norm) 全错
  ——AL=0 + 文本分歧的总根因)+ topk 布局两段连续(SliceView 平铺偏移
  行内错位)+ encode 含 bonus 行(行 m+2;噪声前缀未初始化读)+ 探针
  脚手架(probe_roots 逐层/单遍 multi 收割)
- **恒等门 42tok 逐位一致**(真草稿,DUMB/NOENCODE 全关)—— 正确性 ✓
- **遗留定性:L0 输出爆炸** —— probe(单遍 multi 修正后):embed ✓、
  L0 出口和 5019(均值 |x|≈9.8,健康 ~1-3,×878)、L1+ 全 NaN(饱和)。
  AL=0 = 草稿与错位分布完全脱节。嫌疑:①conv 系数量纲(base/delta 装载
  或核语义)②NC attention 读池未初始化行 ③kernel_projection F16 溢出。
  探针伪影教训:独立 eval_ops 重执行副作用层(原地 fused_add 累加)=
  NaN 假象,必须单遍 multi(§testkit 重放语义警告实战兑现)
- 下次首刀:conv 量纲对拍(真实 base_kernel/delta 逐元素 vs sglang 公式,
  H=5120 真形状)→ 修 → AL 应跳出 0

### 6.11 NaN 定位到算子级(同日六,未决)

probe 修正(单遍 multi)后根序解码:root1 = conv_in = **5019 有限**(conv
✓)→ root2 = attn_raw = **NaN** → **NC attention 真形状产 NaN**(tiny 金标
绿,真实形状 hq32/hkv8/hd128/prefix19 首爆)。L0 出口 5019 = NaN 下游
饱和非爆炸(×878 判定撤销)。候选:①池未初始化行(轮 1 的 bonus 行
时序)②score 溢出路径 ③NC 核真形状参数(page/x/头数换算)。探针基建
(probe_roots 逐层 + 单遍 multi)保留,下次直接 harvest NC 的 q/k 输入
定谳。decode 现值:真草稿 24.4 tok/s(AL=0),DUMB 38.1,baseline 33.1。

### 6.12 NaN 定谳:encode 链产出(同日七,未决)

- **NC 核洗清**:草稿池 K 行 0 = NaN(encode 输出本身坏)——NaN 在
  encode 链:memory(fc marlin W4A16)→ k_proj(**marlin M=19 小批量**)
  → k_norm+rope → K0 写。NC attention/conv/topk/select 全洗清
- 嫌疑排序:①**marlin GEMM M=19**(全仓 marlin 只验过 cyankiwi AWQ
  M=1-4 与 0.8B;M 非 16 倍数的行掩码/越界病)②memory 真实 T 行
  (冒烟只测单行 [1,fan])③k_norm_rope 真形状
- 四源调研再确认:xinfer 草稿 attention **不读池**(ctx k/v 现算 cat 后
  candle 标准算子一次前向;k_norm 在 cat 后整块;f32 norm)—— 若 marlin
  M 病难修,可摘此路线(前缀 k/v 每轮现算,免池免写,代价 = 每轮重算
  k/v GEMM)
- 下次首刀:probe memory(encode 前)二分 → marlin M=19 复现小测 →
  修行掩码或摘 xinfer 现算路线

### 6.13 golden 对拍定位:首个发散 = L1 fused_add(同日八)

- **golden 基建落成**(用户三指示兑现):tools/dflash2_golden.py(torch
  f32,真实 W4A16 反量化,sglang 逐式;随机公式输入隔离 target)→
  testdata/dflash2_golden.safetensors 逐层激活;owl 侧
  gpu_dflash2_golden_matches(marlin 同权重同输入)逐层对拍
- **结果**:L0_conv_in / attn_raw / attn_fin / mlp_out **全部对拍通过**
  (marlin W4A16 / conv / NC attention / mlp 在真实权重+真实形状下数值
  正确!)→ **首个发散 = L1_conv_in(bad 40592/40960)**
- L0 input_ln = plain rmsnorm(对);L1 input_ln = **LN_FUSED_ADD_RMSNORM
  (fused_add 路径,首个 fused 用户)→ 定位收敛到该核的参数/语义**
  (下一刀:读 fused.cu 的 fused_add_rmsnorm 参数序 + 与 decoder.rs
  用法逐参对照;金标侧可直接推 res 流期望值)
- 恒等门 AL=0 的主嫌疑就此收敛:草稿前向从 L1 起 NaN/错位 → drafts
  全废。修掉这一个算子,AL 应跳出 0

### 6.14 定谳翻案:模型全链无罪;三层取证伪影 + f16 溢出真凶(同日九)

- **"L1 fused_add 发散"系三层测试伪影叠加,核/模型无罪**:
  1. **逐根 harvest 重执行污染**——golden 对拍循环每根 probe 独立
     `eval_ops`(memo 每调用新建)→ 整棵子图重跑 → 原地 fused_add 把
     残差流反复累加进共享 embed 缓冲:L1 起全错 + "embed 终态 NaN" 全为
     取证伪影(6.11/6.12 的 NaN 定位同源)。修 = 单遍
     `eval_ops_multi`(共享 memo,CSE 去重)+ 从返回块直读(dtoh)
  2. **手写 safetensors 扫描器被 `__metadata__` 吞键**——前 6 键解析错
     位,对拍配对全乱。修 = `safetensors::SafeTensors::deserialize`
  3. **logits 对拍切片错位**——块为 [7, 248320] 行主序,前 112 元素 =
     行 0 前 112 列,不能与 golden [7,16] 拍平比;修 = 按 `r·V+c` 取头
- **修正后 golden 全绿**:25/25 探针(conv_in 残 48-74/40960 = f16 累积
  漂移噪声)+ hidden 全对 + logits_head 0/112(相对容差 4e-3,f16 ulp
  在 |logit|~9e3 处 = 8 ≫ 绝对差)+ **drafts 7/7**
- **golden drafts 键重生成**:原生成器用 f32 hidden 走 walk,而 f16 引擎
  下 logits 出现大量精确平局(topv 全 9264.0),平局取序崩塌——f32 参考
  对 f16 引擎不可达。修 = 生成器 walk 前 `hid.half().float()`,存量文件
  已按同式原地重写(f16 walk 与 owl 输出逐位一致)
- **引擎 AL=0 真凶(第一层):f16 溢出**——真实 target taps(元素
  ~7-50,深层残差流)× fc(权重量级 ~0.5)→ fc 输出 ~5×10⁴,f16 加链
  47456+16168+… → INF → memory NaN → 草稿池前缀 NaN → propose 塌缩 →
  drafts 全 248320 哨兵。golden 测试 taps ~0.7 从不触发(单测全绿引擎
  崩的原因)。**修 = project_memory 预乘 2⁻⁸**(f16 纯指数移位零舍入损
  失;rms 对行尺度不变 → norm 后语义严格等价;sglang bf16 无此问题)。
  修后 memory 有限、drafts 出真 token
- **同日十(未决):drafts 仍退化(" Hick"×7),AL 仍 0**——修后引擎
  marlin fc 输出有**结构性缺列**(row0 前 16 列精确零,真实 GEMM 统计
  上不可能;同输入在 models 测试进程逐位正确,设备无关/device 0/1 双验,
  env 无关,输入形态无关<视图/OPS_NARROW 物化/from_host 三态同象>,
  ws 尺寸/选型 tk=128 tn=64 m_blocks=2 与测试全同)。测试进程多 marlin
  连发全对 → **引擎进程态专属的 plain-GEMM_W4A16 写缺**;target AWQ
  (kU4)同进程正确(文本 = baseline)。探针链已证:fc 段量级 fc0=255/
  fc1=5224/fc2=16168/fc3=47456/fc4=86(块间悬殊,可疑),acc 行0 全零
  而它行 INF(未缩放对照)。诊断口 debug_fc0/debug_acc 保留;下次首刀
  = 引擎进程内最小复现(纯 fc_0 marlin 双连发对拍 host 参考)或
  compute-sanitizer 直接照 marlin

### 6.15 同日十二:权重读取双重缺陷定谳(装载器 BF16 误读 + 生成器 bf2f 值转换)

- **z-lab 源真值相关性检验**(决定性):syvai W4A16 的 fc 权重反量化对
  z-lab BF16 源做相关 —— **F16 读 = 0.9774(正确)/ BF16 读 = 0.0163
  (噪声)**;|w|max 20.875 vs 20.90 逐位吻合
- **缺陷①(owl 装载器)**:w4a16.rs 全部 scale 读取走 `bf16_par`
  (BF16 位型),而 syvai W4A16 的 weight_scale = **F16 标注 + F16 字节**
  → 草稿全部 W4A16 权重(fc + 5 层 qkvo/mlp)= 噪声 → 草稿前向整体废
  → drafts = " Hick"×7 语境盲 junk → AL=0。**修 = `f16_par` +
  `scales_par(dtype)` 按 RawEntry 声明分派**(dequant_linear /
  fc_block_bytes / build 三处;awq.rs 的 cyankiwi 线不动,BF16 读法在那
  边是对的)。修后 m-sweep:五块 fc 输出零元 0/102400(原 99.9% 零)
- **缺陷②(golden 生成器)**:同族 —— bf2f-on-F16-数组 = astype 值转换
  → scales ≈ 0 → **python 参考的层激活全零**(golden 文件 L0_attn_raw/
  attn_fin/mlp_out 全零已实证)→ 此前 golden"全绿"= **零比零空转**
  (fc/memory/层路径从未被真实验证;conv_in/hidden = embed 链非空转才
  真)。修 = dequant 按 dtype 分派(F16 值读);golden 文件已重生成,
  现含真实激活(含 inf:见下)
- **新暴露的真问题(下一案)**:真实权重下草稿激活幅度超 f16 —— 再生
  成 golden 的 L0_mlp_out 已含 inf(python f32 前向 >65504,z-lab 权重
  带 |w|~200 离群;sglang BF16 范围 3e38 无恙)。owl f16 管线需**全局
  权重预缩放**(2⁻⁸ 幂次,经 rmsnorm/argmax 尺度不变结点严格等价;
  embed 与 target 共表需图内 mul)—— 未实施,当前引擎 drafts 仍退化
- 引擎 bisect 探针(OWL_DFLASH_PROBE)进 dflash_encode:①独立 eval ②
  multi 根 ③池非零计,三段读回。注意:本轮清理曾误删
  prefill_dflash_encode 的 memory/encode 主体,已恢复(函数曾变 no-op)
- 回归:models 114 ✓ engine 33 ✓(golden 对拍现为诊断打印,无断言)

### 6.16 同日十三:BF16 激活 W4A16 marlin 变体落地(对齐 sglang 方案,用户拍板"他们用什么我们用什么")

- **决策**:DFlash2 真实权重激活超 f16(L0_mlp_out >65504,见 6.15),
  sglang = BF16 全程 → owl 对齐 = **草稿路径 BF16 化**(而非权重预缩放,
  预缩放在 conv 双线性项/silu 上有等价性缺口)
- **marlin 侧(本段完成,BF16 parity 全绿)**:
  - `sm80_kernel_bfloat16_u4b8_bfloat16.cu` 新实例文件(f16 版 sed 生成,
    a/c/s 全 BF16;模板体系原生支持,`MarlinScalarType<kBFloat16>` 全套
    traits 在位)
  - kernel_selector.h 追加 75 对分支(1:1 对应实例;**教训:生成脚本的
    替换必须覆盖全部四个模板参** —— 首版只换 a/b,c/s 残留 kFloat16 →
    混合签名未定义引用,链接错误绕了三圈)
  - marlin_host.cu 新增 `marlin_gemm_v2_bf16_ffi`(is_bf16=true);Rust
    `GEMM_W4A16_BF16` 核名 + `gemm_v2_raw_bf16` + foreign 分派臂 +
    Linear::forward 按激活 dtype 选臂(BF16 → BF16 核)
  - parity:bf16 三形状(m=20/8/64 × fc 及 512 形状)噪声比 0.13-0.26%
    全绿;**marlin_parity 7/7**(f16 + AWQ + BF16 全过)
- **构建系统两坑(记档)**:①`--features marlin` 缺失时 build_marlin()
  整段跳过,MARLIN_FORCE_BUILD 静默失效;②.h 文件不在 rerun-if-changed
  → selector 改动不触发重编,host .o 落后一代。已修:build.rs 补 .h 追踪
- **prebuilt .a 事故与恢复**:ar 替换/组装多轮中误删 marlin_cuda_kernel.o
  (上游弃用宿主,内嵌全量 selector,含未实例化的 bf16 分支引用)又因
  /tmp/marlin_host.o(20:53 stale)与 /tmp/marlin_host_fresh.o(21:18 新)
  两文件混用反复回退 —— 最终态:prebuilt .a = 上游全成员(含
  marlin_cuda_kernel.o)+ marlin_host.o(全 BF16 selector 编译)+
  bf16inst.o(BF16 实例);member 名以 ar r 原名替换为准
- **剩余(BF16 化第二期)**:草案路径原生核 bf16 变体(rmsnorm/fused_add/
  grouped conv/NC attention/norm_rope/K0 write/topk16/cast ×2)+ 池
  dtype BF16 + cast 边界(embed f16→bf16、lm_head 前 bf16→f16 保 cublas/
  topk 不变)+ selector 投影 cast。工程量:核变体机械 + 装配点清点

### 6.17 同日十四:BF16 化第二期完工(草稿路径全链 BF16,sglang 精确对齐)

- **装载面定谳(本段起点)**:W4A16 检查点的 passthrough 权重(norms ×12 /
  base_kernel ×10 / kernel_projection ×10 / hidden_projection / codebooks ×2)
  **原生 BF16 非料想 F16** —— sglang 对齐即免转换直载,比 F16 中转多保 3 位
  尾数且省一次全量转换;装载器 BF16 want 臂补齐(w4a16 passthrough 直拷
  + safetensors 源 BF16→BF16 直拷/F16→BF16 值转换;WeightSource 默认臂补
  BF16,testkit HashMap 源可直喂)
- **核变体(9 个新核,机械变体 + 两处签名混合)**:
  - 全 bf16 同型:owl_dflash_conv_bf16 / owl_naive_attn_nc_bf16 /
    owl_silu_and_mul_bf16 / owl_fused_add_rmsnorm_bf16(gamma 原生 bf16)/
    vllm_reshape_and_cache_bf16(K0 草稿池)/ owl_add_bf16 / owl_mul_bf16 /
    owl_rmsnorm_bf16(ops_pair 宏桥免费预留兑现)
  - 混合签名×2:owl_norm_rope_bf16(x/w/out bf16,**cos/sin 保持 f16** ——
    rope 表与 target 共享免双表)/ owl_dflash_select_bf16(proj/a_tab/b_tab
    bf16,cand/unary/anchor/out 保持 f32 契约 5)
  - 铸边界:owl_cast_f16_bf16 / owl_cast_bf16_f16(embed 查表后入草稿 /
    lm_head 前出草稿;**topk16 无 bf16 变体** —— logits 恒 f16,cast 保
    cublas/topk 链不动)
- **分派链**:driver::DType 增 BF16(7 处 GDN/naive/sigmoid 臂同步
  `BF16 | U32 => unimplemented!` 守门);norm_rope/fused_add_rmsnorm/
  silu_and_mul/k0_write 四臂按 dt 分派;ddt()/dname() 放行 BF16;
  registry +14 条目
- **cublas BF16**:GEMM_BF16 外来核(CUDA_R_16BF ×3 + COMPUTE_32F,
  映射逐式同 f16 臂);handle_cublas_gemm 参数化(bf16: bool)零重复;
  eval Matmul/MatmulNt BF16 路由(消费点 = kernel_projection / hidden_
  projection 两个非量化投影)
- **模型层 dtype 线程化**:DFlash2Draft 新增 `dt` 字段全链穿透(GroupedConv/
  DfAttn/DfLayer/CandidateSelector);`new_with_plan_dt` 全参构造(存量
  new/new_with_plan 保持 F16 = 测试面零迁移);base_kernel/codebooks 改
  `Weight::new_typed(.., dt)`;ensure_dt() 铸边界助手(dtype 同直通)
- **project_memory 双臂**:BF16 = **免 2⁻⁸ 预缩放**(bf16 指数域同 f32,
  溢出护栏仅 f16 面)+ taps owl_cast_f16_bf16 铸入;F16 存量臂原样保留
- **引擎侧**:草稿池 dtype BF16 独立于 dims.dtype(state.rs;同 2B/elem,
  page/x 几何不变);探针解析 dtype 感知(draftK 行检查 + probe roots);
  load_27b_dflash2 → LoaderCtx.dtype = BF16(双形态:W4A16 + BF16 原生)
- **验收**:bf16 冒烟三件套(conv bf16 host 对拍 + cast 往返零损 + select
  bf16 walk 同序,新 `gpu_dflash_bf16_smoke`)全绿;全量回归 models 115 /
  kernels 8 / engine 33 / marlin_parity 7(F16 套件零迁移零回归);
  测试坑两条记档:①bf16 输出量子在 mag~16 处 = 0.0625,host 参考须
  落 bf16 网格后对拍(atol 0.15)②drafts SliceView dtoh 回读整父块
  (f16 selector 同坑,测试侧手动 eval+dtoh 取前 rows)
- **待验(下一步)**:真检查点恒等门重跑 —— 预期 drafts 逃离 " Hick" 退化、
  AL 从 0 跳升(目标 ~2.4,吞吐 ~90 tok/s);若仍低 → 分域 AL 验收 +
  taps 链复核(6.16 遗留观察)

### 6.18 同日十五:AL=0 三连环破案 + DFlash2 首次真接受(BF16 二期收口)

BF16 化后首跑恒等门:恒等绿(42tok 逐位)但 **AL 仍 0.00**。三连环排查
(每环都由 dump+对拍实锤,非推测):

- **①propose 自块槽表错位(真 bug,已修)**:dflash_propose 槽表误填
  0..8,每轮把池位置 0..7 的 memory 前缀 K/V 磨成噪声块残写(永不重编)
  → 窗口头部永久污染。修 = 物理槽 pos..pos+8(sglang
  assign_extend_cache_locs [prefix_len, prefix_len+block) 同位)。伴生雷:
  propose 跑在 fp = 下轮 fed 位,**先于**下轮 scheduler 的 ensure_for_len
  → 跨页越界 panic;且 propose 内 ensure 必须伴 **write_bt 同步律**
  (否则 scheduler grew=false 跳过重写,烘焙块表停代 → gate9 文本分歧 /
  gate10 空轮两案真凶)
- **②多 chunk prefill dtoh 契约雷(真 bug,已修)**:非末 chunk 路径
  dtoh(b, t·hidden) 是 MTP 时代遗留 —— DFlash2 下 b = concat 根
  [(n_tp+1)t, hidden],整父块契约炸(长城 prompt 单 chunk 从未触发;
  数学 prompt 双 chunk 当场曝)。修 = 等大读全块
- **③draft 全系 norms 语义错(真根因,已修)**:sglang dflash.py 的
  draft 六种 norms(q/k/input/post/hidden_norm/norm)**全部 = plain
  RMSNorm ×w**(「matching HF Qwen3」;无 offset);owl 全误用 ×(1+w)
  —— rmsnorm 尺度不变性让逐层前向表面正常(层层重新归一),但
  **注意力温度(q·k 二次放大)与 selector 码本打分(非尺度不变处)**
  被系统性毒化 → drafts = 语境盲 junk → AL=0。golden 双方同错所以
  自洽通过(共享语义误读的教训:对拍只能证内部一致性,不能证对 sglang
  的语义正确性)。修 = 全部改 RmsNorm::new + norm_rope w_off 随 norm
  (曾硬编码 1);golden 生成器同步改 + testdata 重生成

**破案方法论(逐环 dump+对拍,可复用)**:
1. 对齐探针:verify 行 argmax(ids) vs drafts 逐位对照(附近命中 = 位移
   bug;全散 = 质量问题)—— 排除对齐类;
2. sglang 地面真值:sglang-dflash-patch 分支打 env 门控 dump 补丁
   (SGLANG_DFLASH_DUMP;注意 .item() 图捕获期非法,须
   is_current_stream_capturing() 守卫),同 prompt 对拍 —— sglang
   hid1_sum=4.6e3 / drafts=[13,198,248069,…] 常见 token,owl drafts
   junk → 确认.owl 侧问题;
3. python 参照逐阶段 bisect:memory 引擎 dump → python 反量化重算
   (corr 0.99999 ✓)→ encode K 池 dump 解交织对拍(corr 0.9944 ✓)
   → 31 探针根逐根独立 eval 落盘 + python 同输入逐阶段对拍 →
   **首个发散 = L0_attn_raw**(conv_in 0.4% 吻合)→ 收敛到 norms;
4. 探针工程坑:multi eval 后段会把早根块回收复用 → dtoh 读陈旧块
   (root0 曾读出 1e6 级垃圾"爆炸"假象)—— 逐根独立 eval 才可信;
   切片视图 dtoh 整父块契约再次命中(探针自身也要守)。

**修复后验收(2026-10-08)**:
- 恒等门:prose(长城 42tok)**逐位一致** + math(123×456 73tok)
  **逐位一致**(双域全绿);
- **AL:0.00 → prose 1.08 / math 2.75**(xinfer 域定律参照:prose 1.13 /
  math 4.49 —— prose 达标,math 方向正确尚有差距);
- 吞吐:math 域 30.5 vs baseline 29.4 tok/s(持平)—— 净增益待
  DF-4 propose 图化(现 eager propose ~26ms 吃掉 AL 收益);
- 回归:models 115 / kernels 8 / engine 33 全绿;
- 附带修复:pw 误杀 3080 对现役 vLLM(pkill -f qwen-server 两引擎同名,
  **第 N 次验证 kill 循环必须 PID 列表**,后已按标准姿势重启);
  sglang dump 补丁留在工作树未提交(家规)。

**余下(DF-4)**:propose 图化(eager → 图回放,-20ms 级)+ fold 批量
+ 分域净增益验收(std/MATH × on/off;净增转正线 = AL×(1/轮成本) 折算)。

### 6.19 同日十五续:DF-4 propose 单图落地 —— 数学域 +103% 净增益

- **轮账(profile 实测,AL=2.9)**:verify 30.5ms(图回放;target 前向
  23ms 带宽墙附近)+ propose 20.5ms(eager ~130 发射的 actor 往返税)
  + fold/snap 1.5ms;基线 decode 29-32 tok/s
- **DFlash2 propose 单图(E5-DF4;比 MTP 桶图更优 —— 固定几何单图
  覆盖,无需 m 桶)**:
  - 前置改造:NC 核 prefix_len 去烘焙(`owl_naive_attn_nc_{f16,bf16}`
    签名 `i32 prefix_len` → `T kv_len_ptr`,核内 prefix = kv_len − T;
    契约 5 f32 过线同 slots/pos)—— fp 每轮变,宿主标量烘图必错;
  - **encode 行数钉 8**(verify 块全行):多编行(被拒 draft 行)下轮
    被自块写/重编覆盖,幂等安全;附带修 eager 路 m=7 时 rows=m+2=9
    超 taps 块 8 行的越界读(min(8));
  - 图 = memory(5×fc marlin)→ 5 层 encode_kv(K0 写,输出槽触发执行)
    → 噪声块 5 层 → lm_head cublas → topk16 → select;输入槽 7 个
    (tokens/anchor/enc_pos/enc_slots/prop_pos/prop_slots/kv_len),
    烘焙叶 = verify taps 槽 + 草稿池叶 + bt 叶;输出 = drafts(select
    缓冲 offset 0,形状 = 整缓冲遵守 dtoh 整父块契约)+ encode 哑根
    e0..e4(BF16 [1],触发 K0 写入 DAG);
  - 回退链:捕获降级/OWL_DFLASH_EAGER=1 → eager 路(验收一致);
- **验收(math 域,AL 2.90)**:**59.6 tok/s vs baseline 29.4(+103%)**;
  prose 域(AL 0.79)**42.6 vs ~32(+29%)**;双域恒等门逐位一致;
  eager 回退验收一致(AL 1.08);回归 models 115 / kernels 8 / engine 33 /
  marlin_parity 7 全绿;
- **分相细账(dflash-prof)**:graph_launch = **150µs**(回放本身);
  read dtoh 同步等流排空 = 18ms(verify GPU 尾部 + propose 图 GPU);
  下一刀需要 nsys 级深挖:①propose 图 GPU 构成(理论 ~5ms:草稿权重
  1.1GB 带宽 1.3ms + lm_head 2.4GB 2.8ms + 小 GEMM,实测 ~18ms 差 3×);
  ②verify 30ms 里的 host 交织;理论轮下限 ~36ms → math ~108 tok/s。
  注:60 tok/s = 超越 vLLM decode 基线(43)+ xinfer 157 参照同量级。

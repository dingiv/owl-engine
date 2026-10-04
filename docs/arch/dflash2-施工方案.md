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

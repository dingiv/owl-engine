# E5 投机解码 —— 四家调研与立项设计

> 2026-10-04。触发:decode 45.6 t/s 达标(用户令:E5 解锁线)。方法论:REQ-DESIGN
> 先抄后改;四家侦察 = xinfer(自家史,+33.8% 实绩)/ vLLM / sglang(dflash 上游
> + 工作区 fc-quant 补丁)/ ninfer-3090(MTP3+ReplaySSM,单卡榜 85.34)。
> 状态:调研卷;施工里程碑见 §六。

## 一、四家机制对照

| 维度 | xinfer(冻结档) | vLLM | sglang(v2 worker) | ninfer-3090 |
|---|---|---|---|---|
| 草稿类型 | DFlash2(外部 5 层草稿) | MTP/EAGLE/ngram/DFlash(**上游也有 dflash.py**)| DFlash2/DSpark/EAGLE/frozen-KV-MTP 全家 | **MTP3**(模型内嵌 MTP 头) |
| 草稿结构 | 5 层 hidden 5120,40H/8KV,cross-attn over **投影后目标层 hidden**([5,19,33,47,61] 层) | fc(hidden*2→hidden)+ 1 个完整 decoder layer(共享 embed/lm_head) | 同 xinfer 架构(DFlash2 同源);**draft KV = 物化的目标 hidden** | MTP 层 ×3 步自回归 |
| 草稿上下文 | **rolling window ≤128 行逐步全重算**(xinfer 自评"正解 = KV 物化") | — | **增量物化**(_append_target_hidden_to_draft_kv;compact 滑窗或共享映射) | — |
| verify | target 8 行 mini-prefill(pos = len-1 起,**anchor 复用**:行 0 重打分上步 bonus,省独立 64 层 anchor 前向) | target_verify 批处理 | TARGET_VERIFY 前向模式 + spec-v2 attn backends | verify_ids=[anchor,drafts×k,pad] 设备侧构建 |
| 接受 | greedy 逐位(argmax 对拍)+ continuation | 含 rejection sampling(随机域) | greedy/sampling 双路(greedy_mask 进草稿图) | **单核内 device 拒绝采样**(一热 q,truncated target 残差重采样)|
| GDN 状态回滚 | **快照**(mtp_rollback_mamba_at)| —(qwen3_next spec 未通单卡)| 三档:ReplaySSM fold(record+fold)/ ring write / 逐 draft 快照回退 | **ReplaySSM:record per-token (k,v,g) + fold 重放**(免快照拷贝)|
| 图化 | verify 定形图(mtp_capturer 按 verify_len)+ **draft eager(未入图 = 最大挂账)** | piecewise | 草稿图含采样路径(greedy_mask 静态槽) | 全轮图化 |
| 每步预算 | draft 15.4ms + verify 29.4ms + host 0.15ms | — | — | MTP3 = 85.34 t/s 实绩 |

## 二、DFlash2 草稿模型解剖(z-lab/Qwen3.8-27B-DFlash2,owl 可直载)

```
DFlash2DraftModel:  5 层,hidden 5120,40H/8KV(GQA),mlp 17408,≈1.9B bf16
输入:  noise_embedding = target.embed([anchor, MASK×7])     # 8 行查询
       memory        = kernel_projection(concat(target 层 [5,19,33,47,61] hidden))  # 上下文
forward: 每层 hidden = layer(noise, memory, cos/sin)(cross-attn:noise=Q,memory=KV)
输出:  末 7 行 hidden → target.lm_head → select_candidates(top-k 路径,锚定 anchor)
config: block_size=8(1 锚 + 7 草稿),mask_token_id=248070
```

量化档在库:z-lab BF16 / syvai W4A16(compressed-tensors,sglang 工作区分支
2c396f5 已打通其 fc 量化)/ lued W8 / GGUF。**W4A16 草稿 ≈1.1GB,owl marlin
直通,24G 贴顶卡可容**(boot 19.8 + draft 1.1 + 上下文缓冲 ~7MB)。

## 三、实测数据(xinfer p5 终卷,27B DFlash2,greedy)

### 3.1 分域接受率(最重要的一张表)

| 域 | per-token accept | 真实 AL(acc×7+1) | decode t/s | vs eager |
|---|---:|---:|---:|---:|
| MATH(算式 CoT) | **49.9%** | **4.49** | 139.6 | **+183%** |
| AGENT(工具调用) | 16.7% | 2.17 | 97.0 | +97% |
| CODE | 24.8% | 2.74 | 58.2 | +18% |
| **PROSE(散文/随机词)** | **1.9%** | **1.13** | **27.3** | **−45%** |

**⚠️ 域定律:草稿增益强域依赖;随机词池/开放散文 = 负收益**(zero-accept 步占
91%,AL 1.13 = 每步只赚 bonus)。owl 基准 std 口径 = 随机词池 → **期望净负**;
E5 的验收必须分域(llm_speedtest 已有 `--prompt-domain math|prose`)+ C7 护栏
(净增益 >0 才默认开,sglang adaptive_spec_params 的静态版)。

### 3.2 步成本账(xinfer 双卡时代,anchor 复用后)

df_step 45.1ms = df_draft **15.4ms(34%,eager candle 慢路径 ~4× 于地板)**
+ df_verify 29.4ms(65%)+ host 0.15ms。draft 带宽地板 ≈ 4ms(1.9B bf16)
→ **草稿入图/快路径 = 步成本 -13ms**,是 xinfer 未完成的头号挂账。owl 从
第一天就图化草稿(W4A16 marlin + 定形图),直接绕过此税。

## 四、GDN 状态回滚三派(27B 混合架构的特有难题)

verify 的 8 行会推进 GDN 递推状态;部分接受时须回滚。三派:

1. **快照**(xinfer 用之;owl E2c 基建现成):verify 前拍 48 层状态
   (151MB D2D ≈ 0.3ms),回滚 = restore。**owl E5-D1 选此**:最低风险,
   基建已有(capture_snap/restore_snap 按 slot)。
2. **record+fold**(ninfer ReplaySSM,sglang fold 档):verify 期间逐 token
   记录 (k,v,g,conv 输入),回滚 = 从记录重放递推(免 151MB 拷贝,一次
   确定性重算)。**E5-D2 优化项**。
3. **ring write**(sglang ring 档):草稿期直写环形记录,commit 时 fold。
   与 2 同族,投机深水区,不立项。

## 五、轮结构(共同骨架,四家一致)

```
每轮:
  1. 快照 GDN 48 层状态( E2c 通道)
  2. draft:投影目标 hidden 窗口 + [anchor, MASK×7] → 草稿 7 token
  3. verify:target 8 行 mini-prefill(pos = len-1 起,anchor 复用;
     KV 写 len-1..len+6 槽,GDN 推进)—— owl 即现有 prefill 通道 T=8
  4. accept:greedy 逐位;continuation = 边界行 argmax
  5. 部分接受 → GDN restore;KV 槽自然覆写(kv_len 回退 accepted+1)
  6. 发射 accepted + bonus;verify 捕获的 5 层 hidden 投影后追加进窗口
```

## 六、owl E5 施工设计

### D1 草稿模型装载 + 独立对拍
- `crates/models/src/models/dflash.rs`:DFlash2DraftModel(5 层 GQA
  cross-attn + mlp + norm;复用 owl attention/MLP 组件;memory = 投影后
  target hidden;Q=embed(anchor+MASK))
- loader:z-lab BF16(safetensors 直读)先行;syvai W4A16(marlin)第二步
- kernel_projection(Concat5×5120→5120 GEMM)入库;金标 = 官方?无 →
  **对拍锚 = xinfer candle 实现同输入同输出**(冻结档只读可执行)
- 验收:candidates 与 xinfer 逐位一致(同权重同输入)

### D2 目标侧 hidden tap + 窗口
- verify 前向的 SSA 树 tap 层 [5,19,33,47,61] 输出(BlockSlice 免费)
- concat → kernel_projection → 窗口缓冲(rolling ≤128 行/seq,持久块)
- 验收:窗口内容 = xinfer store_decode_hidden 语义

### D3 verify 通路 + accept + 快照回滚
- Scheduler 新 `StepAction::SpecRound`:快照 → draft → verify(prefill 通道
  T=8,pos=len-1)→ greedy accept → restore(部分接受)→ 发射
- KV 槽:verify_out = [len-1..len+6](覆写 bonus 槽幂等);kv_len 回退
- GDN:capture_snap/restore_snap 现成通道,SNAP_MAX≥1 已保证
- 验收:27B e2e 无 spec 文本 vs spec 文本逐位一致(greedy 下 spec 输出
  必须与无 spec 逐步等价 —— 接受是恒等的)

### D4 图化 + 性能
- verify 8 行定形图(与 decode 图并族,multi-root 已修);draft 图
  (W4A16 marlin,8 行 × 5 层,marlin m=8)
- 三段账 metrics(df_draft/df_verify/df_step + acceptance 直方图,REQ-DEC-04)
- 验收:MATH 域净增益 >0(C7);std/prose 域自动回退 v1(adaptive 静态版
  = 域开关 env,动态版挂 adaptive_spec_params 调研)

### D5 验收线
- MATH 域:相对 45.6 基线 ≥ +60%(保守;xinfer 同域 +183% 但其基线弱)
- prose/std:自动回退,零损失
- 27B 长程召回/文本金标不回归

## 七、MTP 路线(第二期,不并行)

27B FP8 检查点 `num_nextn_predict_layers: None` → HF 侧无 MTP 权重;
ninfer-v2/qwen3_8_27b.ninfer 包内含 MTP3 权重(85.34 实绩)→ 可提取转置。
MTP 草稿 = fc(hidden×2)+1 层 decoder,owl 结构齐备,装其即用。
**DFlash2 先行,MTP 为 DFlash 验证完管线后的第二草稿**(管线是共享的,
草稿可插拔 = sglang base_spec_worker 形态)。

## 八、风险

1. **域负收益**(§3.1):验收必须分域;std 随机词池预期净负 —— 立项书
   明确 E5 的成功标准是"MATH 域净增益 + prose 零损失",不是 std 涨分。
2. **贴顶显存**:19.8 + W4A16 草稿 1.1 + 窗口 7MB ≈ 21GB ✓;BF16 草稿
   3.8GB 需 OWL_POOL_TOKENS 降档。
3. **verify 与 FI 的交互**:8 行 chunk 走 FI 需 plan(小 qo);回退 paged
   prefill 亦可(T=8 的 prefill attention 成本可忽略)。
4. **快照成本**:151MB D2D ≈ 0.3ms/轮 —— 相对 AL 增益可忽略;record+fold
   列 D2。
5. **吞吐口径**:spec 的 t/s 仍按 usage completion_tokens(含 bonus)。

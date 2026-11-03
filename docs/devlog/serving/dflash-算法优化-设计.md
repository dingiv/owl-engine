# DFlash/DFlash2 投机解码：三家对齐与优化空间（设计）

> 2026-11-03 立案。问题入口：**owl 的 DFlash2 算法层还有没有优化空间？** 对照
> sglang 与 vLLM 的实现。
>
> 结论先行：**算法骨架三家已经收敛（selector lattice + 一次并行 draft + 链式
> verify + 精确拒绝采样），owl 的接受数学是教科书式的、没有错**；差异集中在
> 三处**工程取舍**：walk 在 host 还是 device、深度策略、以及"重叠调度"。
> 本文给一张优化点总表 + 便宜档的详细设计（本轮施工），中等档只留提要。
>
> 状态：**便宜档 P0-C1/C2/C3 已施工并过门**（见 §6）。施工中又查出一件更要紧的事：
> **双 3080 TP2（词表切分）档根本不走 q 路径**（`spec.q.none` 计数 = 16/16 轮）。
> 已把 C4（TP2 补 q）**实现并真机 A/B**：**中性**（AL 2.10 → 2.13、tok/s
> 69.9 → 70.2）⇒ **未合入**（结论与实现要点留在 §4.4，将来重测直接照抄）。
> **S0（条件格 vs 边际）已实测结案（§11）**：边缘分≈0 信息 ⇒ 树按边际造；
> 名次直方图给出树的真实天花板只有 **+15~20%**（不是论文的 2~3×）⇒ **真树暂不立项**，
> 优先级改到"每轮固定开销 9.5 ms"与"按 AL 切 spec/裸 decode"。
> **P1-⑤（深度双向阶梯）已实现并实测证伪**：verify 的行边际成本只有
> **0.57 ms/行**（每步被权重流带宽支配）⇒ "少喂几行"没有收益，**AL 才是唯一杠杆**，
> 详见 §8。阶梯判据保留（MTP 生效、DFlash2 待图族），生产档仍单档 d7。

## §0 一句话

owl 的 DFlash2 = 「草稿头一次前向出 select-16 候选格 + host 侧 walk + K+1 行链式
verify + `min(1,p/q)` 精确拒绝采样 + lookup 点质量融合 + 位置门/收益门自适应深度」。
**没有算法错误**；有 4 处可优化：①`hist` 污染（真 bug）②q 未走请求的采样变换
（白付拒绝）③walk 的 host 往返（关键路径同步）④深度只降不升（一次错判付费 10~20 轮）。

## §1 三家骨架（已收敛）

```
draft :  目标 hidden 作 context KV + [anchor, MASK×K] 一次前向
         → 每步 top-16 候选 id + 边分 scores[K,16,16]
         → selector lattice 条件 walk（按上一步**选中**的候选走）
verify:  K+1 行单次前向（anchor + d1..dK），链式（无树）
accept:  min(1, p(d)/q(d)) 接受；拒 ⇒ 从 max(p−q,0) 归一抽 correction
         全收 ⇒ 第 K 行抽 bonus
lookup:  n-gram 后缀召回 ⇒ q 改点质量，把草稿补到 K 之后
depth :  自适应（三家都有，判据不同）
```

**为什么 owl 的稀疏接受是精确的**（本轮复核，纠正一个曾经的怀疑）：
p 被限制在设备 top-32 候选上（`distribution_of_cands`，与裸 decode 同一套
temp→top-k→softmax→top-p 变换，截断质量**重均摊**而非重归一）；
q 限制在 16 候选上归一。对任意提案分布 q 与任意目标分布 p̃，拒绝采样
`min(1,p̃/q)` + 残差 `(p̃−q)+` 的输出分布**恒等于 p̃**（标准定理）。
残差在 p 的支撑上算**也不引入偏置**：因为 `min(p̃,q)=0` 在支撑之外
⇒ `Σ_{S}(p̃−q)+ = 1 − Σ_all min(p̃,q)`，与全词表口径相等。
⇒ 唯一的"口径差"是 **p̃ 本身被 top-32 截断**（见 C3），不是接受数学。

## §2 三家差异（决定优化方向）

| 维度 | owl | sglang | vLLM |
|---|---|---|---|
| lattice walk | **host**（DTOH `sel`+`topk_all` 两次读 → `DraftQ::from_scores`） | device（`_selector_lattice`+`sample_path`） | **device Triton 核**，直接写 tokens，并缓存用完的 q 行 |
| q 的变换 | softmax(边分/T) 归一（**自成一套**，不含请求 top-k/top-p） | `_dflash_top_k/top_p_renorm_prob` | 按请求 top-k/top-p 截断 q 后重归一，注释写明"mass outside support = guaranteed rejection" |
| q 的表达 | 稀疏 [K,16] + host `q_of`/one-hot | scatter 成稠密 [bs,γ,vocab] | device 缓存 q 行，verify 核按候选读 |
| 深度策略 | ①位置门 `depth_switch_ctx` ②收益门（EMA，**8 轮 warmup + 2 低窗确认**） | EMA(accept_len)+1 双向 + hysteresis（warmup 10 批 / 每 5 批更新） | **copy 检测**（lookup 连续两步饱和才升深，`sticky` 续期；短 ctx 下长块几乎免费） |
| 重叠调度 | 无（accept 的 host 决策在关键路径） | spec-v2 overlap（commit_lens / prefix-valid write / decoupled io） | 有（`next_num_draft_tokens` 用**一步陈旧**的决策；注释实测同步 `.item()` 值 5%） |
| 多请求批 | 与 `max_batch>1` 互斥 | 支持 + **draft KV 压实** | 支持（`sample_indices` 打包） |
| verify logits 调整 | rep_penalty + top-k/top-p（无 grammar/logit_bias） | rep penalty / **grammar mask** / logit_bias / custom processor | sampling_states 同族 |
| GDN 回滚 | ①`spec_fold_to` 记录折叠 ②fallback 重放 m+1 行 | hybrid 后端 + mamba align | mamba-align dense boundary（补丁） |

## §3 优化点总表

优先级 = 收益/成本。**「便宜」= 纯 host 逻辑、零设备面改动、可 CPU 门。**

| # | 项 | 现状 | 改法 | 预期收益 | 成本 | 风险 | 优先 |
|---|---|---|---|---|---|---|---|
| **C1** | `hist` 污染 | MTP/Dumb 采样档把 `live+1` 个抽样 token 全部推进惩罚历史，实际只提交 `m+1` 个 | 先判接受、再按 `s[..=m]` 入账 | 惩罚语义与裸 decode 一致（`rep_penalty=1.0` 时无感，开启后是真错） | 30 min | 极低 | **P0 ✅** |
| **C2** | q 未走请求采样变换 | q = softmax(边分/T) 归一；请求 top-k/top-p 只作用在 p 上 | q 改用 `sampler::distribution`（与 p **同一套**变换：temp→top-k→softmax→top-p→质量重均摊） | 强截断请求（top_k≤8 / top_p≤0.9）下不再提出必被拒的草稿 ⇒ AL 上升；同时消灭"两套采样规范" | 2 h | 低（q 可以是任意分布，精确性不受影响） | **P0 ✅** |
| **C3** | 口径不可见 | ①请求关掉截断时（top_k≥16 且 top_p≥1）spec 的 p 仍是 top-32 ⇒ 比裸 decode 尖；②q 被截掉多少质量无人知 | 记账 `spec.q.trunc`（截断步数）/`spec.q.drop_ppm`（丢弃质量 ppm）；关截断时一次性 WARN | 把"已知口径差"变成可观测，不再靠猜 | 1 h | 极低 | **P0 ✅** |
| ~~**C4**~~ | **TP2 档没有 q**（**已实测：中性，未合入** §4.4） | 词表切分档走 `dflash_select_merged`：selector 图只吐 drafts，**不吐 scores/候选** ⇒ `spec_q_host = None` ⇒ 接受退到"目标抽样==draft"路，草稿是 selector **argmax（贪心）** | 让 merged 收割面也出 `scores[K,16,16]` + `cand[K,16]`（两 rank 半表在 rank0 合并后同一份 host walk），复用 C2 的 `from_scores` | 让生产档拿到"温度感知的草稿 + 精确 `min(1,p/q)` 接受" ⇒ 采样档 AL 上升（当前生产档白丢） | 1~2 d | 中（图输出槽 + 半表合并） | **实测中性 ⇒ 留档** |
| **P1-③** | walk 的 host 往返 | 每轮 2 次 DTOH（`sel` ≤7×256 f32 + `topk_all` 224）+ host softmax/抽样，**在关键路径上同步** | 把 walk+采样搬进 `dflash_select` 图（设备 RNG：`rand32(seed,pos)` 位置派生态，语义与现 splitmix 对齐） | 省 1 次流排水 + ~9 KB 拷贝；**解锁 ④** | 2~3 d | 中（采样档确定性语义要重立门） | P1 |
| **P1-④** | accept 在关键路径 | `verify → accept(host) → replay/fold → extend+propose` 串行 | **乐观 propose**：verify 完立刻按"最可能接受的 anchor"发下一轮 propose 图，accept 回来再采用/丢弃（vLLM `next_num_draft_tokens` 同构，一步陈旧） | 每轮藏掉 propose 的 3~8 ms | 3 d | 中高（双缓冲 + 条件丢弃） | P1 |
| ~~**P1-⑤**~~ | 深度只降不升 | 位置门 + 收益门（降）；无升档；warmup 8 轮 + 2 低窗 | 阶梯判据已做成**双向 + 滞回**（`exec::spec::depth_ladder_step`，5 条单测），但**降档在 DFlash2 上不可用**：切档要给每个 depth 各捕获一整套 propose/fold 图族（boot 硬门 `depth=3 图族不完整 fold 1/3 propose 0/4`），且**收益已被证伪**（§8：verify 行边际仅 0.57 ms） | — | — | — | **实测无收益 ⇒ 留档** |
| **P2-⑥** | spec × 并发 | 与 `max_batch>1` 结构互斥 | 多请求 draft 打包 + **draft KV 压实**（sglang `_rebuild_compact_draft_cache` 同构） | 并发 2~4 时唯一上限差 | 1~2 w | 高 | P2 |
| **P2-⑦** | verify 的 logits 调整面 | 只有 rep_penalty/top-k/top-p | 将来接 grammar/logit_bias 时**必须同时进 verify 的 K+1 行**（sglang `apply_dflash_verify_logits_adjustments` 的职责） | 防"spec 静默绕过约束" | 接语法时才做 | — | P2 |

## §4 便宜档详细设计（本轮施工）

### C1 `hist` 只吃已提交 token

现状（`exec/spec.rs` 非 DFlash2 采样档）：

```rust
for (i, prow) in p_rows.iter().enumerate().take(live + 1) {
    let tok = crate::sampler::sample_sparse(prow, &mut rng);
    s.push(tok);
    hist.push(tok);            // ← 全量入账（live+1 个）
}
let m = spec_sampled_accept_prefix(&drafts[..live], &s[..live]);
(m, s[..=m].to_vec(), s[m])    // ← 实际只提交 m+1 个
```

**契约**：`hist` 是"已提交 token 的惩罚历史窗"（`SPEC_HIST_KEEP = 64`），
必须与 `act.out` 同源。改法 = 抽一个纯函数
`fn hist_commit(hist: &mut Vec<u32>, emitted: &[u32])`，两条采样路径
（稀疏拒绝采样 / 抽样等式）都只在"确定提交"之后调用；`emitted` 含 bonus。

顺带把 DFlash2 路径的 `accept_rejection_sparse` 也改成**返回后统一入账**
（现在它在函数内部边比边 push，语义正确但两套写法）。

### C2 q 走请求的采样变换（与 p 单源）

现状：`DraftQ::from_scores(scores, cand, depth, temperature, seed, base_pos)`
里自带 `softmax(边分/T)`；而 p 侧走 `sampler::distribution_of_cands`
（temp → top-k → softmax → top-p → 截断质量重均摊）。

改法：签名改成收 `&SamplerCfg`，每步：

```rust
let mut vals: Vec<f32> = row.to_vec();               // 边分 = 提案 logits
let dist = crate::sampler::distribution(&mut vals, cfg, &[]);  // 与 p 同一套
// dist: Vec<(cand_index, p)>，p 之和 = 1
```

- 抽样仍用 `uniform(seed, pos+e, PURPOSE_PROPOSAL)` 累积取；
- `prev = 选中下标`（格布局 `scores[e, prev, c]` 不变）；
- `q_of(e, tok)` 在 dist 之外返回 0（⇒ 该步必然拒绝，但 walk 不会再选中它）。

**为什么不破坏精确性**：q 只要是个分布，拒绝采样就精确（§1 的定理）；
C2 只改变"提案在哪"，不改目标分布。

**为什么能提 AL**：请求 top_k≤8 时，q 的第 9~16 名候选 p̃ 恒为 0 ⇒ 抽到它们
= 白白整轮拒。截断后 q 的支撑 ⊆ p̃ 的支撑，白付消失。

**已知不对称（记录，不改）**：`rep_penalty` 在设备侧已折进 p 的候选值，
q 的边分没折（边分不是 token 级 logit，硬折没有语义）⇒ 惩罚档下接受率略低，
但**分布仍精确**。

### C4 TP2（词表切分）档补 q —— **实测中性，未合入**（留作复测蓝图）

> 实现已跑通（`spec.q.round` 从 0 → 207、`spec.q.trunc` 725、丢质量 ≈3.1%/步），
> **但真机 A/B 证明它不转化为吞吐**，故从主干撤掉、只留本文档。

实测（2026-11-03，8135 = 双 3080 TP2 生产档，temperature=0.7 跑 16 轮）：

```
spec.q.none  counter = 16     ← 16 轮全部"没有 q"
spec.q.round counter = 0
```

代码面（`exec/draft.rs`）：`head_split_active()` 为真时走
`dflash_select_merged()`，其文档写得很清楚——图只出两槽 `topk`（值区‖索引区）
与 `sel_proj`，收割 = 合并两半候选 → rank0 跑 selector → **只返回 drafts**。
`spec_q_host` 在合并路里恒 `None`。

后果（不是错，是白丢）：

- 草稿 = selector **argmax**（贪心），与请求 `temperature` 无关；
- 接受退到"目标抽样 == draft"等价式（C1 修的正是这条路）⇒ 精确，但
  **draft 不是按请求温度抽的** ⇒ 采样档的接受率比"温度匹配 draft"低
  （温度越低差距越小，`temp=0` 时两者等价）；
- 因此 C2/C3 只在**单卡（非切分）档**点亮。

改法（与 C2 同源，不新造规范）：

1. `dflash_select` 图增加输出 `scores`（每步 `[16]` 边分，按选中前驱走的那一行）
   与 `cand`（`[K,16]` 全局 id）—— 图里 selector 本来就算了这些，只是没落槽；
2. 收割面把 peer 半表的 `scores/cand` 一并取回（与现在"一次 dtoh 当同步点"
   同一次读，不新增往返）；
3. 直接复用 `DraftQ::from_scores(..., &act.sampler, ...)` ⇒ 与单卡档逐位同构。

#### 实测（2026-11-03，8135 双 3080 TP2，temp=0.7 / top_p=0.95，同 prompt × 5 轮）

| 臂 | AL（spec_tokens / rounds） | tok/s |
|---|---|---|
| **q 关**（selector argmax 草稿 + 抽样等式接受） | 2.00 / 2.14 / 2.04 / 2.14 / 2.16 → **mean 2.10** | 69.9 |
| **q 开**（C4：温度感知 q + `min(1,p/q)`） | 2.10 / 1.86 / 2.32 / 2.07 / 2.32 → **mean 2.13** | 70.2 |

**中性，落在噪声里。** 机理（复核后）：

- q 关：草稿 = selector **argmax** ⇒ q 实际是 one-hot ⇒ `min(1,p/q) = p(d)`，
  抽到最优点就按它的概率接受；拒绝时从 p 重抽 ⇒ **分布精确且每步接受率最高**；
- q 开：`distribution` 把 q 摊在 16 个候选上（哪怕最优点 `q(d) < 1`）⇒ 每步
  接受率**反而不高于** `p(d)`；"提案更贴温度"的收益被这一项抵消；
- 根因是**选择器边分不是校准 logits**（低秩码本内积 + unary），
  所以"用请求的 top-p 去截断 q"只能去掉尾巴，改不了头部形状。

#### 复测前提（别重复踩）

1. 选择器边分先做校准（或改用目标头 top-16 的 logits 当 q）——否则仍会中性；
2. 复测时**必须**同 prompt、同 temp、≥20 轮取均值（5 轮时差 0.03 完全被噪声吃掉）；
3. 实现要点（照抄即可）：`dflash_select` 的 `sel` 槽已是
   `[S·(K·K+1)] = toks‖scores` ⇒ scores 尾段**零成本可读**；缺的只有合并后的
   `cand`，读一次本 rank `topk` 槽 + `merge_topk_two_halves`（与设备合并逐位同）
   即可；然后 `DraftQ::from_scores(&sel[S..], &cand, rows, &act.sampler, seed, fp)`。

#### 若将来启用，验收门

`spec.q.round` 在生产档 > 0（现在恒 0）；采样档 AL ≥ 基线 + 5%（≥20 轮均值）；
greedy 恒等门不受影响（`temp=0` 时 `distribution` 出 one-hot = selector argmax）。

### C3 口径记账 + 一次性 WARN

| 指标 | 含义 | 健康值 |
|---|---|---|
| `spec.q.trunc` | 本进程 q 行被请求 top-k/top-p 截断的步数 | >0 正常（top_p<1 就会） |
| `spec.q.drop` | 被丢掉的质量（ppm，逐轮累加） | 越小越好；持续 >1000ppm 说明 q 头部很平 |
| WARN（每进程一次） | 请求关截断（top_k=0 或 top_p≥1.0）⇒ spec 的 p 是 top-32 口径，比裸 decode 尖 | 出现即"口径不同"提示 |

## §5 中等档提要（未施工，按此顺序）

1. **③ walk 上设备**：先量（`spec.propose.sel` 占一轮的比例），再决定是否值得
   重写采样档的确定性语义。收益主要来自"去掉同步"而不是那 9 KB。
2. **④ 乐观 propose**：owl 的 propose 已是图态 + 地址稳定，门槛在于 accept
   结果的双缓冲与条件丢弃；做之前先把 ③ 落地（否则 host 决策仍挡在前面）。
3. **⑤ 深度双向 + copy 检测**：判据直接抄 vLLM 的"连续两步 lookup 饱和 →
   长档 + sticky"（pi agent 的复制粘贴形状），配 `AL EMA` 做兜底。
4. **⑥ 并发**：只在确有并发目标时做。

## §6 施工结果与门（本轮）

### 已落地（P0 便宜档）

| 项 | 代码面 | 说明 |
|---|---|---|
| C1 | `exec/mod.rs::sample_accept` / `hist_commit`；`exec/spec.rs` 两条采样路 | 罚历史只吃已提交 token；顺带修掉"第 i 行抽样看到前 i-1 行"的口径错 |
| C2 | `exec/proposal_q.rs::from_scores(cfg)` | q 改用 `sampler::distribution`（与 p **同一套**变换），`temp=0` 自然退 one-hot |
| C3 | `exec/draft.rs` + `spec/spec_runtime.rs` | `spec.q.trunc` / `spec.q.drop_ppm` 两个 counter；关截断时一次性 WARN |

### 门

| 门 | 位置 | 判据 |
|---|---|---|
| C1 | `exec::spec::tests::hist_only_gets_committed_tokens` | 构造 4 行 p、第 2 位拒 ⇒ `hist` 恰好 = 已提交（含 bonus），不含被拒/未提交 |
| C2-a | `exec::proposal_q::tests::q_honours_request_topk` | 请求 top_k=4 ⇒ q 支撑 ⊆ 4 个候选、和为 1 |
| C2-b | `exec::proposal_q::tests::q_is_one_hot_at_zero_temperature` | temp=0 ⇒ q 为 argmax one-hot（与裸 decode 同语义） |
| C2-c | `exec::proposal_q::tests::q_matches_p_transform` | 同一行边分与 p 用同一 `SamplerCfg` 变换后，`distribution_of_cands` 与 q 的支撑/概率逐项一致 |
| C3 | `exec::proposal_q::tests::q_records_dropped_mass` | 截断后 `dropped > 0` 且 `trunc_steps > 0` |
| 回归 | `cargo test -p owl-engine --lib spec` / `proposal_q` / `--workload gui` | 全绿；spec greedy 恒等门不受影响（C2 只动采样档） |

实测（2026-11-03）：`proposal_q` 10/10 绿（含 20k 次抽样的
`rejection_preserves_target_distribution` 精确性门）；`spec_accept_tests` 4/4；
clippy 零告警；8135 真机 temp=0.7 冒烟通过（但走的是 C1 路径，见 C4）。

## §8 实测：一轮的成本结构（**判死了"减行数"这条路**）

2026-11-03，8135 = 双 3080 TP2 / fp8 KV / d7 / 27B，temp 0.7、300 tok 一轮，
指标按"本次请求增量"统计：

| 项 | 每次 | 总计 | 占 wall（6.28 s） |
|---|---|---|---|
| `spec.round`（整轮） | **28.1 ms** × 186 | 5225 ms | **83%** |
| ├ `spec.verify`（8 行） | **21.3 ms** | 3967 ms | 63% |
| ├ `spec.propose` | 4.8 ms | 874 ms | 14% |
| │  └ `sel.read1`（peer dtoh = TP2 同步点） | **2.8 ms** | 513 ms | 8% |
| ├ `spec.fold` | 1.2 ms | 225 ms | 4% |
| └ `spec.snap` | 0.75 ms | 139 ms | 2% |
| `decode.step.deg`（降级期裸步） | **17.0 ms** × 40 | 757 ms | 12% |

**两个硬结论**：

1. **行数几乎免费**：裸步 1 token = 17.0 ms，而 verify 8 行 = 21.3 ms
   ⇒ **边际 0.57 ms/行**（每步被 13.5 GB 权重流支配，760 GB/s 下限 ≈ 17 ms）。
   ⇒ 任何"少喂几行"的优化（浅档 verify、少提草稿）**都不值**：d3 只省 ~2 ms/轮
   却掉 AL。**AL（每轮推进几个 token）是唯一杠杆。** 这也解释了 copy 负载能跑
   **265–274 tok/s**（lookup 饱和 + AL 高），创作类只有 **47–49 tok/s**。
2. **盈亏线会被踩到**：本负载 AL ≈ 1.39，盈亏线 = 轮/裸 = 28.1/17.0 = **1.65**
   ⇒ 亏 15%（实测 spec **47.6** tok/s vs 裸 **56.9** tok/s）。降级机制**确实触发**
   （40 个 `decode.step.deg`），代价 = "确认制 2 窗"前 ~32 轮白付
   + 降级期每 8 步一次 28 ms 探测轮（≈ +3.5 ms/token）。

**⇒ 后续优先级重排**（原 §5 的 P1 顺序作废）：

| 新顺序 | 项 | 依据 |
|---|---|---|
| 1 | **提高 AL**：草稿质量 / lookup 覆盖（`nmax`/`nstrong`/`agree`）/ 选择器校准 | 唯一杠杆（§8.1） |
| 2 | **每轮固定开销**：peer dtoh 2.8 + fold 1.2 + snap 0.75 ≈ 4.8 ms（17%） | 低 AL 段是纯水分 |
| 3 | 降级判据的**前 32 轮**（确认制代价） | 亏本负载上白付 ~0.9 s |
| ~~4~~ | ~~浅档 verify / 深度阶梯~~ | **已证伪**（§8.1） |

## §9 源码坐标（三家）

| 关注点 | owl | sglang | vLLM |
|---|---|---|---|
| 轮执行 | `crates/engine/src/exec/spec.rs:82` `execute_spec_round` | `dflash_worker_v2.py` `forward_draft_extend/verify` | `v1/spec_decode/dflash.py` `DFlashProposer` |
| 接受 | `spec.rs:214`（④）+ `exec/proposal_q.rs:382` | `dflash_utils.py:775` `compute_dflash_correct_drafts_and_bonus` / `:868` 采样版 | `dflash2/speculator.py` + `_selector_walk_kernel` |
| 草稿 | `exec/draft.rs:274` `dflash_propose`、`:1319` `dflash_select_merged` | `dflash_worker_v2.py:1106` `_propose_selector_block` | `dflash2/speculator.py:393` `_generate_draft` |
| q 构造 | `exec/proposal_q.rs:45` `DraftQ::from_scores` | `dflash_worker_v2.py:219` `_SelectorDraftSampler` | `dflash2/speculator.py:25` `_selector_walk_kernel` |
| 深度 | `crates/engine/src/spec/spec_runtime.rs:124` `depth_candidates` | `adaptive_spec_params.py:142` `AdaptiveStepSlot` | `dflash2/speculator.py:289` `next_num_draft_tokens` |
| lookup | `exec/lookup.rs` + `draft.rs:233` | 独立 `ngram_worker.py` | `dflash2/lookup.py` |


## §10 外部研究：**DDTree（Diffusion Draft Tree）** 能不能救 AL？（2026-11-03 调研）

> 结论：**这是目前唯一能真正抬 AL 的算法升级**，且和我们 §8 的硬件事实（行几乎免费）
> 天生互补；但**Qwen3.5-27B 的 GDN（混合线性注意力）是硬门槛**——真正的树验证
> 需要每分支一份 GDN 状态。可落地的第一步不是树，而是"用边际分布选链"（零内核改动）。

### 10.1 论文

**Accelerating Speculative Decoding with Block Diffusion Draft Trees**
Liran Ringel, Yaniv Romano（Technion），arXiv **2604.12989**
（[论文](https://arxiv.org/abs/2604.12989) · [项目页](https://liranringel.github.io/ddtree/) ·
[官方实现](https://github.com/liranringel/ddtree)，HF Transformers + PyTorch）

**一句话**：DFlash 一轮只验**一条**草稿轨迹，浪费了 block diffusion 一次前向产出的
**每位置分布**；DDTree 用这些边际分布造一棵 **B 个节点**的草稿树，用**一次**
target 前向 + **tree attention（只允许看祖先）** 验完整棵树，走最长的匹配路径。

**算法（关键就三条，实现极简）**

1. 拿草稿器一次前向的逐位置边际 `q_i(v)`（top-K 即可）；
2. **best-first 堆**枚举"rank 元组"并按 `σ(ρ)=Σ log q_i(ρ_i)` 降序弹 **B** 次
   （弹出时只压入"下一个兄弟"和"第一个孩子"两个候选，分数增量更新）；
   论文证明：前缀闭合 ⇒ 取概率最高的 B 个前缀**就是**代理目标的最优解
   （Proposition 1/2/3，与 OPT-Tree 同源，但只需一次前向而不是每层一次）；
3. 把这棵树拍平成 `[bonus, node_1..node_B]`：**position_id = 树深度**，
   attention 用祖先可见的稠密 mask；验证走"目标解码规则 → 查子节点表"，
   把匹配路径提交、第一个不匹配的目标 token 当下一轮 bonus，KV 压实到该路径。

**收益（论文 Table 1，temp=0.0 / 1.0；τ = 平均接受长度，含 bonus）**

| 模型 / 数据集 | DFlash τ | **DDTree τ** | DFlash 加速 | **DDTree 加速** |
|---|---|---|---|---|
| Qwen3-8B / MATH-500 | 7.79 | **10.73** | 5.56× | **7.52×** |
| Qwen3-8B / MT-Bench | 4.28 | **6.58** | 2.56× | **4.10×** |
| Qwen3-8B / Alpaca | 3.12 | 5.09 | 2.07× | 3.36× |
| Qwen3-8B / SWE-bench Lite | 3.60 | 5.91 | 2.65× | 4.23× |
| Qwen3-30B-MoE / HumanEval | 8.02 | 10.72 | 6.09× | 8.22× |

- **60/60 组（10 数据集 × 3 模型 × 2 温度）全部提升**；
- 预算扫描：τ 随 B 单调涨，**加速在 B≈256–512 达峰**（再大就被验证成本吃掉）；
- 论文明确写"**最优预算随硬件/实现漂移**"。

### 10.2 社区落地状态（很关键）

| 项目 | 状态 |
|---|---|
| 官方实现 | HF Transformers + PyTorch，单卡、无分页 KV、无 CUDA graph（教学级） |
| **sglang PR #27509** | +1621/−94 / 16 文件；改 flash/triton **attention backend** 支持"树状 custom mask"、改 cuda graph runner、加 `--speculative-ddtree-budget`。**2026-09-16 因 93 天无进展被关（stale）**。论文作者本人评论："我也在优化，**要优化过才打得过 DFlash**——即便预算 16" |
| **vLLM issue #40809 / PR #42910** | 实测 **+37.8% 吞吐**（相对 DFlash），**但依赖 tree attention**；上游 PR #42121 的 tree attention **已停止维护** ⇒ issue 2026-10-28 以 not-planned 关闭 |
| vLLM 实现者踩的坑（原话要点） | ①每请求 mask 不同 ⇒ 需要自定义 RoPE/位置处理；②prefill 与 spec 混合批要**回退不用树 mask**；③**temp>0 需要"树感知的拒绝采样器"**（比分布而非 argmax）；④他们在单卡上只能跑 B=64（显存） |

⇒ **这不是"抄个 PR 就能用"的东西**：三家里两家都是"做出来了但没维护下来"，核心依赖是
tree attention + 树感知采样。

### 10.3 对 owl 的可行性（三个硬门槛，按严重度）

| # | 门槛 | 现状 | 严重度 |
|---|---|---|---|
| 1 | **GDN（混合线性注意力）不能分支** | owl 的 GDN 状态是**每 turn 一个槽**（`gdn_slot` 标量），递推沿"链"前进（`spec.fold`/回滚/重放都建立在这个假设上）。树的每个分支需要**各自**的状态 | **硬** |
| 2 | **tree attention 无处可挂** | 全仓 `attn_mask`/`custom_mask` 零命中：verify 走分页 FI 的 `slots/lens/cu_seqlens`（纯链），没有 mask 通道 | 大（要新核或 eager 回落） |
| 3 | 树验证后 **KV 压实进页** | 接受路径在树里是散落的行；owl 的 KV 是 32 tok/页 + 内容寻址块身份 ⇒ 需要一次 ≤d+1 行的 gather（≈4.4 MB ≈ 10 µs）+ 块表重建（`write_bt` 已有） | 小 |

**为什么 1 是硬门槛**：论文的实验是 **Qwen3-4B/8B/Coder-30B-A3B**（纯 attention / MoE，
**没有 GDN**）。Qwen3.5-27B 是混合 GDN，树验证时同一层的 GDN 递推要在 B 个分支上
同时走——单份状态表达不了。想绕开只有三条路：

- **(a) "扇"（fan of F chains）**：F 条**互相独立**的链（各自有 GDN 状态），
  验证后取最长匹配那条提交。分布仍然无损（提交的永远是目标模型自己选出的路径）；
  代价 = **每多一条链 +1 份 GDN 状态（159 MB/rank）+ F×行数**。
  `gdn_slots` 旋钮已经存在（当前 2）⇒ F=2 结构上现成。
- (b) 只在主路径算 GDN、所有分支共用该状态 ⇒ **不无损**，违反 owl 的
  "greedy spec ≡ no-spec 逐位一致"硬门 ⇒ **否决**。
- (c) 每分支重放 GDN ⇒ 等于 F 次前向，收益归零 ⇒ 否决。

### 10.4 我们硬件上的预算估算（结合 §8）

每轮成本 ≈ `17 ms（权重流）+ 0.57 ms × 行数`：

| 方案 | 行数 | 轮成本 | τ（乐观/保守） | ms/token（乐观） |
|---|---|---|---|---|
| 现状 DFlash d7 | 8 | 21.3 ms | 1.4（创作）/ 2.1 | **20.2 ms**（比裸 17.0 还慢） |
| **DDTree B=16** | 17 | **~22 ms** | 2.4 / 3.0 | **6.5 ms** ⇒ **3×** |
| DDTree B=32 | 33 | ~36 ms | 3.2 / 4.2 | 8.5 ms |
| DDTree B=64（vLLM 同档） | 65 | ~54 ms | 3.8 / 5.0 | 9.8 ms |

⇒ 我们这代的**甜点预算明显低于 H200 的 256–512**：B=16~32 就够，
多出来的行几乎不花钱（这正是 §8 结论的正面用法）。**创作类负载（现在亏 15%）
是最大受益者**——DDTree 的接受分布直方图正是把"短接受"的质心往长里推。

### 10.5 建议的落地顺序（先做 0，它零内核改动）

> 📌 **施工交接已单独立卷**：`roadmap.local/26-11-03-ddtree-AL升级-交接.md`
> （含 S0/S1/S2 的详细设计、文件坐标、门、测量口径、踩坑清单）。
> 另补一篇关键论文：**Bole**（arXiv 2608.01651，混合 GDN 模型的树投机闭式解
> `O = D_P·Q·S_pre + C·(I+G)^{-1}·D_β·(V − D_P·K·S_pre)`，Qwen3.5 家族 MAT 4.98–7.08）——
> 它才是"Qwen3.5-27B 能不能上真树"的答案。

| 阶段 | 做什么 | 改动面 | 预期 | 风险 |
|---|---|---|---|---|
| **S0** | **用逐位置边际选链**（不是条件格 walk）：把 `from_scores` 的 walk 换成"按 `q_i` 的边际取 top-1/抽样"（`DraftQ` 里两套数据都有） | ~30 行 host + 门 | 先量出"条件格 walk vs 边际"的 AL 差；论文的理论基础就是边际 | 零 |
| **S1** | **扇 F=2**：两条独立链 + 各自 GDN 状态 + 取最长匹配 | `gdn_slots`→3、accept/回滚/fold 走"多路径"分支 | τ 1.4→~2.0（乐观 3×）；这是**GDN 兼容的树的最小形态** | 中（fold/快照账要扩） |
| **S2** | 真树（B=16~32）+ tree attention | verify 侧新 mask 通道（FI 或 eager 回落）+ 树感知拒绝采样 + KV gather | τ→2.4~3.2 | 大（等价于 sglang PR 的 16 文件量级） |

**决策建议（已被 §11 的实测取代）**：S0 已做完 —— 边际 ≈ 条件（边缘分无信息）；
且按名次直方图重估，B=16 的树只值 **+15~20%**（要多周的 GDN 树核）⇒
**先砍每轮固定开销与按 AL 切档，树隧道待 GDN 树核有现成弹药再谈**。

## §11 实测：S0「条件格 vs 边际」+ 边际名次直方图（**本轮结论：树必须按边际造；树本身不值**）

> 2026-11-03 第二会话。施工件：`[spec] dflash_walk`（三臂消融量具，缺省
> `conditional` = 设备核原律，生产零回归）+ 名次探针 `spec.cand.r{0,4,8,16}/miss` +
> 测法脚本 `tools/fixture/spec_al_ab.py`。门：`exec::proposal_q` 15/15（含 5 条新门）、
> `exec::` 56/56、config 解析门；真机自检 `spec.walk.check` ≈ 4000 轮 **零 mismatch**
> ⇒ host 条件走格与设备 `sel` 逐位同律。

### 11.1 三臂（8135 = 双 3080 TP2 / d7 / fp8 KV；temp 0.7 / top_p 0.95 / 300 tok）

| 臂 | 读哪一行 | 创作 AL（n） | 数学 AL（n） | 复制 AL | tok/s（创作/数学） |
|---|---|---|---|---|---|
| **conditional**（生产） | `scores[e, 上一步选中, ·]` | **1.373**（8）/ 1.380（5）/ 1.348（5） | **4.034**（8）/ 4.212（5） | 7.868 | 49.5 / 135.5 |
| **fixed_pred** | `scores[e, 0, ·]`（固定参考前驱） | 1.361（5） | **3.511**（5，−13%） | 7.868 | 48.8 / 114.4 |
| **unary**（纯边际） | `unary[e, ·]` | **1.423**（16 = 8+8） | **3.866**（16 = 8+8） | 7.868 | 49.2~50.3 / 124~131 |

三条读法：

1. **边缘分（低秩码本内积）几乎不携带信息**：`unary` 与 `conditional` 的差
   **落在噪声里**（创作 +3.6%、数学 −4%，而单轮 AL 的逐轮方差就有 ±0.3）。
   ⇒ DDTree/Bole 那类"按逐位置边际造树"的配方**不需要**条件边分行
   （省掉 S1/S2 的"条件兄弟"整条设计分支）。
2. **条件用错前驱会倒扣**：`fixed_pred` 把边分应用在 `cand[e-1][0]` 上 ⇒ 数学
   −13%（5 轮）。⇒ 边分不是"可丢弃的噪声"，是"**只有对齐路径时才有用、错位时
   反向拉偏**"的一项 —— 与 C4 的"选择器边分不是校准 logits"同一根因。
3. **复制负载三臂全同**（7.868 = lookup 饱和）：lookup 一开，走格策略被旁路。

### 11.2 边际名次直方图（本轮真正的新数据）

**量法**：目标在**第 0 个草稿位**选中的 token（`emitted[0]`；m=0 时它就是
bonus = 目标自己的抽样）在草稿**逐位置边际候选 top-16** 里的名次。只在消融臂
记账（消融本来就多读一次 `mcand` 槽，零额外设备成本）。

| 负载 | P(rank<1) | <2 | <3 | <4 | <8 | <16 | miss |
|---|---|---|---|---|---|---|---|
| 创作（n=685 轮） | **0.340** | 0.426 | 0.477 | 0.509 | 0.613 | **0.686** | 0.314 |
| 数学（n=367 轮） | **0.755** | 0.839 | 0.894 | 0.929 | 0.951 | **0.965** | 0.035 |
| 复制（n=74 轮） | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |

`P(rank<1)` 与链式 AL 自洽：创作为 0.340 ⇒ AL ≈ 1 + 0.34 + 0.34·p₂ ≈ 1.44（实测 1.42）。

**它给出的三条硬结论**：

- **单层扇宽是"最便宜的 AL"**：创作从 1 档到 16 档，首 token 命中率 0.34 → 0.686
  （+35 个点）；数学 0.755 → 0.965。
- **但也有天花板**：创作有 **31.4%** 的轮次目标 token 根本不在草稿 top-16 里
  —— 任何树都救不了那部分（草稿头分布 vs 目标分布的 KL 是真上限）。
- **扇宽换 AL 的汇率很差**：每步行成本 0.57 ms，而一轮的固定成本（权重流）是 17 ms。

### 11.3 重估树预算（§8 的成本模型 + 11.2 的命中率）

轮成本 ≈ `17 ms + 0.57 ms×行数 + propose 4.8 + fold/snap 2.0`：

| 方案 | 行数 | 轮成本 | 创作 AL（估计） | ms/token（创作） | 相对现在 |
|---|---|---|---|---|---|
| 现状（d7 链） | 8 | 28.1 ms | 1.37 | 20.3 | — |
| 星形 W=16（只有第一层扇开） | 17 | 33.3 ms | ≥1.69（名次数据下界） | 19.7 | **+3%** ❌ |
| best-first 树 B=16（宽度按概率摊到各层） | 17 | 33.3 ms | 1.9~2.1 | 15.8~17.5 | +14~22% |
| best-first 树 B=32 | 33 | 42.4 ms | 2.3~2.6 | 16.3~18.4 | +10~20% |

数学负载同型（B=16：AL 3.9 → ~5.5，7.6 → 6.0 ms/token，+20%），复制负载不动。

⇒ **树的真实回报是 +15~20%，不是论文里的 2~3×**。原因是我们与论文的实验域
不同：论文的基线 τ 已经是 4~8（temp 0/1、MATH/SWE 这类低熵域），
而我们最吃亏的创作类 AL 只有 1.37、且目标分布的重尾比 DFlash 草稿宽
（31% 的 token 连 top-16 都不在）。

### 11.4 决策（据本轮数据改写 §10.5 的顺序）

| 项 | 判决 | 依据 |
|---|---|---|
| 真树 + GDN 闭式解（原 S2） | **暂不立项** | 天花板 +15~20%，代价是多周（tree mask + 树扫描 GDN 核 + 因子化状态） |
| 扇 of F 链（原 S1） | **不做** | 宽度换 AL 的汇率差（11.3 首行）；且 GDN 状态每链 +159 MB/rank |
| ~~条件兄弟树~~ | **判死** | 11.1：边缘分 ≈ 0 信息；错位还倒扣 |
| **每轮固定开销（9.5 ms = 34%）** | **优先** | propose 4.8（其中 peer dtoh 2.8）+ fold 1.2 + snap 0.75；把它压到 ~2 ms ⇒ 轮 21~23 ms ⇒ 创作 **+20~30%**，数学/复制同幅受益 |
| **按 AL 切换 spec/裸 decode** | **优先** | 创作 AL 1.37 < 盈亏线 1.65 ⇒ spec 实亏 15%（49.6 vs 裸 56.9）；现有降级判据要 ~32 轮才生效 |

**给未来树隧道的备忘**（若哪天有 GDN 树核可用）：树必须**按边际**造
（`mcand` 值半区，host 纯函数即可，无需新增设备读面）；预算 B=16 是甜点，
不要超过 32（行成本虽便宜，但每加一行都要摊 AL 的固定成本）。

---

## §12 实测：每轮固定开销 → 自适应切档（2026-11-03 第二会话下半场）

> S0 结案后按 §11.4 的判决动"每轮固定开销"。本轮实际落地的是**降级判据的
> 经济学修正**（+ 降级期同步节流），真机 **创作 +12%**，数学/复制不变。

### 12.1 起点数据（8135，temp 0.7 / top_p 0.95 / 300 tok，n=6）

| 负载 | 现状 | 裸 decode（spec 关，§8） | 结论 |
|---|---|---|---|
| 创作 | 49.5 tok/s（AL 1.37，轮 28.3 ms） | 56.9 | spec 亏 13% |
| 复制 | 277–279 | — | lookup 饱和 |
| 数学 | 129–135 | — | spec 赢 2.3× |

### 12.2 查出的三处失真（都是"仪表错 ⇒ 决策错"）

1. **轮成本被少算 6 ms/轮**。降级判决发生在 accept 之后、轮尾（fold / 边界快照 /
   ⑦propose）之前 ⇒ 就地量得 **22.3 ms**，而全程是 **28.3 ms**
   ⇒ 盈亏线 22/19 = 1.16 被 `BREAK_EVEN_MIN` 夹到 **1.35**，真值是 **1.49**
   ⇒ 引擎以为投机比实际便宜 ⇒ **创作档 65% 的轮白亏**。
   修法：`SpecRuntime::last_round_ms`（引擎级保鲜，轮尾写、下一轮决策点读）。
2. **降级期草稿同步贵 3.7×**。注释写"≈0.5 ms(2%)"，真机 `decode.sync` =
   **1.83 ms/步**，占裸步 18.9 ms 的 **10%**。
   修法：节流 —— 每 8 步同步一次 + **探测前最后一步必同步**（洞 ≤7 行，在既有的
   "毒化带 ≤8 token"容忍内）。
3. **恢复判据太松 ⇒ 降级↔恢复振荡**。旧实现一次 `m ≥ 1` 就恢复；真机 363 轮里
   **4 降 3 恢复**（12 s 请求抖 7 次），而创作档 `P(m≥1) ≈ 0.29`。
   修法：**连续两次命中**或**一次决定性命中（m ≥ 3）**；健康档第一探就决定性
   （数学/复制 `P(m≥3)` 高）⇒ 恢复延迟 ≈ 8 token。

另加两道护栏：**深饿一窗确认**（`AL < 0.95×线` 视为已在亏，不必等第二窗）与
**窗口最少 8 轮**（冷首轮曾一轮成窗 ⇒ 复制档首个请求误降级 194 tok/s）。

### 12.3 结果（同 prompt / 同参数，n=6）

| 负载 | 前 | 后 | Δ |
|---|---|---|---|
| **创作** | 49.5 tok/s | **55.4** | **+12%**（= 纯裸 decode 天花板 56.9 的 97%） |
| 复制（稳态） | 277–279 | 279–284 | 0 |
| 数学 | 129–135 | 130–134 | 0 |

真机日志判读：创作档在第 1~2 个窗口（16~32 轮）后降级并**长留**裸 decode
（`spec.degrade` 6 次 / `spec.undegrad` 0 次，全部落在创作请求内）；
数学/复制一次都没降。降级步实测 **17.5 ms**（节流前 18.9）。

**两处已知残留**（都只值一轮）：
- 进程/批次**首个请求**偏慢（冷图 + 脏草稿池）：创作后紧接的复制首轮 218 vs 稳态 280；
- 创作档距离纯裸 decode 还差 2.6%（探测轮 + 检测延迟）。

### 12.4 下一步（按 ROI）

| 项 | 现状 | 可省 | 风险 |
|---|---|---|---|
| ⑦ propose（4.8 ms/轮，14%） | 串在 verify 之后；其中 peer dtoh 等待 2.8 ms | 2–3 ms（需锚点猜测；猜错 +4.8） | 中 |
| `spec.fold` 1.28 + `spec.snap` 0.82 | 每轮 ~2.5 趟 GDN 状态流量（159 MB/趟） | ~1.5 ms（fold 一次写池+快照） | 中高（回滚面） |
| verify 21.3 ms（8 行） | 已在权重流带宽地板（13.5 GB / 760 GB/s ≈ 17.8 ms） | 0 | — |

⇒ 下一刀应在 **propose 与 verify 的重叠**（收益最大），fold/snap 合并次之。

---

## §13 实测：⑦ propose 能不能与 verify 重叠（**结论：不能，只值 0.6 ms**）

> 2026-11-03 第三刀。目标（§12.4 第 1 行）：propose 4.8 ms/轮里那 2.8 ms 的
> peer dtoh 等待。**实测证伪**，但同时落地了一个安全的 0.6 ms 重排（发射/收割两段式）。

### 13.1 先把相位账算清

| 段 | 墙钟 | 里面是什么 |
|---|---|---|
| `spec.verify` | 21.3 | 目标前向 8 行（权重流 13.5 GB ⇒ **带宽地板**） |
| `spec.propose.launch` | 1.36 | 两 rank 的图入队（host） |
| `spec.propose.sel.read1` | 2.81 | **等 peer propose 图算完**（= 图 GPU 时间） |
| `spec.propose.sel.read` | 0.28 | 本 rank sel 槽回读 |
| `spec.fold` | 1.28 | 快照恢复拷贝（159 MB D2D）+ fold 图 |
| `spec.snap` | 0.82 | 轮首快照拷贝（159 MB D2D） |

⇒ 一轮 GPU 实忙 ≈ 21.3 + 2.8 + 1.3 + 0.8 = **26.2 ms**（轮 28.3）⇒ **GPU 占用 93%**，
可藏的空间只有 ~2 ms 的 host 入队时间。**propose 的 2.8 ms 是 GPU 在干活，藏不掉。**

### 13.2 为什么"猜锚点"不划算

乐观 propose（vLLM `next_num_draft_tokens` 同构）要猜 m（本轮接受数）。
本机 m 分布（`spec.m0..m7` 计数差）：

| 负载 | P(m=0) | ΣP(m)²（= 猜"上轮 m"的命中率） |
|---|---|---|
| 创作（363 轮） | 0.71 | 0.55 |
| 数学（158 轮） | 0.32 | **0.19** |

命中最多省 ~0.6 ms（真能藏的部分），未命中要**重发整图 4.8 ms**
⇒ 期望值 = 0.19×0.6 − 0.81×4.8 **≪ 0**（数学档）；创作档 0.55×0.6 − 0.45×4.8 仍为负。
⇒ **不实现猜测路**。

### 13.3 落地的（无猜测）重排：发射提前到 fold 之前

`⑦` 拆成两段：`dflash_propose_launch`（构输入 + 两 rank 发图，只用 verify 的 taps /
anchor / m）与 `dflash_propose_harvest`（读两半候选 + select + lookup 融合）。
发射挪到 **⑥ fold/边界快照之前** —— 两者写的是不同池（草稿池 vs 主池），无数据依赖。

**实测收益：每轮 28.3 → 27.7 ms（−0.6 ms / 2%）**，账目上体现为
`spec.fold` 1.28 → **4.03**（它现在真的在等 propose 图）、`spec.propose` 4.8 → **2.77**。

为什么只有 0.6：fold 的拷贝与 propose 图**在同一流上排队**，GPU 总工作量不变；
变的是 host 只等一次而不是两次。**要拿更多必须让两段真并发**（双流），而那会
把带宽争用摊回原样 —— 省下的仍是 ~1.3 ms 的入队延迟，收益/风险比不划算。

### 13.4 顺带定谳的一个测量坑（很重要）

**复制档的 AL 依赖"上一批请求的草稿环内容"**：同一二进制、同一 prompt，
`copy` 打头跑 = **AL 7.868（38 轮）**，`math` 之后跑 = **AL 7.475（40 轮）**
（tok/s 282 → 268，−5%）。⇒ 跨负载连续跑时"复制档 tok/s"有 ±5% 的伪差；
**做 A/B 时要么单独起机测复制档，要么只比同批内的首条**。

### 13.5 结论与下一刀

| 项 | 判决 |
|---|---|
| propose 与 verify 重叠（P1-④） | **证伪**：GPU 占用 93%，可藏仅 ~0.6 ms；猜锚点期望值为负 |
| 发射/收割两段式 | **保留**（−0.6 ms，且是将来双流/跨轮流水的前置件） |
| **真正还有肉的地方** | `fold` 的 **restore 拷贝**与 `snap` 的 **capture 拷贝**：每轮各 159 MB D2D，且这对拷贝只是为了"fold 幂等"这个契约而存在（图形路径里 restore 的源就等于池里现值 ⇒ 多半是 no-op）。改成**就地 fold** + 懒快照，估 **1.2~1.6 ms/轮（4~6%）**，但要先审回滚不变量 |

---

## §14 实测：fold/snap 的 GDN 状态账（**每轮 5%，两条便宜路都证伪**）

> 2026-11-03 第四刀。目标（§13.5）：capture/restore 那对 159 MB 拷贝。

### 14.1 先把账拆开（新增 4 条有界 metric）

| 段 | 每轮 | 说明 |
|---|---|---|
| `spec.snap` | **0.79** | 轮首快照（`capture_spec_snap`） |
| ├ `spec.snap.build` | 0.003 | 组 64 段拷贝表（可忽略） |
| └ `spec.snap.send` | **0.72** | `copy_batch` 往返（64 段） |
| `spec.restore.send` | **0.71** | fold 开头的反向拷贝（只部分接受轮跑） |
| `spec.fold.step` | 2.5 | fold 图（当前含等待提前发射的 propose） |

几何真值（boot 日志）：**16 个 GDN 层 × (rec 4.5 MB + conv_q/k/v 各 12 KB)**
⇒ 72 MB / **64 段**，其中 **48 段只有 12 KB（0.8% 的字节）**。合计每轮 **~5%**
（创作档 1.43 ms/轮），与 §12 的估算一致。

### 14.2 证伪一：`cuMemcpyBatchAsync`（CUDA 12.8+ 一次调用搬 N 段）

接了运行时符号解析 + 失败回落逐次（`cuMemcpyBatchAsync_v2`，本机 615.71 有该符号）。
**实测：`snap.send` 0.72 → 0.720、`restore.send` 0.703 → 0.712 —— 零收益。**
⇒ 单段成本在**驱动内部**（每段的事务/页表处理），不在 ABI 次数上。
（改动已回退，不留死复杂度。）

### 14.3 证伪二：少拷 / 懒拍

- **跳过轮首 capture**：不行 —— fold 的 restore 无条件读它（`fold_step_dev` 第一步
  就是 `restore_spec_snap`），而 verify **确实推进了池里的状态**（`⑤` 的轮不变量
  注），所以"capture + restore"是一对，不能只砍一半。
- **只在跨页边界轮拍**：同一理由 —— 非边界轮也要 restore。
- **不拍 conv 三段**：fold 的 conv 重放以快照为基态，砍了就错。

### 14.4 剩下的两条真路（都要动状态语义，估 4~5%）

| 方案 | 做法 | 收益 | 代价/风险 |
|---|---|---|---|
| **A. 影子状态槽**（推荐） | verify 写"影子槽"、真槽永远停在 state@pos ⇒ **capture 消失**；fold 就地推进真槽（空接受轮也要 fold，需要 fold 家族扩到 `rows = depth+1`）；边界轮保留 capture+restore（25% 的轮） | **~0.7 ms/轮**（2.5%） | `gdn_slots` 翻倍（+72 MB/rank）；边界轮的双折仍要基态 |
| **B. verify 读写分离** | GDN 谱核吃两个槽（读 A 写 B）⇒ capture/restore **全消失**；满接受轮把 turn 的 gdn_slot 换到 B（双缓冲，零拷贝） | **~1.4 ms/轮**（5%） | 要改 **FLA chunked 六核**（Triton/PTX 契约）的 state 出口 —— 研究级 |
| ~~C. 批量拷贝 API~~ | §14.2 | 0 | 已证伪 |
| ~~D. 跳过 capture / 少拷~~ | §14.3 | 0 | 已证伪 |

**下一步建议**：A 是唯一"中等工作量 + 确定收益"的路；B 才有 5%，但要动 GDN 谱核。
两条都**必须**以 `spec.m0..m7` 的 AL 分布 + temp=0 连贯性当金丝雀（状态一旦错，
AL 立刻塌），并用 `[spec] gdn_shadow = false` 这类显式开关做 A/B。

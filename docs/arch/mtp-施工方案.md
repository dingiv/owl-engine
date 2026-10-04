# E5-MTP 施工方案 —— owl 投机解码(MTP 先行,DFlash2 续作,接口多算法)

> 2026-10-05。前置:四家调研(`spec-decode-调研与立项.md`)完卷;decode 45.6 达标。
> 本文档 = MTP 施工设计。原则:接口一次成型支持多草稿算法(现役 MTP + DFlash2
> 两家, trait 留扩展位);正确性门 = **greedy 下 spec 输出与无 spec 逐位等价**
> (接受是恒等变换,最强验收);性能门 = C7(分域净增益 >0 才默认开)。

## 一、MTP 几何事实(侦察定谳)

- **MTP 层 = 与 target 同款门控注意力 decoder layer**:ninfer `mtp_pack` 实测
  fc 入力 = concat(embedding_norm, hidden_norm) [2×5120];attn rows 14336 =
  q 6144(24×256)+ k 1024(4×256)+ gate 6144 + v 1024(与 target attention 的
  q⊕gate 融合同构 = owl Attention 层原生支持);mlp 17408。vLLM
  `Qwen3NextMultiTokenPredictor` 同构(fc + 1×Qwen3NextDecoderLayer(full) +
  pre_fc_norm_hidden/embedding + norm,共享 embed/lm_head)。
- **权重来源(2026-10-05 订正,用户质询触发)**:**cyankiwi 检查点自带完整
  MTP 头** —— 15 个 `mtp.*` 键全 BF16(config 未声明 ≠ 权重不存在;量化
  配置照例 ignore MTP 层与 lm_head 同待遇):mtp.fc [5120,10240] +
  layers.0.self_attn(q_proj [12288,5120] = q⊕gate 融合，与 target 同款
  门控注意力)/k/v/o + q/k_norm + mlp + pre_fc_norm×2 + norm —— 与 vLLM
  `Qwen3NextMultiTokenPredictor` 逐键对齐。**M0 缩水 = 现有 AwqSource 读已
  索引的 mtp.* 键(F16 形态兑底通道现成)，零新格式**；原「.ninfer 解析」
  作废(不引入小众格式支持)。ninfer 几何实测仍有效(互证)。
- **MTP 层有独立 KV 表**(ninfer `mtp_kv_table_rows` 与 text 表分立):草稿链
  注意力吃自己的 paged KV。几何 1 层 × hkv4 × hd256 × f16 = **4KB/token**,
  8704 ctx 全量 = 35MB(微不足道;"vLLM/sglang 单卡装不下"的元凶是他们框架
  的草稿图/显存管理,不是这块 KV)。

## 二、接口设计(多算法,一家一个 Drafter)

### 2.1 trait(放 `crates/models/src/spec/mod.rs`;引擎侧经 Scheduler 消费)

```rust
/// 草稿器:一轮 = propose → (引擎做 verify) → observe。
/// MTP 与 DFlash2 各一实现;verify/accept/状态回滚是引擎公共件。
pub trait Drafter {
    /// 本轮草稿数(自适应降档留位;v1 恒定:MTP=3,DFlash2=7)
    fn depth(&self) -> usize;
    /// verify 块行数(depth + anchor 行;MTP=4,DFlash2=8)
    fn verify_width(&self) -> usize;

    /// 出草稿。输入 = 引擎供给的轮上下文(anchor token + anchor 位的
    /// target 末层 hidden;两者都是上一轮 verify 的产出,MTP 与 DFlash2
    /// 同源消费)。返回草稿 token(device 块;图内链式产出)。
    fn propose(&mut self, ctx: &ProposeCtx) -> Result<TensorOps>;

    /// verify 后回填:引擎交回 verify 块的 argmax 行/接受长度/本块
    /// hidden taps(MTP 不用;DFlash2 取 5 层投影进窗口)。
    fn observe(&mut self, o: &ObserveCtx) -> Result<()>;
}
```

- `ProposeCtx { anchor_token: u32(device 槽), anchor_hidden: TensorOps(末层
  hidden 行), kv_len: usize }` —— **两家消费同一名词**:
  - MTP:fc(concat(embed_norm(anchor), hidden_norm(h_anchor))) 进草稿链
  - DFlash2:窗口尾 append(h_anchor 投影)后 noise 块进草稿
- `ObserveCtx { verify_hidden_taps: &[TensorOps](各层末行输出), accepted: usize,
  bonus: u32 }`
- 引擎不感知算法差异:**调度只看 `depth()/verify_width()`,执行只调
  propose/observe**,verify/accept/回滚全公共(§四)。

### 2.2 引擎集成点(最小侵入)

- `SchedulerOutput` 新 `StepAction::SpecRound`(决策条件:drafters 在场 +
  活跃会话 decode 相;否则回落 DecodeBatch —— C7 回退语义的挂点)
- `exec.rs` 新 `execute_spec_round`:快照 → propose → verify(prefill 通道
  T=verify_width,pos 从 B 起)→ accept → (restore)→ 发射
- 事件流不变:`Token` 事件逐个发(accepted + bonus);usage 口径不变

## 三、MTP 模块(新 model 组件)

```
crates/models/src/models/mtp.rs:
  MtpPredictor {
    fc: Linear(2*hidden → hidden, 无 bias),
    layer: Attention(24H/4KV/256, gated — owl Attention 原生)+ mlp(17408)+ norms,
    pre_fc_norm_hidden / pre_fc_norm_embedding / norm: RmsNorm ×3,
    // embed/lm_head 共享 target(不持有;propose 由引擎喂 embed 行/lm_head 走 target)
  }
```

- 草稿链(k=3)全 SSA 一棵树:step i = fc(cat(embed_norm(tok_i),
  hidden_norm(h_{i-1}))) → layer → norm → lm_head → argmax → tok_{i+1};
  h_i = step i 输出 hidden(feedback = 树内边,无循环);tok_0 = anchor,
  h_0 = anchor_hidden ✓ 图友好(单图捕获,零 host 往返)
- **MTP 层 KV**:独立 mtp 链(BlockManager 第二链/会话,页 32 同款)——
  草稿 step i 在 mtp 链上 pos = B+i 读写;已提交 token 的 mtp KV 由
  **draft-extend**(≤4 行 × 1 层,≈30µs)在 accept 后补写(ninfer
  alignment 同语义)
- 权重:M0 从 .ninfer 提取 → 转出格式(mtp.fc / mtp.layers.0.{attn,mlp} /
  三 norm;groupwise-int4 → 落 f16 或直接转 AWQ-marlin,先 f16 简单)

## 四、轮结构不变量(设计核心,逐符号推导)

记号:L = 已提交 token 数(槽 [0,L));anchor w_B(B = L-1)= 上轮 bonus,
**KV 已写、GDN 未处理、target 末层 hidden h_B 已知**。

**轮不变量**:GDN = state@B-1(即 token B 未进递推)。

1. draft:从 (w_B, h_B) 出 d1..dk(位置 B+1..B+k 的提名)
2. verify:块 [w_B, d1..dk](位置 B..B+k,共 k+1 行)从 state@B-1 推进:
   - 行 r 的 logits 预测位置 B+r+1 → row0 验 d1 … row(k-1) 验 dk,
     row k 的 argmax = bonus b(位置 B+k+1 候选)
   - KV 写 B..B+k(槽 B 覆写 = 幂等;GDN 推进 k+1 token)
3. **全接受(m=k)**:发射 d1..dk + b;新 L' = B+k+1;**GDN 现值 =
   state@B+k = state@L'-1 = 下轮不变量**(B' = L'-1 = B+k)—— **零拷贝
   零回滚,verify 末态直接延续**;h_{B'} = 行 k 的末层 hidden ✓
4. **部分接受(m<k)**:发射 d1..dm + b = 行 m argmax;restore 快照
   (@B-1);新 L' = B+m+1,下轮不变量需 state@B'-1 = state@B+m-1 =
   state@B-1 + [w_B, d1..dm]。两案:
   - **v1 窗口重处理**(先做,零新核):下一轮 verify 块 =
     [w_B, d1..dm, b, e1..e_k']从快照重推(块长 m+k'+2 动态;已接受行
     重算恒等 —— 同快照同输入确定性同输出);快照本身不还仓,连续部分
     接受时块前缀累积(有界:一次全接受即清零复位)
   - **v2 fold 核**(D4 后优化):verify 期间记录各行 (k,v,g,β,conv 行)
     (SSA 现成),restore 后单核重放 m+1 行(48 层一次 launch,~0.1ms)
5. 快照时机:轮首(48 层 × 3.1MB D2D ≈ 0.3ms;全接受主流路径下这是纯税,
   可换 v2 的 record 消除 —— 挂账)

**kv_len 记账**:accept 后 L' = B+m+1;KV 槽 B+m+1..B+k 的写入作废
(下轮 verify 从 B' 起重写,页内覆写幂等;BlockManager 链按 L' 收口)。

## 五、图与调度

- **draft 链图**(定形:k 步 × 1 token):输入槽 = anchor token + anchor
  hidden 行 + mtp 链块表;输出 = d1..dk + 链末 hidden。与 decode 图同族
  (multi-root ✓)
- **verify**:v1 走现有 prefill 通道 eager(T=4 小块,FI 需 plan 或回退
  paged——T=4 attention 成本可忽略);D4 桶形图(width=4 定形)
- **decode 图保留**:C7 回退路径 + 无草稿器时主路径(调度二选一)
- accept:v1 host(verify 4 行 argmax 已是图输出,dtoh 4×4B + 3 草稿 id
  比对 ≈ 50µs);D4 换 ninfer 式单核 device accept(+rejection 采样)

## 六、内存预算(24G 贴顶)

| 项 | 增量 |
|---|---|
| MTP 权重(f16 提取;fc 52M + attn 99M + mlp 267M ≈ 420M 参数)| ≈ 840MB |
| mtp_kv 链(4KB/tok × 8704)| 35MB |
| GDN 快照(现役 SNAP 通道)| 151MB(已备) |
| 草稿链/accept 缓冲 | <10MB |
| **合计** | **≈ 1.0GB → boot ~21GB ✓**(超限时 OWL_POOL_TOKENS 降档) |

W4A16 权重可再省 600MB(二期;先 f16 简单正确)。

## 七、里程碑与验收门

| # | 内容 | 验收门 |
|---|---|---|
| **M0** | 装载器读 checkpoint 自带 `mtp.*` 键(BF16→f16,现有 mmap 源)+ 几何 dump | 15 键全载；形状与 vLLM MTP 结构逐键对齐 |
| **M1** | MtpPredictor 组件 + 装载 + 草稿链单跑 | host 逐式参考对拍(fc/norm/attn 逐算子 ≤2e-2);k=3 链 argmax 有限性 |
| **M2** | SpecRound e2e(快照+draft+verify+accept+restore) | **恒等门:greedy spec 输出 ≡ 无 spec 输出(逐 token 位数等价,≥512 tok × 双 prompt)** |
| **M3** | 性能 + 分域 | MATH 域净增益 >0(C7);prose/std 自动回退零损失;acceptance 直方图 + 三段账可视 |
| **M4** | verify 桶形图 + device accept + fold 核(可选) | 图化后净增益不回退;host/步 <1ms |
| **M5** | 默认开裁决 | 数据驱动(C7) |

## 八、风险与开放问题

1. **MTP 权重质量未知**:ninfer 容器的 MTP 头与 cyankiwi AWQ target 的
   匹配度(架构同、量化不同)—— M2 恒等门与 M3 接受率直接回答;若
   接受率 <20% 则评估 lued/syvai 之外的新草稿训练(超范围,先测)。
2. **不变量破坏 = 文本乱码**:§四的推导必须以 M2 恒等门机器验证,不得
   肉眼判文本(刀1.6/页配对律的教训:状态双推/错配的文本症状 = 复读
   吸引子,肉眼不可靠)。
3. **快照 0.3ms/轮纯税**:全接受主流路径下吃掉 ~13% 增益 → v2 fold 核
   (record 消除)排 D4。
4. **verify eager(T=4)的发射开销**:prefill 通道 64 层 × T=4 ≈ 830 发
   eager ≈ 与 decode 图同量级 —— M3 数据驱动是否提前 D4。
5. **draft-extend 的 mtp 链记账**(accept 后补写 ≤4 行):与 BlockManager
   双链(mtp_bt)交互,M2 内一并验。


## 九、M2b 源码定谳与管线修订(2026-10-06,vLLM 源码级对证后增补)

> 对证源:`packages/vllm` `vllm/v1/spec_decode/llm_base_proposer.py`
> `set_inputs_first_pass`(EAGLE/MTP 共用轮结构)+ `qwen3_next_mtp.py`
> (模块同构互证)。本节取代 §四 中与本文冲突的记号。

### 9.1 配对语义(核心悬案定谳)

vLLM 草稿首遍输入装配(ids 移位,hidden/positions **不移位**):

```python
self.input_ids[:n-1] = target_token_ids[1:]      # token 左移一位
self.input_ids[采样位]   = next_token_ids          # 末槽 = 采样 token
self.hidden_states[:n]  = target_hidden_states    # hidden 原位
```

⇒ **pair(emb(t_{p+1}), h_p) 处理在位置 p(hidden 的位置)**。mtp KV 的
position 0 有条目 (h_0, emb(t_1)),**无 gap**(本卷此前担心的 h_{-1}
缺口不存在)。M1 propose 的 pos 约定(wiring 测试 pos=0 起)与该约定
同构 —— 引擎接线只需把 base 对齐到 verify 块基址(见 9.3)。

**anchor_hidden = 行 m(最后被接受行)的 target 末层 post-final-norm
hidden**(padded 路径 `query_end_loc -= num_rejected_tokens` 实证:被拒
行被裁掉,采样 token 与其「预测者 hidden」配对)。owl `model.last_hidden`
含 final_norm ✓ 即消费面。

### 9.2 轮结构修订:extend 并入 propose,零跨轮 hidden 持久化

记号沿用 §四(F = 本轮 verify 块基址,m = 接受数,b = bonus):

```
每轮(SpecRound 一次 pump 迭代内):
  1. 快照 GDN(M2a 不变)
  2. verify 块 [w_F, d1..dk] @ F..F+k(M2a 不变)
     → 产出 argmax ids + [k+1, hidden] hidden 块(树内,本迭代存活)
  3. accept(m, b;host 比对 ids vs drafts dtoh ~8 标量)
  4. extend+propose(单棵 SSA 树,同迭代):
     a. extend 行 i = (h_{F+i}, emb(t_{F+i+1})) @ 位置 F+i,i = 0..m
        (t_{F+i+1}: i<m = 已接受草稿;i=m = bonus —— 全部 host 已知)
        → 批量 MTP 层前向(prefill 形 ctx,fi=None 走 paged)
        → 末行(i=m)输出 = 链种子 hidden',argmax = d1
     b. 链步 j = 1..k-1:(emb(d_j), h'_prev) @ 位置 F+m+j(decode 形 ctx)
        → argmax = d_{j+1}
     → 草稿 d1..dk(device 块 [k,1],持久到下一轮)
  5. 部分接受 → GDN restore + 前缀重放(M2a 修复,不变)
```

**关键红利:hidden 块生命周期 = 单次 pump 迭代**(extend 就地消费 verify
树,无跨轮设备块持久化;唯一跨轮态 = 草稿 device 块 k×4B + host m/b)。

首轮衔接:prefill 末块 forward_last 已产 [1,hidden](= h_{P-1})+ 首
token → harvest 两件持久 → 首个 SpecRound 无草稿在场时先跑
propose(&[t_P], h_{P-1}, base=P-1, k)(即 M1 propose 本尊,m=0 特例)。

**prefill extend**:prefill 逐 chunk 完成时对该 chunk 的 last_hidden 跑
extend 行(chunk_base..chunk_end-1 的 pair(h_p, emb(t_{p+1}))),覆盖
mtp KV 0..P-2;位置 P-1 由首轮 propose 的 extend 行补齐 —— **mtp KV
0..F'-1 全程无缝**。逐 chunk 就地 extend,免 hidden 累积驻留。

### 9.3 mtp 链 KV 记账(取代 §八.5)

- **覆盖不变量**:propose 开工时 mtp KV 已覆盖 0..F-1;extend+链写
  F..F+m+k-1;下一轮需 0..F'-1 = 0..F+m ✓(恒被覆盖,含全接受)。
- **免回滚**:部分接受时链步写在 F+m+1..F+m+k-1(越过提交点)= 陈旧
  条目,但注意力 kv_len 有界(≤ 提交点)**永不读**;后续轮 extend/链
  顺位覆写 —— 与 target KV 槽幂等覆写同律,零 mtp 回滚动作。
- **承载**:会话第二块链(`Session.mtp_block_table`)+ 独立页池
  (复用 BlockManager,不调 cache_seq —— mtp KV 依赖 hidden 不可
  token 哈希,无前缀缓存)。页 32 同款;27B 全量 35.7MB(在案)。
- **MTP 层算子路径**:恒 paged(fi=None / attn_v2=None,ctx 驱动零新
  旋钮);T=m+1≤4 attention 成本可忽略(§八.3 同判)。

### 9.4 工程清单(M2b 施工序)

1. `mtp.rs propose_ext(tokens, hiddens, base, k)`(extend 批行 +
   k-1 链步;M1 propose = 其 m=0 特例,保留作 wiring 锚)。
2. verify_forward 返回值扩 `(ids, hidden 块)`;**顺手修 M2a 存量雷:
   `_arena` 未 free(每轮泄漏 verify 中间块)** —— hidden 块从候选表
   retain,余量照焚。
3. 引擎:Session.mtp_block_table + mtp 页池;prefill 末块 harvest
   (hidden, token);execute_spec_round 按 §9.2 重排;drafts 块跨轮
   持久(pump 字段,轮首消费轮末替换)。
4. fc 列半拆双 Linear(M0 挂账兑现;消 2×50MB/轮物化,n=5120 过
   marlin 谓词可入 marlin)。
5. 恒等门升级:27B 真权重 A/B(spec ≡ no-spec 逐位;哑草稿 0.8B 门
   保留为机制回归)+ **接受路径覆盖断言**(m=0 / 部分接受 / 全接受
   三路都被踩到 —— 哑草稿门几乎只踩 m=0,真草稿才踩全接受)。
6. C7 回退挂钩:无 MTP 头的模型(0.8B)spec_depth 强制回落
   DecodeBatch(现 spec_depth 与草稿器解耦,接线时收拢)。

### 9.5 成本预算(27B,数据驱动待 M3 复核)

extend 行 ≈ 0.4ms + 链步 ≈ 0.4ms(MTP 层)+ **lm_head GEMV 2.66ms/次
× (1 extend + k-1 链) ≈ 8ms @k=3 = propose 大头(53%)**;verify 侧
lm_head 2.66ms(forward_last 单行)。round ≈ 15ms,AL=2 → ~133 t/s
(基线 43,+210%)。**已知优化挂 M3**:greedy 域 d1 可免 propose
(verify 行 0 argmax 免费产出,省一链步一 lm_head ≈ -3ms;采样域需
rejection 才要真 d1,vLLM 保留是为 rejection 采样,owl greedy 专属可裁)。

# 中断(客户端断流/取消)后的前缀缓存命中 —— 立案与结案(2026-11-03)

> 症状(前端实测):模型思考中客户端中断提交,随后再提新需求 ⇒ prefill 明显变慢。
> 结案:客户端中断时,**本 turn 已算完的 prompt 前缀没有进内容树** ⇒ 下一轮
> 内容寻址命中 0 ⇒ 整段重算。修好后同形状 5850 tok prompt 的 TTFT
> **2395 ms → 186 ms**,`cache` **0.5% → 99.3%**。
>
> 档位:`profiles/qwen38_27b-3080_tp2.toml`(双 3080 / TP2 / 27B AWQ-INT4 / fp8 KV),端口 8135。

---

## 一、定位(真机取证,不靠读码猜)

`apps/server` 在流式写失败时发 `EngineReq::Abort` ⇒ `drop_turn` ⇒
`RunningEngine::cancel_turn` ⇒ `abandon_active`(见 `apps/server/src/executor.rs:393`)。

复现脚本(5850 tok 随机词池 prompt + `session` 键;`curl` 级断流即可):

```
turn 1  受理 …prompt=5850 cached=0        ← 中断(建流后 3 个 chunk 断)
[WARN]  [executor] 事件投递失败(请求方已断开 ⇒ 本 turn 弃置): channel closed
turn 2  受理 …prompt=5865 cached=0        ← 同前缀 + 新增一句
[kvmem] 复用面:matched=1 m_raw=1 …
指标:   ttft=2395ms; prefill=2388ms/5833tok; cache=32/5865 (0.5%)
```

`matched=1` 是**上一个无关会话**的首块(系统提示同内容),即本会话的整段前缀
在树里根本不存在。对照:同一 prompt **正常收尾**的 turn,下一轮
`cache=99.8%`、prefill 9 tok。

### 根因(单点)

`store_release` 的 H3 链登记原来**只给 `ReleaseKind::TurnEnd`**:

```rust
// 旧:正常 turn 收尾才发布链;abort 的半途链不进缓存
if release.kind == ReleaseKind::TurnEnd { let up = register_chain_owned(&s.tokens, ...)?; }
```

而会话账(`s.tokens` / `s.cached_len`)只在 `commit_turn` 里推进,`Abort` 不跑
`commit_turn`(见 `burn_ephemeral` 注释:引擎错误后账面不可信)。于是

- KV 里**已经有** `[0, fed)` 的真内容(prefill 全量 + 若干生成 token);
- 但树里**没有**这些内容身份;
- `release_turn_table` 把表引用归零后,这些页变成**无身份的冷页**,
  下一轮 `match_prefix` 看不到它们 ⇒ 全量重算。

中断恰好在"生成中"发生时最痛:这一轮 prompt 通常是 agent 场景里最大的一笔
(工具回包/粘贴材料),白算一次就是 2 s 量级。

---

## 二、修法

### 1. 中断落账(`exec/turn_end.rs::ledger_written`)

在 `Release(Abort)` **之前**把"已写入"的前缀写回会话账(表此刻还没归还 ⇒
树才有窗口对它 `incref`):

- 口径 `written = min(fed, prompt_len)`:`fed` 是 KV 推进指针,KV 有效域恒为
  `[0, fed)`,prefill 每 chunk / decode 每步都推进它 ⇒ 中断在 **prefill 中段**
  也只登记真的算完的那一截;
- 位置账 / 图像段账**同源截短**(`ledger_prefix`),否则下一轮 `unpad` /
  leave-gap 会按错坐标算;
- 位置账长度对不上 = 结构错 ⇒ 清空(leave-gap 自会按"位置账缺失"跳过),
  不写错值。

### 2. 发布口放行(两种终局都发)

```rust
if release.publish && self.dev.pool.geom.paged { … register_chain_owned(…) }
// 盘索引 / 空转拍仍只给 TurnEnd:中断路径要便宜
```

`Release` 新增 `publish: bool`(**不是**把 `Abort` 一律放行):客户端中断 =
`true`,引擎错误 = `false`(`abandon(err)` 路径账面不可信,发布错内容 =
静默错值)。判据来源 = `AbandonPublish`(`WrittenPrefix` / `Skip`)。

### 3. **生成段故意不入账**(这一条是踩出来的)

第一版把 `prompt + out` 全写回账本(语义上 KV 里确实有),结果第二次中断后
命中面**又归零**,日志给出的线索是:

```
[img-evict] leave-gap 会话拒绝跨会话复用:session=0x… turn=9 tail_owner=Some(0x…) ⇒ Fresh
```

链路:`positions_after_splice` 的 gap 判据是
`deletion.is_some() || tail_start < old_tokens.len()` —— 账本带生成尾巴 ⇒
下一轮客户端**丢弃**了那段生成文本 ⇒ 新 prompt 与旧账在 prompt 末尾分叉 ⇒
`tail_start < old_tokens.len()` ⇒ `positioned = true` ⇒
"同 token 不同 position 不许跨会话复用" ⇒ 命中面归零。

⇒ 账本只留**已确认的 prompt 前缀**;生成段那 ≤ 几个页不入缓存(客户端本来
也不重发它们)。修后 `positioned` 保持 `false`,跨会话复用照常。

---

## 三、实测(8135,双 3080 TP2)

三段对照,同一 prompt 形状(5850 → 5865 tok):

| 路线 | 复用面 | prefill | TTFT | `cache` |
|---|---|---|---|---|
| 修前:中断后再提交 | 1 块 | 5833 tok / 2388 ms | 2395 ms | 0.5% |
| 修后:中断 → 同会话再提交 | 182 块 | 41 tok / 179 ms | **187 ms** | **99.3%** |
| 修后:中断 → **跨会话**再提交 | 183 块 | 9 tok / 122 ms | 130 ms | 99.8% |
| 基准:**正常收尾**再续写 | 183 块 | 9 tok / 123 ms | 130 ms | 99.8% |

⇒ 中断路线与正常收尾路线的**复用面完全一致**(183 块),跨会话也成立
(内容寻址,不靠会话守卫)。

### 语义门(不是"能跑出字")

深埋 token 级探针:pin 放在 ~4000 tok 处的口令 `QX-7731-PLUM`,中断后再问
"口令是什么" —— 3/3 命中,且三次输出**逐字一致**(KV 与 GDN 快照都真的对)。

> ⚠️ 不要用"两臂逐字节一致"当质量门:同一 prompt 走**不同复用面**
> (182 vs 183 块)或不同快照捕获路径(块边界快照是 prefill 拍的还是
> decode fold 拍的)时,首 token 的 greedy 选择会出现**合法漂移**
> (实测残差 3e-3 量级即可翻字)。判据应是:复用面一致 + 深埋检索命中
> + 输出连贯。

### 单测门(CPU mock,`abort_publishes_written_prompt_prefix`)

`crates/engine/src/tests.rs`:`A` 在生成中 `cancel_turn` → `B`(不同会话键)
= A 的 prompt + 尾巴,机械断言:

```
[test] page=32 A prompt=307 tok / A chunks=9 / B chunks=1 · B 复用面=288 tok
```

反证(把 `publish` 强置 `false`):`B 复用面=0 tok / B chunks=10` ⇒ 门会红。

---

## 四、边界与遗留

- **引擎错误弃置仍不发布**(`abandon(err)` → `AbandonPublish::Skip`):
  那时账面不可信,发布 = 静默错值。
- **未落盘的收益**:中断那一轮的生成段(≤ 页级)不入缓存;若前端把中断的
  部分输出**保留进历史**(某些 UI 会),这段就退化成"重算几个 token"。
- **rebind 不改 owner**:`register_chain_owned` 只给 `up.inserted` 记归属,
  `rebound`(同内容写新槽)不改 —— 现在只影响 `positioned` 会话的
  `leave-gap` 跨会话判据(偏保守,安全),不是本 bug 的一部分,留案。
- **备选更彻底的形态**(未做):把"逐块登记"从 `select_partial` 档解锁到
  所有档(即 vLLM 的"写一块、登记一块"),那样**任何**异常中断(含进程被
  杀前)都自然有缓存;代价是每块一次哈希 + 缓存页更早进"不可沉"面。
  本刀先按 turn 终局补账,足够覆盖客户端中断这一现实场景。

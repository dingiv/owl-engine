# 投机播种裸步:预产草稿错位 + 分母重采风暴(冷启动 AL 损失)—— 立案与结案(2026-11-12)

> 入口:11-09 交接卷 §5.1 留下的独立立案 —— **启动后第 1 次请求 AL 固定低 ~1.1**
> (两轮 boot 同一签名),当时把它归到"draft 侧预热状态 / 首请求图形状"。
>
> 结案 = **两个叠加缺陷**,都在调度面上,不在草稿核:
> ① **播种裸步吃掉预产草稿**:上一轮 ⑦ 按"本步做 verify"预产的 `spec_drafts_host`,
> 本步改走裸 decode ⇒ 位置已被吃掉;下一轮照旧 `take()` ⇒ 整体错位一格 ⇒ **必然 m=0**
> (2026-11-03 修过降级裸步的同款,播种裸步漏了);
> ② **分母重采风暴**:裸步成本的 ctx 配对门只有相对门(`pos > ctx*1.25`)且**在播种中也会
> 触发** ⇒ 短 ctx 下每 20–30 token 重新播种一次(`48 → 62` 就是 +29%),而 `seed_due`
> 又要求 `ewma == 0` ⇒ 文档写的 2 样本从未采完,永远是 1 样本后作废重来。
>
> 修后(双 3090 / 27B AWQ / DFlash2 d7,3 次独立 boot):
> **冷启动 AL 6.50 → 7.91**(热态 7.97),轮数 **38 → 32**,输出 sha 逐字节不变;
> 残留差 = 2 个播种裸步(≈36 ms)+ 冷 prefill,不再是 m=0 轮。

档位:双 3090 Ti / TP2 / 27B AWQ-INT4 / DFlash2 d7（本地档，`[parallel]` 内联）,8136。

---

## 一、现场数据(先证据,再推断)

口径:`llm_speedtest` 同族计数任务(`Count from 1 to 500, one number per line...`),
system 固定、`temperature=0`、thinking off、`max_tokens=256`,同一进程连发两次。

| | 冷(启动后第 1 次) | 热(第 2 次) |
|---|---:|---:|
| spec 轮 | **38** | 32 |
| accepted | 209 | 223 |
| AL(`1+accepted/rounds`) | **6.50** | 7.97 |
| m 直方图 | `m0=7, m2=1, m4=1, m7=29` | `m6=1, m7=31` |
| 输出 sha(16) | `25f47191…` | `25f47191…` |
| decode | 754 ms/255 tok | 719 ms/255 tok |

**输出逐字节相同** ⇒ 不是内容/轨迹差异(11-09 那轮已经证明是 prompt 渲染变量),
是草稿面在特定轮次的真实退化。逐轮 debug 日志(`spec-round` + `spec-seed`):

```
[spec-seed] 分母未测 ⇒ 本步发播种裸 decode(还欠 1 步)
[spec-seed] 分母已播种:裸步成本 = 18.0ms(pos=48)
[spec-round] acquire pos=49 src=eager
[spec-round] 接受完成 m=4
[spec-round] acquire pos=54 src=stored
[spec-round] 接受完成 m=7
[spec-seed] 分母未测 ⇒ 本步发播种裸 decode(还欠 1 步)     ← 25% 门又触发
[spec-seed] 分母已播种:裸步成本 = 17.2ms(pos=62)
[spec-round] acquire pos=63 src=stored                     ← 消费播种前预产的草稿
[spec-round] 对齐 pos=63 m=0 drafts=[198,24,198,16,…] ids=[24,248046,198,16,…]
[spec-round] acquire pos=64 src=stored
[spec-round] 接受完成 m=7                                  ← 一错即自愈
…(pos 80 / 106 / 140 / 182 / 232 / 298 再各来一次)
```

`spec-seed` 在首个 turn 里出现 **8 次**(不是设计里的 2 次),每次后面紧跟着恰好一轮
`m=0`、再下一轮满接受。热态(第 2 次请求)一次都没有 ⇒ 这就是"冷启动"的全部来源。

## 二、根因

### 2.1 裸步不再产出草稿,但草稿账没跟着作废

`spec_drafts_host` 是**上一轮 ⑦ 为本轮预产**的草稿(位置 = 上一轮 `pos+m+1`)。
播种裸步(以及降级裸步)只做一步双卡 decode + `prefill_dflash_encode`(1 行),
**不产草稿**;但下一轮的 `acquire_drafts` 只看 `spec_degraded`:

- 降级裸步:2026-11-03 已修(`degraded_now ⇒ spec_drafts_host = None`),
- **播种裸步:没修** ⇒ 下一轮照旧 `take()`。

裸步已经吃掉了那个位置 ⇒ 草稿整体错位一格 ⇒ 第一枚草稿必错 ⇒ `m=0`。
对齐日志的指纹很干净:`d[i] ≈ ids[i-1]`(草稿前移)/首枚 = 上一轮的换行 token。

### 2.2 分母重采门:短 ctx 下的风暴 + 2 样本永远采不完

`bare_cost_ewma_ms`(裸步成本)是 B4 盈亏线的分母;它随 ctx 上行
(实测 24.1 ms@564 → 41.7 ms@32k),所以设计上"ctx 涨过 25% 就作废重采"。
但旧实现:

```rust
if a.bare_cost_ctx > 0 && pos as f32 > a.bare_cost_ctx as f32 * 1.25 {
    a.bare_cost_ewma_ms = 0.0;
    a.bare_seed_left = SPEC_SEED_SAMPLES;   // 2
}
let seed_due = a.bare_cost_ewma_ms <= 0.0 && a.bare_seed_left > 0;
```

- 播种中也会进这个 `if` ⇒ 第 1 个样本把 `ewma` 置非零、`bare_cost_ctx` 更新到当前
  pos;下一次 25% 增长又把它清零重来 ⇒ **`SPEC_SEED_SAMPLES = 2` 名不副实**
  (永远只采到 1 个样本);
- 只有相对门 ⇒ 短 ctx 下 25% 只有十几个 token(48 → 62 就是 +29%)
  ⇒ 首个 turn 里反复播种,每次打掉一轮 verify。

### 2.3 为什么热态没事

`bare_cost_ctx` 跨 turn 保鲜(`carry_ctx`),第 2 次请求的起点在 `carry_ctx*1.25`
之内 ⇒ 不播种。所以缺陷的可见面 = **每次 ctx 跨过重采门后的那一轮**,最集中的
就是启动后第 1 个 turn(从 0 起涨)。

## 三、修法(三处,都不碰草稿核/verify)

1. **裸步即作废预产草稿**(`sched/scheduler.rs` 播种分支):
   `spec_drafts_host = None; spec_q_host = None;`
   与降级裸步同一条语义;下一轮走当步 eager propose(`src=eager`)。
2. **重采门做成纯函数** `exec::bare_cost_rearm(pos, ctx, ewma_ms, seed_left)`:
   - 相对门 `pos > ctx * 1.25` **且** 绝对下限 `pos - ctx >= 4096`
     (≤4k 时裸步成本接近平坦,频繁重采只亏不赚);
   - `seed_left > 0`(播种中)一律不重采;
   - `ewma == 0` / `ctx == 0`(未测)不重采 —— 初始播种由 turn 初负责。
3. **播种欠账只看 `bare_seed_left > 0`**(去掉 `ewma <= 0` 的第二个门)
   ⇒ 2 个样本背靠背采完,EWMA 用既有 α=0.15 混合,不再"采一个就作废"。

顺带补了两条 debug 诊断面(零成本,仅 `--log-level debug` + tag 过滤时开):
`spec-round` 的逐位对齐行(`drafts` vs `ids`)与 `acquire src=stored|eager`。

## 四、验收

### 4.1 真机门(双 3090 Ti / 8136,3 次独立 boot,同一脚本)

| boot | 冷 AL | 冷轮数 | 热 AL | 冷/热输出 sha |
|---|---:|---:|---:|---|
| 1 | 7.91 | 32 | 7.97 | 同 |
| 2 | 7.91 | 32 | 7.97 | 同 |
| 3 | 7.91 | 32 | 7.97 | 同 |

冷态 `m0 = 0`;唯一非满轮是预算尾(`m4`)。wall:冷 0.90–0.93 s / 热 0.79–0.81 s,
差值 = 冷 prefill(48 tok,~78 ms)+ 2 个播种裸步(~36 ms),decode 步时
`spec.round` 冷热都是 22.5 ms(没有隐藏的预热税)。

### 4.2 长 ctx 重采不受影响

4.2k prompt(跨过 4096 绝对下限):重采只发生 1 次(2 个裸步),AL 7.62 / 8 轮,
无连播;`spec.degrade = 0`。

### 4.3 单测

`cargo test -p owl-engine --lib bare_cost_rearm -- --test-threads=1` ⇒ 4/4:
短 ctx 不重采 / 长 ctx 保留相对门 / 播种中不重采 / 未测不重采。

## 五、边界与未做

- **不是 draft 质量问题**:同一 token 流上 draft 健康(11-09 门 B),本次不改
  selector/verify/encode;11-09 的 `reasoning_effort` 结论仍然成立。
- **播种成本是设计成本**:首 turn 2 个裸步 ≈ 36 ms / 0.06 AL;它的收益是让 B4
  盈亏线有实测分母(替代旧兜底常量 2.5)。要再省只能改成"首窗结束时才采"或
  把播种挪到 boot,属于策略变更,不在本卷。
- 降级态探测/恢复(2026-11-03 两卷)语义未动;降级裸步的草稿作废是既有行为。

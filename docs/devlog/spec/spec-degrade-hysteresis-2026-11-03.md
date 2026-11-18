# 投机降级:降了再也不回头(B4 三态)+ 降级门槛太敏感 —— 立案与结案(2026-11-03)

> 用户报:**降速(降级裸 decode)频繁触发,而且一旦降了就再也不回投机** ——
> 要求"提高降速门槛(容忍性)"+"必须留反悔机会"。
>
> 结案 = **两个独立缺陷**:
> ① **真 bug**:降级态探测轮消费的是**8~64 步之前的陈旧草稿** ⇒ 探测必然 m=0
> ⇒ `spec.undegrad = 0`(真机 53 次探测全灭),turn 内永久裸 decode;
> ② **判据太敏感**:**一个**低收益窗口(16 轮 / 600 ms)就降级,而代价是
> 不对称的(误降 2.1× vs 误留 1.24×)。
>
> 修后同一形状:解码 **83.4 → 116.8 t/s**,降级步 **758/2399 → 0**;
> 旧门槛下同一负载 3 次降级 **3 次全部自愈**(修前 0/53)。

档位:`profiles/qwen38_27b-3080_tp2.toml`(双 3080 / TP2 / 27B AWQ-INT4 / DFlash2 d7),8135。

---

## 一、现场数据(先说证据,再讲推断)

`/debug/metrics`(旧进程,跑过一段 101k prompt 的 agent 会话):

| 计数 | 值 | 读法 |
|---|---|---|
| `spec.degrade` | **1** | 只降了一次 |
| `spec.undegrad` | **0(整表里根本没这个 tag)** | **一次都没恢复** |
| `spec.deg.rounds` | 53 | 降级期只发过 53 轮探测 |
| `decode.step.deg` / `decode.step` | **3238 / 3241** | 裸步 3238 = 全是降级步 |
| `spec.round` | 2218 | 正常投机轮次 |
| `spec.m0` | 714(32%) | 普通轮里 m=0 只占三成 |

对应 turn 的诊断行:

```
turn 3: decode=113222ms/6445tok=56.9 t/s · spec rounds=1204 accepted=2003 al=2.66 · deg=3238/6445
turn 4: decode=10259ms/1174tok=114.4 t/s · spec rounds=305  accepted=871  al=3.86 · deg=0/1174
```

⇒ 同一个会话、相邻两 turn:**一半的解码步在裸跑,吞吐腰斩**。

关键矛盾:**普通轮 m=0 占 32%,而 53 次探测**全部** m=0**。按 0.32^53 算是
统计上不可能事件 ⇒ 探测轮拿到的草稿不是"当步的草稿"。

---

## 二、根因①:探测轮消费陈旧草稿(真 bug)

### 机理

`spec_drafts_host` 的语义是"**上一轮 ⑦ 为本轮预产的草稿**"——位置 = 上一轮结束
时的 `fed`。正常态每轮紧挨着消费,没问题。

但**降级态**只在 `probe_every`(8→16→32→64)步之后开一轮探测,**中间隔着 8~64 枚
裸 decode**,而 `acquire_drafts` 照旧 `take()` 那份草稿:

```rust
// 旧
let drafts = if let Some(mut d) = self.tspec()?.spec_drafts_host.take() { d }  // ← 8~64 步前的预测
             else if DFlash2 { eager propose(token, pos) }                    // ← 本该走这条
```

更糟的是 ⑦ 在**探测失败后仍然继续产草稿**(`if !round_done { propose … }`),
于是下一次探测又拿到同样陈旧的一份 —— 自我维持。

### 取证(靠新加的探测轮打点)

修前(降级期探测轮,`d` = 草稿 7 枚、`ids` = 当步目标验证行 8 枚):

```
pos=1768 m=0 d=[198, 2719, 16, 19, 318, 16, 18] ids=[12, 12, 16, 19, 318, 16, 18, 16]
pos=2208 m=0 d=[17, 17, 318, 17, 16, 16, 12]   ids=[6154, 17, 15, 17, 16, 16, 12, 17]
```

读数:`d[0]` **恒不等于** `ids[0]`,但 `d[2..] == ids[2..]` / `d[4..] == ids[4..]` ——
草稿本身是"这段文本的合法续写",只是**位置落后了若干 token**。对照同 turn 的
正常轮(修后同一打点):

```
pos=133 m=4 d=[4087, 1156, 579, 1622, …] ids=[4087, 1156, 579, 1622, 303, …]
```

逐位对齐、m=4 正常接受。⇒ 草稿器没坏,**是账错了**。

### 修法

1. `acquire_drafts`:**降级态一律丢弃** `spec_drafts_host`,改走当步 eager propose;
2. ⑦:`degraded_now` 时**不产草稿**(省一次 propose 税,也不再留陈旧账);
3. 降级/恢复时把 `spec_steps_degraded` 归零(旧值可能是上一次降级 episode 的残留
   ⇒ 降级第一拍就白探一次)。

### 验证(A/B:门槛临时改回旧口径,隔离出恢复通路)

同一 prompt,把 `DEGRADE_CONFIRM_WINDOWS` 临时置 0 + 去掉连击的窗口要求
(即"旧门槛"),只保留上面的草稿修复:

```
[WARN] 降级裸 decode(probe 每 8 tok):m=0×10 … → 降级
[WARN] 探测命中 m=5 → 恢复 spec(降级步 8 / 探测周期 8)
[WARN] 降级裸 decode … m=0×7 → 降级
[WARN] 探测命中 m=3 → 恢复 spec(降级步 16)
[WARN] 降级裸 decode … m=0×6 → 降级
[WARN] 探测命中 m=7 → 恢复 spec(降级步 24)
turn: decode=21994ms/2399tok=109.1 t/s · deg=24/2399(1%)
```

**3 次降级 3 次自愈**;**修前同一档位 = 0/53**。降级步从 758 掉到 24。

---

## 三、根因②:门槛太敏感(确认制)

旧判据 = **一个**窗口(`16 轮` 或 `600 ms`)`AL+1 < 盈亏线` 就降级。真机现场是
"整体健康的 turn 里夹着几段 16 轮全 m=0"(工具调用 / JSON 结构 / 随机词列表),
于是反复降-试探-恢复。

代价是**不对称**的:

| | 每 token 成本 | 相对裸 decode |
|---|---|---|
| 健康档走 spec | ≈ 9.8 ms(26 ms/轮 ÷ AL 2.66) | **0.47×**(赚 2.1 倍) |
| 死档留在 spec | ≈ 26 ms/轮 ÷ 1 token | 1.24×(亏 24%) |

⇒ 误降的罚是误留的 **8 倍**。据此改成**确认制**(纯函数
`exec::spec_degrade_verdict`,单测 `spec_degrade_verdict_requires_confirmation_and_can_recover`):

1. **两个连续低收益窗口**才降(≈1.2 s 实证);
2. **连击只减半确认**:`m=0×N` 达阈值 **且** 已有 ≥1 个低窗 **且** 部分窗 AL 也低于
   盈亏线 —— 连击不再单独构成依据;
3. 健康窗口把连续计数清零(滞回);
4. **探测命中(m≥1)必须恢复** —— 除非恰好凑满两个连续低窗(证据压过单次命中);
5. `DEGRADE_PROBE_MAX` **64 → 32**:探测是保险不是惩罚(一轮探测摊到 32 token
   ≈ 1 ms/token = 5%,而"该恢复还在裸跑"是 2.1×)。

---

## 四、实测汇总

| 场景 | 解码 | 降级步 | 备注 |
|---|---|---|---|
| 旧代码(repro:随机词清单 2400 tok) | **83.4 t/s** | 758/2399(32%) | 一降到底,0 次恢复 |
| 旧代码(生产 agent turn,101k prompt) | **56.9 t/s** | 3238/6445(50%) | 同会话相邻 turn 健康态 114.4 t/s |
| 草稿修复 + 旧门槛(A/B) | 109.1 t/s | 24/2399(1%) | 3 降 3 恢复 |
| **草稿修复 + 确认制(终态)** | **116.8~120.0 t/s** | **0** | 两个不同 repro 都不再降级 |

样本(temperature=1.0 同 prompt):无降级、AL 3.46、115.3 t/s。

`cargo test --workspace --lib` **976 全绿**;clippy 零 lint。

---

## 五、现在的日志面(排障入口)

降级期每次探测打一行 info(此前完全没有这一面,是本次定谳的关键):

```
[spec-probe] 探测轮(周期 32 tok)⇒ 未命中,继续裸 decode(明细 pos=… m=0 d=[…] ids=[…])
[spec-degrade] 降级裸 decode(probe 每 8 tok):m=0×10 连续低窗=2 盈亏线=1.35 因=双窗确认 窗口AL=1.00
[spec-degrade] 探测命中 m=5 → 恢复 spec(降级步 8 / 探测周期 8)
```

判读口诀:`d[i] ≈ ids[i-k]` = 草稿位置账错(k>0 的滞后);`d` 与 `ids` 完全无关 =
草稿分布质量差(该降);`d == ids` 前几位而偶发中断 = 正常边界。

---

## 六、遗留

- **真死域的确认延迟**:`1.2 s`(两窗)是刻意取的容忍度。若某类负载的"死段"
  短于 1.2 s 但很频繁,spec 会一直留在"略亏"区(1.24×)。可考虑把窗口切成两档
  (短窗看 AL 断崖 + 长窗看均值),现阶段不做。
- **turn 边界**:`spec_degraded` 是 turn 域状态 ⇒ 下一 turn 自动从投机态起(本次
  的"永不回头"只发生在 turn 内)。跨 turn 的分母(`bare_cost_ewma_ms`)保鲜规则不变。
- **A/B 只能靠单元门**:本机 30 秒内没能造出"≥1.2 s 真死域"的负载(随机词/随机
  hex/temperature=1 都被草稿器扛住了)⇒ "两个连续低窗必降"由纯函数单测钉住,
  真机只验证了"不再误降 + 探测必恢复"。

---

## 七、2026-11-18 回归:计数器只在同步分支自增 ⇒ 降级永不探测

2026-11-18 真实 agent 场景复现:**同一 turn 前段 ~100 t/s,decode 中段跌到 ~50 t/s
并一直保持**;日志 `spec.degrade=8 / spec.undegrad=0 / spec.probe.round=0`,
每个 turn 只记 1 个 `decode.step.deg`,`deg` 只占 1/1584(实际约 2/3 的 token
在裸 decode)。

根因不是 11-03 的陈旧草稿(那条仍在修),而是后来加入的同步节流:

```rust
// 旧(错):计数器只在“同步草稿池”分支里自增
if (degraded || seeding) && sync_due && taps > 0 {
    ...
    if degraded { spec_steps_degraded += 1; spec_degraded_steps += 1; }
}
```

`sync_due` 本身由 `spec_steps_degraded % 8 == 0 || steps + 1 >= probe_every` 判定
⇒ 第一次同步后计数 = 1,之后 `1 % 8 != 0` 且 `1 + 1 < probe_every` ⇒ **永远不再
同步、也永远到不了探测点**。调度面 `spec_steps_degraded >= spec_probe_every`
因此恒假,降级态一次探测都不发,只能整段裸 decode。

修法(2026-11-18):

- `spec_steps_degraded` / `spec_degraded_steps` **每个降级裸步都自增**(与同步解耦);
- 同步仍按 `degraded_sync_due`(每 8 步 + 探测前最后一步);
- 调度面探测判据抽成 `degraded_probe_due`,与 `degraded_sync_due` 同源;
- `decode.step.deg` 改为每降级步记账(此前只在同步步记,`deg` 被系统性低估)。

真机验证(双 3080 / 8135 / 同一 prose+thinking 负载):

```text
修前: degrade=8  undegrad=0  probe.round=0   decode 50.0 t/s
修后: degrade=3  undegrad=2  probe.round=31  decode 61.0 t/s(prose,AL≈1.3~2.1)
```

另测高 AL 对照:

```text
1..1200 数字(短 ctx): AL 7.99 / 292 t/s / deg 0
147k 长文档 + 1..600: AL 7.68 / 148.8 t/s / deg 0
```

⇒ 修复恢复的是**降级后的反悔通路**,不是把低 AL 内容变高。真实 agent 的
thinking/prose 段 AL 仍会从 ~7 掉到 ~1.3,那是草稿域质量(模型侧),不是位置/
长 ctx 结构 bug;`spec` 在该段只能贴近盈亏线,裸 decode 约 47 t/s。

### 7.1 vLLM 交叉验证(2026-11-18)

同模型/同量化/同 DFlash2 d7,换 vLLM 0.27.2-dev + 双 3090 Ti TP2(8133),
同一句“凌晨四点的便利店”prose prompt,`max_tokens=1000`:

```text
vLLM: 952 tok / 18.39s = 51.8 t/s
  0-100 tok: 78.4 t/s    100-200: 57.2    200-300: 53.0
  300-400:   45.1        400-900: 40.5~56.8
  AL 窗口: 0-2s 2.80 → 2-6s 1.74 → 6-18s 1.61/1.62
```

对照“只输出 1..1000 数字”(thinking off):

```text
vLLM: 1200 tok / 5.37s = 223.4 t/s
  每 100 tok: 203~277 t/s(首桶含 warmup)
  AL 全程恒 8.0(7 草稿全收)
```

⇒ **vLLM 也复现同样的 AL/速度衰减**,且数字任务同样不衰减。这排除
“owl 降级计数器”是衰减的唯一原因;计数器 bug 只负责把低 AL 段**永久锁死**
在裸 decode。prose/thinking 段的 AL 从 ~7 掉到 ~1.6 是 DFlash2 草稿的
域质量(模型侧),不是长 ctx / decode 位置 / owl 特有结构。

### 7.2 147k follow-up 的功耗/吞吐 A/B(2026-11-18)

同 3080 双卡、同 147,904-token 长文档、同 320-token 输出、同 DFlash2 d7：

| 引擎/档 | decode | 每卡平均功率 | J/token |
|---|---:|---:|---:|
| vLLM 0.27.2 + FP8 KV + W4A16 | **93.9 t/s** | 256–260W | **5.49** |
| owl W4A8 + INT8 KV(主档) | 60.5 t/s | 259–260W | 8.58 |
| owl W4A8 + F16 KV | 73.2 t/s | 257–259W | 7.06 |
| **owl W4A16 + F16 KV** | **78.7 t/s** | 261–263W | **6.67** |

定谳：**功耗不是主问题**。147k 下 owl 与 vLLM 都是 ~260W；owl 是
“同功率、少 20~35% token”。第一杠杆是 **KV dtype**：int8 主池在长 ctx 的
verify 读核明显吃亏，f16 换回 +30% decode；第二杠杆仍是 spec round：

```text
owl W4A16+F16: round 35.8ms = verify 27.3 + fold 5.7 + propose 1.8
vLLM 同 AL 2.86 ≈ 30ms/round
```

⇒ 剩余 ~5.8ms 主要在 fold/verify；不再动草稿模型。f16 池 150k 已占
~4.7GiB，不能直接拉到 262k；262k 仍走主档 int8。

新增落档：`profiles/qwen38_27b-3080_tp2-150k.toml`(W4A16 + F16 KV +
`max_context_length=150000` + 单并发)。FP8 KV 当前会因缺
`owl_qknorm_rope_kv_insert_f16_fp8kv_knhd` 准入声明而拒启，暂以 f16 桥接。

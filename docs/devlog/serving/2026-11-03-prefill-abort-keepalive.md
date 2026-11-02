# 长 prefill 期间客户端断流:引擎照旧跑完 prefill —— 立案与结案(2026-11-03)

> 用户报:**引擎陷在长 prefill 里,前端断开连接后,它还是把整段 prefill 跑完**。
>
> 结案 = **两个缺陷**,第二个是修第一个之后才暴露的:
> ① 流式应答在 prefill 期间**一个字节都不写** ⇒ 手写 HTTP 的 chunked 写永远不会失败
> ⇒ 对端 FIN 无人察觉(实测:198495 tok 的 turn 断流后仍 prefill 到 100%);
> ② 中断点的链发布了(前一刀的 `中断落账`),但**边界上没有快照**(本档非虚拟化,
> 只有"末边界"才拍)⇒ 下一轮 `m_raw=1` ⇒ 中断前算完的几万 token **白算**。
>
> 修后:断流 **≤1s(保活)+1 个 chunk** 内停;**已算完的前缀可被下一轮复用**
> (实测 18464 tok 全复用,修前 `m_raw=1`)。

档位:`profiles/qwen38_27b-3080_tp2.toml`(双 3080 / TP2 / 27B / DFlash2 d7),8135。

---

## 一、根因①:prefill 期间流上零写入 ⇒ 断流不可见

服务面是**手写 HTTP/1.1 + chunked**。断流唯一的暴露路径 = **写失败**
(`emit!` 宏 → `sink_rx.close()` → `abort_turn`)。而 prefill 阶段:

```
TurnEvent::Prefill { total, .. } => { usage.prompt = total; }   // ← 只改账本,不写字节
```

⇒ 一个 100k prompt 的 prefill(≈50 s)里,连接上没有任何写操作 ⇒ 客户端 FIN 躺在
socket 里没人读 ⇒ 引擎跑完整段,直到**首个 token 写失败**才弃置。现场日志(旧代码):

```
[WARN] [executor] 事件投递失败(请求方已断开 ⇒ 本 turn 弃置): channel closed
[INFO] [prefix] 中断落账:… prompt 前缀 198495 tok 入账          ← 整个 prompt 都算完了
```

### 修法:静默保活帧(`[runtime] keepalive_ms`,缺省 1000)

- 流上超过 `keepalive_ms` 没写过字节 ⇒ 写一帧 **SSE 注释**(`: keepalive\n\n`,
  chunked 封装,`http::sse_comment`);
- 写失败 = 连接已死 ⇒ 与 `emit!` 同一处置:先 `sink_rx.close()`(让 executor 下一笔投递
  失败),再指名 `abort_turn` + 等回执 ⇒ **prefill 中段也能停**;
- `0` = 关(逃生门:某些自研客户端若对注释帧过敏,把档里这一项设 0 即可)。

**规范依据**:SSE 要求解析器**忽略 `:` 开头的行**(官方 openai-python 的 `SSEDecoder`
对"注释行 + 空行"返回 `None`,不会派发空 data 事件);OpenAI / Anthropic / OpenRouter
在长思考期用的都是同一手法。`data:` 帧的语义与顺序一字未改。

### 踩坑:保活定时器会被 `Prefill` 事件饿死(第一版实测无效)

第一版把保活放在 `select!` 分支里,并在**每次收到事件**时重置计时。`Prefill` 每
chunk 就来一笔(0.7~1.2 s),而它**不写字节** ⇒ 计时被无限推后 ⇒ 28k prompt 的
prefill 期间**一个保活帧都没发**(客户端 10 s 只收到 105 B = SSE 头)。

修法两条(缺一不可):

1. **到点即发**:保活判断放在**等事件之前**(事件流再密也不能把它饿死);
2. **只有真写字节的事件**才重置计时(`Token`/`Completed`/`Failed`;`Prefill`/`Session`
   只改账本)。

修后实测收包时刻:`0.0, 1.1, 2.1, 3.1, 4.1, …`(每 ~1 s 一帧,直到断流)。

---

## 二、根因②:中断点的链发布了,但没有边界快照

前一刀(`中断落账` + `Release::publish`)把"已写入的前缀"发布进内容树。但
`reuse_face` 还要求**边界上有 GDN 快照**,而本档(非虚拟化)的快照节奏是
`last_boundary || 每 SNAP_EVERY 个边界 || 消息边界` —— 客户端在 prefill **中段**
断开时既不是末边界、也未必撞上锚点 ⇒ 实测:

```
[kvmem] 复用面:matched=769 m_raw=1      ← 链匹配 769 块,复用面塌到 1 块
```

⇒ 中断前算完的 24k token 下一轮全部重算。

### 修法:中断点补拍(`capture_abort_snap`)

- `cancel_turn` 在 **`take_active` 之前**(补拍要用活跃 turn 的 GDN 行/位置)判:
  `fed < prompt_len`(仍在 prefill)且 `fed` 页对齐 且 前缀缓存+快照池在场 ⇒
  `capture_snap_at_cursor(Durable)`;
- `capture_gdn_snap_tail` 的"拍快照"半抽成 `capture_snap_at_boundary`(**单一产地**:
  正常边界与补拍共用槽位选择/双 rank 镜像/登记账簿/保护面),补拍走"当前游标链尾"
  (正常路径的幂等门是"同一 fed 不重复拍",事后补拍必须绕过它);
- 补拍失败只 WARN(它是加速面,不挡收摊)。

实测(28k prompt,客户端 6 s 断开 → 78k prompt 重提):

```
[snap] 中断补拍:边界 577 块 的快照已就位(中断前的前缀可复用)
[prefix] 中断落账:prompt 前缀 18464 tok 入账
[kvmem] 复用面:matched=577 m_raw=577 尾快照层=Some(Gpu)      ← 18464 tok 全复用
```

---

## 三、实测汇总

| 场景 | 旧行为 | 新行为 |
|---|---|---|
| 28k prompt,6 s 断开 | 跑完整段(28203 tok) | 停在 **18464 tok**(65%),≤1 s 检出 |
| 78k prompt(前缀复用),4.1 s 断开 | 跑完整段 | 停在 **28704 tok**,并补拍 897 块快照 |
| 下一轮同前缀请求 | `m_raw=1`(全量重算) | **`m_raw=577`(18464 tok 复用)** |
| 正常收尾 turn 的下一轮 | cache 99.3% | cache **99.3%**(未回归) |

检出延迟 = 保活间隔(≤1 s)+ 一个 prefill chunk(≈0.7~1.2 s)。

`cargo test --workspace --lib` **976 全绿**;`cargo clippy --workspace --all-targets`
零 lint(顺手把 `emit_items` 的既有 8 参 lint 按"站点级 allow + 理由"收口)。

---

## 四、配置面

```toml
[runtime]
keepalive_ms = 1000   # SSE 静默保活(ms;0=关):长 prefill 期间探客户端断流
```

12 个档(入口档 + measurement + local)已按"完全显式字段"纪律补齐;缺字段会被
完整度门拒启(`runtime.keepalive_ms`)。

---

## 五、遗留

- **非流式(JSON)应答仍无法探断流**:整段 prefill + 生成期间一个字节都不写,只有
  最终 `respond_json` 才知道对端在不在。要治需另立机制(连接级 EOF 探测 / 读半关闭),
  收益低(agent 客户端一律流式),暂不做。
- **补拍只在页对齐的 prefill 中段生效**:`fed % page != 0` 时没有边界身份可挂
  (prefill 的 chunk 尾通常页对齐,生成段走 Transient 逐页边界快照,不在此列)。
- **保活帧对客户端的要求**:必须按 SSE 规范忽略 `:` 注释行。若某自研客户端不守规范,
  把 `keepalive_ms = 0` 即回到旧行为(代价 = 本 bug 复现)。

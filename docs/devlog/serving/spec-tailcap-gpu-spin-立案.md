# DFlash2 尾批裁剪 × verify 图档 —— 设备核自旋立案(2026-10-24)

> 状态:**已修(2026-10-25,根因定谳 + 真机 2 boot / 双形状验证)**。
> 严重度:**P0**(整台引擎被一个 turn 钉死,只能重启;
> `max_concurrency=1` 时后续请求全部排队饿死)。修复见 §十。
> 影响档:`profiles/qwen38_27b-3090.toml`、`profiles/qwen38_27b-3080_tp2.toml`
> (凡 `spec.depth > 0` + 图档 decode 臂皆可能命中)。

## 一、现场

单 3090 Ti / 27B AWQ-INT4 + DFlash2 d7 + FI + CUDA graph,`max_seq=49152`:

```text
[DEBUG] [spec-round] pos=60 接受完成 m=7 bonus=17 emitted=8
[DEBUG] [spec-round] 轮开始 token=17 pos=68 depth=7 vlm_delta=0
[DEBUG] [owl-fi] [fi-plan] ctx_total=72 nb_pages=3 T=4 page=32
[DEBUG] [owl-fi] [plan] padded_batch=42 total_rows=4 split=1 ...
[DEBUG] [owl-fi] [mask] 图档 plan:T=4 ctx=72 padded=42 tiles=1 ⇒ 掩 41 个尾部 CTA
[DEBUG] [owl-fi] [plan] int_ws 激活(t=4 padded=42 tiles=1 cap=false)
        ← 此后再无任何日志,请求永不返回
```

同一时刻抓的外证(关键判据,决定是"核在跑"还是"主机在等"):

```text
nvidia-smi: GPU util = 100 %(连续 4 次采样恒定不变)、memory util = 42 %
进程线程:   主线程 + 全部 tokio worker = futex_do_wait / ep_poll
```

⇒ **不是主机死锁**:设备侧有核在自旋等一个永不到来的信号,主机在等价地等它。
显存没有被耗尽,池也没有压力腾槽(`N6 压力节点 = 0`)——与容量无关。

## 二、触发形状(实测矩阵)

同一台机、同一份档,只改请求的 `max_tokens` 与 `spec.depth`:

| prompt token | `max_tokens` | `spec.depth` | 结果 |
|---|---|---|---|
| 130 | **16** | 7 | **STUCK(30s+ 永不回)** |
| 130 | 256 | 7 | ok 1.9s |
| 3101 | **16** | 7 | ok 但 **24.0s**(只出 16 token) |
| 3101 | 256 | 7 | ok 3.6s |
| 21510 | 64 | 7 | ok 29.8s |
| 130 | 16 | **2** | ok 0.4s |
| 130 | 16 | **0** | ok 0.5s |

⇒ 触发轴 = **`depth` 大(=7)且"剩余预算 ≤ depth"**,即 `live_depth_for_budget`
进入裁剪的那几步;不是"短 prompt",也不是"贪心采样"(最初两个猜测都被证伪)。

裁剪函数本身是纯函数且有单测锚(`crates/engine/src/exec/spec.rs`):

```rust
/// rem=1 ⇒ P=0(裸 decode,调用方不应发 spec);rem>=2 ⇒ P=min(K, rem-1)
fn live_depth_for_budget(depth: usize, remaining: usize) -> usize {
    if remaining <= 1 { 0 } else { depth.min(remaining - 1) }
}
```

`rem=4 ⇒ live=3 ⇒ target_rows=live+1=4` —— 与上面日志里的 `T=4` 逐位吻合。

## 三、机理假设(待证)

注释写明设计意图是:**proposal 仍满 W=K+1,只有 verify 的 target 行/提交行裁到 P+1**
(`spec.rs` 轮内注释)。但现场显示裁剪同时把 **FI 的 plan 形状**换掉了
(`total_rows=4`,而正常轮是 8),并且**按"图档"路径**去掩尾部 CTA
(`掩 41 个尾部 CTA`)。

于是两条可疑链:

1. **图档形状错配**:CUDA graph 是按某个桶烤的(烤进去的 CTA 数/掩码是常量);
   尾批把 target 行改成 4 之后,复用了按 8 行烤的图(或反过来),被掩掉的 CTA
   仍在等一个由满宽度同伴写来的标志 ⇒ 自旋。
2. **`int_ws`(内部工作区)按 `t=4/padded=42` 激活**,而该形状的收尾同步点
   只在 `t=8` 分支里被置位 ⇒ 等待永不满足。

旁证(不是结论):卡死前最后一次成功轮是 `T=8`;`T=4` 那次日志正好停在
`int_ws 激活` 之后、`verify 完成` 之前。

## 四、复现步骤

```bash
cd packages/owl && cargo build --release -p server
./target/release/server -p profiles/qwen38_27b-3090.toml \
  --set runtime.max_seq=49152 --set parallel.device='["GPU-…"]' \
  --log-level debug &
curl -s http://127.0.0.1:8134/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"What is 2+2?"}],
       "max_tokens":16,"temperature":0.7}'
# 永不返回;nvidia-smi 看 GPU util 恒 100%
```

关掉投机(`--set spec.depth=0 --set spec.lookup=false`)同请求 0.4s 返回。

## 五、规避(修复前的临时手段;2026-10-25 修复后不再需要)

1. **客户端给足 `max_tokens`**:`≥ 64` 基本不进裁剪,agent 场景(pi = 8192)不受影响;
   但**"跑到 `max_tokens` 上限"的回合仍会进尾批** ⇒ 这不是免疫,只是低概率。
2. 短请求/评测臂用 `--set spec.depth=0 --set spec.lookup=false`(恒等门本来就要求 spec-off)。
3. 一旦被钉死:只能 kill 重启(`SIGTERM` 收不掉在飞的 turn,见 §七)。

## 六、下一步取证建议(修复前记录;§十 已定谳)

- 在 `verify_round` 入口断言:`graph 烤进的 rows` 与 `target_rows` 不等 ⇒ 响亮拒
  (而不是带着错形状发射),先把它从"自旋"变成"报错",顺便确认假设 1;
- 尾批强制走 eager verify(`probes.propose_eager` 同族开关)对照:若不再自旋 ⇒ 图档侧;
- 查 `int_ws` 的置位/等待条件在 `t<桶宽` 分支是否成对(假设 2);
- 补一条**引擎级回归门**:`max_new ∈ {depth+1 … depth+8}` 扫一遍,防止再退化。

## 七、同场发现(另案,未修)

`SIGTERM` 后日志已打「优雅关停:收到停止信号 ⇒ 释放卡锁」,但 HTTP 面**继续受理新 turn**
(关停行之后的日志里仍有 `受理 … turn N`)。收摊没收住监听/准入 ⇒ 关停语义不完整。
这与本案的"卡死 turn 无法被摘除"叠加,会让 kill 也救不回来(实测需 `kill -9`)。

## 八、与 store 层幽灵案的关系

同日另一起「`块 … 已在宿主层` 粘性毒化」(宿主副本留尸)**已修**,
判据与回归见 `crates/store/tests/store_e2e.rs::owned_recompute_leaves_no_host_ghost`
与 `/debug/vram` 的 `kv.host_duplicate_landings` 指纹。

**两案无关**:本案设备侧自旋,那案宿主账漂。记录在此是为了避免后来人
把两种"turn 不返回"混成一起。

## 九、追加取证(2026-10-25,pi 功能性验收副产品)

**同一台机、`qwen38_27b-3090.toml`(未改档,spec d7)、两次独立 boot、同一请求 2/2 复现**:

```bash
curl -s http://127.0.0.1:8134/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"用一句话说明什么是目录。不要思考。"}],\
       "max_tokens":64,"temperature":0,"stream":false}'
# prompt=56 / max_new=64 ⇒ 永不返回;nvidia-smi GPU util 恒 100%
# /health 仍 200(HTTP 面活着),chat 面彻底不返回
```

debug 档停在(删去无关行):

```text
[spec-round] 轮开始 token=97152 pos=109 depth=7 vlm_delta=0
[owl-fi] [plan] padded_batch=42 total_rows=8 ...
[spec-round] pos=109 接受完成 m=2 bonus=12 emitted=3
...
[spec-round] 轮开始 token=12 pos=112 depth=7 vlm_delta=0
[owl-fi] [plan] padded_batch=42 total_rows=7 ...
[owl-fi] [plan] int_ws 激活(t=7 padded=42 tiles=1 cap=false)
        ← 此后无任何日志,请求永不返回
```

⇒ 与 §二/§三 同一条链:**T=8(满宽)轮正常完成,T=7(裁剪)轮发射后自旋**。
本次增量事实:

1. **`max_tokens ≥ 64` 不是免疫线** —— §五 第 1 条的"基本不进裁剪"只对早停回合成立;
   本次 `max_new=64` 在 `pos=112`(已生成 56)进入裁剪。任何跑到上限附近的回合都会进,
   长思考 agent 回合(pi `max_tokens=8192` 但思考跑满)同样会;
2. 两次独立 boot 均复现(**确定性**),排除"布局运气"(与 A13 不同);
3. 卡死判定双判据可复用:`GPU util 恒 100%` + `/health` 200 但 chat 不返回;
4. 功能验收请在 spec-off 下跑(`--set spec.depth=0 --set spec.lookup=false`),
   本次 pi 全链(spec-off)35 轮零错误、`host_duplicate_landings=0`。


## 十、修复(2026-10-25)

### 根因(假设 1 成立,假设 2 不是根因)

`execute_spec_round` 把 **`live + 1`** 当成 `verify_round` 的 `target_rows`
传进去。`verify_round` 又把这个值用于:

- `fi_plan_spec(target_rows, ctx_total)` —— FI plan 的 `total_rows`；
- `gdn_cu = target_rows`；
- `xqa_seq = pos + target_rows`。

但 verify 的 **CUDA graph 是按 `depth+1` 行捕获的**。尾批 `live < depth` 时，
目标行变成 `T=7/4/...`，却复用满宽 `T=8` 的图；FI 的尾部 CTA 被掩掉后仍在
等一个由满宽同伴写来的信号 ⇒ 设备自旋、GPU 100%、主机全 futex。

§三的假设 1（图档形状错配）成立；假设 2（`int_ws` 置位条件）不是根因。

### 修复

**预算截断只作用于提交面，绝不改 verify 图的 target 形状。**

- `execute_spec_round` 的 `live` 仍按 `live_depth_for_budget` 计算，只用于
  接受/提交上限；
- `verify_round` **不再收 `target_rows` 参数**；内部固定
  `target_rows = block.len() (= depth+1)`；
- FI plan/GDN/XQA 全部按满宽 `depth+1` 发射，与捕获图一致；
- 未接受的行由既有 fold/回滚路径丢弃（KV 覆写幂等）。

结构性防回归：调用点无法再传一个更小的 target_rows（签名已删）。

### 真机验证

固定二进制 = 本修复后的 `server`，`qwen38_27b-3090.toml`（spec d7，
`max_seq=49152`）与 `qwen38_27b-3080_tp2.toml` 各起一次：

| 用例 | 修复前 | 修复后 |
|---|---|---|
| `max_tokens=16`, "What is 2+2?" | 可能卡死 | **0.17s 返回，16 tok** |
| `max_tokens=64`, "用一句话说明什么是目录。不要思考。" | 2/2 boot 卡死 | **1.2s 返回，64 tok** |
| `max_tokens ∈ {2,3,4,5,6,7,8,9,10,12,14,15,16,17,20,32,64}` | — | **17/17 返回，无自旋** |
| 第二次独立 boot + 同请求 | — | **通过** |
| 3080 TP2 + 同请求 | — | **通过** |
| spec-on vs spec-off 同 prompt/temp=0/max_tokens=64 | — | **content/reasoning 逐字节一致** |

关键日志判据（debug 档）：尾批轮仍打印
`[owl-fi] [plan] ... total_rows=8 ...`，不再出现 `total_rows=7` 后无后续；
请求完成后 `nvidia-smi` GPU util 回到 0。

### 残留

- 本修复不改变 `SIGTERM` 收不住在飞 turn 的 §七 问题；
  卡死已不再发生，但关停语义仍应另案补。
- 尾批仍会多算未提交的 draft 行（full width），这是换取图形状稳定的代价；
  只发生在接近 `max_tokens` 的尾部，量级可忽略。

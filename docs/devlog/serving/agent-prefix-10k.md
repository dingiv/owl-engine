# Agent 10K 多轮前缀命中率：定位与定谳（2026-10-24）

> 场景：agent/pi 长会话，每轮请求约 10K token（系统提示 + 工具 + AGENTS +
> 历史），只追加一小段用户/工具增量。关心的不是冷灌 10K 的速度，而是下一轮
> `usage.prompt_tokens_details.cached_tokens`。
>
> 判据器：`apps/server/tests/agent_prefix_10k_probe.py`（本文件同轮入仓）。
> 结果：**snap_max=0 时结构性 0%；snap_max=4 + snap_host>0 时同会话/跨会话
> 98–100%**。生产 AWQ + DFlash2 档（`snap_max=4`）实测通过。

## 1. 探针形状

固定一段 ~10K 背景（300 行随机词，Qwen 两种 tokenizer 都约 10K），然后：

| 步骤 | 内容 | 期望 |
|---|---|---|
| T1 | 会话 A 首轮，用户任务 1 | 冷；唯一 salt 保证不撞旧缓存 |
| T2 | A 追加 assistant（长输出 160 tok）+ 短用户轮 | 同会话增量命中 |
| T3 | A 再追加一轮 | 同会话继续命中 |
| T4 | 会话 B，原样 T1 messages | 跨会话全前缀命中 |
| T5 | 会话 C，原样 T2 messages | 跨会话增量命中 |
| T6 | 不带 `session` 键（pi 行为） | 隐式会话命中 |
| T7 | 会话 D，同 system、不同用户任务 | 固定 system 前缀分叉命中 |

T1 的 assistant 输出故意超过 32 token：生成段会跨页，触发 Transient 边界
快照，用来暴露 `snap_max=1` 的 durable 快照被顶掉问题。

## 2. 实测矩阵

引擎：GGUF Qwen3.8-27B Q4_K_M，单 3090 Ti，`max_seq=32768`，f16 KV，
`prefix_cache=true`，`query_replay=true`（KVMem 行；新 GGUF agent 档为
`query_replay=false`），`implicit_session=true`。
不同行只改 `snap_max` / `snap_host_mb`。

| 配置 | T2 同会话 | T4 跨会话 | T5 跨会话增量 | T6 无 session | T7 system 分叉 |
|---|---:|---:|---:|---:|---:|
| `snap_max=0` | **0 / 10124 = 0%** | **0 / 9942 = 0%** | **0%** | **0%** | **0%** |
| `snap_max=1`, `snap_host=0` | 10080 / 10113 = 99.7% | **0 / 9931 = 0%** | 99.7% | 100% | 未测（T4 同因） |
| `snap_max=4`, `snap_host=0` | 10080 / 10113 = 99.7% | 9920 / 9931 = 99.9% | 100% | 100% | 8192 / 9921 = 82.6% |
| `snap_max=4`, `snap_host=0`（长输出 310 行） | 10400 / 10449 = 99.5% | 10240 / 10267 = 99.7% | 10432 / 10449 = 99.8% | 99.8% | 10240 / 10257 = 99.8% |
| `qwen38_27b-3090-gguf-agent`（`snap_max=4`, `snap_host=2048`, **消息边界开**） | 10080 / 10124 = 99.6% | 9920 / 9942 = 99.8% | 10112 / 10124 = 99.9% | 99.9% | **9888 / 9932 = 99.6%** |

生产 agent 档（AWQ-INT4 + DFlash2 d7，`snap_max=4`，`snap_host=3072`，
`prefix_cache=true`，`implicit_session=true`）：

```text
T1 cold        prompt= 9942 cached=    0
T2 same        prompt=10124 cached= 9920 (98.0%)
T3 same        prompt=10148 cached=10112 (99.6%)
T4 x-sess      prompt= 9942 cached= 9920 (99.8%)
T5 x-sess      prompt=10124 cached=10112 (99.9%)
T6 no-session  prompt=10124 cached=10112 (99.9%)
T7 sys-branch  prompt= 9932 cached= 9888 (99.6%)
cold/cached equality: YES
```

T2 的 98.0% 不是零命中：与最新整块 10112 差 192 token（6 个页边界）。T3/T5/T6 已恢复到
10112，说明尾快照随后补齐；与 spec 全接受轮跳过边界快照（已知缺口）或
生成段页边界落点有关，另账。

## 3. 机制定谳

### 3.1 `snap_max=0` = 结构性 0，不是“命中率低”

`capture_gdn_snap_tail` 的快照面门：

```rust
let snap_on = snap_keep && self.dev.pool.snap_capacity() > 0
           && self.boot.cfg.knobs.prefix_cache;
```

`snap_max=0` ⇒ `snap_capacity()==0` ⇒ 边界事件只做链哈希/索引，不拍 GDN
快照。`reuse_face` 的复用面判据是“最后一个命中边界有快照在场”（设备或宿主），
没有快照就逐块 `m -= 1`，一路裁到 0。结果是：

- 同会话增量 0；
- 跨会话块级缓存 0；
- pi 不带 session 时也 0；
- 只有 `cached_tokens=0` 这一个外在表现，**没有错误日志**。

这正是本轮 10K agent 场景的主因。旧 `qwen38_27b-3090-gguf-kvmem*` 档把
`snap_max=0` 当作“query replay 用专用缓冲”的省显存选择，但同时在配置里
写了 `prefix_cache=true` —— 两者语义冲突。

### 3.2 `snap_max=1` + `snap_host=0`：同会话看似正常，跨会话掉 0

生成段每跨一个页边界会拍一条 `Transient` 快照；它逐出即丢、不落宿主。
`snap_max=1` 时它需要槽位，就会把唯一的 `Durable` 前缀边界顶掉，且
`snap_host=0` 无法下沉。于是：

- T2 仍能命中 **最新的生成段尾快照**（10080），同会话看起来 99.7%；
- T4 需要的是首轮 prompt 的 9920 边界，已被顶掉 ⇒ **0%**；
- T5 因为带着更长的历史，命中 10080 尾快照，仍 99.7%。

这条对 agent 很危险：单个长会话会掩盖问题，新会话/子代理/compaction 后
立刻暴露。

### 3.3 `snap_max=4`：同会话/跨会话闭环

4 个槽位把“durable 前缀边界”和“生成段 Transient 尾快照”分开：Transient
在槽内 churn，durable 保留。实测 T2/T4/T5/T6 全部 98–100%，且 T1 冷路径与
T4 缓存路径输出 sha 完全一致（`query_replay=true` 下本 prompt 未出现状态
不一致；仍需更大样本复核 replay × snapshot 的通用等价性）。

### 3.4 残余损失与消息边界快照（2026-10-24 已落地）

旧的 `SNAP_EVERY=4`（每 4 个 prefill chunk，即 2048 token）只在粗锚点拍快照；
T7 的 system 结尾不在锚点上时只复用 8192/9932 = 82.5%。这不是结构性问题，
但它是 agent 跨会话/子代理场景的下一档收益。

现在新增**消息边界快照**（A-P0/A-P1）：

```text
prompt_ids
→ 用可配置的“消息开启标记”token 序列扫描（不是写死字符串）
→ 每个下一个标记前 = 上一条消息结束
→ floor 到页边界 = snap point
→ prefill 在 point 切块 + 拍 Durable 快照
```

实测（同一 GGUF agent 档，`snap_msg_boundary=true`）：

```text
T7 sys-branch  prompt= 9932 cached= 9888 (99.6%)
T2 same        prompt=10124 cached=10080 (99.6%)
```

T7 只损失 44 token（页边界取整 + 少量尾差），system 前缀分叉基本拉满。
同 build 单键 A/B（GGUF 32K 档）：

```text
snap_msg_boundary=false: T7 8192/9932 = 82.5% · T1 10.63s
snap_msg_boundary=true : T7 9888/9932 = 99.6% · T1 10.61s
```

T7 命中 +17.1pp，冷 prefill 墙钟在噪声内。GGUF 档 T2 也是 99.6%；
生产 AWQ 档 T2 仍 98.0%（剩余 192 token 属 spec 全接受轮的页边界缺口，
另账）。

分隔符**不是写死的**：

```toml
[pool]
snap_msg_boundary = true
snap_msg_delim = ""        # 空 = 模型 ChatFormat.im_open；可填任意自定义标记
snap_msg_min_gap = 2048    # 第一条无条件保留，后续按 token 间隔稀疏
snap_msg_min_span = 0      # 0 = 即使边界贴近 chunk 起点也切；>0 可抑制碎块
```

引擎 boot 期把 `snap_msg_delim`（或模型 `im_open`）tokenize 成 `open_ids`，
扫描函数 `message_snap_points(prompt_ids, open_ids, page, min_gap)` 只收 token
序列参数 —— 换模板/换标记 = 改配置或模型声明，不改扫描代码。

**第二个踩坑（已修）**：上一块恰好在边界结束时，下一块 `fed` 就等于该点；
若切块判据允许 `b == fed` 会得到 `step=0`，触发
`owl_gguf_dequant_rows_q4k out_elems=0`。判据已收紧为 `b > fed`，并加了单测。

## 4. 已落代码

- 新增 `apps/server/tests/agent_prefix_10k_probe.py`：T1–T7 + 四条硬门
  （T2/T4/T5/T6 `cached/prompt >= 0.80`），可选 `--check-equality`；
  `--min-branch-hit 0.95` 可把 T7 也纳入硬门。
- 新增 `profiles/qwen38_27b-3090-gguf-agent.toml`：GGUF Q4_K_M / 32K /
  `prefix_cache=true` / `snap_max=4` / `snap_host_mb=2048` /
  `implicit_session=true` + 消息边界快照，不启用 KVMem 选择层。
- 新增 `crates/engine/src/msg_boundary.rs`：`message_snap_points` 纯函数
  （`open_ids` 是参数，不写死分隔串）+ 单测。
- `prefill_step_chunk_msg` 接 `msg_snaps`；scheduler/running/phases 四处
  生产调用点同源；`Turn`/`ActiveTurn` 携带 `msg_snaps`。
- `capture_gdn_snap` 的 `snap_keep` 增加 `msg_snap`；边界切块拍 Durable。
- 生产 AWQ DFlash2 档也打开 `snap_msg_boundary`（T7 从 82.5% → 99.6%）。
- `boot_kvmem_check` 新增两条 WARN（不硬拒，纯 attention 档仍可合法无快照）：
  - `prefix_cache=true && snap_max=0`；
  - `prefix_cache=true && snap_max=1 && snap_host_mb=0`。
- `qwen38_27b-3090-gguf-kvmem.toml` 的 `snap_max=0` 注释补上
  “前缀复用结构性关闭，仅冷长上下文实验”。

## 5. 复现

```bash
# 生产 agent 档（AWQ + DFlash2）
./target/release/server -p profiles/qwen38_27b-3090-w4a16-dflash.toml \
  --set server.bind=127.0.0.1:8137 \
  --set runtime.implicit_session=true --log-level info

python3 apps/server/tests/agent_prefix_10k_probe.py \
  --base-url http://127.0.0.1:8137 --model qwen3.8-27b \
  --min-branch-hit 0.95
# RESULT: PASS（T2 98.0% / T4 99.8% / T5 99.9% / T6 99.9% / T7 99.6%）

# GGUF 32K agent 档（无 KVMem 选择层）
./target/release/server -p profiles/qwen38_27b-3090-gguf-agent.toml --log-level info
python3 apps/server/tests/agent_prefix_10k_probe.py \
  --base-url http://127.0.0.1:8137 --model qwen3.8-27b-gguf \
  --min-branch-hit 0.95
# RESULT: PASS（T2 99.6% / T4 99.8% / T5 99.9% / T6 99.9% / T7 99.6%）

# 反例：KVMem 档 snap_max=0
./target/release/server -p profiles/qwen38_27b-3090-gguf-kvmem.toml \
  --set runtime.max_seq=32768 --set runtime.implicit_session=true
# boot 出现 WARN；探针 RESULT: FAIL（T2/T4/T5/T6 全 0）
```

## 6. 结论

1. agent 10K 多轮命中率的主因不是块级缓存树，也不是渲染对称律，而是
   **边界 GDN 快照被关掉/被顶掉**。`prefix_cache=true` 不等于前缀可复用。
2. 生产 agent 档 `snap_max=4 + snap_host_mb>0` 已达到 98–100%；本地 GGUF
   10K 档用 `profiles/qwen38_27b-3090-gguf-agent.toml`（32K，无 KVMem），
   不要再使用 `snap_max=0` 跑 agent。
3. 固定 system 前缀的分叉场景已由消息边界快照补到 **99.6%**；剩余 T2 的
   98.0% 属 spec 全接受轮的页边界缺口，下一步优先补那里，而不是继续加
   `snap_max`。

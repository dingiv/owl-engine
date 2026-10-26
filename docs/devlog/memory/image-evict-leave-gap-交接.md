# 图像无感删除：Leave-Gap 改造方案（设计与交接）

> 状态：**P0-P2 核心 + 0.8B spec 门已实现；P3 27B/TP2 与 P4 consolidation 未做**（2026-10-25）
> 目标：彻底移除 `image_evict_delta_limit=4` 导致的“每 5 次删除一次整段 prefill”
> 尖峰，同时保持三图窗口、前缀缓存和 FP8 KV。
>
> 基线现场：`perf-runs/ttft-prefill-20261025/`
> 基线结论：`docs/devlog/serving/ttft-prefill-round-20261025.md`
> 当前实现：`docs/devlog/memory/prefix-deletion.md`
>
> **本文是下一个会话的施工交接，不是完成记录。**

---

## §0bis 施工结果（2026-10-25）

已落地（0.8B 门验收）：

- 配置：`[pool].image_evict_mode = "rebase" | "leave_gap"`；leave-gap 不受
  FP8 `delta_limit ∈ 1..=4` 约束，默认仍是 `rebase`；
- `AgentSession.positions / max_pos / positioned / position_epoch`；
- `img_evict::{compact_positions, positions_after_splice}`：删除后前缀/后缀
  position 原样保留、新尾巴从 `max_pos+1` 追加；
- `ChunkTables` 显式 positions；decode 用 `ActiveTurn.next_rope_pos`；
  VLM cos/sin 由 `Vlm::prepare_with_positions` 生成；
- `scheduler::plan_image_leave_gap`：摘页 + 快照元数据重挂，**不调用 K 重转**；
  保留 rebase 旧路径；
- positioned 会话的跨会话内容命中收窄为**本会话链**；
- target spec verify/replay 用 `next_rope_pos - 1` 的 `rope_delta`（草稿仍走既有
  token 位置，只影响 AL，不影响 greedy 恒等）；
- 0.8B 真机门：原 `gpu_image_evict_gate` 已改造成 leave-gap 版（两次连续
  删除、K 逐字节不变、零重建、幸存 position 逐 token 不变、删点起旧链身份
  释放）；门内**手动拆 begin 与 prefill**，在 `execute_begin` 后断言被删图
  槽 `slot_refs == 0`（删除前 >0），直接证明缓存引用已解除而非“等 prefill
  复用后再看”。`gpu_image_evict_spec_identity_gate` 在
  `OWL_GATE_LEAVE_GAP=1` 下 spec on/off 逐位相同。
- **27B 真机 TTFT A/B（2026-10-25，单 3090，22k prompt / 三图窗口）**：
  leave-gap 删首图轮 `ttft=1342ms` / cache 98.3% / prefill 368 tok；
  同请求 `--no-evict` 控制臂 `ttft=18070ms` / cache 0%。删除轮还实测
  宿主回灌 676 页后吸收（不再因 host 占位退回全量）。

未做：P4 consolidation（position 上限）、草稿池/DFlash2 的绝对 rope 位置、
27B TP2 长跑、生产 profile 默认切换。

---

## §0 一句话

当前实现是 **“删 token + 把后缀 K 原地旋转”**，旋转用的 K 已经量化过，
所以 FP8 每次重转都会累积误差；`delta_limit=4` 是精度保险丝，到顶只能整段
prefill 重置。

新方案改成 **“删 token/KV block，但不旋转、不移动幸存 token 的 position”**
（leave-gap）：

```text
旧图 token 真删、KV block 真释放
幸存 token 的 position 保持不变
新 token 从 max_position+1 继续追加
K 一个字节都不改写
```

这样：

- 不再有 K 重转 → 没有 FP8 漂移；
- 不再有 `delta_limit` → 没有每 5 次删除的全量重建；
- 图片删除变成纯 block 摘除 + 位置账记账；
- 旧 K 和前缀缓存继续可用；
- 代价是 position 空间出现“洞”，到模型最大 position 前需要一次
  **consolidation**（把幸存窗口重新压到 position 0；频率从“每 5 次”
  降到“每几百轮”，可尝试放 idle）。

---

## §1 为什么当前方案必须重建

正常删图：

```text
保留前缀 K
删掉图片 token
后缀 token 位置前移 S
后缀 K 乘 R(−S)
```

但第 N 次删除旋转的是**已经旋转过 N−1 次、且经过 FP8 量化的 K**：

```text
K_N = quant(rotate(quant(rotate(... quant(K_0)))))
```

FP8 e4m3 只有 3 位尾数；`dequant → rotate → requant` 每次都会重新落格。
实测（`prefix-deletion.md` §5sexies）：

| 叠转次数 | FP8 最差误差 |
|---:|---:|
| 4 | ≈1 个 e4m3 码步 |
| 16 | 4.5 个码步 |
| 32 | 8.9 个码步 |
| 64 | 21 个码步 |

所以当前实现取 `delta_limit=4`：第 5 次删除必须 `plan_fresh`，用一次
43K–57K 的 cold prefill 把 K 和 GDN 状态归零。

**这是正确性门，不是缓存 bug。**

---

## §2 外部研究结论

### 2.1 vLLM #53876：留空位 + consolidation

[vLLM PR #53876](https://github.com/vllm-project/vllm/pull/53876) /
RFC #51948 对长流式会话采用：

- 删除中间 token 的 KV block；
- **幸存 token 保留原 position**；
- 不旋转 K；
- position 接近模型上限时做 consolidation（重新 prefill 幸存窗口）。

其 Qwen3-VL-8B / H100 实测：

```text
300 帧、window=4：
TTFT 约 41ms（首帧 322ms）
300 帧只 consolidation 7 次
```

这说明“删除后不压紧 position”在真实多模态会话中可以做到稳定、低延迟。

### 2.2 vLLM #57672：raw-survivor / query-side rotation

[vLLM RFC #57672](https://github.com/vllm-project/vllm/pull/57672) 走另一条：

- K 存 raw（未旋转）；
- 每次删除时从 raw K **直接旋转一次**到当前 position；
- 不做 correction-on-correction，因此不累积漂移；
- 代价是 raw K 存储 + query-side rotation / 额外 kernel。

其 0.5B needle 实验：

```text
未校正：n=48 时 6.7%
校正/RSQR：95–97%
```

这是 leave-gap 之外的第二条候选；本方案先做 leave-gap，因为它不需要
raw K 双份存储，也不需要改 attention 的 query 旋转。

### 2.3 结论

对 GUI 三图窗口：

- **Leave-gap 是首选**：不改 K，不改 FP8 精度，不重建；
- RSQR 作为未来“必须压紧 position”的备选；
- f16 KV（delta_limit=32）只是降低频率，不是彻底解决。

---

## §3 目标与非目标

### 目标

1. 删除已标记图片时，不调用 K 旋转核；
2. 不因删除触发 `plan_fresh` / 全量 prefill；
3. 图片 KV 真正释放，prompt/KV 内存有界；
4. 热轮 `cached_tokens` 维持高位，TTFT 只包含新尾巴 prefill；
5. 保持三图窗口、FP8 KV、投机解码、TP2；
6. 到 position 上限时能显式 consolidation，且可观测；
7. 保留旧 `rebase` 模式作为回退（配置开关）。

### 非目标

- 不实现任意文本段删除（仍只支持 `owl_evict` 图像段）；
- 不改变“删点之后是搬运记忆”的语义；
- 不追求 position 空间永不增长；
- 不在本阶段实现异步 idle consolidation（先做同步，P2 再优化）；
- 不改 3090 默认档，先在 3080 TP2 和 0.8B 上验证。

---

## §4 核心模型：两套坐标

### 4.1 紧凑 token 坐标

`AgentSession.tokens` / `block_table` / `cached_len` 仍表示**当前实际存在的
token 序列**：

```text
tokens:       [t0, t1, t2, ...]          # 删掉的图片 token 不在其中
block_table:  [page0, page1, ...]        # 与 tokens 页对齐
cached_len:   tokens.len()
```

这套坐标用于：

- KV block 表；
- 前缀匹配 / 生成；
- tokenizer / 模板 / 图像段账。

### 4.2 绝对 position 账

新增每 token 的 **m-RoPE position**：

```rust
struct PositionLedger {
    /// 与 tokens 等长；每个 token 的 (T,H,W) 绝对位置
    pos: Vec<[i32; 3]>,
    /// 当前最大 position（三轴取 max；通常 T 轴最大）
    max_pos: i32,
    /// 最近一次 consolidation 的代数（诊断/缓存键）
    epoch: u32,
}
```

- 文本 token：`[p,p,p]`；
- 图像 token：`MropeCfg::vision_position_ids(base, grid)` 产生的
  `[t,h,w]`，其中 `base = max_pos + 1`；
- 新尾巴：从 `max_pos + 1` 顺序分配；
- **删除中间图时只删 `pos[a..b]`，后面的 position 不变**；
- `max_pos` 不因删除而回退（除非删的是最后一个 token，按实际重算）。

> 为什么存每 token 而不是只存 offset：图像是 m-RoPE 3D，删除后不同
> segment 的 position 不可由一个标量 `S` 推导；per-token 账是唯一真源。

---

## §5 删除流程（leave-gap）

### 5.1 提交面

`running.rs::submit_session_ex` 仍做：

1. 客户端全量 prompt tokenize；
2. 与 `AgentSession.tokens` 的 unpadded 文本做删除识别
   （沿用 `img_evict::detect_text_delete`）；
3. 找到被删的 marked 图像 span；
4. 生成新的紧凑 token 序列 + 新 position 表；
5. 把 `Turn` 携带 `(new_tokens, new_positions, deletion_range, new_img_spans)`。

### 5.2 删除定位

第一阶段**保留当前 pad 律**：

- 可删图像段仍页对齐（前后 pad）；
- 删除区间仍是整页；
- 这样 block_table 可按页摘除，避免半页搬运。

以后可再做“无 pad leave-gap”优化（不在本阶段）。

### 5.3 新序列构造

给定旧 padded 序列、旧 positions、删除区间 `[a,b)`、客户端 raw 新 prompt：

```text
new_tokens =
    old_tokens[..a]                  # 删点前，逐字节不变
  + old_tokens[b..b+k]               # 旧后缀中与 raw 相同的部分（保留原 position）
  + new_tail_tokens                  # 客户端新增的尾巴（分配新 position）

new_positions =
    old_positions[..a]
  + old_positions[b..b+k]            # 不移动！
  + assign_new_positions(new_tail)   # 从 max_pos+1 开始
```

实现上可以：

- 复用 `detect_text_delete` 得到 `k`（共同后缀 token 数）；
- 用旧的 padded tokens/positions 做 splice；
- 只对 `new_tail_tokens` 重新 tokenize / 计算 m-RoPE；
- 避免对整段历史重新跑 `rope_index`（否则 position 会被重算成紧凑值）。

### 5.4 账本更新

`AgentSession`：

```rust
tokens = new_tokens;
positions = new_positions;
block_table = old_block_table 删除 [a/page, b/page) 对应页;
cached_len = tokens.len();
img_spans / pads = 按新坐标重建;
max_pos = max(new_positions);
rot_rounds 删除（leave-gap 不再使用）;
```

**不调用**：

- `positions_shift`；
- `k_shift` / `execute_k_shift`；
- `snap_rekey` / 旧身份废弃（K 没变）；
- `delta_limit` / `rebase`。

### 5.5 GDN 状态

GDN 是因果递归状态，删除图片后仍沿用旧 state（“搬运记忆”语义），
与当前 image_evict 一致；leave-gap 不改变这一点。

consolidation 时才 `reset_gdn` + 从头 prefill。

---

## §6 position 如何进入前向

### 6.1 prefill

当前 `ChunkTables::build_view` 的 `pos = seq(base, t)` 用 token 下标当
position。leave-gap 需要改为：

```rust
ChunkTables.pos = positions[base..base+t]  // 显式 per-token 位置
```

`lens` / `fi` / `xqa.seq_len` 仍按**token 数**计算（注意力看到的 token 数
不变），只有 `pos` 槽改为绝对位置。

### 6.2 decode

当前 `decode_step_inputs`：

```rust
pos        // KV length / token index
pos_rope   // 只有 VLM delta 偏移
```

需要拆成：

```rust
kv_pos     // token 数（kv_len / 块表）
rope_pos   // 绝对 m-RoPE position（下一 token）
```

`pos` 槽喂 `rope_pos`；`kv_len` / `xqa_seq` 仍用 token 数。

### 6.3 m-RoPE / 图像

`MropeCfg::rope_index` 增加一个“从绝对 base 开始”的入口，或直接对
新图使用 `vision_position_ids(base, grid)`：

- 文本新 token：`base+i` 三轴同值；
- 新图像：`vision_position_ids(base, grid)`，更新 `max_pos = base + advance(grid)`；
- 旧图像/旧文本：直接用 positions 账，不重新生成。

### 6.4 spec / MTP / DFlash2

`ActiveTurn` 增加：

```rust
next_rope_pos: i32,     // 下一个生成 token 的绝对 position
```

- verify 块 `pos..pos+depth` 中的 `pos` 用 `next_rope_pos`；
- draft/propose 的槽表、RoPE 位置都用 `next_rope_pos`；
- KV 槽表仍按 token 数；
- 生成 token 逐个追加 positions（连续，不产生洞）；
- 若新 prompt 尾巴含图像，prefill 已把 positions 算好，生成从 `max_pos+1` 开始。

### 6.5 快照 / 消息边界

`msg_snaps` 仍是 token 下标；`SnapKey` 的链身份需增加
`position_epoch`/position fingerprint（见 §7），防止“同 token 不同
position”错误命中。

---

## §7 前缀缓存与快照

这是改造的第二难点。

### 7.1 问题

当前 block 链哈希只吃 token 内容：

```text
hash(parent, block_tokens)
```

leave-gap 后同样的 token 序列可能有不同的 position（洞的位置不同），
K 值不同；跨会话/跨 epoch 直接按内容命中会读错 K。

### 7.2 方案

P0（先正确）：

- `AgentSession` 标记 `positioned: bool`；
- `register_chain` 只登记**非 positioned**链，或给 positioned 链加
  `position_fingerprint`；
- positioned 会话的跨会话内容缓存默认关闭；
- 同会话仍用 `block_table` 直接复用，不依赖 content tree。

P1（优化）：

- `ChainKey = hash(token_block, position_block)`；
- snapshot key 同样带 position fingerprint；
- 只对相同 position 账的链开放跨会话复用。

### 7.3 快照

- 设备/宿主/盘层 snapshot 的 key 必须带 position epoch；
- leave-gap 不再 `snap_rekey`（K 没变）；删除只摘 block；
- GDN 快照仍在消息/页边界拍；删除后的快照语义仍是“搬运记忆”。

---

## §8 配置与兼容

建议新增：

```toml
[pool]
image_evict_mode = "leave_gap"   # rebase | leave_gap
image_evict_delta_limit = 4      # 仅 rebase 模式使用

[runtime]
max_position = 0                 # 0 = 由模型/配置推导；leave-gap 的 consolidation 阈值
```

兼容策略：

- 默认仍是 `rebase`，生产 profile 不动；
- 测试 profile / 新 GUI 档显式 `image_evict_mode = "leave_gap"`；
- leave-gap 下 boot 不要求 `delta_limit ∈ 1..=4`；
- `image_evict` 的旧 boot 门保留，但按 mode 分支；
- 旧 `rot_rounds` / `k_shift` 代码保留，直到 leave-gap 通过 27B 门；
- 不删掉回退路径。

观测：

```text
image_evict.absorbed
image_evict.blocks_freed
image_evict.positions_gap
image_evict.max_position
image_evict.consolidate
image_evict.consolidate_reason
```

---

## §9 consolidation

### 9.1 触发

当：

```text
max_pos + expected_new_tokens + max_new > max_position_budget
```

（`max_position_budget` 建议先取 `runtime.max_seq` 或模型 max position；
不要和 token 池容量混为一谈。）

### 9.2 动作

```text
保留当前 compact tokens / images
positions = 0..n-1（按当前序列重新计算 mRoPE）
GDN reset
全量 prefill 一次
block table 重建 / 旧链释放
epoch += 1
```

这与当前 `plan_fresh` 类似，但**不是每 5 次删除**；先做成下一请求同步触发，
P2 再尝试 idle 预热。

### 9.3 验收

- 100 轮三图窗口：consolidation 0 次；
- 300 轮：consolidation ≤1 次；
- consolidation 日志含 `max_pos`、prompt len、耗时；
- consolidation 后下一轮 cached 正常。

---

## §10 代码触点（交接清单）

### 配置 / 数据

- `crates/shared/src/config/mod.rs`
  - `ImageEvictMode`；`max_position`；
  - boot gate 按 mode 分支。
- `crates/engine/src/session.rs`
  - `AgentSession.positions: Vec<[i32;3]>`
  - `max_pos`、`positioned`、`epoch`
  - `reset` / `rebase` / `commit_turn` 同步清/写 positions
  - `guard` 仍比 tokens；position 不参与 guard，但参与缓存键。
- `crates/engine/src/turn.rs`
  - `Turn` 携带新 positions / deletion；
  - `ActiveTurn.next_rope_pos`；
  - `TurnStats` 增加 `max_pos`/`gap_tokens`（可选）。

### 提交 / 调度

- `crates/engine/src/running.rs::submit_session_ex`
  - 删除识别后构造 `new_tokens/new_positions`；
  - 不再 `rebuild_after_delete` 后走 K 旋转路径。
- `crates/engine/src/scheduler.rs`
  - `plan_image_delete` 改成 `plan_image_evict_leave_gap`：
    - 校验页对齐；
    - 删 block_table 页；
    - 不 `positions_shift`；
    - 不 `snap_rekey`；
    - 不 `incref` 后缀旋转页（后缀页仍属新链）；
  - 删除/绕过 `image_evict_rebase_due`（leave-gap mode）。
- `crates/engine/src/exec/turn_end.rs`
  - `commit_turn` 写 positions；
  - consolidation 检查。

### 前向 / 位置

- `crates/engine/src/exec/phases.rs`
  - `ChunkTables` 接收 positions；
  - decode `kv_pos` vs `rope_pos`；
  - `prepare_vlm_rank` 的 mRoPE 位置来源。
- `crates/engine/src/exec/prefill_chunk.rs`
  - `ChunkTables.pos` 从 `seq(base,t)` → positions slice；
  - FI/XQA 的 `seq_len` 仍用 token 数，不改成 max_pos。
- `crates/models/src/layers/mrope.rs`
  - `rope_index_from(base, ...)` 或直接 `vision_position_ids(base, grid)`；
  - text/image 新尾巴的 position 生成。
- `crates/engine/src/exec/spec.rs` / `exec/draft.rs`
  - spec verify / propose / MTP 的 `pos` 改 `next_rope_pos`；
  - 槽表仍用 token 下标。

### 缓存 / 存储

- `crates/store` / `ChainCursor`
  - position-aware chain key（P1）；
  - snapshot key 带 epoch（P0 至少不让跨会话命中）。
- `crates/engine/src/scheduler.rs::plan_prefix_reuse`
  - positioned 会话的复用规则。
- `crates/engine/src/exec/img_evict.rs`
  - leave-gap 不再调用 `execute_k_shift`；
  - 保留 `rebase` 模式旧路径。

### 服务面

- `apps/server/src/openai.rs`
  - request dump 已就绪（`[server] request_dump_dir`）；
  - 不需要协议改动。
- `profiles/full.toml`
  - 新键显式；
  - `leave_gap` 测试档单独开。

---

## §11 分阶段施工（建议顺序）

### P0：位置账 + CPU 纯测（0.5–1 天）

- `PositionLedger`；
- 新序列构造（删除图片、保留幸存 position、追加新尾巴）；
- 单元测试：
  - 删除中间图，survivor positions 不变；
  - 新尾 position = max+1；
  - 文本/图像 mRoPE；
  - reset/commit 账一致。

### P1：0.8B GPU leave-gap（1–2 天）

- 0.8B VLM，三图窗口 20 轮；
- 关闭 spec、关闭 prefix 跨会话；
- 验证：
  - 不再调用 K shift；
  - 每轮只 prefill 新尾巴；
  - 输出/困惑度与 baseline 对照；
  - 位置洞不产生 NaN/越界。

### P2：前缀缓存 + 快照（1–2 天）

- positioned chain key / 禁用跨会话；
- snapshot key epoch；
- 20–100 轮 cached 稳定；
- 删除后 `cached` 不塌。

### P3：27B TP2 + spec（2–4 天）

- 3080 TP2，三图窗口 100 轮；
- spec verify/propose/MTP 的 position；
- TTFT / decode / AL；
- 无 Xid。

### P4：consolidation（1 天）

- 触发/全量 prefill/epoch；
- 300 轮长跑；
- 先同步，后考虑 idle。

### P5：切默认（验收后）

- `image_evict_mode = "leave_gap"` 进 3080 GUI 档；
- 保留 `rebase` 回退；
- 更新 `profiles/README` 与 `prefix-deletion.md`。

---

## §12 验收线

| 指标 | 当前 rebase | leave-gap 目标 |
|---|---:|---:|
| 删除触发全量 prefill | 每 ~5 次 | **0（除 consolidation）** |
| 三图 100 轮 cached | 95% 热轮，周期 0 | **≥95% 全程** |
| TTFT 中位 | ~2s，周期 31–43s | **≤2s，无周期尖峰** |
| KV 内存 | 有界 | 有界 |
| FP8 K 重转 | 有 | **0** |
| consolidation | 每 5 次 | **0/100 轮，≤1/300 轮** |
| decode/AL | 111 t/s / 4.6 | 不退化 |
| 3090 生产档 | 不动 | 不动 |

---

## §13 风险与未决问题

1. **模型质量**：position 洞是否影响 Qwen3.8 的 GUI 任务；需 0.8B/27B 对照，
   不能只看 loss。
2. **m-RoPE 图像**：`vision_position_ids(base, grid)` 的 base 加法是否
   与 HF 语义完全一致；需要和 HF `get_rope_index` 对拍。
3. **GDN**：删除后的 GDN state 仍含旧图记忆；这是既有语义，但 leave-gap
   不重置；consolidation 才重置。
4. **prefix cache**：position-aware key 的实现成本；P0 可以先禁用
   positioned 会话的跨会话命中，同会话 block_table 继续复用。
5. **spec**：DFlash2/MTP 的 verify 位置、槽表、图输入；必须单独门。
6. **max_position**：`runtime.max_seq` 是 token 容量，不等于 position 上限；
   需要单独的 consolidation 阈值。
7. **共享块**：删除只摘自己的 block；旧链死枝回收仍走 store 的叶/LRU 规则。
8. **pads**：P0 保留 pad 律；未来可评估去掉 pad 以减少 token 开销。

---

## §14 交接检查表

开始实现前：

```text
[ ] git status 干净；当前 commit / 未提交项明确
[ ] 读 §4/§5/§6/§7
[ ] 先写 PositionLedger 单测，不碰 GPU
[ ] 用 0.8B + 三图窗口做 P1
[ ] 保留 rebase 模式，不删旧路径
[ ] 不在 3090 生产档直接切默认
```

第一个 PR 的最小闭环：

```text
PositionLedger + 删除后 survivor positions 不变
+ ChunkTables 用显式 positions
+ 0.8B 三图 20 轮不再调用 K shift
```

---

## §15 参考

- 当前实现与漂移数据：`docs/devlog/memory/prefix-deletion.md` §5quinquies/§5sexies
- 本轮 TTFT 基线：`docs/devlog/serving/ttft-prefill-round-20261025.md`
- 现场：`perf-runs/ttft-prefill-20261025/`
- vLLM leave-gap / retention：
  <https://github.com/vllm-project/vllm/pull/53876>
- vLLM block-range eviction：
  <https://github.com/vllm-project/vllm/pull/53871>
- vLLM RSQR（raw survivor / query-side rotation）：
  <https://github.com/vllm-project/vllm/pull/57672>
- vLLM session eviction + re-rotation plumbing：
  <https://github.com/vllm-project/vllm/pull/43374>
- NVIDIA kvpress key rerotation：
  <https://github.com/NVIDIA/kvpress/blob/main/kvpress/presses/key_rerotation_press.py>

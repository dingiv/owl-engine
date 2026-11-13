# 交接：前缀命中已修，TTFT 还差 prefill 固定税（2026-11-13）

> **后续更新（2026-11-13 晚）**：本文 P0 的“92 tok 一块 195ms”已被拆细 ——
> 实际是**页对齐/消息边界把 82 tok 切成 80+2 两块**（每块 ~100ms）。
> 调度器已修：已有锚点的小尾块（≤1024 tok）不再为快照切碎 ⇒ 文本 append
> 195→105ms / TTFT 235→145ms，3 图 542→456ms / 794→611ms。
> 详见 `2026-11-13-ttft-append-scheduler-fix.md`；本文 P0 的“图”仍是
> 单块 105ms 的下一刀，但 int8 图有 `L/base_pos` 烘死的新阻断（同卷 §5）。

> 关联：
> - `docs/devlog/serving/2026-11-12-session-aware-kv-snap-设计.md`
> - `docs/devlog/serving/2026-11-12-snapshot-admission-handoff.md`
>
> worktree：`packages/owl-worktrees/session-kv-snap`
> 分支：`feat/session-kv-snap`（基于 `main@39ed54e5`，**全部未提交**）

---

## 0. 一句话现状

- 前缀缓存/快照准入/leave-gap 的“只算新增消息”已经打通：
  append 轮 `cache=36672/36764 (99.7%)`，只 prefill 92 tok。
- 新增了**服务面图像预处理 LRU**，3 图场景 client TTFT **2076ms → 794ms**。
- 但 **92 tok 的 prefill 仍要 542ms（带图）/195ms（纯文本）**，离用户要求的
  **<100ms** 还差一个 **prefill 固定 host 税**。
- 这不是配置能解的：`prefill_chunk=128` 实测 append 不变（195ms），cold 30k
  反而 3.8s→20.5s。下一步必须动 prefill 的 graph/persistent kernel 快路径。

---

## 1. 运行态（交接时）

| 项 | 值 |
|---|---|
| 8135 server | PID **280577** |
| binary | `/home/div/Documents/codes/packages/owl/target/release/server`（worktree 构建） |
| profile | `profiles/qwen38_27b-3080_tp2.toml`（cwd = worktree） |
| log | `/tmp/owl-8135-snap9.log` |
| request dump | `/tmp/owl-req-8135-snap9/` |
| 3090 server | PID **201670**（未动） |
| 主二进制备份 | `/tmp/owl-main-server-backup-20261113-013520` |

> 注意：`packages/owl/target` 是共享 target；当前 release binary 是 **worktree**
> 构建。主仓库若要恢复旧 binary，用上面的备份；但 worktree 源码优先。

构建：

```bash
cd /home/div/Documents/codes/packages/owl-worktrees/session-kv-snap
GDN_TRITON_PY=/home/div/Documents/codes/packages/owl/crates/kernels/.venv/bin/python \
CARGO_TARGET_DIR=/home/div/Documents/codes/packages/owl/target \
  cargo build --release -p server
```

启动：

```bash
cd /home/div/Documents/codes/packages/owl-worktrees/session-kv-snap
setsid nohup /home/div/Documents/codes/packages/owl/target/release/server \
  -p profiles/qwen38_27b-3080_tp2.toml \
  --set server.request_dump_dir=/tmp/owl-req-8135-snap9 \
  --log-level info > /tmp/owl-8135-snap9.log 2>&1 < /dev/null &
```

---

## 2. 本轮已落地（worktree 未提交）

### 2.1 gapped 链同会话复用

- `AgentSession.block_hashes: Vec<ChainHash>`：逐块内容身份，随 capture /
  leave-gap / 前缀命中维护；长度 = `chain.blocks()`。
- `sched::session_local_hit()`：用会话自己的 `block_table + block_hashes`
  构造 `ChainHit`，绕过共享树（gapped 链不发布，位置基准不在身份里）。
- `plan_prefix_reuse`：positioned 会话优先 session-local；**逐块截断**：
  从共同前缀最后一块往前找第一个 tags/positions 不一致的块，只复用到该块前。
  修了“新截图最后一块不同 ⇒ 整个 session-local 被否决 ⇒ 回退树命中 ⇒ m=0”。
- 指标：`kvmem.session_local_hit/blocks`；日志：
  `session-local 截断: full=... usable=...`、
  `session-local 未用: chain=... bh=... table=... common=... tree_m=...`。

### 2.2 leave-gap 用会话自己的旧链

`plan_image_leave_gap` 原来用 `match_prefix_tagged` 取旧链；gapped suffix 不在
共享树里 ⇒ `matched` 只到删点前 ⇒ `keep=0` ⇒ 整个 130k 重算（snap7 turn 29:
`leave-gap 后缀边界无快照 keep=0` + `matched=2455 m_raw=0` + TTFT 72s）。

现在它先用 `session_local_hit(s, page, s.tokens.len())` 拿完整旧链；只有本地
账不齐才退回树。`release_chains` 里未知哈希的 `drop_chain` 幂等（false），安全。

### 2.3 服务面图像预处理 LRU

`apps/server/src/vision_processor.rs`：

- `from_data_url` 增加全局 LRU（容量 8），key = data URL + `max_pixels`；
- 同一张图跨轮重发不再 base64→解码→resize→patchify；
- 3 图场景 client TTFT **2076ms → 794ms**；
- `vlm.cost` 日志：append 轮 `新图=0 | tower=0.0ms | combine=1.8ms`（塔缓存
  本来就在 `RunningEngine.vlm_cache` 跨轮命中）。

### 2.4 配置

- `vram_target` → **`vram_budget`**（代码/配置/文档全改；旧键走退役键门，
  报错提示新键）。`vram_budget_mb` 是绝对硬上限，二者不同量。
- 3080/3090 leaf 删掉与 full 默认重复的 4 个键：
  `prefix_cache` / `snap_msg_delim` / `snap_msg_min_gap` / `snap_msg_min_span`。
- 修正 full.toml 的旧 INT8 注释、gdn_slots 注释、snap_max vs snap_session_cap
  区分、vram_budget vs vram_budget_mb 二选一说明。
- `full.toml` 仍是全量模板（完整度门要求每个字段显式）；生产 leaf 只写差异。
- 3080 实际 `--print-config`：`vram_budget=1.0`（parts/pool/int8.toml 的
  30 系最优值），`gdn_slots=2`，`snap_max=6`，`snap_session_cap=20`。

### 2.5 快照准入（上一刀，仍在）

`StateStore::admit_snap` + `Rel::drop_snap_asset` + `SnapClass` 档位淘汰 +
`msg_snap_keys` 重分类 + 会话关闭 detach。详见 2026-11-12 两份文档。

---

## 3. TTFT 实测（自测，36k prompt + ~100 字符 append）

| 场景 | client TTFT | server prefill | 备注 |
|---|---:|---:|---|
| 3 图，修复前 | 2076 ms | 555 ms / 92 tok | 图像预处理每轮重做 |
| 3 图，vision LRU 后 | 794 ms | 542 ms / 92 tok | VLM tower 0ms；cache 99.7% |
| 纯文本 | 235 ms | 195 ms / 92 tok | 固定 host 税 |
| `prefill_chunk=128` | — | 195 ms / 92 tok | cold 30k 20.5s，append 不变 |

拆解：

```text
542ms（带图） = ~195ms prefill 固定 host 税 + ~350ms 图像 KV attention
195ms（纯文本） = prefill 固定 host 税
```

“固定 host 税” = 每 chunk ~6395 条 alloc/launch/free（见
`exec/prefill_chunk.rs` / `parallel/tp.rs` 注释），与 chunk token 数基本无关。
`decode=134ms/7tok` 是 7 个生成 token，不是 TTFT 主项。

测试脚本：

```text
/tmp/ttft_append_test.py
  base 图像：/tmp/owl-req-8135-snap7/req-1794511419210289169-262930-0.json
  base 纯文本：/tmp/owl-text-36k.json
```

脚本会 cold 一次，然后每轮 append 100 字符的 user 消息，SSE 测首 token。

---

## 4. 下一步（按优先级）

### P0：prefill 固定 host 税（要 <100ms 的唯一硬仗）

> **2026-11-13 晚更新**：先修了“切碎税”（见新卷 §1–§3），单块 82 tok 现已
> 105ms；本文下面的 195ms 是**两块**的旧账。单块拆解与 int8 图阻断见新卷 §4–§5。

现状：92 tok append 的 prefill 195ms（文本）/542ms（图）。目标 <50ms 量级。

- 入口：`crates/engine/src/exec/prefill_chunk.rs`（eager 路，注释明确
  “不含图路”）、`crates/engine/src/parallel/tp.rs`（prefill 桶族已退役
  2026-10-21）、`docs/devlog/graph/prefill-graph.md`、
  `docs/devlog/graph/prefill-bucket-qslot.md`、
  `docs/devlog/perf/w4a16-prefill-bucket-ab.md` / `w4a8-prefill-bucket-ab.md`。
- 历史结论：prefill 桶图对**长 chunk** 无收益（A/B 已判）；但**短 append**
  （≤128/256 tok）可能只有它能消掉 6395 条 host 发射。建议做**小 chunk 专用**
  graph/bucket，长 chunk 保持 eager，不要全局改 `prefill_chunk`。
- 注意 FI plan 的 `fi_plan_sync` / `fi_plan_restore` 与动态 KV 长度；
  decode/verify 的图路可参考 `fi_cuda_graph` / `fi_plan = "baked"`。
- 验收：同一 36k base，append 100 字符，`prefill < 50ms`，client TTFT
  `< 100ms`（文本先达标；图再叠 P1）。

### P1：图像 KV attention 的 ~350ms

- 文本-only 195ms vs 带图 542ms；VLM tower 已 0ms、预处理已缓存 ⇒
  剩余在 **92 q × 36k KV（含 6120 图像 token）的 attention/GDN 路径**。
- 对比 text-only/image 的 `prefill`、attention kernel 选择；查图像 token 的
  K/V 布局、int8 解量化、是否走了不同 mask/分派。
- 不要先动 `max_pixels`/降分辨率；那是质量换速度，用户没批准。

### P2：复现 leave-gap 修复

- snap7 的 turn 29 失败是旧 binary；snap8/9 已加 session-local 旧链 + 逐块截断，
  但**还没在真实“删图 + 新截图”多轮上复现成功**。
- 用 snap7 的 dump 或自建多轮：先让会话进入 positioned，再发带 `owl_evict`
  的删除请求，检查：
  - `leave-gap: ... 保留前缀 2455 + 后缀 ...`（不是 `keep=0`）；
  - `session-local 截断: full=... usable=...`；
  - `m_raw` 不再为 0；
  - 最终 `cache` 接近上一条 prompt 的整块边界。

### P3：快照容量/宿主预算

- 单会话 cap 20、host 3G；130k 长会话 + 多图时二分锚点可能被逐。
- 若 P2 复现后仍出现“删点前无快照”，评估：
  - `snap_session_cap` / `snap_host_mb` 上调；
  - 二分 `Binary` 锚点的保留策略（不是 pin，只是档位/配额）；
  - `msg_snap_keys` 的 L1 老化与 orphan 回收。

### P4：配置完整度门

- 用户明确不想在 leaf 里堆配置。当前 `full.toml` 仍是全量模板，
  `Completeness::Enforce` 缺字段拒启。
- 如果要做“只写硬件/模型事实”，需要放宽 `Completeness::Enforce`（单独一刀，
  会影响“启动点可复现”约束），不要偷偷加 `#[serde(default)]` 绕过。

### P5：3080 功耗 250W（未装车）

- 改好的脚本：`/tmp/cap-3090-power.sh`（3090/3090 Ti 350W，3080 250W，
  4080 跳过）。
- 当前文件 `/usr/local/sbin/cap-3090-power.sh` 是 root:root 0755，agent 没有
  免密 sudo，**尚未安装**。需要用户执行：
  ```bash
  sudo install -m 755 /tmp/cap-3090-power.sh /usr/local/sbin/cap-3090-power.sh
  sudo /usr/local/sbin/cap-3090-power.sh
  ```
- 验证：`nvidia-smi --query-gpu=index,name,power.limit --format=csv,noheader`。

---

## 5. 不要重复的弯路

1. 不要再怀疑 prefix cache/快照准入：append 轮 `cache=99.7%`，prefill 只有 92 tok。
2. 不要把 `prefill_chunk` 全局改小：128 实验 append 不变，cold 30k 慢 5 倍。
3. `vram_target` 已退役；旧档会响亮拒。新键 `vram_budget`。
4. session-local 的 tags/positions 必须**逐块截断**，不能整前缀一票否决。
5. `plan_image_leave_gap` 必须用 session-local 旧链，不能只用共享树。
6. 图像预处理 LRU 在服务面；VLM tower 缓存本来就有，别重复做。
7. 3090 server（PID 201670）不要动；它跑主仓库 binary + 主 local profile。

---

## 6. 已验证的测试

```bash
# 快照/准入/重分类/会话关闭
GDN_TRITON_BUILD=0 CARGO_TARGET_DIR=/home/div/Documents/codes/packages/owl/target \
  cargo test -p owl-store --lib
# 176 passed

# msg boundary / snap_guard / carry
GDN_TRITON_BUILD=0 CARGO_TARGET_DIR=/home/div/Documents/codes/packages/owl/target \
  cargo test -p owl-engine --lib msg_boundary
# 7 passed
GDN_TRITON_BUILD=0 CARGO_TARGET_DIR=/home/div/Documents/codes/packages/owl/target \
  cargo test -p owl-engine --lib snap_guard
# 2 passed
GDN_TRITON_BUILD=0 CARGO_TARGET_DIR=/home/div/Documents/codes/packages/owl/target \
  cargo test -p owl-engine --lib carry_msg_snap_keys
# 1 passed

# 配置/档面（独立 target，避免共享 target 误读主仓库 profiles）
GDN_TRITON_BUILD=0 CARGO_TARGET_DIR=/tmp/owl-shared-worktree-target \
  cargo test -p owl-shared --lib
# 132 passed
```

> 需要模型资产的 engine 测试（如 `snapshot_victim_follows_store_order`）在
> worktree 里因 `crates/models/assets/Qwen3.5-0.8B` 缺失跑不了；存量环境问题。

---

## 7. 关键文件索引

```text
crates/engine/src/sched/scheduler.rs       session_local_hit / plan_prefix_reuse /
                                           plan_image_leave_gap / 逐块截断
crates/engine/src/serve/session.rs         AgentSession.block_hashes / carry_msg_snap_keys
crates/engine/src/exec/spec.rs             capture 回填 block_hashes / 快照准入
crates/engine/src/exec/turn_end.rs         L1 写回 / 旧 tail 降级
crates/engine/src/running.rs               submit 重分类 / close_session detach
crates/store/src/state.rs                  admit_snap / reclassify / detach
crates/store/src/rel.rs                    drop_snap_asset / 包装
crates/store/src/hooks.rs                  order_state_in_session
apps/server/src/vision_processor.rs        data URL LRU
crates/shared/src/config/mod.rs            vram_budget / 退役键
profiles/full.toml                         全量模板 + 注释清理
profiles/qwen38_27b-3080_tp2.toml          20/4、gdn_slots=2、snap_max=6
```

---

## 8. 交接时未决问题（给下一个会话）

1. **prefill host 税**：小 chunk graph 是否可行？先做文本-only 92 tok <50ms 的
   最小原型，再叠图。
2. **图像 KV attention 350ms**：文本 vs 图的 prefill 差异到底在 attention
   kernel 还是 GDN/图像布局。
3. **leave-gap 修复的真实复现**：需要一次带 `owl_evict` 的多轮删图回放，
   确认 `keep=0` 不再出现。
4. **快照 cap/host**：130k 会话的二分锚点是否够用；是否要上调
   `snap_session_cap` / `snap_host_mb`。
5. **配置完整度门**：是否放宽以允许极简 leaf（用户倾向“默认最优、少配置”）。

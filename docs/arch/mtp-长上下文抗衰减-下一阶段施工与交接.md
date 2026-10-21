# MTP 长上下文抗衰减 —— 下一阶段施工与交接

> 2026-10-21 · 交接文档
> 工作树：`packages/owl-worktrees/mtp-spec`
> 分支：`feat/mtp-spec` @ `9bb7414`
> **Phase 0（Windowed-MTP）+ T1a/T1b v2（自适应 depth）已落地；本文接续未完成验证与 Phase 2。**
>
> 阅读顺序：
> 1. `packages/owl/AGENTS.md`
> 2. `docs/arch/mtp-长上下文抗衰减-设计与施工.md`（总设计与 Phase 0 细节）
> 3. 本文（任务板 / 施工卡 / 交接纪律）
> 4. `perf-runs/mtp-longctx-2026-10-21/REPORT.md`（本地原始数据，未入库）

---

## 0. 一句话交接

长上下文 MTP 的 **草稿读成本** 已由 Windowed-MTP 压到平台（64k `propose.read`
10.7 → 6.1ms，−43%），生产 MTP 档已写 `mtp_window=4032` / `mtp_sink=64` /
`mtp_window_min_ctx=12288`。T1a 两档图族与 T1b v2 两连低窗门已落地（默认关）。
下一阶段的主线是：

```text
T0 已收尾（W=4032 最优；W=8192 已补）
  → T1b v2 已落地（两连低窗；DOCS 68.91 / 16k 不误切）
  → 采样/TP2 补测
  → T3 MTP 专用 C7（需先补草稿 KV 追赶，不是小改）
  → T2 置信度截断（DSpark 式 confidence-scheduled verify）
  → T4 采样域 rejection / T5 ring buffer / T6 tree / T7 头量化
```

**采样首轮已测（REPORT §9）；T2/T3 立项卷已写。下一施工 = T3 草稿 KV 追赶。**

---

## 1. 当前状态

### 1.1 Git / 工作树

- 分支：`feat/mtp-spec`
- HEAD：`9bb7414`（merge latest main）
- **大量改动未提交**：Phase 0 代码、设计文档、profile 重命名、测试引用同步
- **不要 `git checkout -- .` / `git reset --hard`**：这些改动是连续的施工结果
- 未 commit / 未 push（遵守 AGENTS：不主动提交/推送）

新增文件：

```text
tools/mtp_curve.py
crates/engine/src/exec/mtp_window.rs
docs/arch/mtp-长上下文抗衰减-设计与施工.md
docs/arch/mtp-长上下文抗衰减-下一阶段施工与交接.md   # 本文
```

核心改动：

```text
crates/shared/src/config/mod.rs          SpecCfg.mtp_window/mtp_sink + 校验
crates/engine/src/engine.rs              DraftCfg / SpecBootInput / boot v2 门
crates/engine/src/spec_runtime.rs        SpecRuntime.mtp_window + 模式/深度校验
crates/engine/src/exec/mod.rs            mod mtp_window
tools/mtp_curve.py
crates/engine/src/exec/mtp_window.rs     MtpWindowPlan（纯函数 + 9 单测）
crates/engine/src/exec/draft.rs          propose_first/round/graph_step 接线
crates/engine/src/graph_fleet.rs         chain c{j}_bt 输入槽
crates/engine/src/tests.rs               window 恒等门（长 prompt）
profiles/full.toml                       schema 缺省 0/64
profiles/qwen38_27b-3090-w4a16-mtp.toml  生产值 4032/64
docs/dev/config.md / debug-tools.md / profiles/README.md
```

### 1.2 已落地能力

- MTP 草稿读视图：`sink=64 + 最近 W=4032`；target verify 全上下文。
- 三条 propose 路径都接：
  - `propose_first`
  - `propose_round`
  - `propose_graph_step`
- 图态 chain 新增 `c{j}_bt` 输入槽（否则 chain 仍读完整 `bt_mtp`）。
- 写路径保持完整链：`prefill_extend` / `write_bt_mtp` 不 window。
- `full.toml` 只保留 schema 缺省 `0/64`；性能值只写具体 MTP 档。
- 生产 MTP 档已转正：`profiles/qwen38_27b-3090-w4a16-mtp.toml`：
  ```toml
  [spec]
  depth = 3
  draft = "mtp"
  mtp_window = 4032
  mtp_sink = 64
  ```
- 回退：`--set spec.mtp_window=0`。

### 1.2b T1a 自适应 depth（2026-10-21 晚已落地）

- 配置：`spec.depth=7` + `spec.adaptive_depth=true` + `spec.depth_min=3` +
  `spec.depth_switch_ctx=49152`（缺省关，旧档不变；`full.toml` 已补 schema 缺省）。
- 图族：`GraphFleet.depths: Vec<DepthGraphs>`，每档独立 verify/fold/propose；
  d3+d7 两档共 24 个 face（单档 16）。只补 verify 而共用 propose/fold 是假自适应。
- 调度：`StepAction::SpecRound { depth }` 显式携带 active depth；热路径按 depth 选图。
- 首轮：MATH 8.1k d7 `139.18` / AL 5.023；MATH 31.9k d7 `110.41` / AL 3.981；
  MATH 64.4k d3 `92.93` / AL 2.442；DOCS 56.8k d3 `69.41` / AL 1.559。
- **8k 窗口回归已修**：`mtp_window_min_ctx=12288`（8k+输出跨 8192，阈值 8192
  仍会压到尾部）；修复前 126.60/AL 4.521。
- **DOCS 29.7k 域漂移**：d7 `62.42`/AL 1.796 vs 静态 d3 `66.09`/AL 1.398。
  d7 AL 更高但成本更大 ⇒ 静态 ctx 阈值不能直接转正，T1b 必须做 AL/成本门。
- **T1b v2（两连低窗）**：turn 级 8 轮窗口，`tokens/round < adaptive_min_tpr`
  （缺省 3.0）记低窗；**连续 2 窗低**才降 d3、只降不升。DOCS 29.7k 实测
  `68.91`（无门 62.42，静态 d3 66.09），约 16 轮内降档；16k MATH `105.23`
  ≈ 静态 d7 105.46，不误切。4.0 不需要（两连低窗下不会更早）。
- 恒等门：`OWL_GATE_SPEC_DEPTH=3` 与 `=7` 各一次，均 PASS（window 4032）。
- 未测：采样域、TP2、GGUF MTP；16k 自动档未测。

### 1.3 首轮实测（Phase 0）

口径：单 3090 Ti（`42:00.0`），greedy，256 tok，单并发，MATH 词池；
`window=0` 为同二进制 full 对照。

| 点 | decode full → win | AL full → win | propose.read full → win |
|---|---:|---:|---:|
| MATH 8k（win 1 轮） | 113.60 → **114.99** | 2.688 → **2.718** | 6.39 → **6.08** |
| MATH 32k（2 轮） | 102.46 → **106.69** | 2.693 → **2.601**（−3.4%） | 8.22 → **6.09** |
| MATH 64k（2 轮） | 86.62 → **93.60** | 2.545 → **2.424**（−4.8%） | 10.67 → **6.08** |
| DOCS 56.8k（1 轮） | 58.58 → **69.39** | 1.324 → **1.574**（+18.9%） | 11.18 → **6.08** |

结论：

- `propose.read` 在 32k/64k 被压到 **~6.1ms 平台**，不再随 ctx 线性增长。
- MATH 64k decode **+8.1%**（低于设计原定 +10% 门），AL 只掉 **4.8%**。
- DOCS 长文本 **AL +18.9% / decode +18.5%**（窗口去掉远距离稀释）。
- 短 ctx 仍 full；512 点无回归。

**T0 补测（W=6144，2026-10-21 晚）**：MATH 64k `92.79` / AL `2.401`，与
4032 基本持平；DOCS 56.8k 单轮 `62.53` / AL `1.304`，低于 4032，但只有 1 轮。
结论：**4032/64 继续作为生产值**；W=8192 仍被外部 `SIGTERM` 杀掉，未取得数据。

**已知缺口**：W=8192 的 64k 对照两次被外部会话 `SIGTERM`，尚未补；
DOCS 29.7k 已补首轮（d7 62.42 < 静态 d3 66.09，见 T1a 节）；采样域未测。

### 1.4 验收现状

已通过：

```text
cargo check --workspace                                      ✅
cargo test -p owl-shared --lib                               ✅ 117/117
cargo test -p server --test config_check                     ✅ 6/6
cargo test -p owl-engine --lib mtp_window                    ✅ 9/9
OWL_GATE_MTP_WINDOW=0      f16 恒等门                        ✅
OWL_GATE_MTP_WINDOW=4032   f16 恒等门（长 prompt >4k）       ✅
OWL_GATE_MTP_WINDOW=4032   fp8 主池恒等门                    ✅
```

恒等门命令：

```bash
GDN_TRITON_PY=/home/div/Documents/codes/packages/owl/crates/kernels/.venv/bin/python \
OWL_TEST_DEVICE=42:00.0 \
OWL_E2E_ONLY=gpu_spec_identity_gate_27b \
OWL_GATE_MTP_WINDOW=4032 OWL_GATE_MTP_SINK=64 \
cargo test -p owl-engine --features e2e --test engine_e2e
```

> `OWL_TEST_DEVICE` 必须是 **BDF**（`42:00.0`）；纯数字会被拒。

---

## 2. 环境与操作纪律（必读）

### 2.1 外部会话会击杀 GPU 进程

本机存在另一个 `packages/owl-worktrees/gguf` 会话，其脚本在启动前执行：

```bash
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
  kill "$p" 2>/dev/null
done
```

本会话的 server 已被误杀两次（DOCS 29.7k 第二轮、W=8192 对照中断）。
**下一会话起服前先查：**

```bash
nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_memory --format=csv
ss -ltnp | grep -E ':8140|:8141'
```

规矩：

- 不杀别人的 server；端口被占就换自己的端口（如 8141）。
- 优先用 **GPU2 `42:00.0`**（本会话测试卡）；GPU0/1 常有外部会话。
- 若外部脚本正在跑，宁等不抢；单次 A/B 被误杀后先记录，不要重跑污染结论。

### 2.2 端口

- 外部 `gguf` 会话常用 `8140`。
- 本会话 A/B 用 `8141`：
  ```bash
  --set server.bind=127.0.0.1:8141
  ```
- 用 `ss -ltnp` 确认自己服务的 PID；不要 `pkill -f`。

### 2.3 构建

```bash
cd packages/owl-worktrees/mtp-spec
GDN_TRITON_PY=/home/div/Documents/codes/packages/owl/crates/kernels/.venv/bin/python \
  cargo build --release -p server --features perf
```

工作树缺 `crates/kernels/.venv` 时借用主树；不要为建 venv 重装 triton。

### 2.4 配置纪律

- 性能值写具体 profile；`full.toml` 只保留 schema 缺省。
- 不用 env 控制引擎行为；测试面 `OWL_GATE_MTP_WINDOW` 仅诊断门用。
- 改动后先 `--check`，再 `--print-config --format json` 确认生效值。
- 不主动 commit / push。

---

## 3. 任务板

| ID | 优先级 | 任务 | 量级 | 依赖/阻塞 |
|---|---|---|---|---|
| **T0** | ✅ | W=6144/8192 扫点（MATH/DOCS 64k） | S | 4032 最优；数据在 `REPORT.md` §6.1 |
| **T1a** | ✅ | 静态 ctx 自适应 depth（d3/d7 全套图族） | M | 8k/16k/32k/64k + DOCS 已补 |
| **T1b** | ✅ | 两连低窗 v2（DOCS 68.91 / 16k 不误切） | M | 采样/TP2 后接 EMA |
| **T2** | P2 | 置信度截断 / confidence-scheduled verify | M | 立项卷已写；需先加 `topk/conf` 输出 |
| **T3** | P1.5 | MTP 专用 C7 / 自动降级 | M | 首轮试做回退；`gpu_mtp_kv_parity` PASS（t=1 无罪）⇒ 查 seed 同源/链页表 |
| **T4** | P3 | 采样域 rejection / 草稿分布 | M | 采样首轮已测（REPORT §9） |
| **T5** | P3 | MTP KV ring buffer / 物理回收 | M | window 读视图稳定 |
| **T6** | P4 | Tree / 多候选 MTP | L | verify 行预算 |
| **T7** | P3 | MTP 头 W4A16/W4A8 | M | 恒等/AL 门 |
| **T8** | P1/P2 | GGUF MTP / IQ4XS MTP / VLM / TP2 / 128k 验证 | S–M | 各自资产/卡 |
| **T9** | 研究 | FOLD / DSpark / L-MTP / AdaMTP / post-training / Frozen-KV | XL | 训练/新 checkpoint |

**建议执行序：T3（草稿 KV 追赶）→ T2（conf 输出 + 浅档 verify）→ T4/T5/T7 → T6；T9 不混入本轮工程。**

---

## 4. T0：W 扫点收尾

### 结论（2026-10-22 已收尾）

- 生产值：`4032/64`；W=6144/8192 均已补（`REPORT.md` §6.1）。
- MATH 64k：4032 `93.60` / 6144 `92.79` / 8192 `93.77`（基本持平）。
- DOCS 56.8k：4032 `69.39` / 6144 `62.53` / 8192 `63.82`（**4032 最优**）。
- 测量脚本已收编：`tools/mtp_curve.py`（不再依赖 `/tmp`）。

### 命令

```bash
# 复现（脚本 = tools/mtp_curve.py；GPU2/8141 避开外部 8140）
./target/release/server \
  -p profiles/qwen38_27b-3090-w4a16-mtp.toml \
  --set server.bind=127.0.0.1:8141 \
  --set runtime.max_seq=73728 --set pool.pool_tokens=73728 \
  --set spec.mtp_window=6144 --set spec.mtp_sink=64

# 64k MATH（2 轮；W=4032 已有基线）
python3 tools/mtp_curve.py --url http://127.0.0.1:8141 \
  --out perf-runs/mtp-longctx-2026-10-21/w6144 \
  --length 65536 --rounds 2

# 64k DOCS（1–2 轮；prompt-file 同 REPORT）
python3 tools/mtp_curve.py --url http://127.0.0.1:8141 \
  --out perf-runs/mtp-longctx-2026-10-21/w6144-docs \
  --length 65536 --rounds 1 --domain prose \
  --prompt-file perf-runs/mtp-longctx-2026-10-21/corpus-docs.txt

# 再对 W=8192 重复
```

### 判据

- AL ≥ full 的 95%（同 ctx/同域）。
- 64k decode 尽量 ≥ full +10%；DOCS 不应回退。
- `propose.read` 随 W 增大而上升，但应仍显著低于 full。
- 若 4032 仍最优，保持 profile；把数据写入 `REPORT.md` / 设计文档 §4.13。

### 交接提示

- 先跑 T0 再动 T1；T1 的 EMA 阈值需要这些点。
- 不要同时跑两个 27B server（24G 卡贴顶）。

---

## 5. T1：自适应 depth（T1a ✅ / T1b v2 ✅）

### 目标

短 ctx / 高 AL 用 d7，长 ctx / 低 AL 用 d3；至少不劣于两静态档的最优。

### 已知事实

- MATH：d7 8k `132.79`、32k `107.80`、64k `84.04`；
  d3 对应 `113.60`、`102.46`、`86.62`。
- 交叉点约 **50k**；d7 `pos7` 从 8k `.389` 掉到 64k `.218`。
- 固定 d3：8k 少赚 ~17%；固定 d7：64k 少赚 ~3%，且 d7 的 propose 成本约 d3 的 2×。

### 当前结构（施工锚点）

- `crates/engine/src/graph_fleet.rs`
  - `GraphFleet.verify: Option<GraphPlan>`（单张，T=depth+1）
  - `plan_verify()`：`let depth1 = boot.spec.depth + 1;`
  - `plan_propose_family()`：`m=0..spec_depth` 桶；`extra_faces` 按 depth 预算
- `crates/engine/src/exec/spec.rs::execute_spec_round(..., depth)`
  - verify 用 `graphs.verify_mut(depth)`
  - propose 用 `graphs.propose_mut(depth, m)`
  - fold 用 `graphs.fold_mut(depth, rows)`
- `crates/engine/src/scheduler.rs`
  - `StepAction::SpecRound { depth }`（轮级 active depth 显式携带）
- `crates/engine/src/spec_runtime.rs`
  - `SpecRuntime.depth`（上限）+ `depth_min/adaptive_depth/depth_switch_ctx`
- `crates/engine/src/turn.rs`
  - `TurnSpecState`（turn 域 spec 状态；不要把 active depth 放进跨 turn 的全局）

### T1a 实施结果（2026-10-21）

- 已加 `SpecCfg.adaptive_depth/depth_min/depth_switch_ctx`（缺省 false/3/49152）。
- `GraphFleet` 改为 `depths: Vec<DepthGraphs>`，每档 verify/fold/propose 全独立；
  `fleet_faces([3,7]) = 24`（原 d7 单档 16）。
- `SpecRound { depth }` + `execute_spec_round(..., depth)` 显式传递；
  `acquire_drafts`/`propose_first`/`propose_round`/`propose_graph_step`/
  `verify_round`/`spec_fold_to` 全链按 active depth 选图。
- 恒等门新增 `OWL_GATE_SPEC_DEPTH=3|7` 参数；两档均 PASS。
- 目前只测 32k/64k MATH；8k/DOCS 未测，生产 profile 仍是静态 d3+window 4032。

### T1b 施工卡（下一步）

1. T1b v2 已落地：每 8 轮窗口，连续 2 窗 `tokens/round < adaptive_min_tpr`
   （缺省 3.0）才只降不升到 depth_min。DOCS 29.7k 68.91（无门 62.42），
   16k d7 105.23（不误切）。
2. 下一步：补采样/TP2，再上 EMA + hysteresis；4.0 不需要（两连低窗下不会更早）。
3. 切档只换 `DepthGraphs` 引用；禁止在线重捕。
4. 门：
   - 8k/32k 自动选 d7，64k 自动选 d3；
   - 每点不劣于对应静态最优 >2%；
   - 恒等门 active=d3 / d7 各一次（已过）；
   - 切档无图重捕、无 CUDA 错误。

**T1b EMA + hysteresis（后续）**

- 参考 SGLang `adaptive_spec_params.py`：
  `target_steps = clamp(round(ema_accept_len)+1, min, max)`；
  warmup / update interval / up/down hysteresis / ceiling。
- 以 `spec.accepted/spec.rounds` 的滚动窗口为输入；不要用单轮 m。
- 切档时只换已捕获图引用；`propose`/`fold`/`verify` 三面必须同档。
- 先按 batch=1 做；多请求/BS 分档后置。

### 风险

- **图族爆炸**：d3 + d7 两套 propose/fold/verify，`extra_faces` 和 slab 预算
  都要重算；先在 GPU 上看 boot `slab 总账` 与 `vram-gov`。
- **切档污染**：active depth 必须 turn/轮级，不能跨会话残留（参考
  `ActiveTurn::spec` 历史教训）。
- **d7 在 64k 的 AL 真降**：如果 window 后 d7 仍不划算，T1 的收益变小；
  用 T0 数据决定是否只做“静态 ctx 阈值”。

---

## 6. T2：置信度截断 / confidence-scheduled verify

> 立项卷：`docs/arch/mtp-置信度截断-设计与施工.md`（2026-10-22）。

### 目标

按草稿每步置信度裁掉低价值尾部，减少 verify 行数；AL 基本不变。

### 依据

- DSpark（arXiv 2607.05147）confidence head + 按 prefix survival 调度 verify；
- SGLang `dspark_components/dspark_block_accept_estimator.py` 是现成参考；
- AdaMTP 的熵增停止思想可用于推理期截断（但 AdaMTP 本身是训练方案）。

### 当前缺口

- MTP `propose_ext` 图只输出 `drafts`，没有 top1-top2 margin / entropy / q。
- verify 图是固定 T=depth+1；即使 host 截断，仍发同一张图 ⇒ 省不到 verify。
- 需要 **至少两档 verify 图**（如 T=4 / T=8），或动态宽度（不建议首版）。

### 施工建议

1. 在 `crates/models/src/layers/mtp.rs` 的链步 logits 上计算：
   - `margin = top1_logit - top2_logit`，或
   - `entropy` / `top1_prob`；
   - 图输出新增 `conf`（`[depth,1]` F32）。
2. 在 `graph_fleet::plan_propose_family` 加输出槽；`exec/draft.rs` 读取。
3. host 侧定阈值（离线扫）→ `depth_eff`；scheduler 选对应 verify 图。
4. 门：
   - AL 不低于 full 的 95–98%；
   - verify 行数/round 下降；
   - 单请求 decode 有正收益；若无，降级为 batch 优化再评估。
5. 风险：margin 与 target 接受率相关性弱；必须用真实 MTP 头校准，不能拍阈值。

---

## 7. T3：MTP 专用 C7 / 自动降级

> 前置立项卷：`docs/arch/mtp-草稿KV追赶-设计与施工.md`（2026-10-22）。
> **首轮试做（2026-10-22）**：追赶链路可跑但 probe 全 m=0、drafts 与 target
> 无关；已回退，默认安全。**KV 行对拍已做**：`gpu_mtp_kv_parity` PASS，
> t=1 写核无罪；剩余候选 = seed hidden 同源 / 链页表时序。见立项卷 §7–8。

### 现状

`execute_spec_round` 的 B4 降级态机被：

```rust
if self.dev.spec_rt.dflash_tap_count > 0 && self.boot.probes.degrade_after > 0
```

门控。MTP 的 `dflash_tap_count = 0` ⇒ **MTP 目前没有自动降级**。
DOCS/随机域可能净亏，靠 profile 手工关 spec 不够。

### 施工前必须解决的结构性缺口（2026-10-22 发现）

B4 降级期的裸 decode **不写 MTP KV**：

- DFlash2 有 `dflash_tap_count > 0` 分支：decode taps → encode，逐步追草稿池；
- MTP 没有这条臂，且 `spec_seed_hidden` 只在 prefill 末块 harvest 一次；
- 降级后若直接重入 `SpecRound`，`acquire_drafts` 要么用陈旧
  `spec_drafts_host`，要么走 `propose_first` 但 seed 已消费 ⇒ MTP 链有洞。

⇒ T3 不是“把门改成 drafter.is_some()”的小改；必须先做
**MTP 草稿 KV 追赶**（每裸步把 target hidden 写入 MTP 链，或提供可重入的
seed 重放），否则降级 = 静默坏草稿。

### 施工（前置完成后再做）

- 先实现 MTP 裸步的草稿 KV 追赶（或 turn 级禁 spec + 下 turn 强制完整
  prefill 重写 MTP 链）；
- 再把门改成“有草稿器（MTP 或 DFlash2）且 `degrade_after > 0`”；
- 保留现有 `break_even_line` / 窗口 / streak 逻辑；
- MTP 的 `m=0` 不必然等于亏本：继续用窗口 AL vs 盈亏线，不要单看连击；
- 增加 `spec.depth.active` / `spec.degrade` 分域指标；
- 门：MATH/DOCS 不误降级；prose/随机域能自动回退；恒等门不回归。

### 量级

M（不是 S）：草稿 KV 追赶 + B4 门 + 真机 A/B。

---

## 8. T4：采样域 rejection / 草稿分布

### 现状

- greedy 路：target argmax 逐位比对，精确。
- 采样路：target 每行采样后与 draft 比对（`spec_sampled_accept_prefix`），
  分布正确；但 **draft 仍是 argmax**，等价于接受概率 `P_target(draft)`，
  不是标准 `min(1,p/q)` 拒绝采样。
- 512 点采样档 AL 明显低于 greedy（约 1.75 vs 2.65）。

### 施工

- MTP 链步保留草稿分布 q 的 top-k（或全 logits 的 logprob）；
- 标准 speculative sampling：
  `accept ~ min(1, p(x)/q(x))`，拒绝时从 residual `(p-q)+` 重采样；
- 需要 device 侧或 host 侧 q/p 对齐；先做 host 版验证分布，再做图；
- 门：采样分布 parity（固定 seed 的统计门），不是文本逐位；
- 注意：当前 `mtp` 图输出 `drafts` 是 f32 token 槽，新增 q 输出会改图槽；
  先做 eager 验证。

### 优先级

P3；不影响 greedy 生产，但影响真实 chat 温度。

---

## 9. T5：MTP KV ring buffer / 物理回收

### 现状

Windowed-MTP 只改 **读视图**；`mtp_block_table` 和 MTP KV 页池仍按全链分配。

### 目标

把 draft KV 物理存储限制到 `sink + W + d`，回收远距离 draft KV；论文在 1M
省 7.7–11% 总 KV。MTP 只有 1 层、约 4KB/token，64k 全链约 256MB（与 depth
无关；以 pool 实测为准），收益是显存，不是 decode 速度。

### 施工难点

- 当前 `write_bt_mtp` / `mtp_block_table` 以绝对位置映射物理页；
- ring 需要把逻辑位置映射到固定环槽，并同步 K0 写、v2 读表、chain；
- 与部分驻留/快照/前缀复用的交互要逐项审计；
- 首版只做“window 读 + 全量写”已经够用；ring 是显存优化，不要和 T1/T2 混做。

### 门

- 显存账下降；AL 不变；`gpu_paged_view_aliases_selected_pages` 同构门；
- 128k/1M 才明显；64k 收益有限，优先级低于 T1/T2。

---

## 10. T6：Tree / 多候选 MTP

### 现状

MTP 每步 top-1 链；verify 只有 8 行（anchor + 7）。树形需要多分支 hidden，
draft 计算和 verify 行预算都会上升。

### 施工前提

- 先做 T1/T2，确认 8 行预算的边际价值；
- 需要 tree attention mask、分支 KV/页表、接受路径的树遍历；
- 不建议首版做；若做，先离线仿真 AL/成本，不要直接改图。

### 门

- 同 verify 行预算下 AL 提升 > 成本；
- 恒等门；图捕获不炸。

---

## 11. T7：MTP 头量化（W4A16/W4A8）

### 现状

MTP 头 420M 参数，F16 ≈840MB；`MtpPredictor` 用 F16 Linear。
AWQ/GGUF MTP 头装载后统一 F16。

### 目标

省权重显存/带宽；但 MTP 头是草稿质量核心，量化可能降 AL。

### 施工

- 复用 `Linear` 的 W4A16 marlin 路径；MTP 头需单独的 QuantPlan / loader；
- 先对 AWQ MTP 头做离线量化 + AL A/B（不要直接上生产）；
- 恒等门（greedy 输出仍由 target 决定）+ AL 门（≥ full 的 95%）；
- 量级 M；优先级低于 T1/T2。

---

## 12. T8：验证矩阵缺口

| 项 | 说明 | 判据 |
|---|---|---|
| **GGUF MTP window** | `qwen38_27b-3090-iq4-mtp` / `iq4xs-mtp` 未开 window | 同 d3 恒等门 + AL/read |
| **4B MTP window** | 4B 本身净亏；window 可能改变结论 | 不默认开，先测 |
| **VLM + window** | 图像 × 投机（V5）已有；window 未测 | 图像请求内容正确 + AL |
| **TP2 + window** | 两 rank 各自视图表 | 双 rank 恒等门；无 AR 失步 |
| **128k 点** | 131k pool + reserve 下请求级 OOM；147k boot 失败 | 先解决请求级显存，再谈 window |
| **采样域曲线** | temp=1.0/top_p=0.1 未测 | 与 greedy 分开报 |
| **W=6144/8192** | T0 | 同 T0 判据 |

---

## 13. T9：研究/训练线（不在本轮工程）

| 方案 | 结论 | 何时考虑 |
|---|---|---|
| **FOLD**（arXiv 2610.20392） | 外部 block drafter（DFlash/DSpark）专用；MTP 单链不适用 | 若转 DFlash2 长上下文 |
| **DSpark**（arXiv 2607.05147） | 半自回归 drafter + confidence verify；需新 checkpoint/训练 | 若 MTP 头极限不够 |
| **Frozen-KV MTP** | 只读 target KV；SGLang 仅 Gemma4 assistant | Qwen 需重训，暂不 |
| **L-MTP**（arXiv 2505.17505） | 跳跃预测，训练线 | 模型侧立项 |
| **AdaMTP**（arXiv 2608.00434） | 训练期 adaptive horizon | 模型侧立项 |
| **Post-training MTP heads**（arXiv 2610.00888） | 冻结模型后训练 MTP 头 | 有训练资源时 |
| **长上下文专用 MTP 微调** | 成本最高 | 最后手段 |

**纪律：不要在没有训练资源/新 checkpoint 的情况下，把 T9 混进 T1/T2 的代码路径。**

---

## 14. 测量协议

### 14.1 标准曲线

- 单 3090 Ti，greedy（`temperature=0`, `top_p=1.0`），`ignore_eos`，256 tok，c=1。
- MATH：随机数字词池；DOCS：`perf-runs/.../corpus-docs.txt`（真实长文本）。
- 每点 2 轮；64k 至少 1 轮 + 复跑一次。
- 读 `usage.completion_tokens`，不要数 SSE chunk。
- 采服务端 `/debug/metrics` 前后差分：
  - `spec.round / verify / propose / propose.read / propose.step`
  - `spec.m0..m7` 计数器
  - `spec.mtp.window.{rounds,full,visible_tokens}`
- AL = `spec.accepted / spec.rounds`；`tokens/round = AL+1`。
- 分位置接受率：`pos_i = count(m >= i+1)/rounds`。

### 14.2 必跑门

- `--check` + `--print-config --format json`（确认 window 生效）。
- `gpu_spec_identity_gate_27b`：
  - f16 主池：window=0 / 4032；
  - fp8 主池：window=4032；
  - 长 prompt 必须 > sink+window，否则等于没测。
- `cargo test -p owl-shared --lib`（profile/config）
- `cargo test -p owl-engine --lib mtp_window`（窗口算法）
- `cargo test -p server --test config_check`

### 14.3 读数陷阱

- `spec.extend` 是 **prefill** 税，不是 decode；不要混进 `spec.round`。
- `spec.propose.read` 是 host 同步 + GPU 读，window 后应到平台。
- 单次 spec 方差 ±20%；看 2 轮/中位，不要单点调参。
- 外部 session 的 kill 脚本会让“服务消失但日志无错”；先看 `ss`/PID。

---

## 15. 交接清单（下一会话开始前）

- [ ] 读 `packages/owl/AGENTS.md` + 设计文档 + 本文。
- [ ] `git status` 确认大量未提交改动是预期的；**不要 reset**。
- [ ] `nvidia-smi` 查外部会话；优先 GPU2 `42:00.0`；端口避让 8140。
- [ ] 构建 release（`GDN_TRITON_PY`）。
- [x] 跑 T0：W=6144/8192 64k MATH/DOCS（4032 最优）。
- [x] 把 T0 数据写入 `perf-runs/.../REPORT.md` 与设计文档 §4.13。
- [x] T1b v2：两连低窗，DOCS 29.7k 68.91 / 16k 105.23。
- [ ] 若 T0 确认 4032 最优，保持 profile；否则更新 profile 并重跑恒等门。
- [ ] T1：先做静态 ctx 阈值（d7→d3），再考虑 EMA。
- [ ] T3：先做草稿 KV 追赶（`docs/arch/mtp-草稿KV追赶-设计与施工.md`），再把 MTP 纳入 C7。
- [ ] T2：确认 T1 图族后再做置信度截断。
- [ ] 不 commit / 不 push，等用户明确指示。

---

## 16. 已知坑（按踩坑概率排序）

1. **外部 `gguf` 会话 kill 全部 compute apps**：服务消失、日志无错；先查 PID/端口。
2. **`OWL_TEST_DEVICE` 纯数字被拒**：用 BDF `42:00.0`。
3. **27B 贴顶**：不能同时起两个 27B server；boot 需要全空卡。
4. **`full.toml` 完整性门**：`mtp_window/mtp_sink` 必须在 full 里作为 schema 缺省，
   否则 `--check full` 拒；性能值只写具体 MTP profile。
5. **window 读表与写表混用**：`slots` 必须用完整 `bt`；`visible` 只给 v2 读。
6. **`extend_v2=None` 仍传 window lens**：会落 chunked prefill 的绝对长度语义；
   代码已有 full fallback，改路径时不要拆掉。
7. **chain 仍读 `bt_mtp` 叶子**：G1 的 `c{j}_bt` 输入槽是 window 对 chain 生效的关键；
   改图族时不要回退。
8. **profile 重命名**：`spec-dflash`/`awq27b-3090-max` 是旧名；测试/工具已同步到
   `qwen38_27b-3090-w4a16-dflash` / `qwen38_27b-3090-w4a16`。
9. **128k**：当前 131k pool + reserve 下请求级激活 OOM；window 不解决 target verify。
10. **采样域**：greedy 门的结论不能直接套到 temp=1.0；需单独统计门。

---

## 17. 关键文件索引

```text
# 设计/数据
docs/arch/mtp-草稿KV追赶-设计与施工.md
docs/arch/mtp-置信度截断-设计与施工.md
docs/arch/mtp-长上下文抗衰减-设计与施工.md
docs/arch/mtp-长上下文抗衰减-下一阶段施工与交接.md
perf-runs/mtp-longctx-2026-10-21/REPORT.md
perf-runs/mtp-longctx-2026-10-21/{spec,spec-d7,nospec,spec-docs,nospec-docs,window-test,window-test2,window-docs}/

# 代码
tools/mtp_curve.py
crates/engine/src/exec/mtp_window.rs
crates/engine/src/exec/draft.rs
crates/engine/src/exec/spec.rs
crates/engine/src/graph_fleet.rs
crates/engine/src/spec_runtime.rs
crates/engine/src/scheduler.rs
crates/engine/src/tests.rs
crates/shared/src/config/mod.rs
crates/models/src/layers/mtp.rs
crates/models/src/layers/attention.rs

# 配置
profiles/full.toml
profiles/qwen38_27b-3090-w4a16-mtp.toml
profiles/qwen38_27b-3090-iq4-mtp.toml
profiles/qwen38_27b-3090-iq4xs-mtp.toml
```

---

## 18. 给下一会话的开场白

```text
工作树 = packages/owl-worktrees/mtp-spec（feat/mtp-spec，未提交）。
先读 docs/arch/mtp-长上下文抗衰减-设计与施工.md 与
docs/arch/mtp-长上下文抗衰减-下一阶段施工与交接.md。

Phase 0 Windowed-MTP + T1a/T1b v2 已落地；生产 MTP 档仍静态 d3，
mtp_window=4032/sink=64/min_ctx=12288；adaptive 默认关。
下一步：补采样/TP2；T0b W=8192 已补（4032 仍最优）。
恒等门 d3/d7 已绿（OWL_GATE_SPEC_DEPTH=3|7）。

注意：外部 gguf worktree 会 kill 全部 GPU 进程（已三次）；用 GPU2 42:00.0，
端口避让 8140；不要 reset 工作树；不要 commit/push。
```

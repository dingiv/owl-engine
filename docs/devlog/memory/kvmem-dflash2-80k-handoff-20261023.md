# 交接：KVMem × DFlash2 同时使用 / 80k 池选择轨迹（2026-10-23）

> 目的：下一个会话继续定位 **80,480 tok 池下 KVMem 选择轨迹分叉**。
> 工作树：`packages/owl-worktrees/metrics-optimize`
> 基线提交：`6ba666b`（未提交任何新改动）
> GPU：RTX 3090 Ti 单卡，BDF `41:00.0`，测试端口 **8141**。

---

## 1. 一句话结论

**视图化 verify 已落地并可用**；90k/131k/260k 不崩、AL 正常。
**80k 池的问题不是 view/verify/页 pin/单页内容**，而是：

> KVMem 的 **选择轨迹是路径依赖的**：
> 池布局 → 下沉/懒沉时机 → 某边界 view → target hidden/query →
> Mean-K 分数 → top-k → 下一边界 view。
> 80k 与 76k 是两条合法但不同的近似轨迹；80k 恰好选出一组让随机词
> prompt 退化的块。

因此下一刀在 **选择层（query/index/top-k 确定性）**，不在 verify。

---

## 2. 设计/研究文档

```text
docs/devlog/memory/kvmem-dflash2-crosscheck-20261023.md
docs/devlog/memory/kvmem-dflash2-view-verify-design.md   # §9 是最终定位
```

`view-verify-design.md` 已含：

- §2–3 机制与接口；
- §9.1 已落地件；
- §9.2 实测（AL/verify/60k dense 对照）；
- §9.3 80k 问题的逐层收窄（含 top8 分数、bhash、页面哈希）；
- §9.4 验收门现状（未达生产）。

---

## 3. 已落地代码（全部未提交）

### 3.1 视图化 verify

| 文件 | 内容 |
|---|---|
| `crates/engine/src/exec/coverage.rs` | `SelectViewAt`、`view_pages_at`、`select_view_at`、`select_view_step` 复用；驻留/哈希/top8 探针 |
| `crates/engine/src/exec/spec.rs` | `VerifyViewTables`、`verify_view_tables`、verify_round 接线、`spec.verify.view` 计数、`write_bt` 视图跳过、`xqa_pt_of` 截断、FI 表按臂构建 |
| `crates/engine/src/select.rs` | `SelectView.blocks: Vec<ChainHash>` |
| `crates/shared/src/config/mod.rs` | `select_partial + spec` 只允许 `verify_arm=v2`、`decode_arm=v2`；单测更新 |
| `crates/engine/src/engine.rs` | `boot_kvmem_check` 纵深臂门 |
| `profiles/qwen38_27b-3090-w4a16-dflash-kvmem.toml` | 27B AWQ + DFlash2 d7 + KVMem overcommit（8141） |

### 3.2 verify 视图语义（重要）

不是逐行不同页表；最终实现是 **同一张页表（最后一行，尾部最长）+ 逐行 `kv_lens`**：

```text
last = pos + rows - 1
pages = selected_pages ++ chain[recall/page ..= last/page]
for r:
  len_r  = selected_tokens + (pos+r+1 - recall_pos)
  slot_r = chain[(pos+r)/page] * page + (pos+r)%page
bt8[r] = pages（零填充到 nb）
kv_lens[r] = len_r
kv_slots[r] = slot_r
pos[r] = logical pos
```

- `kv_slots` 仍写会话链物理页；K/V 写入位不变；
- K 是原位置 post-RoPE，不做 re-RoPE；
- 所有视图 key 都在 query 之前，因果掩码全可见。

### 3.3 关键防坑

- **verify 图恒声明 FI 四件套槽**；`fi_slots_on` 必须再与 `verify_arm.uses_fi()` 取与，否则 v2 档会按会话链铺 FI 表，overcommit 时 `idx` 越界 panic。
- **`xqa_pt_of` 必须按池页数截断**；overcommit 链长 > 池页数，否则输入槽装填越界。
- **`write_bt` 在视图态跳过**（且 `session` 图有 `sel_pt` 时）；否则 spec 轮 `grew`/DFlash2 propose 写全量 bt 撞宿主占位符守卫。
- **`carrier_of` 返回的是 page id，不是 flat slot**；驻留校验必须 `slot == view_page`，不要 `slot/page`。

### 3.4 前一轮遗留（同一工作树，未提交）

- `crates/engine/src/state.rs`：超卖档允许 auto pool `max_seq > cap`。
- `crates/store/src/rel.rs`：`host_used` 含腿自报 `Host KvBlock`。
- `docs/devlog/perf/vram-map.md`：上述记录。

---

## 4. 测试状态

```text
cargo test -p owl-engine --lib   → 132 passed
cargo test -p owl-shared --lib   → 122 passed
```

新增：

```text
view_pages_at_selected_plus_tail
view_pages_at_page_zero_is_treated_as_one
select_partial_spec_requires_v2
```

构建：

```bash
GDN_TRITON_PY=/home/div/Documents/codes/packages/vllm-test/.venv/bin/python \
  cargo build --release -p server --features perf,cupti
```

GPU 当前：15 MiB，已归零。

---

## 5. 实测数据

### 5.1 视图化 verify 收益

| ctx | 路径 | AL | spec 轮 | verify | 备注 |
|---:|---|---:|---:|---:|---|
| 65k | dense（无洞） | 0.75 | 346ms | 327ms | v2 扫全 65k，退化 |
| 90k | 视图 | 2.70 | 51ms | 36ms | `spec.verify.view=10` |
| 131k | 视图 | 2.00 | 48ms | 36ms | 零 IMA |
| 260k | 视图 | 3.00 | 48ms | 36ms | 零 IMA |

视图 verify 比 dense v2 **快 ~9×**。

### 5.2 80k 输出退化矩阵

同一 90k no-thinking prompt：

| 配置 | 池 | 输出 |
|---|---:|---|
| spec off + 视图 | 76,640 | 连贯 |
| spec on + dumb + 视图 | 23,264 | 连贯 |
| spec on + DFlash2 + 视图 | 74,464 | 连贯 |
| spec off + 视图 | **80,480** | **`Yes / user / assistant`** |
| spec on + dumb + 视图 | **80,480** | **退化** |
| spec on + DFlash2 + 视图 | **80,480** | **退化** |

### 5.3 定位证据

- `SelectView.blocks` + `carrier_of`：全部 `Device(slot)`，`slot == view_page`，**没有页失守**。
- 同一选中块 idx2：
  - 80k 页 2448，16 层 K/V 全量哈希 `fe6653c2c199f040`
  - 76k 页 2336，同哈希 `fe6653c2c199f040`
  - ⇒ 单页内容一致，不是 page 腐败。
- 选中块全列表哈希不同：
  - 80k `bhash=4cc648da4e019847`
  - 76k `bhash=3c57d56185382fa0`
  - first/last 相同，中间块不同。
- 末边界 top8 分数（同一 block 也不同）：

```text
80k: 116843...:0.188614994  124373...:0.187117472  241168...:0.185506120 ...
76k: 124373...:0.194004610  116843...:0.193152875  241168...:0.189139202 ...
```

⇒ 不是 tie-break 噪声，是 **query/index 真的变了**。

---

## 6. 已加诊断开关/日志（`probes.dflash_probe=true`）

日志 tag：`[view-diag]`

- `set_view ctx=... pages=... tokens=... blocks=... bhash=... first/last=...`
- `view-page-hash idx=2 page=... hash=...`（16 层 K/V 全量）
- `decode pos=... len=... pages=... first/last=...`
- `verify pos=... rows=... recall=... selected=... view_tokens=... pages=... len0=... slot0=...`
- `top8 ctx=... block:score ...`（**目前只在末边界**）
- 视图块/页长度、`carrier_of` 失守、页不在设备等错误日志。

**注意**：`top8` 目前只在 `last_boundary` 打印；下一步要改成每个边界。

---

## 7. 复现命令

### 7.1 80k 档（退化）

```bash
OUT=perf-runs/metrics-optimize/kvmem-dflash2-verify/repro-80k
./target/release/server \
  -p profiles/qwen38_27b-3090-w4a16-dflash-kvmem.toml \
  --set spec.depth=0 \
  --set pool.vram_reserve_mb=2950 \
  --set probes.dflash_probe=true \
  --log-level info > $OUT/server.log 2>&1 &
```

### 7.2 76k 档（连贯）

```bash
--set pool.vram_reserve_mb=3072
```

### 7.3 固定 prompt

```bash
python3 perf-runs/metrics-optimize/kvmem-dflash2-verify/fixed_prompt.py \
  http://127.0.0.1:8141 90000 /tmp/out.txt
```

固定 prompt 生成：`random.Random('fixed-kvmem-dflash2-20261023')`，
90000 个词，`chat_template_kwargs={"enable_thinking": false}`，
`max_tokens≈8–32`，`temperature=0`。

### 7.4 关键日志

```bash
grep "set_view ctx=89984"  $OUT/server.log
grep "top8 ctx=89984"      $OUT/server.log
grep "view-page-hash"      $OUT/server.log
grep -E "视图页失守|视图页不在设备" $OUT/server.log
```

---

## 8. 下一步调研计划（按序）

### P0：找第一次分叉的边界

- 把 `top8` 从 `last_boundary` 改成**每个边界**打印（或只在 `recall_now`）。
- 同 prompt 跑 80k/76k，逐边界 diff `top8`。
- 找到第一个 `ctx` 使同一 block 分数不同。

### P1：分叉点三选一

在第一个分叉边界比较：

1. **query 向量**（`state.query_sum` / Q 面）是否逐位相同；
2. **同一 block 的 Mean-K 索引字节**是否逐位相同；
3. **上一轮 view**（`SelectView.pages/blocks`）是否相同。

判决：

- query 不同 ⇒ 追上一轮 view / target prefill hidden；
- 索引不同 ⇒ 查 `collect_index_for_block` 在 80k 下沉/回灌边界读到的 K 是否来自正确页；
- query/index 都相同 ⇒ 查 `score_blocks` 的 FP 聚合与 `select` 排序（随机词分数接近，排序对 1 ulp 敏感）。

### P2：修复方向（二选一）

**A. 让选择轨迹与池布局解耦（推荐）**

- 固定选择调度（例如固定每 N 个边界强制完整召回，不受 `lazy_keep`/池余量影响）；
- 或把 query/index 做成池无关（例如用固定窗口/固定 view 的 hidden 计算打分）；
- 目标：同 prompt、同索引下 80k/76k `bhash` 相同。

**B. 接受“不同池 = 不同轨迹”，固定生产池档**

- 先固定 74k/76k 档（已验证连贯），80k 不作为生产档；
- 仍要加确定性门：同池两次跑 `bhash` 必须相同；
- 文档明确“池容量改变选择轨迹”。

### P3：验收门

- 同 prompt 同池两次跑：`bhash` 相同；
- 跨池（若索引相同）：`bhash` 相同；
- 80k spec off/on 输出与 76k 一致或至少连贯；
- 90k/131k/260k 零 IMA，AL > 0；
- 输出一致性/召回不退化；
- `spec.verify.view` 计数 > 0；
- 再决定是否允许 `select_partial + spec` 生产默认。

---

## 9. 重要风险/坑

1. **不要 `pkill -f` 带 server 模式的命令**：会匹配当前 shell 自杀；用 PID kill。
2. 引擎线程 panic 后进程有时不退出；`kill -TERM` 后必要时 `kill -KILL`，确认 GPU 归零。
3. `select_partial + spec` 只允许 `verify_arm=v2`、`decode_arm=v2`；FI/XQA 会被配置门拒。
4. `fi_slots_on` 不能只看槽在场；v2 图也声明 FI 槽。
5. `xqa_pt` 长度必须 ≤ 池页数。
6. 视图态 `write_bt` 必须跳过，否则撞占位符。
7. `carrier_of` 返回 page id；不要当 flat slot。
8. 80k 档不要用 `vram_reserve_mb=512` 做“正确性”结论；它是已知退化轨迹。
9. 随机词 prompt 的分数接近，top-k 对微小差异敏感；质量门最好用真实任务 prompt 复核。
10. `snap_host_mb=0` 会刷“快照下沉失败”WARN；profile 已设 1024。

---

## 10. 原始数据目录

```text
perf-runs/metrics-optimize/kvmem-dflash2-verify/
├─ run-90k/                      # 90k spec on，AL 2.70，视图 verify
├─ run-131k-d/                   # 131k spec on，AL 2.00
├─ run-260k/                     # 260k spec on，AL 3.00
├─ base-65k/                     # 65k dense v2 对照
├─ run-90k-specoff-small/        # 90k spec off + 76k 池，连贯
├─ run-90k-spec-76k/             # 90k spec on + 74k 池，连贯
├─ bhash-80k/ bhash-76k/         # 选中集合 bhash 对比
├─ hashall-80k/ hashall-76k/     # 单页 16 层 K/V 哈希
├─ top8-80k/ top8-76k/           # 末边界 top8 分数
└─ fixed_prompt.py               # 固定 prompt 生成器
```

---

## 11. 交接人一句话

先把 **每个边界的 top8** 打出来，找 80k/76k 第一次分数分叉的 ctx；
然后只查那一个边界的 **query / Mean-K 索引 / 上一轮 view**。
**不要再去查 verify 页表或页 pin——已经排除。**

---

## 12. 续查结论（2026-10-23，同一工作树）

### 12.1 第一次分叉不在“选择”，在 prefill 走视图的时机

把 `top8`/`set_view` 改成**每个召回边界**打印后，80k 与 76k 的逐边界对照：

```text
ctx=73728  bhash 相同 · qfnv 相同 · top8 8/8 相同
ctx=74240  bhash 相同 · qfnv 相同 · top8 8/8 相同
ctx=74752  bhash 不同 · qfnv 不同 · top8 1/8 相同   ← 第一次分叉
```

- 74240 之前：池未满，80k 与 76k 都走 dense prefill，选择完全一致。
- 74240：76k 池先满，开始 `select_partial` 下沉（日志 `下沉 2250 块`）；
  80k 仍有余量，`lazy_keep` 保留整链。
- 74752 起：76k 的 prefill 读“选中块 + 尾部”的视图；80k 仍读全链
  ⇒ **target hidden / query 从这一步开始不同**，后续 top8/bhash 分叉是结果，
  不是选择器在同一输入上算错。

### 12.2 直接按上游 KVMem 修打分：不够

对照 `repos/kvmem/kvmem/src/host/mean_key_retrieval.cpp`，
owl 原 `ScoreMode::LogitMean` 是**点积/logit**，上游是**逐 (层, KV 头) 余弦**
（`docs/memory-backend-contract.md`：scores remain raw cosine values）。
已新增 `ScoreMode::CosineMean`（key 模长归一，GQA 组均值）并切为生产快路。

**单独换余弦，80,480 池 + 2240 工作集仍然输出 `Yes/user/assistant`**，
说明打分口径不是这次退化的主因。

### 12.3 根因：工作集 2240 tok 把 90k 压到 2.2k

| 工作集 | 池 | 输出 | 备注 |
|---:|---:|---|---|
| 2240 | 80,480 | `Yes / user / assistant` | 70 页 = sink 2 + recent 4 + top-k 64 |
| 8192 | 80,480（spec off） | 连贯：识别为 word salad | 256 页 = top-k 250 |
| 8192 | 80,480（DFlash2 d7） | 同上 | AL=3.00；轮 115ms / 裸 28ms |
| 16384 | 80,480（spec off） | 连贯：开始写 story | 512 页 |
| 16384 | 80,480（DFlash2 d7） | 同上 | AL=1.60；轮 194ms / 裸 35ms |

上游产品默认 `B=36864 tok`、`sink=12288`、`recent=1024`
（`repos/kvmem/docs/recommended-config-performance.md`），
owl 的 2240 是 **70 页**，其中 top-k 只有 64 块；90k 上下文下
选择层不是“算错分”，而是**预算太小、被迫丢掉几乎全部历史**。
同一个 80,480 池，只把工作集从 2240 提到 8192 就恢复连贯；
所以修复点是预算，不是 tie-break / 页 pin / verify。

### 12.4 已落地

```text
profiles/qwen38_27b-3090-w4a16-dflash-kvmem.toml
  workspace_per_req = 8192     # 原 2240

crates/store/src/select.rs
  + ScoreMode::CosineMean      # 产品口径；LogitMean 保留作对照
crates/engine/src/exec/coverage.rs
  生产打分切 CosineMean（每边界 top8 + qfnv 取证保留）
```

单测：`cosine_mode_removes_key_norm_bias` 过；`select::tests` 12/12。

### 12.5 仍挂账

1. **8192 档 DFlash2 仍触发 `spec-degrade`**：AL=3.00 < 盈亏线 4.07，
   自动降级裸 decode。16384 档 AL=1.60，更差。若要以 spec 为主，
   需要另立“视图 verify 行数/带宽”刀，不能靠继续加工作集。
2. **query 窗已从 8 改为 64**（`W_QUERY_TOKENS`）。上游是 last-user span
   最多 512（token-only 回退 1024）；64 已把 10k needle 的完整问句纳回窗，
   但 90k 距离下仍答 No（见 12.7，属距离问题）。
3. **长档复测**：131k/260k 在 8192 工作集下尚未跑；页表/verify 行数上限
   未变（视图 256+tail 页），需要一次零 IMA 冒烟。

### 12.6 260k 冒烟（8192 工作集，DFlash2 d7）

```text
prompt=260012 tok / completion=8 tok / finish=length
pool=80480 tok / workspace=8192 tok (256 页)
final set_view ctx=259072: pages=256 tokens=8192
verify pos=260013: rows=8 recall=260000 selected=256 pages=257 len0=8206
```

零 `CUDA_ERROR_ILLEGAL_ADDRESS`、零 `视图页失守/不在设备`、零 panic。
（8 token 太短，未触发 spec 统计；本项只作页表/容量冒烟。）

**未跑**：131k 档位（260k 已覆盖更大视图+尾部；若 131k 有专属形状问题再补）。

### 12.7 关键反证：选择层确实选中了针块，stage-in 内容也没坏

用同一 needle prompt（code 在 token 10,000 = 逻辑块 312）做 40k vs 90k：

```text
40k: block 312 在 final set；target-hash hash=48c9a8f016f32bbf；答案 BLUEBIRD-42
90k: block 312 在 final set（bids 含 312）；同一 target-hash=48c9a8f016f32bbf；答案 No
```

- 90k 的 `bids` 明确含 `312,313,314,315,316,318,319`；
- stage-in 后块 312 的 K/V 页哈希与 40k **逐位一致**；
- 所以 **选择器选中了正确块、页表/搬运也没有丢内容**；90k 的 `No` 是
  **80k 相对距离下的模型检索失败**（或 view 原位置稀疏 attention 的有效距离问题），
  不是“选错块”。45k needle 在 80k 上下文中曾通过，进一步支持“距离”而非“选择”。

⇒ 对 80k 退化的最终归因仍是 **2240 工作集**；不要再把 90k needle 的 `No`
当成选择器 bug。若要继续追，应立“80k 距离 attention/位置”刀，而不是改 top-k。

### 12.8 最终变更清单（本续查）

```text
crates/store/src/select.rs       + ScoreMode::CosineMean（KVMem 产品余弦）
crates/engine/src/exec/coverage.rs  W_QUERY_TOKENS 8→64；每边界 top8/qfnv；bids；
                                    target-hash 诊断
crates/engine/src/running.rs       StepProbes.view_hash_block
crates/shared/src/config/mod.rs    probes.view_hash_block
profiles/...kvmem.toml             workspace_per_req 2240→8192
docs/...                           本节 + design §10 + selection §10
```

单测：engine 132 / shared 122 / store 116 全绿。
GPU 冒烟：80k 8192 连贯；260k 8192 spec on 零 IMA。

### 12.9 needle 位置扫描：有效检索距离 ≈ 45k，不是选择层

固定 90k 上下文、workspace 8192、同一 needle prompt，只移动 code 位置：

| code 位置 | 逻辑块 | 选中? | stage-in 后哈希 | 回答 |
|---:|---:|---|---|---|
| 10k | 312 | 是（312–319） | `48c9a8f016f32bbf` | `No` |
| 30k | 937 | 是（937–945） | `d89ce44a7cda9636` | `No` |
| 45k | 1406 | 是（1406–1412） | `3a1b75c757a99f20` | `BLUEBIRD-…`（16 tok 截断，已答出码） |
| 60k | 1875 | 是（1875–1879） | `3d43be8a9c338d4e` | `…secret code is explicitly stated as:`（16 tok 截断，倾向答出） |

补充：30k 位置 + workspace **16384**（512 块，块 937–945 连续选中，
哈希同上）仍答 `I cannot provide a secret code…`。

⇒ 选块、页表、stage-in 三件全部正确；**失败边界在“query 到 code 的相对距离
≈ 45–60k”**。8192 → 16384 工作集不能恢复，说明不是保留块太少。
下一刀应查 RoPE/YaRN/长距离 attention，而不是继续加 top-k 或改选择器。

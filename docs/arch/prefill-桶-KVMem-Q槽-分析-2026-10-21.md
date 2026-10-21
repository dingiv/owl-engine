> **后续（2026-10-21）**：该分析完成后，prefill 桶已按用户裁决**退役**；
> 备份分支 `backup/prefill-bucket-2026-10-21`。本文保留为 Q 槽阻断的历史证据。
>
# prefill 桶 × KVMem：Q 槽阻断分析（2026-10-21）

> 对象：`perf-review @ main@23116c7`，`prefill_plan.rs` / `exec/prefill_chunk.rs`。
> 结论先行：**KVMem 档的 prefill 桶不是“没配”，是代码里被 `select_decode` 硬旁路；
> 根因是桶图没有输出选择层需要的 Q 窗。** 桶能捕获、能回放，但 KVMem 下 100% 回 eager。

## 1. 一句话结论

- `bucket_prefill_chunk` 在 `select_decode`（KVMem 生产面）为真时直接 `return Ok(None)`；
- 因此 KVMem（`gguf27b-262k` / `1m`）即使显式开 `geobuckets`，也是 **每个 chunk 都 eager**，
  桶图一次不回放；
- 若显式开桶，boot 仍会花 **571 MiB slab + ~2.5 s** 捕获 6 个桶（t32…t1024），
  然后全部闲置 —— **纯显存/启动开销，零收益**；
- 阻断点不是 FI 图，也不是 GDN 表，而是 **Q 窗**：选择层打分需要每层全注意力
  post-RoPE Q 的末 8 行；eager 路用 `ctx.q_sink` 收集，桶图只输出 hidden/taps，
  没有 Q 窗出口。

## 2. 实测证据（2026-10-21，单 3090 Ti 42:00.0）

### 2.1 KVMem 开桶 ⇒ 全部旁路

档：`profiles/legacy/gguf27b-262k.toml` + `graph.prefill_bucket=geobuckets`
+ `dispatch.fi_plan_restore=false` + `dispatch.fi_cuda_graph=true`；8k prompt。

boot 自证：6 桶全部 `Captured`，`prefill桶 571MiB`，引擎 live 21 198 MiB。

请求后 metrics（counter 差分）：

| 指标 | 值 |
|---|---:|
| `prefill.bucket.bypass.diag` | **10** |
| `prefill.bucket.hit` | **0** |
| `prefill.eager.chunks` | **10** |
| `prefill.eager.tokens` | **8 761** |

日志一次 WARN：

```text
诊断档/B 轴选择层在场(.../select_decode)⇒ prefill 桶全部让位 eager
(Q 槽只在 eager 路接线;要图就关它们)
挂账:桶图路学会 Q 槽后,生产档(select_decode)就不必再让位
```

⇒ KVMem 的 prefill 吞吐仍是 eager 口径（本机 8k ≈ **1 623.6 t/s**），
桶的 571 MiB 是死重。

### 2.2 非 KVMem 开桶：桶本身是有效的

同一二进制、同一 27B Q4_K_M、`profiles/legacy/gguf27b.toml`
（`select_decode=false`），8k prompt：

| 臂 | prefill | TTFT | 桶命中 |
|---|---:|---:|---:|
| eager | 1 824.42 t/s | 4 491.81 ms | 0/9 |
| **geobuckets** | **1 841.84 t/s（+0.95%）** | **4 449.36 ms（−42.45 ms）** | **8/9** |

- 8 个整 1024 chunk 全命中 `t1024`；`prefill.graph` 0.436 ms/chunk；
- 第 9 个尾块 569 tok：`pick_bucket→t1024`，pad 455 > `bucket_pad_max_rows(1024)=4`
  ⇒ `bypass.pad=1`，回 eager（这是护栏的正确行为，不是漏配）；
- 该档 `prefill.chunk` 总和 4 505.7 → 4 072.3 ms（−433 ms），
  但 TTFT 只降 42 ms（客户端/调度/尾部同步摊掉一部分）。

⇒ 桶的收益在 GGUF 上比 AWQ 小（AWQ std 五点实测 +1.1…+3.6%），
但方向明确；KVMem 目前拿不到这笔收益。

## 3. 代码路径：为什么 Q 槽是唯一缺口

### 3.1 旁路点

`crates/engine/src/exec/prefill_chunk.rs::bucket_prefill_chunk`：

```rust
if host.knobs.pf_bisect
    || host.knobs.pf_stages.is_some()
    || host.probes.dflash_probe
    || host.knobs.coverage_probe
    || host.knobs.select_decode
{
    mcnt("prefill.bucket.bypass.diag", 1);
    ... warn ...
    return Ok(None);
}
```

KVMem 的 `select_decode=true` 是**硬条件**（选择层/视图/下沉全挂在它上面），
所以这里必然命中旁路。

### 3.2 eager 路的 Q 面

`exec/prefill_chunk.rs` eager 前向：

```rust
let cov_sink = (coverage_probe || select_decode) && paged && boundary
    .then(|| QSink::new(Vec::new()));
...
ctx.q_sink = Some(s);
...
let (hfb, arena1, cov_q) = self.eval_root_and_q(&hidden_full, cov_sink.as_ref()).await?;
self.coverage_take_q(cov_q, base, t, ...).await?;
```

- `attention.rs::tap_q` 只把每层 **post-RoPE Q**（`[t, hq*hd]`）的节点引用塞进
  `ctx.q_sink[fi_kvi]`，不建节点；
- `eval_root_and_q` 用 `eval_ops_multi` 把主根与 Q 节点一次归约（避免双执行
  推进 GDN 状态）；
- `coverage_take_q` 只 D2H **末 `W_QUERY_TOKENS=8` 行**，解码进
  `CoveragePending`；`coverage_analyze` 再用它打分/选择/协调。

⇒ 选择层要的不是完整 Q，而是 **每层末 8 行的 Q 窗**。

### 3.3 桶图没有 Q 出口

`prefill_plan.rs::capture_one`：

- `ForwardCtx` 里 `ctx.q_sink` **从未设置**；
- `outputs` 只有 `root`（hidden/taps 的 concat）；
- `BucketOut` 只有 `root` + `taps`；
- `BucketGraph::replay` 回放后只返回 `root/taps`。

⇒ 桶图跑完，选择层拿不到 Q；`coverage_analyze` 没有输入，只能让位 eager。

## 4. 次要问题（不是本次主阻断，但会咬人）

| # | 问题 | 现状 | 影响 |
|---|---|---|---|
| 1 | **t1024 的 padding 护栏 = 4 行** | `bucket_pad_max_rows(1024)` 落 `_ => 4` | 513…1019 tok 的 chunk 全部 `bypass.pad`；尾块几乎必回 eager。生产 chunk 恒 1024 时只损失尾块；`prefill_tail_merge` 开或调度块长漂移时损失放大 |
| 2 | **阶梯顶 = 1024** | `ladder(Geobuckets)=[32,64,128,256,512,1024]` | `prefill_chunk>1024`（2048/4096 的 prefill 收益档）全部 `pick_bucket=None` ⇒ eager；要抬 chunk 必须先加桶 + slab 预算 + 捕获成本 + allowed_ops 台账 |
| 3 | **GDN host 槽烘焙** | `gdn_host_slot_arm(t_pad≥64)` ⇒ `gdn_slot_baked=Some(0)`；`BucketGraph::usable` 只放槽 0 | `gdn_slots>1`/并发 >1 时只有槽 0 吃桶，其余 eager；KVMem 当前 `gdn_slots=1` 无碍，未来并发档是硬缺口 |
| 4 | **`fi_plan_sync` 每桶回放前刷表** | `fi_plan_restore=false` 是桶的硬配；`fi_plan_sync` 首轮 ≈1.5 ms/chunk（同 ctx 第二轮起缓存） | 小桶/短 chunk 上可能吃掉一部分固定节余；大 chunk 可忽略 |
| 5 | **没有绿硬等价门** | `gpu_prefill_bucket_matches_eager` 在干净 main 上仍红（0.8B ILLEGAL_ADDRESS @ `record_tail`） | 桶路目前只有 27B 真机探针 + std 证据；KVMem 接 Q 槽后必须补 27B 图≡eager 门，不能只靠吞吐 |
| 6 | **开桶 = +571 MiB / +2.5 s boot** | 6 桶 slab 571 MiB（预算 640） | KVMem 24G 卡余量约 2.7G，能放；但若旁路则纯浪费。`graphs=auto` 对 overcommit 直接降级为 off，只有手写档会踩“开了不用” |

## 5. 修法：把 Q 窗做成桶图输出（推荐 P0）

### 5.1 捕获面

1. `capture_one` 里挂 `ctx.q_sink = Some(QSink)`；
2. 每次 `tap_q` 已把每层 Q 节点收进 sink；模型前向结束后，对每个 Q 节点取
   **末 W 行视图**（`W=8`，与 `W_QUERY_TOKENS` 单一来源）；
3. 把 `n_q` 个 `[W, hq*hd]` 视图 concat 成一个输出槽
   `q_window: [n_q*W, hq*hd]`（27B：16 层 × 8 × 24×256 × 2B ≈ **1.5 MiB**，
   6 桶合计 ≈ 9 MiB，可忽略）；
4. `BucketOut` 增加 `q_window`（按层切视图）。

### 5.2 回放面

1. `bucket_prefill_chunk` 从旁路条件里**删掉 `select_decode`**（保留
   `pf_bisect/pf_stages/dflash_probe` 这些“图里没有的诊断面”）；
2. 回放后把 `BucketOut.q_window` 交给一个**窗口版** `coverage_take_q_window`
   （直接按 `[W, hq*hd]` D2H，不再要求完整 `[t, hq*hd]` 块）；
3. 填 `host.cov = Some(CoveragePending{...})`，后续 `coverage_analyze`
   原样跑（打分/选择/视图/协调/下沉都不变）。

### 5.3 为什么不是“图内跑选择层”

- 选择层是 **host 打分 + store 账 + swap 编排**（`coverage_analyze`），
  不是纯 GPU 核；把它塞进图会把 host 逻辑和 I/O 带进捕获窗，违反 G2；
- 只要图把 **Q 窗**交出来，选择层仍在图外、在 chunk 边界跑，语义与 eager 完全同构；
- 这也符合现有 `fi_plan_sync` 的“图外刷表、图内只读”分工。

### 5.4 验收门（必须带）

- 27B / KVMem / 8k+32k：`prefill.bucket.hit > 0`、`bypass.diag = 0`；
- 同 prompt：桶图 vs eager 的 **needle/内容**一致；选择层 `scores/top-k` 与 eager
  在容差内一致（Q 窗同源 ⇒ 应逐位或近逐位）；
- `gpu_prefill_bucket_matches_eager` 转绿，或新增 27B 的图≡eager 门；
- 吞吐：8k prefill 预计 +1% 量级（非 KVMem 实测 +0.95%，AWQ +1.1…3.6%），
  TTFT 8k 约 −40 ms；KVMem 还需复测，不能直接套非 KVMem 数字。

## 6. 建议顺序

1. **P0：Q 窗输出 + `coverage_take_q_window` + 去掉 `select_decode` 旁路** ——
   这是 KVMem 吃桶的唯一前置；改动集中在 `prefill_plan.rs` / `coverage.rs` /
   `prefill_chunk.rs`，不动 FI/GDN 核。
2. **P1：补 27B 图≡eager 门**，再谈默认开桶；当前 0.8B 门红是硬风险。
3. **P2：chunk >1024 的桶阶梯**（t2048/t4096）+ slab 预算 + allowed_ops 台账；
   在这之前不要把 `prefill_chunk` 抬过 1024 并期望桶收益。
4. **P3：GDN host 槽设备化（P-6c）**，解并发多槽的桶可用性；KVMem 单槽档不急。

## 7. 复现

```bash
# KVMem 开桶（会被旁路）
./target/release/server -p profiles/legacy/gguf27b-262k.toml \
  --set model.dir=.../Qwen3.8-27B-Q4_K_M.gguf \
  --set runtime.bind=127.0.0.1:8142 --set runtime.device.bdf=42:00.0 \
  --set graph.prefill_bucket=geobuckets \
  --set dispatch.fi_plan_restore=false --set dispatch.fi_cuda_graph=true

# 请求后看
curl -s 127.0.0.1:8142/debug/metrics/prefill | jq '.rows[] | select(.tag|startswith("prefill.bucket") or startswith("prefill.eager"))'
```

原始记录：`/tmp/owl-bucket-kvmem.log`、`/tmp/owl-bucket-8k.json`、
`/tmp/owl-bucket-eager-8k.json`、`/tmp/owl-bucket-geo-8k.json`。

# 单图追加 TTFT：1.3k image token 的账（2026-11-13）

> 场景：agent 已有 ~30k 文本前缀；单轮新增 **1 张 1536×864 PNG**
> （Qwen 视觉网格 ≈ **1296 image token**）+ 39 tok 文本，`max_tokens=8`。
> 图用新内容（红/蓝/绿边框各一张）确保 **不是旧图缓存**。
>
> worktree：`packages/owl-worktrees/session-kv-snap`
> 分支：`feat/session-kv-snap`（未提交）

## 0. 结论先行

- 新图首轮：**TTFT 1430ms / prefill 1310ms / 1335 tok**；把 DFlash2 草稿
  encode 挪到首 token 后（§2.5）后 **TTFT 1389ms**（`dflash.drain` 67ms）。
- 其中 **vision tower ≈285ms**（每张**新图**一次；同图后续轮 tower=0）。
- 剩余 ~1.0s 是 1335-token prefill：**GPU ≈1.0s**（int8 attention 297ms、
  GDN 145ms、vision flash/rope/GEMM 等），**真实 host CPU ≈275ms**
  （`srv.op.*` 267ms，其中 GDN chunked 170ms；`srv.seg.*` 8ms）。
  `srv.wait.htod` 461ms **不是 host**：Htod 在 COMPUTE 流上排队，ack 等
  前面的 GPU 核；`srv.seg.htod`（真 host 发射）只有 **6.7ms**。
- **不是“页对齐切碎”问题**：把 1312+26 合并成 1344 后，新图 TTFT
  1.45s → 1.45s（无收益）；反而丢失下轮 image KV 复用（下轮从
  1367 tok 变 2675 tok）。所以**不要**为单图关 `prefill_align_tail`。
- 真正的下一刀与文本同一处：**int8 小 chunk prefill 图**（host 侧）
  + **attention split-K/Br 调大**（GPU 侧）。tower 285ms 是第二刀。

## 1. 实测（默认 3080 对档；30k 文本 base 先 cold）

| 轮次 | client TTFT | server prefill | cache | 备注 |
|---|---:|---:|---:|---|
| 新图首轮 | **1430ms** | **1310ms / 1335 tok** | 30528/31863 | tower 285ms |
| 同图第二轮（图/前缀已缓存） | 784ms | 733ms / 1349 tok | 30528/31877 | 仍走 1.3k prefill；tower=0 |
| 同图第三轮（prompt 已有 2 图） | 1347ms | 1247ms / 2675 tok | 30528/33203 | 第二张图重复内容，tower=0 |

注：第二轮 `cached=30528` 只命中文本前缀；**图片 KV 本身要再算**，所以
不是“同图就免费”。只有同一会话的图片块进了前缀快照，下一轮才可能只算
新增文本。

## 2. 成本拆解

### 2.1 vision tower（每张新图一次）

```text
vlm-cost: images=1 新图=1 | host=12.6ms tower=285ms combine=1.9ms | prompt=31863
```

- tower 内部（CUPTI，单图 1296 token）：
  - `flash_fwd_kernel` ≈157ms
  - vision GEMM（cutlass）≈110ms
  - `owl_rope_half_partial_f16` ≈140ms（108 发，1 block/token —— 见 §4）
- tower 在 TP2 两 rank **并行**，wall ≈285ms；不是 ×2。
- 同图后续轮 `新图=0 tower=0.0ms`（VLM 内容 LRU），这是既有缓存。

### 2.2 1.3k-token prefill（每轮都付）

- `prefill.chunk` 1310ms（两块 ×2 rank；页对齐 1312 + 26）。
- CUPTI `owl_int8_prefill_paged`：64 发 ≈297ms（两 rank 之和；
  wall ≈150ms 量级）。
- **真实 host CPU（prefill 窗口）**：`srv.op.*` 合计 **267ms**
  （GDN chunked 170ms、int8 attention 22ms、cast/silu/norm 等 ~75ms）；
  `srv.seg.*` 合计 **8ms**（其中 `srv.seg.htod` 仅 **6.7ms**）。
- `srv.wait.htod` 461ms 是**客户端等 ack**；Htod 落在 COMPUTE 流，ack
  必须等前面的 GPU 核跑完 ⇒ 它量的是 **GPU 排队**，不是 host。
- 佐证：块池把 `interp.host.node.gemm` 从 **280ms → 2.3ms**，TTFT
  1448 → 1440ms **不变**；DFlash 逐层 Htod 去重 50 → 34 个节点，
  `srv.wait.htod` 仍 ~450ms、TTFT 不变。
- 结论：**图场景是 GPU-bound**（GPU ~1.0s / host ~0.275s）；文本 82 tok
  才是 host-bound。图越大，host 占比越低。

### 2.3 已砍：草稿 encode 移出首 token 路径

- DFlash2 的 `prefill_dflash_encode` 只服务**首 token 之后**的 spec propose，
  不参与首 token 本身；旧实现把它放在 prefill chunk 内。
- 改成：chunk 内只把 tap 根块挂到 `RankDev.pending_dflash`（cap 4096 tok），
  `execute_prefill` 在 `sample_and_emit` **之后** drain；根块在
  `drain_turn_stale` 的 keep 集里，跨 chunk 不被当残块清掉；turn 中断走
  `discard_pending_dflash`。
- 实测：图场景 `dflash.drain` 67ms（两 rank 合计）/ TTFT **1430 → 1389ms**；
  文本 82 tok 无回归（150.9 vs 144ms，噪声内）。收益不大是因为 **1.3k 图的
  encode 本身只有 ~33ms/rank**；264ms 的 p50 来自冷 30k 的大 chunk。
- 这不是图场景的主刀；主刀仍是 §4 P0 的 GPU attention。

### 2.4 页对齐 26-token 尾巴

- 当前 `prefill_align_tail` 在 31840 切一刀，尾 26 tok 另起一块；
  这块仍有 ~300ms 量级固定税。
- 但把它并成 1344 一块后，新图首轮 **1.45s → 1.45s**（没快），
  且下轮 cache 从 31840 掉到 30528，下轮多算 ~1.3k。
- ⇒ 尾巴不是主要矛盾；**保留页对齐快照**，不要为单图关它。

## 3. 已试/已排除

| 刀 | 结果 |
|---|---|
| `prefill_align_tail=false` + `snap_msg_min_span=2048` | 新图 1.45s（无收益）；下轮 1.35s / 2675 tok（缓存丢失，负收益） |
| `kv_dtype=fp8`（FI prefill） | 新图 1.38s、tower 0 时 752ms；与 int8 持平，不换默认 |
| `block_pool_bytes=256MiB`（图场景） | `interp.host.node.gemm` 280→2.3ms，但 prefill 1300→1270ms、TTFT 1448→1440ms：**GEMM host 不在关键路径** |
| DFlash 逐层 Htod 去重（50→34 节点） | `srv.wait.htod`/TTFT 不变：**Htod 数量不是关键路径** |
| `int8_prefill_sync=false` | 文本 append 无变化；图未单独 A/B，风险不值 |
| 同图重复 | tower 0，但图片 KV 仍要按位置重算；不是免费 |

## 4. 下一刀（按收益）

### P0 attention split-K / Br=128（GPU ~300ms → ~150ms；图场景首选）

图场景是 GPU-bound：L=30k、T=1335，Br=64 ⇒ 21 q blocks × Hq，每 q block
重复读 30k KV。KV 维切 4 片 + partial/reduce，或 Br=128（8 warps/CTA），
把 KV 读次数减半；预计 attention 300 → 150ms、TTFT 1.43 → 1.25s。
参考 `cu/attn/prefill_split_f16.cu` 与 `owl_int8_decode_*_split`。

**文本 82 tok 的 host 图**仍是另一条线（那里 host-bound）；不要把它当
图场景的主刀。

### P1 int8 小 chunk prefill 图（只对文本 host-bound 场景）

文本 82 tok 的 195/105ms 里 host 发射是主项；图能把那一段压掉。
但**图场景 1.3k token 是 GPU-bound**，图不解决 attention；优先级低于 P0。

**int8 图阻断（已定位）**：`owl_int8_prefill_paged` 的 `L`/`base_pos`
是发射标量，捕获会烘死；append 的 base 每轮变。需要：
1. 核加 dyn 变体（`const int* L_ptr, const int* base_ptr`，核首读）；
2. 恢复 dump 页 + `t_pad/t_real` 分离（老 `prefill_plan.rs` 有现成骨架，
   但它只服务 FI，不能直接套 int8）；
3. 输入槽：ids/pos/kv_slots/kv_lens/gdn_slot/bt + 两个 dyn 标量。

### P2 vision tower rope（140ms → ~20ms）

`owl_rope_half_partial_f16` 目前 **1 block/token**，块内串行 heads
（27B 视觉 16 heads）。加一个 `grid=(tokens, heads)` 的 heads 并行变体，
只给 `layers/vision.rs` 用（文本 rope 不动），预计 tower 285 → ~170ms。
这是独立小刀，可与 P0/P1 并行。

### P3 **不要**做 Htod 合批（461ms 是 GPU 排队，不是 host）

`interp.host.node.unk.htod` 的 449ms 看起来像逐条 Htod ack，但
`srv.seg.htod`（真 host 发射）只有 **6.7ms**；Htod 在 COMPUTE 流上，
ack 必须等前面的 GPU 核。把 6 张表合成一条 `HtodChunks` 只会减少
命令数，不会缩短 TTFT。DFlash 逐层 Htod 去重（50→34 节点）已实测
`wait.htod`/TTFT 不变，就是反证。

## 5. 复现

```bash
# 生成 1536×864（≈1296 token）新图
python3 - <<'PY'
# 见 /tmp/owl-image-green.b64（红/蓝/绿边框各一张，内容不同）
PY

# 默认档 server 后：
python3 /tmp/ttft_image_append.py /tmp/owl-text-36k.json \
  /tmp/owl-image-green.b64 finalimg 1
```

当前运行态：8135 默认档，`/tmp/owl-8135-imgfinal.log`；3090 server 未动。

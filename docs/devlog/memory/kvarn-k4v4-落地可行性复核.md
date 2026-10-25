# KVarN k4v4 落地可行性复核

> 2026-10-25。基线 `fix/spec-tailcap@0a62b747`；工作树
> `<worktree>/kvarn-k4v4`，分支 `feat/kvarn-k4v4`。
> 本文回答一个问题：**KVarN 能不能在 owl 里落地，值不值得按 k4v4/g128 落地。**
>
> 结论先行：
>
> 1. **技术上可落地，但不是“加一个 KV dtype”。** 它是一条独立的
>    page128 / 单 record / 带 f16 sink+tail 的 KV 后端，改动面覆盖
>    pool、store、scheduler、attention、graph、spec、config。
> 2. **k4v4 在当前单 3090 档上到不了 262k。** 按现役 3090 档的预算
>    （vision + DFlash2 + 1 快照）估算，k4v4 约 **160–180k**；把 vision、
>    快照、投机全关也只到 **~235k**。要 262k，应该用 **k4v2**（约
>    211–238k，关 vision 后 ~306k），而不是 k4v4。
> 3. **3080 TP2 已经用 FP8 跑到 262k。** KVarN 在那里的价值不是“更长”，
>    而是“同长度下更多并发”：k4v4 约可容 2 条 262k，k4v2 约 3 条。
> 4. **没有 KVarN verify，就不能开 DFlash2。** 首版 spec-off 意味着
>    3090 agent 档从 175 t/s 级掉到 spec-off 档；KVarN 不能直接替换现役
>    生产档，只能作为独立长上下文档。
> 5. 建议先做 **Phase 0 单 tile/单层 spike**，不要直接改引擎；在
>    Phase 0/1 用 **k4v2 与 k4v4 两档对拍**，用容量与质量同时裁决。

---

## 1. 参考实现的真实形态（先纠正几个容易低估的点）

参考实现 = 外部存档 `kvarn/files/vllm/...`（vLLM 0.30 port）。

### 1.1 KVarN 的存储不是“4-bit 元素数组”

一个 `(block=128, kv_head)` 是 **一条 35,072 B record**（hd256/k4v4/g128）：

```
K packed 16,384 | K s_col 512 | K zp 512 | K s_row 256
V packed 16,384 | V s_col 512 | V s_row 256 | V zp 256
= 35,072 B = 128 × 274 B
```

27B/TP1（16 full layers × 4 kv heads）：**17,536 B/token**，是 FP8 的
**1.87×** 容量；k4v2 是 13,440 B/token，**2.44×**。这与上游
`long-context.md` 的 k4v2 数字一致。

关键含义：

- scale/zero 是 **tile 共享**，所以 **tile 必须等于 block**；
- 一个 `(block,head)` 的 K/V 必须放在同一条 record 里，不能再拆成
  现有 `KvBlocks { k_cache, v_cache }` 两个平面；
- `KvQuant::size_bytes()` 这种“每元素字节”抽象对 KVarN 不成立，必须
  新增“每 token/每 record 字节”的布局抽象。

### 1.2 sink 和 tail 不是可选项

上游默认 `sink_tokens = 128`，并且实际实现是：

- 每个 request 的 **block 0 常驻 fp16 pool**，不量化（attention sink）；
- 当前未满 128 的 **tail** 常驻 fp16 pool；
- 满 128 的 block 由 builder 在两次 forward 之间 **host 侧 flush** 成 int4；
- request 结束后 sink 先 retired（继续 fp16 常驻），pool 压力大时才
  flush 成 int4；部分 tail 直接丢弃（不可前缀复用）。

`KVarNMetadataBuilder.build()` 的顺序是：

```
mark sinks → flush 已满 blocks（释放槽）→ allocate 新 tails
```

这样 live 槽峰值约为 `2×并发`（sink + tail）。上游 `KVarNConfig` 的
`pool_slots()` 还按 `2*max_num_seqs + prefill_blocks + 8` 预留 prefill
期间的 block 数。

**对 owl 的结论：**

- 施工文档里“S1：所有满 128 tile 都进 int4，不做 sink”是**偏离上游**
  的方案；sink 的代价只是每会话每层 0.5 MiB（27B/TP1 约 8 MiB/会话），
  不是主要成本。首版应默认 **sink + tail 都保留**，把“去 sink”作为
  质量对照臂，而不是反过来。
- tail 不能是“每 turn 的 scratch”：owl 的会话跨 turn，若 turn 末的
  未满 128 tail 被丢弃，下一 turn 就丢最后 <128 个 token 的 KV。tail
  必须 **session 级持久**，或者显式落成 f16 副本。
- 未满 128 的 tail **无法用 packed record 表达**，因此它所在页不能被
  store 下沉/驱逐；要么 pin 在设备，要么单独做 f16 宿主副本。这是
  store/eviction 的新约束。

### 1.3 store 的 RTN/pack 上游是 PyTorch，不是 Triton

上游 port 里：

- `kvarn_sinkhorn_triton` 是 Triton；
- `kvarn_store_tile_k_batch_from_sinkhorn` /
  `kvarn_store_tile_v_batch_from_sinkhorn` 是 **PyTorch**
  （`amin/amax/round/位运算/pack`）；
- record 组装是 `torch.cat` + `index_copy`；
- `_kvarn_scatter_store_kernel` 是 Triton（把已旋转的 K/V 写进 tail pool）。

port 中 **没有** `triton_kvarn_store.py`（`kvarn_store.py` 头注提到的
Triton store 只存在于原始 KVarN，不在这个 port 的 `files/` 里）。

**对 owl 的结论：** 不能把“AOT 编译上游 store”当作 Phase 1。必须新写
**RTN + 4-bit pack + record 字段写** 的 GPU 核（Triton 或 native CUDA），
或者写一个“Sinkhorn 后处理”核。上游 PyTorch 只能作数值参考。

### 1.4 prefill 是 materialize + FlashAttention

上游 prefill：

- 首个 chunk（无 cached history）：直接用当前 K/V 做 FA2；
- 后续 chunk / verify：`_kvarn_build_packed_kv_kernel` 把 int4 + tail
  **materialize 成连续 f16 [total_tokens, Hk, D]**，再跑
  `flash_attn_varlen_func(causal=True)`；
- 这就是“O(context)/chunk”的来源，上游用 2048-token 的 chunk 把它
  摊薄；`KVARN_FA_SCRATCH_CAP` 默认 262,144 token。

owl 的现状：

- prefill chunk 默认 **512**；若直接套 materialize，262k 要 512 个 chunk，
  总 materialize 量约 67M token/层，代价不可接受；
- 但 owl **已经有 vendored FA2 varlen**（`cu/flash_attn/owl_fa2.cu`），
  当前只支持 hd64/72、非因果、无 GQA（视觉塔）。vendor 核体本身支持
  hdim256；wrapper 要补 **hd256 + GQA（`h_k`/`h_h_k_ratio`）+
  `is_causal` + 独立 `seqlen_k`**。这是 host 胶水 + 实例化工作，不是
  重写 attention。
- 因此首版可以走 **materialize + FA2 causal**，但有两个代价：
  1. **chunk 冲突**：owl 的 `prefill_chunk=512` 是 Marlin W4A16 的调优值
     （`full.toml` 明确警告 >1024 会拆 M、每段重读权重）；materialize
     需要更大 chunk 摊薄 O(ctx)/chunk。不能简单全局调到 2048；长期要
     packed prefill，或把 attention materialize 拆成独立子块。
  2. **scratch 内存**：262k 时约 1 GiB f16 scratch；3090 现役档没有这个
     余量，3080 TP2 每 rank 约 0.5 GiB 相对宽裕。3090 要 262k 必须
     packed prefill，而不是 materialize。

### 1.5 decode / verify 是两套核

- decode：`_kvarn_fused_decode_kernel` / `stage1+stage2`，in-kernel
  dequant + online softmax，读 int4 + tail pool；这是 KVarN 的速度核心。
- verify：`_kvarn_fused_verify_stage1`（uniform QLEN）或 materialize 路径；
  与 decode 不是同一个核。
- 上游的单用户 KVarN + MTP 在 112k 是 FP8 的 **2.13×** step time；
  batch 100k 约 1.22×。spec 的接受率还会掉约 7%。

**对 owl 的结论：** “首版 spec-off”是合理的，但必须明确：**KVarN 首版
不能用于现役 DFlash2 生产档**。要恢复 DFlash2，需要单独移植 verify
（或先用 materialize+FA2 做正确性臂，但长 ctx 会慢）。这不是 P0 那种
一天能收的尾。

---

## 2. owl 现状与硬冲突

| 面 | 现状 | KVarN 需要 | 冲突级别 |
|---|---|---|---|
| 页宽 | `kv_paged_policy(F16)` 恒 32；`PoolGeom.page` 来自 dtype | page = group = 128 | **硬冲突**，所有 32 页假设要重核 |
| 量化类型 | `KvQuant` = None/Fp8，`size_bytes()` 每元素 | 每 (block,head) record，非元素 | **硬冲突**，需新布局抽象 |
| 池形状 | `KvBlocks { k_cache, v_cache }` 两平面 | 每层一条 record（K+V+scales） | **硬冲突**，enum/方法面改造 |
| 层视图 | `KvBuffers` 只有 k/v/slots/kv_lens/bt | 还要 tail K/V、block→slot、packed record、seq lens | 大改（59 个构造点） |
| 写 KV | K0 / 融合插池直接写 f16/fp8 池 | 旋转后写 tail；满 tile 再 quantize | 新写路径 |
| flush | 无 tile 概念 | 每 128 token host 侧批量 flush | 新调度/生命周期 |
| 前缀缓存 | block 32；store 支持可变 block_tokens | block 128；sink/tail 特殊 | 中改（store 主体已参数化） |
| 快照 | GDN 快照按页边界；`snap_msg_boundary` 可落在 32 对齐 | 必须 128 对齐；partial tail 不可快照 | 中改/门控 |
| image_evict | 依赖 K 的 Δ-rope | K 已旋转/量化，不可重转 | **首版禁用** |
| 选择层/KVMem | 32-token 视图、K 回读 | 不可用 | **首版禁用** |
| graph | decode 图含 K0/attention | 含 WHT、tail scatter、KVarN decode；flush 在窗外 | 中改 |
| spec | verify 写 T 行；按 committed 截断 | KVarN verify 核 + 只 flush committed 满 tile | **首版关闭** |
| TP2 | 每 rank hkv 减半 | record 按 rank hkv 减半；tail 同理 | 可支持 |

### 2.1 store 其实没有想象中那么硬

`KvStore` / `Rel` / `BlockLeg` 的 `block_tokens` 已经是参数：
`SessionBank` 从 `pool.geom.page` 传，`Rel` 校验
`block_tokens == inputs.page_tokens`，`KvStore::new` 也收 `block_tokens`。
真正写死 32 的只有：

- `PrefixId::blocks_for()`（当前无生产调用者）；
- `ChainCursor::advance_tokens()`（生产走 `advance(page, ...)`）；
- 大量注释/测试；
- `disk_index` 的注释（格式本身带 `block_tokens`）。

**所以 store 的 page128 改造主要是“语义/测试/边界”工作，不是重写。**
但 `scheduler` 的 `prefill_align_page`、`decode_boundary_in` 会自动跟
`pool.page`；`msg_boundary` 也收 `page`。这块比施工文档估计的轻。

### 2.2 真正的硬点

1. **`KvQuant` / `PerTokBytes` / `ctx_max` 的元素口径**
   - `PerTokBytes::main_kv = n_full * 2 * hkv * hd * kv_elem`
   - `ctx_max` 用 `page_guess = 32`
   - `StatePool::alloc_pages` 按 `kv_elems = nb*hkv*hd*page` 分配
   - KVarN 必须改成 `record_bytes * hkv * n_full / page` 的 token 口径；
     `size_bytes()` 不能再用。

2. **`KvBlocks` 单 record**
   - 现有 `k_block_bytes()` 被 `image_evict` / rope shift / 索引使用；
     KVarN 下必须 **显式报错**，不能因为 `dims.dtype == F16` 就被
     `k_shift_supported()` 误放行。

3. **tail/sink 生命周期**
   - 每个 session 的 sink（block0）与 tail（当前未满 block）要跨 turn 存活；
   - 部分 tail 页不能下沉/驱逐；
   - session 关闭时：sink 可 flush 成 int4 供前缀缓存；partial tail 丢弃；
   - prefix hit 的 block0 可能已经是 int4，不能重新标成空 sink（上游
     issue #10 的重复循环就是空 sink 槽读垃圾）。

4. **flush 时序**
   - 上游在 `build()` 里、forward 之前 flush **已提交的满 block**；
   - owl 的 decode 图在 forward 内写 K/V，flush 必须放在图外；
   - prefill chunk 可能在 chunk 内跨多个 tile：要么 chunk 后再 flush，
     要么在 chunk 前 flush 上一轮满 tile；当前 chunk 的满 tile 由
     attention 直接读 raw K/V（materialize/FA2 路径天然支持）。

5. **prefix/snapshot 的粒度**
   - 128-token 前缀粒度会降低命中率（尤其 32/64 token 的短前缀）；
   - `snap_msg_boundary` 要向下取整到 128，且 `snap_keep`/`capture_gdn_snap`
     必须只在 128 边界；
   - 部分 tail 不能作为前缀缓存内容；跨 turn 的前缀命中只能到最后一个
     完整 128 块。

6. **prefill 内存**
   - materialize scratch 在 262k / hkv4 / hd256 / f16 = 约 1 GiB；
   - 3090 现役档 post-graph free ~1.1 GiB，但**硬可用 = free − reserve**
     只有 ~0.59 GiB（1 GiB 是 N6 告警目标，不是预留）；因此 1 GiB
     scratch 装不下。3090 上要么关 vision/降 ctx，要么做 packed prefill。
     3080 TP2 每 rank 约 0.5 GiB，相对宽裕。

7. **spec verify**
   - DFlash2 verify 一次写 T 行（含未接受），KVarN 只能 flush
     `committed` 以下的满 tile；tail 里未接受的槽会在下一步被覆盖；
   - 需要 KVarN verify 核，或 materialize+FA2（慢）；
   - 首版 spec-off 是硬边界，不是“顺手开”。

---

## 3. 容量测算（用现役 3090 档的真实 boot 账）

现役 3090 档 boot（本地实测档，2026-10-25 的 3090 日志）：

```
权重 18,041.5 MiB
KV 池 2,496 MiB（FP8，79,872 tok × 32 KiB）
快照 303.8 MiB（snap_max=1）
DFlash2 草稿 ring 40.6 MiB
GDN 151.9 MiB
预算 cap = 90,025 tok（per_tok 32,768 B）
```

由 `cap × per_tok` 反推 pool 预算约 **2,813 MiB**（已扣 fixed）。
KVarN 首版还要加 tail pool；按上游 `2*S + prefill_blocks + 8` 槽、
每槽 8 MiB（27B/TP1）估算，chunk512 约 112 MiB，chunk2048 约 208 MiB。
下表用 **112 MiB**（乐观）：

| 3090 档 | k4v4 (17,536 B/tok) | k4v2 (13,440 B/tok) |
|---|---:|---:|
| vision + snap + spec-off | **~161k** | **~211k** |
| 再关 snap（+304 MiB） | ~182k | ~238k |
| 再关 vision（+~880 MiB） | **~235k** | **~306k** |
| 再把 reserve 512→256（+256 MiB） | ~250k | ~325k |

> 这些是**算式估算**，不是 boot 实测；KVarN 的图 slab、tail pool 实际
> 槽数、spec-off 释放的 draft/graph 内存都会让数字上下浮动。
> 结论不变：**k4v4 在保留 vision 的 3090 档约 160k；即使全关也只在
> 235k 左右，到不了 262k。k4v2 才是 262k 档的候选。**

3080 TP2（现役日志）：

```
每 rank per_tok = 16,384 B；cap = 427,128 tok；pool 4,096 MiB/rank
```

KVarN k4v4 每 rank 8,768 B/token，k4v2 6,720 B/token。粗算：

| 3080 TP2 | k4v4 | k4v2 |
|---|---:|---:|
| 每 rank 可容 token | ~780k | ~1.0M |
| 2 × 262k 会话 | 可 | 可 |
| 3 × 262k 会话 | 边缘 | 可 |

但现役 `max_concurrency=1`、`gdn_slots=1`；要用这个容量，必须另开
多会话/GDN 槽，且首版 spec-off。**这不是“免费”的并发。**

---

## 4. 代码爆炸半径（按子系统）

| 子系统 | 要做的事 | 规模/风险 |
|---|---|---|
| `shared/config` | `KvLayout`/preset、page 由 quant 定、拒启门（image_evict/select/K rope shift/spec） | 中 |
| `models/env` | `KvQuant::Kvarn{...}` 或独立 `KvStorage`；`KvEnv` 选择子；`kv_paged_policy` 改 quant 感知 | 中 |
| `engine/state` | 单 record 分配；`KvBlocksKind`；tail pool；`block_bytes/k_block_bytes`；`pool_layout` 单 piece/层 | **大** |
| `engine/session` | page128 块腿；tail/sink 生命周期；snap 门控 | 中 |
| `engine/scheduler` | `ctx_max` page、prefill 对齐、前缀/flush 边界、materialize chunk 与 Marlin 512 的冲突 | 中 |
| `engine/exec` | flush 编排、graph 外的 block→slot 表、spec 关闭门 | 中 |
| `models/layers/attention` | KVarN 分支（先于 `kv_paged_policy(F16)`）、WHT、tail write、decode/FA2 路径 | **大** |
| `models/module` | `KvBuffers` 加 KVarN 视图（59 个构造点） | 大但机械 |
| `kernels` | WHT、Sinkhorn AOT、RTN/pack（新）、tail scatter AOT、decode AOT、materialize build AOT、FA2 hd256/GQA/causal | **大** |
| `store` | block128 语义/测试；partial tail pin；sink/int4 两态 | 中 |
| `spec` | 首版关闭；后续 verify 核 + committed flush | 大（二期） |

**最小可行纵切**（0.8B，spec-off、prefix off、vision off、image_evict off）：

```
config/pool/state 的单 record + page128
→ 0.8B 单层 KVarN store/decode
→ 0.8B 整模 decode + materialize prefill
→ 质量/速度/显存门
```

这个纵切也要覆盖 `KvBuffers`、graph、flush、tail 生命周期；不是“只改
一个 dtype”。

---

## 5. 对原施工文档的修正

1. **§5.4 S1“所有满 tile 进 int4、partial tail 每 turn scratch、不做 sink”**
   → 应改为 **sink + session 级 tail 常驻**。理由见 §1.2。sink 成本
   8 MiB/会话（TP1），远小于一次长上下文失败。
2. **§6“优先 AOT Triton，store 参考上游”**
   → 上游 store 的 RTN/pack 是 PyTorch；**没有现成 Triton store 可 AOT**。
   Phase 1 必须新增 RTN/pack/record 写核。Sinkhorn 可以 AOT。
3. **§5.5 prefill“materialize 只用于短 ctx/质量门”**
   → 首版可以 materialize + **扩展 owl 已有 FA2**（hd256/GQA/causal），
   但 materialize 需要大 chunk 摊薄，而 owl 的 512 是 Marlin 调优值，
   二者冲突；且 262k 的 1 GiB scratch 在 3090 上不可行。长期要 packed
   prefill，不能只靠“把 chunk 调到 2048”。
4. **§5.6 spec“首版 spec-off，后续移植 verify”**
   → 正确，但要把“首版不能用于 DFlash2 生产档”写成硬边界；verify 不是
   小移植，且上游 KVarN+spec 在 112k 有 2.13× step tax。
5. **§3 容量“k4v4 1.87×，262k 可期”**
   → 1.87× 是相对 FP8 的 KV 容量，不是相对 owl 的**可用上下文**。owl
   3090 的 fixed 开销（weights/vision/graph/snap/GDN）吃掉大部分
   headroom；k4v4 约 160k，k4v2 才接近 262k。

---

## 6. 建议路线与 Go/No-Go

### Phase -1：先定目标（0.5 天，不写代码）

三选一：

- **A. 单 3090 262k**：选 **k4v2/g128**，接受 spec-off、vision off、
  快照收紧；k4v4 不作为主档。
- **B. 单 3090 质量优先 >150k**：k4v4/g128 可以，但目标写 150–180k，
  不写 262k。
- **C. 3080 TP2 262k 多并发**：k4v4 可容 2 条，k4v2 可容 3 条；需要
  多会话/GDN 槽与 spec-off 的单独验收。

### Phase 0：单 tile 数值 spike（3–5 天）

- Python golden：Hadamard、Sinkhorn 8 iter、K/V RTN、pack、record 偏移。
- 新核：WHT、Sinkhorn AOT、RTN/pack（K/V 各一）、dequant。
- 门：packed 字节逐字节一致；scales f16 逐位一致；dequant cosine
  ≥ 0.999；record 大小/偏移与 Python 一致。
- **不过门就停**，不要进引擎。

### Phase 1：单层/0.8B decode（3–5 天）

- tail pool + block→slot + host flush；KVarN decode 单层对拍
  “dequant → f16 attention”。
- 门：cosine ≥ 0.999；长 ctx 无 NaN；WHT 往返恒等。
- 此时可先不做 sink，用 k4v2/k4v4 两档对拍，量出质量差。

### Phase 2：0.8B 整模纵切（1 周）

- page128、单 record、sink+tail、spec-off、prefix off、materialize
  prefill + FA2 causal（先扩 hd256/GQA）。
- 门：GSM8K n=50 不劣化 >2 点；无 Xid/自旋；显存账与模型一致。
- **Go/No-Go：若 decode 比 FP8 慢 >2×，或 0.8B 质量明显掉，停。**

### Phase 3：27B 3090 目标档（1–2 周）

- 目标 ctx（A=262k / B=160k）、prefix cache、snapshot、host offload。
- 门：needle 16k/64k/128k/240k；GSM8K 200；std 五点三轮；与 FP8
  基线（spec-off）对比；boot `[ctx-max]` 与实测一致。
- **Go/No-Go：容量不达目标或 decode tax >50%，停。**

### Phase 4：spec verify / 3080 多并发（二期，按目标）

- KVarN verify 或 materialize verify；committed-only flush。
- 3080 TP2 多会话/GDN 槽；或 single-3090 DFlash2 恢复。
- 这一期才谈“替代生产档”。

---

## 7. 一句话建议

**KVarN 可以落地，但不要把“k4v4”当成默认答案：k4v4 是质量优先档，
在 owl 3090 上约 160k；容量优先档应是 k4v2。先做 Phase 0 单 tile/
单层 spike，用 k4v2 与 k4v4 双档实测再决定是否改引擎。**

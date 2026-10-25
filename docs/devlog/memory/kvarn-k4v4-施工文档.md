# KVarN k4v4 / g128 施工文档（交接）

> 2026-10-25 立项。目标：在 owl 中实现 **KVarN k4v4、tile/block=128、g128**
> 主 KV 档。本文件是**交接给下一个会话的施工底稿**，不是完成记录。
>
> 代码基线：`fix/spec-tailcap@0a62b747`（含 DFlash2 tail-cap P0 修复）。
> 工作树：`packages/owl-worktrees/kvarn-k4v4`，分支 `feat/kvarn-k4v4`，未提交。
> 构建用**本工作树自己的 `target/`**，不要用主工作树 target。
> Triton AOT 只借 `packages/owl/crates/kernels/.venv/bin/python` 作为工具环境。

## 0. 一句话

KVarN 不是一个 dtype，而是 **128-token tile 级量化 + 独立 attention backend**：
Hadamard 旋转 → 迭代方差归一（Sinkhorn）→ K/V 非对称 RTN（K4/V4）→
tile 共享 scale/zero 打包。owl 现有 page32 / 前缀缓存 / K/V 双平面 / FI 图
都需要为它开一条**独立后端路径**，不能靠改一个 `kv_fp8` bool 完成。

## 1. 范围与裁决

### 1.1 已定

- 档位：**k4v4**（K=4-bit，V=4-bit）。
- tile = block = **128 tokens**（KVarN 设计点 g128）。
- 目标硬件：单 3090 Ti / 双 3080 TP2，27B AWQ-INT4，FP8 为对照。
- 首版优先质量/正确性；性能后置。
- 首版不支持：`image_evict`（K 已旋转/量化，无法做 f16 重转）。

### 1.2 明确不做（首版）

- KVarN k4v2 / g64 多档（先把 k4v4/g128 打通）。
- 与 FI / XQA / fused qknorm-insert 混用（KVarN 走自己的 attention）。
- KVarN + DFlash2 spec verify（首版 spec-off；后续再移植 fused verify）。
- 运行期自适应 tile 大小。

## 2. 参考实现与事实

必读（外部存档）：

- `repos/HyperQwen/kvarn/files/vllm/model_executor/layers/quantization/kvarn/config.py`
- `repos/HyperQwen/kvarn/files/vllm/model_executor/layers/quantization/kvarn/sinkhorn.py`
- `repos/HyperQwen/kvarn/files/vllm/v1/attention/ops/kvarn_store.py`
- `repos/HyperQwen/kvarn/files/vllm/v1/attention/ops/kvarn_decode.py`
- `repos/HyperQwen/kvarn/files/vllm/v1/attention/ops/triton_kvarn_sinkhorn.py`
- `repos/HyperQwen/kvarn/files/vllm/v1/attention/ops/triton_kvarn_decode.py`
- `repos/HyperQwen/kvarn/files/vllm/v1/attention/backends/kvarn_attn.py`
- `repos/HyperQwen/kvarn/README.md`（0.30 移植流水/布局/坑）

关键事实：

- `KVARN_PRESETS` 含 `k4v2_g128`、`k4v4_g128`、`k4v2_g64`、`k4v4_g64`；
  K store 里 `assert bits == 4`，**当前 K 固定 4-bit**；V 预设为 2/4。
- KVarN 的 **tile 必须等于 block**：scale 是 tile 共享，block 边界就是 tile 边界。
- K 方向：`[D, G]`（channel × token），Sinkhorn 后 **per-channel** RTN；
  V 方向：`[G, D]`（token × channel），**per-token** RTN。
- KVarN 用 Hadamard（Sylvester 归一化，H=Hᵀ、H·H=I）沿 head_dim 旋转；
  attention 在旋转后的坐标系计算，输出再乘 H 回到原坐标系。
- KVarN 的 fp16 sink/tail：前 128 token 与未满 tile 的尾部不进 int4，放在
  fp16 pool；这是它质量的重要部分，也是 owl 集成最麻烦的部分。

## 3. 数据布局（k4v4 / g128 / hd256）

一个 `(block, kv-head)` 是一个 **35072 字节** 的 record：

| 字段 | 字节 | 偏移 | 说明 |
|---|---:|---:|---|
| K packed | 16384 | 0 | `[D, G/2]` uint8，4-bit 两两一字节，低 nibble=偶索引 |
| K s_col | 512 | 16384 | `[D]` f16，吸收后的 per-channel scale |
| K zp | 512 | 16896 | `[D]` f16，吸收后的 per-channel zero |
| K s_row | 256 | 17408 | `[G]` f16，per-token Sinkhorn scale |
| V packed | 16384 | 17664 | `[G, D/2]` uint8，4-bit 两两一字节 |
| V s_col | 512 | 34048 | `[D]` f16，per-channel Sinkhorn scale |
| V s_row | 256 | 34560 | `[G]` f16，吸收后的 per-token scale |
| V zp | 256 | 34816 | `[G]` f16，吸收后的 per-token zero |

`35072 = 128 × 274`，可被 128 整除；U32 字数 = 8768。

27B（16 full layers × 4 KV heads × hd256）：

- 每 block 每层：`4 × 35072 = 140,288 B`
- 每 token 每层：`140,288 / 128 = 1,096 B`
- 每 token 全模型：`16 × 1,096 = 17,536 B`
- 相对 FP8（32,768 B/token）：**1.87× 容量**；相对 f16：3.74×。

> hd128/g128/k4v4 的 record = 17,920 B（与 KVarN 文档一致），
> 说明我们的偏移公式与上游一致；27B 用 hd256。

## 4. 量化数学（必须逐式复刻）

### 4.1 Hadamard

`H` = Sylvester 归一化矩阵，`H = Hᵀ`，`H·H = I`。对 K/V 的每个 token 向量
沿 `D` 做 `x ← x·H`；attention 的 Q 也要先转，输出再乘 H 回来。

首版建议：实现一个 `kvarn_hadamard_f16`（WHT 蝶形，8 阶 for D=256），
不要用矩阵乘；矩阵乘会多一份 256×256 权重/GEMM 启动。

### 4.2 Sinkhorn（log-domain，8 iters）

参考 `sinkhorn.py::variance_normalize`：

```
m = tile.float()
log_s_col = 0[C]; log_s_row = 0[R]
cur = m / exp(log_s_col) / exp(log_s_row)
imb_best = imbalance(cur); sc_best=exp(log_s_col); sr_best=exp(log_s_row)
repeat 8:
  col_std = clamp(cur.std(dim=0), 1e-3, 1e3)
  log_s_col = clamp(log_s_col + log(col_std), -0.3, 10.0)
  cur = m / exp(log_s_col) / exp(log_s_row)
  row_std = clamp(cur.std(dim=1), 1e-3, 1e3)
  log_s_row = clamp(log_s_row + log(row_std), -0.3, 10.0)
  cur = m / exp(log_s_col) / exp(log_s_row)
  imb = imbalance(cur)
  if imb <= imb_best: imb_best=imb; sc_best=...; sr_best=...
balanced = m / sc_best / sr_best
```

`imbalance = (col_std.max/col_std.min) + (row_std.max/row_std.min)`。

K tile `[D,G]`：`s_col` 是 per-token（G 轴），`s_row` 是 per-channel（D 轴）。
V tile `[G,D]`：`s_col` 是 per-channel（D 轴），`s_row` 是 per-token（G 轴）。

### 4.3 RTN / 吸收

K（`[D,G]`）：
```
lo = min(balanced, dim=1); hi = max(balanced, dim=1)
rtn_scale = max((hi-lo)/15, 1e-10); rtn_zp = lo
q = clamp(round((balanced - rtn_zp)/rtn_scale), 0, 15)
s_col_K = s_chan * rtn_scale    # s_chan = s_row_sinkhorn
zp_K    = s_chan * rtn_zp
s_row_K = s_tok                 # s_col_sinkhorn
```

V（`[G,D]`）：
```
lo = min(balanced, dim=1); hi = max(balanced, dim=1)
rtn_scale = max((hi-lo)/15, 1e-10); rtn_zp = lo
q = clamp(round((balanced - rtn_zp)/rtn_scale), 0, 15)
s_col_V = s_chan                 # s_col_sinkhorn
s_row_V = s_tok * rtn_scale      # s_tok = s_row_sinkhorn
zp_V    = s_tok * rtn_zp
```

反量化：
```
K = (q * s_col_K + zp_K) * s_row_K
V = (q * s_row_V + zp_V) * s_col_V
```

打包：`bits=4`，每字节两个；索引 `c` 在 byte `c/2`，偶索引低 nibble、
奇索引高 nibble（与 `_pack_lowbit` 一致）。

## 5. owl 侧改动面

### 5.1 配置 / 类型

- 新增 `KvQuant::KvarnK4V4`（或 `Kvarn { k_bits, v_bits, group }`，首版固定）。
- `pool.kv_dtype = "kvarn_k4v4"`；与 `kv_fp8` 互斥。
- boot 门：
  - 只允许 `page=128`、`hd=256`、`g128`；
  - `image_evict=false`；
  - `qkv_fuse=false`、`prefill_split=false`、`xqa_prefill=false`；
  - `decode_arm` 必须是 `kvarn`（新增）；`verify_arm` 首版要求 `v2` 且
    `spec.depth=0`，或新增 `kvarn` verify 后再开 spec；
  - `fi` 面关闭。
- `KvEnv` 增加 KVarN 选择子：store / decode / prefill / hadamard；
  `is_fp8()=false`、`is_k8v4()=false`。

### 5.2 池几何

- **KVarN 档 page=block=128**。`kv_paged_policy` 不能再固定 32；改为
  `kv_page_for(quant)`：KVarN→128，其余→32。
- `StatePool::PoolGeom.page` 跟档；store/前缀缓存/快照的 `block_tokens`
  自动变成 128（`SessionBank` 从 `pool.geom.page` 取）。
- `runtime.max_seq` 必须 128 对齐（现有 79872/262144 都满足）。
- `V2_PARTITION_TOKENS=512` 仍可用（4 页/分区）；`snap_decode_stride` 等
  按 128 页解释。
- `image_evict` 直接拒（K 已量化，无法重转）。

### 5.3 K/V 存储形状

KVarN 每个 `(block,head)` 是**一个连续 record**，不是 owl 现有 K/V 双平面。
建议显式化，不要用 dummy：

```rust
pub(crate) enum KvBlocksKind {
    Split { k: BlockN, v: BlockN },     // f16/fp8/K8V4
    Kvarn { rec: BlockN },              // KVarN record
}
```

`KvBlocks` 改为持有 `KvBlocksKind`（或 `rec: Option<BlockN>` + 原有 k/v）；
`pool_layout` 对 KVarN 只发一个 `PieceSpec`（每页字节 = `hkv*35072`），
`block_bytes()` 直接用它；`k_block_bytes()` 在 KVarN 档返回错误/0
（image_evict/rope shift 不支持）。

### 5.4 Store / 前缀缓存 / 三层

- store 的 block 粒度 = 128；`KvStore::new(..., block_tokens=page=128, block_bytes=...)`
  已由 `SessionBank` 从池几何取，但需核对所有 `32` 字面量与 `x` 参数。
- 前缀哈希按 128-token 内容；命中/回灌/宿主/磁盘都按 128 页搬运。
- **sink/tail 是最大分叉**：
  - 首版建议 **S1：所有满 128 tile 都进 KVarN int4，部分尾 tile 用每 turn
    f16 scratch，不进前缀缓存**；先打通正确性/质量，再补 sink。
  - 若质量不达标，再做 **S2：前 128 token 的 f16 sink + 未满 tail 放每 turn
    f16 pool**；sink 不参与前缀缓存（命中时从 prompt 重 prefill 首 tile）。
  - S3：把 sink 也做成 store 资产（key=chain hash），但需要 store 支持
    “同 block 两种字节形态”，不建议首版。
- `prefix_cache` 首版可以先关（`pool.prefix_cache=false`）做质量门；
  打开后必须验证 128-token 链的命中/驱逐/复活。

### 5.5 Attention

KVarN 的 Q/输出都在旋转坐标系：

- Q（post-RoPE）先 `H` 旋转；
- K/V store 时 `H` 旋转；
- attention 输出再 `H` 逆旋转（H 对称，同一矩阵）。

首版路径：

- **decode**：KVarN fused decode（packed tile + f16 tail）→ 输出旋转域 →
  inverse H。
- **prefill**：两条路可选：
  - P1（先通）：materialize 历史 KVarN tile 为 f16（用 KVarN build-packed
    等价核），拼当前 chunk 的 raw K/V，走现有 f16 prefill；只在短 ctx/
    质量门用，长 ctx O(ctx) scratch 不可接受。
  - P2（目标）：写 KVarN paged prefill（q=T，读 packed tile，dequant on the fly，
    因果 mask），避免 materialize。
- **spec verify**：首版 spec-off；后续移植 `_kvarn_fused_verify_stage1`
  （VQ_INDIRECT）或先走 KVarN decode 逐 token。

### 5.6 调度

- 新增 `DecodeArm::Kvarn`；`KvEnv::decode` 返回 KVarN 核。
- prefill 走 KVarN 自己的分派，不复用 FI/split/chunked f16 核。
- `KvMem` 选择层/虚拟化首版不接（KVarN tile=128 与 32-token 选择层视图冲突）。

## 6. 内核路线：优先 AOT Triton，不先重写 CUDA

owl 已有 Triton AOT 先例（`crates/kernels/triton/`，GDN 六核）。KVarN 的
Sinkhorn/decode 本来就是 Triton，建议：

1. vendor 需要的 KVarN 源到 `crates/kernels/triton/kvarn/`：
   - `sinkhorn.py`（205 行）
   - `triton_kvarn_sinkhorn.py`
   - `triton_kvarn_decode.py` 的 stage1/stage2/verify 子集
   - `kvarn_store.py`/`kvarn_decode.py` 仅作为 host 参考，不进 cubin
2. 写 `triton/kvarn_aot_compile.py` + `kvarn_k4v4_g128_hd256.json`
   （固定 BLOCK_N/num_warps/num_stages/F16=false；KVarN 的 autotune 必须烘焙）。
3. `crates/kernels/build.rs` 加 `build_kvarn_assets`（照 `build_gdn_assets`）。
4. `crates/kernels/src/family/kvarn.rs` 放 ABI/资产/launch 常量；
   `server/kvarn.rs` 做多核编排；`client/kvarn.rs` 放槽序契约。
5. 先只接 **store（WHT+Sinkhorn+RTN+pack）** 和 **decode stage1/2**；
   prefill 可先 materialize 参考，再接 packed prefill。

不建议首版从 Triton 重写成 CUDA：decode 1.2k 行 + Sinkhorn 205 行，
重写风险高；AOT 可以把“数值等价”交给上游源。

## 7. 分阶段施工与门

### Phase 0：参考与 host 对拍（1–2 天）

- 用 Python 参考（`sinkhorn.py`、`kvarn_store.py`、`kvarn_decode.py`）生成
  固定随机 tile 的 golden：
  - K/V 旋转、Sinkhorn scales、packed 字节、反量化误差。
- 在 Rust/CUDA 或 Triton AOT 侧实现 store/dequant，逐字节/逐值对拍。
- 门：`max|Δ| ≤ 1e-3`（f16 源）/ packed 字节逐字节一致。

### Phase 1：单 tile 写/读核（2–4 天）

- `kvarn_store_tile`（WHT + Sinkhorn 8 iter + K/V RTN + pack + scales）。
- `kvarn_dequant_tile`（packed + scales → rotated f16）。
- 门：随机 128×256 tile，K/V cosine ≥ 0.999；packed 字节与 Python 一致；
  scales 逐 f16 相等；Sinkhorn 8 iter 与参考一致。

### Phase 2：decode 核（3–5 天）

- stage1/stage2 split-K decode，hd256/g128/K4V4，f16 dequant 路径。
- 门：与“dequant tile → f16 attention”host 金标 cosine ≥ 0.999；
  长 ctx 不 NaN；rotated Q/O 往返恒等。

### Phase 3：引擎 short-ctx 闭环（3–5 天）

- `KvQuant::KvarnK4V4`、page128、单 buffer record、S1 sink/tail、
  `DecodeArm::Kvarn`、prefill materialize 路径。
- 0.8B 模型：生成冒烟、GSM8K n=5/50、无 NaN；
- 27B 3090：短 prompt + max_tokens 64/256，spec-off。
- 门：与 FP8 同题 GSM8K 不劣化 >2 点；无 Xid/自旋。

### Phase 4：长 ctx / 质量 / 速度（5–10 天）

- packed prefill（P2）或 materialize 分块；needle 16k/64k/128k/240k；
- GSM8K 200、needle、PPL/KLD（若 logprobs 已补）；
- std 速度；与 FP8 基线报告对比；
- 决定是否补 sink/S2、spec verify、prefix cache。

## 8. 风险清单

1. **128-token 前缀粒度**：owl 现有 32-token 假设遍布 store/快照/image；
   必须逐处扫 `32` 字面量（`rg 'page|block_tokens|32'`）。
2. **K/V 单 record**：现有 `KvBlocks {k,v}` 被大量代码消费；改 enum 会触及
   store/state/session/engine，需一次性完成，不要用 dummy v_cache。
3. **Sink/tail**：不做会掉质量；做会破坏“统一块”假设。先 S1 量化全 tile，
   用质量门决定是否上 S2。
4. **Prefill**：materialize O(ctx) 在 262k 不可用；P2 packed prefill 是长 ctx
   的关键，可能比 decode 更难。
5. **Spec**：KVarN verify 与 DFlash2 的 T=8 图/锚复用需要单独设计；首版 spec-off。
6. **Triton AOT**：KVarN 源依赖 `vllm` 符号，需要 shim；autotune 必须烘焙；
   AOT 配置表要与 cubin `shared` 回读校验。
7. **性能**：KVarN decode 本身有 20%+ 税；owl 的 v2/图路径不能直接套用。
8. **质量数字不可跨栈**：HyperQwen 的 +0.16% PPL 是 k4v2；k4v4 需 owl 自测。

## 9. 开工顺序（建议）

1. 先读 `sinkhorn.py` + `kvarn_store.py`，把 Phase 0 golden 脚本放在
   `crates/kernels/triton/kvarn/reference/`（不参与构建）。
2. 做 store/dequant 单 tile，不碰引擎；对拍通过再写 decode。
3. 同时做 `kv_page_for(quant)` 和 `KvBlocksKind` 的结构改造（纯 host 单测），
   把 page128 路径先跑通空池/预算。
4. 最后接引擎；不要先改 decode 再补 store。

## 10. 交接检查表

- [ ] 工作树 `packages/owl-worktrees/kvarn-k4v4`，分支 `feat/kvarn-k4v4`
- [ ] 基线 `fix/spec-tailcap@0a62b747`（含 P0 修复）
- [ ] 本工作树 `target/` 构建；不要用主工作树 target
- [ ] Triton 工具环境：`GDN_TRITON_PY=/home/div/Documents/codes/packages/owl/crates/kernels/.venv/bin/python`
- [ ] 参考文件路径见 §2
- [ ] 首版裁决：g128/block128、k4v4、spec-off、image_evict=false
- [ ] 第一个门：单 tile store/dequant 对拍，不接引擎

---

**交接一句话**：KVarN k4v4 的核心不是“把 K/V 压到 4-bit”，而是 **128-token
tile + Hadamard + Sinkhorn + tile 共享 scale**；owl 需要一条 page128 的独立
KV 后端。先做单 tile 数值对拍，再做 decode，最后接引擎。

---

## 11. 2026-10-25 可行性复核补记（优先读）

后续调研结论见
[`kvarn-k4v4-落地可行性复核.md`](kvarn-k4v4-落地可行性复核.md)。对本文件
以下判断作修正：

1. **§5.4 的 S1（满 tile 全 int4、partial tail 每 turn scratch、不做 sink）
   不采用。** 上游默认 **sink(block0 f16) + session 级 tail f16**；sink
   只多 ~8 MiB/会话（TP1），不应先省。partial tail 必须跨 turn 存活，
   且不能以 packed record 形式下沉/驱逐。
2. **§6 的“store 优先 AOT Triton”要拆开看。** 上游 port 里只有
   Sinkhorn 是 Triton；RTN/pack/record 组装是 PyTorch，且 port 中没有
   `triton_kvarn_store.py`。Phase 1 必须新写 RTN+pack+record 写核。
3. **§5.5 的 prefill 路线可先借 owl 已有 FA2。** `cu/flash_attn/owl_fa2.cu`
   当前只支持 hd64/72、非因果、无 GQA；扩成 hd256+GQA+causal 后，
   materialize + FA2 可作为首版 prefill/verify 正确性臂。但 262k 的
   f16 scratch 约 1 GiB，3090 装不下，且 chunk 需从 512 提到 2048。
4. **§3 的容量结论要按 owl 的 fixed 开销重算。** k4v4 1.87× 是相对
   FP8 的 KV 容量；现役 3090 档（vision+DFlash2+1 快照）估算只有
   **~160–180k**，全关 vision 也约 **235k**，到不了 262k。262k 应选
   **k4v2**（约 211–238k，关 vision 后 ~306k）。
5. **§5.6 的 spec-off 是硬边界。** 首版 KVarN 不能用于现役 DFlash2
   生产档；verify 是独立核/独立路径，不是顺手开关。

**修正后的第一决策**：先定目标（单 3090 262k → k4v2；质量优先 150–180k
→ k4v4；3080 TP2 多并发 → 两档都测），再做 Phase 0 单 tile spike。

# verify 算子优化：Marlin m8 tile + MLP gate/up 合并（2026-10-23）

> 接 `dflash-optimization-handoff.md` / `dflash-k15-proposalq.md`。
> 目标 = 常规 verify 吞吐；本轮不碰 mask/q。
>
> 注：旧 nsys 构成里的 TP AR 是**未融合两核**（`tp_ar_fused=false`）；
> 2026-10-23 已补 §3.3 的 fused 受控 A/B，**融合只省一次发射，不省数据腿**，
> 端到端约 −0.5~−0.6%，不是把 AR 减半。
> 硬件 = 2×3080（CUDA ordinal 2/3）；target = Qwen3.8-27B AWQ + DFlash2 d7。

## 0. 结论

- **Marlin m8 tile**：3080/3090 上 m=1/7/8 的 n=17408/5120 形状，
  `{thread_k=64, thread_n=128, threads=128}` 比 auto 的 `{128,128,256}`
  快 **1~6%**（n=5120 约 5~6%，n=17408 约 1%）。
- **MLP gate+up 合并**：two×17408 → one×34816，3080 微基准
  m=7 **159.5→152.0µs（−4.7%）**，m=8 159.3→153.4µs（−3.7%），
  m=1 155.8→150.6µs（−3.4%）；64 层理论省 **~0.4–0.5ms/轮**。
- **引擎实测**：
  - K7 / 1k：verify **27.66→27.28ms（−0.39ms，−1.4%）**，token 逐位相同；
  - 32K fixed-tail：verify **68.79→68.37ms（−0.42ms，−0.6%）**，
    round 80.66→80.50ms，**64/64 token 与 f16 全等**；
  - old i8 32K 输出与新 i8 **逐位相同**。
- 两项合计目前约 **0.4ms/轮**，不是数量级；但都是零语义变化、可叠加。
- **TP AR fused（P0）**：K7 N=192 round **34.750→34.586ms（−0.47%）**、
  verify **27.237→27.002ms（−0.86%）**，token 逐位相同；nsys AR 核时
  **198.0→173.3ms（−12.5%）**。融合只省一次发射、不省跨卡数据腿，
  端到端 −0.5~−0.6%，**不是把 18% 砍半**。生产档已开。
- **AR-2（2026-10-23 晚）**：verify/propose 80KB 小包走单 block、v1
  produced/consumed 相对游标 + release/acquire；去掉尾部 rendezvous/
  wait_done/blocks_done。K7 1k **round 34.586→33.316ms（−3.7%）**、
  verify **27.002→25.975ms（−3.8%）**；f8 档 **34.392→32.968ms（−4.1%）**、
  verify **26.978→25.764ms（−4.5%）**，token 逐位相同。80KB AR 核时
  **19.4µs**。生产档已开 `tp_ar2=true`。

## 1. Marlin m8 tile

文件：`crates/kernels/cu/marlin/marlin_host.cu`

- 新增 `small_batch_m8_thread_configs = [{64,128,128}, {128,128,256},
  {128,64,128}]`。
- `determine_exec_config` 在 `m_block_size_8 && thread_m_blocks==1` 时
  先用**完整** `max_shared_mem` 判一次 `{64,128,128}`，命中直接返回
  （不走下面的 `−512` 保守余量）。
- 非 m8 路径顺序不变。

微基准（`marlin_bw_bench::marlin_tile_sweep`，AWQ g32）：

| 设备 | 形状 | auto | k64/n128/t128 | k128/n128/t256 |
|---|---|---:|---:|---:|
| 3080 | 17408×5120 m8 | 78.1µs | **78.0** | 78.4 |
| 3080 | 5120×17408 m8 | 28.2µs | **27.2** | 28.0 |
| 3090Ti | 17408×5120 m8 | 61.8µs | **61.2** | 61.7 |
| 3090Ti | 5120×17408 m8 | 22.5µs | **21.3** | 21.5 |

> 注：auto 与显式 cfg 的 1µs 差主要来自台子测序/缓存；selector 已改为
> 优先 m8 档，生产收益以引擎 A/B 为准（见 §3）。

## 2. MLP gate+up 合并

### 2.1 装载：虚拟合并键

- `formats::split_gate_up(base)`：`gate_up_proj = gate_proj ⊕ up_proj`
  （同 k 行堆叠 = 字节拼接）。
- `formats::split_merged` = qkvz ∪ gate_up；AWQ / W4A16 / safetensors
  三个源统一走它。
- `SegKind::GateUp` + `rule_of("gate_up_proj") = SegmentedCol(GateUp)`
  ⇒ TP2 每 rank 得到 `gate[local] | up[local]` 两段，和拆键
  `UniformCol` 的分片语义一致。
- `Qwen35Convention` 增加 `gate_up_proj` → `mlp.gate_up_proj`。
- `Mlp::layout` 用 `.with_segments_for("gate_up_proj", [inter, inter])`
  声明**本 rank** 段长（与 GDN `in_proj_qkvz` 同律）。

### 2.2 执行

- `Mlp::new_merged` 只在 **Uniform W4A16 / W4A16Awq** 时启用；
  F16/GGUF/DFlash2 仍用 `Mlp::new`（拆键路径零变化）。
- 目标 `DecoderLayer::new_full/new_gdn` 改用 `Mlp::new_merged`。
- 合并 GEMM 输出 `[T, 2·inter_local]`；新增
  `owl_silu_and_mul_merged_{f16,bf16}`，单输入块内按前半 gate / 后半 up
  计算 `silu(g)*u`，输出 `[T, inter_local]`。
- 为什么不用两个 `slice_view` + 旧 `SiluAndMul`：`slice_view` 的
  `src_dim != out_dim` 会物化两次；merged 核保持一次发射。

### 2.3 微基准（`marlin_mlp_merge_ab`，3080）

| m | two×17408 | merged 34816 | Δ |
|---:|---:|---:|---:|
| 1 | 155.8µs | **150.6µs** | −3.4% |
| 7 | 159.5µs | **152.0µs** | −4.7% |
| 8 | 159.3µs | **153.4µs** | −3.7% |

## 3. 引擎 A/B

### K7 / 1k（`tp_head_i8_gate`，6 prompt × 8 tok，greedy）

| 项 | 旧 | 新 |
|---|---:|---:|
| verify | 27.66ms | **27.28ms** |
| round | 36.41ms | 36.11ms |
| tokens | — | **逐位相同** |

### 32K fixed-tail（N=64，i8，DFlash2 d7）

| 项 | 旧 | 新 |
|---|---:|---:|
| verify | 68.79ms | **68.37ms** |
| round | 80.66ms | 80.50ms |
| decode | 38.84ms/tok | 37.90ms/tok（采样轮次波动） |

- 新 i8 输出 vs old i8：**逐位相同**；
- 新 i8 输出 vs 32K f16：**64/64 逐位相同**。

### 3.3 TP AR fused A/B（2026-10-23，P0）

测试档此前走 `TpEngine::assemble` = `TpOpts::default()`（`tp_ar_fused=false`）。
本轮给 `tp_head_i8_gate` 加 `OWL_TP_AR_FUSED=1`，显式 `assemble_with(TpOpts)`；
并加路径断言（未开 fused 时必须 `tp.ar.v1>0`，开了必须 `tp.ar.fused>0`）。

**K7 / 1k / 6 prompt（i8 head，N=192，greedy）**

| 项 | 两核 v1 | **一核 fused** | Δ |
|---|---:|---:|---:|
| round | 34.750ms | **34.586ms** | **−0.47%** |
| verify | 27.237ms | **27.002ms** | **−0.86%** |
| propose | 5.762ms | 5.875ms | +2.0%（噪声） |
| tokens | — | — | **逐位相同** |

N=96 复测 round 35.079→34.712（−1.05%）、verify 27.256→27.090（−0.61%），
同向。生产档同参 `OWL_TP_AR_F8=1`（prefill 走 f8 两核、decode 走 fused）：
N=192 round 34.602→34.392（−0.61%）、verify 27.158→26.978（−0.66%），
**tokens 逐位相同**。

**nsys node trace（N=32，同一 prompt，16 轮）**

| 核 | 两核（ms / 占） | fused（ms / 占） |
|---|---:|---:|
| `owl_comm_ar_stage_f16` | 86.41 | — |
| `owl_comm_ar_reduce_f16` | 94.98 | — |
| `owl_comm_ar_fused_f16` | — | **158.31** |
| AR 合计（含 bf16） | **198.00 / 16.1%** | **173.25 / 14.3%** |
| 全部 kernel | 1228.7 | 1209.5 |

结论：

- fused **省掉一次 kernel 发射**，AR 核时 −12.5%（198.0→173.3ms）；
- 但数据腿仍是“本端 stage 写 + 跨卡 reduce 读”，没有少传字节，
  所以**不是** 18%→9% 的减半；端到端只有 **−0.5~−0.6%**；
- 剩余 AR 时间主要是跨卡读 + 双 rank rendezvous/skew，不是 launch；
  继续抠要动协议/负载均衡，性价比低于 GDN/head；
- 生产档已开 `[parallel] tp_ar_fused = true`；大载荷仍按 `tp_ar_f8`
  走两核 f8 腿（融合核只非 fp8）。回退删该行即可。

## 4. 失败的尝试：GDN conv q/k/v 三段融合

试过 `owl_gdn_conv_fwd3_f16`（一次发射，输出 `[q|k|v]` 连续块）：

- 核本身可行，但 **fold 记录**是 `conv` 输出的 `slice_view`；
  `GraphPlan` 输出槽校验看**父块长度**，会报
  `r5 归约 40960 ≠ 声明 24576`；
- 单输出块 + 零拷贝切片与现有 fold 记录机制不兼容；
- 若要继续：需要三段独立输出块（多个 output 槽）或 fold 记录物化，
  成本和风险都高于收益，**已回退**。

## 5. 下一步候选

AR 已收口：`ar_fused` + `ar2` 两档都落地。当前 nsys 构成约
Marlin 52% / AR 14~16% / GDN 13.5% / head 8% / attention 2.7% / 其它。
继续按下面的顺序，不要回头磨 Marlin tile：

1. **AR-2 已完成（2026-10-23）**：小包(verify m=8 ≈80KB)走单 block、
   v1 `produced/consumed` 相对游标 + release/acquire；去掉尾部 rendezvous、
   `wait_done/blocks_done` 与全线程 system fence。
   K7 1k **round 34.586→33.316ms（−3.7%）**、verify **27.002→25.975ms（−3.8%）**；
   f8 档 **34.392→32.968ms（−4.1%）**、verify **26.978→25.764ms（−4.5%）**；
   token 逐位相同。80KB AR 核时 **19.4µs**（SGLang 9.3µs 是 10KB 口径，
   不能直接套到 80KB；实际收益不是 8%）。详见
   [`allreduce-sglang-gap-2026-10-23.md`](allreduce-sglang-gap-2026-10-23.md)。
2. **GDN fold/tap 输出槽**：`q_n/k_n/g/beta` 都是 tap 记录，
   `l2norm(q,k)` 双发、`gating_g`+`sigmoid(beta)` 融合若输出单块再切片，
   会与 conv3 一样撞父块长度校验。需要
   `GraphPlan` 输出槽/ fold 记录支持 offset/stride，或真正的多输出块。
3. **GDN 小核融合**：在 2 之后做 `l2norm(q,k)` 双段、
   `gating_g`+`sigmoid(beta)`（各约 0.2ms/轮）。
4. **GDN conv q/k/v**：同一输出槽前提下的三段单发（核已试过，回退）。
5. Attention q/k/v 合并；Marlin 只剩 m=16/prefill 形状可扫。

## 6. 验证命令

```bash
# 微基准（3080 BDF 01:00.0）
OWL_TEST_DEVICE=01:00.0 cargo test --release -p owl-kernels --features marlin \
  --test marlin_bw_bench -- marlin_mlp_merge_ab --ignored --nocapture
OWL_TEST_DEVICE=01:00.0 OWL_TILE_M=8 cargo test --release -p owl-kernels \
  --features marlin --test marlin_bw_bench -- marlin_tile_sweep --ignored --nocapture

# 引擎门（两核基线）
OWL_TP_TEST_DEV_A=2 OWL_TP_TEST_DEV_B=3 OWL_TP_HEAD_ARM=i8 \
OWL_TP_HEAD_SPEC=1 OWL_TP_HEAD_DEPTH=7 OWL_TP_HEAD_N=192 OWL_TP_HEAD_CTX=1024 \
cargo test --release -p owl-engine --test tp_head_i8_gate -- --nocapture

# TP AR fused 臂（加 OWL_TP_AR_FUSED=1；生产同参再叠 OWL_TP_AR_F8=1）
OWL_TP_TEST_DEV_A=2 OWL_TP_TEST_DEV_B=3 OWL_TP_HEAD_ARM=i8 \
OWL_TP_HEAD_SPEC=1 OWL_TP_HEAD_DEPTH=7 OWL_TP_HEAD_N=192 OWL_TP_HEAD_CTX=1024 \
OWL_TP_AR_FUSED=1 OWL_TP_AR_F8=1 \
cargo test --release -p owl-engine --test tp_head_i8_gate -- --nocapture

# 注：并发 cargo 构建后若遇到 lld 的 anon hidden symbol 未定义，
# 是增量缓存污染，不是代码错误；用 CARGO_INCREMENTAL=0 重跑即可。
```

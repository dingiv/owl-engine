# DFlash2 优化交接（2026-10-23）

> **2026-10-23 P0 续（AR fused 已结）**：给 `tp_head_i8_gate` 加
> `OWL_TP_AR_FUSED` → 显式 `TpOpts`；K7 N=192 受控 A/B
> round **34.750→34.586ms（−0.47%）**、verify **27.237→27.002ms（−0.86%）**，
> token 逐位相同；生产同参 `tp_ar_f8=1` 复测 round −0.61%、verify −0.66%。
> nsys 核时 AR **198.0→173.3ms（−12.5%）**，但 fused 只省一次发射、
> 不省跨卡数据腿 ⇒ **不是 18%→9% 的减半**。生产档已开
> `[parallel] tp_ar_fused = true`；f8 大载荷仍走两核。**AR-2 也已落地**：
> 小包(80KB)走单 block、v1 produced/consumed 相对游标、无尾部 rendezvous；
> K7 1k round **34.586→33.316ms(−3.7%)**、verify **27.002→25.975ms(−3.8%)**；
> f8 档 round **34.392→32.968ms(−4.1%)**、verify **26.978→25.764ms(−4.5%)**，
> token 逐位相同；80KB AR 核时 19.4µs（SGLang 9.3µs 是 10KB 口径，不能直接套）。
> 生产档已开 `tp_ar2=true`。独立算子台（`ar_channel_bench`，图内 N=64）：
> AR2 10KB 10.81 / 80KB 20.29 / 160KB 29.49 / 256KB 41.71µs；1-stage
> 12.5 / 18.9 / 26.9 / 34.7µs（80KB+ 更快，10KB 输 AR2；但 27B 生产
> 1-stage 路径历史不触发）。下一步 = GDN fold/tap 输出槽，暂缓 Marlin。
> 详见 `allreduce-sglang-gap-2026-10-23.md` / `verify-op-optimization.md` §5。

> **2026-10-23 续**：K=1..15 单 block、target live P+1、proposal_q +
> 稀疏拒绝采样、32K fixed-tail 质量门已完成/实测。见
> [`dflash-k15-proposalq.md`](dflash-k15-proposalq.md)。一句结论：
> K=15 能跑但比 K7 慢（52.7ms/AL1.81 vs ~35ms/AL1.98），默认仍 d7；
> 32K W8A16 TC f16/i8 **64/64 全等**；采样 temp0.1 **384/384 全等**，
> temp0.7 322/384（采样放大，非 0.95 等价门）。
> masked-compute P+1 已实现且语义逐位一致，但 **verify 无可见收益**
> （K15 42.19→41.65ms，K7 27.66→27.70ms；权重带宽主导），只省末轮 propose。
> **verify 算子优化已起步**：Marlin m8 tile + MLP gate/up 合并，
> K7 verify 27.66→27.28ms、32K 68.79→68.37ms，见
> [`verify-op-optimization.md`](verify-op-optimization.md)。

> 给下一个会话的**唯一入口文档**。先读本文，再读
> `docs/devlog/perf/tp2-decode-fast.md` §10.13/§10.14。
> 当前分支 `main`，**未 push**；本轮（K15/proposal_q/verify-op/AR fused）
> 改动**未提交**，见顶部新注。当前没有 owl server 在跑，GPU 全空。

## 0. 一句话状态

- **默认 head = `rtn-int8-row`（W8A16）已自动走 f16 Tensor Core**：
  HBM 读 int8 权重、寄存器 `int8→half`、f16 MMA；激活不量化。
- **W8A8（`rtn-int8-row-a8`）已退回诊断档**：int8 TC 与 W8A16 TC 同速，
  但 strict token 门 0.6962 vs 0.9896，不再推荐。
- **DFlash2 lookup 已转正**，TP2 主档 `[spec] lookup = true`。
- **W4A8 Marlin int8 m8 已 port，默认关**（3080 无速度收益）。
- **草稿池 ring 已落地**：262k 档 per_tok 16384，`[ctx-max]` 442k。
- 下一步真正的 DFlash 算法空间在 **K=1..15 单 block**（NInfer 路线），
  不是两个 8 行 block 拼接。

## 1. 本轮做了什么

### 1.1 DFlash2 lookup（已转正）

- 实现：`crates/engine/src/exec/lookup.rs`
- 接线：`exec/draft.rs` 的 `dflash_lookup_for_tail` / `apply_dflash_lookup`；
  `exec/spec.rs` 轮末 `toks` 作为 tail。
- 配置：`[spec] lookup / lookup_nmin / lookup_nmax / lookup_nstrong / lookup_agree`
- 语义：在当前请求历史里找当前后缀最近一次出现；强匹配直接覆盖草稿头，
  弱匹配需草稿一致；只覆盖草稿器自己的 7 个位置，不改 verify 图。
- 数据（2×3080，greedy）：
  - copy：AL 5.907 → **6.641–7.000**，decode ~238 → **273–286 t/s**
  - 32K fixed-tail：AL 1.69 → 1.87，轮时 33.79 → 33.92 ms（噪声）
- 回退：`--set spec.lookup=false`。

### 1.2 W8A16 多头 head（已转正）

旧逐行 GEMV 在 spec verify m=8 / DFlash2 propose m=7 时每个 token 重读
0.635 GB 权重。`gemv_i8w_f16.cu` 增加 mt8/mt7：

- `TOK_TILE=8`、`ROWS_PER_WARP=4`、`WARPS=8`、`K_TILE=512`
- 权重只读一遍；m=7 专用 mt7 少算第 8 行
- m=1（prefill 末行/MTP）仍走旧核

核级（124160×5120）：

| 实现 | tokens=8 | tokens=7 |
|---|---:|---:|
| 旧逐行 | 7,039.9 µs | 6,159.5 µs |
| **W8A16 mt8 / mt7** | **2,069.8** | **1,451.0** |

### 1.3 W8A16 f16 Tensor Core（当前默认；本轮最大收益）

文件：`crates/kernels/cu/elems/gemm_i8w_a16_tc.cu`

- HBM 读 **int8 权重**（1 B/参数），shared 仍存 int8
- 每个 MMA fragment 在寄存器里 `int8 → half/bf16`
- `mma.m16n8k16.row.col.f32.f16.f16.f32`（bf16 同形）
- M=16（7/8 token 补零），N=8 权重行/warp，CTA 64 行，K_TILE=256
- 静态 shared 26.3 KB，3 block/SM；driver `smem=0`（**不能**再报 dynamic）
- 激活保持 f16/bf16，**不做 W8A8 激活量化**

核级（3080，124160×5120）：

| 实现 | tokens=8 | tokens=7 | 带宽 |
|---|---:|---:|---:|
| W8A16 FMA mt8/mt7 | 2,069.8 µs | 1,451.0 µs | 307/438 GB/s |
| **W8A16 f16 TC** | **983.8 µs** | **980.5 µs** | **646/648 GB/s** |
| W8A8 int8 TC | 969.3 µs | 964.1 µs | 656/659 GB/s |

引擎 spec 门（6 prompt × 96 tok，DFlash2 d7，2×3080）：

| tag | 旧 W8A16 FMA | **W8A16 TC** | W8A8 TC |
|---|---:|---:|---:|
| `spec.round` avg | 36.649 ms | **35.281 ms** | 35.310 ms |
| `spec.propose` avg | 6.256 ms | **5.970 ms** | 5.993 ms |
| `spec.verify` avg | 28.617 ms | **27.453 ms** | 27.477 ms |

质量门（f16 vs i8，6×96）：

- **570/576 = 0.9896**，逐字全等 **5/6**，过 0.95 门
- 唯一分歧在 CSV prompt idx 90，两边成文，late near-tie
- 与旧 W8A16 FMA 的 token 账也基本一致

### 1.4 W8A8 head（dp4a → int8 TC；已退回诊断）

- `gemv_i8w_a8.cu`：per-token 激活 int8 + `__dp4a`
- `gemm_i8w_a8_tc.cu`：SM86 int8 IMMA `mma.m16n8k32.s8s8s32`
  （30 系**只有 int8 TC，没有 fp8 TC**）
- `quant_i8_act.cu`：per-token scale/q
- 质量门：i8 vs a8 **401/576 = 0.6962**，逐字全等 4/6
- AL 1.98→1.93；语义输出未观察到事实错误，但 strict 门未过
- **裁决：`rtn-int8-row-a8` 只作 opt-in 诊断档，不转默认**；
  W8A16 TC 同速且质量 0.9896，W8A8 已被支配。

### 1.5 草稿池 ring（已落地）

- `engine/src/state.rs`：DFlash2 草稿池按 `window+block` 的 ring 分配，
  不再按 `max_seq` 线性。
- `naive_attn_nc.cu`：新增 `ring_cap`，逻辑位 `% ring_slots`；
  `ring_cap < 2048 + T` 直接 `__trap()`。
- 主档 `lookup=true` 不变；262k 实测与 184k 持平。
- 草稿 ring 槽数 = 2080（2048 窗口 + 8 自块，页对齐）。

### 1.6 W4A8 Marlin int8 m8（已 port，默认关）

- 来源：`vllm-ampere-optimized/flashampere/marlin` @ f2e081d
- `marlin_template.h` 4 处 int8/m8 布局修复 + epilogue per-row scale
- `kernel_selector.h` 增加 18 条 `kS8 × kU4/kU4B8` g32/64/128 m8=true
- 门：m8 与 16-row **逐位相同**；host 1.413e-4；oracle RMS 0.00016
- 3080 A/B：**无速度收益**（m=7/8 持平/略慢），默认
  `[kernel] marlin_w4a8_m8 = false`；`MARLIN_NO_M8` 仅诊断。

## 2. 质量门结论（不要重新踩）

- `rtn-int8-row`（W8A16）：**0.9896，过现行 0.95 门**；默认。
- `rtn-int8-row-a8`（W8A8）：**0.6962，未过**；诊断档。
- f16 vs i8 只有 1 个后段分歧；a8 早期就分叉 ⇒ W8A8 激活量化是主因。
- 现行门是 6×96、spec m=7/8、`max_seq=1024`；**32K fixed-tail 的
  W8A16 TC / W8A8 尚未跑**。
- 不要用“语义看起来对”替代 strict 门；如果要把 a8 转默认，先重新定
  top-1/语义门判据并举证。

## 3. 最终 decode 速度（口径警告）

- 核级：W8A16 TC 相对旧 FMA **2.10× / 1.48×**（8/7 tok）。
- 引擎同门：round **−3.7%**、propose −4.6%、verify −4.1%。
- 生产 `lookup=true` A/B：旧 FMA 与新 TC 的 **token 账、rounds、AL 直方图
  逐字相同**；round 36.65→**35.12 ms** ⇒ 同 AL 折算约 **+4% decode**。
- 单次 std（262k，greedy，15 点）：**decode avg 132.65 t/s**
  （范围 74.4–222.8），prefill 1616.5。旧文档基线 138.1 是另一轮 std，
  prompt/AL 波动很大，**不是受控 A/B**；不要拿单次均值判回归。
- 结论：**head 优化成功，但端到端只有 ~4%**（head 只占 round ~2–3 ms）。
  想要更大 decode 收益，必须动 verify 主干（Marlin/GDN/attention），
  或改算法（K 宽度/更多行），不是继续磨 head。

## 4. 代码地图

### head

- `crates/kernels/cu/elems/gemv_i8w_f16.cu` — W8A16 mt8/mt7
- `crates/kernels/cu/elems/gemm_i8w_a16_tc.cu` — **当前默认 f16 TC**
- `crates/kernels/cu/elems/gemv_i8w_a8.cu` — W8A8 dp4a
- `crates/kernels/cu/elems/gemm_i8w_a8_tc.cu` — W8A8 int8 TC
- `crates/kernels/cu/elems/quant_i8_act.cu` — 激活量化
- `crates/kernels/src/ops/elems.rs` / `driver.rs` — 声明/几何/选路
- `crates/models/src/ops.rs` / `layers/embedding.rs` — 语义动作与选路
- `crates/kernels/tests/gemv_i8w_bench.rs` — 三个 bench + host 对拍

### lookup / ring

- `crates/engine/src/exec/lookup.rs`
- `crates/engine/src/exec/draft.rs`
- `crates/engine/src/exec/spec.rs`
- `crates/engine/src/engine.rs` / `spec_runtime.rs`
- `crates/engine/src/state.rs`
- `crates/kernels/cu/attn/naive_attn_nc.cu`
- `crates/kernels/src/ops/attn.rs`

### tests / profiles

- `crates/engine/tests/tp_head_i8_gate.rs` — spec m=7/8 门
- `profiles/qwen38_27b-3080_tp2-w4a16.toml` — 主档（lookup / TC）
- `profiles/qwen38_27b-3080_tp2-w4a16-a8.toml` — W8A8 诊断档
- `profiles/qwen38_27b-3080_tp2-w4a8.toml` — W4A8 prefill 档

## 5. 怎么跑

### head 门

```bash
cd packages/owl
# 两臂各一进程；N=96
OWL_TP_TEST_DEV_A=2 OWL_TP_TEST_DEV_B=3 OWL_TP_HEAD_SPEC=1 \
OWL_TP_HEAD_ARM=i8 OWL_TP_HEAD_N=96 \
cargo test --release -p owl-engine --test tp_head_i8_gate -- --nocapture
# a8 换 OWL_TP_HEAD_ARM=a8
# 比较（i8 vs a8；预期红）
OWL_TP_HEAD_ARM=compare_a8 \
cargo test --release -p owl-engine --test tp_head_i8_gate -- --nocapture
```

注意：

- `OWL_TP_HEAD_OUT` 是**单臂落盘名**，两臂比较不要设置它；
  默认 `/tmp/owl-tp-head-<arm>.txt`。
- `compare` = f16 vs i8；`compare_a8` = i8 vs a8（**按设计会红**）。
- `OWL_TP_HEAD_SPEC=1` 才走 m=7/8；不开只会测 m=1。

### head 核 bench

```bash
OWL_TEST_DEVICE=C1:00.0 \
cargo test --release -p owl-kernels --features device \
  --test gemv_i8w_bench -- gemm_i8w_a16_tc_bench --nocapture
# 或 gemm_i8w_a8_tc_bench / gemv_i8w_a8_bench / gemv_i8w_mt_bench
```

### 其它门

```bash
cargo test -p owl-kernels --test op_tables
cargo build --release -p server
```

## 6. 下一会话的优先事项

### P0：先确认现状，不要重复劳动

1. `git log --oneline -8`，确认 6 个提交（见 §7）。
2. 读 `tp2-decode-fast.md` §10.13/§10.14。
3. 除非要改算法，否则**不要重新跑 W8A8 质量门**：结论 0.6962 已定。
4. 默认是 W8A16 TC；如果要回退，回退的是 `rtn-int4-g32` 或改代码，
   目前没有 `head_a16_tc` 配置开关（可另立）。

### P1：DFlash2 K=1..15（NInfer 路线，最大算法空间）

关键事实（`repos/ninfer-kvmem-actkv/backends/ninfer`）：

- NInfer DFlash2 支持 `--draft-tokens 1..15`；
- 物理块宽 **W=K+1**，**一个 masked block 一次前向**；
- 不是两个 8 行 block 拼接；
- proposal 计算全部 W 列，target 只验证 `P+1` live 列
  （`proposal_valid_columns` vs `target_valid_columns` / `current_extents`）；
- draft attention 非因果 ⇒ 更长 block 会改变所有 proposal，
  短 block 不是长 block 的前缀；
- checkpoint 的 K 是运行时选择，不是权重维度。

owl 当前：`BLOCK=8` / `DEPTH=7` 硬编码，verify 固定 8 行。
要做 K=15，需要：

- `DFlash2Draft` 的 BLOCK/DEPTH 改为运行时 K；
- verify 图/fold/快照/`CONCAT_ROWS_SLOTS`/`kv_slots` 扩到 16 行；
- proposal/target live 宽度解耦（P/Q）；
- 重新跑 AL/质量门（更长 block 的 draft 质量未知）。

**不要**用两个 8 行 block 链式拼接：第二块依赖第一块未验证 draft，
只有第一块全接受才有效；当前 m7 全接受约 6.8%，大概率白算。

### P2：proposal_q + sparse rejection（采样正确性）

- NInfer DFlash2 selector 输出 `proposal_q FP32 [16,K,B]`；
- `speculative_accept_sparse_drafts` 用 `min(1, p/q)` 接受，
  拒绝时从 `max(p-q,0)` 归一化采样 correction；
- owl 目前 `spec_sampled_accept_prefix` 只比 sampled token == draft，
  **temperature>0 时不等价于目标分布**。
- greedy 生产不受影响；agent 采样档要补。

### P3：verify 算子级优化（不要先做 head）

- verify 27.45 ms，head 0.98 ms；64 层（48 GDN + 16 full）。
- 权重带宽是墙；Marlin W4A16 m8 已走 m8 路径，但 selector/tile 未逐形状扫。
- 可做：
  1. Marlin m8 tile sweep（`threads/n/k`）按真实形状选；
  2. MLP gate+up 合并、Attention q/k/v 合并（少发射、改善小 N 带宽）；
  3. GDN T=8 小核融合（conv q/k/v、l2norm q/k）；
  4. attention 臂 `fi/xqa/v2` 在 8k/32k A/B（32k FI 已 2.29× SM，
     不要盲目 split-KV）。
- 预估：这几条合计 **1–3 ms/round**，不是数量级。

## 7. 本轮提交

```text
3fee883 config: 注册 DFlash2 lookup / W8A8 head / W4A8 m8 开关
34a2350 feat(spec): DFlash2 lookup 草稿 + fixed-tail 测量工具
4843305 feat(head): W8A16 多头快路 + W8A8 int8/TC + W8A16 f16 TC
bd06cfd feat(marlin): 移植 W4A8 int8 m8 路径 + 构建增量修正
10898e8 feat(spec): DFlash2 草稿池 ring + NC 注意力 ring 槽
ab1bc74 test(spec): TP2 head 门覆盖 spec m=7/8 与 W8A8 质量门
```

提交顺序上，`34a2350` 的 `draft.rs` ring 引用到 `10898e8` 的
`StatePool` 方法；**最终 HEAD 是完整可构建的**，但中间单提交不一定
逐提交 build（本轮未逐提交验证）。下一个会话以 HEAD 为准。

本提交之后的 docs 提交包含本文与
`tp2-decode-fast` / `optimization-candidates` / `w4a8-family` /
`operator-matrix` / `device-command-ledger` / `profiles/README`。

## 8. 坑清单（已踩过）

1. **W8A16 TC 的 shared 是 static**：driver `smem` 必须写 0；
   static+dynamic 双算会 `CUDA_ERROR_INVALID_VALUE`。
2. **`tp_head_i8_gate` 的 head plan 与 `EngineKnobs.head_quant` 必须同源**：
   只传 loader plan、knob 留 `Off`，tag filter 会拒 `quant.int8` 算子。
3. **`OWL_TP_HEAD_OUT` 是单臂路径**；比较两臂不要设置，否则读同一个文件。
4. **`compare_a8` 按设计红**（0.6962）；不要为了变绿改阈值。
5. **W8A16 TC 不是 int8 TC**：30 系只有 int8 IMMA，没有 fp8 TC；
   W8A16 必须走 f16/bf16 MMA + 在线反量化。
6. **`MARLIN_NO_M8` 是旧诊断 env**；新路径用
   `[kernel] marlin_w4a8_m8`，默认 false。
7. **`build.rs` marlin 源登记**：`rerun-if-changed` 必须在 stamp 命中
   分支之外，否则改 `marlin_host.cu` 会静默用旧 `.a`。
8. **`spec.verify.read` 含整栈同步**；不要把它当纯 D2H 或 host 开销。
9. **std 单次 decode t/s 方差很大**；同 prompt、同 AL 的受控 round
   才是判据；跨轮 std 均值不要直接对比。
10. **`tp_draft=true` 是 TP2 spec 硬门**；缺它会在装载后拒启。

## 9. 当前环境

- 分支 `main`，未 push。
- 6 个代码提交 + 本文所在 docs 提交。
- 无 owl server 在跑；`nvidia-smi` 无 compute apps。
- 3080 对 = CUDA ordinal **2/3**（BDF `C1:00.0` / `01:00.0` 视 boot；
  以 `nvidia-smi --query-gpu=index,pci.bus_id` 为准）。
- 3090 对 = ordinal 0/1；GPU1 可能被其他会话占用，起 owl 前先查。
- 模型：`models/cyankiwi/Qwen3.8-27B-AWQ-INT4`，
  draft `models/syvai/Qwen3.8-27B-DFlash2-W4A16`。

## 10. 交接判据

下一个会话如果只做一件事：

> **先确认默认 W8A16 TC 的 32K fixed-tail 与采样档质量门；
> 然后评估 DFlash2 K=1..15 单 block（不是两段拼接）。**

如果只做性能：

> **不要再磨 head**；去 verify 的 Marlin m8 tile / GEMM 合并 / GDN
> 小核融合，或评估 proposal_q + sparse accept 对采样档的收益。

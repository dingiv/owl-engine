# INT8 KV × Attention 专项设计（2026-11-03）

> 状态：**P8 生产态已落地（2026-11-03 晚，见 §0.20）**；
> 早期 P24/P32/f16-PV 章节保留为历史排查记录，结论以 §0.20 为准。
> 目标不是“给 owl 加一个 int8 缓存字段”，而是**新增一条 INT8 KV 的端到端
> Attention 线路**：缓存格式、量化写入、scale 生命周期、attention 后端、
> planner、workspace、CUDA graph、前缀缓存、TP2，全部成套。
>
> 本文只定方向、接口、调用链、里程碑和 kill criteria；具体实测数据见
> `docs/bench/int8-kv-research-2026-11-03.md` 与
> `docs/bench/int8-pv-s1-2026-11-03.md`（工作区根 docs）。

---

## 0. 结论摘要

1. **两条主线并行立项**：
   - **主线 B（先走）**：**pth-flash-attn 接入 + 性能优化**。
     pth 已经有完整的 int8 KV attention 技术栈（int8 cache + per-token/head
     scale + paged/GQA/causal/split-KV/decode/prefill/CUDA graph + 纯 C ABI），
     且不依赖 PyTorch 就能调用。我们先把它接进 owl，再针对 D256 长 prefill
     做优化。
   - **主线 A（同步设计，后施工）**：**爆改 FlashInfer，单独开一条 int8 线路**。
     先做 A1（int8 存储 + 反量化 + 现有 FP16-PV），再做 A2（full int8 QK+PV）。
     A1 与 pth 共用同一个 `int8_per_token_head` 缓存格式，便于 A/B；
     A2 需要第二套 V scale（沿 K 恒定），不共用缓存。
2. **共同底座 P0 必须先做**：新增 `int8_per_token_head` KV cache 格式 +
   量化写入 kernel + scale 生命周期 + 内存规划。没有这个，两条主线都无法
   A/B，也无法回答“int8 到底比 fp8 强在哪”。
3. **不把 ninfer 当底座**：ninfer 的模型/引擎/布局与我们不同，改造成本大；
   只作为算法与 kernel 参考（QK int8、G64 scale、V 反量化、decode 分片）。
4. **不再做 int8 QK-only**：FlashAmpere、pth、ninfer 已经三次负结果；
   本专项只做“完整 int8 KV 链路”，不做“只换 QK 的局部补丁”。
5. **性能判决线**（沿用上一轮用户裁决）：
   - pth / FI-A1：先追平当前 `fp8 + FP16-PV`（T=2048/65k ≈ 50.5 ms），
     再谈超越；若只能到 55 ms 以上，只能按“质量/容量”项目保留，不按速度项目。
   - FI-A2 full int8：T=2048/65k **≤30.5 ms** 且 cos ≥0.9999；否则停止。
6. **默认不改变生产档**：新线路由显式配置键开启，默认仍是 f16/fp8 +
   FP16-PV；任何 int8 路径都不能影响现有 f16/fp8 的逐字行为。

### 0.1 两条主线一眼对照

| 维度 | 主线 B：pth-flash-attn | 主线 A1：FI-int8-storage | 主线 A2：FI-int8-full |
|---|---|---|---|
| 技术栈完整度 | **完整**（cache/scale/paged/GQA/split/graph/C ABI） | 需新增 FI int8 模板/loader/scale | 需新增 QK+PV int8/P 量化/V scale |
| 计算方式 | int8→fp16 反量化 + f16 MMA | int8→fp16 反量化 + f16 MMA + FP16-PV | **int8 张量核 QK+PV** |
| 性能天花板 | 接近 f16 KV | 接近 pth / fp8+FP16PV | 理论 2× PV（唯一可能超 f16+FP16PV） |
| 风险 | 中（集成/planner/缓存写入） | 中高（FI typed 模板侵入） | **高**（P 量化 O(L²)；S1 已 NO-GO 一次） |
| 缓存格式 | `int8_per_token_head`（§3） | 同 §3（可与 pth 共用） | `int8_g64_pv`（V scale 沿 K 恒定） |
| 先行判据 | 先复测 pth vs FI，再优化 FP16-PV | 与 pth 同缓存 A/B | 单路 QK+softmax <30.5ms 才继续 |
| 本项目优先级 | **P1（先走）** | P2（同步设计） | P3（只在 A1 有空间时打） |

---

## 0.2 2026-11-03 基线修正(双 half 档之后)

原设计里的 30.5ms 门是拿 **2026-11-01 的 FI f16+FP16-PV(38.3ms)** 算的。
`fi-fp16acc` 合流后,基线已经变了(3090 Ti,hd256,hkv4,page32,T=2048):

| 档 | ctx=65k | ctx=139k |
|---|---:|---:|
| FI 双 half(f16 池) | **32.98 ms** | **73.4 ms** |
| FI 双 half(fp8 池) | **42.89 ms** | 90.8 ms |
| pth int8(2026-11-01 实测) | 58.57 ms | 126.30 ms |

⇒ 两条修正:

1. **30.5ms 已不是“反超”门槛**,只是与双 half 档打平。新门槛 = **T=2048/65k ≤26ms**
   (相对 32.98ms ≥20%),冲刺 ≤24ms;ctx=139k 目标 ≤58ms(相对 73.4ms)。
   cos ≥0.9999 不变。
2. **pth 只做 P0/缓存/planner 参考,不作为性能主线**。pth 当前 prefill 比双 half FI
   慢 78%,即使接进 owl 也先输 25ms;它的价值是完整的 int8 cache/scale/C ABI
   和 decode 路径。性能主线 = **FI-A2:在 FI 的 4-warp/tile 流水里做 full int8
   QK+PV**,A1(int8 存储 + f16 MMA)只作为正确性/缓存接线台阶。

P0 已开工(`packages/owl-worktrees/fi-int8`,分支 `fi-int8`):
`owl_reshape_and_cache_int8_knhd` + host oracle 已通过(逐字节 + scale 位型),
下一步是 FI-A2 的 kernel spike,不是继续接 pth。

## 1. 需求与边界

### 1.1 要解决的问题

owl 目前：

- 主池：**fp8 e4m3 KV**（1 byte/token/head，per-tensor scale，无 per-token
  scale 平面）；
- prefill：FI 0.7.2 + 本地 FP16-PV；
- decode：FI graph / XQA fp8；
- 长上下文质量受 fp8 e4m3 的 3-bit mantissa 限制；
- 现有 int8 讨论的“容量优势”不成立：int8 与 fp8 都是 1 byte；int8 还要多
  scale 平面（per-token/head 在 D256 约 +1.56%，G64 约 +6.25%）。

所以本专项的**真实价值**只有两个：

1. **同字节更高精度**：int8 per-token/head（或 G64）比 fp8 per-tensor 精度更高，
   用于长上下文质量；
2. **full int8 PV 的 2× 张量核**（仅 A2）：如果 P 量化税能被摊薄，才有
   prefill 速度收益。

### 1.2 目标（按优先级）

| 优先级 | 目标 | 判据 |
|---|---|---|
| P0 | int8 KV 缓存 + 写入 + scale 生命周期可用 | 与 pth/vLLM writer oracle 逐字节一致；prefix cache / snapshot 正确 |
| P1 | pth 后端接入 owl | D256/GQA24/4/page32/causal，T=1/8/2048，ctx 4k/65k 与 f32 参考 cos ≥0.9999 |
| P2 | pth 与 FI-A1 的 prefill 性能 A/B | 同缓存、同形状；至少追平 fp8+FP16-PV，目标 ≤38.3 ms（f16+FP16-PV） |
| P3 | FI-A2 full int8 PV | T=2048/65k ≤30.5 ms，cos ≥0.9999；否则停 |
| P4 | 生产接线 | 配置键、graph、prefix cache、3090 e2e、3080 TP2 |

### 1.3 非目标

- 不替换 f16/fp8 默认路径；
- 不做“只改 QK 的 int8”局部补丁；
- 不做运行时 env 开关（配置键 + 构建 feature 是唯一入口）；
- 不在本专项里做 INT4 KV / TurboQuant / rk8v4；
- 不改动 ninfer 作为底座。

---

## 2. 两条主线与备选

### 2.1 主线 B：pth-flash-attn（先走）

pth-flash-attn 0.1.0（Apache-2.0，commit `b0d4f7ee`）已经具备：

- **KV cache**：int8 `[blocks, block_size, kv_heads, head_dim]` +
  `k_scale/v_scale` fp32 `[blocks, block_size, kv_heads]`，per-token/head；
- **attention**：D64/128/256，GQA（G≤64），paged，causal，sliding window
  （D128），split-KV，decode / verify / prefill，CUDA graph；
- **数值**：int8 K/V 在寄存器里 exact 转 fp16，Q fp16 + per-row 2^k prescale，
  全部 fp16×fp16→fp32 MMA；
- **接口**：`pth_attn_attention` / `pth_attn_combine` 的纯 C ABI
  （`attn_args.h`），不依赖 PyTorch；vLLM 插件只是 Python glue；
- **测试**：fp64 reference + per-element error bound、simulator、GPU grid、
  0 spill/255 reg 构建门。

**它不是什么**：

- 不是 int8 张量核；K/V 先反量化成 fp16，再 f16 MMA；
- 不是 owl 的现成后端：缓存写入、planner、workspace、graph、prefix cache
  都要我们接；
- 不是无代价：`2D+8` 字节/token/head，D256 比 fp8 多 1.56%，池少约 1.5%。

**为什么先走它**：

- 技术栈完整，C ABI 可被 Rust 直接调用；
- 与 FI-A1 可以共用同一个缓存格式，A/B 成本最低；
- 它的 decode 分片/prefill 分块/planner 都是现成的工程答案；
- 即使 prefill 不赚，它也能作为“int8 质量/长上下文”后端保留。

### 2.2 主线 A：FlashInfer int8 专用线路（设计同步，后施工）

FI 0.7.2 当前只认 f16/fp8/fp4 KV；要接 int8，必须新增一条**独立 TU/模板
分支**，不能把 int8 塞进现有 f16/fp8 实例化。

分两级：

- **A1：FI-int8-storage**
  - `DTypeKV=int8_t`，K/V 反量化到 f16；
  - K/V scale 用 pth 的 per-token/head 格式（与主线 B 共用缓存）；
  - PV 继续用现有 FP16-PV（`o_acc` half）；
  - 预期：与 pth 同量级；可能因 FI 的 tile/pipeline 更好而略快；
  - 风险：FI 0.7.2 的 `compute_qk`/`compute_sfm_v` 是 f16/fp8 typed，
    要加 int8 loader、scale 读取、反量化；不能影响现有 f16/fp8 路径。

- **A2：FI-int8-full**
  - QK int8（m16n8k32.s8）+ PV int8（m16n8k32.u8.s8 或 s8.s8）；
  - P uint8，per-row/64-key tile scale；
  - V int8，**per-channel / per-64-dim scale，沿 K 恒定**；
  - 每个 tile 的 int32 PV 结果反量化到 float O；
  - 预期：唯一可能 2× PV 的路线；FI 的 `compute_sfm_v` 由**所有 warp**
    执行，P 量化可以摊到全部 warp，正好避开 ninfer 的 4-producer 瓶颈；
  - 风险：P 量化仍是 O(L²)；S1（ninfer scaffold）实测 full int8 53.7ms，
    QK+softmax 单路 39.5ms 已超 30.5ms；FI 版必须证明其 warp 结构能把
    P 量化税压下去，否则停。

### 2.3 备选（简要，不立项）

| 方案 | 结论 |
|---|---|
| **ninfer-3090** | 只作参考：QK int8 + V f16，prefill 比 BF16 慢 20~27%，decode W=1/4 快 27~36%；模型/引擎/布局不同，改造成本大 |
| **LMDeploy TurboMind** | int8 KV 只做 decode；prefill 注册表只实例化 half/bf16；不满足 prefill 主线 |
| **vLLM Triton `int8_per_token_head`** | 完整链路，但 Triton + 反量化到 Q dtype，不是 int8 张量核；性能不是 FI 对手 |
| **TRT-LLM FMHA v2** | full int8 QK+PV 算法参考；per-tensor scale、BERT 类 QKV、NVIDIA 专有许可；不能直接复制 |
| **XQA** | int8/fp8 KV 只 decode/verify，反量化到 f16；owl 已有，不作为 prefill 起点 |
| **pth-flash-attn** | 主线 B，见 §4 |

---

## 3. 共同底座 P0：int8_per_token_head 缓存

两条主线（pth 与 FI-A1）共用这一套缓存；A2 需要第二套 V scale，见 §5.3。

### 3.1 数据格式

逻辑视图（pth 的 C ABI 已支持任意 byte stride）：

```text
k_cache: [num_blocks, block_size, num_kv_heads, head_dim] int8
v_cache: [num_blocks, block_size, num_kv_heads, head_dim] int8
k_scale: [num_blocks, block_size, num_kv_heads] fp32
v_scale: [num_blocks, block_size, num_kv_heads] fp32
```

owl 的 page-major 布局可以表达为：

```text
block_size = 32 (owl 现有 page)
num_blocks = pages
k_cache page = [slot 0..31, kv_head 0..3, d 0..255] int8
k_scale page = [slot 0..31, kv_head 0..3] fp32
```

pth 的 `PthAttnStream` 用 byte stride 描述，允许 NHD/HND/padded page；
owl 只需给出 `stride_block/slot/head`，不需要改 pth 的 kernel。

字节账（D=256）：

```text
fp8:   K 256 + V 256                    = 512 B/token/head
int8:  K 256 + V 256 + ks 4 + vs 4      = 520 B/token/head  (+1.56%)
```

### 3.2 量化写入约定（必须与 pth/vLLM 逐字节一致）

每个 `(token, kv_head)` 行，float32：

```text
scale = max(max|x| / 127, 1e-6)
q = x * (1/scale)
q = q >= 0 ? q + 0.5 : q - 0.5
q = clamp(q, -128, 127)
store int8(q)   // 向零截断
```

K 和 V 各自一行一 scale；scale 存 fp32。这个约定是 pth 的 `integration.md`
§3 和 vLLM 的 writer 镜像，必须用 pth 的 Python oracle 做逐字节对拍。

### 3.3 写入 kernel

新增 owl 算子族 `int8_kv_write`（暂名）：

- 输入：K/V（模型 dtype，bf16/f16）、slot_mapping、block_table；
- 输出：int8 K/V cache + fp32 K/V scale cache；
- 每个 `(token, kv_head)` 一个 work unit，行内 absmax/127，RNE/round-half-away
  按 §3.2；与 pth 的 `quantize_rows` 逐字节一致；
- 与 owl 现有 `reshape_and_cache` 的 page 分配/前缀缓存/snapshot 生命周期对齐；
- 支持 slot < 0（padding token 跳过）。

### 3.4 生命周期 / 前缀缓存 / snapshot

int8 缓存不是“只换 data plane”：

- 前缀缓存的身份必须包含 **data + scale 两个平面**；
- snapshot/eviction/页回收必须同时处理 data/scale；
- 页哈希/内容校验要覆盖 scale（否则 scale 陈旧、data 新，长上下文会漂）；
- host 腿/swap 的字节账要按 `2D+8` 而不是 `2D`；
- graph plan 的 workspace 不走 cache，单独算。

### 3.5 配置与内存规划

新增显式配置（名称待定，建议）：

```toml
[kernel]
kv_format = "fp8_per_tensor"   # 默认；可选 "int8_per_token_head"
int8_backend = "pth"           # "pth" | "fi_storage" | "fi_full"（实验）
```

- 不用环境变量；`--check` 必须显示解析后的 `kv_format` / `int8_backend`；
- `MemoryPlan` 增加 scale plane 字节；D256 约 +1.56%；
- 默认 fp8，不改变现有 profile；
- int8 profile 显式写 `kv_format`，避免静默切换。

---

## 4. 主线 B：pth-flash-attn 接入与优化

### 4.1 接入方式：走纯 C ABI，不走 Python/torch

pth 的 `csrc/pth_attn/attn_capi.h` 暴露：

```c
int32_t pth_attn_attention(const PthAttnArgs* args, void* stream);
int32_t pth_attn_combine(const PthAttnCombineArgs* args, void* stream);
```

`PthAttnArgs` 只含指针/stride/标量，不含 torch 类型；`attn_args.h` 是纯 C
结构。**owl 不使用 pth 的 Python 包、vLLM 插件、torch stable ABI**，而是：

1. 把 `csrc/pth_attn` 的必要源文件 vendor 到
   `crates/kernels/cu/pth_attn/`（Apache-2.0，保留 NOTICE/版权头）；
2. `build.rs` 用 nvcc 编译 `attn_capi.cu` + `instantiate_d256.cu` +
   `attn_combine.cu` + launcher，产出 `libowl_pth_attn.a`；
3. Rust 侧用 `extern "C"` 绑定 `pth_attn_attention` / `pth_attn_combine`，
   传 `PthAttnArgs`；`#[repr(C)]` 镜像必须带 `abi_version` / `struct_bytes`
   并在调用前校验，另加 `size_of` / `offset_of` 单测，避免 ABI 漂移；
4. 构建门沿用 pth 的 0 spill / ≤255 reg / SASS lint；owl 只编 sm86。

> 这样既不引 PyTorch，也不把 pth 的 vLLM 后端搬进 owl；pth 只作为
> attention kernel 库。

### 4.2 缓存 / 元数据映射

pth 的 `PthAttnArgs` 需要：

| 字段 | owl 来源 |
|---|---|
| `q/out` | prefill chunk / decode 的激活，`[T,Hq,D]` f16 |
| `k_cache/v_cache` | §3 的 int8 page-major 视图 |
| `k_scale/v_scale` | §3 的 fp32 scale 视图 |
| `block_table` | owl 的页表（int32，kernel-block id） |
| `seq_lens` | 每请求 KV 长度（含本 chunk） |
| `query_start_loc` | 本 slice 的 query 前缀和 |
| `max_query_len` | host 上界，只选 tile |
| `bq` / `num_q_tiles` / `num_splits` | 由 owl 的 planner 给出 |
| `ws_o/ws_lse` | owl 预留的 split workspace |
| `window/causal` | owl 因果/窗口语义；D256 只支持 causal、window=0 |

pth 的 op **只读** cache，不写；写入由 owl 的 `int8_kv_write` 负责。
pth 的 block_size 可以是 32（owl page），但 pth 的 vLLM 后端要求 16 的倍数；
32 满足。pth 的 `PthAttnStream` stride 允许 page-major；owl 只需暴露
`stride_block/slot/head`。

### 4.3 Planner：Rust 端口 + 对拍

pth 的 `python/pth_flash_attn/planner.py` 是纯算术，不读序列长度：

- `plan_launch(...)` 返回 `wm`、`bq`、`q_tiles_upper`、`num_splits`；
- `workspace_token_splits(...)` / `workspace_shapes(...)` / `workspace_bytes(...)`；
- `split_tile_range(...)` 与 kernel 的分片公式一致。

owl 不能在运行时跑 Python；在 `crates/kernels/src/.../pth_planner.rs` 做
**纯 Rust 端口**，并用 pth 的 Python planner 输出做 golden 对拍（CPU，
不需要 GPU）。至少覆盖：

- prefill：T=2048/8192，G=6，Hkv=4，D=256，num_sms=84/68；
- decode：T=1/4/8，ctx 8k/32k/128k；
- workspace 形状/字节；
- `wm`/`bq`/`num_splits`/`split_tile_range` 与 Python 完全相等。

### 4.4 workspace / graph

- pth 需要 `ws_o` float32 `[S,T,Hq,D]` + `ws_lse` float32 `[S,T,Hq]`；
  `S` 由 planner 给出，decode 可能到 ~34，prefill S=1；
- 在 owl 的 `MemoryPlan` 里预留（模型装载期），不能按需分配；
- pth 的 op 不分配、不同步、不读 host 设备数据，可进 CUDA graph；
- `num_splits>1` 时先 `pth_attn_attention` 再 `pth_attn_combine`，两条
  launch 都要进图；owl 的图捕获/回放要包住它们；
- pth 的 `seq_lens/query_start_loc` 从设备读，graph 回放改变长度安全；
- 这与 owl 的 `fi_plan_sync` 不同：pth 没有 plan 表，只有 workspace 指针。

### 4.5 性能现状与优化点

pth 自己的公开数据（RTX 3080 20GB，D256，12q/2kv，TP2 shape）：

| 形状 | pth int8 | Triton int8 | 备注 |
|---|---:|---:|---|
| q=1, KV 8k | 25.0 µs | 97.6 µs | 3.9× |
| q=1, KV 32k | 63.5 µs | 373.9 µs | 5.9× |
| q=1, KV 128k | 209.8 µs | 1464 µs | 7.0× |
| q=4, KV 128k | 247.2 µs | 15561 µs | 62.9× |
| q=8192 prefill, no prefix | 10.46 ms | 41.17 ms | 3.9× |

但上一轮 owl 的实测（handoff）是：**pth prefill 比 FI f16 慢 15~22%，
decode 快 3~10%，e2e <1%**。两者不矛盾：pth 的公开数据是对 Triton int8，
不是对 FI f16；owl 的 FI f16/fp8 本身很强。**所以 P1 的第一件事不是优化，
而是同形状 head-to-head 复测**：

- T=2048/65k、T=8192/65k、T=512/65k；
- decode T=1/8/65k，graph 开/关；
- 同缓存、同页表、同 workspace，pth vs FI f16/fp8/FP16-PV。

只有在复测确认 pth 的 prefill 差距后，才进入优化。优化顺序：

1. **PV 改 FP16 累加**（最高优先）：
   pth 的 `pv_subtile` 现在是 `float o[D/8][4]` fp32 累加；照我们 FI
   FP16-PV 的做法，改成 half-packed `o_acc` + `mma.f16f16f16`，KV loop
   结束后 materialize 回 float。D256 的 PV 寄存器/带宽压力最大，预计
   +15~25% PV；同时可能把 prefill 的 32-token sub-tile 放宽到 64。
   必须过 pth 的 0 spill / 255 reg 门。
2. **QK int8（可选）**：
   pth 现在把 int8 K 反量化成 fp16 再 f16 MMA；如果 PV 改 half 后仍有
   余量，可在 `qk_subtile` 加 m16n8k32.s8 分支（参考 FlashAmpere /
   ninfer），但 QK-only 历史负结果，必须和 FP16-PV 组合测，不能单开。
3. **Tile / pipeline**：
   pth D256 是 16-row tile、64-token load、2 stages、4 warps、SUB=32；
   评估 8 warps / 128-row tile、3 stages、SUB=64 的收益，但先过 smem/reg
   门。不要盲目改，pth 的 geometry 有 static_assert 保护。
4. **Dequant / loader**：
   pth 的 `cvt4_s8_f16x2` 已是 magic-number 4-wide；可评估 `cvt`/`prmt`
   变体，但优先级低于 PV。
5. **Planner / split**：
   用 owl 的真实 shape 扫 `wm`、`bq`、`num_splits`；prefill S=1，decode
   split 数按 3090/3080 的 SM 数与 Hkv 调。
6. **GQA / L2**：
   pth 的 grid 是 `(q_tiles, Hkv, splits)`；G=6 时同一 KV head 的 6 个
   q_tile 会重复读 K/V；用 L2 persistence / 调度顺序优化，但不要改
   cache 语义。

### 4.6 主线 B 的 Go/No-Go

- **B0**：C ABI 接入 + planner 对拍 + f32 参考 cos ≥0.9999 → 必须过。
- **B1**：pth 原版 head-to-head：
  - prefill T=2048/65k **≤55 ms**（至少不显著输给 fp8+FP16-PV 50.5）；
  - decode T=1/8/65k 不低于当前 XQA fp8 的 95%；
  - 若 prefill >55 ms 且 decode 无优势 → 只保留“质量/长上下文”定位，
    不按速度项目继续。
- **B2**：FP16-PV 优化后：
  - prefill **≤50.5 ms**（追平 fp8+FP16-PV）；
  - 目标 **≤38.3 ms**（追平 f16+FP16-PV）。
- **B3**：e2e 60k ≥ 当前 fp8+FP16-PV 的 +0%（先不输），
  needle/4-seed 引文一致；否则停。

---

## 5. 主线 A：FlashInfer int8 专用线路

### 5.1 原则

- **不改现有 f16/fp8 路径**：新增 `owl_fi_attention_int8.cu` /
  `owl_fi_attention_int8_common.h`，单独 ABI 符号；
- 复用 FI 的 plan/页表/图档/前缀缓存/CTA 调度，但 int8 的 loader、
  scale、PV 累加器走新分支；
- 先 A1（存储/反量化），再 A2（full int8）；A1 与 pth 共用 §3 缓存，
  A2 用第二套 V scale。

### 5.2 A1：FI-int8-storage

- `DTypeKV=int8_t`，`DTypeQ=half`，`DTypeO=half`；
- Params 增加 `k_scale/v_scale` 指针（per-token/head fp32）+ stride；
- `compute_qk`：
  - 读 int8 K，`cvt` 到 f16（或直接 `cvt.rn.f16.s8` 4-wide），乘 `k_scale`；
  - 继续 f16 MMA；
- `compute_sfm_v`：
  - 读 int8 V，反量化到 f16，乘 `v_scale`；
  - PV 继续 FP16-PV（`o_acc` half）；
- 预期：与 pth 同量级；FI 的 tile/pipeline 可能更好；仍需实测；
- 风险：FI 0.7.2 的 `produce_kv` / `ldmatrix` 对 int8 的 swizzle/宽度
  与 f16 不同；要按 FlashAmpere 的 int8-QK 经验处理 K 的 b128 swizzle、
  Q 的 `load_q_global_smem` dtype-aware 计数、causal mask 的 5e4 问题。

### 5.3 A2：FI-int8-full

- QK：`m16n8k32.s8`，Q int8 per-row/G64，K int8 per-token/G64；
- P：uint8 per-row/64-key tile，`mma.m16n8k32.u8.s8.s32`；
- V：int8 per-**channel/64-dim** scale，沿 K 恒定（这是 A2 的缓存格式，
  与 §3 的 per-token V scale 不兼容）；
- PV：int32 累加 → 每 tile 反量化到 float O（`p_scale * v_scale`）；
- P 量化在 `compute_sfm_v` 里由**所有 warp**执行（FI 的天然优势）；
- 先 standalone FI probe（不接 owl 缓存），形状 T=2048/65k；
- **S1 已证明 ninfer scaffold 的 full int8 53.7ms / QK+softmax 39.5ms**；
  FI 版必须证明：
  - QK+softmax 单路 <30.5ms；
  - full <30.5ms；
  - cos ≥0.9999；
  - 否则按 kill criteria 停止，不进入 owl。

### 5.4 FI 路线的 Go/No-Go

- **A1**：
  - 与 pth 同缓存、同形状；prefill 至少 ≤55ms，目标 ≤50.5ms；
  - 若 A1 明显快于 pth，选 A1 作为 int8 读取后端；pth 保留 decode/参考。
- **A2**：
  - 单路 QK+softmax <30.5ms；
  - full ≤30.5ms + cos ≥0.9999；
  - 否则停止 A2，回到 A1/pth。

---

## 6. 统一接口与调用链

### 6.1 Rust 侧接口

不要把两套后端硬编码进 engine；用一个小枚举 + trait：

```rust
pub enum Int8KvBackend { Pth, FiStorage, FiFull }

pub struct Int8KvPlan {
    pub wm: u32,
    pub bq: u32,
    pub num_q_tiles: u32,
    pub num_splits: u32,
    pub ws_o_bytes: usize,
    pub ws_lse_bytes: usize,
}

pub trait Int8KvAttention {
    fn format(&self) -> KvFormat;
    fn workspace_bytes(&self, cfg: &PlanCfg) -> Result<usize>;
    fn plan(&self, meta: &BatchMeta) -> Result<Int8KvPlan>;
    fn launch(&self, plan: &Int8KvPlan, args: &LaunchArgs) -> Result<()>;
}
```

- `format` 决定 cache 视图（pth/FI-A1 共用 `Int8PerTokenHead`；
  FI-A2 用 `Int8G64Pv`）；
- `plan` 由 pth planner 端口 / FI plan 产生；
- `launch` 分别调 pth C ABI / FI int8 符号；
- engine 只依赖 trait，不感知 pth/FI 的差异。

### 6.2 调用链（prefill）

```text
engine prefill chunk
  -> kv_format == int8_per_token_head ?
       -> int8_kv_write(K/V, slot_mapping)       # data + scales
       -> Int8KvAttention::plan(meta)
       -> Int8KvAttention::launch(plan, args)     # pth / FI-A1 / FI-A2
       -> (num_splits > 1) combine
  -> else -> 现有 FI fp8/f16 + FP16-PV
```

### 6.3 调用链（decode / verify）

```text
engine decode step
  -> int8_kv_write(new tokens)
  -> plan(meta)                                  # pth split-KV / FI graph
  -> launch
  -> (split > 1) combine
```

### 6.4 配置

```toml
[kernel]
kv_format = "int8_per_token_head"   # 默认 fp8_per_tensor
int8_backend = "pth"                # pth | fi_storage | fi_full
```

- `--check` 显示解析结果；
- 构建 feature：`int8-kv-pth` / `int8-kv-fi`，默认关；不允许 env；
- 默认 profile 不变；int8 profile 显式写全。

---

## 7. 测试与验收

### 7.1 单元 / oracle

- writer：与 pth `quantize_rows` / vLLM writer 逐字节一致（含 round-half-away、
  1e-6 floor、clamp）。
- planner：Rust 端口 vs pth Python planner 的 `wm/bq/num_splits/ws shapes/
  split_tile_range` 全等。
- cache 生命周期：prefix cache / snapshot / eviction 覆盖 data+scale；
  scale 不陈旧。
- 内存：`MemoryPlan` 的 scale plane 字节与实测一致（D256 +1.56%）。

### 7.2 kernel parity

- pth：D256/GQA24/4/page32/causal，T=1/8/2048，ctx 4k/65k，vs f32 参考
  cos ≥0.9999；与 pth 自带 reference 的 per-element bound 对比。
- FI-A1：同形状，vs pth 输出/ f32 参考。
- FI-A2：vs f32 参考 cos ≥0.9999；P/V 量化误差单独记录。

### 7.3 性能 head-to-head

同一 3090 Ti / 3080 对，同缓存、同页表、同 workspace：

| 形状 | 基线 | pth | FI-A1 | FI-A2 |
|---|---|---|---|---|
| T=2048/65k | fp8+FP16PV 50.5 / f16+FP16PV 38.3 | ? | ? | ? |
| T=8192/65k | 待测 | ? | ? | ? |
| T=512/65k | 待测 | ? | ? | ? |
| decode T=1/65k | XQA fp8 0.165ms/层 | ? | ? | — |
| decode T=8/65k | XQA fp8 | ? | ? | — |

- 每个数字至少 3 次，取 median；同时记录 REG/STACK/SMEM 和 graph 开/关。
- 先复测 pth 原版，再谈优化；不要拿 pth 公开的“vs Triton”数字当“vs FI”。

### 7.4 e2e / 质量

- 60k speedtest（shadow on/off）：prefill t/s、TTFT；
- 4-seed 引文复制：逐字节一致；
- needle 59.7k：正确；
- GSM8K n=200：int8 vs fp8 的 Δ 在 ±1pp 内（参考 27B 质量门）；
- 3080 TP2：262k 池、零 Xid、长 ctx 正确。

---

## 8. 里程碑与 kill criteria

| 里程碑 | 内容 | 时间盒 | Go |
|---|---|---|---|
| **M0** | 设计冻结；§3 cache/writer/planner oracle | 2 天 | writer/planner 对拍全绿 |
| **M1** | pth C ABI 接入 + f32 parity | 3 天 | cos ≥0.9999，无非法访存 |
| **M2** | pth 原版 head-to-head | 2 天 | prefill ≤55ms；否则仅质量定位 |
| **M3** | pth FP16-PV 优化 | 3 天 | prefill ≤50.5ms；目标 ≤38.3ms |
| **M4** | FI-A1（int8 storage） | 4 天 | 与 pth 同量级或更快 |
| **M5** | FI-A2（full int8）S1 | 3 天 | QK+softmax + full 均 ≤30.5ms，cos ≥0.9999 |
| **M6** | owl 生产接线（graph/prefix/3090 e2e） | 3 天 | e2e 不输当前 fp8+FP16PV |
| **M7** | 3080 TP2 / 262k | 2 天 | 零 Xid、长 ctx 正确 |

**Kill criteria**：

- M0 writer/planner 不一致 → 不进入 M1；
- M1 parity 失败 → 停 pth；
- M2 pth prefill >55ms 且 decode 无优势 → pth 仅保留质量定位，不按速度继续；
- M3 优化后仍 >50.5ms → 不把 pth 作为 prefill 速度后端；
- M4 FI-A1 不快于 pth 且不快于 fp8+FP16PV → 不选 FI-A1；
- M5 FI-A2 单路 QK+softmax ≥30.5ms → 停 A2（S1 已有 39.5ms 前科）；
- M6 e2e 输给当前 fp8+FP16PV → 不默认开，只保留 opt-in。

---

## 9. 风险

1. **pth 的 prefill 历史负结果**：上一轮 owl 实测 pth prefill 比 FI f16 慢
   15~22%；pth 公开数据是对 Triton，不是对 FI。M2 必须先复测，不能假设
   “优化一下就能赢”。
2. **pth 是反量化到 fp16，不是 int8 张量核**：它的性能天花板接近 f16 KV；
   真正的 2× 只在 FI-A2。
3. **pth 的 scale 内存开销**：D256 +1.56%，池少约 1.5%；262k 档要重算
   max_seq。
4. **pth 的 C ABI/planner 版本锁定**：vendor 固定 commit；升级要重跑
   planner oracle 和 GPU parity。
5. **FI int8 模板侵入性**：0.7.2 的 typed kernel 需要新 TU；不能污染
   f16/fp8 路径；要保留 FP16-PV 的逐字行为。
6. **FI-A2 的 P 量化税**：S1 已证明 ninfer scaffold 的 QK+softmax 单路
   39.5ms；FI 的全 warp `compute_sfm_v` 是唯一希望，但未验证。
7. **prefix cache/snapshot**：scale 平面必须参与身份/驱逐/回灌；否则长
   上下文会出现“data 新、scale 旧”的静默错误。
8. **graph**：pth 的 combine 是第二条 launch；owl 的图捕获/回放要包住
   两条，且 workspace 指针固定。
9. **TP2**：pth/FI 都只做单卡 attention；owl 的 TP all-reduce 在外部，
   per-rank Hq/Hkv 映射必须与 cache writer 一致。

---

## 10. 代码落点

| 模块 | 文件/动作 |
|---|---|
| 缓存格式/写入 | `crates/kernels/cu/pth_attn/`（vendor）+ 新 `int8_kv_write` TU |
| pth C ABI | `crates/kernels/cu/pth_attn/attn_capi.cu` + Rust FFI |
| pth planner | `crates/kernels/src/.../pth_planner.rs`（Rust 端口） |
| FI-A1 | 新 `crates/kernels/cu/flashinfer/owl_fi_attention_int8.cu` + 共享头 |
| FI-A2 | 新 `owl_fi_attention_int8full.cu` + `prefill_int8.cuh` 变体 |
| 配置 | `crates/engine/src/kernels/kernel_cfg.rs` / `profiles/*` 新增 `kv_format`、`int8_backend` |
| 内存 | `MemoryPlan` 增加 scale planes / pth workspace |
| 图 | `exec/phases.rs` / `running.rs` 包 pth attention+combine、FI int8 launch |
| 测试 | `crates/kernels/tests/` 新增 int8 probe；pth oracle/Python planner 对拍 |

**归属**：pth 源文件保留 Apache-2.0 头注/NOTICE；ninfer 只作参考，不 vendor；
FI 的 int8 改动保留本地 patch 记录（`patches/`）和 `docs/devlog/kernels/`。

---

## 11. 开放问题

1. `kv_format` 的配置键放 `[kernel]` 还是 `[pool]`？建议 `[kernel]`（与
   attention 绑定），但需要 profile 兼容性检查。
2. pth 的 `block_size=32` 是否与 pth 的 vLLM 后端假设冲突？pth 的 op 支持
   ≥1，但 vLLM 插件要求 16 的倍数；owl 不走 vLLM 插件，32 可用。
3. pth 的 `num_splits` 在 3090/3080 上如何选？先按 pth planner 的 `s_alone`
   公式，再用 owl 实测扫。
4. FI-A2 的 V scale 用 per-channel 还是 per-64-dim group？先 per-64-dim
   group（与 ninfer 一致），精度/寄存器更平衡。
5. int8 是否要同时替换 decode？pth decode 可能优于 FI/XQA，但 owl 的 XQA
   fp8 已经很快；先测，不预设。

---

## 12. 一句话

**先把 pth 的完整 int8 KV 栈接进来，验证“int8 缓存 + 成熟 attention”在 owl
形状上到底值多少；同时把 FI 的 int8 线路作为第二条读端，A1 共用缓存，
A2 只在 A1 证明有速度空间时才打。**

---

## 0.3 2026-11-03 S1 all-warp / QK-only 复测(重要负结果与唯一活口)

在 `fi-int8` 工作树的 `experiments/int8pv` 里把 S1 的 4 producer warp 改成
**16 warp 全部参与 QK**(4 Q tile × 4 KV slice,每 warp 16 key),P 仍按 64-key
tile 量化。3090 Ti,T=2048/ctx=65k:

| 档 | 中位 ms | 说明 |
|---|---:|---|
| S1 原始 full int8 | 54.1 | 4 producer + 12 consumer |
| all-warp full int8 | **43.1** | 16 warp 全 QK;correctness 不变(cos 0.999953) |
| all-warp QK+softmax only(NO_PV) | **29.8** | P 量化也跳过 |
| raw QK,无 softmax | 20.5 | 只 QK MMA + K load + scale |
| raw QK,无 scale | 16.0 | 只 QK MMA + K load |
| QK+softmax,**per-row/per-head scale** | **26.1** | 去掉 per-group scale 的 fma/load |
| **FI f16 QK+softmax**(临时跳过 PV MMA) | **28.1** | 同形状、同卡 |
| FI 双 half 全核 | **33.0** | 生产基线 |

判决:

1. **full int8 PV 仍死**:43.1ms,PV+P 量化 13.3ms,怎么调都过不了 33ms。
2. **per-group scale 是 S1 的额外税**:改成每 token/head 一个 scale 后,
   QK+softmax 29.8→26.1ms,已经略快于 FI 的 QK+softmax(28.1ms)。
   我们的 P0 writer 本来就是 per-token/head,所以这才是真实设计,不是近似。
3. **QK-only 是唯一活口**:26.1ms(QK+softmax)+ 约 5ms f16 PV ≈ **31ms**,
   对 33ms 只有 ~6% 优势,落在噪声边缘;必须把 softmax 再压到 ~5ms,
   或把 PV 压到 ~4ms,才有 ≥10% 的确定性。
4. S1 的 16-warp/128-reg 结构占满 64K 寄存器 ⇒ 1 CTA/SM,占用率差;
   这是它 QK 只有 ~40 TFLOPS 的原因。后续若继续,应走 **FI 的 4-warp
   pipeline + int8 QK**,而不是继续堆 S1 的 warp 数。

**下一步(未做)**:实现 `K int8(per-token/head) + Q int8(per-token/head) +
f16 V/FP16-PV` 的 QK-only 核:
- QK 用 `mma.m16n8k32.s8.s8.s32`,D 循环后每个 score 只乘一次 `q_scale*k_scale`;
- P 保持 half/float,**不量化成 uint8**;
- PV 用现有 f16 MMA,不做 int8 PV;
- 先测核级 T=2048/65k,门槛 ≤30ms(相对 FI 33ms ≥10%),cos ≥0.9999。
若过,再接 owl;若不过,INT8 KV 只保留质量/容量定位,速度主线停止。

---

## 0.4 2026-11-03 PV f16 微基准:S1 快修路线也关死

在 `experiments/int8pv/pv_f16_bench.cu` 用 **naive WMMA** 跑同形状 PV
(64x64 P × 64x256 V → 64x256 O,16 warps,grid=768,iters=1024):

| PV 实现 | 时间 |
|---|---:|
| naive WMMA f16 PV(仅 MMA,smem 常驻) | **54.8 ms** |
| S1 手写 int8 PV(含 P 量化/V load/scale) | 13.3 ms |
| FI 手写 FP16-PV(从 skip-PV 探针差值估) | ~5 ms |

结论:

- **不能假设 S1 换个 f16 PV 就只要 5ms**。naive WMMA 比 S1 的 int8 PV 还慢 4×;
  FI 的 5ms 是高度调优的 4-warp pipeline + FP16-PV,不是“换个 dtype”就有。
- 即使 QK+softmax 做到 26.1ms,加一个现实的手写 f16 PV(9~13ms)仍是
  **35~39ms > 33ms**。S1 这条线没有赢面。
- 唯一还剩的路 = **把 int8 QK 塞进 FI 的 4-warp pipeline**,复用 FI 的
  K/V load、softmax、FP16-PV 和 occupancy;这是大改,且 QK-only 的收益
  目前只看到 2ms(26.1 vs 28.1),风险高。

⇒ 建议:S1 实验线收口;若继续,只做 FI 端口,不再在 S1 上堆优化。

---

## 0.5 2026-11-03 4-warp FI-like 探针:**正面结果,路还活着**

S1 的 16-warp 结构占满 64K 寄存器(1 CTA/SM)。改成 **4 warp / 128 thread**
(FI 的 occupancy 形态)后,同形状 3090 Ti,T=2048/ctx=65k:

| 4-warp 探针 | 时间 |
|---|---:|
| raw int8 QK(无 scale/softmax/PV) | **10.35 ms** |
| int8 QK + softmax(无 scale,无 PV) | **11.65 ms** |
| f16 PV(手写 ldmatrix + m16n8k16,f16 acc;P/V smem 常驻) | **11.36 ms** |
| S1 16-warp QK+softmax(单 scale) | 26.1 ms |
| FI f16 QK+softmax | 28.1 ms |

关键结论:

1. **4-warp 形态把 int8 QK+softmax 从 26.1ms 打到 11.65ms**(2.2×)。
   之前的 16-warp 结果主要输在 occupancy,不是 int8 本身。
2. QK+softmax 11.65 + f16 PV 11.36 ≈ **23ms**(MMA 级,未含 V global load)。
   即使 V load/同步再吃 5~10ms,总时仍在 **28~33ms**,有真实赢面。
3. 下一步 = 把 V load(优先 int8 + dequant 到 f16,或 f16 直读)接进
   4-warp kernel,测完整 QK+softmax+PV:
   - 门槛 ≤28ms(相对 FI 33ms ≥15%),cos ≥0.9999;
   - 若 ≤25ms,直接进 owl 接线。
4. 仍需补:per-token Q/K scale 的乘法(当前探针用 1.0)、Q 量化 prepass、
   V 的 int8→f16 dequant/预打包布局。

S1 16-warp 线正式收口;后续只在 4-warp/FI 形态上开发。

---

## 0.6 2026-11-03 完整 4-warp 探针:**K int8 + V f16 shadow = 28.3ms,首次反超**

在 `experiments/int8pv/full_4w_vf16.cu` 把 4-warp QK+softmax 与 f16 PV 合成
一个核(3090 Ti,T=2048/ctx=65k,Bc=32,128 threads):

| 档 | 时间 |
|---|---:|
| QK+softmax only | 14.5 ms |
| 完整 QK+softmax+V f16 load+PV | **28.3 ms** |
| FI 双 half 全核 | 33.0 ms |

⇒ **28.3ms 对 33.0ms = +14.2%,第一次真实反超。**
smem = Q16 + K8 + V_f16 16 + P4 = 44KB,2 CTA/SM(8 warps);
寄存器 169,无 spill。

关键取舍:

- **V 不能每 tile 从 int8 反量化**:同结构下 per-tile int8→f16 dequant 实测
  **68ms**(O(T×ctx×D) 的逐元素转换,不是带宽问题)。
- 必须用 **V f16 shadow**:int8 V 池在 prefill 前/写池时一次性转 f16
  (O(ctx),不是 O(T×ctx));attention 只读 f16 V。
- 因此本方案是 **K int8 + V int8 池 + V f16 shadow**:
  - QK 吃 int8 张量核;
  - PV 吃 f16 张量核,不付 P 量化税;
  - attention 读带宽 = K 1B + V 2B = 1.5× f16,比 FI 的 2× f16 少 25%。
- 探针尚未含:per-token Q/K scale、online softmax rescale、正确输出、
  Q 量化 prepass、V shadow 转换成本。这些是下一步接线工作。

**判决:速度线复活。** 下一步把 per-token scale + online softmax + 输出
补齐,目标 ≤30ms(含 scale)且 cos ≥0.9999;然后接 owl 的 K int8/V shadow。

---

## 0.7 2026-11-03 修正:28.3ms 探针无效;正确核 32.9ms ≈ 平手

**0.6 的 28.3ms 不能作为结论。** 那个探针的 P swizzle 用了 `row & 7`,
而 Bc=32 的 P 行只有 4 个 8-column 组 ⇒ 行 7 的 P 被写到行外,
`T=8` 时 row7 输出直接错(cos 0.36~0.95)。它快,部分是因为 P 布局越界/错误。

修正后的核(`experiments/int8pv/full_4w_correct.cu`)补齐了:

- per-token/head Q/K scale(score 乘 `q_scale * k_scale`,K scale 预乘 softmax_scale);
- 真正的 online softmax(running m/l + 每 tile rescale acc);
- 正确 P 布局(Bc=32 用 `row&3` 或 padding)、V f16 shadow `[key][dim]` + `ldmatrix_x4_t`;
- 正确输出写回。

正确性(host fp32 参考):

| T | L | cos | max\|Δ\| |
|---:|---:|---:|---:|
| 8 | 64 | 0.999999865 | 3.5e-4 |
| 8 | 512 | 0.999999236 | 3.9e-4 |
| 64 | 4096 | 0.999995064 | 7.2e-4 |

性能(3090 Ti,T=2048/ctx=65k,Bc=32,4 warp,2 CTA/SM):

| 档 | 时间 |
|---|---:|
| 正确核(全量) | **33.5 ms** |
| 正确核 + P padding | **32.9 ms** |
| 去掉 Q/K scale | 32.25 ms |
| FI 双 half | 33.0 ms |

⇒ **正确核 ≈ 32.9~33.5ms,与 FI 33.0ms 基本平手,不是 28.3ms 的胜利。**
scale 成本约 1.4ms;online/output 成本很小;28.3ms 的差额主要来自无效 P 布局
和更少的簿记。

**判决(修正版):**

- K int8 + V f16 shadow + int8 QK 在当前 4-warp/Bc32 形态下 **没有确定性优势**;
- 剩余可试的杠杆只有:
  1. 8 warp / 2 KV slice 或 CTA_TILE_Q=128(降低 V 重读、增 occupancy);
  2. 把 P 留在寄存器直接喂 PV(去掉 P smem round-trip);
  3. 更激进的 V 预取/双缓冲。
- 这些都需要重写主循环,预期收益 3~5ms,风险高。若不做,**INT8 KV 在 30 系
  prefill 上应定位为“质量/容量”而非“速度反超”**。

---

## 0.8 公平对比:FP8 无 shadow vs INT8 无 shadow(用户裁决口径)

之前拿 INT8 + V f16 shadow(32.9ms)去对 FI 双 half(33.0ms)是不公平的:
V f16 shadow 付了额外显存/带宽。按用户裁决,改成 **都不开 shadow**:

| 档 | 时间 | 显存 |
|---|---:|---|
| **FP8 无 shadow**（FI fp8 池 + 双 half 核） | **42.89 ms** | 1B/elem |
| **INT8 无 shadow**（4-warp, K/V int8, P uint8, int8 PV） | **29.24 ms** | 1B/elem + scale |
| FP8 + f16 shadow | 32.98 ms | 2B/elem |
| INT8 + V f16 shadow | 32.9 ms | K 1B + V 2B |

⇒ **同无 shadow 口径下,INT8 29.24ms vs FP8 42.89ms = +32%。**
这是本项目第一次在**公平显存口径**下拿到明确速度优势。

INT8 无 shadow 的形态(`experiments/int8pv/full_4w_i8pv_bench.cu`):

- 4 warp / 128 thread,Bc=32
- K int8 池 + per-token/head scale(QK 后乘)
- V int8 预打包 `[Hkv][D][L/2]` b16 + per-(head,64-dim) scale(沿 K 恒定)
- P 每 (row, 32-key tile) 量化 uint8,`mma.m16n8k32.u8.s8.s32`
- int32 PV 每 tile 反量化到 float acc
- smem = Q16 + K8 + V8 + P2 ≈ 34KB,2 CTA/SM
- 无 V f16 shadow,无额外显存

注意:29.24ms 是 **timing 探针**(未含 Q/K scale、online rescale、输出写回)。
按 f16-shadow 正确核的拆账,scale+online+output 约 +2~3ms ⇒ **预期正确核 31~32ms**,
仍比 FP8 无 shadow 42.89ms 快 **25~28%**。

**当前结论:无 shadow 的 INT8 路线成立。** 下一步:
1. 给 4-warp int8 PV 补 per-token scale / online softmax / 正确输出;
2. host 对拍 cos ≥0.9999(参考 S1 full int8 的 0.99995);
3. 若正确核 ≤32ms,接 owl 的 int8 池 + int8 PV 线路(不再需要 V f16 shadow)。

---

## 0.9 2026-11-03 质量门通过:INT8 无 shadow 正确核 34.64ms

4-warp int8 QK + int8 PV(无 V shadow)补齐 per-token scale / online softmax /
正确输出后,host fp32 对拍:

| T | L | cos | max\|Δ\| |
|---:|---:|---:|---:|
| 1 | 64 | 0.999998088 | 1.1e-3 |
| 8 | 64 | 0.999997940 | 1.8e-3 |
| 8 | 512 | 0.999997992 | 4.7e-4 |
| 64 | 4096 | 0.999997908 | 1.6e-4 |

**质量门通过,余量很大。**

性能(3090 Ti,T=2048/ctx=65k,Bc=32,4 warp,无 V shadow):

| 档 | 时间 |
|---|---:|
| FP8 无 shadow(FI 双 half) | 42.89 ms |
| **INT8 无 shadow(正确核)** | **34.64 ms** |
| INT8 无 shadow(timing 探针) | 29.24 ms |

⇒ **同无 shadow 口径,INT8 34.64ms vs FP8 42.89ms = +19.2%,且无额外显存。**
timing 探针到正确核的 +5.4ms 来自 scale/online/output 簿记,是后续 profile 的
第一优化面。

关键 bug(已修):P 是 uint8,必须用 `mma...s32.u8.s8.s32`;此前误用
`mma_s8`(signed A),P>127 被当负数,cos 变负。改用 `mma_u8s8` 后 cos 0.999998。

**结论:INT8 无 shadow 路线成立(质量 + 速度 + 显存三项都过)。**
下一步:接 owl,然后按用户要求用 nsys/寄存器/占用率 profile 找瓶颈。

---

## 0.10 2026-11-03 owl 原生核接入(第一步)

已把验证过的 4-warp int8 核搬进 owl 原生 AOT 面:

| 文件 | 内容 |
|---|---|
| `crates/kernels/cu/attn/int8_attn_helpers.cuh` | 自包含 inline-PTX helpers(ldmatrix/mma/cp.async/warp reduce) |
| `crates/kernels/cu/attn/int8_prefill.cu` | `owl_int8_prefill_paged`(4 warp/Bc32/K int8/V prepacked/P uint8) |
| `crates/kernels/src/ops/attn.rs` | `Int8PrefillPaged` spec + `OWL_INT8_PREFILL_PAGED` |
| `crates/kernels/tests/int8_prefill_probe.rs` | 原生发射探针(T=8/L=64 smoke + T=2048/65k timing) |

探针结果(3090 Ti):

```text
[int8-probe] T=8 L=64 finite=true nonzero=true max|o|=0.580928
[int8-probe] T=2048 L=65536 3 runs: 103.3ms  (~34.4ms/run)
```

与实验核一致。当前 spec 的已知缺口:

- V prepacked 布局(`[Hkv][D][L/2]` b16)还没有 `LayoutTag`/池访问声明;
- 目前只声明 K 的 `KV_MAIN` int8 读;scale/V 的 writer/池账待补(FIXME 已写);
- 还没有接 engine 的 KV cache 分配、prefix cache/snapshot、图路径。

**下一步:V prepack writer + scale plane + engine 配置接线。**

---

## 0.11 owl 原生 V prepack + scale(第二步)

新增原生核:

| 文件/符号 | 作用 |
|---|---|
| `cu/attn/pack_v_int8.cu::owl_v_scale_int8` | 每 (kv_head, 64-dim group) 扫全序列 absmax → fp32 scale |
| `cu/attn/pack_v_int8.cu::owl_pack_v_int8` | V f16 → `[Hkv][D][ceil(T/2)]` b16 int8 预打包 |
| `ops/attn.rs::VScaleInt8 / PackVInt8` | AOT spec + statics |

设计:

- V scale 必须**沿 K 恒定**(per-head/64-dim group),所以是 **两 pass**:
  先扫全序列求 scale,再量化打包;这是 O(ctx) 一次性成本,不是 O(T×ctx)。
- 打包格式 `low byte = key0, high byte = key1`,尾部奇数 token 的 key1 写 0。
- 量化约定与 K writer 一致:round-half-away、clamp [-127,127]、scale=max(absmax/127,1e-6)。

探针(host 逐字节):

```text
[pack-v] OK 1536 pairs, 8 scales
```

V scale 与 packed bytes 均 bit-exact。

当前 native 链路已具备:

```text
K writer (owl_reshape_and_cache_int8_knhd)
V scale + pack (owl_v_scale_int8 / owl_pack_v_int8)
prefill attention (owl_int8_prefill_paged)
```

下一步:engine 侧 KV cache 分配(int8 K + packed V + scale plane)、config 档、
prefix/snapshot 生命周期;之后再接 decode。

---

## 0.12 Engine 接线勘察(下一步的准确落点)

V pack/scale 已进 owl 原生核。Engine 侧要接的四件事与落点:

### 1. 池分配

- `KvQuant` 加 `Int8`(dtype = `I8`,1B/elem);
- `kv_pool_dtype` 走 `quant.dtype()`(size 1 ⇒ I8);
- **生产 V 布局改为 per-page packed**:`[nb, hkv, hd, page/2]` b16。
  page=32 ⇒ 每页 16 个 key pair,**字节数与 int8 标准池相同**(2B/pair vs 1B/token),
  不需要额外池;K 仍 `[nb, page, hkv, hd]` int8。
- scale plane:
  - K: `[nb, page, hkv]` fp32(per-token/head);
  - V: `[hkv, 4]` fp32(per-head/64-dim group,沿 K 恒定,整段一次算);
- 现有 `StatePool::alloc` / `KvBank` / `KvPageFormat` 的 `kv_fp8` 布尔要扩成
  三态(`F16/Fp8/Int8`)或加 `kv_int8`;这是最大改动面。

### 2. 配置档

- `[pool] kv_int8 = true`(与 `kv_fp8` 互斥);
- `KvEnv::int8_paged()`(page 32);
- prefill 选择子走 `owl_int8_prefill_paged`;
- decode 暂**响亮拒**(`unimplemented`/配置校验失败),不静默回退。

### 3. 生命周期

- prefix cache / snapshot / store: K、packed V、K scale、V scale **四件套**
  必须同生命周期搬移;任何一件漏搬 = 静默错值。
- `KvPageFormat` 增加 Int8 分支;`exec/kvpage.rs` 的页拷贝/对账要按四件套走。

### 4. Decode

- 暂不接;int8 decode 需要单独的 paged V packed 读核与 per-head/group scale。
- 配置校验在 int8 + decode 路径直接拒,避免“能启动但出垃圾”。

**下一步代码顺序**:`KvQuant::Int8` → `StatePool::alloc` 三态 → per-page V pack
writer → prefill 选择子 → prefix/snapshot 四件套。decode 最后。

---

## 0.13 Paged 契约落地(engine 接线的前置)

已把 native int8 核改成**真正按页读**:

- K: `[nb, page, hkv, hd]` int8,经 `block_table[tile_k0/page]` 寻址;
- V: `[nb, hkv, hd, page/2]` b16,同样按页寻址;
- Bc=32 = page=32 ⇒ 每个 KV tile 恰好一页,无跨页 tile。

`owl_pack_v_int8` 也改成 per-page 输出:

```text
vp[((page_idx*hkv + h)*D + d)*(page/2) + pair]
```

探针:

```text
[pack-v] OK 8192 pairs, 8 scales   (page=32, bit-exact)
[int8-probe] T=2048 L=65536 3 runs: 114ms (~38ms/run, paged)
```

## 还差的 engine 层一跳

现有 prefill 层是 **K0 写池 + 23 参 paged kernel** 的融合路径;int8 需要一条
**独立层路径**:

```text
Q f16 -> int8 q + qscale           (QuantInt8RowF16, 已有)
K f16 -> int8 k + kscale           (K-only writer; 现有 int8 writer 会顺带写 V)
V f16 -> vscale (全段) -> per-page pack
int8 prefill kernel (owl_int8_prefill_paged)
out f32 -> f16 -> GateMul
```

这条路径不能靠一个 config 开关切;需要:
`KvQuant::Int8` → `StatePool` 三态 → `KvBuffers` int8/packed 形状 →
`paged_prefill_int8_output` → prefix/snapshot 四件套 → decode 拒绝。

当前 native 核/写者/pack 已全部就位,下一步只差这条层路径与池分配。

---

## 0.14 交接收口(2026-11-03)

- 原生 paged 核、K writer、V scale/pack、per-page 契约全部就位；
- native 探针补了 host cos 对拍：**T=1/L=32 cos=0.999996**；
- V scale 改为 atomicMax + finalize，65k 三连 scale+pack **2.55~2.67ms**
  ⇒ **0.85~0.89ms/次**；
- paged 核 65k ≈ **38~39ms**；加 V pack ≈ **39ms**，
  对 FP8 无 shadow 42.89ms **+8~10%**（连续实验核 34.64ms 是上界，不作生产数）。

下一会话的完整步骤、代码地图、复现命令、风险清单见：

```text
roadmap.local/26-11-03-int8-kv-交接.md
```

---

## 0.15 Engine 接线完成(2026-11-03)

INT8 KV 已从「核级可用」推进到**可启动、可 prefill、可 decode、可前缀复用**的
完整引擎路径。工作树 `packages/owl-worktrees/fi-int8`。

### 1. 设计修正:V scale 改为**每页**

0.13 的全局 `Vscale [Hkv,4]` 在 chunked prefill / prefix cache / decode 追加
下不成立:同一序列的 V 会分多次写入,全局单 scale 要么要求全序列先统计,
要么让旧页用新 scale(错值)。

现布局:

```text
K:      [nb, page, hkv, hd]          int8
V:      [nb, hkv, hd, page/2]        b16
Kscale: [nb, page, hkv]              fp32(per-token/head)
Vscale: [nb, hkv, 4]                 fp32(per-page/head/64-dim)
page = 32
```

PV 每个 tile(Bc=32=page)用**该页**的 Vscale;页内恒定,跨页可变。这样
prefill 整页统计、decode 追加复用页 scale、prefix cache 四件套随页搬移,
三者自洽。

同时修正了旧核的 `kscale * 1/16`:host 参考与核同步去掉 `/16`。新的
独立 fp32 对拍(无 /16)cos **0.9999999**,证明 int8 MMA 的原始点积是精确的;
旧 `/16` 只是参考与核一起错,会压平 softmax。

### 2. 新增生产核

| 核 | 作用 |
|---|---|
| `owl_reshape_and_cache_int8_k_only_knhd` | 只写 K + Kscale(V 不再白写) |
| `owl_quant_int8_qk_f16` | Q per-(token,q_head) int8 + qscale(8 warp/block) |
| `owl_v_pack_int8_prefill` | prefill 整页 Vscale + packed V(按 slot) |
| `owl_v_append_int8_paged` | decode 单 token 追加 V(复用/新建页 scale) |
| `owl_int8_decode_paged_split` | decode KV 按 8 片并行,出 partial (acc,m,l) |
| `owl_int8_decode_reduce` | 合并 partial(LSE) |

### 3. 池与四件套

- `KvQuant::Int8`;`StatePool` 三态(F16/Fp8/Int8);
- 每层分配 K int8、packed V、Kscale、Vscale;
- `block_bytes` / `pool_layout` 把四件套都算进页字节与 store 片;
  prefix cache 实测 **7904/7907 cached tokens** 且答案正确;
- `KvPageFormat::Int8`(诊断用;index/coverage/select/image_evict 暂拒);
- `[pool] kv_int8`(与 `kv_fp8` 互斥)。

### 4. Layer 路径

- decode:Q quant → K-only writer → V append → split decode → reduce →
  f32→f16 → GateMul;
- prefill:Q quant → K-only writer → V 整页 pack → `owl_int8_prefill_paged` →
  f32→f16 → GateMul;
- `qkv_fuse` 在 int8 档显式跳过(无融合插池臂);
- prefill 要求 `ctx_base % page == 0`(V 每页 scale 不能跨 chunk 重算);
- decode split `NPARTS = 8`(32 头 × 8 片,4080 上 decode 从 25 → 34 t/s)。

### 5. Boot 门

`kv_int8` 只允许:

- `hd=256` / `page=32` / paged;
- `spec.depth=0`(verify 未接);
- 不开 `index_probe` / `coverage_probe` / `select_decode` / `select_partial` /
  `image_evict`(这些面按 f16/fp8 K 页解码);
- `force_naive_prefill=false`。

违反 = 装配期结构化拒启。

### 6. 真机验收(4080 SUPER 32G;3090 被外部服务占用,待补)

档:`profiles/measurement/qwen38_27b-3090-int8.toml` + `model.quant=w4a8`。

- boot 9.14s;池 int8 8192 页 / 262144 tok;26.3GB used;
- 短数学 `17*23` = **391**;
- 7.9k needle:`7391` 正确;第二次同 prompt **cached 7904/7907**,答案仍正确;
- `llm_speedtest` 单轮 512/2048/8192:
  - prefill **3171 / 3447 / 3343 t/s**(另一次 8k 单点 3038);
  - decode **36.5 / 36.9 / 34.5 t/s**(split 前 8192 为 25.5);
  - 同卡 fp8 参考(w4a8)prefill 3084 / decode 35.9 ⇒ int8 已不输,prefill 略优;
- 17k needle 在 split decode 下出现推理绕圈(旧单块 decode 曾答对);单元
  对拍 L=4096/NPARTS=8 与旧核 **max|Δ|=1e-6**,判断为 greedy 轨迹对浮点
  求和顺序敏感,非注意力错值。待用 GSM8K/needle 集复核。

### 7. 测试

```text
int8_pipeline_probe: Q/K/V writer bit-exact + decode vs host + split/reduce
int8_prefill_probe:  cos 0.9999999(T=1/L=32)+ 65k timing
pack_v_int8_probe:   old global probe(仅参考核)
selfcheck_oracles:   K0 int8 毒值律 ✓
```

已知工作树基线红(与 int8 无关,缺声明/资产):
`cu_sig::every_compiled_symbol_is_declared` /
`knhd_variants_mirror_classic_signature` /
`production_set_is_fully_built` / `tag_filter` 四条。

### 8. 下一步

1. 3090 Ti 上补 A/B(与 fp8 同 w4a8、同 ctx、std 五点三轮);
2. decode split 调参(NPARTS=4/8/16、reduce 合并);
3. 17k 级 quality set(GSM8K / needle)确认轨迹差异不构成质量回归;
4. 可选:V pack 与 Q quant 融合、K writer 与 V pack 融合;
5. prefix cache 四件套的 eviction/restore 压测(当前只验证了未驱逐命中)。

---

## 0.16 双 3090 Ti TP2 实测(2026-11-03)

档:

- `profiles/measurement/qwen38_27b-3090_tp2-int8.toml`(8142)
- `profiles/measurement/qwen38_27b-3090_tp2-fp8.toml`(8143,同形对照)
- 两者同 `model.quant=w4a8`、`max_seq=262144`、`chunk=2048`、spec off。
- 卡位 UUID:rank0 `GPU-e565…`(41:00.0)/ rank1 `GPU-89df…`(42:00.0)。

### 短/中/长 prefill + decode(单轮,output 128,ignore_eos)

| prompt | INT8 prefill | FP8 prefill | INT8 decode | FP8 decode |
|---:|---:|---:|---:|---:|
| 512 | 2246 t/s | 2777 t/s | 63.6 | 63.4 |
| 2048 | 2895 | 3086 | 61.4 | 63.6 |
| 8192 | 3007 | 3010 | 55.1 | 62.3 |
| 32768 | 2749 | 2639 | 47.3 | 60.5 |
| 65536 | 2441 | 2280 | 38.7 | 58.9 |

- **prefill**:8k 基本平手;32k **+4.2%**、65k **+7.1%**;短上下文(512/2k)因
  Q quant / K-only writer / V pack 的固定税,落后 6~19%。
- **decode**:短上下文平手;长上下文显著落后(65k 38.7 vs 58.9)。根因 =
  当前 decode PV 是**标量**逐维累加;FI decode 走 tensor-core PV。NPARTS
  256 已是当前峰值(512 反而略降)。
- **结论**:INT8 KV 的收益在**长上下文 prefill**;若生产同时看重长上下文
  decode,下一刀必须是 int8 decode PV tensor-core(或 prefill int8 / decode
  另用 f16 shadow 的混合档,但会付显存)。

### 正确性

- 双 3090 TP2,`17*23 = 391`;
- 7.9k needle:256 tok 推理预算会用尽,512 tok 答对 `7391`;
  二次同 prompt `cached=7904/7907`,答案仍正确;
- kernel 对拍(3090,NPARTS=256):
  `split/reduce max|Δ|=1e-6`;Q/K/V writer bit-exact。

### 启动

- TP2 装配 11.02s;每 rank 池 4096MiB(8192 页 × 32 tok),used ~14.7GB/24G;
- `[boot] 现役臂: prefill=int8 | ... | kv=int8`;
- GDN chunked AOT 必须按 **arch-sm86** 单独构建(双代际时 `single_arch()` 取最后
  一项 sm89,30 系会 `CUDA_ERROR_NO_BINARY_FOR_GPU`)。

### 仍待

1. int8 decode PV tensor-core(长 ctx decode 的唯一有效杠杆);
2. 短上下文 prefill 税:Q quant + K writer + V pack 融合/裁剪;
3. 17k 级质量集与 fp8 的 greedy 轨迹差异复核;
4. NPARTS/分片粒度的正式 sweep(当前 256 是单点最优)。

性能优化的下一会话交接见：

```text
roadmap.local/26-11-03-int8-kv-性能优化-交接.md
```

重点：decode PV 标量 → dp4a/tensor-core；prefill K writer + V pack 融合；
短 T 变体；质量门与复现命令。

## 0.17 单 3090 Ti 性能优化与质量边界(2026-11-03 二轮)

工作树 `packages/owl-worktrees/fi-int8`(分支 `fi-int8`,base `05a1314e`)。
本轮只动 decode 核与测量档,**未提交 git**。

### 0.17.1 decode:GQA 共享 K/V + tensor-core QK/PV

旧 `owl_int8_decode_paged_split` 每 `(q_head, part)` 一个 block,同 KV head 的
6 个 q head 各读一遍 K/V;PV 是逐维标量 FMA。新
`owl_int8_decode_gqa_tc_split`:

- grid = `(Hkv, B, nparts)`,一个 block 服务同 KV head 的 `grp=Hq/Hkv` 个 q head;
- K/V 每页只读一遍,staging 进 shared;
- QK/PV 都用 `mma.m16n8k32.u8.s8.s32`;PV 8 warp ×32 dim,每 warp 4 个 n-tile;
- P 为 uint8 per-row/per-tile scale,V 仍 per-page/64-dim scale;
- 保留原 `out_dummy/dep_k/dep_v` 依赖边 ABI,partial 布局与 reduce 不变。

独立核 benchmark(3090 Ti,65k,NPARTS=256,host 对拍旧 split):

| ctx | 旧 split | TC GQA split | 加速 |
|---:|---:|---:|---:|
| 512 | 0.021 ms | 0.011 ms | 1.87× |
| 2048 | 0.051 ms | 0.015 ms | 3.51× |
| 8192 | 0.174 ms | 0.032 ms | 5.48× |
| 32768 | 0.669 ms | 0.091 ms | 7.37× |
| 65536 | 1.311 ms | 0.167 ms | 7.86× |

对拍:`cos=0.999999955`,`max|Δ|=2.8e-4`(P uint8 量化固有误差)。
生产 kernel probe(L=4096)对照旧 split:`max|Δ|=4.99e-3`,已入
`crates/kernels/tests/int8_pipeline_probe.rs`(阈值 1e-2,真实质量走 needle)。

单 3090 Ti / 27B / **w4a16** / int8 KV / ctx 131072 / chunk 2048:

| prompt | prefill 旧 decode | prefill TC | decode 旧 | decode TC |
|---:|---:|---:|---:|---:|
| 512 | ~1398 | ~1393 | 47.5 | 47.3 |
| 2048 | ~1436 | ~1434 | 46.4 | 47.7 |
| 8192 | ~1408 | ~1407 | 42.6 | 47.0 |
| 32768 | 1296.7* | 1264.0* | 31.7 | 44.3* |
| 65536 | 1169.8* | 1141.0* | 25.1 | 42.65* |

`*` = 3 轮均值;其余 3 轮。TC decode 把 65k decode 从 25.1 提到 42.65,
与 FP8 同形档的 44.02 只差 3.1%;32k 44.3 vs 45.7(-3.1%)。

### 0.17.2 单卡 FP8 同形 A/B(w4a16,3 轮)

| prompt | INT8 prefill | FP8 prefill | INT8 decode | FP8 decode |
|---:|---:|---:|---:|---:|
| 32768 | 1264.0 | 1207.9 | 44.30 | 45.70 |
| 65536 | 1141.0 | 1046.8 | 42.65 | 44.02 |

即长上下文 prefill **+4.6% / +9.0%**,decode **-3.1%**。
短上下文 prefill 基本平手(512/2k 仍略低),8k 也平手。

### 0.17.3 w4a8:速度很香,但质量门红

单卡 w4a8 int8(same profile override)实测:

| prompt | prefill | decode |
|---:|---:|---:|
| 512 | 2387 | 44.6 |
| 2048 | 2483 | 45.2 |
| 8192 | 2358 | 44.6 |
| 32768 | 2043 | 42.8 |
| 65536 | 1711 | 40.3 |

65k prefill 比 w4a16 的 1141 高 **50%**,比 FP8 w4a8 的 1502 高 **13.9%**。
但 **8k needle 实测 w4a8+int8 丢失**(同 prompt w4a8+fp8 命中、w4a16+int8
命中);因此 w4a8 不能进生产档。根因候选 = int8 KV/P 量化与 w4a8 激活
outlier 的交互,不是 TC decode(旧标量 decode 同样失败)。

### 0.17.4 长上下文质量边界(必须记住)

- 8k needle(`ZXCVBNM12345`,depth 50%,max_tokens 512):
  - w4a16 + int8:命中;
  - w4a8 + int8:丢失;
  - w4a8 + fp8:命中。
- **32k needle:w4a16 + int8 丢失,同形 FP8 命中**;旧标量 decode 与 TC
  decode 都丢失 ⇒ 问题在 int8 prefill/KV 路径,不在 decode 核。
- 因此当前 int8 KV 的“长上下文 prefill 性能”只在**速度**上成立;
  **32k 质量门未过**,不能宣称生产可用。下一步必须先定位:
  1) P uint8 per-tile 量化(小权重被压掉) vs 2) K per-token / V per-page
  量化(长上下文误差累积);
  3) 用 16-bit 两段 P(hi/lo 两遍 mma)或 f16 PV 做质量对照;
  4) 若确认 K/V,再做 per-64-group K scale / per-token V scale 的代价评估。

### 0.17.5 结论

- **decode 性能战已打赢**:TC GQA split 7.86×,生产 65k 25.1→42.65,
  与 FP8 差 3.1%,基本持平。
- **prefill 速度小胜**:w4a16 档 32k/65k +4.6%/+9.0%;w4a8 档 +50%
  但质量门红,不可用。
- **长上下文质量是当前真正的硬仗**:32k needle int8 失败,必须先修
  int8 prefill/KV 精度或 P 量化,再谈 w4a8 与更大 prefill 收益。

### 0.17.6 100k 逐核归因与 attention 屋顶(2026-11-03 续)

用引擎自带 CUPTI(`--perf-level gpu`,epoch 视图)对单 3090 Ti / w4a16 /
int8 KV 跑 100k prompt:

| 项 | 数值 |
|---|---:|
| TTFT | 94.65 s |
| prefill | 1056.5 t/s |
| `owl_int8_prefill_paged` | **25.85 s / 27.3%** |
| decode | 44.4 t/s |
| `owl_int8_prefill_paged` 发射 | 784 = 49 chunk × 16 层 |

65k 同口径:attention 11.10s / TTFT 56.2s = 19.7%。
CUPTI 单 tag 4096 样本封顶,所以 Marlin 的 total 只能看 nsys(65k 约 42s);
但 attention 发射数 < cap,25.85s 是全量真值。

**核内已到屋顶**:

- QK/PV 的 int8 mma 实测约 115–143 TMAC/s(3090 Ti int8 tensor 峰值约
  142–160 TMAC/s);
- softmax 的 exp2 约 3.2G 次/chunk 级,已接近 SFU 峰值;
- 试过 2-head 共享 K/V、Br=128、8 warp、warp-specialized QK/PV 流水、
  head-fast grid、f16 acc、`ex2.approx.f16x2`:
  - K/V 共享/head-fast 无收益 ⇒ 不是带宽瓶颈;
  - 8w/流水版因 256 线程 × 128 acc 寄存器压力 spill(local 96–464B),
    反而 0.83–0.98×;
  - f16 acc 0.95× 且 cos 0.999998;
  - 唯一干净收益 = `ex2.approx.ftz.f32` 替换 `exp2f`,核 −5~6%,
    输出与旧核逐位一致(max|Δ|=0)。

**结论**:int8 attention 核不是“还能靠调参再砍一半”的对象;它的
QK/PV 在 tensor roof、softmax 在 SFU roof。要继续显著降 attention,
只能改算法(减 attend 的 KV / 更粗的 KV 压缩 / 近似),或者接受
w4a8 的速度但先修 32k 质量。当前代码保留 `ex2f_fast`(commit
`954c3ea6`),其余实验不落生产。

### 0.17.7 100k 同卡 A/B:INT8 KV vs FP8 KV 的 attention 占比

同一单 3090 Ti、同 27B w4a16、同 `max_seq=131072`、同 `prefill_chunk=2048`、
spec off、同 100k prompt、同 output=8；两边都开引擎 CUPTI
(`--perf-level gpu`，epoch 视图)。

| 指标 | FP8 KV | INT8 KV | 变化 |
|---|---:|---:|---:|
| TTFT | 106.05 s | **91.94 s** | **−13.3%** |
| prefill | 943.0 t/s | **1087.7 t/s** | **+15.3%** |
| attention kernel 总时间 | **37.66 s** | **23.04 s** | **−38.8%(1.63×)** |
| attention 占 prefill | **35.5%** | **25.1%** | −10.4 pt |
| decode | 44.25 t/s | 44.37 t/s | 持平 |

归因核名：

- FP8:`flashinfer::BatchPrefillWithPagedKVCacheKernel` 37.66s(n=896);
  另有 `PersistentVariableLengthMergeStatesKernel` 48ms(可忽略);
- INT8:`owl_int8_prefill_paged` 23.04s(n=784)。

**结论**:100k 下 INT8 KV 把 attention 从 37.66s 压到 23.04s,
attention 占 prefill 从 35.5% 降到 25.1%,端到端 prefill +15.3%。
这就是 INT8 KV 在长上下文的核心收益;它不是“attention 占比越低越没用”,
而是把 FI 那笔 O(T·ctx) 的大头换成了更便宜的 int8 路径。

剩余 23.04s 的核内账(65k microbench 比例外推):QK/PV 已接近 int8
tensor roof,softmax 接近 SFU roof;`ex2.approx` 已再省 ~2.8s(25.85→23.04s)。
要继续压,只能改算法(减 attend 的 KV / 更粗压缩 / 近似)或做跨 warp
流水,而流水版当前因 128 acc 寄存器压力 spill,未过性能门。

### 0.17.8 算子流水线审计:去掉 INT8 专属的 F32→F16 空转

100k CUPTI 显示 INT8 与 FP8 的 attention 前处理链差异:

| 阶段 | FP8 KV | INT8 KV(优化前) | INT8 KV(优化后) |
|---|---|---|---|
| Q | NormRope F16 → FI | NormRope F16 → QuantInt8 | 同左 |
| K | NormRope F16 → K0 dual write | NormRope F16 → K-only writer | 同左 |
| V | 同上(K0 同发) | v_proj F16 → V pack | 同左 |
| attention out | FI 直接 F16 | `owl_int8_prefill_paged` **F32** | **F16** |
| 后处理 | GateMul | **CastF32F16 → GateMul** | **GateMul** |

`CastF32F16` 是 FP8 路径不需要、INT8 路径继承下来的纯空转:
attention 已经算出最终值,却先写 F32、再单独读回转 F16。已改为
kernel epilogue 直接写 F16,删除该节点。

实测(100k,CUPTI epoch):

| 项 | 优化前 | 优化后 |
|---|---:|---:|
| `owl_int8_prefill_paged` | 23.04 s | **22.91 s** |
| `owl_cast_f32_f16` | 存在 | **0** |
| TTFT | 91.94 s | **91.70 s** |
| prefill | 1087.7 t/s | **1090.5 t/s** |

**负结果**:也试过把 `GateMul` 的 sigmoid 一起塞进 attention epilogue
(彻底去掉 cast+gate 两发)。结果 attention 23.04→23.57s,端到端反而
变慢——9.8G 次 sigmoid 被塞进 attention 关键路径,而独立 GateMul
是访存型、并行度高。已回退,只保留“输出 F16”,GateMul 仍独立。

**剩余可融合项与上限**:

- Q 的 NormRope + int8 quant:可省 `owl_quant_int8_qk_f16`(~35ms/100k)
  与一次 Q F16 写读;还能让 int8 量化在 F16 舍入前发生,略提精度。
- K 的 NormRope + int8 quant + K writer:同理,省一次 K F16 写读。
- K writer + V pack 合并:两发合一,省一次发射;但 V 的 per-page scale
  仍需整页归约。
- 以上合计 <0.3% 端到端;真正的大头仍是 QK/PV/softmax 的硬件屋顶。

commit:`6b5adbce`(F16 输出 + 去 cast)、`954c3ea6`(ex2.approx)。

### 0.17.9 算子级优化:V 预取与 QK 重叠(2026-11-03 六轮)

100k attention 23s 的核内结构继续拆,试了一组算子级改动:

| 实验 | kernel 结果 | 结论 |
|---|---:|---|
| `ldmatrix.x4` 配对 K/V 两个 n-tile | 65k −1.3%,100k 0% | 正确但收益不稳,未落 |
| PV swizzle 关掉 | 与 baseline 持平 | bank conflict 不是瓶颈 |
| Br=128 / 8 warp | 0.91×,输出也不对 | 1 block/SM + 因果浪费,废弃 |
| 全 half2 P(exp2+量化) | −1.2%,cos 0.99985 | 精度/收益都不划算,废弃 |
| **V 预取与 QK 重叠** | **65k −6.9%,100k −6.1%** | **落生产** |

V 预取的做法:

```cpp
// 每个 key tile:
K cp.async → commit(group 0)
V cp.async → commit(group 1)
cp.async.wait_group 1;   // 只等 K
__syncthreads();
QK(...);                 // 此时 V 还在飞
cp.async.wait_group 0;   // 等 V
__syncthreads();
PV(...);
```

K/V 在 shared 里本来就是前后半区互不重叠,所以 V 的 cp.async 可以
在 QK 计算期间完成,不需要额外 double buffer。

**关键坑**:V 预取改变代码调度后,PV 的 `acc += c*s` 被编译器排成
另一种 FMA/非 FMA 组合,输出整体差 1 个 F16 ULP。8k needle 从
`ZXCVBNM12345` 变成 `ZXCVBNM`(稳定复现)。把 PV 累加显式写成
`__fmaf_rn(c, s, acc)` 后,8k needle 恢复;说明 greedy 长文检索对
单 ULP 也敏感,不能把“数学等价”当“质量等价”。

100k CUPTI(同卡同档):

| 项 | V 预取前 | V 预取后 |
|---|---:|---:|
| `owl_int8_prefill_paged` | 22.91 s | **21.35 s(−6.8%)** |
| TTFT | 91.70 s | **90.73 s** |
| prefill | 1090.5 t/s | **1102.2 t/s** |
| attention 占 prefill | 25.0% | **23.5%** |

同口径 FP8 attention 37.66s ⇒ INT8/FP8 attention = **1.76×**。
8k needle 通过;32k needle 的已知 INT8 长文退化仍未解决。
commit `b9d0bb05`。

### 0.17.10 算子级优化:K 双缓冲预取与 PV 重叠(2026-11-03 七轮)

V 预取之后,继续把 K 也做成双缓冲:

```text
shared: Q 16K | K0 8K | K1 8K | V 8K | P 2K | Kscale0/1 2×128B = 43,264B
        (2 blocks/SM = 86.5K,仍放得下)

prologue: load K0 + kscale0
tile kb:
  V(kb) cp.async → commit
  QK(kb) 用 K[kb&1]          // V 在飞
  wait V; sync
  K(kb+1) cp.async → K[kb&1^1],同时写 kscale[next]
  PV(kb)                     // K(kb+1) 在飞
  wait K; sync
```

两个 global load 都藏进计算:

- V(kb) 与 QK(kb) 重叠;
- K(kb+1) 与 PV(kb) 重叠。

独立 kernel(3090 Ti,128 thread,4 warp):

| 场景 | V 预取 | V+K 双预取 | 增益 |
|---|---:|---:|---:|
| T=2048 / L=65k | 33.56 ms | **30.64 ms** | **−8.7%** |
| T=2048 / L=100k | 51.87 ms | **47.81 ms** | **−7.8%** |
| T=2048 / L=8k | 3.86 ms | **3.43 ms** | **−11%** |

100k CUPTI(同卡同档):

| 项 | V 预取 | V+K 双预取 |
|---|---:|---:|
| `owl_int8_prefill_paged` | 21.35 s | **18.97 s(−11.2%)** |
| TTFT | 90.73 s | **88.52 s** |
| prefill | 1102.2 t/s | **1129.7 t/s** |
| attention 占 prefill | 23.5% | **21.4%** |

对 FP8 KV 同口径:

| 指标 | FP8 | INT8(新) | 变化 |
|---|---:|---:|---:|
| attention | 37.66 s | **18.97 s** | **1.99×** |
| TTFT | 106.05 s | **88.52 s** | **−16.5%** |
| prefill | 943.0 t/s | **1129.7 t/s** | **+19.8%** |

8k needle 通过;PV 显式 `__fmaf_rn` 保留(否则单 ULP 会翻掉
`ZXCVBNM12345`)。commit `f17b49b3`。

### 0.18 每 tile barrier 下限 + NormRope/QK 量化融合(2026-11-03 八轮)

#### (1) 单 barrier 尝试:51200B shared ⇒ 1 block/SM,失败

想把每 tile 的两个 `__syncthreads` 压成一个,方案 = K/V 都双缓冲 +
一个 barrier 放在 PV 之后:

```text
Q 16K + K0/K1 16K + V0/V1 16K + P 2K = 51,200B
```

结果:

- 51200B/block 在 3090 上只能放 **1 block/SM**(4 warp),
  而当前 43.3KB 配置是 **2 blocks/SM**(8 warp);
- 独立 kernel:1bar 46.1ms vs 当前 30.9ms,**0.67×**。

结论:**2 barrier/tile 是当前 2-block 占用下的结构下限**;要单 barrier
必须牺牲一个 block 的占用,不划算。K/V 预取已把 global load 隐藏,
barrier 本身不是主要成本。

#### (2) NormRope + Q int8 / K int8 写池融合

新增两个 kernel(`cu/attn/norm_rope.cu`):

- `owl_norm_rope_q_int8_f16`:norm+rope 后直接算 q8 + qscale;
- `owl_norm_rope_k_int8_write_f16`:norm+rope 后直接写 paged K int8 池 +
  k_scale(槽映射内联)。

关键细节:

- **先落 F16 再量化**:`v = __half2float(__float2half_rn(v))`。
  不做这一步,17×23 会陷入 greedy 循环;做了之后与旧
  NormRope(F16)+QuantInt8/K-writer 链数值一致,17×23 = 391。
- 归约从 256-thread 树改成 **warp shuffle + 8 partials**,把每个
  reduction 的 8 个 barrier 压到 2 个;Q kernel 257→135ms,K 48→29ms。
- 去掉 `owl_quant_int8_qk_f16` 与 `owl_reshape_and_cache_int8_k_only_knhd`
  两个节点;Q/K 的 F16 中间张量不再落显存。

100k CUPTI(同卡同档):

| 项 | 融合前 | 融合后 |
|---|---:|---:|
| `owl_int8_prefill_paged` | 18.97 s | 18.90 s |
| `owl_norm_rope_q_int8_f16` | — | 134.8 ms |
| `owl_norm_rope_k_int8_write_f16` | — | 29.3 ms |
| `owl_quant_int8_qk` + K writer | ~40 ms | **0** |
| Q/K NormRope F16 | ~154 ms | **0** |
| TTFT | 88.52 s | **88.31 s** |
| prefill | 1129.7 t/s | **1132.4 t/s** |

端到端约 +0.25%,与预期一致;主要价值是**少两个算子节点、少两次
F16 写读**,并为后续质量实验提供干净的 Q/K 量化入口。

8k needle 通过;17×23 = 391。注意:同一 server 先跑数学再跑 needle,
implicit session 的前缀缓存会让 needle 偶发失败;质量门必须 **fresh
server**,否则会把缓存态误判成数值回归。

commit `a946f6db`。

---

## 0.19 2026-11-03 性能新方案:P32 退场,`int8 QK + f16 PV`(设计)

> 背景:§0.17/§0.18 的 P32 单 K 质量态已让 27B 8k/16k/32k/64k needle
> 全过,但 PV 是 **4 个 `u8s8` mma**(P32 四个字节),T=2048/65k
> probe 约 188ms/3 runs(≈62.7ms/run),相比 P8 时代的 34.6ms 慢近 2×。
>
> 三个参考项目的结论见
> `roadmap.local/26-11-03-int8-kv-三个参考项目-性能借鉴.md`:
> **ninfer / pthfa 的生产 int8 attention 都不是 P32,而是 int8 QK + f16 PV。**
> 本章是据此设计的下一版性能态,不是重复 §0.6/§0.7 的旧 f16 试验。

### 0.19.1 目标与不变量

```text
保留:
  QK: int8 m16n8k32 + per-32 Q/K scale(数值路径不动)
  KV 池: K int8 / V int8(1B/elem 质量态)
  prefill 4 warp / Bc=32 / Br=64 / 2 CTA/SM 的 occupancy 形态

替换:
  P32 四个 uint8 平面 + 4×u8s8 PV
  → P f16(带 2 的幂 headroom)+ 2×f16 m16n8k16 PV

质量护栏:
  27B 8k/16k/32k/64k needle(fresh server)
  17×23=391
  0.8B int8-vs-fp8 逐层不退化
```

**明确不做**:

- 不回到 16-warp S1(§0.3 已证明 1 CTA/SM 是负结果);
- 不做 naive WMMA f16 PV(§0.4 的 54.8ms 是反例);
- 不把 P 换成普通 f16 直接 exp(会重蹈 P24 的 2^-24 下限);
- 不在 prefill 每 tile 做 int8→f16 全量 dequant 到 shared
  (§0.6 实测 68ms,是 O(T×ctx×D) 的税)。

### 0.19.2 核心:给 f16 P 加 2 的幂 headroom

普通 f16 P 的下限是 `2^-24`(subnormal),和 P24 一样;32k needle 的
小权重会归零。新方案沿用 pthfa 的 per-subtile 2 的幂归一化,并额外
留 15 bit headroom:

```text
每 tile:
  c_new = ceil(log2(max_t vscale_t))        // vscale = V 的 per-token scale
  P''_t = fp16( p_t * vscale_t * 2^(15 - c_new) )
  P''_t <= 2^15 = 32768 < f16 max 65504

跨 tile:
  acc *= 2^(c_old - c_new)                  // 精确 2 的幂
  l   += sum p_t                            // l 仍是未缩放的 p
输出:
  out = acc * inv_l * 2^(c_cur - 15)
```

有效下限:

```text
f16 subnormal 2^-24 / (vscale * 2^(15-c))
vscale ≈ 2^c
⇒ 有效 p 下限 ≈ 2^-39
```

比 P32 的 `2^-32` 还宽,同时保留 f16 的 11-bit mantissa。
P'' 的量化误差约 0.05%,远小于 FP8 路径的 P 误差。

### 0.19.3 PV 形态:每 32 key 2 个 f16 mma

当前 P32:

```text
A = P32 四个 uint8 平面(hi/mid/lo/lo2)
B = V int8 packed(两 key 一个 uint16)
PV = 4 × mma.u8.s8.m16n8k32
```

新方案:

```text
A = P'' f16, 16 rows × 16 keys
B = V raw int8 → f16(精确;-128..127 在 f16 中可精确表示)
PV = 2 × mma.f16.f16.f32.m16n8k16 (k=16 × 2)
```

V 的 per-token scale 不进入 V,而是折进 P'';因此 V f16 只是
`f16(code)`,不需要每 key 乘 scale。

PV mma 数:

```text
P32: 4 mma / 32 keys
f16: 2 mma / 32 keys   ⇒ −50%
```

同时去掉 P32 的 8KB shared 平面、P store 和 P ldmatrix(若 P 片段
直接在 softmax 寄存器里生成;第一版可先落 P f16 shared 4KB)。

### 0.19.4 V 的两条实现路线

#### 路线 A(先落地,复用 §0.6/§0.7 的正确核经验)

**V f16 shadow**,但 shadow 存 **raw int8 code 的 f16**(不是 dequant 后的 V):

```text
K int8 池 + V int8 池 + V_f16 shadow(token-major [slot][hkv][hd])
shadow 内容 = f16(int8_code),无 vscale
P'' 已含 vscale
```

- VPack / V append 在写 int8 V 时顺手写 shadow;
- prefill 读 V_f16 shadow 做 `ldmatrix_x2_t`;
- 显存代价 = V 从 1B → 2B(整体 K/V 1.5B/elem);
- 历史正确核 §0.7 的 32.9ms 就是这个形态,但当时 P 是普通 f16;
  本方案只改 P 的 2 的幂 headroom。

优点:改动小,能快速验证质量 + 速度;缺点:显存/带宽 +1B/V elem。

#### 路线 B(显存中性,第二阶段)

**V int8 寄存器 cvt4**,不建 shadow:

```text
V 池保持 1B;
V 的 shared 布局改 token-major(或保留 dim-major + 设计对应 ldmatrix);
PV 内用 pthfa cvt4:
  4 个 int8 → 2 个 f16x2(1 LOP3 + 2 PRMT + 2 HSUB2)
  直接形成 B 片段,不写 16KB V_f16 staging
```

- 显存回到 1B/elem;
- 代价是 PV 每条 B 片段多 4 PRMT/4 cvt4;
- §0.6 的“68ms”是 per-tile 全量 dequant 到 shared,不是这个寄存器路径;
  pthfa 的 cvt4 是按 PV 实际消费的片段做,量级不同。

建议:**A 先验收,B 作为 A 达标后的显存优化**,不要把两件事绑死。

### 0.19.5 调用链(第一版 4 warp)

```text
owl_int8_prefill_paged_f16pv:
  prologue: Q int8 → smem; K int8 → smem; K/V scale → smem
  per key tile:
    V int8/f16 load 与 QK 重叠
    QK int8 m16n8k32 → score f32
    mask + online softmax:
      c_new = ceil_log2(max vscale)
      P'' = f16(p * vscale * 2^(15-c_new))
      acc *= 2^(c_old-c_new); l += p
    PV f16:
      A = P''(可先写 P_f16 smem 4KB,再 ldmatrix_x4)
      B = V_f16(路线 A)或 cvt4(V int8,路线 B)
      2 × mma.m16n8k16 per n-tile
    K(next) 预取与 PV 重叠
  输出:
    out = acc * inv_l * 2^(c_cur-15)
```

shared 预算(路线 A):

```text
Q 16K + K 8K + V_f16 16K + P_f16 4K + Kscale 1K + Vscale 128B ≈ 45.1KB
⇒ 2 CTA/SM(90.2KB)仍成立
```

路线 B:

```text
Q 16K + K 8K + V_int8 8K + P_f16 4K + scales ≈ 37.1KB
⇒ 更大余量,可再考虑 Bc=64
```

### 0.19.6 验收线

kernel(T=2048 / ctx=65k,3090 Ti,同卡):

```text
当前 P32 单 K        : ≈62.7 ms/run(188ms/3)
目标 f16 PV 路线 A   : ≤40 ms(先恢复),冲刺 ≤33 ms(§0.7 形态)
目标 f16 PV 路线 B   : ≤35 ms,且显存回到 1B/elem
```

质量:

```text
host 参考 cos ≥ 0.9999
27B 8k/16k/32k/64k needle fresh server 全过
32k depth 10/50/90
17×23 = 391
0.8B T=64/1024 int8-vs-fp8 不退化
```

端到端:

```text
65k prefill ≥ 1060 t/s(先恢复),冲刺 ≥ 1130
100k TTFT ≤ 100s,冲刺 ≤ 95s
```

### 0.19.7 里程碑 / kill criteria

| 阶段 | 内容 | 通过条件 | 失败处置 |
|---|---|---|---|
| F1 | P f16×2^15 + V f16 shadow,4 warp | host cos≥0.9999;32k/64k needle 过;≤40ms | 若 P 质量不过,保留 P32,停止 f16 P |
| F2 | F1 + K/V 预取重叠 + P 片段寄存器化 | ≤35ms;65k/100k e2e 达标 | 回退 F1,不接生产 |
| F3 | V cvt4 无 shadow(路线 B) | 显存 1B/elem;≤35ms;质量不降 | 保留 F1/F2 shadow 档,作为性能 mode |
| F4 | 生产接线 + profile | 100k TTFT ≤95s;质量门全绿 | 回退 P32 单 K |

**硬 kill**:任一路线在 32k/64k needle 上出现 INT8 vs FP8 分叉,立即
停止该路线;**不允许用 0.8B 短上下文代替 27B 长文质量门**。

### 0.19.8 与历史负结果的关系

- §0.4 `naive WMMA f16 PV 54.8ms`:不是本方案;本方案是 4-warp、
  P/V smem 常驻、手写 ldmatrix + m16n8k16,§0.5/§0.7 已证 11.36/32.9ms。
- §0.7 正确 f16 PV 32.9ms:P 是普通 f16,长上下文下限未解决;
  本方案只增加 `2^(15-c)` 的 P 归一化,PV 结构不变。
- §0.17.9/§0.17.10:V/K 预取与在线 softmax 的调度保留,不重新发明。
- §0.18:2 barrier/tile 与 4 warp/2 CTA 的 occupancy 结论保留;
  本方案不回到 1 barrier 或 16 warp。

### 0.19.9 一句话

**不要再给 P32 省第 4 个 mma;把 P 换成带 2 的幂 headroom 的 f16,
V 换成 f16(先 shadow、后 cvt4),PV 从 4 个 int8 mma 降到 2 个 f16 mma。
这是当前质量态下唯一有希望把 62.7ms 拉回 33ms 量级、且不牺牲
32k/64k needle 的路线。**

### 0.19.10 首轮探针(2026-11-03,experiments/int8pv)

在 `experiments/int8pv` 上先跑已有核 + 两个 P scale 变体,同卡 3090 Ti、
T=2048/ctx=65536、4 warp / Bc32 / V f16 shadow:

| 变体 | P/V | acc | cos(T=8,L=512) | ms |
|---|---|---|---:|---:|
| `full_4w_vf16`(timing only) | P f16 普通,V f16 | f16 | — | **28.256** |
| `full_4w_correct`(per-token scale/online/output) | P f16 普通,V f16 | f16 | 0.999997992 | **33.521** |
| `full_4w_correct_p15` | P f16×2^15,V f16 | **f32** | 0.999999980 | **43.256** |
| `full_4w_correct_p8` | P f16×2^8,V f16 | **f16** | 0.999999236 | **33.709** |
| `full_4w_correct_p12` | P f16×2^12,V f16 | f16 | NaN/inf(溢出) | 33.247 |

关键结论:

1. **f16 PV 的正确核 33.5ms**,比当前 P32 单 K 的 62.7ms/run 快约
   **1.87×**,说明方向成立。
2. **P×2^8 + f16 acc** 是当前最好点:
   - 有效 p 下限 = `2^-24 / 2^8 = 2^-32`,与 P32 相同;
   - cos 0.999999236,仍高于 P24 的 0.999999977? 略低但同量级;
   - 33.7ms,几乎不付 f32 acc 的代价。
3. **P×2^15 必须配 f32 acc**:f16 acc 在 P≥2^12 时溢出(NaN),
   f32 acc 版本 43.3ms,是质量保险档,不是首选。
4. 因此 F1 采用:
   - **P f16 × 2^8 + f16 accumulator + V f16 shadow(先 dequantized)**;
   - P×2^15/f32 作为 fallback,若 32k/64k needle 显示 P8 精度不够再切。

下一步:

- 把 `full_4w_correct_p8` 的 P scale 接进生产核(先 V f16 shadow);
- 27B 8k/16k/32k/64k needle fresh 验收;
- 若 P8 通过,再做 V cv4/无 shadow 的路线 B;
- 若 P8 长文失败,切 P15+f32 保险档(43ms 仍优于 P32 62.7ms)。

### 0.19.11 生产接线实跑:8k/16k 过,32k 失败,已回退(2026-11-03)

> **已被 §0.20 推翻**:32k 失败根因是当时尚未修完的 norm_rope/K/V scale
> 旧 bug,不是 f16 P 的 11-bit 精度。修复后 P8 单平面即可过 100k。

按 §0.19 做了第一版生产接线:

- 新核 `owl_int8_prefill_paged_f16pv`(`cu/attn/int8_prefill_f16pv.cu`):
  QK int8 / P f16×2^15 / V f16 shadow / f16 PV f32 acc;
- VPack/VAppend 同步写 V f16 shadow;
- 池新增 `KvInt8Blocks.v_f16` 平面;
- `KvEnv::int8_paged` 暂切 f16-PV 核。

实测:

```text
kernel host parity T=1/32:  cos=0.999999956  max|d|=1.69e-4  PASS
0.8B T=8    int8-vs-fp8 fin=3.00e-2
0.8B T=64   int8-vs-fp8 fin=1.75e-2
0.8B T=1024 int8-vs-fp8 fin=1.42e-2   (优于 P32 的 1.69e-2)
27B 8k  needle RETRIEVED
27B 16k needle RETRIEVED
27B 32k needle MISSED ("ZXCVBN!!!!")   ❌
```

同时 V f16 shadow 把 per_tok 从 35,072B 抬到 67,840B:

```text
max_seq 131072 放不下;只能 --set runtime.max_seq≈70k
```

判决:

- **f16 P 的 11-bit mantissa 对 27B 32k needle 不够**:
  16k 过、32k 挂;与 P24 的失败点同形,但原因不是范围
  (P15 有效下限 2^-39 已优于 P32),而是 P 的**相对精度**。
- 0.8B 长文误差反而优于 P32,说明这是 27B needle 的敏感边界,
  不能用 0.8B cos 代替质量门。
- V shadow 的 1.5B/elem 显存代价还会把 131k 档打到 ~70k,
  与当前容量目标冲突。

处置:

- **生产已回退到 P32 单 K**(`KvEnv::int8_paged` 仍选
  `Int8PrefillPaged`);V f16 shadow 平面/写者/池字段全部撤回。
- 保留 `int8_prefill_f16pv.cu` 与新 KernelSpec 作为研究核 +
  `int8_prefill_f16pv_correctness_small` 对拍;不接生产。
- 下一步若继续 f16 PV,必须解决 **P 的 11-bit 相对精度**:
  候选 = P high/low 双 f16(4 mma,无 mma 收益)、
  或 P f16 只用于短 ctx / decode、长 ctx 仍 P32;
  在没有 32k/64k needle 证据前不再接生产。

---

## 0.20 2026-11-03 翻案:P8 单平面才是生产态(P32/P24 退场)

### 0.20.1 结论

**此前 P8/P24/P32 的质量分叉不是 P 精度,而是 writer/scale 旧 bug。**
在 norm_rope fused writer、per-32 K scale、per-token V scale 全部修完后,
重新做 27B fresh-server needle:

| P 平面数 | 表示 | 32k d10/d50/d90 | 64k d50 | 100k d50 | 结论 |
|---:|---|---|---|---|---|
| 1(P8) | 高 8 bit | 全 RETRIEVED | RETRIEVED | RETRIEVED | **生产** |
| 2(P16) | 高 16 bit | d50 过 | d50 过 | — | 冗余 |
| 3(P24) | 高 24 bit | d50 过 | d50 过 | — | 冗余 |
| 4(P32) | 32 bit | 全过 | 全过 | — | 冗余 |

补充门:

```text
27B 17×23 = 391
P8 32k depth 10/50/90 三次 fresh server 全过
P8 100k depth 50 RETRIEVED
```

因此 **P32 的“4 个 uint8 平面 + 4 个 u8s8 mma”不是质量必需**;
P8 单平面即可,PV 回到 **1 个 mma / 32 keys**。

### 0.20.2 代码改动

`cu/attn/int8_prefill.cu`:

```text
P:  p_hi/mid/lo/lo2 四平面 → 单 p_b16 平面
PV: 4 × mma.u8.s8.m16n8k32 → 1 × mma.u8.s8.m16n8k32
    acc += c * (pmax/255)
smem: 42,112B → 35,968B
```

`cu/attn/pack_v_int8_paged.cu`:

```text
prefill V pack: 每 token 一次 block_max + 2 barrier
  → 8 warp 各管一个 token 对,warp shuffle 求 absmax,
    同一 warp 写完整 uint16,零 block barrier
decode V append: 256-thread block_max
  → 单 warp(32 lane × 8 dim)warp_max + 8 维写回
```

`int8_decode.cu`:

```text
P32 四平面保留(见下);dequant 改为
  hi16 = h*256+m, lo16 = l*256+o
  acc += hi16*ps*65536 + lo16*ps
比逐平面 4 次 I2F/FMA 少一半簿记。
```

### 0.20.3 性能(3090 Ti / 27B w4a16 / ctx 131072)

kernel probe(T=2048/L=65536,3 runs):

```text
P32 四平面: 184.0 ms / 3 ≈ 61.4 ms
P8 单平面: 120.4 ms / 3 ≈ 40.1 ms   (−35%)
```

端到端(llm_speedtest,单并发,output 128,ignore_eos,nonce 强制 miss):

| prompt | P32 prefill | P8 prefill | 变化 | P32 TTFT | P8 TTFT |
|---:|---:|---:|---:|---:|---:|
| 65k | 1067.6 t/s | **1174.8 t/s** | **+10.0%** | 61.4 s | **55.8 s** |
| 100k | 916.6 t/s | **1040.2 t/s** | **+13.5%** | 109.1 s | **96.1 s** |

decode 仍走 P32 TC split,65k/100k 约 **38~39 t/s**,与 P8 单平面 decode
(39.1/38.3)在噪声内 ⇒ **decode 不做 P8**(且 P8 decode 的合成门
`tc split max|Δ|=0.0317` 超过 1e-2,收益不足,已回退)。

### 0.20.4 当前生产状态

```text
prefill: owl_int8_prefill_paged, P8 单平面
decode : owl_int8_decode_gqa_tc_split, P32 四平面
KV     : K int8 + Kscale[8]/token/head + V int8 packed + Vscale/token/head
池     : per_tok 35,072B,131k 档可用
```

质量门(最终二进制):

```text
P8 prefill + P32 decode:
  27B 32k d50 RETRIEVED
  27B 64k d50 RETRIEVED
  27B 100k d50 RETRIEVED
  17×23 = 391
```

### 0.20.5 仍未收口

1. **P8 kernel 仍 255 regs / 144B spill**;虽然 40ms 已可用,但
   qsg/acc 的寄存器压力还有下降空间。
2. **decode P32 仍是 4 mma**;P8 decode 无收益,若将来要动,
   应直接换 f16 PV/XQA,而不是再压 P 字节。
3. **per-32 K scale 的 QK 税**未单独量化;per-token K scale 的 A/B
   已试过一版(全局 max + 8 槽同值):QK 乘法减少,但 ptxas spill
   140→200B,kernel 40.1→41.3ms,已回退。说明当前瓶颈不是 scale
   乘法,别再从这条线省;要动先解决寄存器压力。
4. f16-PV 研究核仍保留但**未接生产**,且其 `v16` 语义要求
   “已反量化的 V shadow”;接之前必须补齐 vscale 或写者契约。
5. TP2(双 3090 / 双 3080)未复测 P8;3080 双卡按用户要求不占用。

### 0.20.6 双 3090 Ti TP2 复测(P8 + w4a8)

档:`profiles/measurement/qwen38_27b-3090_tp2-int8.toml`(8142),
UUID 钉 3090 对(`GPU-e565…` / `GPU-89df…`),**未碰 3080 双卡**。

质量(fresh/连续同 server):

```text
w4a8 + int8 + P8:
  8k  needle RETRIEVED
  32k needle RETRIEVED
  64k needle RETRIEVED
```

此前“w4a8 + int8 8k 丢失”是旧 writer/scale bug 的另一面;修复后
w4a8 不再单独触发质量分叉。

性能(单并发,output 128,ignore_eos,nonce 强制 miss):

| prompt | INT8 TP2 P8 prefill | decode | TTFT |
|---:|---:|---:|---:|
| 65k | **2483.8 t/s** | **58.97** | **26.39 s** |
| 100k | **2222.5 t/s** | **57.48** | **45.00 s** |

同形旧数(TP2 P32/w4a8,§0.16):65k prefill 2441 / decode 38.7;
P8 后 prefill +1.8%,decode 因后续 TC GQA 与 w4a8 叠加到 59。
单卡对照:65k 1174.8 / 38.6,TP2 约 2.1× prefill;100k 1040.2 / 38.3,
TP2 约 2.14× prefill。

**TP2 已可作为 int8 KV 的生产测试档**;下一步只剩 3080 双卡(用户占用中,
暂不测)与 131k/262k 长档压力。

### 0.20.7 65k 算子级归因(nsys + CUPTI,2026-11-03)

同卡 65k/out4 单请求,INT8 P8 vs FP8:

| 项 | INT8 P8 | FP8 | 差 |
|---|---:|---:|---:|
| Marlin W4A16 | 42.02s(75.0%) | 41.27s(67.5%) | +0.75s(同量级) |
| Attention | **10.23s(18.3%)** | **16.21s(26.5%)** | **−5.98s(1.58×)** |
| 其余 | ~3.8s | ~3.8s | ≈0 |
| GPU kernel 合计 | ~56.0s | ~61.3s | −5.3s |
| TTFT | **55.74s** | **60.94s** | **−5.20s** |

关键:

- **INT8 的收益全部来自 attention**,与 TTFT 差一一对应;
  Marlin 是固定大头(75%),两条路径相同。
- 65k prefill **GPU-bound**:GPU kernel 合计 ≈ TTFT;没有 HtoD 长尾
  (nsys HtoD 1.39s;CUPTI 的 25.34s htod 是等待/重叠假象)。
- **CUPTI 单 tag 4096 封顶**:Marlin CUPTI 只报 5.15s/4096 发,
  真值 42.0s/17,696 发;Marlin 总账必须用 nsys。
- 下一刀优先级:若压长上下文 TTFT,先看 Marlin/W4A8(75%),
  再谈 P8 kernel(18%);INT8 attention 已不是主矛盾。
- 报告:`docs/bench/int8-kv-operator-profile-2026-11-03.md`。

### 0.20.8 W4A8 复测:INT8 的长上下文优势真正拉开(2026-11-03)

单 3090 Ti,INT8 P8,同档只切 `model.quant`:

| prompt | W4A16 prefill | W4A8 prefill | W4A16 TTFT | W4A8 TTFT |
|---:|---:|---:|---:|---:|
| 65k | 1171.5 t/s | **1628.4 t/s** | 55.95s | **40.25s(−28%)** |
| 100k | 1027.9 t/s | **1490.3 t/s** | 97.28s | **67.10s(−31%)** |

W4A8 下同卡 INT8 vs FP8:

| prompt | FP8 prefill | INT8 prefill | FP8 TTFT | INT8 TTFT | TTFT 差 |
|---:|---:|---:|---:|---:|---:|
| 65k | 1584.4 | 1628.4 | 41.36s | 40.25s | −2.7% |
| 100k | 1272.1 | **1490.3** | 78.61s | **67.10s** | **−14.6%** |
| 120k | 1161.2 | **1417.2** | 103.34s | **84.67s** | **−18.1%** |

质量:INT8 P8 + W4A8,27B 8k/32k/64k/100k needle 全过。
旧“w4a8 8k 丢失”是 norm_rope/K/V scale bug,已翻案。

TP2(双 3090 Ti,W4A8)INT8 vs FP16 KV:

| prompt | FP16 prefill | INT8 prefill | FP16 TTFT | INT8 TTFT | TTFT 差 |
|---:|---:|---:|---:|---:|---:|
| 100k | 2154.8 | 2222.5 | 46.41s | 45.00s | −3.0% |
| 120k | 1763.1 | **2087.9** | 68.06s | **57.48s** | **−15.5%** |

注意:TP2 FP16 的 decode(32.5~35.5)明显低于 INT8(55.6~57.5),
但含 FI f16 decode 路径差异,不能全归因 KV;100k FP16 点偏高,待复跑。

判决:

1. **W4A8 是当前最大杠杆**,单卡 INT8 TTFT −28~31%。
2. 同 W4A8 下,INT8 对 FP8 的优势随 ctx 增长:65k 打平,
   100k −14.6%,120k −18.1%。
3. 对 FP16 KV:100k 仅 −3%,120k −15.5%;FP16 单卡 131k 装不下,
   容量也是 INT8 的硬优势。
4. 报告:`docs/bench/int8-kv-w4a8-2026-11-03.md`。

### 0.20.9 W4A8 生产化验收(2026-11-03 晚)

单 3090 Ti,W4A8 + INT8 P8,chunk 2048:

| prompt | 3 次 prefill | 中位 | 中位 TTFT | decode 中位 |
|---:|---|---:|---:|---:|
| 65k | 1763.7 / 1657.6 / 1691.4 | **1691.4 t/s** | **38.75s** | 37.77 |
| 100k | 1488.4 / 1455.3 / 1560.2 | **1488.4 t/s** | **67.19s** | 36.38 |

GSM8K n=200(thinking off/greedy/1536):**193/200 = 96.5%**,
FP8 参考 97.0%,−0.5pp,在 1pp 门内;4 题 finish=length。
needle 8k/32k/64k/100k 全过。

⇒ `profiles/measurement/qwen38_27b-3090-int8.toml` 已把
`model.quant` 从 `w4a16` 改为 **`w4a8`**。

负结果:

- `prefill_chunk=4096` 单卡启动可,但 65k 请求撞 `vram-hard-budget`
  ⇒ 2048 是 24G 天花板。
- per-token 单 Q/K scale 变体:`T=2048/L=65536` **40.7ms**,
  当前 per-32 group **40.1ms**;scale 乘法不是瓶颈,旧 f16-PV 33.5ms
  的差异在 PV/P 精度,不在 K scale。

decode 账(65k,7 步):step 25.54ms/token;Marlin 16.54ms(64.7%),
INT8 decode attention+reduce 3.39ms(13.3%),其余 ~22%。
decode 也是 Marlin-bound;P8 decode 无收益,因为 attention 是
KV 带宽/布局 bound,不是 P 字节 bound。

### 0.20.10 decode 下滑的 nparts/occupancy 探针(负结果,2026-11-03)

目标:把 INT8 decode 对 FP8 的 ~4% 下滑拉回来。先做两项不改布局的尝试。

#### (1) nparts sweep(65k,T=1)

`int8_decode_gqa_tc_split + reduce`,3 次总时:

| nparts | split | split+reduce | 每 token 折算 |
|---:|---:|---:|---:|
| 32 | 668 µs | 690 µs | 230 µs |
| 64 | 716 µs | 738 µs | 246 µs |
| 128 | 608 µs | **659 µs** | **220 µs** |
| 256 | 572 µs | 682 µs | 227 µs |

128 比 256 约 −3% attention。但端到端 100k W4A8 decode
(3 次中位)为 **36.70 t/s vs 36.38 t/s**,没有可靠收益,已回退到 256。

#### (2) 双缓冲 / 4 CTA occupancy

- 双缓冲 K/V/scale:shared 26→43KB,CTA 从 3 降到 2;
  nparts=32 从 690→668µs,但旧版 128 已是 659µs,**没有净赢**。
- 把 `s_score` 与 `s_p24` 共用 1KB,加 `__launch_bounds__(256,4)`,
  做到 64 regs / 25,088B / 4 CTA。但 4 CTA 需要 ~100KB shared
  carveout,**L1 被挤没**,split 从 0.6ms 崩到 **2.9ms**;已回退。
  3 CTA 版本(76 regs)回到 0.6~0.7ms,与基线持平/略慢,也未保留。

#### 判决

- decode 下滑不是 nparts 或 occupancy 能解决的:
  attention 仅 3.39ms / 25.54ms step(13.3%),且是 **KV 带宽/布局 bound**。
- 当前 256 nparts + 单缓冲保持生产;要再往下走,必须动
  **V 的池布局/读流水**(token-major / pthfa 式分组 load)或
  **KV 压缩**,属于独立的 decode 布局项目。探针保留在
  `int8_pipeline_probe::int8_decode_nparts_timing_65k`。

### 0.20.11 INT8 verify 接线（P1，2026-11-03）

目标：让 `spec.depth=7`（DFlash2 d7）在 INT8 KV 上跑 verify，不再被
boot/config 门拒绝；同时保持与 no-spec decode 同源，避免 verify 自己
成为新的数值/质量变量。

#### 实现

- **复用 decode 分片，不新写 attention 核**：
  `paged_verify_int8_output` 对 T=depth+1 行调用
  `owl_int8_decode_gqa_tc_split`（B=T，`bt8` 每行一条页链，`kv_lens` 逐行），
  再走 `owl_int8_decode_reduce`。
  - 分片 = **256**（与 no-spec 单 token decode 完全一致），页偶数
    `per=ceil(pages/256)`；不使用 verify v2 的 512-token 分片。
  - partial/stat 用 **f32**，直接复用 `AttnV2Scratch` 的 f32 verify
    scratch（`vattn2.tmp_out`，槽宽 = `hd+2`：partial + 交错 (m,l)）；
    T=8 时约 50MiB，一块 scratch 跨 16 层复用，图内零逐层 f32 分配。
- Q/K 走与 no-spec decode **同一条链**：
  `NormRope`（f16）→ `QuantInt8QkF16` / `K0WriteInt8KOnly`；
  不再用 prefill 的 fused `NormRopeQ/KInt8`。V 仍逐 token `VAppend`，
  并在 kernel 后加 `dep` 边把同页两 token 的 read-modify-write 串行化。
- boot 门：删除 `engine.rs` 的 `spec.depth != 0` 拒启；删除
  `shared/config` 的 `kv_int8 + spec` 语义门。
- `KvEnv::decode_v2_seq_op` 改为 `Result`：INT8 返回显式 `Err`
  （通用 v2 臂不适用），`verify_v2_arm` 收到错误时毒化，不再
  `unimplemented!` panic。
- 尝试过的新核（固定 512 分区 + f16 partial + 通用 v2 reduce）能过
  host parity（cos 0.99999998），但与 no-spec decode 分片不同，greedy
  轨迹仍有差异；未接生产，已删除，避免两套 verify 语义。

#### 验证（单 3090 Ti，W4A8 + INT8 P8，spec d7，GPU2）

| 项目 | 结果 |
|---|---|
| 短问答 | `17*23` → 391 |
| needle 8k | RETRIEVED，3.76s |
| needle 32k | RETRIEVED，16.32s |
| needle 64k | RETRIEVED，37.31s（max_seq=81920） |
| needle 100k | RETRIEVED，68.67s（max_seq=101376；`vram_reserve_mb=256` + `prefill_chunk=1024`） |
| GSM8K n=200 | **194/200 = 97.0%**（4 题 finish=length，7.6min） |

显存/上限：

- 24G 单卡 + DFlash2 d7 + f32 verify scratch 时，`max_seq=98304` 下
  64k prompt 会撞 `vram-hard-budget`（chunk 2048 需 136MiB，
  驱动 free−reserve 仅 ~10MiB）；降到 81920 后 64k 通过。
- 100k 需要约 101376 max_seq；把 `pool.vram_reserve_mb` 从 512 降到 256
  且 chunk 1024 后才够 transient。**这是单卡 24G 的容量边界，不是核错误**；
  3080 TP2 / 3090 TP2 应重新做 20G/rank 预算（P3）。
- verify scratch 改 f32 使 `vattn2` 从 18MiB → ~50MiB（T=8），
  仍在预算内；图 slab 实测 0MiB（scratch 是持久块，不进图 slab）。

#### 质量口径说明

- `greedy spec ≡ no-spec` 的“逐位”门在**当前 DFlash2 生产档本身**也未成立：
  用 FP8 3090 生产档做对照，5 条短 prompt 中 3 条 spec-on/off 文本有
  措辞差异（同一批 prompt，INT8 为 2~3 条）。差异集中在首词/近义改写，
  不是 INT8 特有；INT8 verify 已把 Q/K/分片/reduce 全部对齐 no-spec
  decode，未再引入额外的 attention 分片差异。
- 因此本轮质量判据采用：长上下文 needle 全过 + GSM8K 不劣化 + 与
  FP8 生产档同量级的 spec 文本波动；若后续要求严格逐 token，需要先
  把 DFlash2 的采样/接受链本身改成严格 greedy，而不是只改 KV 核。

### 0.20.12 P2 页解码/index/coverage/select（部分，2026-11-03）

#### KvPageCodec（trait 化）

旧 `KvPageFormat::decode_row` 只有一条 kNHD 行式，INT8 V（dim-major
packed b16）会被当成 K 行读错；K 也没有乘 Kscale。P2 把它改成：

```text
trait KvPageCodec {
    elem_bytes / page_bytes
    k_scale_elems / v_scale_elems      // f16/fp8 = 0
    k_row(page, KvRowScale, t, h, ...) // INT8 = code × Kscale[group]
    v_row(page, KvRowScale, t, h, ...) // INT8 = packed b16 × Vscale[token]
}
```

- `KvPageFormat::codec()` 是唯一的格式 switch；调用点只拿
  `&dyn KvPageCodec`。
- `KvRowScale` 是 page-local 的 `&[f32]` K/V scale 视图；f16/fp8
  传空切片。
- INT8 K/V 单测：`int8_k行乘scale` / `int8_v行解packed`，以及原有
  f16/fp8 逐位门全部绿。
- `index`/`coverage` 回读时按页把 **K 页 + Kscale +（可选）V 页 +
  Vscale** 一起 D2H；scale 字节一次转 f32，再按 `(block,layer)` 切片。
- `index_gpu` 只有 f16/fp8 Mean-K 核；INT8 暂时走 host 回读（一次
  分支，不是调用点散落）。

#### 选择层/门控

- `index` / `coverage_probe` / `select_decode` 对 INT8 放行。
- `select_decode` 纯文本 8k needle：`ZXCVBNM12345` RETRIEVED（4.0s），
  `view` 页表/页链与 INT8 decode split 正常配合。
- coverage-only 8k：needle RETRIEVED；覆盖率探针输出 top1/top8 曲线
  （top1 10.8% @8k，与 FP8 11.4% 同量级）。
- 仍拒：
  - `select_partial`：INT8 稀疏 prefill 的子集页链/因果口径未验收。
    8k 实测打开后 prefill 后段读错页（答成复读吸引子），config/engine
    双层拒启。
  - `image_evict`：INT8 视觉/m-RoPE prefill 未验收。同图 FP8 12/12
    命中，INT8 坐标全错/退化；不是 `pool_layout` 页搬移的问题，已拒启。

P2b 入口：INT8 sparse prefill（`fi_plan_ctx` 视图口径已作为前置接进
`paged_prefill_int8_output`，但还未通过选择层质量门）与 VLM/m-RoPE
INT8 prefill。

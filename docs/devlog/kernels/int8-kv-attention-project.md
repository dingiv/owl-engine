# INT8 KV × Attention 专项设计（2026-11-03）

> 状态：**设计冻结，待 M0 施工**。
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
| 配置 | `crates/engine/src/kernel_cfg.rs` / `profiles/*` 新增 `kv_format`、`int8_backend` |
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

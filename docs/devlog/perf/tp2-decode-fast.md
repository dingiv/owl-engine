# TP2 双 3080：decode 快档 + W4A8 定档（2026-10-22）

> 目的：把 `feat/w4a8-p1` 工作树里“TP2 decode 档面体检 / W4A8 / marlin 测量修正”
> 三件候选在当前 `main@feb0131` 上**逐项重测**，只把真赢的配方转正。
> 本文所有数字 = 同一二进制、同一对 3080、同一模型、同一 prompt（固定 nonce），
> 单变量 A/B；不再拿不同日期/不同口径的旧读数直接对比。

## 0. 结论（先看这八条）

1. **FI 单图是 decode 的主杠杆**：v2 裸 decode `46.2` → FI 单图 `56.6`（**+21.7%**）；
   再叠 DFlash2 d7（`tp_draft`）后 std 五点三轮 decode **138.1 t/s @262k**
   （184k 同量级 `139.8`；p50 138.1，范围 82.7–202.8；逐 prompt AL 波动大）。
2. **`tp_draft` 是 TP2 spec 的硬门**：主干的 TP2 档此前 `depth=7` 却没有
   `tp_draft=true` —— `tp.rs` 会在**装载完权重之后**响亮拒启（不是配置期）。
   本轮已在两个 TP2 生产档补上。
3. **`head_quant=rtn-int8-row` / `tp_vocab_split` 在 spec 档是负收益**：
   `@512/@2048` 实测 int8 头 `92.2/60.3`、词表切 `139.8/85.5`、两者+融合全开
   `106.0/68.8`，均低于“int4 头 + 全表”的 `165.2/99.6`。它们只对非 spec 档
   有账面收益；**不要在 spec 档叠**。
4. **`tp_ar_fused` 中性偏好**：`169.4/101.7` vs `165.2/99.6`（quick 两点），
   在噪声带内；且字段注写明“只非 fp8 载荷”，本档 `tp_ar_f8=true` 时语义不明确
   ⇒ 生产档先不开。
5. **W4A8 是 prefill/RAG 档，不是 decode 档**：std 五点三轮 @262k prefill
   `1616.8 → 2342.7`（**+44.9%**）、TTFT@8192 `5068.3 → 3421.2 ms`（**−32.5%**），
   decode `138.1 → 82.1`（**−40.5%**；m=1 权重带宽墙 + 激活量化 + AL 下降）。
   184k 同量级 `+47.4% / −47.4%`。⇒ 单独成档 `…-tp2-w4a8`，按负载选。
6. **草稿池 ring（2026-10-22）**：草稿 config `sliding_window=2048`，NC 核本就只读
   最近 2048 行；但池按 `max_seq` 全长分配（10240B/token，184k = 1800MiB）。
   改 ring（2080 slots = 20MiB）后 `per_tok 26624→16384`、`[ctx-max] 322k→442k`；
   **262k 档 std 与 184k 持平**（138.1/1616.8），4096 点 `90.4`（ring 前 91.4）。
7. **W4A8 decode 下降不是固定内核税**：nsys 单轮 `+1.0ms`（quant 0.91 + marlin/attn），
   但 AL `5.00→4.91@512 / 3.78→3.35@2048 / 2.99→2.20@4096`；decode 差 = AL + verify
   税。草稿 KV fp8 **不改变 AL、propose ×1.9 慢**（只省显存），不要再开。
8. **W8A16 head 已上 f16 Tensor Core（2026-10-22 深夜；Marlin 同款流水）**：
   HBM 读 int8 权重、B 在 shared 仍 int8、MMA fragment 在寄存器里
   `int8 → half`，激活保持 f16。核级 7/8 tok **0.980/0.984ms**
   （旧 FMA 1.451/2.070；W8A8 TC 0.964/0.969）。引擎 spec 同门
   round 36.65→**35.28ms**、propose 6.26→**5.97ms**、verify 28.62→**27.45ms**，
   **与 W8A8 TC 基本同速**，但 f16 vs i8 仍是 **0.9896（过门）**。
   ⇒ 默认 W8A16 已吃 TC 速度；W8A8 被**同速更差质量**支配，退回诊断档。

## 1. 口径

| 项 | 值 |
|---|---|
| 二进制 | `main@feb0131` + 本轮 FI/decode 快档/ring 改动 + `cargo build --release -p server --features perf` |
| 卡 | 2× RTX 3080 20G（TP2） |
| 模型 | Qwen3.8-27B AWQ-INT4 + DFlash2 d7（`syvai/...-W4A16`），fp8 KV，**262k 池**（ring 后；184k 为 ring 前基线） |
| 客户端 | `llm_speedtest`(外部存档)；`usage.completion_tokens` 口径；输出 256 tok |
| prompt | `--prompt-domain math`，`--prompt-seed 20261019`，**固定 nonce `owl-tp2-ab`**（跨臂同 prompt，sha 逐点相同） |
| 温度 | greedy（`temperature=0`, `top_p=1`, `--ignore-eos`） |
| std | 五点三轮：512/1024/2048/4096/8192 × 3；quick = 512/2048 × 1 |
| 单变量 | 每臂只改一个维度；同一进程只起一张 27B server；每臂后等显存回落到 <300 MiB |
| 读数 | 优先 `avg_output_speed` / `avg_prefill_speed` / `avg_ttft_ms`（客户端口径），辅以 `/debug/metrics/decode` |

> ⚠️ spec 档 decode 的 AL 随 prompt 内容波动，**单点不可作判据**；本文以
> std 三点中位 + 三点范围同时给出。

## 2. 臂矩阵（quick 两点，math）

共同底座：`qwen38_27b-3080_tp2-w4a16`（184k / fp8 / int4 头 / 无词表切），
除 A0 外都 `tp_draft=true`。

| 臂 | 变体 | decode @512 | decode @2048 | prefill @2048 |
|---|---|---:|---:|---:|
| A0 | v2 注意力，**spec off** | 46.34 | 46.56 | 1639.8 |
| A1 | v2 + spec d7 | 146.04 | 81.54 | 1605.8 |
| E | **FI 单图**，spec off | **56.53** | **56.56** | 1644.7 |
| **B** | **FI 单图 + spec d7** | **165.22** | **99.58** | **1604.7** |
| b1 | B + `head_quant=rtn-int8-row` | 92.24 | 60.33 | 1603.5 |
| b2 | B + `tp_vocab_split=true` | 139.79 | 85.49 | 1603.0 |
| b3 | B + `tp_ar_fused=true` | 169.44 | 101.65 | 1599.6 |
| C | b1 + b2 + b3（分支原配方） | 106.03 | 68.75 | 1605.6 |
| D2 | **B + `quant=w4a8`** | 170.28 | 61.89 | **2382.1** |

读法：

- `A0 → E`：**纯 FI 单图的账 = +21.7%**（无 spec 干扰，最干净的杠杆）。
- `A0 → B`：**v2 裸 → FI+spec = 3.0×**（std 口径 46.2 → 139.8）。
- `B → b3`：融合 AR 在噪声内；`B → b1/b2/C`：int8 头/词表切在 spec 档明显回退。
- `B → D2`：W4A8 的 prefill 收益很大，但 decode 在 2048 点掉到 61.9；std 见下。

## 3. std 五点三轮（math；同 prompt、同二进制）

| 臂 | prefill 中位 | decode 中位 | decode 均值 | decode 范围 | TTFT@8192 |
|---|---:|---:|---:|---:|---:|
| A0（v2 + 无 spec） | 1646.9 | **46.43** | 46.20 | 44.79–46.99 | 4966.6 ms |
| **B（FI + spec d7）** | 1619.3 | **138.87** | 139.79 | 83.78–204.49 | 5060.6 ms |
| D2（W4A8 + FI + spec） | **2386.3** | 73.05 | 83.01 | 44.86–172.28 | **3405.2 ms** |

B 的逐点中位：

| prompt | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---:|---:|---:|---:|---:|
| prefill (t/s) | 1593.9 | 1619.3 | 1664.3 | 1638.1 | 1598.1 |
| decode (t/s) | **168.6** | **154.7** | 98.1 | **166.8** | 100.3 |
| TTFT (ms) | 336.9 | 655.2 | 1219.0 | 2470.6 | 5060.6 |

B 的 decode 在 2048/8192 点明显低于 512/4096 —— 同一 prompt 集里这两点的 AL
偏低（不是 ctx 项：`propose.read` 未随 8192 线性增长）。这是 spec 档的
**prompt 域敏感**，不是档位回退。

### 3.1 262k std（草稿池 ring 后；同口径）

| 档 | prefill 均值 | decode 均值 | decode 范围 | TTFT@8192 |
|---|---:|---:|---:|---:|
| `…-tp2-w4a16`（262k） | 1616.8 | **138.1** | 82.7–202.8 | 5068.3 ms |
| `…-tp2-w4a8`（262k） | **2342.7（+44.9%）** | 82.1（−40.5%） | 44.2–168.9 | **3421.2 ms（−32.5%）** |

⇒ 262k 与 184k 的 std 基本持平（W4A16 139.8→138.1 在噪声带内）；ring 让
**上下文从 184k 免费涨到模型原生 262k**，decode/prefill 不付代价。

### 3.2 W4A8 decode 归因（nsys + metrics，2026-10-22）

**nsys**（63-token prompt / 64 输出；`--cuda-graph-trace=node`；单轮 = 一次图回放）：

| 项 | W4A16 | W4A8 | 差 |
|---|---:|---:|---:|
| marlin 小计 | 12739 µs | 12849 µs | +110 |
| `owl_quant_int8_row_f16` | — | **912 µs** | +912 |
| AR（stage+reduce） | 4110 µs | 4104 µs | −6 |
| GDN recurrence | 1214 µs | 1209 µs | −5 |
| 设备合计 | 20932 µs | 21957 µs | **+1025（+4.9%）** |
| span（每轮） | 20.99 ms | 23.29 ms | +2.30 ms |
| 核数/轮 | 1756 | 2108 | +352 |

**metrics**（同 prompt nonce；`spec.rounds/tokens/verify` 差分）：

| ctx | 档 | AL | round | verify | propose | decode |
|---:|---|---:|---:|---:|---:|---:|
| 512 | W4A16 | **5.000** | 29.65 ms | 21.76 | 5.86 | 167.7 |
| 512 | W4A8 | **4.907** | 31.15 ms | 23.54 | 5.74 | 170.7 |
| 2048 | W4A16 | **3.775** | 32.38 ms | 21.81 | 8.32 | 115.2 |
| 2048 | W4A8 | **3.354** | 33.89 ms | 23.51 | 8.24 | 101.1 |
| 4096 | W4A16 | **2.989** | 32.43 ms | 21.73 | 8.37 | 91.4 |
| 4096 | W4A8 | **2.202** | 34.07 ms | 23.52 | 8.24 | 65.5 |

定谳：

- **不是内核固定变慢**：marlin 在 m=1 没有变快（W4A8 是 `s8_u4` 11.47ms + 1.05ms，
  W4A16 12.39+0.34），只多 **0.91ms 激活量化**；AR/GDN 相同。
- **主因是 AL**：512 点 AL 几乎相同（5.00 vs 4.91）⇒ decode 持平；2048 AL −11%
  ⇒ decode −12%；4096 AL −26% ⇒ decode −28%。每轮 verify 固定 +1.7ms 放大了它。
- `spec.round.deg=0` / `degrade=0`：**没有降级**，全是 spec 轮；所以不是 B4 降级。
- **草稿 KV fp8**：AL 不变（2.989/2.220），propose 8.37→15.88 / 8.24→15.65ms
  （×1.9）⇒ 只省显存（184k 省 900MiB），**不是 decode 修复**。
- 社区对照：SGLang 的 DFlash 草稿窗口/`adaptive_spec_params`、vLLM DFlash2 的
  lookup + 自适应 verify 长度都在治“长 ctx AL/verify 税”；owl 的 NC 核已有
  2048 草稿滑窗，缺的是 **草稿池 ring**（本轮已补）与 **自适应 verify 长度/
  lookup**（未做）。

## 4. 移植物

### 4.1 `profiles/qwen38_27b-3080_tp2-w4a16.toml`（已改）

```toml
[spec]
depth = 7
tp_draft = true

[kernel]
decode_arm = "fi"
verify_arm = "fi"
fi_plan_bake = true
fi_plan_restore = false
fi_cuda_graph = true
```

保留：`head_quant = "rtn-int4-g32"`、`tp_vocab_split` 不写、`tp_ar_fused` 不写。
**262k**：`max_seq = pool_tokens = 262144`(圆顶 = 模型原生上限；ring 后实测可服务)。
头注写入了本轮 A/B、ring 与“不要再叠 int8 头/词表切”的警告。

### 4.2 `profiles/qwen38_27b-3080_tp2-w4a8.toml`（新增）

只 `extends` 上述 W4A16 档 + `[model] quant = "w4a8"`；其余完全继承
（含 FI 单图 / `tp_draft` / **262k** / 草稿 ring / int4 头 / 无词表切）。
头注写明“prefill 档，decode 回退”，避免被当默认档。262k std:
prefill 2342.7(+44.9%) / decode 82.1(−40.5%) / TTFT@8192 3421ms(−32.5%)。

### 4.3 未移植（明确记录）

| 候选 | 判决 | 依据 |
|---|---|---|
| `[spec] method = none\|dflash2\|mtp` | **不采纳** | 与 main 的 `[spec] draft = auto` 是两套 schema；main 已有显式 `draft=mtp/dflash2` 覆盖，迁移会动所有 profiles/resolve/tests，价值不足 |
| `head_quant=rtn-int8-row` + `tp_vocab_split` 快档 | **不采纳** | spec 档实测回退（§2 b1/b2/C）；非 spec 档的 +9.5%/+3.4% 另案，不与 spec 混用 |
| `tp_ar_fused=true` | **暂缓** | quick +2% 在噪声内；字段注写“只非 fp8 载荷”，与 `tp_ar_f8=true` 的语义待定 |
| `marlin_host.cu` thread_local + 窗口均值 | **已接**（探针-only） | main 当前 `static PhaseAcc` 在 TP2 双 rank 同进程下混算；已改 `thread_local` + 每 256 条窗口，见 §5 |
| 分支的 VLM 262k 快档 | **已收编** | ring 后 262k 可服务；`-vlm` 档改为主档兼容名，不再保留 int8 头/词表切，见 §6 |

## 5. marlin 测量修正（已接，探针-only）

`feat/w4a8-p1` 的 `d2b1277` 把 `marlin_host.cu` 的 `static PhaseAcc g_phase`
改成 `thread_local`，并把“从进程起算的累计均值”改成“每 256 条窗口重算”。

为什么必须改：TP2 是**同一进程双 rank 线程**，文件级 static 会把两个 rank 的
发射耗时混在一起；于是“`kernel<<<>>>` 随发射数单调上升”看起来像驱动回退，
实际是两 rank 混算 + 累计平均的伪曲线。分支实测（修正后，单 rank 逐窗）：

```text
launch 逐窗(µs): 4.2  4.5  4.6  64.2  31.0  82.5  2.9  395.3
attr   逐窗(µs): 3.5  0.1  1.4   0.1   0.1   0.8  0.1    0.1
cfg    逐窗(µs): 0.7  0.8  1.0   1.0   0.8   0.7  0.6    0.6
```

⇒ 前 ~768–1024 条自由入队（~4.5 µs/条），之后 `cuLaunchKernel` 阻塞到 GPU
排空一个槽（30–90 µs，最紧 ~400 µs）—— CUDA pending-launch 队列被灌满后的
驱动回压，**不是可优化的宿主代码**。本修正只改探针（`MARLIN_TIME=1` 才生效），
不动生产路径；已保留 main 的 `OWL_LOG_DEBUG`（没有回退成 `fprintf`）。

**顺带修了构建闸的一个真 bug**：`crates/kernels/build.rs::build_marlin` 原先把
源文件的 `cargo:rerun-if-changed` 只写在“重建”分支里 —— 一旦 `libmarlin.a`
stamp 命中，改 `marlin_host.cu` 不会触发 build.rs 重跑，旧 `.a` 静默留用
（“改了探针但读数不变”的根因）。现已把源文件登记移到 stamp 检查**之前**；
本轮的 `.cu` 修正是靠 `MARLIN_FORCE_BUILD=1` 首次编入的，之后普通增量即可。

## 6. VLM 262k：ring 后已可服务（历史失败记录保留）

ring 前（1800MiB 草稿池）的失败现场：

- `vram_reserve_mb=256`：池 `19383 MiB`、驱动 free `242–274 MiB`；请求级
  5/30 MiB 瞬态被 `vram-hard-budget` 拒发（512 出 0 token；2048 170s 失败）。
- `vram_reserve_mb=64` 能服务 512/2048，但每步刷 N6 压力；8192 未测。

ring 后（草稿池 20MiB，`per_tok 16384`）：

- `qwen38_27b-3080_tp2-w4a16-vlm.toml` 改为**主档兼容名**（`extends` 主档；
  主档已 262k + 视觉 + FI 单图 + `tp_draft` + ring），不再保留 int8 头/词表切。
- 262k 实测：512/2048 decode **165.7/100.7**、prefill **1585**；std 十五点
  decode 138.1 / prefill 1616.8（与非 VLM 主档持平）。**262k 视觉档现在
  与主档同配方、同性能**；旧 int8/词表切版本（decode 86）已退役。
- 残留：真正的**图像请求** smoke 未在本轮跑（文本 262k 已过）；图像 × 长 ctx
  的显存/正确性需另立门。`vram_reserve_mb=256` 维持（ring 后余量足够）。

## 7. 复现

```bash
# 同一二进制
cargo build --release -p server --features perf

# 卡位/模型按站点改;下面是 2×3080 的示例(--set 只是覆盖层)
./target/release/server -p profiles/qwen38_27b-3080_tp2-w4a16.toml \
  --log-level info

# 客户端(另一个终端)
cd <llm_speedtest>
uv run cli --api-url http://127.0.0.1:<port>/v1/chat/completions \
  --model qwen3.8-27b --flavor std --rounds 3 \
  --output-length 256 --ignore-eos --temperature 0 --top-p 1 \
  --prompt-domain math --prompt-nonce owl-tp2-ab
```

### 7.1 转正档冒烟（用**实际 profiles 文件**跑，非 /tmp 覆盖档）

| 档 | 启动 | 现役臂自证 | 512 点 prefill | 512 点 decode | TTFT |
|---|---|---|---|---|---|
| `…-tp2-w4a16.toml` | ✅ 32 s | `verify=fi / decode=fi / spec=depth7` | 1581.1 | 167.6 | 336.2 ms |
| `…-tp2-w4a8.toml` | ✅ 37 s | 同上 | 1994.0 | 169.9 | 266.9 ms |

> 冒烟就是 §2 的 B / D2 臂（同 profile 内容、同二进制、同 prompt）；
> 这里验证的是 **profiles 文件本身**（相对 `../models` 路径、`tp_draft`、
> FI 单图标志）能被引擎接受并跑出同一量级。

## 8. 遗留

1. **图像请求 smoke**：262k 文本已过；图像 × 长 ctx（视觉塔输出/快照/段边界）
   仍未跑。`-vlm` 现在是主档兼容名，图像路径的显存门要在主档上补。
2. **自适应 verify 长度 / lookup**：
   - **lookup 已实现并转正**（§10；copy AL 5.91→7.00，prose +10.6%）；
   - 自适应 verify 的 d3/d7 实测仅 ~1%（32k 固定尾部：AL 1.476 vs 1.602、
     轮时 32.09 vs 34.12 ms）⇒ 暂不转正；长块 tail 需配合更大的 verify 图族。
3. **W4A8 spec 调参**：本轮 W4A8 用 d7；长 prompt + 长生成可试 d3/自适应，
   把 decode 回退收窄（prefill 收益不受影响）。
4. **采样域（temp=1.0）未测**：本轮全部 greedy；生产 agent 档是采样。
5. **草稿 ring 的 per-session 分区**：当前 `gdn_slots=1` 线性 ring；B5 合批
   （多 turn 共享逻辑空间）需 per-session ring 基址（与既有 slot 案同族）。

## 9. 草稿池 ring + 账本修正（本轮实现，2026-10-22）

### 9.1 为什么是 ring

DFlash2 草稿检查点 `config.json` 的事实：

```text
layer_types = [sliding_attention] × 5
sliding_window = 2048
use_sliding_window = true
```

NC 核早就按 **2048 滑窗**读前缀（`OWL_NC_SLIDING_WINDOW`，自块恒全可见），
但草稿池仍按 `max_seq` **全长线性分配**（10240B/token/rank）：
184k = **1800 MiB**，262k = **2.56 GiB**，而核永远只读最后 2048 行。
⇒ 这是上下文上不去、显存对不上账的最大单项。

实现（`window 2048 + block 8`，页对齐；页 32 ⇒ **2080 slots = 65 页 = 20MiB**）：

| 面 | 改动 |
|---|---|
| `models/layers/dflash2.rs` | `DRAFT_SLIDING_WINDOW=2048` / `DRAFT_BLOCK=8` / `draft_ring_slots(page)`；`DfAttn.ring_cap` + `DFlash2Draft::with_ring_cap` |
| `kernels/cu/attn/naive_attn_nc.cu` | 全部 NC 变体加 `int ring_cap`；前缀读 `slot = srow % ring_cap`（0 = 线性）；核内 `__trap` 守 `ring_cap >= 2048 + T` |
| `kernels/src/ops/attn.rs` | 9 张 NC 表 `PARAMS` 加 `ring_cap`（槽序测试同步） |
| `engine/state.rs` | `DraftBank.ring_slots`；`draft_dflash_{encode,self}_slot`；`PerTokBytes` 的 DFlash2 草稿项归 0；`FixedAfter.draft_ring`；`DraftBank` 按 `ring_nb` 分配 |
| `engine/spec_runtime.rs` | 按 `base_env.kv.page` 算 `ring_cap`，装载后 `d2.with_ring_cap(...)` |
| `engine/exec/draft.rs` | graph/encode/propose/prefill-encode 四路槽号全走 ring |

### 9.2 账本修正

- **`pool_cap` 改用驱动 free**（`base = min(free, free − total×(1−target))`），
  不再用引擎账 `live` 近似（漏掉 CUDA ctx / FI ws / allocator 惰性 reserved /
  碎片，实测差 ~2–3 GiB）。`live` 仍打印但标“引擎账；仅参考”。
- `FixedAfter` 增 **草稿 ring** 固定项；`PerTokBytes` 只留主池 + MTP 链。
- `[ctx-max] 池落账` 改为 **主池 + 草稿池(MTP 链/DFlash2 ring)** 全量。

**效果（W4A16 TP2，同一档）**：

| 项 | ring 前 | ring 后 |
|---|---:|---:|
| `per_tok` | 26624B | **16384B** |
| 草稿池 | 1800 MiB | **20 MiB** |
| `fixed_after` | 0.59G | **0.61G**(+ring 20MiB) |
| `[vram-gov] budget` | 8.00G（虚高） | **6.74G（free 实测）** |
| `[ctx-max] cap` | 322496 | **441584** |
| 262k std decode / prefill | 不可服务(262k) | **138.1 / 1616.8** |
| 4096 decode | 91.4(184k) | **90.4(262k)** |

GGUF 192k MTP 档也复验：`[ctx-max] cap=275156`、192k 正常、512 点
decode 118.5 / prefill 1419.8（MTP 链 384MiB 在 `[ctx-max] 池落账` 里以
“草稿池(MTP链/DFlash2 ring)”列出）。

### 9.3 门与验证

- `cargo test -p owl-kernels --lib`：`every_compiled_symbol_is_declared` /
  `every_declared_op_has_asset_and_symbols` / `knhd_variants_mirror_classic_signature` /
  `args_derive_from_params` 全绿；`nc_v2_slot_order_derives_from_params` 更新
  为 12 参（含 `ring_cap`）后绿。
- 真机：`main262-std`（15 点）与 `w4a8-262-std`（15 点）；ring 前 4096
  91.4 → 后 90.4（同 prompt/同卡，噪声级）；512/2048 165.2/99.6 → 165.7/100.7。
- VLM 262k：ring 后 512/2048 正常（165.7/100.7；int8/词表切旧版 104/68 已退役）。
- **恒等门** `gpu_dflash2_identity_gate_27b`：**OK**（prose AL 1.25；spec 93 tok
  与 baseline 逐字相同）；**VLM 图首轮** `gpu_vlm_first_turn_probe_27b_imagefirst`：**OK**
  （四轮带图/纯文）。
- 单元面：engine lib **130/130**、models lib **301/301**、kernels lib **118/118**；
  `nc_v2_slot_order` 更新为 12 参后绿；`profiles` 完整度门 2/2。
- 核内硬门：`ring_cap != 0 && ring_cap < 2048 + gridDim.x` ⇒ `__trap()`
  （不允许静默错槽）。

### 9.4 剩余

- **图像请求 smoke**（262k 文本已过；视觉塔 × 长 ctx 未跑）。
- **自适应 verify 长度 / lookup**（vLLM DFlash2 两条；见 §8）。
- **per-session ring 基址**（B5 合批多 turn；当前 `gdn_slots=1` 线性 ring）。

## 10. DFlash2 lookup 草稿（LABD，2026-10-22 实现并转正）

### 10.1 为什么先做 lookup

本轮对 AL 的定位结论（详见 §11 的对照）：

- AL 不是 ctx 的单调函数。**旧 harness 的“唯一前缀”组（3.33/1.79/2.33/1.60）
  已作废**：`prose-zh.txt` 只有 50,301 字符，旧脚本 `src[60000:68000]`
  取到空 tail ⇒ 实际不是固定尾部，只是不同前缀。修正 harness（重复源 +
  真 8K tail）与 vLLM 同 prompt 对齐见 §10.8。可复现的结论是：在固定填充
  尾部上，草稿窗 2048→4096 没有恢复 AL，propose 反而 +40%；
- 自适应 verify（d3 vs d7）在 32k 固定尾部上 AL **1.476 vs 1.602**、
  轮时 **32.09 vs 34.12 ms**，折算 decode 仅 **~+1%**；target verify 是
  权重带宽主导，少 verify 几行几乎不省钱。⇒ 不是第一优先；
- W4A8 不是 AL 主因：16k 有一次 `rounds=10 / AL=0` 是 B4 降级后的无效
  样本，健康复跑 AL **2.739**；32k 健康样本 **4.12**。W4A8 decode 回退
  是 target 的 int8 激活/marlin 成本，不是 AL。

⇒ 先做 **lookup**：它直接抬 copy 类长上下文的 AL，且不需要新 kernel。

### 10.2 实现（vLLM DFlash2 同式，head-only）

参考：`packages/vllm/vllm/v1/worker/gpu/spec_decode/dflash2/lookup.py`
（`suffix_lookup` + `_fuse_draft_kernel` 的 head 分支）。

- **历史**：`prompt_ids + out + 本轮 verify 块`（`out` 末位 = 当前 token；
  轮末 tail = `ids[0..=m]`，末位 bonus 是下一轮 anchor）。首轮 tail 为空。
- **查找**：在历史里找「当前后缀」最近一次出现；先比 `nmin` 个 token，
  命中后向更早扩到 `nmax`；最长优先、等长取更近；允许与后缀重叠
  （周期重复）；返回匹配点之后 `min(k, len-1-e)` 个 token。
- **融合**：`match_len >= nstrong` 直接覆盖草稿头；或
  `match_len >= nmin && agree >= agree_min`；`agree` = 草稿与 lookup
  的前缀一致长度（按 valid 截断）。缺省 `nmin=nstrong=6` / `nmax=12` /
  `agree=0`，与 vLLM 缺省同。
- **范围**：只覆盖草稿器自己的 7 个位置（verify 图仍 depth+1 定形）；
  **不做 tail 长块**（需要更大的 verify 桶，另案）；greedy spec 无需
  point-mass 重写（草稿仍由 target 逐位 verify）。

落点：

| 文件 | 改动 |
|---|---|
| `crates/engine/src/exec/lookup.rs` | **新增**：`LookupCfg` / `LookupProposal` / `suffix_lookup` / `fuse_lookup` + 7 个单测 |
| `crates/engine/src/exec/draft.rs` | `dflash_lookup_for_tail`（历史/查找/`spec.lookup.*` 指标）、`apply_dflash_lookup`；graph/eager 两路融合 |
| `crates/engine/src/exec/spec.rs` | 轮末 `toks` 作为 tail；首轮空 tail |
| `crates/shared/src/config/mod.rs` | `[spec] lookup` / `lookup_nmin|nmax|nstrong|agree` + 代数门 |
| `crates/engine/src/engine.rs` | `DraftCfg` / `SpecBootInput` 接线 |
| `crates/engine/src/spec_runtime.rs` | `SpecRuntime` 字段 + **仅 DFlash2** 的 boot 门 + 日志 |
| `crates/engine/src/tests.rs` | `OWL_GATE_LOOKUP` 恒等门臂 |
| `profiles/full.toml` / `profiles/qwen38_27b-3080_tp2-w4a16.toml` | 缺省关；TP2 主档转正 `lookup=true` |

### 10.3 实测（2×3080 TP2，W4A16，greedy）

**A. copy 任务**（随机 4-hex 词串，80 组；要求逐字重复，`ignore_eos`，
输出 300 tok；同二进制/同 prompt/同卡）：

| 档 | AL | tokens/round | rounds | round_ms | 折算 decode |
|---|---:|---:|---:|---:|---:|
| lookup off | 5.907 | 6.907 | 43 | 28.98 | ~238 t/s |
| lookup on | **6.641–7.000** | **7.641–8.000** | **38–39** | **27.96–28.00** | **~273–286 t/s** |
| Δ | **+12.4–18.5%** | +10.6–15.8% | −9.3–11.6% | −3.5% | **+15–20%** |

lookup on 的 `spec.lookup.hit=38–39/39–40`、`m7=37–38`（几乎全部 7 草稿
被接受）。**range 的来源**：同一 prompt/同 seed 的两次 boot 在最后一块
（输出预算边界）差一轮，AL 由 6.641 变 7.000；内容逐字节相同，不是 lookup
行为差异。`spec.lookup` 宿主耗时 **5.5 µs/轮**（371 token 历史）。

**B. 非 copy 回归（旧 harness 的“固定尾部”tail 为空，已作废）**。

旧表 `lookup off 1.688 / on 1.867` 不能再用作 lookup 的 prose 收益证据。
修正 harness 后的 32K fixed-tail：owl lookup on **AL 3.213 / mean 4.213 /
acceptance 45.9%**，同 prompt vLLM **mean 3.864 / acceptance 40.9%**
（§10.8）；lookup off 尚未在修正 harness 上重测。lookup 的 **copy** 收益
仍然有效（§10.3 A、§10.5 恒等门）。

host 查找在 propose 图发射前执行，`spec.lookup` 计时可单独观察（371 token
历史实测 **5.5 µs/轮**；32k 的轮时增量在噪声带内）。

**C. 恒等门**：`OWL_GATE_LOOKUP=1 gpu_dflash2_identity_gate_27b` **OK**
（单 3090 Ti，d7，copy prompt；baseline 209 tok 与 spec 逐字相同，AL 4.50）。
第二次把 `OWL_GATE_MAX_NEW` 抬到 120：baseline/spec 均 **278 tok 逐字相同**，
spec AL **4.80**（lookup 命中的草稿也全部经 target verify，恒等律不破）。

**D. W4A8 备注**：W4A8 档继承 `lookup=true`，但同一 copy prompt 上
W4A8 target 自身输出退化（`1\n41\n1…`），AL 2.79、lookup hit 60/79；
这不是 lookup 的回归（lookup 仍只覆盖草稿头，target 逐位 verify），
而是 W4A8 target 在该 prompt 上的精度/退化问题。W4A8 的 lookup 收益
需在它自己的健康域（长 prompt/RAG）另测。

### 10.4 配置与回退

```toml
[spec]
lookup = true          # TP2 主档已开
lookup_nmin = 6
lookup_nmax = 12
lookup_nstrong = 6
lookup_agree = 0
```

回退：`--set spec.lookup=false`。非 DFlash2 档写 `lookup=true` = boot 拒
（不静默忽略）。

### 10.5 限制 / 下一刀

1. **head-only**：只覆盖 7 个草稿位；vLLM 的 tail 长块（copy 可一次填满
   16/32 个 verify 位）需要更大的 verify 图族，收益主要在长 copy；
2. **采样档**：当前 spec 轮恒 greedy，未做 point-mass/rejection 重写；
3. **阈值**：缺省跟 vLLM（`nmin=nstrong=6`）；`lookup_agree` 已留但未调；
4. **自适应 verify**：本轮 d3/d7 实测仅 ~1%，暂不转正；若将来做长块
   tail，需要配合 `next_num_draft_tokens` 式长/短块切换。

### 10.6 草稿精度 A/B（BF16 vs W4A16，负结果）

社区 Q8 draft > Q4 draft 的 3090 实测（GGUF Q4_K_M → Q8_0，160K coding
acceptance 72.2%→77.2%）提示“4-bit 草稿可能是长上下文 AL 瓶颈”。本轮在
TP2 上用 **z-lab BF16 草稿**（`models/z-lab/Qwen3.8-27B-DFlash2`，
`draft-mem` ~2.25GB）对 **syvai W4A16**（~0.64GB）做同二进制/同 prompt
A/B；lookup 同开：

| 任务 | W4A16 | BF16 | Δ |
|---|---:|---:|---:|
| copy AL | 6.641 | 6.641 | 0 |
| copy round_ms | 28.00 | 30.03 | +7.2% |
| copy propose_ms | 4.75 | 6.98 | **+47%** |
| 32k 固定尾部 AL | 1.867 | 1.910 | +2.3%（噪声） |
| 32k round_ms | 33.92 | 36.05 | +6.3% |
| 32k 折算 decode | ~84.5 | ~80.7 | **−4.5%** |

⇒ **BF16 草稿不抬 AL，propose 反而慢 ~47%**；4-bit W4A16 不是长上下文
AL 的瓶颈。两个口径差异要记住：

- 社区 Q8>Q4 的“Q4”是 **GGUF Q4_K_M**，其 4-bit 质量与 AWQ/marlin W4A16
  不是同一东西；我们的 W4A16 已经是较高精度的 4-bit 路线；
- `lued/Qwen3.8-27B-DFlash2-W8` 模型卡自述 **W8 vs BF16 acceptance 相等**
  （2.65 vs 2.72/draft），所以 W8 也不是“精度升级”；它只省显存。
  且我们 loader 的量化草稿分支只有 `QuantPlan::W4A16`（W8 会被误当 W4A16
  读），要试 W8 需先加 W8A16 plan/kernel —— 在 BF16 已证明无 AL 收益的
  前提下不值得。

**结论：草稿精度路线关闭，生产保留 W4A16。**

### 10.7 200K 实际上下文的对照（2026-10-22 晚；回应“vLLM/SGLang 200K 稳得住”）

社区在 200K 附近的“稳”有两个口径，不能混：

1. **上下文容量**（max_model_len=131K/200K/262K）≠ 实际序列长度；
2. **workload**：code / copy / 长生成 的 last-2048 足够，远程语义 RAG 的
   last-2048 不够。

本轮补测（W4A16 target + W4A16 draft，lookup=true，degrade_after=0）：

| 形态 | 实际 ctx | AL | mean tok/pass | 备注 |
|---|---:|---:|---:|---|
| fresh code，400 tok | 44 | 2.91–3.40 | 3.91–4.40 | 两次 boot 波动大；输出不同 |
| fresh code，8K 长生成 | 44→8K | **6.27** | **7.27** | m7=961/1124；长生成进入稳定节奏 |
| code-rust ×6 + fresh-code 指令 | **241,736** | **4.97** | **5.97** | lookup hit 39/68；输出退化（重复 TypeVar），但 AL 维持 |
| DGX Spark EDIT-heavy（6.4K prompt / 3K out） | 6,399 | **6.44** | **7.44** | acceptance 92.0%；输出退化（重复 `#`），不是干净 edit |
| 固定尾部 prose（旧 harness tail 为空；已作废） | 32K | 1.60–1.87 | 2.60–2.87 | 不是真 fixed-tail；见 §10.8 勘误 |
| 固定尾部 prose（修正 harness，真 8K tail） | 34,205 | **3.213** | **4.213** | owl；同 prompt vLLM mean **3.864 / 40.9%** |

社区同量级：

- DGX Spark vLLM，NVFP4 target + BF16 draft，32K 容量、fresh code：
  acceptance **64.4%** / mean **5.51**；同一 harness 的 EDIT-heavy：
  acceptance **99.3%** / mean **7.95**；
- z-lab 3090，160K **容量**、实际 ~11K 序列（889 prompt + 10K gen）：
  mean **4.09**、acceptance 77.2%；
- club-3090 dual 3090，vLLM + W4A16 drafter：AL **3.35**、acceptance 47.8%；
- DFlash 论文 LongBench base：16K AL 3.61 / 32K AL 2.09（长上下文微调后
  6.05 / 3.56）。

**结论修正**：我们的引擎在 **code / copy / 长生成** 的 200K+ 实际上下文上
并不衰减（242K AL 4.97、8K 长生成 AL 6.27），与社区同量级；旧 fixed-tail
prose 的 1.6–1.9 是 **harness 空 tail 的假象**，不是 remote-semantic 证据。
修正 harness 后 32K fixed-tail：owl mean **4.213 / 45.9%**，同 prompt vLLM
mean **3.864 / 40.9%**，二者同量级、owl 略高。DFlash2 的 2048 draft window
在 vLLM/SGLang 同样存在（checkpoint `sliding_window=2048`）；“200K 容量”
不等于“200K 远程语义稳定性”，同 prompt 对齐见 §10.8。

### 10.8 vLLM 同 prompt 对齐 + fixed-tail harness 勘误（2026-10-22 晚）

**勘误**：旧 `curve_fixedtail.py` 取 `prose-zh.txt[60000:68000]` 作为“固定
尾部”，但该文件只有 **50,301 字符** ⇒ `tail` 为空，`tail_tokens=1`，实际
prompt 只是不同长度的前缀。§10.1/§10.3/§10.7 里那组
**3.33 / 1.79 / 2.33 / 1.60** 和 **1.60–1.87** 因此**不能**作为
“固定尾部 / remote-semantic”证据。修正 harness = 把源重复 4 次，再取
`[60000:68000]` 作为真 8K 字符 tail，前缀长度按目标 token 数变化。
已入档 `tools/fixedtail_prompt.py`：tail 为空/越界会**响亮拒**，
不再像旧脚本那样静默退化成“无 tail”。

**同 target/draft/ prompt 的 vLLM 对照**（2×3080）：
- vLLM：`qwen38_3080` flavor = Qwen3.8-27B-AWQ-INT4 + DFlash2-W4A16 +
  fp8 KV，`spec_tokens=7`，port 8134；
- owl：同一 target/draft，W4A16 profile + `lookup=true` +
  `degrade_after=0`；
- 同一 prompt 文件、`temperature=0`、`enable_thinking=false`；
- vLLM `AL = 1 + accepted/drafts`，`acceptance = accepted/draft_tokens`；
  owl `mean = AL+1`，`acceptance = AL/7`。

| prompt | vLLM mean / acceptance | owl mean / acceptance |
|---|---:|---:|
| fixed-tail 32K（修正 harness，34,205 tok） | 3.864 / 40.9% | **4.213 / 45.9%** |
| fixed-tail 16K（修正 harness，16,350 tok） | 4.000 / 42.9% | 3.836 / 40.5% |
| fresh code 400 tok | 4.640 / 52.0% | 4.402 / 48.6% |
| edit-heavy 3,000 tok | **7.976 / 99.7%** | 7.468 / 92.4% |

读法：

- **没有系统性 owl 落后**：32K fixed-tail owl 略高，16K 略低，fresh code 低
  ~2.4pp，edit-heavy 低 7.3pp；差异都在几 pp 量级。
- edit-heavy 的低是**内容差异**：owl 的 target 输出退化成重复 `#` 注释，
  而 vLLM 输出干净完整文件；acceptance 被输出形态污染，不是引擎 AL 差。
- vLLM 在**同一 32K fixed-tail** 上也只有 **40.9%** acceptance，不是
  64.4%；64.4% 是 DGX Spark 的 **fresh code**，不是 200K 远程语义。
- 结论：旧 1.6–1.9 的“owl 长上下文 AL 崩”是**prompt/harness 伪影**；
  同 prompt 下 owl 与 vLLM 同量级。真正的差异来自 workload（code/copy/
  fresh generation vs 长前缀语义续写）和 target/draft 精度，不是 owl
  的 DFlash2 实现。

### 10.9 head int4 → int8（2026-10-22 用户裁决；质量优先）

用户裁决：`rtn-int4-g32` 头太小，默认改 **`rtn-int8-row`**。已改
`profiles/full.toml` 默认 + 3080 TP2 主档（W4A8/VLM 继承）+ 3090
w4a16/dflash/mtp/iq4-mtp 四档；GGUF 档保持 `off`。实测代价：

| 任务 | int4 头 | int8 头 | Δ |
|---|---:|---:|---:|
| edit-heavy round_ms | 32.14 | **57.01** | **+77%** |
| edit-heavy propose_ms | 7.25 | **18.70** | **+158%** |
| edit-heavy AL / mean / acceptance | 6.468 / 7.468 / 92.4% | 5.852 / 6.852 / 83.6% | 输出内容不同 |
| 32K fixed-tail round_ms | 33.48 | **58.59** | **+75%** |
| 32K fixed-tail propose_ms | 7.80 | **19.61** | **+151%** |
| 32K fixed-tail AL / mean | 3.213 / 4.213 | 2.159 / 3.159 | 输出内容不同 |
| 权重 | 10,853.9 MiB | 11,364.5 MiB | +510.6 MiB |
| `[ctx-max]` cap | 441,584 | **409,740** | 仍 >262k |

输出质量：int4 的 edit-heavy 输出退化成重复 `#` 注释；int8 输出变成
可读的 placeholder 实现（`pass` / docstring），质量明显更好。AL 差异
主要是**输出内容不同**，不能直接当精度收益；**速度代价是确定的**。

结论：int8 默认按用户裁决保留；int4 是显式速度回退。上表的 int8 数字是
**旧逐行 GEMV**（`grid.y=tokens`，每个 token 重读一遍权重）的代价；
同日已由 §10.10 的多 token 快路替代，旧数字仅作历史对照。

### 10.10 W8A16 多 token head（2026-10-22 晚；int8 头提速转正 → §10.14 TC）

#### 10.10.1 根因：不是 “GEMV 慢”，是 m=7/8 重读权重

- Marlin 在本仓只有 **W4A16 / W4A8**（vLLM `sm80_kernel_*_u4*`），
  **没有 W8A16 变体**；`w8a16 marlin` 不是可用的下一刀。
- 旧 `owl_gemv_i8w_*` 是 m=1 核，`grid.y = tokens`：spec verify 的 8 行、
  DFlash2 propose 的 7 行各自把所有 0.635 GB 权重读一遍 ⇒ **5.08 GB / 7×0.635 GB**。
- 这解释了旧 int8 头 propose **18.7 ms**（int4 7.2 ms）的绝大部分。

#### 10.10.2 新核：权重读一遍，同时算 8 个 token

`owl_gemv_i8w_mt_{f16,bf16}`（同 `cu/elems/gemv_i8w_f16.cu`）：

| 维度 | 取值 | 作用 |
|---|---:|---|
| `TOK_TILE` | 8 | spec d7 的 7 行 + 1 行 pad；不足补 0，内层零 `if (t<nt)` |
| `ROWS_PER_WARP` | 4 | 一个权重字节复用到 4 行 × 8 token = 32 个累加器 |
| `WARPS` | 8 | 32 行/块；`rows % 32 == 0` 才走快路，否则回退 m=1 核 |
| `K_TILE` | 512 | 每 token shared = 544 float；8 token = **17 KiB**，3 blocks/SM |
| 权重 | 一次 | 每 lane 16 B(`uint4`)；4 行 ×2 base，覆盖整行 |
| x | shared float | `k + (k>>5)` swizzle，固定 i 下 32 lane 命中 32 个 bank |

m=1（prefill 末行、MTP）仍走旧核，逐位不变。数值与旧核同序：
先乘 per-row scale，再 `__shfl_down` 归约；实测量化误差 **≤1 ULP**
（旧/新逐位差 0–164/993280 个 f16 值，远小于 int8 自身量化误差）。

#### 10.10.3 核级读数（3080 单卡，124160×5120）

| 实现 | µs/发 | 权重有效带宽 | 备注 |
|---|---:|---:|---|
| 旧逐行 GEMV（tokens=8） | 7,039.9 | 90.3 GB/s（实际读 5.08 GB） | `grid.y=8`，每 token 重读 |
| **W8A16 mt8** | **2,069.8** | **307.1 GB/s（只读 0.635 GB）** | **3.40×** |
| 旧逐行 GEMV（tokens=7） | 6,159.5 | 103.2 GB/s（实际读 4.45 GB） | `grid.y=7` |
| **W8A16 mt7（专用）** | **1,451.0** | **438.1 GB/s** | **4.25×；比 mt8 少 30%** |

#### 10.10.4 引擎读数（2×3080 TP2，同 prompt，greedy，max_tokens=256）

| tag | int4 头 | int8 旧逐行 | **int8 mt8** | **int8 mt7（→§10.14 TC）** | mt7 vs int4 |
|---|---:|---:|---:|---:|---:|
| `spec.round` | 29.23 | 57.0* | **34.30** | **33.37** | **+4.14 ms (+14.2%)** |
| `spec.propose` | 5.08 | 18.7* | **7.50** | **6.71** | **+1.63 ms** |
| `spec.verify` | 21.95 | — | **24.54** | **24.54** | +2.59 ms |

\* 旧逐行的 57.0/18.7 来自 §10.9 的 edit-heavy 口径；同 prompt 未重跑，
仅作量级对照。新 int8 比旧 int8 的 round 约 **−23 ms**、propose **−11 ms**。
两臂 AL/输出内容不同（int4 54 轮 / int8 83 轮），round/propose 只作
**同口径每轮耗时**，不作 tok/s 直接换算。

32K fixed-tail（34,240 tok，max_tokens=64，lookup on）：

| tag | int4（§10.9 口径） | int8 旧逐行 | **int8 mt8** | **int8 mt7（现役）** |
|---|---:|---:|---:|---:|
| `spec.round` | 33.48 | 58.59 | **39.60** | **39.01** |
| `spec.propose` | 7.80 | 19.61 | **10.76** | **10.11** |
| `spec.verify` | — | — | **26.27** | **26.30** |

新 int8 mt7 相对旧逐行：round **约 −33.4%**、propose **−48.4%**；
相对 int4 仍 +16.5% / +29.6%。

#### 10.10.5 配置 / 回退

- 默认自动生效：`lm_head_matmul` 在 `tokens>1 && rows%32==0` 时选
  `owl_gemv_i8w_mt_*`；其中 **`tokens==7` 选 7-token 专用实例 mt7**，
  其余多 token 选 mt8；`tokens==1` 或行数不整除 32 时走旧核。
- 无新配置键；回退整头 = `--set model.head_quant=rtn-int4-g32`。
- `topk16`（45.7 µs/轮）未融合：相对 1.5–2.1 ms 头是噪声，融合不划算。
- 剩余 gap：int8 仍比 int4 每轮 +4.1 ms（propose 用 mt7 1.45 ms +
  verify 用 mt8 2.07 ms，int4 ~0.4 ms×2）。W8A8 dp4a 已落地（§10.12，
  opt-in，round 32.05）；再往下走是 **int8 tensor core**
  （激活也量化，需过 top-1/AL 质量门），不是 W8A16 marlin。

#### 10.10.6 文件

- `crates/kernels/cu/elems/gemv_i8w_f16.cu`：mt8/mt7 核 + 固定 token shared
- `crates/kernels/src/ops/elems.rs`：`GemvI8wMt{,7}{F16,Bf16}` 声明
- `crates/kernels/src/driver.rs`：`tokens>1 && rows%32==0` 选路；tokens==7 走 mt7
- `crates/models/src/layers/embedding.rs`：mt 时补 `tokens` 标量槽
- `crates/kernels/tests/gemv_i8w_bench.rs`：BDF 测试门 + 旧/新 A/B + 1 ULP 容差

### 10.11 权重读放大审计（2026-10-22 晚；W4A16 是否也重复读？）

判据：同一权重在 `m=1` 与 `m=8` 的 µs 应基本持平（权重只读一遍）；
若 `m=8 ≈ 8×m=1`，就是逐 token 重读。微基准 =
`cargo test -p owl-kernels --features marlin --test marlin_bw_bench
-- --ignored marlin_bandwidth_27b_shapes`（3080，AWQ g32）。

| 路径 | m=1 µs | m=8 µs | 有效带宽 | 判定 |
|---|---:|---:|---:|---|
| marlin gate+up 34816×5120 | 150.9 | **153.5** | 657–665 GB/s | **1×，无重读** |
| marlin down 5120×17408 | 79.8 | **80.0** | 629–631 GB/s | **1×** |
| marlin GDN in 16384×5120 | 74.8 | **75.8** | 627–632 GB/s | **1×** |
| marlin GDN out 5120×6144 | 32.2 | **31.7** | 550–564 GB/s | **1×**（小形状效率略低） |
| 旧 int8 head 124160×5120 | 885.7 | **7038.5** | 90 GB/s（实际 5.08 GB） | **8×，已修** |
| 新 int8 mt8 head | — | **2069.8** | 307.1 GB/s（0.635 GB） | **1×** |
| 新 int8 mt7 head（7 行专用） | — | **1451.0** | 438.1 GB/s（0.635 GB） | **1×** |

结论：

- **W4A16 主路没有重复读权重**；m=8 与 m=1 时间持平，657 GB/s ≈ 3080
  峰值的 86%。W4A16 慢不在“读放大”，而在 tensor-core 小 m 效率 +
  attention/GDN/glue。
- 旧 int8 head 是唯一实锤的 8× 权重读放大；§10.10 已修。
- 小权重旁路：GDN `owl_gemv_dual_f16` 仍是 `grid.y=tokens`（T=8 重读
  b/a 各 0.5 MB×2/层），但全在 L2，48 层合计上限约 0.2–0.5 ms/轮，
  不是主矛盾；要做可另立多 token 版。
- GGUF 路：`m≥4` 已走 MMQ2（m=8 不再重读）；仅 m<4 用 MMVQ。
- **prefill 注意**：marlin 在 `m>1024` 会按 `max_par` 拆 M，每段各读
  一遍权重；当前 `chunk=512` 不触发。若把 prefill chunk 提到 2048/4096，
  先核 `max_par` 与权重读放大，不要直接调大。
- 下一步若继续压 W4A16，应先做 verify 的逐核 CUPTI/nsys 分解（marlin
  占比、attention/GDN/glue），而不是再找“权重重读”。

### 10.12 W8A8 head（2026-10-22；dp4a，opt-in → 10.13 TC）

目标：把 int8 head 从 W8A16 的 ~1.5–2.1 ms 再压到 ~1 ms。**权重不动**
（同一份 per-row int8），只把 hidden 按 token 量化成 int8，内积走
`__dp4a`。这是“W8A8 系列”的第一刀；int8 tensor core 仍留作后续。

#### 10.12.1 新增算子

| 算子 | 作用 | 形状 |
|---|---|---|
| `owl_quant_i8_act_scale_{f16,bf16}` | 每 token `sx=max|row|/127` | `[tokens, cols] → [tokens] f32` |
| `owl_quant_i8_act_q_{f16,bf16}` | `q=RN-even(x/sx)` 截 ±127 | `[tokens, cols] → [tokens, cols] i8` |
| `owl_gemv_i8w_a8_mt{,_7}_{f16,bf16}` | `y=(sw·sx)·Σ w·q`，dp4a | 与 W8A16 mt 同几何 |

- 量化拆 scale/q 两发（输出槽末位，与既有权重版同纪律）；每发 ~6 µs。
- GEMV：`TOK_TILE=8`、`ROWS_PER_WARP=4`、`WARPS=8`、`K_TILE=512`；
  x 的 4-byte group 存 shared word `j+(j>>2)`，固定 g 下 32 lane 命中
  32 bank；一个权重字节复用到 4 行 × 8 token。
- `int32` 累加 + 跨 lane 整数归约（整数加法结合，无顺序误差），
  最后乘 `sw[r]·sx[t]` 再转 f16/bf16。

#### 10.12.2 数值

小形状 host 整数点积对拍（rows=64, cols=128, tokens=8）：
`max|Δ|/max|ref| = 2.89e-4`（同一份 int8 权重/激活，只剩 f16 输出舍入）。

引擎级只做了 **server 同 prompt A/B**（TP2 3080，greedy，max 400）：

| prompt | i8 vs a8 | 备注 |
|---|---|---|
| 长城（reasoning 吃满 400） | 两边 content 都为空 | 无法判 |
| 水池注水 | **逐字相同** | 正确 |
| Rust 中位数 | **不同**（两边都成文，风格不同） | 首个字符即分歧 |

> 2026-10-22 晚已补 **spec m=7/8** 质量门（§10.12.7）：DFlash2 verify
> 走 target head m=8、propose 复用 target head m=7，strict token 一致率
> **0.696 < 0.95**。W8A8 维持 opt-in，不转默认。

#### 10.12.3 速度

核级（3080，124160×5120）：

| 实现 | tokens=7 | tokens=8 |
|---|---:|---:|
| W8A16 mt7 / mt8 | 1,451.0 µs | 2,069.8 µs |
| **W8A8（scale+q+gemv）** | **1,233.5 µs**（−15%） | **1,326.4 µs**（−36%） |

引擎（同 prompt，greedy，256 tok）：

| tag | int4 | W8A16 mt7 | **W8A8** | W8A8 vs int4 |
|---|---:|---:|---:|---:|
| `spec.round` | 29.23 | 33.37 | **32.05** | **+2.82 ms (+9.6%)** |
| `spec.propose` | 5.08 | 6.71 | **6.38** | **+1.30 ms** |
| `spec.verify` | 21.95 | 24.54 | **23.46** | **+1.51 ms** |

W8A8 相对 W8A16：round −1.32 ms、propose −0.33、verify −1.08。
未在 32K fixed-tail 上重测（W8A16 mt7 39.01/10.11 仍是长 ctx 基线）。

#### 10.12.4 质量风险（未转默认）

- server 同 prompt A/B：水池注水题逐字相同；Rust 中位数题从首字符起不同
  （两边都成文，风格不同）；长散文 prompt 出现过一次 `将将将…` 重复
  （I8W 同 prompt 正常）。这可能是 near-tie 被激活量化推过边界。
- **spec m=7/8 门已补，strict token 一致率 0.696（§10.12.7）** ⇒
  `rtn-int8-row-a8` 保持 **opt-in 实验档**，默认仍是 `rtn-int8-row`
  （W8A16）。转默认前不得只看“语义还行”，必须先过 token 级门（或把
  门的判据改成有依据的 top-1/语义门，并重新举证）。
- 8-bit 实现已换成 **SM86 int8 Tensor Core**（§10.13；TC 与 dp4a
  逐位同，只提速，不改变质量门）。**per-128 group 激活 scale** 仍未
  施工：误差可降约 2×，但每组需多一层 float 累加/缩放，预计吃掉大部分
  收益；在没有质量门通过证据前不投入。

#### 10.12.5 用法

```bash
# 实验档
./target/release/server -p profiles/qwen38_27b-3080_tp2-w4a16-a8.toml ...
# 或临时覆盖
--set model.head_quant=rtn-int8-row-a8
```

- 权重装载/键族完全复用 `rtn-int8-row`（`lm_head.i8q/i8s`），无新显存。
- m=1（prefill 末行/MTP）仍走 W8A16，避免小 shape 量化开销。
- 回退：`--set model.head_quant=rtn-int8-row`。

#### 10.12.6 文件

- `crates/kernels/cu/elems/quant_i8_act.cu`：激活 scale/q 四入口
- `crates/kernels/cu/elems/gemv_i8w_a8.cu`：dp4a mt8/mt7 四入口
- `crates/kernels/src/ops/elems.rs` / `driver.rs`：声明、几何、选路
- `crates/models/src/ops.rs`：`QuantI8Act*` / `GemvI8wA8*` 语义动作
- `crates/models/src/module.rs` / `layers/embedding.rs`：`QuantPlan::I8W8`
- `crates/shared/src/config/mod.rs` / `resolve.rs`：`HeadQuant::RtnInt8RowA8`
- `profiles/qwen38_27b-3080_tp2-w4a16-a8.toml`：opt-in 档

#### 10.12.7 spec m=7/8 质量门（2026-10-22 晚；strict 未过）

`tp_head_i8_gate` 增加 `OWL_TP_HEAD_SPEC=1` 臂：按生产 DFlash2 d7 启动
（`tp_draft=true`、草稿 syvai W4A16），target verify = **8 行**，DFlash2
`propose_block` 复用 target lm_head 的 `slice_view(DEPTH=7)` ⇒ 正好覆盖
W8A8 的 m=8 / m=7 两条快路。

口径：2×3080 TP2，greedy，`ignore_eos`，6 prompt × 96 tok；`i8` 与
`a8` 两个独立进程各跑一臂，再比 token 账。spec 账（`/debug/metrics`
同源）证明不是 m=1 假门：

| 臂 | rounds | m 直方图 [m0..m7] | AL=Σm/rounds | 576 tok 轮数 |
|---|---:|---|---:|---:|
| i8（W8A16） | 190 | [68,36,25,16,17,6,9,13] | 1.984 | 190 |
| a8（W8A8） | 195 | [73,38,22,17,15,8,10,12] | 1.933 | 195 |

token 一致率：

| 对照 | 相同/总 | 一致率 | 逐字全等 |
|---|---:|---:|---:|
| f16 vs i8 | 570/576 | **0.9896** | 5/6 |
| f16 vs a8 | 395/576 | **0.6858** | 3/6 |
| **i8 vs a8** | **401/576** | **0.6962** | **4/6** |

- 分歧不是系统性错答：4/6 prompt 逐字全等；水池题在 idx 4 分叉、长文题在
  idx 3 分叉，两边都成文；水池答案两边都正确（i8 更简洁，a8 更展开）。
  长文题两边都出现退化，a8 反而更连贯；本组未观察到事实性错答。
- 但 strict token 门（≥0.95）**明确未过**；f16→i8 只有 1 个后段分歧，
  f16→a8 早期就分叉 ⇒ 差异是 W8A8 激活量化带来的 near-tie 翻转，
  不是 DFlash2/TP2 管线问题。`compare_a8` 按设计会红。
- AL 没塌（1.98 → 1.93，−2.6%），但轮数 +5；同 prompt 速度 A/B 的
  W8A8 收益仍只有 ~1.3 ms/round。**质量/收益比不足以转默认。**
- 生产 `lookup=true` 复跑同两组 prompt：i8 vs a8 仍是 **401/576**、
  首个分歧位置不变（lookup 只改草稿候选，不改 head 路径）。

复现（本机 3080 对 = CUDA ordinal 2/3；两臂各一进程）：

```bash
OWL_TP_TEST_DEV_A=2 OWL_TP_TEST_DEV_B=3 OWL_TP_HEAD_SPEC=1 \
OWL_TP_HEAD_ARM=i8 OWL_TP_HEAD_N=96 \
cargo test --release -p owl-engine --test tp_head_i8_gate -- --nocapture
# a8 同命令换 OWL_TP_HEAD_ARM=a8；落盘默认 /tmp/owl-tp-head-<arm>.txt
OWL_TP_HEAD_ARM=compare_a8 \
cargo test --release -p owl-engine --test tp_head_i8_gate -- --nocapture
```

### 10.13 W8A8 int8 Tensor Core（2026-10-22 深夜；SM86 IMMA）

**硬件定调（用户裁决）**：30 系（GA102/SM86）只有 **int8 Tensor Core
（IMMA）**，**没有 fp8 Tensor Core**；W8A8 的 8-bit 矩阵乘必须走
`mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`，不能写 fp8 MMA。
本刀只换算术，不改 W8A8 的量化契约（同一份 `q`/`sx`/`sw`）。

#### 10.13.1 核

`cu/elems/gemm_i8w_a8_tc.cu`：

| 项 | 取值 | 作用 |
|---|---:|---|
| MMA | `m16n8k32.s8s8s32` | 一条 16×8×32 int8 点积；int32 累加 |
| M | 16（固定） | spec 7/8 token 补零到 16；只消费 row 0..7 |
| N | 8 权重行/warp | 一个 CTA 8 warp = 64 行 |
| A | shared `16×528` | 激活按 K_TILE=512 搬入；行 >= tokens 补 0 |
| B | shared `64×528` | **global 16B/row 连续搬**，再供 8 warp 取 fragment |
| 输出 | shared `8×64` | 先按 token 行落 shared，再连续写 `y[t, rows]` |
| CTA | 256 线程 / 64 行 | `grid.x = ceil(rows/64)` |
| 静态 shared | **43,264 B** | <48KB 默认顶；2 block/SM 驻留 |

关键不是“用了 MMA”，而是 **B 的装载**：旧 dp4a 版每 warp/每 16B 散读
权重；TC 版一个 CTA 先把 64 行 × 512B 连续搬进 shared（每线程 16B），
再由 8 warp 取 fragment。3080 上这一步把有效带宽从 477 拉到
**656 GB/s**。`A` 也按块共享，避免每个 warp 重复读激活。

坑：`__shared__` 是**静态** shared；driver 的 `smem` 必须写 0，不能再报
一次 dynamic（static+dynamic 双算 = `CUDA_ERROR_INVALID_VALUE`，实测）。

#### 10.13.2 数值：与 dp4a 逐位同

- host 整数点积对拍（rows=64/cols=512/tokens=8，f16 输出 **bit** 比）：
  **逐位一致**（不是容差）。
- 引擎 spec 同一 prompt、同一 `q`/`sx`：TC 的 token 账与 dp4a
  **逐字全等**（6/6 prompt；`spec.m` 直方图也相同）。
- 因此 §10.12.7 的质量门原样成立：i8 vs a8 仍 **401/576 = 0.6962**、
  逐字全等 4/6；TC **只提速，不改善激活量化误差**。

#### 10.13.3 核级速度（3080，124160×5120）

| 实现 | tokens=8 | tokens=7 | 有效带宽 |
|---|---:|---:|---:|
| W8A16 mt8 / mt7 | 2,069.8 µs | 1,451.0 µs | 307 / 438 GB/s |
| W8A8 dp4a（scale+q+gemv 合计） | 1,326.4 µs | 1,233.5 µs | 479 / 515 GB/s |
| **W8A8 TC（gemv；另加 scale+q ≈12 µs）** | **969.2 µs** | **964.1 µs** | **656 / 659 GB/s** |

TC gemv 相对 dp4a 总耗时：**−26% / −21%**；相对 W8A16 mt8：**−53%**。
tokens=7 与 8 几乎同速（M=16 补零；权重流量不变）。

#### 10.13.4 引擎（同 §10.12.7 的 6×96 spec 门）

| tag | i8（W8A16 mt） | a8（TC） | Δ |
|---|---:|---:|---:|
| `spec.round` avg | 36.649 ms | **35.310 ms** | **−1.34 ms** |
| `spec.propose` avg | 6.256 ms | **5.993 ms** | −0.26 ms |
| `spec.verify` avg | 28.617 ms | **27.477 ms** | −1.14 ms |

两臂 AL/输出不同（§10.12.7 已解释 near-tie），所以 round 只作**每轮**
对照，不作 tok/s 直接换算。TC 相对 dp4a 的额外收益约为 **0.3 ms/round**
（dp4a 32.05 → TC 预计 ~31.7；同 prompt 尚未单独复测，核级差 0.36 ms）。

#### 10.13.5 裁决

- **W8A8 的 8-bit 实现定为 int8 TC**；dp4a 保留为不支持 `cols%32==0` /
  极端形状的回退。默认 head 仍是 W8A16（`rtn-int8-row`）。
- 质量门未过（0.6962）⇒ `rtn-int8-row-a8` 仍是 **opt-in 实验档**。
- **2026-10-22 深夜更新**：W8A16 也上了 f16 TC（§10.14），核级/引擎
  与 W8A8 TC 基本同速，且质量门 0.9896 过。**W8A8 由此被支配**：
  不再有速度理由，只保留为对照/诊断档。
- 文件：
  - `crates/kernels/cu/elems/gemm_i8w_a8_tc.cu`
  - `crates/kernels/src/ops/elems.rs`（`GemmI8wA8Tc{F16,Bf16}`）
  - `crates/kernels/src/driver.rs`（`gemm_i8w_a8_tc_*` + eligible）
  - `crates/models/src/ops.rs` / `layers/embedding.rs`（TC 优先，dp4a 回退）
  - `crates/kernels/tests/gemv_i8w_bench.rs`（`gemm_i8w_a8_tc_bench`）

### 10.14 W8A16 f16 Tensor Core（2026-10-22 深夜；Marlin 同款“反量化”流水）

**动机**：W8A16 不能直接喂 int8 IMMA（激活是 f16）；但可以学 Marlin
W4A16 的流水：**HBM 读压缩权重（int8，1 B/参数）→ shared 仍 int8 →
MMA fragment 在寄存器里 `int8 → half/bf16` → f16/bf16 MMA**。这样权重
流量不增、激活不量化，质量仍是 W8A16；比当前 FMA GEMV 多一条 tensor
core 计算路。

#### 10.14.1 核

`cu/elems/gemm_i8w_a16_tc.cu`：

| 项 | 取值 |
|---|---|
| MMA | `mma.m16n8k16.row.col.f32.f16.f16.f32`（bf16 版同形） |
| M/N | 16（7/8 token 补零）/ 8 权重行/warp |
| CTA | 8 warp = 64 行；`grid.x=ceil(rows/64)` |
| K_TILE | 256 |
| A | shared f16/bf16 `16×264`（8.4 KB） |
| B | shared **int8** `64×272`（17.4 KB）；HBM 16B/row 连续 |
| 输出 | shared `8×64`；按 token 行连续写 |
| 静态 shared | **26.3 KB** ⇒ 3 block/SM |
| 反量化 | 每个 MMA fragment 读 2×2 int8，`__int2half_rn`/`__int2bfloat16_rn` 后 pack 成 half2/bf162 |

关键与 Marlin 同律：**压缩权重只读一次，反量化发生在近端（寄存器）**；
没有把整份权重展开成 f16 再落 HBM。B 的 shared 保持 1 B，避免
“反量化后 2 B shared”把带宽吃掉。

#### 10.14.2 数值 / 质量

- 小形状 host 对拍（rows=64/cols=512/tokens=8）：
  `max|Δ|/max|ref| = 2.67e-4`（f16 输出舍入）。
- 引擎 spec 门（同 §10.12.7 的 6×96）：
  - f16 vs W8A16-TC：**570/576 = 0.9896**，逐字全等 **5/6**
  - 唯一分歧仍在 CSV prompt 的 **idx 90**，两边成文、语义同类
  - 与旧 W8A16 FMA 的 token 账也基本一致（同一处 late near-tie）
- 结论：**W8A16 TC 保持 W8A16 的质量门**，不像 W8A8 那样掉到 0.6962。

#### 10.14.3 核级速度（3080，124160×5120）

| 实现 | tokens=8 | tokens=7 | 带宽 |
|---|---:|---:|---:|
| W8A16 FMA mt8 / mt7 | 2,069.8 µs | 1,451.0 µs | 307 / 438 GB/s |
| **W8A16 f16 TC** | **983.8 µs** | **980.5 µs** | **646 / 648 GB/s** |
| W8A8 int8 TC | 969.3 µs | 964.1 µs | 656 / 659 GB/s |

W8A16 TC 相对旧 FMA：**−52% / −32%**；只比 W8A8 TC 慢 **1.5%**，
但质量好一个档。

#### 10.14.4 引擎（同 §10.12.7 门）

| tag | 旧 W8A16 FMA | **W8A16 TC** | W8A8 TC |
|---|---:|---:|---:|
| `spec.round` avg | 36.649 ms | **35.281 ms** | 35.310 ms |
| `spec.propose` avg | 6.256 ms | **5.970 ms** | 5.993 ms |
| `spec.verify` avg | 28.617 ms | **27.453 ms** | 27.477 ms |

W8A16 TC 与 W8A8 TC 在同一条门里**基本同速**（差 0.03 ms/round），
但 W8A16 TC 的 AL/输出与 f16 更一致（quality gate 0.9896）。

> **std 单次复测（非受控 A/B）**：262k / greedy / 15 点，decode avg
> **132.65 t/s**（范围 74.4–222.8）、prefill 1616.5。旧基线 138.1 是
> 另一轮 std，prompt 实际长度与 AL 波动大，**不能拿单次均值判回归**。
> 受控门（lookup=true，同 prompt）旧 FMA 与新 TC 的 token 账/rounds
> **逐字相同**，round 36.65→**35.12ms** ⇒ 同 AL 下折算约 **+4%**。

#### 10.14.5 裁决：默认切 W8A16 TC，W8A8 退回诊断

- 默认 `rtn-int8-row`（W8A16）现在自动走 **f16 Tensor Core**；
  不需要新配置键，`gemv` 形状不满足时回退 FMA 多头核。
- **W8A8 被支配**：同速、质量门 0.6962；`rtn-int8-row-a8` 只保留为
  对照/诊断档，不再作为速度档推荐。
- W8A8 的 dp4a/int8 TC 代码不删（对照与回退资产）；a8 profile 头注
  标记 deprecated。
- 文件：
  - `crates/kernels/cu/elems/gemm_i8w_a16_tc.cu`
  - `crates/kernels/src/ops/elems.rs`（`GemmI8wA16Tc{F16,Bf16}`）
  - `crates/kernels/src/driver.rs`（`gemm_i8w_a16_tc_*` + eligible）
  - `crates/models/src/ops.rs` / `layers/embedding.rs`（W8A16 TC 优先）
  - `crates/kernels/tests/gemv_i8w_bench.rs`（`gemm_i8w_a16_tc_bench`）

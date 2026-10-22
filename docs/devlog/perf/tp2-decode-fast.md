# TP2 双 3080：decode 快档 + W4A8 定档（2026-10-22）

> 目的：把 `feat/w4a8-p1` 工作树里“TP2 decode 档面体检 / W4A8 / marlin 测量修正”
> 三件候选在当前 `main@feb0131` 上**逐项重测**，只把真赢的配方转正。
> 本文所有数字 = 同一二进制、同一对 3080、同一模型、同一 prompt（固定 nonce），
> 单变量 A/B；不再拿不同日期/不同口径的旧读数直接对比。

## 0. 结论（先看这五条）

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
2. **自适应 verify 长度 / lookup**（vLLM DFlash2 的两条社区方案，未做）：
   - 长块只在“复述上下文”时用（`next_num_draft_tokens` 式；多一个 verify 位
     在 1.5k 只贵 6%、25k 贵 27%）；
   - lookup-augmented drafting 治 copy/GUI 长上下文（owl 无）。
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

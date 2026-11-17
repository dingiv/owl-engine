# TP2 decode 两根因 —— 修正版（2026-11-17）

> 上一版把「TP2 草稿 AL 7.30→4.57」列为根因①。同模型/同量化/同 token 流的
> 公平 A/B 推翻了它：**AL 7.96 vs 7.99（无塌方）**。真实问题是
> **TP2 每轮 0.745×（理想 0.5×）+ AR/GDN/conv 残余 + 每卡固定开销**
> ⇒ 第二张卡只换到 1.34× 吞吐，功耗翻倍（1.42× 能量/token）。
> 本文保留修正后的两根因与修复顺序。

---

### 0.1 复核：单卡 vs TP2 的 decode + 平均功率（1000 tok，2026-11-17 二次实测）

同模型 bytkim W4A8、thinking off、greedy、1000 tok；功率 = 稳态窗口
（请求中段、util≥50% 采样）的平均值，脚本 `/tmp/owl_power_ab.py`。

| | 单 3090 | TP2 3090 | 变化 |
|---|---:|---:|---:|
| decode | **257.6 t/s** | **332.7 t/s** | **+29.2%** |
| 平均功率 | **281.5W** | **475.9W**（220.6 + 255.3） | **+69%** |
| 能量/token | **1.093 J** | **1.431 J** | **+31%** |
| AL / 轮数 | 7.99 / 125 | 7.99 / 125 | 持平 |

边际账：第二张卡 **+75 t/s 换 +194W** ⇒ 边际 **2.6 J/tok**，是单卡
1.09 J/tok 的 **2.4×**。即“双卡确实更快（+29%），但第二张卡的能效
远差于第一张”。

## 0. 公平 A/B（唯一有效口径）

**同模型 `bytkim W4A8` · 同 draft `syvai W4A16` · 同 prompt · thinking off ·
greedy · 600 tok**：

| | 单 3090 | TP2 3090 | 比例 |
|---|---:|---:|---:|
| decode | 252.0 t/s | **338.0 t/s** | **1.34×** |
| AL | 7.96 | **7.99** | 1.00× |
| spec 轮数 | 75 | 75 | 1.00× |
| 每轮墙钟 | 31.7ms | **23.6ms** | **0.745×** |
| 稳态功耗 | 295W | ~280W ×2 | 1.90× |
| 能量/token | **1.17 J** | **1.66 J** | **1.42×** |

⇒ 第二张卡把每轮算力降到 0.745×（不是 0.5×），吞吐 +34%，功耗 +90%。
**每卡并没有“把对面的计算再做一遍”**，但**也没有拿到应有的并行收益**。

### 0.2 vLLM 单卡 vs 双卡（spec none / vision 0 / ctx 4096，2026-11-17）

vLLM 单卡 27B **带 spec d7/draft/vision 直接 OOM**：权重 19.8G + FP16 LM head
2.37G + draft/vision > 24G（日志 `torch.OutOfMemoryError`）。只能退到
`--spec none --vision 0 --context 4096`；双卡用**同参数**对照：

| | 单 3090 | 双 3090 TP2 | 变化 |
|---|---:|---:|---:|
| decode | **47.0 t/s** | **74.4 t/s** | **+58%** |
| 平均功率 | **305.0W** | **604.8W**（298.9 + 306.0） | **+98%** |
| 能量/token | **6.49 J** | **8.13 J** | **+25%** |

- vLLM 的 TP2 在 **plain decode** 下是 1.58×（比 owl spec d7 的 1.29× 好：
  plain 是 weight-bound，切权重直接减半每卡流量）；但功耗仍 2×。
- vLLM 生产双卡 **spec d7** 早前测得 ~163.5 t/s / 248W 每卡 / 3.03 J/tok
  （725 prompt、vision on、fp8 KV；不同口径，仅参考）。
- 结论：**TP2 功耗翻倍是引擎无关的**；第二张卡买的是容量/延迟，不是能效。

## 1. 上一版根因①为什么作废（记录，避免重犯）

两处对照污染：

1. **模型/量化不同**：单 3090 档 `qwen38_27b-3090.toml` 继承
   `cyankiwi W4A16`；TP2 档 `qwen38_27b-3090_tp2.local.toml` 显式
   `bytkim W4A8`。prompt 64 vs 69 tok 已经暴露模板/分词不同。
2. **thinking on 的 token 流分叉**：TP 的浮点归约顺序让 **第一个 reasoning
   token** 就翻转（同 bytkim 单卡 anchor=35760，TP2 anchor=96227）；之后两
   引擎在**不同 token 流**上跑，AL 不可比。thinking off 后两条流逐位一致，
   AL 7.96 vs 7.99。

⇒ **P2.1-c 的「TP2-spec 恒等门」对 near-tie 的 thinking 流过严**：它要求
token 逐位一致，但 TP 的 FP 顺序本来就会在近平分处翻 greedy。门的正确口径 =
thinking off（或固定 prefix 教师强制）下的 AL/接受账一致，而不是整段
reasoning 逐位一致。

## 2. 纯 decode nsys（同模型/同 AL/同 75 轮，前缀已热）

**单 3090 vs TP2 3090（bytkim W4A8，thinking off，600 tok，节点级 nsys）**。
TP2 数值已按「每卡 = 总量 / 2」归一：

| 核 | 单卡 | TP2 每卡 | 每卡/单卡 | 判据 |
|---|---:|---:|---:|---|
| Marlin target | 1.253s | 0.676s | **0.539×** | 切分正确 |
| GDN recurrence | 152ms | 79ms | **0.519×** | 切分正确 |
| GDN conv_fwd | 44.6ms | 41.8ms | **0.937×** | 未切（3.8µs/发，latency-bound） |
| GDN l2norm | 11.2ms | 10.4ms | 0.929× | 未切（小核） |
| GDN norm_act | 10.9ms | 10.1ms | 0.926× | 未切（小核） |
| GDN gating | 5.2ms | 5.0ms | 0.96× | 未切（小核） |
| int8 量化 | 64.6ms | 53.2ms | 0.824× | hidden 全量，固有 |
| **AR barrier** | **0** | **200ms** | **新增** | **12.6% GPU 时间** |
| AR stage f8 | 0 | 60ms | 新增 | prefill 大包（一次） |
| 每卡总核时 | ~2.17s | ~1.59s | **0.73×** | 与墙钟 0.745× 吻合 |

结论：

- **Marlin / GDN recurrence 已正确切半**（0.54×/0.52×）——不是重复计算；
- **AR barrier 是剩余差距的第一大项**：200ms/卡（12.6%），单卡为 0；
  128 发/轮（64 target + 64 draft）×20.8µs。
- GDN conv/l2norm/norm_act/gating 合计 ~67ms/卡（3%），是 **3–4µs 小核的
  发射延迟**，不是算力重复；q/k/v 三段 conv 合并可省 ~27ms/卡。
- 若 AR 归零，每卡约 1.33s = 0.61× 单卡；再合并小核可到 ~0.58×。

### 2.1 形状取证：该切的都已经切了（2026-11-17）

boot 日志 `[tp-shape]`（新探针，纯 host 打印，TP2 安全）：

| 形状 | 单卡（全量） | TP2 每 rank | 判定 |
|---|---|---|---|
| target hq/hkv | 24 / 4 | **12 / 2** | 已切半 |
| target nk/nv | 16 / 48 | **8 / 24** | 已切半 |
| target inter | 17408 | **8704** | 已切半 |
| draft hq/hkv | 32 / 8 | **16 / 4** | 已切半 |
| draft inter | 17408 | **8704** | 已切半 |

⇒ 每卡 0.93–0.94× 的核（GDN conv/l2norm/norm_act/gating、draft GEMM/
attention）**不是宽度没切**，而是每发 3–4µs 的**发射延迟地板**；动作应是
**融合发射**，不是再切。

## 3. 修正后的根因

### ① 每轮只到 0.745×，不是 0.5×（P0）

TP2 的 Marlin target 单发时长确实减半（0.54×），但每轮总时只到 0.745×。
差额 = 每轮 ~7.7ms 的固定/未切面：

- **AR barrier**：thinking-on nsys 里每卡 3.1ms/轮（~44 次 AR × 20µs），
  占 TP2 单轮的 13%；单卡为 0。
- **GDN recurrence/conv**：按轮归一，recurrence 0.71×、conv 1.29×
  （理想都是 0.5×）；conv 是唯一**反向**的核。
- **每卡固定面**：权重/embedding/采样/图节点开销不随 TP 减半。

### ② 能量/token 1.42×（P1，由①派生）

TP2 的每卡工作量 > 单卡一半，但每卡功耗几乎不降（280W vs 295W）：
- 单卡做 1 份工作 / 295W；TP2 每卡做 ~0.37 份 / 280W。
- 只要每轮压到 ~0.6×，能量就能回到 ~1.2 J/tok。

### 3.5 深度 A/B（2026-11-17，同模型/同 prompt/thinking off/600 tok）

| depth | AL | 轮数 | decode | 每轮 |
|---|---:|---:|---:|---:|
| **d7** | **7.99** | 75 | **338.0 t/s** | 23.6ms |
| d5 | 5.97 | 100 | 261.5 t/s | 22.9ms |
| d4 | 4.97 | 120 | 225.4 t/s | 22.1ms |

⇒ **越浅越慢**：每轮只省 0.7–1.5ms（AR/小核固定开销不随深度降），
轮数却涨 33–60%，AL 掉更多。d7 在 AL 7.99 时 verify 8 位几乎全收，
Marlin 没有浪费。动态 d4/d5 **不是当前负载的杠杆**；只可能在 AL<3 的
低接受率场景做 fallback，且收益边际。**原 P0-4 降级为 P2。**

## 4. 修复清单（按收益/风险）

### P0: 形状面已无可切；只剩“融合 + AR 协议 + 动态深度”

1. **不再做“切分”修复**：`[tp-shape]` 证明 target/draft 全部已切半；
   0.93× 小核是发射延迟，不是全量。
2. **AR 账（已拆清；C 方案作废）**：
   - target = 128 发/轮（64 层 × o_proj/down_proj）×20.8µs = **200ms/卡**；
   - draft = 10 发/轮（5 层 × 2）×25.3µs = **19ms/卡**；
   - **草稿复制/单 rank 撤回**：最多省 1.2%，不值 +0.3GB/卡。
   - **decode 走的是 host barrier 单核**（`ArFusedCall.with_barrier`，
     `n_bytes ≤ 128KB`）⇒ 20.8µs 就是该路径地板；P1.2 的 1-stage 选择面
     只影响**设备路径**（prefill/大载荷），修了也碰不到 decode ⇒ **C 作废**。
   - 要降只能**减少发数**：动态深度已实测负收益（§3.5）；即 AR 是 TP2
     decode 的结构性地板（12.6%）。
3. **GDN 小核融合（~67ms/卡，3%）**：q/k/v 三段 conv_upd 合一发；
   l2norm/norm_act/gating 并进 recurrence/相邻核；省 ~27–40ms/卡。
4. **动态深度 d4/d5 —— 已实测，降级为 P2（见 §3.5）**：d4/d5 在
   AL 5–8 的负载上全面慢于 d7；只保留“AL<3 时 fallback”的理论价值，
   不作为当前修复项。
5. quant（53ms/卡，3.3%）hidden 全量，**固有**，不动。

### P1: 验收

| 项 | 当前 | 目标 |
|---|---:|---:|
| TP2 decode | 338 t/s | **≥390 t/s**（≈1.55× 单卡） |
| 每轮墙钟 | 23.6ms | ≤20ms（≈0.63×） |
| AR barrier | 200ms/卡 | ≤200ms（地板；靠动态深度降载荷） |
| GDN 小核合计 | 67ms/卡 | ≤40ms/卡（融合） |
| 能量/token | 1.66 J | ≤1.35 J |
| AL | 7.99 | ≥7.9（thinking off） |

### 方法学（防止再被污染）

- A/B 必须**同 profile 的 model/quant/draft**；单卡对比用
  `--set model.path=bytkim… --set model.quant=w4a8` 强制同源。
- AL 对比必须 **thinking off** 或**固定 prefix**；thinking on 的 token 流
  只比“各自流上的 AL”，不跨引擎比 AL。
- 功耗取**稳态**（≥600 tok decode；200 tok 的 0.6s 采样会抓到爬坡）。

## 5. 复现（3090 对空闲时）

```bash
# 单卡（同模型）
./target/release/server -p profiles/qwen38_27b-3090.toml \
  --set runtime.max_seq=32768 --set runtime.prefill_chunk=1024 \
  --set model.path=../models/bytkim/Qwen3.8-27B-pi-AWQ-INT4 \
  --set model.quant=w4a8
# TP2
./target/release/server -p profiles/qwen38_27b-3090_tp2.local.toml \
  --set runtime.prefill_chunk=1024
# 请求（thinking off）
{"messages":[{"role":"user","content":"请只输出 1 到 600 的数字，每行一个。"}],
 "max_tokens":600,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}
```

## 6. 附：`draft_trace` 诊断键（已落）

`[probes] draft_trace = true` 只打印**已在关键路径读回**的 draft/accept 账
（`exec/draft.rs` 收割面 + `exec/spec.rs` 接受账），不新增 dtoh/sync ⇒
**TP2 可开**（`dflash_probe` 的 prefill 读探针仍被 TP v1 拒启）。
用于同模型单卡/TP2 的 `pos/anchor/drafts/ids` 逐位对照。

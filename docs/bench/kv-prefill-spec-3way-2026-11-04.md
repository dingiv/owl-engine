# KV 格式 × DFlash 投机：单 3090 Ti 预填速度三点两轮（2026-11-04 夜）

> 口径：单 **3090 Ti**（`0000:41:00.0`）· Qwen3.8-27B AWQ **W4A8** · `prefill_chunk=2048`
> · `pool.prefix_cache=false` · **spec = DFlash2 d7（`syvai/Qwen3.8-27B-DFlash2-W4A16`，
> BF16 草稿、`lookup=true`）** · 视觉塔 **关** · 每点 **fresh server** · 每点 **两轮独立 boot**。
> 目的：回答「带投机解码时，三种 KV 池格式的**预填**速度差多少」以及「f16 池能不能用」。

## 1. 变体（只差池格式；draft / 权重 / chunk 全同）

| 变体 | profile | KV 主池 | per_slot | prefill 注意力核 |
|---|---|---|---:|---|
| `int8` | `measurement/qwen38_27b-3090-int8.toml` | K int8 + packed V（token-major，无 shadow） | 1 122 304 B | 本仓 P8 核 |
| `fp8s` | `measurement/qwen38_27b-3090-fp8.toml` + `kernel.prefill_shadow=true` | fp8 e4m3 + **f16 staging shadow** | 1 048 576 B | FI `flashinfer_prefill_paged_f16` |
| `fp16` | 同上 + `pool.kv_dtype=f16` | f16 | 2 097 152 B | FI `flashinfer_prefill_paged_f16` |

`[kernel-env]` 实测标签：`kv.main.int8` / `kv.main.fp8` / `kv.main.f16` + 三者
`kv.draft.bf16 · feat.dflash · feat.spec`，`vision=false` ✓。

## 2. 结果（预填 t/s；两轮 → 均值）

| 长度 | int8 r1 / r2 → **均值** | fp8+shadow r1 / r2 → **均值** | f16 r1 / r2 → **均值** | fp8s vs int8 |
|---:|---|---|---|---:|
| 8 192 | 2337.3 / 2340.5 → **2338.9** | 2366.1 / 2377.1 → **2371.6** | 2340.4 / — → **2340.4** | **+1.4%** |
| 32 768 | 2074.4 / 2074.9 → **2074.7** | 2131.9 / 2105.3 → **2118.6** | 2048.1 / 2064.5 → **2056.3** | **+2.1%** |
| 65 536 | 1790.0 / 1788.8 → **1789.4** | 1855.1 / 1852.5 → **1853.8** | **装不下（见 §4）** | **+3.6%** |

TTFT（ms，与上表同序）：

| 长度 | int8 | fp8+shadow | f16 |
|---:|---:|---:|---:|
| 8k | 3 507 / 3 502 | 3 464 / 3 448 | 3 502 |
| 32k | 15 798 / 15 794 | 15 372 / 15 566 | 16 001 / 15 874 |
| 65k | 36 615 / 36 638 | 35 330 / 35 380 | — |

**复现性**：同变体两轮差 ≤1.2%（int8 三点 ≤0.07%）；跨变体排序 **6/6 轮一致**
（fp8+shadow 始终最快）。⚠️ 同期 GPU2（`42:00`）上有另一路 MTP 实验在跑，
共享功耗/频率 ⇒ 绝对值的 ±2% 应视为跨机时波动，**排序**是可靠的。

## 3. 结论

1. **带 DFlash 投机时，预填最快的是 fp8+shadow，不是 int8**：8k +1.4% /
   32k +2.1% / 65k **+3.6%**，且优势随上下文变长扩大（= 注意力占比变大）。
   与「spec 关」时的三向对照（int8 领先 3~4.5%）**方向相反**，差异来源是
   合并进来的 **V token-major** 布局：它让 decode +4%，但预填每块多付
   一次字节转置（实测该转置的残余税 = 预填 −2~4%，见
   `devlog/kernels/int8-prefill-opt-community-2026-11-04.md` §9~§12）。
2. **f16 不是「更准所以更慢一点」那么简单**：32k 比 int8 还慢 **0.9%**
   （与 fp8+shadow 的差 = 4.0%），却要 **2× 池**，在 24G 上直接把可用上下文
   砍半（§4）。f16 池的预填核与 fp8+shadow **是同一个** FI f16 核 ⇒
   差别只在「KV 写 2× 字节 + 无 shadow 拷贝」，实测净效果是 f16 反而更慢。
3. **投机解码的 decode 数字在这套口径下不可比**：32 token 样本 + 草稿接受率
   逐 prompt 变化（本轮 decode 在 37~139 t/s 之间摆动）。要评 decode 请用
   ≥256 token 且固定 seed/prompt（见 `docs/bench/kv-format-3way-2026-11-04.md`
   的 spec-off 口径）。

## 4. f16 池在 24G 上的三个硬约束（本轮实测踩全）

| 现象 | 现场 | 说明 |
|---|---|---|
| **开不起来**（`target=0.95` + `reserve=2048`） | `budget=0.89G ⇒ cap=14555 tok < 36864`，boot 直接拒 | f16 `per_tok=65536B`（16 个全注意力层 × K+V × 2B）⇒ 2.3 GB 池才算 35k 上下文 |
| **能开、请求时炸**（`max_seq=40960`，池 2.56G） | `PoolExhausted: vram-hard-budget 需 25 165 824B 余 13 172 736B`（`reserve=1024`）⇒ `丢本 turn`，API 回**空补全** | 每 chunk 的 `gdn_conv_fwd` / `narrow_strided` / `embed` 各要 ~20~24 MiB 瞬态；池把余量吃干后**硬预算门**拒发 |
| **可用档** | `max_seq=34816`（1088 槽 × 2 MiB = 2.18 GB，`vram_target=0.99`、`reserve=1024`）⇒ 8k/32k 均跑通 | 65k 需 ≥ 65 568 tok ⇒ 池 ≥ 4.1 GB ⇒ 24G 单卡（权重 17.2G + 固定 1.07G）**装不下** |

⚠️ **测量陷阱（务必记）**：f16 池 OOM 时，`llm_speedtest` 会**降级**成
「本地估算 + 0 输出 token」，打印出 `prefill=324204 t/s / TTFT=27ms` 这种
假数据（日志里那行 `⚠prefill边界降级` 才是真相）。判读时先看
`out 32(API真值)` 与 `decode≈0`。

## 5. 复现

```bash
# int8（fp8s 换 -p ...-fp8.toml + kernel.prefill_shadow=true；f16 再 + pool.kv_dtype=f16）
./target/release/server -p profiles/measurement/qwen38_27b-3090-int8.toml \
  --set 'parallel.device=["0000:41:00.0"]' --set 'model.quant="w4a8"' \
  --set 'pool.prefix_cache=false' --set 'runtime.max_seq=73728' \
  --set 'spec.depth=7' --set 'spec.lookup=true' \
  --set 'spec.draft_path="'"$PWD"'/models/syvai/Qwen3.8-27B-DFlash2-W4A16"' \
  --set 'server.bind="127.0.0.1:8151"' --log-level info &
cd ../llm_speedtest && uv run cli --api-url http://127.0.0.1:8151/v1/chat/completions \
  --lengths 32768 --rounds 1 -c 1 --output-length 32 --timeout 900000 \
  --prompt-nonce kvbench-int8-32k --json /tmp/p.json
```

> 端口用 8151 是因为同期 8141/8144 上有别的实验；判决前先 `curl /health`。
> 原始 JSON（每点 TTFT/decode 逐请求）落在 `/tmp/kvbench/`，按上面命令可重跑。

---

# 6. 配置层收紧：显存预留太保守（2026-11-04 用户裁决后落地）

§4 的 OOM 不是算错，是**预留口径太保守**。`pool_cap` 的账是：

```text
base   = min(free, free − total×(1−target))      # target 只作上限
budget = base − reserve − fixed_after            # fixed_after = 权重外的固定项
cap    = budget / per_tok
运行时闸 = driver_free − reserve ≥ 本次申请字节    # 每条 alloc/htod 都过
```

24G 单卡上，旧档光「白留」就有 **~1.7 GiB**：`target=0.99/0.97`（236/708 MiB）
+ `reserve=1024`（f16） + `snap_max=2` 的第二槽（152 MiB）+ store 复用池 256 MiB。

## 6.1 改了什么

| 项 | 旧 | 新 | 位置 | 依据 |
|---|---|---|---|---|
| `vram_target` | 0.99（int8/fp8）· 0.97（f16） | **1.0** | `parts/pool/{int8,fp8,f16}.toml` | `1.0` 是合法档（`base` 直接用驱动 free）；runtime 闸仍以 `reserve` 兜底 |
| `vram_reserve_mb` | f16 `1024` · 测量档 `512` | **f16 512 · 测量档 384** | 同上 + 两个测量档 | 池外瞬态实测峰值 **20~25 MiB/块**（`gdn_conv`/`narrow_strided`/`embed`）⇒ 384~512 已是 15~20× |
| `transient_reserve_mb` | 1024 | **512** | 同上 | 该项只服务 `[kvmem]` **超卖档**；非超卖档本来就无感（§代码注 2026-10-28） |
| `snap_max` | 2（测量档） | **1** | `measurement/qwen38_27b-3090-{int8,fp8}.toml` | GDN 快照池 297→148 MiB；单流测量用不到第二槽（`prefix_cache=false` 时更用不到） |
| `pool_cap_mb` | 256 | **64** | 同上 | store 搬运复用池（设备面）；单流 64 MiB 足够（超限逐块顶出回驱动，只慢不错） |

> 生产档（`qwen38_27b-3090.toml`、3080/4080/5070ti 各档）只吃到「`target→1.0`、
> `transient→512`」这两条由池组件带下去的变化；`snap_max`/`pool_cap_mb` 的收紧
> **只作用于两个测量档**，不动生产的前缀缓存/搬运面。

## 6.2 实测效果（f16，`max_seq=40960`，同 profile 同参数）

| | `[vram-gov]` | 请求结果 |
|---|---|---|
| 旧（`target=99%` + `reserve=1024`） | `budget=2.83G cap=43160 tok` | 预填后 **24 MiB 瞬态被拒**（`free=1036 − reserve=1024 = 12 MiB`）⇒ `PoolExhausted` ⇒ **空补全** |
| 新（`target=100%` + `reserve=384` + `snap_max=1`） | `budget=3.84G cap=62887 tok` | **跑通**：prefill **2090.4 t/s** / TTFT 15.68 s / decode 54.9 t/s |

f16 可用上下文（单 24G，权重 17.2G，spec d7）：

| 档 | 上限 | 证据 |
|---|---:|---|
| 旧配置 | ~34.8k | `max_seq=36864` 时 `cap=14555`（`target=0.95` 更糟）；`40960` 请求期 OOM |
| **新配置（`snap_max=1`）** | **40 960 跑通** | cap 62 887 |
| **新配置 + `snap_max=0`** | **49 152 跑通** | prefill **1952.4 t/s** / TTFT 25.18 s / decode 55.9 t/s（`fixed_after` 0.92G→0.78G） |
| 65k | 仍差 ~0.2 GB | `fixed_after` 里 `图slab = 320 MiB` 是**保守上界**（同档实测 0.0 MiB）⇒ 要吃它得改 `crates/store/src/plan.rs`（代码层） |

int8 回归（同批配置收紧后，单次）：32k **2059.8** t/s（旧 2074.7，−0.7%）、
65k **1775.2**（旧 1789.4，−0.8%）——在单次噪声内；`cap=117 511 tok` ≫ `max_seq`，
说明 int8/fp8 的限制本来就在 `max_seq`，收紧只把余量还给运行时闸。

## 6.3 压力行为（收紧后仍然优雅）

`target=1.0` 把最后的缓冲交给 `reserve` + runtime 闸：
日志实测多次走「硬预算前回收默认池缓存 320/256/224 MiB ⇒ 复判通过」——
即先 trim 池缓存再复判，只有确实腾不出才拒发。收紧**不是**拆掉这道闸。

> **裁决（2026-10-21）**：本 A/B 后 prefill 桶已退役，代码已从 `perf-review` 工作树删除（待合并/待提交）；
> 备份分支 `backup/prefill-bucket-2026-10-21`。本文保留为决策依据。
>
# W4A8 safetensors × prefill 桶 A/B（2026-10-21）

> ⚠️ **2026-10-21 存证说明**:本卷引用的 `profiles/legacy/*` 已随「配置档全字段显式」裁决删除;数字与命令保持原样(当时现场),复跑请用 `profiles/` 下同名生产档。

> 目的：判断 W4A8 主场景下 prefill CUDA Graph（`prefill_bucket`）是否值得保留。
> 工作树：`perf-review @ main@23116c7`；单张 RTX 3090 Ti（42:00.0）。
> 模型：`cyankiwi/Qwen3.8-27B-AWQ-INT4`，`model.quant = "w4a8"`。
> 档：`profiles/legacy/spec-dflash.toml` + `testdata/args/w4a8-ab-27b.toml`，
> `prefill_chunk=1024`、`prefill_tail_merge=1024`、fp8 KV、`prefix_cache=true`。
> 采样：greedy、`--ignore-eos`、输出 256、固定 prompt seed/nonce。

## 0. 先说测试口径（重要）

W4A8 主场景默认会带：

- `head_quant = rtn-int4-g32`（lm_head W4A16 核）；
- `spec.depth = 7`（DFlash2 草稿是 W4A16）。

在本工作树上这两者都会让 W4A8 boot **被算子准入拒**：

```text
contract marlin_gemm_w4a16_awq:
  tag_filter 期望 facet `quant.weight` 命中 [quant.w4a8, quant.awq, ...]
  实得 tags=["...","quant.w4a16","quant.awq","feat.graph"]
```

即 **W4A8 主干 + W4A16 头/W4A16 草稿** 的 tag 面没有并集。  
为了只测“桶”这一个变量，本 A/B 显式关掉：

```text
model.head_quant = off
spec.depth        = 0
```

⇒ 只跑 **W4A8 主干**；这是当前可启动的 W4A8 隔离档。
（W4A8 生产档要带 int4 头/DFlash2，需另案修 tag 面，不属本 A/B。）

## 1. 结果（两轮 cold，同 nonce 对）

### 1.1 prefill t/s

| prompt | off run1 | on run1 | off run2 | on run2 | on−off |
|---:|---:|---:|---:|---:|---:|
| 512 | 2 483.62 | **2 514.43** | 2 482.72 | **2 510.61** | **+1.2% / +1.1%** |
| 2048 | 2 548.99 | **2 565.44** | 2 470.97 | **2 492.10** | **+0.6% / +0.9%** |
| 8192 | 2 270.82 | **2 299.63** | 2 290.76 | **2 308.48** | **+1.3% / +0.8%** |
| **avg** | **2 434.47** | **2 459.83** | **2 414.82** | **2 437.06** | **+1.0% / +0.9%** |

### 1.2 TTFT

| prompt | off run1 | on run1 | off run2 | on run2 | on−off |
|---:|---:|---:|---:|---:|---:|
| 512 | 207.81 | **205.31** | 207.88 | **205.61** | −2.5 / −2.3 ms |
| 2048 | 805.17 | **800.09** | 830.62 | **823.67** | −5.1 / −7.0 ms |
| 8192 | 3 609.27 | **3 564.16** | 3 577.91 | **3 550.61** | **−45.1 / −27.3 ms** |

### 1.3 decode（对照，不参与桶收益）

- off：41.80 / 41.83 t/s
- on：41.72 / 42.00 t/s  
⇒ 无变化（桶只影响 prefill）。

### 1.4 桶命中/显存

on 臂 3 点共 18 个 chunk：

```text
prefill.bucket.hit          = 15
prefill.bucket.hit.t512     = 15
prefill.bucket.bypass.pad   = 1
prefill.eager.chunks        = 3
prefill.eager.tokens        = 3636
prefill.graph               = 1.476 ms/chunk
```

- 6 桶全 `Captured`；桶族 slab **555 MiB**；
- `vram::live`：off **21 040 MiB** → on **21 596 MiB**（+556 MiB）；
- boot 多约 **2.5 s**；
- 尾块/合并块 `pad > 4` 自动回 eager（3 个 chunk）—— 与设计一致。

## 2. 结论

**W4A8 主干：桶的 prefill 收益约 +0.9…+1.0%，TTFT@8192 约 −27…−45 ms（−0.8…−1.3%）。**

对比成本：

- +555 MiB 常驻（24G 卡约 2.3% 显存）；
- boot +2.5 s；
- 6 张图的捕获/生命周期/测试维护；
- `gpu_prefill_bucket_matches_eager` 在干净 main 上仍红（无绿硬等价门）；
- KVMem 下还被 `select_decode` 100% 旁路（另卷分析）。

历史 W4A16/AWQ 的桶收益略高（std 五点 +1.1…+3.6%），但 **2026-10-21 现场补测未复现**：
W4A16 + DFlash2 两对 +1.1% / −2.2%（六点合计 −0.6%，见
[`w4a16-prefill-bucket-ab-2026-10-21.md`](w4a16-prefill-bucket-ab-2026-10-21.md)）。
W4A8 主场景只有 ~1%。

⇒ **对 W4A8 主场景，桶不值得保留；建议退役/删除。**  
若想保留 W4A16 的 3.6% 上限，可只留 W4A16 非 KVMem 的窄路；但代码/显存/维护成本与收益不匹配，**全删更干净**。

## 3. 如果删：建议清理面

1. 配置面：`GraphCfg.prefill_bucket` / `PrefillBucket` 枚举、`validate` 两条硬配、`graphs=auto` 的 geobuckets 派生；
2. 图族：`GraphFleet.prefill` 字段、`PrefillBuckets` / `BucketGraph` / `capture_one` / `replay`；
3. 执行面：`bucket_prefill_chunk` / `bucket_tail`、`prefill.bucket.*` metrics、`prefill_dump_page`（若只桶用）；
4. 池面：`dump_slot` / dump 页（若只桶用）；
5. 档位：`spec-dflash.toml` / `awq27b-3090-max.toml` 的 `prefill_bucket`；
6. 测试/文档：`prefill_bucket_levels_boot` / `prefill_bucket_pick_and_ladder` / `prefill_bucket_usable_conditions` / `gpu_prefill_bucket_matches_eager`，以及 `prefill-graph` 设计卷；
7. **保留** `fi_plan_sync`（decode/verify 还在用），只删桶路的 `needs_plan_sync` 分支。

建议单开清理分支，不要混在当前 `perf-review` 的 tag WIP 里。

## 4. 独立缺陷（比桶更值得先修）

W4A8 生产档带 `head_quant=rtn-int4-g32` 或 DFlash2（W4A16 草稿）时，当前 tag filter
只允许 `quant.w4a8`，而头/草稿实际发 `quant.w4a16` 核 ⇒ **boot 直接拒**。

这不是桶的问题，但它是 **W4A8 主场景能否带 int4 头/DFlash2 上线** 的前置问题；
建议优先于桶的清理立项。

**2026-10-21 后续（已修 + 真机验证）**：`FeatureCaps.head_int4`、
`tag_filter_from_config` 与 DFlash2 草稿 env 已改为“主干 + 辅助 plan 的 tag
并集”（`quant.w4a8` 与 `quant.w4a16` 共存），纯 W4A8 仍拒 W4A16。
单测：`w4a8_main_plus_int4_head_allows_both_weight_facets` /
`w4a8_main_with_int4_head_allows_aux_w4a16` /
`w4a8_main_with_int4_head_and_dflash_allows_aux_w4a16` /
`w4a8_without_aux_keeps_w4a16_out`。

真机启动（GPU2 单张 3090 Ti，4k 短档）：
`profiles/legacy/spec-dflash.toml + testdata/args/w4a8-ab-27b.toml`，
`model.quant=w4a8`、`head_quant=rtn-int4-g32`、`spec.depth=7`（DFlash2 W4A16）：

```text
[kernels] tag 派生激活集:98 项(池写集审计通过)
[selfcheck] marlin_gemm_w4a16_awq ✓
[selfcheck] marlin_gemm_w4a8_awq ✓
[boot] 就绪 http://127.0.0.1:8143(model=qwen3.8-27b)
GET /health → {"ok":true,...}
POST /v1/chat/completions → HTTP 200（8 token 上限，reasoning_content 有输出）
```

`[w4a16] 物化 layers.0..4` 是 DFlash2 草稿的五层，不是主干；主干走 W4A8。
启动约 30s，`live=16.26G`、`free=5.58G`，无 tag filter 拒绝；测试后进程组已收摊。

## 5. 原始产物

- `/tmp/w4a8-off.json` / `/tmp/w4a8-off2.json`
- `/tmp/w4a8-on.json` / `/tmp/w4a8-on2.json`
- `/tmp/owl-w4a8-off3.log` / `/tmp/owl-w4a8-on.log`

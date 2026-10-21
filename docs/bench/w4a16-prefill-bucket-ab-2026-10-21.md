# W4A16 safetensors × prefill 桶 A/B（2026-10-21）

> 目的：补测 W4A16 主场景下 prefill CUDA Graph（`prefill_bucket`）的真实差距。
> 二进制：`/tmp/owl-server-baseline-perfreview-23116c7`（保留桶代码，
> md5 `6855ef60…`；与 W4A8 A/B 同一基线树）。
> 档：`profiles/legacy/spec-dflash.toml` + `testdata/args/base.toml`
> （W4A16 AWQ + DFlash2 d7，fp8 KV，chunk 1024 / tail_merge 1024）。
> 卡：单张 RTX 3090 Ti 42:00.0；greedy、输出 256、固定 prompt seed/nonce。

## 0. 结论

**W4A16 现场复测没有可复现的桶收益，落在噪声带。**

- 第一对：prefill 平均 **1357.3 → 1372.4 t/s（+1.1%）**
- 第二对：**1358.7 → 1328.5 t/s（−2.2%）**
- 六点合计均值：**1358.0 → 1350.4 t/s（−0.6%）**

历史文档里的 W4A16/AWQ “+1.1…+3.6%”在当前 `chunk 1024 + tail_merge 1024`
形状下**没有复现**；本次桶命中只有 **8/11**，尾块合并出的 3 个 chunk 仍回 eager。
且开桶后图 slab **896.5 MiB**，超过旧预算 640 MiB **+256.5 MiB**（boot 已有
`图 slab 预算对账` WARN）。

⇒ 这次补测进一步支持 **退役 prefill 桶**。

## 1. run 1（off → on）

| prompt | off prefill | on prefill | Δ | off TTFT | on TTFT |
|---:|---:|---:|---:|---:|---:|
| 512 | 1346.35 | **1369.49** | +1.7% | 381.95 | 375.56 |
| 2048 | 1397.65 | **1403.32** | +0.4% | 1467.05 | 1461.30 |
| 8192 | 1327.79 | **1344.41** | +1.3% | 6171.43 | 6095.33 |
| **avg** | **1357.26** | **1372.40** | **+1.1%** | 2673.48 | 2644.06 |

decode 基本不变：off 103.25 / on 103.41 t/s（spec 档）。

## 2. run 2（off → on，热态）

| prompt | off prefill | on prefill | Δ |
|---:|---:|---:|---:|
| 512 | 1345.38 | 1272.95 | −5.4% |
| 2048 | 1395.59 | 1400.94 | +0.4% |
| 8192 | 1335.20 | 1311.51 | −1.8% |
| **avg** | **1358.72** | **1328.47** | **−2.2%** |

TTFT 也反向：avg 2662.9 → 2705.3 ms。  
⇒ 第一对的小幅正收益在第二对被热漂移/波动吃掉；**不能称为稳定收益**。

## 3. 桶命中与显存

on 臂（3 点共 11 chunk）：

```text
prefill.bucket.hit             = 8
  t1024                        = 6
  t512                         = 1
  t32                          = 1
prefill.bucket.bypass.pad      = 1
prefill.bucket.pad_rows_skipped= 480
prefill.eager.chunks           = 3
prefill.eager.tokens           = 4640
prefill.graph                  = 0.535 ms/chunk
```

- 3 个 eager chunk = 尾块/合并块（pad > 4），符合旧护栏；
- **图 slab 总账 = 896.5 MiB**（boot WARN：> 预算 640 MiB，+256.5 MiB）；
- 这对 24G 卡是实打实的余量压力；桶关档的图族约 167 MiB（同树 smoke：session 10 + verify 42 + fold 105 + dflash 10）。

## 4. 为什么与历史数据不一致

历史 +3.6% 来自旧形状/旧预算的 `spec-dflash` 测量；当前档：

- `prefill_chunk=1024`、`prefill_tail_merge=1024` ⇒ 尾块合并后常 >1024，
  桶顶 1024 吃不到；
- 桶命中 8/11，3 个 chunk 回 eager；
- 桶 slab 896.5MiB，挤占 KV/激活余量；
- 单次 prefill 只有 ~6s（8192），±1–2% 的热漂移足以淹没桶的固定节余。

## 5. 与 W4A8 对照

| 场景 | 桶收益（现场） | 结论 |
|---|---:|---|
| W4A8 主干 | +0.9…+1.0% | 边际，已退役 |
| W4A16 + DFlash2 | +1.1% / −2.2%（两对） | **不稳定，已退役** |
| GGUF Q4_K_M | +0.95% | 边际，已退役 |
| KVMem | 0（被 select_decode 旁路） | 已退役 |

⇒ W4A16 的现场结果没有改变退役裁决；反而说明桶的收益对 chunk 形状/热态非常敏感，
不是生产主线的可靠杠杆。

## 6. 原始产物

- `/tmp/w4a16-off.json` / `/tmp/w4a16-off2.json`
- `/tmp/w4a16-on.json` / `/tmp/w4a16-on2.json`
- `/tmp/owl-w4a16-bucket-off.log` / `/tmp/owl-w4a16-bucket-on.log`

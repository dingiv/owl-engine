# FI prefill 半精度累加档(`fi-fp16acc`)—— 30 系 attention 的算术宽度

> 2026-11-02 立项即结案。本文 = **决策 + 数字 + 判据**;逐轮施工/踩坑实录在本地台账
> `roadmap.local/26-11-02-fi-fp16累加-30系attention.md`。

## 1. 为什么只有这一条路

长 ctx prefill 的 GPU 时间账(2026-10-31 nsys,139k 冷 prefill):**FI paged prefill 注意力
47.7%**(第一热点)、marlin W4A8 24%、TP allreduce 13.1%、GDN 7.5%。

注意力这一项已经**没有结构性空间**:
- `CTA_TILE_Q=128` 在 hd=256 上被**寄存器墙**挡死(1088 ≫ 256,`prefill-attention-swap.md` §5.2);
- 实测 54.1 TFLOPS ≈ **fp32-acc 峰值的 68%**(3090 Ti,hd256,ctx 139k,T=512);
- fp8→f16 一次性反量化(`[kernel] prefill_shadow`,2026-10-31)已经把"反量化税"收掉(−10.9% e2e)。

⇒ 剩下唯一的大头是**张量核算术宽度**:消费级 GA10x 的 **fp32 累加只有半速**
(fp16-acc = 2× 峰值)。而默认路径正是半速档 `mma.sync...f32.f16.f16.f32`。

**被否掉的两条**(都有人试过,不要重挖):

| 方案 | 判据 |
|---|---|
| INT8 KV + int8 张量核 | famp `docs/ROADMAP.md` NO-GO:int8-QK 只对 **D≤128** 的纯全注意力有赚,**D=256 不支持**;int8-PV「softmax 概率 int8 不可接受」。我们 hd=**256** |
| 换 Triton/vLLM 后端 | 本仓无 Triton attention 后端;vLLM 的 Triton 路也没有 int8 KV,Ampere 上同样要交反量化税 |

## 2. 档位与用法

```bash
cargo build --release -p server   # 缺省 = 开(2026-11-02 用户裁决:`apps/server` default 带出)
cargo build --release -p server --no-default-features --features arch-sm86   # 显式要 fp32 基线
cargo build --release -p server --no-default-features --features arch-sm120  # 换代际(必须摘掉本档)
```

**⚠️ 代际**:half 累加走 `mma...f16.f16.f16.f16`(m16n8k16)⇒ **sm_80 起**;
`prefill.cuh` 里有 `static_assert(__CUDA_ARCH__ >= 800)`,sm_75 会在编译期响亮拒(不是运行期哑弹)。
`fi-fp16pv` 更严(只验过 sm_86,且其模板参依赖 `__CUDA_ARCH__`)⇒ **多代际 fatbin 与两档不可共存**,
一律 `--no-default-features` 整体摘掉。

三层转发:`apps/server` → `owl-cuda` → `owl-kernels`。**是 Cargo feature 不是 env**
(构建档必须进缓存键,同 `arch-smXX` 的裁决)。feature 在 ⇒ `build.rs` 给 FI 三个 TU 加
`-DFP16_QK_REDUCTION_SUPPORTED -std=c++20 --expt-relaxed-constexpr`,并**同时**把
`owl_fi_attention_common.h` 的 `OWL_FP16_QK_REDUCTION` 置 true(C++ 侧单一真源)。

核内形态:QK^T 走 `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16`(累加器 = half),
online softmax 整条走 half2(`ex2.approx.f16x2` / `hmax2` / `shfl.bfly.b32`),
PV 仍是 fp32 累加(`o_frag` 未改)。

## 3. 数字(3090 Ti / hd256 / hkv4 / page32 / **ctx=139264**)

同树同卡同次 A/B(feature on/off 各建一次):

| T | KV 池 | fp32-acc | fp16-acc | 变化 |
|---|---|---:|---:|---:|
| 512 | f16(**生产路径:影子池**) | 32.46 ms | **25.56 ms** | **1.270×** |
| 807 | f16 | 42.15 ms | **33.59 ms** | **1.255×** |
| 2048 | f16 | 104.24 ms | **84.46 ms** | **1.234×** |
| 512 | fp8 | 39.90 ms | **33.76 ms** | **1.182×** |
| 807 | fp8 | 51.83 ms | **44.59 ms** | 1.162× |
| 2048 | fp8 | 128.71 ms | **110.33 ms** | 1.167× |

短 ctx(T=512,fp8):ctx 2048/4096/8192/16384 = **0.497 / 1.026 / 2.083 / 4.200 ms**
(2026-11-01 基线 0.568 / 1.176 / 2.432 / 4.829)。

## 4. 数值判据

| 形状(host oracle) | fp32-acc | fp16-acc |
|---|---:|---:|
| **fp8 池**(= 生产 KV dtype):T=1/ctx=3072 | 2.7e-4 | 5.5e-3 |
| fp8 池:T=8/ctx=3072 | 5.0e-4 | 9.5e-3 |
| fp8 池:T=54/ctx=3072 | 5.0e-4 | 1.23e-2 |
| fp8 池:T=128/ctx=512 | 4.7e-4 | 1.23e-2 |
| fp8 池:末页半满族(T=1/ctx=53·1057·3073·3080) | ≤5e-4 | 5.5e-3 |
| **f16 池**(满 15 bit 数据,|S| 大得多):T=8/ctx=3072·32768 | **9.3e-4** | **4.0e-2** |

⇒ fp8 池(生产)附加噪声 **1.2e-2 最坏 / 2~5e-3 典型**,cos = **1.00000**,零值 0。
f16 池(大 logits)最坏 **4.0e-2 = 3.6% 相对**。

**噪声全部来自 QK^T 的 half 累加**(逐 mma 把累加器舍到 half,|S| 越大越糟):
两条"降噪"改法实测**零改善**且慢 1%,已写进代码注释禁止再试 ——
(a) 先减后乘 `(S−m)*scale`;(b) exp 参数走 float(`ex2.approx.f32`)。
⇒ 想再降噪只能回 fp32 累加(即关本 feature);**要上生产必须走 e2e 质量门**。

探针:`FI_PF_GATE=2e-2`(缺省仍是 1e-2,只对 fp16 档放宽):

```bash
OWL_TEST_DEVICE=<BDF> FI_PF_GATE=2e-2 \
  cargo test --release -p owl-kernels --features device,flashinfer,fi-fp16acc \
  --test fi_prefill_probe fi_prefill_speed_and_parity -- --test-threads=1 --nocapture
```

## 4.1 与 `fi-fp16pv` 叠加(2026-11-02 合流后实测)

两个档正交:本档改 **QK^T + softmax**,`fi-fp16pv` 改 **PV 的 `o_frag` 累加器**。
同树同卡(3090 Ti,hd256,hkv4,page32,**ctx=139264**):

| T | f16 池:缺省 | 仅 acc | 仅 PV | **acc + PV** |
|---|---:|---:|---:|---:|
| 512 | 31.4 ms | 25.7 | ~26.9 | **21.7 ms(1.45×)** |
| 807 | 42.2 | 33.6 | — | **28.2 ms(1.50×)** |
| 2048 | 104.8 | 83.7 | — | **73.4 ms(1.43×)** |
| 512 / 2048(fp8 池) | 39.9 / 128.7 | 33.3 / 110.8 | — | **28.1 / 90.8 ms(1.42×)** |

数值(叠加档,fp8 池全形状套件):`max|Δ|` ≤ **1.24e-2**、**cos ≥ 0.9999966**、
`mean|Δ|` ≤ **1.02e-3** ⇒ PV-half 在 QK-half 之上**几乎不追加误差**
(两者都是同一类"half 累加"噪声,不叠加放大)。

构建:`cargo build --release -p server`(**两个都已进缺省**,2026-11-02 用户裁决)。

## 5. 上游该档的两个坑(升级闭包必看)

1. **它在本仓从未被编译过**:需要闭包里不存在的 `fp16.h`(上游那份要 `<bit>`/C++20 +
   `boost/math/ccmath/fabs.hpp`)。我们补的是**硬件 `cvt` 版 shim**(OWL 自有,
   `cu/flashinfer/include/flashinfer/fp16.h`)——顺带把 device 侧十几条位运算变成一条指令。
2. **它带着一个数值 bug**:`logits_transform` 把 `half` 直接喂给
   `fp16_ieee_to_fp32_value(uint16_t)`,重载决议走 `__half::operator unsigned short()`
   (**数值**转换:负值饱和 0、正值取整)⇒ logits 全毁 ⇒ **S 全 0 ⇒ 注意力退化成均匀平均**
   (实测 T=1/ctx=64:输出与真值 cos=0.04、范数 1/88,而"能跑出数")。
   修法 = 两行 `__half_as_ushort` / `__ushort_as_half` 显式给位型。

⇒ **判据纪律**:"能编、能跑、出数"不是数值门;门必须是 `max|FI−host|` + **cos**。
2026-11-01 那次"fp16 归约慢 17%"的负结论也因此只对一半:0.6.18 是补偿式路径,
0.7.2 是干净 half 累加,慢的是**软件转换函数**,不是归约本身。

## 6. 还没做

1. **PV 也换 half 累加**(`o_frag` → `uint32_t[4]`,famp patch 0007 的路子):PV 占 mma ~48%
   ⇒ 再 ~1.25×(合计 1.5× 量级)。代价 = 25 处触点 + 长 ctx half 累加器精度门。
2. **e2e 验收**:3080 TP2 生产档真实 dump 回放 + greedy 恒等/数值题,确认 1.2e-2 附加噪声
   在 27B 上无感。**验收前不要把 feature 当默认**。
3. `NUM_MMA_KV` 扫档(预算内 auto 已取 4/1,预期无油水)。

## 7. 3080 TP2 质量门(2026-11-02 实测,通过)

现场 = 生产档 `qwen38_27b-3080_tp2.toml`(W4A8 / DFlash2 d7 / fp8 KV / 影子开),
两臂独立 boot、同一套脚本(gsm8k greedy / needle / `llm_speedtest std`)。

| 指标 | ref(缺省) | ref 第二遍 | exp(acc+PV) |
|---|---:|---:|---:|
| gsm8k n=100 | 95/100 | **96/100** | 94/100 |
| pred 与 ref 第一遍逐字相同 | — | **97/100** | 95/100 |
| needle 60k TTFT | 29.1 s | — | **26.8 s(−7.9%)** |
| needle 100k TTFT | 56.3 s | — | **49.9 s(−11.4%)** |
| std prefill avg(≤8k) | 2402.9 t/s | — | 2401.0 t/s(≈0) |

**判据**:不看"两臂逐字节一致",看**基线自身重复性** —— ref 连跑两遍自己就有
3/100 条链不同(97/100 逐字)、准确率 ±1 题;跨档差(95/100 逐字、1 题翻转)在噪声带内。
噪声源 = 隐式会话 0 的前缀缓存状态 + 投机解码平局翻转,不是核精度。
唯一翻转题 gsm8k #37("卖光玩具"陷阱题)**孤立单问 ref 也答 0**(与 exp 批跑同)⇒ 非本档引入。

**速度**:≤8k 无收益(GEMM 主导);收益区在长 ctx(60k −7.9% / 100k −11.4% TTFT),
与核级 1.42~1.50× + nsys"注意力 47.7%"份额一致。
报告与原始数据:`docs/bench/fp16-pv-2026-11-02.md`(workspace)§3080 TP2。

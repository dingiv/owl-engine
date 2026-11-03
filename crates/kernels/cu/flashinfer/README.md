# `cu/flashinfer` —— FlashInfer paged prefill(生产核;**上游 v0.7.2.post1**)

**来源(2026-11-01 升级)**:`repos/flashinfer` @ `beabf5004`(`v0.7.2.post1`,
2026-10-23)的 **paged prefill 头闭包**。上一版是 `fdc56343`
(nightly-v0.6.18-20260819-317,2026-09-17)—— 注意它的 `version.txt` 实际是
**0.7.0**,即本次是 0.7.0-nightly → 0.7.2.post1 的补丁级跟进,不是 0.6→0.7。
原先 vendor 的是 `guoqingbao/flashinfer` fork @`0f06c230`(2026-08-07)。

**2026-11-01 升级面**(闭包集合不变,仅 3 头有差异):
`attention/scheduler.cuh`(graph 档 `block_valid_mask` 修复 `e0a18900c`)、
`pos_enc.cuh`(RoPE 网格优化;owl `PosEncodingMode::kNone` 不走)、`utils.cuh`(+3 行)。
prefill 核体逐字未变 ⇒ adapter **零改动**、速度中性(15/15 输出逐字节一致)。
实测与决策见 `docs/bench/flashinfer-0.7-upgrade-2026-11-01.md`。

**2026-11-02 owl 本地补丁**:`include/flashinfer/attention/prefill.cuh` 带
**FP16-PV** 实验补丁(`patches/fp16-pv-0.7.2.patch`)。PV 累加器可从 fp32
切到 half(`f16f16f16` HMMA + half2 rescale),由 Cargo feature
`fi-fp16pv` → `-DOWL_FP16_PV=1` 打开;**默认宏 0,核体走原 fp32 路径**。
只对 sm86 GeForce 的 f16/fp8 非 VO-split 预填生效,bf16 硬拒。详见
`docs/devlog/kernels/fp16-pv-prefill-2026-11-02.md`。升级 FI 时必须重放
该 patch(或按同文件重锚)。

**为什么换**(用户令"换其他引擎的同类核 → 先单测 → 再替换"):
1. **实测 1.4×**:同形状(T=512 × ctx,hd256,GQA 24/4,page32,kNHD,**fp8 池**)
   上游 0.112/0.568/1.176/2.432/4.829 ms @ctx 512/2048/4096/8192/16384,
   旧 fork 0.152/0.798/1.657/3.405/6.816 ⇒ **1.36–1.41×**(f16 池 1.8×);
   引擎内 CUPTI 逐核注意力 **49.6 vs 63.0 ms(−21%)**。
2. fork 的 **plan/split-kv 是坏的**(`docs/devlog/perf/session-2026-10-14.md` §1.2
   单测复现 `max|FI−host|=1.39`),且它是一条**个人分支**;上游 plan 用
   **q 瓦片伪批**(本形状 `padded_batch_size=48` ⇒ 192 CTA)把单请求铺满 SM。

**闭包**(25 头 / ~0.85 MB;**无 cutlass**;`nvcc -M` 扫一个含
`BatchPrefillWithPagedKVCacheDispatched` 的 TU 即复现,清单见 `closure.txt`)。

## 半精度累加档(2026-11-02;feature `fi-fp16acc`)

消费级 GA10x 的 **fp32 累加只有半速** ⇒ 默认的 `mma...f32.f16.f16.f32` 是半速档。
开 `fi-fp16acc` 后 QK^T 走 `mma...f16.f16.f16.f16`,整条 online softmax 也走 half2
(`ex2.approx.f16x2` / `hmax2`)。

| 形状(3090 Ti / hd256 / hkv4 / ctx=139264) | fp32-acc | **fp16-acc** |
|---|---:|---:|
| T=512,f16 池(**生产路径**) | 32.46 ms | **25.56 ms(1.270×)** |
| T=807 / T=2048,f16 池 | 42.15 / 104.24 ms | **33.59 / 84.46 ms(1.255× / 1.234×)** |
| T=512 / 807 / 2048,fp8 池 | 39.90 / 51.83 / 128.71 ms | **33.76 / 44.59 / 110.33 ms(1.18×)** |

数值:host oracle `max|FI−host|` 5e-4 → **1.2e-2 最坏 / 2~5e-3 典型**(cos=1.00000;
远小于 fp8 KV 自身量化噪声)。全案见 `roadmap.local/26-11-02-fi-fp16累加-30系attention.md`。

**⚠️ 这个上游档在本仓以前从未被编译过,而且带着一个数值 bug**(`half` 被当整数
截断 ⇒ S 全毁 ⇒ 注意力退化成均匀平均)。本目录的四处 in-tree 改动就是为了让它能编 + 算对,
**升级闭包时必须重新对照**(详见下表)。

### in-tree changes(FROZEN 档的登记项,同步 `THIRD_PARTY.md` §2)

| 文件 | 改动 |
|---|---|
| `include/flashinfer/fp16.h` | **新增文件**(OWL 自有,非上游拷贝):`fp16_ieee_{to,from}_fp32_value` 的**硬件 `cvt` 版**(上游是软件位运算 + boost;device 侧十几条位运算 vs 一条 `cvt`) |
| `attention/prefill.cuh` | ① `logits_transform` 的两处 half↔float 往返**显式给位型**(`__half_as_ushort`/`__ushort_as_half`)—— 原写法 `half` → `uint16_t` 形参走的是**数值**转换(负值饱和 0、正值取整)⇒ S 全毁;② `getNegInf<half>()` 用常量位型 `0xFC00`(原运行期转换函数不是 constexpr);③ hd≥512 vo-split 路 3 处 `half` 二义显式转 float(只影响编译面) |
| `attention/variant_helper.cuh` | `OutputTransform` 的 `m != -inf` 在 `T_M=half` 时二义 ⇒ `float(m)` |
| `build.rs`(非闭包) | feature 在 ⇒ FI 各 TU 加 `-DFP16_QK_REDUCTION_SUPPORTED -std=c++20 --expt-relaxed-constexpr` |

**ABI 逐字未变**(`owl_fi_prefill_plan` / `owl_fi_prefill_run` /
`owl_fi_prefill_run_fp8kv`;15×int64 plan blob、槽序、`FI_SPLIT_KV`/`FI_DEBUG`
旋钮全保留)⇒ Rust 侧(`family/server/探针`)**零改动**。

## 40/50 系原生 FP8 MMA(2026-11-04;arch-sm89/120)

`owl_fi_attention_fp8mma.cu` 是第四个 FI TU:实例化
`BatchPrefillPagedParams<__nv_fp8_e4m3, __nv_fp8_e4m3, half>`，Q/K/V 全走
`m16n8k32 e4m3.e4m3.f32` MMA，不再把 K/V 反量化成 half。Q 由同 TU 的
`owl_fi_quant_f16_to_fp8` 预先量化(scale=1.0，qk-norm 后 RMS≈1)。

- **选臂**:`KernelPick` 按运行期 `OpEnv.hw.arch` 选 —— Sm89/Sm120+FP8 → 原生臂，
  其他代际 → 旧 dequant 臂；`build.rs` 只在 `single_arch() >= Arch::Sm89` 时
  编入设备符号（sm86 不编，sm120 编）；选到 native 而二进制未编 = 结构化拒；
- **布局约束**:沿用上游 `INVALID_FP8_MMA` —— `NUM_WARPS_Q == 1`、
  `CTA_TILE_KV == 64`、`USE_VO_SPLIT`、float QK 累加。计划侧
  `owl_fi_prefill_plan_fp8mma` 强制 `CTA_TILE_Q=16`；
- **实测(4080 SUPER)**:std 1024–8192 均值 3113.4 t/s(旧 3029.8，+2.8%)，
  32k 2844.5(旧 2379.3，+19.6%)，65k 2434.6(旧 2099.4，+16.0%)；decode 不变。
- 全案:`docs/bench/fp8-sm89-native-mma-2026-11-04.md`。

**本目录文件**
| 文件 | 说明 |
|---|---|
| `include/` | 上游闭包 + owl 两个实验档的 in-tree changes(**两个宏都缺省 0 ⇒ 默认档逐字等价**;升级 = 重扫闭包 + 重放 `patches/fp16-pv-0.7.2.patch` + 对照下表) |
| `patches/fp16-pv-0.7.2.patch` | owl 本地 FP16-PV 改动(相对上游 0.7.2.post1) |
| `owl_fi_attention_fp8mma.cu` | 40/50 系原生 FP8 MMA(Q 量化 + e4m3 Q/KV run;arch>=sm89 编) |
| `owl_fi_attention.cu` | owl adapter(换源记录见文件头注:plan 前置 `PrefillPlanWorkspaceSize` 尺寸校验 / `kv_dtype_bytes` / params 改逐字段赋值 / 模板加 `SAME_KV_STRIDES` / fp8 免宏) |
| `bench/` | 候选核单测台架(不参与构建):`FI_SRC=$(cd .. && pwd) ./run.sh`(f16)/ `FP8=1 ...`(fp8 池) |

**全案**:`docs/devlog/kernels/prefill-attention-swap.md`(§1 三候选数字表 /
§3 集成配方 / §6 落地与验收证据)。

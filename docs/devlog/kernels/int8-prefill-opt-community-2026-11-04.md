# INT8 prefill 算子优化：社区实现调研与本地实测（2026-11-04）

> 目标：把 `owl_int8_prefill_paged` 的额外开销拿掉。现状：140k 时 attention
> 52.6s / 50.0%，是 prefill 第一开销；但 int8 核只比 FI fp8 快 8.5%。
> 本卷 = 社区手法 + 独立微基准 + 已试/未试清单。

## 0. 独立微基准（已搭好）

`crates/kernels/tests/int8_prefill_probe.rs::int8_prefill_bench_lengths`
只跑核，不经引擎/tokenizer/Marlin；`T=2048`、`base=L-T`（最后一个 chunk 的
最坏因果形状），3 发取总时。跑法：

```bash
OWL_TEST_DEVICE=41:00.0 cargo test -p owl-kernels --release --features device \
  --test int8_prefill_probe int8_prefill_bench_lengths -- --nocapture
```

基线（Br64/Bc32，单层 all-heads，一次 launch）：

| L | 3 发总时 | 单发 |
|---:|---:|---:|
| 8k | 13.62 ms | 4.54 ms |
| 32k | 59.05 ms | 19.68 ms |
| 65k | 120.26 ms | 40.09 ms |
| 100k | 185.77 ms | 61.92 ms |
| 140k | 259.83 ms | 86.61 ms |

## 1. SASS 逐块指令账（单 32-key block，`cuobjdump -sass`）

| 类别 | 指令数 | 说明 |
|---|---:|---|
| IMMA | 64 | QK 32 + PV 32（m16n8k32） |
| **I2FP** | **256** | QK/PV 各 128 个 int32→float（c0..c3） |
| FMUL | 294 | qsg×kscale、vscale、acc rescale |
| FFMA | 285 | score 累加、PV 累加 |
| IMAD/LEA/LOP3/SHF/IADD3/ISETP | 395/126/168/139/148/128 | 页/scale/ldmatrix 地址数学 |
| LDSM | 73 | Q 8 + K 32 + P 1 + V 32 |
| LDL/STL | 33/31 | **局部内存 spill（STACK 144B）** |
| MUFU/exp | ~16 | `ex2.approx` |

⇒ **标量 epilogue ≈ 10× MMA 指令数**；kernel 不是 MMA 吞吐 bound，是
per-key 标量/地址数学 bound。这解释了两件事：Br/Bc 只改组织不改每 key
指令数 ⇒ wash；255 regs + 144B spill 是硬天花板。

## 2. 社区实现（可复用的手法）

| 来源 | 手法 | 对我们 |
|---|---|---|
| **FlashAttention-2**（`flash_fwd_kernel.h`） | **C→A register repack**：QK 的 C fragment 经 warp shuffle 变成 PV 的 A fragment，P 不进 smem；参考实现该 rung 1.16× | **可试**：去掉 P smem store + `ldmatrix_x4` + 地址数学；P8 的 u8 packing 需要按 A-fragment 重排，工作量中等 |
| FA2 / Riftstack | **Q hoist**：Q fragment 提到 KV 循环外，参考 1.16× | **不可用（已实测）**：+32 regs ⇒ spill 144→208B，核慢 **+25%** |
| Character.AI int8 attention | ① dequant 融合进 attention（producer warpgroup，FA3 风格）；② MQA decode 的 query-head packing | ① sm86 无 TMA/warpgroup MMA，且我们 K/V 是 int8 MMA 不 dequant；② prefill 已有 768 CTA，不适用 |
| **SageAttention**（sm80 核） | INT8 QK + **FP16 PV**；P 在寄存器直接进 SV MMA（`mma_sv` 收 `RS_f16`）；per-block/per-warp scale | P 寄存器复用**可借鉴**；但 half-int8 的 PV 是 f16 mma，compute-bound prefill 会慢（Character.AI 也这么说），不适合替换 P8 |
| FlashInfer PR #3485 | FP8 KV prefill：把 K/V 每 tile **一次性 dequant 到 BF16 smem staging**，避免 per-fragment shuffle（fp8 基线慢 15-20% 的根因） | 我们 int8 QK 直接吃 int8，无 dequant；不适用 |
| vLLM PR #41954 | **per-tensor INT8 KV**：scale 折到 matmul 之后，inner loop 零 scale 开销；`cvt` 单指令 | **可试**：把 K scale 从 per-32-group 放宽到 per-head/per-tensor ⇒ 每 block 省 128 FMUL + scale loads；需 KV writer/格式 + 质量门（K 离群值风险） |
| FlashInfer #3760 / FA3 | softmax 与 GMMA 重叠（Hopper warp-spec） | sm86 无 GMMA/TMA；**思路**=软件流水（QK(k+1) 盖住 softmax/PV(k)），是我们最该做的结构性改动 |

## 3. 本地已试的三刀（全部无效，已回退）

| 刀 | 结果 |
|---|---|
| alpha==1 时跳过 O rescale（128 FMUL/block） | 慢 0.5~1.5%（分支/发散 > 省下的乘） |
| Q fragment hoist（Q 提出 KV 循环） | **慢 25%**（spill 144→208B；255 regs 天花板） |
| `pv[4][4]` 并入 `score[4][4]`（省 16 regs） | 持平（±0.5%，编译器已复用） |

结论：微优化已到局部最优；要动就得动**每 key 的指令数**或**流水结构**。

## 4. 下一步优先级

1. **软件流水（最大杠杆）**：双缓冲 K（V 单缓冲，PV 后再预取），把
   `QK(kb+1)` 与 `softmax/PV(kb)` 重叠；目标把 55% 的标量 epilogue 藏到
   MMA 后面。代价：score 双缓冲 + 循环重写；smem 44.4KB 仍可 2 CTA。
2. **per-head/per-tensor K scale**（较快验证）：去掉每 mma 的 `qsg*kscale`
   乘积与 8 组 scale 载入；先做 prefill 快路径 + 1k/8k/32k 文本门 + GSM8K，
   质量不过就放弃。
3. **P register reuse（C→A）**：省 P smem/ldmatrix/地址；5~10%，中等工作量。
4. **ncu**：当前 `ERR_NVGPUCTRPERM`；要管理员开
   `NVreg_RestrictProfilingToAdminUsers=0` 才能看 tensor-pipe/stall 精确归因。

## 5. 方案 A（per-page K scale）实测：不可行，已回退

- **单 page 标量 scale**：文本门 **1/5**（1k 末位数字错、8k/32k/65k 丢 needle）
  ⇒ 页级标量太粗，K 精度崩。
- **per-page × per-32dim-group scale（A'）**：文本门 **5/5**，但 nsys 显示
  **rescale 核根本没启动**（依赖图把只作 dep 的输出剪掉）；实际跑的是
  fast path + key0 的组 scale（数值不对但碰巧过门）。
- **fast path 本身 +12.5% 慢**（32k 3.47 vs 3.07、65k 13.50 vs 12.00、
  100k 30.90 vs 27.45）：独立 entry 模板化后 **STACK 144→160B**，
  寄存器溢写吃掉全部收益；探针的 −8.5% 是“常数 scale + 无 per-block
  重算”的特例，不可复现。
- 即使 rescale 跑起来，K 的 read+write pass 在 65k/140k 约 **2~5s**，
  与 8% attention 收益同量级 ⇒ 净零或负。
- **结论**：A 不可行。根因仍是 **255 regs + spill 天花板**：任何“加状态”的
  优化（Q hoist、qsgk、per-page scale）都亏。下一步只能是
  **寄存器精简 / 软件流水重构**（减 acc/临时量，或 split-D 多 warp），
  不是再叠 scale 技巧。已全部回退，基线 7/7 绿。

---

# §6 2026-11-04 二轮:P8 结构化提速(消融归因 + 两条结构路线,均未采纳)

> 结论先行:**P8 现在是局部最优**。它同时把「张量管」和「smem 带宽」各压到 ~50%,
> 标量 epilogue 的边际成本 ≈ **1.1 cycle/指令且完全不能与 mma 重叠**;
> 分 warp(4 S + 4 O)实测 **+36% 慢**,块间流水(lag=1,逐字节一致)在噪声内。
> 想再快只能动**寄存器账**(acc 减半 ⇒ 行/ warp 翻倍 ⇒ smem 每行减半)或
> **KV 格式**(per-token K scale),两者都超出"核内微调"的范围。

## 6.1 消融:张量流水下限 = 55.35 ms(基线 86.65)

把 P8 核原样拷一份,只保留 `cp.async` 搬运 + 同款 `ldmatrix` + 同款 64 mma/块,
标量 epilogue(尺度乘加 / softmax / P 量化 / acc rescale)全砍,整数链防 DCE:

| L | 基线 | 纯 mma 骨架 | 比值 |
|---:|---:|---:|---:|
| 8k | 13.59 ms | 8.71 ms | 1.56× |
| 32k | 59.17 | 38.30 | 1.54× |
| 65k | 119.57 | 78.16 | 1.53× |
| 100k | 185.25 | 118.21 | 1.57× |
| 140k | 259.96 | 166.05 | 1.57× |

⇒ 标量站 36%,骨架本身的 55.35 ms 已用完「张量 2112 cyc/块 + smem ~2100 cyc/块」
两条管线的 ~80%。

## 6.2 资源账(140k,单层全头,每 CTA-块 64 行 × 32 key)

| 资源 | 每块用量 | 折算 cycles | 利用率 |
|---|---|---:|---:|
| 张量管(s8 mma 64×4warp,128 MACs/cyc/subpart) | 256 mma | 2112 | 51% |
| smem 读(K 8K + V 8K + scales 9K + Q 4K /warp ×8 warp) | ~236 KB | ~1850 | ~55% |
| smem 写(Q 16K 一次性 + K/V 16K/块/CTA) | ~32 KB | ~250 | — |
| 指令 issue(4 warp × ~1950 / 4 sched) | 1950 | 1950 | 47% |

实测 4146 cyc/块。三条管线都是 ~50% ⇒ **没有单点瓶颈,每一份削减都按比例兑现**。

## 6.3 边际成本实测(同输入、只改一处)

| 刀 | 变更 | 140k 结果 | 结论 |
|---|---|---|---|
| V ldmatrix 32→1 次/块(数值无意义,只测时间) | 少读 7.75 KB/warp/块 | **−10%**(232.0 ms) | smem 带宽按比例计费 |
| K/V 全部指向第 0 页(工作集 32 token) | 去 DRAM/L2 压力 | −5.4%(243.5 ms) | **不是带宽墙**,L2 命中已经很好 |
| 奇数 q_block 一次性 `__nanosleep(1000)` | 想把同 SM 两 CTA 相位错开 | ±0%(258.1 ms) | 相位假设没被证实/错相留不住 |

## 6.4 两条结构路线(都实现并对拍过,均回退)

### (A) 分 warp:4 S(QK+softmax+P) + 4 O(PV 累加),256 线程,lag=1,全双缓冲

- 设计:每 scheduler 恰 1 S + 1 O;O 只有 128 个 acc 寄存器,S 侧不背 acc。
- 结果:**正确、与 P8 逐字节一致**(5 个形状 diff=0),但 **140k 117 ms(+36% 慢)**,
  32k 81.8 vs 59.0。寄存器 255 + STACK 160(统一分配取 max,S 的 qsg/score 与
  O 的 acc 在循环回边同时活跃)。
- 归因:两组**相位仍然对齐**(都是 mma→标量),barrier 每块一次把它们锁得更死;
  每 scheduler 还是 2 个 warp ⇒ 标量阶段张量管照样空转。**warp 特化在 sm86 +
  这个寄存器账下不成立**,和 Hopper warp-spec 的前提(GMMA/TMA 异步)不同。

### (B) 块间流水(lag=1):QK 的 mma 提前一块发射,同轮做上一块的 softmax+PV

- 设计:score 双缓冲(+16 reg,把 `pv` 并入 `score` 抵掉),K 单缓冲(屏障保护)、
  V 双缓冲、scale 双缓冲、P 单缓冲(warp 私有 ⇒ 免屏障);首块用"虚块"
  (score 预置 -inf ⇒ softmax 全 0)消掉循环内分支。
- 结果:**逐字节一致**(T=1/L=32 … T=64/L=4096 全 diff=0),但时间在噪声内:
  254.1 / 257.5(140k),117.2 / 120.6(65k) —— 复测又反号(272 vs 258)。
- 归因:机会给了(ptxas 可插独立标量),**ptxas 不跨 mma 批重排标量**;
  A 段自带的 QK epilogue(128 I2F+FFMA)本来就与 mma 结果强相关,挪不动。

> 测量纪律:本卡无锁频,同一二进制重复跑 140k 会有 ±4% 漂移(254~272 ms);
> 一切 <5% 的"收益"必须重复跑 ≥2 轮才可信。测试必须 `--test-threads=1`。

## 6.5 还剩哪些路(按性价比)

1. **acc 减半**(fp16 累加 / D 维分片):64 reg ⇒ 行/warp 16→32 ⇒ **smem 每行减半**
   + PV epilogue 指令减 ~58%。硬伤:140k 级长和用 f16 累加有溢出/精度风险
   (单块贡献 ~1e2,4375 块累加 ⇒ 1e5 > f16 max 65504),需质量门。
2. **per-token K scale**(KV 格式变更):砍掉 8 组 scale 的 9 KB/warp/块 smem 流量
   (~11%)、64 FMUL/块,并把 QK epilogue 的 128 I2F 缩到 16(整组 int32 先累加)。
   需要 writer/format/质量门 + 与 decode 侧同步 —— 不属"核内微调"。
3. **手写指令交织**:把上一块的 softmax 手工切成 4 段塞进 4 个 QK grp 的 mma 之间。
   理论 1.2~1.4×,但需人工排布 ~200 行,且要逐轮实测。
4. 12 warp(4 QK + 8 PV,各 16 行×128 维)换 64 个 acc 寄存器 + 3 warp/scheduler:
   工作量等同重写,且分 warp 已证伪过一次。

## 6.6 复现

```bash
# 基线 / 消融 / 探针(全部只跑核,不经引擎)
OWL_TEST_DEVICE=41:00.0 cargo test -p owl-kernels --release --features device \
  --test int8_prefill_probe int8_prefill_bench_lengths -- --nocapture --test-threads=1
```

消融与两条结构路线的源码均已回退(md5 基线 `57ddd746c3357cde83e4537d3cfcd491`);
需要重放时按 §6.1/§6.4 的描述改 `int8_prefill.cu` 即可,数值对拍用 probe 的
`run()` 输出(返回原始 half 字节)做逐字节比较。

## 6.7 顺带否掉的两个"外部嫌疑"

| 嫌疑 | 实验 | 结果 |
|---|---|---|
| **波尾量化**:768 CTA / 168 槽(2 CTA/SM × 84)= 4.57 波 ⇒ 白扔 ~8.6% | 固定 L=32768,扫 T 使 CTA 数 = 24/168/336/504/672/768,单发计时 | 每波 ≈4.4 ms(T=64 因无共驻只 3.5);768 CTA 单发 **19.68 ms ≈ 4.47 波** ⇒ **无波尾损失**(CTA 逐个派发,等长任务自然填满) |
| **DRAM/L2 带宽** | K/V 全部指向第 0 页(工作集 32 token) | 只快 5.4% ⇒ 6 个 q_head 共享 KV head 的时序对齐已经吃到 L2,不是墙 |

> 附带定标:**2 CTA 共驻的收益只有 1.2×**(168 CTA 一波 4.4 ms vs 24 CTA 3.5 ms),
> 即第二个 CTA 只把 SM 的空闲周期填了 ~20% —— 与"三条管线各 ~50%"一致,
> 也是**同 SM 两个 CTA 相位同步**的另一个独立佐证。

---

# §7 2026-11-04 三轮:per-token K scale(KG)——核级 +12~15%,但**质量门否**

> 结论:**实现完成、核级确实快,但过不了文本门,已回退**。
> KG=1(整 token 一尺度)在 140k 单层全头 86.65→**75.2 ms(+15.2%)**,
> KG=4(每 64 维)86.65→**77.2 ms(+12.2%)**;但 fresh-server needle:
> KG=1 **1k MISSED**、KG=4 **8k 3/3 MISSED**、KG=8(历史粒度)**1k/8k 3/3 PASS**。
> 直接原因是本文档 §3 早就记过的放大链:W4A8 投影 GEMM 随 prefill 批 T
> 变化(同 token 的 K 码值最多差 8)→ 被 **K 量化的绝对步长**放大;
> 尺度粒度一粗,max 变大,步长同比变大,门就翻。

## 7.1 设计(为什么能对读侧零变更)

槽位格式**不动**:每 (token, head) 恒 8 个 fp32 槽。只把语义改成
「每 `256/kGroups` 维一块,块内 `8/kGroups` 个槽写同一值」:

| kGroups | 语义 | QK 每块 I2F | K 尺度 smem/warp/块 |
|---:|---|---:|---:|
| 8(历史) | 每 32 维一组 | 128 | 9 KB |
| 4 | 每 64 维 | 32 | 4.5 KB |
| 1 | 整 token | 16 | 1.1 KB |

- 写入侧:`quant_int8_qk` / `reshape_and_cache_int8::k_only` 按块算 amax,
  块内各槽写同值。
- 读取侧:**decode / f16pv 等"逐槽乘自己尺度"的消费者数值不变**
  (组内相等 ⇒ 逐组乘 = 整块乘),因此**不与 decode 会话/格式迁移打架**。
- 只有 P8 prefill 能吃:8 个 k-chunk 的 mma 结果先进 int32 累加,
  每块只做一次 int→float。

## 7.2 核级实测(3090 Ti,T=2048,单层全头,3 发)

| L | 基线(KG=8) | KG=4 | KG=1 |
|---:|---:|---:|---:|
| 8k | 13.59 ms | 12.17 | 11.37 |
| 65k | 119.6 | 107.6 | 103.5 |
| 140k | 259.96 | 231.5(**+12.2%**) | 225.7(**+15.2%**) |

KG=8 走同一份重构代码会**慢 13%**(293.8 ms):每块要清零 `cb[4][4]`(128 MOV)。
⇒ 若要重新启用,折叠路径必须**独立成核**或按 `kGroups<8` 条件编译,
不能与历史路径共用一份代码。

## 7.3 质量门(fresh server,单请求,`prefix_cache=true`,chunk 2048)

| 配置 | 1k | 8k | 32k |
|---|---|---|---|
| KG=8(历史) | 2/2 PASS | 3/3 PASS | PASS |
| KG=4 | 3/3 PASS | **3/3 MISSED**(拒绝式:"I cannot provide…") | PASS |
| KG=1 | **MISSED**(同上) | — | — |

门必须一尺寸一 fresh server(跨尺寸会吃前缀缓存;单卡 24G 连跑多档会
`PoolExhausted` 报 500 —— 本轮踩到两次)。

## 7.4 判据与后续

- **粗化 K 尺度 = 质量回归**,不是测量噪声(3/3 确定性复现)。
- 真正的拦路石是 §3 已立项的 **W4A8 投影 GEMM 行分块与批 T 解耦**
  (固定 M tile / 禁 split-K),让同 token 的 K 在任意 chunk 下逐位一致。
  那一刀落地后,本节的 KG 路径(**+12~15%**)才谈得上转正;
  在此之前 int8 的 K 量化必须保持 per-32-group。
- 备份:KG 实现是 3 处小改(header 常量 + 两个 writer + prefill 的 QK epilogue),
  已按 §7.1 的记录回退,可按需要复原。

## 7.5 另一条"数值等价"尝试:尺度读改 shuffle 广播(亏 3.7%)

K 尺度在每个 warp 内被 8 个 `gid` 组重复读(9 KB/warp/块)。
改成「只让 `lane<4` 读 4 对(32B)+ `__shfl_sync` 广播」后:
140k 259.4 → **269.0 ms(慢 3.7%)**,65k 120.8 → 124.3。
原因:ptxas 已把尺度读压到 36 条 `LDS.64`,换成 32 条 predicated LDS + 64 SHFL
是净增指令,而 smem 带宽本来就不是墙(见 §6.3)。已回退。

## 7.6 顺带修掉的构建坑:`cu/*.cuh` 改动原先**不触发重编**

- 现象:改 `int8_scale_groups.cuh` 后 `cargo build` 0.1 s "全增量命中",
  跑出来还是上一代语义(一度以为是测量噪声)。
- 根因:native cubin 的增量闸 `fingerprint()` 只看 `.cu` 自身 + 旗子,
  cargo 也只对 `include_str!` 命中的 `.cu` 发 `rerun-if-changed`;
  **头文件两头都不在**。
- 修法(`crates/kernels/build.rs`):`native_headers()` 扫 `cu/**` 的
  `.cuh/.h`(排除 cutlass/vllm_marlin/vendor/flashinfer/triton 子树),
  逐个发 `rerun-if-changed`,并把它们的 `len:mtime` 摘要(`fnv1a`)折进
  `fingerprint`。**保守**:任一本地头变 ⇒ 73 个 native 源全量重编(~1.5 min),
  但不会再静默用旧码。实测:`touch int8_scale_groups.cuh` ⇒
  `本次重编 73 个` ✓。

---

# §8 2026-11-04 四轮:评估"W4A8 投影 GEMM 批 T 解耦" —— **不必做(已证伪)**

> 用户批准评估路线 1 的前置(GEMM 与批 T 解耦)。**结论:这刀不该做。**
> 新探针证明 W4A8 投影 GEMM **本来就与 M 无关(逐位)** ⇒ 它不是
> per-token K scale 的闸门;KG 的闸门是纯精度。
> 唯一产物:`crates/kernels/tests/marlin_m_determinism.rs`(1.5 s 回归门)。

## 8.1 账面上的 M 依赖点(读码)

`crates/kernels/cu/marlin/marlin_host.cu`:

| 旋钮 | 代码 | 随 M 变化 |
|---|---|---|
| M 分段 | `max_par`(16 / n≤4096 时 128)× `prob_m_split` | 每段一次发射 |
| 瓦片高度 | `thread_m_blocks = min(ceil(M/16), 4)` | 1(M≤16)→4(M≥64) |
| 线程档表 | `thread_m_blocks > 1 ? large{64,256,256} : small{128,128,256}` | **thread_k 128↔64** |
| K-slice | `global_mn_tiles = ceil(M/(tmb*16))*ceil(n/thread_n)` vs `gridDim.x = sms`(84) | tiles<84 ⇒ 沿 K 切段 + `global_reduce` |

静态看,三条(档位表 / 瓦片高度 / K-slice)都会改"谁先加谁",所以在
§7 的门失败归因里看起来很顺理成章。**但这是错的**:见 8.2。

## 8.2 实测(探针 `marlin_m_determinism.rs`)

同一份权重 + 同一批激活(前 M 行相同),以 M=2048 为基准逐位比对 half 输出:

| 形状 | M 扫描 | 结果 |
|---|---|---|
| n=5120 k=5120 | 1, 8, 16, 64, 128, 256, 320, 512, 1024 | **差异 0 / 全等**,max\|Δ\|=0 |
| n=34816 k=5120 | 16, 64, 320, 1024 | **差异 0 / 全等** |
| 强制档位对照 | M=16 强制 tk=128/tn=128 | 位同(其余档位 launcher 报 err=3 不可用) |

M=16 与 M=2048 之间**跨了**线程档表(thread_k 128 vs 64)与 K-slice 分支,
输出仍然逐位相同。

## 8.3 为什么(机制)

W4A8 的整条链路是**整数**:激活 int8 × 权重 int4 → **int32 累加器**
(int16 尺度当整数乘进去,见 `marlin_template.h` kS8 分支);
per-token `a_scale` 与 per-block `b_scale` 只在**每个输出元素**上乘一次
(→ f16 输出)。整数加法**精确且可交换/结合** ⇒ 分块、切片、部分和归约
都不改位型。唯一会引入 M 依赖的写法是 FP 累加(W4A16 的 f16 路径就是
FP 累加,但它不写 KV ⇒ 与 INT8 K 量化无关)。

⇒ 文档 §4 里"固定 M tile / 禁 split-K"这条前置**作废**;
`docs/bench/int8-1k-quality-gate-2026-11-04.md` 已加 §7 更正。

## 8.4 对路线 1 的最终判决

- KG=1(+15.2%)/KG=4(+12.2%)的失败是**纯精度**:粗化 K 尺度 ⇒ 每元素
  绝对步长变大 ⇒ logits 噪声变大 ⇒ 门翻。**确定性不是瓶颈**,
  所以"先把 GEMM 解耦再说"这条路走不通,不要投入。
- 要拿到那 12~15%,只能从**精度侧**想办法(给 K 更多有效位),例如:
  1. per-32-group 不变、但**组内做 2-bit 残差修正**(格式变更,K 变 40 bit/32 维);
  2. K 保持 int8 但**per-channel(KIVI 式)**:需要转置布局 + writer 改写;
  3. 接受现状,把力气放到 §6.5 的 acc 减半(核级 ~+40% 上限,但要过 f16 累加门)。
- 顺带:现在的 per-32-group int8 门是"贴着临界点过的"(KG=4 只差一档就翻),
  任何改 K 量化的动作都应先跑 `marlin_m_determinism` + 1k/8k needle 再动。

---

# §9 2026-11-04 五轮:A/B 两条按用户要求各做定量判决 + **核是 issue-bound 的定帧**

## 9.0 一句话

P8 核 **不是 stall-bound,是 issue-bound**:`§6.1` 消融省下 1498 cyc/块、对应少
~1350 条指令/线程/块 ⇒ **3.76 IPC(上限 4)**。这一条定帧把前面所有负结果都解释了,
也把优化目标收窄成唯一一句:**减少每 key 的指令数**。

## 9.1 A(f16 PV 累加器 / 让出 64 寄存器)——**否**

改 `acc[32][4]` fp32(128 reg)为 `__half2 acch[32][2]`(64 reg),PV epilogue 走
`cvt.rn.f16x2` + `HADD2`/`HMUL2`(数值仅用于测速):

| 指标 | 基线 | f16 acc |
|---|---|---|
| 寄存器 / 栈溢写 | 255 reg / **144 B** | 255 reg / **8 B**(溢出基本消失) |
| 140k 单发 | 86.65 ms | 89.8 ms(**+3.9% 慢**) |

⇒ **寄存器墙不是墙**:把 64 个寄存器还回去、溢出清零,反而更慢(每 2 个值多一次
cvt/hadd 指令,而核按指令计费)。连带结论:同族的「减少 acc → 提高行/warp →
摊薄 smem」路线,收益前提(寄存器是瓶颈)不成立,不再投入。

## 9.2 B(per-32-group + 残差位 = 保精度 + 吃 int32 折叠)——**否**

残差必然要多跑一遍 QK mma(hi/lo 两次)。先测"mma 翻倍"的边际成本
(每次 QK mma 后再发一发同操作数 mma,廉价消费防 DCE):

| 配置 | 140k 单发 | 变化 |
|---|---|---|
| 基线 | 86.65 ms | — |
| QK mma ×2(含 ~96 条消费指令) | 102.5 ms | **+18.5%**(纯 mma 部分 ≈ +16%) |

而 int32 折叠(KG=1)只省 ~15%(§7.2)⇒ **净亏 ~1~3%,还要付格式变更
(K 变 40 bit/32 维)+ writer 重写**。B 判死。

## 9.3 顺带拿到的真收益:掩码只在**对角块**做(+3%)

`key_blocks` 是按 tile 内**最大** query 位置圈定的 ⇒ 只有在
`tile_k0 + Bc - 1 <= base_pos + q0`(块尾 key 都在最小 query 之前)的块上,
因果掩码对所有行**恒不触发**。原核每块无条件做 16 条 ISETP + 16 条 FSEL
(约 40 条指令),现在:
- 非对角块:0 条;
- 对角块:照旧;
- 越界行(`row >= tile_rows`)不写出 ⇒ 不需要掩码(其 NaN/垃圾只留在自己行内)。

实测(3090 Ti,T=2048):

| L | 基线 | 掩码优化 | 增益 |
|---:|---:|---:|---:|
| 8k | 13.59 ms | 13.27 | +2.4% |
| 65k | 119.57 | 116.85 | +2.3% |
| 140k | 259.96 | 251.73 | **+3.0%** |

- 探针 7/7 绿(含 `multi_page` 对角块用例);
- 引擎门(fresh server,`profiles/measurement/qwen38_27b-3090-int8.toml`):
  needle **8k 3/3、1k 2/2 全 RETRIEVED**;
- 对已写出的输出**逐位等价**(非对角块的掩码分支在数学上是 no-op)。

## 9.4 之后的优化目标(按 issue-bound 模型重排)

指令账(基线 1950 条/线程/块,`§6.1` SASS):标量 epilogue ~1350,其中
**寻址/整数数学 ≈ 900**(IMAD 395 / LEA 126 / LOP3 168 / SHF 139 / IADD3 148),
QK 尺度乘加 320,softmax ~450,PV epilogue 384,ldmatrix 73,mma 64。

| 顺位 | 刀 | 预期 | 状态 |
|---|---|---|---|
| 1 | **寻址/整数数学审计**(900 条里有多少是循环不变量被重复计算) | 若能砍 300~400 条 ⇒ **+15~20%** | 未做(需逐段 SASS 对读) |
| 2 | per-token K scale(int32 折叠) | +12~15% | 已实现,被精度门否(§7) |
| 3 | 掩码/边角分支 | +3% | **已落地**(§9.3) |
| — | 换 warp / 叠流水 / 让寄存器 / 换 dtype | ≈0 或负 | 全部实测否(§6/§9.1) |

> 新钉的纪律:任何改 P8 核的动作,先按"指令数 × 0.28 cyc(每块跨 SM 计)"
> 估一下收益再动手 —— 在 issue-bound 核上,这几乎就是全部真理。

---

# §10 2026-11-04 六轮:SASS 审计 —— staging 地址数学整体提出块循环(**+17~20%**)

## 10.1 方法:先切循环体,再按"指令密集段"定位

```bash
cuobjdump -sass <fatbin>   # 唯一回边 = 块循环
```
基线(§9 掩码版):整核 2733 条,**循环体 1858 条**;按 opcode 归类后
**IMAD 214 + LOP3 97 + IADD3 70 + LEA 62 + SHF 44 = 487(26%)** 是寻址/整数数学,
其中最大的一段是 **idx 95–168 连续 74 条纯整数指令**——夹在两条
`LDGSTS`(cp.async)之间。

对读源码后定谳:**三处 staging(K 块 / V 块 / scale)的地址数学是每 chunk 重算的**,
而且大部分可证冗余:

| 冗余 | 原写法 | 代价 |
|---|---|---|
| 有符号除法 | `key_l = chunk / (D/16)`、`g = i / Bc` | SASS 上是 `SHF.R.S32.HI + LEA.HI + LOP3(0xffffffe0) + IADD3` 四件套/次 |
| swizzle 重算 | `swz(key_l, d/2)`、`swz_v(d, pair0)` | 每 chunk 一套 XOR/SHF/LEA |
| 页号重载 | `block_table[nk]` 在**每个 chunk** 里再读一次 | 每 chunk 一次 `LDG.E.CONSTANT + F2I.TRUNC` |
| 溢出重载 | 基址/步长挤到栈上 | `LDL` 28 条/块,每条后面还跟着一段重算 |

而这三处的结构其实极简单:**线程 t 在块内固定负责几个 chunk,相邻 chunk 只差常数步长,
且 swizzle 用到的低位(`key_l&7` / `d&1` / `pair0`)恒定** ⇒ 整段地址数学可以
**提出块循环**(每 CTA 算一次),块内只剩 1 次页号读取 + 常数加法。

## 10.2 改法(`StageGeom`)

```cuda
// 线程 t 负责 chunk = t + Threads*j ⇒
//   K: key_l = t/16 + (Threads/16)j,  d 不变  ⇒ smem 步长 (Threads/16)·D
//   V: d     = t/2  + (Threads/2)j,   pair 不变 ⇒ 步长按 Threads/2 算
//   S: g     = t/32 + (Threads/32)j,  key 不变  ⇒ smem 是 [g][key] ⇒ 步长 (Threads/32)·Bc
//      (gmem 是 [key][g] ⇒ g 的步长就是 +Threads/32,**不是** ×Hkv·8)
```
`stage_geom()` 每 CTA 一次算出 14 个不变量;序幕与块内 K/V/scale 三处复用,
块内只留 `block_table[nk]` 一次 LDG。

> ⚠️ 踩坑记录:第一次改完输出 NaN,原因是把 **smem 的 [g][key] 步长**
> (`SStep·Bc`)套到了 **gmem 的 [key][g]** 上(应为 `+SStep`)。判据:
> scale 读错 ⇒ logits 爆到 inf ⇒ `exp2(inf-inf)=NaN`。

## 10.3 结果(3090 Ti,T=2048,单层全头,3 发)

| L | 基线 | §9 掩码版 | **§10 staging 版** | 累计 |
|---:|---:|---:|---:|---:|
| 8k | 13.59 ms | 13.27 | **11.29** | **+20.4%** |
| 32k | 59.02 | — | **49.48** | +19.3% |
| 65k | 119.57 | 116.85 | **102.19** | +17.0% |
| 100k | 185.25 | — | **158.19** | +17.1% |
| 140k | 259.96 | 251.73 | **221.90** | **+17.2%**(86.65→73.97 ms/发) |

指令账对得上模型(时间 ∝ 指令数):

| | 基线 | 现在 |
|---|---:|---:|
| 循环体 | 1858 | **1537**(−17%) |
| 寻址/整数(IMAD/LOP3/IADD3/LEA/SHF) | 487 | **~225**(−54%) |
| IMAD | 214 | 110 |
| LEA | 62 | 22 |
| SHF | 44 | 22 |
| ISETP | 66 | 33 |
| LDL(栈溢出) | 28 | **0**(`STACK` 144B→**0**) |
| 标量 FP(FMUL/FFMA/I2FP) | 816 | 816(未动) |

- 探针 7/7 绿(host 参考 cos ≥ 0.999997);
- 引擎门(fresh server,int8 测量档):needle **8k 3/3、1k 2/2 RETRIEVED**;
- 65k needle 一并跑通:**RETRIEVED 35.8 s**(参照:档面记录的基线 TTFT 38.75 s ——
  口径不同,只当指示性,未做受控 A/B)。

## 10.4 下一轮的目标账(已是纯 FP epilogue 为主)

现在循环体 1537 条:FMUL 292 + FFMA 268 + I2FP 256 = **816(53%)**,
LDSM 73 + IMMA 64 = 9%,寻址 ~225(15%),其余 softmax/杂项。

| 顺位 | 刀 | 上限 | 状态 |
|---|---|---|---|
| 1 | **QK int32 折叠**(per-token K scale) | −300 条 ⇒ **+15%** | 已实现,精度门否(§7) |
| 2 | 残余寻址 ~225 条里再抠(软/硬边界:P 存 8 条 swizzle、softmax key 索引、`MOV 56`) | −80~120 ⇒ +4~6% | 可试 |
| 3 | PV epilogue 的 I2F+FFMA(256 条) | 需换 dtype/格式 | 实测否(§9.1) |

## 10.5 顺位 2 落地:残余寻址再收一刀(累计 **+22~26%**)

按 §10 的指令账继续剥,三处已落(数值不变,探针 7/7):

1. **`block_table[tile_k0 / page]` 的运行期除法**(~15 条/块)
   SASS 上是 `I2F.RP + MUFU.RCP + IMAD.HI.U32 ×2 + F2I` 一整套 —— 因为 `page`
   是运行期参数。而本核 K 预取早就按 `block_table[nk]`(每块一页)寻址,
   即**隐含要求 `page == Bc`**。现把它写成显式不变量:
   `if (page != Bc) return;` + V 侧改用 `block_table[kb]`,不留两套寻址。
2. **scale/vscale 装载去分支**:原写法是「条件表达式包着 gmem 读」,nvcc 生成
   `BSSY/BSYNC` + 条件 `LDG`。改为**无条件读 + FSEL**(页必已分配、槽在池内,
   读越界行安全)⇒ 换成 20 条 `FSEL`,分支消失。
3. **K 预取整块快路径**:`key < L` 原来每 chunk 判一次;改为
   `if (nk*Bc + Bc <= L)` 一次判完(常态 >99%),尾块才走逐 chunk 守护版。

| L | 基线 | §10 | **§10.5** | 累计 |
|---:|---:|---:|---:|---:|
| 8k | 13.59 ms | 11.29 | **10.76** | **+26.3%** |
| 32k | 59.02 | 49.48 | **47.80** | +23.5% |
| 65k | 119.57 | 102.19 | **97.64** | +22.5% |
| 100k | 185.25 | 158.19 | **151.93** | +21.9% |
| 140k | 259.96 | 221.90 | **211.40** | **+23.0%**(86.65→70.47 ms/发) |

循环体 **1537 → 1513** 条(IMAD 110→102、ISETP 33→29、FSEL +20、分支消失)。

**试过但无效(已回退)**:把 P/V 的 ldmatrix 基址提出块循环 —— 只省 1 条
(ptxas 已折成立即数,受寄存器压力又重算)。K 的 8 个 k-chunk 地址**不能**
折成立即数:swizzle 是 `((kk_abs*2+h) ^ (b_rin)` 对 `b_rin` 非线性。

## 10.6 现在的账(还剩下的)

| 项 | 条/块 | 占比 | 可动性 |
|---|---:|---:|---|
| FMUL 292 | 128 QK 尺度乘积 + 128 acc 在线重标定 + ~36 | 19% | 重标定需"2 块共享 max"重构(改数值稳定化常数,需质量门) |
| FFMA 268 | QK 累加 128 + PV 累加 128 | 18% | 格式/精度相关 |
| I2FP 256 | QK/PV 各 128 | 17% | 只有 int32 折叠能吃(KG,精度门否) |
| IMAD+LOP3+SHF+IADD3+ISETP | ~247 | 16% | K 的 XOR 寻址 + P 存 swizzle + softmax 索引,**已接近底** |
| LDSM 73 + IMMA 64 | 137 | 9% | 结构性 |
| FMNMX 58 + FSEL 20 + FADD 20 + SHF | 98 | 6% | softmax 归约(每块 4 次 warp 归约) |

结论:**在"int8 QK + P8 PV + 无 shadow"这个格式内,预填核已到 ~1.2× 的边际**;
再往下走都必须动格式或精度(§7 KG / §9.1 dtype),那需要质量门配合。

---

# §11 剩余瓶颈与优化方向总账(2026-11-04 收官盘点)

> 口径:单 3090 Ti、27B W4A8、token-major V 版 P8 核(239.0 ms/140k 单层全头;
> e2e 预填 2104.9 / 1814.8 / 1579.7 t/s @32k/64k/100k;decode 42.66 / 40.47 / 38.61)。
> 逐核归因(140k):**attention 50.0% / Marlin 40.7%**(见
> `docs/bench/int8-kv-longctx-attribution-2026-11-04.md`)⇒ attention 的
> +10% 等于 e2e +5%(140k)、+3.5%(65k)、+2.5%(32k)。

## 11.1 账本(所有方向都落在这五本里)

| 账 | 现状 | 还剩什么 |
|---|---|---|
| **指令(issue)** | 3.76/4 IPC;循环体 1716 条 | 只有"减指令"有效(见 11.2 前三项) |
| 存储(regs/smem/L1-L2) | `STACK 0`、255 regs、44 KB smem、2 CTA/SM;**L2 已吃满**(全指第 0 页只快 5.4%) | 28 位偏移/宽访存;**没有大头** |
| 张量管 | ~50% 占用 | 不需要更快 dtype,问题是喂不饱 |
| launch/图 | prefill 无意义(µs vs 70 s);decode 已开 graph | — |
| 格式/精度 | per-32-group K 尺度;token-major V | per-tensor(D 路线)/V 布局取舍 |

## 11.2 attention 核内:四条还没打的刀(按性价比)

| 顺位 | 刀 | 预期(核级) | e2e@140k | 前置 | 风险 |
|---|---|---|---|---|---|
| 1 | **地址 32 位偏移**(`base + int32` 替 `size_t`/`IMAD.WIDE`) | +2~4% | +1~2% | 无 | 低(纯机械,不动数值) |
| 2 | **去掉 acc 在线重标定**(软 max / 两趟 / 固定 max) | +4~5% | +2~2.5% | needle 门 | 中(改数值稳定化常数,逐位不等价) |
| 3 | **KG:per-tensor K 尺度 + SmoothQuant 校准**(算法层融合,int32 折叠) | **+12~15%** | **+6~7.5%** | **权重侧校准 + needle/GSM8K 门** | 中高(精度;§7 已证纯 per-token 会打穿 1k/8k) |
| 4 | uniform 数据通路(UR)+ LDSM 合并 + 掩码零碎 | +1~2% | +0.5~1% | 先看 SASS 里 UR 占比 | 低 |
| — | ~~V 布局回 page-packed~~ | 预填 +2~4% | +1~2% | — | **decode −4%,不划算**(且 token-major 是 decode 的动机) |

## 11.3 attention 之外(按 e2e 权重)

| 项 | 现状/证据 | 判断 |
|---|---|---|
| **Marlin W4A8**(140k 占 40.7%) | M 分段上限 `max_par=16` ⇒ chunk=2048 拆 2 段、权重读 2×;**但按 chunk 摊薄后权重流量只有几十 GB/s**(13.5 GB×2 / 2048 tok × 2000 t/s ≈ 26 GB/s ≪ 936)⇒ 不是瓶颈 | 不动 |
| GQA 打包(6 q_head 共享 kv_head) | L2 已吸掉(全指一页只快 5.4%);每 warp 仍要读完整 K/V tile(16 KB/块) | 无收益(acc 寄存器不允许更多行/warp) |
| decode 轨 | token-major V 换来 +4%;PTH producer/consumer 在另一绘画 | 交回 decode 轨 |
| fp8+shadow | 100k 时 shadow 值 +8.4%,但仍**慢于 INT8 无 shadow**(−1.1%);且多占 200 MiB/rank | 30 系维持 INT8 默认 |

## 11.4 已证伪、不要再试(全部有实测)

warp 特化(+36%)、软件流水/lag 流水(噪声内)、让寄存器/消溢出(f16 acc +3.9%,`STACK` 144B→8B 仍更慢)、
mma 翻倍换精度(+16~18.5%)、scale 读改 shuffle 广播(−3.7%)、V 双布局(+134 MB @131k)、
per-page 标量尺度(质量 1/5)、Marlin 的 M 无关性担忧(已证位同,`marlin_m_determinism`)、
波量化损失(实测为 0)。**共同教训:这个核上唯一稳定有效的是"减指令"。**

## 11.5 建议的执行顺序

1. **地址 32 位化 + 零碎 instructon trim**(1 天,无门):吃到 +3~5% 核级,把 e2e 拉回与 page-packed 持平;
2. **软 max**(半天 + 门):+2% e2e;
3. **SmoothQuant 校准 + per-tensor K 尺度**(2~3 天 + 门):这是唯一还剩两位数收益的刀;
4. 之后**重新跑三向 e2e**(`docs/bench/kv-format-3way-2026-11-04.md` 的协议)定稿,
   并把这轮的"每点 fresh server / 同 prompt / 两轮"纪律写进常规流程。

---

# §12 2026-11-04 七轮(收官):两刀落地 + 四条否掉的刀(含逐核证据)

> 本章是 §11 清单的执行记录。结论:**预填核只剩 ~1% 量级的碎屑**;
> 剩下的两位数收益全在「精度/格式」侧(KG/量化),不在核内。

## 12.1 落地:softmax 起跑值改有限极小值(**+0.5%**,逐位不变)

在线 softmax 原先用 `-inf` 起跑,于是每条 p 都要套一层
`score > -inf ? ex2(...) : 0`(每块 16 条 p ⇒ 32 条 FSETP/FSEL),
alpha 还要一次 `run_m == -inf` 的特判。改用 **有限极小值** `-1e30f` 起跑:

- `nm = max(run_m, bm)` 因此**恒有限** ⇒ 掩码行 `score = -inf` 时
  `ex2(-inf - finite) = +0`(PTX 定义),不需要 select;
- 首块 / 全掩码块 `alpha = ex2(-1e30·L2) = 0` 自然成立,特判一并去掉;
- 越界行(row ≥ tile_rows)不再产生 NaN(顺带把 NaN 从 pad lane 清走)。

| 140k(=140000) | 基线(§10.5) | **§12.1** | 循环体指令 | `STACK` |
|---:|---:|---:|---:|---:|
| ms/发 | 79.70 | **79.31** | 1685 → **1650** | 0 → 0 |

探针 7/7 绿;引擎门(fresh server,`qwen38_27b-3090-int8`):
**1k RETRIEVED 2/2、8k RETRIEVED 3/3**;e2e 32k 预填 **2126.68 t/s**、
decode **42.54 t/s**(与 §11 口径同,无回退)。

## 12.2 否掉(一):`Br=128` / 8 warp(`-11%`)

动机:staging 与每块标量(4 次 warp 归约、alpha、页表读、cp.async 发射)
按行摊薄一半,预期 +4%。实测 **266.1 ms(慢 11%)**。
根因 = smem 61 KB ⇒ 只能 **1 CTA/SM**,丢掉共驻(`§6.3` 实测第二 CTA 值 1.2×);
8 warp 同 CTA 的 barrier 偏斜也更贵。**已回退**。

## 12.3 否掉(二):`acc` 重标定的三面变体(全部被 **255 寄存器** 反杀)

| 变体 | 机制 | 结果 |
|---|---|---|
| A. warp 投票跳段 | `if (__any_sync(alpha<1))` 包 128 条 FMUL | 编译后 `STACK 0→168`,**+74%** |
| B. 公共指数基准(ref) | `acc` 不谈 alpha,只把本块贡献乘 `2^(nm-ref)`(折进 `ps`/`l`) | `STACK 0→176`,**+58%** |
| C. A + `pv` 原地并进 `score`(先让 16 regs) | 同上 | `STACK 168`,**+75%** |

SASS 判据:三条路径都保持 `REG:255`,分支一旦包住「128 个 `acc` 活跃寄存器」,
ptxas 直接把这批值下沉到栈(`STL 64 / LDL 39`)。**在 255 regs 天花板下,
任何条件块包 `acc` 都不可行**——这不是写法问题,是分配器行为。
⇒ §11.2 顺位 2(软 max,预期 +4~5%)**阶段性关闭**;要做得先砍 `acc` 体量
(拆 D 轴 / 半精度累加),而两条都已实测为负(§9.1)或受输出布局限制。

## 12.4 否掉的顺带取证:32 位地址化已经不需要做

§11.2 顺位 1 的前提("SASS 里还有大量 `IMAD.WIDE`")在 §10 之后**已不成立**:
本轮统计循环体只剩 `IMAD.WIDE 5 + IMAD.WIDE.U32 1 + LEA.HI.X 9`,
`IMAD.MOV.U32 70`(`IMAD.SHL.U32 43`)是指针/索引的搬移与左移,不是 64 位地址。
⇒ 该项**作废**(ptxas 已把 staging 地址压到 32 位偏移)。

## 12.5 decode 侧:逐核证据(问题不在 attention)

用 **CUPTI 进程内归因**(`cargo build -p server --features perf,cupti` +
`probes.perf_level="gpu"` 起机;运行期 `/debug/perf/level/gpu` 太晚,`gpu_armed`
在构造期定值)在 32k 单流上取逐核 GPU 时间:

| 面 | 实测 | 判断 |
|---|---|---|
| decode attention(`owl_int8_decode_pth_pc_split`) | **66 µs/层/步** ×16 层 ≈ **1.05 ms** = token 的 **4.4%** | PTH PC 臂已是生产路;同条件下旧 `gqa_tc_split` 65k 是 **1.45 ms**,PTH PC 为 **0.48 ms**(3×)⇒ 已吃到,无剩余 |
| lm_head(`owl_gemv_i8w_f16`,int8 行量化) | **1.33 ms/步**(768 MB @ ~580 GB/s)= **5.6%** | 带宽已近顶;换格式不划算 |
| Marlin W4A8 m=1 单核带宽 | `marlin_bw_bench` 实测 **820~870 GB/s(80~86% peak)**;W4A8 比 W4A16 还快 3~8% | **decode 是权重带宽 bound**,不是核效率问题 |
| prefill(K=32k) | attention 核 **2.655 s / 15.3 s = 17%**;逐块 9.76 ms | 与 §11 归因一致(上下文越长占比越高) |

⇒ **decode 的剩余空间不在 attention**(4%),而在「权重读取的 80~86% → 100%」
与「非 GEMM 的每层碎核」(GDN 6 核 × 16 层、norm/quant/cast 等,单步数十次发射)。
前者是 Marlin 内部(L2/DRAM 调度),后者需要融合/图内减少节点 —— 都不是本文档
的核级刀。⚠️ CUPTI 侧纪律:metrics store 每 tag 4096 样本上限,`drop` 字段
非零 ⇒ **只有均值可信,总量不可加**(本轮 `gpu.kern` 合计 5.09 s ≠ 墙钟 15.3 s
就是 4096 封顶所致)。

## 12.6 收官账

| 项 | 状态 |
|---|---|
| 预填核(相对 2026-11-04 早基线) | **+23% → +23.7%**(86.65 → 79.31 ms/发 @140k) |
| 剩余可动 | 只有 §11.2 顺位 3(**KG + SmoothQuant**,+12~15% 核级)与碎屑(~1%) |
| 已关闭 | warp 特化 / 流水 / 让寄存器 / 换 dtype / 软 max(条件块) / Br=128 / 32 位地址化 |

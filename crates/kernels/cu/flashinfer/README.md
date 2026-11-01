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

**闭包**(25 头 / ~0.85 MB;**无 cutlass**、无额外编译宏;`nvcc -M` 扫一个含
`BatchPrefillWithPagedKVCacheDispatched` 的 TU 即复现,清单见 `closure.txt`)。

**ABI 逐字未变**(`owl_fi_prefill_plan` / `owl_fi_prefill_run` /
`owl_fi_prefill_run_fp8kv`;15×int64 plan blob、槽序、`FI_SPLIT_KV`/`FI_DEBUG`
旋钮全保留)⇒ Rust 侧(`family/server/探针`)**零改动**。

**本目录文件**
| 文件 | 说明 |
|---|---|
| `include/` | 上游闭包 + owl FP16-PV 补丁(默认宏 0;升级 = 重扫闭包 + 重放 `patches/fp16-pv-0.7.2.patch`) |
| `patches/fp16-pv-0.7.2.patch` | owl 本地 FP16-PV 改动(相对上游 0.7.2.post1) |
| `owl_fi_attention.cu` | owl adapter(换源记录见文件头注:plan 前置 `PrefillPlanWorkspaceSize` 尺寸校验 / `kv_dtype_bytes` / params 改逐字段赋值 / 模板加 `SAME_KV_STRIDES` / fp8 免宏) |
| `bench/` | 候选核单测台架(不参与构建):`FI_SRC=$(cd .. && pwd) ./run.sh`(f16)/ `FP8=1 ...`(fp8 池) |

**全案**:`docs/devlog/kernels/prefill-attention-swap.md`(§1 三候选数字表 /
§3 集成配方 / §6 落地与验收证据)。

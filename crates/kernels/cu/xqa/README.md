# owl XQA(kNHD fp8 KV 的 decode/verify 注意力)

**来源**:`repos/vllm-ampere-optimized/flashampere/xqa/csrc/`(avesed/vllm-ampere-optimized
@ f2e081d)= FlashInfer `csrc/xqa` 的 **Ampere 闭包**(裁掉 sm90 GMMA/TMA 与 sm120 MLA;
16 文件 340K,**无 cutlass 依赖**)。上游基线在 `repos/flashinfer/csrc/xqa/`(24 文件)。

**为什么**:XQA 是 decode 形状注意力(KV-split + warp-specialized + GQA 内共享 K/V 读),
**q 维扩展近乎免费**(`SPEC_DEC` 模式 q=1+K)→ 正是 owl verify 缺的形态。
参考 fork 实测 q=1+K 比 FA2 `fwd_kvcache` 快 1.8-4.3×(32-128k ctx,cos=1.0)。

**本目录改动**(相对上游):
- 删 `flashinfer_xqa_binding.cu` / `tvm_ffi_utils.h`(TVM FFI 运行时依赖,owl 不用);
- (2026-10-13 C3a 起)**不再有 C host 发射器**:host 侧搬 Rust
  (`crates/kernels/src/{family,server}/xqa.rs`);本目录只留**设备码源**
  (`mha.cu` + 头)。构建期用 `-DGENERATE_CUBIN`(上游自带的"只出设备码"模式)
  编两个宏变体 fatbin(`xqa_verify` / `xqa_decode`)。
- 已删:`owl_xqa_launch.cu`(裸指针 C ABI 薄 binding —— 职责搬 Rust)、
  `xqa_wrapper.cu`(上游 TVM-FFI binding,owl 从未编译它,且引用不存在的
  `../tvm_ffi_utils.h`)、`prebuilt/*.a`(预编二进制)。

**实例化**(build.rs 传,见 `crates/kernels/build.rs::build_xqa_assets`):TOKENS_PER_PAGE=32 /
HEAD_ELEMS=256 / HEAD_GRP_SIZE=6 / CACHE_ELEM_ENUM=2(fp8 e4m3)/ INPUT_FP16=1 /
SPEC_DEC=1 / 其余 0。**编译只需补 `--expt-relaxed-constexpr`**(实测 12s 出 sm86 目标)。

**布局**:KV = `[num_pages, tokens_per_page, Hkv, D]`(XQA 文档名 NHD)= owl kNHD 主池
逐字对齐;q/out = `[batch,1,q_seq_len,Hq,D]`;page_table = **int32**(owl 现有 bt 是 f32,
需 i32 孪生);seq_lens = **uint32**。

## owl 本地改动(相对上游 / flashampere)

1. **删** `flashinfer_xqa_binding.cu` / `tvm_ffi_utils.h`(TVM FFI 依赖);
   2026-10-13 起连 `owl_xqa_launch.cu` / `xqa_wrapper.cu` 一并删(host 侧搬 Rust)。
2. **`mha.cu` 本地修复(上游 bug)**:`hostSmemSize` 原为**文件作用域 static**,在
   **库装载期**(还没有 CUDA 上下文)就 `cudaMemcpyFromSymbol` → 失败后 `size` 是
   **未初始化栈垃圾**,又被当 `dynamicSmemBytes` 发射 ⇒ `cudaLaunchKernelEx` 抛
   `cudaErrorInvalidValue`。现象极具迷惑性:**同一二进制 device 0 通过 / device 1 崩**
   (栈垃圾逐进程不同)。修法 = 改**首次发射时惰性求值 + `checkCuda` 逐步查错**。
   ⚠️ 2026-10-13 起该 host 段被 `#ifndef GENERATE_CUBIN` 圈住(cubin 形态只出
   设备码;host 职责在 Rust)。
3. **`mha.cu` 本地新增(2026-10-13 cubin 形态)**:`owl_xqa_launch_consts`
   —— 1 线程探针核,把 host 需要的编译期常量(smem/ctaTile/headGrpSize/
   rowsPerBlock/tokensPerPage/headElems/allowMultiBlock/cta 形状)写进缓冲。
   **资产自描述**:单 TU cubin 会把设备变量内部化(STB_LOCAL,`cuModuleGetGlobal`
   取不到),故改由核自己报 ⇒ Rust 侧零常量复制。
4. 其余文件逐字未改(升级 = 与上游 `mha.cu` diff)。

## 调用契约(实测钉死,踩坑留档)

- **`seq_lens` 是"整请求总 KV 长"(batch 维)**,不是逐行前缀长!块内 q_seq_len 行的因果
  结构**只由 mask 位掩码表达**。误传逐行长 ⇒ 行 1..K 各少 1..K 个 key ⇒ 假误差 ~1e-2
  (实测 0.0115),**不是精度问题**。
- **spec-dec 位掩码语义(定谳)**:`[batch, q_seq_len, ceil(q/32)] u32`(u16 打包,行跨
  `ceil(q/32)*2` u16);**行 i = 块内第 i 个 token**(token-major:`idxInGrp/headGrpSize`),
  **bit k = 允许看块内第 k 个 key**,因果 = `bits 0..i`(**含自身位置**)⇒
  `mask[i] = (1<<(i+1))-1`。四路 A/B 佐证:位序反 0.0256 / 极性反 0.0256 / 行序反 0.0141,
  唯因果口径贴 host(**4.5e-4**)。
- **host 参考口径**:因果是闭区间 —— 行 i 的 key 数 = `prefix + i + 1`(首版按"不含自身"
  算 → 每行少 1 个 key,又一层假误差)。
- `q_scale = attn_scale * sqrt(D)`;`q/out` 视图 `[batch,1,q_seq_len,Hq,D]`;KV 元素序
  strides(`[pages, tokens, Hkv, D]` = owl kNHD);page_table **int32**;semaphores
  `u32[round_up(Hkv*batch,2)+2+Hkv*batch+2]` 发射前清零;scratch 持久工作区。

## 实测(2026-10-13,3090 Ti,kernels crate 单算子测试 `tests/xqa_verify_probe.rs`)

| ctx | XQA | v2 主核+reduce | 加速比 | max\|XQA−v2\| | max\|XQA−host\| | max\|v2−host\| |
|---:|---:|---:|---:|---:|---:|---:|
| 2048(掩码全开) | 0.164 ms | 0.775 ms | 4.72× | 4.9e-4 | 3.1e-4 | 2.5e-4 |
| 2048(因果) | 0.164 ms | 0.776 ms | 4.72× | 4.9e-4 | **4.5e-4** | 4.6e-4 |
| **8192(因果)** | **0.495 ms** | **2.673 ms** | **5.40×** | 4.9e-4 | (host 跳过) | — |

⇒ 按 16 个全注意力层/轮折算:verify 注意力 **42.8ms → 7.9ms(−34.9ms/轮)**。
数值与现役 v2 臂同级(4.5e-4 vs 4.6e-4),门 1e-2 有 22× 余量。

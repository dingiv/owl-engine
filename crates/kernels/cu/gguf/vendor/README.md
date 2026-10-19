# vendored: llama.cpp 的 **MMQ**(tensor-core prefill GEMM)

> **本目录是第三方代码**(MIT,见 `LICENSE-llama.cpp`)。上游 = llama.cpp(ggml-cuda)。
> 除下列**两处登记改动**外逐字未改:
>
> | # | 改动 | 原因 | 判据 |
> |---|---|---|---|
> | P1 | `mmq.cu`:`mul_mat_q` 的落点 `float* dst` → `__half* dst` + `__float2half(...)`(13 处声明 + 1 处写) | 本仓输出是 **f16 [m,n] 行主序**,与上游 `dst[col*nrows_dst+row]` **布局逐位相同** ⇒ 直接落 f16,省一个 f32 中间缓冲 + 一次转换核(头层 m×248320×4 = 1 GB 级) | 数值门 `gpu_gguf_mmq_matches_mmvq` |
> | P2 | `common.cuh`:`OWL_MMQ_TENSOR_CORES` 开关(**默认 dp4a 档** = 定义 `GGML_CUDA_FORCE_MMQ`) | 上游张量核档配置 `{x=4,y=32}` 每 CTA 只吃 4 个 token ⇒ 权重重读 `m/4` 次 ⇒ 实测**比 dp4a 档慢 2-3×**(见下) | 两档同跑实测 + `family::gguf::MMQ_TENSOR_CORES` 互锁 |

## 来源(锁定 revision)

| 项 | 值 |
|---|---|
| 仓库 | https://github.com/ggml-org/llama.cpp |
| commit | `1b1e27cb49158123ef4902aa41eb368c9e76e6a1`(2024-05-23,**QK_K=64 下线前的最后一个版本**) |
| 为什么是这一版 | 本仓 `../gguf.cuh` 的快照来自同一代(它还带 `#ifdef GGML_QKK_64`,上游 2024-05-23 起删除)⇒ **块布局/块几何逐位同代**,不会踩"新 MMQ 期待新结构体"的坑 |
| 为什么不用 head 版 | head 已拆成 `mmq.cuh + mmq-load-tiles + mmq-vec-dot + mmq-config-*`(7.8k 行,含 fp4/Q1_0 等本仓没有的型);同代版本是**单文件 `mmq.cu` 1570 行**,且**直吃普通 `block_q8_1`**(36B,正是本仓 `owl_gguf_quant_act_*` 已产出的布局)⇒ 不需要另造激活布局 |

## 文件

| 文件 | 行数 | 说明 |
|---|---:|---|
| `mmq.cu` | 1570 | MMQ 实现(内核 `mul_mat_q<...>` + 逐型启动宏 + 唯一入口 `ggml_cuda_op_mul_mat_q`) |
| `mmq.cuh` | 9 | 上游接口声明(引用了 `ggml_tensor`/`ggml_backend_cuda_context` —— 见 shim) |
| `vecdotq.cuh` | 1280 | 逐型 `vec_dot_*`(与本仓 `gguf.cuh` 同代同物;两者**不可同 TU 共存**,会重定义) |
| `ggml-common.h` | 1853 | 块结构 + IQ 码本表(`GGML_COMMON_DECL_CUDA` + `GGML_COMMON_IMPL_CUDA`) |
| `dequantize.cuh` | 103 | 上游同代反量化小件(MMQ 的 dp4a 分支需要) |
| `common.cuh` | — | **owl 侧适配壳**(非上游):见下 |

## `common.cuh` 壳提供了什么(全部 = 上游 `common.cuh` 的最小子集)

`WARP_SIZE` / `GGML_PAD` / `GGML_UNUSED*` / `GGML_ASSERT` / `GGML_CUDA_ASSUME` /
`NO_DEVICE_CODE` / `dfloat`·`dfloat2` / `kvalues_iq4nl`(上游同代 `common.cuh:476`)/
`ggml_type` 枚举(值 = `ggml.h@rev`,与本仓 `GgmlTy::code()` **同一套数字**)/
`CC_*` 常量 / `ggml_cuda_device_info`·`ggml_cuda_info()`·`ggml_cuda_get_device()`.

**两处有意的语义替换**(owl 纪律):

1. `ggml_cuda_info()` 上游是**运行期设备查询**;本仓 = **编译期常量**
   `OWL_CUDA_CC`(本仓只编 sm_86 代际)⇒ H2(核与驱动零运行期设备查询);
2. `ggml_tensor`/`ggml_backend_cuda_context` 只给**最小外形**(`ne[0]`/`type`/
   `device`)—— 上游那个 `ggml_cuda_op_mul_mat_q(ctx, src0, …)` 入口我们**永不调用**
   (适配层直接走 `mul_mat_q` 内核),它存在只为让声明通过编译。

## 已验证(2026-10-19)

```bash
# 在 crates/kernels/cu/gguf/ 下:
cat > /tmp/mmq_tu.cu <<'EOF'
#include "vendor/mmq.cu"
EOF
nvcc -std=c++17 -arch=sm_86 --expt-relaxed-constexpr -c /tmp/mmq_tu.cu -o /tmp/mmq_tu.o -I.
# ⇒ 编译通过;cuobjdump 可见 mul_mat_q<{q4_K,q5_K,q6_K,q8_0}, fallback=true|false>
```

## 接线清单(下一步,设计 §十八)

1. `gguf_mmq.cu`(本仓 `.cu`):每型一个 `extern "C"`
   `owl_gguf_mmq_<ty>(w, xq, out, m, n, k, out_off, out_ld, stream)` ——
   合成三个最小 `ggml_tensor`(只填 `ne[0]`/`type`)后调
   `ggml_cuda_op_mul_mat_q(ctx{device=0}, …)`。**为什么需要显式 stream**:
   上游用 `<<<…, stream>>>` 启动,而本仓其它核由 server 置流 ⇒ 该入口必须由
   server 侧按名分派(与 `cublas`/`flashinfer` 同一条 foreign 通道)。
2. `ops/gguf.rs`:四型 `KernelSpec`(槽序 = 上面 9 参;`grid` 哨兵 = 空,几何由
   核内/主机侧自算);`driver.rs` 同名拾取;`ops/mod.rs` 白名单登记。
3. `linear.rs` 分派:`m ≥ MMQ_MIN_M` 且 `k % QK8_1 == 0` 且**无内维列序表**
   (MMQ 不做 tiled 置换 ⇒ `ssm_out` 单卡档回退 smem 臂)⇒ 走 MMQ;
   激活仍由 `GgufQuantAct` 产出(布局已对)。
4. 门:vs MMVQ `max|Δ|/max < 2e-2`(同 smem 门口径)+ 1001-token 首 token 一致;
   实测目标:Ampere 上 MMQ ≈ 30-50 TFLOPS ⇒ 27B prefill **400-600 t/s**
   (现 smem 臂 121 t/s)。

## 实测两档(2026-10-19,3090 Ti,1001-token prompt,chunk 512)

| 档 | tile | 每 CTA token | 4B prefill | 27B prefill |
|---|---|---:|---:|---:|
| 张量核(`CUDA_USE_TENSOR_CORES`) | Q4_K/Q5_K/Q6_K `{4,32,4}`、Q8_0 `{4,32,4}` | 4 | 739 t/s | 117 t/s |
| **dp4a(本仓默认)** | Q4_K/Q5_K `{64,128,4}`、Q6_K `{64,64,4}`、Q8_0 `{128,64,4}` | 64-128 | **1829 t/s** | **335 t/s** |

**为什么张量核档反而慢**:该档每个 CTA 只覆盖 4 个 token,权重被重读 `m/4` 次
(权重量 → 带宽墙);dp4a 档把 token 窗拉到 64-128 ⇒ 权重复用 16-32×,点积虽走 int8
dp4a 但**算术密度足够**(每权重字节换 4-8 个 MAC)。⇒ **Ampere 消费卡上,MMQ 的快
来自"大 token 窗",不来自张量核**。若日后要再上一档:需要"张量核 + 大 token 窗"
的配置(上游 head 版已拆出 `mmq-config-*`,可另行 vendoring 试)。

# vendored: llama.cpp 的 **MMQ**(tensor-core prefill GEMM)

> **本目录是第三方代码**(MIT,见 `LICENSE-llama.cpp`)。除 `common.cuh` 与
> `README.md` 外,**逐字未改**。上游 = llama.cpp(ggml-cuda)。

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

// ============================================================================
// **owl 侧适配壳**(非上游文件):`vendor/common.cuh`
//
// 上游 llama.cpp 的 `ggml-cuda/common.cuh`(1719 行)含大量我们不需要的面
// (ggml 后端/张量/池/HIP/ROCm/多设备探测)。本壳只提供 **vendored MMQ 需要的
// 最小子集**,并且把**设备事实**从"运行期查询"改成 **boot 期常量**(owl H2 纪律:
// 核与驱动不得做运行期设备查询)。
//
// 上游对照(同一 commit;逐条对位,勿增删语义):
//   WARP_SIZE / dfloat·dfloat2 / GGML_PAD / GGML_UNUSED / GGML_ASSERT
//   ggml_type 枚举(值与 ggml.h@rev 一致)
//   ggml_cuda_device_info::devices[i].cc(仅此字段被 MMQ 的**主机侧**分派读)
//
// ⚠️ 本文件**不**包含我们的 `../gguf.cuh`:vendored `vecdotq.cuh` 自带同名
// `vec_dot_*`(同代同物),两份同时进一个 TU 会重定义。块结构统一来自
// `vendor/ggml-common.h@同代`(布局与本仓 `gguf.cuh` 的静态断言逐位一致)。
// ============================================================================
#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define WARP_SIZE 32

// ---- 上游 ggml 宏子集 ----
#define GGML_UNUSED(x) (void)(x)
#define GGML_UNUSED_VARS(...) (void)0
#ifndef GGML_PAD
#define GGML_PAD(x, n) (((x) + (n) - 1) & ~((n) - 1))
#endif
// 上游默认开张量核(`GGML_CUDA_FORCE_MMQ` 才关)—— 本仓同样默认开:
// 开 ⇒ ampere 档配置 = {x=4, y=32, nwarps=4}(mma 小 tile);
// 关 ⇒ {x=64/128, y=64/128, nwarps=4..8}(dp4a 大 tile)。
// ⚠️ 本仓 2026-10-19 实测(4B/27B,3090 Ti):
//   张量核档 ampere 配置 = `{x=4, y=32}`(每 CTA 只吃 4 个 token)⇒ 权重被重读
//   `m/4` 次 ⇒ prefill 反而**受权重带宽**限制(27B ≈ 117 t/s);
//   dp4a 档 = `{x=64/128, y=64/128}`(64-128 token/权重读)⇒ 权重复用高 16-32×,
//   代价是点积走 int8 dp4a。本仓默认走 **dp4a 档**(`OWL_MMQ_TENSOR_CORES=1`
//   可切回张量核档;切换必须同时改 `family::gguf::MMQ_CFG`,两边由数值门互锁)。
#ifndef OWL_MMQ_TENSOR_CORES
#define GGML_CUDA_FORCE_MMQ
#endif
#ifndef GGML_CUDA_FORCE_MMQ
#define CUDA_USE_TENSOR_CORES
#endif

// 上游:非 DP4A 代际的"无设备代码"哨兵(只用于 CPU/老架构分支的编译期占位)
#define NO_DEVICE_CODE
#define NO_DEVICE_CODE_DEFINED

// 上游:`__builtin_assume` 的 CUDA 包装(仅优化提示,无语义)
#if defined(__CUDA_ARCH__)
#define GGML_CUDA_ASSUME(x) __builtin_assume(x)
#else
#define GGML_CUDA_ASSUME(x) (void)0
#endif

#ifdef __CUDA_ARCH__
#define GGML_ASSERT(x) do { if (!(x)) { __trap(); } } while (0)
#else
#include <cstdio>
#include <cstdlib>
#define GGML_ASSERT(x) do { if (!(x)) { std::fprintf(stderr, "GGML_ASSERT(%s) failed @ %s:%d\n", #x, __FILE__, __LINE__); std::abort(); } } while (0)
#endif

// ---- 反量化浮点档(上游 `GGML_CUDA_F16` 开 = half;owl 统一 f32 档)----
typedef float  dfloat;
typedef float2 dfloat2;

// ---- 块结构(与 owl `gguf.cuh` 同代:块几何逐位一致,见 host 侧静态断言)----
// 上游 `common.cuh` 同时开 DECL + IMPL:后者提供 IQ 族的**码本表**
// (`iq2xxs_grid` / `ksigns64` / … —— `vecdotq.cuh` 的 IQ 分支引用它们)。
// 本仓只跑 Q4_K/Q5_K/Q6_K/Q8_0,但表是 `static const` 数据、零运行期成本。
#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"

/// ggml 的张量型词表(值 = `ggml.h@rev`;与 `owl_kernels::family::gguf::GgmlTy`
/// 的 `code()` **同一套数字** —— MMQ 的 `type_x` 槽直接吃它)
enum ggml_type {
    GGML_TYPE_F32 = 0,
    GGML_TYPE_F16 = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_Q8_K = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S = 19,
    GGML_TYPE_IQ4_NL = 20,
    GGML_TYPE_IQ3_S = 21,
    GGML_TYPE_IQ2_S = 22,
    GGML_TYPE_IQ4_XS = 23,
    GGML_TYPE_I8 = 24,
    GGML_TYPE_I16 = 25,
    GGML_TYPE_I32 = 26,
    GGML_TYPE_I64 = 27,
    GGML_TYPE_F64 = 28,
    GGML_TYPE_IQ1_M = 29,
    GGML_TYPE_BF16 = 30,
    GGML_TYPE_COUNT,
};

// ---- 计算能力常量(上游 ggml-cuda 同值;owl 只编 sm_86 代际)----
#define GGML_CUDA_CC_PASCAL 600
#define GGML_CUDA_CC_DP4A   610
#define GGML_CUDA_CC_VOLTA  700
#define GGML_CUDA_CC_TURING 750
#define GGML_CUDA_CC_AMPERE 800
#define GGML_CUDA_CC_OFFSET_AMD 1000000
// 上游同名常量的**短名**(`mmq.cu@rev` 的主机侧分派用短名)
#define CC_OFFSET_AMD 1000000
#define CC_RDNA2      (CC_OFFSET_AMD + 1030)
#define CC_VOLTA      700
#define CC_AMPERE     800
#define MIN_CC_DP4A   610

/// **owl 的编译期设备事实**(与构建 arch 一一对应;boot 期 `Hw::arch` 同源)
#ifndef OWL_CUDA_CC
#define OWL_CUDA_CC GGML_CUDA_CC_AMPERE  // sm_86(本仓唯一构建代际)
#endif

// ---- IQ4_NL 的码值表(上游 `common.cuh@rev:476`;`vecdotq.cuh` 引用)----
static const __device__ int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

// ---- 上游接口头里出现、但 owl 侧**不实现也不调用**的类型的**最小外形** ----
// `mmq.cuh@rev` 声明了 `ggml_cuda_op_mul_mat_q(ctx, src0, src1, …)`;owl 的适配层
// 直接调 `mul_mat_q_case<ty>`(自己写 launch),那个入口**永不执行** —— 但为了让它
// 通过编译,需要这些字段存在(`ne[0]`/`type`/`ctx.device` 就是它读的全部)。
struct ggml_tensor {
    int64_t ne[4];
    enum ggml_type type;
};
struct ggml_backend_cuda_context {
    int device;
};

/// 设备条目子集:MMQ 的主机侧分派只读 `cc`
struct ggml_cuda_device_info {
    struct device {
        int cc;
    } devices[1];
};

/// 上游是运行期查询;owl = **编译期常量**(H2:零运行期设备查询)
static inline const ggml_cuda_device_info & ggml_cuda_info() {
    static const ggml_cuda_device_info info = {{{ OWL_CUDA_CC }}};
    return info;
}

static inline int ggml_cuda_get_device() {
    return 0; // 单设备进程内 ordinal 由引擎持有;MMQ 只拿它去查 cc
}

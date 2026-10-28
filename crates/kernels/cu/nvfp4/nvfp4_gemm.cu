// ============================================================================
// owl_nvfp4 —— NVFP4(e2m1 + e4m3/16 块缩放)GEMM 库(sm_120a)。
//
// 溯源:探针 `probes/nvfp4/nvfp4_gemm.cu`(2026-10-27 跑通,4B 真权重 3.17~3.69× MMQ2)。
// 本文件 = 同一份内核的**库形态**:去掉 `main()`,只留 `extern "C"` 面,
// 让 Rust 侧(`owl_kernels::nvfp4`)能预分配、host 打包、按流发射。
//
// ## 为什么必须走 nvcc 编静态库(而不是 NVRTC/cubin 通道)
// FP4 是 **arch-specific** 指令(`mxf4nvf4.block_scale`):目标必须是
// `-gencode arch=compute_120a,code=sm_120a`(**两个都带 `a`** —— 只写
// `-arch=sm_120a` 时 nvcc 会把 PTX 目标退成 compute_120,ptxas 直接拒绝)。
// 且整套 CUTLASS 模板要 C++17 + 头文件闭包 ⇒ 只能 nvcc 出 `.a`。
//
// ## 布局契约(探针实测,不靠推理)
// - A/B 的 fp4 数据:A 行主序 `[M][K/2]`,B 行主序 `[N][K/2]`(每字节两个 k,低 nibble 在前)
// - SF 走 CUTLASS `Sm1xxBlkScaledConfig` 交错布局:
//   `make_tensor(ptr, tile_atom_to_shape_SFA(shape(M,N,K,1)))` 后按 `t(row,16*g,0)` 写
//   ⇒ **不用自己反推交错规则**(数值门②:解析 960 = 实测 960,0/16384 超差)
// - SFA 只依赖 (M,K)、SFB 只依赖 (N,K) ⇒ 权重只打包一次,跨 M 复用
//   (依据:`sm100_blockscaled_layout.hpp::tile_atom_to_shape_SF*` 只取本操作数的范围)
//
// ## 数值门(探针已过,接入后由引擎侧再跑一遍)
// - 数据通路 max|Δ| = 1 个 f16 ULP(0/131072 超差)
// - 真权重(4B `gate_proj`)相对误差 2.35e-04
//
// ## 分流纪律
// M≈128 才开始赢(M=64 → 0.83×、128 → 1.76×、512 → 3.19×、2048 → 3.69×)
// ⇒ 选核必须 `m >= 128 && caps.fp4_blockscale`,小 m 回落 GGUF MMQ2 / Marlin。
// ============================================================================
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <thread>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>

#include "cutlass/gemm/collective/collective_mma_decl.hpp"
#include "cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp"
#include "cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
// epilogue 融合算子(alpha 设备指针面;见 `Epilogue` 的 `fusion::LinearCombination`)
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cute/tensor.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

using namespace cute;
using ElemA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElemB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using TileShape = Shape<_128, _128, _256>;

using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, TileShape, Shape<_1, _1, _1>,
    cutlass::epilogue::collective::EpilogueTileAuto, float, float,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto,
    // **alpha 可来自设备指针**(2026-10-28 进图刀):`fusion::LinearCombination`
    // 的 Args 带 `alpha_ptr`(`Sm90ScalarBroadcast`)⇒ 激活的 per-tensor 全局
    // 缩放可以留在设备侧(`k_scale_from_amax` 直接写 `alpha = g·gb`),
    // 不再需要 host 回读 ⇒ 本族可进 CUDA graph。
    // 默认 epilogue(`DefaultEpilogue`)的 alpha 是**内核参数** ⇒ 一旦要现算
    // 就必须 D2H,那是“结构化不可捕获”的根源。
    cutlass::epilogue::fusion::LinearCombination<
        cutlass::half_t, float, cutlass::half_t, float,
        cutlass::FloatRoundStyle::round_to_nearest>
    >::CollectiveOp;

using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
    ElemA, cutlass::layout::RowMajor, 32,
    ElemB, cutlass::layout::ColumnMajor, 32,
    float, TileShape, Shape<_1, _1, _1>,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;

using Kernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, Mainloop, Epilogue, cutlass::gemm::PersistentScheduler>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
using Sm1xxBlkScaledConfig = typename Mainloop::Sm1xxBlkScaledConfig;
using StrideA = typename Mainloop::StrideA;
using StrideB = typename Mainloop::StrideB;
using ElementA = typename Mainloop::ElementA;
using ElementB = typename Mainloop::ElementB;
using ElementSF = typename Mainloop::ElementSF;

// 动态 shape 的 SF 布局类型(M/N 是运行期值,类型与值无关)
using SFLayoutA = decltype(Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
    make_shape(int(0), int(0), int(0), int(0))));
using SFLayoutB = decltype(Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
    make_shape(int(0), int(0), int(0), int(0))));

// ---------------------------------------------------------------------------
// e2m1 / e4m3 编码(host;与探针逐字相同)
// ---------------------------------------------------------------------------
namespace owl_nvfp4_detail {
// **编码一律用 CUDA 硬件转换**(host/device 同一套语义,RNE):
// 早先用"表内最近值"的表驱动编码,在**恰好等距**的 tie 上与硬件结果不同
// (实测 28/32768 字节),导致 host 打包的权重与设备打包的激活在同一次 GEMM
// 里走两套舍入 —— 不是错,但没法逐位对拍,且 diff 会被误当成数值 bug。
static inline float e4m3_dec(uint8_t c) {
  const __half_raw hr = __nv_cvt_fp8_to_halfraw(c, __NV_E4M3);
  return __half2float(__half(hr));
}

static inline uint8_t e4m3_enc(float x) {
  if (!(x > 0.f)) x = 0.001953125f;
  return (uint8_t)__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
}

/// f16 `[rows,K]` → fp4(`[rows][K/2]`,低 nibble 先)+ SF(CUTLASS 交错布局);
/// 返回 per-tensor 全局缩放 `g`(e4m3 的 scale 内部已除过 `6*448*g`)。
///
/// **行区间版**(2026-10-28 装载提速):`g` 由调用方给 ⇒ 行与行**完全独立**
/// (每个 (row,kg) 只写自己那个 SF 字节 + 自己那段 fp4)⇒ 可按行切给多线程。
template <class TensorSF>
static void quantize_rows(const __half* a, int rows, int K, uint8_t* q, TensorSF& sf,
                          int r0, int r1, float g) {
  const float inv = 1.f / (6.f * g);
  for (int r = r0; r < r1; ++r) {
    for (int kg = 0; kg < K / 16; ++kg) {
      float mx = 0.f;
      const size_t base = (size_t)r * K + (size_t)kg * 16;
      for (int i = 0; i < 16; ++i) mx = fmaxf(mx, fabsf(__half2float(a[base + i])));
      uint8_t sc = e4m3_enc(mx > 0 ? mx * inv : 1.f);
      sf(r, 16 * kg, 0) = sc;  // CUTLASS 交错布局:按 (row, k, l) 写
      float sd = e4m3_dec(sc) * g;
      if (sd <= 0.f) sd = 1.f;
      for (int i = 0; i < 16; i += 2) {
        const float2 pair = make_float2(__half2float(a[base + i]) / sd,
                                        __half2float(a[base + i + 1]) / sd);
        q[(size_t)r * (K / 2) + ((size_t)kg * 16 + i) / 2] =
            __nv_cvt_float2_to_fp4x2(pair, __NV_E2M1, cudaRoundNearest);
      }
    }
  }
}

/// 全局缩放 = 全张量 amax / (6·448)（两遍法的第一遍；只读 ⇒ 也可并发）
static float quantize_g(const __half* a, int rows, int K) {
  float amax = 0.f;
  for (size_t i = 0; i < (size_t)rows * K; ++i) amax = fmaxf(amax, fabsf(__half2float(a[i])));
  return amax > 0 ? amax / (6.f * 448.f) : 1.f;
}

template <class TensorSF>
static float quantize(const __half* a, int rows, int K, uint8_t* q, TensorSF sf) {
  const float g = quantize_g(a, rows, K);
  memset(q, 0, (size_t)rows * (K / 2));
  quantize_rows(a, rows, K, q, sf, 0, rows, g);
  return g;
}
}  // namespace owl_nvfp4_detail

// ---------------------------------------------------------------------------
// 执行计划(M/N/K 绑定;SF 布局与 workspace 尺寸都在这里定死)
// ---------------------------------------------------------------------------
struct OwlNvfp4Plan {
  int M = 0, N = 0, K = 0;
  SFLayoutA l_sfa{};
  SFLayoutB l_sfb{};
  size_t sfa_bytes = 0, sfb_bytes = 0;
  size_t ws_bytes = 0;
  /// SFA 交错布局的**仿射步长**(m%32 / (m/32)%4 / m/128 / (g%4) / g/4);
  /// 设备侧激活量化器用它写 SF —— 防错位靠 `plan_new` 里对 cute 布局的逐点校验。
  int sf_s[5] = {0, 0, 0, 0, 0};
};

extern "C" {

/// A 的 SF 元素数(`[M, ceil(K/16)]`,交错布局)。
///
/// **要用 `cosize` 不是 `size`**:SF 布局里有一个 **stride-0 维**(每个 sf 字节覆盖
/// 16 个 k),`size()` 是 shape 乘积(含那个 0 步长的 16 维)⇒ **恰好偏大 16 倍**;
/// `cosize()` 才是真实字节跨度(= `M*K/16`)。曾经用 `size()` 分配 ⇒ 每个 plan
/// 白白多占 15/16(权重 SF 上就是几十 MB)。
size_t owl_nvfp4_sfa_elems(int M, int K) {
  if (M <= 0 || K <= 0 || (K % 16) != 0) return 0;
  auto l = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, 1, K, 1));
  return (size_t)cosize(l);
}

/// B 的 SF 元素数(**只依赖 N,K** ⇒ 权重只打包一次)
size_t owl_nvfp4_sfb_elems(int N, int K) {
  if (N <= 0 || K <= 0 || (K % 16) != 0) return 0;
  auto l = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(1, N, K, 1));
  return (size_t)cosize(l);
}

/// host 打包激活/权重。`a` = f16 `[rows,K]`;`q` 要 `rows*K/2` 字节;
/// `sf` 要对应 `*_elems` 字节。返回 per-tensor 全局缩放(与另一个操作数的相乘)。
float owl_nvfp4_pack_a(int M, int K, const __half* a, uint8_t* q, uint8_t* sf) {
  auto l = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, 1, K, 1));
  auto t = make_tensor(sf, l);
  return owl_nvfp4_detail::quantize(a, M, K, q, t);
}

float owl_nvfp4_pack_b(int N, int K, const __half* b, uint8_t* q, uint8_t* sf) {
  auto l = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(1, N, K, 1));
  auto t = make_tensor(sf, l);
  return owl_nvfp4_detail::quantize(b, N, K, q, t);
}

/// **并行版权重打包**(2026-10-28 装载提速):与 [`owl_nvfp4_pack_b`] **逐字节
/// 等价**,只是把行按 `nthreads` 切开并行做(行间无共享写)。
///
/// 为何安全:`g` 先由 `quantize_g` 单独算(全张量 amax,纯读),之后每个
/// `(row, kg)` 只写**自己**的 SF 字节与自己的 fp4 段;`q` 的 memset 在并发前完成。
///
/// 为何要:装载 4B 时 host 打包曾是 +40s 的主要成本(`pack_b` 单线程 ≈ 81M 元素/s)。
float owl_nvfp4_pack_b_parallel(int N, int K, const __half* b, uint8_t* q, uint8_t* sf,
                                int nthreads) {
  auto l = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(1, N, K, 1));
  auto t = make_tensor(sf, l);
  const float g = owl_nvfp4_detail::quantize_g(b, N, K);
  memset(q, 0, (size_t)N * (K / 2));
  if (nthreads <= 1 || N < 256) {
    owl_nvfp4_detail::quantize_rows(b, N, K, q, t, 0, N, g);
    return g;
  }
  const int th = nthreads;
  const int per = (N + th - 1) / th;
  std::vector<std::thread> pool;
  pool.reserve((size_t)th);
  for (int i = 0; i < th; ++i) {
    const int r0 = i * per;
    const int r1 = std::min(N, r0 + per);
    if (r0 >= r1) break;
    pool.emplace_back([&, r0, r1] {
      // 每个线程一份独立的 layout 副本(只读;避免共享 Tensor 的引用语义)
      auto li = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(1, N, K, 1));
      auto ti = make_tensor(sf, li);
      owl_nvfp4_detail::quantize_rows(b, N, K, q, ti, r0, r1, g);
    });
  }
  for (auto& w : pool) w.join();
  return g;
}

/// 建计划(`M,N,K` 固定);失败返回 `nullptr`。
void* owl_nvfp4_plan_new(int M, int N, int K) {
  if (M <= 0 || N <= 0 || K <= 0) return nullptr;
  // **M 任意**(2026-10-28 用户裁决:M 不设分流门 ⇒ decode 的 m=1 也走这条臂;
  // CUTLASS 的 tile 谓词会自己处理 M 不足 128 的行)。N 只要求 8 对齐
  // (epilogue 的 f16 向量化),K 必须 16 对齐(块缩放粒度)。
  auto* p = new OwlNvfp4Plan();
  p->M = M; p->N = N; p->K = K;
  p->l_sfa = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
  p->l_sfb = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));
  // **cosize 不是 size**:SF 布局有一个 stride-0 维(每字节覆盖 16 个 k),
  // `size()` 是 shape 乘积 ⇒ 恰好偏大 16 倍(2026-10-28 实测 23.6MB vs 1.47MB)。
  p->sfa_bytes = (size_t)cosize(p->l_sfa);
  p->sfb_bytes = (size_t)cosize(p->l_sfb);
  // ---- SFA 仿射模型(设备侧量化器要用)----
  // 布局 = (32,4) atom × 128 行分块 × 4-组 k 分块;先取步长,再逐点校验。
  if ((K % 16) != 0 || (N % 8) != 0) { delete p; return nullptr; }
  auto sfa_at = [&](int m, int gg) -> long { return (long)p->l_sfa(m, 16 * gg, 0); };
  const int groups = K / 16;
  const long base = sfa_at(0, 0);
  p->sf_s[0] = (int)(sfa_at(1, 0) - base);
  p->sf_s[1] = (int)(sfa_at(32, 0) - base);
  p->sf_s[2] = (int)(sfa_at(128, 0) - base);
  p->sf_s[3] = (int)(sfa_at(0, 1) - base);
  p->sf_s[4] = (int)(sfa_at(0, 4) - base);
  auto affine = [&](int m, int gg) -> long {
    const int m0 = m % 32, m1 = (m / 32) % 4, m2 = m / 128;
    const int g0 = gg % 4, g1 = gg / 4;
    return base + (long)m0 * p->sf_s[0] + (long)m1 * p->sf_s[1] + (long)m2 * p->sf_s[2] +
           (long)g0 * p->sf_s[3] + (long)g1 * p->sf_s[4];
  };
  // 采样校验(边界 + 伪随机);不一致 ⇒ 拒绝建 plan(宁可不跑,不写错位)
  bool ok = true;
  for (int i = 0; i < 4096 && ok; ++i) {
    const int m = (i < 256) ? (i % M) : (int)((unsigned)(i * 2654435761u) % (unsigned)M);
    const int gg = (int)((unsigned)(i * 40503u + 7u) % (unsigned)groups);
    ok = (sfa_at(m, gg) == affine(m, gg));
  }
  if (!ok) {
    delete p;
    return nullptr;  // 布局结构变了(新的 K 分块):设备侧量化器需同步改
  }
  // workspace 与 args 无关的部分(当前实测 0,但不依赖这个巧合:预分配按需真算)
  StrideA stride_A{(int64_t)K, cute::C<1>{}, (int64_t)M * K};
  StrideB stride_B{(int64_t)K, cute::C<1>{}, (int64_t)N * K};
  typename Epilogue::Arguments epi;
  epi.thread.alpha = 1.f;
  epi.thread.beta = 0.f;
  epi.thread.alpha_ptr = nullptr;
  epi.thread.beta_ptr = nullptr;
  epi.ptr_D = nullptr;
  typename Kernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {nullptr, stride_A, nullptr, stride_B, nullptr, p->l_sfa, nullptr, p->l_sfb},
      epi};
  p->ws_bytes = Gemm::get_workspace_size(args);
  return p;
}

void owl_nvfp4_plan_free(void* plan) { delete static_cast<OwlNvfp4Plan*>(plan); }
size_t owl_nvfp4_plan_sfa_bytes(void* plan) { return static_cast<OwlNvfp4Plan*>(plan)->sfa_bytes; }
size_t owl_nvfp4_plan_sfb_bytes(void* plan) { return static_cast<OwlNvfp4Plan*>(plan)->sfb_bytes; }
size_t owl_nvfp4_plan_ws_bytes(void* plan) { return static_cast<OwlNvfp4Plan*>(plan)->ws_bytes; }

/// 发射一次 `D[M,N] = alpha · A(M,K) × B(N,K)ᵀ`。`ws` 长度须 ≥ `plan_ws_bytes`
/// (为 0 时可传 `nullptr`)。返回 0 = 已发射(尚未同步),负 = 主机侧失败。
///
/// `alpha` 由 host 给(**eager/验证面**;装载期 oracle 与探针用这条)。
int owl_nvfp4_run(void* plan, void* stream,
                  const uint8_t* aq, const uint8_t* asf, float ga,
                  const uint8_t* bq, const uint8_t* bsf, float gb,
                  __half* d, void* ws) {
  auto* p = static_cast<OwlNvfp4Plan*>(plan);
  if (!p || !stream) return -1;
  const int M = p->M, N = p->N, K = p->K;
  StrideA stride_A{(int64_t)K, cute::C<1>{}, (int64_t)M * K};
  StrideB stride_B{(int64_t)K, cute::C<1>{}, (int64_t)N * K};
  using StrideC = typename Epilogue::StrideC;
  using StrideD = typename Epilogue::StrideD;
  StrideC stride_C{(int64_t)N, cute::C<1>{}, (int64_t)M * N};
  StrideD stride_D{(int64_t)N, cute::C<1>{}, (int64_t)M * N};
  typename Epilogue::Arguments epi;
  epi.thread.alpha = ga * gb;
  epi.thread.beta = 0.f;
  epi.thread.alpha_ptr = nullptr;
  epi.thread.beta_ptr = nullptr;
  epi.ptr_C = nullptr;
  epi.dC = stride_C;
  epi.ptr_D = reinterpret_cast<typename Epilogue::ElementD*>(d);
  epi.dD = stride_D;
  typename Kernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {reinterpret_cast<ElementA const*>(aq), stride_A,
       reinterpret_cast<ElementB const*>(bq), stride_B,
       reinterpret_cast<ElementSF const*>(asf), p->l_sfa,
       reinterpret_cast<ElementSF const*>(bsf), p->l_sfb},
      epi};
  Gemm gemm_op;
  if (gemm_op.can_implement(args) != cutlass::Status::kSuccess) return -2;
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  if (gemm_op.initialize(args, ws, s) != cutlass::Status::kSuccess) return -3;
  if (gemm_op.run(s) != cutlass::Status::kSuccess) return -4;
  return 0;
}

/// **alpha 来自设备指针**版(2026-10-28 进图刀;生产路径)。
///
/// `alpha_dev` = 设备端 f32[1] = `ga·gb`(`k_scale_from_amax` 写入)。
/// 与 [`owl_nvfp4_run`] **同一条 GEMM**,只换 `alpha_ptr` 的来源 ⇒ 两次发射
/// 数值逐位一致(host 的 `ga*gb` 与设备的 `g*gb` 都是 f32 乘法)。
int owl_nvfp4_run_dev(void* plan, void* stream,
                      const uint8_t* aq, const uint8_t* asf, const float* alpha_dev,
                      const uint8_t* bq, const uint8_t* bsf,
                      __half* d, void* ws) {
  auto* p = static_cast<OwlNvfp4Plan*>(plan);
  if (!p || !stream || !alpha_dev) return -1;
  const int M = p->M, N = p->N, K = p->K;
  StrideA stride_A{(int64_t)K, cute::C<1>{}, (int64_t)M * K};
  StrideB stride_B{(int64_t)K, cute::C<1>{}, (int64_t)N * K};
  using StrideC = typename Epilogue::StrideC;
  using StrideD = typename Epilogue::StrideD;
  StrideC stride_C{(int64_t)N, cute::C<1>{}, (int64_t)M * N};
  StrideD stride_D{(int64_t)N, cute::C<1>{}, (int64_t)M * N};
  typename Epilogue::Arguments epi;
  epi.thread.alpha = 1.f;   // 未用(alpha_ptr 在场时 `Sm90ScalarBroadcast` 优先取指针)
  epi.thread.beta = 0.f;
  epi.thread.alpha_ptr = alpha_dev;
  epi.thread.beta_ptr = nullptr;
  epi.ptr_C = nullptr;
  epi.dC = stride_C;
  epi.ptr_D = reinterpret_cast<typename Epilogue::ElementD*>(d);
  epi.dD = stride_D;
  typename Kernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {reinterpret_cast<ElementA const*>(aq), stride_A,
       reinterpret_cast<ElementB const*>(bq), stride_B,
       reinterpret_cast<ElementSF const*>(asf), p->l_sfa,
       reinterpret_cast<ElementSF const*>(bsf), p->l_sfb},
      epi};
  Gemm gemm_op;
  if (gemm_op.can_implement(args) != cutlass::Status::kSuccess) return -2;
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  if (gemm_op.initialize(args, ws, s) != cutlass::Status::kSuccess) return -3;
  if (gemm_op.run(s) != cutlass::Status::kSuccess) return -4;
  return 0;
}

/// 版本/自述(诊断面:确认编进去的是哪个算术形态)
const char* owl_nvfp4_banner() {
  return "owl_nvfp4 v1 · NVFP4(e2m1+e4m3/16) · CUTLASS SM120 blockscaled mainloop · Tile 128x128x256";
}

// ---------------------------------------------------------------------------
// 设备侧激活量化器(实现在 `nvfp4_quant_act.cu`,同一个 `.a`)
// ---------------------------------------------------------------------------
int owl_nvfp4_quant_act_launch(const __half* x, int M, int K, unsigned char* q,
                               unsigned char* sf, float* d_g, float gb,
                               int s_m0, int s_m1, int s_m2, int s_g0, int s_g1, void* stream);

/// 把 f16 激活 `[M,K]` 在设备上量化成 fp4 + SF,并在 `d_g`(**f32[2]**)写出
/// `g` 与 `alpha = g·gb`。`d_g` **由 launch 自己清零**(调用方不必预清)
/// 返回 0 = 已发射;非 0 = 主机侧失败(plan 形状不匹配 / 仿射布局不受支持)。
int owl_nvfp4_quant_act(void* plan, const void* x, unsigned char* q, unsigned char* sf,
                        float* d_g, float gb, void* stream) {
  auto* p = static_cast<OwlNvfp4Plan*>(plan);
  if (!p || !stream) return -1;
  return owl_nvfp4_quant_act_launch(reinterpret_cast<const __half*>(x), p->M, p->K, q, sf, d_g,
                                    gb,
                                    p->sf_s[0], p->sf_s[1], p->sf_s[2], p->sf_s[3], p->sf_s[4],
                                    stream);
}

}  // extern "C"

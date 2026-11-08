// ============================================================================
// owl **SM100 TMEM 原子 shim** —— 从 `cute/arch/copy_sm100.hpp`（356 KB）里只抄出
// 我们“**仅作名字**”需要的两个 TMEM_LOAD 原子。
// ============================================================================
//
// ## 为什么要它（第 6 刀，2026-10-29）
//
// `cute/arch/copy_sm100.hpp` 定义了 **200+ 个** SM100 原子（TMEM 读写、LDSM/STSM、
// 256-bit 全局访存…）。我们的两个 TU 真正会**实例化**的一个都没有：
//   - FP4 主循环走 TMA（`cute/arch/copy_sm90_tma.hpp`）+ UTCCP；
//   - epilogue 走 **SM90 TMA warp-specialized**，不碰 TMEM。
//
// 但 `cutlass/gemm/dispatch_policy.hpp` 在**模板实参**里直接写了两个名字
// （`Sm100TmaUmmaWarpSpecialized…` 的 TMEM epilogue 分派里）：
//
//     SM100_TMEM_LOAD_16dp256b1x / SM100_TMEM_LOAD_32dp32b32x
//
// 非依赖名 ⇒ **解析期就必须可见**，而它所在的那支 Blackwell epilogue 从不实例化。
// 于是只抄这两个 struct（原样逐字，含 `tcgen05.ld` 的 asm 与 `#if` 保护），
// 让 `copy_sm100.hpp` 整块下树：**356 KB → 本文件 ~60 行**。
//
// ## 纪律
// - **先于 `cute/atom/copy_traits_sm100.hpp` 被 include**（后者末尾的 `using`
//   声明要在这里找到符号）。
// - 抄的是**原样**：将来真要走 TMEM epilogue 时，这两个原子仍是对的（不是空壳）。
//   但那时更需要的是把整份 `copy_sm100.hpp` 从 git 历史恢复回来。
// ============================================================================
#pragma once

#include <cstdint>

#include <cute/config.hpp>   // CUTE_HOST_DEVICE / CUTE_INVALID_CONTROL_PATH
#include <cute/arch/copy.hpp>

namespace cute {
namespace SM100 {
namespace TMEM {
namespace LOAD {

// 16 data path lanes, 256-bit pattern, repeated 1 times
struct SM100_TMEM_LOAD_16dp256b1x {
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void copy(uint32_t const& src_addr, uint32_t& dst0, uint32_t& dst1,
                                    uint32_t& dst2, uint32_t& dst3) {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile(
        "tcgen05.ld.sync.aligned.16x256b.x1.b32"
        "{%0, %1, %2, %3},"
        "[%4];\n"
        : "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
        : "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

// 32 data path lanes, 32-bit pattern, repeated 32 times
struct SM100_TMEM_LOAD_32dp32b32x {
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[32];

  CUTE_HOST_DEVICE static void copy(uint32_t const& src_addr, uint32_t& dst00, uint32_t& dst01,
                                    uint32_t& dst02, uint32_t& dst03, uint32_t& dst04,
                                    uint32_t& dst05, uint32_t& dst06, uint32_t& dst07,
                                    uint32_t& dst08, uint32_t& dst09, uint32_t& dst10,
                                    uint32_t& dst11, uint32_t& dst12, uint32_t& dst13,
                                    uint32_t& dst14, uint32_t& dst15, uint32_t& dst16,
                                    uint32_t& dst17, uint32_t& dst18, uint32_t& dst19,
                                    uint32_t& dst20, uint32_t& dst21, uint32_t& dst22,
                                    uint32_t& dst23, uint32_t& dst24, uint32_t& dst25,
                                    uint32_t& dst26, uint32_t& dst27, uint32_t& dst28,
                                    uint32_t& dst29, uint32_t& dst30, uint32_t& dst31) {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
        "{%0, %1, %2, %3,"
        "%4, %5, %6, %7,"
        "%8, %9, %10, %11,"
        "%12, %13, %14, %15,"
        "%16, %17, %18, %19,"
        "%20, %21, %22, %23,"
        "%24, %25, %26, %27,"
        "%28, %29, %30, %31},"
        "[%32];\n"
        : "=r"(dst00), "=r"(dst01), "=r"(dst02), "=r"(dst03), "=r"(dst04), "=r"(dst05),
          "=r"(dst06), "=r"(dst07), "=r"(dst08), "=r"(dst09), "=r"(dst10), "=r"(dst11),
          "=r"(dst12), "=r"(dst13), "=r"(dst14), "=r"(dst15), "=r"(dst16), "=r"(dst17),
          "=r"(dst18), "=r"(dst19), "=r"(dst20), "=r"(dst21), "=r"(dst22), "=r"(dst23),
          "=r"(dst24), "=r"(dst25), "=r"(dst26), "=r"(dst27), "=r"(dst28), "=r"(dst29),
          "=r"(dst30), "=r"(dst31)
        : "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

}  // namespace LOAD
}  // namespace TMEM
}  // namespace SM100
}  // namespace cute

// ---------------------------------------------------------------------------
// FP4 主循环真的**会实例化**的 4 个 LDSM 原子（`ldmatrix ... b4x16_p64 / b6x16_p32`）
//
// 它们被 `cutlass/gemm/collective/builders/sm120_common.inl` 的 `ss_smem_selector`
// 选做 **fp4 操作数的 smem→reg 拷贝**（非依赖名，所以必须定义，不只是声明）。
// 逐字抄自 `cute/arch/copy_sm100.hpp`（已删，见 git 历史）。
// ---------------------------------------------------------------------------
namespace cute {

struct SM100_SU4_DU8x16_x1_LDSM_N {
  using SRegisters = uint128_t[1];
  using DRegisters = uint32_t[1];

  CUTE_DEVICE static void copy(uint128_t const& smem_src, uint32_t& dst0) {
#if defined(CUTE_ARCH_LDSM_SM100A_ENABLED)
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_src);
    asm volatile("ldmatrix.sync.aligned.m8n16.x1.shared.b8x16.b4x16_p64  {%0}, [%1];\n"
                 : "=r"(dst0)
                 : "r"(smem_int_ptr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use ldmatrix without CUTE_ARCH_LDSM_SM100A_ENABLED.");
#endif
  }
};

struct SM100_SU4_DU8x16_x2_LDSM_N {
  using SRegisters = uint128_t[1];
  using DRegisters = uint32_t[2];

  CUTE_DEVICE static void copy(uint128_t const& smem_src, uint32_t& dst0, uint32_t& dst1) {
#if defined(CUTE_ARCH_LDSM_SM100A_ENABLED)
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_src);
    asm volatile("ldmatrix.sync.aligned.m8n16.x2.shared.b8x16.b4x16_p64  {%0, %1}, [%2];\n"
                 : "=r"(dst0), "=r"(dst1)
                 : "r"(smem_int_ptr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use ldmatrix without CUTE_ARCH_LDSM_SM100A_ENABLED.");
#endif
  }
};

struct SM100_SU4_DU8x16_x4_LDSM_N {
  using SRegisters = uint128_t[1];
  using DRegisters = uint32_t[4];

  CUTE_DEVICE static void copy(uint128_t const& smem_src, uint32_t& dst0, uint32_t& dst1,
                               uint32_t& dst2, uint32_t& dst3) {
#if defined(CUTE_ARCH_LDSM_SM100A_ENABLED)
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_src);
    asm volatile(
        "ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b4x16_p64  {%0, %1, %2, %3}, [%4];\n"
        : "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
        : "r"(smem_int_ptr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use ldmatrix without CUTE_ARCH_LDSM_SM100A_ENABLED.");
#endif
  }
};

struct SM100_SU6_DU8x16_x4_LDSM_N {
  using SRegisters = uint128_t[1];
  using DRegisters = uint32_t[4];

  CUTE_DEVICE static void copy(uint128_t const& smem_src, uint32_t& dst0, uint32_t& dst1,
                               uint32_t& dst2, uint32_t& dst3) {
#if defined(CUTE_ARCH_LDSM_SM100A_ENABLED)
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_src);
    asm volatile(
        "ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b6x16_p32  {%0, %1, %2, %3}, [%4];\n"
        : "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
        : "r"(smem_int_ptr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use ldmatrix without CUTE_ARCH_LDSM_SM100A_ENABLED.");
#endif
  }
};

}  // namespace cute

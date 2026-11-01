// ============================================================================
// owl shim -- `flashinfer/fp16.h` 的硬件指令版(2026-11-02,OWL 自有,非上游拷贝)
//
// 为什么需要:上游 `attention/prefill.cuh` 在 `FP16_QK_REDUCTION_SUPPORTED` 档
// (`#include "../fp16.h"`)里用 `fp16_ieee_{to,from}_fp32_value()` 做 S 片段的
// half<->float 往返。上游那份是 Marat Dukhan / AMD 的软件位运算实现(~120 行,
// 需 `<bit>`/C++20 + `boost/math/ccmath/fabs.hpp`)-- 在 device 侧是十几条
// 位运算 + 一次浮点乘,而 sm_86 有单指令的 `cvt.f32.f16` / `cvt.rn.f16.f32`。
//
// 本文件用 `__half2float` / `__float2half_rn`(PTX cvt 单指令)提供同名函数,
// => prefill.cuh 零改动,且免掉 boost 依赖。数值:RN 舍入与上游软件实现同舍入律
// (两者都是 IEEE round-to-nearest-even);超范围(+-inf/NaN)直通。
//
// 归属:本文件由 owl 编写(不复制上游实现),接口签名与上游同名以保持
// prefill.cuh 逐字可升级;登记见 THIRD_PARTY.md §2 的 in-tree changes 列。
// ============================================================================
#pragma once

#ifndef FLASHINFER_FP16_H
#define FLASHINFER_FP16_H

#include <cuda_fp16.h>
#include <cstdint>

/// half 位型 -> float(单指令 `cvt.f32.f16`;NaN/Inf/denormal 直通)
__device__ __forceinline__ float fp16_ieee_to_fp32_value(uint16_t h) {
  return __half2float(__ushort_as_half(h));
}

/// float -> half 位型(单指令 `cvt.rn.f16.f32`;超范围饱和到 +-inf,与上游同)
__device__ __forceinline__ uint16_t fp16_ieee_from_fp32_value(float f) {
  return __half_as_ushort(__float2half_rn(f));
}

#endif  // FLASHINFER_FP16_H

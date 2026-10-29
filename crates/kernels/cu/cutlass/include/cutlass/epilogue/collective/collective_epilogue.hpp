/***************************************************************************************************
 * Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

// ============================================================================
// **owl 裁剪版(2026-10-28)** —— 不是 NVIDIA 原文件的全集。
//
// 原版把 sm70(vectorized)、sm90(TMA warp-specialized / bias / array)、
// sm100(nosmem / tma / array)的 epilogue collective 全部 include 进来。
// 本仓只用 **sm90 TMA warp-specialized** 那一条:sm120 的 builder 把
// `Sm120TmaWarpSpecialized` 经 `GetBaseDispatchPolicy` 映射到 `Sm90TmaWarpSpecialized`
// (见 `builders/sm120_builder.inl` 尾部),实际例化的就是它。
// `default_epilogue_array` 与 `..._bias_elementwise` 只被 sm90 builder 的
// 分支**按名引用**(非依赖名查找 ⇒ 必须在场),不例化。
//
// 升级 CUTLASS:全量覆盖后按本清单重裁;漏了会在编译期点名。
// ============================================================================
#pragma once

#include <cutlass/detail/dependent_false.hpp>

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue::collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  class DispatchPolicy,
  class... Args
>
class CollectiveEpilogue {
  static_assert(cutlass::detail::dependent_false<DispatchPolicy>, "Could not find an epilogue specialization.");
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::epilogue::collective

/////////////////////////////////////////////////////////////////////////////////////////////////

#include "detail.hpp"

//
// Gemm
//
#include "default_epilogue.hpp"
#include "default_epilogue_array.hpp"
#include "sm90_epilogue_tma_warpspecialized_bias_elementwise.hpp"
#include "sm90_epilogue_tma_warpspecialized.hpp"

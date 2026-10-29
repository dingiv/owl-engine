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
#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////
#include "cutlass/gemm/collective/collective_mma_decl.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

// ============================================================================
// **owl 裁剪点(2026-10-28)**:原版这里按序 include 了 sm90/sm100/sm103/
// sm107/sm120 的全部 builder(30 个 `.inl`,连带整棵 UMMA/GMMA 内核树)。
// 本仓只用 **sm120 blockscaled**(NVFP4 的唯一条路),其余代际走 Marlin/GGUF
// 的自写核 —— 所以只留它,加它真正需要的公共面:
// - `collective_builder_decl.hpp`:主模板声明(不是代际-specific)
// - `sm100_blockscaled_umma_builder.inl`:它定义了
//   `sm100_compute_stage_count_or_override_blockscaled`(blockscaled 流水级数
//   启发式,sm120 builder 直接用)—— 只为一个函数引整个文件,但被引的那份
//   builder 本身已在闭包里(cute/cutlass 的 pipeline/UMMA trait 复用)
// 升级 CUTLASS 时照旧全量覆盖,再按同口径重裁一次。
// ============================================================================
#include "cutlass/gemm/collective/collective_builder_decl.hpp"
#include "cutlass/gemm/collective/builders/sm100_blockscaled_umma_builder.inl"
#include "cutlass/gemm/collective/builders/sm120_blockscaled_mma_builder.inl"

/////////////////////////////////////////////////////////////////////////////////////////////////

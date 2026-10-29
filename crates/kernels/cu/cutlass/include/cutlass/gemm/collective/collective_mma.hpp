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
// 原版把 sm70/sm80/sm90(GMMA,含 sparse/fp8/blockwise)/sm100(UMMA 全家:
// array/rcggemm/emulated/interleaved/planar/mixed-input/sparse/blockwise)/
// sm103/sm107/sm120 的 **40 多个 mainloop** 全部 include 进来,用来支撑
// `CollectiveBuilder` 的任意组合。本仓只用 **sm120 blockscaled**(NVFP4 唯一的
// 那条路;30/40 系走 Marlin/GGUF 的自写核),所以这里只留它。
//
// 升级 CUTLASS:全量覆盖后按本文件的 include 清单重裁;漏了会在编译期
// 直接点名(sm120 builder 找不到 `CollectiveMma` 特化),不会静默走错。
// ============================================================================
#pragma once

#include "cutlass/gemm/collective/collective_mma_decl.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

#include "cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

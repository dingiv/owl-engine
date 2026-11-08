/***************************************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cute/config.hpp>
#include <cute/pointer_sparse.hpp>
#include <cute/tensor_impl.hpp>
#include <cute/arch/mma_sm100.hpp>
#include <cute/arch/mma_sm100_desc.hpp>
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>         // cute::TMEM::

#include <cute/atom/mma_traits.hpp>
// **owl 裁剪点(2026-10-29)**:两行 GMMA traits 已裁 —— 本文件只用到它们的
// smem 布局原子(`GMMA::Layout_*_Atom` / `GMMA::Layout_*_SpAtom`),那些已搬到
// `cu/nvfp4/owl_cutlass_gmma_shim.hpp`(必须在引入本头之前 include)。
#include <cute/atom/copy_traits_sm100.hpp>            // UTCCP smem desc

#include <cute/numeric/numeric_types.hpp>
#include <cute/atom/mma_traits_sm100_frag.hpp>  // UMMA::TmemAllocMode, tmem_frg, tmem_e_frg, tmem_sf_frg, tmem_frg_ws

// Check that aggregate initialization in .with() initializes all fields
#if defined(__GNUG__)
#pragma GCC diagnostic warning "-Wmissing-field-initializers"
#pragma GCC diagnostic error "-Wmissing-field-initializers"
#endif

namespace cute {

namespace UMMA {

//////////////////////////////////////////////////
// Common layouts for UMMA Shared Memory //
//////////////////////////////////////////////////

using cute::GMMA::Layout_MN_INTER_Atom;
using cute::GMMA::Layout_MN_SW32_Atom;
using cute::GMMA::Layout_MN_SW64_Atom;
using cute::GMMA::Layout_MN_SW128_Atom;
using cute::GMMA::Layout_K_INTER_Atom;
using cute::GMMA::Layout_K_SW32_Atom;
using cute::GMMA::Layout_K_SW64_Atom;
using cute::GMMA::Layout_K_SW128_Atom;

using Layout_MN_SW128_32B_Atom_Bits = ComposedLayout<Swizzle<2,5,2>, smem_ptr_flag, Layout<Shape< _1024,_4>,Stride<_1, _1024>>>;

template <class Type>
using Layout_MN_SW128_32B_Atom = decltype(upcast<sizeof_bits<Type>::value>(Layout_MN_SW128_32B_Atom_Bits{}));

//////////////////////////////////////////////////
// Common layouts for Sparse UMMA Shared Memory //
//////////////////////////////////////////////////

using cute::GMMA::Layout_MN_INTER_SpAtom;
using cute::GMMA::Layout_MN_SW32_SpAtom;
using cute::GMMA::Layout_MN_SW64_SpAtom;
using cute::GMMA::Layout_MN_SW128_SpAtom;
using cute::GMMA::Layout_K_INTER_SpAtom;
using cute::GMMA::Layout_K_SW32_SpAtom;
using cute::GMMA::Layout_K_SW64_SpAtom;
using cute::GMMA::Layout_K_SW128_SpAtom;

template <class Type, int S>
using Layout_MN_SW128_32B_SpAtom = ComposedLayout<Swizzle<2,5,2>, smem_sparse_ptr_flag_bits<S,sizeof_bits_v<Type>>,
                                                  decltype(blocked_product(Layout<Shape<_1,Int<S>>>{}, Layout_MN_SW128_32B_Atom<Type>{}.layout_b()))>;

// With UMMA::Major param
template <class Type, int S, UMMA::Major tnsp>
using Layout_INTER_SpAtom = typename conditional<tnsp == UMMA::Major::MN,
                                                 Layout_MN_INTER_SpAtom<Type,S>,
                                                 Layout_K_INTER_SpAtom<Type,S>>::type;
template <class Type, int S, UMMA::Major tnsp>
using Layout_SW32_SpAtom = typename conditional<tnsp == UMMA::Major::MN,
                                                Layout_MN_SW32_SpAtom<Type,S>,
                                                Layout_K_SW32_SpAtom<Type,S>>::type;
template <class Type, int S, UMMA::Major tnsp>
using Layout_SW64_SpAtom = typename conditional<tnsp == UMMA::Major::MN,
                                                Layout_MN_SW64_SpAtom<Type,S>,
                                                Layout_K_SW64_SpAtom<Type,S>>::type;
template <class Type, int S, UMMA::Major tnsp>
using Layout_SW128_SpAtom = typename conditional<tnsp == UMMA::Major::MN,
                                                 Layout_MN_SW128_SpAtom<Type,S>,
                                                 Layout_K_SW128_SpAtom<Type,S>>::type;

// Tile a MN-logical layout atom to an MMA Tile Shape ((MMA_M,MMA_N),M_MMAs,N_MMAs,...)
template <class LayoutAtom, class MMATileShape, class ModeOrder = GenColMajor>
CUTE_HOST_DEVICE constexpr
auto
tile_to_mma_shape(LayoutAtom const& atom, MMATileShape const& mma_tile_shape, ModeOrder const& order = {})
{
  constexpr int R = decltype(rank(mma_tile_shape))::value;
  auto mn_shape = cute::tuple_cat(zip(shape<0>(mma_tile_shape), take<1,3>(mma_tile_shape)), take<3,R>(mma_tile_shape));
  auto mn_tiled = tile_to_shape(atom, mn_shape, order);                      // (BLK_M,BLK_N,...)
  return tiled_divide(mn_tiled, product_each(shape<0>(mma_tile_shape)));     // ((MMA_M,MMA_N),M_MMAs,N_MMAs,...)
}

} // end namespace UMMA

// Special smem_desc_iter tensor entry for UTCCP copy.
template <class UtccpOp, class TEngine, class TLayout>
constexpr auto get_utccp_smem_desc_tensor(Tensor<TEngine, TLayout> const& smem_utccp_partitioned_tensor) {
  using VecLayout = decltype(layout<0>(TLayout{}));
  static_assert(VecLayout::rank == 2 && shape<1>(VecLayout{}) == 1, "Mismatched vec_mode tensor.");
  static_assert(is_smem<TEngine>::value, "Expect vec_mode smem_tesnor.");
  static_assert(is_static<VecLayout>::value, "Utccp copy tensor's vec_mode should be static.");

  using value_type = typename TEngine::value_type;
  using UtccpTaits = Copy_Traits<UtccpOp>;

  // UtccpTaits::ValID: logical_bit_idx -> tmem_offset.
  // We arrange the logical_bit_idx in order of (core_matrix_strided, core_matrix_leading, repeat(only in 64dplw01), broadcast).
  // So we only need the first two modes for src smem_tensor.
  auto utccp_core_matrix_shape = take<0,2>(upcast<sizeof_bits_v<value_type>>(typename UtccpTaits::ValID{}).shape());
  // logical_bit_idx -> smem_addr
  Layout vec_v_layout = flatten(layout<0>(VecLayout{}));
  Layout utccp_core_matrix_layout = vec_v_layout.with_shape(utccp_core_matrix_shape);
  Tensor utccp_core_matrix_tensor = group_modes<0,2>(make_tensor(smem_utccp_partitioned_tensor.data(), utccp_core_matrix_layout));
  Tensor core_matrix_desc_tensor = make_tensor<UMMA::smem_desc<UMMA::Major::K>>(utccp_core_matrix_tensor);
  return make_tensor(core_matrix_desc_tensor.data(), recast_layout<value_type, uint128_t>(smem_utccp_partitioned_tensor.layout()));
}

// ============================================================================
// **owl 剪口（第 7 刀，2026-10-29）**：本文件原 3,866 行 / 172 KB，其中 3,717 行是
// **48 个 `MMA_Traits<SM100_MMA_*>` 特化**（Blackwell tcgen05 UMMA 的算子描述）。
// 本仓的 FP4 主循环用 **sm120 blockscaled atom**（traits 在 `mma_traits_sm120.hpp`），
// `symbols` 模式实测只从这里要 4 个名字：
//   `UMMA::Layout_K_{SW128,SW64,SW32,INTER}_Atom`（`sm120_common.inl` 的 smem 选择器
//   用作**非依赖名**）+ `UMMA::smem_desc` / `get_utccp_smem_desc_tensor`。
// ⇒ 保留 UMMA 布局原子与 UTCCP 辅助，48 个 traits 特化整段删除。
// 需要 Blackwell tcgen05 时从 git 历史恢复本文件。
// ============================================================================

} // end namespace cute

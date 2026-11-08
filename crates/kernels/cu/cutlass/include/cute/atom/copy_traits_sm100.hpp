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

// **owl 剪口（第 6 刀，2026-10-29）**：原来这里 include 的是
// `cute/arch/copy_sm100.hpp`（356 KB、200+ 个 SM100 原子）。我们的两个 TU
// 一个都不实例化（主循环走 TMA+UTCCP，epilogue 走 SM90 TMA warp-specialized），
// 只有 `dispatch_policy.hpp` 的两个模板实参需要**名字** ⇒ 只留一个只含那两个
// TMEM_LOAD 原子的 owl shim，原头整块下树。
#include "../../nvfp4/owl_cutlass_sm100_tmem_atoms.hpp"
#include <cute/arch/tmem_allocator_sm100.hpp>

#include <cute/atom/copy_traits.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/partitioner.hpp>
#include <cute/numeric/numeric_types.hpp>

#include <cute/layout.hpp>

namespace cute
{
////////////////////////////////////////////////////////////////////////////////////////////////////
//
// TMEM Traits and Utilities
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class... Args>
struct Copy_Atom;

/** Generate a TiledCopy from a CopyAtom and a TMEM tensor
 * Example:
 *   Tensor gmem_tensor = ...                                            // (M,N,...)
 *   Tensor tmem_tensor = ...                                            // (M,N,...)
 *   auto tiled_tmem_load = make_tmem_copy(TMEM_LOAD_Operation, tmem_tensor);
 *   auto thr_tmem_load = tiled_tmem_load.get_slice(thread_idx);
 *
 *   Tensor tDtC = thr_tmem_load.partition_S(tmem_tensor);                    // (TMEM_LOAD,TMEM_LOAD_M,TMEM_LOAD_N,...)
 *   Tensor tDgC = thr_tmem_load.partition_D(gmem_tensor);                    // (TMEM_LOAD,TMEM_LOAD_M,TMEM_LOAD_N,...)
 *   Tensor tDrC = make_tensor<ElementAccumulator>(shape(tDgD));         // (TMEM_LOAD,TMEM_LOAD_M,TMEM_LOAD_N,...)
 *
 *   copy(tiled_tmem_load, tDtC, tDrC);       // tmem -> rmem
 *   copy(tDrC, tDgC);                   // rmem -> gmem
 */
template <class CopyOp, class CopyT,
          class TEngine, class TLayout>
CUTE_HOST_DEVICE constexpr
auto
make_tmem_copy(Copy_Atom<CopyOp,CopyT> const& atom,
               Tensor<TEngine,TLayout> const& tmem)
{
  static_assert(is_tmem<TEngine>::value, "Expected TMEM tensor.");
  using T      = typename TEngine::value_type;
  using Traits = typename Copy_Atom<CopyOp, CopyT>::Traits;
  static_assert(sizeof_bits_v<CopyT> == sizeof_bits_v<T>,
                "Expected a CopyAtom with the same type-width as the Tensor.");

  // atom thr idx -> tmem addr    4warps where each warp points to the same position within it's own subpartition
  auto atom_t_layout = Layout<Shape<_32,_4>, Stride<_0, decltype(Int<32>{} * TMEM::DP<T>{})>>{};
  // atom val idx -> tmem addr    Cast the CopyOp's value ids to the proper data width
  auto atom_v_layout = coalesce(upcast<sizeof_bits<T>::value>(typename Traits::ValID{}));

  return make_cotiled_copy(atom, make_layout(atom_t_layout, atom_v_layout), tmem.layout());
}

template <class CopyOp,
          class TEngine, class TLayout>
CUTE_HOST_DEVICE constexpr
auto
make_tmem_copy(CopyOp const&,
               Tensor<TEngine,TLayout> const& tmem)
{
  return make_tmem_copy(Copy_Atom<CopyOp, typename TEngine::value_type>{}, tmem);
}

/** Generate a TV_Tiler from a TMEM tensor
 * Example:
 *   Tensor gmem_tensor = ...                                            // (M,N,...)
 *   Tensor tmem_tensor = ...                                            // (M,N,...)
 *   auto tmem_tiler = make_tmem_warp_partitioner(tmem_tensor);
 *   auto warp_tiler  = tmem_tiler.get_slice(warp_idx);
 *
 *   Tensor tWtC = warp_tiler.partition(tmem_tensor);                    // (WARP_M,WARP_N,...)
 *   Tensor tWgC = warp_tiler.partition(gmem_tensor);                    // (WARP_M,WARP_N,...)
 */
template <class TEngine, class TLayout>
CUTE_HOST_DEVICE constexpr
auto
make_tmem_warp_partitioner(Tensor<TEngine,TLayout> const& tmem)
{
  static_assert(is_tmem<TEngine>::value, "Expected TMEM tensor.");
  using T = typename TEngine::value_type;

  // warp idx -> tmem addr    This is the T in the Layout_TV
  auto atom_t_layout = Layout<_4, decltype(Int<32>{} * TMEM::DP<T>{})>{};

  // tmem coord -> tmem addr
  auto tmem_layout = tmem.layout();
  // tmem addr -> tmem coord    Append 1:0 so off-the-ends get the stride-0
  auto inv_tmem_layout = make_layout(left_inverse(tmem_layout), Layout<_1,_0>{});

  // wid -> tmem_coord
  auto layout_t_tmem = composition(inv_tmem_layout, atom_t_layout);
  //
  // Tiler -- Find the active elements in the TMEM tensor and generate a tiler to extract them
  //

  // Convert to the awkward by-mode tiler to preserve the modes of the tiled TMEM
  auto flat_tmem_shape = product_each(shape(tmem_layout));
  auto flat_tmem_zeros = repeat<rank(flat_tmem_shape)>(Int<0>{});

  auto tiler = transform(make_seq<rank(flat_tmem_shape)>{}, [&](auto i) {
    return filter(composition(make_layout(flat_tmem_shape, replace<i>(flat_tmem_zeros, Int<1>{})), layout_t_tmem));
  });

  //
  // Layout_TV -- Find the (tid,vid) -> tile coord transformation
  //

  // Apply the tiler to a reference and transform the codomain
  // tile_coord -> tmem_coord
  auto tile2tmem = composition(make_layout(flat_tmem_shape), tiler);

  // wid -> tile_coord
  auto layout_tv = composition(left_inverse(tile2tmem), layout_t_tmem);
  return make_tiler_impl(layout_tv, tiler);
}

namespace SM100::TMEM::LOAD {

//
// Specialized copy_unpack implementation for SM100::TMEM::LOAD instructions
//

template <class CopyOp,
          class TS, class SLayout,
          class TD, class DLayout>
CUTE_HOST_DEVICE constexpr
void
copy_unpack(Copy_Traits<CopyOp> const& traits,
            Tensor<TS,SLayout>  const& src,
            Tensor<TD,DLayout>       & dst)
{
  static_assert(is_tmem<TS>::value, "Expected TMEM src.");
  static_assert(is_rmem<TD>::value, "Expected RMEM dst.");

  using SrcType = typename TS::value_type;
  CUTE_STATIC_ASSERT_V((coalesce(layout(src)) == coalesce(upcast<sizeof_bits<SrcType>::value>(typename Copy_Traits<CopyOp>::ValID{}))),
    "Expected src to have the specific TMEM layout required by CopyOp.");

  uint32_t tmem_addr = raw_pointer_cast(src.data());

  using RegTypeDst = typename remove_extent<typename CopyOp::DRegisters>::type;
  Tensor rD = recast<RegTypeDst>(dst);

  constexpr int RegNumDst = extent<typename CopyOp::DRegisters>::value;
  CUTE_STATIC_ASSERT_V(size(rD) == Int<RegNumDst>{},
    "In CopyAtom, dst layout doesn't vectorize into registers. This dst layout is incompatible with this CopyOp.");

  // thread idx <=> DP lane assert.
  // ASSERT TMEM_LOAD thread attemping to access DP lane within sub-partition.
#if defined(__CUDA_ARCH__) && !defined(NDEBUG)
  assert(((uint32_t(threadIdx.x) / 32) % 4) == (((tmem_addr >> 16) / 32) % 4));
#endif

  detail::explode(CopyOp::copy,
                  &tmem_addr, seq<0>{},
                  rD, make_seq<RegNumDst>{});
}

} // end namespace SM100::TMEM::LOAD

namespace SM100::TMEM::STORE {

//
// Specialized copy_unpack implementation for SM100::TMEM::STORE instructions
//

template <class CopyOp,
          class TS, class SLayout,
          class TD, class DLayout>
CUTE_HOST_DEVICE constexpr
void
copy_unpack(Copy_Traits<CopyOp> const& traits,
            Tensor<TS,SLayout>  const& src,
            Tensor<TD,DLayout>       & dst)
{
  static_assert(is_rmem<TS>::value, "Expected RMEM src.");
  static_assert(is_tmem<TD>::value, "Expected TMEM dst.");

  using RegTypeSrc = typename remove_extent<typename CopyOp::SRegisters>::type;
  Tensor rS = recast<RegTypeSrc>(src);

  constexpr int RegNumSrc = extent<typename CopyOp::SRegisters>::value;
  CUTE_STATIC_ASSERT_V(size(rS) == Int<RegNumSrc>{},
    "In CopyAtom, src layout doesn't vectorize into registers. This src layout is incompatible with this tiled copy.");

  using DstType = typename TD::value_type;
  CUTE_STATIC_ASSERT_V((coalesce(layout(dst)) == coalesce(upcast<sizeof_bits<DstType>::value>(typename Copy_Traits<CopyOp>::ValID{}))),
    "Expected dst to have the specific TMEM layout required by CopyOp.");

  uint32_t tmem_addr = raw_pointer_cast(dst.data());

  // thread idx <=> DP lane assert.
  // ASSERT TMEM_LOAD thread attemping to access DP lane within sub-partition.
#if defined(__CUDA_ARCH__) && !defined(NDEBUG)
  assert(((uint32_t(threadIdx.x) / 32) % 4) == (((tmem_addr >> 16) / 32) % 4));
#endif

  detail::explode(CopyOp::copy,
                  rS, make_seq<RegNumSrc>{},
                  &tmem_addr, seq<0>{});
}

} // end namespace SM100::TMEM::STORE

} // namespace cute

// ============================================================================
// **owl 剪口（2026-10-29 第 5 刀）**：本文件原来 3,897 行 / 146 KB，其中
// 3,443 行是 **TMEM_LOAD / TMEM_STORE 的 `Copy_Traits<>` 特化**（Blackwell
// 的 TMEM epilogue 用）。我们的 epilogue 是 **SM90 TMA warp-specialized** ⇒
// 一个都不实例化 ⇒ 整段删除（原文件见 git 历史 / 备份 `TRIMMED.md` §5）。
//
// 唯一还要留的是两个**名字**：`gemm/dispatch_policy.hpp` 在模板实参里直接
// 写了它们（非依赖名 ⇒ 解析期必须可见），但它所在的那支 Blackwell epilogue
// 从未被实例化。所以只补两个 `using`，不补实现。
// ============================================================================
namespace cute {

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp256b1x;
using SM100::TMEM::LOAD::SM100_TMEM_LOAD_32dp32b32x;

} // namespace cute

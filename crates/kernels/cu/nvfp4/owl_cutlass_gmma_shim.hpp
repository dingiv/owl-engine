// ============================================================================
// **owl CUTLASS 裁剪 shim(2026-10-29)** —— 见 `cu/cutlass/TRIMMED.md` §6
// ============================================================================
//
// 为什么有这份文件:本仓的 sm120 块缩放路径**只用到** sm90 家族的
// **smem 布局原子 + 三个选路函数**,而它们住的那三个头
// (`mma_traits_sm90_gmma{,_sparse}.hpp` / `mma_sm90_gmma{,_sparse}.hpp` /
// `sm90_common.inl` / epilogue `sm90_builder.inl`)合计 **2.9 MB**(占裁剪后
// 整棵树的 ~34%),其中 99% 是 GMMA MMA atom 的 trait 特化矩阵 —— 我们一条
// 都不例化。
//
// 所以这里**把它们实际被引用的部分逐字搬出来**(纯类型别名 + 纯 constexpr
// 选路,零代码生成),再把原头从树里去掉。搬迁逐字对齐上游,改动上游时
// 对照 `grep -n "Layout_MN_SW128_Atom_Bits"` 即可。
//
// ⚠️ 本文件必须**先于**任何 `cute/…` 头被 include:上游的
// `cute/atom/copy_traits_sm90_tma.hpp` 在**定义处**就引用
// `GMMA::Layout_MN_SW128_Atom`(非依赖名),那些头没了就得靠这里补齐。
#pragma once

#include <cute/arch/mma_sm90_desc.hpp>   // GMMA::Major / DescriptorIterator / LayoutType
#include <cute/layout_composed.hpp>      // ComposedLayout
#include <cute/pointer_flagged.hpp>      // smem_ptr_flag
#include <cute/pointer_sparse.hpp>       // smem_sparse_ptr_flag_bits
#include <cute/numeric/numeric_types.hpp>
#include <cutlass/detail/dependent_false.hpp>
#include <cutlass/detail/layout.hpp>     // cutlass::gemm::detail::is_major

////////////////////////////////////////////////////////////////////////////////
// GMMA smem 布局原子(逐字搬自 `cute/atom/mma_traits_sm90_gmma.hpp`)
////////////////////////////////////////////////////////////////////////////////
namespace cute::SM90::GMMA {

// `Major` 与 `DescriptorIterator` 上游住在 `mma_sm90_gmma.hpp` /
// `mma_traits_sm90_gmma.hpp`(两个 1MB 级头)。本仓只用到:
// - `Major`(smem 布局选路的参数);
// - `DescriptorIterator` **仅作为类型名**(sm120 mainloop 的 `static_assert`
//   用 `is_base_of` 排除它当 A/B 片段);
// 故逐字搬来最小面。
enum class Major {
  K  = 0,
  MN = 1
};

/// 搬自 `cute/atom/mma_traits_sm90_gmma.hpp`
struct DescriptorIterator
{
  using reference    = GmmaDescriptor;
  using element_type = GmmaDescriptor;
  using value_type   = GmmaDescriptor;

  GmmaDescriptor desc_;

  CUTE_HOST_DEVICE constexpr
  reference operator*() const { return desc_; }

  template <class Index>
  CUTE_HOST_DEVICE constexpr
  reference operator[](Index const& i) const { return *(*this + i); }

  template <class Index>
  CUTE_HOST_DEVICE constexpr
  DescriptorIterator operator+(Index const& offset) const
  {
    GmmaDescriptor ret;
    ret.reg32_[0] = desc_.reg32_[0] + uint32_t(offset);
    ret.reg32_[1] = desc_.reg32_[1];
    return { ret };
  }
};

template <class T>
CUTE_HOST_DEVICE constexpr
GmmaDescriptor
raw_pointer_cast(DescriptorIterator const& ptr) {
  return ptr.desc_;
}

template <class NewT>
CUTE_HOST_DEVICE constexpr
DescriptorIterator
recast_ptr(DescriptorIterator const& iter) {
  static_assert(is_same<NewT, uint64_t>::value, "Can only cast GmmaDescriptorIterator to uint64_t.");
  return iter;
}

template <Major>
struct smem_desc : DescriptorIterator {};

// M|N-major,以 bit 为单位
using Layout_MN_INTER_Atom_Bits = ComposedLayout<Swizzle<0,4,3>, smem_ptr_flag, Layout<Shape< _128,_8>,Stride<_1, _128>>>;
using Layout_MN_SW32_Atom_Bits  = ComposedLayout<Swizzle<1,4,3>, smem_ptr_flag, Layout<Shape< _256,_8>,Stride<_1, _256>>>;
using Layout_MN_SW64_Atom_Bits  = ComposedLayout<Swizzle<2,4,3>, smem_ptr_flag, Layout<Shape< _512,_8>,Stride<_1, _512>>>;
using Layout_MN_SW128_Atom_Bits = ComposedLayout<Swizzle<3,4,3>, smem_ptr_flag, Layout<Shape<_1024,_8>,Stride<_1,_1024>>>;

// K-major,以 bit 为单位
using Layout_K_INTER_Atom_Bits  = ComposedLayout<Swizzle<0,4,3>, smem_ptr_flag, Layout<Shape<_8, _128>,Stride< _128,_1>>>;
using Layout_K_SW32_Atom_Bits   = ComposedLayout<Swizzle<1,4,3>, smem_ptr_flag, Layout<Shape<_8, _256>,Stride< _256,_1>>>;
using Layout_K_SW64_Atom_Bits   = ComposedLayout<Swizzle<2,4,3>, smem_ptr_flag, Layout<Shape<_8, _512>,Stride< _512,_1>>>;
using Layout_K_SW128_Atom_Bits  = ComposedLayout<Swizzle<3,4,3>, smem_ptr_flag, Layout<Shape<_8,_1024>,Stride<_1024,_1>>>;

// 以 Type 为单位
template <class Type>
using Layout_MN_INTER_Atom = decltype(upcast<sizeof_bits<Type>::value>(Layout_MN_INTER_Atom_Bits{}));
template <class Type>
using Layout_MN_SW32_Atom  = decltype(upcast<sizeof_bits<Type>::value>(Layout_MN_SW32_Atom_Bits{}));
template <class Type>
using Layout_MN_SW64_Atom  = decltype(upcast<sizeof_bits<Type>::value>(Layout_MN_SW64_Atom_Bits{}));
template <class Type>
using Layout_MN_SW128_Atom = decltype(upcast<sizeof_bits<Type>::value>(Layout_MN_SW128_Atom_Bits{}));
template <class Type>
using Layout_K_INTER_Atom = decltype(upcast<sizeof_bits<Type>::value>(Layout_K_INTER_Atom_Bits{}));
template <class Type>
using Layout_K_SW32_Atom  = decltype(upcast<sizeof_bits<Type>::value>(Layout_K_SW32_Atom_Bits{}));
template <class Type>
using Layout_K_SW64_Atom  = decltype(upcast<sizeof_bits<Type>::value>(Layout_K_SW64_Atom_Bits{}));
template <class Type>
using Layout_K_SW128_Atom = decltype(upcast<sizeof_bits<Type>::value>(Layout_K_SW128_Atom_Bits{}));

// 带 `GMMA::Major` 参数的形态
template <class Type, Major tnsp>
using Layout_INTER_Atom = typename conditional<tnsp == Major::MN,
                                               Layout_MN_INTER_Atom<Type>,
                                               Layout_K_INTER_Atom<Type>>::type;
template <class Type, Major tnsp>
using Layout_SW32_Atom = typename conditional<tnsp == Major::MN,
                                              Layout_MN_SW32_Atom<Type>,
                                              Layout_K_SW32_Atom<Type>>::type;
template <class Type, Major tnsp>
using Layout_SW64_Atom = typename conditional<tnsp == Major::MN,
                                              Layout_MN_SW64_Atom<Type>,
                                              Layout_K_SW64_Atom<Type>>::type;
template <class Type, Major tnsp>
using Layout_SW128_Atom = typename conditional<tnsp == Major::MN,
                                               Layout_MN_SW128_Atom<Type>,
                                               Layout_K_SW128_Atom<Type>>::type;

////////////////////////////////////////////////////////////////////////////////
// 稀疏(Sp)布局原子(逐字搬自 `cute/atom/mma_traits_sm90_gmma_sparse.hpp`)
// —— 唯一的消费者是 `mma_traits_sm100.hpp` 的 UMMA 稀疏 smem 布局(声明级)。
////////////////////////////////////////////////////////////////////////////////
template <class Type, int S>
using Layout_MN_INTER_SpAtom = ComposedLayout<Swizzle<0,4,3>, smem_sparse_ptr_flag_bits<S,sizeof_bits_v<Type>>,
                                              decltype(blocked_product(Layout<Shape<_1,Int<S>>>{}, Layout_MN_INTER_Atom<Type>{}.layout_b()))>;
template <class Type, int S>
using Layout_MN_SW32_SpAtom  = ComposedLayout<Swizzle<1,4,3>, smem_sparse_ptr_flag_bits<S,sizeof_bits_v<Type>>,
                                              decltype(blocked_product(Layout<Shape<_1,Int<S>>>{}, Layout_MN_SW32_Atom<Type>{}.layout_b()))>;
template <class Type, int S>
using Layout_MN_SW64_SpAtom  = ComposedLayout<Swizzle<2,4,3>, smem_sparse_ptr_flag_bits<S,sizeof_bits_v<Type>>,
                                              decltype(blocked_product(Layout<Shape<_1,Int<S>>>{}, Layout_MN_SW64_Atom<Type>{}.layout_b()))>;
template <class Type, int S>
using Layout_MN_SW128_SpAtom = ComposedLayout<Swizzle<3,4,3>, smem_sparse_ptr_flag_bits<S,sizeof_bits_v<Type>>,
                                              decltype(blocked_product(Layout<Shape<_1,Int<S>>>{}, Layout_MN_SW128_Atom<Type>{}.layout_b()))>;
template <class Type, int S>
using Layout_K_INTER_SpAtom = ComposedLayout<Swizzle<0,4,3>, smem_sparse_ptr_flag_bits<S,sizeof_bits_v<Type>>,
                                             decltype(blocked_product(Layout<Shape<_1,Int<S>>>{}, Layout_K_INTER_Atom<Type>{}.layout_b()))>;
template <class Type, int S>
using Layout_K_SW32_SpAtom  = ComposedLayout<Swizzle<1,4,3>, smem_sparse_ptr_flag_bits<S,sizeof_bits_v<Type>>,
                                             decltype(blocked_product(Layout<Shape<_1,Int<S>>>{}, Layout_K_SW32_Atom<Type>{}.layout_b()))>;
template <class Type, int S>
using Layout_K_SW64_SpAtom  = ComposedLayout<Swizzle<2,4,3>, smem_sparse_ptr_flag_bits<S,sizeof_bits_v<Type>>,
                                             decltype(blocked_product(Layout<Shape<_1,Int<S>>>{}, Layout_K_SW64_Atom<Type>{}.layout_b()))>;
template <class Type, int S>
using Layout_K_SW128_SpAtom = ComposedLayout<Swizzle<3,4,3>, smem_sparse_ptr_flag_bits<S,sizeof_bits_v<Type>>,
                                             decltype(blocked_product(Layout<Shape<_1,Int<S>>>{}, Layout_K_SW128_Atom<Type>{}.layout_b()))>;

} // namespace cute::SM90::GMMA

// 上游 `mma_traits_sm90_gmma.hpp` 在 `namespace cute` 里 `using namespace SM90;`
// —— 这就是 `cute::GMMA::…` 能用的原因。原头被裁后必须继续提供这条可见性。
namespace cute {
using namespace SM90;
} // namespace cute

////////////////////////////////////////////////////////////////////////////////
// 三个选路函数(逐字搬自 gemm/epilogue 的 sm90 builder 面)
////////////////////////////////////////////////////////////////////////////////
namespace cutlass::gemm::collective::detail {

/// cluster shape → TMA 原子(搬自 `gemm/collective/builders/sm90_common.inl`)
template <class UnimodalClusterShape>
constexpr auto
sm90_cluster_shape_to_tma_atom(UnimodalClusterShape) {
  static_assert(cute::rank(UnimodalClusterShape{}) == 1,
    "Use this function to figure out TMA for each mode individually.");

  if constexpr (cute::size(UnimodalClusterShape{}) == 1) {
    return cute::SM90_TMA_LOAD{};
  }
  else {
    return cute::SM90_TMA_LOAD_MULTICAST{};
  }
}

/// SS GMMA smem 布局选路(搬自 `gemm/collective/builders/sm90_common.inl`)
template <cute::GMMA::Major major, class ElementType, class BLK_MN, class BLK_K>
CUTE_HOST_DEVICE constexpr
auto
ss_smem_selector()
{
  using namespace cute;

  auto BLK_MN0 = size<0>(BLK_MN{});
  auto BLK_K0  = size<0>(BLK_K{});

  static_assert(BLK_MN0 % 8 == 0, "BLK_MN0 must be a multiple of 8.");
  static_assert(BLK_K0 % 8 == 0,  "BLK_K0 must be a multiple of 8.");

  if constexpr (major == GMMA::Major::MN) {
    if constexpr (BLK_MN0 % size<0>(GMMA::Layout_MN_SW128_Atom<ElementType>{}) == 0) {
      return GMMA::Layout_MN_SW128_Atom<ElementType>{};
    }
    else if constexpr (BLK_MN0 % size<0>(GMMA::Layout_MN_SW64_Atom<ElementType>{}) == 0) {
      return GMMA::Layout_MN_SW64_Atom<ElementType>{};
    }
    else if constexpr (BLK_MN0 % size<0>(GMMA::Layout_MN_SW32_Atom<ElementType>{}) == 0) {
      return GMMA::Layout_MN_SW32_Atom<ElementType>{};
    }
    else if constexpr (size(BLK_MN{}) % size<0>(GMMA::Layout_MN_INTER_Atom<ElementType>{}) == 0) {
      return GMMA::Layout_MN_INTER_Atom<ElementType>{};
    }
    else {
      static_assert(size(BLK_MN{}) % size<0>(GMMA::Layout_MN_INTER_Atom<ElementType>{}) == 0,
                    "BLK_MN must be a multiple of size<0>(GMMA::Layout_MN_INTER_Atom<ElementType>{})");
    }
  }
  else if constexpr (major == GMMA::Major::K) {
    if constexpr (BLK_K0 % size<1>(GMMA::Layout_K_SW128_Atom<ElementType>{}) == 0) {
      return GMMA::Layout_K_SW128_Atom<ElementType>{};
    }
    else if constexpr (BLK_K0 % size<1>(GMMA::Layout_K_SW64_Atom<ElementType>{}) == 0) {
      return GMMA::Layout_K_SW64_Atom<ElementType>{};
    }
    else if constexpr (BLK_K0 % size<1>(GMMA::Layout_K_SW32_Atom<ElementType>{}) == 0) {
      return GMMA::Layout_K_SW32_Atom<ElementType>{};
    }
    else if constexpr (BLK_K0 % size<1>(GMMA::Layout_K_INTER_Atom<ElementType>{}) == 0) {
      return GMMA::Layout_K_INTER_Atom<ElementType>{};
    }
    else {
      static_assert(BLK_K0 % size<1>(GMMA::Layout_K_INTER_Atom<ElementType>{}) == 0,
                    "BLK_K0 must be a multiple of size<1>(GMMA::Layout_K_INTER_Atom<ElementType>{})");
    }
  }
}

} // namespace cutlass::gemm::collective::detail

namespace cutlass::epilogue::collective::detail {

/// C/D 的 smem 布局原子(搬自 `epilogue/collective/builders/sm90_builder.inl`)
/// ⚠️ 上游写的是 `cutlass::gemm::detail::is_major`（住 `cutlass/gemm/gemm.h`，
/// 而那个头是 2.x 家族的门面）—— 同义实现就在 `cutlass::detail::is_major`
/// （`cutlass/detail/layout.hpp`），两者逐字等价（前者直接转发后者）。
template <class GmemStrideType, class Element, class EpilogueTile_MN>
constexpr auto
sm90_get_epilogue_smem_swizzle_layout_atom() {
  using namespace cute;

  // ColMajor C/D (M-major)
  if constexpr (cutlass::detail::is_major<0>(GmemStrideType{})) {
    return cutlass::gemm::collective::detail::ss_smem_selector<
      cute::GMMA::Major::MN, Element, decltype(get<0>(EpilogueTile_MN{})), decltype(get<1>(EpilogueTile_MN{}))
    >();
  }
  // RowMajor C/D (N-major)
  else if constexpr (cutlass::detail::is_major<1>(GmemStrideType{})) {
    return cutlass::gemm::collective::detail::ss_smem_selector<
      cute::GMMA::Major::K , Element, decltype(get<0>(EpilogueTile_MN{})), decltype(get<1>(EpilogueTile_MN{}))
    >();
  }
  else {
    static_assert(cutlass::detail::dependent_false<GmemStrideType>, "Unsupported gmem layout.");
  }
}

} // namespace cutlass::epilogue::collective::detail

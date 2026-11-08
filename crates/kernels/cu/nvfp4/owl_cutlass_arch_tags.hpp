// ============================================================================
// owl **arch 标签 shim** —— 从 CUTLASS `cutlass/arch/mma.h` 抄出来的那几个
// “操作类别”空标签（纯声明，无实现）。
// ============================================================================
//
// ## 为什么要它（正向增量的做法）
//
// `cutlass/arch/mma.h` 是 2.x 的**算子族总入口**：它定义 12 个 `OpClass*` 标签，
// 但更要命的是它在文件末尾 `#include` 了整棵 2.x arch 树
// （`mma_sm50/60/61/70/75/80/89/90.h` + `mma_sparse_sm80/89.h` + `wmma*.h`），
// 其中 `mma_sm50.h → quaternion.h → matrix.h`（**355 KB**）。
// 我们真正用到的只有**两个空 struct**：
//
//   - `OpClassTensorOp`             —— epilogue collective builder 的算子类别
//   - `OpClassBlockScaledTensorOp`  —— blockscaled 主循环的算子类别
//
// 所以按“用到什么就抄什么”的规矩：把这两个（外加同族的几个同类标签，成本为零）
// 抄成一个 owl 自己的头，然后把 `cutlass/arch/mma.h` 及其 2.x 兄弟头整批删掉。
//
// ## 纪律
// - **必须先于任何 `cutlass/…` collective/ epilogue builder 头 include**
//   （它们把这两个标签用作**非依赖名**模板实参，声明晚一步就是硬编译错）。
// - 只放标签，**不要**在这里补任何 `Mma<…>` 算子模板 —— 我们的算子面是
//   **cute atom**（`cute/arch/mma_sm120.hpp`），2.x 的 `arch::Mma` 一个都没用。
// ============================================================================
#pragma once

namespace cutlass {
namespace arch {

/// 线程级（SIMT）算子
struct OpClassSimt {};
/// 张量核算子（稠密）
struct OpClassTensorOp {};
/// WMMA 张量核算子
struct OpClassWmmaTensorOp {};
/// 张量核 + 结构化稀疏
struct OpClassSparseTensorOp {};
/// 张量核 + **块缩放**（NVFP4 的算子类别）
struct OpClassBlockScaledTensorOp {};
/// 张量核 + 块缩放 + 结构化稀疏
struct OpClassBlockScaledSparseTensorOp {};

}  // namespace arch
}  // namespace cutlass

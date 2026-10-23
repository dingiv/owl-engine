// owl × vLLM FA2 的 **host 编译垫片**（不是算子语义）。
//
// vLLM vendored 的 `flash_attn/src/cuda_check.h` 为了同时服务 PyTorch
// extension 与 standalone 构建，引用了 c10 的 CUDA 异常宏。owl 是纯 CUDA
// C ABI，不链 torch；这里只提供同名的“不吞错”宏：
//   - C10_CUDA_CHECK 保存最后一次 CUDA 错误，由 C ABI 返回；
//   - C10_CUDA_KERNEL_LAUNCH_CHECK 不做额外动作，让调用方 cudaGetLastError。
// 算子核体仍 100% 来自 vLLM vendored FlashAttention，无算法改动。
#pragma once
#include <cuda_runtime.h>

#define C10_CUDA_CHECK(EXPR)                                                   \
  do {                                                                        \
    cudaError_t _owl_cuda_err = (EXPR);                                        \
    (void)_owl_cuda_err;                                                       \
  } while (0)

#define C10_CUDA_KERNEL_LAUNCH_CHECK() ((void)0)

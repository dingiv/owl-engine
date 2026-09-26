// torch/headeronly/util/Exception.h 的最小 shim —— 仅提供 scalar_type.hpp 用到的
// STD_TORCH_CHECK。marlin-ffi 全链零 torch 头文件依赖。
#pragma once

#include <cstdio>
#include <cstdlib>

#ifndef STD_TORCH_CHECK
#define STD_TORCH_CHECK(cond, ...)                                       \
  do {                                                                   \
    if (!(cond)) {                                                       \
      std::fprintf(stderr, "marlin-ffi scalar_type check failed: %s\n",  \
                   #cond);                                               \
      std::abort();                                                      \
    }                                                                    \
  } while (0)
#endif

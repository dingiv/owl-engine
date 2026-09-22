// owl K1 port: vendored from attention.rs rev c0f19f2 (A4 所有权转移,2026-09-22)
// 裁剪:dtype_fp8.cuh 不 port(FP8 KV 归 marlin-ffi 路线,立项文档 §二·2)
#pragma once

#include "attention_generic.cuh"
#include "dtype_float16.cuh"
#include "dtype_float32.cuh"
#include "dtype_bfloat16.cuh"

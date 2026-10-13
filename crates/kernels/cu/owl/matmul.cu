// owl 语义算子族:**matmul f32 手写核**(2026-10-13 C 轮自 cu/owl/ops_pair.cu
// 按**算子族**拆出 —— 一算子一 `.cu`,与 ops/elems.rs 的 `owl_matmul*_f32`
// 声明对望)。
//
// f16 基线 matmul 走 cuBLAS foreign 通道(eval dtype 路由),不养手写变体;
// 本文件 = 单元锚保留(host f32 参考对拍)。
// 槽序契约:三 T + i32 m/k/n;输出块**非末参**(二维 grid 行主序,契约 4 例外)。

// ---- matmul f32 手写核(单元锚保留;[m,k]×[k,n] 行主序,grid 二维)----
// f16 基线 matmul 走 cuBLAS foreign 通道(eval dtype 路由),不养手写变体。
extern "C" __global__ void owl_matmul_f32(
    const float* a, const float* b, float* out,
    const int m, const int k, const int n) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;
    if (r < m && c < n) {
        float acc = 0.0f;
        for (int p = 0; p < k; p++) {
            acc += a[(size_t)r * k + p] * b[(size_t)p * n + c];
        }
        out[(size_t)r * n + c] = acc;
    }
}

// nt 变体:B 按 [n, k] 行主序直读(lm_head/tied embedding 形态;
// 权重保持 checkpoint 原布局,免 host 转置与第二份显存)。
extern "C" __global__ void owl_matmul_nt_f32(
    const float* a, const float* b, float* out,
    const int m, const int k, const int n) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;
    if (r < m && c < n) {
        const float* brow = b + (size_t)c * k;
        float acc = 0.0f;
        for (int p = 0; p < k; p++) {
            acc += a[(size_t)r * k + p] * brow[p];
        }
        out[(size_t)r * n + c] = acc;
    }
}

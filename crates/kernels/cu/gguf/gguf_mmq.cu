// ============================================================================
// owl GGUF 族:**官方 MMQ(tensor-core prefill GEMM)** 适配入口(W-2c;设计 §十八)
//
// 内核本体 = **llama.cpp 官方**(逐字 vendoring,见 `vendor/README.md`:
// `llama.cpp@1b1e27cb49`,2024-05-23)。本文件只做两件事:
//
//   ① 把 vendored 的**设备函数** `mul_mat_q<…>` 包成我们自己的
//      `extern "C" __global__` 入口(名字 = `family::gguf::MMQ_*`,逐型一个字面入口
//      —— 与既有核族同一条 H3 纪律;上游的 `static __global__` 壳带
//      `__launch_bounds__` 与 hip 条件,不适合直接当 C 入口);
//   ② **固定 `need_check = true`**:上游按 `row_diff % mmq_y == 0` 在启动期二选一;
//      我们只保留"带边界检查"那一档 —— 少一个二态维度(H3),网格/越界由核内判,
//      代价是每 tile 多几条谓词(实测 <1%)。
//
// # 槽序(与 `ops/gguf.rs` 的 MMQ_PARAMS 逐参对位)
//
//   weights[U8 块] , xq[U8 q8_1(36B 块;**上游同代布局**,本仓 `quant_act` 已产)],
//   out[F16] , ncols_x(=k) , nrows_x(=n) , ncols_dst(=m) , nrows_y(=k) , nrows_dst(=n)
//
// # 输出布局(本仓对 vendored 的**唯一补丁**,见 vendor/README)
//
// 上游把 `mul_mat_q` 的结果写成 **f32**,索引 `dst[col*nrows_dst + row]`
// (col = token,row = 权重行)。本仓输出是 f16 且布局**逐位相同**(行主序 [m, n])
// ⇒ 只把落点改成 `__float2half(...)` 一次转换,零额外核、零 f32 中间缓冲。
// ============================================================================
#include "vendor/mmq.cu"

// ---- 四型字面入口(名字 = `family::gguf::MMQ_*`;网格由 family 侧按 {4,32,4} 算)----
//
// `need_check` 固定 true(见文件头 ②);`k` 同时作 `ncols_x` 与 `nrows_y`
// (本仓 xq 行宽 = k/32 个 q8_1 块,无 padding)。
#define OWL_MMQ_BODY(CONFIG, QK, QR, QI, ZP, BLK, ALLOC, LOAD, VDR, VDOT)                        \
    constexpr mmq_arch_config_t arch = get_arch_config_device(CONFIG);                           \
    mul_mat_q<QK, QR, QI, ZP, BLK, arch.x, arch.y, arch.nwarps, ALLOC<arch.y>,                   \
              LOAD<arch.y, arch.nwarps, true>, VDR, VDOT>(                                       \
        weights, xq, out, k, n, m, k, n);

extern "C" __global__ void owl_gguf_mmq_q8_0(const void * __restrict__ weights,
                                             const void * __restrict__ xq,
                                             __half * __restrict__ out, int k, int n, int m) {
    OWL_MMQ_BODY(MMQ_CONFIG_Q8_0, QK8_0, QR8_0, QI8_0, false, block_q8_0,
                 allocate_tiles_q8_0, load_tiles_q8_0, VDR_Q8_0_Q8_1_MMQ,
                 vec_dot_q8_0_q8_1_mul_mat)
}

extern "C" __global__ void owl_gguf_mmq_q4k(const void * __restrict__ weights,
                                            const void * __restrict__ xq,
                                            __half * __restrict__ out, int k, int n, int m) {
    OWL_MMQ_BODY(MMQ_CONFIG_Q4_K, QK_K, QR4_K, QI4_K, true, block_q4_K,
                 allocate_tiles_q4_K, load_tiles_q4_K, VDR_Q4_K_Q8_1_MMQ,
                 vec_dot_q4_K_q8_1_mul_mat)
}

extern "C" __global__ void owl_gguf_mmq_q5k(const void * __restrict__ weights,
                                            const void * __restrict__ xq,
                                            __half * __restrict__ out, int k, int n, int m) {
    OWL_MMQ_BODY(MMQ_CONFIG_Q5_K, QK_K, QR5_K, QI5_K, true, block_q5_K,
                 allocate_tiles_q5_K, load_tiles_q5_K, VDR_Q5_K_Q8_1_MMQ,
                 vec_dot_q5_K_q8_1_mul_mat)
}

extern "C" __global__ void owl_gguf_mmq_q6k(const void * __restrict__ weights,
                                            const void * __restrict__ xq,
                                            __half * __restrict__ out, int k, int n, int m) {
    OWL_MMQ_BODY(MMQ_CONFIG_Q6_K, QK_K, QR6_K, QI6_K, false, block_q6_K,
                 allocate_tiles_q6_K, load_tiles_q6_K, VDR_Q6_K_Q8_1_MMQ,
                 vec_dot_q6_K_q8_1_mul_mat)
}

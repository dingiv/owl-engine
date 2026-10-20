// ============================================================================
// owl GGUF 族:**量化表的行解量化**(embedding 查表;设计 §四 L3 `owl_gguf_dequant_rows`)
//
// 为什么必需:两个靶位的 `token_embd.weight` 是**量化块**(4B = Q6_K、
// 27B = Q4_K)、既不是 f16 表也不可解量化驻留(设计 I-3)⇒ 词表查表必须
// **自带解量化**。`gguf.cuh` 只有逐块 `dequantize_block_*`(协作式:块内线程
// 按 `threadIdx.x` 分工写整个块)⇒ 本核的几何必须**与块几何对齐**:
//
//   `grid = (tokens, k / qk)`、`block = (threads_of_ty, 1, 1)`
//   线程数由**型**定(Q4_K/Q5_K = 32,Q6_K = 64;见 `family::gguf::rows_geom`)
//
// 即:**一个 CTA 解一个块**、`blockIdx.x = token`、`blockIdx.y = 行内块号`;
// 与 llama.cpp `convert.cu::dequantize_row_*_cuda`(`<<<nb, threads>>>` +
// 核内 `i = blockIdx.x`)是同一构造,只是多一维 token。
//
// # 与 GEMM 臂的分工
//
//   `owl_gguf_gemm_*`  : dequant-in-kernel 点积(**不解出全行**,MMVQ)
//   `owl_gguf_dequant_rows_*` : **解出全行**(查表;每 token 一行 f16)
//
// # 两条纪律
//
//   1. **H1/H2 零分配零查询**:纯读(`w`/`ids` 入、`out` 出),无 `cudaMalloc*`、
//      无 `cudaGetDeviceProperties`(几何由调用方按型给 ⇒ 核内零分支);
//   2. **越界索引的处置**:`id >= vocab` 时**写 0**(确定性、响亮可见),
//      不读越界内存 —— 调用方(引擎)的 id 来自本模型词表,越界即上游 bug,
//      这里不做静默丢弃(写 0 会在 logits 上立即可见)。
//
// 写法:**共享 `__device__` 模板体 + 逐型字面 wrapper**(与 `gguf_gemm.cu` 同构;
// 块解量化函数是模板 ⇒ 用 tag 结构体做静态分发,零虚表零分支)。
// ============================================================================
#include "gguf.cuh"

/// 型 → 块解量化(静态分发;每个特化只做一件事:把块指针交给 `gguf.cuh` 的函数)
template <int qk, typename block_q_t>
struct GgufRowDequant;

template <>
struct GgufRowDequant<QK_K, block_q4_K> {
    static __device__ __forceinline__ void run(const void * __restrict__ vx, __half * __restrict__ yy) {
        dequantize_block_q4_K<__half>(vx, yy);
    }
};

template <>
struct GgufRowDequant<QK_K, block_q5_K> {
    static __device__ __forceinline__ void run(const void * __restrict__ vx, __half * __restrict__ yy) {
        dequantize_block_q5_K<__half>(vx, yy);
    }
};

template <>
struct GgufRowDequant<QK_K, block_q6_K> {
    static __device__ __forceinline__ void run(const void * __restrict__ vx, __half * __restrict__ yy) {
        dequantize_block_q6_K<__half>(vx, yy);
    }
};


// ---- IQ 族(2026-10-20;REQ-GGUF-04):查表臂用的块解量化(32 线程协作)----
template <>
struct GgufRowDequant<QK_IQ4_NL, block_iq4_nl> {
    static __device__ __forceinline__ void run(const void * __restrict__ vx, __half * __restrict__ yy) {
        dequantize_block_iq4_nl<__half>(vx, yy);
    }
};

template <>
struct GgufRowDequant<QK_K, block_iq4_xs> {
    static __device__ __forceinline__ void run(const void * __restrict__ vx, __half * __restrict__ yy) {
        dequantize_block_iq4_xs<__half>(vx, yy);
    }
};

template <>
struct GgufRowDequant<QK_K, block_iq3_xxs> {
    static __device__ __forceinline__ void run(const void * __restrict__ vx, __half * __restrict__ yy) {
        dequantize_block_iq3_xxs<__half>(vx, yy);
    }
};

template <>
struct GgufRowDequant<QK_K, block_iq3_s> {
    static __device__ __forceinline__ void run(const void * __restrict__ vx, __half * __restrict__ yy) {
        dequantize_block_iq3_s<__half>(vx, yy);
    }
};


// ---- 27B UD 档补六型(2026-10-20)----
template <>
struct GgufRowDequant<QK_K, block_q2_K> {
    static __device__ __forceinline__ void run(const void * __restrict__ vx, __half * __restrict__ yy) {
        dequantize_block_q2_K<__half>(vx, yy);
    }
};

template <>
struct GgufRowDequant<QK_K, block_q3_K> {
    static __device__ __forceinline__ void run(const void * __restrict__ vx, __half * __restrict__ yy) {
        dequantize_block_q3_K<__half>(vx, yy);
    }
};

template <>
struct GgufRowDequant<QK_K, block_iq2_xxs> {
    static __device__ __forceinline__ void run(const void * __restrict__ vx, __half * __restrict__ yy) {
        dequantize_block_iq2_xxs<__half>(vx, yy);
    }
};

template <>
struct GgufRowDequant<QK_K, block_iq2_xs> {
    static __device__ __forceinline__ void run(const void * __restrict__ vx, __half * __restrict__ yy) {
        dequantize_block_iq2_xs<__half>(vx, yy);
    }
};

template <>
struct GgufRowDequant<QK_K, block_iq2_s> {
    static __device__ __forceinline__ void run(const void * __restrict__ vx, __half * __restrict__ yy) {
        dequantize_block_iq2_s<__half>(vx, yy);
    }
};

template <>
struct GgufRowDequant<QK_K, block_iq1_s> {
    static __device__ __forceinline__ void run(const void * __restrict__ vx, __half * __restrict__ yy) {
        dequantize_block_iq1_s<__half>(vx, yy);
    }
};

/// 共享体:`blockIdx.x = token`、`blockIdx.y = 行内块号`
///
/// 入参
///   w     : 量化表(块字节;行 = 词表项,**行跨 `k / qk` 块**)
///   ids   : 每个 token 的词表 id(f32 承载设备侧索引;与 `owl_embed_*` 同约定)
///   out   : [tokens, k] f16
///   k     : 行宽(元素数;必须被 qk 整除 —— 由 `rows_geom` 的调用方保证)
///   rows  : token 数(= out 行数)
///   vocab : 表行数(越界判据;`id >= vocab` ⇒ 该块写 0)
template <int qk, typename block_q_t>
__device__ __forceinline__ void owl_gguf_dequant_rows_body(
    const void * __restrict__ w, const float * __restrict__ ids,
    __half * __restrict__ out, int k, int rows, int vocab) {
    const int row = blockIdx.x;
    const int b = blockIdx.y;
    if (row >= rows) {
        return;
    }
    const int nb = k / qk;
    if (b >= nb) {
        return;
    }
    __half * orow = out + (size_t) row * (size_t) k + (size_t) b * (size_t) qk;
    const unsigned int id = (unsigned int) ids[row];
    if (id >= (unsigned int) vocab) {
        // 越界 id:确定性写 0(不读越界内存;见头注 §纪律 2)
        for (int i = threadIdx.x; i < qk; i += blockDim.x) {
            orow[i] = __float2half(0.0f);
        }
        return;
    }
    const block_q_t * wrow = reinterpret_cast<const block_q_t *>(w) + (size_t) id * (size_t) nb;
    GgufRowDequant<qk, block_q_t>::run(reinterpret_cast<const void *>(wrow + b), orow);
}

// ---- M1 实例集(三型;设计 REQ-GGUF-04 + §11.5 ②):加型 = 加一个 wrapper
//      + `family::gguf::rows_entry_for` 加一项 + `ops/gguf.rs` 加一条声明 ----

extern "C" __global__ void owl_gguf_dequant_rows_q4k(
    const void * __restrict__ w, const float * __restrict__ ids,
    __half * __restrict__ out, int k, int rows, int vocab) {
    owl_gguf_dequant_rows_body<QK_K, block_q4_K>(w, ids, out, k, rows, vocab);
}

extern "C" __global__ void owl_gguf_dequant_rows_q5k(
    const void * __restrict__ w, const float * __restrict__ ids,
    __half * __restrict__ out, int k, int rows, int vocab) {
    owl_gguf_dequant_rows_body<QK_K, block_q5_K>(w, ids, out, k, rows, vocab);
}

extern "C" __global__ void owl_gguf_dequant_rows_q6k(
    const void * __restrict__ w, const float * __restrict__ ids,
    __half * __restrict__ out, int k, int rows, int vocab) {
    owl_gguf_dequant_rows_body<QK_K, block_q6_K>(w, ids, out, k, rows, vocab);
}

extern "C" __global__ void owl_gguf_dequant_rows_iq4nl(
    const void * __restrict__ w, const float * __restrict__ ids,
    __half * __restrict__ out, int k, int rows, int vocab) {
    owl_gguf_dequant_rows_body<QK_IQ4_NL, block_iq4_nl>(w, ids, out, k, rows, vocab);
}

extern "C" __global__ void owl_gguf_dequant_rows_iq4xs(
    const void * __restrict__ w, const float * __restrict__ ids,
    __half * __restrict__ out, int k, int rows, int vocab) {
    owl_gguf_dequant_rows_body<QK_K, block_iq4_xs>(w, ids, out, k, rows, vocab);
}

extern "C" __global__ void owl_gguf_dequant_rows_iq3xxs(
    const void * __restrict__ w, const float * __restrict__ ids,
    __half * __restrict__ out, int k, int rows, int vocab) {
    owl_gguf_dequant_rows_body<QK_K, block_iq3_xxs>(w, ids, out, k, rows, vocab);
}

extern "C" __global__ void owl_gguf_dequant_rows_iq3s(
    const void * __restrict__ w, const float * __restrict__ ids,
    __half * __restrict__ out, int k, int rows, int vocab) {
    owl_gguf_dequant_rows_body<QK_K, block_iq3_s>(w, ids, out, k, rows, vocab);
}

// ---- 27B UD 档补六型(2026-10-20)----

extern "C" __global__ void owl_gguf_dequant_rows_q2k(
    const void * __restrict__ w, const float * __restrict__ ids,
    __half * __restrict__ out, int k, int rows, int vocab) {
    owl_gguf_dequant_rows_body<QK_K, block_q2_K>(w, ids, out, k, rows, vocab);
}

extern "C" __global__ void owl_gguf_dequant_rows_q3k(
    const void * __restrict__ w, const float * __restrict__ ids,
    __half * __restrict__ out, int k, int rows, int vocab) {
    owl_gguf_dequant_rows_body<QK_K, block_q3_K>(w, ids, out, k, rows, vocab);
}

extern "C" __global__ void owl_gguf_dequant_rows_iq2xxs(
    const void * __restrict__ w, const float * __restrict__ ids,
    __half * __restrict__ out, int k, int rows, int vocab) {
    owl_gguf_dequant_rows_body<QK_K, block_iq2_xxs>(w, ids, out, k, rows, vocab);
}

extern "C" __global__ void owl_gguf_dequant_rows_iq2xs(
    const void * __restrict__ w, const float * __restrict__ ids,
    __half * __restrict__ out, int k, int rows, int vocab) {
    owl_gguf_dequant_rows_body<QK_K, block_iq2_xs>(w, ids, out, k, rows, vocab);
}

extern "C" __global__ void owl_gguf_dequant_rows_iq2s(
    const void * __restrict__ w, const float * __restrict__ ids,
    __half * __restrict__ out, int k, int rows, int vocab) {
    owl_gguf_dequant_rows_body<QK_K, block_iq2_s>(w, ids, out, k, rows, vocab);
}

extern "C" __global__ void owl_gguf_dequant_rows_iq1s(
    const void * __restrict__ w, const float * __restrict__ ids,
    __half * __restrict__ out, int k, int rows, int vocab) {
    owl_gguf_dequant_rows_body<QK_K, block_iq1_s>(w, ids, out, k, rows, vocab);
}

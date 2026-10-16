// owl kernels/cu/load/marlin_repack_ct_a8.cu —— ct packed → marlin B **a8** 设备重排
// (2026-10-17;S2 补:与 a16 版 `marlin_repack_ct.cu` **同输入契约、同输出字数**,
//  但 **tile 几何与 nibble 排布不同** —— 核侧 a8 的 B 读法与 a16 不同)
//
// 参照实现(逐位对齐,非凭图反推)= `family/marlin/repack.rs::pack_marlin_b_a8`:
// 该函数已由 W4A8 真机对拍验证(a8 布局喂 a16 表会错列、a16 布局喂 a8 表也错
// —— 见设计卷 §14.1/§14.3 的判别探针与负例)。
//
// 输入契约(与 a16 版逐字同):ct packed `[rows=out, cols=k/8]` u32,
//   word(o, j) nibble t = (k = 8j + t, out = o)。
// 输出:每 **k-tile(32 k)** 段内,n 方向按 **32 列一 tile**,每 tile **128 word**
//   ⇒ 总字数 = (k/32)·(n/32)·128 = **k·n/8**(与 a16 版同额,排布不同)。
//   段内 word 的 (nibble r → k_local / n_local) 映射:
//     warp = word%4, th = word/4, tc_row = (th%4)*4
//     n_local = (warp/2)*16 + th/4 + (warp%2)*8
//     k_local = tc_row + r/2 + 16*(r%2)
//
// 几何:grid `(rows/32, cols/4)`,block **128**(= 128 word/tile)。
// 约束:`rows%32==0 && cols%4==0`(装载侧 marlin_eligible 已保证 out%256 / k%128)。
// shared:`sh[32][33]` u8 = 1056B(k_local 行、o_local 列,33 列 pad 防 bank conflict)。
//
// nvrtc 独立编译:无 stdint include(类型自带;与宿主 ABI = 4/1 字节无符)
typedef unsigned int uint32_t;
typedef unsigned char uint8_t;

// 槽序契约:in, rows, cols, out(输出块末参,owl registry 惯例)
extern "C" __global__ void owl_ct_repack_a8_u32(
    const uint32_t* __restrict__ in,  // ct packed [rows, cols] u32(行 = out,列 = k/8)
    size_t rows,                      // out(总行)
    size_t cols,                      // k/8(总 u32 列)
    uint32_t* __restrict__ out) {     // marlin B a8 [(k/32)·(rows/32)·128] u32
    const size_t nt = blockIdx.x;     // n-tile 基(32 out)
    const size_t kt = blockIdx.y;     // k-tile 基(32 k)
    const int t = threadIdx.x;

    __shared__ uint8_t sh[32][33];    // sh[k_local][o_local]

    // ---- unpack:128 线程 × 1 word(o_local = t/4,jl = t%4)----
    {
        const int o_local = t >> 2;
        const int jl = t & 3;
        const uint32_t v = in[(size_t)(nt * 32 + (size_t)o_local) * cols + kt * 4 + (size_t)jl];
#pragma unroll
        for (int t8 = 0; t8 < 8; t8++) {
            sh[jl * 8 + t8][o_local] = (uint8_t)((v >> (4 * t8)) & 0xF);
        }
    }
    __syncthreads();

    // ---- repack:每线程一个 word(与 pack_marlin_b_a8 逐位同映射)----
    const int warp = t & 3;
    const int th = t >> 2;
    const int tc_row = (th % 4) * 4;
    const int n_local = (warp >> 1) * 16 + (th >> 2) + (warp & 1) * 8;

    uint32_t v = 0;
#pragma unroll
    for (int r = 0; r < 8; r++) {
        const int k_local = tc_row + (r >> 1) + 16 * (r & 1);
        v |= ((uint32_t)sh[k_local][n_local]) << (4 * r);
    }
    const size_t words_per_ktile = (rows / 32) * 128;
    out[kt * words_per_ktile + nt * 128 + (size_t)t] = v;
}

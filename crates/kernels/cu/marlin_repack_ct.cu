// owl kernels/cu/marlin_repack_ct.cu —— ct packed → marlin B 设备重排核
// (2026-10-01 装载提速;主源 = attention.rs marlin_repack.cu 的
// gptq_repack_kernel,extern "C" 零 torch 契约同款;Apache-2.0)
//
// 布局适配(与主源的唯一差异 = 输入侧转置):
// - 主源输入 GPTQ qweight [k/8, n]:word(r, c) = k 元素 8r..8r+8 @ 列 c
//   (行 = k_block,列 = out);
// - 本核输入 ct packed [rows=out, cols=k/8]:word(o, j) = k 元素 8j..8j+8
//   @ 行 o(行 = out,列 = k_block)—— 行列角色互换;
// - repack 输出段(marlin 16k×64n tile → 128 u32)与主源逐行一致:
//   block[bi][r][c] 语义固定 = (bi: n 的 16 列段, r: 16 连续 k,
//   c: 段内 16 n);unpack 段按 ct 语义填充:
//   word(o, j) nibble t = (k = 8j+t, out = o)
//   → 格 (bi = o_local/16, r = 8jl + t8, c = o_local%16),
//     源 word = (row + o_local, col + jl),o_local = wid/2, jl = wid%2。
//
// grid/边界:grid (rows/64, cols/2),block 32 线程;要求 rows%64==0 且
// cols%2==0(装载侧 marlin_eligible 已保证 out%256==0 / k%128==0)。
// shared 4×16×18 u8 = 1152B(18 = 16+2 pad 防 bank conflict,主源同款)。

// nvrtc 独立编译:无 stdint include(类型自带;与宿主 ABI = 4/1 字节无符)
typedef unsigned int uint32_t;
typedef unsigned char uint8_t;

// 槽序契约:in, rows, cols, out(输出块末参,owl registry 惯例)
extern "C" __global__ void owl_ct_repack_u32(
    const uint32_t* __restrict__ in,  // ct packed [rows, cols] i32(行 = out,列 = k/8)
    size_t rows,                      // out(总行)
    size_t cols,                      // k/8(总 u32 列)
    uint32_t* __restrict__ out) {     // marlin B [k/16, rows*2] u32
    uint32_t row = blockIdx.x * 64;   // out 块基(64 out/块)
    uint32_t col = blockIdx.y * 2;    // k_block 基(2 word = 16 k/块)
    int t = threadIdx.x;

    __shared__ uint8_t block[4][16][18];

    // ---- unpack:32 线程 × 4 word = 128 word(64 out × 2 j)----
    int wid = t;
    #pragma unroll
    for (int q = 0; q < 4; q++) {
        int o_local = wid / 2;        // 块内 out(0..63)
        int jl = wid % 2;             // word 列(0..1)
        int bi = o_local / 16;        // n 16 列段
        int c = o_local % 16;         // 段内 n
        uint32_t v = in[(row + o_local) * cols + col + jl];
        int r_base = 8 * jl;          // k = base_k + 8jl + t8
        #pragma unroll
        for (int t8 = 0; t8 < 8; t8++) {
            block[bi][r_base + t8][c] = (v >> (4 * t8)) & 0xF;
        }
        wid += 32;
    }
    __syncthreads();

    // ---- repack:marlin 16×64 tile → 128 u32(主源原样;idx = marlin
    //      tile 内 k 交织(0,8,1,9…)× n 双列)----
    uint32_t srow = (t % 4) * 2;
    uint32_t scol = t / 4;

    uint32_t idx[8][2];
    idx[0][0] = srow;     idx[0][1] = scol;
    idx[1][0] = srow + 8; idx[1][1] = scol;
    idx[2][0] = srow;     idx[2][1] = scol + 8;
    idx[3][0] = srow + 8; idx[3][1] = scol + 8;

    idx[4][0] = srow + 1; idx[4][1] = scol;
    idx[5][0] = srow + 9; idx[5][1] = scol;
    idx[6][0] = srow + 1; idx[6][1] = scol + 8;
    idx[7][0] = srow + 9; idx[7][1] = scol + 8;

    #pragma unroll
    for (int i = 0; i < 4; i += 1) {
        uint32_t v[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            v[j] = block[i][idx[j][0]][idx[j][1]];
        }
        uint32_t pack = (v[7] << 28) | (v[6] << 24) | (v[5] << 20) | (v[4] << 16) |
            (v[3] << 12) | (v[2] << 8) | (v[1] << 4) | v[0];
        // blockIdx.x = out 64-tile(读侧),blockIdx.y = k 16 段(写侧行)
        out[blockIdx.y * rows * 2 + blockIdx.x * 128 + t * 4 + i] = pack;
    }
}

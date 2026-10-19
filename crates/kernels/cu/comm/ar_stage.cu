// owl 通信算子族 —— allreduce **stage**(TP2 S2;协议不变量 I1/I3/I4/I5)。
//
// 出处:`docs/arch/TP2-施工设计.md` §3(布局 / 两核 / 不变量)。
// 与 Rust 侧 `owl_comm::ar` 的常量**逐字节一致**(改 = 双侧同改 + 门重跑):
//
//   header(64 B,两条 cache line;slot 数据从偏移 64 开始)
//     +0   u64 produced    本端已写入调用数(stage 核递增;**本核是唯一写者**)
//     +8   u64 consumed    本端已消费对端调用数(reduce 核递增;对端 stage 当 credit 读)
//     +16  u32 slot_bytes  slot 字节数(自描述)
//     +20  u32 n_slots     slot 数(v1 = 2)
//     +24  u32 magic       0x4F574C41 "OWLA"(串块判据)
//     +28  u32 err         错误码:0 正常 / 1 stage 超时 / 2 reduce 超时 / 3 header 校验失败
//     +32  u64 wait_done   多 block 两级等待令牌(**单调** = 本端 produced+1)
//     +40  u32 blocks_done 完成块计数(最后一块发布 produced)
//     +32..64             保留
//
// 语义:把本端 payload(`local`)第 `slab_off` 起 `bytes` 字节拷进本端 staging 的
// slot[produced % n_slots],系统 fence 后发布 produced++。
//
// 纪律(为什么这么写;每条都有设计出处):
// 1. **单 block**(grid = 1):produced 是**单写者游标** —— 多 block 要原子协调 +
//    "最后一块发布"协议;v1 单 block 循环拷贝(正确优先;S5 再谈流水)。
// 2. **每线程 __threadfence_system()**:发布律 = "数据系统可见 → 再写 flag"。
//    fence 只覆盖**本线程**的写 ⇒ 全线程各自 fence + `__syncthreads()` 之后,才由
//    thread0 写 produced(与 CUDA 编程指南 threadFenceReduction 同构;PCIe 语义上
//    `membar.sys` 是必要条件 —— I4)。
// 3. **credit 自旋(I1)**:`peer.consumed + n_slots > own.produced` 才可覆盖 slot
//    —— 否则会盖掉对端还没读走的数据;超时(5 s)置 err 并**放弃**(D4:宁可整轮
//    失败,不可静默出结果)。
// 4. **相对量(I3)**:两侧只做相对比较 + 设备内自增游标,不烘绝对值 ⇒ 图回放安全。
// 5. **失败路径不提前 return**(thread0 早退 = 其他线程卡死 `__syncthreads`):错误
//    写 shared 标志,全体一致退出。

// ── ★ **前置清单**(2026-10-17;照 NCCL `contrib/custom_algos` 的写法:核头逐条列,
//    误用要能"对着单子查",而不是散在各处)──────────────────────────────────
// ① **两 rank 同 plan**:`slot_bytes` / `n_slots` / `slab_off` 逐参一致(§3 不变量 I5,
//    序列同构的前提)。本核只校验**本端** header 与实参相符(不符 ⇒ err=3)。
// ② **header 必须先初始化**,且 `produced` / `consumed` / `wait_done` / `blocks_done`
//    **四格全 0**。⚠️ 其中 `wait_done` 最险:若留**非 0 残值**且 ≥ 本轮 want,
//    非 0 号 block 会**跳过两级等待** ⇒ 静默读到未就绪的对端数据(无报错)。
//    ⇒ 这就是本核加 `n_slots >= 2` 之外**唯一无法自检**的一条,必须在分配侧保证
//    (见 `ar.rs::hdr` 注释与 boot 期握手)。
// ③ **载荷 ≤ `slot_bytes`**(v1 单块):超了是调用方错,解释器层已响亮拒。
// ④ **`n_slots ≥ 1`**(2026-10-17 **放宽**;原 `≥ 2`,那道守卫把 CE 刀打死了 ——
//    见 TP2-施工设计 §29/§31):
//    · `n_slots == 0` = **没有格可写**(调用方错)⇒ 响亮拒(err=3);
//    · `n_slots == 1` = **严格一深环**:本核必须等 `peer.consumed ≥ produced`
//      (`need = prod − n_slots + 1 = prod`)才覆盖 slot 0。安全性由**两条现役机制**保证:
//      (a) **代发 credit**(见核内 `*own_cons = prod0`)写在 credit 检查**之前**
//          ⇒ 两侧"先发布后检查" ⇒ **不死锁**;
//      (b) **尾部 rendezvous**(发布 produced 后等 `peer.produced ≥ want`)
//          ⇒ 一步之内两端都完成 stage 才放行 ⇒ **单格不会被覆盖在两方之间**。
//      v1 语义下同样安全:credit = 对端 reduce 的**真消费**,等到 `prod` 即已读走 `prod` 次。
//      CE 语义下 `staging.consumed` 的**唯一写者**就是本核的代发(add 核绑到 inbox)。
// ⑤ **发射几何**:block 尺寸纯发射参数(核内全用 `blockDim.x` / `gridDim.x`);
//    grid 由 `server::comm::MAX_THREADS / BLOCK` 算 ⇒ 与核无耦合,故两 rank 之间
//    **不要求同 grid**(它只决定本端工作的摊法,不进协议)。
// ⑥ **超时**:credit 5s ⇒ err=1;数据 5s ⇒ err=2(D4:宁可整轮失败,不静默出结果)。

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>

#define OWL_AR_MAGIC 0x4F574C41u // "OWLA"(与 comm::ar::MAGIC 同值)
#define OWL_AR_HDR_BYTES 64u
#define OWL_AR_OFF_PRODUCED 0u
#define OWL_AR_OFF_CONSUMED 8u
#define OWL_AR_OFF_SLOT_BYTES 16u
#define OWL_AR_OFF_N_SLOTS 20u
#define OWL_AR_OFF_MAGIC 24u
#define OWL_AR_OFF_ERR 28u
#define OWL_AR_OFF_WAIT_DONE 32u
#define OWL_AR_OFF_BLOCKS_DONE 40u

/// 自旋超时(纳秒;设备全局时钟)。5 s —— 比任何正常等待都长,比"挂死"短得多。
#define OWL_AR_TIMEOUT_NS 5000000000ull

/// 紧密自旋次数上限(之后 `__nanosleep` 退避 —— 降 PCIe 读压力)
#define OWL_AR_SPIN_TIGHT 4096u

/// 设备全局时钟(%globaltimer,ns 级;sm_30+)
__device__ __forceinline__ unsigned long long owl_ar_timer_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

/// 字节块拷贝(16 B 对齐走 uint4;否则退化为逐字节)
///
/// ★ **2026-10-17 MLP 律**:主循环 **8 路手工展开 + 逐通道谓词**(先把 8 个 16 B 读
/// 全发出再写回)——— 与 reduce 同病同药:逐条 `load→store` 的在飞访存 = 1/线程,
/// 本地往返 ~0.3µs。
/// **单循环形态(无"余数慢路")**:原"展开主循环 + 逐条余数循环"在小 `n16`
/// (余数段占主体)落回逐条速度 —— 基准里那正是"小载荷比大载荷慢"的怪相来源
/// (`comm_ar_bench` 尺寸阶梯:20KB 慢于 128KB)。谓词展开后**所有尺寸同一条路**。
/// 语义逐字节不变(纯搬移,无算术)。
__device__ __forceinline__ void owl_ar_copy_bytes(
    unsigned char* __restrict__ dst,
    const unsigned char* __restrict__ src,
    const size_t bytes) {
    // **多 block**:全局线程索引 + 全局 stride(块内仍 16 B 连续 ⇒ 合并访问不变)
    const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    const unsigned long long da = reinterpret_cast<unsigned long long>(dst);
    const unsigned long long sa = reinterpret_cast<unsigned long long>(src);
    const bool aligned = ((da | sa) & 15ull) == 0ull;
    if (aligned) {
        const size_t n16 = bytes >> 4;
        uint4* d4 = reinterpret_cast<uint4*>(dst);
        const uint4* s4 = reinterpret_cast<const uint4*>(src);
        for (size_t i = tid; i < n16; i += 8 * stride) {
            uint4 vs[8];
            bool ok[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const size_t j = i + (size_t)k * stride;
                ok[k] = j < n16;
                if (ok[k]) {
                    vs[k] = s4[j];
                }
            }
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                if (ok[k]) {
                    d4[i + (size_t)k * stride] = vs[k];
                }
            }
        }
        const size_t done = n16 << 4;
        for (size_t j = done + tid; j < bytes; j += stride) {
            dst[j] = src[j];
        }
    } else {
        for (size_t i = tid; i < bytes; i += stride) {
            dst[i] = src[i];
        }
    }
}

// ---- **§47 第二步:fp8 载荷**(2026-10-19)----------------------------------
// 背景:§47/§53 已把"数值判据"与"载荷减半的收益"(+46.2%)两条都钉死;
// 本节把**真字节砍半**落到搬运面上:
//   读本端 f16/bf16 的 n 元素 → 按 **64 元素连续区间**分组算 e4m3 尺度 →
//   写对端 data 块 `[fp8 载荷 n B][尺度 ng × f16]`(共 1.031n B;
//   slot 仍按 2n 分配 ⇒ 装得下,`slot_bytes` 不动)。
//
// ⚠️ **分组必须按元素区间连续**(§47.5 的落地要点):第一步的"按线程映射分组"
//   收端**找不到尺度**(线程映射不同);区间化后收端按同一区间索引即可。
// ⚠️ **尺度公式与 §47 逐字同源**(`OWL_AR_E4M3_MAX = 448`、`scale = m/448`、
//   零组取 1.0)—— 两文件各一份(不跨文件 include:构建期增量指纹只覆盖 .cu 自身)。
// ⚠️ **对称律**(§47.1):本核只写 `q(本端)`;收端 reduce 侧对**自己那份**同式量化。
//    送出前量化一次 + 收端反量化 = 与 §47 的数值件同一对值。

// dtype → f32(与 `ar_reduce.cu` **逐字同式**;两文件各一份,不共享头)
template <typename T> __device__ __forceinline__ float owl_ar_to_float(T x);
template <> __device__ __forceinline__ float owl_ar_to_float<float>(float x) { return x; }
template <> __device__ __forceinline__ float owl_ar_to_float<__half>(__half x) { return __half2float(x); }
template <> __device__ __forceinline__ float owl_ar_to_float<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }

#define OWL_AR_E4M3_MAX 448.0f
#define OWL_AR_FP8_G 64

/// **fp8 载荷的尺度区起点**(`[fp8 n B][尺度 ng × f16]` 的第二段)
///
/// ⚠️ 必须**偶字节对齐**:`n` 为奇数时 `dst + n` 是奇地址 ⇒ `__half` 存取
/// 触发 `CUDA_ERROR_MISALIGNED_ADDRESS`(2026-10-19 G1b 的尾组用例抓获;
/// 旧布局 `dst + n` 在 n 奇数时非法,只是从没测过奇数 n)。
/// 本宏与 `ar_reduce.cu` 的同名宏**必须逐字一致**(两处 = 一个线格式)。
#define OWL_AR_FP8_SC_OFF(n) (((n) + 1u) & ~(size_t)1u)

/// 组尺度(全零组取 1.0 ⇒ 量化结果仍为 0;与 ar_reduce.cu 逐字同式)
__device__ __forceinline__ float owl_ar_qdq_scale_stage(float m) {
    return (m > 0.0f) ? (m / OWL_AR_E4M3_MAX) : 1.0f;
}

/// 非量化档的"类型化包装"(统一 XFER 宏签名:量化版与拷贝版同形)
template <typename T>
__device__ __forceinline__ void owl_ar_copy_bytes_t(unsigned char* __restrict__ dst,
                                                     const T* __restrict__ src,
                                                     const size_t bytes) {
    owl_ar_copy_bytes(dst, reinterpret_cast<const unsigned char*>(src), bytes);
}

/// 小端 4 字节打包(元素 e 落在 byte e)—— 供 16 B 向量写用
__device__ __forceinline__ unsigned int owl_ar_pack_u32(const unsigned char* p) {
    return (unsigned int)p[0] | ((unsigned int)p[1] << 8u) | ((unsigned int)p[2] << 16u) |
           ((unsigned int)p[3] << 24u);
}

/// fp8 量化拷贝(元素区间连续分组;shuffle 归约 ⇒ 合并访问)
///
/// **线程映射(2026-10-19 改)**:`sub = lane & 3` 为组内第几个 **16 元素片**,
/// `gsel = (lane>>2)&7` 为 warp 内第几组 ⇒ 一个 warp 同时干 **8 组**,
/// 每组 4 线程 × 16 元素 = 64 元素 ✓ 恰好 OWL_AR_FP8_G(与 reduce 侧同一索引)。
/// `__shfl_xor_sync(1/2)` 在 lane 低 2 位内闭合 ⇒ 归约只在 4 线程组内 ✓。
///
/// ★ **为什么必须 16 元素/lane**(2026-10-19 push×fp8 立案):一元素一字节写
/// ⇒ SASS 是 8 × `STG.E.U8`;本端槽时 L2 吸收无感,但 **push 档写对端**
/// (fp8 stage 写 peer 槽)时**每字节一个 PCIe 事务**:实测 push×fp8 的
/// stage 核 184 µs → **8418 µs**(46×)→ prefill 2321 → 711 t/s。
/// 改后:每 lane 一次 `uint4` 落 16 B(warp 内 8 组连续 ⇒ 512 B 连续),
/// 语义与逐字节写**逐位相同**(同一 `__nv_fp8_e4m3` 构造 + 同一 max 归约)。
/// 指针非 16 B 对齐时降级到标量路(只降速不改语义)。
template <typename T>
__device__ __forceinline__ void owl_ar_quant_fp8_vec(__nv_fp8_e4m3* __restrict__ q,
                                                     __half* __restrict__ sc,
                                                     const T* __restrict__ src,
                                                     const size_t n,
                                                     const size_t ng) {
    const int lane = (int)(threadIdx.x & 31u);
    const int sub = lane & 3;
    const int gsel = (lane >> 2) & 7;
    const size_t wid = ((size_t)blockIdx.x * blockDim.x + (size_t)threadIdx.x) >> 5;
    const size_t nwarp = ((size_t)gridDim.x * blockDim.x) >> 5;
    // ⚠️ 循环上界走**整 warp 的基准组** `g0`(不是每 lane 自己的 g):否则 ng 不是
    //    "每 warp 组数"整倍时 lane 间 trip count 不同 ⇒ `__shfl_xor_sync` 在**发散
    //    warp** 上死等(2026-10-19 G1b 奇数尾组抓获;旧映射同型,只是从未遇到非整倍)。
    for (size_t g0 = wid * 8u; g0 < ng; g0 += nwarp * 8u) {
        const size_t g = g0 + (size_t)gsel;
        const bool active = (g < ng);
        const size_t off = g * OWL_AR_FP8_G + (size_t)sub * 16u;
        float v[16];
        float m = 0.0f;
        const bool full = active && (off + 16u <= n);
        if (full) {
            const uint4 xa = *reinterpret_cast<const uint4*>(src + off); // 8 元素
            const uint4 xb = *reinterpret_cast<const uint4*>(src + off + 8u);
            const T* ha = reinterpret_cast<const T*>(&xa);
            const T* hb = reinterpret_cast<const T*>(&xb);
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                v[e] = owl_ar_to_float<T>(ha[e]);
                m = fmaxf(m, fabsf(v[e]));
            }
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                v[8 + e] = owl_ar_to_float<T>(hb[e]);
                m = fmaxf(m, fabsf(v[8 + e]));
            }
        } else if (active) {
            // 尾部不足一个 16 元素片:仍参与本组归约(补 0),写出时按 j < n 裁
#pragma unroll
            for (int e = 0; e < 16; ++e) {
                const size_t j = off + (size_t)e;
                v[e] = (j < n) ? owl_ar_to_float<T>(src[j]) : 0.0f;
                m = fmaxf(m, fabsf(v[e]));
            }
        } else {
#pragma unroll
            for (int e = 0; e < 16; ++e) {
                v[e] = 0.0f; // 空转 lane:不碰内存,但仍参与归约(否则 shuffle 发散)
            }
        }
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
        if (!active) {
            continue; // 归约已参加完;后面无 warp 级同步 ⇒ continue 安全
        }
        const float scale = owl_ar_qdq_scale_stage(m);
        const float inv = 1.0f / scale;
        if (sub == 0) {
            sc[g] = __float2half(scale);
        }
        if (full) {
            __nv_fp8_e4m3 q8[16];
#pragma unroll
            for (int e = 0; e < 16; ++e) {
                q8[e] = __nv_fp8_e4m3(v[e] * inv);
            }
            const unsigned char* by = reinterpret_cast<const unsigned char*>(q8);
            uint4 pack;
            pack.x = owl_ar_pack_u32(by);
            pack.y = owl_ar_pack_u32(by + 4);
            pack.z = owl_ar_pack_u32(by + 8);
            pack.w = owl_ar_pack_u32(by + 12);
            *reinterpret_cast<uint4*>(q + off) = pack;
        } else {
#pragma unroll
            for (int e = 0; e < 16; ++e) {
                const size_t j = off + (size_t)e;
                if (j < n) {
                    q[j] = __nv_fp8_e4m3(v[e] * inv);
                }
            }
        }
    }
}

/// **标量兜底路**(指针 16 B 未对齐;与向量路**逐位同构**,只降速不改语义)
/// —— 线程映射回到 `sub = lane & 7`(8 元素/lane),写出走单字节。
template <typename T>
__device__ __forceinline__ void owl_ar_quant_fp8_scalar(__nv_fp8_e4m3* __restrict__ q,
                                                        __half* __restrict__ sc,
                                                        const T* __restrict__ src,
                                                        const size_t n,
                                                        const size_t ng) {
    const int lane = (int)(threadIdx.x & 31u);
    const int sub = lane & 7;
    const int gsel = (lane >> 3) & 3;
    const size_t wid = ((size_t)blockIdx.x * blockDim.x + (size_t)threadIdx.x) >> 5;
    const size_t nwarp = ((size_t)gridDim.x * blockDim.x) >> 5;
    for (size_t g0 = wid * 4u; g0 < ng; g0 += nwarp * 4u) { // 上界按整 warp(见向量路注)
        const size_t g = g0 + (size_t)gsel;
        const bool active = (g < ng);
        const size_t off = g * OWL_AR_FP8_G + (size_t)sub * 8u;
        float v[8];
        float m = 0.0f;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const size_t j = off + (size_t)e;
            v[e] = (active && j < n) ? owl_ar_to_float<T>(src[j]) : 0.0f;
            m = fmaxf(m, fabsf(v[e]));
        }
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 4));
        if (!active) {
            continue;
        }
        const float scale = owl_ar_qdq_scale_stage(m);
        const float inv = 1.0f / scale;
        if (sub == 0) {
            sc[g] = __float2half(scale);
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const size_t j = off + (size_t)e;
            if (j < n) {
                q[j] = __nv_fp8_e4m3(v[e] * inv);
            }
        }
    }
}

/// **入口**:16 B 对齐 ⇒ 向量路;否则标量兜底(与 f16 档 `owl_ar_copy_bytes` 同律)
template <typename T>
__device__ __forceinline__ void owl_ar_quant_copy_fp8(unsigned char* __restrict__ dst,
                                                     const T* __restrict__ src,
                                                     const size_t bytes) {
    const size_t n = bytes / sizeof(T);
    if (n == 0) {
        return;
    }
    const size_t ng = (n + OWL_AR_FP8_G - 1) / OWL_AR_FP8_G;
    __nv_fp8_e4m3* q = reinterpret_cast<__nv_fp8_e4m3*>(dst);
    __half* sc = reinterpret_cast<__half*>(dst + OWL_AR_FP8_SC_OFF(n)); // 尺度区(偶对齐;见宏注)
    const bool aligned = ((reinterpret_cast<unsigned long long>(dst) |
                           reinterpret_cast<unsigned long long>(src)) & 15ull) == 0ull;
    if (aligned) {
        owl_ar_quant_fp8_vec<T>(q, sc, src, n, ng);
    } else {
        owl_ar_quant_fp8_scalar<T>(q, sc, src, n, ng);
    }
}

// ---- stage 宏实例:纯字节搬运(dtype 只决定"拷多少字节",不改核体;
//      保留类型化指针让"表 ⇔ .cu 签名"机器互证可查)。----
// `XFER` = 搬运函数(量化档换成 `owl_ar_quant_copy_fp8<T>`;签名统一见上)。
#define OWL_AR_STAGE_KERNEL(NAME, T, XFER, PARAMS)                                    \
    static_assert(sizeof(T) == 4 || sizeof(T) == 2, "AR stage: 仅 f32/f16/bf16");      \
    extern "C" __global__ void NAME PARAMS {                                          \
        unsigned char* own_b = reinterpret_cast<unsigned char*>(own);                      \
        const unsigned char* peer_b = reinterpret_cast<const unsigned char*>(peer);   \
        /* ★ **数据块**(2026-10-18;push 档 §36.7):slot 数据写哪里。            \
         * v1 = `own`(写本端槽);push = `peer`(写对端槽 —— posted write)。   \
         * header(`own`/`peer`)两档都不变 ⇒ 块间令牌留在本端(§36.8)。 */    \
        unsigned char* data_b = reinterpret_cast<unsigned char*>(data);      \
        volatile unsigned long long* own_prod =                                       \
            reinterpret_cast<volatile unsigned long long*>(own_b + OWL_AR_OFF_PRODUCED); \
        volatile unsigned long long* own_cons =                                       \
            reinterpret_cast<volatile unsigned long long*>(own_b + OWL_AR_OFF_CONSUMED); \
        volatile unsigned int* own_err =                                              \
            reinterpret_cast<volatile unsigned int*>(own_b + OWL_AR_OFF_ERR);         \
        const volatile unsigned int* own_magic =                                      \
            reinterpret_cast<const volatile unsigned int*>(own_b + OWL_AR_OFF_MAGIC); \
        const volatile unsigned int* own_slot_bytes =                                 \
            reinterpret_cast<const volatile unsigned int*>(own_b + OWL_AR_OFF_SLOT_BYTES); \
        const volatile unsigned int* own_nslots =                                     \
            reinterpret_cast<const volatile unsigned int*>(own_b + OWL_AR_OFF_N_SLOTS); \
        const volatile unsigned long long* peer_cons =                                \
            reinterpret_cast<const volatile unsigned long long*>(peer_b + OWL_AR_OFF_CONSUMED); \
        const volatile unsigned long long* peer_prod =                                \
            reinterpret_cast<const volatile unsigned long long*>(peer_b + OWL_AR_OFF_PRODUCED); \
        volatile unsigned long long* own_wait =                                        \
            reinterpret_cast<volatile unsigned long long*>(own_b + OWL_AR_OFF_WAIT_DONE); \
        unsigned int* own_blocks_done =                                                \
            reinterpret_cast<unsigned int*>(own_b + OWL_AR_OFF_BLOCKS_DONE);           \
        __shared__ unsigned long long s_prod;                                         \
        __shared__ int s_fail;                                                        \
        if (threadIdx.x == 0) {                                                       \
            s_prod = 0ull;                                                            \
            s_fail = 0;                                                               \
            if (*own_magic != OWL_AR_MAGIC) {                                         \
                *own_err = 3u; /* 串块/未初始化 */                                    \
                s_fail = 1;                                                           \
            } else if ((size_t)(*own_slot_bytes) != slot_bytes) {                     \
                *own_err = 3u; /* 声明与 header 漂移(两处不同改的哨兵) */            \
                s_fail = 1;                                                           \
            } else if (*own_nslots == 0u) {                                           \
                /* 前置清单 ④:0 格 = 无格可写(调用方错)⇒ 响亮拒。                     \
                 * 1 格**合法**(严格一深环;安全性论证见头注 ④/代发 credit)。 */      \
                *own_err = 3u;                                                        \
                s_fail = 1;                                                           \
            } else {                                                                  \
                /* (a) **代发 credit**(2026-10-17 CE 前置;对 v1 亦安全)          \
                 * `own_cons = own_prod` = "我读过的对端调用数"。**值与 reduce 的发布   \
                 * 收敛**(reduce_i 发 i+1;stage_{i+1} 也发 i+1)⇒ 无条件加不会与      \
                 * reduce 打架 —— 这也是它不必进门的原因。CE 路径下 add 绑到 inbox,   \
                 * staging.consumed 就只剩这一个写者。 */                             \
                const unsigned long long prod0 = *own_prod;                           \
                *own_cons = prod0;                                                    \
                const unsigned long long prod = prod0;                                \
                const unsigned long long n_slots = (unsigned long long)(*own_nslots); \
                /* I1:不覆盖未读 —— peer.consumed + n_slots > own.produced */        \
                /* 两级等待(与 reduce 同构):**block 0** 去 PCIe 读对端 consumed;      \
                 * 其余 block 读**本端 header** 的 wait_done(本地 L2)。               \
                 * ⚠️ block 0 **无条件**发布令牌(即便本档无需 credit)——否则小档       \
                 * (`prod < n_slots`)里其余 block 会等到超时(写这刀时踩过的坑)。 */  \
                const unsigned long long need =                                    \
                    prod >= n_slots ? (prod - n_slots + 1ull) : 0ull;              \
                const unsigned long long t0 = owl_ar_timer_ns();                   \
                unsigned int spins = 0u;                                           \
                if (blockIdx.x == 0) {                                             \
                    while (*peer_cons < need) {                                    \
                        if (++spins > OWL_AR_SPIN_TIGHT) {                         \
                            __nanosleep(200u);                                     \
                        }                                                          \
                        if (owl_ar_timer_ns() - t0 > OWL_AR_TIMEOUT_NS) {          \
                            *own_err = 1u;                                         \
                            s_fail = 1;                                            \
                            break;                                                 \
                        }                                                          \
                    }                                                              \
                    if (!s_fail) {                                                 \
                        __threadfence_system();                                    \
                        *own_wait = prod + 1ull; /* 无条件放行其余 block */        \
                    }                                                              \
                } else {                                                           \
                    while (*own_wait < prod + 1ull) {                              \
                        if (++spins > OWL_AR_SPIN_TIGHT) {                         \
                            __nanosleep(200u);                                     \
                        }                                                          \
                        if (owl_ar_timer_ns() - t0 > OWL_AR_TIMEOUT_NS) {          \
                            *own_err = 1u;                                         \
                            s_fail = 1;                                            \
                            break;                                                 \
                        }                                                          \
                    }                                                              \
                    if (!s_fail) {                                                 \
                        __threadfence_system();                                    \
                    }                                                              \
                }                                                                  \
                s_prod = prod;                                                        \
            }                                                                         \
        }                                                                             \
        __syncthreads();                                                              \
        if (s_fail) {                                                                 \
            return; /* 全体一致退出(thread0 不早退 —— 防 __syncthreads 死锁) */    \
        }                                                                             \
        const unsigned int slot =                                                     \
            (unsigned int)(s_prod % (unsigned long long)(*own_nslots));               \
        XFER(data_b + OWL_AR_HDR_BYTES +                                     \
                 (size_t)slot * slot_bytes,                                  \
             reinterpret_cast<const T*>(                                     \
                 reinterpret_cast<const unsigned char*>(local) + slab_off),  \
             bytes);                                                         \
        __threadfence_system(); /* 每线程:本线程写 → 对系统可见 */                 \
        __syncthreads();                                                              \
        /* ★ **最后一块发布 produced**(多 block;本块数据已系统可见)。 */          \
        if (threadIdx.x == 0) {                                                       \
            const unsigned int done = atomicAdd(own_blocks_done, 1u);                  \
            if (done == gridDim.x - 1u) {                                              \
                __threadfence_system();                                                \
                *own_blocks_done = 0u;  /* 回零:同流保序 ⇒ 下一次调用必见 0 */        \
                if (!s_fail) {                                                        \
                    *own_prod = s_prod + 1ull;                                        \
                    /* (b) **尾部 rendezvous**(CE 前置:CE 拷是主机侧的,不能自旋     \
                     * 等对端 ⇒ "对端已发布本轮"必须由核来等)。放在**最后到的那一块** \
                     * 上最省:其余块已退出 ⇒ 核完成时刻 = rendezvous 结束。        \
                     * 对 v1 亦安全:这个等待原在 reduce 里,只是提前到同一条流上。  \
                     * 无死锁:双方都先发布自己的 produced 再等。 */                 \
                    const unsigned long long want = s_prod + 1ull;                    \
                    const unsigned long long t0 = owl_ar_timer_ns();                  \
                    unsigned int spins = 0u;                                          \
                    while (*peer_prod < want) {                                       \
                        if (++spins > OWL_AR_SPIN_TIGHT) {                            \
                            __nanosleep(200u);                                        \
                        }                                                             \
                        if (owl_ar_timer_ns() - t0 > OWL_AR_TIMEOUT_NS) {             \
                            *own_err = 1u;                                            \
                            break;                                                    \
                        }                                                             \
                    }                                                                 \
                    __threadfence_system();                                           \
                }                                                                      \
            }                                                                          \
        }                                                                              \
    }

OWL_AR_STAGE_KERNEL(owl_comm_ar_stage_f32, float, owl_ar_copy_bytes_t<float>, (const float* __restrict__ local, unsigned char* __restrict__ own, const unsigned char* __restrict__ peer, unsigned char* __restrict__ data, const size_t bytes, const size_t slot_bytes, const size_t slab_off))
OWL_AR_STAGE_KERNEL(owl_comm_ar_stage_f16, __half, owl_ar_copy_bytes_t<__half>, (const __half* __restrict__ local, unsigned char* __restrict__ own, const unsigned char* __restrict__ peer, unsigned char* __restrict__ data, const size_t bytes, const size_t slot_bytes, const size_t slab_off))
OWL_AR_STAGE_KERNEL(owl_comm_ar_stage_bf16, __nv_bfloat16, owl_ar_copy_bytes_t<__nv_bfloat16>, (const __nv_bfloat16* __restrict__ local, unsigned char* __restrict__ own, const unsigned char* __restrict__ peer, unsigned char* __restrict__ data, const size_t bytes, const size_t slot_bytes, const size_t slab_off))
// **fp8 载荷档**(§47 第二步):同协议/同几何,只把搬运换成"量化 + 写尺度"
OWL_AR_STAGE_KERNEL(owl_comm_ar_stage_f8_f16, __half, owl_ar_quant_copy_fp8<__half>, (const __half* __restrict__ local, unsigned char* __restrict__ own, const unsigned char* __restrict__ peer, unsigned char* __restrict__ data, const size_t bytes, const size_t slot_bytes, const size_t slab_off))
OWL_AR_STAGE_KERNEL(owl_comm_ar_stage_f8_bf16, __nv_bfloat16, owl_ar_quant_copy_fp8<__nv_bfloat16>, (const __nv_bfloat16* __restrict__ local, unsigned char* __restrict__ own, const unsigned char* __restrict__ peer, unsigned char* __restrict__ data, const size_t bytes, const size_t slot_bytes, const size_t slab_off))

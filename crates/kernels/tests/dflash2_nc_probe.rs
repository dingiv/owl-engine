//! DFlash2 **非因果块注意力(NC)** 单算子门 —— propose 路径的草稿注意力。
//!
//! 语义(生产实况,`owl_naive_attn_nc_*`;`crates/kernels/cu/owl/dflash2.cu`):
//! - 查询 = 自块 T 行(propose 的 8 个位置),**非因果**:每行看
//!   `[srow_begin, kv_len)`,即"前缀滑窗 + 整个自块";
//! - `kv_len = kv_len_p[0]`,`prefix_len = kv_len − gridDim.x`(自块 = 最后 T 行);
//! - `srow_begin = max(0, prefix_len − 2048)`(**D2 滑窗**,前缀只取最近 2048 行;
//!   自块恒全可见);
//! - 前缀行从池读(`kc` 经典布局 `[nb,hkv,D/x,page,x]` / `vc` `[nb,hkv,D,page]`),
//!   自块行从 `k_self/v_self` 直读;
//! - **单头全局 softmax**(非 online 语义;核内 online 是等价重排)。
//!
//! 三条判据:
//!   1) 数值:GPU vs host 参考(上面这套语义)误差 < 1e-2;
//!   2) 速度:同形状 per-call ms(改造前 = 基线;改造后 = 对照);
//!   3) **A/B 同门**:`OWL_NC_KERNEL=<名字>` 可切核(新旧核跑同一份判据)。
//!
//! 运行:`OWL_TEST_DEVICE=1 cargo test --release -p owl-kernels \
//!        --test dflash2_nc_probe -- --test-threads=1 --nocapture`

use owl_macros::owl_test;
use std::time::Instant;

use cudarc::driver::{CudaContext, DevicePtr, LaunchConfig, PushKernelArg};

// ---- 生产形状(DFlash2 spec-dflash 档)----
const T: usize = 8; // 自块行数(= propose 的 8 个位置;grid.x)
const HQ: usize = 32;
const HKV: usize = 8;
const HD: usize = 128;
const PAGE: usize = 32;
const X: usize = 8; // bf16: 16B/2B

fn bf16_bits(v: f32) -> u16 {
    half::bf16::from_f32(v).to_bits()
}
fn bf16_to_f32(b: u16) -> f32 {
    half::bf16::from_bits(b).to_f32()
}
/// 随机 bf16 数据(LCG;幅值 ±2)
fn lcg_bf16(n: usize, seed: u32) -> Vec<u16> {
    (0..n)
        .map(|i| {
            let x = seed
                .wrapping_mul(0x9E37_79B9)
                .wrapping_add((i as u32).wrapping_mul(0x85EB_CA6B));
            bf16_bits(((x & 0xFFFF) as f32 / 65535.0 - 0.5) * 4.0)
        })
        .collect()
}

/// 池元素下标(逐字同核:K 经典布局 / V 经典布局)
fn k_idx(kvh: usize, d: usize, page_off: usize) -> usize {
    let hd_x = HD / X;
    kvh * hd_x * PAGE * X + (d / X) * PAGE * X + page_off * X + (d % X)
}
fn v_idx(kvh: usize, d: usize, page_off: usize) -> usize {
    kvh * HD * PAGE + d * PAGE + page_off
}

/// host 参考:f32 全量 softmax,语义见模块头注。
fn host_ref(
    q: &[u16],
    k_self: &[u16],
    v_self: &[u16],
    kc: &[u16],
    vc: &[u16],
    kv_len: usize,
) -> Vec<f32> {
    let prefix = kv_len - T;
    let srow_begin = prefix.saturating_sub(2048);
    let scale = 1.0 / (HD as f32).sqrt();
    let mut out = vec![0f32; T * HQ * HD];
    for t in 0..T {
        for h in 0..HQ {
            let kvh = h / (HQ / HKV);
            let mut qk = Vec::with_capacity(kv_len - srow_begin);
            for srow in srow_begin..kv_len {
                let mut acc = 0f32;
                for d in 0..HD {
                    let qd = bf16_to_f32(q[(t * HQ + h) * HD + d]);
                    let kv = if srow < prefix {
                        // 池:页号 = srow/PAGE,页内偏移 = srow%PAGE
                        kc[k_idx(kvh, d, srow % PAGE) + (srow / PAGE) * HKV * HD * PAGE]
                    } else {
                        k_self[((srow - prefix) * HKV + kvh) * HD + d]
                    };
                    acc += qd * bf16_to_f32(kv);
                }
                qk.push(acc * scale);
            }
            let m = qk.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let l: f32 = qk.iter().map(|&x| (x - m).exp()).sum();
            for d in 0..HD {
                let mut acc = 0f32;
                for (i, srow) in (srow_begin..kv_len).enumerate() {
                    let v = if srow < prefix {
                        vc[v_idx(kvh, d, srow % PAGE) + (srow / PAGE) * HKV * HD * PAGE]
                    } else {
                        v_self[((srow - prefix) * HKV + kvh) * HD + d]
                    };
                    acc += (qk[i] - m).exp() * bf16_to_f32(v);
                }
                out[(t * HQ + h) * HD + d] = acc / l;
            }
        }
    }
    out
}

fn run_case(prefix: usize, check: bool) {
    let kv_len = prefix + T;
    let nb = prefix.div_ceil(PAGE);
    let ctx = CudaContext::new(1).expect("ctx");
    ctx.bind_to_thread().expect("bind");
    let stream = ctx.default_stream();

    // ---- 数据 ----
    let q = lcg_bf16(T * HQ * HD, 901);
    let k_self = lcg_bf16(T * HKV * HD, 902);
    let v_self = lcg_bf16(T * HKV * HD, 903);
    // 池:整页分配(末页只用前 prefix%PAGE 行)
    let kc_n = nb * HKV * (HD / X) * PAGE * X;
    let vc_n = nb * HKV * HD * PAGE;
    let kc = lcg_bf16(kc_n, 904);
    let vc = lcg_bf16(vc_n, 905);

    let dev = |v: &[u16]| {
        let mut d = unsafe { stream.alloc::<u16>(v.len()) }.unwrap();
        stream.memcpy_htod(v, &mut d).unwrap();
        d
    };
    let d_q = dev(&q);
    let d_ks = dev(&k_self);
    let d_vs = dev(&v_self);
    let d_kc = dev(&kc);
    let d_vc = dev(&vc);
    let d_out = dev(&vec![0u16; T * HQ * HD]);
    let d_len = {
        let mut d = unsafe { stream.alloc::<f32>(1) }.unwrap();
        stream.memcpy_htod(&[kv_len as f32], &mut d).unwrap();
        d
    };

    // ---- 编译 + 取核(OWL_NC_KERNEL 可切名做 A/B)----
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(
        owl_kernels::cu_sig::cu_text(owl_kernels::ops::source("owl_naive_attn_nc_bf16_v2")).as_str(),
        cudarc::nvrtc::CompileOptions {
            arch: Some("compute_86"),
            include_paths: vec!["/usr/local/cuda/include".into()],
            ..Default::default()
        },
    )
    .expect("nvrtc compile dflash2");
    let m = ctx.load_module(ptx).expect("module");
    let name = owl_shared::env_reader::str("OWL_NC_KERNEL")
        .unwrap_or_else(|| "owl_naive_attn_nc_bf16".to_string());
    let f = m.load_function(&name).unwrap_or_else(|e| panic!("load {name}: {e}"));

    let (q_p, ks_p, vs_p, kc_p, vc_p, ln_p, o_p) = (
        d_q.device_ptr(&stream).0,
        d_ks.device_ptr(&stream).0,
        d_vs.device_ptr(&stream).0,
        d_kc.device_ptr(&stream).0,
        d_vc.device_ptr(&stream).0,
        d_len.device_ptr(&stream).0,
        d_out.device_ptr(&stream).0,
    );
    let (hq_i, hkv_i, hd_i, page_i, x_i) =
        (HQ as i32, HKV as i32, HD as i32, PAGE as i32, X as i32);
    let cfg = LaunchConfig {
        grid_dim: (T as u32, HQ as u32, 1),
        block_dim: (HD as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let call = || {
        let mut b = stream.launch_builder(&f);
        b.arg(&q_p)
            .arg(&ks_p)
            .arg(&vs_p)
            .arg(&kc_p)
            .arg(&vc_p)
            .arg(&ln_p)
            .arg(&hq_i)
            .arg(&hkv_i)
            .arg(&hd_i)
            .arg(&page_i)
            .arg(&x_i)
            .arg(&o_p);
        unsafe { b.launch(cfg) }.expect("nc launch");
    };
    call();
    stream.synchronize().expect("sync");

    // ---- 计时 ----
    for _ in 0..3 {
        call();
    }
    stream.synchronize().unwrap();
    let iters = 20;
    let t0 = Instant::now();
    for _ in 0..iters {
        call();
    }
    stream.synchronize().unwrap();
    let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;

    // ---- 数值 ----
    let mut o = vec![0u16; T * HQ * HD];
    stream.memcpy_dtoh(&d_out, &mut o).unwrap();
    let got: Vec<f32> = o.iter().map(|&b| bf16_to_f32(b)).collect();
    let n_zero = got.iter().filter(|&&v| v == 0.0).count();
    let d_h = if check {
        let h = host_ref(&q, &k_self, &v_self, &kc, &vc, kv_len);
        Some(got.iter().zip(h.iter()).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max))
    } else {
        None
    };
    let win = prefix.min(2048);
    println!(
        "=== {name} prefix={prefix} (窗口={win}+自块{T}, grid=({T},{HQ})) ===\n  \
         单发      : {ms:7.3} ms/call\n  \
         数值 max|gpu−host| = {:?}   (零值 {n_zero}/{})\n  \
         有效行数/block = {T} · 总行/block = {}",
        d_h,
        got.len(),
        win + T
    );
    if let Some(dh) = d_h {
        assert!(dh < 1e-2, "{name} vs host 超门:{dh:.6}");
    }
}

#[owl_test(gpu)]
fn dflash2_nc_speed_and_parity() {
    // 生产窗口点(滑窗生效:prefix > 2048)+ host 参考
    run_case(2048, true);
    // 短 ctx(滑窗不生效)+ 长 ctx(滑窗钳死)只测速
    run_case(512, false);
    run_case(8192, false);
}

// ============================================================================
// 诊断用迷你核(只住测试文件):把"K 读 / V 读 / 同步链"三者的代价拆开。
// 逐字复刻 `owl_naive_attn_nc_bf16` 的循环结构与寻址,只按模板参数跳过全局读。
// ============================================================================
const DIAG_SRC: &str = r#"
extern "C" __global__ void __launch_bounds__(512) diag_dummy();
#include <cuda_bf16.h>
#define OWL_NC_SLIDING_WINDOW 2048

template <bool DO_K, bool DO_V>
__device__ __forceinline__ void diag_body(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_self,
    const __nv_bfloat16* __restrict__ v_self,
    const __nv_bfloat16* __restrict__ kc,
    const __nv_bfloat16* __restrict__ vc,
    const float* __restrict__ kv_len_p,
    int q_heads, int kv_heads, int d_dim, int page, int x,
    __nv_bfloat16* __restrict__ out) {
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    const int hd = blockDim.x;
    const int kvh = h / (q_heads / kv_heads);
    const int hd_x = d_dim / x;
    const int kv_len = (int)kv_len_p[0];
    const int prefix_len = kv_len - gridDim.x;
    const __nv_bfloat16* qs = q + ((long long)t * q_heads + h) * d_dim;
    const float scale = rsqrtf((float)d_dim);
    __shared__ float red[512];
    __shared__ float s_m, s_l, s_resc, s_p;
    const float qd = __bfloat162float(qs[d]);
    float acc = 0.0f;
    if (d == 0) { s_m = -3.0e38f; s_l = 0.0f; }
    __syncthreads();
    const int srow_begin =
        (prefix_len > OWL_NC_SLIDING_WINDOW) ? (prefix_len - OWL_NC_SLIDING_WINDOW) : 0;
    for (int srow = srow_begin; srow < kv_len; ++srow) {
        float kv = 0.9f;
        if (DO_K) {
            if (srow < prefix_len) {
                const int b = srow / page;
                const int off = srow % page;
                kv = __bfloat162float(kc[((long long)b * kv_heads + kvh) * hd_x * page * x
                                         + (long long)(d / x) * page * x + off * x + d % x]);
            } else {
                const int st = srow - prefix_len;
                kv = __bfloat162float(k_self[((long long)st * kv_heads + kvh) * d_dim + d]);
            }
        }
        red[d] = qd * kv;
        __syncthreads();
        for (int s2 = hd / 2; s2 > 0; s2 >>= 1) {
            if (d < s2) red[d] += red[d + s2];
            __syncthreads();
        }
        if (d == 0) {
            const float dot = red[0] * scale;
            const float m_new = fmaxf(s_m, dot);
            const float resc = __expf(s_m - m_new);
            const float p = __expf(dot - m_new);
            s_l = s_l * resc + p;
            s_resc = resc;
            s_p = p;
            s_m = m_new;
        }
        __syncthreads();
        float p = s_p;
        if (DO_V) {
            if (srow < prefix_len) {
                const int b = srow / page;
                const int off = srow % page;
                p *= __bfloat162float(vc[((long long)b * kv_heads + kvh) * d_dim * page
                                         + (long long)d * page + off]);
            } else {
                const int st = srow - prefix_len;
                p *= __bfloat162float(v_self[((long long)st * kv_heads + kvh) * d_dim + d]);
            }
        }
        acc = acc * s_resc + p;
        __syncthreads();
    }
    out[((long long)t * q_heads + h) * d_dim + d] = __float2bfloat16(acc / s_l);
}

#define DIAG_ENTRY(NAME, DK, DV)                                                   \
    extern "C" __global__ void NAME(                                               \
        const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ ks, \
        const __nv_bfloat16* __restrict__ vs, const __nv_bfloat16* __restrict__ kc, \
        const __nv_bfloat16* __restrict__ vc, const float* __restrict__ kl,        \
        int qh, int kh, int dd, int pg, int xx, __nv_bfloat16* __restrict__ o) {   \
        diag_body<DK, DV>(q, ks, vs, kc, vc, kl, qh, kh, dd, pg, xx, o);           \
    }
DIAG_ENTRY(diag_kv, true, true)
DIAG_ENTRY(diag_k, true, false)
DIAG_ENTRY(diag_v, false, true)
DIAG_ENTRY(diag_none, false, false)
"#;

fn run_diag(prefix: usize, variant: &str) {
    let kv_len = prefix + T;
    let nb = prefix.div_ceil(PAGE);
    let ctx = CudaContext::new(1).expect("ctx");
    ctx.bind_to_thread().expect("bind");
    let stream = ctx.default_stream();
    let q = lcg_bf16(T * HQ * HD, 901);
    let k_self = lcg_bf16(T * HKV * HD, 902);
    let v_self = lcg_bf16(T * HKV * HD, 903);
    let kc = lcg_bf16(nb * HKV * (HD / X) * PAGE * X, 904);
    let vc = lcg_bf16(nb * HKV * HD * PAGE, 905);
    let dev = |v: &[u16]| {
        let mut d = unsafe { stream.alloc::<u16>(v.len()) }.unwrap();
        stream.memcpy_htod(v, &mut d).unwrap();
        d
    };
    let (d_q, d_ks, d_vs, d_kc, d_vc, d_out) =
        (dev(&q), dev(&k_self), dev(&v_self), dev(&kc), dev(&vc), dev(&vec![0u16; T * HQ * HD]));
    let d_len = {
        let mut d = unsafe { stream.alloc::<f32>(1) }.unwrap();
        stream.memcpy_htod(&[kv_len as f32], &mut d).unwrap();
        d
    };
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(
        DIAG_SRC,
        cudarc::nvrtc::CompileOptions {
            arch: Some("compute_86"),
            include_paths: vec!["/usr/local/cuda/include".into()],
            ..Default::default()
        },
    )
    .expect("nvrtc compile diag");
    let m = ctx.load_module(ptx).expect("module");
    let f = m.load_function(variant).unwrap_or_else(|e| panic!("load {variant}: {e}"));
    let (q_p, ks_p, vs_p, kc_p, vc_p, ln_p, o_p) = (
        d_q.device_ptr(&stream).0, d_ks.device_ptr(&stream).0, d_vs.device_ptr(&stream).0,
        d_kc.device_ptr(&stream).0, d_vc.device_ptr(&stream).0,
        d_len.device_ptr(&stream).0, d_out.device_ptr(&stream).0,
    );
    let (hq_i, hkv_i, hd_i, page_i, x_i) =
        (HQ as i32, HKV as i32, HD as i32, PAGE as i32, X as i32);
    let cfg = LaunchConfig {
        grid_dim: (T as u32, HQ as u32, 1),
        block_dim: (HD as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let call = || {
        let mut b = stream.launch_builder(&f);
        b.arg(&q_p).arg(&ks_p).arg(&vs_p).arg(&kc_p).arg(&vc_p).arg(&ln_p)
            .arg(&hq_i).arg(&hkv_i).arg(&hd_i).arg(&page_i).arg(&x_i).arg(&o_p);
        unsafe { b.launch(cfg) }.expect("diag launch");
    };
    for _ in 0..3 { call(); }
    stream.synchronize().unwrap();
    let iters = 20;
    let t0 = Instant::now();
    for _ in 0..iters { call(); }
    stream.synchronize().unwrap();
    println!("  {variant:<10} {:.3} ms", t0.elapsed().as_secs_f64() * 1e3 / iters as f64);
}

#[owl_test(gpu)]
fn dflash2_nc_bottleneck_split() {
    for prefix in [2048usize, 8192] {
        println!("=== 拆分(prefix={prefix}, 窗口 {} + 自块 {T})===", prefix.min(2048));
        for v in ["diag_kv", "diag_k", "diag_v", "diag_none"] {
            run_diag(prefix, v);
        }
    }
}

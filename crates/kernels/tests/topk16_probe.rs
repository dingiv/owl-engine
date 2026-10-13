//! `owl_topk16_f16` 独立微基准 + split-K 原型(2026-10-14 D2,decode 战线)。
//!
//! 动机:CUPTI 图内逐核报 `gpu.graph.29.owl_topk16_f16` = **1067.9 µs/次**
//! (每轮 1 次)。量纲可疑:rows=7 × vocab=248320 × 2B = 3.48 MB ⇒ 3.2 GB/s
//! (带宽地板 ~900 GB/s)。结构疑点:`driver::topk16(rows)` 的 grid=(rows,1,1)
//! = **7 块**(rows = DEPTH = 7)⇒ 只用 7/132 SM。
//!
//! 账本(全部固定 rows=7 / vocab=248320 / block=256):
//!   A) 生产核直测(grid=(7,1,1));
//!   B) **纯扫描地板**(同循环结构,不选择)= "读 + 循环" 成本;
//!   C) 二维矩阵:`插入法(旧动态索引 vs 新全展开)× 归并(不做 / 做块内 16 轮 argmax)`
//!      ⇒ 定谳 1900 µs 到底花在哪一段;
//!   D) split-K(grid = rows×nparts)各档计时 + 与 host 逐位对拍。
//!
//! 运行:`OWL_TEST_DEVICE=1 cargo test --release -p owl-kernels \
//!        --test topk16_probe -- --test-threads=1 --nocapture`
//! (NVRTC 只住测试域;生产路径零运行期编译。)

use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaContext, CudaModule, DevicePtr, LaunchConfig, PushKernelArg};

// ---- 生产形状(spec-dflash:rows = DEPTH = 7;vocab 248320)----
const ROWS: usize = 7;
const VOCAB: usize = 248320;
const K: usize = 16;
const BLK: usize = 256;

fn f16_bits(v: f32) -> u16 {
    half::f16::from_f32(v).to_bits()
}
fn f16_to_f32(b: u16) -> f32 {
    half::f16::from_bits(b).to_f32()
}

/// 伪随机 f16 数据(LCG;±8 幅值,logits 量级)
fn lcg_f16(n: usize, seed: u32) -> Vec<u16> {
    (0..n)
        .map(|i| {
            let x = seed
                .wrapping_mul(0x9E37_79B9)
                .wrapping_add((i as u32).wrapping_mul(0x85EB_CA6B));
            f16_bits(((x >> 8) as f32 / 16777215.0 - 0.5) * 16.0)
        })
        .collect()
}

/// host 参考:逐行 top-16(值降序;平局取小索引)—— 与核内语义同律
fn host_topk(x: &[u16], rows: usize, vocab: usize) -> (Vec<f32>, Vec<i32>) {
    let mut vals = vec![0f32; rows * K];
    let mut idx = vec![0i32; rows * K];
    for r in 0..rows {
        let mut v: Vec<(f32, i32)> = (0..vocab)
            .map(|i| (f16_to_f32(x[r * vocab + i]), i as i32))
            .collect();
        v.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
        for k in 0..K {
            vals[r * K + k] = v[k].0;
            idx[r * K + k] = v[k].1;
        }
    }
    (vals, idx)
}

// ============================================================================
// 诊断 + 原型核(只住本测试文件;NVRTC 直编)
// ============================================================================

/// 单块一行(生产 grid 形态)的纯扫描地板
const DIAG_SCAN: &str = r#"
#include <cuda_fp16.h>
extern "C" __global__ void diag_scan_f16(const __half* __restrict__ x, int vocab,
                                         float* __restrict__ out) {
    const int row = blockIdx.x, tid = threadIdx.x;
    const __half* xr = x + (long long)row * vocab;
    float m = -3.4e38f;
    for (int i = tid; i < vocab; i += 256) m = fmaxf(m, __half2float(xr[i]));
    if (tid == 0) out[row] = m;
}
"#;

/// split 形态三段分解(模板:INS 0=不插入 1=旧动态索引 2=新全展开;
/// MERGE 0=不归并(逐线程 top-16 直写)1=块内 16 轮 argmax 归并)
const SPLIT_SRC: &str = r#"
#include <cuda_fp16.h>
#define TK 16
#define TB 256

template <int INS, int MERGE>
__device__ __forceinline__ void split_body(
    const __half* __restrict__ x, int vocab, int nparts, int rows,
    float* __restrict__ pv, float* __restrict__ pi) {
    const int row  = blockIdx.x / nparts;
    const int part = blockIdx.x % nparts;
    const int tid  = threadIdx.x;
    const int chunk = (vocab + nparts - 1) / nparts;
    const int lo = part * chunk;
    const int hi = min(vocab, lo + chunk);
    const __half* xr = x + (long long)row * vocab;

    float tv[TK]; int ti[TK];
    int tn = 0;

    if (INS == 0) {
        float m = -3.4e38f;
        for (int i = lo + tid; i < hi; i += TB) m = fmaxf(m, __half2float(xr[i]));
        if (tid == 0) pv[row * nparts + part] = m;
        return;
    } else if (INS == 1) {
        for (int i = lo + tid; i < hi; i += TB) {
            const float v = __half2float(xr[i]);
            if (tn < TK || v > tv[tn - 1]) {
                int a = 0, b = tn;
                while (a < b) { const int mid = (a + b) >> 1; if (tv[mid] >= v) a = mid + 1; else b = mid; }
                if (tn < TK) ++tn;
                for (int j = tn - 1; j > a; --j) { tv[j] = tv[j - 1]; ti[j] = ti[j - 1]; }
                tv[a] = v; ti[a] = i;
            }
        }
    } else {
        #pragma unroll
        for (int k = 0; k < TK; ++k) { tv[k] = -3.4e38f; ti[k] = 0x7fffffff; }
        for (int i = lo + tid; i < hi; i += TB) {
            float v = __half2float(xr[i]); int ii = i;
            if (v > tv[TK - 1]) {
                #pragma unroll
                for (int k = 0; k < TK; ++k) {
                    if (v > tv[k]) { const float t = tv[k]; const int t2 = ti[k]; tv[k] = v; ti[k] = ii; v = t; ii = t2; }
                }
            }
        }
        tn = TK;
    }

    if (MERGE == 0) {
        // 逐线程 top-16 直写(只作分解账;验证用)
        for (int k = 0; k < TK; ++k) {
            const int slot = ((row * nparts + part) * TB + tid) * TK + k;
            pv[slot] = (k < tn) ? tv[k] : -3.4e38f;
            pi[slot] = (k < tn) ? (float)ti[k] : (float)0x7fffffff;
        }
        return;
    }

    __shared__ float sv[TB * TK];
    __shared__ int   si[TB * TK];
    __shared__ float rs_val[TB];
    __shared__ int   rs_idx[TB];
    __shared__ float s_wv; __shared__ int s_wi;
    #pragma unroll
    for (int k = 0; k < TK; ++k) {
        const int slot = tid * TK + k;
        if (k < tn) { sv[slot] = tv[k]; si[slot] = ti[k]; }
        else        { sv[slot] = -3.4e38f; si[slot] = 0x7fffffff; }
    }
    __syncthreads();
    #pragma unroll 1
    for (int k = 0; k < TK; ++k) {
        float best = -3.4e38f; int bi = 0x7fffffff;
        for (int i = tid; i < TB * TK; i += TB) {
            const float v = sv[i];
            if (v > best || (v == best && si[i] < bi)) { best = v; bi = si[i]; }
        }
        rs_val[tid] = best; rs_idx[tid] = bi;
        __syncthreads();
        #pragma unroll
        for (int off = TB / 2; off > 0; off >>= 1) {
            if (tid < off) {
                const float ov = rs_val[tid + off]; const int oi = rs_idx[tid + off];
                if (ov > rs_val[tid] || (ov == rs_val[tid] && oi < rs_idx[tid])) {
                    rs_val[tid] = ov; rs_idx[tid] = oi;
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            pv[(long long)(row * nparts + part) * TK + k] = rs_val[0];
            pi[(long long)(row * nparts + part) * TK + k] = (float)rs_idx[0];
            s_wv = rs_val[0]; s_wi = rs_idx[0];
        }
        __syncthreads();
        {
            const float wv = s_wv; const int wi = s_wi;
            for (int i = tid; i < TB * TK; i += TB) {
                if (sv[i] == wv && si[i] == wi) { sv[i] = -3.4e38f; si[i] = 0x7fffffff; }
            }
        }
        __syncthreads();
    }
    (void)rows;
}

#define WRAP(NAME, INS, MERGE)                                                  \
extern "C" __global__ void NAME(const __half* __restrict__ x, int vocab,        \
    int nparts, int rows, float* __restrict__ pv, float* __restrict__ pi) {     \
    split_body<INS, MERGE>(x, vocab, nparts, rows, pv, pi);                     \
}
WRAP(diag_split_scan_f16, 0, 1)
WRAP(diag_split_old_w_f16, 1, 0)
WRAP(diag_split_old_f16, 1, 1)
WRAP(diag_split_new_w_f16, 2, 0)
WRAP(diag_split_new_f16, 2, 1)
"#;

/// 归并(split-K 第二段):grid = (rows),把 nparts×16 候选归并成 top-16
const MERGE_SRC: &str = r#"
extern "C" __global__ void diag_merge_sync_f16(
    const float* __restrict__ pv, const float* __restrict__ pi,
    int nparts, int rows, float* __restrict__ out) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int n = nparts * 16;
    extern __shared__ float smem[];
    float* sv = smem;
    int*   si = (int*)(smem + 4096);
    for (int i = tid; i < n; i += 256) { sv[i] = pv[(long long)row * n + i]; si[i] = (int)pi[(long long)row * n + i]; }
    __syncthreads();
    __shared__ float rs_val[256];
    __shared__ int   rs_idx[256];
    __shared__ float s_wv; __shared__ int s_wi;
    for (int k = 0; k < 16; ++k) {
        float best = -3.4e38f; int bi = 0x7fffffff;
        for (int i = tid; i < n; i += 256) {
            const float v = sv[i];
            if (v > best || (v == best && si[i] < bi)) { best = v; bi = si[i]; }
        }
        rs_val[tid] = best; rs_idx[tid] = bi;
        __syncthreads();
        #pragma unroll
        for (int off = 128; off > 0; off >>= 1) {
            if (tid < off) {
                const float ov = rs_val[tid + off]; const int oi = rs_idx[tid + off];
                if (ov > rs_val[tid] || (ov == rs_val[tid] && oi < rs_idx[tid])) {
                    rs_val[tid] = ov; rs_idx[tid] = oi;
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            out[row * 16 + k] = rs_val[0];
            out[rows * 16 + row * 16 + k] = (float)rs_idx[0];
            s_wv = rs_val[0]; s_wi = rs_idx[0];
        }
        __syncthreads();
        const float wv = s_wv; const int wi = s_wi;
        for (int i = tid; i < n; i += 256) {
            if (sv[i] == wv && si[i] == wi) { sv[i] = -3.4e38f; si[i] = 0x7fffffff; }
        }
        __syncthreads();
    }
}
"#;

fn compile_module(ctx: &Arc<CudaContext>, src: &str) -> Arc<CudaModule> {
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(
        src,
        cudarc::nvrtc::CompileOptions {
            arch: Some("compute_86"),
            include_paths: vec!["/usr/local/cuda/include".into()],
            options: std::env::var("OWL_NVRTC_OPTS")
                .map(|s| s.split_whitespace().map(str::to_string).collect())
                .unwrap_or_else(|_| vec![]),
            ..Default::default()
        },
    )
    .expect("nvrtc compile");
    ctx.load_module(ptx).expect("load module")
}

#[test]
fn topk16_bench() {
    if !owl_shared::env_reader::flag("OWL_TEST_DEVICE") {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    }
    let dev_ord = owl_shared::env_reader::parse::<usize>("OWL_TEST_DEVICE").unwrap_or(1);
    let ctx = CudaContext::new(dev_ord).expect("ctx");
    ctx.bind_to_thread().expect("bind");
    let stream = ctx.default_stream();

    let x = lcg_f16(ROWS * VOCAB, 2026);
    let mut d_x = unsafe { stream.alloc::<u16>(x.len()) }.unwrap();
    stream.memcpy_htod(&x, &mut d_x).unwrap();
    let d_out = unsafe { stream.alloc::<f32>(2 * ROWS * K) }.unwrap();
    let d_scan = unsafe { stream.alloc::<f32>(ROWS) }.unwrap();
    let (h_vals, h_idx) = host_topk(&x, ROWS, VOCAB);

    let prod_src = owl_kernels::cu_sig::cu_text(owl_kernels::ops::source("owl_topk16_f16"));
    let base_mod = compile_module(&ctx, &prod_src);
    let diag_mod = compile_module(&ctx, &format!("{DIAG_SCAN}\n{prod_src}\n{SPLIT_SRC}\n{MERGE_SRC}"));

    let (x_p, o_p, s_p) = (
        d_x.device_ptr(&stream).0,
        d_out.device_ptr(&stream).0,
        d_scan.device_ptr(&stream).0,
    );
    let vocab_i = VOCAB as i32;
    let rows_i = ROWS as i32;

    let time = |call: &dyn Fn()| -> f64 {
        for _ in 0..3 { call(); }
        stream.synchronize().unwrap();
        let iters = 20;
        let t0 = Instant::now();
        for _ in 0..iters { call(); }
        stream.synchronize().unwrap();
        t0.elapsed().as_secs_f64() * 1e6 / iters as f64
    };
    let bytes = (ROWS * VOCAB * 2) as f64;

    // ---- A) 生产核 ----
    let f_base = base_mod.load_function("owl_topk16_f16").unwrap();
    let cfg1 = LaunchConfig { grid_dim: (ROWS as u32, 1, 1), block_dim: (BLK as u32, 1, 1), shared_mem_bytes: 0 };
    let call_base = || {
        let mut b = stream.launch_builder(&f_base);
        b.arg(&x_p).arg(&vocab_i).arg(&o_p);
        unsafe { b.launch(cfg1) }.expect("base");
    };
    let us_base = time(&call_base);
    println!("=== A) 生产核 grid=({ROWS},1,1)×{BLK} ===");
    println!("  单发 {us_base:9.3} µs ({:.1} GB/s)", bytes / (us_base * 1e-6) / 1e9);

    // ---- B) 单块一行 纯扫描地板 ----
    let f_scan1 = diag_mod.load_function("diag_scan_f16").unwrap();
    let call_scan1 = || {
        let mut b = stream.launch_builder(&f_scan1);
        b.arg(&x_p).arg(&vocab_i).arg(&s_p);
        unsafe { b.launch(cfg1) }.expect("scan1");
    };
    let us_scan1 = time(&call_scan1);
    println!("=== B) 纯扫描地板(同 grid)===");
    println!("  单发 {us_scan1:9.3} µs ({:.1} GB/s)", bytes / (us_scan1 * 1e-6) / 1e9);

    // ---- C) split 分解矩阵(固定 nparts=16)----
    println!("=== C) split 分解(nparts=16,grid={} × {BLK})===", ROWS * 16);
    let nparts = 16usize;
    let np_i = nparts as i32;
    let npart_buf = ROWS * nparts * BLK * K; // 逐线程直写形态(W 变体)需全量槽
    let mut d_pv = unsafe { stream.alloc::<f32>(npart_buf) }.unwrap();
    let mut d_pi = unsafe { stream.alloc::<f32>(npart_buf) }.unwrap();
    let pv_p = d_pv.device_ptr(&stream).0;
    let pi_p = d_pi.device_ptr(&stream).0;
    let cfg_split = LaunchConfig { grid_dim: ((ROWS * nparts) as u32, 1, 1), block_dim: (BLK as u32, 1, 1), shared_mem_bytes: 0 };
    for name in ["diag_split_scan_f16", "diag_split_old_w_f16", "diag_split_old_f16", "diag_split_new_w_f16", "diag_split_new_f16"] {
        let f = diag_mod.load_function(name).unwrap();
        let call = || {
            let mut b = stream.launch_builder(&f);
            b.arg(&x_p).arg(&vocab_i).arg(&np_i).arg(&rows_i).arg(&pv_p).arg(&pi_p);
            unsafe { b.launch(cfg_split) }.expect(name);
        };
        println!("  {name:26} {:.3} µs", time(&call));
    }

    // ---- D) 全链路(段一 + merge)各 nparts + 数值对拍 ----    println!("=== D) 全链路 split+merge ===");
    let f_merge = diag_mod.load_function("diag_merge_sync_f16").unwrap();
    for &np in &[4usize, 8, 16, 32, 64] {
        for (kname, need_w) in [("diag_split_new_f16", false), ("diag_split_old_f16", false)] {
            let _ = need_w;
            let buf = ROWS * np * K;
            let mut d_pv = unsafe { stream.alloc::<f32>(buf) }.unwrap();
            let mut d_pi = unsafe { stream.alloc::<f32>(buf) }.unwrap();
            let pv_p = d_pv.device_ptr(&stream).0;
            let pi_p = d_pi.device_ptr(&stream).0;
            let n_pi = np as i32;
            let cfg_s = LaunchConfig { grid_dim: ((ROWS * np) as u32, 1, 1), block_dim: (BLK as u32, 1, 1), shared_mem_bytes: 0 };
            let cfg_m = LaunchConfig { grid_dim: (ROWS as u32, 1, 1), block_dim: (BLK as u32, 1, 1), shared_mem_bytes: (2 * 4096 * 4) as u32 };
            let f_split = diag_mod.load_function(kname).unwrap();
            let call = || {
                {
                    let mut b = stream.launch_builder(&f_split);
                    b.arg(&x_p).arg(&vocab_i).arg(&n_pi).arg(&rows_i).arg(&pv_p).arg(&pi_p);
                    unsafe { b.launch(cfg_s) }.expect(kname);
                }
                {
                    let mut b = stream.launch_builder(&f_merge);
                    b.arg(&pv_p).arg(&pi_p).arg(&n_pi).arg(&rows_i).arg(&o_p);
                    unsafe { b.launch(cfg_m) }.expect("merge");
                }
            };
            call();
            stream.synchronize().unwrap();
            let mut o2 = vec![0f32; 2 * ROWS * K];
            stream.memcpy_dtoh(&d_out, &mut o2).unwrap();
            let bi = (0..ROWS * K).filter(|&i| (o2[ROWS * K + i] as i32) != h_idx[i]).count();
            let bv = (0..ROWS * K).filter(|&i| (o2[i] - h_vals[i]).abs() > 1e-3).count();
            println!("  np={np:3} {kname:20} {:.3} µs  索引差 {bi} 值差 {bv}", time(&call));
            let _ = d_pv.device_ptr(&stream);
            let _ = d_pi.device_ptr(&stream);
        }
    }

    // ---- F) **生产核**(split-K 两段;单缓冲;源 = 磁盘 .cu 经 NVRTC)----
    println!("=== F) 生产核 split+merge(向量载入 + warp 锦标赛)===");
    let f_split2 = base_mod.load_function("owl_topk16_split_f16").unwrap();
    let f_merge2 = base_mod.load_function("owl_topk16_merge_f16").unwrap();
    for &np in &[4usize, 8, 12, 16, 24, 32] {
        let buf = 2 * ROWS * np * K;
        let mut d_tmp = unsafe { stream.alloc::<f32>(buf) }.unwrap();
        let tmp_p = d_tmp.device_ptr(&stream).0;
        let n_pi = np as i32;
        let cfg_s = LaunchConfig { grid_dim: ((ROWS * np) as u32, 1, 1), block_dim: (BLK as u32, 1, 1), shared_mem_bytes: 0 };
        let cfg_m = LaunchConfig { grid_dim: (ROWS as u32, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };
        let call_s = || {
            let mut b = stream.launch_builder(&f_split2);
            b.arg(&x_p).arg(&vocab_i).arg(&n_pi).arg(&rows_i).arg(&tmp_p);
            unsafe { b.launch(cfg_s) }.expect("split");
        };
        let call_m = || {
            let mut b = stream.launch_builder(&f_merge2);
            b.arg(&tmp_p).arg(&n_pi).arg(&rows_i).arg(&vocab_i).arg(&o_p);
            unsafe { b.launch(cfg_m) }.expect("merge");
        };
        call_s();
        call_m();
        stream.synchronize().unwrap();
        let mut o2 = vec![0f32; 2 * ROWS * K];
        stream.memcpy_dtoh(&d_out, &mut o2).unwrap();
        let bi = (0..ROWS * K).filter(|&i| (o2[ROWS * K + i] as i32) != h_idx[i]).count();
        let bv = (0..ROWS * K).filter(|&i| (o2[i] - h_vals[i]).abs() > 1e-3).count();
        let us2 = time(&|| { call_s(); call_m(); });
        let us_s2 = time(&call_s);
        let us_m2 = time(&call_m);
        println!("  np={np:3}: 合 {us2:7.3} µs(段一 {us_s2:6.3} + 段二 {us_m2:6.3})  索引差 {bi} 值差 {bv}");
        let _ = d_tmp.device_ptr(&stream);
    }

    // ---- E) merge 单核隔离(np 定 16,候选数 = np×16)----
    println!("=== E) merge 单核隔离(grid={ROWS})===");
    for &np in &[4usize, 16, 64] {
        let buf = ROWS * np * K;
        let mut d_pv = unsafe { stream.alloc::<f32>(buf) }.unwrap();
        let mut d_pi = unsafe { stream.alloc::<f32>(buf) }.unwrap();
        let pv_p = d_pv.device_ptr(&stream).0;
        let pi_p = d_pi.device_ptr(&stream).0;
        let n_pi = np as i32;
        let cfg_m = LaunchConfig { grid_dim: (ROWS as u32, 1, 1), block_dim: (BLK as u32, 1, 1), shared_mem_bytes: (2 * 4096 * 4) as u32 };
        let call = || {
            let mut b = stream.launch_builder(&f_merge);
            b.arg(&pv_p).arg(&pi_p).arg(&n_pi).arg(&rows_i).arg(&o_p);
            unsafe { b.launch(cfg_m) }.expect("merge");
        };
        println!("  候选 {n_pi:4}×16 = {:5}: {:.3} µs", np * K, time(&call));
        let _ = d_pv.device_ptr(&stream);
        let _ = d_pi.device_ptr(&stream);
    }
}

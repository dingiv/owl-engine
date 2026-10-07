//! B6.3 证可性探针:chunked prefill 的 fp8 e4m3 KV 读变体。
//!
//! 三线判据(与量化噪声解耦):
//! ① gpu_f16 vs host_f16 参考 —— 布局/读序自检(max < 2e-2,1 ulp 级);
//! ② gpu_fp8 vs host_fp8 参考 —— 核计算正确性硬门(max < 1e-2,贴地);
//! ③ f16 vs fp8 —— 量化噪声信息项(数据相关,只设粗罔 tripwire)。
//! 布局 = 生产 x=8(classic [nb, Hkv, hd/8, bs, 8]);hd128(生产 target)
//! 与 hd256 双形状同测;smem tile 路径(dominant-seq)为主,boundary
//! 路径单序列不触发(E2C 多序列场景另案)。
//! 无 GPU 跳过(OWL_TEST_DEVICE)。
//!
//! 史料(2026-10-10):smem 转换拷贝曾把 `__half` 直接赋给 uint16_t 槽
//! (隐式 float 数值截断,值<1→0/负→UB)= 99.6% 零输出真凶;修 =
//! `__half_as_ushort` 位拷贝。本探针即当时的复现用例。

use std::sync::Arc;
use cudarc::driver::{CudaContext, DevicePtr, LaunchConfig, PushKernelArg};

const PAGE: usize = 32;
const X: usize = 8;
const NUM_HEADS: usize = 8;
const NUM_KV_HEADS: usize = 2;
const NUM_BLOCKS: usize = 16;
const T: usize = 256; // 满 chunk(TOKEN_CHUNK_SIZE)
const SEQ_LEN: usize = 256; // 满 chunk:8 块

fn half_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let man = (bits >> 13) & 0x3FF;
    if exp <= 0 {
        return sign;
    }
    if exp >= 31 {
        return sign | 0x7C00;
    }
    sign | ((exp as u16) << 10) | man as u16
}

fn f16_to_f32(h: u16) -> f32 {
    let s = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let e = ((h >> 10) & 0x1F) as i32;
    let m = (h & 0x3FF) as f32;
    if e == 0 {
        return s * m * 2f32.powi(-24);
    }
    s * (1.0 + m / 1024.0) * 2f32.powi(e - 15)
}

fn e4m3_decode(b: u8) -> f32 {
    let s = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = ((b >> 3) & 0x0F) as i32;
    let m = (b & 0x07) as i32;
    if e == 15 && m == 7 {
        return f32::NAN;
    }
    let v = if e == 0 { (m as f32) * 2f32.powi(-9) } else { (1.0 + m as f32 / 8.0) * 2f32.powi(e - 7) };
    s * v
}

fn e4m3_encode(v: f32) -> u8 {
    if v.is_nan() {
        return 0x7F;
    }
    let mut best = 0u8;
    let mut best_err = f32::INFINITY;
    for code in 0..=255u8 {
        if code & 0x7F == 0x7F {
            continue;
        }
        let d = e4m3_decode(code);
        let err = (d - v).abs();
        if err < best_err || (err == best_err && code & 1 == 0 && best & 1 == 1) {
            best_err = err;
            best = code;
        }
    }
    best
}

fn lcg(n: usize, seed: u32) -> Vec<u16> {
    (0..n)
        .map(|i| {
            let x = seed.wrapping_mul(0x9E37_79B9).wrapping_add((i as u32).wrapping_mul(0x85EB_CA6B));
            half_bits(((x & 0xFFFF) as f32 / 65535.0 - 0.5) * 4.0)
        })
        .collect()
}

/// host 参考:标准 attention f32 逐式(kv_fp8 = 输入池是否 e4m3 字节)。
/// 布局与核同源:K x=8 转置 [hd/8, bs, 8],V 行主 [hd, bs],因果 mask,
/// 单序列块链 [0..SEQ_LEN/PAGE)。验证核计算正确性(与量化噪声解耦的硬门)。
fn host_attn_ref(q: &[u16], kcache: &[u8], vcache: &[u8], kv_fp8: bool, hd: usize) -> Vec<f32> {
    let dec = |pool: &[u8], i: usize| -> f32 {
        if kv_fp8 {
            e4m3_decode(pool[i])
        } else {
            f16_to_f32(pool[2 * i] as u16 | ((pool[2 * i + 1] as u16) << 8))
        }
    };
    let mut out = vec![0f32; T * NUM_HEADS * hd];
    let kbs = NUM_KV_HEADS * hd * PAGE; // 元素 stride(f16 按 2 字节换算)
    let khs = hd * PAGE;
    let scale = 1.0 / (hd as f32).sqrt();
    let mut qk = [0f32; SEQ_LEN];
    for t in 0..T {
        for h in 0..NUM_HEADS {
            let kvh = h / (NUM_HEADS / NUM_KV_HEADS);
            for b in 0..=t {
                let pb = b / PAGE;
                let bb = b % PAGE;
                let mut s = 0f32;
                for d in 0..hd {
                    let gy = d / X;
                    let gx = d % X;
                    let kidx = pb * kbs + kvh * khs + bb * X + gy * (PAGE * X) + gx;
                    s += f16_to_f32(q[(t * NUM_HEADS + h) * hd + d]) * dec(kcache, kidx);
                }
                qk[b] = s * scale;
            }
            let m = qk[..=t].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let l: f32 = qk[..=t].iter().map(|&x| (x - m).exp()).sum();
            for d in 0..hd {
                let mut acc = 0f32;
                for b in 0..=t {
                    let pb = b / PAGE;
                    let bb = b % PAGE;
                    let vidx = pb * kbs + kvh * khs + d * PAGE + bb;
                    acc += (qk[b] - m).exp() * dec(vcache, vidx);
                }
                out[(t * NUM_HEADS + h) * hd + d] = acc / l;
            }
        }
    }
    out
}

fn run_case(
    stream: &Arc<cudarc::driver::CudaStream>,
    m: &Arc<cudarc::driver::CudaModule>,
    hd: usize,
) {
    let pool_elems = NUM_BLOCKS * NUM_KV_HEADS * hd * PAGE;
    let k_f16 = lcg(pool_elems, 101 + hd as u32);
    let v_f16 = lcg(pool_elems, 202 + hd as u32);
    let q_f16 = lcg(T * NUM_HEADS * hd, 303 + hd as u32);
    let k_fp8: Vec<u8> = k_f16.iter().map(|&h| e4m3_encode(f16_to_f32(h))).collect();
    let v_fp8: Vec<u8> = v_f16.iter().map(|&h| e4m3_encode(f16_to_f32(h))).collect();

    let dev_u16 = |v: &[u16]| {
        let mut d = unsafe { stream.alloc::<u16>(v.len()) }.unwrap();
        stream.memcpy_htod(v, &mut d).unwrap();
        d
    };
    let dev_u8 = |v: &[u8]| {
        let mut d = unsafe { stream.alloc::<u8>(v.len()) }.unwrap();
        stream.memcpy_htod(v, &mut d).unwrap();
        d
    };
    let dev_f32 = |v: &[f32]| {
        let mut d = unsafe { stream.alloc::<f32>(v.len()) }.unwrap();
        stream.memcpy_htod(v, &mut d).unwrap();
        d
    };
    let d_q = dev_u16(&q_f16);
    let d_k16 = dev_u16(&k_f16);
    let d_v16 = dev_u16(&v_f16);
    let d_k8 = dev_u8(&k_fp8);
    let d_v8 = dev_u8(&v_fp8);
    // 单序列:块链 = [0..8](256 tok = 8 块满覆盖)
    let d_bt = dev_f32(&(0..8).map(|i| i as f32).collect::<Vec<_>>());
    let d_lens = dev_f32(&[SEQ_LEN as f32]);
    let d_qsl = dev_f32(&[0f32, T as f32]); // 单序列 [0, T]
    let d_alibi = dev_f32(&[0f32]);
    let d_sinks = dev_f32(&[0f32]);

    let run = |tag: &str,
               k_name: &str,
               k_ptr: u64,
               v_ptr: u64,
               out: &cudarc::driver::CudaSlice<u16>| {
        let f = m.load_function(k_name).unwrap_or_else(|e| panic!("[{tag}] load {k_name}: {e:?}"));
        let o_p = out.device_ptr(stream).0;
        let q_p = d_q.device_ptr(stream).0;
        let bt_p = d_bt.device_ptr(stream).0;
        let ln_p = d_lens.device_ptr(stream).0;
        let qsl_p = d_qsl.device_ptr(stream).0;
        let ab_p = d_alibi.device_ptr(stream).0;
        let sk_p = d_sinks.device_ptr(stream).0;
        let (nkv, scale, bts, ns, nqh, nqt) =
            (NUM_KV_HEADS as i32, 1f32 / (hd as f32).sqrt(), 1i32, 1i32, NUM_HEADS as i32, T as i32);
        let (ss, ost, sw, tnb, kbs, khs, ua, us) =
            (1f32, (NUM_HEADS * hd) as i32, -1i32, NUM_BLOCKS as i32,
             (NUM_KV_HEADS * hd * PAGE) as i32, (hd * PAGE) as i32, 0i32, 0i32);
        // extern 序:q,k_cache,v_cache,bt,seq_lens,qsl,alibi,sinks,
        // nkv,sm_scale,bts,ns,nqh,nqt,ss,ost,sw,tnb,kbs,khs,ua,us,out(末参)
        let mut b = stream.launch_builder(&f);
        b.arg(&q_p)
            .arg(&k_ptr)
            .arg(&v_ptr)
            .arg(&bt_p)
            .arg(&ln_p)
            .arg(&qsl_p)
            .arg(&ab_p)
            .arg(&sk_p)
            .arg(&nkv)
            .arg(&scale)
            .arg(&bts)
            .arg(&ns)
            .arg(&nqh)
            .arg(&nqt)
            .arg(&ss)
            .arg(&ost)
            .arg(&sw)
            .arg(&tnb)
            .arg(&kbs)
            .arg(&khs)
            .arg(&ua)
            .arg(&us)
            .arg(&o_p);
        let cfg = LaunchConfig {
            grid_dim: ((NUM_HEADS / NUM_KV_HEADS) as u32, NUM_KV_HEADS as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: (64 + 2 * hd * PAGE * 2) as u32,
        };
        if let Err(e) = unsafe { b.launch(cfg) } {
            panic!("[{tag}] chunked launch {k_name}: {e:?}");
        }
        if let Err(e) = stream.synchronize() {
            panic!("[{tag}] sync 后 ILLEGAL: {e:?}");
        }
        println!("[hd{hd} {tag}] launch+sync OK");
    };

    let d_out16 = dev_u16(&vec![0u16; T * NUM_HEADS * hd]);
    let d_out8 = dev_u16(&vec![0u16; T * NUM_HEADS * hd]);
    run(
        "f16",
        &format!("vllm_chunked_prefill_paged_attn_opt_f16_hd{hd}"),
        d_k16.device_ptr(stream).0,
        d_v16.device_ptr(stream).0,
        &d_out16,
    );
    run(
        "fp8",
        &format!("vllm_chunked_prefill_paged_attn_opt_fp8_hd{hd}"),
        d_k8.device_ptr(stream).0,
        d_v8.device_ptr(stream).0,
        &d_out8,
    );

    let mut o16 = vec![0u16; T * NUM_HEADS * hd];
    stream.memcpy_dtoh(&d_out16, &mut o16).unwrap();
    let mut o8 = vec![0u16; T * NUM_HEADS * hd];
    stream.memcpy_dtoh(&d_out8, &mut o8).unwrap();

    // host 参考(f32 逐式):与量化噪声解耦的硬门
    let k_f16_bytes: Vec<u8> = k_f16.iter().flat_map(|h| h.to_le_bytes()).collect();
    let v_f16_bytes: Vec<u8> = v_f16.iter().flat_map(|h| h.to_le_bytes()).collect();
    let ref_f16 = host_attn_ref(&q_f16, &k_f16_bytes, &v_f16_bytes, false, hd);
    let ref_fp8 = host_attn_ref(&q_f16, &k_fp8, &v_fp8, true, hd);

    let stat = |a: &[f32], b: &[f32]| -> (f32, f32, usize) {
        let mut mx = 0f32;
        let mut sum = 0f32;
        let mut amx = 0usize;
        for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
            let d = (x - y).abs();
            if d > mx {
                mx = d;
                amx = i;
            }
            sum += d;
        }
        (sum / a.len() as f32, mx, amx)
    };
    let to_f32 = |o: &[u16]| -> Vec<f32> { o.iter().map(|&h| f16_to_f32(h)).collect() };

    // ① gpu_f16 vs host_f16 = 布局/读序自检(超门 = host 模型错,非核错)
    let (m16, x16, a16) = stat(&to_f32(&o16), &ref_f16);
    println!("① hd{hd} gpu_f16 vs host_f16 — mean={m16:.6} max={x16:.6} @ {a16}");
    assert!(x16 < 2e-2, "hd{hd} f16 布局自检超门(max {x16})— host 模型或读序错");

    // ② gpu_fp8 vs host_fp8 = 核计算正确性硬门(e4m3→half 无舍入,差应 ≈ f16 输出 ulp)
    let (m8, x8, a8) = stat(&to_f32(&o8), &ref_fp8);
    let (t8, h8, d8) = (a8 / (NUM_HEADS * hd), (a8 / hd) % NUM_HEADS, a8 % hd);
    println!("② hd{hd} gpu_fp8 vs host_fp8 — mean={m8:.6} max={x8:.6} @ t={t8} h={h8} d={d8}");
    assert!(m8 < 2e-3 && x8 < 1e-2, "hd{hd} chunked fp8 对 host 参考超门(mean {m8}, max {x8})");

    // ③ f16 vs fp8 = 量化噪声信息项(数据相关,只作粗罔 tripwire)
    let (mn, xn, _) = stat(&to_f32(&o16), &to_f32(&o8));
    println!("③ hd{hd} f16 vs fp8 量化噪声 — mean={mn:.5} max={xn:.5}");
    assert!(mn < 5e-2 && xn < 3e-1, "hd{hd} f16/fp8 粗罔 tripwire(mean {mn}, max {xn})");
}

#[test]
fn chunked_prefill_fp8_matches_f16() {
    if std::env::var_os("OWL_TEST_DEVICE").is_none() {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    }
    let ctx = CudaContext::new(1).expect("ctx");
    ctx.bind_to_thread().expect("bind");
    let stream = ctx.default_stream();
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(
        owl_kernels::sources::attention::PREFILL_PAGED_ATTN_F16,
        cudarc::nvrtc::CompileOptions {
            arch: Some("compute_86"),
            include_paths: vec!["/usr/local/cuda/include".into()],
            ..Default::default()
        },
    )
    .expect("nvrtc compile prefill");
    let m = ctx.load_module(ptx).expect("module");

    for hd in [128usize, 256usize] {
        run_case(&stream, &m, hd);
    }
    println!("B6.3 探针 PASS(hd128/hd256 双形状):chunked fp8 读核计算贴地,f16 差 = 量化噪声");
}

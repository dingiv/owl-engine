//! kv布局统一契约 P2:chunked prefill kNHD 变体探针。
//!
//! 结构与 chunked_prefill_fp8_probe.rs 同款(host f32 逐式参考,因果 mask,
//! 满 chunk 256 tok / 8 块;dominant-seq smem tile 路径为主);差异 =
//! 池按 kNHD [nb, page, Hkv, hd] 落位,host 参考同寻址,kernel 名 _knhd,
//! kv_head_stride = hd(元素序)。判据:f16 vs host max < 2e-2;
//! fp8 vs host max < 1e-2。无 GPU 跳过(OWL_TEST_DEVICE;卡 = ordinal 1)。

use std::sync::Arc;
use cudarc::driver::{CudaContext, DevicePtr, LaunchConfig, PushKernelArg};

const PAGE: usize = 32;
const NUM_HEADS: usize = 8;
const NUM_KV_HEADS: usize = 2;
const NUM_BLOCKS: usize = 16;
const T: usize = 8; // 子块二分(2026-10-12 real_weights NaN 排查)
const SEQ_LEN: usize = 8; // 子块:1 块

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

/// host 参考(kNHD 寻址:[nb, page, hkv, hd];因果 mask;单序列块链恒等)
fn host_attn_ref_knhd(q: &[u16], kcache: &[u8], vcache: &[u8], kv_fp8: bool, hd: usize) -> Vec<f32> {
    let dec = |pool: &[u8], i: usize| -> f32 {
        if kv_fp8 {
            e4m3_decode(pool[i])
        } else {
            f16_to_f32(pool[2 * i] as u16 | ((pool[2 * i + 1] as u16) << 8))
        }
    };
    let kbs = PAGE * NUM_KV_HEADS * hd; // 元素 stride(两布局同值)
    let khs = hd; // kNHD head stride
    let scale = 1.0 / (hd as f32).sqrt();
    let mut out = vec![0f32; T * NUM_HEADS * hd];
    for t in 0..T {
        for h in 0..NUM_HEADS {
            let kvh = h / (NUM_HEADS / NUM_KV_HEADS);
            let mut qk = [0f32; SEQ_LEN];
            for b in 0..=t {
                let pb = b / PAGE;
                let bb = b % PAGE;
                let mut s = 0f32;
                for d in 0..hd {
                    let kidx = pb * kbs + bb * NUM_KV_HEADS * hd + kvh * hd + d;
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
                    let vidx = pb * kbs + bb * NUM_KV_HEADS * hd + kvh * hd + d;
                    acc += (qk[b] - m).exp() * dec(vcache, vidx);
                }
                out[(t * NUM_HEADS + h) * hd + d] = acc / l;
            }
        }
    }
    out
}

fn run_case(stream: &Arc<cudarc::driver::CudaStream>, m: &Arc<cudarc::driver::CudaModule>, hd: usize) {
    let pool_elems = NUM_BLOCKS * NUM_KV_HEADS * hd * PAGE;
    let k_f16 = lcg(pool_elems, 101 + hd as u32);
    let v_f16 = lcg(pool_elems, 202 + hd as u32);
    let q_f16 = lcg(T * NUM_HEADS * hd, 303 + hd as u32);
    let k_fp8: Vec<u8> = k_f16.iter().map(|&h| e4m3_encode(f16_to_f32(h))).collect();
    let v_fp8: Vec<u8> = v_f16.iter().map(|&h| e4m3_encode(f16_to_f32(h))).collect();

    // kNHD 落位:pool_n[knhd_off] = 值(classic 线性数组直接按 knhd 序排)
    let knhd_arrange = |lin: &[u16]| -> Vec<u16> {
        let mut p = vec![0u16; lin.len()];
        for s in 0..SEQ_LEN {
            for h in 0..NUM_KV_HEADS {
                for d in 0..hd {
                    let b = s / PAGE;
                    let bb = s % PAGE;
                    p[(b * PAGE + bb) * NUM_KV_HEADS * hd + h * hd + d] = lin[s * NUM_KV_HEADS * hd + h * hd + d];
                }
            }
        }
        p
    };
    let k_f16_n = knhd_arrange(&k_f16);
    let v_f16_n = knhd_arrange(&v_f16);
    let k_fp8_n: Vec<u8> = {
        let mut p = vec![0u8; pool_elems];
        for s in 0..SEQ_LEN {
            for h in 0..NUM_KV_HEADS {
                for d in 0..hd {
                    let b = s / PAGE;
                    let bb = s % PAGE;
                    let li = s * NUM_KV_HEADS * hd + h * hd + d;
                    p[(b * PAGE + bb) * NUM_KV_HEADS * hd + h * hd + d] = e4m3_encode(f16_to_f32(k_f16[li]));
                }
            }
        }
        p
    };
    let v_fp8_n: Vec<u8> = {
        let mut p = vec![0u8; pool_elems];
        for s in 0..SEQ_LEN {
            for h in 0..NUM_KV_HEADS {
                for d in 0..hd {
                    let b = s / PAGE;
                    let bb = s % PAGE;
                    let li = s * NUM_KV_HEADS * hd + h * hd + d;
                    p[(b * PAGE + bb) * NUM_KV_HEADS * hd + h * hd + d] = e4m3_encode(f16_to_f32(v_f16[li]));
                }
            }
        }
        p
    };

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
    let d_q = dev_u16(&q_f16);
    let d_k16 = dev_u16(&k_f16_n);
    let d_v16 = dev_u16(&v_f16_n);
    let d_k8 = dev_u8(&k_fp8_n);
    let d_v8 = dev_u8(&v_fp8_n);
    let d_bt = {
        let mut d = unsafe { stream.alloc::<f32>(8) }.unwrap();
        stream.memcpy_htod(&(0..8).map(|i| i as f32).collect::<Vec<_>>(), &mut d).unwrap();
        d
    };
    let d_lens = {
        let mut d = unsafe { stream.alloc::<f32>(1) }.unwrap();
        stream.memcpy_htod(&[SEQ_LEN as f32], &mut d).unwrap();
        d
    };
    let d_qsl = {
        let mut d = unsafe { stream.alloc::<f32>(2) }.unwrap();
        stream.memcpy_htod(&[0f32, T as f32], &mut d).unwrap();
        d
    };
    let d_alibi = {
        let mut d = unsafe { stream.alloc::<f32>(1) }.unwrap();
        stream.memcpy_htod(&[0f32], &mut d).unwrap();
        d
    };
    let d_sinks = {
        let mut d = unsafe { stream.alloc::<f32>(1) }.unwrap();
        stream.memcpy_htod(&[0f32], &mut d).unwrap();
        d
    };

    let run = |tag: &str, k_name: &str, k_ptr: u64, v_ptr: u64, out: &cudarc::driver::CudaSlice<u16>| {
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
             (PAGE * NUM_KV_HEADS * hd) as i32, (hd) as i32, 0i32, 0i32);
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
        &format!("vllm_chunked_prefill_paged_attn_opt_f16_knhd_hd{hd}"),
        d_k16.device_ptr(stream).0,
        d_v16.device_ptr(stream).0,
        &d_out16,
    );
    run(
        "fp8",
        &format!("vllm_chunked_prefill_paged_attn_opt_fp8_knhd_hd{hd}"),
        d_k8.device_ptr(stream).0,
        d_v8.device_ptr(stream).0,
        &d_out8,
    );

    let mut o16 = vec![0u16; T * NUM_HEADS * hd];
    stream.memcpy_dtoh(&d_out16, &mut o16).unwrap();
    let mut o8 = vec![0u16; T * NUM_HEADS * hd];
    stream.memcpy_dtoh(&d_out8, &mut o8).unwrap();

    let k_f16_bytes: Vec<u8> = k_f16_n.iter().flat_map(|h| h.to_le_bytes()).collect();
    let v_f16_bytes: Vec<u8> = v_f16_n.iter().flat_map(|h| h.to_le_bytes()).collect();
    let ref_f16 = host_attn_ref_knhd(&q_f16, &k_f16_bytes, &v_f16_bytes, false, hd);
    let ref_fp8 = host_attn_ref_knhd(&q_f16, &k_fp8_n, &v_fp8_n, true, hd);

    let maxdiff = |a: &[u16], r: &[f32]| -> (f32, f32, usize, usize, usize) {
        let mut mx = 0f32;
        let mut sum = 0f32;
        let mut at = 0usize;
        let mut ah = 0usize;
        let mut ad = 0usize;
        for (i, (&bits, &r)) in a.iter().zip(r.iter()).enumerate() {
            let g = f16_to_f32(bits);
            let dd = (g - r).abs();
            sum += dd;
            if dd > mx {
                mx = dd;
                at = i / (NUM_HEADS * hd);
                ah = (i / hd) % NUM_HEADS;
                ad = i % hd;
            }
        }
        (sum / a.len() as f32, mx, at, ah, ad)
    };
    let (m16, x16, t16, h16, d16) = maxdiff(&o16, &ref_f16);
    println!("① hd{hd} gpu_f16(knhd) vs host — mean={m16:.6} max={x16:.6} @ t={t16} h={h16} d={d16}");
    assert!(x16 < 2e-2, "hd{hd} chunked knhd f16 超门(max {x16})");
    let (m8, x8, t8, h8, d8) = maxdiff(&o8, &ref_fp8);
    println!("② hd{hd} gpu_fp8(knhd) vs host — mean={m8:.6} max={x8:.6} @ t={t8} h={h8} d={d8}");
    assert!(m8 < 2e-3 && x8 < 1e-2, "hd{hd} chunked knhd fp8 超门(mean {m8}, max {x8})");
}

#[test]
fn chunked_knhd_parity() {
    if !owl_shared::env_reader::flag("OWL_TEST_DEVICE") {
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
    .expect("nvrtc compile");
    let m = ctx.load_module(ptx).expect("module");
    for hd in [256usize, 128usize] {
        run_case(&stream, &m, hd);
    }
}

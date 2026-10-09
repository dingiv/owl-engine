//! TG=4 attention 变体证可性探针(2026-10-10 性能刀)。
//!
//! 对照 `chunked_prefill_fp8_probe.rs`(host f32 参考逐式对拍);本探针
//! 只测 **_tg4 变体**(64 token × 4 线程组,acc 64/线程;grid.z =
//! ceil(T/64))。判据同门:gpu_tg4 vs host_fp8 参考(max < 1e-2)。
//! 无 GPU 跳过(OWL_TEST_DEVICE)。

use std::sync::Arc;
use half::f16;
use cudarc::driver::{CudaContext, DevicePtr, LaunchConfig, PushKernelArg};

const PAGE: usize = 32;
const X: usize = 8;
const NUM_HEADS: usize = 8;
const NUM_KV_HEADS: usize = 2;
const NUM_BLOCKS: usize = 16;
const T: usize = 256;
const SEQ_LEN: usize = 256;

fn lcg(n: usize, seed: u32) -> Vec<u16> {
    let mut x = seed;
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            // f16 [0.25, 1.5):指数 14/15,免 denormal/inf
            let e = 0x3C00u16 | ((x >> 7) & 0x0380) as u16;
            e | ((x >> 10) & 0x007F) as u16
        })
        .collect()
}

fn e4m3_encode(v: f32) -> u8 {
    // 粗 e4m3 编码(探针量级 ∈ [±448] 内;位型近似足够 —— 对拍同源)
    if v == 0.0 {
        return 0;
    }
    let s = if v < 0.0 { 0x80u8 } else { 0u8 };
    let a = v.abs();
    let mut e = 7i32;
    let mut m = a;
    while m >= 2.0 {
        m /= 2.0;
        e += 1;
    }
    while m < 1.0 {
        m *= 2.0;
        e -= 1;
    }
    let man = ((m - 1.0) * 8.0).round() as i32;
    let (man, e) = if man == 8 { (0, e + 1) } else { (man, e) };
    if !(1..=15).contains(&e) {
        return s | 0x7F;
    }
    s | (((e + 7) as u8) << 3) | (man as u8) & 0x7
}

fn e4m3_decode(b: u8) -> f32 {
    let s = if b & 0x80 != 0 { -1f32 } else { 1f32 };
    let e = ((b >> 3) & 0x0F) as i32 - 7;
    let m = (b & 0x07) as f32 / 8.0;
    if (b & 0x7F) == 0x7F {
        return f32::NAN;
    }
    s * (1.0 + m) * (2f32).powi(e)
}

fn f16_bytes_to_f32(pool: &[u8], i: usize) -> f32 {
    let h = pool[2 * i] as u16 | ((pool[2 * i + 1] as u16) << 8);
    f16::from_bits(h).to_f32()
}

// host 逐式参考(同 chunked_prefill_fp8_probe)
fn host_attn_ref(q: &[u16], kcache: &[u8], vcache: &[u8], kv_fp8: bool, hd: usize) -> Vec<f32> {
    let dec = |pool: &[u8], i: usize| -> f32 {
        if kv_fp8 {
            e4m3_decode(pool[i])
        } else {
            f16_bytes_to_f32(pool, i)
        }
    };
    let mut out = vec![0f32; T * NUM_HEADS * hd];
    let kbs = NUM_KV_HEADS * hd * PAGE;
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
                    let qv = f16::from_bits(q[(t * NUM_HEADS + h) * hd + d]).to_f32();
                    s += qv * dec(kcache, kidx);
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

#[test]
fn tg4_fp8_hd256_matches_host() {
    if !owl_shared::env_reader::flag("OWL_TEST_DEVICE") {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    }
    // WIP(2026-10-09):tg4 变体挂死 + 数值超门(见 §二十七 优化战报)
    // —— 立案施工中,默认跳过防挂 CI;OWL_TG4_PROBE=1 显式启用
    if !owl_shared::env_reader::flag("OWL_TG4_PROBE") {
        eprintln!("skip: OWL_TG4_PROBE 未设(tg4 变体 WIP)");
        return;
    }
    let hd = 256usize;
    let ctx = CudaContext::new(owl_shared::env_reader::parse::<usize>("OWL_TEST_DEVICE_ORDINAL")
    .unwrap_or(1))
    .expect("ctx");
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
    .expect("nvrtc");
    let m = ctx.load_module(ptx).expect("module");

    let pool_elems = NUM_BLOCKS * NUM_KV_HEADS * hd * PAGE;
    let k_f16 = lcg(pool_elems, 101 + hd as u32);
    let v_f16 = lcg(pool_elems, 202 + hd as u32);
    let q_f16 = lcg(T * NUM_HEADS * hd, 303 + hd as u32);
    let k_fp8: Vec<u8> = k_f16.iter().map(|&h| e4m3_encode(f16::from_bits(h).to_f32())).collect();
    let v_fp8: Vec<u8> = v_f16.iter().map(|&h| e4m3_encode(f16::from_bits(h).to_f32())).collect();

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
    let d_k16 = dev_u16(&k_f16);
    let d_v16 = dev_u16(&v_f16);
    let d_k8 = dev_u8(&k_fp8);
    let d_v8 = dev_u8(&v_fp8);
    let d_bt = {
        let mut d = unsafe { stream.alloc::<f32>(NUM_BLOCKS) }.unwrap();
        stream
            .memcpy_htod(&(0..NUM_BLOCKS).map(|i| i as f32).collect::<Vec<_>>(), &mut d)
            .unwrap();
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
    let d_z = {
        let mut d = unsafe { stream.alloc::<f32>(1) }.unwrap();
        d
    };
    let mut d_out = dev_u16(&vec![0u16; T * NUM_HEADS * hd]);
    let mut d_out_f16 = dev_u16(&vec![0u16; T * NUM_HEADS * hd]);

    let f16_name = "vllm_chunked_prefill_paged_attn_opt_f16_hd256_tg4";
    let fp8_name = "vllm_chunked_prefill_paged_attn_opt_fp8_hd256_tg4";
    let f = m.load_function(fp8_name).expect("load tg4 fp8");
    let f_f16 = m.load_function(f16_name).expect("load tg4 f16");
    let q_p = d_q.device_ptr(&stream).0;
    let k_p = d_k8.device_ptr(&stream).0;
    let v_p = d_v8.device_ptr(&stream).0;
    let k16_p = d_k16.device_ptr(&stream).0;
    let v16_p = d_v16.device_ptr(&stream).0;
    let bt_p = d_bt.device_ptr(&stream).0;
    let ln_p = d_lens.device_ptr(&stream).0;
    let qsl_p = d_qsl.device_ptr(&stream).0;
    let z_p = d_z.device_ptr(&stream).0;
    let o_p = d_out.device_ptr(&stream).0;
    let (nkv, scale, bts, ns, nqh, nqt) =
        (NUM_KV_HEADS as i32, 1f32 / (hd as f32).sqrt(), 1i32, 1i32, NUM_HEADS as i32, T as i32);
    let (ss, ost, sw, tnb, kbs, khs, ua, us) =
        (1f32, (NUM_HEADS * hd) as i32, -1i32, NUM_BLOCKS as i32,
         (NUM_KV_HEADS * hd * PAGE) as i32, (hd * PAGE) as i32, 0i32, 0i32);
    let mut b = stream.launch_builder(&f);
    b.arg(&q_p)
        .arg(&k_p)
        .arg(&v_p)
        .arg(&bt_p)
        .arg(&ln_p)
        .arg(&qsl_p)
        .arg(&z_p)
        .arg(&z_p)
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
    let grid_z = ((T + 63) / 64) as u32;
    let cfg = LaunchConfig {
        grid_dim: ((NUM_HEADS / NUM_KV_HEADS) as u32, NUM_KV_HEADS as u32, grid_z),
        block_dim: (256, 1, 1),
        shared_mem_bytes: (64 + 2 * hd * PAGE * 2) as u32,
    };
    // f16 臂(逻辑对照;KV_FP8=false 特化)
    {
        let mut b = stream.launch_builder(&f_f16);
        b.arg(&q_p)
            .arg(&k16_p)
            .arg(&v16_p)
            .arg(&bt_p)
            .arg(&ln_p)
            .arg(&qsl_p)
            .arg(&z_p)
            .arg(&z_p)
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
        if let Err(e) = unsafe { b.launch(cfg) } {
            panic!("[tg4-f16] launch: {e:?}");
        }
        stream.synchronize().unwrap();
        let mut o = vec![0u16; T * NUM_HEADS * hd];
        stream.memcpy_dtoh(&d_out, &mut o).unwrap();
        let k_f16_bytes: Vec<u8> = k_f16.iter().flat_map(|h| h.to_le_bytes()).collect();
        let v_f16_bytes: Vec<u8> = v_f16.iter().flat_map(|h| h.to_le_bytes()).collect();
        let ref_f16 = host_attn_ref(&q_f16, &k_f16_bytes, &v_f16_bytes, false, hd);
        let got: Vec<f32> = o.iter().map(|&h| f16::from_bits(h).to_f32()).collect();
        let (mut mx, mut amx) = (0f32, 0usize);
        for (i, (&x, &y)) in got.iter().zip(&ref_f16).enumerate() {
            let dd = (x - y).abs();
            if dd > mx { mx = dd; amx = i; }
        }
        let t_i = amx / (NUM_HEADS * hd);
        println!("[tg4-f16] vs host_f16 — max={mx:.6} @ t={t_i} h={} d={}", (amx / hd) % NUM_HEADS, amx % hd);
        // 清零 out 给 fp8 臂
        let zeros = vec![0u16; T * NUM_HEADS * hd];
        stream.memcpy_htod(&zeros, &mut d_out).unwrap();
    }
    if let Err(e) = unsafe { b.launch(cfg) } {
        panic!("[tg4] launch: {e:?}");
    }
    if let Err(e) = stream.synchronize() {
        panic!("[tg4] sync 后 ILLEGAL: {e:?}");
    }
    println!("[tg4] launch+sync OK(grid.z={grid_z})");

    let mut o = vec![0u16; T * NUM_HEADS * hd];
    stream.memcpy_dtoh(&d_out, &mut o).unwrap();
    let ref_fp8 = host_attn_ref(&q_f16, &k_fp8, &v_fp8, true, hd);
    let to_f32 = |o: &[u16]| -> Vec<f32> {
        o.iter().map(|&h| f16::from_bits(h).to_f32()).collect()
    };
    let pairs: Vec<(usize, f32, f32)> = to_f32(&o)
        .iter()
        .zip(&ref_fp8)
        .enumerate()
        .map(|(i, (&x, &y))| (i, x, y))
        .collect();
    let mut errs: Vec<(usize, f32)> = pairs
        .iter()
        .map(|(i, x, y)| (*i, (*x - *y).abs()))
        .collect();
    errs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for &(i, e) in errs.iter().take(8) {
        let (t_i, rem) = (i / (NUM_HEADS * hd), i % (NUM_HEADS * hd));
        println!("[tg4] err top: i={i} t={t_i} h={} d={} |Δ|={e:.4}", rem / hd, rem % hd);
    }
    let mut per_t: std::collections::HashMap<usize, usize> = Default::default();
    for &(i, e) in errs.iter() {
        if e > 1e-2 {
            *per_t.entry(i / (NUM_HEADS * hd)).or_insert(0) += 1;
        }
    }
    let mut bad_ts: Vec<(usize, usize)> = per_t.into_iter().collect();
    bad_ts.sort_by(|a, b| b.1.cmp(&a.1));
    let total_bad: usize = bad_ts.iter().map(|(_, c)| c).sum();
    println!("[tg4] 错误 token 数={} 错误元素数={}", bad_ts.len(), total_bad);
    for (t_i, cnt) in bad_ts.iter().take(6) {
        println!("[tg4]   t={t_i} 错误元素={cnt}");
    }
    let (mut mx, mut sum, mut amx) = (0f32, 0f32, 0usize);
    for (i, (_, e)) in errs.iter().enumerate() {
        if *e > mx { mx = *e; amx = i; }
        sum += e;
    }
    let mean = sum / ref_fp8.len() as f32;
    let t_i = amx / (NUM_HEADS * hd);
    let h_i = (amx / hd) % NUM_HEADS;
    println!("[tg4] vs host_fp8 — mean={mean:.6} max={mx:.6} @ t={t_i} h={h_i} d={}", amx % hd);
    assert!(
        mean < 2e-3 && mx < 1e-2,
        "tg4 对 host 参考超门(mean {mean}, max {mx})"
    );
}

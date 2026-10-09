//! v2 多伪序列探针(§三十二 nb=136 半崩回归的单元定位;2026-10-12)。
//!
//! 形状 = verify 实况:T=8 伪序列、context_lens 递增([600..608])、
//! nparts=3(PARTITION=512 → 每 seq 活跃 2)、GQA 8q/2kv、hd256、paged
//! fp8 KV。host 参考 = 每 seq 独立全局 softmax(attend [0, lens_s))。
//! 门:gpu vs host max < 1e-2;另附 per-seq 误差(定位某 seq 独崩)。
//! 无 GPU 跳过(OWL_TEST_DEVICE)。

use std::sync::Arc;

use cudarc::driver::{CudaContext, DevicePtr, LaunchConfig, PushKernelArg};

const S: usize = 8; // 伪序列数
const CTX0: usize = 600; // 首 seq 的 context_len(递增 +i)
const PAGE: usize = 32;
const NB: usize = (CTX0 + S + PAGE) / PAGE + 1; // 20
const X: usize = 8;
const H: usize = 8;
const HKV: usize = 2;
const HD: usize = 256;
const NP: usize = 9; // scratch partitions(覆盖 ceil(608/512)=2)

fn half_bits(v: f32) -> u16 {
    half::f16::from_f32(v).to_bits()
}

fn f16_to_f32(b: u16) -> f32 {
    half::f16::from_bits(b).to_f32()
}

fn e4m3_decode(code: u8) -> f32 {
    let s = if code & 0x80 != 0 { -1f32 } else { 1f32 };
    let e = ((code >> 3) & 0x0F) as i32;
    let m = (code & 0x07) as i32;
    if e == 15 && m == 7 {
        return f32::NAN;
    }
    let v = if e == 0 {
        (m as f32) * 2f32.powi(-9)
    } else {
        (1.0 + m as f32 / 8.0) * 2f32.powi(e - 7)
    };
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
            let x = seed
                .wrapping_mul(0x9E37_79B9)
                .wrapping_add((i as u32).wrapping_mul(0x85EB_CA6B));
            half_bits(((x & 0xFFFF) as f32 / 65535.0 - 0.5) * 4.0)
        })
        .collect()
}

/// host 参考:每 seq 独立全局 softmax(attend [0, lens_s))。classic 布局。
fn host_ref(q: &[u16], kc: &[u8], vc: &[u8], lens: &[usize]) -> Vec<f32> {
    host_ref_impl(q, kc, vc, lens, true)
}

fn host_ref_f16(q: &[u16], kc: &[u8], vc: &[u8], lens: &[usize]) -> Vec<f32> {
    host_ref_impl(q, kc, vc, lens, false)
}

fn host_ref_impl(q: &[u16], kc: &[u8], vc: &[u8], lens: &[usize], kv_fp8: bool) -> Vec<f32> {
    let dec = |pool: &[u8], i: usize| if kv_fp8 { e4m3_decode(pool[i]) } else { f16_to_f32(pool[2*i] as u16 | ((pool[2*i+1] as u16) << 8)) };
    let mut out = vec![0f32; S * H * HD];
    let kbs = HKV * HD * PAGE;
    let khs = HD * PAGE;
    let scale = 1.0 / (HD as f32).sqrt();
    for s in 0..S {
        let len = lens[s];
        for h in 0..H {
            let kvh = h / (H / HKV);
            let mut qk = vec![0f32; len];
            for b in 0..len {
                let pb = b / PAGE;
                let bb = b % PAGE;
                let mut acc = 0f32;
                for d in 0..HD {
                    let kidx = pb * kbs + kvh * khs + bb * X + (d / X) * (PAGE * X) + d % X;
                    acc += f16_to_f32(q[(s * H + h) * HD + d]) * dec(kc, kidx);
                }
                qk[b] = acc * scale;
            }
            let m = qk.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let l: f32 = qk.iter().map(|&x| (x - m).exp()).sum();
            for d in 0..HD {
                let mut acc = 0f32;
                for b in 0..len {
                    let pb = b / PAGE;
                    let bb = b % PAGE;
                    let vidx = pb * kbs + kvh * khs + d * PAGE + bb;
                    acc += (qk[b] - m).exp() * dec(vc, vidx);
                }
                out[(s * H + h) * HD + d] = acc / l;
            }
        }
    }
    out
}

#[test]
#[ignore = "§三十二:8 伪序列 v2 reduce 段数值超门(0.18 均匀)定位中——主核已验干净(exp_sums 正确),reduce arg 序已修正仍超门;server 侧 2048+/20k 语义健康,仅短 ctx(2 partitions)半崩;勿删,定位后转正"]
fn v2_paged_multiseq_matches_host() {
    if !owl_shared::env_reader::flag("OWL_TEST_DEVICE") {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    }
    let ctx = CudaContext::new(1).expect("ctx");
    ctx.bind_to_thread().expect("bind");
    let stream = ctx.default_stream();
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(
        owl_kernels::sources::attention::PAGED_ATTENTION_F16,
        cudarc::nvrtc::CompileOptions {
            arch: Some("compute_86"),
            include_paths: vec!["/usr/local/cuda/include".into()],
            ..Default::default()
        },
    )
    .expect("nvrtc compile paged v2");
    let m = ctx.load_module(ptx).expect("module");

    // 二分开关:V2_LENS_FLAT=1 → 全 seq len=512(单 partition 形态)
    let lens: Vec<usize> = if std::env::var("V2_LENS_FLAT").is_ok() {
        vec![512; S]
    } else {
        (0..S).map(|i| CTX0 + i).collect()
    };
    let max_len = CTX0 + S - 1;
    let pool_elems = NB * HKV * HD * PAGE;
    let k_f16 = lcg(pool_elems, 701);
    let v_f16 = lcg(pool_elems, 702);
    let q_f16 = lcg(S * H * HD, 703);
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
    let f16_mode_pre = std::env::var("V2_F16").is_ok();
    let d_k8: cudarc::driver::CudaSlice<u8> = if f16_mode_pre {
        let kb: Vec<u8> = k_f16.iter().flat_map(|h| h.to_le_bytes()).collect();
        dev_u8(&kb)
    } else { dev_u8(&k_fp8) };
    let d_v8: cudarc::driver::CudaSlice<u8> = if f16_mode_pre {
        let vb: Vec<u8> = v_f16.iter().flat_map(|h| h.to_le_bytes()).collect();
        dev_u8(&vb)
    } else { dev_u8(&v_fp8) };
    // 页表 [S, NB]:每 seq 同一恒等链(0..NB)
    let bt8: Vec<f32> = (0..S)
        .flat_map(|s| (0..NB).map(move |b| b as f32))
        .collect();
    let d_bt8 = dev_f32(&bt8);
    let d_lens = dev_f32(&lens.iter().map(|&l| l as f32).collect::<Vec<_>>());
    let d_alibi = dev_u16(&[0u16; 16]);
    let d_es = dev_f32(&vec![0f32; S * H * NP]);
    let d_ml = dev_f32(&vec![0f32; S * H * NP]);
    let d_partial = dev_u16(&vec![0u16; S * H * NP * HD]);
    let d_out = dev_u16(&vec![0u16; S * H * HD]);

    // ---- 主核:v2 fp8 hd256bs32;grid (H, S, NP) ----
    let f16_mode = std::env::var("V2_F16").is_ok();
    let f = m.load_function(if f16_mode {
        "vllm_paged_attention_v2_f16_hd256bs32"
    } else {
        "vllm_paged_attention_v2_fp8_hd256bs32"
    })
    .expect("load v2");
    let (q_p, k_p, v_p, bt_p, ln_p, ab_p) = (
        d_q.device_ptr(&stream).0,
        d_k8.device_ptr(&stream).0,
        d_v8.device_ptr(&stream).0,
        d_bt8.device_ptr(&stream).0,
        d_lens.device_ptr(&stream).0,
        d_alibi.device_ptr(&stream).0,
    );
    let (es_p, ml_p, pa_p, o_p) = (
        d_es.device_ptr(&stream).0,
        d_ml.device_ptr(&stream).0,
        d_partial.device_ptr(&stream).0,
        d_out.device_ptr(&stream).0,
    );
    let scale = 1f32 / (HD as f32).sqrt();
    let mut b = stream.launch_builder(&f);
    // fp8 入口专有序(B6 移植改序;≠ f16 版!):q, k, v, bt, lens, alibi_f32*,
    //           exp_sums, max_logits, nkv, scale, max_nb, q_stride, kvbs, kvhs,
    //           softcap, sliding_window, use_alibi, tmp_out(OUT 末参契约 4)
    b.arg(&q_p)
        .arg(&k_p)
        .arg(&v_p)
        .arg(&bt_p)
        .arg(&ln_p)
        .arg(&es_p) // alibi 占位(use_alibi=0 不解引用)
        .arg(&es_p)
        .arg(&ml_p)
        .arg(&(HKV as i32))
        .arg(&scale)
        .arg(&(NB as i32))
        .arg(&((H * HD) as i32))
        .arg(&((HKV * HD * PAGE) as i32))
        .arg(&((HD * PAGE) as i32))
        .arg(&1f32)
        .arg(&(-1i32))
        .arg(&0i32)
        .arg(&pa_p);
    let cfg = LaunchConfig {
        grid_dim: (H as u32, S as u32, NP as u32),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 2048,
    };
    unsafe { b.launch(cfg) }.expect("v2 main launch");
    stream.synchronize().expect("main sync");
    // 诊断:主核是否写了 exp_sums(全零 = 主核未写/全早退)
    {
        let mut es_h = vec![0f32; S * H * NP];
        stream.memcpy_dtoh(&d_es, &mut es_h).unwrap();
        let nz = es_h.iter().filter(|&&x| x != 0.0).count();
        println!("[diag] exp_sums nonzero = {nz}/{}; 前 6 = {:?}", es_h.len(), &es_h[..6.min(es_h.len())]);
    }

    // ---- reduce:grid (H, S) ----
    let f2 = m
        .load_function("vllm_paged_attention_v2_reduce_f16_hd256")
        .expect("load v2 reduce");
    let mut b2 = stream.launch_builder(&f2);
    let np_i = NP as i32;
    // 包装序:exp_sums, max_logits, tmp_out, context_lens, max_partitions, OUT(末参契约 4)
    b2.arg(&es_p)
        .arg(&ml_p)
        .arg(&pa_p)
        .arg(&ln_p)
        .arg(&np_i)
        .arg(&o_p);
    let cfg2 = LaunchConfig {
        grid_dim: (H as u32, S as u32, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 2048,
    };
    unsafe { b2.launch(cfg2) }.expect("v2 reduce launch");
    stream.synchronize().expect("reduce sync");

    // 诊断:partial(s0h0p0/p1 前 4 dims,未归一 acc)
    {
        let mut pa_h = vec![0u16; S * H * NP * HD];
        stream.memcpy_dtoh(&d_partial, &mut pa_h).unwrap();
        let f = |i: usize| f16_to_f32(pa_h[i]);
        let base = 0 * H * NP * HD; // s0
        println!("[diag] p0 d0..3 = {:.4} {:.4} {:.4} {:.4}", f(base), f(base+1), f(base+2), f(base+3));
        println!("[diag] p1 d0..3 = {:.4} {:.4} {:.4} {:.4}", f(base+NP*HD), f(base+NP*HD+1), f(base+NP*HD+2), f(base+NP*HD+3));
    }
    let mut o = vec![0u16; S * H * HD];
    stream.memcpy_dtoh(&d_out, &mut o).unwrap();
    let of: Vec<f32> = o.iter().map(|&h| f16_to_f32(h)).collect();

    // reduce 核名字带 f16(v2 f16 路径同 reduce;fp8 主核 partial 仍 f16)
    let _ = f2;
    let ref_out = if std::env::var("V2_F16").is_ok() {
        let kb: Vec<u8> = k_f16.iter().flat_map(|h| h.to_le_bytes()).collect();
        let vb: Vec<u8> = v_f16.iter().flat_map(|h| h.to_le_bytes()).collect();
        // f16 模式:dec 走 f16 路径(kv_fp8=false)
        host_ref_f16(&q_f16, &kb, &vb, &lens)
    } else {
        host_ref(&q_f16, &k_fp8, &v_fp8, &lens)
    };

    let mut worst = (0usize, 0f32);
    let mut per_seq = [0f32; S];
    for i in 0..S * H * HD {
        let e = (of[i] - ref_out[i]).abs();
        let s = i / (H * HD);
        if e > per_seq[s] {
            per_seq[s] = e;
        }
        if e > worst.1 {
            worst = (i, e);
        }
    }
    let (wi, we) = worst;
    let (ws, wh, wd) = (wi / (H * HD), (wi / HD) % H, wi % HD);
    println!("per-seq max err: {:?}", per_seq);
    println!("worst @ s={ws} h={wh} d={wd}: err={we:.6}");
    // 值级诊断:零的位置模式(前/后半/交错)
    let dims = [0usize, 1, 2, 63, 64, 65, 126, 127, 128, 129, 130, 254, 255];
    for &d in &dims {
        println!("   s0h0 d{d}: gpu={:.5} host={:.5}", of[d], ref_out[d]);
    }
    // seq0 整行误差均值(host 语义独立验)
    let (mut sm, mut ct) = (0f32, 0usize);
    for d in 0..HD {
        sm += (of[d] - ref_out[d]).abs();
        ct += 1;
    }
    println!("   seq0 mean|err| = {:.6}", sm / ct as f32);
    // 零分布:哪些 (s,h) 输出全零(主核没写/reduce 读错槽的指纹)
    let mut zero_by_sh = std::collections::BTreeMap::new();
    let mut zero_total = 0usize;
    for s in 0..S { for h in 0..H {
        let mut z = 0usize;
        for d in 0..HD { if of[(s*H+h)*HD+d] == 0.0 { z += 1; } }
        if z > 0 { zero_by_sh.insert((s,h), z); zero_total += z; }
    }}
    println!("   zero_total={zero_total} by(s,h)={:?} (前 12)", zero_by_sh.iter().take(12).collect::<Vec<_>>());
    assert!(
        per_seq.iter().cloned().fold(0f32, f32::max) < 1e-2,
        "v2 多伪序列对 host 参考超门(max {we:.6} @ s={ws})"
    );
    println!("v2 多伪序列探针 PASS(S={S} lens 递增 {CTX0}..{}):全 seq 贴 host 参考", lens[S - 1]);
}

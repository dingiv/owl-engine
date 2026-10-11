//! XQA **decode 形状**(q=1)速度/数值探针 —— 与现役 v2 分页核(seqs=1)同机同 ctx 头对头。
//!
//! 来源与契约:`crates/kernels/cu/xqa/README.md`(移植自
//! repos/vllm-ampere-optimized/flashampere,Ampere 闭包)。
//!
//! 形状 = owl 裸 decode 实况:batch=1,单 query 行(位置 = ctx−1,可见 [0, ctx)),
//! GQA 24q/4kv,hd256,page 32,kNHD 布局 [page, token, hkv, hd] 的 **fp8 e4m3** 池。
//! 与 `xqa_verify_probe.rs` 的区别:**q_seq_len=1、无因果块、掩码只有一个恒等的 bit0**。
//!
//! ⚠️ 本探针第一目的 = **D1 备选实测**:现有 prebuilt `libowl_xqa.a` 是
//! `SPEC_DEC=1` 实例化(`mha.cu:84` 把 `nbValidRows` 钉死为 `rowsPerBlock=32`);
//! 若 q=1 走它数值正确/速度可用,则裸 decode 不必再编第二份 `mha.cu`;
//! 若慢(32 行里只有 6 行有效 → 5.3× MMA 浪费),则编 `SPEC_DEC=0`(rows=6)第二份。
//!
//! 三条判据:
//!   1) 数值:XQA vs host 参考(单行全局 softmax)误差 < 1e-2;
//!   2) 速度:XQA 单发 vs v2(主核 + reduce 两发)per-call ms,同 ctx;
//!   3) 交叉:XQA vs v2 输出 max diff(v2 ≡ host 已由 xqa_verify_probe 证明)。
//!
//! 运行:`OWL_TEST_DEVICE=1 cargo test --release -p owl-kernels --features xqa \
//!        --test xqa_decode_probe -- --test-threads=1 --nocapture`
#![cfg(feature = "xqa")]

use std::time::Instant;

use cudarc::driver::{CudaContext, DevicePtr, LaunchConfig, PushKernelArg};

const PAGE: usize = 32;
const HKV: usize = 4;
const H: usize = 24;
const HD: usize = 256;
const SM86_SM_COUNT: u32 = 84; // 3090 Ti

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
    let v = if e == 0 { (m as f32) * 2f32.powi(-9) } else { (1.0 + m as f32 / 8.0) * 2f32.powi(e - 7) };
    s * v
}
fn e4m3_encode(v: f32) -> u8 {
    let mut best = 0u8;
    let mut best_err = f32::INFINITY;
    for code in 0..=255u8 {
        if code & 0x7F == 0x7F {
            continue;
        }
        let err = (e4m3_decode(code) - v).abs();
        if err < best_err {
            best_err = err;
            best = code;
        }
    }
    best
}
fn lcg_u8(n: usize, seed: u32) -> Vec<u8> {
    (0..n)
        .map(|i| {
            let x = seed
                .wrapping_mul(0x9E37_79B9)
                .wrapping_add((i as u32).wrapping_mul(0x85EB_CA6B));
            let f = ((x & 0xFFFF) as f32 / 65535.0 - 0.5) * 2.0;
            e4m3_encode(f)
        })
        .collect()
}
fn lcg_f16(n: usize, seed: u32) -> Vec<u16> {
    (0..n)
        .map(|i| {
            let x = seed
                .wrapping_mul(0x9E37_79B9)
                .wrapping_add((i as u32).wrapping_mul(0x85EB_CA6B));
            half_bits(((x & 0xFFFF) as f32 / 65535.0 - 0.5) * 4.0)
        })
        .collect()
}

/// host 参考:kNHD + fp8 池,单行(位置 ctx−1)可见 [0, ctx)。
fn host_ref_knhd(q: &[u16], kc: &[u8], vc: &[u8], len: usize) -> Vec<f32> {
    let mut out = vec![0f32; H * HD];
    let kbs = HKV * HD * PAGE;
    let scale = 1.0 / (HD as f32).sqrt();
    for h in 0..H {
        let kvh = h / (H / HKV);
        let mut qk = vec![0f32; len];
        for b in 0..len {
            let (pb, bb) = (b / PAGE, b % PAGE);
            let mut acc = 0f32;
            for d in 0..HD {
                let kidx = pb * kbs + bb * (HKV * HD) + kvh * HD + d;
                acc += f16_to_f32(q[h * HD + d]) * e4m3_decode(kc[kidx]);
            }
            qk[b] = acc * scale;
        }
        let m = qk.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let l: f32 = qk.iter().map(|&x| (x - m).exp()).sum();
        for d in 0..HD {
            let mut acc = 0f32;
            for b in 0..len {
                let (pb, bb) = (b / PAGE, b % PAGE);
                let vidx = pb * kbs + bb * (HKV * HD) + kvh * HD + d;
                acc += (qk[b] - m).exp() * e4m3_decode(vc[vidx]);
            }
            out[h * HD + d] = acc / l;
        }
    }
    out
}

extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn owl_xqa_launch(
        sm_count: u32,
        nb_k_heads: u32,
        sliding_win_size: u32,
        q_scale: f32,
        output: *mut core::ffi::c_void,
        q: *const core::ffi::c_void,
        k_cache: *mut core::ffi::c_void,
        v_cache: *mut core::ffi::c_void,
        page_table: *const i32,
        max_seq_len: u32,
        seq_lens: *const u32,
        batch_size: u32,
        kv_scale: f32,
        q_seq_len: u32,
        mask: *const u32,
        semaphores: *mut u32,
        scratch: *mut core::ffi::c_void,
        kv_stride_page: u64,
        kv_stride_token: u64,
        kv_stride_head: u64,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    /// 第二份实例化(`SPEC_DEC=1 + M_TILESIZE=16`,写给 q=1 的窄 tile):
    /// 签名与 `owl_xqa_launch` 一致。
    #[allow(clippy::too_many_arguments)]
    fn owl_xqa_launch_decode(
        sm_count: u32,
        nb_k_heads: u32,
        sliding_win_size: u32,
        q_scale: f32,
        output: *mut core::ffi::c_void,
        q: *const core::ffi::c_void,
        k_cache: *mut core::ffi::c_void,
        v_cache: *mut core::ffi::c_void,
        page_table: *const i32,
        max_seq_len: u32,
        seq_lens: *const u32,
        batch_size: u32,
        kv_scale: f32,
        q_seq_len: u32,
        mask: *const u32,
        semaphores: *mut u32,
        scratch: *mut core::ffi::c_void,
        kv_stride_page: u64,
        kv_stride_token: u64,
        kv_stride_head: u64,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

/// q_seq_len(1 = decode 目标形状;>1 = 顺便交叉验证 spec-dec 形状仍可用)
/// `decode_inst`:true → 走 `SPEC_DEC=0` 的第二份实例化(仅 q=1 有意义)
fn run_case(ctx: usize, q_seq_len: usize, decode_inst: bool, check: bool) {
    let nb = ctx.div_ceil(PAGE);
    let nparts = (nb * PAGE).div_ceil(512);
    let ctx_c = CudaContext::new(1).expect("ctx");
    ctx_c.bind_to_thread().expect("bind");
    let stream = ctx_c.default_stream();

    // ---- 数据(kNHD fp8 池 + f16 q) ----
    let pool_elems = nb * HKV * HD * PAGE;
    let k_fp8 = lcg_u8(pool_elems, 801);
    let v_fp8 = lcg_u8(pool_elems, 802);
    let q_f16 = lcg_f16(q_seq_len * H * HD, 803);

    // 行 i 的可见长:decode(q=1)= ctx;spec 形状(q=8)= ctx−q+i+1(因果块,含自身)
    let lens: Vec<usize> = (0..q_seq_len).map(|i| ctx - q_seq_len + i + 1).collect();
    let seq_len_total = ctx as u32; // XQA:整请求总 KV 长(batch 维)

    let dev_u8 = |v: &[u8]| {
        let mut d = unsafe { stream.alloc::<u8>(v.len()) }.unwrap();
        stream.memcpy_htod(v, &mut d).unwrap();
        d
    };
    let dev_u16 = |v: &[u16]| {
        let mut d = unsafe { stream.alloc::<u16>(v.len()) }.unwrap();
        stream.memcpy_htod(v, &mut d).unwrap();
        d
    };
    let d_k = dev_u8(&k_fp8);
    let d_v = dev_u8(&v_fp8);
    let d_q = dev_u16(&q_f16);
    let d_out_xqa = dev_u16(&vec![0u16; q_seq_len * H * HD]);
    let d_out_v2 = dev_u16(&vec![0u16; q_seq_len * H * HD]);

    let bt_i32: Vec<i32> = (0..nb as i32).collect();
    let d_bt_i32 = {
        let mut d = unsafe { stream.alloc::<i32>(bt_i32.len()) }.unwrap();
        stream.memcpy_htod(&bt_i32, &mut d).unwrap();
        d
    };
    let d_seq_u32 = {
        let mut d = unsafe { stream.alloc::<u32>(1) }.unwrap();
        stream.memcpy_htod(&[seq_len_total], &mut d).unwrap();
        d
    };
    // 掩码:行 i 的 bit j = 可看块内第 j 个 key;q=1 只需 bit0 = 1(允许自身)
    let mask: Vec<u32> = (0..q_seq_len).map(|i| (1u32 << (i + 1)) - 1).collect();
    let d_mask = {
        let mut d = unsafe { stream.alloc::<u32>(mask.len()) }.unwrap();
        stream.memcpy_htod(&mask, &mut d).unwrap();
        d
    };
    let sem_len = HKV.div_ceil(2) * 2 + 2 + HKV + 2;
    let d_sem = {
        let mut d = unsafe { stream.alloc::<u32>(sem_len) }.unwrap();
        stream.memcpy_htod(&vec![0u32; sem_len], &mut d).unwrap();
        d
    };
    let d_scratch = unsafe { stream.alloc::<u8>(256 << 20) }.unwrap();

    // v2 基线参数(f32 页表/可见长,与生产同式)
    let bt_f32: Vec<f32> = (0..q_seq_len).flat_map(|_| (0..nb).map(|b| b as f32)).collect();
    let d_bt_f32 = {
        let mut d = unsafe { stream.alloc::<f32>(bt_f32.len()) }.unwrap();
        stream.memcpy_htod(&bt_f32, &mut d).unwrap();
        d
    };
    let d_lens_f32 = {
        let mut d = unsafe { stream.alloc::<f32>(lens.len()) }.unwrap();
        stream.memcpy_htod(&lens.iter().map(|&l| l as f32).collect::<Vec<_>>(), &mut d).unwrap();
        d
    };
    let d_es = {
        let mut d = unsafe { stream.alloc::<f32>(q_seq_len * H * nparts) }.unwrap();
        stream.memcpy_htod(&vec![0f32; q_seq_len * H * nparts], &mut d).unwrap();
        d
    };
    let d_ml = {
        let mut d = unsafe { stream.alloc::<f32>(q_seq_len * H * nparts) }.unwrap();
        stream.memcpy_htod(&vec![0f32; q_seq_len * H * nparts], &mut d).unwrap();
        d
    };
    let d_partial = dev_u16(&vec![0u16; q_seq_len * H * nparts * HD]);

    // ---- XQA 单发 ----
    let (q_p, k_p, v_p) = (
        d_q.device_ptr(&stream).0,
        d_k.device_ptr(&stream).0,
        d_v.device_ptr(&stream).0,
    );
    let (o_p, bt_p, sq_p, mk_p, sem_p, sc_p) = (
        d_out_xqa.device_ptr(&stream).0,
        d_bt_i32.device_ptr(&stream).0,
        d_seq_u32.device_ptr(&stream).0,
        d_mask.device_ptr(&stream).0,
        d_sem.device_ptr(&stream).0,
        d_scratch.device_ptr(&stream).0,
    );
    let stream_raw = stream.cu_stream() as *mut core::ffi::c_void;
    // XQA_SM:覆盖 launcher 的 sm_count(只是用来算 nbSubSeqPerSeq 的 split 数——
    // 放大它 = 每 SM 多放 CTA = 提占用率;用来验证"是否为延迟墙")
    let sm_count: u32 = owl_shared::env_reader::str("XQA_SM")
        .and_then(|s| s.parse().ok())
        .unwrap_or(SM86_SM_COUNT);
    let xqa_call = |out: u64| unsafe {
        if decode_inst {
            assert_eq!(q_seq_len, 1, "decode 实例化只支持 q=1");
            owl_xqa_launch_decode(
                sm_count,
                HKV as u32,
                0,
                1.0,
                out as *mut core::ffi::c_void,
                q_p as *const core::ffi::c_void,
                k_p as *mut core::ffi::c_void,
                v_p as *mut core::ffi::c_void,
                bt_p as *const i32,
                (nb * PAGE) as u32,
                sq_p as *const u32,
                1,
                1.0,
                q_seq_len as u32,
                mk_p as *const u32,
                sem_p as *mut u32,
                sc_p as *mut core::ffi::c_void,
                (PAGE * HKV * HD) as u64,
                (HKV * HD) as u64,
                HD as u64,
                stream_raw,
            )
        } else {
            owl_xqa_launch(
                sm_count,
                HKV as u32,
                0,
                1.0, // q_scale = attn_scale(1/sqrt(256)) * sqrt(256) = 1.0
                out as *mut core::ffi::c_void,
                q_p as *const core::ffi::c_void,
                k_p as *mut core::ffi::c_void,
                v_p as *mut core::ffi::c_void,
                bt_p as *const i32,
                (nb * PAGE) as u32,
                sq_p as *const u32,
                1,
                1.0,
                q_seq_len as u32,
                mk_p as *const u32,
                sem_p as *mut u32,
                sc_p as *mut core::ffi::c_void,
                (PAGE * HKV * HD) as u64,
                (HKV * HD) as u64,
                HD as u64,
                stream_raw,
            )
        }
    };
    let rc = xqa_call(o_p);
    assert_eq!(rc, 0, "owl_xqa_launch 返回 {rc}");
    stream.synchronize().expect("xqa sync");

    // ---- v2 基线(主核 + reduce;与生产同两发) ----
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(
        owl_kernels::sources::attention::PAGED_ATTENTION_F16,
        cudarc::nvrtc::CompileOptions {
            arch: Some("compute_86"),
            include_paths: vec!["/usr/local/cuda/include".into()],
            ..Default::default()
        },
    )
    .expect("nvrtc compile paged v2");
    let m = ctx_c.load_module(ptx).expect("module");
    let f_main = m
        .load_function("vllm_paged_attention_v2_fp8_knhd_hd256bs32")
        .expect("load v2 fp8 knhd hd256bs32");
    let f_red = m
        .load_function("vllm_paged_attention_v2_reduce_f16_hd256")
        .expect("load v2 reduce hd256");
    let (es_p, ml_p, pa_p, ln_p) = (
        d_es.device_ptr(&stream).0,
        d_ml.device_ptr(&stream).0,
        d_partial.device_ptr(&stream).0,
        d_lens_f32.device_ptr(&stream).0,
    );
    let o2_p = d_out_v2.device_ptr(&stream).0;
    let scale = 1f32 / (HD as f32).sqrt();
    let bt_v2 = d_bt_f32.device_ptr(&stream).0;
    let hkv_i = HKV as i32;
    let nb_i = nb as i32;
    let qstride_i = (H * HD) as i32;
    let kvbs_i = (HKV * HD * PAGE) as i32;
    let kvhs_i = HD as i32;
    let one_f = 1f32;
    let neg1_i = -1i32;
    let zero_i = 0i32;
    let sq = q_seq_len as u32;
    let v2_call = || {
        let mut b = stream.launch_builder(&f_main);
        b.arg(&q_p)
            .arg(&k_p)
            .arg(&v_p)
            .arg(&bt_v2)
            .arg(&ln_p)
            .arg(&es_p)
            .arg(&es_p)
            .arg(&ml_p)
            .arg(&hkv_i)
            .arg(&scale)
            .arg(&nb_i)
            .arg(&qstride_i)
            .arg(&kvbs_i)
            .arg(&kvhs_i)
            .arg(&one_f)
            .arg(&neg1_i)
            .arg(&zero_i)
            .arg(&pa_p);
        let cfg = LaunchConfig {
            grid_dim: (H as u32, sq, nparts as u32),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 2048,
        };
        unsafe { b.launch(cfg) }.expect("v2 main");
        let mut b2 = stream.launch_builder(&f_red);
        let np_i = nparts as i32;
        b2.arg(&es_p).arg(&ml_p).arg(&pa_p).arg(&ln_p).arg(&np_i).arg(&o2_p);
        let cfg2 = LaunchConfig {
            grid_dim: (H as u32, sq, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 2048,
        };
        unsafe { b2.launch(cfg2) }.expect("v2 reduce");
    };
    v2_call();
    stream.synchronize().expect("v2 sync");

    // ---- 计时 ----
    let bench = |n: usize, mut f: Box<dyn FnMut()>| -> f64 {
        for _ in 0..3 {
            f();
        }
        stream.synchronize().unwrap();
        let t = Instant::now();
        for _ in 0..n {
            f();
        }
        stream.synchronize().unwrap();
        t.elapsed().as_secs_f64() * 1e3 / n as f64
    };
    let xqa_ms = bench(20, Box::new(|| { xqa_call(o_p); }));
    let v2_ms = bench(20, Box::new(|| { v2_call(); }));

    // ---- 数值 ----
    let n = q_seq_len * H * HD;
    let mut o_xqa = vec![0u16; n];
    stream.memcpy_dtoh(&d_out_xqa, &mut o_xqa).unwrap();
    let mut o_v2 = vec![0u16; n];
    stream.memcpy_dtoh(&d_out_v2, &mut o_v2).unwrap();
    let fx: Vec<f32> = o_xqa.iter().map(|&h| f16_to_f32(h)).collect();
    let fv: Vec<f32> = o_v2.iter().map(|&h| f16_to_f32(h)).collect();
    let mut d_xv = 0f32;
    let mut n_zero = 0usize;
    for i in 0..n {
        d_xv = d_xv.max((fx[i] - fv[i]).abs());
        if fx[i] == 0.0 {
            n_zero += 1;
        }
    }
    let d_xh = if check {
        // decode:host 只算 1 行(q=1);spec 形状逐行算
        let host = host_ref_knhd(&q_f16, &k_fp8, &v_fp8, lens[q_seq_len - 1]);
        Some(fx.iter().zip(host.iter()).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max))
    } else {
        None
    };
    let d_vh = if check {
        let host = host_ref_knhd(&q_f16, &k_fp8, &v_fp8, lens[q_seq_len - 1]);
        Some(fv.iter().zip(host.iter()).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max))
    } else {
        None
    };

    println!(
        "=== ctx={ctx} (pages={nb}, nparts={nparts}, q_seq_len={q_seq_len}, inst={}) ===\n  \
         XQA 单发      : {xqa_ms:7.3} ms/call\n  \
         v2 主核+reduce: {v2_ms:7.3} ms/call\n  \
         加速比        : {:5.2}×\n  \
         数值 max|XQA−v2| = {d_xv:.6}   (XQA 零值 {n_zero}/{n})\n  \
         数值 max|XQA−host| = {:?}   max|v2−host| = {:?}",
        if decode_inst { "decode/M_TILESIZE=16" } else { "verify/M_TILESIZE=32" },
        v2_ms / xqa_ms,
        d_xh,
        d_vh
    );
    if let Some(dh) = d_xh {
        assert!(dh < 1e-2, "XQA vs host 超门:{dh:.6}");
    }
}

#[test]
fn xqa_decode_speed_and_parity() {
    if !owl_shared::env_reader::flag("OWL_TEST_DEVICE") {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    }
    // D1 备选实测:q=1 走现有 SPEC_DEC=1 实例化(M_TILESIZE=32,32 行里只有 6 行有效)
    // XQA_DECODE_ONLY=1 → 只跑 M_TILESIZE=16 实例化;XQA_SM=N → 覆盖 sm_count(split 数)
    let decode_only = owl_shared::env_reader::flag("XQA_DECODE_ONLY");
    let sm_sweep = owl_shared::env_reader::flag("XQA_SM_SWEEP");
    if sm_sweep {
        for sm in [84u32, 168, 252, 336] {
            // SAFETY: 测试域单线程串行;env 读取在 run_case 内
            unsafe { std::env::set_var("XQA_SM", sm.to_string()) };
            println!("--- sm_count={sm} ---");
            run_case(8192, 1, true, false);
        }
        return;
    }
    if !decode_only {
        run_case(2048, 1, false, true);
    }
    // 第二份实例化(SPEC_DEC=0,原生 decode 形状:rows=6 → warpTile.y=16)
    run_case(2048, 1, true, true);
    if decode_only {
        return;
    }
    // 短/长 ctx 速度对照(decode 真实跨度;两份实例化并排)
    for ctx in [512usize, 8192] {
        run_case(ctx, 1, false, false);
        run_case(ctx, 1, true, false);
    }
    // 交叉:spec 形状(q=8)在第二个 ctx 上复算一次,确认实例化未被 q=1 路径污染
    run_case(2048, 8, false, false);
}

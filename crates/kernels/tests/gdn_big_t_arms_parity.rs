//! 大 T prefill 双臂对拍(scalar vs varlen;2026-10-09 语义崩案定位)。
//!
//! 生产症状:tokens >= 64 分臂(scalar/chunked)prefill 输出与终态崩,
//! varlen 恒健康;金标 c1-c5 全绿但 g ∈ [-0.51, -0.01] **浅域零深谷覆盖**。
//! 本对拍:同输入双臂(逐位同源 f16 值),g 域扫 shallow/deep/mixed,
//! o 与终态 max|Δ| 报点。分歧现形 = 核/ABI 层;全域吻合 = 层侧装配。
//!
//! 无 GPU 跳过(OWL_TEST_DEVICE)。

use cudarc::driver::{CudaContext, DevicePtr, LaunchConfig, PushKernelArg};
use half::f16;

const NK: usize = 16;
const NV: usize = 48;
const KD: usize = 128;
const VD: usize = 128;
const SCALE: f32 = 0.08838834764831845; // 1/sqrt(128)

/// 确定性伪随机 [0,1)
fn lcg01(seq: u32, i: usize) -> f32 {
    let x = seq
        .wrapping_mul(0x9E37_79B9)
        .wrapping_add((i as u32).wrapping_mul(0x85EB_CA6B));
    ((x >> 8) as f32) / 16777216.0
}

struct Inputs {
    q: Vec<f16>,
    k: Vec<f16>,
    v: Vec<f16>,
    beta: Vec<f16>,
    g: Vec<f16>,
}

/// 生产量级合成:q/k/v = 投影输出小量级;beta ∈ (0,1);
/// g = nats 负域,g_depth = 深谷上界(金标浅域 = 0.5)。
fn gen_inputs(t: usize, seed: u32, g_depth: f32) -> Inputs {
    let n_qk = t * NK * KD;
    let n_v = t * NV * VD;
    let n_h = t * NV;
    let mut q = Vec::with_capacity(n_qk);
    let mut k = Vec::with_capacity(n_qk);
    let mut v = Vec::with_capacity(n_v);
    let mut beta = Vec::with_capacity(n_h);
    let mut g = Vec::with_capacity(n_h);
    for i in 0..n_qk {
        q.push(f16::from_f32((lcg01(seed, i) - 0.5) * 0.2));
        k.push(f16::from_f32((lcg01(seed + 1, i) - 0.5) * 0.2));
    }
    for i in 0..n_v {
        v.push(f16::from_f32((lcg01(seed + 2, i) - 0.5) * 0.2));
    }
    for i in 0..n_h {
        beta.push(f16::from_f32(0.1 + 0.8 * lcg01(seed + 3, i)));
        g.push(f16::from_f32(-(0.01 + g_depth * lcg01(seed + 4, i))));
    }
    Inputs { q, k, v, beta, g }
}

fn max_dev(a: &[f32], b: &[f32]) -> (usize, f32) {
    let mut worst = 0f32;
    let mut idx = 0;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let d = (x - y).abs();
        if d > worst {
            worst = d;
            idx = i;
        }
    }
    (idx, worst)
}

#[test]
fn scalar_matches_varlen_big_t() {
    if !owl_shared::env_reader::flag("OWL_TEST_DEVICE") {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    }
    let ctx = CudaContext::new(1).expect("ctx");
    ctx.bind_to_thread().expect("bind");
    let stream = ctx.default_stream();

    // scalar 臂 cubin
    let cubin = owl_kernels::family::gdn_scalar::cubin::CHUNK_SCALAR_F32;
    let m_scalar = ctx
        .load_module(cudarc::nvrtc::Ptx::from_binary(cubin.to_vec()))
        .expect("load scalar cubin");
    let f_scalar = m_scalar
        .load_function(owl_kernels::family::gdn_scalar::cubin::KERNEL_F32)
        .expect("fn scalar");
    // varlen 源(nvrtc)
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(
        owl_kernels::sources::text::GDN_F32,
        cudarc::nvrtc::CompileOptions {
            arch: Some("compute_86"),
            include_paths: vec!["/usr/local/cuda/include".into()],
            ..Default::default()
        },
    )
    .expect("nvrtc gdn.cu");
    let m_var = ctx.load_module(ptx).expect("module");
    let f_var = m_var
        .load_function("owl_gdn_recurrence_varlen_gqa_f16")
        .expect("fn varlen");

    let cases: &[(&str, usize, f32)] = &[
        ("shallow_t64", 64, 0.5),
        ("deep_t64", 64, 25.0),
        ("mixed_t64", 64, 3.0),
        ("deep_t65", 65, 25.0),
    ];

    for &(tag, t, g_depth) in cases {
        let inp = gen_inputs(t, 11, g_depth);

        // ── 共享设备输入 ──
        let q16: Vec<u16> = inp.q.iter().map(|x| x.to_bits()).collect();
        let k16: Vec<u16> = inp.k.iter().map(|x| x.to_bits()).collect();
        let v16: Vec<u16> = inp.v.iter().map(|x| x.to_bits()).collect();
        let b16: Vec<u16> = inp.beta.iter().map(|x| x.to_bits()).collect();
        let g16: Vec<u16> = inp.g.iter().map(|x| x.to_bits()).collect();
        let mut d_q = unsafe { stream.alloc::<u16>(q16.len()) }.unwrap();
        stream.memcpy_htod(&q16, &mut d_q).unwrap();
        let mut d_k = unsafe { stream.alloc::<u16>(k16.len()) }.unwrap();
        stream.memcpy_htod(&k16, &mut d_k).unwrap();
        let mut d_v = unsafe { stream.alloc::<u16>(v16.len()) }.unwrap();
        stream.memcpy_htod(&v16, &mut d_v).unwrap();
        let mut d_b = unsafe { stream.alloc::<u16>(b16.len()) }.unwrap();
        stream.memcpy_htod(&b16, &mut d_b).unwrap();
        let mut d_g = unsafe { stream.alloc::<u16>(g16.len()) }.unwrap();
        stream.memcpy_htod(&g16, &mut d_g).unwrap();
        let q_p = d_q.device_ptr(&stream).0;
        let k_p = d_k.device_ptr(&stream).0;
        let v_p = d_v.device_ptr(&stream).0;
        let b_p = d_b.device_ptr(&stream).0;
        let g_p = d_g.device_ptr(&stream).0;

        // f32 镜像(scalar 臂直入 = f16 值的精确 f32)
        let to_f32 = |x: &[f16]| x.iter().map(|h| h.to_f32()).collect::<Vec<f32>>();

        // ── 臂 1:varlen(f16 in;state [1,NV,KD,VD] 零初态)──
        let mut v_state = stream.alloc_zeros::<f32>(NV * KD * VD).unwrap();
        let mut v_slots = stream.alloc_zeros::<f32>(1).unwrap(); // slot 0
        let cu_host = vec![0.0f32, t as f32];
        let mut v_cu = stream.alloc_zeros::<f32>(2).unwrap();
        stream.memcpy_htod(&cu_host, &mut v_cu).unwrap();
        let mut v_out = unsafe { stream.alloc::<u16>(t * NV * VD) }.unwrap();
        {
            let sl_p = v_slots.device_ptr(&stream).0;
            let cu_p = v_cu.device_ptr(&stream).0;
            let st_p = v_state.device_ptr(&stream).0;
            let o_p = v_out.device_ptr(&stream).0;
            let (batch_u, nv_u, nk_u, kd_u, vd_u) =
                (1u64, NV as u64, NK as u64, KD as u64, VD as u64);
            let qs = SCALE; // q 缩放(varlen 乘在 q 上)
            let grid = (((VD + 7) / 8) as u32, NV as u32, 1);
            let block = (32, 8, 1);
            let smem = ((4 * KD + 4) * 4) as u32;
            let mut b = stream.launch_builder(&f_var);
            b.arg(&q_p)
                .arg(&k_p)
                .arg(&v_p)
                .arg(&g_p)
                .arg(&b_p)
                .arg(&st_p)
                .arg(&sl_p)
                .arg(&cu_p)
                .arg(&batch_u)
                .arg(&nv_u)
                .arg(&nk_u)
                .arg(&kd_u)
                .arg(&vd_u)
                .arg(&qs)
                .arg(&o_p);
            let cfg = LaunchConfig { grid_dim: grid, block_dim: block, shared_mem_bytes: smem };
            unsafe { b.launch(cfg) }.unwrap();
        }
        let mut var_o16 = vec![0u16; t * NV * VD];
        stream.memcpy_dtoh(&v_out, &mut var_o16).unwrap();
        let var_o: Vec<f32> = var_o16.iter().map(|&b| f16::from_bits(b).to_f32()).collect();
        let mut var_st = vec![0f32; NV * KD * VD];
        stream.memcpy_dtoh(&v_state, &mut var_st).unwrap();

        // ── 臂 2:scalar(f32 in = f16 值精确镜像)──
        let s_q = to_f32(&inp.q);
        let s_k = to_f32(&inp.k);
        let s_v = to_f32(&inp.v);
        let s_b = to_f32(&inp.beta);
        let s_g = to_f32(&inp.g);
        let mut d_sq = stream.alloc_zeros::<f32>(s_q.len()).unwrap();
        stream.memcpy_htod(&s_q, &mut d_sq).unwrap();
        let mut d_sk = stream.alloc_zeros::<f32>(s_k.len()).unwrap();
        stream.memcpy_htod(&s_k, &mut d_sk).unwrap();
        let mut d_sv = stream.alloc_zeros::<f32>(s_v.len()).unwrap();
        stream.memcpy_htod(&s_v, &mut d_sv).unwrap();
        let mut d_sb = stream.alloc_zeros::<f32>(s_b.len()).unwrap();
        stream.memcpy_htod(&s_b, &mut d_sb).unwrap();
        let mut d_sg = stream.alloc_zeros::<f32>(s_g.len()).unwrap();
        stream.memcpy_htod(&s_g, &mut d_sg).unwrap();
        let mut s_state = stream.alloc_zeros::<f32>(NV * KD * VD).unwrap();
        let seq_off = vec![0i32, t as i32];
        let mut d_soff = stream.alloc_zeros::<i32>(2).unwrap();
        stream.memcpy_htod(&seq_off, &mut d_soff).unwrap();
        let mut s_out = stream.alloc_zeros::<f32>(t * NV * VD).unwrap();
        {
            let sq_p = d_sq.device_ptr(&stream).0;
            let sk_p = d_sk.device_ptr(&stream).0;
            let sv_p = d_sv.device_ptr(&stream).0;
            let sb_p = d_sb.device_ptr(&stream).0;
            let sg_p = d_sg.device_ptr(&stream).0;
            let st_p = s_state.device_ptr(&stream).0;
            let so_p = d_soff.device_ptr(&stream).0;
            let o_p = s_out.device_ptr(&stream).0;
            let (nk_i, hv_i, kd_i) = (NK as i32, NV as i32, KD as i32);
            let mut b = stream.launch_builder(&f_scalar);
            b.arg(&o_p)
                .arg(&sq_p)
                .arg(&sk_p)
                .arg(&sv_p)
                .arg(&sb_p)
                .arg(&sg_p)
                .arg(&st_p)
                .arg(&so_p)
                .arg(&nk_i)
                .arg(&hv_i)
                .arg(&kd_i)
                .arg(&SCALE);
            let cfg = LaunchConfig {
                grid_dim: (1, NV as u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 25472,
            };
            unsafe { b.launch(cfg) }.unwrap();
        }
        let mut s_o = vec![0f32; t * NV * VD];
        stream.memcpy_dtoh(&s_out, &mut s_o).unwrap();
        let mut s_st = vec![0f32; NV * KD * VD];
        stream.memcpy_dtoh(&s_state, &mut s_st).unwrap();

        // ── 对拍 ──
        let (oi, od) = max_dev(&s_o, &var_o);
        let (si, sd) = max_dev(&s_st, &var_st);
        let o_rel = if var_o.iter().any(|x| x.abs() > 1e-6) {
            od / var_o.iter().map(|x| x.abs()).fold(0.0f32, f32::max).max(1e-9)
        } else {
            f32::INFINITY
        };
        println!(
            "[{tag}] g_depth={g_depth} o:max|Δ|={od:.3e}(rel {o_rel:.3e})@{oi} state:max|Δ|={sd:.3e}@{si}"
        );
        // 容差:双臂同 f16 输入 + f32 累加,理论应 ~1e-5 级;放 1e-2 硬门
        assert!(od < 1e-2, "{tag}: o 分歧 {od:.3e} @ {oi}(scalar vs varlen)");
        assert!(sd < 1e-2, "{tag}: 终态分歧 {sd:.3e} @ {si}(scalar vs varlen)");
    }
}

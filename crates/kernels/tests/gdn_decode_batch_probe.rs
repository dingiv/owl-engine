//! B5.0 证可性探针(2026-10-10):GDN decode 核批语义对拍。
//!
//! 判据(立项文档 §五 B5.0):**双格批 decode(B=2 单发)≡ 两次单格
//! 串行(逐位)**;附负 padding 掩码语义(slots[i]<0 = 跳过,状态不动)。
//!
//! 为什么是 GDN:v2 paged attention 核为单序列 launch(grid (hq,1,nparts),
//! 无 seq 维)—— 批化有核改造量;而 GDN decode_step 核签名原生带
//! `slots: [B]`(负 = padding)+ per-slot 状态寻址([max_slots, ...]),
//! 是批 decode 的关键路径(48 层 × 递推)。本探针 = 核级证可的最后一环
//! (marlin 行批已由 verify T=8 证明;attention 走 per-seq launch,
//! slice_view 行切片机制已在)。
//!
//! 无 GPU 跳过(OWL_TEST_DEVICE)。

use cudarc::driver::{CudaContext, DevicePtr, LaunchConfig, PushKernelArg};

const NK: usize = 16;
const NV: usize = 48;
const KD: usize = 128;
const VD: usize = 128;
const K: usize = NK * KD; // 2048
const V: usize = NV * VD; // 6144
const HV: usize = NV;
const CONV_DIM: usize = 2 * K + V; // 10240
const MAX_SLOTS: usize = 8;

/// 确定性伪随机 f16 位型(LCG;等价性测试只需同输入,不需分布)
fn lcg_f16_bits(seq: u32, i: usize) -> u16 {
    let x = seq
        .wrapping_mul(0x9E37_79B9)
        .wrapping_add((i as u32).wrapping_mul(0x85EB_CA6B));
    // f16 正数 [1.0, 2.0) 区间位型:指数 15,尾数取低 10 位(免 denormal/inf)
    0x3C00u16 | ((x >> 6) & 0x03FF) as u16
}

struct Bundle {
    q_c: Vec<u16>,
    k_c: Vec<u16>,
    v_raw: Vec<u16>,
    z: Vec<u16>,
    b_gate: Vec<u16>,
    a_gate: Vec<u16>,
    w: Vec<u16>,
    a_log: Vec<u16>,
    dt_bias: Vec<u16>,
    norm_w: Vec<u16>,
}

fn bundle(b: usize, seed: u32) -> Bundle {
    Bundle {
        q_c: (0..b * K).map(|i| lcg_f16_bits(seed, i)).collect(),
        k_c: (0..b * K).map(|i| lcg_f16_bits(seed + 1, i)).collect(),
        v_raw: (0..b * V).map(|i| lcg_f16_bits(seed + 2, i)).collect(),
        z: (0..b * V).map(|i| lcg_f16_bits(seed + 3, i)).collect(),
        b_gate: (0..b * HV).map(|i| lcg_f16_bits(seed + 4, i)).collect(),
        a_gate: (0..b * HV).map(|i| lcg_f16_bits(seed + 5, i)).collect(),
        w: (0..CONV_DIM * 4).map(|i| lcg_f16_bits(seed + 6, i)).collect(),
        a_log: (0..HV).map(|i| lcg_f16_bits(seed + 7, i)).collect(),
        dt_bias: (0..HV).map(|i| lcg_f16_bits(seed + 8, i)).collect(),
        norm_w: (0..VD).map(|i| lcg_f16_bits(seed + 9, i)).collect(),
    }
}

fn dev_u16(s: &std::sync::Arc<cudarc::driver::CudaStream>, n: usize) -> cudarc::driver::CudaSlice<u16> {
    unsafe { s.alloc::<u16>(n) }.unwrap()
}

#[test]
fn gdn_decode_step_batch2_equals_serial() {
    if std::env::var_os("OWL_TEST_DEVICE").is_none() {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    }
    // nvrtc 懒编译(源 = sources.rs GDN_F32 同文件;自足,仅 <cuda_fp16.h>)
    let ctx = CudaContext::new(1).expect("ctx");
    ctx.bind_to_thread().expect("bind");
    let stream = ctx.default_stream();
    // nvrtc 懒编译(源 = text/gdn.cu;自足,仅 <cuda_fp16.h> 系统头)
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(
        owl_kernels::sources::text::GDN_F32,
        cudarc::nvrtc::CompileOptions {
            arch: Some("compute_86"),
            include_paths: vec!["/usr/local/cuda/include".into()],
            ..Default::default()
        },
    )
    .expect("nvrtc compile gdn.cu");
    let modu = ctx.load_module(ptx).expect("module");
    let f = modu.load_function("owl_gdn_decode_step_f16").expect("fn");

    let bn = bundle(2, 7);
    // 设备面上各输入(批态)
    let mut d_q = dev_u16(&stream, bn.q_c.len());
    stream.memcpy_htod(&bn.q_c, &mut d_q).unwrap();
    let mut d_k = dev_u16(&stream, bn.k_c.len());
    stream.memcpy_htod(&bn.k_c, &mut d_k).unwrap();
    let mut d_v = dev_u16(&stream, bn.v_raw.len());
    stream.memcpy_htod(&bn.v_raw, &mut d_v).unwrap();
    let mut d_z = dev_u16(&stream, bn.z.len());
    stream.memcpy_htod(&bn.z, &mut d_z).unwrap();
    let mut d_bg = dev_u16(&stream, bn.b_gate.len());
    stream.memcpy_htod(&bn.b_gate, &mut d_bg).unwrap();
    let mut d_ag = dev_u16(&stream, bn.a_gate.len());
    stream.memcpy_htod(&bn.a_gate, &mut d_ag).unwrap();
    let mut d_w = dev_u16(&stream, bn.w.len());
    stream.memcpy_htod(&bn.w, &mut d_w).unwrap();
    let mut d_al = dev_u16(&stream, bn.a_log.len());
    stream.memcpy_htod(&bn.a_log, &mut d_al).unwrap();
    let mut d_db = dev_u16(&stream, bn.dt_bias.len());
    stream.memcpy_htod(&bn.dt_bias, &mut d_db).unwrap();
    let mut d_nw = dev_u16(&stream, bn.norm_w.len());
    stream.memcpy_htod(&bn.norm_w, &mut d_nw).unwrap();

    let conv_state_len = MAX_SLOTS * V * 3;
    let rec_state_len = MAX_SLOTS * HV * KD * VD;

    let launch = |slots_host: &[f32],
                  conv: &cudarc::driver::CudaSlice<f32>,
                  rec: &cudarc::driver::CudaSlice<f32>,
                  out: &cudarc::driver::CudaSlice<u16>,
                  rows: usize,
                  input_row_base: usize,
                  out_row: usize| {
        // input_row_base:串行臂复用批态输入缓冲的行偏移(指针推进,字节)
        let mut slots_d = unsafe { stream.alloc::<f32>(slots_host.len()) }.unwrap();
        stream.memcpy_htod(slots_host, &mut slots_d).unwrap();
        let byte_off = |n: usize, r: usize| (r * n * 2) as u64;
        let q_p = d_q.device_ptr(&stream).0 + byte_off(K, input_row_base);
        let k_p = d_k.device_ptr(&stream).0 + byte_off(K, input_row_base);
        let v_p = d_v.device_ptr(&stream).0 + byte_off(V, input_row_base);
        let z_p = d_z.device_ptr(&stream).0 + byte_off(V, input_row_base);
        let bg_p = d_bg.device_ptr(&stream).0 + byte_off(HV, input_row_base);
        let ag_p = d_ag.device_ptr(&stream).0 + byte_off(HV, input_row_base);
        let w_p = d_w.device_ptr(&stream).0;
        let cs_p = conv.device_ptr(&stream).0;
        let al_p = d_al.device_ptr(&stream).0;
        let db_p = d_db.device_ptr(&stream).0;
        let rs_p = rec.device_ptr(&stream).0;
        let sl_p = slots_d.device_ptr(&stream).0;
        let nw_p = d_nw.device_ptr(&stream).0;
        let o_p = out.device_ptr(&stream).0 + byte_off(HV * VD, out_row);
        let (batch_u, nv_u, nk_u, kd_u, vd_u) =
            (rows as u64, NV as u64, NK as u64, KD as u64, VD as u64);
        let (e1, e2, qs) = (1e-6f32, 1e-6f32, 1.0f32);
        let mut b = stream.launch_builder(&f);
        b.arg(&q_p)
            .arg(&k_p)
            .arg(&v_p)
            .arg(&z_p)
            .arg(&bg_p)
            .arg(&ag_p)
            .arg(&w_p)
            .arg(&cs_p)
            .arg(&al_p)
            .arg(&db_p)
            .arg(&rs_p)
            .arg(&sl_p)
            .arg(&nw_p)
            .arg(&batch_u)
            .arg(&nv_u)
            .arg(&nk_u)
            .arg(&kd_u)
            .arg(&vd_u)
            .arg(&e1)
            .arg(&e2)
            .arg(&qs)
            .arg(&o_p);
        let cfg = LaunchConfig {
            grid_dim: ((rows * NV) as u32, 1, 1),
            block_dim: (KD as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(cfg) }.expect("owl_gdn_decode_step_f16 launch");
    };

    let fresh_states = || {
        (
            unsafe { stream.alloc_zeros::<f32>(conv_state_len).unwrap() },
            unsafe { stream.alloc_zeros::<f32>(rec_state_len).unwrap() },
        )
    };

    // ── 批态:B=2,slots [0,1],单发 ──
    let (conv_b, rec_b) = fresh_states();
    let out_b = dev_u16(&stream, 2 * HV * VD);
    launch(&[0.0, 1.0], &conv_b, &rec_b, &out_b, 2, 0, 0);
    stream.synchronize().unwrap();

    // ── 串行臂:两次 B=1(行 0 → 槽 0;行 1 → 槽 1)──
    let (conv_s, rec_s) = fresh_states();
    let out_s = dev_u16(&stream, 2 * HV * VD);
    launch(&[0.0], &conv_s, &rec_s, &out_s, 1, 0, 0);
    launch(&[1.0], &conv_s, &rec_s, &out_s, 1, 1, 1);
    stream.synchronize().unwrap();

    // ── 逐位对拍:out + conv/rec 状态 ──
    let mut ob = vec![0u16; 2 * HV * VD];
    stream.memcpy_dtoh(&out_b, &mut ob).unwrap();
    let mut os = vec![0u16; 2 * HV * VD];
    stream.memcpy_dtoh(&out_s, &mut os).unwrap();
    assert_eq!(ob, os, "批 B=2 单发 ≠ 两次 B=1(out 逐位)");

    let mut cb = vec![0f32; conv_state_len];
    stream.memcpy_dtoh(&conv_b, &mut cb).unwrap();
    let mut cs = vec![0f32; conv_state_len];
    stream.memcpy_dtoh(&conv_s, &mut cs).unwrap();
    assert_eq!(cb, cs, "批 ≠ 串行(conv_v_state 逐位)");

    let mut rb = vec![0f32; rec_state_len];
    stream.memcpy_dtoh(&rec_b, &mut rb).unwrap();
    let mut rs = vec![0f32; rec_state_len];
    stream.memcpy_dtoh(&rec_s, &mut rs).unwrap();
    assert_eq!(rb, rs, "批 ≠ 串行(rec_state 逐位)");

    // ── 负 padding 掩码:B=3 slots [0,1,-5] → 行 2 状态不动 ──
    let (conv_p, rec_p) = fresh_states();
    let out_p = dev_u16(&stream, 3 * HV * VD);
    launch(&[0.0, 1.0, -5.0], &conv_p, &rec_p, &out_p, 3, 0, 0);
    stream.synchronize().unwrap();
    let mut cp = vec![0f32; conv_state_len];
    stream.memcpy_dtoh(&conv_p, &mut cp).unwrap();
    let mut rp = vec![0f32; rec_state_len];
    stream.memcpy_dtoh(&rec_p, &mut rp).unwrap();
    // 行 2 无槽:槽 2..8 区域必须仍为全零(未被 padding 行写穿)
    assert!(
        cp[2 * V * 3..].iter().all(|&x| x == 0.0),
        "padding 行写穿 conv 状态"
    );
    assert!(
        rp[2 * HV * KD * VD..].iter().all(|&x| x == 0.0),
        "padding 行写穿 rec 状态"
    );
    // 行 0/1 与串行一致(前两槽)
    assert_eq!(cp[..2 * V * 3], cs[..2 * V * 3], "padding 批行 0/1 ≠ 串行(conv)");
    assert_eq!(
        rp[..2 * HV * KD * VD],
        rs[..2 * HV * KD * VD],
        "padding 批行 0/1 ≠ 串行(rec)"
    );
    println!("B5.0 探针 PASS:GDN decode 批语义 = 逐位等价 + padding 掩码");
}

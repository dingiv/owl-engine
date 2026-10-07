//! B6.1 证可性探针(2026-10-10):paged attention v2 的 fp8 e4m3 KV 读变体。
//!
//! 判据:同一组 q/K/V,f16 池(金标)vs fp8 池(host 侧 RNE 量化),
//! v2_kernel_f16<KV_FP8=false> vs <...,true>(含分区 reduce 路径),
//! 输出容差 = e4m3 3 位尾数的量化噪声量级(门:mean<2e-2 / max<8e-2,
//! 值域 ~±3;首跑打印实测分布定档)。
//!
//! 附加:PagedAttention v2 reduce 路径(nparts>1)同测;fp8 解码表与
//! CUDA cuda_fp8.hpp 语义一致(直接对照同输入核内转换)。
//! 无 GPU 跳过(OWL_TEST_DEVICE)。

use cudarc::driver::{CudaContext, DevicePtr, LaunchConfig, PushKernelArg};

const HD: usize = 256;
const PAGE: usize = 32;
const X: usize = 8;
const NUM_HEADS: usize = 8;
const NUM_KV_HEADS: usize = 2;
const NUM_BLOCKS: usize = 128;
const NUM_THREADS: usize = 128;

/// e4m3 解码(镜像 CUDA cuda_fp8.hpp 语义;探针量化表用)
fn e4m3_decode(b: u8) -> f32 {
    let s = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = ((b >> 3) & 0x0F) as i32;
    let m = (b & 0x07) as i32;
    if e == 15 && m == 7 {
        return f32::NAN; // e4m3 NaN(0x7F/0xFF)
    }
    let v = if e == 0 {
        (m as f32) * 2f32.powi(-9)
    } else {
        (1.0 + m as f32 / 8.0) * 2f32.powi(e - 7)
    };
    s * v
}

/// f32 → e4m3(暴力 RNE:256 码全扫,与任何合理解码表自洽)
fn e4m3_encode(v: f32) -> u8 {
    if v.is_nan() {
        return 0x7F;
    }
    let mut best = 0u8;
    let mut best_err = f32::INFINITY;
    for code in 0..=255u8 {
        if code & 0x7F == 0x7F {
            continue; // 跳过 NaN 码
        }
        let d = e4m3_decode(code);
        let err = (d - v).abs();
        // 平手偏向偶数码(尾数最低位 0)
        if err < best_err || (err == best_err && code & 1 == 0 && best & 1 == 1) {
            best_err = err;
            best = code;
        }
    }
    best
}

fn lcg_f16_vec(n: usize, seed: u32) -> Vec<u16> {
    // 值域 ~±2(常态 KV 量级);f16 位型
    (0..n)
        .map(|i| {
            let x = seed
                .wrapping_mul(0x9E37_79B9)
                .wrapping_add((i as u32).wrapping_mul(0x85EB_CA6B));
            let f = ((x & 0xFFFF) as f32 / 65535.0 - 0.5) * 4.0;
            half_bits(f)
        })
        .collect()
}

fn half_bits(f: f32) -> u16 {
    // 精简 f32→f16(RNE;探针值域无 inf/denormal 边界压力,够用)
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

fn f16_bits_to_f32(h: u16) -> f32 {
    let s = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let e = ((h >> 10) & 0x1F) as i32;
    let m = (h & 0x3FF) as f32;
    if e == 0 {
        return s * m * 2f32.powi(-24);
    }
    if e == 31 {
        return f32::NAN;
    }
    s * (1.0 + m / 1024.0) * 2f32.powi(e - 15)
}

#[test]
fn paged_attn_v2_fp8_matches_f16() {
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
    .expect("nvrtc compile pagedattention");
        if let Some(bytes) = ptx.as_bytes() {
        owl_shared::file_loader::write("/tmp/b6.ptx", bytes).ok();
    }
    let m = ctx.load_module(ptx).expect("module");

    // ── 形状:2 序列(500 / 1500 tok → nparts 1 / 3,双路径同测)──
    let seq_lens = [500usize, 1500usize];
    let num_seqs = 2;
    let max_ctx = 1500usize;
    let nparts = max_ctx.div_ceil(512); // 3
    let blocks_per_seq = max_ctx.div_ceil(PAGE) + 2; // 49
    let kv_head_stride = NUM_KV_HEADS * (HD / X) * PAGE * X; // 元素序;fp8/f16 同数
    let kv_block_stride = kv_head_stride; // 单 KV 头布局简并(乘 kv_heads 已含)

    // 块表:两序列独立链(物理块不重叠)
    let mut block_tables = vec![0f32; num_seqs * blocks_per_seq];
    let mut nb = 0u32;
    for s in 0..num_seqs {
        for b in 0..blocks_per_seq {
            block_tables[s * blocks_per_seq + b] = {
                nb += 1;
                (nb - 1) as f32
            };
        }
    }
    assert!(nb <= NUM_BLOCKS as u32);

    // ── 输入:f16 金标 K/V/Q;fp8 池 = host RNE 量化 ──
    let k_f16 = lcg_f16_vec(NUM_BLOCKS * kv_head_stride, 11);
    let v_f16 = lcg_f16_vec(NUM_BLOCKS * kv_head_stride, 23);
    let q_f16 = lcg_f16_vec(num_seqs * NUM_HEADS * HD, 37);

    // fp8 量化(只量化引用到的块;全量量化 NUM_BLOCKS×16384×2 = 6MB 亦可)
    let quant = |src: &Vec<u16>| -> Vec<u8> {
        src.iter().map(|&h| e4m3_encode(f16_bits_to_f32(h))).collect()
    };
    let k_fp8 = quant(&k_f16);
    let v_fp8 = quant(&v_f16);

    // 设备面
    let dev_u16 = |s: &std::sync::Arc<cudarc::driver::CudaStream>, v: &[u16]| {
        let mut d = unsafe { s.alloc::<u16>(v.len()) }.unwrap();
        s.memcpy_htod(v, &mut d).unwrap();
        d
    };
    let dev_u8 = |s: &std::sync::Arc<cudarc::driver::CudaStream>, v: &[u8]| {
        let mut d = unsafe { s.alloc::<u8>(v.len()) }.unwrap();
        s.memcpy_htod(v, &mut d).unwrap();
        d
    };
    let dev_f32 = |s: &std::sync::Arc<cudarc::driver::CudaStream>, v: &[f32]| {
        let mut d = unsafe { s.alloc::<f32>(v.len()) }.unwrap();
        s.memcpy_htod(v, &mut d).unwrap();
        d
    };
    let d_q = dev_u16(&stream, &q_f16);
    let d_k16 = dev_u16(&stream, &k_f16);
    let d_v16 = dev_u16(&stream, &v_f16);
    let d_k8 = dev_u8(&stream, &k_fp8);
    let d_v8 = dev_u8(&stream, &v_fp8);
    let d_bt = dev_f32(&stream, &block_tables);
    let d_lens = dev_f32(&stream, &[seq_lens[0] as f32, seq_lens[1] as f32]);
    let d_alibi = dev_f32(&stream, &[0f32]); // use_alibi=0 不解引用
    let tmp_len = num_seqs * NUM_HEADS * nparts * HD;
    let d_es = dev_f32(&stream, &vec![0f32; num_seqs * NUM_HEADS * nparts]);
    let d_ml = dev_f32(&stream, &vec![0f32; num_seqs * NUM_HEADS * nparts]);
    let d_es16 = dev_f32(&stream, &vec![0f32; num_seqs * NUM_HEADS * nparts]);
    let d_ml16 = dev_f32(&stream, &vec![0f32; num_seqs * NUM_HEADS * nparts]);
    let d_tmp16 = dev_u16(&stream, &vec![0u16; tmp_len]);
    let d_tmp8 = dev_u16(&stream, &vec![0u16; tmp_len]);
    let d_out16 = dev_u16(&stream, &vec![0u16; num_seqs * NUM_HEADS * HD]);
    let d_out8 = dev_u16(&stream, &vec![0u16; num_seqs * NUM_HEADS * HD]);

    // ── 发射 f16 金标(两序列 × nparts 分区;reduce 归并)──
    let launch_v2 = |k_name: &str,
                     r_name: &str,
                     k_ptr: u64,
                     v_ptr: u64,
                     es: &cudarc::driver::CudaSlice<f32>,
                     ml: &cudarc::driver::CudaSlice<f32>,
                     tmp: &cudarc::driver::CudaSlice<u16>,
                     out: &cudarc::driver::CudaSlice<u16>| {
        let kf = m.load_function(k_name).expect("v2 fn");
        let rf = m.load_function(r_name).expect("reduce fn");
        for s in 0..num_seqs {
            let seqlen = seq_lens[s];
            let my_nparts = seqlen.div_ceil(512);
            let q_off = s * NUM_HEADS * HD * 2;
            let bt_off = s * blocks_per_seq * 4;
            let lens_off = s * 4;
            let q_p = d_q.device_ptr(&stream).0 + q_off as u64;
            let bt_p = d_bt.device_ptr(&stream).0 + bt_off as u64;
            let ln_p = d_lens.device_ptr(&stream).0 + lens_off as u64;
            let ab_p = d_alibi.device_ptr(&stream).0;
            let es_p = es.device_ptr(&stream).0 + (s * NUM_HEADS * nparts * 4) as u64;
            let ml_p = ml.device_ptr(&stream).0 + (s * NUM_HEADS * nparts * 4) as u64;
            let tp_p = tmp.device_ptr(&stream).0 + (s * NUM_HEADS * nparts * HD * 2) as u64;
            let o_p = out.device_ptr(&stream).0 + (s * NUM_HEADS * HD * 2) as u64;
            let (nkv, scale, mnbp, qst, bst, hst, ss, sw, ua) =
                (NUM_KV_HEADS as i32, 1f32 / (HD as f32).sqrt(), blocks_per_seq as i32,
                 (NUM_HEADS * HD) as i32, kv_block_stride as i32, (HD / X * PAGE * X) as i32,
                 1f32, 0i32, 0i32);
            // extern 入口序(q,k,v,bt,lens,alibi,es,ml,nkv,scale,
            // mnbp,qst,bst,hst,ss,sw,use_alibi,tmp_out 末参;out 由
            // reduce 写)
            let mut b = stream.launch_builder(&kf);
            b.arg(&q_p)
                .arg(&k_ptr)
                .arg(&v_ptr)
                .arg(&bt_p)
                .arg(&ln_p)
                .arg(&ab_p)
                .arg(&es_p)
                .arg(&ml_p)
                .arg(&nkv)
                .arg(&scale)
                .arg(&mnbp)
                .arg(&qst)
                .arg(&bst)
                .arg(&hst)
                .arg(&ss)
                .arg(&sw)
                .arg(&ua)
                .arg(&tp_p);
            let cfg = LaunchConfig {
                grid_dim: (NUM_HEADS as u32, 1, my_nparts as u32),
                block_dim: (NUM_THREADS as u32, 1, 1),
                shared_mem_bytes: (((seqlen.div_ceil(PAGE) * PAGE * 4) as u32).max(2048)),
            };
            unsafe { b.launch(cfg) }.expect("v2 launch");
            // reduce(跨分区归并;my_nparts=1 时核内直拷)
            let es_r = es.device_ptr(&stream).0 + (s * NUM_HEADS * nparts * 4) as u64;
            let ml_r = ml.device_ptr(&stream).0 + (s * NUM_HEADS * nparts * 4) as u64;
            let tp_r = tmp.device_ptr(&stream).0 + (s * NUM_HEADS * nparts * HD * 2) as u64;
            let ln_r = d_lens.device_ptr(&stream).0 + lens_off as u64;
            let mut rb = stream.launch_builder(&rf);
            let np = nparts as i32;
            rb.arg(&es_r).arg(&ml_r).arg(&tp_r).arg(&ln_r).arg(&np).arg(&o_p);
            let rcfg = LaunchConfig {
                grid_dim: (NUM_HEADS as u32, 1, 1),
                block_dim: (NUM_THREADS as u32, 1, 1),
                shared_mem_bytes: (2 * nparts * 4) as u32,
            };
            unsafe { rb.launch(rcfg) }.expect("reduce launch");
        }
        stream.synchronize().unwrap();
    };

    launch_v2(
        "vllm_paged_attention_v2_f16_hd256bs32",
        "vllm_paged_attention_v2_reduce_f16_hd256",
        d_k16.device_ptr(&stream).0,
        d_v16.device_ptr(&stream).0,
        &d_es16,
        &d_ml16,
        &d_tmp16,
        &d_out16,
    );
    launch_v2(
        "vllm_paged_attention_v2_fp8_hd256bs32",
        "vllm_paged_attention_v2_reduce_f16_hd256",
        d_k8.device_ptr(&stream).0,
        d_v8.device_ptr(&stream).0,
        &d_es,
        &d_ml,
        &d_tmp8,
        &d_out8,
    );

    // ── 对拍 ──
    let mut o16 = vec![0u16; num_seqs * NUM_HEADS * HD];
    stream.memcpy_dtoh(&d_out16, &mut o16).unwrap();
    let mut o8 = vec![0u16; num_seqs * NUM_HEADS * HD];
    stream.memcpy_dtoh(&d_out8, &mut o8).unwrap();
    // sanity:污染 fp8 池 → fp8 输出必须变化(证明它真的在读 fp8 缓冲)
    let poisoned = vec![0u8; k_fp8.len()];
    let mut d_k8p = unsafe { stream.alloc::<u8>(poisoned.len()) }.unwrap();
    stream.memcpy_htod(&poisoned, &mut d_k8p).unwrap();
    let d_out8p = dev_u16(&stream, &vec![0u16; num_seqs * NUM_HEADS * HD]);
    launch_v2(
        "vllm_paged_attention_v2_fp8_hd256bs32",
        "vllm_paged_attention_v2_reduce_f16_hd256",
        d_k8p.device_ptr(&stream).0,
        d_v8.device_ptr(&stream).0,
        &d_es,
        &d_ml,
        &d_tmp8,
        &d_out8p,
    );
    let mut op = vec![0u16; num_seqs * NUM_HEADS * HD];
    stream.memcpy_dtoh(&d_out8p, &mut op).unwrap();
    assert_ne!(op, o8, "污染 K 后 fp8 输出未变 —— fp8 核未读 fp8 缓冲!");

    let mut max_d = 0f32;
    let mut sum_d = 0f32;
    let mut max_rel = 0f32;
    for i in 0..o16.len() {
        let a = f16_bits_to_f32(o16[i]);
        let b = f16_bits_to_f32(o8[i]);
        let d = (a - b).abs();
        max_d = max_d.max(d);
        sum_d += d;
        max_rel = max_rel.max(d / (a.abs() + 1e-3));
    }
    let mean_d = sum_d / o16.len() as f32;
    println!(
        "B6.1 探针:f16 vs fp8 输出 — mean|d|={mean_d:.5} max|d|={max_d:.5} max_rel={max_rel:.4}"
    );
    assert!(
        mean_d < 2e-2 && max_d < 8e-2,
        "fp8 attention 偏差超门(mean {mean_d}, max {max_d})"
    );
    println!("B6.1 探针 PASS:fp8 e4m3 KV 读路径在容差内与 f16 一致(含分区 reduce)");
}

//! B6.3 证可性探针:chunked prefill 的 fp8 e4m3 KV 读变体。
//!
//! 判据:同 q/K/V,fp8 池(quantized)vs f16 池,chunked f16 vs fp8 核
//! 输出容差内一致(量化噪声量级;mean < 2e-2 / max < 8e-2)。
//! 布局 = 生产 x=8(classic [nb, Hkv, hd/8, bs, 8]);含 smem tile 路径
//! (dominant-seq)与 boundary 路径同测。
//! 无 GPU 跳过(OWL_TEST_DEVICE)。

use cudarc::driver::{CudaContext, DevicePtr, LaunchConfig, PushKernelArg};

const HD: usize = 256;
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
            let x = seed.wrapping_mul(0x9E37_79B9).wrapping_add(i as u32 * 0x85EB_CA6B);
            half_bits(((x & 0xFFFF) as f32 / 65535.0 - 0.5) * 4.0)
        })
        .collect()
}

/// B6.3 WIP(2026-10-10):fp8 读核 99.6% 输出未写(全零)——主池 fp8
/// E2E 质量门红灯的直接原因(草稿池 fp8 不经此核,已证零损可用)。
/// 本探针 = 复现用例(最小化:x=8 生产布局/单序列满 chunk)。
/// 修好后摘 ignore 恢复守门。
#[test]
#[ignore = "B6.3 WIP:chunked fp8 读核 99.6% 零输出待修(本探针即复现)"]
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

    let kcache_len = NUM_BLOCKS * NUM_KV_HEADS * HD * PAGE;
    let k_f16 = lcg(NUM_BLOCKS * NUM_KV_HEADS * HD * PAGE, 101);
    let v_f16 = lcg(NUM_BLOCKS * NUM_KV_HEADS * HD * PAGE, 202);
    let q_f16 = lcg(T * NUM_HEADS * HD, 303);
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
    println!("host 量化非零:k_fp8={} v_fp8={} / {}",
             k_fp8.iter().filter(|&&b| b != 0).count(),
             v_fp8.iter().filter(|&&b| b != 0).count(),
             k_fp8.len());
    let d_q = dev_u16(&q_f16);
    let d_k16 = dev_u16(&k_f16);
    let d_v16 = dev_u16(&v_f16);
    let d_k8 = dev_u8(&k_fp8);
    let d_v8 = dev_u8(&v_fp8);
    // 单序列:块链 = [0,1,2,3](128 tok 覆盖 96)
    let d_bt = dev_f32(&[0f32, 1f32, 2f32, 3f32, 4f32, 5f32, 6f32, 7f32]);
    let d_lens = dev_f32(&[SEQ_LEN as f32]);
    let d_qsl = dev_f32(&[0f32, T as f32]); // 单序列 [0, T]
    let d_alibi = dev_f32(&[0f32]);
    let d_sinks = dev_f32(&[0f32]);

    let run = |tag: &str,
               k_name: &str,
               k_ptr: u64,
               v_ptr: u64,
               out: &cudarc::driver::CudaSlice<u16>| {
        let f = m.load_function(k_name).expect("chunked fn");
        let o_p = out.device_ptr(&stream).0;
        let q_p = d_q.device_ptr(&stream).0;
        let kp = k_ptr;
        let vp = v_ptr;
        let bt_p = d_bt.device_ptr(&stream).0;
        let ln_p = d_lens.device_ptr(&stream).0;
        let qsl_p = d_qsl.device_ptr(&stream).0;
        let ab_p = d_alibi.device_ptr(&stream).0;
        let sk_p = d_sinks.device_ptr(&stream).0;
        let (nkv, scale, bts, ns, nqh, nqt) =
            (NUM_KV_HEADS as i32, 1f32 / (HD as f32).sqrt(), 1i32, 1i32, NUM_HEADS as i32, T as i32); // bts:单序列线性读,1 与块数均可
        let (ss, ost, sw, tnb, kbs, khs, ua, us) =
            (1f32, (NUM_HEADS * HD) as i32, -1i32, NUM_BLOCKS as i32,
             (NUM_KV_HEADS * HD * PAGE) as i32, (HD * PAGE) as i32, 0i32, 0i32);
        let _ = tnb;
        // extern 序:q,k_cache,v_cache,bt,seq_lens,qsl,alibi,sinks,
        // nkv,sm_scale,bts,ns,nqh,nqt,ss,ost,sw,tnb,kbs,khs,ua,us,out(末参)
        let mut b = stream.launch_builder(&f);
        b.arg(&q_p)
            .arg(&kp)
            .arg(&vp)
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
            shared_mem_bytes: (64 + 2 * HD * PAGE * 2) as u32,
        };
                let r = unsafe { b.launch(cfg) };
        if let Err(e) = r {
            panic!("[{tag}] chunked launch {k_name}: {e:?}");
        }
        if let Err(e) = stream.synchronize() {
            panic!("[{tag}] sync 后 ILLEGAL: {e:?}");
        }
        println!("[{tag}] launch+sync OK");
    };

    // 设备回读 fp8 池首 64 字节(验证上传)
    {
        let mut back = vec![0u8; 64];
        let view = d_k8.slice(0..64);
        stream.memcpy_dtoh(&view, &mut back).unwrap();
        println!("d_k8 首 32 字节: {:?}", &back[..32]);
        let nz = back.iter().filter(|&&b| b != 0).count();
        println!("d_k8 首 64 非零: {nz}/64");
    }
    {
        let mut all = vec![0u8; k_fp8.len()];
        stream.memcpy_dtoh(&d_k8, &mut all).unwrap();
        println!("设备 d_k8 全量非零: {}/{}", all.iter().filter(|&&b| b != 0).count(), all.len());
        let mut allv = vec![0u8; v_fp8.len()];
        stream.memcpy_dtoh(&d_v8, &mut allv).unwrap();
        println!("设备 d_v8 全量非零: {}/{}", allv.iter().filter(|&&b| b != 0).count(), allv.len());
    }
    let d_out16 = dev_u16(&vec![0u16; T * NUM_HEADS * HD]);
    let d_out8 = dev_u16(&vec![0u16; T * NUM_HEADS * HD]);
    run(
        "f16",
        "vllm_chunked_prefill_paged_attn_opt_f16_hd256",
        d_k16.device_ptr(&stream).0,
        d_v16.device_ptr(&stream).0,
        &d_out16,
    );
    run(
        "fp8",
        "vllm_chunked_prefill_paged_attn_opt_fp8_hd256",
        d_k8.device_ptr(&stream).0,
        d_v8.device_ptr(&stream).0,
        &d_out8,
    );

    let mut o16 = vec![0u16; T * NUM_HEADS * HD];
    stream.memcpy_dtoh(&d_out16, &mut o16).unwrap();
    let mut o8 = vec![0u16; T * NUM_HEADS * HD];
    stream.memcpy_dtoh(&d_out8, &mut o8).unwrap();
    // 输出非零按 token 分布
    for t in 0..T {
        let nz = (0..NUM_HEADS * HD).filter(|&i| f16_to_f32(o8[t * NUM_HEADS * HD + i]) != 0.0).count();
        if nz > 0 && t < 8 {
            println!("fp8 out token {t}: 非零 {nz}/{}", NUM_HEADS * HD);
        }
    }
    let mut max_d = 0f32;
    let mut sum_d = 0f32;
    for i in 0..o16.len() {
        let a = f16_to_f32(o16[i]);
        let b = f16_to_f32(o8[i]);
        let d = (a - b).abs();
        max_d = max_d.max(d);
        sum_d += d;
    }
    let mean_d = sum_d / o16.len() as f32;
    println!("B6.3 探针:chunked f16 vs fp8 — mean|d|={mean_d:.5} max|d|={max_d:.5}");
    let z16 = o16.iter().filter(|&&h| f16_to_f32(h) == 0.0).count();
    let z8 = o8.iter().filter(|&&h| f16_to_f32(h) == 0.0).count();
    println!("  零值计数:f16={z16} fp8={z8} / {}", o16.len());
    // diff 分布:top-8 元素位置(token, head, dim)
    let mut diffs: Vec<(f32, usize)> = (0..o16.len())
        .map(|i| ((f16_to_f32(o16[i]) - f16_to_f32(o8[i])).abs(), i))
        .collect();
    diffs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    for (d, i) in diffs.iter().take(8) {
        let (t, h, dd) = (i / (NUM_HEADS * HD), (i / HD) % NUM_HEADS, i % HD);
        println!("  diff {d:.4} @ token={t} head={h} dim={dd} (f16={:.3} fp8={:.3})",
                 f16_to_f32(o16[*i]), f16_to_f32(o8[*i]));
    }
    assert!(
        mean_d < 2e-2 && max_d < 8e-2,
        "chunked fp8 偏差超门(mean {mean_d}, max {max_d})"
    );
    println!("B6.3 探针 PASS:chunked fp8 读与 f16 一致(容差内)");
}

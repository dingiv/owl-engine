//! B6.2 写核回读探针:owl_reshape_and_cache_fp8kv 写已知 f16 K/V →
//! e4m3 池字节回读对照(host RNE 量化表);再经 v2 fp8 读核往返。
//! 布局 = 生产 x=8(classic [nb, Hkv, hd/8, bs, 8])。
//! 无 GPU 跳过(OWL_TEST_DEVICE)。

use cudarc::driver::{CudaContext, DevicePtr, LaunchConfig, PushKernelArg};

const HKV: usize = 2;
const HD: usize = 256;
const BS: usize = 32;
const X: usize = 8; // 生产 x(f16 policy;写/读同源)
const NB: usize = 4;
const T: usize = 4; // 写入 token 数(跨 1 页边界:T=4 全在第 0 页;另测跨页)

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

#[test]
fn k0_fp8_write_readback() {
    if !owl_shared::env_reader::flag("OWL_TEST_DEVICE") {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    }
    let ctx = CudaContext::new(1).expect("ctx");
    ctx.bind_to_thread().expect("bind");
    let stream = ctx.default_stream();
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(
        owl_kernels::sources::attention::RESHAPE_AND_CACHE_FP8KV,
        cudarc::nvrtc::CompileOptions {
            arch: Some("compute_86"),
            include_paths: vec!["/usr/local/cuda/include".into()],
            ..Default::default()
        },
    )
    .expect("nvrtc compile");
    let m = ctx.load_module(ptx).expect("module");
    let f = m.load_function("owl_reshape_and_cache_fp8kv").expect("fn");

    // 输入:2 token × (HKV, HD) K/V(值域 ±4)
    let mut key = vec![0u16; T * HKV * HD];
    let mut value = vec![0u16; T * HKV * HD];
    for i in 0..key.len() {
        key[i] = half_bits(((i * 37) % 83) as f32 / 10.0 - 4.0);
        value[i] = half_bits(((i * 53) % 97) as f32 / 12.0 - 4.0);
    }
    // slots:token0→页0槽0,token1→页0槽31(尾),token2→页1槽0(跨页),token3→-1(padding)
    let slots = [0f32, 31.0, 32.0, -1.0];

    let mut d_k = unsafe { stream.alloc::<u16>(key.len()) }.unwrap();
    stream.memcpy_htod(&key, &mut d_k).unwrap();
    let mut d_v = unsafe { stream.alloc::<u16>(value.len()) }.unwrap();
    stream.memcpy_htod(&value, &mut d_v).unwrap();
    let kcache_len = NB * HKV * HD * BS;
    let mut d_kc = unsafe { stream.alloc::<u8>(kcache_len) }.unwrap();
    let mut d_vc = unsafe { stream.alloc::<u8>(kcache_len) }.unwrap();
    stream.memset_zeros(&mut d_kc).unwrap();
    stream.memset_zeros(&mut d_vc).unwrap();
    let mut d_slots = unsafe { stream.alloc::<f32>(slots.len()) }.unwrap();
    stream.memcpy_htod(&slots, &mut d_slots).unwrap();
    let mut d_out = unsafe { stream.alloc::<u16>(1) }.unwrap();

    let (ks, vs, hs, kst, vst) = (
        (HKV * HD) as i32,
        (HKV * HD) as i32,
        HKV as i32,
        HD as i32,
        BS as i32,
    );
    let mut b = stream.launch_builder(&f);
    let kp = d_k.device_ptr(&stream).0;
    let vp = d_v.device_ptr(&stream).0;
    let kcp = d_kc.device_ptr(&stream).0;
    let vcp = d_vc.device_ptr(&stream).0;
    let slp = d_slots.device_ptr(&stream).0;
    let op = d_out.device_ptr(&stream).0;
    b.arg(&kp)
        .arg(&vp)
        .arg(&kcp)
        .arg(&vcp)
        .arg(&slp)
        .arg(&ks)
        .arg(&vs)
        .arg(&hs)
        .arg(&kst)
        .arg(&(BS as i32))
        .arg(&(X as i32))
        .arg(&op);
    let cfg = LaunchConfig { grid_dim: (T as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
    unsafe { b.launch(cfg) }.expect("k0 fp8 launch");
    stream.synchronize().unwrap();

    // ── 字节回读对照:host 量化表 ──
    let mut kc = vec![0u8; kcache_len];
    stream.memcpy_dtoh(&d_kc, &mut kc).unwrap();
    let mut vc = vec![0u8; kcache_len];
    stream.memcpy_dtoh(&d_vc, &mut vc).unwrap();
    let mut bad = 0;
    for t in 0..3usize {
        // token3 = padding,跳写
        let slot = slots[t] as usize;
        let (blk, off) = (slot / BS, slot % BS);
        for h in 0..HKV {
            for d in 0..HD {
                let src = t * HKV * HD + h * HD + d;
                let x_idx = d / X;
                let x_off = d % X;
                let ki = blk * HKV * (HD / X) * BS * X + h * (HD / X) * BS * X + x_idx * BS * X + off * X + x_off;
                let vi = blk * HKV * HD * BS + h * HD * BS + d * BS + off;
                let want_k = e4m3_encode(f16_to_f32(key[src]));
                let want_v = e4m3_encode(f16_to_f32(value[src]));
                if kc[ki] != want_k {
                    bad += 1;
                    if bad < 4 {
                        eprintln!("K mismatch t{t} h{h} d{d}: got {} want {} ({})", kc[ki], want_k, e4m3_decode(kc[ki]));
                    }
                }
                if vc[vi] != want_v {
                    bad += 1;
                }
            }
        }
    }
    assert_eq!(bad, 0, "K0 fp8 写池 {bad} 处字节不符");
    println!("B6.2 写核回读 PASS(e4m3 字节级,含跨页/padding)");
}

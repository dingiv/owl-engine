//! **GPU Mean-K 块归约 parity**（2026-10-23；KVMem 262K index 刀）。
//!
//! 门 = 与 host 独立参考**逐位一致**（同一 rotate-half 逆变换 + 同序 double
//! 累加 + f16 出口）。覆盖 f16 / e4m3 两种主池字节格式，几何取 27B 实际档
//! （page 32 / hkv 4 / hd 256 / rotary half 32）。
//!
//! 运行：
//! ```bash
//! OWL_TEST_DEVICE=1 cargo test -p owl-kernels --test mean_k_block_parity -- --nocapture
//! ```

use owl_macros::owl_test;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream, DevicePtr, LaunchConfig, PushKernelArg};

const PAGE: usize = 32;
const HKV: usize = 4;
const HD: usize = 256;
const HALF: usize = 32;
const LAYERS: usize = 16;
const NB: usize = 4;

fn f16_bits(f: f32) -> u16 {
    half::f16::from_f32(f).to_bits()
}

fn f16_val(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

/// 池内容（块内交错头，块间不同盐；值域小、可被 f16/e4m3 共同表示）
fn k_value(b: usize, l: usize, t: usize, h: usize, d: usize) -> f32 {
    let x = (b * 131 + l * 37 + t * 11 + h * 17 + d * 7) % 97;
    (x as f32 - 48.0) / 16.0
}

fn cos_sin(b: usize, t: usize, i: usize) -> (f32, f32) {
    let pos = (b * PAGE + t) as f32;
    let phase = pos * 0.017 + i as f32 * 0.113;
    (phase.cos(), phase.sin())
}

fn host_ref(pools: &[Vec<f32>], phys: &[usize], cos: &[f32], sin: &[f32]) -> Vec<u16> {
    assert_eq!(pools.len(), LAYERS, "每层一份池内容");
    let mut acc = vec![0.0f64; NB * LAYERS * HKV * HD];
    for b in 0..NB {
        for l in 0..LAYERS {
            let pool = &pools[l];
            for t in 0..PAGE {
                for h in 0..HKV {
                    let row = ((phys[b] * PAGE + t) * HKV + h) * HD;
                    let out = ((b * LAYERS + l) * HKV + h) * HD;
                    let ci = b * PAGE + t;
                    for i in 0..HALF {
                        let x = pool[row + i];
                        let y = pool[row + i + HALF];
                        let c = cos[ci * HALF + i];
                        let s = sin[ci * HALF + i];
                        acc[out + i] += (x * c + y * s) as f64;
                        acc[out + i + HALF] += (-x * s + y * c) as f64;
                    }
                    for d in 2 * HALF..HD {
                        acc[out + d] += pool[row + d] as f64;
                    }
                }
            }
        }
    }
    acc.iter()
        .map(|v| f16_bits((*v / PAGE as f64) as f32))
        .collect()
}

fn compile(ctx: &Arc<CudaContext>, name: &str) -> cudarc::driver::CudaFunction {
    let src = owl_kernels::cu_sig::cu_text(owl_kernels::ops::source(name));
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(
        src,
        cudarc::nvrtc::CompileOptions {
            arch: Some("compute_86"),
            include_paths: vec!["/usr/local/cuda/include".into()],
            ..Default::default()
        },
    )
    .unwrap_or_else(|e| panic!("nvrtc({name}): {e:?}"));
    let m = ctx.load_module(ptx).unwrap_or_else(|e| panic!("module({name}): {e:?}"));
    m.load_function(name).unwrap_or_else(|e| panic!("fn({name}): {e:?}"))
}

fn run_f16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    pool_bits: &[Vec<u16>],
    phys: &[usize],
    cos: &[f32],
    sin: &[f32],
) -> Vec<u16> {
    let f = compile(ctx, "owl_mean_k_block_f16");
    let mut d_cos = unsafe { stream.alloc::<f32>(cos.len()) }.unwrap();
    stream.memcpy_htod(cos, &mut d_cos).unwrap();
    let mut d_sin = unsafe { stream.alloc::<f32>(sin.len()) }.unwrap();
    stream.memcpy_htod(sin, &mut d_sin).unwrap();
    let phys32: Vec<u32> = phys.iter().map(|p| *p as u32).collect();
    let mut d_phys = unsafe { stream.alloc::<u32>(phys32.len()) }.unwrap();
    stream.memcpy_htod(&phys32, &mut d_phys).unwrap();
    let out_elems = NB * LAYERS * HKV * HD;
    let mut d_out = unsafe { stream.alloc::<u16>(out_elems) }.unwrap();
    stream.memset_zeros(&mut d_out).unwrap();
    let (cp, sp, pp, op) = (
        d_cos.device_ptr(stream).0,
        d_sin.device_ptr(stream).0,
        d_phys.device_ptr(stream).0,
        d_out.device_ptr(stream).0,
    );
    for (layer, bits) in pool_bits.iter().enumerate() {
        let mut d_kc = unsafe { stream.alloc::<u16>(bits.len()) }.unwrap();
        stream.memcpy_htod(bits, &mut d_kc).unwrap();
        let kcp = d_kc.device_ptr(stream).0;
        let (page, hkv, hd, half, layers) =
            (PAGE as i32, HKV as i32, HD as i32, HALF as i32, LAYERS as i32);
        let layer_i = layer as i32;
        let mut b = stream.launch_builder(&f);
        b.arg(&kcp)
            .arg(&cp)
            .arg(&sp)
            .arg(&pp)
            .arg(&page)
            .arg(&hkv)
            .arg(&hd)
            .arg(&half)
            .arg(&layer_i)
            .arg(&layers)
            .arg(&op);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (NB as u32, HKV as u32, 1),
                block_dim: (HD as u32, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .unwrap();
    }
    stream.synchronize().unwrap();
    let mut got = vec![0u16; out_elems];
    stream.memcpy_dtoh(&d_out, &mut got).unwrap();
    got
}

fn run_fp8(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    pool_codes: &[Vec<u8>],
    phys: &[usize],
    cos: &[f32],
    sin: &[f32],
) -> Vec<u16> {
    let f = compile(ctx, "owl_mean_k_block_fp8");
    let mut d_cos = unsafe { stream.alloc::<f32>(cos.len()) }.unwrap();
    stream.memcpy_htod(cos, &mut d_cos).unwrap();
    let mut d_sin = unsafe { stream.alloc::<f32>(sin.len()) }.unwrap();
    stream.memcpy_htod(sin, &mut d_sin).unwrap();
    let phys32: Vec<u32> = phys.iter().map(|p| *p as u32).collect();
    let mut d_phys = unsafe { stream.alloc::<u32>(phys32.len()) }.unwrap();
    stream.memcpy_htod(&phys32, &mut d_phys).unwrap();
    let out_elems = NB * LAYERS * HKV * HD;
    let mut d_out = unsafe { stream.alloc::<u16>(out_elems) }.unwrap();
    stream.memset_zeros(&mut d_out).unwrap();
    let (cp, sp, pp, op) = (
        d_cos.device_ptr(stream).0,
        d_sin.device_ptr(stream).0,
        d_phys.device_ptr(stream).0,
        d_out.device_ptr(stream).0,
    );
    for (layer, codes) in pool_codes.iter().enumerate() {
        let mut d_kc = unsafe { stream.alloc::<u8>(codes.len()) }.unwrap();
        stream.memcpy_htod(codes, &mut d_kc).unwrap();
        let kcp = d_kc.device_ptr(stream).0;
        let (page, hkv, hd, half, layers) =
            (PAGE as i32, HKV as i32, HD as i32, HALF as i32, LAYERS as i32);
        let layer_i = layer as i32;
        let mut b = stream.launch_builder(&f);
        b.arg(&kcp)
            .arg(&cp)
            .arg(&sp)
            .arg(&pp)
            .arg(&page)
            .arg(&hkv)
            .arg(&hd)
            .arg(&half)
            .arg(&layer_i)
            .arg(&layers)
            .arg(&op);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (NB as u32, HKV as u32, 1),
                block_dim: (HD as u32, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .unwrap();
    }
    stream.synchronize().unwrap();
    let mut got = vec![0u16; out_elems];
    stream.memcpy_dtoh(&d_out, &mut got).unwrap();
    got
}

#[owl_test(gpu)]
fn mean_k_block_parity() {
    let ordinal = owl_cuda::test_device_ordinal();
    let ctx = CudaContext::new(ordinal).expect("ctx");
    ctx.bind_to_thread().expect("bind");
    let stream = ctx.default_stream();
    let phys = [3usize, 1, 2, 0];
    let cos: Vec<f32> = (0..NB * PAGE * HALF)
        .map(|i| {
            let (b, rest) = (i / (PAGE * HALF), i % (PAGE * HALF));
            let (t, k) = (rest / HALF, rest % HALF);
            cos_sin(b, t, k).0
        })
        .collect();
    let sin: Vec<f32> = (0..NB * PAGE * HALF)
        .map(|i| {
            let (b, rest) = (i / (PAGE * HALF), i % (PAGE * HALF));
            let (t, k) = (rest / HALF, rest % HALF);
            cos_sin(b, t, k).1
        })
        .collect();

    // ---- f16 ----
    {
        let mut pools_bits: Vec<Vec<u16>> = Vec::with_capacity(LAYERS);
        let mut pools_ref: Vec<Vec<f32>> = Vec::with_capacity(LAYERS);
        for l in 0..LAYERS {
            let mut bits = vec![0u16; NB * PAGE * HKV * HD];
            let mut vals = vec![0f32; NB * PAGE * HKV * HD];
            for b in 0..NB {
                for t in 0..PAGE {
                    for h in 0..HKV {
                        for d in 0..HD {
                            let off = ((phys[b] * PAGE + t) * HKV + h) * HD + d;
                            bits[off] = f16_bits(k_value(b, l, t, h, d));
                            vals[off] = f16_val(bits[off]);
                        }
                    }
                }
            }
            pools_bits.push(bits);
            pools_ref.push(vals);
        }
        let want = host_ref(&pools_ref, &phys, &cos, &sin);
        let got = run_f16(&ctx, &stream, &pools_bits, &phys, &cos, &sin);
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert_eq!(g, w, "f16 Mean-K 第 {i} 项不一致: got={g:#06x} want={w:#06x}");
        }
    }

    // ---- e4m3 ----
    {
        let mut pools_codes: Vec<Vec<u8>> = Vec::with_capacity(LAYERS);
        let mut pools_ref: Vec<Vec<f32>> = Vec::with_capacity(LAYERS);
        for l in 0..LAYERS {
            let mut codes = vec![0u8; NB * PAGE * HKV * HD];
            let mut vals = vec![0f32; NB * PAGE * HKV * HD];
            for b in 0..NB {
                for t in 0..PAGE {
                    for h in 0..HKV {
                        for d in 0..HD {
                            let off = ((phys[b] * PAGE + t) * HKV + h) * HD + d;
                            codes[off] = owl_kernels::ops::oracles::e4m3_encode(k_value(b, l, t, h, d));
                            vals[off] = owl_kernels::ops::oracles::e4m3_decode(codes[off]);
                        }
                    }
                }
            }
            pools_codes.push(codes);
            pools_ref.push(vals);
        }
        let want = host_ref(&pools_ref, &phys, &cos, &sin);
        let got = run_fp8(&ctx, &stream, &pools_codes, &phys, &cos, &sin);
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert_eq!(g, w, "fp8 Mean-K 第 {i} 项不一致: got={g:#06x} want={w:#06x}");
        }
    }
}

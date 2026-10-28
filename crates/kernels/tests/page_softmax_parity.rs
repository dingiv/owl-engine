//! **page-softmax 打分核 parity**（2026-10-28；P1-3 的 GPU scorer 第一步）。
//!
//! 门 = 设备三段（logits → max/Σexp → 逐块累加）与**宿主参考**在同一口径下
//! 一致：`out[b] = (1/(L·hq)) · Σ_t Σ_h softmax_b(q̃[t,h]·mk[b,g(h)]·scale)`。
//!
//! 精度口径 = 参照实现的**显式 FP16 操作数边界**（q 与 mean-K 都取 f16，
//! f32 累加）⇒ 与 f32 宿主参考的容差按 f16 取整量级（≤2e-3 相对）。
//!
//! 运行：
//! ```bash
//! OWL_TEST_DEVICE=1 cargo test -p owl-kernels --test page_softmax_parity -- --nocapture
//! ```

use owl_macros::owl_test;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream, DevicePtr, LaunchConfig, PushKernelArg};

const BT: usize = 32; // 与 .cu 的 OWL_KVSCORE_BT 一致
const W: usize = 8; // query 行数
const HQ: usize = 4; // 查询头
const HKV: usize = 2; // KV 头(GQA 比 = 2)
const HD: usize = 64;
const NB: usize = 40; // 块数(**故意不是 BT 的整数倍**,覆盖尾瓦片)
const LAYERS: usize = 2;

fn gqa_map() -> Vec<u32> {
    (0..HQ).map(|h| (h / (HQ / HKV)) as u32).collect()
}

/// 确定性伪随机(值域小,便于 f16 表示)
fn val(k: usize) -> f32 {
    ((k * 37 + 11) % 61) as f32 / 61.0 - 0.5
}

fn host_q(t: usize, h: usize, d: usize) -> f32 {
    val(t * 7 + h * 13 + d * 3 + 1)
}
fn host_mk(b: usize, g: usize, d: usize) -> f32 {
    val(b * 17 + g * 29 + d * 5 + 101)
}

fn f16(v: f32) -> f32 {
    half::f16::from_f32(v).to_f32()
}

fn f16_bits(v: f32) -> u16 {
    half::f16::from_f32(v).to_bits()
}

/// 宿主参考(与 `owl_store::score_blocks(SoftmaxMean)` 逐式同源;
/// 操作数先过 f16 边界以对齐设备口径)
fn host_ref() -> Vec<f32> {
    let scale = 1.0f32 / (HD as f32).sqrt();
    let gqa = gqa_map();
    let mut out = vec![0.0f32; NB];
    for l in 0..LAYERS {
        let _ = l; // mk 每层各一份;参考里两层同分布(与设备同)
        for t in 0..W {
            let mut acc = vec![0.0f32; NB];
            for h in 0..HQ {
                let g = gqa[h] as usize;
                let mut logits = vec![0.0f32; NB];
                for b in 0..NB {
                    let mut dot = 0.0f32;
                    for d in 0..HD {
                        dot += f16(host_q(t, h, d)) * f16(host_mk(b, g, d));
                    }
                    logits[b] = dot * scale;
                }
                let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let s: f32 = logits.iter().map(|z| (z - m).exp()).sum();
                for b in 0..NB {
                    acc[b] += (logits[b] - m).exp() / s;
                }
            }
            for b in 0..NB {
                out[b] += acc[b] / (LAYERS * HQ) as f32;
            }
        }
    }
    out
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

fn run(ctx: &Arc<CudaContext>, stream: &Arc<CudaStream>) -> Vec<f32> {
    // 设备缓冲：q / mk / gqa / z / ms / out
    let q_bits: Vec<u16> = (0..W * HQ * HD).map(|i| f16_bits(host_q(i / (HQ * HD), (i / HD) % HQ, i % HD))).collect();
    let mut d_q = unsafe { stream.alloc::<u16>(q_bits.len()) }.unwrap();
    stream.memcpy_htod(&q_bits, &mut d_q).unwrap();

    let gqa = gqa_map();
    let mut d_gqa = unsafe { stream.alloc::<u32>(gqa.len()) }.unwrap();
    stream.memcpy_htod(&gqa, &mut d_gqa).unwrap();

    let mut d_z = unsafe { stream.alloc::<u16>(W * HQ * NB) }.unwrap();
    let mut d_ms = unsafe { stream.alloc::<f32>(W * HQ * 2) }.unwrap();
    let mut d_out = unsafe { stream.alloc::<f32>(NB) }.unwrap();
    stream.memset_zeros(&mut d_out).unwrap();

    let f_logits = compile(ctx, "owl_kvscore_logits_f16");
    let f_maxsum = compile(ctx, "owl_kvscore_maxsum");
    let f_acc = compile(ctx, "owl_kvscore_acc");

    for layer in 0..LAYERS {
        // 本层 mean-K（两层同内容 ⇒ 参考里的 `let _ = l` 成立）
        let mk_bits: Vec<u16> = (0..NB * HKV * HD)
            .map(|i| f16_bits(host_mk(i / (HKV * HD), (i / HD) % HKV, i % HD)))
            .collect();
        let mut d_mk = unsafe { stream.alloc::<u16>(mk_bits.len()) }.unwrap();
        stream.memcpy_htod(&mk_bits, &mut d_mk).unwrap();

        let (qp, mkp, gp, zp, msp, op) = (
            d_q.device_ptr(stream).0,
            d_mk.device_ptr(stream).0,
            d_gqa.device_ptr(stream).0,
            d_z.device_ptr(stream).0,
            d_ms.device_ptr(stream).0,
            d_out.device_ptr(stream).0,
        );
        let (w, hq, hkv, hd, nb) = (W as i32, HQ as i32, HKV as i32, HD as i32, NB as i32);
        let scale = 1.0f32 / (HD as f32).sqrt();
        let mut b = stream.launch_builder(&f_logits);
        b.arg(&qp).arg(&mkp).arg(&gp)
            .arg(&w).arg(&hq).arg(&hkv).arg(&hd).arg(&nb)
            .arg(&scale).arg(&zp);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (NB.div_ceil(BT) as u32, HKV as u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: (BT * HD * 2) as u32,
            })
        }
        .unwrap();

        let mut b = stream.launch_builder(&f_maxsum);
        b.arg(&zp).arg(&w).arg(&hq).arg(&nb).arg(&msp);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (W as u32, HQ as u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .unwrap();

        let layers_i = LAYERS as i32;
        let mut b = stream.launch_builder(&f_acc);
        b.arg(&zp).arg(&msp).arg(&w).arg(&hq).arg(&nb).arg(&layers_i).arg(&op);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (NB.div_ceil(BT) as u32, 4, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .unwrap();

        stream.synchronize().unwrap();
    }

    let mut got = vec![0.0f32; NB];
    stream.memcpy_dtoh(&d_out, &mut got).unwrap();
    got
}

#[owl_test(gpu)]
fn page_softmax_matches_host_reference() {
    let ordinal = owl_cuda::test_device_ordinal();
    let ctx = CudaContext::new(ordinal).expect("ctx");
    let stream = ctx.new_stream().expect("stream");
    let got = run(&ctx, &stream);
    let want = host_ref();

    // ① 形状级 sanity:Σ_b out[b] = W(每个 (层,t,h) 的 softmax 沿块和为 1,
    //    再乘 1/(L·hq) ⇒ 只剩 W 倍)
    let sum: f32 = want.iter().sum();
    assert!(
        (sum - W as f32).abs() <= 1e-2 * W as f32,
        "宿主参考自检失败:Σ_b out={sum} ≠ W={W}"
    );

    // ② 逐块相对误差(数值分布有 40 个块 ⇒ 逐块概率 ~2.5%)
    let mut worst = 0.0f32;
    let mut worst_b = 0usize;
    for b in 0..NB {
        let denom = want[b].abs().max(1e-6);
        let rel = (got[b] - want[b]).abs() / denom;
        if rel > worst {
            worst = rel;
            worst_b = b;
        }
    }
    eprintln!("[page-softmax] 逐块最大相对误差 {worst:.3e}(块 {worst_b})");
    eprintln!("[page-softmax] got[0..4]={:?}", &got[..4]);
    eprintln!("[page-softmax] want[0..4]={:?}", &want[..4]);
    assert!(
        worst <= 2e-3,
        "设备与宿主参考不符:最大相对误差 {worst:.3e} > 2e-3(块 {worst_b},got={}, want={})",
        got[worst_b],
        want[worst_b]
    );
}

/// **生产几何基准**（不是门，只测代价）：72k 上下文 ⇒ nb=2240 块，
/// W=64（生产窗）、L=16 全注意力层、hq=24 / hkv=8 / hd=256。
///
/// 验收线（§8.4）：单边界 **≤ 2 ms 量级**（host 版在同几何下 ≈ 4.4 s）。
/// 运行：`OWL_TEST_DEVICE=0000:42:00.0 cargo test --release -p owl-kernels \
///        --test page_softmax_parity -- --nocapture page_softmax_bench`
#[owl_test(gpu)]
fn page_softmax_bench_production_shape() {
    const BW_: usize = 64; // query 行
    const BHQ: usize = 24;
    const BHKV: usize = 8;
    const BHD: usize = 256;
    const BNB: usize = 2240;
    const BL: usize = 16;

    let ordinal = owl_cuda::test_device_ordinal();
    let ctx = CudaContext::new(ordinal).expect("ctx");
    let stream = ctx.new_stream().expect("stream");

    let q_bits: Vec<u16> = (0..BW_ * BHQ * BHD)
        .map(|i| half::f16::from_f32(val(i % 61)).to_bits())
        .collect();
    let mk_bits: Vec<u16> = (0..BNB * BHKV * BHD)
        .map(|i| half::f16::from_f32(val((i * 7) % 61)).to_bits())
        .collect();
    let gqa: Vec<u32> = (0..BHQ).map(|h| (h / (BHQ / BHKV)) as u32).collect();

    let mut d_q = unsafe { stream.alloc::<u16>(q_bits.len()) }.unwrap();
    stream.memcpy_htod(&q_bits, &mut d_q).unwrap();
    let mut d_mk = unsafe { stream.alloc::<u16>(mk_bits.len()) }.unwrap();
    stream.memcpy_htod(&mk_bits, &mut d_mk).unwrap();
    let mut d_gqa = unsafe { stream.alloc::<u32>(gqa.len()) }.unwrap();
    stream.memcpy_htod(&gqa, &mut d_gqa).unwrap();
    let mut d_z = unsafe { stream.alloc::<u16>(BW_ * BHQ * BNB) }.unwrap();
    let mut d_ms = unsafe { stream.alloc::<f32>(BW_ * BHQ * 2) }.unwrap();
    let mut d_out = unsafe { stream.alloc::<f32>(BNB) }.unwrap();
    stream.memset_zeros(&mut d_out).unwrap();

    let f_logits = compile(&ctx, "owl_kvscore_logits_f16");
    let f_maxsum = compile(&ctx, "owl_kvscore_maxsum");
    let f_acc = compile(&ctx, "owl_kvscore_acc");
    let scale = 1.0f32 / (BHD as f32).sqrt();
    let (qp, mkp, gp, zp, msp, op) = (
        d_q.device_ptr(&stream).0,
        d_mk.device_ptr(&stream).0,
        d_gqa.device_ptr(&stream).0,
        d_z.device_ptr(&stream).0,
        d_ms.device_ptr(&stream).0,
        d_out.device_ptr(&stream).0,
    );

    let layers_i = BL as i32;
    let launch_all = || {
        for _ in 0..BL {
            let (w, hq, hkv, hd, nb) =
                (BW_ as i32, BHQ as i32, BHKV as i32, BHD as i32, BNB as i32);
            let mut b = stream.launch_builder(&f_logits);
            b.arg(&qp).arg(&mkp).arg(&gp)
                .arg(&w).arg(&hq).arg(&hkv).arg(&hd).arg(&nb)
                .arg(&scale).arg(&zp);
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: (BNB.div_ceil(BT) as u32, BHKV as u32, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: (BT * BHD * 2) as u32,
                })
            }
            .unwrap();
            let mut b = stream.launch_builder(&f_maxsum);
            b.arg(&zp).arg(&w).arg(&hq).arg(&nb).arg(&msp);
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: (BW_ as u32, BHQ as u32, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .unwrap();
            let mut b = stream.launch_builder(&f_acc);
            b.arg(&zp).arg(&msp).arg(&w).arg(&hq).arg(&nb).arg(&layers_i).arg(&op);
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: (BNB.div_ceil(BT) as u32, 8, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .unwrap();
        }
        stream.synchronize().unwrap();
    };

    launch_all(); // 热身
    let t0 = std::time::Instant::now();
    const REPS: usize = 5;
    for _ in 0..REPS {
        launch_all();
    }
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / REPS as f64;
    // 分段账（定位热点）:只跑 ① / 只跑 ② / 只跑 ③
    let (w, hq, hkv, hd, nb) = (BW_ as i32, BHQ as i32, BHKV as i32, BHD as i32, BNB as i32);
    let only = |which: u8| {
        for _ in 0..BL {
            if which == 1 {
                let mut b = stream.launch_builder(&f_logits);
                b.arg(&qp).arg(&mkp).arg(&gp)
                    .arg(&w).arg(&hq).arg(&hkv).arg(&hd).arg(&nb)
                    .arg(&scale).arg(&zp);
                unsafe { b.launch(LaunchConfig { grid_dim: (BNB.div_ceil(BT) as u32, BW_ as u32, 1), block_dim: (128,1,1), shared_mem_bytes: ((BHQ*BHD + BT*BHD)*2) as u32 }) }.unwrap();
            } else if which == 2 {
                let mut b = stream.launch_builder(&f_maxsum);
                b.arg(&zp).arg(&w).arg(&hq).arg(&nb).arg(&msp);
                unsafe { b.launch(LaunchConfig { grid_dim: (BW_ as u32, BHQ as u32, 1), block_dim: (256,1,1), shared_mem_bytes: 0 }) }.unwrap();
            } else {
                let mut b = stream.launch_builder(&f_acc);
                b.arg(&zp).arg(&msp).arg(&w).arg(&hq).arg(&nb).arg(&layers_i).arg(&op);
                unsafe { b.launch(LaunchConfig { grid_dim: (BNB.div_ceil(BT) as u32, 8, 1), block_dim: (128,1,1), shared_mem_bytes: 0 }) }.unwrap();
            }
        }
        stream.synchronize().unwrap();
    };
    for which in 1..=3u8 {
        only(which);
        let t = std::time::Instant::now();
        for _ in 0..REPS { only(which); }
        eprintln!("[page-softmax bench] 段 ①②③ 之 {which} = {:.2} ms", t.elapsed().as_secs_f64()*1000.0/REPS as f64);
    }
    eprintln!(
        "[page-softmax bench] nb={BNB} W={BW_} L={BL} hq={BHQ} hkv={BHKV} hd={BHD} ⇒ **{ms:.2} ms/边界**(整 16 层)"
    );
    // 验收线：≤ 20 ms（host 同几何 ≈ 4.4 s ⇒ 200× 以上即可装机）
    assert!(ms <= 20.0, "GPU page-softmax 太慢:{ms:.2} ms > 20 ms(装机线)");
}

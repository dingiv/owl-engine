//! marlin W4A16-AWQ 带宽微基准(2026-10-08,E5 性能会战)。
//! 真实 27B 形状(gate/up 合并 / down / GDN qkvz),m ∈ {1, 8},
//! 事件计时 50 发取均值,报有效带宽 vs HBM 峰值(3090 Ti ~1008GB/s)。
//! 目的:判定 verify 轮 marlin 16.93ms(56% 占比)是核本体效率还是
//! 图内空隙 —— 微基准脱离图/发射器,单核直测。
//! 垃圾位型:qweight/zeros 全 0x某常数(nibble 定值,dequant 数值
//! 不发散即可;计时与数值无关)。无 GPU 跳过(OWL_TEST_DEVICE)。

use cudarc::driver::{CudaContext, DevicePtr};

const G: i32 = 32; // AWQ groupsize
const PEAK_GBS: f64 = 1008.0; // 3090 Ti HBM 峰值(规格)

fn f16_one() -> u16 {
    0x3C00
}

/// 单形状单 m 计时(µs/发;50 发事件均值,5 发暖机)
fn bench_shape(
    ctx: &std::sync::Arc<CudaContext>,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    n: usize,
    k: usize,
    m: usize,
) -> (f64, f64) {
    // 分配(垃圾位型;计时与数值无关)
    let a_h: Vec<u16> = vec![f16_one(); m * k]; // A f16 全 1.0
    let b_h: Vec<i32> = vec![0x2222_2222_i32; (k / 8) * n]; // qweight nibble=2
    let c_h: Vec<u16> = vec![0u16; m * n];
    let s_h: Vec<u16> = vec![f16_one(); (k / G as usize) * n]; // scales 全 1.0
    let z_h: Vec<i32> = vec![0_i32; (k / G as usize) * (n / 8) + 16]; // zp=0 位型
    let ws_len = owl_kernels::family::marlin::v2_workspace_len(n);
    let ws_h: Vec<i32> = vec![0_i32; ws_len];
    let tmp_h: Vec<u8> = vec![0u8; 64 << 20]; // c_tmp 64MB(上游 par 归约余量)

    let (mut a, mut b, mut c, mut s, mut z, mut ws, mut tmp);
    unsafe {
        a = stream.alloc::<u16>(a_h.len()).unwrap();
        b = stream.alloc::<i32>(b_h.len()).unwrap();
        c = stream.alloc::<u16>(c_h.len()).unwrap();
        s = stream.alloc::<u16>(s_h.len()).unwrap();
        z = stream.alloc::<i32>(z_h.len()).unwrap();
        ws = stream.alloc::<i32>(ws_h.len()).unwrap();
        tmp = stream.alloc::<u8>(tmp_h.len()).unwrap();
    }
    stream.memcpy_htod(&a_h, &mut a).unwrap();
    stream.memcpy_htod(&b_h, &mut b).unwrap();
    stream.memcpy_htod(&c_h, &mut c).unwrap();
    stream.memcpy_htod(&s_h, &mut s).unwrap();
    stream.memcpy_htod(&z_h, &mut z).unwrap();
    stream.memcpy_htod(&ws_h, &mut ws).unwrap();

    let (a_p, _) = a.device_ptr(&stream);
    let (b_p, _) = b.device_ptr(&stream);
    let (c_p, _) = c.device_ptr(&stream);
    let (s_p, _) = s.device_ptr(&stream);
    let (z_p, _) = z.device_ptr(&stream);
    let (ws_p, _) = ws.device_ptr(&stream);
    let (tmp_p, _) = tmp.device_ptr(&stream);
    let cu_stream = stream.cu_stream() as usize;

    let launch = |a_p, b_p, c_p, s_p, z_p, ws_p, tmp_p| unsafe {
        owl_kernels::family::marlin::gemm_v2_awq_raw(
            a_p as *const u16,
            b_p as *const i32,
            c_p as *mut u16,
            s_p as *const u16,
            z_p as *const i32,
            tmp_p as *const std::ffi::c_void,
            m as i32,
            n as i32,
            k as i32,
            ws_p as *mut i32,
            G,
            ctx.ordinal() as i32,
            cu_stream,
        )
        .expect("marlin awq launch")
    };

    // 暖机 5 发 + sync
    for _ in 0..5 {
        launch(a_p, b_p, c_p, s_p, z_p, ws_p, tmp_p);
    }
    stream.synchronize().unwrap();

    // 墙钟计时 50 发(批粒度;单发 >50µs,发射开销占比可忽略)
    const ITERS: usize = 50;
    let t0 = std::time::Instant::now();
    for _ in 0..ITERS {
        launch(a_p, b_p, c_p, s_p, z_p, ws_p, tmp_p);
    }
    stream.synchronize().unwrap();
    let us = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

    // 有效带宽:int4 权重 + scales(f16)
    let bytes = (n * k / 2 + (k / G as usize) * n * 2 + m * k * 2 + m * n * 2) as f64;
    let gbs = bytes / (us * 1e-6) / 1e9;
    (us, gbs)
}

#[test]
fn marlin_bandwidth_27b_shapes() {
    let Some(dev) = owl_shared::env_reader::parse::<usize>("OWL_TEST_DEVICE")
    else {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    };
    let ctx = CudaContext::new(dev).expect("ctx");
    ctx.bind_to_thread().unwrap();
    let stream = ctx.default_stream();

    // 27B 真实形状:(n, k) — gate/up 合并 / down / GDN qkvz / attn o_proj
    let shapes: &[(usize, usize, &str)] = &[
        (34816, 5120, "mlp gate+up"),
        (5120, 17408, "mlp down"),
        (16384, 5120, "gdn in_qkvz"),
        (5120, 6144, "gdn out_proj"),
    ];
    println!(
        "{:>7} {:>6} {:>10} {:>10} {:>10}",
        "shape", "m", "µs/launch", "GB/s", "%peak"
    );
    for &(n, k, _tag) in shapes {
        for m in [1usize, 8] {
            let (us, gbs) = bench_shape(&ctx, &stream, n, k, m);
            println!(
                "{:>7} {:>6} {:>10.1} {:>10.1} {:>9.1}%",
                format!("{n}x{k}"),
                m,
                us,
                gbs,
                gbs / PEAK_GBS * 100.0
            );
        }
    }
}

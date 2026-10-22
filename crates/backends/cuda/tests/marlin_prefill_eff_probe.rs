//! **marlin 预填效率探针**(2026-10-21):27B **TP2 每 rank 真实形状 × m=512**,
//! 报 时间 / TFLOPS / %峰 / 有效带宽 —— 回答"marlin 占 59.8% GPU 时间是核本体慢,
//! 还是本来就该花这么多"。
//!
//! 判据:
//! - `%峰` = TFLOPS / (SM 数 × boost 时钟 × 1024 FLOP/clk/SM)(fp16 dense TC);
//! - `GB/s` 与 `%HBM` 用来判"计算受限 vs 带宽受限"(W4A16 在 m 大时应当计算受限);
//! - 同形状 `W4A8` 一列(激活 int8)给"换激活精度能拿多少"的直接读数。
//!
//! 运行:
//! ```text
//! OWL_TEST_DEVICE=01:00.0 cargo test --release -p owl-kernels --features marlin \
//!   --test marlin_prefill_eff_probe -- --ignored --nocapture
//! ```
use cudarc::driver::{CudaContext, DevicePtr};

const G: i32 = 32; // AWQ groupsize

/// 27B **TP2 每 rank**的真实形状(m=prefill chunk 512)。
/// 来源:权重头 shape(见 model.safetensors 头)+ TP 规则(行并行切 n / 列并行切 k)。
const SHAPES: &[(usize, usize, &str)] = &[
    (8704, 5120, "mlp.gate/up  行并行"),
    (5120, 8704, "mlp.down     列并行"),
    (5120, 5120, "gdn.in_qkv   行并行"),
    (3072, 5120, "gdn.in_z     行并行"),
    (5120, 3072, "gdn.out      列并行"),
    (6144, 5120, "attn.q       行并行"),
    (512, 5120, "attn.k/v     行并行"),
    (5120, 3072, "attn.o       列并行"),
];

fn launch_awq(
    ctx: &std::sync::Arc<CudaContext>,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    n: usize,
    k: usize,
    m: usize,
) -> f64 {
    let (a, b, c, s, z, ws, tmp);
    let ws_len = owl_kernels::family::marlin::v2_workspace_len(n);
    unsafe {
        a = stream.alloc::<u16>(m * k).unwrap();
        b = stream.alloc::<i32>((k / 8) * n).unwrap();
        c = stream.alloc::<u16>(m * n).unwrap();
        s = stream.alloc::<u16>((k / G as usize) * n).unwrap();
        z = stream.alloc::<i32>((k / G as usize) * (n / 8) + 16).unwrap();
        ws = stream.alloc::<i32>(ws_len).unwrap();
        tmp = stream.alloc::<u8>(64 << 20).unwrap();
    }
    let (a_p, _) = a.device_ptr(stream);
    let (b_p, _) = b.device_ptr(stream);
    let (c_p, _) = c.device_ptr(stream);
    let (s_p, _) = s.device_ptr(stream);
    let (z_p, _) = z.device_ptr(stream);
    let (ws_p, _) = ws.device_ptr(stream);
    let (tmp_p, _) = tmp.device_ptr(stream);
    let cu = stream.cu_stream() as usize;
    let go = || unsafe {
        owl_kernels::family::marlin::gemm_v2_awq_raw(
            a_p as *const u16, b_p as *const i32, c_p as *mut u16,
            s_p as *const u16, z_p as *const i32,
            tmp_p as *const std::ffi::c_void,
            m as i32, n as i32, k as i32, ws_p as *mut i32, G,
            ctx.ordinal() as i32, cu,
        )
        .expect("marlin awq launch")
    };
    for _ in 0..5 {
        go();
    }
    stream.synchronize().unwrap();
    const ITERS: usize = 20;
    let t0 = std::time::Instant::now();
    for _ in 0..ITERS {
        go();
    }
    stream.synchronize().unwrap();
    t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64
}

fn launch_w4a8(
    ctx: &std::sync::Arc<CudaContext>,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    n: usize,
    k: usize,
    m: usize,
) -> f64 {
    let (a, asc, b, c, s, z, ws);
    let ws_len = owl_kernels::family::marlin::v2_workspace_len(n);
    unsafe {
        a = stream.alloc::<i8>(m * k).unwrap();
        asc = stream.alloc::<f32>(m).unwrap();
        b = stream.alloc::<i32>((k / 8) * n).unwrap();
        c = stream.alloc::<u16>(m * n).unwrap();
        s = stream.alloc::<u16>((k / G as usize) * n).unwrap();
        z = stream.alloc::<i32>((k / G as usize) * (n / 8) + 16).unwrap();
        ws = stream.alloc::<i32>(ws_len).unwrap();
    }
    let (a_p, _) = a.device_ptr(stream);
    let (asc_p, _) = asc.device_ptr(stream);
    let (b_p, _) = b.device_ptr(stream);
    let (c_p, _) = c.device_ptr(stream);
    let (s_p, _) = s.device_ptr(stream);
    let (z_p, _) = z.device_ptr(stream);
    let (ws_p, _) = ws.device_ptr(stream);
    let cu = stream.cu_stream() as usize;
    let go = || unsafe {
        owl_kernels::family::marlin::gemm_v2_awq_w4a8_raw(
            a_p as *const i8, b_p as *const i32, c_p as *mut u16,
            s_p as *const u16, z_p as *const i32, asc_p as *const f32,
            std::ptr::null(), m as i32, n as i32, k as i32,
            ws_p as *mut i32, G, ctx.ordinal() as i32, cu,
            false,
        )
        .expect("marlin w4a8 launch")
    };
    for _ in 0..5 {
        go();
    }
    stream.synchronize().unwrap();
    const ITERS: usize = 20;
    let t0 = std::time::Instant::now();
    for _ in 0..ITERS {
        go();
    }
    stream.synchronize().unwrap();
    t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64
}

/// fp16 dense 峰值(TFLOPS)= SM × boost clk × 1024(Ampere GA10x:每 SM 每拍
/// 1024 个 fp16 TC FLOP,FP32 累加口径)。
fn peak_tflops(ctx: &std::sync::Arc<CudaContext>) -> f64 {
    use cudarc::driver::sys::{CUdevice_attribute, cuDeviceGetAttribute};
    let dev = ctx.ordinal() as i32;
    let mut sm = 0i32;
    let mut khz = 0i32;
    unsafe {
        cuDeviceGetAttribute(&mut sm, CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
        cuDeviceGetAttribute(&mut khz, CUdevice_attribute::CU_DEVICE_ATTRIBUTE_CLOCK_RATE, dev);
    }
    sm as f64 * khz as f64 * 1e3 * 1024.0 / 1e12
}

#[ignore = "bench: marlin 预填效率(m=512, TP2 每 rank 形状)"]
#[test]
fn marlin_prefill_eff() {
    let Some(dev) = owl_cuda::test_device_ordinal_opt() else {
        owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
        return;
    };
    let ctx = CudaContext::new(dev).expect("ctx");
    ctx.bind_to_thread().unwrap();
    let stream = ctx.default_stream();
    let peak = peak_tflops(&ctx);
    // 带宽用规格值做参照(3080=760, 3090 Ti=1008 GB/s)——
    // 本探针只回答"计算受限还是带宽受限",不需要实测 HBM 峰值。
    let hbm_gbs = if ctx.name().unwrap_or_default().contains("3090") { 1008.0 } else { 760.0 };
    println!(
        "[device] {} ordinal={dev} · fp16 dense 峰值 {peak:.1} TFLOPS · HBM 参照 {hbm_gbs} GB/s",
        ctx.name().unwrap_or_default()
    );
    println!(
        "{:>11} {:>16} {:>10} {:>9} {:>8} {:>9} {:>7} {:>8}",
        "n x k", "tag", "µs", "TFLOPS", "%峰", "GB/s", "%HBM", "a8 胜出"
    );
    for &(n, k, tag) in SHAPES {
        let m = 512usize;
        let us16 = launch_awq(&ctx, &stream, n, k, m);
        let us8 = launch_w4a8(&ctx, &stream, n, k, m);
        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        let tf16 = flops / (us16 * 1e-6) / 1e12;
        let _tf8 = flops / (us8 * 1e-6) / 1e12; // W4A8 臂:本探针暂只报 W4A16
        // int4 权重 + f16 scales + 激活/输出(f16)
        let bytes = (n * k / 2 + (k / G as usize) * n * 2 + m * k * 2 + m * n * 2) as f64;
        let gbs = bytes / (us16 * 1e-6) / 1e9;
        println!(
            "{:>11} {:>16} {:>10.1} {:>9.2} {:>7.1}% {:>9.1} {:>6.1}% {:>7.1}%",
            format!("{n}x{k}"),
            tag,
            us16,
            tf16,
            tf16 / peak * 100.0,
            gbs,
            gbs / hbm_gbs * 100.0,
            (us16 - us8) / us16 * 100.0
        );
    }
}

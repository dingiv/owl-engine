//! marlin W4A16-AWQ 带宽微基准(2026-10-08,E5 性能会战)。
//! 真实 27B 形状(gate/up 合并 / down / GDN qkvz),m ∈ {1, 8},
//! 事件计时 50 发取均值,报有效带宽 vs HBM 峰值(3090 Ti ~1008GB/s)。
//! 目的:判定 verify 轮 marlin 16.93ms(56% 占比)是核本体效率还是
//! 图内空隙 —— 微基准脱离图/发射器,单核直测。
//! 垃圾位型:qweight/zeros 全 0x某常数(nibble 定值,dequant 数值
//! 不发散即可;计时与数值无关)。无 GPU 跳过(OWL_TEST_DEVICE)。
//!
//! ⚠️ 编译门(2026-10-13):本文件引 `family::marlin`,该模块受
//! `marlin` feature 门控 —— 缺此 `cfg` 则默认 `cargo test -p owl-kernels`
//! (features = [])在**编译期**报 E0433(与 `#[ignore]` 无关:ignore 只跳
//! 运行,不跳编译)。同 `ops_*_isolation` 的 `feature = "device"` 先例。
#![cfg(feature = "marlin")]

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

    let (mut a, mut b, mut c, mut s, mut z, mut ws, tmp);
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

    let (a_p, _) = a.device_ptr(stream);
    let (b_p, _) = b.device_ptr(stream);
    let (c_p, _) = c.device_ptr(stream);
    let (s_p, _) = s.device_ptr(stream);
    let (z_p, _) = z.device_ptr(stream);
    let (ws_p, _) = ws.device_ptr(stream);
    let (tmp_p, _) = tmp.device_ptr(stream);
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

/// **W4A8 臂**单形状计时(与 a16 臂同几何、同 B/scales 分配量;差异 = A 换 int8 +
/// 多一个 a_scales 块,且 **c_tmp 由 C 侧流序自管**(计时含每发一次
/// stream-ordered malloc/free —— 那是生产姿势,如实计入)。
///
/// 垃圾位型取"小整数"档(scale 位型 0x0100 = i16 256、nibble=2、a=1):
/// int16 尺度被**当整数**乘进 int32 累加器(见 marlin_template.h kS8 分支),
/// 用 0x3C00(=15360)会在 m×k 大形状上把 int32 打爆 → 数值垃圾(计时不受影响,
/// 但没必要引入)。返回 (µs/发, GB/s, TFLOPS)。
fn bench_shape_a8(
    ctx: &std::sync::Arc<CudaContext>,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    n: usize,
    k: usize,
    m: usize,
    w4a8_m8: bool,
) -> (f64, f64, f64) {
    let a_h: Vec<i8> = vec![1i8; m * k]; // A int8
    let as_h: Vec<f32> = vec![1.0f32; m]; // per-token 尺度
    let b_h: Vec<i32> = vec![0x2222_2222_i32; (k / 8) * n];
    let c_h: Vec<u16> = vec![0u16; m * n];
    let s_h: Vec<u16> = vec![0x0100u16; (k / G as usize) * n]; // int16 位型 256
    let z_h: Vec<i32> = vec![0_i32; (k / G as usize) * (n / 8) + 16];
    let ws_len = owl_kernels::family::marlin::v2_workspace_len(n);
    let ws_h: Vec<i32> = vec![0_i32; ws_len];

    let (mut a, mut asd, mut b, mut c, mut s, mut z, mut ws);
    unsafe {
        a = stream.alloc::<i8>(a_h.len()).unwrap();
        asd = stream.alloc::<f32>(as_h.len()).unwrap();
        b = stream.alloc::<i32>(b_h.len()).unwrap();
        c = stream.alloc::<u16>(c_h.len()).unwrap();
        s = stream.alloc::<u16>(s_h.len()).unwrap();
        z = stream.alloc::<i32>(z_h.len()).unwrap();
        ws = stream.alloc::<i32>(ws_h.len()).unwrap();
    }
    stream.memcpy_htod(&a_h, &mut a).unwrap();
    stream.memcpy_htod(&as_h, &mut asd).unwrap();
    stream.memcpy_htod(&b_h, &mut b).unwrap();
    stream.memcpy_htod(&c_h, &mut c).unwrap();
    stream.memcpy_htod(&s_h, &mut s).unwrap();
    stream.memcpy_htod(&z_h, &mut z).unwrap();
    stream.memcpy_htod(&ws_h, &mut ws).unwrap();

    let (a_p, _) = a.device_ptr(stream);
    let (as_p, _) = asd.device_ptr(stream);
    let (b_p, _) = b.device_ptr(stream);
    let (c_p, _) = c.device_ptr(stream);
    let (s_p, _) = s.device_ptr(stream);
    let (z_p, _) = z.device_ptr(stream);
    let (ws_p, _) = ws.device_ptr(stream);
    let cu_stream = stream.cu_stream() as usize;

    let launch = |a_p, as_p, b_p, c_p, s_p, z_p, ws_p| unsafe {
        owl_kernels::family::marlin::gemm_v2_awq_w4a8_raw(
            a_p as *const i8,
            b_p as *const i32,
            c_p as *mut u16,
            s_p as *const u16,
            z_p as *const i32,
            as_p as *const f32,
            std::ptr::null(), // c_tmp:C 侧自管
            m as i32,
            n as i32,
            k as i32,
            ws_p as *mut i32,
            G,
            ctx.ordinal() as i32,
            cu_stream,
            w4a8_m8,
        )
        .expect("marlin w4a8 launch")
    };

    for _ in 0..5 {
        launch(a_p, as_p, b_p, c_p, s_p, z_p, ws_p);
    }
    stream.synchronize().unwrap();

    const ITERS: usize = 50;
    let t0 = std::time::Instant::now();
    for _ in 0..ITERS {
        launch(a_p, as_p, b_p, c_p, s_p, z_p, ws_p);
    }
    stream.synchronize().unwrap();
    let us = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

    let bytes = (n * k / 2 + (k / G as usize) * n * 2 + m * k + m * 4 + m * n * 2) as f64;
    let gbs = bytes / (us * 1e-6) / 1e9;
    let tflops = 2.0 * m as f64 * n as f64 * k as f64 / (us * 1e-6) / 1e12;
    (us, gbs, tflops)
}

/// **W4A8 vs W4A16 杠杆实测**(2026-10-16;27B 真实形状 × prefill/decode 两端 m)
///
/// 唯一目的:回答"权威 +27~30% 在**本机 3080** 能不能复现"——bench 全线的
/// 前置判据(W4A8 只在 prefill 有杠杆:decode m=1 是权重带宽墙,int8 激活不减权重流量)。
#[ignore = "bench: marlin W4A8 vs W4A16 杠杆(bench 全线前置判据)"]
#[test]
fn marlin_w4a8_vs_w4a16_lever() {
    let Some(dev) = owl_cuda::test_device_ordinal_opt()
    else {
        owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
        return;
    };
    let ctx = CudaContext::new(dev).expect("ctx");
    ctx.bind_to_thread().unwrap();
    let stream = ctx.default_stream();
    println!(
        "[device] ordinal={dev} name={:?} total_mem={} MiB",
        ctx.name().unwrap_or_default(),
        ctx.total_mem().unwrap_or(0) / (1 << 20)
    );

    let shapes: &[(usize, usize, &str)] = &[
        (34816, 5120, "mlp gate+up"),
        (5120, 17408, "mlp down"),
        (16384, 5120, "gdn in_qkvz"),
        (5120, 6144, "gdn out_proj"),
    ];
    println!(
        "{:>14} {:>6} {:>11} {:>11} {:>9} {:>11} {:>11}",
        "shape", "m", "a16 µs", "a8 µs", "Δus", "a16 TF", "a8 TF"
    );
    for &(n, k, _tag) in shapes {
        for m in [1usize, 7, 8, 512, 2048, 4096] {
            let (us16, _gbs16) = bench_shape(&ctx, &stream, n, k, m);
            let (us8, _gbs8, tf8) = bench_shape_a8(&ctx, &stream, n, k, m, false);
            let tf16 = 2.0 * m as f64 * n as f64 * k as f64 / (us16 * 1e-6) / 1e12;
            println!(
                "{:>14} {:>6} {:>11.1} {:>11.1} {:>8.1}% {:>11.3} {:>11.3}",
                format!("{n}x{k}"),
                m,
                us16,
                us8,
                (us8 - us16) / us16 * 100.0,
                tf16,
                tf8
            );
        }
    }
}

#[ignore = "bench: marlin 带宽微基准(E5 性能会战)"]
#[test]
fn marlin_bandwidth_27b_shapes() {
    let Some(dev) = owl_cuda::test_device_ordinal_opt()
    else {
        owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
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

/// **W4A8 int8 8-row 对拍**(2026-10-22;famp-marlin port 的硬门)。
///
/// 同一组随机数据跑两次:
/// 1. `w4a8_m8=true`(kS8 grouped,m8 8-row);
/// 2. `w4a8_m8=false`(旧 16-row 路)。
/// 两次都与 host 显式反量化参考比;再把 m8 与 16-row 直接比。
/// 形状取 m=8(g32, n=256, k=512)—— 正好命中 spec verify 的 m8 档。
#[test]
fn marlin_w4a8_m8_parity() {
    let Some(dev) = owl_cuda::test_device_ordinal_opt() else {
        owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
        return;
    };
    let ctx = CudaContext::new(dev).expect("ctx");
    ctx.bind_to_thread().unwrap();
    let stream = ctx.default_stream();
    let (m, n, k, g) = (8usize, 256usize, 512usize, G as usize);
    let groups = k / g;

    // 非退化数据:A 有正负、q∈[8,15]、zp∈[0,7]、s 幅值真实。
    let a_i8: Vec<i8> = (0..m * k).map(|i| (((i * 13 + 7) % 255) as i32 - 127) as i8).collect();
    let a_s: Vec<f32> = (0..m).map(|i| 0.02 + i as f32 * 0.0031).collect();
    let q: Vec<u8> = (0..n * k).map(|i| ((i * 7 + 3) % 8 + 8) as u8).collect();
    let s: Vec<f32> = (0..n * groups).map(|i| 0.02 + ((i * 13) % 13) as f32 * 0.004).collect();
    let zp: Vec<u8> = (0..n * groups).map(|i| ((i * 5 + 2) % 8) as u8).collect();

    let b_packed = owl_kernels::family::marlin::repack::pack_marlin_b_a8(&q, k, n);
    let (s_bits, factor) = owl_kernels::family::marlin::repack::pack_marlin_s_a8(&s, n, groups);
    let z_packed = owl_kernels::family::marlin::repack::pack_marlin_z_a8(&zp, n, groups);
    let a_scales: Vec<f32> = a_s.iter().map(|x| x * factor).collect();
    let ws_len = owl_kernels::family::marlin::v2_workspace_len(n);
    let ws = vec![0i32; ws_len];
    let zero = vec![0u16; m * n];

    let (mut da, mut das, mut db, mut ds, mut dz, mut dws, mut dc);
    unsafe {
        da = stream.alloc::<i8>(m * k).unwrap();
        das = stream.alloc::<f32>(m).unwrap();
        db = stream.alloc::<i32>(b_packed.len()).unwrap();
        ds = stream.alloc::<u16>(s_bits.len()).unwrap();
        dz = stream.alloc::<i32>(z_packed.len()).unwrap();
        dws = stream.alloc::<i32>(ws.len()).unwrap();
        dc = stream.alloc::<u16>(m * n).unwrap();
    }
    stream.memcpy_htod(&a_i8, &mut da).unwrap();
    stream.memcpy_htod(&a_scales, &mut das).unwrap();
    stream.memcpy_htod(&b_packed, &mut db).unwrap();
    stream.memcpy_htod(&s_bits, &mut ds).unwrap();
    stream.memcpy_htod(&z_packed, &mut dz).unwrap();
    stream.memcpy_htod(&ws, &mut dws).unwrap();
    stream.memcpy_htod(&zero, &mut dc).unwrap();
    let (a_p, _) = da.device_ptr(&stream);
    let (as_p, _) = das.device_ptr(&stream);
    let (b_p, _) = db.device_ptr(&stream);
    let (c_p, _) = dc.device_ptr(&stream);
    let (s_p, _) = ds.device_ptr(&stream);
    let (z_p, _) = dz.device_ptr(&stream);
    let (ws_p, _) = dws.device_ptr(&stream);
    let cu_stream = stream.cu_stream() as usize;

    let launch = |w4a8_m8: bool| unsafe {
        owl_kernels::family::marlin::gemm_v2_awq_w4a8_raw(
            a_p as *const i8,
            b_p as *const i32,
            c_p as *mut u16,
            s_p as *const u16,
            z_p as *const i32,
            as_p as *const f32,
            std::ptr::null(),
            m as i32,
            n as i32,
            k as i32,
            ws_p as *mut i32,
            G,
            dev as i32,
            cu_stream,
            w4a8_m8,
        )
        .expect("marlin w4a8 launch")
    };
    let run = |w4a8_m8: bool| {
        launch(w4a8_m8);
        stream.synchronize().unwrap();
        let mut buf = vec![0u16; m * n];
        stream.memcpy_dtoh(&dc, &mut buf).unwrap();
        buf.into_iter()
            .map(|b| half::f16::from_bits(b).to_f32())
            .collect::<Vec<f32>>()
    };

    let got_m8 = run(true);
    let got_16 = run(false);

    // host 参考:与 oracle 同一式(原始 q/s/zp + 折因子后的 a_scale)
    let mut want = vec![0f32; m * n];
    for r in 0..m {
        for nn in 0..n {
            let mut acc = 0f32;
            for kk in 0..k {
                let qv = q[nn * k + kk] as f32;
                let zv = zp[nn * groups + kk / g] as f32;
                acc += a_i8[r * k + kk] as f32 * (qv - zv) * s[nn * groups + kk / g];
            }
            // kernel 出口 = a_s × (q−zp) × s(s_bits 已按 max 归一);
            // a_scales 里折的 factor 与 s_bits 的 1/factor 抵消 ⇒ host 用原始 a_s。
            want[r * n + nn] = acc * a_s[r];
        }
    }

    let stat = |got: &[f32]| {
        let mut max_abs = 0.0f32;
        let mut max_ref = 0.0f32;
        for (x, w) in got.iter().zip(&want) {
            max_abs = max_abs.max((x - w).abs());
            max_ref = max_ref.max(w.abs());
        }
        max_abs / max_ref.max(1e-6)
    };
    let e8 = stat(&got_m8);
    let e16 = stat(&got_16);
    let mut diff = 0.0f32;
    for (x, y) in got_m8.iter().zip(&got_16) {
        diff = diff.max((x - y).abs());
    }
    let ref_max = want.iter().fold(0.0f32, |a, b| a.max(b.abs())).max(1e-6);
    println!(
        "[w4a8-m8] m={m} n={n} k={k} g={g}: m8 rel={e8:.3e} · 16row rel={e16:.3e} · m8-vs-16 max|Δ|/max={:.3e}",
        diff / ref_max
    );
    assert!(e8 < 2e-2, "W4A8 m8 与 host 参考偏差过大:{e8}");
    assert!(e16 < 2e-2, "W4A8 16-row 与 host 参考偏差过大:{e16}");
    assert!(diff / ref_max < 2e-2, "W4A8 m8 与 16-row 偏差过大:{:.3e}", diff / ref_max);
}

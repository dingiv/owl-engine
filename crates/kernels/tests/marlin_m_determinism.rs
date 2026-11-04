//! **W4A8 投影 GEMM 的 M 依赖探针**(2026-11-04,per-token K scale 的准入前置评估)。
//!
//! 问题:同一个 token(同一行激活、同一份权重)的输出,是否会随 call 的 M
//! (prefill chunk 行数)变化?若会,INT8 KV 的 K 量化就会把这份差异放大成
//! 质量门翻车(`docs/bench/int8-1k-quality-gate-2026-11-04.md` §3)。
//!
//! 判据:以 M=2048 为基准,逐 M 比对**前 min(M, 基准) 行**的输出 half 位型。
//! 分离三个 M 依赖源:
//!   1. 线程档表:`thread_m_blocks > 1` 走 large_batch{64,256,256},
//!      `== 1` 走 small_batch{128,128,256} ⇒ **thread_k 不同**(K 分区不同);
//!   2. `thread_m_blocks = ceil(M/16) capped 4` ⇒ M≤16 时 1,M≥64 时 4;
//!   3. **K-slice**:`gridDim.x = sms × blocks_per_sm`(=84) 固定,而
//!      `global_mn_tiles = ceil(M/(tmb*16)) * ceil(n/thread_n)`;
//!      tiles < 84 ⇒ 沿 K 切段 + global_reduce(部分和再相加)。
//!
//! 跑法(需 GPU):
//! `OWL_TEST_DEVICE=41:00.0 cargo test -p owl-kernels --release --features "device,marlin" \
//!   --test marlin_m_determinism -- --nocapture --test-threads=1`
#![cfg(all(feature = "marlin", feature = "device"))]

use cudarc::driver::DevicePtr;

const G: i32 = 32; // AWQ groupsize

fn f16_to_f32(h: u16) -> f32 {
    let sgn = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let e = ((h >> 10) & 0x1F) as i32;
    let m = (h & 0x3FF) as f32;
    if e == 0 {
        return sgn * m * 2f32.powi(-24);
    }
    if e == 31 {
        return if m == 0.0 { sgn * f32::INFINITY } else { f32::NAN };
    }
    sgn * (1.0 + m / 1024.0) * 2f32.powi(e - 15)
}

/// 小整数位型(同 `marlin_bw_bench` 的 W4A8 档):nibble=2、int16 scale 256、
/// a=±小整数 ⇒ int32 累加器不会溢出,数值可比较。
fn run_shape(m: usize, n: usize, k: usize) -> Vec<u16> {
    run_shape_cfg(m, n, k, -1, -1).expect("auto 档必须可用")
}

/// `tk`/`tn` = -1 ⇒ auto(生产路径);否则强制线程档(直测"分块是否影响数值")。
/// 返回 None = 该档位对此形状无效(launcher 报 3/4/5)。
fn run_shape_cfg(m: usize, n: usize, k: usize, tk: i32, tn: i32) -> Option<Vec<u16>> {
    let ctx = owl_cuda::test_device_ordinal_opt()
        .map(|o| cudarc::driver::CudaContext::new(o).expect("cuda ctx"))
        .expect("OWL_TEST_DEVICE 未设");
    let stream = ctx.default_stream();

    let a_h: Vec<i8> = (0..m * k).map(|i| ((i * 7 + 3) % 17) as i8 - 8).collect();
    let as_h: Vec<f32> = vec![1.0f32; m];
    let b_h: Vec<i32> = vec![0x2222_2222_i32; (k / 8) * n];
    let c_h: Vec<u16> = vec![0u16; m * n];
    let s_h: Vec<u16> = vec![0x0100u16; (k / G as usize) * n];
    let z_h: Vec<i32> = vec![0_i32; (k / G as usize) * (n / 8) + 16];
    let ws_len = owl_kernels::family::marlin::v2_workspace_len(n);
    let ws_h: Vec<i32> = vec![0_i32; ws_len];
    // c_tmp:W4A8 的 partial-sum 面(slice_count>1 时走 global_reduce_fp32)
    let tmp_h: Vec<u8> = vec![0u8; 64 << 20];

    let (mut a, mut asc, mut b, mut c, mut s, mut z, mut ws, mut tmp);
    unsafe {
        a = stream.alloc::<i8>(a_h.len()).unwrap();
        asc = stream.alloc::<f32>(as_h.len()).unwrap();
        b = stream.alloc::<i32>(b_h.len()).unwrap();
        c = stream.alloc::<u16>(c_h.len()).unwrap();
        s = stream.alloc::<u16>(s_h.len()).unwrap();
        z = stream.alloc::<i32>(z_h.len()).unwrap();
        ws = stream.alloc::<i32>(ws_h.len()).unwrap();
        tmp = stream.alloc::<u8>(tmp_h.len()).unwrap();
    }
    stream.memcpy_htod(&a_h, &mut a).unwrap();
    stream.memcpy_htod(&as_h, &mut asc).unwrap();
    stream.memcpy_htod(&b_h, &mut b).unwrap();
    stream.memcpy_htod(&c_h, &mut c).unwrap();
    stream.memcpy_htod(&s_h, &mut s).unwrap();
    stream.memcpy_htod(&z_h, &mut z).unwrap();
    stream.memcpy_htod(&ws_h, &mut ws).unwrap();

    let (a_p, _) = a.device_ptr(&stream);
    let (as_p, _) = asc.device_ptr(&stream);
    let (b_p, _) = b.device_ptr(&stream);
    let (c_p, _) = c.device_ptr(&stream);
    let (s_p, _) = s.device_ptr(&stream);
    let (z_p, _) = z.device_ptr(&stream);
    let (ws_p, _) = ws.device_ptr(&stream);
    let (tmp_p, _) = tmp.device_ptr(&stream);
    let cu_stream = stream.cu_stream() as usize;

    let mut ok = true;
    unsafe {
        if tk >= 0 && tn >= 0 {
            if let Err(code) = owl_kernels::family::marlin::gemm_v2_awq_w4a8_cfg_raw(
                a_p as *const i8,
                b_p as *const i32,
                c_p as *mut u16,
                s_p as *const u16,
                z_p as *const i32,
                as_p as *const f32,
                m as i32,
                n as i32,
                k as i32,
                ws_p as *mut i32,
                G,
                ctx.ordinal() as i32,
                cu_stream,
                tk,
                tn,
            ) {
                println!("    (cfg tk={tk} tn={tn} m={m} 不可用: err={code})");
                ok = false;
            }
        } else {
        owl_kernels::family::marlin::gemm_v2_awq_w4a8_raw(
            a_p as *const i8,
            b_p as *const i32,
            c_p as *mut u16,
            s_p as *const u16,
            z_p as *const i32,
            as_p as *const f32,
            tmp_p as *const std::ffi::c_void,
            m as i32,
            n as i32,
            k as i32,
            ws_p as *mut i32,
            G,
            ctx.ordinal() as i32,
            cu_stream,
            false, // w4a8_m8:生产缺省关
        )
        .expect("marlin w4a8 launch");
        }
    }
    if !ok { return None; }
    stream.synchronize().unwrap();
    let mut out = vec![0u16; m * n];
    stream.memcpy_dtoh(&c, &mut out).unwrap();
    Some(out)
}

fn diff(a: &[u16], b: &[u16], rows: usize, n: usize) -> (usize, f32) {
    let mut bad = 0usize;
    let mut mx = 0f32;
    for r in 0..rows {
        for j in 0..n {
            let (x, y) = (a[r * n + j], b[r * n + j]);
            if x != y {
                bad += 1;
                let d = (f16_to_f32(x) - f16_to_f32(y)).abs();
                mx = mx.max(d);
            }
        }
    }
    (bad, mx)
}

fn sweep(n: usize, k: usize, ms: &[usize], base_m: usize) {
    println!("=== shape n={n} k={k} 基准 M={base_m} ===");
    let base = run_shape(base_m, n, k);
    for &m in ms {
        let got = run_shape(m, n, k);
        let rows = m.min(base_m);
        let (bad, mx) = diff(&got, &base, rows, n);
        let verdict = if bad == 0 { "位同" } else { "**不同**" };
        println!(
            "  M={m:<5} 比对 {rows} 行 × {n} 列: 差异 {bad:>9} / {:>9}  max|Δ|={mx:.6}  {verdict}",
            rows * n
        );
    }
}

#[test]
fn w4a8_gemm_m_dependence() {
    if owl_cuda::test_device_ordinal_opt().is_none() {
        owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
        return;
    }
    // 27B 的真实投影形状:qkvz / out_proj(n=5120,k=5120)、gate_up 合并(n=34816,k=5120)
    sweep(5120, 5120, &[1, 8, 16, 64, 128, 256, 320, 512, 1024], 2048);
    sweep(34816, 5120, &[16, 64, 320, 1024], 2048);

    // ---- 线程档对照:若 int32 累加确实精确,换 thread_k/thread_n(改 K 分区)
    //      与 M 都该**位同**;这同时否掉"选核档位造成 M 依赖"的假设。----
    println!("=== 线程档对照 n=5120 k=5120(基准 = auto @ M=2048) ===");
    let base = run_shape(2048, 5120, 5120);
    for &(m, tk, tn) in &[(16usize, 64, 256), (16, 128, 128), (16, 128, 64),
                          (1024, 128, 128), (1024, 128, 64), (2048, 64, 128)] {
        let Some(got) = run_shape_cfg(m, 5120, 5120, tk, tn) else { continue };
        let rows = m.min(2048);
        let (bad, mx) = diff(&got, &base, rows, 5120);
        println!("  M={m:<5} tk={tk:<4} tn={tn:<4} 差异 {bad:>9} / {:>9}  max|Δ|={mx:.6}  {}",
                 rows * 5120, if bad == 0 { "位同" } else { "**不同**" });
    }
}

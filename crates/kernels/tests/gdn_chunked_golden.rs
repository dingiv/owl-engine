//! GDN chunked delta rule —— FLA AOT cubin **算子层金标对拍**(2026-10-03)。
//!
//! 金标 = FLA 0.6.0 在本机(16/48/128,chunk 64,owl 引擎逐 chunk 形态)
//! 跑出的五级阶段张量(testdata/gdn_fla/cases/,f32 raw bin)。本测试把
//! server 编排完全剥出去,直接:读金标 → 上传 → 逐核发射 → 逐阶段对拍
//! —— 每一核独立定位。
//!
//! 形状/发射常量与 `gdn_chunked.rs` 契约一致(h 变体 = 32768/w2/BV64/h0)。
//! 无 GPU 跳过(OWL_TEST_DEVICE)。

use cudarc::driver::{CudaContext, DevicePtr, LaunchConfig, PushKernelArg};

const CASES: &[(&str, usize)] = &[
    ("c1_t64", 64),
    ("c2_t128", 128),
    ("c3_t96", 96),
    ("c4_t64_s0", 64),
];

const NK: usize = 16;
const NV: usize = 48;
const KD: usize = 128;
const VD: usize = 128;
const BT: usize = 64;
const SCALE: f32 = 0.08838834764831845; // 1/sqrt(128)
const RCP_LN2: f32 = 1.4426950408889634;

struct Case {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    g: Vec<f32>,
    beta: Vec<f32>,
    s0: Option<Vec<f32>>,
    g_cum: Vec<f32>,
    a: Vec<f32>,
    w: Vec<f32>,
    u: Vec<f32>,
    h: Vec<f32>,
    v_new: Vec<f32>,
    ht: Vec<f32>,
    o: Vec<f32>,
    t: usize,
}

fn read_bin(dir: &str, name: &str) -> Vec<f32> {
    let p = format!("{dir}/{name}.bin");
    let bytes = std::fs::read(&p).unwrap_or_else(|e| panic!("{p}: {e}"));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn load_case(dir: &str, tag: &str, t: usize) -> Case {
    let rd = |n: &str| read_bin(dir, &format!("{tag}.{n}"));
    let has_s0 = std::fs::metadata(format!("{dir}/{tag}.s0.bin")).is_ok();
    Case {
        q: rd("q"),
        k: rd("k"),
        v: rd("v"),
        g: rd("g"),
        beta: rd("beta"),
        s0: has_s0.then(|| rd("s0")),
        g_cum: rd("g_cum"),
        a: rd("A"),
        w: rd("w"),
        u: rd("u"),
        h: rd("h"),
        v_new: rd("v_new"),
        ht: rd("final"),
        o: rd("o"),
        t,
    }
}

fn max_dev(a: &[f32], b: &[f32]) -> (usize, f32) {
    let mut worst = 0f32;
    let mut idx = 0;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let d = (x - y).abs();
        if d > worst {
            worst = d;
            idx = i;
        }
    }
    (idx, worst)
}

#[test]
#[allow(clippy::too_many_lines)]
fn gdn_chunked_golden_stage_by_stage() {
    if std::env::var_os("OWL_TEST_DEVICE").is_none() {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    }
    let ctx = CudaContext::new(1).expect("ctx");
    ctx.bind_to_thread().expect("bind");
    let stream = ctx.default_stream();

    let assets = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/gdn_chunked");
    let load_fn = |cubin: &[u8], fname: &str| -> std::sync::Arc<cudarc::driver::CudaFunction> {
        let m = ctx
            .load_module(cudarc::nvrtc::Ptx::from_binary(cubin.to_vec()))
            .unwrap_or_else(|e| panic!("load {fname}: {e:?}"));
        std::sync::Arc::new(
            m.load_function(fname)
                .unwrap_or_else(|e| panic!("fn {fname}: {e:?}")),
        )
    };
    use owl_kernels::gdn_chunked::cubins;
    use owl_kernels::gdn_chunked::cubins::launch as LC;
    let f_cum = load_fn(cubins::CUMSUM, "chunk_local_cumsum_scalar_kernel");
    let f_kkt = load_fn(cubins::KKT, "chunk_gated_delta_rule_fwd_kkt_solve_kernel");
    let f_wu = load_fn(cubins::WU, "recompute_w_u_fwd_kernel");
    let f_h = load_fn(cubins::H, "chunk_gated_delta_rule_fwd_kernel_h_blockdim64");
    let f_h_noh0 = load_fn(cubins::H_NOH0, "chunk_gated_delta_rule_fwd_kernel_h_blockdim64");
    let f_o = load_fn(cubins::O, "chunk_fwd_kernel_o");
    for (f, sz) in [(&f_wu, LC::WU_SHARED as i32), (&f_o, LC::O_SHARED as i32)] {
        use cudarc::driver::sys::CUfunction_attribute_enum as Attr;
        f.set_attribute(Attr::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, sz)
            .expect("setattr");
    }

    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/gdn_fla/cases");

    for (tag, t) in CASES {
        let case = load_case(dir, tag, *t);
        assert_eq!(case.q.len(), t * NK * KD, "{tag}: q 形状");
        let nt = t.div_ceil(BT);

        let mut q = stream.alloc_zeros::<f32>(case.q.len()).unwrap();
        stream.memcpy_htod(&case.q, &mut q).unwrap();
        let mut k = stream.alloc_zeros::<f32>(case.k.len()).unwrap();
        stream.memcpy_htod(&case.k, &mut k).unwrap();
        let mut v = stream.alloc_zeros::<f32>(case.v.len()).unwrap();
        stream.memcpy_htod(&case.v, &mut v).unwrap();
        let mut g = stream.alloc_zeros::<f32>(case.g.len()).unwrap();
        stream.memcpy_htod(&case.g, &mut g).unwrap();
        let mut beta = stream.alloc_zeros::<f32>(case.beta.len()).unwrap();
        stream.memcpy_htod(&case.beta, &mut beta).unwrap();
        let s0 = case.s0.as_ref().map(|s| {
            let mut d = stream.alloc_zeros::<f32>(s.len()).unwrap();
            stream.memcpy_htod(s, &mut d).unwrap();
            d
        });
        let g_cum = stream.alloc_zeros::<f32>(t * NV).unwrap();
        let a = stream.alloc_zeros::<f32>(t * NV * BT).unwrap();
        let w = stream.alloc_zeros::<f32>(t * NV * KD).unwrap();
        let u = stream.alloc_zeros::<f32>(t * NV * KD).unwrap();
        let h = stream.alloc_zeros::<f32>(nt * NV * KD * VD).unwrap();
        let v_new = stream.alloc_zeros::<f32>(t * NV * VD).unwrap();
        let ht = stream.alloc_zeros::<f32>(NV * KD * VD).unwrap();
        let idx_h: Vec<i64> = (0..nt).flat_map(|i| [0i64, i as i64]).collect();
        let mut idx = stream.alloc_zeros::<i64>(idx_h.len()).unwrap();
        stream.memcpy_htod(&idx_h, &mut idx).unwrap();
        let mut cu = stream.alloc_zeros::<i64>(2).unwrap();
        stream.memcpy_htod(&[0i64, *t as i64], &mut cu).unwrap();
        let mut coff = stream.alloc_zeros::<i64>(2).unwrap();
        stream.memcpy_htod(&[0i64, nt as i64], &mut coff).unwrap();
        let o = stream.alloc_zeros::<f32>(t * NV * VD).unwrap();

        let q_p = q.device_ptr(&stream).0;
        let k_p = k.device_ptr(&stream).0;
        let v_p = v.device_ptr(&stream).0;
        let g_p = g.device_ptr(&stream).0;
        let beta_p = beta.device_ptr(&stream).0;
        let s0_p = s0.as_ref().map(|s| s.device_ptr(&stream).0).unwrap_or(0);
        let gcum_p = g_cum.device_ptr(&stream).0;
        let a_p = a.device_ptr(&stream).0;
        let w_p = w.device_ptr(&stream).0;
        let u_p = u.device_ptr(&stream).0;
        let h_p = h.device_ptr(&stream).0;
        let vnew_p = v_new.device_ptr(&stream).0;
        let ht_p = ht.device_ptr(&stream).0;
        let idx_p = idx.device_ptr(&stream).0;
        let cu_p = cu.device_ptr(&stream).0;
        let coff_p = coff.device_ptr(&stream).0;
        let o = stream.alloc_zeros::<f32>(t * NV * VD).unwrap();
        let o_p = o.device_ptr(&stream).0;
        let ti = *t as i32;
        let scratch: u64 = 0;

        // 1. cumsum:grid (NT, HV);尾部双隐式 scratch 参(global+profile)
        {
            let mut b = stream.launch_builder(&f_cum);
            b.arg(&g_p).arg(&gcum_p).arg(&RCP_LN2).arg(&cu_p).arg(&idx_p).arg(&ti)
                .arg(&scratch).arg(&scratch);
            let cfg = LaunchConfig {
                grid_dim: (nt as u32, NV as u32, 1),
                block_dim: (32 * LC::CUMSUM_WARPS, 1, 1),
                shared_mem_bytes: LC::CUMSUM_SHARED,
            };
            unsafe { b.launch(cfg) }.expect("cumsum");
        }
        // 2. kkt:grid (NT, HV)
        {
            let mut b = stream.launch_builder(&f_kkt);
            b.arg(&k_p).arg(&gcum_p).arg(&beta_p).arg(&a_p).arg(&cu_p).arg(&idx_p).arg(&ti)
                .arg(&scratch).arg(&scratch);
            let cfg = LaunchConfig {
                grid_dim: (nt as u32, NV as u32, 1),
                block_dim: (32 * LC::KKT_WARPS, 1, 1),
                shared_mem_bytes: LC::KKT_SHARED,
            };
            unsafe { b.launch(cfg) }.expect("kkt");
        }
        // 3. wu:grid (NT, HV)
        {
            let mut b = stream.launch_builder(&f_wu);
            b.arg(&k_p).arg(&v_p).arg(&beta_p).arg(&w_p).arg(&u_p).arg(&a_p)
                .arg(&gcum_p).arg(&cu_p).arg(&idx_p).arg(&ti)
                .arg(&scratch).arg(&scratch);
            let cfg = LaunchConfig {
                grid_dim: (nt as u32, NV as u32, 1),
                block_dim: (32 * LC::WU_WARPS, 1, 1),
                shared_mem_bytes: LC::WU_SHARED,
            };
            unsafe { b.launch(cfg) }.expect("wu");
        }
        // 4. h:grid (cdiv(VD,BV) × NV),BV=64(h0 变体;无 s0 传 NULL —— 内核
        //    对 h0==nullptr 的行为由 FLA 语义保证:fresh 序列走零初态分支)
        {
            let mut b = stream.launch_builder(&f_h);
            b.arg(&k_p)
                .arg(&v_p)
                .arg(&w_p)
                .arg(&vnew_p)
                .arg(&gcum_p)
                .arg(&h_p)
                .arg(&s0_p)
                .arg(&ht_p)
                .arg(&cu_p)
                .arg(&coff_p)
                .arg(&ti)
                .arg(&scratch)
                .arg(&scratch);
            let cfg = LaunchConfig {
                grid_dim: ((VD as u32).div_ceil(64) * NV as u32, 1, 1),
                block_dim: (32 * LC::H_WARPS, 1, 1),
                shared_mem_bytes: LC::H_SHARED,
            };
            unsafe { b.launch(cfg) }.expect("h");
        }
        // 5. o:grid (cdiv(VD,BV), NT, NV),BV=64
        {
            let mut b = stream.launch_builder(&f_o);
            b.arg(&q_p)
                .arg(&k_p)
                .arg(&vnew_p)
                .arg(&h_p)
                .arg(&gcum_p)
                .arg(&o_p)
                .arg(&cu_p)
                .arg(&idx_p)
                .arg(&SCALE)
                .arg(&ti)
                .arg(&scratch)
                .arg(&scratch);
            let cfg = LaunchConfig {
                grid_dim: ((VD as u32).div_ceil(64), nt as u32, NV as u32),
                block_dim: (32 * LC::O_WARPS, 1, 1),
                shared_mem_bytes: LC::O_SHARED,
            };
            unsafe { b.launch(cfg) }.expect("o");
        }
        stream.synchronize().expect("sync");

        // 逐阶段对拍
        let dtoh = |s: &cudarc::driver::CudaSlice<f32>| -> Vec<f32> { stream.memcpy_dtov(s).expect("dtoh") };
        let (gi, dg) = max_dev(&dtoh(&g_cum), &case.g_cum);
        let (ai, da) = max_dev(&dtoh(&a), &case.a);
        let (wi, dw) = max_dev(&dtoh(&w), &case.w);
        let (ui, du) = max_dev(&dtoh(&u), &case.u);
        let (hi, dh) = max_dev(&dtoh(&h), &case.h);
        let (vni, dv) = max_dev(&dtoh(&v_new), &case.v_new);
        let oi = dtoh(&o);
        let (oi_i, doo) = max_dev(&oi, &case.o);
        let ht_host = dtoh(&ht);
        let (hti, dht) = max_dev(&ht_host, &case.ht);
        eprintln!(
            "[{tag}] g_cum={dg:.2e}@{gi} A={da:.2e}@{ai} w={dw:.2e}@{wi} u={du:.2e}@{ui} \
             h={dh:.2e}@{hi} v_new={dv:.2e}@{vni} o={doo:.2e}@{oi_i} ht={dht:.2e}@{hti}"
        );
        assert!(dg < 1e-3, "{tag}: g_cum 偏差 {dg}");
        assert!(da < 5e-2, "{tag}: A 偏差 {da}");
        assert!(dw < 5e-2, "{tag}: w 偏差 {dw}");
        assert!(du < 5e-2, "{tag}: u 偏差 {du}");
        assert!(dh < 5e-2, "{tag}: h 偏差 {dh}");
        assert!(dv < 5e-2, "{tag}: v_new 偏差 {dv}");
        assert!(doo < 5e-2, "{tag}: o 偏差 {doo}");
        assert!(dht < 5e-2, "{tag}: ht 偏差 {dht}");
    }
}

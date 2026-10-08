//! GDN chunked delta rule —— FLA fork AOT cubin 算子层金标对拍(M1 修活,2026-10-11)。
//!
//! **配方 = 引擎生产链镜像**(foreign.rs handle_gdn_chunked 逐一对应):
//! q/k/v/beta 铸 bf16 + g 恒 f32(先 f16 量化锁引擎输入)+ state/A f32、
//! Ai/w/u/h/v_new/o = bf16 字节;发射序 cumsum → kkt → merge → wu → h → o。
//!
//! **⚠️ g 语义律(2026-10-11 终案,补录Ⅸ)**:kkt/h/o 的 "g" 参数全部是
//! **cumsum 产物** —— 本测试喂 gcum;喂 raw g 会在 c5 深谷(−24)上爆炸,
//! 在 c1-c4 温和域(≥−0.5)则 A/w/o 静默偏离 —— 两级金标正好双拦截。
//!
//! 金标两级:
//! - c1-c4(2026-10-03,pip-fla 0.6.0 f32):**宽松容差** = 盖 bf16 输入
//!   量化 × f16 g 量化(金标是 f32 链,引擎是 bf16 链);
//! - c5_t128_deep(2026-10-11,gen_c5_deep.py,**fork 阶段函数权威**):
//!   **紧容差**(同源同核,理论逐位;g 域 [−23.98, −0.01] 覆盖引擎深谷)
//!   —— 输入域极值机器门(补录Ⅷ教训#2),raw-g 回归在此必爆。
//!
//! 运行纪律:OWL_TEST_DEVICE 钉卡 + `--test-threads=1`(pitfall §16);
//! 校准模式 OWL_GOLDEN_CALIBRATE=1 只打印偏差不断言。

use cudarc::driver::{CudaContext, DevicePtr, LaunchConfig, PushKernelArg};

const CASES: &[(&str, usize, bool)] = &[
    // (tag, T, tight) tight = fork 同源金标(紧容差);false = pip-fla f32 金标(宽)
    ("c1_t64", 64, false),
    ("c2_t128", 128, false),
    ("c3_t96", 96, false),
    ("c4_t64_s0", 64, false),
    ("c5_t128_deep", 128, true),
];

const NK: usize = 16;
const NV: usize = 48;
const KD: usize = 128;
const VD: usize = 128;
const BT: usize = 64;
const SCALE: f32 = 0.08838834764831845; // 1/sqrt(128)
const RCP_LN2: f32 = 1.4426950408889634;

/// **g_cum 域律**(M1 首跑定谳):fork cumsum 核输出 **nats 域**(RCP_LN2 烘
/// 核内不施放);pip-fla 金标(c1-c4)= **log2 域**(gen.py 显式 scale=
/// RCP_LN2)。下游 exp(Δ) 两域同值,故仅 g_cum 对拍需换算:want_nats =
/// golden_log2 / RCP_LN2。c5(fork 权威)同为 nats,直比。

/// **布局律(M1 首跑定谳)**:fork h 核的 h_buf/ht/ht0 排布 = [.., HV, V, K]
/// (V 主;wrapper new_empty(B,NT,H,V,K) 同款;handler 的
/// owl_state_kv_to_vk/vk_to_kv 与池 [HV,K,V] 互转即此)。
/// 两代金标约定不同 → 比较方式分层:
/// - c1-c4(pip-fla 0.6.0 golden,[.., K,V] FLA 习惯):**需转置**后比;
/// - c5(fork 权威 golden,[.., V,K]):**直比**。
/// (c5 首跑曾现"chunk0 需转置 chunk1 不需"奇象 = gen 喂 s0 未转置 +
///  深谷 chunk 边界 exp 下溢精确抹零 s0 贡献的合谋,gen 已修为 [V,K] 喂入。)
fn needs_transpose(tight: bool) -> bool { !tight }

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
}

fn read_bin(dir: &str, name: &str) -> Vec<f32> {
    let p = format!("{dir}/{name}.bin");
    let bytes = owl_shared::file_loader::read(&p).unwrap_or_else(|e| panic!("{p}: {e}"));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn load_case(dir: &str, tag: &str, t: usize) -> Case {
    let rd = |n: &str| read_bin(dir, &format!("{tag}.{n}"));
    let has_s0 = owl_shared::file_loader::metadata(format!("{dir}/{tag}.s0.bin")).is_ok();
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
    }
}

// ── host 位型转换(与引擎 cast 核同精度:f32→bf16/f16 = RNE;展宽精确)──

fn f32_to_bf16(src: &[f32]) -> Vec<u8> {
    src.iter()
        .flat_map(|f| half::bf16::from_f32(*f).to_le_bytes())
        .collect()
}

fn bf16_to_f32(src: &[u8]) -> Vec<f32> {
    src.chunks_exact(2)
        .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect()
}

/// 引擎 g 路径:f16 块 → cast f32(量化有损,金标容差须覆盖)
fn g_engine_view(src: &[f32]) -> Vec<f32> {
    src.iter()
        .map(|f| half::f16::from_f32(*f).to_f32())
        .collect()
}

/// [HV, X, Y] ↔ [HV, Y, X](state 的 fork↔FLA 布局互转)
fn transpose_kv(src: &[f32], hv: usize, x: usize, y: usize) -> Vec<f32> {
    let mut out = vec![0f32; src.len()];
    for hi in 0..hv {
        for xi in 0..x {
            for yi in 0..y {
                out[hi * x * y + yi * x + xi] = src[hi * x * y + xi * y + yi];
            }
        }
    }
    out
}

/// [NT, HV, X, Y] → [NT, HV, Y, X](h_buf 逐 chunk 转置;M1 首跑勘误:
/// 三维版喂四维缓冲时 NT>1 的后半块不落笔 = 全零假象)
fn transpose_kv_nt(src: &[f32], nt: usize, hv: usize, x: usize, y: usize) -> Vec<f32> {
    let mut out = vec![0f32; src.len()];
    for ti in 0..nt {
        let base = ti * hv * x * y;
        let (s, o) = src[base..base + hv * x * y].split_at(0);
        let _ = (s, o);
        for hi in 0..hv {
            for xi in 0..x {
                for yi in 0..y {
                    out[base + hi * x * y + yi * x + xi] =
                        src[base + hi * x * y + xi * y + yi];
                }
            }
        }
    }
    out
}

/// 组合容差断言:|got − want| ≤ atol + rtol·|want|;返回最大归一化偏差
fn assert_close(name: &str, tag: &str, got: &[f32], want: &[f32], atol: f32, rtol: f32) {
    assert_eq!(got.len(), want.len(), "{tag}/{name}: 长度失配");
    let (mut worst, mut wi, mut wg, mut ww) = (0f32, 0usize, 0f32, 0f32);
    for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
        let d = (a - b).abs();
        let n = d / (atol + rtol * b.abs());
        if n > worst {
            worst = n;
            wi = i;
            wg = *a;
            ww = *b;
        }
    }
    eprintln!("  [{tag}] {name}: max|Δ|={:.3e} @{} (got={wg:.4e} want={ww:.4e}) 预算内 ×{worst:.2}", wg - ww, wi);
    assert!(worst <= 1.0, "{tag}/{name}: 超容差 ×{worst:.2} @ {wi}({wg} vs {ww})");
}

#[test]
#[allow(clippy::too_many_lines)]
fn gdn_chunked_golden_stage_by_stage() {
    if !owl_shared::env_reader::flag("OWL_TEST_DEVICE") {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    }
    let dev: usize = owl_shared::env_reader::parse("OWL_TEST_DEVICE").unwrap_or(1);
    let ctx = CudaContext::new(dev).expect("cuda ctx");
    ctx.bind_to_thread().expect("bind");
    let stream = ctx.default_stream();

    let load_fn = |cubin: &[u8], fname: &str| -> std::sync::Arc<cudarc::driver::CudaFunction> {
        let m = ctx
            .load_module(cudarc::nvrtc::Ptx::from_binary(cubin.to_vec()))
            .unwrap_or_else(|e| panic!("load {fname}: {e:?}"));
        std::sync::Arc::new(
            m.load_function(fname)
                .unwrap_or_else(|e| panic!("fn {fname}: {e:?}")),
        )
    };
    use owl_kernels::family::gdn_chunked::cubins;
    use owl_kernels::family::gdn_chunked::cubins::launch as LC;
    // ⚠️ 符号名 = fork 六核(2026-10-11 资产换血后);pip-fla 旧名
    // (chunk_gated_delta_rule_fwd_kkt_solve_kernel)已随资产退役。
    let f_cum = load_fn(cubins::CUMSUM, "chunk_local_cumsum_scalar_kernel");
    let f_kkt = load_fn(cubins::KKT, "chunk_scaled_dot_kkt_fwd_kernel");
    let f_merge = load_fn(cubins::MERGE, "merge_16x16_to_64x64_inverse_kernel");
    let f_wu = load_fn(cubins::WU, "recompute_w_u_fwd_kernel");
    let f_h = load_fn(cubins::H, "chunk_gated_delta_rule_fwd_kernel_h_blockdim64");
    let f_o = load_fn(cubins::O, "chunk_fwd_kernel_o");
    // shared opt-in(与 handler 同名单:wu/h/o;fork 选中变体 h=49412 超 48K 顶)
    for (f, sz) in [
        (&f_wu, LC::WU_SHARED as i32),
        (&f_h, LC::H_SHARED as i32),
        (&f_o, LC::O_SHARED as i32),
    ] {
        use cudarc::driver::sys::CUfunction_attribute_enum as Attr;
        f.set_attribute(Attr::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, sz)
            .expect("setattr");
    }

    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/gdn_fla/cases");

    for (tag, t, tight) in CASES {
        let case = load_case(dir, tag, *t);
        assert_eq!(case.q.len(), t * NK * KD, "{tag}: q 形状");
        let nt = t.div_ceil(BT);

        // ── host 铸造(引擎配方)──
        let q_b = f32_to_bf16(&case.q);
        let k_b = f32_to_bf16(&case.k);
        let v_b = f32_to_bf16(&case.v);
        let beta_b = f32_to_bf16(&case.beta);
        let g_e = g_engine_view(&case.g); // f16 量化(引擎 g 路径)
        // state:池排布 [HV,KD,VD] → fork [HV,VD,KD](FORK_LAYOUT_VK);零初态 = 零块
        // s0 上传:恒为核期望 [V,K](引擎 kv_to_vk 同款;与金标代际无关)
        let state_in = match &case.s0 {
            Some(s) => transpose_kv(s, NV, KD, VD),
            None => vec![0f32; NV * KD * VD],
        };

        // ── 设备缓冲(dtype = 生产 realloc 同款)──
        let up_f32 = |v: &[f32]| {
            let mut d = stream.alloc_zeros::<f32>(v.len()).unwrap();
            stream.memcpy_htod(v, &mut d).unwrap();
            d
        };
        let up_u8 = |v: &[u8]| {
            let mut d = unsafe { stream.alloc_zeros::<u8>(v.len()) }.unwrap();
            stream.memcpy_htod(v, &mut d).unwrap();
            d
        };
        let qd = up_u8(&q_b);
        let kd_ = up_u8(&k_b);
        let vd_ = up_u8(&v_b);
        let betad = up_u8(&beta_b);
        let gd = up_f32(&g_e);
        let stated = up_f32(&state_in);
        let gcum = stream.alloc_zeros::<f32>(t * NV).unwrap();
        let a = stream.alloc_zeros::<f32>(t * NV * BT).unwrap();
        let ai = unsafe { stream.alloc_zeros::<u8>(t * NV * BT * 2) }.unwrap(); // bf16 字节
let w = unsafe { stream.alloc_zeros::<u8>(t * NV * KD * 2) }.unwrap();
        let u = unsafe { stream.alloc_zeros::<u8>(t * NV * VD * 2) }.unwrap();
        let h = unsafe { stream.alloc_zeros::<u8>(nt * NV * VD * KD * 2) }.unwrap();
        let v_new = unsafe { stream.alloc_zeros::<u8>(t * NV * VD * 2) }.unwrap();
        let state_out = stream.alloc_zeros::<f32>(NV * KD * VD).unwrap();
        let o = unsafe { stream.alloc_zeros::<u8>(t * NV * VD * 2) }.unwrap();
        // meta 表(handler 同款):[cu(2) | coff(2) | idx(2NT)]
        let mut meta_h: Vec<i64> = Vec::with_capacity(4 + nt * 2);
        meta_h.extend_from_slice(&[0i64, *t as i64]);
        meta_h.extend_from_slice(&[0i64, nt as i64]);
        meta_h.extend((0..nt).flat_map(|i| [0i64, i as i64]));
        let metad = {
            let mut d = unsafe { stream.alloc_zeros::<u8>(meta_h.len() * 8) }.unwrap();
            stream.memcpy_htod(bytemuck_meta(&meta_h), &mut d).unwrap();
            d
        };
        let (cu_p, coff_p, idx_p) = {
            let base = metad.device_ptr(&stream).0;
            (base, base + 16, base + 32)
        };

        let q_p = qd.device_ptr(&stream).0;
        let k_p = kd_.device_ptr(&stream).0;
        let v_p = vd_.device_ptr(&stream).0;
        let beta_p = betad.device_ptr(&stream).0;
        let g_p = gd.device_ptr(&stream).0;
        let state_p = stated.device_ptr(&stream).0;
        let gcum_p = gcum.device_ptr(&stream).0;
        let a_p = a.device_ptr(&stream).0;
        let ai_p = ai.device_ptr(&stream).0;
        let w_p = w.device_ptr(&stream).0;
        let u_p = u.device_ptr(&stream).0;
        let h_p = h.device_ptr(&stream).0;
        let vnew_p = v_new.device_ptr(&stream).0;
        let so_p = state_out.device_ptr(&stream).0;
        let o_p = o.device_ptr(&stream).0;
        let ti = *t as i32;
        let scratch: u64 = 0;

        // ── 1. cumsum:raw g(f32)→ gcum;⚠️ fork ABI 无 scale 参(ln2 已烘核内)
        {
            let mut b = stream.launch_builder(&f_cum);
            b.arg(&g_p).arg(&gcum_p).arg(&cu_p).arg(&idx_p).arg(&ti)
                .arg(&scratch).arg(&scratch);
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: (nt as u32, NV as u32, 1),
                    block_dim: (32 * LC::CUMSUM_WARPS, 1, 1),
                    shared_mem_bytes: LC::CUMSUM_SHARED,
                })
            }.expect("cumsum");
        }
        // ── 2. kkt:k/beta/gcum → A(f32)【g 语义律:吃 cumsum 产物】
        {
            let mut b = stream.launch_builder(&f_kkt);
            b.arg(&k_p).arg(&beta_p).arg(&gcum_p).arg(&a_p)
                .arg(&cu_p).arg(&idx_p).arg(&ti)
                .arg(&scratch).arg(&scratch);
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: (nt as u32, NV as u32, 1),
                    block_dim: (32 * LC::KKT_WARPS, 1, 1),
                    shared_mem_bytes: LC::KKT_SHARED,
                })
            }.expect("kkt");
        }
        // ── 3. merge:solve_tril(BT=64)A → Ai(bf16 字节;生产链同款)
        {
            let mut b = stream.launch_builder(&f_merge);
            b.arg(&a_p).arg(&ai_p).arg(&cu_p).arg(&idx_p).arg(&ti)
                .arg(&scratch).arg(&scratch);
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: (nt as u32, NV as u32, 1),
                    block_dim: (32 * LC::MERGE_WARPS, 1, 1),
                    shared_mem_bytes: LC::MERGE_SHARED,
                })
            }.expect("merge");
        }
        // ── 4. wu:k/v/beta/Ai/gcum → w/u(bf16)
        {
            let mut b = stream.launch_builder(&f_wu);
            b.arg(&k_p).arg(&v_p).arg(&beta_p).arg(&w_p).arg(&u_p).arg(&ai_p)
                .arg(&gcum_p).arg(&cu_p).arg(&idx_p).arg(&ti)
                .arg(&scratch).arg(&scratch);
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: (nt as u32, NV as u32, 1),
                    block_dim: (32 * LC::WU_WARPS, 1, 1),
                    shared_mem_bytes: LC::WU_SHARED,
                })
            }.expect("wu");
        }
        // ── 5. h:k/u/w/gcum/state → h_buf/v_new/ht(SASS 审计序:gk 被剪参;
        //        grid 二维 = (cdiv(VD,BV), HV),非 1D 乘积 —— 补录Ⅴ 校正)
        {
            let mut b = stream.launch_builder(&f_h);
            b.arg(&k_p).arg(&u_p).arg(&w_p).arg(&vnew_p).arg(&gcum_p)
                .arg(&h_p).arg(&state_p).arg(&so_p)
                .arg(&cu_p).arg(&coff_p).arg(&ti)
                .arg(&scratch).arg(&scratch);
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: ((VD as u32).div_ceil(LC::H_BV), NV as u32, 1),
                    block_dim: (32 * LC::H_WARPS, 1, 1),
                    shared_mem_bytes: LC::H_SHARED,
                })
            }.expect("h");
        }
        // ── 6. o:q/k/v_new/h/gcum → o(bf16)
        {
            let mut b = stream.launch_builder(&f_o);
            b.arg(&q_p).arg(&k_p).arg(&vnew_p).arg(&h_p).arg(&gcum_p)
                .arg(&o_p).arg(&cu_p).arg(&idx_p).arg(&SCALE).arg(&ti)
                .arg(&scratch).arg(&scratch);
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: ((VD as u32).div_ceil(LC::O_BV), nt as u32, NV as u32),
                    block_dim: (32 * LC::O_WARPS, 1, 1),
                    shared_mem_bytes: LC::O_SHARED,
                })
            }.expect("o");
        }
        stream.synchronize().expect("sync");

        // ── 逐阶段对拍 ──
        let dt = |s: &cudarc::driver::CudaSlice<u8>| -> Vec<f32> {
            bf16_to_f32(&stream.memcpy_dtov(s).expect("dtoh"))
        };
        // 容差分级:tight = fork 同源金标(理论逐位,留 JIT/AOT 变体余量);
        // 宽 = pip-fla f32 金标 × bf16 链(输入量化 × 64 步递推放大)
        let (tol_a, tol_r) = if *tight { (1e-4f32, 1e-3f32) } else { (6e-2f32, 6e-2f32) };
        eprintln!("[{tag}] {}金标 对拍:", if *tight { "紧(深谷) " } else { "宽(f32) " });
        let g_cum_want: Vec<f32> = if *tight {
            case.g_cum.clone() // c5 = fork 权威,nats 域直比
        } else {
            case.g_cum.iter().map(|v| v / RCP_LN2).collect() // log2 → nats
        };
        assert_close("g_cum(nats)", tag, &stream.memcpy_dtov(&gcum).unwrap(), &g_cum_want,
            if *tight { 1e-3 } else { 2.5e-2 }, tol_r);
        assert_close("A(solve后)", tag, &dt(&ai), &case.a, tol_a, tol_r);
        assert_close("w", tag, &dt(&w), &case.w, tol_a, tol_r);
        assert_close("u", tag, &dt(&u), &case.u, tol_a, tol_r);
        let h_host = dt(&h);
        let h_cmp = if needs_transpose(*tight) { transpose_kv_nt(&h_host, nt, NV, KD, VD) } else { h_host };
        assert_close("h", tag, &h_cmp, &case.h, tol_a, tol_r);
        assert_close("v_new", tag, &dt(&v_new), &case.v_new, tol_a, tol_r);
        assert_close("o", tag, &dt(&o), &case.o, if *tight { 1e-4 } else { 5e-2 }, tol_r);
        let ht_host = stream.memcpy_dtov(&state_out).unwrap();
        let ht_cmp = if needs_transpose(*tight) { transpose_kv(&ht_host, NV, VD, KD) } else { ht_host };
        assert_close("ht", tag, &ht_cmp, &case.ht, tol_a, tol_r);
            }
}


/// i64 表按字节上传的便桥(meta 表)
fn bytemuck_meta(v: &[i64]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), v.len() * 8) }
}

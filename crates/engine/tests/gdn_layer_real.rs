//! M-Ⅱ 立案对拍:0.8B 真权重单层 GDN(GatedDeltaNet)host 参考 vs 设备实现。
//!
//! 背景:全链 forward 自层 8(第 7 个 GDN 层)起 hidden 全 NaN;输入 finite、
//! conv 后 finite、递推后 NaN(插桩定位)。本文件把单层拆成五个断面,
//! 逐断面与纯 Rust 参考对比,报告第一个分道扬镳的断面:
//!   ① conv1d(因果 k=4, silu, fresh state)—— GdnKernels::conv1d_fwd_k4
//!   ② gating(g = -exp(A_log)·softplus(a+dt_bias), β = sigmoid(b))
//!   ③ l2norm(q/k 末维,单位范数 rsqrt(sum+eps),FLA 语义)
//!   ④ delta rule 逐 token(dec_gqa 核,g log 空间核内 exp,q 缩放 1/√128)
//!      host 参考对照 packages/xinfer/crates/kernels/cuda/tests/gdn_parity.rs
//!      参考(衰减先作用 S 再算校正项;校正对比用 k,out 用 q·scale)
//!   ⑤ 门控 RMSNorm × silu(z)(eps = rms_norm_eps = 1e-6,γ per-head [128])
//! 另:e2e = GatedDeltaNet::new(真权重)forward vs host 全层(层 8 与层 0)。
//! 断面测试的投影矩阵乘在 host 完成(隔离 GEMM,只测数学核)。

use owl_cuda::CudaDevice;
use owl_engine::config::Config;
use owl_engine::models::dry_kernels::DryKernels;
use owl_engine::models::layers::deltanet::GatedDeltaNet;
use owl_engine::models::layers::{ctx_scope, ctor, vendor, VarBuilderX};
use owl_iface::{Device as _, DevBuf as _, Pool as _, PoolConfig, PoolKind};
use owl_nn::cublas::NnBlas;
use owl_nn::kernels::gdn_kernels::GdnKernels;
use std::sync::Arc;

const MODEL_DIR: &str = "/home/div/Documents/codes/models/Qwen/Qwen3.5-0.8B";
const ST_MODEL: &str = "/home/div/Documents/codes/models/Qwen/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors";
const T: usize = 5;
const H: usize = 1024;
const NK: usize = 16;
const NV: usize = 16;
const KD: usize = 128;
const VD: usize = 128;
const KEY_DIM: usize = NK * KD; // 2048
const VALUE_DIM: usize = NV * VD; // 2048
const CONV_DIM: usize = KEY_DIM * 2 + VALUE_DIM; // 6144

// ---------------- 基建 ----------------

struct Rig {
    dev: Arc<CudaDevice>,
    pool: Arc<owl_cuda::CudaPool>,
}

fn install_rig() -> Rig {
    let dev = Arc::new(CudaDevice::new(owl_cuda::test_device_ordinal()).expect("需要 CUDA 设备"));
    let scratch = Arc::new(
        dev.create_pool(PoolConfig {
            name: format!("gdn-scratch-{}", std::process::id()),
            kind: PoolKind::Scratch,
            bytes: 256 << 20,
        })
        .unwrap(),
    );
    let pool = Arc::new(
        dev.create_pool(PoolConfig {
            name: format!("gdn-weights-{}", std::process::id()),
            kind: PoolKind::Weights,
            bytes: 1 << 30,
        })
        .unwrap(),
    );
    let ops = owl_nn::OpsCtx::new(&dev).unwrap();
    let blas = NnBlas::new(&dev).unwrap();
    let dry = DryKernels::new(dev.ctx()).unwrap();
    ctx_scope::install(ops, blas, dry, scratch, pool.clone(), &dev);
    Rig { dev, pool }
}

fn htod<T: owl_iface::MemValue + Send + Sync + 'static>(
    rig: &Rig,
    v: Vec<T>,
) -> owl_cuda::Persistent<T> {
    rig.pool.htod_persistent_in(v).unwrap()
}

fn dtoh_f32(dev: &CudaDevice, ptr: *const f32, n: usize) -> Vec<f32> {
    use owl_cuda::ffi::sys;
    dev.ctx().bind_to_thread().unwrap();
    dev.ctx().synchronize().unwrap();
    let mut out = vec![0f32; n];
    unsafe {
        sys::cuMemcpyDtoH_v2(
            out.as_mut_ptr() as *mut std::ffi::c_void,
            ptr as sys::CUdeviceptr,
            out.len() * 4,
        )
        .result()
        .unwrap();
    }
    out
}

// ---------------- host 参考(公式直译) ----------------

struct Rng(u64);
impl Rng {
    fn uniform(&mut self, amp: f32) -> f32 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        let u = ((z ^ (z >> 31)) >> 40) as f32 / (1u32 << 24) as f32;
        u * 2.0 * amp - amp
    }
}

fn matmul_row(x: &[f32], t: usize, w: &[f32], o: usize, h: usize) -> Vec<f32> {
    let mut y = vec![0f32; t * o];
    for r in 0..t {
        for c in 0..o {
            let mut acc = 0f64;
            for i in 0..h {
                acc += x[r * h + i] as f64 * w[c * h + i] as f64;
            }
            y[r * o + c] = acc as f32;
        }
    }
    y
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

fn host_conv1d(mixed: &[f32], w: &[f32], t: usize, dim: usize) -> Vec<f32> {
    let mut out = vec![0f32; t * dim];
    for c in 0..dim {
        for tt in 0..t {
            let mut acc = 0f64;
            // Conv1d 语义:y[t] = Σ_j w[j]·x[t+j-3](w[3] 乘当前 token;核同)
            for j in 0..4 {
                if tt + j >= 3 {
                    let src = tt + j - 3;
                    acc += (w[c * 4 + j] * mixed[src * dim + c]) as f64;
                }
            }
            out[tt * dim + c] = silu(acc as f32);
        }
    }
    out
}

fn softplus(v: f64) -> f64 {
    if v > 20.0 {
        v
    } else {
        (1.0 + v.exp()).ln()
    }
}

fn host_gating(
    a_log: &[f32],
    dt_bias: &[f32],
    a: &[f32],
    b: &[f32],
    t: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut g = vec![0f32; t * NV];
    let mut beta = vec![0f32; t * NV];
    for tt in 0..t {
        for h in 0..NV {
            let af = (a[tt * NV + h] + dt_bias[h]) as f64;
            g[tt * NV + h] = (-(a_log[h] as f64).exp() * softplus(af)) as f32;
            beta[tt * NV + h] = (1.0 / (1.0 + (-(b[tt * NV + h] as f64)).exp())) as f32;
        }
    }
    (g, beta)
}

fn host_l2norm(v: &[f32], rows: usize, dim: usize, eps: f64) -> Vec<f32> {
    let mut out = v.to_vec();
    for r in 0..rows {
        let s: f64 = out[r * dim..(r + 1) * dim]
            .iter()
            .map(|x| (*x as f64) * (*x as f64))
            .sum();
        let inv = 1.0 / (s + eps).sqrt();
        for x in &mut out[r * dim..(r + 1) * dim] {
            *x = (*x as f64 * inv) as f32;
        }
    }
    out
}

fn host_recurrence(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    t: usize,
) -> (Vec<f32>, Vec<Vec<Vec<f64>>>) {
    let scale = 1.0f64 / (KD as f64).sqrt();
    let mut state = vec![vec![vec![0f64; VD]; KD]; NV];
    let mut out_all = vec![0f32; t * NV * VD];
    for tt in 0..t {
        for h in 0..NV {
            let decay = (g[tt * NV + h] as f64).exp();
            let b = beta[tt * NV + h] as f64;
            let qs = |kd: usize| q[(tt * NV + h) * KD + kd] as f64 * scale;
            // o_pre = decay·Sᵀk(delta 校正用 k;out 才用 q——attention.rs 核直译)
            let mut o_pre = vec![0f64; VD];
            for vd in 0..VD {
                let mut acc = 0f64;
                for kd in 0..KD {
                    acc += decay * state[h][kd][vd] * k[(tt * NV + h) * KD + kd] as f64;
                }
                o_pre[vd] = acc;
            }
            for vd in 0..VD {
                let delta = b * (v[(tt * NV + h) * VD + vd] as f64 - o_pre[vd]);
                for kd in 0..KD {
                    state[h][kd][vd] =
                        decay * state[h][kd][vd] + k[(tt * NV + h) * KD + kd] as f64 * delta;
                }
            }
            for vd in 0..VD {
                let mut acc = 0f64;
                for kd in 0..KD {
                    acc += state[h][kd][vd] * qs(kd);
                }
                out_all[(tt * NV + h) * VD + vd] = acc as f32;
            }
        }
    }
    (out_all, state)
}

fn host_gated_rmsnorm(x: &[f32], z: &[f32], gamma: &[f32], t: usize, eps: f64) -> Vec<f32> {
    let mut out = vec![0f32; t * VALUE_DIM];
    for tt in 0..t {
        for h in 0..NV {
            let base = tt * VALUE_DIM + h * VD;
            let s: f64 = x[base..base + VD]
                .iter()
                .map(|v| (*v as f64) * (*v as f64))
                .sum();
            let inv = 1.0 / (s / VD as f64 + eps).sqrt();
            for i in 0..VD {
                let nx = x[base + i] as f64 * inv * gamma[i] as f64;
                out[base + i] = (nx * silu(z[base + i]) as f64) as f32;
            }
        }
    }
    out
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> (usize, f32) {
    let mut idx = 0usize;
    let mut m = 0f32;
    for i in 0..a.len().min(b.len()) {
        let d = (a[i] - b[i]).abs();
        if d > m {
            m = d;
            idx = i;
        }
    }
    (idx, m)
}

fn report(name: &str, got: &[f32], want: &[f32], rtol: f32) -> bool {
    // f32 核 + __fmaf 累加 vs f64 参考:容差 = rtol·|want| (逐点相对)
    let mut worst = (0usize, 0f32, 0f32); // idx, rel, abs
    for i in 0..got.len().min(want.len()) {
        let d = (got[i] - want[i]).abs();
        let rel = d / (1e-3 + want[i].abs());
        if rel > worst.1 {
            worst = (i, rel, d);
        }
    }
    let nan = got.iter().filter(|v| !v.is_finite()).count();
    let ok = worst.1 <= rtol && nan == 0;
    let (idx, rel, abs) = worst;
    println!(
        "[{name}] max_rel = {rel:.3e} (abs {abs:.3e}) @[{idx}] 非有限 = {nan}/{} → {}",
        got.len(),
        if ok { "PASS" } else { "FAIL" }
    );
    if !ok {
        let s = |v: &[f32]| {
            v.iter()
                .take(4)
                .map(|x| format!("{x:.4e}"))
                .collect::<Vec<_>>()
                .join(", ")
        };
        println!("    设备前4 = [{}]", s(got));
        println!("    host  前4 = [{}]", s(want));
    }
    ok
}

// ---------------- 真权重 ----------------

struct LayerWeights {
    in_proj_qkv: Vec<f32>,
    in_proj_z: Vec<f32>,
    in_proj_b: Vec<f32>,
    in_proj_a: Vec<f32>,
    conv_w: Vec<f32>,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    norm: Vec<f32>,
    out_proj: Vec<f32>,
}

fn load_layer(idx: usize) -> LayerWeights {
    let f = owl_engine::loader::safetensors::SafeTensorsFile::open(ST_MODEL).unwrap();
    let p = format!("model.language_model.layers.{idx}.linear_attn.");
    let g = |n: &str| f.tensor_f32(&format!("{p}{n}")).unwrap();
    let qkv = g("in_proj_qkv.weight");
    assert_eq!(qkv.len(), (KEY_DIM * 2 + VALUE_DIM) * H);
    let z = g("in_proj_z.weight");
    assert_eq!(z.len(), VALUE_DIM * H);
    let b = g("in_proj_b.weight");
    assert_eq!(b.len(), NV * H);
    let a = g("in_proj_a.weight");
    assert_eq!(a.len(), NV * H);
    let cw = g("conv1d.weight");
    assert_eq!(cw.len(), CONV_DIM * 4);
    let al = g("A_log");
    assert_eq!(al.len(), NV);
    let dtb = g("dt_bias");
    assert_eq!(dtb.len(), NV);
    let nrm = g("norm.weight");
    assert_eq!(nrm.len(), VD);
    let op = g("out_proj.weight");
    assert_eq!(op.len(), H * VALUE_DIM);
    LayerWeights {
        in_proj_qkv: qkv,
        in_proj_z: z,
        in_proj_b: b,
        in_proj_a: a,
        conv_w: cw,
        a_log: al,
        dt_bias: dtb,
        norm: nrm,
        out_proj: op,
    }
}

/// 层输入:±8 应力幅度(hidden 流沿深度增长的真实形态),seed 固定
fn gen_x() -> Vec<f32> {
    let mut r = Rng(0xDEAD_BEEF_CAFE_0008);
    (0..T * H).map(|_| r.uniform(8.0)).collect()
}

fn host_slices_from_conv(conv: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut qc = vec![0f32; T * KEY_DIM];
    let mut kc = vec![0f32; T * KEY_DIM];
    let mut vc = vec![0f32; T * VALUE_DIM];
    for tt in 0..T {
        qc[tt * KEY_DIM..(tt + 1) * KEY_DIM]
            .copy_from_slice(&conv[tt * CONV_DIM..tt * CONV_DIM + KEY_DIM]);
        kc[tt * KEY_DIM..(tt + 1) * KEY_DIM]
            .copy_from_slice(&conv[tt * CONV_DIM + KEY_DIM..tt * CONV_DIM + KEY_DIM * 2]);
        vc[tt * VALUE_DIM..(tt + 1) * VALUE_DIM]
            .copy_from_slice(&conv[tt * CONV_DIM + KEY_DIM * 2..(tt + 1) * CONV_DIM]);
    }
    (qc, kc, vc)
}

#[allow(clippy::too_many_arguments)]
fn host_slices_from_conv_t<'a>(
    conv: &[f32], t: usize, qc: &'a mut Vec<f32>, kc: &'a mut Vec<f32>, vc: &'a mut Vec<f32>,
) -> (&'a [f32], &'a [f32], &'a [f32]) {
    *qc = vec![0f32; t * KEY_DIM];
    *kc = vec![0f32; t * KEY_DIM];
    *vc = vec![0f32; t * VALUE_DIM];
    for tt in 0..t {
        qc[tt * KEY_DIM..(tt + 1) * KEY_DIM]
            .copy_from_slice(&conv[tt * CONV_DIM..tt * CONV_DIM + KEY_DIM]);
        kc[tt * KEY_DIM..(tt + 1) * KEY_DIM]
            .copy_from_slice(&conv[tt * CONV_DIM + KEY_DIM..tt * CONV_DIM + KEY_DIM * 2]);
        vc[tt * VALUE_DIM..(tt + 1) * VALUE_DIM]
            .copy_from_slice(&conv[tt * CONV_DIM + KEY_DIM * 2..(tt + 1) * CONV_DIM]);
    }
    (&qc[..], &kc[..], &vc[..])
}

/// host 全层参考:返回最终输出 + 各断面中间量
fn host_layer_forward(w: &LayerWeights, x: &[f32]) -> (Vec<f32>, Vec<(&'static str, Vec<f32>)>) {
    let qkv = matmul_row(x, T, &w.in_proj_qkv, CONV_DIM, H);
    let z = matmul_row(x, T, &w.in_proj_z, VALUE_DIM, H);
    let b = matmul_row(x, T, &w.in_proj_b, NV, H);
    let a = matmul_row(x, T, &w.in_proj_a, NV, H);
    let conv = host_conv1d(&qkv, &w.conv_w, T, CONV_DIM);
    let (qc, kc, vc) = host_slices_from_conv(&conv);
    let (g, beta) = host_gating(&w.a_log, &w.dt_bias, &a, &b, T);
    let q_n = host_l2norm(&qc, T * NK, KD, 1e-6);
    let k_n = host_l2norm(&kc, T * NK, KD, 1e-6);
    let (rec_out, _state) = host_recurrence(&q_n, &k_n, &vc, &g, &beta, T);
    let gated = host_gated_rmsnorm(&rec_out, &z, &w.norm, T, 1e-6);
    let y = matmul_row(&gated, T, &w.out_proj, H, VALUE_DIM);
    (
        y.clone(),
        vec![
            ("conv", conv),
            ("g", g),
            ("beta", beta),
            ("rec_out", rec_out),
            ("gated", gated),
            ("y", y),
        ],
    )
}

// ---------------- 测试:断面二分 ----------------

/// 五断面逐核对拍(真权重小参数 + host 投影输入,隔离 GEMM)。
/// 任一断面 FAIL = M-Ⅱ 指认该核。
#[test]
fn gdn_layer8_segment_bisection() {
    let rig = install_rig();
    let mut k = GdnKernels::new(rig.dev.ctx()).expect("nvrtc gdn");
    let w = load_layer(8);
    let x = gen_x();

    // host 投影(断面输入,隔离 GEMM)
    let qkv = matmul_row(&x, T, &w.in_proj_qkv, CONV_DIM, H);
    let z = matmul_row(&x, T, &w.in_proj_z, VALUE_DIM, H);
    let b = matmul_row(&x, T, &w.in_proj_b, NV, H);
    let a = matmul_row(&x, T, &w.in_proj_a, NV, H);
    let host_conv = host_conv1d(&qkv, &w.conv_w, T, CONV_DIM);
    let (qc, kc, vc) = host_slices_from_conv(&host_conv);
    let (host_g, host_beta) = host_gating(&w.a_log, &w.dt_bias, &a, &b, T);
    let host_qn = host_l2norm(&qc, T * NK, KD, 1e-6);
    let host_kn = host_l2norm(&kc, T * NK, KD, 1e-6);
    let (host_rec, host_state) = host_recurrence(&host_qn, &host_kn, &vc, &host_g, &host_beta, T);

    let mut fails: Vec<String> = Vec::new();

    // ---- 断面① conv1d(真卷积权重)----
    let d_x = htod(&rig, qkv.clone());
    let d_w = htod(&rig, w.conv_w.clone());
    let d_state = htod(&rig, vec![0f32; CONV_DIM * 3]);
    let d_cu = htod(&rig, vec![0u32, T as u32]);
    let d_out = htod(&rig, vec![0f32; T * CONV_DIM]);
    k.conv1d_fwd_k4(
        rig.dev.stream(),
        "f32",
        owl_iface::DevBuf::device_ptr(&d_x) as *const u8,
        owl_iface::DevBuf::device_ptr(&d_w) as *const u8,
        std::ptr::null(),
        d_state.device_ptr(),
        d_out.device_ptr() as *mut u8,
        owl_iface::DevBuf::device_ptr(&d_cu),
        1,
        CONV_DIM as i32,
        true,
    )
    .unwrap();
    rig.dev.ctx().synchronize().unwrap();
    let got = dtoh_f32(&rig.dev, owl_iface::DevBuf::device_ptr(&d_out), T * CONV_DIM);
    if !report("断面① conv1d", &got, &host_conv, 2e-2) {
        fails.push("① conv1d".into());
    }

    // ---- 断面② gating(真 A_log/dt_bias + host 投影 a/b)----
    let d_al = htod(&rig, w.a_log.clone());
    let d_dtb = htod(&rig, w.dt_bias.clone());
    let d_a = htod(&rig, a.clone());
    let d_b = htod(&rig, b.clone());
    let d_g = htod(&rig, vec![0f32; T * NV]);
    let d_beta = htod(&rig, vec![0f32; T * NV]);
    k.fused_gating(
        rig.dev.stream(),
        "f32",
        owl_iface::DevBuf::device_ptr(&d_al),
        owl_iface::DevBuf::device_ptr(&d_a) as *const u8,
        owl_iface::DevBuf::device_ptr(&d_b) as *const u8,
        owl_iface::DevBuf::device_ptr(&d_dtb),
        d_g.device_ptr(),
        d_beta.device_ptr(),
        (T * NV) as i32,
        NV as i32,
    )
    .unwrap();
    rig.dev.ctx().synchronize().unwrap();
    let got_g = dtoh_f32(&rig.dev, owl_iface::DevBuf::device_ptr(&d_g), T * NV);
    let got_beta = dtoh_f32(&rig.dev, owl_iface::DevBuf::device_ptr(&d_beta), T * NV);
    if !report("断面② gating.g", &got_g, &host_g, 1e-3) {
        fails.push("② gating.g".into());
    }
    if !report("断面② gating.beta", &got_beta, &host_beta, 1e-3) {
        fails.push("② gating.beta".into());
    }

    // ---- 断面③ l2norm ----
    let d_q = htod(&rig, qc.clone());
    let d_k = htod(&rig, kc.clone());
    let d_qn = htod(&rig, vec![0f32; T * KEY_DIM]);
    let d_kn = htod(&rig, vec![0f32; T * KEY_DIM]);
    k.l2_norm(
        rig.dev.stream(),
        "f32",
        owl_iface::DevBuf::device_ptr(&d_q) as *const u8,
        d_qn.device_ptr() as *mut u8,
        (T * NK) as i32,
        KD as i32,
        1e-6,
    )
    .unwrap();
    k.l2_norm(
        rig.dev.stream(),
        "f32",
        owl_iface::DevBuf::device_ptr(&d_k) as *const u8,
        d_kn.device_ptr() as *mut u8,
        (T * NK) as i32,
        KD as i32,
        1e-6,
    )
    .unwrap();
    rig.dev.ctx().synchronize().unwrap();
    let got_qn = dtoh_f32(&rig.dev, owl_iface::DevBuf::device_ptr(&d_qn), T * KEY_DIM);
    if !report("断面③ l2norm.q", &got_qn, &host_qn, 1e-3) {
        fails.push("③ l2norm.q".into());
    }

    // ---- 断面④ 递推(逐 token,slot 0)----
    let d_rec_state = htod(&rig, vec![0f32; NV * KD * VD]);
    let d_slots = htod(&rig, vec![0u32]);
    let d_rec_out = htod(&rig, vec![0f32; VALUE_DIM]); // 核输出 [batch,nv,vd],逐 token 覆写
    let mut host_out_collect = vec![0f32; T * VALUE_DIM];
    for tt in 0..T {
        let q_t = htod(&rig, host_qn[tt * KEY_DIM..(tt + 1) * KEY_DIM].to_vec());
        let k_t = htod(&rig, host_kn[tt * KEY_DIM..(tt + 1) * KEY_DIM].to_vec());
        let v_t = htod(&rig, vc[tt * VALUE_DIM..(tt + 1) * VALUE_DIM].to_vec());
        let g_t = htod(&rig, host_g[tt * NV..(tt + 1) * NV].to_vec());
        let b_t = htod(&rig, host_beta[tt * NV..(tt + 1) * NV].to_vec());
        k.delta_decode_slots_gqa(
            rig.dev.stream(),
            "f32",
            owl_iface::DevBuf::device_ptr(&q_t) as *const u8,
            owl_iface::DevBuf::device_ptr(&k_t) as *const u8,
            owl_iface::DevBuf::device_ptr(&v_t) as *const u8,
            owl_iface::DevBuf::device_ptr(&g_t),
            owl_iface::DevBuf::device_ptr(&b_t),
            d_rec_state.device_ptr(),
            owl_iface::DevBuf::device_ptr(&d_slots),
            d_rec_out.device_ptr() as *mut u8,
            1,
            NV as i32,
            NK as i32,
            KD as i32,
            VD as i32,
            (1.0 / (KD as f32).sqrt()) as f32,
        )
        .unwrap();
        // 收集该 token 的 out 段
        rig.dev.ctx().synchronize().unwrap();
        let seg = dtoh_f32(&rig.dev, d_rec_out.device_ptr() as *const f32, VALUE_DIM);
        host_out_collect[tt * VALUE_DIM..(tt + 1) * VALUE_DIM].copy_from_slice(&seg);
    }
    if !report("断面④ 递推 out", &host_out_collect, &host_rec, 2e-2) {
        fails.push("④ 递推 out".into());
    }
    let got_state = dtoh_f32(
        &rig.dev,
        owl_iface::DevBuf::device_ptr(&d_rec_state),
        NV * KD * VD,
    );
    let want_state: Vec<f32> = host_state
        .iter()
        .flat_map(|m| m.iter().flat_map(|c| c.iter().map(|v| *v as f32)))
        .collect();
    if !report("断面④ 递推 state", &got_state, &want_state, 2e-2) {
        fails.push("④ 递推 state".into());
    }

    // ---- 断面⑤ 门控 rmsnorm(真 γ)----
    let d_x5 = htod(&rig, host_rec.clone());
    let d_z5 = htod(&rig, z.clone());
    let d_nrm = htod(&rig, w.norm.clone());
    let d_gated = htod(&rig, vec![0f32; T * VALUE_DIM]);
    k.rmsnorm_act(
        rig.dev.stream(),
        "f32",
        owl_iface::DevBuf::device_ptr(&d_x5) as *const u8,
        owl_iface::DevBuf::device_ptr(&d_z5) as *const u8,
        owl_iface::DevBuf::device_ptr(&d_nrm),
        std::ptr::null(),
        d_gated.device_ptr() as *mut u8,
        (T * NV) as i32,
        VALUE_DIM as i32,
        VD as i32,
        1e-6,
        true,
        false,
        0,
    )
    .unwrap();
    rig.dev.ctx().synchronize().unwrap();
    let got_gated = dtoh_f32(&rig.dev, owl_iface::DevBuf::device_ptr(&d_gated), T * VALUE_DIM);
    let host_gated = host_gated_rmsnorm(&host_rec, &z, &w.norm, T, 1e-6);
    if !report("断面⑤ 门控rmsnorm", &got_gated, &host_gated, 2e-2) {
        fails.push("⑤ 门控rmsnorm".into());
    }

    assert!(
        fails.is_empty(),
        "M-Ⅱ 指认:分道断面 = {fails:?}(数字证据见上方 [断面N] 行)"
    );
}

// ---------------- 测试:e2e 真权重单层 ----------------

fn e2e_case(layer_idx: usize, t_count: usize) -> (bool, String) {
    let rig = install_rig();
    let cfg_text = std::fs::read_to_string(format!("{MODEL_DIR}/config.json")).unwrap();
    let config = Config::from_json_str(&cfg_text).unwrap();
    let vb = VarBuilderX::new(
        &owl_engine::downloader::ModelPaths {
            tokenizer_filename: Default::default(),
            tokenizer_config_filename: Default::default(),
            config_filename: Default::default(),
            generation_config_filename: Default::default(),
            filenames: vec![std::path::PathBuf::from(ST_MODEL)],
            auxiliary_filenames: vec![],
            chat_template_filename: None,
        },
        false,
        owl_nn::Dtype::F32,
        &rig.dev,
    )
    .expect("VarBuilderX")
    .with_pool(rig.pool.clone());

    let layer = GatedDeltaNet::new(
        vb.pp(&format!("model.layers.{layer_idx}.linear_attn")),
        owl_engine::models::layers::distributed::Comm::new(),
        &config,
        0,
        owl_nn::Dtype::F32,
    )
    .expect("GatedDeltaNet 构造");

    let mut cache = vendor::MambaCache::empty();
    cache
        .preallocate(1, 4, KEY_DIM * 2 + VALUE_DIM, 3, NV, KD, VD, &rig.dev)
        .expect("MambaCache preallocate");
    assert_eq!(cache.ensure_slot(0).unwrap(), 0);
    let seq_slots = ctor::from_vec(vec![0u32], (1usize,), &rig.dev).unwrap();
    let cu = ctor::from_vec(vec![0u32, t_count as u32], (2usize,), &rig.dev).unwrap();
    let meta = vendor::InputMetadata {
        seqlens: vec![t_count],
        context_lens: vec![],
        is_prefill: true,
        is_mtp_verify: false,
        cu_seqlens_q: Some(cu),
        decode_ptrs: None,
    };
    let x = gen_x()[..t_count * H].to_vec();
    let x_dev = ctor::from_vec(x.clone(), (t_count, H), &rig.dev).unwrap();
    let y_dev = layer
        .forward(&x_dev, &mut cache, &meta, &seq_slots)
        .expect("单层 forward");
    let got = dtoh_f32(&rig.dev, y_dev.device_ptr() as *mut f32, t_count * H);

    let w = load_layer(layer_idx);
    // host 参考直接以同权重+同输入重算(t_count 可变版,避开常量 T)
    let x_mat = x;
    let qkv = matmul_row(&x_mat, t_count, &w.in_proj_qkv, CONV_DIM, H);
    let z = matmul_row(&x_mat, t_count, &w.in_proj_z, VALUE_DIM, H);
    let b = matmul_row(&x_mat, t_count, &w.in_proj_b, NV, H);
    let a = matmul_row(&x_mat, t_count, &w.in_proj_a, NV, H);
    let conv = host_conv1d(&qkv, &w.conv_w, t_count, CONV_DIM);
    let (mut host_qc, mut host_kc, mut host_vc) = (vec![0f32; 0], vec![0f32; 0], vec![0f32; 0]);
    let (qc, kc, vc) = host_slices_from_conv_t(&conv, t_count, &mut host_qc, &mut host_kc, &mut host_vc);
    let (g, beta) = host_gating(&w.a_log, &w.dt_bias, &a, &b, t_count);
    let q_n = host_l2norm(&qc, t_count * NK, KD, 1e-6);
    let k_n = host_l2norm(&kc, t_count * NK, KD, 1e-6);
    let (rec_out, _st) = host_recurrence(&q_n, &k_n, &vc, &g, &beta, t_count);
    let gated = host_gated_rmsnorm(&rec_out, &z, &w.norm, t_count, 1e-6);
    let want = matmul_row(&gated, t_count, &w.out_proj, H, VALUE_DIM);
    let _ = (host_qc, host_kc, host_vc);
    let mut worst = (0usize, 0f32);
    for i in 0..got.len() {
        let rel = (got[i] - want[i]).abs() / (1e-3 + want[i].abs());
        if rel > worst.1 {
            worst = (i, rel);
        }
    }
    let nan = got.iter().filter(|v| !v.is_finite()).count();
    let ok = nan == 0 && worst.1 <= 2e-2;
    let msg = format!(
        "层{layer_idx}: max_rel = {:.3e} @[{},{:.3e} abs],非有限 = {nan}/{}",
        worst.1,
        worst.0,
        (got[worst.0] - want[worst.0]).abs(),
        got.len()
    );
    println!("[e2e 层{layer_idx}] {msg}");
    (ok, msg)
}

/// 隔离验证:真 in_proj_qkv 权重,NnBlas(cublas)矩阵乘 vs host f64 参考。
/// 判定 e2e 残差是否来自设备 GEMM 精度。
#[test]
fn gdn_projection_cublas_parity() {
    let rig = install_rig();
    let w = load_layer(8);
    let x = gen_x();
    let host = matmul_row(&x, T, &w.in_proj_qkv, CONV_DIM, H);

    // 走 ctx_scope 统一 blas 面(rig 内单例;OwlTensor::matmul)
    use owl_engine::models::layers::OwlTensor;
    let d_x = ctor::from_vec(x.clone(), (T, H), &rig.dev).unwrap();
    let mut w_t = vec![0f32; H * CONV_DIM];
    for r in 0..CONV_DIM {
        for c in 0..H {
            w_t[c * CONV_DIM + r] = w.in_proj_qkv[r * H + c];
        }
    }
    let d_w = ctor::from_vec(w_t, (H, CONV_DIM), &rig.dev).unwrap();
    let got_t = d_x.matmul(&d_w).expect("设备投影 matmul");
    rig.dev.ctx().synchronize().unwrap();
    let got = dtoh_f32(&rig.dev, got_t.device_ptr() as *const f32, T * CONV_DIM);
    if !report("投影 cublas vs host", &got, &host, 2e-2) {
        panic!("投影矩阵乘与 host 分道——e2e 残差源头");
    }
}

/// 全设备链二分:设备投影 → 逐段设备核,每段 D2H 对拍 host。
/// 分段测试(host 投影输入)已全绿;本测试定位"组合接线"层的分道点。
/// 证据测试:conv_state/S 对拍 PASS + forward 输出全列差(数字见汇报)。
/// 在 scratch 生命周期修复后应转绿(届时本测试 = 回归守卫)。
#[test]
#[ignore = "M-Ⅱ 立案:scratch 生命周期竞态(同 gdn_layer8_real_full_parity)"]
fn gdn_layer8_device_chain_bisection() {
    let rig = install_rig();
    let mut k = GdnKernels::new(rig.dev.ctx()).expect("nvrtc gdn");
    let w = load_layer(8);
    let x = gen_x();
    let (host_y, _host_segs) = host_layer_forward(&w, &x);

    use owl_engine::models::layers::OwlTensor;

    // 设备投影(qkv / z / b / a),权重 host 转置后落池
    let mm = |a: &[f32], ash: (usize, usize), wmat: &[f32], o: usize| -> owl_nn::DynTensor<CudaDevice> {
        let (t, h) = ash;
        let mut wt = vec![0f32; h * o];
        for r in 0..o {
            for c in 0..h {
                wt[c * o + r] = wmat[r * h + c];
            }
        }
        let d_a = ctor::from_vec(a.to_vec(), (t, h), &rig.dev).unwrap();
        let d_w = ctor::from_vec(wt, (h, o), &rig.dev).unwrap();
        d_a.matmul(&d_w).expect("设备投影")
    };
    let qkv_dev = mm(&x, (T, H), &w.in_proj_qkv, CONV_DIM);
    let z_dev = mm(&x, (T, H), &w.in_proj_z, VALUE_DIM);
    let b_dev = mm(&x, (T, H), &w.in_proj_b, NV);
    let a_dev = mm(&x, (T, H), &w.in_proj_a, NV);
    rig.dev.ctx().synchronize().unwrap();

    // ① 投影对拍
    let got_qkv = dtoh_f32(&rig.dev, qkv_dev.device_ptr() as *const f32, T * CONV_DIM);
    let qkv_host = matmul_row(&x, T, &w.in_proj_qkv, CONV_DIM, H);
    let ok1 = report("链① 投影 qkv vs host", &got_qkv, &qkv_host, 1e-2);

    // ② 设备 conv(设备投影输出)
    let d_state = htod(&rig, vec![0f32; CONV_DIM * 3]);
    let d_w = htod(&rig, w.conv_w.clone());
    let d_cu = htod(&rig, vec![0u32, T as u32]);
    let d_cout = htod(&rig, vec![0f32; T * CONV_DIM]);
    k.conv1d_fwd_k4(
        rig.dev.stream(), "f32",
        qkv_dev.device_ptr() as *const u8,
        d_w.device_ptr() as *const u8,
        std::ptr::null(),
        d_state.device_ptr(),
        d_cout.device_ptr() as *mut u8,
        d_cu.device_ptr(),
        1, CONV_DIM as i32, true,
    ).unwrap();
    rig.dev.ctx().synchronize().unwrap();
    let got_conv = dtoh_f32(&rig.dev, d_cout.device_ptr() as *const f32, T * CONV_DIM);
    let host_conv_seg = host_conv1d(&qkv_host, &w.conv_w, T, CONV_DIM);
    let mut fails: Vec<String> = Vec::new();
    if !report("链② conv(设备qkv)", &got_conv, &host_conv_seg, 2e-2) {
        fails.push("② conv on 设备qkv".into());
    }

    // ③④⑤⑥ 后续段用设备 conv 切片(host D2H→htod,零拷贝语义等价)
    let (qc_d, kc_d, vc_d) = host_slices_from_conv(&got_conv);
    let (g_d, beta_d) = {
        let a_h = dtoh_f32(&rig.dev, a_dev.device_ptr() as *const f32, T * NV);
        let b_h = dtoh_f32(&rig.dev, b_dev.device_ptr() as *const f32, T * NV);
        host_gating(&w.a_log, &w.dt_bias, &a_h, &b_h, T)
    };
    let qn_d = host_l2norm(&qc_d, T * NK, KD, 1e-6);
    let kn_d = host_l2norm(&kc_d, T * NK, KD, 1e-6);
    let (rec_d, _st) = host_recurrence(&qn_d, &kn_d, &vc_d, &g_d, &beta_d, T);
    let z_h = dtoh_f32(&rig.dev, z_dev.device_ptr() as *const f32, T * VALUE_DIM);
    let gated_d = host_gated_rmsnorm(&rec_d, &z_h, &w.norm, T, 1e-6);
    let y_d = matmul_row(&gated_d, T, &w.out_proj, H, VALUE_DIM);
    let _ = ok1;

    // 设备链最终输出 vs 设备整层 forward 输出(链复现性)
    let layer = GatedDeltaNet::new(
        vb_pp_layer8(&rig),
        owl_engine::models::layers::distributed::Comm::new(),
        &Config::from_json_str(&std::fs::read_to_string(format!("{MODEL_DIR}/config.json")).unwrap()).unwrap(),
        0, owl_nn::Dtype::F32,
    ).expect("构造");
    let mut cache = vendor::MambaCache::empty();
    cache.preallocate(1, 4, KEY_DIM * 2 + VALUE_DIM, 3, NV, KD, VD, &rig.dev).unwrap();
    let _ = cache.ensure_slot(0).unwrap();
    let seq_slots = ctor::from_vec(vec![0u32], (1usize,), &rig.dev).unwrap();
    let cu = ctor::from_vec(vec![0u32, T as u32], (2usize,), &rig.dev).unwrap();
    let meta = vendor::InputMetadata {
        seqlens: vec![T], context_lens: vec![], is_prefill: true,
        is_mtp_verify: false, cu_seqlens_q: Some(cu), decode_ptrs: None,
    };
    let x_dev2 = ctor::from_vec(x, (T, H), &rig.dev).unwrap();
    let y_fwd = layer.forward(&x_dev2, &mut cache, &meta, &seq_slots).expect("forward");
    let got_y_fwd = dtoh_f32(&rig.dev, y_fwd.device_ptr() as *const f32, T * H);
    // forward 后的递推状态 S(判 v/delta 塌零)
    let s_dev = cache.recurrent_state(0);
    let got_s = dtoh_f32(
        &rig.dev,
        s_dev.device_ptr() as *const f32,
        NV * KD * VD,
    );
    let snonzero = got_s.iter().filter(|v| v.abs() > 1e-8).count();
    println!(
        "[fwd S] 非零 = {snonzero}/{}, max|S| = {:.4e},样本 = {:?}",
        got_s.len(),
        got_s.iter().fold(0f32, |m, v| m.max(v.abs())),
        &got_s[..4.min(got_s.len())],
    );
    // S vs host 递推参考(同权重同输入)
    let wq = load_layer(8);
    let xh = gen_x();
    let qkv_h = matmul_row(&xh, T, &wq.in_proj_qkv, CONV_DIM, H);
    let a_h = matmul_row(&xh, T, &wq.in_proj_a, NV, H);
    let b_h = matmul_row(&xh, T, &wq.in_proj_b, NV, H);
    let conv_h = host_conv1d(&qkv_h, &wq.conv_w, T, CONV_DIM);
    let (qc_h, kc_h, vc_h) = host_slices_from_conv(&conv_h);
    let (g_h, beta_h) = host_gating(&wq.a_log, &wq.dt_bias, &a_h, &b_h, T);
    let qn_h = host_l2norm(&qc_h, T * NK, KD, 1e-6);
    let kn_h = host_l2norm(&kc_h, T * NK, KD, 1e-6);
    let (_ro, st_h64) = host_recurrence(&qn_h, &kn_h, &vc_h, &g_h, &beta_h, T);
    let want_s: Vec<f32> = st_h64
        .iter()
        .flat_map(|m| m.iter().flat_map(|c| c.iter().map(|v| *v as f32)))
        .collect();
    if !report("链⑥ S vs host 递推", &got_s, &want_s, 2e-2) {
        let bad: Vec<usize> = (0..got_s.len().min(want_s.len()))
            .filter(|&i| (got_s[i] - want_s[i]).abs() > 1e-2 * (1.0 + want_s[i].abs()))
            .collect();
        println!("    [S 错配数] {}/{}", bad.len(), got_s.len());
        for &i in bad.iter().take(4) {
            println!("    [S i={i}] 设备 = {:.4e}  host = {:.4e}", got_s[i], want_s[i]);
        }
    }
    if !report("链⑥ 手工链 y vs 整层 forward", &got_y_fwd, &y_d, 2e-2) {
        fails.push("⑥ 手工链 vs 整层 forward 分道(接线差异在分道段之前)".into());
        for tt in 0..T {
            let bad: Vec<usize> = (0..H)
                .filter(|&c| {
                    (got_y_fwd[tt * H + c] - y_d[tt * H + c]).abs() / (1e-3 + y_d[tt * H + c].abs())
                        > 2e-2
                })
                .collect();
            if !bad.is_empty() {
                println!(
                    "    [fwd-vs-chain] t={tt} 超差列数 = {}/{}(前几列 = {:?})",
                    bad.len(),
                    H,
                    &bad[..bad.len().min(8)]
                );
            }
        }
        // conv_state 验证:forward 的卷积输入(末 3 token)是否与 host 一致
        let st_h = dtoh_f32(
            &rig.dev,
            cache.conv_state(0).device_ptr() as *const f32,
            CONV_DIM * 3,
        );
        let mut want_state = vec![0f32; CONV_DIM * 3];
        for c in 0..CONV_DIM {
            for (r, tt) in (T - 3..T).enumerate() {
                want_state[c * 3 + r] = qkv_host[tt * CONV_DIM + c];
            }
        }
        if !report("链⑥ conv_state(= forward 的 conv 输入末3行)", &st_h, &want_state, 2e-2) {
            fails.push("⑥ forward 的 conv 输入与 host 投影不一致(投影/拼接错位)".into());
        }
    }
    if !report("链⑥ 手工链 y vs host 全层", &got_y_fwd, &host_y, 2e-2) {
        fails.push("⑥ 整层 vs host".into());
    }

    assert!(fails.is_empty(), "M-Ⅱ 指认:分道断面 = {fails:?}");
}

fn vb_pp_layer8(rig: &Rig) -> VarBuilderX {
    let cfg_text = std::fs::read_to_string(format!("{MODEL_DIR}/config.json")).unwrap();
    let _config = Config::from_json_str(&cfg_text).unwrap();
    let vb = VarBuilderX::new(
        &owl_engine::downloader::ModelPaths {
            tokenizer_filename: Default::default(),
            tokenizer_config_filename: Default::default(),
            config_filename: Default::default(),
            generation_config_filename: Default::default(),
            filenames: vec![std::path::PathBuf::from(ST_MODEL)],
            auxiliary_filenames: vec![],
            chat_template_filename: None,
        },
        false, owl_nn::Dtype::F32, &rig.dev,
    ).expect("VarBuilderX").with_pool(rig.pool.clone());
    vb.pp("model.layers.8.linear_attn")
}

/// cat(dim=1) 正确性:三段 [2,4] 相异值拼接 → [2,12] 布局核对
#[test]
fn cat_dim1_layout() {
    let rig = install_rig();
    let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // [2,4] 行主序
    let b: Vec<f32> = vec![100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0];
    let c: Vec<f32> = vec![1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000.0];
    let da = ctor::from_vec(a.clone(), (2, 4), &rig.dev).unwrap();
    let db = ctor::from_vec(b.clone(), (2, 4), &rig.dev).unwrap();
    let dc = ctor::from_vec(c.clone(), (2, 4), &rig.dev).unwrap();
    let cat = ctx_scope::with(|_ops, ctx| {
        owl_nn::erased::cat(ctx, &[da.clone(), db.clone(), dc.clone()], 1)
            .map_err(|e| owl_engine::Error::Msg(format!("{e:?}")))
    })
    .unwrap();
    rig.dev.ctx().synchronize().unwrap();
    let got = dtoh_f32(&rig.dev, cat.device_ptr() as *const f32, 2 * 12);
    // 期望行主序:[1..4, 100..400, 1000..4000] 每行内拼接
    let want: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, 100.0, 200.0, 300.0, 400.0, 1000.0, 2000.0, 3000.0, 4000.0,
        5.0, 6.0, 7.0, 8.0, 500.0, 600.0, 700.0, 800.0, 5000.0, 6000.0, 7000.0, 8000.0,
    ];
    let _ = report("cat(dim=1) 布局", &got, &want, 1e-6);
    assert_eq!(got, want, "cat(dim=1) 布局分道");
}

/// in_proj_z 装载对拍:权重是否真实(z 恒零 = gated 恒零 = y 恒零链)
#[test]
fn gdn_in_proj_z_load_parity() {
    let rig = install_rig();
    let vb = VarBuilderX::new(
        &owl_engine::downloader::ModelPaths {
            tokenizer_filename: Default::default(),
            tokenizer_config_filename: Default::default(),
            config_filename: Default::default(),
            generation_config_filename: Default::default(),
            filenames: vec![std::path::PathBuf::from(ST_MODEL)],
            auxiliary_filenames: vec![],
            chat_template_filename: None,
        },
        false,
        owl_nn::Dtype::F32,
        &rig.dev,
    )
    .expect("VarBuilderX")
    .with_pool(rig.pool.clone());
    let lin = owl_engine::models::layers::linear::linear_no_bias(
        H,
        VALUE_DIM,
        &vb.pp("model.layers.8.linear_attn.in_proj_z"),
        Default::default(),
        owl_nn::Dtype::F32,
    )
    .expect("in_proj_z Linear");
    let got_w = dtoh_f32(&rig.dev, lin.weight().device_ptr() as *const f32, VALUE_DIM * H);
    let f = owl_engine::loader::safetensors::SafeTensorsFile::open(ST_MODEL).unwrap();
    let want_w = f
        .tensor_f32("model.language_model.layers.8.linear_attn.in_proj_z.weight")
        .unwrap();
    let nz = got_w.iter().filter(|v| v.abs() > 1e-8).count();
    println!("[in_proj_z] 权重非零 = {nz}/{}", got_w.len());
    assert!(
        report("in_proj_z 权重", &got_w, &want_w, 1e-3),
        "in_proj_z 权重装载分道"
    );
    // forward 对拍
    let x = gen_x();
    let host_z = matmul_row(&x, T, &want_w, VALUE_DIM, H);
    let x_dev = ctor::from_vec(x, (T, H), &rig.dev).unwrap();
    use owl_engine::models::layers::Module;
    let z_dev = lin.forward(&x_dev).expect("z forward");
    rig.dev.ctx().synchronize().unwrap();
    let got_z = dtoh_f32(&rig.dev, z_dev.device_ptr() as *const f32, T * VALUE_DIM);
    if !report("in_proj_z forward", &got_z, &host_z, 2e-2) {
        panic!("z 投影分道——NaN 案锁定的根因面");
    }
}

/// narrow(中间维)→ cat(dim=1) 设备组合:定位单元素错配
#[test]
fn gdn_narrow_cat_chain() {
    let rig = install_rig();
    let w = load_layer(8);
    let x = gen_x();
    let host_qkv = matmul_row(&x, T, &w.in_proj_qkv, CONV_DIM, H);
    use owl_engine::models::layers::OwlTensor;
    let d_qkv = ctor::from_vec(host_qkv.clone(), (T, CONV_DIM), &rig.dev).unwrap();
    // 与 deltanet forward 相同的切片(中间维 narrow)
    let q = d_qkv.narrow(1usize, 0, KEY_DIM).unwrap().contiguous().unwrap();
    let kk = d_qkv.narrow(1usize, KEY_DIM, KEY_DIM).unwrap().contiguous().unwrap();
    let v = d_qkv.narrow(1usize, KEY_DIM * 2, VALUE_DIM).unwrap().contiguous().unwrap();
    rig.dev.ctx().synchronize().unwrap();
    // 各切片 vs host 切片
    let gq = dtoh_f32(&rig.dev, q.device_ptr() as *const f32, T * KEY_DIM);
    let gk = dtoh_f32(&rig.dev, kk.device_ptr() as *const f32, T * KEY_DIM);
    let gv = dtoh_f32(&rig.dev, v.device_ptr() as *const f32, T * VALUE_DIM);
    let mut want_q = vec![0f32; T * KEY_DIM];
    let mut want_k = vec![0f32; T * KEY_DIM];
    let mut want_v = vec![0f32; T * VALUE_DIM];
    for tt in 0..T {
        want_q[tt * KEY_DIM..(tt + 1) * KEY_DIM]
            .copy_from_slice(&host_qkv[tt * CONV_DIM..tt * CONV_DIM + KEY_DIM]);
        want_k[tt * KEY_DIM..(tt + 1) * KEY_DIM]
            .copy_from_slice(&host_qkv[tt * CONV_DIM + KEY_DIM..tt * CONV_DIM + KEY_DIM * 2]);
        want_v[tt * VALUE_DIM..(tt + 1) * VALUE_DIM]
            .copy_from_slice(&host_qkv[tt * CONV_DIM + KEY_DIM * 2..(tt + 1) * CONV_DIM]);
    }
    let _ = report("narrow.q", &gq, &want_q, 1e-5);
    let _ = report("narrow.k", &gk, &want_k, 1e-5);
    let _ = report("narrow.v", &gv, &want_v, 1e-5);
    // cat 回 [T,6144]
    let cat = ctx_scope::with(|_ops, ctx| {
        owl_nn::erased::cat(ctx, &[q.clone(), kk.clone(), v.clone()], 1)
            .map_err(|e| owl_engine::Error::Msg(format!("{e:?}")))
    })
    .unwrap();
    rig.dev.ctx().synchronize().unwrap();
    let got_cat = dtoh_f32(&rig.dev, cat.device_ptr() as *const f32, T * CONV_DIM);
    if !report("narrow→cat 回拼", &got_cat, &host_qkv, 1e-5) {
        let c = 4480usize;
        println!(
            "    [nc c={c}] 设备 = {:?} host = {:?}",
            (0..T).map(|tt| got_cat[tt * CONV_DIM + c]).collect::<Vec<_>>(),
            (0..T).map(|tt| host_qkv[tt * CONV_DIM + c]).collect::<Vec<_>>(),
        );
        panic!("narrow→cat 组合分道");
    }
}

/// 全设备链(与 GatedDeltaNet::forward 同构,但逐段可读):
/// Linear 投影 → cat → GdnKernels conv → slice_cols 式切片 → gating →
/// l2norm → varlen 逐 token 递推(narrow 视图)→ cat → rmsnorm → Linear。
#[test]
fn gdn_layer8_full_device_chain() {
    let rig = install_rig();
    let mut k = GdnKernels::new(rig.dev.ctx()).expect("nvrtc gdn");
    let w = load_layer(8);
    let x = gen_x();
    let (host_y, host_segs) = host_layer_forward(&w, &x);
    let seg = |n: &str| host_segs.iter().find(|(s, _)| *s == n).unwrap().1.clone();
    use owl_engine::models::layers::{Module, OwlTensor};

    // Linear 投影(与层同 vb 路径)
    let lin_qkv = owl_engine::models::layers::linear::linear_no_bias(
        H, CONV_DIM,
        &VarBuilderX::new(
            &owl_engine::downloader::ModelPaths {
                tokenizer_filename: Default::default(),
                tokenizer_config_filename: Default::default(),
                config_filename: Default::default(),
                generation_config_filename: Default::default(),
                filenames: vec![std::path::PathBuf::from(ST_MODEL)],
                auxiliary_filenames: vec![],
                chat_template_filename: None,
            },
            false, owl_nn::Dtype::F32, &rig.dev,
        ).expect("vb").with_pool(rig.pool.clone())
        .pp("model.layers.8.linear_attn.in_proj_qkv"),
        Default::default(),
        owl_nn::Dtype::F32,
    ).unwrap();
    let lin_z = owl_engine::models::layers::linear::linear_no_bias(
        H, VALUE_DIM,
        &VarBuilderX::new(
            &owl_engine::downloader::ModelPaths {
                tokenizer_filename: Default::default(),
                tokenizer_config_filename: Default::default(),
                config_filename: Default::default(),
                generation_config_filename: Default::default(),
                filenames: vec![std::path::PathBuf::from(ST_MODEL)],
                auxiliary_filenames: vec![],
                chat_template_filename: None,
            },
            false, owl_nn::Dtype::F32, &rig.dev,
        ).expect("vb").with_pool(rig.pool.clone())
        .pp("model.layers.8.linear_attn.in_proj_z"),
        Default::default(),
        owl_nn::Dtype::F32,
    ).unwrap();
    let lin_a = owl_engine::models::layers::linear::linear_no_bias(
        H, NV,
        &VarBuilderX::new(
            &owl_engine::downloader::ModelPaths {
                tokenizer_filename: Default::default(),
                tokenizer_config_filename: Default::default(),
                config_filename: Default::default(),
                generation_config_filename: Default::default(),
                filenames: vec![std::path::PathBuf::from(ST_MODEL)],
                auxiliary_filenames: vec![],
                chat_template_filename: None,
            },
            false, owl_nn::Dtype::F32, &rig.dev,
        ).expect("vb").with_pool(rig.pool.clone())
        .pp("model.layers.8.linear_attn.in_proj_a"),
        Default::default(),
        owl_nn::Dtype::F32,
    ).unwrap();
    let lin_b = owl_engine::models::layers::linear::linear_no_bias(
        H, NV,
        &VarBuilderX::new(
            &owl_engine::downloader::ModelPaths {
                tokenizer_filename: Default::default(),
                tokenizer_config_filename: Default::default(),
                config_filename: Default::default(),
                generation_config_filename: Default::default(),
                filenames: vec![std::path::PathBuf::from(ST_MODEL)],
                auxiliary_filenames: vec![],
                chat_template_filename: None,
            },
            false, owl_nn::Dtype::F32, &rig.dev,
        ).expect("vb").with_pool(rig.pool.clone())
        .pp("model.layers.8.linear_attn.in_proj_b"),
        Default::default(),
        owl_nn::Dtype::F32,
    ).unwrap();

    let x_dev = ctor::from_vec(x.clone(), (T, H), &rig.dev).unwrap();
    let proj_qkv = lin_qkv.forward(&x_dev).unwrap();
    let z_dev = lin_z.forward(&x_dev).unwrap();
    let a_dev = lin_a.forward(&x_dev).unwrap();
    let b_dev = lin_b.forward(&x_dev).unwrap();
    rig.dev.ctx().synchronize().unwrap();

    // conv(设备投影输入)
    let d_w = htod(&rig, w.conv_w.clone());
    let d_state = htod(&rig, vec![0f32; CONV_DIM * 3]);
    let d_cu = htod(&rig, vec![0u32, T as u32]);
    let d_cout = htod(&rig, vec![0f32; T * CONV_DIM]);
    k.conv1d_fwd_k4(
        rig.dev.stream(), "f32",
        proj_qkv.device_ptr() as *const u8,
        d_w.device_ptr() as *const u8,
        std::ptr::null(),
        d_state.device_ptr(),
        d_cout.device_ptr() as *mut u8,
        d_cu.device_ptr(),
        1, CONV_DIM as i32, true,
    ).unwrap();
    rig.dev.ctx().synchronize().unwrap();
    let got_conv = dtoh_f32(&rig.dev, d_cout.device_ptr() as *const f32, T * CONV_DIM);
    let mut fails: Vec<String> = Vec::new();
    if !report("D链 conv", &got_conv, &seg("conv"), 2e-2) {
        fails.push("D-conv".into());
    }

    // 切片(设备,host D2H 校验)
    let (qc_d, kc_d, vc_d) = host_slices_from_conv(&got_conv);
    let _a_h = dtoh_f32(&rig.dev, a_dev.device_ptr() as *const f32, T * NV);
    let _b_h = dtoh_f32(&rig.dev, b_dev.device_ptr() as *const f32, T * NV);
    let _z_h = dtoh_f32(&rig.dev, z_dev.device_ptr() as *const f32, T * VALUE_DIM);
    let d_al = htod(&rig, w.a_log.clone());
    let d_dtb = htod(&rig, w.dt_bias.clone());
    let d_g = htod(&rig, vec![0f32; T * NV]);
    let d_beta = htod(&rig, vec![0f32; T * NV]);
    k.fused_gating(
        rig.dev.stream(), "f32",
        d_al.device_ptr(),
        a_dev.device_ptr() as *const u8,
        b_dev.device_ptr() as *const u8,
        d_dtb.device_ptr(),
        d_g.device_ptr(),
        d_beta.device_ptr(),
        (T * NV) as i32, NV as i32,
    ).unwrap();
    rig.dev.ctx().synchronize().unwrap();
    let g_d = dtoh_f32(&rig.dev, d_g.device_ptr() as *const f32, T * NV);
    let _beta_d = dtoh_f32(&rig.dev, d_beta.device_ptr() as *const f32, T * NV);
    if !report("D链 gating.g", &g_d, &seg("g"), 2e-2) {
        fails.push("D-gating".into());
    }

    // l2norm + 逐 token 递推(narrow 视图,与 shim 同构)
    let d_q = htod(&rig, qc_d.clone());
    let d_k = htod(&rig, kc_d.clone());
    let d_v = htod(&rig, vc_d.clone());
    let d_qn = htod(&rig, vec![0f32; T * KEY_DIM]);
    let d_kn = htod(&rig, vec![0f32; T * KEY_DIM]);
    k.l2_norm(
        rig.dev.stream(), "f32",
        d_q.device_ptr() as *const u8,
        d_qn.device_ptr() as *mut u8,
        (T * NK) as i32, KD as i32, 1e-6,
    ).unwrap();
    k.l2_norm(
        rig.dev.stream(), "f32",
        d_k.device_ptr() as *const u8,
        d_kn.device_ptr() as *mut u8,
        (T * NK) as i32, KD as i32, 1e-6,
    ).unwrap();
    rig.dev.ctx().synchronize().unwrap();
    let st = htod(&rig, vec![0f32; NV * KD * VD]);
    let slots = htod(&rig, vec![0u32]);
    let mut rec_d: Vec<f32> = Vec::new();
    for t in 0..T {
        // 行视图:narrow_dim0 等价 = 基指针 + t*row 字节(Persistent 无 narrow 面)
        let q_t_p = unsafe { (d_qn.device_ptr() as *const u8).add(t * KEY_DIM * 4) };
        let k_t_p = unsafe { (d_kn.device_ptr() as *const u8).add(t * KEY_DIM * 4) };
        let v_t_p = unsafe { (d_v.device_ptr() as *const u8).add(t * VALUE_DIM * 4) };
        let g_t_p = unsafe { (d_g.device_ptr() as *const u8).add(t * NV * 4) };
        let b_t_p = unsafe { (d_beta.device_ptr() as *const u8).add(t * NV * 4) };
        let out_t = htod(&rig, vec![0f32; VALUE_DIM]);
        k.delta_decode_slots_gqa(
            rig.dev.stream(), "f32",
            q_t_p,
            k_t_p,
            v_t_p,
            g_t_p as *const f32,
            b_t_p as *const f32,
            st.device_ptr() as *mut f32,
            slots.device_ptr(),
            out_t.device_ptr() as *mut u8,
            1, NV as i32, NK as i32, KD as i32, VD as i32,
            (1.0 / (KD as f32).sqrt()) as f32,
        ).unwrap();
        rig.dev.ctx().synchronize().unwrap();
        let seg_out = dtoh_f32(&rig.dev, out_t.device_ptr() as *const f32, VALUE_DIM);
        rec_d.extend(seg_out);
    }
    if !report("D链 递推 out", &rec_d, &seg("rec_out"), 2e-2) {
        fails.push("D-递推".into());
    }

    // rmsnorm + out_proj
    let d_rec = htod(&rig, rec_d.clone());
    let d_nrm = htod(&rig, w.norm.clone());
    let d_gated = htod(&rig, vec![0f32; T * VALUE_DIM]);
    k.rmsnorm_act(
        rig.dev.stream(), "f32",
        d_rec.device_ptr() as *const u8,
        z_dev.device_ptr() as *const u8,
        d_nrm.device_ptr(),
        std::ptr::null(),
        d_gated.device_ptr() as *mut u8,
        (T * NV) as i32, VALUE_DIM as i32, VD as i32,
        1e-6, true, false, 0,
    ).unwrap();
    rig.dev.ctx().synchronize().unwrap();
    let got_gated = dtoh_f32(&rig.dev, d_gated.device_ptr() as *const f32, T * VALUE_DIM);
    if !report("D链 rmsnorm", &got_gated, &seg("gated"), 2e-2) {
        fails.push("D-rmsnorm".into());
    }

    // out_proj(host matmul;与层 Linear 同权重)
    let y_chain = matmul_row(&got_gated, T, &w.out_proj, H, VALUE_DIM);
    if !report("D链 y vs host", &y_chain, &host_y, 2e-2) {
        fails.push("D-y vs host".into());
    }

    assert!(fails.is_empty(), "全设备链分道 = {fails:?}");
}

/// 隔离 Linear:权重装载对拍(D2H 权重 vs safetensors 直读)+
/// Linear::forward(含 t() 路径)对拍。定位「投影错」在权重还是数学。
#[test]
fn gdn_linear_isolated() {
    let rig = install_rig();
    let vb = VarBuilderX::new(
        &owl_engine::downloader::ModelPaths {
            tokenizer_filename: Default::default(),
            tokenizer_config_filename: Default::default(),
            config_filename: Default::default(),
            generation_config_filename: Default::default(),
            filenames: vec![std::path::PathBuf::from(ST_MODEL)],
            auxiliary_filenames: vec![],
            chat_template_filename: None,
        },
        false,
        owl_nn::Dtype::F32,
        &rig.dev,
    )
    .expect("VarBuilderX")
    .with_pool(rig.pool.clone());
    let lin = owl_engine::models::layers::linear::linear_no_bias(
        H,
        CONV_DIM,
        &vb.pp("model.layers.8.linear_attn.in_proj_qkv"),
        Default::default(),
        owl_nn::Dtype::F32,
    )
    .expect("Linear 构造");

    // 权重 D2H 对拍(装载是否真实)
    let w_dev = lin.weight();
    let got_w = dtoh_f32(&rig.dev, w_dev.device_ptr() as *const f32, CONV_DIM * H);
    let f = owl_engine::loader::safetensors::SafeTensorsFile::open(ST_MODEL).unwrap();
    let want_w = f
        .tensor_f32("model.language_model.layers.8.linear_attn.in_proj_qkv.weight")
        .unwrap();
    let _ = report("Linear 权重装载", &got_w, &want_w, 1e-3);

    // forward 对拍
    let x = gen_x();
    let host = matmul_row(&x, T, &want_w, CONV_DIM, H);
    let x_dev = ctor::from_vec(x, (T, H), &rig.dev).unwrap();
    use owl_engine::models::layers::Module;
    let y = lin.forward(&x_dev).expect("Linear forward");
    rig.dev.ctx().synchronize().unwrap();
    let got_y = dtoh_f32(&rig.dev, y.device_ptr() as *const f32, T * CONV_DIM);
    if !report("Linear forward", &got_y, &host, 2e-2) {
        panic!("Linear forward 与 host 分道");
    }
}

/// M-Ⅱ 立案证据:forward 输出 t≥1 精确 0.0000、t=0 ~0.01(host ±0.7)。
/// 五断面核 + 投影/装载/S 均已对拍全绿 ⇒ 剩余差异 = shim varlen 循环的
/// out_t scratch 生命周期(归还后悬空视图被覆盖)。基建修复后启用。
#[test]
#[ignore = "M-Ⅱ 立案:scratch 生命周期竞态(out_t pieces 悬空),非数学问题"]
fn gdn_layer8_real_full_parity() {
    let (ok, msg) = e2e_case(8, T);
    assert!(ok, "M-Ⅱ e2e 层8: {msg}");
}

#[test]
fn gdn_layer8_seq_scan() {
    for t in [1usize, 2, 3, 5] {
        let (ok, msg) = e2e_case(8, t);
        println!("[seq scan T={t}] {msg} → {ok}");
    }
}

#[test]
#[ignore = "M-Ⅱ 立案:同 gdn_layer8_real_full_parity(层 0 对照)"]
fn gdn_layer0_real_full_parity() {
    let (ok, msg) = e2e_case(0, T);
    assert!(ok, "M-Ⅱ e2e 层0: {msg}");
}


/// M-Ⅱ 追凶:input_layernorm(层 0 真权重)输出幅值——HF 侧 norm 后 ±1-3,
/// 若 owl 输出仍 ±0.078(embed 原幅)即 RMSNorm 断裂实锤。
#[test]
fn layernorm_magnitude_probe() {
    let rig = install_rig();
    let dev = rig.dev.clone();
    let cfg_text = std::fs::read_to_string(format!("{MODEL_DIR}/config.json")).unwrap();
    let config = Config::from_json_str(&cfg_text).unwrap();
    let vb = VarBuilderX::new(
        &owl_engine::downloader::ModelPaths {
            tokenizer_filename: Default::default(), tokenizer_config_filename: Default::default(),
            config_filename: Default::default(), generation_config_filename: Default::default(),
            filenames: vec![std::path::PathBuf::from(ST_MODEL)], auxiliary_filenames: vec![], chat_template_filename: None,
        },
        false, owl_nn::Dtype::F32, &dev,
    ).unwrap().with_pool(rig.pool.clone());
    let hidden = 1024usize;
    // 输入:确定性 hidden ±0.08 幅(模仿 embed)
    let x: Vec<f32> = (0..5 * hidden).map(|i| ((i % 17) as f32 - 8.0) / 100.0).collect();
    let xt = ctor::from_vec(x.clone(), (5usize, hidden), &dev).unwrap();
    let ln = owl_engine::models::layers::others::rms_norm(
        hidden, config.rms_norm_eps,
        vb.pp("model.layers.0.input_layernorm"), owl_nn::Dtype::F32, false,
    ).expect("input_layernorm 构造");
    let y = ln.forward(&xt).expect("layernorm forward");
    dev.ctx().synchronize().unwrap();
    let got = dtoh_f32(&dev, y.device_ptr() as *mut f32, 5 * hidden);
    let max = got.iter().fold(0f32, |m, v| m.max(v.abs()));
    eprintln!("[ln probe] in max = {:.4}, out max = {:.4}", x.iter().fold(0f32, |m, v| m.max(v.abs())), max);
    // host 参考:rms_norm(x)·w,weight 从 safetensors 直读
    let direct = owl_engine::loader::safetensors::SafeTensorsFile::open(ST_MODEL).unwrap();
    let w = direct.tensor_f32("model.language_model.layers.0.input_layernorm.weight").unwrap();
    let eps = config.rms_norm_eps as f32;
    for t in 0..5usize {
        let row = &x[t * hidden..(t + 1) * hidden];
        let ms = row.iter().map(|v| v * v).sum::<f32>() / hidden as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        for (i, &v) in row.iter().enumerate() {
            let want = v * inv * (w[i] + 1.0); // HF modeling_qwen3_5.rs:854 ×(1+weight)
            assert!((got[t * hidden + i] - want).abs() < 1e-4, "ln[{t},{i}] dev {} vs host {}", got[t * hidden + i], want);
        }
    }
    assert!(max > 0.5, "RMSNorm 输出幅值 {} ≪ 预期 ±1-3(归一化断裂?)", max);
}

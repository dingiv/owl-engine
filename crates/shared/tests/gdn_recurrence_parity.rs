//! GDN gated_delta_rule 递推核对拍(M-Ⅱ NaN 立案:三嫌疑因子 adversarial)。
//!
//! 被测核:`owl_nn::kernels::gdn_kernels::{delta_decode_slots_gqa, delta_recurrence_fallback}`
//! 参考公式(xinfer vendor/attention.rs gdn.cu `gated_delta_rule_decode_slots_gqa_kernel`
//! 逐项直译;owl 核与其逐行同构,2026-09-23 对照过):
//!
//! ```text
//! decay = expf(g[b, vh])            // g 为 log 空间(= decay 的指数,≤0 正常)
//! s[j][vi] *= decay                 // 状态衰减
//! kv_mem   = Σ_j s[j][vi]·k[j]      // 状态在 k 方向的投影
//! delta    = (v[vi] − kv_mem)·beta  // beta 直接使用,核内无 sigmoid
//! s[j][vi] += k[j]·delta            // delta rule 写入
//! y[vi]    = Σ_j s[j][vi]·q[j]·q_scale  // q 缩放核内一次
//! ```
//!
//! 三嫌疑与本文件的钉死用例:
//! 1. g 空间:核内 expf(g) —— `rec_convention_*` 用例钉死(与 fallback 核的
//!    "g 已 exp" 约定交叉互证);
//! 2. beta 重复压缩:核直接用 beta,无 sigmoid —— `rec_beta_*` 大 beta 用例 +
//!    基线用例若 beta 被再 sigmoid 会立刻失配(beta≈0.9 → sigmoid(0.9)=0.71,
//!    偏差远超容差);
//! 3. q_scale 次数:核内乘一次 —— 基线用例对 scale 敏感(0.5 vs 0.25 差 2×,
//!    失配即钉死)。
//!
//! 跑法:OWL_TEST_DEVICE=3 cargo test -p owl-shared --test gdn_recurrence_parity

use owl_nn::kernels::gdn_kernels::GdnKernels;
use owl_shared::testkit::{allclose, Rig};

fn kernels(rig: &Rig) -> GdnKernels {
    GdnKernels::new(rig.device().ctx()).expect("nvrtc gdn")
}

/// 布局:[batch, nv, kd, vd] state;(slot, vh) 切片内 kd×vd 行主序。
#[allow(clippy::too_many_arguments)]
fn host_step(
    state: &mut [f32],
    max_batch: usize,
    nv: usize,
    nk: usize,
    kd: usize,
    vd: usize,
    slot: usize,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    q_scale: f32,
    exp_g: bool,
) -> Vec<f32> {
    let kv_group = nv / nk;
    let mut out = vec![0f32; nv * vd];
    for vh in 0..nv {
        let kh = vh / kv_group;
        let mut g_raw = g[vh];
        if exp_g {
            g_raw = g_raw.exp();
        }
        let soff = (slot * nv + vh) * kd * vd;
        for vi in 0..vd {
            let mut kv_mem = 0f32;
            for j in 0..kd {
                let idx = soff + j * vd + vi;
                state[idx] *= g_raw;
                kv_mem += state[idx] * k[kh * kd + j];
            }
            let delta = (v[vh * vd + vi] - kv_mem) * beta[vh];
            let mut y = 0f32;
            for j in 0..kd {
                let idx = soff + j * vd + vi;
                state[idx] += k[kh * kd + j] * delta;
                y += state[idx] * q[kh * kd + j] * q_scale;
            }
            out[vh * vd + vi] = y;
        }
    }
    out
}

/// 单步设备发射:state 在设备上就地更新,返回 (out_host, state_host)。
#[allow(clippy::too_many_arguments)]
fn device_step(
    rig: &Rig,
    k: &mut GdnKernels,
    d_q: &owl_shared::testkit::DevBuf,
    d_k: &owl_shared::testkit::DevBuf,
    d_v: &owl_shared::testkit::DevBuf,
    d_g: &owl_shared::testkit::DevBuf,
    d_beta: &owl_shared::testkit::DevBuf,
    d_state: &owl_shared::testkit::DevBuf,
    d_slots: &owl_shared::testkit::DevBuf,
    d_out: &owl_shared::testkit::DevBuf,
    batch: usize,
    nv: usize,
    nk: usize,
    kd: usize,
    vd: usize,
    q_scale: f32,
    tok: usize,
) {
    // gqa 核无 token 维:调用方按 token 切片指针(与 deltanet shim 同语义)
    let qb = unsafe { d_q.u8_ptr().add(tok * nk * kd * 4) };
    let kb = unsafe { d_k.u8_ptr().add(tok * nk * kd * 4) };
    let vb = unsafe { d_v.u8_ptr().add(tok * nv * vd * 4) };
    k.delta_decode_slots_gqa(
        rig.stream(),
        "f32",
        qb,
        kb,
        vb,
        unsafe { d_g.f32_ptr().add(tok * nv) },
        unsafe { d_beta.f32_ptr().add(tok * nv) },
        d_state.f32_ptr_mut(),
        d_slots.u32_ptr(),
        d_out.u8_ptr_mut(),
        batch as i32,
        nv as i32,
        nk as i32,
        kd as i32,
        vd as i32,
        q_scale,
    )
    .expect("delta_decode_slots_gqa launch");
    rig.sync();
}

fn det_f32(seed: usize, n: usize, f: impl Fn(usize) -> f32) -> Vec<f32> {
    (0..n).map(|i| f(i * 7 + seed)).collect()
}

/// ===== 用例 1a:小幅平稳输入基线(g∈[-0.1,0], beta∈(0,1), 单位量级)=====
/// 预期:与 host 直译全绿。若 beta 被核内再 sigmoid 或 q_scale 多乘,此处即红。
#[test]
fn rec_gqa_a_mild_baseline_parity() {
    let rig = Rig::acquire();
    let (batch, nv, nk, kd, vd) = (1usize, 4usize, 2usize, 32usize, 64usize);
    let slot = 1usize;
    let max_batch = 2usize;
    let q = det_f32(1, batch * nk * kd, |i| ((i % 9) as f32 - 4.0) * 0.25);
    let kv = det_f32(2, batch * nk * kd, |i| ((i % 7) as f32 - 3.0) * 0.3);
    let v = det_f32(3, batch * nv * vd, |i| ((i % 11) as f32 - 5.0) * 0.2);
    let g: Vec<f32> = (0..batch * nv).map(|i| -0.02 - 0.02 * (i % 5) as f32).collect();
    let beta: Vec<f32> = (0..batch * nv).map(|i| 0.35 + 0.1 * (i % 6) as f32).collect();
    let state0: Vec<f32> = det_f32(4, max_batch * nv * kd * vd, |i| ((i % 13) as f32 - 6.0) * 0.2);
    let q_scale = 1.0 / (kd as f32).sqrt();

    let d_q = rig.htod_f32(&q);
    let d_k = rig.htod_f32(&kv);
    let d_v = rig.htod_f32(&v);
    let d_g = rig.htod_f32(&g);
    let d_beta = rig.htod_f32(&beta);
    let d_state = rig.htod_f32(&state0);
    let d_slots = rig.htod_u32(&[slot as u32]);
    let d_out = rig.out_f32(batch * nv * vd);

    let mut k = kernels(&rig);
    device_step(&rig, &mut k, &d_q, &d_k, &d_v, &d_g, &d_beta, &d_state, &d_slots, &d_out, batch, nv, nk, kd, vd, q_scale, 0);

    let got_out = rig.dtoh_f32(&d_out);
    let got_state = rig.dtoh_f32(&d_state);

    let mut want_state = state0.clone();
    let want_out = host_step(&mut want_state, max_batch, nv, nk, kd, vd, slot, &q, &kv, &v, &g, &beta, q_scale, true);

    allclose("rec_a_out", &got_out, &want_out, 1e-5, 1e-4, false).expect("out 基线对拍");
    allclose("rec_a_state", &got_state, &want_state, 1e-5, 1e-4, false).expect("state 基线对拍");
}

/// ===== 用例 1b-1:极端负 g(-30)——decay≈9.4e-14,旧状态应当被抹到 ≈0 =====
/// 钉死:核内 expf(g) 而非直接乘 g(直接乘 −0.05 类负 g 会把状态符号翻转,
/// 基线与 −30 用例都会红)。容忍度用小 atol:数值本身 ~1e-13 量级。
#[test]
fn rec_gqa_b1_extreme_g_neg30() {
    let rig = Rig::acquire();
    let (batch, nv, nk, kd, vd) = (1usize, 4usize, 2usize, 32usize, 64usize);
    let slot = 1usize;
    let max_batch = 2usize;
    let q = det_f32(5, batch * nk * kd, |i| ((i % 9) as f32 - 4.0) * 0.25);
    let kv = det_f32(6, batch * nk * kd, |i| ((i % 7) as f32 - 3.0) * 0.3);
    let v = det_f32(7, batch * nv * vd, |i| ((i % 11) as f32 - 5.0) * 0.2);
    let g = vec![-30.0f32; batch * nv];
    let beta: Vec<f32> = (0..batch * nv).map(|i| 0.5 + 0.05 * (i % 4) as f32).collect();
    let state0: Vec<f32> = det_f32(8, max_batch * nv * kd * vd, |i| ((i % 13) as f32 - 6.0) * 0.2);
    let q_scale = 1.0 / (kd as f32).sqrt();

    let d_q = rig.htod_f32(&q);
    let d_k = rig.htod_f32(&kv);
    let d_v = rig.htod_f32(&v);
    let d_g = rig.htod_f32(&g);
    let d_beta = rig.htod_f32(&beta);
    let d_state = rig.htod_f32(&state0);
    let d_slots = rig.htod_u32(&[slot as u32]);
    let d_out = rig.out_f32(batch * nv * vd);

    let mut k = kernels(&rig);
    device_step(&rig, &mut k, &d_q, &d_k, &d_v, &d_g, &d_beta, &d_state, &d_slots, &d_out, batch, nv, nk, kd, vd, q_scale, 0);

    let got_out = rig.dtoh_f32(&d_out);
    let got_state = rig.dtoh_f32(&d_state);
    let mut want_state = state0.clone();
    let want_out = host_step(&mut want_state, max_batch, nv, nk, kd, vd, slot, &q, &kv, &v, &g, &beta, q_scale, true);

    // 数值 ~1e-13,atol 收紧到 1e-15 才是真钉子
    allclose("rec_b1_out", &got_out, &want_out, 1e-15, 1e-3, false).expect("out(g=-30) 对拍");
    allclose("rec_b1_state", &got_state, &want_state, 1e-15, 1e-3, false).expect("state(g=-30) 对拍");

    // decay 钉子:旧状态内容被抹掉——对比 host 中「关掉 decay」的变体,
    // 若核未做 expf(g)(直接乘 g=-30 或不衰减),state 必然落在另一侧。
    let mut no_decay = state0.clone();
    let _ = host_step(&mut no_decay, max_batch, nv, nk, kd, vd, slot, &q, &kv, &v, &g, &beta, q_scale, false);
    let d_nodecay: f32 = got_state.iter().zip(&no_decay).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let d_withdecay: f32 = got_state.iter().zip(&want_state).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    eprintln!("[b1 定谳] 与带 expf 参考 max|Δ|={d_withdecay:.3e} 与无 decay 参考 max|Δ|={d_nodecay:.3e}");
    assert!(d_withdecay < d_nodecay * 1e-3, "核未按 expf(g) 衰减:带 decay 偏差 {d_withdecay} 不应远小于无 decay 偏差 {d_nodecay}");
}

/// ===== 用例 1b-2:极端正 g(+5)64 token 链——decay≈148/步,指数增长饱和 =====
/// 钉死:核内 expf(g) 无 clamp;记录设备/参考各自的 Inf 起点,要求一致;
/// Inf 之前的每个 token 与 host 对拍。这是"爆炸点记录"立案证据。
#[test]
fn rec_gqa_b2_extreme_g_pos5_saturation() {
    let rig = Rig::acquire();
    let (batch, nv, nk, kd, vd) = (1usize, 4usize, 2usize, 32usize, 64usize);
    let slot = 1usize;
    let max_batch = 2usize;
    let t_len = 64usize;
    let q: Vec<f32> = det_f32(9, t_len * nk * kd, |i| ((i % 9) as f32 - 4.0) * 0.25);
    let kv: Vec<f32> = det_f32(10, t_len * nk * kd, |i| ((i % 7) as f32 - 3.0) * 0.3);
    let v: Vec<f32> = det_f32(11, t_len * nv * vd, |i| ((i % 11) as f32 - 5.0) * 0.2);
    // 注意:g/beta 必须与 token 数等长(核按 tok 偏移读)
    let g = vec![5.0f32; t_len * nv];
    let beta: Vec<f32> = (0..t_len * nv).map(|i| 0.5 + 0.05 * (i % 4) as f32).collect();
    let state0: Vec<f32> = det_f32(12, max_batch * nv * kd * vd, |i| ((i % 13) as f32 - 6.0) * 0.02);
    let q_scale = 1.0 / (kd as f32).sqrt();

    let d_q = rig.htod_f32(&q);
    let d_k = rig.htod_f32(&kv);
    let d_v = rig.htod_f32(&v);
    let d_g = rig.htod_f32(&g);
    let d_beta = rig.htod_f32(&beta);
    let d_state = rig.htod_f32(&state0);
    let d_slots = rig.htod_u32(&[slot as u32]);
    let d_out = rig.out_f32(nv * vd);

    let mut k = kernels(&rig);
    // host 同步推进(state 拷贝一份)
    let mut h_state = state0.clone();

    let mut dev_inf_tok: Option<usize> = None;
    let mut host_inf_tok: Option<usize> = None;
    let mut last_parity_tok: Option<usize> = None;

    for t in 0..t_len {
        device_step(&rig, &mut k, &d_q, &d_k, &d_v, &d_g, &d_beta, &d_state, &d_slots, &d_out, batch, nv, nk, kd, vd, q_scale, t);
        let q_t = &q[t * nk * kd..(t + 1) * nk * kd];
        let k_t = &kv[t * nk * kd..(t + 1) * nk * kd];
        let v_t = &v[t * nv * vd..(t + 1) * nv * vd];
        let want_out = host_step(&mut h_state, max_batch, nv, nk, kd, vd, slot, q_t, k_t, v_t, &g, &beta, q_scale, true);

        let got_out = rig.dtoh_f32(&d_out);
        let got_state = rig.dtoh_f32(&d_state);

        let dev_bad = got_state.iter().any(|x| !x.is_finite());
        let host_bad = h_state.iter().any(|x| !x.is_finite());
        if dev_inf_tok.is_none() && (dev_bad || got_out.iter().any(|x| !x.is_finite())) {
            dev_inf_tok = Some(t);
        }
        if host_inf_tok.is_none() && host_bad {
            host_inf_tok = Some(t);
        }
        if !dev_bad && !host_bad {
            // 有限区段逐 token 对拍(指数增长下相对容差放宽)
            let max_abs = h_state.iter().fold(0f32, |a, x| a.max(x.abs())).abs().max(1.0);
            allclose(
                &format!("rec_b2_state_t{t}"),
                &got_state,
                &h_state,
                1e-3 * max_abs * 1e-3,
                1e-3,
                false,
            )
            .unwrap_or_else(|e| panic!("t={t} 有限区段失配: {e}"));
            allclose(&format!("rec_b2_out_t{t}"), &got_out, &want_out, 1e-3, 1e-3, false)
                .unwrap_or_else(|e| panic!("t={t} out 失配: {e}"));
            last_parity_tok = Some(t);
        }
    }
    eprintln!(
        "[b2 定谳] dev_inf_tok={dev_inf_tok:?} host_inf_tok={host_inf_tok:?} last_parity_tok={last_parity_tok:?}(decay=exp(5)≈148.4/步,f32 上限 ~3.4e38 → 理论饱和 ~token 17)"
    );
    assert_eq!(
        dev_inf_tok, host_inf_tok,
        "设备/参考 Inf 起点不一致(核内 exp 行为与 expf 公式有偏差)"
    );
}

/// ===== 用例 1c:大 beta(≈1)+ 大 ‖k‖²(>10)——delta 大负值路径 =====
#[test]
fn rec_gqa_c_large_beta_large_k() {
    let rig = Rig::acquire();
    let (batch, nv, nk, kd, vd) = (1usize, 4usize, 2usize, 32usize, 64usize);
    let slot = 1usize;
    let max_batch = 2usize;
    // ‖k‖² ≈ 32×0.6² = 11.5 > 10
    let kv = vec![0.6f32; batch * nk * kd];
    let q = vec![0.5f32; batch * nk * kd];
    let v: Vec<f32> = det_f32(13, batch * nv * vd, |i| ((i % 11) as f32 - 5.0) * 0.2);
    let g: Vec<f32> = (0..batch * nv).map(|i| -0.03 - 0.01 * (i % 3) as f32).collect();
    let beta = vec![0.99f32; batch * nv];
    // 非零初始状态 → kv_mem 大 → delta 大负
    let state0: Vec<f32> = det_f32(14, max_batch * nv * kd * vd, |i| ((i % 17) as f32 - 8.0) * 0.5);
    let q_scale = 1.0 / (kd as f32).sqrt();

    let d_q = rig.htod_f32(&q);
    let d_k = rig.htod_f32(&kv);
    let d_v = rig.htod_f32(&v);
    let d_g = rig.htod_f32(&g);
    let d_beta = rig.htod_f32(&beta);
    let d_state = rig.htod_f32(&state0);
    let d_slots = rig.htod_u32(&[slot as u32]);
    let d_out = rig.out_f32(batch * nv * vd);

    let mut k = kernels(&rig);
    device_step(&rig, &mut k, &d_q, &d_k, &d_v, &d_g, &d_beta, &d_state, &d_slots, &d_out, batch, nv, nk, kd, vd, q_scale, 0);

    let got_out = rig.dtoh_f32(&d_out);
    let got_state = rig.dtoh_f32(&d_state);
    let mut want_state = state0.clone();
    let want_out = host_step(&mut want_state, max_batch, nv, nk, kd, vd, slot, &q, &kv, &v, &g, &beta, q_scale, true);

    allclose("rec_c_out", &got_out, &want_out, 1e-4, 1e-4, false).expect("out(大 beta/大 k) 对拍");
    allclose("rec_c_state", &got_state, &want_state, 1e-4, 1e-4, false).expect("state(大 beta/大 k) 对拍");

    // beta 语义钉子:若核内对 beta 再 sigmoid(0.99→0.731),delta 缩 26%,
    // 状态更新量级必然失配——上面 allclose 红即是证据。此处再显式验证
    // 「beta 直接用」:换 beta=0.99→1.0 重算 host,与 got 差应远小于 sigmoid 化差异。
    let mut alt = state0.clone();
    let alt_out = host_step(&mut alt, max_batch, nv, nk, kd, vd, slot, &q, &kv, &v, &g, &beta, q_scale, true);
    let direct_diff: f32 = got_out.iter().zip(&alt_out).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let mut sig = beta.clone();
    for x in &mut sig {
        *x = 1.0 / (1.0 + (-*x).exp());
    }
    let mut sig_state = state0.clone();
    let sig_out = host_step(&mut sig_state, max_batch, nv, nk, kd, vd, slot, &q, &kv, &v, &g, &sig, q_scale, true);
    let sigmoid_diff: f32 = got_out.iter().zip(&sig_out).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    eprintln!("[c 定谳] 直接用 beta 最大偏差 {direct_diff:.3e} vs sigmoid 化偏差 {sigmoid_diff:.3e}");
    assert!(sigmoid_diff > direct_diff * 10.0, "核 beta 语义无法判定为『直接使用』");
}

/// ===== 用例 2:T=64 温和序列逐步推进,与 host 参考同步对比 =====
#[test]
fn rec_gqa_multi_token_t64_stability() {
    let rig = Rig::acquire();
    let (batch, nv, nk, kd, vd) = (1usize, 4usize, 2usize, 32usize, 64usize);
    let slot = 1usize;
    let max_batch = 2usize;
    let t_len = 64usize;
    let q: Vec<f32> = det_f32(15, t_len * nk * kd, |i| ((i % 9) as f32 - 4.0) * 0.25);
    let kv: Vec<f32> = det_f32(16, t_len * nk * kd, |i| ((i % 7) as f32 - 3.0) * 0.3);
    let v: Vec<f32> = det_f32(17, t_len * nv * vd, |i| ((i % 11) as f32 - 5.0) * 0.2);
    let g: Vec<f32> = (0..t_len * nv).map(|i| -0.1 + -0.001 * (i % 90) as f32).collect();
    let beta: Vec<f32> = (0..t_len * nv).map(|i| 0.4 + 0.01 * (i % 50) as f32).collect();
    let state0 = vec![0f32; max_batch * nv * kd * vd];
    let q_scale = 1.0 / (kd as f32).sqrt();

    let d_q = rig.htod_f32(&q);
    let d_k = rig.htod_f32(&kv);
    let d_v = rig.htod_f32(&v);
    let d_g = rig.htod_f32(&g);
    let d_beta = rig.htod_f32(&beta);
    let d_state = rig.htod_f32(&state0);
    let d_slots = rig.htod_u32(&[slot as u32]);
    let d_out = rig.out_f32(nv * vd);

    let mut k = kernels(&rig);
    let mut h_state = state0.clone();
    let mut max_out_mag = 0f32;
    let mut max_state_mag = 0f32;

    for t in 0..t_len {
        device_step(&rig, &mut k, &d_q, &d_k, &d_v, &d_g, &d_beta, &d_state, &d_slots, &d_out, batch, nv, nk, kd, vd, q_scale, t);
        let q_t = &q[t * nk * kd..(t + 1) * nk * kd];
        let k_t = &kv[t * nk * kd..(t + 1) * nk * kd];
        let v_t = &v[t * nv * vd..(t + 1) * nv * vd];
        let g_t = &g[t * nv..(t + 1) * nv];
        let b_t = &beta[t * nv..(t + 1) * nv];
        let want_out = host_step(&mut h_state, max_batch, nv, nk, kd, vd, slot, q_t, k_t, v_t, g_t, b_t, q_scale, true);
        let got_out = rig.dtoh_f32(&d_out);
        let got_state = rig.dtoh_f32(&d_state);
        allclose(&format!("mt_out_t{t}"), &got_out, &want_out, 1e-5, 1e-4, false)
            .unwrap_or_else(|e| panic!("t={t} out 失配(开始发散点): {e}"));
        allclose(&format!("mt_state_t{t}"), &got_state, &h_state, 1e-5, 1e-4, false)
            .unwrap_or_else(|e| panic!("t={t} state 失配(开始发散点): {e}"));
        max_out_mag = max_out_mag.max(got_out.iter().fold(0f32, |a, x| a.max(x.abs())));
        max_state_mag = max_state_mag.max(got_state.iter().fold(0f32, |a, x| a.max(x.abs())));
    }
    eprintln!("[t64 定谳] 全程 64 token 对拍全绿;max|out|={max_out_mag:.4} max|state|={max_state_mag:.4}(温和分布下有界)");
    assert!(max_state_mag.is_finite() && max_state_mag < 1e6, "温和分布下状态发散");
}

/// ===== 用例 3:约定交叉互证——同 g 数值喂两个核,钉死各自空间 =====
/// fallback 核约定「g 已 exp(实空间 decay,直接乘)」;gqa 核「g 为 log 空间,
/// 核内 expf」。同一初始状态:取 log_g=-0.05 → decay=exp(-0.05)≈0.95123。
/// fallback 吃 decay、gqa 吃 log_g → 两核终态应一致;任何一方约定翻转即红。
#[test]
fn rec_convention_cross_kernel_pin() {
    let rig = Rig::acquire();
    let (nv, kd, vd) = (4usize, 32usize, 64usize);
    let (nk, batch) = (2usize, 1usize);
    let log_g = -0.05f32;
    let decay = log_g.exp();
    let beta = 0.7f32;
    let q_scale = 1.0 / (kd as f32).sqrt();

    let q: Vec<f32> = det_f32(21, nk * kd, |i| ((i % 9) as f32 - 4.0) * 0.25);
    let kv: Vec<f32> = det_f32(22, nk * kd, |i| ((i % 7) as f32 - 3.0) * 0.3);
    let v: Vec<f32> = det_f32(23, nv * vd, |i| ((i % 11) as f32 - 5.0) * 0.2);
    let state0: Vec<f32> = det_f32(24, nv * kd * vd, |i| ((i % 13) as f32 - 6.0) * 0.2);

    // gqa 路:state [1(batch slot), nv, kd, vd]
    let d_q = rig.htod_f32(&q);
    let d_k = rig.htod_f32(&kv);
    let d_v = rig.htod_f32(&v);
    let d_g = rig.htod_f32(&vec![log_g; nv]);
    let d_beta = rig.htod_f32(&vec![beta; nv]);
    let d_state_gqa = rig.htod_f32(&state0);
    let d_slots = rig.htod_u32(&[0]);
    let d_out = rig.out_f32(nv * vd);
    let mut k = kernels(&rig);
    device_step(&rig, &mut k, &d_q, &d_k, &d_v, &d_g, &d_beta, &d_state_gqa, &d_slots, &d_out, batch, nv, nk, kd, vd, q_scale, 0);
    let gqa_state = rig.dtoh_f32(&d_state_gqa);

    // fallback 路:state [BH=1, kd, vd](bh 维 = batch*nv 逐头;此处 nv 头全同参)
    // fallback 布局 q/k [BH,S,K],v/out [BH,S,V],state [BH,K,V];S=1
    let d_q_fb = rig.htod_f32(&q); // [1,1,kd]? 不对:BH=nv → 每 bh 一份 q
    let q_fb: Vec<f32> = (0..nv).flat_map(|_| q.iter().copied()).collect();
    let k_fb: Vec<f32> = (0..nv).flat_map(|_| kv.iter().copied()).collect();
    let v_fb: Vec<f32> = (0..nv).flat_map(|vi| v[vi * vd..(vi + 1) * vd].iter().copied()).collect();
    let _ = d_q_fb;
    let d_q_fb = rig.htod_f32(&q_fb);
    let d_k_fb = rig.htod_f32(&k_fb);
    let d_v_fb = rig.htod_f32(&v_fb);
    let d_g_fb = rig.htod_f32(&vec![decay; nv]); // 实空间 decay
    let d_beta_fb = rig.htod_f32(&vec![beta; nv]);
    // state [BH=nv, kd, vd] → 每头一份初始态(与 gqa 的 per-(slot=0,vh) 相同)
    let st_fb: Vec<f32> = (0..nv)
        .flat_map(|vh| state0[vh * kd * vd..(vh + 1) * kd * vd].iter().copied())
        .collect();
    let d_state_fb = rig.htod_f32(&st_fb);
    let d_out_fb = rig.out_f32(nv * vd);
    k.delta_recurrence_fallback(
        rig.stream(),
        "f32",
        d_q_fb.u8_ptr(),
        d_k_fb.u8_ptr(),
        d_v_fb.u8_ptr(),
        d_g_fb.f32_ptr(),
        d_beta_fb.f32_ptr(),
        d_state_fb.f32_ptr_mut(),
        d_out_fb.f32_ptr_mut(),
        nv as i32,
        1,
        kd as i32,
        vd as i32,
    )
    .expect("fallback launch");
    rig.sync();
    let fb_state = rig.dtoh_f32(&d_state_fb);

    // 逐头对比(gqa 的 (slot=0,vh) 切片 vs fallback 的 bh=vh)
    for vh in 0..nv {
        let a = &gqa_state[vh * kd * vd..(vh + 1) * kd * vd];
        let b = &fb_state[vh * kd * vd..(vh + 1) * kd * vd];
        allclose(&format!("cross_state_vh{vh}"), a, b, 1e-6, 1e-4, false)
            .unwrap_or_else(|e| panic!("vh={vh} 交叉互证失配: {e}"));
    }
    eprintln!("[convention 定谳] fallback(g=decay) 与 gqa(g=log,核内 expf) 同参同态同果——两核空间约定互证钉死");
}

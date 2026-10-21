//! N1 探针:`owl_gdn_conv_fwd_f16` **t 维并行改写**的独立验证
//! (2026-10-14;施工背景见 `docs/arch/prefill-graph-施工.md` §12.4b)。
//!
//! 判据(任一失败即改写不成立):
//! 1. **数值**:与 host 参考(f32 逐步同序:先 `x_t*w3`,再 `w0/w1/w2` 逐个
//!    `fmaf_rn`;末写 f16 舍入)**逐位一致**(`silu=0` 档 —— 无超越函数,
//!    无容差余地);`silu=1` 档用容差(`__expf` 是 SFU 近似,host 无法逐位复刻)。
//! 2. **状态律**:`conv_state` = 本序列**末 3 个输入**(不是 conv 输出);
//!    多 chunk 串联(同槽连跑)时后 chunk 用前 chunk 写回的 state,结果与
//!    "整段一次跑完"逐位一致 —— 这是 prefill 分块的真实语义。
//! 3. **边界**:`seq_len < 3`(state 退化为移位寄存器)、`seq_len` 小于/大于
//!    `TILE_ROWS=32`(tiles=1 vs 多 tile)、`d` 跨多个 256 通道块、`batch>1`。
//! 4. **并行正确性**:`tiles>1` 时必须只有 tile 0 读写 state
//!    (t>=start+3 的行纯滑窗)—— 由判据 2 的多 chunk 对拍覆盖。
//!
//! 运行纪律:`OWL_TEST_DEVICE` 钉卡(缺省跳过)+ `--test-threads=1`。

use cudarc::driver::{CudaContext, CudaFunction, DevicePtr, LaunchConfig, PushKernelArg};
use half::f16;

const BLOCK: u32 = 256;
/// picker 侧的网格提示(`driver::gdn::TILE_ROWS`);核内自算 ts
const TILE_ROWS: usize = 32;

fn dev_ordinal() -> Option<usize> {
    owl_cuda::test_device_ordinal_opt()
}

fn conv_asset() -> &'static owl_kernels::family::cubins::CubinAsset {
    let src = owl_kernels::spec::meta_of::<owl_kernels::ops::gdn::ConvFwdF16>().src;
    owl_kernels::family::cubins::asset(src).expect("cu/gdn/gdn_conv_fwd.cu 应有构建期资产")
}

/// 确定性伪随机(与 gdn_big_t_arms_parity 同族,便于跨卷对读)
fn lcg01(seq: u32, i: usize) -> f32 {
    let x = seq
        .wrapping_mul(0x9E37_79B9)
        .wrapping_add((i as u32).wrapping_mul(0x85EB_CA6B));
    ((x >> 8) as f32) / 16777216.0
}

fn bits(v: &[f16]) -> Vec<u16> {
    v.iter().map(|x| x.to_bits()).collect()
}

// ---------------------------------------------------------------------------
// host 参考(与核内逐式同序)
// ---------------------------------------------------------------------------

/// 行 `t` 的第 `i` 个历史输入(hist[0] = 最老 = x[t-3]);pos<start 读旧 state。
/// `st_chan` = **本通道**的 3 槽(内核里是 `conv_state + (slot*d+chan)*3` 指针;
/// 传整块 [d,3] 会永远读到通道 0 —— 2026-10-14 探针首跑的假红就是这个)。
#[allow(clippy::too_many_arguments)]
fn hist_at(
    x: &[f16],
    st_chan: &[f32],
    d: usize,
    chan: usize,
    start: usize,
    t: isize,
    i: usize,
) -> f32 {
    let pos = t - 3 + i as isize;
    if pos >= start as isize {
        x[pos as usize * d + chan].to_f32()
    } else if !st_chan.is_empty() {
        st_chan[(pos - start as isize + 3) as usize]
    } else {
        0.0
    }
}

/// 单序列 host 参考:`x [len, d]`、`w [d, 4]`、`st [d, 3]`(原地更新)、
/// `silu` 门;返回 `out [len, d]`
fn host_ref(x: &[f16], w: &[f16], st: &mut [f32], len: usize, d: usize, silu: bool) -> Vec<f16> {
    let mut out = vec![f16::from_f32(0.0); len * d];
    for chan in 0..d {
        let wv = [
            w[chan * 4].to_f32(),
            w[chan * 4 + 1].to_f32(),
            w[chan * 4 + 2].to_f32(),
            w[chan * 4 + 3].to_f32(),
        ];
        let mut h = [0.0f32; 3];
        {
            let st_c = if st.is_empty() { &st[0..0] } else { &st[chan * 3..chan * 3 + 3] };
            for i in 0..3 {
                h[i] = hist_at(x, st_c, d, chan, 0, 0, i);
            }
        }
        for t in 0..len {
            let x_t = x[t * d + chan].to_f32();
            let mut sum = x_t * wv[3];
            for k in 0..3 {
                sum = h[k].mul_add(wv[k], sum);
            }
            if silu {
                sum /= 1.0 + (-sum).exp();
            }
            out[t * d + chan] = f16::from_f32(sum);
            h[0] = h[1];
            h[1] = h[2];
            h[2] = x_t;
        }
        // 新 state = 末 3 个输入(seq_len<3 时退化为移位寄存器)
        if !st.is_empty() {
            let old = [st[chan * 3], st[chan * 3 + 1], st[chan * 3 + 2]];
            let mut new = [0.0f32; 3];
            for i in 0..3 {
                new[i] = hist_at(x, &old, d, chan, 0, len as isize, i);
            }
            st[chan * 3..chan * 3 + 3].copy_from_slice(&new);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// 发射
// ---------------------------------------------------------------------------

fn tiles_for(len: usize) -> u32 {
    len.div_ceil(TILE_ROWS).max(1) as u32
}

/// 单次发射:同槽、单 batch(生产 prefill 形)。
/// `st` 传入 = 旧 state,传出 = 核写回的新 state。
#[allow(clippy::too_many_arguments)]
fn run_conv(
    ctx: &std::sync::Arc<CudaContext>,
    f: &CudaFunction,
    x: &[f16],
    w: &[f16],
    st: &[f32],
    slots: &[f32],
    cu: &[f32],
    batch: usize,
    len: usize,
    d: usize,
    silu: bool,
) -> (Vec<f16>, Vec<f32>) {
    let stream = ctx.default_stream();
    let dx = stream.clone_htod(&bits(x)).expect("htod x");
    let dw = stream.clone_htod(&bits(w)).expect("htod w");
    let dst = stream.clone_htod(st).expect("htod st");
    let dslots = stream.clone_htod(slots).expect("htod slots");
    let dcu = stream.clone_htod(cu).expect("htod cu");
    let dout = stream.alloc_zeros::<u16>(len * d).expect("alloc out");
    let (px, _) = dx.device_ptr(&stream);
    let (pw, _) = dw.device_ptr(&stream);
    let (pst, _) = dst.device_ptr(&stream);
    let (psl, _) = dslots.device_ptr(&stream);
    let (pcu, _) = dcu.device_ptr(&stream);
    let (po, _) = dout.device_ptr(&stream);
    let (b_i, d_i, s_i) = (batch as i32, d as i32, i32::from(silu));
    // SAFETY:参数槽序与 `owl_gdn_conv_fwd_f16` 实例行签名严格对位
    unsafe {
        stream
            .launch_builder(f)
            .arg(&px)
            .arg(&pw)
            .arg(&pst)
            .arg(&psl)
            .arg(&pcu)
            .arg(&b_i)
            .arg(&d_i)
            .arg(&s_i)
            .arg(&po)
            .launch(LaunchConfig {
                grid_dim: ((d as u32).div_ceil(BLOCK), tiles_for(len), batch as u32),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
            .expect("launch");
    }
    stream.synchronize().expect("sync");
    let out: Vec<u16> = stream.clone_dtoh(&dout).expect("dtoh out");
    let st_out: Vec<f32> = stream.clone_dtoh(&dst).expect("dtoh st");
    (out.into_iter().map(f16::from_bits).collect(), st_out)
}

fn gen(n: usize, seed: u32) -> Vec<f16> {
    (0..n)
        .map(|i| f16::from_f32((lcg01(seed, i) - 0.5) * 0.4))
        .collect()
}

/// f16 输出对拍:`exact=true` 逐位,否则 |Δ|<=1e-3(SFU `__expf` 档)
fn check_bits(tag: &str, got: &[f16], want: &[f16], exact: bool) {
    assert_eq!(got.len(), want.len(), "{tag}: 长度不符");
    let mut bad = 0usize;
    let mut first: Option<(usize, f32, f32)> = None;
    let mut worst = 0.0f32;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let (gf, wf) = (g.to_f32(), w.to_f32());
        let dev = (gf - wf).abs();
        worst = worst.max(dev);
        let miss = if exact { g != w } else { dev > 1e-3 };
        if miss {
            bad += 1;
            if first.is_none() {
                first = Some((i, gf, wf));
            }
        }
    }
    assert_eq!(
        bad, 0,
        "{tag}:{bad}/{} 处不符(首个 {first:?};worst={worst:.3e};exact={exact})",
        got.len()
    );
}

/// f32 state 对拍(逐位)
fn check_f32(tag: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{tag}: 长度不符");
    let mut bad = 0usize;
    let mut first: Option<(usize, f32, f32)> = None;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        if g.to_bits() != w.to_bits() {
            bad += 1;
            if first.is_none() {
                first = Some((i, *g, *w));
            }
        }
    }
    assert_eq!(bad, 0, "{tag}:{bad}/{} 处位型不符(首个 {first:?})", got.len());
}

// ---------------------------------------------------------------------------
// 判据 1+3:单 chunk 数值 + 边界形状
// ---------------------------------------------------------------------------

#[test]
fn conv_fwd_single_chunk_matches_host_bitwise() {
    let Some(ord) = dev_ordinal() else {
        owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
        return;
    };
    let ctx = CudaContext::new(ord).expect("context");
    let asset = conv_asset();
    let module = ctx
        .load_module(cudarc::nvrtc::Ptx::from_binary(asset.bytes.to_vec()))
        .expect("fatbin 载入");
    let f = module.load_function("owl_gdn_conv_fwd_f16").expect("load fn");

    // (len, d, 初态非零?) —— 覆盖 tiles=1 / 多 tile / len<3 / 多通道块
    let cases: &[(usize, usize, bool)] = &[
        (1, 512, true),
        (2, 512, true),   // len<3:state 移位寄存器
        (3, 512, true),   // 恰好=3:state = x[0..3]
        (31, 512, false), // tiles=1
        (32, 1024, false),
        (33, 1024, true), // tiles=2 的第一例(跨块)
        (461, 512, true), // 生产 chunk 形,15 tiles
        (461, 2048, false),
        (1000, 256, true),
    ];
    for &(len, d, nonzero_st) in cases {
        let x = gen(len * d, 7 + len as u32);
        let w = gen(d * 4, 11);
        let mut st0 = vec![0.0f32; d * 3];
        if nonzero_st {
            for (i, v) in st0.iter_mut().enumerate() {
                *v = (lcg01(13, i) - 0.5) * 0.3;
            }
        }
        let slots = vec![0.0f32];
        let cu = vec![0.0f32, len as f32];

        // silu=0:逐位
        let mut want_st = st0.clone();
        let want = host_ref(&x, &w, &mut want_st, len, d, false);
        let (got, got_st) = run_conv(&ctx, &f, &x, &w, &st0, &slots, &cu, 1, len, d, false);
        check_bits(&format!("len={len} d={d} silu=0 out"), &got, &want, true);
        check_f32(&format!("len={len} d={d} silu=0 state"), &got_st, &want_st);

        // silu=1:容差(__expf 近似)
        let mut want_st1 = st0.clone();
        let want1 = host_ref(&x, &w, &mut want_st1, len, d, true);
        let (got1, got_st1) = run_conv(&ctx, &f, &x, &w, &st0, &slots, &cu, 1, len, d, true);
        check_bits(&format!("len={len} d={d} silu=1 out"), &got1, &want1, false);
        check_f32(&format!("len={len} d={d} silu=1 state"), &got_st1, &want_st1);
    }
    eprintln!("N1 判据 1+3 绿:{} 个形状单 chunk 逐位一致", cases.len());
}

// ---------------------------------------------------------------------------
// 判据 2:多 chunk 串联(state 携带)vs 整段一次跑
// ---------------------------------------------------------------------------

#[test]
fn conv_fwd_chunked_state_matches_whole_run() {
    let Some(ord) = dev_ordinal() else {
        owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
        return;
    };
    let ctx = CudaContext::new(ord).expect("context");
    let asset = conv_asset();
    let module = ctx
        .load_module(cudarc::nvrtc::Ptx::from_binary(asset.bytes.to_vec()))
        .expect("fatbin 载入");
    let f = module.load_function("owl_gdn_conv_fwd_f16").expect("load fn");

    let (d, total) = (512usize, 991usize);
    let x = gen(total * d, 21);
    let w = gen(d * 4, 23);
    let st0 = vec![0.0f32; d * 3];
    let slots = vec![0.0f32];

    // 参照:整段一次(len=991 ⇒ tiles=31,单序列)
    let mut want_st = st0.clone();
    let want = host_ref(&x, &w, &mut want_st, total, d, true);

    // 分块:461 + 461 + 69(与生产的 512 级 chunk 同数量级),state 逐块携带
    let chunks = [461usize, 461, 69];
    let mut st = st0.clone();
    let mut base = 0usize;
    for (ci, &len) in chunks.iter().enumerate() {
        // 本 chunk 的 x 切片按 [len, d] 连续给出;cu_seqlens 相对本 chunk 起点
        let xs: Vec<f16> = x[base * d..(base + len) * d].to_vec();
        let cu = vec![0.0f32, len as f32];
        let (got, st_new) =
            run_conv(&ctx, &f, &xs, &w, &st, &slots, &cu, 1, len, d, true);
        let seg = &want[base * d..(base + len) * d];
        check_bits(&format!("chunk{ci} len={len} out"), &got, seg, false);
        st = st_new;
        base += len;
    }
    check_f32("chunked 串联终态", &st, &want_st);
    eprintln!("N1 判据 2 绿:3 段串联(461+461+69)与整段 991 一致,终态一致");
}

// ---------------------------------------------------------------------------
// 判据 3b:batch>1(z 维)+ 负槽(无 state)
// ---------------------------------------------------------------------------

#[test]
fn conv_fwd_batch_and_neg_slot() {
    let Some(ord) = dev_ordinal() else {
        owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
        return;
    };
    let ctx = CudaContext::new(ord).expect("context");
    let asset = conv_asset();
    let module = ctx
        .load_module(cudarc::nvrtc::Ptx::from_binary(asset.bytes.to_vec()))
        .expect("fatbin 载入");
    let f = module.load_function("owl_gdn_conv_fwd_f16").expect("load fn");

    let (d, lens) = (256usize, [70usize, 5, 40]);
    let total: usize = lens.iter().sum();
    let x = gen(total * d, 31);
    let w = gen(d * 4, 33);
    // 序列槽 = [0, 负(全跳), 1];**槽 2 无人使用** ⇒ 它必须一字未动
    // (负槽序列本身没有 state 区域 —— 早先把"槽 1 被写"当成违规是测试自己的槽映射错)
    let slots = vec![0.0f32, -1.0, 1.0];
    let mut st = vec![0.0f32; 3 * d * 3];
    for (i, v) in st.iter_mut().enumerate() {
        *v = (lcg01(37, i) - 0.5) * 0.2;
    }
    // cu_seqlens = [0, 70, 75, 115]
    let mut cu = vec![0.0f32];
    let mut acc = 0usize;
    for &l in &lens {
        acc += l;
        cu.push(acc as f32);
    }

    let (got, got_st) = run_conv(
        &ctx, &f, &x, &w, &st, &slots, &cu, 3, total, d, false,
    );

    // host 参考:逐序列(注意 x 是 varlen 拼接,序列 s 的起始行 = cu[s])
    let mut want_st = st.clone();
    let mut want = vec![f16::from_f32(0.0); total * d];
    for (s, &len) in lens.iter().enumerate() {
        let start = cu[s] as usize;
        let seg = &x[start * d..(start + len) * d];
        let slot = if slots[s] >= 0.0 { slots[s] as usize } else { usize::MAX };
        let mut local_st = if slot == usize::MAX {
            vec![]
        } else {
            want_st[slot * d * 3..(slot + 1) * d * 3].to_vec()
        };
        let _ = &local_st;
        let seg_out = host_ref(seg, &w, &mut local_st, len, d, false);
        want[start * d..(start + len) * d].copy_from_slice(&seg_out);
        if slot != usize::MAX {
            want_st[slot * d * 3..(slot + 1) * d * 3].copy_from_slice(&local_st);
        }
    }
    check_bits("batch=3 out", &got, &want, true);
    check_f32("batch=3 state", &got_st, &want_st);
    // 无人使用的槽 2 必须一字未动(负槽序列 = 无 state 区域,不受影响)
    assert_eq!(
        &got_st[2 * d * 3..3 * d * 3],
        &st[2 * d * 3..3 * d * 3],
        "未用槽不得被写(全跳序列 slot<0 ⇒ 无 state 副作用)"
    );
    eprintln!("N1 判据 3b 绿:batch=3 + 负槽(state 不写)逐位一致");
}

// ---------------------------------------------------------------------------
// 判据 5:延迟(N1 的全部动机)—— 生产形 ×200 发取均值
// (旧版:q/k 各 8 CTA、v 24 CTA,核内对 t 顺序扫描 ⇒ nsys 实测 100/100/170 µs)
// ---------------------------------------------------------------------------

#[test]
fn conv_fwd_latency_probe() {
    let Some(ord) = dev_ordinal() else {
        owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
        return;
    };
    let ctx = CudaContext::new(ord).expect("context");
    let asset = conv_asset();
    let module = ctx
        .load_module(cudarc::nvrtc::Ptx::from_binary(asset.bytes.to_vec()))
        .expect("fatbin 载入");
    let f = module.load_function("owl_gdn_conv_fwd_f16").expect("load fn");

    const ITERS: usize = 200;
    let (d, len) = (6144usize, 461usize);
    let x = gen(len * d, 41);
    let w = gen(d * 4, 43);
    let st = vec![0.0f32; d * 3];
    let slots = vec![0.0f32];
    let cu = vec![0.0f32, len as f32];

    let stream = ctx.default_stream();
    let dx = stream.clone_htod(&bits(&x)).expect("htod x");
    let dw = stream.clone_htod(&bits(&w)).expect("htod w");
    let dst = stream.clone_htod(&st).expect("htod st");
    let dslots = stream.clone_htod(&slots).expect("htod slots");
    let dcu = stream.clone_htod(&cu).expect("htod cu");
    let dout = stream.alloc_zeros::<u16>(len * d).expect("alloc out");
    let (px, _) = dx.device_ptr(&stream);
    let (pw, _) = dw.device_ptr(&stream);
    let (pst, _) = dst.device_ptr(&stream);
    let (psl, _) = dslots.device_ptr(&stream);
    let (pcu, _) = dcu.device_ptr(&stream);
    let (po, _) = dout.device_ptr(&stream);
    let (b_i, d_i) = (1i32, d as i32);
    let s_i = 1i32; // 生产:silu=true
    let cfg = LaunchConfig {
        grid_dim: ((d as u32).div_ceil(BLOCK), tiles_for(len), 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let fire = || unsafe {
        stream
            .launch_builder(&f)
            .arg(&px)
            .arg(&pw)
            .arg(&pst)
            .arg(&psl)
            .arg(&pcu)
            .arg(&b_i)
            .arg(&d_i)
            .arg(&s_i)
            .arg(&po)
            .launch(cfg)
            .expect("launch");
    };
    // 暖机(模块/指令缓存)
    for _ in 0..10 {
        fire();
    }
    stream.synchronize().expect("sync warmup");
    let t0 = std::time::Instant::now();
    for _ in 0..ITERS {
        fire();
    }
    stream.synchronize().expect("sync");
    let us = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;
    eprintln!(
        "N1 判据 5: v=6144 x t={} x {} 发,均值 ~{} us/发(旧版 nsys 实测 ~170 us;含下发税)",
        len, ITERS, us as u64
    );
    assert!(us < 170.0, "改写后反而更慢: ~{} us/发", us as u64);
}

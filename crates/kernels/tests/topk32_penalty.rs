//! `owl_topk32_split_f16` 设备端重复惩罚门(2026-10-26)。
//!
//! 语义(交接文档 §3):先按全局 token id 命中 history,做 llama.cpp 同式
//! penalty(正除负乘),**再**做 top-32。这样被惩罚的重复 token 掉出前 32 时,
//! 原第 33 名能补位;绝不是"先 top-32 再惩罚"。
//!
//! 覆盖:
//! 1. rep=1.0 ⇒ 与旧实现/无惩罚 host 参考逐位一致;
//! 2. rep>1.0 且 hist 命中 ⇒ 值与索引都逐位对 host 参考;
//! 3. 第 33 名补位(惩罚把原第 1 名踢出 top-32);
//! 4. rank1 `vocab_offset` 命中/不命中;
//! 5. 空 history / 全哨兵。
//!
//! 运行:`OWL_TEST_DEVICE=2 cargo test --release -p owl-kernels \
//!        --test topk32_penalty -- --test-threads=1 --nocapture`

use owl_macros::owl_test;
use std::sync::Arc;

use cudarc::driver::{CudaContext, DevicePtr, LaunchConfig, PushKernelArg};

const ROWS: usize = 3;
const VOCAB: usize = 4096;
const NPARTS: usize = 4;
const K: usize = 32;
const H: usize = 256;

fn f16_bits(v: f32) -> u16 {
    half::f16::from_f32(v).to_bits()
}

fn f16_to_f32(b: u16) -> f32 {
    half::f16::from_bits(b).to_f32()
}

fn history(valid: &[u32]) -> Vec<u32> {
    let mut h = valid.to_vec();
    h.sort_unstable();
    h.dedup();
    h.resize(H, u32::MAX);
    h
}

/// host 参考 = penalty → (值降,局部索引升) top-32。
fn host_topk(
    x: &[u16],
    rows: usize,
    vocab: usize,
    hist: &[u32],
    rep: f32,
    offset: usize,
) -> (Vec<f32>, Vec<u32>) {
    let mut out_v = Vec::with_capacity(rows * K);
    let mut out_i = Vec::with_capacity(rows * K);
    for r in 0..rows {
        let mut cand: Vec<(f32, u32)> = (0..vocab)
            .map(|i| {
                let mut v = f16_to_f32(x[r * vocab + i]);
                if rep > 1.0 {
                    let gid = offset as u32 + i as u32;
                    if hist.binary_search(&gid).is_ok() {
                        v = if v > 0.0 { v / rep } else { v * rep };
                    }
                }
                (v, i as u32)
            })
            .collect();
        cand.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.1.cmp(&b.1))
        });
        for &(v, i) in cand.iter().take(K) {
            out_v.push(v);
            out_i.push(i);
        }
    }
    (out_v, out_i)
}

fn dev_index() -> usize {
    owl_shared::env_reader::str("OWL_TEST_DEVICE")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(2)
}

fn make_random(seed: u32) -> Vec<u16> {
    let mut st = seed;
    (0..ROWS * VOCAB)
        .map(|_| {
            st = st.wrapping_mul(1664525).wrapping_add(1013904223);
            f16_bits(((st >> 8) as f32 / 16777215.0 - 0.5) * 16.0)
        })
        .collect()
}

/// 惩罚把 token 5 从第 1 名踢出前 32;旧 top-32 之外的第一名 token 37
/// 应补进来(38 仍是第 33 名,不补)。
fn promotion_case() -> Vec<u16> {
    let mut x = vec![f16_bits(-100.0); ROWS * VOCAB];
    for r in 0..ROWS {
        x[r * VOCAB + 5] = f16_bits(10.0);
        for t in 6..=38 {
            x[r * VOCAB + t] = f16_bits(9.0);
        }
    }
    x
}

fn run_case(
    ctx: &Arc<CudaContext>,
    f_split: &cudarc::driver::CudaFunction,
    f_merge: &cudarc::driver::CudaFunction,
    x: &[u16],
    hist: &[u32],
    rep: f32,
    offset: usize,
    label: &str,
) {
    let stream = ctx.default_stream();
    let (want_v, want_i) = host_topk(x, ROWS, VOCAB, hist, rep, offset);

    let mut d_x = unsafe { stream.alloc::<u16>(x.len()) }.unwrap();
    stream.memcpy_htod(x, &mut d_x).unwrap();
    let mut d_hist = unsafe { stream.alloc::<u32>(hist.len()) }.unwrap();
    stream.memcpy_htod(hist, &mut d_hist).unwrap();
    let mut d_rep = unsafe { stream.alloc::<f32>(1) }.unwrap();
    stream.memcpy_htod(&[rep], &mut d_rep).unwrap();
    let d_tmp = unsafe { stream.alloc::<f32>(2 * ROWS * NPARTS * K) }.unwrap();
    let d_out = unsafe { stream.alloc::<f32>(2 * ROWS * K) }.unwrap();

    let (xp, hp, rp, tp, op) = (
        d_x.device_ptr(&stream).0,
        d_hist.device_ptr(&stream).0,
        d_rep.device_ptr(&stream).0,
        d_tmp.device_ptr(&stream).0,
        d_out.device_ptr(&stream).0,
    );
    let (vocab_i, nparts_i, rows_i, hcap_i, off_i) = (
        VOCAB as i32,
        NPARTS as i32,
        ROWS as i32,
        H as i32,
        offset as i32,
    );

    let cfg_split = LaunchConfig {
        grid_dim: ((ROWS * NPARTS) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b = stream.launch_builder(f_split);
    b.arg(&xp)
        .arg(&hp)
        .arg(&rp)
        .arg(&vocab_i)
        .arg(&nparts_i)
        .arg(&rows_i)
        .arg(&hcap_i)
        .arg(&off_i)
        .arg(&tp);
    unsafe { b.launch(cfg_split) }.expect("split launch");

    let cfg_merge = LaunchConfig {
        grid_dim: (ROWS as u32, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b = stream.launch_builder(f_merge);
    b.arg(&tp)
        .arg(&nparts_i)
        .arg(&rows_i)
        .arg(&vocab_i)
        .arg(&op);
    unsafe { b.launch(cfg_merge) }.expect("merge launch");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; 2 * ROWS * K];
    stream.memcpy_dtoh(&d_out, &mut got).unwrap();
    let got_v = &got[..ROWS * K];
    let got_i: Vec<u32> = got[ROWS * K..].iter().map(|&v| v as u32).collect();

    let v_ok = got_v
        .iter()
        .zip(want_v.iter())
        .all(|(a, b)| a.to_bits() == b.to_bits());
    let i_ok = got_i == want_i;
    println!(
        "{label}: rep={rep} offset={offset} 值逐位 {} / id 逐位 {}",
        if v_ok { "✓" } else { "✗" },
        if i_ok { "✓" } else { "✗" }
    );
    if !v_ok || !i_ok {
        for r in 0..ROWS {
            println!("  row {r} got_v={:?}", &got_v[r * K..(r + 1) * K]);
            println!("  row {r} want_v={:?}", &want_v[r * K..(r + 1) * K]);
            println!("  row {r} got_i={:?}", &got_i[r * K..(r + 1) * K]);
            println!("  row {r} want_i={:?}", &want_i[r * K..(r + 1) * K]);
        }
    }
    assert!(v_ok, "{label}: 候选值不是逐位相同");
    assert!(i_ok, "{label}: 候选 id 不是逐位相同");
}

#[owl_test(gpu)]
fn topk32_device_rep_penalty_matches_host() {
    let ctx = CudaContext::new(dev_index()).expect("ctx");
    ctx.bind_to_thread().expect("bind");
    let src = owl_kernels::cu_sig::cu_text(owl_kernels::ops::source("owl_topk32_split_f16"));
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(
        src.as_str(),
        cudarc::nvrtc::CompileOptions {
            arch: Some("compute_86"),
            include_paths: vec!["/usr/local/cuda/include".into()],
            ..Default::default()
        },
    )
    .expect("nvrtc compile");
    let module = ctx.load_module(ptx).expect("module");
    let f_split = module
        .load_function("owl_topk32_split_f16")
        .expect("split fn");
    let f_merge = module
        .load_function("owl_topk32_merge_f16")
        .expect("merge fn");

    let random = make_random(7);
    // 1) 关惩罚:hist 里即使有 id 也不得改动,必须与旧 topk32 逐位一致。
    run_case(
        &ctx,
        &f_split,
        &f_merge,
        &random,
        &history(&[5, 9, 33]),
        1.0,
        0,
        "rep-off",
    );
    // 2) 空 history / 全哨兵:rep 开了但无命中。
    run_case(
        &ctx,
        &f_split,
        &f_merge,
        &random,
        &history(&[]),
        1.5,
        0,
        "empty-history",
    );
    // 3) rank0 命中若干真实 id。
    run_case(
        &ctx,
        &f_split,
        &f_merge,
        &random,
        &history(&[5, 9, 33, 100, 4095]),
        1.5,
        0,
        "rank0-hit",
    );
    // 4) rank1:hist 是全局 id,local 5 对应 4096+5;局部 id 5 本身不应命中。
    run_case(
        &ctx,
        &f_split,
        &f_merge,
        &random,
        &history(&[4096 + 5, 4096 + 9, 4096 + 33]),
        1.5,
        VOCAB,
        "rank1-offset",
    );
    // 5) 第 33 名补位。
    let promo = promotion_case();
    let h = history(&[5]);
    let (want_v, want_i) = host_topk(&promo, ROWS, VOCAB, &h, 1.5, 0);
    assert!(!want_i.contains(&5), "参考: token 5 已被罚出 top-32");
    assert!(want_i.contains(&37), "参考: 旧 top-32 之外的第一名(37)补位");
    assert!(!want_i.contains(&38), "参考: 下一名(38)仍不补位");
    run_case(
        &ctx,
        &f_split,
        &f_merge,
        &promo,
        &h,
        1.5,
        0,
        "33rd-promotion",
    );
    let _ = want_v;
}

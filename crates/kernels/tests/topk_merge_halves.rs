//! `owl_topk_merge_halves_f32` 单算子门(2026-10-26 propose 刀3)。
//!
//! 语义 = host `merge_topk_two_halves`(`crates/engine/src/running.rs`)逐位同律:
//!   - A 的 K 个候选(全表域)+ B 的 K 个(半表域,id += half);
//!   - 丢弃 id 非有限的哨兵槽;
//!   - 按 **(值降, id 升)** 全序取前 K;
//!   - 输出单缓冲 [2·rows·K](值区 ‖ 索引区)。
//!
//! 判据 = **逐位相同**(值 f32 位型 + id 整数),不是容差 —— 这条核的全部意义
//! 就是替掉 host 的排序,任何一位不同都会让 drafts 漂移。
//!
//! 运行:`OWL_TEST_DEVICE=1 cargo test --release -p owl-kernels \
//!        --test topk_merge_halves -- --test-threads=1 --nocapture`

use owl_macros::owl_test;

use cudarc::driver::{CudaContext, DevicePtr, LaunchConfig, PushKernelArg};

const ROWS: usize = 7;
const K: usize = 16;
/// Qwen3.8-27B 半表宽(rank1 的 id 偏移)
const HALF: usize = 124160;

/// host 参考 —— 与 `merge_topk_two_halves` 同律(照抄,含哨兵过滤)
fn host_merge(
    vals_a: &[f32],
    ids_a: &[f32],
    vals_b: &[f32],
    ids_b: &[f32],
    rows: usize,
    k: usize,
    half: usize,
) -> (Vec<f32>, Vec<u32>) {
    let mut out_v = Vec::with_capacity(rows * k);
    let mut out_i = Vec::with_capacity(rows * k);
    let mut cand: Vec<(f32, u32)> = Vec::with_capacity(2 * k);
    for r in 0..rows {
        cand.clear();
        for j in 0..k {
            let ia = ids_a[r * k + j];
            if ia.is_finite() {
                cand.push((vals_a[r * k + j], ia as u32));
            }
            let ib = ids_b[r * k + j];
            if ib.is_finite() {
                cand.push((vals_b[r * k + j], ib as u32 + half as u32));
            }
        }
        cand.sort_by(|x, y| {
            y.0.partial_cmp(&x.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(x.1.cmp(&y.1))
        });
        for &(v, i) in cand.iter().take(k) {
            out_v.push(v);
            out_i.push(i);
        }
    }
    (out_v, out_i)
}

/// LCG 造数据;`dup` = 刻意造平局(同值不同 id),`holes` = 尾部留非有限哨兵
fn make_case(seed: u32, dup: bool, holes: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = ROWS * K;
    let mut st = seed;
    let mut rnd = move || {
        st = st.wrapping_mul(1664525).wrapping_add(1013904223);
        ((st >> 8) & 0xFFFF) as f32 / 65535.0
    };
    let mut vals_a = vec![0f32; n];
    let mut ids_a = vec![0f32; n];
    let mut vals_b = vec![0f32; n];
    let mut ids_b = vec![0f32; n];
    for r in 0..ROWS {
        for j in 0..K {
            let i = r * K + j;
            // 值:±8 量级;dup 档每行只造 3 个不同值 ⇒ 大量平局
            vals_a[i] = if dup { (j % 3) as f32 - 1.0 } else { (rnd() - 0.5) * 16.0 };
            vals_b[i] = if dup { ((j + 1) % 3) as f32 - 1.0 } else { (rnd() - 0.5) * 16.0 };
            // id:半表域各自递增,互不重叠
            ids_a[i] = (j * 7 + r) as f32;
            ids_b[i] = (j * 5 + r * 3) as f32;
        }
        // 哨兵:每行尾部 holes 个槽的 id = NaN(未填满)
        for h in 0..holes.min(K) {
            let i = r * K + (K - 1 - h);
            ids_a[i] = f32::NAN;
            ids_b[i] = f32::NAN;
        }
    }
    (vals_a, ids_a, vals_b, ids_b)
}

fn dev_index() -> usize {
    owl_shared::env_reader::str("OWL_TEST_DEVICE")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1)
}

fn run_case(seed: u32, dup: bool, holes: usize) {
    let (vals_a, ids_a, vals_b, ids_b) = make_case(seed, dup, holes);
    let (want_v, want_i) = host_merge(&vals_a, &ids_a, &vals_b, &ids_b, ROWS, K, HALF);

    let ctx = CudaContext::new(dev_index()).expect("ctx");
    ctx.bind_to_thread().expect("bind");
    let stream = ctx.default_stream();
    let dev_f32 = |v: &[f32]| {
        let mut d = unsafe { stream.alloc::<f32>(v.len()) }.unwrap();
        stream.memcpy_htod(v, &mut d).unwrap();
        d
    };
    let d_va = dev_f32(&vals_a);
    let d_ia = dev_f32(&ids_a);
    let d_vb = dev_f32(&vals_b);
    let d_ib = dev_f32(&ids_b);
    let d_out = dev_f32(&vec![0f32; 2 * ROWS * K]);

    let src = owl_kernels::cu_sig::cu_text(owl_kernels::ops::source("owl_topk_merge_halves_f32"));
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(
        src.as_str(),
        cudarc::nvrtc::CompileOptions {
            arch: Some("compute_86"),
            include_paths: vec!["/usr/local/cuda/include".into()],
            ..Default::default()
        },
    )
    .expect("nvrtc compile");
    let m = ctx.load_module(ptx).expect("module");
    let f = m.load_function("owl_topk_merge_halves_f32").expect("load");

    let (va, ia, vb, ib, op) = (
        d_va.device_ptr(&stream).0,
        d_ia.device_ptr(&stream).0,
        d_vb.device_ptr(&stream).0,
        d_ib.device_ptr(&stream).0,
        d_out.device_ptr(&stream).0,
    );
    let (half_i, rows_i, k_i) = (HALF as i32, ROWS as i32, K as i32);
    let cfg = LaunchConfig {
        grid_dim: (ROWS as u32, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b = stream.launch_builder(&f);
    b.arg(&va)
        .arg(&ia)
        .arg(&vb)
        .arg(&ib)
        .arg(&half_i)
        .arg(&rows_i)
        .arg(&k_i)
        .arg(&op);
    unsafe { b.launch(cfg) }.expect("launch");
    stream.synchronize().expect("sync");

    let mut out = vec![0f32; 2 * ROWS * K];
    stream.memcpy_dtoh(&d_out, &mut out).unwrap();
    let got_v = &out[..ROWS * K];
    let got_i: Vec<u32> = out[ROWS * K..].iter().map(|&v| v as u32).collect();

    let v_ok = got_v.iter().zip(want_v.iter()).all(|(a, b)| a.to_bits() == b.to_bits());
    let i_ok = got_i == want_i;
    println!(
        "merge_halves seed={seed} dup={dup} holes={holes}: 值逐位 {} / id 逐位 {}",
        if v_ok { "✓" } else { "✗" },
        if i_ok { "✓" } else { "✗" }
    );
    if !v_ok || !i_ok {
        for r in 0..ROWS {
            println!("  row {r}: got_v={:?}", &got_v[r * K..(r + 1) * K]);
            println!("  row {r}: want_v={:?}", &want_v[r * K..(r + 1) * K]);
            println!("  row {r}: got_i={:?}", &got_i[r * K..(r + 1) * K]);
            println!("  row {r}: want_i={:?}", &want_i[r * K..(r + 1) * K]);
        }
    }
    assert!(v_ok, "候选值不是逐位相同(seed={seed} dup={dup} holes={holes})");
    assert!(i_ok, "候选 id 不是逐位相同(seed={seed} dup={dup} holes={holes})");
}

#[owl_test(gpu)]
fn topk_merge_halves_matches_host() {
    // 常规 / 平局密集 / 尾部哨兵 / 混合
    run_case(7, false, 0);
    run_case(11, true, 0);
    run_case(13, false, 3);
    run_case(17, true, 5);
}

//! **FI 批化 plan 形状探针**(B5.1 / R1 前置判据,2026-10-20)。
//!
//! 问题:spec 回合合批(M2)想把 **B 个会话的 verify 行**拼成一次 FI plan
//! (`qo_indptr=[0,8,16,…]`、`kv_indptr=[0,nb1,nb1+nb2,…]`)。若同一张捕获图要在
//! **不同 ctx 组合**上回放,plan 的发射形状(plan15 / cta_tile_q / split / int_bytes)
//! 必须对"各会话 nb 的分配"恒定 —— 否则只能退回 v2 seq 臂(R2 打折)。
//!
//! 本探针直接调 `owl_fi_prefill_plan`(host 规划),对同一 `total_rows / batch`
//! 扫描不同的 nb 切分,打印形状签名。判据:
//! - **同 total、不同切分**下签名全同 ⇒ 形状只由 (T, B, Σnb) 定 ⇒ R1 通过(直接拼);
//! - 若 `split/cta` 随 `max(nb_i)` 变 ⇒ 需要"按上界补页"或 v2 seq 兜底。
//!
//! 用法:
//! ```text
//! OWL_TEST_DEVICE=42:00.0 cargo test --release -p owl-kernels \
//!   --features device,flashinfer --test fi_batch_plan_probe -- --test-threads=1 --nocapture
//! ```
#![cfg(feature = "device")]

mod fi_support;
use fi_support::{TestRes, H, HD, HKV, PAGE};

use owl_kernels::family::flashinfer::{
    owl_fi_prefill_plan, FI_FLOAT_WS_BYTES, FI_HOST_STAGING_BYTES, FI_INT_WS_BYTES,
};

#[derive(Debug, Clone, PartialEq, Eq)]
struct Sig {
    plan0: i64,
    plan3: i64,
    cta: i32,
    split: i32,
    int_bytes: usize,
}

fn plan_one(
    res: &mut TestRes,
    q_per_seq: usize,
    nb: &[usize],
) -> Sig {
    let batch = nb.len();
    let total_rows = q_per_seq * batch;
    let float_ws = unsafe { res.stream.alloc::<u8>(FI_FLOAT_WS_BYTES) }.expect("float_ws");
    let int_ws = unsafe { res.stream.alloc::<u8>(FI_INT_WS_BYTES) }.expect("int_ws");
    let (float_ptr, _fgu) = cudarc::driver::DevicePtr::device_ptr(&float_ws, &res.stream);
    let (int_ptr, _igu) = cudarc::driver::DevicePtr::device_ptr(&int_ws, &res.stream);
    let mut staging = vec![0u8; FI_HOST_STAGING_BYTES];
    let mut qo = Vec::with_capacity(batch + 1);
    let mut kv = Vec::with_capacity(batch + 1);
    qo.push(0i32);
    kv.push(0i32);
    for (i, n) in nb.iter().enumerate() {
        qo.push(((i + 1) * q_per_seq) as i32);
        kv.push(*n as i32 + kv[i]);
    }
    let mut plan15 = [0i64; 15];
    let (mut cta, mut split) = (0i32, 0i32);
    let mut int_bytes = 0usize;
    let rc = unsafe {
        owl_fi_prefill_plan(
            float_ptr as *mut std::ffi::c_void,
            FI_FLOAT_WS_BYTES,
            int_ptr as *mut std::ffi::c_void,
            FI_INT_WS_BYTES,
            staging.as_mut_ptr() as *mut std::ffi::c_void,
            staging.len(),
            plan15.as_mut_ptr(),
            &mut cta,
            &mut split,
            &mut int_bytes,
            qo.as_ptr(),
            kv.as_ptr(),
            total_rows as i32,
            batch as i32,
            H as i32,
            HKV as i32,
            HD as i32,
            PAGE as i32,
            res.stream.cu_stream() as *mut std::ffi::c_void,
        )
    };
    assert_eq!(rc, 0, "plan 失败 rc={rc} (q={q_per_seq} nb={nb:?})");
    res.stream.synchronize().expect("sync");
    Sig { plan0: plan15[0], plan3: plan15[3], cta, split, int_bytes }
}

#[test]
fn batch_plan_shape_probe() {
    if owl_cuda::test_device_bdf().is_none() {
        owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
        return;
    }
    let dev = owl_cuda::test_device_ordinal();
    let mut res = TestRes::new(dev);

    // verify 深度 8(1 + 7 提案);contexts 以页计(page=32 ⇒ 1 页 = 32 token)
    // 4k ctx ≈ 128 页;8k ≈ 256 页;16k ≈ 512 页。
    let combos_2: Vec<Vec<usize>> = vec![
        vec![128, 128],
        vec![200, 56],
        vec![240, 16],
        vec![250, 6],
    ];
    let combos_4: Vec<Vec<usize>> = vec![
        vec![128, 128, 128, 128],
        vec![200, 200, 100, 12],
        vec![256, 128, 128, 0],
        vec![500, 10, 1, 1],
    ];
    let combos_8: Vec<Vec<usize>> = vec![
        vec![128; 8],
        vec![256, 128, 128, 128, 128, 128, 128, 0],
        vec![500, 128, 128, 128, 20, 10, 2, 1],
    ];

    println!("mode,B,total_nb,nb_combo,padded,plan3,cta,split,int_bytes,shape_ok");
    for graph in [0u32, 1u32] {
        let mode = graph == 1;
        std::env::set_var("FI_CUDA_GRAPH", if mode { "1" } else { "0" });
        owl_kernels::server::flashinfer::set_plan_mode(mode);
        for (b, combos) in [(2usize, &combos_2), (4, &combos_4), (8, &combos_8)] {
            let mut baseline: Option<Sig> = None;
            for nb in combos {
                let total: usize = nb.iter().sum();
                let sig = plan_one(&mut res, 8, nb);
                let ok = match &baseline {
                    None => {
                        baseline = Some(sig.clone());
                        true
                    }
                    Some(b0) => b0 == &sig,
                };
                println!(
                    "{graph},{b},{total},{nb:?},p0={},p3={},cta={},split={},ib={},ok={ok}",
                    sig.plan0, sig.plan3, sig.cta, sig.split, sig.int_bytes
                );
            }
        }
        // 单序列基线(串行 = B=1 的 8 行,作对照)
        for nb in [128usize, 256, 512] {
            let sig = plan_one(&mut res, 8, &[nb]);
            println!(
                "{},1,{nb},[{nb}],p0={},p3={},cta={},split={},ib={},ok=true",
                graph, sig.plan0, sig.plan3, sig.cta, sig.split, sig.int_bytes
            );
        }
    }
    std::env::remove_var("FI_CUDA_GRAPH");
}

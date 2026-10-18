//! **FI plan 形状扫描**（2026-10-18 性能专项 · 图档桶化的第 0 号判据）。
//!
//! 为什么要有它：ninfer-3090 能把 decode/verify 全进图，靠的是
//! **形状只由"信封上界"定、真实长度在核内夹取**；而我们的 FI 是**宿主侧按真实 ctx
//! 算的调度表**。要让 FI 进图，必须先把"形状"从"真实 ctx"里剥离出来 ——
//! 也就是挑一组**桶边界**，使桶内 plan 的**发射形状恒定**（于是可以烘图 + 过供给 +
//! 运行期夹取）。本探针就是找这些边界：**对阶梯 ctx 逐个调 plan，打印形状签名，
//! 跳变点即桶边界**。
//!
//! 形状签名 = (padded_batch_size, cta_tile_q, split_kv)；另记 `int_bytes`
//! (= plan 表写进 int_ws 的有效前缀 ⇒ 决定"图自携 plan"快照尺寸与该桶的表布局)。
//! 两档分别扫：`FI_CUDA_GRAPH=0`(eager 定瓦)与 `=1`(**图档定瓦** —— 终局设计用这档)。
//!
//! 用法：
//! ```text
//! CUDA_VISIBLE_DEVICES=<UUID> OWL_TEST_DEVICE=0 \
//!   cargo test --release -p owl-kernels --features device,flashinfer \
//!   --test fi_plan_scan -- --test-threads=1 --nocapture
//! ```
//! 输出：CSV 行 `mode,t,ctx,nb_pages,padded,cta_tile_q,split,int_bytes`（stdout）。
#![cfg(feature = "device")]

mod fi_support;
use fi_support::{TestRes, H, HD, HKV, PAGE};

use owl_kernels::family::flashinfer::{
    owl_fi_prefill_plan, FI_FLOAT_WS_BYTES, FI_HOST_STAGING_BYTES, FI_INT_WS_BYTES,
};

/// 阶梯：覆盖 2^5 → 2^18(=262144)，并在每两个 2 的幂之间取中点（抓细跳变）。
fn ladder() -> Vec<usize> {
    let mut v = vec![];
    let mut p = 32usize;
    while p <= 1 << 18 {
        v.push(p);
        if p * 3 / 2 <= 1 << 18 {
            v.push(p + p / 2);
        }
        p *= 2;
    }
    v.sort_unstable();
    v.dedup();
    v
}

fn plan_one(res: &mut TestRes, t: usize, ctx: usize) -> (i64, i64, i32, usize) {
    let nb = ctx.div_ceil(PAGE);
    // 工作区用 TestRes 的裸分配面(与 FiRuntime 同尺寸:float 64MiB / int 32MiB)
    let float_ws = unsafe { res.stream.alloc::<u8>(FI_FLOAT_WS_BYTES) }.expect("float_ws");
    let int_ws = unsafe { res.stream.alloc::<u8>(FI_INT_WS_BYTES) }.expect("int_ws");
    let (float_ptr, _fgu) = cudarc::driver::DevicePtr::device_ptr(&float_ws, &res.stream);
    let (int_ptr, _igu) = cudarc::driver::DevicePtr::device_ptr(&int_ws, &res.stream);
    let mut staging = vec![0u8; FI_HOST_STAGING_BYTES];
    let qo = [0i32, t as i32];
    let kv = [0i32, nb as i32];
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
            t as i32,
            1,
            H as i32,
            HKV as i32,
            HD as i32,
            PAGE as i32,
            res.stream.cu_stream() as *mut std::ffi::c_void,
        )
    };
    assert_eq!(rc, 0, "plan 失败 rc={rc} (T={t} ctx={ctx} nb={nb})");
    res.stream.synchronize().expect("sync");
    (plan15[0], plan15[3], split, int_bytes)
}

#[test]
fn scan_plan_shape() {
    if owl_cuda::test_device_bdf().is_none() {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    }
    let dev = owl_cuda::test_device_ordinal();
    let mut res = TestRes::new(dev);
    println!("mode,t,ctx,nb_pages,padded,cta_tile_q,split,int_bytes");
    for graph in [0u32, 1u32] {
        // C++ / Rust 两侧都是逐次 getenv（无 OnceLock）⇒ 逐档切档有效
        std::env::set_var("FI_CUDA_GRAPH", if graph == 1 { "1" } else { "0" });
        for t in [1usize, 4, 8, 64] {
            for ctx in ladder() {
                let (padded, cta, split, ib) = plan_one(&mut res, t, ctx);
                println!("{graph},{t},{ctx},{},{padded},{cta},{split},{ib}", ctx.div_ceil(PAGE));
            }
        }
    }
    std::env::remove_var("FI_CUDA_GRAPH");
}

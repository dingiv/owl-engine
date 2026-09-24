//! 基建修复钉测:narrow 中间维 + ctor 真路径(2026-09-23 波次三收口)。
//!
//! - `OwlTensor::narrow` 中间维:修复前 out_dim 误传 len(丢 after 因子),
//!   中间维窄切只拷每行前 len 个元素,其余为 scratch 脏数据;
//! - `ctor::zeros/ones/full`:原 t3_unimpl 桩,现走池三原语
//!   (zeros_tensor/full_tensor,F32/U32 双档)。
//!
//! 独立二进制 = rig OnceLock 进程隔离(dry_run.rs / rotary_truth.rs 先例)。

use owl_cuda::CudaDevice;
use owl_engine::models::dry_kernels::DryKernels;
use owl_engine::models::layers::{ctx_scope, OwlTensor};
use owl_nn::cublas::NnBlas;
use std::sync::Arc;

fn dtoh_f32(dev: &CudaDevice, ptr: *mut f32, n: usize) -> Vec<f32> {
    use owl_cuda::ffi::sys;
    dev.ctx().bind_to_thread().unwrap();
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

fn install_rig() -> Arc<CudaDevice> {
    let dev = Arc::new(CudaDevice::new(owl_cuda::test_device_ordinal(), owl_cuda::TEST_POOL_BYTES).expect("需要 CUDA 设备"));
    let scratch = dev.default_pool();
    let wpool = dev.default_pool();
    let ops = owl_nn::OpsCtx::new(&dev).unwrap();
    let blas = NnBlas::new(&dev).unwrap();
    let dry = DryKernels::new(dev.ctx()).unwrap();
    ctx_scope::install(ops, blas, dry, scratch, wpool, &dev);
    dev
}

/// narrow 中间维:[2,3,4] f32 序列张量,窄切 dim=1[1..3] → [2,2,4],
/// 全部 16 个输出元素逐一核对(修复前每行只有前 4 个对,后 4 个是脏数据)。
#[test]
fn narrow_middle_dim_full_parity() {
    let dev = install_rig();
    let src: Vec<f32> = (0..2 * 3 * 4).map(|i| i as f32 * 1.5).collect();
    let t = owl_engine::models::layers::ctor::from_vec(src.clone(), [2usize, 3, 4], &dev).unwrap();
    let out = t.narrow(1usize, 1, 2).unwrap();
    assert_eq!(out.shape(), &[2, 2, 4]);
    let got = dtoh_f32(&dev, out.device_ptr() as *mut f32, 2 * 2 * 4);
    // 期望:src[r, 1..3, k]
    for r in 0..2usize {
        for j in 0..2usize {
            for k in 0..4usize {
                let want = src[r * 12 + (1 + j) * 4 + k];
                assert_eq!(got[r * 8 + j * 4 + k], want, "r={r} j={j} k={k}");
            }
        }
    }
}

/// ctor 三原语:zeros 全零 / ones 全一 / full 常量,U32 档位型保真。
#[test]
fn ctor_zeros_ones_full_pool_paths() {
    let dev = install_rig();
    let z = owl_engine::models::layers::ctor::zeros([3usize, 5], owl_nn::Dtype::F32, &dev).unwrap();
    assert_eq!(z.shape(), &[3, 5]);
    let gz = dtoh_f32(&dev, z.device_ptr() as *mut f32, 15);
    assert!(gz.iter().all(|&x| x == 0.0));

    let o = owl_engine::models::layers::ctor::ones([4usize], owl_nn::Dtype::F32, &dev).unwrap();
    let go = dtoh_f32(&dev, o.device_ptr() as *mut f32, 4);
    assert!(go.iter().all(|&x| x == 1.0));

    let f = owl_engine::models::layers::ctor::full([2usize, 2], 7.5, owl_nn::Dtype::F32, &dev)
        .unwrap();
    let gf = dtoh_f32(&dev, f.device_ptr() as *mut f32, 4);
    assert!(gf.iter().all(|&x| x == 7.5));

    // U32 位型保真(zeros 路径;slot 语义依赖)
    let u = owl_engine::models::layers::ctor::zeros([8usize], owl_nn::Dtype::U32, &dev).unwrap();
    assert_eq!(u.dtype(), owl_nn::Dtype::U32);

    // 不支持 dtype = 结构化报错(非 panic)
    let e = owl_engine::models::layers::ctor::ones([2usize], owl_nn::Dtype::F16, &dev);
    assert!(e.is_err());
}

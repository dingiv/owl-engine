//! 集成对拍:owl_nn 公开 API vs tests/common 的 f64 参考。
//! 需要真机 CUDA(与 src 内联测试同纪律)。

mod common;

use common::{
    add, assert_allclose_f32, exp, gelu, matmul, mul, rmsnorm, silu, softmax_last_dim, Lcg,
};
use owl_cuda::CudaDevice;
use owl_iface::{Device, MemPhase, PoolConfig, PoolKind};
use owl_nn::cublas::NnBlas;
use owl_nn::ops::OpsCtx;
use owl_nn::tensor::TensorPoolOps;

#[test]
fn ref_ops_all_match_common_reference() {
    let dev = CudaDevice::new(owl_cuda::test_device_ordinal()).expect("需要 CUDA 设备");
    let persist = dev
        .create_pool(PoolConfig {
            name: format!("ref-persist-{}", std::process::id()),
            kind: PoolKind::Weights,
            bytes: 8 << 20,
        })
        .unwrap();
    let scratch = dev
        .create_pool(PoolConfig {
            name: format!("ref-scratch-{}", std::process::id()),
            kind: PoolKind::Scratch,
            bytes: 8 << 20,
        })
        .unwrap();
    let blas = NnBlas::new(&dev).unwrap();
    let mut ops = OpsCtx::new(&dev).unwrap();
    let ctx = ops.ctx(MemPhase::Idle);

    let (m, k, n) = (8usize, 16usize, 12usize);
    let mut seed = Lcg::new(7);

    // ---- matmul ----
    let mut wa = persist.zeros_tensor::<f32>(&[m, k]).unwrap();
    let mut wb = persist.zeros_tensor::<f32>(&[k, n]).unwrap();
    let mut wout = persist.zeros_tensor::<f32>(&[m, n]).unwrap();
    let ha = seed.vec(m * k);
    let hb = seed.vec(k * n);
    ops.fill_from_host(&ctx, &mut wa, &ha).unwrap();
    ops.fill_from_host(&ctx, &mut wb, &hb).unwrap();
    ops.matmul(&ctx, &blas, &wa, &wb, &mut wout).unwrap();
    dev.ctx().synchronize().unwrap();
    let got = wout.to_vec().unwrap();
    let want = matmul(&ha, &hb, m, k, n);
    assert_allclose_f32(&got, &want, 1e-4, 1e-4, "matmul");

    // ---- 逐元素(f32 输入,got vs f64 参考)----
    let xn = 64usize;
    let cols = 8usize;
    let mut x = scratch.scratch_tensor::<f32>(&[xn]).unwrap();
    let mut y = scratch.scratch_tensor::<f32>(&[xn]).unwrap();
    let mut out = scratch.scratch_tensor::<f32>(&[xn]).unwrap();
    let hx = seed.vec(xn);
    let hy = seed.vec(xn);
    ops.fill_from_host(&ctx, &mut x, &hx).unwrap();
    ops.fill_from_host(&ctx, &mut y, &hy).unwrap();

    ops.add(&ctx, &x, &y, &mut out).unwrap();
    let want = add(&hx, &hy, 1, xn);
    assert_allclose_f32(&out.to_vec().unwrap(), &want, 0.0, 1e-5, "add");
    ops.mul(&ctx, &x, &y, &mut out).unwrap();
    let want = mul(&hx, &hy);
    assert_allclose_f32(&out.to_vec().unwrap(), &want, 0.0, 1e-5, "mul");
    ops.exp(&ctx, &x, &mut out).unwrap();
    let want = exp(&hx);
    assert_allclose_f32(&out.to_vec().unwrap(), &want, 1e-6, 1e-6, "exp");
    ops.silu(&ctx, &x, &mut out).unwrap();
    let want = silu(&hx);
    assert_allclose_f32(&out.to_vec().unwrap(), &want, 1e-6, 1e-6, "silu");
    ops.gelu(&ctx, &x, &mut out).unwrap();
    let want = gelu(&hx);
    assert_allclose_f32(&out.to_vec().unwrap(), &want, 1e-5, 1e-5, "gelu");

    // ---- softmax last-dim([rows=8, cols=8])----
    let mut sx = scratch.scratch_tensor::<f32>(&[xn / cols, cols]).unwrap();
    let mut s_out = scratch.scratch_tensor::<f32>(&[xn / cols, cols]).unwrap();
    ops.fill_from_host(&ctx, &mut sx, &hx).unwrap();
    ops.softmax(&ctx, &sx, &mut s_out).unwrap();
    let want = softmax_last_dim(&hx, cols);
    assert_allclose_f32(&s_out.to_vec().unwrap(), &want, 1e-5, 1e-6, "softmax");

    // ---- rmsnorm([rows=8, cols=8] × alpha[8])----
    let mut alpha = persist.zeros_tensor::<f32>(&[cols]).unwrap();
    let halpha = seed.vec(cols);
    let mut r_out = scratch.scratch_tensor::<f32>(&[xn / cols, cols]).unwrap();
    ops.fill_from_host(&ctx, &mut alpha, &halpha).unwrap();
    let normed: Vec<f32> = silu(&hx).iter().map(|&v| v as f32).collect();
    ops.fill_from_host(&ctx, &mut sx, &normed).unwrap();
    ops.rmsnorm(&ctx, &sx, &alpha, &mut r_out, 1e-5).unwrap();
    let want = rmsnorm(
        &normed.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        &halpha,
        cols,
        1e-5,
    );
    assert_allclose_f32(&r_out.to_vec().unwrap(), &want, 1e-4, 1e-5, "rmsnorm");
}

#[test]
fn ref_chain_matmul_add_silu_rmsnorm() {
    let dev = CudaDevice::new(owl_cuda::test_device_ordinal()).expect("需要 CUDA 设备");
    let persist = dev
        .create_pool(PoolConfig {
            name: format!("ref-chain-persist-{}", std::process::id()),
            kind: PoolKind::Weights,
            bytes: 8 << 20,
        })
        .unwrap();
    let _scratch = dev
        .create_pool(PoolConfig {
            name: format!("ref-chain-scratch-{}", std::process::id()),
            kind: PoolKind::Scratch,
            bytes: 8 << 20,
        })
        .unwrap();
    let blas = NnBlas::new(&dev).unwrap();
    let mut ops = OpsCtx::new(&dev).unwrap();
    let ctx = ops.ctx(MemPhase::Idle);

    let (m, k, n) = (8, 16, 12);
    let mut seed = Lcg::new(42);
    let w = persist.from_vec_tensor::<f32>(&[m, k], seed.vec(m * k)).unwrap();
    let a = persist.from_vec_tensor::<f32>(&[k, n], seed.vec(k * n)).unwrap();
    let mut mm_out = persist.zeros_tensor::<f32>(&[m, n]).unwrap();
    let bias = persist.from_vec_tensor::<f32>(&[m, n], seed.vec(m * n)).unwrap();
    let mut add_out = persist.zeros_tensor::<f32>(&[m, n]).unwrap();
    let mut silu_out = persist.zeros_tensor::<f32>(&[m, n]).unwrap();
    let alpha = persist.from_vec_tensor::<f32>(&[n], seed.vec(n)).unwrap();
    let mut out = persist.zeros_tensor::<f32>(&[m, n]).unwrap();

    ops.matmul(&ctx, &blas, &w, &a, &mut mm_out).unwrap();
    dev.ctx().synchronize().unwrap();
    let wv = w.to_vec().unwrap();
    let av = a.to_vec().unwrap();
    let bv = bias.to_vec().unwrap();
    let avv = alpha.to_vec().unwrap();

    // 分段对拍:逐级定位偏差来源
    let r_mm = matmul(&wv, &av, m, k, n);
    assert_allclose_f32(&mm_out.to_vec().unwrap(), &r_mm, 1e-4, 1e-4, "chain/matmul");

    ops.add(&ctx, &mm_out, &bias, &mut add_out).unwrap();
    dev.ctx().synchronize().unwrap();
    let mm32: Vec<f32> = r_mm.iter().map(|&v| v as f32).collect();
    let r_add = add(&mm32, &bv, m, n);
    assert_allclose_f32(&add_out.to_vec().unwrap(), &r_add, 1e-5, 1e-5, "chain/add");

    ops.silu(&ctx, &add_out, &mut silu_out).unwrap();
    dev.ctx().synchronize().unwrap();
    let add32: Vec<f32> = r_add.iter().map(|&v| v as f32).collect();
    let r_silu = silu(&add32);
    assert_allclose_f32(&silu_out.to_vec().unwrap(), &r_silu, 1e-5, 1e-5, "chain/silu");

    ops.rmsnorm(&ctx, &silu_out, &alpha, &mut out, 1e-5).unwrap();
    dev.ctx().synchronize().unwrap();
    let sil64: Vec<f64> = r_silu.clone();
    let r_out = rmsnorm(&sil64, &avv, n, 1e-5);
    assert_allclose_f32(&out.to_vec().unwrap(), &r_out, 1e-4, 1e-4, "chain/rmsnorm");
}

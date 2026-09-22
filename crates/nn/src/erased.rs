//! 擦除层算子面:DynTensor → 强类型 kernel 的桥(T2-三 OwlTensor 回填,
//! layer 侧垫片由主 agent 接到 OwlTensor impl)。
//!
//! 纪律:
//! - S1:二元算子两侧同 dtype;P0 仅 F32 有设备路径(其他 dtype =
//!   结构化报错,不是静默降级);
//! - S2:P0 仅同形二元(broadcast 右对齐 = P1);
//! - 输入经 `downcast` TensorRef(**偏移视图安全**:ptr 已带偏移,活性由
//!   keepalive 兜底);输出经 [`KernelCtx::scratch_tensor`](S6 通道,
//!   捕获期自动 emit 进图租约);
//! - softmax/rmsnorm 一期 2D(与 ops.rs 强类型面同一约束)。

use crate::cublas::NnBlas;
use crate::dtype::Dtype;
use crate::ops::OpsCtx;
use crate::{DynTensor, KernelCtx};
use owl_cuda::CudaDevice;
use owl_iface::BackendError;

fn require_f32(dt: Dtype, what: &str) -> Result<(), BackendError> {
    if dt != Dtype::F32 {
        return Err(BackendError::Init(format!(
            "erased::{what}: dtype {dt} 无设备路径(P0 仅 F32;S1 结构化报错)"
        )));
    }
    Ok(())
}

fn same_shape(a: &DynTensor<CudaDevice>, b: &DynTensor<CudaDevice>, what: &str) -> Result<(), BackendError> {
    if a.shape() != b.shape() {
        return Err(BackendError::Init(format!(
            "erased::{what}: 形状不一致 {:?} vs {:?}(S2 broadcast = P1)",
            a.shape(),
            b.shape()
        )));
    }
    Ok(())
}

fn elems(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// `out = a + b`(F32,同形;偏移视图安全)
pub fn add(
    ops: &mut OpsCtx,
    ctx: &KernelCtx,
    a: &DynTensor<CudaDevice>,
    b: &DynTensor<CudaDevice>,
) -> Result<DynTensor<CudaDevice>, BackendError> {
    require_f32(a.dtype(), "add")?;
    same_shape(a, b, "add")?;
    let ra = a.downcast::<f32>()?;
    let rb = b.downcast::<f32>()?;
    let out = ctx.scratch_tensor::<f32>(a.shape())?;
    ctx.trace_launch("add");
    ops.note_launch();
    for t in [a.token(), b.token(), out.token()] {
        if let Some(t) = t {
            owl_signal::emit(t);
        }
    }
    let n = elems(a.shape());
    let stream = std::sync::Arc::clone(ctx.stream());
    ops.kernels()
        .add_f32(&stream, n, ra.device_ptr(), rb.device_ptr(), out.device_ptr())
        .map_err(BackendError::Init)?;
    Ok(DynTensor::from_f32(&out))
}

/// `out = a * b`(F32,同形;偏移视图安全)
pub fn mul(
    ops: &mut OpsCtx,
    ctx: &KernelCtx,
    a: &DynTensor<CudaDevice>,
    b: &DynTensor<CudaDevice>,
) -> Result<DynTensor<CudaDevice>, BackendError> {
    require_f32(a.dtype(), "mul")?;
    same_shape(a, b, "mul")?;
    let ra = a.downcast::<f32>()?;
    let rb = b.downcast::<f32>()?;
    let out = ctx.scratch_tensor::<f32>(a.shape())?;
    ctx.trace_launch("mul");
    ops.note_launch();
    for t in [a.token(), b.token(), out.token()] {
        if let Some(t) = t {
            owl_signal::emit(t);
        }
    }
    let n = elems(a.shape());
    let stream = std::sync::Arc::clone(ctx.stream());
    ops.kernels()
        .mul_f32(&stream, n, ra.device_ptr(), rb.device_ptr(), out.device_ptr())
        .map_err(BackendError::Init)?;
    Ok(DynTensor::from_f32(&out))
}

/// `out = silu(x)`(F32;偏移视图安全)
pub fn silu(
    ops: &mut OpsCtx,
    ctx: &KernelCtx,
    x: &DynTensor<CudaDevice>,
) -> Result<DynTensor<CudaDevice>, BackendError> {
    require_f32(x.dtype(), "silu")?;
    let rx = x.downcast::<f32>()?;
    let out = ctx.scratch_tensor::<f32>(x.shape())?;
    ctx.trace_launch("silu");
    ops.note_launch();
    for t in [x.token(), out.token()] {
        if let Some(t) = t {
            owl_signal::emit(t);
        }
    }
    let n = elems(x.shape());
    let stream = std::sync::Arc::clone(ctx.stream());
    ops.kernels()
        .silu_f32(&stream, n, rx.device_ptr(), out.device_ptr())
        .map_err(BackendError::Init)?;
    Ok(DynTensor::from_f32(&out))
}

/// last-dim softmax(F32,2D;偏移视图安全)
pub fn softmax_last_dim(
    ops: &mut OpsCtx,
    ctx: &KernelCtx,
    x: &DynTensor<CudaDevice>,
) -> Result<DynTensor<CudaDevice>, BackendError> {
    require_f32(x.dtype(), "softmax")?;
    if x.shape().len() != 2 {
        return Err(BackendError::Init(format!(
            "erased::softmax: 一期仅 2D,得 {:?}",
            x.shape()
        )));
    }
    let rx = x.downcast::<f32>()?;
    let out = ctx.scratch_tensor::<f32>(x.shape())?;
    ctx.trace_launch("softmax");
    ops.note_launch();
    for t in [x.token(), out.token()] {
        if let Some(t) = t {
            owl_signal::emit(t);
        }
    }
    let shape = x.shape();
    let stream = std::sync::Arc::clone(ctx.stream());
    ops.kernels()
        .softmax_f32(&stream, shape[0], shape[1], rx.device_ptr(), out.device_ptr())
        .map_err(BackendError::Init)?;
    Ok(DynTensor::from_f32(&out))
}

/// RMSNorm(F32,2D;alpha last-dim;偏移视图安全)
pub fn rmsnorm(
    ops: &mut OpsCtx,
    ctx: &KernelCtx,
    x: &DynTensor<CudaDevice>,
    alpha: &DynTensor<CudaDevice>,
    eps: f32,
) -> Result<DynTensor<CudaDevice>, BackendError> {
    require_f32(x.dtype(), "rmsnorm")?;
    require_f32(alpha.dtype(), "rmsnorm")?;
    if x.shape().len() != 2 {
        return Err(BackendError::Init(format!(
            "erased::rmsnorm: 一期仅 2D,得 {:?}",
            x.shape()
        )));
    }
    let rx = x.downcast::<f32>()?;
    let ralpha = alpha.downcast::<f32>()?;
    let out = ctx.scratch_tensor::<f32>(x.shape())?;
    ctx.trace_launch("rmsnorm");
    ops.note_launch();
    for t in [x.token(), alpha.token(), out.token()] {
        if let Some(t) = t {
            owl_signal::emit(t);
        }
    }
    let shape = x.shape();
    let stream = std::sync::Arc::clone(ctx.stream());
    ops.kernels()
        .rmsnorm_f32(
            &stream,
            shape[0],
            shape[1],
            rx.device_ptr(),
            ralpha.device_ptr(),
            out.device_ptr(),
            eps,
        )
        .map_err(BackendError::Init)?;
    Ok(DynTensor::from_f32(&out))
}

/// `out[M,N] = a[M,K] × b[K,N]`(F32,2D;cuBLAS,workspace 预钉;偏移视图安全)
pub fn matmul(
    ops: &mut OpsCtx,
    ctx: &KernelCtx,
    blas: &NnBlas,
    a: &DynTensor<CudaDevice>,
    b: &DynTensor<CudaDevice>,
) -> Result<DynTensor<CudaDevice>, BackendError> {
    require_f32(a.dtype(), "matmul")?;
    require_f32(b.dtype(), "matmul")?;
    let (sa, sb) = (a.shape(), b.shape());
    if sa.len() != 2 || sb.len() != 2 || sa[1] != sb[0] {
        return Err(BackendError::Init(format!(
            "erased::matmul: 形状不兼容 {sa:?} × {sb:?}(一期 2D,内积维须相等)"
        )));
    }
    let ra = a.downcast::<f32>()?;
    let rb = b.downcast::<f32>()?;
    let out = ctx.scratch_tensor::<f32>(&[sa[0], sb[1]])?;
    ctx.trace_launch("matmul");
    ops.note_launch();
    for t in [a.token(), b.token(), out.token()] {
        if let Some(t) = t {
            owl_signal::emit(t);
        }
    }
    // workspace 是 cublas 节点的隐藏依赖:捕获路径必须 emit 进租约(A5.2)
    if let Some(t) = blas.workspace_token() {
        owl_signal::emit(t);
    }
    blas.matmul_f32(
        sa[0],
        sb[1],
        sa[1],
        ra.device_ptr(),
        rb.device_ptr(),
        out.device_ptr(),
        ctx.stream(),
    )?;
    Ok(DynTensor::from_f32(&out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::Bf16;
    use crate::TensorPoolOps;
    use owl_cuda::test_device_ordinal;
    use owl_iface::{Device, PoolConfig, PoolKind};

    fn setup() -> (OpsCtx, KernelCtx, CudaDevice, owl_cuda::CudaPool) {
        let dev = CudaDevice::new(test_device_ordinal()).expect("需要 CUDA 设备");
        let ops = OpsCtx::new_with_scratch(&dev, 1 << 20).unwrap();
        let ctx = ops.ctx(owl_iface::MemPhase::Live);
        let pool = dev
            .create_pool(PoolConfig {
                name: format!("erased-t-{}", std::process::id()),
                kind: PoolKind::Weights,
                bytes: 16 << 20,
            })
            .unwrap();
        (ops, ctx, dev, pool)
    }

    /// 确定性伪随机
    struct Lcg(u32);
    impl Lcg {
        fn next(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((self.0 >> 8) as f32 / 16_777_216.0) * 2.0 - 1.0
        }
    }

    fn f32_tensor(
        _dev: &CudaDevice,
        pool: &owl_cuda::CudaPool,
        shape: &[usize],
        data: &[f32],
    ) -> DynTensor<CudaDevice> {
        let t = pool.from_vec_tensor::<f32>(shape, data.to_vec()).unwrap();
        DynTensor::from_f32(&t)
    }

    #[test]
    fn erased_add_matmul_chain_matches_cpu() {
        let (mut ops, ctx, dev, pool) = setup();
        let blas = NnBlas::new(&dev).unwrap();
        let mut seed = Lcg(7);
        let a: Vec<f32> = (0..16).map(|_| seed.next()).collect();
        let b: Vec<f32> = (0..16).map(|_| seed.next()).collect();
        let bias: Vec<f32> = (0..16).map(|_| seed.next()).collect();

        let da = f32_tensor(&dev, &pool, &[4, 4], &a);
        let db = f32_tensor(&dev, &pool, &[4, 4], &b);
        let dneg = f32_tensor(&dev, &pool, &[4, 4], &b.iter().map(|x| -x).collect::<Vec<_>>());
        let dbias = f32_tensor(&dev, &pool, &[4, 4], &bias);

        // add(a, -b) = a - b;再 mul;再 matmul;再加 bias
        let diff = add(&mut ops, &ctx, &da, &dneg).unwrap();
        let prod = mul(&mut ops, &ctx, &diff, &diff).unwrap();
        let mm = matmul(&mut ops, &ctx, &blas, &prod, &db).unwrap();
        let out = add(&mut ops, &ctx, &mm, &dbias).unwrap();

        // CPU 参考(f64)
        let mut want = vec![0f64; 16];
        for i in 0..16 {
            let d = (a[i] - b[i]) as f64;
            want[i] = d * d;
        }
        let mut mm_want = vec![0f64; 16];
        for r in 0..4 {
            for c in 0..4 {
                let mut s = 0f64;
                for k in 0..4 {
                    s += want[r * 4 + k] * b[k * 4 + c] as f64;
                }
                mm_want[r * 4 + c] = s + bias[r * 4 + c] as f64;
            }
        }
        let got = out.typed_f32().unwrap().to_vec().unwrap();
        for i in 0..16 {
            assert!(
                (got[i] as f64 - mm_want[i]).abs() < 1e-3,
                "element {i}: got {} want {}",
                got[i],
                mm_want[i]
            );
        }
    }

    #[test]
    fn erased_softmax_rows_sum_to_one() {
        let (mut ops, ctx, dev, pool) = setup();
        let mut seed = Lcg(11);
        let x: Vec<f32> = (0..8).map(|_| seed.next() * 3.0).collect();
        let dx = f32_tensor(&dev, &pool, &[2, 4], &x);
        let out = softmax_last_dim(&mut ops, &ctx, &dx).unwrap();
        let got = out.typed_f32().unwrap().to_vec().unwrap();
        for r in 0..2 {
            let s: f32 = got[r * 4..r * 4 + 4].iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "row {r} sum {s}");
        }
    }

    #[test]
    fn erased_rmsnorm_matches_cpu() {
        let (mut ops, ctx, dev, pool) = setup();
        let x: Vec<f32> = (0..6).map(|i| (i as f32) - 2.5).collect();
        let alpha = vec![1.1f32, 0.9]; // last-dim = 2
        let dx = f32_tensor(&dev, &pool, &[3, 2], &x);
        let dalpha = f32_tensor(&dev, &pool, &[2], &alpha);
        let out = rmsnorm(&mut ops, &ctx, &dx, &dalpha, 1e-5).unwrap();
        let got = out.typed_f32().unwrap().to_vec().unwrap();
        for r in 0..3 {
            let row = &x[r * 2..r * 2 + 2];
            let ms = row.iter().map(|v| v * v).sum::<f32>() / 2.0;
            let inv = 1.0 / (ms + 1e-5).sqrt();
            for c in 0..2 {
                let want = row[c] * inv * alpha[c];
                assert!((got[r * 2 + c] - want).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn erased_dtype_mismatch_is_structured() {
        let (mut ops, ctx, _dev, pool) = setup();
        // BF16 输入:P0 无路径 → 结构化报错(S1),非 panic
        let t = pool
            .from_vec_tensor::<Bf16>(&[4], vec![Bf16(0x3f80); 4])
            .unwrap();
        let d = DynTensor::from_bf16(&t);
        let r = add(&mut ops, &ctx, &d, &d);
        assert!(r.is_err(), "BF16 add 应结构化报错");
    }

    #[test]
    fn erased_reshape_narrow_views() {
        let (_ops, _ctx, dev, pool) = setup();
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let d = f32_tensor(&dev, &pool, &[2, 3, 4], &data);
        // reshape 真实现(元数据)
        let f = d.flatten_all().unwrap();
        assert_eq!(f.shape(), &[24]);
        let r = f.reshape(&[4, 6]).unwrap();
        assert_eq!(r.shape(), &[4, 6]);
        // narrow_dim0 偏移视图:downcast 携带偏移基址(指针级,非存储克隆)
        let n = r.narrow_dim0(1, 1).unwrap();
        assert_eq!(n.shape(), &[1, 6]);
        let t = n.downcast::<f32>().unwrap();
        assert_eq!(t.shape(), &[1, 6]);
        // 越界 = 结构化报错
        assert!(r.narrow_dim0(5, 2).is_err());
    }

    #[test]
    fn erased_offset_view_ops_read_correct_rows() {
        // 偏移视图做算子输入:必须读到正确的行(指针偏移生效)
        let (mut ops, ctx, dev, pool) = setup();
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let d = f32_tensor(&dev, &pool, &[3, 4], &data);
        let ones = f32_tensor(&dev, &pool, &[1, 4], &vec![1.0f32; 4]);
        let row1 = d.narrow_dim0(1, 1).unwrap(); // [1,4] = [4,5,6,7]
        let out = add(&mut ops, &ctx, &row1, &ones).unwrap();
        let got = out.typed_f32().unwrap().to_vec().unwrap();
        assert_eq!(got, vec![5.0, 6.0, 7.0, 8.0]);
    }
}

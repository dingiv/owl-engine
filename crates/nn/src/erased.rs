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

// ============================================================================
// P1:索引 / 归约 / 组合 / broadcast 族(f32;S4 索引恒 U32)
// ============================================================================

/// 流上 D2D 拷贝(捕获安全;字节级)
fn copy_d2d(
    ctx: &KernelCtx,
    dst: *mut core::ffi::c_void,
    src: *const core::ffi::c_void,
    bytes: usize,
) -> Result<(), BackendError> {
    use owl_cuda::ffi::sys;
    unsafe {
        sys::cuMemcpyDtoDAsync_v2(
            dst as sys::CUdeviceptr,
            src as sys::CUdeviceptr,
            bytes,
            ctx.stream().cu_stream(),
        )
        .result()
        .map_err(|e| BackendError::CopyFailed { dir: "d2d", detail: format!("{e:?}") })
    }
}

/// cat(f32;dim ∈ {0, last};同 dtype 断言;偏移视图安全)
pub fn cat(
    ctx: &KernelCtx,
    parts: &[DynTensor<CudaDevice>],
    dim: usize,
) -> Result<DynTensor<CudaDevice>, BackendError> {
    if parts.is_empty() {
        return Err(BackendError::Init("cat: 空输入".into()));
    }
    let dt = parts[0].dtype();
    require_f32(dt, "cat")?;
    let rank = parts[0].shape().len();
    if dim >= rank || (dim != 0 && dim + 1 != rank) {
        return Err(BackendError::Init(format!(
            "cat: 一期仅支持 dim=0 或 last(收到 dim={dim}/rank={rank})"
        )));
    }
    for p in parts {
        if p.dtype() != dt {
            return Err(BackendError::Init("cat: dtype 不一致(S1)".into()));
        }
        if p.shape().len() != rank {
            return Err(BackendError::Init("cat: rank 不一致".into()));
        }
        for (i, (a, b)) in p.shape().iter().zip(parts[0].shape()).enumerate() {
            if i != dim && a != b {
                return Err(BackendError::Init(format!("cat: 非拼接维 {i} 不一致")));
            }
        }
    }
    let mut out_shape = parts[0].shape().to_vec();
    out_shape[dim] = parts.iter().map(|p| p.shape()[dim]).sum();
    let out = ctx.scratch_tensor::<f32>(&out_shape)?;
    ctx.trace_launch("cat(copy_d2d)");
    let _inner: usize = out_shape[dim + 1..].iter().product();
    let mut dst_off = 0usize;
    if dim == 0 {
        // 行块拷贝:每段 = rows × inner 连续块
        for p in parts {
            let bytes = elems(p.shape()) * 4;
            copy_d2d(
                ctx,
                unsafe { out.device_ptr().add(dst_off) } as *mut _,
                p.device_ptr() as *const _,
                bytes,
            )?;
            dst_off += elems(p.shape());
        }
    } else {
        // last 维:逐行段拷贝(rows = 外面积)
        let rows: usize = out_shape[..dim].iter().product();
        let mut src_row_off = 0usize;
        for p in parts {
            let seg = p.shape()[dim];
            for r in 0..rows {
                copy_d2d(
                    ctx,
                    unsafe { out.device_ptr().add(r * out_shape[dim] + dst_off) } as *mut _,
                    unsafe { p.device_ptr().add(r * seg) } as *const _,
                    seg * 4,
                )?;
            }
            let _ = src_row_off;
            src_row_off += seg;
            dst_off += seg;
        }
        let _ = src_row_off;
    }
    for p in parts {
        if let Some(t) = p.token() {
            owl_signal::emit(t);
        }
    }
    if let Some(t) = out.token() {
        owl_signal::emit(t);
    }
    Ok(DynTensor::from_f32(&out))
}

/// stack(f32;仅 dim=0:新首维,其余形状必须一致)
pub fn stack(
    ctx: &KernelCtx,
    parts: &[DynTensor<CudaDevice>],
) -> Result<DynTensor<CudaDevice>, BackendError> {
    if parts.is_empty() {
        return Err(BackendError::Init("stack: 空输入".into()));
    }
    let mut shape = vec![parts.len()];
    shape.extend_from_slice(parts[0].shape());
    let out = ctx.scratch_tensor::<f32>(&shape)?;
    ctx.trace_launch("stack(copy_d2d)");
    let seg = elems(parts[0].shape()) * 4;
    for (i, p) in parts.iter().enumerate() {
        if p.shape() != parts[0].shape() || p.dtype() != parts[0].dtype() {
            return Err(BackendError::Init("stack: 形状/dtype 不一致".into()));
        }
        copy_d2d(
            ctx,
            unsafe { out.device_ptr().add(i * elems(parts[0].shape())) } as *mut _,
            p.device_ptr() as *const _,
            seg,
        )?;
        if let Some(t) = p.token() {
            owl_signal::emit(t);
        }
    }
    if let Some(t) = out.token() {
        owl_signal::emit(t);
    }
    Ok(DynTensor::from_f32(&out))
}

/// index_select(f32;dim=0 或 last;idx = U32 DynTensor,S4)
pub fn index_select(
    ops: &mut OpsCtx,
    ctx: &KernelCtx,
    src: &DynTensor<CudaDevice>,
    dim: usize,
    idx: &DynTensor<CudaDevice>,
) -> Result<DynTensor<CudaDevice>, BackendError> {
    require_f32(src.dtype(), "index_select")?;
    if idx.dtype() != Dtype::U32 {
        return Err(BackendError::Init(format!(
            "index_select: 索引 dtype {} ≠ U32(S4)",
            idx.dtype()
        )));
    }
    let rank = src.shape().len();
    if dim >= rank || (dim != 0 && dim + 1 != rank) {
        return Err(BackendError::Init(format!(
            "index_select: 一期仅 dim=0 或 last(收到 {dim}/{rank})"
        )));
    }
    let n_idx = elems(idx.shape());
    let inner: usize = src.shape()[dim + 1..].iter().product();
    let mut out_shape = src.shape().to_vec();
    out_shape[dim] = n_idx;
    let out = ctx.scratch_tensor::<f32>(&out_shape)?;
    ctx.trace_launch("index_select");
    ops.note_launch();
    let stream = std::sync::Arc::clone(ctx.stream());
    let rs = src.downcast::<f32>()?;
    let ri = idx.downcast::<u32>()?;
    let ro = out.device_ptr();
    if dim == 0 {
        // rows-gather:out[k,c] = src[idx[k],c](idx 设备侧,捕获安全)
        let outer: usize = src.shape()[..dim].iter().product();
        if outer != 1 {
            return Err(BackendError::Init(
                "index_select(dim=0): 一期仅 leading 维为 1 的行块gather".into(),
            ));
        }
        ops.kernels()
            .rows_gather_f32(&stream, n_idx as u64, inner as u64, rs.device_ptr(), ri.device_ptr(), ro)
            .map_err(BackendError::Init)?;
    } else {
        let flat_n = n_idx * inner;
        ops.kernels()
            .gather_f32(&stream, flat_n, rs.device_ptr(), ri.device_ptr(), ro)
            .map_err(BackendError::Init)?;
    }
    for t in [src.token(), idx.token(), out.token()] {
        if let Some(t) = t {
            owl_signal::emit(t);
        }
    }
    Ok(DynTensor::from_f32(&out))
}

/// gather(f32;last 维语义扁平化:out[i] = src[flat_idx[i]],
/// flat_idx 由调用方按 idx*inner 预展开;P1 通用版 = idx 直读版,见下)
pub fn gather(
    ops: &mut OpsCtx,
    ctx: &KernelCtx,
    src: &DynTensor<CudaDevice>,
    idx: &DynTensor<CudaDevice>,
) -> Result<DynTensor<CudaDevice>, BackendError> {
    require_f32(src.dtype(), "gather")?;
    if idx.dtype() != Dtype::U32 {
        return Err(BackendError::Init("gather: 索引恒 U32(S4)".into()));
    }
    let n = elems(idx.shape());
    let out = ctx.scratch_tensor::<f32>(&[n])?;
    ctx.trace_launch("gather");
    ops.note_launch();
    let stream = std::sync::Arc::clone(ctx.stream());
    let rs = src.downcast::<f32>()?;
    let ri = idx.downcast::<u32>()?;
    ops.kernels()
        .gather_f32(&stream, n, rs.device_ptr(), ri.device_ptr(), out.device_ptr())
        .map_err(BackendError::Init)?;
    for t in [src.token(), idx.token(), out.token()] {
        if let Some(t) = t {
            owl_signal::emit(t);
        }
    }
    Ok(DynTensor::from_f32(&out))
}

/// scatter_add(f32;dim0 语义:src [n, F] 按 idx[n] 累加进 out [num, F],out 预清零)
pub fn scatter_add(
    ops: &mut OpsCtx,
    ctx: &KernelCtx,
    num: usize,
    src: &DynTensor<CudaDevice>,
    idx: &DynTensor<CudaDevice>,
) -> Result<DynTensor<CudaDevice>, BackendError> {
    require_f32(src.dtype(), "scatter_add")?;
    if idx.dtype() != Dtype::U32 {
        return Err(BackendError::Init("scatter_add: 索引恒 U32(S4)".into()));
    }
    let f = src.shape().last().copied().ok_or_else(|| BackendError::Init("scatter_add: 空 src".into()))?;
    if elems(idx.shape()) * f != elems(src.shape()) {
        return Err(BackendError::Init("scatter_add: src 行数 ≠ idx 长度".into()));
    }
    let out = ctx.scratch_tensor::<f32>(&[num, f])?;
    ctx.trace_launch("scatter_add");
    ops.note_launch();
    let stream = std::sync::Arc::clone(ctx.stream());
    let rs = src.downcast::<f32>()?;
    let ri = idx.downcast::<u32>()?;
    ops.kernels()
        .scatter_add_f32(&stream, elems(idx.shape()), rs.device_ptr(), ri.device_ptr(), out.device_ptr())
        .map_err(BackendError::Init)?;
    for t in [src.token(), idx.token(), out.token()] {
        if let Some(t) = t {
            owl_signal::emit(t);
        }
    }
    Ok(DynTensor::from_f32(&out))
}

/// 轴归约(keepdim 恒真:out 同 rank,axis 维 = 1)
pub fn sum_dim(
    ops: &mut OpsCtx,
    ctx: &KernelCtx,
    src: &DynTensor<CudaDevice>,
    dim: usize,
) -> Result<DynTensor<CudaDevice>, BackendError> {
    reduce_axis(ops, ctx, src, dim, true, "sum")
}

pub fn max_dim(
    ops: &mut OpsCtx,
    ctx: &KernelCtx,
    src: &DynTensor<CudaDevice>,
    dim: usize,
) -> Result<DynTensor<CudaDevice>, BackendError> {
    reduce_axis(ops, ctx, src, dim, false, "max")
}

#[allow(clippy::too_many_arguments)]
fn reduce_axis(
    ops: &mut OpsCtx,
    ctx: &KernelCtx,
    src: &DynTensor<CudaDevice>,
    dim: usize,
    is_sum: bool,
    what: &str,
) -> Result<DynTensor<CudaDevice>, BackendError> {
    require_f32(src.dtype(), what)?;
    let rank = src.shape().len();
    if dim >= rank {
        return Err(BackendError::Init(format!("{what}: dim 越界 {dim}/{rank}")));
    }
    // 重排到 [outer, axis, inner]:dim 后面的维进 inner
    let inner: usize = src.shape()[dim + 1..].iter().product();
    let outer: usize = src.shape()[..dim].iter().product();
    // dim 之后还有多个维时 inner 含它们;axis = src.shape[dim] ✓
    let axis = src.shape()[dim];
    let mut out_shape = src.shape().to_vec();
    out_shape[dim] = 1;
    let out = ctx.scratch_tensor::<f32>(&out_shape)?;
    ctx.trace_launch("reduce_axis");
    ops.note_launch();
    let stream = std::sync::Arc::clone(ctx.stream());
    let rs = src.downcast::<f32>()?;
    if is_sum {
        ops.kernels()
            .sum_axis_f32(&stream, outer, axis, inner, rs.device_ptr(), out.device_ptr())
    } else {
        ops.kernels()
            .max_axis_f32(&stream, outer, axis, inner, rs.device_ptr(), out.device_ptr())
    }
    .map_err(BackendError::Init)?;
    if let Some(t) = src.token() {
        owl_signal::emit(t);
    }
    if let Some(t) = out.token() {
        owl_signal::emit(t);
    }
    Ok(DynTensor::from_f32(&out))
}

/// 右对齐 broadcast 加(S2:b 任意轴 = 1;前导维必须相等)
pub fn broadcast_add(
    ops: &mut OpsCtx,
    ctx: &KernelCtx,
    a: &DynTensor<CudaDevice>,
    b: &DynTensor<CudaDevice>,
) -> Result<DynTensor<CudaDevice>, BackendError> {
    bcast(ops, ctx, a, b, true)
}

pub fn broadcast_mul(
    ops: &mut OpsCtx,
    ctx: &KernelCtx,
    a: &DynTensor<CudaDevice>,
    b: &DynTensor<CudaDevice>,
) -> Result<DynTensor<CudaDevice>, BackendError> {
    bcast(ops, ctx, a, b, false)
}

fn bcast(
    ops: &mut OpsCtx,
    ctx: &KernelCtx,
    a: &DynTensor<CudaDevice>,
    b: &DynTensor<CudaDevice>,
    is_add: bool,
) -> Result<DynTensor<CudaDevice>, BackendError> {
    require_f32(a.dtype(), "bcast")?;
    require_f32(b.dtype(), "bcast")?;
    let ra = a.shape();
    let rb = b.shape();
    if rb.len() > ra.len() {
        return Err(BackendError::Init("bcast: b rank > a rank(S2 右对齐)".into()));
    }
    let off = ra.len() - rb.len();
    for (i, &bv) in rb.iter().enumerate() {
        let av = ra[off + i];
        if bv != av && bv != 1 {
            return Err(BackendError::Init(format!(
                "bcast: b 维 {bv} 与 a 维 {av} 不兼容(S2)"
            )));
        }
    }
    // 三段分解:outer(前导积)| mid(b 覆盖区,a 各轴)| inner(b 尾部连续 1 段)
    let b_ones = rb.iter().rev().take_while(|&&v| v == 1).count();
    let outer: usize = ra[..off].iter().product();
    let mid: usize = ra[off..ra.len() - b_ones].iter().product::<usize>().max(if ra.len() - b_ones == off { 1 } else { 0 });
    let inner: usize = if b_ones == 0 { 1 } else { rb[rb.len() - b_ones..].iter().product() };
    // b_mid = b 覆盖区各维之积(空 = 1;含 1 轴时模运算自然广播)
    let b_mid: usize = rb[..rb.len() - b_ones].iter().product::<usize>().max(if rb.len() - b_ones == 0 { 1 } else { 0 });
    let out = ctx.scratch_tensor::<f32>(ra)?;
    ctx.trace_launch(if is_add { "add_bcast" } else { "mul_bcast" });
    ops.note_launch();
    let stream = std::sync::Arc::clone(ctx.stream());
    let r_a = a.downcast::<f32>()?;
    let r_b = b.downcast::<f32>()?;
    let k = ops.kernels();
    let res = if is_add {
        k.add_bcast_f32(&stream, outer, mid, inner, b_mid, r_a.device_ptr(), r_b.device_ptr(), out.device_ptr())
    } else {
        k.mul_bcast_f32(&stream, outer, mid, inner, b_mid, r_a.device_ptr(), r_b.device_ptr(), out.device_ptr())
    };
    res.map_err(BackendError::Init)?;
    for t in [a.token(), b.token(), out.token()] {
        if let Some(t) = t {
            owl_signal::emit(t);
        }
    }
    Ok(DynTensor::from_f32(&out))
}

// ============================================================================
// R2:擦除面 → 裸指针 D2D(图边界 logits 直写通道)
// ============================================================================

/// src(DynTensor)前 `bytes` 字节 → 裸目标指针(流上 D2D,捕获安全)。
/// dst 通常 = GraphPlan 租约绑定缓冲的 device_ptr(地址由租约钉住,
/// 此处不 emit——租约在捕获期已登记);src 侧 token 正常 emit。
pub fn copy_d2d_to_raw(
    ctx: &KernelCtx,
    src: &DynTensor<CudaDevice>,
    dst: *mut core::ffi::c_void,
    bytes: usize,
) -> Result<(), BackendError> {
    if bytes > src.len_bytes() {
        return Err(BackendError::CopyFailed {
            dir: "d2d-raw",
            detail: format!("请求 {bytes} B > src 容量 {} B", src.len_bytes()),
        });
    }
    ctx.trace_launch("copy_d2d_to_raw");
    copy_d2d(ctx, dst, src.device_ptr() as *const core::ffi::c_void, bytes)?;
    if let Some(t) = src.token() {
        owl_signal::emit(t);
    }
    Ok(())
}

// ============================================================================
// P1:repeat / repeat_interleave(单维核复合;S1 f32)
// ============================================================================

/// candle `Tensor::repeat` 语义:counts 逐维 tile(右对齐 pad 1;
/// rank 超出 = 结构化报错)。多维 = 单维核逐维复合。
pub fn repeat(
    ops: &mut OpsCtx,
    ctx: &KernelCtx,
    src: &DynTensor<CudaDevice>,
    counts: &[usize],
) -> Result<DynTensor<CudaDevice>, BackendError> {
    require_f32(src.dtype(), "repeat")?;
    let rank = src.shape().len();
    if counts.len() > rank {
        return Err(BackendError::Init(format!(
            "repeat: counts 维数 {} > src rank {rank}(candle 允许前补 1,一期不支持)",
            counts.len()
        )));
    }
    // 右对齐 pad 1
    let mut cnt = vec![1usize; rank];
    cnt[rank - counts.len()..].copy_from_slice(counts);
    if cnt.iter().any(|&c| c == 0) {
        return Err(BackendError::Init("repeat: counts 含 0(candle 同律)".into()));
    }
    let mut cur = src.clone(); // 视图共享,首维 tile 的读取源
    for (d, &c) in cnt.iter().enumerate() {
        if c == 1 {
            continue;
        }
        let outer: usize = cur.shape()[..d].iter().product();
        let d_size = cur.shape()[d];
        let inner: usize = cur.shape()[d + 1..].iter().product();
        let mut out_shape = cur.shape().to_vec();
        out_shape[d] = d_size * c;
        let out = ctx.scratch_tensor::<f32>(&out_shape)?;
        ctx.trace_launch("owl_tile_dim_f32");
        ops.note_launch();
        let rs = cur.downcast::<f32>()?;
        let ro = out.device_ptr();
        ops.kernels()
            .tile_dim_f32(
                &std::sync::Arc::clone(ctx.stream()),
                outer as u64,
                d_size as u64,
                inner as u64,
                c as u64,
                rs.device_ptr(),
                ro,
            )
            .map_err(BackendError::Init)?;
        for t in [src.token(), out.token()] {
            if let Some(t) = t {
                owl_signal::emit(t);
            }
        }
        cur = DynTensor::from_f32(&out);
    }
    Ok(cur)
}

/// candle `Tensor::repeat_interleave(repeats, dim)` 语义(单维;
/// 多维 = 逐维复合,同 repeat)。
pub fn repeat_interleave(
    ops: &mut OpsCtx,
    ctx: &KernelCtx,
    src: &DynTensor<CudaDevice>,
    repeats: usize,
    dim: usize,
) -> Result<DynTensor<CudaDevice>, BackendError> {
    require_f32(src.dtype(), "repeat_interleave")?;
    let rank = src.shape().len();
    if dim >= rank {
        return Err(BackendError::Init(format!(
            "repeat_interleave: dim {dim} 越界(rank={rank})"
        )));
    }
    if repeats == 0 {
        return Err(BackendError::Init("repeat_interleave: repeats=0".into()));
    }
    let outer: usize = src.shape()[..dim].iter().product();
    let d_size = src.shape()[dim];
    let inner: usize = src.shape()[dim + 1..].iter().product();
    let mut out_shape = src.shape().to_vec();
    out_shape[dim] = d_size * repeats;
    let out = ctx.scratch_tensor::<f32>(&out_shape)?;
    ctx.trace_launch("owl_rep_interleave_dim_f32");
    ops.note_launch();
    let rs = src.downcast::<f32>()?;
    let ro = out.device_ptr();
    ops.kernels()
        .rep_interleave_dim_f32(
            &std::sync::Arc::clone(ctx.stream()),
            outer as u64,
            d_size as u64,
            inner as u64,
            repeats as u64,
            rs.device_ptr(),
            ro,
        )
        .map_err(BackendError::Init)?;
    for t in [src.token(), out.token()] {
        if let Some(t) = t {
            owl_signal::emit(t);
        }
    }
    Ok(DynTensor::from_f32(&out))
}

#[cfg(test)]
mod p1_r2_tests {
    use super::*;
    use crate::TensorPoolOps;
    use owl_cuda::test_device_ordinal;
    use owl_iface::{Device, PoolConfig, PoolKind};

    fn setup() -> (OpsCtx, KernelCtx, CudaDevice, owl_cuda::CudaPool) {
        let dev = CudaDevice::new(test_device_ordinal()).expect("需要 CUDA 设备");
        let ops = OpsCtx::new_with_scratch(&dev, 1 << 20).unwrap();
        let ctx = ops.ctx(owl_iface::MemPhase::Live);
        let pool = dev
            .create_pool(PoolConfig {
                name: format!("p1-r2-t-{}", std::process::id()),
                kind: PoolKind::Weights,
                bytes: 8 << 20,
            })
            .unwrap();
        (ops, ctx, dev, pool)
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
    fn repeat_tiles_rows_and_cols() {
        let (mut ops, ctx, _dev, pool) = setup();
        let src = f32_tensor(&_dev, &pool, &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        // dim0 tile ×2:[4,3]
        let r0 = repeat(&mut ops, &ctx, &src, &[2, 1]).unwrap();
        assert_eq!(r0.shape(), &[4, 3]);
        let got = r0.typed_f32().unwrap().to_vec().unwrap();
        assert_eq!(got, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        // last 维 tile ×3:[2,9]
        let r1 = repeat(&mut ops, &ctx, &src, &[1, 3]).unwrap();
        assert_eq!(r1.shape(), &[2, 9]);
        let got = r1.typed_f32().unwrap().to_vec().unwrap();
        assert_eq!(
            got,
            vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn repeat_interleave_last_dim_matches_cpu() {
        let (mut ops, ctx, _dev, pool) = setup();
        let src = f32_tensor(&_dev, &pool, &[1, 4], &[10.0, 20.0, 30.0, 40.0]);
        let out = repeat_interleave(&mut ops, &ctx, &src, 2, 1).unwrap();
        assert_eq!(out.shape(), &[1, 8]);
        let got = out.typed_f32().unwrap().to_vec().unwrap();
        assert_eq!(got, vec![10.0, 10.0, 20.0, 20.0, 30.0, 30.0, 40.0, 40.0]);
    }

    #[test]
    fn index_select_dim0_rows_gather() {
        let (mut ops, ctx, _dev, pool) = setup();
        let src = f32_tensor(&_dev, &pool, &[3, 2], &[1.0, 1.5, 2.0, 2.5, 3.0, 3.5]);
        // idx 设备侧(U32;S4)
        let idx_t = pool.from_vec_tensor::<u32>(&[2], vec![2, 0]).unwrap();
        let idx = DynTensor::from_u32(&idx_t);
        let out = index_select(&mut ops, &ctx, &src, 0, &idx).unwrap();
        assert_eq!(out.shape(), &[2, 2]);
        let got = out.typed_f32().unwrap().to_vec().unwrap();
        assert_eq!(got, vec![3.0, 3.5, 1.0, 1.5]); // rows 2,0
    }

    #[test]
    fn copy_d2d_to_raw_roundtrip() {
        let (_ops, ctx, _dev, pool) = setup();
        let src = f32_tensor(&_dev, &pool, &[4], &[7.0, 8.0, 9.0, 10.0]);
        let dst_t = pool.from_vec_tensor::<f32>(&[4], vec![0.0; 4]).unwrap();
        let dst = DynTensor::from_f32(&dst_t);
        copy_d2d_to_raw(
            &ctx,
            &src,
            dst.device_ptr() as *mut core::ffi::c_void,
            4 * std::mem::size_of::<f32>(),
        )
        .unwrap();
        let got = dst.typed_f32().unwrap().to_vec().unwrap();
        assert_eq!(got, vec![7.0, 8.0, 9.0, 10.0]);
    }
}

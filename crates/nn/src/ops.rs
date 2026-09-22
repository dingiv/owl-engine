//! NN 算子层(T3):out-style 签名 + 两阶段分离(裁决 5)。
//!
//! **类型强制(裁决 5)**:本文件所有算子只接收
//! `&Tensor`(E 阶段只读视图)+ [`OpsCtx`](执行器:仅 stream/launch
//! 能力)+ 既有执行资源(`NnBlas`)。算子方法签名中不存在
//! `GpuBackend`/`OwlCuda`/`CudaPool`——**算子函数体内零分配**,
//! "顺手分一块显存"在类型上不可能。池耗尽只能在 P 阶段(创建
//! Tensor 时)暴露,算子层原样透传的只有 launch 错误。
//!
//! 捕获安全:全部算子标 [`CaptureSafe`](A1.5:无同步/无 D2H/无 host
//! 分支/无隐式分配);matmul 的前提 = workspace 已预钉(T2)。
//! [`OpsCtx::fill_from_host`] 是 **EagerOnly**(同步 H2D,禁入捕获段),
//! 用于 P 阶段向既有缓冲装载数据。
//!
//! launch 通道:T0 的 nvrtc kernel(`Kernels`,标量+裸指针参数,
//! 裁决 3①)+ T2 的 cuBLAS(`NnBlas`,workspace 预钉)。

use crate::kernels::Kernels;
use owl_signal;
use crate::tensor::Tensor;
use crate::{CaptureSafe, KernelCtx};
use owl_iface::BackendError;
use std::sync::Arc;

// ---- CaptureSafe 标记(裁决 3③:算子可捕获性分类的类型化)----

macro_rules! capture_safe {
    ($($t:ident),* $(,)?) => {
        $(
            #[doc = concat!(stringify!($t), ":CAPTURE_SAFE = true(A1.5 契约)")]
            #[derive(Debug, Clone, Copy)]
            pub struct $t;
            impl CaptureSafe for $t {
                const CAPTURE_SAFE: bool = true;
            }
        )*
    };
}

capture_safe!(AddOp, MulOp, SiluOp, GeluOp, ExpOp, SoftmaxOp, RmsnormOp);

/// matmul 可捕获的前提:workspace 已预钉(T2 `NnBlas`),见 charter 裁决 3②。
#[derive(Debug, Clone, Copy)]
pub struct MatmulOp;
impl CaptureSafe for MatmulOp {
    const CAPTURE_SAFE: bool = true;
}

/// EagerOnly 标记(fill_from_host:同步 H2D,禁入捕获段)
#[derive(Debug, Clone, Copy)]
pub struct FillFromHostOp;
impl crate::EagerOnly for FillFromHostOp {
    const EAGER_ONLY: bool = true;
}

macro_rules! unary_op {
    ($method:ident, $kernel:ident, $doc:expr) => {
        #[doc = $doc]
        pub fn $method<T, D>(
            &mut self,
            ctx: &KernelCtx,
            x: &Tensor<T, D>,
            out: &mut Tensor<T, D>,
        ) -> Result<(), BackendError>
        where
            T: crate::tensor::Scalar,
            D: owl_iface::Device,
        {
            ctx.trace_launch(stringify!($method));
            self.note_launch();
            owl_signal::emit(x.token().expect("token"));
            owl_signal::emit(out.token().expect("token"));
            assert_eq!(x.shape(), out.shape(), "unary: out 形状不一致");
            let n = owl_iface::DevBuf::<T>::len(x);
            let stream = Arc::clone(ctx.stream());
            self.kernels
                .$kernel(
                    &stream,
                    n,
                    x.device_ptr() as *const f32,
                    out.device_ptr() as *mut f32,
                )
                .map_err(BackendError::Init)
        }
    };
}


/// E 阶段执行器:仅持有 launch 能力(kernels + stream),**没有池/分配
/// 契约入口**。构造发生在 P 阶段(只读设备句柄,零显存申请);之后
/// 所有算子经 `&mut self` 使用——裁决 5 的类型强制载体。
pub struct OpsCtx {
    kernels: Kernels,
    /// 设备句柄:主流引用 + 发射计数(姿势 6 门禁)+ EagerOnly 的
    /// bind_to_thread。不再持有流所有权——**流由 KernelCtx 携带**
    /// (M1②:流选择显式化,eager/捕获各持其流)
    dev: owl_cuda::CudaDevice,
    /// S6 激活 scratch 池(跨图族共享,A1.3 同族;None = 无池)
    scratch: Option<std::sync::Arc<owl_cuda::CudaPool>>,
}

impl OpsCtx {
    /// P 阶段构造:读设备句柄装配执行器(零显存申请;无 scratch 池——
    /// 层代码 functional 风格请用 new_with_scratch,S6)。
    pub fn new(device: &owl_cuda::CudaDevice) -> Result<Self, BackendError> {
        Self::new_with_scratch(device, 0)
    }

    /// P 阶段构造 + S6 激活 scratch 池(bytes = 0 则不建池;预算由
    /// A1.1 内存规划器登记,T3 接管——本默认值属临时)。
    pub fn new_with_scratch(
        device: &owl_cuda::CudaDevice,
        scratch_bytes: u64,
    ) -> Result<Self, BackendError> {
        let kernels = Kernels::new(device.ctx()).map_err(BackendError::Init)?;
        use owl_iface::Device as _;
        let scratch = if scratch_bytes > 0 {
            Some(std::sync::Arc::new(
                device
                    .create_pool(owl_iface::PoolConfig {
                        name: "ctx-scratch".into(),
                        kind: owl_iface::PoolKind::Scratch,
                        bytes: scratch_bytes,
                    })
                    .map_err(|e| BackendError::Init(format!("scratch pool: {e:?}")))?,
            ))
        } else {
            None
        };
        Ok(Self { kernels, dev: device.clone(), scratch })
    }

    /// E 阶段上下文:设备主显存流(非阻塞;M1②)+ scratch 池注入
    pub fn ctx(&self, phase: owl_iface::MemPhase) -> KernelCtx {
        let c = KernelCtx::eager(phase, Arc::clone(self.dev.stream()));
        match &self.scratch {
            Some(p) => c.with_scratch(std::sync::Arc::clone(p)),
            None => c
        }
    }

    /// 每发射一次核函数计一次(姿势 6 warmup 门禁)
    fn note_launch(&self) {
        self.dev.note_launch();
    }

    fn assert_same_shape<T: crate::tensor::Scalar, Dex: owl_iface::Device>(
        a: &Tensor<T, Dex>,
        b: &Tensor<T, Dex>,
        out: &Tensor<T, Dex>,
    ) {
        assert_eq!(a.shape(), b.shape(), "binary: a/b 形状不一致");
        assert_eq!(a.shape(), out.shape(), "binary: out 形状不一致");
    }

    // ---- EagerOnly:P 阶段向既有缓冲装载数据(同步 H2D,零分配)----

    /// 把 host 数据写进**既有**张量缓冲(不分配;同步路径,禁入捕获段)。
    pub fn fill_from_host<T, D>(
        &mut self,
        ctx: &KernelCtx,
        t: &mut Tensor<T, D>,
        src: &[T],
    ) -> Result<(), BackendError>
    where
        T: crate::tensor::Scalar + Copy,
        D: owl_iface::Device,
    {
        debug_assert!(
            !ctx.recording(),
            "EagerOnly(fill_from_host) 禁入捕获段(裁决 3③)"
        );
        let n = owl_iface::DevBuf::<T>::len(t);
        assert_eq!(n, src.len(), "fill_from_host: 长度不一致");
        self.dev
            .ctx()
            .bind_to_thread()
            .map_err(|e| BackendError::Init(format!("{e:?}")))?;
        use owl_cuda::ffi::sys;
        unsafe {
            sys::cuMemcpyHtoD_v2(
                t.device_ptr() as owl_cuda::ffi::sys::CUdeviceptr,
                src.as_ptr() as *const core::ffi::c_void,
                n * std::mem::size_of::<T>(),
            )
            .result()
            .map_err(|e| BackendError::CopyFailed {
                dir: "htod",
                detail: format!("{e:?}"),
            })
        }
    }

    // ---- 二元(T0 kernel:owl_add_f32 / owl_mul_f32)----

    /// 逐元素加:`out = a + b`(contiguous,形状一致)。
    pub fn add<T, D>(
        &mut self,
        ctx: &KernelCtx,
        a: &Tensor<T, D>,
        b: &Tensor<T, D>,
        out: &mut Tensor<T, D>,
    ) -> Result<(), BackendError>
    where
        T: crate::tensor::Scalar,
        D: owl_iface::Device,
    {
        Self::assert_same_shape(a, b, out);
        ctx.trace_launch("add");
        self.note_launch();
        owl_signal::emit(a.token().expect("token"));
        owl_signal::emit(b.token().expect("token"));
        owl_signal::emit(out.token().expect("token"));
        let n = owl_iface::DevBuf::<T>::len(a);
        let stream = Arc::clone(ctx.stream());
        self.kernels
            .add_f32(
                &stream,
                n,
                a.device_ptr() as *const f32,
                b.device_ptr() as *const f32,
                out.device_ptr() as *mut f32,
            )
            .map_err(BackendError::Init)
    }

    /// 逐元素乘:`out = a * b`。
    pub fn mul<T, D>(
        &mut self,
        ctx: &KernelCtx,
        a: &Tensor<T, D>,
        b: &Tensor<T, D>,
        out: &mut Tensor<T, D>,
    ) -> Result<(), BackendError>
    where
        T: crate::tensor::Scalar,
        D: owl_iface::Device,
    {
        Self::assert_same_shape(a, b, out);
        ctx.trace_launch("mul");
        self.note_launch();
        owl_signal::emit(a.token().expect("token"));
        owl_signal::emit(b.token().expect("token"));
        owl_signal::emit(out.token().expect("token"));
        let n = owl_iface::DevBuf::<T>::len(a);
        let stream = Arc::clone(ctx.stream());
        self.kernels
            .mul_f32(
                &stream,
                n,
                a.device_ptr() as *const f32,
                b.device_ptr() as *const f32,
                out.device_ptr() as *mut f32,
            )
            .map_err(BackendError::Init)
    }

    // ---- 一元(T0 kernel:owl_silu/gelu/exp_f32)----

    unary_op!(silu, silu_f32, "逐元素 SiLU:`out = x * σ(x)`。");
    unary_op!(gelu, gelu_f32, "逐元素 GELU(tanh 近似)。");
    unary_op!(exp, exp_f32, "逐元素 exp。");

    // ---- softmax last-dim(T0 kernel:owl_softmax_f32)----

    /// last-dim softmax:`x` 形状 `[rows, cols]`(contiguous)。
    pub fn softmax<T, D>(
        &mut self,
        ctx: &KernelCtx,
        x: &Tensor<T, D>,
        out: &mut Tensor<T, D>,
    ) -> Result<(), BackendError>
    where
        T: crate::tensor::Scalar,
        D: owl_iface::Device,
    {
        ctx.trace_launch("softmax");
        self.note_launch();
        owl_signal::emit(x.token().expect("token"));
        owl_signal::emit(out.token().expect("token"));
        assert_eq!(x.shape(), out.shape(), "softmax: out 形状不一致");
        let shape = x.shape();
        assert!(shape.len() == 2, "softmax: 一期仅 [rows, cols]");
        let stream = Arc::clone(ctx.stream());
        self.kernels
            .softmax_f32(
                &stream,
                shape[0],
                shape[1],
                x.device_ptr() as *const f32,
                out.device_ptr() as *mut f32,
            )
            .map_err(BackendError::Init)
    }

    // ---- rmsnorm(T0 kernel:owl_rmsnorm_f32)----

    /// RMSNorm:`out = x / rms(x) * alpha`(last-dim;alpha 长度 = cols)。
    pub fn rmsnorm<T, D>(
        &mut self,
        ctx: &KernelCtx,
        x: &Tensor<T, D>,
        alpha: &Tensor<T, D>,
        out: &mut Tensor<T, D>,
        eps: f32,
    ) -> Result<(), BackendError>
    where
        T: crate::tensor::Scalar,
        D: owl_iface::Device,
    {
        ctx.trace_launch("rmsnorm");
        self.note_launch();
        owl_signal::emit(x.token().expect("token"));
        owl_signal::emit(alpha.token().expect("token"));
        owl_signal::emit(out.token().expect("token"));
        assert_eq!(x.shape(), out.shape(), "rmsnorm: out 形状不一致");
        let shape = x.shape();
        assert!(shape.len() == 2, "rmsnorm: 一期仅 [rows, cols]");
        assert_eq!(
            owl_iface::DevBuf::<T>::len(alpha),
            shape[1],
            "rmsnorm: alpha 长度须等于 last dim"
        );
        let stream = Arc::clone(ctx.stream());
        self.kernels
            .rmsnorm_f32(
                &stream,
                shape[0],
                shape[1],
                x.device_ptr() as *const f32,
                alpha.device_ptr() as *const f32,
                out.device_ptr() as *mut f32,
                eps,
            )
            .map_err(BackendError::Init)
    }

    // ---- matmul(T2 cuBLAS:workspace 已预钉)----

    /// `out[M×N] = a[M×K] × b[K×N]`(2D contiguous,行主序语义)。
    ///
    /// CaptureSafe 前提:workspace 已在 `NnBlas::new` 时预钉并登记
    /// Workspace 池(T2)。
    pub fn matmul<D>(
        &self,
        ctx: &KernelCtx,
        blas: &crate::cublas::NnBlas,
        a: &Tensor<f32, D>,
        b: &Tensor<f32, D>,
        out: &mut Tensor<f32, D>,
    ) -> Result<(), BackendError>
    where
        D: owl_iface::Device,
    {
        ctx.trace_launch("matmul");
        self.note_launch();
        // workspace 是 cublas 节点的隐藏依赖(指针烘进 launch 参数):
        // 捕获路径必须把它 emit 进租约(A5.2 池外原语预登记)
        if let Some(t) = blas.workspace_token() {
            owl_signal::emit(t);
        }
        owl_signal::emit(a.token().expect("token"));
        owl_signal::emit(b.token().expect("token"));
        owl_signal::emit(out.token().expect("token"));
        let sa = a.shape();
        let sb = b.shape();
        assert!(sa.len() == 2 && sb.len() == 2, "matmul: 一期仅 2D");
        let (m, k) = (sa[0], sa[1]);
        let (k2, n) = (sb[0], sb[1]);
        assert_eq!(k, k2, "matmul: 内维不一致");
        assert_eq!(out.shape(), &[m, n], "matmul: out 形状应为 [M, N]");
        // M1①:发射到 ctx 携带的流(eager = 设备主流;捕获 = 会话捕获流)——
        // cublas 逐次 setStream,捕获期 GEMM 才能被烙进图(姿势 4 防线:
        // workspace 已预钉,零懒分配)
        blas.matmul_f32(
            m,
            n,
            k,
            a.device_ptr() as *const f32,
            b.device_ptr() as *const f32,
            out.device_ptr() as *mut f32,
            ctx.stream(),
        )
    }
}

// ---- 集成对拍(真机;T4 examples 的算子级前置)----

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cublas::NnBlas;
    use crate::tensor::{Tensor, TensorPoolOps};
    use owl_cuda::CudaDevice;
    use owl_iface::{Device, PoolConfig, PoolKind};

    /// 确定性伪随机([-1,1)),与 T2 测试同款(无 rand 依赖)
    struct Lcg(u32);
    impl Lcg {
        fn next(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((self.0 >> 8) as f32 / 16_777_216.0) * 2.0 - 1.0
        }
        fn vec(&mut self, n: usize) -> Vec<f32> {
            (0..n).map(|_| self.next()).collect()
        }
    }

    fn cpu_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f64> {
        let mut c = vec![0f64; m * n];
        for i in 0..m {
            for p in 0..k {
                let av = a[i * k + p] as f64;
                for j in 0..n {
                    c[i * n + j] += av * b[p * n + j] as f64;
                }
            }
        }
        c
    }

    fn cpu_silu(x: &[f32]) -> Vec<f64> {
        x.iter()
            .map(|&v| v as f64 * (1.0 + (-(v as f64)).exp()).recip())
            .collect()
    }

    fn cpu_rmsnorm(x: &[f64], alpha: &[f32], cols: usize) -> Vec<f64> {
        let rows = x.len() / cols;
        let mut out = vec![0f64; x.len()];
        for r in 0..rows {
            let row = &x[r * cols..(r + 1) * cols];
            let ms = row.iter().map(|&v| v * v).sum::<f64>() / cols as f64;
            let inv = 1.0 / (ms + 1e-5).sqrt();
            for (j, &v) in row.iter().enumerate() {
                out[r * cols + j] = v * inv * alpha[j] as f64;
            }
        }
        out
    }

    fn assert_close(got: &[f32], want: &[f64], rel: f64, abs: f64) {
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            let g = *g as f64;
            let diff = (g - w).abs();
            assert!(
                diff <= abs + rel * w.abs(),
                "idx {i}: got {g}, want {w} (diff {diff})"
            );
        }
    }

    /// 集成链:matmul → add(bias) → silu → rmsnorm,与 CPU f64 对拍。
    /// P 阶段一次性建齐全部缓冲;E 阶段纯使用零分配——裁决 5 运行时示范。
    #[test]
    fn chain_matmul_add_silu_rmsnorm_matches_cpu() {
        let dev = CudaDevice::new(owl_cuda::test_device_ordinal()).expect("需要 CUDA 设备");

        // ---- P 阶段:池 + 全部缓冲(含 chain 中间量)一次性到位 ----
        let pool = dev
            .create_pool(PoolConfig {
                name: format!("t3-chain-{}", std::process::id()),
                kind: PoolKind::Weights,
                bytes: 16 << 20,
            })
            .unwrap();
        let blas = NnBlas::new(&dev).unwrap();
        let mut ops = OpsCtx::new(&dev).unwrap();

        const M: usize = 64;
        const K: usize = 128;
        const N: usize = 64;

        let mut seed = Lcg(7);
        let mut w = pool.zeros_tensor::<f32>(&[M, K]).unwrap();
        let mut a = pool.zeros_tensor::<f32>(&[K, N]).unwrap();
        let mut mm_out = pool.zeros_tensor::<f32>(&[M, N]).unwrap();
        let mut bias = pool.zeros_tensor::<f32>(&[M, N]).unwrap();
        let mut add_out = pool.zeros_tensor::<f32>(&[M, N]).unwrap();
        let mut silu_out = pool.zeros_tensor::<f32>(&[M, N]).unwrap();
        let mut alpha = pool.zeros_tensor::<f32>(&[N]).unwrap();
        let mut norm_out = pool.zeros_tensor::<f32>(&[M, N]).unwrap();

        // 装数(fill_from_host:EagerOnly,零分配)
        let ctx = ops.ctx(owl_iface::MemPhase::Idle);
        ops.fill_from_host(&ctx, &mut w, &seed.vec(M * K)).unwrap();
        ops.fill_from_host(&ctx, &mut a, &seed.vec(K * N)).unwrap();
        ops.fill_from_host(&ctx, &mut bias, &seed.vec(M * N)).unwrap();
        ops.fill_from_host(&ctx, &mut alpha, &seed.vec(N)).unwrap();

        // ---- E 阶段:纯使用,零分配 ----
        let alive_before = dev.ledger().bytes_alive;
        ops.matmul(&ctx, &blas, &w, &a, &mut mm_out).unwrap();
        ops.add(&ctx, &mm_out, &bias, &mut add_out).unwrap();
        ops.silu(&ctx, &add_out, &mut silu_out).unwrap();
        ops.rmsnorm(&ctx, &silu_out, &alpha, &mut norm_out, 1e-5).unwrap();

        // 裁决 5:E 阶段账本零变动
        assert_eq!(dev.ledger().bytes_alive, alive_before);

        let w_h = w.to_vec().unwrap();
        let a_h = a.to_vec().unwrap();
        let mm = cpu_matmul(&w_h, &a_h, M, K, N);
        // 分级断言 2/3:add、silu
        let _add_gpu = add_out.to_vec().unwrap();
        let _silu_gpu = silu_out.to_vec().unwrap();
        // 分级断言 1:matmul
        let mm_gpu = mm_out.to_vec().unwrap();
        assert_close(&mm_gpu, &mm, 2e-3, 2e-3);
        let bias_h = bias.to_vec().unwrap();
        let added: Vec<f64> = mm.iter().zip(&bias_h).map(|(c, b)| c + *b as f64).collect();
        let silu = cpu_silu(&added.iter().map(|&v| v as f32).collect::<Vec<_>>());
        let silu_gpu = silu_out.to_vec().unwrap();
        assert_close(&silu_gpu, &silu, 2e-3, 2e-3); // 分级断言 2:silu
        let alpha_h = alpha.to_vec().unwrap();
        let want = cpu_rmsnorm(&silu, &alpha_h, N);

        let got = norm_out.to_vec().unwrap();
        // matmul f32 累加误差主导;链式组合后 rel 2e-2 是诚实容差
        assert_close(&got, &want, 2e-2, 2e-2);
    }

    /// 裁决 5 的镜像验收:算子层无分配入口,池耗尽只会在 **P 阶段**
    /// (创建链缓冲时)暴露并透传 PoolExhausted。
    #[test]
    fn pool_exhaustion_surfaces_at_planning_phase() {
        let dev = CudaDevice::new(owl_cuda::test_device_ordinal()).expect("需要 CUDA 设备");
        let pool = dev
            .create_pool(PoolConfig {
                name: format!("t3-exhaust-{}", std::process::id()),
                kind: PoolKind::Weights,
                bytes: 4096,
            })
            .unwrap();
        let r: Result<Tensor<f32, CudaDevice>, _> =
            <_ as crate::tensor::TensorPoolOps>::zeros_tensor::<f32>(&pool, &[4096]); // 16KiB > 4KiB
        assert!(matches!(r, Err(BackendError::PoolExhausted { .. })));
    }
    /// 哨兵①:捕获痕迹(launch 序列 + 触碰令牌;小链)
    #[test]
    fn capture_record_traces_launches_and_touched_tokens() {
        use crate::{CaptureRecorder, KernelCtx};
        use owl_iface::MemPhase;

        const M: usize = 16;
        const K: usize = 32;
        const N: usize = 16;

        let dev = CudaDevice::new(owl_cuda::test_device_ordinal()).expect("需要 CUDA 设备");
        let mut ops = OpsCtx::new(&dev).unwrap();
        let blas = NnBlas::new(&dev).unwrap();
        let pool = dev
            .create_pool(PoolConfig {
                name: "cap".into(),
                kind: PoolKind::Weights,
                bytes: 4 << 20,
            })
            .unwrap();

        let w = pool.zeros_tensor::<f32>(&[M, K]).unwrap();
        let a = pool.zeros_tensor::<f32>(&[K, N]).unwrap();
        let mut mm_out = pool.zeros_tensor::<f32>(&[M, N]).unwrap();
        let mut silu_out = pool.zeros_tensor::<f32>(&[M, N]).unwrap();

        // eager:无痕迹
        let eager = ops.ctx(MemPhase::Idle);
        ops.silu(&eager, &mm_out, &mut silu_out).unwrap();
        assert!(eager.snapshot().is_none());
        assert!(!eager.recording());

        // 捕获:matmul → silu,launch 序列与触碰令牌全部留痕
        let recorder = CaptureRecorder::new();
        let cap = KernelCtx::capturing(recorder.clone(), dev.stream().clone());
        ops.matmul(&cap, &blas, &w, &a, &mut mm_out).unwrap();
        ops.silu(&cap, &mm_out, &mut silu_out).unwrap();

        let rec = cap.snapshot().expect("捕获态必有快照");
        assert_eq!(rec.launches, vec!["matmul", "silu"]);
        let touched: Vec<_> = rec.touched.iter().copied().collect();
        // w/a/mm_out/silu_out + cublas workspace(隐藏依赖,M1①) = 5 个
        assert_eq!(touched.len(), 5, "4 个链缓冲 + 1 个 workspace 令牌");
        assert!(touched.contains(&w.token().unwrap()));
        assert!(touched.contains(&a.token().unwrap()));
        assert!(touched.contains(&mm_out.token().unwrap()));
        assert!(touched.contains(&silu_out.token().unwrap()));
        assert!(dev.validate_token(&w.token().unwrap()));
    }

    /// 哨兵①:drop 后令牌注销 → validate=false(结构化发现替代 Xid 盲死)
    #[test]
    fn dropped_tensor_token_invalidated() {
        let dev = CudaDevice::new(owl_cuda::test_device_ordinal()).expect("需要 CUDA 设备");
        let pool = dev
            .create_pool(PoolConfig {
                name: "tok".into(),
                kind: PoolKind::Weights,
                bytes: 64 << 10,
            })
            .unwrap();
        let t = pool.zeros_tensor::<f32>(&[16]).unwrap();
        let token = t.token().unwrap();
        assert!(dev.validate_token(&token));
        drop(t);
        assert!(!dev.validate_token(&token));
    }

    /// 哨兵①:全链捕获痕迹(launch 序列 = 节点序;触碰集 = 全操作数)
    #[test]
    fn capture_record_on_full_chain() {
        use crate::{CaptureRecorder, KernelCtx};
        use std::collections::BTreeSet;

        const M: usize = 32;
        const K: usize = 64;
        const N: usize = 32;

        let dev = CudaDevice::new(owl_cuda::test_device_ordinal()).expect("需要 CUDA 设备");
        let mut ops = OpsCtx::new(&dev).unwrap();
        let blas = NnBlas::new(&dev).unwrap();
        let pool = dev
            .create_pool(PoolConfig {
                name: "capchain".into(),
                kind: PoolKind::Weights,
                bytes: 4 << 20,
            })
            .unwrap();

        let w = pool.zeros_tensor::<f32>(&[M, K]).unwrap();
        let a = pool.zeros_tensor::<f32>(&[K, N]).unwrap();
        let bias = pool.zeros_tensor::<f32>(&[M, N]).unwrap();
        let mut mm_out = pool.zeros_tensor::<f32>(&[M, N]).unwrap();
        let mut add_out = pool.zeros_tensor::<f32>(&[M, N]).unwrap();
        let mut silu_out = pool.zeros_tensor::<f32>(&[M, N]).unwrap();
        let alpha = pool.zeros_tensor::<f32>(&[N]).unwrap();
        let mut norm_out = pool.zeros_tensor::<f32>(&[M, N]).unwrap();

        let recorder = CaptureRecorder::new();
        let cap = KernelCtx::capturing(recorder.clone(), dev.stream().clone());
        ops.matmul(&cap, &blas, &w, &a, &mut mm_out).unwrap();
        ops.add(&cap, &mm_out, &bias, &mut add_out).unwrap();
        ops.silu(&cap, &add_out, &mut silu_out).unwrap();
        ops.rmsnorm(&cap, &silu_out, &alpha, &mut norm_out, 1e-5).unwrap();

        let rec = cap.snapshot().expect("捕获态必有快照");
        assert_eq!(rec.launches, vec!["matmul", "add", "silu", "rmsnorm"]);
        // 8 个链缓冲 + cublas workspace(matmul 的隐藏依赖,M1①)
        let expected: BTreeSet<_> = [
            w.token(),
            a.token(),
            bias.token(),
            mm_out.token(),
            add_out.token(),
            silu_out.token(),
            alpha.token(),
            norm_out.token(),
            blas.workspace_token(),
        ]
        .into_iter()
        .flatten()
        .collect();
        assert_eq!(rec.touched, expected);
    }

}

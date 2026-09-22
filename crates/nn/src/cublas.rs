//! cuBLAS matmul + BlasWorkspace 预钉(T2)。
//!
//! ported from: repos/candle-gb/candle-core/src/cuda_backend/device.rs
//! `BlasWorkspace`(约 25-110 行)。
//!
//! candle 焊缝的语义:workspace 若在图捕获期间由 cuBLAS 懒分配,会成为
//! graph-owned 内存节点,在 AUTO_FREE 语义池下 replay 后地址被回收 →
//! split-K 写坏无关张量。candle 的解法:启动时 `malloc_sync` 池外预分配
//! 32MiB 常驻,`cublasSetWorkspace` 钉死("mirrors PyTorch")。
//!
//! owl 的移植(A5.2 化):预分配本体**经 OwlCuda 账本与 Workspace 池**
//! (`alloc_persistent_in`),发生在任何捕获之前、生命周期内永不释放——
//! "池外预钉"的不变量(稳定地址 + 捕获期零懒分配)由"预登记"等效满足。

use owl_cuda::ffi::sys::cublas::{
    cublasCreate_v2, cublasDestroy_v2, cublasSetStream_v2, cublasSetWorkspace_v2,
    cublasSgemm_v2, cublasHandle_t, cublasOperation_t, cublasStatus_t,
};
use owl_cuda::{CudaDevice, Persistent};

/// workspace 大小:32 MiB,与 candle 一致(覆盖 decode 稀瘦 matmul 的
/// split-K 归约上限)。
pub const BLAS_WORKSPACE_BYTES: usize = 32 * 1024 * 1024;

/// port 自 candle `BlasWorkspace`:池外预钉的 cuBLAS workspace。
/// owl 化:分配经账本(Workspace 池),缓冲句柄保活 = 地址稳定。
pub struct BlasWorkspace {
    buf: Persistent<u8>,
}

impl BlasWorkspace {
    pub fn new(device: &CudaDevice) -> Result<Self, BackendError> {
        let pool = device.create_pool(PoolConfig {
            name: "workspace".into(),
            kind: PoolKind::Workspace,
            bytes: BLAS_WORKSPACE_BYTES as u64,
        })?;
        let buf = device.alloc_persistent_in::<u8>(&pool, BLAS_WORKSPACE_BYTES)?;
        Ok(Self { buf })
    }

    fn raw_ptr(&self) -> *mut core::ffi::c_void {
        use owl_iface::DevBuf;
        self.buf.device_ptr() as *mut core::ffi::c_void
    }

    /// 哨兵①:workspace 缓冲令牌。cublas 把 workspace 指针烘进 launch
    /// 参数,捕获期它就是图的真实依赖——A5.2"池外原语必须预分配
    /// 并登记"的执行点(matmul 捕获路径必须把它 emit 进租约)。
    pub fn token(&self) -> Option<owl_iface::BufToken> {
        self.buf.token()
    }
}

/// 绑定 workspace 的 cuBLAS 句柄(handle 的销毁先于 workspace 释放,
/// 见 Drop 实现的字段顺序)。
pub struct NnBlas {
    ctx: std::sync::Arc<owl_cuda::ffi::CudaContext>,
    /// 设备句柄:姿势 6 发射计数(matmul 每次发射 note_launch)
    dev: owl_cuda::CudaDevice,
    handle: Option<cublasHandle_t>,
    ws: BlasWorkspace,
}

impl NnBlas {
    /// workspace 令牌(捕获路径 emit 进租约,见 [`BlasWorkspace::token`])
    pub fn workspace_token(&self) -> Option<owl_iface::BufToken> {
        self.ws.token()
    }
}

fn check(st: cublasStatus_t, what: &'static str) -> Result<(), BackendError> {
    if st != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        return Err(BackendError::Init(format!("{what}: {st:?}")));
    }
    Ok(())
}

impl NnBlas {
    pub fn new(device: &CudaDevice) -> Result<Self, BackendError> {
        // cublasCreate 需要当前线程绑定目标 context
        device
            .ctx()
            .bind_to_thread()
            .map_err(|e| BackendError::Init(format!("bind_to_thread: {e:?}")))?;

        let ws = BlasWorkspace::new(device)?;
        let mut handle: cublasHandle_t = std::ptr::null_mut();
        unsafe { check(cublasCreate_v2(&mut handle), "cublasCreate_v2")? };
        // 焊缝核心:首次 matmul 之前钉死 workspace,捕获期零懒分配
        unsafe {
            check(
                cublasSetWorkspace_v2(handle, ws.raw_ptr(), BLAS_WORKSPACE_BYTES),
                "cublasSetWorkspace_v2",
            )?;
        }
        // M1①:绑设备主显存流(非阻塞、可捕获)。legacy NULL 流不可捕获
        // (T4 判例);matmul 逐次 setStream 到实际发射流(见 matmul_f32)
        // driver CUstream 与 cublas cudaStream_t 是两套不透明类型:经
        // c_void 转接(两者同为裸句柄,零拷贝)
        let s = device.stream().cu_stream() as *mut core::ffi::c_void;
        unsafe {
            check(
                cublasSetStream_v2(handle, s as _),
                "cublasSetStream_v2",
            )?;
        }
        Ok(Self {
            ctx: std::sync::Arc::clone(device.ctx()),
            dev: device.clone(),
            handle: Some(handle),
            ws,
        })
    }

    /// row-major 语义的 C[M×N] = A[M×K] × B[K×N]。
    ///
    /// cublas 列主序转换(经典技巧):行主序内存按列主序重解释时,
    /// A ≡ Aᵀ(K×M)、B ≡ Bᵀ(N×K)、C ≡ Cᵀ(N×M),故
    /// `Cᵀ = B × A`(全部不转置),即 cublas 侧 (m=N, n=M, k=K),
    /// 首参传 B(lda=N)、次参传 A(ldb=K)、C ldc=N。
    /// `stream`:发射目标流(eager = 设备主流;捕获 = 会话捕获流)。
    /// cublasSetStream 是 host 侧句柄配置(不触碰流),捕获期安全。
    pub fn matmul_f32(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        stream: &owl_cuda::ffi::CudaStream,
    ) -> Result<(), BackendError> {
        self.dev.note_launch();
        self.ctx
            .bind_to_thread()
            .map_err(|e| BackendError::Init(format!("bind_to_thread: {e:?}")))?;
        let handle = self.handle.expect("NnBlas handle");
        let (m, n, k) = (m as i32, n as i32, k as i32);
        let alpha = 1.0f32;
        let beta = 0.0f32;
        let s = stream.cu_stream() as *mut core::ffi::c_void;
        unsafe {
            check(
                cublasSetStream_v2(handle, s as _),
                "cublasSetStream_v2",
            )?;
            check(
                cublasSgemm_v2(
                    handle,
                    cublasOperation_t::CUBLAS_OP_N,
                    cublasOperation_t::CUBLAS_OP_N,
                    n, // cublas m = 语义 N
                    m, // cublas n = 语义 M
                    k,
                    &alpha,
                    b,    // 首参 = 语义 B(列主序视角 N×K)
                    n,    // lda = N
                    a,    // 次参 = 语义 A(列主序视角 K×M)
                    k,    // ldb = K
                    &beta,
                    c, // C 列主序视角 N×M = 语义 C[M×N]
                    n, // ldc = N
                ),
                "cublasSgemm_v2",
            )?;
        }
        Ok(())
    }
}

impl Drop for NnBlas {
    fn drop(&mut self) {
        // handle 先销毁,再让 ws(账本缓冲)随后 drop——字段声明顺序保证
        if let Some(handle) = self.handle.take() {
            unsafe {
                let _ = cublasDestroy_v2(handle);
            }
        }
    }
}

use owl_iface::{BackendError, Device, PoolConfig, PoolKind};

#[cfg(test)]
mod tests {
    use super::*;
    use owl_iface::DevBuf;

    /// 确定性伪随机([-1,1)),无 rand 依赖
    struct Lcg(u32);
    impl Lcg {
        fn next(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((self.0 >> 8) as f32 / 16_777_216.0) * 2.0 - 1.0
        }
    }

    fn setup(
        dev: &CudaDevice,
    ) -> Result<(NnBlas, owl_cuda::CudaPool), BackendError> {
        let blas = NnBlas::new(dev)?;
        let pool = dev.create_pool(PoolConfig {
            name: format!("nn-test-{}", std::process::id()),
            kind: PoolKind::Weights,
            bytes: 64 << 20,
        })?;
        Ok((blas, pool))
    }

    /// f32 matmul 对拍 CPU f64 参考(裁决 4:rtol/atol = 1e-4)
    #[test]
    fn matmul_f32_matches_cpu_reference() {
        let dev = CudaDevice::new(owl_cuda::test_device_ordinal()).expect("需要 CUDA 设备");
        let stream = dev.stream().clone();
        let (blas, pool) = setup(&dev).expect("setup");

        const M: usize = 128;
        const K: usize = 256;
        const N: usize = 512;
        let mut seed = Lcg(42);
        let a: Vec<f32> = (0..M * K).map(|_| seed.next()).collect();
        let b: Vec<f32> = (0..K * N).map(|_| seed.next()).collect();

        let da = dev.htod_persistent_in::<f32>(&pool, a.clone()).unwrap();
        let db = dev.htod_persistent_in::<f32>(&pool, b.clone()).unwrap();
        let dc = dev.alloc_persistent_in::<f32>(&pool, M * N).unwrap();
        dev.ctx().synchronize().unwrap();

        blas
            .matmul_f32(
                M,
                N,
                K,
                da.device_ptr(),
                db.device_ptr(),
                dc.device_ptr(),
                &stream,
            )
            .unwrap();
        dev.ctx().synchronize().unwrap();

        let mut host_c = vec![0f32; M * N];
        unsafe {
            owl_cuda::ffi::memcpy_dtoh_sync(
                &mut host_c,
                dc.device_ptr() as owl_cuda::ffi::sys::CUdeviceptr,
            )
            .expect("dtoh");
        }

        // CPU f64 参考
        let mut reference = vec![0f64; M * N];
        for i in 0..M {
            for kk in 0..K {
                let av = a[i * K + kk] as f64;
                for j in 0..N {
                    reference[i * N + j] += av * b[kk * N + j] as f64;
                }
            }
        }
        let mut worst = 0.0f64;
        for (i, (got, want)) in host_c.iter().zip(reference.iter()).enumerate() {
            let got = *got as f64;
            let diff = (got - want).abs();
            let tol = 1e-4 + 1e-4 * want.abs();
            assert!(
                diff <= tol,
                "element {i}: got {got}, want {want}, diff {diff}"
            );
            worst = worst.max(diff);
        }
        eprintln!("[T2] matmul 128x256x512 最差绝对误差 = {worst:.3e}");
    }

    /// 同一 handle 连续 1000 次 matmul:账本零漂移(A5.3)
    #[test]
    fn repeated_matmul_no_ledger_drift() {
        let dev = CudaDevice::new(owl_cuda::test_device_ordinal()).expect("需要 CUDA 设备");
        let (blas, pool) = setup(&dev).expect("setup");

        const M: usize = 64;
        const K: usize = 64;
        const N: usize = 64;
        let da = dev
            .htod_persistent_in::<f32>(&pool, vec![0.5f32; M * K])
            .unwrap();
        let db = dev
            .htod_persistent_in::<f32>(&pool, vec![0.25f32; K * N])
            .unwrap();
        let dc = dev.alloc_persistent_in::<f32>(&pool, M * N).unwrap();
        dev.ctx().synchronize().unwrap();

        let stream = dev.stream().clone();
        let before = dev.ledger();
        for _ in 0..1000 {
            blas.matmul_f32(
                M,
                N,
                K,
                da.device_ptr(),
                db.device_ptr(),
                dc.device_ptr(),
                &stream,
            )
            .unwrap();
        }
        dev.ctx().synchronize().unwrap();
        let after = dev.ledger();

        assert_eq!(after.bytes_alive, before.bytes_alive, "A5.3: 账本漂移");
        assert_eq!(
            after.bytes_allocated_total, before.bytes_allocated_total,
            "1000 次 matmul 不应有任何新分配"
        );
        eprintln!(
            "[T2] 1000 matmul 稳定:alive={}B total={}B",
            after.bytes_alive, after.bytes_allocated_total
        );
    }
}

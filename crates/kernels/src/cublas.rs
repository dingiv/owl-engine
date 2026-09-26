//! cuBLAS 封装(f16 基线 F1;2026-09-26 自 backends/cuda 下沉至此)。
//!
//! 算子之家统一口径:外部算子库(cuBLAS/将来 cublasLt/FlashInfer)的
//! 封装住本 crate,server(backends/cuda)退回纯传输 —— 命令面
//! (`Command::Gemm`)不变,执行策略归算子层。
//!
//! 定盘:
//! - **f16 入出 + `CUBLAS_COMPUTE_32F` 累计**(f16 累计精度不够,禁用);
//! - 行主序映射(owl Linear 惯例 B [m,k] 行主序权重):
//!   `C_rm[T,n] = A[T,k] × W[n,k]^T` ⇔ 列主序 `gemm_ex(opA=T on W,
//!   lda=k)`;
//! - 懒初始化交调用方(第一次 gemm 前建句柄);句柄绑 COMPUTE 流,
//!   与 kernel 发射同流保序;
//! - **账外显存已知项**:workspace 池热稳态 ≈ 50 MiB(实测,见
//!   backends/cuda/README.md 显存记账;接 capture 前必须 warmup)。
//!
//! 层级纪律:本模块不依赖 owl-iface(错误 = String,server 侧映射
//! ModelError);裸指针入参(u64 设备地址),零 cudarc driver 泛型面。

use cudarc::cublas::{result as cb, sys, CudaBlas};
use std::sync::Arc;

/// 外部算子名(foreign-kernel 通道):Launch 的核名以此标识外部库执行,
/// server 按名分派到本模块;不进 nvrtc 注册表,无 .cu 源。
///
/// **槽序契约**(LaunchMsg.args,与 .cu 核同构 —— 输出块固定末参前的
/// Block 序 + 类型化标量):
/// `[T a, T b, T out, sz m, sz k, sz n, sz nt]`(nt: 1 = owl Linear nt 形)。
pub const GEMM_F16: &str = "cublas_gemm_f16";

/// foreign 分派谓词(server handle_launch 前置检查)
pub fn is_foreign(name: &str) -> bool {
    name == GEMM_F16
}

pub struct OwlCublas {
    blas: CudaBlas,
}

impl OwlCublas {
    /// 句柄创建并绑流(cublasSetStream;GEMM 与该流上 kernel 同序)
    pub fn new(stream: Arc<cudarc::driver::CudaStream>) -> Result<Self, String> {
        CudaBlas::new(stream).map(|blas| Self { blas }).map_err(|e| format!("cublas init: {e:?}"))
    }

    /// f16 GEMM:nt=true(owl Linear)C[T,n] = A[T,k]×W[m,k]^T;
    /// nt=false:C[T,n] = A[T,k]×B[k,n]。A/B/out = 设备地址(u64)。
    /// fire-and-forget(排队即返回;COMPUTE 流保序)。
    pub fn gemm_f16(
        &self,
        a: u64,
        b: u64,
        out: u64,
        m: usize,
        k: usize,
        n: usize,
        nt: bool,
    ) -> Result<(), String> {
        // 行主序 → 列主序映射:out_cm[m, n] = op(b) · op(a)
        // nt=true:B=W [m,k] rm = cm [k,m],transa=T(lda=k)→ [m,k];
        // nt=false:B [k,n] rm = cm [n,k],transa=T(lda=n)→ [k,n]。
        let (lda, b_ptr) = if nt { (k as i32, b) } else { (n as i32, b) };
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let r = unsafe {
            cb::gemm_ex(
                *self.blas.handle(),
                sys::cublasOperation_t::CUBLAS_OP_T,
                sys::cublasOperation_t::CUBLAS_OP_N,
                m as i32,
                n as i32,
                k as i32,
                &alpha as *const f32 as *const _,
                b_ptr as *const _,
                sys::cudaDataType::CUDA_R_16F,
                lda,
                a as *const _,
                sys::cudaDataType::CUDA_R_16F,
                k as i32,
                &beta as *const f32 as *const _,
                out as *mut _,
                sys::cudaDataType::CUDA_R_16F,
                m as i32,
                sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
        };
        r.map_err(|e| format!("gemm_ex: {e:?}"))
    }
}

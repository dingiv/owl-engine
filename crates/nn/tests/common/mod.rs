//! CPU 参考算子库(对拍基建,T5)。
//!
//! 全部参考实现使用 **f64 累加**,与 GPU 端 f32/f16 计算对拍时提供
//! 高精度基准。语义对齐 `crates/nn/kernels/cu/owl_nn_kernels.cu`
//! (每函数注明对齐的 kernel/出处)。
//!
//! 用法:
//! ```ignore
//! mod common;
//! use common::{Lcg, ref_matmul, assert_allclose};
//! ```

/// 确定性伪随机([-1,1)),与 src 内联测试同款参数(无 rand 依赖)
pub struct Lcg(u32);

impl Lcg {
    pub fn new(seed: u32) -> Self {
        Self(seed)
    }

    pub fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((self.0 >> 8) as f32 / 16_777_216.0) * 2.0 - 1.0
    }

    pub fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next()).collect()
    }
}

/// 函数版断言(集成测试直接调用)
pub fn assert_allclose_f32(
    actual: &[f32],
    expected: &[f64],
    rtol: f64,
    atol: f64,
    label: &str,
) {
    assert_eq!(actual.len(), expected.len(), "[{l}] 长度不一致", l = label);
    let mut worst: (usize, f64) = (0, 0.0);
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let d = (a as f64 - e).abs();
        let tol = atol + rtol * e.abs();
        if d > tol && d > worst.1 {
            worst = (i, d);
        }
    }
    assert!(
        worst.1 == 0.0,
        "[{l}] 对拍失败:最大偏差 {d:.3e} @ {i}(rtol={r}, atol={a})",
        l = label,
        d = worst.1,
        i = worst.0,
        r = rtol,
        a = atol,
    );
}

/// 宏版(可选;同函数语义)
#[macro_export]
macro_rules! assert_allclose {
    ($actual:expr, $expected:expr, $rtol:expr, $atol:expr, $label:expr) => {
        $crate::__private_assert_allclose($actual, $expected, $rtol, $atol, $label)
    };
}

// ---- 参考算子(f64 累加)----

/// port 对齐:cublas Sgemm(行主序 C[M,N] = A[M,K] × B[K,N])
pub fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f64> {
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

/// 逐元素加(b.len()==n 时按行广播,否则同形逐元素)
pub fn add(a: &[f32], b: &[f32], m: usize, n: usize) -> Vec<f64> {
    let _ = m;
    if b.len() == n {
        (0..a.len())
            .map(|i| a[i] as f64 + b[i % n] as f64)
            .collect()
    } else {
        assert_eq!(a.len(), b.len(), "add:同形或按行广播二选一");
        (0..a.len()).map(|i| a[i] as f64 + b[i] as f64).collect()
    }
}

/// 逐元素乘
pub fn mul(a: &[f32], b: &[f32]) -> Vec<f64> {
    debug_assert_eq!(a.len(), b.len());
    (0..a.len()).map(|i| a[i] as f64 * b[i] as f64).collect()
}

/// port 对齐:owl_nn_kernels.cu owl_silu_f32
pub fn silu(x: &[f32]) -> Vec<f64> {
    x.iter()
        .map(|&v| v as f64 * (1.0 + (-(v as f64)).exp()).recip())
        .collect()
}

/// port 对齐:owl_nn_kernels.cu owl_exp_f32
pub fn exp(x: &[f32]) -> Vec<f64> {
    x.iter().map(|&v| (v as f64).exp()).collect()
}

/// port 对齐:owl_nn_kernels.cu owl_gelu_f32(tanh 近似,candle unary.cu gelu_fwd)
pub fn gelu(x: &[f32]) -> Vec<f64> {
    x.iter()
        .map(|&v| {
            let x = v as f64;
            0.5 * x * (1.0 + (0.797_884_560_802_865_4 * (x + 0.044_715 * x * x * x)).tanh())
        })
        .collect()
}

/// softmax last-dim 数值稳定版(减行最大值)
pub fn softmax_last_dim(x: &[f32], cols: usize) -> Vec<f64> {
    let rows = x.len() / cols;
    let mut out = vec![0f64; x.len()];
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let sum: f64 = row.iter().map(|&v| (v as f64 - max).exp()).sum();
        for (j, &v) in row.iter().enumerate() {
            out[r * cols + j] = (v as f64 - max).exp() / sum;
        }
    }
    out
}

/// port 对齐:owl_nn_kernels.cu owl_rmsnorm_f32(eps 默认 1e-5)
pub fn rmsnorm(x: &[f64], alpha: &[f32], cols: usize, eps: f64) -> Vec<f64> {
    let rows = x.len() / cols;
    let mut out = vec![0f64; x.len()];
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let ms = row.iter().map(|&v| v * v).sum::<f64>() / cols as f64;
        let inv = 1.0 / (ms + eps).sqrt();
        for (j, &v) in row.iter().enumerate() {
            out[r * cols + j] = v * inv * alpha[j] as f64;
        }
    }
    out
}

// ---- M3 预留:transformer block 级参考组装 ----

/// 单层 block 形状参数(M3 填充真实值)
#[allow(dead_code)] // M3 预留:签名先冻结,防各测试自造参考导致口径分叉
pub struct BlockCfg {
    pub hidden: usize,
    pub inter: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub eps: f64,
}

/// Transformer block 级参考(M3 预留:届时按模型结构组装
/// rmsnorm→attention→residual→rmsnorm→mlp→residual 的 f64 参考链,
/// 与 GPU 端整层对拍)。签名先冻结,防止各测试自造参考导致口径分叉。
#[allow(dead_code)] // M3 预留(同上)
pub fn transformer_block_ref(
    _x: &[f32],
    _attn_w: &[f32],
    _mlp_w: &[f32],
    _cfg: &BlockCfg,
) -> Vec<f64> {
    unimplemented!("M3:按模型结构组装 f64 参考链(REQ-DESIGN:参考实现单一来源)")
}

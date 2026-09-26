//! 朴素算子实现(CPU 后端的 launch 执行体;自 models CpuInterpreter 提炼)。
//!
//! ⚠️ **对拍锚纪律**:本文件与 `owl_models::reference::CpuInterpreter`
//! 的算子实现**互为独立副本,禁止互相引用**——models 侧参考实现的存在
//! 意义就是对拍后端,共享代码会让对拍失效。两侧语义必须一致
//! (行主序 / k = a.len()/m / rmsnorm ×(1+w) 语义),数值路径允许分化
//! (本侧将来可优化:SIMD/rayon;参考侧永不优化)。

use crate::value::Value;
use owl_iface::contract::{Dtype, ModelError, Shape};

/// host 字节 → 值块(装载)
pub(crate) fn htod(dtype: Dtype, shape: &Shape, bytes: &[u8]) -> Result<Value, ModelError> {
    Value::from_bytes(dtype, shape, bytes)
}

/// 清零分配
pub(crate) fn zeros(dtype: Dtype, shape: &Shape) -> Result<Value, ModelError> {
    Value::zero(dtype, shape)
}

/// [m,k] × [k,n](行主序;m/k/n = launch 标量权威 —— 2026-09-26 修:
/// 原 k 自推 a.shape[0],alloc 块无形状(平面 [len])时 k 被压成 1,
/// 被 engine M0 session 测试逮到;块账长只作容量,维度不参与语义)
pub(crate) fn matmul(a: &Value, b: &Value, m: usize, k: usize, n: usize) -> Result<Value, ModelError> {
    debug_assert_eq!(a.f32.len(), m * k, "matmul: a 账长 = m×k");
    debug_assert_eq!(b.f32.len(), k * n, "matmul: b 账长 = k×n");
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for p in 0..k {
            let av = a.f32[i * k + p];
            for j in 0..n {
                out[i * n + j] += av * b.f32[p * n + j];
            }
        }
    }
    Ok(Value { f32: out, shape: vec![m, n] })
}

pub(crate) fn matmul_nt(a: &Value, b: &Value, m: usize, k: usize, n: usize) -> Result<Value, ModelError> {
    // B 按 [n, k] 行主序直读(nt;与 owl_matmul_nt_f32 同式)
    // m/k/n = launch 标量权威(同上修:alloc 块无形状,k 自推不可靠)
    debug_assert_eq!(a.f32.len(), m * k, "matmul_nt: a 账长 = m×k");
    debug_assert_eq!(b.f32.len(), n * k, "matmul_nt: b 账长 = n×k");
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let acc: f32 = a.f32[i * k..(i + 1) * k]
                .iter()
                .zip(&b.f32[j * k..(j + 1) * k])
                .map(|(x, y)| x * y)
                .sum();
            out[i * n + j] = acc;
        }
    }
    Ok(Value { f32: out, shape: vec![m, n] })
}

/// 同形逐元素加
pub(crate) fn add(a: &Value, b: &Value) -> Result<Value, ModelError> {
    Ok(Value {
        f32: a.f32.iter().zip(&b.f32).map(|(x, y)| x + y).collect(),
        shape: a.shape.clone(),
    })
}

/// 同形逐元素乘
pub(crate) fn mul(a: &Value, b: &Value) -> Result<Value, ModelError> {
    Ok(Value {
        f32: a.f32.iter().zip(&b.f32).map(|(x, y)| x * y).collect(),
        shape: a.shape.clone(),
    })
}

/// 逐元素 SiLU
pub(crate) fn silu(x: &Value) -> Result<Value, ModelError> {
    Ok(Value {
        f32: x.f32.iter().map(|v| v / (1.0 + (-v).exp())).collect(),
        shape: x.shape.clone(),
    })
}

/// 逐元素 sigmoid(attn_output_gate 门;GDN beta 同族)
pub(crate) fn sigmoid(x: &Value) -> Result<Value, ModelError> {
    Ok(Value {
        f32: x.f32.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect(),
        shape: x.shape.clone(),
    })
}

/// RMSNorm(per-channel alpha([cols] 广播,与 GPU owl_rmsnorm_f32 同一语义);w_off = ×(1+w))
pub(crate) fn rmsnorm(
    x: &Value,
    alpha: &Value,
    cols: usize,
    eps: f32,
    w_off: bool,
) -> Result<Value, ModelError> {
    let cols = cols.max(1);
    let rows = x.f32.len() / cols;
    let mut out = vec![0.0f32; x.f32.len()];
    for r in 0..rows {
        let xs = &x.f32[r * cols..(r + 1) * cols];
        let ms = xs.iter().map(|v| v * v).sum::<f32>() / cols as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        for (c, v) in xs.iter().enumerate() {
            let g = if w_off { alpha.f32[c] + 1.0 } else { alpha.f32[c] };
            out[r * cols + c] = v * inv * g;
        }
    }
    Ok(Value { f32: out, shape: x.shape.clone() })
}

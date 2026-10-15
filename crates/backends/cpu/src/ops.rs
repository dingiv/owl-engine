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


/// 旋转位置编码(rotate-half;表驱动):x [tokens, heads*head_dim] × cos/sin [rows, half],
/// pos[t] 选表行;前 2·half 维旋转,余直通(partial)。与 `owl_rope_half_partial_f32` 同式。
pub(crate) fn rope_half_partial(
    x: &Value,
    cos_t: &Value,
    sin_t: &Value,
    pos: &Value,
    heads: usize,
    head_dim: usize,
    half: usize,
) -> Result<Value, ModelError> {
    let row_len = heads * head_dim;
    if row_len == 0 || x.f32.len() % row_len != 0 {
        return Err(ModelError::Msg(format!(
            "owl_rope_half_partial: x 长 {} 不是 heads·head_dim = {row_len} 的整数倍",
            x.f32.len()
        )));
    }
    if cos_t.f32.len() % half.max(1) != 0 || 2 * half > head_dim {
        return Err(ModelError::Msg(format!(
            "owl_rope_half_partial: cos 长 {} / half {half} / head_dim {head_dim} 不齐",
            cos_t.f32.len()
        )));
    }
    let rows = x.f32.len() / row_len;
    let table_rows = cos_t.f32.len() / half;
    let mut out = x.f32.clone();
    for t in 0..rows {
        let p = pos.f32[t] as usize;
        if p >= table_rows {
            return Err(ModelError::Msg(format!(
                "owl_rope_half_partial: pos[{t}] = {p} 越出表行数 {table_rows}"
            )));
        }
        let c = &cos_t.f32[p * half..(p + 1) * half];
        let sr = &sin_t.f32[p * half..(p + 1) * half];
        for h in 0..heads {
            let base = t * row_len + h * head_dim;
            for i in 0..half {
                let a = x.f32[base + i];
                let b = x.f32[base + i + half];
                out[base + i] = a * c[i] - b * sr[i];
                out[base + i + half] = b * c[i] + a * sr[i];
            }
        }
    }
    Ok(Value { f32: out, shape: x.shape.clone() })
}

/// 非连续窄切物化:dst[r·out_dim + d] = src[r·src_dim + start + d]
pub(crate) fn narrow_strided(
    src: &Value,
    outer: usize,
    src_dim: usize,
    start: usize,
    out_dim: usize,
) -> Result<Value, ModelError> {
    if (outer.saturating_sub(1)) * src_dim + start + out_dim > src.f32.len() {
        return Err(ModelError::Msg(format!(
            "owl_narrow_strided: 读越界(outer {outer} src_dim {src_dim} start {start} out_dim {out_dim} / src {})",
            src.f32.len()
        )));
    }
    let mut out = vec![0.0f32; outer * out_dim];
    for r in 0..outer {
        let s = r * src_dim + start;
        out[r * out_dim..(r + 1) * out_dim].copy_from_slice(&src.f32[s..s + out_dim]);
    }
    Ok(Value { f32: out, shape: vec![outer, out_dim] })
}

/// LayerNorm(affine;two-pass mean/var —— 与 torch.nn.LayerNorm 同序，
/// 不做 E[x²]−mean²)。行宽 n 由标量权威（alloc 块无形状）。
pub(crate) fn layernorm(
    x: &Value,
    weight: &Value,
    bias: &Value,
    n: usize,
    eps: f32,
) -> Result<Value, ModelError> {
    let n = n.max(1);
    if weight.f32.len() < n || bias.f32.len() < n {
        return Err(ModelError::Msg(format!(
            "owl_layernorm: weight/bias 短于行宽 {n}(得 {}/{})",
            weight.f32.len(),
            bias.f32.len()
        )));
    }
    let rows = x.f32.len() / n;
    let mut out = vec![0.0f32; rows * n];
    for r in 0..rows {
        let xs = &x.f32[r * n..(r + 1) * n];
        let mean = xs.iter().sum::<f32>() / n as f32;
        let var = xs.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for c in 0..n {
            out[r * n + c] = (xs[c] - mean) * inv * weight.f32[c] + bias.f32[c];
        }
    }
    Ok(Value { f32: out, shape: x.shape.clone() })
}


// ============================================================================
// 视觉塔算子批二(2026-10-15):Gelu / BiasAdd / PosEmbed / AttnFull
// (与 `owl-cpu` 之外的第二份实现 `owl_models::reference` 互为独立副本,禁互引)
// ============================================================================

/// erf 的 f64 参考实现(仅 CPU 对拍用;CUDA 侧走设备 `erff`)。
/// |x| ≤ 4 用交替级数(保留 ~60 项,收敛后 break);|x| > 4 时 erf 距 1 < 1.5e-8
/// ⇒ 直接取 ±1(误差远低于判据 1e-6)。
fn erf64(x: f64) -> f64 {
    if x.abs() > 4.0 {
        return if x > 0.0 { 1.0 } else { -1.0 };
    }
    let mut term = x; // n = 0 项:x
    let mut sum = x;
    let x2 = x * x;
    let mut n = 0u32;
    loop {
        n += 1;
        term *= -x2 / (n as f64);
        let add = term / (2 * n + 1) as f64;
        sum += add;
        if add.abs() < 1e-18 || n > 200 {
            break;
        }
    }
    2.0 / std::f64::consts::PI.sqrt() * sum
}

/// GELU(tanh 近似):0.5x(1+tanh(√(2/π)(x+0.044715x³)))
pub(crate) fn gelu_tanh(x: &Value) -> Result<Value, ModelError> {
    const C: f32 = 0.797_884_56_f32; // √(2/π)
    Ok(Value {
        f32: x
            .f32
            .iter()
            .map(|v| {
                let v = *v;
                0.5 * v * (1.0 + (C * (v + 0.044715 * v * v * v)).tanh())
            })
            .collect(),
        shape: x.shape.clone(),
    })
}

/// GELU(精确 erf):0.5x(1+erf(x/√2))—— **merger 专用**,与 tanh 版不可互替
pub(crate) fn gelu_erf(x: &Value) -> Result<Value, ModelError> {
    Ok(Value {
        f32: x
            .f32
            .iter()
            .map(|v| {
                let vf = *v as f64;
                (0.5 * vf * (1.0 + erf64(vf * std::f64::consts::FRAC_1_SQRT_2))) as f32
            })
            .collect(),
        shape: x.shape.clone(),
    })
}

/// 行广播加:x [rows, cols] + bias [cols]
pub(crate) fn bias_add(x: &Value, bias: &Value, cols: usize) -> Result<Value, ModelError> {
    let cols = cols.max(1);
    if bias.f32.len() < cols {
        return Err(ModelError::Msg(format!(
            "owl_bias_add: bias 长 {} < cols {cols}",
            bias.f32.len()
        )));
    }
    let rows = x.f32.len() / cols;
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[r * cols + c] = x.f32[r * cols + c] + bias.f32[c];
        }
    }
    Ok(Value { f32: out, shape: x.shape.clone() })
}

/// VLM 图像 token 注入:out[t] = idx[t] ≥ 0 ? merged[idx[t]] : h[t]
/// (idx 为 f32 行号;负值 = 文本位保留 h)
pub(crate) fn image_scatter(
    h: &Value,
    merged: &Value,
    idx: &Value,
    dim: usize,
) -> Result<Value, ModelError> {
    let dim = dim.max(1);
    if h.f32.len() % dim != 0 || merged.f32.len() % dim != 0 {
        return Err(ModelError::Msg(format!(
            "owl_image_scatter: h 长 {} / merged 长 {} 不是 dim {dim} 的整数倍",
            h.f32.len(),
            merged.f32.len()
        )));
    }
    let rows = h.f32.len() / dim;
    let n_merged = merged.f32.len() / dim;
    if idx.f32.len() != rows {
        return Err(ModelError::Msg(format!(
            "owl_image_scatter: idx 长 {} ≠ h 行数 {rows}",
            idx.f32.len()
        )));
    }
    let mut out = vec![0.0f32; rows * dim];
    for r in 0..rows {
        let v = idx.f32[r];
        if v < 0.0 {
            out[r * dim..(r + 1) * dim].copy_from_slice(&h.f32[r * dim..(r + 1) * dim]);
            continue;
        }
        let src = v as usize;
        if !v.is_finite() || src >= n_merged {
            return Err(ModelError::Msg(format!(
                "owl_image_scatter: idx[{r}] = {v} 越出 merged 行数 {n_merged}"
            )));
        }
        out[r * dim..(r + 1) * dim].copy_from_slice(&merged.f32[src * dim..(src + 1) * dim]);
    }
    Ok(Value { f32: out, shape: h.shape.clone() })
}

/// 位置嵌入 4-tap 重采样:out[n,d] = Σ_k wts[n,k]·table[idx[n,k],d]
pub(crate) fn pos_embed_bilinear(
    table: &Value,
    idx: &Value,
    wts: &Value,
    n_taps: usize,
    dim: usize,
) -> Result<Value, ModelError> {
    let (n_taps, dim) = (n_taps.max(1), dim.max(1));
    if idx.f32.len() != wts.f32.len() || idx.f32.len() % n_taps != 0 {
        return Err(ModelError::Msg(format!(
            "owl_pos_embed_bilinear: idx 长 {} / wts 长 {} 与 n_taps {n_taps} 不齐",
            idx.f32.len(),
            wts.f32.len()
        )));
    }
    let rows = idx.f32.len() / n_taps;
    if table.f32.len() < dim {
        return Err(ModelError::Msg(format!(
            "owl_pos_embed_bilinear: table 长 {} < dim {dim}",
            table.f32.len()
        )));
    }
    let n_pos = table.f32.len() / dim;
    let mut out = vec![0.0f32; rows * dim];
    for r in 0..rows {
        for k in 0..n_taps {
            let w = wts.f32[r * n_taps + k];
            let row = idx.f32[r * n_taps + k] as usize;
            if row >= n_pos {
                return Err(ModelError::Msg(format!(
                    "owl_pos_embed_bilinear: idx[{r},{k}] = {row} 越出表行数 {n_pos}"
                )));
            }
            for d in 0..dim {
                out[r * dim + d] += w * table.f32[row * dim + d];
            }
        }
    }
    Ok(Value { f32: out, shape: vec![rows, dim] })
}

/// 非因果 varlen 全注意力(逐图;无 GQA;两趟 softmax 参考实现)
pub(crate) fn attn_full(
    q: &Value,
    k: &Value,
    v: &Value,
    cu: &Value,
    rows: usize,
    heads: usize,
    head_dim: usize,
    n_seq: usize,
    scale: f32,
) -> Result<Value, ModelError> {
    if cu.f32.len() < n_seq + 1 {
        return Err(ModelError::Msg(format!(
            "owl_vis_attn_full: cu_seqlens 长 {} < n_seq+1 = {}",
            cu.f32.len(),
            n_seq + 1
        )));
    }
    let mut out = vec![0.0f32; rows * heads * head_dim];
    let at = |buf: &Value, row: usize, head: usize, d: usize| -> f32 {
        buf.f32[(row * heads + head) * head_dim + d]
    };
    for row in 0..rows {
        let (mut start, mut end) = (0usize, 0usize);
        for s in 0..n_seq {
            let a = cu.f32[s] as usize;
            let b = cu.f32[s + 1] as usize;
            if row >= a && row < b {
                start = a;
                end = b;
                break;
            }
        }
        if end <= start {
            continue;
        }
        for head in 0..heads {
            let mut scores = Vec::with_capacity(end - start);
            let mut m = f32::NEG_INFINITY;
            for t in start..end {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += at(q, row, head, d) * at(k, t, head, d);
                }
                let sc = dot * scale;
                if sc > m {
                    m = sc;
                }
                scores.push(sc);
            }
            let mut l = 0.0f32;
            let mut acc = vec![0.0f32; head_dim];
            for (i, t) in (start..end).enumerate() {
                let e = (scores[i] - m).exp();
                l += e;
                for d in 0..head_dim {
                    acc[d] += e * at(v, t, head, d);
                }
            }
            let inv = if l > 0.0 { 1.0 / l } else { 0.0 };
            for d in 0..head_dim {
                out[(row * heads + head) * head_dim + d] = acc[d] * inv;
            }
        }
    }
    Ok(Value { f32: out, shape: vec![rows, heads, head_dim] })
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

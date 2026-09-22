//! RoPE 旋转位置编码(= xinfer layers/rotary_emb.rs 直译)。
//!
//! 翻译注记:
//! - scalar 算术(`(t - min)? / (max-min)?` 等)统一走 [`OwlTensor::affine`]
//!   (S1:擦除层不开自由标量算术面);
//! - yarn 的 CPU 预计算段:owl 无 CPU 设备,inv/freq 表直接落目标卡池
//!   (host 只做 Vec 数学,再 from_vec 装载——语义等价);
//! - fused rope(vendor FusedRope)→ vendor 占位,T3 kernel 回填。

use super::ctor;
use super::{vendor, Device, DType, OwlTensor, Result, Tensor};
use crate::bail;
use crate::config::{Config, RopeScalingValue};

pub trait ApplyRotaryEmbedding {
    /// Apply rotary embedding to packed Q and K tensors with shape
    /// `[tokens, heads, head_dim]`.
    /// Returns:
    /// - `Ok(None)` when in-place operation was performed (CUDA, non-partial)
    /// - `Ok(Some((q, k)))` when new tensors are returned (partial rotary or non-CUDA)
    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        positions: &Tensor,
    ) -> Result<Option<(Tensor, Tensor)>>;

    fn get_original_max_position_embeddings(&self) -> Option<usize>;
    fn get_llama_4_scaling_beta(&self) -> Option<f64>;
}

#[derive(Clone)]
pub struct RotaryEmbedding {
    pub sin: Tensor,
    pub cos: Tensor,
    pub is_rope_i: bool,
    pub rotary_dim: Option<usize>,
    pub original_max_position_embeddings: Option<usize>,
    pub llama_4_scaling_beta: Option<f64>,
}

impl RotaryEmbedding {
    pub fn new(
        dtype: DType,
        cfg: &Config,
        dev: &Device,
        is_rope_i: bool,
        rope_theta: Option<f64>,
        original_max_position_embeddings: Option<usize>,
        llama_4_scaling_beta: Option<f64>,
    ) -> Result<Self> {
        let dim = cfg
            .head_dim
            .unwrap_or(cfg.hidden_size / cfg.num_attention_heads);
        let rotary_dim = cfg
            .partial_rotary_factor
            .map(|factor| (factor * dim as f32) as usize)
            .unwrap_or(dim);
        let rope_theta = rope_theta.unwrap_or(10000.0);
        let inv_freq: Vec<_> = (0..rotary_dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / rotary_dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = ctor::from_vec(inv_freq, (1, inv_freq_len), dev)?;
        let t = ctor::arange(0, cfg.max_position_embeddings, DType::U32, dev)?
            .to_dtype(DType::F32)?
            .reshape((cfg.max_position_embeddings, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
            is_rope_i,
            rotary_dim: if cfg.partial_rotary_factor.is_some() {
                Some(rotary_dim)
            } else {
                None
            },
            original_max_position_embeddings,
            llama_4_scaling_beta,
        })
    }
}

impl ApplyRotaryEmbedding for RotaryEmbedding {
    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        positions: &Tensor,
    ) -> Result<Option<(Tensor, Tensor)>> {
        let (_tokens, _q_heads, _head_dim) = q.dims3()?;
        let (_k_tokens, _k_heads, _k_head_dim) = k.dims3()?;

        // Handle partial rotary (rotary_dim < head_dim) - must return new tensors
        if let Some(rotary_dim) = self.rotary_dim {
            vendor::fused_rope::apply_inplace_partial(
                q,
                k,
                &self.cos,
                &self.sin,
                positions,
                self.is_rope_i,
                rotary_dim,
            )?;
            return Ok(None);
        }

        // Full rotary embedding - use fused kernel with position selection
        // Pass full cos/sin tables and positions - kernel selects on-the-fly
        // This eliminates the index_select kernel launch!
        let (_t, q_heads, head_dim) = q.dims3()?;
        let (_kt, k_heads, _kd) = k.dims3()?;
        vendor::fused_rope::apply_inplace(
            q, k, &self.cos, &self.sin, positions, self.is_rope_i,
            q_heads, k_heads, head_dim,
        )?;
        Ok(None)
    }

    fn get_original_max_position_embeddings(&self) -> Option<usize> {
        self.original_max_position_embeddings
    }

    fn get_llama_4_scaling_beta(&self) -> Option<f64> {
        self.llama_4_scaling_beta
    }
}

fn calculate_default_inv_freq(base: f64, dim: usize) -> Vec<f32> {
    (0..dim)
        .step_by(2)
        .map(|i| 1f32 / base.powf(i as f64 / dim as f64) as f32)
        .collect()
}

#[derive(Clone)]
pub struct ScalingRotaryEmbedding(pub RotaryEmbedding);

impl ScalingRotaryEmbedding {
    pub fn new(
        dtype: DType,
        cfg: &Config,
        dev: &Device,
        is_rope_i: bool,
        rope_theta: Option<f64>,
    ) -> Result<Self> {
        let dim = cfg
            .head_dim
            .unwrap_or(cfg.hidden_size / cfg.num_attention_heads);

        let rotary_dim = cfg
            .partial_rotary_factor
            .map(|factor| (factor * dim as f32) as usize)
            .unwrap_or(dim);

        if let Some(rope_scaling_cfg) = &cfg.rope_scaling {
            let mut rope_scaling = rope_scaling_cfg.clone();

            // Normalize rope_type key
            if !rope_scaling.contains_key("rope_type") && rope_scaling.contains_key("type") {
                let value = rope_scaling.remove("type").unwrap();
                rope_scaling.insert("rope_type".to_string(), value);
            }

            let original_max_position_embeddings: f64 = if let Some(v) = rope_scaling
                .get("original_max_position_embeddings")
                .and_then(|v| v.as_f64())
            {
                v
            } else if let Some(factor) = rope_scaling.get("factor").and_then(|v| v.as_f64()) {
                cfg.max_position_embeddings as f64 / factor
            } else {
                cfg.max_position_embeddings as f64
            };

            let rope_type = rope_scaling
                .get("rope_type")
                .and_then(|v| v.as_str())
                .ok_or_else(|| crate::Error::Msg("rope_type must be a string".to_string()))?;

            let rope_result = match rope_type {
                "linear" => {
                    let factor = rope_scaling
                        .get("factor")
                        .and_then(|v| v.as_f64())
                        .ok_or_else(|| {
                            crate::Error::Msg("Linear rope_type requires factor to be set".into())
                        })?;

                    let rope_theta = cfg.rope_theta.unwrap_or(10000.0);
                    let model_len = (original_max_position_embeddings * factor) as u32;

                    let inv_freq: Vec<_> = calculate_default_inv_freq(rope_theta, rotary_dim);
                    // (inv_freq / factor) → affine(1/factor, 0)(S1 标量面收窄)
                    let inv_freq = ctor::from_vec(
                        inv_freq.into_iter().map(|f| f / factor as f32).collect(),
                        (1, inv_freq_len_of(rotary_dim)),
                        dev,
                    )?;

                    let idx_theta = ctor::arange(0, model_len as usize, DType::U32, dev)?
                        .to_dtype(DType::F32)?
                        .reshape((model_len as usize, 1))?;

                    let idx_theta =
                        idx_theta.matmul(&inv_freq.reshape((1, inv_freq.elem_count()?))?)?;
                    let cos = idx_theta.cos()?.to_dtype(dtype)?;
                    let sin = idx_theta.sin()?.to_dtype(dtype)?;

                    Self(RotaryEmbedding {
                        sin,
                        cos,
                        is_rope_i,
                        rotary_dim: cfg.partial_rotary_factor.map(|_| rotary_dim),
                        original_max_position_embeddings: Some(
                            original_max_position_embeddings as usize,
                        ),
                        llama_4_scaling_beta: None,
                    })
                }

                "llama3" => {
                    let factor = rope_scaling.get("factor").and_then(|v| v.as_f64());
                    let low_freq_factor =
                        rope_scaling.get("low_freq_factor").and_then(|v| v.as_f64());
                    let high_freq_factor = rope_scaling
                        .get("high_freq_factor")
                        .and_then(|v| v.as_f64());

                    let (factor, low_freq_factor, high_freq_factor) = match (
                        factor,
                        low_freq_factor,
                        high_freq_factor,
                    ) {
                        (Some(a), Some(b), Some(c)) => (a, b, c),
                        _ => bail!(
                            "Llama3 rope_type requires factor, low_freq_factor, high_freq_factor to be set"
                        ),
                    };

                    let low_freq_wavelen =
                        (original_max_position_embeddings / low_freq_factor) as f32;
                    let high_freq_wavelen =
                        (original_max_position_embeddings / high_freq_factor) as f32;

                    let rope_theta = cfg.rope_theta.unwrap_or(10000.0);

                    let inv_freq = calculate_default_inv_freq(rope_theta, rotary_dim)
                        .into_iter()
                        .map(|freq| {
                            let wavelen = 2. * std::f32::consts::PI / freq;
                            if wavelen < high_freq_wavelen {
                                freq
                            } else if wavelen > low_freq_wavelen {
                                freq / factor as f32
                            } else {
                                let smooth = (original_max_position_embeddings as f32 / wavelen
                                    - low_freq_factor as f32)
                                    / (high_freq_factor - low_freq_factor) as f32;
                                (1. - smooth) * freq / factor as f32 + smooth * freq
                            }
                        })
                        .collect::<Vec<_>>();

                    let inv_freq_len = inv_freq.len();
                    let inv_freq = ctor::from_vec(inv_freq, (1, inv_freq_len), dev)?;

                    let t = ctor::arange(0, cfg.max_position_embeddings, DType::U32, dev)?
                        .to_dtype(DType::F32)?
                        .reshape((cfg.max_position_embeddings, 1))?;

                    let freqs = t.matmul(&inv_freq)?;
                    let sin = freqs.sin()?.to_dtype(dtype)?;
                    let cos = freqs.cos()?.to_dtype(dtype)?;

                    let llama_4_scaling_beta = rope_scaling
                        .get("llama_4_scaling_beta")
                        .and_then(|v| v.as_f64());

                    Self(RotaryEmbedding {
                        sin,
                        cos,
                        is_rope_i,
                        rotary_dim: cfg.partial_rotary_factor.map(|_| rotary_dim),
                        original_max_position_embeddings: Some(
                            original_max_position_embeddings as usize,
                        ),
                        llama_4_scaling_beta,
                    })
                }

                "default" => Self(RotaryEmbedding::new(
                    dtype, cfg, dev, is_rope_i, rope_theta, None, None,
                )?),

                "dynamic" => {
                    let scaling_factor = rope_scaling
                        .get("alpha")
                        .and_then(|v| v.as_f64())
                        .or_else(|| rope_scaling.get("factor").and_then(|v| v.as_f64()))
                        .ok_or_else(|| {
                            crate::Error::Msg(
                                "Dynamic rope_type requires either alpha or factor to be set".into(),
                            )
                        })?;

                    let rope_theta = cfg.rope_theta.unwrap_or(10000.0);

                    let (rope_theta, max_seq_len) = if rope_scaling.contains_key("alpha") {
                        let max_len = cfg.max_position_embeddings as u32;
                        let rope_theta = (rope_theta * scaling_factor)
                            .powf(rotary_dim as f64 / (rotary_dim - 2) as f64);
                        (rope_theta, max_len)
                    } else {
                        let max_len = (original_max_position_embeddings * scaling_factor) as u32;
                        let rope_theta = (rope_theta
                            * ((scaling_factor * max_len as f64
                                / original_max_position_embeddings)
                                - (scaling_factor - 1.0)))
                            .powf(rotary_dim as f64 / (rotary_dim - 2) as f64);
                        (rope_theta, max_len)
                    };

                    let inv_freq = calculate_default_inv_freq(rope_theta, rotary_dim);
                    let inv_freq_len = inv_freq.len();
                    let inv_freq = ctor::from_vec(inv_freq, (1, inv_freq_len), dev)?;

                    let t = ctor::arange(0, max_seq_len as usize, DType::U32, dev)?
                        .to_dtype(DType::F32)?
                        .reshape((max_seq_len as usize, 1))?;

                    let freqs = t.matmul(&inv_freq)?;
                    let sin = freqs.sin()?.to_dtype(dtype)?;
                    let cos = freqs.cos()?.to_dtype(dtype)?;

                    Self(RotaryEmbedding {
                        sin,
                        cos,
                        is_rope_i,
                        rotary_dim: cfg.partial_rotary_factor.map(|_| rotary_dim),
                        original_max_position_embeddings: Some(
                            original_max_position_embeddings as usize,
                        ),
                        llama_4_scaling_beta: None,
                    })
                }

                "yarn" => {
                    let default_one = RopeScalingValue::Number(1.0);
                    let default_fast = RopeScalingValue::Number(32.0);

                    let extrapolation_factor = rope_scaling
                        .get("extrapolation_factor")
                        .unwrap_or(&default_one)
                        .as_f64();
                    let attn_factor = rope_scaling
                        .get("attn_factor")
                        .unwrap_or(&default_one)
                        .as_f64();
                    let beta_fast = rope_scaling
                        .get("beta_fast")
                        .unwrap_or(&default_fast)
                        .as_f64();
                    let beta_slow = rope_scaling
                        .get("beta_slow")
                        .unwrap_or(&default_one)
                        .as_f64();
                    let factor = rope_scaling.get("factor").and_then(|v| v.as_f64());

                    let (extrapolation_factor, attn_factor, beta_fast, beta_slow, factor) = match (
                        extrapolation_factor,
                        attn_factor,
                        beta_fast,
                        beta_slow,
                        factor,
                    ) {
                        (Some(a), Some(b), Some(c), Some(d), Some(e)) => (a, b, c, d, e),
                        _ => {
                            bail!("yarn rope_type requires factor to be set");
                        }
                    };

                    let rope_theta = cfg.rope_theta.unwrap_or(10000.0);

                    let embed = YarnRotaryEmbedding::new_yarn(
                        dtype,
                        dev,
                        rope_theta as f32,
                        rotary_dim,
                        cfg.max_position_embeddings,
                        original_max_position_embeddings as usize,
                        beta_fast as f32,
                        beta_slow as f32,
                        attn_factor as f32,
                        extrapolation_factor as f32,
                        factor as f32,
                    )?;

                    let llama_4_scaling_beta = rope_scaling
                        .get("llama_4_scaling_beta")
                        .and_then(|v| v.as_f64());

                    Self(RotaryEmbedding {
                        sin: embed.sin,
                        cos: embed.cos,
                        is_rope_i,
                        rotary_dim: cfg.partial_rotary_factor.map(|_| rotary_dim),
                        original_max_position_embeddings: Some(
                            original_max_position_embeddings as usize,
                        ),
                        llama_4_scaling_beta,
                    })
                }

                other => {
                    bail!("Unknown rope_type: {other}");
                }
            };

            Ok(rope_result)
        } else {
            Ok(Self(RotaryEmbedding::new(
                dtype, cfg, dev, is_rope_i, rope_theta, None, None,
            )?))
        }
    }
}

impl ApplyRotaryEmbedding for ScalingRotaryEmbedding {
    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        positions: &Tensor,
    ) -> Result<Option<(Tensor, Tensor)>> {
        self.0.apply_rotary_emb_qkv(q, k, positions)
    }

    fn get_original_max_position_embeddings(&self) -> Option<usize> {
        self.0.original_max_position_embeddings
    }

    fn get_llama_4_scaling_beta(&self) -> Option<f64> {
        self.0.llama_4_scaling_beta
    }
}

/// linear 分支的 inv_freq 长度(rotary_dim/2 向上取整,与 step_by(2) 语义一致)
fn inv_freq_len_of(rotary_dim: usize) -> usize {
    (rotary_dim + 1) / 2
}

pub struct YarnRotaryEmbedding {
    pub sin: Tensor,
    pub cos: Tensor,
}

impl YarnRotaryEmbedding {
    fn yarn_find_correction_dim(
        num_rot: f32,
        dim: usize,
        base: f32,
        max_position_embeddings: usize,
    ) -> f32 {
        (dim as f32 * (max_position_embeddings as f32 / (num_rot * 2. * std::f32::consts::PI)).ln())
            / (2. * base.ln())
    }

    fn yarn_find_correction_range(
        low_rot: f32,
        high_rot: f32,
        dim: usize,
        base: f32,
        max_position_embeddings: usize,
    ) -> (f32, f32) {
        let low =
            Self::yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings).floor();
        let high =
            Self::yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings).ceil();
        (low.max(0.), high.min(dim as f32 - 1.))
    }

    /// 标量算术全走 affine(S1):
    /// `(arange - min) / (max - min)` = affine(1/(max-min), -min/(max-min))
    fn yarn_linear_ramp_mask(min: f32, mut max: f32, dim: usize, dev: &Device) -> Result<Tensor> {
        if min == max {
            max += 0.001;
        }
        let scale = 1.0 / (max as f64 - min as f64);
        ctor::arange(0, dim, DType::U32, dev)?
            .to_dtype(DType::F32)?
            .affine(scale, -(min as f64) * scale)?
            .clamp(0., 1.)
    }

    pub(crate) fn yarn_get_mscale(scale: f32, mscale: f32) -> f32 {
        if scale <= 1. {
            return 1.;
        }
        0.1 * mscale * scale.ln() + 1.
    }

    #[allow(clippy::too_many_arguments)]
    fn new_yarn(
        dtype: DType,
        dev: &Device,
        rope_theta: f32,
        dim: usize,
        max_position_embeddings: usize,
        original_max_position_embeddings: usize,
        beta_fast: f32,
        beta_slow: f32,
        attn_factor: f32,
        extrapolation_factor: f32,
        factor: f32,
    ) -> Result<Self> {
        // 注:原版在 CPU 设备上预计算再 to_device(dev);owl 无 CPU 设备,
        // host Vec 数学照旧,装载直接落目标卡池(语义等价)。
        let freq_extra: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f32 / dim as f32))
            .collect();
        let freq_extra_len = freq_extra.len();
        let freq_extra = ctor::from_vec(freq_extra, (1, freq_extra_len), dev)?;
        let freq_inter: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / (factor * rope_theta.powf(i as f32 / dim as f32)))
            .collect();
        let freq_inter_len = freq_inter.len();
        let freq_inter = ctor::from_vec(freq_inter, (1, freq_inter_len), dev)?;

        let (low, high) = Self::yarn_find_correction_range(
            beta_fast,
            beta_slow,
            dim,
            rope_theta,
            original_max_position_embeddings,
        );
        // (1 - ramp) * extrapolation → affine(-e, e)
        let inv_freq_mask = Self::yarn_linear_ramp_mask(low, high, dim / 2, dev)?
            .affine(-(extrapolation_factor as f64), extrapolation_factor as f64)?;
        // freq_inter*(1-mask) + freq_extra*mask → 两次 affine + broadcast 加乘
        let inv_freq = freq_inter
            .affine(1.0, 0.0)?
            .broadcast_mul(&inv_freq_mask.affine(-1.0, 1.0)?)?
            .broadcast_add(&freq_extra.broadcast_mul(&inv_freq_mask)?)?;

        let scaled_len = (max_position_embeddings as f32 * factor) as usize;
        let t = ctor::arange(0, scaled_len, DType::U32, dev)?
            .to_dtype(DType::F32)?
            .reshape((scaled_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;

        let mscale = Self::yarn_get_mscale(factor, 1.0f32) * attn_factor;
        // freqs * mscale → affine(mscale, 0)
        let sin = freqs
            .sin()?
            .affine(mscale as f64, 0.0)?
            .to_dtype(dtype)?;
        let cos = freqs
            .cos()?
            .affine(mscale as f64, 0.0)?
            .to_dtype(dtype)?;

        Ok(Self { sin, cos })
    }
}

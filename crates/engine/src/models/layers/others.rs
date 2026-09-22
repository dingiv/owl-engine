//! 通用层件:NormX/rms_norm/layer_norm/embedding/conv(= xinfer layers/others.rs 直译)。
//!
//! 翻译注记:
//! - Either<标准, GGUF> 双路 → `VarBuilderX::is_gguf()` 布尔分派;
//! - V4 ATen-order RMSNorm(vendor attention_rs)→ vendor 占位;
//! - MLX NVFP4 / GGUF dequantize → 量化路径 = marlin-ffi,运行里程碑。

use super::{ops, vendor, Conv2dConfig, Conv, DType, Embedding, LayerNorm, OwlTensor, Result, RmsNorm, Shard, Tensor, VarBuilderX};

pub struct NormX {
    norm: NormKind,
    /// When set, forward uses DeepSeek-V4 ATen-order RMSNorm CUDA kernel
    /// (`rms_norm_v4`) instead of Candle's generic reduction. V4's 86 HC
    /// updates amplify last-bit F32 differences from the mean reduction.
    v4_weight: Option<Tensor>,
    v4_eps: f32,
    dtype: DType,
}

enum NormKind {
    Rms(RmsNorm),
    Layer(LayerNorm),
}

impl NormX {
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        if let Some(weight) = &self.v4_weight {
            return self.forward_v4(xs, weight);
        }
        let in_dtype = xs.dtype();
        if xs.dtype() != self.dtype {
            let converted = xs.to_dtype(self.dtype)?;
            let out = match &self.norm {
                NormKind::Rms(norm) => norm.forward(&converted)?,
                NormKind::Layer(norm) => norm.forward(&converted)?,
            };
            out.to_dtype(in_dtype)
        } else {
            let out = match &self.norm {
                NormKind::Rms(norm) => norm.forward(xs)?,
                NormKind::Layer(norm) => norm.forward(xs)?,
            };
            Ok(out)
        }
    }

    /// F32 weight tensor for DeepSeek-V4 ATen-order RMSNorm (used by fused HC pre+norm).
    pub fn v4_weight_f32(&self) -> Option<&Tensor> {
        self.v4_weight.as_ref()
    }

    pub fn v4_eps(&self) -> f32 {
        self.v4_eps
    }

    /// Single ATen-order CUDA RMSNorm launch. Hot path (contiguous BF16 2D +
    /// F32 weight) does no dtype/reshape/weight clones around the kernel.
    fn forward_v4(&self, xs: &Tensor, weight: &Tensor) -> Result<Tensor> {
        let in_dtype = xs.dtype();
        let dims = xs.dims()?;
        let dim = *dims
            .last()
            .ok_or_else(|| crate::Error::Msg("NormX V4: expected non-empty dims".into()))?;
        let rows: usize = if dims.len() <= 1 {
            1
        } else {
            dims[..dims.len() - 1].iter().product::<usize>().max(1)
        };

        // Fast path: already BF16 [rows, dim] — one kernel, no host-side glue.
        if in_dtype == DType::BF16 && dims.len() == 2 && dims[0] == rows && dims[1] == dim {
            let x = if xs.is_contiguous() {
                xs.clone()
            } else {
                xs.contiguous()?
            };
            return vendor::deepseek_v4::rms_norm_v4(&x, weight, dim, self.v4_eps);
        }

        // Slow path: materialize BF16 2D, normalize in-place, restore shape/dtype.
        let x = if in_dtype == DType::BF16 {
            xs.clone()
        } else {
            xs.to_dtype(DType::BF16)?
        };
        let x = x.reshape((rows, dim))?;
        let x = if x.is_contiguous() {
            x
        } else {
            x.contiguous()?
        };
        vendor::deepseek_v4::rms_norm_v4_inplace(&x, weight, dim, self.v4_eps)?;
        let out = if dims.len() == 2 && dims[0] == rows && dims[1] == dim {
            x
        } else {
            x.reshape(dims.to_vec())?
        };
        if in_dtype == DType::BF16 {
            Ok(out)
        } else {
            out.to_dtype(in_dtype)
        }
    }

    /// In-place V4 RMSNorm on an owned contiguous BF16 `[rows, dim]` buffer.
    pub fn forward_v4_inplace(&self, xs: &Tensor) -> Result<()> {
        let weight = self.v4_weight.as_ref().ok_or_else(|| {
            crate::Error::Msg("forward_v4_inplace requires V4 norm weight".into())
        })?;
        let dim = *xs
            .dims()?
            .last()
            .ok_or_else(|| crate::Error::Msg("forward_v4_inplace: empty dims".into()))?;
        vendor::deepseek_v4::rms_norm_v4_inplace(xs, weight, dim, self.v4_eps)
    }
}

pub fn rms_norm(
    size: usize,
    eps: f64,
    vb: VarBuilderX,
    dtype: DType,
    is_gemma: bool,
) -> Result<NormX> {
    rms_norm_sharded(size, eps, vb, dtype, is_gemma, Shard::default())
}

/// DeepSeek-V4 RMSNorm with ATen (128,4) mean-reduction order.
pub fn rms_norm_v4(size: usize, eps: f64, vb: VarBuilderX, dtype: DType) -> Result<NormX> {
    rms_norm_v4_sharded(size, eps, vb, dtype, Shard::default())
}

pub fn rms_norm_v4_sharded(
    size: usize,
    eps: f64,
    vb: VarBuilderX,
    dtype: DType,
    shard: Shard,
) -> Result<NormX> {
    let (weight, dtype) = if !vb.is_gguf() {
        let ws = vb.get_with_hints(size, "weight", shard)?;
        if ws.dtype() != dtype {
            (ws.to_dtype(dtype)?, dtype)
        } else {
            (ws, dtype)
        }
    } else {
        (vb.get(size, "weight")?.dequantize(&vb.device())?, DType::F32)
    };
    let weight_f32 = weight.to_dtype(DType::F32)?;
    Ok(NormX {
        norm: NormKind::Rms(RmsNorm::new(weight_f32.clone(), eps)),
        v4_weight: Some(weight_f32),
        v4_eps: eps as f32,
        dtype,
    })
}

pub fn rms_norm_sharded(
    size: usize,
    eps: f64,
    vb: VarBuilderX,
    dtype: DType,
    is_gemma: bool,
    shard: Shard,
) -> Result<NormX> {
    let (weight, dtype) = if !vb.is_gguf() {
        let ws = vb.get_with_hints(size, "weight", shard)?;
        if ws.dtype() != dtype {
            (ws.to_dtype(dtype)?, dtype)
        } else {
            (ws, dtype)
        }
    } else {
        (vb.get(size, "weight")?.dequantize(&vb.device())?, DType::F32)
    };

    let weight = if is_gemma { weight.affine(1.0, 1.0)? } else { weight };
    Ok(NormX {
        norm: NormKind::Rms(RmsNorm::new(weight, eps)),
        v4_weight: None,
        v4_eps: eps as f32,
        dtype,
    })
}

pub fn layer_norm(
    size: usize,
    eps: f64,
    affine: bool,
    vb: VarBuilderX,
    dtype: DType,
) -> Result<NormX> {
    let (weight, dtype) = if !vb.is_gguf() {
        (
            vb.get_with_hints(size, "weight", Shard::default())?
                .to_dtype(dtype)?,
            dtype,
        )
    } else {
        (vb.get(size, "weight")?.dequantize(&vb.device())?, DType::F32)
    };
    if affine {
        let bias = if !vb.is_gguf() {
            vb.get(size, "bias")?.to_dtype(dtype)?
        } else {
            vb.get(size, "bias")?.dequantize(&vb.device())?
        };
        Ok(NormX {
            norm: NormKind::Layer(LayerNorm::new(weight, bias, eps)),
            v4_weight: None,
            v4_eps: eps as f32,
            dtype,
        })
    } else {
        Ok(NormX {
            norm: NormKind::Layer(LayerNorm::new_no_bias(weight, eps)),
            v4_weight: None,
            v4_eps: eps as f32,
            dtype,
        })
    }
}

pub fn embedding(
    vocab_size: Option<usize>,
    hidden_size: usize,
    vb: VarBuilderX,
    dtype: DType,
) -> Result<(Embedding, usize)> {
    let (embeddings, vocab_size) = if !vb.is_gguf() {
        assert!(
            vocab_size.is_some(),
            "vocab_size must be specified for safetensor models"
        );
        let vs = vocab_size.unwrap();
        if vb.contains_tensor("scales") {
            // MLX NVFP4: quantized embedding with U32 weights + U8 FP8 E4M3 scales.
            // Dequantize at load time since embeddings are looked up, not matmul'd.
            let emb = dequantize_mlx_nvfp4_embedding(&vb, vs, hidden_size, dtype)?;
            (emb, vs)
        } else {
            (vb.get((vs, hidden_size), "weight")?.to_dtype(dtype)?, vs)
        }
    } else {
        let weight = if vocab_size.is_some() {
            vb.get((vocab_size.unwrap(), hidden_size), "weight")?
        } else {
            vb.get_no_shape("weight")?
        }
        .dequantize(&vb.device())?;
        let vocab_size = vocab_size.unwrap_or(weight.dim(0)?);
        (weight, vocab_size)
    };
    Ok((Embedding::new(embeddings), vocab_size))
}

fn dequantize_mlx_nvfp4_embedding(
    vb: &VarBuilderX,
    vocab_size: usize,
    hidden_size: usize,
    dtype: DType,
) -> Result<Tensor> {
    let no_shard = Shard::default();
    let w_u32 = vb.get_with_hints_dtype(
        (vocab_size, hidden_size / 8),
        "weight",
        no_shard,
        DType::U32,
    )?;
    let scales = vb.get_with_hints_dtype(
        (vocab_size, hidden_size / 16),
        "scales",
        no_shard,
        DType::U8,
    )?;

    let out_dtype = match dtype {
        DType::F16 | DType::BF16 => dtype,
        _ => DType::BF16,
    };
    vendor::nvfp4_linear::mlx_dequant_embedding(
        &w_u32,
        &scales,
        vocab_size,
        hidden_size,
        out_dtype,
    )
}

pub fn conv2d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: Conv2dConfig,
    vb: VarBuilderX,
    bias: bool,
) -> Result<Conv> {
    let (ws, bs) = if !vb.is_gguf() {
        let ws = vb.get(
            (
                out_channels,
                in_channels / cfg.groups,
                kernel_size,
                kernel_size,
            ),
            "weight",
        )?;
        let bs = if bias {
            Some(vb.get(out_channels, "bias")?)
        } else {
            None
        };
        (ws, bs)
    } else {
        todo!()
    };

    Ok(Conv::new(ws, bs))
}

pub struct AvgPool2d {
    kernel_size: usize,
    stride: usize,
}

impl AvgPool2d {
    pub fn new(kernel_size: usize, stride: usize) -> Self {
        Self {
            kernel_size,
            stride,
        }
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.avg_pool2d_with_stride(self.kernel_size, self.stride)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conv3dConfig {
    pub padding: usize,
    pub stride: usize,
    pub dilation: usize,
    pub groups: usize,
}

impl Default for Conv3dConfig {
    fn default() -> Self {
        Self {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
        }
    }
}

pub struct Conv3dNoBias {
    conv2d_1: Conv,
    conv2d_2: Conv,
}

impl Conv3dNoBias {
    pub fn from_conv2d_weights(w1: Tensor, w2: Tensor, cfg: Conv2dConfig) -> Result<Self> {
        Ok(Self {
            conv2d_1: Conv::new(w1, None),
            conv2d_2: Conv::new(w2, None),
        })
    }

    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_sizes: [usize; 3],
        cfg: Conv3dConfig,
        vb: VarBuilderX,
    ) -> Result<Self> {
        let expected_shape = (
            out_channels,
            in_channels / cfg.groups,
            kernel_sizes[0],
            kernel_sizes[1],
            kernel_sizes[2],
        );
        let ws = if !vb.is_gguf() {
            match vb.get(expected_shape, "weight") {
                Ok(w) => w,
                Err(_) => {
                    // MLX stores conv weights as (O, T, H, W, C) instead of (O, C, T, H, W).
                    let mlx_shape = (
                        out_channels,
                        kernel_sizes[0],
                        kernel_sizes[1],
                        kernel_sizes[2],
                        in_channels / cfg.groups,
                    );
                    let w = vb.get(mlx_shape, "weight")?;
                    w.permute(&[0, 4, 1, 2, 3])?
                }
            }
        } else {
            panic!("Unsupported quantized format for conv3d")
        };

        // i((.., .., 0, .., ..)) / i((.., .., 1, .., ..)) 的窄切等价:
        // 第 2 维取 [0,1) / [1,2) 再 squeeze。
        let w1 = ws.narrow(2, 0, 1)?.squeeze(2)?;
        let w2 = ws.narrow(2, 1, 1)?.squeeze(2)?;

        let cfg = Conv2dConfig {
            padding: cfg.padding,
            stride: cfg.stride,
            dilation: cfg.dilation,
            groups: cfg.groups,
        };

        Ok(Self {
            conv2d_1: Conv::new(w1.contiguous()?, None),
            conv2d_2: Conv::new(w2.contiguous()?, None),
        })
    }

    pub fn weight(&self) -> Result<Tensor> {
        let w1 = self.conv2d_1.weight().clone().unsqueeze(2)?;
        let w2 = self.conv2d_2.weight().clone().unsqueeze(2)?;
        ops::cat(&[w1, w2], 2)
    }
}

impl Conv3dNoBias {
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // i((.., .., 0, .., ..)) / i((.., .., 1, .., ..)) 的窄切等价
        let xs1 = xs.narrow(2, 0, 1)?.squeeze(2)?;
        let xs2 = xs.narrow(2, 1, 1)?.squeeze(2)?;

        self.conv2d_1
            .forward(&xs1)?
            .broadcast_add(&self.conv2d_2.forward(&xs2)?)?
            .unsqueeze(2)
    }
}

/// 通用掩码填充。value 语义面取 f64(candle 的 WithDType 泛型在
/// owl 擦除层等价收窄;调用点数值字面量兼容)。
pub fn masked_fill(xs: &Tensor, mask: &Tensor, value: f64) -> Result<Tensor> {
    let on_true = ops::full(value, &xs.dims()?, xs)?;
    let on_false = xs;
    mask.broadcast_as(&xs.dims()?)?
        .where_cond(&on_true, on_false)
}

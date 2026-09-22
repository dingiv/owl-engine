//! GatedDeltaNet 线性注意力层(= xinfer layers/deltanet.rs 直译;T2-三 搬运)。
//!
//! 翻译注记:
//! - vendor attention_rs `gdn` kernel 面(causal_conv1d/fused_gdn_gating/
//!   l2_norm/gated_delta_rule_*/gated_rmsnorm_*)经 `gdn_shim` 垫片
//!   (真实 kernel = attention-rs port,运行里程碑;裁决 2026-09-22);
//! - MambaCache 方法面经 `MambaCacheHandle` 包装(vendor::MambaCache 本体
//!   只读;方法需求清单见尾注);
//! - **GGUF 重排分支语义裁剪**:xinfer 的 `undo_tiled_v_heads_*`/
//!   `restore_qwen35_*`/`load_restored_gguf_*` 路径**不搬**——owl loader
//!   (loader/gguf.rs)在 host 反解阶段即恢复张量布局,qvar 分支与标准
//!   分支在此合流(架构决定:重排属于装载层,不属于模型层);
//! - ct-awq merged qkv 专用通道保留分支骨架,体 = WNA16 合并切片,
//!   marlin-ffi 路线(运行里程碑);
//! - TP 多进程/nccl 分支 `#[cfg(feature = "nccl")]` 关断(owl 多卡 A2.6
//!   单进程多实例,另案)。

use super::distributed::{
    shard, Comm, MergedParallelColumnLinear, TensorParallelColumnLinear, TensorParallelRowLinear,
};
use super::vendor;
use super::{collect_key_map, DType, OwlTensor, Result, Shard, Tensor, VarBuilderX};
use crate::config::Config;
use crate::hybrid::resolve_qwen3_hybrid_config;
use std::rc::Rc;

// ---- vendor gdn kernel 垫片(attention-rs port,T3 回填) ----
#[allow(dead_code)] // flashinfer 引入后接通(裁决 2026-09-22)
mod gdn_shim {
    use super::{Result, Tensor};

    /// 因果卷积 prefill(变长;conv_state 就地更新 + 可选快照)
    #[allow(clippy::too_many_arguments)]
    pub fn causal_conv1d_fwd(
        _mixed_qkv: &Tensor,
        _weight: &Tensor,
        _bias: Option<&Tensor>,
        _conv_state: &mut Tensor,
        _snapshots: Option<&Tensor>,
        _cu_seqlens: &Tensor,
        _silu: bool,
    ) -> Result<Tensor> {
        unimplemented!("T3: gdn::causal_conv1d_fwd(attention-rs kernel port)")
    }

    /// 因果卷积 decode(按 slot 更新状态)
    #[allow(clippy::too_many_arguments)]
    pub fn causal_conv1d_update_slots(
        _mixed_qkv: &Tensor,
        _weight: &Tensor,
        _bias: Option<&Tensor>,
        _conv_state: &mut Tensor,
        _seq_slots: &Tensor,
        _silu: bool,
    ) -> Result<Tensor> {
        unimplemented!("T3: gdn::causal_conv1d_update_slots(attention-rs kernel port)")
    }

    /// 融合 GDN 门控(a_log/dt_bias → g, beta)
    pub fn fused_gdn_gating(
        _a_log: &Tensor,
        _a: &Tensor,
        _b: &Tensor,
        _dt_bias: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        unimplemented!("T3: gdn::fused_gdn_gating(attention-rs kernel port)")
    }

    /// 末维 L2 归一
    pub fn l2_norm_last_dim(_t: &Tensor, _eps: f64) -> Result<Tensor> {
        unimplemented!("T3: gdn::l2_norm_last_dim(attention-rs kernel port)")
    }

    /// prefill 变长 GQA 递推(flashinfer 路径;flashinfer 引入后接通)
    #[allow(clippy::too_many_arguments)]
    pub fn gated_delta_rule_prefill_flashinfer_gqa(
        _q: &Tensor,
        _k: &Tensor,
        _v: &Tensor,
        _g: &Tensor,
        _beta: &Tensor,
        _state: &mut Tensor,
        _seq_slots: &Tensor,
        _cu_seqlens: &Tensor,
        _scale: f32,
    ) -> Result<Tensor> {
        unimplemented!("T3: gdn::gated_delta_rule_prefill_flashinfer_gqa(flashinfer 引入后)")
    }

    /// prefill 变长 GQA 递推(主路)
    #[allow(clippy::too_many_arguments)]
    pub fn gated_delta_rule_recurrence_varlen_gqa(
        _q: &Tensor,
        _k: &Tensor,
        _v: &Tensor,
        _g: &Tensor,
        _beta: &Tensor,
        _state: &mut Tensor,
        _seq_slots: &Tensor,
        _cu_seqlens: &Tensor,
        _scale: f32,
        _snapshots: Option<&Tensor>,
    ) -> Result<Tensor> {
        unimplemented!("T3: gdn::gated_delta_rule_recurrence_varlen_gqa(attention-rs port)")
    }

    /// prefill 变长 MHA 递推(k==v 头数)
    #[allow(clippy::too_many_arguments)]
    pub fn gated_delta_rule_recurrence_varlen(
        _q: &Tensor,
        _k: &Tensor,
        _v: &Tensor,
        _g: &Tensor,
        _beta: &Tensor,
        _state: &mut Tensor,
        _seq_slots: &Tensor,
        _cu_seqlens: &Tensor,
        _snapshots: Option<&Tensor>,
    ) -> Result<Tensor> {
        unimplemented!("T3: gdn::gated_delta_rule_recurrence_varlen(attention-rs port)")
    }

    /// decode 按 slot GQA 递推
    #[allow(clippy::too_many_arguments)]
    pub fn gated_delta_rule_decode_slots_gqa(
        _q: &Tensor,
        _k: &Tensor,
        _v: &Tensor,
        _g: &Tensor,
        _beta: &Tensor,
        _state: &mut Tensor,
        _seq_slots: &Tensor,
        _scale: f32,
    ) -> Result<Tensor> {
        unimplemented!("T3: gdn::gated_delta_rule_decode_slots_gqa(attention-rs port)")
    }

    /// 门控 RMSNorm × sigmoid(z)
    #[allow(clippy::too_many_arguments)]
    pub fn gated_rmsnorm_sigmoid_mul(
        _output: &Tensor,
        _z: &Tensor,
        _weight: &Tensor,
        _bias: Option<&Tensor>,
        _eps: f64,
        _head_v_dim: usize,
    ) -> Result<Tensor> {
        unimplemented!("T3: gdn::gated_rmsnorm_sigmoid_mul(attention-rs port)")
    }

    /// 门控 RMSNorm × silu(z)
    #[allow(clippy::too_many_arguments)]
    pub fn gated_rmsnorm_silu_mul(
        _output: &Tensor,
        _z: &Tensor,
        _weight: &Tensor,
        _bias: Option<&Tensor>,
        _eps: f64,
        _head_v_dim: usize,
    ) -> Result<Tensor> {
        unimplemented!("T3: gdn::gated_rmsnorm_silu_mul(attention-rs port)")
    }
}

/// MambaCache 方法面包装(vendor::MambaCache 本体只读;方法需求见尾注)
pub(crate) struct MambaCacheHandle<'a> {
    #[allow(dead_code)] // T3:MambaCache 方法面接通后消费
    cache: &'a mut vendor::MambaCache,
}

impl<'a> MambaCacheHandle<'a> {
    pub fn new(cache: &'a mut vendor::MambaCache) -> Self {
        Self { cache }
    }
    pub fn get_batch_conv_state(&mut self, _layer: usize, _slots: &Tensor) -> Result<Tensor> {
        unimplemented!("T3: MambaCache::get_batch_conv_state")
    }
    pub fn set_batch_conv_state(&mut self, _layer: usize, _slots: &Tensor, _s: &Tensor) -> Result<()> {
        unimplemented!("T3: MambaCache::set_batch_conv_state")
    }
    pub fn conv_state_mut(&mut self, _layer: usize) -> &mut Tensor {
        unimplemented!("T3: MambaCache::conv_state_mut")
    }
    pub fn conv_state(&self, _layer: usize) -> &Tensor {
        unimplemented!("T3: MambaCache::conv_state")
    }
    pub fn recurrent_state_mut(&mut self, _layer: usize) -> &mut Tensor {
        unimplemented!("T3: MambaCache::recurrent_state_mut")
    }
    pub fn set_batch_recurrent_state(&mut self, _layer: usize, _slots: &Tensor, _s: &Tensor) -> Result<()> {
        unimplemented!("T3: MambaCache::set_batch_recurrent_state")
    }
}

enum GdnProjection {
    /// Qwen3Next: in_proj_qkvz + in_proj_ba
    FusedQkvzBa {
        in_proj_qkvz: TensorParallelColumnLinear,
        in_proj_ba: TensorParallelColumnLinear,
    },
    /// Qwen3.5: in_proj_qkv + in_proj_z + in_proj_b + in_proj_a
    SplitQkvZaLegacy {
        in_proj_qkv: TensorParallelColumnLinear,
        in_proj_z: TensorParallelColumnLinear,
        in_proj_b: TensorParallelColumnLinear,
        in_proj_a: TensorParallelColumnLinear,
    },
    /// Qwen3.5 TP-safe split for packed in_proj_qkv [q|k|v].
    SplitQkvZaMerged {
        in_proj_qkv: MergedParallelColumnLinear,
        in_proj_z: TensorParallelColumnLinear,
        in_proj_b: TensorParallelColumnLinear,
        in_proj_a: TensorParallelColumnLinear,
    },
}

pub struct GatedDeltaNet {
    projection: GdnProjection,
    out_proj: TensorParallelRowLinear,
    conv_weight: Tensor,
    conv_bias: Option<Tensor>,
    a_log: Tensor,
    dt_bias: Tensor,
    gdn_norm_weight: Tensor,
    gdn_norm_bias: Option<Tensor>,
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    key_dim: usize,
    value_dim: usize,
    kv_group_size: usize,
    gdn_layer_idx: usize,
    rms_norm_eps: f64,
    scale: f64,
    /// GDN 核心算子 dtype(conv/gating/recurrence;GGUF/F16 模式 = F32)
    gdn_dtype: DType,
    /// 模型原生 dtype(投影输入/权重装载)
    model_dtype: DType,
    /// 输出门激活:false = silu(Qwen3.5),true = sigmoid(Qwen4)
    gate_sigmoid: bool,
    conv_mtp_state: Option<Tensor>,
    recurrent_mtp_state: Option<Tensor>,
}

impl GatedDeltaNet {
    fn is_weight_quantized(vb: &VarBuilderX, quant_method: &str) -> bool {
        if vb.is_qvar_builder() {
            return false;
        }
        match quant_method {
            "fp8" => vb.has_key("weight_scale") || vb.has_key("weight_scale_inv"),
            "mxfp4" => vb.has_key("weight_packed") || vb.has_key("blocks"),
            "nvfp4" => {
                let has_packed = vb.has_key("weight_packed") || vb.has_key("blocks");
                let has_scale = vb.has_key("weight_scale") || vb.has_key("scales");
                let has_nvfp4_second_scale =
                    vb.has_key("weight_scale_2") || vb.has_key("weight_global_scale");
                let is_mlx_nvfp4 = vb.has_key("weight")
                    && vb.has_key("scales")
                    && !has_packed
                    && !has_nvfp4_second_scale;
                (has_packed && has_scale) || (has_nvfp4_second_scale && has_scale) || is_mlx_nvfp4
            }
            "gptq" | "awq" => vb.has_key("qweight") || vb.has_key("B"),
            "compressed-tensors" => vb.has_key("weight_packed") && vb.has_key("weight_scale"),
            _ => true,
        }
    }

    fn is_weight_fp8(vb: &VarBuilderX) -> bool {
        if vb.is_qvar_builder() {
            return false;
        }
        vb.has_key("weight_scale") || vb.has_key("weight_scale_inv")
    }

    /// 单权重量化解析:未实际量化 → (None,None) 走标准路;nvfp4 全局下
    /// 检出 fp8 权重 → 返回 fp8 配置(混合精度)
    fn resolve_quant_for_weight(
        vb: &VarBuilderX,
        quantization_config: &Option<crate::config::QuantConfig>,
        quant: &Option<String>,
    ) -> (Option<crate::config::QuantConfig>, Option<String>) {
        if let Some(cfg) = quantization_config {
            if Self::is_weight_quantized(vb, &cfg.quant_method) {
                return (quantization_config.clone(), quant.clone());
            }
            if cfg.quant_method == "nvfp4" && Self::is_weight_fp8(vb) {
                let mut fp8_cfg = cfg.clone();
                fp8_cfg.quant_method = "fp8".to_string();
                return (Some(fp8_cfg), quant.clone());
            }
        }
        (None, None)
    }

    #[allow(clippy::too_many_arguments)]
    fn load_projection(
        vb: &VarBuilderX,
        hidden_size: usize,
        num_k_heads_global: usize,
        key_dim_global: usize,
        value_dim_global: usize,
        num_v_heads_global: usize,
        head_v_dim: usize,
        comm: Rc<Comm>,
        config: &Config,
        dtype: DType,
        is_quantized: bool,
    ) -> Result<GdnProjection> {
        let (quantization_config, quant) = if is_quantized {
            (config.quantization_config.clone(), config.quant.clone())
        } else {
            (None, None)
        };
        let mut load_errors = Vec::new();
        let projection_pairs = [
            ("in_proj_qkv", "attn_qkv"),
            ("in_proj_z", "attn_gate"),
            ("in_proj_b", "ssm_beta"),
            ("in_proj_a", "ssm_alpha"),
        ];
        let projection_key_map = collect_key_map(vb.is_qvar_builder(), projection_pairs);

        // Qwen3Next 格式:融合 qkvz + 融合 ba
        let projection_size_qkvz = key_dim_global * 2 + value_dim_global * 2;
        let projection_size_ba = num_v_heads_global * 2;

        let vb_qkvz = vb.pp("in_proj_qkvz");
        let (qc_qkvz, q_qkvz) =
            Self::resolve_quant_for_weight(&vb_qkvz, &quantization_config, &quant);
        let fused_qkvz = TensorParallelColumnLinear::load_with_hints(
            hidden_size,
            projection_size_qkvz,
            false,
            vb_qkvz,
            comm.clone(),
            &qc_qkvz,
            &q_qkvz,
            dtype,
        );

        let vb_ba = vb.pp("in_proj_ba");
        let (qc_ba, q_ba) = Self::resolve_quant_for_weight(&vb_ba, &quantization_config, &quant);
        let fused_ba = TensorParallelColumnLinear::load_with_hints(
            hidden_size,
            projection_size_ba,
            false,
            vb_ba,
            comm.clone(),
            &qc_ba,
            &q_ba,
            dtype,
        );

        match (fused_qkvz, fused_ba) {
            (Ok(in_proj_qkvz), Ok(in_proj_ba)) => {
                return Ok(GdnProjection::FusedQkvzBa {
                    in_proj_qkvz,
                    in_proj_ba,
                });
            }
            (qkvz, ba) => {
                if let Err(err) = qkvz {
                    load_errors.push(format!("in_proj_qkvz: {err}"));
                }
                if let Err(err) = ba {
                    load_errors.push(format!("in_proj_ba: {err}"));
                }
            }
        };

        // Qwen3.5 格式:split qkv/z/b/a(GGUF 重排分支已由 loader 层吸收,
        // 见文件头"GGUF 重排分支语义裁剪")
        let split_z = {
            let vb_z = vb.pp(projection_key_map["in_proj_z"]);
            let (qc_z, q_z) = Self::resolve_quant_for_weight(&vb_z, &quantization_config, &quant);
            TensorParallelColumnLinear::load_with_hints(
                hidden_size,
                value_dim_global,
                false,
                vb_z,
                comm.clone(),
                &qc_z,
                &q_z,
                dtype,
            )
        };
        let split_b = {
            let vb_b = vb.pp(projection_key_map["in_proj_b"]);
            let (qc_b, q_b) = Self::resolve_quant_for_weight(&vb_b, &quantization_config, &quant);
            TensorParallelColumnLinear::load_with_hints(
                hidden_size,
                num_v_heads_global,
                false,
                vb_b,
                comm.clone(),
                &qc_b,
                &q_b,
                dtype,
            )
        };
        let split_a = {
            let vb_a = vb.pp(projection_key_map["in_proj_a"]);
            let (qc_a, q_a) = Self::resolve_quant_for_weight(&vb_a, &quantization_config, &quant);
            TensorParallelColumnLinear::load_with_hints(
                hidden_size,
                num_v_heads_global,
                false,
                vb_a,
                comm.clone(),
                &qc_a,
                &q_a,
                dtype,
            )
        };

        match (split_z, split_b, split_a) {
            (Ok(in_proj_z), Ok(in_proj_b), Ok(in_proj_a)) => {
                if comm.world_size > 1 {
                    // TP-safe:packed in_proj_qkv [q|k|v] 按语义段独立切分
                    let split_qkv_merged = {
                        let vb_qkv = vb.pp(projection_key_map["in_proj_qkv"]);
                        let (qc_qkv, q_qkv) =
                            Self::resolve_quant_for_weight(&vb_qkv, &quantization_config, &quant);
                        // ct 非对称 AWQ 合并 QKV:WNA16 语义行号切片
                        // (marlin-ffi 路线,运行里程碑;P5-Q1b 判例)
                        if qc_qkv
                            .as_ref()
                            .map(|c| {
                                c.quant_method == "compressed-tensors" && c.sym == Some(false)
                            })
                            .unwrap_or(false)
                        {
                            unimplemented!(
                                "T3: ct-awq merged qkv WNA16::new_ct_awq_merged_chunk 通道(marlin-ffi)"
                            );
                        }
                        MergedParallelColumnLinear::load_merged_chunks(
                            hidden_size,
                            key_dim_global * 2 + value_dim_global,
                            vec![key_dim_global, key_dim_global, value_dim_global],
                            vb_qkv,
                            &qc_qkv,
                            &q_qkv,
                            dtype,
                        )
                    };

                    match split_qkv_merged {
                        Ok(in_proj_qkv) => {
                            return Ok(GdnProjection::SplitQkvZaMerged {
                                in_proj_qkv,
                                in_proj_z,
                                in_proj_b,
                                in_proj_a,
                            });
                        }
                        Err(err) => {
                            if is_quantized && !vb.is_qvar_builder() {
                                crate::bail!(
                                    "Unable to load TP-safe quantized Qwen3.5 split in_proj_qkv: {}",
                                    err
                                );
                            }
                        }
                    }
                }

                // 单卡(或非 fp8 回退):legacy split 装载
                let split_qkv_legacy = {
                    let vb_qkv = vb.pp(projection_key_map["in_proj_qkv"]);
                    let (qc_qkv, q_qkv) =
                        Self::resolve_quant_for_weight(&vb_qkv, &quantization_config, &quant);
                    TensorParallelColumnLinear::load_with_hints(
                        hidden_size,
                        key_dim_global * 2 + value_dim_global,
                        false,
                        vb_qkv,
                        comm.clone(),
                        &qc_qkv,
                        &q_qkv,
                        dtype,
                    )
                };

                if let Ok(in_proj_qkv) = split_qkv_legacy {
                    return Ok(GdnProjection::SplitQkvZaLegacy {
                        in_proj_qkv,
                        in_proj_z,
                        in_proj_b,
                        in_proj_a,
                    });
                } else if let Err(err) = split_qkv_legacy {
                    load_errors.push(format!("in_proj_qkv: {err}"));
                }
            }
            (z, b, a) => {
                if let Err(err) = z {
                    load_errors.push(format!("in_proj_z: {err}"));
                }
                if let Err(err) = b {
                    load_errors.push(format!("in_proj_b: {err}"));
                }
                if let Err(err) = a {
                    load_errors.push(format!("in_proj_a: {err}"));
                }
            }
        }

        crate::bail!(
            "Unable to load Qwen3.5/Qwen3Next linear attention projection weights: {}",
            load_errors.join("; ")
        )
    }

    fn fix_qwen3next_projection_order(
        &self,
        mixed_qkvz: &Tensor,
        mixed_ba: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let seq_len = mixed_qkvz.dim(0)?;
        let qkvz_group_dim =
            self.head_k_dim + self.head_k_dim + self.kv_group_size * self.head_v_dim * 2;
        let ba_group_dim = 2 * self.kv_group_size;

        let mixed_qkvz = mixed_qkvz.reshape((seq_len, self.num_k_heads, qkvz_group_dim))?;
        let mixed_ba = mixed_ba.reshape((seq_len, self.num_k_heads, ba_group_dim))?;

        let mut offset = 0usize;
        let query = mixed_qkvz.narrow(2usize, offset, self.head_k_dim)?;
        offset += self.head_k_dim;
        let key = mixed_qkvz.narrow(2usize, offset, self.head_k_dim)?;
        offset += self.head_k_dim;
        let value = mixed_qkvz.narrow(2usize, offset, self.kv_group_size * self.head_v_dim)?;
        offset += self.kv_group_size * self.head_v_dim;
        let z = mixed_qkvz.narrow(2usize, offset, self.kv_group_size * self.head_v_dim)?;

        let b = mixed_ba.narrow(2usize, 0, self.kv_group_size)?;
        let a = mixed_ba.narrow(2usize, self.kv_group_size, self.kv_group_size)?;

        Ok((
            query.reshape((seq_len, self.key_dim))?,
            key.reshape((seq_len, self.key_dim))?,
            value.reshape((seq_len, self.value_dim))?,
            z.reshape((seq_len, self.value_dim))?,
            b.reshape((seq_len, self.num_v_heads))?,
            a.reshape((seq_len, self.num_v_heads))?,
        ))
    }

    fn project_inputs(
        &self,
        xs: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let xs = &if xs.dtype() != self.model_dtype {
            xs.to_dtype(self.model_dtype)?
        } else {
            xs.clone()
        };
        match &self.projection {
            GdnProjection::FusedQkvzBa {
                in_proj_qkvz,
                in_proj_ba,
            } => {
                let mixed_qkvz = in_proj_qkvz.forward(xs)?;
                let mixed_ba = in_proj_ba.forward(xs)?;
                self.fix_qwen3next_projection_order(&mixed_qkvz, &mixed_ba)
            }
            GdnProjection::SplitQkvZaLegacy {
                in_proj_qkv,
                in_proj_z,
                in_proj_b,
                in_proj_a,
            } => {
                let proj_qkv = in_proj_qkv.forward(xs)?;
                let q = proj_qkv.narrow(1usize, 0, self.key_dim)?.contiguous()?;
                let k = proj_qkv
                    .narrow(1usize, self.key_dim, self.key_dim)?
                    .contiguous()?;
                let v = proj_qkv
                    .narrow(1usize, self.key_dim * 2, self.value_dim)?
                    .contiguous()?;
                let z = in_proj_z.forward(xs)?;
                let b = in_proj_b.forward(xs)?;
                let a = in_proj_a.forward(xs)?;
                Ok((q, k, v, z, b, a))
            }
            GdnProjection::SplitQkvZaMerged {
                in_proj_qkv,
                in_proj_z,
                in_proj_b,
                in_proj_a,
            } => {
                let qkv = in_proj_qkv.forward(xs)?;
                if qkv.len() != 3 {
                    crate::bail!("Expected 3 chunks from merged in_proj_qkv, got {}", qkv.len());
                }
                let q = qkv[0].clone();
                let k = qkv[1].clone();
                let v = qkv[2].clone();
                let z = in_proj_z.forward(xs)?;
                let b = in_proj_b.forward(xs)?;
                let a = in_proj_a.forward(xs)?;
                Ok((q, k, v, z, b, a))
            }
        }
    }

    pub fn new(
        vb: VarBuilderX,
        comm: Rc<Comm>,
        config: &Config,
        gdn_layer_idx: usize,
        dtype: DType,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let hybrid = resolve_qwen3_hybrid_config(config);
        let world_size = comm.world_size;
        let rank = comm.rank;

        let num_v_heads_global = hybrid.num_v_heads;
        let num_k_heads_global = hybrid.num_k_heads;
        if num_v_heads_global % num_k_heads_global != 0 {
            crate::bail!(
                "linear_num_value_heads ({}) must be divisible by linear_num_key_heads ({})",
                num_v_heads_global,
                num_k_heads_global
            );
        }
        if num_v_heads_global % world_size != 0 || num_k_heads_global % world_size != 0 {
            crate::bail!(
                "linear attention heads must be divisible by tensor parallel world_size (num_v_heads={}, num_k_heads={}, world_size={})",
                num_v_heads_global,
                num_k_heads_global,
                world_size
            );
        }

        let is_quantized = config.quantization_config.is_some();
        let gdn_dtype = if vb.is_qvar_builder() || config.is_f16_mode {
            DType::F32
        } else {
            dtype
        };

        let num_v_heads = num_v_heads_global / world_size;
        let num_k_heads = num_k_heads_global / world_size;
        let head_k_dim = hybrid.key_head_dim;
        let head_v_dim = hybrid.value_head_dim;
        let key_dim_global = num_k_heads_global * head_k_dim;
        let value_dim_global = num_v_heads_global * head_v_dim;
        let key_dim = num_k_heads * head_k_dim;
        let value_dim = num_v_heads * head_v_dim;
        let kv_group_size = num_v_heads / num_k_heads;
        let conv_kernel_size = hybrid.conv_kernel_size;
        let conv_dim_global = key_dim_global * 2 + value_dim_global;

        // 学习参数(A_log/dt_bias)
        let sd = shard(0, comm.rank, comm.world_size);
        let gdn_pairs = [
            ("A_log", "ssm_a"),
            ("dt_bias", "ssm_dt.bias"),
            ("conv1d.weight", "ssm_conv1d.weight"),
            ("conv1d.bias", "ssm_conv1d.bias"),
            ("out_proj", "ssm_out"),
            ("norm.weight", "ssm_norm.weight"),
            ("norm.bias", "ssm_norm.bias"),
        ];
        let gdn_key_map = collect_key_map(vb.is_qvar_builder(), gdn_pairs);
        // GGUF 重排(a_log 恢复/undo_tiled)已由 loader 层吸收,直取
        let mut a_log =
            vb.get_with_hints_dtype((num_v_heads_global,), gdn_key_map["A_log"], sd, DType::F32)?;
        let mut dt_bias = vb.get_with_hints_dtype(
            (num_v_heads_global,),
            gdn_key_map["dt_bias"],
            sd,
            DType::F32,
        )?;
        if vb.is_qvar_builder() {
            a_log = tensor_parallel_chunk(&a_log, 0, rank, world_size, gdn_key_map["A_log"])?;
            dt_bias =
                tensor_parallel_chunk(&dt_bias, 0, rank, world_size, gdn_key_map["dt_bias"])?;
        }

        let projection = Self::load_projection(
            &vb,
            hidden_size,
            num_k_heads_global,
            key_dim_global,
            value_dim_global,
            num_v_heads_global,
            head_v_dim,
            comm.clone(),
            config,
            dtype,
            is_quantized,
        )?;

        // Conv1D 权重全局存;切 rank 本地 q/k/v 通道块
        let conv_weight = if vb.is_qvar_builder() {
            vb.get_with_hints_dtype(
                (conv_dim_global, conv_kernel_size),
                gdn_key_map["conv1d.weight"],
                Shard::default(),
                DType::F32,
            )?
            .unsqueeze(1usize)?
        } else {
            let w = vb.get(
                (conv_dim_global, 1, conv_kernel_size),
                gdn_key_map["conv1d.weight"],
            );
            match w {
                Ok(t) => t,
                Err(_) => {
                    // MLX 存 (out, kernel, 1);转 (out, 1, kernel)
                    vb.get(
                        (conv_dim_global, conv_kernel_size, 1),
                        gdn_key_map["conv1d.weight"],
                    )?
                    .permute(&[0usize, 2usize, 1usize])?
                }
            }
        };
        let q_start = rank * key_dim;
        let k_start = key_dim_global + rank * key_dim;
        let q_w = conv_weight.narrow(0usize, q_start, key_dim)?;
        let k_w = conv_weight.narrow(0usize, k_start, key_dim)?;
        let v_w = conv_weight.narrow(0usize, key_dim_global * 2, value_dim_global)?;
        let v_w = tensor_parallel_chunk(&v_w, 0, rank, world_size, "linear_attn.conv1d.weight[v]")?;
        let conv_weight = {
            let parts = [q_w, k_w, v_w];
            cat_local(&parts, 0)?.to_dtype(gdn_dtype)?
        };

        let conv_bias = vb.get((conv_dim_global,), gdn_key_map["conv1d.bias"]).ok();
        let conv_bias = if let Some(cb) = conv_bias {
            let q_b = cb.narrow(0usize, q_start, key_dim)?;
            let k_b = cb.narrow(0usize, k_start, key_dim)?;
            let v_b = cb.narrow(0usize, key_dim_global * 2, value_dim_global)?;
            let v_b =
                tensor_parallel_chunk(&v_b, 0, rank, world_size, "linear_attn.conv1d.bias[v]")?;
            let parts = [q_b, k_b, v_b];
            Some(cat_local(&parts, 0)?.to_dtype(gdn_dtype)?)
        } else {
            None
        };

        // 输出投影
        let out_proj = {
            let vb_out = vb.pp(gdn_key_map["out_proj"]);
            let (qc_out, q_out) = if is_quantized {
                Self::resolve_quant_for_weight(&vb_out, &config.quantization_config, &config.quant)
            } else {
                (None, None)
            };
            TensorParallelRowLinear::load_with_hints(
                value_dim_global,
                hidden_size,
                vb_out,
                comm.clone(),
                &qc_out,
                &q_out,
                dtype,
            )?
        };

        // GDN 输出归一(门控 RMSNorm;per-head 参数)
        let gdn_norm_weight = vb
            .get_with_hints_dtype(
                (head_v_dim,),
                gdn_key_map["norm.weight"],
                Shard::default(),
                DType::F32,
            )
            .map_err(|err| {
                crate::Error::Config(format!(
                    "Unable to load linear_attn.norm.weight as per-head [{head_v_dim}]: {err}"
                ))
            })?;
        let gdn_norm_bias = vb
            .get_with_hints_dtype(
                (head_v_dim,),
                gdn_key_map["norm.bias"],
                Shard::default(),
                DType::F32,
            )
            .ok();
        let scale = 1.0f64 / (head_k_dim as f64).sqrt();
        let d_conv = key_dim * 2 + value_dim;
        let (conv_mtp_state, recurrent_mtp_state) = if config.mtp_enabled {
            // 覆盖打包 batch verify:batch_size × (num_speculative + 1)
            let max_verify_tokens = config.mtp_max_verify_tokens.max(16);
            let conv = ctor_zeros(
                &[max_verify_tokens, d_conv, conv_kernel_size - 1],
                gdn_dtype,
            )?;
            let rec = ctor_zeros(
                &[max_verify_tokens, num_v_heads, head_k_dim, head_v_dim],
                DType::F32,
            )?;
            (Some(conv), Some(rec))
        } else {
            (None, None)
        };
        Ok(Self {
            projection,
            out_proj,
            conv_weight,
            conv_bias,
            a_log,
            dt_bias,
            gdn_norm_weight,
            gdn_norm_bias,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            key_dim,
            value_dim,
            kv_group_size,
            gdn_layer_idx,
            rms_norm_eps: config.rms_norm_eps,
            scale,
            gdn_dtype,
            model_dtype: if vb.is_qvar_builder() {
                DType::F32
            } else {
                dtype
            },
            gate_sigmoid: config
                .output_gate_type
                .as_deref()
                .is_some_and(|a| a.eq_ignore_ascii_case("sigmoid")),
            conv_mtp_state,
            recurrent_mtp_state,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        mamba_cache: &mut vendor::MambaCache,
        input_metadata: &vendor::InputMetadata,
        seq_slots: &Tensor,
    ) -> Result<Tensor> {
        let mut cache = MambaCacheHandle::new(mamba_cache);
        let slot_count = seq_slots.dim(0)?;
        if slot_count == 0 {
            crate::bail!("Linear attention requires non-empty sequence slots");
        }
        let original_dtype = xs.dtype();

        let (token_count, _hidden) = xs.dims2()?;
        let is_prefill = input_metadata.is_prefill;
        let (q, k, v, z, b, a) = self.project_inputs(xs)?;

        let (q, k, v, z, b, a) = if q.dtype() != self.gdn_dtype {
            (
                q.to_dtype(self.gdn_dtype)?,
                k.to_dtype(self.gdn_dtype)?,
                v.to_dtype(self.gdn_dtype)?,
                z.to_dtype(self.gdn_dtype)?,
                b.to_dtype(self.gdn_dtype)?,
                a.to_dtype(self.gdn_dtype)?,
            )
        } else {
            (q, k, v, z, b, a)
        };
        let mixed_qkv = cat_local(&[q, k, v], 1)?;

        let (kv_conv, prefill_conv_state) = if is_prefill {
            let mut conv_state = cache.get_batch_conv_state(self.gdn_layer_idx, seq_slots)?;
            let cu_seqlens = input_metadata
                .cu_seqlens_q
                .as_ref()
                .expect("cu_seqlens_q must be present in prefill!");

            let conv_snapshots = if input_metadata.is_mtp_verify {
                Some(
                    self.conv_mtp_state
                        .as_ref()
                        .ok_or_else(|| {
                            crate::Error::Schedule(format!(
                                "Missing MTP conv snapshot buffer for GDN layer {}",
                                self.gdn_layer_idx
                            ))
                        })?
                        .narrow(0usize, 0, token_count)?,
                )
            } else {
                None
            };
            let out = gdn_shim::causal_conv1d_fwd(
                &mixed_qkv,
                &self.conv_weight,
                self.conv_bias.as_ref(),
                &mut conv_state,
                conv_snapshots.as_ref(),
                cu_seqlens,
                true,
            )?;
            (out, Some(conv_state))
        } else {
            if token_count != slot_count {
                crate::bail!(
                    "Linear attention decode mismatch: {} tokens vs {} sequence slots",
                    token_count,
                    slot_count
                );
            }
            let out = gdn_shim::causal_conv1d_update_slots(
                &mixed_qkv,
                &self.conv_weight,
                self.conv_bias.as_ref(),
                cache.conv_state_mut(self.gdn_layer_idx),
                seq_slots,
                true,
            )?;
            (out, None)
        };
        if let Some(conv_state) = prefill_conv_state {
            cache.set_batch_conv_state(self.gdn_layer_idx, seq_slots, &conv_state)?;
        }

        // 卷积输出切回 q'/k'/v'
        let q_conv = kv_conv.narrow(1usize, 0, self.key_dim)?;
        let k_conv = kv_conv.narrow(1usize, self.key_dim, self.key_dim)?;
        let v_conv = kv_conv.narrow(1usize, self.key_dim * 2, self.value_dim)?;

        // 融合 GDN 门控
        let (a_expanded, b_expanded) = (a.unsqueeze(0usize)?, b.unsqueeze(0usize)?);
        let (g, beta) = gdn_shim::fused_gdn_gating(&self.a_log, &a_expanded, &b_expanded, &self.dt_bias)?;
        let (g, beta) = (g.squeeze(0usize)?, beta.squeeze(0usize)?);

        let q = q_conv.reshape((token_count, self.num_k_heads, self.head_k_dim))?;
        let k = k_conv.reshape((token_count, self.num_k_heads, self.head_k_dim))?;
        let v = v_conv.reshape((token_count, self.num_v_heads, self.head_v_dim))?;
        let q = gdn_shim::l2_norm_last_dim(&q, 1e-6)?;
        let k = gdn_shim::l2_norm_last_dim(&k, 1e-6)?;

        let output = if is_prefill {
            let cu_seqlens = input_metadata
                .cu_seqlens_q
                .as_ref()
                .expect("cu_seqlens_q must be present in prefill!");

            let global_state = cache.recurrent_state_mut(self.gdn_layer_idx);
            let recurrent_snapshots = if input_metadata.is_mtp_verify {
                Some(
                    self.recurrent_mtp_state
                        .as_ref()
                        .ok_or_else(|| {
                            crate::Error::Schedule(format!(
                                "Missing MTP recurrent snapshot buffer for GDN layer {}",
                                self.gdn_layer_idx
                            ))
                        })?
                        .narrow(0usize, 0, token_count)?,
                )
            } else {
                None
            };

            if self.num_k_heads != self.num_v_heads {
                // flashinfer 低精度 prefill 路径:flashinfer 引入后接通
                let flashinfer_result: Option<Tensor> = None;
                if let Some(out) = flashinfer_result {
                    out
                } else {
                    gdn_shim::gated_delta_rule_recurrence_varlen_gqa(
                        &q,
                        &k,
                        &v,
                        &g,
                        &beta,
                        global_state,
                        seq_slots,
                        cu_seqlens,
                        self.scale as f32,
                        recurrent_snapshots.as_ref(),
                    )?
                }
            } else {
                let q_scaled = q.affine(self.scale, 0.0)?;
                gdn_shim::gated_delta_rule_recurrence_varlen(
                    &q_scaled,
                    &k,
                    &v,
                    &g,
                    &beta,
                    global_state,
                    seq_slots,
                    cu_seqlens,
                    recurrent_snapshots.as_ref(),
                )?
            }
        } else {
            let batch = slot_count;
            let v_b = v.reshape((batch, self.num_v_heads, self.head_v_dim))?;
            let g_b = g.reshape((batch, self.num_v_heads))?;
            let beta_b = beta.reshape((batch, self.num_v_heads))?;
            let global_state = cache.recurrent_state_mut(self.gdn_layer_idx);
            let q_b = q.reshape((batch, self.num_k_heads, self.head_k_dim))?;
            let k_b = k.reshape((batch, self.num_k_heads, self.head_k_dim))?;
            gdn_shim::gated_delta_rule_decode_slots_gqa(
                &q_b,
                &k_b,
                &v_b,
                &g_b,
                &beta_b,
                global_state,
                seq_slots,
                self.scale as f32,
            )?
        };

        // [seq_len, num_v_heads, head_v_dim] -> [seq_len, value_dim]
        let output = output.reshape((token_count, self.value_dim))?;

        // 门控 RMSNorm:silu(Qwen3.5)/ sigmoid(Qwen4)
        let gated_output = if self.gate_sigmoid {
            gdn_shim::gated_rmsnorm_sigmoid_mul(
                &output,
                &z,
                &self.gdn_norm_weight,
                self.gdn_norm_bias.as_ref(),
                self.rms_norm_eps,
                self.head_v_dim,
            )?
        } else {
            gdn_shim::gated_rmsnorm_silu_mul(
                &output,
                &z,
                &self.gdn_norm_weight,
                self.gdn_norm_bias.as_ref(),
                self.rms_norm_eps,
                self.head_v_dim,
            )?
        };

        let out = self
            .out_proj
            .forward(&gated_output.to_dtype(self.model_dtype)?)?;
        if out.dtype() != original_dtype {
            out.to_dtype(original_dtype)
        } else {
            Ok(out)
        }
    }

    /// MTP verify 后状态回滚(保留 keep_tokens 个 token;快照区索引)
    pub fn rollback_mtp_verify(
        &self,
        mamba_cache: &mut vendor::MambaCache,
        seq_slots: &Tensor,
        keep_tokens: usize,
    ) -> Result<()> {
        self.rollback_mtp_verify_at(mamba_cache, seq_slots, keep_tokens, 0)
    }

    pub fn rollback_mtp_verify_at(
        &self,
        mamba_cache: &mut vendor::MambaCache,
        seq_slots: &Tensor,
        keep_tokens: usize,
        snapshot_offset: usize,
    ) -> Result<()> {
        let mut cache = MambaCacheHandle::new(mamba_cache);
        if keep_tokens == 0 {
            return Ok(());
        }
        let idx = snapshot_offset.checked_add(keep_tokens - 1).ok_or_else(|| {
            crate::Error::Schedule(format!(
                "MTP rollback index overflow for GDN layer {}",
                self.gdn_layer_idx
            ))
        })?;

        let conv_mtp_state = self.conv_mtp_state.as_ref().ok_or_else(|| {
            crate::Error::Schedule(format!(
                "Missing MTP conv snapshot buffer for GDN layer {} rollback",
                self.gdn_layer_idx
            ))
        })?;
        if idx >= conv_mtp_state.dim(0)? {
            crate::bail!(
                "MTP conv snapshot index {} out of range (buffer len {}, offset {}, keep {})",
                idx,
                conv_mtp_state.dim(0)?,
                snapshot_offset,
                keep_tokens
            );
        }
        let conv_snapshot = conv_mtp_state.narrow(0usize, idx, 1)?;
        let conv_state_dtype = cache.conv_state(self.gdn_layer_idx).dtype();
        let conv_snapshot = if conv_snapshot.dtype() != conv_state_dtype {
            conv_snapshot.to_dtype(conv_state_dtype)?
        } else {
            conv_snapshot
        };
        cache.set_batch_conv_state(self.gdn_layer_idx, seq_slots, &conv_snapshot)?;

        let recurrent_mtp_state = self.recurrent_mtp_state.as_ref().ok_or_else(|| {
            crate::Error::Schedule(format!(
                "Missing MTP recurrent snapshot buffer for GDN layer {} rollback",
                self.gdn_layer_idx
            ))
        })?;
        if idx >= recurrent_mtp_state.dim(0)? {
            crate::bail!(
                "MTP recurrent snapshot index {} out of range (buffer len {}, offset {}, keep {})",
                idx,
                recurrent_mtp_state.dim(0)?,
                snapshot_offset,
                keep_tokens
            );
        }
        let rec_snapshot = recurrent_mtp_state.narrow(0usize, idx, 1)?;
        cache.set_batch_recurrent_state(self.gdn_layer_idx, seq_slots, &rec_snapshot)?;

        Ok(())
    }
}

// ---- 本文件私有垫片(依赖需求见 mod trait 合并清单) ----

/// cat(missing trait 面;attention.rs missing_shims 同源需求)
fn cat_local(ts: &[Tensor], dim: usize) -> Result<Tensor> {
    unimplemented!("T3: Tensor::cat(需求清单 #1)")
}

/// zeros 构造(池直连工厂接入前的垫片;池上下文经 T3 loader 通道传入)
fn ctor_zeros(_shape: &[usize], _dtype: DType) -> Result<Tensor> {
    unimplemented!("T3: ctor_zeros(Pool 工厂接入)")
}

/// TP 切分(= xinfer tensor_parallel_chunk;体走 OwlTensor::narrow 回填)
pub(crate) fn tensor_parallel_chunk(
    _t: &Tensor,
    _dim: usize,
    _rank: usize,
    _world_size: usize,
    _name: &str,
) -> Result<Tensor> {
    unimplemented!("T3: tensor_parallel_chunk(narrow 通道回填)")
}

//! 注意力层(= xinfer layers/attention.rs 直译;T2-三 搬运)。
//!
//! 翻译注记:
//! - vendor attention_rs `PagedAttention` 面(new/forward)经本文件
//!   `pa_shim` 垫片调用(真实 kernel = attention-rs port,运行里程碑;
//!   vendor 段归宿裁决 2026-09-22:保留 attention-rs,只 port kernel);
//! - 量化装载(fp8 packed qkv)类型/分派骨架完整,体 = marlin-ffi 路线;
//! - OwlTensor 缺面(mean_keepdim/flatten(0,1)/expand/cat 等)经本文件
//!   `missing_shims` 占位,需求清单见文件尾注释(待主 agent 合并进 trait)。

use super::distributed::{
    kv_head_shard, shard, Comm, MergedParallelColumnLinear, ReplicatedLinear,
    TensorParallelColumnLinear, TensorParallelRowLinear,
};
use super::others::{rms_norm, rms_norm_sharded, NormX};
use super::rotary_emb::ApplyRotaryEmbedding;
use super::vendor;
use super::{DType, OwlTensor, Result, Shard, Shape, Tensor, VarBuilderX};
use crate::config::Config;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

// ---- OwlTensor 缺面垫片(需求清单待合并;体 T3 回填) ----
mod missing_shims {
    use super::{Result, Tensor};

    /// 标量常量 → 池内 [1] f32 张量(dry-run 装载基元)
    fn const_tensor(v: f32) -> Result<Tensor> {
        let pool = crate::models::layers::ctx_scope::weights_pool();
        let t = owl_nn::TensorPoolOps::from_vec_tensor(pool.as_ref(), &[1], vec![v])?;
        Ok(owl_nn::DynTensor::from_f32(&t))
    }
    /// candle `Tensor::cat`(沿 dim 拼接)→ erased::cat
    pub fn cat(ts: &[Tensor], dim: usize) -> Result<Tensor> {
        crate::models::layers::ctx_scope::with(|_ops, ctx| {
            owl_nn::erased::cat(ctx, ts, dim).map_err(Into::into)
        })
    }
    /// candle `Tensor::flatten(from,to)`(全区间展平 = [elems])
    pub fn flatten(t: &Tensor, _from: usize, _to: usize) -> Result<Tensor> {
        let n: usize = t.shape().iter().product();
        t.reshape(&[n]).map_err(Into::into)
    }
    /// candle `mean_keepdim(D::Minus1)`(末维均值保维)→ sum_dim + 标量乘
    pub fn mean_keepdim_last(t: &Tensor) -> Result<Tensor> {
        crate::models::layers::ctx_scope::with(|ops, ctx| {
            let d = *t.shape().last().ok_or_else(|| {
                crate::Error::Msg("mean_keepdim_last: 空形状".into())
            })? as f32;
            let s = owl_nn::erased::sum_dim(ops, ctx, t, t.shape().len() - 1)
                .map_err(crate::Error::from)?;
            let inv_t = ctx.scratch_tensor::<f32>(&[1])?;
            let inv = owl_nn::DynTensor::from_f32(&inv_t);
            owl_nn::erased::copy_d2d_to_raw(
                ctx,
                &const_tensor(1.0f32 / d)?,
                inv.device_ptr() as *mut core::ffi::c_void,
                4,
            )?;
            owl_nn::erased::mul(ops, ctx, &s, &inv).map_err(crate::Error::from)
        })
        .map_err(Into::into)
    }
    /// candle `expand`(广播到目标 shape;元数据视图)
    pub fn expand(t: &Tensor, shape: &[usize]) -> Result<Tensor> {
        crate::models::layers::OwlTensor::broadcast_as(t, shape)
    }
    /// candle `broadcast_add(rhs_scalar_tensor)` 的标量便捷面(标量=[1] 张量)
    pub fn broadcast_add_const(t: &Tensor, v: f64) -> Result<Tensor> {
        crate::models::layers::ctx_scope::with(|ops, ctx| {
            let one_t = ctx.scratch_tensor::<f32>(&[1])?;
            let one = owl_nn::DynTensor::from_f32(&one_t);
            owl_nn::erased::copy_d2d_to_raw(
                ctx,
                &const_tensor(v as f32)?,
                one.device_ptr() as *mut core::ffi::c_void,
                4,
            )?;
            owl_nn::erased::broadcast_add(ops, ctx, t, &one).map_err(crate::Error::from)
        })
        .map_err(Into::into)
    }
    /// candle `dims3()`(元数据)
    pub fn dims3(t: &Tensor) -> Result<(usize, usize, usize)> {
        crate::models::layers::OwlTensor::dims3(t)
    }
    /// candle `get_llama4_attn_scale`(utils 自由函数;位置相关注意力缩放)
    pub fn llama4_attn_scale(
        _positions: &Tensor,
        _beta: f64,
        _orig_max: f64,
    ) -> Result<Tensor> {
        unimplemented!("T3: get_llama4_attn_scale(missing_shims)")
    }
}

// ---- vendor PagedAttention 垫片(真实 kernel = attention-rs port,T3) ----
mod pa_shim {
    use super::vendor::InputMetadata;
    use super::{Result, Tensor};

    /// decode naive 路径垫片(dry-run;真核 = attention-rs port,K2 后替换)。
    /// KV 布局:flat [max_slots, Hkv*D](slot 直排,与 dry_kernels 核一致);
    /// slots/kv_lens = bindings 设备指针(u32 位型 = i32,值域 <2^31)。
    pub struct PagedAttention {
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    }

    /// [A,B,C] → [A,C,B](dry transpose12 核)
    fn transpose12(t: &Tensor, a: usize, b: usize, c: usize) -> Result<Tensor> {
        crate::models::layers::ctx_scope::with_dry(|ctx, dry| {
            let out_t = ctx.scratch_tensor::<f32>(&[a, c, b])?;
            let out = owl_nn::DynTensor::from_f32(&out_t);
            dry.transpose12_f32(
                ctx.stream(),
                t.device_ptr() as *const f32,
                out.device_ptr() as *mut f32,
                a,
                b,
                c,
            )
            .map_err(|e| crate::Error::Msg(format!("transpose12: {e}")))?;
            ctx.trace_launch("transpose12");
            Ok(out)
        })
    }

    /// [A,B,C] → [B,A,C](dry transpose01 核;permute stride 语义未落地)
    fn transpose01(t: &Tensor, a: usize, b: usize, c: usize) -> Result<Tensor> {
        crate::models::layers::ctx_scope::with_dry(|ctx, dry| {
            let out_t = ctx.scratch_tensor::<f32>(&[b, a, c])?;
            let out = owl_nn::DynTensor::from_f32(&out_t);
            dry.transpose01_f32(
                ctx.stream(),
                t.device_ptr() as *const f32,
                out.device_ptr() as *mut f32,
                a,
                b,
                c,
            )
            .map_err(|e| crate::Error::Msg(format!("transpose01: {e}")))?;
            ctx.trace_launch("transpose01");
            Ok(out)
        })
    }

    impl PagedAttention {
        #[allow(clippy::too_many_arguments)]
        pub fn new(
            num_heads: usize,
            head_dim: usize,
            _scale: f32,
            num_kv_heads: Option<usize>,
            _sliding_window: Option<usize>,
            _device: Option<()>,
            _fp8_kvcache: bool,
        ) -> Result<Self> {
            Ok(Self {
                num_heads,
                num_kv_heads: num_kv_heads.unwrap_or(num_heads),
                head_dim,
            })
        }

        #[allow(clippy::too_many_arguments)]
        pub fn forward(
            &self,
            q: &Tensor,
            k: &Tensor,
            v: &Tensor,
            mask: Option<&Vec<Tensor>>,
            k_cache: Option<Tensor>,
            v_cache: Option<Tensor>,
            meta: &InputMetadata,
            _softcap: Option<f64>,
        ) -> Result<Tensor> {
            // 分派:decode = naive 核;prefill = eager 逐序列(causal mask + GQA 展开
            // + KV 落槽;M-Ⅰ 实接,2026-09-23)。两路均在 with_dry 面。
            let (seq_len, _hq, _d) = crate::models::layers::OwlTensor::dims3(q)?;
            match (&meta.decode_ptrs, k_cache.clone(), v_cache.clone()) {
                (Some(p), Some(kc), Some(vc)) => {
                    Self::forward_decode(q, k, v, kc, vc, *p, meta, seq_len)
                }
                (None, Some(kc), Some(vc)) if meta.is_prefill => {
                    Self::forward_prefill_eager(q, k, v, kc, vc, mask, meta, seq_len)
                }
                _ => crate::bail!(
                    "pa_shim: decode 需要 meta.decode_ptrs + kv cache 对(空跑面;prefill eager 另案)"
                ),
            }
        }

        /// decode naive 路径(原实现搬入)
        #[allow(clippy::too_many_arguments)]
        fn forward_decode(
            q: &Tensor,
            k: &Tensor,
            v: &Tensor,
            kc: Tensor,
            vc: Tensor,
            ptrs: crate::models::layers::vendor::DecodePtrs,
            meta: &InputMetadata,
            seq_len: usize,
        ) -> Result<Tensor> {
            use crate::models::layers::{ctx_scope, erased, OwlTensor};
            ctx_scope::with_dry(|ctx, dry| {
                let out_t = ctx.scratch_tensor::<f32>(&[seq_len, q.shape()[2] * q.shape()[1]])?;
                let out = owl_nn::DynTensor::from_f32(&out_t);
                dry.naive_decode_attn_f32(
                    ctx.stream(),
                    q.device_ptr() as *const f32,
                    k.device_ptr() as *const f32,
                    v.device_ptr() as *const f32,
                    kc.device_ptr() as *mut f32,
                    vc.device_ptr() as *mut f32,
                    ptrs.slots,
                    ptrs.kv_lens,
                    out.device_ptr() as *mut f32,
                    seq_len,
                    q.shape()[1],
                    k.shape()[1],
                    q.shape()[2],
                )
                .map_err(|e| crate::Error::Msg(format!("naive_decode_attn: {e}")))?;
                ctx.trace_launch("naive_decode_attn");
                Ok(out)
            })
        }

        /// prefill eager:逐序列 naive attention(candle 形态,[H,len,D] 面)。
        /// 单序列或 varlen 打包(cu_seqlens/seqlens 均可);KV 写回 cache 槽行
        /// (行号 = token 全局位;M-Ⅰ 冒烟口径:fresh cache,槽 = 位置)。
        #[allow(clippy::too_many_arguments)]
        fn forward_prefill_eager(
            q: &Tensor,
            k: &Tensor,
            v: &Tensor,
            kc: Tensor,
            vc: Tensor,
            mask: Option<&Vec<Tensor>>,
            meta: &InputMetadata,
            _total_tokens: usize,
        ) -> Result<Tensor> {
            use crate::models::layers::ops;
            use crate::models::layers::{ctx_scope, erased, OwlTensor};
            let (_t, hq, d) = {
                let d3 = OwlTensor::dims(q)?;
                (d3[0], d3[1], d3[2])
            };
            let hkv = OwlTensor::dims(k)?[1];
            let scale = f64::from((d as f32).powf(-0.5));
            let group = hq / hkv;
            let seqlens = if !meta.seqlens.is_empty() {
                meta.seqlens.clone()
            } else {
                vec![OwlTensor::dims(q)?[0]]
            };
            let mut outs = Vec::new();
            let mut start = 0usize;
            for (si, &len) in seqlens.iter().enumerate() {
                let q_s = q.narrow(0usize, start, len)?;
                let k_s = k.narrow(0usize, start, len)?;
                let v_s = v.narrow(0usize, start, len)?;
                // [len, H, D] → [H, len, D]
                let qh = transpose01(
                    &q_s.reshape((len, hq, d))?.contiguous()?,
                    len,
                    hq,
                    d,
                )?;
                let mut kh = transpose01(
                    &k_s.reshape((len, hkv, d))?.contiguous()?,
                    len,
                    hkv,
                    d,
                )?;
                let mut vh = transpose01(
                    &v_s.reshape((len, hkv, d))?.contiguous()?,
                    len,
                    hkv,
                    d,
                )?;
                if group > 1 {
                    // GQA 展开:每个 kv 头连续重复 group 次(h → h/group 映射)
                    let k_parts: Vec<Tensor> = (0..group).map(|_| kh.clone()).collect();
                    let v_parts: Vec<Tensor> = (0..group).map(|_| vh.clone()).collect();
                    kh = ops::cat(&k_parts, 0usize)?;
                    vh = ops::cat(&v_parts, 0usize)?;
                }
                // 写 cache 槽行(行 = token 全局位 start+i)
                let row = hkv * d;
                let k_flat = k_s.reshape((len, row))?.contiguous()?;
                let v_flat = v_s.reshape((len, row))?.contiguous()?;
                ctx_scope::with_dry(|ctx, _dry| {
                    for i in 0..len {
                        let slot_row = start + i;
                        let ki = k_flat.narrow(0usize, i, 1)?;
                        let vi = v_flat.narrow(0usize, i, 1)?;
                        erased::copy_d2d_to_raw(
                            ctx,
                            &ki,
                            unsafe { kc.device_ptr().add(slot_row * row * 4) } as *mut core::ffi::c_void,
                            row * 4,
                        )?;
                        erased::copy_d2d_to_raw(
                            ctx,
                            &vi,
                            unsafe { vc.device_ptr().add(slot_row * row * 4) } as *mut core::ffi::c_void,
                            row * 4,
                        )?;
                    }
                    Ok(())
                })?;
                // 注意力:matmul 一期 2D → 逐头循环
                // qh [Hq,len,D] / kh [Hkv,len,D](已 GQA 展开)/ vh [Hkv,len,D]
                let qh = qh.contiguous()?;
                let kh = kh.contiguous()?;
                let vh = vh.contiguous()?;
                let m2 = match mask.map(|ms| ms[si].reshape((len, len))).transpose() { Ok(m) => m, Err(e) => return Err(e.into()) };
                let mut head_outs = Vec::with_capacity(hq);
                for h in 0..hq {
                    let qh2 = qh.narrow(0usize, h, 1)?.reshape((len, d))?;
                    let kh2 = kh.narrow(0usize, h / group, 1)?.reshape((d, len))?;
                    let mut att = qh2.matmul(&kh2)?; // [len,len]
                    att = att.affine(scale, 0.0)?;
                    if let Some(m) = &m2 {
                        att = att.broadcast_add(m)?;
                    }
                    let att = ops::softmax_last_dim(&att)?;
                    let vh2 = vh.narrow(0usize, h / group, 1)?.reshape((len, d))?;
                    let o = att.matmul(&vh2)?; // [len,D]
                    head_outs.push(o.reshape((1usize, len, d))?);
                }
                let ctx_out = if head_outs.len() == 1 {
                    head_outs.into_iter().next().unwrap()
                } else {
                    ops::cat(&head_outs, 0usize)? // [Hq, len, D]
                };
                let out = transpose01(&ctx_out.contiguous()?, hq, len, d)? // [len, Hq, D]
                    .reshape((len, hq * d))?;
                outs.push(out);
                start += len;
            }
            if outs.len() == 1 {
                Ok(outs.into_iter().next().unwrap())
            } else {
                ops::cat(&outs, 0usize)
            }
        }
    }
}

enum QkvProjection {
    Separate {
        q_proj: TensorParallelColumnLinear,
        k_proj: TensorParallelColumnLinear,
        v_proj: TensorParallelColumnLinear,
    },
    Packed(MergedParallelColumnLinear),
}

pub struct Attention {
    qkv_proj: QkvProjection,
    o_proj: TensorParallelRowLinear,
    q_norm: Option<NormX>,
    k_norm: Option<NormX>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    attn_output_gate: bool,
    attn: pa_shim::PagedAttention,
    softcapping: Option<f64>,
    dtype: DType,
    full_dim_qk_norm: bool,
    is_qvar_builder: bool,
    qk_l2_norm: bool,
    promote_qk_to_f32: bool,
    v_norm_eps: Option<f64>,
}

impl Attention {
    fn normalize_sharded_2d(
        t: Tensor,
        shard: Shard,
        global_dim0: usize,
        global_dim1: usize,
        name: &str,
    ) -> Result<Tensor> {
        if shard.world_size <= 1 {
            return Ok(t);
        }
        if shard.dim > 1 {
            crate::bail!("unexpected shard dim {} for {}", shard.dim, name);
        }
        let (d0, d1) = t.dims2()?;
        if shard.dim == 0 {
            let local = global_dim0 / shard.world_size;
            if d0 == local {
                return Ok(t);
            }
            if d0 == global_dim0 {
                return t.narrow(0usize, shard.rank * local, local)?.contiguous();
            }
            crate::bail!(
                "unexpected {} shape ({}, {}), shard dim 0 expects local {} or global {}",
                name, d0, d1, local, global_dim0
            );
        }
        let local = global_dim1 / shard.world_size;
        if d1 == local {
            return Ok(t);
        }
        if d1 == global_dim1 {
            return t.narrow(1usize, shard.rank * local, local)?.contiguous();
        }
        crate::bail!(
            "unexpected {} shape ({}, {}), shard dim 1 expects local {} or global {}",
            name, d0, d1, local, global_dim1
        );
    }

    fn normalize_sharded_1d(
        t: Tensor,
        shard: Shard,
        global_dim: usize,
        name: &str,
    ) -> Result<Tensor> {
        if shard.world_size <= 1 {
            return Ok(t);
        }
        let d0 = t.dim(0)?;
        let local = global_dim / shard.world_size;
        if d0 == local {
            return Ok(t);
        }
        if d0 == global_dim {
            return t.narrow(0usize, shard.rank * local, local)?.contiguous();
        }
        crate::bail!(
            "unexpected {} shape ({}), expects local {} or global {}",
            name, d0, local, global_dim
        );
    }

    fn load_sharded_bias(
        vb: &VarBuilderX,
        out_dim: usize,
        shard: Shard,
        dtype: DType,
    ) -> Result<Option<Tensor>> {
        if vb.is_qvar_builder() {
            return Ok(None);
        }
        let Ok(bias) = vb.get_with_hints_dtype((out_dim,), "bias", shard, DType::F32) else {
            return Ok(None);
        };
        let bias = Self::normalize_sharded_1d(bias, shard, out_dim, "bias")?;
        if bias.dtype() != dtype {
            Ok(Some(bias.to_dtype(dtype)?))
        } else {
            Ok(Some(bias))
        }
    }

    fn try_load_sharded_fp8_weight_scale(
        vb: &VarBuilderX,
        out_dim: usize,
        in_dim: usize,
        shard: Shard,
        block_size: &[usize],
    ) -> Result<Option<(Tensor, Tensor)>> {
        if !vb.has_key("weight_scale") && !vb.has_key("weight_scale_inv") {
            return Ok(None);
        }
        let by = block_size[0];
        let bx = block_size[1];
        let scale_dim0 = out_dim.div_ceil(by);
        let scale_dim1 = in_dim.div_ceil(bx);

        let weight = match vb
            .get_with_hints_dtype((out_dim, in_dim), "weight", shard, DType::U8) // F8E4M3 变体待 Dtype 扩容(需求清单)
        {
            Ok(w) => w,
            Err(_) => match vb.get_with_hints_dtype((out_dim, in_dim), "weight", shard, DType::U8)
            {
                Ok(w) => w,
                Err(_) => return Ok(None),
            },
        };
        let weight = Self::normalize_sharded_2d(weight, shard, out_dim, in_dim, "weight")?;

        let weight_scale = match vb.get_with_hints_dtype(
            (scale_dim0, scale_dim1),
            "weight_scale",
            shard,
            DType::F32,
        ) {
            Ok(s) => s,
            Err(_) => match vb.get_with_hints_dtype(
                (scale_dim0, scale_dim1),
                "weight_scale_inv",
                shard,
                DType::F32,
            ) {
                Ok(s) => s,
                Err(_) => return Ok(None),
            },
        };
        let weight_scale = Self::normalize_sharded_2d(
            weight_scale,
            shard,
            scale_dim0,
            scale_dim1,
            "weight_scale",
        )?;
        Ok(Some((weight, weight_scale)))
    }

    #[allow(clippy::too_many_arguments)]
    fn try_load_packed_qkv(
        vb: &VarBuilderX,
        hidden_size: usize,
        q_out_dim: usize,
        kv_out_dim: usize,
        attention_bias: bool,
        comm: Rc<Comm>,
        kv_shard: Shard,
        dtype: DType,
        quant_cfg: &Option<crate::config::QuantConfig>,
        quant: &Option<String>,
        k_eq_v: bool,
    ) -> Result<Option<QkvProjection>> {
        if vb.is_qvar_builder() || quant.is_some() {
            return Ok(None);
        }

        let q_shard = shard(0, comm.rank, comm.world_size);
        let q_vb = vb.pp("q_proj");
        let k_vb = vb.pp("k_proj");
        let v_vb = if k_eq_v { vb.pp("k_proj") } else { vb.pp("v_proj") };

        let is_fp8_quant = quant_cfg
            .as_ref()
            .map(|cfg| cfg.quant_method == "fp8")
            .unwrap_or(false);
        if let Some(cfg) = quant_cfg {
            if cfg.quant_method != "fp8" {
                return Ok(None);
            }
        }

        if is_fp8_quant {
            let Some(block_size) = quant_cfg
                .as_ref()
                .and_then(|cfg| cfg.weight_block_size.clone())
            else {
                crate::bail!("LnFp8: weight_block_size must be configured for packed qkv");
            };
            if block_size.len() != 2 {
                crate::bail!("LnFp8: weight_block_size must have 2 elements");
            }

            let Some((q_weight, q_scale)) = Self::try_load_sharded_fp8_weight_scale(
                &q_vb, q_out_dim, hidden_size, q_shard, &block_size,
            )?
            else {
                return Ok(None);
            };
            let Some((k_weight, k_scale)) = Self::try_load_sharded_fp8_weight_scale(
                &k_vb, kv_out_dim, hidden_size, kv_shard, &block_size,
            )?
            else {
                return Ok(None);
            };
            let Some((v_weight, v_scale)) = Self::try_load_sharded_fp8_weight_scale(
                &v_vb, kv_out_dim, hidden_size, kv_shard, &block_size,
            )?
            else {
                return Ok(None);
            };

            let local_q = q_weight.dim(0)?;
            let local_k = k_weight.dim(0)?;
            let local_v = v_weight.dim(0)?;
            let by = block_size[0];
            let q_global_start = q_shard.rank * local_q;
            let k_global_start = q_out_dim + kv_shard.rank * local_k;
            let v_global_start = q_out_dim + kv_out_dim + kv_shard.rank * local_v;
            if q_global_start % by != 0 || k_global_start % by != 0 || v_global_start % by != 0 {
                return Ok(None);
            }

            let packed_weight = missing_shims::cat(&[q_weight, k_weight, v_weight], 0)?;
            let packed_scale = missing_shims::cat(&[q_scale, k_scale, v_scale], 0)?;
            let packed_bias = if attention_bias {
                let q_bias = Self::load_sharded_bias(&q_vb, q_out_dim, q_shard, dtype)?;
                let k_bias = Self::load_sharded_bias(&k_vb, kv_out_dim, kv_shard, dtype)?;
                let v_bias = Self::load_sharded_bias(&v_vb, kv_out_dim, kv_shard, dtype)?;
                match (q_bias, k_bias, v_bias) {
                    (Some(qb), Some(kb), Some(vb_b)) => {
                        Some(missing_shims::cat(&[qb, kb, vb_b], 0)?)
                    }
                    (None, None, None) => None,
                    _ => return Ok(None),
                }
            } else {
                None
            };

            // sm_version:owl 基座 cuda-only;sm 探测 = owl-cuda ffi 面待接线
            let sm_version = 0usize;

            let merged = MergedParallelColumnLinear::from_packed_local_fp8(
                packed_weight,
                packed_scale,
                packed_bias,
                block_size,
                sm_version,
                vec![local_q, local_k, local_v],
            )?;
            return Ok(Some(QkvProjection::Packed(merged)));
        }

        if quant_cfg.is_some() {
            return Ok(None);
        }

        let q_weight =
            q_vb.get_with_hints_dtype((q_out_dim, hidden_size), "weight", q_shard, dtype)?;
        let k_weight =
            k_vb.get_with_hints_dtype((kv_out_dim, hidden_size), "weight", kv_shard, dtype)?;
        let v_weight =
            v_vb.get_with_hints_dtype((kv_out_dim, hidden_size), "weight", kv_shard, dtype)?;

        let local_q = q_weight.dim(0)?;
        let local_k = k_weight.dim(0)?;
        let local_v = v_weight.dim(0)?;
        let packed_weight = missing_shims::cat(&[q_weight, k_weight, v_weight], 0)?;

        let packed_bias = if attention_bias {
            let q_bias = Self::load_sharded_bias(&q_vb, q_out_dim, q_shard, dtype)?;
            let k_bias = Self::load_sharded_bias(&k_vb, kv_out_dim, kv_shard, dtype)?;
            let v_bias = Self::load_sharded_bias(&v_vb, kv_out_dim, kv_shard, dtype)?;
            match (q_bias, k_bias, v_bias) {
                (Some(qb), Some(kb), Some(vb_b)) => {
                    Some(missing_shims::cat(&[qb, kb, vb_b], 0)?)
                }
                (None, None, None) => None,
                _ => return Ok(None),
            }
        } else {
            None
        };

        let merged = MergedParallelColumnLinear::from_packed_local(
            packed_weight,
            packed_bias,
            vec![local_q, local_k, local_v],
        )?;
        Ok(Some(QkvProjection::Packed(merged)))
    }

    pub fn new(
        vb: VarBuilderX,
        comm: Rc<Comm>,
        config: &Config,
        attention_scale: Option<f32>,
        sliding_window: Option<usize>,
        dtype: DType,
    ) -> Result<Self> {
        Self::new_with_options(
            vb, comm, config, attention_scale, sliding_window, dtype, false, false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_options(
        vb: VarBuilderX,
        comm: Rc<Comm>,
        config: &Config,
        attention_scale: Option<f32>,
        sliding_window: Option<usize>,
        dtype: DType,
        k_eq_v: bool,
        qk_l2_norm: bool,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim.unwrap_or(hidden_size / num_heads);
        let is_qvar_builder = vb.is_qvar_builder();
        let arch = config
            .architectures
            .as_ref()
            .and_then(|a| a.first().cloned())
            .unwrap_or_default();
        let is_qwen35_or_next = matches!(
            arch.as_str(),
            "Qwen3_5ForCausalLM"
                | "Qwen3_5ForConditionalGeneration"
                | "Qwen3_5MoeForCausalLM"
                | "Qwen3_5MoeForConditionalGeneration"
                | "Qwen3NextForCausalLM"
                | "Qwen3NextForConditionalGeneration"
                | "Qwen4ExpForConditionalGeneration"
                | "Qwen4ExpForCausalLM"
        );
        let attention_bias = if is_qwen35_or_next {
            config.qkv_bias.or(config.attention_bias).unwrap_or(false)
        } else {
            config.qkv_bias.or(config.attention_bias).unwrap_or(true)
        };
        let attn_output_gate = if is_qwen35_or_next {
            config.attn_output_gate.unwrap_or(true)
        } else {
            config.attn_output_gate.unwrap_or(false)
        };
        let q_out_dim = num_heads * head_dim * if attn_output_gate { 2 } else { 1 };
        let is_gemma = arch == "Gemma3ForConditionalGeneration"
            || arch == "Gemma3ForCausalLM";
        let is_mlx_nvfp4 = config
            .quantization_config
            .as_ref()
            .is_some_and(|q| q.is_mlx_nvfp4);
        let qk_norm_add_one = is_gemma || (is_qwen35_or_next && !is_qvar_builder && !is_mlx_nvfp4);

        let world_size = comm.world_size;
        let attention_heads = num_heads / world_size;
        let (kv_heads, kv_shard) = kv_head_shard(num_kv_heads, comm.rank, world_size)?;

        let qkv_proj = if let Some(packed) = Self::try_load_packed_qkv(
            &vb,
            hidden_size,
            q_out_dim,
            num_kv_heads * head_dim,
            attention_bias,
            comm.clone(),
            kv_shard,
            dtype,
            &config.quantization_config,
            &config.quant,
            k_eq_v,
        )? {
            packed
        } else {
            let q_proj = TensorParallelColumnLinear::load_with_hints(
                hidden_size,
                q_out_dim,
                attention_bias,
                if is_qvar_builder { vb.pp("attn_q") } else { vb.pp("q_proj") },
                comm.clone(),
                &config.quantization_config,
                &config.quant,
                dtype,
            )?;
            let k_proj = TensorParallelColumnLinear::load_with_shard(
                hidden_size,
                num_kv_heads * head_dim,
                attention_bias,
                if is_qvar_builder { vb.pp("attn_k") } else { vb.pp("k_proj") },
                kv_shard,
                &config.quantization_config,
                &config.quant,
                dtype,
            )?;
            let v_proj = TensorParallelColumnLinear::load_with_shard(
                hidden_size,
                num_kv_heads * head_dim,
                attention_bias,
                if is_qvar_builder {
                    if k_eq_v { vb.pp("attn_k") } else { vb.pp("attn_v") }
                } else if k_eq_v {
                    vb.pp("k_proj")
                } else {
                    vb.pp("v_proj")
                },
                kv_shard,
                &config.quantization_config,
                &None,
                dtype,
            )?;
            QkvProjection::Separate {
                q_proj,
                k_proj,
                v_proj,
            }
        };

        let o_proj = TensorParallelRowLinear::load_with_hints(
            num_heads * head_dim,
            hidden_size,
            if is_qvar_builder { vb.pp("attn_output") } else { vb.pp("o_proj") },
            comm.clone(),
            &config.quantization_config,
            &config.quant,
            dtype,
        )?;

        let norm_dtype = if is_qvar_builder
            || config.quant.is_some()
            || config.quantization_config.is_some()
            || config.is_f16_mode
        {
            DType::F32
        } else {
            dtype
        };
        let q_norm_vb = if is_qvar_builder { vb.pp("attn_q_norm") } else { vb.pp("q_norm") };
        let k_norm_vb = if is_qvar_builder { vb.pp("attn_k_norm") } else { vb.pp("k_norm") };

        let q_norm = rms_norm(head_dim, config.rms_norm_eps, q_norm_vb.clone(), norm_dtype, qk_norm_add_one);
        let k_norm = rms_norm(head_dim, config.rms_norm_eps, k_norm_vb.clone(), norm_dtype, qk_norm_add_one);

        let (q_norm, k_norm, full_dim_qk_norm) = if q_norm.is_ok() && k_norm.is_ok() {
            (Some(q_norm.unwrap()), Some(k_norm.unwrap()), false)
        } else {
            let q_shard = shard(0, comm.rank, comm.world_size);
            let q_full = rms_norm_sharded(
                num_heads * head_dim,
                config.rms_norm_eps,
                q_norm_vb,
                norm_dtype,
                qk_norm_add_one,
                q_shard,
            );
            let k_full = rms_norm_sharded(
                num_kv_heads * head_dim,
                config.rms_norm_eps,
                k_norm_vb,
                norm_dtype,
                qk_norm_add_one,
                kv_shard,
            );
            if q_full.is_ok() && k_full.is_ok() {
                (Some(q_full.unwrap()), Some(k_full.unwrap()), true)
            } else {
                (None, None, false)
            }
        };

        let is_gemma4 = arch == "Gemma4ForConditionalGeneration" || arch == "Gemma4ForCausalLM";
        let v_norm_eps = if is_gemma4 { Some(config.rms_norm_eps) } else { None };

        let attn = pa_shim::PagedAttention::new(
            attention_heads,
            head_dim,
            attention_scale.unwrap_or(1. / (head_dim as f32).sqrt()),
            Some(kv_heads),
            sliding_window,
            None,
            config.kvcache_dtype.is_fp8_keys(),
        )?;

        Ok(Self {
            qkv_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads: attention_heads,
            num_kv_heads: kv_heads,
            head_dim,
            attn_output_gate,
            attn,
            softcapping: config.attn_logit_softcapping,
            dtype,
            full_dim_qk_norm,
            is_qvar_builder,
            qk_l2_norm,
            promote_qk_to_f32: is_qvar_builder || config.higher_precision_required() || qk_l2_norm,
            v_norm_eps,
        })
    }

    /// Promote Q/K to F32 when needed and apply per-token or per-head RMSNorm before RoPE.
    fn prepare_qk_for_rope(&self, q: Tensor, k: Tensor, seq_len: usize) -> Result<(Tensor, Tensor)> {
        let (q, k) = if self.promote_qk_to_f32 && q.dtype() != DType::F32 {
            (q.to_dtype(DType::F32)?, k.to_dtype(DType::F32)?)
        } else {
            (q, k)
        };

        let (q, k) = if self.q_norm.is_some() && self.k_norm.is_some() {
            if self.full_dim_qk_norm {
                let q_2d = q.reshape((seq_len, self.num_heads * self.head_dim))?;
                let k_2d = k.reshape((seq_len, self.num_kv_heads * self.head_dim))?;
                let q_2d = self.q_norm.as_ref().unwrap().forward(&q_2d)?;
                let k_2d = self.k_norm.as_ref().unwrap().forward(&k_2d)?;
                (
                    q_2d.reshape((seq_len, self.num_heads, self.head_dim))?,
                    k_2d.reshape((seq_len, self.num_kv_heads, self.head_dim))?,
                )
            } else {
                let q_flat = missing_shims::flatten(&q, 0, 1)?;
                let k_flat = missing_shims::flatten(&k, 0, 1)?;
                let q_flat = self.q_norm.as_ref().unwrap().forward(&q_flat)?;
                let k_flat = self.k_norm.as_ref().unwrap().forward(&k_flat)?;
                (
                    q_flat.reshape((seq_len, self.num_heads, self.head_dim))?,
                    k_flat.reshape((seq_len, self.num_kv_heads, self.head_dim))?,
                )
            }
        } else {
            (q, k)
        };
        Ok((q, k))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        xs: &Tensor,
        rotary_emb: &Option<Arc<dyn ApplyRotaryEmbedding>>,
        attention_mask: Option<&Vec<Tensor>>,
        positions: &Tensor,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &vendor::InputMetadata,
    ) -> Result<Tensor> {
        self.forward_ext(
            xs,
            rotary_emb,
            attention_mask,
            positions,
            cache,
            input_metadata,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_ext(
        &self,
        xs: &Tensor,
        rotary_emb: &Option<Arc<dyn ApplyRotaryEmbedding>>,
        attention_mask: Option<&Vec<Tensor>>,
        positions: &Tensor,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &vendor::InputMetadata,
        q_scale: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (seq_len, _) = xs.dims2()?;

        let (q_raw, k, v) = match &self.qkv_proj {
            QkvProjection::Separate { q_proj, k_proj, v_proj } => (
                q_proj.forward(xs)?,
                k_proj.forward(xs)?,
                v_proj.forward(xs)?,
            ),
            QkvProjection::Packed(qkv_proj) => {
                let qkv = qkv_proj.forward(xs)?;
                if qkv.len() != 3 {
                    crate::bail!(
                        "Expected 3 outputs from packed qkv projection, got {}",
                        qkv.len()
                    );
                }
                (qkv[0].clone(), qkv[1].clone(), qkv[2].clone())
            }
        };

        let local_q_dim = self.num_heads * self.head_dim;
        let (q_linear, gate) = if self.attn_output_gate {
            let q_dim = q_raw.dim(1)?;
            if q_dim != local_q_dim * 2 {
                crate::bail!(
                    "q_proj output dim mismatch for gated attention, expected {}, got {}",
                    local_q_dim * 2,
                    q_dim
                );
            }
            let q_gate = q_raw.reshape((seq_len, self.num_heads, self.head_dim * 2))?;
            let q = q_gate.narrow(2usize, 0, self.head_dim)?;
            let gate = q_gate.narrow(2usize, self.head_dim, self.head_dim)?;
            (
                q.reshape((seq_len, local_q_dim))?,
                Some(gate.reshape((seq_len, local_q_dim))?),
            )
        } else {
            (q_raw, None)
        };

        let q = q_linear.reshape((seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((seq_len, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((seq_len, self.num_kv_heads, self.head_dim))?;

        let (q, k) = self.prepare_qk_for_rope(q, k, seq_len)?;

        let (q, k) = if let Some(rotary_emb) = rotary_emb {
            match rotary_emb.apply_rotary_emb_qkv(&q, &k, positions)? {
                Some((q_new, k_new)) => (q_new, k_new),
                None => (q, k),
            }
        } else {
            (q, k)
        };

        let (q, k) = if self.qk_l2_norm {
            let q_rms = missing_shims::mean_keepdim_last(&q.sqr()?)?;
            let q_rms = missing_shims::broadcast_add_const(&q_rms, 1e-5)?.sqrt()?;
            let k_rms = missing_shims::mean_keepdim_last(&k.sqr()?)?;
            let k_rms = missing_shims::broadcast_add_const(&k_rms, 1e-5)?.sqrt()?;
            let q = q.broadcast_div(&q_rms)?;
            let k = k.broadcast_div(&k_rms)?;
            (q, k)
        } else {
            (q, k)
        };

        let (q, k) = if q.dtype() != self.dtype {
            (q.to_dtype(self.dtype)?, k.to_dtype(self.dtype)?)
        } else {
            (q, k)
        };

        let v = if v.dtype() != self.dtype { v.to_dtype(self.dtype)? } else { v };

        let v = if let Some(eps) = self.v_norm_eps {
            let orig_dtype = v.dtype();
            let v_f32 = v.to_dtype(DType::F32)?;
            let mean_sq = missing_shims::mean_keepdim_last(&v_f32.sqr()?)?;
            let rms = missing_shims::broadcast_add_const(&mean_sq, eps)?.sqrt()?;
            v_f32.broadcast_div(&rms)?.to_dtype(orig_dtype)?
        } else {
            v
        };

        let mut q = q;
        if let Some(rotary_emb) = rotary_emb {
            if let (Some(orig_max), Some(beta)) = (
                rotary_emb.get_original_max_position_embeddings(),
                rotary_emb.get_llama_4_scaling_beta(),
            ) {
                let scale =
                    missing_shims::llama4_attn_scale(positions, beta, orig_max as f64)?;
                let scale = scale.to_dtype(q.dtype())?;
                let scale = scale.squeeze(0usize)?.squeeze(0usize)?;
                let scale = scale.reshape((seq_len, 1, 1))?;
                q = q.broadcast_mul(&scale)?;
            }
        }

        if let Some(scale) = q_scale {
            let scale = scale.reshape((seq_len, 1, 1))?;
            let q_f32 = q.to_dtype(DType::F32)?;
            let scale_f32 = scale.to_dtype(DType::F32)?;
            q = q_f32.broadcast_mul(&scale_f32)?.to_dtype(q.dtype())?;
        }

        let y = self.attn.forward(
            &q,
            &k,
            &v,
            attention_mask,
            cache.map(|(k_, _)| k_.clone()),
            cache.map(|(_, v_)| v_.clone()),
            input_metadata,
            self.softcapping,
        )?;
        // FIX:原误植 (seq_len,) 会把 [seq, hidden] 压成 1 元素
        //(xinfer 原码无此 reshape;语义 = 展平到 hidden 维)
        let y = y.reshape((seq_len, self.num_heads * self.head_dim))?;

        let y = if let Some(gate) = gate {
            let gate = if gate.dtype() != y.dtype() {
                gate.to_dtype(y.dtype())?
            } else {
                gate
            };
            let g = super::ops::sigmoid(&gate)?;
            y.broadcast_mul(&g)?
        } else {
            y
        };

        let y = if self.is_qvar_builder { y } else { y.to_dtype(xs.dtype())? };
        self.o_proj.forward(&y)
    }

    /// Optimized single-token attention without KV cache or paged attention.
    pub fn forward_single_token_no_cache(
        &self,
        xs: &Tensor,
        rotary_emb: &Arc<dyn ApplyRotaryEmbedding>,
        positions: &Tensor,
    ) -> Result<Tensor> {
        let (seq_len, _) = xs.dims2()?;

        let (q_raw, k, v) = match &self.qkv_proj {
            QkvProjection::Separate { q_proj, k_proj, v_proj } => (
                q_proj.forward(xs)?,
                k_proj.forward(xs)?,
                v_proj.forward(xs)?,
            ),
            QkvProjection::Packed(qkv_proj) => {
                let qkv = qkv_proj.forward(xs)?;
                (qkv[0].clone(), qkv[1].clone(), qkv[2].clone())
            }
        };

        let local_q_dim = self.num_heads * self.head_dim;
        let (q_linear, gate) = if self.attn_output_gate {
            let q_gate = q_raw.reshape((seq_len, self.num_heads, self.head_dim * 2))?;
            let q = q_gate.narrow(2usize, 0, self.head_dim)?;
            let gate = q_gate.narrow(2usize, self.head_dim, self.head_dim)?;
            (
                q.reshape((seq_len, local_q_dim))?,
                Some(gate.reshape((seq_len, local_q_dim))?),
            )
        } else {
            (q_raw, None)
        };

        let q = q_linear.reshape((seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((seq_len, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((seq_len, self.num_kv_heads, self.head_dim))?;

        let (q, k) = self.prepare_qk_for_rope(q, k, seq_len)?;

        let (_q, _k) = match rotary_emb.apply_rotary_emb_qkv(&q, &k, positions)? {
            Some((q_new, k_new)) => (q_new, k_new),
            None => (q, k),
        };

        let n_rep = self.num_heads / self.num_kv_heads;
        let y = if n_rep > 1 {
            let v_e = v.unsqueeze(2usize)?;
            let v_e = missing_shims::expand(&v_e, &[seq_len, self.num_kv_heads, n_rep, self.head_dim])?;
            v_e.reshape((seq_len, self.num_heads * self.head_dim))?
        } else {
            v.reshape((seq_len, self.num_heads * self.head_dim))?
        };

        let y = if let Some(gate) = gate {
            let gate = if gate.dtype() != y.dtype() {
                gate.to_dtype(y.dtype())?
            } else {
                gate
            };
            let g = super::ops::sigmoid(&gate)?;
            y.broadcast_mul(&g)?
        } else {
            y
        };

        let y = if self.is_qvar_builder { y } else { y.to_dtype(xs.dtype())? };
        self.o_proj.forward(&y)
    }
}

pub struct NaiveAttention {
    q_proj: ReplicatedLinear,
    k_proj: ReplicatedLinear,
    v_proj: ReplicatedLinear,
    o_proj: ReplicatedLinear,
    scale: f64,
    num_heads: usize,
    head_dim: usize,
    softcapping: Option<f64>,
}

impl NaiveAttention {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vb: VarBuilderX,
        _num_heads: usize,
        hidden_size: usize,
        head_dim: usize,
        softcapping: Option<f64>,
        dtype: DType,
        key_mappings: HashMap<String, String>,
    ) -> Result<Self> {
        let is_qvar_builder = vb.is_qvar_builder();
        let pick = |fallback: &str| -> String {
            if is_qvar_builder {
                match fallback {
                    "q_proj" => "attn_q",
                    "k_proj" => "attn_k",
                    "v_proj" => "attn_v",
                    "o_proj" => "attn_output",
                    _ => fallback,
                }
                .to_string()
            } else {
                fallback.to_string()
            }
        };

        let q_proj = ReplicatedLinear::load_b(
            hidden_size,
            hidden_size,
            true,
            vb.pp(&pick("q_proj")),
            dtype,
        )?;
        let k_proj = ReplicatedLinear::load_b(
            hidden_size,
            hidden_size,
            true,
            vb.pp(&pick("k_proj")),
            dtype,
        )?;
        let v_proj = ReplicatedLinear::load_b(
            hidden_size,
            hidden_size,
            true,
            vb.pp(&pick("v_proj")),
            dtype,
        )?;
        let o_key = key_mappings
            .get("o_proj")
            .cloned()
            .unwrap_or_else(|| pick("o_proj"));
        let o_proj = ReplicatedLinear::load_b(
            hidden_size,
            hidden_size,
            true,
            vb.pp(&o_key),
            dtype,
        )?;

        let scale = (head_dim as f64).powf(-0.5);
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            scale,
            num_heads: _num_heads,
            head_dim,
            softcapping,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        emb: &Arc<dyn ApplyRotaryEmbedding>,
        positions: &Option<Tensor>,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, seq_len, _) = missing_shims::dims3(xs)?;
        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        let shape: Shape = (b, seq_len, self.num_heads, self.head_dim).into();
        let q_packed = q.reshape((seq_len, self.num_heads, self.head_dim))?.contiguous()?;
        let k_packed = k.reshape((seq_len, self.num_heads, self.head_dim))?.contiguous()?;
        let v = v.reshape(shape.dims())?.transpose(1usize, 2usize)?.contiguous()?;

        let (q_packed, k_packed) = if let Some(positions) = positions {
            match emb.apply_rotary_emb_qkv(&q_packed, &k_packed, positions)? {
                Some((q_new, k_new)) => (q_new, k_new),
                None => (q_packed, k_packed),
            }
        } else {
            (q_packed, k_packed)
        };

        let q = q_packed
            .reshape((b, seq_len, self.num_heads, self.head_dim))?
            .transpose(1usize, 2usize)?
            .contiguous()?;
        let k = k_packed
            .reshape((b, seq_len, self.num_heads, self.head_dim))?
            .transpose(1usize, 2usize)?
            .contiguous()?;

        let chunk_size = 1024;
        let mut attn_chunks = Vec::new();
        let num_chunks = (seq_len + chunk_size - 1) / chunk_size;
        for c in 0..num_chunks {
            let offset = c * chunk_size;
            let len = chunk_size.min(seq_len - offset);
            let q_chunk = q.narrow(2usize, offset, len)?.contiguous()?;
            let kt = k.t2()?;
            let mut att = q_chunk.matmul(&kt)?;
            att = att.affine(f64::from(self.scale), 0.0)?;

            if let Some(sc) = self.softcapping {
                att = att.affine(1.0 / sc, 0.0)?.tanh()?;
                att = att.affine(sc, 0.0)?;
            }
            if let Some(mask) = mask {
                let q_chunk_mask = mask.narrow(2usize, offset, len)?;
                att = att.broadcast_add(&q_chunk_mask)?;
            }
            let att_f32 = att.to_dtype(DType::F32)?;
            let att_n = super::ops::softmax_last_dim(&att_f32)?;
            let att_n = att_n.to_dtype(att.dtype())?;

            let att_chunk = att_n.matmul(&v)?;
            attn_chunks.push(att_chunk);
        }

        let att = missing_shims::cat(&attn_chunks, 2)?;
        let att = att.contiguous()?;
        let att = att.squeeze(0usize)?;
        let att = att.transpose(0usize, 1usize)?;
        let att = att.reshape((b, seq_len, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&att)
    }
}

// ============================================================================
// OwlTensor / vendor 新增面需求清单(待主 agent 合并进 mod.rs;本文件经
// missing_shims/pa_shim 自洽编译,合并后垫片可删):
// 1. Tensor::cat(&[Tensor], dim) -> Tensor —— 拼接(调用于 packed qkv / naive attn)
// 2. Tensor::flatten(from, to) -> Tensor —— 区间展平(prepare_qk_for_rope per-head 路径)
// 3. Tensor::mean_keepdim(D) -> Tensor —— 保维均值(qk_l2_norm / v_norm)
// 4. Tensor::broadcast_add_const(f64) -> Tensor —— 标量右加(l2_norm eps)
// 5. Tensor::expand(&[usize]) -> Tensor —— 广播复制(GQA expand)
// 6. get_llama4_attn_scale(positions, beta, orig_max) -> Tensor —— 位置缩放
// 7. Tensor::dims3() -> (usize,usize,usize) —— NaiveAttention forward
// 8. vendor::PagedAttention::new(...)/forward(...) 真实面(attention-rs kernel port)
// ============================================================================

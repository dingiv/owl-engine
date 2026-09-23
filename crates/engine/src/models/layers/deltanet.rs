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
use super::{collect_key_map, ctx_scope, erased, DType, OwlTensor, Result, Shard, Tensor, VarBuilderX};
use crate::config::Config;
use crate::hybrid::resolve_qwen3_hybrid_config;
use std::rc::Rc;

// ---- vendor gdn kernel 垫片(attention-rs port,T3 回填) ----
#[allow(dead_code)] // flashinfer 引入后接通(裁决 2026-09-22)
mod gdn_shim {
    use super::{ctx_scope, Result, Tensor};
    use crate::models::layers::OwlTensor;
    use owl_nn::kernels::gdn_kernels::GdnKernels;
    use std::sync::{Arc, Mutex, OnceLock};

    /// GDN 核族句柄(nvrtc 一次编译,进程级;rig 单卡语义)
    fn kernels() -> Result<Arc<Mutex<GdnKernels>>> {
        static K: OnceLock<Arc<Mutex<GdnKernels>>> = OnceLock::new();
        if let Some(k) = K.get() {
            return Ok(Arc::clone(k));
        }
        let dev = ctx_scope::with_device();
        let k = GdnKernels::new(dev.ctx())
            .map_err(|e| crate::Error::Msg(format!("gdn nvrtc: {e}")))?;
        let _ = K.set(Arc::new(Mutex::new(k)));
        Ok(Arc::clone(K.get().expect("gdn kernels 初始化")))
    }

    /// 因果卷积 prefill(变长;conv_state [nseq, d, 3] 已由调用方 gather,
    /// 核按序列序号就地滚动,写回由 forward 的 scatter 路负责)
    #[allow(clippy::too_many_arguments)]
    pub fn causal_conv1d_fwd(
        mixed_qkv: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        conv_state: &mut Tensor,
        snapshots: Option<&Tensor>,
        cu_seqlens: &Tensor,
        silu: bool,
    ) -> Result<Tensor> {
        if snapshots.is_some() {
            return Err(crate::Error::Schedule(
                "gdn conv1d_fwd: MTP conv 快照未接(MTP verify 域外,结构化报错)".into(),
            ));
        }
        if mixed_qkv.dtype() != owl_nn::Dtype::F32 {
            return Err(crate::Error::Msg(format!(
                "gdn conv1d_fwd: 仅 F32(mamba_ssm_dtype=f32),得到 {:?}",
                mixed_qkv.dtype()
            )));
        }
        let shape = mixed_qkv.shape().to_vec(); // [total, d]
        if shape.len() != 2 {
            return Err(crate::Error::Msg(format!(
                "gdn conv1d_fwd: 期望 [total, d],实际 {shape:?}"
            )));
        }
        let (total, d) = (shape[0], shape[1]);
        // cu_seqlens host 读取(急切 prefill 路径合法;捕获路径不走此处)
        let cu = super::vendor::read_u32_device(cu_seqlens)?;
        if cu.len() < 2 || cu.len() - 1 != conv_state.shape()[0] {
            return Err(crate::Error::Msg(format!(
                "gdn conv1d_fwd: cu_seqlens len {} 与 conv_state 行数 {} 不一致",
                cu.len(),
                conv_state.shape()[0]
            )));
        }
        if (cu[cu.len() - 1] as usize) != total {
            return Err(crate::Error::Msg(format!(
                "gdn conv1d_fwd: cu_seqlens 末值 {} != token 总数 {total}",
                cu[cu.len() - 1]
            )));
        }
        // 核期望 weight [d, 4] 连续(conv1d.weight [d,1,4] 同一内存布局,直传指针)
        let batch = (cu.len() - 1) as i32;
        let k = kernels()?;
        ctx_scope::with_dry(|ctx, _dry| {
            let out = ctx.scratch_tensor::<f32>(&shape)?;
            let mut kk = k.lock().map_err(|_| crate::Error::Msg("gdn kernels 中毒".into()))?;
            kk.conv1d_fwd_k4(
                ctx.stream(),
                "f32",
                mixed_qkv.device_ptr(),
                weight.device_ptr(),
                bias.map(|b| b.device_ptr() as *const u8).unwrap_or(std::ptr::null()),
                conv_state.device_ptr() as *mut f32,
                out.device_ptr() as *mut u8,
                cu_seqlens.device_ptr() as *const u32,
                batch,
                d as i32,
                silu,
            )
            .map_err(|e| crate::Error::Msg(format!("conv1d_fwd: {e}")))?;
            Ok(owl_nn::DynTensor::from_f32(&out))
        })
    }

    /// 因果卷积 decode(按 slot 更新状态)
    #[allow(clippy::too_many_arguments)]
    pub fn causal_conv1d_update_slots(
        mixed_qkv: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        conv_state: &mut Tensor,
        seq_slots: &Tensor,
        silu: bool,
    ) -> Result<Tensor> {
        if mixed_qkv.dtype() != owl_nn::Dtype::F32 {
            return Err(crate::Error::Msg(format!(
                "gdn conv1d: 仅 F32(mamba_ssm_dtype=f32),得到 {:?}",
                mixed_qkv.dtype()
            )));
        }
        let shape = mixed_qkv.shape().to_vec();
        let d = shape[1];
        let batch = shape[0];
        let k = kernels()?;
        ctx_scope::with_dry(|ctx, _dry| {
            let out = ctx.scratch_tensor::<f32>(&shape)?;
            let mut kk = k.lock().map_err(|_| crate::Error::Msg("gdn kernels 中毒".into()))?;
            kk.conv1d_update_slots_k4(
                ctx.stream(),
                "f32",
                mixed_qkv.device_ptr(),
                weight.device_ptr(),
                bias.map(|b| b.device_ptr() as *const u8).unwrap_or(std::ptr::null()),
                conv_state.device_ptr() as *mut f32,
                seq_slots.device_ptr() as *const u32,
                out.device_ptr() as *mut u8,
                batch as i32,
                d as i32,
                silu,
            )
            .map_err(|e| crate::Error::Msg(format!("conv1d_upd: {e}")))?;
            Ok(owl_nn::DynTensor::from_f32(&out))
        })
    }

    /// 融合 GDN 门控(a_log/dt_bias → g, beta)
    /// 融合 GDN 门控(a_log/dt_bias [H] F32;a/b [1,T,H])→ (g, beta) log 空间
    pub fn fused_gdn_gating(
        a_log: &Tensor,
        a: &Tensor,
        b: &Tensor,
        dt_bias: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let shape = a.shape().to_vec();
        let total: usize = shape.iter().product();
        let heads = a_log.shape()[0];
        let k = kernels()?;
        ctx_scope::with_dry(|ctx, _dry| {
            let g = ctx.scratch_tensor::<f32>(&[total])?;
            let beta = ctx.scratch_tensor::<f32>(&[total])?;
            let mut kk = k.lock().map_err(|_| crate::Error::Msg("gdn kernels 中毒".into()))?;
            kk.fused_gating(
                ctx.stream(),
                "f32",
                a_log.device_ptr() as *const f32,
                a.device_ptr(),
                b.device_ptr(),
                dt_bias.device_ptr() as *const f32,
                g.device_ptr() as *mut f32,
                beta.device_ptr() as *mut f32,
                total as i32,
                heads as i32,
            )
            .map_err(|e| crate::Error::Msg(format!("gating: {e}")))?;
            let g = owl_nn::DynTensor::from_f32(&g).reshape(&shape[..])?;
            let beta = owl_nn::DynTensor::from_f32(&beta).reshape(&shape[..])?;
            Ok((g, beta))
        })
    }

    /// 末维 L2 归一
    pub fn l2_norm_last_dim(t: &Tensor, eps: f64) -> Result<Tensor> {
        let shape = t.shape().to_vec();
        let last = *shape.last().expect("l2norm rank>=1");
        let rows: usize = shape.iter().product::<usize>() / last;
        let k = kernels()?;
        ctx_scope::with_dry(|ctx, _dry| {
            let out = ctx.scratch_tensor::<f32>(&shape)?;
            let mut kk = k.lock().map_err(|_| crate::Error::Msg("gdn kernels 中毒".into()))?;
            kk.l2_norm(
                ctx.stream(),
                "f32",
                t.device_ptr(),
                out.device_ptr() as *mut u8,
                rows as i32,
                last as i32,
                eps as f32,
            )
            .map_err(|e| crate::Error::Msg(format!("l2norm: {e}")))?;
            Ok(owl_nn::DynTensor::from_f32(&out))
        })
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

    /// prefill 变长 GQA 递推(主路;g log 空间,q 未缩放 → 核内约定在垫片统一)
    #[allow(clippy::too_many_arguments)]
    pub fn gated_delta_rule_recurrence_varlen_gqa(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        g: &Tensor,
        beta: &Tensor,
        state: &mut Tensor,
        seq_slots: &Tensor,
        cu_seqlens: &Tensor,
        scale: f32,
        snapshots: Option<&Tensor>,
    ) -> Result<Tensor> {
        recurrence_varlen_impl(
            q, k, v, g, beta, state, seq_slots, cu_seqlens, snapshots, scale,
        )
    }

    /// prefill 变长 MHA 递推(k==v 头数;q 已由调用方缩放)
    #[allow(clippy::too_many_arguments)]
    pub fn gated_delta_rule_recurrence_varlen(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        g: &Tensor,
        beta: &Tensor,
        state: &mut Tensor,
        seq_slots: &Tensor,
        cu_seqlens: &Tensor,
        snapshots: Option<&Tensor>,
    ) -> Result<Tensor> {
        recurrence_varlen_impl(
            q, k, v, g, beta, state, seq_slots, cu_seqlens, snapshots, 1.0,
        )
    }

    /// decode 按 slot GQA 递推
    #[allow(clippy::too_many_arguments)]
    pub fn gated_delta_rule_decode_slots_gqa(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        g: &Tensor,
        beta: &Tensor,
        state: &mut Tensor,
        seq_slots: &Tensor,
        scale: f32,
    ) -> Result<Tensor> {
        let bs = q.shape()[0];
        let nk = q.shape()[1];
        let kd = q.shape()[2];
        let nv = v.shape()[1];
        let vd = v.shape()[2];
        let kn = kernels()?;
        ctx_scope::with_dry(|ctx, _dry| {
            let out = ctx.scratch_tensor::<f32>(&[bs, nv, vd])?;
            let mut kk = kn.lock().map_err(|_| crate::Error::Msg("gdn kernels 中毒".into()))?;
            kk.delta_decode_slots_gqa(
                ctx.stream(),
                "f32",
                q.device_ptr(),
                k.device_ptr(),
                v.device_ptr(),
                g.device_ptr() as *const f32,
                beta.device_ptr() as *const f32,
                state.device_ptr() as *mut f32,
                seq_slots.device_ptr() as *const u32,
                out.device_ptr() as *mut u8,
                bs as i32,
                nv as i32,
                nk as i32,
                kd as i32,
                vd as i32,
                scale,
            )
            .map_err(|e| crate::Error::Msg(format!("delta_dec_gqa: {e}")))?;
            Ok(owl_nn::DynTensor::from_f32(&out))
        })
    }

    /// 门控 RMSNorm × act(z)(act: 0=silu 1=sigmoid;weight [head_v_dim] F32 per-group)
    #[allow(clippy::too_many_arguments)]
    fn gated_rmsnorm_act(
        output: &Tensor,
        z: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        eps: f64,
        head_v_dim: usize,
        act: i32,
    ) -> Result<Tensor> {
        let shape = output.shape().to_vec();
        let rows = shape[0];
        let value_dim = shape[1];
        let per_group = weight.shape()[0] == head_v_dim;
        let kn = kernels()?;
        ctx_scope::with_dry(|ctx, _dry| {
            let out = ctx.scratch_tensor::<f32>(&shape)?;
            let mut kk = kn.lock().map_err(|_| crate::Error::Msg("gdn kernels 中毒".into()))?;
            kk.rmsnorm_act(
                ctx.stream(),
                "f32",
                output.device_ptr(),
                z.device_ptr(),
                weight.device_ptr() as *const f32,
                bias.map(|b| b.device_ptr() as *const f32).unwrap_or(std::ptr::null()),
                out.device_ptr() as *mut u8,
                rows as i32,
                value_dim as i32,
                head_v_dim as i32,
                eps as f32,
                per_group,
                bias.is_some(),
                act,
            )
            .map_err(|e| crate::Error::Msg(format!("rmsnorm_act: {e}")))?;
            Ok(owl_nn::DynTensor::from_f32(&out))
        })
    }

    /// 门控 RMSNorm × sigmoid(z)
    #[allow(clippy::too_many_arguments)]
    pub fn gated_rmsnorm_sigmoid_mul(
        output: &Tensor,
        z: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        eps: f64,
        head_v_dim: usize,
    ) -> Result<Tensor> {
        gated_rmsnorm_act(output, z, weight, bias, eps, head_v_dim, 1)
    }

    /// prefill 变长递推统一实现(09-23 深夜改道定谳):
    /// 原 gdn_delta_rec_fb 直通在 bh>1 时 out 写缺失(状态写正确;owl-nn 单独
    /// 对拍却绿——多块并发下 out 路径行为与状态路径不一致,另案查 .cu)。
    /// 现役实现 = delta_decode_slots_gqa 逐 token 推进:语义 = delta rule
fn recurrence_varlen_impl(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        g: &Tensor,
        beta: &Tensor,
        state: &mut Tensor,
        seq_slots: &Tensor,
        cu_seqlens: &Tensor,
        snapshots: Option<&Tensor>,
        q_scale: f32,
    ) -> Result<Tensor> {
        if snapshots.is_some() {
            return Err(crate::Error::Schedule(
                "gdn recurrence: MTP recurrent 快照未接(MTP verify 域外,结构化报错)".into(),
            ));
        }
        if q.dtype() != owl_nn::Dtype::F32 {
            return Err(crate::Error::Msg(format!(
                "gdn recurrence: 仅 F32(mamba_ssm_dtype=f32),得到 {:?}",
                q.dtype()
            )));
        }
        let q_shape = q.shape().to_vec(); // [T, nk, kd]
        if q_shape.len() != 3 {
            return Err(crate::Error::Msg(format!(
                "gdn recurrence: 期望 q [T, nk, kd],实际 {q_shape:?}"
            )));
        }
        let v_shape = v.shape().to_vec(); // [T, nv, vd]
        if v_shape.len() != 3 || v_shape[0] != q_shape[0] {
            return Err(crate::Error::Msg(format!(
                "gdn recurrence: q {q_shape:?} 与 v {v_shape:?} token 数不一致"
            )));
        }
        let (total, nk, kd) = (q_shape[0], q_shape[1], q_shape[2]);
        let (nv, vd) = (v_shape[1], v_shape[2]);
        if nv % nk != 0 {
            return Err(crate::Error::Msg(format!(
                "gdn recurrence: nv={nv} 须被 nk={nk} 整除(GQA 组映射)"
            )));
        }
        // host 读取:cu_seqlens [n+1] + 槽位 [n](急切 prefill 合法;捕获不走此处)
        let cu = super::vendor::read_u32_device(cu_seqlens)?;
        let slots = super::vendor::read_u32_device(seq_slots)?;
        if cu.len() < 2 || cu.len() - 1 != slots.len() {
            return Err(crate::Error::Msg(format!(
                "gdn recurrence: cu_seqlens len {} 与槽位数 {} 不一致",
                cu.len(),
                slots.len()
            )));
        }
        if (cu[cu.len() - 1] as usize) != total {
            return Err(crate::Error::Msg(format!(
                "gdn recurrence: cu_seqlens 末值 {} != token 总数 {total}",
                cu[cu.len() - 1]
            )));
        }
        if state.shape()[0] as u64 <= *slots.iter().max().unwrap_or(&0) as u64 {
            return Err(crate::Error::Msg(format!(
                "gdn recurrence: 槽位越界 max={} state 行数 {}",
                slots.iter().max().unwrap_or(&0),
                state.shape()[0]
            )));
        }
        let kn = kernels()?;
        // 生命周期定谳(2026-09-23):旧实现每 token 租一块 scratch、把视图
        // push 进 pieces —— 租约随 with_dry 出口归还,后续分配覆盖悬空视图
        // (t≥1 输出精确 0)。改为整段一块 slab [total, nv, vd],租约覆盖
        // 全循环;t 连续覆盖 0..total,slab 即最终答案,免 cat。
        let slab = ctx_scope::with_dry(|ctx, _dry| {
            ctx.scratch_tensor::<f32>(&[total, nv, vd])
                .map_err(|e| crate::Error::Msg(format!("recurrence slab: {e:?}")))
        })?;
        ctx_scope::with_dry(|ctx, _dry| {
            let mut kk =
                kn.lock().map_err(|_| crate::Error::Msg("gdn kernels 中毒".into()))?;
            for (i, &slot) in slots.iter().enumerate() {
                let (s, e) = (cu[i] as usize, cu[i + 1] as usize);
                if e <= s {
                    continue; // 空序列跳过
                }
                if slot == u32::MAX {
                    return Err(crate::Error::Schedule(
                        "gdn recurrence: prefill 序列命中无效槽位哨兵(prefill 不应出现)".into(),
                    ));
                }
                // 槽位张量视图([1] U32):decode 核按 slots[0] 寻址常驻状态行
                let slot_view = seq_slots.narrow_dim0(i, 1)?;
                for t in s..e {
                    let q_t = q.narrow_dim0(t, 1)?; // [1, nk, kd] 视图
                    let k_t = k.narrow_dim0(t, 1)?;
                    let v_t = v.narrow_dim0(t, 1)?; // [1, nv, vd]
                    let g_t = g.narrow_dim0(t, 1)?; // [1, nv](log 空间,核内 exp)
                    let beta_t = beta.narrow_dim0(t, 1)?;
                    // 写入 slab 第 t 行(字节偏移;slab 租约覆盖至循环结束)
                    let dst = unsafe {
                        (slab.device_ptr() as *mut f32).add(t * nv * vd) as *mut u8
                    };
                    kk.delta_decode_slots_gqa(
                        ctx.stream(),
                        "f32",
                        q_t.device_ptr(),
                        k_t.device_ptr(),
                        v_t.device_ptr(),
                        g_t.device_ptr() as *const f32,
                        beta_t.device_ptr() as *const f32,
                        state.device_ptr() as *mut f32,
                        slot_view.device_ptr() as *const u32,
                        dst,
                        1,
                        nv as i32,
                        nk as i32,
                        kd as i32,
                        vd as i32,
                        q_scale,
                    )
                    .map_err(|e| crate::Error::Msg(format!("delta_dec(prefill): {e}")))?;
                }
            }
            Ok(())
        })?;
        // slab 本身即最终输出([total, nv, vd];行序 = token 全局序)
        let out = owl_nn::DynTensor::from_f32(&slab);
        OwlTensor::reshape(&out, (total, nv, vd))
    }

    /// 门控 RMSNorm × silu(z)
    #[allow(clippy::too_many_arguments)]
    pub fn gated_rmsnorm_silu_mul(
        output: &Tensor,
        z: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        eps: f64,
        head_v_dim: usize,
    ) -> Result<Tensor> {
        gated_rmsnorm_act(output, z, weight, bias, eps, head_v_dim, 0)
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
    pub fn get_batch_conv_state(&mut self, layer: usize, slots: &Tensor) -> Result<Tensor> {
        self.cache.gather_conv_rows(layer, slots)
    }
    pub fn set_batch_conv_state(&mut self, layer: usize, slots: &Tensor, s: &Tensor) -> Result<()> {
        self.cache.scatter_conv_rows(layer, slots, s)
    }
    pub fn conv_state_mut(&mut self, layer: usize) -> &mut Tensor {
        self.cache.conv_state_mut(layer)
    }
    pub fn conv_state(&self, layer: usize) -> &Tensor {
        self.cache.conv_state(layer)
    }
    pub fn recurrent_state_mut(&mut self, layer: usize) -> &mut Tensor {
        self.cache.recurrent_state_mut(layer)
    }
    pub fn set_batch_recurrent_state(&mut self, layer: usize, slots: &Tensor, s: &Tensor) -> Result<()> {
        self.cache.scatter_rec_rows(layer, slots, s)
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

        // 卷积输出切回 q'/k'/v'(列切片走 transpose 路径,规避 narrow 中间维缺陷,见 slice_cols)
        let q_conv = slice_cols(&kv_conv, 0, self.key_dim)?;
        let k_conv = slice_cols(&kv_conv, self.key_dim, self.key_dim)?;
        let v_conv = slice_cols(&kv_conv, self.key_dim * 2, self.value_dim)?;

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
    ctx_scope::with(|_ops, ctx| Ok(erased::cat(ctx, ts, dim)?))
}

/// 2D 列切片 [rows, cols] → [rows, len](transpose ×2 + narrow_dim0;
/// 规避 OwlTensor::narrow 中间维 dry 调用丢 after 因子的缺陷,见汇报)
fn slice_cols(t: &Tensor, start: usize, len: usize) -> Result<Tensor> {
    let tr = t.transpose(0usize, 1usize)?; // [cols, rows] 物化
    let piece = tr.narrow_dim0(start, len)?; // [len, rows] 视图
    piece.transpose(0usize, 1usize) // [rows, len] 物化
}

/// zeros 构造(P 阶段 ctor 工厂;设备 = rig 单卡)
fn ctor_zeros(_shape: &[usize], _dtype: DType) -> Result<Tensor> {
    super::ctor::zeros(_shape, _dtype, &ctx_scope::with_device())
}

/// TP 切分(= xinfer tensor_parallel_chunk;world_size=1 直通,TP>1 = T3 后续)
pub(crate) fn tensor_parallel_chunk(
    t: &Tensor,
    _dim: usize,
    _rank: usize,
    world_size: usize,
    name: &str,
) -> Result<Tensor> {
    if world_size <= 1 {
        return Ok(t.clone());
    }
    let _ = name;
    unimplemented!("TP>1 分片装载(单卡面已回退;TP = A2.6 另案)")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::dry_kernels::DryKernels;
    use owl_nn::cublas::NnBlas;
    use owl_nn::TensorPoolOps;
    use std::sync::Arc;

    type Dev = owl_cuda::CudaDevice;
    type Pool = owl_cuda::CudaPool;

    /// 进程级测试 rig(与 dry_run 同构;OnceLock 幂等,多测试共享互斥安全)
    fn rig() -> (Dev, Arc<Pool>, Arc<Pool>) {
        static R: std::sync::OnceLock<(Dev, Arc<Pool>, Arc<Pool>)> = std::sync::OnceLock::new();
        R.get_or_init(|| {
            use owl_iface::{Device as _, PoolConfig, PoolKind};
            let dev = Dev::new(owl_cuda::test_device_ordinal(), owl_cuda::TEST_POOL_BYTES).expect("需要 CUDA 设备");
            let scratch = Arc::new(
                dev.create_pool(PoolConfig {
                    name: format!("gdn-e2e-scratch-{}", std::process::id()),
                    kind: PoolKind::Scratch,
                    bytes: 64 << 20,
                })
                .unwrap(),
            );
            let wpool = Arc::new(
                dev.create_pool(PoolConfig {
                    name: format!("gdn-e2e-weights-{}", std::process::id()),
                    kind: PoolKind::Weights,
                    bytes: 128 << 20,
                })
                .unwrap(),
            );
            let ops = owl_nn::OpsCtx::new(&dev).unwrap();
            let blas = NnBlas::new(&dev).unwrap();
            let dry = DryKernels::new(dev.ctx()).unwrap();
            ctx_scope::install(ops, blas, dry, scratch.clone(), wpool.clone(), &dev);
            (dev, scratch, wpool)
        })
        .clone()
    }

    fn htod(wpool: &Pool, shape: &[usize], v: Vec<f32>) -> Tensor {
        let t = wpool.from_vec_tensor(shape, v).unwrap();
        owl_nn::DynTensor::from_f32(&t)
    }

    fn htod_u32(wpool: &Pool, shape: &[usize], v: Vec<u32>) -> Tensor {
        let t = wpool.from_vec_tensor(shape, v).unwrap();
        owl_nn::DynTensor::from_u32(&t)
    }

    fn dtoh_f32(dev: &Dev, t: &Tensor) -> Vec<f32> {
        let n = t.shape().iter().product::<usize>();
        let mut out = vec![0f32; n];
        // P0-3:流序 D2H(memx)
        dev.memcpy_dtoh_f32(dev.stream(), t.device_ptr() as *const f32, &mut out)
            .unwrap();
        out
    }

    /// 微形回归:1 序列/1 token/零态/4 头多 bh + 状态写回,手算可验证
    #[test]
    fn recurrence_micro_sanity() {
        let (_dev, _scratch, wpool) = rig();
        // 固定微形:4 头/1 token/零态/decay=1/beta=1/k=e_0 → y = v(头可辨)
        let (nk, nv, kd, vd) = (4usize, 4usize, 2usize, 2usize);
        let scale = 1.0f32;
        let total = 1usize;
        let q = vec![1.0f32; total * nk * kd];
        let k: Vec<f32> = (0..total * nk * kd).map(|i| if i % kd == 0 { 1.0 } else { 0.0 }).collect();
        let v: Vec<f32> = (0..total * nv * vd)
            .map(|i| (i / vd) as f32 * vd as f32 + (i % vd) as f32 + 1.0)
            .collect();
        let g_log = vec![0.0f32; total * nv];
        let beta = vec![1.0f32; total * nv];
        let state = vec![0.0f32; nv * kd * vd];
        let d_q = htod(&wpool, &[total, nk, kd], q);
        let d_k = htod(&wpool, &[total, nk, kd], k);
        let d_v = htod(&wpool, &[total, nv, vd], v);
        let d_g = htod(&wpool, &[total, nv], g_log);
        let d_beta = htod(&wpool, &[total, nv], beta);
        let mut d_state = htod(&wpool, &[1, nv, kd, vd], state);
        let d_slots = htod_u32(&wpool, &[1], vec![0]);
        let d_cu = htod_u32(&wpool, &[2], vec![0, total as u32]);
        let out = gdn_shim::gated_delta_rule_recurrence_varlen_gqa(
            &d_q, &d_k, &d_v, &d_g, &d_beta, &mut d_state, &d_slots, &d_cu, scale, None,
        )
        .unwrap();
        let got = dtoh_f32(&_dev, &out);
        let st = dtoh_f32(&_dev, &d_state);
        // 期望:每头 y = v[bh](零态 + k=e_0 + beta=1 → s[0] = v,y = s·q = v)
        assert_eq!(got, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    /// 递推垫片对拍:2 序列(3+2)/GQA(nk=2,nv=4)/带初态,fb 公式直译。
    /// g 以 log 空间入参(垫片负责 exp);q 缩放在垫片。
    #[test]
    fn recurrence_varlen_gqa_shim_host_parity() {
        let (dev, _scratch, wpool) = rig();
        let (nk, nv, kd, vd) = (2usize, 4usize, 4usize, 8usize);
        let gs = nv / nk;
        let lens = [3usize, 2usize];
        let total: usize = lens.iter().sum();
        let scale = 1.0f32 / (kd as f32).sqrt();

        let q: Vec<f32> = (0..total * nk * kd).map(|i| ((i % 5) as f32 - 2.0) * 0.3).collect();
        let kv: Vec<f32> = (0..total * nk * kd).map(|i| ((i % 7) as f32 - 3.0) * 0.2).collect();
        let v: Vec<f32> = (0..total * nv * vd).map(|i| ((i % 11) as f32 - 5.0) * 0.25).collect();
        // log 空间负值(exp 后 <1 = 衰减)
        let g_log: Vec<f32> = (0..total * nv).map(|i| -0.05 - 0.02 * (i % 3) as f32).collect();
        let beta: Vec<f32> = (0..total * nv).map(|i| 0.5 + 0.1 * (i % 2) as f32).collect();
        // 初态非零且逐槽可辨(slot1 与 slot0 不同;row2 = 未触碰守卫行)
        let state: Vec<f32> = (0..3 * nv * kd * vd)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.1)
            .collect();

        let d_q = htod(&wpool, &[total, nk, kd], q.clone());
        let d_k = htod(&wpool, &[total, nk, kd], kv.clone());
        let d_v = htod(&wpool, &[total, nv, vd], v.clone());
        let d_g = htod(&wpool, &[total, nv], g_log.clone());
        let d_beta = htod(&wpool, &[total, nv], beta.clone());
        let mut d_state = htod(&wpool, &[3, nv, kd, vd], state.clone());
        let d_slots = htod_u32(&wpool, &[2], vec![1, 0]);
        let mut cu = vec![0u32];
        for l in lens {
            cu.push(cu.last().unwrap() + l as u32);
        }
        let d_cu = htod_u32(&wpool, &[cu.len() as u32 as usize], cu.clone());

        let out = gdn_shim::gated_delta_rule_recurrence_varlen_gqa(
            &d_q, &d_k, &d_v, &d_g, &d_beta, &mut d_state, &d_slots, &d_cu, scale, None,
        )
        .unwrap();
        assert_eq!(out.shape(), &[total, nv, vd]);

        // ---- host 参考(fb 公式直译 + GQA 头映射 + 状态行按 slot)----
        let g_real: Vec<f32> = g_log.iter().map(|x| x.exp()).collect();
        let got = dtoh_f32(&dev, &out);
        let got_state = dtoh_f32(&dev, &d_state);
        let mut tok = 0usize;
        for (seq, &slot) in [1usize, 0].iter().enumerate() {
            let l = lens[seq];
            let mut s = state[slot * nv * kd * vd..(slot + 1) * nv * kd * vd].to_vec();
            for t in 0..l {
                for h in 0..nv {
                    let qh = h / gs; // GQA 头映射
                    let decay = g_real[(tok + t) * nv + h];
                    for j in 0..kd {
                        for vi in 0..vd {
                            s[(h * kd + j) * vd + vi] *= decay;
                        }
                    }
                    let base = (tok + t) * nv * vd + h * vd;
                    let kbase = (tok + t) * nk * kd + qh * kd;
                    for vi in 0..vd {
                        let mut kv_mem = 0.0f32;
                        for j in 0..kd {
                            kv_mem += s[(h * kd + j) * vd + vi] * kv[kbase + j];
                        }
                        let delta = (v[base + vi] - kv_mem) * beta[(tok + t) * nv + h];
                        for j in 0..kd {
                            s[(h * kd + j) * vd + vi] += kv[kbase + j] * delta;
                        }
                    }
                    for vi in 0..vd {
                        let mut y = 0.0f32;
                        for j in 0..kd {
                            y += s[(h * kd + j) * vd + vi] * q[kbase + j] * scale;
                        }
                        let go = got[(tok + t) * nv * vd + h * vd + vi];
                        assert!(
                            (go - y).abs() < 2e-4,
                            "OUT seq={seq} t={t} h={h} vi={vi} got={go} want={y}"
                        );
                    }
                }
            }
            for (i, (a, b)) in got_state[slot * nv * kd * vd..(slot + 1) * nv * kd * vd]
                .iter()
                .zip(&s)
                .enumerate()
            {
                assert!(
                    (a - b).abs() < 1e-4,
                    "STATE seq={seq} idx={i} got={a} want={b}"
                );
            }
            tok += l;
        }
        // 未触碰槽位(row2)状态不变
        assert_eq!(
            got_state[2 * nv * kd * vd..3 * nv * kd * vd],
            state[2 * nv * kd * vd..3 * nv * kd * vd],
            "未触碰槽位状态被改动"
        );
    }

    /// shim 链路 e2e:conv1d_fwd → gating → l2norm → delta 递推 → 门控 rmsnorm,
    /// 小形状(nk=2/nv=4/kd=8/vd=8,变长 2 序列)自建状态张量。
    /// 注:经 MambaCache::preallocate 的全层 forward 因 ctor::zeros 桩(mod.rs,
    /// B3.2 遗留)暂不可行,已上报;本测试覆盖同一算子链与状态连续性。
    #[test]
    fn deltanet_shim_chain_e2e_selfcheck() {
        let (dev, _scratch, wpool) = rig();
        let (nk, nv, kd, vd) = (2usize, 4usize, 8usize, 8usize);
        let key_dim = nk * kd; // 16
        let value_dim = nv * vd; // 32
        let conv_dim = key_dim * 2 + value_dim; // 64
        let hidden = 32usize;
        let lens = [3usize, 2usize];
        let total: usize = lens.iter().sum();

        // ---- 输入与权重(逐张量可辨)----
        let xs: Vec<f32> = (0..total * conv_dim).map(|i| ((i % 13) as f32 - 6.0) * 0.2).collect();
        let cw: Vec<f32> = (0..conv_dim * 4).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
        let cb: Vec<f32> = (0..conv_dim).map(|i| 0.01 * i as f32).collect();
        let a_log: Vec<f32> = (0..nv).map(|i| -0.3 - 0.1 * i as f32).collect();
        let dtb: Vec<f32> = (0..nv).map(|i| 0.2 * i as f32).collect();
        let nrm: Vec<f32> = (0..vd).map(|i| 0.5 + 0.05 * i as f32).collect();

        let d_x = htod(&wpool, &[total, conv_dim], xs);
        let d_w = htod(&wpool, &[conv_dim, 1, 4], cw);
        let d_b = htod(&wpool, &[conv_dim], cb);
        let d_a_log = htod(&wpool, &[nv], a_log);
        let d_dtb = htod(&wpool, &[nv], dtb);
        let d_nrm = htod(&wpool, &[vd], nrm);
        // conv_state 收集副本 [2, conv_dim, 3];递推状态 [3, nv, kd, vd](row2 守卫)
        let mut d_conv_state = htod(&wpool, &[2, conv_dim, 3], vec![0f32; 2 * conv_dim * 3]);
        let st0: Vec<f32> = (0..3 * nv * kd * vd).map(|i| ((i % 23) as f32 - 11.0) * 0.1).collect();
        let mut d_rec = htod(&wpool, &[3, nv, kd, vd], st0.clone());
        let mut cu = vec![0u32];
        for l in lens {
            cu.push(cu.last().unwrap() + l as u32);
        }
        let d_cu = htod_u32(&wpool, &[cu.len()], cu.clone());
        let d_slots = htod_u32(&wpool, &[2], vec![1, 0]);

        // ① conv1d_fwd(prefill 变长,silu)
        let kv_conv = gdn_shim::causal_conv1d_fwd(
            &d_x, &d_w, Some(&d_b), &mut d_conv_state, None, &d_cu, true,
        )
        .expect("conv1d_fwd");
        assert_eq!(kv_conv.shape(), &[total, conv_dim]);

        // ② 列切片 q'/k'/v'(走 slice_cols,同 forward)
        let q_c = slice_cols(&kv_conv, 0, key_dim).unwrap();
        let v_c = slice_cols(&kv_conv, key_dim * 2, value_dim).unwrap();

        // ③ gating(a/b 展开 [T,nv];此处以常数 a/b 走真核)
        let a_e = htod(&wpool, &[total, nv], vec![0.3f32; total * nv]);
        let b_e = htod(&wpool, &[total, nv], vec![0.6f32; total * nv]);
        let (g, beta) = gdn_shim::fused_gdn_gating(&d_a_log, &a_e, &b_e, &d_dtb).unwrap();

        // ④ l2norm(q/k 末维)
        let q_r = q_c.reshape((total, nk, kd)).unwrap();
        let q_n = gdn_shim::l2_norm_last_dim(&q_r, 1e-6).unwrap();

        // ⑤ delta 递推(decode 逐 token;g log 空间入参)
        let v_r = v_c.reshape((total, nv, vd)).unwrap();
        let g2 = g.squeeze(0usize).unwrap();
        let beta2 = beta.squeeze(0usize).unwrap();
        let out = gdn_shim::gated_delta_rule_recurrence_varlen_gqa(
            &q_n, &q_n, &v_r, &g2, &beta2, &mut d_rec, &d_slots, &d_cu,
            (1.0 / (kd as f32).sqrt()) as f32, None,
        )
        .expect("recurrence");
        assert_eq!(out.shape(), &[total, nv, vd]);
        let host_out = dtoh_f32(&dev, &out);
        assert!(host_out.iter().all(|x| x.is_finite()));

        // ⑥ 门控 rmsnorm(z = conv 输出的 v 段走 silu 路径)
        let z = slice_cols(&kv_conv, key_dim * 2, value_dim).unwrap();
        let flat = OwlTensor::reshape(&out, (total, value_dim)).unwrap();
        let gated = gdn_shim::gated_rmsnorm_silu_mul(
            &flat, &z, &d_nrm, None, 1e-5, vd,
        )
        .expect("rmsnorm");
        let host_g = dtoh_f32(&dev, &gated);
        assert!(host_g.iter().all(|x| x.is_finite()));

        // ⑦ 确定性:全链重跑(状态重置)位级一致
        let mut d_conv_state2 = htod(&wpool, &[2, conv_dim, 3], vec![0f32; 2 * conv_dim * 3]);
        let mut d_rec2 = htod(&wpool, &[3, nv, kd, vd], st0.clone());
        let kv2 = gdn_shim::causal_conv1d_fwd(
            &d_x, &d_w, Some(&d_b), &mut d_conv_state2, None, &d_cu, true,
        )
        .unwrap();
        let q_c2 = slice_cols(&kv2, 0, key_dim).unwrap();
        let v_c2 = slice_cols(&kv2, key_dim * 2, value_dim).unwrap();
        let (g3, beta3) = gdn_shim::fused_gdn_gating(&d_a_log, &a_e, &b_e, &d_dtb).unwrap();
        let q_n2 = gdn_shim::l2_norm_last_dim(&q_c2.reshape((total, nk, kd)).unwrap(), 1e-6).unwrap();
        let out2 = gdn_shim::gated_delta_rule_recurrence_varlen_gqa(
            &q_n2, &q_n2, &v_c2.reshape((total, nv, vd)).unwrap(), &g3.squeeze(0usize).unwrap(),
            &beta3.squeeze(0usize).unwrap(), &mut d_rec2, &d_slots, &d_cu,
            (1.0 / (kd as f32).sqrt()) as f32, None,
        )
        .unwrap();
        let host_out2 = dtoh_f32(&dev, &out2);
        assert_eq!(host_out, host_out2, "全链两次结果不一致");
        let st1 = dtoh_f32(&dev, &d_rec);
        let st2 = dtoh_f32(&dev, &d_rec2);
        assert_eq!(st1, st2, "递推状态两次不一致");
        // 状态确实被写(非全零)
        assert!(st1.iter().any(|x| x.abs() > 1e-6));
        // 未触碰槽位 row2 不变
        assert_eq!(
            st1[2 * nv * kd * vd..3 * nv * kd * vd],
            st0[2 * nv * kd * vd..3 * nv * kd * vd]
        );
    }

}

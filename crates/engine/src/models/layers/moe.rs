//! MoE 层(= xinfer layers/moe.rs 公共面 + 分派骨架直译;4426 行源)。
//!
//! 翻译注记:
//! - 权重装载/重排/gguf repack 管线(~3.5k 行)不搬:owl marlin-ffi
//!   (W4A16/W4A8/moe_wna16,已验收)+ loader/gguf 反解各归其位
//!   (REQ-DESIGN:参考实现单一来源);
//! - `MarlinMoeBackend/MarlinMoeOp`(CustomOp3)= marlin-ffi moe_wna16
//!   已验收件的调用面,本文件保 pick_moe_block_size/build 槛查分派骨架;
//! - FusedMoeGGUF / FusedMoeISQ:类型面保留,体 = GGUF 反解路径已由
//!   loader/gguf.rs 吸收,ISQ = 量化运行里程碑;
//! - vendor attention_rs::moe(P3 port 的 moe_marlin)归宿 = 已验收,
//!   调用面在 marlin-ffi crate,不进 vendor 占位。

use super::distributed::Comm;
use super::linear::{linear_no_bias, LinearX};
use super::{DType, Module, OwlTensor, Result, Shard, Tensor, VarBuilderX};
use crate::config::{Config, QuantConfig, MoEConfig};
use std::rc::Rc;
use std::sync::atomic::AtomicBool;

// ---- OwlTensor 缺面垫片(topk/sort 等;需求清单见尾注) ----
#[allow(dead_code)] // 需求垫片:topk/scatter_add/index_select 随 T3 消费
mod moe_shims {
    use super::{Result, Tensor};

    /// candle `Tensor::topk(dim, k, sorted)` → (values, indices)
    pub fn topk(_t: &Tensor, _k: usize, _dim: usize, _sorted: bool) -> Result<(Tensor, Tensor)> {
        unimplemented!("T3: Tensor::topk(routing kernel;需求已上报)")
    }
    /// candle `Tensor::cat`(同 attention 缺面)
    pub fn cat(_ts: &[Tensor], _dim: usize) -> Result<Tensor> {
        unimplemented!("T3: Tensor::cat(moe_shims)")
    }
    /// candle `Tensor::scatter_add(dim, idx, src)`
    pub fn scatter_add(_t: &Tensor, _dim: usize, _idx: &Tensor, _src: &Tensor) -> Result<Tensor> {
        unimplemented!("T3: Tensor::scatter_add(moe_shims)")
    }
    /// candle `Tensor::index_select(dim, idx)`
    pub fn index_select(_t: &Tensor, _dim: usize, _idx: &Tensor) -> Result<Tensor> {
        unimplemented!("T3: Tensor::index_select(moe_shims)")
    }
    /// marlin moe 对齐元数据构建(marlin-ffi moe_wna16 已验收件调用点)
    #[allow(clippy::too_many_arguments)]
    pub fn marlin_moe_gemm(
        _x: &Tensor,
        _w: &Tensor,
        _scales: &Tensor,
        _topk_weights: &Tensor,
        _topk_ids: &Tensor,
        _w_size_n: usize,
        _top_k: usize,
        _group_size: usize,
    ) -> Result<Tensor> {
        unimplemented!("T3: marlin-ffi moe_wna16 gemm(已验收件接线)")
    }
}

// ============================================================================
// MoeRouting:路由配置 + topk 选择(DepSeek/Qwen noaux_tc 语义)
// ============================================================================

pub struct MoeRouting {
    pub e_score_correction_bias: Option<Tensor>,
    pub use_sigmoid_scoring: bool,
    pub n_group: usize,
    pub topk_group: usize,
    pub norm_topk_prob: bool,
    pub routed_scaling_factor: Option<f64>,
    pub num_experts_per_tok: usize,
}

impl MoeRouting {
    /// 从 MoE 配置段构建路由配置
    pub fn from_moe_cfg(cfg: &MoEConfig, bias: Option<Tensor>) -> Self {
        let use_sigmoid = cfg.topk_method.as_deref().is_some_and(|m| m == "noaux_tc")
            || cfg.scoring_func.as_deref().is_some_and(|s| s == "sigmoid");
        Self {
            e_score_correction_bias: bias,
            use_sigmoid_scoring: use_sigmoid,
            n_group: cfg.n_group.unwrap_or(1),
            topk_group: cfg.topk_group.unwrap_or(1),
            norm_topk_prob: cfg.norm_topk_prob,
            routed_scaling_factor: cfg.routed_scaling_factor,
            num_experts_per_tok: cfg.num_experts_per_tok,
        }
    }

    /// token → 专家路由,返回 `(topk_weights, topk_ids)`。
    /// `router_logits` 必须 F32 `[num_tokens, num_experts]`。
    ///
    /// 分派骨架:sigmoid/bias 校正 → 分组 topk → 权重归一 → 缩放;
    /// 归约/topk kernel T3 回填(routing kernel = radix/marlin 面已定谳)。
    pub fn route(&self, router_logits: &Tensor, is_prefill: bool) -> Result<(Tensor, Tensor)> {
        let _ = is_prefill;
        if self.use_sigmoid_scoring {
            let scores = super::ops::sigmoid(router_logits)?;
            let scores_for_choice = if let Some(bias) = &self.e_score_correction_bias {
                let bias_f32 = bias.to_dtype(DType::F32)?;
                scores.broadcast_add(&bias_f32)?
            } else {
                scores
            };
            let (topk_weights, topk_ids) =
                moe_shims::topk(&scores_for_choice, self.num_experts_per_tok, 1, true)?;
            let _ = (self.n_group, self.topk_group, self.norm_topk_prob, self.routed_scaling_factor);
            Ok((topk_weights, topk_ids))
        } else {
            let (topk_weights, topk_ids) =
                moe_shims::topk(router_logits, self.num_experts_per_tok, 1, true)?;
            let _ = (self.norm_topk_prob, self.routed_scaling_factor);
            Ok((topk_weights, topk_ids))
        }
    }
}

// ============================================================================
// FusedMoe:未量化 BF16 通用 MoE(gate → route → 专家加权 → AR)
// ============================================================================

/// 字段面 = 装载契约(T3 回填前部分无读者)
#[allow(dead_code)]
pub struct FusedMoe {
    gate: LinearX,
    gate_up_w: Tensor,
    down_w: Tensor,
    w_size_n: usize,
    act: crate::config::Activation,
    routing: MoeRouting,
    world_size: usize,
    dtype: DType,
    gate_dtype: DType,
}

impl FusedMoe {
    /// 打包装载(gate/up/down 三组专家权重;切割管线 = loader 通道)
    pub fn load_packed(
        _cfg: &Config,
        _experts_vb: VarBuilderX,
        _comm: Rc<Comm>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        unimplemented!("T3: FusedMoe::load_packed(loader 通道)")
    }

    pub fn new(cfg: &Config, vb: VarBuilderX, comm: Rc<Comm>, dtype: DType) -> Result<Self> {
        Self::new_with_gate(cfg, vb.pp("gate"), vb.pp("experts"), &vb, comm, dtype)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_gate(
        cfg: &Config,
        gate_vb: VarBuilderX,
        _experts_vb: VarBuilderX,
        _bias_vb: &VarBuilderX,
        comm: Rc<Comm>,
        dtype: DType,
    ) -> Result<Self> {
        let moe_cfg = cfg.moe_cfg.as_ref().expect("MoE config is not available!");
        let num_experts = moe_cfg.num_experts.expect("MoE config missing num_experts");

        assert!(
            cfg.quantization_config.is_none(),
            "Invalid quantization format!"
        );
        let gate_dtype = if cfg.higher_precision_required() {
            DType::F32
        } else {
            dtype
        };
        let gate_linear = linear_no_bias(
            cfg.hidden_size,
            num_experts,
            &gate_vb,
            Shard::default(),
            gate_dtype,
        )?;
        let gate = LinearX::Linear(gate_linear);

        let (gate_w, up_w, down_w) = Self::load_packed(cfg, _experts_vb.clone(), comm.clone())?;
        let gate_up_w = moe_shims::cat(&[gate_w, up_w], 1)?;
        let world_size = comm.world_size;
        let w_size_n = gate_up_w.dim(1)? / 2;

        Ok(Self {
            gate,
            gate_up_w,
            down_w,
            w_size_n,
            act: cfg.hidden_act,
            routing: MoeRouting::from_moe_cfg(moe_cfg, None),
            world_size,
            dtype,
            gate_dtype,
        })
    }

    pub fn forward(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
        let gate_input = if xs.dtype() != self.gate_dtype {
            xs.to_dtype(self.gate_dtype)?
        } else {
            xs.clone()
        };
        let router_logits = self.gate.forward(&gate_input)?.to_dtype(DType::F32)?;
        let (topk_weights, topk_ids) = self.routing.route(&router_logits, is_prefill)?;
        self.forward_with_routing(xs, topk_weights, topk_ids, is_prefill)
    }

    pub fn forward_with_routing(
        &self,
        xs: &Tensor,
        topk_weights: Tensor,
        topk_ids: Tensor,
        _is_prefill: bool,
    ) -> Result<Tensor> {
        let (_num_tokens, _hidden_dim) = xs.dims2()?;
        let _ = (&topk_weights, &topk_ids, self.w_size_n, self.world_size, self.dtype);
        // 未量化 MoE 通路:专家索引 gather → silu 门控 MLP → 加权归并。
        // 体 = OwlTensor 通用算子链(T3 kernel 回填;非 marlin 路径)
        unimplemented!("T3: FusedMoe 通用算子链回填")
    }
}

// ============================================================================
// FusedMoeWNA16:W4A16/W4A8 marlin MoE(decode 主杠杆;P3 已验收)
// ============================================================================

/// 字段面 = 装载契约(T3 回填前部分无读者)
#[allow(dead_code)]
pub struct FusedMoeWNA16 {
    gate: LinearX,
    /// ct 布局(attention-rs 路径);marlin 后端成功构建时为 None(显存互斥)
    gate_up_packed: Option<Tensor>,
    gate_up_scales: Option<Tensor>,
    down_packed: Option<Tensor>,
    down_scales: Option<Tensor>,
    w_size_n: usize,
    act: crate::config::Activation,
    routing: MoeRouting,
    world_size: usize,
    dtype: DType,
    bits: usize,
    group_size: usize,
    gate_dtype: DType,
    legacy_gptq: bool,
    /// marlin moe 后端(marlin-ffi 已验收件;构建失败 = None 回退 ct 布局)
    marlin_backend: Option<()>,
    marlin_failed: AtomicBool,
}

/// marlin moe 块大小选择(= xinfer pick_moe_block_size 直译;分配对齐律)
pub fn pick_moe_block_size(slots: usize, num_experts: usize) -> usize {
    // 分摊 metrix 构建;与 marlin-ffi moe_align_block_size 的粒度契约一致
    let _ = num_experts;
    if slots <= 4 { 4 } else if slots <= 16 { 16 } else { 32 }
}

impl FusedMoeWNA16 {
    pub fn new(
        cfg: &Config,
        vb: VarBuilderX,
        comm: Rc<Comm>,
        dtype: DType,
        quant_cfg: &QuantConfig,
    ) -> Result<Self> {
        Self::new_with_gate(
            cfg,
            vb.pp("gate"),
            vb.pp("experts"),
            &vb,
            comm,
            dtype,
            quant_cfg,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_gate(
        cfg: &Config,
        _gate_vb: VarBuilderX,
        _experts_vb: VarBuilderX,
        _bias_vb: &VarBuilderX,
        comm: Rc<Comm>,
        dtype: DType,
        quant_cfg: &QuantConfig,
    ) -> Result<Self> {
        let moe_cfg = cfg.moe_cfg.as_ref().expect("MoE config is not available!");
        let bits = quant_cfg.bits;
        let legacy_gptq = quant_cfg.quant_method == "gptq";
        if !matches!(bits, 4 | 8) || quant_cfg.group_size <= 0 {
            crate::bail!(
                "WNA16 MoE requires 4/8 bits and a positive group size, got bits={bits}, group_size={}",
                quant_cfg.group_size
            );
        }

        Ok(Self {
            gate: LinearX::QLinear(super::linear::QLinear {
                inner: None,
                wna16: None,
                bias: None,
                dtype,
            }),
            gate_up_packed: None,
            gate_up_scales: None,
            down_packed: None,
            down_scales: None,
            w_size_n: 0,
            act: cfg.hidden_act,
            routing: MoeRouting::from_moe_cfg(moe_cfg, None),
            world_size: comm.world_size,
            dtype,
            bits,
            group_size: quant_cfg.group_size as usize,
            gate_dtype: if cfg.higher_precision_required() { DType::F32 } else { dtype },
            legacy_gptq,
            marlin_backend: None, // marlin-ffi moe_wna16 构建,T3 接线
            marlin_failed: AtomicBool::new(false),
        })
    }

    pub fn forward(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
        let gate_input = if xs.dtype() != self.gate_dtype {
            xs.to_dtype(self.gate_dtype)?
        } else {
            xs.clone()
        };
        let router_logits = self.gate.forward(&gate_input)?.to_dtype(DType::F32)?;
        let (topk_weights, topk_ids) = self.routing.route(&router_logits, is_prefill)?;
        self.forward_with_routing(xs, topk_weights, topk_ids, is_prefill)
    }

    pub fn forward_with_routing(
        &self,
        xs: &Tensor,
        topk_weights: Tensor,
        topk_ids: Tensor,
        _is_prefill: bool,
    ) -> Result<Tensor> {
        if let Some(_bk) = &self.marlin_backend {
            // marlin-ffi moe_wna16 主路(已验收:3080 对 decode 主杠杆)
            return moe_shims::marlin_moe_gemm(
                xs,
                self.gate_up_packed.as_ref().expect("marlin requires packed w"),
                self.gate_up_scales.as_ref().expect("marlin requires scales"),
                &topk_weights,
                &topk_ids,
                self.w_size_n,
                self.routing.num_experts_per_tok,
                self.group_size,
            );
        }
        // ct 布局回退(marlin 构建失败;attention-rs 通路)
        let _ = (self.bits, self.legacy_gptq, &self.marlin_failed);
        unimplemented!("T3: ct 布局 moe 回退通道(attention-rs 通路)")
    }
}

// ============================================================================
// FusedMoeGGUF / FusedMoeISQ:类型面(GGUF 反解已归 loader;ISQ = 运行里程碑)
// ============================================================================

/// GGUF 量化 MoE(体 = loader/gguf 反解 + marlin 路线)
pub struct FusedMoeGGUF {
    _priv: (),
}

impl FusedMoeGGUF {
    pub fn new_repack(
        _cfg: &Config,
        _vb: VarBuilderX,
        _comm: Rc<Comm>,
        _dtype: DType,
    ) -> Result<Self> {
        unimplemented!("T3: FusedMoeGGUF::new_repack(loader/gguf + marlin)")
    }
    pub fn new(_cfg: &Config, _vb: VarBuilderX, _comm: Rc<Comm>, _dtype: DType) -> Result<Self> {
        unimplemented!("T3: FusedMoeGGUF::new")
    }
    pub fn forward(&self, _xs: &Tensor, _is_prefill: bool) -> Result<Tensor> {
        unimplemented!("T3: FusedMoeGGUF::forward")
    }
}

/// ISQ MoE(量化运行里程碑)
pub struct FusedMoeISQ {
    _priv: (),
}

// ============================================================================
// OwlTensor / vendor 新增面需求清单(moe 侧;待主 agent 合并):
// 1. Tensor::topk(dim, k, sorted) -> (values, indices) —— MoeRouting::route
// 2. Tensor::scatter_add(dim, idx, src) —— 专家 scatter 归并
// 3. Tensor::index_select(dim, idx) —— 专家 gather
// 4. Tensor::cat(同 attention 缺面 #1)
// 5. marlin-ffi moe_wna16 gemm 接线(已验收件;非新增,跨 crate 调用面)
// ============================================================================

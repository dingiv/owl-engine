//! WNA16 权重量化层(= xinfer layers/wna16.rs 签名面 + 分派骨架直译)。
//!
//! 翻译注记:
//! 真实路径 = owl marlin-ffi(W4A16/W4A8/moe_wna16,**已验收件**,
//! 见 xinfer docs/dev/marlin基座-施工设计文档与 P1-P4 报告);candle 的
//! 权重 repack/permute/去量化管线(get_scale_perms/marlin_permute_scales
//! 等 ~600 行)在 owl **不搬**——marlin-ffi 的 repacker 是同一语义的
//! 已验收实现,重复搬运 = 双份真源(REQ-DESIGN:参考实现单一来源)。
//! 本文件保:字段面/构造签名/forward 分派骨架(awq_z 通路/W4A8 通路/
//! 标准 W4A16 通路),kernel 调用点 unimplemented 标注运行里程碑。

use super::{Result, Shard, Tensor, VarBuilderX, DType};
use crate::config::QuantConfig;

/// W4A16/W4A8 量化线性权重包(字段面直译;T3 marlin-ffi 装载消费)。
#[derive(Clone)]
#[allow(dead_code)] // 字段面 = 装载契约,T3 回填前无读者
pub struct WNA16 {
    /// marlin 重排后的设备权重(体 = marlin-ffi repacker 产物,T3 回填)
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub scales: Option<Tensor>,
    pub qzeros: Option<Tensor>,
    pub g_idx: Option<Tensor>,
    pub workspace: Option<Tensor>,
    group_size: i32,
    bits: i32,
    dtype: DType,
    is_awq: bool,
    /// W4A8(XINFER_W4A8=1,ct 路径):激活 per-token int8 + s8 scales 契约
    a8: bool,
    a_global_f: f32,
    /// P5-Q1b:ct pack-quantized 非对称(AWQ zp,kU4)通路 —— Some(z) =
    /// forward 走 gemm_v2_awq_raw(内核 f16-only,x/out 由 forward 做位型回转);
    /// z = pack_marlin_z 输出((k/g, n/8) i32)。
    awq_z: Option<Tensor>,
}

impl WNA16 {
    /// 标准构造(签名与 xinfer 对齐;装载体 = marlin-ffi,T3 回填)
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        _vb: &VarBuilderX,
        _shards: Shard,
        _quant_cfg: &Option<QuantConfig>,
        _bias: bool,
        _dtype: DType,
        _is_gptq: bool,
        _module_path: &str,
        _cpu_vb: Option<&VarBuilderX>,
    ) -> Result<Self> {
        let _ = (in_dim, out_dim);
        unimplemented!("WNA16::new = marlin-ffi 权重装载通道,T3 回填")
    }

    /// ct pack-quantized 非对称 AWQ 合并 QKV 的语义分片构造
    /// (P5-Q1b:merged [q|k|v] 张量按 chunk_start/chunk_len 切语义行;
    /// prefetched = S3-P1 层内单次预取的 packed/scales/zp 三元组)。
    #[allow(clippy::too_many_arguments)]
    pub fn new_ct_awq_merged_chunk(
        _vb: &VarBuilderX,
        in_dim: usize,
        out_full: usize,
        chunk_start: usize,
        chunk_len: usize,
        rank: usize,
        world: usize,
        group_size: usize,
        _dtype: DType,
        _tag: &str,
        _prefetched: Option<(&Tensor, &Tensor, &Tensor)>,
    ) -> Result<WNA16> {
        if chunk_len % world != 0 || in_dim % group_size != 0 || out_full % 8 != 0 {
            crate::bail!(
                "ct-awq merged chunk shard divisibility: chunk_len={chunk_len} world={world} in={in_dim} g={group_size}",
            );
        }
        unimplemented!("WNA16::new_ct_awq_merged_chunk = marlin-ffi pack 通道,T3 回填")
    }

    pub fn is_w4a8(&self) -> bool {
        self.a8
    }

    /// S2.5-B:外部 prequant W4A8 matmul(xq = fused_silu_int8 的 i8 [M,K],
    /// xs = f32[M] per-token scale)。仅 2D 输入;调用方(MLP)负责形状。
    pub fn forward_prequant(&self, xq: &Tensor, xs: &Tensor) -> Result<Tensor> {
        if !self.a8 {
            crate::bail!("forward_prequant: layer is not W4A8 (XINFER_W4A8 off at load)");
        }
        if xq.shape().len() != 2 {
            crate::bail!("forward_prequant: xq must be 2D [M,K]");
        }
        let _scale = self.scales.as_ref().expect("w4a8 requires scales");
        // = xinfer utils::gptq::gptq_matmul_a8_prequant(marlin-ffi W4A8 kernel)
        let _ = xs;
        unimplemented!("W4A8 forward_prequant = marlin-ffi s8 kernel,T3 回填")
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // P5-Q1b:ct-AWQ(kU4 zp)通路 —— 内核 f16-only:x→f16,出→模型 dtype 回转
        if let Some(_z) = &self.awq_z {
            // 骨架:2D 输入补 batch 维 → x→F16 → awq_matmul → 位型回转
            // (= xinfer utils::gptq::awq_matmul / marlin-ffi gemm_v2_awq_raw)
            let _ = x.shape().to_vec();
            unimplemented!("awq_matmul = marlin-ffi,T3 回填");
        }
        // 标准 W4A16 marlin gemm(workspace 惰性初始化在 kernel 面)
        unimplemented!("W4A16 forward = marlin-ffi gemm,T3 回填")
    }
}

//! Qwen3.5 文本主干 layers(容器 + load 钩子 + forward 纯声明)。
//!
//! 生命周期三段(2026-09-25 用户裁决;与 TensorOps 完全对称):
//! - **`new` = 准备容器**(零数据、零副作用:登记槽位键名/形状/布局);
//! - **`load_ops(&self)` = 装载的纯描述**(返回 LoaderOps,零副作用;
//!   执行 = `interpreter::eval_load(ops, face, src)` → LoadedBlocks;
//!   毒值/缺键/长度违约在此收割);`mount(&mut self, blocks)` = 纯内存回填;
//! - **`forward` = 纯声明**(槽 decl():未装载 → 毒值声明,eval 边界收割
//!   —— 全程 total,零 panic 零 expect)。
//!
//! 复杂算子 = Kernel 节点(源经 `crate::kernels` 注册表按名取用,层内
//! 零源码);索引/位置量(ids/pos/slots)以 f32 数值形态过线
//! (<2^24 精度无损;S4 u32 索引律的 dtype 扩展留后)。
//!
//! 模块地图(试水批):
//! - [`linear`] / [`rmsnorm`] / [`mlp`]:纯语义算子层(双 face 可对拍)
//! - [`embedding`] / [`rope`]:Kernel 节点试水件
//! - [`attention`]:M-b 批(qk-norm 复用 rmsnorm w_off + gate 切分/naive attn kernel)
//! - [`gdn`]:M-c 批(GatedDeltaNet 18/24 层;批 1 gating 起逐核小步移植)
//! - [`decoder`]:M-d 批(DecoderLayer Full/Gdn 枚举 + 双残差)
//!
//! 主干(批 8)在 [`crate::model`](共有机制);Qwen3.5 预设在
//! [`crate::specs::qwen35`]。
//!
//! 未搬(试水通过后逐个立项,勿一把梭):
//! vision ViT、MTP、chunked prefill、量化。

pub mod attention;
pub mod decoder;
pub mod embedding;
pub mod gdn;
pub mod linear;
pub mod mlp;
pub mod rmsnorm;
pub mod rope;

// ============================================================================
// 层间共享 Kernel 帮手(声明构造;零状态)
// ============================================================================

use crate::TensorOps;

/// 非连续窄切物化(owl_narrow_strided_{f32,f16}):行展平
/// r ∈ [0, outer),dst[r·out_dim + d] = src[r·src_dim + start + d]。
/// attention q gate 切分 / GDN qkv 投影列切分与 conv 权重行切分共用。
/// dtype 跟随 src 声明(F5 整模切换;纯 gather 拷贝,f16 位型直搬)。
pub(crate) fn narrow_strided(
    src: &TensorOps,
    outer: usize,
    src_dim: usize,
    start: usize,
    out_dim: usize,
    shape: crate::contract::Shape,
) -> TensorOps {
    let dt = src.dtype;
    let name = if dt == crate::tensor::Dtype::F16 {
        "owl_narrow_strided_f16"
    } else {
        "owl_narrow_strided_f32"
    };
    TensorOps::of(crate::kernel::kernel_with(
        name,
        (0, 0, 0), // 哨兵:逐元素核,自动 1D ceil/256
        (256, 1, 1),
        0,
    ))
    .arg(src)
    .arg_usize(outer)
    .arg_usize(src_dim)
    .arg_usize(start)
    .arg_usize(out_dim)
    .with_shape(dt, shape)
}

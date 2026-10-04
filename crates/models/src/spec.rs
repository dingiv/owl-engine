//! E5 投机解码公共件(Drafter 域;施工方案 §2.1)。
//!
//! M4 落地:fold 树构建(§四.4 v2)—— 部分接受后从快照态重放已接受
//! 前缀,**零整模前向**(复用现役 prefill 核消费 verify 图的 GDN 记录)。

use crate::contract::{Dtype, ModelError};
use crate::layers::gdn::{fold_layer, GdnBuffers};
use crate::model::Model;
use crate::tensor::TensorOps;

/// 每层记录数(conv 三 raw + 递推五件;gdn.rs tap 推入序)
pub const REC_PER_LAYER: usize = 8;

/// fold 树(E5-M4):遍历模型的 GDN 层,按 verify 图 tap 记录重放
/// 已接受前缀 m+1 行(行 0 = anchor,行 1..=m = 已接受草稿)。
///
/// - `rec`:记录平铺(8 × n_gdn;引擎侧 = verify 图输出块的 of_block
///   前缀视图,形状 [m+1, ·]);
/// - `gdns`:会话 GDN 状态块组叶子(restore 批量通道已回 state@F-1);
/// - 返回 = 4 × n_gdn 个副作用根(conv ×3 + 递推;multi-root eval
///   消费,状态原地推进 → state@F+m)。
pub fn fold_tree(
    model: &Model,
    rec: &[TensorOps],
    gdns: &[GdnBuffers],
    slots: &TensorOps,
    cu: &TensorOps,
    slot_host: usize,
    m: usize,
    scalar: bool,
) -> Result<Vec<TensorOps>, ModelError> {
    let n_gdn = gdns.len();
    if rec.len() != REC_PER_LAYER * n_gdn {
        return Err(ModelError::Msg(format!(
            "fold_tree: 记录数 {} ≠ 8 × GDN 层数 {}",
            rec.len(),
            n_gdn
        )));
    }
    let m1 = m + 1;
    let _ = m1; // cu 由调用方供给(图 = 输入槽;eager = from_host)
    let mut roots = Vec::with_capacity(4 * n_gdn);
    let mut gi = 0usize;
    for layer in &model.layers {
        if layer.is_full() {
            continue;
        }
        let net = layer
            .gdn_mixer()
            .ok_or_else(|| ModelError::Msg("fold_tree: GDN 层缺 mixer".into()))?;
        roots.extend(fold_layer(
            net,
            &rec[gi * REC_PER_LAYER..(gi + 1) * REC_PER_LAYER],
            &gdns[gi],
            slots,
            slot_host,
            &cu,
            m1,
            scalar,
        ));
        gi += 1;
    }
    if gi != n_gdn {
        return Err(ModelError::Msg(format!(
            "fold_tree: 模型 GDN 层数 {gi} ≠ 状态组 {n_gdn}"
        )));
    }
    Ok(roots)
}

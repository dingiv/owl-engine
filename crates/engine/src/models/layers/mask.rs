//! 注意力因果 mask(= xinfer layers/mask.rs 直译;flash 分支裁剪)。
//!
//! xinfer 生产路径走 flash/flashattn/flashinfer(kernel 内 causal),
//! 本编译口径只保留非 flash 分支的骨架;vendor causal_mask 归宿
//! 随 vendor 三选一裁决,体为占位。

use super::{DType, Device, Result, Tensor};
use crate::bail;

pub fn get_attention_causal_mask(
    device: &Device,
    dtype: DType,
    _: &Tensor,
    seqlens: Vec<u32>,
    sliding_window: Option<usize>,
    is_prefill: bool,
) -> Option<Vec<Tensor>> {
    if !is_prefill {
        return None;
    }
    let mut vec_mask = Vec::new();
    let mut start = 0u32;
    for seq_offset in seqlens.iter() {
        let seq_len = *seq_offset - start;
        let mask = get_causal_mask_internal(device, dtype, seq_len as usize, sliding_window)
            .unwrap_or_else(|e| panic!("causal_mask: {e}"));
        vec_mask.push(mask);
        start = *seq_offset;
    }
    Some(vec_mask)
}

fn get_causal_mask_internal(
    _device: &Device,
    _dtype: DType,
    tgt_len: usize,
    sliding_window: Option<usize>,
) -> Result<Tensor> {
    // vendor attention_rs::mask::causal_mask(归宿裁决随 vendor 三选一;
    // 骨架保留 tgt_len/sliding_window 语义,T3 kernel 回填)
    let _ = (tgt_len, sliding_window);
    bail!("vendor causal_mask: T3 kernel 回填(vendor 归宿裁决待定)")
}

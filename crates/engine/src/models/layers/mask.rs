//! 注意力因果 mask(= xinfer layers/mask.rs 直译;flash 分支裁剪)。
//!
//! xinfer 生产路径走 flash/flashattn/flashinfer(kernel 内 causal),
//! 本编译口径只保留非 flash 分支的骨架;vendor causal_mask 归宿
//! 随 vendor 三选一裁决,体为占位。

use super::{DType, Device, Result, Tensor};
use super::OwlTensor;

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
    device: &Device,
    dtype: DType,
    tgt_len: usize,
    sliding_window: Option<usize>,
) -> Result<Tensor> {
    // candle 经典 additive mask:[1, 1, tgt, tgt];允许位 = 0,禁止位 = -inf。
    // 消费面 = softmax 前 broadcast_add(attention.rs naive 分支)。
    // sliding_window:q 位 i 对 kv 位 j 允许 ⇔ j ≤ i 且 i - j < window。
    let neg_inf = f32::NEG_INFINITY;
    let mut v = vec![0f32; tgt_len * tgt_len];
    for i in 0..tgt_len {
        for j in 0..tgt_len {
            let banned = j > i
                || sliding_window
                    .is_some_and(|w| i - j >= w);
            if banned {
                v[i * tgt_len + j] = neg_inf;
            }
        }
    }
    let t = super::ctor::from_vec(v, (tgt_len, tgt_len), device)?;
    let t = if dtype == DType::F32 {
        t
    } else {
        t.to_dtype(dtype)?
    };
    Ok(t.reshape((1usize, 1, tgt_len, tgt_len))?)
}

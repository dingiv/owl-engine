//! DeepStack 视觉嵌入注入(= xinfer layers/deepstack.rs 直译)。
//! 参考 mistral.rs qwen3_vl_moe text.rs deepstack_process。

use super::{ctor, DType, OwlTensor, Result, Tensor};
use crate::bail;

pub trait ApplyDeepStack {
    fn apply_deep_stack(
        &self,
        visual_pos_masks: &Tensor,
        visual_embeds: &Tensor,
    ) -> Result<Tensor>;
}

impl ApplyDeepStack for Tensor {
    fn apply_deep_stack(
        &self,
        visual_pos_masks: &Tensor,
        visual_embeds: &Tensor,
    ) -> Result<Tensor> {
        deepstack_process(self, visual_pos_masks, visual_embeds)
    }
}

fn deepstack_process(
    hidden_states: &Tensor,
    visual_pos_masks: &Tensor,
    visual_embeds: &Tensor,
) -> Result<Tensor> {
    let device = hidden_states.device();
    let dtype = hidden_states.dtype();

    let mask = visual_pos_masks.to_device(&device)?.to_dtype(DType::F32)?;
    let mask_flat = mask.flatten_all()?;

    let masked_count = mask_flat.sum_all()?.to_scalar::<f32>()? as usize;
    let visual_embeds = visual_embeds.to_device(&device)?.to_dtype(dtype)?;

    if masked_count == 0 {
        if visual_embeds.dim(0)? != 0 {
            bail!(
                "DeepStack visual embeds ({}) provided but mask is empty",
                visual_embeds.dim(0)?
            );
        }
        return Ok(hidden_states.clone());
    }

    if visual_embeds.dim(0)? != masked_count {
        bail!(
            "Mismatch between DeepStack visual embeds ({}) and mask positions ({})",
            visual_embeds.dim(0)?,
            masked_count
        );
    }

    let (total_positions, hidden) = hidden_states.dims2()?;

    let prefix = mask_flat.cumsum(0)?;
    let rank = prefix.sub(&mask_flat)?.mul(&mask_flat)?;
    let rank_u32 = rank.to_dtype(DType::U32)?;

    let positions = ctor::arange(0, total_positions, DType::U32, &device)?;
    let positions_f32 = positions.to_dtype(DType::F32)?;
    let masked_positions = positions_f32.mul(&mask_flat)?;

    let position_per_rank =
        ctor::zeros((masked_count,), DType::F32, &device)?
            .scatter_add(&rank_u32, &masked_positions, 0)?;
    let position_per_rank = position_per_rank.to_dtype(DType::U32)?;

    let linear_index = position_per_rank.unsqueeze(1)?.repeat((1, hidden))?;

    hidden_states.scatter_add(&linear_index, &visual_embeds, 0)
}

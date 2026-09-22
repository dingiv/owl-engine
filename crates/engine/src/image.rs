//! 多模态图像数据结构——从 xinfer `crates/core/src/utils/image.rs` 只搬
//! `ImageData`(纯数据 + serde)。图像处理管线(ImageProcessor/compute_*)
//! 依赖 candle Tensor 与 image crate,随 T3 模型搬运(VL 模型)时再议。

use owl_iface::{Device, Pool};
use owl_nn::tensor::{Tensor, TensorPoolOps};
use serde::{Deserialize, Serialize};

/// 已切片的图像 token 数据(多模态路径的引擎内表示)。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageData {
    pub raw: Vec<u8>,
    pub shape: Vec<usize>,
    pub patches: Vec<(usize, usize)>,
    pub image_idx: i32,
    #[serde(default)]
    pub image_token_offset: usize,
    #[serde(default)]
    pub tokens_per_image: Vec<usize>,
    #[serde(default)]
    pub image_token_id: Option<u32>,
}

impl ImageData {
    /// 原始字节(f32 位流)→ 池装载 f32 张量。
    /// 对应 xinfer `ImageData::to_tensor_f32`(candle `Tensor::from_slice`),
    /// 落点改为 owl 池直连工厂(裁决 5:分配入口在 Pool)。
    pub fn to_tensor_f32<P, D>(
        &self,
        pool: &P,
    ) -> Result<Tensor<f32, D>, owl_iface::BackendError>
    where
        P: Pool<Dev = D>,
        D: Device<Pool = P>,
    {
        let floats: &[f32] = bytemuck::cast_slice(&self.raw);
        pool.from_vec_tensor(&self.shape, floats.to_vec())
    }
}

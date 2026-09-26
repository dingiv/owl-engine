//! f32 值块:CPU 后端的数据平面(host 真值;自 models/interpreter.rs §1 迁入)。
//!
//! GPU 版对应物 = `owl_iface::contract::Bytes`(池块句柄,数据不过线);
//! CPU 版直接持真值 —— 这就是两后端形态差异的根源。

use owl_iface::contract::{Dtype, ModelError, Shape};

/// f32 值块(朴素 host 真值)
#[derive(Clone, Debug, PartialEq)]
pub struct Value {
    pub f32: Vec<f32>,
    pub shape: Shape,
}

impl Value {
    /// 构造(外部 demo/测试用)
    pub fn new(f32: Vec<f32>, shape: Shape) -> Self {
        Self { f32, shape }
    }

    pub(crate) fn zero(dtype: Dtype, shape: &Shape) -> Result<Self, ModelError> {
        if dtype != Dtype::F32 {
            return Err(ModelError::Msg(format!(
                "CPU 后端仅 F32(S1),得 {dtype:?}"
            )));
        }
        Ok(Value {
            f32: vec![0.0; owl_iface::contract::numel(shape)],
            shape: shape.clone(),
        })
    }

    pub(crate) fn from_bytes(dtype: Dtype, shape: &Shape, bytes: &[u8]) -> Result<Self, ModelError> {
        // f16 基线(F5):CPU 面存储仍 f32,f16 字节解码入账(装载冒烟用;
        // 执行面照旧仅 F32 —— CPU 先不搞,用户裁决)。其余 dtype 拒。
        if dtype == Dtype::F16 {
            let n = owl_iface::contract::numel(shape);
            if bytes.len() != n * 2 {
                return Err(ModelError::Msg(format!(
                    "htod(f16): 字节数 {} != {}×2",
                    bytes.len(),
                    n
                )));
            }
            return Ok(Value {
                f32: bytes
                    .chunks_exact(2)
                    .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                    .collect(),
                shape: shape.clone(),
            });
        }
        if dtype != Dtype::F32 {
            return Err(ModelError::Msg("CPU 后端仅 F32/F16-h 工装载;执行仅 F32".into()));
        }
        let n = owl_iface::contract::numel(shape);
        if bytes.len() != n * 4 {
            return Err(ModelError::Msg(format!(
                "htod: 字节数 {} != {}×4",
                bytes.len(),
                n
            )));
        }
        Ok(Value {
            f32: bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            shape: shape.clone(),
        })
    }
}

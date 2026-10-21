//! **GGUF → f32 装载源**(2026-10-21;MTP 头首用)。
//!
//! 定位:不是生产热路径的「零解量化」通道(那条仍是 [`super::source::GgufSource`]),
//! 而是**小件/异型头**的装载适配器:把 GGUF 块在 host 解成 f32,再走既有
//! F16 `Linear` 装载面。首个消费者 = MTP 头(GGUF MTP 块的线性权重逐角色
//! 混 Q6_K/Q8_0/Q4_K;统一解到 F16 比给 DecoderLayer 造逐线性 plan 简单,
//! 且 MTP 头只有 ~0.8GB)。
//!
//! `eh_proj` 的 in 维两半(`fc_e`/`fc_h`)在这里按列切 f32;列宽 2560 是
//! QK_K 的整数倍,但本源只做元素面,不依赖块对齐。

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::contract::{Dtype, ModelError};
use crate::module::{Taken, WeightSource};

use super::dequant::dequant_all;
use super::reader::Gguf;

/// GGUF 解量化装载源(单卡;MTP 头专用)
pub struct GgufDequantSource {
    g: Gguf,
    /// 键 → f32(行主序 [rows, inner];虚拟半键为 [rows, inner/2])
    cache: Mutex<HashMap<String, Arc<Vec<f32>>>>,
}

impl GgufDequantSource {
    pub fn open(path: &std::path::Path) -> Result<Self, ModelError> {
        Ok(Self {
            g: Gguf::open(path)?,
            cache: Mutex::new(HashMap::new()),
        })
    }

    /// `blk.<n>.nextn.eh_proj.{e,h}.weight` → (物理键, high)
    fn fc_half(key: &str) -> Option<(String, bool)> {
        let rest = key.strip_prefix("blk.")?;
        let (block, rest) = rest.split_once('.')?;
        let (seg, tail) = rest.strip_prefix("nextn.eh_proj.")?.split_once('.')?;
        if tail != "weight" {
            return None;
        }
        let high = match seg {
            "e" => false,
            "h" => true,
            _ => return None,
        };
        Some((format!("blk.{block}.nextn.eh_proj.weight"), high))
    }

    /// 读取(带缓存)一个键的 f32 逻辑值
    fn load(&self, key: &str) -> Taken<Arc<Vec<f32>>> {
        {
            let c = self.cache.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(v) = c.get(key) {
                return Ok(Some(v.clone()));
            }
        }
        let (phys, high) = match Self::fc_half(key) {
            Some((p, h)) => (p, Some(h)),
            None => (key.to_string(), None),
        };
        let Some(t) = self.g.get_tensor(&phys) else {
            return Ok(None);
        };
        let rows = t.rows() as usize;
        let inner = t.inner() as usize;
        let ix = self
            .g
            .index_of(&phys)
            .ok_or_else(|| ModelError::Msg(format!("GGUF: 张量 {phys} 无索引")))?;
        let full = dequant_all(t.ty, self.g.bytes(ix), rows, inner)
            .map_err(|e| ModelError::Msg(format!("GGUF 解量化 {phys}: {e}")))?;
        let v: Arc<Vec<f32>> = match high {
            None => Arc::new(full),
            Some(high) => {
                if inner % 2 != 0 {
                    return Err(ModelError::Msg(format!(
                        "GGUF MTP {phys}: 内维 {inner} 不是偶数(不能列半拆)"
                    )));
                }
                let half = inner / 2;
                let skip = if high { half } else { 0 };
                let mut out = Vec::with_capacity(rows * half);
                for r in 0..rows {
                    let b = r * inner + skip;
                    out.extend_from_slice(&full[b..b + half]);
                }
                Arc::new(out)
            }
        };
        let mut c = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        c.insert(key.to_string(), v.clone());
        Ok(Some(v))
    }
}

impl WeightSource for GgufDequantSource {
    fn take(&self, key: &str) -> Taken<Vec<f32>> {
        Ok(self.load(key)?.map(|v| (*v).clone()))
    }

    fn elem_len(&self, key: &str) -> Taken<usize> {
        if let Some((phys, _)) = Self::fc_half(key) {
            let Some(t) = self.g.get_tensor(&phys) else { return Ok(None) };
            let rows = t.rows() as usize;
            let inner = t.inner() as usize;
            if inner % 2 != 0 {
                return Err(ModelError::Msg(format!(
                    "GGUF MTP {phys}: 内维 {inner} 不是偶数(不能列半拆)"
                )));
            }
            return Ok(Some(rows * inner / 2));
        }
        Ok(self.g.get_tensor(key).map(|t| t.rows() as usize * t.inner() as usize))
    }

    fn native_dtype(&self, _key: &str) -> Taken<Dtype> {
        // 元素面已解成 f32;目标 dtype 由装载面 cast(F16/F32 均可)
        Ok(None)
    }

    fn convert_chunk_into_bytes(
        &self,
        key: &str,
        offset: usize,
        len: usize,
        dst: &mut [u8],
        dtype: Dtype,
    ) -> Taken<()> {
        let Some(v) = self.load(key)? else { return Ok(None) };
        let Some(slice) = v.get(offset..offset + len) else { return Ok(None) };
        let esz = match dtype {
            Dtype::F32 => 4,
            Dtype::F16 | Dtype::BF16 => 2,
            other => {
                return Err(ModelError::Msg(format!(
                    "GgufDequantSource: 键 {key} 目标 dtype {other:?} 未接"
                )))
            }
        };
        if dst.len() < len * esz {
            return Err(ModelError::Msg(format!(
                "GgufDequantSource: 键 {key} dst {}B < 需 {}B",
                dst.len(),
                len * esz
            )));
        }
        for (i, x) in slice.iter().enumerate() {
            match dtype {
                Dtype::F32 => dst[i * 4..i * 4 + 4].copy_from_slice(&x.to_le_bytes()),
                Dtype::F16 => dst[i * 2..i * 2 + 2]
                    .copy_from_slice(&half::f16::from_f32(*x).to_le_bytes()),
                Dtype::BF16 => dst[i * 2..i * 2 + 2]
                    .copy_from_slice(&half::bf16::from_f32(*x).to_le_bytes()),
                _ => unreachable!("上方已筛"),
            }
        }
        Ok(Some(()))
    }
}

//! **源精度轴**(`docs/arch/装载管线-源精度x目标格式-设计.md` §3.1/§3.3)。
//!
//! 病根:装载器原来只认"目标槽 dtype"(`Want.dtype`),**源是 f16 还是 bf16/fp8
//! 靠上下文明白** —— fp8 源与 f16 源走同一分支,静默按 f16 读就是错值。
//! 这里把源精度提成**第一轴**:它来自文件元数据([`SrcDtype::from_dtype`] /
//! [`SrcDtype::from_safetensors`]),变换只吃 [`decode_to_f32`] 的产物。
//!
//! 纪律(与 `admit.rs` 的"不许静默降级"同律):
//! - `SrcDtype` **禁止由 plan 反推**(plan 是目标侧愿望,不是源的事实);
//! - 表外/未实现组合 = **响亮拒**,决不许"按 f16 猜着读"。
//!
//! 今天的事实源 = [`crate::module::WeightSource::native_dtype`](它报的是文件
//! 头里那个 dtype,经 [`SrcDtype::from_dtype`] 归类)。fp8 源(P4)接入时要一个
//! **头字符串 → `SrcDtype`** 的口(`F8_E4M3` 与 `F8_E4M3FNUZ` 是两个码表,
//! 不能混);那个函数跟它的**第一个消费者**一起加,不先摆一个没人调的空壳。

use crate::contract::{Dtype, ModelError};

/// 源张量的精度事实(safetensors 头里的 `dtype` 字段)。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SrcDtype {
    F16,
    BF16,
    F32,
    /// fp8(4 位指数 3 位尾数,无 inf;`e4m3_fnuz` 不在本表 —— 见到就拒)
    F8E4M3,
    /// fp8(5 位指数 2 位尾数)
    F8E5M2,
}

impl SrcDtype {
    /// 单元素字节数
    #[must_use]
    pub const fn size_bytes(self) -> usize {
        match self {
            SrcDtype::F16 | SrcDtype::BF16 => 2,
            SrcDtype::F32 => 4,
            SrcDtype::F8E4M3 | SrcDtype::F8E5M2 => 1,
        }
    }

    /// 契约 dtype → 源精度(**只有这五种是"源的事实"**;其余返回 `None`)
    #[must_use]
    pub const fn from_dtype(dt: Dtype) -> Option<SrcDtype> {
        match dt {
            Dtype::F16 => Some(SrcDtype::F16),
            Dtype::BF16 => Some(SrcDtype::BF16),
            Dtype::F32 => Some(SrcDtype::F32),
            Dtype::F8E4M3 => Some(SrcDtype::F8E4M3),
            Dtype::F8E5M2 => Some(SrcDtype::F8E5M2),
            _ => None,
        }
    }

    /// 诊断面短名
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            SrcDtype::F16 => "f16",
            SrcDtype::BF16 => "bf16",
            SrcDtype::F32 => "f32",
            SrcDtype::F8E4M3 => "fp8-e4m3",
            SrcDtype::F8E5M2 => "fp8-e5m2",
        }
    }
}

/// **唯一的精度归一点**:源字节 → f32(`n` 个元素)。
///
/// 为什么集中一处:四种 decode 都是常数代价的位运算/查表,分散实现必然漂移
/// (fp8 的 e4m3 尤其:核内 `e4m3_dec` 与 host 参考两套写法曾差 28/32768 字节)。
/// 之后所有变换只看 `&[f32]`,**不再出现"源是什么精度"的分支**。
///
/// # Errors
/// `src.len()` 不是 `dt.size_bytes()` 的整数倍、或 `out.len() < n` ⇒ 拒。
pub fn decode_to_f32(src: &[u8], dt: SrcDtype, out: &mut [f32]) -> Result<(), ModelError> {
    let esz = dt.size_bytes();
    if src.len() % esz != 0 {
        return Err(ModelError::Msg(format!(
            "源字节长度 {} 不是 {} ({esz}B) 的整数倍",
            src.len(),
            dt.name()
        )));
    }
    let n = src.len() / esz;
    if out.len() < n {
        return Err(ModelError::Msg(format!(
            "decode 输出缓冲太小:{} < {n}({} 个元素)",
            out.len(),
            dt.name()
        )));
    }
    let out = &mut out[..n];
    match dt {
        SrcDtype::F16 => {
            for (o, c) in out.iter_mut().zip(src.chunks_exact(2)) {
                *o = half::f16::from_le_bytes([c[0], c[1]]).to_f32();
            }
        }
        SrcDtype::BF16 => {
            for (o, c) in out.iter_mut().zip(src.chunks_exact(2)) {
                // bf16 → f32 是精确展宽(尾数 7 → 23 位,零舍入)
                *o = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16);
            }
        }
        SrcDtype::F32 => {
            for (o, c) in out.iter_mut().zip(src.chunks_exact(4)) {
                *o = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
        }
        SrcDtype::F8E4M3 => {
            // 与核内 `e4m3_dec` 同式(`cu/nvfp4/*.cu` 走硬件转换,oracle 里
            // 有一份被 GPU 逐字节验证过的解析式 ⇒ 这里复用它,不另写一套)
            let lut = owl_kernels::ops::oracles::e4m3_lut();
            for (o, c) in out.iter_mut().zip(src.iter()) {
                *o = lut[*c as usize];
            }
        }
        SrcDtype::F8E5M2 => {
            for (o, c) in out.iter_mut().zip(src.iter()) {
                *o = e5m2_decode(*c);
            }
        }
    }
    Ok(())
}

/// e5m2(1+5+2)解析式:IEEE 半精度式偏移 15 的指数(无 fnuz 变体)。
fn e5m2_decode(b: u8) -> f32 {
    let s = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let e = ((b >> 2) & 0x1F) as i32;
    let m = (b & 0x03) as f32;
    match e {
        0 => s * m * 2f32.powi(-16),          // 次正规(尾数 × 2^-14 / 4)
        0x1F if m != 0.0 => f32::NAN,         // 指数全 1 + 尾数非零 = NaN
        0x1F => s * f32::INFINITY,            // 指数全 1 + 尾数零 = inf
        _ => s * (1.0 + m / 4.0) * 2f32.powi(e - 15),
    }
}

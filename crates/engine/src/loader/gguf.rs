//! GGUF 格式解析 + 量化反解 + GGufVarBuilder(T2-三 loader 搬运)。
//!
//! port 出处:
//! - 格式解析:vendor/candle/candle-core/src/quantized/gguf_file.rs
//!   (rev 23a6f38;Content/Value/TensorInfo/magic 的 host 侧读取,
//!   设备路径不搬——A4 所有权转移,owl 自有);
//! - 量化块布局与反解:同目录 k_quants.rs + mod.rs 的 `GgmlDType`
//!   表(host dequant;vec_dot/matmul/SIMD 不搬,推理侧另有 kernel 面);
//! - f16/bf16 位型转换:half crate 等价位算法手写(不引依赖)。
//!
//! 与 xinfer 的差异:
//! - `QTensor`/设备缓冲不搬:原始块字节按 dtype 读出为
//!   [`GgufRawTensor`],反解走 [`dequantize_to_f32`] 后经
//!   [`WeightAllocator`](super::alloc::WeightAllocator) 落显存/内存;
//! - IQ 系(1_S/1_M/2_XXS/2_XS/2_S/3_XXS/3_S/4_NL/4_XS)仅登记
//!   type_size/block_size(数据段偏移需要),反解 = 结构化报错
//!   (运行里程碑回填;xinfer 生产面未见 IQ 权重);
//! - 烘焙缓存目录/环境变量 owl 化(见 load_cache.rs)。

use crate::error::{Error, Result};
use owl_nn::Dtype;
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};

pub const DEFAULT_ALIGNMENT: u64 = 32;

// ============================================================================
// LE 读原语(不引 byteorder;全部 little-endian)
// ============================================================================

fn rd_u8<R: Read>(r: &mut R) -> std::io::Result<u8> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    Ok(b[0])
}
fn rd_u16<R: Read>(r: &mut R) -> std::io::Result<u16> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b)?;
    Ok(u16::from_le_bytes(b))
}
fn rd_u32<R: Read>(r: &mut R) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn rd_u64<R: Read>(r: &mut R) -> std::io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}
fn rd_i8<R: Read>(r: &mut R) -> std::io::Result<i8> {
    Ok(rd_u8(r)? as i8)
}
fn rd_i16<R: Read>(r: &mut R) -> std::io::Result<i16> {
    Ok(rd_u16(r)? as i16)
}
fn rd_i32<R: Read>(r: &mut R) -> std::io::Result<i32> {
    Ok(rd_u32(r)? as i32)
}
fn rd_i64<R: Read>(r: &mut R) -> std::io::Result<i64> {
    Ok(rd_u64(r)? as i64)
}
fn rd_f32<R: Read>(r: &mut R) -> std::io::Result<f32> {
    Ok(f32::from_bits(rd_u32(r)?))
}
fn rd_f64<R: Read>(r: &mut R) -> std::io::Result<f64> {
    Ok(f64::from_bits(rd_u64(r)?))
}

// ============================================================================
// half 位型转换(手写;等价 half::f16/bf16::to_f32)
// ============================================================================

/// IEEE 754 binary16 位型 → f32(标准展开算法;无查表,常数路径)
pub(crate) fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;
    let f32_bits = match exp {
        0 => {
            if frac == 0 {
                sign << 31 // ±0
            } else {
                // 次正规:归一化
                let mut e = 0u32;
                let mut f = frac;
                while f & 0x400 == 0 {
                    f <<= 1;
                    e += 1;
                }
                f &= 0x3ff;
                (sign << 31) | ((127 - 15 - e) << 23) | (f << 13)
            }
        }
        0x1f => (sign << 31) | (0xff << 23) | (frac << 13), // inf/nan
        e => (sign << 31) | ((e + 127 - 15) << 23) | (frac << 13),
    };
    f32::from_bits(f32_bits)
}

/// bfloat16 位型 → f32(高 16 位截断;精确)
pub(crate) fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

// ============================================================================
// GgmlDType 表(type_size/block_size 为数据段偏移与反解的公共事实)
// ============================================================================

/// GGML 量化 dtype(= candle `quantized::GgmlDType`,全变体登记:
/// 数据段偏移计算需要全表;反解按用到面实现,见 [`dequantize_to_f32`])。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum GgmlDType {
    F32,
    F16,
    BF16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    IQ2_XXS,
    IQ2_XS,
    IQ3_XXS,
    IQ1_S,
    IQ4_NL,
    IQ3_S,
    IQ2_S,
    IQ4_XS,
    IQ1_M,
}

impl core::fmt::Display for GgmlDType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let s = match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::Q4_0 => "q4_0",
            Self::Q4_1 => "q4_1",
            Self::Q5_0 => "q5_0",
            Self::Q5_1 => "q5_1",
            Self::Q8_0 => "q8_0",
            Self::Q2K => "q2k",
            Self::Q3K => "q3k",
            Self::Q4K => "q4k",
            Self::Q5K => "q5k",
            Self::Q6K => "q6k",
            Self::Q8K => "q8k",
            Self::IQ2_XXS => "iq2_xxs",
            Self::IQ2_XS => "iq2_xs",
            Self::IQ3_XXS => "iq3_xxs",
            Self::IQ1_S => "iq1_s",
            Self::IQ4_NL => "iq4_nl",
            Self::IQ3_S => "iq3_s",
            Self::IQ2_S => "iq2_s",
            Self::IQ4_XS => "iq4_xs",
            Self::IQ1_M => "iq1_m",
        };
        f.write_str(s)
    }
}

impl GgmlDType {
    pub fn from_u32(u: u32) -> Result<Self> {
        let d = match u {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            16 => Self::IQ2_XXS,
            17 => Self::IQ2_XS,
            18 => Self::IQ3_XXS,
            19 => Self::IQ1_S,
            20 => Self::IQ4_NL,
            21 => Self::IQ3_S,
            22 => Self::IQ2_S,
            23 => Self::IQ4_XS,
            29 => Self::IQ1_M,
            30 => Self::BF16,
            _ => return Err(Error::Msg(format!("gguf: unknown tensor dtype {u}"))),
        };
        Ok(d)
    }

    /// 每块字节数(repr(C) 布局逐字节转录;candle `type_size`)
    pub fn type_size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q4_0 => 18,  // d(2) + qs 16
            Self::Q4_1 => 20,  // d + m + qs 16
            Self::Q5_0 => 22,  // d + qh 4 + qs 16
            Self::Q5_1 => 24,  // d + m + qh 4 + qs 16
            Self::Q8_0 => 34,  // d(2) + i8×32
            // K 系(QK_K=256)
            Self::Q2K => 84,   // scales 16 + qs 64 + d + dmin
            Self::Q3K => 110,  // hmask 32 + qs 64 + scales 12 + d
            Self::Q4K => 144,  // d + dmin + scales 12 + qs 128
            Self::Q5K => 176,  // d + dmin + scales 12 + qh 32 + qs 128
            Self::Q6K => 210,  // ql 128 + qh 64 + scales 16 + d
            Self::Q8K => 292,  // d f32 + i8×256 + bsums 16×i16
            // IQ 系(仅偏移用)
            Self::IQ2_XXS => 34, // d + u16×16
            Self::IQ2_XS => 42,  // d + u16×16 + u8×8
            Self::IQ3_XXS => 98, // d + u8×96
            Self::IQ1_S => 50,   // d + u8×32 + u16×8
            Self::IQ4_NL => 18,  // d + u8×16
            Self::IQ3_S => 110,  // d + qs 64 + qh 32 + signs 32 + scales 4
            Self::IQ2_S => 65,   // d + qs 64 + qh 8 + scales 8
            Self::IQ4_XS => 136, // d + scales_h u16 + scales_l 4 + qs 128
            Self::IQ1_M => 56,   // qs 32 + qh 16 + scales 8(无 d)
        }
    }

    /// 每块元素数
    pub fn block_size(self) -> usize {
        const QK_K: usize = 256;
        match self {
            Self::F32 | Self::F16 | Self::BF16 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 => 32,
            Self::IQ4_NL => 32,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => QK_K,
            Self::IQ2_XXS | Self::IQ2_XS | Self::IQ3_XXS | Self::IQ1_S | Self::IQ3_S
            | Self::IQ2_S | Self::IQ4_XS | Self::IQ1_M => QK_K,
        }
    }
}

// ============================================================================
// Value / ValueType(= candle gguf_file::Value;读全量,写测试用最小面)
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    F32,
    F64,
    Bool,
    String,
    Array,
}

impl ValueType {
    fn from_u32(v: u32) -> Result<Self> {
        let t = match v {
            0 => Self::U8,
            1 => Self::I8,
            2 => Self::U16,
            3 => Self::I16,
            4 => Self::U32,
            5 => Self::I32,
            6 => Self::F32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::U64,
            11 => Self::I64,
            12 => Self::F64,
            _ => return Err(Error::Msg(format!("gguf: unrecognized value-type {v:#08x}"))),
        };
        Ok(t)
    }
}

#[derive(Debug, Clone)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
}

macro_rules! value_getter {
    ($fn_name:ident, $t:ty, $variant:ident, $what:literal) => {
        pub fn $fn_name(&self) -> Result<$t> {
            match self {
                Value::$variant(v) => Ok(*v),
                v => Err(Error::Msg(format!("not a {} {v:?}", $what))),
            }
        }
    };
}

impl Value {
    value_getter!(to_u8, u8, U8, "u8");
    value_getter!(to_i8, i8, I8, "i8");
    value_getter!(to_u16, u16, U16, "u16");
    value_getter!(to_i16, i16, I16, "i16");
    value_getter!(to_u32, u32, U32, "u32");
    value_getter!(to_i32, i32, I32, "i32");
    value_getter!(to_f32, f32, F32, "f32");
    value_getter!(to_f64, f64, F64, "f64");
    value_getter!(to_bool, bool, Bool, "bool");

    /// 自动上档(candle 语义:整型/Bool 无损升 u64)
    pub fn to_u64(&self) -> Result<u64> {
        match self {
            Value::U64(v) => Ok(*v),
            Value::U8(v) => Ok(*v as u64),
            Value::U16(v) => Ok(*v as u64),
            Value::U32(v) => Ok(*v as u64),
            Value::Bool(v) => Ok(*v as u64),
            v => Err(Error::Msg(format!("not a u64 or upcastable to u64 {v:?}"))),
        }
    }

    pub fn to_i64(&self) -> Result<i64> {
        match self {
            Value::I64(v) => Ok(*v),
            v => Err(Error::Msg(format!("not a i64 {v:?}"))),
        }
    }

    pub fn to_vec(&self) -> Result<&Vec<Value>> {
        match self {
            Value::Array(v) => Ok(v),
            v => Err(Error::Msg(format!("not a vec {v:?}"))),
        }
    }

    pub fn to_string(&self) -> Result<&String> {
        match self {
            Value::String(v) => Ok(v),
            v => Err(Error::Msg(format!("not a string {v:?}"))),
        }
    }

    fn read<R: Read>(r: &mut R, value_type: ValueType, magic: &VersionedMagic) -> Result<Self> {
        let v = match value_type {
            ValueType::U8 => Self::U8(rd_u8(r)?),
            ValueType::I8 => Self::I8(rd_i8(r)?),
            ValueType::U16 => Self::U16(rd_u16(r)?),
            ValueType::I16 => Self::I16(rd_i16(r)?),
            ValueType::U32 => Self::U32(rd_u32(r)?),
            ValueType::I32 => Self::I32(rd_i32(r)?),
            ValueType::U64 => Self::U64(rd_u64(r)?),
            ValueType::I64 => Self::I64(rd_i64(r)?),
            ValueType::F32 => Self::F32(rd_f32(r)?),
            ValueType::F64 => Self::F64(rd_f64(r)?),
            ValueType::Bool => match rd_u8(r)? {
                0 => Self::Bool(false),
                1 => Self::Bool(true),
                b => return Err(Error::Msg(format!("gguf: unexpected bool value {b}"))),
            },
            ValueType::String => Self::String(read_string(r, magic)?),
            ValueType::Array => {
                let vt = ValueType::from_u32(rd_u32(r)?)?;
                let len = match magic {
                    VersionedMagic::GgufV1 => rd_u32(r)? as usize,
                    VersionedMagic::GgufV2 | VersionedMagic::GgufV3 => rd_u64(r)? as usize,
                };
                let mut vs = Vec::with_capacity(len);
                for _ in 0..len {
                    vs.push(Value::read(r, vt, magic)?);
                }
                Self::Array(vs)
            }
        };
        Ok(v)
    }
}

// ============================================================================
// Content / TensorInfo(host 读取;设备面不存在)
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Magic {
    Gguf,
}

impl TryFrom<u32> for Magic {
    type Error = Error;
    fn try_from(value: u32) -> Result<Self> {
        match value {
            0x46554747 | 0x47475546 => Ok(Self::Gguf),
            _ => Err(Error::Msg(format!("gguf: unknown magic 0x{value:08x}"))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VersionedMagic {
    GgufV1,
    GgufV2,
    GgufV3,
}

impl VersionedMagic {
    fn read<R: Read>(r: &mut R) -> Result<Self> {
        let magic = Magic::try_from(rd_u32(r)?)?;
        let version = rd_u32(r)?;
        match (magic, version) {
            (Magic::Gguf, 1) => Ok(Self::GgufV1),
            (Magic::Gguf, 2) => Ok(Self::GgufV2),
            (Magic::Gguf, 3) => Ok(Self::GgufV3),
            _ => Err(Error::Msg(format!(
                "gguf: unsupported magic/version {magic:?}/{version}"
            ))),
        }
    }
}

fn read_string<R: Read>(r: &mut R, magic: &VersionedMagic) -> Result<String> {
    let len = match magic {
        VersionedMagic::GgufV1 => rd_u32(r)? as usize,
        VersionedMagic::GgufV2 | VersionedMagic::GgufV3 => rd_u64(r)? as usize,
    };
    let mut v = vec![0u8; len];
    r.read_exact(&mut v)
        .map_err(|e| Error::Msg(format!("gguf: string read: {e}")))?;
    while let Some(0) = v.last() {
        v.pop(); // GGUF 字符串规范不带 NUL,实践里有,容忍
    }
    Ok(String::from_utf8_lossy(&v).into_owned())
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub ggml_dtype: GgmlDType,
    pub shape: Vec<usize>,
    pub offset: u64,
}

impl TensorInfo {
    /// 原始块字节读出(布局 = GGML 块表;无任何设备语义)
    pub fn read_raw<R: Read + Seek>(
        &self,
        r: &mut R,
        tensor_data_offset: u64,
    ) -> Result<Vec<u8>> {
        let elems: usize = self.shape.iter().product();
        let block_size = self.ggml_dtype.block_size();
        if elems % block_size != 0 {
            return Err(Error::Msg(format!(
                "gguf: elements {elems} not divisible by block size {block_size}"
            )));
        }
        let size_in_bytes = elems / block_size * self.ggml_dtype.type_size();
        let mut raw = vec![0u8; size_in_bytes];
        r.seek(SeekFrom::Start(tensor_data_offset + self.offset))
            .map_err(|e| Error::Msg(format!("gguf: seek: {e}")))?;
        r.read_exact(&mut raw)
            .map_err(|e| Error::Msg(format!("gguf: tensor data read: {e}")))?;
        Ok(raw)
    }

    /// 分片原始读(块对齐数学 = candle `read_shard` 的 host 版;
    /// 非对齐返回 None = 该分片策略不支持此 dtype/shape 组合)
    pub fn read_shard_raw<R: Read + Seek>(
        &self,
        r: &mut R,
        tensor_data_offset: u64,
        dim: usize,
        rank: usize,
        world_size: usize,
    ) -> Result<Option<Vec<u8>>> {
        if world_size <= 1 {
            return self.read_raw(r, tensor_data_offset).map(Some);
        }
        let dims = &self.shape;
        if dim >= dims.len() {
            return Err(Error::Msg(format!(
                "gguf: cannot shard shape {dims:?} on dim {dim}"
            )));
        }
        if rank >= world_size {
            return Err(Error::Msg(format!(
                "gguf: rank {rank} must be < world_size {world_size}"
            )));
        }
        if dims[dim] % world_size != 0 {
            return Err(Error::Msg(format!(
                "gguf: cannot shard dim {dim} size {} into {world_size} parts",
                dims[dim]
            )));
        }
        let block_size = self.ggml_dtype.block_size();
        let type_size = self.ggml_dtype.type_size();
        let inner_elems: usize = dims[dim + 1..].iter().product();
        let outer_count: usize = dims[..dim].iter().product();
        let local_dim = dims[dim] / world_size;
        let start_dim = rank * local_dim;
        let stride_elems = dims[dim] * inner_elems;
        let segment_start = start_dim * inner_elems;
        let segment_elems = local_dim * inner_elems;
        if segment_start % block_size != 0 || segment_elems % block_size != 0 {
            return Ok(None);
        }
        let bytes_per_segment = segment_elems / block_size * type_size;
        let mut raw = vec![0u8; bytes_per_segment * outer_count];
        let base = tensor_data_offset + self.offset;
        for outer_idx in 0..outer_count {
            let elem_offset = outer_idx * stride_elems + segment_start;
            let byte_offset = elem_offset / block_size * type_size;
            r.seek(SeekFrom::Start(base + byte_offset as u64))
                .map_err(|e| Error::Msg(format!("gguf: shard seek: {e}")))?;
            let dst = &mut raw[outer_idx * bytes_per_segment..(outer_idx + 1) * bytes_per_segment];
            r.read_exact(dst)
                .map_err(|e| Error::Msg(format!("gguf: shard read: {e}")))?;
        }
        Ok(Some(raw))
    }
}

/// GGUF 文件内容(host 解析面;= candle `gguf_file::Content` 去设备化)
#[derive(Debug)]
pub struct Content {
    pub magic: VersionedMagic,
    pub metadata: HashMap<String, Value>,
    pub tensor_infos: HashMap<String, TensorInfo>,
    pub tensor_data_offset: u64,
}

impl Content {
    pub fn read<R: Read + Seek>(r: &mut R) -> Result<Self> {
        let magic = VersionedMagic::read(r)?;
        let tensor_count = match magic {
            VersionedMagic::GgufV1 => rd_u32(r)? as usize,
            VersionedMagic::GgufV2 | VersionedMagic::GgufV3 => rd_u64(r)? as usize,
        };
        let kv_count = match magic {
            VersionedMagic::GgufV1 => rd_u32(r)? as usize,
            VersionedMagic::GgufV2 | VersionedMagic::GgufV3 => rd_u64(r)? as usize,
        };
        let mut metadata = HashMap::new();
        for _ in 0..kv_count {
            let key = read_string(r, &magic)?;
            let vt = ValueType::from_u32(rd_u32(r)?)?;
            let value = Value::read(r, vt, &magic)?;
            metadata.insert(key, value);
        }
        let mut tensor_infos = HashMap::new();
        for _ in 0..tensor_count {
            let name = read_string(r, &magic)?;
            let n_dims = rd_u32(r)? as usize;
            let mut dims: Vec<usize> = match magic {
                VersionedMagic::GgufV1 => (0..n_dims).map(|_| rd_u32(r).map(|v| v as usize)).collect::<std::io::Result<Vec<_>>>()
                    .map_err(|e| Error::Msg(format!("gguf: dims: {e}")))?,
                VersionedMagic::GgufV2 | VersionedMagic::GgufV3 => (0..n_dims)
                    .map(|_| rd_u64(r).map(|v| v as usize))
                    .collect::<std::io::Result<Vec<_>>>()
                    .map_err(|e| Error::Msg(format!("gguf: dims: {e}")))?,
            };
            dims.reverse(); // GGUF 维序反转(candle 同)
            let ggml_dtype = GgmlDType::from_u32(rd_u32(r)?)?;
            let offset = rd_u64(r)?;
            tensor_infos.insert(name, TensorInfo { ggml_dtype, shape: dims, offset });
        }
        let position = r
            .stream_position()
            .map_err(|e| Error::Msg(format!("gguf: tell: {e}")))?;
        let alignment = match metadata.get("general.alignment") {
            Some(Value::U8(v)) => *v as u64,
            Some(Value::U16(v)) => *v as u64,
            Some(Value::U32(v)) => *v as u64,
            Some(Value::I8(v)) if *v >= 0 => *v as u64,
            Some(Value::I16(v)) if *v >= 0 => *v as u64,
            Some(Value::I32(v)) if *v >= 0 => *v as u64,
            _ => DEFAULT_ALIGNMENT,
        };
        let tensor_data_offset = position.div_ceil(alignment) * alignment;
        Ok(Self { magic, metadata, tensor_infos, tensor_data_offset })
    }

    pub fn tensor_raw<R: Read + Seek>(&self, r: &mut R, name: &str) -> Result<GgufRawTensor> {
        let info = self
            .tensor_infos
            .get(name)
            .ok_or_else(|| Error::Msg(format!("gguf: cannot find tensor info for {name}")))?;
        let raw = info.read_raw(r, self.tensor_data_offset)?;
        Ok(GgufRawTensor { dtype: info.ggml_dtype, shape: info.shape.clone(), raw })
    }

    pub fn tensor_shard_raw<R: Read + Seek>(
        &self,
        r: &mut R,
        name: &str,
        dim: usize,
        rank: usize,
        world_size: usize,
    ) -> Result<Option<GgufRawTensor>> {
        let info = self
            .tensor_infos
            .get(name)
            .ok_or_else(|| Error::Msg(format!("gguf: cannot find tensor info for {name}")))?;
        let dims = info.shape.clone();
        let raw = info.read_shard_raw(r, self.tensor_data_offset, dim, rank, world_size)?;
        Ok(raw.map(|raw| {
            let mut shard_dims = dims;
            if world_size > 1 {
                shard_dims[dim] /= world_size;
            }
            GgufRawTensor { dtype: info.ggml_dtype, shape: shard_dims, raw }
        }))
    }
}

/// 原始块张量(dtype + shape + 量化字节;反解见 [`dequantize_to_f32`])
#[derive(Debug, Clone)]
pub struct GgufRawTensor {
    pub dtype: GgmlDType,
    pub shape: Vec<usize>,
    pub raw: Vec<u8>,
}

// ============================================================================
// 反解引擎(11 dtype 真;IQ 系结构化报错)
// ============================================================================

/// 量化块字节 → f32(row-major;布局逐字节对齐 GGML 块表)。
/// 支持:F32/F16/BF16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q2K/Q3K/Q4K/Q5K/Q6K/Q8K;
/// IQ 系 = Err(IQ2S 网格表体量大,xinfer 生产面未见,运行里程碑按需回填)。
pub fn dequantize_to_f32(dtype: GgmlDType, shape: &[usize], bytes: &[u8]) -> Result<Vec<f32>> {
    let n: usize = shape.iter().product();
    let bs = dtype.block_size();
    if n % bs != 0 {
        return Err(Error::Msg(format!(
            "dequantize {dtype}: {n} not divisible by block {bs}"
        )));
    }
    let nb = n / bs;
    if bytes.len() != nb * dtype.type_size() {
        return Err(Error::Msg(format!(
            "dequantize {dtype}: bytes {} != blocks×type_size {}",
            bytes.len(),
            nb * dtype.type_size()
        )));
    }
    let mut ys = vec![0f32; n];
    match dtype {
        GgmlDType::F32 => {
            for (y, c) in ys.iter_mut().zip(bytes.chunks_exact(4)) {
                *y = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
        }
        GgmlDType::F16 => {
            for (y, c) in ys.iter_mut().zip(bytes.chunks_exact(2)) {
                *y = f16_bits_to_f32(u16::from_le_bytes([c[0], c[1]]));
            }
        }
        GgmlDType::BF16 => {
            for (y, c) in ys.iter_mut().zip(bytes.chunks_exact(2)) {
                *y = bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]]));
            }
        }
        GgmlDType::Q4_0 => {
            for (i, blk) in bytes.chunks_exact(18).enumerate() {
                let d = f16_bits_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                let qs = &blk[2..];
                for j in 0..16 {
                    let x0 = (qs[j] & 0x0f) as i32 - 8;
                    let x1 = (qs[j] >> 4) as i32 - 8;
                    ys[i * 32 + j] = x0 as f32 * d;
                    ys[i * 32 + j + 16] = x1 as f32 * d;
                }
            }
        }
        GgmlDType::Q4_1 => {
            for (i, blk) in bytes.chunks_exact(20).enumerate() {
                let d = f16_bits_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                let m = f16_bits_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
                let qs = &blk[4..];
                for j in 0..16 {
                    let x0 = (qs[j] & 0x0f) as f32;
                    let x1 = (qs[j] >> 4) as f32;
                    ys[i * 32 + j] = x0 * d + m;
                    ys[i * 32 + j + 16] = x1 * d + m;
                }
            }
        }
        GgmlDType::Q5_0 => {
            for (i, blk) in bytes.chunks_exact(22).enumerate() {
                let d = f16_bits_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                let qh = u32::from_le_bytes([blk[2], blk[3], blk[4], blk[5]]);
                let qs = &blk[6..];
                for j in 0..16 {
                    let xh_0 = (((qh >> j) << 4) & 0x10) as i32;
                    let xh_1 = ((qh >> (j + 12)) & 0x10) as i32;
                    let x0 = ((qs[j] & 0x0f) as i32 | xh_0) - 16;
                    let x1 = ((qs[j] >> 4) as i32 | xh_1) - 16;
                    ys[i * 32 + j] = x0 as f32 * d;
                    ys[i * 32 + j + 16] = x1 as f32 * d;
                }
            }
        }
        GgmlDType::Q5_1 => {
            for (i, blk) in bytes.chunks_exact(24).enumerate() {
                let d = f16_bits_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                let m = f16_bits_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
                let qh = u32::from_le_bytes([blk[4], blk[5], blk[6], blk[7]]);
                let qs = &blk[8..];
                for j in 0..16 {
                    let xh_0 = ((qh >> j) << 4) & 0x10;
                    let xh_1 = (qh >> (j + 12)) & 0x10;
                    let x0 = ((qs[j] & 0x0f) as i32) | xh_0 as i32;
                    let x1 = ((qs[j] >> 4) as i32) | xh_1 as i32;
                    ys[i * 32 + j] = x0 as f32 * d + m;
                    ys[i * 32 + j + 16] = x1 as f32 * d + m;
                }
            }
        }
        GgmlDType::Q8_0 => {
            for (i, blk) in bytes.chunks_exact(34).enumerate() {
                let d = f16_bits_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                for (j, q) in blk[2..].iter().enumerate() {
                    ys[i * 32 + j] = *q as i8 as f32 * d;
                }
            }
        }
        GgmlDType::Q2K => dequant_q2k(bytes, &mut ys)?,
        GgmlDType::Q3K => dequant_q3k(bytes, &mut ys)?,
        GgmlDType::Q4K => dequant_q4k(bytes, &mut ys)?,
        GgmlDType::Q5K => dequant_q5k(bytes, &mut ys)?,
        GgmlDType::Q6K => dequant_q6k(bytes, &mut ys)?,
        GgmlDType::Q8K => dequant_q8k(bytes, &mut ys)?,
        other => {
            return Err(Error::Msg(format!(
                "dequantize {other}: IQ 反解未回填(运行里程碑;块表已登记,偏移不受影响)"
            )));
        }
    }
    Ok(ys)
}

fn q2_k_scales<'a>(blk: &'a [u8]) -> (&'a [u8], f32, f32) {
    // Q2K 布局:scales[16] + qs[64] + d(2) + dmin(2)
    let d = f16_bits_to_f32(u16::from_le_bytes([blk[80], blk[81]]));
    let dmin = f16_bits_to_f32(u16::from_le_bytes([blk[82], blk[83]]));
    (&blk[..16], d, dmin)
}

fn dequant_q2k(bytes: &[u8], ys: &mut [f32]) -> Result<()> {
    for (bi, blk) in bytes.chunks_exact(84).enumerate() {
        let (scales, d, min) = q2_k_scales(blk);
        let qs = &blk[16..80];
        let mut is = 0usize;
        let base = bi * 256;
        for (y_block, qs32) in ys[base..base + 256]
            .chunks_exact_mut(128)
            .zip(qs.chunks_exact(32))
        {
            let mut shift = 0u32;
            let mut yi = 0usize;
            for _ in 0..4 {
                let sc = scales[is];
                is += 1;
                let dl = d * (sc & 0xF) as f32;
                let ml = min * (sc >> 4) as f32;
                for q in &qs32[..16] {
                    y_block[yi] = dl * ((q >> shift) & 3) as f32 - ml;
                    yi += 1;
                }
                let sc = scales[is];
                is += 1;
                let dl = d * (sc & 0xF) as f32;
                let ml = min * (sc >> 4) as f32;
                for q in &qs32[16..] {
                    y_block[yi] = dl * ((q >> shift) & 3) as f32 - ml;
                    yi += 1;
                }
                shift += 2;
            }
        }
    }
    Ok(())
}

fn dequant_q3k(bytes: &[u8], ys: &mut [f32]) -> Result<()> {
    const KMASK1: u32 = 0x03030303;
    const KMASK2: u32 = 0x0f0f0f0f;
    for (bi, blk) in bytes.chunks_exact(110).enumerate() {
        // 布局:hmask[32] + qs[64] + scales[12] + d(2)
        let hmask = &blk[..32];
        let qs = &blk[32..96];
        let scales_raw = &blk[96..108];
        let d = f16_bits_to_f32(u16::from_le_bytes([blk[108], blk[109]]));

        let mut aux = [0u32; 4];
        for (i, c) in scales_raw.chunks_exact(4).enumerate() {
            aux[i] = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        }
        let tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
        aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
        aux[0] = (aux[0] & KMASK2) | ((tmp & KMASK1) << 4);
        aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
        let scales_bytes: Vec<u8> = aux
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .take(16)
            .collect();
        let scales: Vec<i8> = scales_bytes.iter().map(|&b| b as i8).collect();

        let d_all = d;
        let mut m = 1u8;
        let mut is = 0usize;
        let base = bi * 256;
        for (y128, qs32) in ys[base..base + 256]
            .chunks_exact_mut(128)
            .zip(qs.chunks_exact(32))
        {
            let mut shift = 0u32;
            for y32 in y128.chunks_exact_mut(32) {
                for (scale_index, y16) in y32.chunks_exact_mut(16).enumerate() {
                    let dl = d_all * (scales[is] as f32 - 32.0);
                    for (i, yi) in y16.iter_mut().enumerate() {
                        let sub = if (hmask[i + 16 * scale_index] & m) == 0 { 4 } else { 0 };
                        *yi = dl * ((((qs32[i + 16 * scale_index] >> shift) & 3) as i8 - sub) as f32);
                    }
                    is += 1;
                }
                shift += 2;
                m <<= 1;
            }
        }
    }
    Ok(())
}

fn dequant_q4k(bytes: &[u8], ys: &mut [f32]) -> Result<()> {
    // 布局:d(2) + dmin(2) + scales[12] + qs[128]
    for (bi, blk) in bytes.chunks_exact(144).enumerate() {
        let d = f16_bits_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let min = f16_bits_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
        let scales12 = &blk[4..16];
        let qs = &blk[16..];
        let mut is = 0usize;
        let mut yi = bi * 256;
        for j in (0..256).step_by(64) {
            let q = &qs[j / 2..j / 2 + 32];
            let (sc, m) = get_scale_min_k4(is, scales12);
            let (d1, m1) = (d * sc as f32, min * m as f32);
            let (sc, m) = get_scale_min_k4(is + 1, scales12);
            let (d2, m2) = (d * sc as f32, min * m as f32);
            for q in q {
                ys[yi] = d1 * (q & 0xF) as f32 - m1;
                yi += 1;
            }
            for q in q {
                ys[yi] = d2 * (q >> 4) as f32 - m2;
                yi += 1;
            }
            is += 2;
        }
    }
    Ok(())
}

fn dequant_q5k(bytes: &[u8], ys: &mut [f32]) -> Result<()> {
    // 布局:d(2) + dmin(2) + scales[12] + qh[32] + qs[128]
    for (bi, blk) in bytes.chunks_exact(176).enumerate() {
        let d = f16_bits_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let min = f16_bits_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
        let scales12 = &blk[4..16];
        let qh = &blk[16..48];
        let qs = &blk[48..];
        let mut is = 0usize;
        let mut u1 = 1u8;
        let mut u2 = 2u8;
        let mut yi = bi * 256;
        for j in (0..256).step_by(64) {
            let ql = &qs[j / 2..j / 2 + 32];
            let (sc, m) = get_scale_min_k4(is, scales12);
            let (d1, m1) = (d * sc as f32, min * m as f32);
            let (sc, m) = get_scale_min_k4(is + 1, scales12);
            let (d2, m2) = (d * sc as f32, min * m as f32);
            for (ql, qh) in ql.iter().zip(qh) {
                let to_add = if qh & u1 != 0 { 16f32 } else { 0f32 };
                ys[yi] = d1 * ((ql & 0xF) as f32 + to_add) - m1;
                yi += 1;
            }
            for (ql, qh) in ql.iter().zip(qh) {
                let to_add = if qh & u2 != 0 { 16f32 } else { 0f32 };
                ys[yi] = d2 * ((ql >> 4) as f32 + to_add) - m2;
                yi += 1;
            }
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    Ok(())
}

fn dequant_q6k(bytes: &[u8], ys: &mut [f32]) -> Result<()> {
    // 布局:ql[128] + qh[64] + scales[16](i8)+ d(2)
    for (bi, blk) in bytes.chunks_exact(210).enumerate() {
        let ql = &blk[..128];
        let qh = &blk[128..192];
        let sc = &blk[192..208];
        let d = f16_bits_to_f32(u16::from_le_bytes([blk[208], blk[209]]));
        let out = &mut ys[bi * 256..(bi + 1) * 256];
        for n in (0..256).step_by(128) {
            let idx = n / 128;
            let ys = &mut out[n..];
            let sc = &sc[8 * idx..];
            let ql = &ql[64 * idx..];
            let qh = &qh[32 * idx..];
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i8 - 32;
                let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i8 - 32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 - 32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 - 32;
                ys[l] = d * sc[is] as f32 * q1 as f32;
                ys[l + 32] = d * sc[is + 2] as f32 * q2 as f32;
                ys[l + 64] = d * sc[is + 4] as f32 * q3 as f32;
                ys[l + 96] = d * sc[is + 6] as f32 * q4 as f32;
            }
        }
    }
    Ok(())
}

fn dequant_q8k(bytes: &[u8], ys: &mut [f32]) -> Result<()> {
    // 布局:d(f32) + qs[256](i8)+ bsums[16](i16)
    for (bi, blk) in bytes.chunks_exact(292).enumerate() {
        let d = f32::from_le_bytes([blk[0], blk[1], blk[2], blk[3]]);
        for (j, &q) in blk[4..260].iter().enumerate() {
            ys[bi * 256 + j] = d * q as i8 as f32;
        }
    }
    Ok(())
}

/// K4 系 scale/min 提取(= candle `get_scale_min_k4`)
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

// ============================================================================
// GGufVarBuilder(= xinfer gguf_varbuilder::VarBuilder 去设备化)
// ============================================================================

struct GgufShard {
    content: Content,
    file: std::fs::File,
}

/// GGUF 多分片变量构建器(路径拼接/分片路由/单张量缓存语义与
/// xinfer 一致;产物从 `Arc<QTensor>` 换为 [`GgufRawTensor`])。
#[derive(Clone)]
pub struct GGufVarBuilder {
    shards: std::sync::Arc<std::sync::Mutex<Vec<GgufShard>>>,
    tensor_to_shard: std::sync::Arc<HashMap<String, usize>>,
    cache: std::sync::Arc<std::sync::Mutex<Option<(String, std::sync::Arc<GgufRawTensor>)>>>,
    path: Vec<String>,
    file_path: std::sync::Arc<String>,
}

impl GGufVarBuilder {
    pub fn from_gguf<P: AsRef<std::path::Path>>(p: P) -> Result<Self> {
        Self::from_gguf_files(&[p.as_ref().to_path_buf()])
    }

    pub fn from_gguf_files(paths: &[std::path::PathBuf]) -> Result<Self> {
        assert!(!paths.is_empty(), "no gguf files provided");
        let file_path = paths[0].to_string_lossy().to_string();
        let mut shards = Vec::with_capacity(paths.len());
        let mut tensor_to_shard = HashMap::new();
        for (shard_idx, path) in paths.iter().enumerate() {
            let mut file = std::fs::File::open(path)
                .map_err(|e| Error::Msg(format!("gguf: open {}: {e}", path.display())))?;
            let content = Content::read(&mut file)?;
            for name in content.tensor_infos.keys() {
                tensor_to_shard.insert(name.clone(), shard_idx);
            }
            shards.push(GgufShard { content, file });
        }
        Ok(Self {
            shards: std::sync::Arc::new(std::sync::Mutex::new(shards)),
            tensor_to_shard: std::sync::Arc::new(tensor_to_shard),
            cache: std::sync::Arc::new(std::sync::Mutex::new(None)),
            path: Vec::new(),
            file_path: std::sync::Arc::new(file_path),
        })
    }

    pub fn gguf_path(&self) -> &str {
        &self.file_path
    }

    /// 前缀下钻(点分路径;语义与 xinfer/candle 一致)
    pub fn pp<S: ToString>(&self, s: S) -> Self {
        let mut path = self.path.clone();
        path.push(s.to_string());
        Self {
            shards: self.shards.clone(),
            tensor_to_shard: self.tensor_to_shard.clone(),
            cache: self.cache.clone(),
            path,
            file_path: self.file_path.clone(),
        }
    }

    /// 张量全名(前缀 + name)
    pub fn path(&self, tensor_name: &str) -> String {
        if self.path.is_empty() {
            tensor_name.to_string()
        } else {
            [&self.path.join("."), tensor_name].join(".")
        }
    }

    fn resolve_shard(&self, tensor_path: &str) -> Result<usize> {
        self.tensor_to_shard
            .get(tensor_path)
            .copied()
            .ok_or_else(|| {
                Error::Msg(format!("cannot find tensor {tensor_path} in any gguf shard"))
            })
    }

    /// 原始块张量(带 shape 校验 + 单张量缓存)
    pub fn get(&self, shape: &[usize], name: &str) -> Result<std::sync::Arc<GgufRawTensor>> {
        let path = self.path(name);
        {
            let cache = self.cache.lock().expect("gguf cache 中毒");
            if let Some((cached_name, cached)) = cache.as_ref() {
                if cached_name == &path {
                    if &cached.shape != shape {
                        return Err(Error::Msg(format!(
                            "shape mismatch for {name}, got {:?}, expected {shape:?}",
                            cached.shape
                        )));
                    }
                    return Ok(cached.clone());
                }
            }
        }
        let shard_idx = self.resolve_shard(&path)?;
        let mut shards = self.shards.lock().expect("gguf shards 中毒");
        let shard = &mut shards[shard_idx];
        let tensor = std::sync::Arc::new(shard.content.tensor_raw(&mut shard.file, &path)?);
        *self.cache.lock().expect("gguf cache 中毒") = Some((path.clone(), tensor.clone()));
        if &tensor.shape != shape {
            return Err(Error::Msg(format!(
                "shape mismatch for {name}, got {:?}, expected {shape:?}",
                tensor.shape
            )));
        }
        Ok(tensor)
    }

    /// 分片读取(world_size>1 时按 dim 切;非对齐返回 None)
    pub fn get_sharded(
        &self,
        shape: &[usize],
        name: &str,
        dim: usize,
        rank: usize,
        world_size: usize,
    ) -> Result<Option<std::sync::Arc<GgufRawTensor>>> {
        if world_size <= 1 {
            return self.get(shape, name).map(Some);
        }
        let path = self.path(name);
        if dim >= shape.len() {
            return Err(Error::Msg(format!(
                "cannot shard tensor {path} with shape {shape:?} on dim {dim}"
            )));
        }
        if shape[dim] % world_size != 0 {
            return Err(Error::Msg(format!(
                "cannot shard tensor {path} dim {dim} size {} into {world_size} parts",
                shape[dim]
            )));
        }
        let mut shard_shape = shape.to_vec();
        shard_shape[dim] /= world_size;
        let shard_idx = self.resolve_shard(&path)?;
        let mut shards = self.shards.lock().expect("gguf shards 中毒");
        let shard = &mut shards[shard_idx];
        let Some(t) =
            shard.content.tensor_shard_raw(&mut shard.file, &path, dim, rank, world_size)?
        else {
            return Ok(None);
        };
        if t.shape != shard_shape {
            return Err(Error::Msg(format!(
                "shape mismatch for sharded {name}, got {:?}, expected {shard_shape:?}",
                t.shape
            )));
        }
        Ok(Some(std::sync::Arc::new(t)))
    }

    pub fn get_no_shape(&self, name: &str) -> Result<std::sync::Arc<GgufRawTensor>> {
        let path = self.path(name);
        let shard_idx = self.resolve_shard(&path)?;
        let mut shards = self.shards.lock().expect("gguf shards 中毒");
        let shard = &mut shards[shard_idx];
        Ok(std::sync::Arc::new(
            shard.content.tensor_raw(&mut shard.file, &path)?,
        ))
    }

    /// 反解 + 落 allocator(F32 落池/落内存;loader 不直触设备)
    pub fn materialize_f32(
        &self,
        shape: &[usize],
        name: &str,
        alloc: &dyn super::alloc::WeightAllocator,
    ) -> Result<super::alloc::HostTensor> {
        let raw = self.get(shape, name)?;
        let f32_bytes: Vec<u8> = dequantize_to_f32(raw.dtype, &raw.shape, &raw.raw)?
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        alloc.materialize(shape, Dtype::F32, &f32_bytes)
    }

    pub fn clear_cache(&self) {
        *self.cache.lock().expect("gguf cache 中毒") = None;
    }

    pub fn contains_key(&self, key: &str) -> bool {
        self.tensor_to_shard.contains_key(&self.path(key))
    }

    pub fn tensor_shape(&self, key: &str) -> Option<Vec<usize>> {
        let path = self.path(key);
        let shard_idx = *self.tensor_to_shard.get(&path)?;
        let shards = self.shards.lock().expect("gguf shards 中毒");
        shards[shard_idx]
            .content
            .tensor_infos
            .get(&path)
            .map(|info| info.shape.clone())
    }

    pub fn tensor_dtype(&self, key: &str) -> Option<GgmlDType> {
        let path = self.path(key);
        let shard_idx = *self.tensor_to_shard.get(&path)?;
        let shards = self.shards.lock().expect("gguf shards 中毒");
        shards[shard_idx]
            .content
            .tensor_infos
            .get(&path)
            .map(|info| info.ggml_dtype)
    }

    pub fn first_content_metadata(&self) -> HashMap<String, Value> {
        let shards = self.shards.lock().expect("gguf shards 中毒");
        shards[0].content.metadata.clone()
    }
}

// ============================================================================
// 测试:half 转换 + HostWeightAllocator 往返 + 合成 GGUF 全链
// ============================================================================

#[cfg(test)]
mod tests {
    use super::super::alloc::{HostWeightAllocator, WeightAllocator};
    use super::*;

    #[test]
    fn f16_bits_roundtrip_known_values() {
        // 0x3C00 = 1.0;0x4000 = 2.0;0xD800 = -512;0x3580 ≈ 0.84375...用 1/3 的标准近似
        assert_eq!(f16_bits_to_f32(0x3C00), 1.0);
        assert_eq!(f16_bits_to_f32(0x4000), 2.0);
        assert_eq!(f16_bits_to_f32(0x0000), 0.0);
        assert_eq!(f16_bits_to_f32(0x8000), -0.0);
        assert!(f16_bits_to_f32(0x7C00).is_infinite());
        // 0x4900 = 10.0
        assert_eq!(f16_bits_to_f32(0x4900), 10.0);
    }

    #[test]
    fn bf16_bits_roundtrip_known_values() {
        assert_eq!(bf16_bits_to_f32(0x3F80), 1.0);
        assert_eq!(bf16_bits_to_f32(0x4000), 2.0);
        assert_eq!(bf16_bits_to_f32(0x4120), 10.0);
    }

    #[test]
    fn host_allocator_roundtrip() {
        let alloc = HostWeightAllocator;
        let mut buf = alloc.alloc(&[2, 3], Dtype::F32).unwrap();
        assert_eq!(buf.data.len(), 24);
        let vals: Vec<u8> = [1.0f32, -2.5, 3.25, 0.0, 7.0, -0.5]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        alloc.upload(&mut buf, &vals).unwrap();
        buf.assert_layout().unwrap();
        assert_eq!(buf.data, vals);
        // 长度不符 = 结构化报错
        assert!(alloc.upload(&mut buf, &[0u8; 4]).is_err());
    }

    /// 构造最小 GGUF(V3):1 个 F32 张量 [2,2] + 1 个 Q8_0 张量 [32]
    fn synthetic_gguf() -> Vec<u8> {
        let mut w: Vec<u8> = Vec::new();
        // magic "GGUF" LE = 0x46554747,version 3
        w.extend_from_slice(&0x46554747u32.to_le_bytes());
        w.extend_from_slice(&3u32.to_le_bytes());
        w.extend_from_slice(&2u64.to_le_bytes()); // tensor count
        w.extend_from_slice(&1u64.to_le_bytes()); // kv count
        // kv: general.alignment = U32 32
        w.extend_from_slice(&17u64.to_le_bytes()); // key len
        w.extend_from_slice(b"general.alignment");
        w.extend_from_slice(&4u32.to_le_bytes()); // U32
        w.extend_from_slice(&32u32.to_le_bytes());
        // tensor info 1: t_f32 [2,2](GGUF 原始维序 [2,2] 反转后一致)
        let name1 = b"t_f32";
        w.extend_from_slice(&(name1.len() as u64).to_le_bytes());
        w.extend_from_slice(name1);
        w.extend_from_slice(&2u32.to_le_bytes()); // n_dims
        w.extend_from_slice(&2u64.to_le_bytes());
        w.extend_from_slice(&2u64.to_le_bytes());
        w.extend_from_slice(&0u32.to_le_bytes()); // F32
        w.extend_from_slice(&0u64.to_le_bytes()); // offset
        // tensor info 2: t_q8_0 [32]
        let name2 = b"t_q8_0";
        w.extend_from_slice(&(name2.len() as u64).to_le_bytes());
        w.extend_from_slice(name2);
        w.extend_from_slice(&1u32.to_le_bytes());
        w.extend_from_slice(&32u64.to_le_bytes());
        w.extend_from_slice(&8u32.to_le_bytes()); // Q8_0
        w.extend_from_slice(&16u64.to_le_bytes()); // offset(header 后对齐 32)
        // 对齐到 32
        while w.len() % 32 != 0 {
            w.push(0);
        }
        // data: t_f32 = [1,2,3,4]
        for f in [1.0f32, 2.0, 3.0, 4.0] {
            w.extend_from_slice(&f.to_le_bytes());
        }
        // data: t_q8_0 = d=f16(2.0) + qs=[1..32]
        w.extend_from_slice(&0x4000u16.to_le_bytes()); // f16 2.0
        for i in 0..32u8 {
            w.push(i);
        }
        w
    }

    #[test]
    fn synthetic_gguf_parse_dequant_and_materialize() {
        let dir = std::env::temp_dir().join(format!("owl-gguf-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.gguf");
        std::fs::write(&path, synthetic_gguf()).unwrap();

        let vb = GGufVarBuilder::from_gguf(&path).unwrap();
        assert!(vb.contains_key("t_f32"));
        assert!(!vb.contains_key("missing"));
        assert_eq!(vb.tensor_shape("t_f32"), Some(vec![2, 2]));
        assert_eq!(vb.tensor_dtype("t_q8_0"), Some(GgmlDType::Q8_0));

        // F32 原始读
        let raw = vb.get_no_shape("t_f32").unwrap();
        assert_eq!(raw.dtype, GgmlDType::F32);
        let expect_bytes: Vec<u8> = [1.0f32, 2.0f32, 3.0f32, 4.0f32]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        assert_eq!(raw.raw, expect_bytes);

        // Q8_0 反解:d=2.0,qs=0..31 → [0,2,4,...,62]
        let q = vb.get_no_shape("t_q8_0").unwrap();
        let ys = dequantize_to_f32(q.dtype, &q.shape, &q.raw).unwrap();
        assert_eq!(ys.len(), 32);
        assert_eq!(ys[0], 0.0);
        assert_eq!(ys[31], 62.0);

        // 走 allocator 全链(host)
        let alloc = HostWeightAllocator;
        let ht = vb.materialize_f32(&[2, 2], "t_f32", &alloc).unwrap();
        ht.assert_layout().unwrap();
        let f: Vec<f32> = ht
            .data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(f, vec![1.0, 2.0, 3.0, 4.0]);

        // pp 路径拼接
        assert_eq!(vb.pp("blk").path("attn_q"), "blk.attn_q");
        assert_eq!(vb.path("x"), "x");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn sharded_read_even_split() {
        let dir = std::env::temp_dir().join(format!("owl-gguf-shard-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.gguf");
        // 单个 F32 [4] 张量,分 2 片 → 各 [2]
        let mut w: Vec<u8> = Vec::new();
        w.extend_from_slice(&0x46554747u32.to_le_bytes());
        w.extend_from_slice(&3u32.to_le_bytes());
        w.extend_from_slice(&1u64.to_le_bytes());
        w.extend_from_slice(&0u64.to_le_bytes());
        let name = b"t";
        w.extend_from_slice(&(name.len() as u64).to_le_bytes());
        w.extend_from_slice(name);
        w.extend_from_slice(&1u32.to_le_bytes());
        w.extend_from_slice(&4u64.to_le_bytes());
        w.extend_from_slice(&0u32.to_le_bytes());
        w.extend_from_slice(&0u64.to_le_bytes());
        while w.len() % 32 != 0 {
            w.push(0);
        }
        for f in [1.0f32, 2.0, 3.0, 4.0] {
            w.extend_from_slice(&f.to_le_bytes());
        }
        std::fs::write(&path, &w).unwrap();

        let vb = GGufVarBuilder::from_gguf(&path).unwrap();
        let s0 = vb.get_sharded(&[4], "t", 0, 0, 2).unwrap().unwrap();
        let s1 = vb.get_sharded(&[4], "t", 0, 1, 2).unwrap().unwrap();
        let v0: Vec<f32> = s0
            .raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let v1: Vec<f32> = s1
            .raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(v0, vec![1.0, 2.0]);
        assert_eq!(v1, vec![3.0, 4.0]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}

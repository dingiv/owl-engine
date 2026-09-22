//! HF safetensors 装载通道(阶段二线 A)。
//!
//! 格式(spec: huggingface/safetensors):
//! 8 字节 LE u64 = header 长度 N → N 字节 JSON header(UTF-8)→ 张量原始数据
//! (little-endian,自文件头起 8+N 偏移)。header 条目:
//! `"name": {"dtype": "BF16", "shape": [d0,..], "data_offsets": [begin, end]}`
//! (`data_offsets` 相对数据区起点;`__metadata__` 为字符串 map,忽略)。
//!
//! 职责边界与 gguf.rs 一致:只产 host 字节/懒转换 f32,分配交给
//! [`crate::loader::WeightAllocator`](A4:禁止内部触 CUDA)。
//!
//! dtype:HF 全枚举;数值转换 BF16/F16→F32 用扩位(F16 走 RNE 查表语义,
//! BF16 为 u16 高位直接扩),F64→F32 截断为 `as`,整型/BOOL 按值 cast。

use crate::error::{Error, Result};
use serde_json::Value;
use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

/// safetensors dtype 全枚举(HF 规范)。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafeDtype {
    Bool,
    U8,
    I8,
    I16,
    I32,
    I64,
    U16,
    U32,
    U64,
    F16,
    Bf16,
    F32,
    F64,
}

impl SafeDtype {
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "BOOL" => Ok(Self::Bool),
            "U8" => Ok(Self::U8),
            "I8" => Ok(Self::I8),
            "I16" => Ok(Self::I16),
            "I32" => Ok(Self::I32),
            "I64" => Ok(Self::I64),
            "U16" => Ok(Self::U16),
            "U32" => Ok(Self::U32),
            "U64" => Ok(Self::U64),
            "F16" => Ok(Self::F16),
            "BF16" => Ok(Self::Bf16),
            "F32" => Ok(Self::F32),
            "F64" => Ok(Self::F64),
            v => Err(Error::Msg(format!("safetensors: 未知 dtype {v}"))),
        }
    }

    /// 元素字节宽度(布局 = 逐元素小端,无打包/分块)。
    pub fn elem_size(self) -> usize {
        match self {
            Self::Bool | Self::U8 | Self::I8 => 1,
            Self::I16 | Self::U16 | Self::F16 | Self::Bf16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64 | Self::U64 | Self::F64 => 8,
        }
    }
}

/// 单张量元信息(header 投影;offsets 相对数据区)。
#[derive(Debug, Clone)]
pub struct SafeTensorInfo {
    pub dtype: SafeDtype,
    pub shape: Vec<usize>,
    /// 数据区起点(字节)
    pub begin: usize,
    /// 数据区终点(字节,含检查 begin + elems*elem_size == end)
    pub end: usize,
}

impl SafeTensorInfo {
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }
    pub fn byte_len(&self) -> usize {
        self.end - self.begin
    }
}

/// 已打开的 safetensors 文件:一次性读入(0.8B 单文件 1.7G 可接受;
/// 不引 memmap2 依赖,延迟转换由 [`Self::tensor_f32`] 懒做)。
pub struct SafeTensorsFile {
    /// 原始文件字节([0..8+N) = 头,[8+N..] = 数据区)
    data: Vec<u8>,
    /// 数据区在 `data` 中的起始偏移 = 8 + header_len
    data_offset: usize,
    tensors: HashMap<String, SafeTensorInfo>,
}

impl SafeTensorsFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let mut data = Vec::new();
        std::fs::File::open(path)
            .and_then(|mut f| f.read_to_end(&mut data))
            .map_err(|e| Error::Msg(format!("safetensors: 打开 {} 失败: {e}", path.display())))?;
        Self::from_bytes(data)
    }

    /// 从完整文件字节构造(测试/内存源可复用)。
    pub fn from_bytes(data: Vec<u8>) -> Result<Self> {
        if data.len() < 8 {
            return Err(Error::Msg(format!(
                "safetensors: 文件过短({} 字节,连 header 长度都没有)",
                data.len()
            )));
        }
        let header_len = u64::from_le_bytes(data[..8].try_into().unwrap()) as usize;
        let data_offset = 8usize
            .checked_add(header_len)
            .ok_or_else(|| Error::Msg("safetensors: header 长度溢出".into()))?;
        if data.len() < data_offset {
            return Err(Error::Msg(format!(
                "safetensors: header 声明 {header_len} 字节,文件只剩 {}",
                data.len() - 8
            )));
        }
        let header: HashMap<String, Value> = serde_json::from_slice(&data[8..data_offset])
            .map_err(|e| Error::Msg(format!("safetensors: header JSON 解析失败: {e}")))?;

        let mut tensors = HashMap::new();
        for (name, entry) in header {
            if name == "__metadata__" {
                continue;
            }
            let dtype_s = entry
                .get("dtype")
                .and_then(Value::as_str)
                .ok_or_else(|| Error::Msg(format!("safetensors: {name} 缺 dtype")))?;
            let dtype = SafeDtype::from_str(dtype_s)?;
            let shape = entry
                .get("shape")
                .and_then(Value::as_array)
                .ok_or_else(|| Error::Msg(format!("safetensors: {name} 缺 shape")))?;
            let shape: Vec<usize> = shape
                .iter()
                .map(|d| {
                    d.as_u64()
                        .map(|d| d as usize)
                        .ok_or_else(|| Error::Msg(format!("safetensors: {name} shape 非法: {d}")))
                })
                .collect::<Result<_>>()?;
            let offs = entry
                .get("data_offsets")
                .and_then(Value::as_array)
                .ok_or_else(|| Error::Msg(format!("safetensors: {name} 缺 data_offsets")))?;
            if offs.len() != 2 {
                return Err(Error::Msg(format!(
                    "safetensors: {name} data_offsets 需要 [begin, end],得到 {} 项",
                    offs.len()
                )));
            }
            let mut off = [0usize; 2];
            for (i, v) in offs.iter().enumerate() {
                off[i] = v
                    .as_u64()
                    .map(|v| v as usize)
                    .ok_or_else(|| Error::Msg(format!("safetensors: {name} offset 非法: {v}")))?;
            }
            let (begin, end) = (off[0], off[1]);
            let info = SafeTensorInfo {
                dtype,
                shape,
                begin,
                end,
            };
            if end < begin {
                return Err(Error::Msg(format!(
                    "safetensors: {name} offsets 倒挂 [{begin},{end})"
                )));
            }
            let expect = info
                .num_elements()
                .checked_mul(dtype.elem_size())
                .ok_or_else(|| Error::Msg(format!("safetensors: {name} 字节数溢出")))?;
            if expect != info.byte_len() {
                return Err(Error::Msg(format!(
                    "safetensors: {name} 字节数不符: shape 声明 {expect},offsets 给 {}",
                    info.byte_len()
                )));
            }
            if data_offset
                .checked_add(end)
                .is_none_or(|abs| abs > data.len())
            {
                return Err(Error::Msg(format!(
                    "safetensors: {name} 数据区越界(end={end},数据区只剩 {})",
                    data.len() - data_offset
                )));
            }
            tensors.insert(name, info);
        }
        Ok(Self {
            data,
            data_offset,
            tensors,
        })
    }

    /// 张量数(不含 __metadata__)。
    pub fn len(&self) -> usize {
        self.tensors.len()
    }
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
    /// 全部张量名(无序)。
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }
    pub fn info(&self, name: &str) -> Result<&SafeTensorInfo> {
        self.tensors
            .get(name)
            .ok_or_else(|| Error::Msg(format!("safetensors: 无此张量 {name}")))
    }

    /// 原始字节视图(小端;切片有效性已在 open 时整体校验)。
    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let info = self.info(name)?;
        let s = self.data_offset + info.begin;
        Ok(&self.data[s..s + info.byte_len()])
    }

    /// 懒转 f32(LE 语义):F32 拷贝;BF16 高位扩;F16 解码(RNE);F64 截断;
    /// 整型按值 cast;BOOL → 0.0/1.0。
    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        let info = self.info(name)?;
        let bytes = self.tensor_bytes(name)?;
        let n = info.num_elements();
        let mut out = Vec::with_capacity(n);
        match info.dtype {
            SafeDtype::F32 => {
                out.extend(bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())));
            }
            SafeDtype::Bf16 => {
                out.extend(bytes.chunks_exact(2).map(|c| bf16_bits_to_f32(u16::from_le_bytes(c.try_into().unwrap()))));
            }
            SafeDtype::F16 => {
                out.extend(bytes.chunks_exact(2).map(|c| f16_bits_to_f32(u16::from_le_bytes(c.try_into().unwrap()))));
            }
            SafeDtype::F64 => {
                out.extend(bytes.chunks_exact(8).map(|c| (f64::from_le_bytes(c.try_into().unwrap())) as f32));
            }
            SafeDtype::I8 => out.extend(bytes.iter().map(|&b| i8::from_le_bytes([b]) as f32)),
            SafeDtype::U8 | SafeDtype::Bool => out.extend(bytes.iter().map(|&b| b as f32)),
            SafeDtype::I16 => out.extend(bytes.chunks_exact(2).map(|c| i16::from_le_bytes(c.try_into().unwrap()) as f32)),
            SafeDtype::U16 => out.extend(bytes.chunks_exact(2).map(|c| u16::from_le_bytes(c.try_into().unwrap()) as f32)),
            SafeDtype::I32 => out.extend(bytes.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f32)),
            SafeDtype::U32 => out.extend(bytes.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap()) as f32)),
            SafeDtype::I64 => out.extend(bytes.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f32)),
            SafeDtype::U64 => out.extend(bytes.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap()) as f32)),
        }
        debug_assert_eq!(out.len(), n);
        Ok(out)
    }
}

/// BF16 → F32:u16 直接放高 16 位,低 16 位补零(IEEE 语义无损)。
fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// F16(IEEE binary16)→ F32:手写解码(符号/指数/尾数;RNE 无需——
/// 单向展宽无舍入;含 inf/nan/非规格化)。
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;
    let f32_bits = match (exp, frac) {
        (0, 0) => sign,                                  // ±0
        (0, _) => sign + frac_shift(frac),               // 非规格化
        (0x1f, 0) => sign | 0x7f80_0000,                 // ±inf
        (0x1f, _) => sign | 0x7f80_0000 | (frac << 13),  // nan(载荷保留)
        _ => sign | ((exp + 112) << 23) | (frac << 13),  // 常规:指数字段重置
    };
    f32::from_bits(f32_bits)
}

/// F16 非规格化 → F32 常规位型(frac 逐位找首位)。
fn frac_shift(frac: u32) -> u32 {
    debug_assert!(frac != 0);
    let mut f = frac;
    let mut e = 113u32; // 起始指数字段(非规格化对应 -14 → f32 指数位 109 起)
    loop {
        f <<= 1;
        e -= 1;
        if f & 0x400 != 0 {
            break;
        }
    }
    (e << 23) | ((f & 0x3ff) << 13)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    /// 真模型文件(本地工作区;缺失时跳过并提示——CI 无此文件)。
    const MODEL: &str =
        "/home/div/Documents/codes/models/Qwen/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors";
    const INDEX: &str =
        "/home/div/Documents/codes/models/Qwen/Qwen3.5-0.8B/model.safetensors.index.json";

    fn model_path() -> Option<&'static str> {
        if Path::new(MODEL).exists() {
            Some(MODEL)
        } else {
            eprintln!("skip: 真模型文件不在本机({MODEL})");
            None
        }
    }

    #[test]
    fn bf16_f16_bits() {
        assert_eq!(bf16_bits_to_f32(0x3f80), 1.0);
        assert_eq!(bf16_bits_to_f32(0x4000), 2.0);
        assert_eq!(bf16_bits_to_f32(0xbf80), -1.0);
        assert_eq!(f16_bits_to_f32(0x3c00), 1.0);
        assert_eq!(f16_bits_to_f32(0x4200), 3.0);
        assert!(f16_bits_to_f32(0x7c00).is_infinite());
        assert!(f16_bits_to_f32(0x7e00).is_nan());
        // F16 非规格化:0x0001 = 2^-24
        assert_eq!(f16_bits_to_f32(0x0001), 2f32.powi(-24));
    }

    #[test]
    fn synthetic_roundtrip() {
        // 手工拼一个小文件:2 张量(F32/BF16)+ __metadata__
        let mut hdr = serde_json::map::Map::new();
        hdr.insert("__metadata__".into(), serde_json::json!({"format": "pt"}));
        hdr.insert(
            "a".into(),
            serde_json::json!({"dtype": "F32", "shape": [3], "data_offsets": [0, 12]}),
        );
        hdr.insert(
            "b".into(),
            serde_json::json!({"dtype": "BF16", "shape": [2, 2], "data_offsets": [12, 20]}),
        );
        let hdr_json = serde_json::to_string(&Value::Object(hdr)).unwrap();
        let mut data = Vec::new();
        data.extend_from_slice(&(hdr_json.len() as u64).to_le_bytes());
        data.extend_from_slice(hdr_json.as_bytes());
        data.extend_from_slice(&{
            let mut b = Vec::new();
            b.extend_from_slice(&1.0f32.to_le_bytes());
            b.extend_from_slice(&2.5f32.to_le_bytes());
            b.extend_from_slice(&(-4.0f32).to_le_bytes());
            b
        });
        data.extend_from_slice(&[0x3f80u16.to_le_bytes(), 0x4000u16.to_le_bytes(), 0x0000u16.to_le_bytes(), 0xbf80u16.to_le_bytes()].concat());

        let f = SafeTensorsFile::from_bytes(data).unwrap();
        assert_eq!(f.len(), 2);
        assert_eq!(f.tensor_f32("a").unwrap(), vec![1.0, 2.5, -4.0]);
        assert_eq!(f.tensor_f32("b").unwrap(), vec![1.0, 2.0, 0.0, -1.0]);
        let ib = f.info("b").unwrap();
        assert_eq!(ib.shape, vec![2, 2]);
        assert_eq!(ib.dtype, SafeDtype::Bf16);
        // tensor_bytes 透传
        assert_eq!(f.tensor_bytes("a").unwrap().len(), 12);
        assert!(f.tensor_f32("c").is_err());
    }

    /// 真文件:张量总数 == index weight_map 条目数;数据区无缝全覆盖。
    #[test]
    fn real_file_counts_and_contiguity() {
        let Some(p) = model_path() else { return };
        let f = SafeTensorsFile::open(p).unwrap();
        let index = std::fs::read_to_string(INDEX).unwrap();
        let weight_map: HashMap<String, String> =
            serde_json::from_str(serde_json::from_str::<Value>(&index).unwrap()["weight_map"].to_string().as_str())
                .unwrap();
        assert_eq!(f.len(), weight_map.len(), "张量总数 == weight_map 条目数");
        // index 列出的每个键都存在,且归属同一分片
        for k in weight_map.keys() {
            assert!(f.info(k).is_ok(), "index 键 {k} 在文件头缺失");
        }
        // 无缝衔接:按 begin 排序后 prev.end == next.begin,首 0、末 == 数据区长度
        let mut infos: Vec<&SafeTensorInfo> = f.tensors.values().collect();
        infos.sort_by_key(|i| i.begin);
        assert_eq!(infos[0].begin, 0);
        for w in infos.windows(2) {
            assert_eq!(w[0].end, w[1].begin, "{} 与 {} 之间有缝隙", w[0].shape_debug_name(), w[1].shape_debug_name());
        }
        eprintln!("contiguity: {} 张量无缝全覆盖, 数据区 = {} 字节", infos.len(), infos.last().unwrap().end);
    }

    impl SafeTensorInfo {
        /// 测试报错用可读名(无名字段,拼 shape/dtype 代指)。
        fn shape_debug_name(&self) -> String {
            format!("{}{:?}@{}", "", self.shape, self.begin)
        }
    }

    /// 真文件:抽查 0.8B 关键键的 shape/dtype(注意真键名前缀
    /// `model.language_model.`;visual 塔也在本分片)。
    #[test]
    fn real_file_shapes() {
        let Some(p) = model_path() else { return };
        let f = SafeTensorsFile::open(p).unwrap();
        let check = |name: &str, shape: &[usize], dt: SafeDtype| {
            let i = f.info(name).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(i.shape, shape, "{name} shape");
            assert_eq!(i.dtype, dt, "{name} dtype");
        };
        check(
            "model.language_model.embed_tokens.weight",
            &[248320, 1024],
            SafeDtype::Bf16,
        );
        // GDN 层:q/k/v 拼接 = 3 × (16 头 × 128)
        check(
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            &[6144, 1024],
            SafeDtype::Bf16,
        );
        // full_attention 层:q = 16 头 × head_dim 256(修正:阶段二文档写 8 头)
        check(
            "model.language_model.layers.23.self_attn.q_proj.weight",
            &[4096, 1024],
            SafeDtype::Bf16,
        );
        // 无独立 lm_head(与 embed_tokens 共享);visual 塔在本分片
        assert!(f.info("model.language_model.lm_head.weight").is_err());
        check(
            "model.visual.merger.linear_fc1.weight",
            &[3072, 3072],
            SafeDtype::Bf16,
        );
        // 5-D shape(conv 核)也是合法条目
        check(
            "model.visual.patch_embed.proj.weight",
            &[768, 3, 2, 16, 16],
            SafeDtype::Bf16,
        );
        // MTP 层权重(阶段二情报:mtp.pre_fc_norm_embedding 存在)
        check("mtp.pre_fc_norm_embedding.weight", &[1024], SafeDtype::Bf16);
        eprintln!(
            "shapes: 7 个抽查键全中(embed/in_proj_qkv 6144/q_proj 4096/visual_fc1/patch_embed 5D/mtp)"
        );
    }

    /// 真文件:bf16→f32 数值抽查(与 python 侧独立读出的首元素对照)。
    #[test]
    fn real_file_numeric_spot() {
        let Some(p) = model_path() else { return };
        let f = SafeTensorsFile::open(p).unwrap();
        let v = f
            .tensor_f32("model.language_model.embed_tokens.weight")
            .unwrap();
        assert_eq!(v.len(), 248320 * 1024);
        // python 侧独立读出:首 2 个 bf16 bits = 15456, 15450
        assert_eq!(v[0], bf16_bits_to_f32(15456));
        assert_eq!(v[1], bf16_bits_to_f32(15450));
        // 非零(排除全零假阳性)
        assert!(v[0] != 0.0);
        // 与另一个层权重不同值(排除错位读同一存储)
        let other = f
            .tensor_f32("model.language_model.layers.0.input_layernorm.weight")
            .unwrap();
        assert_ne!(v[0], other[0]);
        eprintln!("numeric: embed[0..2] 位型匹配 python 独立读数, 与 layernorm 非同值");
    }
}

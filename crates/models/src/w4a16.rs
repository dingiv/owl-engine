//! W4A16 装载源(E3-ii,2026-09-27):compressed-tensors pack-quantized
//! 检查点(llm-compressor 社区转换器产物)→ **marlin-ready 字节表**
//! (WeightSource)。
//!
//! 母本格式 = repack.rs 头注(S3-P2 实测定谳):
//! - `weight_packed` i32 [out, in/8](LSB-first 8 nibble/i32)
//! - `weight_scale` bf16 [out, in/128](per out 通道 × 128-k 分组)
//! - `weight_shape` i64 [out, in];U4B8 语义:W = (q - 8) × scale
//!
//! 装载期重排(一次性物化,rayon 并行):
//! - 尺寸门控 eligible(out%256==0 且 in%128==0)→ qweight =
//!   pack_marlin_b(q, k, n) + scales = pack_marlin_s(f32, n, k/128)
//! - 非门控(小线性,如 GDN in_proj_z/b/a 16 行)→ 反量化 f16 `.weight`
//!   (保持 f16 直读,Linear 不使能量化臂)
//! - 其余张量(embed/norm/conv bf16、A_log/dt_bias f32)→ f16 字节归一
//!
//! 与引擎的分工:本源只供字节;`Linear::enable_w4a16` 用同一谓词
//! ([`marlin_eligible`])决定量化臂/f16 臂 —— 判定单一来源。

use crate::contract::{Dtype, ModelError};
use crate::module::WeightSource;
use owl_kernels::marlin::repack::{
    marlin_gather_indices, pack_marlin_b_gather, pack_marlin_s, unpack_nibbles,
};
use std::collections::HashMap;
use std::path::Path;

/// marlin 可承载判定(与 repack pack_marlin_b 的 tile 约束一致):
/// n(out)%256==0 且 k(in)%128==0。不满足 → 反量化 f16。
pub fn marlin_eligible(out_dim: usize, in_dim: usize) -> bool {
    out_dim % 256 == 0 && in_dim % 128 == 0
}

/// 打包 n 的安全档位(E3-ii g128 暗雷对策):.a 核仅对 **512×2^k** 的
/// n 形状正确(实测扫描:512/1024/2048/4096 全绿;1536/2560/3072/4608
/// /5120/5632/6144 → NaN/垃圾,3584 退化)。非安全 n → 垫零列到下一档
/// (U4B8 零权重 = nibble 8,scale 垫 0,双保险零贡献)。
pub fn marlin_n_pack(n: usize) -> usize {
    let mut p = 512usize;
    while p < n {
        p *= 2;
    }
    p
}

/// 字节表条目:(LE 字节, dtype, 元素数)
type Entry = (Vec<u8>, Dtype, usize);

pub struct W4A16Source {
    table: HashMap<String, Entry>,
}

impl W4A16Source {
    /// 装载入口:**烘焙缓存优先**(s3-load 同款)—— 首次重排物化(慢,
    /// 一次性)后落盘 `{dir}/marlin_cache.safetensors`,此后装载直读
    /// 缓存(亚秒)。缓存失配(换检查点)→ 删除该文件重生成。
    pub fn open_dir(dir: &Path) -> Result<Self, ModelError> {
        let t0 = std::time::Instant::now();
        let cache = dir.join("marlin_cache.safetensors");
        if cache.exists() {
            let src = Self::load_cache(&cache)?;
            eprintln!("[w4a16] 缓存命中 {} 项 @ {:?}", src.table.len(), t0.elapsed());
            return Ok(src);
        }
        let src = Self::materialize(dir)?;
        eprintln!("[w4a16] 物化 {} 项 @ {:?}", src.table.len(), t0.elapsed());
        let t1 = std::time::Instant::now();
        src.save_cache(&cache)?;
        eprintln!("[w4a16] 缓存落盘 @ {:?}", t1.elapsed());
        Ok(src)
    }

    fn load_cache(path: &Path) -> Result<Self, ModelError> {
        let raw = std::fs::read(path)
            .map_err(|e| ModelError::Msg(format!("W4A16 缓存读 {path:?}: {e}")))?;
        let st = safetensors::SafeTensors::deserialize(&raw)
            .map_err(|e| ModelError::Msg(format!("W4A16 缓存解析: {e:?}")))?;
        let mut table = HashMap::new();
        for name in st.names() {  // FIXME: 使用哈希表来装每一个张量吗？这太离谱了呀！你这。
            let tv = st
                .tensor(name)
                .map_err(|e| ModelError::Msg(format!("W4A16 缓存 {name}: {e:?}")))?;
            let dtype = match tv.dtype() {
                safetensors::Dtype::U32 => Dtype::U32,
                safetensors::Dtype::F16 => Dtype::F16,
                other => {
                    return Err(ModelError::Msg(format!(
                        "W4A16 缓存 {name}: 不支持 {other:?}"
                    )))
                }
            };
            let elems: usize = tv.shape().iter().product();
            table.insert(name.to_string(), (tv.data().to_vec(), dtype, elems));
        }
        Ok(Self { table })
    }

    fn save_cache(&self, path: &Path) -> Result<(), ModelError> {
        use safetensors::tensor::TensorView;
        let mut data = HashMap::new();
        for (name, (bytes, dtype, elems)) in &self.table {
            let stdtype = match dtype {
                Dtype::U32 => safetensors::Dtype::U32,
                Dtype::F16 => safetensors::Dtype::F16,
                _ => safetensors::Dtype::F32,
            };
            data.insert(
                name.clone(),
                TensorView::new(stdtype, vec![*elems], bytes).expect("cache view"),
            );
        }
        safetensors::serialize_to_file(data, None, path)
            .map_err(|e| ModelError::Msg(format!("W4A16 缓存写: {e:?}")))?;
        Ok(())
    }

    /// 重排物化(慢路径,仅首次):扫检查点 → 逐线性 ct→marlin 重排
    fn materialize(dir: &Path) -> Result<Self, ModelError> {
        let path = dir.join("model.safetensors");
        let raw = std::fs::read(&path)
            .map_err(|e| ModelError::Msg(format!("W4A16: 读 {path:?}: {e}")))?;
        let st = safetensors::SafeTensors::deserialize(&raw)
            .map_err(|e| ModelError::Msg(format!("W4A16: safetensors 解析: {e:?}")))?;

        let mut table: HashMap<String, Entry> = HashMap::new();
        let mut idx_cache: HashMap<(usize, usize), Vec<u32>> = HashMap::new();
        let (mut t_unpack, mut t_pack, mut t_pass) = (0u128, 0u128, 0u128);
        let dbg = std::env::var_os("OWL_DEBUG").is_some();
        let t0 = std::time::Instant::now();
        let mut done = 0usize;
        for name in st.names() {
            done += 1;
            if dbg && done % 32 == 0 {
                let th = std::fs::read_to_string("/proc/self/status")
                    .ok()
                    .and_then(|s| s.lines().find(|l| l.starts_with("Threads")).map(|l| l.to_string()))
                    .unwrap_or_default();
                eprintln!("[dbg w4a16] {done} 张量 @ {:?} {th}", t0.elapsed());
            }
            if name.contains("mtp") || name.contains("visual") {
                continue; // 非主干键(owl 不消费)
            }
            let tv = st.tensor(name).map_err(|e| ModelError::Msg(format!("W4A16 {name}: {e:?}")))?;
            let shape: Vec<usize> = tv.shape().to_vec();
            let elems: usize = shape.iter().product();
            if dbg && name.ends_with(".weight_packed") {
                eprintln!("[dbg w4a16] 重排 {name} (out={}, k={}) @ {:?}", shape[0], shape[1] * 8, t0.elapsed());
            }

            if name.ends_with(".weight_shape") {
                continue; // 元数据([out, in];形状已由 weight_packed 蕴含)
            }
            if let Some(base) = name.strip_suffix(".weight_packed") {
                // 量化三元组 → marlin 重排(或反量化 f16)
                let out = shape[0];
                let k = shape[1] * 8; // 8 nibble/i32
                let packed = i32s(tv.data());
                let scale_bits: Vec<u16> = {
                    let sv = st
                        .tensor(&format!("{base}.weight_scale"))
                        .map_err(|e| ModelError::Msg(format!("W4A16 {base}: {e:?}")))?;
                    debug_assert_eq!(sv.dtype(), safetensors::Dtype::BF16);
                    sv.data().chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
                };
                let scales_f32: Vec<f32> = scale_bits
                    .iter()
                    .map(|&bits| f32::from_bits((bits as u32) << 16))
                    .collect();
                if marlin_eligible(out, k) {
                    // n-pad(暗雷对策):垫零列到安全档 512×2^k
                    let n_pack = marlin_n_pack(out);
                    let pad_rows = n_pack - out;
                    let pad_qrow = vec![8u8; k]; // U4B8 零权重 nibble = 8
                    let _u = std::time::Instant::now();
                    let mut q = unpack_nibbles(&packed, out, k);
                    t_unpack += _u.elapsed().as_nanos();
                    for _ in 0..pad_rows {
                        q.extend_from_slice(&pad_qrow);
                    }
                    // 索引表按形状缓存(同形状层复用;除法分解只算一次)
                    let idx = idx_cache
                        .entry((k, n_pack))
                        .or_insert_with(|| marlin_gather_indices(k, n_pack));
                    let b = pack_marlin_b_gather(&q, idx, q.len() / 8);
                    let mut scales_pad = scales_f32.clone();
                    scales_pad.extend(std::iter::repeat(0.0).take(pad_rows * (k / 128)));
                    let s = pack_marlin_s(&scales_pad, n_pack, k / 128);
                    table.insert(
                        format!("{base}.qweight"),
                        (i32s_le_bytes(&b), Dtype::U32, b.len()),
                    );
                    table.insert(
                        format!("{base}.scales"),
                        (u16s_le_bytes(&s), Dtype::F16, s.len()),
                    );
                    // marlin workspace(零初始化;Alloc 契约 + 本源零表双保证)
                    let ws_len = owl_kernels::marlin::v2_workspace_len(n_pack);
                    table.insert(
                        format!("{base}.marlin_ws"),
                        (vec![0u8; ws_len * 4], Dtype::U32, ws_len),
                    );
                    table.insert(format!("{base}.marlin_ctmp"), (vec![0u8; 4], Dtype::U32, 1));
                } else {
                    // 小线性:反量化 f16,装载走 f16 直读臂
                    // (w[i] = (q[i] - 8) × scale[row=i/k][group=col/128])
                    let groups = k / 128;
                    let w_elems = out * k;
                    let mut w = vec![0u16; w_elems];
                    for (i, cell) in w.iter_mut().enumerate() {
                        let qi = (packed[i / 8] >> (4 * (i % 8))) as u8 & 0xF;
                        let row = i / k;
                        let col = i % k;
                        let v = (qi as f32 - 8.0) * scales_f32[row * groups + col / 128];
                        *cell = half::f16::from_f32(v).to_bits();
                    }
                    table.insert(format!("{base}.weight"), (u16s_le_bytes(&w), Dtype::F16, w_elems));
                }
            } else {
                // 非量化:归一 F16 字节(bf16/f32 → f16)
                let bytes = match tv.dtype() {
                    safetensors::Dtype::BF16 => bf16_to_f16_bytes(tv.data()),
                    safetensors::Dtype::F32 => f32_to_f16_bytes(tv.data()),
                    safetensors::Dtype::F16 => tv.data().to_vec(),
                    other => {
                        return Err(ModelError::Msg(format!(
                            "W4A16 {name}: 不支持的字节 dtype {other:?}"
                        )))
                    }
                };
                table.insert(name.to_string(), (bytes, Dtype::F16, elems));
            }
        }
        Ok(Self { table })
    }
}

fn i32s(bytes: &[u8]) -> Vec<i32> {
    bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn i32s_le_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn u16s_le_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bf16_to_f16_bytes(data: &[u8]) -> Vec<u8> {
    use rayon::prelude::*;
    let mut out = vec![0u8; data.len()];
    out.par_chunks_exact_mut(2)
        .zip(data.par_chunks_exact(2))
        .for_each(|(o, c)| {
            let bits = u16::from_le_bytes([c[0], c[1]]);
            let f = f32::from_bits((bits as u32) << 16);
            let h = half::f16::from_f32(f);
            o[0] = h.to_le_bytes()[0];
            o[1] = h.to_le_bytes()[1];
        });
    out
}

fn f32_to_f16_bytes(data: &[u8]) -> Vec<u8> {
    use rayon::prelude::*;
    let mut out = vec![0u8; data.len() / 2];
    out.par_chunks_exact_mut(2)
        .zip(data.par_chunks_exact(4))
        .for_each(|(o, c)| {
            let h = half::f16::from_f32(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            o[0] = h.to_le_bytes()[0];
            o[1] = h.to_le_bytes()[1];
        });
    out
}

impl WeightSource for W4A16Source {
    fn take(&self, key: &str) -> Option<Vec<f32>> {
        let (bytes, dtype, _) = self.table.get(key)?;
        Some(match dtype {
            Dtype::F16 => bytes
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
            Dtype::U32 => bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
                .collect(),
            _ => bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        })
    }

    fn elem_len(&self, key: &str) -> Option<usize> {
        self.table.get(key).map(|(_, _, n)| *n)
    }

    /// 字节直写(登记 dtype 与 Want dtype 一致 → 纯 memcpy)
    fn convert_chunk_into_bytes(
        &self,
        key: &str,
        offset_elems: usize,
        len: usize,
        dst: &mut [u8],
        dtype: Dtype,
    ) -> Option<()> {
        let dbg = std::env::var_os("OWL_DEBUG").is_some();
        let (bytes, reg_dtype, _elems) = match self.table.get(key) {
            Some(e) => e,
            None => {
                if dbg {
                    eprintln!("[dbg w4a16] convert 缺键 {key}");
                }
                return None;
            }
        };
        if *reg_dtype != dtype {
            if dbg {
                eprintln!("[dbg w4a16] convert dtype 不配 {key}: 表 {:?} vs 要 {:?}", reg_dtype, dtype);
            }
            return None;
        }
        let esz = dtype.size_bytes();
        let start = offset_elems * esz;
        let end = start + len * esz;
        let need = len * esz; // 租约是**块级**缓冲(dst[0..need]),与表偏移无关
        if end > bytes.len() || need > dst.len() {
            if dbg {
                eprintln!(
                    "[dbg w4a16] convert 越界 {key}: end={end} 表={} need={need} dst={} esz={esz} len={len} off={offset_elems}",
                    bytes.len(), dst.len()
                );
            }
            return None;
        }
        dst[..need].copy_from_slice(&bytes[start..end]);
        Some(())
    }
}

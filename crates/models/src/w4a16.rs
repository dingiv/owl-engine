//! W4A16 装载源 **v2:mmap 懒物化**(2026-09-28 治本重构)。
//!
//! v1 形态(整文件 fs::read + 全量预物化进 HashMap + 烘焙缓存)已废除,
//! 三宗罪定谳(实测立案):debug 冷装载 46.7s(串行反量化 ~16s + 串行
//! 字节转换黑段)、RAM 峰值 = 检查点 × 4-5 份全量拷贝(27B 不可行)、
//! 缓存文件比源检查点还大 57%。
//!
//! v2 架构 = 与 f16 基线 [`crate::loader::SafeTensorsSource`] 同源:
//! - **open 零拷贝**:mmap + 条目索引(共享底座 [`crate::loader::open_raw_index`],
//!   多分片仓通用);
//! - **懒物化**:键首触才重排/反量化(装载域本就按窗口分块拉取);
//!   per-key 字节缓存(容量上限清空驱逐,主机驻留 = 在途份 + cap 窗口);
//! - eligible([`marlin_eligible`],与 Linear 构造期 QuantPlan 门控单一来源)
//!   → ct packed+scale → unpack(rayon)→ gather-pack(rayon,索引表按形状
//!   复用)→ `qweight` U32 + `scales` F16;`marlin_ws`/`marlin_ctmp` =
//!   **零填充直写,零存储**;
//! - 非门控 → rayon 反量化 F16(核在 owl_f16c,opt 覆盖 debug 免疫;
//!   每 i32 的 8 nibble 同 group,scale 每 i32 取一次零除法);
//! - passthrough → mmap 视图 F16C 单 pass 直写 dst(bf16/f32→f16)或
//!   memcpy(f16),窗口消费完 DONTNEED 还页。
//!
//! 母本格式(repack.rs 头注,S3-P2 实测定谳):`weight_packed` i32
//! [out, in/8](LSB-first)、`weight_scale` bf16 [out, in/128]、
//! U4B8 语义 W = (q-8) × scale。

use crate::contract::{Dtype, ModelError};
use crate::loader::{open_raw_index, Mmap, RawEntry};
use crate::module::WeightSource;
use owl_f16c::{bf16_bytes_to_f16_bytes, dequant_u4_affine_f16_bytes};
use owl_kernels::marlin::repack::{
    marlin_gather_indices, pack_marlin_b_gather_into, pack_marlin_s, unpack_nibbles_into,
};
use owl_kernels::marlin::v2_workspace_len;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// marlin 可承载判定(**g128 暗雷收紧版**,2026-09-27 形状扫描实锤):
/// .a 核仅对 **n = 512×2^k**(512/1024/2048/4096/8192…)的形状正确;
/// 其余 n(1536/2560/3072/3584/4608/5120/5632/6144)→ NaN/垃圾/挂死
/// (sweep 全录:marlin_shape_sweep)。不满足 → 反量化 f16。
/// k(in)%128==0 照旧(分组对齐)。
pub fn marlin_eligible(out_dim: usize, in_dim: usize) -> bool {
    out_dim % 512 == 0 && out_dim.is_power_of_two() && in_dim % 128 == 0
}

/// 打包 n 的安全档位(g128 暗雷对策):eligible 形状(n = 512×2^k)下
/// 恒等于 n 本身(pad 路线已废:pad 到 2 次幂实测仍挂,根因在 .a 内部
/// config);非安全 n 由 eligible 判定走 f16,不打包。
pub fn marlin_n_pack(n: usize) -> usize {
    let mut p = 512usize;
    while p < n {
        p *= 2;
    }
    p
}

/// 懒物化缓存容量上限(字节;超出整表清空 —— 装载按层序推进,
/// 在途 Arc 由消费方持活,峰值 = cap + 并发在途张量)。
const CACHE_CAP_BYTES: usize = 512 << 20;

/// 键字节解析结果(懒物化后的一致字节 + 登记 dtype)
enum Resolved {
    /// 零填充直写(ws/ctmp;不占存储)
    Zeroed,
    /// mmap 视图直转/直拷(passthrough;f16 基线同款)
    View(RawEntry),
    /// 已物化字节(qweight U32 / scales F16 / 反量化 weight F16)
    Bytes(Arc<[u8]>, Dtype),
}

pub struct W4A16Source {
    maps: Vec<Mmap>,
    index: HashMap<String, RawEntry>,
    /// 懒物化字节缓存(键 → 字节;容量清空驱逐)
    cache: Mutex<(HashMap<String, Arc<[u8]>>, usize)>,
    /// gather 索引表(按 (k, n) 形状复用;除法分解只算一次)
    idx_cache: Mutex<HashMap<(usize, usize), Arc<Vec<u32>>>>,
    /// 构建串行锁(重排内部 rayon 已饱和全核;并发双构建纯浪费)
    build_lock: Mutex<()>,
}

impl W4A16Source {
    /// 装载入口:mmap + 索引(毫秒级,零数据拷贝);重排/反量化全部
    /// 懒到键首触。**无缓存文件**(v1 烘焙缓存随 v1 形态一并废除)。
    pub fn open_dir(dir: &Path) -> Result<Self, ModelError> {
        let t0 = std::time::Instant::now();
        let (maps, index) = open_raw_index(dir)?;
        eprintln!(
            "[w4a16] v2 mmap 索引 {} 项 @ {:?}(懒物化,无缓存)",
            index.len(),
            t0.elapsed()
        );
        Ok(Self {
            maps,
            index,
            cache: Mutex::new((HashMap::new(), 0)),
            idx_cache: Mutex::new(HashMap::new()),
            build_lock: Mutex::new(()),
        })
    }

    /// 诊断口:索引键全集(装载面排查/基准用;不含派生键 qweight/scales/ws)
    pub fn keys(&self) -> Vec<String> {
        self.index.keys().cloned().collect()
    }

    /// 量化线性三元组定位:(out, k) 与 packed/scale 条目。
    /// shape 事实:`weight_packed` [out, k/8] i32。
    fn linear_dims(&self, base: &str) -> Option<(usize, usize, &RawEntry, &RawEntry)> {
        let pe = self.index.get(&format!("{base}.weight_packed"))?;
        let se = self.index.get(&format!("{base}.weight_scale"))?;
        let out = *pe.shape.first()?;
        let k = pe.shape.get(1)? * 8;
        Some((out, k, pe, se))
    }

    /// 键解析(唯一入口;懒物化在此触发)
    fn resolve(&self, key: &str) -> Option<Resolved> {
        if key.ends_with(".marlin_ws") || key.ends_with(".marlin_ctmp") {
            return Some(Resolved::Zeroed);
        }
        if let Some(base) = key.strip_suffix(".qweight") {
            return Some(Resolved::Bytes(self.linear_bytes_for(base, key)?, Dtype::U32));
        }
        if let Some(base) = key.strip_suffix(".scales") {
            return Some(Resolved::Bytes(self.linear_bytes_for(base, key)?, Dtype::F16));
        }
        // passthrough 优先(norm/conv/embed 等原生存量键)
        if let Some(e) = self.index.get(key) {
            return Some(Resolved::View(e.clone()));
        }
        // 非门控线性反量化臂(Linear 与本源同谓词,该键只对 non-eligible 出现)
        if let Some(base) = key.strip_suffix(".weight") {
            if self.index.contains_key(&format!("{base}.weight_packed")) {
                return Some(Resolved::Bytes(self.linear_bytes_for(base, key)?, Dtype::F16));
            }
        }
        None
    }

    /// 量化线性物化(键首触才建;qweight/scales 成对,weight 独立)。
    /// 双检缓存(快路径锁内查;构建在 build_lock 串行内,防并发双建)。
    fn linear_bytes_for(&self, base: &str, key: &str) -> Option<Arc<[u8]>> {
        let (out, k, pe, se) = self.linear_dims(base)?;
        {
            let cache = self.cache.lock().unwrap();
            if let Some(b) = cache.0.get(key) {
                return Some(b.clone());
            }
        }
        self.build_linear(base, out, k, pe, se).ok()?;
        self.cache.lock().unwrap().0.get(key).cloned()
    }

    /// 真构建(串行;内部 rayon 饱和)
    fn build_linear(
        &self,
        base: &str,
        out: usize,
        k: usize,
        pe: &RawEntry,
        se: &RawEntry,
    ) -> Result<(), ModelError> {
        let _g = self.build_lock.lock().unwrap();
        let t0 = std::time::Instant::now();
        // 源视图:i32 packed(rayon 转换,对齐安全)+ bf16 scales → f32
        let packed_bytes = &self.maps[pe.map_ix][pe.start..pe.start + pe.nbytes];
        let packed = i32s_par(packed_bytes);
        let scale_bytes = &self.maps[se.map_ix][se.start..se.start + se.nbytes];
        let scales_f32 = bf16_par(scale_bytes);
        // 源页消费完毕即还(DONTNEED;file-backed clean page 再访重新缺页)
        self.maps[pe.map_ix].dontneed(pe.start, pe.nbytes);
        self.maps[se.map_ix].dontneed(se.start, se.nbytes);

        let mut inserts: Vec<(String, Arc<[u8]>, usize)> = Vec::new();
        if marlin_eligible(out, k) {
            let mut q_buf: Vec<u8> = Vec::new();
            unpack_nibbles_into(&packed, out, k, &mut q_buf);
            let idx = self
                .idx_cache
                .lock()
                .unwrap()
                .entry((k, out))
                .or_insert_with(|| Arc::new(marlin_gather_indices(k, out)))
                .clone();
            let words = out * k / 8;
            let mut b_buf: Vec<i32> = Vec::new();
            pack_marlin_b_gather_into(&q_buf, &idx, words, &mut b_buf);
            let s = pack_marlin_s(&scales_f32, out, k / 128);
            inserts.push((
                format!("{base}.qweight"),
                Arc::from(i32s_le_bytes(&b_buf).into_boxed_slice()),
                b_buf.len() * 4,
            ));
            inserts.push((
                format!("{base}.scales"),
                Arc::from(u16s_le_bytes(&s).into_boxed_slice()),
                s.len() * 2,
            ));
        } else {
            let mut dst = vec![0u8; out * k * 2];
            dequant_u4_affine_f16_bytes(&packed, &scales_f32, out, k, &mut dst);
            inserts.push((
                format!("{base}.weight"),
                Arc::from(dst.into_boxed_slice()),
                out * k * 2,
            ));
        }
        let mut cache = self.cache.lock().unwrap();
        let add: usize = inserts.iter().map(|(_, _, b)| b).sum();
        if cache.1 + add > CACHE_CAP_BYTES {
            cache.0.clear();
            cache.1 = 0;
        }
        for (kk, b, nb) in inserts {
            cache.0.insert(kk, b);
            cache.1 += nb;
        }
        drop(cache);
        eprintln!(
            "[w4a16] 物化 {base} (out={out}, k={k}) @ {:?}",
            t0.elapsed()
        );
        Ok(())
    }

    /// 键全量字节视图(懒物化 / passthrough 归一;take/take_range 用)
    fn key_bytes(&self, key: &str) -> Option<(KeyData, Dtype)> {
        Some(match self.resolve(key)? {
            Resolved::Zeroed => (KeyData::Owned(vec![0; self.elem_len(key)? * 4]), Dtype::U32),
            Resolved::View(e) => {
                // passthrough 归一 F16 字节(want dtype 恒 F16)
                let src = &self.maps[e.map_ix][e.start..e.start + e.nbytes];
                let mut out = vec![0u8; e.nbytes];
                match e.dtype {
                    safetensors::Dtype::F16 => out.copy_from_slice(src),
                    safetensors::Dtype::BF16 => bf16_bytes_to_f16_bytes(src, &mut out),
                    safetensors::Dtype::F32 => out = f32_bytes_to_f16(src),
                    _ => return None, // weight_shape 等元数据键不可取
                }
                (KeyData::Owned(out), Dtype::F16)
            }
            Resolved::Bytes(b, dt) => (KeyData::Shared(b), dt),
        })
    }
}

enum KeyData {
    Owned(Vec<u8>),
    Shared(Arc<[u8]>),
}

impl KeyData {
    fn as_slice(&self) -> &[u8] {
        match self {
            KeyData::Owned(v) => v,
            KeyData::Shared(b) => b,
        }
    }
}

impl WeightSource for W4A16Source {
    fn take(&self, key: &str) -> Option<Vec<f32>> {
        let n = self.elem_len(key)?;
        self.take_range(key, 0, n)
    }

    fn elem_len(&self, key: &str) -> Option<usize> {
        // 元数据键(I64 [out,in])owl 不消费,拒绝可取化
        if key.ends_with(".weight_shape") {
            return None;
        }
        // 派生键:尺寸由基线性形状推导;谓词门控(eligible → qweight/scales/
        // ws/ctmp;non-eligible → weight),错位键报 None 免误消费
        if let Some(base) = key.strip_suffix(".marlin_ws") {
            let (out, k, _, _) = self.linear_dims(base)?;
            if !marlin_eligible(out, k) {
                return None;
            }
            return Some(v2_workspace_len(marlin_n_pack(out)));
        }
        if key.ends_with(".marlin_ctmp") {
            let base = &key[..key.len() - ".marlin_ctmp".len()];
            let (out, k, _, _) = self.linear_dims(base)?;
            if !marlin_eligible(out, k) {
                return None;
            }
            return Some(1);
        }
        if let Some(base) = key.strip_suffix(".qweight") {
            let (out, k, _, _) = self.linear_dims(base)?;
            if !marlin_eligible(out, k) {
                return None;
            }
            return Some(out * k / 8);
        }
        if let Some(base) = key.strip_suffix(".scales") {
            let (out, k, _, _) = self.linear_dims(base)?;
            if !marlin_eligible(out, k) {
                return None;
            }
            return Some(out * (k / 128));
        }
        if let Some(base) = key.strip_suffix(".weight") {
            if !self.index.contains_key(key) {
                let (out, k, _, _) = self.linear_dims(base)?;
                if marlin_eligible(out, k) {
                    return None;
                }
                return Some(out * k);
            }
        }
        self.index.get(key).map(|e| {
            let elems: usize = e.shape.iter().product();
            elems
        })
    }

    fn take_range(&self, key: &str, offset_elems: usize, len: usize) -> Option<Vec<f32>> {
        let (data, dtype) = self.key_bytes(key)?;
        let bytes = data.as_slice();
        let esz = dtype.size_bytes();
        let s = offset_elems.checked_mul(esz)?;
        let e = s.checked_add(len.checked_mul(esz)?)?;
        let win = bytes.get(s..e)?;
        Some(match dtype {
            Dtype::F16 => win
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
            Dtype::U32 => win
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
                .collect(),
            _ => win
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        })
    }

    /// 分块转换直写字节租约(主路径):懒物化/passthrough/零填充三臂。
    fn convert_chunk_into_bytes(
        &self,
        key: &str,
        offset_elems: usize,
        len: usize,
        dst: &mut [u8],
        dtype: Dtype,
    ) -> Option<()> {
        let esz = dtype.size_bytes();
        let need = len.checked_mul(esz)?;
        if need > dst.len() {
            return None;
        }
        match self.resolve(key)? {
            Resolved::Zeroed => {
                dst[..need].fill(0);
                Some(())
            }
            Resolved::View(e) => {
                if dtype != Dtype::F16 {
                    return None;
                }
                let src_esz = match e.dtype {
                    safetensors::Dtype::BF16 | safetensors::Dtype::F16 => 2,
                    safetensors::Dtype::F32 => 4,
                    _ => return None,
                };
                let s = e.start + offset_elems * src_esz;
                let win = &self.maps[e.map_ix][s..s + len * src_esz];
                match e.dtype {
                    safetensors::Dtype::F16 => dst[..need].copy_from_slice(win),
                    safetensors::Dtype::BF16 => bf16_bytes_to_f16_bytes(win, &mut dst[..need]),
                    safetensors::Dtype::F32 => {
                        // F32 passthrough 仅 GDN 标量小张量,标量循环足够
                        for (i, c) in win.chunks_exact(4).enumerate() {
                            let f = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                            dst[i * 2..i * 2 + 2]
                                .copy_from_slice(&half::f16::from_f32(f).to_le_bytes());
                        }
                    }
                    _ => return None,
                }
                self.maps[e.map_ix].dontneed(s, len * src_esz);
                Some(())
            }
            Resolved::Bytes(b, reg) => {
                if reg != dtype {
                    return None;
                }
                let s = offset_elems * esz;
                dst[..need].copy_from_slice(b.get(s..s + need)?);
                Some(())
            }
        }
    }
}

// ============================================================================
// 字节转换助手(rayon;对齐安全 from_le_bytes)
// ============================================================================

fn i32s_par(bytes: &[u8]) -> Vec<i32> {
    use rayon::prelude::*;
    let n = bytes.len() / 4;
    let mut v = vec![0i32; n];
    v.par_chunks_mut(1 << 16)
        .enumerate()
        .for_each(|(t, chunk)| {
            for (i, cell) in chunk.iter_mut().enumerate() {
                let o = (t * (1 << 16) + i) * 4;
                *cell = i32::from_le_bytes([
                    bytes[o],
                    bytes[o + 1],
                    bytes[o + 2],
                    bytes[o + 3],
                ]);
            }
        });
    v
}

fn bf16_par(bytes: &[u8]) -> Vec<f32> {
    use rayon::prelude::*;
    let n = bytes.len() / 2;
    let mut v = vec![0f32; n];
    v.par_chunks_mut(1 << 16)
        .enumerate()
        .for_each(|(t, chunk)| {
            for (i, cell) in chunk.iter_mut().enumerate() {
                let o = (t * (1 << 16) + i) * 2;
                *cell = f32::from_bits(
                    (u16::from_le_bytes([bytes[o], bytes[o + 1]]) as u32) << 16,
                );
            }
        });
    v
}

fn f32_bytes_to_f16(bytes: &[u8]) -> Vec<u8> {
    use rayon::prelude::*;
    let mut out = vec![0u8; bytes.len() / 2];
    out.par_chunks_exact_mut(2)
        .zip(bytes.par_chunks_exact(4))
        .for_each(|(o, c)| {
            let h = half::f16::from_f32(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            o[0] = h.to_le_bytes()[0];
            o[1] = h.to_le_bytes()[1];
        });
    out
}

fn i32s_le_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn u16s_le_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

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
//! - 非门控 → rayon 反量化 F16(核在 crate::f16c(原 opt 覆盖 crate,2026-10-01 并入);
//!   每 i32 的 8 nibble 同 group,scale 每 i32 取一次零除法);
//! - passthrough → mmap 视图 F16C 单 pass 直写 dst(bf16/f32→f16)或
//!   memcpy(f16),窗口消费完 DONTNEED 还页。
//!
//! 母本格式(repack.rs 头注,S3-P2 实测定谳):`weight_packed` i32
//! [out, in/8](LSB-first)、`weight_scale` bf16 [out, in/128]、
//! U4B8 语义 W = (q-8) × scale。

use crate::contract::{Dtype, ModelError};
use crate::formats::mmap::{open_raw_index, Mmap, RawEntry};
use crate::module::WeightSource;
use crate::f16c::{bf16_bytes_to_f16_bytes, dequant_u4_affine_f16_bytes};
use owl_kernels::marlin::repack::{
    marlin_gather_indices, pack_marlin_b_gather_into, pack_marlin_s, unpack_nibbles_into,
};
use owl_kernels::marlin::v2_workspace_len;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// marlin 可承载判定(**g128 暗雷根治后放宽**,2026-09-30):暗雷真凶 =
/// marlin_host 的 use_fp32_reduce=true + 4B c_tmp 占位(slice_count>1
/// 形状 global_reduce_fp32 写穿 c_tmp 踩内存,呈「n=2^k 才安全」假象;
/// 按 a_is_s8 分派修复后 sweep 全绿 512..17408 含全部非 2^k 档与 27B
/// 维度族,m∈{1,32,4096})。判据回归 kernel/pack 真实约束:
/// **n%256==0**(repack tile)且 **in%128==0**(g128 分组)。
pub fn marlin_eligible(out_dim: usize, in_dim: usize) -> bool {
    out_dim % 256 == 0 && in_dim % 128 == 0
}

/// 打包 n 档位(暗雷根治后 = **n 直通**):eligibility 已保证 n%256==0
/// (repack tile 约束),历史 pad 路线(n → 512×2^k)系暗雷误诊产物,
/// 已废(pad 实测仍挂;真凶 = use_fp32_reduce/c_tmp,见
/// [`marlin_eligible`] 文档)。保留函数 = 语义锚点,防再引入 pad。
pub fn marlin_n_pack(n: usize) -> usize {
    debug_assert!(n % 256 == 0, "marlin pack 要求 n%256==0(得 {n})");
    n
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
        // 家族嗅探(2026-10-10 立案):本源只吃 W4A16 量化家族
        // (weight_packed/scale/shape 三件套;syvai 系)。BF16 未量化源
        // (z-lab golden 参考系)喂进来 = 全部权重噪声 → drafts junk →
        // AL=0 全拒(实测立案:数学域 36 → 157 t/s 的 4.3×)。缺
        // weight_packed = 结构化拒启并指路,不静默。
        if !index.keys().any(|k| k.ends_with(".weight_packed")) {
            return Err(ModelError::Msg(format!(
                "{} 缺 weight_packed 量化键:疑似未量化源(BF16/F16 golden 参考系)。\n\
                 DFlash2 草稿须用 W4A16 量化家族(e.g. syvai/Qwen3.8-27B-DFlash2-W4A16)",
                dir.display()
            )));
        }
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

    /// 反量化口(测试/对拍用):量化基线性 → f32(dequant_u4 精确展宽)
    pub fn dequant_linear(&self, base: &str) -> Option<Vec<f32>> {
        let (out, k, pe, se) = self.linear_dims(base)?;
        let packed_bytes = &self.maps[pe.map_ix][pe.start..pe.start + pe.nbytes];
        let packed = i32s_par(packed_bytes);
        let scale_bytes = &self.maps[se.map_ix][se.start..se.start + se.nbytes];
        let scales_f32 = scales_par(se.dtype, scale_bytes);
        let mut dst = vec![0u8; out * k * 2];
        dequant_u4_affine_f16_bytes(&packed, &scales_f32, out, k, &mut dst);
        Some(
            dst.chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
        )
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
        // DFlash2 fc 列块五拆派生(E5-DF3.5):`fc_{i}.qweight/.scales` =
        // fc.weight_packed/scale 列块 i(in 维 [i*5120,(i+1)*5120);
        // packed [out, k/8] 列连续 ↔ in 连续,块宽 640/40)
        if let Some(base) = key.strip_suffix(".qweight") {
            if let Some(i) = fc_block_index(base) {
                return Some(Resolved::Bytes(self.fc_block_bytes(i, false)?, Dtype::U32));
            }
        }
        if let Some(base) = key.strip_suffix(".scales") {
            if let Some(i) = fc_block_index(base) {
                return Some(Resolved::Bytes(self.fc_block_bytes(i, true)?, Dtype::F16));
            }
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

    /// fc 列块物化(E5-DF3.5):fc.weight_packed [5120, 3200] I32 列子阵
    /// + fc.weight_scale [5120, 200] F16 列子阵 → (out=5120, k=5120) 走
    /// 与 build_linear 同式的 marlin 重排(qweight U32 / scales F16)。
    /// 列块恒 eligible(n=5120 % 256 == 0,k=5120 % 128 == 0)。
    fn fc_block_bytes(&self, blk: usize, scales_side: bool) -> Option<Arc<[u8]>> {
        let vkey = format!("fc_block_{blk}_{}", if scales_side { "s" } else { "q" });
        {
            let cache = self.cache.lock().unwrap();
            if let Some(b) = cache.0.get(&vkey) {
                return Some(b.clone());
            }
        }
        let pe = self.index.get("fc.weight_packed")?;
        let se = self.index.get("fc.weight_scale")?;
        let out = *pe.shape.first()?;
        let packed_cols = *pe.shape.get(1)?; // k_all / 8 = 3200
        let n_fc = 5usize;
        let cols = packed_cols / n_fc; // 640
        let scol = *se.shape.get(1)? / n_fc; // 40
        let _g = self.build_lock.lock().unwrap();
        let packed_bytes = &self.maps[pe.map_ix][pe.start..pe.start + pe.nbytes];
        let packed_all = i32s_par(packed_bytes);
        let scale_bytes = &self.maps[se.map_ix][se.start..se.start + se.nbytes];
        let scales_all = scales_par(se.dtype, scale_bytes);
        // 列子阵提取(packed 行主序 [out, 3200] → 行内 [blk*640, +640))
        let k = cols * 8 / n_fc * n_fc / n_fc; // 5120 = cols*8(单块)
        let k_blk = cols * 8; // = 5120
        let _ = k;
        let mut packed = Vec::with_capacity(out * cols);
        for r in 0..out {
            packed.extend_from_slice(&packed_all[r * packed_cols + blk * cols..r * packed_cols + (blk + 1) * cols]);
        }
        let mut scales = Vec::with_capacity(out * scol);
        for r in 0..out {
            scales.extend_from_slice(&scales_all[r * (scol * n_fc) + blk * scol..r * (scol * n_fc) + (blk + 1) * scol]);
        }
        let out_b = out as u32;
        let _ = out_b;
        let buf: Arc<[u8]> = if scales_side {
            let s = pack_marlin_s(&scales, out, k_blk / 128);
            // BF16 激活 marlin 要求 s_type = BF16:f16 位型字节 → bf16 位型
            Arc::from(f16_bytes_to_bf16_bytes(&u16s_le_bytes(&s)).into_boxed_slice())
        } else {
            let mut q_buf: Vec<u8> = Vec::new();
            unpack_nibbles_into(&packed, out, k_blk, &mut q_buf);
            let idx = self
                .idx_cache
                .lock()
                .unwrap()
                .entry((k_blk, out))
                .or_insert_with(|| Arc::new(marlin_gather_indices(k_blk, out)))
                .clone();
            let words = out * k_blk / 8;
            let mut b_buf: Vec<i32> = Vec::new();
            pack_marlin_b_gather_into(&q_buf, &idx, words, &mut b_buf);
            Arc::from(i32s_le_bytes(&b_buf).into_boxed_slice())
        };
        let mut cache = self.cache.lock().unwrap();
        cache.0.insert(vkey.clone(), buf.clone());
        cache.1 += buf.len();
        Some(buf)
    }

    /// 量化线性物化(键首触才建;qweight/scales 成对,weight 独立)。
    /// 双检缓存(快路径锁内查;构建在 build_lock 串行内,防并发双建)。
    fn linear_bytes_for(&self, base: &str, key: &str) -> Option<Arc<[u8]>> {
        // 刀2 虚拟合并:in_proj_qkvz = row-stack(in_proj_qkv, in_proj_z)
        // (同 k 行堆叠 = 字节拼接;两子基各自走完整建管)
        if let Some((b1, b2)) = crate::formats::split_qkvz(base) {
            let suffix = &key[base.len()..];
            let s1 = self.linear_bytes_for(&b1, &format!("{b1}{suffix}"))?;
            let s2 = self.linear_bytes_for(&b2, &format!("{b2}{suffix}"))?;
            let mut buf = Vec::with_capacity(s1.len() + s2.len());
            buf.extend_from_slice(&s1);
            buf.extend_from_slice(&s2);
            return Some(buf.into());
        }
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
        let scales_f32 = scales_par(se.dtype, scale_bytes);
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
                // BF16 激活 marlin:s_type = BF16(位型转换,见 f16_bytes_to_bf16_bytes)
                Arc::from(f16_bytes_to_bf16_bytes(&u16s_le_bytes(&s)).into_boxed_slice()),
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
        // fc 列块派生(E5-DF3.5):fc_{i}.qweight = out*640 U32 /
        // fc_{i}.scales = out*40 F16;fc_{i}.weight = out*5120 F16(反量化)
        if let Some(base) = key.strip_suffix(".qweight") {
            if fc_block_index(base).is_some() {
                return Some(5120 * 640);
            }
        }
        if let Some(base) = key.strip_suffix(".scales") {
            if fc_block_index(base).is_some() {
                return Some(5120 * 40);
            }
        }
        if let Some(base) = key.strip_suffix(".weight") {
            if fc_block_index(base).is_some() {
                return Some(5120 * 5120);
            }
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
                // F16 want(存量主路径)/ BF16 want(E5-DF3 同日十四;DFlash2
                // 草稿 BF16 化 —— passthrough 权重原生 BF16,直拷零转换,
                // sglang .to(bf16) 同位)
                if dtype != Dtype::F16 && dtype != Dtype::BF16 {
                    return None;
                }
                let src_esz = match e.dtype {
                    safetensors::Dtype::BF16 | safetensors::Dtype::F16 => 2,
                    safetensors::Dtype::F32 => 4,
                    _ => return None,
                };
                let s = e.start + offset_elems * src_esz;
                let win = &self.maps[e.map_ix][s..s + len * src_esz];
                match (e.dtype, dtype) {
                    (safetensors::Dtype::F16, Dtype::F16) => dst[..need].copy_from_slice(win),
                    (safetensors::Dtype::BF16, Dtype::F16) => {
                        bf16_bytes_to_f16_bytes(win, &mut dst[..need])
                    }
                    (safetensors::Dtype::F16, Dtype::BF16) => {
                        let conv = f16_bytes_to_bf16_bytes(win);
                        dst[..need].copy_from_slice(&conv);
                    }
                    (safetensors::Dtype::BF16, Dtype::BF16) => dst[..need].copy_from_slice(win),
                    (safetensors::Dtype::F32, Dtype::F16) => {
                        // F32 passthrough 仅 GDN 标量小张量,标量循环足够
                        for (i, c) in win.chunks_exact(4).enumerate() {
                            let f = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                            dst[i * 2..i * 2 + 2]
                                .copy_from_slice(&half::f16::from_f32(f).to_le_bytes());
                        }
                    }
                    (safetensors::Dtype::F32, Dtype::BF16) => {
                        for (i, c) in win.chunks_exact(4).enumerate() {
                            let f = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                            dst[i * 2..i * 2 + 2]
                                .copy_from_slice(&half::bf16::from_f32(f).to_le_bytes());
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
// 字节转换助手(rayon;对齐安全 from_le_bytes)—— pub(crate):awq.rs 共享
// ============================================================================

pub(crate) fn i32s_par(bytes: &[u8]) -> Vec<i32> {
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

/// f16 位型字节 → BF16 位型字节(E5-DF3 同日十二:syvai W4A16 的 scales
/// = F16,而 BF16 激活的 marlin 实例要求 s_type = BF16;值经 f32 中转,
/// 尾数 10→7 位 ≈ 0.4% 相对误差,远小于量化噪声)。
fn f16_bytes_to_bf16_bytes(bytes: &[u8]) -> Vec<u8> {
    bytes
        .chunks_exact(2)
        .map(|c| {
            half::bf16::from_f32(half::f16::from_le_bytes([c[0], c[1]]).to_f32()).to_le_bytes()
        })
        .flatten()
        .collect::<Vec<u8>>()
}

pub(crate) fn bf16_par(bytes: &[u8]) -> Vec<f32> {
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

/// F16 平行读(2026-10-07 E5-DF3 定谳:syvai W4A16 的 weight_scale = F16
/// 标注 + F16 字节,曾全走 bf16_par 按 BF16 位型读 → 草稿全部 W4A16 权重
/// = 噪声 → 草稿前向废 → AL=0。scale 读法必须按 index 声明 dtype 分派。
/// 对照检验:两种读法对 z-lab BF16 源 fc 权重做相关,F16 读 0.977 /
/// BF16 读 0.016。)
pub(crate) fn f16_par(bytes: &[u8]) -> Vec<f32> {
    use rayon::prelude::*;
    let n = bytes.len() / 2;
    let mut v = vec![0f32; n];
    v.par_chunks_mut(1 << 16)
        .enumerate()
        .for_each(|(t, chunk)| {
            for (i, cell) in chunk.iter_mut().enumerate() {
                let o = (t * (1 << 16) + i) * 2;
                *cell = half::f16::from_le_bytes([bytes[o], bytes[o + 1]]).to_f32();
            }
        });
    v
}

/// scale 平行读分派:按 RawEntry 声明 dtype 选 F16/BF16 位型
pub(crate) fn scales_par(dt: safetensors::Dtype, bytes: &[u8]) -> Vec<f32> {
    match dt {
        safetensors::Dtype::F16 => f16_par(bytes),
        _ => bf16_par(bytes),
    }
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

pub(crate) fn i32s_le_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub(crate) fn u16s_le_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// DFlash2 fc 列块键解析:`fc_{i}`(i < 5)→ Some(i);其余 None。
fn fc_block_index(base: &str) -> Option<usize> {
    let idx = base.strip_prefix("fc_")?;
    if idx.len() != 1 || !idx.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let i = idx.parse::<usize>().ok()?;
    (i < 5).then_some(i)
}

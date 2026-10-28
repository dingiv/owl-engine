//! safetensors 权重源(f16 基线;F32/BF16 → f16/f32,流式分块)。
//! 底座(mmap/索引)见 [`crate::formats::mmap`]。
//!
//! **流式形态(2026-09-26 M-e 性能,用户裁决:读一点装一点)**:
//! - 文件 = libc::mmap 只读映射(打开 27µs;页缓存支撑零堆拷贝);
//! - 张量只登记**条目索引**(字节区间 + dtype,堆上 KB 级);
//! - `take(key)` = 查到才从映射区转换出 f32 **并取走所有权** ——
//!   上传后随调用方消亡,主机驻留只剩"在途"份(0.8B 全量常驻 3.4GB
//!   的旧形态作废;visual/mtp 153 张量永不转换,0.5GB 根本不发生)。
//!   量化格式另立 source,勿在此堆积。
//!
//! 形状事实:conv1d 等 3D 权重按扁平字节直读(row-major 连续,
//! [6144,1,4] ≡ [6144,4]),shape 元数据由层侧 Want 声明,此处不搬。

use crate::contract::ModelError;
use crate::formats::mmap::{open_raw_index, Mmap};
use crate::module::{Taken, WeightSource};
use half::slice::{HalfBitsSliceExt, HalfFloatSliceExt};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

// 转换热路径(独立微 crate;debug 档 profile 覆盖 opt-level=3,
// 见 crate 根注释与 workspace Cargo.toml)
use crate::f16c::{bf16_bytes_to_f16_bytes, f32_bytes_to_f16_bytes};

/// 流式权重源:mmap + 条目索引(堆上 KB 级),数据按需转换、取走即弃。
pub struct SafeTensorsSource {
    maps: Vec<Mmap>,
    /// 待取条目(take 即移除;锁只护索引,转换在锁外读映射区)
    index: Mutex<HashMap<String, Entry>>,
}

/// 张量条目(映射区内字节区间 + dtype + **形状**)
///
/// 形状为何要留:`mtp.fc_{e,h}` 两个**虚拟键**是 `mtp.fc.weight` 的
/// in 维前后半(列半拆),拆法必须知道行宽 —— 只靠 `nbytes` 拆不出来。
#[derive(Clone)]
struct Entry {
    map_ix: usize,
    start: usize,
    nbytes: usize,
    dtype: safetensors::Dtype,
    shape: Vec<usize>,
}

impl Entry {
    /// 元素宽(字节):F32=4,BF16=2
    fn esz(&self) -> usize {
        match self.dtype {
            safetensors::Dtype::F32 => 4,
            safetensors::Dtype::BF16 => 2,
            _ => unreachable!("open_dir 只登记 F32/BF16"),
        }
    }
}



impl SafeTensorsSource {
    /// **索引锁的唯一取用口**(2026-10-17 审查 §④ 档③:锁中毒 = 响亮拒)。
    ///
    /// 为何与本文件其它缓存不同:本索引是**消费式状态**(`take` 会 `remove`)
    /// —— 持锁线程 panic 可能恰好落在"已摘键、未返回"之间 ⇒ 就地恢复
    /// (`into_inner`)会把"已经丢了一半的键"继续用(静默错值 > 崩溃),
    /// 所以这里选**结构化 Err**(由装载入口 `?` 上抛到泵循环收摊)。
    fn idx(&self) -> Result<std::sync::MutexGuard<'_, HashMap<String, Entry>>, ModelError> {
        self.index.lock().map_err(|_| {
            ModelError::Msg(
                "safetensors 索引锁中毒:此前有线程持锁 panic —— 索引是消费式状态\
                 (take 会摘键),可能已丢键,故不许就地恢复;本实例不再可用"
                    .into(),
            )
        })
    }

    /// 刀2 虚拟合并键面:in_proj_qkvz.weight = row-stack(in_proj_qkv.weight,
    /// in_proj_z.weight)。返回 (子键1, 元素数1, 子键2, 元素数2);非合并键 None。
    fn merged_parts(&self, key: &str) -> Taken<(String, usize, String, usize)> {
        let Some(base) = key.strip_suffix(".weight") else {
            return Ok(None);
        };
        let Some((b1, b2)) = crate::formats::split_merged(base) else {
            return Ok(None);
        };
        let idx = self.idx()?;
        let (k1, k2) = (format!("{b1}.weight"), format!("{b2}.weight"));
        let (Some(e1), Some(e2)) = (idx.get(&k1), idx.get(&k2)) else {
            return Ok(None);
        };
        Ok(Some((k1, e1.nbytes / e1.esz(), k2, e2.nbytes / e2.esz())))
    }

    /// 打开目录下全部 `*.safetensors`(共享底座 [`open_raw_index`]),
    /// 叠加 f16 基线的 dtype 校验(F32/BF16)。只建映射 + 索引,零数据拷贝。
    pub fn open_dir(dir: &Path) -> Result<Self, ModelError> {
        let (maps, raw) = open_raw_index(dir)?;
        let mut index = HashMap::with_capacity(raw.len());
        for (name, e) in raw {
            if !matches!(e.dtype, safetensors::Dtype::F32 | safetensors::Dtype::BF16) {
                return Err(ModelError::Msg(format!(
                    "safetensors: {name} dtype {:?} 不支持(量化另立 source)",
                    e.dtype
                )));
            }
            index.insert(
                name,
                Entry {
                    map_ix: e.map_ix,
                    start: e.start,
                    nbytes: e.nbytes,
                    dtype: e.dtype,
                    shape: e.shape.clone(),
                },
            );
        }
        Ok(Self { maps, index: Mutex::new(index) })
    }

    pub fn len(&self) -> Result<usize, ModelError> {
        Ok(self.idx()?.len())
    }

    pub fn is_empty(&self) -> Result<bool, ModelError> {
        Ok(self.len()? == 0)
    }
}

// ============================================================================
// 转换热路径(手写 F16C;debug 档也保 GB/s 级)
// ============================================================================
// 微基准定谳(2026-09-26):half 批量转换 release 4-7.7GB/s,但 debug 仅
// 78-153MB/s(8 宽循环包装开销),而装载测试跑 debug —— 转换段曾是装载
// 的大头(探针:convert≈17ms/MB 恒定)。这里用裸 intrinsic 循环(指令即
// 循环体,无 debug 包装税);舍入语义 RNE 与 half::from_f32 一致(checksum
// 逐位门可证)。server 侧逐命令计时同时定谳:装载全程 server 仅 ~355ms,
// 瓶颈全在客户端 CPU。

impl SafeTensorsSource {
    /// 条目区间 → f32(不持锁;F32 直读 / BF16 升位)。
    /// 消费完立即 `madvise(MADV_DONTNEED)` 归还映射页 —— 进程 RSS
    /// 恒定在"在途块"量级,不随仓大小增长(读一点装一点)。
    fn convert_range(&self, e: &Entry, offset_elems: usize, len: usize) -> Vec<f32> {
        let esz = e.esz();
        let s = e.start + offset_elems * esz;
        let nbytes = len * esz;
        let bytes = &self.maps[e.map_ix][s..s + nbytes];
        let mut converted = vec![0f32; len];
        // 大块分线程解码(F5 尾批:与 load.rs 并行转换同款;≥8MB 4 线程)
        const PAR_THRESHOLD: usize = 8 << 20;
        if nbytes >= PAR_THRESHOLD {
            let chunk = len.div_ceil(4);
            let dst_slices: Vec<&mut [f32]> = converted.chunks_mut(chunk).collect();
            let src_slices: Vec<&[u8]> = bytes.chunks(chunk * esz).collect();
            std::thread::scope(|scope| {
                for (dst_slice, src_slice) in dst_slices.into_iter().zip(src_slices) {
                    scope.spawn(move || match e.dtype {
                        safetensors::Dtype::F32 => {
                            for (i, c) in src_slice.chunks_exact(4).enumerate() {
                                dst_slice[i] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                            }
                        }
                        safetensors::Dtype::BF16 => {
                            for (i, c) in src_slice.chunks_exact(2).enumerate() {
                                dst_slice[i] = f32::from_bits(
                                    (u16::from_le_bytes([c[0], c[1]]) as u32) << 16,
                                );
                            }
                        }
                        _ => {}
                    });
                }
            });
        } else {
            let converted = &mut converted;
            match e.dtype {
                safetensors::Dtype::F32 => {
                    for (i, c) in bytes.chunks_exact(4).enumerate() {
                        converted[i] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                    }
                }
                safetensors::Dtype::BF16 => {
                    for (i, c) in bytes.chunks_exact(2).enumerate() {
                        converted[i] =
                            f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16);
                    }
                }
                _ => unreachable!("open_dir 只登记 F32/BF16"),
            }
        }
        self.maps[e.map_ix].dontneed(s, nbytes);
        converted
    }
}

impl SafeTensorsSource {
    /// 索引键存在性(源族探测用;E5-DF3.5)
    pub fn has_key(&self, key: &str) -> Result<bool, ModelError> {
        Ok(self.idx()?.contains_key(key))
    }

    /// **MTP 头是否自带**(2026-10-28 propose 进图刀):两种落盘形态都认 ——
    /// 融合 `mtp.fc.weight`(HF 原样)或已拆好的 `mtp.fc_e/h.weight`。
    /// 调用方据此决定 `mtp_dir`(同一目录当草稿源)。
    pub fn has_mtp_head(&self) -> Result<bool, ModelError> {
        let idx = self.idx()?;
        Ok(idx.contains_key("mtp.fc.weight") || idx.contains_key("mtp.fc_e.weight"))
    }

    /// **fc 列半拆虚拟键**:`mtp.fc_{e,h}.weight` → (`mtp.fc.weight`, `high`)
    ///
    /// 语义:`mtp.fc.weight` = `[out, 2·hidden]` 行主序,`fc_e` 吃 embedding
    /// 侧、`fc_h` 吃 hidden 侧 ⇒ 按 **in 维前后半**拆(`W [out, 2h]` → 两块
    /// `[out, h]`)。与 AWQ 源的同名派生键**同语义**(那里是打包字节形态)。
    ///
    /// 为何不允许"用整张 `mtp.fc` 跑一条 GEMM":两半的**输入不同**(embedding /
    /// residual),融合需要每轮 concat;拆开装载 = 零运行时开销,且与 27B/GGUF
    /// 两条已定谳路径同形。
    fn fc_half(&self, key: &str) -> Taken<(Entry, bool)> {
        let high = match key {
            "mtp.fc_e.weight" => false,
            "mtp.fc_h.weight" => true,
            _ => return Ok(None),
        };
        let Some(e) = self.idx()?.get("mtp.fc.weight").cloned() else {
            return Ok(None);
        };
        if e.shape.len() != 2 || e.shape[1] % 2 != 0 {
            return Ok(None); // 形态不对 ⇒ 按"缺键"报(不猜)
        }
        Ok(Some((e, high)))
    }

    /// fc 半张量的 `[offset, offset+len)` 元素区间 → f32(行内不去交织)
    fn fc_half_range(
        &self,
        e: &Entry,
        high: bool,
        offset_elems: usize,
        len: usize,
    ) -> Option<Vec<f32>> {
        let (out, din) = (e.shape.first().copied()?, e.shape.get(1).copied()?);
        let half = din / 2;
        let total = out * half;
        if half == 0 || offset_elems + len > total {
            return None;
        }
        let esz = e.esz();
        let src = &self.maps[e.map_ix][e.start..e.start + e.nbytes];
        let skip = if high { half } else { 0 };
        let mut v = vec![0f32; len];
        let mut i = 0usize;
        while i < len {
            let flat = offset_elems + i;
            let (r, c) = (flat / half, flat % half);
            let n = (half - c).min(len - i);
            let base = (r * din + skip + c) * esz;
            let chunk = &src[base..base + n * esz];
            match e.dtype {
                safetensors::Dtype::F32 => {
                    for (k, c4) in chunk.chunks_exact(4).enumerate() {
                        v[i + k] = f32::from_le_bytes([c4[0], c4[1], c4[2], c4[3]]);
                    }
                }
                safetensors::Dtype::BF16 => {
                    for (k, c2) in chunk.chunks_exact(2).enumerate() {
                        v[i + k] = f32::from_bits((u16::from_le_bytes([c2[0], c2[1]]) as u32) << 16);
                    }
                }
                _ => return None,
            }
            i += n;
        }
        // 读一点装一点(与 `convert_range` 同律):MTP 头 fc 是 2×hidden 宽的
        // 大件,不去页就会把整段留在 RSS 里(小量,但律不破)。
        self.maps[e.map_ix].dontneed(e.start, e.nbytes);
        Some(v)
    }
}

impl WeightSource for SafeTensorsSource {
    /// 天然 dtype(F32/BF16;合并虚拟键 = None —— 原始通道不做行拼接)
    fn native_dtype(&self, key: &str) -> Taken<crate::contract::Dtype> {
        if self.merged_parts(key)?.is_some() || self.fc_half(key)?.is_some() {
            return Ok(None);
        }
        let Some(e) = self.idx()?.get(key).cloned() else {
            return Ok(None);
        };
        Ok(match e.dtype {
            safetensors::Dtype::F32 => Some(crate::contract::Dtype::F32),
            safetensors::Dtype::BF16 => Some(crate::contract::Dtype::BF16),
            _ => None,
        })
    }

    /// 原始字节直拷(mmap → dst;**纯 memcpy,零元素转换**)。
    /// 消费完 `madvise(DONTNEED)` 归还映射页(与 convert_range 同律)。
    fn copy_native_into(
        &self,
        key: &str,
        offset_elems: usize,
        len: usize,
        dst: &mut [u8],
    ) -> Taken<()> {
        if self.merged_parts(key)?.is_some() || self.fc_half(key)?.is_some() {
            return Ok(None);
        }
        let Some(e) = self.idx()?.get(key).cloned() else {
            return Ok(None);
        };
        let esz = e.esz();
        // 边界判定统一走 **checked**(2026-10-17 审查 §四轮-C5:原先本函数用
        // 普通算式、`awq::copy_native_into` 用 checked —— 同一语义两套写法)
        let (Some(want), Some(end)) = (
            len.checked_mul(esz),
            offset_elems
                .checked_add(len)
                .and_then(|n| n.checked_mul(esz)),
        ) else {
            return Ok(None);
        };
        if end > e.nbytes || dst.len() < want {
            return Ok(None);
        }
        let s = e.start + offset_elems * esz;
        let src = &self.maps[e.map_ix][s..s + want];
        // **大块分线程 memcpy**(公共口 `mmap::copy_par`:单线程 ~19GB/s、
        // 四线程 ~55GB/s,CPU 侧不再当墙 —— DMA 才是应然瓶颈)
        crate::formats::mmap::copy_par(src, dst);
        self.maps[e.map_ix].dontneed(s, want);
        Ok(Some(()))
    }

    fn elem_len(&self, key: &str) -> Taken<usize> {
        if let Some((_, n1, _, n2)) = self.merged_parts(key)? {
            return Ok(Some(n1 + n2));
        }
        if let Some((e, _)) = self.fc_half(key)? {
            return Ok(Some(e.shape[0] * (e.shape[1] / 2)));
        }
        let Some(e) = self.idx()?.get(key).cloned() else {
            return Ok(None);
        };
        Ok(Some(e.nbytes / e.esz()))
    }

    /// 整取(F32 直读 / BF16 升位),条目即从索引移除。
    /// 流式装载主路径走 `take_range`(分块,条目保留)。
    fn take(&self, key: &str) -> Taken<Vec<f32>> {
        if let Some((e, high)) = self.fc_half(key)? {
            // **不摘物理键**(同源另一半还要用)
            let n = e.shape[0] * (e.shape[1] / 2);
            return Ok(self.fc_half_range(&e, high, 0, n));
        }
        if let Some((k1, _, k2, _)) = self.merged_parts(key)? {
            let Some(mut v) = self.take(&k1)? else {
                return Ok(None);
            };
            let Some(tail) = self.take(&k2)? else {
                return Ok(None);
            };
            v.extend(tail);
            return Ok(Some(v));
        }
        // remove 与取值同锁内完成(消费式;锁中毒 = 响亮拒,见 `idx`)
        let mut idx = self.idx()?;
        let Some(e) = idx.remove(key) else {
            return Ok(None);
        };
        drop(idx);
        // 元素数按条目真实位宽(F5 修:原 /4 硬编码对 bf16 条目
        // 返回一半元素 —— 直转路径首次踩中;分块路径按 len 显式未暴露)
        Ok(Some(self.convert_range(&e, 0, e.nbytes / e.esz())))
    }

    /// 区间转换(mmap 直读,**不移除条目** —— 分块上传同键多块重复取)
    fn take_range(&self, key: &str, offset_elems: usize, len: usize) -> Taken<Vec<f32>> {
        if let Some((e, high)) = self.fc_half(key)? {
            return Ok(self.fc_half_range(&e, high, offset_elems, len));
        }
        if let Some((k1, n1, k2, _)) = self.merged_parts(key)? {
            // 行堆叠拼接:子1 元素区 [0,n1),子2 [n1,n1+n2)
            if offset_elems + len <= n1 {
                return self.take_range(&k1, offset_elems, len);
            }
            if offset_elems >= n1 {
                return self.take_range(&k2, offset_elems - n1, len);
            }
            let Some(head) = self.take_range(&k1, offset_elems, n1 - offset_elems)? else {
                return Ok(None);
            };
            let Some(tail) = self.take_range(&k2, 0, offset_elems + len - n1)? else {
                return Ok(None);
            };
            let mut v = head;
            v.extend(tail);
            return Ok(Some(v));
        }
        let Some(e) = self.idx()?.get(key).cloned() else {
            return Ok(None);
        };
        if (offset_elems + len) * e.esz() > e.nbytes {
            return Ok(None);
        }
        Ok(Some(self.convert_range(&e, offset_elems, len)))
    }

    /// 分块转换直写字节租约(F16 基线):mmap 直解码单 pass 直写 dst
    /// (crate::f16c F16C 通道),零中间 Vec、零手写线程;块页读毕 DONTNEED
    /// (读一点装一点,RSS 恒定在在途块量级)。主路径(直接臂)专用。
    fn convert_chunk_into_bytes(
        &self,
        key: &str,
        offset_elems: usize,
        len: usize,
        dst: &mut [u8],
        dtype: crate::contract::Dtype,
    ) -> Taken<()> {
        // **fc 列半拆虚拟键**(2026-10-28):行内去交织 + 目标 dtype 转换
        if let Some((e, high)) = self.fc_half(key)? {
            let Some(v) = self.fc_half_range(&e, high, offset_elems, len) else {
                return Ok(None);
            };
            match dtype {
                crate::contract::Dtype::F32 => {
                    if dst.len() < len * 4 {
                        return Ok(None);
                    }
                    for (i, x) in v.iter().enumerate() {
                        dst[i * 4..i * 4 + 4].copy_from_slice(&x.to_le_bytes());
                    }
                }
                crate::contract::Dtype::F16 => {
                    if dst.len() < len * 2 {
                        return Ok(None);
                    }
                    crate::f16c::f32_slice_to_f16_bytes(&v, &mut dst[..len * 2]);
                }
                crate::contract::Dtype::BF16 => {
                    if dst.len() < len * 2 {
                        return Ok(None);
                    }
                    for (i, x) in v.iter().enumerate() {
                        dst[i * 2..i * 2 + 2]
                            .copy_from_slice(&half::bf16::from_f32(*x).to_le_bytes());
                    }
                }
                _ => return Ok(None),
            }
            return Ok(Some(()));
        }
        // 刀2 合并键面:跨界块拆两子键分别直写(行堆叠拼接)
        if let Some((k1, n1, k2, _)) = self.merged_parts(key)? {
            let out_esz = match dtype {
                crate::contract::Dtype::F32 => 4,
                crate::contract::Dtype::F16 => 2,
                _ => return Ok(None),
            };
            if offset_elems + len <= n1 {
                return self.convert_chunk_into_bytes(&k1, offset_elems, len, dst, dtype);
            }
            if offset_elems >= n1 {
                return self.convert_chunk_into_bytes(&k2, offset_elems - n1, len, dst, dtype);
            }
            let head_elems = n1 - offset_elems;
            let head_bytes = head_elems * out_esz;
            // 两半均须已写(任一 `Ok(None)` = 子键不支持 ⇒ 整体回退)
            let h = self.convert_chunk_into_bytes(&k1, offset_elems, head_elems, &mut dst[..head_bytes], dtype)?;
            if h.is_none() {
                return Ok(None);
            }
            return self.convert_chunk_into_bytes(&k2, 0, len - head_elems, &mut dst[head_bytes..], dtype);
        }
        let Some(e) = self.idx()?.get(key).cloned() else {
            owl_shared::log_info!("st", "[DIAG] 索引未命中: {key}");
            return Ok(None);
        };
        let esz = e.esz();
        if (offset_elems + len) * esz > e.nbytes {
            owl_shared::log_info!("st", "[DIAG] OOB: {key} off{offset_elems} len{len} esz{esz} nbytes{}",
                e.nbytes
            );
            return Ok(None);
        }
        let s = e.start + offset_elems * esz;
        let bytes = &self.maps[e.map_ix][s..s + len * esz];
        // 目标缓冲尺寸不足 = **调用方违约** ⇒ 响亮拒(2026-10-17 审查 §③:
        // 旧版逐臂返回 None ⇒ 被上层收成"数据源缺键",假归因)
        let need = match dtype {
            crate::contract::Dtype::F32 => len * 4,
            crate::contract::Dtype::F16 | crate::contract::Dtype::BF16 => len * 2,
            _ => return Ok(None),
        };
        if dst.len() < need {
            return Err(ModelError::Msg(format!(
                "safetensors::convert_chunk_into_bytes: dst {}B < 需 {need}B({dtype:?}, len {len})",
                dst.len()
            )));
        }
        match (e.dtype, dtype) {
            // F32 源 → F32 目标：LE 直拷（零转换）
            (safetensors::Dtype::F32, crate::contract::Dtype::F32) => {
                dst[..len * 4].copy_from_slice(bytes);
            }
            // F32 源 → f16 目标：F16C 单 pass 直写（舍入 RNE 与 from_f32 一致）
            // —— 走**字节口**(源是 mmap 条目切片,偏移不保证 4 对齐:
            // 2026-10-17 审查,原 `from_raw_parts::<f32>` = UB)
            (safetensors::Dtype::F32, crate::contract::Dtype::F16) => {
                f32_bytes_to_f16_bytes(bytes, &mut dst[..len * 2]);
            }
            // BF16 源 → f16 目标：F16C 单 pass 直写（bf16→f32 精确展宽 +
            // RNE 舍入，与 checksum 锚逐位一致）
            (safetensors::Dtype::BF16, crate::contract::Dtype::F16) => {
                bf16_bytes_to_f16_bytes(bytes, &mut dst[..len * 2]);
            }
            // F32 源 → BF16 目标：完备化臂(2026-10-17 审查：矩阵原缺此格，
            // 而兄弟源 w4a16 早有 `(F32, BF16)` —— 混 dtype 检查点(裸
            // weight 里夹 F32 标量张量)装进 BF16 链时会落到 `_ => None`，
            // 表现为“数据源缺键”的假归因)。RNE 与 `f16c` 的 f16→bf16 同式。
            (safetensors::Dtype::F32, crate::contract::Dtype::BF16) => {
                for (i, c) in bytes.chunks_exact(4).enumerate() {
                    let f = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                    dst[i * 2..i * 2 + 2].copy_from_slice(&half::bf16::from_f32(f).to_le_bytes());
                }
            }
            // BF16 源 → BF16 目标：直拷零转换(E5-DF3 同日十四;DFlash2 草稿
            // BF16 全程 —— passthrough 权重原生直载,sglang 同位)
            (safetensors::Dtype::BF16, crate::contract::Dtype::BF16) => {
                dst[..len * 2].copy_from_slice(bytes);
            }
            // F16 源 → BF16 目标:值转换(E5-DF3 同日十四安全网;主路径 =
            // BF16 原生源直拷)
            (safetensors::Dtype::F16, crate::contract::Dtype::BF16) => {
                crate::f16c::f16_bytes_to_bf16_bytes(bytes, &mut dst[..len * 2]);
            }
            // BF16 源 → f32 目标（f32 链回退路径）：批量展开
            (safetensors::Dtype::BF16, crate::contract::Dtype::F32) => {
                let (_, mid, _) = unsafe { dst[..len * 4].align_to_mut::<f32>() };
                let (_, smid, _) = unsafe { bytes.align_to::<u16>() };
                if mid.len() == len && smid.len() == len {
                    smid.reinterpret_cast::<half::bf16>()
                        .convert_to_f32_slice(mid);
                } else {
                    for (i, c) in bytes.chunks_exact(2).enumerate() {
                        let f = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16);
                        dst[i * 4..i * 4 + 4].copy_from_slice(&f.to_le_bytes());
                    }
                }
            }
            _ => return Ok(None),
        }
        self.maps[e.map_ix].dontneed(s, len * esz);
        Ok(Some(()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::Dtype;
    use std::path::Path;

    /// 合成单条目 safetensors(与 awq 测试的写盘法同款：u64 LE 头长 + JSON + 数据)
    fn write_st(dir: &Path, entry_dtype: &str, shape: &[usize], data: &[u8]) {
        owl_shared::file_loader::create_dir_all(dir).expect("mkdir");
        let header = format!(
            "{{\"w.weight\": {{\"dtype\": \"{entry_dtype}\", \"shape\": [{}], \"data_offsets\": [0, {}]}}}}",
            shape.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","),
            data.len()
        );
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header.as_bytes());
        file.extend_from_slice(data);
        owl_shared::file_loader::write(dir.join("model.safetensors"), file).expect("写 st");
    }

    const VALS: [f32; 4] = [1.0, -2.5, 0.0, 3.25];

    /// 写盘(可指定头尾补白字节数 —— 用来**故意**造非 4 对齐的数据区起点)
    fn write_st_padded(dir: &Path, entry_dtype: &str, data: &[u8], pad: usize) {
        owl_shared::file_loader::create_dir_all(dir).expect("mkdir");
        let mut header = format!(
            "{{\"w.weight\": {{\"dtype\": \"{entry_dtype}\", \"shape\": [2, 2], \"data_offsets\": [0, {}]}}}}",
            data.len()
        );
        header.push_str(&" ".repeat(pad));
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header.as_bytes());
        file.extend_from_slice(data);
        owl_shared::file_loader::write(dir.join("model.safetensors"), file).expect("写 st");
    }

    fn src_bytes(dt: &str) -> Vec<u8> {
        VALS.iter()
            .flat_map(|v| match dt {
                "F32" => v.to_le_bytes().to_vec(),
                "BF16" => half::bf16::from_f32(*v).to_le_bytes().to_vec(),
                "F16" => half::f16::from_f32(*v).to_le_bytes().to_vec(),
                other => panic!("未知源 dtype {other}"),
            })
            .collect()
    }

    fn want_bytes(dt: Dtype) -> Vec<u8> {
        VALS.iter()
            .flat_map(|v| match dt {
                Dtype::F32 => v.to_le_bytes().to_vec(),
                Dtype::BF16 => half::bf16::from_f32(*v).to_le_bytes().to_vec(),
                Dtype::F16 => half::f16::from_f32(*v).to_le_bytes().to_vec(),
                other => panic!("未知目标 dtype {other:?}"),
            })
            .collect()
    }

    /// **dtype 矩阵对白名单封闭**(2026-10-17 审查 §C4 的落地门)：
    /// 本源开箱只放行 F32/BF16(白名单)，而目标 dtype 由调用方选(F16 基线 /
    /// F32 链 / BF16 直通) ⇒ **白名单 × 三目标 = 6 格必须全部可达**。
    /// 缺格 = “开箱通过、装载才炸”，且报的是“数据源缺键”这种假归因。
    #[test]
    fn dtype_matrix_is_total_over_whitelist() {
        for src in ["F32", "BF16"] {
            for tgt in [Dtype::F32, Dtype::F16, Dtype::BF16] {
                let dir = crate::testkit::tmp_path(&format!("st_matrix_{src}_{tgt:?}"));
                let _ = owl_shared::file_loader::remove_file(&dir);
                let shape = [2usize, 2];
                write_st(&dir, src, &shape, &src_bytes(src));
                let s = SafeTensorsSource::open_dir(&dir).expect("开箱");
                let mut dst = vec![0u8; 4 * tgt.size_bytes()];
                let got = crate::module::WeightSource::convert_chunk_into_bytes(
                    &s, "w.weight", 0, 4, &mut dst, tgt,
                )
                .expect("源");
                assert!(got.is_some(), "({src} → {tgt:?}) 这一格缺失(矩阵不满)");
                assert_eq!(dst, want_bytes(tgt), "({src} → {tgt:?}) 字节不符");
            }
        }
    }

    /// **非 4 对齐的源不得 UB**(2026-10-17 审查,实测):safetensors 规范
    /// 不保证条目 4 对齐(F8/奇数元素 BF16 条目、非 padding writer),而
    /// F32→F16 臂原先把条目切片裸铸成 `&[f32]` ⇒ 对齐前置条件违反 =
    /// **非捕获性 abort**(debug 档当场毙;由 dtype 矩阵门首次抓到)。
    /// 本门:头补白 2 字节 ⇒ 数据区 2 mod 4,断言**与对齐时逐位同值**。
    #[test]
    fn f32_to_f16_survives_unaligned_entry_offset() {
        for pad in [0usize, 2, 6] {
            let dir = crate::testkit::tmp_path(&format!("st_unaligned_{pad}"));
            let _ = owl_shared::file_loader::remove_file(&dir);
            write_st_padded(&dir, "F32", &src_bytes("F32"), pad);
            let s = SafeTensorsSource::open_dir(&dir).expect("开箱");
            let mut dst = vec![0u8; 4 * Dtype::F16.size_bytes()];
            let got = crate::module::WeightSource::convert_chunk_into_bytes(
                &s, "w.weight", 0, 4, &mut dst, Dtype::F16,
            )
            .expect("源");
            assert!(got.is_some(), "pad={pad}:(F32 → F16) 必须可达");
            assert_eq!(dst, want_bytes(Dtype::F16), "pad={pad}:值须与对齐时一致");
        }
    }

    /// **白名单的边界是响亮的**：F16 条目在**开箱期**结构化拒(不静默、不拖延到装载)
    /// —— 这条同时钉住“转换矩阵只覆盖白名单”这一前提(§dtype_matrix 的对应面)。
    #[test]
    fn f16_entry_rejected_loudly_at_open() {
        let dir = crate::testkit::tmp_path("st_f16_entry");
        let _ = owl_shared::file_loader::remove_file(&dir);
        write_st(&dir, "F16", &[2, 2], &src_bytes("F16"));
        let err = match SafeTensorsSource::open_dir(&dir) {
            Ok(_) => panic!("F16 条目必须拒"),
            Err(e) => e,
        };
        let msg = format!("{err}");
        assert!(msg.contains("F16"), "错误须点名 dtype：{msg}");
    }

    /// **锁中毒门(§④ 档③)**:索引是消费式状态(`take` 会摘键)⇒ 中毒后
    /// 必须**响亮拒**(结构化 Err),不许 panic、也不许"就地恢复"把丢了一半的
    /// 索引继续用。
    #[test]
    fn poisoned_index_is_loud_not_panic() {
        let dir = crate::testkit::tmp_path("st_poison");
        let _ = owl_shared::file_loader::remove_file(&dir);
        write_st(&dir, "F32", &[2, 2], &src_bytes("F32"));
        let s = SafeTensorsSource::open_dir(&dir).expect("开箱");
        // 故意持锁 panic(锁被毒化)
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _g = s.index.lock().unwrap();
            panic!("poison");
        }));
        assert!(r.is_err(), "本测试就是要毒化锁");
        // 后续一切取用 = 结构化 Err(不是 PoisonError panic,也不是假缺键)
        let e = match s.take("w.weight") {
            Ok(_) => panic!("中毒索引不许装作可用"),
            Err(e) => e,
        };
        assert!(format!("{e}").contains("中毒"), "{e}");
        assert!(matches!(s.elem_len("w.weight"), Err(_)), "elem_len 同样响亮");
    }
}

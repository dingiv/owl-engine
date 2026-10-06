//! safetensors 权重源(f16 基线;F32/BF16 → f16/f32,流式分块)。
//! 底座(mmap/索引)见 [`crate::formats::mmap`]。
//!
//! **流式形态(2026-09-26 M-e 性能,用户裁决:读一点装一点)**:
//! - 文件 = libc::mmap 只读映射(打开 27µs;页缓存支撑零堆拷贝);
//! - 张量只登记**条目索引**(字节区间 + dtype,堆上 KB 级);
//! - `take(key)` = 查到才从映射区转换出 f32 **并取走所有权** ——
//!   上传后随调用方消亡,主机驻留只剩"在途"份(0.8B 全量常驻 3.4GB
//!   的旧形态作废;visual/mtp 153 张量永不转换,0.5GB 根本不发生)。
//! 量化格式另立 source,勿在此堆积。
//!
//! 形状事实:conv1d 等 3D 权重按扁平字节直读(row-major 连续,
//! [6144,1,4] ≡ [6144,4]),shape 元数据由层侧 Want 声明,此处不搬。

use crate::contract::ModelError;
use crate::formats::mmap::{open_raw_index, Mmap};
use crate::module::WeightSource;
use half::slice::{HalfBitsSliceExt, HalfFloatSliceExt};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

// 转换热路径(独立微 crate;debug 档 profile 覆盖 opt-level=3,
// 见 crate 根注释与 workspace Cargo.toml)
use crate::f16c::{bf16_bytes_to_f16_bytes, f32_slice_to_f16_bytes};

/// 流式权重源:mmap + 条目索引(堆上 KB 级),数据按需转换、取走即弃。
pub struct SafeTensorsSource {
    maps: Vec<Mmap>,
    /// 待取条目(take 即移除;锁只护索引,转换在锁外读映射区)
    index: Mutex<HashMap<String, Entry>>,
}

/// 张量条目(映射区内字节区间 + dtype)
#[derive(Clone, Copy)]
struct Entry {
    map_ix: usize,
    start: usize,
    nbytes: usize,
    dtype: safetensors::Dtype,
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
    /// 刀2 虚拟合并键面:in_proj_qkvz.weight = row-stack(in_proj_qkv.weight,
    /// in_proj_z.weight)。返回 (子键1, 元素数1, 子键2, 元素数2);非合并键 None。
    fn merged_parts(&self, key: &str) -> Option<(String, usize, String, usize)> {
        let base = key.strip_suffix(".weight")?;
        let (b1, b2) = crate::formats::split_qkvz(base)?;
        let idx = self.index.lock().unwrap();
        let e1 = idx.get(&format!("{b1}.weight"))?;
        let e2 = idx.get(&format!("{b2}.weight"))?;
        Some((
            format!("{b1}.weight"),
            e1.nbytes / e1.esz(),
            format!("{b2}.weight"),
            e2.nbytes / e2.esz(),
        ))
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
                Entry { map_ix: e.map_ix, start: e.start, nbytes: e.nbytes, dtype: e.dtype },
            );
        }
        Ok(Self { maps, index: Mutex::new(index) })
    }

    pub fn len(&self) -> usize {
        self.index.lock().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
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
    pub fn has_key(&self, key: &str) -> bool {
        self.index.lock().unwrap().contains_key(key)
    }
}

impl WeightSource for SafeTensorsSource {
    fn elem_len(&self, key: &str) -> Option<usize> {
        if let Some((_, n1, _, n2)) = self.merged_parts(key) {
            return Some(n1 + n2);
        }
        let e = self.index.lock().unwrap().get(key).cloned()?;
        Some(e.nbytes / e.esz())
    }

    /// 整取(F32 直读 / BF16 升位),条目即从索引移除。
    /// 流式装载主路径走 `take_range`(分块,条目保留)。
    fn take(&self, key: &str) -> Option<Vec<f32>> {
        if let Some((k1, _, k2, _)) = self.merged_parts(key) {
            let mut v = self.take(&k1)?;
            v.extend(self.take(&k2)?);
            return Some(v);
        }
        let e = self.index.lock().unwrap().remove(key)?;
        // 元素数按条目真实位宽(F5 修:原 /4 硬编码对 bf16 条目
        // 返回一半元素 —— 直转路径首次踩中;分块路径按 len 显式未暴露)
        Some(self.convert_range(&e, 0, e.nbytes / e.esz()))
    }

    /// 区间转换(mmap 直读,**不移除条目** —— 分块上传同键多块重复取)
    fn take_range(&self, key: &str, offset_elems: usize, len: usize) -> Option<Vec<f32>> {
        if let Some((k1, n1, k2, _)) = self.merged_parts(key) {
            // 行堆叠拼接:子1 元素区 [0,n1),子2 [n1,n1+n2)
            if offset_elems + len <= n1 {
                return self.take_range(&k1, offset_elems, len);
            }
            if offset_elems >= n1 {
                return self.take_range(&k2, offset_elems - n1, len);
            }
            let head = self.take_range(&k1, offset_elems, n1 - offset_elems)?;
            let tail = self.take_range(&k2, 0, offset_elems + len - n1)?;
            let mut v = head;
            v.extend(tail);
            return Some(v);
        }
        let e = self.index.lock().unwrap().get(key).cloned()?;
        if (offset_elems + len) * e.esz() > e.nbytes {
            return None;
        }
        Some(self.convert_range(&e, offset_elems, len))
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
    ) -> Option<()> {
        // 刀2 合并键面:跨界块拆两子键分别直写(行堆叠拼接)
        if let Some((k1, n1, k2, _)) = self.merged_parts(key) {
            let out_esz = match dtype {
                crate::contract::Dtype::F32 => 4,
                crate::contract::Dtype::F16 => 2,
                _ => return None,
            };
            if offset_elems + len <= n1 {
                return self.convert_chunk_into_bytes(&k1, offset_elems, len, dst, dtype);
            }
            if offset_elems >= n1 {
                return self.convert_chunk_into_bytes(&k2, offset_elems - n1, len, dst, dtype);
            }
            let head_elems = n1 - offset_elems;
            let head_bytes = head_elems * out_esz;
            self.convert_chunk_into_bytes(&k1, offset_elems, head_elems, &mut dst[..head_bytes], dtype)?;
            return self.convert_chunk_into_bytes(&k2, 0, len - head_elems, &mut dst[head_bytes..], dtype);
        }
        let e = match self.index.lock().unwrap().get(key).cloned() {
            Some(e) => e,
            None => {
                eprintln!("[st][DIAG] 索引未命中: {key}");
                return None;
            }
        };
        let esz = e.esz();
        if (offset_elems + len) * esz > e.nbytes {
            eprintln!(
                "[st][DIAG] OOB: {key} off{offset_elems} len{len} esz{esz} nbytes{}",
                e.nbytes
            );
            return None;
        }
        let s = e.start + offset_elems * esz;
        let bytes = &self.maps[e.map_ix][s..s + len * esz];
        match (e.dtype, dtype) {
            // F32 源 → F32 目标：LE 直拷（零转换）
            (safetensors::Dtype::F32, crate::contract::Dtype::F32) => {
                if dst.len() < len * 4 {
                    return None;
                }
                dst[..len * 4].copy_from_slice(bytes);
            }
            // F32 源 → f16 目标：F16C 单 pass 直写（舍入 RNE 与 from_f32 一致）
            (safetensors::Dtype::F32, crate::contract::Dtype::F16) => {
                if dst.len() < len * 2 {
                    return None;
                }
                f32_slice_to_f16_bytes(
                    // SAFETY:[u8] → [f32] 视图：条目字节序 LE 与主机一致，
                    // 长度 4 对齐由 chunks_exact 保证；只读视图
                    unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, len) },
                    &mut dst[..len * 2],
                );
            }
            // BF16 源 → f16 目标：F16C 单 pass 直写（bf16→f32 精确展宽 +
            // RNE 舍入，与 checksum 锚逐位一致）
            (safetensors::Dtype::BF16, crate::contract::Dtype::F16) => {
                if dst.len() < len * 2 {
                    return None;
                }
                bf16_bytes_to_f16_bytes(bytes, &mut dst[..len * 2]);
            }
            // BF16 源 → BF16 目标：直拷零转换(E5-DF3 同日十四;DFlash2 草稿
            // BF16 全程 —— passthrough 权重原生直载,sglang 同位)
            (safetensors::Dtype::BF16, crate::contract::Dtype::BF16) => {
                if dst.len() < len * 2 {
                    return None;
                }
                dst[..len * 2].copy_from_slice(bytes);
            }
            // F16 源 → BF16 目标:值转换(E5-DF3 同日十四安全网;主路径 =
            // BF16 原生源直拷)
            (safetensors::Dtype::F16, crate::contract::Dtype::BF16) => {
                if dst.len() < len * 2 {
                    return None;
                }
                crate::f16c::f16_bytes_to_bf16_bytes(bytes, &mut dst[..len * 2]);
            }
            // BF16 源 → f32 目标（f32 链回退路径）：批量展开
            (safetensors::Dtype::BF16, crate::contract::Dtype::F32) => {
                if dst.len() < len * 4 {
                    return None;
                }
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
            _ => return None,
        }
        self.maps[e.map_ix].dontneed(s, len * esz);
        Some(())
    }
}

//! AWQ g32-asym 装载源(compressed-tensors pack-quantized,gs=32,对称=false,
//! zp_dtype=int8;2026-10-01 cyankiwi/Qwen3.8-27B-AWQ-INT4 装载线)。
//!
//! 与 [`crate::formats::w4a16::W4A16Source`] 同源架构(mmap 懒物化 +
//! per-key 字节缓存 + DONTNEED 还页),差异只在量化语义与派生键集:
//!
//! - **检查点事实**(2026-10-01 对 cyankiwi 实测 + compressed_tensors
//!   0.19.0 `unpack_from_int32` 官方解码逐位对证):
//!   `weight_packed` I32 [out, in/8](LSB-first 沿 in,q ∈ [0,15] 无符号);
//!   `weight_scale` BF16 [out, in/32];`weight_zero_point` I32 [out/8, in/32]
//!   (LSB-first 沿 **out** —— word c 的 nibble t = 行 8c+t;zp ∈ [0,15]);
//!   反量化 **W = (q − zp_group) × scale**(双无符号域,截断差与有符号域
//!   代数等价);
//! - **eligible**(与 g128 同谓词,`marlin_eligible`:n%256==0 且 in%128==0
//!   —— pack_marlin_b 硬约束 k%128):→ qweight U32(B 打包与 U4B8 同套,
//!   q 原值无 −8;dequant 在内核内做 q − zp)/ scales F16(pack_marlin_s,
//!   组数 = in/32)/ **zeros U32**(`unpack_zp_ct` → `pack_marlin_z`,布局
//!   (k/g, n/8) i32,scale_perm + n-interleave 已烘焙)/ marlin_ws/marlin_ctmp
//!   零填充直写;
//! - **内核臂** = marlin kU4(has_zp;`GEMM_W4A16_AWQ` foreign 通道,
//!   `marlin_gemm_v2_awq_ffi`;group_blocks = 32/16 = 2 幂次合法);
//! - 非门控 → rayon 反量化 F16(W = (q − zp) × scale);passthrough →
//!   mmap 视图 F16C 直写(与 w4a16 同款)。
//!
//! 层侧对应:`Linear::new(.., QuantPlan::W4A16Awq)` 四件套(qweight/scales/
//! zeros/ws)+ ctmp;Qwen35Convention 认 `.zeros` 后缀。

use crate::contract::{Dtype, ModelError};
use crate::formats::mmap::{copy_par, open_raw_index, Mmap, RawEntry};
use crate::formats::split_qkvz;
use crate::formats::w4a16::{bf16_par, i32s_le_bytes, i32s_par, u16s_le_bytes};
use crate::module::WeightSource;
use owl_kernels::family::marlin::repack::{
    marlin_fused_indices, marlin_scales_index, pack_marlin_b_fused, pack_marlin_b_gather_into,
    pack_marlin_s_idx,
    pack_marlin_z, unpack_zp_ct,
};
use owl_kernels::family::marlin::v2_workspace_len;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// 懒物化缓存容量上限(27B 专用档):同 base 三键(qweight/scales/zeros)
/// 被 LPT 分桶拉到不同装载桶的不同位置,存活窗口 = 整个装载期 —— cap 必须
/// 容下全模型物化态(~14GB;RAM 108GB),否则 clear 全清把先建键清掉,
/// 后到键 3.3× 重复构建(2026-10-01 实测:512MB cap → 1183 次构建,
/// 16GB → 355 次)。clear 计数入 dbg.cache_clear 盯防。
const CACHE_CAP_BYTES: usize = 16 << 30;

/// 量化线性源形态(见 [`AwqSource::linear_form`])
enum QuantForm<'a> {
    Ct {
        pe: &'a RawEntry,
        se: &'a RawEntry,
        ze: &'a RawEntry,
    },
    F16 { we: &'a RawEntry },
}

/// 索引表种类(形状索引缓存键的第三轴;原 `bool` 只容 gather/fused 两态)
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum IdxKind {
    /// `marlin_gather_indices`(F16 兜底臂的 nibble gather)
    Gather,
    /// `marlin_fused_indices`(Ct 臂的 packed_word<<3|nib 融合表)
    Fused,
    /// `marlin_scales_index`(scales 转置+perm 的源下标表)
    Scales,
}

/// 键字节解析结果(懒物化后的一致字节 + 登记 dtype)
enum Resolved {
    /// 零填充直写(ws/ctmp;不占存储)
    Zeroed,
    /// mmap 视图直转/直拷(passthrough;f16 基线同款)
    View(RawEntry),
    /// **多段逻辑拼接视图**(合并虚拟键:in_proj_qkvz.packed_raw = 两子基
    /// `weight_packed` 按序拼接)。零中间缓冲:消费侧按 dst 偏移**直写**
    /// (2026-10-16 装载提速定谳:原实现每键新建 41.9MB Vec + 二次拷贝,
    /// anon 首触 + 拷贝共 ≈75ms/键 × 48 键 = 3.9s,纯税)
    Merged(Vec<RawEntry>),
    /// 已物化字节(qweight U32 / scales F16 / zeros U32 / 反量化 weight F16)
    Bytes(Arc<[u8]>, Dtype),
}

pub struct AwqSource {
    maps: Vec<Mmap>,
    index: HashMap<String, RawEntry>,
    /// 懒物化字节缓存(键 → 字节;容量清空驱逐)
    #[allow(clippy::type_complexity)] // 行块缓存(字节镜像 + 计数)
    cache: Mutex<(HashMap<String, Arc<[u8]>>, usize)>,
    /// gather/fused 索引表(按 (k, out, fused?) 形状复用;除法分解只算一次)
    #[allow(clippy::type_complexity)] // 索引缓存(键 → 列序)
    idx_cache: Mutex<HashMap<(usize, usize, IdxKind), Arc<Vec<u32>>>>,
    /// 构建分片锁(16 片按 base hash;同 base 互斥、异 base 并行 ——
    /// 全局单锁时 build 串行链 ≈ 装载墙钟主导,分片后 4 桶并发 build
    /// 与 DMA 流水重叠)
    build_locks: Vec<Mutex<()>>,
    /// 设备重排模式(装载域 set;true = CPU 构建走轻路径:量化大键的
    /// marlin B 由 GPU repack 产出,CPU 仅产 scales/zeros/packed_raw_ct)
    device_repack: std::cell::Cell<bool>,
}

impl AwqSource {
    /// 装载入口:mmap + 索引(毫秒级,零数据拷贝);重排/反量化全部
    /// 懒到键首触。无缓存文件(与 w4a16 v2 同律)。
    pub fn open_dir(dir: &Path) -> Result<Self, ModelError> {
        let t0 = std::time::Instant::now();
        let (maps, index) = open_raw_index(dir)?;
        owl_shared::log_info!("awq", "mmap 索引 {} 项 @ {:?}(懒物化,无缓存)",
            index.len(),
            t0.elapsed()
        );
        Ok(Self {
            maps,
            index,
            cache: Mutex::new((HashMap::new(), 0)),
            idx_cache: Mutex::new(HashMap::new()),
            build_locks: (0..16).map(|_| Mutex::new(())).collect(),
            device_repack: std::cell::Cell::new(false),
        })
    }

    /// 设备重排模式(装载域注入;必须在首键物化前设置):
    /// true = CPU 构建轻路径(量化大键的 marlin B 由 GPU repack 产出)
    pub fn set_device_repack(&self, on: bool) {
        self.device_repack.set(on);
    }

    /// 诊断口:索引键全集(装载面排查/基准用)
    pub fn keys(&self) -> Vec<String> {
        self.index.keys().cloned().collect()
    }

    /// 量化线性形态定位:(out, k) + 源条目。两形态:
    /// - `Ct`:weight_packed + weight_scale + weight_zero_point(量化层);
    /// - `F16`:裸 weight(checkpoint ignore 层,如 layer0 out_proj ——
    ///   尺寸 eligible 但上游未量化)→ 现场量化兜底(见 `build_linear`)。
    fn linear_form(
        &self,
        base: &str,
    ) -> Option<(usize, usize, QuantForm<'_>)> {
        if let Some(pe) = self.index.get(&format!("{base}.weight_packed")) {
            let se = self.index.get(&format!("{base}.weight_scale"))?;
            let ze = self.index.get(&format!("{base}.weight_zero_point"))?;
            let out = *pe.shape.first()?;
            let k = pe.shape.get(1)? * 8;
            Some((out, k, QuantForm::Ct { pe, se, ze }))
        } else {
            let we = self.index.get(&format!("{base}.weight"))?;
            let out = *we.shape.first()?;
            let k = *we.shape.get(1)?;
            Some((out, k, QuantForm::F16 { we }))
        }
    }

    /// fc 列半拆物化(E5-M2b;键首触才建,双检缓存 + build_lock 同款):
    /// 读物理键 mtp.fc.weight(bf16 [out, 2h])→ 逐行 bf16→f16 + 取前/后
    /// 半 → [out, h] f16 连续。rayon 行并行(行长 2 字节对齐,安全)。
    fn fc_half_bytes(&self, high: bool) -> Option<Arc<[u8]>> {
        let vkey = if high { "mtp.fc_h.weight" } else { "mtp.fc_e.weight" };
        {
            let cache = self.cache.lock().unwrap();
            if let Some(b) = cache.0.get(vkey) {
                return Some(b.clone());
            }
        }
        let we = self.index.get("mtp.fc.weight")?;
        let out = *we.shape.first()?;
        let din = *we.shape.get(1)?;
        let half = din / 2;
        let src = &self.maps[we.map_ix][we.start..we.start + we.nbytes];
        let mut dst = vec![0u8; out * half * 2];
        {
            use rayon::prelude::*;
            let skip = if high { half * 2 } else { 0 };
            dst.par_chunks_mut(half * 2).enumerate().for_each(|(r, row)| {
                let base = r * din * 2 + skip;
                crate::f16c::bf16_bytes_to_f16_bytes(
                    &src[base..base + half * 2],
                    &mut row[..half * 2],
                );
            });
        }
        let b: Arc<[u8]> = dst.into();
        {
            let mut cache = self.cache.lock().unwrap();
            cache.0.insert(vkey.to_string(), b.clone());
        }
        Some(b)
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
        if let Some(base) = key.strip_suffix(".zeros") {
            return Some(Resolved::Bytes(self.linear_bytes_for(base, key)?, Dtype::U32));
        }
        // ---- W4A8 键族(S2;a8 布局三件套 + 全局因子)----
        // `scales_a8` 线 dtype = **F16**:int16 归一位型塞在 2 字节槽里,
        // 与 a8 GEMM 的 `scales` 槽(声明 F16)同宽同形
        if is_a8_key(key) {
            let dt = if key.ends_with(".a8_factor") {
                Dtype::F32
            } else if key.ends_with(".scales_a8") {
                Dtype::F16
            } else {
                Dtype::U32
            };
            let base = key.rsplit_once('.').map(|(b, _)| b)?;
            return Some(Resolved::Bytes(self.linear_bytes_for(base, key)?, dt));
        }
        // E5-M2b fc 列半拆:mtp.fc_{e,h}.weight = mtp.fc.weight 的 in 维
        // 前/后半(W [out, 2h] 行主序 → 行内去交织,装载一次性 rayon;
        // bf16→f16 归一)。MTP 头双子 Linear 消费,消每轮 narrow 物化。
        if key == "mtp.fc_e.weight" || key == "mtp.fc_h.weight" {
            return Some(Resolved::Bytes(
                self.fc_half_bytes(key.ends_with("_h.weight"))?,
                Dtype::F16,
            ));
        }
        // passthrough 优先(norm/conv/embed 等原生存量键;含非量化线性
        // 的裸 .weight —— cyankiwi 的 ignore 逐层不同,同键可能两态)
        if let Some(e) = self.index.get(key) {
            return Some(Resolved::View(e.clone()));
        }
        // 设备重排原始键:{base}.packed_raw(CPU 零重排,重排在 GPU)
        // - Ct 形态:weight_packed 原样(View 直拷 I32→U32)
        // - F16 兜底形态:RTN nibbles 打包回 ct 布局(build 期一并产出;
        //   使 ignore 层与量化层走同一 GPU 通路,免特判)
        if let Some(base) = key.strip_suffix(".packed_raw") {
            // 刀2 虚拟合并:in_proj_qkvz.packed_raw = 两子基 weight_packed
            // 行堆叠(同 k,字节拼接;要求两子均 Ct 形态 —— cyankiwi 满足)
            if let Some((b1, b2)) = split_qkvz(base) {
                if let (Some(e1), Some(e2)) = (
                    self.index.get(&format!("{b1}.weight_packed")),
                    self.index.get(&format!("{b2}.weight_packed")),
                ) {
                    // 零中间缓冲:两子基视图按序交给消费侧直写
                    return Some(Resolved::Merged(vec![e1.clone(), e2.clone()]));
                }
            }
            if let Some(pe) = self.index.get(&format!("{base}.weight_packed")) {
                return Some(Resolved::View(pe.clone()));
            }
            if self.index.contains_key(&format!("{base}.weight")) {
                return Some(Resolved::Bytes(
                    self.linear_bytes_for(base, &format!("{base}.packed_raw_ct"))?,
                    Dtype::U32,
                ));
            }
        }
        // 非门控线性反量化臂(Linear 与本源同谓词,该键只对 non-eligible 出现)
        if let Some(base) = key.strip_suffix(".weight") {
            if self.index.contains_key(&format!("{base}.weight_packed")) {
                return Some(Resolved::Bytes(self.linear_bytes_for(base, key)?, Dtype::F16));
            }
        }
        None
    }

    /// scales 源索引表(按 (k, out) 形状缓存;见 [`Self::pack_scales`])
    fn scales_index(&self, out: usize, k: usize) -> Arc<Vec<u32>> {
        self.idx_cache
            .lock()
            .unwrap()
            .entry((k, out, IdxKind::Scales))
            .or_insert_with(|| Arc::new(marlin_scales_index(out, k / 32)))
            .clone()
    }

    /// marlin scales 打包(装载热路径):源索引表按 (k, out) 形状缓存 ⇒ 每元素
    /// 零整数除法。
    ///
    /// 2026-10-16 提速定谳:原 `pack_marlin_s` 每元素现算两次运行时除法
    /// (`/ out` `/ % out`,out 非二次幂)⇒ 0.74GB/s(2.78M 元素 7.5ms/键,
    /// 整载 352 键 ≈2.6s)。整载共 6 种形状,表一次、热路径纯 gather。
    fn pack_scales(&self, scales_f32: &[f32], out: usize, k: usize) -> Vec<u16> {
        let groups = k / 32;
        pack_marlin_s_idx(scales_f32, out, groups, &self.scales_index(out, k))
    }

    /// **bf16 源 → marlin scales 字节**(单遍融合;装载热路径)。
    ///
    /// 原链 = `bf16_par`(源→f32,11MB)+ `pack_marlin_s`(f32→u16,5.6MB)
    /// + `u16s_le_bytes`(u16→字节,5.6MB):三分配两拷贝,每键 ≈22MB
    /// 新堆页。2026-10-16 release 定谳:`.scales` 352 键 **3.42s** 是全载
    /// 最大单项,而三件纯算力只占 ≈3ms/键 —— 余量全在分配/缺页。
    /// 本函数 = 索引表 gather + bf16→f16 + LE 直写 dst,**一分配一路过**,
    /// 且逐位与原链等价(bf16→f32 为零舍入展宽,余下同一 `f16::from_f32`)。
    ///
    /// `parts` = 源 bf16 分段:1 段 = 常规 Ct 键;2 段 = 合并虚拟键左右半
    /// (拼接语义等同于原 `scales_f32` 的 concat)。
    fn pack_scales_bytes(&self, parts: &[&[u8]], out: usize, k: usize) -> Vec<u8> {
        let groups = k / 32;
        let mut dst = vec![0u8; out * groups * 2];
        pack_scales_bf16_into(parts, out, groups, &self.scales_index(out, k), &mut dst);
        dst
    }

    /// 派生/虚拟键(不指向单一 mmap 条目)名判定 —— [`Self::resolve`]
    /// 专项分支表的**谓词镜像**(单一事实源:`resolve` 里的分支与本节
    /// 同名列表;两处同改)。
    ///
    /// 用途:[`WeightSource::native_dtype`] 的廉价筛查(声扫热路,禁物化)。
    /// 反面(不是派生键)也可能解析成 `Bytes`(如 `x.weight` 只有
    /// `weight_packed` 的 F16 兜底层)—— 那由 `index.get(key)` 缺失逼住。
    fn is_derived_key(&self, key: &str) -> bool {
        key.ends_with(".marlin_ws")
            || key.ends_with(".marlin_ctmp")
            || key.ends_with(".qweight")
            || key.ends_with(".scales")
            || key.ends_with(".zeros")
            // W4A8 键族(同一份源数据的 a8 布局产物;S2)
            || key.ends_with(".qweight_a8")
            || key.ends_with(".scales_a8")
            || key.ends_with(".zeros_a8")
            || key.ends_with(".a8_factor")
            || key.ends_with(".packed_raw")
            || key == "mtp.fc_e.weight"
            || key == "mtp.fc_h.weight"
    }


/// 量化线性物化(键首触才建;qweight/scales/zeros/weight 四键共享
    /// 一次构建)。双检缓存 + build_lock 串行(w4a16 同款)。
    fn linear_bytes_for(&self, base: &str, key: &str) -> Option<Arc<[u8]>> {
        // **a8 键族走独立构建**(S2):与 a16 同源不同布局,且**互不物化**
        // —— a8 路只产 a8 四件,不陪跑 a16 的重排(反之亦然)
        if is_a8_key(key) {
            // 设备重排模式下 B 由 GPU 核产出 ⇒ 这里**不做 CPU B 打包**
            // (先前无条件烘 B 会让设备档白跑一趟全模 CPU 重排)
            let want_b = key.ends_with(".qweight_a8");
            if let Err(e) = self.build_linear_a8(base, want_b) {
                owl_shared::log_info!("awq", "a8 物化失败 {base}: {e}");
                return None;
            }
            let cache = self.cache.lock().unwrap();
            return cache.0.get(key).cloned();
        }
        // 刀2 虚拟合并:in_proj_qkvz = row-stack(in_proj_qkv, in_proj_z)。
        // **源域合并后一次构建**(marlin s/z 是全 flat 64-chunk 置换,
        // 构建产物域内拼接错序;源域 packed/scales/zp 均行主序,concat
        // 合法)。要求两子均 Ct 形态(cyankiwi 满足;F16 兜底混态不支持)。
        if let Some((b1, b2)) = split_qkvz(base) {
            let suffix = &key[base.len()..];
            let (n1, k1, f1) = self.linear_form(&b1)?;
            let (n2, k2, f2) = self.linear_form(&b2)?;
            let (QuantForm::Ct { se: se1, ze: ze1, .. }, QuantForm::Ct { se: se2, ze: ze2, .. }) = (&f1, &f2) else {
                return None;
            };
            if k1 != k2 {
                return None;
            }
            {
                let cache = self.cache.lock().unwrap();
                if let Some(b) = cache.0.get(key) {
                    return Some(b.clone());
                }
            }
            let shard = base.bytes().map(|b| b as usize).sum::<usize>() % self.build_locks.len();
            let _g = self.build_locks[shard].lock().unwrap();
            // 双检(锁内)
            {
                let cache = self.cache.lock().unwrap();
                if let Some(b) = cache.0.get(key) {
                    return Some(b.clone());
                }
            }
            let out = n1 + n2;
            let k = k1;
            let groups = k / 32;
            // 源域行堆叠:scales/zp 展开 concat([out, groups] 行主序)
            // **scales 不拼中转**:两段 bf16 视图 + 形状索引表单遍融合直出字节
            let sb1 = &self.maps[se1.map_ix][se1.start..se1.start + se1.nbytes];
            let sb2 = &self.maps[se2.map_ix][se2.start..se2.start + se2.nbytes];
            let zb1 = &self.maps[ze1.map_ix][ze1.start..ze1.start + ze1.nbytes];
            let zb2 = &self.maps[ze2.map_ix][ze2.start..ze2.start + ze2.nbytes];
            let mut zp_u8 = unpack_zp_ct(&i32s_par(zb1), n1, groups);
            zp_u8.extend(unpack_zp_ct(&i32s_par(zb2), n2, groups));
            self.maps[ze1.map_ix].dontneed(ze1.start, ze1.nbytes);
            self.maps[ze2.map_ix].dontneed(ze2.start, ze2.nbytes);
            let bytes: Vec<u8> = match suffix {
                ".scales" => {
                    let b = self.pack_scales_bytes(&[sb1, sb2], out, k);
                    self.maps[se1.map_ix].dontneed(se1.start, se1.nbytes);
                    self.maps[se2.map_ix].dontneed(se2.start, se2.nbytes);
                    b
                }
                ".zeros" => i32s_le_bytes(&pack_marlin_z(&zp_u8, out, groups)),
                ".qweight" | ".packed_raw" => {
                    // packed 源域 [n, k/8] i32 行主序,flat concat 合法
                    let (pe1, pe2) = match (&f1, &f2) {
                        (QuantForm::Ct { pe: p1, .. }, QuantForm::Ct { pe: p2, .. }) => (p1, p2),
                        _ => return None,
                    };
                    let pb1 = &self.maps[pe1.map_ix][pe1.start..pe1.start + pe1.nbytes];
                    let pb2 = &self.maps[pe2.map_ix][pe2.start..pe2.start + pe2.nbytes];
                    let mut packed = i32s_par(pb1);
                    packed.extend(i32s_par(pb2));
                    self.maps[pe1.map_ix].dontneed(pe1.start, pe1.nbytes);
                    self.maps[pe2.map_ix].dontneed(pe2.start, pe2.nbytes);
                    if suffix == ".packed_raw" {
                        i32s_le_bytes(&packed)
                    } else {
                        // host 路径:融合索引 gather(与 build_linear Ct 臂同管)
                        let words = out * k / 8;
                        let idx = self
                            .idx_cache
                            .lock()
                            .unwrap()
                            .entry((k, out, IdxKind::Fused))
                            .or_insert_with(|| Arc::new(marlin_fused_indices(k, out)))
                            .clone();
                        let mut b_buf: Vec<i32> = Vec::new();
                        pack_marlin_b_fused(&packed, &idx, words, &mut b_buf);
                        i32s_le_bytes(&b_buf)
                    }
                }
                _ => return None,
            };
            let mut cache = self.cache.lock().unwrap();
            cache.0.insert(key.to_string(), Arc::from(bytes.into_boxed_slice()));
            return cache.0.get(key).cloned();
        }
        let (out, k, form) = self.linear_form(base)?;
        {
            let cache = self.cache.lock().unwrap();
            if let Some(b) = cache.0.get(key) {
                return Some(b.clone());
            }
        }
        self.build_linear(base, out, k, form).ok()?;
        self.cache.lock().unwrap().0.get(key).cloned()
    }

    /// 真构建(串行;内部 rayon 饱和)。F16 形态 = checkpoint ignore 层
    /// 兜底:裸权重现场 RTN g32-**sym** 量化(zp ≡ 8)为 awq 四件套 ——
    /// kU4 内核(q − 8)×s 语义与对称量化等价;上游 ignore 的动机
    /// (质量保守)让位于统一 kU4 通路,精度损失限于 RTN(无 observer)。
    fn build_linear(
        &self,
        base: &str,
        out: usize,
        k: usize,
        form: QuantForm<'_>,
    ) -> Result<(), ModelError> {
        // 分片锁:同 base 互斥(双检防重复构建),异 base 并行
        let shard = base.bytes().map(|b| b as usize).sum::<usize>() % self.build_locks.len();
        let _g = self.build_locks[shard].lock().unwrap();
        // 双检(锁内):并发首触(同层 qweight/scales/zeros 落不同装载桶)
        // 时先到者已填缓存 —— 重复构建直接短路(cap 16GB 后 clear=0,本路径
        // 仅护并发窗口)
        {
            let cache = self.cache.lock().unwrap();
            let built = cache.0.contains_key(&format!("{base}.qweight"))
                || cache.0.contains_key(&format!("{base}.weight"));
            if built {
                return Ok(());
            }
        }
        let t0 = std::time::Instant::now();
        let groups = k / 32;
        let mut inserts: Vec<(String, Arc<[u8]>, usize)> = Vec::new();
        let mut form_ct = false;
        let device = self.device_repack.get();
        if crate::formats::w4a16::marlin_eligible(out, k) && device {
            // ---- 轻路径(device):CPU 仅产 scales/zeros;qweight 的
            //      marlin B 由 GPU repack 产出(load.rs 设备重排臂),
            //      packed 原样 DMA —— CPU 44MB×3 的重排/拷贝全消 ----
            let s_bytes: Vec<u8>;
            let z_buf: Vec<i32>;
            match form {
                QuantForm::Ct { se, ze, .. } => {
                    form_ct = true;
                    // 设备臂分相(2026-10-12 装载提速):原只有 load.mat.ct
                    // 总账(198ms/层)—— 拆到五段,定谳真凶住哪段。
                    let zp_i32 = owl_shared::timer_scope!("load.zp.zread", {
                        let b = &self.maps[ze.map_ix][ze.start..ze.start + ze.nbytes];
                        i32s_par(b)
                    });
                    s_bytes = owl_shared::timer_scope!("load.zp.spack", {
                        let se_v = &self.maps[se.map_ix][se.start..se.start + se.nbytes];
                        self.pack_scales_bytes(&[se_v], out, k)
                    });
                    self.maps[se.map_ix].dontneed(se.start, se.nbytes);
                    self.maps[ze.map_ix].dontneed(ze.start, ze.nbytes);
                    let zp_u8 = owl_shared::timer_scope!(
                        "load.zp.unpack",
                        unpack_zp_ct(&zp_i32, out, groups)
                    );
                    z_buf = owl_shared::timer_scope!(
                        "load.zp.pack",
                        pack_marlin_z(&zp_u8, out, groups)
                    );
                }
                QuantForm::F16 { we } => {
                    let wb = &self.maps[we.map_ix][we.start..we.start + we.nbytes];
                    let w_f32: Vec<f32> = match we.dtype {
                        safetensors::Dtype::BF16 => bf16_par(wb),
                        safetensors::Dtype::F16 => wb
                            .chunks_exact(2)
                            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                            .collect(),
                        safetensors::Dtype::F32 => wb
                            .chunks_exact(4)
                            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect(),
                        _ => return Err(ModelError::Msg(format!("awq: {base} 裸权重 dtype 不支持"))),
                    };
                    self.maps[we.map_ix].dontneed(we.start, we.nbytes);
                    let scales_f32 = rtn_g32_sym_scales(&w_f32, out, k);
                    s_bytes = u16s_le_bytes(&self.pack_scales(&scales_f32, out, k));
                    z_buf = pack_marlin_z(&vec![8u8; out * groups], out, groups);
                    // packed_raw_ct(设备重排核的输入):RTN nibbles → ct i32
                    let qb = rtn_g32_sym_packed(&w_f32, out, k);
                    let mut packed_ct = vec![0i32; out * (k / 8)];
                    for (idx, &nib) in qb.iter().enumerate() {
                        packed_ct[idx / 8] |= (nib as i32) << (4 * (idx % 8));
                    }
                    inserts.push((
                        format!("{base}.packed_raw_ct"),
                        Arc::from(i32s_le_bytes(&packed_ct).into_boxed_slice()),
                        packed_ct.len() * 4,
                    ));
                }
            }
            let s_len = s_bytes.len();
            inserts.push((format!("{base}.scales"), Arc::from(s_bytes.into_boxed_slice()), s_len));
            inserts.push((
                format!("{base}.zeros"),
                Arc::from(i32s_le_bytes(&z_buf).into_boxed_slice()),
                z_buf.len() * 4,
            ));
        } else if crate::formats::w4a16::marlin_eligible(out, k) {
            // ---- 全路径(Host 臂;裁决唯一产地 = module::RepackPath::resolve)----
            // B 打包与 U4B8 同套(q 原值,无 −8;dequant 在 kU4 内核内做)
            let s_bytes: Vec<u8>;
            let z_buf: Vec<i32>;
            // B 打包双路径:Ct = fused(packed 直提 nibble,消 89MB 中转);
            // F16 兜底 = unpacked u8 → 旧 gather(元素索引域,与 fused 分缓存)
            let (packed, q_buf): (Vec<i32>, Option<Vec<u8>>);
            match form {
                QuantForm::Ct { pe, se, ze } => {
                    form_ct = true;
                    // 源视图:i32 packed / bf16 scales / i32 zp(rayon,对齐安全)
                    let packed_bytes = &self.maps[pe.map_ix][pe.start..pe.start + pe.nbytes];
                    packed = i32s_par(packed_bytes);
                    let scale_bytes = &self.maps[se.map_ix][se.start..se.start + se.nbytes];
                    let zp_bytes = &self.maps[ze.map_ix][ze.start..ze.start + ze.nbytes];
                    let zp_i32 = i32s_par(zp_bytes);
                    // 源页消费完毕即还(DONTNEED)
                    self.maps[pe.map_ix].dontneed(pe.start, pe.nbytes);
                    self.maps[se.map_ix].dontneed(se.start, se.nbytes);
                    self.maps[ze.map_ix].dontneed(ze.start, ze.nbytes);
                    let t_step = std::time::Instant::now();
                    s_bytes = self.pack_scales_bytes(&[scale_bytes], out, k);
                    let zp_u8 = unpack_zp_ct(&zp_i32, out, groups);
                    z_buf = pack_marlin_z(&zp_u8, out, groups);
                    owl_shared::metrics::with_metrics_store(|s| {
                        s.timer_record_tag("load.mat.sz", t_step.elapsed(), file!(), line!())
                    });
                    q_buf = None;
                }
                QuantForm::F16 { we } => {
                    // 裸权重(bf16/f16)→ f32 → RTN g32-sym(zp ≡ 8)
                    let wb = &self.maps[we.map_ix][we.start..we.start + we.nbytes];
                    let w_f32: Vec<f32> = match we.dtype {
                        safetensors::Dtype::BF16 => bf16_par(wb),
                        safetensors::Dtype::F16 => wb
                            .chunks_exact(2)
                            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                            .collect(),
                        safetensors::Dtype::F32 => wb
                            .chunks_exact(4)
                            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect(),
                        _ => return Err(ModelError::Msg(format!("awq: {base} 裸权重 dtype 不支持"))),
                    };
                    self.maps[we.map_ix].dontneed(we.start, we.nbytes);
                    let scales_f32 = rtn_g32_sym_scales(&w_f32, out, k);
                    s_bytes = u16s_le_bytes(&self.pack_scales(&scales_f32, out, k));
                    z_buf = pack_marlin_z(&vec![8u8; out * groups], out, groups);
                    packed = Vec::new();
                    let qb = rtn_g32_sym_packed(&w_f32, out, k);
                    // 设备重排原始键供给:RTN nibbles → ct i32 布局
                    // ([out, k/8],LSB-first 沿 k —— 与 weight_packed 同构)
                    let mut packed_ct = vec![0i32; out * (k / 8)];
                    for (idx, &nib) in qb.iter().enumerate() {
                        packed_ct[idx / 8] |= (nib as i32) << (4 * (idx % 8));
                    }
                    inserts.push((
                        format!("{base}.packed_raw_ct"),
                        Arc::from(i32s_le_bytes(&packed_ct).into_boxed_slice()),
                        packed_ct.len() * 4,
                    ));
                    q_buf = Some(qb);
                }
            }
            let words = out * k / 8;
            let mut b_buf: Vec<i32> = Vec::new();
            let t_gather = std::time::Instant::now();
            if form_ct {
                // Ct:融合索引(packed_word<<3|nib)直提 —— unpack+gather 两步合一
                let idx = self
                    .idx_cache
                    .lock()
                    .unwrap()
                    .entry((k, out, IdxKind::Fused))
                    .or_insert_with(|| Arc::new(marlin_fused_indices(k, out)))
                    .clone();
                pack_marlin_b_fused(&packed, &idx, words, &mut b_buf);
            } else {
                let idx = self
                    .idx_cache
                    .lock()
                    .unwrap()
                    .entry((k, out, IdxKind::Gather))
                    .or_insert_with(|| {
                        Arc::new(owl_kernels::family::marlin::repack::marlin_gather_indices(k, out))
                    })
                    .clone();
                let q = q_buf.as_ref().expect("F16 臂 q_buf 必在");
                pack_marlin_b_gather_into(q, &idx, words, &mut b_buf);
            }
            owl_shared::metrics::with_metrics_store(|s| {
                s.timer_record_tag("load.mat.gather", t_gather.elapsed(), file!(), line!())
            });
            inserts.push((
                format!("{base}.qweight"),
                Arc::from(i32s_le_bytes(&b_buf).into_boxed_slice()),
                b_buf.len() * 4,
            ));
            let s_len = s_bytes.len();
            inserts.push((format!("{base}.scales"), Arc::from(s_bytes.into_boxed_slice()), s_len));
            inserts.push((
                format!("{base}.zeros"),
                Arc::from(i32s_le_bytes(&z_buf).into_boxed_slice()),
                z_buf.len() * 4,
            ));
        } else {
            // 反量化臂:仅 Ct 形态(F16 形态非 eligible 走 passthrough View)
            let QuantForm::Ct { pe, se, ze } = form else {
                return Err(ModelError::Msg(format!(
                    "awq: {base} 非 eligible 裸权重应走 passthrough,不可达"
                )));
            };
            let packed_bytes = &self.maps[pe.map_ix][pe.start..pe.start + pe.nbytes];
            let packed = i32s_par(packed_bytes);
            let scale_bytes = &self.maps[se.map_ix][se.start..se.start + se.nbytes];
            let scales_f32 = bf16_par(scale_bytes);
            let zp_bytes = &self.maps[ze.map_ix][ze.start..ze.start + ze.nbytes];
            let zp_i32 = i32s_par(zp_bytes);
            self.maps[pe.map_ix].dontneed(pe.start, pe.nbytes);
            self.maps[se.map_ix].dontneed(se.start, se.nbytes);
            self.maps[ze.map_ix].dontneed(ze.start, ze.nbytes);
            // W = (q − zp_group) × scale(host rayon;小线性专用)
            let mut dst = vec![0u8; out * k * 2];
            dequant_u4_awq_f16_bytes(&packed, &zp_i32, &scales_f32, out, k, &mut dst);
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
        // metrics:物化分相落账(Ct 重排 / F16 现场量化,直用 API 恒开)
        let tag = if form_ct { "load.mat.ct" } else { "load.mat.f16" };
        owl_shared::metrics::with_metrics_store(|s| {
            s.timer_record_tag(tag, t0.elapsed(), file!(), line!())
        });
        owl_shared::log_info!("awq", "物化 {base} (out={out}, k={k}) @ {:?}", t0.elapsed());
        Ok(())
    }

    /// **a8 四件物化**(W4A8 × AWQ;S2):`{base}.{qweight_a8, scales_a8, zeros_a8, a8_factor}`
    ///
    /// 与 a16 键族**同源不同布局**(同一份 ct packed/scales/zp):
    /// - B  → `pack_marlin_b_a8`(32×32 tile;与 a16 的 16×64+1024-PIPE 不同)
    /// - s  → `pack_marlin_s_a8`(perm_single + **int16 归一**;第二返回值 =
    ///        **全局因子** `max/4096` —— 它折进激活量化核的设备标量块,故单独
    ///        出一个 `{base}.a8_factor` f32[1] 键给层声明 want)
    /// - zp → `pack_marlin_z_a8`(**无 n-interleave**)
    ///
    /// ⚠️ a8 **只有 CPU 打包路**(核侧无 a8 repack GPU 核)⇒ 与 27B 生产默认的
    /// 设备重排不同路;装载耗时增量记 S2 台账(`load.mat.a8` 分相)。
    ///
    /// 双检缓存 + build_lock 串行(与 a16 臂同纪律;插入原子 —— 四件同生同灭)。
    /// `want_b` = 是否烘 CPU 版 B(`{base}.qweight_a8`):**设备重排档为 false**
    /// —— B 交给 `owl_ct_repack_a8_u32`(与 a16 生产档同一条快路)。scales/zp/
    /// 因子三件恒烘(数据小,且与 B 无关)。
    fn build_linear_a8(&self, base: &str, want_b: bool) -> Result<(), ModelError> {
        use owl_kernels::family::marlin::repack::{pack_marlin_b_a8, pack_marlin_s_a8, pack_marlin_z_a8, unpack_nibbles};
        // ---- 刀2 虚拟合并键(in_proj_qkvz = row-stack(qkv, z))的 a8 分支 ----
        // 与 a16 臂同律:**源域行拼接**(scales/zp/nibble 均行主序 ⇒ concat 合法),
        // 拼完再进 a8 三件套 —— 全局因子取**合并后**尺度的 max(一层一个 a_scales)
        if let Some((b1, b2)) = split_qkvz(base) {
            let Some((n1, k1, f1)) = self.linear_form(&b1) else {
                return Err(ModelError::Msg(format!("awq/a8: 子基 {b1} 形态不可解")));
            };
            let Some((n2, k2, f2)) = self.linear_form(&b2) else {
                return Err(ModelError::Msg(format!("awq/a8: 子基 {b2} 形态不可解")));
            };
            if k1 != k2 {
                return Err(ModelError::Msg(format!(
                    "awq/a8: 合并键 {base} 两子基 k 不等({k1} vs {k2})"
                )));
            }
            let (QuantForm::Ct { pe: pe1, se: se1, ze: ze1 }, QuantForm::Ct { pe: pe2, se: se2, ze: ze2 }) =
                (&f1, &f2)
            else {
                return Err(ModelError::Msg(format!(
                    "awq/a8: 合并键 {base} 要求两子均 Ct 形态(cyankiwi 满足)"
                )));
            };
            let out = n1 + n2;
            let k = k1;
            let groups = k / 32;
            let shard = base.bytes().map(|b| b as usize).sum::<usize>() % self.build_locks.len();
            let _g = self.build_locks[shard].lock().unwrap();
            let guard_key = if want_b { format!("{base}.qweight_a8") } else { format!("{base}.scales_a8") };
            {
                let cache = self.cache.lock().unwrap();
                if cache.0.contains_key(&guard_key) {
                    return Ok(());
                }
            }
            let t0 = std::time::Instant::now();
            // 源域拼接:scales(bf16→f32 两段)/ zp(u8 两段)/ nibble(仅 want_b)
            let mut s_f32 = bf16_par(&self.maps[se1.map_ix][se1.start..se1.start + se1.nbytes]);
            s_f32.extend(bf16_par(&self.maps[se2.map_ix][se2.start..se2.start + se2.nbytes]));
            let mut zp_u8 = unpack_zp_ct(
                &i32s_par(&self.maps[ze1.map_ix][ze1.start..ze1.start + ze1.nbytes]),
                n1,
                groups,
            );
            zp_u8.extend(unpack_zp_ct(
                &i32s_par(&self.maps[ze2.map_ix][ze2.start..ze2.start + ze2.nbytes]),
                n2,
                groups,
            ));
            for (m, e) in [(se1.map_ix, se1), (se2.map_ix, se2), (ze1.map_ix, ze1), (ze2.map_ix, ze2)] {
                self.maps[m].dontneed(e.start, e.nbytes);
            }
            let (s_bits, factor) = pack_marlin_s_a8(&s_f32, out, groups);
            let z_buf = pack_marlin_z_a8(&zp_u8, out, groups);
            let mut inserts: Vec<(String, Arc<[u8]>, usize)> = vec![
                (
                    format!("{base}.scales_a8"),
                    Arc::from(u16s_le_bytes(&s_bits).into_boxed_slice()),
                    s_bits.len() * 2,
                ),
                (
                    format!("{base}.zeros_a8"),
                    Arc::from(i32s_le_bytes(&z_buf).into_boxed_slice()),
                    z_buf.len() * 4,
                ),
                (
                    format!("{base}.a8_factor"),
                    Arc::from(factor.to_le_bytes().to_vec().into_boxed_slice()),
                    4,
                ),
            ];
            if want_b {
                let mut q = unpack_nibbles(
                    &i32s_par(&self.maps[pe1.map_ix][pe1.start..pe1.start + pe1.nbytes]),
                    n1,
                    k,
                );
                q.extend(unpack_nibbles(
                    &i32s_par(&self.maps[pe2.map_ix][pe2.start..pe2.start + pe2.nbytes]),
                    n2,
                    k,
                ));
                for (m, e) in [(pe1.map_ix, pe1), (pe2.map_ix, pe2)] {
                    self.maps[m].dontneed(e.start, e.nbytes);
                }
                let b_buf = pack_marlin_b_a8(&q, k, out);
                inserts.push((
                    format!("{base}.qweight_a8"),
                    Arc::from(i32s_le_bytes(&b_buf).into_boxed_slice()),
                    b_buf.len() * 4,
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
            owl_shared::metrics::with_metrics_store(|s| {
                s.timer_record_tag("load.mat.a8", t0.elapsed(), file!(), line!())
            });
            owl_shared::log_info!(
                "awq",
                "物化 {base} a8(合并键 out={out}, k={k}, 因子={factor:.3e}) @ {:?}",
                t0.elapsed()
            );
            return Ok(());
        }
        let Some((out, k, form)) = self.linear_form(base) else {
            return Err(ModelError::Msg(format!("awq/a8: {base} 形态不可解(非量化线性?)")));
        };
        if !crate::formats::w4a16::marlin_eligible(out, k) {
            return Err(ModelError::Msg(format!(
                "awq/a8: {base} (out={out}, k={k}) 不满足 marlin 打包约束(n%256==0 && k%128==0)"
            )));
        }
        let shard = base.bytes().map(|b| b as usize).sum::<usize>() % self.build_locks.len();
        let _g = self.build_locks[shard].lock().unwrap();
        let guard_key = if want_b { format!("{base}.qweight_a8") } else { format!("{base}.scales_a8") };
        {
            let cache = self.cache.lock().unwrap();
            if cache.0.contains_key(&guard_key) {
                return Ok(());
            }
        }
        let t0 = std::time::Instant::now();
        let groups = k / 32;
        // 源数据三件:q(nibble u8 行主序 [out,k])/ scales(f32 [out,groups])/
        // zp(u8 [out,groups])—— ct 形态直取,裸权重形态现场 RTN g32-sym(zp≡8)
        let (q_u8, scales_f32, zp_u8): (Vec<u8>, Vec<f32>, Vec<u8>) = match form {
            QuantForm::Ct { pe, se, ze } => {
                let packed = i32s_par(&self.maps[pe.map_ix][pe.start..pe.start + pe.nbytes]);
                let s = bf16_par(&self.maps[se.map_ix][se.start..se.start + se.nbytes]);
                let zp_i32 = i32s_par(&self.maps[ze.map_ix][ze.start..ze.start + ze.nbytes]);
                self.maps[pe.map_ix].dontneed(pe.start, pe.nbytes);
                self.maps[se.map_ix].dontneed(se.start, se.nbytes);
                self.maps[ze.map_ix].dontneed(ze.start, ze.nbytes);
                (
                    if want_b { unpack_nibbles(&packed, out, k) } else { Vec::new() },
                    s,
                    unpack_zp_ct(&zp_i32, out, groups),
                )
            }
            QuantForm::F16 { we } => {
                // 裸权重 → f32 → RTN int4 g32 对称(与 a16 臂 F16 兜底同源)
                let wb = &self.maps[we.map_ix][we.start..we.start + we.nbytes];
                let w_f32: Vec<f32> = match we.dtype {
                    safetensors::Dtype::BF16 => bf16_par(wb),
                    safetensors::Dtype::F16 => wb
                        .chunks_exact(2)
                        .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                        .collect(),
                    safetensors::Dtype::F32 => wb
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect(),
                    _ => {
                        return Err(ModelError::Msg(format!(
                            "awq/a8: {base} 裸权重 dtype 不支持"
                        )))
                    }
                };
                self.maps[we.map_ix].dontneed(we.start, we.nbytes);
                (
                    if want_b { rtn_g32_sym_packed(&w_f32, out, k) } else { Vec::new() },
                    rtn_g32_sym_scales(&w_f32, out, k),
                    vec![8u8; out * groups],
                )
            }
        };
        let (s_bits, factor) = pack_marlin_s_a8(&scales_f32, out, groups);
        let z_buf = pack_marlin_z_a8(&zp_u8, out, groups);
        let mut inserts: Vec<(String, Arc<[u8]>, usize)> = Vec::new();
        if want_b {
            let b_buf = pack_marlin_b_a8(&q_u8, k, out);
            inserts.push((
                format!("{base}.qweight_a8"),
                Arc::from(i32s_le_bytes(&b_buf).into_boxed_slice()),
                b_buf.len() * 4,
            ));
        }
        inserts.extend([
            (
                format!("{base}.scales_a8"),
                Arc::from(u16s_le_bytes(&s_bits).into_boxed_slice()),
                s_bits.len() * 2,
            ),
            (
                format!("{base}.zeros_a8"),
                Arc::from(i32s_le_bytes(&z_buf).into_boxed_slice()),
                z_buf.len() * 4,
            ),
            (
                format!("{base}.a8_factor"),
                Arc::from(factor.to_le_bytes().to_vec().into_boxed_slice()),
                4,
            ),
        ]);
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
        owl_shared::metrics::with_metrics_store(|s| {
            s.timer_record_tag("load.mat.a8", t0.elapsed(), file!(), line!())
        });
        owl_shared::log_info!(
            "awq",
            "物化 {base} a8 (out={out}, k={k}, 因子={factor:.3e}) @ {:?}",
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
                    safetensors::Dtype::BF16 => {
                        crate::f16c::bf16_bytes_to_f16_bytes(src, &mut out)
                    }
                    safetensors::Dtype::F32 => out = f32_bytes_to_f16(src),
                    _ => return None, // weight_shape 等元数据键不可取
                }
                (KeyData::Owned(out), Dtype::F16)
            }
            Resolved::Merged(parts) => {
                // 合并虚拟键(tied 双槽/元素域取数用):按序展开各段字节
                // (U32 位型 —— 与 `Bytes` 同一取数语义;不走装载热路径)
                let mut out = Vec::with_capacity(parts.iter().map(|p| p.nbytes).sum());
                for p in &parts {
                    out.extend_from_slice(&self.maps[p.map_ix][p.start..p.start + p.nbytes]);
                }
                (KeyData::Owned(out), Dtype::U32)
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

impl WeightSource for AwqSource {
    fn take(&self, key: &str) -> Option<Vec<f32>> {
        let n = self.elem_len(key)?;
        self.take_range(key, 0, n)
    }

    /// 天然 dtype(passthrough 原生键的原始字节通道;量化为 I32/派生键无原始浮点字节)
    /// —— 供 "原始字节上卡 + 设备端 cast" 臂判定(2026-10-16 装载提速定谳:
    /// 27B AWQ 的 embed/lm_head(BF16 2.54GB ×2)因本口缺位而走 CPU F16C
    /// ~1.5~3.3GB/s,占整载 ≈2.65s)。
    ///
    /// **必须廉价**(装载暂存区声扫会对全部 Want 逐个问一次)且与
    /// [`Self::copy_native_into`] 同源判定(`Some` ⟹ 字节直拷必成)。
    /// 故走 `Resolved::View` 的**等价谓词**(查索引 + 排派生键名),
    /// **绝不调 `resolve`**:后者对派生键会当场物化整键(qweight/scales/…,
    /// 27B 档 ≈14GB 主机内存)—— 2026-10-16 实测教训:声扫里调 resolve
    /// = 整载多花 3.5s(键循环 2.82s 被拉到 6.31s),且把懒物化提前成
    /// 全量物化。
    fn native_dtype(&self, key: &str) -> Option<Dtype> {
        if self.is_derived_key(key) {
            return None;
        }
        let e = self.index.get(key)?;
        match e.dtype {
            safetensors::Dtype::F32 => Some(Dtype::F32),
            safetensors::Dtype::BF16 => Some(Dtype::BF16),
            safetensors::Dtype::F16 => Some(Dtype::F16),
            _ => None,
        }
    }

    /// 原始字节区间直拷(mmap → dst;**纯 memcpy,零元素转换**)。
    ///
    /// 只服务"原生视图"键(`resolve` = `View`):派生键(qweight/scales/zeros/
    /// fc 半拆)与合并虚拟键回退元素转换路(响亮 None —— 调用方自会回退)。
    fn copy_native_into(
        &self,
        key: &str,
        offset_elems: usize,
        len: usize,
        dst: &mut [u8],
    ) -> Option<()> {
        let Resolved::View(e) = self.resolve(key)? else {
            return None;
        };
        let esz = e.esz();
        let s = e.start.checked_add(offset_elems.checked_mul(esz)?)?;
        let want = len.checked_mul(esz)?;
        if s.checked_add(want)? > e.start + e.nbytes || dst.len() < want {
            return None;
        }
        copy_par(&self.maps[e.map_ix][s..s + want], dst);
        self.maps[e.map_ix].dontneed(s, want);
        Some(())
    }

    fn elem_len(&self, key: &str) -> Option<usize> {
        // 元数据键(I64 [out,in])owl 不消费,拒绝可取化
        if key.ends_with(".weight_shape") {
            return None;
        }
        if let Some(base) = key.strip_suffix(".marlin_ws") {
            let (out, k, _) = self.linear_form(base)?;
            if !crate::formats::w4a16::marlin_eligible(out, k) {
                return None;
            }
            return Some(v2_workspace_len(out));
        }
        if let Some(base) = key.strip_suffix(".marlin_ctmp") {
            let (out, k, _) = self.linear_form(base)?;
            if !crate::formats::w4a16::marlin_eligible(out, k) {
                return None;
            }
            return Some(1);
        }
        if let Some(base) = key.strip_suffix(".qweight") {
            let (out, k, _) = self.linear_form(base)?;
            if !crate::formats::w4a16::marlin_eligible(out, k) {
                return None;
            }
            return Some(out * k / 8);
        }
        if let Some(base) = key.strip_suffix(".scales") {
            let (out, k, _) = self.linear_form(base)?;
            if !crate::formats::w4a16::marlin_eligible(out, k) {
                return None;
            }
            return Some(out * (k / 32));
        }
        if let Some(base) = key.strip_suffix(".packed_raw") {
            // 设备重排键:原始 packed 元素数(out × k/8;恒供,与 eligible 无关)
            let (out, k, _) = self.linear_form(base)?;
            if !self.index.contains_key(&format!("{base}.weight_packed")) {
                return None; // 仅 ct 形态有原始 packed(F16 兜底层无)
            }
            return Some(out * (k / 8));
        }
        if let Some(base) = key.strip_suffix(".zeros") {
            let (out, k, _) = self.linear_form(base)?;
            if !crate::formats::w4a16::marlin_eligible(out, k) {
                return None;
            }
            return Some((k / 32) * (out / 8));
        }
        if let Some(base) = key.strip_suffix(".weight") {
            if !self.index.contains_key(key) {
                let (out, k, _) = self.linear_form(base)?;
                if crate::formats::w4a16::marlin_eligible(out, k) {
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
                // 设备重排原始键:I32 packed → U32 want(4B 同宽原样直拷)
                if e.dtype == safetensors::Dtype::I32 && dtype == Dtype::U32 {
                    let s = e.start + offset_elems * 4;
                    copy_par(&self.maps[e.map_ix][s..s + need], dst);
                    self.maps[e.map_ix].dontneed(s, need);
                    return Some(());
                }
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
                    safetensors::Dtype::F16 => copy_par(win, dst),
                    safetensors::Dtype::BF16 => {
                        crate::f16c::bf16_bytes_to_f16_bytes(win, &mut dst[..need])
                    }
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
            // 合并虚拟键(两子基按序拼接):按 dst 偏移**直写**,零中间缓冲;
            // 支持任意 [offset_elems, offset_elems+len) 区间跨段(与大键分块同律)
            Resolved::Merged(parts) => {
                if dtype != Dtype::U32 {
                    return None;
                }
                let s = offset_elems.checked_mul(4)?;
                let e = s.checked_add(need)?;
                let mut pos = 0usize; // 逻辑拼接内的游标
                let mut w = 0usize; // dst 写入位
                for p in &parts {
                    let p_end = pos.checked_add(p.nbytes)?;
                    if p_end <= s {
                        pos = p_end;
                        continue;
                    }
                    if pos >= e {
                        break;
                    }
                    let from = s.saturating_sub(pos);
                    let to = (e - pos).min(p.nbytes);
                    let len_p = to - from;
                    let src = &self.maps[p.map_ix][p.start + from..p.start + to];
                    copy_par(src, &mut dst[w..w + len_p]);
                    self.maps[p.map_ix].dontneed(p.start + from, len_p);
                    w += len_p;
                    pos = p_end;
                }
                if w != need {
                    return None;
                }
                Some(())
            }
            Resolved::Bytes(b, reg) => {
                if reg != dtype {
                    return None;
                }
                let s = offset_elems * esz;
                copy_par(b.get(s..s + need)?, dst);
                Some(())
            }
        }
    }
}

    /// **a8 键族谓词**(S2):W4A8 的四件产物 —— 与 a16 键族并列(层侧按 plan 二选一)
fn is_a8_key(key: &str) -> bool {
    key.ends_with(".qweight_a8")
        || key.ends_with(".scales_a8")
        || key.ends_with(".zeros_a8")
        || key.ends_with(".a8_factor")
}

/// **bf16 源 → marlin scales 字节**(单遍融合;装载热路径的内部器)。
///
/// `parts` = 源 bf16 分段(1 或 2 段;2 段 = 合并虚拟键左右半,按序拼接);
/// `idx` = [`marlin_scales_index`] 源下标表;逐位等价原链
/// `bf16_par → pack_marlin_s → u16s_le_bytes`。
pub(crate) fn pack_scales_bf16_into(
    parts: &[&[u8]],
    out: usize,
    groups: usize,
    idx: &[u32],
    dst: &mut [u8],
) {
    use rayon::prelude::*;
    let n = out * groups;
    assert_eq!(idx.len(), n, "scales 索引表长度 = (groups,out) 展平");
    assert!(dst.len() >= n * 2, "scales 目标字节不足");
    // 段数只容 1/2(多于两段 = 亮声拒;合并虚拟键只有左右半)
    let (a, b, cut) = match parts {
        [] => panic!("scales 源分段为空"),
        [a] => (*a, *a, usize::MAX),
        [a, b] => (*a, *b, a.len() / 2),
        _ => panic!("scales 源分段 > 2:{} 段未支持", parts.len()),
    };
    // 源覆盖度校验(响亮:索引表指到段外 = 编程错,不得静默读错字节)
    let esrc = a.len() / 2 + if parts.len() == 2 { b.len() / 2 } else { 0 };
    assert!(
        a.len() % 2 == 0 && (parts.len() != 2 || b.len() % 2 == 0),
        "scales 源字节量必须偶数"
    );
    assert!(esrc >= n, "scales 源元素数 {esrc} < 目标 {n}");
    // 分块并行(块 = 2Ki 输出元素;块内串行递推,免逐元素派发税)
    const BLK: usize = 1 << 11;
    dst[..n * 2]
        .par_chunks_mut(BLK * 2)
        .enumerate()
        .for_each(|(t, dchunk)| {
            let base = t * BLK;
            for (k, d2) in dchunk.chunks_exact_mut(2).enumerate() {
                let g = idx[base + k] as usize;
                let (src, o) = if g < cut { (a, g) } else { (b, g - cut) };
                let bits = u16::from_le_bytes([src[o * 2], src[o * 2 + 1]]);
                let f = f32::from_bits((bits as u32) << 16);
                d2.copy_from_slice(&half::f16::from_f32(f).to_le_bytes());
            }
        });
}

/// RTN g32-sym 现场量化(q packed U4B8 语义):q = clamp(round(w/s), −8, 7)+8,
/// s = absmax/7 per (o, g32)。输出 nibble LSB-first 沿 k 的 u8 [out × k]
/// (unpack_nibbles 逆;供 pack_marlin_b gather)。
pub(crate) fn rtn_g32_sym_packed(w: &[f32], out: usize, k: usize) -> Vec<u8> {
    use rayon::prelude::*;
    let groups = k / 32;
    let mut q = vec![0u8; out * k];
    q.par_chunks_mut(k)
        .enumerate()
        .for_each(|(o, row)| {
            for gg in 0..groups {
                let base_w = &w[o * k + gg * 32..o * k + (gg + 1) * 32];
                let amax = base_w.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-12);
                let scale = amax / 7.0;
                for (j, v) in base_w.iter().enumerate() {
                    let qi = (v / scale).round().clamp(-8.0, 7.0) + 8.0;
                    row[gg * 32 + j] = qi as u8;
                }
            }
        });
    q
}

/// RTN g32-sym scales:f32 [out, groups](absmax/7;与 [`rtn_g32_sym_packed`] 同窗)
pub(crate) fn rtn_g32_sym_scales(w: &[f32], out: usize, k: usize) -> Vec<f32> {
    let groups = k / 32;
    let mut s = vec![0f32; out * groups];
    for o in 0..out {
        for gg in 0..groups {
            let amax = w[o * k + gg * 32..o * k + (gg + 1) * 32]
                .iter()
                .fold(0f32, |m, v| m.max(v.abs()))
                .max(1e-12);
            s[o * groups + gg] = amax / 7.0;
        }
    }
    s
}

/// AWQ 非对称反量化(host rayon;非 eligible 小线性专用):
/// W[o, i] = (nibble_i(q_packed[o, i/8]) − zp[o, g]) × scale[o, g],g = i/32。
/// 输出 f16 LE [out × k]。
pub(crate) fn dequant_u4_awq_f16_bytes(
    packed: &[i32],
    zp_packed: &[i32],
    scales: &[f32],
    out: usize,
    k: usize,
    dst: &mut [u8],
) {
    use rayon::prelude::*;
    assert_eq!(packed.len(), out * (k / 8));
    assert_eq!(zp_packed.len(), (out / 8) * (k / 32));
    assert_eq!(scales.len(), out * (k / 32));
    assert_eq!(dst.len(), out * k * 2);
    dst.par_chunks_mut(k * 2)
        .enumerate()
        .for_each(|(o, row)| {
            let groups = k / 32;
            for g in 0..groups {
                let z = ((zp_packed[(o / 8) * groups + g] as u32) >> (4 * (o % 8))) & 0xF;
                let s = scales[o * groups + g];
                let base = &mut row[g * 32 * 2..(g + 1) * 32 * 2];
                for j in 0..32 {
                    let i = g * 32 + j;
                    let q = ((packed[o * (k / 8) + i / 8] as u32) >> (4 * (i % 8))) & 0xF;
                    let w = (q as f32 - z as f32) * s;
                    let h = half::f16::from_f32(w).to_le_bytes();
                    base[j * 2] = h[0];
                    base[j * 2 + 1] = h[1];
                }
            }
        });
}

/// passthrough F32→F16(rayon;w4a16 同款)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::tmp_path;

    /// 合成 ct 检查点目录(weight_packed I32 / weight_scale BF16 /
    /// weight_zero_point I32 / weight_shape I64)。
    fn write_ct_dir(
        dir: &Path,
        base: &str,
        packed: &[i32],
        scales_bf16: &[u8],
        zp: &[i32],
        out: usize,
        k: usize,
    ) {
        write_ct_named(dir, "model.safetensors", base, packed, scales_bf16, zp, out, k);
    }

    /// 同上但可指定文件名(多子基检查点:合并虚拟键的对拍需要两个文件共存)
    #[allow(clippy::too_many_arguments)]
    fn write_ct_named(
        dir: &Path,
        file_name: &str,
        base: &str,
        packed: &[i32],
        scales_bf16: &[u8],
        zp: &[i32],
        out: usize,
        k: usize,
    ) {
        owl_shared::file_loader::create_dir_all(dir).expect("mkdir");
        let mut header = String::from("{");
        let mut blobs: Vec<Vec<u8>> = Vec::new();
        let mut offset = 0usize;
        let entry = |header: &mut String, name: &str, dtype: &str, shape: &[usize], bytes: Vec<u8>, offset: &mut usize| {
            let n = bytes.len();
            header.push_str(&format!(
                "{}\"{}\": {{\"dtype\": \"{}\", \"shape\": [{}], \"data_offsets\": [{}, {}]}}",
                if *offset == 0 { "" } else { ", " },
                name, dtype,
                shape.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","),
                *offset, *offset + n
            ));
            *offset += n;
            bytes
        };
        blobs.push(entry(&mut header, &format!("{base}.weight_packed"), "I32", &[out, k / 8],
            packed.iter().flat_map(|v| v.to_le_bytes()).collect(), &mut offset));
        blobs.push(entry(&mut header, &format!("{base}.weight_scale"), "BF16", &[out, k / 32],
            scales_bf16.to_vec(), &mut offset));
        blobs.push(entry(&mut header, &format!("{base}.weight_zero_point"), "I32", &[out / 8, k / 32],
            zp.iter().flat_map(|v| v.to_le_bytes()).collect(), &mut offset));
        blobs.push(entry(&mut header, &format!("{base}.weight_shape"), "I64", &[2],
            [(out as i64), (k as i64)].iter().flat_map(|v| v.to_le_bytes()).collect(), &mut offset));
        header.push('}');
        let mut file = Vec::new();
        file.extend_from_slice(&(header.len() as u64).to_le_bytes());
        file.extend_from_slice(header.as_bytes());
        for b in blobs {
            file.extend_from_slice(&b);
        }
        owl_shared::file_loader::write(dir.join(file_name), file).expect("写 ct 检查点");
    }

    /// **合并虚拟键直写对拍**(2026-10-16 装载提速):
    /// `in_proj_qkvz.packed_raw`(两子基 `weight_packed` 按序拼接)的
    /// **零中间缓冲直写**必须 ≡ 参考实现"先拼接全量再按区间切片",
    /// 且覆盖:整键/头部/跨段边界/尾部/乱序窗口 + 越界响亮拒。
    #[test]
    fn merged_packed_raw_direct_write_parity() {
        let (out1, out2, k) = (256usize, 128usize, 256usize);
        let dir = tmp_path("awq_merged_raw");
        let n1 = out1 * (k / 8);
        let n2 = out2 * (k / 8);
        let p1: Vec<i32> = (0..n1).map(|i| (i as i32).wrapping_mul(0x0BB6_7AE1) ^ 0x5a5a).collect();
        let p2: Vec<i32> = (0..n2).map(|i| (i as i32).wrapping_mul(0x9E37_79B1_u32 as i32)).collect();
        let mk_scales = |n: usize| f32_to_bf16_bytes(&vec![0.25f32; n]);
        let mk_zp = |n: usize| vec![0i32; n];
        write_ct_named(
            &dir,
            "a.safetensors",
            "l.in_proj_qkv",
            &p1,
            &mk_scales(out1 * (k / 32)),
            &mk_zp((out1 / 8) * (k / 32)),
            out1,
            k,
        );
        write_ct_named(
            &dir,
            "b.safetensors",
            "l.in_proj_z",
            &p2,
            &mk_scales(out2 * (k / 32)),
            &mk_zp((out2 / 8) * (k / 32)),
            out2,
            k,
        );
        let src = AwqSource::open_dir(&dir).expect("open");
        let key = "l.in_proj_qkvz.packed_raw";
        assert!(
            matches!(src.resolve(key), Some(Resolved::Merged(_))),
            "合并键应解析为 Merged 视图"
        );
        let mut want: Vec<u8> = Vec::with_capacity((n1 + n2) * 4);
        want.extend(p1.iter().flat_map(|v| v.to_le_bytes()));
        want.extend(p2.iter().flat_map(|v| v.to_le_bytes()));
        let n = n1 + n2;
        let windows = [
            (0usize, n),
            (0, 1),
            (0, 7),
            (n1 - 3, 9),
            (n1, n2),
            (n1 - 64, 128),
            (n - 5, 5),
            (12, 1000),
            (77, 4096),
            (n / 2, n - n / 2),
        ];
        for (off, len) in windows {
            let mut got = vec![0u8; len * 4];
            assert!(
                src.convert_chunk_into_bytes(key, off, len, &mut got, Dtype::U32).is_some(),
                "窗口 [{off},{}) 应可直写",
                off + len
            );
            assert_eq!(
                got,
                want[off * 4..(off + len) * 4],
                "窗口 [{off},{}) 字节不一致",
                off + len
            );
        }
        // 越界/宽度不符 = 响亮 None(禁静默截断)
        let mut small = vec![0u8; 4];
        assert!(src.convert_chunk_into_bytes(key, n - 1, 2, &mut small, Dtype::U32).is_none());
        assert!(src.convert_chunk_into_bytes(key, 0, n, &mut small, Dtype::U32).is_none());
        assert!(src.convert_chunk_into_bytes(key, 0, 1, &mut small, Dtype::F16).is_none());
    }

    /// **scales 单遍融合**逐位对拍:融合直出字节 ≡ 原链
    /// (`bf16_par` → `pack_marlin_s` → `u16s_le_bytes`),单段与两段都验。
    #[test]
    fn scales_fused_bytes_parity() {
        use owl_kernels::family::marlin::repack::{marlin_scales_index, pack_marlin_s};
        let (out, k) = (256usize, 256usize);
        let groups = k / 32;
        let dir = tmp_path("awq_scales_fused");
        let scales: Vec<f32> = (0..out * groups).map(|i| 0.013 + (i % 521) as f32 * 2e-4).collect();
        let sb = f32_to_bf16_bytes(&scales);
        write_ct_dir(&dir, "l", &vec![0i32; out * (k / 8)], &sb, &vec![0i32; (out / 8) * groups], out, k);
        let src = AwqSource::open_dir(&dir).expect("open");
        let view = &src.maps[src.index["l.weight_scale"].map_ix]
            [src.index["l.weight_scale"].start..src.index["l.weight_scale"].start + src.index["l.weight_scale"].nbytes];
        // 单段
        let idx = marlin_scales_index(out, groups);
        let want = u16s_le_bytes(&pack_marlin_s(&bf16_par(view), out, groups));
        let got = src.pack_scales_bytes(&[view], out, k);
        assert_eq!(got, want, "单段融合逐位");
        assert_eq!(idx.len(), out * groups);
        // 两段(合并虚拟键语义:右半拼在左半之后)
        let (o1, o2) = (out / 2, out - out / 2);
        let (v1, v2) = (view, view);
        let merged = src.pack_scales_bytes(&[v1, v2], out, k);
        let mut f32_merged = bf16_par(v1);
        f32_merged.extend(bf16_par(v2));
        let want2 = u16s_le_bytes(&pack_marlin_s(&f32_merged[..out * groups], out, groups));
        assert_eq!(merged, want2, "两段融合逐位");
        let _ = (o1, o2);
    }

    /// **a8 四件物化对拍**(S2):同源数据经 `AwqSource` 键族取出的字节
    /// 必须与**直接调用 a8 三件套**逐位相等(+ 全局因子键 = `max/4096`)
    #[test]
    fn awq_source_a8_keys_match_packers() {
        use owl_kernels::family::marlin::repack::{
            pack_marlin_b_a8, pack_marlin_s_a8, pack_marlin_z_a8, unpack_nibbles,
        };
        let (out, k) = (256usize, 256usize); // marlin_eligible:n%256==0 && k%128==0
        let groups = k / 32;
        let dir = tmp_path("awq_a8_keys");
        let packed: Vec<i32> = (0..out * (k / 8))
            .map(|i| (i as i32).wrapping_mul(0x0BB6_7AE1).wrapping_mul(0x9E37_79B1_u32 as i32))
            .collect();
        let zp: Vec<i32> = (0..(out / 8) * groups)
            .map(|i| (i as i32).wrapping_mul(0x85EB_CA6B_u32 as i32) ^ 0x1234_5678)
            .collect();
        let scales: Vec<f32> = (0..out * groups).map(|i| 0.01 + (i % 23) as f32 * 0.002).collect();
        let sb = f32_to_bf16_bytes(&scales);
        write_ct_dir(&dir, "l", &packed, &sb, &zp, out, k);
        let src = AwqSource::open_dir(&dir).expect("open");
        // ⚠️ 期望必须用**检查点里真正存的** bf16 位型(a8 归一对尺度精度敏感:
        // 拿裸 f32 做期望会差一个 bf16 舍入 —— 2026-10-16 首跑实测)
        let scales_bf16 = bf16_par(&sb);

        let take = |key: &str| -> (Vec<u8>, Dtype) {
            let (d, dt) = src.key_bytes(key).unwrap_or_else(|| panic!("{key} 取不到"));
            (d.as_slice().to_vec(), dt)
        };
        // 期望 = 直接过 a8 三件套(a16 臂同源的 unpack 两步)
        let want_b = i32s_le_bytes(&pack_marlin_b_a8(&unpack_nibbles(&packed, out, k), k, out));
        let (s_bits, factor) = pack_marlin_s_a8(&scales_bf16, out, groups);
        let want_s = u16s_le_bytes(&s_bits);
        let want_z = i32s_le_bytes(&pack_marlin_z_a8(&unpack_zp_ct(&zp, out, groups), out, groups));

        let (b, dt_b) = take("l.qweight_a8");
        let (s_, dt_s) = take("l.scales_a8");
        let (z, dt_z) = take("l.zeros_a8");
        let (f, dt_f) = take("l.a8_factor");
        assert_eq!(b, want_b, "qweight_a8 逐位(32×32 tile 布局)");
        assert_eq!(s_, want_s, "scales_a8 逐位(perm_single + int16 归一)");
        assert_eq!(z, want_z, "zeros_a8 逐位(无 n-interleave)");
        assert_eq!(f, factor.to_le_bytes().to_vec(), "a8_factor = max/4096");
        assert_eq!((dt_b, dt_s, dt_z, dt_f), (Dtype::U32, Dtype::F16, Dtype::U32, Dtype::F32), "线 dtype");
        // 尺寸自查(位数与 GEMM 槽契约对望)
        assert_eq!(b.len(), (k / 16) * (out * 16 / 8) * 4, "B 字数 = (k/16, n*2) i32");
        assert_eq!(s_.len(), groups * out * 2, "scales 字数 = (k/g, n) 2B");
        assert_eq!(z.len(), groups * (out / 8) * 4, "zp 字数 = (k/g, n/8) i32");
        assert_eq!(f.len(), 4, "因子 = f32[1]");
    }

    /// `pack_marlin_s` 索引表化**逐位对拍**(冻结参考实现 = 原现算表式)
    #[test]
    fn marlin_scales_index_pack_parity() {
        use owl_kernels::family::marlin::repack::{marlin_scale_perm, pack_marlin_s};
        fn pack_marlin_s_ref(scales: &[f32], out: usize, groups: usize) -> Vec<u16> {
            let sp = marlin_scale_perm();
            let mut packed = vec![0u16; groups * out];
            for gi in 0..groups {
                for o in 0..out {
                    let idx = gi * out + o;
                    let src_idx = (idx / 64) * 64 + sp[idx % 64];
                    let (sg, so) = (src_idx / out, src_idx % out);
                    packed[idx] = half::f16::from_f32(scales[so * groups + sg]).to_bits();
                }
            }
            packed
        }
        for (out, groups) in [(64usize, 2usize), (128, 4), (256, 1), (5120, 544)] {
            let n = out * groups;
            let scales: Vec<f32> = (0..n).map(|i| ((i * 2654435761) % 65536) as f32 / 8192.0).collect();
            let idx = marlin_scales_index(out, groups);
            let got = pack_marlin_s_idx(&scales, out, groups, &idx);
            assert_eq!(got, pack_marlin_s_ref(&scales, out, groups), "(out={out}, groups={groups}) 逐位");
            assert_eq!(pack_marlin_s(&scales, out, groups), got, "壳函数与索引路一致");
        }
    }

    fn f32_to_bf16_bytes(v: &[f32]) -> Vec<u8> {
        v.iter()
            .flat_map(|f| ((f.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    /// 非 eligible 线性(out=64, in=32):反量化臂数值 = (q − zp) × s。
    #[test]
    fn awq_source_dequant_non_eligible() {
        let (out, k) = (64usize, 32usize);
        let groups = k / 32;
        let dir = tmp_path("awq_src_deq");
        let packed: Vec<i32> = (0..out * (k / 8))
            .map(|i| (i as i32).wrapping_mul(0x0BB6_7AE1).wrapping_mul(0x9E37_79B1_u32 as i32))
            .collect();
        let zp: Vec<i32> = (0..(out / 8) * groups)
            .map(|i| (i as i32).wrapping_mul(0x85EB_CA6B_u32 as i32) ^ 0x1234_5678)
            .collect();
        let scales: Vec<f32> = (0..out * groups).map(|i| 0.01 + (i % 17) as f32 * 0.003).collect();
        write_ct_dir(&dir, "l", &packed, &f32_to_bf16_bytes(&scales), &zp, out, k);

        let src = AwqSource::open_dir(&dir).expect("open");
        // 非 eligible:qweight 不可取,weight 反量化可取
        assert!(src.elem_len("l.qweight").is_none(), "非 eligible 不得供 qweight");
        assert_eq!(src.elem_len("l.weight"), Some(out * k), "反量化臂尺寸");
        let w = src.take("l.weight").expect("take weight");
        // host 参考:nibble t of word c = 行 8c+t(zp 打包维 = out)
        for o in 0..out {
            for g in 0..groups {
                let z = ((zp[(o / 8) * groups + g] as u32) >> (4 * (o % 8))) & 0xF;
                for j in 0..32 {
                    let i = g * 32 + j;
                    let q = ((packed[o * (k / 8) + i / 8] as u32) >> (4 * (i % 8))) & 0xF;
                    let want = (q as f32 - z as f32) * scales[o * groups + g];
                    assert!(
                        (w[o * k + i] - want).abs() < 2e-2,
                        "({o},{i}): {} vs {want}",
                        w[o * k + i]
                    );
                }
            }
        }
    }

    /// eligible 线性(out=256, in=128):派生键尺寸 + passthrough + 裸键。
    #[test]
    fn awq_source_marlin_keys_surface() {
        let (out, k) = (256usize, 128usize);
        let groups = k / 32;
        let dir = tmp_path("awq_src_marlin");
        let packed: Vec<i32> = (0..out * (k / 8)).map(|i| i as i32 * 7 + 3).collect();
        let zp: Vec<i32> = (0..(out / 8) * groups).map(|i| i as i32 * 13 + 1).collect();
        let scales: Vec<f32> = (0..out * groups).map(|i| 0.02 + (i % 7) as f32 * 0.004).collect();
        write_ct_dir(&dir, "l", &packed, &f32_to_bf16_bytes(&scales), &zp, out, k);

        let src = AwqSource::open_dir(&dir).expect("open");
        assert_eq!(src.elem_len("l.qweight"), Some(out * k / 8));
        assert_eq!(src.elem_len("l.scales"), Some(out * groups));
        assert_eq!(src.elem_len("l.zeros"), Some(groups * (out / 8)));
        assert_eq!(
            src.elem_len("l.marlin_ws"),
            Some(owl_kernels::family::marlin::v2_workspace_len(out))
        );
        assert_eq!(src.elem_len("l.marlin_ctmp"), Some(1));
        // zeros 键可取(U32 位型;字节级正确性由 GPU 对拍核内验证)
        let z = src.take("l.zeros").expect("take zeros");
        assert_eq!(z.len(), groups * (out / 8));
        // take_range 分块一致性(w4a16 同律)
        let whole = src.take("l.scales").expect("take scales");
        let half1 = src.take_range("l.scales", 0, whole.len() / 2).expect("r1");
        assert_eq!(&whole[..half1.len()], &half1[..], "分块 = 整取前缀");
    }

    /// 本进程累计磁盘读字节(`/proc/self/io`)—— 区分"页缓存命中"与"真磁盘读"
    fn io_read_bytes() -> u64 {
        std::fs::read_to_string("/proc/self/io")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find_map(|l| l.strip_prefix("read_bytes: ").and_then(|v| v.trim().parse().ok()))
            })
            .unwrap_or(0)
    }

    /// N 线程分块拷贝(与 `safetensors::copy_native_into` 同款)
    fn par_copy(src: &[u8], dst: &mut [u8], threads: usize) {
        if threads <= 1 {
            dst.copy_from_slice(src);
            return;
        }
        let chunk = src.len().div_ceil(threads);
        let ds: Vec<&mut [u8]> = dst.chunks_mut(chunk).collect();
        let ss: Vec<&[u8]> = src.chunks(chunk).collect();
        std::thread::scope(|sc| {
            for (d, s) in ds.into_iter().zip(ss) {
                sc.spawn(move || d.copy_from_slice(s));
            }
        });
    }

    /// 一段计账(逐次打印 + 磁盘读增量)
    fn bench_seg(tag: &str, bytes: usize, reps: usize, mut f: impl FnMut(usize)) {
        let r0 = io_read_bytes();
        let mut times = Vec::new();
        for i in 0..reps {
            let t = std::time::Instant::now();
            f(i);
            times.push(t.elapsed().as_secs_f64());
        }
        let dr = io_read_bytes().saturating_sub(r0);
        let all: Vec<String> = times
            .iter()
            .map(|d| format!("{:>7.1}ms/{:>5.2}GB/s", d * 1e3, bytes as f64 / 1e9 / d))
            .collect();
        eprintln!(
            "[iobench] {tag:<32} {} 磁盘读 {:+.0}MB",
            all.join("  "),
            dr as f64 / 1e6
        );
    }

    /// **scales 构建链微基准**(门控 `OWL_AWQ_IOBENCH=1`):旧链
    /// (`bf16_par` → `pack_marlin_s` → `u16s_le_bytes`)vs 新单遍融合
    /// (`pack_scales_bytes`)—— 装载提速的回归哨(2026-10-16:21ms/键 → 8ms/键)。
    #[test]
    fn bench_scales_chain() {
        if !owl_shared::env_reader::flag("OWL_AWQ_IOBENCH") {
            eprintln!("skip: OWL_AWQ_IOBENCH 未开");
            return;
        }
        use owl_kernels::family::marlin::repack::pack_marlin_s;
        // 27B mlp 档尺度:out=5120, k=17408 ⇒ groups=544, scales=2.78M 元素
        let (out, k) = (5120usize, 17408usize);
        let groups = k / 32;
        let n = out * groups;
        let scales_f32: Vec<f32> = (0..n).map(|i| 0.01 + (i % 977) as f32 * 1e-5).collect();
        let src_bf16 = f32_to_bf16_bytes(&scales_f32);
        eprintln!("[iobench] scales 链(27B mlp 档):out={out} groups={groups} 元素={n}");
        bench_seg("旧链 bf16_par+spack+bytes", n * 2, 3, |_| {
            std::hint::black_box(u16s_le_bytes(&pack_marlin_s(&bf16_par(&src_bf16), out, groups)));
        });
        let idx = marlin_scales_index(out, groups);
        let mut dst = vec![0u8; n * 2];
        bench_seg("索引表 gather(u16)", n * 2, 3, |_| {
            std::hint::black_box(pack_marlin_s_idx(&scales_f32, out, groups, &idx));
        });
        bench_seg("单遍融合直出字节", n * 2, 3, |_| {
            pack_scales_bf16_into(&[&src_bf16], out, groups, &idx, &mut dst);
        });
    }

    /// **装载 I/O 通路微基准**(瓶颈定位用;门控 `OWL_AWQ_IOBENCH=1`)。
    ///
    /// 逐键量:① `resolve()` 单独(合并虚拟键的构建成本)② 现行
    /// `convert_chunk_into_bytes`(resolve + 拷贝 + DONTNEED)③ 裸拷 mmap 视图
    /// (1T/4T —— 内存带宽上限)④ 合并键的两子基直拷(理想实现)。
    /// dst 先预热(隔离目标页缺页);每段带 `/proc/self/io` 磁盘读增量。
    #[test]
    fn bench_raw_copy_paths() {
        if !owl_shared::env_reader::flag("OWL_AWQ_IOBENCH") {
            eprintln!("skip: OWL_AWQ_IOBENCH 未开");
            return;
        }
        let dir = std::path::PathBuf::from(owl_shared::env_reader::str("OWL_CKPT_27B_AWQ").unwrap_or_else(|| {
            "/home/div/Documents/codes/models/cyankiwi/Qwen3.8-27B-AWQ-INT4".into()
        }));
        let src = AwqSource::open_dir(&dir).expect("源");
        let want = owl_shared::env_reader::str("OWL_AWQ_IOBENCH_KEYS").unwrap_or_else(|| {
            "model.language_model.layers.0.linear_attn.in_proj_qkvz.packed_raw,model.language_model.layers.0.mlp.down_proj.packed_raw,model.language_model.embed_tokens.weight".into()
        });
        for key in want.split(',').map(str::trim).filter(|k| !k.is_empty()) {
            let first = src.resolve(key);
            let nbytes = match &first {
                Some(Resolved::View(e)) => e.nbytes,
                Some(Resolved::Bytes(b, _)) => b.len(),
                other => {
                    eprintln!("[iobench] 跳过 {key}(臂 {} 不适用)", matches!(other, None));
                    continue;
                }
            };
            let dtype = if key.ends_with(".packed_raw") { Dtype::U32 } else { Dtype::F16 };
            let elems = nbytes / dtype.size_bytes();
            let arm = match src.resolve(key) {
                Some(Resolved::View(e)) => format!("View({:?},{}MB)", e.dtype, e.nbytes / 1_000_000),
                Some(Resolved::Bytes(b, d)) => format!("Bytes({d:?},{:.1}MB)", b.len() as f64 / 1e6),
                Some(Resolved::Merged(ps)) => format!(
                    "Merged({} 段,{:.1}MB)",
                    ps.len(),
                    ps.iter().map(|p| p.nbytes).sum::<usize>() as f64 / 1e6
                ),
                Some(Resolved::Zeroed) => "Zeroed".to_string(),
                None => "None".to_string(),
            };
            eprintln!("[iobench] 键 {key}");
            eprintln!("[iobench]   臂={arm} 字节={:.1}MB", nbytes as f64 / 1e6);
            let mut dst = vec![0u8; nbytes];
            dst.fill(0x5a); // dst 预热(不把目标缺页算进拷贝)
            // ① resolve 单独(合并虚拟键 = 每次调用重建整个缓冲)
            bench_seg("① resolve() 单独", nbytes, 3, |_| {
                std::hint::black_box(src.resolve(key));
            });
            // ② 现行 API(装载域唯一入口)
            bench_seg("② convert_chunk_into_bytes", nbytes, 3, |_| {
                src.convert_chunk_into_bytes(key, 0, elems, &mut dst, dtype).expect("convert");
            });
            // ③ 裸拷:mmap 视图(仅 View 臂可比)
            if let Some(Resolved::View(e)) = src.resolve(key) {
                let r0 = io_read_bytes();
                {
                    let view = &src.maps[e.map_ix][e.start..e.start + e.nbytes];
                    bench_seg("③ 裸拷 mmap 视图 1T", nbytes, 3, |_| dst.copy_from_slice(view));
                    bench_seg("③' 裸拷 mmap 视图 4T", nbytes, 3, |_| par_copy(view, &mut dst, 4));
                }
                let _ = io_read_bytes().saturating_sub(r0);
            }
            // ④ 合并虚拟键的两子基直拷(理想:零中间缓冲)
            if let Some(base) = key.strip_suffix(".packed_raw") {
                if let Some((b1, b2)) = crate::formats::split_qkvz(base) {
                    let (e1, e2) = (
                        src.index.get(&format!("{b1}.weight_packed")).cloned(),
                        src.index.get(&format!("{b2}.weight_packed")).cloned(),
                    );
                    if let (Some(e1), Some(e2)) = (e1, e2) {
                        bench_seg("④ 两子基直拷 1T(理想)", nbytes, 3, |_| {
                            let (a, b) = dst.split_at_mut(e1.nbytes);
                            a.copy_from_slice(&src.maps[e1.map_ix][e1.start..e1.start + e1.nbytes]);
                            b.copy_from_slice(&src.maps[e2.map_ix][e2.start..e2.start + e2.nbytes]);
                        });
                        bench_seg("④' 两子基直拷 4T(理想)", nbytes, 3, |_| {
                            let (a, b) = dst.split_at_mut(e1.nbytes);
                            par_copy(&src.maps[e1.map_ix][e1.start..e1.start + e1.nbytes], a, 4);
                            par_copy(&src.maps[e2.map_ix][e2.start..e2.start + e2.nbytes], b, 4);
                        });
                        // ⑤ 中间缓冲成本:每轮新建 Vec + 两拷贝 + drop
                        bench_seg("⑤ 中间 Vec 建+拷+drop", nbytes, 3, |_| {
                            let mut buf = Vec::with_capacity(nbytes);
                            buf.extend_from_slice(
                                &src.maps[e1.map_ix][e1.start..e1.start + e1.nbytes],
                            );
                            buf.extend_from_slice(
                                &src.maps[e2.map_ix][e2.start..e2.start + e2.nbytes],
                            );
                            dst.copy_from_slice(&buf);
                            std::hint::black_box(buf.len());
                        });
                        // ⑤' 同链但缓冲复用(只省 alloc/首触)
                        let mut keep = Vec::with_capacity(nbytes);
                        bench_seg("⑤' 中间 Vec 复用", nbytes, 3, |_| {
                            keep.clear();
                            keep.extend_from_slice(
                                &src.maps[e1.map_ix][e1.start..e1.start + e1.nbytes],
                            );
                            keep.extend_from_slice(
                                &src.maps[e2.map_ix][e2.start..e2.start + e2.nbytes],
                            );
                            dst.copy_from_slice(&keep);
                            std::hint::black_box(keep.len());
                        });
                    }
                }
            }
        }
    }
}

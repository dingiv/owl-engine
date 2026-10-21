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
use crate::formats::util::{f32_bytes_to_f16, marlin_eligible};
use owl_kernels::family::marlin::v2_workspace_len;

use crate::module::{Taken, WeightSource};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// 懒物化缓存容量上限(27B 专用档):同 base 三键(qweight/scales/zeros)
/// 被 LPT 分桶拉到不同装载桶的不同位置,存活窗口 = 整个装载期 —— cap 必须
/// 容下全模型物化态(~14GB;RAM 108GB),否则 clear 全清把先建键清掉,
/// 后到键 3.3× 重复构建(2026-10-01 实测:512MB cap → 1183 次构建,
/// 16GB → 355 次)。clear 计数入 dbg.cache_clear 盯防。
pub(crate) const CACHE_CAP_BYTES: usize = 16 << 30;

/// 量化线性源形态(见 [`AwqSource::linear_form`])
pub(crate) enum QuantForm<'a> {
    Ct {
        pe: &'a RawEntry,
        se: &'a RawEntry,
        ze: &'a RawEntry,
    },
    F16 { we: &'a RawEntry },
}

/// 索引表种类(形状索引缓存键的第三轴;原 `bool` 只容 gather/fused 两态)
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum IdxKind {
    /// `marlin_gather_indices`(F16 兜底臂的 nibble gather)
    Gather,
    /// `marlin_fused_indices`(Ct 臂的 packed_word<<3|nib 融合表)
    Fused,
    /// `marlin_scales_index`(scales 转置+perm 的源下标表)
    Scales,
    /// `marlin_scales_index_a8`(同上,a8 的 32-chunk perm_single 档)
    ScalesA8,
}

/// 键字节解析结果(懒物化后的一致字节 + 登记 dtype)
pub(crate) enum Resolved {
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
    pub(crate) maps: Vec<Mmap>,
    pub(crate) index: HashMap<String, RawEntry>,
    /// 懒物化字节缓存(键 → 字节;容量清空驱逐)
    #[allow(clippy::type_complexity)] // 行块缓存(字节镜像 + 计数)
    pub(crate) cache: Mutex<(HashMap<String, Arc<[u8]>>, usize)>,
    /// gather/fused 索引表(按 (k, out, fused?) 形状复用;除法分解只算一次)
    #[allow(clippy::type_complexity)] // 索引缓存(键 → 列序)
    pub(crate) idx_cache: Mutex<HashMap<(usize, usize, IdxKind), Arc<Vec<u32>>>>,
    /// 构建分片锁(16 片按 base hash;同 base 互斥、异 base 并行 ——
    /// 全局单锁时 build 串行链 ≈ 装载墙钟主导,分片后 4 桶并发 build
    /// 与 DMA 流水重叠)
    pub(crate) build_locks: Vec<Mutex<()>>,
    /// 设备重排模式(装载域 set;true = CPU 构建走轻路径:量化大键的
    /// marlin B 由 GPU repack 产出,CPU 仅产 scales/zeros/packed_raw_ct)
    pub(crate) device_repack: std::cell::Cell<bool>,
}

impl AwqSource {
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
    pub(crate) fn linear_form(
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


    /// 键解析(唯一入口;懒物化在此触发)。
    ///
    /// 三态(2026-10-17 审查 §③):`Ok(None)` = 本键不归我管(正常语义);
    /// `Err` = 物化真失败(经 `linear_bytes_for` 一路上抛到装载入口)。
    fn resolve(&self, key: &str) -> Taken<Resolved> {
        if key.ends_with(".marlin_ws") || key.ends_with(".marlin_ctmp") {
            return Ok(Some(Resolved::Zeroed));
        }
        if let Some(base) = key.strip_suffix(".qweight") {
            let Some(__b) = self.linear_bytes_for(base, key)? else { return Ok(None) };
            return Ok(Some(Resolved::Bytes(__b, Dtype::U32)));
        }
        if let Some(base) = key.strip_suffix(".scales") {
            let Some(__b) = self.linear_bytes_for(base, key)? else { return Ok(None) };
            return Ok(Some(Resolved::Bytes(__b, Dtype::F16)));
        }
        if let Some(base) = key.strip_suffix(".zeros") {
            let Some(__b) = self.linear_bytes_for(base, key)? else { return Ok(None) };
            return Ok(Some(Resolved::Bytes(__b, Dtype::U32)));
        }
        // ---- P1.5-a:int8 per-row 头(packed-⌚ lm_head;源 = bf16 权重)----
        if let Some(base) = key.strip_suffix(".i8q") {
            let Some(__b) = self.linear_bytes_for(base, key)? else { return Ok(None) };
            return Ok(Some(Resolved::Bytes(__b, Dtype::I8)));
        }
        if let Some(base) = key.strip_suffix(".i8s") {
            let Some(__b) = self.linear_bytes_for(base, key)? else { return Ok(None) };
            return Ok(Some(Resolved::Bytes(__b, Dtype::F32)));
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
            let Some(base) = key.rsplit_once('.').map(|(b, _)| b) else {
                return Ok(None);
            };
            let Some(__b) = self.linear_bytes_for(base, key)? else { return Ok(None) };
            return Ok(Some(Resolved::Bytes(__b, dt)));
        }
        // E5-M2b fc 列半拆:mtp.fc_{e,h}.weight = mtp.fc.weight 的 in 维
        // 前/后半(W [out, 2h] 行主序 → 行内去交织,装载一次性 rayon;
        // bf16→f16 归一)。MTP 头双子 Linear 消费,消每轮 narrow 物化。
        if key == "mtp.fc_e.weight" || key == "mtp.fc_h.weight" {
            let Some(__b) = self.fc_half_bytes(key.ends_with("_h.weight"))? else {
                return Ok(None);
            };
            return Ok(Some(Resolved::Bytes(__b, Dtype::F16)));
        }
        // passthrough 优先(norm/conv/embed 等原生存量键;含非量化线性
        // 的裸 .weight —— cyankiwi 的 ignore 逐层不同,同键可能两态)
        if let Some(e) = self.index.get(key) {
            return Ok(Some(Resolved::View(e.clone())));
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
                    return Ok(Some(Resolved::Merged(vec![e1.clone(), e2.clone()])));
                }
            }
            if let Some(pe) = self.index.get(&format!("{base}.weight_packed")) {
                return Ok(Some(Resolved::View(pe.clone())));
            }
            if self.index.contains_key(&format!("{base}.weight")) {
                let Some(__b) = self.linear_bytes_for(base, &format!("{base}.packed_raw_ct"))? else {
                    return Ok(None);
                };
                return Ok(Some(Resolved::Bytes(__b, Dtype::U32)));
            }
        }
        // 非门控线性反量化臂(Linear 与本源同谓词,该键只对 non-eligible 出现)
        if let Some(base) = key.strip_suffix(".weight") {
            if self.index.contains_key(&format!("{base}.weight_packed")) {
                let Some(__b) = self.linear_bytes_for(base, key)? else { return Ok(None) };
            return Ok(Some(Resolved::Bytes(__b, Dtype::F16)));
            }
        }
        Ok(None)
    }

    /// 派生/虚拟键(不指向单一 mmap 条目)名判定 —— [`Self::resolve`]
    /// 专项分支表的**谓词镜像**(单一事实源:`resolve` 里的分支与本节
    /// 同名列表;两处同改)。
    ///
    /// 用途:[`WeightSource::native_dtype`] 的廉价筛查(声扫热路,禁物化)。
    /// 反面(不是派生键)也可能解析成 `Bytes`(如 `x.weight` 只有
    /// `weight_packed` 的 F16 兜底层)—— 那由 `index.get(key)` 缺失逼住。
    pub(crate) fn is_derived_key(&self, key: &str) -> bool {
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
            // P1.5-a:int8 per-row 头两键(名判定镜像;与 resolve 同改)
            || key.ends_with(".i8q")
            || key.ends_with(".i8s")
            || key == "mtp.fc_e.weight"
            || key == "mtp.fc_h.weight"
    }

    /// 键全量字节视图(懒物化 / passthrough 归一;take/take_range 用)
    fn key_bytes(&self, key: &str) -> Taken<(KeyData, Dtype)> {
        let Some(resolved) = self.resolve(key)? else {
            return Ok(None);
        };
        Ok(Some(match resolved {
            Resolved::Zeroed => {
                let Some(n) = self.elem_len(key)? else {
                    return Ok(None);
                };
                (KeyData::Owned(vec![0; n * 4]), Dtype::U32)
            }
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
                    _ => return Ok(None), // weight_shape 等元数据键不可取
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
        }))
    }
}

pub(crate) enum KeyData {
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
    fn take(&self, key: &str) -> Taken<Vec<f32>> {
        let Some(n) = self.elem_len(key)? else {
            return Ok(None);
        };
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
    fn native_dtype(&self, key: &str) -> Taken<Dtype> {
        if self.is_derived_key(key) {
            return Ok(None);
        }
        let Some(e) = self.index.get(key) else {
            return Ok(None);
        };
        Ok(match e.dtype {
            safetensors::Dtype::F32 => Some(Dtype::F32),
            safetensors::Dtype::BF16 => Some(Dtype::BF16),
            safetensors::Dtype::F16 => Some(Dtype::F16),
            _ => None,
        })
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
    ) -> Taken<()> {
        let Some(resolved) = self.resolve(key)? else {
            return Ok(None);
        };
        let Resolved::View(e) = resolved else {
            return Ok(None);
        };
        let esz = e.esz();
        let (Some(soff), Some(want)) =
            (offset_elems.checked_mul(esz), len.checked_mul(esz))
        else {
            return Ok(None);
        };
        let Some(s) = e.start.checked_add(soff) else {
            return Ok(None);
        };
        let (Some(end), false) = (s.checked_add(want), dst.len() < want) else {
            return Ok(None);
        };
        if end > e.start + e.nbytes {
            return Ok(None);
        }
        copy_par(&self.maps[e.map_ix][s..s + want], dst);
        self.maps[e.map_ix].dontneed(s, want);
        Ok(Some(()))
    }

    fn elem_len(&self, key: &str) -> Taken<usize> {
        // 元数据键(I64 [out,in])owl 不消费,拒绝可取化
        if key.ends_with(".weight_shape") {
            return Ok(None);
        }
        if let Some(base) = key.strip_suffix(".marlin_ws") {
            let Some((out, k, _)) = self.linear_form(base) else { return Ok(None) };
            if !marlin_eligible(out, k) {
                return Ok(None);
            }
            return Ok(Some(v2_workspace_len(out)));
        }
        if let Some(base) = key.strip_suffix(".marlin_ctmp") {
            let Some((out, k, _)) = self.linear_form(base) else { return Ok(None) };
            if !marlin_eligible(out, k) {
                return Ok(None);
            }
            return Ok(Some(1));
        }
        if let Some(base) = key.strip_suffix(".qweight") {
            let Some((out, k, _)) = self.linear_form(base) else { return Ok(None) };
            if !marlin_eligible(out, k) {
                return Ok(None);
            }
            return Ok(Some(out * k / 8));
        }
        if let Some(base) = key.strip_suffix(".scales") {
            let Some((out, k, _)) = self.linear_form(base) else { return Ok(None) };
            if !marlin_eligible(out, k) {
                return Ok(None);
            }
            return Ok(Some(out * (k / 32)));
        }
        if let Some(base) = key.strip_suffix(".packed_raw") {
            // 设备重排键:原始 packed 元素数(out × k/8;恒供,与 eligible 无关)
            let Some((out, k, _)) = self.linear_form(base) else { return Ok(None) };
            if !self.index.contains_key(&format!("{base}.weight_packed")) {
                return Ok(None); // 仅 ct 形态有原始 packed(F16 兜底层无)
            }
            return Ok(Some(out * (k / 8)));
        }
        if let Some(base) = key.strip_suffix(".zeros") {
            let Some((out, k, _)) = self.linear_form(base) else { return Ok(None) };
            if !marlin_eligible(out, k) {
                return Ok(None);
            }
            return Ok(Some((k / 32) * (out / 8)));
        }
        if let Some(base) = key.strip_suffix(".weight") {
            if !self.index.contains_key(key) {
                let Some((out, k, _)) = self.linear_form(base) else { return Ok(None) };
                if marlin_eligible(out, k) {
                    return Ok(None);
                }
                return Ok(Some(out * k));
            }
        }
        Ok(self.index.get(key).map(|e| {
            let elems: usize = e.shape.iter().product();
            elems
        }))
    }

    fn take_range(&self, key: &str, offset_elems: usize, len: usize) -> Taken<Vec<f32>> {
        let Some((data, dtype)) = self.key_bytes(key)? else {
            return Ok(None);
        };
        let bytes = data.as_slice();
        let esz = dtype.size_bytes();
        let (Some(s), Some(lb)) = (offset_elems.checked_mul(esz), len.checked_mul(esz)) else {
            return Ok(None);
        };
        let Some(e) = s.checked_add(lb) else {
            return Ok(None);
        };
        let Some(win) = bytes.get(s..e) else {
            return Ok(None);
        };
        Ok(Some(match dtype {
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
        }))
    }

    /// 分块转换直写字节租约(主路径):懒物化/passthrough/零填充三臂。
    fn convert_chunk_into_bytes(
        &self,
        key: &str,
        offset_elems: usize,
        len: usize,
        dst: &mut [u8],
        dtype: Dtype,
    ) -> Taken<()> {
        let esz = dtype.size_bytes();
        let Some(need) = len.checked_mul(esz) else {
            return Ok(None);
        };
        // 目标缓冲不足 = **调用方违约** ⇒ 响亮拒(旧版返回 None ⇒ 被上层收成"缺键")
        if need > dst.len() {
            return Err(ModelError::Msg(format!(
                "awq::convert_chunk_into_bytes: dst {}B < 需 {need}B({dtype:?}, len {len})",
                dst.len()
            )));
        }
        let Some(resolved) = self.resolve(key)? else {
            return Ok(None);
        };
        match resolved {
            Resolved::Zeroed => {
                dst[..need].fill(0);
                Ok(Some(()))
            }
            Resolved::View(e) => {
                // 设备重排原始键:I32 packed → U32 want(4B 同宽原样直拷)
                if e.dtype == safetensors::Dtype::I32 && dtype == Dtype::U32 {
                    let s = e.start + offset_elems * 4;
                    copy_par(&self.maps[e.map_ix][s..s + need], dst);
                    self.maps[e.map_ix].dontneed(s, need);
                    return Ok(Some(()));
                }
                if dtype != Dtype::F16 {
                    return Ok(None);
                }
                let src_esz = match e.dtype {
                    safetensors::Dtype::BF16 | safetensors::Dtype::F16 => 2,
                    safetensors::Dtype::F32 => 4,
                    _ => return Ok(None),
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
                    _ => return Ok(None),
                }
                self.maps[e.map_ix].dontneed(s, len * src_esz);
                Ok(Some(()))
            }
            // 合并虚拟键(两子基按序拼接):按 dst 偏移**直写**,零中间缓冲;
            // 支持任意 [offset_elems, offset_elems+len) 区间跨段(与大键分块同律)
            Resolved::Merged(parts) => {
                if dtype != Dtype::U32 {
                    return Ok(None);
                }
                let Some(s) = offset_elems.checked_mul(4) else {
                    return Ok(None);
                };
                let Some(e) = s.checked_add(need) else {
                    return Ok(None);
                };
                let mut pos = 0usize; // 逻辑拼接内的游标
                let mut w = 0usize; // dst 写入位
                for p in &parts {
                    let Some(p_end) = pos.checked_add(p.nbytes) else {
                        return Ok(None);
                    };
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
                    return Ok(None);
                }
                Ok(Some(()))
            }
            Resolved::Bytes(b, reg) => {
                if reg != dtype {
                    return Ok(None);
                }
                let s = offset_elems * esz;
                let Some(win) = b.get(s..s + need) else {
                    return Ok(None);
                };
                copy_par(win, dst);
                Ok(Some(()))
            }
        }
    }
}

/// **a8 键族谓词**(S2):W4A8 的四件产物 —— 与 a16 键族并列(层侧按 plan 二选一)
pub(crate) fn is_a8_key(key: &str) -> bool {
    key.ends_with(".qweight_a8")
        || key.ends_with(".scales_a8")
        || key.ends_with(".zeros_a8")
        || key.ends_with(".a8_factor")
}

#[cfg(test)]
mod tests {
    use super::*;
    // 搬去 awq_build 的打包器 + util 助手(测试面专用;放这里 ⇒ 生产构建零 unused)
    use crate::formats::awq_build::pack_scales_bf16_into;
    use owl_kernels::family::marlin::repack::{marlin_scales_index, pack_marlin_s_idx, unpack_zp_ct};
    use crate::formats::util::{bf16_par, i32s_le_bytes, u16s_le_bytes};
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
            matches!(src.resolve(key).expect("源"), Some(Resolved::Merged(_))),
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
                src.convert_chunk_into_bytes(key, off, len, &mut got, Dtype::U32).expect("源").is_some(),
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
        // 三态分明(2026-10-17 审查 §③):
        //  - **越界** = 正常语义 `Ok(None)`(调用方收割为缺键/形状错);
        //  - **dst 太小** = 调用方违约 ⇒ `Err`(旧版返回 None,被上层收成"缺键"假归因);
        //  - **宽度不符**(F16 目标要 4B/元素?见下)= `Ok(None)`(目标 dtype 不支持)。
        let mut small = vec![0u8; 4];
        // 越界用例给足 dst —— 否则先撞"dst 尺寸门"(两道门都响亮,顺序上先尺寸)
        let mut dst8 = vec![0u8; 8];
        assert!(src.convert_chunk_into_bytes(key, n - 1, 2, &mut dst8, Dtype::U32).expect("源").is_none(),
            "越界 = Ok(None)");
        assert!(src.convert_chunk_into_bytes(key, 0, n, &mut small, Dtype::U32).is_err(),
            "dst 太小 = 响亮 Err");
        assert!(src.convert_chunk_into_bytes(key, 0, 1, &mut small, Dtype::F16).expect("源").is_none(),
            "目标 dtype 不支持 = Ok(None)");
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
            let (d, dt) = src.key_bytes(key).expect("源").unwrap_or_else(|| panic!("{key} 取不到"));
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
        assert!(src.elem_len("l.qweight").expect("源").is_none(), "非 eligible 不得供 qweight");
        assert_eq!(src.elem_len("l.weight").expect("源"), Some(out * k), "反量化臂尺寸");
        let w = src.take("l.weight").expect("源").expect("take weight");
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
        assert_eq!(src.elem_len("l.qweight").expect("源"), Some(out * k / 8));
        assert_eq!(src.elem_len("l.scales").expect("源"), Some(out * groups));
        assert_eq!(src.elem_len("l.zeros").expect("源"), Some(groups * (out / 8)));
        assert_eq!(
            src.elem_len("l.marlin_ws").expect("源"),
            Some(owl_kernels::family::marlin::v2_workspace_len(out))
        );
        assert_eq!(src.elem_len("l.marlin_ctmp").expect("源"), Some(1));
        // zeros 键可取(U32 位型;字节级正确性由 GPU 对拍核内验证)
        let z = src.take("l.zeros").expect("源").expect("take zeros");
        assert_eq!(z.len(), groups * (out / 8));
        // take_range 分块一致性(w4a16 同律)
        let whole = src.take("l.scales").expect("源").expect("take scales");
        let half1 = src.take_range("l.scales", 0, whole.len() / 2).expect("源").expect("r1");
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
            owl_shared::testkit::report::skip("OWL_AWQ_IOBENCH 未开");
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
            owl_shared::testkit::report::skip("OWL_AWQ_IOBENCH 未开");
            return;
        }
        let dir = std::path::PathBuf::from(owl_shared::testkit::resource("ckpt:27b-awq").unwrap_or_else(|| {
            "/home/div/Documents/codes/models/cyankiwi/Qwen3.8-27B-AWQ-INT4".into()
        }));
        let src = AwqSource::open_dir(&dir).expect("源");
        let want = owl_shared::env_reader::str("OWL_AWQ_IOBENCH_KEYS").unwrap_or_else(|| {
            "model.language_model.layers.0.linear_attn.in_proj_qkvz.packed_raw,model.language_model.layers.0.mlp.down_proj.packed_raw,model.language_model.embed_tokens.weight".into()
        });
        for key in want.split(',').map(str::trim).filter(|k| !k.is_empty()) {
            let first = src.resolve(key).expect("源");
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
            let arm = match src.resolve(key).expect("源") {
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
                std::hint::black_box(src.resolve(key).expect("源"));
            });
            // ② 现行 API(装载域唯一入口)
            bench_seg("② convert_chunk_into_bytes", nbytes, 3, |_| {
                src.convert_chunk_into_bytes(key, 0, elems, &mut dst, dtype).expect("源").expect("convert");
            });
            // ③ 裸拷:mmap 视图(仅 View 臂可比)
            if let Some(Resolved::View(e)) = src.resolve(key).expect("源") {
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

    /// **锁中毒门(§④ 档①)**:物化缓存是**可重算**派生品 ⇒ 中毒后必须
    /// 就地取回 + 清空 + 告警,**继续可用**(重建那一键),而不是 PoisonError panic。
    #[test]
    fn poisoned_cache_recovers_and_clears() {
        let (out, k) = (256usize, 256usize);
        let dir = tmp_path("awq_poison_cache");
        let packed: Vec<i32> = (0..out * (k / 8)).map(|i| (i as i32) ^ 0x5a5a).collect();
        let scales = f32_to_bf16_bytes(&vec![0.125f32; out * (k / 32)]);
        let zp = vec![8i32; (out / 8) * (k / 32)];
        write_ct_dir(&dir, "l", &packed, &scales, &zp, out, k);
        let src = AwqSource::open_dir(&dir).expect("open");
        // 先正常取一次(填缓存)
        let first = src.key_bytes("l.qweight").expect("源").expect("qweight");
        // 毒化:持锁 panic
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _g = src.cache.lock().unwrap();
            panic!("poison");
        }));
        assert!(r.is_err(), "本测试就是要毒化锁");
        // 取回 + 清空 ⇒ 重建后仍逐字节相同(可重算性的判据)
        let again = src.key_bytes("l.qweight").expect("源").expect("qweight");
        assert_eq!(first.0.as_slice(), again.0.as_slice(), "清空重算后字节一致");
    }
}

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
use crate::formats::mmap::{open_raw_index, Mmap, RawEntry};
use crate::formats::w4a16::{bf16_par, i32s_le_bytes, i32s_par, u16s_le_bytes};
use crate::module::WeightSource;
use owl_kernels::marlin::repack::{
    marlin_fused_indices, pack_marlin_b_fused, pack_marlin_b_gather_into, pack_marlin_s,
    pack_marlin_z, unpack_zp_ct,
};
use owl_kernels::marlin::v2_workspace_len;
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

/// 键字节解析结果(懒物化后的一致字节 + 登记 dtype)
enum Resolved {
    /// 零填充直写(ws/ctmp;不占存储)
    Zeroed,
    /// mmap 视图直转/直拷(passthrough;f16 基线同款)
    View(RawEntry),
    /// 已物化字节(qweight U32 / scales F16 / zeros U32 / 反量化 weight F16)
    Bytes(Arc<[u8]>, Dtype),
}

pub struct AwqSource {
    maps: Vec<Mmap>,
    index: HashMap<String, RawEntry>,
    /// 懒物化字节缓存(键 → 字节;容量清空驱逐)
    cache: Mutex<(HashMap<String, Arc<[u8]>>, usize)>,
    /// gather/fused 索引表(按 (k, out, fused?) 形状复用;除法分解只算一次)
    idx_cache: Mutex<HashMap<(usize, usize, bool), Arc<Vec<u32>>>>,
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
        eprintln!(
            "[awq] mmap 索引 {} 项 @ {:?}(懒物化,无缓存)",
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

    /// 量化线性物化(键首触才建;qweight/scales/zeros/weight 四键共享
    /// 一次构建)。双检缓存 + build_lock 串行(w4a16 同款)。
    fn linear_bytes_for(&self, base: &str, key: &str) -> Option<Arc<[u8]>> {
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
            let s_pack: Vec<u16>;
            let z_buf: Vec<i32>;
            match form {
                QuantForm::Ct { se, ze, .. } => {
                    form_ct = true;
                    let scale_bytes = &self.maps[se.map_ix][se.start..se.start + se.nbytes];
                    let scales_f32 = bf16_par(scale_bytes);
                    let zp_bytes = &self.maps[ze.map_ix][ze.start..ze.start + ze.nbytes];
                    let zp_i32 = i32s_par(zp_bytes);
                    self.maps[se.map_ix].dontneed(se.start, se.nbytes);
                    self.maps[ze.map_ix].dontneed(ze.start, ze.nbytes);
                    s_pack = pack_marlin_s(&scales_f32, out, groups);
                    let zp_u8 = unpack_zp_ct(&zp_i32, out, groups);
                    z_buf = pack_marlin_z(&zp_u8, out, groups);
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
                    s_pack = pack_marlin_s(&scales_f32, out, groups);
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
            inserts.push((
                format!("{base}.scales"),
                Arc::from(u16s_le_bytes(&s_pack).into_boxed_slice()),
                s_pack.len() * 2,
            ));
            inserts.push((
                format!("{base}.zeros"),
                Arc::from(i32s_le_bytes(&z_buf).into_boxed_slice()),
                z_buf.len() * 4,
            ));
        } else if crate::formats::w4a16::marlin_eligible(out, k) {
            // ---- 全路径(Host 臂;裁决唯一产地 = module::RepackPath::resolve)----
            // B 打包与 U4B8 同套(q 原值,无 −8;dequant 在 kU4 内核内做)
            let s_pack: Vec<u16>;
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
                    let scales_f32 = bf16_par(scale_bytes);
                    let zp_bytes = &self.maps[ze.map_ix][ze.start..ze.start + ze.nbytes];
                    let zp_i32 = i32s_par(zp_bytes);
                    // 源页消费完毕即还(DONTNEED)
                    self.maps[pe.map_ix].dontneed(pe.start, pe.nbytes);
                    self.maps[se.map_ix].dontneed(se.start, se.nbytes);
                    self.maps[ze.map_ix].dontneed(ze.start, ze.nbytes);
                    let t_step = std::time::Instant::now();
                    s_pack = pack_marlin_s(&scales_f32, out, groups);
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
                    s_pack = pack_marlin_s(&scales_f32, out, groups);
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
                    .entry((k, out, true))
                    .or_insert_with(|| Arc::new(marlin_fused_indices(k, out)))
                    .clone();
                pack_marlin_b_fused(&packed, &idx, words, &mut b_buf);
            } else {
                let idx = self
                    .idx_cache
                    .lock()
                    .unwrap()
                    .entry((k, out, false))
                    .or_insert_with(|| {
                        Arc::new(owl_kernels::marlin::repack::marlin_gather_indices(k, out))
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
            inserts.push((
                format!("{base}.scales"),
                Arc::from(u16s_le_bytes(&s_pack).into_boxed_slice()),
                s_pack.len() * 2,
            ));
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
        eprintln!("[awq] 物化 {base} (out={out}, k={k}) @ {:?}", t0.elapsed());
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
        if key.ends_with(".marlin_ctmp") {
            let base = &key[..key.len() - ".marlin_ctmp".len()];
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
                    dst[..need].copy_from_slice(&self.maps[e.map_ix][s..s + need]);
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
                    safetensors::Dtype::F16 => dst[..need].copy_from_slice(win),
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
        std::fs::create_dir_all(dir).expect("mkdir");
        let mut header = String::from("{");
        let mut blobs: Vec<Vec<u8>> = Vec::new();
        let mut offset = 0usize;
        let mut entry = |header: &mut String, name: &str, dtype: &str, shape: &[usize], bytes: Vec<u8>, offset: &mut usize| {
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
        std::fs::write(dir.join("model.safetensors"), file).expect("写 ct 检查点");
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
            Some(owl_kernels::marlin::v2_workspace_len(out))
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
}

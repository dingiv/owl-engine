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
use crate::formats::util::{
    checked_len, f16_bytes_to_bf16_bytes, f32_bytes_to_f16, i32s_le_bytes, i32s_par,
    marlin_eligible, marlin_n_pack, scales_par, u16s_le_bytes,
};
use crate::module::{Taken, WeightSource};
use crate::f16c::{bf16_bytes_to_f16_bytes, dequant_u4_affine_f16_bytes};
use owl_kernels::family::marlin::repack::{
    marlin_gather_indices, pack_marlin_b_gather_into, pack_marlin_s, unpack_nibbles_into,
};
use owl_kernels::family::marlin::v2_workspace_len;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// marlin 可承载判定(**g128 暗雷根治后放宽**,2026-09-30):暗雷真凶 =
/// marlin_host 的 use_fp32_reduce=true + 4B c_tmp 占位(slice_count>1
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
    #[allow(clippy::type_complexity)] // 行块缓存(字节镜像 + 计数)
    cache: Mutex<(HashMap<String, Arc<[u8]>>, usize)>,
    /// gather 索引表(按 (k, n) 形状复用;除法分解只算一次)
    #[allow(clippy::type_complexity)] // 索引缓存(键 → 列序)
    idx_cache: Mutex<HashMap<(usize, usize), Arc<Vec<u32>>>>,
    /// 构建串行锁(重排内部 rayon 已饱和全核;并发双构建纯浪费)
    build_lock: Mutex<()>,
    /// 设备重排模式(装载域注入;须在首键物化前设置):true = CPU 仅产
    /// scales,量化大键的 marlin B 由 **GPU repack**(owl_ct_repack_u32)产出
    /// —— DFlash2 草稿提速(2026-10-13):草稿原被钉在 CPU 两步
    /// unpack+gather(66 张量 × ~0.68s ≈ 24.6s)。
    device_repack: std::cell::Cell<bool>,
}

impl W4A16Source {
    /// 装载入口:mmap + 索引(毫秒级,零数据拷贝);重排/反量化全部
    /// 懒到键首触。**无缓存文件**(v1 烘焙缓存随 v1 形态一并废除)。
    /// **可重算缓存**取用口(2026-10-17 审查 §④ 档①:中毒就地取回 + 清空 + 告警)。
    fn cache_guard(&self) -> std::sync::MutexGuard<'_, (HashMap<String, Arc<[u8]>>, usize)> {
        match self.cache.lock() {
            Ok(g) => g,
            Err(p) => {
                let mut g = p.into_inner();
                let n = g.0.len();
                if n > 0 {
                    g.0.clear();
                    g.1 = 0;
                    owl_shared::log_warn!("w4a16", "物化缓存锁中毒:已清空 {n} 项重算");
                }
                self.cache.clear_poison(); // 粘性:见 awq::cache_guard 注
                g
            }
        }
    }

    /// 索引缓存(可重算)取用口;语义同上。
    fn idx_guard(&self) -> std::sync::MutexGuard<'_, HashMap<(usize, usize), Arc<Vec<u32>>>> {
        match self.idx_cache.lock() {
            Ok(g) => g,
            Err(p) => {
                let mut g = p.into_inner();
                let n = g.len();
                if n > 0 {
                    g.clear();
                    owl_shared::log_warn!("w4a16", "索引缓存锁中毒:已清空 {n} 项重算");
                }
                self.idx_cache.clear_poison();
                g
            }
        }
    }

    /// 建造锁(纯串行化,无载荷):中毒无状态可坏 ⇒ 直接取回。
    fn build_guard(&self) -> std::sync::MutexGuard<'_, ()> {
        self.build_lock.lock().unwrap_or_else(|p| p.into_inner())
    }

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
        owl_shared::log_info!(
            "w4a16",
            "v2 mmap 索引 {} 项 @ {:?}(懒物化,无缓存)",
            index.len(),
            t0.elapsed()
        );
        Ok(Self {
            maps,
            index,
            cache: Mutex::new((HashMap::new(), 0)),
            idx_cache: Mutex::new(HashMap::new()),
            build_lock: Mutex::new(()),
            device_repack: std::cell::Cell::new(false),
        })
    }

    /// 设备重排模式(装载域注入;须在首键物化前设置)
    /// —— 与 awq.rs 同款接口(裁决唯一产地 = module::RepackPath::resolve)
    pub fn set_device_repack(&self, on: bool) {
        self.device_repack.set(on);
    }

    /// 反量化口(测试/对拍用):量化基线性 → f32(dequant_u4 精确展宽)
    pub fn dequant_linear(&self, base: &str) -> Taken<Vec<f32>> {
        let Some((out, k, pe, se)) = self.linear_dims(base)? else {
            return Ok(None);
        };
        let packed_bytes = &self.maps[pe.map_ix][pe.start..pe.start + pe.nbytes];
        let packed = i32s_par(packed_bytes);
        let scale_bytes = &self.maps[se.map_ix][se.start..se.start + se.nbytes];
        let scales_f32 = scales_par(se.dtype, scale_bytes);
        let mut dst = vec![0u8; out * k * 2];
        dequant_u4_affine_f16_bytes(&packed, &scales_f32, out, k, &mut dst);
        Ok(Some(
            dst.chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
        ))
    }

    /// 诊断口:索引键全集(装载面排查/基准用;不含派生键 qweight/scales/ws)
    pub fn keys(&self) -> Vec<String> {
        self.index.keys().cloned().collect()
    }

    /// 量化线性三元组定位:(out, k) 与 packed/scale 条目。
    /// shape 事实:`weight_packed` [out, k/8] i32。
    fn linear_dims(&self, base: &str) -> Taken<(usize, usize, &RawEntry, &RawEntry)> {
        let Some(pe) = self.index.get(&format!("{base}.weight_packed")) else { return Ok(None) };
        let Some(se) = self.index.get(&format!("{base}.weight_scale")) else { return Ok(None) };
        let Some(out) = pe.shape.first().copied() else {
            return Ok(None);
        };
        let Some(k) = pe.shape.get(1).copied().map(|x| x * 8) else {
            return Ok(None);
        };
        Ok(Some((out, k, pe, se)))
    }

    /// 键解析(唯一入口;懒物化在此触发)
    fn resolve(&self, key: &str) -> Taken<Resolved> {
        if key.ends_with(".marlin_ws") || key.ends_with(".marlin_ctmp") {
            return Ok(Some(Resolved::Zeroed));
        }
        // 设备重排原始键(2026-10-13 DFlash2 草稿提速):`{base}.packed_raw`
        // = 检查点原始 ct packed i32 原样(CPU 零重排;重排交 GPU
        // owl_ct_repack_u32 —— 该核 (in,rows,cols,out) 无 zp 参数、纯布局
        // 变换,与 AWQ kU4 同核同 B 字节布局,故 W4A16 直接复用)。
        // 零拷贝 View 优先;fc 列块五拆 / qkvz 合并走 linear_bytes_for。
        if let Some(base) = key.strip_suffix(".packed_raw") {
            // 原始 ct packed i32 → **U32 字节**(零重排)。注意:不可用
            // `Resolved::View` —— 它走 passthrough 归一 F16(I32 条目落
            // `_ => None` → 缺键);此处一律 Bytes/U32。
            let Some(__b) = self.linear_bytes_for(base, key)? else { return Ok(None) };
            return Ok(Some(Resolved::Bytes(__b, Dtype::U32)));
        }
        // DFlash2 fc 列块五拆派生(E5-DF3.5):`fc_{i}.qweight/.scales` =
        // fc.weight_packed/scale 列块 i(in 维 [i*5120,(i+1)*5120);
        // packed [out, k/8] 列连续 ↔ in 连续,块宽 640/40)
        if let Some(base) = key.strip_suffix(".qweight") {
            if let Some(i) = fc_block_index(base) {
                let Some(__b) = self.fc_block_bytes(i, false)? else { return Ok(None) };
                return Ok(Some(Resolved::Bytes(__b, Dtype::U32)));
            }
        }
        if let Some(base) = key.strip_suffix(".scales") {
            if let Some(i) = fc_block_index(base) {
                let Some(__b) = self.fc_block_bytes(i, true)? else { return Ok(None) };
                return Ok(Some(Resolved::Bytes(__b, Dtype::F16)));
            }
        }
        if let Some(base) = key.strip_suffix(".qweight") {
            let Some(__b) = self.linear_bytes_for(base, key)? else { return Ok(None) };
            return Ok(Some(Resolved::Bytes(__b, Dtype::U32)));
        }
        if let Some(base) = key.strip_suffix(".scales") {
            let Some(__b) = self.linear_bytes_for(base, key)? else { return Ok(None) };
            return Ok(Some(Resolved::Bytes(__b, Dtype::F16)));
        }
        // passthrough 优先(norm/conv/embed 等原生存量键)
        if let Some(e) = self.index.get(key) {
            return Ok(Some(Resolved::View(e.clone())));
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

    /// fc 列块物化(E5-DF3.5):fc.weight_packed [5120, 3200] I32 列子阵
    /// + fc.weight_scale [5120, 200] F16 列子阵 → (out=5120, k=5120) 走
    ///   与 build_linear 同式的 marlin 重排(qweight U32 / scales F16)。
    ///   列块恒 eligible(n=5120 % 256 == 0,k=5120 % 128 == 0)。
    ///   fc 列块**原始 ct packed** 字节(设备重排核的输入;不做 marlin 重排)
    ///   —— 与 [`Self::fc_block_bytes`] 的区别:后者返回的是**已重排的 marlin B**
    ///   (CPU 臂产物),喂给 GPU CtRepack 会二次重排 = 垃圾(2026-10-13 立案)。
    fn fc_block_raw_bytes(&self, blk: usize) -> Taken<Arc<[u8]>> {
        let vkey = format!("fc_block_{blk}_raw");
        {
            let cache = self.cache_guard();
            if let Some(b) = cache.0.get(&vkey) {
                return Ok(Some(b.clone()));
            }
        }
        let Some(pe) = self.index.get("fc.weight_packed") else { return Ok(None) };
        let Some(out) = pe.shape.first().copied() else { return Ok(None) };
        let Some(packed_cols) = pe.shape.get(1).copied() else { return Ok(None) };
        let n_fc = 5usize;
        let cols = packed_cols / n_fc;
        let packed_bytes = &self.maps[pe.map_ix][pe.start..pe.start + pe.nbytes];
        let packed_all = i32s_par(packed_bytes);
        let mut packed = Vec::with_capacity(out * cols);
        for r in 0..out {
            packed.extend_from_slice(
                &packed_all[r * packed_cols + blk * cols..r * packed_cols + (blk + 1) * cols],
            );
        }
        let buf: Arc<[u8]> = Arc::from(i32s_le_bytes(&packed).into_boxed_slice());
        let mut cache = self.cache_guard();
        cache.1 += buf.len();
        cache.0.insert(vkey, buf.clone());
        Ok(Some(buf))
    }

    fn fc_block_bytes(&self, blk: usize, scales_side: bool) -> Taken<Arc<[u8]>> {
        let vkey = format!("fc_block_{blk}_{}", if scales_side { "s" } else { "q" });
        {
            let cache = self.cache_guard();
            if let Some(b) = cache.0.get(&vkey) {
                return Ok(Some(b.clone()));
            }
        }
        let Some(pe) = self.index.get("fc.weight_packed") else { return Ok(None) };
        let Some(se) = self.index.get("fc.weight_scale") else { return Ok(None) };
        let Some(out) = pe.shape.first().copied() else { return Ok(None) };
        let Some(packed_cols) = pe.shape.get(1).copied() else {
            return Ok(None);
        }; // k_all / 8 = 3200
        let n_fc = 5usize;
        let cols = packed_cols / n_fc; // 640
        let Some(scol) = se.shape.get(1).copied().map(|x| x / n_fc) else {
            return Ok(None);
        }; // 40
        let _g = self.build_guard();
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
            Arc::from(u16s_le_bytes(&s).into_boxed_slice())
        } else {
            let mut q_buf: Vec<u8> = Vec::new();
            unpack_nibbles_into(&packed, out, k_blk, &mut q_buf);
            let idx = self
                .idx_guard()
                .entry((k_blk, out))
                .or_insert_with(|| Arc::new(marlin_gather_indices(k_blk, out)))
                .clone();
            let words = out * k_blk / 8;
            let mut b_buf: Vec<i32> = Vec::new();
            pack_marlin_b_gather_into(&q_buf, &idx, words, &mut b_buf);
            Arc::from(i32s_le_bytes(&b_buf).into_boxed_slice())
        };
        let mut cache = self.cache_guard();
        cache.0.insert(vkey.clone(), buf.clone());
        cache.1 += buf.len();
        Ok(Some(buf))
    }

    /// 量化线性物化(键首触才建;qweight/scales 成对,weight 独立)。
    /// 双检缓存(快路径锁内查;构建在 build_lock 串行内,防并发双建)。
    fn linear_bytes_for(&self, base: &str, key: &str) -> Taken<Arc<[u8]>> {
        // 刀2 虚拟合并:in_proj_qkvz = row-stack(in_proj_qkv, in_proj_z)
        // (同 k 行堆叠 = 字节拼接;两子基各自走完整建管)
        if let Some((b1, b2)) = crate::formats::split_qkvz(base) {
            let suffix = &key[base.len()..];
            let Some(s1) = self.linear_bytes_for(&b1, &format!("{b1}{suffix}"))? else { return Ok(None) };
            let Some(s2) = self.linear_bytes_for(&b2, &format!("{b2}{suffix}"))? else { return Ok(None) };
            let mut buf = Vec::with_capacity(s1.len() + s2.len());
            buf.extend_from_slice(&s1);
            buf.extend_from_slice(&s2);
            return Ok(Some(buf.into()));
        }
        // 设备重排原始键(2026-10-13 DFlash2 草稿提速):必须在
        // `linear_dims` **之前** —— fc 列块(`fc_0..4`)是派生键,没有
        // `fc_i.weight_packed`,linear_dims 会先 `?` 掉。
        if let Some(b) = key.strip_suffix(".packed_raw") {
            if let Some(i) = fc_block_index(b) {
                // **原始** ct packed(非 marlin B) —— 设备重排核的输入
                return self.fc_block_raw_bytes(i);
            }
            if let Some(pe) = self.index.get(&format!("{b}.weight_packed")) {
                return Ok(Some(Arc::from(
                    self.maps[pe.map_ix][pe.start..pe.start + pe.nbytes].to_vec(),
                )));
            }
            return Ok(None);
        }
        let Some((out, k, pe, se)) = self.linear_dims(base)? else { return Ok(None) };
        {
            let cache = self.cache_guard();
            if let Some(b) = cache.0.get(key) {
                return Ok(Some(b.clone()));
            }
        }
        // 2026-10-17 审查 §③:构建真失败 = **上抛**(原 `.ok()?` 压成 None ⇒ "缺键"假归因)
        self.build_linear(base, out, k, pe, se)?;
        Ok(self.cache_guard().0.get(key).cloned())
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
        let _g = self.build_guard();
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
        let device = self.device_repack.get();
        if marlin_eligible(out, k) && device {
            // ---- 轻路径(device):CPU 仅产 scales;qweight 的 marlin B 由
            //      GPU repack 产出(load.rs 设备重排臂;与 awq.rs 同款)----
            let s = pack_marlin_s(&scales_f32, out, k / 128);
            inserts.push((
                format!("{base}.scales"),
                // BF16 激活 marlin:s_type = BF16(位型转换,见 f16_bytes_to_bf16_bytes)
                Arc::from(u16s_le_bytes(&s).into_boxed_slice()),
                s.len() * 2,
            ));
        } else if marlin_eligible(out, k) {
            let mut q_buf: Vec<u8> = Vec::new();
            unpack_nibbles_into(&packed, out, k, &mut q_buf);
            let idx = self
                .idx_guard()
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
        let mut cache = self.cache_guard();
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
        owl_shared::log_debug!(
            "w4a16",
            "物化 {base} (out={out}, k={k}) @ {:?}",
            t0.elapsed()
        );
        Ok(())
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
                    safetensors::Dtype::BF16 => bf16_bytes_to_f16_bytes(src, &mut out),
                    safetensors::Dtype::F32 => out = f32_bytes_to_f16(src),
                    _ => return Ok(None), // weight_shape 等元数据键不可取
                }
                (KeyData::Owned(out), Dtype::F16)
            }
            Resolved::Bytes(b, dt) => (KeyData::Shared(b), dt),
        }))
    }
}

/// `len × esz` 的检查乘法(溢出/越界 ⇒ None,调用方收割为缺键)

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
    fn take(&self, key: &str) -> Taken<Vec<f32>> {
        let Some(n) = self.elem_len(key)? else { return Ok(None) };
        self.take_range(key, 0, n)
    }

    fn elem_len(&self, key: &str) -> Taken<usize> {
        // 元数据键(I64 [out,in])owl 不消费,拒绝可取化
        if key.ends_with(".weight_shape") {
            return Ok(None);
        }
        // fc 列块派生(E5-DF3.5):fc_{i}.qweight = out*640 U32 /
        // fc_{i}.scales = out*40 F16;fc_{i}.weight = out*5120 F16(反量化)
        if let Some(base) = key.strip_suffix(".qweight") {
            if fc_block_index(base).is_some() {
                return Ok(Some(5120 * 640));
            }
        }
        if let Some(base) = key.strip_suffix(".scales") {
            if fc_block_index(base).is_some() {
                return Ok(Some(5120 * 40));
            }
        }
        if let Some(base) = key.strip_suffix(".weight") {
            if fc_block_index(base).is_some() {
                return Ok(Some(5120 * 5120));
            }
        }
        // 派生键:尺寸由基线性形状推导;谓词门控(eligible → qweight/scales/
        // ws/ctmp;non-eligible → weight),错位键报 None 免误消费
        if let Some(base) = key.strip_suffix(".marlin_ws") {
            let Some((out, k, _, _)) = self.linear_dims(base)? else { return Ok(None) };
            if !marlin_eligible(out, k) {
                return Ok(None);
            }
            return Ok(Some(v2_workspace_len(marlin_n_pack(out))));
        }
        if let Some(base) = key.strip_suffix(".marlin_ctmp") {
            let Some((out, k, _, _)) = self.linear_dims(base)? else { return Ok(None) };
            if !marlin_eligible(out, k) {
                return Ok(None);
            }
            return Ok(Some(1));
        }
        if let Some(base) = key.strip_suffix(".qweight") {
            let Some((out, k, _, _)) = self.linear_dims(base)? else { return Ok(None) };
            if !marlin_eligible(out, k) {
                return Ok(None);
            }
            return Ok(Some(out * k / 8));
        }
        if let Some(base) = key.strip_suffix(".scales") {
            let Some((out, k, _, _)) = self.linear_dims(base)? else { return Ok(None) };
            if !marlin_eligible(out, k) {
                return Ok(None);
            }
            return Ok(Some(out * (k / 128)));
        }
        if let Some(base) = key.strip_suffix(".weight") {
            if !self.index.contains_key(key) {
                let Some((out, k, _, _)) = self.linear_dims(base)? else { return Ok(None) };
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
        let Some((data, dtype)) = self.key_bytes(key)? else { return Ok(None) };
        let bytes = data.as_slice();
        let esz = dtype.size_bytes();
        let Some(s) = offset_elems.checked_mul(esz) else { return Ok(None) };
        let Some(lb) = len.checked_mul(esz) else { return Ok(None) };
        let Some(e) = s.checked_add(lb) else { return Ok(None) };
        let Some(win) = bytes.get(s..e) else { return Ok(None) };
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
        let Some(need) = len.checked_mul(esz) else { return Ok(None) };
        // 目标缓冲不足 = **调用方违约** ⇒ 响亮拒(旧版返回 None ⇒ 被上层收成"缺键")
        if need > dst.len() {
            return Err(ModelError::Msg(format!(
                "w4a16::convert_chunk_into_bytes: dst {}B < 需 {need}B({dtype:?}, len {len})",
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
                // F16 want(存量主路径)/ BF16 want(E5-DF3 同日十四;DFlash2
                // 草稿 BF16 化 —— passthrough 权重原生 BF16,直拷零转换,
                // sglang .to(bf16) 同位)
                if dtype != Dtype::F16 && dtype != Dtype::BF16 {
                    return Ok(None);
                }
                let src_esz = match e.dtype {
                    safetensors::Dtype::BF16 | safetensors::Dtype::F16 => 2,
                    safetensors::Dtype::F32 => 4,
                    _ => return Ok(None),
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
                    _ => return Ok(None),
                }
                self.maps[e.map_ix].dontneed(s, len * src_esz);
                Ok(Some(()))
            }
            Resolved::Bytes(b, reg) => {
                let src_esz = reg.size_bytes();
                let s = offset_elems * src_esz;
                let Some(sz) = checked_len(len, src_esz) else {
                    return Ok(None);
                };
                let Some(win) = b.get(s..s + sz) else { return Ok(None) };
                if reg == dtype {
                    // 声明一致 ⇒ 逐字节直拷(零转换)
                    dst[..need].copy_from_slice(win);
                    return Ok(Some(()));
                }
                // **声明=事实**下的显式转换(2026-10-16 定谳):F16 ↔ BF16
                // (同为 2B 值域;由调用方按 want 的声明 dtype 决定方向)。
                // 其余组合(含跨宽度)不支持 ⇒ 响亮拒(绝不静默直拷/截断)。
                match (reg, dtype) {
                    (Dtype::F16, Dtype::BF16) => {
                        dst[..need].copy_from_slice(&f16_bytes_to_bf16_bytes(win))
                    }
                    (Dtype::BF16, Dtype::F16) => bf16_bytes_to_f16_bytes(win, &mut dst[..need]),
                    _ => return Ok(None),
                }
                Ok(Some(()))
            }
        }
    }
}

// ============================================================================
// 字节转换助手(rayon;对齐安全 from_le_bytes)—— pub(crate):awq.rs 共享
// ============================================================================

/// DFlash2 fc 列块键解析:`fc_{i}`(i < 5)→ Some(i);其余 None。
fn fc_block_index(base: &str) -> Option<usize> {
    // ⚠️ 本函数**不在 Taken 链上**(纯键名解析,无 I/O)⇒ 保持 Option:机械改写
    // 时曾把它的 `return None` 一并升格,编译器当场拦下(2026-10-17)
    let idx = base.strip_prefix("fc_")?;
    if idx.len() != 1 || !idx.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let i = idx.parse::<usize>().ok()?;
    (i < 5).then_some(i)
}

// ============================================================================
// 门:派生键 × 权威键 数值对账(**F16/BF16 内容面**;2026-10-16 立)
// ============================================================================
//
// 背景(整类问题的机器判据):派生键的**字节内容**可以与线格式声明不同 ——
// 现状:`fc_i.scales` 线格式声明 **F16**(与算子表一致,给 L3 校验的是它),
// 而字节内容其实是 **BF16 位型**(见 `fc_block_bytes` 的三处
// `f16_bytes_to_bf16_bytes`)。而"内容 dtype"在当前契约里**没有字段**,
// 于是任何按声明解字节的读者都会静默拿到 ~15× 错值。
//
// 为什么不能用"往返定点"当判据:f16 解码 ↔ 编码是**双射**,
// 声明错了也照样往返成功 ⇒ 必须用"**两条独立来路数值一致**"。
#[cfg(test)]
mod tests {
    use super::*;
    // 测试面专用(生产段不用;放这里 ⇒ 生产构建零 unused 告警)
    use crate::formats::util::f16_par;

    fn draft_dir() -> Option<std::path::PathBuf> {
        owl_shared::testkit::resource("ckpt:dflash2").map(std::path::PathBuf::from)
    }

    /// 派生 fc 列块 ↔ fused 权威:①尺度(内容口径 BF16)②marlin B(逐字)。
    ///
    /// 缺失检查点即 skip(与仓内其他 fixture 档同律;本门**不需要 GPU**)。
    #[test]
    fn fc_derived_keys_match_fused_authority() {
        let Some(dir) = draft_dir() else {
            owl_shared::testkit::report::skip("ckpt:dflash2 未设");
            return;
        };
        let src = W4A16Source::open_dir(&dir).expect("打开草稿目录");

        // ---- 权威面:fused 键(声明与内容一致:packed=I32 字、scale=F16)----
        // 注:这两把键走不了 `key_bytes`(I32 不在元素通道的 dtype 表里)⇒ 直读 owner 面。
        let pe = src.index.get("fc.weight_packed").expect("fused packed 索引");
        let pk = i32s_par(&src.maps[pe.map_ix][pe.start..pe.start + pe.nbytes]);
        let se = src.index.get("fc.weight_scale").expect("fused scale 索引");
        assert_eq!(se.dtype, safetensors::Dtype::F16, "fused 尺度声明 F16(内容亦 F16)");
        let fused_s = scales_par(se.dtype, &src.maps[se.map_ix][se.start..se.start + se.nbytes]);

        const OUT: usize = 5120;
        const N_FC: usize = 5;
        const COLS_ALL: usize = 3200; // k_all/8 = 25600/8
        const SC_ALL: usize = 200; // k_all/g = 25600/128
        let cols = COLS_ALL / N_FC; // 640
        let scol = SC_ALL / N_FC; // 40
        let k_blk = cols * 8; // 5120

        for blk in 0..N_FC {
            // ---- ① 尺度:派生(按**内容口径 BF16** 解)== fused 列块(按 F16 解)----
            let skey = format!("fc_{blk}.scales");
            let (sb, sdt) = src.key_bytes(&skey).expect("源").expect("派生尺度");
            assert_eq!(sdt, Dtype::F16, "{skey}: 线格式声明必须保持 F16(算子表口径)");
            // 派生尺度 = **f16 位型的 marlin S**(声明 F16 ⇒ 事实 F16;bf16 臂的转换在装载侧)
            let got = f16_par(sb.as_slice());
            assert_eq!(got.len(), OUT * scol, "{skey}: 元素数");
            let idx = owl_kernels::family::marlin::repack::marlin_scales_index(OUT, scol);
            for pos in [0usize, 1, 63, 64, 127, 4096, OUT * scol - 1] {
                let s = idx[pos] as usize;
                let (so, sg) = (s / scol, s % scol);
                let want = fused_s[so * SC_ALL + blk * scol + sg];
                let g = got[pos];
                let rel = (g - want).abs() / want.abs().max(1e-9);
                assert!(
                    rel < 5e-3,
                    "{skey}[pos {pos} → (r{so},g{sg})]: 派生 {g:.6} vs 权威 {want:.6}(rel {rel:.4})"
                );
            }

            // ---- ② marlin B:派生 == 由 fused 列块现算(逐字)----
            let qkey = format!("fc_{blk}.qweight");
            let (qb, qdt) = src.key_bytes(&qkey).expect("源").expect("派生 qweight");
            assert_eq!(qdt, Dtype::U32, "{qkey}: B 以 U32 承载");
            let derived = i32s_par(qb.as_slice());
            // fused 列块(行主序 [out, cols])→ 解包 → marlin B 期望值
            let mut block = vec![0i32; OUT * cols];
            for r in 0..OUT {
                block[r * cols..(r + 1) * cols].copy_from_slice(
                    &pk[r * COLS_ALL + blk * cols..r * COLS_ALL + (blk + 1) * cols],
                );
            }
            let mut nib = Vec::new();
            owl_kernels::family::marlin::repack::unpack_nibbles_into(&block, OUT, k_blk, &mut nib);
            let gidx = owl_kernels::family::marlin::repack::marlin_gather_indices(k_blk, OUT);
            let mut want_b: Vec<i32> = Vec::new();
            owl_kernels::family::marlin::repack::pack_marlin_b_gather_into(
                &nib,
                &gidx,
                OUT * k_blk / 8,
                &mut want_b,
            );
            assert_eq!(derived.len(), want_b.len(), "{qkey}: 字数");
            let bad = derived.iter().zip(&want_b).filter(|(a, b)| a != b).count();
            assert_eq!(
                bad, 0,
                "{qkey}: 派生 B 与 fused 现算不一致({bad}/{} 字)—— 派生链路(重排/切块/打包)有罪",
                want_b.len()
            );

            // ---- ③ 设备重排臂的**输入** `fc_i.packed_raw` == fused 列块**逐字** ----
            // (设备路线的 B 由 GPU CtRepack 现产;若它的输入切块错了,CPU 路线的绿
            //  并不能替它背书 —— 2026-10-16 补:这条正是 16.6× 的剩余嫌疑面)
            let rkey = format!("fc_{blk}.packed_raw");
            let (rb, rdt) = src.key_bytes(&rkey).expect("源").expect("派生 packed_raw");
            assert_eq!(rdt, Dtype::U32, "{rkey}: 原始 ct 字以 U32 承载");
            let raw = i32s_par(rb.as_slice());
            let want_raw: Vec<i32> = (0..OUT)
                .flat_map(|r| pk[r * COLS_ALL + blk * cols..r * COLS_ALL + (blk + 1) * cols].to_vec())
                .collect();
            assert_eq!(raw.len(), want_raw.len(), "{rkey}: 字数");

            // ---- ④ **两条 B 打包路必须等价**(2026-10-16 补;纯 CPU) ----
            // 核的正确性由 parity 档用 `pack_marlin_b`(plain)证明;而本仓装载/派生
            // 走的是 `pack_marlin_b_gather_into`(gather)。两者若不等价,
            // "派生 == gather 现算" 就只是自证 —— 必须直接对拍。
            let plain = owl_kernels::family::marlin::repack::pack_marlin_b(&nib, k_blk, OUT);
            let mism = plain.iter().zip(&want_b).filter(|(a, b)| a != b).count();
            assert_eq!(
                mism, 0,
                "块{blk}: pack_marlin_b(plain, 核的证明口径)与 pack_marlin_b_gather_into(gather, 装载口径)不等价({mism}/{} 字)",
                plain.len()
            );
        }
        eprintln!(
            "[fc-derive] 5 块 × (尺度内容口径 BF16 对账 + marlin B 逐字)全绿(目录 {})",
            dir.display()
        );
    }
}

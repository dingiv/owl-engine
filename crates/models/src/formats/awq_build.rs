//! **AWQ 源:字节物化面**(2026-10-17 审查 §五轮-C3 从 `awq.rs` 拆出)。
//!
//! 分工(与 `awq.rs` 的界):
//! - `awq.rs` = **索引 / 形态解析 / 契约实现**(`open_dir`、`linear_form`、`resolve`、
//!   `key_bytes`、`impl WeightSource`,以及全部测试);
//! - 本文件 = **字节物化** —— 物化缓存/索引缓存/分片建造锁的取用口(`*_guard`)、
//!   `fc_half_bytes`、`linear_bytes_for`(物化唯一入口)、`build_linear`、
//!   `build_linear_a8`,以及各打包器(`pack_scales*` / `rtn_g32_sym_*` /
//!   `dequant_u4_awq_f16_bytes`)。
//!
//! 判据(为什么这么切):物化面全部围绕**三张缓存 + 懒构建**,与键面解析无关;
//! 拆后 `awq.rs` 回千行档,物化面自带单一入口。
//! `impl AwqSource` 第二块合法(类型可vis ⇒ 同 crate 可另起 impl;相应字段转 `pub(crate)`)。

use crate::formats::awq::{is_a8_key, AwqSource, IdxKind, QuantForm, CACHE_CAP_BYTES};
use crate::contract::ModelError;
use crate::formats::split_merged;
use crate::formats::util::marlin_eligible;
use crate::module::Taken;
use crate::formats::util::{bf16_par, i32s_le_bytes, i32s_par, u16s_le_bytes};
use owl_kernels::family::marlin::repack::{
    marlin_fused_indices, marlin_scales_index, marlin_scales_index_a8, pack_marlin_b_fused,
    pack_marlin_b_gather_into, pack_marlin_s_idx, pack_marlin_z, unpack_zp_ct,
};
use std::collections::HashMap;
use std::sync::Arc;

impl AwqSource {
    /// 装载入口:mmap + 索引(毫秒级,零数据拷贝);重排/反量化全部
    /// 懒到键首触。无缓存文件(与 w4a16 v2 同律)。
    /// **可重算缓存**的取用口(2026-10-17 审查 §④ 档①)。
    ///
    /// 锁中毒 ⇒ 就地取回 + **清空** + 告警:内容纯由源算出(丢了重算),
    /// 但半成品/半插入不许留 —— 比 §52 的"就地恢复"更强一档。
    /// 恢复必须**可见**(告警 + 该键重算的代价),不许静默。
    fn cache_guard(&self) -> std::sync::MutexGuard<'_, (HashMap<String, Arc<[u8]>>, usize)> {
        match self.cache.lock() {
            Ok(g) => g,
            Err(p) => {
                let mut g = p.into_inner();
                let n = g.0.len();
                if n > 0 {
                    g.0.clear();
                    g.1 = 0;
                    owl_shared::log_warn!(
                        "awq",
                        "物化缓存锁中毒:已清空 {n} 项重算(此前有线程持锁 panic)"
                    );
                }
                // ⚠️ **毒化是粘性的**:不清标志位 ⇒ 之后每次取用都走 Err 分支
                // (缓存被反复清空 ⇒ 永不命中)。2026-10-17 由新加的锁中毒门
                // 当场抓出:`poisoned_cache_recovers_and_clears` 第一次跑红。
                self.cache.clear_poison();
                g
            }
        }
    }

    /// 索引缓存(可重算)取用口;语义同 [`Self::cache_guard`]。
    fn idx_guard(
        &self,
    ) -> std::sync::MutexGuard<'_, HashMap<(usize, usize, IdxKind), Arc<Vec<u32>>>> {
        match self.idx_cache.lock() {
            Ok(g) => g,
            Err(p) => {
                let mut g = p.into_inner();
                let n = g.len();
                if n > 0 {
                    g.clear();
                    owl_shared::log_warn!("awq", "索引缓存锁中毒:已清空 {n} 项重算");
                }
                self.idx_cache.clear_poison(); // 粘性(同上)
                g
            }
        }
    }

    /// 分片建造锁(纯串行化,**无载荷**)取用口:中毒无状态可坏 ⇒ 直接取回。
    fn build_guard(&self, shard: usize) -> std::sync::MutexGuard<'_, ()> {
        self.build_locks[shard].lock().unwrap_or_else(|p| p.into_inner())
    }

    /// fc 列半拆物化(E5-M2b;键首触才建,双检缓存 + build_lock 同款):
    /// 读物理键 mtp.fc.weight(bf16 [out, 2h])→ 逐行 bf16→f16 + 取前/后
    /// 半 → [out, h] f16 连续。rayon 行并行(行长 2 字节对齐,安全)。
    pub(crate) fn fc_half_bytes(&self, high: bool) -> Taken<Arc<[u8]>> {
        let vkey = if high { "mtp.fc_h.weight" } else { "mtp.fc_e.weight" };
        {
            let cache = self.cache_guard();
            if let Some(b) = cache.0.get(vkey) {
                return Ok(Some(b.clone()));
            }
        }
        let Some(we) = self.index.get("mtp.fc.weight") else {
            return Ok(None);
        };
        let (Some(out), Some(din)) = (we.shape.first().copied(), we.shape.get(1).copied()) else {
            return Ok(None);
        };
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
            let mut cache = self.cache_guard();
            cache.0.insert(vkey.to_string(), b.clone());
        }
        Ok(Some(b))
    }

    /// scales 源索引表(按 (k, out) 形状缓存;见 [`Self::pack_scales`])
    fn scales_index(&self, out: usize, k: usize) -> Arc<Vec<u32>> {
        self.idx_guard()
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

    /// a8 版 scales 源索引表(perm = 32 元素 `scale_perm_single`;
    /// [`Self::scales_index`] 的 a8 兄弟,按 (k, out) 形状缓存)。
    fn scales_index_a8(&self, out: usize, k: usize) -> Arc<Vec<u32>> {
        self.idx_guard()
            .entry((k, out, IdxKind::ScalesA8))
            .or_insert_with(|| Arc::new(marlin_scales_index_a8(out, k / 32)))
            .clone()
    }

    /// **a8 scales 单遍融合**(装载热路径;2026-10-17):bf16 源 → 转置+32-perm
    /// (_a8 表)→ int16 归一(q = round(v/max·4096) clamp 正域)→ LE 2B 直写。
    ///
    /// 原链(a8 首版)= `bf16_par`(11MB)+ 标量双循环转置(214MB 级
    /// 非连续写)+ 逐 32-chunk `to_vec` perm + 独立 normalize 遍历:
    /// 27B 单键 ~65ms(150 键 ≈10s);本函数 = 一分配一路过(同 a16 的
    /// `pack_scales_bf16_into` 纪律;`bf16→f32` 零舍入展宽 ⇒ 与旧链**逐位同**)。
    fn pack_scales_a8_bytes(
        &self,
        parts: &[&[u8]],
        out: usize,
        k: usize,
        max: f32,
    ) -> Vec<u8> {
        let groups = k / 32;
        let mut dst = vec![0u8; out * groups * 2];
        if max == 0.0 {
            // 全零尺度(旧链同路:早退产全零 + 因子 0)
            return dst;
        }
        pack_scales_a8_bf16_into(parts, out, groups, &self.scales_index_a8(out, k), max, &mut dst);
        dst
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
    pub(crate) fn pack_scales_bytes(&self, parts: &[&[u8]], out: usize, k: usize) -> Vec<u8> {
        let groups = k / 32;
        let mut dst = vec![0u8; out * groups * 2];
        pack_scales_bf16_into(parts, out, groups, &self.scales_index(out, k), &mut dst);
        dst
    }

/// 量化线性物化(键首触才建;qweight/scales/zeros/weight 四键共享
    /// 一次构建)。双检缓存 + build_lock 串行(w4a16 同款)。
    pub(crate) fn linear_bytes_for(&self, base: &str, key: &str) -> Taken<Arc<[u8]>> {
        // ---- P1.5-a:int8 per-row 头(专用分支;**不经** `build_linear`)----
        //
        // 为什么独立:lm_head 走 `build_linear` 会连带把 int4 三件套(scales/zeros/
        // marlin B)也烘一遍 —— 那是纯浪费(636M nibble 重排),而且会往缓存塞
        // 三份用不到的字节。本分支只读 bf16 源、只产 i8q/i8s。
        if key.ends_with(".i8q") || key.ends_with(".i8s") {
            {
                let cache = self.cache_guard();
                if let Some(b) = cache.0.get(key) {
                    return Ok(Some(b.clone()));
                }
            }
            let Some((out, k, form)) = self.linear_form(base) else { return Ok(None) };
            let QuantForm::F16 { we } = form else {
                // 只有"未量化的 bf16 权重"才做 int8 头;Ct(packed)源不反解
                // (从 int4 反解再压 int8 = 双重损失,响亮拒)
                return Err(ModelError::Msg(format!(
                    "{base}: int8 头要求 bf16 源,**不得**从 packed int4 反解"
                )));
            };
            let wb = &self.maps[we.map_ix][we.start..we.start + we.nbytes];
            // BF16 头占整载 2.9s(2026-10-24 冷/热载实测):旧链先展开
            // 5.1GB f32,再逐元素除/round。BF16 直接两遍(行 amax + 量化)
            // —— 读源 2.54GB×2,不建 f32 中间态;语义与旧链逐位同源。
            let (q8, s8) = if we.dtype == safetensors::Dtype::BF16 {
                rtn_i8_row_sym_bf16(wb, out, k)
            } else {
                let w_f32: Vec<f32> = match we.dtype {
                    safetensors::Dtype::F16 => wb
                        .chunks_exact(2)
                        .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                        .collect(),
                    safetensors::Dtype::F32 => wb
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect(),
                    _ => return Err(ModelError::Msg(format!("awq: {base} i8 源 dtype 不支持"))),
                };
                rtn_i8_row_sym(&w_f32, out, k)
            };
            self.maps[we.map_ix].dontneed(we.start, we.nbytes);
            // **两键一次落缓存**:q/s 同源,分别按请求算会把整表(27B embed
            // 5G 级 f32)重算两遍。q 键 = `{base}.i8q`,s 键 = `{base}.i8s`。
            let qk = format!("{base}.i8q");
            let sk = format!("{base}.i8s");
            let mut cache = self.cache_guard();
            cache.0.insert(qk, Arc::from(q8.into_boxed_slice()));
            cache.0.insert(sk, Arc::from(s8.into_boxed_slice()));
            return Ok(cache.0.get(key).cloned());
        }

        // **a8 键族走独立构建**(S2):与 a16 同源不同布局,且**互不物化**
        // —— a8 路只产 a8 四件,不陪跑 a16 的重排(反之亦然)
        if is_a8_key(key) {
            // 设备重排模式下 B 由 GPU 核产出 ⇒ 这里**不做 CPU B 打包**
            // (先前无条件烘 B 会让设备档白跑一趟全模 CPU 重排)
            let want_b = key.ends_with(".qweight_a8");
            // 2026-10-17 审查 §③:物化真失败 = **上抛**(原先打一条 log_info
            // 再返回 None ⇒ 上层收成"数据源缺键",第二条静默路)
            self.build_linear_a8(base, want_b)?;
            let cache = self.cache_guard();
            return Ok(cache.0.get(key).cloned());
        }
        // 刀2 虚拟合并:in_proj_qkvz = row-stack(in_proj_qkv, in_proj_z)。
        // **源域合并后一次构建**(marlin s/z 是全 flat 64-chunk 置换,
        // 构建产物域内拼接错序;源域 packed/scales/zp 均行主序,concat
        // 合法)。要求两子均 Ct 形态(cyankiwi 满足;F16 兜底混态不支持)。
        if let Some((b1, b2)) = split_merged(base) {
            let suffix = &key[base.len()..];
            let Some((n1, k1, f1)) = self.linear_form(&b1) else { return Ok(None) };
            let Some((n2, k2, f2)) = self.linear_form(&b2) else { return Ok(None) };
            let (QuantForm::Ct { se: se1, ze: ze1, .. }, QuantForm::Ct { se: se2, ze: ze2, .. }) = (&f1, &f2) else {
                return Ok(None);
            };
            if k1 != k2 {
                return Ok(None);
            }
            {
                let cache = self.cache_guard();
                if let Some(b) = cache.0.get(key) {
                    return Ok(Some(b.clone()));
                }
            }
            let shard = base.bytes().map(|b| b as usize).sum::<usize>() % self.build_locks.len();
            let _g = self.build_guard(shard);
            // 双检(锁内)
            {
                let cache = self.cache_guard();
                if let Some(b) = cache.0.get(key) {
                    return Ok(Some(b.clone()));
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
                        _ => return Ok(None),
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
                            .idx_guard()
                            .entry((k, out, IdxKind::Fused))
                            .or_insert_with(|| Arc::new(marlin_fused_indices(k, out)))
                            .clone();
                        let mut b_buf: Vec<i32> = Vec::new();
                        pack_marlin_b_fused(&packed, &idx, words, &mut b_buf);
                        i32s_le_bytes(&b_buf)
                    }
                }
                _ => return Ok(None),
            };
            let mut cache = self.cache_guard();
            cache.0.insert(key.to_string(), Arc::from(bytes.into_boxed_slice()));
            return Ok(cache.0.get(key).cloned());
        }
        let Some((out, k, form)) = self.linear_form(base) else { return Ok(None) };
        {
            let cache = self.cache_guard();
            if let Some(b) = cache.0.get(key) {
                return Ok(Some(b.clone()));
            }
        }
        self.build_linear(base, out, k, form)?;
        Ok(self.cache_guard().0.get(key).cloned())
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
        let _g = self.build_guard(shard);
        // 双检(锁内):并发首触(同层 qweight/scales/zeros 落不同装载桶)
        // 时先到者已填缓存 —— 重复构建直接短路(cap 16GB 后 clear=0,本路径
        // 仅护并发窗口)
        {
            let cache = self.cache_guard();
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
        if marlin_eligible(out, k) && device {
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
                    // (P1.5-a 的 int8 两键**不在这里**:那样会连带把 int4 三件套
                    //  也给 lm_head 烘一遍 —— 专用分支见 `linear_bytes_for`)
                }
            }
            let s_len = s_bytes.len();
            inserts.push((format!("{base}.scales"), Arc::from(s_bytes.into_boxed_slice()), s_len));
            inserts.push((
                format!("{base}.zeros"),
                Arc::from(i32s_le_bytes(&z_buf).into_boxed_slice()),
                z_buf.len() * 4,
            ));
        } else if marlin_eligible(out, k) {
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
                    .idx_guard()
                    .entry((k, out, IdxKind::Fused))
                    .or_insert_with(|| Arc::new(marlin_fused_indices(k, out)))
                    .clone();
                pack_marlin_b_fused(&packed, &idx, words, &mut b_buf);
            } else {
                let idx = self
                    .idx_guard()
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
        if let Some((b1, b2)) = split_merged(base) {
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
            let _g = self.build_guard(shard);
            let guard_key = if want_b { format!("{base}.qweight_a8") } else { format!("{base}.scales_a8") };
            {
                let cache = self.cache_guard();
                if cache.0.contains_key(&guard_key) {
                    return Ok(());
                }
            }
            let t0 = std::time::Instant::now();
            // 源域拼接:scales(bf16 两段 → 单遍索引表直出 a8 字节)/
            // zp(u8 两段)/ nibble(仅 want_b)
            let sb1 = &self.maps[se1.map_ix][se1.start..se1.start + se1.nbytes];
            let sb2 = &self.maps[se2.map_ix][se2.start..se2.start + se2.nbytes];
            let max = bf16_parts_max(&[sb1, sb2]);
            let s_a8 = self.pack_scales_a8_bytes(&[sb1, sb2], out, k, max);
            self.maps[se1.map_ix].dontneed(se1.start, se1.nbytes);
            self.maps[se2.map_ix].dontneed(se2.start, se2.nbytes);
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
            for (m, e) in [(ze1.map_ix, ze1), (ze2.map_ix, ze2)] {
                self.maps[m].dontneed(e.start, e.nbytes);
            }
            let factor = max / 4096.0;
            let z_buf = pack_marlin_z_a8(&zp_u8, out, groups);
            let mut inserts: Vec<(String, Arc<[u8]>, usize)> = vec![
                (format!("{base}.scales_a8"), Arc::from(s_a8.into_boxed_slice()), out * groups * 2),
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
        if !marlin_eligible(out, k) {
            return Err(ModelError::Msg(format!(
                "awq/a8: {base} (out={out}, k={k}) 不满足 marlin 打包约束(n%256==0 && k%128==0)"
            )));
        }
        let shard = base.bytes().map(|b| b as usize).sum::<usize>() % self.build_locks.len();
        let _g = self.build_guard(shard);
        let guard_key = if want_b { format!("{base}.qweight_a8") } else { format!("{base}.scales_a8") };
        {
            let cache = self.cache_guard();
            if cache.0.contains_key(&guard_key) {
                return Ok(());
            }
        }
        let t0 = std::time::Instant::now();
        let groups = k / 32;
        // 源数据三件:q(nibble u8 行主序 [out,k])/ scales / zp(u8 [out,groups])
        // —— ct 形态直取(scale 保留 bf16 字节段,单遍索引表直出 a8 字节),
        // 裸权重形态现场 RTN g32-sym(zp≡8;scale 走 f32 旧链)。
        let (q_u8, zp_u8, s_a8, factor): (Vec<u8>, Vec<u8>, Vec<u8>, f32) = match form {
            QuantForm::Ct { pe, se, ze } => {
                let packed = i32s_par(&self.maps[pe.map_ix][pe.start..pe.start + pe.nbytes]);
                let sb = &self.maps[se.map_ix][se.start..se.start + se.nbytes];
                let zp_i32 = i32s_par(&self.maps[ze.map_ix][ze.start..ze.start + ze.nbytes]);
                self.maps[pe.map_ix].dontneed(pe.start, pe.nbytes);
                self.maps[se.map_ix].dontneed(se.start, se.nbytes);
                self.maps[ze.map_ix].dontneed(ze.start, ze.nbytes);
                let max = bf16_parts_max(&[sb]);
                (
                    if want_b { unpack_nibbles(&packed, out, k) } else { Vec::new() },
                    unpack_zp_ct(&zp_i32, out, groups),
                    self.pack_scales_a8_bytes(&[sb], out, k, max),
                    max / 4096.0,
                )
            }
            QuantForm::F16 { we } => {
                // 裸权重 → f32 → RTN int4 g32 对称(与 a16 臂 F16 兜底同源;
                // 此支非 AWQ 生产路 —— 27B ct 档走上面那条快路)
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
                let s_f32 = rtn_g32_sym_scales(&w_f32, out, k);
                let (s_bits, factor) = pack_marlin_s_a8(&s_f32, out, groups);
                (
                    if want_b { rtn_g32_sym_packed(&w_f32, out, k) } else { Vec::new() },
                    vec![8u8; out * groups],
                    u16s_le_bytes(&s_bits),
                    factor,
                )
            }
        };
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
                Arc::from(s_a8.into_boxed_slice()),
                out * groups * 2,
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
    let (a, b, cut) = split_scale_parts(parts, n);
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

/// 源分段解析(1/2 段;合并虚拟键只有左右半)+ **覆盖度校验**
/// (索引表指到段外 = 编程错,亮声拒,不得静默读错字节)。
/// 返回 `(左段, 右段, 右段起点)`;单段时 `cut = usize::MAX`(永不命中右段)。
pub(crate) fn split_scale_parts<'a>(parts: &[&'a [u8]], n: usize) -> (&'a [u8], &'a [u8], usize) {
    // 段数只容 1/2(多于两段 = 亮声拒;合并虚拟键只有左右半)
    let (a, b, cut) = match parts {
        [] => panic!("scales 源分段为空"),
        [a] => (*a, *a, usize::MAX),
        [a, b] => (*a, *b, a.len() / 2),
        _ => panic!("scales 源分段 > 2:{} 段未支持", parts.len()),
    };
    assert!(
        a.len() % 2 == 0 && (parts.len() != 2 || b.len() % 2 == 0),
        "scales 源字节量必须偶数"
    );
    let esrc = a.len() / 2 + if parts.len() == 2 { b.len() / 2 } else { 0 };
    assert!(esrc >= n, "scales 源元素数 {esrc} < 目标 {n}");
    (a, b, cut)
}

/// **a8 scales 单遍融合 gather**(2026-10-17;装载热路径):与
/// [`pack_scales_bf16_into`] 同骨架(同分段解析/同分块并行),差别在写侧
/// —— 目标字节 = **int16 归一后的位型**(`q = round(v/max·4096)` → clamp
/// 到 i16 正域),与旧链 [`owl_kernels::family::marlin::repack::pack_marlin_s_a8`]
/// 的算术**逐位同源**(同 bf16→f32 展宽、同表达式序 `f / max * 4096.0`、
/// 同 `round().clamp`)。
///
/// `max` = [`bf16_parts_max`] 的产值;`max == 0.0`(全零尺度)由调用侧早退。
pub(crate) fn pack_scales_a8_bf16_into(
    parts: &[&[u8]],
    out: usize,
    groups: usize,
    idx: &[u32],
    max: f32,
    dst: &mut [u8],
) {
    use rayon::prelude::*;
    assert!(max > 0.0, "a8 scales 需正的全局因子(0 = 调用侧早退)");
    let n = out * groups;
    assert_eq!(idx.len(), n, "a8 scales 索引表长度 = (groups,out) 展平");
    assert!(dst.len() >= n * 2, "a8 scales 目标字节不足");
    let (a, b, cut) = split_scale_parts(parts, n);
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
                let q = (f / max * 4096.0).round();
                let qi = (q as i64).clamp(0, i16::MAX as i64) as i16 as u16;
                d2.copy_from_slice(&qi.to_le_bytes());
            }
        });
}

/// bf16 源分段的全局最大(装载热路径;a8 全局因子 `max/4096` 的输入)。
///
/// 值与 `bf16_par(..).iter().fold(0.0, f32::max)` **逐位同**(bf16→f32 是
/// 零舍入展宽;`f32::max` 是选择而非算术 ⇒ 与遍历序无关),但零 f32 中转
/// 分配、单遍直读源字节。
pub(crate) fn bf16_parts_max(parts: &[&[u8]]) -> f32 {
    use rayon::prelude::*;
    if parts.is_empty() {
        return 0.0;
    }
    for p in parts {
        assert!(p.len() % 2 == 0, "bf16 源字节量必须偶数");
    }
    parts
        .par_iter()
        .map(|p| {
            p.par_chunks(1 << 13)
                .map(|chunk| {
                    chunk.chunks_exact(2).fold(0f32, |m, b| {
                        let bits = u16::from_le_bytes([b[0], b[1]]);
                        m.max(f32::from_bits((bits as u32) << 16))
                    })
                })
                .reduce(|| 0f32, f32::max)
        })
        .reduce(|| 0f32, f32::max)
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

/// **RTN int8 per-row-sym**(P1.5-a;lm_head 专用):`s_o = absmax/127`(全零行 = 1),
/// `q = clamp(RN-even(w/s), -127, 127)`。返回 `(i8 字节 [out×k] 行主, f32 LE [out])`。
///
/// ⚠️ **两条与设备核逐元素一致的纪律**(台子 `quant_i8_row_w_bench` 与
/// `gemv_i8w_bench` 背书):
/// 1. **除法不换倒数**(两侧都写 `w/s`;`w*(1/s)` 会在舍入上分家);
/// 2. **RN-even**(`round_ties_even` ↔ `__float2int_rn`);`round()` 是 half-away,**不可用**。
pub(crate) fn rtn_i8_row_sym(w: &[f32], out: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    use rayon::prelude::*;
    let mut q = vec![0u8; out * k];
    let sc: Vec<f32> = (0..out)
        .into_par_iter()
        .map(|o| {
            let row = &w[o * k..(o + 1) * k];
            let amax = row.iter().fold(0f32, |m, v| m.max(v.abs()));
            if amax > 0.0 { amax / 127.0 } else { 1.0 }
        })
        .collect();
    q.par_chunks_mut(k).enumerate().for_each(|(o, dst)| {
        let s = sc[o];
        for (i, v) in w[o * k..(o + 1) * k].iter().enumerate() {
            let qi = (v / s).round_ties_even().clamp(-127.0, 127.0);
            dst[i] = (qi as i8) as u8;
        }
    });
    let mut sb = Vec::with_capacity(out * 4);
    for v in &sc {
        sb.extend_from_slice(&v.to_le_bytes());
    }
    (q, sb)
}

/// **BF16 源 → per-row int8 对称**:与 `rtn_i8_row_sym(bf16_par(..))` 逐位同源,
/// 但不建 5.1GB f32 中间数组 —— 两遍直读 BF16(行 amax / 量化)。
pub(crate) fn rtn_i8_row_sym_bf16(bytes: &[u8], out: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    use rayon::prelude::*;
    assert_eq!(bytes.len(), out * k * 2, "bf16 源字节数 = out*k*2");
    let sc: Vec<f32> = (0..out)
        .into_par_iter()
        .map(|o| {
            let row = &bytes[o * k * 2..(o + 1) * k * 2];
            // 第一遍只做**位型 abs-max**：BF16 正域编码单调 ⇒ 清符号位后的
            // u16 最大者就是 absmax（无需逐元素展开 f32/abs）。最后才把
            // 这一个位型展宽成 f32 求 scale，与旧式逐位一致。
            let mut max_bits = 0u16;
            for b in row.chunks_exact(2) {
                max_bits = max_bits.max(u16::from_le_bytes([b[0], b[1]]) & 0x7fff);
            }
            let amax = f32::from_bits((max_bits as u32) << 16);
            if amax > 0.0 { amax / 127.0 } else { 1.0 }
        })
        .collect();
    let mut q = vec![0u8; out * k];
    q.par_chunks_mut(k).enumerate().for_each(|(o, dst)| {
        let s = sc[o];
        let row = &bytes[o * k * 2..(o + 1) * k * 2];
        quant_i8_row_bf16(row, dst, s);
    });
    let mut sb = Vec::with_capacity(out * 4);
    for v in &sc {
        sb.extend_from_slice(&v.to_le_bytes());
    }
    (q, sb)
}

/// 单行 BF16 → i8（`q = RN-even(v/s)`，clamp ±127）。
/// AVX2 加速（除法/round/饱和都是逐元素精确同式）；其余架构/尾部走标量。
#[inline]
fn quant_i8_row_bf16(row: &[u8], dst: &mut [u8], s: f32) {
    debug_assert_eq!(row.len(), dst.len() * 2);
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: 调用前已做运行期 AVX2 检测；row/dst 长度由 debug_assert
            // 与生产调用点保证（out*k*2 / out*k）。
            unsafe { quant_i8_row_bf16_avx2(row, dst, s) };
            return;
        }
    }
    quant_i8_row_bf16_scalar(row, dst, s);
}

fn quant_i8_row_bf16_scalar(row: &[u8], dst: &mut [u8], s: f32) {
    for (i, b) in row.chunks_exact(2).enumerate() {
        let bits = u16::from_le_bytes([b[0], b[1]]);
        let v = f32::from_bits((bits as u32) << 16);
        let qi = (v / s).round_ties_even().clamp(-127.0, 127.0);
        dst[i] = (qi as i8) as u8;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn quant_i8_row_bf16_avx2(row: &[u8], dst: &mut [u8], s: f32) {
    use std::arch::x86_64::*;
    let n = dst.len();
    let mut i = 0usize;
    // SAFETY: 循环条件保证 [i, i+8) 在 row（2*n 字节）与 dst（n 字节）内；
    // loadu/storeu 允许未对齐。
    unsafe {
        let sv = _mm256_set1_ps(s);
        let lo = _mm256_set1_ps(-127.0);
        let hi = _mm256_set1_ps(127.0);
        while i + 8 <= n {
            let bits = _mm_loadu_si128(row.as_ptr().add(i * 2).cast::<__m128i>());
            let w = _mm256_cvtepu16_epi32(bits);
            let f = _mm256_castsi256_ps(_mm256_slli_epi32(w, 16));
            let q = _mm256_div_ps(f, sv);
            let r = _mm256_round_ps(
                q,
                (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC) as i32,
            );
            let r = _mm256_min_ps(_mm256_max_ps(r, lo), hi);
            let iv = _mm256_cvttps_epi32(r);
            let mut tmp = [0i32; 8];
            _mm256_storeu_si256(tmp.as_mut_ptr().cast::<__m256i>(), iv);
            for (j, v) in tmp.iter().enumerate() {
                dst[i + j] = (*v as i8) as u8;
            }
            i += 8;
        }
    }
    if i < n {
        quant_i8_row_bf16_scalar(&row[i * 2..], &mut dst[i..], s);
    }
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

#[cfg(test)]
mod i8_row_tests {
    use super::*;

    /// AVX2 与标量逐位同门：统一用 `rtn_i8_row_sym_bf16`（内部按 CPU 分派），
    /// 再与逐元素标量参考对拍。覆盖 0 / 极值 / 恰好半程（ties-even）附近。
    #[test]
    fn bf16_i8_row_avx2_matches_scalar_including_ties() {
        let (out, k) = (37usize, 1031usize); // k 非 8 倍数：同时锁尾部
        let mut vals = Vec::with_capacity(out * k);
        for i in 0..out * k {
            let x = match i % 13 {
                0 => 0.0,
                1 => 127.0,
                2 => -127.0,
                3 => 0.5,
                4 => -0.5,
                5 => 1.5,
                6 => -1.5,
                _ => ((i as f32 * 0.019).sin() * 9.0) + (i % 7) as f32 * 0.125,
            };
            vals.push(x);
        }
        let bf: Vec<u8> = vals
            .iter()
            .flat_map(|f| ((f.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let (q, _) = rtn_i8_row_sym_bf16(&bf, out, k);
        for o in 0..out {
            let row = &bf[o * k * 2..(o + 1) * k * 2];
            let amax = row
                .chunks_exact(2)
                .fold(0f32, |m, b| {
                    m.max(f32::from_bits(
                        (u16::from_le_bytes([b[0], b[1]]) as u32) << 16,
                    ).abs())
                });
            let s = if amax > 0.0 { amax / 127.0 } else { 1.0 };
            let mut want = vec![0u8; k];
            quant_i8_row_bf16_scalar(row, &mut want, s);
            assert_eq!(&q[o * k..(o + 1) * k], &want[..], "row {o} 逐位");
        }
    }
}

//! **引擎侧 VLM 请求态 + 视觉塔输出缓存**(2026-10-20 视觉接线 V2;
//! 2026-10-26 塔输出跨 turn 复用)。
//!
//! 模型层 [`owl_models::vlm::Vlm`] 负责塔/表/注入;本模块持:
//! - 一个 turn 的已备料(host 表 + 组合后的塔输出设备叶块);
//! - 引擎级**逐图 merged 缓存**(按像素内容指纹;跨 turn 存活)。
//!
//! 为什么可以按图缓存:塔输出只依赖
//! `像素 + grid + 塔权重/dtype`,**不依赖文本 position**。文本位置只影响
//! 后续 m-RoPE 注入;图片在历史里移动/删除不会改变它自己的 `merged` 行。
//!
//! 生命周期:
//! - 缓存条目归 `Owner::Cache`,不随 turn 回收;
//! - `ActiveTurn.vlm.merged` 是**本 turn 新组合**的 `[N,D]` 块,归 turn;
//!   它不直接引用缓存块,所以缓存 LRU 出清不会踩到活跃 turn。

use owl_iface::contract::Bytes;
use owl_models::contract::{Dtype, ModelError};
use owl_models::vlm::{VlmImage, VlmInputs};
use owl_models::TensorOps;

/// 每个 rank 的视觉塔输出缓存上限(字节)。
///
/// 27B:一张 16×16 图 = 64 行 × 5120 × f16 ≈ 640 KiB;一屏 30 张历史图
/// ≈ 19 MiB。256 MiB 留足多会话/大图余量,同时不会把 3080 的 20G 吃掉。
const VLM_MERGED_CACHE_CAP_BYTES: usize = 256 << 20;

/// 一个 turn 的 VLM 备料(`ActiveTurn.vlm`)
pub(crate) struct VlmPrepared {
    /// 位置号/行号/ m-RoPE 表(host 副本 + 设备张量;chunk 切片用)
    pub(crate) ins: VlmInputs,
    /// 塔输出 `merged [N,D]` 的设备叶块(所有权 = turn;叶子只借 id)
    pub(crate) merged: Option<(Bytes, Vec<u64>, TensorOps)>,
}

/// 一张图的缓存键(内容 + 形状)。
///
/// 双 64 位指纹只是**避免碰撞**;缓存是进程内值,不需要跨进程稳定。
/// 像素按 `f32.to_bits()` 逐位哈希 ⇒ 预处理输出的任何变化都会 miss。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct VlmImageKey {
    h1: u64,
    h2: u64,
    pixels: usize,
    grid: (usize, usize, usize),
}

impl VlmImageKey {
    pub(crate) fn of(image: &VlmImage) -> Self {
        let mut h1 = 0xcbf2_9ce4_8422_2325u64;
        let mut h2 = 0x9e37_79b9_7f4a_7c15u64;
        for &v in &image.pixel_values {
            let bits = v.to_bits() as u64;
            h1 = (h1 ^ bits).wrapping_mul(0x0000_0100_0000_01b3);
            h2 ^= bits
                .wrapping_add(0x9e37_79b9_7f4a_7c15)
                .wrapping_add(h2 << 6)
                .wrapping_add(h2 >> 2);
        }
        Self {
            h1,
            h2,
            pixels: image.pixel_values.len(),
            grid: image.grid,
        }
    }
}

struct VlmMergedEntry {
    key: VlmImageKey,
    rows: usize,
    block: Bytes,
    leaf: TensorOps,
}

/// 引擎级逐图塔输出缓存(每 rank 一份;LRU)。
pub(crate) struct VlmMergedCache {
    entries: Vec<VlmMergedEntry>,
    used_bytes: usize,
    cap_bytes: usize,
    pub(crate) hits: u64,
    pub(crate) misses: u64,
    pub(crate) evictions: u64,
}

impl Default for VlmMergedCache {
    fn default() -> Self {
        Self::new(VLM_MERGED_CACHE_CAP_BYTES)
    }
}

impl VlmMergedCache {
    pub(crate) fn new(cap_bytes: usize) -> Self {
        Self {
            entries: Vec::new(),
            used_bytes: 0,
            cap_bytes,
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn used_bytes(&self) -> usize {
        self.used_bytes
    }

    /// 命中:把条目移到 LRU 尾部,返回 `(leaf, rows)`。
    pub(crate) fn get(&mut self, key: VlmImageKey) -> Option<(TensorOps, usize)> {
        let pos = self.entries.iter().position(|e| e.key == key)?;
        let e = self.entries.remove(pos);
        let out = (e.leaf.clone(), e.rows);
        self.entries.push(e);
        self.hits += 1;
        Some(out)
    }

    /// 插入(同键旧条目只会出现在异常调用;返回旧块 id 由调用方释放)。
    pub(crate) fn insert(
        &mut self,
        key: VlmImageKey,
        rows: usize,
        block: Bytes,
        leaf: TensorOps,
    ) -> Option<u64> {
        self.misses += 1;
        let old = if let Some(pos) = self.entries.iter().position(|e| e.key == key) {
            let e = self.entries.remove(pos);
            self.used_bytes = self.used_bytes.saturating_sub(block_bytes(&e.block));
            Some(e.block.id)
        } else {
            None
        };
        self.used_bytes += block_bytes(&block);
        self.entries.push(VlmMergedEntry { key, rows, block, leaf });
        old
    }

    /// 从最旧开始出清到上限,返回需要交给设备面 `free` 的块 id。
    ///
    /// 调用点必须在**本次组合求值完成之后**调用 —— 组合张量只借用缓存块
    /// 作为输入,求值结束才可安全释放。
    pub(crate) fn evict_to_cap(&mut self) -> Vec<u64> {
        let mut ids = Vec::new();
        while self.used_bytes > self.cap_bytes && !self.entries.is_empty() {
            let e = self.entries.remove(0);
            self.used_bytes = self.used_bytes.saturating_sub(block_bytes(&e.block));
            ids.push(e.block.id);
            self.evictions += 1;
        }
        ids
    }
}

fn block_bytes(b: &Bytes) -> usize {
    b.len.saturating_mul(b.dtype.size_bytes())
}

/// 单张图的 merged 行数(塔输出 `[N,D]` 的 N)。
pub(crate) fn merged_rows(
    grid: (usize, usize, usize),
    merge: usize,
) -> Result<usize, ModelError> {
    let (t, h, w) = grid;
    if merge == 0 || t == 0 || h == 0 || w == 0 {
        return Err(ModelError::Msg(format!(
            "VLM 图像网格非法:grid={grid:?} merge={merge}"
        )));
    }
    if !h.is_multiple_of(merge) || !w.is_multiple_of(merge) {
        return Err(ModelError::Msg(format!(
            "VLM 图像网格 {grid:?} 不是 merge={merge} 的整数倍"
        )));
    }
    t.checked_mul(h / merge)
        .and_then(|n| n.checked_mul(w / merge))
        .ok_or_else(|| ModelError::Msg(format!("VLM 图像 merged 行数溢出:grid={grid:?}")))
}

/// 把逐图缓存块组合成本 turn 的连续 `[N,D]` merged。
///
/// **为什么不用 `concat_rows`**:那张核要求每段行数相同,而不同截图的
/// merged 行数可以不同。这里用 `image_scatter` 的稀疏行选择做通用拼接:
/// 先建 `[N,D]` 零块,再对每张图发一次 scatter(只写自己那段行),
/// 组合块归 turn,缓存块只作为输入;求值完成后缓存可自由出清。
pub(crate) fn combine_merged(
    leaves: &[(TensorOps, usize)],
    hidden: usize,
) -> Result<TensorOps, ModelError> {
    let total = leaves.iter().try_fold(0usize, |acc, (_, rows)| {
        acc.checked_add(*rows)
            .ok_or_else(|| ModelError::Msg("VLM merged 行数溢出".into()))
    })?;
    if total == 0 || hidden == 0 {
        return Err(ModelError::Msg(format!(
            "VLM merged 组合非法:rows={total} hidden={hidden}"
        )));
    }
    let mut x = TensorOps::zeros(Dtype::F16, vec![total, hidden]);
    let mut offset = 0usize;
    for (leaf, rows) in leaves {
        if leaf.dtype() != Dtype::F16 {
            return Err(ModelError::Msg(format!(
                "VLM merged 组合:缓存块 dtype {:?} ≠ F16",
                leaf.dtype()
            )));
        }
        let want = [*rows, hidden];
        if leaf.shape() != want {
            return Err(ModelError::Msg(format!(
                "VLM merged 组合:缓存块 {:?} ≠ {want:?}",
                leaf.shape()
            )));
        }
        if *rows == 0 {
            return Err(ModelError::Msg("VLM merged 组合:图片 merged 行数为 0".into()));
        }
        let mut idx = vec![-1.0f32; total];
        for (r, slot) in idx[offset..offset + rows].iter_mut().enumerate() {
            *slot = r as f32;
        }
        let idx_bytes: Vec<u8> = idx.iter().flat_map(|v| v.to_le_bytes()).collect();
        let idx = TensorOps::from_host(Dtype::F32, vec![total], &idx_bytes);
        x = x.image_scatter(leaf, &idx);
        offset += rows;
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(salt: f32, grid: (usize, usize, usize)) -> VlmImage {
        VlmImage {
            pixel_values: (0..16).map(|i| salt + i as f32 * 0.25).collect(),
            grid,
        }
    }

    #[test]
    fn key_tracks_pixels_and_grid() {
        let a = image(1.0, (1, 4, 4));
        assert_eq!(VlmImageKey::of(&a), VlmImageKey::of(&a));
        assert_ne!(VlmImageKey::of(&a), VlmImageKey::of(&image(1.5, (1, 4, 4))));
        assert_ne!(VlmImageKey::of(&a), VlmImageKey::of(&image(1.0, (1, 4, 8))));
    }

    #[test]
    fn lru_evicts_oldest_after_cap() {
        let key = |n| VlmImageKey {
            h1: n,
            h2: n,
            pixels: 16,
            grid: (1, 4, 4),
        };
        let mut cache = VlmMergedCache::new(0);
        let block = |id| Bytes::new(id, 16, Dtype::F16, 0);
        for id in 0..3 {
            cache.insert(key(id), 4, block(id), TensorOps::of_block(id, Dtype::F16, vec![4, 4]));
        }
        assert_eq!(cache.len(), 3);
        let evicted = cache.evict_to_cap();
        assert_eq!(evicted, vec![0, 1, 2], "零上限必须从最旧开始全部出清");
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.used_bytes(), 0);
    }

    #[test]
    fn merged_rows_validates_grid() {
        assert_eq!(merged_rows((1, 16, 16), 2).unwrap(), 64);
        assert!(merged_rows((1, 15, 16), 2).is_err());
        assert!(merged_rows((0, 16, 16), 2).is_err());
    }
}

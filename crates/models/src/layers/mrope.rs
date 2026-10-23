//! M-RoPE:三网格(T/H/W)位置编码 —— **Qwen3.5-VL 文本侧**专用(host 纯函数,零核)。
//!
//! 事实面(HF 5.19 `modeling_qwen3_5.py`;逐行对照见
//! `docs/arch/vision-tower-设计与CPU对拍.md` §1.4):
//!
//! - **位置号**(`Qwen3_5Model.get_rope_index`):按 token 类型分段 ——
//!   文本段 `arange(len) + cur`,三网格同值;视觉段 = `get_vision_position_ids`
//!   的 h-major 网格(H/W 各除 `spatial_merge_size`,T 网格整段恒等于段首位),
//!   段末 `cur += max(h, w) / merge`;
//! - **频率重组**(`Qwen3_5TextRotaryEmbedding.recomposition_frequencies`):
//!   频率维 j 按 `slice(dim, section[dim]*3, 3)` 交错认网格(interleaved),
//!   `[11, 11, 10]` 恰好铺满 `half = rotary_dim/2 = 32`;输出 `cat(f, f)`
//!   ⇒ 配对 `(j, j + half)` 与现役 rotate-half 核同构 —— 故本模块产出
//!   `[L, half]` 行表,行内逐 j 取"j 所属网格"的位置;
//! - **纯文本退化**:三网格同值 ⇒ 表与现役 1D rope 表逐元素同(零回归,R5)。
//!
//! 产出姿势 = **host 表**(与视觉塔 `VisionTables::build` 的 axial cos/sin 同款):
//! 表经 `TensorOps::from_host` 上传后,现役 rope 核按"行号 = token 号"直取
//! —— 零新核、零新语义算子。**契约**:表路径要求 pos 通道 = 行号(0..L),
//! 即 prefill 单块整发(`pos = arange(L)`);decode 步三网格同值 ⇒ 仍走现役
//! 标量 pos 路径(表不必换)。

use crate::contract::ModelError;
use crate::layers::vision::GridThw;

/// token 类型(与 HF `mm_token_type_ids` 同位;文本 0 / 图像 1 / 视频 2)
pub const TYPE_TEXT: u8 = 0;
pub const TYPE_IMAGE: u8 = 1;
pub const TYPE_VIDEO: u8 = 2;

/// M-RoPE 超参(`config.json:text_config.rope_parameters` + `vision_config`)
pub struct MropeCfg {
    /// 三网格频率段 `[11, 11, 10]`(和 = half)
    sections: [usize; 3],
    /// 旋转维(partial_rotary_factor × head_dim;4B = 64)
    rotary_dim: usize,
    /// rope_theta(1e7)
    theta: f32,
    /// `vision_config.spatial_merge_size`(2)
    merge: usize,
}

impl MropeCfg {
    /// 超参校验(结构非法 = 容器语义不成立,非数据错)
    pub fn new(
        sections: [usize; 3],
        rotary_dim: usize,
        theta: f32,
        merge: usize,
    ) -> Result<MropeCfg, ModelError> {
        if !rotary_dim.is_multiple_of(2) || rotary_dim == 0 {
            return Err(ModelError::Msg(format!(
                "MropeCfg::new: rotary_dim {rotary_dim} 非法(须正偶数)"
            )));
        }
        if sections.iter().sum::<usize>() != rotary_dim / 2 {
            return Err(ModelError::Msg(format!(
                "MropeCfg::new: sections {sections:?} 和 ≠ half = {}",
                rotary_dim / 2
            )));
        }
        if merge == 0 {
            return Err(ModelError::Msg("MropeCfg::new: spatial_merge_size = 0".into()));
        }
        if !(theta > 0.0) {
            return Err(ModelError::Msg(format!("MropeCfg::new: theta {theta} 非法")));
        }
        Ok(MropeCfg { sections, rotary_dim, theta, merge })
    }

    /// 旋转半维(表宽 = 每行频率数)
    pub fn half(&self) -> usize {
        self.rotary_dim / 2
    }

    /// 频率维 j → 网格号(0 = T,1 = H,2 = W;HF `recomposition_frequencies`
    /// 的 `slice(offset, section[dim]*3, 3)` 交错语义的直译)
    pub fn freq_grid(&self) -> Vec<usize> {
        let half = self.half();
        let mut grid = vec![0usize; half];
        for (dim, section) in self.sections.iter().enumerate().skip(1) {
            // 段长 section×3:j ∈ (dim, dim+3·section) 步长 3 → 恰 section 个
            for j in (dim..section.saturating_mul(3)).step_by(3) {
                if j < half {
                    grid[j] = dim;
                }
            }
        }
        grid
    }

    /// 单张图的 3 网格位置号(h-major;HF `get_vision_position_ids(start, grid, 1, merge)`)。
    ///
    /// `grid` = 原始 patch 网格 `(t, h, w)`;H/W 各除 merge,T 不减。
    pub fn vision_position_ids(
        &self,
        start: i64,
        grid: GridThw,
    ) -> Result<Vec<[i64; 3]>, ModelError> {
        let (t, h, w) = grid;
        if t == 0 || h == 0 || w == 0 {
            return Err(ModelError::Msg(format!(
                "vision_position_ids: 空网格 {grid:?}(图/帧必须三轴非零)"
            )));
        }
        if h % self.merge != 0 || w % self.merge != 0 {
            return Err(ModelError::Msg(format!(
                "vision_position_ids: 网格 {grid:?} 的 h/w 不是 merge = {} 的整数倍",
                self.merge
            )));
        }
        let (lh, lw) = (h / self.merge, w / self.merge);
        let mut out = Vec::with_capacity(t * lh * lw);
        for _ti in 0..t {
            for hi in 0..lh {
                for wi in 0..lw {
                    out.push([start, start + hi as i64, start + wi as i64]);
                }
            }
        }
        Ok(out)
    }

    /// 一段位置的"段末推进量"(HF `cur += max(grid[1], grid[2]) // merge`)
    fn advance(&self, grid: GridThw) -> i64 {
        (grid.1.max(grid.2) / self.merge) as i64
    }

    /// 位置号全序列(按类型分段推进;HF `get_rope_index` 单 batch 行)。
    ///
    /// `ids` 仅用于长度对齐(HF 的 `input_ids` 在算法内不参与取值);
    /// `ttypes` = 逐 token 类型;`images` / `videos` = 各类网格按出现序
    /// (视频侧 HF 先按帧展开:每帧 T=1 —— 见 [`split_video_frames`])。
    pub fn rope_index(
        &self,
        ids: &[u32],
        ttypes: &[u8],
        images: &[GridThw],
        videos: &[GridThw],
    ) -> Result<MropeIndex, ModelError> {
        if ids.len() != ttypes.len() {
            return Err(ModelError::Msg(format!(
                "rope_index: ids 长 {} ≠ token 类型长 {}",
                ids.len(),
                ttypes.len()
            )));
        }
        let mut pos: Vec<[i64; 3]> = Vec::with_capacity(ttypes.len());
        let (mut img_it, mut vid_it) = (0usize, 0usize);
        let (mut cur, mut i) = (0i64, 0usize);
        while i < ttypes.len() {
            let ty = ttypes[i];
            let mut j = i;
            while j < ttypes.len() && ttypes[j] == ty {
                j += 1;
            }
            let run_len = j - i;
            match ty {
                TYPE_TEXT => {
                    for k in 0..run_len {
                        let p = cur + k as i64;
                        pos.push([p, p, p]);
                    }
                    cur += run_len as i64;
                }
                TYPE_IMAGE | TYPE_VIDEO => {
                    let (grids, grid_start, kind) = if ty == TYPE_IMAGE {
                        (&images, img_it, "图像")
                    } else {
                        (&videos, vid_it, "视频")
                    };
                    // 一段连续视觉 token 可能包含**多张相邻图/多帧**;
                    // 按 grid 顺序逐个消费,而不是把整段当成单图。
                    let mut grid_idx = grid_start;
                    let mut used = 0usize;
                    while used < run_len {
                        let grid = *grids.get(grid_idx).ok_or_else(|| {
                            ModelError::Msg(format!(
                                "rope_index: {kind}段(起点 {i},长 {run_len})无对应网格(已用 {grid_idx} 张)"
                            ))
                        })?;
                        let vp = self.vision_position_ids(cur, grid)?;
                        let vlen = vp.len();
                        if used + vlen > run_len {
                            return Err(ModelError::Msg(format!(
                                "rope_index: {kind}段 token 数 {run_len} 不足网格 {grid:?} 展开 {vlen}（已用 {used}）"
                            )));
                        }
                        pos.extend(vp);
                        used += vlen;
                        cur += self.advance(grid);
                        grid_idx += 1;
                    }
                    if ty == TYPE_IMAGE {
                        img_it = grid_idx;
                    } else {
                        vid_it = grid_idx;
                    }
                }
                other => {
                    return Err(ModelError::Msg(format!(
                        "rope_index: token 类型 {other} 非法(0 文本 / 1 图像 / 2 视频)"
                    )))
                }
            }
            i = j;
        }
        // 空输入 = 退化输入(克分子表为空);原 `unwrap_or(0)` 会算出一个
        // 无意义的 delta(1 − ttypes.len(),可为负)传给位置计算 —— 静默错值
        // ⇒ 2026-10-17 审查改响亮拒
        let Some(maxpos) = pos.iter().flat_map(|p| p.iter()).copied().max() else {
            return Err(ModelError::Msg(
                "rope_index: 位置表为空(token 类型表为空?需至少一个 token)".into(),
            ));
        };
        let delta = maxpos + 1 - ttypes.len() as i64;
        Ok(MropeIndex { pos, delta })
    }

    /// cos/sin 行表(`[L, half]` f32;行内逐 j 取 "j 所属网格" 的位置)。
    ///
    /// 数值口径与现役 [`crate::layers::rope::Rope`] 表同式
    /// (`inv_freq[j] = theta^(-2j/rotary_dim)`,f32 直算)。
    pub fn cos_sin(&self, pos: &[[i64; 3]]) -> (Vec<f32>, Vec<f32>) {
        let half = self.half();
        let grid = self.freq_grid();
        let inv: Vec<f32> = (0..half)
            .map(|j| self.theta.powf(-(2.0 * j as f32) / self.rotary_dim as f32))
            .collect();
        let mut cos = vec![0f32; pos.len() * half];
        let mut sin = vec![0f32; pos.len() * half];
        for (t, p) in pos.iter().enumerate() {
            for j in 0..half {
                let ang = p[grid[j]] as f32 * inv[j];
                cos[t * half + j] = ang.cos();
                sin[t * half + j] = ang.sin();
            }
        }
        (cos, sin)
    }
}

/// 位置号结果(全序列 + 增量步的 rope delta)
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MropeIndex {
    /// `[L, 3]`:逐 token 的 (T, H, W) 位置
    pub pos: Vec<[i64; 3]>,
    /// decode 增量位 `past + delta`(HF `mrope_position_deltas`)
    pub delta: i64,
}

/// 逐 token 类型(serving 侧便利函数:input_ids 占位符 → `mm_token_type_ids`)。
///
/// 图像 token 现值 248056 / 视频 248057(4B `config.json`);
/// 二者之外的连续文本一律 0。
pub fn token_types(ids: &[u32], image_token: u32, video_token: Option<u32>) -> Vec<u8> {
    ids.iter()
        .map(|&id| {
            if id == image_token {
                TYPE_IMAGE
            } else if Some(id) == video_token {
                TYPE_VIDEO
            } else {
                TYPE_TEXT
            }
        })
        .collect()
}

/// 视频网格按帧展开(HF `get_rope_index` 入口的 `repeat_interleave(t, dim=0)`
/// + `grid[:, 0] = 1`):每帧独立成段,T 轴恒 1。
pub fn split_video_frames(grids: &[GridThw]) -> Vec<GridThw> {
    let mut out = Vec::new();
    for &(t, h, w) in grids {
        for _ in 0..t {
            out.push((1, h, w));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::rope::Rope;

    /// 4B 实测档(`Qwen3.5-4B/config.json`)
    fn cfg4b() -> MropeCfg {
        MropeCfg::new([11, 11, 10], 64, 1e7, 2).expect("4B 档超参应合法")
    }

    #[test]
    fn config_validation_is_loud() {
        assert!(MropeCfg::new([11, 11, 10], 64, 1e7, 2).is_ok());
        // 段和 ≠ half
        assert!(MropeCfg::new([11, 11, 11], 64, 1e7, 2).is_err(), "段和 33 ≠ 32 应拒");
        // 奇数旋转维
        assert!(MropeCfg::new([1, 1, 1], 7, 1e7, 2).is_err(), "奇数 rotary_dim 应拒");
        // merge = 0
        assert!(MropeCfg::new([1, 1, 1], 6, 1e7, 0).is_err(), "merge = 0 应拒");
        // theta ≤ 0
        assert!(MropeCfg::new([1, 1, 1], 6, 0.0, 2).is_err(), "theta = 0 应拒");
    }

    /// 交错映射:j 的三分律(11 T / 11 H / 10 W),与 HF slice(offset, ·, 3) 同
    #[test]
    fn freq_grid_is_interleaved() {
        let g = cfg4b().freq_grid();
        assert_eq!(g.len(), 32);
        let count = |d: usize| g.iter().filter(|&&x| x == d).count();
        assert_eq!((count(0), count(1), count(2)), (11, 11, 10), "三网格段数");
        // j % 3 直判([11,11,10] 的巧合形态;段和恒等 ⇒ 交错律)
        for (j, &d) in g.iter().enumerate() {
            assert_eq!(d, j % 3, "频率维 {j} 网格映射");
        }
    }

    /// 单图网格位置号 + 文本段推进(手算锚:3 文本 + 6 视 + 2 文本)
    #[test]
    fn rope_index_hand_computed() {
        let c = cfg4b();
        // 网格 (1, 4, 6) → llm (1, 2, 3) = 6 token
        let mut ids = vec![0u32; 3];
        ids.extend(vec![248056u32; 6]);
        ids.extend(vec![0u32; 2]);
        let tt = token_types(&ids, 248056, Some(248057));
        let out = c.rope_index(&ids, &tt, &[(1, 4, 6)], &[]).expect("rope_index");
        let want = vec![
            [0, 0, 0], // 文本 0
            [1, 1, 1],
            [2, 2, 2],
            // 图像段:start = 3,T 恒 3;H = 3 + hi;W = 3 + wi(h-major)
            [3, 3, 3], [3, 3, 4], [3, 3, 5],
            [3, 4, 3], [3, 4, 4], [3, 4, 5],
            // 段末 cur += max(4, 6) / 2 = 3 → 6
            [6, 6, 6],
            [7, 7, 7],
        ];
        assert_eq!(out.pos, want, "位置号逐 token");
        // delta = max + 1 − L = 7 + 1 − 11 = −3
        assert_eq!(out.delta, -3);
    }

    /// **相邻多图**:同一连续 image run 里按 grid 顺序消费
    /// (旧实现把整段当单图 ⇒ 第二张起直接拒/错位)。
    #[test]
    fn rope_index_multiple_adjacent_images() {
        let c = cfg4b();
        let ids = vec![248056u32; 6 + 2];
        let tt = token_types(&ids, 248056, Some(248057));
        let out = c
            .rope_index(&ids, &tt, &[(1, 4, 6), (1, 2, 4)], &[])
            .expect("相邻两图");
        assert_eq!(out.pos.len(), 8);
        assert_eq!(out.pos[0], [0, 0, 0], "图一 T/H/W 起点");
        assert_eq!(out.pos[5], [0, 1, 2], "图一末行");
        assert_eq!(out.pos[6], [3, 3, 3], "图二起点 = 图一段末 advance=3");
        assert_eq!(out.pos[7], [3, 3, 4], "图二末行");
        assert_eq!(out.delta, -3, "maxpos 4 + 1 - L 8");
    }

    /// 响亮拒绝:占位 token 数与网格不符 / 网格缺失 / 类型非法
    #[test]
    fn rope_index_rejects_mismatch() {
        let c = cfg4b();
        // 5 个图像 token ≠ 网格 6
        let mut ids = vec![0u32, 0];
        ids.extend(vec![248056u32; 5]);
        let tt = token_types(&ids, 248056, None);
        let e = c.rope_index(&ids, &tt, &[(1, 4, 6)], &[]);
        assert!(e.is_err(), "占位 token 数与网格不符应拒");
        // 无网格
        assert!(c.rope_index(&ids, &tt, &[], &[]).is_err(), "网格缺失应拒");
        // 类型非法
        let mut tt_bad = tt.clone();
        tt_bad[0] = 7;
        assert!(c.rope_index(&ids, &tt_bad, &[(1, 4, 6)], &[]).is_err());
        // ids/ttypes 长度不符
        assert!(c.rope_index(&ids, &tt[..3], &[(1, 4, 6)], &[]).is_err());
    }

    /// 视频:按帧展开成 T=1 段(每帧独立推进;帧间须有文本分隔 ——
    /// HF 的分段按 **类型连续段**,帧间时间戳 token 正是分隔符)
    #[test]
    fn video_frames_are_split() {
        let c = cfg4b();
        let frames = split_video_frames(&[(2, 4, 6)]);
        assert_eq!(frames, vec![(1, 4, 6), (1, 4, 6)]);
        // [文本][帧一 6][文本][帧二 6]
        let mut ids = vec![7u32];
        ids.extend(vec![248057u32; 6]);
        ids.push(7);
        ids.extend(vec![248057u32; 6]);
        let tt = token_types(&ids, 248056, Some(248057));
        let out = c.rope_index(&ids, &tt, &[], &frames).expect("视频两帧");
        assert_eq!(out.pos.len(), 14);
        assert_eq!(out.pos[0], [0, 0, 0], "文本段");
        assert_eq!(out.pos[1], [1, 1, 1], "帧一 T 段首");
        assert_eq!(out.pos[7], [4, 4, 4], "帧间文本:cur 1 + max(4,6)/2 = 4");
        assert_eq!(out.pos[8], [5, 5, 5], "帧二 T 段首 = 4 + 1");
    }

    /// R5 零回归(**表级**独立锚):纯文本序列的三网格同值 ⇒ 本表与现役
    /// 1D rope 表逐元素同(`Rope::new` 同式独立副本)
    #[test]
    fn text_only_table_equals_1d_rope() {
        let c = cfg4b();
        let l = 24usize;
        let ids = vec![7u32; l];
        let tt = token_types(&ids, 248056, None);
        let idx = c.rope_index(&ids, &tt, &[], &[]).expect("纯文本");
        assert_eq!(idx.delta, 0, "纯文本 delta 恒 0");
        let (cos, sin) = c.cos_sin(&idx.pos);

        // 现役 1D 表(Rope::new 的 f32 表;half = 32;取前 l 行)
        use crate::module::WeightSource;
        let rp = Rope::new(l, 256, 64, 1e7).expect("rope");
        let table = rp.tables();
        let cos_ref = table.take("cos_table").expect("源").expect("cos 表源");
        let sin_ref = table.take("sin_table").expect("源").expect("sin 表源");
        for t in 0..l {
            for j in 0..32 {
                let i = t * 32 + j;
                assert!(
                    (cos[i] - cos_ref[i]).abs() <= 1e-7,
                    "cos[{t},{j}] {} vs 现役表 {}",
                    cos[i],
                    cos_ref[i]
                );
                assert!(
                    (sin[i] - sin_ref[i]).abs() <= 1e-7,
                    "sin[{t},{j}] {} vs 现役表 {}",
                    sin[i],
                    sin_ref[i]
                );
            }
        }
    }

    /// HF parity(`OWL_HF_PARITY=1`):位置号**逐元素整数相等**(f32 形态无损)+
    /// delta 相等 + cos/sin `atol 1e-6`。四例:纯文本 / 单图 / 双图 / 视频两帧。
    #[test]
    fn parity_matches_hf() {
        use crate::testkit::{hf_python, parity_enabled, skip_note, st_read, st_write, tmp_path};
        if !parity_enabled() {
            skip_note();
            return;
        }
        const IM: u32 = 248056;
        const VID: u32 = 248057;
        let c = cfg4b();
        let mut cases: Vec<(&str, Vec<u32>, Vec<GridThw>, Vec<GridThw>)> = Vec::new();
        // ① 纯文本(无视觉段)
        cases.push(("text", vec![10, 11, 12], vec![], vec![]));
        // ② 单图 (1,4,6) → llm (1,2,3) = 6 token
        {
            let mut ids = vec![10u32];
            ids.extend(vec![IM; 6]);
            ids.push(11);
            cases.push(("image", ids, vec![(1, 4, 6)], vec![]));
        }
        // ③ 双图(不同网格,验网格迭代器逐一消费)
        {
            let mut ids = vec![IM; 6];
            ids.push(12);
            ids.extend(vec![IM; 4]);
            cases.push(("two_images", ids, vec![(1, 4, 6), (1, 2, 8)], vec![]));
        }
        // ④ 视频两帧(**python 侧收原始网格 (2,4,6)** —— HF 在 get_rope_index
        // 内部按帧展开;Rust 侧先 split_video_frames,两边应一致)
        {
            let mut ids = vec![7u32];
            ids.extend(vec![VID; 6]);
            ids.push(8);
            ids.extend(vec![VID; 6]);
            cases.push(("video", ids, vec![], vec![(2, 4, 6)]));
        }

        let half = c.half();
        for (name, ids, images, videos) in cases {
            let ttypes = token_types(&ids, IM, Some(VID));
            let l = ids.len();
            let inp = tmp_path(&format!("mrope_in_{name}"));
            let outp = tmp_path(&format!("mrope_out_{name}"));
            let f32v = |v: &[u32]| v.iter().map(|&x| x as f32).collect::<Vec<f32>>();
            let flat_grid = |g: &[GridThw]| {
                g.iter().flat_map(|&(t, h, w)| [t as f32, h as f32, w as f32]).collect::<Vec<f32>>()
            };
            let (gi, gv) = (flat_grid(&images), flat_grid(&videos));
            st_write(
                &inp,
                &[
                    ("ids", &f32v(&ids), vec![l]),
                    ("ttypes", &f32v(&ttypes.iter().map(|&x| x as u32).collect::<Vec<u32>>()), vec![l]),
                    ("image_grid_thw", &gi, vec![images.len(), 3]),
                    ("video_grid_thw", &gv, vec![videos.len(), 3]),
                ],
            );
            let contract = format!(
                r#"{{"merge": 2, "head_dim": 256, "rotary_dim": 64, "theta": 1e7, "sections": [11, 11, 10]}}"#
            );
            let manifest = hf_python("mrope.py", &[inp.to_str().unwrap(), outp.to_str().unwrap(), &contract]);
            eprintln!("[hf manifest] mrope/{name}: {manifest}");

            let want_pos = st_read(&outp, "pos_ids");
            let want_delta = st_read(&outp, "delta");
            let want_cos = st_read(&outp, "cos");
            let want_sin = st_read(&outp, "sin");

            // 视频侧:Rust API 收**已展开帧表**(`split_video_frames`),python 侧收
            // 原始网格(HF 在内部展开)—— 两边等价即验证了展开律本身
            let frames = split_video_frames(&videos);
            let idx = c
                .rope_index(&ids, &ttypes, &images, &frames)
                .expect("rope_index");
            assert_eq!(want_pos.len(), 3 * l, "[{name}] HF 位置号尺寸");
            for t in 0..l {
                for g in 0..3 {
                    assert_eq!(
                        idx.pos[t][g] as f32,
                        want_pos[g * l + t],
                        "[{name}] pos[{t}][{g}]"
                    );
                }
            }
            assert_eq!(want_delta.len(), 1, "[{name}] delta 尺寸");
            assert_eq!(idx.delta as f32, want_delta[0], "[{name}] rope delta");

            let (cos, sin) = c.cos_sin(&idx.pos);
            assert_eq!(want_cos.len(), l * half, "[{name}] cos 表尺寸");
            assert_eq!(want_sin.len(), l * half, "[{name}] sin 表尺寸");
            for i in 0..l * half {
                assert!(
                    (cos[i] - want_cos[i]).abs() <= 1e-6,
                    "[{name}] cos[{i}] {} vs HF {}",
                    cos[i],
                    want_cos[i]
                );
                assert!(
                    (sin[i] - want_sin[i]).abs() <= 1e-6,
                    "[{name}] sin[{i}] {} vs HF {}",
                    sin[i],
                    want_sin[i]
                );
            }
            owl_shared::file_loader::remove_file(&inp).ok();
            owl_shared::file_loader::remove_file(&outp).ok();
        }
    }

    /// 视觉段的表结构:T 维恒段首位 ⇒ j%3==0 的频率逐行同;H/W 维随行号
    #[test]
    fn vision_rows_follow_grid() {
        let c = cfg4b();
        let idx = c.rope_index(
            &[1u32, 248056, 248056, 248056, 248056],
            &[TYPE_TEXT, TYPE_IMAGE, TYPE_IMAGE, TYPE_IMAGE, TYPE_IMAGE],
            &[(1, 2, 4)], // llm (1,1,2) = 2 token? 需 4 → 用 (1,4,4) 见下
            &[],
        );
        assert!(idx.is_err(), "占位 4 ≠ 网格 token 数应拒");

        let (cos, _sin) = {
            let idx = c
                .rope_index(
                    &[1u32, 248056, 248056, 248056, 248056],
                    &[TYPE_TEXT, TYPE_IMAGE, TYPE_IMAGE, TYPE_IMAGE, TYPE_IMAGE],
                    &[(1, 4, 4)],
                    &[],
                )
                .expect("(1,4,4) → llm (1,2,2) = 4 token ✓");
            c.cos_sin(&idx.pos)
        };
        // 行 1..4 属图像:T 位 = 1 ⇒ j%3==0 的槽位行间相同
        let half = c.half();
        for t in 2..5 {
            for j in (0..half).step_by(3) {
                assert_eq!(cos[t * half + j], cos[half + j], "[{t},{j}] T 维应恒定");
            }
        }
        // H 位随 hi(j%3==1),W 位随 wi(j%3==2):行 1 = (h0,w0),行 2 = (h0,w1) ⇒
        // j%3==1 槽位同、j%3==2 槽位异
        assert_eq!(cos[half + 1], cos[2 * half + 1], "同 h 行:H 维相同");
        assert_ne!(cos[half + 2], cos[2 * half + 2], "同 h 行:W 维不同");
    }

    /// **空输入门**(2026-10-17 审查 §五轮):位置表为空 = 退化输入 ⇒ 响亮拒
    /// (修前 `max().unwrap_or(0)` 会算出无意义 delta 传下去)。
    #[test]
    fn rope_index_rejects_empty_input() {
        let c = cfg4b();
        let e = c.rope_index(&[], &[], &[], &[]);
        assert!(e.is_err(), "空输入必须拒,得 {e:?}");
    }
}

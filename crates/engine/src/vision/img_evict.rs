//! **前缀无感删除(图像段)** —— host 侧纯函数面(识别 / 页对齐 pad / 自证)。
//!
//! # 需求(2026-10-23 用户裁决)
//!
//! GUI agent 的历史是**图文交错**的。真正的需求只有一句话:
//! 「harness 从历史里抽掉一张旧截图,前缀缓存不许失效、删点之后不许重算。」
//!
//! 引擎**不自作主张**:客户端在图像上打「将来可能被删」标记(`TurnImage::evict`),
//! 且 `[vision] image_evict = true` 才启用。没有标记的图 → 逐字节旧路径。
//!
//! # 为什么是 leave-gap(而不是旋转 K)
//!
//! 删掉区间 `[a, b)` 后,若把后缀 K 乘 `R(−S)` 压紧位置,语义上精确,但
//! **每轮都在已经量化/旋转过的 K 上再转一次** ⇒ FP8 网格反复舍入,无法
//! 无限次丢弃旧图片。最终方案改为 **leave-gap**:
//!
//! - 删除只摘页,**不转 K、不移动幸存 token 的 position**;
//! - 新 token 从 `max_pos+1` 追加;位置账保留洞(不可复原);
//! - 因此可以**无限次**丢弃旧图片缓存,没有叠转漂移,也没有周期整段重建。
//!
//! 位置账/拼接的唯一产地 = [`compact_positions`] / [`positions_after_splice`];
//! 启动/执行面见 `session.rs`、`scheduler.rs`。
//!
//! # 为什么需要"重建"(不是纯 diff)
//!
//! 页对齐 pad 是**引擎自己插进 prompt 的 token**;客户端从不发它们。删掉一张图
//! 之后,那张图的**前置 pad 块**在新文本里没有对应物 —— 纯 diff 会从 pad 块
//! 开始分叉,删除区间就不落页边界了。
//!
//! 解法:**删点之前的 token 序列原样沿用**(`old[..span.start]`,含旧 pad 块),
//! 只把**删点之后的客户端文本**接上去,再对整条序列重跑 pad 律。于是
//! `new[..a] == old[..a]` 按构造成立,而 pad 律仍然幂等(位置差是整页)。
//!
//! 为了下一轮还能做"文本坐标"的 diff,会话账同时记住**本序列的全部 pad 区间**
//! ([`ImgSpan`] 的 `pad_before/pad_after`);`unpad` 把这些块剥掉就得到客户端
//! 可见的 token 序列。
//!
//! # 语义边界(必须先认账)
//!
//! 被删段**之后**的 KV 是"当年看得见被删内容"时算出的 ⇒ 这是**搬运记忆**,
//! 不是重算。删点**之前**是真等价(可做逐位门)。
//!
//! # 本模块的边界
//!
//! 只做 **host 纯函数**(零设备 IO、零 store 访问):段 discovery、pad 律、
//! 删除识别、位置同移自证。动手(取链 / 拼接 / 位置账 / 盘层收尾)在
//! [`crate::scheduler`] 与 `exec::img_evict`。

use owl_models::layers::mrope::MropeCfg;
use owl_models::layers::vision::GridThw;

/// 图像段/位置账类型已随会话账下沉 `owl-store::session`（2026-11-15）；
/// 这里保留 `crate::img_evict::{ImgSpan, Pos3}` 路径兼容。
pub use owl_store::session::{ImgSpan, Pos3};

/// pad 律的产物:整条序列 + 每段的坐标账 + **全部 pad 区间**(padded 坐标,
/// 升序、互不重叠 —— 会话账原样存它,`unpad` 用它还原客户端可见序列)。
#[derive(Clone, Debug, Default)]
pub struct PaddedSeq {
    pub ids: Vec<u32>,
    pub spans: Vec<ImgSpan>,
    pub pads: Vec<(usize, usize)>,
}

/// 删除识别命中:要删的是**哪一段**(文本坐标)以及它在 padded 序列里的区间。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextDelete {
    /// padded 删除区间 `[a, b)`(整页;`new[..a] == old[..a]` 由重建保证)
    pub a: usize,
    /// padded 删除终点(不含)
    pub b: usize,
    /// 文本坐标的删除区间(诊断/日志)
    pub text_a: usize,
    pub text_b: usize,
}

/// **多段删除计划**（2026-11-13）：一次请求里删掉多段**不连续**的旧图。
///
/// 与 [`TextDelete`] 的区别：`ranges` 按文本/padded 坐标升序排列，段之间
/// 可以夹着幸存文本。调用方仍需保证每段 `a/b` 页对齐；只有所有旧图都被
/// 删除、且客户端 raw 以“旧文本去掉这些段”为前缀时才返回。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeletionPlan {
    pub ranges: Vec<TextDelete>,
}

impl DeletionPlan {
    pub fn first(&self) -> Option<TextDelete> {
        self.ranges.first().copied()
    }
    pub fn last(&self) -> Option<TextDelete> {
        self.ranges.last().copied()
    }
}

/// **多段 leave-gap 的页映射**（2026-11-13）。
///
/// 删除计划编译成显式页表，后续 hash/table/release/snapshot 都只读这一份，
/// 不再各自用 `survivors/common_pages/used_len` 反推坐标。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GapPageLayout {
    /// 新序列前 `used_pages` 页对应的旧页号（升序、无重复）。
    pub survivors: Vec<usize>,
    /// 新链第一个**需要重算身份**的页 = 第一段删除的页号。
    pub first_changed_page: usize,
    /// 按 token 公共前缀得到的安全复用上限（页）。
    pub target_pages: usize,
    /// 实际复用页数（再受 GDN 快照边界约束）。
    pub used_pages: usize,
    /// GDN 恢复点：旧尾快照，或某个旧边界上的快照。
    pub restore: GapRestore,
}

/// 多段删除后的 GDN 恢复来源。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GapRestore {
    /// 旧链末尾的 Dense 尾锚点（仅当全部幸存页都保留）。
    TailSnapshot,
    /// 旧链某个边界上的快照（含被删页内的边界也可：GDN 态包含“删掉的记忆”）。
    OldBoundary(usize),
}

impl GapPageLayout {
    /// 新链身份可从旧链直接沿用的页数。
    pub fn hash_prefix_pages(&self) -> usize {
        self.used_pages.min(self.first_changed_page)
    }

    /// 需要按新 prompt 重算身份的页区间。
    pub fn hash_recompute_pages(&self) -> std::ops::Range<usize> {
        let from = self.hash_prefix_pages();
        from..self.used_pages
    }
}

/// **编译多段删除计划**（纯函数，无设备/无 store）。
///
/// `snap_pages` = 旧链上有快照的边界页号（升序）；`tail_snap` = 旧链末尾
/// 是否有 Dense 尾锚点。返回 `None` = 没有可用恢复点（调用方退回普通复用/重建）。
///
/// 关键约束：
/// - `used_pages` 永远不超过 `target_pages`（token 公共前缀）；
/// - 尾快照**只在全部幸存页都保留时**可用；否则 GDN 态会包含缓存面之外的 token；
/// - 其余快照按“覆盖的幸存页数”截断 `used_pages`，绝不让恢复点越过缓存面。
pub fn layout_multi_delete(
    old_blocks: usize,
    page: usize,
    ranges: &[TextDelete],
    common_tokens: usize,
    snap_pages: &[usize],
    tail_snap: bool,
) -> Result<Option<GapPageLayout>, String> {
    if page == 0 || old_blocks == 0 || ranges.is_empty() {
        return Ok(None);
    }
    let old_len = old_blocks
        .checked_mul(page)
        .ok_or_else(|| "layout_multi_delete: old_blocks×page 溢出".to_string())?;
    let mut cursor = 0usize;
    for r in ranges {
        if r.a >= r.b || r.a % page != 0 || r.b % page != 0 || r.b > old_len {
            return Err(format!(
                "layout_multi_delete: 删除区间 [{},{}) 非整页/越界(old={old_len}, page={page})",
                r.a, r.b
            ));
        }
        if r.a < cursor {
            return Err(format!(
                "layout_multi_delete: 删除区间 [{},{}) 重叠/未升序(cursor={cursor})",
                r.a, r.b
            ));
        }
        cursor = r.b;
    }
    let deleted: Vec<(usize, usize)> = ranges.iter().map(|r| (r.a / page, r.b / page)).collect();
    let mut survivors = Vec::with_capacity(old_blocks);
    let mut d = 0usize;
    for b in 0..old_blocks {
        while d < deleted.len() && deleted[d].1 <= b {
            d += 1;
        }
        if d < deleted.len() && deleted[d].0 <= b && b < deleted[d].1 {
            continue;
        }
        survivors.push(b);
    }
    if survivors.is_empty() {
        return Ok(None);
    }
    let target_pages = survivors.len().min(common_tokens / page);
    if target_pages == 0 {
        return Ok(None);
    }
    let first_changed_page = deleted[0].0;

    // 尾快照代表“旧链全部幸存页 + 旧生成尾”。只有**最后一个旧页仍幸存**时才
    // 对齐：若尾部页被删，尾快照的 GDN 态包含新 prefix 之外的已删 token。
    if tail_snap && target_pages == survivors.len() && survivors.last() == Some(&(old_blocks - 1)) {
        return Ok(Some(GapPageLayout {
            survivors,
            first_changed_page,
            target_pages,
            used_pages: target_pages,
            restore: GapRestore::TailSnapshot,
        }));
    }

    // 非尾快照必须是**幸存页**上的边界：落到被删页 = GDN 态包含新 prefix
    // 之外的 token（即使它后面没有幸存页，也已经“算过头”）。
    let mut best: Option<(usize, usize)> = None; // (覆盖幸存页数, 旧页号)
    for (idx, &p) in survivors.iter().enumerate().take(target_pages) {
        if snap_pages.binary_search(&p).is_ok() {
            best = Some((idx + 1, p));
        }
    }
    let Some((used_pages, old_page)) = best else {
        return Ok(None);
    };
    Ok(Some(GapPageLayout {
        survivors,
        first_changed_page,
        target_pages,
        used_pages,
        restore: GapRestore::OldBoundary(old_page),
    }))
}

/// 图像占位游程(`[start, end)`,**仅 image token 本身**,不含标记)。
fn image_runs(ids: &[u32], image_token: u32) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < ids.len() {
        if ids[i] == image_token {
            let start = i;
            while i < ids.len() && ids[i] == image_token {
                i += 1;
            }
            out.push((start, i));
        } else {
            i += 1;
        }
    }
    out
}

/// 把图像账截到 padded 前缀 `len`：只保留**完整落在前缀内**的 spans/pads。
///
/// leave-gap 删除会在 turn 中途把会话 `tokens` 截到 cached 前缀；若此时
/// 客户端断流走 Abort，不会再 `commit_turn`。因此落账点必须同时截
/// `img_spans`/`pads`，否则下一轮 `unpad` 会读到超出序列尾的 pad 区间。
pub fn ledger_prefix(
    spans: &[ImgSpan],
    pads: &[(usize, usize)],
    len: usize,
) -> (Vec<ImgSpan>, Vec<(usize, usize)>) {
    let spans = spans.iter().copied().filter(|sp| sp.end <= len).collect();
    let pads = pads.iter().copied().filter(|(_, hi)| *hi <= len).collect();
    (spans, pads)
}

/// 剥掉 pad 区间 → 客户端可见的 token 序列。
///
/// `pads` 必须升序且互不重叠(由 [`pad_image_spans`] 保证);越界 = `Err`
/// (账与 tokens 没对齐是结构错,不许静默截断)。
pub fn unpad(ids: &[u32], pads: &[(usize, usize)]) -> Result<Vec<u32>, String> {
    let mut out = Vec::with_capacity(ids.len());
    let mut cursor = 0usize;
    for &(lo, hi) in pads {
        if lo > hi || hi > ids.len() || lo < cursor {
            return Err(format!(
                "unpad: pad 区间 [{lo}, {hi}) 非法(序列长 {}, 上一 cursor {cursor})",
                ids.len()
            ));
        }
        out.extend_from_slice(&ids[cursor..lo]);
        cursor = hi;
    }
    out.extend_from_slice(&ids[cursor..]);
    Ok(out)
}

/// 把 **unpadded 坐标**下的前缀长度映射回 **padded 序列**的切点。
///
/// `raw_len` = 客户端可见 token 数;返回值 = 覆盖这 `raw_len` 个可见 token
/// 的 padded 前缀长度,**并吞掉紧跟在切点后的 pad 块**(孤儿 pad 必须留下,
/// 否则下一轮从裸文本重算 pad 时它会消失,缓存退到删点之前)。
///
/// `Err` = `raw_len` 超过可见 token 数 / pad 账非法(不静默截断)。
pub fn padded_prefix_for_raw(
    tokens: &[u32],
    pads: &[(usize, usize)],
    raw_len: usize,
) -> Result<usize, String> {
    let mut pad_i = 0usize;
    let mut raw = 0usize;
    let mut i = 0usize;
    while i <= tokens.len() {
        while pad_i < pads.len() && pads[pad_i].0 == i {
            let (lo, hi) = pads[pad_i];
            if lo != i || hi < lo || hi > tokens.len() {
                return Err(format!(
                    "padded_prefix_for_raw: pad 区间 [{lo},{hi}) 非法(tokens={})",
                    tokens.len()
                ));
            }
            i = hi;
            pad_i += 1;
        }
        if raw == raw_len {
            return Ok(i);
        }
        if i >= tokens.len() {
            break;
        }
        i += 1;
        raw += 1;
    }
    Err(format!(
        "padded_prefix_for_raw: raw_len={raw_len} > 可见 token {raw}(padded={})",
        tokens.len()
    ))
}

/// **页对齐 pad 律**(只在 `[vision] image_evict` 且该图带标记时执行)。
///
/// 对每段**可删**图像占位(含 `<|vision_start|>` / `<|vision_end|>` 标记,
/// 若在场):
///
/// ```text
/// pad_before = (page − 段起点 % page) % page
/// pad_after  = (page − (段起点 + pad_before + 段长) % page) % page
/// ```
///
/// `inherited` = 输入序列里**已经存在**的 pad 区间(上一轮留下的;输入坐标)。
/// 它们会被原样搬进输出并映射到输出坐标(会话账要能一路追下去)。
/// **已补过的图不会二次补 pad**:before 块以图首为尾、after 块以图尾为首,
/// 命中即只记 `pad_before/pad_after` 账,不发射 token(重建路径的前缀
/// 逐字节不变靠这一条;见 `delete_middle_image_keeps_padded_prefix`)。
///
/// 返回 [`PaddedSeq`]。`marks.len()` 必须 = 游程数(调用方按占位顺序对齐;
/// 不符 = 响亮的 `Err`,不许静默错位)。未标记的段**一个 pad 都不加**。
// 结构性豁免(2026-11-02 clippy 清理):派发/契约 ABI:参数表就是调用面,拆结构体会把每个调用点变成二次装箱
#[allow(clippy::too_many_arguments)]
pub fn pad_image_spans(
    ids: &[u32],
    marks: &[bool],
    grids: &[GridThw],
    image_token: u32,
    vision_start: Option<u32>,
    vision_end: Option<u32>,
    page: usize,
    pad_id: u32,
    inherited: &[(usize, usize)],
) -> Result<PaddedSeq, String> {
    let runs = image_runs(ids, image_token);
    if runs.len() != marks.len() || runs.len() != grids.len() {
        return Err(format!(
            "pad_image_spans: 占位游程 {} 段 / 标记 {} 个 / 网格 {} 个 —— 三者必须相等",
            runs.len(),
            marks.len(),
            grids.len()
        ));
    }
    let mut out: Vec<u32> = Vec::with_capacity(ids.len());
    let mut spans: Vec<ImgSpan> = Vec::with_capacity(runs.len());
    let mut pads: Vec<(usize, usize)> = Vec::new();
    // 已搬迁的 inherited 个数(区间升序 ⇒ 两指针扫)
    let mut inh = 0usize;
    let mut cursor = 0usize;
    // 把一个输入区间 [lo,hi) 原样搬到输出(顺带搬 inherited 里的 pad 区间)
    // 把一个输入区间 [from, to) 原样搬到输出(顺带把 inherited pad 区间映射过去)。
    // `lo < from` = 区间跨了拷贝边界(账不平)⇒ **响亮拒**,不许静默丢 pad
    // (丢一个 pad ⇒ 下一轮 `unpad` 与 tokens 对不上 ⇒ 静默错值)。
    let copy_upto = |out: &mut Vec<u32>,
                     pads: &mut Vec<(usize, usize)>,
                     inh: &mut usize,
                     from: usize,
                     to: usize|
     -> Result<(), String> {
        let base = out.len();
        out.extend_from_slice(&ids[from..to]);
        while *inh < inherited.len() && inherited[*inh].1 <= to {
            let (lo, hi) = inherited[*inh];
            if lo < from {
                return Err(format!(
                    "pad_image_spans: inherited pad [{lo}, {hi}) 跨拷贝边界 [from={from})"
                ));
            }
            pads.push((base + (lo - from), base + (hi - from)));
            *inh += 1;
        }
        Ok(())
    };
    for (k, &(rs, re)) in runs.iter().enumerate() {
        // 段 = [vision_start?][image×n][vision_end?]
        let start = match vision_start {
            Some(t) if rs > 0 && ids[rs - 1] == t => rs - 1,
            _ => rs,
        };
        let end = match vision_end {
            Some(t) if re < ids.len() && ids[re] == t => re + 1,
            _ => re,
        };
        copy_upto(&mut out, &mut pads, &mut inh, cursor, start)?;
        // 重建路径会把**删点之前**的旧 padded 前缀原样带进来；那些已经
        // 补过 pad 的图不能在新一轮 pad 律里再补一份（否则前缀被推歪，
        // 调度面会因 `old[..a] != new[..a]` 拒吸收）。`inherited` 里的
        // pad 块就是“已补过”的权威事实：before 块以图首为尾，after 块
        // 以图尾为首。命中则只记账、不发射 token；旧 pad 随后由
        // `copy_upto` 原样搬进输出。
        let inherited_before = inherited
            .iter()
            .find(|(_, hi)| *hi == start)
            .map(|(lo, hi)| hi - lo);
        let inherited_after = inherited
            .iter()
            .find(|(lo, _)| *lo == end)
            .map(|(lo, hi)| hi - lo);
        let before_pad_lo = out.len();
        let pad_before = if let Some(n) = inherited_before {
            n
        } else if marks[k] && page > 0 {
            let p = (page - out.len() % page) % page;
            out.extend(std::iter::repeat_n(pad_id, p));
            p
        } else {
            0
        };
        // 删除起点 = pad 块**之后**(页对齐);before-pad 留在前缀里(它被重建保留)
        let span_start = out.len();
        out.extend_from_slice(&ids[start..end]);
        let text_end = out.len();
        let pad_after = if let Some(n) = inherited_after {
            // 不发射 token；旧 pad 仍留在 `ids[end..]`，由后续 copy_upto 搬运。
            n
        } else if marks[k] && page > 0 {
            let p = (page - out.len() % page) % page;
            out.extend(std::iter::repeat_n(pad_id, p));
            p
        } else {
            0
        };
        // 新补分支：out 已经含 pad；继承分支：旧 pad 还在 `ids[end..]`，
        // 由后续 copy_upto 搬入，span_end 先按输出坐标记到它之后。
        let span_end = if inherited_after.is_some() {
            out.len() + pad_after
        } else {
            out.len()
        };
        // `pads` 由 copy_upto 在搬旧 pad 时登记；这里只为新补的 pad 登记。
        if inherited_after.is_none() && pad_after > 0 {
            pads.push((text_end, span_end));
        }
        if inherited_before.is_none() && pad_before > 0 {
            pads.push((before_pad_lo, span_start));
        }
        // 文本坐标(客户端可见序列)= 输入坐标剥掉 inherited pad 之后的下标
        let text_idx = |i: usize| -> usize {
            i - inherited
                .iter()
                .filter(|(_, hi)| *hi <= i)
                .map(|(lo, hi)| hi - lo)
                .sum::<usize>()
        };
        spans.push(ImgSpan {
            start: span_start,
            end: span_end,
            text_start: text_idx(start),
            text_end: text_idx(end),
            pad_before,
            pad_after,
            grid: grids[k],
            evict: marks[k],
            img_tag: 0, // 由调用方按图像像素盖章(本函数只有网格/标记,没有像素)
        });
        cursor = end;
    }
    copy_upto(&mut out, &mut pads, &mut inh, cursor, ids.len())?;
    if inh < inherited.len() {
        return Err(format!(
            "pad_image_spans: inherited pad 区间有 {} 个越过序列尾(账不平)",
            inherited.len() - inh
        ));
    }
    pads.sort_by_key(|p| p.0);
    Ok(PaddedSeq {
        ids: out,
        spans,
        pads,
    })
}

/// **删除识别(文本坐标)**:客户端这一轮是不是"正好删了一段已标记图像"。
///
/// 判据:
/// 1. `old_text[..ta] == raw[..ta]`(删点之前的文本逐 token 相同);
/// 2. `old_text[tb..]` 与 `raw[ta..]` 至少有 **1 个 token** 相同(后缀可能
///    很长;分叉点由调用方按 padded 序列精确算)。
///
/// **旧账兼容口**（`img_tag == 0` / 无指纹）：相邻已标记图像段先合并，
/// 只能识别“整组一起删”。新账请走 [`detect_text_delete_tagged`]。
/// 返回命中的文本区间 + 对应的 padded 删除区间。
pub fn detect_text_delete(old_text: &[u32], raw: &[u32], spans: &[ImgSpan]) -> Option<TextDelete> {
    detect_text_delete_inner(old_text, raw, spans, None)
}

/// **带内容指纹的删除识别**（2026-11-13）。
///
/// 相邻的已标记图像在文本坐标里首尾相接；旧实现把整段相邻游程合并成一个
/// 候选，只能识别“整组一起删”。真实 agent 形态是**只删其中第一张、保留
/// 第二张**（截一张新图替换旧图）——合并候选的后缀从第二张之后开始，与
/// 客户端 raw 的第二张对不上，`k=0` ⇒ 整轮只命中图片前的 26k。
///
/// 本函数遍历游程内**所有连续子段**，并用新旧图像内容指纹确认“哪些旧图
/// 真的不在了”：`new_tags` 必须以 `old_tags` 删除该子段后的序列为前缀
/// （尾部允许追加新图）。token 占位符相同但内容不同时，只有指纹能区分
/// 删的是第一张还是第二张，避免把旧图 KV 错绑到新图上。
pub fn detect_text_delete_tagged(
    old_text: &[u32],
    raw: &[u32],
    spans: &[ImgSpan],
    new_tags: &[u64],
) -> Option<TextDelete> {
    detect_text_delete_inner(old_text, raw, spans, Some(new_tags))
}

/// **多段删除识别**（仅覆盖“所有旧图都被删”的形态）。
///
/// 真实 agent 会一次替换整组截图：旧图可能分散在多个 tool result 里，
/// 新图追加在末尾。单段 `detect_text_delete_tagged` 无法把这种形态拼成
/// 一个连续区间；这里直接删除**全部旧图段**，保留夹在中间的所有文本，
/// 并验证客户端 raw 的前缀 = “旧文本去掉这些段”。
///
/// 要求：
/// - 旧 spans 全部带 `evict`（有一张未标记就不吸收）；
/// - 新图集合里没有任何旧图内容（同内容新图也保守放弃）；
/// - 至少两段（单段仍走原路径，保留旧 pad/快照逻辑）。
pub fn detect_all_images_deleted_tagged(
    old_text: &[u32],
    raw: &[u32],
    spans: &[ImgSpan],
    new_tags: &[u64],
) -> Option<DeletionPlan> {
    owl_shared::log_debug!(
        "img-evict",
        "multi try: spans={} evict={:?} tags={:?} new={:?} old={} raw={}",
        spans.len(),
        spans.iter().map(|s| s.evict).collect::<Vec<_>>(),
        spans.iter().map(|s| s.img_tag).collect::<Vec<_>>(),
        new_tags,
        old_text.len(),
        raw.len()
    );
    if spans.len() < 2 || spans.iter().any(|s| !s.evict || s.img_tag == 0) {
        return None;
    }
    if spans
        .iter()
        .any(|s| new_tags.iter().any(|t| *t == s.img_tag))
    {
        return None;
    }
    let mut ranges: Vec<TextDelete> = spans
        .iter()
        .map(|s| TextDelete {
            a: s.start,
            b: s.end,
            text_a: s.text_start,
            text_b: s.text_end,
        })
        .collect();
    ranges.sort_by_key(|r| r.text_a);
    if ranges.len() < 2 || ranges.iter().any(|r| r.text_a >= r.text_b) {
        return None;
    }
    // 去掉所有旧图后的客户端文本；raw 必须以其为前缀（后面才是新尾巴）。
    let mut survivors = Vec::with_capacity(old_text.len());
    let mut cursor = 0usize;
    for r in &ranges {
        if r.text_a < cursor || r.text_b > old_text.len() {
            return None;
        }
        survivors.extend_from_slice(&old_text[cursor..r.text_a]);
        cursor = r.text_b;
    }
    survivors.extend_from_slice(&old_text[cursor..]);
    let common = raw
        .iter()
        .zip(&survivors)
        .take_while(|(a, b)| a == b)
        .count();
    // raw 至少要包含最后一段删除之前的幸存前缀；后面可以接新图/新尾巴，
    // 旧 session 的生成尾巴也可以不在新 prompt 里（common 会提前停）。
    let last_b = ranges.last().map(|r| r.text_b).unwrap_or(0);
    let after_last = old_text.len().saturating_sub(last_b);
    let required = survivors.len().saturating_sub(after_last);
    owl_shared::log_debug!(
        "img-evict",
        "multi calc: common={} required={} survivors={} last_b={} ranges={:?}",
        common,
        required,
        survivors.len(),
        last_b,
        ranges
            .iter()
            .map(|r| (r.text_a, r.text_b, r.a, r.b))
            .collect::<Vec<_>>()
    );
    if common < required || common == 0 {
        return None;
    }
    Some(DeletionPlan { ranges })
}

fn detect_text_delete_inner(
    old_text: &[u32],
    raw: &[u32],
    spans: &[ImgSpan],
    new_tags: Option<&[u64]>,
) -> Option<TextDelete> {
    // 指纹不可用（旧账 tag=0 / 未传）时保持旧行为：只试完整相邻游程，
    // 宁可少吸收，也不做 token 盲猜。
    let use_tags = new_tags.is_some() && !spans.is_empty() && spans.iter().all(|s| s.img_tag != 0);
    let old_tags: Vec<u64> = spans.iter().map(|s| s.img_tag).collect();
    // 相邻已标记图像 = 一个游程（下标闭开区间）。
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for (i, s) in spans.iter().enumerate() {
        if !s.evict {
            continue;
        }
        match runs.last_mut() {
            Some((_, hi)) if spans[*hi - 1].text_end == s.text_start => *hi = i + 1,
            _ => runs.push((i, i + 1)),
        }
    }
    let mut best: Option<(usize, usize, TextDelete)> = None; // (suffix_k, deleted_bytes, hit)
    for (lo, hi) in runs {
        let starts: Vec<usize> = if use_tags {
            (lo..hi).collect()
        } else {
            vec![lo]
        };
        for i in starts {
            let ends: Vec<usize> = if use_tags {
                (i + 1..=hi).collect()
            } else {
                vec![hi]
            };
            for j in ends {
                let (ta, tb) = (spans[i].text_start, spans[j - 1].text_end);
                if tb > old_text.len() || ta >= tb || ta > old_text.len() || ta > raw.len() {
                    continue;
                }
                if old_text[..ta] != raw[..ta] {
                    continue;
                }
                let k = old_text[tb..]
                    .iter()
                    .zip(&raw[ta..])
                    .take_while(|(x, y)| x == y)
                    .count();
                if k == 0 {
                    continue;
                }
                if use_tags {
                    let nt = new_tags.expect("use_tags ⇒ Some");
                    let mut expected = Vec::with_capacity(old_tags.len() - (j - i));
                    expected.extend_from_slice(&old_tags[..i]);
                    expected.extend_from_slice(&old_tags[j..]);
                    if !nt.starts_with(&expected) {
                        continue;
                    }
                }
                let delete_bytes = spans[j - 1].end - spans[i].start;
                let cand = TextDelete {
                    a: spans[i].start,
                    b: spans[j - 1].end,
                    text_a: ta,
                    text_b: tb,
                };
                let better = match best {
                    None => true,
                    Some((bk, bd, _)) => k > bk || (k == bk && delete_bytes > bd),
                };
                if better {
                    best = Some((k, delete_bytes, cand));
                }
            }
        }
    }
    best.map(|(_, _, hit)| hit)
}

/// **重建**:`old[..a]`(含旧 pad 块)+ 客户端删点之后的文本。
///
/// 这是"删点之前逐字节不变"的**构造性保证** —— 纯 diff 做不到(见模块头注)。
/// `hit.a` 必须落页边界(pad 律保证),否则 `Err`。
pub fn rebuild_after_delete(
    old_padded: &[u32],
    raw: &[u32],
    hit: &TextDelete,
) -> Result<Vec<u32>, String> {
    // 页对齐由 pad 律保证(调用方用真实 `page` 校验一次);这里只防越界。
    if hit.a > old_padded.len() || hit.text_a > raw.len() {
        return Err(format!(
            "rebuild_after_delete: 越界(a={} old={} ta={} raw={})",
            hit.a,
            old_padded.len(),
            hit.text_a,
            raw.len()
        ));
    }
    let mut out = Vec::with_capacity(hit.a + raw.len() - hit.text_a);
    out.extend_from_slice(&old_padded[..hit.a]);
    out.extend_from_slice(&raw[hit.text_a..]);
    Ok(out)
}

/// **逐块多模态指纹**(`owl_store::BlockTag` 的唯一产地)。
///
/// 块身份只看 token ⇒ 两张不同截图(占位符字面相同)会被判成同一条链 ⇒ 复用
/// 陈旧图像 KV(2026-11-03 定谳)。这里把每张图的 `img_tag` 铺到它覆盖的块上,
/// 同块多图用**可交换折叠**(与 span 顺序无关,幂等)。
///
/// 返回空 = 全是纯文本块(调用方传 `&[]`,`hash_block` 走旧身份,零回归)。
pub fn block_tags(token_len: usize, page: usize, spans: &[ImgSpan]) -> Vec<u64> {
    if page == 0 || spans.iter().all(|s| s.img_tag == 0) {
        return Vec::new();
    }
    let n = token_len / page;
    if n == 0 {
        return Vec::new();
    }
    let mut out = vec![0u64; n];
    for s in spans {
        if s.img_tag == 0 {
            continue;
        }
        let lo = s.start / page;
        let hi = s.end.saturating_sub(1) / page;
        for slot in out.iter_mut().take((hi + 1).min(n)).skip(lo) {
            *slot = fold_tag(*slot, s.img_tag);
        }
    }
    out
}

/// 同块多图的**可交换**折叠(不依赖 span 顺序;0 是单位元)。
fn fold_tag(acc: u64, tag: u64) -> u64 {
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    (acc ^ tag).wrapping_mul(PRIME).rotate_left(7)
}

/// **带位置签名的块指纹**（2026-11-13）。
///
/// 前缀链的内容身份原先只有 token + 图像指纹；两条 token 完全相同、但
/// position 基准不同的链（leave-gap 后的非紧凑序列 vs 普通紧凑序列）会
/// 撞成同一个 `ChainHash` ⇒ 跨会话复用错 K。这里把**非紧凑位置**折进
/// 每个整块的 tag：
///
/// - 位置紧凑（`p[i] == i`）⇒ 返回原 [`block_tags`]，老链身份逐字节不变；
/// - 位置非紧凑 ⇒ 每块 tag 混入该块内逐 token 的 `(index, [T,H,W])`；
///   父链哈希已包含前块签名，所以任意前缀位置不同都会断开命中。
///
/// 调用方必须对**同一条链**使用同一份 positions（submit 的 `turn.positions`、
/// 会话账的 `s.positions`、活跃 turn 的 `act.positions`）。
pub fn block_tags_pos(
    token_len: usize,
    page: usize,
    spans: &[ImgSpan],
    positions: &[Pos3],
) -> Vec<u64> {
    let mut tags = block_tags(token_len, page, spans);
    let n = if page == 0 { 0 } else { token_len / page };
    if n == 0 || positions.len() < n * page {
        return tags;
    }
    let positioned = positions
        .iter()
        .take(n * page)
        .enumerate()
        .any(|(i, p)| *p != [i as i32; 3]);
    if !positioned {
        return tags;
    }
    if tags.is_empty() {
        tags.resize(n, 0);
    }
    for (b, slot) in tags.iter_mut().enumerate().take(n) {
        let mut acc = *slot;
        let mut block_positioned = false;
        for (off, p) in positions[b * page..(b + 1) * page].iter().enumerate() {
            let idx = b * page + off;
            // 只把**本块内确实离开紧凑坐标**的 token 折进去：洞之前的块
            // 保持旧身份（共享前缀不因后面的洞而失效）。
            if *p == [idx as i32; 3] {
                continue;
            }
            block_positioned = true;
            let sig = (p[0] as u32 as u64)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .rotate_left(17)
                ^ (p[1] as u32 as u64)
                    .wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
                    .rotate_left(31)
                ^ (p[2] as u32 as u64)
                    .wrapping_mul(0x1656_67B1_9E37_79F9)
                    .rotate_left(47)
                ^ (idx as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93)
                | 1;
            acc = fold_tag(acc, sig);
        }
        if block_positioned {
            *slot = acc;
        }
    }
    tags
}

/// **可删图前的快照锚点**（2026-11-13，纯函数）。
///
/// 返回**每一张 `evict` 标记图段起点**的页边界（升序、去重）。
///
/// # 为什么需要它
///
/// 真实形态“截图在 prompt 尾部”下，预填充末快照的键落在**被删页**上
/// （`prompt_end` 的最后一个整块就是那页截图）⇒ `layout_multi_delete` 与
/// 单段路径都会安全拒绝尾快照，幸存页上又没有边界快照 ⇒ 整段退回 fresh。
/// 在每张可删图的段起点拍一张快照，删除发生时就能从**被删图之前**
/// 的那个 GDN 态续跑；同一游程里只删后几张时，也能复用到前几张的页。
///
/// # 语义与成本
///
/// - 锚点只对 `evict` 标记的图生效 —— 未标记的图是引擎“不得自作主张”的红线；
/// - 页边界由 pad 律保证（`span.start` 已页对齐）；`0` 与 `>= prompt_len`
///   的点没有可复用前缀 / 不在链内，直接丢弃；
/// - 稳态成本 ≈ **本轮新进 prompt 的可删图数**：旧图的锚点键是内容身份，
///   前缀命中时 prefill 不会跨过那些 offset ⇒ 同键 Touch，不占新槽；
/// - 设备/宿主/盘的快照预算仍由 `admit_snap`（`snap_max`/`snap_session_cap`/
///   `SnapRole`）兜底 —— 满了就跳过，不会静默错值。
pub fn evict_snap_points(spans: &[ImgSpan], prompt_len: usize, page: usize) -> Vec<usize> {
    if page == 0 || spans.is_empty() || prompt_len <= page {
        return Vec::new();
    }
    let mut pts: Vec<usize> = spans
        .iter()
        .filter(|sp| sp.evict)
        .map(|sp| sp.start / page * page)
        .filter(|&a| a > 0 && a < prompt_len)
        .collect();
    pts.sort_unstable();
    pts.dedup();
    pts
}

/// **模板 trim 边界复原**（2026-11-15，纯函数）。
///
/// # 病灶（真机定谳）
///
/// Qwen 系快照模板对消息内容做 `|trim`（`chat_template.jinja`）；服务端给
/// 多段 content 的 text part 补的 `\n` 只在**内部 part 边界**幸存。于是
/// “截图在消息尾部”的形态里：
/// - turn N：text part 被补 `…。\n`，随后才是图片；
/// - turn N+1：图片被删、同一段文本退化成单一 content ⇒ `|trim` 把 `\n`
///   剪掉 ⇒ raw 在删点前**少一个空白 token**，`common = ta-1`，
///   删除识别/锚点全部失配。
///
/// 这里在识别之前把旧前缀里的这段空白**插回 raw**（只允许纯空白、只发生在
/// 可删图段的起点之前、且跳过空白后的下一个 token 必须与旧幸存序列一致）。
/// raw 因此回到规范形式，后续识别/重建/位置账全部走原路径。
///
/// `is_ws` = 单 token 是否**可见空白**（特殊 token 解码为空 ⇒ 必须返回 false）。
/// 返回插入的 token 数；`None` = 不需要/不能安全复原。
pub fn restore_trimmed_prefix(
    old_text: &[u32],
    input: &mut Vec<u32>,
    spans: &[ImgSpan],
    is_ws: impl Fn(u32) -> bool,
) -> Option<usize> {
    const MAX_TRIM: usize = 8;
    if spans.is_empty() || old_text.is_empty() || input.is_empty() {
        return None;
    }
    let n = old_text.len().min(input.len());
    let mut c = 0usize;
    while c < n && old_text[c] == input[c] {
        c += 1;
    }
    if c == 0 || c == n {
        return None; // 无分歧 / 一方是另一方前缀
    }
    // 候选 1：全删（旧文本去掉**所有**可删段）。
    let all_deleted: Option<Vec<u32>> = {
        let mut out: Vec<u32> = Vec::with_capacity(old_text.len());
        let mut cursor = 0usize;
        let mut ok = true;
        for sp in spans.iter().filter(|sp| sp.evict) {
            let (lo, hi) = (sp.text_start, sp.text_end);
            if lo < cursor || hi > old_text.len() || lo >= hi {
                ok = false;
                break;
            }
            out.extend_from_slice(&old_text[cursor..lo]);
            cursor = hi;
        }
        if ok {
            out.extend_from_slice(&old_text[cursor..]);
            Some(out)
        } else {
            None
        }
    };
    for sp in spans.iter().filter(|sp| sp.evict) {
        let ta = sp.text_start;
        if ta <= c || ta - c > MAX_TRIM || ta > old_text.len() {
            continue;
        }
        if !old_text[c..ta].iter().all(|&t| is_ws(t)) {
            continue;
        }
        let k = ta - c;
        // 单删这一段 / 全删：跳过空白后的下一个 token 必须与旧幸存序列一致。
        let single_match = sp.text_end <= old_text.len()
            && old_text[sp.text_end..]
                .first()
                .zip(input.get(c))
                .is_some_and(|(a, b)| a == b);
        let all_match = all_deleted
            .as_ref()
            .and_then(|p| p.get(c + k).zip(input.get(c)))
            .is_some_and(|(a, b)| a == b);
        if !single_match && !all_match {
            continue;
        }
        let inserted: Vec<u32> = old_text[c..ta].to_vec();
        input.splice(c..c, inserted.iter().copied());
        return Some(k);
    }
    None
}

/// **紧凑位置账**:从 token 0 起按 m-RoPE 重新计算(普通 turn / 整段重建)。
pub fn compact_positions(
    ids: &[u32],
    spans: &[ImgSpan],
    image_token: u32,
    mrope: &MropeCfg,
) -> Result<(Vec<Pos3>, i32), String> {
    if ids.is_empty() {
        return Err("compact_positions: 空序列".to_string());
    }
    let tt = owl_models::layers::mrope::token_types(ids, image_token, None);
    let grids: Vec<GridThw> = spans.iter().map(|s| s.grid).collect();
    let index = mrope
        .rope_index(ids, &tt, &grids, &[])
        .map_err(|e| format!("compact_positions: {e}"))?;
    let pos = index
        .pos
        .into_iter()
        .map(|p| {
            Ok([
                i32::try_from(p[0]).map_err(|_| "position T 越 i32".to_string())?,
                i32::try_from(p[1]).map_err(|_| "position H 越 i32".to_string())?,
                i32::try_from(p[2]).map_err(|_| "position W 越 i32".to_string())?,
            ])
        })
        .collect::<Result<Vec<_>, String>>()?;
    let max_pos = pos
        .iter()
        .flat_map(|p| p.iter())
        .copied()
        .max()
        .unwrap_or(0);
    Ok((pos, max_pos))
}

/// **删除/追加后的位置账**(leave-gap 的唯一产地)。
///
/// - `old_tokens` / `old_pos` = 会话上一版 padded 序列与逐 token 位置;
/// - `new_ids` / `spans` = 本轮重建后的 padded 序列与图像段账;
/// - `deletion = Some(hit)` 时按 `[a,b)` 删一段:前缀 `[..a)` 与后缀
///   `[b..b+k)` 的 position **原样保留**,新尾巴从 `max_pos+1` 追加;
/// - `deletion = None` 时按普通追加:公共前缀保留旧 position,尾巴从
///   `max_pos+1` 追加(leave-gap 会话跨 turn 继续留洞)。
///
/// 返回 `(positions, max_pos, positioned)`;`positioned=true` 表示位置已
/// 离开紧凑坐标,调用方须禁用跨会话内容缓存。
pub fn positions_after_splice(
    old_tokens: &[u32],
    old_pos: &[Pos3],
    new_ids: &[u32],
    spans: &[ImgSpan],
    deletion: Option<TextDelete>,
    image_token: u32,
    mrope: &MropeCfg,
) -> Result<(Vec<Pos3>, i32, bool), String> {
    if old_tokens.len() != old_pos.len() {
        return Err(format!(
            "positions_after_splice: 旧 token {} ≠ 旧位置 {}",
            old_tokens.len(),
            old_pos.len()
        ));
    }
    if new_ids.is_empty() {
        return Err("positions_after_splice: 新序列为空".to_string());
    }
    let mut pos: Vec<Pos3> = Vec::with_capacity(new_ids.len());
    let tail_start;
    if let Some(del) = deletion {
        let (a, b) = (del.a, del.b);
        if a > b || b > old_tokens.len() || a > new_ids.len() {
            return Err(format!(
                "positions_after_splice: 删除区间 [{a},{b}) 越界(旧 {} / 新 {})",
                old_tokens.len(),
                new_ids.len()
            ));
        }
        if old_tokens[..a] != new_ids[..a] {
            return Err("positions_after_splice: 删点之前 token 不一致".to_string());
        }
        pos.extend_from_slice(&old_pos[..a]);
        let k = old_tokens[b..]
            .iter()
            .zip(&new_ids[a..])
            .take_while(|(x, y)| x == y)
            .count();
        pos.extend_from_slice(&old_pos[b..b + k]);
        tail_start = a + k;
    } else {
        let k = old_tokens
            .iter()
            .zip(new_ids)
            .take_while(|(x, y)| x == y)
            .count();
        pos.extend_from_slice(&old_pos[..k]);
        tail_start = k;
    }
    // 位置账只能从旧账**连续**拼;跨 tail_start 的图像段说明删除点落在图内,
    // 这种形状不允许(页对齐 pad 律保证删点在图段边界)。
    if spans
        .iter()
        .any(|s| s.start < tail_start && s.end > tail_start)
    {
        return Err(format!(
            "positions_after_splice: 图像段跨越尾巴起点 {tail_start}(spans={:?})",
            spans.iter().map(|s| (s.start, s.end)).collect::<Vec<_>>()
        ));
    }
    // **紧凑文本的截断/改写 = 无洞**（2026-11-06 双并发门定谳）：
    // 纯文本会话的位置就是下标；客户端把旧尾巴截短（最常见：不带上一轮生成段
    // 重发前缀）时，正确做法是让新尾巴从**分歧点** `tail_start` 续写，而不是从
    // 旧 `max+1` 起（后者会凭空造一个位置洞 ⇒ 会话被标 `positioned` ⇒
    // 跨会话复用被永久关闭，真机表现为"明明有 KV 却命中 0"）。
    // mRoPE/图像会话（old_pos 非紧凑）保持旧语义：分歧/删除 = 真位置洞。
    let compact_old = old_pos
        .iter()
        .enumerate()
        .all(|(i, p)| p.iter().all(|v| *v == i as i32));
    let tail = &new_ids[tail_start..];
    if !tail.is_empty() {
        let grids: Vec<GridThw> = spans
            .iter()
            .filter(|s| s.start >= tail_start)
            .map(|s| s.grid)
            .collect();
        let tt = owl_models::layers::mrope::token_types(tail, image_token, None);
        let index = mrope
            .rope_index(tail, &tt, &grids, &[])
            .map_err(|e| format!("positions_after_splice: 新尾巴 rope_index: {e}"))?;
        let base: i32 = if deletion.is_none() && compact_old {
            i32::try_from(tail_start).map_err(|_| "positions_after_splice: tail_start 越 i32")?
        } else {
            old_pos
                .iter()
                .flat_map(|p| p.iter())
                .copied()
                .max()
                .unwrap_or(-1)
                .saturating_add(1)
        };
        for p in index.pos {
            let add = |v: i64| -> Result<i32, String> {
                i32::try_from(v)
                    .ok()
                    .and_then(|x| x.checked_add(base))
                    .ok_or_else(|| format!("positions_after_splice: 位置 {v}+{base} 越 i32"))
            };
            pos.push([add(p[0])?, add(p[1])?, add(p[2])?]);
        }
    }
    if pos.len() != new_ids.len() {
        return Err(format!(
            "positions_after_splice: 位置 {} ≠ 新序列 {}",
            pos.len(),
            new_ids.len()
        ));
    }
    let max_pos = pos
        .iter()
        .flat_map(|p| p.iter())
        .copied()
        .max()
        .unwrap_or(0);
    // 洞的判据 = 删除（leave-gap）**或** 非紧凑序列上的分歧；紧凑文本的分歧
    // 已按 `tail_start` 连续续写，不再产生洞。
    let gap = deletion.is_some() || (tail_start < old_tokens.len() && !compact_old);
    Ok((pos, max_pos, gap))
}

/// **多段删除后的位置账**（2026-11-13）：保留所有幸存段的旧 position，
/// 新尾巴从旧 `max_pos+1` 追加。要求新 prompt 前缀 = 旧序列去掉所有
/// `plan.ranges` 后的幸存序列（多段全部页对齐、段间文本逐 token 不变）。
pub fn positions_after_deletes(
    old_tokens: &[u32],
    old_pos: &[Pos3],
    new_ids: &[u32],
    spans: &[ImgSpan],
    plan: &DeletionPlan,
    image_token: u32,
    mrope: &MropeCfg,
) -> Result<(Vec<Pos3>, i32, bool), String> {
    if old_tokens.len() != old_pos.len() {
        return Err(format!(
            "positions_after_deletes: 旧 token {} ≠ 旧位置 {}",
            old_tokens.len(),
            old_pos.len()
        ));
    }
    if new_ids.is_empty() {
        return Err("positions_after_deletes: 新序列为空".to_string());
    }
    let mut surv_tokens: Vec<u32> = Vec::with_capacity(old_tokens.len());
    let mut surv_pos: Vec<Pos3> = Vec::with_capacity(old_pos.len());
    let mut cursor = 0usize;
    for r in &plan.ranges {
        if r.a < cursor || r.a > r.b || r.b > old_tokens.len() {
            return Err(format!(
                "positions_after_deletes: 删除区间 [{},{}) 重叠/越界(旧 {} / cursor {})",
                r.a,
                r.b,
                old_tokens.len(),
                cursor
            ));
        }
        surv_tokens.extend_from_slice(&old_tokens[cursor..r.a]);
        surv_pos.extend_from_slice(&old_pos[cursor..r.a]);
        cursor = r.b;
    }
    surv_tokens.extend_from_slice(&old_tokens[cursor..]);
    surv_pos.extend_from_slice(&old_pos[cursor..]);
    let k = surv_tokens
        .iter()
        .zip(new_ids)
        .take_while(|(a, b)| a == b)
        .count();
    if k == 0 {
        return Err("positions_after_deletes: 新 prompt 与幸存序列无公共前缀".to_string());
    }
    // `k` 可以小于幸存序列（旧生成尾巴/客户端新尾巴在分叉点）：只保留
    // k 之前的旧 position，之后按新尾巴从 max+1 重算。
    let mut pos = surv_pos[..k].to_vec();
    let tail = &new_ids[k..];
    if !tail.is_empty() {
        let grids: Vec<GridThw> = spans
            .iter()
            .filter(|s| s.start >= k)
            .map(|s| s.grid)
            .collect();
        let tt = owl_models::layers::mrope::token_types(tail, image_token, None);
        let index = mrope
            .rope_index(tail, &tt, &grids, &[])
            .map_err(|e| format!("positions_after_deletes: 新尾巴 rope_index: {e}"))?;
        let base = old_pos
            .iter()
            .flat_map(|p| p.iter())
            .copied()
            .max()
            .unwrap_or(-1)
            .saturating_add(1);
        for p in index.pos {
            let add = |v: i64| -> Result<i32, String> {
                i32::try_from(v)
                    .ok()
                    .and_then(|x| x.checked_add(base))
                    .ok_or_else(|| format!("positions_after_deletes: 位置 {v}+{base} 越 i32"))
            };
            pos.push([add(p[0])?, add(p[1])?, add(p[2])?]);
        }
    }
    if pos.len() != new_ids.len() {
        return Err(format!(
            "positions_after_deletes: 位置 {} ≠ 新序列 {}",
            pos.len(),
            new_ids.len()
        ));
    }
    let max_pos = pos
        .iter()
        .flat_map(|p| p.iter())
        .copied()
        .max()
        .unwrap_or(0);
    Ok((pos, max_pos, true))
}

#[cfg(test)]
mod tests {
    use super::*;

    const IMG: u32 = 1000;
    const VS: u32 = 1001;
    const VE: u32 = 1002;
    const PAD: u32 = 9;
    const PAGE: usize = 32;

    fn grid() -> GridThw {
        (1, 16, 16) // h/merge = 8 ⇒ 位置推进 8;占位 token = 8·8 = 64
    }

    /// 拼一条图文交错序列:`text(n) [vs img×64 ve] text(n) ...`
    fn build(images: &[usize], text_between: usize) -> Vec<u32> {
        let mut v = Vec::new();
        for (k, &n) in images.iter().enumerate() {
            v.extend((0..text_between).map(|i| (k * 100 + i) as u32 % 900));
            if n > 0 {
                v.push(VS);
                v.extend(std::iter::repeat_n(IMG, 64));
                v.push(VE);
            }
        }
        v.extend((0..text_between).map(|i| (500 + i) as u32));
        v
    }

    fn pad(ids: &[u32], marks: &[bool], grids: &[GridThw]) -> PaddedSeq {
        pad_image_spans(ids, marks, grids, IMG, Some(VS), Some(VE), PAGE, PAD, &[]).expect("pad")
    }

    #[test]
    fn pad_law_aligns_every_marked_item() {
        let ids = build(&[1, 1], 5);
        let seq = pad(&ids, &[true, true], &[grid(), grid()]);
        assert_eq!(seq.spans.len(), 2);
        for (i, s) in seq.spans.iter().enumerate() {
            assert_eq!(s.start % PAGE, 0, "第 {i} 段起点必须页对齐");
            assert_eq!((s.end - s.start) % PAGE, 0, "第 {i} 段长必须整页");
            assert_eq!(s.end - s.start, 96, "段 = 1 + 64 + 1 = 66 ⇒ pad 到 96");
            assert_eq!(s.text_end - s.text_start, 66, "文本坐标段长 = 66");
        }
        assert_eq!(seq.pads.len(), 4, "两段各两个 pad 块");
        // pad 是 pad_id,标记与占位原样
        assert_eq!(seq.ids.iter().filter(|t| **t == PAD).count(), 114);
        assert_eq!(seq.ids.iter().filter(|t| **t == IMG).count(), 128);
    }

    #[test]
    fn unmarked_image_gets_no_pad_and_unpad_round_trips() {
        let ids = build(&[1, 1], 5);
        let seq = pad(&ids, &[false, true], &[grid(), grid()]);
        assert_eq!(
            seq.spans[0].start, seq.spans[0].text_start,
            "未标记段不 pad"
        );
        assert_eq!(
            seq.ids[..seq.spans[0].text_start],
            ids[..seq.spans[0].text_start]
        );
        let back = unpad(&seq.ids, &seq.pads).expect("unpad");
        assert_eq!(back, ids, "unpad(pad(x)) 必须回到客户端可见序列");
    }

    #[test]
    fn pad_law_is_idempotent() {
        let ids = build(&[1, 1, 1], 7);
        let marks = [true, true, true];
        let grids = [grid(), grid(), grid()];
        let s1 = pad(&ids, &marks, &grids);
        let s2 = pad(&ids, &marks, &grids);
        assert_eq!(s1.ids, s2.ids, "pad 律必须幂等(同文本 ⇒ 同 pad 数)");
        assert_eq!(s1.spans, s2.spans);
        assert_eq!(s1.pads, s2.pads);
    }

    /// **全链闭环**:pad → 删一张 → 文本识别 → 重建 → 重新 pad
    /// 判据:删点之前逐字节不变;后缀逐 token 相同(⇒ 可复用)。
    #[test]
    fn delete_pipeline_keeps_prefix_and_suffix() {
        let ids = build(&[1, 1], 5);
        let seq = pad(&ids, &[true, true], &[grid(), grid()]);
        let del = seq.spans[0];
        // 客户端可见文本 = unpad(seq);删掉第一段图像项 ⇒ 客户端新一轮原文
        let text = unpad(&seq.ids, &seq.pads).expect("unpad");
        let raw: Vec<u32> = text[..del.text_start]
            .iter()
            .chain(&text[del.text_end..])
            .copied()
            .collect();
        // 识别(文本坐标)
        let hit = detect_text_delete(&text, &raw, &seq.spans).expect("必须识别");
        assert_eq!((hit.text_a, hit.text_b), (del.text_start, del.text_end));
        assert_eq!((hit.a, hit.b), (del.start, del.end));
        // 重建 + 重新 pad(继承旧 pad 区间)
        let rebuilt = rebuild_after_delete(&seq.ids, &raw, &hit).expect("重建");
        let next = pad_image_spans(
            &rebuilt,
            &[true],
            &[grid()],
            IMG,
            Some(VS),
            Some(VE),
            PAGE,
            PAD,
            &seq.pads
                .iter()
                .copied()
                .filter(|(_, hi)| *hi <= hit.a)
                .collect::<Vec<_>>(),
        )
        .expect("pad2");
        let old = &seq.ids;
        // ① 删点之前逐字节不变(构造性)
        assert_eq!(&next.ids[..hit.a], &old[..hit.a], "前缀必须逐字节不变");
        // ② 后缀逐 token 相同(整页对齐的复用面)
        let k = old[hit.b..]
            .iter()
            .zip(&next.ids[hit.a..])
            .take_while(|(x, y)| x == y)
            .count();
        assert!(k >= PAGE, "后缀可复用长度 {k} 至少一整页");
        assert_eq!(k, old.len() - hit.b, "本形态下整条后缀都该可复用");
        // 实际可拼接的是**整块**部分(尾部不足一页的部分重新 prefill)
        assert_eq!(k / PAGE, 4, "133 tok ⇒ 4 个整页可复用");
        // ③ 下一轮的文本坐标账仍然自洽(unpad 回到"删掉一张图"的文本)
        let back = unpad(&next.ids, &next.pads).expect("unpad2");
        assert_eq!(back, raw, "重建后的文本坐标必须 = 客户端原文");
    }

    /// **非连续多图全删**（2026-11-13 真机：旧图分散在多个 tool result，
    /// 新图整体追加到末尾）。验证：多段范围全部识别、幸存文本保留。
    #[test]
    fn all_old_images_deleted_multi_splice() {
        // 三张图各夹在文本之间，模拟 message 14 / 19 / 138 的分隔。
        let ids = build(&[1, 1, 1], 5);
        let mut seq = pad(&ids, &[true, true, true], &[grid(), grid(), grid()]);
        seq.spans[0].img_tag = 0xA1;
        seq.spans[1].img_tag = 0xB2;
        seq.spans[2].img_tag = 0xC3;
        let text = unpad(&seq.ids, &seq.pads).expect("unpad");
        let mut survivors = text.clone();
        for sp in seq.spans.iter().rev() {
            survivors.drain(sp.text_start..sp.text_end);
        }
        let mut raw = survivors.clone();
        raw.extend([777u32, 778, 779]);
        let plan = super::detect_all_images_deleted_tagged(&text, &raw, &seq.spans, &[0xD4])
            .expect("三张旧图全删应识别");
        assert_eq!(plan.ranges.len(), 3);
        assert_eq!(plan.ranges[0].text_a, seq.spans[0].text_start);
        assert_eq!(plan.ranges[2].text_b, seq.spans[2].text_end);

        let (old_pos, _) = compact_positions(
            &seq.ids,
            &seq.spans,
            IMG,
            &MropeCfg::new([11, 11, 10], 64, 1e7, 2).expect("mrope"),
        )
        .expect("pos");
        let next_spans: Vec<ImgSpan> = Vec::new();
        let mut pos_ids = seq.ids.clone();
        for r in plan.ranges.iter().rev() {
            pos_ids.drain(r.a..r.b);
        }
        pos_ids.extend([777u32, 778, 779]);
        let (pos, _max, gap) = positions_after_deletes(
            &seq.ids,
            &old_pos,
            &pos_ids,
            &next_spans,
            &plan,
            IMG,
            &MropeCfg::new([11, 11, 10], 64, 1e7, 2).expect("mrope"),
        )
        .expect("多段位置");
        assert!(gap);
        // 幸存文本的旧 position 逐 token 保留；新尾巴从旧 max+1 起。
        let mut expected = Vec::new();
        let mut cursor = 0usize;
        for r in &plan.ranges {
            expected.extend_from_slice(&old_pos[cursor..r.a]);
            cursor = r.b;
        }
        expected.extend_from_slice(&old_pos[cursor..]);
        assert_eq!(&pos[..expected.len()], expected.as_slice());
        let base = old_pos
            .iter()
            .flat_map(|p| p.iter())
            .copied()
            .max()
            .unwrap_or(0)
            + 1;
        assert_eq!(pos[expected.len()], [base; 3]);
    }

    /// **相邻图像只删第一张**（2026-11-13 真机：截图替换旧图）。
    ///
    /// 相邻图像在文本坐标里首尾相接；旧实现合并整个游程，只能识别“整组删”。
    /// 这里验证：legacy 识别不了半删，带内容指纹的新识别能精确选第一/第二张，
    /// 且不把保留图的 KV 错绑到删除图内容上。
    #[test]
    fn partial_adjacent_image_delete_uses_content_tags() {
        let mut ids = build(&[1, 1], 0);
        ids.extend([7, 8, 9, 10]);
        let mut seq = pad(&ids, &[true, true], &[grid(), grid()]);
        seq.spans[0].img_tag = 0xA1;
        seq.spans[1].img_tag = 0xB2;
        let a = seq.spans[0];
        let b = seq.spans[1];
        let text = unpad(&seq.ids, &seq.pads).expect("unpad");

        // 只删第一张：旧合并识别必须失败（k 从第二张之后起，对不上 raw 的第二张）。
        let raw_first: Vec<u32> = text[..a.text_start]
            .iter()
            .chain(&text[a.text_end..])
            .copied()
            .collect();
        assert!(
            detect_text_delete(&text, &raw_first, &seq.spans).is_none(),
            "legacy 合并游程不能识别半删"
        );
        let hit = detect_text_delete_tagged(&text, &raw_first, &seq.spans, &[b.img_tag])
            .expect("内容指纹应识别第一张删除");
        assert_eq!((hit.text_a, hit.text_b), (a.text_start, a.text_end));
        assert_eq!((hit.a, hit.b), (a.start, a.end));

        // 只删第二张：同样必须精确命中第二张，而不是第一张。
        let raw_second: Vec<u32> = text[..b.text_start]
            .iter()
            .chain(&text[b.text_end..])
            .copied()
            .collect();
        let hit2 = detect_text_delete_tagged(&text, &raw_second, &seq.spans, &[a.img_tag])
            .expect("内容指纹应识别第二张删除");
        assert_eq!((hit2.text_a, hit2.text_b), (b.text_start, b.text_end));

        // 两张一起删：tail 为空标签，整段游程命中。
        let raw_both: Vec<u32> = text[..a.text_start]
            .iter()
            .chain(&text[b.text_end..])
            .copied()
            .collect();
        let hit3 =
            detect_text_delete_tagged(&text, &raw_both, &seq.spans, &[]).expect("两张一起删");
        assert_eq!((hit3.text_a, hit3.text_b), (a.text_start, b.text_end));
    }

    /// **中间图删除**:三图删第二张，删点之前必须仍逐字节不变。
    ///
    /// 这个形态会保留一张**在删点之前、且已经带 pad** 的图；重建后重跑
    /// pad 律时不能给这张图再补一份 pad（否则前缀会被推歪，调度面拒吸收）。
    #[test]
    fn delete_middle_image_keeps_padded_prefix() {
        let ids = build(&[1, 1, 1], 5);
        let seq = pad(&ids, &[true, true, true], &[grid(), grid(), grid()]);
        let text = unpad(&seq.ids, &seq.pads).expect("unpad");
        let del = seq.spans[1];
        let raw: Vec<u32> = text[..del.text_start]
            .iter()
            .chain(&text[del.text_end..])
            .copied()
            .collect();
        let hit = detect_text_delete(&text, &raw, &seq.spans).expect("必须识别中间图");
        assert_eq!((hit.text_a, hit.text_b), (del.text_start, del.text_end));
        let rebuilt = rebuild_after_delete(&seq.ids, &raw, &hit).expect("重建");
        let inherited: Vec<(usize, usize)> = seq
            .pads
            .iter()
            .copied()
            .filter(|(_, hi)| *hi <= hit.a)
            .collect();
        let next = pad_image_spans(
            &rebuilt,
            &[true, true],
            &[grid(), grid()],
            IMG,
            Some(VS),
            Some(VE),
            PAGE,
            PAD,
            &inherited,
        )
        .expect("pad2");
        assert_eq!(
            &next.ids[..hit.a],
            &seq.ids[..hit.a],
            "删点之前必须逐字节不变"
        );
        assert_eq!(unpad(&next.ids, &next.pads).expect("unpad2"), raw);
    }

    /// 🔴 **陈旧图像 KV 红包(2026-11-03)**:
    /// 同 token 流 + 不同截图 ⇒ **不得**判成同一条链。
    ///
    /// 真机口径(修复前):同文本、只换图 ⇒ `cached_tokens` 1088/1100,模型输出
    /// 逐点等于**上一张图**的方块位置。根因 = 块身份只看 token,而图像只是字面
    /// 相同的占位符;`block_tags` 是修法的一半(另一半是 `hash_block` 混入指纹)。
    #[test]
    fn different_image_content_never_shares_chain() {
        use owl_store::kv::KvStore;
        const PAGE: usize = 32;
        // text32 [img×64] text16 [img×64] text32 = 208 tok ⇒ 6 个整块
        // (首块纯文本 ⇒ 换图后仍能复用到第一张图之前;这正是"少损失"的判据)
        let mut ids: Vec<u32> = (0..32).map(|i| 100 + i).collect();
        let s1 = ids.len();
        ids.extend(std::iter::repeat_n(IMG, 64));
        let e1 = ids.len();
        ids.extend((0..16).map(|i| 200 + i));
        let s2 = ids.len();
        ids.extend(std::iter::repeat_n(IMG, 64));
        let e2 = ids.len();
        ids.extend((0..32).map(|i| 300 + i));

        let spans = vec![
            ImgSpan {
                start: s1,
                end: e1,
                text_start: 32,
                text_end: 96,
                pad_before: 0,
                pad_after: 0,
                grid: (1, 8, 8),
                evict: true,
                img_tag: 0,
            },
            ImgSpan {
                start: s2,
                end: e2,
                text_start: 112,
                text_end: 176,
                pad_before: 0,
                pad_after: 0,
                grid: (1, 8, 8),
                evict: true,
                img_tag: 0,
            },
        ];
        // 纯文本:空 tags(零回归面)
        assert!(
            block_tags(ids.len(), PAGE, &spans).is_empty(),
            "无指纹时必须是纯文本身份"
        );

        // 两张**内容不同**的图 ⇒ 两次请求的指纹不同
        let mut spans_a = spans.clone();
        spans_a[0].img_tag = 0xAAAA_1111_2222_3333;
        spans_a[1].img_tag = 0xBBBB_4444_5555_6666;
        let mut spans_b = spans.clone();
        spans_b[0].img_tag = 0xAAAA_1111_2222_3333; // 第一张相同
        spans_b[1].img_tag = 0xCCCC_7777_8888_9999; // 第二张换了
        let tags_a = block_tags(ids.len(), PAGE, &spans_a);
        let tags_b = block_tags(ids.len(), PAGE, &spans_b);
        assert_eq!(tags_a.len(), ids.len() / PAGE, "逐块指纹数 = 整块数");
        assert_eq!(tags_a[0], 0, "纯文本块 tag 必须为 0(身份与旧实现一致)");
        assert_ne!(tags_a[1], 0, "含图块必须有指纹");
        assert_eq!(tags_a[1], tags_b[1], "同一张图 ⇒ 同指纹(跨会话仍可共享)");
        assert_eq!(tags_a[2], tags_b[2], "块 2 只覆盖第一张图 ⇒ 不随第二张图变");
        assert_ne!(tags_a[4], tags_b[4], "换图 ⇒ 该图覆盖的块指纹必变");

        // 端到端(store 面):A 登记后,B 不得命中任何图像块
        let mut kv = KvStore::new(8, PAGE, 1 << 20, 16);
        let mut table = Vec::new();
        kv.ensure_for_len(&mut table, ids.len()).expect("块表");
        kv.register_chain_tagged(&ids, &table, &tags_a)
            .expect("登记 A");

        let hit_a = kv.match_prefix_tagged(&ids, &tags_a);
        assert_eq!(hit_a.matched_blocks, ids.len() / PAGE, "同图必须整链命中");
        let hit_b = kv.match_prefix_tagged(&ids, &tags_b);
        // 块 0 = 纯文本、块 1/2 = 第一张图(未变)、块 3 起涉及第二张图(已变)
        assert_eq!(
            hit_b.matched_blocks, 3,
            "换第二张图后复用面必须截到块 2 为止(现状:整链命中 ⇒ 陈旧图像 KV)"
        );
        // 忘了带指纹(纯文本口径)也不许**吃到**图像链:含图块身份带指纹 ⇒
        // 空 tags 只能命中纯文本块。失败模式退化为"少复用",而不是"错用旧图"。
        let hit_plain = kv.match_prefix_tagged(&ids, &[]);
        assert_eq!(
            hit_plain.matched_blocks, 1,
            "空 tags 只能复用纯文本块(宁少不错)"
        );
    }

    /// **可删图锚点门（2026-11-13）**：每张 `evict` 图段起点都要出锚点
    /// （同一游程内也一样 —— 只删后几张时才能复用到前几张的页）；未标记图、
    /// 首块之前、链外、`page = 0` 一律不出点。
    #[test]
    fn evict_snap_points_cover_each_marked_image() {
        let page = 32usize;
        let span =
            |start: usize, len: usize, text_start: usize, text_end: usize, evict: bool| ImgSpan {
                start,
                end: start + len,
                text_start,
                text_end,
                pad_before: 0,
                pad_after: 0,
                grid: (1, 8, 8),
                evict,
                img_tag: if evict { 1 } else { 0 },
            };
        // 无图 / 全部未标记 ⇒ 不拍
        assert!(evict_snap_points(&[], 320, page).is_empty());
        assert!(evict_snap_points(&[span(64, 64, 32, 96, false)], 320, page).is_empty());
        // 一张可删图 ⇒ 段起点
        assert_eq!(
            evict_snap_points(&[span(64, 64, 32, 96, true)], 320, page),
            vec![64]
        );
        // 同游程两段（无文本间隔）⇒ **各段起点都出点**（只删后一张时，
        // 前一张的页仍可复用）；顺序升序。
        let run = [span(64, 64, 32, 96, true), span(160, 64, 96, 160, true)];
        assert_eq!(evict_snap_points(&run, 320, page), vec![64, 160]);
        // 中间夹未标记图 ⇒ 只出可删的那两张
        let mixed = [
            span(32, 32, 16, 48, true),
            span(64, 64, 48, 112, false),
            span(160, 64, 112, 176, true),
        ];
        assert_eq!(evict_snap_points(&mixed, 320, page), vec![32, 160]);
        // 首块之前 / 链外 ⇒ 丢弃（0 边界没有可复用前缀；96 == prompt_len 不在链内）
        assert_eq!(
            evict_snap_points(
                &[span(0, 64, 0, 64, true), span(96, 64, 64, 128, true)],
                96,
                page
            ),
            Vec::<usize>::new(),
            "0 与 >= prompt_len 的锚点都不可用"
        );
        // prompt 太短 / page 关 ⇒ 空
        assert!(evict_snap_points(&[span(64, 64, 32, 96, true)], 64, page).is_empty());
        assert!(evict_snap_points(&[span(64, 64, 32, 96, true)], 320, 0).is_empty());
    }

    /// **模板 trim 边界复原门（2026-11-15）**：删点前被模板 `|trim` 剪掉的
    /// 尾部空白可以补回；内容改动、未标记图、图仍在场一律不猜。
    #[test]
    fn restore_trimmed_prefix_only_bridges_whitespace() {
        let ws = 7u32; // 测试用“空白” token（谓词注入，不依赖 tokenizer）
        let is_ws = |t: u32| t == ws;
        let (text, img, tail) = ([10u32, 11, 12], [20u32, 21, 22, 23], [30u32, 31]);
        let mut old: Vec<u32> = text.to_vec();
        old.push(ws);
        old.extend_from_slice(&img);
        old.extend_from_slice(&tail);
        let span = ImgSpan {
            start: 4,
            end: 8,
            text_start: 4,
            text_end: 8,
            pad_before: 0,
            pad_after: 0,
            grid: (1, 8, 8),
            evict: true,
            img_tag: 1,
        };
        // ① 图片被删 + 模板剪掉空白 ⇒ 补回 1 个空白
        let mut raw = text.to_vec();
        raw.extend_from_slice(&tail);
        assert_eq!(
            restore_trimmed_prefix(&old, &mut raw, &[span], is_ws),
            Some(1)
        );
        assert_eq!(raw, [10, 11, 12, ws, 30, 31], "空白插回删点之前");
        // ② 内容 token 改了（不是纯空白）⇒ 不插
        let mut edited = [10u32, 11, 99].to_vec();
        edited.extend_from_slice(&tail);
        assert_eq!(
            restore_trimmed_prefix(&old, &mut edited, &[span], is_ws),
            None
        );
        // ③ 未标记的图 ⇒ 引擎不自作主张
        let mut unmarked = span;
        unmarked.evict = false;
        let mut raw2 = text.to_vec();
        raw2.extend_from_slice(&tail);
        assert_eq!(
            restore_trimmed_prefix(&old, &mut raw2, &[unmarked], is_ws),
            None
        );
        // ④ 图片仍在场（raw 在删点处是图片首 token）⇒ 不插
        let mut keep = text.to_vec();
        keep.extend_from_slice(&img);
        keep.extend_from_slice(&tail);
        assert_eq!(
            restore_trimmed_prefix(&old, &mut keep, &[span], is_ws),
            None
        );
        // ⑤ 只复原**第一个分歧**处的空白：模板 `|trim` 只会剪消息内容的
        //    外尾部 ⇒ 真实形态只有一个缺口；后续缺口交给识别拒绝（宁少不错）。
    }

    /// **多段 leave-gap 页映射门（2026-11-13）**：显式 PageMap 必须把
    /// 公共前缀不足、尾快照越界、最后一张图在尾部三种边界都收干净。
    #[test]
    fn multi_delete_layout_bounds_prefix_and_snapshot() {
        let page = 32usize;
        let old_blocks = 10usize;
        let del = |a: usize, b: usize| TextDelete {
            a,
            b,
            text_a: a,
            text_b: b,
        };
        let ranges = vec![del(2 * page, 3 * page), del(5 * page, 6 * page)];

        // ① 公共前缀 < 第一段删除页：只能复用洞前页，hash 不得越过 used。
        let l = layout_multi_delete(old_blocks, page, &ranges, 2 * page - 1, &[0], false)
            .expect("layout")
            .expect("应可复用第 0 页");
        assert_eq!(l.target_pages, 1);
        assert_eq!(l.used_pages, 1);
        assert_eq!(l.first_changed_page, 2);
        assert_eq!(l.hash_prefix_pages(), 1);
        assert_eq!(l.hash_recompute_pages(), 1..1);
        assert_eq!(l.survivors[..1], [0]);

        // ② 公共前缀跨过第一段删除：身份从第一段删除页起重算，长度仍 = used。
        let l = layout_multi_delete(old_blocks, page, &ranges, 3 * page, &[3], false)
            .expect("layout")
            .expect("边界 3 应可恢复");
        assert_eq!(l.target_pages, 3);
        assert_eq!(l.used_pages, 3);
        assert_eq!(l.survivors[..3], [0, 1, 3]);
        assert_eq!(l.hash_prefix_pages(), 2);
        assert_eq!(l.hash_recompute_pages(), 2..3);

        // ③ 有尾快照但公共前缀不足：绝不能拿尾快照恢复（GDN 态越过缓存面）。
        let l = layout_multi_delete(old_blocks, page, &ranges, 7 * page, &[8], true)
            .expect("layout")
            .expect("应退回边界 8");
        assert_eq!(l.used_pages, 7);
        assert_eq!(l.restore, GapRestore::OldBoundary(8));
        assert!(l.used_pages < l.survivors.len());

        // ④ **最后一张图在尾部**：尾快照落在被删页 ⇒ 不得用（GDN 态越过
        //    新 prefix）；幸存页 8 上有边界快照时，用它且 used 仍 = 全部幸存页。
        let tail_del = vec![del(9 * page, 10 * page)];
        assert!(
            layout_multi_delete(old_blocks, page, &tail_del, 9 * page, &[], true)
                .expect("layout")
                .is_none(),
            "尾部页被删且无幸存边界快照 ⇒ 不许拿尾快照"
        );
        let l = layout_multi_delete(old_blocks, page, &tail_del, 9 * page, &[8], true)
            .expect("layout")
            .expect("幸存页 8 的快照可用");
        assert_eq!(l.used_pages, 9);
        assert_eq!(l.restore, GapRestore::OldBoundary(8));
        assert_eq!(l.hash_prefix_pages(), 9);
        assert_eq!(l.hash_recompute_pages(), 9..9);

        // ④b 尾页仍幸存 ⇒ 尾快照合法。
        let mid_del = vec![del(2 * page, 3 * page)];
        let l = layout_multi_delete(old_blocks, page, &mid_del, 9 * page, &[], true)
            .expect("layout")
            .expect("尾页幸存时尾快照可用");
        assert_eq!(l.used_pages, 9);
        assert_eq!(l.restore, GapRestore::TailSnapshot);
        assert_eq!(l.hash_prefix_pages(), 2);
        assert_eq!(l.hash_recompute_pages(), 2..9);

        // ⑤ 非整页/重叠区间必须响亮拒，不静默截断。
        assert!(layout_multi_delete(old_blocks, page, &[del(1, 2)], 64, &[0], false).is_err());
        assert!(layout_multi_delete(
            old_blocks,
            page,
            &[del(2 * page, 4 * page), del(3 * page, 5 * page)],
            128,
            &[0],
            false
        )
        .is_err());
    }

    /// **位置签名门（2026-11-13）**：同 token 不同 position 基的链不得共享
    /// 块身份；位置紧凑时保持旧身份（零回归）。
    #[test]
    fn position_signature_separates_gapped_and_compact_chains() {
        let ids = build(&[1, 1], 5);
        let mut seq = pad(&ids, &[true, true], &[grid(), grid()]);
        seq.spans[0].img_tag = 0xA1;
        seq.spans[1].img_tag = 0xB2;
        let n = seq.ids.len() / PAGE;
        let compact: Vec<Pos3> = (0..seq.ids.len()).map(|i| [i as i32; 3]).collect();

        let old = block_tags(seq.ids.len(), PAGE, &seq.spans);
        let compact_tags = block_tags_pos(seq.ids.len(), PAGE, &seq.spans, &compact);
        assert_eq!(compact_tags, old, "紧凑位置必须逐字节复用旧身份");
        assert_eq!(compact_tags.len(), n);

        // 只在第 2 页之后产生位置洞：之前的块身份不变，之后的必须全变。
        let hole_at = 2 * PAGE + 5;
        let mut gapped = compact.clone();
        for p in gapped.iter_mut().skip(hole_at) {
            p[0] += 100;
            p[1] += 100;
            p[2] += 100;
        }
        let gapped_tags = block_tags_pos(seq.ids.len(), PAGE, &seq.spans, &gapped);
        assert_eq!(gapped_tags.len(), n, "非紧凑链也必须逐块出 tag");
        assert_eq!(
            &gapped_tags[..2],
            &compact_tags[..2],
            "洞之前的块身份不得变化"
        );
        assert!(
            gapped_tags[2..]
                .iter()
                .zip(&compact_tags[2..])
                .all(|(a, b)| a != b),
            "洞之后的块身份必须全部不同（含纯文本块）"
        );

        // store 面硬门：gapped 链登记后，紧凑会话只能命中洞前前缀。
        let mut kv = owl_store::kv::KvStore::new(16, PAGE, 1 << 20, 16);
        let mut table = Vec::new();
        kv.ensure_for_len(&mut table, seq.ids.len()).expect("块表");
        kv.register_chain_tagged(&seq.ids, &table, &gapped_tags)
            .expect("登记 gapped 链");
        let hit = kv.match_prefix_tagged(&seq.ids, &compact_tags);
        assert_eq!(
            hit.matched_blocks, 2,
            "紧凑位置只能复用洞前前缀，不得吃到 gapped 链"
        );

        // positions 不完整 = 结构账未就绪 ⇒ 退回旧身份，不猜。
        let short = block_tags_pos(seq.ids.len(), PAGE, &seq.spans, &gapped[..PAGE]);
        assert_eq!(short, old, "位置账不足整块时必须退回旧身份");
    }

    #[test]
    fn ledger_prefix_drops_out_of_range_pads_and_spans() {
        let sp = |start, end| ImgSpan {
            start,
            end,
            text_start: 0,
            text_end: 0,
            pad_before: 0,
            pad_after: 0,
            grid: (1, 2, 2),
            evict: true,
            img_tag: 0,
        };
        let spans = [sp(0, 4), sp(8, 16)];
        let pads = [(0, 4), (12, 16)];
        let (s, p) = ledger_prefix(&spans, &pads, 10);
        assert_eq!(s.len(), 1, "只保留完整落在前缀内的 span");
        assert_eq!(p, vec![(0, 4)], "超出前缀尾的 pad 必须丢掉");
    }

    #[test]
    fn leave_gap_preserves_survivor_positions() {
        let mrope = MropeCfg::new([11, 11, 10], 64, 1e7, 2).expect("mrope");
        let ids = build(&[1, 1, 1], 5);
        let seq = pad(&ids, &[true, true, true], &[grid(), grid(), grid()]);
        let (old_pos, old_max) =
            compact_positions(&seq.ids, &seq.spans, IMG, &mrope).expect("旧位置账");
        let text = unpad(&seq.ids, &seq.pads).expect("unpad");
        let del = seq.spans[1];
        let raw: Vec<u32> = text[..del.text_start]
            .iter()
            .chain(&text[del.text_end..])
            .copied()
            .collect();
        let hit = detect_text_delete(&text, &raw, &seq.spans).expect("识别中间图");
        let rebuilt = rebuild_after_delete(&seq.ids, &raw, &hit).expect("重建");
        let inherited: Vec<(usize, usize)> = seq
            .pads
            .iter()
            .copied()
            .filter(|(_, hi)| *hi <= hit.a)
            .collect();
        let next = pad_image_spans(
            &rebuilt,
            &[true, true],
            &[grid(), grid()],
            IMG,
            Some(VS),
            Some(VE),
            PAGE,
            PAD,
            &inherited,
        )
        .expect("重新 pad");
        let (pos, max_pos, gap) = positions_after_splice(
            &seq.ids,
            &old_pos,
            &next.ids,
            &next.spans,
            Some(hit),
            IMG,
            &mrope,
        )
        .expect("leave-gap 位置");
        assert!(gap, "删除后必须标记位置洞");
        assert_eq!(
            &pos[..hit.a],
            &old_pos[..hit.a],
            "删点之前位置逐 token 不变"
        );
        let k = seq.ids[hit.b..]
            .iter()
            .zip(&next.ids[hit.a..])
            .take_while(|(x, y)| x == y)
            .count();
        assert_eq!(
            &pos[hit.a..hit.a + k],
            &old_pos[hit.b..hit.b + k],
            "幸存后缀位置不移动"
        );
        assert_eq!(max_pos, old_max, "无新尾巴时 max_pos 不回退");
        assert_eq!(pos.len(), next.ids.len());

        // 普通追加:旧前缀位置不动,新尾巴从 max_pos+1 起。
        let mut appended = seq.ids.clone();
        appended.extend([777u32, 778, 779]);
        let (pos2, max2, gap2) =
            positions_after_splice(&seq.ids, &old_pos, &appended, &seq.spans, None, IMG, &mrope)
                .expect("append 位置");
        assert!(!gap2, "纯追加仍是紧凑连续位置");
        assert_eq!(&pos2[..seq.ids.len()], &old_pos[..], "旧前缀位置不动");
        assert_eq!(pos2[seq.ids.len()], [old_max + 1; 3], "新尾巴从 max+1 起");
        assert_eq!(max2, old_max + 3);
    }

    #[test]
    fn detect_rejects_text_edits_and_unmarked() {
        let ids = build(&[1, 1], 5);
        let seq = pad(&ids, &[true, true], &[grid(), grid()]);
        let text = unpad(&seq.ids, &seq.pads).expect("unpad");
        let del = seq.spans[0];
        // ① 删点之前被改写 ⇒ 不许吸收(正确性优先)
        let mut raw: Vec<u32> = text[..del.text_start]
            .iter()
            .chain(&text[del.text_end..])
            .copied()
            .collect();
        raw[3] = 424242;
        assert!(
            detect_text_delete(&text, &raw, &seq.spans).is_none(),
            "前缀改写不许吸收"
        );
        // ② 未标记的段不许吸收
        let un = pad(&ids, &[false, true], &[grid(), grid()]);
        let utext = unpad(&un.ids, &un.pads).expect("unpad");
        let ud = un.spans[0];
        let uraw: Vec<u32> = utext[..ud.text_start]
            .iter()
            .chain(&utext[ud.text_end..])
            .copied()
            .collect();
        assert!(
            detect_text_delete(&utext, &uraw, &un.spans).is_none(),
            "未标记不许吸收"
        );
        // ③ 尾部追加(真实 agent 形态)仍然识别
        let mut raw2: Vec<u32> = text[..del.text_start]
            .iter()
            .chain(&text[del.text_end..])
            .copied()
            .collect();
        raw2.extend([777u32, 778]);
        assert!(
            detect_text_delete(&text, &raw2, &seq.spans).is_some(),
            "追加尾部必须仍识别"
        );
    }
}

#[cfg(test)]
mod padded_prefix_tests {
    use super::*;

    #[test]
    fn raw_prefix_mapping_keeps_trailing_orphan_pad() {
        // raw: [10,11, 12,13]; padded: [10,11, 99,98, 12,13]
        let tokens = [10, 11, 99, 98, 12, 13];
        let pads = [(2usize, 4usize)];
        assert_eq!(padded_prefix_for_raw(&tokens, &pads, 0).unwrap(), 0);
        assert_eq!(padded_prefix_for_raw(&tokens, &pads, 1).unwrap(), 1);
        assert_eq!(
            padded_prefix_for_raw(&tokens, &pads, 2).unwrap(),
            4,
            "切点后的 pad 必须留下"
        );
        assert_eq!(padded_prefix_for_raw(&tokens, &pads, 4).unwrap(), 6);
        assert!(padded_prefix_for_raw(&tokens, &pads, 5).is_err());
    }
}

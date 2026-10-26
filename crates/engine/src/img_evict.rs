//! **前缀无感删除(图像段)** —— host 侧纯函数面(识别 / 页对齐 pad / 自证)。
//!
//! # 需求(2026-10-23 用户裁决)
//!
//! GUI agent 的历史是**图文交错**的。真正的需求只有一句话:
//! 「harness 从历史里抽掉一张旧截图,前缀缓存不许失效、删点之后不许重算。」
//!
//! 引擎**不自作主张**:客户端在图像上打「将来可能被删」标记(`TurnImage::evict`),
//! 且 `[pool] image_evict = true` 才启用。没有标记的图 → 逐字节旧路径。
//!
//! # 为什么"删一段"能退化成**一个标量旋转**
//!
//! RoPE 只吃**相对位置**。删掉区间 `[a, b)` 后,其后所有 token 的位置整体
//! 前移**同一个量 `S`**(位置推进,不是 token 位移)⇒ 只要把**后缀的 K** 乘
//! `R(−S)`:
//!
//! - 后缀内部:两端同时旋转 ⇒ 相对位置不变 ⇒ 注意力**精确不变**;
//! - 后缀 ↔ 前缀:相对位置恰好等于"真的删了"之后的值 ⇒ **精确**;
//! - **V 不用动**(V 无位置),**前缀一个字节都不用碰**(连 K 都不转)。
//!
//! 唯一的机械约束:块表是**页粒度**(32 tok/页)。要让"删"变成**零数据搬运**,
//! 被删区间必须是整页数、且起点落页边界 ⇒ 可删图像段强制**页对齐**(前后补 pad)。
//!
//! **2026-10-25 leave-gap**:`[pool].image_evict_mode = "leave_gap"` 时不再
//! 旋转后缀 K;删除只摘页,幸存 token 的 position 原样保留,新 token 从
//! `max_pos+1` 追加。位置账/拼接的唯一产地 = [`compact_positions`] /
//! [`positions_after_splice`];启动/执行面见 `session.rs`、`scheduler.rs`。
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
//! 删除识别、位置同移自证。动手(取链 / 拼接 / 旋转 / 记账)在
//! [`crate::scheduler`] 与 `exec::img_evict`。

use owl_models::layers::mrope::MropeCfg;
use owl_models::layers::vision::GridThw;

/// 一段图像占位(**含** `<|vision_start|>` / `<|vision_end|>` 标记)的坐标账。
///
/// 两套坐标(必须同时记):
/// - `[start, end)` = **padded** 序列里的删除候选(整页对齐;连 pad 一起删);
/// - `[text_start, text_end)` = **unpadded**(客户端可见)序列里的图像项 ——
///   下一轮的删除识别在文本坐标里做。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImgSpan {
    /// padded 段首(= pad_before 块起点;**页对齐**)
    pub start: usize,
    /// padded 段尾(不含;含 pad_after 块)
    pub end: usize,
    /// unpadded 段首(客户端文本坐标)
    pub text_start: usize,
    /// unpadded 段尾(不含)
    pub text_end: usize,
    /// 段首 pad 块长度(padded `start` → 图像项起点)
    pub pad_before: usize,
    /// 段尾 pad 块长度(padded `end` 前的尾巴)
    pub pad_after: usize,
    /// 图像 patch 网格(重建 m-RoPE 位置表要用 —— 位置推进 `max(h,w)/merge`)
    pub grid: GridThw,
    /// 客户端标记:这段将来可能被删(未标记 = 永不吸收删除)
    pub evict: bool,
}

impl ImgSpan {
    /// padded 序列里的两个 pad 块(`(lo, hi)` 半开区间)。
    pub fn pad_ranges(&self) -> [(usize, usize); 2] {
        [
            (self.start - self.pad_before, self.start),
            (self.end - self.pad_after, self.end),
        ]
    }
}

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

/// **页对齐 pad 律**(只在 `[pool] image_evict` 且该图带标记时执行)。
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
    Ok(PaddedSeq { ids: out, spans, pads })
}

/// **删除识别(文本坐标)**:客户端这一轮是不是"正好删了一段已标记图像"。
///
/// 判据:
/// 1. `old_text[..ta] == raw[..ta]`(删点之前的文本逐 token 相同);
/// 2. `old_text[tb..]` 与 `raw[ta..]` 至少有 **1 个 token** 相同(后缀可能
///    很长;分叉点由调用方按 padded 序列精确算)。
///
/// 相邻的已标记图像段先合并(连删两张时文本区间才是连续的)。
/// 返回命中的文本区间 + 对应的 padded 删除区间。
pub fn detect_text_delete(
    old_text: &[u32],
    raw: &[u32],
    spans: &[ImgSpan],
) -> Option<TextDelete> {
    let mut merged: Vec<(usize, usize, usize, usize)> = Vec::new(); // (ta,tb,a,b)
    for s in spans.iter().filter(|s| s.evict) {
        match merged.last_mut() {
            Some(last) if last.1 == s.text_start => {
                last.1 = s.text_end;
                last.3 = s.end;
            }
            _ => merged.push((s.text_start, s.text_end, s.start, s.end)),
        }
    }
    for (ta, tb, a, b) in merged {
        if tb > old_text.len() || ta >= tb || ta > raw.len() {
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
        return Some(TextDelete { a, b, text_a: ta, text_b: tb });
    }
    None
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

/// **位置三元组**(与 `MropeCfg` 的 `[T,H,W]` 同序;i32 足够覆盖 262k)。
pub type Pos3 = [i32; 3];

/// **紧凑位置账**:从 token 0 起按 m-RoPE 重新计算(普通 turn / rebase)。
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
    let max_pos = pos.iter().flat_map(|p| p.iter()).copied().max().unwrap_or(0);
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
    if spans.iter().any(|s| s.start < tail_start && s.end > tail_start) {
        return Err(format!(
            "positions_after_splice: 图像段跨越尾巴起点 {tail_start}(spans={:?})",
            spans.iter().map(|s| (s.start, s.end)).collect::<Vec<_>>()
        ));
    }
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
    let max_pos = pos.iter().flat_map(|p| p.iter()).copied().max().unwrap_or(0);
    Ok((pos, max_pos, deletion.is_some() || tail_start < old_tokens.len()))
}

/// **位置同移自证**(G-自证):删掉 `[a,b)` 之后,后缀每个 token 的三网格位置
/// 是否都只差**同一个标量** `S`。
///
/// 用**真正喂给前向的同一份** `MropeCfg::rope_index` 两侧各算一遍(不抄公式 ——
/// 抄公式是"静默位移半格"的头号来源),再逐 token 逐分量比对:
/// `pos_old[b+j] − pos_new[a+j] == S`。
///
/// 返回 `Some(S)`(S = 后缀位置前移量;K 乘 `R(−S)`);不成立 = `None`。
pub fn positions_shift(
    old: &[u32],
    new: &[u32],
    old_spans: &[ImgSpan],
    new_spans: &[ImgSpan],
    image_token: u32,
    mrope: &MropeCfg,
    a: usize,
    b: usize,
    keep_tokens: usize,
) -> Option<i64> {
    if b + keep_tokens > old.len() || a + keep_tokens > new.len() {
        return None;
    }
    // 网格按**出现序**喂给 rope_index;段数与占位数不符 = 结构错(响亮 None)
    let runs_old = image_runs(old, image_token);
    let runs_new = image_runs(new, image_token);
    if runs_old.len() != old_spans.len() || runs_new.len() != new_spans.len() {
        return None;
    }
    let grids_old: Vec<GridThw> = old_spans.iter().map(|s| s.grid).collect();
    let grids_new: Vec<GridThw> = new_spans.iter().map(|s| s.grid).collect();
    let tt_old = owl_models::layers::mrope::token_types(old, image_token, None);
    let tt_new = owl_models::layers::mrope::token_types(new, image_token, None);
    let pi_old = mrope.rope_index(old, &tt_old, &grids_old, &[]).ok()?;
    let pi_new = mrope.rope_index(new, &tt_new, &grids_new, &[]).ok()?;
    let s0 = pi_old.pos.get(b)?[0] - pi_new.pos.get(a)?[0];
    for j in 0..keep_tokens {
        let po = pi_old.pos.get(b + j)?;
        let pn = pi_new.pos.get(a + j)?;
        for c in 0..3 {
            if po[c] - pn[c] != s0 {
                return None;
            }
        }
    }
    Some(s0)
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
        pad_image_spans(ids, marks, grids, IMG, Some(VS), Some(VE), PAGE, PAD, &[])
            .expect("pad")
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
        assert_eq!(seq.spans[0].start, seq.spans[0].text_start, "未标记段不 pad");
        assert_eq!(seq.ids[..seq.spans[0].text_start], ids[..seq.spans[0].text_start]);
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
    /// 判据:删点之前逐字节不变;后缀逐 token 相同(⇒ 可复用 + 单标量旋转)。
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
        let inherited: Vec<(usize, usize)> =
            seq.pads.iter().copied().filter(|(_, hi)| *hi <= hit.a).collect();
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
        assert_eq!(&next.ids[..hit.a], &seq.ids[..hit.a], "删点之前必须逐字节不变");
        assert_eq!(unpad(&next.ids, &next.pads).expect("unpad2"), raw);
    }

    #[test]
    fn leave_gap_preserves_survivor_positions() {
        let mrope = MropeCfg::new([11, 11, 10], 64, 1e7, 2).expect("mrope");
        let ids = build(&[1, 1, 1], 5);
        let seq = pad(&ids, &[true, true, true], &[grid(), grid(), grid()]);
        let (old_pos, old_max) = compact_positions(&seq.ids, &seq.spans, IMG, &mrope)
            .expect("旧位置账");
        let text = unpad(&seq.ids, &seq.pads).expect("unpad");
        let del = seq.spans[1];
        let raw: Vec<u32> = text[..del.text_start]
            .iter()
            .chain(&text[del.text_end..])
            .copied()
            .collect();
        let hit = detect_text_delete(&text, &raw, &seq.spans).expect("识别中间图");
        let rebuilt = rebuild_after_delete(&seq.ids, &raw, &hit).expect("重建");
        let inherited: Vec<(usize, usize)> =
            seq.pads.iter().copied().filter(|(_, hi)| *hi <= hit.a).collect();
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
        assert_eq!(&pos[..hit.a], &old_pos[..hit.a], "删点之前位置逐 token 不变");
        let k = seq.ids[hit.b..]
            .iter()
            .zip(&next.ids[hit.a..])
            .take_while(|(x, y)| x == y)
            .count();
        assert_eq!(&pos[hit.a..hit.a + k], &old_pos[hit.b..hit.b + k], "幸存后缀位置不移动");
        assert_eq!(max_pos, old_max, "无新尾巴时 max_pos 不回退");
        assert_eq!(pos.len(), next.ids.len());

        // 普通追加:旧前缀位置不动,新尾巴从 max_pos+1 起。
        let mut appended = seq.ids.clone();
        appended.extend([777u32, 778, 779]);
        let (pos2, max2, gap2) = positions_after_splice(
            &seq.ids,
            &old_pos,
            &appended,
            &seq.spans,
            None,
            IMG,
            &mrope,
        )
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
        let mut raw: Vec<u32> = text[..del.text_start].iter().chain(&text[del.text_end..]).copied().collect();
        raw[3] = 424242;
        assert!(detect_text_delete(&text, &raw, &seq.spans).is_none(), "前缀改写不许吸收");
        // ② 未标记的段不许吸收
        let un = pad(&ids, &[false, true], &[grid(), grid()]);
        let utext = unpad(&un.ids, &un.pads).expect("unpad");
        let ud = un.spans[0];
        let uraw: Vec<u32> = utext[..ud.text_start]
            .iter()
            .chain(&utext[ud.text_end..])
            .copied()
            .collect();
        assert!(detect_text_delete(&utext, &uraw, &un.spans).is_none(), "未标记不许吸收");
        // ③ 尾部追加(真实 agent 形态)仍然识别
        let mut raw2: Vec<u32> = text[..del.text_start].iter().chain(&text[del.text_end..]).copied().collect();
        raw2.extend([777u32, 778]);
        assert!(detect_text_delete(&text, &raw2, &seq.spans).is_some(), "追加尾部必须仍识别");
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
        assert_eq!(padded_prefix_for_raw(&tokens, &pads, 2).unwrap(), 4, "切点后的 pad 必须留下");
        assert_eq!(padded_prefix_for_raw(&tokens, &pads, 4).unwrap(), 6);
        assert!(padded_prefix_for_raw(&tokens, &pads, 5).is_err());
    }
}

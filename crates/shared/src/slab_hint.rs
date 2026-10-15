//! 捕获 slab 定量(E8 工程债 → A1 解锁,2026-10-10)。
//!
//! 病:每图 `graph_begin` 预留**满额 slab**(`OWL_CAPTURE_SLAB_MB`,缺省
//! 64MiB),slab 经 carve 输出块被图持久引用 → 常驻。图族(verify +
//! fold×7 + propose ≈ 10 图)× 满额 = 0.6-1.3GiB 纯预留浪费,24G 贴顶
//! 场景(27B + draft)直接把 propose 图挤成 eager 降级(2026-10-10 A1
//! 复测实录:64/96/128MB 三档 carve 全崩,160MB slab 本体 OOM)。
//!
//! 修:**warmup 计量定量** —— GraphPlan 的 warmup(姿势 6 lite)eager
//! dry 一遍闭包,`meter_take()` 取闭包真分配字节 → hint = ×1.15 + 1MiB
//! → `graph_begin` 消费。mid-capture 分配非法(malloc 在捕获窗内非法,
//! 这正是 slab 预分配存在的原因),故只能预定量、不能边捕边长。
//!
//! - env 显式设 `OWL_CAPTURE_SLAB_MB` = 旧固定档(测试/诊断兼容,忽略 hint);
//! - env 缺省 = hint 定量(hint 缺席/为 0 → 缺省 64MiB 兜底);
//! - `CaptureState.cap` 记本图实际 cap,carve 校验对齐同一口(旧实现
//!   carve 校验读 env,与 hint 定量的 slab 脱口)。
//!
//! 全进程单引擎假设(actor 串行 boot;多引擎共驻非现行形态)。

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

static METER: AtomicU64 = AtomicU64::new(0);
static HINT: AtomicUsize = AtomicUsize::new(0);

/// 归因标签回填:求值器在分配点以此挂块名(见 `interpreters::eval` 的三个
/// 分配臂)。名字可缺 —— 缺的归 `(未命名)`,不静默。(块 id 空间与
/// [`probe_alloc`] 同源:两端都是同一块句柄。)
///
/// 设备侧真分配计量(server Alloc 处理器、非捕获窗分支调用;字节)。
pub fn meter_add(n_bytes: u64) {
    METER.fetch_add(n_bytes, Ordering::Relaxed);
}

/// 取走并清零累计计量字节(GraphPlan warmup 前弃一次旧账,后取真账)。
pub fn meter_take() -> u64 {
    METER.swap(0, Ordering::Relaxed)
}

/// 置本图 slab hint(字节;GraphPlan warmup 后、捕获前)。
pub fn set_hint(n_bytes: usize) {
    HINT.store(n_bytes, Ordering::Relaxed);
}

/// 消费 hint(graph_begin;None = 缺席)。
pub fn take_hint() -> Option<usize> {
    let v = HINT.swap(0, Ordering::Relaxed);
    if v == 0 { None } else { Some(v) }
}

/// 丢弃 hint(捕获失败/未请求捕获的收尾;防陈旧 hint 污染下一图)。
pub fn clear_hint() {
    HINT.store(0, Ordering::Relaxed);
}

// ============================================================================
// §2 slab 内回收分配器 + 需求探针(M1;自 `feat/prefill-graph` 分支 S1.5 择取,
//    2026-10-15 —— 设计见 docs/arch/prefill-graph-设计与实现.md §4.3)
// ============================================================================
//
// **为什么要它**(分支真机定谳,施工文档 §9):旧实现两点叠加 ——
// ①`try_capture` 走 **plain(无回收)** 归约 ⇒ 捕获窗内所有中间块**常驻**;
// ②`free_blocks` 对 `Block::Carved` 是 no-op ⇒ slab 空间不复用。
// 合起来 ⇒ slab 容量下限 = **分配总和**(27B/512-tok chunk 实测 **9362 MiB**)
// ⇒ `graph_begin` OOM ⇒ 图退化为 eager。
//
// **修法**:捕获窗改走 scoped(回收)归约 + `free_blocks` 把 Carved 块**推回**
// 本 arena 的尺寸类回收表 ⇒ 容量代偿回到**峰值活跃集**(liveness 后)。
//
// ⚠️ **安全前提**(2026-10-15 二轮;见 state.rs `GraphRefs`):回收=地址可复用,
// 而图里烘的是裸指针 ⇒ 复用的安全性由"归还的块不再被任何**已记录**的图节点
// 引用"保证(scoped 归约只归还不可达中间块,捕获按求值序记录 ⇒ 复用点在所有
// 消费者之后)。**输入/输出槽不参与归还**。

/// slab 内分配器:bump + **空闲块有序表(相邻合并 + 顶部回收)**(对齐 256B)
///
/// **为什么不再用尺寸类 best-fit**(2026-10-15 归因定谳):旧实现按尺寸类存空闲块,
/// 且**从不合并相邻空闲块**、**从不回退 bump 顶** ⇒ 27B/512-tok 桶实测:
/// 容量定尺 193MiB 而真·同时活跃只有 81MiB(**58% = 空洞**;见 `graph_plan` 的
/// `slab 峰值归因` 行)。病根 = 大/小块交替的分配序列(8000+ 次 narrow/add/rmsnorm
/// 小块 + MLP 大中间张量)不断把 bump 顶高,而释放回来的碎片洞支撑不了一个
/// 34MB 的大请求。
///
/// **修法**:空闲块按**偏移**有序存(BTreeMap)+ 释放时**左右相邻合并** +
/// **贴顶回收**(贴住 bump 顶的空闲块 ⇒ `used` 退回)。层间同形状序列(64 层 ×
/// 同批张量)因此复用同一段地址 ⇒ 容量趋近真活跃集,而非"分配足迹上界"。
/// 不变量不变:256B 对齐 / 零字节退化块 / `high` = 历史最大 bump(定尺口径)。
#[derive(Default, Debug)]
pub struct SlabArena {
    /// bump 顶(可被 [`SlabArena::trim_top`] 回退)
    used: usize,
    /// 历史最大 bump(= slab 容量需求;只增不减)
    high: usize,
    /// 空闲块:**off → cap**(均 256B 对齐;相邻即 `off+cap == next.off`)
    free: std::collections::BTreeMap<usize, usize>,
}

const ARENA_ALIGN: usize = 256;

fn align_up(n: usize) -> usize {
    n.div_ceil(ARENA_ALIGN) * ARENA_ALIGN
}

impl SlabArena {
    /// 切一块 ≥ `n`(返回 `(off, cap_n)`;`cap_n` = 该块的可用容量,回收凭它)
    pub fn carve(&mut self, n: usize, slab_cap: usize) -> Option<(usize, usize)> {
        let need = align_up(n);
        if need == 0 {
            // 零字节退化块:偏移 = 当前 bump 顶(不占容量、不抬水位)
            return Some((align_up(self.used), 0));
        }
        // best-fit:取 ≥ need 里最小者(合并后大洞可填小请求 ⇒ 余量拆回)
        let mut best: Option<(usize, usize)> = None; // (off, cap)
        for (&off, &cap) in self.free.iter() {
            if cap < need {
                continue;
            }
            match best {
                Some((_, bc)) if bc <= cap => {}
                _ => best = Some((off, cap)),
            }
            if cap == need {
                break; // 完美匹配,无更优
            }
        }
        if let Some((off, cap)) = best {
            self.free.remove(&off);
            let rest = cap - need;
            if rest >= ARENA_ALIGN {
                self.insert_free(off + need, rest);
            }
            return Some((off, need));
        }
        // bump
        let off = align_up(self.used);
        if off + need > slab_cap {
            return None;
        }
        self.used = off + need;
        self.high = self.high.max(self.used);
        Some((off, need))
    }

    /// 归还(捕获窗内 `free_blocks` 对 Carved 块调用):入表 → 合并 → 贴顶回收
    pub fn release(&mut self, off: usize, cap_n: usize) {
        if cap_n == 0 {
            return;
        }
        self.insert_free(off, cap_n);
        self.trim_top();
    }

    /// 入表并**合并左右相邻**(256B 对齐 ⇒ 相邻即 `off+cap == 邻.off`)
    fn insert_free(&mut self, off: usize, cap: usize) {
        let (mut off, mut cap) = (off, cap);
        if let Some((&loff, &lcap)) = self.free.range(..off).next_back() {
            if loff + lcap == off {
                self.free.remove(&loff);
                off = loff;
                cap += lcap;
            }
        }
        if let Some((&roff, &rcap)) = self.free.range(off..).next() {
            if off + cap == roff {
                self.free.remove(&roff);
                cap += rcap;
            }
        }
        self.free.insert(off, cap);
    }

    /// **贴顶回收**:空闲块贴住 bump 顶 ⇒ `used` 退回(可连续回退)
    fn trim_top(&mut self) {
        loop {
            let Some((&off, &cap)) = self.free.iter().next_back() else {
                return;
            };
            if off + cap == self.used {
                self.free.remove(&off);
                self.used = off;
            } else {
                return;
            }
        }
    }

    /// 高水位 = 本 arena 需要的 slab 容量(探针口径)
    pub fn high(&self) -> usize {
        self.high
    }
}

static PROBE_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// 探针状态(= 影子 arena + **归因账**)。
///
/// 归因口径(2026-10-15 用户令"先归因"):
/// - `peak_ids`:高水位被刷新的那一刻**活跃块集合** ⇒ 它就是"峰值构成";
/// - `bytes`/`names`:块 → 真请求字节 / 名字(求值器在分配点回填;可缺);
/// - `per_name_sum`:按名字的**分配总量**(累计,不随 free 减) —— 回答"谁总在要内存"。
struct ProbeState {
    arena: SlabArena,
    ids: std::collections::HashMap<u64, (usize, usize)>,
    /// 块 → 真请求字节(**插入即留** —— 峰值快照里的块可能已释放,
    /// 渲染时仍要能读到尺寸;2026-10-15 实测踩过:free 时删除 ⇒ 报告低估 6×)
    bytes: std::collections::HashMap<u64, usize>,
    /// 块 → 名字(同 bytes:插入即留)
    names: std::collections::HashMap<u64, String>,
    per_name_sum: std::collections::HashMap<String, usize>,
    /// 当前活跃字节合计(用于"活跃峰值"口径)
    live_sum: usize,
    /// 口径 A:**容量峰值**(影子 arena 高水位最大那次)的活跃集合
    peak_bump_ids: Vec<u64>,
    peak_bump_high: usize,
    peak_bump_live: usize,
    /// 口径 B:**活跃峰值**(同时在用字节最多那次)的活跃集合
    peak_live_ids: Vec<u64>,
    peak_live_live: usize,
    peak_live_high: usize,
}

static PROBE: std::sync::Mutex<Option<ProbeState>> = std::sync::Mutex::new(None);
/// 最后一次 `probe_end` 的归因报告(文本;调用方取走后自行打日志)
static PROBE_REPORT: std::sync::Mutex<String> = std::sync::Mutex::new(String::new());

/// 归因报告的 top-N 条数
const ATTR_TOP: usize = 8;

/// 探针开/关(GraphPlan warmup 窗口;关时零开销)
pub fn probe_on(on: bool) {
    if let Ok(mut g) = PROBE.lock() {
        *g = if on {
            Some(ProbeState {
                arena: SlabArena::default(),
                ids: std::collections::HashMap::new(),
                bytes: std::collections::HashMap::new(),
                names: std::collections::HashMap::new(),
                per_name_sum: std::collections::HashMap::new(),
                live_sum: 0,
                peak_bump_ids: Vec::new(),
                peak_bump_high: 0,
                peak_bump_live: 0,
                peak_live_ids: Vec::new(),
                peak_live_live: 0,
                peak_live_high: 0,
            })
        } else {
            None
        };
        PROBE_ON.store(on, Ordering::Relaxed);
    }
}

/// 影子分配(warmup 的真分配点调用;探针关 = 立即返回)
pub fn probe_alloc(id: u64, n_bytes: usize) {
    if !PROBE_ON.load(Ordering::Relaxed) {
        return;
    }
    if let Ok(mut g) = PROBE.lock() {
        if let Some(st) = g.as_mut() {
            // slab 容量按"无上限"跑,只求高水位(真容量由 high 决定)
            if let Some((off, cap_n)) = st.arena.carve(n_bytes, usize::MAX) {
                st.ids.insert(id, (off, cap_n));
                st.bytes.insert(id, n_bytes);
                st.live_sum += n_bytes;
                // 两个口径各记一份快照(名字晚到、尺寸插入即留 ⇒ 渲染时都能解析):
                //  A **容量峰值**(bump 高水位最大那次)—— 定尺靠它
                //  B **活跃峰值**(同时在用字节最多那次)—— "谁占大头"靠它
                let high = st.arena.high();
                if high > st.peak_bump_high {
                    st.peak_bump_high = high;
                    st.peak_bump_live = st.live_sum;
                    st.peak_bump_ids = st.ids.keys().copied().collect();
                }
                if st.live_sum > st.peak_live_live {
                    st.peak_live_live = st.live_sum;
                    st.peak_live_high = high;
                    st.peak_live_ids = st.ids.keys().copied().collect();
                }
            }
        }
    }
}

/// 归因标签(求值器在分配点回填块名;名气可缺 —— 缺的归"(未命名)")
pub fn probe_name(id: u64, name: &str) {
    if !PROBE_ON.load(Ordering::Relaxed) {
        return;
    }
    if let Ok(mut g) = PROBE.lock() {
        if let Some(st) = g.as_mut() {
            if let Some(n) = st.bytes.get(&id) {
                *st.per_name_sum.entry(name.to_string()).or_insert(0) += *n;
            }
            st.names.insert(id, name.to_string());
        }
    }
}

/// 影子归还(warmup 的 free 点;未登记/探针关 = 无动作)
pub fn probe_free(id: u64) {
    if !PROBE_ON.load(Ordering::Relaxed) {
        return;
    }
    if let Ok(mut g) = PROBE.lock() {
        if let Some(st) = g.as_mut() {
            if let Some((off, cap_n)) = st.ids.remove(&id) {
                st.arena.release(off, cap_n);
            }
            // ⚠️ `bytes`/`names` **不删**:峰值快照可能引用已释放的块
            st.live_sum = st.live_sum.saturating_sub(*st.bytes.get(&id).unwrap_or(&0));
        }
    }
}

/// 关探针并取高水位(= 捕获所需 slab 容量)。归因报告留在
/// [`take_probe_report`](文本为空 = 无数据/探针未开)。
pub fn probe_end() -> usize {
    PROBE_ON.store(false, Ordering::Relaxed);
    let mut peak = 0;
    let mut report = String::new();
    if let Ok(mut g) = PROBE.lock() {
        if let Some(st) = g.as_ref() {
            peak = st.arena.high();
            report = render_report(st);
        }
        *g = None;
    }
    if let Ok(mut r) = PROBE_REPORT.lock() {
        *r = report;
    }
    peak
}

/// 取走最后一次归因报告(取走即清)
pub fn take_probe_report() -> String {
    PROBE_REPORT.lock().map(|mut r| std::mem::take(&mut *r)).unwrap_or_default()
}

/// 报告渲染:双口径 —— ①**容量峰值**(slab 定尺靠它)②**活跃峰值**(谁占大头)
/// 各自的构成 top-N,加"每名字分配总量"top-N。
/// 两者差值高 = 尺寸类空洞(碎片);差值低 = 真·同时活跃集。
fn render_report(st: &ProbeState) -> String {
    if st.peak_bump_ids.is_empty() && st.peak_live_ids.is_empty() {
        return String::new();
    }
    let compose = |ids: &[u64]| -> (String, usize) {
        let mut comp: Vec<(String, usize)> = Vec::new();
        for id in ids {
            let name = st.names.get(id).cloned().unwrap_or_else(|| "(未命名)".to_string());
            let n = *st.bytes.get(id).unwrap_or(&0);
            match comp.iter_mut().find(|(k, _)| *k == name) {
                Some((_, v)) => *v += n,
                None => comp.push((name, n)),
            }
        }
        comp.sort_unstable_by_key(|(_, n)| std::cmp::Reverse(*n));
        let total: usize = comp.iter().map(|(_, n)| *n).sum();
        let s = comp
            .iter()
            .take(ATTR_TOP)
            .map(|(k, n)| {
                format!(
                    "{k} {:.1}MB({:.0}%)",
                    *n as f64 / 1048576.0,
                    if total == 0 { 0.0 } else { *n as f64 * 100.0 / total as f64 }
                )
            })
            .collect::<Vec<_>>()
            .join(" + ");
        (s, total)
    };
    let (bump_part, bump_live) = compose(&st.peak_bump_ids);
    let (live_part, live_sum) = compose(&st.peak_live_ids);
    let mut sums: Vec<(String, usize)> = st.per_name_sum.iter().map(|(k, v)| (k.clone(), *v)).collect();
    sums.sort_unstable_by_key(|(_, n)| std::cmp::Reverse(*n));
    let sum_total: usize = sums.iter().map(|(_, n)| *n).sum();
    let sum_part = sums
        .iter()
        .take(ATTR_TOP)
        .map(|(k, n)| {
            format!(
                "{k} {:.0}MB({:.0}%)",
                *n as f64 / 1048576.0,
                if sum_total == 0 { 0.0 } else { *n as f64 * 100.0 / sum_total as f64 }
            )
        })
        .collect::<Vec<_>>()
        .join(" + ");
    format!(
        "容量峰值 {:.1}MiB(活跃 {:.1}MiB / {} 块,差 {:.1}MiB=空洞):{bump_part} ‖ 活跃峰值 {:.1}MiB(容量位 {:.1}MiB / {} 块):{live_part} ‖ 分配总量 {:.1}GB:{sum_part}",
        st.peak_bump_high as f64 / 1048576.0,
        bump_live as f64 / 1048576.0,
        st.peak_bump_ids.len(),
        (st.peak_bump_high.saturating_sub(bump_live)) as f64 / 1048576.0,
        live_sum as f64 / 1048576.0,
        st.peak_live_high as f64 / 1048576.0,
        st.peak_live_ids.len(),
        sum_total as f64 / 1073741824.0,
    )
}

#[cfg(test)]
mod arena_tests {
    use super::*;

    /// 探针是**进程级静态** ⇒ 两个探针用例并行会互相清台(2026-10-15 实测:
    /// 单跑绿、套跑红)。用锁把"碰全局态"的用例串行化(其余纯 arena 用例不受影响)。
    static PROBE_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// 对齐:bump 分配按 256B 对齐(向量化核基底要求)
    #[test]
    fn carve_is_256b_aligned() {
        let mut a = SlabArena::default();
        let (o0, c0) = a.carve(1, 4096).expect("carve 1");
        let (o1, _) = a.carve(1, 4096).expect("carve 2");
        assert_eq!((o0, c0), (0, 256));
        assert_eq!(o1, 256, "第二块须对齐到 256B");
        let (o2, _) = a.carve(300, 4096).expect("carve 300");
        assert_eq!(o2, 512);
        assert_eq!(a.high(), 1024, "300→512 对齐后总高水位");
    }

    /// 归还后再切:地址复用(同尺寸类最佳匹配;剩余 ≥256B 拆回表)
    #[test]
    fn release_then_carve_reuses_address() {
        let mut a = SlabArena::default();
        let (o0, c0) = a.carve(1000, 1 << 20).expect("carve");
        assert_eq!(o0, 0);
        assert_eq!(c0, 1024);
        a.release(o0, c0);
        let (o1, _) = a.carve(1000, 1 << 20).expect("再切");
        assert_eq!(o1, 0, "须复用归还地址(这就是峰值口径的来源)");
        assert_eq!(a.high(), 1024, "复用不得抬高水位");
        // 小块切大块:拆余量
        a.release(0, 1024);
        let (o2, c2) = a.carve(256, 1 << 20).expect("切小");
        assert_eq!((o2, c2), (0, 256));
        let (o3, c3) = a.carve(256, 1 << 20).expect("承余量");
        assert_eq!((o3, c3), (256, 256), "余量 ≥256B 须拆回表");
        assert_eq!(a.high(), 1024);
    }

    /// **相邻合并**(2026-10-15 归因之修):两块分别释放 ⇒ 合成一块大洞,
    /// 大请求直接复用,不得顶高水位
    #[test]
    fn coalesce_adjacent_releases_serves_big_request() {
        let mut a = SlabArena::default();
        let (o1, c1) = a.carve(1024, 1 << 20).expect("blk1");
        let (o2, c2) = a.carve(1024, 1 << 20).expect("blk2");
        assert_eq!((o1, c1, o2, c2), (0, 1024, 1024, 1024));
        // 分两次释放(非相邻顺序:先右后左 ⇒ 仍须合并)
        a.release(o2, c2);
        a.release(o1, c1);
        let (o3, c3) = a.carve(2048, 1 << 20).expect("合并后的大块");
        assert_eq!((o3, c3), (0, 2048), "两块 1024 合并后须能一次切 2048");
        assert_eq!(a.high(), 2048, "合并复用不得抬水位");
    }

    /// **贴顶回收**:释放贴顶块 ⇒ bump 退回(层间同形状序列靠它)
    #[test]
    fn trim_top_returns_bump_pointer() {
        let mut a = SlabArena::default();
        let (o1, c1) = a.carve(4096, 1 << 20).expect("blk1");
        let (o2, c2) = a.carve(2048, 1 << 20).expect("blk2");
        assert_eq!(a.high(), 6144);
        a.release(o2, c2); // 贴顶 ⇒ used 退回 4096
        a.release(o1, c1); // 又贴顶 ⇒ used 退回 0
        let (o3, _) = a.carve(6144, 1 << 20).expect("再切 6144");
        assert_eq!(o3, 0, "贴顶回收后须从 0 重新长(而不是接着 6144 长)");
        assert_eq!(a.high(), 6144, "水位 = 历史最大,不因回收而下降");
    }

    /// **层间同形状序列**(27B prefill 的真实形状):64 轮"大-小-小"交替 ⇒
    /// 水位 = 单轮峰值(旧尺寸类实现会被交替尺寸拖到多轮叠加)
    #[test]
    fn repeated_layer_pattern_fits_one_layer_peak() {
        let mut a = SlabArena::default();
        for _ in 0..64 {
            let big = a.carve(34 * 1024 * 1024, usize::MAX).expect("大块");
            let mid = a.carve(17 * 1024 * 1024, usize::MAX).expect("中块");
            for _ in 0..64 {
                let small = a.carve(64 * 1024, usize::MAX).expect("小块");
                a.release(small.0, small.1);
            }
            a.release(mid.0, mid.1);
            a.release(big.0, big.1);
        }
        let one = align_up(34 << 20) + align_up(17 << 20) + align_up(64 << 10);
        assert_eq!(a.high(), one, "64 轮同形状序列必须只占单轮峰值(实测 {})", a.high());
    }

    /// 容量上限:超限返回 None(调用方结构化拒,不静默截断);
    /// 零字节 = 退化块(不占容量、不抬水位 —— 与旧 bump 语义逐字一致)
    #[test]
    fn carve_over_capacity_is_none() {
        let mut a = SlabArena::default();
        assert!(a.carve(1024, 1024).is_some());
        assert!(a.carve(1, 1024).is_none(), "耗尽须 None");
        let (o, c) = a.carve(0, 1024).expect("零字节不耗容量(需量 0 ⇒ 恒命中)");
        assert_eq!((o, c), (1024, 0), "零字节块:偏移内的首地址 + 容量 0");
        assert_eq!(a.high(), 1024, "零字节不抬水位");
    }

    /// 探针:alloc/free 配对 ⇒ 高水位 = 峰值活跃集(含未兑现的归还)
    #[test]
    fn probe_measures_peak_live_set() {
        let _g = PROBE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        probe_on(true);
        probe_alloc(1, 4096);
        probe_alloc(2, 4096);
        probe_free(1); // 归还 ⇒ 下一个可复用
        probe_alloc(3, 4096);
        let peak = probe_end();
        assert_eq!(peak, 8192, "峰值 = 两块(复用归还块,不涨高水位)");
        assert_eq!(probe_end(), 0, "取过即清(二次 = 0)");
        // 关态:零动作(不 panic、不计)
        probe_alloc(4, 1 << 20);
        assert_eq!(probe_end(), 0);
    }

    /// **峰值归因**(2026-10-15):快照 = 高水位刷新时刻的活跃集;
    /// 名字晚到(求值器在分配后回填)也必须被解析到;free 掉的块不得入报告。
    #[test]
    fn probe_attribution_reports_peak_composition_and_sums() {
        let _g = PROBE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _ = take_probe_report(); // 清前一次
        probe_on(true);
        // 峰值时刻:1(4KiB gemm)+ 2(2KiB norm)同时活
        probe_alloc(1, 4096);
        probe_name(1, "marlin.w4a16");
        probe_alloc(2, 2048); // 此处高水位 = 6144(对齐后 4096+2048)
        probe_name(2, "owl_norm_rope_f16");
        probe_free(2); // 峰值后归还 ⇒ 不得出现在"峰值构成"里(但留在分配总量)
        probe_alloc(3, 1024);
        probe_name(3, "gdn_chunked");
        let peak = probe_end();
        let rep = take_probe_report();
        assert_eq!(peak, 6144, "峰值 = 前两块(复合后不高过)");
        assert!(rep.contains("容量峰值") && rep.contains("活跃峰值"), "报告:{rep}");
        assert!(rep.contains("marlin.w4a16"), "容量峰值构成须含 marlin(报告:{rep})");
        assert!(rep.contains("owl_norm_rope_f16"), "活跃峰值构成须含 norm(报告:{rep})");
        assert!(rep.contains("分配总量"), "报告:{rep}");
        assert!(rep.contains("gdn_chunked"), "总量须含晚到的名字(报告:{rep})");
        assert_eq!(take_probe_report(), "", "取走即清");
        // 未开探针:名字/报告皆空(零开销路径)
        probe_name(9, "x");
        assert_eq!(take_probe_report(), "");
    }
}

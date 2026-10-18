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
    /// **逐块引用计数快照**(块 id → (users, pending);求值器在改动点推。
    /// 2026-10-16):"活着一大片"时唯一能回答**"它凭什么是活的"**的面 ——
    /// `users` 抬着(谁还引用)vs `pending` 没减(预波的消费者计数)。
    /// 插入即留(与 `bytes`/`names` 同律:峰值快照里的块可能已死,报告仍要能画)
    refs: std::collections::HashMap<u64, (u32, u32)>,
}

static PROBE: std::sync::Mutex<Option<ProbeState>> = std::sync::Mutex::new(None);
/// `probe_refs` 被调用的次数(诊断面:**证明"求值器真的在推"** —— 端到端
/// 断言不该依赖日志级别/报告渲染;`probe_on` 清零)
static REFS_PUSHED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// 最后一次 `probe_end` 的归因报告(文本;调用方取走后自行打日志)
static PROBE_REPORT: std::sync::Mutex<String> = std::sync::Mutex::new(String::new());

// ============================================================================
// §4 活性回收账(诊断;2026-10-16 —— “活着一大片却一字节没回收”的指认面)
// ============================================================================
//
// 探针(`slab 峰值归因`)只答"多少活着",不答"为什么没回收"。当存活构成与分配
// 总量构成**逐名同比例**(= 一字节没 free)时,唯一要问的是"回收出口的哪个判定
// 把块拦下了" —— 本模块给每个出口各记一笔,一行日志指认。
//
// 复位口径与探针同窗(`probe_on(true)` 清台)⇒ 与"slab 峰值归因"同源可比。
// 增量为 relaxed 原子加(每事件 ~1ns,常开不影响热路径形状)。
pub mod live_diag {
    use std::sync::atomic::{AtomicU64, Ordering};

    /// 新块登记(`EvalCtx::track_new`;竞技场原生块)
    pub static NEW_BLOCKS: AtomicU64 = AtomicU64::new(0);
    /// 真回收(`release_block` 走到 `face.free` 的块数;池命中不算)
    pub static FREE_BLOCKS: AtomicU64 = AtomicU64::new(0);
    /// 拒因:父节点还有消费者(`pending > 0`)
    pub static REFUSE_PENDING: AtomicU64 = AtomicU64::new(0);
    /// 拒因:块仍有别名用户(`block_users > 0`)
    pub static REFUSE_USERS: AtomicU64 = AtomicU64::new(0);
    /// 拒因:父块即本节点输出(透传别名)
    pub static REFUSE_ALIAS: AtomicU64 = AtomicU64::new(0);
    /// 拒因:块非竞技场原生(`arena_set` 缺席 —— 池件/外部块)
    pub static REFUSE_NOT_ARENA: AtomicU64 = AtomicU64::new(0);
    /// 拒因:块在 keep 白名单(跨段存活)
    pub static REFUSE_KEPT: AtomicU64 = AtomicU64::new(0);
    /// 退场块不在 `block_users` 里 —— 实测 = **叶子/外部块**(图输入槽、持久件、
    /// 池件的父块退场)的正常路径:它们本就不经 `track_new`/`track_alias`。
    ///
    /// ⚠️ **实测语义(2026-10-17 审查)**:生产 boot 实测 **2651** 次 ⇒
    /// 原文案"记账缺席 —— 不该发生"**误导**(会让人去追一个不存在的 bug)。
    /// 判据:唯一能绕过登记的就是 `Op::Block` 叶子(`Bytes::new(id, …)`)。
    pub static REFUSE_NOT_TRACKED: AtomicU64 = AtomicU64::new(0);
    /// **同一块被 `track_new` 两次**(记账重复 ⇒ users 恒多 1 ⇒ 永不归还;不该发生)
    pub static DOUBLE_NEW: AtomicU64 = AtomicU64::new(0);
    /// 别名登记时该块**已有归属**(`track_alias` 落到已计数块)。
    ///
    /// ⚠️ **实测语义(2026-10-17 审查)**:实现是"`users > 0` 就记一笔"
    /// (`eval.rs::track_alias`),而生产块的常态正是"产者已置 1 → 视图再 +1"
    /// ⇒ 本计数在生产 boot 里**恒非 0**(实测 672),**不是**"不该发生"。
    /// 原文案("同一块被 track_alias 两次以上")与实现不符,属**假现场**一类;
    /// 未改判定(改判定要按块记别名次数 = 新状态),此处先把话写实。
    pub static ALIAS_MULTI: AtomicU64 = AtomicU64::new(0);

    /// 回收出口的拒因(枚举 = 判定分支单点)
    #[derive(Clone, Copy)]
    pub enum Refuse {
        Pending,
        Users,
        Alias,
        NotArena,
        Kept,
        NotTracked,
    }

    pub fn new_block() {
        NEW_BLOCKS.fetch_add(1, Ordering::Relaxed);
    }

    /// `track_new` 的**重复登记**检测(`insert` 返回 false = 已在竞技场里);
    /// 前 12 例连 id + 名字上屏(直指来源:哪个张量/哪条臂)
    pub fn double_new(id: u64, node_id: u64, overrides: usize, memo_len: usize) {
        let n = DOUBLE_NEW.fetch_add(1, Ordering::Relaxed);
        if n < 12 {
            let name = crate::slab_hint::probe_name_of(id)
                .unwrap_or_else(|| "(未命名)".to_string());
            eprintln!(
                "[live-dup] block={id} node={node_id} overrides={overrides} memo={memo_len} name={name}"
            );
        }
    }

    /// 别名重复登记
    pub fn alias_multi() {
        ALIAS_MULTI.fetch_add(1, Ordering::Relaxed);
    }

    pub fn freed() {
        FREE_BLOCKS.fetch_add(1, Ordering::Relaxed);
    }

    /// **回收出口按名分账**(2026-10-16;定谳面):只记**决定点**
    /// (`pending==0` 那一瞬)—— 即"这个块本来可以还了,最后结果是什么"。
    /// 前 k-1 次消费**(仍有消费者)不算决定点,不入账(否则重复计)。
    /// 账目 = (字节, 次数, 残余 users 最大值) 按 (出口, 名字) 聚合 ⇒
    /// 一行报告就能指认"谁被 users 抬着 / 谁平白无故活着"。
    pub static BY_NAME: std::sync::Mutex<std::collections::BTreeMap<(u8, String), (u64, u64, u64)>> =
        std::sync::Mutex::new(std::collections::BTreeMap::new());

    fn code(r: Option<Refuse>) -> u8 {
        match r {
            None => 0,
            Some(Refuse::Pending) => 1,
            Some(Refuse::Users) => 2,
            Some(Refuse::Alias) => 3,
            Some(Refuse::NotArena) => 4,
            Some(Refuse::Kept) => 5,
            Some(Refuse::NotTracked) => 6,
        }
    }

    fn label(c: u8) -> &'static str {
        match c {
            0 => "真回收",
            1 => "pending",
            2 => "users",
            3 => "alias",
            4 => "非竞技场",
            5 => "keep",
            _ => "未登记",
        }
    }

    /// 决定点落账(`reason = None` = 真回收;`left_u` = 该刻残余 users)
    pub fn outcome(block_id: u64, bytes: usize, reason: Option<Refuse>, left_u: usize) {
        let name = crate::slab_hint::probe_name_of(block_id)
            .unwrap_or_else(|| "(未命名)".to_string());
        if let Ok(mut m) = BY_NAME.lock() {
            let e = m.entry((code(reason), name)).or_insert((0, 0, 0));
            e.0 += bytes as u64;
            e.1 += 1;
            e.2 = e.2.max(left_u as u64);
        }
    }

    /// 按名分账的一行摘要(报告尾部追加;空 = 无账)
    pub fn by_name_report() -> String {
        let m = match BY_NAME.lock() {
            Ok(m) => m,
            Err(_) => return String::new(),
        };
        if m.is_empty() {
            return String::new();
        }
        let mut parts: Vec<String> = Vec::new();
        for c in [0u8, 2, 1, 3, 4, 5, 6] {
            let rows: Vec<(&String, &(u64, u64, u64))> =
                m.iter().filter(|((k, _), _)| *k == c).map(|((_, n), v)| (n, v)).collect();
            if rows.is_empty() {
                continue;
            }
            let tot: u64 = rows.iter().map(|(_, v)| v.0).sum();
            let cnt: u64 = rows.iter().map(|(_, v)| v.1).sum();
            let mut top: Vec<(&String, &(u64, u64, u64))> = rows.clone();
            top.sort_unstable_by_key(|(_, v)| std::cmp::Reverse(v.0));
            let tops = top
                .iter()
                .take(4)
                .map(|(n, v)| format!("{n} {:.0}MB×{}", v.0 as f64 / 1048576.0, v.1))
                .collect::<Vec<_>>()
                .join(" / ");
            let maxu = rows.iter().map(|(_, v)| v.2).max().unwrap_or(0);
            parts.push(format!(
                "{} {:.0}MB×{cnt}(残余users≤{maxu}): {tops}",
                label(c),
                tot as f64 / 1048576.0
            ));
        }
        format!(" ‖ 回收出口按名: {}", parts.join(" ｜ "))
    }

    pub fn refuse(r: Refuse) {
        let _ = match r {
            Refuse::Pending => REFUSE_PENDING.fetch_add(1, Ordering::Relaxed),
            Refuse::Users => REFUSE_USERS.fetch_add(1, Ordering::Relaxed),
            Refuse::Alias => REFUSE_ALIAS.fetch_add(1, Ordering::Relaxed),
            Refuse::NotArena => REFUSE_NOT_ARENA.fetch_add(1, Ordering::Relaxed),
            Refuse::Kept => REFUSE_KEPT.fetch_add(1, Ordering::Relaxed),
            Refuse::NotTracked => REFUSE_NOT_TRACKED.fetch_add(1, Ordering::Relaxed),
        };
    }

    /// 清台(与 `probe_on(true)` 同步)
    pub fn reset() {
        if let Ok(mut m) = BY_NAME.lock() {
            m.clear();
        }
        for c in [
            &NEW_BLOCKS,
            &FREE_BLOCKS,
            &REFUSE_PENDING,
            &REFUSE_USERS,
            &REFUSE_ALIAS,
            &REFUSE_NOT_ARENA,
            &REFUSE_KEPT,
            &REFUSE_NOT_TRACKED,
        ] {
            c.store(0, Ordering::Relaxed);
        }
    }

    /// **非消费快照**(2026-10-17 审查补):上面 `take_report` 是"读后清零",
    /// 而它只在**图规划期**被取一次 —— 之后才发生的重复登记/拒因就再没有出口
    /// (实测:一次 boot 的 `[live-dup]` 刷了 12 条,而报告里 `重复记账:new 0`)。
    /// 本函数只读不清,供 `/debug/metrics` 之类的**随时可查**面使用
    /// ⇒ A1 那类"记账重复"回归从此可断言(counter == 0)。
    pub fn counters() -> Vec<(&'static str, u64)> {
        let g = |c: &AtomicU64| c.load(Ordering::Relaxed);
        vec![
            ("live.new_blocks", g(&NEW_BLOCKS)),
            ("live.free_blocks", g(&FREE_BLOCKS)),
            ("live.refuse.pending", g(&REFUSE_PENDING)),
            ("live.refuse.users", g(&REFUSE_USERS)),
            ("live.refuse.alias", g(&REFUSE_ALIAS)),
            ("live.refuse.not_arena", g(&REFUSE_NOT_ARENA)),
            ("live.refuse.kept", g(&REFUSE_KEPT)),
            ("live.retire.untracked", g(&REFUSE_NOT_TRACKED)),
            // **非 0 即异常**(块被登记两遍 ⇒ users 恒多 1 ⇒ 永不中途回收)
            ("live.dup_new", g(&DOUBLE_NEW)),
            ("live.alias.on_tracked", g(&ALIAS_MULTI)),
        ]
    }

    /// 一行账(**读后清零**;与 `slab 峰值归因` 同窗 ⇒ 同源可比)
    pub fn take_report() -> String {
        let g = |c: &AtomicU64| c.swap(0, Ordering::Relaxed);
        format!(
            "新块 {} / 真回收 {}(拒:pending {} users {} alias {} 非竞技场 {} keep {} 未登记 {})／重复记账:new {} alias {}",
            g(&NEW_BLOCKS),
            g(&FREE_BLOCKS),
            g(&REFUSE_PENDING),
            g(&REFUSE_USERS),
            g(&REFUSE_ALIAS),
            g(&REFUSE_NOT_ARENA),
            g(&REFUSE_KEPT),
            g(&REFUSE_NOT_TRACKED),
            g(&DOUBLE_NEW),
            g(&ALIAS_MULTI),
        ) + &by_name_report()
    }
}

/// 归因报告的 top-N 条数
const ATTR_TOP: usize = 8;

/// 探针开/关(GraphPlan warmup 窗口;关时零开销)
pub fn probe_on(on: bool) {
    // 活性账与探针**同窗**:开探针 = 清台(否则上一个窗口的拒因混进来)
    live_diag::reset();
    REFS_PUSHED.store(0, Ordering::Relaxed);
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
                refs: std::collections::HashMap::new(),
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

/// **逐块引用计数推送**(求值器在 `block_users`/`pending` 改动点调用;
/// 2026-10-16)。探针关 = 零动作(一次 relaxed 原子读)。
///
/// 为什么由求值器推而不是探针自己算:两个计数都是**求值器的活状态**
/// (CPU 侧 HashMap),探针只持"快照"。推送点 = 改动点(三处:
/// `track_new`/`track_alias`/死亡减一),每次 ~1ns 哈希写 —— 与既有
/// `live_diag` 原子计数同级。
pub fn probe_refs(id: u64, users: usize, pending: usize) {
    if !PROBE_ON.load(Ordering::Relaxed) {
        return;
    }
    REFS_PUSHED.fetch_add(1, Ordering::Relaxed);
    if let Ok(mut g) = PROBE.lock() {
        if let Some(st) = g.as_mut() {
            st.refs.insert(id, (users as u32, pending as u32));
        }
    }
}

/// 引用计数推送次数(诊断/门面;探针关时恒 0)。
///
/// 存在的意义:让"求值器在改动点推了引用计数"**可机械判定** ——
/// 报告渲染受日志级别影响,计数不受。
pub fn refs_pushed() -> u64 {
    REFS_PUSHED.load(Ordering::Relaxed)
}

/// 读某块的**引用计数快照** `(users, pending)`(诊断/测试面;探针关 = None)。
///
/// 与 [`probe_refs`] 配对:求值器推、外部读。存在的意义是让"求值器真的推了"
/// 这件事**可判定**(端到端 e2e 里日志可能被级别过滤,机械探针不受影响)。
pub fn probe_refs_of(id: u64) -> Option<(u32, u32)> {
    let g = PROBE.lock().ok()?;
    g.as_ref().and_then(|st| st.refs.get(&id).copied())
}

/// 探针登记过的块名(诊断:回收出口按名分账用;未登记/探针关 = None)
pub fn probe_name_of(id: u64) -> Option<String> {
    let g = PROBE.lock().ok()?;
    g.as_ref().and_then(|st| st.names.get(&id).cloned())
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
    // **未命名块画像**(诊断;2026-10-16):"活跃构成里冒出一砦未命名"时必须能一眼
    // 看出它是**哪种张量**(尺寸直方图)与**什么时候分配的**(块 id 单调 ⇒ 时序面)。
    // 本轮就是靠它把段化图的 2× 峰值定到 "1452 块 ~90KB 未命名"上。
    let uname_part = {
        let mut un: Vec<(u64, usize)> = st
            .peak_live_ids
            .iter()
            .filter(|id| !st.names.contains_key(id))
            .map(|id| (*id, *st.bytes.get(id).unwrap_or(&0)))
            .collect();
        if un.is_empty() {
            String::new()
        } else {
            un.sort_unstable_by_key(|(id, _)| *id);
            let total: usize = un.iter().map(|(_, n)| *n).sum();
            let mut hist: std::collections::BTreeMap<usize, usize> =
                std::collections::BTreeMap::new();
            for (_, n) in &un {
                *hist.entry(*n).or_insert(0) += 1;
            }
            let mut hs: Vec<(usize, usize)> = hist.into_iter().collect();
            hs.sort_unstable_by_key(|(_, c)| std::cmp::Reverse(*c));
            let top = hs
                .iter()
                .take(3)
                .map(|(sz, c)| format!("{sz}B×{c}"))
                .collect::<Vec<_>>()
                .join(" / ");
            format!(
                " ‖ ⚠未命名块画像:{} 块 {:.1}MB;id ∈ [{}, {}];尺寸 top3 = {top}",
                un.len(),
                total as f64 / 1048576.0,
                un[0].0,
                un[un.len() - 1].0
            )
        }
    };
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
    ) + &uname_part
        + &render_refs(st)
}

/// **"它凭什么是活的"**(逐块引用计数视图;2026-10-16)。
///
/// 输入 = 求值器在改动点推的 `(users, pending)` 快照。三条判读:
/// - **用户峰值前 N**:按 `users` 降序取前几块(谁被最多消费者抬着);
///   users 越大越说明"这一块的寿命被谁决定" —— 想缩峰值就去找那个节点。
/// - **users 直方图**:按字节/块数分桶(1 / 2–3 / ≥4)—— 快速看"大部分
///   内存是被独占的还是一堆共享的"。
/// - **⚠ 幽灵块**:`users == 0 && 已死` 之外**仍计入活跃**的块(不该出现:
///   归零那一刻就该走回收出口)。它非 0 = 活性记账漏了口(历史上真出过)。
fn render_refs(st: &ProbeState) -> String {
    if st.refs.is_empty() {
        return String::new();
    }
    // 只统计**当前活跃**的块(已死块留在表里是为了渲染,不该进引用分布)
    // 只统计**此刻仍在账**的块:峰值集合是**历史快照**(高水位那一刻),而
    // `refs` 是**最新**计数 —— 直接相交会把"峰值之后正常死掉的块"读成幽灵
    // (2026-10-16 收尾轮实测:engine 离线图因此报关 2 块假红)。
    // 判据改成 "峰值集 ∩ 当前账"(= 归零了却**还活着**才是漏口)。
    let live: Vec<(u64, (u32, u32))> = st
        .peak_live_ids
        .iter()
        .filter(|id| st.ids.contains_key(*id))
        .filter_map(|id| st.refs.get(id).map(|r| (*id, *r)))
        .collect();
    if live.is_empty() {
        return String::new();
    }
    let name_of = |id: u64| -> String {
        let n = st.names.get(&id).cloned().unwrap_or_else(|| "(未命名)".to_string());
        let bytes = *st.bytes.get(&id).unwrap_or(&0);
        format!("{n}({:.0}MB)", bytes as f64 / 1048576.0)
    };
    // ① 用户峰值前 3
    let mut top: Vec<(u64, (u32, u32))> = live.clone();
    top.sort_by(|a, b| b.1.0.cmp(&a.1.0));
    let tops = top
        .iter()
        .take(3)
        .map(|(id, (u, p))| format!("{}:u{u}/p{p}", name_of(*id)))
        .collect::<Vec<_>>()
        .join(" / ");
    // ② users 直方图(块数 + 字节)
    let mut buckets = [(0usize, 0u64); 3]; // 1 / 2-3 / >=4
    for (id, (u, _)) in &live {
        let bytes = *st.bytes.get(id).unwrap_or(&0) as u64;
        let idx = match *u {
            0 | 1 => 0,
            2..=3 => 1,
            _ => 2,
        };
        buckets[idx].0 += 1;
        buckets[idx].1 += bytes;
    }
    // ③ 幽灵块(users==0 却在活跃集里)
    let ghosts: Vec<&(u64, (u32, u32))> = live.iter().filter(|(_, (u, _))| *u == 0).collect();
    let ghost_part = if ghosts.is_empty() {
        String::new()
    } else {
        let names = ghosts
            .iter()
            .take(3)
            .map(|(id, _)| name_of(*id))
            .collect::<Vec<_>>()
            .join(" / ");
        format!(
            " ‖ ⚠ 幽灵块 {} 个(users=0 却仍活跃 ⇒ 回收记账漏口):{names}",
            ghosts.len()
        )
    };
    format!(
        " ‖ 引用分布({} 块):u=1 {}块/{:.0}MB, u=2-3 {}块/{:.0}MB, u≥4 {}块/{:.0}MB ‖ users 前3:{tops}{ghost_part}",
        live.len(),
        buckets[0].0,
        buckets[0].1 as f64 / 1048576.0,
        buckets[1].0,
        buckets[1].1 as f64 / 1048576.0,
        buckets[2].0,
        buckets[2].1 as f64 / 1048576.0,
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

    /// **引用分布视图**(2026-10-16):"它凭什么是活的" —— users 桶 / users 前 3 /
    /// 幽灵块(users=0 却仍活跃 = 记账漏口)。三条判据各一条断言。
    #[test]
    fn probe_refs_view_buckets_top_and_ghosts() {
        let _g = PROBE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _ = take_probe_report();
        probe_on(true);
        // 三块进峰值集(尺寸 1MiB/1MiB/2MiB),各推不同引用计数
        probe_alloc(11, 1 << 20);
        probe_name(11, "kv.a");
        probe_alloc(12, 1 << 20);
        probe_name(12, "kv.b");
        probe_alloc(13, 2 << 20);
        probe_name(13, "kv.c");
        probe_refs(11, 1, 0); // u=1 桶
        probe_refs(12, 3, 2); // u=2-3 桶,且 pending 非零
        probe_refs(13, 5, 0); // u>=4 桶
        let peak = probe_end();
        assert!(peak > 0);
        let rep = take_probe_report();
        assert!(rep.contains("引用分布(3 块)"), "分布段必须在:{rep}");
        assert!(rep.contains("u=1 1块/1MB"), "u=1 桶:{rep}");
        assert!(rep.contains("u=2-3 1块/1MB"), "u=2-3 桶:{rep}");
        assert!(rep.contains("u≥4 1块/2MB"), "u≥4 桶:{rep}");
        assert!(rep.contains("users 前3:kv.c(2MB):u5/p0"), "users 降序前 3:{rep}");
        assert!(!rep.contains("幽灵块"), "无幽灵时不得出现警报:{rep}");
    }

    /// **幽灵块必须响亮**:users 归零却仍在活跃集 = 回收记账漏口(历史真出过),
    /// 这条是那个不变式的负例门。
    #[test]
    fn probe_refs_ghost_blocks_are_loud() {
        let _g = PROBE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _ = take_probe_report();
        probe_on(true);
        probe_alloc(21, 1 << 20);
        probe_name(21, "leaky.block");
        probe_refs(21, 0, 0); // u=0 却仍活 ⇒ 幽灵
        let _ = probe_end();
        let rep = take_probe_report();
        assert!(rep.contains("⚠ 幽灵块 1 个"), "幽灵必须显形:{rep}");
        assert!(rep.contains("leaky.block"), "幽灵要点名:{rep}");
    }

    /// **幽灵判据的时间基**(2026-10-16 收尾轮):峰值集合 = 高水位那一刻的
    /// 快照(历史),`refs` = 最新计数 ⇒ "峰值后正常走完生命周期"的块不许被
    /// 读成幽灵(engine 离线图曾因此报 2 块假红)。判据 = 峰值集 ∩ **当前账**。
    #[test]
    fn dead_after_peak_is_not_a_ghost() {
        let _g = PROBE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _ = take_probe_report();
        probe_on(true);
        probe_alloc(31, 1 << 20);
        probe_name(31, "died.after.peak");
        probe_refs(31, 1, 0);
        probe_alloc(32, 1 << 20);
        probe_name(32, "alive.healthy");
        probe_refs(32, 2, 0);
        // 高水位拍下 {31,32} 之后,31 正常走完生命周期:归零(死) + 真回收
        // (`probe_end` 渲染那一刻才算报告 ⇒ 死亡必须在它之前发生)
        probe_refs(31, 0, 0);
        probe_free(31);
        let _ = probe_end();
        let rep = take_probe_report();
        assert!(rep.contains("引用分布(1 块)"), "只剩健康块:{rep}");
        assert!(!rep.contains("幽灵块"), "峰值后正常死掉的块不是幽灵:{rep}");
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

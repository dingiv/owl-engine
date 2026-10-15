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

/// slab 内分配器:bump + **尺寸类回收**(对齐 256B;与旧 carve 逐字同对齐)
#[derive(Default, Debug)]
pub struct SlabArena {
    used: usize,
    high: usize,
    /// class(ceil_log2(cap_n))→ [(off, cap_n)]
    free: std::collections::HashMap<u32, Vec<(usize, usize)>>,
}

const ARENA_ALIGN: usize = 256;

fn align_up(n: usize) -> usize {
    n.div_ceil(ARENA_ALIGN) * ARENA_ALIGN
}

fn class_of(n: usize) -> u32 {
    let p = n.max(1).next_power_of_two();
    usize::BITS - p.leading_zeros() - 1
}

impl SlabArena {
    /// 切一块 ≥ `n`(返回 `(off, cap_n)`;`cap_n` = 该块的可用容量,回收凭它)
    pub fn carve(&mut self, n: usize, slab_cap: usize) -> Option<(usize, usize)> {
        let need = align_up(n);
        // 尺寸类最佳匹配(取 ≥ need 里最小者;避免大切小用)
        let mut best: Option<(u32, usize)> = None; // (class, idx)
        for (c, v) in self.free.iter() {
            if *c < class_of(need) {
                continue;
            }
            for (i, (_, cap_n)) in v.iter().enumerate() {
                if *cap_n >= need {
                    let better = match best {
                        None => true,
                        Some((bc, bi)) => self.free[&bc][bi].1 > *cap_n,
                    };
                    if better {
                        best = Some((*c, i));
                    }
                }
            }
        }
        if let Some((c, i)) = best {
            let (off, cap_n) = self.free.get_mut(&c).unwrap().swap_remove(i);
            let rest = cap_n - need;
            if rest >= ARENA_ALIGN {
                self.release(off + need, rest);
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

    /// 归还(捕获窗内 `free_blocks` 对 Carved 块调用)
    pub fn release(&mut self, off: usize, cap_n: usize) {
        if cap_n == 0 {
            return;
        }
        self.free.entry(class_of(cap_n)).or_default().push((off, cap_n));
    }

    /// 高水位 = 本 arena 需要的 slab 容量(探针口径)
    pub fn high(&self) -> usize {
        self.high
    }
}

static PROBE_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static PROBE: std::sync::Mutex<Option<(SlabArena, std::collections::HashMap<u64, (usize, usize)>)>> =
    std::sync::Mutex::new(None);

/// 探针开/关(GraphPlan warmup 窗口;关时零开销)
pub fn probe_on(on: bool) {
    if let Ok(mut g) = PROBE.lock() {
        if on {
            *g = Some((SlabArena::default(), std::collections::HashMap::new()));
        } else {
            *g = None;
        }
        PROBE_ON.store(on, Ordering::Relaxed);
    }
}

/// 影子分配(warmup 的真分配点调用;探针关 = 立即返回)
pub fn probe_alloc(id: u64, n_bytes: usize) {
    if !PROBE_ON.load(Ordering::Relaxed) {
        return;
    }
    if let Ok(mut g) = PROBE.lock() {
        if let Some((arena, ids)) = g.as_mut() {
            // slab 容量按"无上限"跑,只求高水位(真容量由 high 决定)
            if let Some((off, cap_n)) = arena.carve(n_bytes, usize::MAX) {
                ids.insert(id, (off, cap_n));
            }
        }
    }
}

/// 影子归还(warmup 的 free 点;未登记/探针关 = 无动作)
pub fn probe_free(id: u64) {
    if !PROBE_ON.load(Ordering::Relaxed) {
        return;
    }
    if let Ok(mut g) = PROBE.lock() {
        if let Some((arena, ids)) = g.as_mut() {
            if let Some((off, cap_n)) = ids.remove(&id) {
                arena.release(off, cap_n);
            }
        }
    }
}

/// 关探针并取高水位(= 捕获所需 slab 容量)
pub fn probe_end() -> usize {
    PROBE_ON.store(false, Ordering::Relaxed);
    let mut peak = 0;
    if let Ok(mut g) = PROBE.lock() {
        if let Some((arena, _)) = g.as_ref() {
            peak = arena.high();
        }
        *g = None;
    }
    peak
}

#[cfg(test)]
mod arena_tests {
    use super::*;

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
}

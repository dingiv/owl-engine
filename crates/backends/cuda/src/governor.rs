//! 治理句柄：相位机 + 双账本 + 预算 + 延迟队列（A1.2/A5）。
//! P1-2（2026-09-24）：捕获窗口的缓冲钉住账在此——窗口内出生的块
//! （capture_borns，出生即强持，防 Arc 提前归零）与窗口内死亡的块
//! （capture_parked，释放闭包停放，地址可能已烙进图）都由
//! CaptureSession 在定影时收编进 DeviceGraph keepalive。

use super::pool::PoolBufInner;
use cudarc::driver::CudaContext;
use owl_iface::{BufToken, MemPhase, MemStats};
use parking_lot::{Mutex, RwLock};
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Weak};

pub(crate) type DeferredFree = Box<dyn FnOnce() + Send>;

#[derive(Default)]
pub(crate) struct Governor {
    pub(crate) ctx: Option<Arc<CudaContext>>,
    pub(crate) phase: RwLock<MemPhase>,
    pub(crate) deferred: Mutex<Vec<DeferredFree>>,
    /// 捕获窗口相栈(push_phase/restore_phase 作用域用)
    pub(crate) phase_stack: Mutex<Vec<MemPhase>>,
    pub(crate) stats: Mutex<MemStats>,
    /// 池注册表（2026-09-24 破环：强注册表移 DeviceInner——Governor 持
    /// Arc<CudaPool> 且 CudaPool 持 Arc<Governor> 成环，池/账本永不可回收；
    /// 根 = 设备，Device 强持池、池强持治理，治理不再回指池）
    pub(crate) next_pool_id: AtomicU64,
    /// 哨兵①词汇:缓冲令牌发放与存活登记(id → gen)
    pub(crate) next_buf_id: AtomicU64,
    pub(crate) alive: Mutex<HashMap<u64, u64>>,
    /// A5.2/哨兵①:存活池缓冲登记表(id → Weak;仅身份/反查用途——
    /// 强保活由租约持有方(GraphLease keepalive)负责;登记表若持强引用,
    /// "最后一个句柄"判定永不成立 → 账本泄漏(2026-09-22 判例)
    pub(crate) live_bufs: Mutex<HashMap<u64, Weak<PoolBufInner>>>,
    /// 哨兵①:设备地址区间索引(base → (len, buf_id));emit_by_ptr 反查
    pub(crate) intervals: Mutex<BTreeMap<u64, (u64, u64)>>,
    /// 姿势 6(warmup 门禁):本设备 eager kernel 发射计数——
    /// "effect 未完整执行过一次不可 seal"的 M1 形态(capture 前必须 >0)
    pub(crate) eager_launches: AtomicU64,
    /// A5 预算播种种子（set_budget 广播到全部池账本；新池建账时继承。
    /// 预算是设备级策略，账本本体在各池 Ledger 上）
    pub(crate) budget_seed: Mutex<Option<Budget>>,
    /// P1-2：捕获窗口出生钉住账（出生即强持 → 窗口内 Arc 不可能归零；
    /// CaptureSession 定影时按 watermark 收编进图 keepalive）
    pub(crate) capture_borns: Mutex<Vec<Arc<PoolBufInner>>>,
    /// P1-2：捕获窗口死亡停放账（pre-window 出生、窗口内归零者的释放
    /// 闭包；定影时随图走，图 drop 才真正释放——地址已烙进图）
    pub(crate) capture_parked: Mutex<Vec<DeferredFree>>,
    /// P0-4 毒化探测器开关（OWL_POISON_FREED=1；归还即填 0xFF，悬空读
    /// 立即现形。P1-2 修复后接线，测试可覆写）
    pub(crate) poison_freed: AtomicBool,
}

impl Governor {
    pub(crate) fn phase(&self) -> MemPhase {
        *self.phase.read()
    }

    /// 捕获窗口相压栈(push Capturing;窗口结束弹出恢复调用方原相)。
    /// 与 set_phase 分立:窗口是作用域语义,不应把用户设的 Live 永久覆盖。
    pub(crate) fn push_phase(&self, phase: MemPhase) {
        self.phase_stack.lock().push(*self.phase.read());
        *self.phase.write() = phase;
    }

    pub(crate) fn restore_phase(&self) {
        if let Some(prev) = self.phase_stack.lock().pop() {
            *self.phase.write() = prev;
        }
    }

    pub(crate) fn set_phase(&self, phase: MemPhase) {
        *self.phase.write() = phase;
        if phase == MemPhase::Idle {
            // 净空窗口:统一归还延迟队列(A1.2);归还动作由各缓冲自己的
            // 闭包完成(含池账 + 全局账)
            let mut q = self.deferred.lock();
            let n = q.len();
            for free in q.drain(..) {
                free();
            }
            self.stats.lock().drained_frees += n as u64;
        }
    }

    pub(crate) fn defer(&self, free: DeferredFree) {
        self.stats.lock().deferred_frees += 1;
        self.deferred.lock().push(free);
    }

    // ---- P1-2：捕获窗口钉住账 ----

    pub(crate) fn is_capturing(&self) -> bool {
        self.phase() == MemPhase::Capturing
    }

    /// 窗口内出生的块：出生即强持（防 Arc 在窗口内归零→释放→图悬空）
    pub(crate) fn pin_capture_born(&self, inner: Arc<PoolBufInner>) {
        self.capture_borns.lock().push(inner);
    }

    /// 窗口水位线（capture 开始时打点；收编 [mark..)）
    pub(crate) fn capture_born_mark(&self) -> usize {
        self.capture_borns.lock().len()
    }

    /// 收编 mark 之后的出生钉住账（split_off：保留 [..mark] 给更早窗口）
    pub(crate) fn take_capture_borns_from(&self, mark: usize) -> Vec<Arc<PoolBufInner>> {
        self.capture_borns.lock().split_off(mark)
    }

    /// 窗口内死亡的块：停放释放闭包（图 drop 才执行）
    pub(crate) fn park_capture(&self, free: DeferredFree) {
        self.capture_parked.lock().push(free);
    }

    pub(crate) fn take_capture_parked(&self) -> Vec<DeferredFree> {
        std::mem::take(&mut *self.capture_parked.lock())
    }

    /// 捕获失败清理：弃出生钉 + 执行停放闭包（无图存活，释放安全）。
    /// 顺序关键：先弃 borns（归零者会 park），再统一执行 parked。
    pub(crate) fn discard_capture_state(&self, born_mark: usize) {
        drop(self.take_capture_borns_from(born_mark));
        for f in self.take_capture_parked() {
            f();
        }
    }

    /// P0-4 毒化探测器（测试/诊断可覆写；默认取 OWL_POISON_FREED）
    pub(crate) fn set_poison_freed(&self, on: bool) {
        self.poison_freed.store(on, Ordering::Relaxed);
    }

    /// 哨兵①:签发缓冲令牌并登记存活
    pub(crate) fn issue_token(&self) -> BufToken {
        let id = self.next_buf_id.fetch_add(1, Ordering::Relaxed);
        let gen = id.wrapping_mul(2654435761) | 1;
        self.alive.lock().insert(id, gen);
        BufToken { id, gen }
    }

    /// 哨兵①:世代校验(replay 前检查;死亡令牌 = 结构化报错的依据)
    pub(crate) fn validate(&self, t: &BufToken) -> bool {
        self.alive.lock().get(&t.id).is_some_and(|&g| g == t.gen)
    }

    /// 哨兵①:令牌注销(drop 即死;延迟的只是物理回收,不是身份)。
    /// 同时清理 Weak 登记表条目(防 live_bufs 无界累积)。
    pub(crate) fn retire(&self, t: &BufToken) {
        self.alive.lock().remove(&t.id);
        self.live_bufs.lock().remove(&t.id);
    }

    /// 姿势 6:记一次 kernel 发射(eager 与捕获期都计;门禁只认 >0)
    pub(crate) fn note_launch(&self) {
        self.eager_launches.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn eager_launches(&self) -> u64 {
        self.eager_launches.load(Ordering::Relaxed)
    }

}

/// A5 硬预算:启动时声明,生命周期恒不超(见 charter 公理 A5)。
#[derive(Debug, Clone, Copy)]
pub struct Budget {
    pub bytes: u64,
    pub reserve_floor: u64,
}

/// A5.4 账本快照:违约/诊断时的完整内存叙事
/// (2026-09-23 裁决:字节账本在池上;快照由 Device 聚合默认池账本 +
/// 设备侧计数器拼装)
#[derive(Debug, Clone, Copy)]
pub struct LedgerSnapshot {
    pub bytes_alive: u64,
    pub bytes_allocated_total: u64,
    pub stats: MemStats,
    pub phase: MemPhase,
}

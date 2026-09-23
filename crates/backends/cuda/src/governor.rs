//! 治理句柄:相位机 + 双账本 + 预算 + 延迟队列(A1.2/A5)。

use super::pool::{CudaPool, PoolBufInner};
use cudarc::driver::CudaContext;
use owl_iface::{BufToken, MemPhase, MemStats};
use parking_lot::{Mutex, RwLock};
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
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
    /// 池注册表(2026-09-23 裁决:账本归池持有,Governor 只存身份索引供
    /// Device::pool(id) 反查;Arc 强持 = 池注册表为设备生命周期账,
    /// 与命名池进程级常驻语义一致)
    pub(crate) pools: Arc<Mutex<HashMap<u64, Arc<CudaPool>>>>,
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
    /// A5 预算播种种子(set_budget 广播到全部池账本;新池建账时继承。
    /// 预算是设备级策略,账本本体在各池 Ledger 上)
    pub(crate) budget_seed: Mutex<Option<Budget>>,
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

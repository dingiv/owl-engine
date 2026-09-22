//! 治理句柄:相位机 + 双账本 + 预算 + 延迟队列(A1.2/A5)。

use super::pool::{CudaPool, PoolBufInner, PoolLedger};
use cudarc::driver::CudaContext;
use owl_iface::{BackendError, BufToken, MemPhase, MemStats};
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
    /// A5.2 全局账本:存活字节 / 累计分配字节
    pub(crate) bytes_alive: Mutex<u64>,
    pub(crate) bytes_allocated_total: Mutex<u64>,
    /// A5.1 预算(None = 未设;设置后每次分配断言不超)
    pub(crate) budget: Mutex<Option<Budget>>,
    /// 显存池账本(A5.2);Arc 供池对象跨线程归账
    pub(crate) pools: Arc<Mutex<HashMap<u64, PoolLedger>>>,
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

    pub(crate) fn charge(&self, bytes: u64) -> Result<(), BackendError> {
        // 校验先行:预算超支直接拒绝,**不触碰账本**(调用方失败路径
        // 无需回滚——2026-09-22 双重扣减下溢判例)
        if let Some(b) = *self.budget.lock() {
            let cur = *self.bytes_alive.lock();
            if cur + bytes > b.bytes {
                return Err(BackendError::LawViolation(
                    "A5.4 显存超支(引擎账本超预算,详见账本快照日志)",
                ));
            }
            // A5.3 第三道闸:driver 侧对账,free 低于警线即告警
            if let Some(ctx) = &self.ctx {
                let (free, _total) = ctx
                    .mem_get_info()
                    .map_err(|e| BackendError::Init(format!("{e:?}")))?;
                if (free as u64) < b.reserve_floor {
                    eprintln!(
                        "[owl-mem] WARN: free {free}B < reserve_floor {}B(A5 警线,账本 alive={})",
                        b.reserve_floor, cur
                    );
                }
            }
        }
        *self.bytes_alive.lock() += bytes;
        *self.bytes_allocated_total.lock() += bytes;
        Ok(())
    }

    pub(crate) fn uncharge(&self, bytes: u64) {
        let mut alive = self.bytes_alive.lock();
        if *alive < bytes {
            eprintln!(
                "[owl-mem][BUG] uncharge underflow: alive={} bytes={} governor={:p}",
                *alive, bytes, self as *const _
            );
        }
        *alive = alive.saturating_sub(bytes);
    }

    pub(crate) fn uncharge_pool(&self, id: u64, bytes: u64) {
        if let Some(led) = self.pools.lock().get_mut(&id) {
            led.used = led.used.saturating_sub(bytes);
        }
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

    /// 池余量校验(A5.4 第一道):池耗尽即违约。通过后调用方已占池账。
    pub(crate) fn charge_pool(&self, pool: &CudaPool, bytes: u64) -> Result<(), BackendError> {
        let mut pools = self.pools.lock();
        let led = pools
            .get_mut(&pool.id.0)
            .ok_or(BackendError::UnknownPool(pool.id.0))?;
        let available = led.capacity - led.used;
        if bytes > available {
            return Err(BackendError::PoolExhausted {
                pool: led.name.clone(),
                needed: bytes,
                available,
                capacity: led.capacity,
            });
        }
        led.used += bytes;
        led.peak = led.peak.max(led.used);
        Ok(())
    }
}

/// A5 硬预算:启动时声明,生命周期恒不超(见 charter 公理 A5)。
#[derive(Debug, Clone, Copy)]
pub struct Budget {
    pub bytes: u64,
    pub reserve_floor: u64,
}

/// A5.4 账本快照:违约/诊断时的完整内存叙事
#[derive(Debug, Clone, Copy)]
pub struct LedgerSnapshot {
    pub bytes_alive: u64,
    pub bytes_allocated_total: u64,
    pub stats: MemStats,
    pub phase: MemPhase,
}

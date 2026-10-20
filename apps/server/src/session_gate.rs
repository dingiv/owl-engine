//! **会话闸**（per-session serialization；B5.1 合批档的前置契约）。
//!
//! 为什么在服务层：引擎在 `[runtime] max_batch > 1` 时对"**同会话**已有活跃/排队
//! turn"**硬拒**（`RunningEngine::submit_session_opts`；同会话两个 turn 会共享同一份
//! KV/GDN 状态 = 结构性互踩，不是调度能修的）。契约是"上层应按会话串行" ——
//! 本模块就是那一层。
//!
//! 语义边界：
//! - **同 session id 串行**（一次只有一个 turn 在引擎里），**跨会话并行** ——
//!   后者正是合批的原料（B 个会话才凑得出一批）；
//! - 闸只在 HTTP 轮次期间持有（提交 → 终态 → 响应写完）；等待期间不占引擎资源、
//!   不占并发 permit（permit 是另一层，见 `ConnCtx::gate`）；
//! - 无会话（`session = None`，临时会话）= 不需要闸（每个 turn 独占临时账）；
//! - 表**有界**：条目是 `Weak`，取不到就重建；每次取用顺手清死项 ⇒ 长期运行不
//!   随会话数增长。
//!
//! ⚠️ 已知产品面后果：`[agent] implicit_session = true` 时所有**无 session 键**的
//! 请求都落隐式会话 0 ⇒ 它们会被本闸串行（合批对这类客户端无收益）。要拿合批
//! 收益，客户端需带 `session` 键（每 agent 一个）。boot 期有配套告警。

use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};

use tokio::sync::{Mutex as AsyncMutex, OwnedMutexGuard};

/// 会话闸表（进程级一份；由 `ConnCtx` 持有，显式传参 —— 不做全局变量）。
#[derive(Default)]
pub struct SessionGates {
    /// 会话 id → 该会话的互斥体（弱引用：无人持有/等待时可回收）
    inner: Mutex<HashMap<u64, Weak<AsyncMutex<()>>>>,
}

impl SessionGates {
    pub fn new() -> Self {
        Self::default()
    }

    /// 取得该会话的独占权（RAII：guard Drop 即释放）。
    ///
    /// 锁毒化：表锁内**没有 panic 面**（只做 map 操作），真毒化也只能来自
    /// 外部 abort —— 取回内层继续用（不 panic，符合"长驻服务不硬退"）。
    pub async fn acquire(&self, session: u64) -> OwnedMutexGuard<()> {
        let m = {
            let mut g = match self.inner.lock() {
                Ok(g) => g,
                Err(poison) => poison.into_inner(),
            };
            // 清死项（有 holder / waiter 的 Weak 才活）：表规模 = 在跑会话数
            g.retain(|_, w| w.strong_count() > 0);
            match g.get(&session).and_then(Weak::upgrade) {
                Some(m) => m,
                None => {
                    let m = Arc::new(AsyncMutex::new(()));
                    g.insert(session, Arc::downgrade(&m));
                    m
                }
            }
        };
        m.lock_owned().await
    }

    /// 当前在册会话数（诊断/门用；`/debug/agent` 之类可读）。
    pub fn tracked(&self) -> usize {
        match self.inner.lock() {
            Ok(g) => g.len(),
            Err(poison) => poison.into_inner().len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// 同会话串行：第二个 acquire 必须等第一个 guard 释放。
    #[tokio::test]
    async fn same_session_is_serialized() {
        let g = Arc::new(SessionGates::new());
        let a = g.acquire(7).await;
        let g2 = g.clone();
        let t = tokio::spawn(async move {
            let _b = g2.acquire(7).await;
            1u8
        });
        // 200ms 内不应完成（闸在 a 手里）
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(!t.is_finished(), "同会话第二次 acquire 不该通过");
        drop(a);
        let v = tokio::time::timeout(Duration::from_secs(2), t)
            .await
            .expect("释放后应立刻通过")
            .expect("join");
        assert_eq!(v, 1);
    }

    /// 跨会话并行：不同 id 互不阻塞。
    #[tokio::test]
    async fn cross_session_is_parallel() {
        let g = Arc::new(SessionGates::new());
        let _a = g.acquire(1).await;
        let g2 = g.clone();
        let t = tokio::spawn(async move {
            let _b = g2.acquire(2).await;
            2u8
        });
        let v = tokio::time::timeout(Duration::from_millis(500), t)
            .await
            .expect("跨会话不该被闸住")
            .expect("join");
        assert_eq!(v, 2);
    }

    /// 表有界：guard 释放后条目可回收（下次取用清死项）。
    #[tokio::test]
    async fn table_does_not_grow_without_bound() {
        let g = SessionGates::new();
        for id in 0..64u64 {
            let _x = g.acquire(id).await;
        }
        // 全部释放后，再取一次触发 retain ⇒ 死项清零
        let _y = g.acquire(999).await;
        assert!(g.tracked() <= 1, "死项应被清掉（tracked={}）", g.tracked());
    }
}

//! **收摊**(2026-10-16 立 / 2026-11-03 重写:不再 `process::exit`)。
//!
//! 三档退出共用**一条**路径:信号 / 引擎致命错 / 设计外 panic。
//!
//! ```text
//! SIGTERM/SIGINT ─────────┐
//! 引擎致命错(is_fatal) ───┼─▶ accept 循环 break ─▶ drop(EngineReq 发送端)
//! owl-engine 线程 panic ──┘        │                        │
//!                                  │                        ▼
//!                                  │              executor 泵循环读到 None ⇒ 返回 ExitReason
//!                                  │                        │
//!                                  └────────────▶ join executor 线程 ◀┘
//!                                                           │
//!                                                    释放卡锁(全 rank)
//!                                                           │
//!                                              main 返回 ExitCode(正常走 Drop)
//! ```
//!
//! **为什么不再 `process::exit`**(2026-11-03 用户裁决):`exit` 跳过 Drop、跳过
//! 栈展开、也跳过 `main` 之后的一切 —— 它是"假装优雅"的兜底,不是收摊。现在
//! 每一步都是**显式**的:关 listener → 关请求通道 → 等 executor 自己停 → 释放卡锁
//! → 返回。退出码由返回值表达(`ExitCode`),supervisor 仍能分辨"该不该重启"。
//!
//! 卡锁仍然显式 `release`:锁判据是**进程 pid 存活**(`gpu_lock::live_locks_in`),
//! 进程正常退出后锁也会被下一次启动判为死锁并回收 —— 但"靠别人回收"是兜底,
//! 不是路径;显式释放让运维日志里能看见"我放干净了"。

use std::path::PathBuf;

/// 收摊上下文(卡锁路径 + 设备 UUID 的日志面)。boot 后只读。
#[derive(Clone, Debug)]
pub struct Shutdown {
    /// **全部 rank** 的卡锁文件(TP 档每 rank 一把)。
    lock_paths: Vec<PathBuf>,
    /// 设备 UUID(日志面)。
    uuid: String,
}

impl Shutdown {
    pub fn new(lock_paths: Vec<PathBuf>, uuid: String) -> Self {
        Self { lock_paths, uuid }
    }

    /// 释放**全部** rank 的卡锁(幂等;失败只 WARN —— 退出路径不许被它挡住)。
    pub fn release_locks(&self) {
        for p in &self.lock_paths {
            owl_shared::gpu_lock::release(p);
        }
        owl_shared::log_info!("gpu-lock", "卡锁已全部释放({} 把)", self.lock_paths.len());
    }

    pub fn uuid(&self) -> &str {
        &self.uuid
    }
}

/// **executor 线程的退出原因**(决定退出码与收摊日志)。
///
/// 语义边界:
/// - [`ExitReason::Normal`]:请求通道关闭(运维停服)⇒ 退出码 **0**;
/// - [`ExitReason::Fatal`]:引擎不可恢复错误(`ModelError::is_fatal`)⇒ **1**;
/// - [`ExitReason::Panic`]:设计外 panic(catch_unwind 兜到)⇒ **70**(EX_SOFTWARE,
///   与"配置错/致命错"分开,supervisor 据此决定是否重启)。
#[derive(Clone, Debug)]
pub enum ExitReason {
    Normal,
    Fatal(String),
    Panic(String),
}

impl ExitReason {
    /// 进程退出码(`sysexits` 习惯:0 正常 / 1 致命 / 70 软件缺陷)。
    pub fn code(&self) -> u8 {
        match self {
            Self::Normal => 0,
            Self::Fatal(_) => 1,
            Self::Panic(_) => 70,
        }
    }

    /// 收摊日志行(空 = 正常停服也只打一行 INFO)。
    pub fn describe(&self) -> String {
        match self {
            Self::Normal => "正常停服(请求通道关闭)".to_string(),
            Self::Fatal(e) => format!("引擎不可恢复错误:{e}"),
            Self::Panic(e) => format!("owl-engine 线程 panic(设计外 —— 应转显式 Result):{e}"),
        }
    }
}

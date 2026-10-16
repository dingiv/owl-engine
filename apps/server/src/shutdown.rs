//! **优雅收摊**(2026-10-16):释放卡锁 + 有序退出(退出码 0)。
//!
//! 两条触发源走**同一条路径**:
//! 1. `SIGTERM`/`SIGINT`(运维停服);
//! 2. **引擎不可恢复错误**(`ModelError::is_fatal`:显存硬预算拒发 / 设备 server 关闭)
//!    —— 泵循环里遇到就跳出并收摊,而不是让进程带着半截状态继续跑。
//!
//! 为什么必须显式:`std::process::exit` **不跑 Drop** ⇒ 卡锁必须显式释放
//! (与 `5981ec4` 礼仪启动/优雅关停同源)。

/// 收摊上下文(卡锁路径 + 设备 UUID)。
#[derive(Clone, Debug)]
pub struct Shutdown {
    /// 卡锁文件路径(`/run/user/<uid>/owl/gpu-locks/...`)。
    pub lock_path: std::path::PathBuf,
    /// 设备 UUID(日志面)。
    pub uuid: String,
}

impl Shutdown {
    /// 收摊:`reason` 上报(日志面),释放卡锁,退出码 0。**不返回**。
    pub fn graceful(&self, reason: &str) -> ! {
        owl_shared::log_warn!(
            "boot",
            "优雅关停:{reason} ⇒ 释放卡锁 {} + 退出(exit 0)",
            self.uuid
        );
        owl_shared::gpu_lock::release(&self.lock_path);
        std::process::exit(0);
    }
}

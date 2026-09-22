//! 引擎层错误与 Result 别名(T2 前置,翻译基座)。
//!
//! 映射(见 candle-api-mapping.md §三):
//! - `candle_core::Result<T>` → [`Result<T>`](Result)
//! - `candle_core::bail!(...)`(369 处,搬运最大宗)→ 本模块 [`bail!`],
//!   保持源码形态最接近、搬运成本最低;
//! - `candle_core::Error::wrap` → `From<BackendError>`;

use owl_iface::BackendError;

/// 引擎层统一错误。变体按搬运面增补;不追求枚举穷尽,
/// Msg 兜底(candle Error::Msg 73 处的直接对应)。
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// `candle_core::Error::Msg` 直译(占搬运错误面 ~70%)
    #[error("{0}")]
    Msg(String),
    /// 后端/账本错误透传(`#[from]` = candle Error::wrap 的直译)
    #[error(transparent)]
    Backend(#[from] BackendError),
    /// 配置/装载类(搬运 config/gguf 层时逐步细分)
    #[error("config: {0}")]
    Config(String),
    /// 调度/序列类(scheduler/runner 搬运期增补)
    #[error("schedule: {0}")]
    Schedule(String),
}

/// 引擎层 Result 别名:搬运点把 `candle_core::Result<T>` 机械替换为本类型。
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// `candle_core::bail!` 直译:搬运点保留原调用形态,
/// 只改 use 路径即可编译(格式化参数语义与 candle 一致)。
#[macro_export]
macro_rules! bail {
    ($msg:literal $(,)?) => {
        return Err($crate::Error::Msg($msg.to_string()))
    };
    ($fmt:expr, $($arg:tt)*) => {
        return Err($crate::Error::Msg(format!($fmt, $($arg)*)))
    };
}

/// `candle_core::ensure!` 直译(条件不满足即 bail)。
#[macro_export]
macro_rules! ensure {
    ($cond:expr, $($arg:tt)*) => {
        if !($cond) {
            $crate::bail!($($arg)*);
        }
    };
}

//! 环境变量统一读取面(env reader;2026-10-10 clean-code 轮立)。
//!
//! **定位**:全 workspace 唯一的 `std::env` 访问口 —— 应用入口、各 crate、
//! 测试用例一律经本模块读环境变量,禁止新增散点直呼 `std::env::var*`。
//! (例外:构建脚本 `build.rs` 属构建期基础设施,不经运行时读取面。)
//!
//! **为什么收口**:
//! - **降级可见**:直呼 `var(k).unwrap_or(d)` 会把 `OWL_DEVICE=abc` 静默
//!   变 0;本模块解析失败路径一律 `eprintln!` 告知(降级 ≠ 无声);
//! - **热路径禁律**(running.rs boot 探针表的执行前提):本模块只提供
//!   读取原语,boot 一次解析、进程内冻结的纪律不因收口而放松;
//! - **测试防踩踏**:`std::env` 是进程全局,并行测试 `set_var` 互踩 ——
//!   新测试一律用 [`EnvGuard`](RAII 设/恢复;同键并行用例仍需外部串行化)。
//!
//! **命名**:键名全文传入(`OWL_*` 家族),本模块不做键名拼接魔法。

use std::ffi::OsString;

/// 探针/开关族:键存在即真(不看值;`OWL_DEBUG` 风格)
pub fn flag(key: &str) -> bool {
    std::env::var_os(key).is_some()
}

/// 字符串值(未设置 = None)
pub fn str(key: &str) -> Option<String> {
    std::env::var(key).ok()
}

/// 字符串值带缺省(未设置 = default)
pub fn str_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

/// 解析值(`T: FromStr`;未设置 = None,解析失败 = None + eprintln 可见)
pub fn parse<T: std::str::FromStr>(key: &str) -> Option<T> {
    let raw = std::env::var(key).ok()?;
    match raw.parse::<T>() {
        Ok(v) => Some(v),
        Err(_) => {
            eprintln!("[env] {key}={raw:?} 解析失败,按未设置处置");
            None
        }
    }
}

/// 解析值带缺省(未设置或解析失败 = default;失败路径 eprintln 可见)
pub fn parse_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    parse(key).unwrap_or(default)
}

/// 写入(测试/引导专用;生产路径零写入 —— 进程配置 boot 后视为冻结)
pub fn set(key: &str, value: &str) {
    std::env::set_var(key, value);
}

/// 移除(测试专用;配对 [`set`])
pub fn remove(key: &str) {
    std::env::remove_var(key);
}

/// RAII 环境写入守卫(测试防踩踏):构造即写,Drop 恢复原值(原未设则移除)。
/// guard 只保证**恢复**,不保证**互斥** —— 同键并行用例请合并用例或外部加锁。
pub struct EnvGuard {
    key: &'static str,
    prev: Option<OsString>,
}

impl EnvGuard {
    pub fn set(key: &'static str, value: &str) -> Self {
        let prev = std::env::var_os(key);
        std::env::set_var(key, value);
        Self { key, prev }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        match self.prev.take() {
            Some(v) => std::env::set_var(self.key, v),
            None => std::env::remove_var(self.key),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, MutexGuard};

    /// std::env 进程全局 —— 触碰 env 的用例经此互斥(模块内测试串行)
    fn env_lock() -> MutexGuard<'static, ()> {
        static LOCK: Mutex<()> = Mutex::new(());
        LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    const K: &str = "OWL_ENV_READER_TEST_KEY";

    #[test]
    fn flag_str_or_parse_or_缺省与可见降级() {
        let _g = env_lock();
        let _guard = EnvGuard::set(K, "42");
        assert!(flag(K));
        assert_eq!(str(K).as_deref(), Some("42"));
        assert_eq!(parse::<usize>(K), Some(42));
        assert_eq!(parse_or(K, 7usize), 42);

        remove(K);
        assert!(!flag(K));
        assert_eq!(str(K), None);
        assert_eq!(str_or(K, "d"), "d");
        assert_eq!(parse::<usize>(K), None);
        assert_eq!(parse_or(K, 7usize), 7);
    }

    #[test]
    fn parse_坏值降级为_none() {
        let _g = env_lock();
        let _guard = EnvGuard::set(K, "not-a-number");
        assert_eq!(parse::<usize>(K), None);
        assert_eq!(parse_or(K, 3usize), 3);
    }

    #[test]
    fn env_guard_drop_恢复原值() {
        let _g = env_lock();
        let _outer = EnvGuard::set(K, "old");
        {
            let _inner = EnvGuard::set(K, "new");
            assert_eq!(str(K).as_deref(), Some("new"));
        }
        assert_eq!(str(K).as_deref(), Some("old"));

        // 原本未设的键:guard 退出后应回到未设
        remove(K);
        {
            let _inner = EnvGuard::set(K, "temp");
            assert!(flag(K));
        }
        assert!(!flag(K));
    }
}

//! **测试面统一报告出口**（`skip` / `note`）——测试域唯一的"说话"地方。
//!
//! # 为什么不走 `log_*!` / `metrics`
//!
//! - `cargo test` 按用例捕获 stdout/stderr ⇒ 只有**直写 stderr** 才会归属到
//!   用例（日志框架的 logger 需要 init + level 门，缺省 WARN；在测试里临时
//!   init 全局单例又会与 `--test-threads` 打架）；
//! - 测试输出**不该被级别门吞掉**：`skip:` 行是"这条用例为什么没跑"的唯一
//!   证据；被静默 = 假绿。
//!
//! ⇒ 测试域**唯一合法出口就是本模块**。生产面仍走 `owl_shared::log`
//! （事件流）/ `owl_shared::metrics`（数值账），三者分工不重叠。
//!
//! # 格式契约（grep 锚点，改动 = 破坏既有脚本）
//!
//! ```text
//! skip: <原因>                  ← [`skip`]；前缀由本模块唯一给出
//! [<tag>] <正文>                ← [`note`]；与生产 `log_*!` 的 `[tag] msg` 同形
//! ```
//!
//! 探针行统一用 [`note`]：`note("gdn", format_args!("T={t} nz={nz}"))`
//! ⇒ `[gdn] T=512 nz=0`（宏形态 `owl_shared::test_note!(…)`，导出在 crate 根）。**不要**再写裸 `eprintln!("[gdn] …")` —— 那会让
//! tag 格式一文件一样，跨用例的 grep 通道就断了。

use std::fmt;

/// `skip: <原因>` —— 用例门未过（GPU/对拍/资源缺）的唯一出口
///
/// `note` 只给原因**正文**，`skip: ` 前缀由本函数加（单点，防重复前缀）。
pub fn skip(note: impl fmt::Display) {
    // ⚠️ 本函数是 skip 前缀的**唯一**给出点；不许再调自己(自我递归)
    eprintln!("skip: {note}");
}

/// `[<tag>] <正文>` —— 用例内探针行（与生产 `log_*!` 渲染同形）
///
/// `tag` 用短名（`gdn` / `fi` / `gpu` / `tp`…），不带方括号。
pub fn note(tag: &str, args: fmt::Arguments<'_>) {
    eprintln!("[{tag}] {args}");
}

/// 裸行（多行转储/表格；**不是**探针行 —— 用前先想 [`note`] 是否更合适）
pub fn line(args: fmt::Arguments<'_>) {
    eprintln!("{args}");
}

/// `note` 的宏形态（免写 `format_args!`；tag 必须是字面量）
#[macro_export]
macro_rules! test_note {
    ($tag:expr, $($arg:tt)*) => {
        $crate::testkit::report::note($tag, ::std::format_args!($($arg)*))
    };
}

/// `skip` 的宏形态（免对常量再套一层）
#[macro_export]
macro_rules! test_skip {
    ($($arg:tt)*) => {
        $crate::testkit::report::skip(::std::format_args!($($arg)*))
    };
}

#[cfg(test)]
mod tests {

    /// 前缀单点：`skip()` 不再吃带前缀的正文（否则会出现 `skip: skip: …`）
    #[test]
    fn skip_prefix_is_single_sourced() {
        // 本测试只校验契约文案常量不含前缀（输出面由 cargo test 捕获）
        assert!(!crate::testkit::HF_PARITY_SKIP.starts_with("skip:"));
        assert!(!crate::testkit::missing_reason("ckpt:definitely-not-listed")
            .starts_with("skip:"));
    }
}

//! 两族错误,分而治之(design §六):
//! - **描述层逻辑违约**(形状/dtype)→ 毒值 [`LazyError`] 随声明流动,边界收割;
//!   算子对毒恒等(透传,不重复报);
//! - **执行层资源错误**(池耗尽/令牌死亡/拷贝失败)→ 不经毒值,发生在
//!   解释器/server,直接 `Err` 过线。
//!
//! 词汇权威:2026-09-25 下沉 `owl-iface::contract`(前后端共同依赖;
//! server 回执面必须能被后端表达)。本模块保留描述层毒值 [`LazyError`]
//! (server 不感知),`ModelError` 为 re-export(路径稳定)。

/// 毒值载荷:案发坐标 + 细节。反向树可沿 parents 反演——错误现场永远可重放。
#[derive(Debug, Clone)]
pub struct LazyError {
    /// 案发节点深度(归约链上的坐标,供回溯)
    pub at_depth: u32,
    /// 案发算子 + 双方元数据(如 "add: 形状不符 [4] vs [3]")
    pub detail: String,
}

pub use owl_iface::contract::ModelError;

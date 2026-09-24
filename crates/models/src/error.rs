//! 两族错误,分而治之(design §六):
//! - **描述层逻辑违约**(形状/dtype)→ 毒值 [`LazyError`] 随声明流动,边界收割;
//!   算子对毒恒等(透传,不重复报);
//! - **执行层资源错误**(池耗尽/令牌死亡/拷贝失败)→ 不经毒值,发生在
//!   解释器/server,直接 `Err` 过线。

use crate::tensor::Dtype;

/// 毒值载荷:案发坐标 + 细节。反向树可沿 parents 反演——错误现场永远可重放。
#[derive(Debug, Clone)]
pub struct LazyError {
    /// 案发节点深度(归约链上的坐标,供回溯)
    pub at_depth: u32,
    /// 案发算子 + 双方元数据(如 "add: 形状不符 [4] vs [3]")
    pub detail: String,
}

/// 边界错误:装载期(new)与执行期(interpret/to_host)的结构化报错。
impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelError::MissingKey { path, key } => write!(f, "MissingKey: {path}::{key}"),
            ModelError::ShapeMismatch { path, expected, got } => {
                write!(f, "ShapeMismatch: {path} 期望 {expected:?} 实得 {got:?}")
            }
            ModelError::DtypeMismatch { path, expected, got } => {
                write!(f, "DtypeMismatch: {path} 期望 {expected:?} 实得 {got:?}")
            }
            ModelError::PoolExhausted { pool, needed, available } => {
                write!(f, "PoolExhausted: {pool} 需 {needed}B 余 {available}B")
            }
            ModelError::DeadBlock { id } => write!(f, "DeadBlock: {id}"),
            ModelError::CaptureViolation { detail } => write!(f, "CaptureViolation: {detail}"),
            ModelError::ServerClosed => write!(f, "ServerClosed"),
            ModelError::Msg(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for ModelError {}

/// 词汇表是我们对 **server 错误面**的期望——server 侧必须能表达这些。
#[derive(Debug, Clone)]
pub enum ModelError {
    // ---- 装载期(new;P 阶段)----
    /// 权重键缺失(vb 路径 + 键名;键名双约定问题的结构化出口)
    MissingKey { path: String, key: String },
    /// 形状不符(期望 vs 实际)
    ShapeMismatch { path: String, expected: Vec<usize>, got: Vec<usize> },
    /// dtype 不符
    DtypeMismatch { path: String, expected: Dtype, got: Dtype },

    // ---- 执行期(interpret/to_host;server 回执)----
    /// 池容量不足
    PoolExhausted { pool: String, needed: u64, available: u64 },
    /// 池块句柄死亡后被引用(令牌世代校验失败;哨兵①词汇)
    DeadBlock { id: u64 },
    /// 捕获事务违约(空窗/窗内非法操作/审计不一致)
    CaptureViolation { detail: String },
    /// server 不可达/已关闭
    ServerClosed,
    /// 兜底(server 侧透传的结构化细节)
    Msg(String),
}

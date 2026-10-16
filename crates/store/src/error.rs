//! crate 级错误面(响亮:禁静默降级 —— 见 `arch.md` 律⑤)。
//!
//! code-master:错误分类 = 调用方能据以决策的类别,不是来源位置的镜像。

/// crate 结果别名。
pub type StoreResult<T> = std::result::Result<T, StoreError>;

/// 状态存储的错误(结构化:调用方可据此分流)。
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum StoreError {
    /// 预算分解失败(A5.1:没有实测数字的项不许进预算式)。
    #[error("预算分解失败:{0}")]
    Budget(String),
    /// 准入拒绝(**响亮**;`Strict` turn 在显存不足时必须走这里,不许降级成近似)。
    #[error("准入拒绝:{0}")]
    Reject(String),
    /// 资源耗尽(行/层容量),调用方应排空或等待。
    #[error("资源耗尽:{0}")]
    Exhausted(String),
    /// **占用中**(资源被别的持有者引用;例:块仍被活跃 turn 持着不可下沉)。
    /// 与 [`StoreError::Invariant`] 分开:调用方**据类别**决定"跳过该候选"
    /// (保洁换出)还是"停下来修账" —— 不靠字符串嗅探。
    #[error("占用中:{0}")]
    Busy(String),
    /// 不变量违反(调用序错误 / 账目不平)—— 编程错,必须修。
    #[error("不变量违反:{0}")]
    Invariant(String),
    /// 身份未登记。
    #[error("身份未登记:{0}")]
    Unknown(String),
}

/// 契约面转换(engine 用 `ModelError` 作为错误出口;store 是叶子 crate,
/// 转换实现放在**拥有 `StoreError` 的一侧** —— 孤儿规则唯一合法落点)。
///
/// 语义:**不丢判别**(`Exhausted`/`Invariant`/`Budget`/`Unknown` 进文案),
/// 便于上层按文案/日志定位;需要结构化时再扩 `ModelError` 变体。
impl From<StoreError> for owl_iface::contract::ModelError {
    fn from(e: StoreError) -> Self {
        owl_iface::contract::ModelError::Msg(format!("store: {e}"))
    }
}

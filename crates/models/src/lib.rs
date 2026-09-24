//! owl-models —— 声明式 Tensor 模型层(样板 crate,2026-09-23)。
//!
//! 取代 `crates/nn` + `crates/engine/src/models`。三句话:
//!
//! 1. **模型层是纯函数树**:`forward(&self, tx, xs, kv) -> Tensor`——
//!    无 Result、无 async、无相位分支;错误只在两个边界存在
//!    (构造期装载 `new`、执行期资源操作 `interpret`);
//! 2. **TensorOps 是声明链**:反向多叉树(值语义,深拷贝)上的节点视图;
//!    运算 = 往节目单追加节点,不执行。数据张量 = eval 之后的产物。
//! 3. **server 是解释器**:本 crate 的 [`client`] 模块定义了我们期望
//!    GPU server 提供的**能力契约**——cuda 包照此实现。
//!
//! 模块地图:
//! - [`dtype`] / [`shape`]:标注词汇
//! - [`error`]:毒值 + 两族错误(描述层逻辑违约 / 执行层资源错)
//! - [`plan`]:反向多叉树(值语义)+ 语义 Op 枚举
//! - [`tensor`]:声明式链式 API(客户主入口)
//! - [`tx`]:发射上下文(相位无感:eager 直发 / 捕获录制)
//! - [`client`]:**对 server 的能力期望**(malloc/htod/launch/capture/replay)
//! - [`interpreter`]:解释器契约(eager / capture-bake / CPU 参考)
//!
//! 设计文档:docs/arch/declarative-tensor.md(async-runtime.md 契约五)。

pub mod actions;
pub mod client;
pub mod demo;
pub mod device;
pub mod rt;
pub mod dtype;
pub mod kernel;
pub mod error;
pub mod interpreter;
pub mod plan;
pub mod shape;
pub mod tensor;

pub use dtype::Dtype;
pub use kernel::{Kernel, Scalar};
pub use error::{LazyError, ModelError};
pub use tensor::TensorOps;
pub use rt::Tensor;
pub use device::{Cpu, Device, DeviceKind};

/// 本 crate 的结果别名:只用于**边界**(构造期装载 / 执行收割)。
/// 描述层内部禁止出现(契约五)。
pub type Result<T> = std::result::Result<T, crate::error::ModelError>;

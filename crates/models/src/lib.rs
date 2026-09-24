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
//! - [`tensor`]:声明链 TensorOps + 运行时 Tensor<D> + Dtype 标注词汇
//! - [`shape`]:形状词汇
//! - [`error`]:毒值 + 两族错误(描述层逻辑违约 / 执行层资源错)
//! - [`plan`]:语义 Op 枚举
//! - [`kernel`]:Kernel 值(name + source + 发射配置)
//! - [`client`]:**对 server 的能力期望**(五原语 DeviceClient + eval 声明树求值)
//! - [`actions`]:预定义算子动作表(具名算子 → LaunchMsg 的唯一 lower 通道)
//! - [`interpreter`]:CpuFace(CPU 参考执行器;与 GPU server 同一契约)
//! - [`device`]:跨设备统一表达契约(CPU / GPU server 同一形状)
//!
//! 设计文档:docs/arch/async-runtime.md(v0.2,含声明式 Tensor 合并)。
//!
//! 调试:`OWL_DEBUG=1` 开启解释层发射日志(默认静默)。

pub mod actions;
pub mod client;
pub mod demo;
pub mod device;
pub mod kernel;
pub mod error;
pub mod interpreter;
pub mod plan;
pub mod shape;
pub mod tensor;

pub use kernel::{Kernel, LaunchShape, Scalar};
pub use error::{LazyError, ModelError};
pub use tensor::{Dtype, Tensor, TensorOps};
pub use device::{Cpu, Device, DeviceKind};

/// 本 crate 的结果别名:只用于**边界**(构造期装载 / 执行收割)。
/// 描述层内部禁止出现(契约五)。
pub type Result<T> = std::result::Result<T, crate::error::ModelError>;

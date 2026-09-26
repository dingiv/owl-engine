//! owl-models —— 声明式 Tensor 模型层(样板 crate,2026-09-23)。
//!
//! 取代 `crates/nn` + `crates/engine/src/models`。三句话:
//!
//! 1. **模型层是纯函数树**:`forward(&self, xs, ctx) -> TensorOps`——
//!    无 Result、无 async、无相位分支;错误只在两个边界存在
//!    (构造期装载 `eval_load`、执行收割 `eval`);
//! 2. **TensorOps 是声明链**:反向多叉树(值语义,深拷贝)上的节点视图;
//!    运算 = 往声明树追加节点,不执行。数据张量 = eval 之后的产物;
//! 3. **server 是解释器**:声明经解释器翻译为 LaunchMsg 原语,face 注入
//!    (`owl-cpu::CpuFace` / `owl-cuda::GpuClient` 同一契约)。
//!
//! # 模块地图(2026-09-26 重组:碎壳四合一,声明/执行分域)
//!
//! **契约门面**
//! - [`contract`]:owl-iface 契约词汇统一出口(线格式/能力契约/标注词汇/
//!   错误;权威在 owl-iface,此处零语义增量)
//!
//! **声明面(纯描述,零执行)**
//! - [`tensor`]:声明链 TensorOps + 毒值 LazyError(运行时数据 =
//!   池块 Bytes(memo)+ reference::Value(host 锚);旧同步 Device/
//!   Tensor<D> 面已废,唯一标准 = iface DeviceClient,2026-09-26 收束)
//! - [`ops`]:语义 Op 枚举 + lower 动作表(具名算子 → LaunchMsg 唯一通道)
//! - [`kernel`]:Kernel 值 + 名字→源 注册表(源码之家 = owl-kernels cu/)
//! - [`module`]:层协议双面 —— Module/ForwardCtx(计算)+ Loadable/Weight/
//!   Want/LoaderOps(装载)
//! - [`layers`]:Qwen3.5 文本主干层(容器 + layout + forward)
//! - [`loader`]:safetensors 权重源(通用;F32/BF16 → host f32)
//! - [`model`]:通用解码器主干(共有机制;embed + 层链 + norm + lm_head,
//!   整模单树 C5 + C10 装载)
//! - [`specs`]:模型规格集(每档一文件;纯参数事实 + 键名约定,拆分律见模块头)
//! - [`tokenizer`]:分词器(通用机制;家族事实由 specs 注入)
//!
//! **执行面(解释器)**
//! - [`interpreters`]:解释器集合(计算 eval/装载 load/观测 tap/生成
//!   generate;face 注入;变体见模块头)
//! - [`reference`]:同步参考解释器(reduce/CpuInterpreter;对拍锚,永不优化)
//!
//! 设计文档:docs/arch/async-runtime.md、docs/arch/qwen3-mini-demo.md、
//! roadmap.local/api-stabilize-plan.md(API 稳定化挂账)。
//!
//! 调试:`OWL_DEBUG=1` 开启解释层发射日志(默认静默)。

pub mod contract;
pub mod interpreters;
pub mod kernel;
pub mod layers;
pub mod loader;
pub mod model;
pub mod module;
pub mod ops;
pub mod reference;
pub mod specs;
pub mod tensor;
pub mod tokenizer;

// ============================================================================
// 公共门面(权威出口;推荐上层从这里取)
// ============================================================================

pub use contract::{DeviceClient, Dtype, ModelError, Shape};
pub use kernel::{Kernel, LaunchShape};
pub use module::{ForwardCtx, KvBuffers, Module};
pub use tensor::{LazyError, TensorOps};

/// 测试套件(就近测试的公共件;仅测试构建编译)
#[cfg(test)]
pub mod testkit;

/// 本 crate 的结果别名:只用于**边界**(构造期装载 / 执行收割)。
/// 描述层内部禁止出现(契约五)。
pub type Result<T> = std::result::Result<T, ModelError>;

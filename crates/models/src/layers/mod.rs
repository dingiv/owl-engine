//! Qwen3.5 文本主干 layers(容器 + load 钩子 + forward 纯声明)。
//!
//! 生命周期三段(2026-09-25 用户裁决;与 TensorOps 完全对称):
//! - **`new` = 准备容器**(零数据、零副作用:登记槽位键名/形状/布局);
//! - **`load_ops(&self)` = 装载的纯描述**(返回 LoaderOps,零副作用;
//!   执行 = `interpreter::eval_load(ops, face, src)` → LoadedBlocks;
//!   毒值/缺键/长度违约在此收割);`mount(&mut self, blocks)` = 纯内存回填;
//! - **`forward` = 纯声明**(槽 decl():未装载 → 毒值声明,eval 边界收割
//!   —— 全程 total,零 panic 零 expect)。
//!
//! 复杂算子 = Kernel 节点(源经 `crate::kernels` 注册表按名取用,层内
//! 零源码);索引/位置量(ids/pos/slots)以 f32 数值形态过线
//! (<2^24 精度无损;S4 u32 索引律的 dtype 扩展留后)。
//!
//! 模块地图(试水批):
//! - [`linear`] / [`rmsnorm`] / [`mlp`]:纯语义算子层(双 face 可对拍)
//! - [`embedding`] / [`rope`]:Kernel 节点试水件
//!
//! 未搬(试水通过后逐个立项,勿一把梭):
//! full attention(qk-norm + output gate + paged/naive attn)、
//! GatedDeltaNet(18/24 层,conv1d + delta rule 五 kernel 族)、
//! DecoderLayer 编排、vision ViT、MTP、chunked prefill、量化。

pub mod embedding;
pub mod linear;
pub mod mlp;
pub mod rmsnorm;
pub mod rope;

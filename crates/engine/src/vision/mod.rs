//! **视觉域**:引擎侧 VLM 请求态 + 视觉塔输出缓存,以及图像段的**前缀无感删除**
//! (host 纯函数面:识别 / 页对齐 pad / 自证)。
//!
//! ⚠️ 名字易混:`exec/img_evict.rs` 是同一特性的**盘层 I/O 收尾**(快照重挂 / 旧身份丢弃),
//! 本目录这份是 **host 纯函数面**(识别 / 页对齐 pad / 位置自证)—— 决策票在 `sched/scheduler`。
//!
//! 视觉塔本身的层实现在 `owl-models`;tower 的注意力核走 `cu/flash_attn`(FA2 hdim72)。
//! 缓存失效三层的受控实验见 `docs/llama.cpp/mtmd-hybrid-缓存失效-排查实录.md`。

pub(super) mod img_evict;
pub(super) mod vlm;

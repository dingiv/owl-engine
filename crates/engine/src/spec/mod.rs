//! **投机解码运行域**:模式 / 深度 / 草稿器 / 统计 / 诊断门 / B4 分母保鲜。
//!
//! 轮内的 propose/verify 执行在 `exec/`(草稿器面 = `exec/draft`、轮机器 =
//! `exec/spec`);本域只持**引擎级 spec 状态**。设计见 `docs/devlog/spec/`。

pub(super) mod spec_runtime;

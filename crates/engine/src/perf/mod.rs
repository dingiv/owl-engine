//! **诊断 / 取证域**:诊断 dump 路径的唯一构造点(避免散落的 `/tmp/owl-*` 字面量)
//! 与相关探针开关。性能口径与排障顺序见 `docs/devlog/dev/debug-tools.md`。

pub(super) mod probe_dump;

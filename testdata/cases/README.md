# testdata/cases —— 用例 overlay（约定路径，零 env）

**一个用例一个文件**，由 `#[owl_test]` 按源码位置**自动解析**：

```
crates/<crate>/src/<模块路径>.rs  +  用例名 <fn>
  ⇒ testdata/cases/<crate>/<模块路径>/<fn>.toml
```

例：`crates/models/src/layers/gdn.rs::gpu_l2norm_matches_host`
→ `testdata/cases/models/layers/gdn/gpu_l2norm_matches_host.toml`

## 怎么写

只写**差异项**（继承链：`extends → OWL_TEST_CONF 环境档 → 本文件`）：

```toml
# 只改几何:其余全继承(生产档或环境档)
[runtime]
max_seq = 2048

# 也可自带基准(显式路径)
# extends = "profiles/spec-dflash.toml"
```

- **存在才加载**；删除文件 = 还原；**不会影响任何兄弟用例**（天然隔离）；
- 集成测试（`crates/*/tests/*.rs`）无 `/src/` 段 ⇒ 没有 overlay；
- 本目录**不进版本**（本地临时档）；规范见 `docs/dev/testing.md`。

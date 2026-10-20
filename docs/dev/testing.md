# 测试工作流（v2，2026-10-18）

> **一句话**：用例只写被测逻辑；配置/资源/设备全部**按约定自动解析**，
> 不在用例里读 env、不重复抄配置、不互相污染。
> 设计/施工记录见 `docs/arch/owl-test-macro.md`；本文是**怎么用**的规范。

---

## 1. 三个文件面（全部在 `testdata/` 下）

| 面 | 文件 | 作用域 | 进版本? |
|---|---|---|---|
| **环境档** | `testdata/conf/<名>.toml`（路径任意，用 `OWL_TEST_CONF` 指） | **整轮测试**（所有用例） | 否（示例可进） |
| **用例 overlay** | `testdata/cases/<crate>/<模块路径>/<用例名>.toml` | **单个用例**（自动解析） | 否（随手写/删） |
| **资源清单** | `testdata/resources.toml` | **整机**（检查点/金标/开关） | 否（`*.example` 进版本） |

配置**层链**（后者覆盖前者）：

```
flavor(可选,预设名) → 环境档(OWL_TEST_CONF) → 用例 overlay(按路径自动找)
```

- 任何一层都可写 **`extends = "<路径>"`**（显式相对/绝对路径;不再有 `flavor` 名字键）
  ⇒ 基础口径永远复用生产档，不用抄；
- **missing = 该层不参与**（全缺 = `OwlConfig::default()`）；
- 每层都走唯一 loader：`deny_unknown_fields`、枚举严格、`flavor`/`extends`
  环检测 ⇒ 写错键 = 响亮报错。

### 1.1 用例 overlay 的自动解析（本工作流的核心）

宏把 `file!()` 传进 `case_config`，路径由**源码位置**推导：

```
crates/models/src/layers/gdn.rs + 用例 gpu_l2norm_matches_host
  ⇒ testdata/cases/models/layers/gdn/gpu_l2norm_matches_host.toml
```

- **存在才加载** ⇒ 加配置 = 丢一个文件；删除 = 还原；**不会影响任何兄弟用例**；
- 模块路径天然带在路径里 ⇒ 不同模块的同名用例不冲突；
- 集成测试（`crates/<c>/tests/*.rs`）无 `/src/` 段 ⇒ 无 overlay（仍可吃环境档）；
- 文件只写**差异项**（从 flavor / 环境档继承）。

## 2. 用例侧：`#[owl_test]`

```rust
use owl_macros::owl_test;
use owl_shared::config::OwlConfig;

#[owl_test]                       // 纯 async 包装(= #[tokio::test])
async fn cpu_case(cfg: &OwlConfig) { /* cfg 自动注入 */ }

#[owl_test(gpu)]                  // GPU 门:OWL_TEST_DEVICE 未设 ⇒ skip
async fn gpu_case() { }

#[owl_test(gpu, hf_parity)]       // HF 对拍门(OWL_HF_PARITY)
async fn parity_case() { }

#[owl_test(gpu, needs = "ckpt:27b-awq")]   // 资源门(清单里没有 ⇒ skip)
async fn needs_ckpt_case() { }

#[owl_test(gpu)]                  // 同步体也收:只做门控/配置,不建 runtime
fn sync_probe() { }
```

- 参数**最多一个** `&OwlConfig`；门控用例必须返回 `()`；
- 函数名不变（`cargo test <名>` 过滤 / `--list` / `#[ignore]` 全兼容）；
- skip 文案统一并**点名原因**：`skip: OWL_TEST_DEVICE 未设` /
  `skip: 资源清单缺 "ckpt:27b-awq"(…resources.toml…)`；
- 调用点必须 `#[cfg(test)]` 在宏之上（dev-dep 纪律）。

## 3. 资源与设备

- **资源**：`testdata/resources.toml`

  ```toml
  "ckpt:27b-awq" = "/path/to/Qwen3.8-27B-AWQ-INT4"
  "ckpt:dflash2" = "/path/to/Qwen3.8-27B-DFlash2-W4A16"
  "flag:hf-parity" = true
  ```

  读取面：`testkit::{resource, resource_or, resource_flag, resource_ok, missing_reason}`；
  **旧 `OWL_*_DIR` 等 17 键已退役**（仍设置 = 响亮 panic + 迁移指引）。

- **设备**：唯一 env `OWL_TEST_DEVICE`，值 = **PCI BDF**（`41:00.0`；
  `nvidia-smi --query-gpu=pci.bus_id` 查）。纯数字 = 响亮拒。
  需要 ordinal 的底层测试用 `owl_cuda::test_device_ordinal(_opt)`（唯一解析入口）。

## 4. 决策表：我要做什么 → 改哪里

| 诉求 | 动作 |
|---|---|
| 新写一个 CPU 单测 | `#[owl_test] async fn …(cfg: &OwlConfig)` |
| 新写一个 GPU 探针（同步） | `#[owl_test(gpu)] fn …()` |
| 某用例要特殊几何/开关 | 丢一个 `testdata/cases/…/<用例>.toml`（只写差异） |
| 整轮测试换口径 | 写 `testdata/conf/x.toml`（可含 `flavor`），`OWL_TEST_CONF=` 指它 |
| 新检查点/金标 | 加一行到 `testdata/resources.toml`，用例加 `needs = "<key>"` |
| 换一张卡跑 | `OWL_TEST_DEVICE=<BDF> cargo test …` |
| 跑全部/单个 | `cargo test -p <crate> --lib [用例名]` |

## 5. 反模式（不要做）

- ❌ 用例里 `env_reader::…` 读业务旋钮/资源路径（唯一例外：设备经统一入口）；
- ❌ 把配置抄进用例常量或每个文件一份大 toml（用 flavor + overlay 差量）；
- ❌ 用 CUDA 序号钉卡（会漂；已禁用）；
- ❌ 在 overlay 里写"无差异的整档"（继承就好；否则失去单一来源）；
- ❌ 新开一个"只差一行"的测试文件（先看 `#[owl_test]` 选项 + overlay 够不够）。

## 6. 现状（迁移状态）

- models 内联测试面 **53 例**已收编；kernels 同步探针 **24 例**已收编；
- 旧 env：每用例配置键（`OWL_TEST_CONFIG_*`）与 17 个数据门键**全部退役**；
- 测试域**只剩 1 个 env**（`OWL_TEST_DEVICE` = BDF）；
- 尾量：e2e 场景（`pub async fn gpu_*`）仍由 e2e 入口统一驱动，随其改造另案。

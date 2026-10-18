# testdata —— 测试资产与"快速测试系统"落地目录

## 1. 测试配置从哪来（单一入口）

`#[owl_test]` 用例的配置**层链**（后者覆盖前者）：

```
flavor（可选，预设档名）  →  OWL_TEST_CONF（整轮测试环境档）  →  用例 overlay
```

| 层 | 放哪 | 干什么 |
|---|---|---|
| `flavor` | `flavors/<名>.toml`（仓库顶层） | 直接复用生产/档位预设：`flavor = "spec-dflash"` |
| 环境档 | 任意路径，用 **`OWL_TEST_CONF`** 指定 | 整轮共同口径（位置/几何/探针）—— 各会话/各 agent 用同一份，测出来的配置一致 |
| 用例 overlay | `testdata/cases/<crate>/<模块路径>/<用例名>.toml` | 只改一个用例；**删文件即还原**，不留 env 键 |

```bash
# 整轮口径（环境档自带 flavor 基准 + 覆盖）
OWL_TEST_CONF=testdata/conf/local.toml cargo test -p owl-models --lib
```

`testdata/conf/local.toml` 例：

```toml
flavor = "spec-dflash"      # 复用生产档（flavors/spec-dflash.toml）

[runtime]
max_seq = 4096              # 测试口径覆盖
```

单用例 overlay 例（`testdata/cases/models/layers/gdn/gpu_l2norm_matches_host.toml`）：

```toml
[runtime]
max_seq = 2048
```

> 键名规则：`file!()` 形如 `crates/<crate>/src/<模块路径>.rs`，
> overlay = `testdata/cases/<crate>/<模块路径>/<用例名>.toml`。
> 缺文件 = 不上这一层；环境档缺失 = 响亮 panic。

## 2. 设备与资源

- **设备**：唯一 env `OWL_TEST_DEVICE`，值 = **PCI BDF**（如 `41:00.0`；用 `nvidia-smi --query-gpu=pci.bus_id` 查）。
  未设 = GPU 用例 auto-skip；**纯数字（CUDA 序号）会被响亮拒绝**。
- **资源清单**：`testdata/resources.toml`（本机，不进版本；模板见 `resources.toml.example`）——
  取代 `OWL_AWQ27B_DIR` / `OWL_DFLASH2_DIR` / `OWL_VLM_CKPT` 一族 env：

  ```toml
  "ckpt:27b-awq"   = "/path/to/Qwen3.8-27B-AWQ-INT4"
  "ckpt:dflash2"   = "/path/to/Qwen3.8-27B-DFlash2-W4A16"
  "ckpt:qwen35-4b" = "/path/to/Qwen3.5-4B"
  "flag:hf-parity" = true          # 开关（未列 = false）
  ```

  用例侧：`owl_shared::testkit::resource("ckpt:27b-awq")`（缺 = skip，文案点名 key）；
  宏门控：`#[owl_test(needs = "ckpt:27b-awq")]`。**旧 env 若仍被设置 = 响亮 panic**（附迁移指引）。

## 3. 目录一览

| 目录/文件 | 内容 | 进版本? |
|---|---|---|
| `cases/` | 用例 overlay（本地临时档，可删） | 否（随手写、随手删） |
| `conf/` | 测试环境档（本地；`OWL_TEST_CONF` 指向） | 否 |
| `resources.toml` | **资源清单**（本地；检查点/金标路径 + 开关） | 否（`resources.toml.example` 进版本） |
| `gdn_fla/` | GDN 金标：`gen.py` 生成 `cases/*.bin` | 脚本+manifest 是；大件否 |
| `args/` | 引擎启动档（索引见 `args/README.md`） | 仅 `perf-ctxladder-27b.toml` |
| `server/` | agent-serving 自有档 | 否 |
| `dflash2_golden.safetensors` 等 | 算子金标 | 否（tools/ 生成） |

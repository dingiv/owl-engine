# testdata —— 测试资产与"快速测试系统"索引

> **规范**:`docs/dev/testing.md`(测试工作流 v2) · **设计**:`docs/arch/owl-test-macro.md`
> **一句话**:用例只写被测逻辑;配置/资源/设备**按约定自动解析**,不读 env、不抄配置、不互相污染。

## 1. 测试配置从哪来(层链 + 自动解析)

```
extends(可选,基准档路径)  →  OWL_TEST_CONF(整轮环境档)  →  用例 overlay(按路径自动找)
```

| 层 | 放哪 | 干什么 | 进版本? |
|---|---|---|---|
| `extends` | 任意路径(`profiles/...`) | 复用基准档:`extends = "profiles/qwen38_27b-3090-w4a16-dflash.toml"` | ✅ |
| 环境档 | 任意路径,用 **`OWL_TEST_CONF`** 指定 | 整轮共同口径(各会话/各 agent 一致) | 否(`conf/` 本地) |
| **用例 overlay** | `testdata/cases/<crate>/<模块路径>/<用例名>.toml` | 只改**一个**用例;**删文件即还原** | 否(随手写/删) |

- **overlay 自动解析**(零 env):`file!()` 形如 `crates/<crate>/src/<模块路径>.rs`
  → `testdata/cases/<crate>/<模块路径>/<用例名>.toml`;
  例:`crates/models/src/layers/gdn.rs::gpu_l2norm_matches_host`
  → `testdata/cases/models/layers/gdn/gpu_l2norm_matches_host.toml`;
- **存在才加载**;三层全缺 = 引擎缺省;**天然按用例隔离**(不同模块同名不冲突);
- 只写**差异项**(从基准/环境档继承);集成测试(`crates/*/tests/*.rs`)无 overlay;
- 每层都走唯一 loader:键错/枚举错/`extends` 环 = **响亮拒**。

```toml
# testdata/conf/local.toml(整轮口径)
extends = "profiles/qwen38_27b-3090-w4a16-dflash.toml"

[runtime]
max_seq = 4096
```

```toml
# testdata/cases/models/layers/gdn/gpu_l2norm_matches_host.toml(单用例差异)
[runtime]
max_seq = 2048
```

```bash
OWL_TEST_CONF=testdata/conf/local.toml cargo test -p owl-models --lib   # 整轮
```

## 2. 用例侧:`#[owl_test]` 选项速查

```rust
#[owl_test]                                async fn cpu_case(cfg: &OwlConfig) {}
#[owl_test(gpu)]                           async fn gpu_case() {}       // OWL_TEST_DEVICE 未设 ⇒ skip
#[owl_test(gpu, hf_parity)]                async fn parity() {}         // OWL_HF_PARITY 门
#[owl_test(gpu, needs = "ckpt:27b-awq")]   async fn with_ckpt() {}      // 资源清单门(缺 ⇒ 点名 key)
#[owl_test(gpu)]                           fn sync_probe() {}           // 同步体也收(不建 runtime)
```

## 3. 设备与资源

- **设备**:唯一 env **`OWL_TEST_DEVICE`** = **PCI BDF**(如 `41:00.0`;
  `nvidia-smi --query-gpu=pci.bus_id` 查)。未设 = GPU 用例 auto-skip;
  **纯数字(CUDA 序号)会被响亮拒绝**;底层要 ordinal 用
  `owl_cuda::test_device_ordinal(_opt)`(唯一解析入口)。
- **资源清单**:`testdata/resources.toml`(本机,不进版本;模板 `resources.toml.example`
  进版本)—— 取代 `OWL_AWQ27B_DIR` / `OWL_DFLASH2_DIR` / `OWL_VLM_CKPT` 一族 env:

  ```toml
  "ckpt:27b-awq"   = "/path/to/Qwen3.8-27B-AWQ-INT4"
  "ckpt:dflash2"   = "/path/to/Qwen3.8-27B-DFlash2-W4A16"
  "ckpt:qwen35-4b" = "/path/to/Qwen3.5-4B"
  "flag:hf-parity" = true          # 开关(未列 = false)
  ```

  用例侧 `testkit::{resource, resource_or, resource_flag, missing_reason}`;
  **17 个旧 env 已退役**(仍设置 = 响亮 panic + 迁移指引)。

## 4. 决策表:要做什么 → 改哪里

| 诉求 | 动作 |
|---|---|
| 某用例要特殊几何/开关 | 丢一个 `testdata/cases/…/<用例>.toml`(只写差异) |
| 整轮测试换口径 | 写 `testdata/conf/x.toml`(可含 `extends`),`OWL_TEST_CONF=` 指它 |
| 新检查点/金标 | 加一行 `resources.toml`,用例加 `needs = "<key>"` |
| 换一张卡跑 | `OWL_TEST_DEVICE=<BDF> cargo test …` |
| 跑全部 / 单个 | `cargo test -p <crate> --lib [用例名]` |

**反模式**:用例里读 env 旋钮 / 每文件一份大 toml / CUDA 序号钉卡 /
overlay 抄整档 / 为"只差一行"新开测试文件。

## 5. 目录一览

| 目录/文件 | 内容 | 进版本? |
|---|---|---|
| `cases/` | **用例 overlay**(约定路径;规范见 `cases/README.md`) | 仅该 README |
| `conf/` | 测试环境档(本地;`OWL_TEST_CONF` 指向) | 否 |
| `resources.toml` | **资源清单**(检查点/金标路径 + 开关) | 否(`resources.toml.example` 是) |
| `args/` | 引擎启动档(索引见 `args/README.md`);入库的 `perf-ctxladder-27b.toml` **自包含**(路径写死在档内) | 仅该档 |
| `gdn_fla/` | GDN 金标:脚本 + `cases/*.manifest.json`;`*.bin` 由 `gen.py` 生成(大件可跨树符号链接) | 脚本+manifest 是;大件否 |
| `server/` | agent-serving 自有档 | 否 |
| `dflash2_golden.safetensors` 等 | 算子金标(tools/ 生成) | 否 |

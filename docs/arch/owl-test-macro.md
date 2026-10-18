# owl 测试过程宏(owl_test)—— 用例参数化与样板收编

> 2026-10-15 立项(用户指令:研究仓库测试样板 → 写过程宏减样板 → 隐藏
> 环境变量传每用例配置文件 → 样板解析配置包装成变量传给真测试函数,
> **让测试函数带参数**)。姊妹卷:[`test-architecture.md`](test-architecture.md)
> (两类测试 + e2e 入口唯一化 —— 本宏服务其中的**单元测试层**)。

---

## 一、研究结论:测试样板的四处啰嗦

全仓盘点(生产源码,不含 `tests/` 目录):

| # | 样板 | 实测量 | 例证 |
|---|---|---|---|
| S1 | `#[tokio::test]` + 手建 runtime 等价物 | **106 处**(models 90 / engine 6 / shared 6 / server 4) | 每个 async 测试都要宏 |
| S2 | **GPU 门控四行**:`let Some(ordinal) = … else { eprintln!("skip: OWL_TEST_DEVICE 未设"); return }` | **59 处** `OWL_TEST_DEVICE 未设` | 同一段逐字重复,分散在各测试头部 |
| S3 | `EngineConfig { device_ordinal, max_seq, prefill_chunk, knobs }` **字面量构造** | **24 处** | 每个测试 hardcode 一套配置,换档 = 改代码 |
| S4 | 测试参数无通路 | — | 配置类参数只能 env 直读(违反显式依赖律的测试域豁免)或写死 |

**根因**:libtest 的 `#[test]` 无参 ⇒ 测试所需的一切(设备门控、引擎配置)
只能靠「env 散读 + 字面量」两条样板路。S2/S3 相互叠加 —— 一个 GPU 引擎
测试的头部平均 8~12 行与被测逻辑无关的接线。

## 二、方案:`#[owl_test]` 属性过程宏

### 2.1 形态

```rust
use owl_macros::owl_test;   // 过程宏 crate `owl-macros`(仅导宏;后续新宏统一进此)
use owl_shared::config::OwlConfig;

/// 无门控用例:配置变量由宏注入
#[owl_test]
async fn cpu_case(cfg: &OwlConfig) {
    assert!(cfg.runtime.max_seq >= 64);   // ← 配置文件真的到了
}

/// GPU 门控用例:OWL_TEST_DEVICE 未设 ⇒ 自动 skip(样板 S2 消失)
#[owl_test(gpu)]
async fn gpu_case(cfg: &OwlConfig) {
    let ordinal = cfg.runtime.test_device.unwrap();  // 宏已回填
    // ...
}

/// 叠用门控(2026-10-18):对拍门 OWL_HF_PARITY 未开也 skip(与手写 skip_note 同文案)
#[owl_test(gpu, hf_parity)]
async fn hf_parity_case() { /* 真跑需要 OWL_HF_PARITY=1 + OWL_TEST_DEVICE */ }
```

**选项清单**(逗号分隔、顺序无关,未知项 = compile_error):
`gpu` / `hf_parity`;判据与 skip 文案全在 `owl_shared::testkit`(宏零逻辑)。

### 2.2 配置从哪来:**单一 env + 约定 overlay**(2026-10-18 定稿)

层链(**后者覆盖前者**):

```
flavor(可选,预设档名) → OWL_TEST_CONF(整轮环境档) → 用例 overlay(约定路径)
```

| 层 | 位置 | 说明 |
|---|---|---|
| `flavor` | `flavors/<名>.toml`(仓库顶层) | 配置档里写 `flavor = "spec-dflash"` 即**复用预设当基准**(loader 顶层键,等效 `extends` 到 `flavors/<名>.toml`);测试与站点档同用 |
| `OWL_TEST_CONF` | 任意路径(**唯一**测试 env) | 整轮测试的环境档(TOML,引擎同款 `OwlConfig` 格式):共同口径 + 本机覆盖。未设 = 无此层;设了但文件缺失/解析失败 = **响亮 panic** |
| 用例 overlay | `testdata/cases/<crate>/<模块路径>/<用例名>.toml` | **零 env**:由宏传入的 `file!()` 推导。临时改一个用例 = 丢个文件;删文件即还原;集成测试(`tests/*.rs`)无此层 |

- 三层全不在 ⇒ `OwlConfig::default()`(测试零配置可跑,不 skip);
- **旧键 `OWL_TEST_CONFIG_<函数名大写>` 已退役**:设了 = 响亮 panic +
  迁移指引(给出该用例应放 overlay 的**确切路径**);
- 解析走唯一 loader(`load_chain`)—— deny_unknown_fields、枚举严格、
  `flavor`/`extends` 的环/超深检测对测试配置同样生效(错键名 = panic,
  与引擎拒启同语义);
- 目录与用法速查:`testdata/README.md`。

### 2.3 宏展开契约(属性宏,`syn`/`quote`)

输入:`async fn 名(参数?) { 体 }`,参数 **0 个或 1 个**,1 个时类型必须是
`&OwlConfig`(否则 compile_error,响亮拒绝)。属性选项:`gpu`(可与其他共存)。

输出(伪码):

```rust
// ① 原 async fn 降为具名 body(私有;同名 test 不冲突)
async fn __owl_body_名(cfg: &OwlConfig) { 原体 }

// ② 同名 #[test] 包装(同步;libtest 照常发现/过滤)
#[test]
fn 名() {
    let mut __cfg = ::owl_shared::testkit::case_config("名");   // 隐藏 env → OwlConfig
    #[gpu 门控:__cfg.runtime.test_device 无值 ⇒ eprintln skip + return]  // S2 收编
    // ③ 手建 current_thread runtime(= #[tokio::test] 缺省形态),block_on body
    // ④ 无参用例省略 body 参数;body 名带原函数名反查(错误可读)
}
```

- **不依赖 `#[tokio::test]`**:wrapper 是同步 `#[test]`,runtime 在 wrapper 内
  手建(`new_current_thread().enable_all()`)—— 展开引用 `::tokio`,消费方需
  tokio dev-dep(全仓测试已有);
- **函数名保持原样** ⇒ `cargo test cpu_case` / `--exact` 过滤、`--list`、
  `#[ignore]` 透传全部不受影响;
- 编译期校验(响亮拒绝清单):非 async / 参数个数 >1 / 参数类型非
  `&OwlConfig` ⇒ `compile_error!` 指明位置与修法。

### 2.4 职责分界

| 层 | 职责 |
|---|---|
| `crates/macros`(新,proc-macro only) | 签名校验 + 展开;**零逻辑**(逻辑在 helper,宏只搬运) |
| `owl_shared::testkit::case_config(name)` | 读隐藏 env → 唯一 loader 解析 → 返回 `OwlConfig`;GPU 门控读 `test_device` 回填 |
| 消费测试 | 写 `#[owl_test]` + 带参签名;tokio dev-dep |

## 三、范围与非目标

**做**:宏 crate + helper + 账外键登记 + **示范改造**(engine 2 个 CPU 测试
+ graph_plan 1 个 GPU 门控测试,共 3 处证明 S1/S2/参数化全链)+
配置文件真实生效的回归测试 + 文档。

**做(2026-10-18 起,试点已落地)**:按 crate 分批迁移 `#[tokio::test]` + 手写门
—— 试点 `models/layers/gdn.rs`:22 个 tokio 测试中 21 个迁到宏(19 个 `gpu` +
2 个 `gpu, hf_parity`),并**注册了一个漏写属性的死测试**(`gpu_gating_g_matches_host`,
注册后真机通过);净 **−66 行**。余量见 §六。

**不做**(列挂账):
- e2e 场景(`pub async fn gpu_*`,无 `#[test]`)不适用本宏 —— 它们由
  e2e 入口统一驱动,配置化随 e2e 入口改造另案;
- `stack_mb` 深栈选项(e2e 入口的 64MiB 栈问题在单测域未复现)、
  `flavor` 内联预设(文件已够)—— 有需求再加选项。

## 四、验收

1. `cargo test -p owl-macros -p owl-shared -p owl-engine --lib` 全绿;
2. **参数化实证**:设 `OWL_TEST_CONF` 指向一份改了 `runtime.max_seq` 的 TOML
   (或丢一个用例 overlay),断言测试收到该值;env 清除/无 overlay 则收 default
   —— 双臂/三层单测,用 `EnvGuard` 防踩踏(`testkit::case` 自带 4 项);
3. 示范测试在 GPU 环境下行为不变(GPU 门控测试:env 未设 = skip 打印逐字同);
4. `cargo check --workspace` 零新告警;
5. 账外键清单登记 `OWL_TEST_CONF`(旧 `OWL_TEST_CONFIG_*` 退役)。

## 五、施工记录(2026-10-15 同日完工)

- **新 crate `crates/macros`**(proc-macro only,workspace 成员,
  syn2+quote):`#[owl_test]` / `#[owl_test(gpu)]`;校验清单(响亮拒绝)
  = 非 async / 泛型 / 参数 >1 / 参数非 `&OwlConfig` / gpu 非 unit 返回,
  各自带修法提示。
- **helper `owl_shared::testkit::case_config(name)`**(testkit/case.rs):
  键派生 `OWL_TEST_CONFIG_<NAME大写>` → 唯一 loader → `test_device`
  无条件回填 `OWL_TEST_DEVICE`;+3 单测(键派生 / 注入双臂与唯一
  loader 逐字段同 / 坏配置响亮 panic)。
- **账外键登记**:config.rs 账外键清单 +`OWL_TEST_CONFIG_*`。
- **示范改造 3 处**:engine 2 个 CPU 测试(EngineConfig 全件从用例
  配置注入,原字面量 64/16/default knobs ⇒ cfg.runtime.*)+ graph_plan
  `gpu_capture_matches_eager`(S2 门控四行 ⇒ `#[owl_test(gpu)]`,skip
  文案逐字保持);tests.rs 加 `case_config_injection` 端到端实证。
- **验收实测**:① shared 42 / engine 34 全绿;② 注入双臂实测绿
  (缺省臂 + `OWL_TEST_CONFIG_CASE_CONFIG_INJECTION=<toml> OWL_TEST_DEVICE=2`
  文件臂,与独立复算逐字段同);③ GPU 门控双臂实测绿(env 未设 =
  逐字 `skip: OWL_TEST_DEVICE 未设`,设了 = 真跑 0.31s);④
  `cargo check --workspace --tests` 0 error;⑤ **负例探针**:三条违约
  (错参数型/双参数)编译期 compile_error 文案实测到位。
- **施工中抓到并修复的宏 bug**:无参形态调用点缺括号(`#body` 空 token
  ⇒ 拿到 fn 指针 `is not a future`)—— 负例探针的价值实证,已修(空参
  改 `()`)。
- **⚠️ 非测试构建纪律**:宏调用点必须 `#[cfg(test)]` 在 `#[owl_test]`
  **之上**(cfg 先求值 ⇒ dev-dep 在 e2e feature 档不触宏名);
  `cargo check -p owl-engine --features e2e` 实测通过。
- **遗留**(已在 §三列范围外):106 处 `#[tokio::test]` 分批迁移、
  e2e 场景配置化(随 e2e 入口另案)。


---

## 六、迁移台账(2026-10-18)

### 已迁(models 面:共 **53** 个用例)
| 文件 | 迁入 | 备注 |
|---|---:|---|
| `layers/gdn.rs` | 22 | 含 2 个 `gpu, hf_parity`;顺手注册回一个漏写属性的死测试 |
| `layers/attention.rs` | 12 | |
| `layers/dflash2.rs` | 7 | |
| `port_tests.rs` | 6 | |
| `layers/vision.rs` | 3 | |
| `layers/decoder.rs` | 2 | 其中一个的门**被注释掉过**(等于无门在跑)—— 本次恢复 |
| `formats/qwen35.rs` | 1 | |
| **合计** | **53** | models 内联测试面基本收编完毕 |

### 未迁(有意保留,各有理由)
- `decoder.rs::pinned_roundtrip_probe`、`dflash2.rs::gpu_dflash2_stage_bench`:
  返回 `Result<(), E>` ⇒ 门控要 `return Ok(())`,宏现只支持 unit 返回(裸 `return`);
- `qwen35.rs::gpu_dflash2_draft_load_bench`:`#[tokio::test(flavor = "multi_thread", …)]`,
  宏固定 current_thread(选项见 §三"不做")；
- `port_tests.rs::gpu_prefill_paged_attn_f16_hd256_impl`:非测试 helper(由测试调用),门保留;
- **数据门**(`OWL_*_DIR` / 金标缺失)另属一类,见下节提案。

### 不在本宏覆盖
- **kernels 集成测试**(`crates/kernels/tests/*.rs`,~40 个文件):全部是**同步** `#[test]`
  + 四行门,宏当前只收 `async fn`。若要把这批也收编,需给宏加
  **同步体模式**(不建 runtime,仅注入门/配置)—— 待定;e2e 入口(`pub async fn gpu_*`)
  仍按原计划随 e2e 改造另案。


---

## 七、走向"零散 env"的测试姿势(**提案**;2026-10-18)

现状:测试域 env 键分四类。AGENTS.md 的立场是"配置显式,env 退出正式链;
测试域可放但不推荐"。归位方案与**进度**:

| 类 | 旧姿势 | 新姿势 | 状态 |
|---|---|---|---|
| **用例配置**(引擎旋钮) | `OWL_TEST_CONFIG_<FN>` 每用例一键 | **`OWL_TEST_CONF` 单一入口 + `testdata/cases/**` 约定 overlay**(宏传 `file!()` 推导;`flavor` 复用 `flavors/`) | ✅ **已落地**(2026-10-18) |
| **整轮共同口径** | 各会话各写一套档 | 同上一入口:一份测试环境档定 flavor + 本机覆盖,大家测的配置一致 | ✅ 已落地 |
| **设备**(哪张卡) | `OWL_TEST_DEVICE` | 保留为设备专用 env;值 = **PCI BDF**(`41:00.0`;**CUDA 序号已禁用**,纯数字响亮拒);进一步可收进 `testdata/machine.toml` | ✅ 已落地(2026-10-18) |
| **数据**(检查点/金标) | `OWL_AWQ27B_DIR` / `OWL_DFLASH2_DIR` 等十余键 | **资源清单** `testdata/resources.toml`(`key = "path"`);用例声明 `#[owl_test(needs = ["ckpt:27b-awq"])]`,缺失 = skip 且**点名缺哪个 key** | ⏳ 提案 |
| **昂贵套件**(parity/bench) | `OWL_HF_PARITY` 等 | 已进宏选项(`hf_parity`);后续并入资源清单 | 部分 |

下一步:资源清单(吃数据门)→ 设备收进 machine 文件。终态 = `#[owl_test]`
用例**零散 env 归零**(仅保留设备选择器)。


### 2026-10-18:单一入口落地(替代每用例 env)

- `OWL_TEST_CONF` 成为**唯一**测试 env(整轮环境档);每用例 env 键退役
  (设了 = 响亮 panic + 迁移指引);
- 用例 overlay 改**约定路径** `testdata/cases/<crate>/<模块路径>/<用例名>.toml`
  (零 env);
- **`flavor` 升格为配置层顶层键**(loader,与 `extends` 同族):任何配置档都能
  自带预设基准 ⇒ 测试环境档只写差异,复用 `flavors/` 生产档;
- `flavors/` 目录自 `assets/flavors/` **提到仓库顶层**(模块 `shared::flavors`);
- loader 新增 `OwlConfig::load_chain(base, &[layers])`(多层叠加,每层支持
  `flavor`/`extends`);
- 实证:`testdata/README.md` 速查;`testkit::case` 4 项单测(路径派生/环境档双臂/
  退役键拒/overlay 生效)+ `engine::tests::case_config_injection` 端到端复算;
  真机演示:`OWL_TEST_CONF=<flavor+override> cargo test -p owl-engine --lib
  case_config_injection` 绿。

### 2026-10-18(晚):设备门改 BDF

- `OWL_TEST_DEVICE` **唯一形态 = PCI BDF**(如 `41:00.0`);纯数字(CUDA 序号)
  在解析层**响亮拒绝**("CUDA 序号已禁用");未设 = skip。
- 读取面 `owl_shared::device::test_device_bdf()`;解析面
  `owl_cuda::{resolve_ordinal_by_bdf, test_device_ordinal, test_device_ordinal_opt}`
  (设备表枚举 + `DeviceSpec` 同一套归一;未命中 = 报可用 BDF 表)。
- `[runtime] test_device` 类型 `Option<usize>` → `Option<String>`(BDF);
  `#[owl_test(gpu)]` 门控语义不变。
- 全仓测试面改走唯一入口:kernels(12 文件;dev-dep owl-cuda)、engine
  (tests/e2e/graph_plan/examples)、cuda 集成测试补 GPU 门(9 文件 / 22 处)、
  cuda examples。实证:BDF 真机跑通、纯数字响亮拒、未设 skip。

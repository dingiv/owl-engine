# owl 测试架构(test-architecture)

> 2026-10-13 立项。定"**两类测试 + 一 crate 一个端到端入口**"的纪律。
> 动机(实证,见 §四):端到端测试反复重装载(engine 套件 6× 27B)、
> 跨用例显存账本污染(单跑绿/套件红)、全量永红致结果失读。
> 与 pitfall §15(teardown 异步赛跑)/§16(批量 GPU 必须串行)配套。

---

## 一、两类测试(唯一分类,按"是否装载模型"判定)

| 类 | 判据 | 速度预算 | 位置 | 入口 |
|---|---|---|---|---|
| **单元测试** | 只测**单个函数/对象/单核**;**不装载模型** | < 5s | 与被测代码同文件 `#[cfg(test)] mod tests`;静态门放 `tests/<名>.lint.rs` | `#[test]` / `#[tokio::test]` |
| **端到端测试** | **涉及模型装载**(loader / engine / server 全链) | 秒~分 | `crates/<crate>/tests/<域>.e2e.rs` | **每 crate 唯一一个** |

**判定口诀:要装载模型 → 端到端;不装载(哪怕跑真 GPU 核)→ 单元。**

- 核 parity 探针(单核、自配张量、无模型)= **单元**;
- `gpu_*_identity_gate` / `*_e2e` / 装载 / 服务器全链 = **端到端**;
- 静态 lint(引用方向、注册表一致性)= **单元**(放 `tests/` 是因为要
  从 crate 外部审视,不代表它是端到端)。

---

## 二、端到端入口唯一化(硬纪律)

1. **路径**:`crates/<crate>/tests/<域>.e2e.rs`。
   例:`crates/models/tests/loader.e2e.rs`、`crates/engine/tests/engine.e2e.rs`。
2. **每 crate 只允许一个 `.e2e.rs`**。新增被测域 = 在该文件里加**场景**,
   **不许**开新文件、不许加新的 `#[tokio::test]`。
   理由:每个 `#[tokio::test]` 就是**一次完整装载**;入口唯一 ⇔ 装载唯一。
3. 该文件里**只有一个**测试函数:

```rust
// crates/models/tests/loader.e2e.rs
#[tokio::test]
async fn e2e() {
    // ① 环境**只在这里读一次**(显式依赖律:场景内零 env 读取)
    let Some(device) = env_usize("OWL_TEST_DEVICE") else { eprintln!("skip: 未设"); return };
    let Some(dir) = env_str("OWL_DFLASH2_DIR") else { return };

    // ② **唯一装载点**
    let mut gpu = owl_models::testkit::gpu_client(device).await;

    // ③ 顺序跑全部场景;逐个收集,不中断(失败汇总到最后)
    let mut fails: Vec<String> = Vec::new();
    for (name, r) in [
        ("draft_load_bench",  owl_models::formats::w4a16::e2e::draft_load_bench(&dir, &mut gpu).await),
        ("draft_behavior_ab", owl_models::formats::w4a16::e2e::draft_behavior_ab(&dir, &mut gpu).await),
        ("awq_load_smoke",    owl_models::formats::awq::e2e::awq_load_smoke(&awq_dir, &mut gpu).await),
    ] {
        eprintln!("[e2e] {name:<20} {}", if r.is_ok() { "PASS" } else { "FAIL" });
        if let Err(e) = r { fails.push(format!("{name}: {e}")); }
    }
    // ④ 汇总门:任一失败 → 红,并列出全部失败(不是只报第一个)
    assert!(fails.is_empty(), "E2E 场景失败 {} 项: {fails:#?}", fails.len());
}
```

4. **必须在 Cargo.toml 显式登记目标**(2026-10-13 实测坑):文件名按约定带点
   (`<域>.e2e.rs`),而 cargo 目标名**必须合法标识符**(含点报
   `invalid character '.' in crate name`)→ 显式 path/name 两件套:

   ```toml
   [[test]]
   name = "engine_e2e"           # 目标名(合法标识符)
   path = "tests/engine.e2e.rs"  # 约定文件名(带点)
   ```

   运行:`cargo test -p owl-engine --features e2e --test engine_e2e -- --test-threads=1`。

5. **遗留多入口文件为祖父条款**(`crates/kernels/tests/*.rs` 16 个、
   `crates/backends/cuda/tests/*.rs` 9 个):**只减不增** —— 要么并入该
   crate 的 `.e2e.rs`,要么降级为单元(单核、无装载)。

---

## 三、场景函数:写在源文件里,**禁止挂 `#[test]`**

场景体住在被测代码同文件(看得见私有项),但**不带** `#[test]`/`#[tokio::test]`,
由端到端入口**手动导入并调用**:

```rust
// crates/models/src/formats/w4a16.rs
/// E2E 场景面(仅 e2e feature 编译;生产构建整块剔除)
#[cfg(feature = "e2e")]
pub mod e2e {
    use super::*;

    /// 场景:草稿装载计时 + 设备重排抽样逐 u32 对拍
    pub async fn draft_load_bench(dir: &Path, gpu: &mut owl_cuda::GpuClient)
        -> Result<(), String> { /* ... 断言 ... */ }

    /// 场景:CPU 臂 vs 设备臂,草稿 memory 输出逐位一致
    pub async fn draft_behavior_ab(dir: &Path, gpu: &mut owl_cuda::GpuClient)
        -> Result<(), String> { /* ... */ }
}
```

### 3.1 可见性:走 feature,不许裸 `pub`

`tests/*.e2e.rs` 是**集成测试**,只能看见 crate 的公开面 —— 而 lib 编译时
**没有 `cfg(test)`**(`#[cfg(test)]` 里的场景**导不出来**)。故标准做法:

```toml
# crates/models/Cargo.toml
[features]
e2e = []          # 只开场景面;生产/普通测试零污染
```

```bash
cargo test -p owl-models --features e2e --test loader.e2e -- --test-threads=1
```

轻量变体(可接受):`#[doc(hidden)] pub mod e2e`(不设 feature),代价 =
公开面多一个隐藏模块。**禁止**把场景函数裸 `pub` 到正常路径上。

### 3.1b feature=e2e 的 lib 构建**不链接 dev-dependencies**(实测坑)

集成测试(`tests/`)能看见 dev-deps,但**它链接的 lib 不能** —— 而场景体住在
lib 里。故场景签名**只许用常规依赖的类型**:

| 不许(dev-dep) | 改法 |
|---|---|
| `&mut owl_cuda::GpuClient` | 泛型 `&mut D where D: DeviceClient`(owl_iface = 常规依赖) |
| `crate::testkit::gpu_client()`(`#[cfg(test)] pub mod testkit`) | **由入口构造**再传入(入口是集成测试,dev-deps 可用) |

副产品:客户端由入口构造并传入 = 「**一次构造、多场景复用**」的天然载体(§四.1)。
实例:engine 片(`crates/engine/tests/engine.e2e.rs`)已落地,`--lib` 从
38 用例 / ~6min → **22 用例 / 39s**。

### 3.1c 入口必须跑在**大栈线程**上(实测:栈溢出)

`.e2e.rs` 的入口把多层 async 内联进一个 future,而场景含**图捕获的声明树走查**
(深递归)→ libtest 默认栈(8MiB)**实测溢出**(`fatal runtime error: stack overflow`
@ dflash propose 图捕获)。故入口用 **非 async 测试 + 自建大栈线程 + current_thread runtime**:

```rust
#[test]
fn e2e() {
    std::thread::Builder::new().stack_size(64 << 20)
        .spawn(|| tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(e2e_body()))
        .unwrap().join().unwrap();
}
async fn e2e_body() { ... }
```

### 3.2 依赖显式(显式依赖律,与生产同律)

- 场景函数**收参数**(`&mut GpuClient`、模型目录、旋钮值),**不在场景内读 env**;
- env/配置只在 `.e2e.rs` 入口读一次,再显式传下去;
- 需要"关掉某个重档"的,用**显式参数**(`opt: &E2eOpts`),不用散点 env。

### 3.3 场景间不得共享可变状态

- 场景各自建 session / 实例;**需要同一实例的**,由入口持有并显式传入;
- 场景失败**不中断**后续(collect-then-assert),但**失败必须红**(不许
  `eprintln!` 后放过 —— 那是"测试面失效"的头号来源)。

### 3.3b **共享资源范式**(gpu_server.rs 样板;只合"贵的")

样板 = `crates/backends/cuda/tests/gpu_server.rs`:**一个大测试、一个资源、步骤顺序跑**
(步骤函数收资源为参数;头注列步骤清单)。

落地口径(2026-10-13 engine 片):

```rust
// 装载工厂:装载本身**即一步测试**(构造+装载+装配+捕获态断言+计时)
pub async fn boot_27b_default() -> RunningEngine<GpuClient> { ... }

// 行为步骤:收 Option<&mut RunningEngine> —— Some = 用入口的共享装载,
// None = 单跑自带装载(排障仍可单独跑)
pub async fn gpu_awq27b_marlin_e2e(shared: Option<&mut RunningEngine<GpuClient>>) {
    let mut own;
    let running: &mut RunningEngine<GpuClient> = match shared {
        Some(r) => r,
        None => { own = boot_27b_default().await; &mut own }
    };
    ... // 体内对 &mut 的再借用:drive_turn(&mut *running, ..)
}
```

**共享组的准入判据(实测教训,2026-10-13)**:并入同一组前,必须确认 **boot 期旋钮真的
兼容** —— 不只是"看起来一样"。反例:`gpu_schedule_decision_surface` 依赖
`max_seq=64/chunk=4` 的**切块语义**,并入 `4096/32` 的共享档后**实测红**
(断言"首块切块")→ 切块/预算敏感的用例一律自带装载。

**只合"贵的"**:贵 = **模型装载**(27B ~24-32s/次)。便宜的不动(0.8B ~2-5s,
合了省不到哪去,反而增加耦合)。**新步骤默认并入共享装载组**;
只有 boot 期配置不同的才自带装载(草稿类型 / 池几何 / fp8 / capture_slab / 采样开关)。

### 3.3c **带草稿的引擎:装载一次、两轮**(投机门专用,2026-10-13 用户令)

投机恒等门天生要**两轮**(一轮不投机 baseline、一轮投机)。旧写法 = 每轮各建一个
引擎(**2 次 27B 装载 + 2×10s 收尾**)→ 纯浪费。

`RunningEngine.spec_depth` 是 **`pub(crate)` 运行时可切字段**(`exec/spec.rs` 每轮读;
`0` = 回落 DecodeBatch 非投机)→ **一次装载两轮**即可:

```rust
let mut running = boot_gate_engine(ordinal, &dir, &ddir, max_new).await; // 带草稿装载一次
let (ta, ta_tokens) = round(&mut running, 0, &prompt, max_new, "baseline").await; // 不投机
let (tb, tb_tokens) = round(&mut running, depth, &prompt, max_new, "spec").await; // 投机
assert_eq!(ta_tokens, tb_tokens);   // 恒等门
drop(running); std::thread::sleep(Duration::from_secs(10));  // 收尾屏障仅一次
```

实测(engine):dflash2 门 **75s → 35s**、MTP 门 **75s → 34s**(各省一次装载 + 一次 10s 收尾),
两门 AL 与 token 流结论不变(dflash2 AL=1.17 / MTP rounds=17 sum_m=22,均逐位一致)。
**草稿档一律按投机配置 boot**,baseline 轮靠运行时切 `spec_depth=0`。

### 3.4 重场景必须自带预算

| 档 | 触发 | 例 |
|---|---|---|
| 默认档 | 入口无条件跑 | 装载计时 + 抽样对拍(≤15s) |
| 重档 | **显式开关** `OWL_DRAFT_AB=1` 或 `#[ignore]` + `gpu_slow_` 前缀 | 全量 40 项对拍 + 双臂装载(~90s) |

---

## 四、为什么值得这么严(三条收益,均有实证)

1. **时间**:engine 套件有 6 个 27B 用例(`chat_inference` / `raw_completion_diag` /
   `awq27b_marlin_e2e` / `decode_marginal_bench` / `dflash2_identity_gate_27b` /
   `spec_identity_gate_27b`)→ **6 次整模装载**;装载修好后 24s/次 × 6 = 144s,
   装载修好前 100s/次 × 6 = 10min。合一后 **1 次**。
2. **正确**:跨用例**显存账本污染** —— 27B 用例跑完后治理面 `live` 变
   `155G` 假值 → 后续 0.8B 用例 `budget=-134G` → 池上限 0 tok →
   `KV 块池耗尽` 必红,**单跑即绿**(pitfall §15 的套件级联假回归同源)。
   引擎实例少了,这类"必须单跑"的税自然消失。
3. **可读**:一张 `PASS/FAIL` 表代替"哪个 test 挂了、挂在哪一步"。

---

## 五、档位与唯一 runner

**唯一入口 = `tools/test.sh`**(禁止手敲 `cargo test` 组合命令,避免旋钮靠口口相传):

| 档 | 内容 | 约时 |
|---|---|---|
| `tools/test.sh fast` | 单元(CPU,无 GPU/无模型)+ 编译门 | < 30s |
| `tools/test.sh unit [filter]` | 单元 + 单核 GPU 探针 | 1~3 min |
| `tools/test.sh e2e` | 各 crate 的 `.e2e.rs`(串行) | 依场景 |
| `tools/test.sh full` | 全档 + 超时看门狗 | 10~15 min |

runner 内固化(不许各人自配):

- `OWL_TEST_DEVICE=1` + 模型目录(`OWL_AWQ27B_DIR` / `OWL_DFLASH2_DIR`);
- GPU 一律 `-- --test-threads=1`(**pitfall §16**:并行 = 多 CUDA 上下文
  同卡并发图捕获 → `STREAM_CAPTURE_INVALIDATED` / ILLEGAL_ADDRESS 连环假挂);
- **每档带 `timeout`**(防挂死静默吃半小时);
- 默认**不开** `--nocapture`(除排障;它会让输出爆炸并拖慢)。

---

## 六、改动面 → 测试子集(跑什么,不跑什么)

**禁止"改一行跑全量"**(磨洋工);全量只在里程碑/结案跑。

| 改了什么 | 只跑 |
|---|---|
| 单核 `kernels/cu/**` | 该核 parity 探针 + 1 个端到端冒烟场景 |
| `kernels/family/marlin/**` | repack 位等门(`zp_parallel_matches_serial_reference`)+ `marlin_parity` 相关项 |
| `formats/{awq,w4a16}.rs` | 该格式位等门 + `models/tests/loader.e2e.rs` |
| `layers/linear.rs` | AWQ e2e 场景 + dflash2 场景 |
| `env.rs` / KV 布局 | kNHD parity 三件 + nospec A/B 场景 |
| `graph_plan.rs` | 启动分相计时 + 1 个图族场景 |
| 文档 / 配置 | `cargo check`(+ 配置单测) |

---

## 七、测试面自净纪律

- **不许长期挂红**:要么修,要么删,要么 `#[ignore = "wip: 原因"]` 并注明
  出处。**"全量永红" = 测试面失效**(本次立案的既有红:测试自写错形状的
  `gpu_real_weights_prefill`、examples API 漂移、已回滚核的 tg4 探针)。
- **不许"静默放过"**:`eprintln!("skip")` 只允许用于**环境缺失**
  (未设 `OWL_TEST_DEVICE` / 模型目录),不允许用于"断言没过但算了"。
- **测试也受预算约束**:新场景必须报耗时;超预算的走 §3.4 的重档开关。
- **跑过什么要留痕**:结案/施工文档里记"跑过哪些档、结果如何"。

---

## 八、迁移清单(现状 → 目标)

| 现状 | 目标 |
|---|---|
| ✅ **已落地(2026-10-13)**:`crates/engine/src/tests.rs` 16 个 GPU 用例 → `pub async fn`(无 `#[test]`);入口 = `crates/engine/tests/engine.e2e.rs`(单 `#[tokio::test] e2e()`,env 一次读,`OWL_E2E_ONLY=<子串>` 可单跑)。**`--lib` 38 用例/~6min → 22 用例/39s** | ✅ **Step B(2026-10-13)**:两个投机门(DFlash2 / MTP)**各装载一次跑两轮**(省 2×~40s;见 §3.3c)+ 0.8B 默认档 **5 步共用一次装载**(`boot_08b`,实测 4.1s 一次;切块敏感的 `schedule_decision_surface` 退出该组自带装载)+ 27B 默认档 **2 步共用一次装载**(`boot_27b_default`,实测装载 32.3s 一次;步骤收 `Option<&mut RunningEngine>` 范式见 §3.3b)。其余 4 个 27B 步骤 boot 期配置各不相同(草稿类型/池/fp8/slab)故各自装载;0.8B 档按"只合贵的"不合并 |
| `crates/models/src/specs/qwen35.rs`(8)/`layers/dflash2.rs`(8)/`layers/attention.rs`(13)`tokio::test` | **待做**:装载类(≈12 个:`gpu_real_weights_*` / `gpu_dflash2_*` / `gpu_model_step_diag` / `gpu_matmul_dim_probe` / `gpu_vram_manifest_checksum` / `cpu_real_weights_load` / `mtp_head_loads_from_awq27b`)→ `crates/models/tests/loader.e2e.rs`;`attention.rs` 的单核探针(`gpu_narrow_matches_host`/`gpu_split_probe`/`gpu_fi_prefill_probe` 等)保留为单元。**前置**:按 §3.1b 把签名泛型化(`owl_cuda` 是 models 的 dev-dep) |
| `crates/kernels/tests/*.rs`(16 文件) | 单核探针**单元化保留**;跨模块链式 → 并入 `crates/kernels/tests/kernels.e2e.rs` |
| `crates/backends/cuda/tests/*.rs`(9 文件) | 同上(`cuda.e2e.rs`) |
| `crates/models/tests/facedir_lint.rs` | 保留(静态门 = 单元类) |
| `apps/server/tests/e2e.rs` | 已是 server crate 的唯一 e2e 入口,按本文件口径收口(env 一次读、场景化) |

迁移顺序建议:① 先落 `tools/test.sh` + §六 映射表(立刻止痛);
② engine 套件场景化(省 2min/轮 + 消污染);③ models 装载类收敛到 `loader.e2e.rs`;
④ 清既有红。

---

## 九、参考

- `docs/arch/common-dev-pitfall.md` §15(teardown 异步赛跑)/§16(批量 GPU 必须串行)
- `docs/arch/requirements.md`(验收线口径)
- `docs/arch/charter.md` A1(图治理)/A5(显存预算) —— 测试档位与治理账本同律

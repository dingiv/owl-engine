# logger 框架(log)—— 事件流出口,与 metrics 同风格不同定位

> 2026-10-14 立项(用户指令:metrics 是 metrics、logger 是 logger;同款宏风格;
> 不开特性就编译期消失)。姊妹卷:`perf-framework.md`
> (数值账)。落点:`crates/shared/src/log.rs`。

---

## 〇、为什么要有它

性能框架(P1-P4)把**数值账**收口了,但仓里还有 **710 处裸打印**
(`eprintln!/println!`):装配报告、装载进度、探针分账、错误回卷……
它们的共病:

| 病 | 后果 |
|---|---|
| 无级别 | 装配报告与错误同权重;`--quiet` 无从下手 |
| 无开关 | 生产/CI 里刷屏,只能靠改代码 |
| 无 tag 过滤 | 排障时想看 `[gl-prof]`,却被 `[st]` 刷掉 |
| 无落盘/切分 | 长跑日志只能重定向,无 rotation,盘会被吃满 |
| 无统一渲染 | 40+ 个 tag 各写各的格式 |

## 一、定位分界(与 metrics 不互相替代)

| | [`crate::metrics`] | **本模块** |
|---|---|---|
| 语义 | **数值账**(计时/计数魔集) | **事件流**(离散、带级别) |
| 用途 | 性能判读(可聚合/可导出) | 排障、装配报告、异常回卷 |
| 形态 | 采样 → 聚合报告(`p50/p99/self`) | 一行一条,顺序不可丢 |
| 丢失容忍 | 丢样本只损精度(有 `dropped` 计数) | 丢一行就是丢一个事实 |
| 出口 | `/debug/perf`、CSV/JSON | stderr + 日志文件(tee) |

## 二、三层机制(与 metrics 对齐)

1. **宏风格**:`log_error!` / `log_warn!` / `log_info!` / `log_debug!` /
   `log_trace!(tag, fmt, args...)` —— 与 `timer_start!/perf_span!` 同款;
2. **编译期门**:cargo feature `log`,**已入 `owl-shared` 的 `default`**
   (用户裁决 2026-10-14:默认开)。不开时五级宏全部是编译期剔除
   (双定义法 ⇒ 下游零 `cfg` 样板、零 `unexpected_cfgs` 告警)。
   空缺档宏体 = `if false { let _ = (tag, format_args!(..)); }`:
   **参数不被求值**(优化器整块消去),但仍是**词法使用** ⇒ 收编后
   不会出现"变量只在日志里用过"的 `unused_variables` 告警;
3. **运行期门**(logger 内):级别阈值 + **tag 前缀白/黑名单**(黑名单优先)
   + tee 落盘 + **按尺寸 rotation**。

### 缺省级别 = WARN(用户裁决 2026-10-14)

**开箱只出 warn/error** —— 装配报告/装载进度等 info 面默认不输出:

```bash
cargo build --release -p server --features perf   # log 已在 default 里
./target/release/server …                         # 只有 warn/error + 就绪横幅
./target/release/server … --log-level info         # 要装配报告
```

- 「安静」由**级别**承担(默认 `warn`),feature `log` 默认编译进来;
- 要**连编译都去掉**(逐字节静默):依赖侧把 `owl-shared` 改成
  `default-features = false`(或 `--no-default-features` 构建该包),
  此时五级宏展开为空语句、`[tag]` 与参数都不求值;
- 拒启类错误(引擎装配失败)**恒输出**(裸 `eprintln!`,不经级别门)+
  就绪横幅恒输出 ⇒ 运维至少知道"起来了 / 起不来 + 端口"。

## 三、渲染格式

```text
[INFO] [boot] 引擎就绪(捕获态 Captured)
[WARN] [actor] pump 错误: …
[INFO] [gl-prof] 图节点 = 1044(类型 {1: 1044})
```

- `[LEVEL] [tag] msg`;`timestamp = true` 时前置 `2026-10-14T00:12:33.123Z `
  (零依赖自算 civil date);
- `[tag] msg` 部分与旧 `eprintln!("[tag] …")` **逐字相同** ⇒ 收编是机械
  替换,既有脚本/测试的 grep 不受影响;
- ⚠️ 与旧输出的唯一差别 = 新增 `[LEVEL] ` 前缀;按行匹配 `"[boot] …"` 的
  脚本仍命中(`contains` 语义),按**行首**匹配的需同步。

## 四、落盘与 rotation

- `Target`: `Stderr`(缺省)/ `Memory`(测试断言/宿主自取)/ `Tee`(stderr+文件)
  / `File`(仅文件);
- `max_mb > 0` 时按尺寸切:`srv.log` → `srv.log.1` → `srv.log.2` …,
  最多 `keep` 份(超出丢最旧);`keep = 0` = 不留备份直接截断;
- 全部 I/O 走 [`crate::file_loader`](file_loader.rs)(统一入口律:本次为它
  新增 `append`/`rename` 两个原语)。

## 五、配置与出口

```toml
[log]
level = "info"        # off|error|warn|info|debug|trace
tags_allow = []       # 空 = 全放行(前缀匹配)
tags_deny  = []       # 优先于 allow
file = "/var/log/owl/srv.log"   # 缺省 None = 仅 stderr
max_mb = 64           # 0 = 不切
keep = 3
timestamp = false
```

CLI 覆盖:`--log-level <L>` / `--log-file <PATH>`。

运行期(免重启):

```bash
curl 127.0.0.1:8136/debug/log                       # 状态(level/tags/build)
curl 127.0.0.1:8136/debug/log/level/debug           # 调级
curl "127.0.0.1:8136/debug/log/tags/boot,gl-prof/-" # allow=boot,gl-prof;deny 空
```

## 六、收编台账(生产面 ≈238 处,分三批)

| 批 | 范围 | 处数 | 状态 |
|---|---|---|---|
| ① | `apps/server/src/**` + engine boot/装配(`engine.rs`/`state.rs`/`blocks.rs`) | **45** | ✅ 2026-10-14 |
| ①′ | **开机噪音面**:models(`formats/*`/`kernel.rs`)+ cuda(`state.rs`/`server/mod.rs`/`harvest.rs`/`launch.rs`)+ `shared/metrics.rs` | **50** | ✅ 2026-10-14 |
| ② | engine 剩余(探针/分账/`[sched]`/`[*-prof]`/`[*-probe]`) | 58 | ✅ 2026-10-15 |
| ③ | models 层内探针(`layers/**` 的 `[gdn-*]`/`[enc-forensic]`/`[m-sweep]`) | ≈90 | 待办 |
| ④ | **`crates/kernels` Rust 面**(装配报告 / 自检 / `[gdn-diag]` / `[l3-audit]` / `[owl-fi]`) | **23** | ✅ 2026-10-21(见 §八) |
| ⑤ | **`cu/**` C/C++ 面**(`[owl-fi]` / `[marlin-*]`)+ 一处无门设备 printf | **13** | ✅ 2026-10-21(见 §八) |
| ⑥ | **engine 探针面**(`cov` / `recon` / `chain-inc` / `snap` / `idx` / `vram` / `pf-tap` / `live-diag` / `prefix` / `hook`) | **24** | ✅ 2026-10-21(§八 同律;bool 保留 = 采集,打印走 `log_debug!`) |
| ⑦ | **models 装载报告**(`load` / `load-verify` / `dump` / `w4a16` / `gguf` / `dflash2` / `boot` / `store` / `disk` / `tp` / `turn-ws`) | **22** | ✅ 2026-10-21(装配/装载 → `info`;逐键 → `debug`) |
| — | `crates/*/tests` + `examples` | 184 | **不动**(测试输出本就该直出;`skip:` 行见 §九) |
| — | `build.rs`(kernels) | 45 | **不动**(构建期基础设施) |

**实测(2026-10-14,27B 生产档开机)**:收编批① / ①′ 之后
**默认输出 337 行 → 10 行**(其中 9 行 = 未收编的 kernels 面 + 1 行就绪横幅);
`--log-level info` 恢复完整 335 行装配报告(325 行 `[INFO]`)。

级别约定:`[boot]/[load]/[srv]` → `info`;`[conn]/[actor]/[engine]` →
`warn`;`[*-probe]/[*-dbg]/[*-prof]/[blk]` → `debug`/`trace`。
**格式保持原位**(只做前缀替换),P5 散装仪表并轨时再统一考虑
"日志 + metrics 双出口"。

## 七、纪律

1. tag **必须是字面量**(`&'static str`)—— 与 metrics 同款契约;
2. 新增打印一律用宏,禁止新裸 `eprintln!`(测试/`build.rs` 除外)。下层
   (`crates/kernels`、`cu/**`)用 §八 的注入口 —— **没有"零依赖 crate 例外"了**;
2b. **拒启/致命路径用裸 `eprintln!`**(必须绕过级别门 —— 静默的拒启是运维事故);
3. 诊断转储(`[dump]`)走 `log_debug!` + 保留落盘失败可见;
4. 日志宏**不进热路径**(逐 token/逐节点)—— 那是 metrics 的活。

## 八、下层与 C/C++ 输出口(2026-10-21;kernels + `cu/**` 一次收编)

### 8.1 为什么不能"kernels 依赖 owl-shared"

`owl-shared → owl-iface → owl-kernels` 是既成边 ⇒ `kernels → shared` =
**cyclic package dependency**(cargo 实测报错)。而 kernels 又必须出声。

解法 = **把 `LogSink` 这个纯函数指针包搬到依赖图叶子**:

```
owl-shared ──► owl-iface ──► owl-kernels
     │                            │
     └──────► owl-sink ◄──────────┘        (零依赖 ABI crate,唯一 LogSink 定义)
```

`crates/sink`(package `owl-sink`)只装:级别常量 `LVL_*` / `level_name_of` /
`LogSink`。`owl_shared::log` 与 `owl_kernels::sink` 都是它的**使用者**
(不各持一份镜像结构 —— AGENTS.md 禁"接口不唯一")。

### 8.2 三道门(与 Rust 侧逐条同构)

| 门 | Rust(kernels) | C/C++(`cu/**`) |
|---|---|---|
| 编译期 | cargo feature `log`(缺省开);关 ⇒ `klog_*!` 空展开 | 同 feature → `build.rs` 决定是否 `-DOWL_LOG_ENABLED`;关 ⇒ `OWL_LOG_*(…)` = `((void)0)` |
| 运行期分级 | `LogSink::enabled(level, tag)` | 同一函数(`owl_log_on` → 同一 `LogSink`) |
| tag 过滤 | `/debug/log/tags/a,b/-x` | 同一个白黑名单(判据只有一套) |
| 调用点 | 先准入再 `format_args!` | 先 `owl_log_on` 再 `vsnprintf`(1024B 栈缓冲,超长截断) |

实测(对象级):`-DOWL_LOG_ENABLED` 关时目标文件**不含任何 `owl_log_*` 引用**
(`nm -u` 为空;679B → 104B)。

### 8.3 接线(唯一写点)

```rust
// owl-cuda::GpuServer boot(与既有采集旋钮同一处)
owl_kernels::server::configure_diag(&DiagTune {
    sink: self.sink,              // ← LogSink;C 与 Rust 同时可用
    l3_audit: self.diag.l3_audit, // 采集
    gdn: /* GdnDiag */,           // 采集
})?;                              // 重入不同值 = 拒启(不静默覆盖)
```

`kernels::sink::configure` 是**唯一写点**(`OnceLock`,boot 后冻结;同值幂等、
异值 `SinkError::AlreadyConfigured`)。C ABI `owl_log_on` / `owl_log_emit`
只读它 —— 没有第二条注入路径。

### 8.4 C 侧写法

```c
#include "owl_log.h"          // -Icu

OWL_LOG_INFO("selfcheck", "%s ✓", name);          // 装配报告
OWL_LOG_DEBUG("gdn-diag", "T=%d nz=%zu", t, nz);  // 探针(level=debug 才出)
OWL_LOG_ERROR("owl-fi", "workspace 不足: 需 %zuB", need);  // 错误码路径
if (OWL_LOG_ON(OWL_LVL_TRACE, "ar")) { /* 热循环里先问再组参 */ }
OWL_DEV_PRINTF("...");   // __device__ 用;仅 OWL_LOG_DEVICE 编译门(默认关)
```

tag 约定与 Rust 面同名同义:`owl-fi` / `marlin-cfg` / `marlin-v2` /
`marlin-time` / `selfcheck` / `xqa`。

### 8.5 本轮一并退役的"纯打印开关"(裁决:**bool 管采集,日志管可见性**)

| 退役项 | 原形态 | 现形态 |
|---|---|---|
| `[probes] fi_debug` | 配置键 → `DIAG` → C++ `getenv("FI_DEBUG")` | 删;`owl-fi` tag + level 决定 |
| `FI_DEBUG` env 桥 | `std::env::set_var("FI_DEBUG", …)` | 删(不再有 env 参与) |
| `MARLIN_CONFIG_DEBUG` / `MARLIN_DEBUG` | C++ 内 `getenv` 纯打印开关 | 删;`marlin-cfg` / `marlin-v2` tag |
| `[knhd-chunked-W]` 设备 printf | 每次 prefill 无条件打 8 行 | 删(取证残留;设备面只留 `OWL_DEV_PRINTF`) |

**保留为"采集"的 bool**(它们不是打印开关,是额外工作量):
`[probes] l3_audit`(登记去重集)、`gdn_diag/gdn_touch/…`(dtoh 回读/往返)、
`trace_gate` / `step_gate` / `pf_fence`(路径与 A/B 语义)。

### 8.6 保留裸打印(白名单;别再审这些)

| 位置 | 为什么必须裸打 |
|---|---|
| `apps/server/src/main.rs`(12) | **拒启/参数非法/logger 初始化失败** —— logger 还没起来,或用户显式 `--help`/`--print-config` 的产品输出 |
| `apps/server/src/config.rs`(4) | **选卡发生在 logger init 之前**(源码里已注明"logger 前 ⇒ 裸输出") |
| `apps/server/src/config.rs`(2 `println!`) | `--print-config` 的 **stdout 产品**(要被 `jq`/测试解析,不许加前缀) |
| `apps/cli/src/main.rs`(10) | 交互式 CLI 的**用户界面输出**(不是日志) |
| `crates/backends/cuda/src/vmm.rs`(3) | 回滚失败 = **泄漏风险**,必须绕过级别门 |
| `crates/backends/cuda/src/gpu_client.rs`(1) / `server/mod.rs`(1) | server 线程退出 / 回执槽越界 = 致命或编程错误 |
| `crates/shared/src/env_reader.rs`(1) | 解析失败发生在 **logger init 之前**,且是"静默降级"的反面证据 |
| `crates/shared/src/testkit/report.rs`(3) | 测试面出口**本尊**(见 §九) |
| `crates/models/src/port_tests.rs`(2) | 文件由 `#[cfg(test)] mod port_tests;` 门控(非生产) |
| `crates/kernels/*` 的 `#[cfg(test)]` 模块 | 测试输出 |

判据: **错过这一行会不会让人看不见事故?** 会 ⇒ 允许裸打(纪律 2b 的扩展);
只是"想知道细节" ⇒ 走 `log_*!` + level/tag。

### 8.6b HAL 契约面(`owl-iface`)的注入(2026-10-21 补齐)

`owl-iface` 也不能依赖 `owl-shared`(`owl-shared → owl-iface` 成环),但**可以**
依赖叶子 `owl-sink` —— 于是 `DeviceHandle` 直接持一个 `LogSink`:

```rust
pub struct DeviceHandle<D: DeviceClient> { inner, ledger, ctx, sink }
impl<D: DeviceClient> DeviceHandle<D> {
    pub fn new(inner, ledger) -> Self      // sink = LogSink::none()(测试/纯转发的缺省)
    pub fn with_sink(self, sink) -> Self   // 上层注入
}
```

- **唯一注入点** = `engine` 构造句柄处(`on_with_ledger`):
  `.with_sink(owl_shared::log::sink())`;
- 克隆路径(`face_clone` / `loader_faces`)**同源携带**(克隆不改可见性);
- `CtxScope` 也随身带一份(Drop 里的锁中毒告警要能落地);
- 7 处 `[device-handle]` 全部收编:池排空/回填释放失败、捕获窗失败、
  `graph_abort` 失败 → `WARN`;`PoolExhausted ⇒ 排空复用池` → `INFO`(自愈,不是故障);
- `Ctx::bump` 按值收 `LogSink`(`MutexGuard` 独占借用时按引用会撞借用检查),
  **未注入 = 静默但计数照记**(`hook_fail` 是机读面,不因日志静音而丢账);
- 单测 `handle::tests::ctx_bump_goes_through_sink` 锁这两条。

### 8.7 未收编(下一轮,明账)

- `MARLIN_TIME` / `MARLIN_NO_M8` / `MARLIN_THREADS` / `MARLIN_NO_OVERRIDE`
  —— C++ 内 `getenv` 行为/计时旋钮(前者注释已写"测完即删");按"配置面
  显式化"处理,不与打印混在一起改;
- 生产面已从 126 → **39**,全部是 §8.6 白名单(无待收编项)。

## 九、测试面:`skip:` / 探针行的唯一出口

`cargo test` 按用例捕获 stderr ⇒ 测试输出**不能**走 logger(需要 init +
level 门,缺省 WARN 会吞掉 skip 行 = 假绿)。故测试域有独立出口:

```rust
use owl_shared::testkit::report;

report::skip("OWL_TEST_DEVICE 未设");           // → "skip: OWL_TEST_DEVICE 未设"
report::note("gdn", format_args!("T={t}"));     // → "[gdn] T=512"
owl_shared::test_note!("gdn", "T={t}");         // 宏形态(导出在 crate 根)
```

- `skip:` 前缀**只由 `report::skip` 给**(常量/`missing_reason` 返回正文);
- `#[owl_test]` 的三个门(GPU / HF 对拍 / 资源清单)全部改走它 ——
  270 处手写 `eprintln!("skip: …")` 已统一(2026-10-21);
- 探针行统一 `[tag] 正文`(与生产 `log_*!` 渲染同形,跨用例 grep 通道一致)。

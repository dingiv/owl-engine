# logger 框架(log)—— 事件流出口,与 metrics 同风格不同定位

> 2026-10-14 立项(用户指令:metrics 是 metrics、logger 是 logger;同款宏风格;
> 不开特性就编译期消失)。姊妹卷:[`perf-framework.md`](perf-framework.md)
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
| ② | engine 剩余(探针/分账/`[sched]`/`[*-prof]`/`[*-probe]`) | ≈75 | 待办 |
| ③ | models 层内探针(`layers/**` 的 `[gdn-*]`/`[enc-forensic]`/`[m-sweep]`) | ≈90 | 待办 |
| ⚠️ | **`crates/kernels`(13 处)**:kernels 的 **lib 零依赖律**禁止依赖 `owl-shared` | 13 | **挂账**:需 `LogSink` **注入**设计(fn 指针经 Exec/registry 初始化面显式传入,不破依赖图) |
| — | `crates/*/tests` + `examples` | 184 | **不动**(测试输出本就该直出) |
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
2. 新增打印一律用宏,禁止新裸 `eprintln!`(测试/`build.rs`/**零依赖 crate** 除外);
2b. **拒启/致命路径用裸 `eprintln!`**(必须绕过级别门 —— 静默的拒启是运维事故);
3. 诊断转储(`[dump]`)走 `log_debug!` + 保留落盘失败可见;
4. 日志宏**不进热路径**(逐 token/逐节点)—— 那是 metrics 的活。

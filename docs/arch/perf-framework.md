# 性能框架(perf)—— metrics 增强 + 解释器关键路径钩子

> 2026-10-14 立项。动机:性能会战进入「prefill 与 decode 双线优化」阶段,
> 需要一个**统一、可分级、可导出**的耗时分布观测面 —— 现有 `metrics.rs`
> 是通用打点器,不是性能框架;现有性能仪表(`[sched]`/`[gl-prof]`/
> `srv_timing`/`launch_time`/`d2h_prof`/`gpu_prof`)散落在各层,格式各异、
> 无统一出口、无法离线比对。
>
> 定位:框架**不新增数据面哲学** —— 它把「模型解释器是唯一执行者」这条
> 既有结构红利兑现为观测红利:全部节点归约收口在 `_eval_node` 一处,
> 于是「每个算子的宿主耗时 + 设备耗时 + 调用树」可以一次声明、全图生效。
>
> 上游:`charter.md`(A1 图治理 / A5 严格显存预算)、`interpreter-tap.md`
> (观测面先例:值观测)、`common-dev-pitfall.md`(口径律)。
>
> **姊妹卷**:事件流出口见 [`log-framework.md`](log-framework.md)
> (logger;同为宏风格 + 编译期 feature 门,但定位是排障/装配而非性能判读)。

---

## 〇、现状与三条软肋

`crates/shared/src/metrics.rs`(804 行)已提供:全局 store + tag 字符串的
timer/counter/gauge + 过滤器 `tag/prefix/contains/min_count/since/sort/limit`
+ 聚合 `count/total/min/max/avg/p50/p99/open/orphan/dropped` + `/debug/metrics`
JSON 出口 + 直用 API(`with_metrics_store`)。

| # | 软肋 | 后果 |
|---|---|---|
| S1 | **宏是 debug-only**(`#[cfg(debug_assertions)]` 空展开) | 性能测量全在 **release**;主力宏在性能场景里是哑的,只能靠 `mrec/mcnt` 直用 API 绕行 |
| S2 | **无层级 / 无排他 / 无 epoch** | tag 平铺累计:`engine.pump` 与 `engine.pump.schedule` 双计;读不出「self 时间」、读不出 % of wall、读不出「只看本轮」 |
| S3 | **设备侧缺席** | store 只装宿主墙时;GPU 逐核时间在 cuda `DiagOpts`,不进 store、口径不一、不可导出 |

---

## 一、分级与可见性(裁决 S1)

**两级机制(用户裁决 2026-10-14)**:

1. **编译期**:`owl-shared` 新增 cargo feature `perf`。宏与实现桥的 cfg 由
   `debug_assertions` 放宽为 `any(debug_assertions, feature = "perf")`。
   生产二进制不带 feature = **逐字节零成本**(不新增任何运行期分支);
   基准二进制 `cargo build --release -p server --features perf` 全速可观测。
2. **运行期分级**(store 内一个 `AtomicU8`,读一次 ≈ 1ns):

| 级 | 名 | 记录什么 |
|---|---|---|
| 0 | `off` | 全关(**零成本基线的对照臂** —— 用于量出仪表本身的税) |
| 1 | `counters` | counter / gauge(计数无时延噪声) |
| 2 | `host` | + 宿主计时(**默认**;与历史行为逐位一致) |
| 3 | `gpu` | + 设备侧采集(eager 逐发射 event 对 / 图态节点账) |

- 默认 = `host`:历史打点行为**零变化**(`off` 是新增能力,不是新缺省)。
- 配置面:`[probes] perf_level = "off"|"counters"|"host"|"gpu"`(boot 一次
  `set_level`);`/debug/perf/level/<name>` 可在线切换(免重启 A/B)。
- 门在 **store 入口**(`timer_begin`/`timer_record`/`counter_add`),不在宏体
  —— 直用 API 与宏走同一道门,不可能绕。

## 二、tag 命名律

```
<plane>.<class>[.<detail>]
```

- plane ∈ {`engine`, `interp`, `srv`, `gpu`, `load`, `spec`}(小写点分);
- detail ∈ {算子类名, 核名, 阶段名} —— **必须有界**(算子类 = 登记表声明集;
  核名 = `[gl-prof]` 同一集合),禁止把 token id / 块 id 灌进 tag
  (动态 tag 增殖已有 `fact_probe` 门控先例);
- 已有 tag(`engine.pump.schedule` / `spec.*` / `load.*` / `srv.*`)**不动**
  —— 命名律向后兼容,新面按律生长。

本框架新增面:

| tag | 含义 |
|---|---|
| `engine.phase.prefill` / `.decode` / `.spec` | 阶段包络(父节点,给下面所有钩子定相) |
| `interp.host.node.<class>` | 单节点宿主总时(按算子类) |
| `interp.host.seg.<seg>` | `Call` 臂宿主分相:`resolve` / `build` / `alloc` / `lower` / `launch` |
| `interp.count.node` / `.launch` / `.cse_hit` / `.view` | 计数面 |
| `gpu.kern.<kernel>` | 逐核设备时间(eager event 对;步骤 2) |
| `gpu.graph.node.<kernel>` | 图材料单(捕获期节点清单入账) |

## 三、层级 / 排他 / epoch / 导出(治 S2)

- **父指针(pending stack)**:每线程一条 open span 栈(原按 (tag,thread)
  分栈改为**单栈**),`timer_end` 记录 `parent` = 其下最近未闭合 span 的 tag。
  ⇒ 样本自带**真实调用树**,不需要调用方声明层次。容错:end 找不到栈顶同
  tag 时向下找最近匹配(计 `out_of_order`),再无则 orphan(计 `orphan_ends`)。
- **exclusive/self**:`collect` 时一次扫描得 `parent → Σdur`,则
  `self = total − Σ(直接子样本)`,报告出 `self%`(真正的热点判据:平铺
  total 会把父节点重复计一遍)。
- **rollup**:按前缀折并(`render_tree` 用),树形缩进输出。
- **epoch**:`mark_epoch()` 记时刻,配合既有 `since` 过滤 —— 「只看这一轮」
  不必 reset(历史上「同 session 二发走前缀缓存污染差分」就是缺 epoch 的代价)。
- **导出**:CSV / JSON 两种落盘(离线比对多配置);`--perf-dump <path>`
  在每轮结束覆写一次(进程被 kill 也不丢账)。

## 四、解释器钩子(L2;兑现「唯一执行者」红利)

落点 = `crates/models/src/interpreters/eval.rs::_eval_node`(**单一归约点**)。

```
_eval_node(t)
  ├ CSE memo 命中                        → count.cse_hit
  ├ 父递归(展开)
  ├ 分支配重 → class(算子类)
  │   Htod → "htod" | Zeros → "alloc" | Matmul/MatmulNt → "gemm"
  │   Reshape/SliceView/SlotWrite/Block → "view"
  │   Call  → **pick.name**(= 核名,登记表有界集)
  │        ├ seg.resolve    = driver::resolve(纯 CPU 选核)
  │        ├ seg.build      = kernel 构造 + dtype 守门
  │        ├ seg.alloc      = alloc_uninit(await)
  │        ├ seg.lower      = lower_kernel(线格式构造)
  │        └ seg.launch     = face.launch(await;进 server actor)
  ├ reclaim free                         → count.reclaim
  └ 节点总时                            → host.node.<class>
```

- **level 读一次**:eval 入口读 `metrics::level()`,存 `EvalCtx.perf`,热路径
  零全局查询(承接「显式依赖律」:boot 一次 / 入口一次)。
- **零开销律**:`off`/`counters` 级下 `Instant::now()` 都不调(单条
  `if perf >= HOST` 分支)。
- **阶段定相**不写进 tag(会把 tag 面 ×3),由 `engine.phase.*` 包络 span 的
  父指针承担 —— 看「prefill 的分布」= 树里那一棵子树。
- 钩子**只读不写**:不改变计算、不新增 launch/alloc(与 tap 同律)。

## 五、散装仪表并轨(L3;同一 store、同一出口)

| 现有 | 并轨为 |
|---|---|
| `[sched]` 累计打印 | `engine.pump.schedule`(已有)+ 保留打印 |
| `[gl-prof]` 图材料单 | `gpu.graph.node.<kernel>` counter + `gpu.graph.nodes` gauge |
| `srv_timing` 逐命令计时 | `srv.cmd.<name>` timer + `srv.cmd.ns` counter |
| `launch_time` | `srv.launch` timer |
| `d2h_prof` | `srv.dtoh.wait` / `srv.dtoh.copy` timer |
| `gpu_prof` 逐发射同步差分 | `gpu.kern.<name>` timer(与 event 对同 tag,口径在报告里标) |

纪律:**旧打印保留**(工具链与既有判据依赖其格式),并轨 = 同数据多一个出口,
不是替换。逐点收编,不追求一次清完。

## 六、口径纪律(承接既有律)

1. **仪表自身成本必须可测**:`off` vs `host` 两跑同一负载,差值 = 税
   (eager 逐发射 event 对的税最大,预期 >20%,仅诊断用)。
2. **宿主时间 ≠ 设备时间**:`interp.host.*` 是 CPU 侧提交成本(GIL 级),
   `gpu.kern.*` 才是 GPU 执行时间;两者不可相加当墙时。
3. **图态与 eager 不可混谈**:图回放期宿主钩子**不触发**(节点是驱动侧的),
   `interp.host.*` 只在捕获/trace 期有意义;图态分布走 `gpu.graph.node.*`
   (节点计数)+ 分段探针。
4. **报告必须带 epoch/协议标签**(长度/prompt 域/spec 开关/池配置)。
5. **一次绿不算绿**:性能结论需 ≥3 次同协议复跑,取中位。

## 七、施工分解

| 步 | 内容 | 状态 |
|---|---|---|
| P1 | feature `perf` + `PerfLevel` 分级门 + `off` 基线臂 | ✅ 本日 |
| P2 | 单栈父指针 + exclusive/self + rollup + epoch + CSV/JSON 导出 | ✅ 本日 |
| P3 | 解释器钩子(`interp.host.node.*` / `seg.*` / 计数面)+ 阶段包络 | ✅ 本日 |
| P4 | `/debug/perf[/epoch][/json\|<prefix>]` 树状报告 + `--perf-dump` + 在线分级切换 | ✅ 本日 |
| P5 | `gpu.kern.*`(eager event 对)+ 图材料单入账 + 散装仪表并轨 | 下一步 |
| P6 | 长跑守门:仪表税回归门(`off` vs `host` 同负载差 ≤ 门限) | 待立 |

---

## 八、首次实测(2026-10-14;P1-P4 交付当日)

**协议**:生产 27B AWQ(cyankiwi)/ 单 3090 Ti(`bench-27b-std.toml`:f16 池 9472 tok /
max_seq 9216 / gdn_chunked + FI + prefill_split)/ device=1 / release + `--features perf` /
prompt 24 tok + 128 输出(ignore_eos)/ greedy-decode 档(sampling,非 spec)。
原始数据:`/tmp/owl-perf/{epoch.json, run1.csv, boot1.log}`。

### 8.1 总账(末轮 epoch 视图)

| 量 | 值 | 占比 |
|---|---:|---:|
| 轮墙(`decode.step` ×127) | 3.116 s | 100% |
| `engine.phase.decode` | 3.12 s | ← 127 步 × 24.6 ms |
| `engine.phase.prefill` | 90.6 ms | 2.9%(1 个 chunk) |
| **`interp.host.node` 合计** | **58.1 ms** | **1.9%** |
| ├ prefill 分支(4799 节点) | 57.8 ms | — |
| └ decode 分支 | **0.0 ms(零节点)** | ← 见 8.3 沉默律 |
| `interp.host.seg.*` 合计 | 47.0 ms / 1332 发 | — |

### 8.2 宿主链分相(`Call` 臂 1332 发;prefill chunk)

| 段 | self | 每次 | 占比 |
|---|---:|---:|---:|
| **launch** | 24.29 ms | **18.2 µs** | 51.7% |
| **alloc** | 19.36 ms | **14.5 µs** | 41.2% |
| lower | 1.42 ms | 1.1 µs | 3.0% |
| resolve(选核) | 1.04 ms | 0.8 µs | 2.2% |
| build(构件+守门) | 0.88 ms | 0.7 µs | 1.9% |

⇒ **宿主链 93% 集中在 launch + alloc 两段**;选核/构件/线格式构造三段合计 <7%
(与「专利面已收窄」相符)。这是 quota 首次可量化 —— 后续 prefill 宿主优化
只需盯这两段。

### 8.3 类热点(prefill chunk;每发宿主 µs)

| 类 | n | self | 每次 |
|---|---:|---:|---:|
| `marlin_gemm_w4a16_awq` | 352 | 13.96 ms | **39.7 µs** |
| `owl_narrow_strided_f16` | 209 | 7.29 ms | 34.9 µs |
| `owl_gdn_conv_fwd_f16` | 144 | 5.13 ms | 35.6 µs |
| `htod`(H2D 搬运) | 85 | 4.76 ms | **56.0 µs** |
| `gemm`(cublas) | 97 | 4.19 ms | 43.2 µs |
| `owl_gdn_l2norm_f16` | 96 | 3.26 ms | 34.0 µs |
| `owl_rmsnorm_f16` | 65 | 2.32 ms | 35.7 µs |
| 其余(每发 ~30-35 µs 的固定档) | — | — | — |

### 8.4 两条新方法论(入册)

1. **图回放沉默律**:decode 走捕获图 ⇒ `_eval_node` **不执行** ⇒
   `interp.host.*` 对 decode **结构性为零**(本测 decode 分支节点数 = 0,
   而 prefill 分支 4799)。⇒ 宿主分布只在 (a) prefill chunk、(b) 捕获
   warmup、(c) `no_graph` eager 档 有意义;**decode 的耗时分布必须走
   设备侧口径**(P5)—— 这把 P5 从「增强」升为「decode 分析的唯一入口」。
2. **warmup 污染律**:since-boot 累积视图里首核/装载 outlier(prefill.gemm
   max = 14.5 ms vs p50 = 44.5 µs)会独占榜首 ⇒ **性能判读一律用
   `epoch` 视图**(每 turn `execute_begin` 打标;`/debug/perf/epoch`)。

### 8.5 仪表税(off vs host;在线切换,同 session 六发)

| 级 | 轮墙(3 发) |
|---|---|
| `host` | 3.2308 / 3.2192 / 3.2099 s(中位 3.2192)|
| `off` | 3.2142 / 3.2051 / 3.2099 s(中位 3.2099)|

⇒ **税 ≈ 0.29%** —— 本档 decode 为图态(无 interp 节点),税主要来自阶段包络
与引擎侧启动打点。⚠️ **p.refill/`no_graph` 档的税未测**(那才是钩子全开面),
挂 P6。

### 8.6 首个性能结论(待立案)

- prefill 宿主链每次发射 ~18 µs(launch)+ ~15 µs(alloc)⇒ 1332 发 ≈ 44 ms;
  这是 **host 已经可见的硬成本**,在长 prompt(如 8192 tok)下按 chunk 数线性放大。
- 但当前 24 tok chunk 下宿主只占 1.9% ⇒ **宿主不是本档瓶颈**;优先序仍应是
  设备侧面(GEMM/GDN/attention 的 GPU 时间分布),否则会在无意义的微观优化上花时间。
  ⇒ **先交 P5(GPU 面),再定刀**。

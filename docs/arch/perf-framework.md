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

---

## 九、P5 第一步:注入式下沉口(LogSink)+ 服务端分相(2026-10-14 用户指令)

### 9.1 为什么要"注入"而不是"依赖"

设备面(`crates/backends/cuda`)与算子面(`crates/kernels`)**不能反过来依赖
日志/指标 crate**:kernels 的 **lib 零依赖律**(owl-shared 只在它的
dev-dependencies)是硬约束;cuda 虽有 owl-shared,但让下层自己去找全局
单例也违背「显式依赖律」。

⇒ 做法:**一个零状态可 Copy 的函数指针包**,由上层在**构造函数**注入:

```rust
// owl_shared::log::LogSink —— 一个对象管三件事
log(level, tag, msg)   // 事件流(→ logger)
timer(tag, dur)        // 数值账(→ metrics timer_record_tag)
count(tag, n)          // 计数
```

- `GpuServer::with_diag_sink(rx, selector, boot, diag, debug, sink)`
  —— **GPU server 构造函数持 sink**;`with_diag` 沿用旧签名(注入
  `LogSink::none()` = 全静默);
- 引擎 boot 处一次性组装:`GpuClient::spawn_with(…, owl_shared::log::sink())`
  (`sink()` = 全局 logger + metrics 的现成接线);
- cuda 内所有打印改走 `self.sink.logf(LEVEL, tag, format_args!(…))`
  —— **先准入再 format**(不过门时零分配),`LogSink::none()` 时全静默;
- 顺带解决 kernels 13 处挂账的**同型方案**(将来把 sink 经 `Exec`/registry
  初始化面传进 kernels 即可,不动依赖图)。

### 9.2 服务端分相(发射/回收路径)

| tag | 位置 | 含义 |
|---|---|---|
| `srv.op.<核名>` | `handle_launch` 包络 | 逐算子服务端总时(两路都覆盖) |
| `srv.seg.prep` | `issue_launch` 前段 | 参数槽装配 + 取核 + smem 守卫 + 准入 |
| `srv.seg.issue` | `issue_launch` 后段 | 真正的 `cuLaunchKernel` 提交 |
| `srv.seg.fam` | 家族(foreign)路 | registry.execute(FFI/marshal) |
| `srv.seg.alloc` / `srv.seg.free` | `handle_alloc`/`handle_free` | 池块分配 / 回收 |
| `srv.seg.htod` / `srv.seg.dtoh` | harvest | 搬运两向(`dtoh` 前等 COMPUTE 排空) |
| `srv.count.free_blocks` | `handle_free` | 回收块数 |

### 9.3 首次对账(同 epoch;27B/3090Ti/f16 池;2 请求 × 32 tok)

| 段 | 客户端观测 | 服务端真活 | **管道/调度** |
|---|---:|---:|---:|
| 原生发射 | `interp.host.seg.launch` **22.4 µs** | prep 0.57 + issue 4.7 = **5.3 µs** | **17.1 µs(76%)** |
| 分配 | `interp.host.seg.alloc` **16.1 µs** | `srv.seg.alloc` **4.5 µs** | **11.6 µs(72%)** |
| 家族路发射 | (含在上面) | `srv.seg.fam` **13.4 µs** | ≈9 µs |
| 回收 | — | `srv.seg.free` 4.2 µs | — |
| H2D | `interp.host.node.*.htod` 60 µs | `srv.seg.htod` 15.3 µs | ≈45 µs |
| 逐算子总 | `interp.host.node` 66.6 ms / 4799 节点 | `srv.op` 17.4 ms / 1429 发 | — |

**结论(第一杠杆)**:解释器宿主链的**七成以上是「到 actor 的往返」**,
服务端真干活只有 4-13 µs/命令。`resolve/lower/build` 合计 <3 µs
(选核/构件/线格式**不是**问题)。

**免费代理**:`srv.seg.dtoh` = **21.9 ms/步**(它必须等 COMPUTE 排空)
⇒ 这是**图态 decode 的步 GPU 时代理**,不必等逐核 event 对就能看趋势
(带宽地板 ≈13 ms/步 ⇒ 表面 1.7× 空间)。

### 9.4 管道税拆解(第二轮探针:tx / queue / op / wait 四表)

四只表:`srv.tx.<kind>`(客户端入队)、`srv.queue.<kind>`(入队→服务端出队,
客户端在 `Ack` 上盖 `created` 时间戳、服务端出队时相减)、`srv.op.<核名>`
(服务端处理全程)、`srv.wait.<kind>`(客户端等回执全程)。

**一次 `launch` 命令的完整生命周期(1429 次,同 epoch,单位 µs/次)**

| 段 | 每次 | 归属 |
|---|---:|---|
| `srv.tx.launch` | **1.84** | 客户端:锁 mpsc + 入队 + unpark |
| `srv.queue.launch` | **6.67** | **入向唤醒延迟**(线程 park→run,跨核) |
| `srv.op`(真活) | **10.86** | 服务端 |
| ├ `srv.seg.prep` | 0.48 | 参数装配 + 取核 + smem 守卫 |
| ├ `srv.seg.issue` | **4.05** | 真正的 `cuLaunchKernel` |
| ├ `srv.seg.fam` | 11.52(449 发) | 家族路 FFI/marshal |
| └ **余(准入+分派+回执记账)** | **≈6.3** | 白名单线性扫描等 |
| 回执唤醒 + 客户端 poll | ≈1.44 | ack `waker.wake` → 引擎线程 |
| **= `srv.wait.launch`** | **20.81** | 客户端看到的等回执全程 |
| 客户端 `interp.host.seg.launch` | **23.19** | 再 +2.4 = boxed-future 链 poll 重入 |

其余 kind(同 epoch):`alloc` tx 1.83 / queue 6.74 / seg 3.72 / wait 14.31;
`free` tx 1.84 / queue 6.74 / seg 3.19 / **wait 16.09**(残余 4.3 = 分派 +
回收记账,含每批 event 钉序);`dtoh` wait 22.14 ms vs seg 21.91 ms ⇒
**税仅 ~7 µs,全是等排空**(GPU 边界)。

**结论(推翻了上一轮的猜测)**:
1. **免回执收益很小**(回执侧唤醒仅 ~1.4 µs + await 重入 2.4 µs);
2. **最大单项 = 入向唤醒 6.67 µs/命令**(跨核 park→run);
3. **服务端真活 10.86 µs 里,纯发射只有 4.5 µs,余 ≈6.3 µs 是准入/分派/记账**;
4. 每 `Call` 节点两条命令(alloc+launch)⇒ 税 ≈ **25 µs/节点**,服务端真活
   ≈22 µs/节点(其中"真的发起 GPU 动作"只有 ~9 µs)。

---

## 十、decode 测量(2026-10-14;含图态/eager A/B 与三条方法论修正)

档:`testdata/args/decode-{graph,eager}-27b.toml`(同 prompt 410 tok、同 32 输出、
`--ignore-eos`、bare f16 池、无 spec、单会话;3090 Ti)。

### 10.1 基线(图态,256 tok 全程)

| 量 | 值 |
|---|---:|
| 轮墙(含 TTFT) | 7.85 s |
| `engine.phase.prefill` | 357.6 ms(410 tok ⇒ ~1150 t/s) |
| `engine.phase.decode` | 6.470 s / 255 = **25.37 ms/步** |
| `srv.seg.dtoh` | 23.34 ms/步(**等 COMPUTE 排空 + D2H** = 图态 GPU 时) |
| `srv.wait.graph_launch` | 36.5 µs/步(整步一次提交:tx 2.8 + queue 15.7 + 回放) |
| host 采样/决策 | 25.37 − 23.66 = **~1.7 ms/步(6.7%)** |
| 解码图材料单 | **836 节点**(marlin 352 / farn 64 / rmsnorm 65 / silu 64 / add 64 / gdn×3 144 / attn 32 / 其余) |

⇒ **decode 92% 是 GPU 边界,6.7% 是 host 采样链,0.2% 是提交税。**

### 10.2 图态 vs eager(同 prompt / 同 32 tok / 同 ctx 区间,冷启动已排除)

| | graph | eager |
|---|---:|---:|
| 步时 `decode.step` | **25.18 ms** | **47.61 ms** |
| `srv.seg.dtoh`/步 | **22.91 ms** | **2.06 ms** |
| 命令管线等待/步 | ~0.04 ms | **~27 ms**(616 launch×23.9µs + 616 alloc×20.4µs) |
| 树声明(`declare_outputs`)+ 其余 | ~0(捕获一次) | **~17 ms** |
| host 采样 | ~1.7 ms | ~1 ms |

⇒ **图替我们省 22.4 ms/步(47%)**,省的是「每步重建 SSA 树 ~17ms + 命令管道 ~27ms」,
代价是设备侧节点派发(已含在 22.9 ms 里)。

### 10.3 三条方法论修正(入册)

1. **`srv.seg.dtoh` 只在图态是「GPU 时」代理** —— 图态 22.91 ms/步
   (整步一次性提交后等排空 = 真 GPU 时);eager 态只有 **2.06 ms/步**
   (host 提交慢,GPU 早已空)⇒ **同一指标两态语义相反**,跨态不可比。
2. **样本容量 4096/tag 的静默截断** —— 256 tok 档里 `marlin`/`block`/`view`/
   `interp.host.seg.*` 全部 **count = 4096(顶到 cap)**,`dropped` 才看得出。
   判读纪律:**看 `drop` 字段;per-sample 均值仍有效,总数与会话级计数无效**。
3. **`unk` 相位 = GraphPlan 缺 env** —— eager 臂的节点全落
   `interp.host.node.unk.*`,因为 `GraphPlan::replay → eval_current →
   eval_ops_multi()` **不带 EnvProvider**(`EnvProvider::default().phase = Unk`)。
   修法:把 env 穿进 GraphPlan(小改)。

### 10.4 ✅ 已修:eager 臂 decode 每步泄漏 → ~100 步 OOM(真 bug)

`GraphPlan::replay` 的 `Mode::Eager` 走 [`eval_current`](../crates/engine/src/graph_plan.rs)
→ **`eval_ops_multi`(plain 口径:块常驻,无回收)** ⇒ 每步中间块全留存,
实测 ~100 步后 `CUDA_ERROR_OUT_OF_MEMORY`,且**失败后池不可恢复**(后续请求立刻再 OOM)。
生产走 Captured 所以从未暴露。

**修复(2026-10-14,用户指令「先修 bug」)**:
1. `eval_current` 改 **`eval_ops_multi_scoped_env`**(per-node 活性回收 +
   末次 arena−roots 清扫)⇒ 中间块当轮即还;
2. **上一轮声明输出显式回收**(入口处 `free`)—— 契约:`output_*` 句柄
   仅在**本次回放内**有效,跨回放使用 = 悬空 id → 服务端结构化
   `DeadBlock`(**fail loud**,不会静默错值);捕获臂不受影响(slab 常驻);
3. `plan()` 的 warmup 收尾从 `last.clear()`(只丢句柄)改**显式 free**;
4. **顺带修 `unk` 相位**(同一条链路):`EnvProvider` 贯入 `GraphPlan`,
   并新增 `RunningEngine::set_phase`(引擎 env + 5 类 plan 的 env 快照同步)
   ⇒ eager 臂节点从 `interp.host.node.unk.*` 正名为 `.decode.*`。

**验收**:eager 臂 410 tok prompt + **256 tok × 2 次全通**,`[vram-trace]`
两轮 `free=2869MiB` **零增长**;graph 臂回归 25.25 ms/步(改前 25.18,噪声内);
engine lib **22/22**;engine e2e 全场景 **1 passed(255.5s,含 spec/DFlash2
恒等门)**;clippy 0 警。

---

## 十一、CUPTI 归因面 —— **引擎自己分析自己**(2026-10-14 PoC 落地)

### 11.1 动机:图里发生了什么,两套仪器都看不见

decode / verify 走**捕获图回放** ⇒ ①解释器钩子结构性为零(`interp.host.*` 0 节点)
②`gpu_prof` 与图互斥(逐发射同步,只在 eager 有意义)③外部 nsys 能看,但
**离开引擎**(不能进我们的账/报告/在线开关)。CUPTI(= nsys 的内核)是唯一
能在**生产图态**拿逐核 GPU 时间、且零核改动的路。

### 11.2 形态(注入式,与 sink 同源)

```text
CUPTI 驱动回调线程 ──shim(只 memcpy)──► 旁路缓冲 ──► 排空线程 ──► sink.timer
                                                        gpu.kern.<核名>
                                                        gpu.graph.<gid>.<核名>
                                                        gpu.nodes.<gid>
```

| 件 | 落点 | 说明 |
|---|---|---|
| `native/cupti_shim.c` | `crates/backends/cuda/` | 吃真头文件,**C shim** 把 `CUpti_ActivityKernel11` 压成自包含 plain struct(核名内联 64B,无悬垂指针)⇒ Rust 侧零 ABI 猜测 |
| `build.rs` | 同上 | **feature `cupti` 关 = 零动作**(不编 shim、不引 libcupti);开则 `cc`+`ar` 静态归档 + `-lcupti` + rpath |
| `src/cupti.rs` | 同上 | 排空线程 + tag 构造;**回调只投递**(`try_lock` 拿不到即丢并计数) |
| 武装点 | `GpuServer::run` | `if sink.gpu_armed()` ⇒ 才调 `owl_cupti_start`;`perf level = gpu` 是唯一开关(默认 `host` 不启动) |

### 11.3 PoC 三条验收(全过)

1. **图内逐核**:8k ctx 生产 spec 档两轮 → **45/45 行非零**,`graphId` 归属清晰
   (5=verify / 29=draft / 8·11·14·17·23=fold 桶);
2. **开销可测**:**+5.4%**(`level=host` 6.96s vs `level=gpu` 7.33s;同 prompt/同输出/3 轮中位);
3. **零回归**:默认档 `gpu.*` 行数 = **0**(不调 `owl_cupti_start`);不开 feature 的
   默认构建里 `cupti` 零命中(build.rs 早退)。

### 11.4 ⚠️ CUPTI 语义坑(踩实,必记)

**`CUPTI_ACTIVITY_KIND_KERNEL`(legacy kind 3)对图内节点只报名字、不报时间戳**
(`start=end=0`,实测 11/49 非零);换 **`CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL`
(kind 10,nsys 同款)** 后 **45/45 全非零**。⇒ 采集面一律用 kind 10。

### 11.5 首战结论:verify 30 ms/轮的真身(重新定谳 item 1)

**闭环校验**:图内逐核累加 = **30.3 ms/轮**,与我们自己的 `spec.verify.read`
**31.1 ms** 吻合 ✓(两套仪器互证)。

| verify 图内(8k ctx,归一到 16.2 轮) | ms/轮 | 占比 |
|---|---:|---:|
| **marlin GEMM** | **17.8** | **59%** |
| **`kernel_mha`(XQA;469 µs/层 × 16 层)** | **7.5** | **25%** |
| `owl_gdn_recurrence_varlen_gqa` | 2.0 | 7% |
| farn/rmsnorm/conv/argmax/narrow/embed | ~2.5 | 8% |

⇒ **"8 伪序列单遍化 / fp8 反量化"都不是主项**(XQA 只占 25%);verify 的成本
≈ **一遍权重带宽(GEMM 59% 是常数项,T=8 与 T=1 同价)** + attention。
**真正的 ctx 相关项只有 XQA**:512 档 ~4.4 ms → 8k 档 7.5 ms(+70%),
**绝对头寸 = 3.1 ms/轮**(不是 31 ms)——长 ctx 若要动,动的是 XQA/fp8 读核。

**draft 图(graph 29)**:`owl_naive_attn_nc_bf16_v2` **1127.8 µs/层调用** × 5 层
+ `owl_topk16_f16` **1067.9 µs/次** + `owl_dflash_conv_bf16` 442 µs ⇒ 这是
propose 轮(~11 ms)的大头,与 `spec.propose` 分相自洽。

### 11.6 交付纪律(2026-10-14 补录,D2 战中现场抓)

**⚠️ 排空线程必须周期 flush**。CUPTI 的 activity 记录**不是**实时交付:它们先进
CUPTI 内部缓冲,**只有内部缓冲满、或显式 `cuptiActivityFlushAll`** 才经
`on_buffer_completed` 交给我们(我们自己申请的 2 MiB 缓冲只是"交付容器")。
首版实现只在 `stop()` 时 flush ⇒ **低体量负载样本恒为 0**(实测:几十轮 decode
≈ 几 MB,不满分片 ⇒ `/debug/perf/epoch/gpu` 一行没有,而"武装成功"日志照常打印
—— 静默失效,最坏的一种)。修法 = 排空线程**每 50 ms 先 `owl_cupti_flush()` 再排空**
(专用线程,与驱动线程回调解耦;nsys 同款姿势)。

**开销口径(复测)**:同一负载 `--perf-level host` vs `gpu` 的 `spec.round` =
42.4 vs 43.6 ms/轮 ⇒ **+3%**(旧口径的 +5.4% 是另一种负载形状;两者同量级)。
即:**逐核账可信,但别拿 gpu 档的数当绝对性能**。

**判读纪律补充**:`gpu.graph.<gid>.<核名>` 的 `n` 是**回放次数**,
`self/n` = 单次核时(ngraphId 归属能力见 §11.1)。同一核可能同时出现在
verify / draft / fold 三个图里 —— 看 `gid` 分辨,别看总数。

### 11.7 feature 转发链(2026-10-14 实测坑)

`perf` 是**逐 crate 的 feature**:`server --features perf` 只点亮 `server`/`engine`/
`shared` 的宏,而**解释器钩子住在 `crates/models`** ⇒ 若 `owl-engine/perf` 不转发
`owl-models/perf`,则 `interp.host.node.*` / `interp.host.seg.*` **一行都不会出现**
(`/debug/perf/interp` 空),而 `engine.phase.*` / `srv.*` / `gpu.*` 照常有 —— 
**半哑状态,无任何告警**。修法:`crates/engine/Cargo.toml`
`perf = ["owl-shared/perf", "owl-models/perf"]`(log 同款)。

**纪律**:凡新增跨 crate 的 feature 门,必须①补齐转发链 ②真机跑一次并确认
目标 tag 前缀**非空**(空 = 静默失效,比报错更贵)。

## 十二、单步 / 粘滞调试门(step gate;2026-10-14 立项当日落地)

### 12.1 定位

前三节是**观察**(metrics 数值账 / log 事件流 / CUPTI 设备归因);本节是**干预**:
让解释器在检查点**停下来**,把"看不清楚的执行序"变成"可以一步步走出来的执行序"。
与另两者共用同一套纪律:**宏 + 编译期 feature 门 + 运行期档位 + 全局单例(在线可切)**。

### 12.2 两种模式

| 模式 | 唤醒条件 | 用途 |
|---|---|---|
| `Sticky`(**粘滞**) | **定时器**:每个检查点睡 `step_gate_ms` | 整体放慢,便于盯日志/指标/显存;**不改执行序** |
| `Single`(**单步**) | **外部显式 step 信号**(tx/rx 通道) | 逐检查点手动推进:发一步走一步 |
| `Off` | — | 零干预(缺省) |

### 12.3 检查点(三个粒度)

| 站点 | 位置 | 粒度 |
|---|---|---|
| `interp.node` | `crates/models/src/interpreters/eval.rs::_eval_node` 顶部(CSE 判定**之前**) | **最细**:一步 = 一次节点访问 |
| `engine.prefill` | `crates/engine/src/exec/phases.rs::execute_prefill` | 每 prefill 块 |
| `engine.decode` | 同上 `execute_decode` | 每 decode 步 |

`step_gate_every = N` 把最细那档**粗化**(每 N 个检查点才停一次)⇒ 层粒度/块粒度都能凑出来
(27B 实测:一个短 prompt 的 prefill+32 decode ≈ **6441 个检查点** ⇒ `every=200` ⇒ 32 次等待)。

### 12.4 零开销律(宏)

```rust
owl_shared::step_gate!("interp.node").await;
```

* **不带 `stepgate` feature 的构建**:宏展开为 `core::future::ready(())`
  —— **参数不求值、无分支、无全局查询**(与 `perf`/`log` 的"生产零成本"同款);
* 带 feature:一次检查点(档位判定 + 命中计数 + 可选等待);
* 调试二进制:`cargo build --release -p server --features stepgate`
  (server 的 feature 转发到 `owl-shared/stepgate`;见 §11.7 feature 转发链)。

### 12.5 控制面

```bash
# 启动参数(或配置 [probes] step_gate*)
server -p profiles/legacy/spec-dflash.toml owl.toml --step-gate sticky --step-gate-ms 20 --step-gate-every 200

# 在线(HTTP;与 /debug/perf/level、/debug/log/level 同风格)
curl /debug/step                          # 状态快照
curl /debug/step/mode/single              # 切档(立即生效)
curl /debug/step/release/1                # **发一步**(n=0 或 inf = 不限步)
curl /debug/step/ms/20                    # 粘滞间隔
curl /debug/step/every/200                # 粗化步长
```

状态面字段:`mode / compiled / armed / every / interval_ms / hits / advanced / waits /
waited_ms / credits / released / waiting`。
`compiled=false` = 本二进制未编入 feature(此时配非 off 档会在 boot 时**可见告警**,不静默失效)。

### 12.6 指标面(接 metrics 框架)

| 键 | 语义 |
|---|---|
| `stepgate.hits` | 进入检查点次数(counter) |
| `stepgate.advanced` | **真正通过**门的检查点数(等待结束/吃信用/被 every 过滤都算) |
| `stepgate.waits` | 真等了几次 |
| `stepgate.wait` | 每次等待时延(timer;按 `file!/line!` 分站点) |

查询:`GET /debug/metrics/stepgate`。

### 12.7 纪律(实测踩中,写进契约)

1. **boot 完成后才武装**:boot 期的 warmup/图捕获也跑解释器,而那时 HTTP 面还没起来
   ⇒ 一开始就 Single 会**自锁**(实测:main 卡在 `ready_rx.recv()`,端口不监听,
   GPU 显存已占 ⇒ 表现为"起来了但连不上")。故 `configure` 只在就绪横幅**之前一行**调用,
   状态面用 `armed` 标明;
2. **通道只做唤醒,不携带状态**:载荷式消息会在通道里**排队**,后来的检查点可能吃到
   **早先的**旧档位 ⇒ "翻旧账"(实测:切回 single 后请求跑到第 3 个检查点被更早的
   `Mode(off)` 放跑整轮)。⇒ **状态唯一住址 = 原子量**,被叫醒者一律重读;
3. **通道关闭 = 放行**:宁可少等一步,也不把引擎焊死;
4. **等待必须 async**(`tokio::sync::Mutex` + `mpsc`):阻塞运行时线程会让 HTTP 面
   回不了 step 信号 = 死锁;
5. `release(n)`:**本检查点吃 1 步**,余下 n−1 入信用池;`release(0)` = 不限步
   (⚠️ 不能写 `release(n-1)`:n=1 时 `0` 会被当成"不限步"⇒ 整轮跑飞,实测踩中)。

### 12.8 首测数据(27B/3090Ti/spec-dflash;短 prompt;`every=200`)

| 档 | TTFT | 说明 |
|---|---|---|
| `off` | **0.40s** | 基线(`hits=0`,门连计数都不走) |
| `sticky` 20ms | **0.76s** | host 累计睡 674ms,**墙钟只 +0.36s** ⇒ 等待与 GPU 计算重叠(顺带量出 host 领先量) |
| `single` | 31.4s | 由外部按 3 次 `release/1` + 一次 `release/0` 驱动;`release/1` ⇒ 恰好前进 1 个检查点(200 节点) |
| 切回 `off` | 0.42s | 在线切档立即生效,下一请求零等待 |

### 12.9 变更记录

| 日期 | 版本 | 内容 |
|---|---|---|
| 2026-10-14 | v1.0 | 立项当日落地:`owl_shared::stepgate`(宏 + feature 门 + 两模式 + tx/rx 唤醒)+ 三处检查点(interp.node / engine.prefill / engine.decode)+ 配置/CLI/HTTP 三面控制 + metrics 四键 + 6 单测(含"翻旧账"回归)+ 真机两档验收 |

## 十三、GPU server 发射账(trace ring;2026-10-14 立项当日落地)

### 13.1 定位

前十二节里,CUPTI 答"**设备侧**时间花在哪",`[call]` 解释器账答"**解释器**发了什么";
本节答第三问:**server 侧的事实发射序** —— 它是全部设备动作的唯一咽喉
(`launch` / `htod` / `dtoh` / `alloc` / `memset` / `free` / `copy` / `graph`),
而 `[call]` 账**看不到**搬运/回收/图。

### 13.2 形态

`crates/backends/cuda/src/trace.rs`:定容环形账(`VecDeque` + 单调 `seq`),
在 `GpuServer::dispatch_inner` **派发之前**记一条;`htod` 额外在**落定**时补记一条
`htod_landed`(带块 id + **设备地址** —— 地址复用是"跨流 UAF"类缺陷的唯一钥匙)。

每条账目:`seq / host_ns / cmd / stream(compute|h2d|d2h) / name / grid / block /
smem / bytes / ids / ptrs / out_id`(输出块 id —— 契约:最后一个 Block 实参 = 输出)。

### 13.3 开关与查询

```bash
# 开关(配置面;0 = 关 = 逐命令零开销)
[cuda]
trace_ring = 65536        # 建议 4096~65536(27B 一次 prefill+decode ≈ 3.6 万条)

# 查询(app 侧直读发布出来的账;不占 DeviceClient 契约)
curl /debug/launches                      # 最近 200 条(缺省)
curl /debug/launches/5000                 # 最近 5000 条(按发射序,旧→新)
curl /debug/launches/50/htod_landed       # 只看上传落定(带地址)
curl /debug/launches/50/h2d               # 按流过滤(compute|h2d|d2h)
curl /debug/launches/50/marlin            # 按核名子串过滤
```

### 13.4 首战(27B/spec-dflash/一次请求,`trace_ring=65536`)

* 一次请求 **36481 条**命令,涉及 **10977 个地址**;
* **三流密交织**:decode 每步固定形状 = `htod_chunks(37624B) → graph_launch →
  dtoh(32B) → htod_chunks(4B) → graph_launch → copy_batch(159MB) → htod_chunks(232B)
  → graph_launch → dtoh(7196B)` —— 一屏就能看出"哪些动作之间没有序";
* **地址复用密集**:同一地址在同一请求内落定最多 **15 次**;
* **输入侧潜在冲突 276 起**(某核读 P 之后 `htod_landed` 覆写 P,中间**无 sync**);
  落定尺寸高度集中:**272B ×215 / 32B ×49 / 4B ×12**;
* **272B 那条是"每轮一次"**:间隔恒定 ≈270 条命令 / 4.6ms(一个 decode 轮),
  上下文恒为 `… → htod(272B) → htod_landed → graph_launch → …`。

⇒ **机制候选(待正面验证)**:spec 轮的**图输入槽**由 H2D 上传,紧随其后的
`graph_launch` 在 COMPUTE 上读同一地址,**两条流之间没有栅栏** ⇒ 回放可能读到
半写入槽。与仓内 2026-10-03 "FI 表渐进腐败案"(同型;那次靠"预物化"修了 FI 表,
**图输入槽没修**)以及 `[cuda] launch_sync=true` 能治好 A13 病臂(恒等 H2D/D2H)
的现象**一致**。验证路径:给该上传→回放之间加一道**定向事件栅栏**,看 A13 病臂是否
在 `gdn_chunked` 开着的情况下转健康。

### 13.5 纪律

1. **账是"发射序",不是"执行序"**:同流内发射序 = 执行序(CUDA 保证),跨流之间
   **无序** —— 账里两条相邻条目完全可能并发执行(这正是要看的东西);
2. `host_ns` = host 单调时钟(相对建账),不是 GPU 时间;设备时间轴用 CUPTI(§十一);
3. 记账不许拖垮执行:容量 0 直接返回,锁中毒就地恢复;
4. `htod_chunks`(分块写入路径)目前只记**发射**,未记落定 —— 图输入槽走的正是它,
   需要时补(见 §13.4 的候选机制)。

### 13.6 变更记录

| 日期 | 版本 | 内容 |
|---|---|---|
| 2026-10-14 | v1.0 | `cuda::trace` 环形账(命令级 + htod 落定)+ `[cuda] trace_ring` + `GET /debug/launches[/<n>[/<kind>]]` + 3 单测;首战产出"图输入槽无栅栏"机制候选(§13.4) |

---

## 十四、A13 结案:NULL 流同步拷贝 × 流序异步分配(2026-10-15)

> **一句话**:`DeviceRes::upload`(家族 runtime scratch 上传)用 **NULL 流同步
> `cuMemcpyHtoD`** 写一个由 **`stream.alloc`(流序 `cudaMallocAsync`)** 刚分配的
> 地址 —— 两者之间**没有任何序**,写会落进**尚未落定**的内存并被静默丢弃。
> chunked GDN 的 **meta 表(cu/coff/idx)** 走的正是这条路 ⇒ 分块索引/偏移读成
> 上一次占用者的残值 ⇒ prefill 末行 hidden 被确定性算错 ⇒ pi 场景"不发工具"。

### 14.1 病灶(compute-sanitizer 定谳)

```
========= Use-before-alloc on allocation of size 1,024 bytes at 0x31a6000600
=========     Address 0x31a6000600 is potentially accessed before it is allocated
=========   访问点: cuMemcpyHtoD_v2 → GpuRes as DeviceRes::upload_block → marlin oracle
=========   分配点: cuMemAllocAsync → CudarcStream::alloc → GpuRes as DeviceRes::alloc_block
```

生产面(`crates/kernels/src/server/gdn_chunked.rs:422`):

```rust
let nb = res.alloc(host.len() * 8, "gdn.meta")?;   // 流序异步分配(COMPUTE)
res.upload(nb.ptr, &bytes)?;                        // NULL 流同步拷贝 ⇒ 无序
```

### 14.2 定谳实验(同 boot / 同 prompt / 只切一位)

新增 `[cuda] sync_mask`(同步点分解掩码,见 [`sync_bit`])后逐点 A/B
(27B AWQ,chunked GDN 臂,ptok=1403,判据 = `[pf-probe]` 末行 logits):

| mask | 同步点 | 末行 logits | 行为 |
|---|---|---|---|
| 0 | 无(旧缺省) | `28` / gap 0.914 | ❌ 塌陷 |
| 1 | HTOD(H2D DMA 后 sync) | `28` / 0.914 | ❌ 逐位无变化 |
| 4 | FENCE(回收栅栏旧语义) | `28` / 0.914 | ❌ 无变化 |
| 16 | GRAPH(回放后 sync) | `28` / 0.914 | ❌ 无变化 |
| 2 | DTOH(整流 `COMPUTE.sync`) | `198` / 0.453 | ⚠️ 第三种错值(半程定序) |
| 8 | LAUNCH(逐发射 sync) | `760` / 2.5625 | ⚠️ argmax 对但残差 |
| 32 | ALLOC(alloc 后 sync) | `760` / 2.5625 | ⚠️ 同上 |
| 63 | 全部(`launch_sync`) | `760` / 2.5625 | ⚠️ 同上 |
| **64** | **scratch 上传前排空 COMPUTE** | **`760` / 4.328** | ✅ 与 varlen 臂一致 |
| 128 | 上传退回 NULL 流同步(复现位) | `28` / 0.914 | ❌ **A13 逐字节复现** |

**读法**:① 拷贝流(HTOD/FENCE)与本病历**无关** ⇒ "跨流 UAF"假设彻底出局;
② 位 8/32 这两个互不相干的点各自"治好",说明治愈机制是**把 host 摁住让异步
分配落定**(同族位 2 只摁住 71 次 dtoh ⇒ 得到第三种错值,失败是分级的);
③ 位 8/32/63 的 gap 2.5625 ≠ 真值 4.328 ⇒ **旧"解药" `launch_sync` 其实是
治了一半**(argmax 碰对),单看 argmax 会把运气当因果。

### 14.3 发射账侧证据(为什么先前抓不到)

- 病灶路径**不是 `Command`**(家族 runtime 直连 `DeviceRes`),所以发射账/htod
  栅栏全看不见它;只表现为"块 id 计数比命令数多 14"这类旁证;
- 两份 trace(mask=0 vs 8)对拍:窗口内**命令序列逐项同构**(cmd/核名/几何/字节
  全同),唯一差异是**设备地址**(窗口地址集合交集仅 508/4779)⇒ 当时的
  "布局敏感"观感成立,但**不是布局在起作用,而是布局决定了那笔写落在
  "已映射"还是"未落定"的内存上**;
- 同一 mask 不同的两次 boot 布局也只 50% 重合而**值逐字节相同** ⇒ "布局是原因"
  当场被证伪,才转向直接仪器(memcheck)。

### 14.4 修法与代价

`DeviceRes::upload` / `upload_block` 改为**与分配同流的 async 拷贝**
(`memcpy_htod_async(…, COMPUTE)`:分配 → 拷贝 → 后续核 三者同流严格有序;
捕获分支本来就是这么写的);`download`(NULL 流同步 D2H)补**回读前排空
COMPUTE**(仓内 2026-10-09 / xqa.rs:69 / gdn_chunked.rs:453 **三次本地绕行**
的病根即此)。`SCRATCH_UP_LEGACY`(bit 7)保留旧行为以便复现。

| 口径 | 旧(病) | 新 |
|---|---|---|
| TTFT @ptok1403 | 1.18s | **1.19s**(零代价) |
| 末行 logits | 28 / 0.914 | 760 / 4.328 |
| finish / 工具 | `length`,无 tool_calls | **`tool_calls`,`{"command":"echo owl-agent"}`** |
| 端点(varlen 参照) | 1.34s / 1.80s | 1.65s(chunked 提速仍在) |

广泛验收:ptok 543/753/1403/1753/2453 各一次 + 1403 重复一次 = **6/6 全绿**,
重复请求 gap 逐位相同(4.32812)⇒ 确定性恢复。

### 14.5 门与探针(新增/沿用)

| 门/探针 | 用途 |
|---|---|
| `[cuda] sync_mask` + `sync_bit::*` | 同步点**逐位**归因(`launch_sync` 是合称,不可再当单变量用) |
| `sync_bit::SCRATCH_UP_LEGACY` | **A13 复现开关**(仪器验证/回归对照) |
| `cuda.trace_ring` + `alloc_done`/`free` 地址 | 生命周期判据(本轮补上 alloc 落定 + memset/free 的地址) |
| step gate(§十二)/ CUPTI(§十一) | host 领先深度与设备时间轴 |

### 14.6 变更记录

| 日期 | 版本 | 内容 |
|---|---|---|
| 2026-10-15 | v1.0 | A13 结案:根因 NULL 流同步拷贝 × 流序异步分配;修 upload/upload_block/download;新增 `sync_mask` 分解面与 `SCRATCH_UP_LEGACY` 复现位;发射账补 `alloc_done`/地址 |

---

## 十五、家族面收口(临时补救;2026-10-15,A13 暴露的结构性缺口)

> A13 的病根能藏一整轮,根因不是"没人看",而是**家族面不在任何仪器的视场内**:
> 家族 runtime 直连 `DeviceRes` 的设备动作**不产生 `Command`**,而发射账(§十三)
> 只在命令派发处记一条。本轮把"再犯"与"看不见"两件事一起收掉(治理层欠账另记)。

### 15.1 收口审计(改前)

| # | 面 | 谁能到 | 经 `Command`? | 进发射账? | 定序纪律 | 捕获护栏 |
|---|---|---|---|---|---|---|
| 1 | 引擎→server | `GpuClient`(唯一客户端) | ✅ | ✅ | ✅ 三流路由 + dtoh 事件栅栏 | ✅ G2 结构拒绝 |
| 2 | 家族核发射 | `Exec::launch` → `launch_vals` | ❌ | ❌ | ✅ 同 COMPUTE 流 | ⚠️ 查了 `capturing()` 但 `record_capture` 是**空实现** |
| 3 | 家族直连 FFI | marlin host launcher / cublas / FI plan+run | ❌ | ❌ | ✅ 用 `res.stream()` 的 cu_stream | ❌ 连 `capturing()` 都不查 |
| 4 | 家族 scratch | `GpuRes::alloc` → `alloc_zeros`(append-only) | ❌ | ❌ | ✅ 同流 | ❌ 靠"族 grave 律"人工约定 |
| 5 | 家族上传/回读 | `GpuRes::{upload,upload_block,download}` | ❌ | ❌ | ✅ **§十四 刚补** | ❌ |

### 15.2 临时补救三件套(已落)

**① 可观测:`DeviceRes::note_device`**（`crates/kernels/src/device.rs`）

```rust
/// 家族面设备动作入账钩子 —— 通知(best-effort),缺省空实现(关账零成本)
fn note_device(&mut self, _kind: &'static str, _name: &str, _bytes: usize, _ptrs: &[u64]) {}
```
- `GpuRes` 覆写 → 写进**同一个**发射环(`crate::trace`,§十三),`stream = "family"`;
- 接线面:`Exec::launch`(覆盖 gdn_chunked / gdn_scalar / xqa / FI 的**全部核发射**,
  值表里的 `Ptr` 参数顺带落账)+ `GpuRes::{alloc,upload,upload_block,download}`
  + 三个 **FFI 直发族**各自自报一条(marlin 两臂 / cublas / FI run);
- 词表(有界):`fam_alloc` / `fam_upload` / `fam_upload_block` / `fam_download` /
  `fam_launch` / `fam_ffi_launch`。
- ⚠️ `Exec` 在 `kernels` crate,不能依赖 `owl-cuda` 的 trace 模块 ⇒ 必须走
  `DeviceRes` 注入面(不能直接调),这是本钩子存在的形式理由。

**② 文档纠错**(孕育 A13 的错误前提)

`DeviceRes::upload` 原文"同步语义,回执即落地"→ 改为"**同流有序;回执 ≠ host 侧落地**",
并把 memcheck 判据(`Use-before-alloc @ cuMemcpyHtoD ← cuMemAllocAsync`)挂在注释上;
`download` 补"实现侧在回读前排空生产流"的序语义。

**③ 原语收口 + 机器判据门**

- `crates/backends/cuda/src/ffi.rs`:`memcpy_htod_sync` / `memcpy_dtoh_sync`
  (raw driver = **NULL 流**)**降级 `pub(crate)`** —— 不对 crate 外开这条后门;
- 新门 `crates/backends/cuda/tests/raw_sync_primitives_gate.rs`:
  ① 全仓 `crates/**/*.rs` 扫代码面,白名单外命中 = **红**(注释里提名字允许 —— A13 的病史注释不该被自己的门绊倒);
  ② `ffi.rs` 的再导出必须 `pub(crate)`;
  ③ 白名单项必须**真实命中**(防白名单自己腐坏)。

### 15.3 验收

| 判据 | 结果 |
|---|---|
| 家族面入账(一次 ptok=1403 请求,ring=65536) | `fam_launch` 1826 + `fam_ffi_launch` 2172(marlin AWQ 967 / BF16 705 / cublas f16 260 / bf16 196 / FI fp8 44)+ `fam_alloc`/`fam_upload` |
| 收口门 | `raw_sync_primitives_gate` ✅(命中 5 处,全部白名单内) |
| 单测 | owl-shared 36/36 · owl-cuda 5/5 · owl-kernels 53/53 |
| 回归(6 请求 × ptok 543…2453) | 6/6 `finish=tool_calls` + 参数正确 + logits 逐位同修后值 |
| **零代价** | TTFT 与补救前**逐位相同**(0.54/0.63/1.14/1.43/1.99s) |
| 病态复现位 | `sync_mask=128` 仍逐字节复现(`28`/0.91406) |

### 15.4 未收口(立项级,不是临时补救能覆盖的)

1. **`GpuRes::record_capture` 是空实现**(自注"A1.6 数据结构化随 graph 治理层立项(挂账)")
   ⇒ `Exec::launch` 捕获窗内**登记了但没地方落**;
2. **两条族路在捕获窗不对称**:`Exec` 路"查 `capturing()` + 登记(空)",marlin/cublas/FI
   **什么都不查**(进捕获窗全靠驱动原生 capture 兜);
3. **G2 护栏只管命令面**(捕获期只放行 Launch/Alloc)—— 家族面没有对应护栏,
   捕获窗内 `res.alloc`/`download` 这类非法动作只能靠"族 grave 律"人工约定拦住。

### 15.5 变更记录

| 日期 | 版本 | 内容 |
|---|---|---|
| 2026-10-15 | v1.0 | 家族面收口三件套:`DeviceRes::note_device`(家族面入账)+ 文档纠错 + NULL 流原语 `pub(crate)` 与机器判据门;审计表与治理层欠账入档 |

---

## 十六、声明驱动路由(2026-10-15):`KernelSpec::FAMILY` + 拒启路径修复

> 起因:A13 让人看清**家族面不在任何仪器视场内**(§十五);进一步追问发现更基础的
> 一层问题 —— **"某个算子归哪个族执行"这件事在算子侧不可见**,族成员关系只能
> **反向**声明(`KernelFamily::ops()` 手写表),于是路由表、元数据表、族表三份
> 靠 boot 事后校验勉强绑住。

### 16.1 改前结构(三份表 + 事后校验)

| 处 | 内容 | 被谁校验 |
|---|---|---|
| `ops/<域>.rs` `PARAMS` | 槽序/类型/律 | `check_from_params`(L3)+ 池写集审计 |
| `KernelFamily::ops()` | **手写** `const OPS: &[...]`(6 族各抄一份) | boot:"ops ⊆ names"(事后) |
| `OpRegistry::names` | 名字集合 → `knows()`/`execute()` **线性扫族×名** | boot:"家族名全库唯一" |

代价:① 归属在声明处不可见;② `names() = 族主名 ∪ ops()` 的"部分建表"陷阱
(2026-10-13 实测:FI f16 主名 + gdn_scalar 静默从路由表消失);③ 每次发射线性扫;
④ 族 id 在 `server/<族>::id()` 与 `ops/<域>::FAMILY` **两处提及**,可各自改。

### 16.2 改后:`FAMILY` 声明 + 派生 + 索引 + 硬门

```rust
// spec.rs —— 声明面新增一维
pub trait KernelSpec: 'static {
    /// 执行家族(NATIVE = 原生 AOT 直发)
    const FAMILY: OpId = OpId(crate::contract::names::NATIVE);
    ...
}
// ops/marlin.rs —— 声明即归属
impl KernelSpec for MarlinGemmW4a16Awq { const FAMILY: OpId = family_id::MARLIN; ... }
```

**族身份为什么住 `contract::family_id`(零门总账)而不是族结构体**:族 id 要被
**两侧**引用 —— 声明面 `ops/<域>.rs`(**零 feature 门**,models 以 `features=["marlin"]`
依赖本 crate 就要拿元数据)与执行面 `server/<族>.rs`(`device` 门)。**零门面引用有门面
= 编译断**(同一约束见 `ops/mod.rs` 头注);而 `client/{cublas,marlin,xqa,flashinfer}.rs`
与 `family/{cublas,marlin,flashinfer}.rs` 也带 feature 门 ⇒ 本节是唯一对六个族都合法
的零门位置。**终局**:各族 face 零门化后,本节整体下移到 `client/<族>.rs`(那才是
`client/mod.rs` 说的"face = 名字常量")。

配套三件事:
1. `KernelFamily::ops()` **降级为派生**:缺省 = `ops::of_family(self.id())` —— 6 族的
   手写 `const OPS` 全删,单一真源 = `KernelSpec::FAMILY`;
2. `OpRegistry` 建 **name → 族下标** 索引(boot 一次)⇒ `execute`/`knows` 从
   "线性扫族×名"变 O(1);
3. 两条**声明驱动硬门**(取代事后校验)+ 一条可见性告警:
   - **A 孤儿声明**:非 Native 的算子,其 `FAMILY` 必须真实存在(否则元数据发不出去);
   - **B 声明与路由分家**:路由表里的名字若已建表,其 `FAMILY` 必须指向本族;
   - **C 空 ops()**:告警(只能靠主名路由)。

### 16.3 验收

| 判据 | 结果 |
|---|---|
| 路由面零漂移 | `已声明 129 张(原生表 + 家族臂 **11**)` —— 与改前逐字同 |
| 单测 | owl-kernels 53/53 · owl-cuda 5/5 · owl-shared 36/36 · NULL 流门 1/1 |
| 真机(6 请求 × ptok 543…2453) | 6/6 `tool_calls` + logits/TTFT **与改前逐位相同** |
| **门会红(负例)** | `ops/gdn_scalar.rs` 的 `FAMILY` 改成不存在的族 ⇒ 拒启,错误**指名道姓**:`孤儿声明(家族不存在):["gdn_scalar_delta_rule_fwd"]` |

### 16.4 顺手挖出并修掉:拒启路径三缺陷(运维级)

负例验收时暴露 —— 拒启触发后**进程永挂、无监听、`warn` 档下零输出**:

| # | 缺陷 | 位置 | 后果 |
|---|---|---|---|
| 1 | **boot 握手过早**:`boot.send(Ok(()))` 放在"设备 ctx 上线"处,而注册表 P4 门在**其后** | `server/mod.rs::run` | 拒启被"已 Ok 过"掩盖 ⇒ `spawn_with` 返回成功 ⇒ 引擎继续启动 ⇒ 向已退出的 server 发命令 ⇒ 引擎线程卡住 ⇒ `ready` 永不发 ⇒ **main 永挂** |
| 2 | **可失败返回值被丢弃**:`let _ = server.run();` | `gpu_client.rs::spawn_with` | 真 Err 静默(AGENTS 纪律点名:"不允许丢弃一个可失败的函数的返回值") |
| 3 | 拒启消息 `LVL_INFO` | `server/mod.rs` | 生产 `warn` 档**看不见**(而同文件注释写着"拒启必须可见") |

修法:① 握手移到 **boot 全链成功之后**,失败路径**显式上握手送真 Err**;② 改
`if let Err(e) = server.run() { eprintln!("[gpu-client] server 线程退出(Err): {e}") }`
(裸 eprintln,不经级别门);③ 消息升 **`LVL_ERROR`**。

**负例验收(warn 档)**:`exit=1`(不再挂)且三处错误全链可见:
```
[ERROR] [kernels] ❌ 算子注册表装配失败(拒启): ... 孤儿声明(家族不存在):["gdn_scalar_delta_rule_fwd"]
[boot] 引擎装配失败: engine: gpu 绑定失败 "算子注册表装配失败(拒启): ..."
[gpu-client] server 线程退出(Err): 算子注册表装配失败(拒启): ...
```
> 这类"活着但没引擎"的假进程,正是本会话里反复把人误导成"卡死/OOM"的现象源。

### 16.5 未做(下一轮)

`Call`/`parse` **仍是第二份槽序语义**:`client/<族>::parse` 手写下标,只断言
**个数**(`blocks.len() != 7`)不断言序与类型 —— `PARAMS` 里两个同类型槽对调,
`check_from_params` 照放行、`parse` 却会把 state 当 gate 读(静默错值)。
修法 = `parse` 由 `PARAMS` **按名取槽**(`const fn slot_index(name)` + `slots.block("q")?`),
代价 ≈ 6 族改写;兜底低成本版 = boot 期对每个已建表算子**干跑 parse** 并断言与
`PARAMS` 派生序列一致。

### 16.6 变更记录

| 日期 | 版本 | 内容 |
|---|---|---|
| 2026-10-15 | v1.0 | `KernelSpec::FAMILY` + `contract::family_id` 族身份总账 + `ops()` 派生 + name→族 O(1) 索引 + 三条声明驱动门;修拒启路径三缺陷(握手点/错误丢弃/级别),负例验收 |

---

## 十七、server 参数校验审计与补口 + 族身份字面量住址重构(2026-10-15)

### 17.1 审计:校验只在 `Launch` 一路,六条命令裸奔

| 命令 | 校验 | 缺口 |
|---|---|---|
| `Launch` | ✅ L3 全量:槽数对位 / 槽类型对位 / 张量非零长+对齐 / 标量 set·min·max;**白名单逐发射强制** | ① `registry.meta(name)==None` ⇒ **直接 `Ok(())` 跳过**(未建表算子零校验;白名单在场时被白名单挡住,dev 档无);② 块不在账本 ⇒ `arg_len=None` ⇒ 零长/对齐**静默跳过**(非法块仍会在 resolve 期报错,只是晚一步);③ **几何(grid/block/smem/out_elems)默认不校验**(`Geom` 只给自定义 check 用) |
| `Dtoh` | ✅ `want_bytes > n` 拒(前缀读有意放行) | — |
| `GraphLaunch` | ✅ 图存在性 | — |
| `Alloc` / `Htod` | 无 | 无上限/预算校验(引擎侧负责) |
| **`HtodChunk(s)` / `UploadPinned` / `MemsetZero` / `CopyBlock` / `CopyBatch`** | **无** | **`offset + len ≤ block_len` 全无校验** ⇒ 越界写直接落**邻块** |

**为什么要紧**:设备内**无越界保护**,`ptr + offset` 超尾就写邻块;而池复用使邻块身份随布局变
⇒ 表现成"布局敏感的确定性错值" —— **A13 期间"邻块 clobber"正是头号嫌疑**,最后靠
compute-sanitizer 才排除。与其每次靠仪器挖,不如在唯一咽喉把它变成结构化拒绝。

### 17.2 补口(六条命令)+ 验活

- `GpuCtx::check_range(id, off, len, what)`(块在册 + `range_fits`)+ 纯函数
  `range_fits(n, off, len)`(溢出按越界;`off+len == n` 合法)+ 单测 8 例边界;
- 接线:`HtodChunk` / `HtodChunks`(逐写)/ `UploadPinned` / `MemsetZero` /
  `CopyBlock`(源+目标)/ `CopyBatch`(逐 copy 源+目标)。
- **零假阳**:6 请求 × ptok 543…2453 全绿,logits/TTFT 逐位同,越界拒绝计数 **0**;
- **门会红(off-by-one 注入)**:`n-1` 注入后第一个权重上传即爆出可执行错误 ——
  ```
  [boot] 引擎装配失败: Weight 'model.language_model.embed_tokens.weight':
    [upload-pinned-arg] id=1099511627787 越界:offset=2415919104 + len=126877696 > block_len=2542796799
  ```
  且 `2415919104+126877696 = 2542796800`(真实块长)⇒ **合法路径正好写满**,
  `≤` 语义与边界同时被证明。

### 17.3 槽序回归锁(为"parse 仍是第二份语义"补的机器判据)

族侧 `client/<族>::parse` 与 `ins_args`/`scalar_args` 都**手写下标**,`PARAMS` 是另一份;
同类型槽对调时 L3 只看类型 ⇒ 放行,而解析器把 `state` 当 `gate` 读 = 静默错值。
新增 `ops::tests::slot_order_lock_family_ops`:把 **11 个族算子**的 `PARAMS` 名序列钉死
(键引族身份常量),并带**覆盖率门**(族算子一个不漏)。**同类型对调验红**:
注入 `q↔v` 对调(同类型同 rank ⇒ L3 放行)⇒ 锁立刻红,消息指名"请同步审 parse 与 builder"。
配套 `KernelMeta::{param_names, slot_index}`(按名寻址 = 未来 parse/builder 按名取槽的前置件)。

> **未做(下一轮)**:只改 `parse` 会造成**新的不对称**(builder 仍手写序)⇒ 正解是
> **builder + parse 成对按名取址**(6 族 + models 调用点,≈半天);现在先有锁兜着。

### 17.4 族身份字面量住址重构(用户裁决:字面量住各族自己)

**删掉两个总表**:`contract::names`(12 条字面量)与 `contract::family_id`(7 条族身份)。
**新住址**:

| 面 | 住址 |
|---|---|
| 族名/算子名字面量 | **各族自己的 `family/<族>.rs`**(身份面**零门**) |
| 族身份 `OpId` | **族 runtime 结构体上**:`server/<族>::ID`(`id()` 返回 `Self::ID`) |
| 声明面 | `ops/<域>::ENTRY` 与 `::FAMILY` 都引用**族字面量** |
| "无族"标记 | `spec::NATIVE`(它不是族字面量,是"无族"哨兵) |

**门控处理**(零门面引用有门面 = 编译断):三个带 feature 门的族文件把**身份面零门化** ——
`family/cublas.rs`(cudarc 项 `#[cfg(feature="cublas")]`)、`family/marlin/mod.rs`
(`repack` 随 `feature="marlin"`)、`family/flashinfer.rs`(本就零依赖);
`family/mod.rs` 三个模块声明去门。**`cargo check -p owl-kernels`(零 feature)通过** = 零门律保住。

**名字门补齐**:`crates/backends/cuda/tests/server_name_lint.rs`(server 子树禁裸字面量)
此前**漏了 XQA 两名**(覆盖缺口),已补;文档改成新住址纪律(清单 = 住址表两侧互证)。
另:`ops::tests::slot_order_lock` 的键改引族常量(ops 面不再躺族名字面量);
`models/{ops,env,kernel}.rs`、`kernels/driver.rs`、`ops/oracles/*` 全部改引族常量。

**验收**:owl-kernels 54/54 · owl-cuda 6/6 + 名字门 ✅ + NULL 流门 ✅ · owl-shared 36/36 ·
零门编译 ✅ · 真机 6/6 `tool_calls` + logits/TTFT **逐位同改前**。

### 17.5 变更记录

| 日期 | 版本 | 内容 |
|---|---|---|
| 2026-10-15 | v1.0 | 参数校验审计(Launch-only 三盲点)+ 六条命令边界校验补口(off-by-one 验活)+ 槽序回归锁(同类型对调验红)+ 族身份字面量住址重构(删两总表,字面量归各族,身份面零门化,名字门补 XQA) |

---

## 十八、双路独立审查整改(2026-10-15;fork 子代理 + 父会话复核)

两路 **fork 子代理**(继承本轮全部调查史)分别审 `crates/kernels/**` 与 CUDA 后端
`server/**`;报告存 `roadmap.local/26-10-15-审查-{kernels,server}.md`。
**父会话逐条复核**后落地"简单项",复杂项留新会话(见 §18.4)。

### 18.1 本轮落地(9 项)

| # | 修什么 | 位置 | 判据/证据 |
|---|---|---|---|
| 1 | **notify 失败路径统一**(3 处 `ack.take().unwrap()` **必 panic** + 1 处 `let _ = e` 让调用方**永挂**):ack 住 `Arc<Mutex<Option<Ack>>>` cell(与闭包二选一原子成立)+ `notify_failed()` 留 counter/LVL_ERROR | `server/harvest.rs` 四处 | 编译 + 真机验收(`srv.notify_fail` 计数 0) |
| 2 | **家族面写路径边界**:`upload` 命中 scratch 账本只证明"基址在册"⇒ 补 `range_fits(buf_len,0,src.len())`;`upload_block` 补 `check_range` | `server/mod.rs` | 与命令面六条同口径;零假阳 |
| 3 | **`download` EagerOnly 守卫**(捕获窗内 `cudaStreamSynchronize` 非法)+ GDN diag 块 `!capturing` | `server/mod.rs`、`kernels/server/gdn_chunked.rs` | **独立双路径命中**:审查者静态指出 + 父会话做 prefill 图设计时独立推出 |
| 4 | **`Arg::BlockSlice.byte_offset ≤ block_len`** 粗错门 | `server/launch.rs` | 越界访存落邻块(设备内无保护) |
| 5 | **`note_device` 守门 + 入账挪到发射成功后**:`note_enabled()`(缺省 false,GpuRes 覆写)⇒ 关账不建载荷 | `kernels/device.rs`、`server/mod.rs` | 关账零分配;账与事实一致 |
| 6 | **GDN 两族 `slot` 跨参校验**(host 标量乘进状态块指针,唯一未校验处):`check()` 覆写 + `checked_mul` + 账长比较 | `ops/gdn_{chunked,scalar}.rs` | **正例 2 + 负例 2** 单测(越界拒 / 未知块拒 / 边界格放行) |
| 7 | **门补强三处**:① `no_runtime_compile` **删白名单短路**(一行同时含 `Ptx::from_binary` 与禁词即被豁免 = 一行级绕过)+ ROOTS 自检(顶层带 Cargo.toml 的目录必须在扫描面);② `slot_order_lock` **扩 dtype/rank 类型签名锁**;③ 名字门**补正例半边**(清单改从 `ops::all()` **派生**,不再手写) | 三处门 | 各门自测 + "混合行必报"反证 + 名字门 2/2 |
| 8 | **清理**:`src/server.rs.bak`(2000 行死文件)→ `docs/archive/`;`cu_sig.rs` 误导 FIXME(删它会断掉表↔签名互证门);`crate::list` 死引 ×5;`cmd_name` 与 `Command::kind()` **合一**;`carve_block` 区间防溢出(`range_fits`);`gpu_client::htod` **dtype 交叉校验**(不再 `let _ = dtype`);注释漂移(捕获护栏清单/名字住址/gdn parse 边界);fmt nit | — | 编译 + 门 |
| 9 | **删死字段** `DiagOpts::legacy_sync`(复核结论见 §18.2) | `state.rs`、`harvest.rs` | 编译 + 单测 |

### 18.2 复核纠正的一条审查结论(不成立)

审查者 P1「`sync_mask` 位耦合:单开 bit1(DTOH) 会连带把 free 栅栏切成旧语义 ⇒
A13 分解表 `mask=2 → 198` 那格是两个变量」—— **实测不成立**:`fence_targets` 早已改成
`sync_at(FENCE)`、dtoh 走 `sync_at(DTOH)`,`legacy_sync` **只剩派生式没有读者**
(审查者读到的是 `state.rs:876` 的**注释**,不是代码)。
⇒ **§十四 的分解表仍是单变量,A13 结论无需更正**;残留只是"死字段 + 两处过时注释",
本轮已一并删净。

### 18.3 验收

| 判据 | 结果 |
|---|---|
| 单测/门 | owl-kernels **57/57**(+3 slot 门)· owl-cuda **6/6** · owl-shared **36/36** · `no_runtime_compile` **2/2** · `server_name_lint` **2/2** · `raw_sync_primitives_gate` **1/1** |
| 真机(6 请求 × ptok 543…2453) | **6/6 `tool_calls`** + logits/TTFT **与整改前逐位相同** |
| 新门零假阳 | 越界拒绝 **0**、`srv.notify_fail` **0**、boot 门全过(家族臂 11 / 自检 5) |

### 18.4 留给新会话的复杂项(明确不做)

| 项 | 为什么复杂 |
|---|---|
| **L3 补 dtype/rank/role 比对**(`check_from_params` 只比 token 类) | 改 L3 语义 ⇒ 所有已建表算子的发射都要回归;是"声明驱动"的最后一块 |
| **`DeviceRes::resolve_len`**(家族无法自证入参尺寸:marlin `ws`/`c_tmp`、xqa `scratch` 256MiB) | 动 trait 实现面(2 个 impl)+ 三族调用点 |
| **`parse` 与 builder 成对按名取址** | 6 族 + models 调用点;只改一侧会造**新的**不对称(双锁已就位兜底) |
| `prefill` 上 CUDA Graph | 见 §17 之外的独立设计讨论(slab 预算 / M0 去 from_host 是前置) |

## 十九、L3 事实面(dtype/rank/角色)+ 声明面审计(2026-10-15;台账 **A20**)

§18.4 的第一项("声明驱动最后一块")当日落地。**用户裁决**:用 rank(不是 shape);
**能有多严格就多严格,不搞静默丢弃**;不一致**逐条按性质分**(声明错改声明 / 调用点
扇平改调用点 / 阶数无关写 0);fp8 池句柄的 dtype **修引擎折叠**而不是改声明。

### 19.1 架构(用户裁决:**事实随线格式走,服务端唯一门**)

初版把检查放在客户端(`models::ops::check_call_facts`,靠闭包探针读 SSA 图的事实)。
**用户当场纠正**:客户端自查 = 运动员兼裁判,覆盖不到 oracle/测试/未来新路 ——
"你客户端那些参数为什么不发过去呢?"。改后:

| 面 | 变化 |
|---|---|
| **线格式** | `Arg::Block { id, **dtype, rank** }` / `Arg::BlockSlice { …, dtype, rank }`;`Bytes` 变**自描述**(`{ id, len, dtype, rank }`,`alloc` 后 rank 0 = 未定形,`view()` 定形) |
| **服务端 L3** | 事实检查**并入 `check_from_params`**(与 token/账长/几何/角色同一函数)⇒ 唯一咽喉 `admission_check` 覆盖所有发射路 |
| **客户端** | **不再自查**(删 `check_call_facts` 与 4 个调用点);只负责**如实填事实** |
| 顺带 | `DeviceRes::resolve(&Bytes)` → **`resolve(id: u64)`**(原签名逼出满地 `Bytes{id,len:0}` 占位块) |

语义(用户裁决:能多严就多严,不静默):
- 张量槽 `facts()` 缺失 ⇒ 拒;`dtype` 精确相等;
- `rank`:声明 `0` = **显式阶数无关**(核按计数扁平索引);发送方 `rank = 0`
  (**未定形**)在张量槽 = **拒**(不许用"未定形"蒙混);
- `Out` 槽必须是全块 `Block`。

### 19.2 槽序单源

`models::ops::lower_kernel` 的槽序规则(输出槽/父张量槽/标量槽)走
`spec::slot_sources` ⇒ 装配序与检查序**单源**(原 `is_out` 闭包就地推导,必漂)。
事实取自**父节点视图**(`p.dtype` / `p.shape.len()`),不是块句柄的标注 ——
`Reshape`/`SliceView` 透传底块时句柄标注会滞后(实测踩到:delta_dec 收 rank2),
故两臂同时按本节点视图 `view()` 重定形。

### 19.3 服务端补强 4 条(同一张表)

| # | 补什么 | 旧行为 |
|---|---|---|
| 1 | **未知块 ⇒ 拒**(`arg_len == None`) | 静默跳过零长/对齐检查 |
| 2 | **几何通用律**:`out_elems ≥ 1`;`grid.x ≥ 1`(全零 `(0,0,0)` = 显式"runtime 自持"哨兵,家族路专用) | `ctx.geom` **从不被读** ⇒ 任意几何都过 |
| 3 | `Out` 槽禁 `BlockSlice`(线格式完整性) | 无 |
| 4 | 删 `t_out` 的**假 range**(张量槽 range 从不被求值) | 假声明 |

### 19.4 审计方法(影子扫描;**已删**,不入库)

真机跑不动全量算子,故先用**临时全槽扫描**(`OWL_L3_AUDIT=1`:逐槽登记不拒发,
去重打印)跑真机取**全表**,逐条裁决后再转严格。**审计代码已删**(生产路径无影子档)。

- 起点:**155 条**(fp8 档 27B,一次 boot + 一次请求);终局 **0 条**。
- 迭代轮次:5 轮(每轮修完再跑);f16 档另跑 3 轮。

### 19.5 审计发现(**两条是真 bug**)

| 发现 | 性质 |
|---|---|
| `owl_qknorm_rope_kv_insert_f16{,_knhd}` 的声明表是**旧签名**(`q, q_raw, kw, beta, cos, sin, k, v, kc, vc, …`),`.cu` 早已是 `q_raw, k, v, key_cache, value_cache, slots, q_w, k_w, …` —— **token 串逐字相同**(11×T+f32+4×i32+T)⇒ 老门全绿,而**语义槽位全错位** | **f16-KV 档融合插池 = 读写全错**(静默);新增**名字级互证门**焊死这类 |
| `vllm_paged_attention_v2_{f16,fp8}_knhd_*` **共用一张表**,而 fp8 池句柄是 U32 扁平字节账、f16 池句柄是 kNHD 分页视图 ⇒ 一表两用必错 | 拆两表(按 `$pool_dt` 选) |
| `block_leaf_dt`(引擎叶子注解)把非 2B 类型**一律折叠成 F32**(legacy f32 基线残留)⇒ fp8 池块以 U32 分配却注解 F32,与声明(U8)、调用点三方各说各话 | 修引擎:**如实传递** |
| `dflash_kv_leaves` 草稿池叶子注解用 `self.dims.dtype`(目标池 F16),而草稿池是 **BF16 分配** | 修引擎:BF16 |
| XQA 四件套(page_table/seq_lens/mask/semaphores)= **图输入槽**(写 f32 数值、核按 u32 读 = 位型走私)⇒ 声明写 F32 + 注记 | 对齐 |
| 24 条 rank:层侧顺手扇平(`key/k/q` = `[T, HK·HD]`)而核真按 3D 索引 ⇒ **改调用点 reshape**(零成本元数据) | 对齐 |
| `beta/gate`(gdn 两臂)、`a_log/dt_bias`、`pos/ids/anchor/gamma/conv_state/slots/cu_seqlens` … 声明陈旧 ⇒ 改声明;逐元素核(add/mul/sigmoid/silu/cast/argmax/embed-ids/concat 行/rope pos)⇒ **显式 `rank = 0`** | 对齐 |
| `owl-models` 测试目标**从未编译过**(`model_prefill` 少一参 + 测试里用 `?` 返回 `()`);修好后 123/123,并当场抓出**测试自身**的 dtype 错(`LoaderCtx::default()` = F32 的 rope 表喂进 `const __half* cos_t` 的融合插池核) | 顺手修(测试域) |

### 19.6 新门

| 门 | 判据 |
|---|---|
| `op_tables::op_tables_match_cu_param_names`(**新**) | 声明表形参**名** == `.cu` 形参名(逐位含序);121 张 native 表全绿。取证口与 token 级**同一实现**(`cu_sig::cu_params_of` 三形态:宏展开/单层宏实例化/字面核) |
| `spec::tests` +7 | 事实面正例 / dtype 负例 / rank 负例 / **事实缺失必拒** / `rank=0` 通配 / Out 禁 BlockSlice / 未知块拒 / 几何律 / `slot_sources` 与 O 位一致 |
| **`op_tables::non_family_out_slot_is_last_tensor`(新)** | 非族算子 `out_pos()` 必须 = **最后一个张量槽**(现役回传 ABI:"末位 Block = 输出";家族豁免 —— marlin AWQ 的 out 在槽 2,其 runtime 按名取 out)。**负例验红**(把 concat 的 out 移到首位 ⇒ 点名报错) |
| **`op_tables::pool_writers_have_a_carrying_slot`(新)** | ① 声明 `write`/`rw` 池 ⇒ 必有 `InOut`/`Out` 槽承载写入 ② `rw` 池 ⇒ 必有 `InOut` 槽(反向不查:激活域原地槽合法)。落地时**修 6 处角色不如实**(gdn `conv_upd`/`delta_dec`/`decode_step`:声明 `rw(GDN_STATE)` 却写 `In`) |
| A17 锁表 | 11 个族算子的 (名序, 类型签名) 按新声明**重生成**(`gdn_chunked` beta/gate = F16x2;`gdn_scalar` = F32x3/F32x2;xqa/FI/cublas/marlin 池句柄口径) |

### 19.7 验收

| 判据 | 结果 |
|---|---|
| 单测/门 | kernels **63/63** · **op_tables 10/10**(+`out_pos` 门 +池写集↔角色门)· no_runtime_compile **2/2** · models **123/123**(目标首次可编译)· cuda **6/6 + 2/2 + 1/1** · shared **39/39** · server **33/33 + 2/2 + 1/1** |
| 真机 fp8 档(6 请求 × ptok 543…2453) | **6/6 `tool_calls`** + `[L3 事实面]` 拒绝 **0** |
| 真机 **f16-KV 档**(同模型,`kv_fp8=false`) | boot/plan/请求全绿 + 拒绝 **0**(这条档就是上面那个静默 bug 的现场) |
| 事实面口径 | **服务端** `check_from_params` 唯一门(客户端零自查);`Arg`/`Bytes` 携带 dtype/rank;`DeviceRes::resolve(id)` |
| 存量失效 → **已修** | `cuda/tests/gpu_server.rs` 的内联 `owl_scale_f32` → **真资产核 `owl_add_f32`**(`KernelSource.source` 须给资产 id)⇒ 契约套件 **2/2 复活**(自 NVRTC 退役以来首次可跑);`apps/server/tests/e2e.rs` 的 **I4 假红**同批修(SSE `arguments` 转义截断 ⇒ 改按 SSE-JSON 解析拼 delta)⇒ e2e 转绿 |
| 数值指纹 | 同一探针点(t=31/379/405)gap = **4.57812 / 4.32812 / 4.56250** 与改动前**逐位相同** |

### 19.8 诚实边界(未验路径)

- 审计覆盖 = **该档跑到的路径**(fp8 档:XQA/FI/K0/fp8-pool/草稿 BF16;f16 档:FI f16/
  paged v2 f16/fused insert f16/草稿 BF16)。**未跑到**的臂(naive decode、classic 池臂、
  0.8B W4A16 档、holo/nex 档)只做了**静态对齐**(按池口径推断 + 名字级门 + `.cu` 视图),
  未做真机验证 —— 谁开那些档,门会当场把不对齐的地方**指名报出来**(不会再静默)。
- `rank = 0` 目前用在:逐元素核(计数扁平索引)、行栈/位型表/池句柄等**视图随臂变**的槽。
  它是**显式**声明(不是漏写),但确实弱于具体阶数 ⇒ 复核时优先看这些槽。
- `ShapeRule`(符号维)仍是**空机制**(121 张表 0 处填)⇒ shape 级检查(维度顺序/转置)
  仍未落地;本轮把事实面接口开在 `TensorFacts` 上,将来加 shape 只需扩该结构 + 求值器。

## 二十、声明面收尾:尺寸自证 + 按名取址 + rank=0 复核(2026-10-15;台账 §F F2/F3/F5c)

§18.4 遗留三项里的前两项当日落地(第三项 = prefill 上 CUDA Graph,另账)。

### 20.1 F2:家族面**入参尺寸自证**(`DeviceRes::resolve_len`)

**问题**:家族 runtime 直连 FFI/launch(不经 `Command`),入参**尺寸**只在注释、
装载器、公式三处各自记账 —— 少分配时 FFI/内核**不会当场报错**,而是越界写邻块
(A13 期间"邻块 clobber"正是头号嫌疑)。

**落地**:

| 面 | 变化 |
|---|---|
| `DeviceRes` | 新方法 `resolve_len(id) -> usize`(**必实现**,不给缺省:每个资源面自己声明"账长怎么来的");与 `resolve` 走**同一条解析路径**(`GpuRes::resolve_span`,错措辞只有一份) |
| 实现面 | `GpuRes` 覆写(账房 `block_ptr` 的物理长度)+ `&mut dyn DeviceRes` 转发 + **4 处测试替身**(registry `NopRes` / gdn chain / cublas isolation / oracles) |
| marlin | `ws` 严格比公式(`v2_workspace_len(n) × 4 B`)——公式与拒语原文住 `family::marlin`(server 只做 `String → OpError` 映射,**不复制算式**);`c_tmp` 守**非空下限 1 word**(固化路径 `use_fp32_reduce=false` 不触碰,精确公式依赖运行时 m/sms ⇒ 只守"槽位有实体,不可零容量/空 alias") |
| xqa | `scratch` 守**下限** `XQA_SCRATCH_FLOOR_BYTES = 1 MiB`;精确需求随 geometry 变且住内核 `MemSegmenter`,host 侧重推 = **复制内核常量** ⇒ 违本仓"零常量复制"律 ⇒ 只守量级(抓 1 word/4 KiB 类错,不抓 1.5× 少分配;生产定档 32 MiB、测试 64 MiB 远在其上) |
| 边界语义 | `BlockSlice` 兼容:可用字节 = **账长 − 块内偏移**;不足 ⇒ `OpError::Contract`(field 指名槽位,`expect` 给需求字节,`got` 给实得字节) |

**验收**:纯函数单测 3(够长放行 / 短一截拒 / 零容量拒;含"拒语须指名槽位")+
**接线负例 1**(`server/marlin.rs::f2_gate`:账长替身 + AWQ 消息 ⇒ 门在 FFI 前拒,
`field="ws"`/`"c_tmp"`;够长时放行到替身的 `stream()` 上,证明拦它的不是门)+
真机 `selfcheck_oracles` **8/8**(marlin oracle 走的就是公式尺寸 ⇒ 无假阳)。

### 20.2 F3:`parse` 与 builder **成对按名取址**

**问题**:6 个 client parse 全是**位置切片**(`blocks[0]`/`scalars[0]`) + 个数断言
⇒ "槽序"知识在**声明表 `PARAMS`** 与 parse 各存一份;同型槽对调时 L3 放行而
解析器**静默错值**(gdn 两族的注释里就写着这条已知缺口)。

**落地**:

| 面 | 变化 |
|---|---|
| `client::NamedSlots`(新) | 线格式的**按名视图**:`new(msg)` 要求算子**已建表**(未建表 = 结构化拒,不回落位置切片)→ `block(name)` / `flat_block(name)`(零偏移口径,gdn 面)/ `out_block(name, elems)`(dtype/rank 取**声明**侧)/ `u64(name)`(sz 与 F32Bits 走私槽) |
| `client::BlockRef` | **唯一住址**(原 cublas/flashinfer/marlin/xqa 四份同构副本删除,各族 `pub use` 保持旧路径) |
| 6 个 parse | cublas / marlin(两臂)/ flashinfer / xqa / gdn_chunked / gdn_scalar 全部改按名取址 |
| **`PARSE_SLOTS` 锁**(新,`client::tests::parse_slots_are_exactly_declared`) | 每个 parse 的槽名表与声明表**集合逐字相等** ⇒ 改名/插槽/漏跟 = **测试期红**(不等到运行时)。**另一半锁** = `ops::tests::slot_order_lock_family_ops` 的名字**序列** ⇒ **同型槽对调**由它拦。两把锁缺一不可:集合挡"漏跟",序列挡"对调" |

**判据边界(诚实)**:线格式不带槽名 ⇒ **手搓消息**(测试/oracle/探针)若把同型槽
摆错位,按名取址读到的仍是"声明意义上的那个槽";这类由 L3 的**槽类型对位**+
各族对拍门兜。真正的收益是:**解析器不再持有第二份槽序**(改声明必须连带改的具体
位置只剩 `PARSE_SLOTS` 一处,而它被机器锁着)。

**顺手抓到真 bug**:**`gdn_scalar` 的声明把线上第 3..6 个标量整体错标一格**
(线上真语义 `ns, nv, nk, kd`,旧表写 `nv, nk, kd, vd`)⇒ 该文件 `check` 里
`slot` 越界校验的 stride 算成了 `ns·nk·nk·4`(**过宽**,真越界不报;正品是
`hv·kd·kd·4`)。修正 = 声明改名 + `check` 变量名 + A17 锁表项随改。

### 20.3 F5c:`rank = 0` 复核(真机 rank 表;工具入库)

A20 把逐元素核/多臂共用槽显式标成 `rank=0`(阶数无关)——弱于具体阶数。要判
"能不能收紧"不能靠读调用点猜,故:

- **工具**:`OWL_L3_AUDIT=1`(在 `check_from_params` 事实面里,只对声明 `rank=0`
  的槽生效;**OnceLock 门 ⇒ 关账 = 一次原子读**,去重键 = (op, 槽名, rank) ⇒
  每个三元组只打一行)。工具**入库**(F5a 未验档的验证同样用它)。
- **数据**(真机 27B,每档 boot + 3 个请求 90/950/2000 词):

| 档 | 三元组数 | 备注 |
|---|---:|---|
| `ls-off`(fp8 KV + XQA + chunked) | 70 | 含 `vllm_chunked_prefill_paged_attn_opt_fp8_knhd_hd256` 四件套 = rank1 |
| `ls-f16`(f16 KV + FI/`paged v2`) | 70 | 同一槽可能两档都见;差异 = 池族专属算子(`_f16_knhd` / `_fp8kv_knhd`) |

- **结论(逐类)**:
  1. **同一 (op, 槽) 出现多个 rank ⇒ 必须维持 0**。实测多例:
     `owl_norm_rope_*/pos`(1,2)、`owl_embed_f16::ids`(1,2)、
     `owl_concat_rows_*::i0..i7`(1,2)、`owl_reshape_and_cache_*::slot_mapping`(1,2)、
     `owl_dflash_select_bf16::anchor`(1,2)、`gdn_conv_fwd::cu_seqlens/slots`(1,2)、
     `gdn_conv_fwd::conv_state`(1,2,3)、`gdn_recurrence_varlen_gqa::state`(1,4)、
     `vllm_paged_attention_v2_f16_knhd_hd256bs32::alibi_slopes`(1,2)、
     `xqa::{mask, wr}`(1,2,**表内已注记**表两臂共用)。
  2. **占位/旗标槽 ⇒ 诚实维持 0**:`...::sinks`(层侧传 `kv.slots` 占位、
     `use_sinks` 关)、`alibi_slopes`(多数臂传常量块)——它们**语义上无阶数**。
  3. **单档单 rank 的候选(auto-tighten)不采**:该 op 是**全仓共用**的
     (逐元素/铸核:add/sigmoid/cast/argmax/…),其他 flavor(0.8B/nex/holo/未验臂)
     可能喂别的阶数 ⇒ 现在收紧 = 拿 F5a 未验路径去撞门。**收紧的前置** =
     逐档跑 `OWL_L3_AUDIT=1` 拿到跨档一致的单 rank 表(工具已就位)。

### 20.4 验收(同配置 A/B,**逐位**)

| 判据 | 结果 |
|---|---|
| 单测/门 | kernels `--lib` **68/68**(默认 feature;+F2/F3 新增 6)· `--lib --features device,cublas,marlin,flashinfer,xqa` **76/77** —— 唯一红 = `registry::tests::knows_and_execute_share_the_table`(**A16 起就红**:空工厂表必撞 A16 孤儿声明门;非本轮引入,见 §F F8)· op_tables **10/10** · no_runtime_compile 2/2 · models `--lib` **123/123** · cuda **2/2**(gpu_server)+ 1/1 + 2/2 · shared 39/39 · server `ops_whitelist` 2/2 |
| 真机(GPU 0,selfcheck) | `selfcheck_oracles` **8/8**(marlin oracle = F2 门的正例路径) |
| 真机 27B **fp8/chunked 档**(6 请求,ptok 543…2453) | **6/6 `tool_calls`** + 探针 `gap = 4.57812 / 4.25000 / 4.32812 / 4.67188 / 4.56250 / 4.32812` —— 与 A20 基线**逐位相同** ⇒ F2/F3 **零代价、零漂移** |
| 真机 27B **scalar 档**(`gdn_chunked=false`;F3 改名命中面) | boot + **3/3 `tool_calls`**(`gap ≈ 4.48 / 4.27 / 4.47`,与 chunked 档同健康区) |

### 20.5 追加(同日,F6 批量 + F8)

| # | 项 | 结论 |
|---|---|---|
| F6a | 生产档 `gdn_chunked` 回落**已撤**(用户裁决) | `testdata/server/agent-27b.toml` → `[dispatch] gdn_chunked = true`(A13 回落注释改写为撤回依据)。撤后真机(8137,chunked + prefix_cache):boot 绿 + **3/3 `tool_calls`**,TTFT 0.83/1.05/1.51s(scalar 臂对照 1.01/1.62/2.60s)⇒ 恢复 FLA 六核 prefill(~2100µs → ~383µs / 层 chunk) |
| F6b | `reference.rs` 的 `add`/`mul` FIXME | **不是过时方法**:`reduce` 的 `Op::Call{SemanticKernel::Add/Mul}` 臂就派发到它们,层侧(decoder 残差/dflash2 累加/mtp)真发 Add/Mul ⇒ `CpuInterpreter` 作为对拍锚(与 `owl-cpu`/`owl-cuda` 互为独立副本)必须实现。FIXME → 判据注记 |
| F6c | `cuda/build.rs` 的 `CARGO_FEATURE_CUPTI` | 改 `cfg!(feature = "cupti")`(cargo 把本包 feature 作为 `--cfg` 交给 build script;`CARGO_FEATURE_*` 是实现细节、且会被外部环境继承污染)。实测:开 feature 真编 shim、关 feature 零动作 |
| F6d | 热路径计时器 | `handle_launch` 原**无条件** `Instant::now` + `kernel.name.clone()` + `format!`(而 `srv.tx`/`srv.queue` 两段早已由 `sink.timing()` 门控 ⇒ `srv.op` 是唯一漏网)⇒ **先问 `sink.timing()` 再建载荷**;gpu_prof 臂的注释同处改注(该臂由 bool 门控,且开启档本就要逐核 `synchronize()`)。真机 A/B:探针 gap 仍**逐位相同** |
| F8 | `registry::tests::knows_and_execute_share_the_table`(A16 起在 device-feature lib 跑下必红) | 孤儿门(A16)**射程改为激活集**:`allowed = Some(白名单)` 时只审白名单内声明(语义 = "会被发射的声明必须有族"),`None` 保持全量严格。空激活集 ⇒ 零孤儿 ⇒ 裸注册表可构造,该测试复活。**kernels lib 带全 feature 77/77 全绿** |

### 20.6 agent 服务面复验(A10 结案 + pi 真机端到端;2026-10-15)

**A10(跨会话块级前缀缓存 \"失效\")定谳 = 档口径,不是引擎**:首轮复测所用的两档
(`split-on/off.toml`,以及 `ls-off/ls-f16.toml` —— 都是 `ls-off` 副本)全部
`prefix_cache = false` ⇒ 跨会话块级缓存**根本没开**,T5 `cached_tokens=0` 是应然。
生产档(`agent-27b.toml`,`prefix_cache = true`)同探针:`T2 同会话 217/239`、
**`T5 跨会话 1536/1732 = 88.7%`**、content sha 等价 ⇒ **G4 全绿**;
`gdn_chunked = true/false` 两臂都命中(与 prefill 臂无关)。
残留**设计限制**(D1 已记,非 bug):快照只在分块 prefill 末位 32 对齐处拍照 ⇒
短 prompt 常无快照可复用(≥2 分块的长 prompt 稳定命中)。

**真机 pi → owl 端到端**(判据 = M-A 的\"工具闭环\"):
`pi --provider owl --model qwen3.8-27b -p \"用 bash 写 owl_probe.txt,再用 read 读回\"`
⇒ 3 轮、12 tools、**模型真发 tool calls**(bash 建文件 + read 回读)、文件落地内容
`owl-agent-ok`、末答正确;服务端侧:末轮 prompt **12950 / cached 12931**(增量只 19 行
⇒ §4.3 对称律的会话增量复用生效),零 abort 零 error(唯一 WARN = pi 多发的
`store` 字段,收下未生效)。

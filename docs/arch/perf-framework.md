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

档:`testdata/server/decode-{graph,eager}-27b.toml`(同 prompt 410 tok、同 32 输出、
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

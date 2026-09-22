# owl GraphLease 依赖追踪 —— Signal 语义设计(设计稿,未实现)

> 2026-09-22。动机:Vue 的 signal 语义证明了"依赖追踪可以不靠人工登记"——
> effect 执行时**读过的信号自动成为依赖**。我们把它搬进 graph 捕获:
> **捕获闭包执行时,读过的缓冲自动成为图的租约**。替代手工 `session.lease()`
> 调用,消灭"漏登一个依赖"这一整类 bug。
> 状态:设计稿。M1 实现;本文档为唯一需求源。

---

## 一、Vue 语义 → owl 语义的映射表

| Vue | owl | 语义 |
|---|---|---|
| `ref(value)` | `Tensor/PoolBuf`(自带 `BufToken{id,gen}`) | 被追踪的值;出生即有身份 |
| `effect(fn)` | `capture_session().capture(\|frame\| {…})` | 追踪作用域:执行期**读**操作自动登记依赖 |
| 读取 `ref.value`(在 effect 内) | `frame.ptr(&tensor)` / kernel 参数引用 tensor | **追踪点**:登记 token → 当前作用域 |
| `ref.value = new` | `gen += 1`(generation bump) | 触发:依赖此信号的效果"过期" |
| effect 重跑 | re-capture 或 validate-fail | 过期响应策略 |
| `effectScope` | `CaptureSession` | 作用域边界;scope 死 = 依赖全解 |
| `untracked(fn)` | `fill_from_host` / 旁路写入 | 刻意不追踪的操作(E 阶段邮箱写入) |
| computed 缓存 | —(一期不需要) | |

**核心不变量**:replay 是"冻结的 effect 重跑"。Vue 里 effect 读信号 → 依赖自动
收集 → 信号变化触发重跑;owl 里捕获期读缓冲 → 租约自动收集 → 缓冲世代变化
触发校验。**两句话是同一个语义。**

## 二、机制设计

### 2.1 追踪开关 = thread-local 追踪栈

```rust
// owl-cuda
thread_local! { static TRACK: RefCell<Vec<ScopeId>> = ...; }
pub fn tracking_scope<R>(f: impl FnOnce() -> R) -> R;   // 压栈
pub fn current_scope() -> Option<ScopeId>;               // None = untracked
```

- `capture_session().capture(body)` 内部 = `tracking_scope(|| body())`;
- KernelCtx/Tensor 的 `device_ptr()`/launch 封装在取地址前查
  `current_scope()`:Some → 登记 token;None → 零开销直通
  (一个 thread-local 判读,热路径无关——E 阶段根本不开 scope);
- 这就是**哨兵①的自动化**:不再要求 ops 手工 `trace_launch/trace_buf`,
  launch 封装自动做(哨兵②闭包借用检查保留,负责别名冲突)。

### 2.2 依赖记录 = 区间→token 反查(哨兵①已有地基)

kernel launch 参数里的地址是裸 u64。登记走既有设施:
`Pool.malloc` 时把 `(addr_range → token)` 注册进设备的区间索引;
launch 拦截点拿到地址做 O(log n) 反查,命中 → `scope.subscribe(token)`。
未命中 = 逃逸缓冲(非池分配)→ 捕获期直接 LawViolation
(封闭性证明的运行时执行)。

### 2.3 触发 = 世代 + 存活台账(已有地基)

- `PoolBuf` drop → `retire(token)`(已实现):**依赖失效信号**;
- `DeviceGraph.launch()` 前逐租约 `validate`(已实现接口):失效 →
  结构化报错(A2.7 语义,不 Xid);
- M1+:`invalidate` 还可以主动通知(图标记 dirty → 决策 re-capture
  或 fail),这是"响应式"的完整形态,一期只做校验形态。

### 2.4 粒度:token 级,不是字段级

一期追踪粒度 = 整缓冲(Vue 的 ref 也是整值)。分字段追踪(张量的切片)
留待真实需求出现——切片分配属于 pool 的子分配问题,不归 signal 管。

### 2.5 与三哨兵/既有资产的关系

| 哨兵 | 旧形态 | Signal 化后 |
|---|---|---|
| ① launch 拦截 | ops 手工 trace | **launch 封装自动 subscribe**(本设计) |
| ② 闭包借用检查 | capture 闭包 &/&mut | 不变(编译期,与 signal 正交) |
| ③ cuGraphGetNodes 审计 | 人工 diff | 不变(驱动真相 vs signal 记录的对账) |

A1.2 延迟归还 = 强租约的执行臂(不变);定影协议 = footprint 对账(不变);
EagerOnly(fill_from_host)= untracked(不变,且语义上终于"名正言顺")。

## 二·五、统一不变量:八大死亡姿势的 Signal 收编(2026-09-22 设计)

> 推演结论:把追踪对象从"缓冲"扩展到"一切 CUDA 资源操作",
> 图鉴 10 姿势中的 8 条被同一个不变量统一封死:
>
> **【不变量 Φ】捕获作用域内,一切 CUDA 资源操作(发射/分配/拷贝/
> 流切换)必须经由追踪栈;栈外操作 = LawViolation。**

四个执法 choke point(全部是我们自己的代码,零第三方依赖):

| Choke point | 拦截的姿势 | 执法 |
|---|---|---|
| ① launch 封装(Kernels) | 2 依赖登记、8 节点记录 | 自动 subscribe + 地址反查 |
| ② Pool.malloc 原语 | 5 池内不当分配、7 池外分配、4 库懒分配(预登记后即合规) | Capturing 相仅允许已登记池的 scratch;未登记分配 = 违约 |
| ③ ffi 白名单包装(相位感知) | 1 D2H 回读、10 host 暂存 | `memcpy_dtoh_sync` 等 Capturing 相调用 = 违约 |
| ④ stream 所有权(session.stream) | 9 野流、1 legacy 流 | launch 仅接 session 流 |

姿势 3(存活期回收)升格为不变量:"**retire 有活跃订阅者的 token = 违约**"
(现有延迟归还为其执行臂)。姿势 6(warmup)= "effect 未完整执行过一次
不可 seal"(定影协议已隐含,制度化)。姿势 8 = 触发→响应的完整形态
(M1+ 策略化)。

**Signal 解不了的(诚实清单)**:BAR1/VMM 物理约束(A2.8,物理定律);
性能策略(cublas 入图时机、档位数量);warmup 的内容清单(过程性知识)。
Signal 提供的是"违规必被抓住",不是"正确决策自动做出"。

## 三、风险与边界

1. **thread-local 与 runner 线程模型**:A3 规定每卡一线程,tracking
   栈天然按线程隔离——与单进程多实例(A2.6)兼容;多线程并发捕获
   (两卡同时)各自独立栈,无干扰;
2. **逃逸缓冲**:非池分配的裸地址在捕获段出现 = 未命中 → 直接
   LawViolation。这比 vLLM 强(它靠纪律),但要求 ffi 白名单 kernel
   全部走封装——准入铁律覆盖;
3. **性能**:追踪只存在于捕获期(冷);E 阶段 scope 栈为空,
   `current_scope()` 一次 thread-local 读,热路径零影响;
4. **嵌套捕获**(capture 内再 begin 子流):栈天然支持,一期禁止嵌套
   (不变量断言)。

## 四、实现拆解(M1 内,预计 0.5-1 天)

1. iface:`ScopeId`/`subscribe` 词汇(0.5h);
2. owl-cuda:tracking 栈 + `Pool.malloc` 区间登记 + launch 封装
   subscribe(1h);
3. owl-cuda:`CaptureSession` 自动化——capture 闭包包 tracking_scope,
   session.end 消费 subscriptions → leases(替代手工 lease 调用)(1h);
4. 测试:漏 lease 自动收集(对照手工版)、逃逸缓冲 LawViolation、
   untracked 旁路、E 阶段零追踪开销冒烟(1h)。

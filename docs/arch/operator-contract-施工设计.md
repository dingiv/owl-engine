# 算子契约权威 —— kernels 双面基础设施与名字无关 server(Result 全链)

> 版本:v0.5(2026-10-12 深夜;**状态:全部里程碑完工**——M1 金标归位 / M2 骨架 / M3 全家族面+models 切换(gdn_chunked 全链样板)/ M4 一键切换(registry 现役,foreign.rs 退役,净删 ~1320 行)/ M5 manifest 审计面+双侧名字面量门)。
> **v0.4→v0.5 变更(依赖现实修订,M2 实测)**:
> ① **契约老家 = owl-kernels::contract**(用户裁决:iface 依赖 kernels、
>    kernels 不依赖 iface,则 iface 的线格式 Bytes/LaunchMsg/Arg/Dtype
>    **搬进 kernels**,iface 经 re-export 零破坏续用)——依赖方向
>    iface→kernels 不变,契约单源在 kernels。新增词汇:OpError(Result
>    全链错误模型)/OpId/Linkage/Law/FieldStats/Stage/InvariantBox;
> ② **服务端面 = kernels::device(DeviceRes+Exec,feature="device")+
>    kernels::registry(FamilyRuntime/OpRegistry)**;feature 名为 device
>    非 cuda —— 后者 cu/ops.cu 预编链为存量断链(backends 从不启用);
>    runtime 家族落 backends/cuda::ops(Exec 需真 stream/cubin 柄,
>    iface 无 cuda 装不下),server 主循环零算子名不变;
> ③ models 侧引用方向 lint 已落地(crates/models/tests/facedir_lint.rs)。
> **v0.3→v0.4 变更(用户裁决)**:**执行动作全部下沉 kernels**——链路形态
> 是 server 可感知的事实,但装载/编译/发射原语的**实现**属 kernels。
> 原 DeviceCtx(server 实现的厚接口)拆为:DeviceRes 资源面(server 实现,
> 只剩状态:ctx/stream/块表/账本/捕获态)+ Exec 执行引擎(kernels 实现,
> 全部机械:nvrtc 缓存/cubin 模块表/launch/staging/体检)。
> **v0.2→v0.3 变更(用户裁决)**:
> ① **范围 = 全部算子**,不做"样板族+后续跟进"——五 foreign 家族全量
>   同批改造,native 面客户端面全量升级;gdn_chunked 在文中仅为示例展开
>   (它契约债最重),不是范围裁剪;
> ② **server 允许感知链路形态**(静态编译/链接 vs 动态载入/编译,§4.5):
>   全仓实为三种(StaticLib/AotCubin/Nvrtc),装载时机、校验门、失败面
>   随形态完全不同,升格为注册表 schema 一等字段;server 感知形态,
>   仍不感知名字。
> 触发:g 语义案(2026-10-11,perf-roadmap 补录Ⅳ-Ⅸ)复盘 + 三方对质实证
> (gdn 槽序契约字符串与真实协议不符,§1.2)。
> **v0.1→v0.2 变更(用户裁决)**:
> ① **server 零算子名感知**——连分派表都不该在 server,server 不该知道
>   任何具体算子的存在;
> ② **kernels 升格为双面基础设施**:它现在只是个注册表,没有任何实际
>   逻辑;要在这一层加**契约逻辑 + 底层 API**,分两套——
>   **客户端面**(面向 models:注册名 + 函数类型框架,保证调用不可能错)
>   与**服务端面**(面向 server:契约执行 + 底层发射原语,供其通用调用);
> ③ models 层只拿名字和类型框架,不碰参数摆位。
> 关系:charter A1.6 / A2.7 / A5.2 算子域落地;KvEnv 收口与 DraftPoolAddr
> 同款"单一权威"模式的推广至算子全域。

---

## 一、病灶(证据链,当日核实)

### 1.1 事故谱 × 防线缺口

| 案 | 现象 | 该接住的防线 | 为什么没接住 |
|---|---|---|---|
| 断链案(10-03 冻结根因) | handler 输入指向从未灌数据的死 scratch | 契约校验 | 槽序是字符串约定,死指针也是"合法的 T 槽" |
| SASS 剪参案(补录Ⅴ) | h 核 gk 被 triton prune,14 参错发 13 参 | ABI 机器校验 | 只有人肉 cuobjdump,cubin 签名零机器对账 |
| **raw g 案(补录Ⅸ)** | kkt/h/o 吃 raw g,e²⁴ 爆炸,染毒 60 层 | 类型/不变量/金标 | 类型数量全对只有语义错;不变量门不存在;金标红灯 |
| slot≥1 崩塌案 | 草稿池两套寻址隐式协定 | 寻址权威 | (已治:DraftPoolAddr——本设计全域复制该模式) |
| 契约腐烂案(**当日新发现**) | `GDN_CHUNKED_SLOTS` 字符串与真实协议不符(g/beta 块序颠倒、标量 7≠6、o_f32 已废) | —— | **契约字符串全库零消费,烂了没人知道** |
| 金标平行实现案(结构性) | golden 测试手搓发射链 vs handler 手搓发射链,两份编排 | 单一实现 | 两份手抄天然可分歧(golden 喂 gcum / handler 喂 raw g 正是分歧现场) |

### 1.2 三方对质(gdn_chunked,当日实证)

| 层 | 位置 | 它眼里的协议 |
|---|---|---|
| models(调用方) | `layers/gdn.rs:649` 手摆 `Vec<Arg>` | q,k,v,**beta,g**,state + **6** 标量 |
| kernels("契约") | `gdn_chunked.rs:38` `GDN_CHUNKED_SLOTS` 字符串 | q,k,v,**g,beta**,state + **7** 标量——**零引用,已腐烂** |
| server(执行方) | `foreign.rs:511` `parse_foreign_slots(&msg, 7, 6)` 按数量盲切 | q,k,v,**beta,g**,state + 6 标量 |

### 1.3 资产与哨兵失明

- cubin 资产 gitignore:资产换血 git 零 diff,代码-资产一致性无机器门;
- 金标红灯两度恶化(ILLEGAL_ADDRESS 存量案 → 资产换血后 NOT_FOUND,
  测试源还是 pip-fla 符号名);金标 g 域 [−0.51,−0.01] 不覆盖真实深谷 −24。

### 1.4 Result 缺位(病根)

```rust
// 现状:函数返回 (),错误走 ack 侧信道,编译器管不了
pub(super) fn handle_gdn_chunked(&mut self, msg: LaunchMsg,
                                 ack: Ack<Result<Bytes, ModelError>>) {
    // ...30+ 处手工模式:
    if let Err(e) = unsafe { b.launch(cfg) } {
        return ack.send(Err(ModelError::Msg(format!("gdn kkt: {e:?}"))));
    }
    // log-and-continue 也存在:
    Err(e) => eprintln!("[blas-ws] 分配失败 {e:?}(回退池分配)"),
}
```

三宗罪:①send-exactly-once 无结构保证(漏发=悬挂,多发=未定义,全靠人肉
枚举早退路径);②`()` 返回 = 错误不可 `?` 组合,30 个手工 if-let 就是
30 个遗漏机会;③检测与处置合谋——降级/拒服/吞掉散落同层,没有策略位。

### 1.5 职责倒挂(名字感知污染)

server(`backends/cuda/server/foreign.rs`,1147 行)内嵌五个算子家族的
**全部**业务逻辑:cubin 装配、scratch 池、meta 表、多核编排、dtype 铸造。
server 本该是"通用内核发射器",现在是一个知道每个算子私事的管家。
对照:**native kernel 通道早就是对的形状**——`LaunchMsg{name, source}`
进来,通用编译缓存 + 通用发射,server 零算子知识。foreign 臂破坏了这个
不变式;本设计把它恢复,并把恢复后的形状统一到全部算子。

### 1.6 全量盘点(范围 = 全部算子,v0.3 用户裁决)

**A. server 名字感知污染区(foreign,5 家族 / 10 名字绑定,全部改造):**

| 家族 | 名字绑定 | server 现状 | 契约债 | 客户端面落点(models) |
|---|---|---|---|---|
| cublas GEMM | `GEMM_F16` / `GEMM_BF16` | handle_cublas_gemm(含 ensure_blas 句柄/工作区) | bf16 三参枚举、m/k/n 手摆 | linear.rs 等量化/GEMV 调用点 |
| marlin GEMM | `GEMM_W4A16` / `_AWQ` / `_BF16` | handle_marlin_gemm ×3 变体 | 变体选择散点、workspace | linear.rs:168(动态 name+sig 字符串) |
| FlashInfer prefill | `PREFILL_FI` / `_FP8KV` | handle_fi_prefill(plan/run 分离、页表构造) | kv/f16 变体选择、页表契约 | attention.rs:602 / 2118 |
| GDN chunked | `GDN_CHUNKED_FWD` | handle_gdn_chunked(~500 行,六核编排) | **契约字符串已腐烂**(§1.2) | gdn.rs:649(**全库唯一手抄名字面量**) |
| GDN scalar | `GDN_SCALAR_FWD` | handle_gdn_scalar(含 soff/o_f32 捕获战果) | scale_bits 手操、T 分臂 | gdn.rs ~627 |

**B. native 面(通道已名字无关,客户端面同病,同批改造):**

- 注册表 94 Entry(kernel.rs):手抄名 + `args: "T,T,sz"` sig 字符串,
  现仅 KV 写者族有 boot manifest 门(kv-manifest),其余 0 机器校验;
- models 15 处 driver 直发 + 全部 `Kernel::new(...)` 调用点:名字字面量/
  常量混用,sig 与实参零对账;
- 迁移目标:OpId 全量化(名字字面量全库仅 registry 一处)+ Entry schema 化
  (sig 退役,客户端面同一套 Call 框架)+ boot 符号校验全量覆盖。

---

## 二、北极星:驱动模型

```
models(设备消费者)      只见:注册名 OpId + 类型化调用框架
                            │  调用(类型框架保证不可能错)
                            ▼
kernels(设备层,双面)   ┌─ 客户端面:OpId / Call 框架 / builder / newtype
   契约单源 + 双面 API    └─ 服务端面:Runtime / 发射原语 / 不变量门 / ABI
                            │  注册(boot 期) + 执行(Result)
                            ▼
server(通用总线)        只见:一个注册表句柄 + 一个通用执行入口
   名字无关               resolve(block ptr) / 流 / 账本 / 图捕获 —— 纯基建
```

- server 对算子的全部知识 = "有一个注册表,给它 LaunchMsg,它还我
  `Result<Bytes, OpError>`"。**server 里不存在任何算子名字字符串、不存在
  任何 `match name` 分派臂、不存在任何算子专属 handler。**
- server 感知且仅感知一个算子维度:**链路形态**(静态编译/链接 vs 动态
  载入/编译,§4.5)——这是装载基建知识(决定装载时机/校验门/缓存策略),
  不是算子知识。
- 分派(名字 → 具体 runtime)发生在 kernels 注册表内部;编译期穷尽
  (单二进制,无插件加载),枚举 match = 静态分发零虚表(REQ-CODE-01 C2:
  dyn 仅限注册表期——而注册本身只发生在 boot)。
- models 对算子的全部知识 = `OpId` 常量 + `XxxCall` 框架。摆参、槽序、
  dtype 铸造、标量编码(scale_bits 手撸)全部从调用点消失。

---

## 三、设计原则(六条,评审门禁)

> **P0 名字无关**:server 零算子知识(名字、逻辑、状态各为零);算子全部
> 逻辑住 kernels。违反即返工。
> **P1 契约单源**:一份 schema,三处视图;builder/parser/文档同源派生,
> 字符串槽序退役。
> **P2 类型即文档**:语义不同的值用不同 newtype;非法传递编译期不可表达。
> **P3 Result 全链**:算子域零 `()` 返回、零 log-and-continue、零 ack
> 侧信道;`?` 串联到唯一出口;send-exactly-once 结构化;检测与处置分离。
> **边界裁定**:凡资源获取/IO/副作用/治理态读取一律 `Result`,接口面零
> 例外;裸返回仅限**纯且全函数**(类型已排除失败,如 to_launch 序列化
> ——类型保证不可失败,另配往返 property 测试锁死)。
> **P4 装配即校验**:符号存在性 + 资产指纹 boot 期校验,缺失 = 结构化拒启。
> **P5 输出即体检**:数学不变量闸门,违例产出类型化 Err,策略层裁决。

---

## 四、架构:kernels 双面基础设施

### 4.1 模块布局与依赖方向

```
crates/kernels/src/
  op/                          ← 算子家族(契约 + 运行时同居一地,单源)
    gdn_chunked/
      mod.rs                   — pub use 客户端面 + 服务端面
      call.rs                  — 【客户端面】GdnChunkedCall + newtype + builder
      runtime.rs               — 【服务端面】GdnChunkedRuntime(init/run/不变量门/scratch)
      abi.rs                   — cubin 声明、符号名、发射常量、SASS ABI 注记
    flashinfer/ cublas/ marlin/ gdn_scalar/ …(五家族全量同构,§1.6)
  registry.rs                  — registry! 宏 + OpRegistry(名字→schema/runtime)
  device.rs                    — DeviceRes trait(server 实现的资源面,窄)
                                 + Exec 执行引擎(kernels 实现,装载/编译/发射全机械)
  driver.rs …                  — 既有底层 API(归入服务端面)
```

依赖方向(箭头=depends on):

```
models ──→ kernels(客户端面:op/*/call + registry 的 OpId)
server ──→ kernels(服务端面:registry::init + RuntimeRegistry + Exec 执行引擎)
server 实现 kernels 定义的 DeviceRes 资源面(仅状态:块表/账本/捕获态)
kernels ──→ cudarc(底层;不经 server)
```

零环。server 不 import 任何 `op::gdn_chunked::*`;models 不 import 任何
`runtime::*`。**这是机器可查的门**:CI 加一条 lint(或编译期 assert),
禁止 `backends/*/` 路径下出现 `op::` 的客户端面引用、禁止 `models/` 出现
`runtime::` 引用。

### 4.2 契约 schema(单源)

```rust
// kernels/src/op/gdn_chunked/call.rs —— 客户端面(契约的"名字+类型框架")

/// 语义 newtype:字段私有,唯一产地铸造型
pub struct GateRaw(Bytes);       // raw 门控 f16;唯一产地 = 层侧 g 块
pub struct CumsumGate(Bytes);    // cumsum 产物 f32;唯一产地 = runtime 的 cumsum 步
pub struct StateSlot { pool: Bytes, slot: u32 }   // 寻址权威(DraftPoolAddr 同款)

#[derive(Clone, Copy)]
pub struct GdnShape { pub t: u32, pub nv: u32, pub nk: u32, pub kd: u32, pub scale: f32 }

/// 类型化调用框架 —— models 只见这个
pub struct GdnChunkedCall { /* 字段私有 */ }

impl GdnChunkedCall {
    /// builder:字段名即契约;类型不对编译不过;摆参/槽序/dtype 不可见
    pub fn builder(shape: GdnShape) -> GdnChunkedCallBuilder;
    /// 单点落图:产物与现有 interpreter LaunchMsg 完全同构(发射通道零改动)
    pub(crate) fn to_launch(&self, out: Bytes) -> LaunchMsg;
}

/// 服务端面的镜像(同 schema 派生):server 通用执行器用它收消息
impl GdnChunkedCall {
    pub(crate) fn parse(msg: &LaunchMsg) -> Result<GdnChunkedCall, OpError>;
}
```

同源保证:builder 与 parse 从同一字段表派生(宏或手写双实现 + **往返
property 测试**锁死:`parse(to_launch(x)) == x` 进套件,腐烂不可能再溜进)。

### 4.3 客户端面(models 视角)

models 层调用点(gdn.rs:649 现在 25 行手摆参数)变成:

```rust
let y = owl_kernels::op::gdn_chunked::GdnChunkedCall::builder(GdnShape {
        t: tokens, nv: self.nv, nk: self.nk, kd: self.hk_dim,
        scale: 1.0 / (self.hk_dim as f32).sqrt(),
    })
    .q(&q_n)? .k(&k_n)? .v(&v_c)? .beta(&beta)? .gate_raw(&g)? .state(&gdn.rec)?
    .launch(ctx)?;   // 内部:to_launch → interpreter 发射通道;错误 = Result
```

调用不可能错的三个机制:
1. **类型框架**:q/k/v 的 shape/dtype 由 builder 校验(不匹配 = `Err(Contract)`
   带 field 名;更进一步的编译期 shape 编码按需再收);
2. **newtype 产地律**:`gate_raw(&g)` 收 f16 原块;不存在"把 cumsum 产物当
   raw 传"或反向的表达(类型不同);
3. **名字单源**:models 不再出现 `"gdn_chunked_delta_rule_fwd"` 字面量
   (现状 `Kernel::new("gdn_chunked_delta_rule_fwd","")` 手抄名),一律
   `use owl_kernels::registry::GDN_CHUNKED` 的 OpId 常量——改名 = 编译错,
   不是运行期 NOT_FOUND。

### 4.4 服务端面(server 视角)

#### 资源面与执行面分离 —— DeviceRes(server 实现)× Exec(kernels 实现)

链路形态是 server 可感知的**事实**(§4.5),但装载/编译/发射这些**动作的
实现**全部下沉 kernels。切分判据 = **状态 vs 机械**:

```rust
// device.rs —— 资源面:server 是唯一实现者;只含治理状态,零执行逻辑
// P3 无例外:资源获取/状态读取/副作用一律 Result;裸返回仅限纯且全函数
pub trait DeviceRes {
    fn stream(&self) -> Result<StreamHandle, OpError>;            // 资源获取
    fn resolve(&self, b: &Bytes) -> Result<DevPtr, OpError>;      // 块表/账房
    fn alloc(&mut self, bytes: usize, tag: &'static str)
        -> Result<ScratchBuf, OpError>;               // 账本+捕获 slab+A1.7 租约
    fn capturing(&self) -> Result<bool, OpError>;     // 治理态读取
    fn record_capture(&mut self, rec: CaptureRecord)  // 副作用(A1.6 登记)
        -> Result<(), OpError>;
}

// device.rs —— 执行引擎:kernels 实现;全部机械,吃资源面干活
pub struct Exec {   // 私有态:nvrtc 编译缓存 / cubin 模块表 / staging 盒池
}

impl Exec {
    pub fn resolve_function(&mut self, res: &mut dyn DeviceRes,
        key: FunctionKey) -> Result<CudaFunction, OpError>;   // §4.5 三链路
    pub fn launch(&mut self, res: &mut dyn DeviceRes,
        f: &CudaFunction, args: LaunchArgs) -> Result<(), OpError>;
    pub fn memcpy_htod(&mut self, res: &mut dyn DeviceRes,
        dst: DevPtr, src: InvariantBox) -> Result<(), OpError>;   // 不变盒 staging
    pub fn sample_stats(&mut self, res: &mut dyn DeviceRes,
        p: DevPtr, n: usize) -> Result<FieldStats, OpError>;     // 体检核(§4.8)
}
```

归位明细:
- **留 server(治理状态)**:块表/账本(A5.2)、捕获状态机与图治理(A1.6/
  A1.7)、三流模型——DeviceRes 五方法即全部,server 侧再无执行代码;
- **下沉 kernels(执行机械)**:nvrtc 编译缓存、cubin 模块表、launch 组装、
  捕获窗守卫与 CaptureRecord 的**机械部分**(查 `res.capturing()` → 拒或
  登记)、staging 不变盒、体检核发射——连同 §4.5 的三套链路缓存,实体
  全在 `Exec`,server 只持有它;
- 横切纪律一次写成:`Exec.launch` 内部统一做"捕获期守卫 + res.record_
  登记",五家族 runtime 零感知(今天各 handler 手工绕的守卫全部退役)。

价值:捕获窗守卫、图登记、账本记账这些**横切纪律只写一次**(Exec 机械 ×
DeviceRes 状态),五个算子家族的 runtime 不再各自手工绕(capture 内首次
初始化拒绝、栈临时 staging 雷等事故的根源就是横切纪律靠每 handler 自觉)。

#### RuntimeRegistry —— boot 注册,执行期零名字

```rust
// kernels/src/registry.rs —— 注册表(编译期穷举,枚举分派)
registry! {
    ops {
        // 名字字面量全库仅此一处;Linkage 决定装载时机与校验门(§4.5)
        GdnChunked  => ("gdn_chunked_delta_rule_fwd", AotCubin,  GdnChunkedRuntime),
        GdnScalar   => ("gdn_scalar_fwd",             AotCubin,  GdnScalarRuntime),
        FlashInfer  => ("flashinfer_prefill_paged_*", StaticLib, FlashInferRuntime),
        Marlin      => ("marlin_gemm_w4a16*",         StaticLib, MarlinRuntime),
        Cublas      => ("cublas_gemm_*",              StaticLib, CublasRuntime),
    }
}

impl OpRegistry {
    /// boot 期:init 全部 runtime(符号校验/manifest 对账/scratch 预分配),
    /// 缺一 = Err 拒启。产物交给 server 持有。
    pub fn init_all(res: &mut dyn DeviceRes, exec: &mut Exec,
                    assets: &AssetManifest) -> Result<OpRegistry, OpError>;   // boot:装载/预钉全走 Exec
    /// server 唯一执行入口:名字 → runtime 的分派在 reg 内部(match 枚举,静态分发)
    pub fn execute(&self, msg: &LaunchMsg, res: &mut dyn DeviceRes,
                   exec: &mut Exec) -> Result<Bytes, OpError>;
}
```

server 侧的全部算子知识收缩为(`handle_launch` 内一处,启动时接好):

```rust
let out = self.registry.execute(&msg, &mut Ctx(self));   // 名字无关,永远这三行
ack.send(out.map_err(ModelError::from));                 // 唯一 ack 出口
```

`is_foreign()`、五个 `handle_*`、foreign.rs 1147 行 → **整体退役**,server
回到 native 通道的同构形状:一切 launch 都是"通用发射 + 注册表兜底"。

### 4.5 链路形态(Linkage):静态编译 vs 动态载入

server 允许感知**链路形态**这一个维度——有些算子静态编译/链接,有些动态
载入/编译。这是装载基建知识,不是算子知识;但必须是注册表 schema 的一等
字段,因为**装载时机、校验门、失败面**三者随形态完全不同。全仓实际三种:

| Linkage | 家族 | 产物形态 | 装载时机 | 校验门 | 失败面 |
|---|---|---|---|---|---|
| `StaticLib` | cublas / marlin / flashinfer | 构建期链接的 .a / 系统库(extern "C") | 进程启动(链接期) | 链接期符号解析(免费)+ init 期句柄/workspace 预钉 | 构建失败;运行期=句柄/工作区态 |
| `AotCubin` | gdn_chunked / gdn_scalar | 资产 cubin(include_bytes!,gitignore)+ manifest.toml | **boot 期** load_module + load_function | P4 双门:符号逐一校验 × manifest sha256 对账 | **boot 拒启**(带清单) |
| `Nvrtc` | owl_* native(94 Entry) | 源码字符串(sources.rs,在 git 里) | 首用编译,(source hash, name) 缓存 | Entry schema + boot 源码登记核对;编译错天然结构化 | 首用编译 Err(带 nvrtc 日志) |

分工就此定型:

- **server 感知**:链路形态 + 三套通用缓存(静态句柄表 / cubin 模块表 /
  nvrtc 编译缓存)——全是基建,缓存键里没有算子语义;
- **server 不感知**:哪个名字属于哪个家族、家族有几核、编排顺序、scratch
  语义——全部在 runtime 内,经 `resolve_function` 取函数、`alloc_scratch`
  要内存、`launch` 发射;
- **boot 门矩阵(P4 落地形态)**:StaticLib = 句柄/workspace 预钉
  (ensure_blas 案 → 构造期急切预钉已成律);AotCubin = 符号 × manifest
  双门,失配拒启带清单;Nvrtc = 源码登记表核对(编译错首用暴露可接受,
  或 boot 预热一笔带过,留评审)。
- **"捕获内首次初始化"事故类被结构性消灭**:三种链路的装载/预钉全部
  发生在图捕获**前**(server 构造期),捕获窗守卫成为永不触发的保险。

### 4.6 Runtime 内部(以 gdn_chunked 为例展开;五家族同构,见 §1.6 全量盘点)

```rust
pub struct GdnChunkedRuntime {
    cubins: CubinSet,          // init 期 load_function 逐一校验符号
    scratch: GdnScratch,       // 私有 scratch 池(经 res.alloc,账本内)
    meta: MetaCache,           // (T,NT) 键 host 表驻留(现状迁移)
}

impl Runtime for GdnChunkedRuntime {
    fn run(&mut self, call: GdnChunkedCall, res: &mut dyn DeviceRes,
          exec: &mut Exec) -> Result<Bytes, OpError> {
        let gcum = self.cumsum(&call.gate, ctx)?;        // → CumsumGate(唯一产地)
        ctx.check_law(&gcum, Law::GateCumsumNonPositive)?;  // 全负律(§4.8)
        let a = self.kkt(&call.k, &call.beta, &gcum, ctx)?; // 类型上吃不到 raw
        let ai = self.solve_tril(&a, ctx)?;
        let (w, u) = self.wu(&call.k, &call.v, &call.beta, &ai, &gcum, ctx)?;
        let (h, v_new, ht) = self.h(&call.k, &u, &w, &gcum, &call.state, ctx)?;
        let o = self.o(&call.q, &call.k, &v_new, &h, &gcum, ctx)?;
        ctx.check_finite(&o)?;                           // 出口体检(§4.8)
        Ok(o)
    }
}
```

注意发射函数签名:`kkt(&self, k, beta, gcum: &CumsumGate, …)` —— **raw-g
案在类型层面不可表达**;且每步 `?` 串联,零 `()`、零 ack、零 eprintln。

### 4.7 错误模型与策略表(检测与处置分离)

```rust
pub enum OpError {
    Contract { op: OpId, field: &'static str, expect: String, got: String },
    Asset    { op: OpId, detail: String },
    Launch   { op: OpId, stage: Stage, source: DriverError },
    Invariant{ op: OpId, law: Law, stats: FieldStats },
}
```

| 错误类 | 默认处置(引擎策略层,显式决策点) |
|---|---|
| Contract / Asset | 拒启(boot 期)或该请求结构化失败 |
| Launch(捕获期) | A2.7 全引擎 Paused,带 CaptureRecord/账本快照 |
| Launch(eager) | 该请求 Err;连续 N 次 → A2.7 |
| Invariant | 该请求 Err + 全文落账;同 turn 复发 → A2.7 |

`FieldStats { min, max, nan, inf }` 随 Invariant 落账——下一个"+142"自带
数值证据。策略力度是评审可调的表,不是事故现场即兴发挥。

### 4.8 不变量闸门(debug / `[probes] op_health` 启用;release 编译期剔除零开销)

| 算子·阶段 | 律 | 拦截的历史案 |
|---|---|---|
| gdn_chunked·cumsum 后 | 全负律 `g_cum ≤ 0+ε` | raw g 案(+142 一轮即爆) |
| gdn_chunked·solve 后 | A isfinite | 条件数爆(补录Ⅷ现象) |
| 各算子·出口 | out isfinite 采样扫 | 一切染毒的下游传播 |

体检含 D2H = EagerOnly,捕获段调用在 Exec 层被守卫拒(A1.5/A1.6);
合法窗口 = 捕获前 warmup + probes 事后抽样。图回放热路径不受影响。

### 4.9 资产 manifest 门

`crates/kernels/assets/<family>/manifest.toml`(**进 git**,cubin 继续
ignore):采集来源 fork@rev、生成脚本、每文件 sha256 + 字节数 + 符号名清单。
`init_all` 对账失配 = `OpError::Asset` 拒启并打印 diff——资产在测试脚底下
换血这种事,git 看不见 manifest 看得见(xinfer s3-load checksum.txt 先例)。

### 4.10 金标与运行时合流(消除平行实现)

现状金标测试**手搓一条与 handler 平行的发射链**——这正是"golden 喂 gcum、
handler 喂 raw g"能各自为政的结构根源。v0.2 后金标测试改为:真实
`GdnChunkedRuntime` + 真实 `DeviceRes`/`Exec`(生产同款)跑同一 `run()`,对拍
金标张量。**runtime 只有一份,金标测的就是生产路径**;handler/golden
分歧面从结构上消失。修活清单不变(fork 符号名/merge 核/ABI/分层容差)+
c5 深谷新金标(g 至 −24)+ 坑册三条入册。

---

## 五、迁移路径(全量施工,顺序仅是风险序不是范围裁剪)

> 范围承诺(§1.6):**5 个 foreign 家族全部 + native 客户端面全量**,终点
> 状态 = `is_foreign`/`handle_*`/foreign.rs 退役、server 全域零算子名字、
> models 全 OpId。家族间不做能力分级,只排施工顺序(依赖 + 风险)。

| 里程碑 | 内容 | 验收 |
|---|---|---|
| M1 金标归位 | §4.10 修活 + c5 + 坑册(暂以现有通道,不动架构) | 金标 5 case 全绿;恒等门四域逐位不变 |
| M2 双面骨架 | device.rs(DeviceRes + Exec)+ registry!(宏,先注册 5 家族占位)+ OpError + CI 引用方向 lint | 编译绿;lint 上线(此刻起禁新增违例) |
| M3 客户端面全量 | **5 家族 call.rs 全部落地** + models 全部调用点切 builder(gdn×2/linear/attention×2/cublas)+ native 面 OpId 化 + 往返 property 测试 ×5 | 每家族往返同构;恒等门四域逐位;全库 grep:models 零名字字面量 |
| M4 服务端面全量 | **5 家族 runtime.rs 全部落地** + registry 枚举穷举 + server 切通用执行入口 + 金标测试切真实 runtime + foreign.rs 删除 | server grep 零算子名;全套件绿;std 方差内;负例注入双拦 |
| M5 装配门 | manifest.toml ×全部 cubin 家族 + init_all 符号/指纹校验 + native 94 Entry boot 门全量覆盖 | 注入演练:换任一 cubin → 拒启含 diff;缺失符号 → boot 拒启清单 |

施工序依据:M1 先恢复裁判(后续每步有独立真值站岗);M2 骨架使后续每
家族都是纯搬运;M3 先行是因为客户端面不依赖 runtime 迁移(to_launch 与
现 LaunchMsg 同构),可全量铺开且随时可停(每家族独立合入);M4 切换
server 入口是唯一动热路径的里程碑,一次做完避免双轨长期共存;M5 纯增益门
closing。工期量级:M3 每家族 0.5 天 ×5;M4 每家族 0.5-1 天 ×5(含回归);
整体 4-5 天。

## 六、风险与兼容

| 风险 | 对策 |
|---|---|
| 重构改数值 | 恒等门四域逐位 + 金标逐核双保险;任何一位漂移 = 停 |
| DeviceRes dyn 化引入虚表开销 | 每次发射一次 dyn 调用,ns 级 vs kernel ms 级;且 launch 密度 = capture/eager 期,图回放零经过。如 profiling 立功可换枚举静态分发(C2 合规双方案备好) |
| 图态回归 | 图捕获路径仅迁移不改序;CaptureRecord 登记数逐 launch 对账 |
| LaunchMsg 通道兼容 | to_launch 产物与现状逐字节同构(property 测试锁);interpreter 零改动 |
| B5 并发前瞻 | Call/Runtime 全 owned 数据天然 Send;StateSlot 预留 per-session 分区 |

## 七、评审待决点

1. §4.7 Invariant 默认处置:请求级 Err + 复发升级(本稿默认)vs 一律 A2.7 Paused?
2. §4.8 不变量门 release 姿态:编译期剔除(本稿默认)vs probes 常开低采样?
3. DeviceRes 走 dyn trait(本稿,灵活性优先)vs 枚举静态分发(C2 纯度优先)?
4. M5 收编序:cublas 先(最简单,练手)还是 FI 先(与影子池修复案合并施工)?

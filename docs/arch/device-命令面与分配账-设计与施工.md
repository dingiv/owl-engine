# DeviceHandle:分配账 + 命令面(设计与施工)· 2026-10-19

> 用户定调(三条,同一层):
> ① **收编所有显存申请** —— engine/models 不许再手动碰分配;但 **store 是纯状态机(不做 I/O)**,
>    所以"摸显存"的只能是**客户端侧**的咽喉;钩子**挂在客户端**,不许挂到 server(跨线程)。
> ② 客户端的 device client interface 依赖一个**接口**,该接口**由 store 实现**(方案甲:client 直持 `Rc<RefCell<dyn AllocLedger>>`)。
> ③ 新增**命令批发送**:`DeviceHandle` 提供 `alloc!`/`free!` **宏**、`exec()` 单条、`batch()` 多条;
>    新增 `Command` 枚举 + **链式构造**,如 `Command::Launch(..).args(..)`。
>
> 纪律(不变):**iface 包一层,`cuda` 里实现的 `DeviceClient` 一行不改**。

---

## §一 现状对账(设计的地基;全部实测过)

| 事实 | 位置 | 判读 |
|---|---|---|
| **client 已是唯一咽喉** | `owl-cuda` 的 `GpuClient`;engine/models/kernels/server 都直接依赖 `owl-cuda` | 收编点就在这里,**不必新增 crate** |
| **分配已带点名** | `models/interpreters/eval.rs`:`face.alloc(dtype, n_elems, **name**)`("全量记名纪律") | 归属只需再加一个维度,**调用点不必改** |
| **server 侧已有**:slab 切块 / 归池 / 块账房 | `cuda/server/mod.rs`:`handle_alloc_inner`(捕获期 `carve_block(n_bytes,name)`)、`handle_free`→`free_blocks`(**归池 = 复用已存在**)、`trace_alloc_done`/`block_ledger_dump`(`/debug/blocks`) | 复用与账的**机制**都在;**缺的是"事件导出 + owner 归属"** |
| **engine 直 poke 规模** | 83 处 `alloc/free/htod/dtoh`;52 处 `bank.rel.{leg,state}.*`(变更面 ~10) | 迁移是**有界**的 |
| **store 已具备账本** | `TierLedger`(三层 × 类别 + **owner 维**,2026-10-19 第 0/0.5 刀已落) | store 只差"被喂事件" |
| **引擎单线程** | 引擎 actor = `current_thread` 运行时(收件与泵不相交) | ⇒ `Rc/RefCell` 足够,**无锁、无跨线程** |
| **批发送的真实需求** | prefill 线 M0"命令批/零回执"(每 chunk ~250ms 宿主税) | `batch()` 的直接动机 |

**⇒ 结论**:三件事(收编分配 / 记账 / 批发送)**同层同点** —— 都在 `DeviceHandle` 这一个新面上落。

---

## §二 分层设计(一张图)

```text
                    owl-iface(契约层;新三件都在这里)
        ┌───────────────────────────────────────────────────────────┐
        │ trait DeviceClient        ← 现有;owl-cuda 实现(**不改**)  │
        │ trait AllocLedger         ← 新:on_alloc / on_free / on_move │
        │ enum  Command             ← 新:统一命令面(+ 链式构造)      │
        │ struct DeviceHandle<D>    ← 新:薄转发 + 钩子 + exec/batch  │
        │   inner:  D                                                 │
        │   ledger: Rc<RefCell<dyn AllocLedger>>                      │
        │   owner:  Owner(节点设置)                                   │
        └───────────────────────────────────────────────────────────┘
                 ▲                                    ▲
                 │ impl AllocLedger                   │ 使用
        ┌────────┴─────────┐              ┌───────────┴────────────┐
        │ owl-store        │              │ owl-engine / owl-models│
        │ TierLedger(纯账) │              │ 只见 DeviceHandle      │
        └──────────────────┘              └────────────────────────┘
```

**依赖方向**(单向,无环):
`iface ← {cuda, cpu, store, engine, models}`;`store` 只实现 iface 的 trait(**不依赖 cuda**);
`engine` 同时持 client 与 store ⇒ **接线在 engine**(boot)。

**三条律**(与既有架构一致):
1. **store 纯**:它只被调用(`on_alloc/on_free` 改自己的账),**零 I/O、零句柄**;
2. **不跨线程**:钩子在 **client 侧**(调用者线程 = 引擎线程)发;**server 线程完全不碰 store**;
3. **咽喉唯一**:engine/models 只见 `DeviceHandle`;`DeviceClient::alloc/free` 收窄为"只有 iface 能调"(§六 封印)。

---

## §三 `AllocLedger`:账本接口(store 实现)

**落点**:`crates/backends/iface/src/contract.rs` 是 `DeviceClient` 的家
(`pub trait DeviceClient: Send`;方法为 RPITIT 形 `fn alloc(&mut self, dtype, elems, name) -> impl Future<Output = Result<Bytes, ModelError>> + Send` —— 已有 `name` 必填"全量记名"律)。
本设计新增件建议单开一个模块 **`crates/backends/iface/src/handle.rs`**(trait + 事件 + `Command` + `DeviceHandle`),
由 `lib.rs` 重导出;`contract.rs` **一行不改**。

```rust
// iface/handle.rs
/// 资产种类(与 store 的 AssetKind 同构;iface 不依赖 store ⇒ 此处用中性名)
pub enum AssetClass { KvBlock, Floor, Elastic, Snap, Row, Meta, Index, Temp }

pub struct AllocEvent {
    pub id: u64,          // 设备块 id(server 块账房同源)
    pub bytes: usize,
    pub name: String,     // 调用点点名(models 已有)
    pub class: AssetClass,
    pub owner: Owner,     // 见 §五:由 DeviceHandle 的 owner 上下文填
}
pub struct FreeEvent  { pub id: u64, pub bytes: usize }
pub struct MoveEvent  { pub id: u64, pub bytes: usize, pub dir: MoveDir }  // Htod/Dtoh/SwapOut/SwapIn

/// **账本接口**(由 store 实现;client 只依赖它 —— 依赖倒置)
pub trait AllocLedger {
    fn on_alloc(&mut self, ev: &AllocEvent);
    fn on_free(&mut self, ev: &FreeEvent);
    fn on_move(&mut self, ev: &MoveEvent);
}
```

**store 侧实现**(方案甲):
```rust
// store
impl AllocLedger for TierLedger {
    fn on_alloc(&mut self, ev: &AllocEvent) {
        // 出借登记 + 入账(owner 维);class/owner 直落 TierLedger
        let _ = self.add_owned(ev.owner, Tier::Gpu, ev.class.into(), ev.bytes);
    }
    fn on_free(&mut self, ev: &FreeEvent) { /* remove_owned(按登记反查) */ }
    fn on_move(&mut self, ev: &MoveEvent) {
        // **换层 ≠ 归还**:SwapOut = Gpu→Host(改 tier,不注销);SwapIn 反之
    }
}
```
`Rel.ledger` 由 `TierLedger` 变 **`Rc<RefCell<TierLedger>>`**(约 10 处 `borrow_mut()`;`report()` 也走同一把 Rc ⇒ 两边看同一本账)。

**三律(写进 trait 文档)**:
1. **下沉 ≠ 归还**:`MoveEvent` 只换层;只有 `FreeEvent`/逐出才注销;
2. **双持与幽灵归还响亮拒**:`on_alloc` 遇到同 id 已登记 = `StoreError::Invariant`;`on_free` 未登记同此(不许静默吞);
3. **失败必留痕**:`on_*` 的失败**不许丢**(store 侧记 `ledger.hook_fail` 计数 + 一条 WARN;记账失败不得中断主流程,但**必须可见**)。

---

## §四 `DeviceHandle`:薄转发 + 钩子 + 命令面

```rust
// iface
pub struct DeviceHandle<D: DeviceClient> {
    inner: D,
    ledger: Option<Rc<RefCell<dyn AllocLedger>>>,
    owner: Owner,                  // 当前归属上下文(节点设置)
    pending: Vec<AllocEvent>,      // 批内的延迟入账(见 §五.4)
}

impl<D: DeviceClient> DeviceHandle<D> {
    /// 构造(唯一入口;ledger = None ⇒ 纯转发 = 零行为变化的迁移态)
    pub fn new(inner: D, ledger: Option<Rc<RefCell<dyn AllocLedger>>>) -> Self;

    /// **归属上下文**(N1/N4/N5 调;同线程、无锁)
    pub fn set_owner(&mut self, o: Owner);

    /// 单条命令(钩子在此发:命令 → 事件)
    pub async fn exec(&mut self, cmd: Command) -> Result<Outcome, ModelError>;

    /// **命令批**:一次投递 + 一次收口(钩子逐条入账,但只一次往返)
    pub async fn batch(&mut self, cmds: impl IntoIterator<Item = Command>)
        -> Result<BatchOutcome, ModelError>;

    /// 只读直通(不记账;查询类不走命令面)
    pub fn client(&self) -> &D;
    pub fn client_mut(&mut self) -> &mut D;   // 仅限"不分配"的用途;审计见 §六
}

/// 批的结果(逐条)
pub struct BatchOutcome {
    pub outcomes: Vec<Result<Outcome, ModelError>>,
    /// 首个失败的**批内下标**(None = 全成)
    pub first_err: Option<usize>,
    /// 实际投递段数(打点:批 vs 逐条的提交税对比)
    pub segments: usize,
}
```

**宏**(调用点短;自动带 owner + 自动记账):
```rust
#[macro_export] macro_rules! alloc {
    ($dev:expr, $cls:ident, $elems:expr, $name:expr) => {
        $dev.exec($crate::Command::alloc($cls, $elems, $name)).await
    };
}
#[macro_export] macro_rules! free { ($dev:expr, $id:expr) => { $dev.exec($crate::Command::free($id)).await }; }
```
> 宏只做"构造 + exec",**不隐藏错误**(返回 `Result`,与全仓"不许吞返回值"同律);
> 之所以用宏而不是函数:`$cls`/`$name` 这类**编译期字面量**要进事件(便于 grep/打点),且调用点更短。

---

## §五 `Command`:统一命令面 + 链式构造

```rust
// iface
pub enum Command {
    Alloc  { class: AssetClass, dtype: Dtype, elems: usize, name: String, zero: bool },
    Free   { id: u64 },
    Htod   { id: u64, src: HostRef, off: usize, bytes: usize },
    Dtoh   { id: u64, dst: HostRef, off: usize, bytes: usize },
    Launch { kernel: KernelRef, grid: Dim3, block: Dim3, args: ArgPack, stream: StreamId },
    Swap   { dir: SwapDir, pieces: Vec<Piece>, table: SlotTable },
    Sync   { stream: StreamId },
    Barrier{ streams: Vec<StreamId> },
}

impl Command {
    // **链式声明**(用户要的语法:`Command::Launch(..).args(..).grid(..)`)
    pub fn args(self, a: ArgPack) -> Self;
    pub fn grid(self, g: Dim3) -> Self;
    pub fn block(self, b: Dim3) -> Self;
    pub fn stream(self, s: StreamId) -> Self;
    pub fn name(self, n: &str) -> Self;
    pub fn zero(self, z: bool) -> Self;
    // 构造器糖(宏用)
    pub fn alloc(class: AssetClass, elems: usize, name: &str) -> Self;
    pub fn free(id: u64) -> Self;
}
```

### 5.1 语义(必须钉住)

| 面 | 语义 |
|---|---|
| **顺序** | 批内 **FIFO**,同流保证序;跨流要显式 `stream(..)`(不许隐式跨流重排) |
| **错误** | 批内第 k 条失败 ⇒ `first_err = Some(k)`,**已投递的不回滚**(命令流语义);账**如实**记(失败条不发 `on_alloc`) |
| **收口** | `batch` = **一次投递 + 一次收口**(`segments` 打点);`exec` = `batch([cmd])` 的特例(实现单源) |
| **钩子** | 事件在**客户端侧**逐条生成;`on_alloc` 在**成功收口后**才入账(失败不入账) |
| **回执** | 两阶段:①**语义等价批**(逐条转发 + 聚合 await,后端零改动)②**原生批**(后端把 N 条合成一次 ack) |

### 5.2 为什么"批"值钱(与 prefill 线 M0 对齐)

现状:每条命令一次 `await`(投递 + ack 往返)⇒ prefill 每 chunk ~250ms 的**宿主税**。
`batch()` 把 N 条合成一次往返 ⇒ **提交税 ÷N**;而**执行面完全不变**(仍是同一串命令)——
⇒ 验收口径 = **token 逐位恒等门**(结果不变)+ `segments`/`srv.seg.*` 计数下降。

### 5.3 与既有 swap 路径的关系

`apply_verdict` 现在自己发 `face.swap(...)`(engine 编排搬运)—— 迁移后改成
`dev.batch([Command::Swap{..}, ..])`,**编排仍在 engine**(§21 裁决不变),只是出口统一到命令面。

---

## §六 封印:让"绕过"成为编译错误(而不是纪律)

| 手段 | 内容 |
|---|---|
| **类型封印** | `DeviceClient` 的 `alloc/free` 增加 `permit: AllocPermit` 参数;**`AllocPermit` 只能由 iface 构造**(`pub(crate)` 构造器)⇒ engine/models 结构上无法直调 |
| **依赖收窄** | `engine`/`models` 的 `Cargo.toml` 去掉 `owl-cuda`,只依赖 iface(经 handle)+ arena 语义 |
| **审计** | CI 加规则:`crates/{engine,models}` 出现 `\.alloc\(|\.free\(` 直调 = 红(白名单:handle 内部/测试) |
| **只读例外** | 查询类(`slots_total`/`block_bytes`/`stats`)不封印(不产生分配) |

---

## §七 接线与生命周期节点

```rust
// engine boot(loader.rs)
let ledger: Rc<RefCell<dyn AllocLedger>> = Rc::new(RefCell::new(store_ledger));
let dev = DeviceHandle::new(gpu_client, Some(ledger.clone()));
// store 侧 Rel.ledger 持同一个 Rc ⇒ 一本账

// engine 主循环(节点设 owner)
N1 TurnBegin   → dev.set_owner(Owner::Turn(t));
N5 SessionOpen → dev.set_owner(Owner::Session(s));
N4 TurnEnd     → dev.set_owner(Owner::System);      // 复位;无上下文时的分配记 System + 计数(不静默)
```
**归属规则**(与 §三 一致):

| 资产 | owner | 来源 |
|---|---|---|
| KV 链 | `Session(s)` | 会话级(跨 turn) |
| 快照 Durable / Transient | `Cache` / `Turn(t)` | G1 的 `SnapDurability` 已在手 |
| GDN 行 | `Turn(t)` | 行借给活跃 turn |
| 临时工作区/激活面 | `Turn(t)` | **本设计的主要新增可见面** |
| 权重/图 slab/池骨架 | `System` | 常驻 |

---

## §八 不变量(响亮拒清单)

1. 每块**唯一 owner + 唯一 tier**;重复出借 / 幽灵归还 = `Invariant`;
2. `MoveEvent` **不改 owner**(只换层);`FreeEvent` 才注销;
3. 批:失败条**不发** `on_alloc`;已成功的条**必须**有事件(否则账漏);
4. 钩子失败**不中断主流程**,但必须计数 + 一条 WARN(`ledger.hook_fail`);
5. `exec/batch` 的返回**不许丢**(全仓律 #1);批的 `first_err` 必须由调用方处置或显式记日志。

---

## §九 施工序(5 步,每步可编译、可停、可验收)

| 步 | 文件 | 内容 | 判据 |
|---|---|---|---|
| **1** | `crates/backends/iface/src/handle.rs`(新)+ `lib.rs` 重导出 | `AssetClass`/`AllocEvent`/`FreeEvent`/`MoveEvent` + `trait AllocLedger` + `Command` + `DeviceHandle`(薄转发 + 钩子 + `exec/batch` + 宏) | 单测:钩子事件序列正确;`batch` 的 `first_err` 语义;**cuda 零改动** |
| **2** | `crates/store/src/{tier.rs,rel.rs}` | `impl AllocLedger for TierLedger`;`Rel.ledger` → `Rc<RefCell<TierLedger>>`(约 10 处) | 单测:alloc→free 账平;换层不改 owner;幽灵归还拒 |
| **3** | `crates/engine/src/{loader.rs,running.rs}` | boot 接线 + N1/N4/N5 设 owner(先只记账,**零行为变化**) | `/debug/vram` 出现 `turn:N`/`system` 临时面字节,且与块账房对齐 |
| **4** | `crates/models`(face 类型) | `GpuClient` → `DeviceHandle<GpuClient>`(**调用点签名不变**) | 全量单测绿;`/debug/vram` 的临时面归属正确 |
| **5** | 后端(可选) | **原生批**(N 条合成一次 ack)+ 封印(§六) | `segments`/`srv.seg.*` 下降 + **token 逐位恒等门**;审计 grep 归零 |

---

## §十 风险与未决

1. **`Rc<RefCell>` 借用纪律**:`on_*` 期间不得再借同一本账(`borrow_mut` 重入 = panic)⇒ 钩子内**只改账**,不做任何回调(写进 trait 文档 + 单测覆盖重入);
2. **批的原生实现**:后端需支持"一条命令 N 段"或"多命令一次 ack"——先做等价批,原生批按 `srv.seg.*` 收益定档;
3. **跨流顺序**:批内混流是**显式**的;隐式重排 = 不许(与既有流纪律一致);
4. **models 的 face 类型替换面**:`face` 出现在 interpreters 多处;类型换名是机械替换,但**测试替身**(CPU face)要同步实现 `DeviceClient` 即可(无需改测试语义);
5. **封印的时机**:步 5 才做(步 1-4 期间允许并存,便于灰度对比)。

---

## §十二 步 1 实施记(架子已搭;2026-10-19)

### 12.1 已落

| 件 | 内容 |
|---|---|
| `crates/backends/iface/src/handle.rs`(**新**) | `Owner`(带 `label()/default_owner()`)/`AssetClass`(8 类)/`AllocEvent·FreeEvent·MoveEvent`/`MoveDir`/`trait AllocLedger`/`Command`(+`grid/block/shared_mem/args/name/kind` 链式)/`Outcome`/`BatchOutcome`/`DeviceHandle<D>`(薄转发 + 钩子 + `set_owner`/`hook_fail` + `exec/batch`) |
| `crates/backends/iface/src/lib.rs` | 挂 `pub mod handle` |
| `crates/backends/iface/src/contract.rs` | **仅一处改名**:裸 client 的 3 参 `alloc!` → **`client_alloc!`**(`alloc!` 让位给 DeviceHandle;8 个 cuda tests + 4 个 models 文件随改) |
| `crates/backends/cpu/tests/device_handle.rs`(**新**) | 行为门 7 条(真实 `CpuFace` 作内层 client) |

### 12.2 实测门(全绿)

| 门 | 判据 |
|---|---|
| ① `exec` 钩子 | owner/class/name/bytes/id **全对**(bytes = 元素 × `dtype.size_bytes()`) |
| ② `free` | `FreeEvent.id` 与分配同源;owner 随节点切换 |
| ③ 批 `first_err` | 失败条**不入账**、成功条必入账(`first_err=Some(1)`、`segments=3`) |
| ④ 未接线命令 | `Htod/Dtoh` **响亮拒**(信息含"尚未接线") |
| ⑤ 纯转发态 | `ledger=None` ⇒ 零记账、`hook_fail=0` |
| ⑥ 批 `Sync` | 逐条成功 + 一次收口 |
| ⑦ `Launch` 转发 | 失败信息来自 **client**(非"尚未接线")⇒ 确已转发 |

`owl-iface` **9/9** · `owl-cpu` **11/11**(含 7 行为门)· `owl-models` **217/217** · `server` **46/46** · `cargo check --workspace --all-targets` **干净**。

### 12.3 三条实施教训(值得记)

1. **`alloc!` 归位**:用户裁定 `alloc!` 属 **DeviceHandle**(5 参:class/dtype/elems/语义名),
   裸 client 的 3 参形式改名 `client_alloc!` —— 两形状并存但名字不撞,谁都不被静默劫持;
2. **`Launch` 也产出块**(契约里 `launch` 返回 `Bytes`)⇒ `Outcome::Alloc` 改名 **`Outcome::Buf`**
   (alloc/alloc_uninit/launch 统一);**丢返回值 = 违反律 #1**(骨架期即被测试抓出);
3. **dev-dep 成环会让 crate 编译两份** ⇒ `CpuFace` 实现的是**另一实例**的 `DeviceClient`
   (trait 不通用,`the trait bound CpuFace: DeviceClient is not satisfied`)。故行为测试放
   `crates/backends/cpu/tests/`(owl-iface 在那里是普通依赖,同一实例);iface 自身只留纯单测。

### 12.5 步 2 实施记(store 侧实现 `AllocLedger`;2026-10-19)

**已落**:

| 件 | 内容 |
|---|---|
| `crates/store/src/tier.rs` | ① **`Owner` 改为复用 `owl_iface::handle::Owner`**(删本地定义 —— 单一真源,避免"两套接口");② `AssetKind += Temp` + `impl From<AssetClass> for AssetKind`(单点映射);③ **`TierLedger::move_owned(owner, kind, from, to, bytes)`**(换层:owner/entries 不变,源层无账/同层/超账**响亮拒**);④ **`LedgerHandle`**(`Rc<RefCell<TierLedger>>` + **出借登记** `id → (owner, kind, tier, bytes)` + `hook_fail`)实现 `AllocLedger` |
| `crates/store/src/rel.rs` | `ledger: TierLedger` → **`Rc<RefCell<TierLedger>>`**(3 处使用点加 `borrow()/borrow_mut()`)+ `Rel::ledger_handle()`(给 boot 接线) |
| `crates/store/src/lib.rs` | 无需改(`tier::*` 已导出 `Owner`/`LedgerHandle` 经 `pub use tier::{..}`? —— 见下注) |

**三门(全绿)**:

| 门 | 判据 |
|---|---|
| ① 同一本账 | `on_alloc` 后 store 自己立刻看到(`kind_bytes`/`owner_bytes` 同值);`on_free` 按**登记字节**扣(事件只带 id)⇒ 账平、登记清空 |
| ② 双持/幽灵 | 同 id 重复 alloc / 未登记 free **计数 + 不 panic**(`hook_fail` 1→2),合法条目仍在场 |
| ③ 换层 ≠ 归还 | `MoveDir::Out`:设备层清、宿主层在位、**owner 不变**、**entries 跟着走(不多出一条)**;方向不符**响亮拒** |

`owl-store` **86/86** · `owl-engine` **54/54** · `server` **46/46** · `cargo check --workspace --all-targets` **干净**。

**设计澄清(重要)**:事件账(`LedgerHandle`)与**腿自报对账**(§8.3 的 `leg_accounting`)是**两个独立来源**——
前者是"出借/归还"的**事件流**,后者是"腿里此刻真的躺着多少"。二者**互相校验**:
步 3 接线后,`/debug/vram` 同时给两面,**对不上就是 bug**(这是比"单一来源"更强的判据)。
因此 `report()` 暂**不**合并事件账(避免双计);步 3 视接线情况再决定是否切换主账源。

### 12.4 下一步(步 3/4/5)

步 2:`impl AllocLedger for TierLedger` + `Rel.ledger → Rc<RefCell<..>>`;
步 3:boot 接线 + N1/N4/N5 设 owner(零行为变化);
步 4:`face` 类型替换;步 5:原生批 + 封印(§六)。

## §十一 变更记录

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-10-19 | v0.3 | **步 2 完成**:store 侧 `impl AllocLedger`(`LedgerHandle`:出借登记 + 双持/幽灵拒 + 换层不注销)+ `TierLedger::move_owned` + `Owner` 复用 iface(单一真源)+ `Rel.ledger → Rc<RefCell<..>>`;三门全绿;澄清"事件账 vs 腿自报"互为校验 |
| 2026-10-19 | v0.2 | **步 1 完成**:`iface/handle.rs` 落地(Owner/AssetClass/三事件/AllocLedger/Command+链式/Outcome/BatchOutcome/DeviceHandle + `alloc!`·`free!` 宏);`contract.rs` 仅把裸 client 的 `alloc!` 改名 `client_alloc!`;行为门 7 条全绿(真实 CpuFace 作内层);三条教训入档(宏归位 / `Outcome::Buf` / dev-dep 成环双实例) |
| 2026-10-19 | v0.1 | 立项:三事同层(收编分配 / 账本接口 / 命令批发送);方案甲(`Rc<RefCell<dyn AllocLedger>>`)+ iface 薄包装(cuda 零改动);`Command` 链式构造 + `exec/batch` + `alloc!/free!` 宏;5 步施工序与判据 |

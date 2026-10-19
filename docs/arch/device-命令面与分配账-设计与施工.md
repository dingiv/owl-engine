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


### 12.6 步 3 实施记(全出生/死亡钩子 + 图窗 + 相位上下文;2026-10-19)

**用户裁决(四条)**:① 账本原语改 `Arc<Mutex>`(真 `Send`);② 词表扩 4 类 + 相位上下文;
③ 覆盖口径 = **全出生/死亡 + 全等门**(事件账 ids ↔ 块账房 ids);④ 先落步 1+2 提交。

#### (1) 账本原语:`Rc<RefCell>` → `Arc<Mutex>`(方案甲字面被**硬约束**推翻)

`DeviceClient: Send` 是契约硬要求,而 `/debug/blocks` 的 `face_clone()` **真的**把 face
句柄送进 HTTP 线程(`main.rs` 的 `DbgFace = Arc<Mutex<GpuClient>>` 槽)—— `Rc` 在那条路径上
是**真的 UB**,不是理论问题。落地:

| 件 | 内容 |
|---|---|
| `iface/handle.rs` | `pub type LedgerCell = Arc<Mutex<dyn AllocLedger>>`(trait 加 `Send` 超界);`DeviceHandle.ctx: Arc<Mutex<Ctx>>`(**共享上下文**:face_clone 出来的句柄与本体同源 ⇒ 节点设一次全族生效) |
| `store/tier.rs` | `LedgerBook { tiers, open, hook_fail }`(三层账 + 出借登记**同屋**:一笔出借同时改两处,分两处必分叉)+ `SharedLedger` + `book_lock()`(**锁中毒不许 panic 级联**:取回内部数据 + 计数可见)+ `LedgerEntry`(`AllocLedger` 的 store 适配器)+ `shared_ledger()`/`ledger_entry()`(boot 唯一入口) |
| `store/rel.rs` | `Rel.ledger: SharedLedger`;`Rel::open(.., ledger)` **交叉校验**账本预算与 `VramBudget::tier_budget()` 逐字相等(不符 = boot 算式漂了 ⇒ 响亮拒) |
| `store/lib.rs` + `engine` | `tier_budget_of(host_cache_mb, snap_host_mb)` = **中层预算唯一算式**(boot 造账与 `BudgetInputs.host_bytes` 同源) |

两层锁的分工:**外层** = 事件入口对象(`LedgerCell`),**内层** = 账本本体(store 自查也要它);
两层都只在同步段短暂持有、绝不跨 await、持内层时不回调外部代码。

#### (2) 词表扩 4 类 + 相位上下文

`AssetClass`/`AssetKind` 各 8 → **12**:`Weight`(装载)/`Graph`(图 slab)/`Pool`(池骨架)/
`Unclassified`(兜底,**可见**)。映射单射(store 单测钉住)。相位上下文用 **RAII**(`CtxScope`,
出作用域自动还原 —— 相位里任何 `?` 早退都不会漏还原):

| 相位 | 设置点 | owner / class |
|---|---|---|
| 装载 | `loader.rs` 两个 `load_*` 入口 `scoped_ctx` | `System` / `Weight` |
| 池骨架 | `boot_rank` 的 `StatePool::alloc` 外包一层 | `System` / `Pool` |
| 捕获窗 | **自动**:窗内出生 ⇒ `Graph`(机制事实:捕获期后端恒走 slab 切片) | 窗内覆盖 |
| 执行 | N1 设 `Turn(t)` / `Temp`;N4 落 `Session(s)`(会话仍活)或 `System`(焚毁)+ `Unclassified` | 见 §七 |

上下文面同时进 `DeviceClient` **契约**(带缺省空实现:`set_owner/set_class/scoped_ctx/ledger_probe`
—— 引擎内核按 `D: DeviceClient` 泛型写,拿不到具名句柄类型;裸 face 缺省 = 空实现,
不是静默失败:无账本 ⇒ 无归属可谈)。

#### (3) 覆盖口径:全出生/死亡 + 图窗(**契约缺口用"后端自述"补**)

- **出生** = `alloc` / `alloc_uninit` / `htod` / `htod_on` / `htod_f32`(手心:字节 = 元素×dtype
  / 源字节);**死亡** = `free`(成功收口才销);**换层** = `swap`(片段按逻辑块**聚合**成一条
  `MoveEvent`;部分搬运由账本响亮拒);
- **`launch` 不算出生** —— 实测 `cuda/src/launch.rs`:输出块 = 实参里最后一个 `Block`
  (调用方先 `alloc` 再发射,服务端零分配)⇒ 出生只发生在分配族(这条推翻了我方文档原先的
  "Launch 也产出块"的措辞);
- **图窗**:`graph_begin` 开窗 → 窗内出生块记账 → `graph_end` 成功挂到 `GraphId` 名下
  → `graph_destroy` 补发归还;
- **契约缺口(新发现)**:窗失败/退休时"那些块去哪了"**两档后端语义真的不同** ——
  CUDA 整批回收(`carve_block` ⇒ `graph_destroy`/空窗失败移出账房),host mock 交调用方 `free`
  (CpuFace 注释原文)。**猜 = 静默错账** ⇒ 新增后端自述
  [`DeviceClient::window_blocks_auto_reclaimed`](crate/backends/iface/src/contract.rs)(默认 false,
  CUDA 覆写 true;契约三条:abort 成功 / destroy 成功 / **空窗失败** 都整批回收)。
  `DeviceHandle` 按自述镜像:自述回收的场合补发事件;不回收/去向不明的场合**保留窗与 id**
  并计 `window_unclosed`(可见),由调用方显式 `free` 收。

#### (4) 报告面:`/debug/vram` 三面同刻

`VramSnapshot { report, blocks(逐 rank 块账房快照), hook_fail, window_unclosed, open_window_blocks }`
——**同一 actor turn 内取三面**(引擎不在 pump ⇒ 无交错,差集才是判据)。JSON 新增:

- `events[]`(rank × owner × tier × kind;`EventRow`)、`handle{}`(`hook_fail`/`window_unclosed`)、
- `recon[]`(**全等门**):逐 rank 比事件账 id 集 ↔ 块账房 id 集,给出 `equal`/`only_event`/
  `only_block` + 前 16 个差集 id(纯比较,零决策)。

#### (5) 实测门(全绿)

| 门 | 判据 | 结果 |
|---|---|---|
| 单测 | iface 11 · store 87(+3+3)· cpu 16(行为门 16 条:含窗/自述/接管/同源)· models 217 · engine 54 · server 46(+1+2) | ✅ |
| `cargo check --workspace --all-targets` | — | ✅ 干净(仅 2 条 models 存量 warning) |
| **CPU mock 集成门**(新 `crates/engine/tests/device_ledger.rs`) | 装载 ⇒ `weight` 类 1.43 GiB / 304 条;装配 ⇒ `pool` 237 MiB / 161 条 + `graph` 485 KiB / 2 条;turn 后 `hook_fail=0` `window_unclosed=0` | ✅ |
| 真机 27B(见 §12.7) | `recon.equal` + owner 曲线 | 待跑 |

**三条实施教训**(值得记):
1. **上下文读必须落变量**:`self.note_birth(&b, bytes, self.ctx().class, ..)` 里实参的临时
   guard 活到**语句末**,而 `note_birth` 内部还要锁同一把 ⇒ **自锁死**(std Mutex 不可重入)。
   同理 `match self.ctx().windows.pop()` 的 scrutinee guard 活过整个 `match`
   ⇒ 分支里再锁 = 死锁。两处都实测踩到(make(1) 挂死),已改成"先落变量再调用"并写进注释。
2. **`launch` 不产块**(见上):文档措辞与代码事实不符时,**以代码事实重写文档**。
3. **`Engine::on_with_ledger` / `from_handle`**:TP 的 staging 握手要把 face 借出来再装回去,
   而句柄承载账本身份 ⇒ 拆装只能"接着用"不能"重包"(重包 = 双包 `DeviceHandle<DeviceHandle<..>>`
   + 上下文丢失)。TP 两 rank **合流同一本账**(`adopt_ledger`,仅 boot 期零事件;rank1 自己
   那本必须为空 = 合流自证),事件带 `rank` 维区分卡。


### 12.7 步 3 真机实测(27B AWQ INT4;2026-10-19)

**档位**:`--flavor spec-dflash`(depth 7)· 3090 Ti(CUDA ordinal 1)· ctx 16384 / pool 16896 /
gdn_slots 1 / `prefill_bucket = off`(临时档 `/tmp/owl-recon-27b.toml`,不入库)·
**引擎装配(run)20.29s** · 启动即 `捕获态 Captured`。

#### 事件账(启动后 + 1 轮请求)

| rank | owner | kind | bytes | 条数 | 判读 |
|---|---|---|---|---|---|
| 0 | system | **Weight** | 16 652 MiB | 2217 | 27B AWQ INT4 权重,量级对(相位上下文生效) |
| 0 | system | **Pool** | 1 772.8 MiB | 630 | 池骨架(KV/GDN/快照/草稿) |
| 0 | system | **Unclassified** | 1 287.5 MiB | 254 | **未接线相位**(草稿模型装载等)——迁移进度指标 |
| 0 | system | **Graph** | 61.9 MiB | 1749 | 捕获窗自产块(图族 slab;随图存活) |
| 0 | **turn:1** | Temp | 0.02 MiB | 27 | N1 生效:turn 域临时面**可数** |

`handle`: **`hook_fail = 0`** · `window_unclosed = 0` · `open_window_blocks = 1749`(图 slab 自产块,待图退休)。

#### 全等门(`recon[0]`)

```
blocks = 4895    events = 4877    only_event = 0    only_block = 18
block_bytes = 20 735 053 308 B   events.bytes = 19 774.3 MiB(≈ 同值)
only_block_ids = [1099511627777 … +17]   ⇒ 账本 id 1 的前 18 个块
```

⇒ **18 块 = 服务端装配期自检块**,是**设计内**的账外分配:`GpuCtx::alloc_block` 的自检面
(源码注释原文:*"装配期一次性小额…不计 slab/vram 音账"*,`cuda/src/server/mod.rs` §1915 附近)。
⇒ **残差被解释干净**:36 个字节级差值之外**逐块全等**(id 集 + 字节),且**不随轮次增长**。

#### 逐轮曲线(泄漏判据:4 轮同档请求)

| 轮 | `turn:*` 在借条目 | `turn:*` 字节 | 全局 entries | 全局字节 | recon `only_block` |
|---|---|---|---|---|---|
| 1 | 27 | 0.020 MiB | 4877 | 19774.3 MiB | 18 |
| 2 | 34 (**+7**) | 0.033 MiB | 4884 | 19774.3 MiB | 18 |
| 3 | 41 (**+7**) | 0.050 MiB | 4891 | 19774.3 MiB | 18 |
| 4 | 48 (**+7**) | 0.067 MiB | 4898 | 19774.3 MiB | 18 |

⇒ **"未归还"第一次可数到块**:每 turn **+7 块 / +0.017 MiB**(`turn:N` 桶,内容极小但**条目线性增长**)
= store 立案 §缺口 1「N4 无条件归还清单」缺位的**实测指纹**(此前只有"逐 turn 爬坡 ~480 MB"这种
总量口径,看不见是谁)。**N4 清单 = 下一刀**,判据就写在这里:该列必须**不再随轮增长**。

⇒ 另注:事件账面**看不到 480 MB/turn 量级的增长** ⇒ 那部分若复现,不在"设备块"这一层
(属于腿/池的**复用缓存**面或视窗口径),与本节不冲突 —— 但**说明"爬坡"必须两面一起看**
(事件账 = 块面;腿自报 = 池面),这正是 `/debug/vram` 三面同刻的理由。


### 12.8 刀 N4-a:副作用哑根治理(7 块/轮 → 2 块/轮;2026-10-19)

**侦查口径**:`/debug/vram` 新增 `events.open[]`(**lifecycle 在借清单**,逐条带
`id/name/owner/bytes`)—— 这就是 N4「未归还清单」的原料。27B 真机 3 轮实测:

| 名字 | 条数/轮 | 字节 | 判读 |
|---|---|---|---|
| `vllm_reshape_and_cache_bf16` | **5**(turn:1 额外多 15,共 20) | ≈0 KiB | DFlash2 草稿 **5 层**的 KV 写:核就地写池,末位出参只是**契约 4 占位**(`with_shape(dt, vec![1])` = 1 元素哑根) |
| `owl_dflash_select_bf16` | 1 | 7 KiB | propose 路选择器输出 |
| `owl_narrow_strided_f16` | 1 | 10 KiB | propose 路 narrow(树序/视图) |

**根因**:`eval_ops_multi_scoped_env` 的收尾口径 = `候选 = 竞技场 − 根` ⇒ **副作用哑根
(只为触发执行而传的"根")永久保留**。DFlash2 encode 每层一个 ⇒ 5 块/轮,`turn:N` 列
逐轮爬坡的主因(它**不是** 480MB 量级的那种泄漏,是**条目**泄漏)。

**修法(源头上分口径,不猜不扫)**:
- `interpreters/eval.rs` 新增 **`eval_ops_side_effect_env`**(与 scoped 同一实现体,
  `keep_roots = false` ⇒ **根也在候选里**);契约写明:**调用方不得在返回后再读这些块**;
- 换口三处(全部是 `encode_kv` 的哑根):`exec/draft.rs` 两处 + 后一处(`prefill_dflash_encode`
  的第二个 encode 站点 —— 第一轮只改了其中一处,实测**没变**,正是"同一函数两处调用点"的坑)。

**实测(27B,3090 Ti,同档 3 轮)**:

| 轮 | lifecycle 在借 | 内容 |
|---|---|---|
| 1 | **2** 条 / 17.0 KiB | select 1 + narrow 1 |
| 2 | **4** 条 / 34.1 KiB | 每轮 +2(不再是 +7) |
| 3 | **6** 条 / 51.1 KiB | `only_block` 恒 18 · `hook_fail` 0 · `window_unclosed` 0 |

⇒ **每轮 +7 → +2**;两项各自与"轮"一一对应,`recon` 残差不变。

**遗留(下一刀 N4-b)**:`owl_dflash_select_bf16` + `owl_narrow_strided_f16` 各 1/轮。
持有面 = propose 路 `eval_ops_scoped_env(drafts.step())`(单根 scoped:根保留、`arena`
由调用方 `face.free(&arena)` 收)。二者是 `drafts` 的**上游**块,却不在归还的 `arena` 里 ⟹
**待查假设**:它们经 `BlockPool.give_back` 进了池(`= 零命令复用`),而池在函数出口被 drop
而**未 flush** ⇒ 块既不在 arena 也不被 free。**下一步 = 在 `eval_ops_scoped_env` 出口
`flush_pool`(与 scoped 多根同口径)并加门**;或证明它们是"读数据的根"而需显式在轮末归还。


### 12.9 刀 N4-b:根块无条件归还(2 块/轮 → 1 块/轮;2026-10-19)

**取证方式(按"不猜"纪律)**:开 `[cuda] trace_ring`,把在借块的 id 拿到**发射账**里比对 ——
`owl_narrow_strided_f16`:`alloc_done(10240B) → launch owl_narrow_strided_f16 → launch
marlin_gemm_w4a16_awq(吃它) → 此后无 free`;`owl_dflash_select_bf16`:`alloc → select →
**dtoh(引擎回读)** → 此后无 free`。

**定谳(两条)**:
1. `exec/draft.rs::dflash_propose` 的 harvest:根块 `b` = **drafts 的 select 单缓冲**
   (`b` 只是它的 SliceView);`face.dtoh(&b, ..)` 回读后**只还了 arena,没还 `b`** ⇒ 每轮一条。
   修:`let mut ids = arena; ids.push(b.id);` 一并还。
2. `exec/draft.rs::dflash_spec_extend` 的扩展 eval:根块原写作 **`_b`(弃值)** ——
   那正是漏点的形状(没人读、也没人还)。修:`let (b, arena)` + 与 arena 一起还。

**实测(同档 3 轮)**:`1 / 2 / 3` 条(每轮 **+1**,只剩 `owl_narrow_strided_f16` 10 KiB);
`recon only_block` 恒 18 · `hook_fail` 0 · `window_unclosed` 0。
累计:**每轮 +7 → +1**。

**遗留(N4-c)**:`owl_narrow_strided_f16` 1 条/轮。发射账已给出形状:它是草稿 fc 路
(`五拆 marlin:fc_i 整块 GEMM 直吃 tap_i`)里被 GEMM 消费的**中间块**,却没进 scoped 的
`arena` 候选 —— 候选 = `arena − root`;**下一步假设**:该 eval 走的是 plain/multi(块常驻,
为捕获铺路)口径,或 tap 的 narrow 块在 eval 之外构造。取证入口同上(开 trace_ring 看
`alloc_done` 之后有没有进 arena/有没有 free)。


### 12.10 刀 N4-c 立案:最后一个 1 块/轮(10 KiB)的发射账实证(2026-10-19)

**证据(发射账窗口,`trace_ring`;seq 连号即同时序)**:

```
53443 launch  vllm_reshape_and_cache_bf16        out=…421            ← 草稿 KV 写(哑根,N4-a 已治)
53444-53447 free [418,419,420] / free [401,406,411,416,421]          ← 该 eval 的活块正常回收 ✓
53448 alloc   uninit:owl_narrow_strided_f16      10240B
53449 alloc_done                                                 →…422
53450 launch  owl_narrow_strided_f16  ids=[…380(concat_rows),…422] out=…422
53451 alloc   uninit:marlin_gemm_w4a16_awq       496640B
53453 launch  marlin_gemm_w4a16_awq   ids=[…422(A), 7800(权重), …423…] out=7804   ← 422 被当 A 吃
53455/57      alloc/launch owl_argmax_f32idx_f16 (吃 423) → …424      ← top-1 走词表
53458 copy_block  496640  [423 → 32502(常驻块)]                      ← 真结果 copy 进常驻面
53459 free [423]                                                      ← GEMM workspace 回收 ✓
53460 free []                                                         ← 某 eval 的 flush_pool(空)
53461 free [380]                                                      ← concat_rows 的块回收 ✓
（全 trace 里对 …422 的 free:**一条都没有**）
```

**判读(不猜,只登记事实)**:
- …422 的**生产者与消费者都在同一段**(narrow 53450 → GEMM 53453),且其输入 …380(concat_rows)与
  同段其它临时块(…418/419/420/423)都被回收 —— **只有它没进任何一次 free 列表**;
- 因此它不是"没人知道它在哪"(那是 N4-a/N4-b 之前的形状),而是**"知道却不收"**:
  假设 = 该块由 `alloc_uninit` 铸出后被当作**透传/别名臂**处理(块归属按块计数那条路),
  ⇒ `arena_set` 未登记(或 `block_users` 计数没归零),于是滑出 `arena − root` 候选;
- **下一步(刀 N4-c 正刀)**:在 `interpreters/eval.rs` 加一条**收尾不变量检验** ——
  scoped 出口把 `arena_set` 与"本段 `alloc`/`alloc_uninit` 命令数"对账,不等 = 响亮留痕
  (`OWL_LIVE_DIAG` 一路),把"漏登记"从"看不见"变成"每次都报";再按报出的名字定位登记缺口。

**当前进度**:每轮 **+7 → +2 → +1**(22→29→36→43 变成 1→2→3);`recon` 残差恒 18 ·
`hook_fail` 0 · `window_unclosed` 0。


### 12.11 刀 N4-c:未归还清单**上节点**(可见性落地)+ 最后 1 块的机制判死(2026-10-19)

**(1) N4 清单上节点**(store 立案 §缺口1 的第一半,落地):turn 收尾
(`complete`/`abandon` 两路)调 `RunningEngine::n4_report_unreturned(turn)` —— 从账本
读该 turn 名下**仍开着**的块,按名聚合,非空即 **WARN**。实测输出:

```
[WARN] [store] N4 未归还清单:turn 1 名下仍有 1 块 / 10 KiB 开着 —— owl_narrow_strided_f16 ×1 / 10240B
```

⇒ 从此"每轮 +N"**在节点上就报名字**,不必等 `/debug/vram` 的聚合数字、也不必开
`trace_ring` 拼发射账(本次定位两次泄漏都靠后者,代价高)。

**(2) 最后 1 块的机制判死**(复核 + 发射账):`alloc_out` 的三处调用点**都**调了
`track_new`(竞技场登记无缺口)⇒ 排除"漏登记"。真正的形状是**两次 eval**:

```
eval A: narrow(concat_rows 输出) → **A 的根** → 返回给调用方(扫尾按 `arena − root` ⇒ 根留下)
eval B: marlin_gemm_w4a16_awq(A 的返回值当输入)→ argmax → copy_block 进常驻
        扫尾只 free B 自己的 arena([380] 与 [423] 都在列,唯独 [422] 不在)
```

⇒ 与 N4-b **同型**:前一个 eval 的**根**被后一个 eval 当输入消费掉,而**根从未归还**。
现场 = 草稿 select/lm_head 走词表的那条 walk(`concat_rows → narrow → marlin GEMM → argmax → copy`);
**下一刀 = 在 walk 的调用点把这两个中间根(窄块 + GEMM 输出)归还**(或整条 walk 改走
`eval_ops_side_effect_env` 口径)。

## §十一 变更记录

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-10-19 | v0.4 | **步 3 完成**:账本原语 `Arc<Mutex>`(真 Send)+ 词表扩 4 类 + 相位 RAII 上下文 + **全出生/死亡钩子**(含图窗自述契约)+ `/debug/vram` 三面同刻(事件账/句柄计数/**全等门**对账);CPU mock 集成门实测 weight/pool/graph 三类账落地;三条教训入档(自锁死两处 / launch 不产块 / TP 拆装只能接着用) |
| 2026-10-19 | v0.3 | **步 2 完成**:store 侧 `impl AllocLedger`(`LedgerHandle`:出借登记 + 双持/幽灵拒 + 换层不注销)+ `TierLedger::move_owned` + `Owner` 复用 iface(单一真源)+ `Rel.ledger → Rc<RefCell<..>>`;三门全绿;澄清"事件账 vs 腿自报"互为校验 |
| 2026-10-19 | v0.2 | **步 1 完成**:`iface/handle.rs` 落地(Owner/AssetClass/三事件/AllocLedger/Command+链式/Outcome/BatchOutcome/DeviceHandle + `alloc!`·`free!` 宏);`contract.rs` 仅把裸 client 的 `alloc!` 改名 `client_alloc!`;行为门 7 条全绿(真实 CpuFace 作内层);三条教训入档(宏归位 / `Outcome::Buf` / dev-dep 成环双实例) |
| 2026-10-19 | v0.1 | 立项:三事同层(收编分配 / 账本接口 / 命令批发送);方案甲(`Rc<RefCell<dyn AllocLedger>>`)+ iface 薄包装(cuda 零改动);`Command` 链式构造 + `exec/batch` + `alloc!/free!` 宏;5 步施工序与判据 |

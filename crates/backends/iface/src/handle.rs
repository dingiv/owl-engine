//! **DeviceHandle:分配账 + 命令面**(设计卷 `docs/arch/device-命令面与分配账-设计与施工.md`)。
//!
//! 三件事同层同点(2026-10-19 用户定调):
//! ① **收编所有显存申请** —— engine/models 不再手动碰分配,但 **store 是纯状态机(不做 I/O)**,
//!    所以"摸显存"的只能是**客户端侧**的咽喉;钩子**挂在客户端**(不许挂到 server:跨线程)。
//! ② **接口倒置**:[`AllocLedger`] 由 **store 实现**;client 侧只依赖这个 trait
//!    (方案甲:`Arc<Mutex<dyn AllocLedger>>`,真 `Send` —— 见 §1.1)。
//! ③ **命令面**:[`Command`](链式构造)+ [`DeviceHandle::exec`] 单条 / [`DeviceHandle::batch`] 多条
//!    + [`alloc!`](crate::alloc) / [`free!`](crate::free) 宏。
//!
//! # 纪律
//! - **本模块一行不改 `cuda` 里实现的 [`DeviceClient`]**:这里是**薄转发 + 钩子**;
//! - **钩子在客户端侧发**(调用者线程 = 引擎线程)⇒ 无跨线程、无锁争用;
//! - **下沉 ≠ 归还**:[`MoveEvent`] 只换层,只有 [`FreeEvent`] 才注销(账本律);
//! - **不许静默**:钩子失败计入 `hook_fail`;骨架期未接线的命令**响亮拒**(返回结构化错误)。
//!
//! # 归属与类别上下文(2026-10-19 用户裁决 ②:扩 4 类 + 相位上下文)
//! `owner`(谁持有:turn/session/cache/system)由**生命周期节点**设(N1/N4/N5);
//! `class`(哪类资产)由**相位**设(装载=Weight / 捕获=Graph / 池=Pool / 执行=Temp),
//! 缺省 = [`AssetClass::Unclassified`](**可见**的兜底桶 —— 不许猜、不许谎报)。
//! 二者都住在**共享上下文**(`Arc<Mutex<Ctx>>`):`face_clone()` 出来的句柄与本体
//! **同源** ⇒ 节点设一次,全族生效(图族每张图各持一个句柄,逐个设 = 漏设的温床)。
//!
//! # 钩子覆盖(2026-10-19 用户裁决 ③:全出生/死亡 + 全等门)
//! 块**出生** = `alloc` / `alloc_uninit` / `htod` / `htod_on` / `htod_f32`;
//! 块**死亡** = `free`(成功收口后才销);**换层** = `swap`(Out/In ⇒ 每逻辑块一条 `MoveEvent`);
//! **图窗**:`graph_begin` 开窗 → 窗内出生的块记在窗里(捕获期后端走 slab 切片 ⇒ 块**随图存活**)
//! → `graph_end` 成功 ⇒ 挂到 `GraphId` 名下(等 `graph_destroy` 补发归还);
//! `graph_end` 失败 / `graph_abort` 的销账口径由**后端自述**决定
//! ([`DeviceClient::window_blocks_auto_reclaimed`](crate::contract::DeviceClient::window_blocks_auto_reclaimed)
//! 契约三条:CUDA = 整批回收;host mock = 交调用方 free)—— 本层**不猜**:
//! 后端不回收的场合保留窗与 id,计 `window_unclosed`(可见),由调用方显式 `free` 收。
//!
//! **`launch` 不算出生**:输出块 = 实参里最后一个 `Block`(调用方先 `alloc` 再发射;
//! 服务端 `launch.rs` 只发核,零分配)⇒ 出生只在分配族。

use std::collections::BTreeMap;
use std::future::Future;
use std::sync::{Arc, Mutex, MutexGuard};

use crate::contract::{
    BlockLedgerRow, Bytes, Connector, DeviceClient, Dtype, FiPlanSpec, GraphId, LaunchMsg,
    ModelError, PinnedRegion, Shape, SwapDir, SwapPiece, UpChunk, UpStream,
};

// ============================================================================
// §1 归属与资产类别(与 store 的 TierLedger 同构;store 侧 impl 时做映射)
// ============================================================================

/// **归属主体**(owner 维;账本的判据面:某 owner 单调爬坡 = 泄漏)。
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Owner {
    /// 活跃 turn(逐 turn 归还的临时/弹性面;N4 的对象)。
    Turn(u64),
    /// 会话持有(跨 turn:KV 链 / GDN 行;N5 的对象)。
    Session(u64),
    /// 缓存面(前缀缓存块 / 快照 —— 无单会话归属)。
    Cache,
    /// 系统常驻(权重 / 图 slab / 池骨架 / **未接线归属的兜底**)。
    System,
}

impl Owner {
    /// 缺省归属(`DeviceHandle::new` 的初值;未设上下文时的一切分配都落这里 ——
    /// **可见**而不是静默丢弃)。
    pub const fn default_owner() -> Self {
        Owner::System
    }

    /// 是否属于**可随生命周期回收**的面(turn/session)—— 报告面高亮用。
    pub fn is_lifecycle(&self) -> bool {
        matches!(self, Owner::Turn(_) | Owner::Session(_))
    }

    /// 短名(`turn:7` / `session:3` / `cache` / `system`)。
    pub fn label(&self) -> String {
        match self {
            Owner::Turn(t) => format!("turn:{t}"),
            Owner::Session(s) => format!("session:{s}"),
            Owner::Cache => "cache".into(),
            Owner::System => "system".into(),
        }
    }
}

/// **资产类别**(中性名;与 store 的 `AssetKind` 同构,映射在 store 侧做)。
///
/// 前八类是**store 自管面**的词表(KV/地板/弹性/快照/行/元数据/索引/临时);
/// 后四类是 2026-10-19 用户裁决新增的**转发面**词表 —— 裸 `DeviceClient` 的
/// 分配没有类别参数,靠相位上下文补;缺省落 `Unclassified`(可见兜底)。
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AssetClass {
    /// attention 历史块(可驱逐缓存面)。
    KvBlock,
    /// 活跃 turn 的直接显存地板(保留面)。
    Floor,
    /// 地板之上的弹性余量(可驱逐)。
    Elastic,
    /// GDN 状态快照(可驱逐;可下沉)。
    Snap,
    /// 活跃 turn 占用的 GDN 行(保留面)。
    Row,
    /// 会话元数据(µs 级;不参与驱逐)。
    Meta,
    /// 选择层检索索引(host 驻留)。
    Index,
    /// **临时工作区**(chunk 激活面 / plan sync 缓冲 / eager 回退 / logits 中转)。
    Temp,
    /// **权重与装载产物**(常驻;装载相位)。
    Weight,
    /// **图 slab 与图内自产块**(随图存活;捕获窗)。
    Graph,
    /// **池骨架**(KV/GDN/快照/草稿池的容器块;boot 期)。
    Pool,
    /// **未分类兜底**(尚未接线相位的分配;**可见** —— 它的字节是迁移进度指标)。
    Unclassified,
}

impl AssetClass {
    /// 短名(报告/日志)。
    pub fn label(&self) -> &'static str {
        match self {
            AssetClass::KvBlock => "kv",
            AssetClass::Floor => "floor",
            AssetClass::Elastic => "elastic",
            AssetClass::Snap => "snap",
            AssetClass::Row => "row",
            AssetClass::Meta => "meta",
            AssetClass::Index => "index",
            AssetClass::Temp => "temp",
            AssetClass::Weight => "weight",
            AssetClass::Graph => "graph",
            AssetClass::Pool => "pool",
            AssetClass::Unclassified => "unclassified",
        }
    }
}

// ============================================================================
// §1.1 账本接口(由 store 实现;client 只依赖它 —— 依赖倒置)
// ============================================================================

/// **分配事件**(出借;`id` = 设备块 id,与 server 块账房同源)。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AllocEvent {
    pub id: u64,
    pub bytes: usize,
    /// 分配点点名(调用方给;`DeviceClient::alloc` 的 `name` 必填律已在)。
    pub name: String,
    pub class: AssetClass,
    /// 归属(由 [`DeviceHandle`] 的 owner 上下文填)。
    pub owner: Owner,
    /// **设备序(TP 两 rank 共用一本账时区分卡)**;单卡恒 0。
    pub rank: u8,
}

/// **归还事件**(注销)。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FreeEvent {
    pub id: u64,
    /// 字节(事件可留 0 —— 账本按**出借登记**扣,登记才是权威)。
    pub bytes: usize,
}

/// **换层方向**。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoveDir {
    /// 设备 → 宿主(下沉;内容留)。
    Out,
    /// 宿主 → 设备(回加载)。
    In,
}

/// **换层事件**:只换 tier,**不改 owner、不注销**(账本律:下沉 ≠ 归还)。
///
/// `bytes` = 本次搬运的**逻辑块全量字节**(引擎按块聚合后发一条;
/// 服务端 `swap` 是多片命令,片在 handle 侧聚合)。部分搬运(片和 ≠ 登记的
/// 整块字节)由账本**响亮拒**,不许把"沉了一半"记成"整块下沉"。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MoveEvent {
    pub id: u64,
    pub bytes: usize,
    pub dir: MoveDir,
}

/// **账本接口**(由 store 实现;`DeviceHandle` 在客户端侧调用)。
///
/// 三条律(实现方必须守):
/// 1. **下沉 ≠ 归还**:`on_move` 只换层;只有 `on_free`/逐出才注销;
/// 2. **双持与幽灵归还响亮拒**:同 id 重复 `on_alloc` / 未登记 `on_free` = 不变量违反;
/// 3. **失败必留痕**:实现侧记 `hook_fail` 计数 + 一条 WARN,**不得**让记账失败中断主流程。
///
/// `Send` = 硬要求(见 [`LedgerCell`]):账本进 `Arc<Mutex<..>>`,而
/// [`DeviceClient`] 自身就是 `Send`(句柄要能随 face 走)。
pub trait AllocLedger: Send {
    /// 出借(登记 + 入账)。
    fn on_alloc(&mut self, ev: &AllocEvent);
    /// 归还(注销 + 出账)。
    fn on_free(&mut self, ev: &FreeEvent);
    /// 换层(只改 tier/owner,不注销)。
    fn on_move(&mut self, ev: &MoveEvent);

    /// **池取块**(缺口 4):**同 rank + 同 kind + 同尺寸**才给(P0-1:TP 两 rank
    /// 共写一本账 ⇒ 键必须带 rank);`None` = 未命中(调用方真分配)。缺省 = 无池面。
    fn pool_acquire(&mut self, _rank: u8, _class: AssetClass, _bytes: usize) -> Option<u64> {
        None
    }

    /// **池还块**:返回**必须真归还驱动**的 id 列表(空 = 已入池)。
    /// 不可池化 ⇒ `[id]`;超池上限顶出的旧块 ⇒ `[old…]`(**可能多块**,
    /// 2026-10-20 定谳:早先只返还最后一块 ⇒ 多顶出时其余块永久泄漏)。
    fn pool_release(&mut self, id: u64) -> Vec<u64> {
        vec![id]
    }

    /// **池排空**(压力路径 N6):清池并交出全部 id(调用方真归还驱动)。
    fn pool_drain(&mut self) -> Vec<u64> {
        Vec::new()
    }

    /// **账本身份令牌**(TP 两 rank 合流判据):同一个"书"必须给**同一个非零值**;
    /// `0` = 不表态(无身份的实现 —— 一律按"另一本书"处理)。
    ///
    /// 用途:[`DeviceHandle::adopt_ledger`] 区分两种接管 ——
    /// ① **同一本书**(令牌相等):接管**不丢任何事件** ⇒ 任何时点都幂等允许
    /// (TP:调用方已在**装载前**用 `Engine::new_with_ledger` 合流,装配期那次
    /// adopt 只是复用同一本;装载事件本就落在这本上);
    /// ② **另一本书**:只有 boot 期(零事件)才许接管(否则事件进孤儿账)。
    fn book_token(&self) -> usize {
        0
    }
}

/// **账本单元**(共享句柄;**同一本账**被 client 钩子与 store 自查共看)。
///
/// 为什么是 `Arc<Mutex>` 而不是方案甲字面的 `Rc<RefCell>`(2026-10-19 用户裁决):
/// [`DeviceClient`] 的 `Send` 是**硬约束**,而 `/debug/blocks` 的 `face_clone()`
/// 会把句柄**真的**送进 HTTP 线程 ⇒ `Rc` 在那条路径上是真的 UB,不是理论问题。
/// 单线程热路径上无争用锁 ≈ ns 级(相对一次分配/一发命令的 µs 可忽略)。
pub type LedgerCell = Arc<Mutex<dyn AllocLedger>>;

// ============================================================================
// §2 命令面(Command + 链式构造)
// ============================================================================

/// 三维尺寸(grid/block)。
pub type Dim3 = (u32, u32, u32);

/// **GPU server 命令**(统一出口;链式构造见 [`Command::args`] 等)。
///
/// 注:不带 `Clone/Debug` 派生 —— 内含 `LaunchMsg`(契约类型,未实现这两者);
/// 排障需要时用 [`Command::kind`](短名)。
pub enum Command {
    /// 清零分配(块登记 + 流序 memset;捕获期走 slab 切块)。
    Alloc {
        class: AssetClass,
        dtype: Dtype,
        elems: usize,
        name: String,
    },
    /// 免零化分配(uninit scratch;kernel 全量覆写的输出块不烙 memset 节点)。
    AllocUninit {
        class: AssetClass,
        dtype: Dtype,
        elems: usize,
        name: String,
    },
    /// 归还(→ server 归池;复用即在此层发生)。
    Free { ids: Vec<u64> },
    /// 核发射(链式:`Command::Launch(msg).grid(..).block(..).args(..)`)。
    Launch { msg: LaunchMsg, name: String },
    /// 流同步。
    Sync,
    /// 设备 → 宿主(下沉)。骨架期**未接线**(响亮拒)。
    Htod { id: u64, off: usize, bytes: usize },
    /// 宿主 → 设备(回加载)。骨架期**未接线**(响亮拒)。
    Dtoh { id: u64, off: usize, bytes: usize },
}

impl Command {
    /// 链式:设置 grid(覆盖 `LaunchMsg.grid`)。
    pub fn grid(mut self, g: Dim3) -> Self {
        if let Command::Launch { msg, .. } = &mut self {
            msg.grid = g;
        }
        self
    }

    /// 链式:设置 block(覆盖 `LaunchMsg.block`)。
    pub fn block(mut self, b: Dim3) -> Self {
        if let Command::Launch { msg, .. } = &mut self {
            msg.block = b;
        }
        self
    }

    /// 链式:设置共享内存字节(覆盖 `LaunchMsg.shared_mem`)。
    pub fn shared_mem(mut self, bytes: u32) -> Self {
        if let Command::Launch { msg, .. } = &mut self {
            msg.shared_mem = bytes;
        }
        self
    }

    /// 链式:设置核参数表。
    pub fn args(mut self, args: Vec<owl_kernels::contract::Arg>) -> Self {
        if let Command::Launch { msg, .. } = &mut self {
            msg.args = args;
        }
        self
    }

    /// 链式:改命令名(打点/账房聚合用)。
    pub fn name(mut self, n: &str) -> Self {
        match &mut self {
            Command::Alloc { name, .. }
            | Command::AllocUninit { name, .. }
            | Command::Launch { name, .. } => *name = n.to_string(),
            _ => {}
        }
        self
    }

    /// 构造糖(宏用):清零分配。
    pub fn alloc(class: AssetClass, dtype: Dtype, elems: usize, name: &str) -> Self {
        Command::Alloc { class, dtype, elems, name: name.to_string() }
    }

    /// 构造糖(宏用):归还。
    pub fn free(id: u64) -> Self {
        Command::Free { ids: vec![id] }
    }

    /// 短名(打点/日志用;`Command` 无 `Debug` 派生 ⇒ 这是排障入口)。
    pub fn kind(&self) -> &'static str {
        match self {
            Command::Alloc { .. } => "alloc",
            Command::AllocUninit { .. } => "alloc_uninit",
            Command::Free { .. } => "free",
            Command::Launch { .. } => "launch",
            Command::Sync => "sync",
            Command::Htod { .. } => "htod",
            Command::Dtoh { .. } => "dtoh",
        }
    }
}

/// 单条命令的执行结果。
///
/// ⚠️ `Launch` 也**产出块句柄**(输出块)—— 契约里 `launch` 的返回就是 `Bytes`,
/// 不许丢(全仓律 #1)⇒ 统一走 [`Outcome::Buf`]。
///
/// 注意:**产出句柄 ≠ 块出生**。服务端 `launch.rs` 的输出块 = 实参里最后一个
/// `Block`(调用方先 `alloc` 再发射)⇒ 记账意义上的"出生"只发生在分配族
/// ([`AllocEvent`] 只在 `alloc/alloc_uninit/htod` 系上发)。
#[derive(Clone, Debug)]
pub enum Outcome {
    /// 产出一个设备块句柄(`Alloc` / `AllocUninit` / `Launch` 的输出块)。
    Buf(Bytes),
    /// 无返回值(`Free` / `Sync`)。
    Done,
}

/// 批的结果(逐条;`first_err` = 首个失败下标,**已投递的不回滚**)。
#[derive(Clone, Debug)]
pub struct BatchOutcome {
    pub outcomes: Vec<Result<Outcome, ModelError>>,
    pub first_err: Option<usize>,
    /// 实际投递段数(打点:批 vs 逐条的提交税对比)。
    pub segments: usize,
}

impl BatchOutcome {
    /// 全成?(便捷判据;失败细节看 `outcomes`/`first_err`)
    pub fn all_ok(&self) -> bool {
        self.first_err.is_none()
    }
}

// ============================================================================
// §3 共享上下文(owner/class/图窗/失败计数)
// ============================================================================

/// **捕获窗**(`graph_begin` → `graph_end`):窗内出生的块随图存活,
/// 服务端在 `graph_destroy`/`graph_abort` 时**整批移出块账房**(客户端不发 `free`)
/// ⇒ 本窗登记就是"图销毁时要补发的归还清单"。
#[derive(Default, Clone, Debug)]
struct Window {
    /// 窗内出生、**尚未归还**的块 id(客户端归还过的会被摘掉)。
    ids: Vec<u64>,
    /// 窗内发射数(镜像服务端 `cap.nodes` 空判据:零发射 ⇒ 空窗,服务端整批回收)。
    nodes: u64,
}

/// 句柄的**共享上下文**(所有克隆同源:face_clone / loader_faces 出来的句柄
/// 与本体看同一份归属、同一条窗栈、同一个失败计数)。
struct Ctx {
    owner: Owner,
    class: AssetClass,
    /// 本引擎的设备序(单卡 0;TP rank1 = 1)—— 事件按它分组,卡不混账。
    dev_rank: u8,
    /// 池命中/入池计数(诊断面;缺口 4)
    pool_hits: u64,
    pool_released: u64,
    /// 本上下文发出的事件条数(诊断 + `adopt_ledger` 的 boot 期判据;
    /// 放共享上下文 ⇒ 克隆句柄的发事件也算“本引擎已开始记账”)。
    events_sent: u64,
    hook_fail: u64,
    /// 捕获窗栈(服务端禁嵌套;栈式写法对"误嵌套"也天然正确)。
    windows: Vec<Window>,
    /// `graph_end` 成功后按图存档的"待补归还"清单(`graph_destroy` 时销)。
    graphs: BTreeMap<GraphId, Vec<u64>>,
    /// 服务端留尸计数(窗失败且窗内有发射 ⇒ 服务端不回收;**可见**不静默)。
    window_unclosed: u64,
}

impl Ctx {
    fn new() -> Self {
        Self {
            owner: Owner::default_owner(),
            class: AssetClass::Unclassified,
            dev_rank: 0,
            pool_hits: 0,
            pool_released: 0,
            events_sent: 0,
            hook_fail: 0,
            windows: Vec::new(),
            graphs: BTreeMap::new(),
            window_unclosed: 0,
        }
    }

    /// 记账失败:计数 + 一条裸日志(iface 无日志依赖;机读面 = `hook_fail`)。
    fn bump(&mut self, what: &str, detail: String) {
        self.hook_fail = self.hook_fail.saturating_add(1);
        eprintln!("[device-handle] {what}:{detail}");
    }
}

/// **归属/类别作用域**(RAII:出作用域自动还原;装载/捕获/池/执行相位用)。
///
/// 为什么用 RAII 而不是"设完记得还原":相位里**任何**早退(`?`)都会让手工还原
/// 漏掉 ⇒ 后续分配归错账(比不记账更坏)。
pub struct CtxScope {
    /// 持有共享单元本身(**不借句柄**)—— 否则"作用域内还要 &mut 用句柄"
    /// 会撞借用检查(实测:`face.scoped_ctx(..)` + `StatePool::alloc(&mut face, ..)`)。
    /// `None` = 空作用域(无账本的裸 client:没有归属可谈)。
    ctx: Option<Arc<Mutex<Ctx>>>,
    saved: (Owner, AssetClass),
}

impl CtxScope {
    /// 空作用域(裸 client 的缺省实现:进出都不改任何东西 —— 不是"静默失败",
    /// 而是**无账本 ⇒ 无归属可谈**)。
    pub fn noop() -> Self {
        Self { ctx: None, saved: (Owner::default_owner(), AssetClass::Unclassified) }
    }
}

impl Drop for CtxScope {
    fn drop(&mut self) {
        let Some(ctx) = self.ctx.as_ref() else { return };
        let mut c = lock_ctx(ctx);
        c.owner = self.saved.0;
        c.class = self.saved.1;
    }
}

/// 上下文加锁(**锁中毒不许 panic 级联**:取回内层数据 + 计数可见)。
fn lock_ctx(m: &Mutex<Ctx>) -> MutexGuard<'_, Ctx> {
    let poisoned = m.is_poisoned();
    let mut g = match m.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    if poisoned {
        g.bump("ctx-lock", "上下文锁中毒(已续用内部数据)".into());
    }
    g
}

/// 按块聚合 `swap` 的片段 ⇒ 每逻辑块一条 [`MoveEvent`](整块下沉/回加载)。
///
/// 为什么在客户端聚合:服务端 `swap` 是**多片一条命令**(一个逻辑块 = 层数 × 2 片),
/// 而账本是按**逻辑块**记账的(一条资产一行)⇒ 片必须在进账前合成块。
fn swap_moves(dir: SwapDir, pieces: &[SwapPiece]) -> Vec<MoveEvent> {
    let mut by_block: BTreeMap<u64, usize> = BTreeMap::new();
    for p in pieces {
        *by_block.entry(p.block).or_insert(0) += p.len;
    }
    let d = match dir {
        SwapDir::Out => MoveDir::Out,
        SwapDir::In => MoveDir::In,
    };
    by_block.into_iter().map(|(id, bytes)| MoveEvent { id, bytes, dir: d }).collect()
}

// ============================================================================
// §4 DeviceHandle:薄转发 + 钩子 + 命令面
// ============================================================================

/// **设备句柄**(engine/models 唯一能碰的设备面)。
///
/// - `ledger = None` ⇒ **纯转发**(迁移态:零行为变化,便于灰度);
/// - `ledger = Some(..)` ⇒ 每次分配/归还/换层发事件(客户端侧,同线程);
/// - `owner`/`class` = **共享**归属与类别上下文(生命周期节点 / 相位设置)。
pub struct DeviceHandle<D: DeviceClient> {
    inner: D,
    ledger: Option<LedgerCell>,
    ctx: Arc<Mutex<Ctx>>,
}

impl<D: DeviceClient> DeviceHandle<D> {
    /// 构造(唯一入口)。`ledger = None` = 纯转发(迁移态)。
    pub fn new(inner: D, ledger: Option<LedgerCell>) -> Self {
        Self { inner, ledger, ctx: Arc::new(Mutex::new(Ctx::new())) }
    }

    // ── 上下文面(节点/相位调;`&self` 即可 —— 内部共享可变)──────────────

    /// **归属上下文**(生命周期节点调用:N1 `Turn(t)` / N5 `Session(s)` / N4 复位 `System`)。
    pub fn set_owner(&self, o: Owner) {
        self.ctx().owner = o;
    }

    /// 当前归属。
    pub fn owner(&self) -> Owner {
        self.ctx().owner
    }

    /// **类别上下文**(相位设置:装载=Weight / 捕获=Graph / 池=Pool / 执行=Temp)。
    pub fn set_class(&self, c: AssetClass) {
        self.ctx().class = c;
    }

    /// 当前类别。
    pub fn class(&self) -> AssetClass {
        self.ctx().class
    }

    /// **设备序**(TP:rank1 的 boot 装配在**任何分配之前**调一次;单卡不用)。
    pub fn set_dev_rank(&self, rank: u8) {
        self.ctx().dev_rank = rank;
    }

    /// 当前设备序。
    pub fn dev_rank(&self) -> u8 {
        self.ctx().dev_rank
    }

    /// **boot 期账本接管**(TP:两 rank 必须写**同一本**账)。
    ///
    /// 判据 = 本句柄**尚未发过任何事件**(发过就不是 boot 期了)；违反 ⇒ 响亮拒
    /// (接管会把已有事件丢进孤儿账 —— 静默丢账比拒更坏)。
    pub fn adopt_ledger(&mut self, ledger: LedgerCell) -> Result<(), ModelError> {
        // **幂等支路(同一本书)**:接管不改"账在哪"⇒ 零事件丢失风险 ⇒ 放行。
        // 判据 = 令牌相等(`Arc::ptr_eq` 先行:同一 cell 双锁会自死锁)。
        if let Some(cur) = self.ledger.as_ref() {
            let same = if Arc::ptr_eq(cur, &ledger) {
                true
            } else {
                // 中毒 = 有 panic 发生在持账期间:取回内层数据续用(与 `ctx` 同律)
                let a = match cur.lock() { Ok(g) => g, Err(p) => p.into_inner() };
                let b = match ledger.lock() { Ok(g) => g, Err(p) => p.into_inner() };
                let t = a.book_token();
                t != 0 && t == b.book_token()
            };
            if same {
                self.ledger = Some(ledger);
                return Ok(());
            }
        }
        let sent = self.ctx().events_sent;
        if sent != 0 {
            return Err(ModelError::Msg(format!(
                "adopt_ledger: 本引擎已发过 {sent} 条事件,只允许 boot 期(零事件)接管                  —— 接管会把已有事件丢进孤儿账(静默丢账比拒更坏)"
            )));
        }
        self.ledger = Some(ledger);
        Ok(())
    }

    /// **作用域化设置**(RAII 还原;相位代码用这个,不要手工 set/还原)。
    pub fn scoped_ctx(&self, owner: Owner, class: AssetClass) -> CtxScope {
        let saved = {
            let mut c = self.ctx();
            let saved = (c.owner, c.class);
            c.owner = owner;
            c.class = class;
            saved
        };
        CtxScope { ctx: Some(self.ctx.clone()), saved }
    }

    /// 记账失败计数(诊断面;非零 = 有钩子调用失败过)。
    pub fn hook_fail(&self) -> u64 {
        self.ctx().hook_fail
    }

    /// **诊断三元组** `(hook_fail, window_unclosed, 窗内未归还块数)`。
    ///
    /// ⚠️ 三条读数**必须各自落变量**:元组表达式里的临时 guard 活到**语句末**
    /// (不是子表达式末)⇒ 直接写 `(self.hook_fail(), self.window_unclosed(), ..)`
    /// 会在第二次取锁时**自锁死**(std Mutex 不可重入;实测踩到过)。
    pub fn ledger_probe(&self) -> (u64, u64, usize) {
        let hf = self.hook_fail();
        let wu = self.window_unclosed();
        let owb = self.open_window_blocks();
        (hf, wu, owb)
    }

    /// **窗未收次数**(诊断面;非零 = 有窗收尾时去向未定,见 [`Window`])。
    pub fn window_unclosed(&self) -> u64 {
        self.ctx().window_unclosed
    }

    /// **池命中/入池计数**(诊断面;缺口 4)。
    pub fn pool_counters(&self) -> (u64, u64) {
        let c = self.ctx();
        (c.pool_hits, c.pool_released)
    }

    fn note_pool_hit(&self) {
        let mut c = self.ctx();
        c.pool_hits = c.pool_hits.saturating_add(1);
    }

    fn note_pool_released(&self, n: usize) {
        let mut c = self.ctx();
        c.pool_released = c.pool_released.saturating_add(n as u64);
    }

    /// 当前捕获窗内**尚未归还**的块数(诊断面;非 0 且非捕获期 = 有图未退休)。
    pub fn open_window_blocks(&self) -> usize {
        let c = self.ctx();
        c.windows.iter().map(|w| w.ids.len()).sum::<usize>()
            + c.graphs.values().map(|v| v.len()).sum::<usize>()
    }

    /// 只读直通(查询类不走命令面、不记账)。
    pub fn client(&self) -> &D {
        &self.inner
    }

    /// 可变直通(**仅限不产生分配的用途**;封印见设计卷 §六)。
    pub fn client_mut(&mut self) -> &mut D {
        &mut self.inner
    }

    // ── 命令面 ──────────────────────────────────────────────────────────

    /// **单条命令**([`Self::batch`] 的特例:实现单源)。
    pub async fn exec(&mut self, cmd: Command) -> Result<Outcome, ModelError> {
        let mut out = self.batch([cmd]).await?;
        // 单条语义:把批的错误就地展开(不静默)
        match out.outcomes.pop() {
            Some(r) => r,
            None => Err(ModelError::Msg("DeviceHandle::exec: 批为空(内部错误)".into())),
        }
    }

    /// **命令批**:一次投递 + 一次收口(钩子逐条入账,但只一次往返)。
    ///
    /// 骨架期 = **语义等价批**(逐条转发 + 聚合结果,后端零改动);
    /// 原生批(N 条合成一次 ack)在后续步骤接线 —— 判据 = `segments` 下降 +
    /// token 逐位恒等门(结果不变)。
    pub async fn batch(
        &mut self,
        cmds: impl IntoIterator<Item = Command>,
    ) -> Result<BatchOutcome, ModelError> {
        let mut outcomes: Vec<Result<Outcome, ModelError>> = Vec::new();
        let mut first_err: Option<usize> = None;
        let mut segments = 0usize;
        // 本批的延迟入账(收口后统一写账:失败条**不**入账)
        let mut events: Vec<AllocEvent> = Vec::new();
        let mut frees: Vec<FreeEvent> = Vec::new();
        let mut moves: Vec<MoveEvent> = Vec::new();
        for (i, cmd) in cmds.into_iter().enumerate() { // FIXME: 这里没有达到批处理的目的，你要知道。这个Command，它是要让后端实现去感知的，你懂吧？为什么要搞批处理？搞批处理是为了省掉这个线程间同步开销啊。对吧？这一来一回，走这个线程间同步啊，那么就让这个每一个命令时间拖长了。
            segments += 1;
            let r = self.run_one(cmd, &mut events, &mut frees, &mut moves).await;
            if r.is_err() && first_err.is_none() {
                first_err = Some(i);
            }
            outcomes.push(r);
        }
        // 收口:入账(失败条已在上方被拦 ⇒ 这里全是成功事件)
        self.send_events(&events, &frees, &moves);
        Ok(BatchOutcome { outcomes, first_err, segments })
    }

    /// 单条执行(内部;把事件塞进本批的缓冲)。
    async fn run_one(
        &mut self,
        cmd: Command,
        events: &mut Vec<AllocEvent>,
        frees: &mut Vec<FreeEvent>,
        _moves: &mut Vec<MoveEvent>, // 命令面无换层命令(swap 走 DeviceClient 转发)
    ) -> Result<Outcome, ModelError> {
        match cmd {
            Command::Alloc { class, dtype, elems, name } => {
                let b = self.inner.alloc(dtype, elems, &name).await?;
                self.note_birth(&b, elems * dtype.size_bytes(), class, &name, events);
                Ok(Outcome::Buf(b))
            }
            Command::AllocUninit { class, dtype, elems, name } => {
                let b = self.inner.alloc_uninit(dtype, elems, &name).await?;
                self.note_birth(&b, elems * dtype.size_bytes(), class, &name, events);
                Ok(Outcome::Buf(b))
            }
            Command::Free { ids } => {
                // **缺口 4**:可池化的块**不回驱动**(入 store 池 ⇒ 零命令复用);
                // 池内那条**不**发 FreeEvent(块仍在借 —— 只是变得可复用)。
                let (rest, pooled) = self.pool_release_all(&ids);
                if pooled > 0 {
                    self.note_pool_released(pooled);
                }
                if !rest.is_empty() {
                    self.inner.free(&rest).await?;
                }
                for id in rest {
                    // 字节数在账本侧反查(FreeEvent 只带 id);此处留 0 = 由账本按登记扣
                    frees.push(FreeEvent { id, bytes: 0 });
                }
                for id in &ids {
                    self.untrack(*id);
                }
                Ok(Outcome::Done)
            }
            Command::Launch { msg, .. } => {
                // 输出块不许丢(律 #1):原样带出。**不算出生**(块 = 实参里的 Block)
                Ok(Outcome::Buf(self.inner.launch(msg).await?))
            }
            Command::Sync => {
                self.inner.sync().await?;
                Ok(Outcome::Done)
            }
            // 骨架期未接线:**响亮拒**(不是静默 no-op)
            Command::Htod { .. } => Err(ModelError::Msg(
                "DeviceHandle: Command::Htod 尚未接线(骨架期;见设计卷 §九 步 5)".into(),
            )),
            Command::Dtoh { .. } => Err(ModelError::Msg(
                "DeviceHandle: Command::Dtoh 尚未接线(骨架期;见设计卷 §九 步 5)".into(),
            )),
        }
    }

    // ── 钩子(全部同步短锁;绝不跨 await 持锁)────────────────────────────

    fn ctx(&self) -> MutexGuard<'_, Ctx> {
        lock_ctx(&self.ctx)
    }

    /// 出生记账:事件入账 + 记进当前捕获窗(图销毁时整批补发归还)。
    fn note_birth(
        &self,
        b: &Bytes,
        bytes: usize,
        class: AssetClass,
        name: &str,
        sink: &mut Vec<AllocEvent>,
    ) {
        let (owner, rank, in_window) = {
            let c = self.ctx();
            (c.owner, c.dev_rank, !c.windows.is_empty())
        };
        // **窗内出生 ⇒ Graph 类**(服务端捕获期恒走 `carve_block`,块随图存活;
        // 这不是猜测而是机制事实 —— 窗内不可能出 Owned 块)。
        let class = if in_window { AssetClass::Graph } else { class };
        sink.push(AllocEvent { id: b.id, bytes, name: name.to_string(), class, owner, rank });
        let mut c = self.ctx();
        if let Some(w) = c.windows.last_mut() {
            w.ids.push(b.id);
        }
    }

    /// 归还记账:从捕获窗/图存档里摘掉(它们只在**未归还**时才需要补发)。
    fn untrack(&self, id: u64) {
        let mut c = self.ctx();
        for w in c.windows.iter_mut() {
            w.ids.retain(|x| *x != id);
        }
        for v in c.graphs.values_mut() {
            v.retain(|x| *x != id);
        }
        c.graphs.retain(|_, v| !v.is_empty());
    }

    /// 窗内发射计数(镜像服务端 `cap.nodes`:零发射 ⇒ 空窗 ⇒ 服务端整批回收)。
    fn note_node(&self) {
        let mut c = self.ctx();
        if let Some(w) = c.windows.last_mut() {
            w.nodes += 1;
        }
    }

    /// **池取块**(真分配之前问一次;命中 ⇒ 零命令复用)。
    ///
    /// **P0-3(安全门)**:捕获窗开着时**一律不取池** —— 窗内块若来自池(而非 slab 切片),
    /// 它的指针会被烘进图,之后被复用/顶出/排空 ⇒ 图回放踩已改嫁的块。
    fn pool_acquire(&self, class: AssetClass, bytes: usize) -> Option<u64> {
        let cell = self.ledger.as_ref()?;
        let (rank, in_window) = {
            let c = self.ctx();
            (c.dev_rank, !c.windows.is_empty())
        };
        if in_window {
            return None;
        }
        let mut g = match cell.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        g.pool_acquire(rank, class, bytes)
    }

    /// **压力自愈**(N6 第一刀):池排空 ⇒ 真归还驱动 ⇒ 销账。返回释放块数。
    ///
    /// 为什么放在句柄:它是唯一同时握有**池(store 账本)**与**驱动口(face)**的一层
    /// —— 决策仍在 store(排空哪些 = `pool_drain`),执行在这里。
    pub async fn drain_pool_to_driver(&mut self) -> usize {
        let Some(cell) = self.ledger.as_ref() else { return 0 };
        let ids = {
            let mut g = match cell.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            g.pool_drain()
        };
        if ids.is_empty() {
            return 0;
        }
        if let Err(e) = self.inner.free(&ids).await {
            // 释放失败**不许静默**:本块仍归池(不销账),下次再来
            eprintln!("[device-handle] 池排空释放失败({} 块):{e}", ids.len());
            // 回填入池可能**顶出旧块** —— 那些 id 必须真还驱动(否则又是泄漏)
            let evicted = {
                let mut g = match cell.lock() {
                    Ok(g) => g,
                    Err(p) => p.into_inner(),
                };
                let mut evicted = Vec::new();
                for id in &ids {
                    evicted.extend(g.pool_release(*id));
                }
                evicted
            };
            if !evicted.is_empty() {
                if let Err(e2) = self.inner.free(&evicted).await {
                    eprintln!("[device-handle] 池回填顶出释放失败({} 块):{e2}", evicted.len());
                } else {
                    let frees: Vec<FreeEvent> =
                        evicted.iter().map(|id| FreeEvent { id: *id, bytes: 0 }).collect();
                    self.send_events(&[], &frees, &[]);
                    for id in &evicted {
                        self.untrack(*id);
                    }
                }
            }
            return 0;
        }
        let frees: Vec<FreeEvent> = ids.iter().map(|id| FreeEvent { id: *id, bytes: 0 }).collect();
        self.send_events(&[], &frees, &[]);
        for id in &ids {
            self.untrack(*id);
        }
        ids.len()
    }

    /// **池还块**(逐 id):返回 `(必须真归还的 id 列表, 入池条数)`。
    ///
    /// 两种"必须归还":①不可池化(同一 id)②超池上限**顶出**的旧块
    /// (**可以多条** —— 一次还块可能要腾多块;旧实现只收最后一条 ⇒ 泄漏)。
    /// 入池条数 = 本次真正留在池里的块数(打点;顶出块在 store 侧另计 `pool_evicted`)。
    fn pool_release_all(&self, ids: &[u64]) -> (Vec<u64>, usize) {
        let Some(cell) = self.ledger.as_ref() else { return (ids.to_vec(), 0) };
        let mut g = match cell.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let mut must_free = Vec::new();
        let mut pooled = 0usize;
        for id in ids {
            let real = g.pool_release(*id);
            if real.is_empty() {
                pooled += 1;
            } else {
                must_free.extend(real);
            }
        }
        (must_free, pooled)
    }

    /// 发事件(客户端侧;**短锁**,绝不跨 await)。
    ///
    /// 锁中毒 = 有 panic 发生在持账期间(设计上不应发生):取回内层数据继续,
    /// 并**计数可见**(不许静默,也不许 panic 级联)。
    fn send_events(&self, births: &[AllocEvent], frees: &[FreeEvent], moves: &[MoveEvent]) {
        if births.is_empty() && frees.is_empty() && moves.is_empty() {
            return;
        }
        let Some(cell) = self.ledger.as_ref() else { return }; // 纯转发态
        let poisoned = cell.is_poisoned();
        {
            // 借用范围收在块内 ⇒ 绝不跨 await(本函数同步)
            let mut guard = match cell.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            for ev in births {
                guard.on_alloc(ev);
            }
            for ev in frees {
                guard.on_free(ev);
            }
            for ev in moves {
                guard.on_move(ev);
            }
        }
        let n = (births.len() + frees.len() + moves.len()) as u64;
        {
            let mut c = self.ctx();
            c.events_sent = c.events_sent.saturating_add(n);
            if poisoned {
                c.bump("ledger-lock", "事件入口锁中毒(已续用内部数据)".into());
            }
        }
    }

    /// 补发归还(图窗整批回收:abort / 空窗失败 / graph_destroy)。
    fn emit_free(&self, ids: &[u64]) {
        if ids.is_empty() {
            return;
        }
        let frees: Vec<FreeEvent> = ids.iter().map(|id| FreeEvent { id: *id, bytes: 0 }).collect();
        self.send_events(&[], &frees, &[]);
    }

}

// ============================================================================
// §5 `DeviceClient` 实现:薄转发(39 法)+ 出生/死亡/换层/图窗钩子
// ============================================================================
//
// **纪律**:所有方法一律**显式转发**到 `self.inner` —— 绝不用 trait 缺省实现。
// 缺省实现会**静默换语义**(例:`alloc_uninit` 缺省 = 委托 `alloc` ⇒ 捕获窗内
// 多烙 memset 节点;`htod_on` 缺省 = 退化 H2D 流 ⇒ 顺序义务错位)。
impl<D: DeviceClient> DeviceClient for DeviceHandle<D> {
    // ── 上下文面(泛型面:引擎内核按 `D: DeviceClient` 写,拿不到具名句柄类型
    //    ⇒ 这三条必须在 trait 上;具名的 [`DeviceHandle::set_owner`] 等是同一操作
    //    的具名入口,行为逐字一致)────────────────────────────────────
    fn set_owner(&self, o: Owner) {
        self.ctx().owner = o;
    }

    fn set_class(&self, c: AssetClass) {
        self.ctx().class = c;
    }

    fn scoped_ctx(&self, o: Owner, c: AssetClass) -> CtxScope {
        DeviceHandle::scoped_ctx(self, o, c)
    }

    fn ledger_probe(&self) -> (u64, u64, usize) {
        DeviceHandle::ledger_probe(self)
    }

    fn drain_temp_pool(&mut self) -> impl Future<Output = usize> + Send {
        // UFCS 钉死走具名实现(见契约面注释:同名解析事故)
        async move { DeviceHandle::drain_pool_to_driver(self).await }
    }

    fn op_env(&self) -> Option<owl_kernels::driver::OpEnv> {
        self.inner.op_env()
    }

    fn device_kernels(&self) -> bool {
        self.inner.device_kernels()
    }

    fn graph_begin(&mut self) -> impl Future<Output = Result<(), ModelError>> + Send {
        async move {
            self.inner.graph_begin().await?;
            self.ctx().windows.push(Window::default());
            Ok(())
        }
    }

    fn graph_end(&mut self) -> impl Future<Output = Result<GraphId, ModelError>> + Send {
        async move {
            let r = self.inner.graph_end().await;
            let win = self.ctx().windows.pop();
            match (r, win) {
                (Ok(gid), Some(w)) => {
                    if !w.ids.is_empty() {
                        self.ctx().graphs.entry(gid).or_default().extend(w.ids);
                    }
                    Ok(gid)
                }
                (Ok(gid), None) => Ok(gid), // 窗栈空(内层自管捕获;本层不记账)
                (Err(e), Some(w)) => {
                    // **窗失败 ⇒ 不猜**:块去向是后端知识(见 `window_blocks_auto_reclaimed` 契约)。
                    // 唯一可判的空窗例外(契约③:后端已整批回收且窗已关)才销账;
                    // 其余情况**保留窗与 id**并计数可见 —— 后续 `graph_abort` 成功
                    // (契约①)或调用方显式 `free` 才是真销账点。
                    let auto = self.inner.window_blocks_auto_reclaimed();
                    if auto && w.nodes == 0 {
                        self.emit_free(&w.ids);
                    } else {
                        let n = w.ids.len();
                        self.ctx().window_unclosed += 1;
                        if n > 0 {
                            eprintln!(
                                "[device-handle] 捕获窗失败(窗内发射 {}):{n} 块去向由后端定,先不销账(graph_end 错:{e})",
                                w.nodes
                            );
                        }
                        // 窗放回栈顶(调用方的 abort/显式 free 才收)
                        self.ctx().windows.push(w);
                    }
                    Err(e)
                }
                (Err(e), None) => Err(e),
            }
        }
    }

    fn graph_launch(&mut self, graph: GraphId) -> impl Future<Output = Result<(), ModelError>> + Send {
        self.note_node();
        self.inner.graph_launch(graph)
    }

    fn configure_fi_plan(&mut self, self_carry: bool, cuda_graph: bool) -> Result<(), ModelError> {
        self.inner.configure_fi_plan(self_carry, cuda_graph)
    }

    fn fi_plan_sync(&mut self, spec: FiPlanSpec) -> impl Future<Output = Result<(), ModelError>> + Send {
        self.inner.fi_plan_sync(spec)
    }

    fn graph_abort(&mut self) -> impl Future<Output = Result<(), ModelError>> + Send {
        async move {
            let r = self.inner.graph_abort().await;
            // 同律:scrutinee 的临时 guard 活过整个 match ⇒ 先落变量再 match
            let popped = self.ctx().windows.pop();
            match popped {
                // 契约①:`abort` 成功 ⇒ 后端整批回收 ⇒ 补发归还
                Some(w) if r.is_ok() => {
                    if self.inner.window_blocks_auto_reclaimed() {
                        self.emit_free(&w.ids);
                    }
                }
                // `abort` 失败 = 窗已不在(或收尾失败):不销账、放回、计数可见
                Some(w) => {
                    self.ctx().window_unclosed += 1;
                    if !w.ids.is_empty() {
                        eprintln!(
                            "[device-handle] graph_abort 失败:{}(窗未收,{} 块去向未定)",
                            r.as_ref().err().map(|e| e.to_string()).unwrap_or_default(),
                            w.ids.len()
                        );
                    }
                    self.ctx().windows.push(w);
                }
                None => {}
            }
            r
        }
    }

    fn graph_destroy(&mut self, graph: GraphId) -> impl Future<Output = Result<(), ModelError>> + Send {
        async move {
            let r = self.inner.graph_destroy(graph).await;
            if r.is_ok() {
                // 契约②:图退休 ⇒ 后端把本图自产块整批移出账房 ⇒ 补发归还
                let ids = self.ctx().graphs.remove(&graph).unwrap_or_default();
                if self.inner.window_blocks_auto_reclaimed() {
                    self.emit_free(&ids);
                }
            }
            r
        }
    }

    fn alloc(
        &mut self,
        dtype: Dtype,
        elems: usize,
        name: &str,
    ) -> impl Future<Output = Result<Bytes, ModelError>> + Send {
        // 池命中 ⇒ **零命令复用**;`alloc` 是零化语义 ⇒ 命中后补一次 memset(不许静默降级)
        let class = self.ctx().class; // 先落变量(上下文临时 guard 不许跨到下一句)
        let hit = DeviceHandle::pool_acquire(self, class, elems * dtype.size_bytes());
        async move {
            if let Some(id) = hit {
                self.note_pool_hit();
                let b = Bytes::new(id, elems, dtype, 0);
                self.inner.memset_zero(&b, elems * dtype.size_bytes()).await?;
                return Ok(b);
            }
            let b = match self.inner.alloc(dtype, elems, name).await {
                Ok(b) => b,
                Err(e) if matches!(e, ModelError::PoolExhausted { .. }) => {
                    let n = self.drain_pool_to_driver().await;
                    if n > 0 {
                        eprintln!("[device-handle] 分配遇 PoolExhausted ⇒ 排空复用池 {n} 块后重试");
                        self.inner.alloc(dtype, elems, name).await?
                    } else {
                        return Err(e);
                    }
                }
                Err(e) => return Err(e),
            };
            let mut ev = Vec::new();
            // 上下文读**先落变量**:实参里的 `self.ctx()` 临时 guard 会活到语句末,
            // 而 `note_birth` 内部还要再锁同一把 —— std Mutex 不可重入 ⇒ 自锁死。
            let class = self.ctx().class;
            self.note_birth(&b, elems * dtype.size_bytes(), class, name, &mut ev);
            self.send_events(&ev, &[], &[]);
            Ok(b)
        }
    }

    fn alloc_uninit(
        &mut self,
        dtype: Dtype,
        elems: usize,
        name: &str,
    ) -> impl Future<Output = Result<Bytes, ModelError>> + Send {
        // 池命中 ⇒ 零命令 + **免零化**(kernel 全量覆写的块;设计原文的复用红利)
        let class = self.ctx().class; // 先落变量(上下文临时 guard 不许跨到下一句)
        let hit = DeviceHandle::pool_acquire(self, class, elems * dtype.size_bytes());
        async move {
            if let Some(id) = hit {
                self.note_pool_hit();
                return Ok(Bytes::new(id, elems, dtype, 0));
            }
            let b = match self.inner.alloc_uninit(dtype, elems, name).await {
                Ok(b) => b,
                // **N6 压力自愈**:硬预算/池耗尽 ⇒ 先把复用池还给驱动,再试一次
                Err(e) if matches!(e, ModelError::PoolExhausted { .. }) => {
                    let n = self.drain_pool_to_driver().await;
                    if n > 0 {
                        eprintln!("[device-handle] 分配遇 PoolExhausted ⇒ 排空复用池 {n} 块后重试");
                        self.inner.alloc_uninit(dtype, elems, name).await?
                    } else {
                        return Err(e);
                    }
                }
                Err(e) => return Err(e),
            };
            let mut ev = Vec::new();
            // 上下文读**先落变量**:实参里的 `self.ctx()` 临时 guard 会活到语句末,
            // 而 `note_birth` 内部还要再锁同一把 —— std Mutex 不可重入 ⇒ 自锁死。
            let class = self.ctx().class;
            self.note_birth(&b, elems * dtype.size_bytes(), class, name, &mut ev);
            self.send_events(&ev, &[], &[]);
            Ok(b)
        }
    }

    fn htod(
        &mut self,
        dtype: Dtype,
        shape: &Shape,
        src: &[u8],
        name: &str,
    ) -> impl Future<Output = Result<Bytes, ModelError>> + Send {
        async move {
            let b = self.inner.htod(dtype, shape, src, name).await?;
            let mut ev = Vec::new();
            let class = self.ctx().class;
            self.note_birth(&b, src.len(), class, name, &mut ev);
            self.send_events(&ev, &[], &[]);
            Ok(b)
        }
    }

    fn htod_on(
        &mut self,
        stream: UpStream,
        dtype: Dtype,
        shape: &Shape,
        src: &[u8],
        name: &str,
    ) -> impl Future<Output = Result<Bytes, ModelError>> + Send {
        async move {
            let b = self.inner.htod_on(stream, dtype, shape, src, name).await?;
            let mut ev = Vec::new();
            let class = self.ctx().class;
            self.note_birth(&b, src.len(), class, name, &mut ev);
            self.send_events(&ev, &[], &[]);
            Ok(b)
        }
    }

    fn dtoh(&mut self, b: &Bytes, out: &mut [u8]) -> impl Future<Output = Result<(), ModelError>> + Send {
        self.inner.dtoh(b, out)
    }

    fn dtoh_at(
        &mut self,
        b: &Bytes,
        off_bytes: usize,
        out: &mut [u8],
    ) -> impl Future<Output = Result<(), ModelError>> + Send {
        self.inner.dtoh_at(b, off_bytes, out)
    }

    fn dtoh_batch(
        &mut self,
        pieces: &[(&Bytes, usize, usize)],
    ) -> impl Future<Output = Result<Vec<u8>, ModelError>> + Send {
        self.inner.dtoh_batch(pieces)
    }

    fn htod_batch(
        &mut self,
        writes: &[(&Bytes, usize, &[u8])],
    ) -> impl Future<Output = Result<(), ModelError>> + Send {
        // **不算出生**:写进已存在的块(dst 早已在账上)
        self.inner.htod_batch(writes)
    }

    fn swap(
        &mut self,
        dir: SwapDir,
        pieces: &[SwapPiece],
        conn: Box<dyn Connector + Send>,
    ) -> impl Future<Output = Result<Box<dyn Connector + Send>, ModelError>> + Send {
        let moves = swap_moves(dir, pieces);
        async move {
            let conn = self.inner.swap(dir, pieces, conn).await?;
            // 换层 ≠ 归还:只报 tier 迁移(失败不发事件 —— 整条 swap 要么全成要么响亮失败)
            self.send_events(&[], &[], &moves);
            Ok(conn)
        }
    }

    fn launch(&mut self, msg: LaunchMsg) -> impl Future<Output = Result<Bytes, ModelError>> + Send {
        self.note_node();
        self.inner.launch(msg)
    }

    fn sync(&mut self) -> impl Future<Output = Result<(), ModelError>> + Send {
        self.inner.sync()
    }

    fn free(&mut self, ids: &[u64]) -> impl Future<Output = Result<(), ModelError>> + Send {
        async move {
            // **缺口 4**:可池化 ⇒ 入池(不回驱动);其余真归还。池内那条不发 FreeEvent。
            let (rest, pooled) = self.pool_release_all(ids);
            if pooled > 0 {
                self.note_pool_released(pooled);
            }
            if !rest.is_empty() {
                self.inner.free(&rest).await?;
            }
            let frees: Vec<FreeEvent> =
                rest.iter().map(|id| FreeEvent { id: *id, bytes: 0 }).collect();
            self.send_events(&[], &frees, &[]);
            for id in ids {
                self.untrack(*id);
            }
            Ok(())
        }
    }

    fn htod_f32(
        &mut self,
        shape: &Shape,
        data: Vec<f32>,
        name: &str,
    ) -> impl Future<Output = Result<Bytes, ModelError>> + Send {
        let bytes = data.len() * std::mem::size_of::<f32>();
        async move {
            let b = self.inner.htod_f32(shape, data, name).await?;
            let mut ev = Vec::new();
            let class = self.ctx().class;
            self.note_birth(&b, bytes, class, name, &mut ev);
            self.send_events(&ev, &[], &[]);
            Ok(b)
        }
    }

    fn alloc_pinned(
        &mut self,
        bytes: usize,
        name: &str,
    ) -> impl Future<Output = Result<Box<dyn PinnedRegion + Send>, ModelError>> + Send {
        // pinned 是**宿主**面(服务端 pinned 池自成一本账,不在设备块账房)⇒ 不进设备账
        self.inner.alloc_pinned(bytes, name)
    }

    fn upload_pinned_on(
        &mut self,
        stream: UpStream,
        buf: Box<dyn PinnedRegion + Send>,
        dst: &Bytes,
        offset_bytes: usize,
        len_bytes: usize,
    ) -> impl Future<Output = Result<(), ModelError>> + Send {
        self.inner.upload_pinned_on(stream, buf, dst, offset_bytes, len_bytes)
    }

    fn upload_pinned(
        &mut self,
        buf: Box<dyn PinnedRegion + Send>,
        dst: &Bytes,
        offset_bytes: usize,
        len_bytes: usize,
    ) -> impl Future<Output = Result<(), ModelError>> + Send {
        self.inner.upload_pinned(buf, dst, offset_bytes, len_bytes)
    }

    fn write_block_f32(
        &mut self,
        dst: &Bytes,
        offset_elems: usize,
        data: &[f32],
    ) -> impl Future<Output = Result<(), ModelError>> + Send {
        self.inner.write_block_f32(dst, offset_elems, data)
    }

    fn write_block_f32_on(
        &mut self,
        stream: UpStream,
        dst: &Bytes,
        offset_elems: usize,
        data: &[f32],
    ) -> impl Future<Output = Result<(), ModelError>> + Send {
        self.inner.write_block_f32_on(stream, dst, offset_elems, data)
    }

    fn write_blocks_f32_on(
        &mut self,
        stream: UpStream,
        writes: &[(&Bytes, usize, &[f32])],
    ) -> impl Future<Output = Result<(), ModelError>> + Send {
        self.inner.write_blocks_f32_on(stream, writes)
    }

    fn write_blocks_f32(
        &mut self,
        writes: &[(&Bytes, usize, &[f32])],
    ) -> impl Future<Output = Result<(), ModelError>> + Send {
        self.inner.write_blocks_f32(writes)
    }

    fn memset_zero(
        &mut self,
        dst: &Bytes,
        len_bytes: usize,
    ) -> impl Future<Output = Result<(), ModelError>> + Send {
        // memset 是合法图节点 ⇒ 记节点(空窗判据)
        self.note_node();
        self.inner.memset_zero(dst, len_bytes)
    }

    fn upload_and_launch(
        &mut self,
        dst: &Bytes,
        chunks: Vec<UpChunk>,
        msg: LaunchMsg,
    ) -> impl Future<Output = Result<Bytes, ModelError>> + Send {
        self.note_node();
        self.inner.upload_and_launch(dst, chunks, msg)
    }

    fn launch_and_harvest(
        &mut self,
        msg: LaunchMsg,
        id: u64,
        want_bytes: usize,
    ) -> impl Future<Output = Result<Vec<u8>, ModelError>> + Send {
        self.note_node();
        self.inner.launch_and_harvest(msg, id, want_bytes)
    }

    fn block_ledger_dump(&mut self) -> impl Future<Output = Result<Vec<BlockLedgerRow>, ModelError>> + Send {
        self.inner.block_ledger_dump()
    }

    fn mem_pool_trim(&mut self) -> impl Future<Output = Result<u64, ModelError>> + Send {
        self.inner.mem_pool_trim()
    }

    fn copy_block_at(
        &mut self,
        src: &Bytes,
        src_off_bytes: usize,
        dst: &Bytes,
        dst_off_bytes: usize,
        len_bytes: usize,
    ) -> impl Future<Output = Result<(), ModelError>> + Send {
        self.note_node();
        self.inner.copy_block_at(src, src_off_bytes, dst, dst_off_bytes, len_bytes)
    }

    fn face_clone(&self) -> Option<Self> {
        // 克隆 = **同一本账 + 同一份上下文**(节点设一次全族生效)
        self.inner.face_clone().map(|c| Self {
            inner: c,
            ledger: self.ledger.clone(),
            ctx: self.ctx.clone(),
        })
    }

    fn sm_count(&mut self) -> impl Future<Output = Result<usize, ModelError>> + Send {
        self.inner.sm_count()
    }

    fn copy_batch(
        &mut self,
        copies: &[(Bytes, usize, Bytes, usize, usize)],
    ) -> impl Future<Output = Result<(), ModelError>> + Send {
        self.note_node();
        self.inner.copy_batch(copies)
    }

    fn memset_zero_at(
        &mut self,
        dst: &Bytes,
        offset_bytes: usize,
        len_bytes: usize,
    ) -> impl Future<Output = Result<(), ModelError>> + Send {
        self.note_node();
        self.inner.memset_zero_at(dst, offset_bytes, len_bytes)
    }

    fn loader_faces(&self, k: usize) -> Option<Vec<Self>> {
        self.inner.loader_faces(k).map(|v| {
            v.into_iter()
                .map(|c| Self { inner: c, ledger: self.ledger.clone(), ctx: self.ctx.clone() })
                .collect()
        })
    }
}

// ============================================================================
// §6 宏(调用点短;自动带 owner + 自动记账)
// ============================================================================

/// 清零分配(handle 形式):`alloc!(dev, AssetClass::Temp, Dtype::F16, 4096, "kv.workspace")`
/// ⇒ `dev.exec(..).await`。
///
/// 宏只做"构造 + exec",**不隐藏错误**(返回 `Result`;全仓"不许吞返回值"律)。
///
/// 名字沿革:`alloc!` 原属**裸 client**(3 参位置名)⇒ 2026-10-19 用户裁定
/// **`alloc!` 归 DeviceHandle**(5 参:class/dtype/elems/语义名 + 归属),
/// 裸 client 形式改名 `client_alloc!`(cuda tests/probes 已随改)。
#[macro_export]
macro_rules! alloc {
    ($dev:expr, $cls:expr, $dtype:expr, $elems:expr, $name:expr) => {
        $dev.exec($crate::handle::Command::alloc($cls, $dtype, $elems, $name)).await
    };
}

/// 归还(handle 形式):`free!(dev, id)` ⇒ `dev.exec(Command::free(id)).await`。
#[macro_export]
macro_rules! free {
    ($dev:expr, $id:expr) => {
        $dev.exec($crate::handle::Command::free($id)).await
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 骨架门 ①:**链式构造**(Launch 的 grid/block/shared_mem/args/name)
    #[test]
    fn command_builder_is_chainable() {
        let msg = LaunchMsg {
            kernel: owl_kernels::contract::KernelSource { name: "k".into(), source: String::new() },
            args: vec![],
            grid: (1, 1, 1),
            block: (1, 1, 1),
            shared_mem: 0,
            out_elems: 0,
        };
        let cmd = Command::Launch { msg, name: "k0".into() }
            .grid((8, 1, 1))
            .block((128, 1, 1))
            .shared_mem(4096)
            .args(vec![])
            .name("k1");
        assert_eq!(cmd.kind(), "launch");
        let Command::Launch { msg, name } = cmd else { panic!("应是 Launch") };
        assert_eq!(msg.grid, (8, 1, 1));
        assert_eq!(msg.block, (128, 1, 1));
        assert_eq!(msg.shared_mem, 4096);
        assert_eq!(name, "k1");
    }

    /// 骨架门 ②:构造糖 + `kind()`(打点/排障面)
    #[test]
    fn command_constructors_and_kind() {
        assert_eq!(Command::alloc(AssetClass::Temp, Dtype::F16, 8, "x").kind(), "alloc");
        assert_eq!(Command::free(3).kind(), "free");
        assert_eq!(Command::Sync.kind(), "sync");
        assert_eq!(
            Command::Htod { id: 1, off: 0, bytes: 8 }.kind(),
            "htod",
            "未接线命令也要能报名字"
        );
    }

    /// 骨架门 ③:owner 短名(报告面口径)
    #[test]
    fn owner_labels() {
        assert_eq!(Owner::Turn(7).label(), "turn:7");
        assert_eq!(Owner::Session(3).label(), "session:3");
        assert_eq!(Owner::Cache.label(), "cache");
        assert_eq!(Owner::System.label(), "system");
        assert_eq!(Owner::System, Owner::default_owner(), "缺省归属 = system");
    }

    /// 步 3 门 ④:**类别词表**(12 类;短名唯一 —— 报告面聚合键)
    #[test]
    fn asset_class_vocabulary_is_complete_and_labelled() {
        let all = [
            AssetClass::KvBlock,
            AssetClass::Floor,
            AssetClass::Elastic,
            AssetClass::Snap,
            AssetClass::Row,
            AssetClass::Meta,
            AssetClass::Index,
            AssetClass::Temp,
            AssetClass::Weight,
            AssetClass::Graph,
            AssetClass::Pool,
            AssetClass::Unclassified,
        ];
        let mut names: Vec<&str> = all.iter().map(|c| c.label()).collect();
        names.sort_unstable();
        let n = names.len();
        names.dedup();
        assert_eq!(names.len(), n, "短名必须唯一(报告面聚合键):{names:?}");
        assert_eq!(all.len(), 12);
    }

    /// 步 3 门 ⑤:**swap 片段按逻辑块聚合**(多片 → 每块一条 MoveEvent)
    #[test]
    fn swap_pieces_aggregate_per_block() {
        let piece = |block: u64, len: usize| SwapPiece {
            block,
            dev_off: 0,
            conn_off: 0,
            len,
            slot: crate::contract::ConnSlot(1),
        };
        let pieces = [piece(7, 4096), piece(9, 2048), piece(7, 2048)];
        let out = swap_moves(SwapDir::Out, &pieces);
        assert_eq!(out.len(), 2, "两块 ⇒ 两条(不是三板):{out:?}");
        assert_eq!(out[0].id, 7);
        assert_eq!(out[0].bytes, 6144, "同块片段求和");
        assert_eq!(out[0].dir, MoveDir::Out);
        assert_eq!(out[1].id, 9);
        let back = swap_moves(SwapDir::In, &pieces);
        assert_eq!(back[0].dir, MoveDir::In);
    }
}

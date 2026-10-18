//! **DeviceHandle:分配账 + 命令面**(设计卷 `docs/arch/device-命令面与分配账-设计与施工.md`)。
//!
//! 三件事同层同点(2026-10-19 用户定调):
//! ① **收编所有显存申请** —— engine/models 不再手动碰分配,但 **store 是纯状态机(不做 I/O)**,
//!    所以"摸显存"的只能是**客户端侧**的咽喉;钩子**挂在客户端**(不许挂到 server:跨线程)。
//! ② **接口倒置**:[`AllocLedger`] 由 **store 实现**;client 侧只依赖这个 trait
//!    (方案甲:`Rc<RefCell<dyn AllocLedger>>`,引擎单线程 ⇒ 无锁、无跨线程)。
//! ③ **命令面**:[`Command`](链式构造)+ [`DeviceHandle::exec`] 单条 / [`DeviceHandle::batch`] 多条
//!    + [`alloc!`](crate::alloc) / [`free!`](crate::free) 宏。
//!
//! # 纪律
//! - **本模块一行不改 `cuda` 里实现的 [`DeviceClient`]**:这里是**薄转发 + 钩子**;
//! - **钩子在客户端侧发**(调用者线程 = 引擎线程)⇒ 无跨线程、无锁;
//! - **下沉 ≠ 归还**:[`MoveEvent`] 只换层,只有 [`FreeEvent`] 才注销(账本律);
//! - **不许静默**:钩子失败计入 `hook_fail` 并 WARN;骨架期未接线的命令**响亮拒**(返回结构化错误)。
//!
//! # 骨架期边界(诚实标注)
//! 本轮只搭架子:`exec/batch` 已接线 `Alloc/AllocUninit/Free/Sync/Launch`;
//! `Htod/Dtoh/Swap` 的转发待接线(**响亮拒**,不是静默 no-op)。

use std::cell::RefCell;
use std::rc::Rc;

use owl_kernels::contract::{Bytes, Dtype, LaunchMsg};

use crate::contract::{DeviceClient, ModelError};

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
}

// ============================================================================
// §2 账本接口(由 store 实现;client 只依赖它 —— 依赖倒置)
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
}

/// **归还事件**(注销)。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FreeEvent {
    pub id: u64,
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
pub trait AllocLedger {
    /// 出借(登记 + 入账)。
    fn on_alloc(&mut self, ev: &AllocEvent);
    /// 归还(注销 + 出账)。
    fn on_free(&mut self, ev: &FreeEvent);
    /// 换层(只改 tier/owner,不注销)。
    fn on_move(&mut self, ev: &MoveEvent);
}

// ============================================================================
// §3 命令面(Command + 链式构造)
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
    Launch {
        msg: LaunchMsg,
        name: String,
    },
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
/// ⚠️ `Launch` 也**产出块**(输出块)—— 契约里 `launch` 的返回就是 `Bytes`,
/// 不许丢(全仓律 #1)⇒ 统一走 [`Outcome::Buf`]。
#[derive(Clone, Debug)]
pub enum Outcome {
    /// 产出一个设备块(`Alloc` / `AllocUninit` / `Launch` 的输出块)。
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
// §4 DeviceHandle:薄转发 + 钩子 + 命令面
// ============================================================================

/// **设备句柄**(engine/models 唯一能碰的设备面)。
///
/// - `ledger = None` ⇒ **纯转发**(迁移态:零行为变化,便于灰度);
/// - `ledger = Some(..)` ⇒ 每次分配/归还/换层发事件(客户端侧,同线程);
/// - `owner` = 当前归属上下文(由生命周期节点设置:N1/N4/N5)。
pub struct DeviceHandle<D: DeviceClient> {
    inner: D,
    ledger: Option<Rc<RefCell<dyn AllocLedger>>>,
    owner: Owner,
    /// 记账失败计数(**不许静默**:失败也看得见)。
    hook_fail: u64,
}

impl<D: DeviceClient> DeviceHandle<D> {
    /// 构造(唯一入口)。`ledger = None` = 纯转发(迁移态)。
    pub fn new(inner: D, ledger: Option<Rc<RefCell<dyn AllocLedger>>>) -> Self {
        Self { inner, ledger, owner: Owner::System, hook_fail: 0 }
    }

    /// **归属上下文**(生命周期节点调用:N1 `Turn(t)` / N5 `Session(s)` / N4 复位 `System`)。
    pub fn set_owner(&mut self, o: Owner) {
        self.owner = o;
    }

    /// 当前归属。
    pub fn owner(&self) -> Owner {
        self.owner
    }

    /// 记账失败计数(诊断面;非零 = 有钩子调用失败过)。
    pub fn hook_fail(&self) -> u64 {
        self.hook_fail
    }

    /// 只读直通(查询类不走命令面、不记账)。
    pub fn client(&self) -> &D {
        &self.inner
    }

    /// 可变直通(**仅限不产生分配的用途**;封印见设计卷 §六)。
    pub fn client_mut(&mut self) -> &mut D {
        &mut self.inner
    }

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
        for (i, cmd) in cmds.into_iter().enumerate() {
            segments += 1;
            let r = self.run_one(cmd, &mut events, &mut frees, &mut moves).await;
            if r.is_err() && first_err.is_none() {
                first_err = Some(i);
            }
            outcomes.push(r);
        }
        // 收口:入账(失败条已在上方被拦 ⇒ 这里全是成功事件)
        self.commit_events(&events, &frees, &moves);
        Ok(BatchOutcome { outcomes, first_err, segments })
    }

    /// 单条执行(内部;把事件塞进本批的缓冲)。
    async fn run_one(
        &mut self,
        cmd: Command,
        events: &mut Vec<AllocEvent>,
        frees: &mut Vec<FreeEvent>,
        _moves: &mut Vec<MoveEvent>, // 骨架期无命令产生 MoveEvent(Htod/Dtoh 待接线)
    ) -> Result<Outcome, ModelError> {
        match cmd {
            Command::Alloc { class, dtype, elems, name } => {
                let b = self.inner.alloc(dtype, elems, &name).await?;
                events.push(AllocEvent {
                    id: b.id,
                    bytes: b.len * dtype.size_bytes(),
                    name,
                    class,
                    owner: self.owner,
                });
                Ok(Outcome::Buf(b))
            }
            Command::AllocUninit { class, dtype, elems, name } => {
                let b = self.inner.alloc_uninit(dtype, elems, &name).await?;
                events.push(AllocEvent {
                    id: b.id,
                    bytes: b.len * dtype.size_bytes(),
                    name,
                    class,
                    owner: self.owner,
                });
                Ok(Outcome::Buf(b))
            }
            Command::Free { ids } => {
                self.inner.free(&ids).await?;
                for id in ids {
                    // 字节数在账本侧反查(FreeEvent 只带 id);此处留 0 = 由账本按登记扣
                    frees.push(FreeEvent { id, bytes: 0 });
                }
                Ok(Outcome::Done)
            }
            Command::Launch { msg, .. } => {
                // 输出块不许丢(律 #1):原样带出
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

    /// 收口入账(钩子调用点;**失败必须可见**,但不中断主流程)。
    fn commit_events(
        &mut self,
        events: &[AllocEvent],
        frees: &[FreeEvent],
        moves: &[MoveEvent],
    ) {
        let Some(ledger) = self.ledger.clone() else { return }; // 纯转发态
        let mut l = match ledger.try_borrow_mut() {
            Ok(g) => g,
            Err(_) => {
                // 重入 = 实现侧违约(设计卷 §十 风险 1);响亮计数,不 panic
                self.hook_fail += 1;
                // iface 无日志依赖 ⇒ 裸 eprintln(机读面 = `hook_fail` 计数)
                eprintln!(
                    "[device-handle] 账本借用重入(实现侧违约)⇒ 本批 {} 条事件未入账",
                    events.len() + frees.len() + moves.len()
                );
                return;
            }
        };
        for ev in events {
            l.on_alloc(ev);
        }
        for ev in frees {
            l.on_free(ev);
        }
        for ev in moves {
            l.on_move(ev);
        }
    }
}

// ============================================================================
// §5 宏(调用点短;自动带 owner + 自动记账)
// ============================================================================

/// 清零分配(handle 形式):`dev_alloc!(dev, AssetClass::Temp, Dtype::F16, 4096, "kv.workspace")`
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
}

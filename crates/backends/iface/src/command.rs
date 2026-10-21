//! **设备命令面:SDK ↔ server 的唯一单位**(2026-10-21 用户裁决)。
//!
//! 旧形态 = `DeviceClient` 把每个能力展开成一个方法(39 个),后端逐个实现、
//! 每条命令一次信箱往返 —— 调用点与后端**各执一套"命令"**(方法名 vs
//! `cuda::Command` 枚举),批处理在后端结构上不可达。
//!
//! 新形态 = **一个 `Command` = 一次设备操作**,契约只留两条执行入口:
//!
//! ```text
//! DeviceClient::exec(cmd)        —— 单条
//! DeviceClient::exec_batch(cmds) —— 一条信箱消息交给后端(真正的批)
//! ```
//!
//! 三条纪律:
//! 1. **后端只认命令**,不认具名方法 —— 具名方法(alloc/launch/…)是
//!    [`crate::contract::DeviceClient`] 上的**缺省糖**,内部构造本文件的
//!    `Command` 调 `exec`(单一真源,调用点零改动);
//! 2. **命令带全语义**(dtype/shape/偏移/流),后端不需要再回问客户端;
//! 3. **结果统一为 [`Outcome`]**:无泛型的窄和类型,读错变体 = 响亮错误
//!    (不是 `as_*` 静默 0)。
//!
//! 词汇表当前 = **核心集**(用户裁决:先做 alloc/free/launch/htod/dtoh/sync/graph)。
//! 其余能力(swap/peer/复合口/池账…)仍走具名方法,后续按同一张表补齐 ——
//! 新增一条 = 本文件加一个变体 + 后端 `exec` 加一臂,不新增方法。

use crate::contract::{Bytes, Dtype, GraphId, LaunchMsg, ModelError, Shape, UpStream};

/// **一条设备命令**。
///
/// 注:不含流 id 的地方走 [`UpStream`](语义流),与旧契约一致 ——
/// 流的物理路由是 server 内部策略,客户端只说"这条搬运属于哪个面"。
pub enum Command {
    /// 分配。`zero = false` = 免清零(scratch;kernel 全量覆写)。
    Alloc {
        dtype: Dtype,
        elems: usize,
        zero: bool,
        name: String,
    },
    /// 归还(server 侧归池;复用发生在 server)。
    Free { ids: Vec<u64> },
    /// host → device(整块)。
    Htod {
        dtype: Dtype,
        shape: Shape,
        src: Vec<u8>,
        stream: UpStream,
        name: String,
    },
    /// device → host(块内偏移读;`off_bytes = 0` 即整块前缀读)。
    Dtoh {
        id: u64,
        off_bytes: usize,
        want_bytes: usize,
    },
    /// kernel 发射(COMPUTE 流)。
    Launch { msg: LaunchMsg },
    /// 全设备栅栏(三条流此前工作落定)。
    Sync,

    // ── 图族(捕获/回放/退休)────────────────────────────────────────
    /// 进入捕获窗(不可嵌套;窗内只放行 Launch/Alloc/Memset/Free/收尾)。
    GraphBegin,
    /// 结束捕获并实例化(空窗拒绝);返回图 id。
    GraphEnd,
    /// 图回放(流序异步提交)。
    GraphLaunch { graph: GraphId },
    /// 捕获窗作废(窗内自产块整批回收 —— 后端自知语义)。
    GraphAbort,
    /// 销毁已登记图(排空回放 → 解租约 → slab 退休)。
    GraphDestroy { graph: GraphId },
}

impl Command {
    /// 命令类名(打点/日志/发射账维度;`Command` 无 `Debug` 派生 —— 这是排障入口)。
    pub fn kind(&self) -> &'static str {
        match self {
            Command::Alloc { .. } => "alloc",
            Command::Free { .. } => "free",
            Command::Htod { .. } => "htod",
            Command::Dtoh { .. } => "dtoh",
            Command::Launch { .. } => "launch",
            Command::Sync => "sync",
            Command::GraphBegin => "graph_begin",
            Command::GraphEnd => "graph_end",
            Command::GraphLaunch { .. } => "graph_launch",
            Command::GraphAbort => "graph_abort",
            Command::GraphDestroy { .. } => "graph_destroy",
        }
    }
}

/// **单条命令的结果**(窄和类型;每个变体带自己的真类型)。
///
/// 为什么不是泛型返回:`exec_batch` 要装**异质**命令的结果,类型擦除是批的
/// 前提。这里的代价由 [`Outcome`] 的强类型访问器补齐 —— 读错变体返回
/// `Err`(带实得变体名),绝不静默取默认值。
#[derive(Clone, Debug)]
pub enum Outcome {
    /// 无返回值(Free / Sync / 图三件套)。
    Done,
    /// 产出一个块句柄(Alloc / Htod / Launch 的输出块)。
    Buf(Bytes),
    /// 一段主机字节(Dtoh)。
    Bytes(Vec<u8>),
    /// 图 id(GraphEnd)。
    Graph(GraphId),
}

impl Outcome {
    /// 取块句柄(变体不符 ⇒ 响亮错)。
    pub fn into_buf(self, what: &str) -> Result<Bytes, ModelError> {
        match self {
            Outcome::Buf(b) => Ok(b),
            other => Err(mismatch(what, "Buf", &other)),
        }
    }

    /// 取主机字节。
    pub fn into_bytes(self, what: &str) -> Result<Vec<u8>, ModelError> {
        match self {
            Outcome::Bytes(b) => Ok(b),
            other => Err(mismatch(what, "Bytes", &other)),
        }
    }

    /// 取图 id。
    pub fn into_graph(self, what: &str) -> Result<GraphId, ModelError> {
        match self {
            Outcome::Graph(g) => Ok(g),
            other => Err(mismatch(what, "Graph", &other)),
        }
    }

    /// 断言"无返回值"。
    pub fn into_done(self, what: &str) -> Result<(), ModelError> {
        match self {
            Outcome::Done => Ok(()),
            other => Err(mismatch(what, "Done", &other)),
        }
    }

    /// 变体短名(错误信息/打点用)。
    pub fn kind(&self) -> &'static str {
        match self {
            Outcome::Done => "Done",
            Outcome::Buf(_) => "Buf",
            Outcome::Bytes(_) => "Bytes",
            Outcome::Graph(_) => "Graph",
        }
    }
}

fn mismatch(what: &str, want: &str, got: &Outcome) -> ModelError {
    ModelError::Msg(format!(
        "{what}: 结果变体不符 —— 期望 {want},实得 {}",
        got.kind()
    ))
}

/// **一批命令的结果**(`exec_batch` 的回执)。
#[derive(Clone, Debug)]
pub struct BatchOutcome {
    /// 逐条结果(**下标与投递顺序严格对应**)。
    pub outcomes: Vec<Result<Outcome, ModelError>>,
    /// 首个失败下标(None = 全成)。**已投递的不回滚**(命令流语义)。
    pub first_err: Option<usize>,
    /// 实际投递段数(批 = 1;逐条 = N。打点:批税对比的唯一判据)。
    pub segments: usize,
}

impl BatchOutcome {
    /// 空批(全成)。
    pub fn empty() -> Self {
        Self { outcomes: Vec::new(), first_err: None, segments: 0 }
    }

    /// 全成?
    pub fn all_ok(&self) -> bool {
        self.first_err.is_none()
    }

    /// 从逐条结果收口(segments 由调用方给:CPU 面逐条 = N,GPU 原生批 = 1)。
    pub fn collect(outcomes: Vec<Result<Outcome, ModelError>>, segments: usize) -> Self {
        let first_err = outcomes.iter().position(|r| r.is_err());
        Self { outcomes, first_err, segments }
    }
}

//! Client:对 GPU server 的**能力期望**——cuda 包照此实现。
//!
//! 本模块回答:"上层(Tensor/layer/解释器)需要 server 提供什么?"
//! 这份清单就是 cuda server 的实现任务书。client 不感知 server 内部
//! (池/租约/相位/哨兵如何做,是 server 自己的事);client 只消费句柄。
//!
//! 两层:
//! - [`GpuFace`]:同步能力面(actor 线程上的执行原语)——解释器消费;
//! - [`AsyncClient`]:异步门面(Tokio)——run/capture/replay/sync/close;
//!   server 侧以专用线程 + 命令队列实现(async-runtime.md §二)。


// ============================================================================
// §1 句柄词汇(client 视角的 server 产物;不透明,不可克隆内容)
// ============================================================================

/// 池块句柄(字节面;server 签发;drop = 延迟归还语义由 server 自持)
#[derive(Clone, Debug)]
#[allow(dead_code)] // 句柄 id:server 记账用,client 侧暂不读
pub struct Bytes(pub(crate) u64);

/// 图句柄(捕获产物;烘焙后的可回放计划)
#[derive(Clone, Debug)]
#[allow(dead_code)] // 句柄 id:server 记账用,client 侧暂不读
pub struct Graph(pub(crate) u64);

/// kernel 句柄(server 侧 nvrtc 编译产物;装载经 load_kernel)
#[derive(Clone, Debug)]
#[allow(dead_code)] // 句柄 id:server 记账用,client 侧暂不读
pub struct Kernel(pub(crate) u64);

/// KV 上下文:动态依赖的参数形态(每步由 runner 构造传入)。
/// manager 本体是静态依赖(layer 构造捕获);Ctx 只是"这一步用哪些格"。
#[derive(Clone, Debug)]
pub struct KvCtx {
    pub step: u64,
    pub slots: Vec<u32>,
}

// ============================================================================
// §2 同步能力面(actor 线程上的原语;解释器消费)
// ============================================================================
//
// 这些方法的**全部实现都在 server/actor 线程**;本 trait 是 cuda 包对
// owl-models 的能力承诺。阻塞(等待/拷贝)合法——调用者就是 actor 线程。

pub trait GpuFace {
    // 分配(字节;清零;Capturing 相允许 = 捕获安全分配)
    // PSEUDO: fn malloc(&mut self, bytes: usize) -> Result<Bytes, ModelError>;

    // host → device 写入式分配(pinned 码头 + 池流 async+sync;契约 3)
    // PSEUDO: fn htod(&mut self, src: &[u8]) -> Result<Bytes, ModelError>;

    // device → host 收割(阻塞等待 + 拷贝;完成 ready)
    // PSEUDO: fn dtoh(&mut self, b: &Bytes, out: &mut [u8]) -> Result<(), ModelError>;

    // kernel 发射(提交即回:发射成功即返回,不等 GPU)
    // PSEUDO: fn launch(&mut self, kernel: &Kernel, args: &LaunchArgs) -> Result<(), ModelError>;

    // 全设备同步
    // PSEUDO: fn sync(&mut self) -> Result<(), ModelError>;

    // kernel 装载(nvrtc 源码 → 句柄;A2 期:LoadKernel 命令)
    // PSEUDO: fn load_kernel(&mut self, src: &str, names: &[&str]) -> Result<Vec<Kernel>, ModelError>;

    // 捕获事务(原子;窗内发射经 launch 通道落 capture 流)
    // PSEUDO: fn capture<F>(&mut self, step: F) -> Result<Graph, ModelError>
    // PSEUDO: where F: FnOnce(&mut Self) -> Result<(), ModelError>;

    // 图回放(租约校验 + launch;提交即回)
    // PSEUDO: fn replay(&mut self, graph: &Graph) -> Result<(), ModelError>;
}

// ============================================================================
// §3 异步门面(生产入口;Tokio 面零阻塞)
// ============================================================================
//
/// 对 server 的异步期望:专用 GPU 线程 + 命令队列(async-runtime.md §二)。
/// 模型层的工厂/forward 全部同步;只有这里的 eval/to_host/sync 是 async。
pub struct AsyncClient {
    /* proto: ProtocolClient —— 协议封装在 server 侧内部;client 不感知 */
    _priv: (),
}

impl AsyncClient {
    // 启动 server 并握手(uuid 钉卡;pool_bytes 构造参数指定)
    // PSEUDO: pub async fn spawn(uuid: &str, pool_bytes: u64) -> Result<Self, ModelError>;

    // 评述:节目单 → server 逐节点解释(档一:提交即回)。
    // 返回收割句柄(对任意中间节点 to_host)。
    // PSEUDO: pub async fn eval(&self, t: &crate::tensor::TensorOps) -> Result<EvalHandle, ModelError>;

    // 收割:同步等待 + D2H(全链唯一拿数据的地方;完成 ready)
    // PSEUDO: pub async fn to_host(&self, t: &crate::tensor::TensorOps) -> Result<Vec<u8>, ModelError>;

    // 捕获事务(原子)→ 图句柄
    // PSEUDO: pub async fn capture(&self, step: CaptureStep) -> Result<Graph, ModelError>;

    // 图回放(提交即回)
    // PSEUDO: pub async fn replay(&self, graph: Graph) -> Result<(), ModelError>;

    // 全设备同步
    // PSEUDO: pub async fn sync(&self) -> Result<(), ModelError>;

    // 关机
    // PSEUDO: pub async fn close(&self) -> Result<(), ModelError>;
}

/// 评述产物:节点 → 收割/数据访问的路由(实施期定形态)
pub struct EvalHandle {
    _priv: (),
}

// ============================================================================
// §4 对 server 的错误面期望(见 error.rs ModelError 执行期段)
// ============================================================================
//
// server 必须能结构化表达:PoolExhausted / DeadBlock / CaptureViolation /
// ServerClosed / Msg。禁止 panic 跨线;禁止错误静默吞并。

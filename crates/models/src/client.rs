//! Client:对 GPU server 的**能力期望**(五原语;owl-cuda 照此实现)。
//!
//! 职责定稿(2026-09-23 四层架构):
//! - server(硬件层)= 哑执行器:只认 LaunchMsg / Alloc / Htod / Dtoh / Sync;
//! - 解释器 = 把声明树翻译成原语消息;
//! - 上层(TensorOps/Module)= 纯声明,零设备知识。
//!
//! 签名形态:RPITIT(`-> impl Future + Send`),无 async_trait 依赖;
//! eval 为泛型静态分发,无 dyn。

/// 图身份证(server 签发;graph_end 成功后可 graph_launch 重放)
pub type GraphId = u64;

use crate::error::ModelError;
use crate::plan::Op;
use crate::shape::Shape;
use crate::tensor::TensorOps;
use std::future::Future;
use std::pin::Pin;

/// 池块句柄:server 签发的身份证(id → server 账房 → 显存)。
#[derive(Clone, Debug)]
pub struct Bytes {
    pub id: u64,
    /// 元素数(f32)
    pub len: usize,
}

impl Bytes {
    pub fn new(id: u64, len: usize) -> Self {
        Self { id, len }
    }
}

/// KV 动态上下文:每步由 runner 构造。
#[derive(Clone, Debug)]
pub struct KvCtx {
    pub step: u64,
    pub slots: Vec<u32>,
}

/// kernel 描述:入口名 + 源码。后端按 (源码哈希, 名) 懒编译缓存。
#[derive(Clone, Debug)]
pub struct KernelSpec {
    pub name: String,
    pub source: String,
}

/// 发射参数槽(有序;与 kernel 签名严格对位)
/// 类型化:标量按 kernel 形参宽度入槽(CUDA 参数空间自然对齐,
/// 8 字节槽顶 4 字节形参会错位读參)
#[derive(Clone, Debug)]
pub enum Arg {
    Block { id: u64 },
    U64(u64),
    I32(i32),
    F32(f32),
}

/// 发射消息
pub struct LaunchMsg {
    pub kernel: KernelSpec,
    pub args: Vec<Arg>,
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub shared_mem: u32,
    pub out_elems: usize,
}

/// 设备客户端能力契约(五原语 + 流管理;全异步)。
///
/// 流语义:server 按业界三流模型固定路由(H2D/COMPUTE/D2H,客户端不可
/// 自创也不可见;多并发靠算子维 batching,不靠多流)——
/// htod→H2D,launch/alloc/graph→COMPUTE,dtoh→D2H,sync 排空全部。
/// 流内保序(依赖维),流间并发(传输/计算重叠维)。跨流依赖(事件边)
/// 待服务端扩展。
///
/// **回执语义分级**(Future resolve 时机):
/// - `alloc` / `launch`:Ok = 提交成功 + 账房登记(fire-and-forget;
///   数据正确性由同流保序保证,GPU 侧错误 sticky 延迟暴露,在下次
///   `dtoh`/`sync` 收割)
/// - `htod` / `dtoh`:Ok = GPU 真完成(pinned 码头生命周期/数据收割
///   要求真实完成点;完成通知走 host 回调 cuLaunchHostFunc)
/// - `sync`:栅栏,该流此前所有工作全部落定
pub trait DeviceClient: Send {
    /// 图捕获开始:进入捕获模式(此后 Launch 进图;
    /// Alloc 从捕获 slab 切块,零 cudaMalloc)。护栏:
    /// - 不可嵌套/并发捕获;捕获期 Htod/Dtoh/Sync/GraphLaunch 拒绝
    /// - 数据须在图外先物化(Block 叶子)—— warmup 契约:先 eager 跑
    ///   同一声明树,验证逻辑正确后再捕获
    fn graph_begin(&mut self) -> impl Future<Output = Result<(), ModelError>> + Send;
    /// 图捕获结束:实例化并登记,返回图 id(空捕获窗拒绝 —— 发射数须 > 0)
    fn graph_end(&mut self) -> impl Future<Output = Result<GraphId, ModelError>> + Send;
    /// 图重放(流序异步提交;写入捕获时的同一批池块 —— 块只增不减,
    /// 指针稳定,由 server 账房保证)
    fn graph_launch(
        &mut self,
        graph: GraphId,
    ) -> impl Future<Output = Result<(), ModelError>> + Send;
    fn alloc(&mut self, n_bytes: usize)
        -> impl Future<Output = Result<Bytes, ModelError>> + Send;
    fn htod(
        &mut self,
        dtype: crate::tensor::Dtype,
        shape: &Shape,
        src: &[u8],
    ) -> impl Future<Output = Result<Bytes, ModelError>> + Send;
    fn dtoh(&mut self, b: &Bytes, out: &mut [u8])
        -> impl Future<Output = Result<(), ModelError>> + Send;
    fn launch(&mut self, msg: LaunchMsg)
        -> impl Future<Output = Result<Bytes, ModelError>> + Send;
    fn sync(&mut self) -> impl Future<Output = Result<(), ModelError>> + Send;
}

/// 声明树求值:自叶向根,逐节点翻译为原语调用。
pub fn eval<'a, D>(
    t: &'a TensorOps,
    face: &'a mut D,
) -> Pin<Box<dyn Future<Output = Result<Bytes, ModelError>> + Send + 'a>>
where
    D: DeviceClient + 'a,
{
    Box::pin(_eval(t, face))
}

async fn _eval<D: DeviceClient>(t: &TensorOps, face: &mut D) -> Result<Bytes, ModelError> {
    if let Some(e) = &t.err {
        return Err(ModelError::Msg(format!(
            "[毒值落地 @depth {}] {}",
            e.at_depth, e.detail
        )));
    }
    let mut ins: Vec<Bytes> = Vec::with_capacity(t.parents.len());
    for p in &t.parents {
        ins.push(eval(p, face).await?);
    }

    let dtype = t.dtype;
    let shape = t.shape.clone();
    let n_bytes = shape.iter().product::<usize>() * dtype.size_bytes();

    match &t.op {
        Op::Htod { bytes } => face.htod(dtype, &shape, bytes).await,
        Op::Zeros => face.alloc(n_bytes).await,
        Op::Block { id } => Ok(Bytes { id: *id, len: 0 }),
        Op::Add => {
            let out = face.alloc(n_bytes).await?;
            let msg = crate::actions::lower_add(&ins, &out);
            face.launch(msg).await?;
            Ok(out)
        }
        Op::Silu => {
            let out = face.alloc(n_bytes).await?;
            let msg = crate::actions::lower_silu(&ins, &out);
            face.launch(msg).await?;
            Ok(out)
        }
        Op::Matmul => {
            let (m, n) = (shape[0], shape[1]);
            let k = ins[0].len;
            let out = face.alloc(m * n * dtype.size_bytes()).await?;
            let msg = crate::actions::lower_matmul(&ins, &out, m, k, n);
            face.launch(msg).await?;
            Ok(out)
        }
        Op::Rmsnorm { eps, w_off } => {
            let out = face.alloc(n_bytes).await?;
            let msg = crate::actions::lower_rmsnorm(&ins, *eps, *w_off, &out);
            face.launch(msg).await?;
            Ok(out)
        }
        Op::Kernel { kernel } => {
            let out = face.alloc(n_bytes).await?;
            let msg = crate::actions::lower_kernel(kernel, &t.args, &ins, &out);
            face.launch(msg).await?;
            Ok(out)
        }
        Op::SlotWrite => Ok(ins[0].clone()),
        other => Err(ModelError::Msg(format!("eval 未覆盖: {other:?}"))),
    }
}

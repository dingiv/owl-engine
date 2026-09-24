//! Client:对 GPU server 的**能力期望**(五原语;owl-cuda 照此实现)。
//!
//! 职责定稿(2026-09-23 四层架构):
//! - server(硬件层)= 哑执行器:只认 LaunchMsg / Alloc / Htod / Dtoh / Sync;
//! - 解释器 = 把声明树翻译成原语消息;
//! - 上层(TensorOps/Module)= 纯声明,零设备知识。
//!
//! 签名形态:RPITIT(`-> impl Future + Send`),无 async_trait 依赖;
//! eval 为泛型静态分发,无 dyn。

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

/// 设备客户端能力契约(五原语;全异步)。
pub trait DeviceClient: Send {
    fn alloc(&mut self, n_bytes: usize) -> impl Future<Output = Result<Bytes, ModelError>> + Send;
    fn htod(
        &mut self,
        dtype: crate::dtype::Dtype,
        shape: &Shape,
        src: &[u8],
    ) -> impl Future<Output = Result<Bytes, ModelError>> + Send;
    fn dtoh(
        &mut self,
        b: &Bytes,
        out: &mut [u8],
    ) -> impl Future<Output = Result<(), ModelError>> + Send;
    fn launch(
        &mut self,
        msg: LaunchMsg,
    ) -> impl Future<Output = Result<Bytes, ModelError>> + Send;
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
        Op::SlotWrite => Ok(ins[0].clone()),
        Op::Kernel { .. } => Err(ModelError::Msg(
            "Op::Kernel: 需 server 侧 load+launch(动作表二期)".to_string(),
        )),
        other => Err(ModelError::Msg(format!("eval 未覆盖: {other:?}"))),
    }
}

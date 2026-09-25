//! 解释器:同一份声明,多种执行(每后端一个解释器)。
//!
//! 解释器是对上层模型层(layer)的封装:layer 只写声明(TensorOps 链),
//! 解释器负责把声明翻译成后端调用(alloc / lower / launch)——算子的
//! 落地逻辑由解释器兜住,上层零后端知识。毒值在解释器边界落地
//! (结构化报错 + depth 归因)。
//!
//! | 解释器 | 形态 | 后端 | 场景 |
//! |---|---|---|---|
//! | [`eval`] | 异步(face 注入,泛型静态分发) | `owl-cpu::CpuFace`(CPU:进程内同步直调,无 server)/ `owl-cuda::GpuClient`(GPU:actor server) | 引擎主路径 |
//! | [`reduce`] + [`CpuInterpreter`] | 同步(纯 Rust 朴素归约) | 无后端(参考实现) | 测试/对拍基准 |
//!
//! 后端契约 = `owl-iface::contract::DeviceClient`(线格式同源);
//! 解释器本身不含后端代码 —— 后端经 face 泛型注入。
//! 毒值传播契约:构造期违约随链流动,`is_poisoned()` 可查,边界收割。

use crate::error::ModelError;
use crate::loader::{Layout, LoaderOps, WeightSource};
use crate::plan::Op;
use crate::shape::{numel, Shape};
use crate::tensor::{Dtype, TensorOps};
use crate::client::{Bytes, DeviceClient};
use std::future::Future;
use std::pin::Pin;

// ============================================================================
// §1 异步解释器:eval(层入口)/ eval_ops(树归约;引擎主路径)
// ============================================================================

/// 计算执行(层级入口):驱动层的 `Module::forward` 声明并归约。
/// 解释器自己调用层钩子并为它传递 ctx —— 使用者只给层与输入。
///
/// ```rust,ignore
/// let out = eval(&mlp, &xs, face, KernelCtx { tokens: 1 }).await?;
/// ```
pub async fn eval<M, D>(
    layer: &M,
    xs: &TensorOps,
    face: &mut D,
    ctx: &crate::module::KernelCtx,
) -> Result<Bytes, ModelError>
where
    M: crate::module::Module,
    D: crate::client::DeviceClient,
{
    let ops = layer.forward(xs, ctx);   // 层产出声明(ctx 引用透传)
    eval_ops(ops.step(), face).await    // 解释器归约
}

/// 声明树求值(内部件;供手工子树/非层根的归约):自叶向根,
/// 逐节点翻译为原语调用(face = 后端句柄)。
pub fn eval_ops<'a, D>(
    t: &'a TensorOps,
    face: &'a mut D,
) -> Pin<Box<dyn Future<Output = Result<Bytes, ModelError>> + Send + 'a>>
where
    D: crate::client::DeviceClient + 'a,
{
    Box::pin(_eval(t, face))
}

async fn _eval<D: crate::client::DeviceClient>(
    t: &TensorOps,
    face: &mut D,
) -> Result<Bytes, ModelError> {
    if let Some(e) = &t.err {
        return Err(ModelError::Msg(format!(
            "[毒值落地 @depth {}] {}",
            e.at_depth, e.detail
        )));
    }
    let mut ins: Vec<Bytes> = Vec::with_capacity(t.parents.len());
    for p in &t.parents {
        ins.push(eval_ops(p, face).await?);
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
        Op::Mul => {
            let out = face.alloc(n_bytes).await?;
            let msg = crate::actions::lower_mul(&ins, &out);
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
            let cols = shape.last().copied().unwrap_or(0);
            let rows = shape.iter().product::<usize>() / cols.max(1);
            let msg = crate::actions::lower_rmsnorm(&ins, *eps, *w_off, &out, rows, cols);
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

// ============================================================================
// §1.5 装载解释器:eval_load(识别 LoaderOps 指令;装载域的执行能力)
// ============================================================================

/// 装载执行(层级入口):驱动层的 `Loadable::layout` 需求并物化。
/// 解释器自己调用层钩子并为它传递 LoaderCtx —— 使用者只给层与源。
///
/// ```rust,ignore
/// eval_load(&mlp, face, &src, LoaderCtx::default()).await?;
/// ```
pub async fn eval_load<M, D, S>(
    layer: &M,
    face: &mut D,
    src: &S,
    ctx: &crate::loader::LoaderCtx,
) -> Result<(), ModelError>
where
    M: crate::loader::Loadable,
    D: DeviceClient,
    S: WeightSource + ?Sized,
{
    let want = layer.layout(ctx);   // 层产出需求清单(ctx 引用透传)
    eval_want(&want, face, src).await
}

/// 需求清单求值(内部件):按清单从源取数 → 布局变换 → face 物化 →
/// **经 Want.sink 自动填回容器空包**(mount 已被吸收)。
/// 缺键/长度不符/htod 失败 → 结构化 Err(带槽键归因)。
async fn eval_want<D: DeviceClient, S: WeightSource + ?Sized>(
    want: &LoaderOps,
    face: &mut D,
    src: &S,
) -> Result<(), ModelError> {
    use crate::loader::f32b;
    for w in want.wants() {
        let data = src.get(w.key).ok_or_else(|| {
            ModelError::Msg(format!("Weight '{}': 数据源缺键", w.key))
        })?;
        let n: usize = w.shape.iter().product();
        let (shape, bytes) = match w.layout {
            Layout::Direct => {
                if data.len() != n {
                    return Err(ModelError::Msg(format!(
                        "Weight '{}': 元素 {} != shape {:?}({n})",
                        w.key,
                        data.len(),
                        w.shape
                    )));
                }
                (w.shape.clone(), f32b(data))
            }
            Layout::Transposed => {
                // 目标 [cols, rows];源 [rows, cols](行主序)
                let (cols, rows) = (w.shape[0], w.shape[1]);
                if data.len() != n {
                    return Err(ModelError::Msg(format!(
                        "Weight '{}': 元素 {} != 源形状 {rows}×{cols}",
                        w.key,
                        data.len()
                    )));
                }
                let mut t = vec![0.0f32; n];
                for r in 0..rows {
                    for c in 0..cols {
                        t[c * rows + r] = data[r * cols + c];
                    }
                }
                (w.shape.clone(), f32b(&t))
            }
        };
        let b = face
            .htod(w.dtype, &shape, &bytes)
            .await
            .map_err(|e| ModelError::Msg(format!("Weight '{}': {e}", w.key)))?;
        w.sink.deliver(crate::client::Bytes::new(b.id, n));
    }
    Ok(())
}

// ============================================================================
// §2 同步解释器 trait + 归约驱动(CPU 参考;测试/对拍)
// ============================================================================

/// 执行单节点:输入已就绪(归约序保证),产出输出。
pub trait Interpreter {
    /// Htod:host 字节入块
    fn htod(&mut self, dtype: Dtype, shape: &Shape, bytes: &[u8]) -> Result<Value, ModelError>;
    /// Zeros:清零分配
    fn zeros(&mut self, dtype: Dtype, shape: &Shape) -> Result<Value, ModelError>;
    /// Matmul:[m,k]×[k,n]
    fn matmul(&mut self, a: &Value, b: &Value, dtype: Dtype, shape: &Shape) -> Result<Value, ModelError>;
    /// Add(同形)
    fn add(&mut self, a: &Value, b: &Value) -> Result<Value, ModelError>;
    /// Mul(同形逐元素乘;MLP 门控/输出门)
    fn mul(&mut self, a: &Value, b: &Value) -> Result<Value, ModelError>;
    /// Silu
    fn silu(&mut self, x: &Value) -> Result<Value, ModelError>;
    /// Rmsnorm(parents = [x, alpha];per-channel alpha([cols] 广播);w_off = ×(1+w))
    fn rmsnorm(
        &mut self,
        x: &Value,
        alpha: &Value,
        eps: f32,
        w_off: bool,
    ) -> Result<Value, ModelError>;
    /// SlotWrite:KV 写槽(副作用;返回透传值)
    fn slot_write(&mut self, v: &Value) -> Result<Value, ModelError> {
        Ok(v.clone()) // CPU 参考实现:无状态,透传
    }
    /// Block 叶子解析:按 id 取已物化数据(server 侧 = 池账房反查)
    fn block(&mut self, id: u64, dtype: Dtype, shape: &Shape) -> Result<Value, ModelError>;
}

/// f32 值块(同步参考解释器的值形态;后端版为池块句柄 /
/// owl-cpu 的 [`owl_cpu::Value`] host 真值 —— 与后端实现互为独立副本)。
#[derive(Clone, Debug, PartialEq)]
pub struct Value {
    pub f32: Vec<f32>,
    pub shape: Shape,
}

impl Value {
    /// 构造(外部 demo/测试用)
    pub fn new(f32: Vec<f32>, shape: Shape) -> Self {
        Self { f32, shape }
    }
}

impl Value {
    fn zero(dtype: Dtype, shape: &Shape) -> Result<Self, ModelError> {
        if dtype != Dtype::F32 {
            return Err(ModelError::Msg(format!(
                "CPU 解释器仅 F32(S1),得 {dtype:?}"
            )));
        }
        Ok(Value {
            f32: vec![0.0; numel(shape)],
            shape: shape.clone(),
        })
    }

    fn from_bytes(dtype: Dtype, shape: &Shape, bytes: &[u8]) -> Result<Self, ModelError> {
        if dtype != Dtype::F32 {
            return Err(ModelError::Msg("CPU 解释器仅 F32".into()));
        }
        let n = numel(shape);
        if bytes.len() != n * 4 {
            return Err(ModelError::Msg(format!(
                "Htod: 字节数 {} != {}×4",
                bytes.len(),
                n
            )));
        }
        Ok(Value {
            f32: bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            shape: shape.clone(),
        })
    }
}

/// 同步归约驱动:自叶向根。毒值在此落地;资源错误直接 Err 过线。
pub fn reduce(
    t: &TensorOps,
    itp: &mut impl Interpreter,
) -> Result<Value, ModelError> {
    // 毒值落地(案发 = depth + detail;LazyError 可回溯)
    if let Some(e) = &t.err {
        return Err(ModelError::Msg(format!(
            "[毒值落地 @depth {}] {}",
            e.at_depth, e.detail
        )));
    }
    let ins: Vec<Value> = t
        .parents
        .iter()
        .map(|p| reduce(p, itp))
        .collect::<Result<_, _>>()?;
    match &t.op {
        Op::Htod { bytes } => itp.htod(t.dtype, &t.shape, bytes),
        Op::Zeros => itp.zeros(t.dtype, &t.shape),
        Op::Matmul => itp.matmul(&ins[0], &ins[1], t.dtype, &t.shape),
        Op::Add => itp.add(&ins[0], &ins[1]),
        Op::Mul => itp.mul(&ins[0], &ins[1]),
        Op::Silu => itp.silu(&ins[0]),
        Op::Rmsnorm { eps, w_off } => itp.rmsnorm(&ins[0], &ins[1], *eps, *w_off),
        Op::SlotWrite => itp.slot_write(&ins[0]),
        Op::Block { id } => itp.block(*id, t.dtype, &t.shape),
        // 逃逸舱:client 侧闭包就地执行(不经 GpuFace;server 派发表不见它)
        Op::Kernel { kernel, .. } => Err(ModelError::Msg(format!(
            "Kernel 节点 \"{}\" 需 GPU server 执行(后端编译 + 发射;CPU 参考解释器不支持)",
            kernel.name
        ))),
        other => Err(ModelError::Msg(format!(
            "CPU 参考解释器未覆盖: {other:?}(server 侧实现)"
        ))),
    }
}

// ============================================================================
// §3 CpuInterpreter:同步 CPU 解释器(朴素实现;对拍锚)
// ============================================================================

/// 调试门控:OWL_DEBUG=1 开启解释层发射日志
fn dbg_on() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("OWL_DEBUG").is_some())
}

/// CPU 参考解释器(朴素实现;**永不优化** —— 它的存在意义就是对拍
/// 后端实现:`owl-cpu::CpuFace` / `owl-cuda::GpuClient`。两侧算子实现
/// 互为独立副本,禁止互相引用,否则对拍失效)。
pub struct CpuInterpreter {
    /// Block 叶子登记表(id → 值;rt::Tensor 物化时由绑定方注册)
    blocks: std::collections::HashMap<u64, Value>,
}

impl Default for CpuInterpreter {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuInterpreter {
    pub fn new() -> Self {
        Self { blocks: std::collections::HashMap::new() }
    }

    /// 登记 Block 叶子的数据(id = rt::Tensor 的全局 id)
    pub fn bind(&mut self, id: u64, v: Value) {
        self.blocks.insert(id, v);
    }
}

impl Interpreter for CpuInterpreter {
    fn htod(&mut self, dtype: Dtype, shape: &Shape, bytes: &[u8]) -> Result<Value, ModelError> {
        Value::from_bytes(dtype, shape, bytes)
    }

    fn zeros(&mut self, _dtype: Dtype, shape: &Shape) -> Result<Value, ModelError> {
        Value::zero(_dtype, shape)
    }

    fn matmul(
        &mut self,
        a: &Value,
        b: &Value,
        _dtype: Dtype,
        shape: &Shape,
    ) -> Result<Value, ModelError> {
        if dbg_on() {
            eprintln!("[dbg mm] a.shape={:?} a.len={} b.shape={:?} b.len={} shape={:?}", a.shape, a.f32.len(), b.shape, b.f32.len(), shape);
        }
        // k = 内维 = a 的元素数 / m(行主序)
        let m_rows = a.shape[0].max(1);
        let k = a.f32.len() / m_rows;
        let (m, n) = (shape[0], shape[1]);
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for p in 0..k {
                let av = a.f32[i * k + p];
                for j in 0..n {
                    out[i * n + j] += av * b.f32[p * n + j];
                }
            }
        }
        Ok(Value { f32: out, shape: shape.clone() })
    }

    fn add(&mut self, a: &Value, b: &Value) -> Result<Value, ModelError> {
        Ok(Value {
            f32: a.f32.iter().zip(&b.f32).map(|(x, y)| x + y).collect(),
            shape: a.shape.clone(),
        })
    }

    fn mul(&mut self, a: &Value, b: &Value) -> Result<Value, ModelError> {
        Ok(Value {
            f32: a.f32.iter().zip(&b.f32).map(|(x, y)| x * y).collect(),
            shape: a.shape.clone(),
        })
    }

    fn block(&mut self, id: u64, _dtype: Dtype, _shape: &Shape) -> Result<Value, ModelError> {
        self.blocks.get(&id).cloned().ok_or_else(|| {
            ModelError::Msg(format!("Block {id} 未绑定(需先登记物化数据)"))
        })
    }

    fn silu(&mut self, x: &Value) -> Result<Value, ModelError> {
        Ok(Value {
            f32: x.f32.iter().map(|v| v / (1.0 + (-v).exp())).collect(),
            shape: x.shape.clone(),
        })
    }

    fn rmsnorm(
        &mut self,
        x: &Value,
        alpha: &Value,
        eps: f32,
        w_off: bool,
    ) -> Result<Value, ModelError> {
        // per-channel alpha([cols] 广播;与 GPU owl_rmsnorm_f32 同一语义)
        let cols = x.shape.last().copied().unwrap_or(0).max(1);
        let rows = x.f32.len() / cols;
        let mut out = vec![0.0f32; x.f32.len()];
        for r in 0..rows {
            let xs = &x.f32[r * cols..(r + 1) * cols];
            let ms = xs.iter().map(|v| v * v).sum::<f32>() / cols as f32;
            let inv = 1.0 / (ms + eps).sqrt();
            for (c, v) in xs.iter().enumerate() {
                let g = if w_off { alpha.f32[c] + 1.0 } else { alpha.f32[c] };
                out[r * cols + c] = v * inv * g;
            }
        }
        Ok(Value { f32: out, shape: x.shape.clone() })
    }
}

//! 异步解释器:同一份声明,落地为后端原语(引擎主路径)。
//!
//! 解释器是对上层模型层(layer)的封装:layer 只写声明(TensorOps 链),
//! 解释器负责把声明翻译成后端调用(alloc / lower / launch)——算子的
//! 落地逻辑由解释器兜住,上层零后端知识。毒值在解释器边界落地
//! (结构化报错 + depth 归因)。
//!
//! | 入口 | 形态 | face |
//! |---|---|---|
//! | [`eval`] | 计算执行(层入口;驱动 Module::forward + 归约) | `owl-cpu::CpuFace` / `owl-cuda::GpuClient` |
//! | [`eval_ops`] | 计算归约(内部件;手工子树/非层根) | 同上 |
//! | [`eval_load`] | 装载执行(层入口;驱动 Loadable::layout) | 同上 |
//!
//! 后端契约 = `owl-iface::contract::DeviceClient`(线格式同源);
//! 解释器本身不含后端代码 —— 后端经 face 泛型注入。
//! 同步参考解释器(reduce/CpuInterpreter,对拍锚)已拆至 [`crate::reference`]。
//! 毒值传播契约:构造期违约随链流动,`is_poisoned()` 可查,边界收割。
//!
//! **维度源头单一律(C1,2026-09-26 定案)**:本文件一切维度推导只读
//! 声明 shape(`t.shape` / `t.parents[i].shape`);`Bytes.len` 不参与语义
//! (Block 叶子 len=0),仅在边界做断言。

use crate::contract::{Bytes, DeviceClient, ModelError};
use crate::module::{Layout, LoaderOps, WeightSource};
use crate::ops::Op;
use crate::tensor::TensorOps;
use std::future::Future;
use std::pin::Pin;

// ============================================================================
// §1 异步解释器:eval(层入口)/ eval_ops(树归约;引擎主路径)
// ============================================================================

/// 计算执行(层级入口):驱动层的 `Module::forward` 声明并归约。
/// 解释器自己调用层钩子并为它传递 ctx —— 使用者只给层与输入。
///
/// ```rust,ignore
/// let out = eval(&mlp, &xs, face, &ForwardCtx::minimal(1)).await?;
/// ```
pub async fn eval<M, D>(
    layer: &M,
    xs: &TensorOps,
    face: &mut D,
    ctx: &crate::module::ForwardCtx<'_>,
) -> Result<Bytes, ModelError>
where
    M: crate::module::Module,
    D: crate::contract::DeviceClient,
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
    D: crate::contract::DeviceClient + 'a,
{
    Box::pin(_eval(t, face))
}

async fn _eval<D: crate::contract::DeviceClient>(
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

    // C1 边界断言:非 Block 父节点的块账长 = 声明元素数(Block 叶子 len=0 无
    // 语义)。维度推导一律不用 len(见下方各分支);此处只在 debug 构建拦账变。
    debug_assert!(t.parents.iter().zip(&ins).all(|(p, b)| {
        b.len == 0
            || b.len == p.shape.iter().product::<usize>()
    }), "块账长 != 声明元素数(node id {})", t.id);

    let dtype = t.dtype;
    let shape = t.shape.clone();
    let n_elems: usize = shape.iter().product::<usize>();
    let n_bytes = n_elems * dtype.size_bytes();

    match &t.op {
        Op::Htod { bytes } => face.htod(dtype, &shape, bytes).await,
        Op::Zeros => face.alloc(n_bytes).await,
        Op::Block { id } => Ok(Bytes { id: *id, len: 0 }),
        Op::Reshape => Ok(ins[0].clone()), // 纯元数据视图:透传父块(零拷贝)
        Op::Add => {
            let out = face.alloc(n_bytes).await?;
            let msg = crate::ops::lower_add(&ins, &out, n_elems);
            face.launch(msg).await?;
            Ok(out)
        }
        Op::Mul => {
            let out = face.alloc(n_bytes).await?;
            let msg = crate::ops::lower_mul(&ins, &out, n_elems);
            face.launch(msg).await?;
            Ok(out)
        }
        Op::Silu => {
            let out = face.alloc(n_bytes).await?;
            let msg = crate::ops::lower_silu(&ins, &out, n_elems);
            face.launch(msg).await?;
            Ok(out)
        }
        Op::Sigmoid => {
            let out = face.alloc(n_bytes).await?;
            let msg = crate::ops::lower_sigmoid(&ins, &out, n_elems);
            face.launch(msg).await?;
            Ok(out)
        }
        Op::Matmul => {
            let (m, n) = (shape[0], shape[1]);
            // k = 内维 = x 声明 shape 末维(C1:只读声明 shape;原取 ins[0].len
            // 在多行 [m,k] 时越界读 —— 此 bug 被"历届测试都单行"掩盖)
            let k = t.parents[0].shape.last().copied().unwrap_or(0);
            let out = face.alloc(m * n * dtype.size_bytes()).await?;
            let msg = crate::ops::lower_matmul(&ins, &out, m, k, n);
            face.launch(msg).await?;
            Ok(out)
        }
        Op::Rmsnorm { eps, w_off } => {
            let out = face.alloc(n_bytes).await?;
            // 归一化宽度由 alpha 的声明 shape 定义(per-head 行归一化:
            // [T, H×HD] × alpha [HD])。不能取 ins[1].len —— Block 叶子 len=0。
            let alpha_shape = t.parents[1].shape.clone();
            let cols: usize = alpha_shape.iter().product::<usize>().max(1);
            let rows = shape.iter().product::<usize>() / cols;
            let msg = crate::ops::lower_rmsnorm(&ins, *eps, *w_off, &out, rows, cols);
            face.launch(msg).await?;
            Ok(out)
        }
        Op::Kernel { kernel } => {
            let out = face.alloc(n_bytes).await?;
            let msg = crate::ops::lower_kernel(kernel, &t.args, &ins, &out, n_elems);
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
    ctx: &crate::module::LoaderCtx,
) -> Result<(), ModelError>
where
    M: crate::module::Loadable,
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
    use crate::module::f32b;
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
        w.sink.deliver(crate::contract::Bytes::new(b.id, n));
    }
    Ok(())
}


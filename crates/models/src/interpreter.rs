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
use crate::module::{LoaderOps, WeightSource};
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
    Box::pin(async move {
        let mut ctx = EvalCtx { face, memo: std::collections::HashMap::new() };
        _eval_rec(t, &mut ctx).await
    })
}

/// 求值上下文(CSE 备忘录 + 后端句柄)
struct EvalCtx<'a, D> {
    face: &'a mut D,
    memo: std::collections::HashMap<u64, Bytes>,
}

/// DAG 求值入口(装箱:async 递归要求;'b 短借用 reborrow)。
fn _eval_rec<'a, 'b, D>(
    t: &'a TensorOps,
    ctx: &'b mut EvalCtx<'a, D>,
) -> Pin<Box<dyn Future<Output = Result<Bytes, ModelError>> + Send + 'b>>
where
    D: crate::contract::DeviceClient + 'a,
    D: Send,
{
    Box::pin(_eval_node(t, ctx))
}

async fn _eval_node<'a, 'b, D>(
    t: &'a TensorOps,
    ctx: &'b mut EvalCtx<'a, D>,
) -> Result<Bytes, ModelError>
where
    D: crate::contract::DeviceClient,
{
    // CSE / DAG 求值(2026-09-26 定案):共享子树(如 DecoderLayer 的残差
    // h:既喃 ln2 又喃残差加)只求值一次 —— 无 CSE 则 mixer 的状态副作用
    // kernel(conv/delta)执行两遍,第二遍读到滑过的状态 → 语义破坏。
    // memo 键 = 节点 id(全局自增,克隆保留 → 同节点 = 同值)。
    if let Some(b) = ctx.memo.get(&t.id) {
        return Ok(b.clone());
    }
    if let Some(e) = &t.err {
        return Err(ModelError::Msg(format!(
            "[毒值落地 @depth {}] {}",
            e.at_depth, e.detail
        )));
    }
    let mut ins: Vec<Bytes> = Vec::with_capacity(t.parents.len());
    for p in &t.parents {
        ins.push(_eval_rec(p, ctx).await?);
    }

    // C1 边界断言:非 Block 父节点的块账长 = 声明元素数(Block 叶子 len=0 无
    // 语义)。维度推导一律不用 len(见下方各分支);此处只在 debug 构建拦账变。
    if cfg!(debug_assertions) {
        for (i, (p, b)) in t.parents.iter().zip(&ins).enumerate() {
            let want: usize = p.shape.iter().product();
            if b.len != 0 && b.len != want {
                panic!(
                    "[C1] 块账长 != 声明元素数: node {} op {:?} parent#{i} id {} op {:?} 声明 {:?} 账长 {}",
                    t.id, t.op, p.id, p.op, p.shape, b.len
                );
            }
        }
    }

    let dtype = t.dtype;
    let shape = t.shape.clone();
    let n_elems: usize = shape.iter().product::<usize>();
    let n_bytes = n_elems * dtype.size_bytes();

    let out: Bytes = match &t.op {
        Op::Htod { bytes } => ctx.face.htod(dtype, &shape, bytes).await?,
        Op::Zeros => ctx.face.alloc(n_bytes).await?,
        Op::Block { id } => Bytes { id: *id, len: 0 },
        Op::Reshape => ins[0].clone(), // 纯元数据视图:透传父块(零拷贝)
        Op::Add => {
            let out = ctx.face.alloc(n_bytes).await?;
            let msg = crate::ops::lower_add(&ins, &out, n_elems);
            ctx.face.launch(msg).await?;
            out
        }
        Op::Mul => {
            let out = ctx.face.alloc(n_bytes).await?;
            let msg = crate::ops::lower_mul(&ins, &out, n_elems);
            ctx.face.launch(msg).await?;
            out
        }
        Op::Silu => {
            let out = ctx.face.alloc(n_bytes).await?;
            let msg = crate::ops::lower_silu(&ins, &out, n_elems);
            ctx.face.launch(msg).await?;
            out
        }
        Op::Sigmoid => {
            let out = ctx.face.alloc(n_bytes).await?;
            let msg = crate::ops::lower_sigmoid(&ins, &out, n_elems);
            ctx.face.launch(msg).await?;
            out
        }
        Op::Matmul | Op::MatmulNt => {
            let (m, n) = (shape[0], shape[1]);
            // k = 内维 = x 声明 shape 末维(C1:只读声明 shape;原取 ins[0].len
            // 在多行 [m,k] 时越界读 —— 此 bug 被"历届测试都单行"掩盖)
            let k = t.parents[0].shape.last().copied().unwrap_or(0);
            let out = ctx.face.alloc(m * n * dtype.size_bytes()).await?;
            let msg = if matches!(t.op, Op::MatmulNt) {
                crate::ops::lower_matmul_nt(&ins, &out, m, k, n)
            } else {
                crate::ops::lower_matmul(&ins, &out, m, k, n)
            };
            ctx.face.launch(msg).await?;
            out
        }
        Op::Rmsnorm { eps, w_off } => {
            let out = ctx.face.alloc(n_bytes).await?;
            // 归一化宽度由 alpha 的声明 shape 定义(per-head 行归一化:
            // [T, H×HD] × alpha [HD])。不能取 ins[1].len —— Block 叶子 len=0。
            let alpha_shape = t.parents[1].shape.clone();
            let cols: usize = alpha_shape.iter().product::<usize>().max(1);
            let rows = shape.iter().product::<usize>() / cols;
            let msg = crate::ops::lower_rmsnorm(&ins, *eps, *w_off, &out, rows, cols);
            ctx.face.launch(msg).await?;
            out
        }
        Op::Kernel { kernel } => {
            let out = ctx.face.alloc(n_bytes).await?;
            let msg = crate::ops::lower_kernel(kernel, &t.args, &ins, &out, n_elems);
            ctx.face.launch(msg).await?;
            out
        }
        Op::SlotWrite => ins[0].clone(),
        other => return Err(ModelError::Msg(format!("eval 未覆盖: {other:?}"))),
    };
    ctx.memo.insert(t.id, out.clone());
    Ok(out)
}

// ============================================================================
// §1.5 装载解释器:eval_load(识别 LoaderOps 指令;装载域的执行能力)
// ============================================================================

/// 装载并发度(2026-09-26 M-e loader 性能,用户裁决:4 协程流水 ——
/// 每协程一束键组,数据单副本流动:take 取走 → 变换/move 上传 → 即弃,
/// 主机驻留只剩"在途"份)。
const LOAD_WORKERS: usize = 4;

/// 装载执行(层级入口):驱动层的 `Loadable::layout` 需求并物化。
/// 解释器自己调用层钩子并为它传递 LoaderCtx —— 使用者只给层与源。
/// face 提供并发句柄(`DeviceClient::loader_faces`)且 Want 多于一条
/// 时走分桶流水;否则顺序。
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
    let want = layer.layout(ctx); // 层产出需求清单(ctx 引用透传)
    eval_want(&want, face, src).await
}

/// 需求清单求值:并发流水(能力可用)或顺序。
async fn eval_want<D: DeviceClient, S: WeightSource + ?Sized>(
    want: &LoaderOps,
    face: &mut D,
    src: &S,
) -> Result<(), ModelError> {
    let handles = face.loader_faces(LOAD_WORKERS);
    match handles {
        Some(handles) if handles.len() > 1 && want.wants().len() > 1 => {
            eval_want_parallel(want, handles, src).await
        }
        _ => eval_want_sequential(want, face, src).await,
    }
}

/// 键分组:同键多 Want(tied 双槽 = w 直读 + w_t 转置读)原子成组,
/// take 一次供全组;组序 = 清单首次出现序。
fn key_groups<'a>(wants: &'a [crate::module::Want]) -> Vec<Vec<&'a crate::module::Want>> {
    let mut order: Vec<&'a str> = Vec::new();
    let mut map: std::collections::HashMap<&'a str, Vec<&'a crate::module::Want>> =
        std::collections::HashMap::new();
    for w in wants {
        if !map.contains_key(w.key.as_str()) {
            order.push(w.key.as_str());
            map.insert(w.key.as_str(), Vec::new());
        }
        map.get_mut(w.key.as_str()).unwrap().push(w);
    }
    order.into_iter().map(|k| map.remove(k).unwrap()).collect()
}

fn want_bytes(w: &crate::module::Want) -> usize {
    w.shape.iter().product::<usize>() * 4
}

/// 单键组装载(数据单副本流动):
/// 1. `src.take(key)` 取走所有权(源驻留下降);
/// 2. Transposed Want 先行(只读 data,产出转置 Vec → htod_f32 move);
/// 3. Direct Want 收尾:htod_f32(data) 直接 move —— 零变换零拷贝
///    (良构校验内联;多 Direct 同键时末位以外克隆,现模型不出现)。
async fn load_group<D: DeviceClient, S: WeightSource + ?Sized>(
    face: &mut D,
    group: &[&crate::module::Want],
    src: &S,
) -> Result<(), ModelError> {
    let key = group[0].key.clone();
    let data = src
        .take(&key)
        .ok_or_else(|| ModelError::Msg(format!("Weight '{key}': 数据源缺键")))?;
    let mut direct: Option<&crate::module::Want> = None;
    for w in group {
        match w.layout {
            crate::module::Layout::Transposed => {
                let n_want: usize = w.shape.iter().product();
                let t = transpose_into_vec(data.as_slice(), w, n_want)?;
                let b = face
                    .htod_f32(&w.shape, t)
                    .await
                    .map_err(|e| ModelError::Msg(format!("Weight '{}': {e}", w.key)))?;
                let n: usize = w.shape.iter().product();
                w.sink.deliver(crate::contract::Bytes::new(b.id, n));
            }
            crate::module::Layout::Direct => direct = Some(w),
        }
    }
    if let Some(w) = direct {
        let n: usize = w.shape.iter().product();
        if data.len() != n {
            return Err(ModelError::Msg(format!(
                "Weight '{}': 元素 {} != shape {:?}({n})",
                w.key,
                data.len(),
                w.shape
            )));
        }
        let b = face
            .htod_f32(&w.shape, data)
            .await
            .map_err(|e| ModelError::Msg(format!("Weight '{}': {e}", w.key)))?;
        w.sink.deliver(crate::contract::Bytes::new(b.id, n));
    }
    Ok(())
}

/// TILE² 分块转置 → 新 Vec(源 [rows, cols] → 目标 [cols, rows] f32)
fn transpose_into_vec(data: &[f32], w: &crate::module::Want, n: usize) -> Result<Vec<f32>, ModelError> {
    const TILE: usize = 64;
    let (cols, rows) = (w.shape[0], w.shape[1]);
    if data.len() != n {
        return Err(ModelError::Msg(format!(
            "Weight '{}': 元素 {} != 源形状 {rows}×{cols}",
            w.key,
            data.len()
        )));
    }
    let mut out = vec![0f32; n];
    for r0 in (0..rows).step_by(TILE) {
        let r1 = (r0 + TILE).min(rows);
        for c0 in (0..cols).step_by(TILE) {
            let c1 = (c0 + TILE).min(cols);
            for r in r0..r1 {
                for c in c0..c1 {
                    out[c * rows + r] = data[r * cols + c];
                }
            }
        }
    }
    Ok(out)
}

/// 顺序路径(CpuFace / 单 Want / 无并发能力)
async fn eval_want_sequential<D: DeviceClient, S: WeightSource + ?Sized>(
    want: &LoaderOps,
    face: &mut D,
    src: &S,
) -> Result<(), ModelError> {
    for group in key_groups(want.wants()) {
        load_group(face, &group, src).await?;
    }
    Ok(())
}

/// 并发流水(GpuClient 等提供多句柄的 face):
/// 键组按字节 LPT 分给最轻协程(embed 双槽 2GB 独占一束);
/// join_all 单线程协作驱动 —— 组 A await 设备拷贝期间,组 B 的
/// take/变换在同一线程推进,与 server 线程 memcpy 流水重叠。
/// 交付序无关(Want 各带各的 sink;分配无跨 Want 定序)。
async fn eval_want_parallel<D: DeviceClient, S: WeightSource + ?Sized>(
    want: &LoaderOps,
    handles: Vec<D>,
    src: &S,
) -> Result<(), ModelError> {
    let mut groups = key_groups(want.wants());
    groups.sort_by_key(|g| {
        std::cmp::Reverse(g.iter().map(|w| want_bytes(w)).sum::<usize>())
    });
    let k = handles.len().min(groups.len()).max(1);
    let mut buckets: Vec<(D, Vec<Vec<&crate::module::Want>>, usize)> = handles
        .into_iter()
        .take(k)
        .map(|f| (f, Vec::new(), 0usize))
        .collect();
    for g in groups {
        let bytes: usize = g.iter().map(|w| want_bytes(w)).sum();
        let b = buckets.iter_mut().min_by_key(|(_, _, total)| *total).unwrap();
        b.1.push(g);
        b.2 += bytes;
    }
    let futs = buckets.into_iter().map(|(mut face, bundles, _)| async move {
        for bundle in &bundles {
            load_group(&mut face, bundle, src).await?;
        }
        Ok::<(), ModelError>(())
    });
    let results = futures_util::future::join_all(futs).await;
    results.into_iter().collect::<Result<Vec<_>, _>>()?;
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::{assert_close, f32b, gpu_client, gpu_enabled, harvest, skip_note};

    fn inputs() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let x: Vec<f32> = (0..4 * 16).map(|i| ((i as f32 * 0.31) - 3.0).sin()).collect();
        let w1: Vec<f32> = (0..16 * 8).map(|i| (i as f32 * 0.11) - 0.6).collect();
        let gamma: Vec<f32> = (0..8).map(|i| (i as f32 * 0.23) - 0.8).collect();
        (x, w1, gamma)
    }

    /// 全语义算子复合树(多行):matmul → add → sigmoid → silu → mul → rmsnorm
    fn composite_tree(x: &TensorOps, w1: &TensorOps, gamma: &TensorOps, w_off: bool) -> TensorOps {
        let h1 = x.matmul(w1); // [4,16]×[16,8] → [4,8](多行 k;防单行掩盖)
        let h2 = h1.add(&h1.sigmoid().silu().mul(&h1));
        h2.rmsnorm(gamma, 1e-6, w_off)
    }

    /// GPU 后端 vs CPU 参考锚(同一棵声明树,双 face 分跑;
    /// 数值到 HF 的对拍在各层 parity 测试)
    #[tokio::test]
    async fn gpu_semantic_ops_match_cpu_multirow() {
        if !gpu_enabled() {
            skip_note();
            return;
        }
        use crate::tensor::Dtype;
        let (x, w1, gamma) = inputs();
        let mut cpu = owl_cpu::CpuFace::new();
        let mut gpu = gpu_client().await;

        for w_off in [false, true] {
            let tree = composite_tree(
                &TensorOps::from_host(Dtype::F32, vec![4, 16], &f32b(&x)),
                &TensorOps::from_host(Dtype::F32, vec![16, 8], &f32b(&w1)),
                &TensorOps::from_host(Dtype::F32, vec![8], &f32b(&gamma)),
                w_off,
            );
            let cpu_out = harvest(&mut cpu, &tree).await;
            let gpu_out = harvest(&mut gpu, &tree).await;
            assert_close(&gpu_out, &cpu_out, 1e-5, &format!("w_off={w_off}"));
        }

        gpu.close().await.expect("server 关机");
    }
}

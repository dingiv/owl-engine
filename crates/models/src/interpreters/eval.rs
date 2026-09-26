//! 计算(推理)解释器:声明树 DAG → 后端原语调用。
//!
//! 职责(单域):`TensorOps` 树的自叶向根归约 —— 逐节点翻译为
//! alloc / lower / launch,带 CSE 备忘录(同 id 单次求值)+ 毒值落地
//! (结构化报错 + depth 归因)+ C1 维度断言。
//!
//! 变体定位:这是**计算域**解释器;装载域见 [`super::load`]
//! (Loadable::layout → Want 清单 → 分桶流水)。二者共享 face 注入
//! 形态(`owl-iface::contract::DeviceClient`)与"声明/执行分离"范式,
//! 但互不依赖 —— 新变体(图捕获回放、量化装载、其他平台)另起文件。

use crate::contract::{Bytes, Dtype, ModelError};
use crate::ops::Op;
use crate::tensor::TensorOps;
use std::future::Future;
use std::pin::Pin;

use super::observe::{BlockRef, BlockStats, NodeEvent, Tap, Want};

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
/// 块字节 → host f32(按块 dtype;f16 基线后观测面统一走此解码。
/// 字节口径纪律:读回长度一律 `elem 数 × dtype.size_bytes()`,禁硬编码)
pub(crate) fn decode_host(dtype: Dtype, buf: &[u8]) -> Vec<f32> {
    match dtype {
        Dtype::F16 => buf
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        Dtype::BF16 => buf
            .chunks_exact(2)
            .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        Dtype::F32 => buf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        // U32(frontier 等 id 块):位型直释(观测统计只对浮点块有意义)
        Dtype::U32 => buf
            .chunks_exact(4)
            .map(|c| f32::from_bits(u32::from_le_bytes([c[0], c[1], c[2], c[3]])))
            .collect(),
    }
}

pub fn eval_ops<'a, D>(
    t: &'a TensorOps,
    face: &'a mut D,
) -> Pin<Box<dyn Future<Output = Result<Bytes, ModelError>> + Send + 'a>>
where
    D: crate::contract::DeviceClient + 'a,
{
    Box::pin(async move {
        let mut ctx =
            EvalCtx { face, memo: std::collections::HashMap::new(), tap: None };
        _eval_rec(t, &mut ctx).await
    })
}

/// 带观测的求值(interpreter-tap.md §4.2):归约语义与 [`eval_ops`]
/// 完全一致,仅在每个节点 After 窗口发事件 —— 数据读取(Stats/Bytes)
/// 由解释器代执行,tap 零执行权(观测窗口律,候选 §四 律 25)。
pub fn eval_ops_tap<'a, D>(
    t: &'a TensorOps,
    face: &'a mut D,
    tap: &'a mut dyn Tap,
) -> Pin<Box<dyn Future<Output = Result<Bytes, ModelError>> + Send + 'a>>
where
    D: crate::contract::DeviceClient + 'a,
{
    Box::pin(async move {
        let mut ctx =
            EvalCtx { face, memo: std::collections::HashMap::new(), tap: Some(tap) };
        _eval_rec(t, &mut ctx).await
    })
}

/// 求值上下文(CSE 备忘录 + 后端句柄 + 观测者)
struct EvalCtx<'a, D> {
    face: &'a mut D,
    memo: std::collections::HashMap<u64, Bytes>,
    /// 观测面(interpreter-tap.md;None = 零开销直通)
    tap: Option<&'a mut dyn Tap>,
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
        let detail = format!("[毒值落地 @depth {}] {}", e.at_depth, e.detail);
        // Tap:Poison 事件(毒值节点无输出块,无 After;归约照旧 Err)
        if let Some(tap) = ctx.tap.as_deref_mut() {
            tap.on_poison(t.id, &detail);
        }
        return Err(ModelError::Msg(detail));
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
        Op::Zeros => ctx.face.alloc(dtype, n_elems).await?,
        Op::Block { id } => Bytes { id: *id, len: 0 },
        Op::Reshape => ins[0].clone(), // 纯元数据视图:透传父块(零拷贝)
        Op::Add => {
            let out = ctx.face.alloc(dtype, n_elems).await?;
            let msg = crate::ops::lower_add(&ins, &out, n_elems, dtype);
            ctx.face.launch(msg).await?;
            out
        }
        Op::Mul => {
            let out = ctx.face.alloc(dtype, n_elems).await?;
            let msg = crate::ops::lower_mul(&ins, &out, n_elems, dtype);
            ctx.face.launch(msg).await?;
            out
        }
        Op::Silu => {
            let out = ctx.face.alloc(dtype, n_elems).await?;
            let msg = crate::ops::lower_silu(&ins, &out, n_elems, dtype);
            ctx.face.launch(msg).await?;
            out
        }
        Op::Sigmoid => {
            let out = ctx.face.alloc(dtype, n_elems).await?;
            let msg = crate::ops::lower_sigmoid(&ins, &out, n_elems, dtype);
            ctx.face.launch(msg).await?;
            out
        }
        Op::Matmul | Op::MatmulNt => {
            let (m, n) = (shape[0], shape[1]);
            // k = 内维 = x 声明 shape 末维(C1:只读声明 shape;原取 ins[0].len
            // 在多行 [m,k] 时越界读 —— 此 bug 被"历届测试都单行"掩盖)
            let k = t.parents[0].shape.last().copied().unwrap_or(0);
            let out = ctx.face.alloc(dtype, m * n).await?;
            if dtype == Dtype::F16 {
                // f16 基线:foreign-kernel 通道(cuBLAS;nt = owl Linear 惯例)。
                // gemm 侧 cm 约定:m = n_out(权重行)/ n = T(2026-09-26 修正:
                // 原误传 (m=T, n=n_out),T=1 即 B 操作数越界读 → ILLEGAL_ADDRESS)
                let nt = matches!(t.op, Op::MatmulNt);
                let msg = crate::ops::lower_gemm(&ins, &out, n, k, m, nt);
                ctx.face.launch(msg).await?;
            } else {
                let msg = if matches!(t.op, Op::MatmulNt) {
                    crate::ops::lower_matmul_nt(&ins, &out, m, k, n)
                } else {
                    crate::ops::lower_matmul(&ins, &out, m, k, n)
                };
                ctx.face.launch(msg).await?;
            }
            out
        }
        Op::Rmsnorm { eps, w_off } => {
            let out = ctx.face.alloc(dtype, n_elems).await?;
            // 归一化宽度由 alpha 的声明 shape 定义(per-head 行归一化:
            // [T, H×HD] × alpha [HD])。不能取 ins[1].len —— Block 叶子 len=0。
            let alpha_shape = t.parents[1].shape.clone();
            let cols: usize = alpha_shape.iter().product::<usize>().max(1);
            let rows = shape.iter().product::<usize>() / cols;
            let msg = crate::ops::lower_rmsnorm(&ins, *eps, *w_off, &out, rows, cols, dtype);
            ctx.face.launch(msg).await?;
            out
        }
        Op::Kernel { kernel } => {
            // f16 基线守门(2026-09-26):注册表 dtype 标注对账声明 dtype,
            // 不符 = 结构化报错 —— 堵死「f32 核读 f16 字节 = 静默垃圾」。
            // 逃生舱(非注册,带 sig)跳过(仅测试域)。
            if let Some(e) = crate::kernel::lookup(kernel.name) {
                if e.dtype != dtype {
                    return Err(ModelError::Msg(format!(
                        "[dtype 守门] kernel \"{}\" 登记为 {:?},声明为 {:?} \
                         —— f16 变体未注册(F2-F4 迁移中)",
                        kernel.name, e.dtype, dtype
                    )));
                }
            }
            let out = ctx.face.alloc(dtype, n_elems).await?;
            let msg = crate::ops::lower_kernel(kernel, &t.args, &ins, &out, n_elems);
            ctx.face.launch(msg).await?;
            out
        }
        Op::SlotWrite => ins[0].clone(),
        other => return Err(ModelError::Msg(format!("eval 未覆盖: {other:?}"))),
    };
    ctx.memo.insert(t.id, out.clone());
    // ── Tap:After 窗口(输出已入 memo;父输入仍在 memo 存活)──
    // 观测窗口律:数据读取由解释器代执行并在窗口内完成;窗口外读块
    // 未定义(scratch 块 eval 结束后可被池复用)。tap 无 launch/alloc
    // 权 —— 观测在结构上不可能改变计算(对照 harvest 重放污染)。
    // f16 基线(F5):观测读回按块 dtype 解码(f32 硬解码会读歪 rms)
    if let Some(tap) = ctx.tap.as_deref_mut() {
        let ev = NodeEvent {
            id: t.id,
            op: &t.op,
            dtype,
            shape: &t.shape,
            depth: t.depth,
            tag: t.label.as_deref(),
            out: BlockRef { block_id: out.id, len: out.len },
        };
        match tap.on_node(&ev) {
            Want::Quiet | Want::Keep => {} // Keep:MVP 未实施,按 Quiet 落空
            Want::Stats => {
                let mut buf = vec![0u8; n_bytes];
                ctx.face.dtoh(&out, &mut buf).await?;
                let host = decode_host(dtype, &buf);
                tap.on_stats(t.id, BlockStats::of(&host));
            }
            Want::Bytes => {
                let mut buf = vec![0u8; n_bytes];
                ctx.face.dtoh(&out, &mut buf).await?;
                let host = decode_host(dtype, &buf);
                tap.on_bytes(&ev, &host);
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod dtype_guard_tests {
    use super::*;
    use crate::contract::{DeviceClient, Dtype};
    use crate::tensor::TensorOps;

    /// CPU 面:f16 语义算子结构化报错(CPU 先不搞,裁决)——不许静默
    #[tokio::test]
    async fn cpu_f16_semantic_unsupported() {
        let mut face = owl_cpu::CpuFace::new();
        // Block 叶子(CPU htod 对 f16 本就拒;守门测的是声明 dtype 对账)
        let a = TensorOps::of_block(901, Dtype::F16, vec![2]);
        let b = TensorOps::of_block(902, Dtype::F16, vec![2]);
        let y = a.add(&b);
        let err = eval_ops(y.step(), &mut face).await.unwrap_err();
        let msg = format!("{err}");
        // CPU 拒绝点随面能力漂移(alloc 拒 / 未注册动作 / DeadBlock /
        // dtype 守门),唯一不变量 = **显式报错,绝不静默错值** —— 这才是
        // 本测试守护的性质
        assert!(
            !(msg.is_empty()),
            "f16 语义算子在 CPU 面必须显式报错,得静默通过: {msg}"
        );
    }

    /// f16 基线守门:注册核(f32 条目)遇 f16 声明 = 结构化报错
    #[tokio::test]
    async fn f16_registered_kernel_is_rejected() {
        let mut face = owl_cpu::CpuFace::new();
        // narrow_strided 登记 F32;f16 声明 → 守门拦截
        let src = TensorOps::of_block(903, Dtype::F16, vec![4]);
        let decl = TensorOps::of(crate::kernel::kernel_with(
            "owl_narrow_strided_f32",
            (0, 0, 0),
            (256, 1, 1),
            0,
        ))
        .arg(&src)
        .arg_usize(1)
        .arg_usize(4)
        .arg_usize(0)
        .arg_usize(2)
        .with_shape(Dtype::F16, vec![2]);
        let err = eval_ops(decl.step(), &mut face).await.unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("dtype 守门") && msg.contains("owl_narrow_strided_f32"), "{msg}");
    }

    /// GPU:f16 语义五算子 vs host f32 参考(F2 收口锚;OWL_TEST_DEVICE 门控)
    #[tokio::test]
    async fn gpu_f16_semantic_matches_host() {
        if !crate::testkit::gpu_enabled() {
            eprintln!("skip: OWL_TEST_DEVICE 未设");
            return;
        }
        use crate::testkit::gpu_client;
        let n = 1024usize;
        let x: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.21) - 5.0).sin() * 2.0).collect();
        let yv: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.13) - 1.0).cos() * 1.5).collect();
        let alpha: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01) - 0.5).collect();
        let xb: Vec<u8> = x.iter().flat_map(|f| half_le(*f)).collect();

        let mut gpu = gpu_client().await;
        let shape = crate::contract::Shape::from(vec![n]);
        let dx_b = gpu.htod(Dtype::F16, &shape, &xb).await.expect("htod x");
        let dy_b = gpu.htod(Dtype::F16, &shape, &yv.iter().flat_map(|f| half_le(*f)).collect::<Vec<u8>>()).await.expect("htod y");
        let da_b = gpu.htod(Dtype::F16, &shape, &alpha.iter().flat_map(|f| half_le(*f)).collect::<Vec<u8>>()).await.expect("htod alpha");
        let dx = TensorOps::of_block(dx_b.id, Dtype::F16, vec![n]);
        let dy = TensorOps::of_block(dy_b.id, Dtype::F16, vec![n]);
        let da = TensorOps::of_block(da_b.id, Dtype::F16, vec![n]);

        // y = silu(x + y) * sigmoid(x);z = rmsnorm(y ×w_off, alpha)
        let s1 = dx.add(&dy);
        let s2 = s1.silu();
        let g = dx.sigmoid();
        let m = s2.mul(&g);
        let z = m.rmsnorm(&da, 1e-6, true);
        let bytes = eval_ops(z.step(), &mut gpu).await.expect("eval f16 chain");
        let mut buf = vec![0u8; n * 2];
        gpu.dtoh(&bytes, &mut buf).await.expect("dtoh");
        let got: Vec<f32> = buf
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect();

        // host f32 参考
        let mut max_diff = 0f32;
        for i in 0..n {
            let s = x[i] + yv[i];
            let silu = s / (1.0 + (-s).exp());
            let sig = 1.0 / (1.0 + (-x[i]).exp());
            let m_val = silu * sig;
            // rmsnorm 行内需全行 rms —— 算全行
            let _ = max_diff;
            let _ = m_val;
        }
        let mut sumsq = 0f32;
        for i in 0..n {
            let s = x[i] + yv[i];
            let silu = s / (1.0 + (-s).exp());
            let sig = 1.0 / (1.0 + (-x[i]).exp());
            let mv = silu * sig;
            sumsq += mv * mv;
        }
        let inv = 1.0 / (sumsq / n as f32 + 1e-6).sqrt();
        for (i, g_val) in got.iter().enumerate() {
            let s = x[i] + yv[i];
            let silu = s / (1.0 + (-s).exp());
            let sig = 1.0 / (1.0 + (-x[i]).exp());
            let want = silu * sig * inv * (alpha[i] + 1.0);
            let diff = (g_val - want).abs();
            max_diff = max_diff.max(diff);
            assert!(diff < 5e-2 * (1.0 + want.abs()), "[{i}] f16 {g_val} vs host {want}");
        }
        eprintln!("[f16 语义链] max_diff = {max_diff:.5}");
        gpu.close().await.expect("关机");
    }

    fn half_le(f: f32) -> [u8; 2] {
        half::f16::from_f32(f).to_le_bytes()
    }

    /// f32 正常路径不受守门影响(回归哨)
    #[tokio::test]
    async fn f32_path_unaffected() {
        let mut face = owl_cpu::CpuFace::new();
        let a = TensorOps::from_host(Dtype::F32, vec![2], &crate::testkit::f32b(&[1.0, 2.0]));
        let b = TensorOps::from_host(Dtype::F32, vec![2], &crate::testkit::f32b(&[3.0, 4.0]));
        let y = a.add(&b);
        let bytes = eval_ops(y.step(), &mut face).await.expect("f32 add");
        let mut buf = vec![0u8; 8];
        face.dtoh(&bytes, &mut buf).await.expect("dtoh");
        let got: Vec<f32> = buf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(got, vec![4.0, 6.0]);
    }
}

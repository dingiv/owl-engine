//! 同步参考解释器(CPU 朴素归约;对拍锚)。
//!
//! 与 [`crate::interpreter`](异步解释器)的分工:
//! - 异步面 = 引擎主路径(face 注入 CpuFace/GpuClient,launch 原语);
//! - 本模块 = **永不优化**的朴素归约(纯 Rust;测试/对拍基准)。
//!
//! ⚠️ **对拍锚纪律**:本文件与 `owl_cpu::ops` 的算子实现**互为独立副本,
//! 禁止互相引用** —— 参考实现的存在意义就是对拍后端,共享代码会让对拍
//! 失效。两侧语义必须一致(行主序 / rmsnorm alpha 定宽 ×(1+w)),数值
//! 路径允许分化(owl-cpu 将来可 SIMD/rayon;参考侧永不优化)。
//!
//! 语义权威:与异步解释器/lower_*/.cu 注释同源 —— 唯一权威出处见
//! `docs/arch/qwen3-mini-demo.md` §四(C8 定案后回写)。

use crate::contract::{ModelError, Shape};
use crate::contract::numel;
use crate::interpreters::observe::{BlockRef, BlockStats, NodeEvent, Tap, Want};
use crate::ops::{KernelArg, Op, SemanticKernel};
use crate::tensor::{Dtype, TensorOps};

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
    /// MatmulNt:[m,k]×(B [n,k] 直读)→ [m,n]
    fn matmul_nt(&mut self, a: &Value, b: &Value, dtype: Dtype, shape: &Shape) -> Result<Value, ModelError>;
    /// Add(同形)—— **不是过时方法**(2026-10-15 复核):`reduce` 的
    /// `Op::Call { SemanticKernel::Add }` 臂就派发到这里,而层侧
    /// (decoder 残差 / dflash2 累加 / mtp)真发 Add/Mul 语义算子 ⇒
    /// `CpuInterpreter` 作为**对拍锚**(与 `owl-cpu` / `owl-cuda` 互为独立副本)
    /// 必须实现它们;删掉 = 参考面留白洞(图里有 Add 时对拍直接失明)。
    fn add(&mut self, a: &Value, b: &Value) -> Result<Value, ModelError>;
    /// Mul(同形逐元素乘;MLP 门控/输出门)—— 同 `add`,勿删(`Op::Call{Mul}` 臂派发于此)
    fn mul(&mut self, a: &Value, b: &Value) -> Result<Value, ModelError>;
    /// Silu
    fn silu(&mut self, x: &Value) -> Result<Value, ModelError>;
    /// Sigmoid(逐元素;attn_output_gate 门 / GDN beta 同族)
    fn sigmoid(&mut self, x: &Value) -> Result<Value, ModelError>;
    /// Rmsnorm(parents = [x, alpha];per-channel alpha([cols] 广播);w_off = ×(1+w))
    fn rmsnorm(
        &mut self,
        x: &Value,
        alpha: &Value,
        eps: f32,
        w_off: bool,
    ) -> Result<Value, ModelError>;
    /// Layernorm(affine;视觉塔):parents = [x, weight, bias];two-pass mean/var
    fn layernorm(
        &mut self,
        x: &Value,
        weight: &Value,
        bias: &Value,
        n: usize,
        eps: f32,
    ) -> Result<Value, ModelError>;
    /// GELU(tanh 近似;视觉塔 MLP)
    fn gelu_tanh(&mut self, x: &Value) -> Result<Value, ModelError>;
    /// GELU(精确 erf;视觉塔 merger)
    fn gelu_erf(&mut self, x: &Value) -> Result<Value, ModelError>;
    /// 行广播加(x [rows,cols] + bias [cols])
    fn bias_add(&mut self, x: &Value, bias: &Value, cols: usize) -> Result<Value, ModelError>;
    /// 位置嵌入 4-tap 重采样(table / idx / wts)
    fn pos_embed_bilinear(
        &mut self,
        table: &Value,
        idx: &Value,
        wts: &Value,
        n_taps: usize,
        dim: usize,
    ) -> Result<Value, ModelError>;
    /// VLM 图像 token 注入:h [L,D] 按 idx(≥0 行号 / <0 保留)换成 merged 行
    fn image_scatter(
        &mut self,
        h: &Value,
        merged: &Value,
        idx: &Value,
        dim: usize,
    ) -> Result<Value, ModelError>;
    /// 非因果 varlen 全注意力(q/k/v/cu;无 GQA)
    #[allow(clippy::too_many_arguments)]
    fn attn_full(
        &mut self,
        q: &Value,
        k: &Value,
        v: &Value,
        cu: &Value,
        rows: usize,
        heads: usize,
        head_dim: usize,
        n_seq: usize,
        scale: f32,
    ) -> Result<Value, ModelError>;
    /// Rope(rotate-half 表驱动;pos 选表行,前 2·half 维旋转)
    #[allow(clippy::too_many_arguments)]
    fn rope_half_partial(
        &mut self,
        x: &Value,
        cos_t: &Value,
        sin_t: &Value,
        pos: &Value,
        heads: usize,
        head_dim: usize,
        half: usize,
    ) -> Result<Value, ModelError>;
    /// 非连续窄切物化
    fn narrow_strided(
        &mut self,
        src: &Value,
        outer: usize,
        src_dim: usize,
        start: usize,
        out_dim: usize,
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
    reduce_tap(t, itp, None)
}

/// 带观测的归约(interpreter-tap.md §五):事件词汇与计算域
/// `eval_ops_tap` 同源;**同树双锚求值,id 在声明期已定,天然对齐**
/// —— 逐节点自动 bisect 的 CPU 侧挂点。值已在手上(CPU 无池块),
/// 统计本地即算,观测窗口律天然满足。
pub fn reduce_tap(
    t: &TensorOps,
    itp: &mut impl Interpreter,
    tap: Option<&mut dyn Tap>,
) -> Result<Value, ModelError> {
    let mut tap = tap;
    reduce_rec(t, itp, &mut tap)
}

fn reduce_rec(
    t: &TensorOps,
    itp: &mut impl Interpreter,
    tap: &mut Option<&mut dyn Tap>,
) -> Result<Value, ModelError> {
    // 毒值落地(案发 = depth + detail;LazyError 可回溯)
    if let Some(e) = &t.err {
        let detail = format!("[毒值落地 @depth {}] {}", e.at_depth, e.detail);
        if let Some(tp) = tap.as_deref_mut() {
            tp.on_poison(t.id, &detail);
        }
        return Err(ModelError::Msg(detail));
    }
    let ins: Vec<Value> = t
        .parents
        .iter()
        .map(|p| reduce_rec(p, itp, tap))
        .collect::<Result<_, _>>()?;
    let value: Value = match &t.op {
        Op::Htod { bytes } => itp.htod(t.dtype, &t.shape, bytes),
        Op::Zeros => itp.zeros(t.dtype, &t.shape),
        Op::Matmul => itp.matmul(&ins[0], &ins[1], t.dtype, &t.shape),
        Op::MatmulNt => itp.matmul_nt(&ins[0], &ins[1], t.dtype, &t.shape),
        // 纯算子动作(2026-10-12 瘦身):Op::Call 枚举臂 —— CPU 解释器
        // match 同一词表落 host 实现(硬件无关动作契约的 CPU 面)
        Op::Call { op: SemanticKernel::Add, .. } => itp.add(&ins[0], &ins[1]),
        Op::Call { op: SemanticKernel::Mul, .. } => itp.mul(&ins[0], &ins[1]),
        Op::Call { op: SemanticKernel::Silu, .. } => itp.silu(&ins[0]),
        Op::Call { op: SemanticKernel::Sigmoid, .. } => itp.sigmoid(&ins[0]),
        Op::Call { op: SemanticKernel::Rmsnorm, .. } => {
            // 标量槽序 = 线契约:cols(i32)/ eps(f32)/ w_off(i32)
            let eps = match t.args.get(1) {
                Some(KernelArg::F32(v)) => *v,
                other => panic!("rmsnorm 标量槽 1 期望 f32 eps,实得 {other:?}"),
            };
            let w_off = match t.args.get(2) {
                Some(KernelArg::I32(v)) => *v != 0,
                other => panic!("rmsnorm 标量槽 2 期望 i32 w_off,实得 {other:?}"),
            };
            itp.rmsnorm(&ins[0], &ins[1], eps, w_off)
        }
        Op::Call { op: SemanticKernel::Layernorm, .. } => {
            // 标量槽序 = vis.layernorm 线契约:n(i32) / eps(f32)
            let n = match t.args.first() {
                Some(KernelArg::I32(v)) => *v as usize,
                other => panic!("layernorm 标量槽 0 期望 i32 n,实得 {other:?}"),
            };
            let eps = match t.args.get(1) {
                Some(KernelArg::F32(v)) => *v,
                other => panic!("layernorm 标量槽 1 期望 f32 eps,实得 {other:?}"),
            };
            itp.layernorm(&ins[0], &ins[1], &ins[2], n, eps)
        }
        Op::Call { op: SemanticKernel::Rope, .. } => {
            // 标量槽序 = ops.rope 线契约:heads(sz)/head_dim(sz)/half(sz)
            let sz = |i: usize, what: &str| match t.args.get(i) {
                Some(KernelArg::Bits(v)) => *v as usize,
                other => panic!("rope 标量槽 {i}({what})期望 sz,实得 {other:?}"),
            };
            itp.rope_half_partial(&ins[0], &ins[1], &ins[2], &ins[3], sz(0, "heads"), sz(1, "head_dim"), sz(2, "half"))
        }
        Op::Call { op: SemanticKernel::Narrow, .. } => {
            // 标量槽序 = ops.narrow 线契约:outer/src_dim/start/out_dim(sz)
            let sz = |i: usize, what: &str| match t.args.get(i) {
                Some(KernelArg::Bits(v)) => *v as usize,
                other => panic!("narrow 标量槽 {i}({what})期望 sz,实得 {other:?}"),
            };
            itp.narrow_strided(&ins[0], sz(0, "outer"), sz(1, "src_dim"), sz(2, "start"), sz(3, "out_dim"))
        }
        Op::Call { op: SemanticKernel::GeluTanh, .. } => itp.gelu_tanh(&ins[0]),
        Op::Call { op: SemanticKernel::GeluErf, .. } => itp.gelu_erf(&ins[0]),
        Op::Call { op: SemanticKernel::BiasAdd, .. } => {
            // 标量槽序 = vis.bias_add 线契约:cols(sz)
            let cols = match t.args.first() {
                Some(KernelArg::Bits(v)) => *v as usize,
                other => panic!("bias_add 标量槽 0 期望 sz cols,实得 {other:?}"),
            };
            itp.bias_add(&ins[0], &ins[1], cols)
        }
        Op::Call { op: SemanticKernel::PosEmbedBilinear, .. } => {
            // 标量槽序 = vis.pos_embed 线契约:n_taps(sz) / dim(sz)
            let n_taps = match t.args.first() {
                Some(KernelArg::Bits(v)) => *v as usize,
                other => panic!("pos_embed 标量槽 0 期望 sz n_taps,实得 {other:?}"),
            };
            let dim = match t.args.get(1) {
                Some(KernelArg::Bits(v)) => *v as usize,
                other => panic!("pos_embed 标量槽 1 期望 sz dim,实得 {other:?}"),
            };
            itp.pos_embed_bilinear(&ins[0], &ins[1], &ins[2], n_taps, dim)
        }
        Op::Call { op: SemanticKernel::ImageScatter, .. } => {
            // 标量槽序 = vis.image_scatter 线契约:dim(sz)
            let dim = match t.args.first() {
                Some(KernelArg::Bits(v)) => *v as usize,
                other => panic!("image_scatter 标量槽 0 期望 sz dim,实得 {other:?}"),
            };
            itp.image_scatter(&ins[0], &ins[1], &ins[2], dim)
        }
        Op::Call { op: SemanticKernel::AttnFull, .. } => {
            // 标量槽序 = vis.attn_full 线契约:rows/heads/head_dim(sz) / n_seq(i32) / scale(f32)
            let sz = |i: usize, what: &str| match t.args.get(i) {
                Some(KernelArg::Bits(v)) => *v as usize,
                other => panic!("attn_full 标量槽 {i}({what})期望 sz,实得 {other:?}"),
            };
            let n_seq = match t.args.get(3) {
                Some(KernelArg::I32(v)) => *v as usize,
                other => panic!("attn_full 标量槽 3 期望 i32 n_seq,实得 {other:?}"),
            };
            let scale = match t.args.get(4) {
                Some(KernelArg::F32(v)) => *v,
                other => panic!("attn_full 标量槽 4 期望 f32 scale,实得 {other:?}"),
            };
            itp.attn_full(
                &ins[0], &ins[1], &ins[2], &ins[3],
                sz(0, "rows"), sz(1, "heads"), sz(2, "head_dim"), n_seq, scale,
            )
        }
        // 其余动作(gdn/attn/ln/mlp/dflash/marlin/fi…)= GPU/引擎路径;
        // CPU 参考面结构化拒绝,绝不静默
        Op::Call { op, .. } => Err(ModelError::Msg(format!(
            "CPU 参考解释器未覆盖动作 {op:?}(server/引擎路径实现)"
        ))),
        Op::SlotWrite => itp.slot_write(&ins[0]),
        Op::Reshape => Ok(Value { f32: ins[0].f32.clone(), shape: t.shape.clone() }),
        Op::SliceView { .. } => Ok(Value { f32: ins[0].f32.clone(), shape: t.shape.clone() }),
        Op::Block { id } => itp.block(*id, t.dtype, &t.shape),
    }?;
    // ── Tap:After 窗口(CPU 无池块,block_id = NO_POOL 哨兵)──
    if let Some(tp) = tap.as_deref_mut() {
        let ev = NodeEvent {
            id: t.id,
            op: &t.op,
            dtype: t.dtype,
            shape: &t.shape,
            depth: t.depth,
            tag: t.label.as_deref(),
            out: BlockRef { block_id: BlockRef::NO_POOL, len: value.f32.len() },
        };
        match tp.on_node(&ev) {
            Want::Quiet | Want::Keep => {} // Keep:MVP 未实施,按 Quiet 落空
            Want::Stats => tp.on_stats(t.id, BlockStats::of(&value.f32)),
            Want::Bytes => tp.on_bytes(&ev, &value.f32),
        }
    }
    Ok(value)
}

// ============================================================================
// §3 CpuInterpreter:同步 CPU 解释器(朴素实现;对拍锚)
// ============================================================================

/// CPU 参考解释器(朴素实现;**永不优化** —— 它的存在意义就是对拍
/// 后端实现:`owl-cpu::CpuFace` / `owl-cuda::GpuClient`。两侧算子实现
/// 互为独立副本,禁止互相引用,否则对拍失效)。
pub struct CpuInterpreter {
    /// Block 叶子登记表(id → 值;rt::Tensor 物化时由绑定方注册)
    blocks: std::collections::HashMap<u64, Value>,
    /// 解释层发射日志(对拍调试;显式依赖律,原 OWL_DEBUG OnceLock 直读
    /// —— 测试/入口构造时传,缺省关)
    pub debug: bool,
}

impl Default for CpuInterpreter {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuInterpreter {
    pub fn new() -> Self {
        Self { blocks: std::collections::HashMap::new(), debug: false }
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
        if self.debug {
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

    fn matmul_nt(
        &mut self,
        a: &Value,
        b: &Value,
        _dtype: Dtype,
        shape: &Shape,
    ) -> Result<Value, ModelError> {
        let m_rows = a.shape[0].max(1);
        let k = a.f32.len() / m_rows;
        let (m, n) = (shape[0], shape[1]);
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let brow = &b.f32[j * k..(j + 1) * k];
                let acc: f32 = a.f32[i * k..(i + 1) * k]
                    .iter()
                    .zip(brow)
                    .map(|(x, y)| x * y)
                    .sum();
                out[i * n + j] = acc;
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

    fn sigmoid(&mut self, x: &Value) -> Result<Value, ModelError> {
        Ok(Value {
            f32: x.f32.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect(),
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
        // 归一化宽度由 alpha 定义(per-channel alpha 广播;与 GPU owl_rmsnorm_f32 同一语义;
        // x 任意前导维折叠为行 —— [T, H×HD] × alpha [HD] = per-head 行归一化)
        let cols = alpha.f32.len().max(1);
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

    /// Layernorm(affine;two-pass mean/var —— 与 torch.nn.LayerNorm 同序;
    /// 与 `owl-cpu` / CUDA 实现**互为独立副本**,禁止互引)
    fn layernorm(
        &mut self,
        x: &Value,
        weight: &Value,
        bias: &Value,
        n: usize,
        eps: f32,
    ) -> Result<Value, ModelError> {
        let n = n.max(1);
        if weight.f32.len() < n || bias.f32.len() < n {
            return Err(ModelError::Msg(format!(
                "layernorm: weight/bias 短于行宽 {n}(得 {}/{})",
                weight.f32.len(),
                bias.f32.len()
            )));
        }
        let rows = x.f32.len() / n;
        let mut out = vec![0.0f32; rows * n];
        for r in 0..rows {
            let xs = &x.f32[r * n..(r + 1) * n];
            let mean = xs.iter().sum::<f32>() / n as f32;
            let var = xs.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
            let inv = 1.0 / (var + eps).sqrt();
            for c in 0..n {
                out[r * n + c] = (xs[c] - mean) * inv * weight.f32[c] + bias.f32[c];
            }
        }
        Ok(Value { f32: out, shape: x.shape.clone() })
    }

    fn rope_half_partial(
        &mut self,
        x: &Value,
        cos_t: &Value,
        sin_t: &Value,
        pos: &Value,
        heads: usize,
        head_dim: usize,
        half: usize,
    ) -> Result<Value, ModelError> {
        let row_len = heads * head_dim;
        if row_len == 0 || x.f32.len() % row_len != 0 {
            return Err(ModelError::Msg(format!(
                "rope: x 长 {} 不是 heads·head_dim = {row_len} 的整数倍",
                x.f32.len()
            )));
        }
        let rows = x.f32.len() / row_len;
        let table_rows = cos_t.f32.len() / half.max(1);
        let mut out = x.f32.clone();
        for t in 0..rows {
            let p = pos.f32[t] as usize;
            if p >= table_rows {
                return Err(ModelError::Msg(format!(
                    "rope: pos[{t}] = {p} 越出表行数 {table_rows}"
                )));
            }
            for h in 0..heads {
                let base = t * row_len + h * head_dim;
                for i in 0..half {
                    let a = x.f32[base + i];
                    let b = x.f32[base + i + half];
                    let (c, s) = (cos_t.f32[p * half + i], sin_t.f32[p * half + i]);
                    out[base + i] = a.mul_add(c, -(b * s));
                    out[base + i + half] = b.mul_add(c, a * s);
                }
            }
        }
        Ok(Value { f32: out, shape: x.shape.clone() })
    }

    fn narrow_strided(
        &mut self,
        src: &Value,
        outer: usize,
        src_dim: usize,
        start: usize,
        out_dim: usize,
    ) -> Result<Value, ModelError> {
        if (outer.saturating_sub(1)) * src_dim + start + out_dim > src.f32.len() {
            return Err(ModelError::Msg(format!(
                "narrow_strided: 读越界(outer {outer} src_dim {src_dim} start {start} out_dim {out_dim})"
            )));
        }
        let mut out = Vec::with_capacity(outer * out_dim);
        for r in 0..outer {
            let s = r * src_dim + start;
            out.extend_from_slice(&src.f32[s..s + out_dim]);
        }
        Ok(Value { f32: out, shape: vec![outer, out_dim] })
    }

    fn gelu_tanh(&mut self, x: &Value) -> Result<Value, ModelError> {
        // 0.5x(1+tanh(√(2/π)(x+0.044715x³)));与 owl-cpu 互为独立副本
        let c = (2.0f64 / std::f64::consts::PI).sqrt();
        let vals: Vec<f32> = x
            .f32
            .iter()
            .map(|v| {
                let vf = *v as f64;
                (0.5 * vf * (1.0 + (c * (vf + 0.044715 * vf * vf * vf)).tanh())) as f32
            })
            .collect();
        Ok(Value { f32: vals, shape: x.shape.clone() })
    }

    fn gelu_erf(&mut self, x: &Value) -> Result<Value, ModelError> {
        // 0.5x(1+erf(x/√2));erf 走 f64 级数(|x|>4 饱和 —— 误差 < 1.5e-8)
        fn erf_series(x: f64) -> f64 {
            if x.abs() > 4.0 {
                return x.signum();
            }
            let mut t = x;
            let mut acc = x;
            for n in 1..=200u32 {
                t *= -x * x / f64::from(n);
                let term = t / f64::from(2 * n + 1);
                acc += term;
                if term.abs() < 1e-18 {
                    break;
                }
            }
            2.0 / std::f64::consts::PI.sqrt() * acc
        }
        let vals: Vec<f32> = x
            .f32
            .iter()
            .map(|v| {
                let vf = *v as f64;
                (0.5 * vf * (1.0 + erf_series(vf / std::f64::consts::SQRT_2))) as f32
            })
            .collect();
        Ok(Value { f32: vals, shape: x.shape.clone() })
    }

    fn bias_add(&mut self, x: &Value, bias: &Value, cols: usize) -> Result<Value, ModelError> {
        let cols = cols.max(1);
        if bias.f32.len() < cols {
            return Err(ModelError::Msg(format!(
                "bias_add: bias 长 {} < cols {cols}",
                bias.f32.len()
            )));
        }
        let rows = x.f32.len() / cols;
        let mut out = x.f32.clone();
        for r in 0..rows {
            for c in 0..cols {
                out[r * cols + c] += bias.f32[c];
            }
        }
        Ok(Value { f32: out, shape: x.shape.clone() })
    }

    fn pos_embed_bilinear(
        &mut self,
        table: &Value,
        idx: &Value,
        wts: &Value,
        n_taps: usize,
        dim: usize,
    ) -> Result<Value, ModelError> {
        let (n_taps, dim) = (n_taps.max(1), dim.max(1));
        let rows = idx.f32.len() / n_taps;
        let n_pos = table.f32.len() / dim;
        let mut out = vec![0.0f32; rows * dim];
        for r in 0..rows {
            let mut acc = vec![0.0f32; dim];
            for k in 0..n_taps {
                let row = idx.f32[r * n_taps + k] as usize;
                if row >= n_pos {
                    return Err(ModelError::Msg(format!(
                        "pos_embed: idx[{r},{k}] = {row} 越出表行数 {n_pos}"
                    )));
                }
                let w = wts.f32[r * n_taps + k];
                let src = &table.f32[row * dim..(row + 1) * dim];
                for (d, s) in src.iter().enumerate() {
                    acc[d] += w * s;
                }
            }
            out[r * dim..(r + 1) * dim].copy_from_slice(&acc);
        }
        Ok(Value { f32: out, shape: vec![rows, dim] })
    }

    /// 图像 token 注入(独立副本;与 owl-cpu / CUDA 互为对照,禁止互引):
    /// 逐行判定,行号来自 f32 索引量,越界响亮拒。
    fn image_scatter(
        &mut self,
        h: &Value,
        merged: &Value,
        idx: &Value,
        dim: usize,
    ) -> Result<Value, ModelError> {
        let dim = dim.max(1);
        let rows = h.f32.len().checked_div(dim).filter(|r| *r * dim == h.f32.len()).ok_or_else(|| {
            ModelError::Msg(format!("image_scatter: h 长 {} 不是 dim {dim} 的整数倍", h.f32.len()))
        })?;
        let n_merged = merged.f32.len() / dim;
        let mut out = Vec::with_capacity(h.f32.len());
        for r in 0..rows {
            let pick = idx.f32.get(r).copied().ok_or_else(|| {
                ModelError::Msg(format!("image_scatter: idx 缺第 {r} 行(长 {})", idx.f32.len()))
            })?;
            // 负 = 文本位保留 embed;非负 = 取塔输出行(越界响亮拒)
            let (src, base) = if pick < 0.0 {
                (&h.f32, r * dim)
            } else {
                let row = pick as usize;
                if row >= n_merged {
                    return Err(ModelError::Msg(format!(
                        "image_scatter: idx[{r}] = {pick} 越出 merged 行数 {n_merged}"
                    )));
                }
                (&merged.f32, row * dim)
            };
            out.extend_from_slice(src.get(base..base + dim).ok_or_else(|| {
                ModelError::Msg(format!(
                    "image_scatter: 第 {r} 行源区间 [{base}, {base} + {dim}) 越界"
                ))
            })?);
        }
        Ok(Value { f32: out, shape: vec![rows, dim] })
    }

    fn attn_full(
        &mut self,
        q: &Value,
        k: &Value,
        v: &Value,
        cu: &Value,
        rows: usize,
        heads: usize,
        head_dim: usize,
        n_seq: usize,
        scale: f32,
    ) -> Result<Value, ModelError> {
        // 两趟 softmax(先 max 后加权和);与 CUDA 的在线 softmax 数学同值
        let mut out = vec![0.0f32; rows * heads * head_dim];
        let elem = |b: &Value, row: usize, head: usize, d: usize| b.f32[(row * heads + head) * head_dim + d];
        for row in 0..rows {
            let mut span = (0usize, 0usize);
            for s in 0..n_seq {
                let a = cu.f32[s] as usize;
                let b = cu.f32[s + 1] as usize;
                if a <= row && row < b {
                    span = (a, b);
                    break;
                }
            }
            let (start, end) = span;
            if start >= end {
                continue;
            }
            for head in 0..heads {
                let mut m = f32::NEG_INFINITY;
                let mut scores = vec![0.0f32; end - start];
                for (i, t) in (start..end).enumerate() {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot = elem(q, row, head, d).mul_add(elem(k, t, head, d), dot);
                    }
                    scores[i] = dot * scale;
                    m = m.max(scores[i]);
                }
                let mut l = 0.0f32;
                let mut acc = vec![0.0f32; head_dim];
                for (i, t) in (start..end).enumerate() {
                    let e = (scores[i] - m).exp();
                    l += e;
                    for d in 0..head_dim {
                        acc[d] = elem(v, t, head, d).mul_add(e, acc[d]);
                    }
                }
                let inv = if l > 0.0 { 1.0 / l } else { 0.0 };
                for d in 0..head_dim {
                    out[(row * heads + head) * head_dim + d] = acc[d] * inv;
                }
            }
        }
        Ok(Value { f32: out, shape: vec![rows, heads, head_dim] })
    }
}

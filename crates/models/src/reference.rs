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
use crate::ops::Op;
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
    /// Add(同形)
    fn add(&mut self, a: &Value, b: &Value) -> Result<Value, ModelError>;
    /// Mul(同形逐元素乘;MLP 门控/输出门)
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
        Op::Sigmoid => itp.sigmoid(&ins[0]),
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
}

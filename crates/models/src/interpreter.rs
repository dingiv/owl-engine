//! 解释器:同一份声明,多种执行(三种游走)。
//!
//! | 解释器 | Op 执行体 | 场景 |
//! |---|---|---|
//! | eager(档一) | 逐节点经 GpuFace 提交(提交即回) | decode 热路径 |
//! | capture(烘焙) | 逐节点"录"→ 整单 instantiate → 哨兵③对账 | 图捕获 |
//! | CPU(参考) | 纯 Rust 闭包(本文件提供) | nn 测试保持同步 |
//!
//! walk 共用:自叶向根归约(递归;深度 = 链长,几百层内无忧)。
//! 毒值在此落地:遇 Poisoned 节点 = 结构化报错(LazyError 归因)。

use crate::dtype::Dtype;
use crate::error::ModelError;
use crate::plan::Op;
use crate::shape::{numel, Shape};
use crate::tensor::TensorOps;

// ============================================================================
// §1 值(CPU 参考解释器的形态;server 版 = 不透明 Block 句柄)
// ============================================================================

/// f32 值块(CPU 参考;server 版为 GPU 池块句柄)
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

// ============================================================================
// §2 解释器 trait(cuda server 实现它;CPU 参考实现随附)
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
    /// Silu
    fn silu(&mut self, x: &Value) -> Result<Value, ModelError>;
    /// Rmsnorm(parents = [x, alpha];w_off = ×(1+w))
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

// ============================================================================
// §3 归约驱动(三种解释器共用的 walk)
// ============================================================================

/// 自叶向根归约。毒值在此落地;资源错误直接 Err 过线。
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
        Op::Silu => itp.silu(&ins[0]),
        Op::Rmsnorm { eps, w_off } => itp.rmsnorm(&ins[0], &ins[1], *eps, *w_off),
        Op::SlotWrite => itp.slot_write(&ins[0]),
        Op::Block { id } => itp.block(*id, t.dtype, &t.shape),
        // 逃逸舱:client 侧闭包就地执行(不经 GpuFace;server 派发表不见它)
        Op::UnaryFn { f } => Ok(f(ins[0].clone())),
        Op::BinaryFn { f } => Ok(f(ins[0].clone(), ins[1].clone())),
        other => Err(ModelError::Msg(format!(
            "CPU 参考解释器未覆盖: {other:?}(server 侧实现)"
        ))),
    }
}

// ============================================================================
// §4 CPU 参考解释器(朴素实现;对拍基准)
// ============================================================================

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
        let ms = x.f32.iter().map(|v| v * v).sum::<f32>() / x.f32.len() as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        Ok(Value {
            f32: x
                .f32
                .iter()
                .zip(&alpha.f32)
                .map(|(v, a)| {
                    let g = if w_off { a + 1.0 } else { *a };
                    v * inv * g
                })
                .collect(),
            shape: x.shape.clone(),
        })
    }
}

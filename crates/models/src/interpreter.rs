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
        eprintln!("[dbg mm] a.shape={:?} a.len={} b.shape={:?} b.len={} shape={:?}", a.shape, a.f32.len(), b.shape, b.f32.len(), shape);
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

// ============================================================================
// §5 CpuFace:CPU 解释器适配为 GpuFace(与 GPU server 同一契约)
// ============================================================================

use crate::client::{Arg, Bytes as HandleBytes, DeviceClient, LaunchMsg};
use std::collections::HashMap;

/// CPU 参考执行器(适配 GpuFace):与 GPU server 同一契约、同一管线,
/// 测试/对拍时作为 server 的替身。
pub struct CpuFace {
    itp: CpuInterpreter,
    blocks: HashMap<u64, Value>,
    next: u64,
}

impl Default for CpuFace {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuFace {
    pub fn new() -> Self {
        Self { itp: CpuInterpreter::new(), blocks: HashMap::new(), next: 1 }
    }
}

impl DeviceClient for CpuFace {
    /// 显存分配(清零;CPU = Vec)
    async fn alloc(&mut self, n_bytes: usize) -> Result<HandleBytes, ModelError> {
        let v = Value::zero(Dtype::F32, &vec![n_bytes / 4])?;
        let id = self.next;
        self.next += 1;
        self.blocks.insert(id, v);
        Ok(HandleBytes::new(id, 0))
    }

    /// 装载(host 字节 → f32 值块)
    async fn htod(
        &mut self,
        dtype: Dtype,
        shape: &Shape,
        src: &[u8],
    ) -> Result<HandleBytes, ModelError> {
        let v = self.itp.htod(dtype, shape, src)?;
        let id = self.next;
        self.next += 1;
        self.blocks.insert(id, v);
        Ok(HandleBytes::new(id, 0))
    }

    /// 收割(值 → LE 字节;长度须与块一致)
    async fn dtoh(&mut self, b: &HandleBytes, out: &mut [u8]) -> Result<(), ModelError> {
        let v = self
            .blocks
            .get(&b.id)
            .ok_or_else(|| ModelError::DeadBlock { id: b.id })?;
        let bytes: Vec<u8> = v
            .f32
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        if bytes.len() != out.len() {
            return Err(ModelError::Msg(format!(
                "dtoh: 块 {} 字节 {} != 收割 {}",
                b.id,
                bytes.len(),
                out.len()
            )));
        }
        out.copy_from_slice(&bytes);
        Ok(())
    }

    /// 发射:按 kernel 名路由到 CPU 参考实现(动作表的 CPU 面)
    async fn launch(&mut self, msg: LaunchMsg) -> Result<HandleBytes, ModelError> {

        eprintln!("[dbg launch] {} args={:?}", msg.kernel.name, msg.args);
        // 参数槽解析:Block → 值块;类型化标量
        let mut vals: Vec<Value> = Vec::with_capacity(msg.args.len());
        let mut u64s: Vec<u64> = Vec::new();
        let mut i32s: Vec<i32> = Vec::new();
        let mut f32s: Vec<f32> = Vec::new();
        let mut out_id: Option<u64> = None;
        for a in &msg.args {
            match a {
                Arg::Block { id } => {
                    out_id = Some(*id);
                    vals.push(
                        self.blocks
                            .get(id)
                            .cloned()
                            .ok_or(ModelError::DeadBlock { id: *id })?,
                    )
                }
                Arg::U64(v) => u64s.push(*v),
                Arg::I32(v) => i32s.push(*v),
                Arg::F32(v) => f32s.push(*v),
            }
        }
        let out = match msg.kernel.name.as_str() {
            "owl_add_f32" => self.itp.add(&vals[0], &vals[1])?,
            "owl_silu_f32" => self.itp.silu(&vals[0])?,
            "owl_matmul_f32" => {
                // 标量与 lower_matmul 对位:m/k/n
                let (m, _k, n) =
                    (i32s[0] as usize, i32s[1] as usize, i32s[2] as usize);
                let shape = crate::shape::Shape::from(vec![m, n]);
                self.itp.matmul(&vals[0], &vals[1], Dtype::F32, &shape)?
            }
            "owl_rmsnorm_f32" => {
                // 标量与 lower_rmsnorm 对位:n(I32)/eps(F32)/w_off(I32)
                self.itp.rmsnorm(&vals[0], &vals[1], f32s[0], i32s[1] != 0)?
            }
            other => {
                return Err(ModelError::Msg(format!(
                    "CpuFace::launch: 未注册的动作 {other}"
                )))
            }
        };
        // 写回 eval 预 alloc 的 out 块(与 GPU 语义一致:发射原位写输出)
        let id = out_id.ok_or_else(|| {
            ModelError::Msg("launch: args 中无输出块".to_string())
        })?;
        self.blocks.insert(id, out);
        Ok(HandleBytes::new(id, msg.out_elems))
    }

    async fn sync(&mut self) -> Result<(), ModelError> {
        Ok(()) // CPU:无在飞操作
    }
}


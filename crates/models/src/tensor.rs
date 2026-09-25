//! Tensor:声明链(TensorOps)+ 运行时数据(Tensor<D>)+ 标注词汇(Dtype)。
//!
//! 三层区分(2026-09-23/24 定稿):
//!   TensorOps  = 声明链(懒描述;反向多叉树;无数据)—— §2
//!   Tensor<D>  = 运行时数据(设备池块 + dtype/shape 标注)—— §3
//!   Dtype      = 标注词汇(S4 语义表的 dtype 维;权威在 iface::contract,
//!                此处 re-export)—— §1
//!
//! **结构定稿**:
//! - 值语义:深拷贝输入子树(配置面一次性成本);
//! - 无 Arc、无 Tx、无共享别名——纯值世界;
//! - 每节点携带全局唯一自增 id(跨线程;server 对账/缓存键),声明链与
//!   运行时张量共用同一套进程级身份证。
//! 
//! 
//! 那么这种设计，它的一个期望是说，哎，我让上层的这些代码量最大的这个layer声明和models声明，让它们的一些重复琐碎的代码尽可能的减少。然后呢，把副作用呢移出这些层。
//! 也不是说没有代价的，对吧？那我们啊需要定义更加复杂的这些结构和解释器去执行上层的这些声明式的语法。并且呢，它的能力是受限的。那么，但其实这也不能说是一个缺点啊，但是它是一个。啊，不方便的一个点。啊，但是这没有关系啊，我们可以在它能力不足的时候啊，我们可以为它扩充一些新的能力.

use crate::device::Device;
use crate::error::{LazyError, ModelError};
use crate::kernel::Kernel;
use crate::plan::{KernelArg, Op};
use crate::shape::{numel, Shape};
use std::sync::atomic::{AtomicU64, Ordering};

// ============================================================================
// §1 Dtype:标注词汇(权威在 iface::contract;re-export 保路径)
// ============================================================================

pub use owl_iface::contract::Dtype;

// ============================================================================
// §2 TensorOps:声明式链式 API(客户主入口)
// ============================================================================

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

fn next_id() -> u64 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

/// 张量声明值。clone = 深拷贝遍历子树(配置面一次性成本)。
#[derive(Clone)]
pub struct TensorOps {
    /// 全局唯一 id(跨线程自增;进程级身份证)
    pub(crate) id: u64,
    /// 反向边:本节点归约所需的全部输入(叶子为空;深拷贝子树)
    pub(crate) parents: Vec<TensorOps>,
    /// 拓扑深度(归约排序 + 错误归因坐标)
    pub(crate) depth: u32,
    /// 语义运算
    pub(crate) op: Op,
    /// 标注(client 侧;server 是字节世界)
    pub(crate) dtype: Dtype,
    pub(crate) shape: Shape,
    /// Kernel 节点的有序参数槽(T = 张量依赖 / Bits = 标量位型)
    pub(crate) args: Vec<KernelArg>,
    /// 毒值(构造期违约;随子树透传)
    pub(crate) err: Option<LazyError>,
}

impl TensorOps {
    // ======================================================================
    // 查询
    // ======================================================================

    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// 拓扑深度(错误归因坐标;flatten 的排序键)
    pub fn depth(&self) -> u32 {
        self.depth
    }

    pub fn is_poisoned(&self) -> bool {
        self.err.is_some()
    }

    /// 自身(解释器归约入口;值语义下 TensorOps 就是根节点)
    pub fn step(&self) -> &TensorOps {
        self
    }

    /// 节点身份证(错误归因/server 对账/缓存键)
    pub fn id(&self) -> u64 {
        self.id
    }

    /// 展平:自根收集整棵遍历子树,按深度升序(审计/烘焙的原材料)。
    /// 值语义下子树无共享(深拷贝),朴素 DFS 即完备。
    pub fn flatten(&self) -> Vec<&TensorOps> {
        let mut all = Vec::new();
        let mut stack = vec![self];
        while let Some(t) = stack.pop() {
            all.push(t);
            for p in &t.parents {
                stack.push(p);
            }
        }
        all.sort_by_key(|t| t.depth);
        all
    }

    // ======================================================================
    // 工厂(声明"需要一个块";执行在 eval 边界)
    // ======================================================================

    /// 清零分配声明
    pub fn zeros(dtype: Dtype, shape: Shape) -> TensorOps {
        TensorOps {
            id: next_id(),
            parents: vec![],
            depth: 0,
            op: Op::Zeros,
            dtype,
            shape,
            args: vec![],
            err: None,
        }
    }

    /// host 数据声明(pinned 码头直填归 server;契约 3)。
    /// 值语义:数据随树走。
    pub fn from_host(dtype: Dtype, shape: Shape, data: &[u8]) -> TensorOps {
        debug_assert_eq!(
            shape.iter().product::<usize>() * dtype.size_bytes(),
            data.len(),
            "from_host: 字节数不符"
        );
        TensorOps {
            id: next_id(),
            parents: vec![],
            depth: 0,
            op: Op::Htod { bytes: data.to_vec() },
            dtype,
            shape,
            args: vec![],
            err: None,
        }
    }

    /// 从已物化数据块构造声明叶子(Op::Block;数据由 server 按 id 解析)。
    /// 注:声明叶子的 id = 物化数据的 id(同一身份证;server 侧账房按它反查)。
    pub fn of_block(id: u64, dtype: Dtype, shape: Shape) -> TensorOps {
        TensorOps {
            id,
            parents: vec![],
            depth: 0,
            op: Op::Block { id },
            dtype,
            shape,
            args: vec![],
            err: None,
        }
    }

    /// Kernel 节点声明:`TensorOps::of(kernel).arg(..).arg(..)`
    /// 参数按序入槽;参数与真实 kernel 签名的一致性由后端最终裁决
    /// (INVALID_VALUE = 结构化报错)。
    pub fn of(kernel: Kernel) -> TensorOps {
        TensorOps {
            id: next_id(),
            parents: vec![],
            depth: 0,
            op: Op::Kernel { kernel },
            dtype: Dtype::F32,
            shape: vec![],
            args: vec![],
            err: None,
        }
    }

    /// Kernel 节点输出形状标注(of 默认 F32 空形状;eval 按此 alloc 输出块)
    pub fn with_shape(mut self, dtype: Dtype, shape: Shape) -> TensorOps {
        self.dtype = dtype;
        self.shape = shape;
        self
    }

    /// 毒值声明工厂(外部违约出口:槽未装载等;eval 边界收割,零 panic)
    pub fn poisoned(dtype: Dtype, shape: Shape, detail: impl Into<String>) -> TensorOps {
        TensorOps {
            id: next_id(),
            parents: vec![],
            depth: 0,
            op: Op::Zeros,
            dtype,
            shape,
            args: vec![],
            err: Some(LazyError { at_depth: 0, detail: detail.into() }),
        }
    }

    /// 视图:纯元数据收窄,零节点追加,零 server 往返
    pub fn narrow(&self, dim: usize, _start: usize, len: usize) -> TensorOps {
        let mut shape = self.shape.clone();
        shape[dim] = len;
        TensorOps {
            id: next_id(),
            parents: vec![],
            depth: self.depth,
            op: Op::Zeros,
            dtype: self.dtype,
            shape,
            args: vec![],
            err: None,
        }
        // 注:narrow 的 zero-op 占位实施期改为视图节点(共享 parents,不复制)
    }

    // ======================================================================
    // 链式运算(声明;total——毒传播,永不失败)
    // ======================================================================

    /// [m,k] × [k,n] → [m,n]
    pub fn matmul(&self, b: &TensorOps) -> TensorOps {
        if let Some(e) = self.shape_rule(b, "matmul", |a, b| a.last() == b.first()) {
            return self.poisoned_local(e);
        }
        let mut shape = self.shape.clone();
        let last = shape.len() - 1;
        shape[last] = b.shape.last().copied().unwrap_or(0);
        let meta = (self.dtype, shape);
        self.join(Op::Matmul, Some(b), meta, vec![])
    }

    pub fn add(&self, b: &TensorOps) -> TensorOps {
        if let Some(e) = self.shape_rule(b, "add", |a, b| a == b) {
            return self.poisoned_local(e);
        }
        let meta = (self.dtype, self.shape.clone());
        self.join(Op::Add, Some(b), meta, vec![])
    }

    /// 同形逐元素乘(MLP 门控 / 注意力输出门)
    pub fn mul(&self, b: &TensorOps) -> TensorOps {
        if let Some(e) = self.shape_rule(b, "mul", |a, b| a == b) {
            return self.poisoned_local(e);
        }
        let meta = (self.dtype, self.shape.clone());
        self.join(Op::Mul, Some(b), meta, vec![])
    }

    pub fn silu(&self) -> TensorOps {
        let meta = (self.dtype, self.shape.clone());
        self.join(Op::Silu, None, meta, vec![])
    }

    /// ×(1+w) 语义(w_off = true;use_norm_offset)
    /// 二元(x + alpha 双 parent):alpha 必须入树,归约才有输入
    pub fn rmsnorm(&self, alpha: &TensorOps, eps: f32, w_off: bool) -> TensorOps {
        if let Some(e) = self.shape_rule(alpha, "rmsnorm", |a, b| a.last() == b.last()) {
            return self.poisoned_local(e);
        }
        let meta = (self.dtype, self.shape.clone());
        self.join(
            Op::Rmsnorm { eps, w_off },
            Some(alpha),
            meta,
            vec![],
        )
    }

    // ======================================================================
    // 参数槽(Kernel 节点;按序入包)
    // ======================================================================

    /// 参数入包(张量 → 依赖 + 有序槽)
    pub fn arg(self, t: &TensorOps) -> TensorOps {
        let mut out = self;
        out.args.push(KernelArg::T { id: t.id });
        out.parents.push(t.clone());
        out
    }

    /// 标量参数入包(位型原样)
    pub fn arg_bits(self, bits: u64) -> TensorOps {
        let mut out = self;
        out.args.push(KernelArg::Bits(bits));
        out
    }

    /// 4 字节浮点参数(与 kernel `float` 形参严格对位)
    pub fn arg_f32(self, v: f32) -> TensorOps {
        let mut out = self;
        out.args.push(KernelArg::F32(v));
        out
    }

    pub fn arg_usize(self, v: usize) -> TensorOps {
        self.arg_bits(v as u64)
    }

    /// 4 字节整数参数(与 kernel `int` 形参严格对位)
    pub fn arg_i32(self, v: i32) -> TensorOps {
        let mut out = self;
        out.args.push(KernelArg::I32(v));
        out
    }

    pub fn arg_bool(self, v: bool) -> TensorOps {
        self.arg_bits(v as u64)
    }

    // ======================================================================
    // 内部
    // ======================================================================

    fn shape_rule(
        &self,
        b: &TensorOps,
        who: &str,
        ok: impl Fn(&[usize], &[usize]) -> bool,
    ) -> Option<LazyError> {
        (!ok(&self.shape, &b.shape)).then(|| LazyError {
            at_depth: self.depth + 1,
            detail: format!("{who}: 形状不符 {:?} vs {:?}", self.shape, b.shape),
        })
    }

    fn poisoned_local(&self, e: LazyError) -> TensorOps {
        TensorOps {
            id: next_id(),
            parents: vec![self.clone()],
            depth: self.depth + 1,
            op: Op::Zeros,
            dtype: self.dtype,
            shape: self.shape.clone(),
            args: vec![],
            err: self.err.clone().or(Some(e)),
        }
    }

    /// append:深拷贝输入子树进新节点(值语义;配置面一次性成本)
    fn join(&self, op: Op, rhs: Option<&TensorOps>, meta: (Dtype, Shape), args: Vec<KernelArg>) -> TensorOps {
        let mut parents = vec![self.clone()];
        if let Some(r) = rhs {
            parents.push(r.clone());
        }
        let err = self.err.clone().or(rhs.and_then(|r| r.err.clone()));
        let depth = parents.iter().map(|p| p.depth).max().unwrap_or(0) + 1;
        let (dtype, shape) = meta;
        TensorOps {
            id: next_id(),
            parents,
            depth,
            op,
            dtype,
            shape,
            args,
            err,
        }
    }
}


// ============================================================================
// §3 Tensor<D>:运行时数据张量(跨设备统一表达)
// ============================================================================
// id 与 TensorOps 共用同一套进程级身份证(§2 的 NEXT_ID)

/// 运行时数据张量。
/// - `block`:设备池块(数据所在);`offset`:块内字节偏移(视图);
/// - clone = 共享同一底仓 + 同一偏移(视图克隆,零拷贝);
/// - narrow/reshape 纯元数据;数据写入经设备(装载/算子),不经本类型。
pub struct Tensor<D: Device> {
    dev: D,
    block: D::Bytes,
    /// 块内字节偏移(视图;数据本体从 offset 起读)
    offset: usize,
    dtype: Dtype,
    shape: Shape,
    id: u64,
}

impl<D: Device> Tensor<D> {
    /// 从既有池块装配(纯元数据;容量守卫——cat 案教训)
    pub fn from_bytes(
        dev: D,
        dtype: Dtype,
        shape: Shape,
        block: D::Bytes,
    ) -> Result<Self, ModelError> {
        let n = numel(&shape);
        let need = n * dtype.size_bytes();
        let cap = dev.capacity(&block);
        if need > cap {
            return Err(ModelError::Msg(format!(
                "Tensor::from_bytes: 需 {need}B > 池块 {cap}B"
            )));
        }
        Ok(Self {
            dev,
            block,
            offset: 0,
            dtype,
            shape,
            id: next_id(),
        })
    }

    /// 装载声明 → 执行(host 数据经设备 htod;同步面)
    pub fn from_host(dev: D, dtype: Dtype, shape: Shape, data: &[u8]) -> Result<Self, ModelError> {
        let n = numel(&shape);
        let want = n * dtype.size_bytes();
        if data.len() != want {
            return Err(ModelError::Msg(format!(
                "from_host: 字节数 {} != {want}(dtype×shape)",
                data.len()
            )));
        }
        let block = dev.htod(data)?;
        Ok(Self {
            dev,
            block,
            offset: 0,
            dtype,
            shape,
            id: next_id(),
        })
    }

    /// 清零分配
    pub fn zeros(dev: D, dtype: Dtype, shape: Shape) -> Result<Self, ModelError> {
        let n = numel(&shape);
        let block = dev.alloc(n * dtype.size_bytes())?;
        Ok(Self {
            dev,
            block,
            offset: 0,
            dtype,
            shape,
            id: next_id(),
        })
    }

    // ======================================================================
    // 查询
    // ======================================================================

    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn nbytes(&self) -> usize {
        numel(&self.shape) * self.dtype.size_bytes()
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn device(&self) -> D {
        self.dev.clone()
    }

    /// 视图:dim 维收窄(零拷贝;offset 随 start 走)
    /// (仅演示最简形态:dim=0 且行主序时 offset += start × 行宽)
    pub fn narrow_dim0(&self, start: usize, len: usize) -> Result<Self, ModelError> {
        if self.shape.is_empty() || start + len > self.shape[0] {
            return Err(ModelError::Msg(format!(
                "narrow_dim0: [{start}..{}) 越界 shape {:?}",
                start,
                self.shape
            )));
        }
        let row = if self.shape.len() > 1 {
            self.shape[1..].iter().product::<usize>() * self.dtype.size_bytes()
        } else {
            self.dtype.size_bytes()
        };
        let t = Tensor {
            dev: self.dev.clone(),
            block: self.block.clone(),
            offset: self.offset + start * row,
            dtype: self.dtype,
            shape: {
                let mut s = self.shape.clone();
                s[0] = len;
                s
            },
            id: next_id(),
        };
        Ok(t)
    }

    // ======================================================================
    // 数据出入(边界;同步面。GPU 的 async 包装在 client.rs 一层)
    // ======================================================================

    /// device → host 收割(全部数据)
    /// 桥:物化数据 → 声明叶子(Block 节点引用本块的 id)。
    /// 数据本体不动;声明图 eval 时该叶子零操作。
    pub fn as_declaration(&self) -> TensorOps {
        TensorOps::of_block(
            self.id,
            self.dtype,
            self.shape.clone(),
        )
    }

    pub fn to_host(&self) -> Result<Vec<u8>, ModelError> {
        let mut out = vec![0u8; self.nbytes()];
        // 视图收割:块内 offset 起(容量守卫在设备层)
        self.dev.dtoh(&self.block, &mut out)?;
        // offset 版本:设备需支持子区读——一期约束 = narrow 视图暂不支持
        // 跨 offset 收割(需要 server dtoh 带 offset 参数,A3 期扩)。
        if self.offset != 0 {
            return Err(ModelError::Msg(
                format!("to_host: 带偏移视图收割待 server dtoh 扩 offset 参数(offset={})", self.offset),
            ));
        }
        Ok(out)
    }
}

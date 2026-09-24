//! TensorOps:声明式链式 API(客户主入口)。
//!
//! TensorOps = 反向多叉树上某节点的视图 + dtype/shape 标注。
//! 命名注记:它是**声明链**(算子的懒描述),不是数据张量;
//! 数据张量 = eval 之后的产物(将来另立类型或句柄)。
//! 运算 = 声明(构造新节点,值语义深拷贝输入子树),不执行。
//! 链式:一元 = 方法;二元 = 方法吃 &TensorOps;毒传播;视图零往返。
//!
//! **结构定稿(2026-09-23 裁决:Step/TensorMeta 解体,字段全上 TensorOps)**:
//! 值语义 + 无 Arc + 无 Tx + 无中间类型——一个 struct 就是全部。

use crate::dtype::Dtype;
use crate::error::LazyError;
use crate::plan::Op;
use crate::shape::Shape;
use std::sync::atomic::{AtomicU64, Ordering};

/// 全局自增节点 id:跨线程唯一(AtomicU64;进程级身份证)。
/// 用途:错误归因坐标(LazyError.at_id)/ server 侧节点对账 / 缓存键。
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

fn next_id() -> u64 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

/// 张量声明值。clone = 深拷贝遍历子树(值语义;配置面一次性成本)。
pub struct TensorOps {
    /// 全局唯一 id(跨线程自增;进程级身份证)
    pub(crate) id: u64,
    /// 反向边:本节点归约所需的全部输入(叶子为空)
    pub(crate) parents: Vec<TensorOps>,
    /// 拓扑深度(归约排序 + 错误归因坐标)
    pub(crate) depth: u32,
    /// 语义运算
    pub(crate) op: Op,
    /// 标注(client 侧;server 是字节世界)
    pub(crate) dtype: Dtype,
    pub(crate) shape: Shape,
    /// 毒值(构造期违约;随子树透传)
    pub(crate) err: Option<LazyError>,
}

impl Clone for TensorOps {
    /// 深拷贝 = 整棵子树重排**全新 id**(每个节点都是新身份证)。
    /// id 唯一性因此是全进程不变量:任何时刻不存在两个同 id 的 TensorOps。
    fn clone(&self) -> Self {
        let mut parents = Vec::with_capacity(self.parents.len());
        for p in &self.parents {
            parents.push(p.clone());
        }
        TensorOps {
            id: next_id(),
            parents,
            depth: self.depth,
            op: self.op.clone(),
            dtype: self.dtype,
            shape: self.shape.clone(),
            err: self.err.clone(),
        }
    }
}

impl TensorOps {
    /// 从已物化数据块构造声明叶子(Op::Block;数据由 server 按 id 解析)。
    /// 注:声明叶子的 id = 物化数据的 id(同一身份证;server 侧账房按它反查)
    pub fn of_block(id: u64, dtype: Dtype, shape: Shape) -> TensorOps {
        TensorOps {
            id,
            parents: vec![],
            depth: 0,
            op: Op::Block { id },
            dtype,
            shape,
            err: None,
        }
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
            err: None,
        }
    }

    /// 视图:纯元数据收窄,零节点追加,零 server 往返
    pub fn narrow(&self, dim: usize, _start: usize, len: usize) -> TensorOps {
        let mut shape = self.shape.clone();
        shape[dim] = len;
        TensorOps { id: next_id(), parents: vec![], depth: self.depth, op: Op::Zeros, dtype: self.dtype, shape, err: None }
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
        self.join(Op::Matmul, Some(b), self.dtype, shape)
    }

    pub fn add(&self, b: &TensorOps) -> TensorOps {
        if let Some(e) = self.shape_rule(b, "add", |a, b| a == b) {
            return self.poisoned_local(e);
        }
        self.join(Op::Add, Some(b), self.dtype, self.shape.clone())
    }

    pub fn silu(&self) -> TensorOps {
        self.join(Op::Silu, None, self.dtype, self.shape.clone())
    }

    /// ×(1+w) 语义(w_off = true;use_norm_offset)
    /// 二元(x + alpha 双 parent):alpha 必须入树,归约才有输入
    pub fn rmsnorm(&self, alpha: &TensorOps, eps: f32, w_off: bool) -> TensorOps {
        if let Some(e) = self.shape_rule(alpha, "rmsnorm", |a, b| a.last() == b.last()) {
            return self.poisoned_local(e);
        }
        self.join(
            Op::Rmsnorm { eps, w_off },
            Some(alpha),
            self.dtype,
            self.shape.clone(),
        )
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
            err: self.err.clone().or(Some(e)),
        }
    }

    /// append:深拷贝输入子树进新节点(值语义;配置面一次性成本)
    pub(crate) fn join(&self, op: Op, rhs: Option<&TensorOps>, dtype: Dtype, shape: Shape) -> TensorOps {
        let mut parents = vec![self.clone()];
        if let Some(r) = rhs {
            parents.push(r.clone());
        }
        let err = self.err.clone().or(rhs.and_then(|r| r.err.clone()));
        let depth = parents.iter().map(|p| p.depth).max().unwrap_or(0) + 1;
        TensorOps { id: next_id(), parents, depth, op, dtype, shape, err }
    }
}

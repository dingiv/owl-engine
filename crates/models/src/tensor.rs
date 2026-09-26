//! Tensor:声明链(TensorOps)+ 标注词汇(Dtype)。
//!
//! 两层区分(2026-09-26 收束:旧 Tensor<D>/Device 同步面已废,唯一标准
//! = owl-iface DeviceClient,见 iface contract「顺序语义」):
//!   TensorOps  = 声明链(懒描述;反向多叉树;无数据)
//!   Dtype      = 标注词汇(S4 语义表的 dtype 维;权威在 iface::contract,
//!                此处 re-export)
//! 运行时数据形态 = 池块句柄 `Bytes`(eval memo;server 账房)+
//! [`crate::reference::Value`](host 锚)。若将来需要带标注的运行时
//! 张量视图,以 DeviceClient 为底重立,勿复活同步 Device 面。
//!
//! ═══════════════════════════════════════════════════════════════════
//! §0.0 TensorOps 语义规格(组件契约;含 2026-09-26 重复求值案定谳)
//! ═══════════════════════════════════════════════════════════════════
//!
//! **设计期望**:让代码量最大的层声明 / models 声明,重复琐碎的样板尽
//! 量减少;把副作用移出层——层只产出纯描述(值),执行归解释器。
//! 代价:需要更复杂的结构与解释器来执行声明式语法,且描述能力受限。
//! 这不算缺点,但确实不方便——能力不足时,为它扩充新能力即可。
//!
//! **图语义:算子单出度,值多出度。**
//! - 算子:多入度、**单出度**(一个算子产出一个输出值;C2 契约,
//!   kernel 输出块固定末参)——这条不变;
//! - 值:出度 = 被多少消费者引用,**可为任意数**(残差网络的定义:
//!   每个残差连接都是一个值出度 ≥ 2 的菱形)。
//! 旧文档把这两者混为一谈,把"值出度"钉死为 1(树语义)——这正是
//! 2026-09-26 重复求值案的语义根因(见下)。
//!
//! **值形态与两个后果**:
//! 1. `Clone` = 深拷贝整棵子树;**id 字段原样保留,不重新分配**。
//!    ⇒ 同一逻辑节点无论被克隆多少份、嵌在树的多少个位置,id 相同
//!    ⇒ "同 id = 同节点 = 同值",这是全组件最重要的不变量;
//! 2. 由于深拷贝,共享节点在声明树里**物理存在多份**(DAG 的树形展开)。
//!
//! **⚠️ 重复求值陷阱(2026-09-26 实录,勿再踩)**:
//! 解释器若无 CSE,按树递归求值会把共享节点**执行多次**:
//! - 纯函数 kernel:只是浪费算力,数值不变;
//! - **带状态副作用的 kernel(conv/delta 写状态):第二次执行读到第一
//!   次滑过的状态 → 输出不同 → 语义破坏**。实测:DecoderLayer 残差
//!   h 同时喃 ln2 与残差加,mixer 执行两遍,mlp 段输出恒定 ×1.017
//!   偏差,bisect 两小时定位。
//! 因此解释器(_eval)必须带 **CSE memo**(键 = 节点 id,作用域 = 单次
//! eval):同节点只执行一次,多消费者共享同一输出块。菱形、任意出度
//! 菱形都在保护范围内;跨 eval 不复用(每步状态必须推进)。
//!
//! **方案裁决(用户拍板)**:曾在构造期去重(图注册表:同节点直接连接)
//! 与求值期 memo 之间二选一——两者对【求值结果】完全等价,遂取实现
//! 简单的后者(求值期 memo),构造期保持纯值形态。句柄化/图 arena
//! 暂不立项,列为 M-f 图捕获期再评估(届时 memo 可升级为节点表)。
//!
//! **id 规则(三条)**:
//! 1. 新节点构造 → `next_id()` 全局自增;
//! 2. 克隆 → id 原样保留(同节点不变量,memo 键正确的根基);
//! 3. `of_block` 声明叶子 **也走 next_id()** —— 块 id 是 server 侧另一套
//!    计数,直接复用会撞 memo 键(实测 Rmsnorm gamma 撞 [8] 声明的
//!    64 账长块)。块 id 只存在 `Op::Block{id}` 数据字段里。
//!
//! **结构定稿**:
//! - 值语义:深拷贝输入子树(配置面一次性成本;执行期由 memo 保证
//!   DAG 语义,见上);
//! - 无 Arc、无 Tx、无共享别名——纯值世界;
//! - 每节点携带全局唯一自增 id(跨线程;server 对账/缓存键),声明链
//!   与运行时张量共用同一套进程级身份证。
//!
//! **这套设计的期望**:让代码量最大的 layer 声明和 models 声明,重复
//! 琐碎的样板尽可能减少,并把副作用移出层(层零执行)。代价:需要
//! 更复杂的结构(解释器/图求值)去执行上层的声明式语法,且描述能力
//! 受限(无控制流、无动态形状)。这不算缺点,但不方便——能力不足
//! 时,为它扩充新能力即可(历次扩项:Reshape/Kernel 节点/CSE/DAG)。

use crate::contract::{ModelError, Shape};
use crate::kernel::Kernel;
use crate::ops::{KernelArg, Op};
use std::sync::atomic::{AtomicU64, Ordering};

// ============================================================================
// §0 LazyError:描述层毒值载荷(server 不感知)
// ============================================================================

/// 毒值载荷:案发坐标 + 细节。反向树可沿 parents 反演——错误现场永远可重放。
/// (执行层资源错误是另一族:ModelError 直接 Err 过线,不经毒值;
///  两族错误分而治之,原 error.rs 已并入本文件与 contract.rs。)
#[derive(Debug, Clone)]
pub struct LazyError {
    /// 案发节点深度(归约链上的坐标,供回溯)
    pub at_depth: u32,
    /// 案发算子 + 双方元数据(如 "add: 形状不符 [4] vs [3]")
    pub detail: String,
}

// ============================================================================
// §1 Dtype:标注词汇(权威在 iface::contract;re-export 保路径)
// ============================================================================

pub use crate::contract::Dtype;

// ============================================================================
// §2 TensorOps:声明式链式 API(客户主入口)
// ============================================================================

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

fn next_id() -> u64 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

/// 张量声明值。clone = 浅拷贝(parents 为 Arc 共享,克隆 O(1))。
#[derive(Clone)]
pub struct TensorOps {
    /// 全局唯一 id(跨线程自增;进程级身份证)
    pub(crate) id: u64,
    /// 反向边:本节点归约所需的全部输入(叶子为空;Arc 共享,物理 DAG)
    pub(crate) parents: Vec<std::sync::Arc<TensorOps>>,
    /// 拓扑深度(归约排序 + 错误归因坐标)
    pub(crate) depth: u32,
    /// 语义运算
    pub(crate) op: Op,
    /// 标注(client 侧;server 是字节世界)
    pub(crate) dtype: Dtype,
    pub(crate) shape: Shape,
    /// Kernel 节点的**标量**参数槽(有序;Bits/I32/F32)。
    /// 张量依赖 = parents(自身输出类型声明即 f32 块,消费方查父即可,
    /// 2026-09-26 裁决:args 不再重复标 T);槽序权威 = kernel 签名。
    pub(crate) args: Vec<KernelArg>,
    /// 毒值(构造期违约;随子树透传)
    pub(crate) err: Option<LazyError>,
    /// 语义坐标标注(观测面 tap 的定位键;纯标注非身份 —— **同 id 必同
    /// label**,克隆保留;执行路径零参与。见 interpreters/observe.rs 与
    /// docs/arch/interpreter-tap.md §三)
    pub(crate) label: Option<std::sync::Arc<str>>,
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

    /// 语义坐标(观测面;None = 未打标)
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// 打标:语义坐标(观测面 tap 的定位键;interpreter-tap.md §三)。
    /// clone 本节点设标注 —— 同 id 不变(CSE memo/双锚对齐不受影响),
    /// O(1)(parents Arc 共享,浅拷贝)。层根自动打标见 Model::last_hidden。
    pub fn tag(self, s: impl Into<std::sync::Arc<str>>) -> TensorOps {
        let mut c = self;
        c.label = Some(s.into());
        c
    }

    /// 展平:自根收集整棵遍历子树,按深度升序(审计/烘焙的原材料)。
    /// 物理共享(Arc parents,2026-09-26)下按 id 去重 —— 同节点只收一次。
    pub fn flatten(&self) -> Vec<&TensorOps> {
        let mut all = Vec::new();
        let mut seen = std::collections::HashSet::new();
        let mut stack = vec![self];
        while let Some(t) = stack.pop() {
            if !seen.insert(t.id) {
                continue;
            }
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
            label: None,
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
            label: None,
        }
    }

    /// 从已物化数据块构造声明叶子(Op::Block;数据由 server 按 id 解析)。
    /// 注:声明叶子 id 必须走 next_id() 全局节点空间 —— 块 id 是 server 侧
    /// 另一套计数,直接复用会与普通节点 id 冲突(CSE memo 键撞车,
    /// 2026-09-26 实测:Rmsnorm 的 gamma 撞上 [8] 声明的 64 账长块)。
    pub fn of_block(id: u64, dtype: Dtype, shape: Shape) -> TensorOps {
        TensorOps {
            id: next_id(),
            parents: vec![],
            depth: 0,
            op: Op::Block { id },
            dtype,
            shape,
            args: vec![],
            err: None,
            label: None,
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
            label: None,
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
            label: None,
        }
    }

    /// 视图:narrow 收窄(未实装,显式毒 —— 勿用)。
    ///
    /// 窄切需求现走 Kernel 节点 `owl_narrow_strided_f32`(物化拷贝,
    /// 见 layers/attention 的 gate 切分);真视图(共享底仓 + 偏移)牵动
    /// 全 kernel 的 stride 语义,归线 B / view 立项(roadmap.local/
    /// api-stabilize-plan.md C3)。调用即毒,eval 边界收割。
    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> TensorOps {
        let mut shape = self.shape.clone();
        shape[dim] = len;
        let _ = start; // 语义占位:真视图实现时用于偏移
        self.poisoned_local(LazyError {
            at_depth: self.depth + 1,
            detail: "narrow: 未实装(显式封雷)—— 窄切走 owl_narrow_strided_f32 kernel 节点".into(),
        })
    }

    /// 视图:重解释形状(纯元数据;元素数守恒,违约即毒)。
    /// eval 透传父块(零 kernel 零拷贝);qk-norm 的 [T×H, HD] / [T, H×HD]
    /// 双形态需求即本节点(2026-09-26 C6 立项)。
    pub fn reshape(&self, shape: Shape) -> TensorOps {
        let want: usize = shape.iter().product();
        let have: usize = self.shape.iter().product();
        if want != have {
            return self.poisoned_local(LazyError {
                at_depth: self.depth + 1,
                detail: format!("reshape: 元素数不符 {have}({:?}) → {want}({:?})", self.shape, shape),
            });
        }
        self.join(Op::Reshape, None, (self.dtype, shape), vec![])
    }

    // ======================================================================
    // 链式运算(声明;total——毒传播,永不失败)
    // ======================================================================

    /// [m,k] × [k,n] → [m,n]
    /// nt 矩阵乘:B 以 [n, k] 行主序直读(tied lm_head:权重保持
    /// checkpoint [vocab, hidden] 原布局,零 host 转置零第二份显存)
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

    /// nt 矩阵乘:B 以 [n, k] 行主序直读(tied lm_head:权重保持
    /// checkpoint [vocab, hidden] 原布局,零 host 转置零第二份显存)
    pub fn matmul_nt(&self, b: &TensorOps) -> TensorOps {
        if let Some(e) = self.shape_rule(b, "matmul_nt", |a, b| a.last() == b.last()) {
            return self.poisoned_local(e);
        }
        let mut shape = self.shape.clone();
        let last = shape.len() - 1;
        // nt:n = B 的**行数**(B [n,k],k = 内维 = a.last == b.last)
        shape[last] = b.shape.first().copied().unwrap_or(0);
        let meta = (self.dtype, shape);
        self.join(Op::MatmulNt, Some(b), meta, vec![])
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

    /// 逐元素 sigmoid(注意力输出门;GDN beta 同族)
    pub fn sigmoid(&self) -> TensorOps {
        let meta = (self.dtype, self.shape.clone());
        self.join(Op::Sigmoid, None, meta, vec![])
    }

    /// ×(1+w) 语义(w_off = true;use_norm_offset)
    /// 二元(x + alpha 双 parent):alpha 必须入树,归约才有输入。
    /// 形状律:归一化宽度由 alpha 定义 —— x 任意前导维折叠为行,
    /// x.total % alpha.len() == 0 即可(Qwen3.5 qk-norm:[T, H×HD] ×
    /// alpha [HD] = per-head 行归一化,同 HF flatten(0,1) 语义)
    pub fn rmsnorm(&self, alpha: &TensorOps, eps: f32, w_off: bool) -> TensorOps {
        if let Some(e) = self.shape_rule(alpha, "rmsnorm", |a, b| {
            let cols: usize = b.iter().product();
            let total: usize = a.iter().product();
            cols > 0 && total % cols == 0
        }) {
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
        out.parents.push(std::sync::Arc::new(t.clone()));
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
            parents: vec![std::sync::Arc::new(self.clone())],
            depth: self.depth + 1,
            op: Op::Zeros,
            dtype: self.dtype,
            shape: self.shape.clone(),
            args: vec![],
            err: self.err.clone().or(Some(e)),
            label: None,
        }
    }

    /// append:深拷贝输入子树进新节点(值语义;配置面一次性成本)
    fn join(&self, op: Op, rhs: Option<&TensorOps>, meta: (Dtype, Shape), args: Vec<KernelArg>) -> TensorOps {
        let mut parents = vec![std::sync::Arc::new(self.clone())];
        if let Some(r) = rhs {
            parents.push(std::sync::Arc::new(r.clone()));
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
            label: None,
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::ModelError;
    use crate::reference::{reduce, CpuInterpreter};

    fn t(v: &[f32], shape: Shape) -> TensorOps {
        TensorOps::from_host(Dtype::F32, shape, &crate::testkit::f32b(v))
    }

    #[test]
    fn chain_matches_host_reference() {
        let out = t(&[1.0, 2.0, -3.0, 4.0], vec![1, 4]).silu();
        let got = reduce(out.step(), &mut CpuInterpreter::new()).unwrap();
        let want: Vec<f32> = [1.0f32, 2.0, -3.0, 4.0]
            .iter()
            .map(|v| v / (1.0 + (-v).exp()))
            .collect();
        assert!(got.f32.iter().zip(&want).all(|(g, w)| (g - w).abs() < 1e-6));
    }

    #[test]
    fn shape_mismatch_becomes_poison_not_panic() {
        let a = t(&[1.0; 4], vec![4]);
        let b = t(&[1.0; 3], vec![3]);
        let chain = a.add(&b).silu();
        assert!(chain.is_poisoned(), "描述期零失败(毒随链流)");
        let err = reduce(chain.step(), &mut CpuInterpreter::new()).unwrap_err();
        assert!(matches!(err, ModelError::Msg(ref m) if m.contains("毒值落地")));
    }

    #[test]
    fn poison_flows_through_downstream_ops() {
        let a = t(&[1.0; 4], vec![4]);
        let bad = t(&[1.0; 3], vec![3]);
        let mid = a.add(&bad);
        assert!(mid.is_poisoned());        
        let out = mid.silu().add(&mid);
        assert!(out.is_poisoned(), "毒必须流过下游,不能中途消失");
        let err = reduce(out.step(), &mut CpuInterpreter::new()).unwrap_err();
        assert!(format!("{err:?}").contains("形状不符"));
    }

    #[test]
    fn matmul_shapes() {
        let a = t(&[1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let b = t(&[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0], vec![4, 2]);
        let got = reduce(a.matmul(&b).step(), &mut CpuInterpreter::new()).unwrap();
        assert_eq!(got.f32, vec![4.0, 6.0]);
        assert_eq!(got.shape, vec![1, 2]);
    }

    #[test]
    fn flatten_covers_whole_tree() {
        let a = t(&[1.0, 2.0], vec![2]);
        let s = a.silu();
        let sum = s.add(&s);
        let flat = sum.flatten();
        // 物理共享(Arc parents,2026-09-26):重复消费 = 同节点两条入边,
        // 物理 DAG 中只存在一份,flatten 按 id 去重 → 3 个物理节点。
        // (旧值语义下 = 深拷贝 5 节点,已随 Arc 修订作废)
        assert_eq!(flat.len(), 3, "物理共享:重复消费去重");
        assert!(flat.windows(2).all(|w| w[0].depth() <= w[1].depth()));
        assert!(flat.iter().all(|x| !x.is_poisoned()));
        assert_eq!(flat.last().unwrap().id, sum.id, "根(最深)在末尾");
    }

    #[test]
    fn zeros_leaf_reduces() {
        let got = reduce(
            TensorOps::zeros(Dtype::F32, vec![2, 3]).step(),
            &mut CpuInterpreter::new(),
        )
        .unwrap();
        assert_eq!(got.f32, vec![0.0; 6]);
    }

    #[test]
    fn sigmoid_semantic_op_matches_host() {
        let x = vec![0.0, 1.0, -2.0, 10.0, -10.0, 0.25];
        let xs = t(&x, vec![2, 3]);
        let got = reduce(xs.sigmoid().step(), &mut CpuInterpreter::new())
            .unwrap()
            .f32;
        for (g, v) in got.iter().zip(&x) {
            let want = 1.0 / (1.0 + (-v).exp());
            assert!((g - want).abs() < 1e-6, "{g} vs {want}");
        }
    }

    #[test]
    fn reshape_is_metadata_passthrough() {
        let x: Vec<f32> = (0..6).map(|i| i as f32 * 0.5).collect();
        let xs = t(&x, vec![2, 3]);

        let out = xs.reshape(vec![3, 2]);
        assert!(!out.is_poisoned());
        assert_eq!(out.shape(), &[3, 2]);
        let got = reduce(out.step(), &mut CpuInterpreter::new()).unwrap();
        assert_eq!(got.f32, x, "reshape 透传父块,元素序不变");

        let bad = xs.reshape(vec![7]);
        assert!(bad.is_poisoned(), "元素数不守恒 → 毒");
        let err = reduce(bad.step(), &mut CpuInterpreter::new()).unwrap_err();
        assert!(format!("{err:?}").contains("reshape"), "{err:?}");
    }
}

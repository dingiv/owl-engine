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
//!   旧文档把这两者混为一谈,把"值出度"钉死为 1(树语义)——这正是
//!   2026-09-26 重复求值案的语义根因(见下)。
//!
//! **值形态与两个后果**:
//! **值形态与两个后果**(2026-10-12 修订:parents 已改 Arc 共享 ——
//! `Clone` = 浅拷贝 O(1),物理 DAG;"深拷贝树形展开"为历史描述,已废):
//! 1. `Clone` = 浅拷贝(parents 为 `Arc` 共享);**id 字段原样保留,不重新分配**。
//!    ⇒ 同一逻辑节点无论被克隆多少份、嵌在树的多少个位置,id 相同
//!    ⇒ "同 id = 同节点 = 同值",这是全组件最重要的不变量;
//! 2. 共享节点在声明树里**物理只有一份**(真 DAG;flatten 按 id 去重)。
//!
//! **⚠️ 重复求值陷阱(2026-09-26 实录,勿再踩)**:
//! 解释器若无 CSE,按树递归求值会把共享节点**执行多次**:
//! - 纯函数 kernel:只是浪费算力,数值不变;
//! - **带状态副作用的 kernel(conv/delta 写状态):第二次执行读到第一
//!   次滑过的状态 → 输出不同 → 语义破坏**。实测:DecoderLayer 残差
//!   h 同时喃 ln2 与残差加,mixer 执行两遍,mlp 段输出恒定 ×1.017
//!   偏差,bisect 两小时定位。
//!   因此解释器(_eval)必须带 **CSE memo**(键 = 节点 id,作用域 = 单次
//!   eval):同节点只执行一次,多消费者共享同一输出块。菱形、任意出度
//!   菱形都在保护范围内;跨 eval 不复用(每步状态必须推进)。
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
//! - Arc 物理 DAG:arg/join 均浅挂共享(配置面一次性;执行期由 memo 保证
//!   DAG 语义,见上)。
//! - 无 Arc、无 Tx、无共享别名——纯值世界;
//! - 每节点携带全局唯一自增 id(跨线程;server 对账/缓存键),声明链
//!   与运行时张量共用同一套进程级身份证。
//!
//! **这套设计的期望**:让代码量最大的 layer 声明和 models 声明,重复
//! 琐碎的样板尽可能减少,并把副作用移出层(层零执行)。代价:需要
//! 更复杂的结构(解释器/图求值)去执行上层的声明式语法,且描述能力
//! 受限(无控制流、无动态形状)。这不算缺点,但不方便——能力不足
//! 时,为它扩充新能力即可(历次扩项:Reshape/Kernel 节点/CSE/DAG)。

use crate::contract::Shape;
use crate::ops::{KernelArg, Op, SemanticKernel};
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
    /// **TP 角色标记**(S4 ③';**纯标注,零决策**)——
    /// 由层在**构造期**从规则表算出(`Linear::new` 缓存一次)并透传到本节点;
    /// **解释器**在求值尾部按它统一追加 allreduce(行并行 ⇒ 输出是部分和)。
    /// 层不构造任何通信节点,只"给标记"(用户裁决 2026-10-17)。
    pub(crate) tp_role: crate::plan::rules::WeightRule,
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

    /// **标量槽数**(宿主标量槽下标校验用;槽序权威 = `KernelSpec::PARAMS`)
    pub fn n_scalar_args(&self) -> usize {
        self.args.len()
    }

    /// **操作数只读视图**(归约所需的全部输入;叶子为空)。
    ///
    /// 为什么要有:分段/重排计划器需要"这个节点的输入是什么"来**推导段根**
    /// (`piecewise` 的每段根 = 下一个断点的操作数)—— 那是计划面的事,
    /// 不该让计划器去摸内部字段。克隆是浅的(Arc 共享父节点)。
    pub fn operands(&self) -> Vec<TensorOps> {
        self.parents.iter().map(|p| (**p).clone()).collect()
    }

    /// 语义坐标(观测面;None = 未打标)
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// 打标:语义坐标(观测面 tap 的定位键;interpreter-tap.md §三)。
    /// clone 本节点设标注 —— 同 id 不变(CSE memo/双锚对齐不受影响),
    /// O(1)(parents Arc 共享,浅拷贝)。层根自动打标见 Model::last_hidden。
    /// **TP 角色标记**(S4 ③';纯标注 —— 见 [`TensorOps::tp_role`] 字段注)。
    /// 层构造期算一次、forward 期透传;**决策在解释器**。
    pub fn with_tp_role(mut self, role: crate::plan::rules::WeightRule) -> TensorOps {
        self.tp_role = role;
        self
    }

    /// 本节点的 TP 角色(解释器读它)
    pub fn tp_role(&self) -> crate::plan::rules::WeightRule {
        self.tp_role
    }

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
            tp_role: crate::plan::rules::WeightRule::Replicate,
        }
    }

    /// **无清零分配声明**(scratch;内容未定义 —— 见 [`Op::Uninit`] 契约)
    pub fn uninit(dtype: Dtype, shape: Shape) -> TensorOps {
        TensorOps {
            id: next_id(),
            parents: vec![],
            depth: 0,
            op: crate::ops::Op::Uninit,
            dtype,
            shape,
            args: vec![],
            err: None,
            label: None,
            tp_role: crate::plan::rules::WeightRule::Replicate,
        }
    }

    /// **写入既有块的语义调用**(dest = `dst`;返回节点值 = 同块别名)。
    ///
    /// 之后按普通 `Call` 的写法链式追加**输入**与标量(不含 dest —— 它已是
    /// `parents[0]`);`with_shape` 应声明 dest 的形状(同块)。
    pub fn call_dst(dst: &TensorOps, op: crate::ops::SemanticKernel) -> TensorOps {
        TensorOps {
            id: next_id(),
            parents: vec![std::sync::Arc::new(dst.clone())],
            depth: 0,
            op: crate::ops::Op::CallDst { op, aux: Vec::new() },
            dtype: dst.dtype,
            shape: dst.shape.clone(),
            args: vec![],
            err: None,
            label: None,
            tp_role: crate::plan::rules::WeightRule::Replicate,
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
            tp_role: crate::plan::rules::WeightRule::Replicate,
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
            tp_role: crate::plan::rules::WeightRule::Replicate,
        }
    }

    /// 语义调用声明(**model 层面向解释器的唯一新语句**,Driver 立项):
    /// `TensorOps::call(ops::SemanticKernel::GdnGatingG).arg(..).with_shape(..)`
    /// 零核名/零发射参数 —— 名/变体/grid/block/smem 由解释器执行期经
    /// owl-kernels::driver::resolve(OpEnv 必传)拾取。
    pub fn call(op: crate::ops::SemanticKernel) -> TensorOps {
        TensorOps {
            id: next_id(),
            parents: vec![],
            depth: 0,
            op: crate::ops::Op::Call { op, aux: Vec::new() },
            dtype: Dtype::F32,
            shape: vec![],
            args: vec![],
            err: None,
            label: None,
            tp_role: crate::plan::rules::WeightRule::Replicate,
        }
    }

    /// 拾取推导常数(层语义几何:kd/vd/batch/rows…;**非核参数**,不进
    /// 签名 —— 序 = 各族 driver 文档;核签名标量仍走 arg_usize/arg_i32)
    pub fn aux(mut self, v: &[usize]) -> TensorOps {
        if let crate::ops::Op::Call { aux, .. } | crate::ops::Op::CallDst { aux, .. } = &mut self.op {
            aux.extend_from_slice(v);
        }
        self
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
            tp_role: crate::plan::rules::WeightRule::Replicate,
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
        // 刀3b 修正(2026-10-04):SliceView 父的重解释 = 声明期合成的新
        // SliceView(offset 保留,形状换新)—— 若走 Op::Reshape 透传节点,
        // eval 侧父匹配非 SliceView → Arg::Block **丢偏移**(消费核读到块首
        // 而非切片;C1 账长断言仅 debug 构建可见,release 静默错值)。
        if let Op::SliceView { .. } = &self.op {
            return self.slice_view(0, shape);
        }
        self.join(Op::Reshape, None, (self.dtype, shape), vec![])
    }

    /// 刀1:块内连续切片视图(零内核零节点派发;消费面发射
    /// Arg::BlockSlice)。offset = 平铺元素偏移(声明期含父链合成:
    /// 视图的视图相加,父仍链式保留);shape = 切片后声明形状。
    /// 仅连续切片入此臂(narrow_strided 快路守门),非连续仍物化。
    pub fn slice_view(&self, offset_elems: usize, shape: Shape) -> TensorOps {
        let base = match &self.op {
            Op::SliceView { offset_elems: prev } => *prev,
            _ => 0,
        };
        self.join(
            Op::SliceView { offset_elems: base + offset_elems },
            None,
            (self.dtype, shape),
            vec![],
        )
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
        // rank ≥ 1 律(2026-10-17 审查:原 `shape.len() - 1` 在空形状上
        // **usize 下溢 panic**,现在先取末维 ⇒ 毒化)
        let (Some(n), Ok(_)) = (b.shape.last().copied(), self.dim_last()) else {
            return self.poisoned_local(LazyError {
                at_depth: self.depth + 1,
                detail: format!("matmul: 空形状(self {:?} / b {:?},需 rank ≥ 1)", self.shape, b.shape),
            });
        };
        let mut shape = self.shape.clone();
        let last = shape.len() - 1;
        shape[last] = n;
        let meta = (self.dtype, shape);
        self.join(Op::Matmul, Some(b), meta, vec![])
    }

    /// nt 矩阵乘:B 以 [n, k] 行主序直读(tied lm_head:权重保持
    /// checkpoint [vocab, hidden] 原布局,零 host 转置零第二份显存)
    pub fn matmul_nt(&self, b: &TensorOps) -> TensorOps {
        if let Some(e) = self.shape_rule(b, "matmul_nt", |a, b| a.last() == b.last()) {
            return self.poisoned_local(e);
        }
        // rank ≥ 1 律(同上:空形状 ⇒ 毒化,不 panic;nt 的 n = B 行数)
        let (Some(n), Ok(_)) = (b.shape.first().copied(), self.dim_last()) else {
            return self.poisoned_local(LazyError {
                at_depth: self.depth + 1,
                detail: format!("matmul_nt: 空形状(self {:?} / b {:?},需 rank ≥ 1)", self.shape, b.shape),
            });
        };
        let mut shape = self.shape.clone();
        let last = shape.len() - 1;
        shape[last] = n;
        let meta = (self.dtype, shape);
        self.join(Op::MatmulNt, Some(b), meta, vec![])
    }

    pub fn add(&self, b: &TensorOps) -> TensorOps {
        if let Some(e) = self.shape_rule(b, "add", |a, b| a == b) {
            return self.poisoned_local(e);
        }
        let n: usize = self.shape.iter().product();
        let meta = (self.dtype, self.shape.clone());
        self.join(
            Op::Call { op: SemanticKernel::Add, aux: vec![] },
            Some(b),
            meta,
            vec![KernelArg::Bits(n as u64)],
        )
    }

    /// 同形逐元素乘(MLP 门控 / 注意力输出门)
    pub fn mul(&self, b: &TensorOps) -> TensorOps {
        if let Some(e) = self.shape_rule(b, "mul", |a, b| a == b) {
            return self.poisoned_local(e);
        }
        let n: usize = self.shape.iter().product();
        let meta = (self.dtype, self.shape.clone());
        self.join(
            Op::Call { op: SemanticKernel::Mul, aux: vec![] },
            Some(b),
            meta,
            vec![KernelArg::Bits(n as u64)],
        )
    }

    pub fn silu(&self) -> TensorOps {
        let n: usize = self.shape.iter().product();
        let meta = (self.dtype, self.shape.clone());
        self.join(
            Op::Call { op: SemanticKernel::Silu, aux: vec![] },
            None,
            meta,
            vec![KernelArg::Bits(n as u64)],
        )
    }

    /// 逐元素 sigmoid(注意力输出门;GDN beta 同族)
    pub fn sigmoid(&self) -> TensorOps {
        let n: usize = self.shape.iter().product();
        let meta = (self.dtype, self.shape.clone());
        self.join(
            Op::Call { op: SemanticKernel::Sigmoid, aux: vec![] },
            None,
            meta,
            vec![KernelArg::Bits(n as u64)],
        )
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
            cols > 0 && total.is_multiple_of(cols)
        }) {
            return self.poisoned_local(e);
        }
        let cols: usize = alpha.shape.iter().product();
        let meta = (self.dtype, self.shape.clone());
        self.join(
            Op::Call { op: SemanticKernel::Rmsnorm, aux: vec![] },
            Some(alpha),
            meta,
            // 标量槽序 = 线契约:cols(i32)/ eps(f32)/ w_off(i32)
            vec![
                KernelArg::I32(cols as i32),
                KernelArg::F32(eps),
                KernelArg::I32(w_off as i32),
            ],
        )
    }

    /// LayerNorm(affine;**视觉塔专用**):y = (x−mean)·rsqrt(var+eps)·w + b。
    /// 父序 = x [rows,n] / weight [n] / bias [n](与 PARAMS 同序);
    /// 标量槽序 = n(i32) / eps(f32)。
    /// 形状律:n 由 weight 定,x 行数 = total/n(任意前导维折叠),bias 同宽。
    pub fn layernorm(&self, weight: &TensorOps, bias: &TensorOps, eps: f32) -> TensorOps {
        let n: usize = weight.shape.iter().product();
        if let Some(e) = self.shape_rule(weight, "layernorm", |a, b| {
            let cols: usize = b.iter().product();
            let total: usize = a.iter().product();
            cols > 0 && total.is_multiple_of(cols)
        }) {
            return self.poisoned_local(e);
        }
        let bn: usize = bias.shape.iter().product();
        if bn != n {
            return self.poisoned_local(LazyError {
                at_depth: self.depth + 1,
                detail: format!("layernorm: bias 宽 {bn} ≠ weight 宽 {n}"),
            });
        }
        let meta = (self.dtype, self.shape.clone());
        self.join(
            Op::Call { op: SemanticKernel::Layernorm, aux: vec![] },
            Some(weight),
            meta,
            vec![KernelArg::I32(n as i32), KernelArg::F32(eps)],
        )
        .arg(bias)
    }

    /// GELU(tanh 近似;视觉塔 MLP)。逐元素一元;标量槽 = n(sz)。
    pub fn gelu_tanh(&self) -> TensorOps {
        let n: usize = self.shape.iter().product();
        let meta = (self.dtype, self.shape.clone());
        self.join(
            Op::Call { op: SemanticKernel::GeluTanh, aux: vec![] },
            None,
            meta,
            vec![KernelArg::Bits(n as u64)],
        )
    }

    /// GELU(精确 erf;视觉塔 merger)。与 [`Self::gelu_tanh`] **不可互替**。
    pub fn gelu_erf(&self) -> TensorOps {
        let n: usize = self.shape.iter().product();
        let meta = (self.dtype, self.shape.clone());
        self.join(
            Op::Call { op: SemanticKernel::GeluErf, aux: vec![] },
            None,
            meta,
            vec![KernelArg::Bits(n as u64)],
        )
    }

    /// 行广播加:x [rows, cols] + bias [cols](视觉塔每条 Linear 的 bias)。
    /// 形状律:bias 宽 = x 末维,且 x 行数 = total/cols。标量槽 = cols(sz)。
    pub fn bias_add(&self, bias: &TensorOps) -> TensorOps {
        let cols: usize = bias.shape.iter().product();
        if let Some(e) = self.shape_rule(bias, "bias_add", |a, b| {
            let bc: usize = b.iter().product();
            a.last().copied().unwrap_or(0) == bc && bc > 0
        }) {
            return self.poisoned_local(e);
        }
        let meta = (self.dtype, self.shape.clone());
        self.join(
            Op::Call { op: SemanticKernel::BiasAdd, aux: vec![] },
            Some(bias),
            meta,
            vec![KernelArg::Bits(cols as u64)],
        )
    }

    /// VLM 图像 token 注入(self = h [L, D];`merged` = 塔输出 [N, D];`idx` [L] f32)。
    ///
    /// `idx[t] ≥ 0` ⇒ 取 `merged[idx[t]]`;`idx[t] < 0` ⇒ 保留 `h[t]`
    /// (文本 token)。**响亮拒绝**:末维不齐 / 行号非有限或越出 N。
    /// 标量槽 = dim(sz);父序 = h / merged / idx(**构造序** —— idx 在末位,与
    /// `vis.pos_embed` 的 table/idx/wts 同款“索引量殿后”惯例)。
    pub fn image_scatter(&self, merged: &TensorOps, idx: &TensorOps) -> TensorOps {
        let dim = match self.dim_last() {
            Ok(d) => d,
            Err(e) => return self.poisoned_local(e),
        };
        if dim == 0 || self.shape.len() != 2 || merged.shape.len() != 2 {
            return self.poisoned_local(LazyError {
                at_depth: self.depth + 1,
                detail: format!("image_scatter: h {:?} / merged {:?} 需为 2 维且末维非零", self.shape, merged.shape),
            });
        }
        if merged.shape[1] != dim {
            return self.poisoned_local(LazyError {
                at_depth: self.depth + 1,
                detail: format!("image_scatter: merged 宽 {:?} ≠ h 宽 {dim}", merged.shape),
            });
        }
        let rows = match self.dim_first() {
            Ok(r) => r,
            Err(e) => return self.poisoned_local(e),
        };
        if idx.shape.iter().product::<usize>() != rows {
            return self.poisoned_local(LazyError {
                at_depth: self.depth + 1,
                detail: format!("image_scatter: idx 长 {:?} ≠ h 行数 {rows}", idx.shape),
            });
        }
        let meta = (self.dtype, self.shape.clone());
        self.join(
            Op::Call { op: SemanticKernel::ImageScatter, aux: vec![] },
            Some(merged),
            meta,
            vec![KernelArg::Bits(dim as u64)],
        )
        .arg(idx)
    }

    /// 位置嵌入 4-tap 重采样(self = table [n_pos, dim];idx/wts [n, n_taps] 恒 f32)。
    /// 形状律:idx 与 wts 同形;idx 末维 = taps;table 末维 = dim。
    /// 标量槽 = n_taps(sz), dim(sz);父序 = table / idx / wts。
    pub fn pos_embed_bilinear(&self, idx: &TensorOps, wts: &TensorOps) -> TensorOps {
        if idx.shape != wts.shape {
            return self.poisoned_local(LazyError {
                at_depth: self.depth + 1,
                detail: format!("pos_embed: idx 形状 {:?} ≠ wts 形状 {:?}", idx.shape, wts.shape),
            });
        }
        let dim = match self.dim_last() {
            Ok(d) => d,
            Err(e) => return self.poisoned_local(e),
        };
        let taps = match idx.dim_last() {
            Ok(t) => t,
            Err(e) => return self.poisoned_local(e),
        };
        if dim == 0 || taps == 0 {
            return self.poisoned_local(LazyError {
                at_depth: self.depth + 1,
                detail: format!("pos_embed: dim/taps 为 0(table {dim}, idx {taps})"),
            });
        }
        let rows = match idx.dim_first() {
            Ok(r) => r,
            Err(e) => return self.poisoned_local(e),
        };
        let meta = (self.dtype, vec![rows, dim]);
        self.join(
            Op::Call { op: SemanticKernel::PosEmbedBilinear, aux: vec![] },
            Some(idx),
            meta,
            vec![KernelArg::Bits(taps as u64), KernelArg::Bits(dim as u64)],
        )
        .arg(wts)
    }

    /// 非因果 varlen 全注意力(self = q [rows, heads, dim];k/v 同 hkv 头;cu f32 段界)。
    /// 形状律:q/k/v 末维一致、头数一致(k 与 v 同形);cu 长度 = n_seq+1。
    /// 标量槽 = rows(sz), heads(sz), head_dim(sz), n_seq(i32), scale(f32)。
    pub fn attn_full(
        &self,
        k: &TensorOps,
        v: &TensorOps,
        cu: &TensorOps,
        n_seq: usize,
        scale: f32,
    ) -> TensorOps {
        let q = self.shape.clone();
        if q.len() != 3 || k.shape.len() != 3 || v.shape != k.shape {
            return self.poisoned_local(LazyError {
                at_depth: self.depth + 1,
                detail: format!("attn_full: q {:?} / k {:?} / v {:?} 形状不符(需 3 维且 k==v)", q, k.shape, v.shape),
            });
        }
        if q[1] != k.shape[1] || q[2] != k.shape[2] {
            return self.poisoned_local(LazyError {
                at_depth: self.depth + 1,
                detail: format!("attn_full: q 头数/维度 {:?} ≠ k {:?}(本核无 GQA)", q, k.shape),
            });
        }
        if cu.shape.iter().product::<usize>() != n_seq + 1 {
            return self.poisoned_local(LazyError {
                at_depth: self.depth + 1,
                detail: format!("attn_full: cu_seqlens 长 {:?} ≠ n_seq+1 = {}", cu.shape, n_seq + 1),
            });
        }
        let meta = (self.dtype, q.clone());
        self.join(
            Op::Call { op: SemanticKernel::AttnFull, aux: vec![] },
            Some(k),
            meta,
            vec![
                KernelArg::Bits(q[0] as u64),
                KernelArg::Bits(q[1] as u64),
                KernelArg::Bits(q[2] as u64),
                KernelArg::I32(n_seq as i32),
                KernelArg::F32(scale),
            ],
        )
        .arg(v)
        .arg(cu)
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

    /// **标量槽改写**(拷贝语义:同父/同 op/同形状,**新 id**)。
    ///
    /// 用途(piecewise 宿主标量槽;见 `docs/piecewise-断点图-设计与实现.md`):
    /// 断点核里那些"每轮会变的宿主标量"(FI 的 `ctx_total` = plan key)
    /// 不能在捕获期烘死 ⇒ 回放期由驱动件按**契约下标**把本轮值补进来,
    /// 再送出窗外求值。id 换新 = memo/覆盖按新节点记账,旧条目不参与。
    ///
    /// `idx` 越界 = 上游违约(调用点先校验):返回 `None`,不静默自吞。
    pub fn with_arg_bits(self, idx: usize, bits: u64) -> Option<TensorOps> {
        if idx >= self.args.len() {
            return None;
        }
        let mut out = self;
        out.args[idx] = KernelArg::Bits(bits);
        out.id = next_id();
        Some(out)
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

    /// 声明形状**末维**(律:**rank ≥ 1**)
    ///
    /// 空形状 = 声明失守 ⇒ `Err`(调用点 `poisoned_local` 收口)。2026-10-17
    /// 审查 §六轮:原先七处写成 `shape.last().copied().unwrap_or(0)` —— 把
    /// "形状失守"与"维度是 0"混成同一个值(读起来像静默回退,且 0 会一路
    /// 流进核标量/几何推导)。
    fn dim_last(&self) -> Result<usize, LazyError> {
        self.shape.last().copied().ok_or_else(|| LazyError {
            at_depth: self.depth + 1,
            detail: format!("形状为空(需 rank ≥ 1;op {:?})", self.op),
        })
    }

    /// 声明形状**首维**(律同 [`Self::dim_last`])
    fn dim_first(&self) -> Result<usize, LazyError> {
        self.shape.first().copied().ok_or_else(|| LazyError {
            at_depth: self.depth + 1,
            detail: format!("形状为空(需 rank ≥ 1;op {:?})", self.op),
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
            tp_role: crate::plan::rules::WeightRule::Replicate,
        }
    }

    /// append:新节点浅挂输入(parents Arc 共享;同 id 不变量的根基)
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
            tp_role: crate::plan::rules::WeightRule::Replicate,
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

    /// 宿主标量槽的载体:`with_arg_bits` = 拷贝 + 换一个标量槽 + **新 id**
    /// (piecewise 回放期每轮补值;旧 id 由调用方摘账)
    #[test]
    fn with_arg_bits_patches_one_slot_and_refreshes_id() {
        let x = TensorOps::zeros(Dtype::F32, vec![2]);
        let n = TensorOps::call(crate::ops::SemanticKernel::Concat)
            .arg(&x)
            .arg_usize(7)
            .arg_usize(9)
            .with_shape(Dtype::F32, vec![2]);
        let old_id = n.id();
        assert_eq!(n.n_scalar_args(), 2);

        let p = n.clone().with_arg_bits(1, 42).expect("槽 1 存在");
        assert_ne!(p.id(), old_id, "补丁 = 新节点 id(memo/覆盖按新节点记账)");
        assert_eq!(p.n_scalar_args(), 2);
        assert_eq!(p.shape(), n.shape());
        // 除被补的那一槽,节点其余部分逐位沿用(操作数 id / 形状 / 槽数)
        assert_eq!(
            p.operands().iter().map(|o| o.id()).collect::<Vec<_>>(),
            n.operands().iter().map(|o| o.id()).collect::<Vec<_>>(),
            "补丁只动标量槽:操作数(依赖边)逐位沿用"
        );

        // 越界 = 上游违约:结构化 None(调用点 plan 期拒捕),不静默自吞
        assert!(n.clone().with_arg_bits(2, 1).is_none());
        assert!(x.clone().with_arg_bits(0, 1).is_none(), "叶子无标量槽");
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

    /// **rank ≥ 1 律门**(2026-10-17 审查 §六轮):空形状声明失守 ⇒ 相关 op
    /// **毒化**(修前:`shape.len() - 1` **usize 下溢 panic**;另 7 处
    /// `shape.last().copied().unwrap_or(0)` 读起来像静默回退)。
    #[test]
    fn empty_shape_poisons_instead_of_zero_or_panic() {
        let a = t(&[1., 2., 3., 4.], vec![2, 2]);
        let empty = TensorOps::zeros(Dtype::F32, vec![]);
        // matmul / matmul_nt:空 b 或空 self ⇒ 毒(不 panic、不静默 0)
        assert!(a.matmul(&empty).is_poisoned(), "matmul(空 b)");
        assert!(empty.matmul(&a).is_poisoned(), "matmul(空 self)");
        assert!(a.matmul_nt(&empty).is_poisoned(), "matmul_nt(空 b)");
        assert!(empty.matmul_nt(&a).is_poisoned(), "matmul_nt(空 self)");
        // image_scatter / pos_embed:空 self/idx ⇒ 毒
        assert!(empty.image_scatter(&a, &a).is_poisoned(), "image_scatter(空 h)");
        assert!(empty.pos_embed_bilinear(&a, &a).is_poisoned(), "pos_embed(空 table)");
        assert!(a.pos_embed_bilinear(&empty, &empty).is_poisoned(), "pos_embed(空 idx)");
        // 对照组:正常形状不毒
        assert!(!a.matmul(&a).is_poisoned(), "正常 matmul 不该毒");
        assert!(!a.image_scatter(&a, &t(&[1., 2.], vec![2])).is_poisoned(), "正常 scatter 不该毒");
    }
}

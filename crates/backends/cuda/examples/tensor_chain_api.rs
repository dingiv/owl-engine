//! ⚠️ 设计伪代码(2026-09-23):Tensor 链式声明语法 + 执行解释器。
//!
//! 代数基础:funio Pack 三律(用户自有先验,见 repos/funio):
//!   1. 不可变:append 产出新值,旧值永不变(fork = 结构共享,免费);
//!   2. 毒值:Either<Val, Err> 随声明流动,边界收割;
//!   3. 双色透明:同一份声明,sync 解释器(测试)与 async 解释器(生产)
//!      都能吃——描述不关心执行色。
//!
//! 数据结构定案:**cons-DAG 节目单**。
//!   - append = 包一个新 Spine 节点指回 parent → O(1),不可变;
//!   - 多输入运算(两支链合并)= 节点多父 → 天然 DAG;
//!   - fork(多档位捕获)= 同一 parent 多个孩子 → 结构共享零拷贝。
//! eval = 沿 parent 指针收集节点,按深度拓扑序送解释器。
//!
//! 本文件不编译;是 A2(server 落肉)之前必须定稿的接口契约。

#![allow(dead_code, unused_variables)]

// ============================================================================
// §1 计划图(cons-DAG):描述的唯一容器
// ============================================================================

/// 计划图 = 不可变 DAG。所有 Tensor 都是它上面某个节点的视图。
/// Clone 便宜(Arc);fork = 多个孩子共享同一 parent(零拷贝)。
#[derive(Clone)]
pub struct Plan {
    head: Option<Arc<Step>>,
}

struct Step {
    parent: Option<Arc<Step>>,
    depth: u32,          // 拓扑深度(构造时 = parent.depth + 1;eval 排序用)
    op: Op,
    out: TensorMeta,     // 本节点产出的 dtype/shape(client 侧断言与调试)
    err: Option<LazyError>, // 毒值(构造期违约:shape/dtype 不符)
}

/// 节点运算(语义枚举;server 派发到 kernel;协议面转 u64 参数)
enum Op {
    Malloc { bytes: u64 },
    Htod,                          // host 数据入块(pinned 码头,server 持有)
    Matmul { transposed: bool },
    Rmsnorm { eps: f32, w_off: bool },
    Silu,
    Add,
    Rope { theta_base: f64 },
    Embedding,
    /// 唯一的显式副作用节点:KV 写槽(SSA 外形,物理原地由 server 解释;§迁移账 2)
    SlotWrite { /* slots, kv_lens */ },
    ReplayInput,                   // 捕获档:bindings 预绑定缓冲(窗内只读)
}

struct TensorMeta {
    dtype: Dtype,
    shape: Vec<usize>,
}

struct LazyError {
    at_depth: u32,
    detail: String,
}

enum Dtype {
    F32,
    BF16,
}

// ============================================================================
// §2 Tensor:计划图上某节点的视图(dtype/shape 标注随行)
// ============================================================================

#[derive(Clone)] // 不可变:Clone = 同一节点的多消费者(免费)
pub struct Tensor {
    plan: Plan, // 该节点所在的 spine 头(含本节点)
    node: Arc<Step>,
    meta: TensorMeta,
}

impl Tensor {
    // ---- 工厂(声明;提交/执行在边界)----

    /// 声明分配(bytes = shape.prod() × dtype 宽,由标注推得,server 不问)
    pub fn zeros(c: &Client, dtype: Dtype, shape: &[usize]) -> Tensor {
        let _ = c;
        unimplemented!("Plan::root(Op::Malloc) + meta 标注")
    }

    /// host 数据声明(数据进 server 的 pinned 码头;契约 3 归 server)
    pub fn from_host(c: &Client, shape: &[usize], data: &[f32]) -> Tensor {
        unimplemented!()
    }

    // ---- 视图:零节点追加(纯元数据;server 无感)----

    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Tensor {
        unimplemented!("同 block,meta 收窄;Poisoned 透传(毒不消失)")
    }

    // ======================================================================
    // §2.1 链式声明语法:一元 = 方法;二元 = 方法吃 &Tensor
    // ======================================================================

    /// y = rmsnorm(x, alpha, eps, w_off) —— 链式一元
    /// 毒传播: Poisoned 自返(不追加节点;毒随流,边界收割)
    pub fn rmsnorm(&self, alpha: &Tensor, eps: f32, w_off: bool) -> Tensor {
        // PSEUDO: if let Poisoned = self → 自返毒(带上游 LazyError)
        // PSEUDO: 断言(alpha 一维 & len == last_dim;不符 → Poisoned 携带本案)
        // PSEUDO: Tensor { plan: Plan::extend(&self.plan, Op::Rmsnorm{..}), ... }
        unimplemented!()
    }

    pub fn silu(&self) -> Tensor {
        unimplemented!()
    }

    /// out = a × b —— 链式二元(两支链在 DAG 里合并;O(1) append)
    pub fn matmul(&self, b: &Tensor) -> Tensor {
        // PSEUDO: 断言 self.shape[-1] == b.shape[0](不符 → Poisoned)
        // PSEUDO: Plan::join(&self.plan, &b.plan, Op::Matmul)  ← DAG 多父合并
        unimplemented!()
    }

    pub fn add(&self, b: &Tensor) -> Tensor {
        unimplemented!()
    }
}

// ============================================================================
// §3 执行边界:解释器(对同一张 DAG 的两种游走)
// ============================================================================

impl Client {
    /// eager 解释器(生产):收集拓扑序 → 逐节点提交 server(提交即回)。
    /// 共享节点(NodeId 同源)去重:只提交一次,Block 缓存供多消费者取用。
    /// 返回句柄面:可对任意中间节点 to_host(档二事件等它)。
    // PSEUDO: pub async fn eval(&self, t: &Tensor) -> Result<EvalHandle, BackendError> {
    // PSEUDO:     let order = t.plan.topo();                 // 深度排序(迭代,不递归)
    // PSEUDO:     let mut live: HashMap<NodeId, BlockRef> = ...;
    // PSEUDO:     for step in order {
    // PSEUDO:         if let Some(e) = &step.err { return Err(毒值落地: LazyError → 结构化); }
    // PSEUDO:         match step.op {
    // PSEUDO:             Op::Malloc => live[id] = self.proto_malloc(step).await?,
    // PSEUDO:             Op::Matmul => self.proto_launch(kernel_of(step), live_of(step)).await?,
    // PSEUDO:             ...
    // PSEUDO:         }
    // PSEUDO:     }
    // PSEUDO: }

    /// 收割(D2H;完成 ready——档二事件等前置节点,链路 server 内部消化)
    // PSEUDO: pub async fn to_host(&self, t: &Tensor) -> Result<Vec<f32>, BackendError>;

    /// sync 解释器(测试专用;同一条 DAG,阻塞游走):
    /// 同一 walk 代码,submit 换成阻塞版——**描述与执行色无关**的验收形态。
    // PSEUDO: pub fn eval_blocking(&self, t: &Tensor) -> Result<Vec<f32>, BackendError>;
}

// ============================================================================
// §4 捕获 = 节目单的另一种解释器(free win 的正式形态)
// ============================================================================

impl Client {
    /// 捕获解释器:同一张 DAG,游走时"录"而非"发";整单烘焙成图。
    /// 哨兵③的对账对象 = 这张 DAG 本身(契约 2 类型化:窗内不存在旁路)。
    ///
    /// 多档位 fork(funio 结构共享的直接兑现):
    ///   let base  = Tensor::bindings(...)            // 底稿:预绑定+装载
    ///   let body4 = base.chain_forward(4);           // fork:4 档的尾巴
    ///   let body8 = base.chain_forward(8);           // fork:8 档的尾巴(共享 base!)
    ///   let g4 = client.bake(&body4)?;               // 各自烘焙成图
    ///   let g8 = client.bake(&g8_base=body8)?;       // base 部分的节点号共享
    // PSEUDO: pub async fn bake(&self, t: &Tensor) -> Result<GraphRef, BackendError>;
}

// ============================================================================
// §5 端到端样例(验收脚本草案;同 tensor_client_vision §4 的声明式版)
// ============================================================================

async fn decode_step(c: &Client, base: &Tensor, tokens: &[u32]) -> Result<(), BackendError> {
    // 描述期:同步,纯,total(毒值在流动,但没有一个 ? / .await)
    let frontier = Tensor::from_host(c, &[4], tokens_by(tokens));
    let x = frontier.add(base);
    let logits = x.matmul(&x).rmsnorm(&x, 1e-5, true); // 形状示意

    // 执行边界:全链唯一的 await + ?(哨兵①登记/保活/提交都在 server 侧)
    let out = c.eval(&logits).await?;

    // 采样(描述层继续;或 to_host 收割走 radix)
    let _ = out;
    Ok(())
}

fn tokens_by(t: &[u32]) -> Vec<u8> {
    t.iter().flat_map(|v| v.to_le_bytes()).collect()
}

// ============================================================================
// §6 毒值与错误边界(设计 §四契约五的落地细则)
// ============================================================================

// - 构造期违约(shape/dtype/断言)= Poisoned(LazyError{depth, detail}),
//   随链流动;任何 op 对 Poisoned = 恒等透传(不追加节点,不重复报);
// - eval 到 Poisoned 节点 = 结构化报错落地(带 depth → 案发节点回溯);
// - **报错点后移的补偿**:LazyError 携带 depth + op 描述;DAG 可从 head
//   反演整链(不可变的第二个红利:错误现场永远可重放);
// - 资源错(池耗尽/令牌死亡)不经毒值——它们发生在解释器,直接 Err 过线
//   (毒值只承载"描述层的逻辑违约",两族错误不混)。

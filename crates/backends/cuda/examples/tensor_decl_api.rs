//! ⚠️ 设计伪代码(2026-09-23):Tensor 声明式 API——描述运算,不执行。
//!
//! 回答的问题:上层(nn/Tensor)如何描述运算而**不实际执行**?
//! 答案的形状:Tensor 是**不可变的声明值**(像 DataFrame 惰性计划、
//! cuTile DeviceOp、React 元素——同一族),运算产生新声明,执行只在
//! 显式边界发生。配合 design §四契约五:
//!   描述层 = 纯 · total · 同步 · 无 Result 无 async。
//!
//! 与现状的关键差异(迁移成本所在):
//!   现状:ops.matmul(ctx, blas, &w, &x, &mut out)?   ← 就地执行,&mut 出参
//!   本设计:let out = matmul(&w, &x);                 ← SSA,产出新声明
//! 迁移 = 把"出参风格"翻成"返回值风格"(机械,但面广——owl-nn 全部算子)。

#![allow(dead_code, unused_variables)]

// ============================================================================
// §1 声明值:Tensor 的两种存在形态
// ============================================================================

/// 张量:一个声明。它**只是值**——持有"我从哪来"的证明,不碰 GPU。
///
/// - `Materialized`:块已物化(装载/上一轮 eval 的产物;decode 的
///   bindings/weights 是这一态的常驻民);
/// - `Deferred`:尚未执行的运算节点(节目单中的中间值);
/// - `Poisoned`:毒值(构造期违约/上游执行失败),随运算流动,终点检验。
pub struct Tensor {
    src: Src,
    dtype: Dtype,
    shape: Vec<usize>,
}

enum Src {
    Materialized(BlockRef),
    Deferred(NodeId),
    Poisoned(LazyError),
}

/// 节点号 = 节目单下标(SSA:每个运算产生一个新声明,无 &mut,无别名)。
struct NodeId(usize);

/// 惰性错误:案发时抓上下文(Op 序号 + 描述),毒值携带到边界。
struct LazyError {
    at_node: usize,
    detail: String,
}

// ============================================================================
// §2 节目单:描述的容器(纯数据;一份描述,三个消费者)
// ============================================================================

/// 运算节目单:声明式 API 的全部产出就是往这里追加节点。
///
/// 三个消费者(同一份数据):
///   1. eager 解释器:server 逐节点执行(提交即回);
///   2. 捕获烘焙:整单烘成 CUDA 图(哨兵③审计的原材料就是它——契约 2
///      从"纪律"变"类型事实":窗内不存在旁路发射的可能性);
///   3. 错误对账:毒值/失败节点按 NodeId 定位。
///
/// 线程纪律:节目单随 Tx 走,Tx 随线程走(单写者);跨线程 = 整单 Send。
pub struct Program {
    ops: Vec<OpNode>,
    arena: Vec<TensorMeta>, // 每节点的 dtype/shape(供 client 侧断言与调试)
    poisoned: Vec<LazyError>,
}

struct OpNode {
    kind: OpKind,       // Matmul/Rmsnorm/Embedding/... (语义枚举;协议面仍转 u64)
    inputs: Vec<NodeOrBlock>,
    output: usize,      // 本节点的产出节点号(SSA)
}

enum OpKind {
    // —— 语义枚举:server 派发到真实 kernel;协议面上仍转成裸 u64 参数 ——
    Matmul { transposed: bool },
    Rmsnorm { eps: f32, w_off: bool },
    Silu,
    Add,
    Rope { theta_base: f64 },
    Embedding,
    PagedAttn { /* 槽位参数 */ },
    Malloc { bytes: u64 },   // 分配也是声明(工厂产出 = Malloc 节点 + 标注)
    Htod,                    // host 数据入块(装载声明)
    // 采样/argmax ...
}

// ============================================================================
// §3 声明式 API:运算 = 纯函数(Total:毒值传播,永不失败)
// ============================================================================

// ---- 工厂(声明"需要一个块";server 在 eval 时兑现)----

impl Tensor {
    /// 声明分配。注意:**不 await**——节目单追加 Malloc 节点,返回声明。
    /// 提交与完成在边界(design §五档一"提交即回")。
    pub fn zeros(c: &TensorClient, dtype: Dtype, shape: &[usize]) -> Tensor {
        // PSEUDO: c.program.emit(OpKind::Malloc { bytes }); ...
        unimplemented!()
    }

    /// host 数据声明(契约 3 的落点:server 持有 pinned 码头直填/复用)。
    /// 返回的是声明;数据何时上卡 = eval 的事,与我无关。
    pub fn from_host(c: &TensorClient, shape: &[usize], data: &[f32]) -> Tensor {
        unimplemented!()
    }

    /// 视图:零 server 往返(纯元数据;狭义 reshape/narrow 是声明上的声明)。
    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Tensor {
        unimplemented!()
    }
}

// ---- 运算:SSA 返回值风格(替换现状的 &mut out 风格)----

/// out[M,N] = a[M,K] × b[K,N]。纯函数:读声明,产声明,永不失败。
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    // 断言(shape/dtype 相容)在**这里**做:
    //   - 相容 → 追加节点,返回 Deferred 声明;
    //   - 不符 → 返回 Poisoned(逻辑 bug 的性质;毒到边界集中爆,
    //     携带本节点号与两侧 shape——LazyError 的补偿机制)。
    // 两个断言都是纯计算,无 GPU 触碰。
    unimplemented!()
}

pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    unimplemented!()
}
pub fn silu(x: &Tensor) -> Tensor {
    unimplemented!()
}
pub fn rmsnorm(x: &Tensor, alpha: &Tensor, eps: f32, w_off: bool) -> Tensor {
    unimplemented!()
}

// ============================================================================
// §4 执行边界:整个调用链唯一触碰 async/Result 的地方
// ============================================================================

impl TensorClient {
    /// 评述:把节目单交给 server 解释(档一:提交即回;节点完成走档二)。
    /// 返回完成句柄——对 program 里任何 Deferred 声明,之后可取真实结果。
    // PSEUDO: pub async fn eval(&self, program: Program) -> EvalHandle;

    /// 收割:同步等待 + D2H(全链唯一拿数据的地方)。
    // PSEUDO: pub async fn to_host(&self, t: &Tensor) -> Result<Vec<f32>, BackendError>;
}

// ============================================================================
// §5 端到端样例:一次完整的声明 → 执行
// ============================================================================

async fn demo(c: &TensorClient) {
    // ---- 描述期:同步,纯,total,零 await 零 ? ----
    let w = Tensor::from_host(c, &[64, 64], &weights());
    let x = Tensor::zeros(c, Dtype::F32, &[64]);

    // 一条链:每个调用返回新声明;哪怕 w 的装载将来在 GPU 上失败,
    // 这段代码也**不会失败**——毒值流动,边界收割。
    let mid = matmul(&w, &x);
    let out = rmsnorm(&mid, &mid, 1e-5, true);

    // ---- 执行边界:全程序仅有的 async/Result ----
    let data: Vec<f32> = c.to_host(&out).await.unwrap();

    assert_eq!(data.len(), 64);
}

fn weights() -> Vec<f32> {
    vec![]
}

// ============================================================================
// §6 与捕获的关系:声明即图(免费 win)
// ============================================================================

// 捕获事务在这个模型下的形态:
//
//   let g = client.capture(plan, |p: &mut Program| {
//       // 描述层原样复用:同样的 block_forward(p.tx(), ...) 调用链
//       // p 就是节目单;窗内 = 节目单全长
//   }).await?;   // server:整单烘焙 → instantiate → 哨兵③对账(对账对象 =
//                //            节目单本身,契约 2 由类型保证)
//
// 对比今天:捕获窗是"执行时录 VCR";声明式后,捕获窗是"把排练稿打印装订"。
// P1-2/悬空/污染三案在声明式下结构性不存在:窗内没有执行,只有数据。

// ============================================================================
// §7 迁移账(诚实条款)
// ============================================================================
//
// 1. ops 全部 &mut out → SSA 返回值:机械但面广(owl-nn 全算子 + engine
//    消费点);收益 = 声明纯度的前提(别名消除);
// 2. 就地语义的算子(如 KV 写槽 paged attention):SSA 表达不了"原地"——
//    建模为带显式副作用声明的节点(SlotWrite { slots, ... }),节点仍是
//    SSA,物理原地由 server 解释;这是唯一需要新语义的地方;
// 3. 节目单先行:显存峰值按节目单全长计;毒值报错点后移(靠 LazyError
//    携带节点号补偿);
// 4. 现有 eager 热路径:声明式的每节点 append = 一次 Vec push + 枚举构造,
//    纳秒级;decode 图回放不受影响(图在捕获期烘焙,回放不经节目单)。

//! 观测面(tap):节点级单步调试的事件词汇与协议。
//!
//! 定位(设计文档:`docs/arch/interpreter-tap.md`):owl 三层把全部数据流
//! 收口在解释器节点归约点 —— 每个中间值都是 memo 中的真实池块,因此
//! **声明一次,任一中间量可寻址、可回读、可统计,零重放零污染**。
//! 本模块只定义词汇与 tap 协议;挂点在 [`super::eval`](计算域)与
//! [`crate::reference::reduce_tap`](同步参考域),两域事件同源。
//!
//! **核心裁决:tap 只声明想要什么,执行归解释器**(与"层只声明、执行归
//! 解释器"同构 —— 观测也归解释器,tap 连执行的权力都没有):
//! - face 是 async 且独占借用,tap 若自己摸数据要么 async trait(dyn 不可用)
//!   要么借用冲突;裁决 = 解释器代执行 dtoh/统计,tap 是纯同步声明者;
//! - 观测在结构上不可能改变计算(对照 harvest 重放污染:病根是观测自带
//!   执行 —— 带状态层每观测一次就多执行一遍)。
//!
//! **观测窗口律(候选 §四 律 25)**:数据读取由解释器在 After 窗口内
//! 代执行完毕;块存活期 = memo 存活期,窗口外读块未定义。tap 无
//! launch/alloc/htod 权。

use crate::ops::Op;

// ============================================================================
// §1 事件词汇
// ============================================================================

/// 块引用(输出块的只读投影;BlockRef 不赋予读权 —— 读走 Want 协议)
#[derive(Clone, Copy, Debug)]
pub struct BlockRef {
    /// server 池块 id(CPU 参考侧无池块,哨兵 [`BlockRef::NO_POOL`])
    pub block_id: u64,
    /// 元素账长(Block 叶子 len=0,C1 惯例)
    pub len: usize,
}

impl BlockRef {
    /// CPU 参考侧哨兵(无池块世界)
    pub const NO_POOL: u64 = u64::MAX;
}

/// 节点归约完成事件(元数据借用声明节点;数据读取走 Want 协议)
pub struct NodeEvent<'a> {
    /// 节点身份证(同 id = 同节点 = 同值,C2;双锚对齐键 —— 同一声明树
    /// 双锚求值,id 在声明期已定,天然对齐)
    pub id: u64,
    /// 语义算子
    pub op: &'a Op,
    pub dtype: crate::contract::Dtype,
    /// 声明形状(权威维度,C1;与块账长无关)
    pub shape: &'a [usize],
    /// 拓扑深度
    pub depth: u32,
    /// 语义坐标(§2 tag;层根标签 = Model 声明侧注入)
    pub tag: Option<&'a str>,
    /// 输出块(已入 memo)
    pub out: BlockRef,
}

/// 节点数值统计(观测指纹的最小集;rms = √mean(v²))
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlockStats {
    pub rms: f32,
    pub min: f32,
    pub max: f32,
    /// 塌零检测的直接判据(精确 0.0 计数)
    pub zeros: usize,
    /// 头部样本(对拍粗定位)
    pub first8: [f32; 8],
}

impl BlockStats {
    pub fn of(host: &[f32]) -> Self {
        if host.is_empty() {
            return Self { rms: 0.0, min: 0.0, max: 0.0, zeros: 0, first8: [0.0; 8] };
        }
        let mut sum2 = 0f32;
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut zeros = 0usize;
        for &v in host {
            sum2 += v * v;
            min = min.min(v);
            max = max.max(v);
            if v == 0.0 {
                zeros += 1;
            }
        }
        let n = host.len() as f32;
        let mut first8 = [0f32; 8];
        let k = host.len().min(8);
        first8[..k].copy_from_slice(&host[..k]);
        Self { rms: (sum2 / n).sqrt(), min, max, zeros, first8 }
    }
}

// ============================================================================
// §2 tap 协议
// ============================================================================

/// 观测意愿(on_node 的返回;执行归解释器)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Want {
    /// 不观测(默认;零 dtoh)
    Quiet,
    /// 解释器 dtoh → [`BlockStats`] → [`Tap::on_stats`]
    Stats,
    /// 解释器 dtoh 全量(host f32)→ [`Tap::on_bytes`]
    Bytes,
    /// 钉住本块(eval 结束后仍可读)。**二期未实施**:MVP 按 Quiet 处理
    /// (scratch 块窗口外存活无保证;见设计文档 §六)
    Keep,
}

/// 观测者协议(全默认方法:实现一个方法即完整 tap;`&mut dyn` 可用)。
///
/// 三事件:`on_node`(每节点 After;返回 Want)→ `on_stats`/`on_bytes`
/// (解释器代执行后回传);`on_poison`(毒值落地,归约照旧 Err)。
pub trait Tap: Send {
    /// 每节点归约完成(输出已入 memo)调用
    fn on_node(&mut self, _ev: &NodeEvent<'_>) -> Want {
        Want::Quiet
    }
    /// 统计回传(on_node 返 [`Want::Stats`] 时)
    fn on_stats(&mut self, _id: u64, _s: BlockStats) {}
    /// 全量回传(on_node 返 [`Want::Bytes`] 时;host = 声明形状全量 f32)
    fn on_bytes(&mut self, _ev: &NodeEvent<'_>, _host: &[f32]) {}
    /// 毒值落地(Err 路径;观测流里也能看到"死在哪")
    fn on_poison(&mut self, _id: u64, _detail: &str) {}
}

/// 空观测(eval_ops 默认;与 tap: None 等价,省一层 Option 分支)
pub struct QuietTap;

impl Tap for QuietTap {}

/// 双 tap 链(诊断场景:曲线打印 + 引用记录同行);Want 取两臂最响。
/// 更多臂的需求出现再泛化(现勿过度设计)。
pub struct TapChain<'a>(pub &'a mut dyn Tap, pub &'a mut dyn Tap);

impl Tap for TapChain<'_> {
    fn on_node(&mut self, ev: &NodeEvent<'_>) -> Want {
        let a = self.0.on_node(ev);
        let b = self.1.on_node(ev);
        if a == Want::Bytes || b == Want::Bytes {
            Want::Bytes
        } else if a == Want::Stats || b == Want::Stats {
            Want::Stats
        } else if a == Want::Keep || b == Want::Keep {
            Want::Keep
        } else {
            Want::Quiet
        }
    }

    fn on_stats(&mut self, id: u64, s: BlockStats) {
        self.0.on_stats(id, s);
        self.1.on_stats(id, s);
    }

    fn on_bytes(&mut self, ev: &NodeEvent<'_>, host: &[f32]) {
        self.0.on_bytes(ev, host);
        self.1.on_bytes(ev, host);
    }

    fn on_poison(&mut self, id: u64, detail: &str) {
        self.0.on_poison(id, detail);
        self.1.on_poison(id, detail);
    }
}

// ============================================================================
// §3 内建 tap:StatsTap(统计收集 + 曲线打印)
// ============================================================================

/// 单节点统计记录
#[derive(Clone, Debug)]
pub struct StatRecord {
    pub id: u64,
    pub tag: Option<String>,
    pub stats: BlockStats,
}

/// 统计收集 tap(内建;塌零排查的层根曲线即本 tap 的 curve 姿势)。
///
/// - `tagged_only = true`:只观测带 tag 节点(层根曲线;weights 等
///   Block 叶子无 tag 天然跳过,零大块 dtoh);
/// - `log = true`:每条记录 eprintln(现场肉眼诊断);
/// - `records`:全部落账(golden 指纹锁的原料,双锚拉链比对同款)。
#[derive(Default)]
pub struct StatsTap {
    tagged_only: bool,
    log: bool,
    step: usize,
    pending: Option<String>,
    pub records: Vec<StatRecord>,
}

impl StatsTap {
    /// 全节点统计(不打印;weights Block 叶子也会统计 —— 大块 dtoh,慎用)
    pub fn all() -> Self {
        Self { tagged_only: false, log: false, step: 0, ..Default::default() }
    }

    /// 层根曲线姿势(带 tag 节点 + 打印;step = 步号标注)
    pub fn curve(step: usize) -> Self {
        Self { tagged_only: true, log: true, step, ..Default::default() }
    }

    /// 静默收集姿势(带 tag 节点,不打印;golden 锁用)
    pub fn golden() -> Self {
        Self { tagged_only: true, log: false, step: 0, ..Default::default() }
    }
}

impl Tap for StatsTap {
    fn on_node(&mut self, ev: &NodeEvent<'_>) -> Want {
        if self.tagged_only && ev.tag.is_none() {
            return Want::Quiet;
        }
        self.pending = ev.tag.map(str::to_string);
        Want::Stats
    }

    fn on_stats(&mut self, id: u64, s: BlockStats) {
        let tag = self.pending.take();
        if self.log {
            eprintln!(
                "[tap s{}]{:<16} rms={:.6} zeros={:<5} min={:.3e} max={:.3e}",
                self.step,
                tag.as_deref().unwrap_or("-"),
                s.rms,
                s.zeros,
                s.min,
                s.max
            );
        }
        self.records.push(StatRecord { id, tag, stats: s });
    }
}

//! GraphPlan:图三态编排面(P0/M1;`docs/arch/session-plan.md` 的新世界实现)。
//!
//! **命名裁定(2026-09-26 用户裁决)**:本类型原名 `Session`,占用了
//! 应用层会话词汇,正名 **`GraphPlan`** —— 它是 decode 图的捕获/回放
//! 计划(底层执行概念),与客户端 Session(会话)/ Turn(一条消息)/
//! Step(推理段)三层应用词汇无关。客户端会话见 [`crate::session`]。
//!
//! 编排纪律(session-plan.md §一):算子编排闭包只
//! 只写一份,执行三态(eager/捕获/回放)由 GraphPlan 承载。
//!
//! **移植自 engine-bak/session.rs(464 行)的流程纪律**,数据面全面换代:
//! ETensor/KernelCtx/CudaPool/捕获会话 → TensorOps 声明 + DeviceClient
//! (唯一设备标准)。
//!
//! 三态(M1,2026-09-26):
//! - **eager**:槽装填(write_block 原位)→ 闭包声明 → 计算解释器归约;
//! - **捕获**:plan 尾部 `graph_begin` → 全输出归约(launch 入图,alloc 走
//!   slab;捕获期 Htod/Dtoh/Sync 拒绝)→ `graph_end`;
//! - **回放**:step = 槽装填 + `graph_launch`(指针稳定契约:槽/状态/
//!   输出块全部持久,由捕获窗账房强租约保证)。
//!
//! **两道捕获预检**(A1.4 精神:预检降级,禁止撞死):
//! 1. 树审计——输出树含 `Htod` 即降级(捕获期 Htod 禁;运行期装填一律
//!    write_block 原位,不再 htod 重绑);
//! 2. face 能力——`graph_begin` Err(如 CpuFace)即 EagerFallback。
//!    捕获窗内 Err = 硬错(窗已脏;graph abort 原语挂账 iface)。
//!
//! warmup = 姿势 6 lite:init 数据 eager dry 一遍(声明违约 plan 边界即
//! 暴露;捕获不执行 kernel,状态不被捕获推进)。

use std::cell::RefCell;
use std::collections::HashMap;

use owl_iface::contract::{Bytes, DeviceClient, Dtype, GraphId, ModelError};
use owl_models::interpreters::{eval_ops_multi, eval_ops_multi_scoped_env};
use owl_models::module::TpCtx;
use owl_models::ops::{Op, PlanNode};
use owl_models::TensorOps;

type Result<T> = std::result::Result<T, ModelError>;

// ============================================================================
// §1 槽词汇(声明;数据面在 GraphPlan)
// ============================================================================

/// 输入槽:命名持久设备缓冲,每步 write_block 原位装填(指针稳定 ——
/// 捕获回放的根基;decode:frontier/positions/slots…)
#[derive(Clone, Debug)]
pub struct InputSlot {
    pub name: String,
    /// 容量(元素;M0.5 步数据须等长,档位 narrow 视图随 batching 引入)
    pub len: usize,
    /// 声明的 dtype(缺省 F32;FI 表件槽 = U32 位型)
    pub dtype: Dtype,
    /// 声明的形状(缺省 `[1, len]` = 历史面;prefill 桶槽面用 [`Self::rank1`])
    pub shape: Vec<usize>,
    /// 初始数据(warmup dry 执行用;缺省全零。**U32 槽按位型承载** ——
    /// 调用方写 `f32::from_bits(真 u32)` 即可)
    pub init: Vec<f32>,
}

impl InputSlot {
    pub fn f32(name: impl Into<String>, len: usize) -> Self {
        Self {
            name: name.into(),
            len,
            dtype: Dtype::F32,
            shape: vec![1, len],
            init: Vec::new(),
        }
    }

    /// **rank-1 声明**(形状 = `[len]`)。prefill 桶槽面必须用它:闭包里的
    /// 张量要与 eager 路**同形**(eager 的 `from_host(F32, vec![t], …)` 是
    /// rank-1;算子准入表按 rank 对位)。
    #[must_use]
    pub fn rank1(mut self) -> Self {
        self.shape = vec![self.len];
        self
    }

    /// **U32 位型槽**(FI 的 `q_cu`/`indices`/`indptr`/`last_len` 四件;
    /// 声明 dtype 进准入表 —— 历史 F32 声明会被 FI 的线格式门当场拒:
    /// "q_cu 期望 dtype = U32 实得 F32")
    pub fn u32(name: impl Into<String>, len: usize) -> Self {
        Self {
            name: name.into(),
            len,
            dtype: Dtype::U32,
            shape: vec![len],
            init: Vec::new(),
        }
    }

    pub fn init(mut self, v: Vec<f32>) -> Self {
        self.init = v;
        self
    }
}

/// 输出槽:闭包登记的声明树,解释器归约后按名收割
#[derive(Clone, Debug)]
pub struct OutputSlot {
    pub name: String,
    /// 声明形状(收割量守卫)
    pub shape: Vec<usize>,
    /// 收割 dtype(f16 基线:logits f16;read_output_f32 按此解码)
    pub dtype: Dtype,
}

impl OutputSlot {
    pub fn f32(name: &'static str, shape: &[usize]) -> Self {
        Self { name: name.to_string(), shape: shape.to_vec(), dtype: Dtype::F32 }
    }

    /// f16 输出(F5 整模切换;read_output_f32 自动解码回 f32)
    pub fn f16(name: &'static str, shape: &[usize]) -> Self {
        Self { name: name.to_string(), shape: shape.to_vec(), dtype: Dtype::F16 }
    }
}

/// 会话描述
pub struct GraphPlanDesc {
    pub inputs: Vec<InputSlot>,
    pub outputs: Vec<OutputSlot>,
    /// 捕获三态开关(true = 尝试捕获,预检不过自动 EagerFallback)
    pub capture: bool,
    /// 捕获 slab 固定档占用中(knobs.capture_slab_mb.is_some();true =
    /// 跳过 warmup 计量 hint,用固定档 —— 测试/诊断兼容臂)
    pub slab_fixed: bool,
    /// 槽回读诊断打印(knobs.probes.debug;显式依赖律,原 OWL_DEBUG)
    pub debug_probe: bool,
    /// **捕获窗锚定活性回收**(M1;缺省 false = 历史 plain 口径)。
    /// true ⇒ 捕获窗内死块即时 free ⇒ slab 容量从"分配总和"降到"峰值活跃集"
    /// (27B/512-tok chunk 实测:总和 9362 MiB ⇒ OOM;峰值 193 MiB ⇒ Captured)。
    /// 安全依据见 `crates/backends/cuda/src/state.rs::GraphRefs` + `free_blocks`:
    /// scoped 归约只归还**不可达**中间块,而捕获按求值序记录 ⇒ 复用点在所有
    /// 消费者之后。prefill 图族开;decode/verify 族保持 false(它们的总和本来
    /// 就只有几百 MB,改动 = 行为改动,不在本刀范围)。
    pub reclaim_capture: bool,
    /// **跳过 warmup 预跑**(缺省 true = 历史姿势 6 lite)。
    ///
    /// false 的语义:`plan()` 不再 eager 跑一遍闭包 ⇒ ①**不计量** ⇒ 必须
    /// 配合固定 slab 档(`slab_fixed`,否则结构化拒)②**不产生状态副作用**
    /// ⇒ 捕获可脱离"静止点律"(惰性按需捕获 / LRU 驱逐的前置件)。
    /// 代价:声明树门禁不再于捕获前预检(窗口内才暴露 ⇒ 触发 abort 收窗)。
    pub warmup: bool,
}

impl Default for GraphPlanDesc {
    /// **手写 Default**(不用 derive):`warmup` 的"安全缺省值"是 **true**
    /// —— `bool::default()` 会给出 false,而 false 要求 `slab_fixed`,是
    /// 个只有在 `..Default::default()` 简写里才会踩的坑。
    fn default() -> Self {
        Self {
            inputs: Vec::new(),
            outputs: Vec::new(),
            capture: false,
            slab_fixed: false,
            debug_probe: false,
            reclaim_capture: false,
            warmup: true,
        }
    }
}

/// plan 结果(A1.4:预检失败 = EagerFallback,同闭包直发)
#[derive(Clone, Debug)]
pub enum PlanOutcome {
    Captured,
    EagerFallback { reason: String },
}

// ============================================================================
// §2 PlanCtx:闭包取输入声明 / 登记输出声明(纯描述;零 await)
// ============================================================================

pub struct PlanCtx {
    inputs: HashMap<String, TensorOps>,
    outputs: RefCell<Vec<(String, TensorOps)>>,
}

impl PlanCtx {
    /// 输入槽声明(本步设备块的 Block 叶子;克隆共享同块,零拷贝)
    pub fn input(&self, name: &str) -> Result<TensorOps> {
        self.inputs
            .get(name)
            .cloned()
            .ok_or_else(|| ModelError::Msg(format!("step: 未声明输入槽 {name}")))
    }

    /// 可选输入槽(刀D 取证面:探针槽缺省 = None,不违约)
    pub fn input_opt(&self, name: &str) -> Result<Option<TensorOps>> {
        Ok(self.inputs.get(name).cloned())
    }


    /// 输出登记:该声明树即本步输出(解释器归约;块生命周期归会话)。
    /// E5-M4:String 名(verify 图的 fold 记录槽 = 每层 × 8 动态名)
    pub fn output(&self, name: impl Into<String>, t: &TensorOps) -> Result<()> {
        self.outputs.borrow_mut().push((name.into(), t.clone()));
        Ok(())
    }

    /// 取登记(会话在闭包返回后收割;消费即清空)
    fn take_outputs(&self) -> Vec<(String, TensorOps)> {
        self.outputs.borrow_mut().split_off(0)
    }

}

// ============================================================================
// §3 GraphPlan:一次声明,三态执行
// ============================================================================

struct InSlotDev {
    name: String,
    len: usize,
    /// 声明形状/dtype(闭包与写入侧各自用;见 [`InputSlot`])
    shape: Vec<usize>,
    dtype: Dtype,
    block: Bytes,
}

struct OutVal {
    block: Bytes,
    len: usize,
    dtype: Dtype,
}

#[derive(Clone, Copy, Debug)]
enum Mode {
    Eager,
    Captured { graph: GraphId },
}

pub struct GraphPlan<D: DeviceClient> {
    face: D,
    /// 解释器执行环境(显式依赖律,引擎 boot 组装后下传)。
    /// 用途:①eager 臂的 scoped 回收/env 定相(perf tag 不再落 `unk`)
    /// ②层声明侧的分派旋钮(与 ForwardCtx.env 同源)
    env: owl_models::env::EnvProvider,
    /// **张量并行上下文(S4 ⑤;`None` = 单卡)** —— 与 env 同律：
    /// 装配期烘焙进本图,eager/捕获/piecewise 三条求值路逐段带进解释器
    /// (行并行层的 allreduce 注入靠它;层只给标记)。
    tp: Option<TpCtx>,
    /// 槽回读诊断打印(desc.debug_probe;显式依赖律)
    debug_probe: bool,
    /// 捕获窗锚定活性回收(M1;见 `GraphPlanDesc` 同名字段)
    reclaim_capture: bool,
    inputs: Vec<InSlotDev>,
    out_specs: HashMap<String, usize>, // name → 元素数
    out_dt: HashMap<String, Dtype>,    // name → 收割 dtype(F5 logits f16)
    last: HashMap<String, OutVal>,
    mode: Mode,
    #[allow(clippy::type_complexity)] // 捕获闭包(crate 内部契约)
    forward: Box<dyn Fn(&PlanCtx) -> Result<()>>,
    /// **本 plan 的捕获 slab 占用(字节;0 = eager/未捕获/固定档)** ——
    /// 图显存总账的账源(2026-10-16 立)。口径 = `graph_begin` 真分配的容量
    /// (= 定尺 hint;单图 ⇒ 一条)。
    slab_bytes: usize,
}

impl<D: DeviceClient> GraphPlan<D> {
    /// **face 句柄克隆**(诊断投影出口;偶极:server 是 actor,克隆发送端即可)
    pub(crate) fn face_clone(&self) -> Option<D> {
        self.face.face_clone()
    }

    /// 本 plan 的捕获 slab 占用(字节;图显存总账账源;0 = eager/未捕获)
    pub fn slab_bytes(&self) -> usize {
        self.slab_bytes
    }

    /// 声明 + warmup + (可选)捕获。流程纪律全部在此闭合:
    /// 姿势 6 warmup 门禁(lite)/ 两道捕获预检 / EagerFallback。
    pub async fn plan(
        mut face: D,
        env: owl_models::env::EnvProvider,
        tp: Option<TpCtx>,
        desc: GraphPlanDesc,
        forward: impl Fn(&PlanCtx) -> Result<()> + 'static,
    ) -> Result<(Self, PlanOutcome)> {
        // 槽设备面:持久块,htod(init 缺省补零);此后只 write_block 原位
        let mut ins: Vec<InSlotDev> = Vec::new();
        for i in desc.inputs {
            let mut init = i.init;
            init.resize(i.len, 0.0);
            // U32 位型槽:init 值按 `f32::to_bits` 语义承载(F32 槽同字节)
            let bytes: Vec<u8> = if i.dtype == Dtype::U32 {
                init.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect()
            } else {
                f32b(&init)
            };
            // **参数面走 COMPUTE 流**(2026-10-19 定谳;桶门 0.8B IMA 的根因):
            // 槽块由 `cuMemAllocAsync` **按流**落定,而槽的每步装填
            // (`fill_slots` → `write_blocks_f32_on(Compute)`)与消费它的发射都在
            // COMPUTE 上 ⇒ 若槽块在 **H2D** 流分配/首次上传,COMPUTE 读者与
            // 该分配之间**没有序**(A13 同族:跨流"用前未分配")。
            // 大模型 boot 慢、中间有栅栏 ⇒ 侥幸不炸;0.8B 小几何 boot 极快 ⇒
            // 首个 warmup 就撞上(compute-sanitizer:`owl_embed_f16` 读
            // `plan.in.frontier`、地址"potentially made before memory is allocated")。
            // 修法 = 与 2026-10-17 参数同流律对齐:**同流 FIFO,结构上无竞态**。
            let block = face
                .htod_on(
                    owl_iface::contract::UpStream::Compute,
                    i.dtype,
                    &i.shape,
                    &bytes,
                    &format!("plan.in.{}", i.name),
                )
                .await?;
            ins.push(InSlotDev {
                name: i.name,
                len: i.len,
                shape: i.shape,
                dtype: i.dtype,
                block,
            });
        }

        let mut sess = Self {
            face,
            env,
            tp,
            debug_probe: desc.debug_probe,
            reclaim_capture: desc.reclaim_capture,
            inputs: ins,
            out_specs: desc
                .outputs
                .iter()
                .map(|o| (o.name.clone(), o.shape.iter().product::<usize>()))
                .collect(),
            out_dt: desc
                .outputs
                .iter()
                .map(|o| (o.name.clone(), o.dtype))
                .collect(),
            last: HashMap::new(),
            mode: Mode::Eager,
            forward: Box::new(forward),
            slab_bytes: 0,
        };

        // 输入槽装填(init 值落位;捕获/eager 两路都要)
        sess.fill_slots(&[]).await?;
        // ── warmup(姿势 6 lite):init 数据全链 eager dry 一遍 ──
        // **仅捕获前需要**(2026-10-13 定谳):预热唯二产出 = ①slab 定量 hint
        // ②声明树门禁 —— 两者都只服务捕获窗。无捕获(no_graph/eager 档)时
        // 它是纯浪费:27B 实测 26s = 32 次 eager 整模前向(fold 桶 m=0..6
        // 的 m 步链占 ~28 次;每次 ~0.8s = eager 逐算子发射开销,图回放同
        // 规模仅 ~14ms),而捕获本身 ≈0(no_graph 对照 26.25s vs 正常 26.18s)。
        // 跳过预热 → 该档启动立省 26s;声明错误改在首次推理时暴露
        // (no_graph 本就是诊断档,取舍成立)。
        // slab 定量(E8→A1,2026-10-10):warmup 前弃旧账,后取闭包真
        // 分配字节 → hint = ×1.15 + 1MiB。env 显式设 OWL_CAPTURE_SLAB_MB
        // 时固定档优先(state 侧裁决),此处不置 hint 兼容旧档位。
        if desc.capture && !desc.warmup && !desc.slab_fixed {
            // 无 warmup ⇒ 无计量 ⇒ 必须给固定档;否则捕获窗内在 OOM 边缘
            // 试探(窗已脏,只能 abort)。结构性拒,不给"猜一个 64MiB"的兜底。
            return Err(ModelError::Msg(
                "GraphPlanDesc{warmup:false} 必须同时 slab_fixed(capture_slab_mb 固定档);\
                 否则捕获窗容量无来源"
                    .to_string(),
            ));
        }
        if desc.capture && desc.warmup {
            let _ = owl_shared::slab_hint::meter_take();
            // M1:影子 arena 探针(warmup 与捕获窗是同一份 DAG/同一批形状,
            // 且都走 scoped 归约 ⇒ 影子高水位 = 捕获真需要的容量)
            owl_shared::slab_hint::probe_on(true);
            let warm_res = sess.eval_current_dry().await;
            // 探针必须收(即使 warmup 失败),否则 PROBE_ON 残留污染下一图
            let peak = owl_shared::slab_hint::probe_end();
            // 峰值归因(2026-10-15;用户令"先归因"):峰值时刻的构成 + 每名字分配总量
            // —— 回答"193MiB 里谁占大头"(是搬算子还是减死重量,靠它定)
            let attr = owl_shared::slab_hint::take_probe_report();
            if !attr.is_empty() {
                owl_shared::log_info!("graph-plan", "slab 峰值归因:{attr}");
            }
            // 活性回收账(与归因同窗):一行指认"哪个出口把块拦下了"
            // —— 存活构成与分配总量同比例(一字节没 free)时的唯一追问面。
            owl_shared::log_info!(
                "graph-plan",
                "活性回收账:{} ‖ 引用计数推送 {} 次",
                owl_shared::slab_hint::live_diag::take_report(),
                owl_shared::slab_hint::refs_pushed()
            );
            warm_res?;
            sess.face.sync().await?;
            // warmup 输出块归还(原 `clear()` 只丢句柄、块留池外 ⇒ 每次
            // plan 一次性泄漏;此处显式 free)
            let warm: Vec<u64> = sess.last.values().map(|o| o.block.id).collect();
            sess.last.clear();
            if !warm.is_empty() {
                sess.face.free(&warm).await?;
            }
            let metered = owl_shared::slab_hint::meter_take() as usize;
            if !desc.slab_fixed {
                // **口径必须与捕获模式一致**(M1 实测教训):锚定回收 ⇒ 需求 =
                // 峰值活跃集(影子高水位);plain(历史口径)⇒ 窗内块常驻 ⇒
                // 需求 = 分配总和。混用会把族定小 ⇒ 捕获中途耗尽当场炸。
                let base = if desc.reclaim_capture { peak } else { metered }.max(4 << 20);
                // **余量:原 +12.5%+1MiB → +100%+2MiB**(§65.3-2 定谳;2026-10-19):
                // `peak` 是 **warmup(eager)期**的实测影子峰值,而**捕获期**的回收点
                // 更少(节点合并 ⇒ 块活跃期变长):v1 两核融合(§65)把 stage+reduce
                // 并一节点后,实测**需求/base ∈ [1.40, 2.45)**(base=10.1MiB 时,
                // 12.36MiB 的旧 hint 差 139264B 仍爆;24.7MiB 够)。
                // ⇒ 取 **2× + 2MiB**(覆盖 2.0 以下;仍比双保险(下游再 ×2)省 ~40%)。
                // 真根治 = 捕获期 liveness 直接跟踪/或 carve 失败自动重试
                // (后者受 FnOnce 闭包限制,不可行 —— 见 §65.3-2)。
                let hint_bytes = base.saturating_mul(2) + (2 << 20);
                owl_shared::slab_hint::set_hint(hint_bytes);
                sess.slab_bytes = hint_bytes; // 图显存总账账源
                owl_shared::log_info!(
                    "graph-plan",
                    "slab 定尺({}):峰值 {}MiB / 总和 {}MiB ‖ 引擎 live={}MiB",
                    if desc.reclaim_capture { "锚回收·峰值" } else { "plain·总和" },
                    peak >> 20,
                    metered >> 20,
                    owl_shared::vram::live_bytes() >> 20
                );
            }
        }

        // ── 捕获(两道预检;窗内硬错直接上抛)──
        let outcome = if desc.capture {
            let r = sess.try_capture().await;
            // 收尾:graph_begin 已消费 hint;失败路径防陈旧 hint 污染下一图
            owl_shared::slab_hint::clear_hint();
            match r {
                Ok(o) => o,
                Err(e) => {
                    // **失败不留尸**(2026-10-15):窗内已切出的块若无人认领,
                    // 会把整条 slab 钉到进程结束(最坏 GB 级)。收窗 + 槽归还;
                    // 两者失败都**不覆盖**原错(记可见警告,原错优先上抛)。
                    sess.abandon().await;
                    return Err(e);
                }
            }
        } else {
            fallback("未请求捕获")
        };
        Ok((sess, outcome))
    }

    // ── piecewise(分段捕获 + 断点)面已于 2026-10-18 W5 **整体退役** ──
    // 原址:`plan_piecewise` / `replay_piecewise` / `Mode::CapturedPiecewise` /
    // `BreakPoint`·`PiecewiseRt` / `pw_carry`(接力棒)/ `pw_*_stats` 诊断面。
    // 退役理由:W3/W4 之后 FI 核留捕获图内、调度表由窗外 `fi_plan_sync` 刷 ⇒
    // 断点面零消费者;而接力棒"每步造新节点 id ⇒ 只增"正是端到端 O(步数²)
    // 那条慢化病的宿主(设计卷 §3.3)。需要时从 git 历史取回。

    /// 相位定相(perf tag 维度;引擎逐相位调用)。
    /// 构造期 env 是**快照**,相位是**逐相位状态** ⇒ 需显式同步
    /// (引擎侧 `RunningEngine::set_phase` 同时更新自身 env 与各 plan)。
    pub fn set_phase(&mut self, p: owl_models::env::EvalPhase) {
        self.env.phase = p;
    }

    /// 每步执行:装填输入(write_block 原位)→ 回放或 eager 直发。
    /// 未提到的输入槽保持上步内容(典型:常量槽)。
    pub async fn step(&mut self, inputs: &[(&str, &[f32])]) -> Result<()> {        self.fill_slots(inputs).await?;
        // 诊断:槽回读(desc.debug_probe;取证每步标量是否真落烘焙槽 ——
        // 复读病理立案的仪表盘)
        if self.debug_probe {
            for (name, vals) in inputs {
                if let Some(slot) = self.inputs.iter().find(|s| s.name == *name) {
                    let mut buf = vec![0u8; vals.len() * 4];
                    self.face.dtoh(&slot.block, &mut buf).await?;
                    let back: Vec<f32> =
                        buf.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
                    owl_shared::log_debug!("slot-dump", "{name} = {back:?}");
                }
            }
        }
        match self.mode {
            Mode::Captured { graph } => self.face.graph_launch(graph).await?,
            Mode::Eager => self.eval_current().await?,
        }
        Ok(())
    }


    /// 裸回放(E5-DF4 性能判别:不装填输入,纯 launch + 同步由调用方;
    /// 用于测量图独占 GPU 时间 —— 队列排水污染排查)
    pub async fn replay(&mut self) -> Result<()> {
        match self.mode {
            Mode::Captured { graph } => self.face.graph_launch(graph).await?,
            Mode::Eager => self.eval_current().await?,
        }
        Ok(())
    }

    /// 刀D 取证:输入槽原始字节回读(ts_buf 时间线;同步后调用)
    pub async fn read_input_slot_bytes(&mut self, name: &str) -> Result<Vec<u8>> {
        let slot = self
            .inputs
            .iter()
            .find(|s| s.name == name)
            .ok_or_else(|| ModelError::Msg(format!("read_input_slot: 无 {name}")))?;
        let mut buf = vec![0u8; slot.len * 4];
        self.face.dtoh(&slot.block, &mut buf).await?;
        Ok(buf)
    }
    /// 输出块句柄(E5-M4:设备侧消费 —— fold 记录/hidden 上游;
    /// 块 = 最近一次 step/捕获的归约产物,指针在捕获态恒定)
    pub fn output_block(&self, name: &str) -> Option<Bytes> {
        self.last.get(name).map(|o| o.block.clone())
    }

    /// 输出收割(读语义;块 = 最近一次 step/捕获的归约产物)
    pub async fn read_output_f32(&mut self, name: &str) -> Result<Vec<f32>> {
        let out = self
            .last
            .get(name)
            .ok_or_else(|| ModelError::Msg(format!("read_output: 无输出 {name}(先 step)")))?;
        // 按 slot dtype 解码（f16 基线：logits f16 上载，f32 回 host）
        // **BF16 分支（P1.1 × P2.1，2026-10-19）**：草稿组件的 proj 槽是草稿 dt
        // （量化族检查点 = BF16）；旧实现把非 F16 一律当 F32 读 ⇒ BF16 槽读回错值。
        enum Dec {
            F16,
            Bf16,
            F32,
        }
        let (esize, dec) = match out.dtype {
            Dtype::F16 => (2usize, Dec::F16),
            Dtype::BF16 => (2usize, Dec::Bf16),
            _ => (4usize, Dec::F32),
        };
        let mut buf = vec![0u8; out.len * esize];
        self.face.dtoh(&out.block, &mut buf).await?;
        Ok(match dec {
            Dec::F16 => buf
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
            Dec::Bf16 => buf
                .chunks_exact(2)
                .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
            Dec::F32 => buf
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        })
    }

    /// face 访问器(crate 内编排层用:turn 切换时的状态重置等
    /// 槽外设备操作;对外仍零暴露 —— face 不出 engine crate)
    pub(crate) fn face_mut(&mut self) -> &mut D {
        &mut self.face
    }

    /// 只读面(诊断:句柄计数/账本读 —— 不产生分配)
    pub(crate) fn face_ref(&self) -> &D {
        &self.face
    }

    // ------------------------------------------------------------------
    // 内部:装填 / 声明 / 归约 / 捕获
    // ------------------------------------------------------------------

    /// 槽装填:write_block 原位(指针稳定 —— 回放根基)
    /// 本图是否声明了某输入槽(可选槽的调用方决策面;缺失时 `step` 会报错)
    pub fn has_input(&self, name: &str) -> bool {
        self.inputs.iter().any(|s| s.name == name)
    }

    /// 本图是否声明了某输出槽(诊断/门面读出口的守卫;`read_output_f32` 前查)
    pub fn has_output(&self, name: &str) -> bool {
        self.last.contains_key(name) || self.out_specs.contains_key(name)
    }

    /// 某输入槽的元素数(None = 未声明)
    pub fn input_len(&self, name: &str) -> Option<usize> {
        self.inputs.iter().find(|s| s.name == name).map(|s| s.len)
    }

    async fn fill_slots(&mut self, fills: &[(&str, &[f32])]) -> Result<()> {
        // S1:批量装填 —— 一次设备往返(校验全过才发;错误归因仍逐槽)
        let mut writes: Vec<(&Bytes, usize, &[f32])> = Vec::with_capacity(fills.len());
        for (name, data) in fills {
            let slot = self
                .inputs
                .iter()
                .find(|s| s.name == *name)
                .ok_or_else(|| ModelError::Msg(format!("step: 未声明输入槽 {name}")))?;
            // 0 长槽 = 上游形状 bug(声明期就应拦住);放行会在后端炸
            // `malloc_host(0)` 空指针(2026-10-13 挂账 #7 的源头,守卫在这里)
            if slot.len == 0 {
                return Err(ModelError::Msg(format!(
                    "step: 槽 {name} 声明为 0 元素(零长块写 ⇒ 后端锁页分配空指针;上游形状 bug)"
                )));
            }
            if data.len() != slot.len {
                return Err(ModelError::Msg(format!(
                    "step: 槽 {name} 装填 {} 元素 ≠ 容量 {}",
                    data.len(),
                    slot.len
                )));
            }
            writes.push((&slot.block, 0, data));
        }
        // **参数面 → COMPUTE 流**(2026-10-17):输入槽与紧随其后的 GraphLaunch
        // 同流 FIFO ⇒ 结构上不可能"图读到还没落地的槽"(桶族复用格错值案的根因);
        // 小表(几 KB)放 COMPUTE 无带宽代价,且省掉"等回执"的主机往返。
        self.face
            .write_blocks_f32_on(owl_iface::contract::UpStream::Compute, &writes)
            .await
    }

    /// 闭包声明 + 守卫:返回登记输出(名须在 out_specs)
    /// 声明一趟:跑闭包收输出树(断点表已随 piecewise 退役)。
    fn declare_outputs(&self) -> Result<Vec<(String, TensorOps)>> {
        let mut inputs = HashMap::new();
        for s in &self.inputs {
            inputs.insert(
                s.name.clone(),
                TensorOps::of_block(s.block.id, s.dtype, s.shape.clone()),
            );
        }
        let ctx = PlanCtx { inputs, outputs: RefCell::new(Vec::new()) };
        (self.forward)(&ctx)?;
        let outs = ctx.take_outputs();
        for (name, _) in &outs {
            if !self.out_specs.contains_key(name) {
                return Err(ModelError::Msg(format!("step: 未声明输出槽 {name}")));
            }
        }
        Ok(outs)
    }

    /// eager 归约:全输出 eval 入 last(读序由 dtoh 读语义保证)
    ///
    /// **块生命周期(2026-10-14 修复)**:本函数是 eager 臂唯一入口,
    /// 两处回收缺一不可 ——
    /// 1. **上一轮声明输出**:本函数重写 `last`,旧块到此作废 →
    ///    入口处显式 `free`。契约:`output_*` 句柄仅在**本次回放内**有效;
    ///    跨回放使用 = 悬空 id → 服务端结构化 `DeadBlock`(**fail loud**,
    ///    不会静默错值)。捕获臂不受影响(输出块在 slab 常驻)。
    /// 2. **本轮中间块**:`eval_ops_multi_scoped_env`(per-node 活性回收 +
    ///    末次 arena−roots 清扫)。原实现用 plain `eval_ops_multi`
    ///    (块常驻)**每步泄漏全部中间块** ⇒ 实测 ~100 步
    ///    `CUDA_ERROR_OUT_OF_MEMORY`,且失败后池不可恢复。
    /// **warmup/干跑归约**(S4 ⑤-K3):与 [`Self::eval_current`] 同路,但
    /// **TP 上下文置空** —— 装配期(warmup/分段干跑)只服务计量与声明门禁,
    /// **不得真发射 AR**:AR 是对端配对协议,warmup 单边先行会撞 5s 自旋超时 +
    /// 游标永久失步(2026-10-17 G8 定谳);真 AR 只在回放期(两 rank 同序入队)。
    async fn eval_current_dry(&mut self) -> Result<()> {
        let saved = self.tp.take();
        let r = self.eval_current().await;
        self.tp = saved;
        r
    }

    async fn eval_current(&mut self) -> Result<()> {
        if !self.last.is_empty() {
            let old: Vec<u64> = self.last.values().map(|o| o.block.id).collect();
            self.last.clear();
            self.face.free(&old).await?;
        }
        // E5-M4:**多根共享 memo 单次归约**(eval_ops_multi)—— 逐根
        // eval_ops 会把共享子树(整模前向)每根重执行一遍:verify 图
        // 389 根(384 记录 = 整模内部节点)实测 warmup 387 次整模前向
        // ≈ 19GB 中间块泄漏 + 分钟级耗时(刀 1.6 同款病,warmup/inline
        // 路径补修;捕获窗 eval_ops_multi 早已正确)
        let outs = self.declare_outputs()?;
        let trees: Vec<&TensorOps> = outs.iter().map(|(_, t)| t).collect();
        let results = eval_ops_multi_scoped_env(&trees, &mut self.face, self.env, self.tp.as_ref()).await?;
        for ((name, _), block) in outs.iter().zip(results) {
            let want = self.out_specs[name];
            if block.len != 0 && block.len != want {
                return Err(ModelError::Msg(format!(
                    "step: 输出 {name} 归约 {} 元素 ≠ 声明 {want}",
                    block.len
                )));
            }
            let dtype = self.out_dt[name];
            self.last.insert(name.clone(), OutVal { block, len: want, dtype });
        }
        Ok(())
    }

    /// 捕获(两道预检 → graph_begin → 全输出归约入图 → graph_end)。
    /// 窗内 Err = 硬错(窗已脏;graph abort 原语挂账 iface)。
    async fn try_capture(&mut self) -> Result<PlanOutcome> {
        // 预检 1:树审计 —— Htod 捕获期禁(运行期装填已全部 write_block)
        let outs = self.declare_outputs()?;
        if outs.iter().any(|(_, t)| {
            t.flatten().iter().any(|n| matches!(n.op(), Op::Htod { .. }))
        }) {
            return Ok(fallback("输出树含 Htod(捕获期禁)"));
        }

        // 预检 2:face 能力(CpuFace 等结构化拒绝 → EagerFallback)
        if let Err(e) = self.face.graph_begin().await {
            return Ok(fallback(&format!("face 无图能力: {e}")));
        }

        // 捕获窗:launch 入图 / alloc 走 slab;输出块 id 捕获期即定。
        // 刀1.6:**多根共享 memo 单次归约** —— 输出树共享模型子树(同
        // 节点 id),逐根独立 eval 会把整模捕获 2+ 遍(图内 2.64× 整模
        // 内核,回放税 ~2× 的历史包袱,2026-10-03 派发税立案定谳)。
        let trees: Vec<&TensorOps> = outs.iter().map(|(_, t)| t).collect();
        let mut win_err: Option<ModelError> = None;
        let mut blocks: Vec<(String, Bytes, usize, Dtype)> = Vec::new();
        // M1:锚定回收族走 scoped 归约(窗内死块即时 free ⇒ 回 slab arena
        // 复用 ⇒ 容量 = 峰值活跃集)。plain 族保持历史口径(窗内块常驻)。
        let eval_res = if self.reclaim_capture {
            eval_ops_multi_scoped_env(&trees, &mut self.face, self.env, self.tp.as_ref()).await
        } else {
            eval_ops_multi(&trees, &mut self.face, self.tp.as_ref()).await
        };
        match eval_res {
            Ok(results) => {
                for ((name, _), b) in outs.iter().zip(results) {
                    let want = self.out_specs[name];
                    if b.len == 0 || b.len == want {
                        let dtype = self.out_dt[name];
                        blocks.push((name.clone(), b, want, dtype));
                    } else {
                        win_err = Some(ModelError::Msg(format!(
                            "capture: 输出 {name} 归约 {} 元素 ≠ 声明 {want}",
                            b.len
                        )));
                        break;
                    }
                }
            }
            Err(e) => win_err = Some(e),
        }
        let gid = self.face.graph_end().await;

        match (win_err, gid) {
            (None, Ok(graph)) => {
                self.mode = Mode::Captured { graph };
                self.last = blocks
                    .into_iter()
                    .map(|(name, block, len, dtype)| (name, OutVal { block, len, dtype }))
                    .collect();
                Ok(PlanOutcome::Captured)
            }
            // 窗已脏:不可静默降级 ⇒ **abort 收窗**(销毁半成品图 + 窗内自产块
            // 整批作废 ⇒ slab 释放),然后把原错上抛。abort 自身失败不覆盖原错
            // (记可见警告);night 若 `graph_end` 已把窗收掉,abort 会结构化拒
            // ⇒ 同样只记警告。
            (Some(e), _) => {
                self.abort_window().await;
                Err(e)
            }
            (None, Err(e)) => {
                self.abort_window().await;
                Err(ModelError::Msg(format!("capture: graph_end 失败 {e}")))
            }
        }
    }

    /// 捕获窗作废(abort)+ 说明性日志。**幂等**:若窗已被 `graph_end` 收掉,
    /// face 会结构化拒 ⇒ 这里只记可见警告,不掩盖既有错。
    async fn abort_window(&mut self) {
        if let Err(e) = self.face.graph_abort().await {
            owl_shared::log_warn!("graph-plan", "捕获窗作废失败(窗可能已收;slab 或留尸): {e}");
        }
    }

    /// **失败路径清理**(`plan()` 捕获失败时调用):作废窗 + 归还输入槽。
    ///
    /// 输入槽是 `plan()` 期 `htod` 的 `Owned` 块,由图引用租约保护;捕获失败
    /// ⇒ 没有图 ⇒ 租约未生效(`adopt` 未跑)⇒ 可安全归还,否则每次失败捕获都
    /// 漏一批槽到进程结束。
    async fn abandon(&mut self) {
        self.abort_window().await;
        let ids: Vec<u64> = self.inputs.iter().map(|s| s.block.id).collect();
        if !ids.is_empty() {
            if let Err(e) = self.face.free(&ids).await {
                owl_shared::log_warn!("graph-plan", "失败路径归还输入槽失败(泄漏到进程结束): {e}");
            }
        }
        self.inputs.clear();
        self.last.clear();
    }

    /// **单图退休**(LRU 驱逐 / plan 生命期收尾的底层):销毁图 + 归还输入槽
    /// + 丢弃输出句柄。次序**不可颠倒**(图先、槽后):图在时槽被租约钉住,
    /// 先 free 只会被 `free_blocks` 响亮拒收。
    pub async fn close(&mut self) -> Result<()> {
        if let Mode::Captured { graph } = self.mode {
            self.face.graph_destroy(graph).await?;
            self.mode = Mode::Eager;
        }
        let ids: Vec<u64> = self.inputs.iter().map(|s| s.block.id).collect();
        self.inputs.clear();
        self.last.clear();
        if !ids.is_empty() {
            self.face.free(&ids).await?;
        }
        Ok(())
    }
}

fn fallback(reason: &str) -> PlanOutcome {
    PlanOutcome::EagerFallback { reason: reason.to_string() }
}

pub(crate) fn f32b(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

// ============================================================================
// §4 测试(CPU:闭环 + 降级;GPU 门控:回放 == eager 数值对拍)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use owl_cpu::CpuFace;
    use owl_models::interpreters::eval_ops;
    use owl_macros::owl_test;


    fn ones_w() -> TensorOps {
        let bytes: Vec<u8> = [1.0f32; 16].iter().flat_map(|f| f.to_le_bytes()).collect();
        TensorOps::from_host(Dtype::F32, vec![4, 4], &bytes)
    }

    #[tokio::test]
    async fn eager_cpu_end_to_end() {
        // out = (x + y) × W,W 全 1 → out = (x+y) 的行和
        let w = ones_w();
        let face = CpuFace::new();
        let desc = GraphPlanDesc {
            inputs: vec![
                InputSlot::f32("x", 4),
                InputSlot::f32("y", 4).init(vec![10.0, 20.0, 30.0, 40.0]),
            ],
            outputs: vec![OutputSlot::f32("s", &[1, 4])],
            capture: false,
            slab_fixed: false,
            debug_probe: false,
            // 本族不加锚定回收(plain 口径:窗内块常驻 —— 历史行为零变化)
            reclaim_capture: false,
            warmup: true,
        };
        let (mut sess, outcome) = GraphPlan::plan(
            face,
            owl_models::env::EnvProvider::default(),
            None,
            desc,
            move |sc: &PlanCtx| -> Result<()> {
                let x = sc.input("x")?;
                let y = sc.input("y")?;
                sc.output("s", &x.add(&y).matmul(&w))?;
                Ok(())
            },
        )
        .await
        .expect("plan(warmup 门禁)");
        assert!(matches!(outcome, PlanOutcome::EagerFallback { .. }));

        // 业务步:y 未装填 → 保持 init 内容(常量槽语义)
        sess.step(&[("x", &[1.0, 2.0, 3.0, 4.0])]).await.expect("step");
        let s = sess.read_output_f32("s").await.expect("read");
        assert_eq!(s, vec![110.0; 4], "(x+y)=[11,22,33,44] × 全 1 阵 = 行和 110");

        sess.step(&[("x", &[0.0; 4]), ("y", &[1.0; 4])]).await.expect("step2");
        let s = sess.read_output_f32("s").await.expect("read2");
        assert_eq!(s, vec![4.0; 4]);
    }

    /// `warmup:false` 必须配固定 slab 档(否则容量无来源)⇒ **结构化拒**,
    /// 不给"猜一个 64MiB"的兜底(踩了会在捕获窗 OOM,窗脏只能 abort)。
    #[tokio::test]
    async fn warmup_false_without_fixed_slab_is_rejected() {
        let face = CpuFace::new();
        let desc = GraphPlanDesc {
            inputs: vec![InputSlot::f32("x", 4)],
            outputs: vec![OutputSlot::f32("s", &[1, 4])],
            capture: true,
            warmup: false,
            ..Default::default()
        };
        let r = GraphPlan::plan(face, owl_models::env::EnvProvider::default(), None, desc, |sc: &PlanCtx| {
            let x = sc.input("x")?;
            sc.output("s", &x)
        })
        .await;
        let e = match r {
            Err(e) => e,
            Ok(_) => panic!("warmup:false + slab_fixed:false 必须拒"),
        };
        assert!(
            format!("{e}").contains("warmup:false"),
            "拒语应指明病因,实得 {e}"
        );
    }

    /// `close()`(单图退休原语):eager 档 = 归还输入槽 + 幂等(第二次无槽可还
    /// 仍是 Ok)。捕获档的图销毁面由 cuda 侧 `gpu_server` 步 11 覆盖。
    #[tokio::test]
    async fn close_is_idempotent_and_frees_slots() {
        let w = ones_w();
        let face = CpuFace::new();
        let desc = GraphPlanDesc {
            inputs: vec![InputSlot::f32("x", 4), InputSlot::f32("y", 4)],
            outputs: vec![OutputSlot::f32("s", &[1, 4])],
            capture: false,
            ..Default::default()
        };
        let (mut sess, outcome) = GraphPlan::plan(
            face,
            owl_models::env::EnvProvider::default(),
            None,
            desc,
            move |sc: &PlanCtx| -> Result<()> {
                let x = sc.input("x")?;
                let y = sc.input("y")?;
                sc.output("s", &x.add(&y).matmul(&w))?;
                Ok(())
            },
        )
        .await
        .expect("plan");
        assert!(matches!(outcome, PlanOutcome::EagerFallback { .. }));
        sess.close().await.expect("close 应成功");
        sess.close().await.expect("重复 close 幂等");
    }

    #[tokio::test]
    async fn warmup_catches_declaration_poison() {
        // 闭包带形状违约([4] + [3] → 毒):plan 边界暴露,不污染业务步
        let face = CpuFace::new();
        let desc = GraphPlanDesc {
            inputs: vec![InputSlot::f32("a", 4), InputSlot::f32("b", 3)],
            outputs: vec![OutputSlot::f32("s", &[1, 4])],
            capture: true,
            slab_fixed: false,
            debug_probe: false,
            // 本族不加锚定回收(plain 口径:窗内块常驻 —— 历史行为零变化)
            reclaim_capture: false,
            warmup: true,
        };
        let r = GraphPlan::plan(face, owl_models::env::EnvProvider::default(), None, desc, |sc: &PlanCtx| -> Result<()> {
            let a = sc.input("a")?;
            let b = sc.input("b")?;
            sc.output("s", &a.add(&b))?;
            Ok(())
        })
        .await;
        let err = match r {
            Ok(_) => panic!("warmup 应拦截毒值声明"),
            Err(e) => format!("{e}"),
        };
        assert!(err.contains("毒值"), "拦截原因为毒值落地,得 {err}");
    }

    #[tokio::test]
    async fn capture_falls_back_on_cpu() {
        // CPU face 无图能力:capture=true 须结构化降级,闭环不受影响。
        // W 块化(否则被预检 1 的 Htod 审计先拦,测不到能力预检)
        let mut face = CpuFace::new();
        let w_bytes: Vec<u8> = [1.0f32; 16].iter().flat_map(|f| f.to_le_bytes()).collect();
        let w_b = eval_ops(TensorOps::from_host(Dtype::F32, vec![4, 4], &w_bytes).step(), &mut face)
            .await
            .expect("W 块化");
        let w = TensorOps::of_block(w_b.id, Dtype::F32, vec![4, 4]);
        let desc = GraphPlanDesc {
            inputs: vec![InputSlot::f32("x", 4)],
            outputs: vec![OutputSlot::f32("s", &[1, 4])],
            capture: true,
            slab_fixed: false,
            debug_probe: false,
            // 本族不加锚定回收(plain 口径:窗内块常驻 —— 历史行为零变化)
            reclaim_capture: false,
            warmup: true,
        };
        let (mut sess, outcome) = GraphPlan::plan(
            face,
            owl_models::env::EnvProvider::default(),
            None,
            desc,
            move |sc: &PlanCtx| -> Result<()> {
                let x = sc.input("x")?;
                sc.output("s", &x.matmul(&w))?;
                Ok(())
            },
        )
        .await
        .expect("plan");
        assert!(
            matches!(outcome, PlanOutcome::EagerFallback { ref reason } if reason.contains("图能力")),
            "CPU 应因无图能力降级,得 {outcome:?}"
        );
        sess.step(&[("x", &[1.0, 2.0, 3.0, 4.0])]).await.expect("step");
        assert_eq!(sess.read_output_f32("s").await.expect("read"), vec![10.0; 4]);
    }

    /// GPU 门控:回放 == eager 数值对拍(捕获正确性的直接证据)。
    /// 权重块化(捕获期禁 Htod);同卡双 actor,纯函数树无状态污染。
    /// 门控由 `#[owl_test(gpu)]` 注入;设备由 `OWL_TEST_DEVICE`(**BDF**)解析。
    #[owl_test(gpu)]
    async fn gpu_capture_matches_eager() {
        let ordinal = owl_cuda::test_device_ordinal(); // OWL_TEST_DEVICE(BDF)→ ordinal(唯一入口)
        let f_eager = owl_cuda::GpuClient::spawn(owl_cuda::DeviceSelector::Ordinal(ordinal))
            .expect("gpu boot(eager 锚)");
        let mut f_cap = owl_cuda::GpuClient::spawn(owl_cuda::DeviceSelector::Ordinal(ordinal))
            .expect("gpu boot(捕获线)");

        // 权重块化(捕获线用;Block 叶子,物化在捕获窗外)
        let w_bytes: Vec<u8> = (0..16)
            .flat_map(|i| ((i % 5) as f32 - 2.0).to_le_bytes())
            .collect();
        let w_b = eval_ops(TensorOps::from_host(Dtype::F32, vec![4, 4], &w_bytes).step(), &mut f_cap)
            .await
            .expect("W 块化");
        let w_cap = TensorOps::of_block(w_b.id, Dtype::F32, vec![4, 4]);
        let w_eager = TensorOps::from_host(Dtype::F32, vec![4, 4], &w_bytes);

        let desc = |capture: bool| GraphPlanDesc {
            inputs: vec![InputSlot::f32("x", 4), InputSlot::f32("y", 4)],
            outputs: vec![OutputSlot::f32("s", &[1, 4])],
            capture,
            slab_fixed: false,
            debug_probe: false,
            // 本族不加锚定回收(plain 口径:窗内块常驻 —— 历史行为零变化)
            reclaim_capture: false,
            warmup: true,
        };

        // eager 锚
        let w1 = w_eager.clone();
        let (mut s_eager, o_e) = GraphPlan::plan(f_eager, owl_models::env::EnvProvider::default(), None, desc(false), move |sc: &PlanCtx| -> Result<()> {
            let x = sc.input("x")?;
            let y = sc.input("y")?;
            sc.output("s", &x.add(&y).matmul(&w1))
        })
        .await
        .expect("eager plan");
        assert!(matches!(o_e, PlanOutcome::EagerFallback { .. }));

        // 捕获线
        let (mut s_cap, o_c) = GraphPlan::plan(f_cap, owl_models::env::EnvProvider::default(), None, desc(true), move |sc: &PlanCtx| -> Result<()> {
            let x = sc.input("x")?;
            let y = sc.input("y")?;
            sc.output("s", &x.add(&y).matmul(&w_cap))
        })
        .await
        .expect("capture plan");
        assert!(matches!(o_c, PlanOutcome::Captured), "捕获应成功,得 {o_c:?}");

        // 两步异构数据,逐步对拍
        for (step, (x, y)) in [
            (&[1.0f32, 2.0, 3.0, 4.0], &[0.5f32, -1.0, 2.0, 0.25]),
            (&[-2.0f32, 0.0, 7.0, 1.5], &[1.0f32; 4]),
        ]
        .into_iter()
        .enumerate()
        {
            s_eager.step(&[("x", x), ("y", y)]).await.expect("eager step");
            s_cap.step(&[("x", x), ("y", y)]).await.expect("cap step");
            let a = s_eager.read_output_f32("s").await.expect("read eager");
            let b = s_cap.read_output_f32("s").await.expect("read cap");
            assert_eq!(a.len(), b.len());
            for (i, (u, v)) in a.iter().zip(&b).enumerate() {
                assert!((u - v).abs() < 1e-5, "step{step}[{i}]: eager {u} vs 回放 {v}");
            }
        }
    }

    /// **M1 真机门**:`reclaim_capture = true`(窗内 scoped 回收 + slab arena
    /// 复用 + 影子探针定尺)下,捕获/回放仍逐位等于 eager。
    ///
    /// 为什么必须单独测:reclaim 改的是**捕获窗内的块回收** —— 若活性判据错
    /// (把还被图节点引用的块归还并复用),回放会读到被别人覆盖的内存 ⇒
    /// **静默错值**。本测就是这个雷的守门(与 gpu_capture_matches_eager 同形,
    /// 只差 `reclaim_capture` 一位)。
    #[tokio::test]
    async fn gpu_capture_reclaim_matches_eager() {
        let Some(ordinal) = owl_cuda::test_device_ordinal_opt() else {
            owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
            return;
        };
        let f_eager = owl_cuda::GpuClient::spawn(owl_cuda::DeviceSelector::Ordinal(ordinal))
            .expect("gpu boot(eager 锚)");
        let mut f_cap = owl_cuda::GpuClient::spawn(owl_cuda::DeviceSelector::Ordinal(ordinal))
            .expect("gpu boot(捕获线)");

        let w_bytes: Vec<u8> = (0..16)
            .flat_map(|i| ((i % 5) as f32 - 2.0).to_le_bytes())
            .collect();
        let w_b = eval_ops(TensorOps::from_host(Dtype::F32, vec![4, 4], &w_bytes).step(), &mut f_cap)
            .await
            .expect("W 块化");
        let w_cap = TensorOps::of_block(w_b.id, Dtype::F32, vec![4, 4]);
        let w_eager = TensorOps::from_host(Dtype::F32, vec![4, 4], &w_bytes);

        let desc = |capture: bool| GraphPlanDesc {
            inputs: vec![InputSlot::f32("x", 4), InputSlot::f32("y", 4)],
            outputs: vec![OutputSlot::f32("s", &[1, 4])],
            capture,
            slab_fixed: false,
            debug_probe: false,
            reclaim_capture: true, // ← 唯一差别
            warmup: true,
        };

        let w1 = w_eager.clone();
        let (mut s_eager, _) = GraphPlan::plan(
            f_eager,
            owl_models::env::EnvProvider::default(),
            None,
            desc(false),
            move |sc: &PlanCtx| -> Result<()> {
                let x = sc.input("x")?;
                let y = sc.input("y")?;
                sc.output("s", &x.add(&y).matmul(&w1))
            },
        )
        .await
        .expect("eager plan");

        let (mut s_cap, o_c) = GraphPlan::plan(
            f_cap,
            owl_models::env::EnvProvider::default(),
            None,
            desc(true),
            move |sc: &PlanCtx| -> Result<()> {
                let x = sc.input("x")?;
                let y = sc.input("y")?;
                sc.output("s", &x.add(&y).matmul(&w_cap))
            },
        )
        .await
        .expect("capture plan");
        assert!(matches!(o_c, PlanOutcome::Captured), "锚回收档捕获应成功,得 {o_c:?}");

        for (step, (x, y)) in [
            (&[1.0f32, 2.0, 3.0, 4.0], &[0.5f32, -1.0, 2.0, 0.25]),
            (&[-2.0f32, 0.0, 7.0, 1.5], &[1.0f32; 4]),
        ]
        .into_iter()
        .enumerate()
        {
            s_eager.step(&[("x", x), ("y", y)]).await.expect("eager step");
            s_cap.step(&[("x", x), ("y", y)]).await.expect("cap step");
            let a = s_eager.read_output_f32("s").await.expect("read eager");
            let b = s_cap.read_output_f32("s").await.expect("read cap");
            for (i, (u, v)) in a.iter().zip(&b).enumerate() {
                assert!((u - v).abs() < 1e-5, "reclaim step{step}[{i}]: eager {u} vs 回放 {v}");
            }
        }
    }
}

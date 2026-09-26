//! Session 编排面(P0/M1;`docs/arch/session-plan.md` 的新世界实现)。
//!
//! **命名分域(phase3 立项勘定,2026-09-26)**:本模块的 Session =
//! **Capture Session**(图三态编排,底层执行概念);用户层 agent 会话
//! (长连接、工具回灌、跨 step 状态存续)= `engine::agent` 的
//! **AgentSession**,是另一个东西 —— 勿混同(roadmap.local/
//! phase3-agent-session.md §〇)。
//!
//! 动机(session-plan.md 原文):消灭上层手工流程纪律——算子编排闭包
//! 只写一份,执行三态(eager/捕获/回放)由 Session 承载。
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
//! 捕获窗内 Err = 硬错(窗已脏;graph abort 原语挂账 iface)。
//!
//! warmup = 姿势 6 lite:init 数据 eager dry 一遍(声明违约 plan 边界即
//! 暴露;捕获不执行 kernel,状态不被捕获推进)。

use std::cell::RefCell;
use std::collections::HashMap;

use owl_iface::contract::{Bytes, DeviceClient, Dtype, GraphId, ModelError};
use owl_models::interpreters::eval_ops;
use owl_models::ops::{Op, PlanNode};
use owl_models::TensorOps;

type Result<T> = std::result::Result<T, ModelError>;

// ============================================================================
// §1 槽词汇(声明;数据面在 Session)
// ============================================================================

/// 输入槽:命名持久设备缓冲,每步 write_block 原位装填(指针稳定 ——
/// 捕获回放的根基;decode:frontier/positions/slots…)
#[derive(Clone, Debug)]
pub struct InputSlot {
    pub name: &'static str,
    /// 容量(元素;M0.5 步数据须等长,档位 narrow 视图随 batching 引入)
    pub len: usize,
    /// 初始数据(warmup dry 执行用;缺省全零)
    pub init: Vec<f32>,
}

impl InputSlot {
    pub fn f32(name: &'static str, len: usize) -> Self {
        Self { name, len, init: Vec::new() }
    }

    pub fn init(mut self, v: Vec<f32>) -> Self {
        self.init = v;
        self
    }
}

/// 输出槽:闭包登记的声明树,解释器归约后按名收割
#[derive(Clone, Debug)]
pub struct OutputSlot {
    pub name: &'static str,
    /// 声明形状(收割量守卫)
    pub shape: Vec<usize>,
}

impl OutputSlot {
    pub fn f32(name: &'static str, shape: &[usize]) -> Self {
        Self { name, shape: shape.to_vec() }
    }
}

/// 会话描述
#[derive(Default)]
pub struct SessionDesc {
    pub inputs: Vec<InputSlot>,
    pub outputs: Vec<OutputSlot>,
    /// 捕获三态开关(true = 尝试捕获,预检不过自动 EagerFallback)
    pub capture: bool,
}

/// plan 结果(A1.4:预检失败 = EagerFallback,同闭包直发)
#[derive(Clone, Debug)]
pub enum PlanOutcome {
    Captured,
    EagerFallback { reason: String },
}

// ============================================================================
// §2 StepCtx:闭包取输入声明 / 登记输出声明(纯描述;零 await)
// ============================================================================

pub struct StepCtx {
    inputs: HashMap<&'static str, TensorOps>,
    outputs: RefCell<Vec<(&'static str, TensorOps)>>,
}

impl StepCtx {
    /// 输入槽声明(本步设备块的 Block 叶子;克隆共享同块,零拷贝)
    pub fn input(&self, name: &str) -> Result<TensorOps> {
        self.inputs
            .get(name)
            .cloned()
            .ok_or_else(|| ModelError::Msg(format!("step: 未声明输入槽 {name}")))
    }

    /// 输出登记:该声明树即本步输出(解释器归约;块生命周期归会话)
    pub fn output(&self, name: &'static str, t: &TensorOps) -> Result<()> {
        self.outputs.borrow_mut().push((name, t.clone()));
        Ok(())
    }

    /// 取登记(会话在闭包返回后收割;消费即清空)
    fn take_outputs(&self) -> Vec<(&'static str, TensorOps)> {
        self.outputs.borrow_mut().split_off(0)
    }
}

// ============================================================================
// §3 Session:一次声明,三态执行
// ============================================================================

struct InSlotDev {
    name: &'static str,
    len: usize,
    block: Bytes,
}

struct OutVal {
    block: Bytes,
    len: usize,
}

#[derive(Clone, Copy, Debug)]
enum Mode {
    Eager,
    Captured { graph: GraphId },
}

pub struct Session<D: DeviceClient> {
    face: D,
    inputs: Vec<InSlotDev>,
    out_specs: HashMap<&'static str, usize>, // name → 元素数
    last: HashMap<&'static str, OutVal>,
    mode: Mode,
    forward: Box<dyn Fn(&StepCtx) -> Result<()>>,
}

impl<D: DeviceClient> Session<D> {
    /// 声明 + warmup + (可选)捕获。流程纪律全部在此闭合:
    /// 姿势 6 warmup 门禁(lite)/ 两道捕获预检 / EagerFallback。
    pub async fn plan(
        mut face: D,
        desc: SessionDesc,
        forward: impl Fn(&StepCtx) -> Result<()> + 'static,
    ) -> Result<(Self, PlanOutcome)> {
        // 槽设备面:持久块,htod(init 缺省补零);此后只 write_block 原位
        let mut ins: Vec<InSlotDev> = Vec::new();
        for i in desc.inputs {
            let mut init = i.init;
            init.resize(i.len, 0.0);
            let block = face.htod(Dtype::F32, &vec![i.len], &f32b(&init)).await?;
            ins.push(InSlotDev { name: i.name, len: i.len, block });
        }

        let mut sess = Self {
            face,
            inputs: ins,
            out_specs: desc
                .outputs
                .iter()
                .map(|o| (o.name, o.shape.iter().product::<usize>()))
                .collect(),
            last: HashMap::new(),
            mode: Mode::Eager,
            forward: Box::new(forward),
        };

        // ── warmup(姿势 6 lite):init 数据全链 eager dry 一遍 ──
        sess.fill_slots(&[]).await?;
        sess.eval_current().await?;
        sess.face.sync().await?;
        sess.last.clear();

        // ── 捕获(两道预检;窗内硬错直接上抛)──
        let outcome = if desc.capture { sess.try_capture().await? } else { fallback("未请求捕获") };
        Ok((sess, outcome))
    }

    /// 每步执行:装填输入(write_block 原位)→ 回放或 eager 直发。
    /// 未提到的输入槽保持上步内容(典型:常量槽)。
    pub async fn step(&mut self, inputs: &[(&str, &[f32])]) -> Result<()> {
        self.fill_slots(inputs).await?;
        match self.mode {
            Mode::Captured { graph } => self.face.graph_launch(graph).await?,
            Mode::Eager => self.eval_current().await?,
        }
        Ok(())
    }

    /// 输出收割(读语义;块 = 最近一次 step/捕获的归约产物)
    pub async fn read_output_f32(&mut self, name: &str) -> Result<Vec<f32>> {
        let out = self
            .last
            .get(name)
            .ok_or_else(|| ModelError::Msg(format!("read_output: 无输出 {name}(先 step)")))?;
        let mut buf = vec![0u8; out.len * 4];
        self.face.dtoh(&out.block, &mut buf).await?;
        Ok(buf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    /// face 访问器(crate 内编排层用:turn 切换时的状态重置等
    /// 槽外设备操作;对外仍零暴露 —— face 不出 engine crate)
    pub(crate) fn face_mut(&mut self) -> &mut D {
        &mut self.face
    }

    // ------------------------------------------------------------------
    // 内部:装填 / 声明 / 归约 / 捕获
    // ------------------------------------------------------------------

    /// 槽装填:write_block 原位(指针稳定 —— 回放根基)
    async fn fill_slots(&mut self, fills: &[(&str, &[f32])]) -> Result<()> {
        for (name, data) in fills {
            let slot = self
                .inputs
                .iter_mut()
                .find(|s| s.name == *name)
                .ok_or_else(|| ModelError::Msg(format!("step: 未声明输入槽 {name}")))?;
            if data.len() != slot.len {
                return Err(ModelError::Msg(format!(
                    "step: 槽 {name} 装填 {} 元素 ≠ 容量 {}",
                    data.len(),
                    slot.len
                )));
            }
            self.face.write_block_f32(&slot.block, 0, data).await?;
        }
        Ok(())
    }

    /// 闭包声明 + 守卫:返回登记输出(名须在 out_specs)
    fn declare_outputs(&self) -> Result<Vec<(&'static str, TensorOps)>> {
        let mut inputs = HashMap::new();
        for s in &self.inputs {
            inputs.insert(
                s.name,
                TensorOps::of_block(s.block.id, Dtype::F32, vec![1, s.len]),
            );
        }
        let ctx = StepCtx { inputs, outputs: RefCell::new(Vec::new()) };
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
    async fn eval_current(&mut self) -> Result<()> {
        for (name, tree) in self.declare_outputs()? {
            let want = self.out_specs[&name];
            let block = eval_ops(tree.step(), &mut self.face).await?;
            if block.len != 0 && block.len != want {
                return Err(ModelError::Msg(format!(
                    "step: 输出 {name} 归约 {} 元素 ≠ 声明 {want}",
                    block.len
                )));
            }
            self.last.insert(name, OutVal { block, len: want });
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

        // 捕获窗:launch 入图 / alloc 走 slab;输出块 id 捕获期即定
        let mut blocks: Vec<(&'static str, Bytes, usize)> = Vec::new();
        let mut win_err: Option<ModelError> = None;
        for (name, tree) in outs {
            let want = self.out_specs[&name];
            match eval_ops(tree.step(), &mut self.face).await {
                Ok(b) if b.len == 0 || b.len == want => blocks.push((name, b, want)),
                Ok(b) => {
                    win_err = Some(ModelError::Msg(format!(
                        "capture: 输出 {name} 归约 {} 元素 ≠ 声明 {want}",
                        b.len
                    )));
                    break;
                }
                Err(e) => {
                    win_err = Some(e);
                    break;
                }
            }
        }
        let gid = self.face.graph_end().await;

        match (win_err, gid) {
            (None, Ok(graph)) => {
                self.mode = Mode::Captured { graph };
                self.last = blocks
                    .into_iter()
                    .map(|(name, block, len)| (name, OutVal { block, len }))
                    .collect();
                Ok(PlanOutcome::Captured)
            }
            (Some(e), _) => Err(e), // 窗已脏:不可静默降级(挂账:graph abort)
            (None, Err(e)) => Err(ModelError::Msg(format!("capture: graph_end 失败 {e}"))),
        }
    }
}

fn fallback(reason: &str) -> PlanOutcome {
    PlanOutcome::EagerFallback { reason: reason.to_string() }
}

fn f32b(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

// ============================================================================
// §4 测试(CPU:闭环 + 降级;GPU 门控:回放 == eager 数值对拍)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use owl_cpu::CpuFace;

    fn ones_w() -> TensorOps {
        let bytes: Vec<u8> = vec![1.0f32; 16].iter().flat_map(|f| f.to_le_bytes()).collect();
        TensorOps::from_host(Dtype::F32, vec![4, 4], &bytes)
    }

    #[tokio::test]
    async fn eager_cpu_end_to_end() {
        // out = (x + y) × W,W 全 1 → out = (x+y) 的行和
        let w = ones_w();
        let face = CpuFace::new();
        let desc = SessionDesc {
            inputs: vec![
                InputSlot::f32("x", 4),
                InputSlot::f32("y", 4).init(vec![10.0, 20.0, 30.0, 40.0]),
            ],
            outputs: vec![OutputSlot::f32("s", &[1, 4])],
            capture: false,
        };
        let (mut sess, outcome) = Session::plan(
            face,
            desc,
            move |sc: &StepCtx| -> Result<()> {
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

    #[tokio::test]
    async fn warmup_catches_declaration_poison() {
        // 闭包带形状违约([4] + [3] → 毒):plan 边界暴露,不污染业务步
        let face = CpuFace::new();
        let desc = SessionDesc {
            inputs: vec![InputSlot::f32("a", 4), InputSlot::f32("b", 3)],
            outputs: vec![OutputSlot::f32("s", &[1, 4])],
            capture: true,
        };
        let r = Session::plan(face, desc, |sc: &StepCtx| -> Result<()> {
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
        let w_bytes: Vec<u8> = vec![1.0f32; 16].iter().flat_map(|f| f.to_le_bytes()).collect();
        let w_b = eval_ops(TensorOps::from_host(Dtype::F32, vec![4, 4], &w_bytes).step(), &mut face)
            .await
            .expect("W 块化");
        let w = TensorOps::of_block(w_b.id, Dtype::F32, vec![4, 4]);
        let desc = SessionDesc {
            inputs: vec![InputSlot::f32("x", 4)],
            outputs: vec![OutputSlot::f32("s", &[1, 4])],
            capture: true,
        };
        let (mut sess, outcome) = Session::plan(
            face,
            desc,
            move |sc: &StepCtx| -> Result<()> {
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
    #[tokio::test]
    async fn gpu_capture_matches_eager() {
        let Some(ordinal) =
            std::env::var("OWL_TEST_DEVICE").ok().and_then(|v| v.parse::<usize>().ok())
        else {
            eprintln!("skip: OWL_TEST_DEVICE 未设");
            return;
        };
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

        let desc = |capture: bool| SessionDesc {
            inputs: vec![InputSlot::f32("x", 4), InputSlot::f32("y", 4)],
            outputs: vec![OutputSlot::f32("s", &[1, 4])],
            capture,
        };

        // eager 锚
        let w1 = w_eager.clone();
        let (mut s_eager, o_e) = Session::plan(f_eager, desc(false), move |sc: &StepCtx| -> Result<()> {
            let x = sc.input("x")?;
            let y = sc.input("y")?;
            sc.output("s", &x.add(&y).matmul(&w1))
        })
        .await
        .expect("eager plan");
        assert!(matches!(o_e, PlanOutcome::EagerFallback { .. }));

        // 捕获线
        let (mut s_cap, o_c) = Session::plan(f_cap, desc(true), move |sc: &StepCtx| -> Result<()> {
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
}

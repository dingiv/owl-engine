//! Session 编排面(P0;`docs/arch/session-plan.md` 的新世界实现)。
//!
//! 动机(session-plan.md 原文):消灭上层手工流程纪律——算子编排闭包
//! 只写一份,执行三态(eager/捕获/回放)由 Session 承载。
//!
//! **移植自 engine-bak/session.rs(464 行)的流程纪律**,数据面全面换代:
//! ETensor/KernelCtx/CudaPool/捕获会话 → TensorOps 声明 + DeviceClient
//! (唯一设备标准)。M0 闭环 = **eager 态**:槽装填(htod 重绑)+ 闭包
//! 纯描述 + 计算解释器执行;捕获/回放三态(graph_begin/end/launch +
//! 姿势 6 门禁 + A1.4 预检 + 租约)M1 接线,槽词汇/闭包形状不变。
//!
//! 职责分界(与 models 解释器的分工):闭包 = 纯声明(零 await 零 ?);
//! 计算执行 = models::interpreters::eval_ops;本模块 = **编排**——槽的
//! 生命周期(分配/装填/重绑)、闭包驱动、输出登记与收割、流程纪律。

use std::cell::RefCell;
use std::collections::HashMap;

use owl_iface::contract::{Bytes, DeviceClient, Dtype, ModelError};
use owl_models::interpreters::eval_ops;
use owl_models::TensorOps;

type Result<T> = std::result::Result<T, ModelError>;

// ============================================================================
// §1 槽词汇(声明;数据面在 Session)
// ============================================================================

/// 输入槽:每步 H2D 装填的命名设备缓冲(decode:frontier/positions/slots…)
#[derive(Clone, Debug)]
pub struct InputSlot {
    pub name: &'static str,
    /// 容量(元素;M0 步数据须等长,档位 narrow 视图 M1 随捕获引入)
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

/// 会话描述(捕获档位/scratch/预算字段 M1 随三态引入,词汇先行)
#[derive(Default)]
pub struct SessionDesc {
    pub inputs: Vec<InputSlot>,
    pub outputs: Vec<OutputSlot>,
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
// §3 Session:一次声明,三态执行(M0 = eager 态)
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

pub struct Session<D: DeviceClient> {
    face: D,
    inputs: Vec<InSlotDev>,
    out_specs: HashMap<&'static str, usize>, // name → 元素数
    last: HashMap<&'static str, OutVal>,
    forward: Box<dyn Fn(&StepCtx) -> Result<()>>,
}

impl<D: DeviceClient> Session<D> {
    /// 声明 + warmup(M0 态;捕获/回放与 A1.4 预检 M1 在本函数尾部接线,
    /// 槽词汇与闭包形状不变)。
    ///
    /// warmup = 姿势 6 的 eager dry 执行:init 数据走一遍闭包 + 输出归约
    /// —— 声明违约(毒值/形状)在 plan 边界即暴露,不污染首个业务步。
    pub async fn plan(
        mut face: D,
        desc: SessionDesc,
        forward: impl Fn(&StepCtx) -> Result<()> + 'static,
    ) -> Result<Self> {
        // 槽设备面:输入 htod(init 缺省补零);输出只留规格(声明树产出块)
        let mut ins: Vec<InSlotDev> = Vec::new();
        for i in desc.inputs {
            let mut init = i.init;
            init.resize(i.len, 0.0);
            let block = face
                .htod(Dtype::F32, &vec![i.len], &f32b(&init))
                .await?;
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
            forward: Box::new(forward),
        };

        // ── warmup(姿势 6 lite):init 数据全链 dry 一遍 ──
        sess.run_step(&[]).await?;
        sess.face.sync().await?;
        sess.last.clear();
        Ok(sess)
    }

    /// 每步执行:装填输入(htod 重绑;M0 新块,M1 捕获态改原块填装)→
    /// 闭包声明 → 输出归约。未提到的输入槽保持上步块(典型:常量槽)。
    pub async fn step(&mut self, inputs: &[(&str, &[f32])]) -> Result<()> {
        self.run_step(inputs).await
    }

    /// 输出收割(读语义;块 = 最近一次 step 的归约产物)
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

    /// 收割原始字节(非 f32 输出场景预留;M0 输出恒 f32)
    #[allow(dead_code)]
    pub async fn read_output_bytes(&mut self, name: &str) -> Result<Vec<u8>> {
        let out = self
            .last
            .get(name)
            .ok_or_else(|| ModelError::Msg(format!("read_output: 无输出 {name}")))?;
        let mut buf = vec![0u8; out.len * 4];
        self.face.dtoh(&out.block, &mut buf).await?;
        Ok(buf)
    }

    /// 步机制( warmup 与 step 共用;差异只在数据来源)
    async fn run_step(&mut self, fills: &[(&str, &[f32])]) -> Result<()> {
        // 1. 装填:htod 新块重绑(eager 态;捕获态 M1 = 原块 write_block,
        //    指针稳定契约 —— iface「块只增不减,指针稳定」)
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
            slot.block = self
                .face
                .htod(Dtype::F32, &vec![slot.len], &f32b(data))
                .await?;
        }

        // 2. 闭包声明(纯描述;输入槽 = 当前块 Block 叶子,行向量形态
        //    [1,len] —— 与 models [T,k] 行约定一致)
        let mut inputs = HashMap::new();
        for s in &self.inputs {
            inputs.insert(
                s.name,
                TensorOps::of_block(s.block.id, Dtype::F32, vec![1, s.len]),
            );
        }
        let ctx = StepCtx { inputs, outputs: RefCell::new(Vec::new()) };
        (self.forward)(&ctx)?;

        // 3. 输出归约(计算解释器;登记名须在 out_specs 且元素数守卫)
        for (name, tree) in ctx.take_outputs() {
            let want = *self
                .out_specs
                .get(name)
                .ok_or_else(|| ModelError::Msg(format!("step: 未声明输出槽 {name}")))?;
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
}

fn f32b(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

// ============================================================================
// §4 测试(CPU face 即测即用;GPU 捕获态 M1 另批)
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
            outputs: vec![OutputSlot::f32("s", &[1, 4]), OutputSlot::f32("sum", &[1, 4])],
        };
        let mut sess = Session::plan(
            face,
            desc,
            move |sc: &StepCtx| -> Result<()> {
                let x = sc.input("x")?;
                let y = sc.input("y")?;
                let s = x.add(&y);
                sc.output("sum", &s)?;
                sc.output("s", &s.matmul(&w))?;
                Ok(())
            },
        )
        .await
        .expect("plan(warmup 门禁)");

        // 业务步:y 未装填 → 保持 init 块(常量槽语义)
        sess.step(&[("x", &[1.0, 2.0, 3.0, 4.0])]).await.expect("step");
        let sum = sess.read_output_f32("sum").await.expect("read sum");
        eprintln!("[probe] sum = {sum:?}");
        let s = sess.read_output_f32("s").await.expect("read");
        assert_eq!(s, vec![110.0; 4], "(x+y)=[11,22,33,44] × 全 1 阵 = 行和 110");

        // 换 y:两槽都装填
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
}

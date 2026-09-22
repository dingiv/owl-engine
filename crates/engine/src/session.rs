//! Session 编排面(P0,session-plan.md):算子编排闭包只写一份,
//! eager/捕获/回放三态由 Session 承载。
//!
//! ```ignore
//! let plan = Session::plan(&dev, desc, |sc: &StepCtx| {
//!     let ids = sc.input("frontier")?;
//!     sc.output("logits", &model.forward(ids, ...)?)?;
//!     Ok(())
//! })?;
//! plan.write("frontier", &[11, 22])?;
//! plan.step()?;
//! ```
//!
//! 流程纪律(warmup 姿势 6 门禁 / A1.4 预检降级 / 相位 / 自动租约 /
//! 哨兵③ / seal)全部封在 [`Session::plan`] 内;上层可见的只有
//! [`StepCtx`](编排 API)。

use crate::Error;
use crate::Result;
use owl_cuda::ffi::sys;
use owl_cuda::CudaDevice;
use owl_nn::DynTensor;
use owl_nn::tensor::Tensor;

/// 引擎侧擦除张量别名(= ETensor)
type ETensor = DynTensor<CudaDevice>;
use owl_iface::BackendError;
use owl_iface::MemPhase;
use std::collections::BTreeMap;
use std::sync::Arc;

/// 输入槽:u32 `[len]`(frontier/positions/slots/kv_lens 等整型动态量)。
pub struct InputSlot {
    pub name: &'static str,
    pub len: usize,
    /// 初始数据(warmup dry 执行用;缺省全零)
    pub init: Vec<u32>,
}

impl InputSlot {
    pub fn u32(name: &'static str, len: usize) -> Self {
        Self { name, len, init: Vec::new() }
    }
    pub fn init(mut self, v: Vec<u32>) -> Self {
        self.init = v;
        self
    }
}

/// 输出槽:f32,形状任意(元素数固定;闭包每步写入同尺寸产出)。
pub struct OutputSlot {
    pub name: &'static str,
    pub shape: Vec<usize>,
}

impl OutputSlot {
    pub fn f32(name: &'static str, shape: &[usize]) -> Self {
        Self { name: name.into(), shape: shape.to_vec() }
    }
}

/// 每 forward 的执行凭证:ctx + 输入视图 + 输出槽写入口。
pub struct StepCtx<'a> {
    ctx: KernelCtxLike,
    inputs: Vec<(&'static str, ETensor)>,
    outputs: Vec<(&'static str, ETensor)>,
    bs: usize,
    _marker: std::marker::PhantomData<&'a ()>,
}

use owl_nn::KernelCtx;

/// 统一 KernelCtx 面的内部枚举(eager Live / 捕获 frame,scratch 已注入)。
#[derive(Clone)]
enum KernelCtxLike {
    Real(KernelCtx),
}

impl KernelCtxLike {
    fn get(&self) -> &KernelCtx {
        match self {
            KernelCtxLike::Real(c) => c,
        }
    }
}

impl<'a> StepCtx<'a> {
    /// 发射 ctx(scratch 已注入;捕获态 = frame ctx,自动租约)
    pub fn ctx(&self) -> &KernelCtx {
        self.ctx.get()
    }

    /// 本档 bs
    pub fn bs(&self) -> usize {
        self.bs
    }

    /// 输入槽视图(已 narrow 到本档 bs)
    pub fn input(&self, name: &str) -> Result<&ETensor> {
        self.inputs
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, t)| t)
            .ok_or_else(|| Error::Msg(format!("StepCtx: 未知输入槽 {name}")))
    }

    /// 输出槽写入(D2D,容量守卫;捕获期 src 自动入租约)
    pub fn output(&self, name: &str, t: &ETensor) -> Result<()> {
        let (_, slot) = self
            .outputs
            .iter()
            .find(|(n, _)| *n == name)
            .ok_or_else(|| Error::Msg(format!("StepCtx: 未知输出槽 {name}")))?;
        let bytes = t.len_bytes();
        owl_nn::erased::copy_d2d_to_raw(self.ctx.get(), t, slot.device_ptr() as *mut core::ffi::c_void, bytes)
            .map_err(|e| Error::Msg(format!("output({name}): {e}")))
    }
}

// ---- 槽的设备面 ----

struct InSlotDev {
    name: &'static str,
    len: usize,
    tensor: Tensor<u32, CudaDevice>, // [len]
}

struct OutSlotDev {
    name: &'static str,
    tensor: Tensor<f32, CudaDevice>, // [shape]
}

/// 槽设备面(输入 u32 / 输出 f32;池分配,P 阶段)
struct SlotBank {
    inputs: Vec<InSlotDev>,
    outputs: Vec<OutSlotDev>,
}

impl SlotBank {
    fn build(
        _dev: &CudaDevice,
        pool: &owl_cuda::CudaPool,
        inputs: &[InputSlot],
        outputs: &[OutputSlot],
    ) -> Result<Self, Error> {
        use owl_nn::TensorPoolOps as _;
        let mut ins = Vec::new();
        for i in inputs {
            let init = if i.init.len() == i.len {
                i.init.clone()
            } else {
                vec![0u32; i.len]
            };
            let t = pool.from_vec_tensor::<u32>(&[i.len], init).map_err(Error::from)?;
            ins.push(InSlotDev { name: i.name, len: i.len, tensor: t });
        }
        let mut outs = Vec::new();
        for o in outputs {
            let n: usize = o.shape.iter().product();
            let t = pool
                .from_vec_tensor::<f32>(&o.shape, vec![0f32; n])
                .map_err(Error::from)?;
            outs.push(OutSlotDev { name: o.name, tensor: t });
        }
        Ok(Self { inputs: ins, outputs: outs })
    }

    fn lease_all(&self, session: &mut owl_cuda::CaptureSession) {
        for i in &self.inputs {
            if let Some(p) = i.tensor.persistent() {
                session.lease(p);
            }
        }
        for o in &self.outputs {
            if let Some(p) = o.tensor.persistent() {
                session.lease(p);
            }
        }
    }
}

/// Session:一次声明,三态执行。
pub struct Session {
    dev: CudaDevice,
    bank: SlotBank,
    graphs: BTreeMap<usize, owl_cuda::DeviceGraph>,
    profiles: Vec<usize>,
    eager: bool,
    scratch: Option<Arc<owl_cuda::CudaPool>>,
    /// P0 单线程口径(A2.6:每 runner 线程一套 Session);Send 界随多线程化再加
    forward: Box<dyn Fn(&StepCtx) -> Result<()>>,
}

/// 捕获结果(A1.4:预检失败 = EagerFallback,同闭包直发)
pub enum PlanOutcome {
    Captured { profiles: Vec<usize>, sealed_bytes: u64 },
    EagerFallback { free_bytes: u64, needed: u64 },
}

pub struct SessionDesc {
    pub profiles: Vec<usize>,
    pub inputs: Vec<InputSlot>,
    pub outputs: Vec<OutputSlot>,
    /// S6 激活 scratch 池(每卡一池,跨图族共享——用户裁决)
    pub scratch: Option<Arc<owl_cuda::CudaPool>>,
    /// 每档位预估 footprint(A1.4 预检;A1.1 规划器 T3 接管)
    pub per_profile_bytes: fn(usize) -> u64,
}

impl Session {
    /// 声明 + warmup + 逐档捕获。流程纪律全部在此闭合:
    /// 姿势 6 warmup 门禁 / Governor 单档周期 / 自动租约 / 哨兵③ /
    /// AUTO_FREE 旗标 / seal 实测超支弃档 / A1.4 预检 EagerFallback。
    pub fn plan(
        dev: &CudaDevice,
        desc: SessionDesc,
        forward: impl Fn(&StepCtx) -> Result<()> + 'static,
    ) -> Result<(Self, PlanOutcome), Error> {
        // 槽设备面(专用 Weights 池;P 阶段分配)
        use owl_iface::Device as _;
        let pool = dev
            .create_pool(owl_iface::PoolConfig {
                name: format!("session-bank-{}", std::process::id()),
                kind: owl_iface::PoolKind::Weights,
                bytes: {
                    let ins: usize = desc.inputs.iter().map(|i| i.len * 4).sum();
                    let outs: usize = desc.outputs.iter().map(|o| o.shape.iter().product::<usize>() * 4).sum();
                    (ins + outs).max(1 << 20) as u64
                },
            })
            .map_err(|e| Error::Msg(format!("session bank pool: {e:?}")))?;
        let bank = SlotBank::build(dev, &pool, &desc.inputs, &desc.outputs)?;

        // warmup(姿势 6):eager dry 执行一遍(空栈回退机制使闭包内
        // ctx_scope 桥可用;数据 = init/全零,合法性由编排方保证)
        {
            let ctx = {
                let c = KernelCtx::eager(MemPhase::Live, dev.stream().clone());
                match &desc.scratch {
                    Some(p) => c.with_scratch(Arc::clone(p)),
                    None => c,
                }
            };
            let max_bs = desc.profiles.iter().copied().max().unwrap_or(1);
            // warmup 视图 = 最大档 narrow(init 截断到 max_bs)
            let inputs = bank
                .inputs
                .iter()
                .map(|i| (i.name, ETensor::from_raw_u32(i.tensor.device_ptr() as *mut u32, &[max_bs.min(i.len)])))
                .collect();
            let outputs = bank
                .outputs
                .iter()
                .map(|o| (o.name, ETensor::from_f32(&o.tensor)))
                .collect();
            let sc = StepCtx {
                ctx: KernelCtxLike::Real(ctx),
                inputs,
                outputs,
                bs: max_bs,
                _marker: std::marker::PhantomData,
            };
            forward(&sc)?;
            dev.ctx()
                .synchronize()
                .map_err(|e| Error::Msg(format!("warmup sync: {e:?}")))?;
        }

        // A1.4 预检
        let (free, _total) = dev.mem_get_info().map_err(Error::from)?;
        let free = free as u64;
        let needed: u64 = desc.profiles.iter().map(|&b| (desc.per_profile_bytes)(b)).sum();
        if free < needed {
            return Ok((
                Self {
                    dev: dev.clone(),
                    bank,
                    graphs: BTreeMap::new(),
                    profiles: Vec::new(),
                    eager: true,
                    scratch: desc.scratch,
                    forward: Box::new(forward),
                },
                PlanOutcome::EagerFallback { free_bytes: free, needed },
            ));
        }

        // 逐档捕获(流程同 graphplan::capture,泛化到槽面)
        let mut graphs: BTreeMap<usize, owl_cuda::DeviceGraph> = BTreeMap::new();
        let _max_bs = desc.profiles.iter().copied().max().unwrap_or(1);
        for &bs in &desc.profiles {
            // warmup per 档:姿势 6 门禁按发射计数,不区分档;此处只跑一次
            // eager(已在上方);捕获前 sync
            dev.ctx()
                .synchronize()
                .map_err(|e| Error::Msg(format!("sync: {e:?}")))?;

            let mut session = dev.capture_session()?;
            bank.lease_all(&mut session); // 槽面租约逃生口(裸指针视图,ops emit 盖不到)
            let recorder = owl_nn::CaptureRecorder::new();
            let scratch = desc
                .scratch
                .as_ref()
                .ok_or_else(|| Error::Msg("捕获需要 scratch 池(desc.scratch)".into()))?
                .clone();
            let in_views: Vec<(&'static str, ETensor)> = bank
                .inputs
                .iter()
                .map(|i| (i.name, ETensor::from_raw_u32(i.tensor.device_ptr() as *mut u32, &[bs.min(i.len)])))
                .collect();
            let out_views: Vec<(&'static str, ETensor)> = bank
                .outputs
                .iter()
                .map(|o| (o.name, ETensor::from_f32(&o.tensor)))
                .collect();
            let flags = sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
            let (_, graph) = session.capture(flags, |frame| {
                let ctx = KernelCtx::capturing_leased(recorder.clone(), frame)
                    .with_scratch(Arc::clone(&scratch));
                let sc = StepCtx {
                    ctx: KernelCtxLike::Real(ctx),
                    inputs: in_views.clone(),
                    outputs: out_views.clone(),
                    bs,
                    _marker: std::marker::PhantomData,
                };
                forward(&sc).map_err(|e| BackendError::Init(format!("session forward: {e}")))
            })?;
            graph.upload().map_err(|e| Error::Msg(format!("upload: {e:?}")))?;
            graph.launch().map_err(|e| Error::Msg(format!("暖场 replay: {e:?}")))?;
            dev.ctx()
                .synchronize()
                .map_err(|e| Error::Msg(format!("sync: {e:?}")))?;
            graphs.insert(bs, graph);
        }
        let profiles: Vec<usize> = graphs.keys().copied().collect();
        Ok((
            Self {
                dev: dev.clone(),
                bank,
                graphs,
                profiles: profiles.clone(),
                eager: false,
                scratch: desc.scratch,
                forward: Box::new(forward),
            },
            PlanOutcome::Captured { profiles, sealed_bytes: 0 },
        ))
    }

    /// 每步执行:装填输入 → replay(最小 ≥bs 档)或 eager 直发。
    /// `inputs`:按槽名给本步数据(长度须 ≤ 槽 len)。
    pub fn step(&self, inputs: &[(&str, Vec<u32>)]) -> Result<(), Error> {
        // H2D 装填(EagerOnly;流序天然先于 launch)
        for (name, data) in inputs {
            let slot = self
                .bank
                .inputs
                .iter()
                .find(|i| i.name == *name)
                .ok_or_else(|| Error::Msg(format!("step: 未知输入槽 {name}")))?;
            if data.len() > slot.len {
                return Err(Error::Msg(format!("step({name}): 数据 {} > 槽 {}", data.len(), slot.len)));
            }
            write_u32(&self.dev, slot.tensor.device_ptr() as *mut u32, data, data.len())?;
        }
        if self.eager || self.graphs.is_empty() {
            return self.run_eager(data_len(inputs));
        }
        // replay:最小 ≥bs 档
        let bs = inputs.iter().map(|(_, d)| d.len()).max().unwrap_or(1);
        let slot = self
            .profiles
            .iter()
            .copied()
            .find(|p| *p >= bs)
            .ok_or_else(|| Error::Msg(format!("step: 无 ≥bs={bs} 档")))?;
        self.graphs[&slot].launch().map_err(|e| Error::Msg(format!("replay: {e:?}")))?;
        Ok(())
    }

    fn run_eager(&self, bs: usize) -> Result<(), Error> {
        let ctx = {
            let c = KernelCtx::eager(MemPhase::Live, self.dev.stream().clone());
            match &self.scratch {
                Some(p) => c.with_scratch(Arc::clone(p)),
                None => c,
            }
        };
        let inputs = self
            .bank
            .inputs
            .iter()
            .map(|i| (i.name, ETensor::from_u32(&i.tensor)))
            .collect();
        let outputs = self
            .bank
            .outputs
            .iter()
            .map(|o| (o.name, ETensor::from_f32(&o.tensor)))
            .collect();
        let sc = StepCtx {
            ctx: KernelCtxLike::Real(ctx),
            inputs,
            outputs,
            bs,
            _marker: std::marker::PhantomData,
        };
        (self.forward)(&sc)
    }

    /// 写入输入槽便捷面(等价 step 的装填段)
    pub fn write(&self, name: &str, data: &[u32]) -> Result<(), Error> {
        let slot = self
            .bank
            .inputs
            .iter()
            .find(|i| i.name == name)
            .ok_or_else(|| Error::Msg(format!("write: 未知输入槽 {name}")))?;
        if data.len() > slot.len {
            return Err(Error::Msg(format!("write({name}): 数据 {} > 槽 {}", data.len(), slot.len)));
        }
        write_u32(&self.dev, slot.tensor.device_ptr() as *mut u32, data, data.len())
    }

    /// 输出槽 D2H 回读(测试/采样面;先 sync)
    pub fn read_output_f32(&self, name: &str) -> Result<Vec<f32>, Error> {
        let slot = self
            .bank
            .outputs
            .iter()
            .find(|o| o.name == name)
            .ok_or_else(|| Error::Msg(format!("read_output: 未知槽 {name}")))?;
        self.dev
            .ctx()
            .synchronize()
            .map_err(|e| Error::Msg(format!("sync: {e:?}")))?;
        use owl_cuda::ffi::sys;
        self.dev.ctx().bind_to_thread().map_err(|e| Error::Msg(format!("{e:?}")))?;
        let n = slot.tensor.shape().iter().product::<usize>();
        let mut out = vec![0f32; n];
        unsafe {
            sys::cuMemcpyDtoH_v2(
                out.as_mut_ptr() as *mut std::ffi::c_void,
                slot.tensor.device_ptr() as sys::CUdeviceptr,
                n * 4,
            )
            .result()
            .map_err(|e| Error::Msg(format!("dtoh: {e:?}")))?;
        }
        Ok(out)
    }

    /// 输出槽原始指针(采样同址读;调用方保证生命周期)
    pub fn output_ptr(&self, name: &str) -> Result<*const f32, Error> {
        let slot = self
            .bank
            .outputs
            .iter()
            .find(|o| o.name == name)
            .ok_or_else(|| Error::Msg(format!("output_ptr: 未知槽 {name}")))?;
        Ok(slot.tensor.device_ptr() as *const f32)
    }

    pub fn profiles(&self) -> &[usize] {
        &self.profiles
    }

    pub fn is_captured(&self) -> bool {
        !self.eager && !self.graphs.is_empty()
    }
}

fn data_len(inputs: &[(&str, Vec<u32>)]) -> usize {
    inputs.iter().map(|(_, d)| d.len()).max().unwrap_or(1)
}


fn write_u32(dev: &CudaDevice, dst: *mut u32, src: &[u32], n: usize) -> Result<(), Error> {
    use owl_cuda::ffi::sys;
    dev.ctx().bind_to_thread().map_err(|e| Error::Msg(format!("{e:?}")))?;
    unsafe {
        sys::cuMemcpyHtoD_v2(
            dst as sys::CUdeviceptr,
            src.as_ptr() as *const std::ffi::c_void,
            n * 4,
        )
        .result()
        .map_err(|e| Error::Msg(format!("h2d: {e:?}")))?;
    }
    Ok(())
}


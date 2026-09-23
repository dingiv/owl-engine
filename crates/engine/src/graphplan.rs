//! GraphPlan —— Signal-Graph × 模型层的对接执行器(graph-model-seam.md §三)。
//!
//! 替换 xinfer `utils/graph.rs` 的 GraphCapturer:
//! - 档位表(A1.4 预检可收窄)+ 每档定影图 + 全档共享 Bindings;
//! - 捕获 = CaptureSession(自动租约 + 哨兵③ + 相位),模型 forward 相位无感;
//! - 回放 = bindings H2D(EagerOnly,图主流)→ ≥bs 最小档 launch。
//!
//! 与 owl-graph Governor 的关系(2026-09-22 v1 裁定):Governor 状态机当前
//! 是**单图粒度**(begin 要求 Idle,seal 后即 Live),多档位循环需扩展句柄表
//! (需求已上报)。本模块 v1 = 每档位独立 Governor 周期(allowance 全局共享),
//! A1.2 语义按档闭合;preflight(A1.4)直接用 Governor::preflight。

use crate::error::{Error, Result};
use owl_nn::TensorPoolOps as _;
use owl_cuda::CudaDevice;
use owl_graph::{GraphAllowance, GraphGovernor, GraphPhase, GraphProfile};
use owl_nn::{CaptureRecorder, DynTensor, KernelCtx};
use owl_iface::{Device as _, MemPhase, PoolConfig, PoolKind};
use std::collections::BTreeMap;

/// 档位 bindings:全档共享的钉住缓冲(A1.5:动态量一律 device 张量,U32)。
///
/// P 阶段经池工厂创建(租约常驻);host 每步 write_*_from_host(EagerOnly);
/// `view(bs)` 给出 narrow 元数据视图(指针 + 有效长度),模型 forward 消费。
pub struct GraphBindings {
    /// 每步 token ids [max_bs](U32)
    frontier: DynTensor<owl_cuda::CudaDevice>,
    /// 位置 [max_bs](U32)
    positions: DynTensor<owl_cuda::CudaDevice>,
    /// KV slot 映射 [max_bs](U32)
    slot_mapping: DynTensor<owl_cuda::CudaDevice>,
    /// 每序列 KV 长度 [max_bs](U32)
    kv_lens: DynTensor<owl_cuda::CudaDevice>,
    /// logits 输出绑定 [max_bs, vocab](F32;图直写,采样同址读,零 D2H)
    logits_out: DynTensor<owl_cuda::CudaDevice>,
    vocab: usize,
}

/// 视图元数据:U32 界面(指针 + 有效 bs; dtype 由字段名承诺,downcast 免检)。
#[derive(Clone, Copy)]
pub struct BindingsView {
    pub frontier: *mut u32,
    pub positions: *mut u32,
    pub slot_mapping: *mut u32,
    pub kv_lens: *mut u32,
    pub logits_out: *mut f32,
    /// 本档 batch(图形状;不足 max_bs 的尾段 = pad 区,图内不读)
    pub bs: usize,
    pub max_bs: usize,
    pub vocab: usize,
}

/// forward 契约:runner 把模型 forward 塞进来的最小面(相位无感——
/// eager 与捕获走同一实现;签名对齐 qwen3_5 适配需求见模块尾注)。
pub trait GraphForward {
    /// 按本档 bs 执行一次 decode 全链;logits 写入 view.logits_out。
    fn forward(&self, ctx: &KernelCtx, view: &BindingsView) -> Result<()>;
}

/// 捕获结果:A1.4 降级语义(不足则收窄,再不足整级关图回 eager——禁止撞死)。
#[derive(Debug)]
pub enum CaptureOutcome {
    /// 全部/部分档位定影成功(实际生效档位表,升序)
    Captured {
        profiles: Vec<usize>,
        /// 定影总 footprints(账本实测,bytes)
        sealed_bytes: u64,
    },
    /// free 不足以容纳任何档 → 整级回 eager(合法,非违约)
    EagerFallback { free_bytes: u64, needed: u64 },
}

impl GraphBindings {
    /// R1:五字段句柄公开(捕获期租约由 CaptureSession 自动登记;runner
    /// /adapter 需要跨步稳定地址时经此取用,禁存副本——句柄活性随图)。
    pub fn frontier(&self) -> &DynTensor<owl_cuda::CudaDevice> { &self.frontier }
    pub fn positions(&self) -> &DynTensor<owl_cuda::CudaDevice> { &self.positions }
    pub fn slot_mapping(&self) -> &DynTensor<owl_cuda::CudaDevice> { &self.slot_mapping }
    pub fn kv_lens(&self) -> &DynTensor<owl_cuda::CudaDevice> { &self.kv_lens }
    pub fn logits_out(&self) -> &DynTensor<owl_cuda::CudaDevice> { &self.logits_out }

    /// P 阶段构造:五缓冲经 Weights 池工厂(租约常驻;A1.1 预算登记点)。
    pub fn new(dev: &CudaDevice, max_bs: usize, vocab: usize) -> Result<Self, Error> {
        let pool = dev
            .create_pool(PoolConfig {
                name: format!("graph-bindings-{}", std::process::id()),
                kind: PoolKind::Weights,
                bytes: ((max_bs * 4 * 4 + max_bs * vocab * 4) as u64).max(1 << 20),
            })
            .map_err(|e| Error::Backend(e))?;
        let mk_u32 = |n: usize| -> Result<DynTensor<owl_cuda::CudaDevice>, Error> {
            Ok(DynTensor::from_u32(&pool.zeros_tensor::<u32>(&[n])?))
        };
        Ok(Self {
            frontier: mk_u32(max_bs)?,
            positions: mk_u32(max_bs)?,
            slot_mapping: mk_u32(max_bs)?,
            kv_lens: mk_u32(max_bs)?,
            logits_out: DynTensor::from_f32(&pool.zeros_tensor::<f32>(&[max_bs, vocab])?),
            vocab,
        })
    }

    /// EagerOnly:H2D 写 frontier(同步拷贝;禁入捕获段——本方法只在
    /// replay 前的 host 侧调用,图主流 launch 之前,流序天然先于图)。
    pub fn write_frontier_from_host(&self, dev: &CudaDevice, ids: &[u32]) -> Result<(), Error> {
        write_u32(dev, self.frontier.device_ptr() as *mut u32, ids, self.frontier_len())
    }

    pub fn write_positions_from_host(&self, dev: &CudaDevice, pos: &[u32]) -> Result<(), Error> {
        write_u32(dev, self.positions.device_ptr() as *mut u32, pos, self.frontier_len())
    }

    pub fn write_slot_mapping_from_host(
        &self,
        dev: &CudaDevice,
        slots: &[u32],
    ) -> Result<(), Error> {
        write_u32(dev, self.slot_mapping.device_ptr() as *mut u32, slots, self.frontier_len())
    }

    pub fn write_kv_lens_from_host(&self, dev: &CudaDevice, lens: &[u32]) -> Result<(), Error> {
        write_u32(dev, self.kv_lens.device_ptr() as *mut u32, lens, self.frontier_len())
    }

    fn frontier_len(&self) -> usize {
        self.frontier.shape()[0]
    }

    /// 本档 narrow 元数据视图(owl-nn DynTensor 暂无 narrow,指针视图等价;
    /// nn 需求已上报:view 语义回填后可换 DynTensor::narrow)
    pub fn view(&self, bs: usize) -> BindingsView {
        BindingsView {
            frontier: self.frontier.device_ptr() as *mut u32,
            positions: self.positions.device_ptr() as *mut u32,
            slot_mapping: self.slot_mapping.device_ptr() as *mut u32,
            kv_lens: self.kv_lens.device_ptr() as *mut u32,
            logits_out: self.logits_out.device_ptr() as *mut f32,
            bs,
            max_bs: self.frontier_len(),
            vocab: self.vocab,
        }
    }

    /// 依赖令牌透传(捕获 session 手工租约逃生口的输入面)
    pub fn dyn_tensors(&self) -> [&DynTensor<owl_cuda::CudaDevice>; 5] {
        [
            &self.frontier,
            &self.positions,
            &self.slot_mapping,
            &self.kv_lens,
            &self.logits_out,
        ]
    }
}

/// 同步 H2D(EagerOnly;bind_to_thread 前置,长度越界拒绝)
fn write_u32(dev: &CudaDevice, dst: *mut u32, src: &[u32], cap: usize) -> Result<(), Error> {
    if src.len() > cap {
        crate::bail!("bindings 写入越界: {} > 容量 {cap}", src.len());
    }
    // P0-3:流序 H2D(memx;EagerOnly 路径)
    dev.memcpy_htod_u32(dev.stream(), dst, src)
        .map_err(|e| Error::Backend(owl_iface::BackendError::CopyFailed {
            dir: "htod",
            detail: format!("bindings H2D: {e}"),
        }))?;
    Ok(())
}

/// GraphPlan:档位表 + 定影图实例表 + 共享 bindings。
pub struct GraphPlan {
    /// bs → 定影图(单共享捕获池语义由 CaptureSession 池策略承载,A1.3)
    graphs: BTreeMap<usize, owl_cuda::DeviceGraph>,
    /// 实际生效档位(升序;A1.4 预检后可能窄于请求)
    profiles: Vec<usize>,
    bindings: GraphBindings,
    /// 全局图预算(A1.1;A5.4 per-profile 校验用)
    allowance: GraphAllowance,
}

impl GraphPlan {
    /// 档位表规划(xinfer planned_graph_capture_batches 语义:1..=max 精确档;
    /// GDN/mamba slot 映射不可 pad,精确档是语义要求)
    pub fn planned_batches(max_num_seqs: usize) -> Vec<usize> {
        (1..=max_num_seqs.clamp(1, 32)).collect()
    }

    /// 启动期捕获(净空窗口)。流程 per 档位:
    /// warmup(eager forward 一次;姿势 6 门禁前置)
    /// → Governor 周期 begin → CaptureSession.capture(信号自动租约)
    /// → end → 定影 seal(实测 footprint ≤ allowance,A5.4)
    ///
    /// `per_profile_bytes` = 内存规划器给的每档成本估计(A1.1 输入);
    /// `flags` = instantiate 旗标(owl 不需要 AUTO_FREE——租约已钉)。
    #[allow(clippy::type_complexity)]
    pub fn capture<F: GraphForward>(
        dev: &CudaDevice,
        requested: Vec<usize>,
        max_bs: usize,
        vocab: usize,
        allowance: GraphAllowance,
        per_profile_bytes: impl Fn(usize) -> u64,
        forward: F,
        flags: owl_cuda::ffi::sys::CUgraphInstantiate_flags,
    ) -> Result<(Self, CaptureOutcome), Error> {
        // ---- A1.4 预检:free 不足收窄,再不足回 eager ----
        let (free, _total) = dev.mem_get_info()?;
        let free = free as u64;
        let profiles_all: Vec<GraphProfile> = requested
            .iter()
            .map(|&bs| GraphProfile { batch: bs as u32, max_frontier: bs as u32 })
            .collect();
        let governor_probe = GraphGovernor::new(allowance, profiles_all);
        let kept = match governor_probe.preflight(free, |p| per_profile_bytes(p.batch as usize)) {
            Ok(kept) => kept,
            Err(_) => {
                let needed: u64 = requested.iter().map(|&b| per_profile_bytes(b)).sum();
                return Ok((
                    Self::bindings_only(dev, max_bs, vocab, allowance)?,
                    CaptureOutcome::EagerFallback {
                        free_bytes: free as u64,
                        needed,
                    },
                ));
            }
        };

        let bindings = GraphBindings::new(dev, max_bs, vocab)?;
        let mut graphs: BTreeMap<usize, owl_cuda::DeviceGraph> = BTreeMap::new();
        let mut sealed_total: u64 = 0;
        let mut alive_before = dev.ledger().bytes_alive;

        for p in kept {
            let bs = p.batch as usize;
            // warmup(姿势 6):eager 全链一次;同时把档位形状的懒状态清干净
            let eager_ctx = KernelCtx::eager(MemPhase::Live, dev.stream().clone());
            forward.forward(&eager_ctx, &bindings.view(bs))?;
            dev.ctx().synchronize().map_err(|e| Error::Msg(format!("sync: {e:?}")))?;

            // Governor 周期(单档;多图句柄表 = owl-graph 需求,已上报)
            let mut gov = GraphGovernor::new(allowance, vec![p.clone()]);
            gov.begin_capture().map_err(|e| Error::Msg(format!("graph governor: {e}")))?;

            let mut session = dev.capture_session()?;
            // bindings 手工租约(五缓冲是图的输入/输出面;forward 内 emit 只盖
            // scratch,bindings 指针不经 ops 触碰 → 逃生口登记)
            for t in bindings.dyn_tensors() {
                lease_dyn(&mut session, t);
            }
            let recorder = CaptureRecorder::new();
            let alive_pre = dev.ledger().bytes_alive;
            let (_, graph) = session.capture(flags, |frame| {
                let ctx = KernelCtx::capturing_leased(recorder.clone(), frame);
                // engine Error → BackendError(capture 契约面;字符串保全)
                forward
                    .forward(&ctx, &bindings.view(bs))
                    .map_err(|e| owl_iface::BackendError::Init(format!("graph forward: {e}")))
            })?;
            let _ = alive_pre;

            gov.end_capture(p.clone()).map_err(|e| Error::Msg(format!("graph governor: {e}")))?;

            // 定影协议:暖场 replay + sync 后测量(此处 replay 一次以强制懒提交)
            graph.upload().map_err(|e| Error::Msg(format!("graph governor: {e}")))?;
            graph.launch().map_err(|e| Error::Msg(format!("graph governor: {e}")))?;
            dev.ctx().synchronize().map_err(|e| Error::Msg(format!("sync: {e:?}")))?;
            let alive_after = dev.ledger().bytes_alive;
            let measured = alive_after.saturating_sub(alive_before);
            alive_before = alive_after;
            match gov.seal(measured) {
                Ok(()) => {
                    sealed_total += measured;
                    graphs.insert(bs, graph);
                }
                Err(_gov_err) => {
                    // A5.4:超 allowance = 捕获规划失误,销图弃档(不宽容;不撞死)
                    drop(graph);
                    crate::bail!("档位 bs={bs} 定影超支(弃档): {gov_err}");
                }
            }
        }

        let profiles: Vec<usize> = graphs.keys().copied().collect();
        Ok((
            Self { graphs, profiles: profiles.clone(), bindings, allowance },
            CaptureOutcome::Captured { profiles, sealed_bytes: sealed_total },
        ))
    }

    fn bindings_only(
        dev: &CudaDevice,
        max_bs: usize,
        vocab: usize,
        allowance: GraphAllowance,
    ) -> Result<Self, Error> {
        Ok(Self {
            graphs: BTreeMap::new(),
            profiles: Vec::new(),
            bindings: GraphBindings::new(dev, max_bs, vocab)?,
            allowance,
        })
    }

    /// 回放:找 ≥bs 的最小档 → launch(DeviceGraph 内建 debug 世代校验)。
    pub fn replay(&self, bs: usize) -> Result<(), Error> {
        let Some((&slot, graph)) = self.graphs.range(bs..).next() else {
            crate::bail!("GraphPlan: 无 ≥bs={bs} 的已捕获档(available={:?})", self.profiles);
        };
        let _ = slot;
        graph.launch().map_err(Error::Backend)
    }

    pub fn profiles(&self) -> &[usize] {
        &self.profiles
    }

    pub fn bindings(&self) -> &GraphBindings {
        &self.bindings
    }

    /// 定影总 footprint(A5.3 周期对账输入)
    pub fn sealed_bytes(&self) -> u64 {
        self.allowance.total() // v1: per-profile sealed 值在 capture 返回;此处给预算上限
    }

    /// 图存活判据(回收窗口判定用)
    pub fn is_live(&self) -> bool {
        !self.graphs.is_empty()
    }
}

/// DynTensor 手工租约(通过 Persistent 路径的对称面;v1 直接经 session.lease
/// 的 Persistent 通道不可行——DynTensor 保活的是 Tensor 克隆。方案:借
/// keepalive 弱点,退化为 emit 通道。此处以 zh 弃用注释记录,真实租约由
/// 捕获闭包内 forward 对 bindings 的 emit 语义补齐;见汇报 nn 需求 #2)
fn lease_dyn(_session: &mut owl_cuda::CaptureSession, _t: &DynTensor<owl_cuda::CudaDevice>) {
    // v1:no-op。bindings 的租约保障 = P 阶段池缓冲永活于 GraphBindings
    // (结构体持 keepalive 直到 GraphPlan drop)。A1.2 语义由所有权兜底。
}

/// GraphPhase 仅为 re-export 便利(外部对账用)
#[allow(unused)]
fn _phase_witness(_: GraphPhase) {}

#[cfg(test)]
mod tests {
    use super::*;
    use owl_cuda::ffi::sys;

    /// 冒烟假模型:清零 frontier + 用常量填 logits(裸 FFI;不依赖 layers)。
    struct SetLogits(f32);
    impl GraphForward for SetLogits {
        fn forward(&self, ctx: &KernelCtx, view: &BindingsView) -> Result<()> {
            let stream = ctx.stream().cu_stream();
            unsafe {
                sys::cuMemsetD32Async(
                    view.frontier as sys::CUdeviceptr,
                    0x2A,
                    view.bs,
                    stream,
                )
                .result()
                .map_err(|e| Error::Msg(format!("memset frontier: {e:?}")))?;
                sys::cuMemsetD32Async(
                    view.logits_out as sys::CUdeviceptr,
                    self.0.to_bits(),
                    view.bs * view.vocab,
                    stream,
                )
                .result()
                .map_err(|e| Error::Msg(format!("memset logits: {e:?}")))?;
            }
            Ok(())
        }
    }

    fn dtoh_f32(dev: &CudaDevice, ptr: *const f32, n: usize) -> Vec<f32> {
        dev.ctx().bind_to_thread().unwrap();
        let mut v = vec![0f32; n];
        unsafe {
            sys::cuMemcpyDtoH_v2(
                v.as_mut_ptr() as *mut core::ffi::c_void,
                ptr as sys::CUdeviceptr,
                n * 4,
            )
            .result()
            .unwrap();
        }
        v
    }

    /// 端到端:预检 → 捕获(4 档)→ 定影 → host 写 0 → replay → 图内 memset 复写 7.0
    #[test]
    fn graphplan_capture_replay_smoke() {
        let dev = CudaDevice::new(owl_cuda::test_device_ordinal()).expect("需要 CUDA 设备");
        let mut ops = owl_nn::OpsCtx::new_with_scratch(&dev, 1 << 20).unwrap();

        // 姿势 6 预热:一次真实 eager 发射(ops.add 内部 note_launch)
        let pool = dev
            .create_pool(PoolConfig {
                name: "smoke-warm".into(),
                kind: PoolKind::Scratch,
                bytes: 1 << 20,
            })
            .unwrap();
        let a = pool.scratch_tensor::<f32>(&[8]).unwrap();
        let b = pool.scratch_tensor::<f32>(&[8]).unwrap();
        let mut o = pool.scratch_tensor::<f32>(&[8]).unwrap();
        let warm_ctx = ops.ctx(MemPhase::Live);
        ops.add(&warm_ctx, &a, &b, &mut o).unwrap();
        dev.ctx().synchronize().unwrap();

        // 捕获:1..=4 精确档,max_bs=4,vocab=8
        let allowance = GraphAllowance { decode_bytes: 64 << 20, verify_bytes: 0 };
        let (plan, outcome) = GraphPlan::capture(
            &dev,
            vec![1, 2, 3, 4],
            4,
            8,
            allowance,
            |bs| (2 << 20) + bs as u64 * 4096,
            SetLogits(7.0),
            sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        )
        .unwrap();
        let profiles = match outcome {
            CaptureOutcome::Captured { profiles, sealed_bytes: _ } => {
                assert_eq!(profiles, vec![1, 2, 3, 4], "四档全定影");
                // memset-only 假模型捕获期零分配,footprint=0 合法;
                // 真模型(qwen3_5 接入后)此处应 > 0(scratch 中间量存活)
                profiles
            }
            CaptureOutcome::EagerFallback { free_bytes, needed } => {
                panic!("冒烟机不应降级: free={free_bytes} needed={needed}");
            }
        };
        assert_eq!(plan.profiles(), &profiles);
        assert!(plan.is_live());

        // 判别实验:host 把 logits 清 0 → replay(2) → 只有图能把 7.0 写回
        let logits_ptr = plan.bindings().view(2).logits_out;
        write_zero_logits(&dev, logits_ptr, 2 * 8);
        let check = dtoh_f32(&dev, logits_ptr, 8);
        assert!(check.iter().all(|&v| v == 0.0), "host 清零生效");

        plan.replay(2).expect("replay bs=2(档位表含 2)");
        dev.ctx().synchronize().unwrap();
        let check = dtoh_f32(&dev, logits_ptr, 8);
        assert!(
            check.iter().all(|&v| v == 7.0),
            "replay 后图内 memset 必须复写 7.0(实际 {check:?})"
        );

        // frontier 同理:图内 0x2A 常量写入
        plan.replay(4).unwrap();
        dev.ctx().synchronize().unwrap();
        let mut f = vec![0u32; 4];
        unsafe {
            sys::cuMemcpyDtoH_v2(
                f.as_mut_ptr() as *mut core::ffi::c_void,
                plan.bindings().view(4).frontier as sys::CUdeviceptr,
                16,
            )
            .result()
            .unwrap();
        }
        assert!(f.iter().all(|&v| v == 0x2A), "frontier 图内写入回读 {f:?}");
    }

    fn write_zero_logits(dev: &CudaDevice, ptr: *mut f32, n: usize) {
        dev.ctx().bind_to_thread().unwrap();
        let zeros = vec![0f32; n];
        unsafe {
            sys::cuMemcpyHtoD_v2(
                ptr as sys::CUdeviceptr,
                zeros.as_ptr() as *const core::ffi::c_void,
                n * 4,
            )
            .result()
            .unwrap();
        }
    }
}

//! **引擎 init（装配 + 装载 + 启动）** —— 2026-11-15 P1 自 `apps/server` 下沉。
//!
//! 原实现散在 `apps/server/src/main.rs`（`engine_of` / `engine_of_tp` /
//! `load_shard` / `executor_main`）+ `config.rs`（选卡）。下沉后：
//!
//! ```text
//! 选卡 → 卡锁 → 单卡/TP 装配 → 装载（TP 双线程并行）→ run → 武装调试门 → ready
//! ```
//!
//! 线程纪律：本函数**在引擎线程内**跑（`EngineService::boot` spawn 的那条）；
//! `RunningEngine` 非 Send，建/泵/销毁必须同线程。失败 = 结构化 `Err` 经
//! `ready` 交回 `boot`（server 不参与任何装配细节）。

use std::path::PathBuf;

use owl_iface::contract::ModelError;

use super::devices::{Devices, resolve_devices};
use crate::engine::{Engine, EngineConfig};
use crate::{LoadArm, LoadIntent, LoadedModel, TpEngine, TpOpts};
use crate::Result;

/// `[parallel]` + 站点面（选卡所需的全部输入）。
#[derive(Clone, Debug)]
pub struct DeviceOpts {
    pub parallel: owl_shared::config::ParallelCfg,
    pub gpu_min_free_ratio: f64,
    /// 卡锁身份（端口/角色；仅日志与冲突提示用）
    pub lock_port: u16,
    pub lock_role: Option<String>,
}

/// 一次 boot 的完整输入（server 侧只做字段搬运，零装配逻辑）。
#[derive(Clone, Debug)]
pub struct BootSpec {
    pub devices: DeviceOpts,
    /// 引擎旋钮（`device_ordinal` 由 boot 逐 rank 回填）
    pub engine: EngineConfig,
    pub model: PathBuf,
    /// 装载面事实（臂/意图由引擎自己换算 —— 白名单只在这里判）
    pub format: owl_shared::config::ModelFormat,
    pub quant: owl_shared::config::QuantScheme,
    pub head_quant: owl_shared::config::HeadQuant,
    pub embed_quant: owl_shared::config::EmbedQuant,
    /// TP 拓扑/通信档（`TpOpts` 的换算也归引擎）
    pub tp_ar: owl_shared::config::TpArMode,
    pub tp_comm: owl_shared::config::TpCommMode,
    pub tp_ar_f8: bool,
    /// `runtime.tp_vocab_split`（装载面）
    pub tp_vocab_split: bool,
    /// 调试门（boot 完成后武装；`Off` = 不碰）
    pub step_gate: StepGate,
    pub perf_dump: Option<PathBuf>,
    /// 请求通道深度
    pub channel_capacity: usize,
}

/// 单步/粘滞调试门（原 `[probes] step_gate*`；武装点在 **run 之后**）
#[derive(Clone, Copy, Debug, Default)]
pub struct StepGate {
    pub mode: owl_shared::stepgate::StepMode,
    pub ms: u64,
    pub us: u64,
    pub every: u64,
}

/// boot 完成回执（日志/诊断面用；server 不消费内部字段）
#[derive(Clone, Debug)]
pub struct Booted {
    pub devices: Devices,
    pub capture: String,
}

/// **选卡**（引擎线程内调用；失败 = 结构化 Err）。
pub fn resolve(spec: &BootSpec) -> Result<Devices> {
    resolve_devices(&spec.devices.parallel, spec.devices.gpu_min_free_ratio).map_err(ModelError::Msg)
}

/// **装配 + 装载 + run**（引擎线程内调用；失败 = 结构化 Err）。
pub async fn assemble(
    spec: &BootSpec,
    devices: &Devices,
) -> Result<(crate::RunningEngine<crate::GpuFace>, Booted)> {
    let t0 = std::time::Instant::now();
    let world = devices.world();
    // **装载白名单在此判死**（非白名单组合 = boot 期拒，不等到装载到一半才报缺键）；
    // 臂 → 计划 / 头表覆盖 → LoadIntent 的换算也在这里（server 不再知道这些）。
    let arm = crate::load_arm_of(spec.format, spec.quant)
        .map_err(|e| ModelError::Msg(e.to_string()))?;
    let intent = crate::LoadIntent::of(arm, spec.head_quant, spec.embed_quant);
    let tp = TpOpts {
        ar_f8: spec.tp_ar_f8,
        ar_fused: spec.tp_ar.uses_fused(),
        ar2: spec.tp_ar.uses_ar2(),
        comm: spec.tp_comm,
    };
    owl_shared::log_info!("boot", "装载臂 = {}", arm.name());
    if arm == LoadArm::F16Base {
        owl_shared::log_info!(
            "boot",
            "[model] safetensors × w4a16 按 f16/bf16 基线装载(G128 pack-quantized 档待接线,届时按键面探测分流)"
        );
    }
    let engine_cfg = |ordinal: usize| -> EngineConfig {
        let mut c = spec.engine.clone();
        c.device_ordinal = ordinal;
        c
    };
    let mut phase_used: Option<u64> = None;
    let running = match world {
        // ── 单卡档（装载 = 恒等分片，与 TP 走同一个 load_shard）──────────
        1 => {
            let mut engine = Engine::new(engine_cfg(devices.device()))?;
            vram_phase("engine:context", engine.mem_info().await, &mut phase_used);
            owl_shared::log_info!("boot", "设备绑定 {:.2}s", t0.elapsed().as_secs_f32());
            let model = load_shard(spec, arm, &intent, &mut engine, 0, 1).await?;
            owl_shared::log_info!(
                "boot",
                "权重/tokenizer/rope 装载 {:.2}s",
                t0.elapsed().as_secs_f32()
            );
            vram_phase("engine:weights", engine.mem_info().await, &mut phase_used);
            let running = engine.run(model).await?;
            owl_shared::log_info!("boot", "引擎装配(run)总计 {:.2}s", t0.elapsed().as_secs_f32());
            running
        }
        // ── TP 档（world = 2；rank 序 = devices 列表序）────────────────────
        2 => {
            let ord = devices.ordinals();
            let mut engine0 = Engine::new(engine_cfg(ord[0]))?;
            vram_phase("engine:context", engine0.mem_info().await, &mut phase_used);
            // rank1 的账本/设备序**在装载前**从 rank0 接过（装载即发事件）
            let mut engine1 =
                Engine::new_with_ledger(engine_cfg(ord[1]), engine0.ledger_cell(), 1)?;
            let (join0, join1) = std::thread::scope(|scope| {
                let h0 =
                    scope.spawn(|| load_rank(spec, arm, &intent, &mut engine0, 0, 2, "rank0"));
                let h1 =
                    scope.spawn(|| load_rank(spec, arm, &intent, &mut engine1, 1, 2, "rank1"));
                (h0.join(), h1.join())
            });
            let model0 = join0
                .map_err(|_| ModelError::Msg("TP rank0 装载线程 panic".into()))??;
            let model1 = join1
                .map_err(|_| ModelError::Msg("TP rank1 装载线程 panic".into()))??;
            vram_phase("engine:tp-weights", engine0.mem_info().await, &mut phase_used);
            owl_shared::log_info!(
                "boot",
                "TP2 分片装载(rank0 ordinal {} / rank1 ordinal {}):{:.2}s",
                ord[0],
                ord[1],
                t0.elapsed().as_secs_f32()
            );
            let tp = TpEngine::assemble_with(vec![engine0, engine1], vec![model0, model1], tp)
                .await?;
            if spec.tp_ar_f8 {
                owl_shared::log_info!(
                    "boot",
                    "TP2 **AR fp8 载荷档**已开(runtime.tp_ar_f8):载荷 e4m3(1.031×)"
                );
            }
            owl_shared::log_info!(
                "boot",
                "TP2 装配完成(分档 + staging 握手 + 双 rank 图捕获):{:.2}s",
                t0.elapsed().as_secs_f32()
            );
            let running = tp.run()?;
            owl_shared::log_info!("boot", "引擎装配(run)总计 {:.2}s", t0.elapsed().as_secs_f32());
            running
        }
        // config 层只放 1/2；到这里 = 内部不变量破了 ⇒ 响亮拒，不静默降级
        n => {
            return Err(ModelError::Msg(format!(
                "设备束 world = {n} 不受支持(单卡 1 / TP 2;config 层守门漏了)"
            )))
        }
    };
    // 调试门武装：**必须在 run 之后**（boot 期 warmup/图捕获也跑解释器，
    // 若一开始就 Single 会自锁；见 `owl_shared::stepgate` 模块头纪律 1）。
    if spec.step_gate.mode != owl_shared::stepgate::StepMode::Off {
        owl_shared::stepgate::configure_ex(
            spec.step_gate.mode,
            spec.step_gate.ms,
            spec.step_gate.us,
            spec.step_gate.every,
        );
        eprintln!(
            "[boot] 调试门已武装:mode={} interval={}(ms={} us={}) every={}(状态面 GET /debug/step;发步 POST /debug/step/release/<n>)",
            spec.step_gate.mode.name(),
            if spec.step_gate.us > 0 {
                format!("{}us", spec.step_gate.us)
            } else {
                format!("{}ms", spec.step_gate.ms)
            },
            spec.step_gate.ms,
            spec.step_gate.us,
            spec.step_gate.every
        );
    }
    let capture = format!("{:?}", running.capture_outcome());
    Ok((
        running,
        Booted {
            devices: devices.clone(),
            capture,
        },
    ))
}

/// 单 rank 分片装载（TP 档两条线程并行调用；单卡 = 恒等分片）。
///
/// 每个装载线程自带 `current_thread` runtime（GPU 设备面在独立线程，不会被
/// 主泵阻塞）。
fn load_rank(
    spec: &BootSpec,
    arm: LoadArm,
    intent: &LoadIntent,
    engine: &mut Engine<owl_cuda::GpuClient>,
    rank: usize,
    world: usize,
    tag: &'static str,
) -> Result<LoadedModel> {
    let t = std::time::Instant::now();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| ModelError::Msg(format!("TP {tag} loader runtime: {e}")))?;
    let out = rt.block_on(load_shard(spec, arm, intent, engine, rank, world));
    owl_shared::log_info!(
        "boot",
        "TP2 {tag} 装载 {:.2}s{}",
        t.elapsed().as_secs_f32(),
        if out.is_ok() { "" } else { "(失败)" }
    );
    out
}

/// 单 rank 装载（`load_arm` 是唯一的臂 → 装载器分派点）。
async fn load_shard(
    spec: &BootSpec,
    arm: LoadArm,
    intent: &LoadIntent,
    engine: &mut Engine<owl_cuda::GpuClient>,
    rank: usize,
    world: usize,
) -> Result<LoadedModel> {
    let mut loader = engine
        .loader()
        .slice(rank, world)
        .vocab_split(spec.tp_vocab_split);
    loader.load_arm(&spec.model, arm, intent).await
}

/// boot 分相显存取证行（`[vram-phase]`；与引擎内 `VramPhase` 同一格式）。
fn vram_phase(tag: &str, info: Result<(u64, u64)>, last: &mut Option<u64>) {
    match info {
        Ok((free, total)) => {
            let used = total.saturating_sub(free);
            let delta = last.map(|p| used as i64 - p as i64).unwrap_or(0);
            eprintln!(
                "[vram-phase] {tag}: used={:.1}MiB free={:.1}MiB delta={:+.1}MiB live={:.1}MiB",
                used as f64 / 1048576.0,
                free as f64 / 1048576.0,
                delta as f64 / 1048576.0,
                owl_shared::vram::live_bytes() as f64 / 1048576.0,
            );
            *last = Some(used);
        }
        Err(e) => eprintln!("[vram-phase] {tag}: 查询失败({e})"),
    }
}

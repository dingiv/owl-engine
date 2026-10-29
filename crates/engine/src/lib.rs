//! owl-engine —— 编排/内存/图治理层(M0 骨架,2026-09-26 立;旧 engine
//! 移至 engine-bak 存档,25k 行产品化残留不复用为主干,复用审计见下)。
//!
//! 唯一设备标准 = `owl_iface::contract::DeviceClient`(线格式/回执分级/
//! 顺序语义);模型 = owl-models 声明层(层/Model 纯描述 + 解释器执行)。
//! engine 的存在理由:把「槽管理 + 步循环 + 图三态 + 内存预算」从每个
//! 上层调用者手里收编为**治理对象**(charter A1 / session-plan.md)。
//!
//! # 模块地图(M0)
//!
//! - [`graph_plan`]:GraphPlan 图三态编排面(P0 eager 闭环 + M1 捕获/
//!   回放;原 `Session` 正名,见模块头命名裁定)
//! - [`session`]:**客户端 Session**(agent 连续会话;引擎侧感知 = KV
//!   cache 区域 + 会话 token 账。Session/Turn/Step 三层词汇权威定义)
//!
//! # 复用审计(engine-bak → 本 crate)
//!
//! | 老模块 | 处置 |
//! |---|---|
//! | `session.rs`(464 行流程纪律) | ✅ **移植改写**:warmup 门禁/槽装填/eager 直发保留;ETensor/KernelCtx/CudaPool → TensorOps/DeviceClient |
//! | graphplan 捕获循环(姿势 6/Governor/租约/哨兵③) | ✅ M1 移植(经 DeviceClient graph_begin/end/launch 泛化) |
//! | `kvcache/allocator.rs` + `block_manager` | 参考重写 M1(页表/槽位治理;数据面已换 KV 直排) |
//! | `runner` / `scheduler` / `sequence` | 挂账 M2(每卡一线程 A2.6 + continuous batching) |
//! | `sampler` | M1(host 贪心已在 models::interpreters::generate;设备采样随 Argmax kernel) |
//! | `loader`(gguf/safetensors/load_cache) | ❌ 废弃(owl-models loader + specs 流式装载取代) |
//! | `models/`(candle 旧世界) | ❌ 废弃(owl-models 取代;移植母本价值已录 qwen3-mini-demo §七) |
//! | `server` | ❌ 废弃(backends/cuda GpuClient actor 取代) |
//! | `downloader`/`image`/`multi_node`/`transfer`/`distributed`/`env` | ❌ 产品化残留废弃(需要时按 charter 重立) |
//!
//! 里程碑对齐 charter:M0 骨架 → M1 图治理最小闭环 → M2 TP2 comm →
//! M3 模型 + DFlash2 对齐线 → M4 冲刺线。

pub mod graph_plan;
pub mod session;
pub mod turn;

// engine 面(2026-10-01 拆分:原单文件 1782 行 → 六户;对外路径经
// re-export 保持不变 —— owl_engine::{Engine, EngineConfig,
// LoadedModel, ModelLoader, RunningEngine})
pub(crate) mod engine;
pub(crate) mod disk_io;
pub(crate) mod exec;
pub(crate) mod graph_fleet;
// **状态域户**(2026-10-21 收户):选择层运行态 / 保洁账 —— 字段私有,
// 只经语义方法读写(见各模块头)。
pub(crate) mod housekeeping;
/// 消息边界 → 页对齐 GDN 快照点(纯 host;2026-10-24)
pub(crate) mod msg_boundary;
/// 前缀无感删除:host 纯函数面(识别 / 页对齐 pad / 位置自证)
pub(crate) mod img_evict;
pub(crate) mod loader;
pub(crate) mod running;
pub(crate) mod sampler;
// **调度政策**:准入 + 相位两个独立状态机(纯数据/零 I/O;事实由引擎每拍注入)。
// 本刀只落骨架与算式,尚未接线 ⇒ 允许死码(接线下一步)。
#[allow(dead_code)]
pub(crate) mod sched_policy;
pub(crate) mod probe_dump;
pub(crate) mod scheduler;
pub(crate) mod select;
pub(crate) mod spec_runtime;
pub(crate) mod state;
/// Store 三钩子契约（Acquire / Release / Maintain；2026-10-25 Phase 1）
pub(crate) mod store_hook;
pub(crate) mod tp;
pub(crate) mod vlm;
// 测试分层(test-architecture.md):单元测试走 cfg(test);
// e2e 场景面(源文件里的 pub async fn,无 #[test])走 feature `e2e`,
// 由 crates/engine/tests/engine.e2e.rs 手动调用。
#[cfg(any(test, feature = "e2e"))]
pub mod tests;
/// GGUF 档 e2e 门(纯测试模块;非测试构建不编译)
#[cfg(test)]
pub mod tests_gguf;

pub use engine::{Engine, EngineConfig, EngineKnobs};
pub use owl_iface::handle::{AssetClass, DeviceHandle, Owner};

/// **GPU 设备面类型糖**(内核后端 + 事件账钩子):`RunningEngine`/`TpEngine` 的
/// 公共签名用它 —— 上层不必手写嵌套泛型(2026-10-19 步 3)。
pub type GpuFace = DeviceHandle<owl_cuda::GpuClient>;
pub use loader::{LoadedModel, ModelLoader};

/// lm_head 量化方案映射(**唯一点**:配置枚举 → owl_models QuantPlan;
/// app 层不直接触碰 owl_models,新方案 = 加枚举值 + 此处加臂 +
/// loader 质量门台账)
pub fn head_plan_of(hq: owl_shared::config::HeadQuant) -> owl_models::module::QuantPlan {
    use owl_models::module::QuantPlan;
    match hq {
        owl_shared::config::HeadQuant::Off => QuantPlan::F16,
        owl_shared::config::HeadQuant::RtnInt4G32 => QuantPlan::W4A16Awq,
        owl_shared::config::HeadQuant::RtnInt8Row => QuantPlan::I8W,
    }
}
/// embedding 表量化映射(P8;`None` = 检查点原表计划,`Some(I8W)` = int8)。
pub fn embed_plan_of(
    eq: owl_shared::config::EmbedQuant,
) -> Option<owl_models::module::QuantPlan> {
    match eq {
        owl_shared::config::EmbedQuant::F16 => None,
        owl_shared::config::EmbedQuant::I8 => Some(owl_models::module::QuantPlan::I8W),
    }
}
pub use running::{RunningEngine, TokenizeReply};
pub use tp::{TpEngine, TpOpts, TpStaging};
pub use graph_plan::{GraphPlan, GraphPlanDesc, PlanOutcome};
pub use session::{AgentSession, SessionTable};
pub use turn::{TurnEvent, TurnImage, TurnOpts, TurnSampler, TurnSpec};

/// **装载臂白名单**(2026-10-30 用户裁决:只吃预量化产物)。
///
/// 引擎**不做任何格式转换**:离线转换器已从树中移除,装载期"从全精度现量化"
/// 也不再是对外能力(转换研究在另一条线,产出物再交回 owl 直接装载)。
/// 四族之外 = boot 期**响亮拒**([`load_arm_of`] 点名白名单),不再是旧
/// [`quant_plan_of`] 那种"兜底回 f16 静默跑"。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadArm {
    /// **AWQ-INT4**:pack-quantized / g32 非对称(cyankiwi 系;int4 已由产出方烘焙)
    AwqInt4,
    /// **AWQ-INT4 · W4A8**:与上一臂**同源预量化权重**,装载期把三件套重排成 a8
    /// 布局(纯 repack,不产生新精度),激活走 per-token int8。**生产档**
    /// (prefill +48~50% / TTFT −33% @32k;3090 与 3080 档均跑这一臂)
    AwqA8,
    /// **NVFP4** 块缩放(e2m1 数据 + 每 16 个 k 一个 e4m3 缩放)
    NvFp4,
    /// **GGUF 块型**(Q4_K / IQ4 / IQ3;逐张量型仍以文件为准)
    Gguf(owl_models::formats::gguf::GgmlTy),
    /// **f16/bf16 基线**(无量化;0.8B 开发档与附属件,非生产档)
    F16Base,
}

impl LoadArm {
    /// 臂 → owl_models `QuantPlan`(白名单臂的**唯一**换算处)
    pub fn quant_plan(self) -> owl_models::module::QuantPlan {
        use owl_models::module::QuantPlan as P;
        match self {
            LoadArm::AwqInt4 => P::W4A16Awq,
            LoadArm::AwqA8 => P::W4A8Awq,
            LoadArm::NvFp4 => P::NvFp4,
            LoadArm::Gguf(ty) => P::Gguf(ty),
            LoadArm::F16Base => P::F16,
        }
    }

    /// 诊断名(日志/错误信息用)
    pub fn name(self) -> &'static str {
        match self {
            LoadArm::AwqInt4 => "awq-int4",
            LoadArm::AwqA8 => "awq-w4a8",
            LoadArm::NvFp4 => "nvfp4",
            LoadArm::Gguf(_) => "gguf",
            LoadArm::F16Base => "f16/bf16",
        }
    }
}

/// **server 的唯一装载接线点**(2026-10-30):`(format, quant)` → [`LoadArm`]。
///
/// 白名单(直接加载,不做转换):
///
/// | format | quant | 臂 |
/// |---|---|---|
/// | `awq` | `w4a16` | [`LoadArm::AwqInt4`] —— AWQ-INT4 预量化 |
/// | `awq` | `w4a8` | [`LoadArm::AwqA8`] —— 同源权重 + int8 激活(生产档) |
/// | `safetensors` | `nvfp4` | [`LoadArm::NvFp4`] —— NVFP4 块缩放 |
/// | `gguf` | `q4k` / `iq4` / `iq3` | [`LoadArm::Gguf`] —— GGUF 块型 |
/// | `safetensors` | `w4a16` | [`LoadArm::F16Base`] —— 裸 f16/bf16 基线(开发档) |
///
/// **路线图内、尚未接线**:**G128**(W4A16 g128-sym,llm-compressor / ct
/// pack-quantized:`format=safetensors` + `quant=w4a16` + `.weight_packed` 三件套)。
/// 装载路径已存在(`QuantPlan::W4A16`),挡在 marlin g128 形状雷上(n 必须
/// `512·2^k`);放行前需按**键面探测**(`CheckpointKind::PackQuantized`)与裸 f16
/// 目录分流 —— 今天两者都答 [`LoadArm::F16Base`],g128 产物会按 f16 读而报缺键。
///
/// 其余组合 = `Err`(错误信息点名白名单;不猜、不兜底)。加第五族是**决策**,
/// 不是配置开关:枚举加臂 + 此表加行 + 质量门台账。
pub fn load_arm_of(
    f: owl_shared::config::ModelFormat,
    q: owl_shared::config::QuantScheme,
) -> Result<LoadArm> {
    use owl_shared::config::{ModelFormat as F, QuantScheme as Q};
    match (f, q) {
        (F::Awq, Q::W4a16) => Ok(LoadArm::AwqInt4),
        (F::Awq, Q::W4a8) => Ok(LoadArm::AwqA8),
        (F::Safetensors, Q::Nvfp4) => Ok(LoadArm::NvFp4),
        (F::Safetensors, Q::W4a16) => Ok(LoadArm::F16Base),
        (F::Gguf, Q::Q4k) => Ok(LoadArm::Gguf(owl_models::formats::gguf::GgmlTy::Q4K)),
        (F::Gguf, Q::Iq4) => Ok(LoadArm::Gguf(owl_models::formats::gguf::GgmlTy::Iq4Nl)),
        (F::Gguf, Q::Iq3) => Ok(LoadArm::Gguf(owl_models::formats::gguf::GgmlTy::Iq3Xxs)),
        (f, q) => Err(owl_iface::contract::ModelError::Msg(format!(
            "[model] format = {f:?} × quant = {q:?} 不在装载白名单内。\
             owl 只直接加载**预量化产物**(不做格式转换,离线转换器不在本仓):\
             awq+w4a16(AWQ-INT4)/ awq+w4a8(W4A8)/ safetensors+nvfp4(NVFP4)\
             / gguf+q4k|iq4|iq3(GGUF)/ safetensors+w4a16(f16 基线,开发档;\
             G128 待接线,键面带 .weight_packed 时才是 g128)。\
             换格式请先在外部产出相应 4bit 检查点"
        ))),
    }
}

/// 主干量化方案映射(**测试/内部用**;server 接线已改为 [`load_arm_of`] ——
/// 本表的两条兜底臂(`(_, Nvfp4)`、`(Safetensors, W4a8)`)不再可从配置面到达)。
///
/// 同 [`head_plan_of`] 的范式 —— app 层不碰 owl_models,新方案 =
/// 加枚举值 + 此处加臂 + 质量门台账。
///
/// 为什么同时要 `format`:同一个 `w4a16` 在 AWQ 档是 **pack-quantized 布局**
/// (`W4A16Awq`),在普通 safetensors 档是**装载期重排**(`W4A16`)——
/// 语义由"权重怎么摆"决定,量化只给比特数。GGUF 档则由块型直接给
/// `Gguf(ty)`(实际逐张量型仍以文件为准)。
pub fn quant_plan_of(
    f: owl_shared::config::ModelFormat,
    q: owl_shared::config::QuantScheme,
) -> owl_models::module::QuantPlan {
    use owl_shared::config::{ModelFormat as F, QuantScheme as Q};
    match (f, q) {
        (F::Gguf, Q::Q4k) => owl_models::module::QuantPlan::Gguf(owl_models::formats::gguf::GgmlTy::Q4K),
        (F::Gguf, Q::Iq4) => owl_models::module::QuantPlan::Gguf(owl_models::formats::gguf::GgmlTy::Iq4Nl),
        (F::Gguf, Q::Iq3) => owl_models::module::QuantPlan::Gguf(owl_models::formats::gguf::GgmlTy::Iq3Xxs),
        // validate 已拦;这里回 F16 而不是 panic(防御:配置门漏了也不炸引擎)
        (F::Gguf, _) => owl_models::module::QuantPlan::F16,
        (F::Awq, Q::W4a16) => owl_models::module::QuantPlan::W4A16Awq,
        (F::Awq, Q::W4a8) => owl_models::module::QuantPlan::W4A8Awq,
        (F::Safetensors, Q::W4a16) => owl_models::module::QuantPlan::W4A16,
        // NVFP4:装载期从 f16 现量化(权重+激活块缩放)
        (F::Safetensors, Q::Nvfp4) => owl_models::module::QuantPlan::NvFp4,
        // 普通 safetensors 的 W4A8 尚无独立 plan(AWQ 布局是其唯一实现)
        (F::Safetensors, Q::W4a8) => owl_models::module::QuantPlan::W4A8Awq,
        // 未实现/错配(validate 已拒):保持 f16 直读,不 panic
        (_, Q::Nvfp4) => owl_models::module::QuantPlan::NvFp4,
        (_, Q::Q4k | Q::Iq4 | Q::Iq3) => {
            owl_models::module::QuantPlan::F16
        }
    }
}

/// **safetensors 目录的“装载期现量化”计划**(**测试/内部用**;server 接线已改为
/// [`load_arm_of`] —— 白名单下唯一允许的 safetensors 量化臂是 `nvfp4`,其余一律
/// 响亮拒,不再有"回 f16 静默跑"的兜底)。
///
/// 为何不直接用 [`quant_plan_of`]:那份表对 `(Safetensors, W4a16)` 答
/// `W4A16` —— 那是**pack-quantized 源**(ct 已打包的 qweight/scales)
/// 的语义,而普通 HF 目录里的 `[model] quant = "w4a16"` 历史上是
/// **空档**(旧 loader 恒 F16;档案注释也写着“safetensors 档忽略”)。
/// 把两者混为一种计划,第一次就会把纯 bf16 检查点按 marlin 键去要
/// `…qweight` 而当场拒启(实测现场)。
///
/// 所以本表只回答一件事:**普通 HF 目录能“在装载期从全精度现算出”哪些
/// 量化格式**。今天只有 `nvfp4`(host 打包,见 `formats::transform::NvFp4Pack`);
/// 其余一律 F16 直读(与历史逐字节一致)。
pub fn safetensors_load_plan(
    q: owl_shared::config::QuantScheme,
) -> owl_models::module::QuantPlan {
    use owl_shared::config::QuantScheme as Q;
    match q {
        Q::Nvfp4 => owl_models::module::QuantPlan::NvFp4,
        _ => owl_models::module::QuantPlan::F16,
    }
}

/// 本 crate 的结果别名:错误权威 = iface `ModelError`(线上一族,不另立)
pub type Result<T> = std::result::Result<T, owl_iface::contract::ModelError>;

#[cfg(test)]
mod load_arm_tests {
    use super::{load_arm_of, LoadArm};
    use owl_models::formats::gguf::GgmlTy;
    use owl_shared::config::{ModelFormat as F, QuantScheme as Q};

    /// 白名单臂(AWQ-INT4 / W4A8 / NVFP4 / GGUF / f16 基线)= 放行;其余 =
    /// 响亮点名拒(2026-10-30 裁决:只吃预量化产物;W4A8 是生产档)
    #[test]
    fn whitelist_arms_and_loud_refusal() {
        let ok = [
            (F::Awq, Q::W4a16, LoadArm::AwqInt4),
            (F::Awq, Q::W4a8, LoadArm::AwqA8),
            (F::Safetensors, Q::Nvfp4, LoadArm::NvFp4),
            (F::Safetensors, Q::W4a16, LoadArm::F16Base),
            (F::Gguf, Q::Q4k, LoadArm::Gguf(GgmlTy::Q4K)),
            (F::Gguf, Q::Iq4, LoadArm::Gguf(GgmlTy::Iq4Nl)),
            (F::Gguf, Q::Iq3, LoadArm::Gguf(GgmlTy::Iq3Xxs)),
        ];
        for (f, q, want) in ok {
            assert_eq!(load_arm_of(f, q).expect("白名单臂应放行"), want, "{f:?}×{q:?}");
        }
        // 旧 `quant_plan_of` 的兜底臂:纯 HF 目录 × 块型 / awq × nvfp4 / 裸目录 × w4a8
        for (f, q) in [
            (F::Safetensors, Q::Q4k),
            (F::Safetensors, Q::W4a8),
            (F::Awq, Q::Nvfp4),
            (F::Awq, Q::Q4k),
            (F::Gguf, Q::W4a16),
            (F::Gguf, Q::Nvfp4),
        ] {
            let e = load_arm_of(f, q).expect_err("非白名单组合必须响亮拒");
            let msg = e.to_string();
            assert!(msg.contains("白名单"), "错误信息须点名白名单,实得: {msg}");
        }
    }
}

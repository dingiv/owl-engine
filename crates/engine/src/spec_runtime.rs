//! spec(投机解码)运行域(2026-10-14 拆分自 running.rs:RunningEngine
//! 万能结构体按状态域拆解 —— 本模块收编**引擎级 spec 状态**:模式/深度/
//! 草稿器/统计/诊断门/B4 分母保鲜)。
//!
//! 生命周期注记:turn 域 spec 状态(TurnSpecState,七件套)随
//! ActiveTurn 生死,住 [`crate::turn`];本模块的字段是**跨 turn 引擎级**
//! 的(草稿器装载一次、统计全程累计、B4 分母跨 turn 保鲜)—— 两层
//! 不许混居,历史教训见 ActiveTurn::spec 头注(跨 turn 残留 = GDN 污染)。

use std::sync::Arc;

use owl_iface::contract::DeviceClient;
use owl_models::layers::rope::Rope;

use crate::Result;

/// E5-M2b:spec 模式三态(C7 回退挂钩)。裁决在 boot:
/// depth>0 且检查点有 mtp.* → Mtp;dumb 诊断位 → Dumb(哑草稿 =
/// anchor 重复);否则 Off(无草稿器不白发 verify 税)。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum SpecMode {
    Off,
    Dumb,
    Mtp,
    /// DFlash2 码本选择器草稿(E5-DF;dflash2_dir)
    DFlash2,
}

/// 草稿器枚举(E5-DF;MTP 链式 / DFlash2 噪声块式双族)
pub(crate) enum Drafter {
    Mtp(Arc<owl_models::layers::mtp::MtpPredictor>),
    DFlash2(Arc<owl_models::layers::dflash2::DFlash2Draft>),
}

/// spec 接受账(轮级;恒等门/接受率观测面)
#[derive(Default, Debug, Clone, Copy)]
pub(crate) struct SpecStats {
    pub rounds: usize,
    pub sum_m: usize,
    pub full: usize,
    pub partial: usize,
    pub zero: usize,
}

/// 引擎级 spec 运行态(组合叶;RunningEngine 持有)。字段全
/// pub(crate) 供 exec 热路径直读,复合逻辑做成方法。
pub(crate) struct SpecRuntime {
    /// E5-M2:投机轮深(0 = 关;>0 = 每轮草稿数,调度发 SpecRound)
    pub depth: usize,
    /// E5-M2b:spec 模式三态(诊断面)
    #[allow(dead_code)] // 诊断面字段(写入后当前无读者;boot 裁决记录)
    pub mode: SpecMode,
    /// 草稿头(Mtp/DFlash2;embed/lm_head 共享 target 不持有;Arc =
    /// propose 图闭包共享)
    pub drafter: Option<Drafter>,
    /// DFlash2 草稿 rope(θ 1e7;DFlash2 模式 Some)
    pub draft_rope: Option<Arc<Rope>>,
    /// DFlash2 target taps 数(dflash 模式 = 5;其余 0)
    pub dflash_tap_count: usize,
    /// spec 接受账(M2b 恒等门覆盖断言:三路 m 分布 + 账目闭合)
    pub stats: SpecStats,
    /// taps/memory 探针落盘一次性门(OWL_DFLASH_PROBE;AL=0 排查)
    pub mem_dumped: bool,
    /// concat 父块落盘一次性门(同上)
    pub mem_dumped2: bool,
    /// propose hidden 落盘一次性门(同上)
    pub hid_dumped: bool,
    /// B4 判据跨 turn 保鲜(2026-10-14):裸步成本 EWMA(ms)+ 其采样
    /// ctx。**为什么必须跨 turn**:TurnSpecState 随 turn 销毁 ⇒ 每 turn
    /// 起步分母 = 0 ⇒ 首个窗口只能用保守兜底线(2.5),在"真线 1.4 而
    /// AL≈2"的域会误降级并抖动。带 ctx 保鲜后,同档续聊直接沿用真线。
    pub carry_ms: f32,
    /// B4 保鲜 ctx(pos;涨过 25% 即作废,换档后旧采样失真)
    pub carry_ctx: usize,
}

// ===========================================================================
// spec 域装载(2026-10-15 拆分自 engine.rs run():模式裁决 + 草稿器
// 装载 + 草稿 rope —— 引擎级 spec 事实的一次性收拢)
// ===========================================================================

/// spec 装载输入(boot 旋钮 + 检查点事实的显式快照;
/// 引擎内部零隐式依赖 —— 全部经此传入)
pub(crate) struct SpecBootInput<'a> {
    /// 投机深度原始值(0 = 关)
    pub(crate) depth_raw: usize,
    /// DFlash2 草稿检查点目录(None = 无该族草稿器)
    pub(crate) dflash2_dir: Option<&'a str>,
    /// MTP 草稿检查点目录(检查点自带 mtp.* 时 Some)
    pub(crate) mtp_dir: Option<&'a std::path::Path>,
    /// 哑草稿诊断位(无真草稿器时的 Dumb 档)
    pub(crate) dumb: bool,
    /// 草稿池 e4m3(B6 偷显存;AL 有损需 A/B)
    pub(crate) draft_kv_fp8: bool,
    /// DFlash2 taps 退回 last_hidden(排查开关)
    pub(crate) notaps: bool,
}

/// **E5-M2b/C7:spec 模式三态裁决** + 草稿器装载(草稿 embed/lm_head
/// 均共享 target 不持有)+ DFlash2 草稿 rope(θ 1e7,rotary 全维 128;
/// z-lab 检查点家族)。
///
/// 裁决律:depth>0 且 dflash2_dir → DFlash2;depth>0 且检查点有 mtp.*
/// → Mtp;dumb 诊断位 → Dumb;否则 Off(spec_depth 归零,调度回落
/// DecodeBatch —— 无草稿器不白发 verify 税)。深度上限:DFlash2 = 7
/// (block_size 8 = 1 锚 + 7 草稿,检查点家族定形)/ 其余 = 3
/// (kv_slots [u32;4] 契约)。
pub(crate) async fn load_spec_runtime<D: DeviceClient + 'static>(
    face: &mut D,
    inp: SpecBootInput<'_>,
) -> Result<SpecRuntime> {
    let mode = if inp.depth_raw == 0 {
        SpecMode::Off
    } else if inp.dflash2_dir.is_some() {
        SpecMode::DFlash2
    } else if inp.mtp_dir.is_some() {
        SpecMode::Mtp
    } else if inp.dumb {
        SpecMode::Dumb
    } else {
        owl_shared::log_info!("boot", "OWL_SPEC_DEPTH>0 但无草稿器(非 mtp/dflash2 检查点)→ C7 回落 DecodeBatch");
        SpecMode::Off
    };
    let depth = if mode == SpecMode::Off {
        0
    } else if mode == SpecMode::DFlash2 {
        inp.depth_raw.min(7)
    } else {
        inp.depth_raw.min(3)
    };
    if mode != SpecMode::Off {
        owl_shared::log_info!("boot", "spec 模式 = {:?} depth={depth}(注:spec 轮恒 greedy,采样 parity 挂 D4 device accept)",
            mode
        );
    }
    // 草稿器装载(C7;embed/lm_head 均共享 target)
    let drafter = match mode {
        SpecMode::Mtp => {
            let dir = inp.mtp_dir.expect("Mtp 模式必有 mtp_dir(上游模式裁决已守)");
            // 声明构造走起点事实(同一份 27B 检查点的 spec;报告已由主模型装载
            // 路径上屏,此处不重报);装载走通用入口
            let (spec, _rep) = owl_models::model::ModelSpec::from_ckpt(
                std::path::Path::new(dir),
                owl_models::formats::spec::SpecSource::family(
                    &owl_models::formats::qwen35::qwen3_8_27b(),
                ),
            )?;
            let mtp = owl_models::layers::mtp::MtpPredictor::from_spec(&spec);
            let opts = owl_models::formats::load::LoadOpts::new(face, owl_models::contract::Dtype::F16);
            owl_models::formats::load::load_from_dir(std::path::Path::new(dir), face, &mtp, opts).await?;
            owl_shared::log_info!("boot", "MTP 草稿头装载(spec depth={depth})");
            Some(Drafter::Mtp(Arc::new(mtp)))
        }
        SpecMode::DFlash2 => {
            let dir = inp.dflash2_dir.expect("DFlash2 模式必有 dflash2_dir(上游模式裁决已守)");
            let t_draft = std::time::Instant::now();
            // 声明:草稿检查点自己的 config(组件档)+ 起点格式探测 → fc 臂;
            // 装载走通用入口(BF16 全程;设备重排 = 能力自述裁决)
            let d2 = owl_models::formats::qwen35::dflash2_draft(
                std::path::Path::new(dir),
                inp.draft_kv_fp8,
            )?;
            let opts = owl_models::formats::load::LoadOpts::new(face, owl_models::contract::Dtype::BF16);
            let draft_manifest =
                owl_models::formats::load::load_from_dir(std::path::Path::new(dir), face, &d2, opts).await?;
            owl_models::formats::load::manifest_mem_ledger(&draft_manifest, "draft-mem");
            owl_shared::log_info!("boot", "DFlash2 草稿装载(depth={depth};1.92B BF16 激活,sglang 对齐) {:.2}s", t_draft.elapsed().as_secs_f32());
            Some(Drafter::DFlash2(Arc::new(d2)))
        }
        _ => None,
    };
    // DFlash2 草稿 rope(θ 1e7;z-lab 检查点家族)
    let draft_rope = match &drafter {
        Some(Drafter::DFlash2(_)) => {
            let dr = Rope::new(262_144, 128, 128, 1.0e7)?;
            let ctx = owl_models::module::LoaderCtx {
                dtype: owl_models::contract::Dtype::F16,
                shard: 1,
                device_repack: false,
                verify: false,
                debug_tap: false,
            };
            owl_models::interpreters::eval_load(&dr, face, &dr.tables(), &ctx).await?;
            Some(Arc::new(dr))
        }
        _ => None,
    };
    // DFlash2 target taps 数(dflash 模式 = 5 层;notaps 退 last_hidden = 0)
    let dflash_tap_count = if mode == SpecMode::DFlash2 && !inp.notaps { 5 } else { 0 };
    Ok(SpecRuntime {
        depth,
        mode,
        drafter,
        draft_rope,
        dflash_tap_count,
        stats: SpecStats::default(),
        mem_dumped: false,
        mem_dumped2: false,
        hid_dumped: false,
        // B4 分母跨 turn 保鲜(初值 0 = 未采样,首窗口用保守兜底线)
        carry_ms: 0.0,
        carry_ctx: 0,
    })
}

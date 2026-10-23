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
use owl_shared::config::SpecDraft;

use crate::Result;
use owl_iface::contract::ModelError;

use crate::exec::mtp_window::MtpWindowCfg;

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
    /// **DFlash2 taps 的 target 层号(0-based,升序)** —— 单一事实源:
    /// HF 草稿 = 家族常量 `[5,19,33,47,61]`;GGUF 草稿 = `dflash.target_layers - 1`
    /// (GGUF 侧是 1-based)。**禁止在调用点写死**:错一层 = AL 静默塌陷。
    /// 无 spec / notaps ⇒ 空。
    pub dflash_tap_ids: std::sync::Arc<Vec<usize>>,
    /// 噪声块 mask token(GGUF 草稿可能未导出 → 家族兜底,见 DraftSpec)
    pub dflash_mask_token: u32,
    /// **草稿池几何**(层数, hkv, hd;**分片后**;P2.1-b)—— 池分配/预算的单源。
    /// `None` = 非 DFlash2 档(池不分配)
    pub draft_geom: Option<(usize, usize, usize)>,
    /// **草稿环窗口**(token;DFlash2 = 检查点 sliding_window / 配置覆盖)。
    /// 0 = 非 DFlash2。⚠️ 当前 main 的 NC ring 固定 2048（K15 前），
    /// 本字段保留给后续可配窗口；不要在热路径依赖它。
    #[allow(dead_code)]
    pub draft_window: usize,
    /// **草稿算子环境**(2026-10-19):池面之外的另一半几何事实 ——
    /// `head_dim`/`grp` 取**草稿自己的**(DFlash2 = 128 / hq÷hkv),其余字段
    /// (硬件档/池量化/分派旋钮)与引擎 env 同源。
    ///
    /// **为什么单列**:草稿与主模型共用同一个 face ⇒ `face.op_env()` 只有
    /// 目标几何(hd=256);草稿的 `attn.naive_nc`(hd=**128**)若按目标几何解析,
    /// F16 变体在几何门响亮拒,BF16 变体的 ShapeFamily 曾过宽 ⇒ **静默按错几何**
    /// (2026-10-10 悬案的真因)。MTP/无草稿档 = 引擎 env(同一份)。
    pub draft_env: owl_models::env::EnvProvider,
    /// 草稿**权重/激活/池**的 dtype(单源 `qwen35::draft_exec_dt` = 恒 BF16;
    /// F16 路在 propose 残差流上溢出 —— 定谳见该函数注)
    pub draft_dt: owl_models::contract::Dtype,
    /// MTP 草稿注意力窗口(0 = full;仅 MTP 有消费者)
    pub mtp_window: MtpWindowCfg,
    /// 自适应 depth（仅 MTP；`depth` = 上限，`depth_min` = 长 ctx 下限）
    pub adaptive_depth: bool,
    pub depth_min: usize,
    pub depth_switch_ctx: usize,
    /// T1b 滚动降档门（tokens/round；仅 adaptive 生效）
    pub adaptive_min_tpr: f32,
    /// DFlash2 lookup-augmented drafting（LABD；仅 DFlash2）
    pub lookup: bool,
    pub lookup_nmin: usize,
    pub lookup_nmax: usize,
    pub lookup_nstrong: usize,
    pub lookup_agree: usize,
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

impl SpecRuntime {
    /// 本档需要预捕获的 depth 列表（升序；无自适应 = 单档）。
    pub(crate) fn depth_candidates(&self) -> Vec<usize> {
        depth_candidates_of(self.depth, self.adaptive_depth, self.depth_min)
    }

    /// 按当前绝对位置选本轮 active depth。
    pub(crate) fn depth_for_pos(&self, pos: usize) -> usize {
        depth_for_pos_of(
            self.depth,
            self.adaptive_depth,
            self.depth_min,
            self.depth_switch_ctx,
            pos,
        )
    }
}

/// 纯函数（单测锚）：需要预捕获的 depth 列表。
fn depth_candidates_of(depth: usize, adaptive: bool, depth_min: usize) -> Vec<usize> {
    if depth == 0 {
        vec![0]
    } else if adaptive && depth_min > 0 && depth_min < depth {
        vec![depth_min, depth]
    } else {
        vec![depth]
    }
}

/// 纯函数（单测锚）：按绝对位置切档；越界/非法配置一律回上限，不 panic。
fn depth_for_pos_of(
    depth: usize,
    adaptive: bool,
    depth_min: usize,
    switch_ctx: usize,
    pos: usize,
) -> usize {
    if adaptive && depth_min > 0 && depth_min < depth && pos >= switch_ctx {
        depth_min
    } else {
        depth
    }
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
    /// 草稿族(auto/mtp/dflash2;配置显式选择)
    pub(crate) family: SpecDraft,
    /// DFlash2 草稿检查点目录(None = 无该族草稿器)
    pub(crate) dflash2_dir: Option<&'a str>,
    /// MTP 草稿检查点目录 / GGUF 单文件(检查点自带 mtp.* 或 `blk.N.nextn.*` 时 Some)
    pub(crate) mtp_dir: Option<&'a std::path::Path>,
    /// 主模型 spec(GGUF MTP 与 target 同维;路径档不读它)
    pub(crate) spec: &'a owl_models::model::ModelSpec,
    /// 哑草稿诊断位(无真草稿器时的 Dumb 档)
    pub(crate) dumb: bool,
    /// 草稿池 e4m3(B6 偷显存;AL 有损需 A/B)
    pub(crate) draft_kv_fp8: bool,
    /// 草稿环窗口(0 = auto = 检查点 sliding_window)
    pub(crate) draft_window: usize,
    /// MTP 草稿注意力窗口(tok;0 = 关)
    pub(crate) mtp_window: usize,
    /// MTP 草稿注意力 sink(tok;仅 window>0 生效)
    pub(crate) mtp_sink: usize,
    /// 窗口生效的最小绝对 ctx(tok;0 = 不设额外门槛)
    pub(crate) mtp_window_min_ctx: usize,
    /// 自适应 depth（仅 MTP；两档图族 boot 期预捕获）
    pub(crate) adaptive_depth: bool,
    /// 自适应档下限 depth
    pub(crate) depth_min: usize,
    /// 切档 ctx 阈值（pos >= 阈值用下限）
    pub(crate) depth_switch_ctx: usize,
    /// T1b 滚动降档门（tokens/round；仅 adaptive 生效）
    pub(crate) adaptive_min_tpr: f32,
    /// DFlash2 lookup-augmented drafting（LABD；仅 DFlash2）
    pub(crate) lookup: bool,
    pub(crate) lookup_nmin: usize,
    pub(crate) lookup_nmax: usize,
    pub(crate) lookup_nstrong: usize,
    pub(crate) lookup_agree: usize,
    /// **target 侧共享事实**(词表/层数/hidden;草稿不自带 —— 仅 GGUF 草稿
    /// 的声明校验与码本行数用)
    pub(crate) target: owl_models::formats::gguf::TargetFacts,
    /// DFlash2 taps 退回 last_hidden(排查开关)
    pub(crate) notaps: bool,
    /// **本 rank 的 TP 计划**(P2.1-a;`None`/`is_single()` = 单卡 ⇒ 逐字节旧行为)。
    /// 草稿族双 rank 化的**唯一形状/装载单源**:草稿 spec 按它切、草稿权重按它装、
    /// 草稿 rope 按它分片 —— 与主干同 TpPlan(vLLM/SGLang 形态:同 TP group)。
    pub(crate) tp: Option<owl_models::plan::tp::TpPlan>,
    /// **装载旋钮**(`[load]`;与主干 loader 同源 —— 一处配置两处消费)
    pub(crate) load: owl_shared::config::LoadCfg,
    /// **引擎 env 快照**(已合入目标几何/硬件档;草稿环境以它为基底 ——
    /// 只覆盖草稿自己的几何,见 [`SpecRuntime::draft_env`])
    pub(crate) base_env: owl_models::env::EnvProvider,
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
    } else {
        match inp.family {
            SpecDraft::Mtp => {
                if inp.mtp_dir.is_none() {
                    return Err(ModelError::Msg(
                        "spec.draft = \"mtp\" 但当前检查点没有 mtp.* / blk.N.nextn.* 头(27B AWQ/带 MTP 的 GGUF 自带;0.8B/外部草稿档请用 dflash2)"
                            .into(),
                    ));
                }
                SpecMode::Mtp
            }
            SpecDraft::Dflash2 => {
                if inp.dflash2_dir.is_none() {
                    return Err(ModelError::Msg(
                        "spec.draft = \"dflash2\" 但 model.dflash2_dir 为空".into(),
                    ));
                }
                SpecMode::DFlash2
            }
            SpecDraft::Auto => {
                if inp.dflash2_dir.is_some() {
                    SpecMode::DFlash2
                } else if inp.mtp_dir.is_some() {
                    SpecMode::Mtp
                } else if inp.dumb {
                    SpecMode::Dumb
                } else {
                    owl_shared::log_info!("boot", "spec.depth>0 但无草稿器(非 mtp/dflash2 检查点)→ C7 回落 DecodeBatch");
                    SpecMode::Off
                }
            }
        }
    };
    // **深度 = 配置唯一真源**(2026-10-21):此前这里按族硬钳(DFlash2 `min(7)`、
    // MTP `min(3)`),MTP 因此只能跑 3 步 —— 而实测第 4~7 步接受率仍 70~80%
    // (depth 7 比 3 在 8k 上快 ~10%、512 上快 ~21%)。上限由 `[spec] depth`
    // 的校验面统一挡(`SPEC_DEPTH_MAX`,结构性行容量),此处不再截断。
    let depth = if mode == SpecMode::Off { 0 } else { inp.depth_raw };
    // **MTP 窗口只服务 MTP**(2026-10-21):非 MTP/Off 写非 0 = boot 拒,
    // 不静默忽略;window=0 时 sink 无消费者(允许缺省 64)。
    let mtp_window = MtpWindowCfg {
        window: inp.mtp_window,
        sink: inp.mtp_sink,
        min_ctx: inp.mtp_window_min_ctx,
    };
    // **自适应 depth 的 boot 门**（2026-10-21）：只在 MTP 上成立；两档
    // 都必须在启动时预捕获，运行期只换图引用。配置面已做形状门，这里
    // 再按**实际裁决出的模式**拒一次（auto 可能落到非 MTP）。
    if inp.adaptive_depth {
        if mode != SpecMode::Mtp {
            return Err(ModelError::Msg(format!(
                "[spec] adaptive_depth 只服务 MTP，但当前模式 = {:?}（depth={}, draft={:?}）",
                mode, depth, inp.family
            )));
        }
        if inp.depth_min == 0 || inp.depth_min >= depth {
            return Err(ModelError::Msg(format!(
                "[spec] adaptive_depth 需 0 < depth_min < depth（实得 depth_min={}, depth={}）",
                inp.depth_min, depth
            )));
        }
        if inp.depth_min > owl_shared::config::SPEC_DEPTH_MAX {
            return Err(ModelError::Msg(format!(
                "[spec] depth_min={} 超结构上限 {}",
                inp.depth_min,
                owl_shared::config::SPEC_DEPTH_MAX
            )));
        }
        if inp.depth_switch_ctx == 0 {
            return Err(ModelError::Msg(
                "[spec] adaptive_depth 需 depth_switch_ctx > 0".into(),
            ));
        }
        if !inp.adaptive_min_tpr.is_finite() || inp.adaptive_min_tpr <= 0.0 {
            return Err(ModelError::Msg(format!(
                "[spec] adaptive_min_tpr 必须 > 0 且有限（实得 {}）",
                inp.adaptive_min_tpr
            )));
        }
        owl_shared::log_info!(
            "boot",
            "spec adaptive depth: max={} min={} switch_ctx={} min_tpr={}（两档 verify/fold/propose 图族预捕获）",
            depth,
            inp.depth_min,
            inp.depth_switch_ctx,
            inp.adaptive_min_tpr
        );
    }
    if inp.lookup {
        if mode != SpecMode::DFlash2 {
            return Err(ModelError::Msg(format!(
                "[spec] lookup 只服务 DFlash2，但当前模式 = {:?}（depth={}, draft={:?}）",
                mode, depth, inp.family
            )));
        }
        if depth == 0 {
            return Err(ModelError::Msg(
                "[spec] lookup = true 但 depth = 0（无投机可查）".into(),
            ));
        }
        if inp.lookup_nmin == 0 || inp.lookup_nmax < inp.lookup_nmin {
            return Err(ModelError::Msg(format!(
                "[spec] lookup 阈值非法：nmin={} nmax={}（需 1 <= nmin <= nmax）",
                inp.lookup_nmin, inp.lookup_nmax
            )));
        }
        if inp.lookup_nstrong > inp.lookup_nmax {
            return Err(ModelError::Msg(format!(
                "[spec] lookup_nstrong={} 必须 <= lookup_nmax={}",
                inp.lookup_nstrong, inp.lookup_nmax
            )));
        }
        if inp.lookup_nmax > 128 || inp.lookup_agree > owl_shared::config::SPEC_DEPTH_MAX {
            return Err(ModelError::Msg(format!(
                "[spec] lookup 越界：nmax={}（上限 128）agree={}（上限 {}）",
                inp.lookup_nmax,
                inp.lookup_agree,
                owl_shared::config::SPEC_DEPTH_MAX
            )));
        }
        owl_shared::log_info!(
            "boot",
            "DFlash2 lookup: nmin={} nmax={} nstrong={} agree={}（host 后缀查找 + 草稿头融合）",
            inp.lookup_nmin,
            inp.lookup_nmax,
            inp.lookup_nstrong,
            inp.lookup_agree
        );
    }
    if mtp_window.active() {
        if mode != SpecMode::Mtp {
            return Err(ModelError::Msg(format!(
                "[spec] mtp_window={} 只服务 MTP,但当前模式 = {:?}(depth={}, draft={:?});请关 window 或改用 mtp 档",
                mtp_window.window, mode, depth, inp.family
            )));
        }
        if mtp_window.sink >= mtp_window.window {
            return Err(ModelError::Msg(format!(
                "[spec] mtp_sink={} 必须小于 mtp_window={}",
                mtp_window.sink, mtp_window.window
            )));
        }
        owl_shared::log_info!(
            "boot",
            "MTP window: window={} sink={} min_ctx={} (page 对齐在 StatePool 分配后校验/打印)",
            mtp_window.window,
            mtp_window.sink,
            mtp_window.min_ctx
        );
    }
    if mode != SpecMode::Off {
        owl_shared::log_info!("boot", "spec 模式 = {:?} depth={depth}(注:spec 轮恒 greedy,采样 parity 挂 D4 device accept)",
            mode
        );
    }
    // 草稿器装载(C7;embed/lm_head 均共享 target)
    // 草稿 dtype:缺省 BF16(量化族/无草稿);DFlash2 分支按**检查点事实**改写
    // DFlash2 草稿池 ring(2026-10-22):检查点 config `sliding_window=2048`,
    // NC 核只读最近 2048 行;池只留窗口+自块(页对齐)。MTP/线性档为 0。
    let dflash_ring_cap =
        owl_models::layers::dflash2::draft_ring_slots(inp.base_env.kv.page.max(1));

    let mut draft_dt = owl_models::contract::Dtype::BF16;
    // DFlash2 草稿事实(HF / GGUF 两源共同填写;单一事实源)
    let mut dflash_tap_ids: std::sync::Arc<Vec<usize>> = std::sync::Arc::new(Vec::new());
    let mut dflash_mask_token = owl_models::layers::dflash2::MASK_TOKEN_ID;
    // (max_pos, rotary_dim, theta)
    let mut draft_rope_facts: Option<(usize, usize, f32)> = None;
    // (layers, hkv, hd, hq)
    let mut draft_geom_raw: Option<(usize, usize, usize, usize, usize)> = None;
    let drafter = match mode {
        SpecMode::Mtp => {
            let dir = inp.mtp_dir.ok_or_else(|| ModelError::EngineInvariant { what: "Mtp 模式缺 mtp_dir(模式裁决与输入不一致)".into() })?;
            // MTP 头 = 检查点内 BF16 权重 / GGUF Q6_K 块,经装载源归一到 F16;
            // 草稿池也按目标模型 dtype(F16)分配。**不是** BF16 草稿池。
            draft_dt = owl_models::contract::Dtype::F16;
            let is_gguf = dir.extension().and_then(|s| s.to_str()) == Some("gguf");
            let spec = if is_gguf {
                // GGUF 单文件:MTP 与 target 同维(同一 spec);块号 = 解码层数
                shard_draft_plan(inp.spec.clone(), inp.tp)?
            } else {
                // 声明构造走起点事实(同一份 27B 检查点的 spec;报告已由主模型
                // 装载路径上屏,此处不重报);装载走通用入口
                let (spec, _rep) = owl_models::model::ModelSpec::from_ckpt(
                    dir,
                    owl_models::formats::spec::SpecSource::family(
                        &owl_models::formats::qwen35::qwen3_8_27b(),
                    ),
                )?;
                // P2.1-a:草稿与主干同 TpPlan(单卡 ⇒ `sharded` 恒等)
                shard_draft_plan(spec, inp.tp)?
            };
            // GGUF 档:MTP 头走**原生块**(逐键计划表)而不是 host 解到 F16 —
            // 27B 头 = Q6_K 12 张 + Q8_0 3 张 ⇒ 0.35 GB(对 0.86 GB),
            // 省下的 0.5 GB 直接还给 KV 池(fp8 池下 ≈ +16k 上下文),
            // 且草稿 GEMM 不再付解量化。
            // ⚠️ TP 档暂走 F16 头:原生块的 `eh_proj` 半拆只实现了**单卡全量**
            // (分片下的行/列窗口是另一套块算术);TP 需要的形状由 `shard_draft_plan` 给。
            let tp1 = inp.tp.as_ref().map_or(true, |p| p.world() == 1);
            let (mtp, native_blocks) = if is_gguf && tp1 {
                let g = owl_models::formats::gguf::Gguf::open(dir)?;
                let table = owl_models::formats::gguf::DraftGgufPlan::new(&g);
                let conv = owl_models::formats::gguf::GgufConvention::new();
                let plan = owl_models::module::PlanSource::keyed(
                    &table,
                    &conv,
                    owl_models::module::SourceFacts::GGUF,
                );
                let mtp = owl_models::layers::mtp::MtpPredictor::from_spec_gguf(
                    &spec,
                    spec.layer_types.len(),
                    plan,
                );
                // 表里没解析到的键(型未移植/形状不符)会静默回落 F16 —— 必须上屏
                use owl_models::module::PlanTable as _;
                let un = table.take_unresolved();
                if !un.is_empty() {
                    owl_shared::log_warn!(
                        "boot",
                        "MTP 头有 {} 个键回落 F16(原生块表未解析):{:?}",
                        un.len(),
                        un
                    );
                }
                (mtp, true)
            } else {
                (
                    owl_models::layers::mtp::MtpPredictor::from_spec(&spec),
                    false,
                )
            };
            let mut opts = draft_load_opts(face, owl_models::contract::Dtype::F16, inp.tp, &inp.load);
            // 原生块路径:源直接给压缩块字节(槽 U8),不再走 host 解量化
            opts.gguf_dequant = is_gguf && !native_blocks;
            owl_models::formats::load::load_from_dir(dir, face, &mtp, opts).await?;
            owl_shared::log_info!(
                "boot",
                "MTP 草稿头装载(spec depth={depth};{};权重={})",
                if is_gguf { "GGUF" } else { "HF/AWQ" },
                if native_blocks { "原生块" } else { "F16" }
            );
            Some(Drafter::Mtp(Arc::new(mtp)))
        }
        SpecMode::DFlash2 => {
            // (draft_dt 在外层声明:下面按检查点事实改写)
            let dir = inp.dflash2_dir.ok_or_else(|| ModelError::EngineInvariant { what: "DFlash2 模式缺 dflash2_dir(模式裁决与输入不一致)".into() })?;
            let t_draft = std::time::Instant::now();
            // 草稿装载 dtype = **草稿执行 dtype 单源**(恒 BF16;F16 路在
            // propose 残差流上溢出 —— 定谳见 `qwen35::draft_exec_dt` 注)
            draft_dt = owl_models::formats::qwen35::draft_exec_dt();
            let path = std::path::Path::new(dir);
            // **草稿源枚举**(2026-10-20 DFlash2-GGUF 线):目录 = HF 检查点;
            // 文件 = GGUF 草稿。一个配置键两种源 —— 由**路径形态**判别,
            // 不新增配置项(`site.draft_dir` 原样可用)。
            if path.is_file() {
                if let Some(tp) = inp.tp {
                    if !tp.is_single() {
                        return Err(ModelError::Msg(
                            "GGUF 草稿 v1 只支持单卡(TP 草稿分片挂 P4;请关 tp_draft 或换 HF 草稿)"
                                .into(),
                        ));
                    }
                }
                let boot = owl_models::formats::gguf::GgufDraftBoot::open(path, inp.target)?;
                let gs = &boot.spec;
                owl_shared::log_info!(
                    "boot",
                    "GGUF 草稿:{} · {} 层 {}H/{}KV/{}D · taps {:?}(GGUF 1-based 原值已归一) · \
                     block {} / conv {}·{} / selector rank {} top {} · mask {}",
                    path.display(),
                    gs.layers, gs.hq, gs.hkv, gs.hd, gs.tap_ids,
                    gs.block, gs.conv_taps, gs.conv_group, gs.rank, gs.top_k, gs.mask_token
                );
                let d2 = boot.draft(draft_dt, inp.draft_kv_fp8)?;
                let opts = draft_load_opts(face, draft_dt, inp.tp, &inp.load);
                let manifest = boot.load(face, &d2, opts).await?;
                owl_models::formats::load::manifest_mem_ledger(&manifest, "draft-mem");
                dflash_tap_ids = std::sync::Arc::new(gs.tap_ids.clone());
                dflash_mask_token = gs.mask_token;
                draft_rope_facts = Some((gs.rope_max_pos, gs.hd, gs.rope_theta));
                draft_geom_raw = Some((gs.layers, gs.hkv, gs.hd, gs.hq,
                    gs.sliding_window.unwrap_or(owl_models::layers::dflash2::NC_WINDOW)));
                owl_shared::log_info!(
                    "boot",
                    "DFlash2 GGUF 草稿装载(depth={depth};块字节驻留 + 码本反量化) {:.2}s",
                    t_draft.elapsed().as_secs_f32()
                );
                Some(Drafter::DFlash2(Arc::new(
                    d2.with_ring_cap(dflash_ring_cap).with_block_size(depth + 1),
                )))
            } else {
                // 声明:草稿检查点自己的 config(组件档)+ 起点格式探测 → fc 臂;
                // 装载走通用入口(BF16 全程;设备重排 = 能力自述裁决)
                let d2 = match inp.tp {
                    Some(tp) if !tp.is_single() => {
                        owl_models::formats::qwen35::dflash2_draft_sharded(path, inp.draft_kv_fp8, tp)?
                    }
                    _ => owl_models::formats::qwen35::dflash2_draft(path, inp.draft_kv_fp8)?,
                };
                let opts = draft_load_opts(face, draft_dt, inp.tp, &inp.load);
                let draft_manifest =
                    owl_models::formats::load::load_from_dir(path, face, &d2, opts).await?;
                owl_models::formats::load::manifest_mem_ledger(&draft_manifest, "draft-mem");
                dflash_tap_ids =
                    std::sync::Arc::new(owl_models::layers::dflash2::DRAFT_TAP_IDS.to_vec());
                dflash_mask_token = owl_models::layers::dflash2::MASK_TOKEN_ID;
                owl_shared::log_info!("boot", "DFlash2 草稿装载(depth={depth};1.92B BF16 激活,sglang 对齐) {:.2}s", t_draft.elapsed().as_secs_f32());
                Some(Drafter::DFlash2(Arc::new(
                    d2.with_ring_cap(dflash_ring_cap).with_block_size(depth + 1),
                )))
            }
        }
        _ => None,
    };
    // DFlash2 草稿 rope(θ 1e7;z-lab 检查点家族)
    let draft_rope = match &drafter {
        Some(Drafter::DFlash2(_)) => {
            let (max_pos, dim, theta) = draft_rope_facts.unwrap_or((262_144, 128, 1.0e7));
            let dr = Rope::new(max_pos, dim, dim, theta)?;
            let (shard, slice_id) = match inp.tp {
                Some(tp) => (tp.world(), tp.rank()),
                None => (1, 0),
            };
            let ctx = owl_models::module::LoaderCtx {
                dtype: owl_models::contract::Dtype::F16,
                shard,
                slice_id,
                device_repack: false,
                verify: false,
                debug_tap: false,
                staging_mb: inp.load.staging_mb,
                dump: false,
            };
            owl_models::interpreters::eval_load(&dr, face, &dr.tables(), &ctx).await?;
            Some(Arc::new(dr))
        }
        _ => None,
    };
    // DFlash2 target taps 数(dflash 模式 = 5 层;notaps 退 last_hidden = 0)
    let dflash_tap_count = if mode == SpecMode::DFlash2 && !inp.notaps {
        dflash_tap_ids.len()
    } else {
        0
    };
    if mode == SpecMode::DFlash2 && !inp.notaps {
        // 层号合法性(启动期判死:`tapped_hidden` 的结构保证是"层号必在域内")
        if let Some(&last) = dflash_tap_ids.last() {
            if last >= inp.target.layers {
                return Err(ModelError::Msg(format!(
                    "DFlash2 taps {dflash_tap_ids:?} 越界(target {} 层)",
                    inp.target.layers
                )));
            }
        }
    }
    // 草稿池几何(**分片后**;P2.1-b):从检查点声明 + 本 rank 计划算(不再用写死的 const)
    let (draft_geom, draft_window, draft_env) = match &drafter {
        Some(Drafter::Mtp(_)) => {
            // MTP 草稿链池 = 独立页池,**存储 dtype 由 `[spec] draft_kv_fp8` 定**
            // (2026-10-21:此前恒 F16 —— 那条 864 MiB@192k 就是这么白吃的):
            // - fp8 ⇒ e4m3(U32 扁平字节账;每 token 2 KiB)
            // - 否则 ⇒ `draft_dt`(F16;每 token 4 KiB)
            // env 必须**与池同档**:池 F16 而 env 说 fp8(或反之)= 读写核错位
            // ⇒ 草稿全垃圾(2026-10-13 守卫就抓过这条)。
            let mut env = inp.base_env;
            env.quant = owl_models::module::QuantPlan::F16;
            env.kv.quant = if inp.draft_kv_fp8 {
                owl_models::env::KvQuant::Fp8E4M3
            } else {
                owl_models::env::KvQuant::None
            };
            env.kv.dtype = if inp.draft_kv_fp8 {
                owl_models::contract::Dtype::U32
            } else {
                draft_dt
            };
            // MTP extend(T=m+1≤4)在层内走 v2 伪序列臂:草稿池 F16、
            // 每行一条 MTP 页链、长 ctx 下远比 chunked prefill 的
            // per-query 全历史扫描便宜。只影响草稿 env,主 verify 仍按
            // 自己的 dispatch.verify_v2/FI 档。
            env.attn.verify_arm = owl_shared::config::VerifyArm::V2;
            (None, 0, env)
        }
        Some(Drafter::DFlash2(_)) => {
            // GGUF 草稿:几何已在开箱时定(不重开容器);HF:从 config + 本 rank 计划算
            let g = match draft_geom_raw {
                Some((layers, hkv, hd, hq, window)) => {
                    owl_models::formats::qwen35::DraftGeom { layers, hq, hkv, hd, window }
                }
                None => {
                    let dir = inp.dflash2_dir.ok_or_else(|| {
                        ModelError::EngineInvariant { what: "DFlash2 模式缺 dflash2_dir".into() }
                    })?;
                    let tp = inp.tp.unwrap_or_else(owl_models::plan::tp::TpPlan::single);
                    owl_models::formats::qwen35::dflash2_draft_geom(std::path::Path::new(dir), tp)?
                }
            };
            // 草稿算子环境 = 引擎 env + **草稿几何**(几何门对账的单一来源;
            // 主模型路径不消费本字段 ⇒ 逐字节不变)
            let mut env = inp.base_env;
            env.kv.head_dim = g.hd;
            env.kv.grp = g.grp();
            // **草稿池的存储真值**:主池 fp8 不代表草稿池 fp8。草稿池
            // 默认 BF16;`draft_kv_fp8` 才切 e4m3。env 必须跟着走,
            // 否则 KernelSelectionEnv 会把主池的 fp8 误报给草稿 picker。
            env.kv.quant = if inp.draft_kv_fp8 {
                owl_models::env::KvQuant::Fp8E4M3
            } else {
                owl_models::env::KvQuant::None
            };
            env.kv.dtype = draft_dt;
            // **草稿权重 plan 是独立事实**:DFlash2 量化族 = W4A16(非 AWQ),
            // 未量化族 = F16。若沿用主干的 W4A8 tag,草稿 `marlin_gemm_w4a16_*`
            // 会在 picker 的 tag 过滤处被拒(2026-10-21 W4A8 组合实测)。
            let dir = inp.dflash2_dir.ok_or_else(|| ModelError::EngineInvariant {
                what: "DFlash2 模式缺 dflash2_dir(draft env 构造)".into(),
            })?;
            env.quant = if owl_models::formats::qwen35::draft_is_quantized(
                std::path::Path::new(dir),
            )? {
                owl_models::module::QuantPlan::W4A16
            } else {
                owl_models::module::QuantPlan::F16
            };
            // 环窗口:配置覆盖优先(0=auto=检查点);当前 NC 核只实例化 2048
            let window = if inp.draft_window > 0 { inp.draft_window } else { g.window };
            let cap = owl_models::layers::dflash2::NC_WINDOW;
            if window != cap {
                return Err(ModelError::Msg(format!(
                    "[spec] draft_window={} 与草稿 NC 核当前窗口 {} 不一致:环容量必须等于核窗口。\
                     4k/8k 需先给 `owl_naive_attn_nc_*` 加运行时 window(或模板实例);\
                     当前只支持 {}。",
                    window, cap, cap
                )));
            }
            (Some((g.layers, g.hkv, g.hd)), window, env)
        }
        _ => (None, 0, inp.base_env),
    };
    Ok(SpecRuntime {
        depth,
        mode,
        drafter,
        draft_rope,
        dflash_tap_count,
        dflash_tap_ids,
        dflash_mask_token,
        draft_geom,
        draft_window,
        draft_env,
        draft_dt,
        mtp_window,
        adaptive_depth: inp.adaptive_depth,
        depth_min: inp.depth_min,
        depth_switch_ctx: inp.depth_switch_ctx,
        adaptive_min_tpr: inp.adaptive_min_tpr,
        lookup: inp.lookup,
        lookup_nmin: inp.lookup_nmin,
        lookup_nmax: inp.lookup_nmax,
        lookup_nstrong: inp.lookup_nstrong,
        lookup_agree: inp.lookup_agree,
        stats: SpecStats::default(),
        mem_dumped: false,
        mem_dumped2: false,
        hid_dumped: false,
        // B4 分母跨 turn 保鲜(初值 0 = 未采样,首窗口用保守兜底线)
        carry_ms: 0.0,
        carry_ctx: 0,
    })
}


/// **草稿装载 opts**(P2.1-a):与 [`LoadOpts::new`] 同律,只把分片协议换成
/// 本 rank 的(`tp = None`/单卡 ⇒ 与旧路径逐字节同物)。
fn draft_load_opts<D: DeviceClient>(
    face: &D,
    dtype: owl_models::contract::Dtype,
    tp: Option<owl_models::plan::tp::TpPlan>,
    load: &owl_shared::config::LoadCfg,
) -> owl_models::formats::load::LoadOpts {
    let mut base = owl_models::formats::load::LoadOpts::new(face, dtype);
    // **与主干 loader 同源**(一处配置两处消费)
    base.verify = load.verify;
    base.debug_tap = load.debug_tap;
    base.staging_mb = load.staging_mb;
    base.dump = load.dump;
    if let Ok(Some(dev)) = load.repack_device() {
        base.device_repack = Some(dev);
    }
    match tp {
        Some(tp) if !tp.is_single() => owl_models::formats::load::LoadOpts {
            shard: tp.world(),
            slice_id: tp.rank(),
            ..base
        },
        _ => base,
    }
}

/// **草稿 spec 分片**(P2.1-a;纯函数委托 —— 形状单源在
/// `formats::qwen35::shard_draft_spec`,此处只做 `Option` 适配)。
fn shard_draft_plan(
    spec: owl_models::model::ModelSpec,
    tp: Option<owl_models::plan::tp::TpPlan>,
) -> Result<owl_models::model::ModelSpec> {
    match tp {
        Some(tp) if !tp.is_single() => owl_models::formats::qwen35::shard_draft_spec(spec, tp),
        _ => Ok(spec),
    }
}


#[cfg(test)]
mod adaptive_depth_tests {
    use super::*;

    #[test]
    fn candidates_and_switch_are_monotone() {
        assert_eq!(depth_candidates_of(3, false, 3), vec![3]);
        assert_eq!(depth_candidates_of(7, true, 3), vec![3, 7]);
        assert_eq!(depth_candidates_of(0, true, 3), vec![0]);
        assert_eq!(depth_for_pos_of(7, true, 3, 49152, 0), 7);
        assert_eq!(depth_for_pos_of(7, true, 3, 49152, 49151), 7);
        assert_eq!(depth_for_pos_of(7, true, 3, 49152, 49152), 3);
        assert_eq!(depth_for_pos_of(7, false, 3, 49152, 999999), 7);
        // 非法 min >= max：不切，保持上限（boot 已拒，此处只证明纯函数不 panic）
        assert_eq!(depth_for_pos_of(3, true, 3, 49152, 999999), 3);
    }
}

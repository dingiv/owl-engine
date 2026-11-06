//! **TP2 双 rank 装配面**(S4 ⑤-K2;world = 2)。
//!
//! 设计出处:`docs/arch/TP2-施工设计.md` §14.2(S4 分刀)/ §15.6(双 actor 形态)。
//!
//! 职责四件(顺序有语义,勿重排):
//! 1. **分档判定**(`owl_cuda::comm::peer::PeerPlan`):UUID + BAR1 + P2P 能力
//!    (混插/不支持 ⇒ **响亮拒**,D5)⇒ 单笔上限与 staging 额度核算;
//! 2. **staging 的 VMM 对称握手**(`peer_alloc` → fd 交叉 → `peer_import`):
//!    **必须在建图之前完成** —— 捕获期拒三个 `Peer*` 命令(护栏在
//!    server 侧;这条纪律的调用方就是本模块);
//! 3. **每 rank 设备侧装配**(`boot_rank`:分片 spec/池/图,`TpCtx` 注入);
//! 4. **收尾**(`shutdown`:先卸图(图租约钉块)后 `PeerRelease` —— 账本闭合)。
//!
//! 两侧**同一份代码**(对称);每 rank 装载走 `ModelLoader::slice(id, world)`
//! —— 确定性分片协议 ⇒ 零协调(装载本身在调用方手里)。

use owl_cuda::comm::ar::StagingLayout;
use owl_cuda::comm::peer::{PeerCaps, PeerPlan};
use owl_cuda::{DeviceClient, GpuClient};
use owl_iface::contract::{Bytes, Dtype, ModelError, Shape};
use owl_iface::handle::DeviceHandle;
use owl_models::model::ModelSpec;
use owl_models::module::TpCtx;
use owl_models::plan::tp::TpPlan;
use owl_models::tokenizer::Tokenizer;
use owl_shared::config::TpCommMode;

use crate::engine::{boot_rank, Engine, EngineConfig, RankDev};
use crate::graph_plan::PlanOutcome;
use crate::loader::LoadedModel;
use crate::running::RunningEngine;

type Result<T> = std::result::Result<T, ModelError>;

fn loud<T>(msg: impl Into<String>) -> Result<T> {
    Err(ModelError::Msg(msg.into()))
}

/// **TP 装配的纯配置护栏**（无设备动作；`assemble_with` 在 staging 握手前
/// 先调一次，`assemble_with_staging` 仍保留为直接调用者的兜底）。
fn check_assemble_guards(cfg: &EngineConfig, loadeds: &[LoadedModel]) -> Result<()> {
    // ① 投机草稿:verify/fold/propose 族图与草稿 prefill extend 是**双 rank
    //    未接线面**(S6+);
    // ② prefill 桶族已退役(2026-10-21),不再有该面。
    // ③ 读数类探针:prefill 入队半里的 dtoh/sync 会撞 AR 自旋(见 on_peer 契约)。
    let k = &cfg.knobs;
    if k.draft.depth != 0 && !k.draft.tp_draft {
        return loud(format!(
            "TP v1 不支持投机草稿(draft.depth = {};双 rank 草稿族未接线)。\
             若要走 P2.1 的草稿双 rank 路,显式开 `[spec] tp_draft = true`\
             (草稿 spec/权重按本 rank 计划分片 + 草稿池几何按分片 hkv)",
            k.draft.depth
        ));
    }
    let bad_probes: Vec<&str> = [("dflash_probe", k.probes.dflash_probe)]
        .into_iter()
        .filter(|(_, on)| *on)
        .map(|(n, _)| n)
        .collect();
    if !bad_probes.is_empty() {
        return loud(format!(
            "TP v1 不支持读数类探针 {bad_probes:?}(入队半必须零同步;见 on_peer 契约)"
        ));
    }
    // ④ 量化档:两 rank 的量化方案必须同档(混档 = 权重语义不同)。
    if loadeds.len() == 2 && loadeds[0].quant_plan() != loadeds[1].quant_plan() {
        return loud(format!(
            "TP 两 rank 量化方案不一致(rank0={:?} / rank1={:?})—— 权重语义不同,拒绝",
            loadeds[0].quant_plan(),
            loadeds[1].quant_plan()
        ));
    }
    Ok(())
}

// ============================================================================
// §1 staging(双块 + 布局量)
// ============================================================================

/// **装配选项**(2026-10-18;`TpEngine::assemble_with` 传入)。
///
/// 为什么是**显式选项**而不是 env:owl 的显式依赖律 —— 引擎内部零 env 读取,
/// 开关由入口(server config / 测试用例)经结构体显式传入。
#[derive(Debug, Clone, Copy, Default)]
pub struct TpOpts {
    /// **fp8 载荷档**(§47 第二步;2026-10-19):AR 载荷走 e4m3(载荷 1.031×)。
    /// 由入口(server config)显式传入 —— 引擎内部零 env 读取。
    pub ar_f8: bool,
    /// **v1 两核融合档**(§11.2;2026-10-19):AR 一核做完两段(只非 fp8 载荷)。
    /// 由入口显式传入(缺省关);两 rank 必须同档。
    pub ar_fused: bool,
    /// **barrier-only AR-2**(2026-10-23):小包 graph 专用;与 fused 同线格式。
    /// 复用 v1 的 produced/consumed 相对游标 + 双槽,但去掉尾部 rendezvous、
    /// wait_done/blocks_done 与全线程 system fence。peer 只读别名即可。
    pub ar2: bool,
    /// **卡间通信通道**(2026-11-04):`P2p` = VMM peer alias(现役);
    /// `Host` = pinned host-mapped 中转(无 P2P 驱动兜底)。由入口显式传入。
    pub comm: TpCommMode,
}

/// staging 的双块与布局量(布局算术单源 = `owl_cuda::comm::ar::StagingLayout` ——
/// 本模块不重写 header/slot 偏移)。
#[derive(Debug, Clone)]
pub struct TpStaging {
    /// 本端 staging(own;stage 核写它 / reduce 核在此发布 `consumed`)
    own: Bytes,
    /// 对端 staging 的跨卡别名(peer;stage 读 `consumed` / reduce 读 slots)
    peer: Bytes,
    /// **fp8 载荷档**(与 `TpOpts::ar_f8` 同值;`ctx()` 把它写进 `TpCtx`)
    ar_f8: bool,
    /// **v1 两核融合档**(与 `TpOpts::ar_fused` 同值;§11.2)
    ar_fused: bool,
    /// **barrier-only AR-2**(与 `TpOpts::ar2` 同值;2026-10-23)
    ar2: bool,
    /// **host-mapped 通道**(与 `TpOpts::comm` 同值;2026-11-04)
    host_mode: bool,
    /// host staging 的 host 视图(base;0 = P2P)。header 初始化/诊断直读用。
    host_own: u64,
    /// 对端 host staging 的 host 视图(base;0 = P2P)
    host_peer: u64,
    /// host f8 CE 的**设备侧量化暂存**(payload → fp8 + scales;None = P2P/raw host)
    host_quant: Option<Bytes>,
    /// host f8 CE 的**量化专用 dummy header**(64B 设备块;不参与真实协议)
    host_dummy: Option<Bytes>,
    pub(crate) layout: StagingLayout,
}

impl TpStaging {
    /// 本端 staging 块
    pub fn own(&self) -> &Bytes {
        &self.own
    }
    /// 对端 staging 的别名块(只读槽)
    pub fn peer(&self) -> &Bytes {
        &self.peer
    }
    /// slot 字节数(两侧同参)
    pub fn slot_bytes(&self) -> u64 {
        self.layout.slot_bytes
    }
    /// 整块字节数(= VMM 授予额度 / host 分配额度)
    pub fn total_bytes(&self) -> u64 {
        self.layout.total_bytes()
    }

    /// 是否 host-mapped 通道
    pub fn host_mode(&self) -> bool {
        self.host_mode
    }

    /// 本端 host 视图(0 = P2P)
    pub fn host_own(&self) -> Option<u64> {
        (self.host_own != 0).then_some(self.host_own)
    }

    /// 对端 host 视图(0 = P2P)
    pub fn host_peer(&self) -> Option<u64> {
        (self.host_peer != 0).then_some(self.host_peer)
    }

    fn ctx(&self, plan: TpPlan) -> TpCtx {
        TpCtx {
            plan,
            own: self.own.clone(),
            peer: self.peer.clone(),
            slot_bytes: self.layout.slot_bytes,
            ar_f8: self.ar_f8,
            ar_fused: self.ar_fused,
            ar2: self.ar2,
            host_own: self.host_own,
            host_quant: self.host_quant.clone(),
            host_dummy: self.host_dummy.clone(),
        }
    }
}

/// staging header(64B)字节(**布局权威 = `owl_cuda::comm::ar::hdr` 常量**;与 cuda 门
/// `write_hdr` 同法):`produced`/`consumed` 归零 + 自描述 `slot_bytes`/`n_slots` + `magic`(核的串块/未初始化校验门)。
fn staging_header_bytes(layout: StagingLayout) -> Vec<u8> {
    let mut b = vec![0u8; owl_cuda::comm::ar::HEADER_BYTES as usize];
    b[16..20].copy_from_slice(&(layout.slot_bytes as u32).to_le_bytes());
    b[20..24].copy_from_slice(&layout.slots.to_le_bytes());
    b[24..28].copy_from_slice(&owl_cuda::comm::ar::MAGIC.to_le_bytes());
    // ★ **前置清单②(2026-10-17 显式化)**:`produced` / `consumed` /
    // `wait_done` / `blocks_done` **四格必须全 0**。这里靠"Vec 初始化为零 +
    // 只写 0..28"成立 —— 但那是**隐含**的,故补一条断言把它钉住:
    // `wait_done` 若留非 0 残值且 ≥ 本轮 want,非 0 号 block 会**跳过两级等待**
    // ⇒ 静默读到未就绪的对端数据(核**无法自检**,只能在这里保证)。
    debug_assert!(
        b.len() == owl_cuda::comm::ar::HEADER_BYTES as usize
            && b[28..].iter().all(|&x| x == 0),
        "staging header 的 28..64(含 wait_done@32 / blocks_done@40)必须全 0;\
         核无法自检这一条 —— 残值会导致非 0 号 block 跳过等待、静默读到未就绪数据"
    );
    b
}

/// **建图前**把 header 落进本端 staging(`upload_pinned` 走 H2D;随后 `sync`
/// —— 核在 COMPUTE 流,两流之间无隐式序)。
///
/// ⚠️ **漏这一步 = 核读 `magic = 0` ⇒ `err = 3` 静默空转**(2026-10-17 G8 首跑
/// 实测定谳:两 rank logits 差 20.04 = AR 一次都没生效 —— header 全零、
/// `produced`/`consumed` 恒 0 而 host 侧毫无报错)。协议面在
/// `cu/comm/ar_stage.cu` 头注;宿主侧初始化的唯一责任点就是这里。
async fn init_staging_header<D: DeviceClient>(
    face: &mut D,
    own: &Bytes,
    layout: StagingLayout,
    host: Option<u64>,
    tag: &str,
) -> Result<()> {
    let n = owl_cuda::comm::ar::HEADER_BYTES as usize;
    // host-mapped:CPU 直接写 host 段(建图之前;无 H2D/无流序)
    if let Some(h) = host {
        let b = staging_header_bytes(layout);
        unsafe { std::ptr::copy_nonoverlapping(b.as_ptr(), h as *mut u8, n) };
        return Ok(());
    }
    let mut pin = face.alloc_pinned(n, tag).await?;
    // ⚠️ 租约是**池化区域**(容量 ≥ 请求)⇒ 只写前 n 字节
    pin.slice_bytes_mut()[..n].copy_from_slice(&staging_header_bytes(layout));
    face.upload_pinned(pin, own, 0, n).await?;
    face.sync().await
}

/// 读回 header(装配后自证用)
async fn read_staging_header<D: DeviceClient>(
    face: &mut D,
    own: &Bytes,
    host: Option<u64>,
) -> Result<[u8; 64]> {
    let mut out = [0u8; 64];
    if let Some(h) = host {
        unsafe { std::ptr::copy_nonoverlapping(h as *const u8, out.as_mut_ptr(), out.len()) };
        return Ok(out);
    }
    let mut buf = vec![0u8; owl_cuda::comm::ar::HEADER_BYTES as usize];
    face.dtoh(own, &mut buf).await?;
    out.copy_from_slice(&buf);
    Ok(out)
}

// ============================================================================
// §2 每 rank 运行态束
// ============================================================================

/// 单 rank 的运行态(每 rank 各一份的东西;host 侧账本/会话是单份,不在此)。
pub(crate) struct TpRank<D: DeviceClient> {
    /// 本 rank 的分片计划(确定性分片协议:`(slice_id, world)`)
    plan: TpPlan,
    /// staging(VMM 常驻;建图前完成握手)
    staging: TpStaging,
    /// rank 束(模型件 + 池/图/环境/spec 运行态;K3 收编)
    dev: RankDev<D>,
}

/// **双 rank 引擎**(TP2;v1 world = 2)。
///
/// 装配入口 = [`TpEngine::assemble`](真机:自动探测 caps)/
/// [`TpEngine::assemble_with_caps`](显式能力事实:测试与自定义后端)。
pub struct TpEngine<D: DeviceClient> {
    /// **共享账本单元**(两 rank 合流到同一本;`RunningEngine` 侧 store 用它)
    ledger: owl_store::SharedLedger,
    /// 两 rank 束(**face 类型 = `DeviceHandle<D>`**:D 是内核后端客户端的类型,
    /// 对上层签名保持 `TpEngine<GpuClient>` 不变)
    ranks: Vec<TpRank<DeviceHandle<D>>>,
    /// rank0 的 tokenizer(host 侧唯一;两侧各解析一份纯浪费)
    tokenizer: Tokenizer,
    /// rank0 的旋钮/预算(装配期已消费;出口给诊断面)
    cfg: EngineConfig,
    /// rank0 的 boot 显存快照(host 侧单份 ⇒ `into_running` 的预算装配用它)
    boot: crate::engine::BootVram,
    /// 池前两 rank 合计 live 摊回单卡的权重/杂项字节(`into_running` 准入账)
    weights_per_card: u64,
}

impl<D: DeviceClient + 'static> TpEngine<D> {
    /// **装配内核**(通用):两侧的 staging 已由调用方备好(见
    /// [`TpEngine::assemble`] 的真机握手路)—— 本函数只做:参数校验 →
    /// 每 rank `boot_rank`(分片 spec/池/图 + `TpCtx` 注入)。
    async fn assemble_with_staging(
        engines: Vec<Engine<D>>,
        loadeds: Vec<LoadedModel>,
        staging0: TpStaging,
        staging1: TpStaging,
    ) -> Result<Self> {
        let world = loadeds.len();
        if world != 2 {
            return loud(format!("TP v1 只支持 world = 2(得 {world};单卡走 Engine::run)"));
        }
        if engines.len() != world {
            return loud(format!(
                "TP 装配参数不齐:engines={} loadeds={}(都必须 = {world})",
                engines.len(),
                loadeds.len()
            ));
        }
        // boot 快照(host 单份;`into_running` 的预算装配用 rank0)
        let boot0 = engines[0].boot_vram();
        let boot1 = engines[1].boot_vram();
        let boot = boot0;
        let mut it = engines.into_iter();
        let (face0, mut cfg0, cell0, _boot0) =
            it.next().expect("world = 2 已守门").into_parts();
        let (face1, mut cfg1, cell1, _boot1) =
            it.next().expect("world = 2 已守门").into_parts();
        // **合流自证**(可判):rank1 自己的那个单元在装配期**必须为空** ——
        // 非空就说明 rank1 在接管之前已经记过账(事件落进了没人读的孤儿账)。
        //
        // 例外 = **两 rank 本来就同一本书**(调用方已在装载前用
        // `Engine::new_with_ledger` 合流):此时 `cell1` 与 `cell0` 是同一个
        // `Arc`,“rank1 自己的单元”就**是**主账 —— 它的在借记录正是两侧装载
        // 刚记下的,不是孤儿账。判据归零由同一性接管。
        if !std::sync::Arc::ptr_eq(&cell0, &cell1) {
            let n = owl_store::book_lock(&cell1).open_count_rank(None);
            if n != 0 {
                return loud(format!(
                    "TP rank1 事件账未合流:它自己的单元已有 {n} 条在借记录(孤儿账)—— \
                     接管必须发生在任何分配之前;修法 = rank1 用 \
                     `Engine::new_with_ledger(cfg1, rank0.ledger_cell(), 1)` 构造"
                ));
            }
        }
        // ── v1 范围护栏（纯配置：`assemble_with` 在 staging 握手前也会先调，
        //    避免坏配置白分配 staging）──────────────────────────────────
        check_assemble_guards(&cfg0, &loadeds)?;

        let mut loadeds = loadeds.into_iter();
        let (l0, l1) = (
            loadeds.next().expect("world = 2 已守门"),
            loadeds.next().expect("world = 2 已守门"),
        );
        let (rm0, tok0) = l0.into_parts();
        let (rm1, tok1) = l1.into_parts();
        drop(tok1); // host 侧 tokenizer 唯一(rank1 的直接放)
        let tokenizer = tok0;
        if staging1.slot_bytes() != staging0.slot_bytes() {
            return loud("TP staging 布局两侧不一致(协议对称性前提)")
        }
        let plan0 = TpPlan::new(0, world).map_err(ModelError::Msg)?;
        let plan1 = TpPlan::new(1, world).map_err(ModelError::Msg)?;
        // ── staging header 初始化(**建图之前**;漏了 = 核 err=3 空转)──
        let (mut face0, mut face1) = (face0, face1);
        init_staging_header(
            &mut face0,
            staging0.own(),
            staging0.layout,
            staging0.host_own(),
            "tp.hdr.r0",
        )
        .await?;
        init_staging_header(
            &mut face1,
            staging1.own(),
            staging1.layout,
            staging1.host_own(),
            "tp.hdr.r1",
        )
        .await?;
        let ctx0 = staging0.ctx(plan0);
        let ctx1 = staging1.ctx(plan1);
        // **池前 live 基线**(per-device!2026-10-24):每 rank 读自己的 account
        // 快照 —— 不再把进程级 LIVE 除以 world。两 rank 的权重/杂项对称,
        // 但并行 boot 时共享全局账会串值,这里必须各算各的。
        let snap0 = face0.vram_snapshot();
        let snap1 = face1.vram_snapshot();
        let baseline0 = snap0.live;
        let baseline1 = snap1.live;
        let weights0 = baseline0.saturating_sub(boot0.live_at_new());
        let weights1 = baseline1.saturating_sub(boot1.live_at_new());
        let weights_per_card = (weights0.saturating_add(weights1)) / world.max(1) as u64;
        // ── 两 rank **并行**装配(per-device 账落地后;2026-10-24)──────────
        // 装配期的 AR 已被 `eval_current_dry` 从 warmup/干跑里摘掉 ⇒ 每 rank
        // 独立完成,无需对端在场(确定性分片协议的"零协调"落实到装配面)。
        let t_boot = std::time::Instant::now();
        let (r0, r1) = tokio::join!(
            boot_rank(face0, &mut cfg0, rm0, Some(ctx0), baseline0, t_boot),
            boot_rank(face1, &mut cfg1, rm1, Some(ctx1), baseline1, t_boot),
        );
        let (dev0, dev1) = match (r0, r1) {
            (Ok(a), Ok(b)) => (a, b),
            (r0, r1) => {
                let msg = match (&r0, &r1) {
                    (Err(e), _) => format!("TP rank0 装配失败:{e}"),
                    (_, Err(e)) => format!("TP rank1 装配失败:{e}"),
                    _ => "TP 装配失败(未知)".to_string(),
                };
                // F-05：成功的那一 rank 已捕图/建池 —— 先关图族（尽力），
                // 再让 Drop 收其余；staging 是 GpuClient 固有命令族，泛型面
                // 无法在这里 PeerRelease（进程退出回收，显式留痕）。
                if let Ok(mut d) = r0 {
                    close_graphs_best_effort(&mut d, "rank0").await;
                }
                if let Ok(mut d) = r1 {
                    close_graphs_best_effort(&mut d, "rank1").await;
                }
                owl_shared::log_warn!(
                    "tp",
                    "TP 装配失败 ⇒ staging 双块未 PeerRelease（泛型面不可达；进程退出回收）"
                );
                return Err(ModelError::Msg(msg));
            }
        };
        let cfg = cfg0;
        // ── 装配后自证:warmup 的 AR 已**对称配对**(produced == consumed、err = 0)──
        if let Err(e) = verify_staging_headers(&dev0, &dev1).await {
            // 自证失败 = 已捕图的半装配态，同样先关图族再上抛。
            let mut dev0 = dev0;
            let mut dev1 = dev1;
            close_graphs_best_effort(&mut dev0, "rank0").await;
            close_graphs_best_effort(&mut dev1, "rank1").await;
            owl_shared::log_warn!(
                "tp",
                "TP staging 自证失败 ⇒ staging 双块未 PeerRelease（泛型面不可达；进程退出回收）"
            );
            return Err(e);
        }
        let ranks = vec![
            TpRank { plan: plan0, staging: staging0, dev: dev0 },
            TpRank { plan: plan1, staging: staging1, dev: dev1 },
        ];
        Ok(Self {
            ranks,
            tokenizer,
            cfg,
            boot,
            weights_per_card,
            ledger: cell0,
        })
    }

    /// **进执行态**(S4 ⑤-K3):rank0 束 = 主设备面,rank1 束 = `peer`
    /// (设备侧程序镜像目标);host 侧账本单一(tokenizer/会话/旋钮取 rank0)。
    pub fn run(self) -> Result<crate::running::RunningEngine<DeviceHandle<D>>> {
        let boot = self.boot; // Copy:先取(下一行 `self.ranks` 会被 part-move)
        let mut it = self.ranks.into_iter();
        let r0 = it.next().ok_or_else(|| ModelError::Msg("TP run: rank0 缺".into()))?;
        let r1 = it.next().ok_or_else(|| ModelError::Msg("TP run: rank1 缺".into()))?;
        crate::engine::into_running(
            self.cfg,
            self.tokenizer,
            boot,
            r0.dev,
            Some(r1.dev),
            self.ledger,
            self.weights_per_card,
        )
    }

    /// rank 数(= world;v1 恒 2)
    pub fn world(&self) -> usize {
        self.ranks.len()
    }

    /// 本 rank 的分片计划
    pub fn plan(&self, rank: usize) -> TpPlan {
        self.ranks[rank].plan
    }

    /// 本 rank 的分片 spec(形状单源)
    pub fn spec(&self, rank: usize) -> &ModelSpec {
        &self.ranks[rank].dev.boot.spec
    }

    /// 本 rank 的 staging(own/peer 双块 + 布局量)
    pub fn staging(&self, rank: usize) -> &TpStaging {
        &self.ranks[rank].staging
    }

    /// slot 字节数(两侧同参;comm 布局单源)
    pub fn slot_bytes(&self) -> u64 {
        self.ranks[0].staging.slot_bytes()
    }

    /// host 侧唯一 tokenizer
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// 主会话图的捕获 slab(> 0 = 真捕到图;0 = eager 降级/未捕)
    pub fn session_slab_bytes(&self, rank: usize) -> usize {
        self.ranks[rank].dev.graphs.session.slab_bytes()
    }

    /// 主会话图捕获结果(`Captured` = 回放态)
    pub fn session_captured(&self, rank: usize) -> bool {
        matches!(self.ranks[rank].dev.graphs.capture_outcome, PlanOutcome::Captured)
    }

    /// 本 rank 的 face 句柄克隆(诊断/门面:同一个 actor 的另一个发送端)
    pub fn face_clone(&self, rank: usize) -> Option<DeviceHandle<D>> {
        self.ranks[rank].dev.graphs.face_clone()
    }

    /// 旋钮(诊断面出口)
    pub fn config(&self) -> &EngineConfig {
        &self.cfg
    }

}

// ============================================================================
// §3 真机能力探测(GpuClient 特化;装配期一次 —— A2.4 热路径零探测)
// ============================================================================

impl TpEngine<GpuClient> {
    /// **真机装配**:分档(UUID/BAR1/P2P 自动探测;混插/不支持 ⇒ 响亮拒)
    /// → staging VMM 对称握手(**建图之前**)→ 每 rank `boot_rank`。
    ///
    /// 装载在调用方手里完成(每 rank `engine.loader().slice(id, world)` ——
    /// 确定性分片协议,零协调)。
    pub async fn assemble(engines: Vec<Engine<GpuClient>>, loadeds: Vec<LoadedModel>) -> Result<Self> {
        Self::assemble_with(engines, loadeds, TpOpts::default()).await
    }

    /// 同 [`Self::assemble`],但显式给装配选项(生产入口用;测试默认走 `assemble`)。
    pub async fn assemble_with(
        engines: Vec<Engine<GpuClient>>,
        loadeds: Vec<LoadedModel>,
        opts: TpOpts,
    ) -> Result<Self> {
        if engines.len() != 2 || loadeds.len() != 2 {
            return loud(format!(
                "TP v1 只支持 world = 2(engines={} loadeds={})",
                engines.len(),
                loadeds.len()
            ));
        }
        let (oa, ob) = (engines[0].device_ordinal(), engines[1].device_ordinal());
        // host CE-push:数据目标偏移必须**主机侧可知** ⇒ 固定单槽(slot 0)。
        // P2P 保持双槽乒乓(数据目标由设备侧 produced 决定)。
        let layout = if opts.comm == TpCommMode::Host {
            StagingLayout::new(StagingLayout::default_layout().slot_bytes, 1)
        } else {
            StagingLayout::default_layout()
        };
        let (uuid0, uuid1) = if opts.comm == TpCommMode::Host {
            let (u0, u1) = probe_uuids_pair(oa, ob)?;
            owl_shared::log_info!(
                "tp",
                "TP2 通信通道 = host(pinned mapped)‖ staging {}/rank = {}×{}MiB + {}B header \
                 —— 跳过 BAR1/P2P 分档",
                layout.total_bytes() >> 20,
                layout.slots,
                layout.slot_bytes >> 20,
                owl_cuda::comm::ar::HEADER_BYTES,
            );
            (u0, u1)
        } else {
            let (caps0, caps1) = probe_caps_pair(oa, ob)?;
            // ── ① 分档:混插/不支持 ⇒ 响亮拒(D5);额度 = staging 总额 ──────
            let plan = PeerPlan::plan(&caps0, &caps1, layout.total_bytes())
                .map_err(|e| ModelError::Msg(format!("TP 分档被拒:{e}")))?;
            owl_shared::log_info!(
                "tp",
                "TP2 分档:{:?}(单笔上限 {}MiB)‖ staging {}/rank = {}×{}MiB + {}B header",
                plan.mechanism,
                plan.per_alloc_limit >> 20,
                layout.total_bytes() >> 20,
                layout.slots,
                layout.slot_bytes >> 20,
                owl_cuda::comm::ar::HEADER_BYTES,
            );
            (caps0.uuid.clone(), caps1.uuid.clone())
        };
        if uuid0 == uuid1 {
            return loud(format!(
                "TP 两侧是同一张卡({uuid0}) —— 双 rank 协议无意义(请给两张卡)"
            ));
        }
        // ── ② staging 对称握手(**建图之前**)───────────────────────────
        let mut it = engines.into_iter();
        let (face0, cfg, cell0, boot0) = it.next().expect("world = 2 已守门").into_parts();
        let (mut face1, cfg1, _cell1, boot1) =
            it.next().expect("world = 2 已守门").into_parts();
        // 纯配置护栏在 staging 握手之前（F-05：坏配置不白分配 staging）。
        check_assemble_guards(&cfg, &loadeds)?;
        let mut face0 = face0;
        // **事件账合流(TP 纪律)**:两 rank 写**同一本**账(单卡档 = 同一本;
        // 分卡靠事件里的 `rank` 维,不靠两本账)。
        //
        // ⚠️ **合流必须在装载之前**(2026-10-20 定谳):装载即发事件(0.8B 实测
        // 312 条)⇒ 事后接管会把这批事件丢进孤儿账(后续归还全是幽灵)。
        // 调用方式 = `Engine::new_with_ledger(cfg1, rank0.ledger_cell(), 1)`;
        // 此处这次 adopt 是**幂等复核**(同一本书 ⇒ 零事件丢失风险 ⇒ 放行),
        // 未按纪律构造的调用方在这里**响亮拒**。
        face1.set_dev_rank(1);
        face1
            .adopt_ledger(owl_store::ledger_entry(&cell0))
            .map_err(|e| {
                ModelError::Msg(format!(
                    "TP rank1 事件账合流失败:{e}\n  ⇒ 修法:rank1 必须**装载前**用 \
                     `Engine::new_with_ledger(cfg1, rank0.ledger_cell(), 1)` 构造 \
                     (装载即发事件,boot 期判据容不下事后接管)"
                ))
            })?;
        owl_shared::log_info!("tp", "事件账:两 rank 合流到同一本(rank 维区分卡)");
        let (staging0, staging1) =
            handshake_staging(&mut face0, &mut face1, &uuid0, &uuid1, layout, opts).await?;
        if let (Some(own), Some(peer)) = (staging0.host_own(), staging0.host_peer()) {
            owl_shared::log_info!(
                "tp",
                "host staging rank0: own={own:#x} peer={peer:#x}(rank1 镜像 attach)"
            );
        }
        // ── ③ 通用装配(每 rank boot_rank;图捕获在此发生)──────────────
        let engines = vec![
            Engine::from_handle(cfg, face0, cell0.clone(), boot0)
                .map_err(|e| ModelError::Msg(format!("TP 重装 rank0 引擎壳失败:{e}")))?,
            // **rank1 用自己的 cfg/boot 快照**（F-04）：`boot1.live_at_new` 是
            // rank1 装载前的基线，`assemble_with_staging` 的 per-card 权重账
            // 靠它；传 `boot0` 会把 rank0 的基线摊到 rank1。
            Engine::from_handle(cfg1, face1, cell0.clone(), boot1)
                .map_err(|e| ModelError::Msg(format!("TP 重装 rank1 引擎壳失败:{e}")))?,
        ];
        Self::assemble_with_staging(engines, loadeds, staging0, staging1)
            .await
    }

    /// **同卡双 rank 装配(T3 位级 transport 档;测试/取证面)** —— 与
    /// [`Self::assemble`] 同构,唯一差别:**跳过 `PeerPlan` 分档**(两张 rank
    /// 查同一个 ordinal ⇒ 自探测的 P2P 能力恒 false,分档无意义)。
    ///
    /// staging 仍走 VMM `peer_alloc`/`peer_import`(**同卡自导入是 VMM 的
    /// 合法用法**:同一物理页在第二个 context 建只读别名 ⇒ AR 协议一字不改)。
    ///
    /// 用途 = G8-c②(§19.2):同卡档(无 PCIe 跨卡路径)与跨卡档逐位对拍
    /// ⇒ **位级隔离 transport**(协议/序/驱动窗口任何回归当场红)。
    /// 生产路径**不**走本入口(同卡双 rank 无性能意义)。
    ///
    /// ⚠️ 两 rank 在**同一张卡**上并发推进(自旋核与计算核共享 SM)⇒ 门的
    /// 超时即判据之一:若这里挂死,说明该卡上的核并发调度不成立(而不是
    /// 协议错 —— 协议面已由跨卡档覆盖)。
    pub async fn assemble_same_device(
        engines: Vec<Engine<GpuClient>>,
        loadeds: Vec<LoadedModel>,
    ) -> Result<Self> {
        if engines.len() != 2 || loadeds.len() != 2 {
            return loud(format!(
                "同卡档只支持 world = 2(engines={} loadeds={})",
                engines.len(),
                loadeds.len()
            ));
        }
        let (oa, ob) = (engines[0].device_ordinal(), engines[1].device_ordinal());
        if oa != ob {
            return loud(format!(
                "同卡档要求两 rank 同一个 ordinal(得 {oa}/{ob});跨卡请走 TpEngine::assemble"
            ));
        }
        // 同卡 ⇒ 同一份 caps(UUID/BAR1 同;P2P 自探测无意义不在本档用)
        let (caps0, _caps1) = probe_caps_pair(oa, oa)?;
        let layout = StagingLayout::default_layout();
        owl_shared::log_info!(
            "tp",
            "TP2 同卡档(T3):ordinal {oa} ×2;staging {}/rank = {}×{}MiB + {}B header",
            layout.total_bytes() >> 20,
            layout.slots,
            layout.slot_bytes >> 20,
            owl_cuda::comm::ar::HEADER_BYTES,
        );
        let mut it = engines.into_iter();
        let (face0, cfg, cell0, boot0) = it.next().expect("world = 2 已守门").into_parts();
        let (mut face1, cfg1, _cell1, boot1) =
            it.next().expect("world = 2 已守门").into_parts();
        check_assemble_guards(&cfg, &loadeds)?;
        let mut face0 = face0;
        face1.set_dev_rank(1);
        face1
            .adopt_ledger(owl_store::ledger_entry(&cell0))
            .map_err(|e| {
                ModelError::Msg(format!(
                    "同卡档 rank1 事件账合流失败:{e}\n  ⇒ 修法:rank1 必须**装载前**用 \
                     `Engine::new_with_ledger(cfg1, rank0.ledger_cell(), 1)` 构造"
                ))
            })?;
        let uuid = caps0.uuid.clone();
        let (staging0, staging1) =
            handshake_staging(&mut face0, &mut face1, &uuid, &uuid, layout, TpOpts::default())
                .await?;
        let engines = vec![
            Engine::from_handle(cfg, face0, cell0.clone(), boot0)
                .map_err(|e| ModelError::Msg(format!("同卡档重装 rank0 引擎壳失败:{e}")))?,
            // 同卡档两 rank 共用一张卡，但 rank1 的 boot 快照仍是它自己构造时的
            // 读数（同卡场景两者通常相同）；保持 per-rank 语义，别把 rank0 的
            // 基线摊过来（与 `assemble_with` 同律）。
            Engine::from_handle(cfg1, face1, cell0.clone(), boot1)
                .map_err(|e| ModelError::Msg(format!("同卡档重装 rank1 引擎壳失败:{e}")))?,
        ];
        Self::assemble_with_staging(engines, loadeds, staging0, staging1)
            .await
    }

    /// **收尾**:先卸图(图租约钉住 staging ⇒ 必须先销毁图),再
    /// `PeerRelease` / `HostRelease` 释放 staging 双块(own = 分配块,
    /// peer = 别名/attach 块;D2 全有或全无)。
    /// 失败不短路(全量尝试;首错上抛)。
    ///
    /// ⚠️ 收尾面是 **GpuClient 特化**(`peer_release` 是 GpuClient 固有方法,
    /// 不在 `DeviceClient` trait 上)—— 见本块位置。
    pub async fn shutdown(mut self) -> Result<()> {
        let mut first: Option<ModelError> = None;
        // host 通道的 owner 在 rank0;必须先收 rank1 的 attach,再 free owner。
        let host = self.ranks.iter().any(|r| r.staging.host_mode());
        if host {
            for i in (0..self.ranks.len()).rev() {
                if let Some(e) = retire_rank(&mut self.ranks[i].dev, &format!("rank{i}")).await {
                    if first.is_none() {
                        first = Some(e);
                    }
                }
            }
        } else {
            for (i, r) in self.ranks.iter_mut().enumerate() {
                if let Some(e) = retire_rank(&mut r.dev, &format!("rank{i}")).await {
                    if first.is_none() {
                        first = Some(e);
                    }
                }
            }
        }
        match first {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

/// **装配失败时的图族尽力收尾**（F-05；泛型面，无 staging 命令）。
async fn close_graphs_best_effort<D: DeviceClient>(dev: &mut RankDev<D>, tag: &str) {
    if let Err(e) = dev.graphs.close().await {
        owl_shared::log_warn!("tp", "{tag} 装配失败收尾:图族关闭失败: {e}");
    }
}

/// **staging header 装配后自证**：produced == consumed == 0 且 err == 0。
async fn verify_staging_headers<D: DeviceClient>(
    dev0: &RankDev<D>,
    dev1: &RankDev<D>,
) -> Result<()> {
    for (i, dev) in [dev0, dev1].into_iter().enumerate() {
        let Some(tp_ctx) = dev.boot.tp.as_ref() else {
            return loud(format!("TP rank{i} 装配后无 TpCtx(装配路径错)"));
        };
        let Some(mut f) = dev.graphs.face_clone() else {
            return loud(format!("TP rank{i} 装配后 face 句柄耗尽"));
        };
        let h = read_staging_header(
            &mut f,
            &tp_ctx.own,
            (tp_ctx.host_own != 0).then_some(tp_ctx.host_own),
        )
        .await?;
        let produced = u64::from_le_bytes(h[0..8].try_into().unwrap());
        let consumed = u64::from_le_bytes(h[8..16].try_into().unwrap());
        let err = u32::from_le_bytes(h[28..32].try_into().unwrap());
        owl_shared::log_info!(
            "tp",
            "rank{i} staging 自证:produced={produced} consumed={consumed} err={err}"
        );
        if err != 0 {
            return loud(format!(
                "TP rank{i} staging err={err}(1=stage 超时 / 2=reduce 超时 / 3=header 校验失败) —— 装配期 AR 未能配对(见 tp.rs 的 header 初始化与并发 boot 注记)"
            ));
        }
        if produced != 0 || consumed != 0 {
            return loud(format!(
                "TP rank{i} 装配期出现了 AR 流量:produced={produced} consumed={consumed}\
                 (装配/warmup 必须零 AR —— 真 AR 只在回放期两 rank 同序入队)"
            ));
        }
    }
    Ok(())
}

/// **单 rank 收尾**(卸图 → `PeerRelease` / `HostRelease` 双块;装配态与执行态共用):
/// 图租约钉住 staging ⇒ **必须先销毁整族图**;失败**不短路**(记录首错,
/// 继续走 PeerRelease —— 调用方的多 rank 循环才做"全量尝试")。
/// 返回 `None` = 干净;`Some(err)` = 首错(已告警)。
///
/// `GpuClient` 特化:`peer_release` / `host_release` 是 GpuClient 固有方法
/// (非 trait 面)。
async fn retire_rank(
    dev: &mut RankDev<DeviceHandle<GpuClient>>,
    tag: &str,
) -> Option<ModelError> {
    let mut first: Option<ModelError> = None;
    let note = |e: ModelError, first: &mut Option<ModelError>| {
        owl_shared::log_warn!("tp", "{e}");
        if first.is_none() {
            *first = Some(e);
        }
    };
    if let Err(e) = dev.graphs.close().await {
        note(ModelError::Msg(format!("{tag} 图族退役失败:{e}")), &mut first);
    }
    match dev.graphs.face_clone() {
        None => note(
            ModelError::Msg(format!("{tag} 收尾:face 句柄耗尽,无法 PeerRelease")),
            &mut first,
        ),
        Some(mut face) => match dev.boot.tp.as_ref() {
            None => note(
                ModelError::Msg(format!("{tag} 收尾:无 TpCtx(非 TP 态?)")),
                &mut first,
            ),
            Some(tp) => {
                // host f8 CE 的设备侧 scratch/dummy 是普通 Owned 块,
                // 图族已关 ⇒ 先走普通 Free(失败不阻断 host 段收尾)。
                let mut dev_ids = Vec::new();
                if let Some(q) = tp.host_quant.as_ref() {
                    dev_ids.push(q.id);
                }
                if let Some(d) = tp.host_dummy.as_ref() {
                    dev_ids.push(d.id);
                }
                if !dev_ids.is_empty() {
                    // raw GpuClient 分配(绕开 DeviceHandle 块池/事件账)⇒ 原路 raw 释放
                    if let Err(e) = face.client_mut().free(&dev_ids).await {
                        note(
                            ModelError::Msg(format!("{tag} host f8 scratch 释放失败:{e}")),
                            &mut first,
                        );
                    }
                }
                // staging 两块(own + peer 别名)
                let ids = vec![tp.own.id, tp.peer.id];
                // peer/host 面不经 `DeviceClient`(自成命令族)⇒ 走可变直通口
                // (设计卷 §六 白名单:peer_* 指令;**这些块不入事件账** —— 见 §十二 步 3 注)
                let released = if tp.host_own != 0 {
                    face.client_mut().host_release(ids).await.map(|_| "HostRelease")
                } else {
                    face.client_mut().peer_release(ids).await.map(|_| "PeerRelease")
                };
                if let Err(e) = released {
                    note(ModelError::Msg(format!("{tag} staging 释放失败:{e}")), &mut first);
                }
            }
        },
    }
    first
}

/// 单 rank 的 staging 游标三元组(`(produced, consumed, err)`;见
/// [`RunningEngine::tp_staging_state`])。
async fn staging_state_of(dev: &mut RankDev<DeviceHandle<GpuClient>>) -> Result<(u64, u64, u32)> {
    let Some(tp) = dev.boot.tp.clone() else {
        return loud("TP staging: 无 TpCtx(装配路径错)");
    };
    let Some(mut f) = dev.graphs.face_clone() else {
        return loud("TP staging: face 句柄耗尽");
    };
    let mut buf = [0u8; 32];
    if tp.host_own != 0 {
        unsafe {
            std::ptr::copy_nonoverlapping(tp.host_own as *const u8, buf.as_mut_ptr(), buf.len())
        };
    } else {
        f.dtoh(&tp.own, &mut buf).await?;
    }
    Ok((
        u64::from_le_bytes(buf[0..8].try_into().unwrap()),
        u64::from_le_bytes(buf[8..16].try_into().unwrap()),
        u32::from_le_bytes(buf[28..32].try_into().unwrap()),
    ))
}

impl RunningEngine<DeviceHandle<GpuClient>> {
    /// **TP staging 游标三元组**(诊断出口;文档 §20.4 的工具口):
    /// `(rank0 的 (produced, consumed, err), rank1 的同款)`;单卡 = `None`。
    ///
    /// 判据(**每 turn 之后**):两侧都必须 `produced == consumed` 且 `err == 0`
    /// —— 否则 AR 未配对(超时 / 头校验失败 / 序列不同步)。装配期自证在
    /// [`TpEngine::assemble`];本口是每 turn 之后的同一判据。
    pub async fn tp_staging_state(&mut self) -> Result<Option<((u64, u64, u32), (u64, u64, u32))>> {
        let Some(p) = self.peer.as_mut() else { return Ok(None) };
        let a = staging_state_of(&mut self.dev).await?;
        let b = staging_state_of(p).await?;
        Ok(Some((a, b)))
    }


    /// **执行态 TP 收尾**(T1 门与生产停止路径共用):rank0 束 + peer 束各走
    /// [`retire_rank`](卸图 → `PeerRelease`)。
    ///
    /// 与 [`TpEngine::shutdown`] 的分工:后者是"装配后未进执行态"的收尾
    /// (rank 束 = `TpRank`),本函数是"跑过 turn 的执行态"的收尾
    /// (rank 束 = `RankDev`,staging 句柄在 `dev.boot.tp` 里 —— 同一笔分配)。
    /// 非 TP 态(无 peer 束)⇒ **响亮拒**(收尾路径错,不是静默空跑)。
    pub async fn shutdown_tp(mut self) -> Result<()> {
        if self.peer.is_none() {
            return loud("shutdown_tp: 本引擎不是 TP 态(无 peer 束;单卡请走 Drop)");
        }
        let mut first: Option<ModelError> = None;
        let host = self
            .dev
            .boot
            .tp
            .as_ref()
            .is_some_and(|tp| tp.host_own != 0);
        if host {
            // host owner 在 rank0:先收 rank1 的 attach
            if let Some(peer) = self.peer.as_mut() {
                if let Some(e) = retire_rank(peer, "rank1").await {
                    first = Some(e);
                }
            }
            if let Some(e) = retire_rank(&mut self.dev, "rank0").await {
                if first.is_none() {
                    first = Some(e);
                }
            }
        } else {
            if let Some(e) = retire_rank(&mut self.dev, "rank0").await {
                first = Some(e);
            }
            if let Some(peer) = self.peer.as_mut() {
                if let Some(e) = retire_rank(peer, "rank1").await {
                    if first.is_none() {
                        first = Some(e);
                    }
                }
            }
        }
        match first {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

/// **staging 对称握手**(生产与同卡档唯一实现):两侧 `peer_alloc` → fd 交叉 →
/// `peer_import`(两侧;所有权归分配侧 —— 导入侧绝不 close,§11.3 裁决 2)。
/// `peer_uuid` = **对端** uuid(导入的是对端的分配)。
async fn handshake_staging(
    face0: &mut DeviceHandle<GpuClient>,
    face1: &mut DeviceHandle<GpuClient>,
    uuid0: &str,
    uuid1: &str,
    layout: StagingLayout,
    opts: TpOpts,
) -> Result<(TpStaging, TpStaging)> {
    // **peer 面 = 独立命令族**(不在 `DeviceClient` trait 上;见 TP2 施工设计 §7.0
    // 裁决 D1)⇒ 走可变直通口。代价:**这些块不进 DeviceHandle 事件账**
    // (TP 档对账的己知残差;单卡档不受影响 —— 见设计卷 §十二 步 3 注)。
    let face0 = face0.client_mut();
    let face1 = face1.client_mut();
    let bytes = layout.total_bytes() as usize;

    // ── host-mapped 兜底通道(2026-11-04)────────────────────────────────
    //
    // 两段 host staging 都由 rank0 的 actor 分配(owner 记账在 rank0),
    // rank1 只 attach(不拥有、不 free)。rank0 own=H0 / peer=H1;
    // rank1 own=H1 / peer=H0。这样 shutdown 时先收 rank1 的 attach,再收
    // rank0 的 owner,host 段生命周期严格覆盖两份账。
    if opts.comm == TpCommMode::Host {
        if layout.slots != 1 {
            return loud(format!(
                "host CE-push staging 需要 n_slots = 1(实得 {});\
                 多槽的 slot 偏移由设备侧 produced 决定,主机侧无法烘进 CopyBlock",
                layout.slots
            ));
        }
        if opts.ar_f8 {
            owl_shared::log_info!(
                "tp",
                "host CE-push f8:device quant scratch → CE 搬紧凑 fp8 载荷 → 本地 host reduce 反量化"
            );
        }
        let (own0, h0) = face0
            .host_alloc(bytes, "tp.host.r0")
            .await
            .map_err(|e| ModelError::Msg(format!("TP host staging 分配失败(rank0/H0):{e}")))?;
        let (own1, h1) = match face0.host_alloc(bytes, "tp.host.r1").await {
            Ok(v) => v,
            Err(e) => {
                if let Err(er) = face0.host_release(vec![own0.id]).await {
                    owl_shared::log_warn!("tp", "host staging 回滚 H0 失败: {er}");
                }
                return Err(ModelError::Msg(format!(
                    "TP host staging 分配失败(rank0/H1):{e}"
                )));
            }
        };
        let p0 = match face1.host_attach(h1, bytes, "tp.host.own.r1").await {
            Ok(b) => b,
            Err(e) => {
                if let Err(er) = face0.host_release(vec![own0.id, own1.id]).await {
                    owl_shared::log_warn!("tp", "host staging 回滚 H0/H1 失败: {er}");
                }
                return Err(ModelError::Msg(format!(
                    "TP host staging attach 失败(rank1 ← H1):{e}"
                )));
            }
        };
        let p1 = match face1.host_attach(h0, bytes, "tp.host.peer.r1").await {
            Ok(b) => b,
            Err(e) => {
                if let Err(er) = face1.host_release(vec![p0.id]).await {
                    owl_shared::log_warn!("tp", "host staging 回滚 attach H1 失败: {er}");
                }
                if let Err(er) = face0.host_release(vec![own0.id, own1.id]).await {
                    owl_shared::log_warn!("tp", "host staging 回滚 H0/H1 失败: {er}");
                }
                return Err(ModelError::Msg(format!(
                    "TP host staging attach 失败(rank1 ← H0):{e}"
                )));
            }
        };
        // host f8 CE 的可选设备侧臂:payload 先量化进本端 device scratch,
        // 再由 CE 把 fp8 载荷(约 0.515×)搬到对端 host。dummy header 让
        // 现役 stage f8 核可以只做“量化+写 device”,不触碰真实 produced。
        let (host_quant0, host_dummy0) = if opts.ar_f8 {
            let q = face0
                .alloc_uninit(
                    Dtype::U8,
                    (layout.slot_bytes + owl_cuda::comm::ar::HEADER_BYTES) as usize,
                    "tp.host.quant.r0",
                )
                .await
                .map_err(|e| ModelError::Msg(format!("TP host f8 quant scratch 分配失败(rank0):{e}")))?;
            let d = face0
                .htod(
                    Dtype::U8,
                    &Shape::from(vec![owl_cuda::comm::ar::HEADER_BYTES as usize]),
                    &staging_header_bytes(StagingLayout::new(0, 1)),
                    "tp.host.dummy.r0",
                )
                .await
                .map_err(|e| ModelError::Msg(format!("TP host f8 dummy header 写入失败(rank0):{e}")))?;
            (Some(q.view(Dtype::U8, 1)), Some(d))
        } else {
            (None, None)
        };
        let (host_quant1, host_dummy1) = if opts.ar_f8 {
            let q = face1
                .alloc_uninit(
                    Dtype::U8,
                    (layout.slot_bytes + owl_cuda::comm::ar::HEADER_BYTES) as usize,
                    "tp.host.quant.r1",
                )
                .await
                .map_err(|e| ModelError::Msg(format!("TP host f8 quant scratch 分配失败(rank1):{e}")))?;
            let d = face1
                .htod(
                    Dtype::U8,
                    &Shape::from(vec![owl_cuda::comm::ar::HEADER_BYTES as usize]),
                    &staging_header_bytes(StagingLayout::new(0, 1)),
                    "tp.host.dummy.r1",
                )
                .await
                .map_err(|e| ModelError::Msg(format!("TP host f8 dummy header 写入失败(rank1):{e}")))?;
            (Some(q.view(Dtype::U8, 1)), Some(d))
        } else {
            (None, None)
        };
        owl_shared::log_info!(
            "tp",
            "TP2 通信通道 = host(pinned mapped;rank0 H0={h0:#x}/H1={h1:#x};无需 P2P;f8={})",
            if opts.ar_f8 { "CE quant" } else { "raw" }
        );
        return Ok((
            TpStaging {
                own: own0,
                peer: own1,
                ar_f8: opts.ar_f8,
                ar_fused: false,
                ar2: false,
                host_mode: true,
                host_own: h0,
                host_peer: h1,
                host_quant: host_quant0,
                host_dummy: host_dummy0,
                layout,
            },
            TpStaging {
                own: p0,
                peer: p1,
                ar_f8: opts.ar_f8,
                ar_fused: false,
                ar2: false,
                host_mode: true,
                host_own: h1,
                host_peer: h0,
                host_quant: host_quant1,
                host_dummy: host_dummy1,
                layout,
            },
        ));
    }

    // **P2P 握手失败回滚（F-06）**：与 host 分支对称 —— 已分配的 VMM 块/别名
    // 必须在失败路径显式 `PeerRelease`，否则半握手会在进程内累积 VMM/句柄。
    let (own0, cred0) = match face0.peer_alloc(bytes, "tp.staging.r0").await {
        Ok(v) => v,
        Err(e) => {
            return Err(ModelError::Msg(format!("TP staging 分配失败(rank0):{e}")));
        }
    };
    let (own1, cred1) = match face1.peer_alloc(bytes, "tp.staging.r1").await {
        Ok(v) => v,
        Err(e) => {
            if let Err(er) = face0.peer_release(vec![own0.id]).await {
                owl_shared::log_warn!("tp", "P2P 握手回滚:释放 rank0 staging 失败: {er}");
            }
            return Err(ModelError::Msg(format!("TP staging 分配失败(rank1):{e}")));
        }
    };
    // staging 别名**恒只读导入**(v1 数据腿:stage 写本端槽 → reduce 读对端槽)。
    // push(写对端)与 1-stage 对称 arena 两个实验档已在 2026-10-27 配置面剪枝中退役
    // (3090 对读写对称 / 对称 arena 实测中性,详见 `roadmap.local/26-10-27-配置面-review.md`)。
    let p0 = match face0
        .peer_import(cred1, bytes, uuid1.to_string(), "tp.peer.r0")
        .await
    {
        Ok(v) => v,
        Err(e) => {
            if let Err(er) = face0.peer_release(vec![own0.id]).await {
                owl_shared::log_warn!("tp", "P2P 握手回滚:释放 rank0 staging 失败: {er}");
            }
            if let Err(er) = face1.peer_release(vec![own1.id]).await {
                owl_shared::log_warn!("tp", "P2P 握手回滚:释放 rank1 staging 失败: {er}");
            }
            return Err(ModelError::Msg(format!(
                "TP staging 导入失败(rank0 ← rank1):{e}"
            )));
        }
    };
    let p1 = match face1
        .peer_import(cred0, bytes, uuid0.to_string(), "tp.peer.r1")
        .await
    {
        Ok(v) => v,
        Err(e) => {
            if let Err(er) = face0.peer_release(vec![p0.id, own0.id]).await {
                owl_shared::log_warn!("tp", "P2P 握手回滚:释放 rank0 staging/别名失败: {er}");
            }
            if let Err(er) = face1.peer_release(vec![own1.id]).await {
                owl_shared::log_warn!("tp", "P2P 握手回滚:释放 rank1 staging 失败: {er}");
            }
            return Err(ModelError::Msg(format!(
                "TP staging 导入失败(rank1 ← rank0):{e}"
            )));
        }
    };
    Ok((
        TpStaging {
            own: own0,
            peer: p0,
            ar_f8: opts.ar_f8,
            ar_fused: opts.ar_fused,
            ar2: opts.ar2,
            host_mode: false,
            host_own: 0,
            host_peer: 0,
            host_quant: None,
            host_dummy: None,
            layout,
        },
        TpStaging {
            own: own1,
            peer: p1,
            ar_f8: opts.ar_f8,
            ar_fused: opts.ar_fused,
            ar2: opts.ar2,
            host_mode: false,
            host_own: 0,
            host_peer: 0,
            host_quant: None,
            host_dummy: None,
            layout,
        },
    ))
}

/// 一对 ordinal 的 peer 能力(驱动枚举 UUID + nvidia-smi BAR1 +
/// `cuDeviceCanAccessPeer` 双向)。
fn probe_uuids_pair(a: usize, b: usize) -> Result<(String, String)> {
    let rows = owl_cuda::ffi::device_rows()
        .map_err(|e| ModelError::Msg(format!("TP 探测:设备枚举失败 {e}")))?;
    let row_of = |o: usize| -> Result<&owl_shared::device::DeviceRow> {
        rows.iter().find(|r| r.ordinal == o).ok_or_else(|| {
            ModelError::Msg(format!("TP 探测:ordinal {o} 不在设备表(共 {} 张)", rows.len()))
        })
    };
    let (ra, rb) = (row_of(a)?, row_of(b)?);
    Ok((
        owl_shared::device::uuid_norm(&ra.uuid),
        owl_shared::device::uuid_norm(&rb.uuid),
    ))
}

fn probe_caps_pair(a: usize, b: usize) -> Result<(PeerCaps, PeerCaps)> {
    let rows = owl_cuda::ffi::device_rows()
        .map_err(|e| ModelError::Msg(format!("TP 探测:设备枚举失败 {e}")))?;
    let row_of = |o: usize| -> Result<&owl_shared::device::DeviceRow> {
        rows.iter().find(|r| r.ordinal == o).ok_or_else(|| {
            ModelError::Msg(format!("TP 探测:ordinal {o} 不在设备表(共 {} 张)", rows.len()))
        })
    };
    let (ra, rb) = (row_of(a)?, row_of(b)?);
    let (ua, ub) = (
        owl_shared::device::uuid_norm(&ra.uuid),
        owl_shared::device::uuid_norm(&rb.uuid),
    );
    // BAR1 按 BDF 对账(nvidia-smi 详查面给 BDF;驱动枚举也带 BDF ⇒ 两源归一)
    let bar1 = query_bar1_bytes()?;
    let bar1_of = |r: &owl_shared::device::DeviceRow| -> Result<u64> {
        let bdf = owl_shared::device::bdf_norm(&r.bdf);
        bar1.get(&bdf).copied().ok_or_else(|| {
            ModelError::Msg(format!(
                "TP 探测:nvidia-smi 无 BDF {bdf} 的 BAR1(表 {} 项)",
                bar1.len()
            ))
        })
    };
    let (ba, bb) = (bar1_of(ra)?, bar1_of(rb)?);
    let p2p = |x: usize, y: usize| -> Result<bool> {
        owl_cuda::ffi::peer_supported(x, y)
            .map_err(|e| ModelError::Msg(format!("TP 探测:cuDeviceCanAccessPeer({x},{y}) 失败:{e}")))
    };
    let (p_ab, p_ba) = (p2p(a, b)?, p2p(b, a)?);
    let both = p_ab && p_ba; // 单向不成立 ⇒ 组合不可用(双向都要)
    Ok((
        PeerCaps::new(ua, ba, both),
        PeerCaps::new(ub, bb, both),
    ))
}

/// BAR1 表(BDF 规范 → bytes):`nvidia-smi -q -d MEMORY` 详查面。
/// 顶层 I/O(装配期一次);命令失败/输出无有效行 = Err(不静默回退)。
fn query_bar1_bytes() -> Result<std::collections::HashMap<String, u64>> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["-q", "-d", "MEMORY"])
        .output()
        .map_err(|e| ModelError::Msg(format!("TP 探测:nvidia-smi 起不来 {e}")))?;
    if !out.status.success() {
        return loud(format!("TP 探测:nvidia-smi 退出码 {:?}", out.status.code()));
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let m = owl_shared::device::parse_nvidia_smi_bar1(&text);
    if m.is_empty() {
        return loud("TP 探测:bar1 表为空(nvidia-smi -q -d MEMORY 输出无 BAR1 段)");
    }
    Ok(m)
}


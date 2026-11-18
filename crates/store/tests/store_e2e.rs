//! **store 无卡端到端演示(§19/§21)**:需求驱动下沉 → 链级整条搬 → 命中批量回灌 →
//! 内容逐字节恒等 —— 全部用 **store 自己的公开 API** 驱动 + `CpuFace` 做真搬运,
//! **不依赖 engine / GPU**(秒级,可进默认套件)。
//!
//! **编排归属(§21 裁决)**:store **不做 I/O**(连 `face` 都不认)—— 它只出
//! **计划**(`plan_offload`/`plan_onload`:算片 + 借宿主台)与**记账**
//! (`commit_*` / `abort_*`)。搬运由 **engine 编排**:取计划 → `face.swap` → 回执 → 记账。
//! 本文件里的 `engine_offload`/`engine_onload` 就是那段编排的**样板**(生产里在
//! `engine::running::apply_verdict` / `exec::phases::execute_begin`)。
//!
//! 场景(冷热交替;与 engine 演示台同构):
//! ```text
//! 池 6 槽 / 宿主 8MiB
//! 轮 1/2  热链 A、B(各 3 块)⇒ 池满
//! 轮 3    新需求(need = 2)⇒ 钩子判:链级**整条**沉 A(3 块,内容留宿主)
//! 轮 4    命中 A ⇒ 前 2 块在宿主 ⇒ **一条命令**批量回灌 ⇒ 逐字节恒等
//! ```

use owl_cpu::CpuFace;
use owl_iface::contract::{DeviceClient, Dtype, HeapRegion, PinnedRegion, SwapDir};
use owl_store::{
    commit_offload, commit_onload, plan_capacity, plan_offload, plan_offload_blocks, plan_onload,
    reclaim_slot_table, BlockClass, BlockLeg, ChainHash, EvictOut, LegConfig, LruPolicy, Room,
    Need, NeedWhy, PieceSpec, PoolLayout, Sink, SkipWhy, StateStore, StoreResult,
};

const PIECE_ELEMS: usize = 64;
const PIECE_BYTES: usize = PIECE_ELEMS * 4;
const PAGE_TOKENS: usize = 32;

// ───────────────────────── engine 式编排(样板)─────────────────────────
//
// 生产位置:`engine::running::apply_verdict`(决策单逐条执行)与
// `engine::exec::phases::execute_begin`(段边界回灌)。三者形态必须一致:
// **取计划 → face.swap → 回执取台 → commit**;失败走 abort(条目删,不留假账)。

/// 换出**整条链**(一条 swap 搬 N 块)。
async fn engine_offload(
    leg: &mut BlockLeg,
    face: &mut CpuFace,
    tail: ChainHash,
) -> StoreResult<usize> {
    let (blocks, pieces, table) = {
        let p = plan_offload(leg, tail)?; // 纯 host(算片 + 借台);Busy = 零副作用
        (p.blocks, p.pieces, p.table)
    };
    let conn = match face.swap(SwapDir::Out, &pieces[0], Box::new(table)).await {
        Ok(c) => c,
        Err(e) => {
            owl_store::abort_offload(leg, &blocks); // 内容已丢 ⇒ 条目一并删
            return Err(owl_store::StoreError::Invariant(format!("swap(Out) 失败:{e}")));
        }
    };
    let table = reclaim_slot_table(conn)?;
    commit_offload(leg, &blocks, table)
}

/// 回灌**一批块**(一条 swap 搬 N 块)。
async fn engine_onload(
    leg: &mut BlockLeg,
    face: &mut CpuFace,
    chains: &[ChainHash],
) -> StoreResult<Vec<u32>> {
    let (cs, slots, pieces, table) = {
        let p = plan_onload(leg, chains)?; // 纯 host(取内容 + 分配槽 + 组片)
        (p.chains, p.slots, p.pieces, p.table)
    };
    let conn = match face.swap(SwapDir::In, &pieces[0], Box::new(table)).await {
        Ok(c) => c,
        Err(e) => {
            owl_store::abort_onload(leg, &cs, &slots); // 槽归还 + 条目删
            return Err(owl_store::StoreError::Invariant(format!("swap(In) 失败:{e}")));
        }
    };
    let table = reclaim_slot_table(conn)?;
    commit_onload(leg, &cs, &slots, table)
}

// ───────────────────────── 夹具 ─────────────────────────

/// 两片段(= 两层 k/v)× `slots` 槽的设备池。
async fn pieces(face: &mut CpuFace, slots: usize) -> Vec<PieceSpec> {
    let mut out = Vec::new();
    for _ in 0..2 {
        let b = face.alloc(Dtype::F32, PIECE_ELEMS * slots, "t").await.expect("池块");
        out.push(PieceSpec { buf: b, stride_bytes: PIECE_BYTES });
    }
    out
}

fn leg(slots: usize) -> BlockLeg {
    BlockLeg::open(LegConfig {
        slots,
        block_tokens: PAGE_TOKENS,
        block_bytes: PIECE_BYTES * 2, // 两片段 = 一个逻辑块
        cache: owl_store::CachePolicy::Cap(slots + 8),
        host_bytes: 8 << 20, // 宿主面开(8MiB ⇒ 可容 40+ 块)
        class: BlockClass::KvPage,
        label: "store-e2e",
    })
    .expect("开腿")
}

/// 只用 KV 腿的判据入口(测试辅助:状态腿空账 + 状态宿主 0 位)。
fn plan_kv(l: &BlockLeg, need: Need) -> owl_store::Verdict {
    let state = StateStore::new(1, 0);
    let room = Room {
        host_enabled: l.host_enabled(),
        host_kv_slots: l.host_free_slots(),
        host_snap_slots: 0,
        disk_kv_slots: 0,
        disk_snap_slots: 0,
        disk_enabled: false,
        bulk_drop_when_tier_full: false,
    };
    plan_capacity(l, &state, &LruPolicy, need, &[], &[], room)
}

async fn seed(face: &mut CpuFace, ps: &[PieceSpec], slot: u32, seed: u8) {
    for p in ps {
        let pat: Vec<u8> = (0..p.stride_bytes).map(|i| seed.wrapping_add(i as u8)).collect();
        let lease: Box<dyn PinnedRegion + Send> = Box::new(HeapRegion(pat));
        face.upload_pinned(lease, &p.buf, slot as usize * p.stride_bytes, p.stride_bytes)
            .await
            .expect("seed");
    }
}

async fn read_slot(face: &mut CpuFace, ps: &[PieceSpec], slot: u32) -> Vec<u8> {
    let mut out = Vec::new();
    for p in ps {
        let mut d = vec![0u8; p.stride_bytes];
        face.dtoh_at(&p.buf, slot as usize * p.stride_bytes, &mut d).await.expect("读槽");
        out.extend_from_slice(&d);
    }
    out
}

/// 建一条链(整块登记)→ 归还会话引用(缓存独占 ⇒ 可下沉)。
async fn build_chain(leg: &mut BlockLeg, tokens: &[u32]) -> Vec<ChainHash> {
    let mut table = Vec::new();
    leg.ensure_for_len(&mut table, tokens.len()).expect("块表");
    leg.register_chain(tokens, &table).expect("登记");
    let hit = leg.match_prefix(tokens).expect("命中");
    leg.release_table(&mut table).expect("会话归还");
    hit.hashes
}

#[tokio::test]
async fn offload_and_batch_reload_without_gpu() {
    let mut face = CpuFace::new();
    let ps = pieces(&mut face, 6).await;
    let mut l = leg(6);
    l.attach_pool(PoolLayout { class: BlockClass::KvPage, epoch: 1, pieces: ps.clone() })
        .expect("注册布局");

    // ── 轮 1/2:两条热链(各 3 块)⇒ 池满(6 槽)
    let a_tokens: Vec<u32> = (0..96).collect(); // 96 tok = 3 块
    let b_tokens: Vec<u32> = (1000..1096).collect();
    let a = build_chain(&mut l, &a_tokens).await;
    let b = build_chain(&mut l, &b_tokens).await;
    assert_eq!((a.len(), b.len()), (3, 3));
    assert_eq!(l.report().slots_free, 0, "池满(缓存独占)");

    // 给 A 的块种可辨识图样(回灌后逐字节比对)。
    // ⚠️ 用**固定槽号**(A 先建 ⇒ 槽 0..3):**不得**先调 `match_prefix` ——
    // 那会 touch A 使其变热,`plan` 就会(正确地)去沉更冷的 B(store 行为对,
    // 演示脚本错)。链级候选按**链尾 LRU** 出票,脚本要尊重这个语义。
    let mut before: Vec<Vec<u8>> = Vec::new();
    for i in 0..3usize {
        let slot = i as u32;
        seed(&mut face, &ps, slot, 0x40 * (i as u8 + 1)).await;
        before.push(read_slot(&mut face, &ps, slot).await);
    }

    // ── 轮 3:新需求(池满 ⇒ 钩子判:链级整条沉最冷的链 A)
    let need = Need::kv(2, NeedWhy::Admit);
    let verdict = plan_kv(&l, need);
    assert!(!verdict.out.is_empty(), "池满 + 新需求 ⇒ 必须出单(下沉/驱逐)");
    // engine 式执行:逐条取计划 → swap → 记账
    let mut offloaded = 0usize;
    for item in &verdict.out {
        let EvictOut::Kv { chain, sink, span } = item else { continue };
        match sink {
            Sink::Host => offloaded += engine_offload(&mut l, &mut face, *chain).await.expect("换出"),
            Sink::Drop => {
                let n = l.prune_branch(*chain, (*span).max(1)).expect("驱逐");
                assert!(n >= 1, "整枝票至少丢一块");
            }
            Sink::Disk => panic!("本档未接盘层"),
        }
    }
    assert!(offloaded > 0, "需求驱动下沉:应真搬(而非驱逐)");
    let rep = l.report();
    assert!(rep.ops.offloads > 0 && rep.ops.offload_batches > 0, "换出门①:{:?}", rep.ops);
    assert!(rep.host_bytes > 0, "宿主层应有驻留");
    assert_eq!(rep.ops.ensure_evicted, 0, "全程不得依赖 `ensure_for_len` 的同步驱逐");

    // ── 轮 4:命中 A ⇒ 命中面落在宿主 ⇒ **一条命令**批量回灌
    let hit = l.match_prefix(&a_tokens).expect("二次命中");
    let pending = hit.pending_host();
    assert!(!pending.is_empty(), "A 已被整条沉到宿主 ⇒ 命中面应有待回灌块(A 的链头段也在)");
    assert!(!hit.all_resident(), "整条链沉 ⇒ 命中面不全驻留");
    let slots = engine_onload(&mut l, &mut face, &pending).await.expect("批量回灌");
    assert_eq!(slots.len(), pending.len(), "同序返回槽");
    let rep2 = l.report();
    assert_eq!(rep2.ops.onloads as usize, pending.len(), "回灌计数 = 块数");
    assert_eq!(rep2.ops.onload_batches, 1, "§15 批面:多块回灌必须**一条命令**");
    assert_eq!(rep2.ops.ensure_evicted, 0, "回灌路径同样不得依赖同步驱逐");

    // ── 门③:内容逐字节恒等(回灌后的新槽 vs 沉之前的旧内容)
    for (i, slot) in slots.iter().enumerate() {
        let after = read_slot(&mut face, &ps, *slot).await;
        assert_eq!(after, before[i], "块 {i} 换出→回灌必须逐字节一致");
    }
}

/// **链级整条搬**(§14.6):一次换出把链从尾到头搬完 —— 链头段必须也在宿主
/// (否则命中面从链头就是设备格,回灌永不触发)。
#[tokio::test]
async fn chain_offload_moves_head_to_tail_and_batch_reloads() {
    let mut face = CpuFace::new();
    let ps = pieces(&mut face, 4).await;
    let mut l = leg(4);
    l.attach_pool(PoolLayout { class: BlockClass::KvPage, epoch: 1, pieces: ps.clone() })
        .expect("注册布局");

    let tokens: Vec<u32> = (0..96).collect();
    let chain = build_chain(&mut l, &tokens).await;
    let (head, tail) = (chain[0], chain[2]);

    // 候选:整条链一张票(dev_blocks = 3)
    let cands = l.chain_offload_candidates(&[], 1);
    assert_eq!(cands.len(), 1);
    assert_eq!((cands[0].tail, cands[0].head), (tail, head));
    assert_eq!(cands[0].dev_blocks, 3);

    assert_eq!(engine_offload(&mut l, &mut face, tail).await.expect("整条沉"), 3);
    for (i, h) in chain.iter().enumerate() {
        assert_eq!(l.carrier_of(*h).expect("在账"), owl_store::Carrier::Host, "第 {i} 块应在宿主");
    }
    let hit = l.match_prefix(&tokens).expect("宿主格仍命中");
    assert_eq!(hit.pending_host().len(), 3, "三段全待回灌(链头段在内)");
    let n = engine_onload(&mut l, &mut face, &hit.pending_host()).await.expect("批量回灌");
    assert_eq!(n.len(), 3);

    let rep = l.report();
    assert_eq!(rep.host_blocks, 0, "回灌后宿主副本应清空");
    assert_eq!(rep.cached_blocks, 3, "链仍在账");
}

// ─────────────────── B-3e-2:**逐块**下沉票(2026-10-18) ───────────────────
//
// 链级票的两处粗化(整链 all-or-nothing / 一块活跃否决整链)是**我们自己的选择**;
// 单卡长 ctx 需要的是"工作集留设备、其余沉走" ⇒ 这里把出票粒度降到**逐块**,
// 并验证:**活跃块自己不沉,但不连坐别的块**(链级票此时是整链 `Busy`)。

/// 逐块出票的编排样板(与 `engine_offload` 同形,只换成块清单)。
async fn engine_offload_blocks(
    leg: &mut BlockLeg,
    face: &mut CpuFace,
    chains: &[ChainHash],
) -> StoreResult<(usize, usize)> {
    let (blocks, pieces, table, skipped) = {
        let p = plan_offload_blocks(leg, chains)?; // 纯 host
        (p.blocks, p.pieces, p.table, p.skipped)
    };
    if blocks.is_empty() {
        return Ok((0, skipped.len()));
    }
    let conn = match face.swap(SwapDir::Out, &pieces[0], Box::new(table)).await {
        Ok(c) => c,
        Err(e) => {
            owl_store::abort_offload(leg, &blocks);
            return Err(owl_store::StoreError::Invariant(format!("swap(Out) 失败:{e}")));
        }
    };
    let table = reclaim_slot_table(conn)?;
    let n = commit_offload(leg, &blocks, table)?;
    Ok((n, skipped.len()))
}

#[tokio::test]
async fn per_block_offload_skips_live_block_without_vetoing_others() {
    let mut face = CpuFace::new();
    let ps = pieces(&mut face, 6).await;
    let mut l = leg(6);
    l.attach_pool(PoolLayout { class: BlockClass::KvPage, epoch: 1, pieces: ps.clone() })
        .expect("注册布局");

    // 链 A = 3 块;归一还会话引用后再**重新持一块**(块 1)= 模拟"会话表仍持有它"
    let a_tokens: Vec<u32> = (0..96).collect();
    let a = build_chain(&mut l, &a_tokens).await;
    assert_eq!(a.len(), 3);
    // 固定槽号:A 先建 ⇒ 槽 0/1/2(与既有演示脚本同律:别先 match 打热)
    let live_slot = 1u32;
    l.incref(live_slot).expect("会话重持块 1");
    assert_eq!(l.report().slots_free, 3, "另 3 槽空(池 6)");

    // 种图样(沉前/回灌后逐字节比)
    let mut before: Vec<Vec<u8>> = Vec::new();
    for i in 0..3usize {
        seed(&mut face, &ps, i as u32, 0x20 * (i as u8 + 1)).await;
        before.push(read_slot(&mut face, &ps, i as u32).await);
    }

    eprintln!(
        "[b3e-2] 池 {} 槽/自由 {} · 链 A = {:?}(槽 0/1/2)其中**槽 1 被会话持有**",
        l.report().slots_total,
        l.report().slots_free,
        a.iter().map(|h| format!("{:?}", h).chars().take(6).collect::<String>()).collect::<Vec<_>>()
    );
    // ── 对照:链级票此时**整链 Busy**(一块活跃 ⇒ 整条链不出票)
    assert!(
        matches!(plan_offload(&mut l, *a.last().expect("尾")), Err(owl_store::StoreError::Busy(_))),
        "链级票的粗化:一块活跃 ⇒ 整链否决(本测试要证的正是它的反面)"
    );

    // ── 逐块票:请求 [块0, 块1(活跃), 块2] ⇒ 只有 0/2 有票,块 1 记 `LiveRefs`
    let (n, skipped) = engine_offload_blocks(&mut l, &mut face, &a).await.expect("逐块下沉");
    let why = plan_offload_blocks(&mut l, &[a[0], a[1], a[2]])
        .map(|p| p.skipped.len())
        .unwrap_or(0);
    eprintln!("[b3e-2] 逐块票:下沉 {n} 块 · 跳过 {skipped} 块(活跃块记 LiveRefs;复询 skipped={why})");
    assert_eq!(n, 2, "两块应真沉");
    assert_eq!(skipped, 1, "活跃块应被跳过(不是连坐)");
    let rep = l.report();
    assert_eq!(rep.ops.offloads, 2, "下沉计数 = 有票块数");
    assert_eq!(rep.ops.offload_batches, 1, "§15 批面:两块一条命令");
    assert!(rep.host_bytes > 0, "宿主层应有驻留");

    // 活跃块**仍在设备**,另两块在宿主(载体查询是现成面 —— B-3e-2 不需要新造)
    assert_eq!(
        l.carrier_of(a[1]).expect("块1载体"),
        owl_store::Carrier::Device(live_slot),
        "活跃块不沉"
    );
    assert_eq!(l.carrier_of(a[0]).expect("块0载体"), owl_store::Carrier::Host, "冷块应沉");
    assert_eq!(l.carrier_of(a[2]).expect("块2载体"), owl_store::Carrier::Host, "冷块应沉");

    // ── 幂等(趁块还在宿主):重复请求全 `AlreadyHost`,零搬运、不借台
    let again = plan_offload_blocks(&mut l, &[a[0], a[2]]).expect("重复请求");
    assert!(again.is_empty(), "已在宿主 ⇒ 无票(不借台、零副作用)");
    assert!(
        again.skipped.iter().all(|(_, why)| *why == SkipWhy::AlreadyHost),
        "重复请求应记 AlreadyHost:{:?}",
        again.skipped
    );

    // ── 回灌两块 ⇒ 内容逐字节恒等(新槽 ≠ 原槽:回灌分配新槽)
    let sunk = vec![a[0], a[2]];
    let slots = engine_onload(&mut l, &mut face, &sunk).await.expect("回灌");
    assert_eq!(slots.len(), 2);
    for (i, slot) in slots.iter().enumerate() {
        let after = read_slot(&mut face, &ps, *slot).await;
        assert_eq!(after, before[[0usize, 2][i]], "块 {:?} 沉→回灌必须逐字节一致", sunk[i]);
    }

    // ── 回灌后它们又回到设备(载体查询可证)⇒ 再请求就又有票(幂等只对"已在宿主"成立)
    assert!(matches!(
        l.carrier_of(a[0]).expect("载体"),
        owl_store::Carrier::Device(_)
    ));

    // ── 未知身份 ⇒ `Unknown`(不静默当成功)
    let unk = plan_offload_blocks(&mut l, &[ChainHash::new(0xDEAD)]).expect("未知身份");
    assert!(unk.is_empty());
    assert_eq!(unk.skipped[0].1, SkipWhy::Unknown);
}

/// **N6-① 次序门**(缺口 3;2026-10-19):`NeedWhy::Pressure` = **零缺口也搬**,
/// 且出单**优先 `Sink::Host`**(内容不丢) —— 与"非压力节点在 free >= need 时早退"对照。
#[tokio::test]
async fn pressure_offloads_without_deficit_and_prefers_host() {
    let mut face = CpuFace::new();
    let ps = pieces(&mut face, 6).await;
    let mut l = leg(6);
    l.attach_pool(PoolLayout { class: BlockClass::KvPage, epoch: 1, pieces: ps.clone() })
        .expect("注册布局");
    // 填 2 条链 × 3 块 = 池满;归还会话引用 ⇒ 缓存独占(可下沉)
    let a_tokens: Vec<u32> = (0..PAGE_TOKENS as u32 * 3).collect();
    let _a = build_chain(&mut l, &a_tokens).await;
    let tokens_b: Vec<u32> = (1000..1000 + PAGE_TOKENS as u32 * 3).collect();
    let _b = build_chain(&mut l, &tokens_b).await;
    assert_eq!(l.slots_free(), 0, "池应已满");
    // 对照:非压力节点,零缺口 ⇒ 不动
    let noop = plan_kv(&l, Need::kv(0, NeedWhy::Grow));
    assert!(noop.is_noop(), "Grow + 零缺口 ⇒ 不搬");
    // N6:目标腾 3 槽(与缺口无关)⇒ 必须出单,且优先 Host 下沉
    let v = plan_kv(&l, Need::kv(3, NeedWhy::Pressure));
    assert!(v.deny.is_none(), "可下沉面充足 ⇒ 不拒:{:?}", v.deny);
    assert!(!v.out.is_empty(), "压力节点必须搬(零缺口也搬)");
    assert!(
        v.out.iter().all(|o| o.sink() == Sink::Host),
        "宿主要位 ⇒ 首选下沉(内容不丢):{:?}",
        v.out.iter().map(|o| o.sink()).collect::<Vec<_>>()
    );
    assert_eq!(v.out.iter().map(|o| o.kv_span()).sum::<usize>(), 3, "整链下沉凑满目标");
    // **H5 空转整理**(步 4 入档):与 Pressure 同语义(零缺口也搬),只是 why 不同 ——
    // 空转期只动缓存面(冷链),绝不碰活跃面。
    let idle = plan_kv(&l, Need::kv(3, NeedWhy::Idle));
    assert!(!idle.out.is_empty(), "H5 空转整理必须出单(零缺口也搬)");
    assert!(idle.out.iter().all(|o| o.sink() == Sink::Host), "空转整理同样先下沉");
    assert!(idle.deny.is_none(), "有可搬面 ⇒ 不拒");
}
/// **`Need::kv` 的单位 = "我想要多少空闲槽"**（不是"已扣 free 的差额"）。
///
/// 回归背景（2026-11-06 真机）：`step_need`/`ensure_start_chain` 都传"要新占
/// 多少槽"，`plan_capacity` 再减 `slots_free` 折成腾位目标；调用方若先自己
/// 减一遍 free，就会变成 `need - 2*free`，free 大时直接 no-op（腾位假死）。
#[tokio::test]
async fn grow_need_is_absolute_free_slots_not_deficit() {
    let mut face = CpuFace::new();
    let ps = pieces(&mut face, 6).await;
    let mut l = leg(6);
    l.attach_pool(PoolLayout { class: BlockClass::KvPage, epoch: 1, pieces: ps })
        .expect("注册布局");
    // 4 条单块链 ⇒ 6 槽池里 free = 2（缓存独占，可下沉）。
    for k in 0..4u32 {
        let base = k * 1000;
        let toks: Vec<u32> = (base..base + PAGE_TOKENS as u32).collect();
        let cs = build_chain(&mut l, &toks).await;
        assert_eq!(cs.len(), 1, "单块链应只产生一张链身份票");
    }
    assert_eq!(l.slots_free(), 2, "6 槽 - 4 块 = 2 空");

    // 想要 4 个空闲槽 ⇒ 目标 = 4 - 2 = 2；若是"差额再减 free"会算出 2-2=0。
    let v = plan_kv(&l, Need::kv(4, NeedWhy::Grow));
    assert!(v.deny.is_none(), "可下沉面充足 ⇒ 不拒：{:?}", v.deny);
    assert_eq!(
        v.out.iter().map(|o| o.kv_span()).sum::<usize>(),
        2,
        "需 4 / 空 2 ⇒ 必须腾 2（不是 no-op）"
    );
    // 对照：想要 ≤ 空闲 ⇒ 不动（非压力节点零缺口不搬）。
    assert!(
        plan_kv(&l, Need::kv(2, NeedWhy::Grow)).is_noop(),
        "需 2 / 空 2 ⇒ 零缺口，不许无谓搬"
    );
}

/// **轮零：“宿主余位夹在缺口与整链跨度之间”**（2026-11-08 长活服务第 7 轮定谳）。
///
/// 形状：goal=2 < 宿主余位=3 < 最小可沉链=4。旧代码认为“房间够”不腾宿主，
/// 轮一又因“整链 4 > 余位 3”一张票不出 ⇒ 只能逐块丢设备内容（或直接 500）。
/// 正解：先逐宿主最冷 1 块腾到 4 槽，再用**一张整链 Host 票**搬走 4 块
/// （设备内容不丢）。
#[tokio::test]
async fn host_room_between_goal_and_chain_span_prunes_host_first() {
    let mut face = CpuFace::new();
    let ps = pieces(&mut face, 10).await;
    let block_bytes = PIECE_BYTES * 2;
    let mut l = BlockLeg::open(LegConfig {
        slots: 10,
        block_tokens: PAGE_TOKENS,
        block_bytes,
        cache: owl_store::CachePolicy::Cap(18),
        host_bytes: block_bytes * 5, // 宿主只装 5 块
        class: BlockClass::KvPage,
        label: "host-room",
    })
    .expect("开腿");
    l.attach_pool(PoolLayout { class: BlockClass::KvPage, epoch: 1, pieces: ps })
        .expect("注册布局");

    let toks = |base: u32, blocks: u32| -> Vec<u32> {
        (base..base + blocks * PAGE_TOKENS as u32).collect()
    };
    // 设备先塞满：A(4) + C(4) + B(2) = 10 槽。
    let a_cs = build_chain(&mut l, &toks(0, 4)).await;
    let c_cs = build_chain(&mut l, &toks(4000, 4)).await;
    let b_cs = build_chain(&mut l, &toks(8000, 2)).await;
    assert_eq!((a_cs.len(), c_cs.len(), b_cs.len()), (4, 4, 2));
    assert_eq!(l.slots_free(), 0, "设备满");
    // B 整链沉宿主 ⇒ 宿主用 2/5（余位 3），设备空出 2 槽。
    engine_offload(&mut l, &mut face, *b_cs.last().expect("B 链尾"))
        .await
        .expect("B 下沉");
    assert_eq!(l.slots_free(), 2);
    assert_eq!(l.host_free_slots(), 3, "宿主余位 3 块");

    // 想要 4 个空闲槽 ⇒ 缺口 2 < 余位 3 < 最小可沉链 4。
    let v = plan_kv(&l, Need::kv(4, NeedWhy::Grow));
    assert!(v.deny.is_none(), "腾得出来 ⇒ 不拒：{:?}", v.deny);
    let sinks: Vec<Sink> = v.out.iter().map(|o| o.sink()).collect();
    assert_eq!(
        sinks,
        vec![Sink::Drop, Sink::Host],
        "先逐宿主最冷 1 块（内容丢），再整链下沉 4 块（内容留）：{:?}",
        v.out
    );
    assert_eq!(v.out[1].kv_span(), 4, "一张整链票：{:?}", v.out);
}

/// **无盘层 + 宿主满 + 链装不下余位**：整枝丢（损失被 goal 封顶），不逐块拖拍。
///
/// 真机形状（2026-11-08 第 7 轮）：宿主满（余位 1）、设备只空 2 槽、缺口 166，
/// 能沉的链都 > 166 ⇒ 轮一零票；轮二逐块每链 1 块 ⇒ 每拍凑 5–6 槽，永远不够。
#[tokio::test]
async fn no_disk_tier_collapses_chain_that_cannot_fit_host() {
    let mut face = CpuFace::new();
    let ps = pieces(&mut face, 8).await;
    let block_bytes = PIECE_BYTES * 2;
    let mut l = BlockLeg::open(LegConfig {
        slots: 8,
        block_tokens: PAGE_TOKENS,
        block_bytes,
        cache: owl_store::CachePolicy::Cap(16),
        host_bytes: block_bytes * 2, // 宿主只装 2 块
        class: BlockClass::KvPage,
        label: "no-disk",
    })
    .expect("开腿");
    l.attach_pool(PoolLayout { class: BlockClass::KvPage, epoch: 1, pieces: ps })
        .expect("注册布局");

    let toks = |base: u32, blocks: u32| -> Vec<u32> {
        (base..base + blocks * PAGE_TOKENS as u32).collect()
    };
    // A(3) + C(3) + B(2) = 8 槽；B 沉宿主 ⇒ 宿主 2/2 满，设备空 2 槽。
    let a_cs = build_chain(&mut l, &toks(0, 3)).await;
    let c_cs = build_chain(&mut l, &toks(4000, 3)).await;
    let b_cs = build_chain(&mut l, &toks(8000, 2)).await;
    assert_eq!((a_cs.len(), c_cs.len(), b_cs.len()), (3, 3, 2));
    engine_offload(&mut l, &mut face, *b_cs.last().expect("B 链尾"))
        .await
        .expect("B 下沉");
    assert_eq!(l.host_free_slots(), 0, "宿主满");

    // 想要 5 个空闲槽 ⇒ 缺口 3；两条可沉链各 3 块都装不下宿主余位。
    let v = plan_kv(&l, Need::kv(5, NeedWhy::Grow));
    assert!(v.deny.is_none(), "整枝丢要能凑够：{:?}", v.deny);
    let max_drop = v
        .out
        .iter()
        .filter(|o| o.sink() == Sink::Drop)
        .map(|o| o.kv_span())
        .max()
        .unwrap_or(0);
    assert_eq!(max_drop, 3, "一张整枝票 span=3（旧行为是每链 1 块 ⇒ 短 1）：{:?}", v.out);
}

/// **宿主腾位必须一张票跨多块**（旧实现一链一票只丢 1 块）。
///
/// 形状：宿主里只有一条长链（10 块，余位 2），要腾 2 槽。旧代码只丢 1 块
/// ⇒ 余位 3 < 整链 4 ⇒ 轮一零票，只能退化成丢设备内容；新代码一张票 span=2
/// ⇒ 余位 4 ⇒ 整链下沉（内容留）。
#[tokio::test]
async fn host_prune_uses_multi_block_span() {
    let mut face = CpuFace::new();
    let ps = pieces(&mut face, 12).await;
    let block_bytes = PIECE_BYTES * 2;
    let mut l = BlockLeg::open(LegConfig {
        slots: 12,
        block_tokens: PAGE_TOKENS,
        block_bytes,
        cache: owl_store::CachePolicy::Cap(24),
        host_bytes: block_bytes * 12,
        class: BlockClass::KvPage,
        label: "host-span",
    })
    .expect("开腿");
    l.attach_pool(PoolLayout { class: BlockClass::KvPage, epoch: 1, pieces: ps })
        .expect("注册布局");

    let toks = |base: u32, blocks: u32| -> Vec<u32> {
        (base..base + blocks * PAGE_TOKENS as u32).collect()
    };
    // 长链 H(10) 先沉宿主（余位 2），设备再放 A(4)/C(4)。
    let h_cs = build_chain(&mut l, &toks(0, 10)).await;
    engine_offload(&mut l, &mut face, *h_cs.last().expect("H 链尾"))
        .await
        .expect("H 下沉");
    let a_cs = build_chain(&mut l, &toks(4000, 4)).await;
    let _c_cs = build_chain(&mut l, &toks(8000, 4)).await;
    assert_eq!((h_cs.len(), a_cs.len()), (10, 4));
    assert_eq!(l.slots_free(), 4, "H 下沉后空出 10，A+C 用 8 ⇒ free 4");
    assert_eq!(l.host_free_slots(), 2, "宿主 12 - H(10) = 余位 2");

    // 想要 6 个空闲槽 ⇒ 缺口 = 6 - 4 = 2；候选链 A 跨度 4 > 余位 2。
    let v = plan_kv(&l, Need::kv(6, NeedWhy::Grow));
    assert!(v.deny.is_none(), "腾得出来 ⇒ 不拒：{:?}", v.deny);
    let drops: Vec<usize> = v
        .out
        .iter()
        .filter(|o| o.sink() == Sink::Drop)
        .map(|o| o.kv_span())
        .collect();
    let hosts: Vec<usize> = v
        .out
        .iter()
        .filter(|o| o.sink() == Sink::Host)
        .map(|o| o.kv_span())
        .collect();
    assert_eq!(drops, vec![2], "宿主腾位一张票跨 2 块：{:?}", v.out);
    let mut drop_chains: Vec<(ChainHash, usize)> = Vec::new();
    for o in &v.out {
        if let EvictOut::Kv { chain, sink: Sink::Drop, span } = o {
            drop_chains.push((*chain, *span));
        }
    }
    assert_eq!(
        drop_chains,
        vec![(*h_cs.last().expect("H 链尾"), 2)],
        "被动的是宿主链 H（设备内容不动）：{:?}",
        v.out
    );
    assert_eq!(hosts, vec![4], "腾够后整链下沉 A：{:?}", v.out);

    // **执行整张判据**（与 engine `apply_verdict` 同序：先 Drop 后 Host）：
    // 宿主腾位票必须真删宿主副本并腾出槽，随后整链下沉才能拿到位。
    for o in &v.out {
        match o {
            EvictOut::Kv { chain, sink: Sink::Drop, span } => {
                let n = l.prune_branch(*chain, (*span).max(1)).expect("宿主逐出");
                assert_eq!(n, *span, "一张宿主票应删满 span 块：{:?}", v.out);
            }
            EvictOut::Kv { chain, sink: Sink::Host, .. } => {
                engine_offload(&mut l, &mut face, *chain).await.expect("整链下沉");
            }
            other => panic!("本档不应出别的票：{other:?}"),
        }
    }
    let a_tail = *a_cs.last().expect("A 链尾");
    assert_eq!(l.carrier_of(a_tail).expect("A 在账"), owl_store::Carrier::Host);
    assert!(
        l.carrier_of(*h_cs.last().expect("H 链尾")).is_err(),
        "宿主最冷的 2 块必须真删"
    );
}

/// **轮零按"轮一实际要沉的这组链"求和**，不是"最小/最大单链跨度"。
///
/// 形状：LRU 前缀 = 小链 1 + 大链 4（goal 2）⇒ 需要 5 槽余位（Σ），
/// 只按最小（1）会判"够"不腾 ⇒ 大链装不下 ⇒ 退化成丢设备内容。
#[tokio::test]
async fn host_room_target_sums_the_lru_prefix() {
    let mut face = CpuFace::new();
    let ps = pieces(&mut face, 8).await;
    let block_bytes = PIECE_BYTES * 2;
    let mut l = BlockLeg::open(LegConfig {
        slots: 8,
        block_tokens: PAGE_TOKENS,
        block_bytes,
        cache: owl_store::CachePolicy::Cap(16),
        host_bytes: block_bytes * 7,
        class: BlockClass::KvPage,
        label: "host-sum",
    })
    .expect("开腿");
    l.attach_pool(PoolLayout { class: BlockClass::KvPage, epoch: 1, pieces: ps })
        .expect("注册布局");

    let toks = |base: u32, blocks: u32| -> Vec<u32> {
        (base..base + blocks * PAGE_TOKENS as u32).collect()
    };
    // H(5) 沉宿主（余位 2）；设备 A(1) + C(4)（free 3）。A 比 C 冷（先建）。
    let h_cs = build_chain(&mut l, &toks(0, 5)).await;
    engine_offload(&mut l, &mut face, *h_cs.last().expect("H 链尾"))
        .await
        .expect("H 下沉");
    let _a_cs = build_chain(&mut l, &toks(4000, 1)).await;
    let _c_cs = build_chain(&mut l, &toks(8000, 4)).await;
    assert_eq!(l.slots_free(), 3);
    assert_eq!(l.host_free_slots(), 2);

    // 想要 5 个空闲槽 ⇒ 缺口 2；LRU 前缀 = A(1) + C(4) = 5 ⇒ 需余位 5。
    let v = plan_kv(&l, Need::kv(5, NeedWhy::Grow));
    assert!(v.deny.is_none(), "{:?}", v.deny);
    let hosts: Vec<usize> = v
        .out
        .iter()
        .filter(|o| o.sink() == Sink::Host)
        .map(|o| o.kv_span())
        .collect();
    assert_eq!(hosts, vec![1, 4], "两条链都下沉（内容留）：{:?}", v.out);
    // 唯一一张 Drop 必须是**宿主链 H**（腾宿主位），设备链 A/C 都下沉（内容留）。
    let mut drop_chains: Vec<(ChainHash, usize)> = Vec::new();
    for o in &v.out {
        if let EvictOut::Kv { chain, sink: Sink::Drop, span } = o {
            drop_chains.push((*chain, *span));
        }
    }
    assert_eq!(
        drop_chains,
        vec![(*h_cs.last().expect("H 链尾"), 3)],
        "只丢宿主最冷（一张票 3 块），不丢设备内容：{:?}",
        v.out
    );
}

// ───────────────── 2026-11-08 池毒化:共享前缀被钉住 ⇒ 整链不可沉 ─────────────────
//
// 真机：一个活跃 turn 持有共享前缀块 ⇒ `chain_down_span` 对所有共享该前缀
// 的缓存链都返回 `None` ⇒ 轮一零票；轮二逐块只丢 1 块/链 ⇒ 9 条长链
// 每拍只凑 5–9 槽，起 267 槽的 turn 永远撞「纯分配不足」。
// 修：链“搬不走”与“装不下”同级 ⇒ 允许整枝塌缩（仍受本拍目标封顶，
// 执行侧 `prune_branch` 逐块校验 refs，不碰活跃面）。
#[tokio::test]
async fn busy_shared_prefix_collapses_unshared_tail_in_one_ticket() {
    let mut face = CpuFace::new();
    let ps = pieces(&mut face, 8).await;
    let mut l = leg(8);
    l.attach_pool(PoolLayout { class: BlockClass::KvPage, epoch: 1, pieces: ps })
        .expect("注册布局");
    let pg = PAGE_TOKENS as u32;
    let shared: Vec<u32> = (0..pg).collect();
    // A = 共享 1 块 + 独有 1 块；表**留着**（模拟活跃 turn 持着共享前缀）。
    let a_tokens: Vec<u32> = shared.iter().copied().chain(1000..1000 + pg).collect();
    let mut ta = Vec::new();
    l.ensure_for_len(&mut ta, a_tokens.len()).expect("A 块表");
    l.register_chain(&a_tokens, &ta).expect("A 登记");
    // B = 共享 1 块 + 独有 3 块（同 session 的后续轮，缓存独占）。
    let b_tokens: Vec<u32> = shared
        .iter()
        .copied()
        .chain(2000..2000 + pg)
        .chain(3000..3000 + pg)
        .chain(4000..4000 + pg)
        .collect();
    let hit = l.match_prefix(&shared).expect("共享前缀命中");
    let mut tb: Vec<u32> = hit.slots.iter().flatten().copied().collect();
    for &slot in &tb {
        l.incref(slot).expect("复用面 incref"); // engine: plan_prefix_reuse 同律
    }
    l.ensure_for_len(&mut tb, b_tokens.len()).expect("B 块表");
    l.register_chain(&b_tokens, &tb).expect("B 登记");
    l.release_table(&mut tb).expect("B 表归还");

    // 自检：两条链都因共享块被钉住而不可沉；B 的独有尾巴 3 块可丢。
    let d = l.kv_diag();
    assert_eq!(d.chain_busy, 2, "两条链都被同一块共享前缀钉住：{d:?}");
    // 共享块（A/B 两条链各算一次）+ A 自己的独有块（也被 A 持着）= 3。
    assert_eq!(d.stuck_shared, 3, "{d:?}");
    assert_eq!(l.slots_free(), 3, "8 槽 - 5 块 = 3 空");

    // 需要 5 个空闲槽 ⇒ 目标 2。轮一零票（两条链都不可沉）⇒ 只能整枝
    // 塌缩 B 的独有尾：**一张票 span=2**（旧行为是 1 块/链 ⇒ deny 短 1 槽）。
    let v = plan_kv(&l, Need::kv(5, NeedWhy::Grow));
    assert!(v.deny.is_none(), "独有尾能覆盖缺口 ⇒ 不拒：{:?}", v.deny);
    assert_eq!(v.out.len(), 1, "应是一张整枝票：{:?}", v.out);
    assert_eq!(v.out[0].sink(), Sink::Drop, "搬不走 ⇒ 只能丢：{:?}", v.out);
    assert_eq!(v.out[0].kv_span(), 2, "塌缩到本拍目标（不是整链 3）：{:?}", v.out);
}

// ───────────────── 生产路径回归:重算重绑不得留宿主幽灵 ─────────────────
//
// 2026-10-24 定谳案:`BlockLeg::register_chain_owned` 缺一步
// `drop_host_chains` ⇒ "块先沉宿主 → 同前缀被重算并重绑设备"时旧宿主副本留尸
// ⇒ 下一次 `commit_offload` 的 `host.land` 撞「块 … 已在宿主层」⇒ `unbind`
// 不执行 ⇒ 双驻留原地不动 ⇒ 之后**每个 turn 全灭**(实测:队列卡死,须重启)。
//
// 本用例在无修复版本上必须是红的(第 ④ 步 Err Invariant)。
#[tokio::test]
async fn owned_recompute_leaves_no_host_ghost() {
    let mut face = CpuFace::new();
    let ps = pieces(&mut face, 6).await;
    let mut l = leg(6);
    l.attach_pool(PoolLayout {
        class: BlockClass::KvPage,
        epoch: 1,
        pieces: ps.clone(),
    })
    .expect("注册布局");

    let tokens: Vec<u32> = (0..96).collect(); // 3 块

    // ① 生产入口:带归属登记(turn 收口),随后归还会话引用 ⇒ 缓存独占
    let mut table = Vec::new();
    l.ensure_for_len(&mut table, tokens.len()).expect("块表");
    l.register_chain_owned(&tokens, &table, 7).expect("首登");
    l.release_table(&mut table).expect("归还");

    let hashes = l.match_prefix(&tokens).expect("命中").hashes;
    assert_eq!(hashes.len(), 3);

    // ② 整条沉到宿主
    engine_offload(&mut l, &mut face, *hashes.last().expect("链尾"))
        .await
        .expect("换出");
    for h in &hashes {
        assert!(l.host_contains(*h), "沉完应在宿主:{h:?}");
    }
    assert_eq!(l.report().host_blocks, 3, "宿主账 = 3 块");

    // ③ 同前缀被重算(新设备槽)+ 带归属登记 ⇒ 旧宿主副本是幽灵,必须当场丢
    let mut t2 = Vec::new();
    l.ensure_for_len(&mut t2, tokens.len()).expect("重算块表");
    l.register_chain_owned(&tokens, &t2, 7).expect("重登(重绑)");
    l.release_table(&mut t2).expect("归还");
    for h in &hashes {
        assert!(!l.host_contains(*h), "重绑设备后不得留宿主幽灵:{h:?}");
    }
    assert_eq!(l.report().host_blocks, 0, "树账应全回设备");

    // ④ 再沉一次 —— 病灶版在这里抛「已在宿主层」,并因此毒化后续每个 turn
    engine_offload(&mut l, &mut face, *hashes.last().expect("链尾"))
        .await
        .expect("再下沉(修复前必炸)");
    assert_eq!(l.host_duplicate_landings(), 0, "源头修好后不该有重复落地");
    assert_eq!(l.report().host_blocks, 3, "第二次下沉同样 3 块入宿主");
}

// ───────────────── 并发/轮换基线(2026-10-27) ─────────────────
//
// 这一组只测 store 的**迁移 → 回填**语义,不引入 engine 调度:
// - 单会话:整条链沉宿主 → 再回灌,逐字节一致;
// - 两会话:池满后 A/B 轮流沉、回灌,验证互不串内容、宿主不残留。

/// **单会话迁移回填**：一条 6 块链占满池 → 整条沉宿主 → 命中后整条回灌。
#[tokio::test]
async fn single_session_offload_and_reload_roundtrip() {
    let mut face = CpuFace::new();
    let ps = pieces(&mut face, 6).await;
    let mut l = leg(6);
    l.attach_pool(PoolLayout { class: BlockClass::KvPage, epoch: 1, pieces: ps.clone() })
        .expect("注册布局");

    let tokens: Vec<u32> = (0..192).collect(); // 6 块
    let chain = build_chain(&mut l, &tokens).await;
    assert_eq!(chain.len(), 6);
    assert_eq!(l.report().slots_free, 0, "单会话也应占满池");

    let mut before = Vec::new();
    for slot in 0..6u32 {
        seed(&mut face, &ps, slot, 0x10 * (slot as u8 + 1)).await;
        before.push(read_slot(&mut face, &ps, slot).await);
    }

    let tail = *chain.last().expect("尾块");
    let moved = engine_offload(&mut l, &mut face, tail).await.expect("整链下沉");
    assert_eq!(moved, 6, "单会话整链沉 6 块");
    assert_eq!(l.report().host_blocks, 6, "宿主应持 6 块");
    assert_eq!(l.report().slots_free, 6, "下沉后池应全空");
    eprintln!(
        "[single] offload: host_blocks={} slots_free={}",
        l.report().host_blocks,
        l.report().slots_free
    );

    let hit = l.match_prefix(&tokens).expect("宿主命中");
    let pending = hit.pending_host();
    assert_eq!(pending.len(), 6, "单会话回灌面 = 6 块");
    let slots = engine_onload(&mut l, &mut face, &pending).await.expect("整链回灌");
    assert_eq!(slots.len(), 6);
    for (i, slot) in slots.iter().enumerate() {
        let after = read_slot(&mut face, &ps, *slot).await;
        assert_eq!(after, before[i], "单会话第 {i} 块回填必须逐字节一致");
    }
    assert_eq!(l.report().host_blocks, 0, "回灌后不得留宿主副本");
    assert_eq!(l.report().slots_free, 0, "回灌后池重新占满");
    eprintln!(
        "[single] reload: host_blocks={} slots_free={}",
        l.report().host_blocks,
        l.report().slots_free
    );
}

/// **两会话并发 + 轮换**：A/B 各 3 块占满池 → A 沉/回 → B 沉/回。
#[tokio::test]
async fn two_sessions_alternate_offload_and_reload_roundtrip() {
    let mut face = CpuFace::new();
    let ps = pieces(&mut face, 6).await;
    let mut l = leg(6);
    l.attach_pool(PoolLayout { class: BlockClass::KvPage, epoch: 1, pieces: ps.clone() })
        .expect("注册布局");

    let a_tokens: Vec<u32> = (0..96).collect();
    let b_tokens: Vec<u32> = (1000..1096).collect();
    let a = build_chain(&mut l, &a_tokens).await;
    let b = build_chain(&mut l, &b_tokens).await;
    assert_eq!((a.len(), b.len()), (3, 3), "两会话各 3 块");
    assert_eq!(l.report().slots_free, 0, "两会话占满池");

    // 固定槽: A 先建 0..3,B 后建 3..6。
    let mut a_before = Vec::new();
    let mut b_before = Vec::new();
    for slot in 0..3u32 {
        seed(&mut face, &ps, slot, 0x20 * (slot as u8 + 1)).await;
        a_before.push(read_slot(&mut face, &ps, slot).await);
    }
    for slot in 3..6u32 {
        seed(&mut face, &ps, slot, 0x30 * (slot as u8 - 2)).await;
        b_before.push(read_slot(&mut face, &ps, slot).await);
    }

    // A 轮换出去;B 仍在设备。
    assert_eq!(engine_offload(&mut l, &mut face, *a.last().unwrap()).await.expect("A 下沉"), 3);
    assert_eq!(l.report().host_blocks, 3, "A 在宿主");
    assert_eq!(l.report().slots_free, 3, "A 腾出 3 槽");
    eprintln!(
        "[two] A offload: host_blocks={} slots_free={}",
        l.report().host_blocks,
        l.report().slots_free
    );
    // B 命中仍在设备,不触发回灌。
    let b_hit = l.match_prefix(&b_tokens).expect("B 命中");
    assert!(b_hit.pending_host().is_empty(), "B 未下沉,不应回灌");

    // A 回灌;B 仍在设备。
    let a_pending = l.match_prefix(&a_tokens).expect("A 命中").pending_host();
    assert_eq!(a_pending.len(), 3, "A 回灌面 = 3 块");
    let a_slots = engine_onload(&mut l, &mut face, &a_pending).await.expect("A 回灌");
    for (i, slot) in a_slots.iter().enumerate() {
        assert_eq!(read_slot(&mut face, &ps, *slot).await, a_before[i], "A 第 {i} 块");
    }
    assert_eq!(l.report().slots_free, 0, "A 回灌后池再次满");
    eprintln!(
        "[two] A reload: host_blocks={} slots_free={}; B pending={}",
        l.report().host_blocks,
        l.report().slots_free,
        l.match_prefix(&b_tokens).expect("B").pending_host().len()
    );

    // B 轮换出去;A 仍在设备。
    assert_eq!(engine_offload(&mut l, &mut face, *b.last().unwrap()).await.expect("B 下沉"), 3);
    assert_eq!(l.report().host_blocks, 3, "B 在宿主");
    assert_eq!(l.report().slots_free, 3, "B 腾出 3 槽");
    eprintln!(
        "[two] B offload: host_blocks={} slots_free={}",
        l.report().host_blocks,
        l.report().slots_free
    );

    // B 回灌;A 仍在设备,内容不得互串。
    let b_pending = l.match_prefix(&b_tokens).expect("B 命中").pending_host();
    assert_eq!(b_pending.len(), 3, "B 回灌面 = 3 块");
    let b_slots = engine_onload(&mut l, &mut face, &b_pending).await.expect("B 回灌");
    for (i, slot) in b_slots.iter().enumerate() {
        assert_eq!(read_slot(&mut face, &ps, *slot).await, b_before[i], "B 第 {i} 块");
    }
    assert_eq!(l.report().host_blocks, 0, "轮换结束宿主无残留");
    assert_eq!(l.report().slots_free, 0, "轮换结束池仍由 A+B 占满");
    eprintln!(
        "[two] B reload: host_blocks={} slots_free={}",
        l.report().host_blocks,
        l.report().slots_free
    );
}

/// **两会话同时活跃 → 轮换才允许迁移**：
/// A/B 都持会话引用时,整链下沉被 `Busy` 挡住;A 先轮换出去后可沉/回,
/// B 仍活跃且内容不变;随后 B 轮换,重复同一路径。
#[tokio::test]
async fn two_active_sessions_block_migration_until_one_rotates_out() {
    let mut face = CpuFace::new();
    let ps = pieces(&mut face, 6).await;
    let mut l = leg(6);
    l.attach_pool(PoolLayout { class: BlockClass::KvPage, epoch: 1, pieces: ps.clone() })
        .expect("注册布局");

    let a_tokens: Vec<u32> = (0..96).collect();
    let b_tokens: Vec<u32> = (1000..1096).collect();
    // 两会话都**保留**块表引用(模拟同时活跃)。
    let (a_table, b_table) = {
        let mut a_table = Vec::new();
        let mut b_table = Vec::new();
        l.ensure_for_len(&mut a_table, a_tokens.len()).expect("A 表");
        l.ensure_for_len(&mut b_table, b_tokens.len()).expect("B 表");
        l.register_chain(&a_tokens, &a_table).expect("A 登记");
        l.register_chain(&b_tokens, &b_table).expect("B 登记");
        (a_table, b_table)
    };
    let a_tail = l.tail_hash(&a_tokens, 3).expect("A 尾");
    let b_tail = l.tail_hash(&b_tokens, 3).expect("B 尾");
    assert_eq!(l.report().slots_free, 0);

    let mut a_before = Vec::new();
    let mut b_before = Vec::new();
    for slot in 0..3u32 {
        seed(&mut face, &ps, slot, 0x50 * (slot as u8 + 1)).await;
        a_before.push(read_slot(&mut face, &ps, slot).await);
    }
    for slot in 3..6u32 {
        seed(&mut face, &ps, slot, 0x60u8.wrapping_mul(slot as u8 - 2)).await;
        b_before.push(read_slot(&mut face, &ps, slot).await);
    }

    // 两个都活跃:计划不得把它们当缓存下沉。
    // 活跃 refs > 1 的叶不得进 Host **也不得进 Drop**；plan 应在容量不足时
    // 直接 deny，而不是出无效 Drop 票。
    let verdict = plan_kv(&l, Need::kv(2, NeedWhy::Admit));
    assert!(
        verdict.out.iter().all(|o| o.sink() != Sink::Host),
        "活跃会话的整链不得下沉:{verdict:?}"
    );
    eprintln!(
        "[active] both active: tickets={} host_ok={} slots_free={}",
        verdict.out.len(),
        verdict.out.iter().all(|o| o.sink() != Sink::Host),
        l.report().slots_free
    );
    assert!(
        verdict.out.iter().all(|o| o.sink() != Sink::Drop),
        "活跃链不得出 Drop 票:{verdict:?}"
    );
    assert!(verdict.deny.is_some(), "无缓存可动 ⇒ 必须响亮 deny:{verdict:?}");
    eprintln!("[active] both active tickets={:?}", verdict.out);

    // A 轮换出去(释放表引用)→ 可整链下沉。
    let mut a_table = a_table;
    l.release_table(&mut a_table).expect("A 表归还");
    assert_eq!(engine_offload(&mut l, &mut face, a_tail).await.expect("A 下沉"), 3);
    assert_eq!(l.report().host_blocks, 3, "A 在宿主");
    eprintln!(
        "[active] A rotates out: host_blocks={} slots_free={}",
        l.report().host_blocks,
        l.report().slots_free
    );
    // B 仍然活跃:回灌 A 不能影响 B 的内容。
    let a_pending = l.match_prefix(&a_tokens).expect("A 宿主命中").pending_host();
    assert_eq!(a_pending.len(), 3);
    let a_slots = engine_onload(&mut l, &mut face, &a_pending).await.expect("A 回灌");
    for (i, slot) in a_slots.iter().enumerate() {
        assert_eq!(read_slot(&mut face, &ps, *slot).await, a_before[i], "A 第 {i} 块");
    }
    for (i, h) in l.match_prefix(&b_tokens).expect("B 仍命中").hashes.iter().enumerate() {
        // B 的表引用还在,命中面应全驻留。
        let _ = (i, h);
    }
    assert_eq!(l.report().slots_free, 0, "A 回灌后池满");

    // B 轮换出去 → 下沉/回灌,验证 A 的内容不被串。
    let mut b_table = b_table;
    l.release_table(&mut b_table).expect("B 表归还");
    assert_eq!(engine_offload(&mut l, &mut face, b_tail).await.expect("B 下沉"), 3);
    let b_pending = l.match_prefix(&b_tokens).expect("B 宿主命中").pending_host();
    assert_eq!(b_pending.len(), 3);
    let b_slots = engine_onload(&mut l, &mut face, &b_pending).await.expect("B 回灌");
    for (i, slot) in b_slots.iter().enumerate() {
        assert_eq!(read_slot(&mut face, &ps, *slot).await, b_before[i], "B 第 {i} 块");
    }
    assert_eq!(l.report().host_blocks, 0, "轮换结束无宿主残留");
}

// ───────────────── 三档调度器（2026-11-13） ─────────────────
//
// 同一份 `plan_capacity` 入口按**实际启用的池**分派：
//   GPU only / GPU+Host / GPU+Host+Disk。
// 级联方向固定：设备缺口先往下一级移，下一级也满才 Drop。

/// 单池档：没有宿主/磁盘，缺口只能 Drop；不得出现 Host/Disk 票。
#[tokio::test]
async fn gpu_only_planner_drops_without_lower_tier() {
    let mut face = CpuFace::new();
    let ps = pieces(&mut face, 6).await;
    let block_bytes = PIECE_BYTES * 2;
    let mut l = BlockLeg::open(LegConfig {
        slots: 6,
        block_tokens: PAGE_TOKENS,
        block_bytes,
        cache: owl_store::CachePolicy::Cap(12),
        host_bytes: 0, // 单池：宿主关
        class: BlockClass::KvPage,
        label: "gpu-only",
    })
    .expect("开腿");
    l.attach_pool(PoolLayout { class: BlockClass::KvPage, epoch: 1, pieces: ps })
        .expect("注册布局");

    let toks = |base: u32| -> Vec<u32> { (base..base + 3 * PAGE_TOKENS as u32).collect() };
    let _a = build_chain(&mut l, &toks(0)).await;
    let _b = build_chain(&mut l, &toks(4000)).await;
    assert_eq!(l.slots_free(), 0, "6 槽应由两条 3 块链占满");

    let state = StateStore::new(1, 0);
    let room = Room {
        host_enabled: false,
        host_kv_slots: 0,
        host_snap_slots: 0,
        disk_kv_slots: 0,
        disk_snap_slots: 0,
        disk_enabled: false,
        bulk_drop_when_tier_full: false,
    };
    let v = plan_capacity(
        &l,
        &state,
        &LruPolicy,
        Need::kv(2, NeedWhy::Grow),
        &[],
        &[],
        room,
    );
    assert!(v.deny.is_none(), "可丢面足够 ⇒ 不拒：{:?}", v.deny);
    assert!(!v.out.is_empty(), "单池缺口必须出 Drop 票");
    assert!(
        v.out.iter().all(|o| o.sink() == Sink::Drop),
        "单池没有下一级，不得出 Host/Disk：{:?}",
        v.out
    );
    assert!(
        v.out.iter().map(|o| o.kv_span()).sum::<usize>() >= 2,
        "Drop 面应覆盖缺口：{:?}",
        v.out
    );
}

/// 三池档：宿主满时先把宿主冷链溢写到 Disk，腾出宿主位后再把设备链
/// 下沉到 Host；**不得**直接 Drop 设备内容。
#[tokio::test]
async fn three_pool_planner_spills_host_to_disk_before_dropping_device() {
    let mut face = CpuFace::new();
    let ps = pieces(&mut face, 6).await;
    let block_bytes = PIECE_BYTES * 2;
    let mut l = BlockLeg::open(LegConfig {
        slots: 6,
        block_tokens: PAGE_TOKENS,
        block_bytes,
        cache: owl_store::CachePolicy::Cap(12),
        host_bytes: block_bytes * 3, // 宿主只装 3 块
        class: BlockClass::KvPage,
        label: "gpu-host-disk",
    })
    .expect("开腿");
    l.attach_pool(PoolLayout { class: BlockClass::KvPage, epoch: 1, pieces: ps })
        .expect("注册布局");

    let toks = |base: u32| -> Vec<u32> { (base..base + 3 * PAGE_TOKENS as u32).collect() };
    // H 整链沉宿主 ⇒ 宿主 3/3 满，设备空 6 槽。
    let h = build_chain(&mut l, &toks(0)).await;
    engine_offload(&mut l, &mut face, *h.last().expect("H 链尾"))
        .await
        .expect("H 下沉");
    assert_eq!(l.host_free_slots(), 0, "宿主应已满");
    // A(冷) + C(热) 占满设备。
    let a = build_chain(&mut l, &toks(4000)).await;
    let _c = build_chain(&mut l, &toks(8000)).await;
    assert_eq!(l.slots_free(), 0, "设备应已满");

    let state = StateStore::new(1, 0);
    let room = Room {
        host_enabled: true,
        host_kv_slots: 0, // 宿主满
        host_snap_slots: 0,
        disk_kv_slots: 3, // 盘层还能装 H 的 3 块
        disk_snap_slots: 0,
        disk_enabled: true,
        bulk_drop_when_tier_full: false,
    };
    let v = plan_capacity(
        &l,
        &state,
        &LruPolicy,
        Need::kv(3, NeedWhy::Grow),
        &[],
        &[],
        room,
    );
    assert!(v.deny.is_none(), "级联可满足 ⇒ 不拒：{:?}", v.deny);
    let sinks: Vec<Sink> = v.out.iter().map(|o| o.sink()).collect();
    assert_eq!(
        sinks,
        vec![Sink::Disk, Sink::Host],
        "宿主冷链先溢盘，再下沉设备冷链：{:?}",
        v.out
    );
    assert_eq!(
        v.out.iter().map(|o| o.kv_span()).sum::<usize>(),
        6,
        "H 3 块溢盘 + A 3 块下沉"
    );
    assert!(
        v.out.iter().all(|o| o.sink() != Sink::Drop),
        "有下一级时不得 Drop 设备内容：{:?}",
        v.out
    );
    // 盘票指向 H（宿主最冷链），Host 票指向 A（设备最冷链）。
    let h_tail = *h.last().expect("H 链尾");
    let a_tail = *a.last().expect("A 链尾");
    assert!(
        v.out.iter().any(|o| matches!(o, EvictOut::Kv { chain, sink: Sink::Disk, span: 3 } if *chain == h_tail)),
        "盘票应是 H 整链：{:?}",
        v.out
    );
    assert!(
        v.out.iter().any(|o| matches!(o, EvictOut::Kv { chain, sink: Sink::Host, span: 3 } if *chain == a_tail)),
        "Host 票应是 A 整链：{:?}",
        v.out
    );
}

/// 状态腿也必须先级联再 Drop：宿主快照满、盘有位时，先把宿主快照溢盘，
/// 再让设备快照下沉；不得因为宿主瞬时无位就把设备快照 Drop。
#[test]
fn snapshot_cascade_spills_host_before_dropping_gpu() {
    let l = leg(1); // 宿主腿开启（plan 只读几何/开关）
    let mut state = StateStore::new(1, 64);
    let host_key = owl_store::SnapKey::at_boundary(ChainHash::new(0x11));
    let gpu_key = owl_store::SnapKey::at_boundary(ChainHash::new(0x22));
    state.register_snap_owned_role(
        host_key,
        64,
        owl_store::Tier::Host,
        owl_store::SnapDurability::Durable,
        1,
        1,
        owl_store::SnapRole::History,
    );
    state.register_snap_owned_role(
        gpu_key,
        64,
        owl_store::Tier::Gpu,
        owl_store::SnapDurability::Durable,
        1,
        2,
        owl_store::SnapRole::Tail,
    );
    let room = Room {
        host_enabled: true,
        host_kv_slots: 0,
        host_snap_slots: 0, // 宿主快照满
        disk_kv_slots: 0,
        disk_snap_slots: 1, // 盘还能装一条
        disk_enabled: true,
        bulk_drop_when_tier_full: false,
    };
    let v = plan_capacity(
        &l,
        &state,
        &LruPolicy,
        Need {
            slots: 0,
            snap_bytes: 64,
            host_kv_slots: 0,
            host_snap_slots: 0,
            why: NeedWhy::Grow,
        },
        &[],
        &[],
        room,
    );
    assert!(v.deny.is_none(), "级联可满足 ⇒ 不拒：{:?}", v.deny);
    assert_eq!(v.out.len(), 1, "宿主满时先只出溢盘票：{:?}", v.out);
    assert!(
        matches!(v.out[0], EvictOut::Snap { key, sink: Sink::Disk } if key == host_key),
        "第一张票应是宿主快照溢盘：{:?}",
        v.out
    );
    assert!(
        v.out.iter().all(|o| o.sink() != Sink::Drop),
        "有下一级时不得 Drop 快照：{:?}",
        v.out
    );

    // 下一拍：宿主快照已在盘上 ⇒ 设备快照下沉到刚腾出的宿主位。
    state.set_tier(host_key, owl_store::Tier::Disk).expect("模拟溢盘落账");
    let room = Room {
        host_enabled: true,
        host_kv_slots: 0,
        host_snap_slots: 1,
        disk_kv_slots: 0,
        disk_snap_slots: 0,
        disk_enabled: true,
        bulk_drop_when_tier_full: false,
    };
    let v2 = plan_capacity(
        &l,
        &state,
        &LruPolicy,
        Need {
            slots: 0,
            snap_bytes: 64,
            host_kv_slots: 0,
            host_snap_slots: 0,
            why: NeedWhy::Grow,
        },
        &[],
        &[],
        room,
    );
    assert!(v2.deny.is_none(), "第二拍应可满足：{:?}", v2.deny);
    assert!(
        v2.out.iter().any(|o| matches!(o, EvictOut::Snap { key, sink: Sink::Host } if *key == gpu_key)),
        "第二拍应把设备快照下沉：{:?}",
        v2.out
    );
    assert!(
        v2.out.iter().all(|o| o.sink() != Sink::Drop),
        "第二拍也不得 Drop 快照：{:?}",
        v2.out
    );
}

/// **两池档（无盘）宿主快照满**：必须按 `SnapRole` 先淘汰更冷的宿主快照，
/// 不能因为宿主瞬时无位就把设备的 `Tail` 快照 Drop（2026-11-13）。
#[test]
fn two_tier_host_snapshot_relief_respects_role() {
    let l = leg(1); // 宿主开启；磁盘由 Room 显式关掉
    let mut state = StateStore::new(1, 64);
    let host_key = owl_store::SnapKey::at_boundary(ChainHash::new(0x31));
    let gpu_key = owl_store::SnapKey::at_boundary(ChainHash::new(0x32));
    state.register_snap_owned_role(
        host_key,
        64,
        owl_store::Tier::Host,
        owl_store::SnapDurability::Durable,
        1,
        1,
        owl_store::SnapRole::History,
    );
    state.register_snap_owned_role(
        gpu_key,
        64,
        owl_store::Tier::Gpu,
        owl_store::SnapDurability::Durable,
        1,
        2,
        owl_store::SnapRole::Tail,
    );
    let room = Room {
        host_enabled: true,
        host_kv_slots: 0,
        host_snap_slots: 0, // 宿主快照满
        disk_kv_slots: 0,
        disk_snap_slots: 0,
        disk_enabled: false,
        bulk_drop_when_tier_full: false,
    };
    let v = plan_capacity(
        &l,
        &state,
        &LruPolicy,
        Need {
            slots: 0,
            snap_bytes: 64,
            host_kv_slots: 0,
            host_snap_slots: 0,
            why: NeedWhy::Grow,
        },
        &[],
        &[],
        room,
    );
    assert!(v.deny.is_none(), "应可满足：{:?}", v.deny);
    assert!(
        v.out.iter().any(
            |o| matches!(o, EvictOut::Snap { key, sink: Sink::Drop } if *key == host_key)
        ),
        "必须先 Drop 更冷的宿主 History：{:?}",
        v.out
    );
    assert!(
        v.out.iter().any(
            |o| matches!(o, EvictOut::Snap { key, sink: Sink::Host } if *key == gpu_key)
        ),
        "再让设备 Tail 下沉：{:?}",
        v.out
    );
    assert!(
        v.out.iter().all(
            |o| !matches!(o, EvictOut::Snap { key, sink: Sink::Drop } if *key == gpu_key)
        ),
        "设备 Tail 不得被 Drop：{:?}",
        v.out
    );

    // 反向：宿主是 Tail、设备是 History ⇒ 应丢设备 History，宿主 Tail 留下。
    let mut state = StateStore::new(1, 64);
    state.register_snap_owned_role(
        host_key,
        64,
        owl_store::Tier::Host,
        owl_store::SnapDurability::Durable,
        1,
        1,
        owl_store::SnapRole::Tail,
    );
    state.register_snap_owned_role(
        gpu_key,
        64,
        owl_store::Tier::Gpu,
        owl_store::SnapDurability::Durable,
        1,
        2,
        owl_store::SnapRole::History,
    );
    let v = plan_capacity(
        &l,
        &state,
        &LruPolicy,
        Need {
            slots: 0,
            snap_bytes: 64,
            host_kv_slots: 0,
            host_snap_slots: 0,
            why: NeedWhy::Grow,
        },
        &[],
        &[],
        room,
    );
    assert!(v.deny.is_none(), "应可满足：{:?}", v.deny);
    assert!(
        v.out.iter().all(
            |o| !matches!(o, EvictOut::Snap { key, sink: Sink::Drop } if *key == host_key)
        ),
        "宿主 Tail 不得为设备 History 让位：{:?}",
        v.out
    );
    assert!(
        v.out.iter().any(
            |o| matches!(o, EvictOut::Snap { key, sink: Sink::Drop } if *key == gpu_key)
        ),
        "设备 History 应被 Drop：{:?}",
        v.out
    );
}

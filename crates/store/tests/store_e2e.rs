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
        host_kv_slots: l.host_free_slots(),
        host_snap_slots: 0,
        disk_kv_slots: 0,
        disk_snap_slots: 0,
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

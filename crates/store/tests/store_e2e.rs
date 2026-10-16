//! **store 无卡端到端演示(§19)**:需求驱动下沉 → 链级整条搬 → 命中批量回灌 →
//! 内容逐字节恒等 —— 全部用 **store 自己的公开 API** 驱动 + `CpuFace` 做真搬运,
//! **不依赖 engine / GPU**(秒级,可进默认套件)。
//!
//! 为什么单独有它:engine 侧的 `OWL_OFFLOAD_PROBE` 需要真模型前向(无卡档受
//! 执行域 dtype 约束,见设计卷 §18.5.1);而 store 的**账与搬运**闭环不需要
//! 模型 —— 这里把同一套三门口用 store API 复刻,作为 store 自己的回归锚。
//!
//! 场景(冷热交替;与 engine 演示台同构):
//! ```text
//! 池 6 槽 / 宿主 8MiB
//! 轮 1/2  热链 A、B(各 3 块)⇒ 池满
//! 轮 3    新需求(need = 2)⇒ 钩子判:链级**整条**沉 A(3 块,内容留宿主)
//! 轮 4    命中 A ⇒ 前 2 块在宿主 ⇒ **一条命令**批量回灌 ⇒ 逐字节恒等
//! ```

use owl_cpu::CpuFace;
use owl_iface::contract::{DeviceClient, Dtype, HeapRegion, PinnedRegion};
use owl_store::{
    apply, offload_chain, onload_blocks, plan, BlockClass, BlockLeg, LegConfig, Need, NeedWhy,
    PieceSpec, PoolLayout,
};

const PIECE_ELEMS: usize = 64;
const PIECE_BYTES: usize = PIECE_ELEMS * 4;
const PAGE_TOKENS: usize = 32;

/// 两片段(= 两层 k/v)× `slots` 槽的设备池。
async fn pieces(face: &mut CpuFace, slots: usize) -> Vec<PieceSpec> {
    let mut out = Vec::new();
    for _ in 0..2 {
        let b = face.alloc(Dtype::F32, PIECE_ELEMS * slots).await.expect("池块");
        out.push(PieceSpec { buf: b, stride_bytes: PIECE_BYTES });
    }
    out
}

fn leg(slots: usize) -> BlockLeg {
    BlockLeg::open(LegConfig {
        slots,
        block_tokens: PAGE_TOKENS,
        block_bytes: PIECE_BYTES * 2, // 两片段 = 一个逻辑块
        cache_cap: slots + 8,
        host_bytes: 8 << 20, // 宿主面开(8MiB ⇒ 可容 40+ 块)
        class: BlockClass::KvPage,
        label: "store-e2e",
    })
    .expect("开腿")
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
        face.dtoh_at(&p.buf, slot as usize * p.stride_bytes, &mut d)
            .await
            .expect("读槽");
        out.extend_from_slice(&d);
    }
    out
}

/// 建一条链(整块登记)→ 归还会话引用(缓存独占 ⇒ 可下沉)。
async fn build_chain(leg: &mut BlockLeg, tokens: &[u32]) -> Vec<owl_store::ChainHash> {
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
    let need = Need { slots: 2, why: NeedWhy::Admit };
    let verdict = plan(&l, need, &[]);
    assert!(!verdict.out.is_empty(), "池满 + 新需求 ⇒ 必须出单(下沉/驱逐)");
    let applied = apply(&mut l, &mut face, &verdict).await.expect("执行决策单");
    assert!(applied.offloaded > 0, "需求驱动下沉:应真搬(而非驱逐):{applied:?}");
    let rep = l.report();
    assert!(rep.ops.offloads > 0 && rep.ops.offload_batches > 0, "换出门①:{:?}", rep.ops);
    assert!(rep.host_bytes > 0, "宿主层应有驻留");
    assert_eq!(rep.ops.ensure_evicted, 0, "全程不得依赖 `ensure_for_len` 的同步驱逐");

    // ── 轮 4:命中 A ⇒ 命中面落在宿主 ⇒ **一条命令**批量回灌
    let hit = l.match_prefix(&a_tokens).expect("二次命中");
    let pending = hit.pending_host();
    assert!(!pending.is_empty(), "A 已被整条沉到宿主 ⇒ 命中面应有待回灌块(A 的链头段也在)");
    assert!(!hit.all_resident(), "整条链沉 ⇒ 命中面不全驻留");
    let slots = onload_blocks(&mut l, &mut face, &pending).await.expect("批量回灌");
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

/// **链级整条搬**(§14.6):一次 `offload_chain` 把链从尾到头搬完 —— 链头段必须
/// 也在宿主(否则命中面从链头就是设备格,回灌永不触发)。
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

    assert_eq!(offload_chain(&mut l, &mut face, tail).await.expect("整条沉"), 3);
    for (i, h) in chain.iter().enumerate() {
        assert_eq!(l.carrier_of(*h).expect("在账"), owl_store::Carrier::Host, "第 {i} 块应在宿主");
    }
    let hit = l.match_prefix(&tokens).expect("宿主格仍命中");
    assert_eq!(hit.pending_host().len(), 3, "三段全待回灌(链头段在内)");
    let n = onload_blocks(&mut l, &mut face, &hit.pending_host()).await.expect("批量回灌");
    assert_eq!(n.len(), 3);
    let rep = l.report();
    assert_eq!(rep.host_blocks, 0, "回灌后宿主副本应清空");
    assert_eq!(rep.cached_blocks, 3, "链仍在账");
}

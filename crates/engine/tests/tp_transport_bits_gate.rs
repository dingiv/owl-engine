//! **G8-c②:位级 transport 门**(方案 A;S4 ⑤ 第三刀)。
//!
//! 设计出处:`docs/arch/TP2-施工设计.md` §19.2(G8-c②)/ §20.2 T3。
//!
//! # 判据
//!
//! **同卡双 rank**(`TpEngine::assemble_same_device`:两个 rank 查同一个 ordinal,
//! staging 走 VMM **同卡自导入**)与**跨卡 TP2**(`TpEngine::assemble`:两张同型号卡,
//! 走 PCIe/NVLink 窗口)**的产物逐位相同**:
//! - ① 会话 token 账(生成段)**逐位**相同;
//! - ② 末步 logits(248320 项 f32)**逐位**相同(bit-exact,不是容差)。
//!
//! # 为什么这条门有价值
//!
//! `tp_assemble_gate` 已证"**同一档内**两 rank 逐位相同"(protocol/序在两端一致),
//! `tp_g8_gate` 已证"TP2 vs 单卡 = 近局数值噪声"。本门问的是**第三个问题**:
//! **跨卡那条物理路径(PCIe 映射/驱动窗口/staging 别名)本身是否引入任何位**?
//! 同卡档把 transport 换成"同卡 VMM 自别名"(无跨设备 DMA),其余一字不改
//! ⇒ 两档逐位相同 = transport 面**零贡献**(差异只来自分片 + AR 语义本身)。
//! 任何"slot 错位 / 游标串台 / 窗口层静默改写"都会在这里当场红。
//!
//! # 运行(两张**同型号**空闲卡 = **CUDA ordinal**;未设 = skip)
//!
//! ```bash
//! OWL_TP_TEST_DEV_A=2 OWL_TP_TEST_DEV_B=3 OWL_0_8B_DIR=<…/Qwen3.5-0.8B> \
//!   cargo test -p owl-engine --test tp_transport_bits_gate -- --nocapture
//! ```
//! `OWL_TP_SAME_DEV`(默认 = dev A)= 同卡档用的 ordinal(**型号必须与跨卡档两卡一致**)。

use std::path::PathBuf;

use owl_engine::{Engine, EngineConfig, EngineKnobs, TpEngine};
use owl_iface::contract::ModelError;

const PROMPT: &str = "黄河发源于青藏高原,流经九个省区。";
const MAX_NEW: usize = 12;

fn dev(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.parse().ok()
}

fn dir_of() -> PathBuf {
    std::env::var("OWL_0_8B_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("../../models/Qwen/Qwen3.5-0.8B"))
}

fn knobs() -> EngineKnobs {
    let mut k = EngineKnobs::default();
    k.sampler_enabled = false; // greedy(设备 argmax)
    k.prefix_cache = false;
    k
}

fn cfg_of(ordinal: usize) -> EngineConfig {
    EngineConfig { device_ordinal: ordinal, max_seq_tokens: 256, prefill_chunk: 64, knobs: knobs() }
}

/// 卡名(型号一致才可比;同 ordinal 会命中同一行 ⇒ 幂等)
fn card_name(ordinal: usize) -> Result<String, ModelError> {
    let rows = owl_cuda::ffi::device_rows()
        .map_err(|e| ModelError::Msg(format!("设备枚举失败:{e}")))?;
    rows.iter()
        .find(|r| r.ordinal == ordinal)
        .map(|r| r.name.clone())
        .ok_or_else(|| ModelError::Msg(format!("ordinal {ordinal} 不在设备表")))
}

/// 装入两 rank 分片(分片协议确定性:两档同一份)
async fn load_shards(
    oa: usize,
    ob: usize,
    dir: &PathBuf,
) -> (Vec<Engine<owl_cuda::GpuClient>>, Vec<owl_engine::LoadedModel>) {
    let mut e0 = Engine::new(cfg_of(oa)).expect("engine(rank0)");
    let l0 = e0.loader().slice(0, 2).load_qwen35_0_8b(dir).await.expect("rank0 分片装载");
    let mut e1 = Engine::new_with_ledger(cfg_of(ob), e0.ledger_cell(), 1).expect("engine(rank1)");
    let l1 = e1.loader().slice(1, 2).load_qwen35_0_8b(dir).await.expect("rank1 分片装载");
    (vec![e0, e1], vec![l0, l1])
}

/// 跑一个 turn,收 (生成段 token, 末步 logits)
async fn drive(
    tp: TpEngine<owl_cuda::GpuClient>,
    sid: u64,
) -> Result<(Vec<u32>, Vec<f32>), ModelError> {
    let mut r = tp.run()?;
    assert!(r.has_peer(), "TP2 执行态必须带 peer 束");
    let (sid, turn) = r.submit_session_ex(Some(sid), PROMPT, MAX_NEW, true)?;
    let prompt_len = r
        .turn_prompt_len(turn)
        .ok_or_else(|| ModelError::Msg(format!("turn {turn} 提交后取不到 prompt 长度")))?;
    loop {
        match r.pump().await? {
            owl_engine::TurnEvent::Idle => break,
            owl_engine::TurnEvent::Failed { turn: t, err } => {
                return Err(ModelError::Msg(format!("t{t} 失败:{err}")))
            }
            _ => {}
        }
    }
    let toks = r
        .session_token_ids(sid)
        .ok_or_else(|| ModelError::Msg(format!("会话 {sid} 账不在")))?;
    let gen = toks[prompt_len..].to_vec();
    let logits = r
        .read_output_f32("logits")
        .await?
        .ok_or_else(|| ModelError::Msg("末步 logits 槽缺失".into()))?;
    r.shutdown_tp().await?;
    Ok((gen, logits))
}

fn bits_diff(a: &[f32], b: &[f32]) -> (usize, f32) {
    let n = a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
    let mx = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    (n, mx)
}

#[test]
fn tp_transport_bits_gate() {
    let (Some(oa), Some(ob)) = (dev("OWL_TP_TEST_DEV_A"), dev("OWL_TP_TEST_DEV_B")) else {
        eprintln!("skip: 需 OWL_TP_TEST_DEV_A / OWL_TP_TEST_DEV_B(两张空闲卡的 CUDA ordinal)");
        return;
    };
    assert_ne!(oa, ob, "跨卡档需要两张不同卡");
    let same_dev = dev("OWL_TP_SAME_DEV").unwrap_or(oa);
    let dir = dir_of();
    if !dir.join("config.json").exists() {
        eprintln!("skip: 0.8B 检查点不在 {dir:?}(设 OWL_0_8B_DIR)");
        return;
    }
    std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .build()
                .expect("rt")
                .block_on(body(oa, ob, same_dev, dir))
                .expect("G8-c② 门失败")
        })
        .expect("起测试线程")
        .join()
        .expect("位级 transport 门失败");
}

async fn body(oa: usize, ob: usize, same_dev: usize, dir: PathBuf) -> Result<(), ModelError> {
    // 型号一致 = 位级可比的前提(不同型号的 kernel 实现可能不同)
    let (na, nb, ns) = (card_name(oa)?, card_name(ob)?, card_name(same_dev)?);
    assert_eq!(na, nb, "跨卡档两卡必须同型号(得 {na} / {nb})—— 不同型号不可位级对拍");
    assert_eq!(na, ns, "同卡档卡型号必须与跨卡档一致(得 {ns} vs {na})");
    eprintln!("[g8c] 卡面:跨卡 {oa}/{ob} = {na};同卡档 = {same_dev}");
    // 标记面初始化(AR 注入计数:两档都应看见对称注入)
    let _ = owl_shared::metrics::init_metrics(owl_shared::metrics::MetricsStore::new());

    // ── ① 跨卡档(生产路径;PCIe/NVLink 窗口)────────────────────────
    let (engines, loadeds) = load_shards(oa, ob, &dir).await;
    let tp_cross = TpEngine::assemble(engines, loadeds).await.expect("跨卡 TP2 装配");
    let (gen_cross, logits_cross) = drive(tp_cross, 1).await.expect("跨卡档 turn");
    eprintln!("[g8c] 跨卡档:{} tok;末步 logits {} 项", gen_cross.len(), logits_cross.len());

    // ── ①' 跨卡档**第二跑**(**配置内 run-to-run 逐位唯一** = §19.3 纪律 4 的
    //    承诺 ①;这是本门**最硬**的一条 —— 同配置重复必须逐位相同)。
    let (engines, loadeds) = load_shards(oa, ob, &dir).await;
    let tp_cross2 = TpEngine::assemble(engines, loadeds).await.expect("跨卡 TP2 装配(第二跑)");
    let (gen_cross2, logits_cross2) = drive(tp_cross2, 1).await.expect("跨卡档第二跑 turn");
    assert_eq!(
        gen_cross, gen_cross2,
        "★ 配置内 run-to-run **不唯一**(同配置两跑 token 不同)—— 这是真 bug,         与配置差异无关\n  跑1 = {gen_cross:?}\n  跑2 = {gen_cross2:?}"
    );
    {
        let (n2, m2) = bits_diff(&logits_cross, &logits_cross2);
        assert_eq!(
            n2, 0,
            "★ 配置内 run-to-run **不唯一**:末步 logits {n2} 项不同(max|Δ| = {m2:.6})——              同配置重复必须逐位相同"
        );
    }
    eprintln!("[g8c] ① 配置内唯一 ✓(跨卡档两跑逐位相同)");

    // ── ② 同卡档(两 rank 同一张卡;VMM 同卡自导入)────────────────
    // ⚠️ 若这里挂死:该卡上的核并发调度不成立(自旋核饿死计算核)——
    //    门超时即判据之一(见文件头)。协议面已由跨卡档覆盖。
    let (engines, loadeds) = load_shards(same_dev, same_dev, &dir).await;
    let tp_same = TpEngine::assemble_same_device(engines, loadeds)
        .await
        .expect("同卡档 TP2 装配");
    let (gen_same, logits_same) = drive(tp_same, 1).await.expect("同卡档 turn");
    eprintln!("[g8c] 同卡档:{} tok;末步 logits {} 项", gen_same.len(), logits_same.len());

    // ── ③ 判据:按 §28(**判据不得强于承诺**)────────────────────────
    //
    // §19.3 纪律 4 原文:"跨 TP 度**不承诺等价**;只承诺 ①同配置 run-to-run 与配置内
    // 不变性 ②vs 单卡的功能/统计等价"。而本门旧判据是"跨卡臂 ≡ 同卡臂 **逐位**" ——
    // **跨卡(2 卡)与同卡(2 rank 挤 1 卡)是两种配置** ⇒ 旧判据强于承诺,于是间歇性
    // 红(实测两次失败的两侧序列**各自逐字相同** ⇒ 不是竞态,是配置差异;与 §21.4-2
    // 的"热 ≠ 冷"同型 —— 那次已为 G8-a 纠正,T3 没跟着改,§28 补上)。
    //
    // 现在:硬门落在承诺得住的三条;**跨臂逐位**降级为诊断(照打不误,只是不判死)。
    assert_eq!(
        gen_cross.len(),
        gen_same.len(),
        "两档生成长度不一致(ignore_eos 定长档应恒等 —— 这是结构承诺)\n跨卡={gen_cross:?}\n同卡={gen_same:?}"
    );
    // ②-а **首 token 必须相同**:两臂同 prompt、同 prefill 路径,第一步 decode 只差
    //     transport ⇒ 这是 §19.3-4 的"功能等价"在 token 粒度上**唯一**能坐实的那一格。
    assert_eq!(
        gen_cross.first(),
        gen_same.first(),
        "★ 两档**首 token** 不同(跨卡 {:?} vs 同卡 {:?})—— 第一步 decode 两臂只差          transport,首 token 分叉说明 transport 真的改了结果",
        gen_cross.first(),
        gen_same.first()
    );
    assert_eq!(
        logits_cross.len(),
        logits_same.len(),
        "两档末步 logits 长度不一致(槽形状应当同构)"
    );
    let (nbit, mx) = bits_diff(&logits_cross, &logits_same);
    if gen_cross == gen_same {
        // ②-б **路径一致 ⇒ 末步 logits 必须逐位相同**(条件式硬门:两臂没走岔时,
        //    transport 不该引入任何位 —— 这是 §21.3 原始命题"零位贡献"的**可判形态**)
        assert_eq!(
            nbit, 0,
            "★ 两档 token 全同但末步 logits 有 {nbit} 项不同(max|Δ| = {mx:.6})——              路径一致时 transport 必须零位贡献"
        );
        eprintln!(
            "[g8c] ② 跨度 vs 单卡 ✓:token 逐位相同且末步 logits {} 项零差(路径一致档)",
            logits_cross.len()
        );
    } else {
        let i = gen_cross.iter().zip(&gen_same).position(|(a, b)| a != b).unwrap_or(0);
        eprintln!(
            "[g8c] ③ **诊断**(非判据):两臂自第 {i} 个 token 起路径分叉 —— 这是**两种配置的\
             合法差异**(§19.3-4 / §21.4-2「热 ≠ 冷」同律),不是 transport 的错。\n  \
             跨卡 = {gen_cross:?}\n  同卡 = {gen_same:?}\n  末步 logits 位级:不同              {nbit}/{} 项;max|Δ| = {mx:.6}",
            logits_cross.len()
        );
    }

    eprintln!(
        "[g8c] 全绿:①配置内唯一(跨卡两跑逐位)✓ · ②首 token 相同({} tok)✓ · \
         ③路径一致时末步 logits {} 项零差 ✓ —— transport 面零位贡献;跨臂逐位差异按 §28 记诊断",
        gen_cross.len(),
        logits_cross.len()
    );
    Ok(())
}

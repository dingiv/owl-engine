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
    let mut e1 = Engine::new(cfg_of(ob)).expect("engine(rank1)");
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

    // ── ② 同卡档(两 rank 同一张卡;VMM 同卡自导入)────────────────
    // ⚠️ 若这里挂死:该卡上的核并发调度不成立(自旋核饿死计算核)——
    //    门超时即判据之一(见文件头)。协议面已由跨卡档覆盖。
    let (engines, loadeds) = load_shards(same_dev, same_dev, &dir).await;
    let tp_same = TpEngine::assemble_same_device(engines, loadeds)
        .await
        .expect("同卡档 TP2 装配");
    let (gen_same, logits_same) = drive(tp_same, 1).await.expect("同卡档 turn");
    eprintln!("[g8c] 同卡档:{} tok;末步 logits {} 项", gen_same.len(), logits_same.len());

    // ── ③ 判据:逐位 ────────────────────────────────────────────────
    assert_eq!(
        gen_cross.len(),
        gen_same.len(),
        "两档生成长度不一致(ignore_eos 定长档应恒等)\n跨卡={gen_cross:?}\n同卡={gen_same:?}"
    );
    if gen_cross != gen_same {
        let i = gen_cross.iter().zip(&gen_same).position(|(a, b)| a != b).unwrap_or(0);
        panic!(
            "G8-c② 违约:第 {i} 个 token 起分歧(transport 引入差异?)\n  跨卡 = {gen_cross:?}\n  同卡 = {gen_same:?}"
        );
    }
    assert_eq!(
        logits_cross.len(),
        logits_same.len(),
        "两档末步 logits 长度不一致(槽形状应当同构)"
    );
    let (nbit, mx) = bits_diff(&logits_cross, &logits_same);
    eprintln!("[g8c] 末步 logits 位级:不同 {nbit}/{} 项;max|Δ| = {mx:.6}", logits_cross.len());
    assert_eq!(
        nbit, 0,
        "G8-c② 违约:末步 logits 有 {nbit} 项不同(max|Δ| = {mx:.6})—— \
         同卡档与跨卡档应当**逐位相同**(transport 不该引入任何位)"
    );

    eprintln!(
        "[g8c] 全绿:同卡双 rank ≡ 跨卡 TP2 —— {} tok 逐位相同 + 末步 logits {} 项逐位相同(transport 面零贡献)",
        gen_cross.len(),
        logits_cross.len()
    );
    Ok(())
}

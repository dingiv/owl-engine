//! **S4 ② 装载门** —— 确定性分片协议在**真检查点**上的字节一致性。
//!
//! 判据(设计 §15.3b 的硬判据):对同一检查点分别做
//! ① 全量装载(`shard = 1`)与 ② 分片装载(`(slice_id, shard) = (0, 2)`),
//! 则**分片块的每个元素 == 全量块的对应元素**(按规则表切出的那一段)——
//! 切分是纯数据搬运,零数值自由度。
//!
//! # 为什么用 CpuFace(零 GPU)
//!
//! 装载面(源读取 → 区间映射 → 目标块)与设备无关;`CpuFace` 把权重块落在
//! host f32 值域(它的 `alloc` 注释即为此类"装载门禁测试"设计)⇒ **逐元素
//! 对拍**不需要 GPU,也不需要 dtoh 的 dtype 换算。
//!
//! # 运行
//!
//! ```bash
//! OWL_0_8B_DIR=<…/Qwen3.5-0.8B> cargo test -p owl-engine --test tp_load_gate -- --nocapture
//! ```
//! 未设 `OWL_0_8B_DIR` ⇒ 尝试默认相对路径;目录不存在即 skip。

use std::path::Path;

use owl_cpu::CpuFace;
use owl_iface::contract::DeviceClient;
use owl_models::contract::ModelError;
use owl_models::formats::load::{load_from_dir, LoadOpts};
use owl_models::formats::qwen35::{qwen3_5_0_8b, Qwen35Convention};
use owl_models::formats::spec::SpecSource;
use owl_models::model::{Model, ModelSpec};
use owl_models::module::{LoadEntry, LoadManifest, QuantPlan};
use owl_models::plan::rules::{rule_of, want_axis, WeightRule};
use owl_models::plan::span::plan_span;
use owl_models::plan::tp::TpPlan;

/// 装载一次(全量或分片);返回 manifest(键 → 块/形状)。
async fn load_once<F: DeviceClient>(
    dir: &Path,
    face: &mut F,
    slice: (usize, usize),
) -> Result<LoadManifest, ModelError> {
    let (id, n) = slice;
    let (spec, rep) = ModelSpec::from_ckpt(dir, SpecSource::family(&qwen3_5_0_8b()))?;
    let spec = if n > 1 {
        spec.sharded(TpPlan::new(id, n).map_err(ModelError::Msg)?)
            .map_err(ModelError::Msg)?
    } else {
        spec
    };
    let model = Model::new(
        &spec,
        Qwen35Convention::new(rep.keys_base.trim_end_matches('.')),
        QuantPlan::F16,
        QuantPlan::F16,
    ).expect("Model::new");
    let opts = LoadOpts {
        dtype: spec.dtype,
        shard: n,
        slice_id: id,
        head_split: false,
        device_repack: Some(false), // CPU 面不设备重排
        verify: false,
        debug_tap: false,
        staging_mb: 128,
        dump: false,
        gguf_dequant: false,
    };
    load_from_dir(dir, face, &model, opts).await
}

/// 整块收割为 f32 序列(CpuFace 的块恒 f32 值域)
async fn dtoh_f32(face: &mut CpuFace, id: u64, elems: usize) -> Vec<f32> {
    let b = owl_iface::contract::Bytes::new(id, elems, owl_iface::contract::Dtype::F32, 0);
    let mut buf = vec![0u8; elems * 4];
    face.dtoh(&b, &mut buf).await.expect("dtoh");
    buf.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// manifest 里按**键尾**找条目(规避键基座前缀差异)
fn entry_by_tail<'a>(m: &'a LoadManifest, tail: &str) -> Option<&'a LoadEntry> {
    m.0.iter().find(|e| e.key.ends_with(tail))
}

/// 按规则把全量向量切成"本 slice 的那一段"(参考实现 = plan/tp 原语)
fn expected_slice(
    key: &str,
    rule: WeightRule,
    shape: &[usize],
    plan: TpPlan,
    segs: &[usize],
    full: &[f32],
) -> Vec<f32> {
    // 参考 = plan_span 的"全量 → 分片"正向读序列(与装载面同一条纯算术;
    // ⚠️ 行宽两维不同:目标 = 分片行宽,源 = **全量**行宽 —— 列切时二者不等,
    // 这正是 2026-10-17 首跑踩到的错位)
    let (span, shard_shape) = plan_span(rule, shape, plan, segs, want_axis(key)).expect("plan_span").expect("需切分");
    let tgt_w = shard_shape[1..].iter().product::<usize>().max(1);
    let src_w = shape[1..].iter().product::<usize>().max(1);
    let n_tgt: usize = shard_shape.iter().product();
    let mut out = Vec::with_capacity(n_tgt);
    for (src_off, len) in span.map(0, n_tgt, tgt_w, src_w).expect("map") {
        out.extend_from_slice(&full[src_off..src_off + len]);
    }
    out
}

#[test]
fn tp_load_gate() {
    let dir = std::env::var("OWL_0_8B_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("../../models/Qwen/Qwen3.5-0.8B"));
    if !dir.join("config.json").exists() {
        owl_shared::testkit::report::skip(format_args!("0.8B 检查点不在 {dir:?}(设 OWL_0_8B_DIR)"));
        return;
    }
    // 大盘装载:大栈 + current_thread rt(与 engine e2e 同律)
    std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .build()
                .expect("rt")
                .block_on(tp_load_body(&dir))
        })
        .expect("spawn")
        .join()
        .expect("join");
}

async fn tp_load_body(dir: &Path) {
    let mut face = CpuFace::new();
    let t0 = std::time::Instant::now();
    let full = load_once(dir, &mut face, (0, 1)).await.expect("全量装载");
    eprintln!("[tp-load] 全量装载 {} 键 / {:.1}s", full.0.len(), t0.elapsed().as_secs_f64());
    let t1 = std::time::Instant::now();
    let shard = load_once(dir, &mut face, (0, 2)).await.expect("分片装载(0/2)");
    eprintln!("[tp-load] 分片装载 {} 键 / {:.1}s", shard.0.len(), t1.elapsed().as_secs_f64());

    let plan = TpPlan::new(0, 2).unwrap();
    // 抽样键(覆盖 UnformCol / UniformRow / SegmentedCol / PerHead / Replicate)
    let keys: &[&str] = &[
        "mlp.gate_proj.weight",            // UniformCol(行切)
        "mlp.down_proj.weight",            // UniformRow(列切;stride)
        "self_attn.q_proj.weight",         // UniformCol(per-head 交错，整行等分)
        "self_attn.o_proj.weight",         // UniformRow
        "linear_attn.in_proj_qkvz.weight", // SegmentedCol(QKVZ 四段)
        "linear_attn.in_proj_a.weight",    // PerHead
        "input_layernorm.weight",          // Replicate(小)
        // ⭐ 2026-10-17 G8 定谳补列:**大 Replicate 键**(embed/lm_head 各 508MB,
        // 走"超槽键流式"分段路)—— G8 的分歧在 embed 就出现,嫌疑就在这条路上。
        "embed_tokens.weight",             // Replicate(大;流式 16 段)
        // (0.8B 是 tied 检查点:无 lm_head.weight —— 大键只 embed 一个) 
    ];
    let mut checked = 0usize;
    for tail in keys {
        let ef = entry_by_tail(&full, tail).unwrap_or_else(|| panic!("全量 manifest 缺 {tail}"));
        let es = entry_by_tail(&shard, tail).unwrap_or_else(|| panic!("分片 manifest 缺 {tail}"));
        let rule = rule_of(&ef.key);
        // ── Replicate:两 rank 都是全量(形状与内容都不得变) ──
        if rule == WeightRule::Replicate {
            assert_eq!(es.shape, ef.shape, "{tail}: Replicate 形状不得变");
            let n: usize = ef.shape.iter().product();
            let vf = dtoh_f32(&mut face, ef.block.id, n).await;
            let vs = dtoh_f32(&mut face, es.block.id, n).await;
            assert_eq!(vs, vf, "{tail}: Replicate 内容不得变");
            eprintln!("[tp-load] {tail:<40} Replicate 全量直通 ✓");
            checked += 1;
            continue;
        }
        // 段长(融合权重;与层侧 with_segments_for 同值 —— 这里按 spec 展开)
        let spec_full = qwen3_5_0_8b();
        let segs: Vec<usize> = match rule {
            WeightRule::SegmentedCol(k) => k.segments(&spec_full).expect("段长"),
            _ => Vec::new(),
        };
        // ① 形状:分片 shape == plan_span 的期望
        let (_span, want_shape) = plan_span(rule, &ef.shape, plan, &segs, want_axis(&ef.key))
            .expect("plan_span")
            .unwrap_or_else(|| panic!("{tail} 应当被切分"));
        assert_eq!(es.shape, want_shape, "{tail}: 分片形状");
        // ② 值:整块对拍(元素级)
        let nf: usize = ef.shape.iter().product();
        let ns: usize = es.shape.iter().product();
        let vf = dtoh_f32(&mut face, ef.block.id, nf).await;
        let vs = dtoh_f32(&mut face, es.block.id, ns).await;
        let want = expected_slice(&ef.key, rule, &ef.shape, plan, &segs, &vf);
        assert_eq!(vs.len(), want.len(), "{tail}: 元素数");
        if vs != want {
            let pos = vs.iter().zip(want.iter()).position(|(a, b)| a != b);
            panic!(
                "{tail}: 分片装载 ≠ 全量切片(rule={rule:?});首个不同元素 @ {pos:?}\n  got[0..4]={:?}\n  want[0..4]={:?}",
                &vs[..vs.len().min(4)],
                &want[..want.len().min(4)]
            );
        }
        eprintln!(
            "[tp-load] {tail:<40} {rule:?} {:?} → {:?} 逐元素一致 ✓",
            ef.shape, es.shape
        );
        checked += 1;
    }
    eprintln!("tp_load_gate 全绿:{checked} 个抽样键(全量 vs 分片 0/2)逐元素一致");
    // ── GPU 口径计时(生产 face;OVL_TEST_DEVICE 门控)──────────────────
    gpu_timing(dir).await;
}

/// **GPU 口径**(生产路径)的装载计时 + 分片形状检查。
///
/// 与上面 CPU 段的区别很大:GPU face 走 **DMA + 设备 cast/重排**(原始字节上卡),
/// 而 CpuFace 是**逐元素主机解码**(f16 → f32 值域)—— 后者只为"零 GPU 逐元素
/// 对拍"而存在,**不是生产口径**。本段给生产口径的数字(证明 S4 ② 的改动
/// 在 `shard = 1` 下零开销、在 `shard = 2` 下真省一半读取)。
async fn gpu_timing(dir: &Path) {
    let Some(dev) = owl_shared::env_reader::parse::<usize>("OWL_TEST_DEVICE") else {
        eprintln!("[tp-load] GPU 段 skip(OWL_TEST_DEVICE 未设)");
        return;
    };
    let mut face = match owl_cuda::GpuClient::spawn(owl_cuda::DeviceSelector::Ordinal(dev)) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("[tp-load] GPU 段 skip(spawn 失败: {e})");
            return;
        }
    };
    let t = std::time::Instant::now();
    let full = load_once(dir, &mut face, (0, 1)).await.expect("GPU 装载(全量)");
    let dt_full = t.elapsed();
    eprintln!(
        "[tp-load] GPU 生产口径:全量 shard=1 —— {} 键 / {:.2}s",
        full.0.len(),
        dt_full.as_secs_f64()
    );
    let t = std::time::Instant::now();
    let shard = load_once(dir, &mut face, (0, 2)).await.expect("GPU 装载(分片)");
    let dt_shard = t.elapsed();
    eprintln!(
        "[tp-load] GPU 生产口径:分片 shard=2 slice=0 —— {} 键 / {:.2}s(比值 {:.2}×)",
        shard.0.len(),
        dt_shard.as_secs_f64(),
        dt_full.as_secs_f64() / dt_shard.as_secs_f64()
    );
    // 分片形状检查(正确性由 CPU 段逐元素负责;此处只证形状与"确实变小")
    let ef = entry_by_tail(&full, "mlp.gate_proj.weight").expect("gate_proj");
    let es = entry_by_tail(&shard, "mlp.gate_proj.weight").expect("gate_proj 分片");
    assert_eq!(es.shape[0] * 2, ef.shape[0], "GPU 分片形状 = 全量/2");
    let _ = face.close().await;
}

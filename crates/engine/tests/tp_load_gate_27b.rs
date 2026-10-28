//! **27B AWQ 分片装载门**(S6 2026-10-17)—— 真量化检查点上的"先全量后切"。
//!
//! 设计出处:`docs/arch/TP2-施工设计.md` §15.3b / §20.4("权重面取证:同卡
//! `load_from_dir(dir, face, (0,1))` vs `((0,2))` 的同键对拍")。
//!
//! # 为什么另立一门(0.8B 门不够)
//!
//! `tp_load_gate` 只跑 **0.8B f16**(tied、safetensors 直读)⇒ 它覆盖不到:
//! ① **派生键**(`qweight/packed_raw/scales/zeros`:由 `AwqSource` 惰性物化);
//! ② **虚拟合成键**(`in_proj_qkvz` = 源侧 `in_proj_qkv` ⊕ `in_proj_z`);
//! ③ **分段切**(QGate/QKVZ/ConvQKV);
//! ④ 量化族的 `elem_len` 口径(`out*k/8` 打包字)。27B 上这四件全在。
//!
//! # 判据
//!
//! 对每个键:分片装载读出的元素序列 == `plan_span` 的"全量 → 本 slice"
//! 正向读序列(`expected_slice`,与装载面同一条纯算术)。**逐元素相等**;
//! 首个不等处即报(键 + 位置 + 两侧值)。
//!
//! # 只装两层(速度)
//!
//! 检查点仍是真 27B(键名/源格式/派生规则全真),但 spec 的 `layer_types`
//! 截成 2 层(一 GDN + 一 full attention)⇒ 装载量 ~百 MB,秒级;GPU 不参与
//! (`CpuFace` 把块落在 host f32 值域,与 `tp_load_gate` 同法)。
//!
//! # 运行
//!
//! ```bash
//! OWL_27B_DIR=<…/Qwen3.8-27B-AWQ-INT4> \
//!   cargo test -p owl-engine --test tp_load_gate_27b -- --nocapture
//! ```
//! 未设 `OWL_27B_DIR` ⇒ 试 workspace 缺省路径;不存在即 skip。
//!
//! ⚠️ **本门三测都要吞多 GB host 内存(每个 ~11GB 释放账)** ⇒ 并行跑会
//! 互相挤爆(2026-10-17 `cargo test --workspace` 实测假红,`join: Any`)。
//! **必须 `-- --test-threads=1`**(单跑三测全绿 150s)。

use std::path::{Path, PathBuf};

use owl_cuda::{DeviceClient as _, DeviceSelector, GpuClient};
use owl_iface::contract::Bytes;
use owl_models::contract::ModelError;
use owl_models::formats::load::{load_from_dir, LoadOpts};
use owl_models::formats::qwen35::{qwen3_8_27b, Qwen35Convention};
use owl_models::formats::spec::SpecSource;
use owl_models::model::{Model, ModelSpec};
use owl_models::module::{LoadEntry, LoadManifest, QuantPlan};
use owl_models::plan::rules::{is_scratch_key, rule_of, want_axis, WantAxis, WeightRule};
use owl_models::plan::span::{plan_span, segments_in_key_units};
use owl_models::plan::tp::TpPlan;

fn dir_of() -> PathBuf {
    std::env::var("OWL_27B_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/cyankiwi/Qwen3.8-27B-AWQ-INT4")))
}

/// 四层 spec(真检查点 + 真维度,只截层表;前 4 层 = 3×GDN + 1×full)
fn two_layer_spec(dir: &Path) -> (ModelSpec, String) {
    let (spec, rep) = ModelSpec::from_ckpt(dir, SpecSource::family(&qwen3_8_27b()))
        .expect("27B spec 解析");
    let base = rep.keys_base.trim_end_matches('.').to_string();
    let mut spec = spec;
    // 前 4 层(3×GDN + 1×full)⇒ 两类层都被覆盖;键名仍是真索引
    spec.layer_types = spec.layer_types[..4].to_vec();
    (spec, base)
}

/// 装载一次(全量或分片);返回 manifest(键 → 块/形状)。
async fn load_once(
    dir: &Path,
    face: &mut GpuClient,
    slice: (usize, usize),
    quant: QuantPlan,
    device_repack: bool,
) -> Result<LoadManifest, ModelError> {
    let (id, n) = slice;
    let (spec, base) = two_layer_spec(dir);
    let spec = if n > 1 {
        spec.sharded(TpPlan::new(id, n).map_err(ModelError::Msg)?)
            .map_err(ModelError::Msg)?
    } else {
        spec
    };
    let model = Model::new(
        &spec,
        Qwen35Convention::new(&base),
        quant,
        QuantPlan::F16,
    ).expect("Model::new");
    let opts = LoadOpts {
        dtype: spec.dtype,
        shard: n,
        slice_id: id,
        head_split: false,
        device_repack: Some(device_repack),
        verify: false,
        debug_tap: false,
        staging_mb: 128,
        dump: false,
        gguf_dequant: false,
    };
    load_from_dir(dir, face, &model, opts).await
}

/// 整块收割为**原始字节**(设备块;dtype 由 entry 声明)
async fn dtoh_bytes(face: &mut GpuClient, e: &LoadEntry) -> Vec<u8> {
    let n: usize = e.shape.iter().product();
    let b = Bytes::new(e.block.id, n, e.dtype, 0);
    let mut buf = vec![0u8; n * e.dtype.size_bytes()];
    face.dtoh(&b, &mut buf).await.expect("dtoh");
    buf
}

/// 全量 → 本 slice 的**正向读序列**(参考实现 = `plan_span`;与装载面同一条
/// 纯算术 —— 本门验的是"源是否按这条序列供数",不是重推切分数学)。
fn expected_slice(
    rule: WeightRule,
    shape: &[usize],
    plan: TpPlan,
    segs: &[usize],
    axis: WantAxis,
    full: &[u8],
    esz: usize,
) -> (Vec<u8>, Vec<usize>) {
    // 段长换算(与装载面同一实现;`zeros` 这类打包键的 n 侧单位不同)
    let segs_key: Vec<usize> = if matches!(rule, WeightRule::SegmentedCol(_)) {
        let dim = if axis == WantAxis::KMajor { 1 } else { 0 };
        segments_in_key_units(segs, shape[dim], "ref", dim).expect("段长换算")
    } else {
        Vec::new()
    };
    let (span, shard_shape) = plan_span(rule, shape, plan, &segs_key, axis)
        .expect("plan_span")
        .expect("需切分的键");
    let tgt_w = shard_shape[1..].iter().product::<usize>().max(1);
    let src_w = shape[1..].iter().product::<usize>().max(1);
    let n_tgt: usize = shard_shape.iter().product();
    let mut out = Vec::with_capacity(n_tgt * esz);
    for (src_off, len) in span.map(0, n_tgt, tgt_w, src_w).expect("map") {
        out.extend_from_slice(&full[src_off * esz..(src_off + len) * esz]);
    }
    (out, shard_shape)
}

/// 按键尾找条目
fn entry_by_key<'a>(m: &'a LoadManifest, key: &str) -> Option<&'a LoadEntry> {
    m.entries().iter().find(|e| e.key == key)
}

#[test]
fn tp_load_gate_27b() {
    let dir = dir_of();
    if !dir.join("config.json").exists() {
        owl_shared::testkit::report::skip(format_args!("27B 检查点不在 {dir:?}(设 OWL_27B_DIR)"));
        return;
    }
    std::thread::Builder::new()
        .stack_size(128 << 20)
        .spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .build()
                .expect("rt")
                .block_on(body(&dir))
        })
        .expect("spawn")
        .join()
        .expect("join");
}

async fn body(dir: &Path) {
    let quant = QuantPlan::W4A16Awq;
    let ordinal: usize = std::env::var("OWL_TP_TEST_DEV")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    let mut face = GpuClient::spawn_with(
        DeviceSelector::Ordinal(ordinal),
        owl_cuda::DiagOpts::default(),
        false,
        owl_shared::log::sink(),
    )
    .expect("GpuClient(设备被占?)");
    eprintln!("[tp-load-27b] 设备 ordinal {ordinal}");
    let repack = std::env::var("OWL_TP_LOAD_REPACK").map(|v| v != "0").unwrap_or(true);
    eprintln!("[tp-load-27b] device_repack = {repack}");
    let t0 = std::time::Instant::now();
    let full = load_once(dir, &mut face, (0, 1), quant, repack).await.expect("全量装载");
    eprintln!("[tp-load-27b] 全量 {} 键 / {:.1}s", full.0.len(), t0.elapsed().as_secs_f64());
    if std::env::var("OWL_TP_DUMP").is_ok() {
        for e in full.entries() {
            eprintln!("  DUMP {:<70} {:?} {:?} rule={:?}", e.key, e.shape, e.dtype, rule_of(&e.key));
        }
    }

    let mut shards: Vec<LoadManifest> = Vec::new();
    for id in 0..2usize {
        let t = std::time::Instant::now();
        let m = load_once(dir, &mut face, (id, 2), quant, repack).await.expect("分片装载");
        eprintln!("[tp-load-27b] slice{id} {} 键 / {:.1}s", m.0.len(), t.elapsed().as_secs_f64());
        shards.push(m);
    }

    let plan0 = TpPlan::new(0, 2).expect("plan0");
    let plan1 = TpPlan::new(1, 2).expect("plan1");
    let (_, base) = two_layer_spec(dir);
    let full_spec = {
        let (s, _) = ModelSpec::from_ckpt(dir, SpecSource::family(&qwen3_8_27b())).expect("spec");
        s
    };

    let mut n_rep = 0usize;
    let mut n_split = 0usize;
    let mut n_rearr = 0usize;
    let mut n_scratch = 0usize;
    let mut fails: Vec<String> = Vec::new();
    for e in full.entries() {
        let key = &e.key;
        let rule = rule_of(key);
        let full_vals = dtoh_bytes(&mut face, e).await;

        let segs: Vec<usize> = match rule {
            WeightRule::SegmentedCol(kind) => {
                kind.segments(&full_spec).expect("段长展开")
            }
            _ => Vec::new(),
        };

        for (id, (plan, m)) in [(plan0, &shards[0]), (plan1, &shards[1])].into_iter().enumerate() {
            let Some(se) = entry_by_key(m, key) else {
                fails.push(format!("{key}: slice{id} 缺键"));
                continue;
            };
            let got = dtoh_bytes(&mut face, se).await;

            // DeviceRearrange 键(`packed_raw`):上卡后由 GPU 重排 ⇒ 分片的字节
            // **不是**全量的子矩形(重排不可交换)⇒ 只做"形状一致 + 装载成功"
            // 层面的覆盖(真正的切分正确性由 `build_shard_pass` 的
            // `expect_shard == w.shape` 硬门 + 端到端输出连贯门兜)
            if matches!(e.layout, owl_models::module::Layout::DeviceRearrange { .. }) {
                n_rearr += 1;
                continue;
            }
            // scratch 件(`marlin_ws`/`marlin_ctmp`/`a8_factor`):内容是**内核
            // 暂存区/标量**,不参与区间翻译(形状已按分片 spec 声明 ⇒ 长度
            // 已正确减半);**不比内容**(比了是假红)
            if is_scratch_key(key) {
                n_scratch += 1;
                continue;
            }
            if rule == WeightRule::Replicate {
                n_rep += 1;
                if got != full_vals {
                    // 诊断:**是否 = 全量前段**(= 区间翻译没做 ⇒ 两 rank 读同一段)
                    let head = got.len() <= full_vals.len() && got == full_vals[..got.len()];
                    fails.push(format!(
                        "{key}: Replicate 键在 slice{id} 变了(len {} vs {};== 全量前段? {head})",
                        got.len(),
                        full_vals.len()
                    ));
                }
                continue;
            }
            n_split += 1;
            let (want, want_shape) = expected_slice(
                rule,
                &e.shape,
                plan,
                &segs,
                want_axis(key),
                &full_vals,
                e.dtype.size_bytes(),
            );
            if se.shape != want_shape {
                fails.push(format!(
                    "{key}: slice{id} 形状 {} ≠ 参考 {}",
                    se.shape.iter().map(|v| v.to_string()).collect::<Vec<_>>().join("x"),
                    want_shape.iter().map(|v| v.to_string()).collect::<Vec<_>>().join("x"),
                ));
                continue;
            }
            if got != want {
                let esz = e.dtype.size_bytes().max(1);
                let first = got.iter().zip(&want).position(|(a, b)| a != b);
                let fi = first.unwrap_or(0) / esz;
                fails.push(format!(
                    "{key}: slice{id} 内容不等(首差元素 idx={fi} / {} 元素;{:?})",
                    got.len() / esz,
                    got.get(first.unwrap_or(0)).zip(want.get(first.unwrap_or(0)))
                        .map(|(g, w)| format!("got=0x{g:02x} want=0x{w:02x}")),
                ));
            }
        }
    }
    eprintln!(
        "[tp-load-27b] 复制键 {n_rep} / 分片键 {n_split} / 重排键跳过 {n_rearr} / scratch 跳过 {n_scratch};失败 {}",
        fails.len()
    );
    for f in fails.iter().take(30) {
        eprintln!("  ✗ {f}");
    }
    assert!(fails.is_empty(), "27B 分片装载对拍失败 {} 条(见上)", fails.len());
    let _ = base;
}

// ============================================================================
// §2 **源级对拍门**(纯 host,无 GPU/无设备重排)—— 覆盖 `packed_raw`
// ============================================================================
//
// 为什么另立一段:上段的字节对拍**必须跳过** `Layout::DeviceRearrange` 的键
// (marlin B 重排不可交换,分片不是全量的子矩形)—— 而 `packed_raw` 正是
// **量化权重的真实数据面**。本段绕开重排:直接对 `AwqSource` 按
// `build_shard_pass` 的 span 读 **源字节**,与全量的对应子矩形逐字节比。
// 这样"权重切分"这条链(规则 → span → 源读)全在门内,零重排干扰。

/// 读某键的**全量源字节**(按 dtype 的元素宽)
fn read_all<S: owl_models::module::WeightSource + ?Sized>(
    src: &S,
    key: &str,
    dtype: owl_models::contract::Dtype,
) -> Option<Vec<u8>> {
    let n = src.elem_len(key).ok()??;
    let mut buf = vec![0u8; n * dtype.size_bytes()];
    src.convert_chunk_into_bytes(key, 0, n, &mut buf, dtype).ok()??;
    Some(buf)
}

#[test]
fn tp_source_reads_27b() {
    let dir = dir_of();
    if !dir.join("config.json").exists() {
        owl_shared::testkit::report::skip(format_args!("27B 检查点不在 {dir:?}(设 OWL_27B_DIR)"));
        return;
    }
    std::thread::Builder::new()
        .stack_size(128 << 20)
        .spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .build()
                .expect("rt")
                .block_on(source_body(&dir))
        })
        .expect("spawn")
        .join()
        .expect("join");
}

async fn source_body(dir: &Path) {
    use owl_models::formats::awq::AwqSource;
    use owl_models::module::{Loadable, LoaderCtx, WeightSource};
    use owl_models::plan::span::plan_span;

    let src = AwqSource::open_dir(dir).expect("AwqSource");
    src.set_device_repack(true);
    let (spec, base) = two_layer_spec(dir);
    let model = Model::new(
        &spec,
        Qwen35Convention::new(&base),
        QuantPlan::W4A16Awq,
        QuantPlan::F16,
    ).expect("Model::new");
    // 全量口径(声明 = 层侧全量形状)
    let ctx_full = LoaderCtx { dtype: spec.dtype, shard: 1, slice_id: 0, device_repack: true, ..Default::default() };
    let ops_full = model.layout(&ctx_full);

    let mut checked = 0usize;
    let mut skipped = 0usize;
    let mut fails: Vec<String> = Vec::new();
    for plan_i in 0..2usize {
        let plan = TpPlan::new(plan_i, 2).expect("plan");
        let ctx_s = LoaderCtx {
            dtype: spec.dtype,
            shard: 2,
            slice_id: plan_i,
            device_repack: true, // 生产档:声明 `packed_raw`(源级读同一条)
            ..Default::default()
        };
        // ⚠️ 分片臂的模型必须用**分片 spec** 构造(装载面同款:
        //    `ModelSpec::sharded` 先行 —— 层按分片 spec 建 Want 形状)
        let spec_s = spec.sharded(plan).expect("sharded spec");
        let model_s = Model::new(
            &spec_s,
            Qwen35Convention::new(&base),
            QuantPlan::W4A16Awq,
            QuantPlan::F16,
        ).expect("Model::new");
        let mut ops_s = model_s.layout(&ctx_s);
        let spans = owl_models::formats::shard::build_shard_pass(&mut ops_s, &ctx_s)
            .expect("build_shard_pass");
        let sharded = owl_models::formats::shard::ShardedSource::new(&src, spans);

        for w_full in ops_full.wants() {
            let key = &w_full.key;
            let rule = rule_of(key);
            let axis = want_axis(key);
            let Some(full_bytes) = read_all(&src, key, w_full.dtype) else {
                skipped += 1;
                continue;
            };
            let w_s = ops_s
                .wants()
                .iter()
                .find(|w| &w.key == key)
                .unwrap_or_else(|| panic!("分片 want 缺 {key}"));
            if rule == WeightRule::Replicate || is_scratch_key(key) {
                skipped += 1;
                continue;
            }
            let n_s: usize = w_s.shape.iter().product();
            let mut got = vec![0u8; n_s * w_s.dtype.size_bytes()];
            sharded
                .convert_chunk_into_bytes(key, 0, n_s, &mut got, w_s.dtype)
                .unwrap_or_else(|_| panic!("{key}: 分片源读失败"))
                .unwrap_or_else(|| panic!("{key}: 分片源缺键"));

            // 参考 = plan_span 的正向读序列(与装载面同一纯算术)
            let segs: Vec<usize> = match rule {
                WeightRule::SegmentedCol(kind) => kind.segments(&spec).expect("段长"),
                _ => Vec::new(),
            };
            let segs_key: Vec<usize> = if matches!(rule, WeightRule::SegmentedCol(_)) {
                let dim = if axis == WantAxis::KMajor { 1 } else { 0 };
                owl_models::plan::span::segments_in_key_units(&segs, w_full.shape[dim], key, dim)
                    .expect("段长换算")
            } else {
                Vec::new()
            };
            let (span, shard_shape) = plan_span(rule, &w_full.shape, plan, &segs_key, axis)
                .expect("plan_span")
                .expect("应被切分");
            assert_eq!(w_s.shape, shard_shape, "{key}: 分片形状");
            let esz = w_s.dtype.size_bytes();
            let tgt_w = shard_shape[1..].iter().product::<usize>().max(1);
            let src_w = w_full.shape[1..].iter().product::<usize>().max(1);
            let n_tgt: usize = shard_shape.iter().product();
            let mut want = Vec::with_capacity(n_tgt * esz);
            for (src_off, len) in span.map(0, n_tgt, tgt_w, src_w).expect("map") {
                want.extend_from_slice(&full_bytes[src_off * esz..(src_off + len) * esz]);
            }
            if got != want {
                let first = got.iter().zip(&want).position(|(a, b)| a != b);
                fails.push(format!(
                    "{key}: slice{plan_i} 源字节不等(首差 byte {first:?} / {} B;shape {:?})",
                    got.len(),
                    w_s.shape
                ));
            } else {
                checked += 1;
            }
        }
    }
    eprintln!("[tp-src-27b] 源级对拍 {checked} 次通过 / 跳过 {skipped} / 失败 {}", fails.len());
    for f in fails.iter().take(20) {
        eprintln!("  ✗ {f}");
    }
    assert!(fails.is_empty(), "源级对拍失败 {} 条(见上)", fails.len());
}

// ============================================================================
// §3 **重排面门**(GPU 面;引擎装载路径)—— 覆盖 staging + DeviceRearrange
// ============================================================================
//
// §1 验的是"分片字节对拍"(只对 Direct 布局有效,重排键必须跳过);
// §2 验的是"源读取";**两者之间还有一段没验过**:源 → **装载暂存槽(staging)**
// → GPU `owl_ct_repack` → 设备块。本段直拍这一段:
//   设备上的 `packed_raw` 字节 必须 == **CPU 参考**(把 §2 读出的源矩形喂
//   `pack_marlin_b_fused`)。
// 差异 = staging 槽/偏移/尺寸 或 重排发射几何 错(而非源读取错)。

#[test]
fn tp_repack_27b() {
    let dir = dir_of();
    if !dir.join("config.json").exists() {
        owl_shared::testkit::report::skip(format_args!("27B 检查点不在 {dir:?}(设 OWL_27B_DIR)"));
        return;
    }
    let ordinal: usize = std::env::var("OWL_TP_TEST_DEV")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    std::thread::Builder::new()
        .stack_size(128 << 20)
        .spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .build()
                .expect("rt")
                .block_on(repack_body(&dir, ordinal))
        })
        .expect("spawn")
        .join()
        .expect("join");
}

async fn repack_body(dir: &Path, ordinal: usize) {
    use owl_models::formats::awq::AwqSource;
    use owl_models::module::{Loadable, LoaderCtx, WeightSource};

    let mut face = GpuClient::spawn_with(
        DeviceSelector::Ordinal(ordinal),
        owl_cuda::DiagOpts::default(),
        false,
        owl_shared::log::sink(),
    )
    .expect("GpuClient");
    let src = AwqSource::open_dir(dir).expect("AwqSource");
    src.set_device_repack(true);

    let mut checked = 0usize;
    let mut fails: Vec<String> = Vec::new();
    // **对照臂先行**(2026-10-17):`(shard=1, slice=0)` = 全量装载 —— 它走
    // 同一段 staging+重排,但**没有切分**。若它也不匹配 ⇒ 是本门的参考写错
    // (假红);若它匹配而分片不匹配 ⇒ 病灶在分片路径 ✓。
    for (shard, plan_i) in [(1usize, 0usize), (2, 0), (2, 1)] {
        let plan = TpPlan::new(plan_i, shard).expect("plan");
        let (spec, base) = two_layer_spec(dir);
        let spec_s = if shard > 1 { spec.sharded(plan).expect("sharded spec") } else { spec.clone() };
        let model = Model::new(&spec_s, Qwen35Convention::new(&base), QuantPlan::W4A16Awq, QuantPlan::F16).expect("Model::new");
        let ctx = LoaderCtx {
            dtype: spec.dtype,
            shard,
            slice_id: plan_i,
            device_repack: true,
            ..Default::default()
        };
        // ① 引擎路径装载(GPU 重排;走 staging)
        let opts = LoadOpts {
            dtype: spec.dtype,
            shard,
            slice_id: plan_i,
            head_split: false,
            device_repack: Some(true),
            verify: false,
            debug_tap: false,
        staging_mb: 128,
        dump: false,
        gguf_dequant: false,
        };
        let manifest = load_from_dir(dir, &mut face, &model, opts).await.expect("装载");
        // ② 参考:同一 want 的源矩形 → CPU marlin B
        let mut ops = model.layout(&ctx);
        let spans = owl_models::formats::shard::build_shard_pass(&mut ops, &ctx).expect("pass");
        let sharded = owl_models::formats::shard::ShardedSource::new(&src, spans);
        for e in manifest.entries() {
            let key = &e.key;
            if !key.ends_with(".packed_raw") {
                continue;
            }
            let n: usize = e.shape.iter().product();
            let mut got = vec![0u8; n * e.dtype.size_bytes()];
            face.dtoh(&e.block, &mut got).await.expect("dtoh 设备块");
            // 源矩形(与装载同一条 ShardedSource)
            let mut raw = vec![0u8; n * 4]; // U32 字
            let ok = sharded.convert_chunk_into_bytes(key, 0, n, &mut raw, e.dtype).ok().flatten().is_some();
            if !ok {
                fails.push(format!("{key}: 源矩形读失败(参考面)"));
                continue;
            }
            // CPU marlin B:rows=out(shape[0]), cols=k/8(shape[1])
            let (rows, cols) = (e.shape[0], e.shape[1]);
            let k = cols * 8;
            let packed: Vec<i32> = raw.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
            let fused = owl_kernels::family::marlin::repack::marlin_fused_indices(k, rows);
            let mut b_ref: Vec<i32> = Vec::new();
            owl_kernels::family::marlin::repack::pack_marlin_b_fused(&packed, &fused, rows * k / 8, &mut b_ref);
            let ref_bytes: Vec<u8> = b_ref.iter().flat_map(|v| v.to_le_bytes()).collect();
            if ref_bytes.len() != got.len() {
                fails.push(format!("{key}: 参考字数 {} ≠ 设备块 {}B", ref_bytes.len() / 4, got.len()));
                continue;
            }
            if got != ref_bytes {
                let first = got.iter().zip(&ref_bytes).position(|(a, b)| a != b);
                fails.push(format!("{key}: shard{shard}/slice{plan_i} 设备字节 ≠ CPU 参考(首差 @ {first:?})"));
            } else {
                checked += 1;
            }
        }
    }
    eprintln!("[tp-repack-27b] 重排面门:{checked} 键通过 / 失败 {}", fails.len());
    for f in fails.iter().take(10) {
        eprintln!("  ✗ {f}");
    }
    let _ = face.sync().await;
    assert!(fails.is_empty(), "重排面门失败 {} 条(见上)", fails.len());
}

// ============================================================================
// **草稿腿**(P2.1-a,2026-10-19):草稿族双 rank 化的装载判据
// ============================================================================

fn draft_dir_of() -> PathBuf {
    std::env::var("OWL_DFLASH_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/syvai/Qwen3.8-27B-DFlash2-W4A16")))
}

/// 草稿 spec(真检查点维度;层表截成前 2 层 ⇒ 秒级)
fn draft_spec(dir: &Path) -> (ModelSpec, String) {
    let (spec, rep) = ModelSpec::from_ckpt(dir, SpecSource::family(&qwen3_8_27b()))
        .expect("草稿 spec 解析");
    let base = rep.keys_base.trim_end_matches('.').to_string();
    let mut spec = spec;
    let n = spec.layer_types.len().min(2);
    spec.layer_types = spec.layer_types[..n].to_vec();
    (spec, base)
}

/// 草稿装载一次(全量或分片)。判据面与主模型腿同律:形状/规则/读序列
async fn load_draft_once(
    dir: &Path,
    face: &mut GpuClient,
    slice: (usize, usize),
) -> Result<LoadManifest, ModelError> {
    let (id, n) = slice;
    let (spec, _base) = draft_spec(dir);
    let spec = if n > 1 {
        owl_models::formats::qwen35::shard_draft_spec(spec, TpPlan::new(id, n).map_err(ModelError::Msg)?)
            .map_err(|e| ModelError::Msg(e.to_string()))?
    } else {
        spec
    };
    let draft = owl_models::layers::dflash2::DFlash2Draft::from_spec(
        &spec,
        QuantPlan::W4A16,
        owl_models::contract::Dtype::BF16,
        false,
    );
    // **CPU 物化面关掉**(用户裁决 2026-10-19:"CPU 面先关掉,先把 GPU 打通")
    // —— 分片档只走设备重排臂;CPU 臂在 shard>1 下本就响亮拒(不可矩形切)
    let opts = LoadOpts {
        dtype: owl_models::contract::Dtype::BF16,
        shard: n,
        slice_id: id,
        head_split: false,
        device_repack: Some(true),
        verify: false,
        debug_tap: false,
        staging_mb: 128,
        dump: false,
        gguf_dequant: false,
    };
    load_from_dir(dir, face, &draft, opts).await
}

/// **草稿分片装载门**:分片读出的元素序列 == `plan_span` 的正向读序列(逐元素)
#[test]
fn tp_load_gate_27b_draft() {
    let dir = draft_dir_of();
    if !dir.join("config.json").exists() {
        owl_shared::testkit::report::skip(format_args!("DFlash2 草稿检查点不在 {dir:?}(设 OWL_DFLASH_DIR)"));
        return;
    }
    std::thread::Builder::new()
        .stack_size(128 << 20)
        .spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .build()
                .expect("rt")
                .block_on(body_draft(&dir))
        })
        .expect("spawn")
        .join()
        .expect("join");
}

async fn body_draft(dir: &Path) {
    // 判据面:`load-rearrange` 几何账走 log_debug(仓里日志宏)⇒ 用**配置面**
    // (LogConfig{level: debug})初始化 logger —— 不走 env,与装载旋钮同律。
    // `set_level` 只在 logger 已初始化时生效;此处 logger 尚未初始化 ⇒ 直接 init。
    if !owl_shared::log::logger_ready() {
        let _ = owl_shared::log::init_logger(owl_shared::log::LogConfig {
            level: 3, // LVL_DEBUG
            ..Default::default()
        });
    }
    let ordinal: usize = std::env::var("OWL_TP_TEST_DEV")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    let mut face = GpuClient::spawn_with(
        DeviceSelector::Ordinal(ordinal),
        owl_cuda::DiagOpts::default(),
        false,
        owl_shared::log::sink(),
    )
    .expect("GpuClient(设备被占?)");
    eprintln!("[tp-load-draft] 设备 ordinal {ordinal};dir={dir:?}");
    let full = load_draft_once(dir, &mut face, (0, 1)).await.expect("草稿全量装载");
    eprintln!("[tp-load-draft] 全量 {} 键", full.entries().len());
    if std::env::var("OWL_TP_DUMP").is_ok() {
        for e in full.entries() {
            eprintln!("  DUMP {:<70} {:?} {:?} rule={:?}", e.key, e.shape, e.dtype, rule_of(&e.key));
        }
    }
    let mut shards: Vec<LoadManifest> = Vec::new();
    for id in 0..2usize {
        let m = load_draft_once(dir, &mut face, (id, 2)).await.expect("草稿分片装载");
        eprintln!("[tp-load-draft] slice{id} {} 键", m.0.len());
        shards.push(m);
    }
    let plan0 = TpPlan::new(0, 2).expect("plan0");
    let plan1 = TpPlan::new(1, 2).expect("plan1");
    let full_spec = { let (s, _) = draft_spec(dir); s };

    let mut n_rep = 0usize;
    let mut n_split = 0usize;
    let mut n_scratch = 0usize;
    let mut n_rearr = 0usize;
    let mut fails: Vec<String> = Vec::new();
    for e in full.entries() {
        let key = &e.key;
        // 参考规则走**同一条推导**(带段长事实;P2.1-a):草稿的 `q_proj` 无门控 ⇒
        // 无段长声明 ⇒ `UniformCol`(与装载面 `rule_of_want` 一致)
        let segs_try: Vec<usize> = match rule_of(key) {
            WeightRule::SegmentedCol(kind) => kind.segments(&full_spec).unwrap_or_default(),
            _ => Vec::new(),
        };
        let segs_ok = !segs_try.is_empty() && segs_try.iter().sum::<usize>() == e.shape[0];
        let rule = owl_models::plan::rules::rule_of_want(key, if segs_ok { Some(&segs_try) } else { None });
        let full_vals = dtoh_bytes(&mut face, e).await;
        let segs: Vec<usize> = match rule {
            WeightRule::SegmentedCol(_) => segs_try,
            _ => Vec::new(),
        };
        for (id, (plan, m)) in [(plan0, &shards[0]), (plan1, &shards[1])].into_iter().enumerate() {
            let Some(se) = entry_by_key(m, key) else {
                fails.push(format!("{key}: slice{id} 缺键"));
                continue;
            };
            if matches!(e.layout, owl_models::module::Layout::DeviceRearrange { .. }) {
                n_rearr += 1;
                continue;
            }
            if is_scratch_key(key) {
                n_scratch += 1;
                continue;
            }
            let got = dtoh_bytes(&mut face, se).await;
            if rule == WeightRule::Replicate {
                n_rep += 1;
                if got != full_vals {
                    fails.push(format!("{key}: Replicate 键在 slice{id} 变了(len {} vs {})", got.len(), full_vals.len()));
                }
                continue;
            }
            n_split += 1;
            let (want, want_shape) = expected_slice(
                rule, &e.shape, plan, &segs, want_axis(key), &full_vals, e.dtype.size_bytes(),
            );
            if se.shape != want_shape {
                fails.push(format!("{key}: slice{id} 形状 {:?} ≠ 参考 {:?}", se.shape, want_shape));
                continue;
            }
            if got != want {
                let esz = e.dtype.size_bytes().max(1);
                let first = got.iter().zip(&want).position(|(a, b)| a != b).unwrap_or(0);
                fails.push(format!("{key}: slice{id} 内容不等(首差元素 idx={} / {})", first / esz, got.len() / esz));
            }
        }
    }
    eprintln!(
        "[tp-load-draft] 复制键 {n_rep} / 分片键 {n_split} / 重排跳过 {n_rearr} / scratch 跳过 {n_scratch};失败 {}",
        fails.len()
    );
    for f in fails.iter().take(30) {
        eprintln!("  ✗ {f}");
    }
    assert!(fails.is_empty(), "草稿分片装载对拍失败 {} 条(见上)", fails.len());
}

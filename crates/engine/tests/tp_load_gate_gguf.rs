//! **GGUF + TP 分片读的字节门**(W-1;2026-10-19;零 GPU)。
//!
//! 判据(设计 §15 的不变式):
//!
//! > 每一 rank 装载后的张量 == 全量逆变换(grouped 序)张量的**连续块**。
//!
//! 做法(全部 CPU;`GgufSource` 自担分片,不需要 face):
//! ① 全量源读字节 → `dequant_all` → **逆变换**(行置换 / 列置换)→ grouped 全量 G;
//! ② 分片源(rank r)读字节 → `dequant_all` → 本 rank 的 S;
//! ③ 逐元素 `S[t][c] == G[grow(t)][col_first + c]`(逐键按**手算**的窗口表)。
//!
//! 为什么不用"分片 vs 全量字节对拍"(其它族的门那样):GGUF 的分片是
//! **行置换 × 块对齐列切**的组合 ⇒ 分片字节在文件里**不是**一段连续区间,
//! 必须按 grouped 序取值。而归一化到 grouped 序之后,判据就退化成"连续块"
//! 这一条 —— 也顺带把"块边界错位/半行错位"这类静默错值全部抓住。
//!
//! 运行:
//! ```bash
//! cargo test -p owl-engine --test tp_load_gate_gguf -- --nocapture
//! ```
//! 资源清单缺 `gguf:27b` ⇒ skip。

use owl_iface::contract::Dtype;
use owl_models::formats::gguf::dequant::dequant_all;
use owl_models::formats::gguf::shard::ShardSel;
use owl_models::formats::gguf::transform::vhead_deperm;
use owl_models::formats::gguf::{spec_from_gguf, GgufConvention, GgufPlan, GgufSource};
use owl_models::model::Model;
use owl_models::module::{
    Loadable, LoaderCtx, PlanSource, PlanTable, SourceFacts, WeightSource,
};
use owl_models::plan::tp::TpPlan;
use owl_shared::testkit::{resource, resource_ok};

/// 读全量字节(量化件,B=U8)
fn read_full(src: &GgufSource, key: &str) -> Vec<u8> {
    let n = src.elem_len(key).expect("elem_len").expect("键在" );
    let mut v = vec![0u8; n];
    src.convert_chunk_into_bytes(key, 0, n, &mut v, Dtype::U8)
        .expect("全量读")
        .expect("全量读(有值)");
    v
}

/// 读分片字节(量化件)
fn read_shard(src: &GgufSource, key: &str) -> Vec<u8> {
    let n = src.elem_len(key).expect("elem_len(shard)").expect("键在");
    let mut v = vec![0u8; n];
    // 刻意**分块**读(装载面就是分块调的;顺带验跨行/跨段边界)
    let step = 1000usize;
    let mut off = 0;
    while off < n {
        let len = step.min(n - off);
        src.convert_chunk_into_bytes(key, off, len, &mut v[off..off + len], Dtype::U8)
            .expect("分片读")
            .expect("分片读(有值)");
        off += len;
    }
    v
}

/// **grouped 全量**(判据的参照):全量读的**行**已经是 grouped 序
/// (`GgufSource::convert_chunk_into_bytes` 逐目标行查行置换表)⇒ 只需把
/// **列**从 tiled 序扭回 grouped 序(`ssm_out`:量化件不可字节级做,正是 A2 表
/// 在单卡承担的那一步)。
fn grouped_full(src: &GgufSource, key: &str) -> (usize, usize, Vec<f32>) {
    let g = src.gguf();
    let t = g.get_tensor(key).expect("张量在");
    let (rows, inner) = (t.rows() as usize, t.inner() as usize);
    let bytes = read_full(src, key);
    let mut v = dequant_all(t.ty, &bytes, rows, inner).expect("dequant");
    let xf = src.xf(key);
    if let owl_models::formats::gguf::RowPerm::ColVHead { nk, nv, hd } = xf.row_perm {
        let map = vhead_deperm(nk, nv, hd);
        let mut out = vec![0f32; v.len()];
        for r in 0..rows {
            for (j, s) in map.iter().enumerate() {
                out[r * inner + j] = v[r * inner + *s as usize];
            }
        }
        v = out;
    }
    (rows, inner, v)
}

/// **逐键逐位门(全键面)**:blk.0(GDN)+ blk.3(full-attn)两层的**每一个**
/// 分片键 —— 分片读出的值 == grouped 全量对应窗口(行窗口 × 列起点)。
///
/// 期望窗口由 `shard::geom_of` 给出(几何本身另有手算钉值门);本门查的是
/// **读实现**:行置换、块对齐列切、逐头列聚合、1-D 元素窗口、原生行窗口。
#[test]
fn gguf_shard_reads_equal_grouped_windows_all_keys() {
    if !resource_ok("gguf:27b") {
        owl_shared::testkit::report::skip("资源清单缺 \"gguf:27b\"");
        return;
    }
    use owl_models::formats::gguf::shard::{geom_of, ShardSel, Shape, TensorFacts};
    let path = std::path::PathBuf::from(resource("gguf:27b").expect("列了就有"));
    let full = GgufSource::open(&path).expect("全量源");
    let d = *full.dims();
    // 两层足够覆盖全部角色:blk.0 = GDN(含 conv/a/dt/alpha/beta/z/qkv/out),
    // blk.3 = full(q 融合 / k / v / o),两层都有 ffn 三件
    let keys: Vec<String> = full
        .gguf()
        .tensors
        .iter()
        .filter(|t| t.name.starts_with("blk.0.") || t.name.starts_with("blk.3."))
        .map(|t| t.name.clone())
        .collect();
    assert!(keys.len() >= 25, "取样键太少({})", keys.len());
    let mut n_split = 0usize;
    let mut n_skip = 0usize;
    for key in &keys {
        let t = full.gguf().get_tensor(key).expect("张量在");
        let f = TensorFacts { ty: t.ty, rows: t.rows() as usize, inner: t.inner() as usize };
        let native = t.ty.native_dtype().is_some();
        // grouped 全量(行置换源侧已做;列 tiled → grouped 在这里补)
        let (_rows_f, inner_f, grouped) = if native {
            let v = full.take(key).expect("take").expect("原生键有值");
            (f.rows, f.inner, v)
        } else {
            grouped_full(&full, key)
        };
        for rank in 0..2usize {
            let sel = ShardSel { id: rank, world: 2, head_split: false };
            let Some(g) = geom_of(key, &d, &f, Some(sel)).expect("geom") else {
                n_skip += 1;
                continue;
            };
            let shard = GgufSource::open_shard(&path, Some(sel)).expect("分片源");
            let got: Vec<f32> = if native {
                shard.take(key).expect("take(shard)").expect("原生键有值")
            } else {
                let bytes = read_shard(&shard, key);
                dequant_all(t.ty, &bytes, g.rows(), g.row_elems()).expect("dequant(shard)")
            };
            match &g.shape {
                Shape::Elem { first, len } => {
                    assert_eq!(got.len(), *len, "[{key}] rank{rank} 1-D 元素数");
                    for j in 0..*len {
                        assert_eq!(
                            got[j].to_bits(),
                            grouped[first + j].to_bits(),
                            "[{key}] rank{rank} 元素 {j}(grouped {})",
                            first + j
                        );
                    }
                }
                Shape::Mat { wins, rows } => {
                    let rw = g.row_elems();
                    assert_eq!(got.len(), rows * rw, "[{key}] rank{rank} 交付元素数");
                    let mut ti = 0usize;
                    for (g0, n) in wins {
                        for r in *g0..*g0 + *n {
                            for c in 0..rw {
                                let (a, b) =
                                    (got[ti * rw + c], grouped[r * inner_f + g.col_first + c]);
                                assert_eq!(
                                    a.to_bits(),
                                    b.to_bits(),
                                    "[{key}] rank{rank} 行 {ti}(grouped {r}) 列 {c}:{a} ≠ {b}"
                                );
                            }
                            ti += 1;
                        }
                    }
                    assert_eq!(ti, *rows);
                }
            }
            n_split += 1;
        }
    }
    eprintln!(
        "[gguf-tp] blk.0+blk.3 全键面:{} 个 (键, rank) 组合逐位对拍通过;{} 个 (键, rank) 为 Replicate",
        n_split, n_skip
    );
}

/// **27B 分片装载账**(零 GPU):层声明的每个 want 的形状 × = 源交付的
/// `elem_len`(分片口径)—— 计划表 + 几何 + 层声明三方对齐。
///
/// 这条门替代了通用分片 pass 的"形状贯通"硬门(`owns_shard` 的源跳过它):
/// 任一角色把声明维放大错、或几何的行宽/列窗口算错,都会在这里以
/// "want n vs elem_len m"当场红。
#[test]
fn sharded_plan_shapes_match_source_elem_len() {
    if !resource_ok("gguf:27b") {
        owl_shared::testkit::report::skip("资源清单缺 \"gguf:27b\"");
        return;
    }
    let path = std::path::PathBuf::from(resource("gguf:27b").expect("列了就有"));
    let full = GgufSource::open(&path).expect("全量源");
    let (spec0, _rep) = spec_from_gguf(full.gguf()).expect("spec");
    for rank in 0..2 {
        let sel = ShardSel { id: rank, world: 2, head_split: false };
        let spec = spec0.sharded(TpPlan::new(rank, 2).expect("plan")).expect("shard spec");
        let src = GgufSource::open_shard(&path, Some(sel)).expect("分片源");
        let plan = GgufPlan::for_shard(src.gguf(), *src.dims(), sel);
        let keys = GgufConvention::new();
        let ps = PlanSource::keyed(&plan, &keys, SourceFacts::GGUF_TP);
        let model = Model::new(&spec, keys, ps, ps).expect("Model::new(分片)");
        let ctx = LoaderCtx {
            dtype: Dtype::F16,
            shard: 2,
            slice_id: rank,
            ..Default::default()
        };
        let ops = model.layout(&ctx);
        let mut bytes = 0usize;
        let mut n_quant = 0usize;
        for w in ops.wants() {
            let n: usize = w.shape.iter().product();
            let got = src.elem_len(&w.key).expect("elem_len").unwrap_or_else(|| {
                panic!("[rank{rank}] 键 {} 在源里不存在", w.key)
            });
            assert_eq!(
                got, n,
                "[rank{rank}] 键 {} 的源交付量 {got} ≠ 声明形状 {:?} = {n}",
                w.key, w.shape
            );
            if w.dtype == Dtype::U8 {
                bytes += n;
                n_quant += 1;
            }
        }
        let unresolved = plan.take_unresolved();
        assert!(unresolved.is_empty(), "[rank{rank}] 计划表未解析:{unresolved:?}");
        eprintln!(
            "[gguf-tp] rank{rank}:声明 {} 件(量化 {n_quant} 件)· 量化字节 {:.2} GiB",
            ops.wants().len(),
            bytes as f64 / (1u64 << 30) as f64
        );
        // 量化驻留 = 文件的一半 + 复制件(embed/output/norm);给个宽松上限
        let file = std::fs::metadata(&path).expect("stat").len() as usize;
        assert!(
            bytes < file / 2 + (file / 5),
            "[rank{rank}] 量化字节 {bytes} 超出 文件/2 + 20%({})",
            file / 2 + file / 5
        );
    }
}

/// **GGUF 词表切装载账**(2026-10-28):`output.weight` 开 `head_split` 后
/// 计划表/源/层声明三方对齐,且每 rank 只交付半表。
#[test]
fn gguf_head_split_plan_and_source_match() {
    if !resource_ok("gguf:27b") {
        owl_shared::testkit::report::skip("资源清单缺 \"gguf:27b\"");
        return;
    }
    let path = std::path::PathBuf::from(resource("gguf:27b").expect("列了就有"));
    let full = GgufSource::open(&path).expect("全量源");
    let full_head = full
        .elem_len("output.weight")
        .expect("elem_len(full)")
        .expect("output.weight 在");
    let (mut spec0, _rep) = spec_from_gguf(full.gguf()).expect("spec");
    spec0.head_split = true;
    for rank in 0..2 {
        let sel = ShardSel { id: rank, world: 2, head_split: true };
        let spec = spec0
            .sharded(TpPlan::new(rank, 2).expect("plan"))
            .expect("shard spec");
        assert_eq!(spec.head_vocab * 2, spec.vocab, "rank{rank} 半表");
        let src = GgufSource::open_shard(&path, Some(sel)).expect("分片源");
        let plan = GgufPlan::for_shard(src.gguf(), *src.dims(), sel);
        let keys = GgufConvention::new();
        let ps = PlanSource::keyed(&plan, &keys, SourceFacts::GGUF_TP);
        let model = Model::new(&spec, keys, ps, ps).expect("Model::new(head split)");
        let ctx = LoaderCtx {
            dtype: Dtype::F16,
            shard: 2,
            slice_id: rank,
            ..Default::default()
        };
        let ops = model.layout(&ctx);
        for w in ops.wants() {
            let n: usize = w.shape.iter().product();
            let got = src.elem_len(&w.key).expect("elem_len").unwrap_or_else(|| {
                panic!("[rank{rank}] 键 {} 在源里不存在", w.key)
            });
            assert_eq!(
                got, n,
                "[rank{rank}] 键 {} 的源交付量 {got} ≠ 声明形状 {:?} = {n}",
                w.key, w.shape
            );
        }
        let head = src
            .elem_len("output.weight")
            .expect("elem_len(head)")
            .expect("output.weight 在");
        assert_eq!(head * 2, full_head, "[rank{rank}] output.weight 应为半表");
        let unresolved = plan.take_unresolved();
        assert!(unresolved.is_empty(), "[rank{rank}] 计划表未解析:{unresolved:?}");
    }
}

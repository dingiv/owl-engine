//! W4A16 装载剖析(v2 mmap 懒物化;host-only,不碰 GPU)。
//! v2 无缓存文件:open_dir = mmap 索引(毫秒级),物化全部懒到键首触。
//! 默认仅验索引;`--touch` 以 elem_len+convert 首块触达全部消费键,
//! 模拟 eval_load 物化全链并打印分相耗时。
//!
//! 用法:cargo run -p owl-models --example w4a16_load_bench [--release] [-- --touch]

use std::path::PathBuf;
use std::time::Instant;

use owl_models::contract::Dtype;
use owl_models::formats::w4a16::W4A16Source;
use owl_models::module::WeightSource;

fn main() {
    let touch = std::env::args().any(|a| a == "--touch");
    let candidates = [
        PathBuf::from("crates/models/assets/Qwen3.5-0.8B-W4A16"),
        PathBuf::from("assets/Qwen3.5-0.8B-W4A16"),
    ];
    let dir = candidates
        .into_iter()
        .find(|p| p.exists())
        .expect("找不到 W4A16 检查点目录(从仓库根或 crates/models 下运行)");
    println!("[bench] dir={} 模式={}", dir.display(), if touch { "索引+全键物化" } else { "仅索引" });

    let t = Instant::now();
    let src = W4A16Source::open_dir(&dir).expect("open_dir");
    println!("[bench] open_dir {:?}", t.elapsed());
    if !touch {
        return;
    }

    // 全键触达:passthrough 键直接消费;量化键派生 qweight/scales/ws/ctmp/
    // weight(按 marlin_eligible 命中其一,elem_len None 的自然跳过)
    let mut touched = 0usize;
    let mut missing = 0usize;
    let mut derived: Vec<String> = Vec::new();
    for name in src.keys() {
        let base = match name.strip_suffix(".weight_packed") {
            Some(b) => b.to_string(),
            None => {
                // passthrough(F16/BF16/F32)
                if let Some(n) = src.elem_len(&name) {
                    let len = n.min(64);
                    let mut dst = vec![0u8; len * 2];
                    assert!(
                        src.convert_chunk_into_bytes(&name, 0, len, &mut dst, Dtype::F16)
                            .is_some(),
                        "passthrough 触达失败 {name}"
                    );
                    touched += 1;
                } else {
                    missing += 1;
                }
                continue;
            }
        };
        // 量化键:派生家族按键存在性消费(eligible → qweight/scales/ws/ctmp;
        // non-eligible → weight;谓词判定单一来源在源内,.elem_len None 自然跳过)
        for (suffix, dtype, esz) in [
            (".qweight", Dtype::U32, 4usize),
            (".scales", Dtype::F16, 2),
            (".marlin_ws", Dtype::U32, 4),
            (".marlin_ctmp", Dtype::U32, 4),
            (".weight", Dtype::F16, 2),
        ] {
            let key = format!("{base}{suffix}");
            if let Some(n) = src.elem_len(&key) {
                let len = n.min(64);
                let mut dst = vec![0u8; len * esz];
                assert!(
                    src.convert_chunk_into_bytes(&key, 0, len, &mut dst, dtype).is_some(),
                    "触达失败 {key}"
                );
                touched += 1;
                derived.push(key);
            }
        }
    }
    println!("[bench] 触达 passthrough {touched} / 派生 {} / 缺 elem_len {missing}", derived.len());
    println!("[bench] 总耗时 {:?}", t.elapsed());
}

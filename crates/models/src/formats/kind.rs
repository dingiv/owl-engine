//! **起点格式探测**(`CheckpointKind`):从目录读出"这是什么检查点"。
//!
//! 装载器"解释器化"的**起点轴**(设计见
//! `docs/arch/装载器-解释器化-设计与迁移.md` §2.2/§2.3):终点格式由模型文件
//! 的 `Loadable::layout` 声明,起点格式由本模块**探测**得出 —— 装载动作
//! 由 (起点原生 dtype/键面 × 目标 Want) 决定,**不由函数名决定**。
//!
//! 规矩:
//! - **纯索引嗅探,零数据拷贝**(mmap 索引;`mmap::open_raw_index` 全 dtype 可读);
//! - **命中即定,歧义/零命中 = 响亮拒**(禁静默猜格式);
//! - 规则只认"键面 + 原生 dtype"两类事实,不认目录名/文件名。
//!
//! 三族签名(2026-10-16 本机检查点实测,facts 写死):
//!
//! | 族 | 判据 | 实测样本 |
//! |---|---|---|
//! | [`CheckpointKind::SafeTensors`] | 无量化后缀;全条目 dtype ∈ {F32, BF16, F16} | 0.8B 488 张(BF16+F32)· 4B-VL 738 张 · DFlash2 草稿 81 张(BF16) |
//! | [`CheckpointKind::PackQuantized`] | `.weight_packed` ∧ `.weight_scale` 在场 | cyankiwi 27B AWQ-INT4 2396 张(399×3 后缀 + I64/I32/BF16) |
//! | [`CheckpointKind::AwqGemm`] | `.qweight` ∧ `.scales` 在场(经典 AWQ 命名) | 仓内 `formats/awq.rs` 兼容族(本机暂无样本,规则留位) |
//! | [`CheckpointKind::Gguf`] | **单文件** magic `"GGUF"`(不由扩展名定;扩展名只作提示) | 9B Q5_K_M 427 张 · 27B Q4_K_M 851 张(2026-10-19 实测) |
//!
//! **路径语义(2026-10-19 阔展)**:本入口吃**目录或单文件** —— GGUF 是单文件格式
//! (设计 §四 L4)。目录内若同时有 `.gguf` 与 `.safetensors` = **响亮拒**(不猜);
//! 目录内多个 `.gguf` = 分片 GGUF(本仓未支持)⇒ 也拒。

use crate::contract::ModelError;
use std::path::{Path, PathBuf};

/// 起点格式族(装载器解释器化的"格式词表";新增格式 = 加一臂 + 嗅探规则)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointKind {
    /// 裸 safetensors(F32/BF16/F16;f16 基线直转源)
    SafeTensors,
    /// compressed-tensors **pack-quantized**(`.weight_packed/.weight_scale/.weight_shape`;
    /// W4A16/AWQ g32-asym 同族,按原点选源:w4a16.rs / awq.rs)
    PackQuantized,
    /// 经典 AWQ GEMM 命名(`.qweight/.scales/.zeros[/.g_idx]`)
    AwqGemm,
    /// **GGUF 单文件**(压缩驻留 + 内嵌词表/模板;`formats::gguf` 起点族)
    Gguf,
}

/// 量化**方案**词表(config 实测分族;`quant_method` 不足以分族 —— 两者同为
/// compressed-tensors,见 `load::quant_family`)
pub type QuantScheme = &'static str;
/// g32 + 非对称(+zp;cyankiwi AWQ-INT4 实测 group_size=32/symmetric=false/zp int8)
pub const QUANT_G32_ASYM: QuantScheme = "g32-asym";
/// g128 + 对称(llm-compressor W4A16 产物)
pub const QUANT_G128_SYM: QuantScheme = "g128-sym";

/// 探测结论(事实面:命中哪族 + 条目规模 + 原生 dtype 分布)
#[derive(Debug, Clone)]
pub struct Detection {
    pub kind: CheckpointKind,
    /// 参与索引的分片文件(`*.safetensors`,排序)
    pub files: Vec<PathBuf>,
    /// 条目数
    pub entries: usize,
    /// 原生 dtype 分布(物化标签 + 计数;`u32` = safetensors dtype)
    pub dtypes: Vec<(String, usize)>,
}

impl Detection {
    /// 原生 dtype 是否单一(装载决策常问;多 dtype = 逐条判定)
    pub fn single_dtype(&self) -> Option<&str> {
        match self.dtypes.as_slice() {
            [(d, _)] => Some(d),
            _ => None,
        }
    }
}

/// 探测**路径**的起点格式(唯一入口;目录或单文件都收)。
///
/// 目录:既有三族(键面 + dtype 判据)+ **GGUF 臂**(目录内 `.gguf`);
/// 文件:直接按容器判(magic 权威,扩展名只作提示)。
pub fn detect(path: &Path) -> Result<Detection, ModelError> {
    if path.is_file() {
        return detect_gguf_file(path);
    }
    if !path.is_dir() {
        return Err(ModelError::Msg(format!("detect: {path:?} 既不是目录也不是文件")));
    }
    // 文件清单(与 open_raw_index 同口径:字典序;分片键天然互斥)
    let files: Vec<PathBuf> = owl_shared::file_loader::read_dir(path)
        .map_err(|e| ModelError::Msg(format!("detect: 读目录 {path:?}: {e}")))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .collect();
    // ---- GGUF 臂(先于 safetensors:两者混存 = 响亮拒,禁猜)----
    let mut ggufs: Vec<PathBuf> = files
        .iter()
        .filter(|p| p.extension().is_some_and(|x| x == "gguf"))
        .cloned()
        .collect();
    ggufs.sort();
    if !ggufs.is_empty() {
        let n_st = files.iter().filter(|p| p.extension().is_some_and(|x| x == "safetensors")).count();
        if n_st > 0 {
            return Err(ModelError::Msg(format!(
                "detect: {path:?} 同时含 .gguf 与 .safetensors(起点格式歧义;不许猜)"
            )));
        }
        if ggufs.len() > 1 {
            return Err(ModelError::Msg(format!(
                "detect: {path:?} 含 {} 个 .gguf(分片 GGUF 本仓未支持;给单文件路径)",
                ggufs.len()
            )));
        }
        return detect_gguf_file(&ggufs[0]);
    }

    // ---- safetensors 三族(既有规则不动)----
    let mut files: Vec<PathBuf> = files
        .into_iter()
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    files.sort();
    let (_, raw) = crate::formats::mmap::open_raw_index(path)?;
    let entries = raw.len();
    let mut counts: Vec<(String, usize)> = Vec::new();
    for e in raw.values() {
        let d = format!("{:?}", e.dtype);
        match counts.iter_mut().find(|(k, _)| *k == d) {
            Some((_, n)) => *n += 1,
            None => counts.push((d, 1)),
        }
    }
    counts.sort_by(|a, b| a.0.cmp(&b.0));

    let has = |suffix: &str| raw.keys().any(|k| k.ends_with(suffix));
    let (pack, awq) = (has(".weight_packed"), has(".qweight"));
    // 伴随后缀必须成对在场(只有半套 = 结构残缺,响亮拒)
    if pack != has(".weight_scale") {
        return Err(ModelError::Msg(
            "detect: pack-quantized 签名残缺(.weight_packed/.weight_scale 必须成对)".into(),
        ));
    }
    if awq != has(".scales") {
        return Err(ModelError::Msg(
            "detect: AWQ 签名残缺(.qweight/.scales 必须成对)".into(),
        ));
    }
    let kind = match (pack, awq) {
        (true, true) => {
            return Err(ModelError::Msg(
                "detect: 量化签名歧义(pack-quantized 与 AWQ 命名同时在场;不许猜)".into(),
            ))
        }
        (true, false) => CheckpointKind::PackQuantized,
        (false, true) => CheckpointKind::AwqGemm,
        (false, false) => {
            // 裸源:条目 dtype 必须全在 f16 基线白名单内(量化族的 I32/I64 会在此露出原形)
            let bad: Vec<&String> = raw
                .iter()
                .filter(|(_, e)| {
                    !matches!(
                        e.dtype,
                        safetensors::Dtype::F32 | safetensors::Dtype::BF16 | safetensors::Dtype::F16
                    )
                })
                .map(|(k, _)| k)
                .collect();
            if let Some(first) = bad.first() {
                return Err(ModelError::Msg(format!(
                    "detect: 无量化后缀但条目 dtype 非 f32/bf16/f16(首例 {first})—— 起点格式未覆盖"
                )));
            }
            CheckpointKind::SafeTensors
        }
    };
    Ok(Detection { kind, files, entries, dtypes: counts })
}

/// **GGUF 单文件探测**(magic 权威;扩展名只作提示)
fn detect_gguf_file(file: &Path) -> Result<Detection, ModelError> {
    let g = crate::formats::gguf::Gguf::open(file)?;
    let mut counts: Vec<(String, usize)> = Vec::new();
    for t in &g.tensors {
        match counts.iter_mut().find(|(k, _)| k == t.ty.name()) {
            Some((_, n)) => *n += 1,
            None => counts.push((t.ty.name().to_string(), 1)),
        }
    }
    counts.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(Detection { kind: CheckpointKind::Gguf, files: vec![file.to_path_buf()], entries: g.tensors.len(), dtypes: counts })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 资源目录(清单优先,缺则默认路径;`key` = 资源清单键)
    fn dir(key: &str, default: &str) -> Option<PathBuf> {
        let p = PathBuf::from(owl_shared::testkit::resource_or(key, default));
        p.is_dir().then_some(p)
    }

    /// 裸 safetensors 族(0.8B / 4B-VL / DFlash2 草稿三档实测)
    #[test]
    fn detects_safetensors_family() {
        for (env, default, entries) in [
            ("ckpt:0.8b", "/home/div/Documents/codes/models/Qwen/Qwen3.5-0.8B", 488),
            ("ckpt:qwen35-4b", "/home/div/Documents/codes/models/Qwen/Qwen3.5-4B", 738),
            (
                "ckpt:dflash2-bf16",
                "/home/div/Documents/codes/models/z-lab/Qwen3.8-27B-DFlash2",
                81,
            ),
        ] {
            let Some(d) = dir(env, default) else {
                eprintln!("skip: {default} 不在(资源清单 {env})");
                continue;
            };
            let det = detect(&d).expect("detect");
            assert_eq!(det.kind, CheckpointKind::SafeTensors, "{}", d.display());
            assert_eq!(det.entries, entries, "{} 条目数", d.display());
            assert!(!det.files.is_empty());
        }
    }

    /// pack-quantized 族(cyankiwi 27B AWQ-INT4:399×3 量化后缀 + I32/I64/BF16 混存)
    #[test]
    fn detects_pack_quantized_family() {
        let Some(d) = dir(
            "ckpt:27b-awq",
            "/home/div/Documents/codes/models/cyankiwi/Qwen3.8-27B-AWQ-INT4",
        ) else {
            eprintln!("skip: 资源 ckpt:27b-awq 不在(testdata/resources.toml)");
            return;
        };
        let det = detect(&d).expect("detect");
        assert_eq!(det.kind, CheckpointKind::PackQuantized);
        assert_eq!(det.entries, 2396);
        assert!(det.single_dtype().is_none(), "混存 dtype(实测 I32/I64/BF16)");
    }

    /// 响亮拒:空目录 / 不存在的路径
    #[test]
    fn rejects_empty_dir() {
        let tmp = std::env::temp_dir().join("owl_kind_empty_probe");
        std::fs::create_dir_all(&tmp).expect("mkdir");
        assert!(detect(&tmp).is_err(), "空目录必须响亮拒");
        assert!(detect(&tmp.join("nope")).is_err());
    }

    /// **GGUF 族**(2026-10-19):单文件 / 目录(单文件)/ 混存拒 / 多文件拒。
    #[test]
    fn detects_gguf_family() {
        for (key, default, entries, want_types) in [
            // M1 靶位(2026-10-19 由 9B 改 4B;原 9B 目录已删 —— 用例改随靶位,
            // 不留"默认路径不存在 ⇒ 静默 skip"的空档)
            (
                "gguf:4b",
                "/home/div/Documents/codes/models/bartowski/Qwen_Qwen3.5-4B-GGUF/Qwen_Qwen3.5-4B-Q4_K_M.gguf",
                441,
                ["F32", "Q4_K", "Q5_K", "Q6_K", "Q8_0"].as_slice(),
            ),
            (
                "gguf:27b",
                "/home/div/Documents/codes/models/ggml-org/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf",
                851,
                ["F32", "Q4_K", "Q6_K", "Q8_0"].as_slice(),
            ),
        ] {
            let p = PathBuf::from(owl_shared::testkit::resource_or(key, default));
            if !p.is_file() {
                eprintln!("skip: {default} 不在(资源清单 {key})");
                continue;
            }
            let det = detect(&p).expect("detect(gguf 文件)");
            assert_eq!(det.kind, CheckpointKind::Gguf, "{key}");
            assert_eq!(det.entries, entries, "{key} 张量数");
            assert_eq!(det.files, vec![p.clone()]);
            for want in want_types {
                assert!(
                    det.dtypes.iter().any(|(d, _)| d == want),
                    "{key} 型直方图缺 {want}(实得 {:?})",
                    det.dtypes
                );
            }
            assert!(det.single_dtype().is_none(), "{key} 混型(单型直方图不计)");

            // 目录形态(单文件在场)⇒ 同结论
            let tmp = std::env::temp_dir().join(format!("owl_kind_gguf_probe_{}", det.entries));
            std::fs::create_dir_all(&tmp).expect("mkdir");
            let link = tmp.join(p.file_name().expect("文件名"));
            let _ = std::fs::remove_file(&link);
            std::os::unix::fs::symlink(&p, &link).expect("symlink");
            let det_dir = detect(&tmp).expect("detect(目录内单 gguf)");
            assert_eq!(det_dir.kind, CheckpointKind::Gguf, "{key} 目录形态");
            assert_eq!(det_dir.entries, entries, "{key} 目录形态张量数");

            // 混存 ⇒ 响亮拒(不许猜)
            let st = tmp.join("dummy.safetensors");
            std::fs::write(&st, [0u8; 8]).expect("write");
            assert!(detect(&tmp).is_err(), "{key} gguf+safetensors 混存必须拒");
            std::fs::remove_file(&st).expect("rm");

            // 多 gguf ⇒ 拒(分片未支持)
            let second = tmp.join("another.gguf");
            let _ = std::fs::remove_file(&second);
            std::os::unix::fs::symlink(&p, &second).expect("symlink");
            assert!(detect(&tmp).is_err(), "{key} 多 gguf 必须拒(分片未支持)");
            std::fs::remove_file(&second).expect("rm");
            std::fs::remove_file(&link).expect("rm");
        }
    }
}

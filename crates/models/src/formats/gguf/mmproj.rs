//! **GGUF mmproj → Qwen3.5-VL 视觉塔**(2026-10-21)。
//!
//! GGUF 主模型只带文本主干;视觉塔在**同目录的 `mmproj-*.gguf`** 里
//! (`general.architecture = "clip"`,`clip.projector_type = "qwen3vl_merger"`)。
//! 本模块做两件事:
//!
//! 1. **事实面**:`clip.vision.*` KV + 张量形状 → [`VisionCfg`]
//!    (mmproj 没有 `num_position_embeddings` / `temporal_patch_size` 两个键,
//!    分别从 position 表行数与 patch_embd 时间片张量数取事实);
//! 2. **权重面**:把 [`VisionTower`](crate::layers::vision::VisionTower) 的
//!    HF 键面映射到 mmproj 键面,并解量化成 f32 交给既有装载器。
//!
//! # 为什么要解量化(而不是生产零解量化通道)
//!
//! 视觉塔是 **prefill-only 的小件**(4B ≈ 0.67GB / 27B ≈ 0.9GB),且 mmproj
//! 各版本会用 F16/F32/Q8_0 混排。复用 [`GgufDequantSource`](super::dequant_source)
//! 的 host 元素面最省接口;文本主干的 `GgufSource` 零解量化通道不变。
//! 同键缓存一次,装载完即随源释放。
//!
//! # patch_embd 的两个时间片
//!
//! HF 是 `Conv3d` 权重 `[out, in, t=2, p, p]`;mmproj 拆成
//! `v.patch_embd.weight` / `v.patch_embd.weight.1` 两个 `[p,p,in,out]` 张量
//! (ggml 序;每个 = 一个时间片,行内 `[in][p][p]`)。拼接必须还原成 HF 的
//! `[out][in][t][p][p]`——**按 in 通道交错**,不是简单首尾相接(已用 4B
//! HF 权重逐元素对拍,见测试)。

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use std::path::{Path, PathBuf};

use crate::contract::ModelError;
use crate::layers::vision::VisionCfg;
use crate::module::{Taken, WeightSource};

use super::dequant::dequant_all;
use super::reader::{Gguf, TensorInfo};

/// **mmproj 路径解析**(loader 与 `--check` 共用):显式优先;缺省只在
/// 同目录**唯一** `mmproj*.gguf` 时自动选。多个/没有 = 响亮拒并列出候选。
pub fn resolve_path(text_path: &Path, explicit: Option<&Path>) -> Result<PathBuf, ModelError> {
    if let Some(p) = explicit {
        if !p.is_file() {
            return Err(ModelError::Msg(format!(
                "[model].mmproj 不是文件: {}",
                p.display()
            )));
        }
        return Ok(p.to_path_buf());
    }
    let dir = text_path.parent().ok_or_else(|| {
        ModelError::Msg(format!(
            "GGUF 视觉塔:无法从 {} 推出同目录",
            text_path.display()
        ))
    })?;
    let mut candidates = Vec::new();
    let rd = owl_shared::file_loader::read_dir(dir).map_err(|e| {
        ModelError::Msg(format!("GGUF 视觉塔:列目录 {} 失败: {e}", dir.display()))
    })?;
    for entry in rd {
        let entry = entry.map_err(|e| {
            ModelError::Msg(format!("GGUF 视觉塔:读目录项 {} 失败: {e}", dir.display()))
        })?;
        let p = entry.path();
        let Some(name) = p.file_name().and_then(|s| s.to_str()) else { continue };
        if name.starts_with("mmproj") && name.ends_with(".gguf") && p.is_file() {
            candidates.push(p);
        }
    }
    candidates.sort();
    match candidates.as_slice() {
        [one] => Ok(one.clone()),
        [] => Err(ModelError::Msg(format!(
            "GGUF 视觉塔:[model].vision = true,但 {} 下没有 `mmproj*.gguf`\
             —— 请把 [model].mmproj 指向塔文件",
            dir.display()
        ))),
        many => Err(ModelError::Msg(format!(
            "GGUF 视觉塔:{} 下有 {} 个 mmproj 候选（{}）—— 请用 [model].mmproj 显式指定",
            dir.display(),
            many.len(),
            many.iter()
                .map(|p| p.file_name().and_then(|s| s.to_str()).unwrap_or("?"))
                .collect::<Vec<_>>()
                .join(" | ")
        ))),
    }
}

/// mmproj 的视觉超参(唯一入口;缺键/形态不符 = 响亮拒)
pub fn vision_cfg(g: &Gguf) -> Result<VisionCfg, ModelError> {
    let ctx = |m: String| ModelError::Msg(format!("gguf mmproj({}): {m}", g.path.display()));
    let arch = g.arch().map_err(|e| ModelError::Msg(format!("gguf mmproj: {e}")))?;
    if arch != "clip" {
        return Err(ctx(format!(
            "general.architecture = `{arch}`,期望 `clip`(这不是 mmproj?)"
        )));
    }
    let kv = &g.kv;
    if !kv.opt_bool("clip.has_vision_encoder").map_err(&ctx)?.unwrap_or(false) {
        return Err(ctx("clip.has_vision_encoder 不为 true(没有视觉编码器)".into()));
    }
    let proj = kv.str("clip.projector_type").map_err(&ctx)?;
    if proj != "qwen3vl_merger" {
        return Err(ctx(format!(
            "clip.projector_type = `{proj}`,本仓只接 `qwen3vl_merger`"
        )));
    }
    if !kv.opt_bool("clip.use_gelu").map_err(&ctx)?.unwrap_or(false) {
        return Err(ctx("clip.use_gelu 不为 true(塔的 GELU 语义不成立)".into()));
    }
    if kv
        .opt_bools("clip.vision.is_deepstack_layers")
        .map_err(&ctx)?
        .is_some_and(|v| v.iter().any(|x| *x))
    {
        return Err(ctx(
            "clip.vision.is_deepstack_layers 含 true:本仓只接无 deepstack 的 Qwen3.5-VL 塔".into(),
        ));
    }

    let depth = kv.usize("clip.vision.block_count").map_err(&ctx)?;
    let hidden = kv.usize("clip.vision.embedding_length").map_err(&ctx)?;
    let intermediate = kv.usize("clip.vision.feed_forward_length").map_err(&ctx)?;
    let heads = kv.usize("clip.vision.attention.head_count").map_err(&ctx)?;
    let out_hidden = kv.usize("clip.vision.projection_dim").map_err(&ctx)?;
    let patch = kv.usize("clip.vision.patch_size").map_err(&ctx)?;
    let merge = kv.usize("clip.vision.spatial_merge_size").map_err(&ctx)?;
    let eps = kv
        .opt_f32("clip.vision.attention.layer_norm_epsilon")
        .map_err(&ctx)?
        .unwrap_or(1e-6);
    // 预处理面(srv `vision_processor`)固定 mean/std=0.5;mmproj 若声明别的值,
    // 塔的输入分布就不成立 —— 装载期拒,别让服务面静默错值。
    for (key, want) in [("clip.vision.image_mean", 0.5f32), ("clip.vision.image_std", 0.5f32)] {
        if let Some(v) = kv.opt_f32s(key).map_err(&ctx)? {
            if v.iter().any(|x| (*x - want).abs() > 1e-6) {
                return Err(ctx(format!(
                    "{key}={v:?} ≠ [{want};3](本仓 vision_processor 的归一化固定 0.5/0.5)"
                )));
            }
        }
    }
    if depth == 0 || hidden == 0 || heads == 0 || out_hidden == 0 || patch == 0 || merge == 0 {
        return Err(ctx(format!(
            "超参含 0(depth={depth} hidden={hidden} heads={heads} out={out_hidden} \
             patch={patch} merge={merge})"
        )));
    }
    if hidden % heads != 0 {
        return Err(ctx(format!(
            "hidden={hidden} 不被 heads={heads} 整除(head_dim 不可定)"
        )));
    }
    if !(eps.is_finite() && eps > 0.0) {
        return Err(ctx(format!("layer_norm_epsilon={eps} 非法")));
    }

    let tensor = |name: &str| -> Result<&TensorInfo, ModelError> {
        g.get_tensor(name)
            .ok_or_else(|| ctx(format!("缺张量 `{name}`(mmproj 不完整)")))
    };
    // position 表行数 = num_position_embeddings;mmproj 没有这个 KV。
    let pos = tensor("v.position_embd.weight")?;
    if pos.ne.len() != 2 || pos.ne[0] as usize != hidden {
        return Err(ctx(format!(
            "`v.position_embd.weight` ne={:?},期望 [hidden={hidden}, num_pos]",
            pos.ne
        )));
    }
    let num_pos = pos.ne[1] as usize;

    // patch_embd:两个时间片各一个 4-D 张量 [p,p,in,out]。
    let p0 = tensor("v.patch_embd.weight")?;
    if p0.ne.len() != 4 {
        return Err(ctx(format!(
            "`v.patch_embd.weight` 维数 {} ≠ 4(期望 [p,p,in,out])",
            p0.ne.len()
        )));
    }
    let (pw, ph, in_channels, out_ch) = (
        p0.ne[0] as usize,
        p0.ne[1] as usize,
        p0.ne[2] as usize,
        p0.ne[3] as usize,
    );
    if pw != patch || ph != patch {
        return Err(ctx(format!(
            "patch_embd 空间核 [{pw},{ph}] ≠ clip.patch_size={patch}"
        )));
    }
    if out_ch != hidden {
        return Err(ctx(format!(
            "patch_embd 输出维 {out_ch} ≠ embedding_length={hidden}"
        )));
    }
    let temporal = if g.index_of("v.patch_embd.weight.1").is_some() {
        let p1 = tensor("v.patch_embd.weight.1")?;
        if p1.ne != p0.ne {
            return Err(ctx(format!(
                "时间片张量形态不一致:{:?} vs {:?}",
                p0.ne, p1.ne
            )));
        }
        2
    } else {
        1
    };

    let cfg = VisionCfg {
        depth,
        hidden,
        intermediate,
        heads,
        out_hidden,
        num_pos,
        patch,
        temporal,
        merge,
        in_channels,
        eps,
    };
    // `num_position_embeddings` 必须是非零完全平方(否则插值表边长不可定)
    cfg.grid_side().map_err(|e| ctx(e.to_string()))?;
    Ok(cfg)
}

/// mmproj 权重源(HF 塔键面 → mmproj 张量;host 解量化 + 同键缓存)
pub struct MmprojSource {
    g: Gguf,
    cfg: VisionCfg,
    cache: Mutex<HashMap<String, Arc<Vec<f32>>>>,
}

impl MmprojSource {
    /// 打开 mmproj 单文件
    pub fn open(path: &std::path::Path) -> Result<Self, ModelError> {
        Self::from_gguf(Gguf::open(path)?)
    }

    /// 从已解析容器构造(避免二次 mmap;loader 与 `vision_cfg` 共用同一份)
    pub fn from_gguf(g: Gguf) -> Result<Self, ModelError> {
        let cfg = vision_cfg(&g)?;
        Ok(Self { g, cfg, cache: Mutex::new(HashMap::new()) })
    }

    pub fn vision_cfg(&self) -> VisionCfg {
        self.cfg
    }

    pub fn gguf(&self) -> &Gguf {
        &self.g
    }

    fn is_patch_weight(key: &str) -> bool {
        key == "model.visual.patch_embed.proj.weight"
    }

    /// 塔键 → mmproj 物理键(未知键 = 响亮拒:塔声明与文件键面不匹配)
    fn physical(key: &str) -> Result<String, ModelError> {
        let rest = key.strip_prefix("model.visual.").ok_or_else(|| {
            ModelError::Msg(format!(
                "mmproj 源:键 `{key}` 不带 `model.visual.` 前缀(塔装载声明错?)"
            ))
        })?;
        let fixed = match rest {
            "patch_embed.proj.bias" => "v.patch_embd.bias",
            "pos_embed.weight" => "v.position_embd.weight",
            "merger.norm.weight" => "v.post_ln.weight",
            "merger.norm.bias" => "v.post_ln.bias",
            "merger.linear_fc1.weight" => "mm.0.weight",
            "merger.linear_fc1.bias" => "mm.0.bias",
            "merger.linear_fc2.weight" => "mm.2.weight",
            "merger.linear_fc2.bias" => "mm.2.bias",
            _ => "",
        };
        if !fixed.is_empty() {
            return Ok(fixed.to_string());
        }
        let blk = rest.strip_prefix("blocks.").ok_or_else(|| {
            ModelError::Msg(format!("mmproj 源:未映射的塔键 `{key}`"))
        })?;
        let (idx, sub) = blk.split_once('.').ok_or_else(|| {
            ModelError::Msg(format!("mmproj 源:块键缺局部名 `{key}`"))
        })?;
        let idx: usize = idx.parse().map_err(|_| {
            ModelError::Msg(format!("mmproj 源:块号不是整数 `{key}`"))
        })?;
        let local = match sub {
            "norm1.weight" => "ln1.weight",
            "norm1.bias" => "ln1.bias",
            "norm2.weight" => "ln2.weight",
            "norm2.bias" => "ln2.bias",
            "attn.qkv.weight" => "attn_qkv.weight",
            "attn.qkv.bias" => "attn_qkv.bias",
            "attn.proj.weight" => "attn_out.weight",
            "attn.proj.bias" => "attn_out.bias",
            "mlp.linear_fc1.weight" => "ffn_up.weight",
            "mlp.linear_fc1.bias" => "ffn_up.bias",
            "mlp.linear_fc2.weight" => "ffn_down.weight",
            "mlp.linear_fc2.bias" => "ffn_down.bias",
            other => {
                return Err(ModelError::Msg(format!(
                    "mmproj 源:块内键 `{other}` 未映射(塔键 `{key}`)"
                )))
            }
        };
        Ok(format!("v.blk.{idx}.{local}"))
    }

    fn dequant(&self, phys: &str, rows: usize, inner: usize) -> Taken<Vec<f32>> {
        let Some(t) = self.g.get_tensor(phys) else { return Ok(None) };
        let ix = self.g.index_of(phys).ok_or_else(|| {
            ModelError::Msg(format!("mmproj 源:张量 `{phys}` 无索引"))
        })?;
        dequant_all(t.ty, self.g.bytes(ix), rows, inner)
            .map(Some)
            .map_err(|e| ModelError::Msg(format!("mmproj 解量化 `{phys}`: {e}")))
    }

    /// 两个时间片 → HF `[out][in][t][p][p]` 行主序
    fn load_patch(&self) -> Taken<Vec<f32>> {
        let Some(p0) = self.g.get_tensor("v.patch_embd.weight") else {
            return Ok(None);
        };
        if p0.ne.len() != 4 {
            return Err(ModelError::Msg(format!(
                "mmproj patch_embd:ne={:?} 不是 4 维",
                p0.ne
            )));
        }
        let (out, channels, spatial) = (
            p0.ne[3] as usize,
            p0.ne[2] as usize,
            p0.ne[0] as usize * p0.ne[1] as usize,
        );
        let inner = channels * spatial;
        let a = self
            .dequant("v.patch_embd.weight", out, inner)?
            .ok_or_else(|| ModelError::Msg("mmproj patch_embd:缺 `v.patch_embd.weight`".into()))?;
        if self.cfg.temporal == 1 {
            return Ok(Some(a));
        }
        let b = self
            .dequant("v.patch_embd.weight.1", out, inner)?
            .ok_or_else(|| {
                ModelError::Msg("mmproj patch_embd:cfg.temporal=2 但缺 `.weight.1`".into())
            })?;
        let mut dst = Vec::with_capacity(out * inner * 2);
        for o in 0..out {
            for c in 0..channels {
                let base = (o * channels + c) * spatial;
                dst.extend_from_slice(&a[base..base + spatial]);
                dst.extend_from_slice(&b[base..base + spatial]);
            }
        }
        Ok(Some(dst))
    }

    fn load(&self, key: &str) -> Taken<Arc<Vec<f32>>> {
        {
            let c = self.cache.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(v) = c.get(key) {
                return Ok(Some(v.clone()));
            }
        }
        let v = if Self::is_patch_weight(key) {
            self.load_patch()?
        } else {
            let phys = Self::physical(key)?;
            let t = self.g.get_tensor(&phys).ok_or_else(|| {
                ModelError::Msg(format!(
                    "mmproj 源:塔键 `{key}` → 物理键 `{phys}`,但文件没有该张量"
                ))
            })?;
            self.dequant(&phys, t.rows() as usize, t.inner() as usize)?
        };
        let Some(v) = v else { return Ok(None) };
        let arc = Arc::new(v);
        let mut c = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        c.insert(key.to_string(), arc.clone());
        Ok(Some(arc))
    }
}

impl WeightSource for MmprojSource {
    fn take(&self, key: &str) -> Taken<Vec<f32>> {
        Ok(self.load(key)?.map(|v| (*v).clone()))
    }

    fn elem_len(&self, key: &str) -> Taken<usize> {
        if Self::is_patch_weight(key) {
            let Some(t) = self.g.get_tensor("v.patch_embd.weight") else { return Ok(None) };
            if t.ne.len() != 4 {
                return Err(ModelError::Msg(format!(
                    "mmproj patch_embd:ne={:?} 不是 4 维",
                    t.ne
                )));
            }
            return Ok(Some(self.cfg.hidden * self.cfg.patch_flat()));
        }
        let phys = Self::physical(key)?;
        Ok(self
            .g
            .get_tensor(&phys)
            .map(|t| (t.rows() * t.inner()) as usize))
    }

    fn native_dtype(&self, _key: &str) -> Taken<crate::contract::Dtype> {
        // 元素面已解成 f32;目标 dtype 由装载面 cast(F16/F32)
        Ok(None)
    }

    fn convert_chunk_into_bytes(
        &self,
        key: &str,
        offset: usize,
        len: usize,
        dst: &mut [u8],
        dtype: crate::contract::Dtype,
    ) -> Taken<()> {
        use crate::contract::Dtype;
        let Some(v) = self.load(key)? else { return Ok(None) };
        let Some(slice) = v.get(offset..offset + len) else { return Ok(None) };
        let esz = match dtype {
            Dtype::F32 => 4,
            Dtype::F16 | Dtype::BF16 => 2,
            other => {
                return Err(ModelError::Msg(format!(
                    "mmproj 源:键 `{key}` 目标 dtype {other:?} 未接"
                )))
            }
        };
        if dst.len() < len * esz {
            return Err(ModelError::Msg(format!(
                "mmproj 源:键 `{key}` dst {}B < 需 {}B",
                dst.len(),
                len * esz
            )));
        }
        for (i, x) in slice.iter().enumerate() {
            match dtype {
                Dtype::F32 => dst[i * 4..i * 4 + 4].copy_from_slice(&x.to_le_bytes()),
                Dtype::F16 => dst[i * 2..i * 2 + 2]
                    .copy_from_slice(&half::f16::from_f32(*x).to_le_bytes()),
                Dtype::BF16 => dst[i * 2..i * 2 + 2]
                    .copy_from_slice(&half::bf16::from_f32(*x).to_le_bytes()),
                _ => unreachable!("上方已筛"),
            }
        }
        Ok(Some(()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::Dtype;
    use crate::layers::vision::VisionTower;
    use crate::module::{Loadable, LoaderCtx};

    fn mmproj_path(key: &str, sibling_of: &str, fallback: &str) -> Option<std::path::PathBuf> {
        if let Some(p) = owl_shared::testkit::resource(key) {
            let p = std::path::PathBuf::from(p);
            if p.is_file() {
                return Some(p);
            }
        }
        if let Some(g) = owl_shared::testkit::resource(sibling_of) {
            let g = std::path::PathBuf::from(g);
            if let Some(dir) = g.parent() {
                let mut hit = None;
                if let Ok(rd) = owl_shared::file_loader::read_dir(dir) {
                    for e in rd.flatten() {
                        let p = e.path();
                        let Some(n) = p.file_name().and_then(|s| s.to_str()) else { continue };
                        if n.starts_with("mmproj") && n.ends_with(".gguf") {
                            hit = Some(p);
                            break;
                        }
                    }
                }
                if let Some(p) = hit {
                    return Some(p);
                }
            }
        }
        let p = std::path::PathBuf::from(fallback);
        p.is_file().then_some(p)
    }

    #[test]
    fn facts_4b_mmproj() {
        let Some(path) = mmproj_path(
            "mmproj:4b",
            "gguf:4b",
            "/home/div/Documents/codes/models/bartowski/Qwen_Qwen3.5-4B-GGUF/mmproj-Qwen_Qwen3.5-4B-f16.gguf",
        ) else {
            eprintln!("skip: 4B mmproj 不在");
            return;
        };
        let g = Gguf::open(&path).expect("open mmproj");
        let cfg = vision_cfg(&g).expect("facts");
        assert_eq!(cfg.depth, 24, "4B 塔深");
        assert_eq!(cfg.hidden, 1024, "4B 塔 hidden");
        assert_eq!(cfg.intermediate, 4096, "4B 塔 ffn");
        assert_eq!(cfg.heads, 16, "4B 塔 heads");
        assert_eq!(cfg.out_hidden, 2560, "4B 塔 out");
        assert_eq!(cfg.num_pos, 2304, "position 表行数");
        assert_eq!((cfg.patch, cfg.temporal, cfg.merge, cfg.in_channels), (16, 2, 2, 3));
        assert!((cfg.eps - 1e-6).abs() < 1e-9, "eps");
    }

    #[test]
    fn facts_27b_mmproj() {
        let Some(path) = mmproj_path(
            "mmproj:27b",
            "gguf:27b",
            "/home/div/Documents/codes/models/ggml-org/Qwen3.8-27B-GGUF/mmproj-Qwen3.8-27B-Q8_0.gguf",
        ) else {
            eprintln!("skip: 27B mmproj 不在");
            return;
        };
        let g = Gguf::open(&path).expect("open mmproj");
        let cfg = vision_cfg(&g).expect("facts");
        assert_eq!(cfg.depth, 27, "27B 塔深");
        assert_eq!(cfg.hidden, 1152, "27B 塔 hidden");
        assert_eq!(cfg.intermediate, 4304, "27B 塔 ffn");
        assert_eq!(cfg.out_hidden, 5120, "27B 塔 out");
        assert_eq!(cfg.num_pos, 2304, "position 表行数");
        assert_eq!((cfg.patch, cfg.temporal, cfg.merge, cfg.in_channels), (16, 2, 2, 3));
    }

    #[test]
    fn source_dequant_27b_q8_block() {
        let Some(path) = mmproj_path(
            "mmproj:27b",
            "gguf:27b",
            "/home/div/Documents/codes/models/ggml-org/Qwen3.8-27B-GGUF/mmproj-Qwen3.8-27B-Q8_0.gguf",
        ) else {
            eprintln!("skip: 27B mmproj 不在");
            return;
        };
        let src = MmprojSource::open(&path).expect("source");
        // 27B mmproj 的 qkv 是 Q8_0(4B f16 档覆盖不到 Q8_0 解量化臂)
        let v = src
            .take("model.visual.blocks.0.attn.qkv.weight")
            .expect("take qkv")
            .expect("qkv 在场");
        assert_eq!(v.len(), 3 * 1152 * 1152, "qkv 元素数");
        assert!(v.iter().all(|x| x.is_finite()), "Q8_0 解量化不得出 NaN/Inf");
        assert!(v.iter().any(|x| *x != 0.0), "Q8_0 解量化不得全零");
    }

    #[test]
    fn source_covers_tower_keys() {
        let Some(path) = mmproj_path(
            "mmproj:4b",
            "gguf:4b",
            "/home/div/Documents/codes/models/bartowski/Qwen_Qwen3.5-4B-GGUF/mmproj-Qwen_Qwen3.5-4B-f16.gguf",
        ) else {
            eprintln!("skip: 4B mmproj 不在");
            return;
        };
        let src = MmprojSource::open(&path).expect("source");
        let tower = VisionTower::new(src.vision_cfg());
        let ctx = LoaderCtx { dtype: Dtype::F16, ..Default::default() };
        let ops = tower.layout(&ctx);
        for w in ops.wants() {
            let n: usize = w.shape.iter().product();
            let got = src.elem_len(&w.key).unwrap_or_else(|e| panic!("{}: {e}", w.key));
            assert_eq!(got, Some(n), "键 `{}` 元素数/形状不符", w.key);
        }
        // patch 是唯一双时间片合成键:取一次验长度,防止 elem_len 只报单片的回归
        let n = src
            .take("model.visual.patch_embed.proj.weight")
            .expect("patch take")
            .expect("patch 在场")
            .len();
        let cfg = src.vision_cfg();
        assert_eq!(n, cfg.hidden * cfg.patch_flat(), "patch 合成后的元素数");
    }
}

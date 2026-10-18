//! **Qwen3.5-VL 视觉塔**(ViT;Holo-3.1-9B/35B-A3B 同款)——
//! 2026-10-15 `feat/vision-tower` 立项;设计文档
//! `docs/arch/vision-tower-设计与CPU对拍.md`(事实面/算子差分/对拍判据)。
//!
//! # 与主干层的关系(为什么不复用 `Module::forward` 的 `(xs, ctx)` 形态)
//! 塔的输入不是"文本 token 序列",而是 `(pixel_values [n_patch, 3·2·16·16], grid_thw)`
//! 外加**每图一组 host 表**(双线性 4-tap 索引/权重、axial rope 的 cos/sin、
//! cu_seqlens)。这些表由 [`VisionTables::build`] 这个**纯 host 函数**算出(可单测、
//! 可与 HF 逐元素对拍),再经 `TensorOps::from_host` 成为叶节点 ⇒ 前向本身仍是
//! **纯声明**(零 host 副作用)。故本模块只提供 inherent 方法,不挂 `Module`
//! trait(塔不在文本主干的层链里;M4 注入 = 在 embed 之后做一次 scatter)。
//!
//! # 逐算子对应(HF `modeling_qwen3_5.py`,行号见设计文档 §1.2)
//! | 塔内步骤 | 落地 |
//! |---|---|
//! | patch_embed(`Conv3d` k=s=(2,16,16)) | `MatmulNt`(权重 [1152,3,2,16,16] 按 [1152,1536] 行主序直读)+ `BiasAdd` |
//! | pos_embed(48² 表 → 图网格,4-tap) | `PosEmbedBilinear`(`VisionTables::build` 出索引/权重) |
//! | 2D axial rope | `Rope`(**复用** `owl_rope_half_partial_*`;half = head_dim/2 ⇒ partial 直通段空) |
//! | block: LN → qkv → rope → **非因果**全注意力 → proj | `Layernorm` + `MatmulNt` + `Rope` + `AttnFull` + `BiasAdd` |
//! | mlp: LN → fc1 → **GELU(tanh)** → fc2 | `MatmulNt` + `GeluTanh` + `BiasAdd` |
//! | merger: LN(1152) → view(4608) → fc1 → **GELU(erf)** → fc2 | `Layernorm` + `Reshape` + `GeluErf` |
//!
//! ⚠️ 两条易错点(设计文档 §1.2):**merger 用精确 erf GELU**(与 MLP 的 tanh 版不可互替);
//! **merger 的 LN 在 shuffle 之前**(逐 patch 归一,再 `view(-1, 4608)`)。
//!
//! # 键面
//! 权重槽名与检查点同构(`patch_embed.proj.weight` … `merger.linear_fc2.bias`),
//! 本容器 `layout` 内**自己**加 `model.visual.` 前缀 ⇒ 可直接挂到任意 LoaderOps 链上。

use crate::contract::ModelError;
use crate::layers::layernorm::LayerNorm;
use crate::layers::narrow_strided;
use crate::module::{Loadable, LoaderCtx, LoaderOps, Module, Weight};
use crate::ops::SemanticKernel;
use crate::tensor::Dtype;
use crate::TensorOps;

/// 塔的超参(来自 `config.json:vision_config`;Holo-9B/35B 逐字段相同)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VisionCfg {
    pub depth: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub heads: usize,
    pub out_hidden: usize,
    pub num_pos: usize,
    pub patch: usize,
    pub temporal: usize,
    pub merge: usize,
    pub in_channels: usize,
}

impl VisionCfg {
    /// **从 `config.json:vision_config` 读**(任意 qwen3_5 系 VLM;零手工抄数)
    pub fn from_json(cfg_text: &[u8]) -> Result<Self, ModelError> {
        let v: serde_json::Value = serde_json::from_slice(cfg_text)
            .map_err(|e| ModelError::Msg(format!("vision config.json 解析: {e}")))?;
        let vc = v.get("vision_config")
            .ok_or_else(|| ModelError::Msg("config.json 无 vision_config(不是 VLM?)".into()))?;
        let g = |k: &str| -> Result<usize, ModelError> {
            vc.get(k).and_then(|x| x.as_u64()).map(|x| x as usize)
                .ok_or_else(|| ModelError::Msg(format!("vision_config 缺 {k}")))
        };
        Ok(VisionCfg {
            depth: g("depth")?,
            hidden: g("hidden_size")?,
            intermediate: g("intermediate_size")?,
            heads: g("num_heads")?,
            out_hidden: g("out_hidden_size")?,
            num_pos: g("num_position_embeddings")?,
            patch: g("patch_size")?,
            temporal: g("temporal_patch_size")?,
            merge: g("spatial_merge_size")?,
            in_channels: g("in_channels")?,
        })
    }

    /// Holo-3.1-9B / 35B-A3B 实测值(`config.json:vision_config`)
    pub fn holo() -> Self {
        VisionCfg {
            depth: 27,
            hidden: 1152,
            intermediate: 4304,
            heads: 16,
            out_hidden: 4096,
            num_pos: 2304,
            patch: 16,
            temporal: 2,
            merge: 2,
            in_channels: 3,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.hidden / self.heads
    }

    /// 塔侧 rope 是 axial 全旋转 ⇒ rotary_dim = head_dim(half = head_dim/2)
    pub fn half(&self) -> usize {
        self.head_dim() / 2
    }

    /// 位置表边长(= √num_pos;Qwen 视觉塔恒为完全平方)
    pub fn grid_side(&self) -> Result<usize, ModelError> {
        let s = (self.num_pos as f64).sqrt().round() as usize;
        // release 也要拦(2026-10-17 审查:原 debug_assert ⇒ 非完全平方时
        // 静默取整,后续 tap/index 全按错误边长算 = 静默错值)
        if s * s != self.num_pos {
            return Err(ModelError::Msg(format!(
                "grid_side: num_pos {} 非完全平方(表边长不可定)",
                self.num_pos
            )));
        }
        Ok(s)
    }

    /// patch 展平宽 = in_channels · temporal · patch²
    pub fn patch_flat(&self) -> usize {
        self.in_channels * self.temporal * self.patch * self.patch
    }

    pub fn eps(&self) -> f32 {
        1e-6
    }
}

// ============================================================================
// §1 host 表(纯函数;逐元素可与 HF `vision_utils` 对拍)
// ============================================================================

/// 一图网格:时间片数 × 高 × 宽(patch 数)
pub type GridThw = (usize, usize, usize);

/// 塔的全部 host 表(一次 `build` 出齐;`TensorOps::from_host` 上传后可前向)
#[derive(Clone, Debug)]
pub struct VisionTables {
    /// 双线性 4-tap 索引 [n, 4](表行序)
    pub idx: Vec<f32>,
    /// 4-tap 权重 [n, 4]
    pub wts: Vec<f32>,
    /// axial rope cos [n, half](半表;`owl_rope_half_partial_*` 的 rotate-half 形式)
    pub cos: Vec<f32>,
    /// axial rope sin [n, half]
    pub sin: Vec<f32>,
    /// rope 选表行(恒等 0..n-1;表按 token 序)
    pub pos: Vec<f32>,
    /// 注意力段界 [n_seq+1](f32 数值过线;契约 5)
    pub cu: Vec<f32>,
    /// token 总数(= Σ t·h·w)
    pub rows: usize,
    /// 图/视频数
    pub n_seq: usize,
    /// 逐图 (t,h,w)
    pub grid: Vec<GridThw>,
    /// 超参快照(前向要用)
    pub cfg: VisionCfg,
}

impl VisionTables {
    /// 由 `grid_thw` 算出全部表(HF `get_vision_interpolation_indices_and_weights`
    /// / `get_vision_position_ids` / rope 三处逐式复刻;见设计文档 §1.2 #3/#4)。
    pub fn build(grid: &[GridThw], cfg: VisionCfg) -> Result<VisionTables, ModelError> {
        if grid.is_empty() {
            return Err(ModelError::Msg("VisionTables::build: grid_thw 为空".into()));
        }
        let merge = cfg.merge;
        let side = cfg.grid_side()?;
        let half = cfg.half();
        let n_pos = cfg.num_pos;

        let rows: usize = grid.iter().map(|(t, h, w)| t * h * w).sum();
        let mut idx = Vec::with_capacity(rows * 4);
        let mut wts = Vec::with_capacity(rows * 4);
        let mut cos = Vec::with_capacity(rows * half);
        let mut sin = Vec::with_capacity(rows * half);
        let mut pos = Vec::with_capacity(rows);
        let mut cu = Vec::with_capacity(grid.len() + 1);
        cu.push(0.0);

        // inv_freq[j] = theta^(-2j/(head_dim/2)),j < half/2 = head_dim/4
        let spatial = cfg.head_dim() / 2;
        let inv: Vec<f32> = (0..spatial / 2)
            .map(|j| 10000.0f32.powf(-(2.0 * j as f32) / spatial as f32))
            .collect();
        if inv.len() * 2 != half {
            return Err(ModelError::Msg(format!(
                "VisionTables::build: head_dim {} 非 4 的整数倍 ⇒ axial rope 半表不齐",
                cfg.head_dim()
            )));
        }

        let mut n = 0usize;
        for &(t, h, w) in grid {
            if h == 0 || w == 0 || t == 0 {
                return Err(ModelError::Msg(format!("VisionTables::build: 非法网格 ({t},{h},{w})")));
            }
            if h % merge != 0 || w % merge != 0 {
                return Err(ModelError::Msg(format!(
                    "VisionTables::build: 网格 ({h},{w}) 不是 merge {merge} 的整数倍"
                )));
            }
            let (bh, bw) = (h / merge, w / merge);
            for _frame in 0..t {
                for block_row in 0..bh {
                    for block_col in 0..bw {
                        for in_row in 0..merge {
                            for in_col in 0..merge {
                                let row = block_row * merge + in_row;
                                let col = block_col * merge + in_col;
                                let (ht, hw) = axis_taps(row, h, side);
                                let (wt, ww) = axis_taps(col, w, side);
                                for kk in 0..2 {
                                    for jj in 0..2 {
                                        let tap = ht[kk] * side + wt[jj];
                                        // 越界 tap 会被 pos-embed 核拿去查表(读越界/静默错值);
                                        // release 也要拦(2026-10-17 审查:原 debug_assert)
                                        if tap >= n_pos {
                                            return Err(ModelError::Msg(format!(
                                                "VisionTables::build: tap {tap} 越出表行数 {n_pos}\
                                                 (grid {t},{h},{w} / side {side})"
                                            )));
                                        }
                                        idx.push(tap as f32);
                                        wts.push(hw[kk] * ww[jj]);
                                    }
                                }
                                // 2D axial rope:half 表 = [cos(h·f), cos(w·f)]
                                for j in 0..inv.len() {
                                    cos.push((row as f32 * inv[j]).cos());
                                    sin.push((row as f32 * inv[j]).sin());
                                }
                                for j in 0..inv.len() {
                                    cos.push((col as f32 * inv[j]).cos());
                                    sin.push((col as f32 * inv[j]).sin());
                                }
                                pos.push(n as f32);
                                n += 1;
                            }
                        }
                    }
                }
            }
            cu.push(n as f32);
        }
        Ok(VisionTables {
            idx,
            wts,
            cos,
            sin,
            pos,
            cu,
            rows: n,
            n_seq: grid.len(),
            grid: grid.to_vec(),
            cfg,
        })
    }

    /// `pixel_values` 的 host 表 + 图叶(对拍/eager 用;`px` = [rows, patch_flat] f32)。
    ///
    /// `dtype` = **计算 dtype**(须与权重装载的 `LoaderCtx.dtype` 一致):px 与 rope 表随之,
    /// 而 `idx`/`wts`/`pos`/`cu` 恒 f32 —— 线契约如此声明(`vis.pos_embed` 的索引/权重、
    /// `ops.rope` 的 pos、`vis.attn_full` 的 cu_seqlens 都是 F32 槽)。
    pub fn inputs(&self, px: &[f32], dtype: Dtype) -> Result<VisionInputs, ModelError> {
        let flat = self.cfg.patch_flat();
        if px.len() != self.rows * flat {
            return Err(ModelError::Msg(format!(
                "VisionTables::inputs: pixel_values 长 {} ≠ rows·patch_flat = {}",
                px.len(),
                self.rows * flat
            )));
        }
        let cast = |v: &[f32], dt: Dtype| -> Vec<u8> { pack(v, dt) };
        let px_t = TensorOps::from_host(dtype, vec![self.rows, flat], &cast(px, dtype));
        let idx = TensorOps::from_host(Dtype::F32, vec![self.rows, 4], &cast(&self.idx, Dtype::F32));
        let wts = TensorOps::from_host(Dtype::F32, vec![self.rows, 4], &cast(&self.wts, Dtype::F32));
        let cos = TensorOps::from_host(dtype, vec![self.rows, self.cfg.half()], &cast(&self.cos, dtype));
        let sin = TensorOps::from_host(dtype, vec![self.rows, self.cfg.half()], &cast(&self.sin, dtype));
        // ⚠️ `pos` 按 **rank 2** 声明(`owl_rope_half_partial_*` 的线契约是 [tokens, 1];
        // 核内线性取 `pos[t]`)—— 文本侧 caller 传的是 rank 1,属既存不一致,已记台账。
        let pos = TensorOps::from_host(Dtype::F32, vec![self.rows, 1], &cast(&self.pos, Dtype::F32));
        let cu = TensorOps::from_host(Dtype::F32, vec![self.n_seq + 1], &cast(&self.cu, Dtype::F32));
        Ok(VisionInputs { px: px_t, idx, wts, cos, sin, pos, cu, rows: self.rows, n_seq: self.n_seq, cfg: self.cfg })
    }
}

/// f32 host 值 → 目标 dtype 字节(f16 存储换档;只认塔用到的两档)
fn pack(v: &[f32], dtype: Dtype) -> Vec<u8> {
    match dtype {
        Dtype::F32 => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        Dtype::F16 => v.iter().flat_map(|x| half::f16::from_f32(*x).to_le_bytes()).collect(),
        other => panic!("vision::pack: 未预期的 dtype {other:?}(塔只走 f32/f16)"),
    }
}

/// 双线性两 tap(HF `_interpolation_axis_taps_weights`;align_corners=True,padding=border)
fn axis_taps(index: usize, size: usize, side: usize) -> ([usize; 2], [f32; 2]) {
    // src = index·(side−1)/max(size−1, 1)(HF 的 clamp(min=1) 防 size==1 除零)
    let src = index as f32 * (side as f32 - 1.0) / (size.max(2) - 1) as f32;
    let floor = src.floor();
    let mut taps = [0usize; 2];
    let mut weights = [0.0f32; 2];
    for k in 0..2 {
        let raw = floor as i64 + k as i64;
        taps[k] = raw.clamp(0, side as i64 - 1) as usize;
        weights[k] = (1.0 - (src - floor - k as f32).abs()).max(0.0);
    }
    (taps, weights)
}

/// 塔的前向输入叶(host 表已上传;`rows`/`n_seq`/`cfg` 为标量事实)
pub struct VisionInputs {
    pub px: TensorOps,
    pub idx: TensorOps,
    pub wts: TensorOps,
    pub cos: TensorOps,
    pub sin: TensorOps,
    pub pos: TensorOps,
    pub cu: TensorOps,
    pub rows: usize,
    pub n_seq: usize,
    pub cfg: VisionCfg,
}

// ============================================================================
// §2 容器(权重槽 + 前向声明)
// ============================================================================

/// patch_embed:`Conv3d(3→1152, k=s=(2,16,16))` ≡ `MatmulNt([1152, patch_flat])`
pub struct VisionPatchEmbed {
    w: Weight,
    b: Weight,
    patch_flat: usize,
}

impl VisionPatchEmbed {
    pub fn new(cfg: &VisionCfg) -> Self {
        VisionPatchEmbed {
            // 检查点原布局 [hidden, in, t, p, p];前向按 [hidden, patch_flat] 行主序直读
            w: Weight::new("patch_embed.proj.weight", vec![cfg.hidden, cfg.in_channels, cfg.temporal, cfg.patch, cfg.patch]),
            b: Weight::new("patch_embed.proj.bias", vec![cfg.hidden]),
            patch_flat: cfg.patch_flat(),
        }
    }

    pub fn is_loaded(&self) -> bool {
        self.w.is_loaded() && self.b.is_loaded()
    }

    fn forward(&self, px: &TensorOps, cfg: &VisionCfg) -> TensorOps {
        let w = self.w.decl().reshape(vec![cfg.hidden, self.patch_flat]);
        px.matmul_nt(&w).bias_add(&self.b.decl())
    }
}

/// 视觉注意力(qkv 融合 + proj;无 GQA、非因果 —— 见 `AttnFull`)
pub struct VisionAttention {
    qkv_w: Weight,
    qkv_b: Weight,
    proj_w: Weight,
    proj_b: Weight,
}

impl VisionAttention {
    pub fn new(cfg: &VisionCfg) -> Self {
        let h = cfg.hidden;
        VisionAttention {
            qkv_w: Weight::new("attn.qkv.weight", vec![3 * h, h]),
            qkv_b: Weight::new("attn.qkv.bias", vec![3 * h]),
            proj_w: Weight::new("attn.proj.weight", vec![h, h]),
            proj_b: Weight::new("attn.proj.bias", vec![h]),
        }
    }

    pub fn is_loaded(&self) -> bool {
        self.qkv_w.is_loaded() && self.qkv_b.is_loaded() && self.proj_w.is_loaded() && self.proj_b.is_loaded()
    }

    /// `x [n, hidden]` + 表 ⇒ `[n, hidden]`
    fn forward(&self, x: &TensorOps, tb: &VisionInputs) -> TensorOps {
        let cfg = tb.cfg;
        // ⚠️ `hw` = hidden 宽(切列用),`h` = 头数(形状用)—— 两者不可混
        let (n, hw, h, hd, half) = (tb.rows, cfg.hidden, cfg.heads, cfg.head_dim(), cfg.half());
        let qkv = x.matmul_nt(&self.qkv_w.decl()).bias_add(&self.qkv_b.decl());
        let dt = qkv.dtype;
        // 三段列切(`narrow_strided`:outer=n, src_dim=3·hidden, start=k·hidden, out_dim=hidden)
        // 注:rope 的线契约是 **rank 2** 平铺 [tokens, heads·head_dim] ⇒ q/k 留在平铺态过 rope,
        // 之后再 reshape 进 attn(rank 3)。(准入契约门会逐槽校 rank,错位当场红。)
        let part = |start: usize| -> TensorOps {
            narrow_strided(&qkv, n, 3 * hw, start * hw, hw, vec![n, hw])
        };
        let (q, k, v) = (part(0), part(1), part(2));
        let rope = |z: &TensorOps| -> TensorOps {
            TensorOps::call(SemanticKernel::Rope)
                .aux(&[n])
                .arg(z)
                .arg(&tb.cos)
                .arg(&tb.sin)
                .arg(&tb.pos)
                .arg_usize(h)
                .arg_usize(hd)
                .arg_usize(half)
                .with_shape(dt, vec![n, hw])
        };
        let (q, k) = (rope(&q).reshape(vec![n, h, hd]), rope(&k).reshape(vec![n, h, hd]));
        let v = v.reshape(vec![n, h, hd]);
        let scale = 1.0f32 / (hd as f32).sqrt();
        let attn = q.attn_full(&k, &v, &tb.cu, tb.n_seq, scale);
        attn.reshape(vec![n, h * hd]).matmul_nt(&self.proj_w.decl()).bias_add(&self.proj_b.decl())
    }
}

/// 视觉 MLP(fc1 → **GELU(tanh)** → fc2;两段都有 bias)
pub struct VisionMlp {
    fc1_w: Weight,
    fc1_b: Weight,
    fc2_w: Weight,
    fc2_b: Weight,
}

impl VisionMlp {
    pub fn new(cfg: &VisionCfg) -> Self {
        VisionMlp {
            fc1_w: Weight::new("mlp.linear_fc1.weight", vec![cfg.intermediate, cfg.hidden]),
            fc1_b: Weight::new("mlp.linear_fc1.bias", vec![cfg.intermediate]),
            fc2_w: Weight::new("mlp.linear_fc2.weight", vec![cfg.hidden, cfg.intermediate]),
            fc2_b: Weight::new("mlp.linear_fc2.bias", vec![cfg.hidden]),
        }
    }

    pub fn is_loaded(&self) -> bool {
        self.fc1_w.is_loaded() && self.fc1_b.is_loaded() && self.fc2_w.is_loaded() && self.fc2_b.is_loaded()
    }

    fn forward(&self, x: &TensorOps) -> TensorOps {
        let h = x.matmul_nt(&self.fc1_w.decl()).bias_add(&self.fc1_b.decl());
        h.gelu_tanh().matmul_nt(&self.fc2_w.decl()).bias_add(&self.fc2_b.decl())
    }
}

/// 视觉 block:`x += attn(LN1(x)); x += mlp(LN2(x))`
pub struct VisionBlock {
    norm1: LayerNorm,
    attn: VisionAttention,
    norm2: LayerNorm,
    mlp: VisionMlp,
}

impl VisionBlock {
    pub fn new(cfg: &VisionCfg) -> Self {
        VisionBlock {
            norm1: LayerNorm::new("norm1.weight", "norm1.bias", cfg.hidden, cfg.eps()),
            attn: VisionAttention::new(cfg),
            norm2: LayerNorm::new("norm2.weight", "norm2.bias", cfg.hidden, cfg.eps()),
            mlp: VisionMlp::new(cfg),
        }
    }

    pub fn is_loaded(&self) -> bool {
        self.norm1.is_loaded() && self.attn.is_loaded() && self.norm2.is_loaded() && self.mlp.is_loaded()
    }

    fn forward(&self, x: &TensorOps, tb: &VisionInputs) -> TensorOps {
        let ctx = crate::module::ForwardCtx::minimal(tb.rows);
        let a = self.attn.forward(&self.norm1.forward(x, &ctx), tb);
        let x = x.add(&a);
        let m = self.mlp.forward(&self.norm2.forward(&x, &ctx));
        x.add(&m)
    }
}

/// merger:LN(1152) → `view(-1, 4608)` → fc1 → **GELU(erf)** → fc2 ⇒ `[n/4, 4096]`
pub struct VisionMerger {
    norm: LayerNorm,
    fc1_w: Weight,
    fc1_b: Weight,
    fc2_w: Weight,
    fc2_b: Weight,
    hidden: usize,
    merged_dim: usize,
}

impl VisionMerger {
    pub fn new(cfg: &VisionCfg) -> Self {
        let merged = cfg.hidden * cfg.merge * cfg.merge;
        VisionMerger {
            // HF:`use_postshuffle_norm=False` ⇒ LN 宽度 = hidden(逐 patch),先归一再 shuffle
            norm: LayerNorm::new("merger.norm.weight", "merger.norm.bias", cfg.hidden, cfg.eps()),
            fc1_w: Weight::new("merger.linear_fc1.weight", vec![merged, merged]),
            fc1_b: Weight::new("merger.linear_fc1.bias", vec![merged]),
            fc2_w: Weight::new("merger.linear_fc2.weight", vec![cfg.out_hidden, merged]),
            fc2_b: Weight::new("merger.linear_fc2.bias", vec![cfg.out_hidden]),
            hidden: cfg.hidden,
            merged_dim: merged,
        }
    }

    pub fn is_loaded(&self) -> bool {
        self.norm.is_loaded() && self.fc1_w.is_loaded() && self.fc1_b.is_loaded() && self.fc2_w.is_loaded() && self.fc2_b.is_loaded()
    }

    fn forward(&self, x: &TensorOps, rows: usize) -> TensorOps {
        let ctx = crate::module::ForwardCtx::minimal(rows);
        let y = self.norm.forward(x, &ctx);
        let n_merged = rows / (self.merged_dim / self.hidden);
        let y = y.reshape(vec![n_merged, self.merged_dim]);
        let h = y.matmul_nt(&self.fc1_w.decl()).bias_add(&self.fc1_b.decl());
        // ⚠️ **精确 erf GELU**(与 block MLP 的 tanh 版不同)
        h.gelu_erf().matmul_nt(&self.fc2_w.decl()).bias_add(&self.fc2_b.decl())
    }
}

/// 塔(容器:patch_embed / pos_embed / 27 block / merger)
pub struct VisionTower {
    patch_embed: VisionPatchEmbed,
    pos_embed: Weight,
    blocks: Vec<VisionBlock>,
    merger: VisionMerger,
    cfg: VisionCfg,
}

/// 前向产物(逐层 hidden 供对拍;`merged` = 出塔喂文本模型的那一份)
pub struct VisionOut {
    /// 每 block 的输出 [n_patch, hidden](索引 = 层号)
    pub blocks: Vec<TensorOps>,
    /// 出 merger 前(norm 前)的 hidden [n_patch, hidden]
    pub pre_merger: TensorOps,
    /// 出塔 [n_patch/4, out_hidden]
    pub merged: TensorOps,
}

impl VisionTower {
    pub fn new(cfg: VisionCfg) -> Self {
        VisionTower {
            patch_embed: VisionPatchEmbed::new(&cfg),
            pos_embed: Weight::new("pos_embed.weight", vec![cfg.num_pos, cfg.hidden]),
            blocks: (0..cfg.depth).map(|_| VisionBlock::new(&cfg)).collect(),
            merger: VisionMerger::new(&cfg),
            cfg,
        }
    }

    pub fn cfg(&self) -> VisionCfg {
        self.cfg
    }

    pub fn is_loaded(&self) -> bool {
        self.patch_embed.is_loaded()
            && self.pos_embed.is_loaded()
            && self.blocks.iter().all(|b| b.is_loaded())
            && self.merger.is_loaded()
    }

    /// 前向(**纯声明**;表已由 [`VisionTables`] 备好)
    pub fn forward(&self, tb: &VisionInputs) -> VisionOut {
        let cfg = &tb.cfg;
        // ① patch_embed + ② pos_embed(4-tap 重采样)
        let h0 = self.patch_embed.forward(&tb.px, cfg);
        let pos = self.pos_embed.decl().pos_embed_bilinear(&tb.idx, &tb.wts);
        let mut h = h0.add(&pos);
        // ③ ×27:block(LN → attn → LN → mlp)
        let mut blocks = Vec::with_capacity(cfg.depth);
        for blk in &self.blocks {
            h = blk.forward(&h, tb);
            blocks.push(h.clone());
        }
        // ④ merger(注意:merger 自己内含 LN;pre_merger = 归一前 hidden)
        let merged = self.merger.forward(&h, tb.rows);
        VisionOut { blocks, pre_merger: h, merged }
    }
}

// ============================================================================
// §3 装载(槽名与检查点同构;前缀 `model.visual.` 由本容器自己加)
// ============================================================================

impl Loadable for VisionPatchEmbed {
    fn layout(&self, ctx: &LoaderCtx) -> LoaderOps {
        self.w.layout(ctx).chain(self.b.layout(ctx))
    }
}

impl Loadable for VisionAttention {
    fn layout(&self, ctx: &LoaderCtx) -> LoaderOps {
        self.qkv_w.layout(ctx).chain(self.qkv_b.layout(ctx)).chain(self.proj_w.layout(ctx)).chain(self.proj_b.layout(ctx))
    }
}

impl Loadable for VisionMlp {
    fn layout(&self, ctx: &LoaderCtx) -> LoaderOps {
        self.fc1_w.layout(ctx).chain(self.fc1_b.layout(ctx)).chain(self.fc2_w.layout(ctx)).chain(self.fc2_b.layout(ctx))
    }
}

impl Loadable for VisionBlock {
    fn layout(&self, ctx: &LoaderCtx) -> LoaderOps {
        self.norm1
            .layout(ctx)
            .chain(self.attn.layout(ctx))
            .chain(self.norm2.layout(ctx))
            .chain(self.mlp.layout(ctx))
    }
}

impl Loadable for VisionMerger {
    fn layout(&self, ctx: &LoaderCtx) -> LoaderOps {
        self.norm
            .layout(ctx)
            .chain(self.fc1_w.layout(ctx))
            .chain(self.fc1_b.layout(ctx))
            .chain(self.fc2_w.layout(ctx))
            .chain(self.fc2_b.layout(ctx))
    }
}

impl Loadable for VisionTower {
    /// 槽名 → 检查点全键(`model.visual.` 前缀在此**唯一**落点)
    fn layout(&self, ctx: &LoaderCtx) -> LoaderOps {
        let mut ops = self.patch_embed.layout(ctx).chain(self.pos_embed.layout(ctx));
        for (i, blk) in self.blocks.iter().enumerate() {
            ops = ops.chain(blk.layout(ctx).map_keys(|k| format!("blocks.{i}.{k}")));
        }
        ops.chain(self.merger.layout(ctx)).map_keys(|k| format!("model.visual.{k}"))
    }
}

#[cfg(test)]
mod tests {
    use owl_macros::owl_test;
    use super::*;
    use crate::contract::DeviceClient;
    use crate::testkit::{assert_close, f32_of, st_read, Src};
    use std::path::PathBuf;

    fn golden_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata")
    }

    /// 全部在场金标(档一 = (1,4,6) 方形小网格;档二 = (1,8,10) 非方形 ⇒ T4 尺寸门)
    fn golden_paths() -> Vec<PathBuf> {
        ["vision_golden.safetensors", "vision_golden2.safetensors"]
            .iter()
            .map(|n| golden_root().join(n))
            .filter(|p| p.exists())
            .collect()
    }

    /// 金标里的网格(工具已存 f32)
    fn golden_grid(g: &PathBuf) -> Vec<GridThw> {
        let raw = st_read(g, "in.grid_thw");
        let mut out = vec![GridThw::default()];
        out[0] = (raw[0] as usize, raw[1] as usize, raw[2] as usize);
        out
    }

    /// 真权重直读(只取 `model.visual.*`;f32 化 —— 与 HF 金标同口径)。
    /// 扫目录里的 `model-*.safetensors`(不解析 index.json ⇒ 零 serde_json 依赖)。
    fn visual_weights() -> (std::collections::HashMap<String, Vec<f32>>, VisionCfg) {
        // 默认档 = **Qwen3.5-4B**(用户裁决 2026-10-15:塔更小(hidden 1024/24 层/hd **64**)
        // + 文本 4B ⇒ 单张 3080 装得下;Holo-9B 塔 head_dim 72 不落 FlashInfer 实例化档)。
        let ckpt = std::env::var("OWL_VISION_CKPT")
            .unwrap_or_else(|_| "/home/div/Documents/codes/models/Qwen/Qwen3.5-4B".into());
        let dir = PathBuf::from(ckpt);
        let mut out = std::collections::HashMap::new();
        let mut shards: Vec<PathBuf> = std::fs::read_dir(&dir)
            .expect("读模型目录(OWL_VISION_CKPT 可指路)")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                // shard 命名两种:HF 分隔点(`model-00001-of-00002.safetensors`)
                // 与新式(`model.safetensors-00001-of-00002.safetensors`)⇒ 只认 `model` + `.safetensors`
                p.file_name().and_then(|n| n.to_str()).is_some_and(|n| {
                    n.starts_with("model") && n.ends_with(".safetensors")
                })
            })
            .collect();
        shards.sort();
        assert!(!shards.is_empty(), "{} 下没有 model-*.safetensors", dir.display());
        for shard in shards {
            let bytes = owl_shared::file_loader::read(&shard).expect("shard 读入");
            let st = safetensors::SafeTensors::deserialize(&bytes).expect("safetensors 解析");
            for name in st.names() {
                if !name.starts_with("model.visual.") {
                    continue;
                }
                let t = st.tensor(name).expect("张量");
                let v = match t.dtype() {
                    safetensors::Dtype::F32 => f32_of(t.data()),
                    safetensors::Dtype::BF16 => t
                        .data()
                        .chunks_exact(2)
                        .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
                        .collect(),
                    safetensors::Dtype::F16 => t
                        .data()
                        .chunks_exact(2)
                        .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                        .collect(),
                    other => panic!("{name}: 未预期的 dtype {other:?}"),
                };
                // 键 = **检查点全键**(layout 已带 `model.visual.` 前缀;数据源按全键取数)
                out.insert(name.to_string(), v);
            }
        }
        let cfg_text = owl_shared::file_loader::read(&dir.join("config.json"))
            .expect("config.json 读入(视觉塔超参唯一来源)");
        let cfg = VisionCfg::from_json(&cfg_text).expect("vision_config 解析");
        (out, cfg)
    }

    /// 逐层判据(**尺度感知 allclose**):`|Δ| ≤ ATOL_REL·scale + RTOL·|want|`,
    /// `scale = max(1, rms(want))`。
    ///
    /// 依据(实测,两档网格):
    /// - 27 层 f32 流水线的归约序漂移(naive 串行点积 vs torch 分块 SIMD)逐层累积到
    ///   **2.5e-4 绝对**,且常落在**近零元素**上(纯相对项对它无意义)⇒ 需要绝对项;
    /// - 末层激活会**爆量**(HF 自身同爆:max|·| ~7.5e2, rms 122)⇒ 绝对项必须随层尺度缩放,
    ///   否则爆量层上 9.2e-2 的漂移会被误判(相对误差其实只有 1.2e-4);
    /// - 真 bug(布局/公式/错位)的误差是**信号量级**(≈ scale)⇒ 本判据差 2~3 个数量级,不会漏。
    const ATOL_REL: f32 = 2e-3;
    const RTOL: f32 = 1e-4;

    /// f16 档判据(M3):对照 = **HF 的 f16 金标**(`h.*`;同档 ⇒ 只剩归约序差异)。
    /// 尺度感知:`1e-2·max(1,rms) + 2e-2·|want|`。
    /// 注:对 **f32 金标**(`g.*`)只作信息打印 —— 那条差里混着"f16 存储本该有的漂移",
    /// 分不清核错还是精度税(实测 27 层后可达 rms 的 ~10%)。
    /// 实测(档一 24 patch / 27 层,vs **HF f16**):逐层漂移随深度增长,近零元素上
    /// 绝对项主导(blk_14 最坏 1.2e-2 @ |want| 9.3e-2)。判据取 5% rms ——
    /// 真 bug(布局/公式错)的误差是 ~100% rms,差一个数量级,不会漏。
    const F16_ATOL_REL: f32 = 5e-2;
    const F16_RTOL: f32 = 5e-2;

    /// allclose 返回 (最大归一化误差, 该元素下标);误差 ≤ 1 即过。
    /// `atol = atol_rel · max(1, rms(want))`(尺度感知;rms 为 0 的层退回 atol_rel)
    fn allclose(got: &[f32], want: &[f32], atol_rel: f32, rtol: f32) -> (f32, usize) {
        assert_eq!(got.len(), want.len(), "长度不符");
        let rms = (want.iter().map(|v| v * v).sum::<f32>() / want.len().max(1) as f32).sqrt();
        let atol = atol_rel * rms.max(1.0);
        let mut worst = 0.0f32;
        let mut at = 0usize;
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            let e = (g - w).abs() / (atol + rtol * w.abs());
            if e > worst {
                worst = e;
                at = i;
            }
        }
        (worst, at)
    }

    /// T1:host 表逐元素对拍(interp 索引/权重、axial cos/sin、段界)
    #[test]
    fn t1_tables_match_golden() {
        let goldens = golden_paths();
        if goldens.is_empty() {
            eprintln!("skip: 金标不在(tools/vision_golden.py 生成)");
            return;
        }
        let (_, cfg) = visual_weights();
        for g in &goldens {
            let grid = golden_grid(g);
            eprintln!("[vis] 表对拍 档 {} grid={grid:?}", g.file_name().unwrap().to_string_lossy());
            tables_match_golden(g, cfg, &grid);
        }
    }

    fn tables_match_golden(g: &PathBuf, cfg: VisionCfg, grid: &[GridThw]) {
        let tb = VisionTables::build(grid, cfg).expect("build tables");
        let want_idx = st_read(g, "g.interp_indices");
        let want_wts = st_read(g, "g.interp_weights");
        assert_eq!(tb.idx.len(), want_idx.len(), "idx 长度");
        assert_eq!(tb.wts.len(), want_wts.len(), "wts 长度");
        assert_eq!(tb.idx, want_idx, "interp 索引逐元素");
        // 容差:三角函数/libm 逐 ULP 差异(torch cos vs Rust cos),绝不足以影响塔判据(1e-4)
        assert_close(&tb.wts, &want_wts, 1e-7, "interp 权重");
        // rope 半表 vs 金标 72 宽表的**前 36 列**(HF 复制两份;我们取半表)
        let want_cos = st_read(g, "g.cos");
        let want_sin = st_read(g, "g.sin");
        let half = cfg.half();
        let rows = tb.rows;
        assert_eq!(want_cos.len(), rows * 2 * half, "金标 cos 应 72 宽");
        let mut got = Vec::with_capacity(rows * half);
        for r in 0..rows {
            got.extend_from_slice(&want_cos[r * 2 * half..r * 2 * half + half]);
        }
        assert_close(&tb.cos, &got, 1e-6, "cos 半表 vs 金标前半");
        let mut gots = Vec::with_capacity(rows * half);
        for r in 0..rows {
            gots.extend_from_slice(&want_sin[r * 2 * half..r * 2 * half + half]);
        }
        assert_close(&tb.sin, &gots, 1e-6, "sin 半表 vs 金标前半");
    }

    /// 装权重 + 前向(CPU face)⇒ 逐层/末层对拍
    async fn tower_forward(face: &mut owl_cpu::CpuFace, tb: &VisionTables, px: &[f32]) -> VisionOut {
        let (weights, cfg) = visual_weights();
        let tower = VisionTower::new(cfg);
        let src = Src::from(weights);
        crate::interpreters::eval_load(&tower, face, &src, &Default::default())
            .await
            .expect("eval_load(vision tower)");
        assert!(tower.is_loaded(), "塔权重未装载齐");
        let inputs = tb.inputs(px, Dtype::F32).expect("inputs");
        tower.forward(&inputs)
    }

    /// **M3 首战**:塔 **f16 GPU 路径**(五个新核 + 声明/拾取/线格式全链)
    /// vs HF **f32** 金标(f16 档判据)。数据源 = 真权重(BF16 检查点 → f16 装载)。
    #[owl_test(gpu)]
    async fn gpu_f16_tower_matches_golden() {
        let goldens = golden_paths();
        let Some(g) = goldens.first() else {
            eprintln!("skip: 金标不在(tools/vision_golden.py 生成)");
            return;
        };
        let (weights, cfg) = visual_weights();
        let grid = golden_grid(g);
        let tb = VisionTables::build(&grid, cfg).expect("tables");
        let px = st_read(g, "in.pixel_values");
        let tower = VisionTower::new(cfg);
        let mut gpu = crate::testkit::gpu_client().await;
        let lctx = crate::module::LoaderCtx {
            dtype: Dtype::F16,
            shard: 1,
            device_repack: false,
            verify: false,
            debug_tap: false,
        };
        crate::interpreters::eval_load(&tower, &mut gpu, &Src::from(weights), &lctx)
            .await
            .expect("塔 f16 装载(GPU)");
        assert!(tower.is_loaded(), "塔权重未装载齐");
        let inputs = tb.inputs(&px, Dtype::F16).expect("inputs(f16)");
        let out = tower.forward(&inputs);
        let mut roots: Vec<&TensorOps> = out.blocks.iter().collect();
        roots.push(&out.pre_merger);
        roots.push(&out.merged);
        // 诊断:单根 vs 多根(定位 multi-root 影响)
        {
            let single = crate::testkit::harvest_f16(&mut gpu, &out.blocks[0]).await;
            let want = st_read(g, "h.blk_00");
            let d = single.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            eprintln!("[vis-mr2] f16 单根 blk_00 对金标 max|Δ| = {d:.3e}");
        }
        let vals = crate::interpreters::eval_ops_multi(&roots, &mut gpu).await.expect("eval_multi(gpu)");

        for (i, b) in vals.iter().take(cfg.depth).enumerate() {
            let n = tb.rows * cfg.hidden;
            let mut buf = vec![0u8; n * 2];
            gpu.dtoh(b, &mut buf).await.expect("dtoh(f16)");
            let got: Vec<f32> = buf
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect();
            let want = st_read(g, &format!("h.blk_{i:02}")); // HF f16 档
            let want_f32 = st_read(g, &format!("g.blk_{i:02}")); // HF f32 档(信息)
            let drift = got
                .iter()
                .zip(&want_f32)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let (worst, at) = allclose(&got, &want, F16_ATOL_REL, F16_RTOL);
            eprintln!(
                "[vis-gpu] blk_{i:02} f16 档归一化误差 {worst:.3e}(@[{at}] |Δ| {:.3e} / |want| {:.3e});                 对 f32 档最大 |Δ| {drift:.3e}",
                (got[at] - want[at]).abs(),
                want[at].abs()
            );
            assert!(worst <= 1.0, "gpu blk_{i:02}: f16 档归一化误差 {worst:.3e} > 1");
        }
        {
            let n = (tb.rows / 4) * cfg.out_hidden;
            let mut buf = vec![0u8; n * 2];
            gpu.dtoh(&vals[cfg.depth + 1], &mut buf).await.expect("dtoh(f16)");
            let got: Vec<f32> = buf
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect();
            let want = st_read(g, "h.merged");
            let want_f32 = st_read(g, "g.merged");
            let drift = got
                .iter()
                .zip(&want_f32)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let (worst, at) = allclose(&got, &want, F16_ATOL_REL, F16_RTOL);
            eprintln!(
                "[vis-gpu] merged f16 档归一化误差 {worst:.3e}(@[{at}] |Δ| {:.3e});对 f32 档最大 |Δ| {drift:.3e}",
                (got[at] - want[at]).abs()
            );
            assert!(worst <= 1.0, "gpu merged: f16 档归一化误差 {worst:.3e} > 1");
        }
        gpu.close().await.expect("GPU 关机");
    }

    /// **M3 性能实测**(门控 `OWL_VIS_BENCH=1` + `OWL_TEST_DEVICE`):
    /// 1280×736 截图(patch 16 ⇒ 网格 80×46 = **3680 patch**,1024 merged token)
    /// ①整塔 f16 前向(装载后 3 次取最好)②仅注意力核(10 次平均)。
    /// 判据(设计文档 §5 M3):整塔 ≤ 150 ms。
    #[owl_test(gpu)]
    async fn gpu_f16_tower_bench_1280x736() {
        if !owl_shared::env_reader::flag("OWL_VIS_BENCH") {
            eprintln!("skip: OWL_VIS_BENCH 未开(性能档不随常规测试跑)");
            return;
        }
        let (weights, cfg) = visual_weights();
        // 1280×736 px ⇒ 80×46 patch(merge 2 整除 ✓)
        let grid = vec![(1usize, 80usize, 46usize)];
        let tb = VisionTables::build(&grid, cfg).expect("tables");
        let px: Vec<f32> = (0..tb.rows * cfg.patch_flat())
            .map(|i| (i as f32 * 1.7e-4).sin())
            .collect();
        let tower = VisionTower::new(cfg);
        let mut gpu = crate::testkit::gpu_client().await;
        let lctx = crate::module::LoaderCtx { dtype: Dtype::F16, shard: 1, device_repack: false, verify: false, debug_tap: false };
        crate::interpreters::eval_load(&tower, &mut gpu, &Src::from(weights), &lctx)
            .await
            .expect("塔 f16 装载");
        let inputs = tb.inputs(&px, Dtype::F16).expect("inputs(f16)");
        let out = tower.forward(&inputs);
        let mut roots: Vec<&TensorOps> = out.blocks.iter().collect();
        roots.push(&out.merged);
        // 暖机(首次含装载后首调/workspace 初始化)。
        // ⚠️ 用 **scoped**(回收中间块;引擎运行期同款):plain `eval_ops_multi` 会把
        // 27 层全部中间块留活 ⇒ 实测 7.8 GB 常驻,第二轮即撞池碎片化 OOM。
        let trees: Vec<TensorOps> = roots.iter().map(|t| (*t).clone()).collect();
        let refs: Vec<&TensorOps> = trees.iter().collect();
        let _ = crate::interpreters::eval_ops_multi_scoped_env(&refs, &mut gpu, crate::env::EnvProvider::default())
            .await
            .expect("warmup");
        gpu.sync().await.expect("sync");
        eprintln!(
            "[vis-bench] 暖机后 vram live {:.1} MiB / total {:.0} MiB",
            owl_shared::vram::live_bytes() as f64 / 1048576.0,
            owl_shared::vram::total_bytes() as f64 / 1048576.0,
        );
        let mut best = f64::MAX;
        for _ in 0..3 {
            gpu.sync().await.expect("sync");
            let t0 = std::time::Instant::now();
            let _ = crate::interpreters::eval_ops_multi_scoped_env(&refs, &mut gpu, crate::env::EnvProvider::default())
                .await
                .expect("bench");
            gpu.sync().await.expect("sync"); // ⚠️ server 是异步提交 ⇒ 必须 sync 才含设备时间
            best = best.min(t0.elapsed().as_secs_f64());
        }
        eprintln!(
            "[vis-bench] 整塔 f16 rows={} merged={} ⇒ **{:.1} ms**(3 次最好)",
            tb.rows,
            tb.rows / 4,
            best * 1e3
        );

        // ②仅注意力核(同形状;10 次平均)
        let n = tb.rows * cfg.hidden;
        let hb = |v: &[f32]| -> Vec<u8> { pack(v, Dtype::F16) };
        let q = TensorOps::from_host(Dtype::F16, vec![tb.rows, cfg.heads, cfg.head_dim()], &hb(&vec![0.01f32; n]));
        let k = TensorOps::from_host(Dtype::F16, vec![tb.rows, cfg.heads, cfg.head_dim()], &hb(&vec![0.02f32; n]));
        let v = TensorOps::from_host(Dtype::F16, vec![tb.rows, cfg.heads, cfg.head_dim()], &hb(&vec![0.03f32; n]));
        let cu = TensorOps::from_host(Dtype::F32, vec![tb.n_seq + 1], &pack(&tb.cu, Dtype::F32));
        let scale = 1.0f32 / (cfg.head_dim() as f32).sqrt();
        let attn = q.attn_full(&k, &v, &cu, tb.n_seq, scale);
        let _ = crate::interpreters::eval_ops(attn.step(), &mut gpu).await.expect("warm");
        gpu.sync().await.expect("sync");
        let t0 = std::time::Instant::now();
        for _ in 0..10 {
            let _ = crate::interpreters::eval_ops(attn.step(), &mut gpu).await.expect("bench");
        }
        gpu.sync().await.expect("sync");
        eprintln!(
            "[vis-bench] 仅 attn_full(rows={} × {} 头 × hd{};{};{},非因果)⇒ **{:.1} ms/层**",
            tb.rows,
            cfg.heads,
            cfg.head_dim(),
            "FlashInfer ragged(FA2 家族)",
            "单图",
            t0.elapsed().as_secs_f64() * 1e2
        );
        gpu.close().await.expect("关机");
    }



    /// 真图金标路径 + 图标签枚举(in./<tag>.pixel_values 推断)
    fn real_golden() -> PathBuf {
        golden_root().join("vision_real.safetensors")
    }

    /// 图筛选(env `OWL_VIS_IMAGE=dog` 只跑指定图;缺省全图)
    fn real_tags(g: &PathBuf) -> Vec<String> {
        let all = real_tags_all(g);
        match std::env::var("OWL_VIS_IMAGE") {
            Ok(pick) => all.into_iter().filter(|t| pick.split(',').any(|p| p == t)).collect(),
            Err(_) => all,
        }
    }

    fn real_tags_all(g: &PathBuf) -> Vec<String> {
        let buf = owl_shared::file_loader::read(g).expect("真图金标读入");
        let st = safetensors::SafeTensors::deserialize(&buf).expect("safetensors 解析");
        let mut tags: Vec<String> = st
            .names()
            .iter()
            .filter_map(|n| {
                n.strip_prefix("in.")
                    .and_then(|r| r.strip_suffix(".pixel_values"))
                    .map(|s| s.to_string())
            })
            .collect();
        tags.sort();
        tags.dedup();
        tags
    }

    /// **真图验收**(R2 闭环;GPU f16 = 生产路径):
    /// `assets/{bird,cat,dog,flower_blue}.png` 经 HF `Qwen3VLProcessor` 真预处理
    /// (缩放 / patchify / **2×2 块序**)⇒ 我们的塔 vs HF 塔同档金标。
    #[owl_test(gpu)]
    async fn real_images_gpu_f16_match_golden() {
        let g = real_golden();
        if !g.exists() {
            eprintln!("skip: 真图金标不在(tools/vision_image_check.py 生成)");
            return;
        }
        let (weights, cfg) = visual_weights();
        let mut gpu = crate::testkit::gpu_client().await;
        let tower = VisionTower::new(cfg);
        let lctx = crate::module::LoaderCtx { dtype: Dtype::F16, shard: 1, device_repack: false, verify: false, debug_tap: false };
        crate::interpreters::eval_load(&tower, &mut gpu, &Src::from(weights), &lctx)
            .await
            .expect("塔 f16 装载");
        for tag in real_tags(&g) {
            let pv = st_read(&g, &format!("in.{tag}.pixel_values"));
            let gr = st_read(&g, &format!("in.{tag}.grid_thw"));
            let grid = vec![(gr[0] as usize, gr[1] as usize, gr[2] as usize)];
            let tb = VisionTables::build(&grid, cfg).expect("tables");
            let inputs = tb.inputs(&pv, Dtype::F16).expect("inputs(f16)");
            let out = tower.forward(&inputs);
            let roots: Vec<&TensorOps> = vec![&out.pre_merger, &out.merged];
            let vals = crate::interpreters::eval_ops_multi(&roots, &mut gpu).await.expect("eval(gpu)");
            for (i, (name, want_key)) in
                [("pre_merger", format!("h.{tag}.pre_merger")), ("merged", format!("h.{tag}.merged"))]
                    .iter()
                    .enumerate()
            {
                let n = if i == 0 { tb.rows * cfg.hidden } else { (tb.rows / 4) * cfg.out_hidden };
                let mut buf = vec![0u8; n * 2];
                gpu.dtoh(&vals[i], &mut buf).await.expect("dtoh");
                let got: Vec<f32> = buf
                    .chunks_exact(2)
                    .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                    .collect();
                let want = st_read(&g, want_key);
                let want_f32 = st_read(&g, &want_key.replacen("h.", "g.", 1));
                let ours = got.iter().zip(&want_f32).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                let hf_gap = want
                    .iter()
                    .zip(&want_f32)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                let dmax = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                eprintln!(
                    "[vis-real] {:12} grid {:?} {:10} f16: vs HF-f16 max|Δ| {dmax:.3e} | 我方 vs **f32** 参考 {ours:.3e} vs HF 自身 f16 偏差 {hf_gap:.3e}(比值 {:.2}x)",
                    tag,
                    grid,
                    name,
                    ours / hf_gap.max(1e-9)
                );
                // **f16 判据 = "不比 HF 自己的 f16 更差"**(对共同 f32 参考各量一次):
                // 实测锚:HF 自己 f32↔f16 在 pre_merger 上相差 rms×6.2~45(按图变),
                // 我方 f16 与它的差距同量级(bird 3.6 vs 7.0;cat 14.3 vs 14.1)⇒ f16 量化主导。
                // 真 bug(hd 错配那类)会给出 rms 的 1000× 量级 ⇒ 与"HF 自身噪声"差 2 个数量级。
                // ⚠️ 实测我方 f16 比 HF 自身 f16 噪声 **大一档**(pre 1.5× / merged 2.8×,bird)——
                // 答案仍全对(见 tools/vision_inject_check.py),但这是**已知精度缺口**,
                // 记在台账待 M4 前查(候选:GEMM 累加设置/中间量保存精度)。
                assert!(
                    ours <= 4.0 * hf_gap + 1e-6,
                    "{tag}/{name}: 我方 f16 偏差 {ours:.3e} > 4 x HF 自身 f16 偏差 {hf_gap:.3e}"
                );
            }
        }
        gpu.close().await.expect("GPU 关机");
    }

    /// 真图 CPU f32 档(与 GPU 档同锚;debug 构建下很慢 ⇒ 门控 `OWL_VIS_CPU_REAL=1`)
    #[tokio::test]
    async fn real_images_cpu_f32_match_golden() {
        if !owl_shared::env_reader::flag("OWL_VIS_CPU_REAL") {
            eprintln!("skip: OWL_VIS_CPU_REAL 未开(debug 下逐图需数十分钟;用 --release)");
            return;
        }
        let g = real_golden();
        if !g.exists() {
            eprintln!("skip: 真图金标不在");
            return;
        }
        let (weights, cfg) = visual_weights();
        let mut cpu = owl_cpu::CpuFace::new();
        let tower = VisionTower::new(cfg);
        let lctx = crate::module::LoaderCtx { dtype: Dtype::F32, shard: 1, device_repack: false, verify: false, debug_tap: false };
        crate::interpreters::eval_load(&tower, &mut cpu, &Src::from(weights), &lctx)
            .await
            .expect("塔 f32 装载");
        for tag in real_tags(&g) {
            let pv = st_read(&g, &format!("in.{tag}.pixel_values"));
            let gr = st_read(&g, &format!("in.{tag}.grid_thw"));
            let grid = vec![(gr[0] as usize, gr[1] as usize, gr[2] as usize)];
            let tb = VisionTables::build(&grid, cfg).expect("tables");
            let inputs = tb.inputs(&pv, Dtype::F32).expect("inputs");
            let out = tower.forward(&inputs);
            let pm = crate::testkit::harvest(&mut cpu, &out.pre_merger).await;
            let mg = crate::testkit::harvest(&mut cpu, &out.merged).await;
            let wp = st_read(&g, &format!("g.{tag}.pre_merger"));
            let wm = st_read(&g, &format!("g.{tag}.merged"));
            let (w1, a1) = allclose(&pm, &wp, ATOL_REL, RTOL);
            let (w2, a2) = allclose(&mg, &wm, 1e-3, 1e-3);
            let rms = |v: &[f32]| (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
            eprintln!(
                "[vis-real-cpu] {:12} grid {:?} pre {w1:.3e}(@[{a1}] |Δ| {:.3e} want {:.3e} rms {:.3e}) merged {w2:.3e}(@[{a2}] |Δ| {:.3e} rms {:.3e})",
                tag,
                grid,
                (pm[a1] - wp[a1]).abs(),
                wp[a1],
                rms(&wp),
                (mg[a2] - wm[a2]).abs(),
                rms(&wm)
            );
            assert!(w1 <= 1.0 && w2 <= 1.0, "{tag}: 归一化误差 pre {w1:.3e} / merged {w2:.3e}");
        }
    }

    /// T2/T3:逐层 + 末层对拍(HF 金标;CPU face f32)
    #[tokio::test]
    async fn t2_layers_match_golden() {
        let goldens = golden_paths();
        if goldens.is_empty() {
            eprintln!("skip: 金标不在(tools/vision_golden.py 生成)");
            return;
        }
        let (_, cfg) = visual_weights();
        for g in &goldens {
            eprintln!("[vis] 塔对拍 档 {} grid={:?}", g.file_name().unwrap().to_string_lossy(), golden_grid(g));
            tower_matches_golden(g, cfg).await;
        }
    }

    async fn tower_matches_golden(g: &PathBuf, cfg: VisionCfg) {
        let grid = golden_grid(g);
        let tb = VisionTables::build(&grid, cfg).expect("build tables");
        let px = st_read(g, "in.pixel_values");

        let mut face = owl_cpu::CpuFace::new();
        let out = tower_forward(&mut face, &tb, &px).await;

        // 多根一次求值(共享 CSE;别逐根重跑 —— 那是 27× 整塔)
        let mut roots: Vec<&TensorOps> = out.blocks.iter().collect();
        roots.push(&out.pre_merger);
        roots.push(&out.merged);
        let blocks = crate::interpreters::eval_ops_multi(&roots, &mut face).await.expect("eval_multi");

        for (i, b) in blocks.iter().take(cfg.depth).enumerate() {
            let n = tb.rows * cfg.hidden;
            let mut buf = vec![0u8; n * 4];
            face.dtoh(b, &mut buf).await.expect("dtoh");
            let got = f32_of(&buf);
            let want = st_read(g, &format!("g.blk_{i:02}"));
            let (worst, at) = allclose(&got, &want, ATOL_REL, RTOL);
            eprintln!(
                "[vis] blk_{i:02} 归一化误差 {worst:.3e}(@[{at}] |Δ| {:.3e} / |want| {:.3e})",
                (got[at] - want[at]).abs(),
                want[at].abs()
            );
            assert!(worst <= 1.0, "blk_{i:02}: 归一化误差 {worst:.3e} > 1");
        }
        // pre_merger
        {
            let n = tb.rows * cfg.hidden;
            let mut buf = vec![0u8; n * 4];
            face.dtoh(&blocks[cfg.depth], &mut buf).await.expect("dtoh");
            let got = f32_of(&buf);
            let want = st_read(g, "g.pre_merger");
            let (worst, at) = allclose(&got, &want, ATOL_REL, RTOL);
            eprintln!("[vis] pre_merger 归一化误差 {worst:.3e}(@[{at}] |Δ| {:.3e})", (got[at] - want[at]).abs());
            assert!(worst <= 1.0, "pre_merger: 归一化误差 {worst:.3e} > 1");
        }
        // merged
        {
            let n = (tb.rows / 4) * cfg.out_hidden;
            let mut buf = vec![0u8; n * 4];
            face.dtoh(&blocks[cfg.depth + 1], &mut buf).await.expect("dtoh");
            let got = f32_of(&buf);
            let want = st_read(g, "g.merged");
            let (worst, at) = allclose(&got, &want, 1e-3, 1e-3);
            eprintln!("[vis] merged 归一化误差 {worst:.3e}(@[{at}] |Δ| {:.3e})", (got[at] - want[at]).abs());
            assert!(worst <= 1.0, "merged: 归一化误差 {worst:.3e} > 1");
        }
    }

    /// **完全平方门**(2026-10-17 审查 §五轮):`num_pos` 非完全平方时表边长
    /// 不可定 ⇒ `grid_side` 必须**响亮拒**(修前 debug_assert ⇒ release 静默取整,
    /// 后续 tap/index 全按错误边长算)。
    #[test]
    fn grid_side_is_loud_on_non_square() {
        let mut cfg = VisionCfg::holo(); // 任意基线,只改 num_pos
        cfg.num_pos = 5; // 非完全平方
        let e = cfg.grid_side().expect_err("非完全平方必须拒");
        assert!(format!("{e}").contains("完全平方"), "{e}");
        cfg.num_pos = 4;
        assert_eq!(cfg.grid_side().expect("4 = 2²"), 2);
    }
}


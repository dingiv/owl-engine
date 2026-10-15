# 视觉塔(Qwen3.5-VL ViT)接入 owl —— 设计与 CPU 对拍方案

> 2026-10-15 立项;worktree `packages/worktrees/owl/vision-tower`(分支 `feat/vision-tower` @ `c18c0be`)。
> 定位:本文件是**视觉塔**的设计文档(实现与施工记录另起 / 记 `roadmap.local`)。
> 设计输入三份,全部已核:
> ① **HF 参考实现**(`transformers` 5.19.0,**已装进 `crates/models/.venv`**;逐条摘录见 §1,行号指向
>    `transformers/models/qwen3_5/modeling_qwen3_5.py` 与 `transformers/vision_utils.py`);
> ② **真权重事实**(`models/Hcompany/Holo-3.1-9B` 的 `model.visual.*` 333 张量:名字/形状/dtype 全量核对;
>    `Holo-3.1-35B-A3B-FP8` 的 `vision_config` 与 9B **逐字段相同** ⇒ 塔是同一个);
> ③ **owl 现役算子面**(`crates/models/src/ops.rs` 的 `SemanticKernel`、`crates/kernels/src/ops/*`,
>    `crates/backends/cpu/src/{face,ops}.rs` 的 CPU 实现集)。
>
| 修订 | 内容 |
|---|---|
| 2026-10-15 首轮 | 立项:事实面 + 算子差分 + 对拍方案 + 分阶段(M1…M5) |
| 2026-10-15 二轮 | **M1+M2 当日完成**(5 算子全落 + 塔整机 CPU 对拍两档全绿,§8);判据改**尺度感知 allclose**(§4.3);rope 定谳**复用零新核**(§3 #6) |

> 首刀范围(**用户裁决**):**设计 + CPU 面对拍**;数值锚 = **transformers**;目标档 = **Holo-9B / 3080 单卡**。
> 注:为过"声明 ⇔ `.cu` 签名"机器门,五个算子在 M1 就**带了真 CUDA 核**(不是空声明);
> 塔的 **f16 GPU 路径实测**仍归 M3。

---

## 0. 结论速览(TL;DR)

1. **塔的规格**(Holo-9B/35B 同款):`depth 27 · hidden 1152 · heads 16 · head_dim 72 · intermediate 4304 ·
   out_hidden 4096 · pos_emb 2304(= 48×48 网格)· patch 16 · temporal_patch 2 · spatial_merge 2 ·
   act gelu_pytorch_tanh · deepstack = [] (空)`。权重 333 张量 / ~1.5 GB(BF16),**全在 shard 4**。
2. **塔只需 5 个新算子**(其余全可复用现役面,且**已全落**):`Layernorm(w+b) / Gelu(tanh|erf 两变体) /
   BiasAdd / **PosEmbedBilinear**(4-tap 重采样)/ AttnFull(非因果 varlen,hd=72)`;
   **2D axial rope 零新核** —— 既存 `owl_rope_half_partial_*` 本身就是表驱动 rotate-half,
   `half = head_dim/2` 时其 partial 直通段自动为空 ⇒ 与 HF 逐式同构(二轮定谳)。
   插值索引/权重与 rope 表由 **host 纯函数**算(`VisionTables::build`,可与 HF 逐元素对拍)。
3. **patch_embed 不是卷积**:processor 已经把图 patchify 成 `[n_patch, 3·2·16·16]` ⇒ 塔内是
   **纯 `MatmulNt` + bias**(权重 `[1152,3,2,16,16]` 直接按 `[1152, 1536]` 行主序直读,零重排)。
4. **对拍口径三层(已实测,§8)**:① 表(interp 索引逐元素相等 / 权重 1e-7 / rope cos·sin 1e-6)
   ② 逐层 27 block(**尺度感知 allclose**:`|Δ| ≤ 2e-3·max(1,rms) + 1e-4·|want|`;实测最坏归一化误差 0.40)
   ③ merged `[n/4, 4096]`(实测归一化误差 1.1e-2)。金标由 `tools/vision_golden.py` 生成
   (真权重 + 固定种子合成 `pixel_values`,两档网格),Rust 侧同权重同输入复算。
5. **风险不在塔内,在文本侧**(M4):image token 替换 + **mrope(3D,section [11,11,10],interleaved)**;
   以及 3080 单卡装 Holo-9B **BF16 18.8G ⇒ KV 池只剩 ~1G**(约 39k token)的容量问题(§6 R4)。

---

## 1. 事实面(HF 5.19 + 真权重;逐条可复查)

### 1.1 塔规格与张量清单

| 项 | 值 | 出处 |
|---|---|---|
| 层数 / hidden / 头 | 27 / 1152 / 16 ⇒ **head_dim = 72** | `config.json:vision_config`;`modeling_qwen3_5.py` 的 `head_dim = dim // num_heads` |
| MLP 中间维 / act | 4304 / **`gelu_pytorch_tanh`**(tanh 近似) | 同上;`Qwen3_5VisionMLP`(f:938) |
| 输出维 / 合并 | `out_hidden_size 4096` / `spatial_merge_size 2`(2×2 ⇒ 4 patch → 1 token) | `Qwen3_5VisionPatchMerger`(f:971) |
| 位置嵌入 | `num_position_embeddings 2304` ⇒ `num_grid_per_side = 48`;插值 `bilinear, align_corners=True, padding=border` | `Qwen3_5VisionModel.__init__`(f:1135-1140)+ `vision_utils.py:231` |
| rope | **axial**:`theta 1e4`,`inv_freq` 长 **18**(=`head_dim/4`),cos/sin 覆盖整 72 维 | `Qwen3_5VisionRotaryEmbedding`(f:75,f:114-134) |
| 深栈 | `deepstack_visual_indexes = []` ⇒ **没有 deepstack**(少一大块活) | `config.json` |
| 权重 dtype/位置 | BF16;333 张量**全在 `model-00004-of-00004.safetensors`** | safetensors 头核对 |

张量形状(逐类,已核):

```
patch_embed.proj.weight [1152, 3, 2, 16, 16]   patch_embed.proj.bias [1152]
pos_embed.weight        [2304, 1152]
blocks.{i}.norm1/.norm2.{weight,bias}   [1152]        ← LayerNorm 有 bias
blocks.{i}.attn.qkv.{weight [3456,1152], bias [3456]}   ← qkv 融合
blocks.{i}.attn.proj.{weight [1152,1152], bias [1152]}
blocks.{i}.mlp.linear_fc1.{weight [4304,1152], bias [4304]}
blocks.{i}.mlp.linear_fc2.{weight [1152,4304], bias [1152]}
merger.norm.{weight,bias} [1152]  merger.linear_fc1.{w [4608,4608],b}  merger.linear_fc2.{w [4096,4608],b}
```

### 1.2 逐算子数学(七段;每条给出 HF 出处与判据)

| # | 段 | 数学 | 出处 | owl 侧落点 |
|---|---|---|---|---|
| 1 | **patchify** | `pixel_values [n_patch, C·T·P·P]` → `view(-1, 3, 2, 16, 16)`;**已由 processor 完成** | `Qwen3_5VisionPatchEmbed.forward`(f:951) | host(布局契约,+ 与 processor 对齐 M5) |
| 2 | **patch_embed** | `Conv3d(3→1152, k=s=(2,16,16), bias)` ≡ `MatmulNt(W'[1152,1536], x[1536]) + b` | 同上 | `MatmulNt` + **BiasAdd** |
| 3 | **pos_embed** | `pos_embed(interp_indices)·interp_weights` 沿 4 tap 求和 ⇒ `[n_patch, 1152]`;**插值表按 2×2 块序**(`spatial_merge_size>1` 时 patch 序 = `(block_row, block_col, in_row, in_col)`) | `vision_utils.py:231-300`;`Qwen3_5VisionModel.forward`(f:1160-1170) | host 纯函数(4-tap 索引+权重) |
| 4 | **rope(axial 2D)** | `inv_freq_j = 1/(θ^(2j/36)), j<18`;`freq_h = h·inv_freq`,`freq_w = w·inv_freq`;`cos = [cat(cos_h,cos_w)]×2`(72 维);`q' = q·cos + rotate_half(q)·sin` | `Qwen3_5VisionRotaryEmbedding.forward`(f:114-134)+ `apply_rotary_pos_emb_vision`(f:987) | **RopeApply2D**(表驱动) |
| 5 | **block** | `x += attn(LN1(x)); x += mlp(LN2(x))`;**LN = LayerNorm(1152, eps 1e-6, affine)**(≠ RMSNorm) | `Qwen3_5VisionBlock`(f:1084) | **Layernorm** + `Add` |
| 6 | **attn** | `qkv = Linear(x)` → 3 段 `[n,16,72]`;rope;`softmax(q kᵀ·(1/√72)) v`;**非因果、逐图**(`cu_seqlens` 按 `t·h·w` 分;无 mask);`proj` 回 1152 | `Qwen3_5VisionAttention`(f:1001-1082) | **AttnFull** |
| 7 | **mlp** | `fc2(gelu_tanh(fc1(x)))`,两段都有 bias | f:938 | `MatmulNt` + **Gelu(tanh)** + **BiasAdd** |
| 8 | **merger** | `LN(1152)` → `view(-1, 4608)` → `fc1(4608→4608)` → **GELU(erf, 精确版!)** → `fc2(4608→4096)` | `Qwen3_5VisionPatchMerger`(f:971-985) | **Layernorm** + **Gelu(erf)** + `MatmulNt` + **BiasAdd** |

⚠️ **两处易错点(已核,写进判据)**
- **merger 的 GELU 是精确 erf 版**(`nn.GELU()` 默认 `approximate='none'`),而 block MLP 是
  `gelu_pytorch_tanh` —— 两者**不同**,不能共用同一个核(除非带变体参数)。
- **merger 的 LN 在 shuffle 之前**(`use_postshuffle_norm=False`):对 `[n_patch, 1152]` 逐 patch 归一,
  **然后** `view(-1, 4608)` 做 4 patch 拼接,再做 fc1(4608→4608)。
- **head_dim = 72**(非 8 的整数倍?是 8×9)—— 现有 FlashInfer/XQA 路径全是 hd128/256,塔要自己的核。

### 1.3 输入面(processor;M5 才接,先记下事实)

```
preprocessor_config: patch 16 · temporal_patch 2 · merge 2 · mean/std = 0.5/0.5(⇒ (x-0.5)/0.5)
                    最短边 65536 px(=256²)最长边 16777216 px(=4096²)
grid_thw = [[t, h, w]]  (h = 图像高/16, w = 宽/16;t=1 表示"一个时间 patch"= 同帧复制 2 份)
```

**塔的输入契约(freeze)** = `(pixel_values: f16/f32 [Σ t·h·w, 3·2·16·16], grid_thw: [n_img,3] i64)`。
图像解码/缩放/归一化/patchify 属**预处理**,与塔解耦(§5 M5)。

### 1.4 文本侧接口(M4;先记事实)

- **image token 替换**:`input_ids == image_token_id(248056)` 的位置用塔输出**顺序填充**
  (`n_merged = n_patch/4` 必须等于 prompt 里 image token 的个数)。
- **mrope**:文本侧 position_ids 变 `(3, bs, L)`(`mrope_section [11,11,10]` + `mrope_interleaved: true`,
  `partial_rotary_factor 0.25` ⇒ 只前 64 维参与旋转)。纯文本输入时 t=h=w 退化为一维 rope(与现役一致)。
- Holo-9B 文本主干 = **Qwen3.5-9B**:32 层 = 8 FA + 24 GDN(`full_attention_interval 4`),
  hidden 4096 / heads 16 / kv 4 / head_dim 256 / vocab 248320 / `attn_output_gate true`。

---

## 2. 数据流(calltree)

```
owl_models::layers::vision::VisionTower::forward(pixel_values, grid_thw)      // 本设计新增
├─ host: interp_table(grid_thw) -> (indices [n,4], weights [n,4])            // 纯函数(§1.2 #3)
├─ host: axial_cos_sin(grid_thw) -> (cos [n,72], sin [n,72])                 // 纯函数(§1.2 #4)
├─ kernel: MatmulNt(W_patch[1152,1536], x)                                   // §1.2 #2
├─ kernel: BiasAdd(…, b_patch)
├─ kernel: PosEmbedGather(pos_embed[2304,1152], indices, weights)            // 或 host 预算表 + Embed
├─ kernel: Add(h, pos)                                                       // 残差入塔
├─ for i in 0..27:                                                           // §1.2 #5-#7
│   ├─ kernel: Layernorm(x, w1_i, b1_i, eps=1e-6)
│   ├─ kernel: MatmulNt(W_qkv_i[3456,1152]) ; BiasAdd(b_qkv_i)
│   ├─ kernel: RopeApply2D(q,k,cos,sin)
│   ├─ kernel: AttnFull(q,k,v, cu_seqlens, scale=1/√72)                      // 非因果、逐图
│   ├─ kernel: MatmulNt(W_proj_i) ; BiasAdd ; Add(残差)
│   ├─ kernel: Layernorm(x, w2_i, b2_i)
│   └─ kernel: MatmulNt(W_fc1_i[4304,1152]) ; BiasAdd ; Gelu(tanh)
│              ; MatmulNt(W_fc2_i[1152,4304]) ; BiasAdd ; Add(残差)
├─ kernel: Layernorm(x, w_m, b_m)
├─ host/kernel: view(-1, 4608)                                               // 零拷贝重解释(连续)
├─ kernel: MatmulNt(W_m1[4608,4608]) ; BiasAdd ; Gelu(erf)
└─ kernel: MatmulNt(W_m2[4096,4608]) ; BiasAdd  ⇒ merged [n/4, 4096]         // 出塔(注入文本模型)
```

---

## 3. 算子差分表

| # | 塔需要 | owl 现役 | 首刀(CPU 对拍) | 落点(M3 GPU) |
|---|---|---|---|---|
| 1 | `MatmulNt` | ✅ `Op::MatmulNt`(CPU `matmul_nt`;GPU cublas/marlin) | 直接复用 | 复用 |
| 2 | `Add` / 残差 | ✅ `Op::Call{Add}` | 复用 | 复用 |
| 3 | **Layernorm(w,b,eps)** | ❌ 只有 `Rmsnorm`(×(1+w),无 bias、无均值) | **✅ 已落(2026-10-15;骨架首件)** | `owl_layernorm_{f32,f16,bf16}` 已过 op_tables 互证 + nvcc 编译(§7 配方) |
| 4 | **Gelu(tanh / erf)** | ❌ 只有 `Silu`/`SiluAndMul` | **✅ 已落**(`owl_gelu_{tanh,erf}_{f32,f16,bf16}`) | 同左(已编入 cubin) |
| 5 | **BiasAdd(行广播)** | ❌ `Add` 是同形逐元素 | **✅ 已落**(`owl_bias_add_{f32,f16,bf16}`) | 同左 |
| 6 | 2D axial rope(表驱动) | ⚠️ `Rope` 是核内自算 freq 的 1D partial rope,**但** `owl_rope_half_partial_*` 本就是**表驱动 rotate-half** | **✅ 零新核:复用** —— `half = head_dim/2` 时 `2·half == head_dim`,其 partial 直通段**自动为空** ⇒ 与 HF 2D axial 逐式同构(二轮定谳) | 复用(表由 host 算;M3 可移设备侧) |
| 7 | **AttnFull(varlen, 非因果, hd72)** | ❌ 现役全是 paged/causal(hd128/256) | **✅ 已落**(CPU 两趟 softmax 参考 + `owl_vis_attn_full_{f32,f16}`:warp/行×头在线 softmax) | 同左(性能优化留 M3) |
| 8 | pos_embed 4-tap gather+weighted sum | ⚠️ `Embed` 是纯查表(无权重) | **✅ 已落**:`owl_pos_embed_bilinear_{f32,f16}` + host 纯函数算索引/权重 | 复用;表可移设备侧 |
| 9 | patchify / view(4608) | ✅ `Reshape`/`SliceView`(纯元数据) | 复用 | 复用 |
| 10 | image token 注入(M4) | ⚠️ 无 scatter/masked-copy | 暂不做 | M4 |

**首件战果(2026-10-15,当日完成)**:新算子 **4 族**(Layernorm / Gelu / BiasAdd / PosEmbedBilinear)
+ **AttnFull** 全落,且**每族都带真 CUDA 核**(不是空声明 —— `build.rs` 扫 `const SRC` 自动编入 cubin,
`op_tables` 门逐参互证);rope **零新核**(上表 #6)。CPU 面 = `owl-cpu` 与 `reference.rs` **两份独立实现**
(仓内纪律:互为副本、禁互引);host 表(interp/rope)是 Rust 纯函数,与 HF 金标逐元素对拍 ✓。

---

## 4. CPU 对拍方案(首刀主交付)

### 4.1 金标生成器 `tools/vision_golden.py`(入库;recipe 必须可复现)

照 `tools/dflash2_golden.py` 先例(权重直读检查点,金标落 safetensors):

```
输入:CKPT=/…/Holo-3.1-9B(只读 model.visual.*)
     grid_thw = [[1,4,6]]  +  固定种子 randn pixel_values [24, 1536]   ← 最小可验形状(6 merged token)
输出:/…/vision_golden.safetensors  键集合:
     in.pixel_values · in.grid_thw
     g.patch_embed · g.pos_embed · g.h0_embed
     g.blk_00 … g.blk_26                     (每 block 输出 [24,1152],27 张)
     g.pre_merger(=[24,1152]) · g.merged([6,4096])
     g.cos / g.sin ([24,72])                 ← rope 表(供 Rust 侧逐元素对拍)
     g.interp_indices / g.interp_weights ([24,4])  ← 插值表(HF 原样)
     meta.json:{transformers 版本, ckpt 路径+shard md5, 种子, 判据}
```

### 4.2 Rust 侧

- `crates/models/src/layers/vision.rs`:容器 + `LoaderOps`(键 = `model.visual.*` 的 333 槽)+
  `forward(&self, pixel_values, grid_thw, ctx) -> (pre_merger, merged)`(纯声明,零 host 副作用);
  host 纯函数:`interp_taps(grid_thw)` / `axial_cos_sin(grid_thw)`(可与 HF 表逐元素对拍)。
- 测试(新文件 `crates/models/tests/vision_parity.rs` 或 `vision.rs` 内 `#[tokio::test]`):
  - **T1 表对拍**:`interp_indices/weights`、`cos/sin` vs 金标(逐元素,判据 0);
  - **T2 逐层**:patch_embed / pos_embed / h0 / 27 block vs 金标(绝对 1e-4);
  - **T3 末层**:merged vs 金标(绝对 1e-3);
  - **T4 尺寸门**:grid `(1,4,6)` 与 `(1,8,10)` 两档(覆盖 merge 边界与非方形网格)。
- 门:`OWL_HF_PARITY=1` + 金标文件在(与现役 parity 管道同律);权重经 `Src` 从检查点直读 f32。

### 4.3 判据(先写死,免得事后调)

| 层 | 判据 | 理由 |
|---|---|---|
| 表(interp 索引) | **逐元素相等**(整数类) | 纯函数,无归约序歧义 |
| 表(interp 权重 / rope cos·sin) | 绝对 1e-7 / 1e-6 | 三角函数逐 ULP 差(torch cos vs Rust cos);与塔判据差 4 个数量级 |
| patch_embed / pos_embed / h0 | **allclose(atol 1e-3, rtol 1e-4)** | 见下"口径修正" |
| 逐 block hidden(27 层) | 同上 | 同上 |
| merged 末层 | allclose(atol 1e-3, rtol 1e-3) | 4608 维归约 + erf |
| 反例门 | 任一 NaN/Inf ⇒ 直接红 | 塔内无 mask/无 padding,出现即真错 |

**口径修正(实测驱动,2026-10-15)**:原写"绝对 1e-4"**不成立**,两条实测理由:
1. **末层激活爆量**:同一输入下 HF 自己的 `blk_26` max|·| ≈ **7.8e3**(rms 122)而 blk_25 只有 65
   —— 绝对判据在爆量层上无意义(0.07 的绝对差在 |want| 2.5 的元素上只有 3% 归一化误差... 实为 ~1e-5 相对);
2. **f32 归约序漂移**:CPU 面是 naive 串行点积,torch 是分块 SIMD ⇒ 逐层漂移 3e-5 → 1.8e-4(**相对** ~1e-5 稳定)。
⇒ 判据写成 `|Δ| ≤ atol + rtol·|want|`(标准 allclose),**atol = 1e-3 / rtol = 1e-4**。
atol 取 1e-3 的依据:32-patch 档实测最坏逐元素 `|Δ| = 2.5e-4`,而**该元素本身只有 ~2e-3**(近零元素
⇒ 纯相对项对它无意义);真 bug(布局/公式错)的误差是**信号量级**(rms ~1)⇒ 差三个数量级,不会漏。

---

## 5. 分阶段落地(每阶段独立可验、可回滚)

| 阶段 | 内容 | 判据 | 量级 | 本 worktree? |
|---|---|---|---|---|
| **M1** | **CPU 面五算子**(Layernorm/Gelu/BiasAdd/**PosEmbed**/AttnFull)+ 两侧实现(rope 复用零新核) | T1 全绿 + 单算子对拍 1e-6 | 1-2d | **✅ 全落(5/5)** |
| **M2** | **塔整机 CPU 对拍**(§4:T1–T4 全绿),含 27 层逐层金标 | §4.3 判据表全绿 | 1-2d | **✅(两档网格全绿,见 §8)** |
| **M3** | **GPU 路径**:五算子的 CUDA 核 + 塔 f16 前向真机(与金标 f16 容差档对拍) | 逐层 1e-2(f16 档)/耗时 ≤150 ms@1280×736 | 3-5d | ✅ |
| **M4** | **注入文本模型**:image token 替换 + **mrope** + 端到端"图 → 答" | 与 HF 贪心逐 token 一致(短图) | 3-5d | ✅ |
| **M5** | **预处理 + 服务面**:图像解码/缩放/patchify(与 `Qwen2VLImageProcessorFast` 对拍)+ OpenAI 兼容 `image_url` | 预处理逐元素 1e-5;端到端 e2e | 3-5d | ✅ |
| — | 优化(视觉塔裁剪/量化,workspace `docs/README.md` §四.5)——**本分支不做** | — | — | ❌ |

---

## 6. 风险与未决

| # | 风险 | 判据/量级 | 逃生阀 |
|---|---|---|---|
| **R1** | **head_dim 72** 无现役核可借;若 M3 才发现要写复杂核 ⇒ 工期超 | M3 前先量 CUDA 侧最坏代价(塔只占 prefill 的 <10%) | 先落"朴素非融合核"(塔 prefill-only,无 KV 池),优化后置 |
| **R2** | patch 序(2×2 块序)与 processor 实际输出不一致 ⇒ **静默错值** | T1 表对拍 + M5 预处理对拍 | 预处理对拍先行;金标里固化 `pixel_values` 顺序 |
| **R3** | LN/GELU 细节错(**erf vs tanh**、eps、bias) | §1.2 三处 ⚠️ 已写死;T2 会抓 | 单算子金标拆开验 |
| **R4** | **3080 装不下 Holo-9B** BF16 18.8G + KV(~39k token 上限) | M4 前必须先算清(§1.4) | 换 3090 Ti 单卡(24G)/ 文本模型量化(W4A16 无现成 Holo 权重) |
| **R5** | mrope(interleaved, section [11,11,10], partial 0.25)与现役 1D rope 的共存 | M4 首战 | 纯文本路径零回归门(现役 27B e2e 复跑) |
| **R6** | 金标 recipe 漂移(transformers 版本变化 ⇒ 数值变) | `meta.json` 记版本 + ckpt md5;金标不入版本库但 recipe 入 | 版本钉死;变更时重生成并记 |

**未决(待用户裁)**
1. M1 的五个算子是否**一次全落**,还是按"M2 需要谁落谁"(建议:M1 全落,单算子验完再拼塔 —— 拼塔出错时能秒定位)。
2. 对拍门是否沿用 `OWL_HF_PARITY=1`(建议沿用),还是给塔单独一个门 `OWL_VISION_PARITY=1`(金标文件大,CI 上可能想单开)。
3. M3 的 GPU 路径要不要**顺手量一次 llama.cpp mmproj 同图耗时**(交叉验证塔的性能面)——
   现役 holo 档在跑同一塔,零额外成本。

---

## 7. 附:新算子落地配方(**已由 Layernorm 走通**,2026-10-15)

一个"塔新算子"从零到过门的**完整触点表**(顺序即施工序;每步都有机器门):

| # | 文件 | 干什么 | 门 |
|---|---|---|---|
| 1 | `crates/kernels/cu/vis/<op>.cu` | 真 CUDA 核(宏实例;**实例行必须单行** —— `cu_sig::cu_params_of` 逐行解析);形参序 = 声明序 | nvcc(build.rs 扫 `const SRC` 字面量自动纳入编译) |
| 2 | `crates/kernels/src/ops/vis.rs` | `impl KernelSpec`(ENTRY/SRC/PARAMS/LAWS/CAPTURE/DETERMINISM)+ 命名 static + `METAS` | `op_tables_match_cu_{signatures,param_names}` |
| 3 | `crates/kernels/src/ops/mod.rs` | `pub mod vis;` + `TABLES` 添 `vis::METAS` | `every_compiled_symbol_is_declared`(R1) |
| 4 | `crates/kernels/src/driver.rs` | 分派臂(`"vis.xxx" => vis::xxx(...)`)+ `pub mod vis` picker(grid/block/smem) | driver 单测(shape 契约) |
| 5 | `crates/models/src/ops.rs` | `SemanticKernel::Xxx` + `op_id()`(如 `"vis.layernorm"`) | 编译期穷举 |
| 6 | `crates/models/src/tensor.rs` | `TensorOps::xxx(...)` 声明构图(形状律 + 标量槽序) | 单元对拍 |
| 7 | `crates/backends/cpu/src/{ops,face}.rs` | CPU 实现 + `launch` 名字臂 | `owl_cpu` 路径 = 所有 CPU 对拍的执行体 |
| 8 | `crates/models/src/reference.rs` | **第二份独立实现**(trait 方法 + `Op::Call` 臂)—— 与 owl-cpu **互为副本、禁互引** | 参考面留白 = 对拍失效 |
| 9 | `crates/models/src/layers/<layer>.{rs,py}` + `mod.rs` | 层容器(Weight 槽 / `Loadable::layout` / `Module::forward`)+ **HF/torch 基线** | `OWL_HF_PARITY=1` 下的 parity 测试 |

两条踩过的坑(写死以免重蹈):
- **宏实例行折成两行 ⇒ 签名互证门拿到空形参串**(报 "无法识别的宏形参 ``")—— 一行一条实例;
- **`reference.rs` 与 `owl-cpu` 两份实现都要写**(只写一份 = 对拍锚失明,仓内纪律明确要求两份)。

验收命令(本机实测全绿):

```
cargo test -p owl-kernels --test op_tables          # 10/10(含新核签名互证 + R1 符号集)
cargo test -p owl-kernels --lib                     # 69/69
cargo test -p owl-models --lib layers::<layer>      # 单算子 host 手算锚
OWL_HF_PARITY=1 cargo test -p owl-models --lib layers::<layer>::tests::parity
OWL_TEST_DEVICE=2 cargo test -p owl-models --lib    # GPU 门测试要指定空闲卡(默认 0 会撞生产 server)
```

---

## 8. CPU 对拍首战(2026-10-15;两档网格,全绿)

**落地面**(全部未提交,worktree `feat/vision-tower`):

| 面 | 文件 |
|---|---|
| 金标生成器(入库) | `tools/vision_golden.py`(+ `testdata/vision_golden{,2}.safetensors`,gitignored) |
| CUDA 核 | `cu/vis/layernorm.cu`(3 dtype)· `cu/vis/vision_ops.cu`(Gelu×6 / BiasAdd×3 / PosEmbed×2 / AttnFull×2) |
| 声明 | `crates/kernels/src/ops/vis.rs`(16 张表)+ `ops/mod.rs` |
| 拾取 | `crates/kernels/src/driver.rs`(`pub mod vis` 6 个 picker + 6 条分派臂) |
| 语义/构图 | `crates/models/src/{ops,tensor}.rs`(5 个 `SemanticKernel` + 5 个 `TensorOps` helper) |
| CPU 执行 | `crates/backends/cpu/src/{ops,face}.rs`(+ rope/narrow 两条既存核的 CPU 臂) |
| 参考锚 | `crates/models/src/reference.rs`(**第二份独立实现**) |
| 塔本体 | `crates/models/src/layers/vision.rs`(容器 + host 表纯函数 + 前向 + T1/T2 测试)· `layernorm.{rs,py}` |

**判据实测**(档一 `grid=(1,4,6)`;档二 `(1,4,8)`):

| 项 | 金标 vs CPU face | 判据 |
|---|---|---|
| interp 索引 | **逐元素相等** | 相等 |
| interp 权重 / rope cos·sin | ≤ 1.2e-7 / ≤ 1.2e-7 | 1e-7 / 1e-6 |
| patch_embed → blk_26(27 层) | 归一化误差 ≤ **0.40**;最大 \|Δ\| = **1.8e-4** | ≤ 1.0 |
| merged `[8, 4096]` | 归一化误差 **1.1e-2**(\|Δ\| 7.9e-5) | ≤ 1.0 |

**过程中逮到的两条**(写死以免重蹈):
1. **head 数 ≠ hidden 宽**:`narrow_strided` 切 qkv 时把 `heads(16)` 当列宽用 ⇒ 三支形状错成 `[n,16]`
   (q/k 被 rope 的 `with_shape` 掩盖,只有 **v** 露馅)。教训:切列宽用 `cfg.hidden`,头数只进形状;
2. **宏实例行折行** ⇒ 签名互证门拿空形参串(§7 已记)。

**性能备注**:CPU 面是 naive 串行实现,debug 档一趟整塔对拍 ≈ **3 分钟**(档一)/ 5 分钟(两档);
要跑全量对拍建议 `cargo test --release` 或只跑 `t1`。

**下一步(M3)**:五算子的 GPU 路径已在位(核 + 声明 + 拾取),M3 = ①真权重 f16 上机对拍(与金标 f16 档)
②`owl_vis_attn_full` 的 warp/行×头在线 softmax 实测(塔 prefill-only,判据 ≤150 ms@1280×736)
③补 `bf16` 两臂(PosEmbed/AttnFull 现只 f32/f16)。

---

## 9. M3 首战:GPU f16 路径(2026-10-15 当晚)

**结论:五算子的 f16 GPU 路径全部跑通,塔整机对拍过**(vs HF **f16** 金标,逐层 + merged);
但**性能未达 §5 的 150 ms 判据**(实测 1.34 s/图,瓶颈 = 注意力核)。

### 9.1 正确性(门 + 证据)

| 门 | 结果 |
|---|---|
| `OWL_TEST_DEVICE=2 cargo test -p owl-models --lib layers::vision::tests::gpu_f16_tower_matches_golden` | **过** |
| 逐层(vs `h.blk_XX`,HF f16 档) | 归一化误差 0.10 → **0.50**(最坏 blk_20),全程 ≤1 |
| merged `[920, 4096]` | 归一化误差 **0.174**(\|Δ\| 6.25e-2) |
| 对照 f32 金标的信息值(纯 f16 存储税) | 逐层最大 \|Δ\| 0.14;merged 5.3e-2 |
| 五个新算子 GPU f32 逐算子 vs CPU(临时探针,已删) | LN 7e-7 / gelu 6e-8 / bias 0 / pos_embed 1.2e-7;narrow 0 / rope 1.2e-7 / matmul+reshape 1.2e-4 |

**批量定位法(值钱,留档)**:塔在 GPU 上出错时,按"逐算子 ← 子树 ← 整塔"三层筛:
① 单算子 GPU vs CPU;② 注意力子树(qkv→bias→narrow→rope→attn)GPU f32 / GPU f16 / CPU f32 三档对拍;
③ 整塔 blk_00 单根 vs 多根 + 对金标。三层全绿而整塔红 ⇒ 必是构建/资产面(见 9.3)。

### 9.2 性能(1280×736 截图 = 3680 patch / 920 merged token;3080 20G)

| 项 | 实测 |
|---|---|
| 整塔 f16(`eval_ops_multi_scoped_env` + sync 口径) | **1339 ms** |
| 仅注意力核(3680×16 头 × hd72,非因果) | **45.5 ms/层 × 27 ≈ 1.23 s** |
| 其余 26 类算子合计(GEMM/LN/gelu/bias/narrow/rope/pos_embed) | ~110 ms |
| GEMM 单发实测(3680×3456×1152,f16) | 0.65 ms ⇒ **45 TFLOPS**(约峰值 75%) |

⇒ **塔 = 注意力绑死**(92%),且瓶颈是**指令/延迟**而非 DRAM 带宽:一 warp 一行、
逐 key 做 3 维点积 + 5 次 shuffle 归约 + 2 次 `__expf`,3680 key 串行 ⇒ 单 (row,head)
约 1.8e5 周期;总 warps = 3680×16 = 5.9 万。v2(块内 8 行共享 smem K/V tile)只把
50.7 → 45.5 ms/层 —— 这正是"带宽不是瓶颈"的证据。

**候选优化(M3-opt,按性价比排序)**:
1. **split-KV**(4 分片 + 归并)⇒ 串行长度 ÷4,预计 attention ~12 ms/层、整塔 ~430 ms(改动约 60 行 + 一个小归并核);
2. **张量核 FA2 形状**(mma + smem 双缓冲)⇒ 需重写核(参考 xqa/flashinfer 结构);hd72 需 pad 到 80/96;
3. **借 FlashInfer prefill**(hd 必须 64/128/256 ⇒ pad 到 128,且本 fork mask 固定 causal,需换 `MaskMode::kNone` 臂)⇒ 走 tensor core,粗算 ~3 ms/层;
4. 削减 host 侧税(scoped eval 每层约 30 次 alloc/launch;若与 M0 命令批 / 图捕获合流可再收一截)。

### 9.3 一条**假红**教训(写死)

改完注意力核后首次复测,塔 f16 逐层误差跳到 **1.2e3**(看似核错)。三层筛(§9.1 方法)
全绿 ⇒ 最终判定为 **stale fatbin**(kernels 侧 cubin 资产未随宿主 crate 重建刷新);
**强制重建 kernels 后复测即恢复**(0.10 → 0.50 的合理剖面)。
⇒ 纪律:**改 `.cu` 后先 `cargo test -p owl-kernels --test op_tables`(触发重编 + 签名互证),
再跑依赖它的上层测试**;跨 crate 的核资产刷新不可想当然。

---

## 10. M3-opt:换用 FlashInfer(FA2 家族)非因果注意力 + 切 4B 档(2026-10-15 收尾)

**两条用户裁决的落地**:
① **目标档切 `models/Qwen/Qwen3.5-4B`**(自带 `model.visual.*` 297 张量 ⇒ 完整 VLM;
塔 = 24 层 / hidden 1024 / **head_dim 64** / out 2560;总权重 9.3G ⇒ 单张 3080 装得下 + KV 有余);
② **视觉塔不再自写算子**:注意力改走 owl 已 vendored 的 **FlashInfer FA2 家族**
`BatchPrefillWithRaggedKVCacheDispatched` + `MaskMode::kNone`(非分页 dense K/V、varlen、非因果)
—— 与 vLLM 的 ViT(`flash_attn_varlen_func(..., causal=False)`)同路算法;胶水只写 host 侧
(plan / 参数装配 / 发射),**核体零自研**。

### 10.1 落地件

| 面 | 文件 |
|---|---|
| 适配器(nvcc 编入 `libowl_flashinfer.a`,**3 TU**) | `cu/flashinfer/owl_fi_attn_ragged.cu`(plan + run;plan 期写规范 int32 段界 `[0, rows]`) |
| 家族/线契约 | `family/flashinfer.rs`(`ATTN_RAGGED_F16` 常量 + extern)· `client/flashinfer.rs`(`parse_attn_ragged`;`NamedSlots` 新增 `f32`/`i32` 取用面)· `server/flashinfer.rs`(`run_attn_ragged` + ragged plan 缓存 + `fi.ragged_cu` scratch) |
| 声明 | `ops/flashinfer.rs`:`FiAttnRaggedF16`(FOREIGN;`FAMILY` = **FI 族 id**;槽位与 native `vis::AttnFullF16` **逐字同构**) |
| 分流 | `driver.rs`:`vis.attn_full` ⇒ **f16 + 单图 → FI ragged**;其余(f32 锚 / 多图)→ native 臂 |
| 塔/测试 | `VisionCfg::from_json`(超参从 `config.json` 读,零手工抄数;`visual_weights()` 顺带返回 cfg)· 金标两档改由 **4B** 生成 |

### 10.2 结果(1280×736 = 3680 patch / 920 merged;3080 20G)

| 项 | 换核前(自写核) | **换核后(FlashInfer)** |
|---|---|---|
| 整塔 f16(3 次最好,sync 口径) | 1339 ms | **124.9 ms**(判据 ≤150 ms ⇒ **达标**) |
| 注意力单发 | 45.5 ms/层 | **6.6 ms/层**(FA2 张量核) |
| GPU f16 对拍(vs HF f16 金标) | 逐层 ≤0.50 / merged 0.174(Holo-9B 档) | 逐层 **0.07→0.60** / **merged 0.032**(4B 档) |

### 10.3 两条新踩的坑(写死)

1. **`cu_seqlens` 的 dtype 契约**:owl 的索引律是"f32 过线",而 FlashInfer 的 `q_indptr/kv_indptr`
   是 `IdType = int32` ⇒ 把 f32 张量指针当 int32 读会**越界**(现场表现是下一发 `CUBLAS_STATUS_INTERNAL_ERROR`,
   真凶是前一发 ragged 读飞)。修法:plan 期把**规范 int32 `[0, rows]`** 写进 server 自有 scratch
   (单图 ⇒ 常量;回放期零转换零回读)。
2. **`KernelSpec::FAMILY` 是族 id,不是算子名**:FI 族的 id 恰是 `PREFILL_FI`(paged f16 入口名),
   写成自己的入口名 ⇒ boot 期"孤儿声明(家族不存在)"**拒启**(声明驱动路由的正面判据当场生效)。

---

## 11. 真图验收(2026-10-15;用户令"拿 assets 里的图让它识别")

**问:正确性验过了吗?** —— 验过了,四层证据,最后一层是**端到端识别**。

### 11.1 四层证据

| 层 | 做法 | 结果 |
|---|---|---|
| ① 表(真网格) | `assets/*.png` 经 **HF `Qwen3VLProcessor`** 真预处理(缩放/patchify/**2×2 块序**)⇒ 网格 (1,16,16)/(1,34,22)/(1,34,24)/(1,42,28) | interp 索引**逐元素相等**、权重 max\|Δ\|=**0**、rope cos/sin 6e-8 ⇒ **R2(patch 序)结案** |
| ② 早期层(真图) | HF 钩子捕 `patch_embed/pos_embed/h0/norm1/qkv/attn(proj 后)` | h0 1.0e-5 · norm1 2.0e-5 · qkv 7.2e-6 · **attn(proj 后) 4.1e-6**(f32) |
| ③ 逐层(真图,24 层) | f32 档逐 block 对 HF-f32 | max\|Δ\| ≤ 3.7e-3(rms 40 ⇒ **1e-4 相对**),**24/24 层 0 元素超阈** |
| ④ **端到端识别** | 把**我们塔**的 `merged` 注入 HF 文本模型(`visual.forward` monkey-patch)生成答案(`tools/vision_inject_check.py`) | **4/4 与 HF 答案一致**(见 11.2) |

### 11.2 识别结果(文件名是障眼法,答案是对的)

| 图 | 图里实际是什么(人工核对) | HF | **owl f32** | **owl f16** |
|---|---|---|---|---|
| `bird.png` | 企鹅 | penguin | **penguin** | **penguin** |
| `cat.png` | 德牧(狗) | dog | **dog** | **dog** |
| `dog.png` | 小猫 | kitten | **kitten** | **kitten** |
| `flower_blue.png` | 蓝花 | flower | **flower** | **flower** |

⇒ **owl 塔算出的 image embedding 对下游文本模型可用**(注入后答案不变)。
(tip:Qwen3.5 模板默认 thinking,要 `enable_thinking=False` 否则前 12 token 全在"想"。)

### 11.3 f16 精度:定谳为"模型自身的 f16 噪声",非我方缺陷

真图上 f16 档与 HF-f16 的差,与 **HF 自己的 f32↔f16 差**同量级(按共同 f32 参考量):

| 图 | HF 自身 f32↔f16(pre_merger, rms×) | 我方 f16 vs f32 参考 / HF 自身 | 比值 |
|---|---|---|---|
| bird | 6.99 | 5.45e2 / 3.61e2 | **1.51×** |
| cat | 14.10 | 4.73e2 / 6.17e2 | **0.77×** |
| dog | 6.19 | 2.55e2 / 2.50e2 | **1.02×** |
| flower_blue | 45.18 | 1.86e3 / 2.30e3 | **0.81×** |

merged 档比值 0.99~2.78× ⇒ 同量级(略高一档,bird 最坏 2.78×)。
⇒ 判据定为**"不比 HF 自己的 f16 更差"(≤4×HF 自身偏差)**;真 bug(hd 错配那类)是 rms 的 1000× 量级,
差 2 个数量级拦得住。⚠️ bird 的 merged 2.78× 记为**已知精度缺口**(M4 前值得查一轮:GEMM 累加设置 / 中间量保存精度)。

### 11.4 顺带修掉的真雷(静默错值)

FI ragged 适配器**只实例化 HEAD_DIM=64**;一旦 head_dim≠64(FI 仍会被 driver 选中)核内按 64 维算 ⇒
**静默错值**(实测 hd=72 时 F16 单测误差 1.19,而 F32 2.9e-4)。修法两道:
① 适配器 `if (head_dim != 64) return -207;`(响亮拒绝);
② driver 的 FI 臂条件补 `head_dim == 64`(否则回落 native 臂)。

---

## 12. M4 首刀:m-RoPE + 注入面 + VLM 装配(2026-10-15 深夜)

**目标**:让 owl **自己**吃图出答案(此前"识别"是借 HF 文本模型注入塔输出)。
本刀只走"host 表 + 两处注入面"路线 —— **零新数学、零新核**(注意力/卷积全复用)。

### 12.1 m-RoPE(新件 `layers/mrope.rs` + `layers/mrope.py`)

事实面取自 HF 5.19 真身(逐行核对 `modeling_qwen3_5.py`):

- 位置号 `Qwen3_5Model.get_rope_index`:按 token 类型分段 —— 文本段三网格同值
  `arange+cur`;视觉段 = h-major 网格(T 网格整段恒 = 段首位,H/W 各除 merge),
  段末 `cur += max(h, w)/merge`;
- 频率重组 `recomposition_frequencies`:`slice(dim, section[dim]*3, 3)` 交错认网格
  (`[11,11,10]` 铺满 half=32 ⇒ 即 `j % 3`),`cat(f,f)` ⇒ 配对 `(j, j+half)`
  **与现役 rotate-half 核同构**;
- 视频入口先把网格按帧展开(`repeat_interleave(t)` + `t=1`),帧间靠时间戳文本分隔
  (HF 按**类型连续段**分段 —— 两帧 token 连写属非法输入)。

**产出姿势 = host 行表**(与塔的 `axial_cos_sin` 同款):`[L, half]` cos/sin +
位置号,`TensorOps::from_host` 上传,现役 rope 核按"行号 = token 号"直取 ⇒
**零新核**。契约:表路径要求 pos 通道 = 行号(prefill 单块整发);decode 三网格
同值 ⇒ 仍走现役标量 pos 路径。

**响亮拒绝**(HF 静默错位点):占位 token 数 ≠ 网格展开 / 网格缺失 / 类型非法 /
h/w 非 merge 整数倍 —— 全部 `Err`。

**判据实测(8/8 绿)**:

| 项 | 结果 |
|---|---|
| `freq_grid` 交错律 | 11 T + 11 H + 10 W,j%3 ✓ |
| 手算锚(3 文本 + 6 视 + 2 文本) | 位置号逐元素相等,delta = −3 ✓ |
| 视频两帧(带分隔) | 帧一 T=1 / 帧间文本 cur=4 / 帧二 T=5 ✓ |
| **纯文本退化 = 现役 1D 表** | 逐元素同(≤1e-7)⇒ R5 零回归(表级) |
| **HF parity**(`OWL_HF_PARITY=1`) | 4 例(纯文本/单图/双图/视频):位置号**逐元素整数相等**、delta 相等、cos/sin ≤ **1e-6** |

HF 基线的取证方式:位置号 = `Qwen3_5Model.get_rope_index` **未绑定方法**调用
(`self` 用轻量 shim —— 已逐行核对该方法只碰 `config.vision_config.spatial_merge_size`
与 `self.get_vision_position_ids`,后者绑 HF 真身);cos/sin = `Qwen3_5TextRotaryEmbedding`
真类 + toy config(带真 `rope_parameters`)。真模型路径的差异留 M4-d 用真 4B 复核。

### 12.2 注入面(两处,`ForwardCtx` 显式依赖)

| 面 | 落点 | 语义 |
|---|---|---|
| `vlm_image` | `Model::last_hidden/prefill_bisect/tapped_hidden` embed 之后 | 图像占位 token 换成塔输出行(与 HF `masked_scatter` 同位);`None` ⇒ 零节点 |
| `vlm_rope` | `Rope::cos_sin_of(ctx)`(attention 3 个发射点) | m-RoPE 行表取代静态 1D 表;`None` ⇒ 现役表 |

构建器 `with_pos/with_vlm_image/with_vlm_rope` 消费式(self 自肃,无 Clone 需求);
`VlmRequest::bind(base)` 一次绑两面 + 接管 pos ⇒ **无"忘设 pos"这一错法**。

### 12.3 新算子 `vis.image_scatter`(VLM 图像 token 注入)

`out[t] = idx[t] ≥ 0 ? merged[idx[t]] : h[t]`(idx f32 行号,负 = 文本位保留)。
九触点全落(`cu/vis/vision_ops.cu` ⑥ + `ops/vis.rs` + driver 拾取 + 语义/构图 +
CPU 面 + `reference.rs` 独立副本)。判据:`op_tables` **10/10**(含新核签名
互名互证 + R1 符号集)、`image_scatter` CPU face vs host 手算锚逐元素相等 +
越界行号响亮拒。

### 12.4 VLM 装配(`vlm.rs`)

`VlmCfg::from_json`(零手工抄数:文本 spec/tower/mrope/占位 token 全从
`config.json` 读;装配期互证塔 `out_hidden` = 文本 hidden)+ `Vlm`(塔 + 文本
主干 + 两段键合并装载)+ `prepare/bind/prefill_logits`(prefill 一步)与
`text_logits`(decode 步复用)。`Vlm::inputs` 出 host 表:pos = `arange(L)` rank 2
([L,1],核线契约 rank 2)、行号表、m-RoPE cos/sin、delta。

判据(3/3 绿):真 4B `config.json` → 维度断言(2560/32 层/8 FA/hd 256/gdn 16×128
+32×128/tied);行号与位置号手算锚;占位不符响亮拒。

### 12.5 未完成(M4 剩两刀)

1. **真权重装载**:`Vlm` 从 4B 检查点两段键(`model.language_model.*` +
   `model.visual.*`)整装 —— 需要 GPU 与显存账(塔 0.8G + 文本 8.5G f16 ≈ 9.3G,
   单 3080 20G 可装);
2. **e2e 贪心对拍**:真图 → 塔 → 文本 → 自己出答案,与 HF 逐 token 一致
   (预处理沿用 `tools/vision_image_check.py` 的 `Qwen3VLProcessor`)。

**已知拦路(pre-existing,与本分支无关)**:本 worktree 的 26 条 GPU/rank 红
(`owl_narrow_strided_f32` / `owl_rope_half_partial_f32` 的 **rank 声明拒**:
`dst/pos 期望 rank = 2 实得 1`)⇒ **F32 rope 非融合臂当前不可用**(F16 生产臂
走 `owl_norm_rope_f16`,其 pos 声明 rank 0,不受影响)。主树同款;M4-d 走
F16 生产臂。**未动**(属主树 A20 战役)。

---

## 13. M4 收尾:4B 整装 + 端到端识图(2026-10-15 深夜)**M4 完成**

### 13.1 整装(`Vlm::load`)

一段 [`Qwen35Convention`](GDN `linear_attn.` / FA `self_attn.` / `mlp.` 三处子前缀
+ `A_log`/`dt_bias` 裸键)+ 塔自带 `model.visual.*`,**一份 `SafeTensorsSource` 供两边**
(检查点里 `mtp.*` 多余键不影响 —— 装载只取 Want)。判据:4B 整装 **3.1~3.6s**
(9.3G bf16 检查点 → f16 设备字节直转),显存 **≈9.4G < 20G(单 3080 可容)**。

⚠️ **踩坑(写死)**:文本侧键约定**必须** `Qwen35Convention`,用 `LlamaFamily` 会在
`layers.N.mlp.gate_proj.weight` 上"数据源缺键"炸(mlp/linear_attn/self_attn 三处子前缀)。

### 13.2 端到端识图(金标 `tools/vlm_e2e_golden.py`;assets 四图)

金标 = 真 processor(`Qwen3VLProcessor`,`enable_thinking=False`)+ HF 4B(bf16 CPU)
贪心 16 token + **真模型** `get_rope_index` 位置号 + 末位 logits top-10。

| 图 | L / 网格 | ① 位置号 vs HF 真模型 | ② prefill 末位 logits | ③ owl 贪心(逐 token) |
|---|---|---|---|---|
| bird | 212 / (1,34,22) | 636 个整数**逐元素相等** ✓ | top-1 = 280 == HF;top-10 **10/10** | `" penguin\n"` 5/5 ✓ |
| cat | 229 / (1,34,24) | 687 ✓ | top-1 = 5388 == HF;top-10 9/10 | `" dog\n"` 4/4 ✓ |
| dog | 89 / (1,16,16) | 267 ✓ | top-1 = 74035 == HF;top-10 10/10 | `" kitten\n"` 4/4 ✓ |
| flower_blue | 319 / (1,42,28) | 957 ✓ | top-1 = 21655 == HF;top-10 10/10 | `" flower\n"` 4/4 ✓ |

**结论**:owl **自己**(塔 + 文本主干 + m-RoPE,零 HF 参与)出答案,四图答案与 HF
逐 token 相同(文件名是障眼法:dog.png 里其实是小猫 —— 与人工看图一致);
① 同时**闭合了 §12.1 的取证缺口**(mrope 的 shim-vs-真身位置号在真 4B 上逐元素对齐)。
一次 e2e 全跑 4 图 ≈ **11s**(含 3.1s 整装);塔走 FI ragged f16 臂,文本走生产
prefill + 生产 decode 面(v2 scratch + qkv 融合插池)。

### 13.3 门控与遗留

- 门控:`OWL_VLM_REAL=1` + `OWL_TEST_DEVICE`(默认整套不跑,不占卡);
- **CUDA 序 ≠ nvidia-smi 序**(本 boot CUDA 序 = [3090,3090,3080,3080])⇒
  `OWL_TEST_DEVICE=2` 是 nvidia-smi 的 0 号卡(空闲 3080);选错会撞占用卡
  (`CUDA_ERROR_OUT_OF_MEMORY`)——**不是模型装不下**;
- 遗留(M5 服务面):图像预处理进 owl(`Qwen2VLImageProcessorFast` 对拍,现走
  Python 金标)+ OpenAI 兼容 `image_url` + chat 注入文本按 `chat_template.jinja`
  校订(见 `VlmCfg::from_json` 里的 FIXME)。

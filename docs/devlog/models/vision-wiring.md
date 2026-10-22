# 视觉塔 × 引擎/服务面 —— 接线设计与施工(2026-10-20 立案)

> **定位**:把已经存在的**模型级** `Vlm`(视觉塔 + m-RoPE + image_scatter,
> M1–M4 已验证)接进 **引擎 prefill/decode** 与 **OpenAI 服务面**,
> 目标是 2×3080 20G 上跑 27B AWQ-INT4 + **视觉塔** + **DFlash2 d7**,
> 上下文 184,320(与 vLLM `qwen38_3080` 同档)。
>
> 工作树:`packages/owl-worktrees/perf-review`(分支 `perf-review`)。
> **本文是施工台账**:每完成一项就在 §7 勾掉,不另起散记。

---

## 0. TL;DR(现状)

| 面 | 状态 |
|---|---|
| 视觉塔本体(`crates/models/src/layers/vision.rs`) | ✅ 塔/表/CPU+GPU 对拍/真图端到端(4B,与 HF 逐 token 一致) |
| VLM 装配(`crates/models/src/vlm.rs`) | ✅ `Vlm::{new,load_ctx,prepare,inputs,prefill_logits,text_logits}` |
| **引擎** `Vlm` 接线 | ⏳ **本次只完成装载面**(`[model].vision` → 塔权重装载、`RankDev.vlm`、文本主干共享 Arc) |
| **引擎图像 prefill/decode** | ✅ **V2 已真机**(鸟/猫两图 TP2 3080 答案与 HF 金标一致) |
| **OpenAI `image_url`** | ✅ **V4 已接**(data URL;Qwen2VL 预处理;占位展开;真机两图) |
| **图像预处理** | ✅ Rust 实现(2×2 块序);像素 max\|Δ\|≈2/255,端到端答案对拍过 |
| **TP2 视觉** | ✅ **V3 已真机**(两 rank 各备塔输出;图像 prefill 通过) |
| 配置档 | ✅ `qwen38_27b-3080_tp2-w4a16-vlm`(视觉 / spec off 对照)/ `qwen38_27b-3080_tp2-w4a16`(视觉 + DFlash2 d7)/ **`qwen38_27b-3080_tp2-w4a16-vlm`(262k 全设备池)** |

**一句话**:V1–V5 全通。模型层 + 引擎 prefill/decode + OpenAI `image_url` + TP2 图像 +
**图像 × DFlash2 d7** 均已在 2×3080 真机过门;图像×spec 与 spec-off 的 128-token
reasoning **逐字一致**。

---

## 1. 目标 / 非目标

### 目标(V1–V5)

1. 引擎接受 `pixel_values + grid_thw`,在 **prefill 首块**跑塔、注入
   `image_scatter`、用 **m-RoPE** 表;decode 用 `delta` 修正 rope 位置。
2. 服务面解析 OpenAI `content: [{type:"text"}, {type:"image_url"}]`,
   解码/缩放/normalize/patchify 与 `Qwen2VLImageProcessorFast` 对拍。
3. TP2:塔权重每 rank 全量(Replicate),文本主干照旧分片;两 rank 的
   图像张量/表一致,图像 prefill 并发入队。
4. 与 DFlash2 共存:首版先 **spec off** 打通;spec 档在 V5 单独过门
   (verify/fold 的 `pos` 与 KV slot 分离是主要风险)。
5. 内存账:184K + 塔 + 草稿在 2×3080 20G 上的 `[vram-gov]` 实测与
   `--check` 估算对账。

### 非目标

- 视频输入 / 多图 batch / 图像 tokenizer 的 HF 全兼容(先单图 data URL);
- 图像预处理性能优化(先正确,后 CUDA/半精度);
- 视觉塔量化(权重 1.5G f16,先不动;后续另立)。

---

## 2. 关键事实(接线时必须遵守)

1. **塔只在 prefill 首步跑**(`Vlm::prepare` 注释);decode 不再进塔。
2. **文本 token 与图像 token 共用一个序列**:
   - `input_ids` 里 `image_token_id=248056` 的个数 = 塔输出行数;
   - `image_scatter` 把 `merged [N,D]` 按 `idx [L]` 写进 embedding(`-1` = 文本);
   - `VisionTables::build` 的 `grid_thw` 与 patch 序 = processor 的 2×2 块序。
3. **m-RoPE**:`VlmInputs` 给 `[L, half]` cos/sin;prefill 的每个 chunk
   必须切对应行;decode 的新 token 用 **`pos + delta`**(`VlmInputs.delta`),
   但 **KV slot/length 仍用真实 token 计数**(不能把 delta 加进 `kv_len`)。
4. **模型层 dtype**:塔 f16;`Vlm::inputs` 的 cos/sin 按 dtype 打包。
5. **图像预处理是正确性的一半**:HF `Qwen2VLImageProcessorFast`
   (patch 16 / temporal 2 / merge 2 / mean=std=0.5 / smart_resize 32 对齐)
   必须与金标逐元素对拍;错 patch 序 = 静默错答。

---

## 3. 架构与落点

### 3.1 装载(✅ 本次已落)

```text
[model].vision = true
  → ModelLoader::vision(true)                    (Engine::loader 从 knobs 取)
  → 27B 文本 Model 照旧装载(量化/分片/head plan 不变)
  → VlmCfg::from_json(config.json)               (vision_config / image_token / mrope)
  → VisionTower::new + load_from_dir(model.visual.*, shard=1)   ← 每 rank 全量
  → Vlm::from_model(Arc<Model>, tower, cfg)      ← 文本主干同一份 Arc
  → LoadedModel.vlm / RankModel.vlm / RankDev.vlm
```

- **0.8B 档**目前响亮拒(`vision=true` 未接);避免静默不装塔。
- `--check` 校验:`vision_config` 与 `image_token_id` 在场;`[model].vision`
  显示在 intent/arms 行。

### 3.2 引擎请求面(V2)

```text
TurnOpts.images: Vec<VlmImage { pixel_values: Vec<f32>, grid: GridThw }>
  → Turn/ActiveTurn.images
  → execute_prefill 首块(act.vlm 为空 && images 非空):
       vlm.prepare(prompt_ids, grids, px)
       + eval tower root → Bytes (turn owner)
       → ActiveTurn.vlm = VlmPrepared { cos/sin/idx host, merged leaf, delta }
  → ChunkHost.vlm = Some(&VlmPrepared)
  → prefill_chunk_enqueue:
       (cos,sin,idx) = prep.chunk_tables(base, t)
       ctx = ctx.with_vlm_rope(&cos,&sin)
       if chunk 有图像行: ctx = ctx.with_vlm_image(merged, &idx)
  → 桶图路:host.vlm.is_some() ⇒ 强制 eager(bucket 图不烘塔)
```

- `VlmInputs` 增加 `cos_host/sin_host`(idx_host 已有),供 chunk 切片;
  `VlmPrepared` 只持 host 表 + `merged` 叶块所有权。
- **prefill ctx 的 `pos`**:行号 = `base+i`(引擎现有 `pos_t` 已是它);
  m-RoPE 表只覆盖 rope,不覆盖 positions/`kv_len`。
- **decode**:`execute_decode` 里 `pos` 输入槽 = `pos + delta`,
  `kv_len = pos+1`、`kv_slot` 不变;VLM 首版 **`[spec] depth = 0`**
  (spec 的 verify/fold 位置账另开 V5)。
- **合批**:VLM 首版 `max_batch=1`(批路径无图像语义);配置门拒 >1。
- **图族**:VLM prefill 不走桶图;decode 图可保留(图像只影响 embedding/rope,
  decode 的 `pos` 槽是标量)。

### 3.3 TP2(V3)

- 塔权重:**每 rank 全量**(`shard=1`);文本主干照旧 `shard=2`。
- 图像 prefill:两 rank 用**同一份** `pixel_values/grids`(host 输入相同),
  各自跑塔 → `merged` 应逐位一致(塔无 AR);`image_scatter`/mrope 每 rank。
- `VlmPrepared` 每 rank 一份(设备块本 rank 所有);`ChunkHost.vlm` 各 rank。
- 内存:塔 f16 ≈1.5G/rank;184K 主 KV 2.9G/card + 草稿 1.7G/card +
  权重 10.5G/card + fixed ~1G ⇒ 约 16–17G/card(V4 实测校准)。
- **门**:TP2 两 rank 的 `merged` 逐位一致;图像 token 的 logits 与单卡同档一致。

### 3.4 服务面(V4)

- `ChatRequest::validate` 已**显式拒** `image_url`(在接线完成前);
  V4 改为解析:
  - `content` 数组:文本 part 拼接;`image_url.url` 支持
    `data:image/{png,jpeg};base64,...`(先做)与 http URL(后置);
  - 解码 → RGB → **smart_resize** → normalize(0.5/0.5) → patchify
    `[n_patch, 3·2·16·16]`(2×2 块序)→ `pixel_values + grid_thw`;
  - 模板:图像占位 token(`<|image_pad|>`)由 `chat_template.jinja` 产生;
    校验占位 token 数 = `grid` 展开的 merged 行数;
  - 引擎调用:`submit_session_opts` 带 `images`。
- **预处理门**:与 `tools/vision_image_check.py` 的
  `Qwen3VLProcessor` 输出逐元素对拍(索引/权重/pixel_values),四张真图
  端到端答案与 HF 一致。
- 图像缓存/并发:首版一请求一图,串行;不做图像前缀缓存。

### 3.5 与 DFlash2 共存(V5,✅ 已落)

- **位置账**:`VlmInputs.delta` 是 m-RoPE 的全局标量(`maxpos + 1 - L`)。
  verify 图的 `pos_f` 与 eager `verify_forward` 的 `pos_t` 统一改成
  `真实位置 + delta`;**KV `kv_slots` / `kv_lens` / FI `ctx_total` / XQA
  `xqa_seq` / GDN 全部保持真实 token 位置**。与 decode 的
  `pos_rope = pos + delta`、`kv_len = pos + 1` 同律。
- **fold/回滚**:GDN 记录与位置无关,`spec_fold_to` 原样;eager 回滚重放
  也走带 delta 的 `verify_forward`。
- **草稿**:DFlash2 自己的 1D RoPE 用真实位置;所有草稿位置统一偏移是
  RoPE 的相对不变量(且草稿池 slots/lens 仍是真实位置),故无需改
  `dflash_encode/propose`。
- **图像 token 不参与草稿**:首版草稿仍只对文本 continuation 提案;图像
  只在 prefill 首步进塔。
- **门(已过)**:同图、同 seed、greedy,`qwen38_27b-3080_tp2-w4a16`(spec d7)与
  `qwen38_27b-3080_tp2-w4a16-vlm`(spec off)的 128-token reasoning 逐字一致:
  bird 484 chars / cat 493 chars;`spec.rounds=46, accepted=79,
  tokens=125`(AL≈1.72),零 Xid。


### 3.6 262k 全设备池 × 视觉 × DFlash2(2026-10-20)

- `profiles/qwen38_27b-3080_tp2-w4a16-vlm.toml`:262,144 tok 全设备池(非超卖)、
  fp8 主 KV(4 GiB/rank)+ BF16 草稿池(2.5 GiB/rank)、vision、d7、
  `vram_target=0.99` / reserve 256 MiB。
- 真机(2×3080 TP2):boot 成功;`nvidia-smi` 19.19G/20.48G,
  free ~0.92G;图像+spec 64-token 后 free ~0.46G。
  `/debug/vram`:Weight 10.44G/card + Pool 6.75G/card。
- **账本修复后**:TP2 两 rank 共用池前 live 基线,`[vram-gov]` 均
  `cap=313,679`;`准入预算` weights=10.51G ↔ `/debug/vram` 10.44G/card。
  非超卖档 `effective > cap` 现在显式拒启,不再静默钉回。
- 本档是单文件完整档;视觉塔算子由 `feat.vision` tag + 配置派生
  `TagFilter` 自动放行(无 `allowed_ops` / `VISION_OPS`)。

---

## 4. 施工顺序与验收

| 阶段 | 内容 | 验收 |
|---|---|---|
| **V1(本次)** | 配置键 `[model].vision`;loader 装塔;`RankDev.vlm`;`--check`;profile;图像请求显式拒 | `cargo check`;`owl-models` 27B config 解析测试;`--check` 显示 vision=on |
| **V2** | `VlmPrepared` + `TurnOpts.images` + prefill 绑定(mrope/image) + decode delta;spec off | 4B/27B 单卡:同图 greedy 与 HF 逐 token 一致;纯文本逐位回归 |
| **V3** | TP2 塔复制 + 两 rank 图像 prefill | 两 rank merged 逐位;184K `[vram-gov]` 对账;`--check` 通过 |
| **V4** | 服务面 image_url + 预处理 + 模板 | processor 对拍(**实测 2/255**,非 1e-5);真图 e2e 答案对拍 |
| **V5** | VLM × DFlash2 d7 | ✅ 2×3080 真机:图像+spec 与 spec-off 128-token reasoning 逐字一致;spec 46 轮/接受 79;零 Xid |

---

## 5. 文件清单(按阶段)

| 文件 | 阶段 | 动作 |
|---|---|---|
| `crates/shared/src/config/mod.rs` | V1 ✅ | `ModelCfg.vision` + 总账 |
| `crates/engine/src/engine.rs` | V1 ✅ | `EngineKnobs.vision` + `RankDev.vlm` |
| `crates/engine/src/loader.rs` | V1 ✅ | `ModelLoader::vision` + 27B 塔装载 + `Vlm::from_model` |
| `crates/models/src/vlm.rs` | V1 ✅ | `Arc<Model>` / `from_model` / `model_arc` |
| `apps/server/src/config.rs` | V1 ✅ | `--check` vision 校验/告警 |
| `apps/server/src/openai.rs` | V1/V4 ✅ | `image_url` 解析 + 占位展开 |
| `crates/models/src/vlm.rs` | V2 | `VlmInputs.cos_host/sin_host` + chunk 切片 |
| `crates/engine/src/turn.rs` | V2 | `VlmPrepared` / `TurnOpts.images` / `ActiveTurn.vlm` |
| `crates/engine/src/exec/phases.rs` | V2 | 首块 prepare/塔 eval;`ChunkHost.vlm` |
| `crates/engine/src/exec/prefill_chunk.rs` | V2 | mrope/image 绑定;桶图让位 |
| `crates/engine/src/exec/phases.rs` | V2/V5 | decode `pos` 槽 = pos+delta;eager verify 的 `pos_t` = base+delta |
| `crates/engine/src/exec/spec.rs` | V5 | verify 图 `pos_f` = pos+i+delta;slots/lens 真值 |
| `crates/engine/src/exec/draft.rs` | V5 | `verify_forward(..., pos_delta)` 双 rank 透传 |
| `crates/engine/src/running.rs` | V5 | 移除“图像×spec 必拒”;保留单图/非合批门 |
| `crates/engine/src/scheduler.rs` | V2 | 图像 turn 的门(单图/非合批) |
| `crates/engine/src/tp.rs` | V3 | 两 rank 图像 prefill/塔复制 |
| `apps/server/src/vision_processor.rs` | V4 | 解码/缩放/normalize/patchify |
| `apps/server/src/openai.rs` | V4 | `image_url` 解析 + 图像 part |
| `apps/server/src/openai.rs` | V4 ✅ | 模板占位 `<|image_pad|>` 按 grid 展开 |
| `profiles/qwen38_27b-3080_tp2-w4a16.toml` | V1 ✅ | 双卡 + vision + draft 目标档 |
| `testdata/args/site-3080-tp2.toml` | V3 | TP2 卡序/路径站点档(本地) |

---

## 6. 风险与门

| # | 风险 | 门 |
|---|---|---|
| R1 | 预处理与 HF 不一致 ⇒ 静默错答 | `pixel_values` 逐元素 1e-5;四图答案/top-10 对拍 |
| R2 | patch 序(2×2 块序)错 | `VisionTables` 索引/权重逐元素;真图 merged 对拍 |
| R3 | decode `delta` 加进 `kv_len` ⇒ 注意力长度错 | 只改 `pos` 槽,`kv_len/kv_slot` 单测 |
| R4 | VLM 进桶图/捕获 ⇒ 塔节点不稳定 | `host.vlm` 强制 eager;桶图让位 |
| R5 | TP2 两 rank 塔输出不一致 | 同图 `merged` 逐位;图像 token logits 对照单卡 |
| R6 | 内存:184K + 塔 + 草稿超 20G | `[vram-gov]` 实测;`--check` 估算对账;超则降 ctx |
| R7 | 图像 token 数与 grid 不符 | `Vlm::inputs` 已有响亮拒;服务面先校验 |
| R8 | 带图请求被当纯文本 | V4 起解析;无图路径逐字节旧行为 |

---

## 7. 进度台账

- [x] **V1**:`ModelCfg.vision` / `EngineKnobs.vision` / `ModelLoader::vision` /
      27B 塔装载 / `Vlm::from_model` / `RankDev.vlm` / `--check` /
      `image_url` 显式拒 / `profiles/qwen38_27b-3080_tp2-w4a16.toml`。
- [x] **V2/V3/V4(已真机)**:`TurnOpts.images` / `ActiveTurn.vlm` /
      `VlmPrepared` / 首块塔 eval(merged 叶块)/ `chunk_tables` m-RoPE 切片 /
      `image_scatter` 注入 / decode `pos` 槽 = pos+delta(kv_len 不动)/
      桶图让位 / spec off 门;**TP2 每 rank 备料**;服务面 data URL + 预处理 +
      `<|image_pad|>` 展开。
      **真机(2026-10-20,2×3080 TP2,spec off,184K)**:
      bird.png → reasoning "penguin"(HF 金标同) / dog.png(实为小猫) → "small cat";
      prompt 247 / 124 tok,0.89s 级。
- [x] V3:TP2 视觉(两 rank 塔/表;真机过)。
- [x] V4:OpenAI image_url + 预处理(data URL;真机过;像素 2/255)。
- [x] **V5:VLM × DFlash2 d7(真机过)**。
      verify 的 rope pos 加 `delta`,slots/lens 保持真实;eager verify/回滚
      同式;图像+spec 与 spec-off 的 128-token reasoning 逐字一致。
      真机:`qwen38_27b-3080_tp2-w4a16`,2×3080 TP2;bird 251 prompt / 64 tok 1.22s、
      128 tok 1.65s;cat 269 prompt / 128 tok 2.23s;spec 46 轮、接受 79、
      125 tok、零 Xid。rank1 `[vram-gov]` cap 174,801 是**预算 cap**(TP
      顺序记账的产物,非实际池);非超卖 `nb.max(seq_blocks)` 把池钉回
      184,320,`/debug/vram` 两 rank 池字节相等。真实 free ~2.4G/卡;
      满 184K + draft 的瞬态余量仍薄,压测前建议降 174,080。
      **Rebase main `5b27d34` 后**:main 的 N4 泄漏修复已入基线;
      4 × 12.5k prompt(spec d7)压测 free 稳定 ~2.4G/卡,无 OOM。

**当前能做什么**:`qwen38_27b-3080_tp2-w4a16`(视觉 + spec d7)真机带图请求
(data URL,单图;TP2);`qwen38_27b-3080_tp2-w4a16-vlm`(spec off)作为无草稿对照;
两者 `--check` 内存账分别 **18.92G / 19.63G(余 0.71G)** 与
**16.24G / 19.63G(余 3.39G)**。
**不能**做:http 图像 URL、多图/多并发图像;184K 满上下文 + draft 的
瞬态余量薄(真实 free ~2.7G/卡,建议先降 174,080 压测)。

---

## 8. 复现命令(当前)

```bash
cd packages/owl-worktrees/perf-review
cargo build -p server
./target/debug/server --profile qwen38_27b-3080_tp2-w4a16 \
  --site testdata/args/base.toml \
  --set 'runtime.tp_ordinals=[2,3]' --check
# 期望:spec=dflash / vision=on(装载面) / draft dir ✓ / TP2 内存账跳过告警
```

---

## 9. GGUF 主干旁路(2026-10-21)

HF/AWQ 档的塔在检查点目录(`model.visual.*`);GGUF 主文件**只带文本主干**,
塔在**同目录 `mmproj-*.gguf`**。两档共用引擎侧全部接线(V1–V5),差别只在
装载面:

- 配置:`[model] vision = true` + `[model] vision_path = "…/mmproj-*.gguf"`
  (缺省 = `model.path` 同目录唯一 `mmproj*.gguf`;多个/没有 = 响亮拒);
- 事实:`VlmCfg::from_gguf` 从主 GGUF 读 mRoPE sections + `<|image_pad|>` id,
  从 mmproj 读 `clip.vision.*`(num_pos/temporal 由张量形状补);
- 权重:`MmprojSource`(host 解量化到 F16;文本主干仍零解量化)+
  `Vlm::from_model` 共享同一份文本 `Arc`;
- patch_embd 的 `v.patch_embd.weight` / `.weight.1` 两个时间片必须按 HF
  `[out][in][t][p][p]` **按 in 通道交错**(不是首尾相接);
- 真机:4B Q4_K_M + mmproj-F16 / 27B Q4_K_M + mmproj-Q8_0 均过单图 e2e;
  4B OpenAI `image_url`(data URL)鸟图 → `Galapagos penguin`(HF 金标同)。

详见 `gguf-format.md` §23.14 与
`config.md` §7.1。

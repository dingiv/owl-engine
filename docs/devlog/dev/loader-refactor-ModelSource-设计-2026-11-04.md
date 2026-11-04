# 装载器改造:`ModelSource` 统一流 —— 设计 + 性能基线（交接卷）

> **交接对象**:下一绘画(装载器专项)。**目标一句话**:把六个 `load_*` 入口
> 收成「一个源接口 + 一条装载流」,**行为逐字不变、装载耗时不得回归**。
>
> 前置已落地(本卷 §2 的起点,别再改回去):
> `LoadIntent` + `ModelLoader::load_arm`(唯一臂分派)、`CheckpointDecl`(声明面)、
> `[pool] kv_dtype`、`[vision]` 段、`DraftHead`(草稿族接口)。
>
> 日期:2026-11-04 · 基线机:单 3090 Ti(`41:00`)/ W4A8 27B + INT8 KV。

---

## 1. 不变量(先看这一节;任何一条破了 = 返工)

1. **行为逐字不变**:权重值、tokenizer、rope 表、视觉塔、草稿声明、分片结果,
   改造前后必须逐字节等价。门见 §6。
2. **装载性能不回归**(用户明令):同档 3 次**热态中位数**,
   `load` ≤ 基线 ×1.05、`key loop` ≤ ×1.05、`wall_to_health` ≤ ×1.05;
   冷态(第 1 次)单独记录、单独解释。基线见 §5。
3. **单点分派**:臂→源的映射只在 `load_arm`;源→字节只在 `WeightSource`;
   源→声明只在 `CheckpointDecl`/`ModelSource`。**禁止第二套 match**(本卷 §3 病史)。
4. **响亮拒绝**:两源能力差异(逐张量计划、内嵌词表、mmproj、nextn MTP)
   必须由**接口回答**,配置给了不适用旋钮 = 报错点名(参照 GGUF 臂现在拒绝
   `head/embed` 覆盖的实现)。
5. **零 env**:装载参数一律从配置(`[load]` 段)来,层内不读环境变量。
6. **逐张量路径不许加税**:27B ≈ 1000+ 张量,任何 per-tensor 额外分配/拷贝
   都会放大到秒级(现状 `key loop 6.4s` 全部是逐键处理)。

---

## 2. 现状地图(改造起点)

### 2.1 已是接口的部分(**别重造**)

| 层 | 接口 | 实现 |
|---|---|---|
| 字节源 | `module::WeightSource`(`take/take_range/prefetch/owns_shard`) | 6 个 impl:`SafeTensorsSource` / `W4A16Source` / `AwqSource` / `GgufSource`(+`GgufDequantSource`/`MmprojSource`/草稿源) |
| 声明面 | `formats::decl::CheckpointDecl`(本批新增) | `HfDecl` / `GgufDecl` |
| 臂分派 | `engine::LoadArm` + `ModelLoader::load_arm`(本批新增) | 3 条臂:AWQ 家族 / safetensors 家族 / GGUF |
| 计划 | `module::QuantPlan` + `PlanSource::Keyed`(GGUF 逐张量) | 单值 vs 逐键两态 |

### 2.2 **重复的**是「装载流程」(本卷要收的)

`crates/engine/src/loader.rs`(810 行)现有 6 个入口,每个都重写同一串步骤:

```
load_qwen35_hf / load_gguf / load_qwen35_0_8b / load_qwen35_safetensors
/ load_qwen35_0_8b_w4a16 / load_qwen38_27b_awq(_with_embed)
```

十二步 × 六份(计数 = 该调用在文件中出现的次数):

| 步骤 | 重复次数 | 差异点 |
|---|---:|---|
| ① 相位上下文 `scoped_ctx(Owner::System, AssetClass::Weight)` | 部分 | 一致 |
| ② spec 事实 | 4 (`ModelSpec::from_ckpt`) + 3 (`spec_from_gguf`) + 预设 | HF=config.json(+形状兜底);GGUF=文件 KV |
| ③ 分片 `self.shard_spec` / `shard_opts` | 5 / 4 | GGUF 源 `owns_shard()=true`(块算术),其余走 `formats::shard` |
| ④ `Model::new` / `new_with_embed` | 5 | head/embed plan 只有 HF 有 |
| ⑤ `LoadOpts` 组装(debug_tap/verify/staging/dump/device_repack) | 4 | 逐字重复 |
| ⑥ `load_from_dir`(HF) / `eval_load(GgufSource)` | 8 | 分派点不同 |
| ⑦ tokenizer | 5 (`load_tokenizer`) | GGUF = 内嵌词表 `from_parts`(目录在场只读打印) |
| ⑧ rope 表 | 6 (`Rope::new`) | HF 族参数硬编码;**GGUF 来自 `rope_facts`**(+YaRN 覆盖) |
| ⑨ rope `eval_load` | 5 | 一致 |
| ⑩ 视觉塔 | 2 入口 | HF=`model.visual.*`;GGUF=`mmproj-*.gguf` 单文件 |
| ⑪ MTP 草稿检测 | 3 | HF=索引里 `mtp.*`;GGUF=`blk.N.nextn.*` 同文件 |
| ⑫ `LoadedModel{quant_plan, mtp_dir, vlm}` 组装 | 6 | `quant_plan=None`(GGUF 逐张量) |

**病史(两次事故,都是"流程里各写一份"导致的)**:
- GGUF 档 `head_quant`/`embed_quant` 曾**静默忽略**(本批已在 `load_arm` 补响亮拒);
- DFlash2 草稿的 hd=128 几何曾用目标 env 解析 ⇒ 静默错几何(见 `draft_env` 头注)。

---

## 3. 目标接口(建议形态;实现可微调,**语义不许变**)

```rust
// owl_models::formats::source(新)— 一个源 = 声明 + 字节流 + 附属件
pub trait ModelSource {
    fn spec(&self) -> &ModelSpec;                 // 已按 rank 分片
    fn convention(&self) -> &dyn Convention;      // Qwen35Convention / GgufConvention
    fn plan(&self) -> PlanFor;                    // Single { quant, head, embed } | Keyed(GgufPlan)
    fn weights(&self) -> &dyn WeightSource;       // 已有 trait(6 impl)
    fn tokenizer(&self) -> TokenizerSource;       // Dir(PathBuf) | Parts(内嵌词表)
    fn rope(&self) -> RopeFacts;                  // max_pos/rotary_dim/theta(+YaRN 覆盖)
    fn vision(&self) -> Option<VisionSource>;     // ModelVisual | Mmproj(PathBuf) | None
    fn draft(&self) -> DraftSource;               // MtpInCkpt | MtpNextnBlock | Dflash2Dir(dir) | None
}
```

```rust
// engine::loader — 唯一装载流(把 §2.2 的十二步只写一遍)
impl<D: DeviceClient> ModelLoader<'_, D> {
    pub async fn load_source(&mut self, src: &dyn ModelSource, intent: &LoadIntent)
        -> Result<LoadedModel>;
}

// load_arm 退化为"构造源 + 调流":
match arm {
    Awq*        => self.load_source(&AwqDirSource::open(dir)?, intent).await,
    NvFp4|F16Base => self.load_source(&HfDirSource::open(dir, arm)?, intent).await,
    Gguf(_)     => self.load_source(&GgufFileSource::open(path)?, intent).await,
}
```

### 3.1 关键设计决定(照着做,别重新发明)

1. **`ModelSource` 不碰设备**:它的方法全是纯 host 事实/句柄(`open()` 可以
   mmap/index,但不 alloc 设备块、不 eval)。设备面只在 `load_source` 里。
2. **`LoadIntent` 保留**:它仍是"用户给了什么",`plan()` 是"这个源能接受什么";
   `load_source` 负责**交叉校验**(如 GGUF + head≠off ⇒ 拒),不许把两者合并。
3. **逐张量计划不压平**:GGUF 的 `PlanSource::Keyed` 是事实;**禁止**为统一而
   引入"代表值 QuantPlan"(历史上这正是静默错臂的来源,见 `LoadedModel.quant_plan` 头注)。
4. **分片所有权在源**:`WeightSource::owns_shard()` 已表达"块算术列切由源负责";
   流层只做 `if !owns_shard { shard_pass }`。
5. **tokenizer/rope/vision 是源的事实,不是分支**:HF 的 `load_tokenizer(dir)` 与
   GGUF 的 `from_parts` 都收进 `TokenizerSource`;GGUF 目录在场只打印(既有行为)。
6. **`--check` 复用同一套事实**:`CheckpointDecl` 已经是 `ModelSource` 的子集;
   S1 完成后可让 `decl` 从 `ModelSource` 派生(一个 impl 两种用途),避免第三份解析。

---

## 4. 分阶段施工(每阶段独立可验收;不许一次性重构)

| 阶段 | 内容 | 验收门 |
|---|---|---|
| **S0** | 冻结基线(§5)。**在本卷记录,不要跳过** | 3 档 × 3 次热态中位数入库 |
| **S1** | 抽 `ModelSource` 的**声明子集**(spec/convention/plan/rope/tokenizer/vision/draft),`HfDirSource` / `GgufFileSource` 两实现;**不改装载流**(`load_*` 各入口改为从 `ModelSource` 取事实,内部行为不变) | `--check` 三档输出逐字不变;三档 boot 成功;§5 门槛全过 → 可单提交 |
| **S2** | 抽 `load_source`:**先让 AWQ 与 GGUF 两条走新流**,`load_qwen35_0_8b*` 暂留(开发档);旧的 `load_qwen38_27b_awq*`/`load_gguf` 变薄壳(转调新流) | HF↔GGUF 交叉对拍(`tests_gguf::gpu_gguf_4b_vs_hf_4b`)绿;`tests_gguf` 全绿;三档 boot + §5;生成文本 32 tok 冒烟与改前逐字一致(同 seed/greedy) |
| **S3** | 删薄壳与 `load_qwen35_0_8b*` 重复体;`LoadArm` 内部只用于构造源;清 `loader.rs` 里 `Rope::new` 硬编码族参数(统一走 `RopeFacts`) | 同上 + `cargo check --workspace --all-targets` 0 error + 全量单测绿 |

> 每阶段**单独提交**,提交信息里贴 §5 的三行性能对照(改前/改后/比值)。

---

## 5. 性能基线(2026-11-04,改装前;**回归判据见 §1.2**)

### 5.1 怎么测(已入库,直接用)

```bash
# 每档连跑 3 次;第 1 次冷(页缓存未热),2~3 次热 ⇒ 判据用热态中位数
packages/owl/tools/bench/boot_time.sh awq-spec     # AWQ 27B W4A8 + INT8 KV + DFlash2 d7(40k 池)
packages/owl/tools/bench/boot_time.sh awq-nospec   # 同档 spec off
packages/owl/tools/bench/boot_time.sh gguf         # GGUF 27B IQ4_XS(16k,spec off,视觉塔开)
```

探针输出:`wall_to_health`(进程起→/health)、`load`(`[boot] 权重/tokenizer/rope 装载`)、
`assemble`(`[boot] 引擎装配(run)总计`)、`draft`(`DFlash2 草稿装载`)。
更细的分相在日志里:

```text
[load] 预取源区间:… 段 / …GB          ← 预取是否生效(冷盘重叠)
[load] eval_load 分相:键循环(host) X.XXs + 总栅栏 0.00s   ← **主权重逐键处理(大头)**
[load] 超槽键流式:<key> … > 槽 … ⇒ N 段
[aot]  算子预加载 323 个核 @ 29.1ms
```

### 5.2 基线(3 次热态;括号 = 极差)

| 档 | wall_to_health | `load` | `assemble` | 主权重 key loop | 草稿 key loop |
|---|---:|---:|---:|---:|---:|
| **AWQ + spec d7** | **9.75 s**(9.743–9.752) | **7.73 s**(7.70–7.78) | **9.46 s**(9.40–9.52) | **6.43 s**(6.42–6.47) | 0.41 s(0.40–0.41) |
| **AWQ,spec off** | **8.72 s**(8.719–8.722) | **7.82 s**(7.78–7.90) | **8.06 s**(7.97–8.09) | **6.64 s**(6.60–6.71) | — |
| **GGUF IQ4_XS(16k)** | **8.70 s**(8.703–8.704) | **7.44 s**(7.42–7.68) | **7.75 s**(7.71–8.00) | **2.86 s** + 视觉塔 **2.72 s** | — |

复现条件:单 3090 Ti `41:00`;机内 `models/` 已热(第 2/3 次);起机期间**不跑别的 GPU 负载**
(实测有一次并发 3080 实验时 `load` 漂到 9.55 s ⇒ 那只作噪声,不进基线);
三档同参数见 `boot_time.sh`。

### 5.3 判据(硬门)

| 指标 | 门槛 |
|---|---|
| 每档 `load` 热态中位数 | ≤ 基线 × **1.05** |
| 每档主权重 `key loop` 热态中位数 | ≤ 基线 × **1.05** |
| 每档 `wall_to_health` 热态中位数 | ≤ 基线 × **1.05** |
| 任一档超出但 < ×1.10 | 必须给出归因(冷盘/并发/新增一次性工作)并**在报告里点明**,由 owner 裁决 |

> ⚠️ 逐张量路径是热区:27B 主权重 `key loop` 6.4 s ≈ **1000+ 张量 × ~6 ms**。
> 任何"每键多一次哈希/多一次 Vec 拷贝/多一次校验"都会被放大 —— 改造时优先
> **零拷贝/复用缓冲**,并把新增工作放在每档一次的位置(如 open/prefetch)。

---

## 6. 行为等价门(与性能门同权重)

| 门 | 怎么跑 |
|---|---|
| 交叉源对拍(HF vs GGUF 同模型) | `cargo test -p owl-engine --test tests_gguf gpu_gguf_4b_vs_hf_4b`(需 `OWL_TEST_DEVICE`;4B 档) |
| GGUF 全量装载/分片/视觉 | `cargo test -p owl-engine --lib tests_gguf`(或 `--test` 对应入口;查该文件头注) |
| 装载臂/wildcard/几何门 | `cargo test -p owl-cuda --lib`(tag filter 单测)+ `--check` 三档 |
| Spec 路径(DFlash2/MTP) | `boot_time.sh awq-spec` 起机 + 32 tok 冒烟(同 seed/greedy,输出逐字比对) |
| 分片协议(TP2) | 有双卡时:`tests_gguf` 的 split 门 + `profiles/qwen38_27b-3090_tp2*.toml` 起机;无卡则**注明未验** |

---

## 7. 已知坑(照着绕)

1. **并发绘画**:`main` 上同时有 MTP T5(`f4e0f611`)与 `kv_dtype`/`[vision]`/`LoadIntent`
   改造。开工前 `git pull --rebase` 一次,装载器改动**只碰** `crates/engine/src/loader.rs`、
   `crates/models/src/formats/{decl.rs,source.rs,gguf/*}`、`apps/server/src/main.rs` 的
   `load_shard`;**不要动** MTP/spec 面。
2. **`PlanSource::Keyed` 不许压平**(见 §3.1-3);GGUF 的 `LoadedModel.quant_plan = None` 是语义。
3. **GGUF 的 tokenizer/rope 来自文件**,不是目录;`[model].path` 同目录有 HF 文件时只打印(既有行为)。
4. **mmproj 解析**:`resolve_mmproj_path`(唯一候选/多个=拒);HF 档给 `[vision] path` = 拒。
5. **`owns_shard`**:GGUF 自己切;通用 `formats::shard` 的"连续源区间"模型表达不了块算术列切。
6. **MTP 头检测**:HF = 索引里 `mtp.*`;GGUF = `blk.N.nextn.*` 同文件 ⇒ `mtp_dir = 同文件`。
7. **0.8B 开发档**(`load_qwen35_0_8b*`):S3 才合并;它有自己的 rope 硬编码与 preset,
   合并时**不要**把 preset 变成隐式分支 —— 让 `ModelSource` 回答,或保留独立源实现。
8. **`load.27b.total` metrics 与 `[load] eval_load 分相`** 是既有观测面,改造后**必须还在**
   (有消费者:`/debug/metrics`;别为了简化删掉计时)。

---

## 8. 交付格式(交回给 owner)

1. 分阶段提交(S1/S2/S3 各一笔),提交信息含:**改了什么 / 门怎么过的 / 性能三行对照**;
2. 报告里贴 §5.3 的表(改前 / 改后 / 比值),并注明是否与并发负载同窗;
3. 未完成项/风险写成清单(别埋在提交里);
4. 若发现本卷前提有误(例如某个差异其实是行为,不是重复),**先停手写清楚**再改代码。

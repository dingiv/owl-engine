# 分支/形状 review 与接口化改造（2026-11-04）

> **用户裁决的靶子**:支持面已经长成四维笛卡尔积,分支开始散落 ⇒
> 「用多态/接口依赖代替散点分支」。本卷 = 一次**实测盘点** + 一条已落地
> 的样板刀 + 后续三刀的验收口径。

## 0. 形状从哪来（四维 + 两道几何门）

| 维度 | 取值 | 谁在分叉 |
|---|---|---|
| KV 存储格式 | `f16` / `fp8-e4m3` / `int8`(＋ e5m2 只有类型) | 池几何、写者、读者、verify 臂 |
| 草稿池 | `bf16` / `fp8` | DFlash2 的 K0/NC 核选择 |
| 模型源格式 | safetensors/AWQ · GGUF | 装载器(两套 10.5k 行实现) |
| 投机族 | off · DFlash2 · MTP | 草稿装载、propose、verify 分片 |
| 相位 | prefill · decode · verify · batch decode | attention 层 7+5 个终端臂 |
| 几何门 | `page∈{32,64}` · `hd` · `paged` | `paged_*_ok()` 谓词 + 毒化声明 |

组合数 ≈ 3×2×2×3×4 —— 但**实际可达组合远少于笛卡尔积**:每个维度都有
能力门。问题的本质不是"组合多",而是**能力门写成散点谓词**(`is_int8()`、
`matches!(format, …)`),新增一档就要回去把每处谓词再想一遍。

## 1. 实测盘点（改造前;可复核）

| 位置 | 计数 | 形态 |
|---|---|---|
| `layers/attention.rs::forward_decl` | 266 行 / 15 `if` / 2 `match` | decode 7 终端臂,三层嵌套(融合插池 → 格式 → scratch) |
| `layers/attention.rs::forward_prefill_decl` | 331 行 / 22 `if` / 1 `match` | prefill 5 终端臂 + int8 分叉 + verify 三级回落 |
| `models/env.rs` | **7 份** `match self.quant`、**9 处** `unimplemented!` | 每种用途一个访问器,每个访问器自带一份 4-arm match(e5m2 文案抄 7 遍) |
| 分派点 `is_int8()/is_fp8()` | attention 4+2、graph_fleet 2+1、spec 1+1 | 用**格式身份**反推"有没有这条臂" |
| `Drafter::` 匹配 | 13 处(spec_runtime 6 / graph_fleet 4 / exec/spec 3) | 草稿族分叉 |
| `ModelFormat::` | 44 处 | 装载器/`--check` 双实现分叉 |

**顺带发现的死代码**:`attention.rs::paged_prefill_split_output`(≈100 行)
**没有任何调用点**,`KvEnv::prefill_split_op()` 也只被它调用 ⇒ split 臂已退役
但表/访问器/内核 allow-list 仍在(见 §4 待裁决)。

## 2. 已落地（slice A）：KV 内核表 + 能力谓词

### 改法

`crates/models/src/env.rs`:

```rust
pub struct KvKernels {                 // 一张表:7 列(用途),4 行(格式)
    k0_write, prefill_paged, prefill_split: Option<_>, decode_v2,
    decode_v2_seq: Option<_>, fused_insert: Option<_>, fi_prefill: Option<&'static str>,
}
impl KvQuant { pub fn kernels(self) -> Result<KvKernels, &'static str> }
```

- **单一分派点**:一条 `match` 描述"这个格式有哪几条臂";`None` = 缺臂
  (是**数据**,调用方按能力选路),`Err` = 整个格式没实现(e5m2)。
- `KvEnv` 的 7 个访问器 → 查表(名字/签名不变 ⇒ 调用面零改动)。
- 新增能力谓词:`supports_fused_insert / supports_prefill_split /
  supports_v2_seq / supports_fi_prefill`。
- 分派点从"格式身份"改成"能力":`attention.rs` 融合插池谓词
  `!kv.is_int8()` → `kv.supports_fused_insert()`。
- 新增矩阵门 `env::kv_quant_tests::kv_kernel_matrix_is_single_sourced`:
  逐格钉住三种格式的臂矩阵 + 谓词与表恒等 + e5m2 必为 `Err`。

### 效果

| 指标 | 前 | 后 |
|---|---:|---:|
| `env.rs` 里 `match self.quant` | 7 | **1** |
| `env.rs` 里 `unimplemented!` | 9 | **0**(缺臂=Option/Err) |
| 新增格式要改的地方 | 7 处 match | **1 行表项** |
| 新增用途要改的地方 | 7 个访问器 | **1 列** |

验证:`cargo check --workspace --all-targets` 0 error;`owl-models` 314 项
(含新门)全绿;int8 / fp8+shadow 两档实机 8k 预填 **2348.0 / 2372.4 t/s**
(改造前 2337~2377,噪声内),decode 正常。

## 2.5 已落地（slice B）：attention 决策/执行分离（`AttnPlan`）

### 改法

`layers/attention.rs` 三个**纯决策函数** + 三个臂枚举，臂体原样搬进 `match`
（不重检前置条件）：

| 决策函数 | 产物 | 首选序（= 行为契约） |
|---|---|---|
| `plan_decode` | `DecodeArm::{Fused, Int8, BatchV2, BatchMissingFace, Paged, Naive}` | 融合插池 → int8 族 → 批 verify → 单行分页 → naive |
| `plan_int8_prefill` | `Int8PrefillArm::{NonF16, MissingScratch, Verify, Plain}` | 结构违约 → verify 面 → 普通 |
| `plan_prefill` | `PrefillArm::{VerifyXqa, VerifyV2, Fi, Chunked, Naive}` | XQA → v2 → FI → chunked → naive |

### 效果（同文件同函数量）

| 函数 | 行数 | `if` | `match` |
|---|---:|---:|---:|
| `forward_decl`（前 → 后） | 266 → **217** | 15 → **6** | 2 |
| `forward_prefill_decl`（前 → 后） | 331 → **249** | 22 → **7** | 3 |
| `plan_decode`（新） | — | 9 | 1 |
| `plan_int8_prefill`（新） | — | 3 | 1 |
| `plan_prefill`（新） | — | 8 | 0 |

净效果：**37 个散点 `if` → 3 个纯决策函数（共 20 个门，全部单点可测）+ 3 个
`match`**；臂体零改动（数值路径逐字不变）。

### 顺带的三处定谳

1. **删死分支**：`forward_prefill_decl` 里 `if is_int8() { poison("只支持 F16 激活") }`
   在 int8 段全路径返回之后 ⇒ **不可达**，已删（-6 行）。
2. **补一个暗道**：融合插池臂的判据原先取 `kv_paged_policy(dtype)`（不看 `paged`）
   ⇒ legacy 直排 + `qkv_fuse` 会选到**写页池**的核（生产不可达，因为
   `qkv_fuse` 只在 paged 档开；但决策矩阵不该有这条暗道）。决策函数里补
   `ctx.env.kv.paged` 门。
3. **钉住既有次序**：`force_naive_prefill` **只拦预填族（FI/chunked），不拦
   verify 的 v2/XQA 臂** —— 单元测试把它写成断言，防未来"顺手统一"改坏。

### 门与验证

- 新增 `plan_decode_matrix_is_ordered_and_total`（8 条断言：融合/int8 优先/
  批 verify/缺批面/force_naive/legacy 直排）
- 新增 `plan_prefill_matrix_is_ordered_and_total`（9 条断言：chunked、XQA 缺面
  落 v2、`verify_arm=fi` 跳过 v2、force_naive 语义、小几何落 naive、int8 三态）
- `owl-models` **316 项全绿**；workspace check 0 error；
  int8 / fp8+shadow / f16 三档实机 8k 预填 **2353.1 / 2373.1 / 2364.8 t/s**
  （改造前 2348/2372/2361，噪声内），spec d7 的 verify 路径同轮跑通。

## 3. 后续两刀(各自独立可验收)

### 3.1 attention:决策与执行分离(`AttnPlan`)

现状:`forward_decl` / `forward_prefill_decl` 一边判断一边算,37 个 `if`
把"是否可用"与"怎么算"缠在一起。目标:

```rust
enum DecodeArm { Fused{pol, v2: bool}, Int8, Batch{bt8}, Paged{pol, v2: bool}, Naive }
enum PrefillArm { Int8Verify, Int8, VerifyXqa, VerifyV2, Fi, Chunked, Naive }
fn plan_decode(ctx, shape) -> DecodeArm      // 纯函数(无 GPU/Mock 可单测)
fn plan_prefill(ctx, shape) -> PrefillArm
// forward_decl = self.run_arm(plan_decode(..), ..); 臂体只跑不判
```

- **收益**:两处 37 个 `if` → 2 个 `match`;矩阵变成纯函数 ⇒ 新增
  `plan_matrix_is_total_and_ordered` 单测(不用 GPU),`note_arm` 计数
  继续做运行期自证。
- **风险控制**:臂体原样不动(只搬条件),boot + `note_arm` 对比。
- **前置**:先给 `paged_*_ok` 谓词补表驱动(与 §2 同款)。

### 3.2 草稿族:`DraftHead` trait(已落地,2026-11-04 第二批)

**判断:两族的差异是结构性的,不是可互换行为** —— MTP 走链式 block
(m+1 自回归 + 链池),DFlash2 走噪声块 + 码本选择器 + 环窗口 + taps;
propose 的签名、图拓扑、KV 布局都不同。把**行为**硬塞进 `Box<dyn>`,
只会把编译期 match 换成运行期 downcast,**分支一个不减**。

落地的接口 = **族判定 + 类型化下探**(共享面),族专行为留在族内:

```rust
pub(crate) trait DraftHead {
    fn family(&self) -> SpecMode;                       // 数据,不是变体
    fn as_dflash2(&self) -> Option<&DFlash2Draft> { None }
    fn dflash2_arc(&self) -> Option<Arc<DFlash2Draft>> { None }
    fn as_mtp(&self) -> Option<&MtpPredictor> { None }
    fn mtp_arc(&self) -> Option<Arc<MtpPredictor>> { None }
}
impl DraftHead for Drafter { /* 全仓唯一一处变体 match */ }
```

**效果**:`Drafter::` 模式匹配从 **17 处(spec_runtime 6 / graph_fleet 4 /
exec/spec 3 / exec/draft 4)→ 0 处**(枚举只在「构造点 + trait impl」出现,
其余全走 `family()/as_*()`)。`generation_kind`、`propose_first/ext`、
`prefill_extend` 的"是不是 MTP"判断改写成 `as_mtp().is_some()` 或
`family() == …`,并顺手把 `unreachable!("DFlash2 不走 propose_first")` 一类的
"先匹配再断言"改成 **`ok_or_else` 结构化错误**(缺句柄 = 装配违约,不是 panic)。

验证:`owl-models` 316 / `owl-shared` 127 / `owl-cuda` 40 / engine config 全绿;
实机 DFlash2 d7 双档(int8 8k **2313.5 t/s**、fp8+shadow 32k **2094.3 t/s**)
propose/select/verify 全路径跑通。

### 3.3 装载器:`CheckpointDecl`(第一步已落地,2026-11-04 第三批)

**先做的判断**:装载器里真正的"两套并行实现"其实是 **权重装载**
(`formats/qwen35` vs `formats/gguf`),而 `LoadArm` 枚举早已是唯一分派点
(`load_shard` 一个 match)。真正在**重复校验逻辑**的是 `--check`:
`check_model_dir`(读 `config.json`)与 `check_gguf_model`(读 GGUF KV +
`rope_facts`)各写一遍「几何对账 + 视觉声明 + 一句话摘要」。

所以第一步收的是**声明面**:

```rust
pub trait CheckpointDecl {           // formats/decl.rs(纯 host,无 async/无设备)
    fn geom(&self) -> &DeclGeom;      // 唯一几何事实(各源自己的解析器填)
    fn format_name(&self) -> &'static str;
    fn summary(&self) -> String;      // 格式 + 关键事实(不含比对结论)
    fn max_pos_policy(&self) -> MaxPosPolicy;   // HF 硬拒 / GGUF 软警告(YaRN)
    fn vision(&self, path: Option<&Path>) -> Result<Option<VisionDecl>, String>;
    fn draft(&self) -> DraftDecl;
}
pub fn open_decl(path, gguf: bool) -> Result<Option<Box<dyn CheckpointDecl>>, String>;
pub fn check_geom_against(decl, profile, max_seq) -> Result<(), String>;  // 源无关对账
```

**效果**:`apps/server/src/config.rs` 里两个格式专用校验函数(合计 ~150 行)
→ **一个 `check_model_decl`(≈35 行)**;调用点从「按 `is_file()` 二选一」
→ 一条路径(只把 `gguf` 位交给 `open_decl`)。几何/`max_pos`/视觉/摘要
全部单点;新增源格式 = 加一个 `impl CheckpointDecl`。

**输出保真**(三档实跑逐行对拍):HF/GGUF/int8 三档的 `model dir:` 行**逐字一致**,
唯一差异 = HF 档在 `vision=on` 时也追加 `· vision=model.visual.*`(GGUF 一直有;
属信息补齐)。负例同样保真:HF 档给 `[vision] path` ⇒ 响亮拒(与装载期同律)、
GGUF 档 `enabled=false` 却给 path ⇒ 自相矛盾、HF 目录缺 `config.json` ⇒
只提示"几何校验跳过"不判失败。

### 3.3b 装载器(第二步,未做):`ModelSource` trait(AWQ/safetensors vs GGUF)

现状:`loader.rs`(810 行)+ `formats/qwen35.rs`(2.2k)+ `formats/gguf/*`(10.5k)
两套并行,`ModelFormat::` 44 处分叉,`--check` 另有一套。目标:

```rust
trait ModelSource { fn spec(&self) -> ModelSpec; fn plan(&self) -> QuantPlan;
                    fn tensors(&self) -> TensorIndex; /* … */ }
```

装载流程(单条)只吃 trait;格式差异(零解量化/mmap/分片键名)封在实现里。

### 3.4 死臂清理(已按用户裁决删除,2026-11-04)

`paged_prefill_split_output`(89 行)+ `KvEnv::prefill_split_op()` +
`KvKernels.prefill_split` 列 + `supports_prefill_split` 能力谓词 —— 全部删除
(该臂自 2026-10-19 起无调用点:FI 恒在 ⇒ split 不可达)。内核 `PrefillSplit*`
仍注册(内核级探针在测),但**不再有层臂入口**;若将来要接回,按 §2 加一行
`Some(..)` 即可。

## 4. 本次确认的纪律(别退化)

1. **单一分派点 + 表** > 每用途一份 match(`KvKernels` / `KvPageCodec` / `Linear` 的 quant plan 同款)。
2. **缺臂 = 数据**(`Option`/`Err`),不是 `unimplemented!`(panic 只许留在装配边界)。
3. **按能力选臂**,不按格式身份(`supports_*`,不是 `is_int8()`)。
4. **决策与执行分离**:决策是纯函数(可单测矩阵),执行体只跑不判。
5. 新增一档 = **加表项/加列**(改 1 处),不是回去巡检 7 处谓词。

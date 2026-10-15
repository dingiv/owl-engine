# 装载器 = 模型解释器(设计与迁移)

> 立项:2026-10-16,用户裁决 ——「把 loader 也变成模型解释器这种架构:
> 每个模型文件里都声明了它希望长成什么样、layout 怎样(终点);
> 从检查点的 config 能读到它的起始格式;起点与终点都有 ⇒ 就能通装。
> 写通用一点,代码多没关系,以后加新模型要好加」。
> 配套:**`specs/` 并入 `formats/`**(模型档 = 一种起点格式);
> 维度事实 = **config.json 优先 + 张量形状兜底**(27B AWQ 的 config 与形状不符 ⇒ 以形状为准 + warn)。

## 1. 病根(现状)

装载侧今天长这样(`crates/models/src/specs/qwen35.rs`):

```
load_0_8b            load_0_8b_w4a16      load_27b_awq
load_27b_mtp         load_27b_dflash2     load_27b_dflash2_with          (+ Vlm::load)
```

每个函数把 **(模型档 × 检查点格式 × 计划/策略)** 三个轴焊死在一个名字里:

| 轴 | 事实 | 今天写在哪 |
|---|---|---|
| 模型档 | 维度(hidden/layers/heads/eps/vocab/tied/layer_types) | 手写常量(`qwen3_5_0_8b()` / `qwen3_8_27b()`) |
| 键约定 | 检查点键基座 + 子前缀(`Qwen35Convention` / 原生直映) | 函数内 `::new("model.language_model")` |
| 起点格式 | safetensors bf16/f16 · pack-quantized(W4A16) · AWQ(g32-asym+zp) | 函数名 + 函数体选源 |
| 计划/策略 | QuantPlan / head_plan / device_repack / dtype | 函数体硬编码 |

⇒ **N 模型 × M 格式 = N·M 个函数**,且每条新线都要重抄一遍 ctx/源/调用序。

## 2. 两轴正交(终点 × 起点)

### 2.1 终点 = 模型声明(**已经存在,勿改**)

每个模型文件已经声明了"我想长成什么样":

- `Loadable::layout(&self, ctx) -> LoaderOps`:键清单(Want = **键 + 形状 + dtype + Layout 动作**);
- `Layout::{Direct, Transposed, DeviceRearrange{op,rows,cols}}`:该槽要什么**装载动作**;
- `Sink::deliver`:装载产物回填口(执行器写入,使用者零感知)。

⇒ 终点侧已经是一份**可求值的声明**,不缺东西。

### 2.2 起点 = 检查点格式(**本设计新增的唯一真源**)

```rust
/// 起点格式(数据 + 一条 I/O 口);**不由函数名决定,由目录探测**
pub trait CheckpointSource: WeightSource {
    /// 目录是否属于本格式(纯索引/文件嗅探,零数据拷贝)
    fn sniff(dir: &Path) -> Result<Option<SniffReport>, ModelError> where Self: Sized;
    /// 原生条目清单(键 → 原生 dtype/形状/字节区间);装载决策的唯一事实源
    fn index(&self) -> &EntryIndex;
    /// 起点侧的"建议计划"(格式 → QuantPlan/head_plan/repack 的**默认值**)
    fn default_plans(&self, spec: &ModelSpec) -> PlanHint;
    fn open(dir: &Path) -> Result<Self, ModelError> where Self: Sized;
}

pub enum CheckpointKind { SafeTensors, PackQuantized, AwqInt4, /* … */ }
pub fn detect(dir: &Path) -> Result<Detection, ModelError>;  // 逐族 sniff,命中即定;多命中/零命中 = 响亮拒
```

**嗅探先例(已在仓内,提炼为规矩)**:`load_27b_dflash2_with` 今天的
"`fc.weight_scale` 在 = compressed-tensors W4A16 g128,缺 = BF16 原生"
就是起点探测 —— 把它从函数体里提出来,变成 `CheckpointKind` 的判定规则表。

### 2.3 模型档 = 起点侧的一种"元数据格式"

`specs/` **并入 `formats/`**:模型档不再是独立的装载入口,而是"如何从
检查点自己读出维度事实"的一种来源:

```rust
impl ModelSpec {
    /// config.json 优先 + 索引形状兜底(VLM 的 `VlmCfg::from_json` 已是先例)
    pub fn from_ckpt(dir: &Path, index: &EntryIndex) -> Result<ModelSpec, ModelError>;
}
```

复核规则(**响亮**):

| 项 | 来源 | 复核 |
|---|---|---|
| hidden / layers / inter / eps / vocab / tied / layer_types | `config.json`(text_config) | 与索引实测层数/形状一致;不一致 ⇒ warn + **以形状为准**(27B AWQ 案) |
| full heads / head_dim | `config.json` | `q_proj` 形状 ↔ `(hq(+gate))·hd`;不符 ⇒ 形状为准 + warn |
| GDN 头数 | `config.json` | `in_proj_qkv(z)` 形状 ↔ `2·nk·kd + …`;同上 |
| 键基座(`model.language_model` / `model` / 原生) | **索引实测**(哪个基座命中 `embed_tokens` 就用哪个) | 零命中 ⇒ 响亮拒 |
| 缺 `config.json` | —— | 降级 = 纯形状推断(门控 `OWL_SPEC_FROM_SHAPE=1` 才允许) |

## 3. 装载引擎 = 解释器(动作表)

`formats/load.rs` 今天已经是"装载解释器"(`interpreters/` 是同族的求值器族:
`eval.rs`/`planner.rs`/`generate.rs`/`observe.rs`):`eval_load` 求值 `LoaderOps`
→ 逐 Want 分派执行臂。**本设计只做两件事**:

1. **入口通用化**(唯一入口,消除具名函数):

```rust
pub struct LoadOpts {            // 调用点只给"与模型无关"的旋钮
    pub plan: PlanSource,        // Auto(∝ 起点格式 + spec) | Explicit(QuantPlan)
    pub head_plan: PlanSource,
    pub device_repack: Option<bool>,
    pub verify: bool,
    pub debug_tap: bool,
}

/// 通用装载:起点(探测)→ 目标(声明)f 求值
pub async fn load<M: Loadable, D: DeviceClient>(
    dir: &Path, face: &mut D, model: &M, opts: LoadOpts,
) -> Result<LoadManifest, ModelError>;

/// 组件同口(MTP 头 / DFlash2 草稿 / tied 头 / 塔 —— 都是 Loadable)
pub async fn load_component<T: Loadable + ?Sized, D: DeviceClient>(
    dir: &Path, face: &mut D, comp: &T, opts: LoadOpts,
) -> Result<LoadManifest, ModelError>;
```

调用点从"选函数"变成"给声明":

```rust
// 旧: let model = load_27b_awq(&dir, &mut gpu).await?;
let spec  = ModelSpec::from_ckpt(&dir, &index)?;            // 起点读
let model = Model::new(&spec, key_convention(&dir, &index), QuantPlan::F16, QuantPlan::F16);
let src   = CheckpointKind::open(&dir)?;                   // 起点开
let opts  = LoadOpts::for_kind(src.kind(), &spec);         // 起点建议计划
load(&dir, &mut gpu, &model, opts).await?;                 // 终点声明求值
```

2. **动作选择由"起点 × 目标"决定,不由函数名决定**(动作表 = 解释器词表):

| 终点 Want | 起点原生 | 装载动作(唯一判定点) |
|---|---|---|
| `dtype=f16` | f16 | 直拷(RawCopy) |
| `dtype=f16` | bf16 | **原始字节上卡 → 设备 cast 核**(2026-10-16 已落:`load_direct_device_cast`) |
| `dtype=f16` | f32 | 元素转换(F16C)或 f32 上卡后设备 cast(量小,择一) |
| `dtype=bf16` | bf16 | 直拷(草稿 BF16 全程) |
| `Layout=Transposed` | 任意 | host 转置臂(量小:tied 头) |
| `Layout=DeviceRearrange` + 起点 pack-quantized | I32 packed | packed 上卡 → 设备重排核 → 原块回收(marlin) |
| 任意 | 起点缺键 | **响亮拒**(缺键归因,禁静默) |

## 4. 目录与模块迁移

```
crates/models/src/
  formats/                    ← 起点族(格式轴);specs/ 并入
    safetensors.rs w4a16.rs awq.rs mmap.rs          现有
    kind.rs                  新增:CheckpointKind + detect + sniff 规则
    spec.rs                  新增:ModelSpec::from_ckpt(原 specs/ 的维度事实)
    qwen35.rs ← specs/qwen35.rs                  移入(只剩"键约定 + 特例")
  specs/                     ← 删除(内容并入 formats/)
  formats/load.rs            入口通用化(load / load_component / LoadOpts)
  layers|model|vlm           终点声明(不动;Vlm::load 改走 load_component)
```

## 5. 迁移刀序(每刀独立可验、可回滚)—— **L1-L4 已落,L5 未开工**

| 刀 | 内容 | 门 | 状态 |
|---|---|---|---|
| **L1** | `formats/kind.rs` 探测 + `ModelSpec::from_ckpt`(**只读不切**) | 三档检查点探测正确 + spec 复核与手写常量逐字段相等(0.8B/27B-AWQ/4B-VL) | ✅ 2026-10-16(roadmap §14.1/14.2) |
| **L2** | 唯一入口 `load/load_component/LoadOpts`;旧六具名函数**内部转调**(薄壳) | 全部装载测试绿(行为逐字节不变) | ✅ 2026-10-16(roadmap §14.3/14.4/14.5) |
| **L3** | 调用点迁移(engine/loader.rs、spec_runtime.rs、Vlm::load、测试) | engine 装载回归 + 4B VLM e2e 四图绿 | ✅ 2026-10-16(roadmap §14.6;附声明构造器 + `from_ckpt` 上行) |
| **L4** | **删旧名**(不留旁路)+ `specs/` 并入 `formats/` | 全仓 grep 零残留;`op_tables`/非门控套件绿 | ✅ 2026-10-16(roadmap §14.7) |
| **L5** | 动作表显式化(§3 第二张表落成一处 `plan_actions(kind, want) -> Action`) | 装载分相账(debug_tap)与今天同档或更好;错值门(seed 对拍) | ⬜ 未开工(独立一轮 + 压测门) |

**落地实况(L3/L4 收尾,与上述伪代码的两处差异)**:

1. 未引入独立 `CheckpointSource` trait —— 起点开箱收在
   `load_from_dir` 一个函数里(`detect` → 按 `CheckpointKind` +
   `quant_family(num_bits, group_size, symmetric)` 开源 → `set_device_repack`
   注入 → `load_component`)。起点族可换源靠 `&dyn WeightSource` 对象
   (trait 全程对象安全)。
2. **维度事实上行到调用点**:engine 三个装载入口已改走
   `ModelSpec::from_ckpt(dir, SpecSource::family(preset))` ⇒ 硬编码维度/键基座
   退役(族预设只供 tokenizer/dtype 两项非维度事实);键基座→约定经
   `convention_base()`(去尾点),失配由 `log_spec_report` 逐条上屏。
3. 组件档(DFlash2 草稿)声明走 `formats/qwen35.rs::dflash2_draft(dir, kv_fp8)`:
   维度 = 草稿自己的 config(无 `embed_tokens` ⇒ 空基座),fc 臂 = `detect()`
   的 `CheckpointKind` —— 与装载器同一探测口。

## 6. 验收门(新增模型 = 只加数据)

- 三档回归:**Qwen3.5-0.8B(f16/W4A16)· Qwen3.8-27B(AWQ + MTP + DFlash2)· Qwen3.5-4B VLM**;
- **新模型零代码门**(本设计的立案判据):给一个只有 `config.json + 权重` 的新目录,
  不改任何 .rs 即可装载并跑通冒烟(键约定命中既有族时);
- 端到端:**4B VLM 真图四图**(答案/top-1/top-10 逐项不变);
- 性能:装载墙时 ≤ 现状(单面 2.3s@4B),分相账无退化。

## 7. 挂账(与本设计相邻,勿混刀)

- 逐键栅栏 ≈1s 的正解 = **H2D→COMPUTE 事件序**(2026-10-16 试验:换流免栅栏 ⇒ 确定性错值);
- 装载探测的 `EntryIndex` 需覆盖量化源(I32/F16 混合条目),`mmap::open_raw_index` 已可供底层。
- **量化族的“形状兜底”有空转**(2026-10-16 收尾发现):`spec.rs::shape_of` 对缺
  `.weight` 的量化键回退查 `<base>_shape`,而真实键是 `<base>.weight_shape`
  ⇒ 从未命中;量化档的 config↔形状复核实际不生效(“零失配” = 无证据而非已复核)。
  修法 = 读 `weight_packed` 元数据形状 + `weight_shape` 张量值(I64[2])。
- `LoadOpts` 无 `shard`(单卡恒 1;TP 立项时补)。

## 8. 本轮收尾实测(2026-10-16,全文见 roadmap §14.5-14.9)

- CPU:`owl-models` 153/153 · `owl-engine` 31/31 · `facedir_lint` 2/2 全绿;
- GPU(空闲 3080):0.8B 真权重 smoke 输出不变 / engine 0.8B e2e 双 turn OK /
  **4B VLM 四图 e2e 逐项不变** / DFlash2 草稿设备≡CPU 重排(0/10240,scales 0/409600) /
  MTP 头 16 键绿;
- **新模型零代码门**(本设计的立案判据):Qwen3.5-9B(代码从未见过)起点侧
  `detect` + `from_ckpt` 零失配,`OWL_NEW_CKPT_LOAD=1` 走 CPU face 整装 403 项 @41.2s
  —— 全程零代码改动;
- 未跑:27B 整装门(3090 Ti ×2 被另一 agent 常驻占用,20G 卡装不下)⇒ 待卡补跑。

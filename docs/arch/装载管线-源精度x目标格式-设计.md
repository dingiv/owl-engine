# 装载管线设计：**源精度 × 目标量化格式**

> 状态：设计（2026-10-28）。范围 = **safetensors 系检查点**（fp16 / bf16 / fp32 / fp8，
> 以及已打包的 W4A16 类"奇形怪状"格式）。**GGUF 不在本设计内**（块型源另立一篇）。
>
> 起因：装载器把「源精度」「目标量化」「TP 分片」三件事写在了同一处 ——
> 每加一个目标（NVFP4）就要新加一个 `Layout` 变体 + 一处派发分支 + 一处 staging
> 估算 + 一行 shard 特判。这篇把它拆成三段。

---

## 1. 现状与病根

今天的装载路径（`formats/load.rs` + `module.rs::Weight::layout`）：

```
Weight::layout(ctx) ──> LoaderOps::want(Want{ key, dtype, shape, layout, sink })
                                    │
                          load.rs:736  按 Layout 派发
                            ├── Direct          → 原样 H2D
                            ├── Transposed      → host 转置后 H2D
                            └── DeviceRearrange → H2D 原始 packed → 设备核重排
```

三个结构性缺陷：

| 缺陷 | 表现 |
|---|---|
| **1:1 假设** | 一个源张量只能喂一个槽。NVFP4 是「一源二槽 + 一个 host 标量」，表达不了 |
| **源精度隐含在 dtype 里** | `Want.dtype` 是**目标槽**的 dtype，源是什么精度靠"上下文明白"；fp8 源与 f16 源走同一分支 ⇒ 静默按 f16 读就是错值 |
| **目标量化内联在 Layout 里** | marlin 重排、AWQ a8 重排、GGUF 块各自是一个 `Layout` + 分支；目标越多，`load.rs` 越长，且 staging/shard 两处还要跟着特判 |

**目标**：把装载拆成 **读源（解精度）→ 变换（到目标格式）→ 落槽（可 1:N）** 三段，
新增目标格式 = 新增一个 `Transform` 实现（一个文件），**不再动 `load.rs` 的派发**。

---

## 2. 三条轴的正交化

```
源(safetensors)                         目标(QuantPlan)                  落地
┌───────────────────────────┐      ┌──────────────────────────┐     ┌────────────┐
│ fp16 / bf16 / fp32        │      │ F16        (直读)        │     │ 1 个块      │
│ fp8-e4m3 / fp8-e5m2       │ ───> │ W4A16      (marlin U4B8) │ ──> │ 1..4 个块   │
│ 已打包:U4B8 pack-quantized │      │ W4A8       (marlin kS8)  │     │ + host 标量 │
│         AWQ g32-asym       │      │ NvFp4      (e2m1+e4m3/16)│     └────────────┘
└───────────────────────────┘      └──────────────────────────┘
        Source                        Target                      Slots
```

**关键判据**：`源精度` 与 `目标格式` 是两个独立维度，**唯一耦合点 = 兼容矩阵**（§5）；
变换数学只依赖「输入是 f32 视图」+「目标布局规则」，与源是 f16 还是 fp8 无关。

---

## 3. 接口

### 3.1 源描述（替代裸 `dtype`）

```rust
/// 源张量的**事实**（装载器读文件时就确定,不是声明）
pub enum SrcDtype { F16, BF16, F32, F8E4M3, F8E5M2 }

pub struct Source {
    pub key: String,          // 检查点键
    pub dtype: SrcDtype,      // ← 新的第一轴
    pub shape: Shape,         // 文件里的原生形状
    /// 已打包型的附加事实(U4B8 的 group/zero 布局、AWQ 的 g32-asym …)
    pub packed: Option<PackedKind>,
}
```

**纪律**：`SrcDtype` 必须来自文件元数据（safetensors header 的 `dtype` 字段），
**禁止由 plan 反推**。`fp8` 源在没有对应目标臂时 = **响亮拒**，不许按 f16 解释
（同 `admit.rs` 的"不许静默降级"）。

### 3.2 目标变换（`Want` 从 1:1 改成 1:N）

```rust
/// 一个槽的落地声明(目标侧形状/dtype 由变换自己算)
pub struct SlotDecl {
    pub name: &'static str,     // "q" / "sf" / "w" / "scales" …
    pub dtype: Dtype,
    pub shape: Shape,
}

/// 变换:一个源 → N 个槽(+ 可选的 host 标量回填)
pub trait Transform {
    fn id(&self) -> &'static str;
    /// 产物声明(装载前就能算;staging 预算用它)
    fn slots(&self, src: &Source) -> Result<Vec<SlotDecl>, LoadError>;
    /// 在哪里算:Host(已在 host 的源字节上做) | Device(已有设备核)
    fn place(&self) -> Place;
    /// 执行:输入 = 源字节 + 源 dtype,输出 = 各槽字节(+ 回填)
    fn run(&self, ctx: &mut TransformCtx<'_>) -> Result<(), LoadError>;
}

pub struct Want {
    pub key: String,
    pub src: SrcDtype,
    pub slots: Vec<(SlotDecl, Sink)>,   // ← 一源多槽
    pub transform: Box<dyn Transform>,
    pub rules: Option<ShardRule>,       // TP 分片策略(§6)
}
```

### 3.3 精度归一（唯一的"源方差"）

```rust
/// 源字节 → f32。四种 decode 都是常数代价,集中实现一次。
pub fn decode_to_f32(src: &[u8], dt: SrcDtype, out: &mut [f32]) -> Result<(), LoadError>;
```

- f16/bf16=f32 的位运算;fp8-e4m3/e5m2 = 查表(与 `cu/nvfp4/*.cu` 的 `e4m3_dec` 同式,**一处实现**);
- 之后所有变换只吃 `&[f32]`（+ 可选的行/列/分段信息），**不再出现"源是什么精度"的分支**。

---

## 4. 两个执行位置（Host / Device）

| 位置 | 适用 | 判据 |
|---|---|---|
| **Host** | 变换是闭合表达式且源字节已在 host（流式读文件时天然如此）：转置、fp8→f16 | 不需要 stream，最简单，装载期一次性 |
| **Device** | 变换已有设备核（marlin U4B8 重排：实测把 141.7 s 降到 37.0 s；**NVFP4 打包**：18.4 s → 5.3 s） | 有现成核 + 数据量大到 host 往返不划算 |

**已落地：同一个 `Transform` 的两个位置共存，接口不变**（2026-10-28 收尾）。
NVFP4 原先只有 Host（理由：“`pack_b` 是纯 host 函数，不新增算子”），但那与既有
量化族（AWQ 的 `ct_repack` = DMA 上卡 + GPU 就地重排）**两个落点**，属架构不一致
（用户裁决）⇒ 现在：

```
设备臂（首选）: fetch → htod(源 f16/bf16 直上卡) → nvfp4_pack_weight → dtoh(g,4B) → free(源)
host 臂（回落）: fetch → pack_weight_b（CPU）→ htod(q/sf)
选臂判据     : face.device_kernels() && 源精度 ∈ {f16,bf16}（f32/fp8 源走 host）
```

- 设备臂是**正式算子**（`nvfp4_pack_weight`，与 GEMM 同族同 plan：SFB 布局只依赖 `N,K`，
  plan 键取 `m=1` ⇒ 装载期建的那份被首轮 decode 直接命中），选核走 `driver::resolve`
  （装载域的重排也归 `KernelPick`，装载器不自建执行面 —— 与 `ct_repack` 同律）。
- 槽声明 `U8`（不透明源字节）+ `src_bf16` 旗标：同一个核吃两种源精度，声明成 F16
  会在 bf16 源上说谎（事实面会当场按声明比对线格式）。
- **逐字节等价是硬门**：校验档（`OWL_LOAD_VERIFY=1`）把设备产物 dtoh 回来与 host
  同源打包 **memcmp**；要对齐三处（bf16 先落 f16 网格、`want = mx*(1/(6g))`、
  退化回退同式），否则真实权重上会差 1 ULP（实测某层 sf 1.34% 组 / q 0.39% 字节）。
- 顺带补源层缺口：**合并虚拟键**（`in_proj_qkvz` = `in_proj_qkv ⊕ in_proj_z`）原先无字节
  通道（"原始通道不做行拼接"）⇒ 24 个键掉回 f32 元素通道；现按"两段同 dtype ⇒
  行拼接 = 两次 memcpy"补齐，这些键也走设备臂，F16 档同受益（6.76 s → 3.01 s）。

---

## 5. 兼容矩阵（唯一耦合点，写死在 `validate`）

| 源 \ 目标 | F16 直读 | W4A16 (marlin) | W4A8 (marlin a8) | NvFp4 |
|---|:--:|:--:|:--:|:--:|
| fp16 / bf16 / fp32 | ✅ 直拷 | ✅ RTN→U4B8 | ✅ RTN→a8 | ✅ **e2m1+e4m3/16** |
| fp8-e4m3 / e5m2 | ✅ 上采 | ✅ | ✅ | ✅（**只多一个 decode 分支**） |
| U4B8 pack-quantized（AWQ 已打包） | ✅ 反量化 | ✅ 直用（布局已对） | ✅ a8 重排 | ✅ 反量化后重打 |
| **GGUF 块型** | —— 另立一篇（本设计不覆盖） —— ||||

矩阵之外的组合 = **boot 期拒**，带"为什么 + 怎么办"（沿用 `admit.rs` 的错文风格）。
今天 `w8a8f`（fp8 目标）仍是 `validate` 拒绝 —— 本设计只让它变成"有位置可放"，
不假装它已实现。

---

## 6. TP 分片（第三轴，别再内联）

`ShardRule` 三选一，**由变换声明**而不是由 `load.rs` 猜：

| 规则 | 语义 | 消费者 |
|---|---|---|
| `Replicate` | 全量装载（norm/embedding） | 现状 |
| `UniformRows` / `UniformCols` | 先切、再变换（切在源布局上合法） | marlin/GGUF |
| `SliceThenRepack { axis }` | **先切、后用同一变换重打包**（NVFP4：切 N 轴后 `sf` 重新按新 N 算） | NVFP4 |

**当前决定**：NVFP4 的 TP 路径**先写明拒**（单卡档不需要；写"静默走错"不如拒启），
等真要 TP 时实现 `SliceThenRepack`。

---

## 7. 记账与校验

1. **staging 预算**（`load.rs:417`）：由 `slots()` 求和得出，不再按 `Layout` 猜
   （现有 `DeviceRearrange` 那条 `rows*cols*4` 的估算就是猜出来的）。
2. **落账**：每个槽照常登记（name/bytes），`Transform` 声明多少就上多少。
3. **校验**：装载完 `face.sync()` 后，**已有** `OWL_LOAD_VERIFY` 路径可选做回读对拍
   （marlin 的 GPU 重排就是这么验的）。NVFP4 的常驻门 = **装配期 oracle**
   （`ops::oracles::nvfp4`，已在位；选 nvfp4 档即生效）。

---

## 8. 迁移路径（每步都可独立落地，且每步 build/boot 绿）

| 步 | 内容 | 影响面 |
|---|---|---|
| **P1** | `SrcDtype` + `decode_to_f32` + `Transform` trait + `Want.slots`（1:N） | `module.rs` 的 `Weight::layout*`、`load.rs` 派发；旧 `Layout` 变体**保留**并包成 `Transform` 适配器 |
| **P2** | **NVFP4 transform**（host `pack_b`）→ 两槽 + `gb` 回填 ⇒ nvfp4 档可起机 | 新增 1 个文件；`Linear::quantize_nvfp4` 两槽改走 `layout_nvfp4()` |
| **P3** | marlin / AWQ / transposed 三件搬进同一形状，删旧 `Layout` 变体 | 机械活；收益 = `load.rs` 缩短、staging 不再猜 |
| **P4** | fp8 源（`SrcDtype::F8E4M3`）接通：decode 一个分支即可 | 等真 fp8 检查点；矩阵与错文已就位 |
| **P5** | 目标 `W4A8` 从 fp8 源的直通（省一次反量化） | 视收益 |

**不做**：GGUF（块型源）、fp8 作为**目标**（`w8a8f`）、TP 的 NVFP4 分片。

---

## 9. 验收判据（不新增测试，按本仓既有纪律）

| 面 | 判据 |
|---|---|
| 逐位不变 | f16 源 → `W4A16` / `F16` 两条老路径，P1 前后**输出块字节完全一致**（一次性的 `cmp` 级核对，不写成常驻测试） |
| NVFP4 数值 | 装配期 oracle：相对偏差 ≤ 1e-2（量化噪声实测 2.2e-4） |
| 显存 | 4B：f16 8 GB → `q`+`sf` ≈ 2.5 GB；`w` 槽在装载末释放 |
| 拒启 | 矩阵外组合 + `fp8 源 × 无 fp8 臂` + NVFP4×TP ⇒ 结构化拒，带"怎么办" |
| 性能 | 装载时间增量（host 打包）打印分相；目标是 4B ≤ +15 s |

---

## 10. P1 施工清单（下一步就按这个做）

> 目标：**只做抽象，不改任何数值路径**。判据 = 老的 `Direct/Transposed` 与
> `W4A16 marlin` 两条装载路径输出块字节与今天完全一致。

1. `crates/models/src/formats/source.rs`（新）
   - `pub enum SrcDtype { F16, BF16, F32, F8E4M3, F8E5M2 }`
   - `pub fn from_safetensors_dtype(s: &str) -> Option<SrcDtype>`（文件头解析处调用）
   - `pub fn decode_to_f32(src: &[u8], dt: SrcDtype, out: &mut [f32]) -> Result<(), LoadError>`
     （fp8 的 decode 与 `cu/nvfp4/nvfp4_gemm.cu::e4m3_dec` 同式）
2. `crates/models/src/formats/transform.rs`（新）
   - `pub struct SlotDecl { name, dtype, shape }`
   - `pub enum Place { Host, Device }`
   - `pub trait Transform { fn id(&self); fn slots(&self, src:&Source)->...; fn place(&self); fn run(&self, cx:&mut TransformCtx)->...; }`
   - `pub struct TransformCtx<'a> { pub src_bytes: &'a [u8], pub src_dt: SrcDtype, pub rows: usize, pub cols: usize, pub sinks: &'a mut dyn SlotSink, pub face: &'a mut dyn LoadFace }`
3. `crates/models/src/module.rs`
   - `Weight`：`layout()` / `layout_as()` / `layout_as_device_rearrange()` **签名保留**，
     实现改成构造 `Want { src, slots, transform: Box<dyn Transform>, .. }`
     - `Direct`      → `Transform::Copy`
     - `Transposed`  → `Transform::HostTranspose`
     - `DeviceRearrange{op,rows,cols}` → `Transform::DeviceOp{op,rows,cols}`（**适配器原样转发**，含 `driver::resolve`）
4. `crates/models/src/formats/load.rs`
   - `:417` staging 估算：`want.slots.iter().map(|s| s.shape.elems()*s.dtype.size())` 求和（旧 `DeviceRearrange` 的 `rows*cols*4` 作为适配器内声明）
   - `:736` 派发：`match want.transform.place() { Host => run_host(..), Device => run_device(..) }`；
     旧的三个 `load_*` 函数**保留**，只在适配器里被调用
5. `crates/models/src/formats/shard.rs:197`
   - `SliceThenRepack` 先写成 `Err("NVFP4×TP 未实现")`；`DeviceRearrange` 走旧路径
6. 验收：起 GGUF 档 + 跑一发（数值路径不变）；4B f16 档若可起就跑一发对比输出文本

---

## 11. 实现记录（2026-10-28 · P1 部分 + P2 完成）

### 11.1 落地清单

| 文件 | 内容 |
|---|---|
| `crates/models/src/formats/source.rs`（新） | `SrcDtype`（F16/BF16/F32/F8E4M3/F8E5M2）+ `from_dtype` / `from_safetensors`（`F8_E4M3FNUZ` 明确返回 `None`，不冒充 fn）+ `decode_to_f32`（唯一归一点；e4m3 复用 oracle 的 `e4m3_lut`，e5m2 为 IEEE 解析式） |
| `crates/models/src/formats/transform.rs`（新） | `SlotDecl` / `Place` / `Transform` trait / `TransformCtx` + **`NvFp4Pack`**（Host 位：按行带解码 → f16 → `pack_weight_b` → `q`+`sf`+`gb`） |
| `crates/models/src/module.rs` | `Layout::NvFp4Weight{rows,cols}`；`Want.extra: Vec<SlotWant>`（1:N 的附加槽）+ `Want.scalar: Option<ScalarSink>`（host 标量回填）；`Weight::layout_as_nvfp4`；`ScalarSink`（`Arc<AtomicU32>`，替代 `Linear` 里的裸原子） |
| `crates/models/src/formats/load.rs` | `load_group` 新臂 → `load_nvfp4`（一源二槽 + `gb`）；`fetch_source_bytes`（原始字节通道优先，元素通道按 32MB 带流式）；`verify_slot_agreement`（声明面 ⇔ 变换面硬门） |
| `crates/models/src/layers/linear.rs` | `layout_arm` 的 NVFP4 分支（`q` 主槽 + `sf` 附加槽 + `gb`）；`nv_gb: ScalarSink` |
| `crates/kernels/src/nvfp4.rs` / `nvfp4_stub.rs`（新） / `lib.rs` | `pack_weight_b`（+ **并行版** `pack_b_parallel`，行切，逐字节等价）；模块**恒存在**（feature 关 ⇒ 同面替身响亮拒，上层零 `cfg`） |
| `crates/kernels/cu/nvfp4/nvfp4_gemm.cu` | `quantize` 拆成 `quantize_g`（全张量 amax）+ `quantize_rows`（行区间，行间无共享写）；新增 `owl_nvfp4_pack_b_parallel` |
| `crates/engine/src/loader.rs` | 新入口 `load_qwen35_safetensors(dir, quant, head, embed)`；`load_qwen35_0_8b` 退化成 F16 薄壳 |
| `crates/engine/src/lib.rs` + `apps/server/src/main.rs` | `safetensors_load_plan`（普通 HF 目录的**装载期现量化**唯一映射：`nvfp4` 才真档，`w4a16/w4a8` 保持历史空档 ⇒ F16） |

### 11.2 与设计的偏差（有意为之）

1. **`Want` 没有改成 `slots: Vec<(SlotDecl, Sink)>`**，而是"主槽沿用旧字段 + `extra: Vec<SlotWant>`"。
   理由：改主槽要同步改分片 pass（按 `shape` 反推全量形状）、manifest 与所有声明点，
   而今天只有一种 1:N 变换 ⇒ 收益 < 风险。**P3 收编时再并**（那时分片规则也由变换声明）。
2. **`Transform` 今天只有 `NvFp4Pack` 一个实现**。`Copy`/`HostTranspose`/`DeviceOp` 仍是
   `load.rs` 的独立臂 —— 它们是"搬运/换序"而非"格式变换"，设计 §8 的 P3 才收编。
3. `Layout` 保留并新增一个变体（而不是 `Want.transform`）：`Layout` 还承担
   manifest / 分片 pass 的"这些字节意味着什么"语义，与 `Place`/`slots` 是两件事。

### 11.3 实测（RTX 5070 Ti / sm_120，4B BF16 → NVFP4）

| 面 | 结果 |
|---|---|
| 权重显存 | 8053.8 MiB → **1253.8 MiB**（live weights；`q`+`sf`，f16 槽未装载） |
| 装载耗时 | F16 6.76s → **18.4s**（+11.6s，达标 ≤+15s）；并行打包前是 48.9s（`pack` 段 300ms/47MB → 92ms） |
| 装载校验 | `[load] verify` 352 键回读 `mismatch=0`（`q`/`sf` 逐字节 == host 打包） |
| 装配期 oracle | `nvfp4_gemm ✓ max|err|=0.000e0 ≤ 1e-2` |
| 文本质量 | `17*23` → **391**；中文散文正常（B/tok 3.79）；与 F16 档同形 |
| decode（eager，m=1） | **82.3 tok/s** vs F16 **76.8 tok/s**（+7%） |
| prefill（eager，m≈20） | 1625 tok/s vs 1312（+24%） |

### 11.4 顺带修掉的两个**引擎级**缺陷（与量化无关，eager 档长期未被跑过）

1. `GraphPlan::output_block` 的句柄**只在捕获态跨步恒定**；prefill 末块无条件
   `copy_block_at` 到该槽 ⇒ `[cuda] graph = false` 档第二次请求 `[copy-dst-arg] DeadBlock`。
   修法：`is_captured()` 门控发布（eager 每步现算 logits，跳过无损）。
2. turn 收尾顺序：`eval_current` 要到**下一步开头**才释放上一步的 `last` 输出，
   而 N4 兜底释放先把借出登记删掉 ⇒ 下一 turn **幽灵归还（双 free）**。
   修法：`GraphPlan::release_eager_outputs()`（捕获态 no-op）在 N4 兜底释放**之前**调用。

### 11.5 未完成 / 后续刀

| 项 | 现状 | 出口 |
|---|---|---|
| ~~NVFP4 进图~~ | **已完成（2026-10-28 进图刀）**：alpha 设备化（见下）⇒ `CAPTURE = NoHostWork` ⇒ session / verify(T=4) / fold 三图实测捕获成功、数值正确 | — |
| NVFP4 × TP | `build_shard_pass` 结构化拒 | 实现 `SliceThenRepack`（先切源、再按新 N 重算 `sf`） |
| fp8 源 | `SrcDtype` 与 `decode_to_f32` 已就位（含 e5m2），**无消费点**（本机没有 fp8 检查点） | 等真 fp8 源；矩阵与错文已备 |
| P3（三臂收编） | 未开工 | 按本文件 §8 |

---

## 12. 进图刀（2026-10-28 · NVFP4 decode / verify / fold 进 CUDA graph）

### 12.1 做了什么

| 面 | 改动 |
|---|---|
| `cu/nvfp4/nvfp4_gemm.cu` | epilogue 换成 `CollectiveBuilder<…, fusion::LinearCombination<…>>`（Args 带 **`alpha_ptr`**）⇒ 新增 `owl_nvfp4_run_dev(…, const float* alpha_dev, …)`；`owl_nvfp4_run`（host alpha）保留给 oracle/探针 |
| `cu/nvfp4/nvfp4_quant_act.cu` | `k_scale_from_amax(d_g, gb)` 把 `g` 与 **`alpha = g·gb`** 一次写定；`d_g` 槽 = **f32[2]**（`[0]`=g 给量化核、`[1]`=alpha 给 GEMM 的 `alpha_ptr`） |
| `kernels/src/nvfp4.rs` / `server/nvfp4.rs` | `quant_act(…, gb)`、`run_dev(alpha_ptr)`；家族内**删掉 `res.download`（4B D2H）与 `capturing()` 拒绝** ⇒ 零 host 工作 |
| `ops/nvfp4.rs` | `CAPTURE = NoHostWork`（宏据此注入 `feat.graph`）；`dg` 账长门 ≥ 8B |
| `models/src/layers/linear.rs` | `dg` 槽 = `uninit(F32, [2])` |
| `engine/src/engine.rs` | 删掉「NVFP4 ⇒ 自动关图捕获」的降级块 |
| `ops/oracles/nvfp4.rs` | 自检块 `dg` 8B（与家族账长门同口径） |

### 12.2 为什么这条路而不是别的

- **不能事后乘 `alpha`**：多一趟 M×N 读写，2048×9216 实测 ≈ +24%（把 prefill 的 3.3× 吃掉一半）。
- **不能把全局缩放折进 `sf`**（等价 `g≡1`）：会在块 amax > 6·448 时 e4m3 饱和（静默削顶），
  且改变已定谳的数值契约。
- 设备 `alpha_ptr` **保持逐位等价**：host 的 `ga*gb` 与设备的 `g*gb` 都是 f32 乘法。

### 12.3 实测（4B BF16 → NVFP4，`[cuda] graph` 缺省）

| 面 | 结果 |
|---|---|
| 捕获 | `引擎就绪(捕获态 Captured)`；`verify 桶形图捕获(T=4)` 成功；`fold 单图捕获(m=3)` 成功 |
| 数值门 | 装配期 oracle `max\|err\|=0.000e0`；`17*23 → 391`；中文散文 B/tok 3.88 |
| decode | **98~101 tok/s**（eager 时 82；F16 eager 77） ⇒ 进图 +20%、对 F16 +28% |
| 多轮 | 连续多次请求正常（进图后不再受 eager 缺陷影响） |

### 12.4 propose 图（同日闭环）

原先只有 verify（dumb 草稿证明「T=depth+1 图内 NVFP4 数值正确」）。propose 需要 MTP 头，
而 4B 检查点**有** `mtp.layers.0.*` + `mtp.fc.weight`，只是 `config.json` 没声明
`num_nextn_predict_layers` ⇒ spec 不为它建层。三处接线：

| 面 | 改动 |
|---|---|
| `formats/safetensors.rs` | `Entry` 保留 `shape`；新增**虚拟键** `mtp.fc_{e,h}.weight` = `mtp.fc.weight`(`[out, 2h]`)的 in 维前/后半（行内去交织；`take` / `take_range` / `convert_chunk_into_bytes` / `elem_len` 四口同源），与 AWQ 源的同名派生键**同语义**；`has_mtp_head()` 供探测 |
| `engine/loader.rs` | 装载完探 `mtp.fc.weight` ⇒ `mtp_dir = Some(目录)`（与 GGUF 探 `blk.N.nextn.*` 同律：**有没有头只能从权重键看出来**） |
| `engine/spec_runtime.rs` | `SpecMode::Mtp` 非 GGUF 分支的几何改为 **target spec**（`shard_draft_plan(inp.spec, tp)`）：MTP 头与 target 同维是它的定义性质 —— 旧实现重读草稿目录 `config.json` 还把 `qwen3_8_27b()` 预设写死，4B/0.8B 档一旦 config 缺字段就落到 27B 默认值 |

实测（4B NVFP4，`[spec] depth = 3, draft = "mtp"`）：
`propose 桶形图捕获(m=0..3)` 四张 + `verify(T=4)` + `fold` 全捕；
**AL 3.06~3.24 · decode 188~190 tok/s**（无投机 98~101 ⇒ ≈1.9×）；
`17*23 → 391`、中文散文正常；9B GGUF 生产档回归逐字节一致。

---

## 13. Review 修正（2026-10-28 收尾）

自审整条链路（装载 → 发射 → 进图 → 投机）后的修正，按严重度排：

| # | 面 | 问题 | 修法 |
|---|---|---|---|
| 1 | `transform.rs::NvFp4Pack::slots` | **SF 尺寸口径错**：门按库的 `N%8` 收，却用 `N*K/16` 报字节数；SFB 的布局按 128 对齐排布 ⇒ `N` 只 8 对齐时 `cosize > N*K/16`（**实测** `N=1000,K=256` ⇒ 16384 vs 16000），槽会**小一号**（上卡时静默踩邻块） | 门提到 `N%128`（与层侧 `quantize_nvfp4` 同门），且尺寸直接取库的 `sf_bytes` **同一个 const fn** |
| 2 | `load.rs::load_nvfp4` | **`gb` 静默丢失**：`if let (Some(cell), Some(gb))` 任一侧缺就跳过 ⇒ 层继续用 1.0，输出被 `1/gb` 静默缩放 | 声明缺 `scalar` 口 = 拒；`scalars` 初值 **NaN**，跑完仍非有限 = 拒（两侧都响亮） |
| 3 | `spec_runtime.rs::SpecMode::Mtp` | **MTP × TP 二次分片**：`inp.spec` 已是本 rank 分片，`shard_draft_plan` 再切一刀（`ModelSpec::sharded` 不幂等）⇒ 双 rank 草稿头错值。旧 GGUF 分支同形，只是从没走到 | `tp>1` **结构化拒**并注明接线点（需要未分片事实） |
| 4 | `config/mod.rs` | safetensors × `head_quant`/`embed_quant`：`i8q/i8s` 是 **AWQ 源**装载期现算的派生键；配置面真生效后会拖到装载期才报“缺键”（旧行为是整链**静默忽略**） | 配置面加门：纯 safetensors 档非 `off`/`f16` ⇒ boot 前拒，错文给两条出路 |
| 5 | `kernels/src/nvfp4.rs`、`server/nvfp4.rs` | 文档/错文漂移：还写着 `M%128`、`m>=128` 分流（已裁决统一臂） | 改成「M 无门、N%8（库）/ N%128（装载 SF 口径）、K%16」并说明两处 N 门为何不同 |
| 6 | `load.rs::fetch_source_bytes` | 注释说 32MB、实际 `32<<20` **元素** = 128MB f32 | 改 `8<<20`（=32MB f32），与注释一致 |
| 7 | `safetensors.rs::fc_half_range` | 读完不 `madvise(DONTNEED)`（`convert_range` 有）⇒ 13MB 级 fc 常驻 RSS | 补上 |
| 8 | `linear.rs::quantize_nvfp4` | 形状门不过时**静默回落 f16**（"哪层没走 fp4" 只能靠反推） | 首次 info + 逐层 debug（实测 4B 只有 `in_proj_a/b` 32×2560 回落） |
| 9 | 死接口 | `SrcDtype::from_safetensors`、`Linear::is_nvfp4` 无消费者 | 删；前者随 fp8 源（P4）与首个消费者一起加回 |

**回归证据**：4B NVFP4+MTP 三图全捕、AL 3.24、decode 190.7 tok/s、`17*23→391`；
9B GGUF 生产档输出逐字节同（137/94/757 B）；`head_quant` 门实测 boot 拒且错文可操作。

**挂账未做**：
- `Nvfp4Runtime.plans` 无淘汰（eager prefill 的 chunk 尺寸若频繁变化会累积；每个 plan 只持 host 布局 POD，量小）。
- `pack_weight_b` 每次调用在 C++ 侧新建线程（738 个张量 = 线程 churn）；总装载 18.4s 已达标，未优化。
- MTP 头仍 F16（既定"二期"）⇒ propose 图内没有 FP4 算子。

### 13.1 两个 SF 口径（逐档实测，2026-10-28）

`check_nvfp4_shapes` 加跑 `M=1/7`、`N=1000`、`K=8` 后钉死的事实 —— **A/B 两侧不同**：

| 侧 | 字节数 | 说明 |
|---|---|---|
| 激活 A | `ceil(M/128)·128 · K/16` | M 按 **128 行块**排。M=1 也要一整块（K=2560 ⇒ 20480B）。权威值 = `plan.sfa_bytes()` |
| 权重 B | `N·K/16`（N 128 对齐时） | 与 M 无关；`N` 只 8 对齐时会偏大（`N=1000,K=256` ⇒ 16384 > 16000）⇒ 装载期只收 128 对齐 |

> 教训（自审现场）：review 时**把 A 侧也按 `M·K/16` 缩了**，起机当场被家族的
> 账长门抓住（`asf ≥ 20480 实得 160`）—— 长度门这层"廉价自证"直接兑现了。
> 同时 `K=8` ⇒ `plan=NULL`（K%16），与文档一致。

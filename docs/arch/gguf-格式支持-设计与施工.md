# GGUF 格式支持 —— 设计与施工

> 立项:2026-10-19(worktree `packages/owl-worktrees/gguf`,分支 `feat/gguf`)。
> 上游:`docs/arch/operator-matrix.md` §Q4 + §七(三型专列矩阵)、
> `docs/arch/装载器-解释器化-设计与迁移.md`(起点族范式)、
> `docs/arch/charter.md` A5(严格显存预算)/A6(池分配者契约)/A1(图治理)。
> 需求基线:xinfer `docs/arch/requirements.md` v1.4「双生态支持:gguf k-quant/i-quant,
> 还有 marlin 的 w4a16 w4a8 模型」;父任务 = owl M1→M4 的**生态面补完**。

---

> **配套卷**:接入一种新格式的**通用检查表**(= 本轮改动面提炼,给下一种格式用)见
> [`新格式接入-改动点清单.md`](新格式接入-改动点清单.md)(A 架构面 / B 起点族 / C 层 /
> D 内核族 / E 引擎 / F 服务 / G 门;四条纪律 + 十条反模式)。

## 一、立项动机与用户裁决

**一句话**:让 owl 能吃 GGUF —— 压缩权重**原样驻留显存**、解量化融进 GEMM,
且**单文件自给**(超参 + 内嵌词表 + chat template 全从 `.gguf` 读,不依赖 HF 目录)。

### 1.1 为什么值得做(三条独有优势)

1. **分发 = 一文件**(权重 + 元数据 + 词表 + 模板),对照现有 safetensors 目录树
   (config.json + tokenizer.json + tokenizer_config.json + N×.safetensors);
2. **mmap 优先**压缩驻留 ⇒ 同显存跑更大模型(27B Q4_K_M = 18G,现役 AWQ-INT4 ≈ 19G,
   但 Q4_K_M 一条文件即可分发);
3. **生态存量**:本机已有 5 个 qwen35/qwen35moe/dflash 族 GGUF(见 §1.3),
   且 llama.cpp **上游本树已实现同三族 arch** ⇒ 数值锚有现成参照实现。

### 1.2 用户裁决(2026-10-19,四条,生效优先级最高)

| # | 裁决 | 落点 |
|---|---|---|
| D1 | **两段式**:M1 = 小档(9B 单卡)全链贯通;M2 = 27B 上双卡生产档 | §七 刀序 |
| D2 | **一步到位 MMQ**:dequant-in-kernel(**无 f16 中间态**),不走"载入期 dequant→f16" | §四 L3 |
| D3 | **按 GGUF 标准操作**:超参 + 内嵌词表 + chat template 全自给;借力 `packages/mistral.rs` / `packages/xinfer` | §四 L1/L2、§三 |
| D4 | 首轮量化型 = **Q4_K_M / IQ4 / IQ3**(+ 目标文件实际含的 F32/Q8_0/Q6_K/Q5_K) | §二 REQ-GGUF-04 |

### 1.3 ⚠️ 为什么 D2 是**被逼出来的**(显存算术,一个都不能省)

`载入期 dequant→f16` 这条最省事的路在 27B 上**结构性不可行**:

```
Qwen3.8-27B-Q4_K_M.gguf = 18G(压缩驻留)
  └ dequant→f16 = 18G × (16bit/4.85bit) ≈ 54G
      └ 3090 Ti 对 = 48G ⇒ 装不下;单卡 24G ⇒ 差 30G
```

⇒ 27B 要么保压缩驻留(D2),要么二次量化(requant→marlin,精度有损)。
用户选 D2。**推论**:`Holo-3.1-9B.Q5_K_M`(6.1G → f16 10.5G)虽然单卡可装,
**仍然走 MMQ 同一条路** —— 不为小档另开一条 dequant 支路(避免"两条语义")。

### 1.4 本机 GGUF 存量(决策依据,已实测)

| 文件 | arch | 大小 | 量化型直方图 | 用途 |
|---|---|---|---|---|
| `ggml-org/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf` | `qwen35` | 18G | Q4_K 193 / Q8_0 288 / Q6_K 17 / F32 353 | **M2 靶位** |
| `bartowski/Qwen_Qwen3.5-4B-GGUF/Qwen_Qwen3.5-4B-Q4_K_M.gguf` | `qwen35` | 3.0G | Q4_K 124 / Q6_K 49 / Q8_0 20 / Q5_K 16 / F32 232 | **M1 靶位(2026-10-19 用户指定;轻量快迭代)** |
| ~~`mradermacher/Holo-3.1-9B-GGUF/…Q5_K_M`~~ | `qwen35` | ~~6.1G~~ | ~~Q5_K/Q6_K/F32~~ | **已失效(目录被删)**;其鹅验已由 4B/27B 接管 |
| `Hcompany/Holo-3.1-35B-A3B-GGUF/q4_k_m.gguf` | `qwen35moe` | 20G | — | M3(留口) |
| `z-lab/Qwen3.8-27B-Dflash2-gguf/*.gguf` | `dflash` | 1.1G / 2.0G | Q4_K 45 / Q6_K 4 / F32 32 | 草稿线(留口) |
| `DimInfer/Qwen3.8-27B-Dspark-v1/…-Q8_0.gguf` | `dflash` | 1.9G | — | 草稿线(留口) |

**缺样本**:IQ4_XS / IQ3_XXS / IQ3_S 本机零存量 ⇒ REQ-GGUF-04 的 IQ 臂
必须先解决样本来源(见 §八 R3)。

---

## 二、需求编号与验收门

| 编号 | 需求 | 优先级 | 验收口径 |
|---|---|---|---|
| **REQ-GGUF-01** | **起点族**:GGUF 容器解析(header / KV / tensor table / alignment / mmap),`CheckpointKind::Gguf` 嗅探。纯 host、零数据拷贝、命中即定、歧义/残缺**响亮拒** | P0 | 解析结果与 `gguf-py`(llama.cpp `GGUFReader`)逐字段对拍:张量数/名/形状/dtype/字节区间;4B(441 张)+27B(851 张)双档零差异(原定 9B,目录被删后由 4B 接管) |
| **REQ-GGUF-02** | **单文件自给**:维度事实读 `qwen35.*`/`qwen35moe.*`/`dflash.*` KV;词表读 `tokenizer.ggml.{tokens,token_type,merges,pre,model}`;**chat template 读 `tokenizer.chat_template`** → owl `ChatFormat` | P0 | 同目录**不含** `config.json`/`tokenizer.json` 时仍可启动;tokenize 结果与 llama.cpp 逐 token 同 |
| **REQ-GGUF-03** | **键约定**:`GgufConvention`(`KeyConvention` 第三套)—— `blk.N.*` ↔ owl `layers.N.*`,含 GDN(`attn_qkv`/`attn_gate`/`ssm_*`)、FA(`attn_q/k/v/output/q_norm/k_norm`)、MLP、embed/lm_head/norm | P0 | 键面覆盖 = 源索引键面(零缺键、零未用键;两侧集合相等,机器判据) |
| **REQ-GGUF-04** | **量化型**:`F32/F16` + `Q8_0` + `Q4_K/Q5_K/Q6_K` + `IQ4_XS/IQ4_NL/IQ3_XXS/IQ3_S`;逐**张量**派生 plan(同模型混型是常态) | P0 | 逐型 dequant 与 `candle-gb k_quants.rs` 参考实现逐位(或 ≤1 ulp);GEMM 与 f16 参考 allclose |
| **REQ-GGUF-05** | **压缩驻留 + dequant-in-kernel**:权重以 GGUF 块字节常驻显存;decode 走 MMVQ(vec_dot)、prefill 走 tile GEMM(同一 `gguf_gemm` 家族按 m 分派);**全程无 f16 权重中间态** | P0 | 27B 双卡驻留 ≤ 每卡预算;decode 收益对照同模型 f16 基线可测 |
| **REQ-GGUF-06** | **账本与图捕获合规(A5.2/A1)**:移植的核**零 `cudaMallocAsync`**、**零运行期 `cudaGetDeviceProperties`**;激活量化临时缓冲由 `Pool` 提供并登记;整链可被 decode 图捕获 | P0 | `gguf/*.cu` 源码级门(零 malloc/memset/props 符号);捕获预检通过;账本无"未命名块" |
| **REQ-GGUF-07** | **数值锚**:①层内 = 与 `candle-gb k_quants.rs` / `gguf-py` 对拍;②端到端 = **首 token 硬门 + 全序列软账 + 分叉点 margin 归因**(2026-10-19 实测定谳,见 §十三 13.4:量化档在 near-tie 处必然分叉 —— 硬门"逐 token 逐位"会把噪声记成错) | P0 | ① 五型逐位全等(已绿);② 首 token 与参考源一致 ≥ 6 题、全序列一致 ≥ 半数、每处分叉能由 oracle 侧 top-2 margin 解释(实测 0.14 / 0.013 nat) |
| **REQ-GGUF-08** | **27B 生产档**:`Qwen3.8-27B-Q4_K_M` 在 3090 对上跑通并出 std 五点三轮读数 | P1 | 三线与现役 AWQ-INT4 档并列落 bench;零 Xid |
| **REQ-GGUF-09** | **MoE 骨架**:`qwen35moe` 键面与 `QuantPlan::Gguf` 在 MoE 层的落点(观察者/路由专家) | P2 | 键面覆盖 + 单层对拍(不承诺 e2e) |
| **REQ-GGUF-10** | **VL 塔(mmproj)**:`mmproj-*.gguf` 走同一族加载 | P2 | 4B-VL 四图 e2e 不回归 |
| **REQ-GGUF-11** | **格式转化器**(requirements §其他需求):GGUF ↔ owl 内部资产(HF safetensors / marlin)离线互转 | P2 | 转化产物装载后可跑;精度差异有表 |

**非目标(明确不做)**:GGUF *写入*(owl 只读);`ggml_file`(旧 GGML 格式);
训练/微调侧;llama.cpp 的 GPU 后端等价物(dflash 草稿的 GGUF 线留口不做 M1/M2)。

---

## 三、借力清单(素材 → 用法 → 所有权)

| 素材 | 位置 | 用法 | 所有权 |
|---|---|---|---|
| **gguf.cuh**(152 KB,llama.cpp 派生:全部 K-quant/IQ 的 `dequantize_block_*` + `vec_dot_*`) | `repos/attention.rs/src/kernels/src/gguf/gguf.cuh` | **CUDA 侧主料**:整 file port 进 `crates/kernels/cu/gguf/` | FROZEN 快照,可 copy 改造(owl 已port过 attention-rs 的 FI 适配器有先例) |
| **gguf_gemm.cu**(654 行:`gguf_gemm_kernel` / `gguf_gemm_prefill_kernel` / IQ4_NL 专用 / `quant_type` 分派) | 同上目录 | 内核族骨架;**必改三点**(§四 L3 硬律) | 同上 |
| **k_quants.rs**(3199 行,全型 CPU dequant + 参考 GEMM) | `repos/candle-gb/candle-core/src/quantized/k_quants.rs` | **层内金标**:block 解量化逐位对拍;block 结构体定义可抄 | FROZEN,只作参照(不引入 candle 依赖) |
| **gguf_file.rs**(623 行,纯 host 解析:header/KV/tensor table) | `repos/candle-gb/candle-core/src/quantized/gguf_file.rs` | 解析器骨架(去掉 candle 错误类型) | FROZEN |
| **archive.rs**(2127 行,含对齐/分片/边界校验/字符串长度上限) | `packages/mistral.rs/mistralrs-quant/src/gguf/archive.rs` | 解析器的**防御性校验**取材(MAX_STRING_LENGTH / MAX_ARRAY_ELEMENTS / 深度上限) | FROZEN |
| **gguf_helper.rs**(753 行:**内嵌词表 → `tokenizers::Tokenizer`**) | `packages/xinfer/crates/core/src/utils/gguf_helper.rs` | **词表复刻主料**(直接对口 owl 已依赖的 `tokenizers` 0.21) | OWNED(我们自己的历史产物),可 port |
| **gguf_varbuilder.rs**(236 行,按名取 QTensor + 形状/dtype 查询) | 同上目录 | 索引面设计参考 | OWNED |
| **gguf_tokenizer.rs / chat_template.rs / normal_bindings.rs**(键映射) | `packages/mistral.rs/mistralrs-core/src/gguf/` | 词表/模板/键映射的**交叉验证**第二源 | FROZEN |
| **`gguf.cuh` 的上游权威** | `repos/llama.cpp/ggml/src/ggml-cuda/`(本树内) | 类型定义(`block_q4_K` 等)与 `gguf-py` 解析器的**判据源** | REFERENCE |
| **llama.cpp 可执行锚** | `repos/llama.cpp/build-new/bin/llama-cli`(build 10945,`LD_LIBRARY_PATH=/home/div/Documents/codes/packages/llama.cpp/build-master2/bin`) | 端到端 greedy token 对拍;`llama-quantize` 可本地产 IQ 样本 | REFERENCE |
| **`gguf-py`** | `repos/llama.cpp/gguf-py` | 解析/词表的**独立第三源**(对拍用) | REFERENCE |

---

## 四、分层设计

```
┌─ L1 起点族(host,零设备) ──────────────────────────────────────────┐
│ crates/models/src/formats/gguf/{reader,meta,tokenizer,convention,source}.rs │
│   reader    : 容器解析 → GgufIndex{ kv, tensors: Vec<Entry{name,shape,ty,offset}> } │
│   meta      : KV → ModelSpec(维度事实;禁 config.json)                    │
│   tokenizer : 内嵌 KV → tokenizers::Tokenizer + ChatFormat                │
│   convention: GgufConvention: KeyConvention(blk.N.* ↔ layers.N.*)      │
│   source    : GgufSource: WeightSource(键 → **块原始字节**,形状翻转)     │
│ kind.rs 扩展: CheckpointKind::Gguf(+嗅探:magic "GGUF" ∧ 目录无非-GGUF 权重)│
├─ L2 终点(声明,零后端) ───────────────────────────────────────────┤
│ QuantPlanner(逐张量 plan 注入;刀 0)                                     │
│ QuantPlan::Gguf(GgmlQTy) → Linear 的 GGUF 臂(权重槽 = 块字节)            │
│ Layout: Gguf 走 Direct(源侧已按 [out,in] 报形状;块内布局 = 内核读序,零重排)│
├─ L3 内核族(crates/kernels) ────────────────────────────────────────┤
│ ops/gguf.rs       : KernelSpec 声明(quant/act-q8_1 + gemm;槽序 = .cu 签名)│
│ family/gguf.rs    : 族身份 + GgmlQTy ↔ quant_type 词表(唯一映射点)        │
│ cu/gguf/gguf.cuh  : port(block 结构 + dequant + vec_dot)                 │
│ cu/gguf/gguf_quant.cu : 激活 per-32 q8_1 量化(**无 malloc**)              │
│ cu/gguf/gguf_gemm.cu  : tile prefill + MMVQ decode(**无 malloc/props**)  │
├─ L4 引擎接线 ──────────────────────────────────────────────────────┤
│ engine/loader.rs: 按 detect() 选 GgufSource;spec 走 from_gguf;          │
│                   tokenizer 走内嵌;flavors/gguf-{9b,27b}.toml           │
└─ L5 锚与工具 ──────────────────────────────────────────────────────┘
  tools/gguf-dump.rs(对拍 gguf-py)/ 层内 dequant 金标测 / llama.cpp A/B 脚本
```

### L1 起点族(契约纪律)

- **源只管「键 → 字节」**:`GgufSource` **不做任何解量化**,返回压缩块字节
  (与现有源"源不懂设备、执行不懂格式"同律);
- **形状翻转律(★)**:GGML 张量 `ne = [ne0, ne1, …]`,其中 **`ne0` 是最内维** ⇒
  Linear 权重 `[in, out]`;**转成 owl 的 `[out, in]` 必须在源侧一次翻转**
  (safetensors 是 `[out, in]`)。此律是 REQ-GGUF-01 的机器判据之一;
- **逐张量 dtype 是事实**:`Entry.ty` 决定该张量的 plan(REQ-GGUF-04);
- **残缺/歧义响亮拒**:缺 KV、张量数不匹配 `block_count`、`n_kv` 越界、
  同一基座两套键面并存 —— 全部 `Err`,禁静默降级(`kind.rs` 既有纪律)。

### L2 终点(★ 本设计要求的一处架构改动:刀 0 —— **已落地**)

**问题**:现役 `Model::new(&spec, convention, plan, head_plan)` 把 plan 定成一
个**全局值**;而 GGUF 的 plan 是**逐张量**的 —— 实测两靶位:

```text
Qwen3.8-27B-Q4_K_M : Q4_K 193 / Q8_0 288 / Q6_K 17 / F32 353   (851 张)
Holo-3.1-9B-Q5_K_M : Q5_K / Q6_K / F32 —— 连**同一角色跨层都不同型**
                     (attn_qkv: Q5_K × 12 层 + Q6_K × 12 层)
```

⇒ "角色 + 形状"不足以定 plan(第二行直接证伪);必须**逐键**查表。

**落地形态**(与初稿的差异:用值语义 `PlanSource` 而非 `dyn QuantPlanner`
—— 零对象、零虚表,且让既有调用点**零改动**):

```rust
// module.rs §5(新)
pub trait PlanTable {                       // 混型源实现(GGUF)
    fn plan_at(&self, full_key: &str, out_dim: usize, in_dim: usize) -> QuantPlan;
    fn lm_head_key(&self) -> &str;           // 源命名事实(HF lm_head.weight / GGUF output.weight)
    fn take_unresolved(&self) -> Vec<String>; // 模型问过而表里没有的键(取走即清)
}

#[derive(Clone, Copy)]
pub enum PlanSource<'a> {
    Uniform(QuantPlan),                              // 既有一切档位(零开销)
    Keyed { table: &'a dyn PlanTable, keys: &'a dyn KeyConvention, layer: usize },
}
impl<'a> From<QuantPlan> for PlanSource<'a> { … }     // ← 既有调用点写字面量即可(单接口)
impl PlanSource<'a> {
    pub fn at_layer(self, i: usize) -> Self;                     // 逐层绑定层号
    pub fn of(&self, local: &str, out: usize, in: usize) -> QuantPlan;  // 层内局部键
    pub fn at(&self, full_key: &str, out: usize, in: usize) -> QuantPlan;
    pub fn head(&self, out: usize, in: usize) -> QuantPlan;      // 向表索取 lm_head 键
}
```

- **键算全复用 `KeyConvention`**(`PlanSource::of` → `keys.layer_key(layer, local)`),
  与 `Model::layout` 的 `map_keys` **同一实现**,不存在第二套键规则;
- 层的构造器签名只把 `plan: QuantPlan` 换成 `plan: impl Into<PlanSource<'a>>`
  ⇒ **全部既有调用点(含 40+ 测试)逐字不变**;
- `Model::new` 型改为 `-> Result<Model, ModelError>`(新增):构造期尾声
  取走 `take_unresolved()`,非空即拒 —— **禁静默落 F16**(那会让显存账静默失真;
  设计 I-2)。为什么必须在**构造期**判:槽形状与显存账在此定形。

**刀 0 实测门(2026-10-19)**:`owl-models` lib 217/217(GPU 真跑,`42:00.0`)、
`owl-engine` lib 53/54、全工作区 lib 全绿(见 §十 施工台账);
唯一红 = `gpu_prefill_bucket_matches_eager`,**在干净 `main @ ae2f8ae` 上同样红
= 既有基线红,非本轮引入**(已记录,不追)。

### L3 内核族(三条硬律 —— port 的**必改点**)

| # | 源(attention.rs)今天的做法 | owl 的硬律 | 改法 |
|---|---|---|---|
| **H1** | `gguf_gemm` 内部 `cudaMallocAsync(input_q8_1)` + `cudaFreeAsync` | **A5.2 一切分配过账本**;图捕获期禁 malloc(capture 白名单只放 Launch/Alloc) | q8_1 缓冲由 **`Pool` 供给**(参数传入);量化核与 GEMM 核**拆成两发** |
| **H2** | `gguf_gemm_device_cc()` 运行期 `cudaGetDeviceProperties` | 热路径禁 host 查询;arch 由 boot 期定 | 编译期 `__CUDA_ARCH__` 分支 + boot 期已解析的 arch 常量走 `HostVal`;**禁**热路径调用 |
| **H3** | `quant_type` 是一个**裸 int** switch(0..14) | 算子准入表里**点名**;裸 int 路由 = 不可审计 | `family/gguf.rs` 立 `GgmlQTy ↔ quant_type` **唯一映射表**,`KernelSpec` 表里逐型点名(≥1 条声明/型) |

补充约束:
- **单 TU 原则**:`gguf.cuh` 是 header-only(152 KB)+ 模板实例化 ⇒
  先只实例化 REQ-GGUF-04 的型(其余留 `#if 0` 或签名占位),避免 nvcc 时间失控(§八 R1);
- **`QK_K = 256`** 固定(禁 `GGML_QKK_64` 分支,减少一半实例);
- **`OutputFinite` 律**照旧;`DETERMINISM = Deterministic`;
- **捕获证据** `CaptureEvidence::NoHostWork`(H1/H2 改完后成立)。

**算子清单(3 条新声明;`ops/gguf.rs`)**:

| 声明 | 用途 | 关键槽 |
|---|---|---|
| `owl_gguf_quant_act` | 激活 per-32 → q8_1(MMVQ 喂料;从 `quantize_q8_1` 拆出) | `x[f32] → xq[U8]` + `kx`/`kx_pad` |
| `owl_gguf_gemm` | dequant-in-kernel GEMM(核内按 m 分派 MMVQ / tile) | `xq[U8], w[U8], out[f16], m, n, k, ty 域` |
| `owl_gguf_dequant_rows` ★ | **embed 查表**(按 id 取行 + 解量化;`token_embd` 在 GGUF 里是 Q4_K/Q5_K,**不是** f16 表) | `w[U8], ids, out[f16], rows, k, ty 域` |

★ **新发现的必需算子**(2026-10-19 侦察):两个靶位 `token_embd.weight` 分别
是 Q4_K / Q5_K ⇒ 词表查表自带解量化。`gguf.cuh` **无** `get_rows`/`dequantize_row_*`
(只有逐块 `dequantize_block_*` + `vec_dot_*`)⇒ 本核要自己写(用其块解量化函数,
约 50 行),不能靠"port 现成件"。`layout` 期 `Embedding` 需新增**量化表臂**
(槽 = `[vocab, row_bytes]` + 该 gather 节点);**tied + 量化表**的组合本轮
不受阻(two 靶位皆 untied:`output.weight` 在场),留作挂账。

### L4 引擎接线

```toml
# flavors/gguf-9b.toml(单卡 3090 Ti)
[model]
kind = "gguf"                 # 新增档;或 dir 探测自动认 GGUF
dir  = "/home/div/Documents/codes/models/mradermacher/Holo-3.1-9B-GGUF/Holo-3.1-9B.Q5_K_M.gguf"
[runtime]
max_seq = 32768
```

`model.dir` 今天语义 = **目录**;GGUF 是**单文件** ⇒ 需在 `model` 模块加
`ModelPath::{Dir, File}`(或 `kind` 决定解释法)。**词表与模板不再从 dir 找**。

---

## 五、端到端调用链(calltree;M1 目标形态)

```
apps/server/main → engine boot
└─ engine/loader.rs::load_model(path)
   ├─ formats::kind::detect(path)                      # 新增 Gguf 臂:magic + 单文件
   │  └─ formats::gguf::reader::open(path)             # header/KV/tensor table + mmap
   ├─ formats::gguf::meta::spec_from_gguf(&index)      # REQ-GGUF-02:qwen35.* KV → ModelSpec
   ├─ formats::gguf::tokenizer::from_gguf(&index)      # 内嵌词表 → tokenizers::Tokenizer
   ├─ Model::new(&spec, GgufConvention, &GgufPlan)     # 刀 0:planner 逐张量给 plan
   └─ formats::load::load_from_dir / load_gguf
      └─ eval_load(LoaderOps) → 逐 Want
         ├─ GgufSource::copy_native_into(键, 块字节)   # 零解量化,原样上卡
         └─ 权重槽 = U8 块张量(设备常驻,压缩态)

engine turn → layers::Linear::forward
├─ (decode m=1)  quant_act_q8_1(x) → x_q8_1(池 scratch)
│                gguf_gemm(x_q8_1, w_blocks, out, m, n, k, ty)   # MMVQ
└─ (prefill m≥8) 同一两发,tile 核分支                             # 分派在核内按 m
```

对拍链(REQ-GGUF-07):

```
tools/gguf-dump(owl 解析)  ⟷  gguf-py(第三源)         # 容器面
owl dequant(CPU 参考)      ⟷  candle-gb k_quants.rs    # 块面(逐位)
owl gguf_gemm(CUDA)        ⟷  owl dequant+cuBLAS(f16 参考)  # GEMM 面(allclose)
owl e2e greedy tokens      ⟷  llama-cli(build-new,同 GGUF)  # 端到端(逐 token)
```

---

## 六、不变量与纪律(缺一条静默错值)

| # | 不变量 | 判据 |
|---|---|---|
| I-1 | **形状翻转律**:GGML `ne`(ne0 = 内维) → owl `[out, in]` 只翻一次 | 索引面张量形状 × 全模型维度账自洽(`in_proj_qkv` 行数 = `2·nk·hd + nv·hv` 等) |
| I-2 | **逐张量 plan**:同模型混型必须逐键取 ty | 装载清单里每张量化张量都带自己的 ty;混型模型(27B)无一张被默认成 F16 |
| I-3 | **零解量化落盘/落 host**:权重永不变成 f16 常驻 | 装载期设备账 ≈ GGUF 文件大小 ×(1+ε),不是 ×3 |
| I-4 | **零账外分配**:`cu/gguf/*.cu` 无 `cudaMalloc*`/`cudaMemset*`/`cudaGetDeviceProperties` | 源码级 grep 门(CI 化) |
| I-5 | **块布局 = 内核读序**(零重排) | 装载字节区间 ≡ gguf-py 报的 offset/length 逐位 |
| I-6 | **词表/模板单一来源**:GGUF 在场时**禁**回落到 HF 目录(防"半自给"假象) | 同目录放一个**故意错**的 tokenizer.json,行为不变 |
| I-7 | **键面集合相等**:源索引键面 ≡ 声明键面(零缺键 / 零未用键) | 机器判据,非人眼(dump 两侧排序后 diff) |
| I-8 | **GGUF 三大非层键各有落点**:`token_embd`(量化表 → gather 解量化)/`output`(量化头 → GEMM,`PlanSource::head` 向表要 `lm_head_key() = "output.weight"`)/`output_norm`(F32 → cast)。缺一即静默错值 | 装载清单里三键各带自己的 plan;`output_norm` 走 f16 cast 臂 |
| I-9 | **合并权重的型一致性**:`attn_qkv` ⊕ `attn_gate` → `in_proj_qkvz`(owl 虚拟合并键)要求**两键同型**;实测两靶位皆同型(Q8_0/Q8_0 与 Q5_K/Q5_K),**异型 ⇒ 响亮拒**(不静默拼) | 装载前查两键 ty,不等即拒 |

---

## 七、刀序(M1)

> 纪律:每刀独立可验、可回滚;**先补观测面再调参**(引入新执行形态的老规矩);
> **app 层不变**:M1 全程不碰 apps/server 的 API 语义。

| 刀 | 内容 | 门 | 卡需求 |
|---|---|---|---|
| **刀 0** | `QuantPlanner` 注入(行为零变;既有调用点 → `UniformPlan`,逐位不变) | 全量既有套件绿(engine/models) | REQ-CODE-01 |
| **刀 1** | `formats/gguf/reader.rs` + `kind::detect` Gguf 臂(纯 host) | `tools/gguf-dump` vs `gguf-py`:9B 427 张 / 27B 851 张,名/形状/dtype/offset 零差异;残缺样本响亮拒 | REQ-GGUF-01 |
| **刀 2** | `tokenizer.rs`(内嵌词表 + chat template)+ 词表门 | tokenize 同 llama.cpp 逐 token;chat prefix 与现役 27B 配方同形可渲染 | REQ-GGUF-02 |
| **刀 3** | `meta.rs`(`ModelSpec::from_gguf`) | 9B/27B 与手写常量 & `from_ckpt` 对拍逐字段 | REQ-GGUF-02 |
| **刀 4** | `convention.rs` + `source.rs`(键映射 + 块字节 + 零解量化) | CPU face 整装;I-7 键面集合相等;I-5 字节区间逐位 | REQ-GGUF-03 |
| **刀 5** | **CPU 参考解量化**(全型;块结构 port 自 `gguf.cuh`/`k_quants.rs`) | 逐位对拍 `k_quants.rs`(F32/F16/Q8_0/Q4_K/Q5_K/Q6_K/IQ4_XS/IQ4_NL/IQ3_XXS/IQ3_S) | REQ-GGUF-04 |
| **刀 6** | CUDA:`cu/gguf/gguf.cuh` + `gguf_quant.cu` + `gguf_gemm.cu`(H1/H2/H3 改造)+ `ops/gguf.rs` 声明 | 对 CPU 参考 allclose;`gpu_gguf_gemm_matches_ref`(逐型);I-4 源码门;单层 9B 真机 | REQ-GGUF-05/06 |
| **刀 7** | `Linear` GGUF 臂 + `Model`/`engine` 接线 → **4B e2e 单卡**(靶位 2026-10-19 由 9B 改 4B,见 §1.4) | ①首 token 硬门 + 全序列软账(见 REQ-GGUF-07);②装载分相/设备账;③账本零未命名 + I-4 源码门;零 Xid — **实测见 §十三** | REQ-GGUF-07/08 |
| **刀 8** | 图捕获合规 + 桶/规格接线 + 分相账 | 捕获预检过;账本"未命名 0";decode 步时账 | REQ-GGUF-06 |

**M2**(27B):刀 9 = 27B Q4_K_M(混型)3090 对 + `flavors/gguf-27b.toml` 生产档 +
std 会战表;刀 10 = MoE 键面骨架(REQ-GGUF-09,仅覆盖 + 单层对拍)。

---

## 八、风险与挂账

| # | 风险 | 影响 | 缓解 |
|---|---|---|---|
| **R1** | `gguf.cuh` 152 KB header-only + 模板膨胀 ⇒ nvcc 时间可能 10 min+ 甚至 `cicc` 病态(mistral.rs 有 flashinfer 单文件 40 min 前例) | 构建循环变慢,拖垮迭代 | 只实例化 REQ-04 的型;`QK_K=256` 单分支;必要时**按型拆 TU**;测构建期增量闸(stamp_hit 同款) |
| **R2** | attention.rs 快照是**dp4a 路线**(`ggml_cuda_dp4a`),prefill 上限低于 fp16 tensor core | M2 的 prefill 读数可能不如现役 marlin 档 | 二期挂账:**prefill 回落 dequant+cuBLAS**(llama.cpp `should_use_mmq` 同款判据);M1/M2 先要正确性 |
| **R3** | IQ4/IQ3 **本机零样本** | REQ-GGUF-04 的 IQ 臂无法验收 | 二选一:①`hf` 下 mradermacher 的 Holo-3.1-9B IQ4_XS/IQ3_XXS;②`llama-quantize` 本地从 fp16 GGUF 产出(IQ 无 imatrix 精度差,但**解量化正确性**仍可对拍) |
| **R4** | `qwen35` 在 GGUF 侧是 **VL 模型**(带 mmproj,`rope.dimension_sections=[11,11,10,0]` 三轴) | 文本路径若与现役 27B 路径不同 ⇒ 新 rope 语义 | 先核 owl 现役 27B(同为 VL 权重)**已跑通**的路径;M1 只做文本(REQ-GGUF-10 挂 M3) |
| **R5** | GDN 键语义差异:`ssm_a`(A_log 取负?)、`ssm_dt.bias`、`ssm_conv1d` 布局、`ssm_norm` | 静默错值(GDN 深谷爆炸是历史教训) | 刀 5 的 CPU 参考 + 与 HF 权重逐位对拍(**同模型 HF 权重在场**:`Hcompany/Holo-3.1-9B`);用 `docs/arch/qwen38/算子盘点.md` 既有语义律 |
| **R6** | `attn_qkv` ⊕ `attn_gate` 两键 ↔ owl 的虚拟合并键 `in_proj_qkvz` | 装载期拼接错位 | 复用既有 `split_qkvz` 合成器(GGUF 源侧实现,不新增机制) |
| **R7** | 词表复刻细节(`token_type` 语义、`pre` = qwen2/qwen35 的预分词规则、特殊 token) | 静默分词错位 ⇒ 输出全错但"看起来能跑" | 刀 2 硬门:**逐 token 对拍** + `prompt_tokens` 计数对拍(AGENTS 测量纪律) |
| **R8** | 27B GGUF 是 Q4_K/Q8_0/Q6_K/F32 **混型** | planner 漏一个键 ⇒ 静默 F16 ⇒ 显存爆 | I-2/I-7 双门 + 装载期显存账对的"预期值"校验(18G ± ε) |
| **R9** | port 漏改一处 `cudaMallocAsync`/props | 图捕获炸 + A5.2 违约(且**可能只在长跑后暴露**) | I-4 源码级 grep 门,做成 CI/测试(`no_ledger_escape.rs` 同款先例:已有 `no_runtime_compile.rs`) |

**格式转化器(REQ-GGUF-11)= P2 挂账**:M1/M2 不做;"二次量化精度评估"是它的前置
(与 requirements §6.4 精度评估合流)。

---

## 九、施工台账(逐刀记)

| 刀 | 状态 | 门/实测 | 日期 |
|---|---|---|---|
| **刀 0** `PlanSource`/`PlanTable` 注入 | ✅ | `owl-models` lib **217/217**（GPU 真跑 `42:00.0`,串行）· `owl-engine` lib **53/54** · 全工作区 lib 全绿;唯一红 `gpu_prefill_bucket_matches_eager` = **既有基线红**(干净 `main @ ae2f8ae` 同样红,已定谳非本轮) | 2026-10-19 |
| **刀 1** GGUF 容器解析 + `GgmlTy` + `detect` Gguf 臂 | ✅ | **与 `gguf-py` 逐字段对拍全绿**:9B(427 张 / 57 KV,`F32×177 Q5_K×217 Q6_K×33`,6.01G)· 27B(851 张 / 39 KV,`F32×353 Q8_0×288 Q4_K×193 Q6_K×17`,17.66G)—— 版本/KV 键面集合/每键(类型·长度·内容指纹)/每张量(名·型·`ne` 序形状·相对偏移·字节数)全等;响亮拒门绿;全工作区 lib **绿(kernels 97 · models 220 · engine 54)** | 2026-10-19 |
| **刀 2** GGUF 内嵌词表 + 模板 | ✅ | ① **tokenize 对拍绿**:与 `llama-tokenize`(llama.cpp 自家实现、同一 GGUF)逐 token 相同,8 例 × 2 模型(中文/emoji/代码/缩写引号/空白边界),decode 往返无损;② **模板对拍绿**:两靶位内嵌模板与 HF 快照 `chat_template.jinja` **逐字节相同**(9B 7756B / 27B 8952B)+ 5 例渲染逐字节等;③ `owl-models` lib **225/225** · `server` bin **47/47** | 2026-10-19 |
| **刀 3** GGUF 超参 → `ModelSpec` | ✅ | **三方对拍绿**:4B(M1 靶位)/27B;27B 与手写预设 `qwen3_8_27b()` **逐字段相同**;超参↔形状 8 条互证;`dflash` 架构响亮拒。**2026-10-19 4B 接入时修正两处判据 bug**:
①**MTP 块不是解码层**(4B `block_count=33` = 32 解码 + 1 MTP;HF `mtp.*` ↔ GGUF `blk.32.nextn.*`);
②**层型表以键面为权威**,`full_attention_interval` 降为统计自检(4B 的块 32 是 full 但不属解码层 ⇒ interval 硬推会把 33 层全判错) | 2026-10-19 |
| 刀 4..8 | ⬜ | 刀 4a(键约定 + I-7 门)/ 刀 5(解量化)/ **刀 6a(CUDA 内核族)** 已完,见下 | |

**刀 4a ✅ 键约定 + I-7 键面覆盖门**(2026-10-19):
- `formats/gguf/convention.rs`:`GgufConvention`(local → `blk.N.*` / 顶层键)+ 新增
  `KeyConvention::head_key`(**第 4 类键**;untied lm_head = GGUF `output.weight`,
  HF `lm_head.weight`)⇒ `Embedding::layout_lm_head` 不再硬编码键名。
- **I-7 门**(零设备,只跑 `Model::layout`):4B 声明 **402 件**(24 件虚拟合并)、
  27B 声明 **803 件**(48 件),**零缺键**;反向检查:未用键全在 MTP 块(`blk.32.*`)。
- 副产物:`PlanTable::lm_head_key()` **删除** —— 头键改向 `KeyConvention::head_key` 索取
  (同一事实的第二个出口 = 接口不唯一,已消除)。

**刀 5 ✅ CPU 参考解量化**(`formats/gguf/dequant.rs`):
- 逐循环翻译自 `ggml-quants.c`;块布局(含 **Q6_K 的 `d` 在块尾**这个陷阱)写在模块头;
- **位级对拍门绿**:Q4_K / Q5_K / Q6_K / Q8_0 / F32 **五型 11 个张量**(含 27B 的
  17408×5120 大张量)与 `gguf-py/quants.py`(独立实现)**位级全等 max|Δ| == 0**;
- 主角油:①刀 6 CUDA 内核的金标;②刀 4c 装载期变换的执行面;
- 未接臂(IQ 族) **响亮拒**并点名挂账(R3)。

**刀 6a ✅ CUDA 内核族(内核侧)**(2026-10-19,用户令「直接上 GPU,别 CPU 解码」):

| 文件 | 内容 |
|---|---|
| `crates/kernels/cu/gguf/gguf.cuh` | attention.rs 快照**整文件搬运**(152 KB;头注登记溯源与改动点),`QK_K=256` 固定 |
| `crates/kernels/cu/gguf/gguf_quant.cu` | 激活 → q8_1(**H1 零分配**;缓冲由调用方给);2 实例(f16/bf16) |
| `crates/kernels/cu/gguf/gguf_gemm.cu` | dequant-in-kernel GEMM(**MMVQ**:warp 一行);共享 `__device__` 模板体 + **逐型字面 wrapper**(4 型);`out_off`/`out_ld` 支持分段合并写 |
| `crates/kernels/src/ops/gguf.rs` | **逐型一条声明**(H3:禁裸 int 路由);`Dtype::U8` 承载不透明块 |
| `crates/kernels/src/family/gguf.rs` §2 | 入口名 + `PORTED` + `entry_for(ty)`(**唯一映射**) + 发射几何(唯一产地);**数值 `kernel_code` 双表已删**(单一映射) |
| `crates/kernels/src/driver.rs` | `driver::gguf::{gemm(ty,m,n), quant_act(dt,rows,k_padded)}` + `resolve` 两臂(aux = `[ty,m,n]` / `[rows,k_padded]`) |

**R1 风险解除(实测)**:含 152 KB 头的两个 `.cu` 重编 = **1.82 s**(只实例化 4+2 个入口,
模板展开量小;不是 mistral.rs flashinfer 那种单文件病态)。
**门**:`op_tables` **10/10**(含**表↔`.cu` 签名逐参互证** + **R1「编出必已声明」**)、
`kernels` lib **97/97**、`models` lib gguf 族 **14/14**。

**刀 6b ✅ GPU 对拍门**(2026-10-19,用户令「直接上 GPU」):
`crates/models/src/formats/gguf/kernel_gate.rs` —— 设备链 = `GgufQuantActF16` → `GgufGemm`
(经完整解释器路径:`driver::resolve` → `with_pick` → dtype 门 → `alloc_out` → `lower_kernel` → `launch`):

```
[gguf-gpu] blk.0.attn_gate.weight(Q4_K, 4096×2560, m=3) max|Δ|/max|ref| = 3.7e-4
[gguf-gpu] blk.0.ssm_out.weight  (Q8_0, 2560×4096, m=4) max|Δ|/max|ref| = 2.9e-4
[gguf-gpu] blk.0.attn_qkv.weight (Q6_K, 8192×2560, m=2) max|Δ|/max|ref| = 4.2e-4
```

三条判据纪律(写进门的模块头):①参考侧**复刻同一激活量化**(MMVQ 的 8bit 损失是设计本身,
不复刻 = 拿算法固有损失当内核错;实测量化损失 2.6e-3 单列记账);②参考用 f64 累加;
③权重字节 = 文件原样(**本门只考内核算术,不含布局变换** —— 变换的门在刀 4c)。
**判读**:残差 3–4e-4 ≈ f16 舍入上界(2⁻¹¹ = 4.9e-4)⇒ 内核位级算术正确;任何错列/错块/
错位移都会给 O(1) 相对误差 ⇒ 阈值 2% 有足够判别力。

**刀 4b/4c ✅ 变换表 + 权重源**(2026-10-19):
| 文件 | 内容 |
|---|---|
| `formats/gguf/transform.rs` | §十一 六类变换的**逐键描述符**(`xf_for(键, dims)`)+ V-head 逆重排公式 + q8_1 **块粒度**置换表;5 单测(含公式互逆/往返) |
| `formats/gguf/source.rs` | `GgufSource: WeightSource` —— 量化键**字节原样**(I-3 压缩驻留,元素面**响亮拒**);行置换在字节面做(按行搬块,零重量化);`ssm_a` 的 `ln(-x)` 与 1-D 元素置换在元素面;`ssm_out` 的列置换**源不动字节**(由调用方在激活侧做) |

**门 = HF 快照对拍**(`models/Qwen/Qwen3.5-4B` 的 bf16 源,16 张量):

```
[blk.0.ssm_alpha/beta/dt_bias]  精确 0        (V-head 行置换 ✓)
[blk.0.ssm_a → A_log]          1.4e-8         (ln(-x) ✓)
[blk.0.ssm_conv1d / ssm_norm]  精确 0         (V 通道 / 不 +1 ✓)
[blk.0.attn_norm / post / q_norm / k_norm / output_norm]  精确 0  (减 1 后 ✓)
[blk.0.attn_qkv]  1.5e-2 · [attn_gate] 4.1e-2 · [blk.3.attn_v] 6.3e-2   (量化噪声)
```

**容差定档依据**(实测,写进测试注释):量化张量的最大单元素相对误差实测
1.5e-2(Q6_K)/ 4.1e-2(Q4_K)/ 6.3e-2(Q5_K)—— **量化噪声**(小幅度元素落粗子块),
与"错置换"签名不同(错置换/错段偏移给 **O(0.5–1.0)**,整行内容换了)⇒ 阈值 1e-1。
原生小件与四个 norm 保持 1e-5 级(**实测全部精确 0**)。

**刀 7 已完成(2026-10-19;逐项证据见 §十三)**:
- ✅ `QuantPlan::Gguf(ty)` + `Linear` 臂(U8 块槽 `[out, 块行字节]`;forward = quant_act + gemm 两发;
  激活按 F16/BF16 分 op_id;形态门 = 内维被块元素数整除)
- ✅ `load_from_dir` 接 `GgufSource`(装载门 = 450 条 / 设备 2.70 GiB ≤ 文件 2.81 GiB)
- ✅ **`norm_delta` 构造期事实**挂 `SourceFacts`(`norm_baked_plus_one`;GGUF 档 `×w`)
- ✅ GDN **两 Linear**(`in_proj_qkv`/`in_proj_z`,4B 24/24 层异型)+ **`ssm_out` 列置换**
  (**落点 A2 = 折进激活量化核**;用户裁决,见 §十三 与 §十二 裁决记录)
- ✅ tied-embed 臂(`owl_gguf_dequant_rows`;**嵌入查表与 lm_head 同一张 Q6_K 表**)
- ✅ 引擎接线(`ModelKind::Gguf` + `load_gguf` + flavor `gguf4b` + server 实测)与 4B 单卡 e2e
  (首 token 6/6 与 HF 快照一致;全 12 token 4/6,两处分叉实测为 oracle 侧 near-tie)
- ⬜ 捕获合规(图清单已见到 `owl_gguf_*` 入图;formal 审计 = 刀 8)

**★ 2026-10-19 重大发现:qwen35 GGUF 布局变换表(§十一)**。转换器
(`llama.cpp/conversion/qwen.py`)写文件时做了**六类变换**,逐条已**实测实证**
(4B × HF 快照)并对照转换器源码定谳:

1. **V-head 反重排**(GDN 层;`nk != nv` 时):`h = (g % nk) * vpr + (g / nk)`;
   实测 `alpha/beta/dt_bias` 逆变换后**精确 0**、`A_log` maxΔ 6e-8;
2. **`A_log` → `-exp(A_log)`**(读时 `ln(-x)`);
3. **norm `+1`**(除 `linear_attn.norm`):实测 `GGUF-(HF+1) == 0` 精确;
   而 `ssm_norm` 直拷精确 0—— 与转换器逐字一致。**解法零变换**:owl 的
   `new_add_one` 就是 `×(1+w)` ⇒ GGUF 档一律 `w_off=false`(新增 `norm_delta` 事实);
4. `conv1d` squeeze + V 通道段重排;5. `in_proj_qkvz` 拆两键;
6. `out_proj` **列**重排(内维)。

**代价分层已钉死**(§11.4):行粒度变换可**字节级**做(行 = 多个完整量化块);
`ssm_out` 的**列**变换跨块 ⇒ 走**激活侧等价置换**(精确无损);F32 小件
(`ssm_a`/`dt_bias`/`conv1d`)host 变换;norm 走语义法(零变换)。
⇒ **权重字节零改动 + 轮内零额外开销**(除 `ssm_out` 一次 gather,可入图)。

**同轮作废/变更**:① I-9(合并键同型)**作废** —— 4B 的 24 个 GDN 层全部
`attn_qkv` Q6_K + `attn_gate` Q4_K **异型** ⇒ 改**分段合并**(分段臂);
② I-8(tied + 量化表)**入 M1**(4B `tie_word_embeddings=True` 且无 `output.weight`)。

**纪律教训(值得复用)**:**不要在 CPU 里硬解排列**(O(n²) python 逐行匹配跑到挂死)。
正解 = **读转换器源码**(`conversion/qwen.py` 里 `_reorder_v_heads` 就是权威),
再用 2–3 个 F32 小张量做**廉价实证**交叉验证。

**刀 0 改动面**(13 文件,+204/−65):`module.rs`(+93,新类型) · `model.rs` ·
`layers/{attention,decoder,gdn,mlp}.rs` · `vlm.rs` · `engine/src/loader.rs` · 测试 3 处。
**关键实现手法**:①`impl From<QuantPlan> for PlanSource` ⇒ 既有调用点零改动;
②层构造器 `plan: impl Into<PlanSource<'a>>`(静态分发,合法于 REQ-CODE-01);
③`Model::new -> Result`(只在 Keyed 臂可能失败;Uniform 臂恒 Ok)。

**刀 1 新增件**:

| 文件 | 内容 |
|---|---|
| `crates/kernels/src/family/gguf.rs` | `GgmlTy`(35 型)+ 块几何 + 内核码双射 + `M1_VALIDATED`;**5 个单测**(表逐条对 `GGML_QUANT_SIZES` 硬编码、双射、几何自洽) |
| `crates/models/src/formats/gguf/reader.rs` | 容器解析(header/KV/张量表/对齐/mmap + 防御上限 + 重叠/越界/对齐三道对账)+ `KvTable` 类型校验取值器族 |
| `crates/models/src/formats/gguf_dump.py` | 对拍协议另一端(canonical 编码 + FNV-1a64;TSV 三行式) |
| `crates/models/src/formats/kind.rs` | `CheckpointKind::Gguf` + 四族嗅探(目录/单文件双形态;混存与多 gguf **响亮拒**) |
| `crates/models/src/formats/load.rs` | Gguf 臂 = **响亮拒**(源未接线的诚实形态;落点刀 4) |

> 为什么对拍用 **canonical 编码 + 指纹**而不是逐字比渲染串:浮点格式化
> (Rust `f32::to_string` vs Python `repr`)、字符串转义两边不同 ⇒ 直接比字符串
> 会把"格式差异"演成"假红"。改为**双方实现同一份字节编码**(整数二补码 / 浮点
> IEEE 位型 / 长度前缀)再比指纹 ⇒ 分歧必是事实分歧。代价 = 词表级大数组也逐字节比
> (两文件 ~21s,可接受)。

**刀 2 新增件**:

| 文件 | 内容 |
|---|---|
| `crates/models/src/formats/gguf/tokenizer.rs` | 内嵌词表 → `tokenizers::Tokenizer`(BPE 臂)+ `VocabModel`/`SpecialIds`/`GgufVocab`;预分词正则表**逐字对 `src/llama-vocab.cpp`**(`qwen35`/`qwen2` 同表);未知 `pre` / Unigram / 非 gpt2 = **响亮拒** |
| `crates/models/src/formats/gguf_dump.py` 之前 | —— |
| `apps/server/src/chat_tpl.rs` | 抽出 `ChatTpl::from_str`(唯一编译入口,`load_with` 与 `load_gguf` 的共同下游)+ `load_gguf`(模板取自 `tokenizer.chat_template` KV)+ `load_snapshot_or_gguf`(**来源优先级唯一产地**) |
| `apps/server/src/main.rs` | 装配点改走 `load_snapshot_or_gguf`(显式 `agent.template_path` > GGUF 内嵌 > 快照) |

**刀 3 新增件**: `crates/models/src/formats/gguf/meta.rs`(`spec_from_gguf`)。

**两条实测事实(入库,影响后续)**:

1. **GGUF 内嵌模板 ≡ HF 快照模板**(逐字节,两靶位)⇒ 既有 HF oracle 金标门
   (`chat_tpl` G0)按构造转移到 GGUF 来源;**但** 27B GGUF(ggml-org 官方 8952B 模板)
   ≠ owl 现役 27B 生产档(cyankiwi 目录下 froggeric 28478B `chat_template.jinja`)
   —— 同一个 27B 的两套模板,若用 GGUF 档跑 27B,前缀与生产档**必然不同**(不是 bug,是事实;
   写进 M2 的对照纪律)。cyankiwi 的 `tokenizer_config.json.chat_template` 与官方同源(8952B)。
2. **`tokenizer.ggml.tokens` 含 276 个占位**(unused 243 + control 27 + user_defined 6),
   HF `vocab` 只有 248044 项 ⇒ 词表大小 248320 是 GGUF 事实(embed 行数用它)。


---

## 十一、qwen35 GGUF 布局变换表(★ 2026-10-19 实证定谳)

> **为什么必须有这一章**:GGUF 转换器(`llama.cpp/conversion/qwen.py`)写文件时对
> qwen35 做了**六类变换**;读回来必须逐类**逆变换**,否则**静默错值**
> (GDN 深谷爆炸那类)。本节每条都有**实测证据**(4B Q4_K_M × 其 HF 快照
> `models/Qwen/Qwen3.5-4B`),不是推测。

### 11.1 变换来源(权威 = 转换器源码)

`conversion/qwen.py::Qwen35Model.modify_tensors` 与 `_LinearAttentionVReorderBase`:

| HF 张量 | 转换器做什么 | 读回来的逆变换 |
|---|---|---|
| `.A_log` | `-exp(A_log)` | `A_log = ln(-x)` |
| `.dt_bias` | 改名 `.dt_proj.bias` | (键名映射) |
| `conv1d` | `squeeze()` + **V 通道段重排** | squeeze 逆 + V 段反重排 |
| **`norm.weight`(非 `linear_attn.norm`)** | **`+1`** | 减 1 **或** 用 `×(1+w)` 语义 |
| `in_proj_qkvz`(合并形态) | 拆成 `attn_qkv` + `attn_gate` 两键 | (虚拟合并键逆) |
| `in_proj_qkv`(分形态) | **仅 V 行**按 `_reorder_v_heads` | 仅 V 段行反重排 |
| `in_proj_z` | 行重排 | 行反重排 |
| `in_proj_a`/`in_proj_b` | 行重排(head_dim=1) | 行反重排 |
| `out_proj` | **列**重排(内维) | 列反重排 / **激活侧等价置换** |

### 11.2 V-head 反重排公式(实测反解,与转换器一致)

转换器把 HF 的 **grouped** 序(`[G0_v0..v(r-1), G1_v0.., ...]`)换成 ggml 的
**tiled** 序(`[G0_v0, G1_v0, ..., G0_v1, G1_v1, ...]`)—— 即 `[nk, vpr, hd] →permute→ [vpr, nk, hd]`。

```text
vpr = nv / nk                 (4B: 32/16 = 2;  27B: 48/16 = 3)
HF 序 h  ↔  GGUF 序 g:
    g = (h % vpr) * nk + (h / vpr)        # HF → GGUF(写)
    h = (g % nk) * vpr + (g / nk)         # GGUF → HF(读,逆变换)
```

**实测证据**(4B layer 0,逐行精确匹配):

| 张量 | 直拷 | 逆变换后 |
|---|---|---|
| `ssm_alpha`(in_proj_a) | rel 9.6e-1 ✗ | **精确 0** ✓ |
| `ssm_beta`(in_proj_b) | rel 1.3e+0 ✗ | **精确 0** ✓ |
| `ssm_dt.bias` | rel 1.2e+0 ✗ | **精确 0** ✓ |
| `ssm_a` → `ln(-x)` | rel 1.1e+0 ✗ | maxΔ **6e-8** ✓ |
| `attn_qkv` V 段 | rel 1.09 ✗ | rel 1.6e-2(= 量化噪声)✓ |
| `attn_gate`(in_proj_z) | rel 1.04 ✗ | rel 4.1e-2(量化)✓ |
| `ssm_out`(列) | rel 1.11 ✗ | rel 3.9e-3 ✓ |
| `ssm_norm` | **精确 0** ✓ | —— |
| full 层 `attn_q/k/v/output` | rel 2–6e-2(= 量化噪声)✓ | —— |

### 11.3 norm 的 `+1`(第二条静默错值陷阱)

**实测**(4B):`attn_norm` / `post_attention_norm` / `attn_q_norm` / `output_norm`:
`GGUF − (HF + 1) == 0`(**精确 0**);而 `ssm_norm`(**`linear_attn.norm`**)直拷 == 0
⇒ **转换器只给前者加 1**(与源码逐字一致:`name.endswith("norm.weight") and not
name.endswith("linear_attn.norm.weight")`)。

**那 owl 怎么办(重要)**:HF 的 Qwen3.5 norm 是 **zero-centered**(`gamma = 1 + delta`,
Gemma 式)—— 这正是 owl `RmsNorm::new_add_one`(`×(1+w)`)的来处,而 HF 路径
`input/post/q/k` 全用 `new_add_one` ✓。GGUF 已把 `w+1` **烧进数值** ⇒
**GGUF 档一律用 `×w`(`w_off=false`)**,零变换、零新算子:

| norm | HF 值(4B) | GGUF 值 | owl HF 路径 | **owl GGUF 路径** |
|---|---|---|---|---|
| `input_layernorm` | 0.667 | **1.667** | `×（1+w)` | **`×w`** |
| `post_attention_norm` | −0.996 | **0.0039** | `×（1+w)` | **`×w`** |
| `attn_q_norm`/`k_norm` | 0.336 | **1.336** | `×（1+w)` | **`×w`** |
| `output_norm` | −0.291 | **0.709** | `×（1+w)` | **`×w`** |
| **`ssm_norm`** | 0.884 | 0.884 | `×w` | **`×w`** ✓ 两者同 |

⇒ 新增一条构造期事实 `norm_delta: bool`(源的事实:HF=true / GGUF=false),
由 `PlanSource` 旁路或同构的 `SourceFacts` 携带(刀 7 落地)。

### 11.4 逆变换的**代价分层**(哪些能字节级做,哪些不能)

关键几何:GGML 量化块沿 **`ne[0]`(内维)**;`QK_K = 256`,而 `hv = 128`
⇒ **一个 v-head = 128 个元素 = 半个块**。于是:

| 变换 | 粒度 | 能否字节级(零重量化) | 采用实现 |
|---|---|---|---|
| `attn_qkv` V 段 / `attn_gate` / `alpha` / `beta` | **行**(行 = hidden 维 = 多个完整块) | ✅ **可以**(块组搬移) | 装载期字节级行置换(`GgufSource` 的 row-perm 能力) |
| `ssm_out` | **列**(内维 = v-head×128) | ❌ 跨块 ⇒ 需重量化 | **激活侧等价置换**(精确、无损):**折进激活量化核**(`owl_gguf_quant_act_*_tiled`,零额外发射);原生件走装载期元素面置换 |
| `ssm_a`(`ln(-x)`)/`ssm_dt.bias`/`conv1d` | 逐元素/通道 | ✅ 全是 **F32** | 装载期 host 变换(极小) |
| norm `+1` | 逐元素 | ✅ **语义法**(`w_off`)零变换 | 构造期 `norm_delta` |

**为什么不用"全量 dequant→permute→requant"**:二次量化在 v-head 边界上引入额外误差,
且毁掉「零解量化」红线(设计 I-3)。上面四类分派做到**权重字节零改动**
(除 F32 小件)+ **轮内零额外开销**(除 `ssm_out` 一次 `[T, nv·hv]` gather,可入图)。

### 11.5 与设计正文的差异(必须回写)

1. **I-9(合并键同型)作废**:4B 的 24 个 GDN 层全是 `attn_qkv` Q6_K +
   `attn_gate` Q4_K(**异型**)⇒ “合并即要求同型”会把 M1 靶位直接拒掉。
   改为:**分段合并**(每段自带 ggml 型;GEMM 逐段发射写进输出列窗)。
   27B 实测 48 层全同型(Q8_0/Q8_0)⇒ 分段臂对它退化为单段。
2. **I-8(tied + 量化表)入 M1**:4B **无 `output.weight`** 且 `tie_word_embeddings=True`
   ⇒ `token_embd`(Q6_K)同时充当 embed 与 lm_head —— 原来挂账的形态现在是**必做**。
3. **MTP 块**:4B 有 `blk.32.nextn.*`(HF `mtp.*`)⇒ **`blk.32` 不是解码层**,
   层型表推导必须排除它(M1 不载 MTP)。
4. **`full_attention_interval` 不是层型表权威**:4B 的 `blk.32` 是 full 但不属
   解码层;32 个解码层里 full 恰为 `i%4==3`(8 个)✓。改为:**键面是权威**,
   interval 只做统计自检(不符记 issues,不拒)。

---

## 十二、裁决记录

| 日期 | 裁决 | 依据 |
|---|---|---|
| 2026-10-19 | 立项;两段式(M1 9B 单卡 → M2 27B 双卡) | 用户 |
| 2026-10-19 | **一步到位 MMQ**(无 f16 中间态) | 用户;§1.3 显存算术 |
| 2026-10-19 | GGUF **标准自给**(超参 + 内嵌词表 + 模板全读);借力 mistral.rs / xinfer | 用户 |
| 2026-10-19 | 首轮量化型 = Q4_K_M / IQ4 / IQ3(+ 目标文件实际含型) | 用户 |
| 2026-10-19 | 引入 `QuantPlanner`(逐张量 plan)作刀 0;GGUF 走**同一** MMQ 路径,小档不另开 dequant 支路 | 本设计 §L2/§1.3 |


---

## 十三、刀 7 施工实录(2026-10-19;A2 落点 · 引擎接线 · e2e 定谳)

> 本章是**结果面**:怎么落的、拿什么门验的、实测数字与踩到的坑。

### 13.1 `ssm_out` 列置换的落点(**用户裁决 = A2**)

候选与取舍(讨论全在会话里,结论记此):

| 落点 | 形态 | 节点/步 | 成本 |
|---|---|---|---|
| A1 独立算子(初稿) | `gather(gated)` → `quant_act` → `gemm` | +1/层 | launch-bound(4B ~+20-40µs,27B ~+50-100µs) |
| **A2 折进激活量化(选定)** | `quant_act(gated, tbl)` → `gemm` | **0** | ≈0(每 warp 一次表广播读) |
| A3 折进 `gdn.norm_act` 写址 | 归一化核按 tiled 写 | 0 | 隐式耦合(把下游权重布局塞进归一化核)—— 否决 |
| A4 装载期按 32 元素子块重排 + scale 重编码 | 字节级 | 0 | **有损**(跨超块要改 `d/dmin`)—— 否决 |

**A2 的形态**:`owl_gguf_quant_act_{f16,bf16}_tiled`(与普通臂**同几何**,多一个 `tbl[U32]` 槽)。
语义:目标块 `j` 的 32 个元素取自源块 `tbl[j]` ⇒ 与"先置换再量化"**逐位等价**
(置换单元 = 32 元素块,块内容不变 ⇒ `d`/`sum`/`q` 全不变)。

**表**:`U32[k/32]`,由 `transform::vhead_q8_blocks`(唯一公式源)生成,经**源侧虚拟键**
`blk.N.ssm_out.perm` 常驻显存(源合成字节,装载门逐字校验 == 公式输出)。

**方向(★ 最易反的一处)**:`vhead_deperm`(行置换,文件→HF)与 `vhead_q8_blocks`
(激活侧,**它的逆表**)是**两个不同的数组**;单测 `vhead_perm_is_inverse_of_deperm`
把两者钉成互逆。写反不会报错,只会整列错位。

**原生臂**(`ssm_out` 为 F16/BF16 的档):源侧装载期做**元素面**列置换
(`transform::col_deperm_in_place`;无损、零运行期成本)—— 与量化臂同一条不变式:
「装载后的权重序 ≡ 喂进去的激活序」。

**门(多层)**:
1. `transform::col_deperm_exact_and_round_trip`(手算小例 + 互逆往返 + 形态门);
2. `kernel_gate::gpu_gguf_ssm_out_tiled_act_matches_grouped_reference`:
   ① **位级**:`tiled(x,tbl)` ≡ `plain(host 置换后的 x)` —— 13824 B **逐字节相同**,
   且**必须 ≠ 自然序期望**(证明真置换生效,防"表退化成恒等"的假绿);
   ② **数值**:tiled + **文件权重字节** ≡ CPU 参考(同权重解量化 + 内维反重排)⇒ `4.70e-4`
   (≈ f16 舍入上界;量化损失 2.57e-3 单列记账)。

### 13.2 四处**实测抓出**的坑(都已修 / 已入档)

| # | 坑 | 症状 | 修法 |
|---|---|---|---|
| ① | **q8_1 激活块是 36 B 不是 40 B** | 模型层按 gguf-py 的**张量型** `Q8_1`(40B)声明激活缓冲 ⇒ 多占 11% 且**留下未写的脏尾**;位级门因此**时绿时红**(单独跑时两侧尾巴恰好都是零) | `family::gguf::Q8_1_BLOCK_BYTES = 36`(=`half2 ds` 2+2 + `qs[32]`),与 `.cuh` 的 `static_assert(sizeof(block_q8_1) == 2*sizeof(ggml_fp16_t) + QK8_0)` 由单测互锁;模型层改引用 |
| ② | **`KvTable::opt_u32s` 每次重解整个数组** | 词表构建里逐 token 调 `tt_of(i)` ⇒ **O(n²)**:248k token 的 `token_type` 循环花掉 **67s**(总启动 68/91s) | 一次读完再索引;`tt_of` 加注警示。**启动分相上屏**(`[gguf] 词表分相` / `BPE 分相` / `装载分相`)—— 没有分相就看不见这类"静默变慢" |
| ③ | **对拍口径**:`submit_session` 默认 `chat_wrap` | 与 llama.cpp 裸 prompt(`/completion`)对拍时,owl 侧其实喂的是**家族 ChatML** ⇒ 首 token = ` thinking`(模板态)vs `" Paris"`(裸态),两边都"连贯"却是两道不同的题 | 用既有的诊断旋钮 `knobs.raw_completion = true`(裸续写);写进 `tests_gguf` 的配置注释 |
| ④ | **量化档不能拿"逐 token 全序列"做硬门** | Q4_K_M 与 f16 两条实现路径在 **near-tie** 处必然分叉(实测 oracle 侧 top-2 间隔 **0.14 nat** / **0.013 nat**,且两次分叉**各由不同一侧**命中 oracle argmax) | 门改口径:**首 token 硬门**(预填充链的 logits 顶)+ **全序列软账**(≥ 半数)+ 分叉点**记 margin** |

### 13.3 引擎接线

| 件 | 内容 |
|---|---|
| `ModelKind::Gguf` | `model.kind = "gguf"`;`model.dir` = **`.gguf` 单文件**(不是目录) |
| `ModelLoader::load_gguf` | ① `GgufSource::open`(容器 + spec 同源)② `GgufPlan` + `PlanSource::keyed(SourceFacts::GGUF)` + `Model::new`(**构造期取走未解析键,非空即拒**)③ `load_from_dir`(压缩块字节原样)④ 词表 `build_vocab`(文件自给;eos 吃文件给的 id)⑤ rope `rope_facts`(KV 三件套)⑥ 装载分相上屏 |
| `ModelLoader::load_qwen35_hf` | **通用 Qwen3.5 HF 目录装载**(预设入参化;rope 三件套显式给)—— 为交叉源对拍而生,顺带让任意 Qwen3.5 HF 档能上引擎 |
| `flavors/gguf4b.toml` | M1 靶位档(单卡 3090 Ti;`max_seq 32768` / `gdn_slots 1` / greedy;**刻意不设 `allowed_ops` 白名单** —— 生产硬化另刀) |
| server 实测 | `target/release/server --flavor gguf4b <用户档>` 起机 → `/v1/chat/completions` 正常(**thinking 走 `reasoning_content`**,与模板一致);图清单里 `owl_gguf_gemm_q4k ×124 / q6k ×76 / q5k ×8 / owl_gguf_quant_act_f16 ×177` —— 真入图 |

### 13.4 4B 单卡 e2e 实测(`OWL_TEST_DEVICE=0000:42:00.0`,3090 Ti)

```text
装载 2.05s(容器+维度 0.07 · 计划/构造 0.00 · 权重 0.70 · 词表/rope 1.05)
设备 live 2.70 GiB vs 文件 2.81 GiB(比 0.951;450 条 = 402 键 + 24 虚拟合并 + 24 列序表)
裸文本贪心 decode ≈ 105 tok/s(单会话、无投机、无 decode 图)
prompt ids = [760, 6511, 314, 9338, 369] ≡ llama.cpp `/tokenize`(逐位)
```

**std 五点三轮(`packages/llm_speedtest --flavor std`;单 3090 Ti,`41:00.0`)**
—— 原始数据 `docs/bench/data/gguf4b-std-2026-10-19.json`:

| 点 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|
| prefill t/s(三轮) | 330 / 324 / 327 | 325 / 326 / 329 | 333 / 324 / 329 | 331 / 328 / 327 | 328 / 330 / 326 |
| decode t/s | 145 / 149 / 149 | 140 / 149 / 149 | 147 / 144 / 142 | 144 / 147 / 142 | 143 / 142 / 141 |
| TTFT ms | ~1553 | ~3148 | ~6146 | ~12400 | ~25000 |

`15/15 成功` · prefill **327.7 t/s**(323.7–333.3)· decode **144.9 t/s**(139.7–149.5)。

**两条口径纪律(读这两行数字前必读)**:
1. **decode 两个数**:客户端 144.9 t/s(7.0ms/token,含 SSE 往返)vs **引擎内 `decode.step` 5.62ms**(≈178 t/s,p50 5.57/p99 5.90;
   来自 `gpu_gguf_4b_smoke` 的步时账)。跨引擎可比的口径 = **客户端**那个;
   判"引擎还有多少余量"用**引擎内**那个(`docs/dev/profiling.md` 的老教训)。
2. **prefill 328 t/s = R2 挂账的量化证据**:当前只有 MMVQ(dequant-in-kernel),
   prefill tile 臂未移植。M2 若要与 llama.cpp/vLLM 同台比 prefill,这条必须补
   (设计 §八 R2 给出方向:prefill 回落 dequant+cuBLAS 或 port tile GEMM)。

**测速档口径(踩过的坑,写进 flavor 注释了)**:`pool_tokens = max_seq`(全 ctx 常驻)
+ `prefix_cache = false`。首轮测速用 `pool_tokens = 8192` + `prefix_cache = true`,
8448-token 那一点撞宿主层预算(`宿主层越界:537919488 > 536870912`)⇒ 请求被弃置、
测速点降级为"本地估算"(prefill 假 10727 t/s / decode 0)—— **看到"prefill 边界降级"
就先查池/宿主预算,别怀疑内核**。

**交叉源门**(同 dims:GGUF Q4_K_M ↔ HF 快照 f16;6 题 × 12 token):

| 题 | 首 token | 全 12 token |
|---|---|---|
| `The capital of France is` | ✅ | 分叉(第 4 token;oracle `A`−1.400 vs `The`−1.543 = **0.14 nat**) |
| `Water is composed of hydrogen and` | ✅ | 分叉(第 3 token;oracle `Which`−2.425 vs `If`−2.438 = **0.013 nat**) |
| 其余四题 | ✅ | ✅ 逐字相同 |

**判读**:首 token 全绿 ⇒ 预填充链(32 层 × 全部权重解释 × 最终 norm × tied 头)无误;
两次分叉各由**不同一侧**命中 oracle 的 argmax ⇒ 分叉是 near-tie 噪声,不是"某一侧错"。
(旁证:`gpu_gguf_gemm_matches_cpu_reference` 现覆盖 4B 全部五型 Q4_K/Q5_K/Q6_K/Q8_0 —— 残差 3.4e-4~9.2e-4。)

### 13.5 刀 8 面已闭两项(2026-10-19 收尾)

| 项 | 门 | 位置 |
|---|---|---|
| **I-4 源码门**(零 `cudaMalloc*`/`cudaMemset*`/`cudaGetDeviceProperties`) | 扫描 `cu/gguf/*.cu|.cuh`(**剥注释** —— 本族头注引用了这些禁词讲 H1/H2 规则,引文不算违规)+ 判据灵敏度反证 | `crates/kernels/tests/no_runtime_compile.rs::gguf_kernels_never_allocate_or_query_device`(4 源文件全绿) |
| **账本"未命名 0"** | 跑完一轮(装载 + 图捕获 + decode)后查块账:`(未命名!)` 零出现;全表 JSON 里 24 张列序表(`wt.blk.N.ssm_out.perm`)与 tied 表都按名在册 | `crates/engine/src/tests_gguf.rs::gpu_gguf_4b_smoke`(在册 672 块 / 3202.4 MiB,零未命名) |

| **捕获预检**(两道:树审计 Htod + face 能力) | `tests_gguf::gpu_gguf_4b_smoke` 断言 `PlanOutcome::Captured`(降级则 panic 并打印 reason) | 同上(实测 `PlanOutcome::Captured`) |
| **decode 步时账** | 同上:查 `decode.step`(引擎既有 `mrec`)—— 样本数 ≥ 生成步数,并上屏 avg/p50/p99 | 实测 23 步 · avg **5.617ms** · p50 5.566 / p99 5.896(≈178 t/s 引擎内真值) |
| **I-4 源码门** | `no_runtime_compile.rs::gguf_kernels_never_allocate_or_query_device`(见上表首行) | 4 源文件零违规 |

**刀 8 至此三面全闭**(源码门 + 运行期账本 + 捕获态/步时账)。仍挂:
`decode` 的**逐相**拆账(step 内的 graph/fetch/sample 分相;现只有总步时)。

### 13.6 生产面的两把小刀(2026-10-19)

| 件 | 内容 |
|---|---|
| **GGUF 实例名回落** | `model.name` 不给时,`kind = "gguf"` 档回落到**文件名词干**(`…-Q4_K_M.gguf` → `Qwen_Qwen3.5-4B-Q4_K_M`)。为什么必须:GGUF 是单文件档,同一目录常并排放多个量化档,固定缺省名会让 `/health`/`/v1/models`/日志全显示别人的名字(实测首版报 `model=qwen3.5-0.8b`;另一档 `base.toml` 会把名字带成 `qwen3.8-27b`)。唯一产地 = `ServerConfig::default_model_name` |
| **`allowed_ops` 白名单(生成 ≠ 手抄)** | 30 项,从**真实发射账**取并集(`[cuda] trace_ring` → `/debug/launches` → 逐请求累积;⚠️ ring 绕圈,单次 dump 实测漏 5 个解码臂)。台账入库 `owl_kernels::ops::PROD_GGUF4B_2026_10_19`,并由 `apps/server/tests/ops_whitelist.rs` **两处锁死**:① flavor 清单 ≡ 台账 ② 每项已建表 + **过池写集穷举审计**(实测 30 项全过)。换文件/换臂 ⇒ 重跑收账流程(步骤写在 flavor 注释里) |

### 13.7 对拍工具链备忘(给下次省时间)

- **`llama-cli` 已无裸续写口径**(本树 build 10945:`-no-cnv` 不存在、`-p` 一律走会话/模板;
  `llama-completion` 空跑);**裸 prompt 用 `llama-server`**:
  `POST /completion`(原生,`temperature:0` = greedy)或 `/v1/completions`(`logprobs:5`
  拿 top-5 与 margin —— 判 near-tie 的关键证据);
- `llama-server` 用 `CUDA_VISIBLE_DEVICES=<UUID>` 钉卡;收工时**按 PID kill**
  (`pkill -f "release/server --flavor gguf4b"` 会连自己的命令行一起匹配 = 自杀,本仓老坑);
- 想知道"模型病 vs 接线病",先跑 **owl 自己两源对拍**(HF 快照在场时),再上 llama.cpp —— 两源不一致 = 格式路径的错,两源一致而 oracle 不一致 = 语义/near-tie 问题。

---

## 十四、刀 9 / M2 开工(2026-10-19):27B 单卡跑通 + 两条**静默错值**定谳

> M2 的原计划是"27B 上 3090 对(TP2)"。施工中的第一刀改了顺序:**先单卡**
> —— 27B 的 17.7 GiB 压缩驻留 + 小池在 24G 卡上装得下(实测 17.69 GiB + 池),
> 而 TP 要动"列切 × 量化块"的根,风险面大得多。**格式面的正确性不该和 TP 一锅炖**。

### 14.1 单卡 27B 实测(靶位 `ggml-org/Qwen3.8-27B-Q4_K_M.gguf`)

```text
装载 6.14s · 设备 live 17.69 GiB vs 文件 17.67 GiB(比 1.001)   ← I-3 的钱行
首 token(单发,裸 prompt "The capital of France is") = " Paris"  ← llama.cpp 同文件 argmax
24 token 生成 = " Paris.\nThe capital of Germany is Berlin.\nThe capital of Italy is Rome.\nThe capital of Spain is"
              ≡ llama.cpp 同文件同 prompt 逐字相同(24/24)
capture = PlanOutcome::Captured · decode.step = 26.25ms/步(≈38 t/s 引擎内)
账本:1329 块 / 18519.8 MiB,零未命名(48 张列序表按名在册)
```

`flavors/gguf27b.toml`(单卡档;TP 仍拒,见 §14.4);**prefill 仍慢**(MMVQ-only,
R2 挂账)—— M2 的 std 读数等 tile 臂。

### 14.2 ★ 新门:**首 token 判据**(最强的一条廉价门)

两个靶位现在都在冒烟门里钉死**同文件同 prompt 的首 token**(= `" Paris"`,
llama.cpp 实测 argmax,top-1 领先次高 ≥ 2.5 nat ⇒ 不是 near-tie):

> 判据强度来自"首 token 由**整条预填充链**(全部权重解释 × 全部层 × 最终 norm ×
> 输出头)的 logits 顶决定"。这一条就能把"能跑出连贯文本"抓不住的那类错全部摁住。

**反面教材(就在本轮)**:27B 第一次跑出的文本是
`-pluginでは导师 Kitset-setsetsetsetsetset时刻-digit光盘...`(重复环)—— 
"字符种类 ≥ 4"的旧连贯判据**放行了它**;首 token 判据一上,当场红。

### 14.3 ★ 两条静默错值(都已修 + 入档)

| # | 缺陷 | 症状 | 根因 | 修法 |
|---|---|---|---|---|
| ① | **untied 的 GGUF 表落到 f16 槽** | 27B 装载期响亮拒(`量化键只能以 U8 交付`) | `Embedding::new_untied_quant` 只给 `lm_head` 传计划,**表自己**按 f16 建槽 —— tied/untied 说"头用不用表",与"表怎么存"正交 | 抽 `set_table_plan`(tied/untied 共用);`Model::new` 的 untied 臂也传 `plan.at(embed_key)` |
| ② | **`lm_head_matmul` 的分支键 = `gguf_ty`** | **静默错值**:首 token 直接垃圾,而权重面逐张量全对(极难归因) | `gguf_ty` 是"**表**是量化块"的事实,而 tied/untied 是"**头**用不用这张表"—— 拿前者当后者 ⇒ untied 模型把头算成 `hidden @ token_embdᵀ` | 分支键改 `lm_head.is_none()`(tied 事实);untied 走独立头 |

**教训(methodology,值得复用)**:两个靶位一个 tied(4B)一个 untied(27B),
**"表"与"头"的四象限里只跑了一半**;跨源逐张量门(权重面)对这种"接错线"完全无感。
⇒ **端到端里必须有一条"能一口否掉整条链"的判据**(首 token),不能只靠"输出看起来像话"。

### 14.4 27B 的**独立第三方**逐张量核验(FP8 原始检查点)

`models/Qwen/Qwen3.8-27B-FP8`(原始精度)+ **llama.cpp 转换器的写侧公式**
(不是我读侧的逆公式)+ `gguf-py` 的解量化 := 三源独立。执行面 =
`crates/models/src/formats/gguf_fp8_probe.py`(**dev 工具**,非 CI 门;
为什么不做成 Rust 门:参照源是 block-wise fp8 + `weight_scale_inv`,
而 Rust 侧没有 fp8 解量化臂 —— 而"三源互不共享代码"正是它的价值)。

> 曾试过"GGUF ↔ 生产档 AWQ-INT4"版本(写进 Rust 门),**失败**:该检查点是
> pack-quantized(只有 `weight_packed`/`_scale`/`_zero_point` 派生键),
> 源侧取不到 plain `.weight`,而在测试里重实现 ct-marlin 反解 = 第二份实现。
> ⇒ 该路废止,改用 FP8 探针(下表为实测):

| 张量 | max\|Δ\|/max\|ref\| | 判读 |
|---|---|---|
| `ssm_alpha/beta`(行置换 vpr=3)、`A_log`(`ln(-x)`+行置换)、`dt_bias`、`conv1d`(V 通道)、`ssm_norm`、两个 layernorm | **0 ~ 4e-3** | 变换**逐位对** |
| `attn_qkv`(仅 V 段)、`attn_gate`、`ssm_out`(列)、`ffn_up`、`token_embd`、`output` | **2.2e-2 ~ 4.1e-2** | = 两源量化噪声(fp8 block-wise vs Q4_K/Q8_0);**错置换会给 O(0.5–1.0)** |
| `attn_q/k/v/o`(identity;注意 `blk.0` 是 GDN 层,full-attn 取 `blk.3`) | 同上 | ✓ |
⇒ **权重解释面在 27B 上无实质错**;这也把 §14.3 ②的排查逼到了运行时(接错线)而不是源面。

### 14.5 剩下(刀 9 的真活)

1. **GGUF + TP2**:现状对 `world > 1` **响亮拒**。要做的事(侦察已完成,见 §14.6):
   块对齐的字节切片 + `ssm_out` 的**按 v-head 整块聚合**(27B 的 `ssm_out` = Q8_0,
   `hv=128 % 32 = 0` ✓ 合法)或"非按头对齐的型 ⇒ TP 拒";
2. **prefill tile 臂**(R2):27B 的 prefill 现在 ~50 t/s 量级,是 M2 读数的前提;
3. `flavors/gguf27b.toml` 的 `allowed_ops` 白名单(按发射账生成 + 入两处锁)。

### 14.6 TP 侦察结论(切片可行性,已实测逐型)

| 张量 | 27B 型 | 切法 | 块账 |
|---|---|---|---|
| `ffn_down` | Q4_K(256) | 行并行 ⇒ 内维切 | 内维 = **17408**(inter)/256 = 68 块 ⇒ /2 = **34 ✓**(W-1 实测;此处原写 5120 是笔误) |
| `attn_output` | Q6_K(256) | 同上 | 6144/256 = 24 ⇒ /2 = **12 ✓** |
| `ffn_gate/up` | Q4_K | 列并行 ⇒ 外维切 | 行是整块链 ✓ |
| `attn_q/k/v`、`attn_qkv`、`attn_gate` | Q8_0(32) | 外维切(`attn_q` = q ⊕ gate 两段;`attn_qkv` = q ⊕ k ⊕ v 三段) | ✓ |
| `ssm_alpha/beta` | Q8_0 | 按头切(外维 = nv=48 ⇒ /2 = 24) | ✓ |
| `ssm_out` | **Q8_0** | 行并行 ⇒ 内维切,且必须**按 v-head 整块聚合** | `hv=128 % 32 = 0` ⇒ 每头 4 块 ⇒ /2 = 24 头 = 96 块 ✓ |
| F32 小件(norm/dt/A_log/conv) | F32 | Replicate / PerHead | ✓ |

**唯一的结构难点**:`ssm_out` 的列置换(A2 表)在 TP 下**不能各切一半**
—— v-head 置换是在**全局** v-head 序上做的,任一 rank 只拿"文件序的一半"会拿到
**非 grouped 连续**的头集 ⇒ 该 rank 的激活映射跨 rank 失效。两条出路:
① **装载期按 grouped 头序聚合**(27B 的 Q8_0 允许:每头 4 块)⇒ 该 rank 直接吃
grouped 激活,**不需要 A2 表**(零运行期成本);
② 非按头对齐的型(`hv % be != 0`,如 K-quant 的 ssm_out)⇒ **TP 拒**(诚实)。

---

## 十五、W-1:GGUF + TP2(2026-10-19 收口;分片几何 · 双卡门 · 一条既有数值定谳)

> 目标(M2 原本的形态):27B Q4_K_M 在 **3090 对(TP2)** 上跑通,判据 = 首 token
> 与单卡/llama.cpp 一致 + 每 rank 只驻留自己那一半 + 零 Xid。

### 15.1 为什么分片几何必须"住在源里"(而不是复用 `formats::shard`)

通用分片 pass 的模型是"**矩形切一个连续的源区间**"(`Rows`/`Cols`/`RowsSeg`/
`ColsSeg`,坐标 = 源的元素序),对 GGUF 有两条事实不成立:

1. **量化块的列切是块算术**:切点必须落在 `be` 元素边界,目标行宽 = 块数 × 块字节
   ⇒ 单位是**原子**(一个块 / 一个原生元素),不是元素;
2. **v-head 是全局置换**:转换器把 HF 的 grouped 序写成 ggml 的 tiled 序
   (`g = (h % vpr) * nk + h / vpr`)。单卡靠 A2 表在**激活侧**扭回;TP 下
   "各切一半文件序"会拿到**非 grouped 连续**的头集 ⇒ 必须在**装载期聚合**。

⇒ 决策:分片几何与读逻辑**归源**(`formats/gguf/shard.rs` + `GgufSource`),新增
`WeightSource::owns_shard()`(默认 `false` = 走通用 pass;GGUF 档 `true` 时
`eval_load` **跳过** pass,由源按 `(slice_id, world)` 交付本 rank 的字节)。

### 15.2 一条贯穿全部角色的**不变式**(测试与实现的锚)

> **每一 rank 装载后的张量 == 全量逆变换(grouped 序)张量的连续块**:
> 行 = grouped 行窗口,列 = grouped 原子窗口;分段件(`attn_qkv`/`attn_q`/
> `ssm_conv1d`)是"每段各取一窗再拼接"。

推论(★ 用户裁决的落地):**TP 档不需要 A2 表** —— 每 rank 的激活本来就是
grouped 序的连续段 ⇒ `SourceFacts::GGUF_TP.vhead_tiled_columns = false`,
`ssm_out` 不再声明虚拟键 `ssm_out_perm`(源侧也会**响亮拒**该虚拟键,防两条
规范并存)。分片档的 `ssm_out` 权重内维 = 本 rank 的 24 个 v-head(grouped 序),
逐头一段(vpr=3 时文件里是 3 段 × 8 头),每段 `hv/be` 个原子块。

### 15.3 角色表(GGUF 名 → 切法;**唯一点** = `shard::role_of`)

| 角色 | 键 | 切法 |
|---|---|---|
| `Out` | `ffn_gate`/`ffn_up`/`attn_k`/`attn_v` | 行窗口(外维) |
| `Segs` | `attn_q`(q⊕gate)、`attn_qkv`(q⊕k⊕v)、`ssm_conv1d` | 每段各取行窗口(V 段 grouped) |
| `OutHead` | `attn_gate`/`ssm_alpha`/`ssm_beta`/`ssm_a`/`ssm_dt.bias` | v-head grouped 行窗口 |
| `In` | `ffn_down`/`attn_output` | 块对齐列窗口 |
| `InHead` | `ssm_out` | 逐头列段(v-head grouped) |
| `Replicate` | 各 norm / `token_embd` / `output` | 全量 |

**三条响亮拒**(开箱即验,`GgufSource::open_shard` 逐键跑一遍):切分维不整除;
量化件列切不落块边界;`hv % be != 0`(按头不可分 ⇒ TP 拒;4B 的 `ssm_out` 是
K-quant 256 > `hv=128`,正是这条 —— 实测 4B 档被**如实拒**,不静默)。

`GgufPlan` 也随档分叉:分片档的层声明是**分片维**,形状门按角色把声明维
**放大 world 倍**再与文件比(`attn_q` 的段长从张量本身反推)。

### 15.4 门(全部真机/真文件,零 GPU 的两条先跑)

| 门 | 判据 | 位置 |
|---|---|---|
| 几何手算 | 27B 逐键角色/行宽/列段/`col_first` 钉死值(含 `ssm_out` 的三段文件头) | `shard::tests::key_geometry_27b_is_pinned` |
| 几何自洽 | 被切的 **640** 键:两 rank 交付量之和 == 全量;切行类列全量、切列类原子两侧之和 == 全量 | `shard::tests::geometry_tiles_full_tensor_27b` |
| **逐键逐位** | blk.0+blk.3 **全键面**:分片读 == grouped 全量对应窗口(36 个 (键,rank) 组合,含 `ssm_out` 逐头列段、`conv1d` 原生行窗、`ssm_a` 1-D 元素窗) | `tests/tp_load_gate_gguf.rs::gguf_shard_reads_equal_grouped_windows_all_keys` |
| 装载账 | 分片 spec + `GgufPlan::for_shard` + 源 `elem_len` 三方一致;rank0/1 各 **851** 件(量化 498),量化字节 **9.64 GiB**(文件 17.67) | 同上 `sharded_plan_shapes_match_source_elem_len` |
| 双卡 e2e | 首 token `" Paris"` 硬门 + 两 rank logits **逐位相同** + AR 注入对称(r0=r1=256)+ 捕获 `Captured` + rank0 live 9.68 GiB | `tests_gguf::gpu_gguf_27b_tp2` |

```bash
GDN_TRITON_PY=<owl>/crates/kernels/.venv/bin/python \
OWL_GGUF_TP_DEV_A=41:00.0 OWL_GGUF_TP_DEV_B=42:00.0 \
  cargo test -p owl-engine --lib gpu_gguf_27b_tp2 -- --nocapture
```

### 15.5 实测(2026-10-19,3090 Ti 对)

```text
分片装载  rank0 9.68 GiB / rank1 9.68 GiB(文件 17.67;整份驻留会超 24G)
捕获态    PlanOutcome::Captured · peer 束 true
首 token  " Paris"          ≡ 单卡 ≡ llama.cpp 同文件 argmax
两 rank   logits 0 项不同(AR 把两侧统一到逐位相同)
AR 注入   r0 = 256 · r1 = 256(v1 两段;256 = 每 rank 每层 2 次 × 64 层 × 2 趟[预热+捕获])
24 token  " Paris.\nThe capital of France is Paris.\n…"(连贯;与单卡文本见下)
decode    30.9 tok/s · 首步 logits 预热+捕获共 ~19s
```

### 15.6 ★ 一条**既有行为**的定谳:TP2 与单卡的首步 logits 不等价

| 档 | TP2 vs 单卡 max\|Δ\| | cos | top-1 |
|---|---|---|---|
| AWQ-27B(`tp_logits_27b`,既有诊断件) | **10.83** | — | 同 |
| **GGUF 27B Q4_K_M(本刀)** | **6.36**(max\|L\| 17.5) | 0.939 | **同** |

**判读**:"TP2 与单卡 logits 不等价"**不是 GGUF 分片的错**(GGUF 的偏差比 AWQ 还小),
而是引擎既有的 numerics:row-parallel 拆半 ⇒ 归约序变化 + 激活量化在两侧各做一次
+ f16 AR 逐层舍入。量化档的 argmax 在 near-tie 处会翻(token 6 的
`Germany`/`France` 即此类),**文本仍然连贯**,首 token(强判据)一致。

⇒ 因此 W-1 的判据写成:**argmax 相同 + 两 rank 逐位相同 + `max|Δ|` 与 AWQ 对照同量级**
(而不是"与单卡逐位/逐 token 相同"——那在 TP2 形态下不成立,量化档尤甚)。
**这条口径已写进测试注释**,避免后来者把它当回归。

### 15.7 与既有设计文档的差异(回写)

- §14.6 的 `ffn_down` 行:内维是 **17408**(不是 5120),68 块 ⇒ 各 34(原文笔误,已改);
- §14.5 的"GGUF + TP2 待做" ⇒ **已收口**(本节);
- `docs/arch/新格式接入-改动点清单.md` 的 TP 一节:新增"源自担分片(`owns_shard`)"一条
  与"分片档源事实(去掉 A2 表)"一条。

---

## 十六、W-2:prefill 权重复用臂(2026-10-19;`vec_dot` 复用 → 算术墙定式)

### 16.1 立案:MMVQ 的流量公式(一条就解释了全部数字)

MMVQ 的网格是 `(n / n_warps, m)`,每个 (权重行, token) 对独立读一遍该行 ⇒

```text
一次前向的权重流量 = m × 权重字节数        ← 与 chunk 无关、与 m 无关
```

实测(单测「prefill 探针」`tests_gguf::gpu_gguf_prefill_probe`,1001-token prompt):

| 档 | 权重 | 带宽推导 | 实测 |
|---|---|---|---|
| 27B Q4_K_M | 17.7 GB | 936 GB/s ÷ 17.7 GB = **53 t/s** | **48.9 t/s** |
| 4B Q4_K_M(历史) | 2.81 GB | 936 ÷ 2.81 = 333 | 327.7(§13.4) |

⇒ decode(m = 1)本来就该读一遍,**prefill 是纯浪费**:m = 1001 时同一行权重被读了 1001 次。

### 16.2 复用臂(`owl_gguf_gemm_tile_*`,已落地)

改动只在**调度**:同一 warp 一次拿住一行权重,连续算 `TILE_MT = 16` 个 token 的
`vec_dot` ⇒ 权重流量 ÷ 16。`vec_dot` 的归约序**一字未动**(每 token 仍是同一串
k 块 + 同一 warp reduce)⇒ 输出与 MMVQ **逐位相同**:

- 门①(位级,GPU):`kernel_gate::gpu_gguf_tile_gemm_matches_mmvq_bitwise`
  —— 四型 × `m = 16 / 20`(整窗 + 尾窗)**逐位相同**;
- 门②(端到端):`gpu_gguf_prefill_probe` 里两臂(`OWL_GGUF_TILE=0|1`)在
  1001-token prompt 上**首 token 相同**。

分派:`m ≥ 8` 走复用臂(`linear.rs::GGUF_TILE_MIN_M`);decode(1)与投机 verify
(2..7)留 MMVQ。白名单(`PROD_GGUF4B` / `flavors/gguf4b.toml`)已补四个 tile 入口。

### 16.3 实测收益与**新墙**

| chunk | MMVQ | 复用臂 | 加速比 |
|---|---|---|---|
| 128 | 48.6 t/s | **80.4 t/s** | 1.65× |
| 512 | 48.9 t/s | 78.0 t/s | 1.60× |

**为什么只有 1.65×**(这条是本刀的真正结论):权重流量已降到 1/16(≈ 0.08 s),
剩下的 12.4 s 全是**算术** —— `vec_dot` 对每个 token **重新解量化**同一个块 ⇒
有效算力只有 **≈ 4.4 TFLOPS**(54 GFLOP/token × 1001 ÷ 12.4 s)。
即:**复用解掉了带宽墙,露出了算术墙**。

### 16.4 W-2b 立案:dequant-to-smem + f16 点积(下一刀)

目标:把"每 token 重新解量化"改成"**每窗解量化一次**",之后是纯 f16 点积。

设计要点(已核过的三条事实,省下次重查):

1. **`threadIdx.x` 即 lane**:现有核的 block 形状是 `(32, n_warps)`,所以
   `gguf.cuh` 的协作式 `dequantize_block_q4_K/q5_K`(用 `threadIdx.x` 分工,
   需 32 线程)**可以被任意单个 warp 直接调用**且语义正确 ⇒ 解量化到 smem
   不需要抄数学;
2. **Q6_K 是例外**:它假定 64 线程(`ip = tid/32`)⇒ 只能在 `blockDim.x = 64`
   时正确;复用臂里保持"Q6_K 走 `vec_dot` 复用"(现成,1.65×),或另写
   lane 参数化的 64 线程版本;
3. **数值门要换口径**:smem 臂喂**原始 f16 激活**(不再经 q8_1 量化)⇒ 与 MMVQ
   **不再位级相同**,门改成 `max|Δ|/max|ref| < 1e-2`(参考 = §13.4 的 CPU f64
   对拍),并保留"两臂首 token 一致"作端到端判据。

预算:smem = `n_tile × k_chunk × 2B`(如 32×256 = 16 KB)+ x 窗(16×256 = 8 KB)
⇒ 每 CTA < 32 KB,4 CTA/SM 可跑;理论算力档:纯 f16 FMA(f32 累加)≈ 40 TFLOPS
⇒ 27B prefill **≈ 600-800 t/s**;再用 `wmma`(f16×f16→f32)可到 ~1000 t/s。

---

## 十七、W-2b:prefill smem 复用臂(2026-10-19;nsys 定谳 → 算术杠杆)

### 17.1 先看分布:nsys 逐核账(4B,1001-token prompt,chunk 512)

```bash
OWL_TEST_DEVICE=41:00.0 OWL_GGUF_TILE=1 \
nsys profile -o /tmp/gguf4b --force-overwrite true --trace=cuda --stats=true \
  cargo test -p owl-engine --lib gpu_gguf_prefill_4b_probe -- --nocapture
```

| 核 | 占比 | 实例 | 备注 |
|---|---:|---:|---|
| `owl_gguf_gemm_tile_q6k` | **45.6%** | 288 | Q6_K = `attn_qkv`/`ffn_down`/`token_embd`(头) |
| `owl_gguf_gemm_tile_q4k` | **40.5%** | 744 | ffn gate/up + 各 Q4_K |
| `owl_gguf_gemm_tile_q8_0` | 4.8% | 72 | `ssm_out` |
| `vllm_chunked_prefill_paged_attn` | 3.0% | 48 | attention |
| `owl_gdn_recurrence_varlen_gqa` | 2.5% | 144 | GDN |
| `owl_gguf_gemm_tile_q5k` | 2.1% | 96 | Q5_K |
| `owl_gguf_quant_act_f16` | 0.7% | 1235 | 激活量化 |
| 其余(conv/norm/rope/silu/narrow…) | <0.5% | | |

两条判读:
1. **GEMM ≈ 93%**,其余全在噪声里 ⇒ 要动就动 GEMM,别碰 attention/GDN;
2. 各型占比 ≈ **FLOPs 占比**(Q6_K 43% FLOPs / 45.6% 时间)⇒ **没有"某型特别慢"**,
   是**整条 `vec_dot` 路径的有效算力只有 3.7-4.4 TFLOPS**(每 token 重新解量化)。

### 17.2 本刀做的事(逐条对上 W-2a 的结论)

| 症结 | 本刀 |
|---|---|
| 每 token 重新解量化 | 权重窗 `[NT=64, KC=256]` 解进 **smem**,每窗**一次** |
| 解量化数学不想抄 | 快照的 `dequantize_block_q4_K` 是 32 线程协作式,而本核 block = `(32, NW)` ⇒ **`threadIdx.x` ≡ lane** ⇒ 单 warp 直接调 ✓(Q5_K/Q6_K 是 **64 线程**型 ⇒ 按 `tid = lane, lane+32` 两趟;Q8_0 快照无该函数 ⇒ 5 行自写) |
| smem 带宽墙 | **寄存器 4×NR 分块**(512 线程 × 16 累加器 ⇒ smem 读 : FMA = 1 : 2) |
| 激活还要 q8_1 | 本臂**直吃 f16**(省一发量化核;数值上更准 —— 门的口径随之从"位级"改成 allclose) |

几何:`grid = (ceil(n/64), ceil(m/64))`,`block = (32, 16) = 512 线程`,
动态 smem = `(64+64) × (256+2) × 2B = 66 048B`(>48KB ⇒ 走既有 opt-in 通道)。

### 17.3 门与实测

| 门 | 判据 | 结果 |
|---|---|---|
| 数值对拍 | `kernel_gate::gpu_gguf_smem_gemm_matches_mmvq`:四型 × `m = 64 / 101`(整窗+尾窗)vs MMVQ,`max|Δ|/max = 3.3e-3 ~ 7.9e-3`(= q8_1 量化噪声量级;错窗/错块会给 O(1)) | ✓ |
| 端到端 | 1001-token prompt 首 token:两臂相同(`" The"`) | ✓ |
| 回归 | 27B/4B 冒烟(短 prompt 走 MMVQ)、`op_tables` 10/10、`ops_whitelist` 2/2、`no_runtime_compile` 4/4 | ✓ |

prefill 实测(1001-token prompt):

| 档 | MMVQ(W-1 前) | W-2a(vec_dot 复用) | **W-2b(smem 复用)** | 总加速 |
|---|---:|---:|---:|---:|
| 4B(chunk 512) | 333 t/s | 468 | **819.5** | **2.46×** |
| 27B(chunk 512) | 48.5 | 78.0 | **121.4** | **2.50×** |

有效算力从 3.7-4.4 升到 **≈6.6 TFLOPS**(4B FLOPs / 时长)。
**仍在算术墙内**(FP32 峰值 36 TFLOPS 的三成不到):下一步杠杆 = `wmma`
(f16×f16→f32 张量核;smem 已按行主序/f16 铺好,可直接 `load_matrix_sync`)+
双缓冲(cp.async 盖住 staging)。

### 17.4 踩过的两个坑(已入档)

1. **`k0` 偏移漏了**:权重窗解量化必须按 `k0 / qk` 退块指针 —— 漏了 ⇒ 每个 k 窗
   都解第 0 块,`max|Δ|/max = 0.92`(门当场红);
2. **Q5_K 也是 64 线程型**(`il = tid/16` 需 0..3):只调一趟 ⇒ 半块是真值、
   半块是上窗残留 ⇒ 偶发 `inf`。**判据**:门的 m=64/101 两档 + 四型全覆盖;
3. (反例)把 x 暂存改成"token 外层"消除法 —— **实测慢 1.35×**(502 vs 687 t/s),
   已回退。原因未深究,但说明**这条路径必须用实测选形**,别凭直觉。

---

## 十八、W-2c 立案:核心算子**不再自研** —— 移植 llama.cpp 官方 MMQ

> 用户裁决(2026-10-19):"核心算子不要自己去写 —— 去 llama.cpp 官方库薅;
> 官方慢的话再去社区找高性能 GGUF 算子库。"

### 18.1 现状盘点:我们已经"薅"了什么、还差什么

| 件 | 来源 | 状态 |
|---|---|---|
| 块结构 / `dequantize_block_*` / `vec_dot_*` | llama.cpp(`ggml-cuda.cu` 时代,经 `repos/attention.rs` 快照 → 本仓 `gguf.cuh`) | ✅ 已在用(MMVQ/行解量化) |
| MMVQ 内核 | 同上(`gguf_gemm.cu`) | ✅ |
| **MMQ(tensor-core prefill)** | llama.cpp **官方** | ⬜ **本刀:已 vendoring + 编译通过** |
| 自研 smem 复用臂(W-2b) | 本仓 | ⚠️ 保留为 MMQ 未覆盖形态的回退(它比 MMVQ 快 2.5×,比 MMQ 预期慢 3-5×) |

### 18.2 关键决策:按**同代 revision** 薅,而不是 head

本仓 `gguf.cuh` 的快照还带 `#ifdef GGML_QKK_64`;上游在 **2024-05-23**(`e84b71c2c`)
删掉了 QK_K=64 ⇒ 我们的块布局 = **该日之前**那一代。据此锁定 vendoring revision:

- **`llama.cpp@1b1e27cb49`(2024-05-23,`e84b71c2c` 的父提交)**
- 该版 MMQ 是 **单文件 `mmq.cu` 1570 行 + `vecdotq.cuh` 1280 行**(head 已拆成
  7.8k 行、且含本仓没有的 fp4/Q1_0 型);
- **额外红利**:该版 MMQ **直吃普通 `block_q8_1`(36B)** —— 正是本仓
  `owl_gguf_quant_act_*` 已经在产的布局 ⇒ **不需要新造激活量化臂**;
- 该版只依赖 `common.cuh` 的两处 host 面(`ggml_cuda_info()` 取 cc + `GGML_UNUSED`),
  且**内核里零分配、零设备查询**(H1/H2 天然满足)。

落地:`crates/kernels/cu/gguf/vendor/`(逐字 + MIT LICENSE + `README.md` 记
revision/文件/shim 差异/接线清单)· 适配壳 `vendor/common.cuh` 把
`ggml_cuda_info()` 改成**编译期 `OWL_CUDA_CC` 常量**(H2),
`kvalues_iq4nl` 等按上游同代 `common.cuh` 逐字补入 ·
编译验证:`nvcc -arch=sm_86` 通过,`cuobjdump` 可见
`mul_mat_q<{q4_K,q5_K,q6_K,q8_0}, fallback=true|false>`。

### 18.3 为什么不是"自研 wmma"(原 W-2c 计划)

自研 wmma 臂只解决"内层循环",而 MMQ 已经把 **tile 调度 / smem 布局 /
逐型解量化进 fragment / dp4a↔mma 分档 / 小 ncols 回退** 都调好了(上游数年打磨)。
我们自研版还停在"寄存器 4×2 分块 + vec_dot 复用"(有效 6.6 TFLOPS),自研到 MMQ
水平等于重走一遍上游的路 —— 违背用户裁决,也不划算。

### 18.4 接线清单(下一刀,详见 `vendor/README.md`)

1. `gguf_mmq.cu`:每型 `extern "C"` 入口(9 参,末位 `stream`)→ 合成最小
   `ggml_tensor` → 调 `ggml_cuda_op_mul_mat_q`;
2. **server 侧 foreign 分派**(`owl_gguf_mmq_*` 按名 → 带 stream 调用,与
   `cublas`/`flashinfer` 同一条通道);
3. `ops/gguf.rs` 四型声明 + `driver.rs` 拾取 + `ops/mod.rs` 白名单;
4. `linear.rs` 分派:`m ≥ 阈值` 且 `k % 32 == 0` 且**无内维列序表** ⇒ MMQ
   (MMQ 不做 tiled 置换 ⇒ 单卡 `ssm_out` 回退 smem 臂);
5. 门:vs MMVQ `max|Δ|/max < 2e-2` + 1001-token 首 token 一致 + 4B/27B prefill 实测。
   预期:Ampere 档 MMQ ≈ 30-50 TFLOPS ⇒ 27B **400-600 t/s**(现 121)。

### 18.5 社区备选(用户要求:官方慢再看社区)

| 候选 | 值不值得 | 备注 |
|---|---|---|
| **ik_llama.cpp**(本地 `repos/ik_llama.cpp` 已有) | 值得(官方接入后对照) | ikawrakow 的 fork,MMQ/注意力有一批 Ampere 优化;同一套块结构 ⇒ 可当"官方 MMQ 更快版本"直接替换块 |
| llamafile `tinyblas.cu` | 观察 | 走 BLAS 路(权重解成 tile 再 GEMM),与本仓"压缩驻留"约束冲突(需 f16 tile 缓冲) |
| vLLM gguf-plugin | 不取 | 也是 dequant + cuBLAS 路(显存放大),与 I-3 冲突 |
| Marlin(EXL/AWQ 生态) | 不适用 | 面向 4bit 分组量化布局,GGUF 的 K-quant 超块不匹配 |

**纪律**:先让官方 MMQ 跑起来并端到端对拍,再拿 ik_llama.cpp 的同一批核做
A/B —— 不允许两套同时接线。

### 18.6 W-2c 收口实录(2026-10-19):两档实测 —— **快的是大 token 窗,不是张量核**

接线完成后按 README 清单落地(适配入口 → `ops`/`driver` 声明 → `linear` 分派 →
对拍门)。先跑**上游默认**(张量核档),结果**不比自研 smem 臂快**:

| 档 | tile(`x`=每 CTA token 数) | 4B prefill | 27B prefill |
|---|---|---:|---:|
| 张量核(`CUDA_USE_TENSOR_CORES`) | Q4_K/Q5_K/Q6_K `{4,32,4}` | 739 t/s | 117 t/s |
| 自研 smem 臂(W-2b,参照) | 64 token/CTA | 819 | 121 |
| **dp4a 档(现役默认)** | Q4_K/Q5_K `{64,128,4}`、Q6_K `{64,64,4}`、Q8_0 `{128,64,4}` | **1829** | **335** |

**根因**(读核源码定谳):张量核档 `mmq_x = 4` ⇒ 每个 CTA 只覆盖 4 个 token,而 k 全量
在 CTA 内流过 ⇒ **权重被重读 `m/4` 次**(27B @m=512:m/4 = 128 遍 = 2.3 TB)。
dp4a 档 `mmq_x = 64/128` ⇒ 权重复用 16-32×,点积走 int8 dp4a 但算术密度足够
(每权重字节换 4-8 个 MAC)。⇒ 世上"MMQ 快"在这个硬件上来自**大 token 窗**;
张量核只是把 tile 内部算得更快,却把 token 窗压到 4。

**纪律红利**:切换档位时 Rust 几何(`MMQ_CFG`)与 C 开关必须同步 —— 不同步时
数值门立刻给 **1.0 相对误差**(实测踩到:`.cu` 没重编,因为 build.rs 的 fatbin 指纹
**原先不含 `#include` 的头**;已修:凡引用 `vendor/` 的 `.cu`,指纹混入 vendor 全树
的长度+mtime ⇒ 改开关必重编)。

**现役分派**(`linear.rs`):
`m ≥ 32` → **官方 MMQ(dp4a 档)**;`m < 32` → MMVQ(官方同款);`ssm_out` 带 A2
内维列序表(单卡、占参数 ~6%)→ 自研 smem 臂回退(MMQ 不做激活侧列序置换)。
`OWL_GGUF_MMQ=0` / `OWL_GGUF_SMEM=0` 可逐级回退;数值门
`gpu_gguf_mmq_matches_mmvq`(四型 × m=32/71,`max|Δ|/max = 2.5e-5 ~ 1.6e-2`)。

**W-2 全程账**(27B,1001-token prompt,chunk 512):

```text
W-2 前(MMVQ)          48.5 t/s      权重流量 = m × 权重字节(带宽墙)
W-2a(vec_dot 复用)     78.0 t/s      流量 ÷16,露出算术墙(≈4.4 TFLOPS)
W-2b(自研 smem 臂)    121.4 t/s      每窗解量化一次(≈6.6 TFLOPS)
W-2c(官方 MMQ·dp4a)   335.1 t/s      大 token 窗 + dp4a int8 点积   ← 现役
```

# NVFP4 块缩放 GEMM 臂 —— 从装载到进图（落地记录）

> 版本 v1.0（2026-10-28）。承接：用户裁决「NVFP4 统一臂，不做 `m ≥ 128` 分流」、
> 「装载管线拆三轴」（`docs/arch/装载管线-源精度x目标格式-设计.md`）、
> 「decode / propose / verify 都进图」。
> 前置：`operator-contract.md`（声明面/执行面分离）、`kernel-admission.md`（代际准入门）、
> `w4a8-family.md`（同一条"新量化族怎么接"的先例）。
> 关联提交：`2986b99` → `42085f6` → `07a0a9a` → `898ba78` → `ecaa30f` → `a48e3cc`。

---

## 一、结论先行（一页纸）

| 面 | 事实（RTX 5070 Ti / sm_120a，4B BF16 检查点） |
|---|---|
| 权重驻留 | 源 8.41 GB → **3.17 GB**（`q` 每字节两个 e2m1 + `sf` 每 16 个 k 一个 e4m3） |
| 装卸 | **5.3 s**（设备侧打包；host 打包版 18.4 s，并行前 48.9 s）；`[load] verify` 352 键回读 **mismatch = 0** |
| 数值 | 装配期 oracle `max|err| = 0`（同规则 host 参考）· `17*23 → 391` · 散文 B/tok 3.7~3.9 |
| 图 | session / verify(T=4) / fold(m=3) / propose(m=0..3) **全部捕获态** |
| 解码 | 190 tok/s（spec d3，AL 3.24）· 无投机 98~101 · F16-eager 77 |
| 回归 | 9B GGUF 生产档输出**逐字节一致** |

一句话：**权重量化格式不再是"另一套装载代码"** —— 它成了装载管线里的一个
`Transform`（一源二槽），而 alpha 设备化之后这条臂与其它族在图上完全平权。

---

## 二、算子家族（`crates/kernels`）

### 2.1 为什么必须是预编静态库

FP4 是 **arch-specific** 指令（`mxf4nvf4.block_scale`）：目标必须
`-gencode arch=compute_120a,code=sm_120a`（两个都带 `a`），且主循环只有 CUTLASS
模板实现 ⇒ 走 nvcc → `libnvfp4.a`，而不是本仓主流的 NVRTC/cubin 通道。
`feature nvfp4` 由 `arch-sm120` 打开；未编入时 `nvfp4_stub.rs` 提供**同面替身**
（`pack_weight_b` 一律响亮拒），上层零 `cfg`。

### 2.2 布局事实（全部实测，不靠推理）

| 事实 | 数值/依据 |
|---|---|
| SF 跨度要用 **`cosize`** 不是 `size()` | 布局里有一个 stride-0 维（每 SF 字节覆盖 16 个 k），`size()` 恰好大 16 倍（4B `gate_proj` 23.6 MB vs 1.47 MB） |
| **A 侧** SF 字节 | `ceil(M/128)·128 · K/16`（M 按 128 行块排；M=1 也要整块） |
| **B 侧** SF 字节 | `N·K/16`，**N 必须 128 对齐**（`N=1000,K=256` ⇒ 16384 > 16000） |
| SFA 仿射模型 | plan 期从 cute 布局取 5 个步长 + 4096 点逐点校验；不一致 ⇒ **拒绝建 plan** |
| 编码一律硬件 RNE | 早期"表内最近值"在等距 tie 上与硬件差 28/32768 字节 |

### 2.3 声明面

`NvFp4Gemm` 进注册表（`dom.nvfp4` / `quant.nvfp4`），选核走 `driver::resolve`
与代际准入（`nvfp4-cutlass` 只在 sm_12x 放行，其它代际**带原因拒**）。数值门 =
boot 期 oracle：经**生产同一条 dispatch** 跑一发真 GEMM，对 host 同规则反量化参考
比较（相对偏差 ≤ 1e-2；实测 0）。

---

## 三、装载：三轴正交化

```
源(f16/bf16/f32/fp8)  ──decode_to_f32──▶  目标(Transform)  ──1:N 槽──▶  设备
```

- **源精度**（`formats/source.rs::SrcDtype`）只从文件元数据来，禁止由 plan 反推；
  fp8 无对应臂时**响亮拒**（`E4M3FNUZ` 明确不当作 `FN` 解）。
- **目标格式**（`formats/transform.rs::Transform`）声明产物（`SlotDecl`）、位置
  （`Place::Host|Device`）、怎么算。NVFP4 有**两条臂**：设备臂（首选，`nvfp4_pack_weight`
  算子，与 AWQ 的 `ct_repack` 同律）与 host 臂（回落：f32/fp8 源、CPU face），见 §六·二。
- **槽**（`Want.extra` + `Want.scalar`）把 1:N 表达出来：`q` 是主槽，`sf` 是附加槽，
  `gb` 走 `ScalarSink`（主机标量 ⇒ kernel 参数，不是设备块）。
- 装载面给两张硬门：`verify_slot_agreement`（层声明 ⇔ 变换产物逐项对齐，写显存**之前**）
  与家族侧的账长门（`aq/asf/dg` 字节数）；`gb` 缺失/未写都会拒（初值 NaN）。
- `build_shard_pass` 对 NVFP4×TP **写明拒**（SF 按目标 N 排布，需 `SliceThenRepack`）。

---

## 四、alpha 设备化（进图的关键一刀）

旧实现：量化器把激活的 per-tensor 全局缩放 `g` 写到设备 → **回读 4 字节** → host
算 `alpha = g·gb` → 作为 CUTLASS 内核参数发射。那一次 D2H + stream 排空是**结构性**
host 工作 ⇒ 本族只能 eager（解码 82 tok/s，且三张图都建不了）。

新实现：epilogue 换成 `fusion::LinearCombination`（Args 带 **`alpha_ptr`**），收尾核直接写
`dg[0] = g`、`dg[1] = alpha = g·gb`（一槽两值），GEMM 读设备指针 ⇒ **零 host 工作**
⇒ `CAPTURE = NoHostWork`。数值与 host 版**逐位等价**（都是 f32 乘法）。

| 口径 | decode | 图 |
|---|---|---|
| eager + 回读 | 82 tok/s | session/verify/fold/propose 全 eager |
| **设备 alpha** | **98~101**（无投机）/ **190**（d3） | 四类图全捕获 |

---

## 五、投机闭环（propose 图）

三处接线才让 propose 有图可捕：

1. **检查点自带 MTP 头**：HF 蒸馏档把 `mtp.*` 放进主检查点，但 `config.json`
   **不声明** `num_nextn_predict_layers` ⇒ 只能从**权重键**看出来
   （`SafeTensorsSource::has_mtp_head()`，与 GGUF 探 `blk.N.nextn.*` 同律）。
2. **`mtp.fc` 列半拆**：MTP 头的 fc 是单个 `[out, 2h]`，而 `MtpPredictor` 消费
   两半（`fc_e` 吃 embedding 侧、`fc_h` 吃 hidden 侧）。`safetensors` 源新增虚拟键
   `mtp.fc_{e,h}.weight`（行内去交织），与 AWQ 源的同名派生键同语义。
3. **草稿几何单源 = target spec**：MTP 头与 target 同维是它的定义性质；旧实现
   重读草稿目录的 `config.json` 并把 `qwen3_8_27b()` 预设写死（4B/0.8B 会落错默认值），
   且 `inp.spec` 已分片 ⇒ 再切一次是静默错值（现 TP>1 直接拒）。

---

## 六、两条踩过的坑（留给后来者）

1. **"用了 atomicMax" ≠ 不确定**：amax 是 max 归约（非负 f32 位序 = 整型序，
   与顺序无关）⇒ 本就位一致。旧声明写 `NonDeterministic` 属误判（那只对**加法**成立）。
2. **alpha 与 g 挤同一 4 字节**：收尾核先写 `g` 再写 `alpha`，量化核读到的就成了
   `g·gb` ⇒ 输出退化成语无伦次的乱码（而且因为 token 变"随机"，B/tok 与速度都虚高，
   看起来像"更快了"）。改成 `f32[2]` 后数值立刻正常 —— **文本质量是进图后的第一道门**。
3. **设备打包想与 host 逐字节等价，必须先在 f16 网格上对齐**（2026-10-28 实测）：
   host 打包器的输入契约是 **f16**，而 bf16 源里有 `|x| < 2^-14` 的值（实测某层 0.6%，
   且**成列聚簇** ⇒ 整组 max 都是次正规的组有 1.34%）。设备若直接吃 bf16，
   这些组的 `want` 就与 host 差 1 ULP ⇒ e4m3 差 1 LSB、q 半字节连带不同
   （当时症状：`layers.2.mlp.down_proj` q 差 0.39% / sf 差 1.34%，而同形状的
   `layers.0/1` 同键**全对** —— 数据相关 ⇒ 一定是舍入口径，不是布局 bug）。
   三处对齐：bf16 源先 `__float2half_rn`、`want = mx * (1/(6g))`（不是 `mx/(6g)`）、
   退化回退同式（`mx<=0⇒1`、`sd<=0⇒1`）。对齐后 **4 档几何 × {f16,bf16} 源全 0 差**。

---

## 六·二、装载改「设备侧打包」（2026-10-28 收尾）

问题（用户裁决）：既有量化族（AWQ）的装载是 **DMA 上卡 + GPU 就地 repack**
（`load_device_rearrange` → `ct_repack`），而 NVFP4 第一版是 **host `pack_weight_b`**
（4B 档 +11.6 s CPU，`pack` 段 300 ms/47 MB）。同一件事两个落点 = 架构不一致。

新落点（与 `ct_repack` 同律）：

```
fetch(文件→host) → htod(源 f16/bf16 直上卡) → nvfp4_pack_weight(三发量化核)
   → dtoh(g, 4B) → free(源块) → 交付 q / sf 两槽 + gb 标量
```

- **它是正式算子**（`ops::nvfp4::NvFp4PackWeight`，与 GEMM **同族同 plan**：SFB 布局只
  依赖 `N,K`，plan 键取 `m=1` ⇒ 装载期建的那份被首轮 decode 直接命中）；名/槽序/解析
  三处各三行（`family/nvfp4.rs` 的 `PACK_WEIGHT` / `client::parse_pack` / `server::run` 按名分派）。
- **槽声明是 `U8`（不透明源字节）**，源精度由 `src_bf16` 决定 —— 声明成 F16 会在 bf16 源
  上说谎，而事实面（`spec::check_tensor_facts`）会当场按声明比对线格式。
- **校验档（`OWL_LOAD_VERIFY=1`）内置逐字节对拍**：设备产物 dtoh 回来，与 host
  `NvFp4Pack` 同源打包**整块 memcmp**，不一致就报差多少字节。生产档只回读 4 B。
- **顺带补掉源层的一个缺口**：合并虚拟键（`in_proj_qkvz` = `in_proj_qkv ⊕ in_proj_z`）
  原先 `native_dtype`/`copy_native_into` 一律 `None`（"原始通道不做行拼接"）⇒ 24 个键
  掉到 f32 元素通道 + host 打包（3.0 s）。补上"两段同 dtype ⇒ 行拼接 = 两次 memcpy"后，
  这些键也走设备臂；**F16 档同受益**（合并键改走设备 cast）。

实测（4B BF16 检查点 → 装载期 NVFP4）：

| 口径 | 装载 | `[load] verify` | 驻留 | 数值 |
|---|---:|---|---:|---|
| host 打包（并行、第一版） | 18.4 s | 352 键 0 差 | 3166.3 MiB | `17*23→391`、B/tok 3.72~3.88 |
| **设备打包（现在）** | **5.32 s** | 352 键 0 差（内部逐字节对拍也 0 差） | **3166.3 MiB（同）** | `391`、B/tok 3.81 |
| 同机体 F16 基线（对照） | 6.76 s → **3.01 s** | — | — | `391`、B/tok 3.76 |

分相（`--set load.verify=false`，404 组）：`htod 3.49 s`（DMA 底）· `fetch 0.65 s` ·
`harvest 0.16 s` · `launch 0.011 s`（176 发）· **`pack` 相消失**（host 打包 0 次）。
回归：9B GGUF 生产档输出逐字节一致（137/94/757 B）。

---

## 七、遗留（详见本地工作单 / 后续立项）

- `propose` 图内的草稿头仍是 F16（既定"二期"）；要草稿也吃 FP4 才谈得上"propose 内有 FP4 算子"。
- NVFP4 × TP（`SliceThenRepack`）、fp8 源（`decode_to_f32` 已备，无消费者）。
- decode 的 m=1 仍走 128 行 tile（128× 算力浪费）⇒ 小 M tile 或 GEMV 分支是下一个大项。
- embed 表（tied，1.21 GiB）是现在最大的单块，需要"查表即反量化"的 gather 臂。

---

## 八、复现

```bash
source .local-state/build-env.sh
cargo build --release -p server --no-default-features --features arch-sm120
# 4B NVFP4（BF16 检查点 → 装载期现量化）
.local-state/... -p <nvfp4 profile>        # [model] format=safetensors quant=nvfp4 head_quant=off
# 判定三件套
[load] verify：checked=N mismatch=0        # 槽字节 == host 打包（设备臂内部也逐字节对拍）
[selfcheck] nvfp4_gemm ✓ max|err|  ≤ 1e-2  # 装配期 oracle
[chat] 指标: decode=… spec rounds=… al=…   # 文本质量（B/tok > 2.5）+ 接受率
```

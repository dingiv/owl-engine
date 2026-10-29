# FP8 → 4bit 离线转换器（施工图）

> 状态：**设计冻结，未实现**（2026-10-29）。
> 定位：`apps/converter/` 独立进程；把普通 HF FP8 检查点转成 owl 可直接装载的
> 4bit 检查点。**运行时不改**：owl 继续只读现成的 W4A16 / AWQ 包。
> 关联：`new-format-checklist.md`、`docs/arch/装载管线-源精度x目标格式-设计.md`、
> `kernels/kernel-admission.md`。

---

## 1. 为什么是离线转换器（而不是装载期现量化）

用户裁决（2026-10-29）：**FP8 → INT4 的精度损失不能靠装载期一把梭**。

**已裁决（同日）**：

- 实现形态 = `apps/converter/` 的 **Python uv 独立项目**（不做 Rust 子进程桥）；
- 默认产出 = **AWQ g32-asym**（`(4,32,false)`）；g128-sym 只作 GPTQ/RTN 备选与 A/B。

| 维度 | 装载期现量化 | 离线转换器 |
|---|---|---|
| 精度 | 只能 RTN；FP8 已经损失过一次，再裸 RTN 到 4bit = 二次损失叠加 | 可用 **AWQ/GPTQ 校准**，按激活敏感度补偿；可做层间混合精度 |
| 质量门 | 起服务时才发现差，回滚成本高 | 离线跑 GSM8K / needle / std，**不合格不进库** |
| 装载器 | 要加 fp8 源 + scale 语义 + W4A16 打包变换 | **零改动**：只加一个 `W4A16Source`/`AwqSource` 能读的目录 |
| 失败隔离 | 转换失败 = 起机失败 | 转换失败 ≠ 服务失败；源检查点原样保留 |
| 可复现 | 与运行时版本耦合 | 工具版本 / 校准集 / 方案全部写进 manifest |

**结论**：转换器是独立工具，owl 引擎不承担“猜 FP8 scale / 现场 RTN”的责任。

**输入不只 FP8**：BF16/F16/F32 **dense** 也是合法源（AWQ 的标准输入；
本机测试模型就是 BF16 VLM）。P0 的 `inspect` 已能识别 `dense-bf16` 并直接
预估 4bit 体积；FP8 的三种 scale 形态用合成档覆盖。


### 非目标

- 不做运行时 FP8 装载（`SrcDtype` 的 fp8 分支仍留给未来的 P4）。
- 不做 GGUF、不做 W4A8、不做 NVFP4（NVFP4 已有运行时臂，另案）。
- 不做 TP 分片格式的重新发明：输出必须能被现有 `formats::shard` 按
  `weight_shape` 切开；27B 的 K/N 维天然满足 g32/g128 对齐。
- 不把转换器塞进 `server` 的启动路径。

---

## 2. 社区工具的真实能力边界（先破三个误解）

| 误解 | 事实 |
|---|---|
| “AutoAWQ 能把 FP8 转 AWQ INT4” | **AutoAWQ 已废弃**；其功能由 `llm-compressor` 接管。`AutoAWQConverter` 只做 **AutoAWQ → compressed-tensors** 的格式转换，不负责“从 FP8 训练出 AWQ” |
| “`convert_checkpoint` 能一步出 4bit” | 它只做**格式/精度转换**。FP8 → 4bit 必须两步：**反量化到 dense** → **校准量化** |
| “AWQ 默认配置 owl 能读” | AWQ 默认 `(num_bits=4, group_size=128, symmetric=false)`；owl 的 `quant_family` **只接受** `(4,32,false)` 与 `(4,128,true)`，默认配置会被 **boot 拒** |

### 2.1 实际可用的工具链

```
compressed-tensors（格式层，model-free，按 shard 流式）
  ├─ FP8BlockDequantizer        blockwise 128×128（DeepSeek 系）
  ├─ CompressedTensorsDequantizer  任意 compressed-tensors 方案 → dense
  └─ AutoAWQConverter           已有 AutoAWQ → CT W4A16（不是 FP8 入口）

llm-compressor（量化层）
  ├─ oneshot + AWQModifier      需要校准集 / GPU（AWQ）
  ├─ oneshot + GPTQModifier     需要校准集 / GPU（GPTQ）
  └─ model_free_ptq             只做 data-free（RTN / FP8 / NVFP4），不支持 AWQ
```

**per-tensor FP8**（`weight_scale` 标量）不一定被 `FP8BlockDequantizer` 覆盖；
转换器需要一个**本地反量化插件**作为兜底（§4.2）。

---

## 3. owl 侧的硬合同（输出必须长这样）

`load_from_dir` 不看 `model.format`，它按**检查点事实**分族：

```
detect(keys) ──> PackQuantized ──> quant_family(config.json)
                                      ├─ (4, 32, false)  ⇒ AwqSource    + QuantPlan::W4A16Awq
                                      └─ (4, 128, true)  ⇒ W4A16Source + QuantPlan::W4A16
                                      └─ 其它            ⇒ 响亮拒
```

### 3.1 两种合法输出

| 方案 | `config_groups.group_0.weights` | owl 源 | 键面（每 Linear） |
|---|---|---|---|
| **AWQ g32-asym** | `num_bits=4, group_size=32, symmetric=false` | `AwqSource` | `weight_packed` I32 `[out, in/8]`、`weight_scale` BF16 `[out, in/32]`、`weight_zero_point` I32 `[out/8, in/32]`、`weight_shape` |
| **W4A16 g128-sym** | `num_bits=4, group_size=128, symmetric=true` | `W4A16Source` | `weight_packed` I32 `[out, in/8]`、`weight_scale` BF16 `[out, in/128]`、`weight_shape` |

反量化语义（owl 源侧已定谳）：

- g32-asym：`W = (q − zp_group) × scale`，`q ∈ [0,15]`；
- g128-sym：`W = (q − 8) × scale`（U4B8，无 zp 张量）。

### 3.2 被拒的方案（不要产出）

- `(4,128,false)`：llm-compressor AWQ 的**默认值**，owl 无 g128-asym 源。
- `(8,*)`：W8 家族未接源族。
- 只有 `qweight/scales/qzeros` 的经典 AutoAWQ 目录：`AwqSource::linear_form`
  只认 `weight_packed` 或裸 `weight`，经典 AWQ 会“缺键”。
- `F8_E4M3FNUZ`、MXFP8（E8M0 块缩放）：格式表外，**转换入口直接拒**。

### 3.3 配置面

| 输出 | `[model]` 配置 | 备注 |
|---|---|---|
| g32-asym | `format = "awq"`、`quant = "w4a16"` | `quant_plan_of` → `W4A16Awq` |
| g128-sym | `format = "safetensors"`、`quant = "w4a16"` | → `W4A16`；loader 仍按 PackQuantized 探测 |

入口档面**不新增**：转换产物用 `--set model.dir=/path/to/converted` 或
`profiles/measurement/` 下的测量档；常驻档门（`profiles.rs`）不放开。

---

## 4. 转换流水线

```
apps/converter（uv 项目；console script `owl-convert`）
  │
  ├─ inspect   ← 只读：config.json + safetensors index
  │    ├─ 判 FP8 形态：per-tensor / per-channel / blockwise(128×128)
  │    ├─ 盘点 dtype / 分片 / tied embeddings / MTP / vision 资产
  │    └─ 输出 plan.json（张量数、字节、预计产物、拒启项）
  │
  ├─ dequant   ← compressed-tensors convert_checkpoint（shard-wise）
  │    ├─ blockwise：FP8BlockDequantizer
  │    ├─ per-tensor/per-channel：本地插件（§4.2）
  │    └─ 产物：dense BF16 临时目录（不覆盖源）
  │
  ├─ quantize  ← llm-compressor（Python 进程内调用）
  │    ├─ rtn：model_free_ptq（data-free，先跑通格式）
  │    ├─ awq：oneshot + AWQModifier（校准集）
  │    └─ gptq：oneshot + GPTQModifier（校准集）
  │
  ├─ normalize ← 校验/改写 config_groups，确保落在 §3.1 两格里
  │
  ├─ verify    ← 结构门 + 数值门 + owl 装载冒烟 + 质量门
  │
  └─ package   ← 拷贝 tokenizer/模板/视觉资产 + manifest.json + README
```

### 4.1 已裁决：Python uv 独立项目

AWQ/GPTQ/compressed-tensors 全部是 Python 生态；用 Rust 包一层只会多出
“子进程 + venv 路径 + 错误桥接”三类故障。2026-10-29 用户裁决：
`apps/converter/` 用 **uv 独立项目**（py3.12，`.venv` 在包内），**不做**
Rust CLI 包装；入口：

```bash
cd apps/converter
uv run owl-convert inspect /path/to/model-fp8
uv run owl-convert run /path/to/model-fp8 /path/to/model-w4a16 \
  --scheme awq-g32 --calib calibration/ultrachat-128.jsonl --device cuda:1
```

> 与 `crates/kernels` 的 triton venv 同纪律：**独立 venv、锁版本、不碰系统
> python**。`apps/converter/pyproject.toml` + `uv.lock` 入库。

### 4.2 FP8 反量化（源侧语义）

| 形态 | 键 | 反量化 |
|---|---|---|
| per-tensor | `*.weight_scale` f32 `[1]` | `w = fp8 × scale` |
| per-channel | `*.weight_scale` f32 `[out]` | 沿 K 广播 |
| blockwise | `*.weight_scale_inv` `[ceil(out/128), ceil(in/128)]` | `w = fp8 / inv`（DeepSeek 系；**以源 config 的 `weight_block_size` 为准**） |
| 激活 scale | `*.input_scale` 等 | **忽略**：输出激活是 F16，权重转换不需要它 |

规则：

- 只反量化 **Linear 权重**；norm / embedding / lm_head / vision tower 保持源
  dtype（BF16/F16）原样拷贝。
- 逐 shard 流式；峰值 = 源 shard + 产物 shard（27B ≈ 2×5GB），不整模型驻留。
- `F8_E4M3FNUZ` / `F8_E5M2`：前者拒；后者允许但记入 manifest（owl 侧未来
  `decode_to_f32` 已支持 e5m2）。
- 反量化后做一次 **抽样对拍**：随机 1% 行，`fp8×scale` 与 torch 反量化逐位
  对齐；不通过 = 拒绝继续。

### 4.3 量化方案（owl 两格）

| 方案 | 算法 | 校准 | 输出 | 适用 |
|---|---|---|---|---|
| `awq-g32` | AWQ（激活感知） | 128–256 条，seqlen 2048 | `(4,32,false)` | **推荐**：贴近 AWQ 原生语义；owl `AwqSource` 已生产验证（cyankiwi 线） |
| `w4a16-g128` | GPTQ 或 RTN | GPTQ 需校准；RTN 不需要 | `(4,128,true)` | 想要“标准 W4A16”/更少 scales；质量需 A/B |
| `rtn-g128` | RTN | 无 | `(4,128,true)` | 只用于**格式冒烟**，不作为质量交付 |

> ✅ **已裁决（2026-10-29）**：默认走 **AWQ g32-asym**（贴近 AWQ 原生语义；
> owl `AwqSource` 已有生产验证）。g128-sym 作为 P3 的 GPTQ/RTN A/B 路线，
> **不**把 AWQ 强行掰成 symmetric；两套都产出后再用 §7 的门定生产档。

量化器的硬约束（写进 `normalize`）：

- `group_size` 必须整除 `in`；27B 的 5120/17408 等维度对 32/128 均可整除。
- `symmetric` 与 `group_size` 只能取 §3.1 两格；其余直接报错。
- `ignore` 至少包含：`lm_head`、`embed_tokens`、`*norm*`、`vision*`、`mtp.*`
  （MTP 头小，保留 F16；DFlash2 草稿是**另一个模型**，不在本转换器范围）。
- 输出 `format` 必须是 `pack-quantized`；写回 `config.json` 的
  `quantization_config` 必须带 `config_groups.group_0.weights` 四元组。

---

## 5. CLI 合同

```text
owl-convert inspect  <src> [--json]
owl-convert dequant  <src> <dense-dir> [--workers N] [--keep-fp8]
owl-convert quantize <dense-dir> <out-dir> --scheme awq-g32|w4a16-g128|rtn-g128
                     [--calib FILE] [--samples 128] [--seqlen 2048]
                     [--device cuda:1] [--max-workers N]
owl-convert normalize <out-dir> --scheme ...
owl-convert verify   <src> <out-dir> [--owl-bin ./target/release/server]
                     [--quality gsm8k,needle] [--std]
owl-convert run      <src> <out-dir> --scheme ... [其余参数透传]
```

约定：

- 每个子命令都可**单独重跑**；中间目录用 `.work/`，失败不删源、不覆盖已有产物。
- `run` = `inspect → dequant → quantize → normalize → verify → package`；
  任一阶段非零退出即停，错误信息带“哪一步/哪个键/怎么办”。
- `--dry-run` 只打印 plan，不写盘。
- 所有随机性（校准采样）用固定 seed，写进 manifest。

### 5.1 产物目录

```
<out-dir>/
  config.json                     # quantization_config 已落到 §3.1
  model.safetensors.index.json
  model-00001-of-000NN.safetensors
  tokenizer.json / tokenizer_config.json / chat_template.jinja
  generation_config.json
  preprocessor_config.json        # VLM 档
  converter-manifest.json         # 见下
  README.converted.md             # 怎么起 owl（测量档/--set 示例）
```

`converter-manifest.json`：

```json
{
  "source": {"path_hash": "...", "dtypes": ["F8_E4M3", "BF16"], "scheme": "fp8-block"},
  "tools": {"owl-convert": "...", "llm-compressor": "...", "compressed-tensors": "...", "torch": "..."},
  "quant": {"algorithm": "awq", "group_size": 32, "symmetric": false, "num_bits": 4},
  "calib": {"path_hash": "...", "samples": 128, "seqlen": 2048, "seed": 0},
  "gates": {"dequant_rel_l2": 0.0, "gsm8k": 0.0, "needle": 0.0, "std_decode": 0.0},
  "notes": ["lm_head/embed/norm/vision/mtp 保持 F16"]
}
```

---

## 6. 显存 / 时间 / 磁盘预算

以 27B、FP8 源 ≈ 27GB、dense BF16 ≈ 54GB、4bit ≈ 14GB 计：

| 阶段 | 峰值 | 预估 | 说明 |
|---|---|---|---|
| inspect | <1GB | 秒级 | 只读 index/config |
| dequant | 源 shard + 产物 shard ≈ 10–15GB | 10–20 min | shard-wise；`max_workers` 受 RAM/磁盘限制 |
| RTN | 1 张卡 / 或 CPU | 10–20 min | data-free |
| AWQ/GPTQ | GPU 显存 ≥ 单层 + 激活；宿主 ≈ 54GB | 1–3 h（单 3090，含 offload） | 27B 不能整模型上单卡；用 `device_map=auto`/顺序管线 |
| verify | <1GB（CLI）+ 起服务时按档 | 分钟级 + 质量门时长 | 质量门见 §7 |

**磁盘**：`源 27G + .work/dense 54G + 产物 14G ≈ 95G`；转换器在 `inspect`
阶段检查剩余空间，低于 1.5× 预算直接拒。保留源检查点，`.work/` 可 `--keep-work`
或成功后清理。

**并行纪律**：dequant 按 shard 并行（`max_workers=auto`，但显式受
`--max-workers` 限制）；AWQ/GPTQ 默认单进程，避免多卡争用与 P2P 雷区。
转换器**不碰** GPU0/3 的 ComfyUI 卡；`--device` 显式指定。

---

## 7. 验收门（不过门不进库）

### 7.1 结构门（必过）

```bash
# 1) 方案四元组
jq '.quantization_config.config_groups.group_0.weights' config.json
# 2) 键面：每个 Linear 都有 weight_packed/weight_scale[/weight_zero_point]/weight_shape
# 3) owl 预检（不加载权重）
./target/release/server -p <测量档> --check
# 4) 真装载冒烟（短生成）
./target/release/server -p <测量档> --log-level info
curl .../v1/chat/completions -d '{"messages":[{"role":"user","content":"17*23"}],"max_tokens":32}'
```

判据：`--check` 绿；装载日志无“缺键/方案未接”；输出 `391`（或同档基线答案）。

### 7.2 数值门

| 门 | 判据 | 工具 |
|---|---|---|
| 反量化对拍 | 抽样 1% 行，与源 FP8×scale 逐位一致 | `owl-convert verify` 内置 |
| 4bit 重构误差 | 每张量相对 L2 ≤ 该方案理论量级（g32 优于 g128） | 内置 dequant 反查 |
| GSM8K 200 | 相对 FP8 **下降 ≤ 1.0 pt**；相对 BF16 下降记录在案 | `tools/quality_eval.py` |
| MMLU-Pro smoke | 不低于 FP8 − 1.5 pt | 同上 |
| needle 32k/131k | 通过率 ≥ FP8 − 2 pt | `tools/needle_eval.py` |
| agent 工具调用 | 9 发 ≥ 8 成功（与 `agent-surface` 同口径） | pi 真机回放 |
| std 测速 | decode ≥ 现有 27B W4A16 档 0.95×；prefill 记录 | `llm_speedtest --flavor std` |

### 7.3 质量归因

同时保留三条臂：

1. **BF16**（源真值，若能放下）；
2. **FP8**（转换输入，vLLM 参考）；
3. **4bit**（产物）。

只报“4bit vs BF16”会把 FP8 自身的损失算到转换器头上；必须报
“4bit vs FP8”才是转换器的真实代价。

---

## 8. 施工分解（P0–P4）

| 阶段 | 内容 | 出口判据 |
|---|---|---|
| **P0 inspect** | 读 config/index，识别 per-tensor/per-channel/blockwise；盘点 ignore 面；空间检查；plan.json | 对 1 个真实 FP8 档 + 1 个合成档输出正确 plan |
| **P1 RTN g128 冒烟** | `convert_checkpoint` 反量化 + `model_free_ptq` W4A16 RTN；normalize/verify | owl 能装载、短生成正确；**只证明格式链路** |
| **P2 AWQ g32** | `oneshot + AWQModifier`（group 32, asym）+ 校准集 + 质量门 | **默认生产线**；GSM8K/needle/agent 三门过；std 数字归档 |
| **P3 GPTQ g128** | `oneshot + GPTQModifier`（group 128, sym）+ 与 P2 A/B | 可选；质量/速度对比表；不达标则只保留 g32 |
| **P4 收口** | 文档、测量档、manifest、失败回滚、工具版本锁 | 新机器按 README 一条命令复现 |

每阶段独立可停；P1 不通过就不要进 P2（格式链路优先于质量调优）。

---

## 9. 风险与开放问题

1. **FP8 源形态未定**：仓库里没有真实 FP8 检查点。P0 先用
   `llm-compressor` 从 BF16 现场造一个 FP8（per-tensor 与 blockwise 各一），
   再找目标模型真档。
2. **g128-sym 是否需要**：已裁决默认 AWQ g32；g128-sym 仅在 P3 做 GPTQ/RTN
   A/B。若 GPTQ 校准成本过高，可只保留 g32 一条生产线。
3. **27B AWQ 校准显存**：单卡 24G 放不下整模型，需 `device_map=auto`/
   顺序管线；若 llm-compressor 的 AWQ 不支持分片校准，退到 GPTQ 或
   2×3090 的层流水线（需另立小刀）。
4. **tokenizer/模板/视觉资产**：必须逐字节拷贝；VLM 的
   `preprocessor_config.json`、image token 配置不能漏。
5. **MTP / DFlash2**：主检查点内的 `mtp.*` 头保持 F16；DFlash2 草稿是
   独立模型，不在本转换器范围（已有 syvai W4A16 线）。
6. **输出兼容性回归**：llm-compressor 版本升级可能改变 `weight_packed`
   布局或 config 字段；`uv.lock` 必须锁死，`normalize` 做结构 gate。
7. **许可/来源**：转换器只处理用户本地权重；不下载、不再分发模型。

---

## 10. 一页结论

- **能做**，但不是在装载器里“顺手转”：FP8→INT4 是**有损二次量化**，
  必须离线 + 校准 + 质量门。
- 社区链路 = `compressed-tensors` 反量化 → `llm-compressor` AWQ/GPTQ；
  AutoAWQ 已废弃，不能作为 FP8 入口。
- 默认产出 = **AWQ g32-asym**；**W4A16 g128-sym** 作为 GPTQ/RTN 备选
  （AWQ 默认的 g128-asym 会被 boot 拒）。
- 转换器 = `apps/converter/` 的独立 Python uv 项目；引擎零改动；产物用
  `--set model.dir=...` 或测量档起服务。
- 先做 P0/P1 证明格式链路，再用 P2/P3 做质量 A/B；不过门不进库。


---

## 11. 实现记录（2026-10-29 · P0）

### 11.1 落地

| 文件 | 内容 |
|---|---|
| `apps/converter/` | Python uv 独立项目（py3.12，无运行时依赖；`uv.lock` 入库） |
| `src/owl_converter/st_index.py` | safetensors 头解析（8B 长度 + JSON）、分片索引、**下载完整性**（头声明 vs 文件实际大小）、重复键/截断头响亮拒 |
| `src/owl_converter/source_profile.py` | `dense-bf16/f16/f32`、`fp8-per-tensor/per-channel/blockwise`、`fp8-mixed`、`pre-quantized-4bit`；FNUZ/缺 scale/未知 scale 形状结构化拒 |
| `src/owl_converter/inventory.py` | `marlin_eligible`(out%256/in%128) 清单、跳过原因、AWQ g32 / W4A16 g128 体积 |
| `src/owl_converter/plan.py` | `owl-convert-plan/1` JSON、人读摘要、磁盘检查、下一步命令 |
| `src/owl_converter/cli.py` | `inspect` + P1/P2 占位；退出码 0/2/3/4/64 |
| `tests/` | 24 项：头解析/不完整/索引/重复键、FP8 形态、清单/体积、CLI |

### 11.2 真实模型（BF16 dense VLM，Qwen3.5 架构）

| 项 | 数值 |
|---|---:|
| 张量 / 体积 | 738 / 8.68 GiB |
| dtype | BF16=738 |
| 可量化 Linear | **200**（另有 48 个 marlin-ineligible） |
| 跳过 | vision 100 · norm 161 · not-weight 196 · non-linear 24 · MTP 8 · embedding 1 |
| 4bit 预估 | q4 1.92 GiB + keep 2.04 GiB = **3.96 GiB** |
| 状态 | complete（下载完成后 `inspect` 自动转绿） |

> 该模型 `tie_word_embeddings=true`，没有独立 `lm_head.weight`；vision/MTP
> 按设计保持 F16，不参与 4bit。

### 11.3 下一步（P2 前置）

- `llmcompressor==0.15.0` 当前要求 `torch>=2.10`、`transformers>=5.15`
  （`compressed-tensors==0.20.1`）；需先验证 `qwen3_5` 架构能否被
  transformers 加载。若 VLM 入口不被 `AutoModelForCausalLM` 接受，改用
  `AutoModelForImageTextToText` 或 `model_free_ptq`（RTN 格式冒烟）。
- P2 默认线仍是 **AWQ g32-asym**；校准集/显存计划见 §6。

### 11.4 P1 实测与 owl 集成修复（2026-10-29，4B BF16 VLM）

**输入**：本机 4B BF16 dense VLM（Qwen3.5 架构，738 张量 / 8.68 GiB）。
**产物**：`model_free_ptq(scheme="W4A16")`（RTN g128-sym）耗时 **6.5 s**，
200 个 Linear 打包，目录 3.75 GiB；`owl-convert normalize` 把 config group
`W4A16` 改名为 owl 唯一可读的 `group_0`。

#### 三个集成缺口（已修）

1. **owl 没有 pack-quantized W4A16 的入口**
   - `format = "safetensors"` + `quant = "w4a16"` 走 `safetensors_load_plan` ⇒ **F16**；
   - `format = "awq"` 只服务 27B 的 g32-asym（`load_qwen38_27b_awq_with_embed`）；
   - 新增 `ModelFormat::PackQuantized`（serde `"pack-quantized"`）+
     `quant_plan_of(PackQuantized, W4a16) = W4A16`，server 走通用
     `load_qwen35_safetensors`；`validate` 只允许 `w4a16`，且 `head_quant`/`embed_quant`
     必须 off/f16（非 AWQ 源没有派生键）。
   - 配置写法：`[model] format = "pack-quantized"`、`quant = "w4a16"`、`head_quant = "off"`、
     `embed_quant = "f16"`。

2. **`W4A16Source` 合并键（`gate_up_proj` / `in_proj_qkvz`）CPU 路径错误**
   - 原实现把两半**已经 marlin 重排**的 `qweight` / `scales` 直接字节拼接；
   - 实测：marlin B 的 N 轴有 tile 交错、marlin S 的 64 元素 perm 按整行 out 排布
     ⇒ **半矩阵拼接 ≠ 整体重排**（`kernels/tests/concat_check.rs` 两道门）；
   - 修法：合并 `.qweight` 先解包两半原始 nibble、拼成 `[2I,k]` 再整体
     `pack_marlin_b_gather_into`；合并 `.scales` 先拼原始 `[out,groups]` 再整体
     `pack_marlin_s`；`.packed_raw` / `.weight` 仍可安全拼接（行主序）。
   - 现场判据：修复前输出乱码/全换行；修复后 `17*23` → **391**。

3. **`normalize` 的 group 命名**
   - `model_free_ptq` 写 `config_groups.W4A16`；owl `quant_family` 硬编码
     `config_groups.group_0.weights` ⇒ 不改名就 boot 拒；
   - `owl-convert normalize` 保留原 group 字段、只换键名，并校验
     packed/scale/zp/shape 的 dtype 与形状。

#### 端到端证据（同一 owl release 二进制）

| 输入 | 结果 |
|---|---|
| BF16 原模型（F16 计划） | `17*23` → 391 ✓ |
| 4bit 产物 → 官方 decompressor → BF16 目录 | 391 ✓（证明转换数值正确） |
| **4bit 产物（W4A16 g128-sym）** | **391 ✓**（修复后） |

#### 仍未收口

- 该路径默认走 **CPU 重排**（`load_qwen35_safetensors` 的 `shard_opts` 用
  `LoadOpts::plain`，`[load] repack` 尚未接线）；设备 `CtRepack` 对合并键的
  `packed_raw` 拼接是正确的（行主序），但尚未在 4B 档实测。
- P2 AWQ g32-asym 与质量门（GSM8K / needle / agent / std）未做；`quantize`
  子命令与 `verify` 仍是占位。
- TP2 下 `PackQuantized` 的分片/合并键路径未验。

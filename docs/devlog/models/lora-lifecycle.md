# LoRA 动态微调（LoRA Lifecycle）提案

> 状态：**提案，未施工**（2026-10-21）
>
> 目标读者：owl 引擎 / models / kernels / server 维护者，以及前端 agent 作者。
>
> 本文覆盖 LoRA 的**全生命周期**：数据采集 → 数据集快照 → 训练 → 评估 →
> adapter 注册 → 推理加载/激活 → 回滚。本文只做提案分析，不涉及代码落地。
>
> 用户已裁决的边界：
> - 第一版训练目标：**0.8B / 4B 验证**；
> - 训练窗口：**停服训练**；
> - 自主程度：**全手工触发**；
> - 数据来源：**前端 agent**；
> - 倾向：**内建训练能力**，而不是把训练完全外包给外部框架；
> - 对外暴露更多接口供 agent 使用。

---

## 0. 一句话结论

先做 **LoRA 推理面**（加载、激活、按请求使用、adapter 生命周期），再做**owl 内建训练面**：
在停服窗口内，用 owl 自己的模型定义与 CUDA 算子训练 LoRA A/B，数据由前端 agent 通过
学习 API 写入、人工创建数据集并手工提交训练任务；训练产物经过评估门后注册为 adapter，
推理侧重新启动并激活。

**不把 27B 原生训练作为第一目标**。当前 owl 没有 autograd/backward/optimizer；0.8B/4B
是验证“数据 → 训练 → 评估 → 上线”闭环的最小载体。27B 的原生训练需要额外的
activation checkpoint、量化权重反向、TP 梯度同步，单独立项。

---

## 1. 需求

### 1.1 功能需求

| 编号 | 需求 | 优先级 |
|---|---|---|
| LF-01 | 推理引擎能加载 PEFT 风格 LoRA 权重（A/B、rank、alpha、target modules） | P0 |
| LF-02 | 一次只激活一个 adapter；chat 请求可显式指定 adapter | P0 |
| LF-03 | adapter 关闭时，输出与无 LoRA 路径逐位一致 | P0 |
| LF-04 | adapter 激活/切换不要求重启整个模型；图按需重捕或 adapter 槽稳定 | P0 |
| LF-05 | 前端 agent 可写入训练样本、创建不可变数据集快照 | P1 |
| LF-06 | 人工创建/批准训练任务；系统不自动训练、不自动上线 | P1 |
| LF-07 | 停服训练：训练进程独占 GPU，推理服务停止 | P1 |
| LF-08 | 首版支持 0.8B/4B 的 SFT LoRA（assistant-only loss） | P1 |
| LF-09 | 训练产物有 base/tokenizer/dataset/config 指纹，评估通过才可激活 | P1 |
| LF-10 | 支持回滚到上一个 adapter 或“无 adapter” | P1 |
| LF-11 | 数据保留、删除、脱敏、来源审计可操作 | P1 |
| LF-12 | 训练失败（OOM/NaN/被杀）不产生半成品 adapter | P1 |
| LF-13 | adapter / base 的 KV、前缀缓存、session 命名空间隔离 | P0 |

### 1.2 非目标（本提案明确不做）

- **不做全参数微调**：只训练 LoRA A/B，base 权重冻结。
- **不做自动自学习闭环**：不自动采集→自动训练→自动上线；全部人工触发/审批。
- **不把 LoRA 当知识库**：文档事实优先 RAG/记忆库；LoRA 用于风格、格式、工具习惯、领域措辞。
- **不承诺 27B 原生训练**：第一版只验证 0.8B/4B；27B 需要单独评估量化反向与显存。
- **不做训练与推理常驻共存**：训练期间停服，释放 KV/图/池。
- **不做多 adapter 同时驻留/混合**：v1 单 active adapter；多 adapter 是后续。
- **不做 DPO/RLHF**：v1 只做 SFT；偏好训练后续。
- **不做跨模型 adapter 复用**：adapter 绑定 base model fingerprint 与 tokenizer。

---

## 2. 当前架构事实

### 2.1 纯推理引擎

- `TensorOps` 是声明式前向 DAG；`Model::forward` / `Module::forward` 只产出前向节点。
- `ForwardCtx` 携带 KV/GDN/rope/FI/图等推理态；没有 `requires_grad` / tape / backward。
- `SemanticKernel` 词表里没有反向算子；`Linear` 的 W4A16/GGUF 臂只实现前向。
- `RunningEngine` 是单模型执行态；训练需要另一套 `TrainingEngine`，不能把训练塞进
  `run_actor` 的泵循环。
- 现有图捕获/`GraphFleet` 是推理图，不能用于训练；训练走 eager + 自己的反向图。

### 2.2 LoRA 推理的接入点很明确

`crates/models/src/layers/linear.rs::Linear::forward_inner` 已经有三个清晰出口：

```text
GGUF 块型 -> gguf_quant_gemm
W4A8     -> quant_int8_row + MarlinW4A8Awq
W4A16    -> MarlinW4A16 / MarlinW4A16_AWQ
f16/bf16 -> matmul_nt
```

LoRA 只是在 base 输出后追加一条低秩支路：

```text
base = Linear::base_forward(x)          // 现有路径，冻结
lora = (x @ A^T) @ B^T * (alpha / rank)
y = base + lora
```

因此 **LoRA 推理**不需要改 base 量化路径，也不需要 dequant 权重；只需要：

- adapter 权重槽（A/B）；
- 低秩 GEMM（可复用 cublas f16/bf16）；
- 加法/广播；
- adapter 生命周期与图策略。

### 2.3 训练面缺的不是“一个 loss”，而是整条反向链

要训练 LoRA，至少需要：

- 反向图/tape；
- 每个前向算子的 backward：
  - matmul / LoRA 低秩支路；
  - attention（QK^T、softmax、PV）；
  - RMSNorm / LayerNorm；
  - RoPE；
  - SiLU / gate / mul；
  - residual / add / concat / narrow；
  - embedding（只读，不需梯度）；
  - cross-entropy / logits 选择；
- GDN（Qwen3.5 混合层）的反向：gating、conv、delta rule、norm-act；
- activation checkpoint / 重算；
- AdamW / grad clip / LR schedule；
- TP 梯度 allreduce（如果多卡训练）。

其中 **GDN backward** 是 Qwen3.5-0.8B/4B 的关键路径：0.8B 是 18 层 GDN + 6 层 full
attention 的 hybrid。只做 full-attention backward 不够；要么实现 GDN backward，
要么第一版只训练最后 full-attention/MLP 段并把 GDN 段 detach（见 §7.3）。

### 2.4 前端 agent 的数据面目前不存在

当前 server 只有：

- `/v1/chat/completions`
- `/tokenize`
- `/debug/*`（诊断/灌流）
- `/v1/models`（单模型）

没有“样本写入 / 数据集 / 训练任务 / adapter 注册”接口。需要新增学习面 API。

---

## 3. 总体架构

### 3.1 三个平面

```text
┌─────────────────────────────────────────────────────────────┐
│ Control plane（agent / operator）                            │
│  - 样本写入、数据集快照、训练任务、人工审批、adapter 激活       │
└───────────────┬─────────────────────────────┬───────────────┘
                │                             │
┌───────────────▼──────────────┐  ┌───────────▼───────────────┐
│ Inference plane              │  │ Learning plane            │
│ owl-server（常驻）            │  │ owl-train（停服窗口）      │
│  - base model + LoRA runtime │  │  - 读数据集快照            │
│  - adapter 激活/回滚         │  │  - forward/backward        │
│  - OpenAI API                │  │  - AdamW / eval            │
└──────────────────────────────┘  └───────────────────────────┘
                │                             │
                └─────────── 共享磁盘 ─────────┘
                     datasets / adapters / evals
```

### 3.2 “内建训练能力”的边界

建议把“内建”定义为：

- 训练器是 **owl 仓库内的 `owl-train`**（Rust + 现有 CUDA 资产），不是外部 Python 服务；
- 使用 owl 的 `ModelSpec` / 层定义 / tokenizer / 数据格式；
- 训练产物是标准 adapter safetensors，可被推理面加载；
- 允许在开发期用 HF/PEFT 做**数值金标**，但生产训练不依赖 PyTorch。

训练器与推理器是**两个进程**，不是同一个 `RunningEngine`：

- 停服训练要求释放 KV/图/池/GPU 上下文；
- 进程边界是最干净的资源回收方式；
- 训练器可以复用 `owl-models` / `owl-kernels` 的代码，但不带 server/HTTP/actor。

如果未来要求“同一进程内切换推理/训练”，那属于模型热替换/训练调度器的后续阶段，
不在本提案 v1。

### 3.3 全手工触发

```text
agent 写入样本
  -> agent/人创建 dataset snapshot
  -> agent 提交 job（状态 = pending_approval）
  -> 人审批 job
  -> 人工停 owl-server
  -> 人工运行 owl-train --job <id>
  -> 训练 + eval
  -> 人审阅 eval 报告
  -> 人启动 owl-server
  -> 人/agent 调用 adapter activate
```

v1 **不做**：
- idle 自动开始训练；
- 自动上线 adapter；
- 自动把用户对话加入训练集（必须显式写入/同意）。

---

## 4. LoRA 推理面

### 4.1 adapter 数据模型

```rust
struct LoraAdapter {
    id: String,                 // "user-style-v3"
    base_fingerprint: String,   // base model + tokenizer + config
    rank: usize,
    alpha: f32,
    dtype: Dtype,               // F16 / BF16
    target_modules: Vec<String>,
    // 每层 A/B
    layers: Vec<LoraLayer>,
    status: AdapterStatus,      // candidate / active / rejected / archived
}

struct LoraLayer {
    module: String,             // q_proj / k_proj / ... / down_proj
    a: Weight,                  // [rank, in_dim]
    b: Weight,                  // [out_dim, rank]
}
```

文件格式：PEFT 兼容 safetensors：

```text
<adapter_dir>/
  adapter_config.json
  adapter_model.safetensors
  owl_adapter.json          # 可选：TP 分片、base fingerprint、eval id
```

`adapter_config.json` 至少含：

- `r` / `lora_alpha` / `lora_dropout`（v1 必须 0）
- `target_modules`
- `fan_in_fan_out`（v1 不支持 true，响亮拒）
- `bias`（v1 只支持 none）
- `use_dora`（v1 不支持，响亮拒）
- `use_rslora`（可选：true 时 scale = alpha / sqrt(r)，必须显式支持，不静默按 alpha/r）
- `base_model_name_or_path`

**键映射**（PEFT → owl）：

```text
PEFT: base_model.model.layers.<i>.self_attn.q_proj.lora_A.weight  [r, in]
      base_model.model.layers.<i>.self_attn.q_proj.lora_B.weight  [out, r]
owl : layers.<i>.self_attn.q_proj  (Linear 的 base key)
```

- 去掉 `base_model.` / `model.` / `language_model.` 前缀后，必须能唯一映射到 owl 的
  `Linear` 键；映射失败 = 拒载，不猜。
- A = `[r, in]`、B = `[out, r]`；与 owl `matmul_nt` 的 `[out, in]` 布局一致，不需要转置。
- 不支持 per-layer 不同 rank；v1 全局 rank/alpha 一致。

### 4.2 forward 注入

在 `Linear` 增加可选 `lora: Option<LoraPair>`：

```text
fn forward_inner(xs):
  base = 现有臂(xs)                  // 冻结
  if adapter active and self.lora:
      l = matmul_nt(xs, A)           // [m, rank]
      d = matmul_nt(l, B)            // [m, out]
      d = d * (alpha / rank)
      return add(base, d)            // dtype 对齐
  return base
```

要点：

- base 输出 dtype 可能是 F16 或 BF16（取决于模型的尺度臂）；LoRA 支路必须**铸到同 dtype**
  再 add，不能混 dtype。
- `m = 1` 时走 GEMV/小 GEMM；`m > 1` 时走 GEMM。可复用现有 cublas f16/bf16 封装。
- LoRA 参数很少（rank 8–32），不需要新的 tensor-core 专用核；先复用 cublas/naive GEMM。
- adapter 关闭时**不建节点**，保证零开销、逐位一致。
- 图捕获：v1 只允许一个 active adapter，adapter 权重块地址稳定；激活新 adapter =
  重新捕获图（或 adapter 权重作为图输入槽，后续优化）。不能在同一张图里热改指针。

### 4.3 TP 分片

LoRA A/B 必须跟 base Linear 的 TP 规则一致：

| base 规则 | 例子 | A | B |
|---|---|---|---|
| `UniformRow`（切 out） | o_proj / down_proj | 全量（输入不切） | 切 out 行 |
| `UniformCol`（切 in） | gate/up/q/k/v | 切 in 列 | 全量（输出不切） |
| `Replicate` | 小层/头 | 全量 | 全量 |

推理时每 rank 只加载自己那份；训练时每 rank 只更新自己那份，梯度按同一规则 allreduce
（27B 多卡训练阶段再启用；0.8B/4B 单卡不需要）。

### 4.4 生成上下文

`ForwardCtx` 增加：

```rust
adapter: Option<&'a LoraRuntime>
```

`LoraRuntime` 只读，包含 active adapter 的层映射与 scale；层通过 `module` 名查自己的
A/B。不要在层内读全局变量/环境变量；显式依赖律不变。

### 4.5 API

```text
GET  /v1/adapters
GET  /v1/adapters/{id}
POST /v1/adapters/load           # 从磁盘登记，不激活
POST /v1/adapters/{id}/activate
POST /v1/adapters/{id}/deactivate
POST /v1/adapters/{id}/rollback  # 激活 previous
DELETE /v1/adapters/{id}
```

chat 请求：

```json
{"model": "qwen3.5-0.8b", "adapter": "user-style-v3", "...": "..."}
```

规则：

- `adapter` 省略 = 当前 active；如果 active 为 none = base。
- `adapter` 不是 active：
  - v1 只允许**一个 active**，返回 `409 adapter_not_active`，要求先激活；
  - 不做请求级多 adapter 并行。
- adapter 在请求开始时冻结；该 turn 内不切换。
- adapter 与 base fingerprint 不匹配 = `409 adapter_base_mismatch`。

### 4.6 adapter 切换

- `activate` 是 host 操作：加载/校验 adapter → 设定 active → 触发图重捕（如果图里烘了
  adapter 指针）。
- 图重捕期间，新请求短暂 503；已有 turn 用旧图跑完。
- `deactivate` 回到 base，同样重捕。
- v1 不做 adapter 的“权重合并进 base”；导出/合并是后续。

### 4.7 KV / 前缀缓存隔离

LoRA 会改变 hidden state，进而改变 K/V（即使 adapter 只挂在 q/o/mlp，早层变化也会传播）。
因此 **KV / 前缀缓存必须按 adapter 命名空间隔离**：

- session key 从 `(model_id, session_id)` 扩为 `(model_id, adapter_id, session_id)`；
- prefix cache 的链哈希必须带 `adapter_id`（或独立目录）；
- 切换 adapter 时，旧 adapter 的 KV 不得被新 adapter 命中；
- base（无 adapter）与任一 adapter 的 KV 也不共享；
- 训练/评估阶段同样不允许用 base KV 冒充 adapter KV。

这是正确性要求，不是性能优化。若实现成本过高，v1 可以在 adapter 激活时**关闭前缀缓存**
（只保留会话内 KV），但绝不能静默复用。

### 4.8 草稿模型 LoRA（DFlash2 / MTP）—— 可选配对层

结论：**可以做，且推理侧成本很低；但训练目标与 target LoRA 不同，必须作为“配对 adapter”管理。**

#### 4.8.1 推理可行性

DFlash2 草稿是 5 层、1.92B BF16 / W4A16 约 1.1GB 的小模型，内部有：

```text
fc (memory projection)
layers.0..4:
  q/k/v/o_proj
  gate/up/down_proj
  attention_conv / mlp_conv（非 Linear）
selector:
  hidden_projection（Linear [256, 5120]）
  predecessor/successor_codebook（非 Linear，冻结）
共享 target 的 embed_tokens / lm_head
```

推理侧 LoRA 与 target 同式：在草稿的 q/k/v/o/gate/up/down（以及可选 `fc` /
`hidden_projection`）后加 `B(Ax)`。rank 8–16 的 draft adapter 只有几十 MB，
对 1.1GB 草稿和 propose 5–7ms 的成本影响很小。

**禁止项（v1）**：

- 不对共享的 `embed_tokens` / `lm_head` 挂 LoRA：target verify、draft selector 和
  target adapter 会互相污染；
- 不训练 `predecessor_codebook` / `successor_codebook`：它们是 selector 的词汇表，
  改动会破坏 candidate 语义；
- 不把 draft adapter 当成独立模型文件：它必须与 target adapter 绑定。

#### 4.8.2 为什么训练不是 SFT

DFlash2 的 propose 是：

```text
noise = embed([anchor, MASK×7])
memory = fc(concat(target hidden taps [5,19,33,47,61]))
hidden = 5 层草稿 backbone(noise, memory)
logits = target.lm_head(hidden)
drafts = selector(logits, hidden_projection, codebooks, anchor)
```

它的输出是 **7 个候选 token**，最终由 target verify 的 greedy accept 决定是否接受。
因此训练信号应该是：

- target（含 target adapter）在同样上下文下的 logits / token 分布；
- 草稿候选与 target 的一致性（KL / CE / candidate ranking）；
- 实际接受率（AL）与分域净增益。

直接拿 assistant 文本做 next-token SFT 可以作为一个 baseline，但它优化的是“草稿像
语言模型”，不是“草稿与 target 的接受行为一致”；两者可能背离。

#### 4.8.3 推荐训练顺序

```text
Step 1: 训练 target LoRA（SFT，冻结 base + draft）
Step 2: 冻结 target + target LoRA，把它当 teacher
        用同一批 prompt/continuation 生成 target hidden taps 与目标分布
        训练 draft LoRA（distillation / acceptance-aware）
Step 3: 分域评估 AL 与净增益；通过才注册为 paired adapter
```

不推荐一开始联合训练 target 和 draft：

- target adapter 每变一次，draft 的输入分布就变一次；
- 联合训练需要同时保存 target/draft 两套梯度和图，复杂度高；
- 先 target 后 draft 的 two-stage 更容易定位“是 target 变了，还是 draft 没跟上”。

#### 4.8.4 配对与失效

adapter bundle：

```text
adapters/<bundle_id>/
  target/adapter_model.safetensors
  draft/adapter_model.safetensors    # 可选
  owl_bundle.json                    # base/draft/depth/target_modules/版本
```

规则：

- `activate` 原子激活 target + draft；不允许 target adapter A 配 draft adapter B；
- target adapter 切换 = **draft KV / encoded target hidden 全部失效**：
  - DFlash2 的 draft KV 是 target hidden 的投影，target 分布变了就不能复用；
  - 必须清空 draft 池并重编码上下文窗口（DFlash2 窗口 ≤2048，成本可测）；
  - prefix cache 也按 `(model, target_adapter, draft_adapter)` 隔离；
- draft adapter 缺失 = 草稿走 base；target adapter 有、draft adapter 无是合法组合，
  但要记录 `draft_adapter=none`；
- target adapter 有、draft adapter 指纹不匹配 = 拒载，或显式 `spec=off` 降级，
  不许静默配错。

#### 4.8.5 净增益门

draft LoRA 不是免费收益：propose 变慢，只有接受率提升足够才划算。近似：

```text
speedup ≈ (1 + AL_new) / (1 + AL_base)
          × (T_verify + T_propose_base) / (T_verify + T_propose_lora)
```

- DFlash2 域定律很强：MATH/AGENT 高接受，PROSE 接近零接受；
- draft LoRA 必须在**分域**上评估，不能只看 std/随机词池；
- 默认 `spec.draft_adapter = none`；只有分域净增益 > 0 且回归不降才启用。

#### 4.8.6 与首版 0.8B/4B 的关系

DFlash2 当前是 27B 的草稿族；0.8B/4B 首版没有可配的 DFlash2 草稿。
因此：

- **L1/L3（0.8B/4B）只做 target LoRA**；
- **draft LoRA 的推理接线**可以随 adapter bundle 设计一起预留；
- **draft LoRA 的训练**放到 27B 预研/专项，不在首版承诺；
- 如果未来 0.8B/4B 也有 DFlash2/MTP 草稿，再复用同一配对协议。

---

## 5. 数据面（前端 agent）

### 5.1 样本 schema

```json
{
  "sample_id": "uuid",
  "model_id": "qwen3.5-0.8b",
  "source": "agent_chat | document | correction | tool_trace",
  "consent": {"train": true, "retain_days": 90},
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "tools": [],
  "loss_mask": "assistant_only",
  "tags": ["style", "zh", "tool-use"],
  "metadata": {"session": "...", "turn": 3},
  "content_hash": "sha256:..."
}
```

规则：

- 只存**原始对话/文档**，不预先 tokenize；训练时用目标模型 tokenizer。
- `loss_mask` 默认 `assistant_only`；system/user 不计 loss。
- 文档样本必须显式标注 `source=document`、`consent.train=true`。
- 拒绝训练的来源（隐私、未同意、内部机密）在写入时就拒，不靠训练时过滤。
- 去重：`content_hash` + 近似去重；重复样本在数据集快照时合并/丢弃。
- PII：写入时做可配置 redaction；redaction 前原文不落训练集。
- 删除：按 sample_id / tag / 时间范围删除；数据集快照是不可变的，删除样本不改变旧快照，
  但会从未来快照中消失。

### 5.2 数据集快照

```text
POST /v1/learning/datasets
{
  "name": "style-2026-10",
  "model_id": "qwen3.5-0.8b",
  "filter": {"tags": ["style"], "since": "...", "sources": ["agent_chat"]},
  "sample_ids": ["..."]        // 与 filter 二选一
}
```

产物：

```text
datasets/<dataset_id>/
  manifest.json       # 样本 id、hash、数量、token 统计、模型/tokenizer 指纹
  train.jsonl
  val.jsonl
  test.jsonl
  stats.json
```

不可变；训练 job 只能引用 dataset id。这样训练可复现、可审计。

### 5.3 学习 API

```text
POST /v1/learning/samples              # 写入单条/批量
GET  /v1/learning/samples/{id}
DELETE /v1/learning/samples/{id}
POST /v1/learning/datasets             # 创建快照
GET  /v1/learning/datasets
GET  /v1/learning/datasets/{id}
DELETE /v1/learning/datasets/{id}

POST /v1/learning/jobs                 # 创建训练任务（pending_approval）
GET  /v1/learning/jobs
GET  /v1/learning/jobs/{id}
POST /v1/learning/jobs/{id}/approve    # 人工审批
POST /v1/learning/jobs/{id}/cancel
GET  /v1/learning/jobs/{id}/events     # 日志/进度
GET  /v1/learning/jobs/{id}/report     # eval 报告
```

`POST /v1/learning/jobs` 只创建任务，不启动 GPU 训练。真正的训练由人工停服后运行：

```text
owl-train --job /var/lib/owl/learning/jobs/<id>.json
```

这样“agent 可准备、可查询、可提交”，但“是否训练、何时训练”仍由人控制。

---

## 6. 训练任务与停服流程

### 6.1 job spec

```json
{
  "job_id": "lora-2026-10-21-01",
  "base_model": "qwen3.5-0.8b",
  "dataset_id": "ds-...",
  "adapter_id": "user-style-v4",
  "method": "sft_lora",
  "lora": {"rank": 16, "alpha": 32, "dropout": 0, "target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]},
  "train": {"seq_len": 2048, "micro_batch": 1, "grad_accum": 8, "lr": 1e-4, "epochs": 1, "warmup": 0.03, "max_steps": 0},
  "eval": {"suite": "default", "max_regression": 0.02},
  "resources": {"gpus": [2, 3], "max_minutes": 120},
  "approval": {"required": true, "approved_by": null}
}
```

校验：

- base model / tokenizer / dataset 指纹必须匹配；
- `target_modules` 必须是已知 Linear 名；
- `rank > 0`、`alpha > 0`、`dropout == 0`；
- `seq_len` 不超过训练配置上限；
- GPU 与推理 profile 的卡位一致（停服后复用同一对卡）；
- 磁盘配额足够放 adapter + checkpoint + eval 输出。

### 6.2 停服顺序

```text
1. 人工停 owl-server（SIGTERM，等待 GPU 释放）
2. 确认 nvidia-smi 无 compute app、GPU 卡锁已释放
3. owl-train 获取 GPU 卡锁
4. 装载 base 模型（BF16；首版不用 AWQ）
5. 装载 LoRA 初始权重（新 adapter = 零初始化 B / 随机 A，或从已有 adapter 继续）
6. 训练循环 + 周期性 eval
7. 写 adapter safetensors + report
8. 释放 GPU、释放卡锁
9. 人工启动 owl-server，登记/激活 adapter
```

训练期间**没有推理服务**；这是用户已确认的停服训练。

### 6.3 断点与失败

- 每 N 步写 checkpoint（LoRA A/B + optimizer state + step + RNG）。
- `max_minutes` 到点：保存最后 checkpoint，标记 `timeout`，不产出可上线 adapter。
- OOM：降低 `seq_len`/`micro_batch`/`rank` 后重试；不静默改配置，必须写进 job 事件。
- NaN/Inf：立即停，保留最后健康 checkpoint，标记 `diverged`。
- 被 SIGTERM：保存 checkpoint，标记 `interrupted`。
- 失败 job 不注册 adapter；半成品放在 `candidates/`，不可激活。

---

## 7. 内建训练核心

### 7.1 训练图

不在 `TensorOps` 推理 DAG 上直接加梯度，而是新增训练专用图：

```text
TrainTensor {
  data: Bytes,          // 设备块
  grad: Option<Bytes>,  // 设备块
  op: TrainOp,
  inputs: Vec<TrainTensor>,
  requires_grad: bool,
}
```

- 前向：由现有 `Module::forward` 的“训练版”生成 `TrainTensor` 图；
- 反向：从 loss 出发，按拓扑逆序生成梯度节点；
- 只对 LoRA A/B 累积梯度；base 权重 `requires_grad=false`；
- 训练图是 eager 执行，不进 `GraphFleet`；不要求图捕获安全。

### 7.2 首版算子范围

**P1（0.8B/4B，BF16 base）需要：**

| 前向 | 反向 |
|---|---|
| embedding | 无（冻结，只读） |
| rmsnorm | dX, dW（若 norm 权重冻结则只 dX） |
| rope | dX（逆旋转） |
| full attention（QK^T/softmax/PV） | dQ/dK/dV + softmax backward |
| MLP（gate/up/silu/mul/down） | dX + dW（LoRA 只对 Linear 权重求 A/B） |
| residual / add / narrow / concat | 反向切片/累加 |
| lm_head / logits | dX（若 head 冻结） |
| cross-entropy（masked） | dLogits |
| LoRA A/B | dA/dB |

**GDN 层：**

- 0.8B/4B 是 hybrid 3:1（18 GDN + 6 full）。完整训练需要 GDN backward：
  - gating（softplus/sigmoid）；
  - causal conv1d 状态更新；
  - gated delta rule；
  - l2norm / norm-act。
- 这是首版最大的风险项。建议两种路径：
  - **P1a（推荐先做）**：只对**最后一个 full-attention block 及其 MLP** 挂 LoRA，
    梯度不穿过 GDN；验证数据/训练/评估/上线闭环。行为改变有限（主要影响输出层附近的
    风格/格式），不承诺知识注入效果；但工程可控，能先把“训练→评估→adapter”跑通。
  - **P1b（后续）**：实现 GDN backward，放开所有层 LoRA；这是让 LoRA 真正影响
    全模型行为的关键里程碑，必须单独立项。
- 如果坚持第一版就全层 LoRA，则必须把 GDN backward 作为独立里程碑，不能低估。

### 7.3 为什么首版用 BF16 base

- W4A16 的 activation-gradient 需要 `dX = dY @ W`，而 W 是 packed int4；要么写
  dequant-in-kernel 的反向 GEMM，要么保留一份 BF16 权重副本。
- 0.8B BF16 ≈ 1.6GB，4B BF16 ≈ 8GB；单卡 20G 可容纳 base + LoRA + activation
  （配合 checkpointing）。
- 27B AWQ 训练是另一个量级：10.5G/rank base + activation + gradient workspace，
  2×20G 非常紧；不应作为首版。
- 首版用 BF16 base 验证训练算法；量化 base 的反向放到 27B 预研阶段。

### 7.4 显存与 checkpoint

0.8B / seq 2048 / micro-batch 1：

- base BF16 ≈ 1.6GB；
- LoRA + optimizer ≈ 数百 MB；
- activation 全存会很大 ⇒ 必须 **gradient checkpointing**：每层只存边界，反向重算；
- logits `[T, vocab]` 也很大（vocab 248k × 2048 × 2B ≈ 1GB）⇒ SFT 只在 assistant token
  上取 logits，避免整段 materialize；或分块计算 cross-entropy。
- 4B 需要更小的 seq / 更激进的 checkpoint；先以 0.8B 为准。

### 7.5 训练 loop

```text
for step:
  batch = next(micro_batch)
  loss = forward_backward(batch) / grad_accum
  if step % accum == 0:
     clip_grad_norm(lora_params)
     adamw_step(lora_params)
     zero_grad()
  if step % eval_every == 0:
     eval(val_set)
  if step % ckpt_every == 0:
     save_checkpoint()
```

优化器：

- AdamW，β=(0.9,0.999)，weight_decay 可配；
- LoRA A/B 用 FP32 master 副本，前向用 BF16；
- grad clip 1.0；
- LR schedule: warmup + cosine/linear；
- 只保存 LoRA 参数，不保存 base。

### 7.6 TP / 多卡

- 0.8B/4B 单卡优先；不需要 TP 训练。
- 27B 若将来做，需要：
  - LoRA A/B 按 base Linear 规则切分；
  - 反向梯度 allreduce；
  - activation/optimizer 分片；
  - 量化 base 的反向 GEMM。
- 这些不在首版承诺内。

---

## 8. 评估与 adapter 注册

### 8.1 评估集

每个 dataset 自动留出：

- `val`：同分布，用于 early stop / 选 checkpoint；
- `test`：冻结，不参与训练；
- `regression`：固定 prompt 集，覆盖通用能力、格式、工具调用、安全拒答；
- `canary`：少量人工审阅样本。

### 8.2 指标

- SFT loss / perplexity（assistant tokens）；
- 任务指标：格式合规率、工具调用正确率、抽取/分类 exact match；
- 回归：base 模型在 regression 集上的通过率下降；
- 安全：拒绝率、敏感内容、prompt injection 抵抗；
- 人工抽检。

### 8.3 上线门

建议默认门：

```text
val loss 下降
test 指标不低于 baseline - 1%
regression 通过率下降 <= 2%
无 NaN / 无安全回归
人工审阅通过
```

任一不满足 = `rejected`，adapter 保留在 candidates，不激活。

### 8.4 adapter 注册表

```text
adapters/<adapter_id>/
  adapter_model.safetensors
  adapter_config.json
  owl_adapter.json       # base/dataset/job/eval 指纹
  report.json
  status.json            # candidate / active / rejected / archived
```

- `base_fingerprint` 不匹配 = 拒绝加载；
- `dataset_id` / `job_id` 可追溯；
- 激活/回滚写审计日志；
- 删除 adapter 前必须 deactivate。

---

## 9. 安全、隐私与数据治理

### 9.1 数据

- 默认只收集**显式写入**的样本；不自动把全部对话加入训练。
- 每条样本有 `consent.train`、`source`、`retain_days`；未同意 = 不写入训练集。
- 写入时 PII redaction（邮箱/电话/密钥/路径可配）；redaction 失败 = 拒写。
- 数据集快照不可变；删除样本只影响未来快照。
- 训练数据目录权限 0700；adapter 目录同。
- 支持按 sample/dataset 删除；删除后不可恢复。

### 9.2 训练安全

- 训练任务必须人工审批；agent 不能直接触发 GPU 训练。
- 训练停服，避免与推理竞争；训练进程获取 GPU 卡锁。
- 数据投毒：来源审计、重复/异常检测、人工抽检；不自动上线。
- 模型漂移：regression + canary；不通过不上线。
- 失败/超时/NaN：不产生可激活 adapter。

### 9.3 行为边界

- LoRA 用于风格/格式/工具习惯/领域措辞；文档事实优先 RAG。
- 不训练“绕过安全策略”的样本；安全拒答样本单独标记。
- 不把用户隐私写进 adapter；adapter 可能记忆训练数据，需在数据侧先脱敏。

---

## 10. 失败与降级

| 故障 | 处理 |
|---|---|
| adapter 文件缺失/损坏 | 加载失败；保持 base 或 previous adapter |
| base fingerprint 不匹配 | 拒载；提示重新训练 |
| rank/alpha/target 不合法 | 拒载（不猜） |
| adapter 图重捕失败 | 回滚 previous adapter；503 短暂 |
| 训练 OOM | 降低 seq/batch/rank 重试；写事件；不静默 |
| 训练 NaN | 停；保留最后健康 checkpoint；标记 diverged |
| 停服失败 | 不启动训练；GPU 卡锁冲突则拒 |
| 磁盘满 | 停止训练；不写半成品；保留最后 checkpoint |
| eval 不通过 | adapter 标记 rejected；不激活 |
| 训练进程被杀 | checkpoint 可续；job 标记 interrupted |

---

## 11. API 汇总（agent 视角）

### 11.1 数据

```text
POST   /v1/learning/samples
GET    /v1/learning/samples/{id}
DELETE /v1/learning/samples/{id}

POST   /v1/learning/datasets
GET    /v1/learning/datasets
GET    /v1/learning/datasets/{id}
DELETE /v1/learning/datasets/{id}
```

### 11.2 训练（全手工）

```text
POST /v1/learning/jobs              # 创建 pending_approval
GET  /v1/learning/jobs
GET  /v1/learning/jobs/{id}
POST /v1/learning/jobs/{id}/approve # 人工
POST /v1/learning/jobs/{id}/cancel
GET  /v1/learning/jobs/{id}/events
GET  /v1/learning/jobs/{id}/report
```

### 11.3 adapter

```text
GET  /v1/adapters
GET  /v1/adapters/{id}
POST /v1/adapters/load
POST /v1/adapters/{id}/activate
POST /v1/adapters/{id}/deactivate
POST /v1/adapters/{id}/rollback
DELETE /v1/adapters/{id}
```

chat：

```json
{"model":"qwen3.5-0.8b","adapter":"user-style-v3","messages":[...]}
```

---

## 12. 施工阶段

### L0：提案与边界（本文）

- 确认推理 LoRA 范围、训练目标、数据治理、API；
- 不写业务代码。

### L1：LoRA 推理最小闭环

- adapter 加载/校验/激活/停用；
- `Linear` 低秩支路；
- 单卡 0.8B，adapter on/off 输出对拍；
- chat `adapter` 字段；
- 图重捕策略；
- 无训练。

验收：adapter off 与 base 逐位一致；adapter on 输出稳定；切换不崩；base fingerprint 拒错。

### L2：数据面

- samples/datasets API；
- 不可变快照；
- PII/consent/去重；
- 数据集统计；
- agent 可写入/查询/删除。

验收：样本可追溯；删除语义正确；训练集不混入未同意数据。

### L3：内建训练核心（0.8B，BF16）

- `owl-train` 进程；
- TrainTensor/反向图；
- full attention + MLP + LoRA backward；
- cross-entropy + AdamW + checkpoint；
- 停服训练；
- 先做“最后 full-attention/MLP 段 LoRA，GDN detach”的 P1a。

验收：0.8B 小数据集 loss 下降；adapter 可加载；base 输出在 adapter off 时不变；
训练不 OOM；失败不产生半成品。

### L4：GDN backward / 全层 LoRA

- gating/conv/delta rule/norm-act 反向；
- 4B 验证；
- 全层 LoRA；
- eval/registry/rollback 完整化。

验收：4B 可训练；regression 门通过；adapter 可回滚。

### L5：27B 预研（不承诺）

- AWQ/W4A16 反向 GEMM；
- TP 梯度同步；
- activation/optimizer 分片；
- 2×20G 可行性报告；
- 若不可行，明确只支持小模型或外部训练。

---

## 13. 性能与容量目标（提案值）

| 项 | 目标 |
|---|---|
| 0.8B LoRA 推理额外延迟 | decode < 5%；prefill < 3% |
| adapter 加载 | < 1s（百 MB 级） |
| adapter 激活/图重捕 | < 5s（0.8B） |
| 0.8B SFT 训练 | 单卡 20G，seq 2048，micro-batch 1，可跑 |
| 4B SFT 训练 | 单卡 20G，seq 1024，checkpointing，可跑或明确不可行 |
| 27B 训练 | 本提案不承诺；需专项预研 |
| adapter 大小 | rank 16 全线性 ≈ 数十 MB 到 200MB |
| 投机解码 | v1 target-only；draft LoRA 是可选配对层（§4.8），启用前必须过分域净增益门 |

---

## 14. 可观测性

日志：

```text
[lora] load id=user-style-v3 rank=16 layers=196 base=... ok
[lora] activate id=... graph_recapture=1240ms
[train] job=... step=120 loss=1.83 lr=... grad_norm=...
[train] eval val_loss=... regression=... gate=pass
[train] save adapter=... bytes=...
```

指标：

- `lora.forward_ms` / `lora.activate_ms` / `lora.active`;
- `train.loss` / `train.lr` / `train.grad_norm` / `train.tokens_per_s`;
- `train.eval.*`；
- `train.ooms` / `train.nan`；
- `adapter.load_total{result}`。

审计：

- 每个 adapter 记录 base 指纹、dataset、job、eval、批准人；
- 激活/回滚记录时间与操作者；
- 训练数据删除记录。

---

## 15. 待拍板问题

| # | 问题 | 当前建议 |
|---|---|---|
| 1 | 首版是否接受“只训练最后 full-attention/MLP 段，GDN detach”？ | **是**，先跑通闭环；GDN backward 单列 L4 |
| 2 | 首版 base 是否用 BF16（不用 AWQ）？ | **是**；0.8B/4B BF16 先验证 |
| 3 | 训练器是否独立 `owl-train` 进程？ | **是**；停服训练，进程边界最干净 |
| 4 | 是否允许 HF/PEFT 作开发期金标？ | **允许**；生产不依赖 |
| 5 | adapter 是全局 active 还是 per-request？ | v1 **全局单 active**；per-request 多 adapter 后续 |
| 6 | 是否支持 adapter merge/export？ | v1 只导出 safetensors；merge 后续 |
| 7 | 训练数据是否默认收集全部对话？ | **否**；必须显式写入 + consent |
| 8 | 评估门具体阈值？ | 先按 §8.3，数据集实测后定档 |
| 9 | 27B 是否必须原生训练？ | 不承诺；先出 2×20G 可行性报告 |
| 10 | agent 是否有权批准训练？ | v1 **否**；人工审批，agent 只能准备/查询 |

---

## 16. 验收清单（提案级）

- [ ] LoRA adapter 可加载、激活、停用；off 时输出逐位不变。
- [ ] 0.8B 单卡上 adapter 推理延迟增量可测且可接受。
- [ ] agent 可写入样本、创建数据集、查询统计；未同意数据不进入训练集。
- [ ] 人工可创建/审批训练任务；系统不自动训练。
- [ ] 停服训练可完成 0.8B SFT；loss 下降；adapter 可被推理面加载。
- [ ] adapter 评估报告包含 val/test/regression/安全；不通过不能激活。
- [ ] 训练失败/超时/NaN 不产生可激活 adapter。
- [ ] adapter 可回滚到 previous 或 base。
- [ ] 数据删除、adapter 删除、审计日志可操作。
- [ ] 4B 可行性有实测结论；27B 只给预研报告，不虚报承诺。

---

## 17. 关键代码落点（施工时参考）

| 模块 | 现状 | 需要 |
|---|---|---|
| `crates/models/src/layers/linear.rs` | 三个 base 臂 | 可选 LoRA A/B 支路；dtype 对齐 |
| `crates/models/src/module.rs` | `ForwardCtx` 推理态 | `adapter` 只读上下文 |
| `crates/models/src/model.rs` | 纯前向 | 训练版 forward / loss mask |
| `crates/models/src/ops.rs` | 前向语义算子 | 训练反向算子/训练图 |
| `crates/engine/src/loader.rs` | 模型装载 | adapter 装载/校验 |
| `crates/engine/src/running.rs` | 推理执行态 | adapter 激活/图重捕 |
| `crates/engine/src/graph_fleet.rs` | 推理图族 | adapter 版本化图/重捕 |
| `apps/server/src/openai.rs` | chat 解析 | `adapter` 字段/校验 |
| `apps/server/src/main.rs` | 单 server | 学习 API 路由；训练不在此进程 |
| `crates/shared/src/config` | 单模型配置 | adapter/learning 配置段 |
| 新 `crates/train` / `apps/train` | 无 | TrainTensor/backward/optimizer/dataset |
| 新 `apps/server/src/learning.rs` | 无 | 样本/数据集/job/adapter API |
| `crates/models/src/layers/gdn.rs` | 前向 GDN | L4 的反向（关键路径） |

---

## 18. 结语

LoRA 推理是“接线问题”：base 冻结，低秩支路独立，现有 GEMM 出口足够。
LoRA 训练是“新引擎问题”：反向图、backward kernel、GDN 反向、optimizer、显存管理
都不是现有推理引擎的附属品。

因此建议把提案拆成两条腿：

1. **先让 LoRA 在推理面真正可用**（L1），这是无论训练做不做都值得的能力；
2. **再用 0.8B/4B 验证内建训练闭环**（L2–L4），把 GDN backward 和 27B 量化训练
   留到有数据、有评估、有显存结论之后。

这样既能尽快让 agent 用上 adapter，又不会在 27B 原生训练上过早投入。

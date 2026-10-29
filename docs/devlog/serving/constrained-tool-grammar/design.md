# Constrained tool grammar —— 原型设计

> 状态：设计稿（未实现）
> 目标读者：后续实现者 / review 者
> 先读：[`README.md`](README.md)

---

## 1. 背景

### 1.1 现象

2026-10-28 真机回放同一 Pi 请求，模型实际输出：

```text
<tool_call>
<function=bash>
<command>
ls /home/ubuntu/gui_agent/desk-pilot/apps/
</parameter>
</function>
</tool_call>
```

`<command>` 不是 `<parameter=command>`，旧解析器得到 `{}`，required 校验失败，服务层按约定返回：

```text
tool_call_missing_required
```

同一批回放还出现：

- `command="find ..."`（整行 key=value）
- `<function=computer_use>`（模型自造未提供的工具）
- 历史会话中的 `<function_calls><invoke ...>`（Anthropic 风格）

### 1.2 根因判断

- **主因是模型格式遵循度**，不是解析器“把 OpenAI 解析成 Anthropic”；
- 模板/长 prompt 是输入分布的一部分：上游模板与 Froggeric 模板都能复现漂移，Froggeric 样本略差但样本不足；
- 服务层旧行为是“事后拒绝”，不是“生成时约束”；
- vLLM 的 `--tool-call-parser qwen3_coder` 只是响应解析器，不是解码约束；它的 `_PARAM_RE` 同样只认 `<parameter=...>`。

### 1.3 现有缓解

`apps/server/src/toolcall.rs` 已支持：

- Anthropic `<function_calls>` wrapper；
- `<parameter name="...">` 与 `<parameter=...>` 混用；
- 裸 `<prop>`、`prop="..."` 的保守修复；
- 只有修复后能通过 schema 校验才接受。

真机同请求 8 发：7 发成功（3 发经修复），1 发因 `computer_use` 仍 502。

**本项目不是替代修复层，而是在修复层之前加一道生成约束。**

---

## 2. 目标与非目标

### 2.1 目标

1. 对“模型决定调用工具”的 token 区域，保证输出是合法 Qwen XML 工具块；
2. 不强制模型在应该文本回答时调用工具（除非显式 `required`）；
3. 普通 agent 客户端零配置即可受益；
4. 3080 TP2 + DFlash2 d7 的稳态 decode 损失控制在 **1–2%**；
5. grammar 编译可缓存，首次编译与 prefill 重叠；
6. grammar 失败必须响亮记录并回退，不静默假装已约束。

### 2.2 非目标

- 不解决“模型该调工具却不调”的语义问题（靠 prompt / GUI 硬规则 / required）；
- 不把参数值按完整 JSON Schema 约束（第一版只约束结构；类型/required 由现有校验）；
- 不实现 Anthropic `/v1/messages`（独立项目）；
- 不改变模型模板、不改 tokenizer、不改 prompt 的默认内容；
- 不做多租户动态 grammar 市场。

---

## 3. 关键裁决

### D1：`llguidance` 优先，xgrammar 备选

- workspace `Cargo.toml` 已声明 `llguidance = "1.7"`，纯 Rust，mask API 直接可用；
- xgrammar 有 Qwen3 coder 内置 structural tag，但 C++/FFI + 构建链更重；
- 第一版用 `llguidance` 的 Lark grammar；xgrammar 只作为性能/兼容对照。

### D2：服务端策略，不依赖客户端

普通 agent 请求实测：

```text
213 个带 tools 的请求
tool_choice：212 个缺省，1 个 auto
strict：0 个
```

因此策略由服务端决定：

```toml
[server]
tool_grammar = "auto"   # off | auto | required
```

| 请求 | 行为 |
|---|---|
| 无 tools | off |
| `tool_choice="none"` | off |
| `tool_choice="required"` | required grammar |
| `tool_choice` 缺省 / auto | auto grammar（默认） |
| 客户端 `strict` | 参数 schema 更严格（后续阶段） |

### D3：第一版只约束结构

Grammar 约束：

- 外层 `<tool_call>` / `</tool_call>`；
- `<function=NAME>` 的 NAME ∈ 请求 tools；
- `<parameter=KEY>` 的 KEY ∈ 该工具 schema properties；
- 多个 tool call、多个参数、自由文本参数值；
- 可选 ` thinking` 段由模板/现有解析器处理。

不约束：

- 参数值的 JSON 类型/枚举/数值范围；
- 参数顺序（第一版允许任意顺序）；
- 文本段内容。

这样 grammar 小、编译快、对质量影响小；类型/required 仍由 `ToolSchemas::validate` 负责。

### D4：`auto` 是默认，`required` 是业务开关

- `auto`：允许“纯文本”或“合法工具块”；不会强迫模型调工具；
- `required`：从第一个输出 token 起只允许工具块；只给 GUI 首轮/显式强制场景；
- 不建议全局 `required`：会破坏正常文本回答，可能引发工具循环。

### D5：mask 必须在 top-32 / argmax 之前

3080 生产档：

```text
DFlash2 d7 + 候选 top-32 + 图内 argmax/top-k
```

如果先采样再检查 grammar：

```text
候选集已选完 → 合法 token 可能不在 top-32
→ 只能拒绝重采样 → spec 收益崩掉
```

正确顺序：

```text
grammar state -> bitmask -> apply to logits -> top-32/argmax -> sample -> commit
```

---

## 4. 架构

```text
POST /v1/chat/completions
        │
        ▼
 ChatRequest.tools ──► GrammarPlan
                        │  hash(tools schema + format + thinking)
                        ▼
                  GrammarCache ──► Compiled Grammar
                        │
                        ▼
                 TurnGrammar (per turn)
                        │
        ┌───────────────┴────────────────┐
        │                                │
   prefill (no mask)              decode loop
                                         │
                              ┌──────────┴──────────┐
                              │                     │
                        plain decode            spec verify
                              │                     │
                       compute_mask            masks for rows
                              │                     │
                       apply to logits        apply before
                       before sample          top-32/argmax
                              │                     │
                       commit_token           commit/rollback
```

### 4.1 组件

| 组件 | 职责 | 位置（未来） |
|---|---|---|
| `GrammarPlan` | 从 `tools` 生成 grammar 文本/结构；决定 auto/required | `apps/server` |
| `GrammarCache` | schema hash → compiled grammar；LRU/上限 | `apps/server` / `crates/engine` |
| `TurnGrammar` | 单 turn 状态机；`compute_mask` / `commit_token` | `crates/engine` |
| `MaskApplier` | host 或 device 应用 bitmask | `crates/engine` / `crates/kernels` |
| `GrammarMetrics` | compile/mask/fallback 计数 | `crates/engine` / `server` |

### 4.2 请求到引擎

`TurnOpts` 增加：

```rust
pub struct TurnOpts {
    // ...
    pub grammar: Option<TurnGrammarHandle>,
}
```

`TurnGrammarHandle` 必须是显式可选：

- `None` = 不约束（普通聊天/无 tools）；
- `Some(auto)` = 允许文本或工具；
- `Some(required)` = 只允许工具。

禁止全局变量/线程变量；由 `openai.rs` 构造，随 `EngineReq::Chat` 传入。

---

## 5. Grammar 设计

### 5.1 Qwen XML 结构

目标形态：

```text
<tool_call>
<function=NAME>
<parameter=KEY>
VALUE
</parameter>
</function>
</tool_call>
```

Grammar 规则（Lark 概念）：

```lark
start: text? toolcall*
text: /(?s:.)*/            # 实际用非贪婪/分段规则，避免吞掉 trigger
toolcall: "<tool_call>\n<function=" TOOL ">\n" param* "</function>\n</tool_call>"
param: "<parameter=" KEY ">\n" VALUE "\n</parameter>\n"
```

实现注意：

- `TOOL` = 请求 tools 的 name 枚举；
- `KEY` = 当前 tool schema 的 properties 枚举；
- `VALUE` 第一版允许任意非 `</parameter>` 文本；
- 多个 tool call 之间允许结构空白；
- `auto` 的 `text` 不能吞掉 `<tool_call>` trigger（用 llguidance 的 lexer/trigger 机制，不用贪婪 `.*`）；
- `required` 的 `start` 直接是 `toolcall+`，不允许纯文本。

### 5.2 与模板的关系

- 模板继续负责“告诉模型 Qwen 格式”；
- grammar 负责“如果模型开始工具块，结构必须合法”；
- 两者不是二选一：模板是软约束，grammar 是硬约束；
- 不把 grammar 写进模板，避免模板与解码器两套规范。

### 5.3 Anthropic wrapper

- grammar **不生成** `<function_calls>`；
- 解析器仍保留 wrapper 兼容，作为 grammar 关闭/回退时的兜底；
- `required` 模式下 wrapper 会被 grammar 拒绝，模型只能走 Qwen 格式。

---

## 6. 采样接缝

### 6.1 普通 decode（host sampler）

现有 3080 生产档 `sampling` 模式每步已经：

```text
read_full_logits -> sampler::sample
```

插入点：

```text
read_full_logits
  -> grammar.compute_mask()
  -> apply_mask(logits, mask)
  -> sampler::sample
  -> grammar.commit_token(token)
```

成本：

- mask 生成 ~0.006ms（xgrammar 微基准，llguidance 同量级）
- host 应用 ~0.05–0.2ms
- 3080 150 t/s 下约 1–2%

### 6.2 候选 top-32 路径

当前 `use_cand`：

```text
device top-32 + rep penalty -> host 候选采样
```

grammar 必须在 **device top-32 之前**应用，否则合法 token 可能被截掉。

第一版可选：

- grammar 开启时关闭 `use_cand`，回退 full logits（简单，但 spec 变慢）；
- 或给 top-32 核增加 bitmask 输入（正确，但工作量大）。

### 6.3 投机 verify

`verify_round` 一次产生 `depth+1` 行 logits/候选：

- 每行 grammar 状态依赖前一行 token；
- 需要逐行 `compute_mask`（k+1 次），生成 `[rows, vocab/32]` bitmask；
- draft token 被接受则 commit；拒绝则 rollback；
- 设备端在 top-32/argmax 前应用 bitmask。

这是生产化的主要工作量。

### 6.4 贪心 3090

3090 档当前 `greedy`，设备 argmax 零 D2H。  
grammar 需要：

- 方案 A：设备 mask 核 + 图输入槽（保留速度）；
- 方案 B：host 回读全量 logits（简单，但 3090 有明显额外成本）；
- 方案 C：3090 档不默认开 grammar（agent 生产档在 3080）。

第一版可只覆盖 3080；3090 显式 opt-in。

---

## 7. 缓存与生命周期

### 7.1 Cache key

```text
hash(
  model_id,
  tool_call_format,
  enable_thinking,
  sorted(tools schema: name + parameters properties/required)
)
```

不把整个 prompt 纳入 key；system prompt 变化不应导致 grammar 重编译。

### 7.2 成本

实测（xgrammar，Qwen3.8 tokenizer）：

| tools | 首次 compile | 缓存后 |
|---|---:|---:|
| 4 GUI tools | ~60ms | ~0.5ms |
| 32 Pi tools | ~1.17s | ~0.5ms |

### 7.3 重叠

- grammar 编译可在请求到达后异步启动；
- 与 prefill 并行（19k prompt prefill ~8s，1.17s 可隐藏）；
- 短 prompt 首轮会感知 TTFT +1.17s，必须缓存。

### 7.4 生命周期

- `TurnGrammar` 随 turn 创建/销毁；
- 无全局单例；缓存由显式 `GrammarCache` 持有；
- turn 结束/abort 时释放，不挂全局变量。

---

## 8. 错误与观测

### 8.1 失败策略

| 失败 | 行为 |
|---|---|
| grammar 编译失败 | WARN + fallback 到现有 repair 层；`strict_grammar=true` 时 400 |
| mask 生成失败 | 该 turn 按协议错误收口（502/SSE error），不静默 |
| commit 失败 | abort turn；记录 token/state |
| cache 超限 | LRU 驱逐；计数 |
| grammar 与 parser 不一致 | 拒绝该 turn，记录 `grammar_parser_mismatch` |

### 8.2 指标

```text
owl_tool_grammar_compile_ms
owl_tool_grammar_cache_hit / miss
owl_tool_grammar_mask_ms
owl_tool_grammar_fallback_total
owl_tool_grammar_malformed_total
owl_tool_grammar_spec_rollback_total
```

`/debug/agent` 增加：

```json
{
  "grammar": {
    "mode": "auto",
    "cache": "hit",
    "compile_ms": 0.5,
    "mask_ms_p50": 0.09,
    "commits": 123,
    "fallbacks": 0
  }
}
```

---

## 9. 性能预算

目标（3080 TP2，140–157 t/s）：

```text
mask + apply + commit：≤ 0.15ms/token
decode 损失：≤ 2%
首次 compile：≤ 1.5s（32 tools）
缓存后：≤ 1ms
```

对照（xgrammar 微基准）：

```text
CPU mask        0.006ms
accept/commit   0.001ms
H2D + GPU apply 0.07ms
合计            0.09ms（p95 0.13ms）
```

llguidance 的 `compute_mask` 可能比 xgrammar 慢（其文档提示可能 >1ms，建议后台线程）；  
必须在本机 32-tool grammar 上实测，若 p95 > 0.5ms，需要预计算/缓存/分段。

---

## 10. 测试门

### G0：grammar 正确性

- 32 tools + 多参数 + 多 tool call + Unicode；
- 必须接受合法 Qwen XML；
- 必须拒绝 `<command>`、Anthropic wrapper、未知工具、未知参数。

### G1：auto/required 语义

- auto：纯文本 PASS；合法工具块 PASS；非法工具块 FAIL；
- required：纯文本 FAIL；合法工具块 PASS。

### G2：与现有解析器一致

- grammar 接受的输出，`ChannelSplitter` 必须解析为同一 tool call；
- parser 修复层在 grammar 关闭时仍工作；
- `tool_call_format=json` 暂不启用 grammar（或单独 grammar）。

### G3：spec 正确性

- grammar 开启/关闭，合法输出逐 token 等价（greedy 口径）；
- draft token 非法时 rollback 正确；
- spec AL 下降不超过预设阈值。

### G4：性能

- decode t/s、TTFT、mask ms、compile ms；
- 3080 TP2 与 3090 单卡分别记录；
- 稳态 decode 损失 ≤ 2%（32 tools）。

### G5：故障

- grammar 编译失败回退；
- cache 驱逐；
- abort/断流释放；
- 无全局状态泄漏。

---

## 11. 风险与开放问题

1. **llguidance 的 `compute_mask` 在 248k 词表上的实际延迟**未知，需要 spike；
2. **Qwen XML 的自由文本 `VALUE` 与 trigger 的 lexer 冲突**需要验证，避免 text 吞掉 `<tool_call>`；
3. **auto grammar 是否真的能阻止 Anthropic wrapper**：如果 wrapper 被当普通文本，grammar 不会拒绝；`required` 才能强制；
4. **spec + grammar 的图输入槽**是最大工程风险；
5. **`tool_choice=required` 的语义**不能由服务端默认开启，否则普通对话被强制工具；
6. **参数 JSON Schema** 第一版不做，后续是否做要看质量/性能；
7. **前端减工具**可能单独把错误率降到可接受，grammar 的优先级需用数据决定。

---

## 12. 与前端减工具的关系

前端减工具是**成本最低的第一刀**：

- 32 tools → 8–12；
- GUI 只暴露 GUI 工具；
- 去掉重复 `<tools>` 文本列表；
- 末尾加强格式规则。

预期：

- prompt 更短，格式指令权重更高；
- 模型选择空间更小，未知工具/`computer_use` 概率下降；
- grammar 的 `TOOL` 枚举更小，编译更快。

两者关系：

```text
前端减工具：降低错误率（软）
repair 层：接住近失（中）
grammar：结构硬保证（硬）
```

不要等 grammar 才做前端减工具；也不要以为前端减工具能 100% 替代 grammar。

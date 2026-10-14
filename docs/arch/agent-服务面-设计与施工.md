# agent 服务面专项:pi agent 对接(设计与施工)

> 版本:**v0.6**(2026-10-14 立项当日;**M-A 已达成**)—— 变更记录见 §十。
> 状态:**M-A ✅ 达成**(pi ↔ owl 生产档跑通工具闭环,§7.7);M-B/M-C 需求已列,待排期。
> 定位:**owl 第一次面向"真实 agent 客户端"的服务面收口** —— 客户 =
> 本机 `pi` coding agent(provider 类型 `openai-completions`)。
> **待办台账** = `roadmap.local/agent-服务面-剩余工作项-2026-10-14.md`(本文只记设计与判据,
> 剩余工作项、量级、执行序住那里)。
> 承接:`docs/arch/server-多会话并发设计.md`(S0 会话/并发面)、
> `docs/arch/requirements.md` v1.0(需求基线,本专项新增 `REQ-AGT-*` 族)、
> `roadmap.local/phase4-agent-session.md`(引擎侧会话面,已落地)。
> 纪律(charter):本专项**只改 `apps/server/`**(协议面 + 模板面),
> 不动 kernels / 不动解释器热路径 / 不动图捕获面;渲染走**快照自带
> `chat_template.jinja`**,不在 Rust 里长第二份模板。

---

## §0 摘要(给三个月后的自己)

1. **pi 侧零成本**:`~/.pi/agent/models.json` 加一个 provider 条目(≈20 行)即可接入;
   **全部工作量在 owl 服务端**。参考现成条目:`3090_dual`/`3080_dual`(vLLM/sglang,已跑通)。
2. **P0 四个硬缺口**(不做 = agent 完全不可用,连不上也跑不动):
   ① 模板真渲染(**`tools` 块**);② 工具调用解析(`<tool_call>` → OpenAI `tool_calls`);
   ③ 历史回填(`tool_calls` / `role:"tool"` / `tool_call_id`,现被**静默丢弃**);
   ④ thinking 分离(`reasoning_content`)。
3. **两条生死项**(不与协议同批,但决定"能不能用"):
   - **前缀复用**:现状多轮 **0% 命中**(e2e 17/17 `@0`),而 pi 每轮重发全历史;
   - **会话身份**:pi **不发我们的 `session` 键** ⇒ 每请求新建会话 ⇒ 守卫必失配。
4. **M-A 交付线**:`pi -p "…"` 跑通 **read → write → bash** 工具闭环(真机冒烟)。**✅ 2026-10-14 达成**(见 §7.7)。
   **M-B/M-C 也已在同日落地**(AGT-06/07/09/10/11/12 + D1/G4,见 §7.9);**仅剩
   两条待攻**:A7(工具调用遵循度)与 A8(spec 臂 token 级非稳定,§7.8)。
5. **明确不在本专项**:prefill 吞吐(那是 P1‴/零回执,`优化候选总表-2026-10-14.md`)、
   decode 批桶(B5.1)、DFlash2 草稿质量。本专项**不做性能承诺**。

---

## §一 病灶与现状对账(证据链)

### 1.1 pi 侧权威口径(唯一判据来源)

| 来源 | 用途 |
|---|---|
| `@earendil-works/pi-ai/dist/api/openai-completions.js`(1344 行) | **wire 契约唯一权威**:请求字段构造 + 流式解析 + `finish_reason` 映射 + usage 语义 |
| `pi-coding-agent/docs/models.md` | `models.json` 配置面、`compat` 标志语义、`samplingParams` |
| `pi-coding-agent/docs/custom-provider.md` | 何时该写 provider extension(本专项**不需要**)|
| `pi-coding-agent/docs/message-types.md` | `ThinkingContent` / `ToolCall` / `Usage(cacheRead)` 语义 |

关键代码证据(行号为 pi-ai v1.0.4 dist):

| 事实 | 出处 |
|---|---|
| 流式**恒发** `stream_options={include_usage:true}` | L583(`supportsUsageInStreaming !== false`) |
| `max_tokens` 字段名由 `compat.maxTokensField` 决定 | L589 |
| `temperature` 由调用方给定(模型 `samplingParams` / thinking 档覆盖) | L597 |
| `tools` 为空但历史含工具 → **发 `tools: []`** | L607-612 |
| thinking 增量字段取 `["reasoning_content","reasoning","reasoning_text"]` 首个非空 | L138 / L400 |
| 工具块**按 `index` 归并**(`id` 仅作兜底) | L289-315 `ensureToolCallBlock` |
| `finish_reason`: `tool_calls`/`function_call` → `toolUse`;未知值 → **error** | L1195-1211 `mapStopReason` |
| 缺 `finish_reason` → 抛 `"Stream ended without finish_reason"` | L485(`supportsFinishReason` 默认 true) |
| usage: `input = prompt_tokens − cached_tokens − cache_write_tokens` | L1165-1174 |
| tool schema 会被塞 `strict`(除非 `supportsStrictMode=false`) | L1148-1160 `convertTools` |
| assistant 消息可带 `reasoning_content`(回填)与 `content: null` | L1004-1050 |
| tool 消息形状 `{role:"tool",content,tool_call_id[,name]}` | L1085-1090 |

### 1.2 逐项对账(pi 要什么 × owl 有什么)

| # | pi 行为 / 要求 | owl 现状(`apps/server/src/openai.rs` + `http.rs`) | 判决 |
|---|---|---|---|
| 1 | `GET /v1/models`、`GET /health` | 有(`main.rs::serve`) | ✅ |
| 2 | `POST /v1/chat/completions` + SSE + `[DONE]` | 有(手写 HTTP,`Connection: close` 定界) | ✅ |
| 3 | `stream_options.include_usage` 末帧真账 | 有(2026-10-13 落地,`choices:[]` + usage) | ✅ |
| 4 | `choices[0].finish_reason` ∈ {stop,length,tool_calls} | 有 `stop`/`length`;**无 `tool_calls`** | ⚠️ 半 |
| 5 | `max_tokens` 字段名 | 收;需在 models.json 设 `maxTokensField:"max_tokens"` | ✅ |
| 6 | **`tools` 参数(function schema)** | **`ChatRequest` 无此字段 → serde 静默丢弃** | ❌ P0 |
| 7 | **assistant `tool_calls` + `content:null`** | `ChatMessage{role,content}` 无该字段 → `text()` 得 `""` → **整条消息被 `continue` 跳过** | ❌ P0(致命) |
| 8 | **`role:"tool"` + `tool_call_id`** | 同上,被丢弃且渲染成 `user` 分支 | ❌ P0(致命) |
| 9 | **`tool_calls` 流式 delta(`index`/`id`/`function.arguments` 片段)** | 无 | ❌ P0 |
| 10 | `reasoning_content` 增量(思考面板) | 无(think 文本混在 `content` 里) | ❌ P0/质量 |
| 11 | `temperature`(及 `top_p`/`top_k` 等 `samplingParams`) | 引擎有 `SamplerCfg{temp,topk,topp,rep_penalty}`(`engine/src/sampler.rs`),但**请求面不收**,只能全局配置 | ❌ P1 |
| 12 | `chat_template_kwargs.enable_thinking`(pi `/thinking` 档) | 无 | P1 |
| 13 | `usage.prompt_tokens_details.cached_tokens` | 无 | P2(便宜,且是自查命中率的窗口) |
| 14 | `tool_choice` / `strict` 字段 | 静默忽略(可接受,但见 §3.1 未知字段纪律) | ⚠️ |
| 15 | 取消(用户 Esc)= 断流 | actor 明确挂账 **"active turn 无 abort 原语"** → 引擎继续烧 GPU 跑完 | ❌ P2 |
| 16 | 并发子代理(多流并存) | `gdn_slots=1`/`max_concurrency=1`;decode 批桶(B5.1)**未开工** → 串行排队 | ⚠️ P2 |

### 1.3 两条生死项(与协议表分开的账)

**(a) 前缀复用零命中**(已在 `server-多会话并发设计.md` §六 立案):

- e2e 17 个请求 **全部全量 prefill(`@0`)**;服务端统计行 `[前缀复用] 全量 N / 增量 M`。
- 三条路分开的账:
  - **同会话守卫**:`render_prompt` 末尾注入 `<|im_start|>assistant\n<think>\n` 预填,
    而该 `<think>` **不在 API 返回的 `content` 里** ⇒ 客户端原样回填后渲染从
    assistant 轮起点起逐 id 不同 ⇒ `AgentSession::guard()` 必失配 ⇒ 全量重算。
  - **跨会话块级缓存**:**只是配置关着** —— `pool.prefix_cache` 默认 true,
    但 `spec-dflash` flavor 锁 `false` ⇒ `enable_prefix_cache()` 不被调
    (`flavor spec-dflash`: `gdn_slots=1 / snap_max=1 / prefix_cache=false`)。
  - **机制本体已验**:`crates/engine/src/tests.rs::gpu_prefix_cache_hit`
    (300 tok 共享前缀跨会话:块链物理共享 + **GDN 快照恢复** + B 的 prefill chunk ≪ A)。

**(b) 会话身份**:pi **不发 `session` 字段** ⇒ `SessionTable::get_or_create(None)` 走
`next_id += 1` **每请求新建会话** ⇒ ① 守卫从零开始(必失配);② 会话表 + GDN 格
被每轮请求轮转(格满后 LRU 逐出**上一个活跃会话**)。

> **结论**:对 pi 而言,**"同会话守卫"这条路线结构性不可达**(除非我们改渲染对称性);
> **唯一可行且与客户端无关的复用手段 = 跨会话块级前缀缓存**(纯 prompt 内容哈希,不需要 session 键)。

### 1.4 一句话判决

> **距离"pi 连上 owl 跑一次真 agent 任务" = P0 四项(1.5~2.5 天);
> 距离"日常可用(多轮长会话不慢崩)" = 再加 P1 前缀复用 + 采样 + 长 ctx 定档(≈1 周)。
> 协议面是小工程,真正的长尾是 prefill 吞吐(与本专项解耦,见 §2.4)。**

---

## §二 需求(REQ-AGT-*)

> 编号族新增于本专项;评审后回写 `docs/arch/requirements.md` §裁决记录。
> 等级:P0(不满足 = 目标不可达)/ P1(不满足 = 可用性显著受损)/ P2(工程完备性)。

### 2.1 P0 功能门

| 编号 | 需求 | 验收口径(判据) |
|---|---|---|
| **REQ-AGT-01** | **模板真渲染**:渲染器读模型快照的 `chat_template.jinja`(**Rust crate `minijinja`**,非 PyPI 同名包 — 见 §7.1 开头),按 §3.2 契约传参;`tools` 渲染进 `# Tools` 块 | **G0 token 等价门**:同输入下,owl 渲染 → tokenize 的 id 序列与参考实现逐 id 相等 |
| **REQ-AGT-02** | **工具调用解析**:从生成文本中抽出 `<tool_call>` 块(xml + json 两种格式),转 OpenAI `tool_calls` 流式 delta,`finish_reason="tool_calls"` | **G1 单测**:两种格式 × 多块 × 跨 chunk 边界 × 多行参数值 × Unicode;`arguments` 恒为合法 JSON 字符串 |
| **REQ-AGT-03** | **历史回填**:解析请求里的 `tool_calls`(assistant,可 `content:null`)与 `role:"tool"`+`tool_call_id`,按模板语义渲染 | **G2 e2e L1**:三轮工具对话端到端语义正确(工具结果被模型引用) |
| **REQ-AGT-04** | **thinking 分离**:`...response` 前的内容走 `reasoning_content` 通道,之后走 `content`;回填时按 `preserve_thinking` 语义还原 | **G0/G1**:回填渲染与生成 token 逐 id 相等(见 §4.3)|
| **REQ-AGT-05** | **未知/未支持字段 fail-fast**:协议关键字段(`tools`/`tool_choice`/`n`/`logprobs`)若未实现,启动或请求期给**结构化 400**,禁静默丢弃 | **G1**:请求带 `n:2` → 400 + 明确 message;`tools:[]` 空数组必须**接受** |

### 2.2 P1 性能门

| 编号 | 需求 | 验收口径 |
|---|---|---|
| **REQ-AGT-06** | **每请求采样参数**:`temperature/top_p/top_k/stop`(及 `seed` 可选)经 `EngineReq` 覆盖引擎 `SamplerCfg`;缺省 = 全局配置 | ✅ **已落地(2026-10-14)**:温度=0 与温度=1 同 prompt 分布可区分;`stop` 命中提前结束。**⚠️ 覆盖面记账**:仅**裸 decode 臂**生效 —— spec 臂的 token 选择不走采样器(见 §7.8 A8) |
| **REQ-AGT-07** | **`enable_thinking`**(pi `chat_template_kwargs`):off ⇒ 预填空 think 块;on ⇒ 自然思考 | 两档在同一 prompt 上输出形态不同(think 段有无) |
| **REQ-AGT-08** | **前缀复用**:跨会话块级缓存开启 + 复用时输出与全量重算**逐字一致**;同会话守卫经 §4.3 渲染对称性顺带命中 | ✅ **PASS(2026-10-14,G4)**:共享前缀跨会话命中 **1536/1732** + 多轮同会话命中 **165-225/187-238**;token 级 sha 等价(spec-off 严格口径)/ content 级(spec-on,因 A8) |
| **REQ-AGT-09** | **会话身份**:无 `session` 键时退化到**隐式默认会话**(守卫失配自动回退全量 = 天然安全) | 多轮同客户端:第 2 轮起 `@N>0`;交错两客户端:正确性不变(退化为全量) |

### 2.3 P2 工程门

| 编号 | 需求 | 说明 |
|---|---|---|
| **REQ-AGT-10** | **取消(abort)**:客户端断流 ⇒ 引擎侧尽早停放当前 turn | ✅ **已落地(2026-10-14)**:服务面写失败 ⇒ `EngineReq::Abort`;actor 侧投递失败 ⇒ 弃置(≤1 采样步);实测 curl 断开后 `弃置 turn N` |
| **REQ-AGT-11** | HTTP 硬化:keep-alive 或至少 undici 兼容性实测;chunked 请求体 | ✅ **chunked 已落地**(单测 4 + curl 实测);keep-alive 挂 hyper 迁移(permit 语义理由见 `http.rs` 头注) |
| **REQ-AGT-12** | `prompt_tokens_details.cached_tokens` 透出前缀命中 | ✅ **已落地**:含**块级缓存命中**真值(`SinkEvent::Session` 校正帧);G4 探针数字与引擎日志逐项吻合 |
| **REQ-AGT-13** | 长 ctx 定档:16384 → 32k/64k,补 std 长点回归 | ✅ **已评估定档**:维持 **16384**;32k/64k 已实测可用(见 §7.9 G5 数字),因 A8 + prefill −17% 不转正 |
| **REQ-AGT-14** | 并发:子代理并行(`max_concurrency`/`gdn_slots`) | ✅ **已实测**:4 并发墙钟 **0.94×**(单槽串行,无并行);真加速依赖 B5.1 批桶(非本专项) |

### 2.4 明确不做(舍弃即设计)

- **不做 provider extension**:`models.json` 的 `openai-completions` 足够(`custom-provider.md` 明确"能用 models.json 就别写 extension")。
- **不做视觉/图像输入**:v1 模型条目声明 `input:["text"]`;模板的 `<|vision_start|>` 分支保留但不启用。
- **不做 prefill/decode 吞吐优化**:那是 `优化候选总表-2026-10-14.md` 的账(P1‴ 零回执 / D1 草稿核 / D3 XQA ctx 税)。
- **不做多会话真并发加速**(B5 批桶)。
- **不做 grammar/结构化输出**(pi 的 `codemode` 系工具输入)。

---

## §三 契约设计(对外接口)

### 3.1 请求面:实际会收到的字段(pi 侧构造)

```json
{
  "model": "Qwen3.8-27B-Coder390",
  "messages": [ ... §3.1.1 ... ],
  "stream": true,
  "stream_options": { "include_usage": true },
  "max_tokens": 32000,
  "temperature": 1.0,
  "tools": [
    { "type": "function",
      "function": { "name": "read", "description": "...",
                    "parameters": { "type": "object", "properties": {...} } } }
  ],
  "tool_choice": "auto",
  "chat_template_kwargs": { "enable_thinking": true, "preserve_thinking": true }
}
```

- `tools: []`(空数组)**必须接受**(pi 在 `hasToolHistory && 无新工具` 时会发,见 L607)。
- `chat_template_kwargs` 仅在 models.json 声明 `thinkingFormat:"qwen-chat-template"` 时出现。
- `strict` 会出现在 `function` 内(除非 `supportsStrictMode:false`)——**须容忍**。

#### 3.1.1 messages 形状(必须全部支持)

| role | 形状 | owl 现状 |
|---|---|---|
| `system` | `content: string \| [{type:"text",text}]` | ✅(仅 string 分支) |
| `user` | 同上 | ✅ |
| `assistant` | `content: string \| null`,`tool_calls?: [{id,type:"function",function:{name,arguments:"<json str>"}}]`,`reasoning_content?: string` | ❌ |
| `tool` | `content: string`,`tool_call_id: string`,`name?: string` | ❌ |

> ⚠️ **未知字段纪律**:`ChatMessage` 必须 `#[serde(deny_unknown_fields)]` **改为显式白名单 + 结构化 400**
> (家规:不许静默隐瞒)。当前"静默丢弃"正是 7/8 号缺口的根因。

**⚠️ 归一(owl 侧实现,2026-10-14 实测发现)**:`assistant.tool_calls[].function.arguments`
在 OpenAI 协议里是 **JSON 字符串**,而**模板期望对象**(Transformers 维护者约定)。vLLM 在
`vllm/entrypoints/chat_utils.py::_postprocess_messages` 明确归一(注释原文:
"tool call arguments … need to be dicts not JSON str")。实证:本机 **0.8B 原版模板遇
字符串参数直接抛 `Can only get item pairs from a mapping`(HF 里同样失败)** ⇒ 不归一是
结构性不可用。owl 同款实现在 `openai::ChatRequest::normalize_tool_calls`:
字符串 → 解析为对象、空串 → `{}`、**空 `tool_calls` 数组剔除**(回普通 assistant 路径)、
非 `function` 类型 → 400。

### 3.2 模板渲染契约(jinja kwargs 表)

**模板来源**:`<model.dir>/chat_template.jinja`(**快照自带**,非内置)。本机实况:
`cyankiwi/Qwen3.8-27B-AWQ-INT4` = **461 行魔改版**(`template_version = "qwen3.8-froggeric-v22.5"`;
同目录 `chat_template.jinja.bak` 169 行 / `.pre-froggeric` 175 行 = 上游原版);
`Qwen/Qwen3.5-0.8B` = 153 行原版。⇒ **模板是每模型资产,必须模板驱动**。

渲染器传参(**模板自己 `{%- set X if X is defined else … %}` 的完整变量面**):

| 变量 | 默认 | 我们传 | 说明 |
|---|---|---|---|
| `messages` | — | 请求 messages(原样,含 tools 轮) | 必需 |
| `tools` | 未定义 | 请求 `tools`(原样) | 有值 ⇒ 进 `# Tools` 块 |
| `add_generation_prompt` | — | `true` | 末尾 assistant 起手 |
| `enable_thinking` | `true` | `chat_template_kwargs.enable_thinking`(缺省 true) | REQ-AGT-07 |
| `preserve_thinking` | `true` | `true` | 历史 assistant 的 think 段保留 ⇒ **§4.3 对称性的基础** |
| `reasoning_effort` | 无 ⇒ `medium` | **请求 `reasoning_effort`(AGT-07)或 `[agent] reasoning_effort`** | 档位指令(**直通模板** —— 模板自持值域 none/low/medium/high/xhigh…,不自造映射) |
| `tool_call_format` | `'xml'` | **`[agent] tool_call_format`(缺省 xml)** | ⚠️ 解析器必须与之一致:xml 模式生成 `<function=…><parameter=…>` |
| `auto_disable_thinking_with_tools` | `false` | **`[agent]` 同名键(缺省 false)** | 有工具就关思考(D2 待 A/B) |
| `max_tool_arg_chars` / `max_tool_response_chars` | `0` | `0`(不截断) | 截断是模板侧启发式 |
| `add_vision_id` | `false` | 不传 | v1 无视觉 |

> ⚠️ **契约要点(T0 实测踩到)**:`chat_template_kwargs` 内的键**必须展平到渲染上下文顶层**
> —— HF 把 kwargs 直接当模板变量;不展平则 `enable_thinking=false` **静默失效**(走默认 `true`,
> 实测差 434 B)。单测必锁此项。

**外部依赖(minijinja 兼容层,§7.1 = 已实测定案)**:
`namespace(...)`(builtins 自带,✅)、`{% macro %}`(✅)、`| tojson`(**必须覆盖为 HF 语义**)、
`raise_exception(...)`(**自注入**)、`.split/.startswith/.lstrip/.rstrip/.endswith`
(**`minijinja-contrib::pycompat` 全覆盖**)、字典 `.items()`(✅)、`ns.attr = v`(✅)、
`~` 拼接 / `is defined/is mapping/is iterable/is string`(✅)。

### 3.3 工具调用 wire 形状(响应面)

生成文本形态(模板指令产出,**xml 为默认格式**):

```
<think>
…思考…
</think>

<tool_call>
<function=bash>
<parameter=command>
ls -la
</parameter>
</function>
</tool_call>
```

或 json 格式(若我们传 `tool_call_format="json"`):
`<tool_call>\n{"name":"bash","arguments":{"command":"ls -la"}}\n</tool_call>`

**响应面 delta 时序**(`index` 必须每帧带,`id` 首帧带即可 —— pi 按 `index` 归并):

```text
data: {"choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}],…}
data: {"choices":[{"index":0,"delta":{"content":"我来看看目录。"},"finish_reason":null}],…}
data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_ab12","type":"function",
        "function":{"name":"bash","arguments":""}}]},"finish_reason":null}],…}
data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"command\":"}}]},"finish_reason":null}],…}
data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"ls -la\"}"}}]},"finish_reason":null}],…}
data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],…}
data: {"choices":[],"usage":{…}}          ← include_usage 时
data: [DONE]
```

**硬约束**:
1. `arguments` 必须是**合法 JSON 字符串**(我们只在 `</tool_call>` 完整闭合后才发射该块 ⇒ 天然满足;
   pi 侧 `parseStreamingJson(partialArgs)` 对未闭合前缀有容错,但最终必须可解析)。
2. 多工具 = **多个独立闭合块**(模板指令明确禁止嵌套)⇒ 依次 `index:0,1,2…`。
3. `finish_reason` **必须是** `stop|length|tool_calls` 三者之一(mapStopReason 里其余值 → pi 直接报错)。
   有工具调用 ⇒ `tool_calls`;**不能**只发 `stop`。
4. 生成中途失败/截断 ⇒ 该 `<tool_call>` 块**不发射**(宁可无工具,不可给非法 arguments)。
5. 工具调用**不经过** `content` 通道(绝不能把 `<tool_call>` 原文吐给 `content`)。

### 3.4 thinking 通道

- 出:`delta.reasoning_content`(字段名固定,pi 用**首个非空**的 reasoning 字段)。
- 入:pi 回填 `assistant.reasoning_content`(因为我们的 thinkingSignature = `reasoning_content`)。
- **回填对称性(§4.3)**:渲染历史 assistant 时必须还原 `<think>{reasoning}\n</think>\n\n{content}`,
  与模板 `_preserve_thinking=true` 分支逐字节一致。

### 3.5 usage 与缓存计数

```json
{"usage":{"prompt_tokens":N,"completion_tokens":M,"total_tokens":N+M,
          "prompt_tokens_details":{"cached_tokens":H},
          "completion_tokens_details":{"reasoning_tokens":T}}}
```
- pi:`cacheRead = H`,`input = N − H` ⇒ **命中可见**(REQ-AGT-12)。
- **H 的真值来源(2026-10-14 落地)**:引擎 `ActiveTurn.cached_start` = 本 turn 起始
  fed(会话守卫命中 **+ 块级前缀缓存命中**),经 `SinkEvent::Session` 校正帧在**该 turn
  首个事件之前**发出(块级命中在 submit 时尚未决定 ⇒ Accepted 只能报估值;短 turn 在
  单次 pump 内跑完,校正帧必须抢在事件前发 —— 否则 sink 已摘除而丢帧,实测踩过)。
- `reasoning_tokens` 仍未透出(可选字段,pi 会把它计入 output 显示);实现需按通道
  计 token 数(现只有文本聚合),归 M-C 挂账。

### 3.6 错误面

| 场景 | 响应 |
|---|---|
| 请求体非法/缺 messages/未支持字段 | `400` + `{"error":{"message":…,"type":"invalid_request_error"}}`(生命周期正确,pi 会用于 compaction/重试判断) |
| 上下文超限(超 `max_seq`) | **须归一化为 `context_length_exceeded` 语义**(pi 据此触发 compaction) |
| 引擎未就绪/断流 | `503`(现状) |
| 生成中错误(已开始 SSE) | `error` 帧 + `[DONE]`(现状 `TurnEvent::Failed` 分支) |

> 注意 pi 会识别**上下文溢出**类错误并自动 compact+retry;**不要把限流/瞬时错误伪装成溢出**。

### 3.7 pi 侧目标配置(`~/.pi/agent/models.json`)

```json
{
  "providers": {
    "owl": {
      "name": "owl engine (3090 Ti, 单卡)",
      "baseUrl": "http://127.0.0.1:8137/v1",
      "api": "openai-completions",
      "apiKey": "local-no-key",
      "compat": { "supportsDeveloperRole": false, "supportsStrictMode": false },
      "models": [
        {
          "id": "Qwen3.8-27B-Coder390",
          "name": "owl · Qwen3.8-27B AWQ INT4(DFlash2)",
          "reasoning": true,
          "input": ["text"],
          "contextWindow": 16384,
          "maxTokens": 8192,
          "cost": {"input":0,"output":0,"cacheRead":0,"cacheWrite":0},
          "compat": {
            "maxTokensField": "max_tokens",
            "thinkingFormat": "qwen-chat-template",
            "supportsReasoningEffort": false
          }
        }
      ]
    }
  }
}
```

- `supportsDeveloperRole:false` ⇒ 用 `system`(与模板一致)。
- `contextWindow` **必须与 owl 实际池上限一致**(M-B 定档 16k→64k 后同步改),否则 pi 的
  compaction 阈值错位。
- `models.json` 的 provider 名不要与已有 `3090_dual`/`3080_dual` 冲突。

---

## §四 架构设计

### 4.1 模块划分与文件面

```text
apps/server/src/
├─ main.rs        路由:不变(仅新增错误分支/可选 /debug 面)
├─ http.rs        不变(M-B 才动 keep-alive/chunked)
├─ config.rs      新增 [agent] module 开关(模板路径缺省、tool 解析开关、隐式会话)
├─ openai.rs      §协议面重排:请求解析(全字段)→ 渲染调用 → 事件流消费(三通道)
├─ chat_tpl.rs    ★新:jinja 渲染器(minijinja 环境 + HF 兼容 filter/global 注入)
└─ toolcall.rs    ★新:增量工具调用解析器(状态机,吃增量文本吐三通道事件)
```

**单一权威**:
- 模板 = **快照文件**(不复制进代码);
- 渲染 = `chat_tpl.rs` **唯一入口**(禁第二处拼 prompt);
- 解析 = `toolcall.rs` **唯一入口**;
- 通道分流(think/tool/content)= `toolcall.rs` + `openai.rs` 事件循环**单点**。

### 4.2 一次工具轮的端到端链路(calltree)

```text
pi ──POST /v1/chat/completions{tools,messages[…tool_calls,tool…]}
  apps/server/src/main.rs::serve()                route
    apps/server/src/openai.rs::chat()
      openai::ChatRequest::from_json()            全字段解析(deny 未知关键字段)
      chat_tpl::render(model_dir, messages, tools, add_generation_prompt, enable_thinking)
        └─ minijinja render → prompt 文本
      actor::EngineReq::Chat{prompt, max_new, sampler:Option<SamplerCfg>, session:Option<u64>}
        openai.rs 事件循环:
          TurnEvent::Token{delta} ─▶ toolcall::Feed::push(delta)
              ├─ Think 通道  → sse delta{reasoning_content}
              ├─ Text 通道   → sse delta{content}
              └─ Tool 通道   → sse delta{tool_calls:[{index,id,function{name,arguments}}]}
          TurnEvent::Completed{finish} → finish_reason = tool_calls|stop|length
          TurnEvent::Prefill{total}    → usage.prompt
  ──SSE 返回§3.3 形状
pi: 执行工具 → 下一轮把{tool_calls + role:"tool"}塞回 messages ⇒ 回到入口
```

### 4.3 渲染对称律(前缀复用)★核心设计

**问题**:pi 每轮重发全历史;我们的渲染必须让"第 N+1 轮渲染出的 prompt"与"第 N 轮进 KV 的
token 序列"在**前缀上逐 id 相等**,否则守卫失配 ⇒ 全量重算。

**律(本专项引入)**:

> **R1(思考对称)**:我们返回给客户端的 `reasoning_content` + `content`,在回填后按
> `<think>{reasoning}\n</think>\n\n{content}` 渲染,必须与**生成时的 token 流**逐 id 相等。
> 其中 `\n</think>\n\n` 的空白形态由**快照模板**决定(不在 Rust 里手写)。

**R2(模板唯一)**:任何"起手/预填/包装"文本只能来自快照模板的渲染结果 —— 禁在
`openai.rs` 里手写 `<|im_start|>assistant\n<think>\n`(现状病灶:手写预填 = 不对称)。

**R3(工具块对称)**:历史工具轮必须按模板的**同一分支**(`_tool_format` 一致)渲染,
否则从 assistant 工具轮起点起逐 id 不同。

**R4(回退安全)**:守卫失配 ⇒ 全量重算(现状语义不变)。**对称是优化,不是正确性前提**。

⇒ **R1~R3 达成时,同会话守卫(REQ-AGT-09 + 隐式会话)自然命中**;
  未达成时,仍有**跨会话块级前缀缓存**(REQ-AGT-08)作为与客户端无关的兜底。

### 4.4 会话身份与增量 prefill

| 情形 | 现行为 | 目标行为 |
|---|---|---|
| 请求带 `session` 键 | 哈希为 u64(不变) | 不变 |
| 请求**不带** `session`(pi) | `next_id += 1` ⇒ **每轮新会话** | **隐式默认会话 0**(`[agent] implicit_session = true`) |

- 隐式会话的**安全性**:守卫失配 → `reset()` + 全量重算,**正确性不依赖客户端配合**;
  两个客户端交错 = 频繁 reset(慢但不坏)。
- `gdn_slots` 必须 ≥ 并发活跃会话数(启动已有 fail-fast 校验)。

### 4.5 流式三通道状态机(带尾巴缓冲)

**约束**:`<tool_call>` **不是 special token**(已核:快照 21 个 special 里无工具相关),
所以它是普通 BPE 文本,**可能跨 token 边界切分**;而 engine 的 `Token{delta}` 是
"全量重解后取后缀差分" ⇒ 中间态可能吐出标记的半截前缀。

因此解析器必须是**尾巴缓冲 + 前缀树匹配**状态机:

```text
状态:Text ──"<think>""──▶ Think ──"</think>"──▶ Text ──"<tool_call>"──▶ InTool
                                                    │
                     InTool ──"</tool_call>"──▶ 完整块 → emit tool_calls(参数解析)
```

- 出栈条件:标记**完整**出现才切通道;半截标记留在缓冲,先不发(延迟 ≤ 数 token)。
- 通道切换时**排空**缓冲(避免把标记原文吐到 `content`)。
- 文本里合法出现的 `<` 等字符不得被误判(前缀树要求"能构成标记前缀"才缓冲)。
- **未知标记**(如模型自创 `<tool_call|>`)⇒ 按普通文本放行 + 一次性 warn。

### 4.6 失败与取消时序

| 事件 | 处置 |
|---|---|
| 客户端断连(`sink.send` 失败 / 写帧失败) | ✅ **已落地(AGT-10,2026-10-14)**:服务面写失败 ⇒ `EngineReq::Abort` 显式弃置;actor 侧投递失败 ⇒ 标记弃置(≤1 采样步)。引擎 `abandon` 走正常回收(块链/快照留存,GDN 格释放)⇒ 后继请求不受影响。实测:curl 3s 断开 ⇒ `事件投递失败…弃置` + `弃置 turn N`,下一请求正常 |
| 引擎 `pump` 错 | `abandon` 烧毁 turn + 广播(现状);已发 SSE ⇒ error 帧 + `[DONE]` |
| 生成中途 EOF(长度/截断)且 `InTool` 未闭合 | **不发射**半截 tool_call;`finish_reason="length"` |
| `<tool_call>` 后模型继续输出 | 允许(多块);`finish_reason="tool_calls"` |
| 空回复 | 至少发 role 首帧 + `finish_reason`(pi 需要 finish 才不会抛错) |

### 4.7 不变量(施工期可断言)

1. **I1**:`finish_reason` ∈ {stop,length,tool_calls} 且**恰好发一次**。
2. **I2**:usage 帧在 `[DONE]` 之前,`choices` 为空数组。
3. **I3**:`content` 通道**永不**含 `<tool_call>`/`</tool_call>` 原文。
4. **I4**:`tool_calls[i].function.arguments` 是合法 JSON 字符串。
5. **I5**:`tool_calls` 的 `index` 连续且每帧携带。
6. **I6**:渲染器输出**只**来自快照模板(除传入的变量外零拼接)。
7. **I7**:未知关键字段 ⇒ 400(不静默)。

---

## §五 判据与验收门(G0~G5)

| 门 | 内容 | 类型 |
|---|---|---|
| **G0 前置 spike** | minijinja(**Rust crate**)跑通快照模板(§7.1 缺件清单逐项);产物 vs **HF `apply_chat_template` oracle** 逐字节/逐 id 相等 | **✅ PASS(2026-10-14,4 用例 + 2 模板全逐字节相等,见 §7.1)** |
| **G1 单测** | 渲染器(含 tools/think/历史工具轮)× 解析器(两格式/多块/跨边界/Unicode/截断)× 状态机不变量 I1~I5 | `cargo test` |
| **G2 wire 冒烟** | `apps/server/tests/agent_wire_smoke.py`(stdlib,对已启动 server 跑契约断言:三通道帧序 / finish_reason / usage 帧 / 工具历史可渲染 / `tools:[]` / 非法与未支持字段 400 / 采样参数不静默) | **✅ 实测 17/17(2026-10-14,0.8B 档)**;`e2e.rs` 内嵌断言归 M-B |
| **G3 pi 真机冒烟** ✅(2026-10-14 PASS,见 §7.7) | `pi --model owl/… -p "在当前目录创建 hello.txt 并读回"` → read/write/bash 闭环;`/session` 里 toolCall 块与 thinking 块可见 | 人工 + 日志 |
| **G4 前缀真值门** | ① 同会话多轮:`[前缀复用] 增量 ≥1` 且输出 sha 与全量重算逐字一致;② 跨会话共享前缀同样通过 | ✅ **PASS(2026-10-14,`apps/server/tests/prefix_reuse_probe.py`)** —— 同会话命中 **165-225/187-238**、跨会话命中 **1536/1732**(块级缓存);**口径**:spec-off 下 token 级 sha 硬判(严格),spec-on 下降为 content 级(因 A8,见 §7.8) |
| **G5 长 ctx 回归** | std 五点 + 32k/64k 单请求 TTFT/decode 记录(定档依据) | llm_speedtest |

**反例/负判据(必须明确失败)**:
- 带工具历史但渲染丢工具轮 ⇒ G2 应红;
- 模型吐半截 `<tool_call>` ⇒ 不得产生 `arguments` 非 JSON 的帧;
- 未知字段被静默吞 ⇒ G2 应红。

---

## §六 里程碑与排期

| 里程碑 | 内容 | 工期 | 出口 |
|---|---|---|---|
| **M-A 最小闭环** | REQ-AGT-01..05 + G0/G1/G2/G3 | **1.5~2.5 天** | ✅ **pi 跑通一次 agent 任务(2026-10-14)** |
| **M-B 稳定可用** | REQ-AGT-06/07/09/10/13 | +1.5 天 | ✅ **已落地(2026-10-14,§7.9)**:采样/停止串真生效、隐式会话、abort、chunked、reasoning 档、ctx 定档决策 |
| **M-C 日常可用** | REQ-AGT-08/12 + G4/G5 + 接受率实测 + 11/14 | +2 天 | ✅ **已落地(同日)**:两层前缀复用 + G4 真值门 / cached_tokens 真值 / G5 三档数字 / agent 域 AL 2.68 / 4 并发墙钟 |

**GPU/端口排期(实测环境,2026-10-14)**:

| 项 | 值 |
|---|---|
| 卡序(**已实测**) | CUDA 序 `[0,1] = 3090 Ti ×2`、`[2,3] = 3080 ×2`;nvidia-smi 序不同(CVD 陷阱) |
| 本专项用卡 | **CUDA ordinal 0**(3090 Ti;`spec-27b-prod.toml` 用的是 ordinal 1) |
| 端口 | **8137**(`spec-27b-prod` = 8136) |
| 与 `prefill-graph` worktree | 并行、**不同卡**;⚠️ 其清场脚本会 `kill -9` 全部 GPU 进程(已误杀过一次),两侧都要改为**只杀自己的 PID** |
| 指标可比性 | 同为 3090 Ti 可比;若被迫落到 3080(20G)则池/ctx 需重定档,**读数不可与 3090 直接比** |

---

## §七 风险与未决

### 7.1 minijinja 支持度(G0 spike 靶子,最高风险)

**先澄清名词**:`minijinja` **本体是 Rust crate**(crates.io,作者 = Jinja2 作者 Armin Ronacher;
本机 cargo 缓存已有 2.14.0/2.24.0,线上 3.0.0),`cargo add minijinja` 即可,零 Python 依赖,
与 serde/toml 同级(符合“纯 Rust 链”原则)。**PyPI 上另有一个同名包**(minijinja-py,
PyO3 绑定) —— **不用**,会把 Python 拖进 server 进程;`minijinja-cli` 也只是 CLI,不用。

**G0 的 oracle 是免费的**:本机 `comfyui-test`/`sglang-test`/`vllm-test` 三个 venv 均含
`transformers + jinja2` ⇒ 用 `tokenizer.apply_chat_template(messages, tools=…,
add_generation_prompt=True, chat_template_kwargs=…)` 对同一输入渲染,**逐字节 diff**
（这是上游 HF 语义的真值面,比自己手抄一份参考强得多）。

| 模板构造 | minijinja 预期 | 处置 |
|---|---|---|
**T0 实测结果:✅ PASS**(2026-10-14,`/tmp/mj-spike`,不落仓库):

| 用例 | 内容 | 结果 |
|---|---|---|
| ctx1 | tools 块 + 单轮 user | ✅ 2229 B 逐字节相等 |
| ctx2 | **工具多轮**(assistant `tool_calls` + `reasoning_content` 回填 + `role:"tool"`×2 + 续轮) | ✅ 3219 B |
| ctx3 | 同 ctx2 但 `enable_thinking=false` | ✅ 2785 B |
| ctx4 | 无工具 + 关思考(纯闲聊路径) | ✅ 106 B |
| **0.8B** | `Qwen/Qwen3.5-0.8B` 原版模板 153 行(家族通用性)| ✅ 1591 B |

**兼容层清单(实测定案 = `chat_tpl.rs` 的构造面)**:

| 项 | 必须做 | 依据 |
|---|---|---|
| 空白控制 | `set_trim_blocks(true)` + `set_lstrip_blocks(true)` | HF env 参数(`chat_template_utils.py:489`),**非** minijinja 默认 |
| Python str 方法 | `set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback)` | 模板只用 5 个:`split`×24 / `startswith`×11 / `lstrip`×3 / `rstrip`×1 / `endswith`×1 —— **pycompat 全覆盖,零手写** |
| **`tojson` 覆盖** | HF 语义 = `json.dumps(x, ensure_ascii=False, sort_keys=False)` | ① **不转义 HTML**(`<`/`>`/`&`/`'`)—— Jinja2 内置会转义,HF 明确覆盖掉(`:480-484` 注释写明);② 非 ASCII **原样**;③ 分隔符 = Python 默认 `", "`/`": "`(**serde_json 紧凑输出少一个空格**);④ 数值表示(`30.0` 要补 `.0`);⑤ 键序 = 插入序(`preserve_order`)|
| globals | `raise_exception`(→ **结构化错误**,不许 panic 逃逸)、`strftime_now`(本模板未用,一并注入防漂移) | HF `:493-494` |
| features | `json` / `builtins` / `macros` / `preserve_order` / `loop_controls` | `namespace()` 由 builtins 提供(`functions.rs:452`),**实测可用** |
| 未用到 | `{% break/continue %}` / `{% generation %}`(HF `AssistantTracker`)/ `include` / `extends` | 模板全文扫描:只有 `set/if/for/macro` |

**T0 踩到的坑(已入契约)**

1. **`chat_template_kwargs` 必须展平到顶层**(§3.2 注):不展平则 `enable_thinking=false`
   **静默失效**,实测多出 `Reasoning effort…` + ` thinking\nBrief explanation…` 两段(差 434 B)。
2. **token 面(实测)**:`<|im_start|>`/`<|im_end|>`/`<tool_call>`/`</tool_call>`/` thinking`/`</think>`
   **全部单 token**(`<tool_call>` 不在 special 表里,但是 base 词表单 token);
   而 `<function=bash>` = **5 token**(`<`/`function`/`=b`/`ash`/`>`) ⇒ **工具解析器只能在文本层做,
   不得按 token 边界假设**(§4.5 尾巴缓冲的实证依据)。
3. HF 侧 `split_special_tokens` 实测 = **False**(`tokenizer_config.json` 里的 `true` 未生效)
   ⇒ 无"特判 token 被打散"风险;与 owl 的 `encode(text, false)` 一致(Rust `tokenizers` 0.21.4 = owl 同版本)。
4. 性能:整模板渲染 **< 1 ms**(含进程启动 3.75 ms/次)⇒ 每 turn 一次可忽略;
   `Environment` 进程内**编译一次**(构造面见 §4.1),禁每请求重编。

**T1/T2 施工期新增事实(2026-10-14,hex 自证)**

| # | 事实 | 影响 |
|---|---|---|
| 5 | 模板里**两种 think 标记并存**:主形态 = `<think>`/`</think>`(产物与 `add_generation_prompt` 段实测);长形态 `<thinking>`/`</thinking>` 出现在**历史 assistant 内容剥除**分支(`content.startswith(...)`;模板计数:主 12/18,长 3/10) | 解析器必须**主 + 别名双认**;`toolcall::markers_match_snapshot` 把这对耦合钉在仓库内金标上 |
| 6 | **证据读法纪律**:`<think>` 形态的字面量在**显示层会被当 HTML 标签吞掉**(显示成 `" think"`),抄"看到的"会写出错字符串 —— 本轮为此白烧 3 个测试周期 | **凡标记/特殊 token 一律看 hex 或码点**,不看渲染文本;写入文件的标记字面量随后务必 hex 复验 |
| 7 | `tojson` 四差异(实测):不转义 HTML / 非 ASCII 原样 / **Python 默认分隔符 `", "`/`": "`**(serde_json 紧凑输出少一个空格)/ 数值表示(Python `repr` 阈值:`exp<-4 || exp>=16` 走科学计数,指数至少两位) | `chat_tpl::py_json/py_float` 逐条实现 + 18 例 Python 期望值单测 |
| 8 | `serde_json` 必须开 **`preserve_order`**:默认 BTreeMap 在**解析时就把键排序** | tools schema 的键序会进 prompt,与 vLLM/HF 的 Python dict 序不一致 ⇒ G0 判据结构性不可能通过(已开) |

| 9 | **`arguments` 必须是对象**:OpenAI 协议传 JSON 字符串,而模板(`tc.arguments \| items`)要求 mapping;0.8B 原版模板在 **HF 里也失败** | vLLM 同款归一(`_postprocess_messages`)已落在 `openai::normalize_tool_calls`;金标生成脚本同步用对象形态 |

**回退方案**:原计划的"手写子集 + 对拍"**不再需要**(留作历史);若将来换模型家族撞上
minijinja 真有缺件,再启用该退路(需再裁决)。

### 7.6 ✅ 结案(当日修复本仓):**XQA 页表恒等链** —— 多会话 KV 污染

**现象(立案时)**:同档连续请求,第 N 个请求的输出对应**第 N−1 个请求的提问**;
冷启动第 1 个请求正确,其后全部"答上一题"(带 tools 时思考段甚至原文回显上一条 prompt)。

**二分矩阵**(单变量,每格 = 一次真实 boot + `tests/lag_probe.py` 4 问;配置在
`testdata/server/lag-*.toml`,全部可复跑):

| 变体 | 与基线的差异 | 结果 |
|---|---|---|
| V2 干净基线 | 基 `dc7027e` + 生产形态 | ❌ 滞后 |
| V3 spec off | `[spec] depth = 0` | ❌ 滞后(⇒ **投机无罪**) |
| V5 graph off | `no_graph = true` + `depth = 0` | ❌ 滞后(⇒ **CUDA 图无罪**) |
| V6 f16 头 | `head_quant = "off"` | ❌ 滞后(⇒ **int4 头无罪**) |
| V4/V1 free_legacy | `[cuda] free_legacy = true`(回 lane C 前保守栅栏) | ❌ 滞后(⇒ **lane C 无罪**) |
| **V7b / V8** | **`[dispatch] kv_fp8 = false`**(f16 池) | **✅ 4/4 各答各题** |
| V8 | 同上,且**只翻这一项**(spec d7/图开/int4 头/采样全同) | **✅ 4/4 —— 单变量定谳** |

#### 7.6.1 定谳更正:触发变量 = **XQA**,不是 fp8 量化本身

V8 的"单变量"其实同时动了两件事(fp8 池 **与** XQA 激活 —— XQA 的前置就是 fp8 池,
见 `state.rs` 池门)。补一格 A/B 即消歧:

| V11 | fp8 池 **+ 关 XQA**(`verify_xqa=false / decode_xqa=false`) | **✅ 4/4 干净** |

⇒ **病灶 = XQA 路径**(f16 池只是"顺带把 XQA 关掉"才看起来无恙)。

**根因(代码级,一读即明)**:

1. `crates/engine/src/state.rs`(XQA 资源面构造)—— 页表是**写死的恒等链**:
   `// 常量页表:单会话恒等链(0..nb)以 i32 位型走私进 f32 账叶`
   ⇒ 只在「单会话独占池、块从 0 起」时成立。
2. 两张图的喂槽**不对称**:verify 图声明了 `xqa_pt` 输入槽却**从无写入**
   (`exec/spec.rs` 只喂 `xqa_seq`/`xqa_sem`);裸 decode 图连槽都没有
   (`engine.rs` 里 `page_table: x.page_table.clone()`)⇒ 两条臂都读恒等链。

⇒ **多会话(pi 每请求新会话;块链从空闲链续)时 XQA 整段读池块 `0..nb` = 读到别家
KV** ⇒ 实测"答上一题"。冷启动第 1 个请求恰好在块 0.. 上 ⇒ 正确 ✓(与观测吻合)。

#### 7.6.2 修法(本仓已落)与验证

| 触点 | 改动 |
|---|---|
| `crates/engine/src/engine.rs`(裸 decode 图) | 声明输入槽 `xqa_pt`(尺寸 = 池 `nb`,init 保恒等链)+ ctx 绑定改 `pt_in` |
| `crates/engine/src/exec/spec.rs` | 新增助手 `xqa_pt_of(sid)`:会话真块链 → `f32::from_bits` 走私编码,尾部补零到 `nb`;verify 步把 `("xqa_pt", …)` 塞进 `vg.step`(借用序:在 `verify_graph.as_mut()` **之前**算好) |
| `crates/engine/src/exec/phases.rs`(裸 decode 步) | `has_input("xqa_pt")` 时每步写会话真链 |

**验证(全真机)**:
| 门 | 结果 |
|---|---|
| `lag_probe.py`,fp8 + XQA 开(原必滞后) | **✅ 4/4** |
| `lag_probe.py`,**真生产形状**(fp8 + XQA + flavor 白名单,无 workaround) | **✅ 4/4** |
| `agent_wire_smoke.py`(27B) | **✅ 17/17** |
| XQA vs v2 恒等探针 `gpu_attn_verify_xqa_matches_v2` | ✅ `max|v2−XQA| = 4.4e-3 ≤ 1e-2` |

⇒ `testdata/server/agent-27b.toml` **收回 flavor 生产形状**(不再需要 `kv_fp8=false`
与空白名单两个 workaround)。

**⚠️ 影响面(必须记账)**:此前所有 **27B fp8 + XQA 的多请求**运行都受此污染 ——
吞吐类 bench 不受影响(单请求为主),但**多请求的接受率/质量类读数**(含 AL 台账、
长 ctx 衰减序列、以及并行 `prefill-graph` 会话的同类测量)**需重跑复核**。

**连带发现(两条,均已立案)**:
1. **`snap_max = 0` 会在运行期 panic**:快照池为空 ⇒ `capture_snap` 的
   `expect("池非空")`(`crates/engine/src/state.rs:859`)爆在 owl-engine 线程,
   请求收到 503。**该组合无 boot 校验**(应 boot 期 fail-fast 或彻底禁用快照)。
2. **白名单缺口**:f16 池需要 spec-dflash flavor 白名单外的算子
   (如 `owl_qknorm_rope_kv_insert_f16_knhd`)⇒ 现以诊断态 `allowed_ops = []` 关闸门绕过;
   正式档应补白名单(或修好 fp8 池后回切)。

**给引擎 lane 的建议修法方向**:查 fp8 池的 **decode 写路径覆盖**与**会话切换时的池失效**;
并补一条"**多请求内容正确性**"回归门(现成的 `lag_probe.py` + wire 冒烟即可)。

### 7.7 ✅ 结案(2026-10-14 当日修复):**是采样器的锅 —— `rep_penalty` 累积惩罚**

> ⚠️ 本节初判"模型/提示侧"**已作废**:同一权重在 vLLM/sglang 上正常 ⇒ 病灶在我们自己。
> 证据链如下。

**决定性单变量 A/B**(同档、同 prompt、同模板/同权重,只动 `rep_penalty`):

| 变体 | 输出 |
|---|---|
| `rep_penalty = 1.15`(owl 自加默认) | ❌ 思考不闭合 → 撑满 4096 token 复读 / 34KB 词沙拉 |
| **`rep_penalty = 1.0`** | ✅ **完全正常**(思考正确理解任务 + 正文给出正确 bash 步骤) |

**排除掉的红鲱(全不是病因)**:投机(关掉仍崩)、模板两版(原版 175 行 / froggeric 461 行)、
`head_quant` 换 f16 头、`prefill_chunk` 放大、fp8/XQA。

**代码级根因**(`crates/engine/src/sampler.rs`):原实现对 `history` 的**每个元素**逐个施加惩罚 ——
同一 token 出现 k 次即被罚 k 次 = `rep_penalty^k` **指数累积**(k=10 → ÷4,k=30 → ÷66,k=60 ≈ 禁掉);
llama.cpp 是位集去重(每个不同 token 只一次)。⇒ **短输出完全无感**(常见词 ≤2 次),
**长思考/长输出必崩** —— 这正是此前"短探针全绿"假象的根源。
**且该旋钮是 owl 自造的**:checkpoint `generation_config`、OpenAI 协议、pi→vLLM(无 `samplingParams`)
都不施加 ⇒ **对参考实现的单方偏离**。

**修法(已落仓)**:① `sampler.rs` 去重后施加(拷贝 + 排序去重,~50µs;新增回归门
`rep_penalty_once_per_distinct_token`);② 默认值 1.15 → **1.0**(`SamplerCfg::default` +
`SamplingCfg` 默认 + 键表文档)。采样器单测 **7/7 绿**。

**T7 真机冒烟 ✅ PASS**(**真生产形状**:快照 froggeric 模板 + 投机 depth 7 + fp8(/XQA)+ 修复后采样):

```
pi turn1   stop=toolUse  toolCalls=[bash: echo "hello owl" > /tmp/pi-smoke/hello.txt && cat …]
服务端     完成(tool_calls;tools=1)
pi turn2   stop=stop     "完成。已在 /tmp/pi-smoke/hello.txt 写入 hello owl,读回确认内容一致"
产物       /tmp/pi-smoke/hello.txt = "hello owl" ✓
```

⇒ **M-A 出口达成**:pi ↔ owl 全链(工具调用 → 工具结果回填 → 三通道 → usage → `finish_reason`)
在真机生产档跑通。

**留下的教训**:① 采样器这类"只在长输出显形"的缺陷**必须用真机长回合做门**(短探针会给假绿);
② 自加旋钮必须记账"与参考实现的偏离",并在 agent 流量上重验。

### 7.8 🆕 立案(2026-10-14):**spec 臂 token 级非路径稳定**(A8)

**现象**:同一 prompt、同 `temperature=0`、三个**冷会话**(互不共享前缀),
reasoning 长度分别是 **240 / 458 / 220B**(content 稳定为 `27`)。**16k 与 32k
档都出现**(不是 ctx 档特有)。

**含义(三条,都要记账)**:
1. `temperature=0` **不再保证逐字可复现** ⇒ 本项目"恒等门/真值门"类判据
   在 spec-on 下不可靠 —— **G4 的 token 级 sha 硬判必须 spec-off 跑**
   (`prefix_reuse_probe.py` 已按开关自动降级为 content 级并打印观测);
2. **请求的 `temperature/top_p/seed` 在 spec 臂基本不生效**(spec 的 token 选择
   是 verify 行 argmax,不走 `sampler::sample`)⇒ **AGT-06 的覆盖面要如实记账**
   (已生效于**裸 decode 臂**;spec 臂待 A8 结案);
3. **与 A7(工具调用遵循度)疑同源**:若 spec 臂实际以某个固定温度在采样,
   pi 发来的 `temperature=1.0` 根本没进采样链 ⇒ "模型不调工具"可能部分由
   此解释。**建议合并排查**。

**假设与修法候选**(未验证):verify 图在**捕获期烘焙了采样器**(图内节点),
per-turn 参数进不了图。候选:① 采样参数走图输入槽(与 `xqa_pt` 同款修法);
② spec 臂在"请求要贪心"时走纯 argmax 数值面(需先确认 verify 行 argmax 与
裸 decode argmax 的数值一致性 —— 目前**不一致**,这也可能正是非稳定的来源);
③ host argmax 校准面(`DiagOpts.host_argmax`)已有先例。

**排查入口(单变量)**:spec 开/关 × 请求 temperature(0 / 1.0) → 看输出分布;
再看 `/debug/metrics` 的 `spec.m*` 直方图是否随请求温度变。

### 7.9 ✅ 第二批交付实录(2026-10-14,AGT-06/07/09/10/11/12 + B1-B4 + A3/A4/A5 + D1/D3/D4)

**原则**:本批把"收下但未生效"全部清干净(AGT-06/07/12),把"能用"推到"长期
可用"(AGT-09/10/11/13),并用**真值门**证明前缀复用不改变答案(D1/G4)。

| 面 | 交付 | 判据/实测 |
|---|---|---|
| AGT-06 | `TurnOpts{sampler,stop,seed}` + 引擎 per-turn 采样合并(部分覆写)+ stop 尾缓冲 | 单测 4 + e2e stop 受控 A/B(贪心必复读提示词:对照吐 `BANANA…`,带 stop 截断且 `finish_reason=stop`)+ wire 22/22 |
| AGT-07 | `reasoning_effort` 直通模板 + pi compat 打开 | 模板值域自持,零自造映射 |
| AGT-09 | `[agent] implicit_session` | lag_probe 4/4(交错提问)+ G4 同会话命中 88% |
| AGT-10 | 断流 ⇒ Abort/弃置 | curl 3s 断开实测:日志 `弃置 turn N`,后继请求正常 |
| AGT-11 | chunked 请求体 | 单测 4 + curl chunked 200;keep-alive 挂 hyper 迁移(permit 语义理由已记) |
| AGT-12 | `cached_tokens` 真值(含块级命中) | G4 探针数字与引擎 `prefill encode @M` 逐项吻合 |
| AGT-13 | 定档决策:**维持 16384** | G5:16k 1264 t/s / 32k 1044(TTFT@30k 28.7s)/ 64k 850(70.6s);32k/64k 可用不转正(A8 + prefill −17%) |
| B1/B2 | `[agent]` 配置模块 + 模板覆盖 | 账外键拒/枚举严格单测 |
| B3 | e2e agent L1 七段 | 工具历史/thinking 回填用 **prompt_tokens A/B** 做硬判(不拿模型脾气做门) |
| B4 | `agent_gates.sh` 三门一键 | lag + wire + e2e |
| A3/A4/A5 | 读数复核 / 质量抽查 / f16 白名单 | 见台账 A 组(全部 ✅) |
| D1/G4 | 两层前缀复用 + 真值门 | 同会话 88% / 跨会话 89%(1536/1732);**机制发现**:跨会话复用需"页面对齐快照"(短 prompt 常无快照可复用) |
| D3/D4 | agent 域 AL / 并发墙钟 | AL **2.68**(156 轮,零降级);4 并发 = 0.94×(单槽串行,无并行) |

**新工具(入仓)**:`prefix_reuse_probe.py`(G4 真值门,自动探测 spec 开关)、
`agent_gates.sh`(三门)。**踩坑入册**:① `kill` 后不等内存释放就重启 ⇒ 新进程 OOM
而旧进程仍在服务(探针打到了旧 server,结论一度被污染)→ **重启纪律 = 等 GPU
占用清零**;② 短 turn 的校正帧必须在首个事件前发(否则 sink 已摘除丢帧);
③ 拿"模型肯不肯发工具调用"做门 = 拿模型脾气做门 ⇒ 改用 prompt_tokens A/B。

### 7.10 🔴 A7+A8 合并排查实录(2026-10-14 深夜;**定责完成,机制部分定谳**)

**定责(vLLM 对照,同模型/同请求/渲染逐字节相同)**:

| 引擎 | 同一条工具请求(temp=1.0,思考开) | 结果 |
|---|---|---|
| **vLLM**(`cyankiwi/Qwen3.8-27B-AWQ-INT4`,TP2,`--tool-call-parser qwen3_coder`) | 同 messages + 同 tools + 同 kwargs | **3/3 真发 `tool_calls`**(`{"command": "echo owl-agent"}`) |
| **owl**(生产档,prefill_split 缺省 false) | 同上 | **0/12**;且 prefill 末行 logits **塔平**(top2 = `28`:9.45 / 9.16) |
| owl(加 `prefill_split = true`) | 同上 | **6/6 真发 `tool_calls`**;同臂 decode **39 → 182 t/s** |

**渲染无罪**:同一请求经 HF `apply_chat_template` 渲染 vs 引擎 `OWL_PROMPT_DUMP` → **逐字节相同**(含 `<tools>` 块/`<IMPORTANT>` 段/`thinking` 预填)。

**已排除的红鳍(全不是病因)**:int4 lm_head(`head_quant=off` 同样塔)、
采样参数(温度 0/0.2/0.5/0.7/1.0 全塔,同一批退化串)、rep_penalty(=1.0 关)、
B4 自适应降级(`degrade_after=0` 不变)、`verify_xqa` / `decode_xqa`(单变量均不变好,
**除了** spec-off + decode_xqa=false 那一格)、prompt 渲染、模型本身(vLLM 无罪)。

**已定谳的机制一:`prefill_split = false`(缺省)的 prefill 会腐坏模型上下文**
—— 同一 prompt 在 split/非 split 下 prefill 末行 logits 差异巨大
(9.45 vs 25.41 的绝对尺度 + argmax 从 `28`("=")变为 `760`("The"))⇒ 输出退化
为复读/短废串。**该缺陷同时压垓投机接受率**(decode 39 vs 182 t/s)。
⇒ agent 档已加 `[dispatch] prefill_split = true`(e2e 档早已是 true ——
这就是为何 e2e 的 agent 面断言全绿而 pi 真机全红)。

**待定谳的机制二:spec 臂的 token 选择不走采样器(贪心性)**
—— spec 路径的 token 全部来自 verify 行 argmax(`exec/spec.rs` 里
`m = drafts[i]==ids[i]` 的贪心接受 + `bonus = ids[m]`),全仓 `sampler::sample`
**只有一个调用点**(裸 decode 步)⇒ **spec 开时,请求的温度/种子基本不生效**,
而贪心正是本项目已定谳的"27B instruct 复读吸引子"姿势。pi 真实流量
(12.7k token 系统提示 + 多工具)在 `prefill_split=true` + 32k 池下仍塔成
`":\n"` 复读至 max_tokens —— 与机制二相符。
**判据(待跑)**:spec-off 下 pi 真机是否完成工具闭环。

**A8(温度=0 不可复现)的最新定位**:已排除 prefix reuse / degrade_after /
两个 xqa 开关 / prefill_split;**实证两件**:
① 输出是"**运行史的函数**"—— 同一 boot 内按 run 序号稳定复现(240/458/220B
三连,4 个独立 boot 一致,1 个 boot 偏差);
② **池内容会改变结果**:干净池 240B vs 先跑一轮 filler 后 428/405B(同一 prompt
同 temperature=0)⇒ 指向"**spec 臂读到了本轮未覆写的残留状态**"(主池尾部/
草稿池/快照其一)。⇒ 与 A7 同族(都在 spec/数值路径),但非同一开关。

**新立案**:
- **A9** = `prefill_split=false` 的 prefill 数值缺陷(已定位到"末行 logits 塔平"
  这一现象级证据;agent 档已绕开,引擎侧待修)
- **A10** = `prefill_split=true` 下**跨会话块级前缀缓存不命中**(D1 ② 失效,
  实测 `cached_tokens=0` ×2;同会话增量仍命中)
- **A11** = spec 臂采样化(随机投机接受 / bonus 采样) —— 修 A7/A8 的正解

### 7.11 🔴 A7/A8 深挖(2026-10-14 深夜):**prefill 末行 logits 的布局相关错读**(未根治)

**定谳到部件(证据链)**:工具 prompt 的**首 token 错** → 全篇复读/幻觉完成:

| 观测 | 值 |
|---|---|
| 健康臂(prefill 末行 logits top2) | `760`("The") / 24.2-27.4,**gap 4-7** |
| 病臂 | `28`("=") / 9.453 或 `50303` / 9.320 —— **塌平**、错 argmax |
| argmax 核 vs host top1 | **恒一致**(28==28、760==760)⇒ 内核无罪,是 **logits 本身错** |
| hidden `hb` | 请求内 hash **稳定**(未被覆写) |
| int4 头 vs f16 头 | **都翻**(不是头) |
| spec / 采样 / B4 降级 | 关掉仍翻(不是 spec;A11 已另修采样) |

**关键判据 = "布局运气律"**:下列**每一个**变量都曾让结果翻转,而它们**都不是因**——
加只读诊断(`dflash_probe` / 一次 `dtoh` / 一次 plain eval)、换 `head_quant`、换
`launch_sync` / `free_legacy`、换 `prefill_split`、甚至**只是重启**(实测 6/6、5/6、
0/6、0/6、5/6 跨 boot)。其中 `prefill_split` 已由**核发射账**证伪(两次 boot 的
`srv.op.*` prefill 核逐项相同 ⇒ 该 flag 对本 prompt 是空操作)。

**⚠️ 跨流 UAF 假设已证伪(2026-10-14 深夜,同一轮深挖)**:① `STREAM_H2D` 在
`server/mod.rs` 里**从未用于任何上传**(只在流初始化出现两次)⇒ 上传/核/alloc/memset
**全在 COMPUTE 一条流**;② 设备地址复用实测**高频**(400 行窗口 152 次)但**全部是
COMPUTE→COMPUTE**(驱动对同流复用有流序保证)。⇒ 病灶不在 server 的流序/栅栏,
而在 **host 跑在 GPU 前面所暴露的"发射顺序/依赖"** —— 任何 host 扰动(加打印、
一次 dtoh、多一次 plain eval、换 boot)都会让症状翻转,正是"窗口被 host 节流关掉"
的特征;FFI 家族(marlin/cublas)的发射已确认是 server 线程内 inline(非异步),
故嫌疑面收窄到**命令序与求值序的一致性**。

**下一个仪器(必做,1-2d)**:server 发射面加**单调序号账** —— 每条
launch/upload/memset 打 `seq + 核名 + 块 id`;对一次工具 prompt 的 prefill 末行
DAG(lm_head gemm → argmax)核对**命令序是否与求值序一致**(gemm 的发射是否可能
排在 argmax 之后)。

**已落缓解(未根治)**:prefill 末行两处 eval(`Narrow` 末行 + `argmax`)改走
**plain(不回收)** —— 方向正确但实测仍随 boot 翻(布局运气)。**立案 A13**
(引擎/backend lane):在 `resolve()`/allocator 侧打**设备地址**账,查
「free 后地址被复用而读者在途」的窗口。

**A11 同时落地(spec 臂采样化)**:verify 图新增 `logits` 出槽 + 投机轮改
**抽-比式采样接受**(`exec/mod.rs::spec_sampled_accept_prefix`:逐行从目标分布
抽 token 与草稿比对,首个不等处改发重抽 —— 分布精确,不依赖草稿分布)+ per-turn
采样参数/RNG/惩罚历史接入。意义:**请求的 `temperature/top_p/seed` 在 spec 开时
真正生效**(原为 verify 行 argmax = 事实贪心)。单测 3 + 真机 `spec.sample.rounds`>0。

**pi 真机的进步与残余**:从"完全不发工具调用"→ 日志出现
`工具块畸形(tool_call_unclosed;0B)已丢弃`(模型**发出了 `<tool_call>` 但未闭合,
思考跑满 8192 token)⇒ 引擎侧已正确,**残余 = 长 prompt 下思考不收敛/格式不闭合**
(模型侧,归 D2 的档位/模板 A/B)。

**方法教训(本轮两次自纠)**:① `prefill_split` 与 `head_quant` 都曾是**空操作或
布局运气**被误当因果 —— 单变量实验必须同时看**核发射账**与**设备地址账**;
② 观测者效应(加打印即变好)本身就是最强的线索:它指向"布局/地址相关",而不是
"语义错"。

### 7.2 魔改模板的复刻度

快照模板 461 行含:工具格式指令块、连续失败警告启发式(`_strong_error`/`_weak_error`)、
`max_tool_*_chars` 截断、"exit code/error json" 判据。**渲染器全量执行它**(minijinja 直读),
不做子集裁剪 —— 但需知:**这些启发式直接塑造模型行为**(如失败警告注入),行为真值门
只能靠 G2/G3 真机验证。

### 7.3 `tool_call_format` 选择

默认 **xml**。若真机发现 27B 在 xml 格式上产废(或 pi 侧解析问题),可切 `json`
(`chat_template_kwargs` 之外我们自控)——**必须与解析器同批切换**,并留 A/B 记录。

### 7.4 模型侧未知数(非本专项能解)

- 27B AWQ INT4 + DFlash2 草稿在 **agent 域(代码/工具 JSON)** 的接受率**未知**
  (已知:数学 175~177 / 竖式 113~118 / 长城 53~54 / 随机域净亏)⇒ M-C 必须实测,
  不行就降 `spec.depth` 或关 spec。
- 工具调用的**格式遵循率**(27B 是否稳定按模板指令输出 `<tool_call>`)未知 ⇒ G3 是唯一判据。

### 7.5 其余

| # | 风险 | 处置 |
|---|---|---|
| 1 | `max_tokens` 缺省 = **24**(`openai.rs::DEFAULT_MAX_NEW`)——pi 会给 32k,但裸 curl 会短 | 默认值上调 + 超池校验 |
| 2 | 会话表无界增长(隐式会话修掉主要来源) | 加 TTL/LRU 上限(M-B) |
| 3 | 前缀缓存 × GDN 快照在 27B spec 档的等价性(机制在 0.8B 已验) | G4 真值门 + `snap_max` 调参 |
| 4 | SSE 长 prefill 期间无字节(undici bodyTimeout 默认 300s) | 当前 60k prompt ≈ 46s,安全;若 P1‴ 前上 100k+ 需心跳 |
| 5 | pi compaction 阈值与真实池不一致 | models.json `contextWindow` 与 flavor 同源(一处改名,两处同步 → 加 boot 校验) |

---

## §八 施工任务单(M-A)

| # | 任务 | 文件面 | 判据 | 依赖 |
|---|---|---|---|---|
| **T0** | **G0 spike**:minijinja × 快照模板(461 行) | `/tmp` 独立小 crate,**不落仓库**(结论回写本文 §7.1) | 逐字节相等 + 缺件清单定案 | — |
| **T1** | `chat_tpl.rs`:渲染器(env 编译一次 + 快照路径 + HF filter/global 注入 + 结构化错误) | 新增 | G0/G1 | T0 |
| **T2** | `toolcall.rs`:三通道状态机 + 两格式解析 + 不变量 | 新增 | G1(I1~I5) | — |
| **T3** ✅ | `openai.rs`:全字段解析(**原样透传 + 结构化校验**)+ `normalize_tool_calls`(vLLM 同款)+ 三通道 SSE + `finish_reason=tool_calls` + 400 面 | 改 | G1/G2 ✅ | T1,T2 |
| **T4** ✅ | `render_prompt` 退役(§4.3 R2);prompt 一律 `ChatTpl`;模板 boot 期装载(失败拒启) | 改 | G0 ✅ | T1 |
| **T5** ✅ | `config.rs`:`[agent]` module(模板路径/隐式会话开关/tool 解析开关/max_new 上下限/tool_call_format/auto_disable/reasoning_effort) | 改 | G1 ✅ | — |
| **T6** ✅ | `agent_wire_smoke.py` 落地并实跑 **17/17** | 新增 | G2 ✅ | T3 |
| **T7** ✅ | **pi 真机冒烟 PASS**(§7.7)+ `~/.pi/agent/models.json` 加 `owl` provider(备份在)+ 本轮另修引擎侧 3 项(XQA 页表恒等链 / `snap_max=0` panic 记录 / 采样器去重与默认值) | 配置 + 文档 | G3 ✅ | T3 |

**施工纪律**(沿用本工作区教训):
① 不主动 git commit;② 只杀自己的 PID(禁 `nvidia-smi | xargs kill -9`);
③ GPU 验收串行;**⚠️ 重启纪律:kill 后必须等 GPU 占用清零再起新实例**
(实测:kill 后 3s 就重启 ⇒ 新进程 OOM 而旧进程仍在服务 ⇒ 探针打到了旧 server,
结论一度被污染);④ 每个新功能先跑**最小独立用例**;⑤ 不许吞可失败返回值;
⑥ 新增旋钮走 `config.rs` 结构体 + `from_config` 登记(禁散点 env 直读);
⑦ **判据门禁拿模型雿气做门** —— 模型行为相关的断言用"受控 A/B + 前提检查"
(如工具历史用 prompt_tokens A/B、stop 串用贪心必复读提示词),否则红绿不可复现。

---

## §九 挂账(不阻塞 M-A)

1. ~~`[agent]` 面尚未接入 `OwlConfig` 的 deny_unknown_fields 全账~~ ✅ **已结(T5/B1)**。
2. ~~reasoning 档位映射~~ ✅ **已结(AGT-07)**:`reasoning_effort` 直通模板。
3. `auto_disable_thinking_with_tools` 的产品选择(有工具时是否关思考)—— **旋钮已就位**,需真机 A/B 数据(D2)。
4. 视觉/图像输入(`<|vision_start|>` 分支)保留未启用。
5. undo/rollback 会话级重置接口(pi 的 compaction 触发后的显式 reset,避免残留块链)。
6. `/debug/agent` 诊断面(最近一次渲染 prompt 原文 + 解析器状态机轨迹;现只有 `OWL_PROMPT_DUMP`)—— **对 A7/A8 排查也有用**。
7. `completion_tokens_details.reasoning_tokens`(需按通道计 token 数;现只有文本聚合)。
8. **A8(spec 臂 token 级非稳定)** —— 见 §7.8;与 A7 疑同源,合并排查。
9. **A7(工具调用遵循度)** —— 台账第一优先;AGT-06 已落地,复测应在新的采样面下做。

---

## §十 变更记录

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-10-14 | v0.9 | **§7.11:A7/A8 深挖** —— 定谳到 **prefill 末行 logits 的布局相关错读**(布局运气律:所有变量都是扰动而非因;`prefill_split` 由核发射账证伪)⇒ 真凶 = 释放块**设备地址复用**(UAF 类),**未根治**,已落 plain-eval 缓解 + 立案 A13;**A11 spec 臂采样化落地**(verify 出 logits 槽 + 抽-比式接受,分布精确,单测+真机);G4 口径定谳(token 级只在 spec-off);pi 真机进步到"发 `<tool_call>` 未闭合"(残余归 D2) |\n| 2026-10-14 | v0.8 | **第二批交付**:AGT-06(per-turn 采样/stop/seed,引擎侧 `TurnOpts`)/07(reasoning_effort 直通)/09(implicit_session)/10(abort)/11(chunked)/12(cached_tokens 真值含块级命中)/13(定档维持 16k);B1-B4(`[agent]` 配置模块 + e2e agent L1 七段 + `agent_gates.sh`);A3/A4/A5 ✅;D1 ✅(**两层前缀复用 + G4 真值门**:同会话 88% / 跨会话 89%)、D3(agent 域 AL 2.68)、D4(4 并发无并行);**新增 §7.8 A8 = spec 臂 token 级非路径稳定**(与 A7 疑同源,恒等/真值门一律 spec-off 跑);实测数字全部入 §7.9。 |
| 2026-10-14 | v0.7 | 清 A 组引擎债三项:**A1 拒启可见**(四项真机实测)/ **A2 `snap_max=0` 不再 panic**(真机 4/4、零 503)/ **A6 think 别名**(+单测);生产档回归绿(wire 17/17、lag 4/4、pi 全链 1 次成功)。**新增 A7:工具调用遵循度实测 1/9**(模型倾向直接作答+幻觉完成,日志证明非解析丢失)= 现第一优先,台账见 `roadmap.local/agent-服务面-剩余工作项-2026-10-14.md`。 |
| 2026-10-14 | v0.6 | **T7 PASS + §7.7 结案**:病灶 = `rep_penalty` 逐元素施加 ⇒ `1.15^k` 指数累积(短输出无感、长思考必崩),已修(去重 + 默认 1.0,单测 7/7);真机生产形状(froggeric + spec d7 + fp8/XQA)pi↔owl 全链跑通,产物文件落地。**M-A 达成**。另记:本轮另修引擎 3 项(XQA 页表恒等链 × 多会话、配置错误静默 exit、`snap_max=0` panic 立案)。 |
| 2026-10-14 | v0.5 | **§7.6 结案并修复**:定谳更正为 **XQA 页表恒等链**(fp8 只是触发条件 —— XQA 前置为 fp8 池;V11 消歧);修法 = 页表改走图输入槽 + `xqa_pt_of` 会话真链,三处触点;验证 = 生产形状 lag 4/4 + 27B 冒烟 17/17 + XQA 恒等探针;agent 档收回生产形状;影响面(多请求接受率/质量读数需复核)已记账。 |
| 2026-10-14 | v0.4 | **V1–V8 二分定谳:T7 拦路虎 = `kv_fp8`(fp8 KV 池跨请求污染)**;f16 池规避后 `lag_probe` 4/4 + **27B 真机 wire 冒烟 17/17**。服务面侧 M-A 收口;新增 §7.7(模板/模型侧第二阻塞:froggeric 模板下工具提示复读塌陷,上游原版模板可出正确 tool_call)+ 两条连带立案(`snap_max=0` 运行期 panic、f16 池白名单缺口)。可用档 = `testdata/server/agent-27b.toml`。 |
| 2026-10-14 | v0.3.1 | **T7 真机冒烟暴露拦路虎(§7.6)**:27B spec-dflash 档多请求「一轮滞后」污染(0.8B 无 / prompt dump 证明服务端无罪 / lane C、GDN 格、session 均排除);另发现 `depth=0` 首请求 panic(`state.rs:859 池非空`)。**M-A 代码面(P0 四项)全部完成并 26 单测 + 17/17 wire 冒烟通过**;G3 待引擎侧修复。 |
| 2026-10-14 | v0.3 | **T3/T4/T6 完工**:`openai.rs` 全面重写(原样透传 + 结构化校验 + 三通道 SSE + 400 面 + `normalize_tool_calls`)、boot 期装载模板(失败拒启)、`render_prompt` 退役、`agent_wire_smoke.py` **17/17 全绿**;单测 26 全绿。**M-A 只剩 T7(pi 真机冒烟)**。 |
| 2026-10-14 | v0.2.1 | **T1/T2 完工**:`chat_tpl.rs`(9 例)+ `toolcall.rs`(12 例)= 21/21 全绿;G0 金标门入仓(`apps/server/tests/fixtures/chat_tpl/` + `gen_fixtures.py`);新增事实 5-8。代码在图 `feat/agent-serving`(worktree `packages/worktrees/owl/agent-serving`)。 |
| 2026-10-14 | v0.2 | **T0(G0)实测 PASS**:minijinja 兼容层定案(trim/lstrip blocks + pycompat + HF `tojson` 语义 + raise_exception 注入);4 用例 + 0.8B 模板逐字节相等;踩坑入契约(`chat_template_kwargs` 展平、`<function=>` 5-token、单 token 标记面、渲染 <1ms);回退方案退役。 |
| 2026-10-14 | v0.1.1 | 名词澄清:`minijinja` = **Rust crate**(PyPI 同名包为 PyO3 绑定,不用);G0 oracle 改用本机 venv 内的 HF `apply_chat_template`(逐字节真值面);`tojson` 走 `json` feature,`raise_exception` 确认两边都需注入。 |
| 2026-10-14 | v0.1 | 立项:pi wire 契约取证(pi-ai v1.0.4)+ 现状对账(15 项)+ REQ-AGT-01..14 + §4.3 渲染对称律 + M-A 任务单 T0~T7。用户裁决:①M-A 最小闭环;②模板路线 = minijinja 直读快照;③与 `prefill-graph` 并行、用另一张 3090 Ti(CUDA ordinal 0)/端口 8137。 |

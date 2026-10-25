# agent 服务面专项:pi agent 对接(设计与施工)

> 版本:**v0.6**(2026-10-14 立项当日;**M-A 已达成**)—— 变更记录见 §十。
> 状态:**M-A ✅ 达成**(pi ↔ owl 生产档跑通工具闭环,§7.7);M-B/M-C 需求已列,待排期。
> 定位:**owl 第一次面向"真实 agent 客户端"的服务面收口** —— 客户 =
> 本机 `pi` coding agent(provider 类型 `openai-completions`)。
> **待办台账** = 本地台账《agent-服务面-剩余工作项-2026-10-14》(本文只记设计与判据,
> 剩余工作项、量级、执行序住那里)。
> 承接:`multi-session.md`(S0 会话/并发面)、
> `requirements.md` v1.0(需求基线,本专项新增 `REQ-AGT-*` 族)、
> 本地台账《phase4-agent-session》(引擎侧会话面,已落地)。
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

**(a) 前缀复用零命中**(已在 `multi-session.md` §六 立案):

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

> 编号族新增于本专项;评审后回写 `requirements.md` §裁决记录。
> 等级:P0(不满足 = 目标不可达)/ P1(不满足 = 可用性显著受损)/ P2(工程完备性)。

### 2.1 P0 功能门

| 编号 | 需求 | 验收口径(判据) |
|---|---|---|
| **REQ-AGT-01** | **模板真渲染**:渲染器读模型快照的 `chat_template.jinja`,经 **`hf-chat-template`**(底层 `minijinja`,transformers 兼容层;2026-10-24 起)渲染,按 §3.2 契约传参;`tools` 渲染进 `# Tools` 块 | **G0 token 等价门**:同输入下,owl 渲染 → tokenize 的 id 序列与参考实现逐 id 相等 |
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
| **REQ-AGT-11** | HTTP 硬化:keep-alive 或至少 undici 兼容性实测;chunked 请求体 | ✅ **均已落地**:chunked 请求体(单测 4 + curl);**keep-alive**(G5④,2026-10-19:请求循环 + 每请求 permit + 空闲超时 30s + SSE 改 chunked 定界;裸 socket 探针 `keep_alive_probe.py` 6 项全绿)。客户端侧仍需真 pi/undici 实测(未做) |
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

**外部依赖(2026-10-24 起由 `hf-chat-template` 承担)**:
`namespace(...)`(builtins 自带,✅)、`{% macro %}`(✅)、`| tojson`(HF Python 语义)、
`raise_exception(...)`、`.split/.startswith/.lstrip/.rstrip/.endswith`(`pycompat`)、
字典 `.items()`(✅)、`ns.attr = v`(✅)、`~` 拼接 / `is defined/is mapping/is iterable/is string`(✅)。
`chat_tpl.rs` 只保留 owl 的模板来源/agent 缺省变量接口，不再自带 `py_json`/`py_float`。

- **tokenizer 特殊 token(2026-10-24)**:`tokenizer_config.json` 的
  `bos_token`/`eos_token`/`pad_token`/`unk_token` 由 `hf-chat-template`
  的 `TokenizerConfig` 视图读取，按 `special tokens → agent defaults →
  request kwargs` 顺序注入；GGUF 单文件路径没有该配置 ⇒ 空。
  配置文件存在但解析失败 = 拒启（不静默丢配置事实）。

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

> **GGUF 裸函数块(2026-10-24)**:GGUF 上游模板实测会漏 `<tool_call>`
> 外壳，只吐 `<function=bash>…</function>`。解析器现把裸块也认作一次
> 工具调用（结构轨迹 `text→bare_tool`）；`parse_tool_calls=false` 时
> 仍按普通文本放行。

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
- **`reasoning_effort` / `enable_thinking` 归一化(2026-10-24)**:
  OpenAI 面只发 `reasoning_effort:"none"` 时，Qwen 模板会据此关思考，
  但 `ChannelSplitter` 旧实现只看 `chat_template_kwargs.enable_thinking`
  （缺省 true）⇒ 正文被整段归到 `reasoning_content`。现按 vLLM
  `build_chat_params` 同款在请求层归一化：`ChatRequest::effective_template_kwargs()`
  同步派生 `enable_thinking = reasoning_effort != none/off`（显式
  `enable_thinking` 优先）；`enable_thinking()` 与 splitter 同口径。
  门 = `server::openai::tests::max_new_clamp_and_thinking_flag`。
  pi 侧 Qwen 模型仍建议配 `compat.thinkingFormat="qwen-chat-template"`
  （`3090_dual` / `3080_dual` 的 27B 条目已配）。

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
- **`reasoning_tokens` 已透出(2026-10-19,G5①)**:真源 = **解析器相位归因** ——
  引擎 1 个 `TurnEvent::Token` = 1 采样步 = 1 token,而协议层每 token 调一次
  `ChannelSplitter::push`,故以 **push 入口时的解析器状态**判该 token 属不属于思考
  相位(`ChannelSplitter::reasoning_steps`;**含 `</think>` 标记本体** —— 它由思考段生成;
  其后首个正文 token 已落 Text 态)。**不用 tokenizer 反数**:标记文本被解析器吞掉,
  反数与真值天然不等,且多一次 CAS 往返。流式(usage 帧)与非流式(`completion_json`)
  **两形态同源**(单点 `usage.reasoning`)。
  真机实测(dev-08b / 3090 Ti):thinking 开 ⇒ 80/80;单 token 边界(思考与答案同
  token)⇒ 1/1;thinking 关 ⇒ 0。门 = `agent_wire_smoke.py`(两形态断言)+
  `toolcall::tests::reasoning_steps_are_phased_by_channel`(三档相位)。

### 3.6 诊断面 `GET /debug/agent`(G5②,2026-10-19)

**路径语**(与 `/debug/launches` 同构,空段 = 全仓;非法段 400 而不静默回缺省):

| 面 | 语义 |
|---|---|
| `GET /debug/agent` | 全仓(旧→新;缺省深 `agent_trace::TURNS = 8`) |
| `GET /debug/agent/<n>` | 最近 n 轮(夹到仓深) |
| `POST /debug/agent/clear` | 清仓 → `{"ok":true,"cleared":N}` |

**一轮记录五面**(`turns[]` 元素):
```json
{"seq":5,"at_unix_ms":1792354106000,
 "req":{"model":"…","stream":false,"messages":1,"roles":"user","tools":0,
        "thinking":true,"reasoning_effort":null,"session":null,"max_new":8,
        "ignore_eos":false,"stop":1},
 "prompt":{"bytes":120069,"chars":…,"head":"<前 8 KiB>","tail":"<后 24 KiB>",
           "omitted_bytes":87303,"truncated":true},
 "usage":{"prompt_tokens":13,"completion_tokens":8,"cached_tokens":0,"reasoning_tokens":8},
 "outcome":{"finish_reason":"length","tools_called":0,"strays":0,"malformed":0,
            "ttft_ms":140,"elapsed_ms":327,"error":null},
 "events":["state think→text","swallow_lf 2","tool #0 bash args=12B","finish text held=0B"]}
```
- `events` = **解析器结构轨迹**(`toolcall::StructEvent::line`):**只记结构**(状态迁移 /
  吞空白 / 游标 / 工具块 / 畸形 / 收尾),**不记 delta 文本** —— 不做第二份输出副本。
- **失败与拒绝同样留痕**:`outcome.error` 非空(如
  `"提交被拒:submit: prompt 24012 + 生成 8 超预算 4096"`;提前返回则由
  `TurnLedger` 的 `Drop` 兑底网记 `未结算`)—— 短 turn / 异常路径不靠“猜日志”。
- **内存上界**:`TURNS × (PROMPT_HEAD + PROMPT_TAIL + 轨迹) ≈ 8 × 32 KiB = 256 KiB`,
  与 ctx 无关(截断 + 显式 `omitted_bytes` 记账);**不落盘、不打日志**,
  转储 = 显式 GET(诊断面不主动泄漏会话内容)。

### 3.7 工具面 `POST /tokenize`(G5③,2026-10-19)

请求(**最小面**;`deny_unknown_fields` **开着** —— owl 自有面,不背兼容史,
多传字段 = 调用方预期落空 ⇒ 响亮拒):
```json
{"model":"qwen3.8-27b","prompt":"…","mode":"raw|chat","add_special_tokens":false}
```
| 字段 | 语义 |
|---|---|
| `prompt` | **必填非空**(本端点不做 `messages` 渲染) |
| `mode` | `raw`(缺省)= 纯文本编码;`chat` = 先套**家族单轮包装**(与裸文本提交同一律:`Tokenizer::chat_wrap`) |
| `add_special_tokens` | 只接 `false`/缺省;`true` → **400** —— 本引擎编码恒 `false`(不静默接受一个不生效的开关);要包装用 `mode="chat"` |

应答:
```json
{"count":6,"max_model_len":4096,"tokens":[…],"mode":"raw","model":"…"}
```
- `max_model_len` = 本档 `max_seq_tokens`(引擎侧唯一产地,经 actor 应答捎回)。
- **单一真源**:tokenizer 常驻引擎(`RunningEngine`),HTTP 层**不二次装载**
  ⇒ 包装律/特殊 token 律只有一份。代价 = 本请求排在**当前 turn 之后**
  (actor 单线程“收件与泵不相交”),工量毫秒级 CPU;生成期间调用会等本轮结束
  (要“随时可查”得把 tokenizer 克隆进 HTTP 层,前提是引擎侧先导出
  `chat_wrap + try_encode` 组合面 —— 另案)。
- **显式失败**:走 `Tokenizer::try_encode`(新增,Result 版);旧 `encode`
  吞错回退空序列(线上 16 处调用,签名升 Result 另案)—— `/tokenize` 不用它:
  “编码失败”与“文本是空的”必须可分。

### 3.8 传输面:keep-alive 与 SSE chunked 定界(G5④,2026-10-19)

| 面 | 语义 |
|---|---|
| 连接复用 | HTTP/1.1 缺省**复用**;`Connection: close` 尊重;HTTP/1.0 只认显式 `Connection: keep-alive`(缺省关) |
| JSON/文本应答 | `Content-Length` 自定界,**不发 `Connection` 头**(关不关由 serve 循环定) |
| SSE 应答 | **`Transfer-Encoding: chunked`**,以终止块 `0\r\n\r\n` 收口(**每个流必调 `http::sse_end`**) |
| 并发闸门 | permit **每请求**获取(旧:每连接 —— keep-alive 下空闲连接会占死并发位) |
| 收摊条件 | `Connection: close` / 空闲 **30s** / 单连接 **1024** 请求 / 非 1.1(505)/ 处理函数返 Err(帧不可信) |
| 非 HTTP/1.1 | **505 + 关**:chunked 应答是 1.1 语法,静默降级成 close 定界会让“复用判定”与“帧语法”两套律打架 |

- **为什么 SSE 必须 chunked**:旧路用“连接关闭”定界 ⇒ 一连接只能服务一个请求;
  pi/undici 连接池会把并发位耗在空闲连接上(keep-alive 的真正动机)。
- **空闲超时/请求数上限是常量**(`main.rs`:`KEEP_ALIVE_IDLE` / `MAX_REQUESTS_PER_CONN`),
  **不是配置键**:纯传输层、不影响引擎行为/测量口径;真要调的落点已定
  (`[server] keep_alive_secs`,添键即插)。
- **客户端要求**:SSE 客户端必须支持 chunked(undici/fetch/urllib/http.client 原生支持;
  手写裸 socket 客户端需自己解帧 —— 本仓 `e2e.rs` 的 `dechunk` 就是样板)。

### 3.9 错误面

| 场景 | 响应 |
|---|---|
| 请求体非法/缺 messages/未支持字段 | `400` + `{"error":{"message":…,"type":"invalid_request_error"}}`(生命周期正确,pi 会用于 compaction/重试判断) |
| 上下文超限(超 `max_seq`) | ✅ **已落地(2026-10-15)**，两段：① **预算收缩**(`engine::running::budget_clamp`):`prompt + max_tokens > max_seq` 时**不直接拒**，先把生成长度收缩到剩余额度(下限 1024 tok)，`finish_reason=length` 由客户端自行续写/压缩；② **真溢出**(连 1024 都放不下) ⇒ `400` + 报文带上 pi 词表的规范短语 `prompt exceeds max length` + `code="context_length_exceeded"`(`openai::error_json_for_reject`；引擎原中文诊断**逐字保留**) ⇒ pi 的 `isContextOverflow()` 认出并走 compaction/上下文修复。词表来源 = pi-ai bundle `OVERFLOW_PATTERNS`(实测 2026-10-15)，锁在 `server` 单测 `budget_reject_is_normalized_for_pi_compaction` |
| 引擎未就绪/断流 | `503`(现状) |
| 生成中错误(已开始 SSE) | `error` 帧 + `[DONE]`(现状 `TurnEvent::Failed` 分支) |

> 注意 pi 会识别**上下文溢出**类错误并自动 compact+retry;**不要把限流/瞬时错误伪装成溢出**。
>
> **为什么必须两段都要(2026-10-15 实测)**:pi 每轮**重发全量上下文**(系统提示 + 工具
> schema + 历史);在工作区根(`/path/to`，`AGENTS.md` 28 KB)它单次
> 系统提示就 ≈ **26.4k tok**(pi 自己的对话窗口只够装 24.6k —— 因为 `models.json.maxTokens
> = 8192` 把 32k 预算吃掉四分之一)。只做归一化 ⇒ 系统提示本身超的部分压不下去，pi
> 会反复 compaction 仍发不出去;只做收缩 ⇒ 真溢出时无报错信息可识别。故两段共存:
> **能跑就先跑(收缩)，真跑不动才报(归一化)**。

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
├─ chat_tpl.rs    ★新:模板入口(快照/GGUF/覆盖 + agent 缺省变量;渲染委托 hf-chat-template)
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
        └─ hf-chat-template render → prompt 文本
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

> **2026-10-24 更新**:本节是当时的 spike 记录。渲染层现已收敛到
> `hf-chat-template` 1.0（minijinja + transformers 兼容层）；owl 不再自维护
> `py_json`/`py_float`。保留本节作为兼容性证据与踩坑来源。

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
| 7 | `tojson` 四差异(实测):不转义 HTML / 非 ASCII 原样 / **Python 默认分隔符 `", "`/`": "`**(serde_json 紧凑输出少一个空格)/ 数值表示(Python `repr` 阈值:`exp<-4 || exp>=16` 走科学计数,指数至少两位) | 旧实现 `chat_tpl::py_json/py_float` 已退役；现由 `hf-chat-template` 提供，接缝单测保留 |
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
| AGT-11 | chunked 请求体 + keep-alive | chunked:单测 4 + curl;keep-alive:**G5④ 已落地**(裸 socket 探针 6 项 + wire smoke + e2e 27B 回归);真 pi/undici 实测挂账 |
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
`max_tool_*_chars` 截断、"exit code/error json" 判据。**渲染器全量执行它**(hf-chat-template 直读模板),
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
6. ~~`/debug/agent` 诊断面(最近一次渲染 prompt 原文 + 解析器状态机轨迹;现只有 prompt 落盘(`[probes] prompt_dump`,原 OWL_PROMPT_DUMP))—— **对 A7/A8 排查也有用**~~ ✅ **已结(G5②,2026-10-19)**:最近 8 轮的请求事实 / prompt 头尾快照 / usage 真账 / 结局 / 解析器结构轨迹;见 §3.6 与 §9.2(2)。
7. ~~`completion_tokens_details.reasoning_tokens`(需按通道计 token 数;现只有文本聚合)~~ ✅ **已结(G5①,2026-10-19)**:解析器相位归因,两形态透出,见 §3.5 与 §9.2。
8. **A8(spec 臂 token 级非稳定)** —— 见 §7.8;与 A7 疑同源,合并排查。
9. **A7(工具调用遵循度)** —— 台账第一优先;AGT-06 已落地,复测应在新的采样面下做。

---

## §9.1 前缀复用闭环实录(2026-10-19;server/engine 服务层专项)

> 目标:**同会话增量 + 跨会话块级缓存**两条路都要真命中,且 `cached_tokens` 是**真值**。
> 工具:`apps/server/tests/prefix_reuse_probe.py`(五判据)+ `testdata/args/prefix-reuse-27b.toml`
> (测装)+ `[probes] prompt_dump/debug`(`[dbg prefix]` / `[dbg snap]`)。

### (1) 已修:渲染对称律 R1 的**最后一块**(Rust 侧结构空白)

模板把 assistant 轮渲染成

```text
<|im_start|>assistant\n<think>\n{trim(reasoning)}\n</think>\n\n{content}<|im_end|>
```

—— `\n</think>\n\n` 是**模板的结构资产**。三通道切分器旧实现只吞标记本体,
把它后面的 `\n\n` 漏进 `content`;客户端回填后渲染**多出两个换行 token**
⇒ 守卫在第一个 content token 失配(实测:`cached_len=13376`、分叉 idx=13360)。

修:`ChannelSplitter` 新增 `swallow_lf`(关思考标记后**至多吞两个 `\n`**,
流式跨 chunk 时挂起不吐)。单测三条:`think_close_boundary_whitespace_is_structural`
(吞两个/吞一个/不吞/只吞两个)、`think_close_boundary_survives_chunk_split`(逐字符切分)、
外加原 `one_shot_xml` 断言按新律更新。
**首块边界不变**:`<think>\n` 在 prompt 里,reasoning 通道从生成流第一位起。

### (2) 已修(新发现 bug):缓存叶驱逐写死 `passes < 4` ⇒ 长链前缀缓存饿死

`KvStore::ensure_for_len` 池不足时逐缓存叶,旧循环上限 **4 趟**。
**单链前缀缓存里只有尾块是叶**(每块都有子链),逐掉尾块后其父才变叶
⇒ 每趟最多腾 1 槽。13k tok prompt(419 页)改一句话 ⇒ 需 419、空闲 306 ⇒
4 趟只腾 4 页 ⇒ 响亮拒"真实容量不足" = **500**。实测复现两次。
修:终止条件改"够用 ‖ `evict_device` 返回空(真无可逐)",不再用趟数当边界。
`owl-store --lib` 81/81 不变;13k 改句子场景不再 500。

### (3) 已落配置:`prefix_cache` 必开 + 快照保留账

| 项 | 事实 |
|---|---|
| `pool.prefix_cache` | **paged 池下不开 = 同会话也不复用**(内容寻址是唯一路径;旧"直接复用会话块链"已退役) |
| `snap_max` | 每条快照 152 MiB(27B);**=1 时解码期每页拍一次 ⇒ 把 prefill 期的检查点顶掉** ⇒ 实测 跨会话 T5 `cached=1536`(8 槽)vs `0`(1 槽) |
| `pool.snap_host_mb` | 状态腿宿主:快照可下沉(不顶掉);`0` = 沉不下就丢 |
| `cache_cap` | `nb/2 + host_slots`(硬编码)—— 长 prompt 链 > 半池时,缓存只留一部分,**复用面受它约束**(13k/16.9k 池实测只复用到 1.5k) |

### (4) 已修/已落:**容量定档**(store 自持策略)+ **末段页对齐**

**律(用户 2026-10-19 提醒,本专项据此落地)**:**store 不感知 I/O** —— 它是显存
管理器:engine 提容量需求,store 回答"谁进显存、谁出去、出去到哪(host/盘/丢)"
(`plan(need, protect) -> Verdict`,纯 host;engine 按 `apply_verdict` 搬)。
⇒ "多少块可以留在显存里等复用"是**驻留决策**,算式必须住 store:

- **`owl_store::CachePolicy`**(新):`Off` / `Cap(n)`(测试精确档)/ `Share{permille}`
  —— 容量 = `份额 × 设备池 + 宿主槽全额`(跨层索引:沉到宿主的条目仍占名额)。
  旧 `nb/2` 的算式从 engine(`session.rs`)移入 store;engine 只传**政策输入**
  (2026-10-22 起 `prefix_cache_share` 已退役:缓存可占整池,新请求逐出到 host/disk)。
- flavor 三层账(`spec-dflash`/`awq27b`):`prefix_cache=true` ·
  `host_cache_mb=4096`(KV 腿宿主:缓存**下沉**而非丢弃) ·
  `snap_max=4` · `snap_host_mb=3072`(状态腿宿主:旧快照下沉 ⇒ `reuse_face`
  能回查到更早的边界)。
- **末段页对齐**(`Scheduler::prefill_step_chunk` 加 `page` 参数 + 新旋钮
  `[pool] prefill_align_tail`,缺省 true):块边界快照只在**页对齐**处拍 ⇒
  若最后一段跨页,切在**最后一个整页边界**上(尾巴 <1 页另起一小步)。
  否则短 prompt **一个快照都没有**(实测 83 tok prompt 复用 0)。
  ⚠️ **与 L4 尾块合并的交互**(踩过并定谳):生产档 `prefill_tail_merge = 1024`
  会把"从头对齐的长尾段"合并成**尾端不对齐**的一大块 —— **前段尾对齐 ≠ 尾段尾
  对齐**。判据必须**只看尾端**("最后一段 + 尾端不对齐 + 有可落的整页边界");
  曾加过"前段已对齐就不切"的守卫,实测把 1.7k 档复用面从 **1728 压到 512**
  (最后一个快照退到第一块)。代价 = 单块档多付一份 ~250ms chunk 宿主税。

### (5) 实测(27B/`spec-dflash`/单张 3090 Ti;测装 = `testdata/args/prefix-reuse-27b.toml`)

| 场景 | 定档前 | **定档后** |
|---|---|---|
| 短 prompt 同会话(spec 开) | 0(全量重算) | **64/187** ✓ |
| 短 prompt 跨会话(spec 开) | 0 | **160/187** ✓ |
| 长 prompt 跨会话(1.7k) | 1536(88%) | **1728/1732(99.8%)** ✓ |
| 长会话(13.4k)同会话 | 1536(11%)· **12.1s** | **13344(98%)· 2.13s** |
| 长会话(13.4k)跨会话 | 1024(8%)· 12.75s | **13568(100%)· 1.79s** |

`prefix_reuse_probe.py` = **✅ G4 全绿(spec 开口径)**:同会话命中 / 跨会话命中 /
sha 等价 / 非空。TTFT 随命中走(12s → 1.2-1.5s)。

**顺带修 2**(测试基建):`#[cfg(test)]` 的四个辅助(`StageStats::{bytes_per_block,
mib_per_s,line}` / `BAxisProfile::line` / `PartialStats::line` /
`RunningEngine::cfg_prefill_chunk`)在 `--features e2e` 构建下不可见(该构建不带
`cfg(test)`,而场景体要它们)⇒ **引擎 e2e 门在 main 上编译不过**;改
`cfg(any(test, feature = "e2e"))`。
另报一条**测试基建限制**:`testdata/resources.toml` 补齐 27B 资源后,单进程内
多个 27B 场景**累积显存**(实测末个场景:卡可用 3.2 GiB / live 账 61 GB ⇒
预算分解拒);
⇒ 27B 场景需按进程隔离或分组,不在本笔范围。

**顺带修 1**(被容量改动暴露):服务面 e2e 夹具 `testdata/args/e2e-27b.toml` 开了
`[graph] prefill_bucket`,与 flavor 的 `[dispatch] fi_plan_bake` **互斥**(validate
拒启)⇒ **自合并起服务面 e2e 根本起不来**(夹具被 `.gitignore` 吞掉,没人跑过)。
现改显式 `off` 并注明解禁条件。

### (6) 立案(未修;按优先级)

a. ✅ **已结(2026-10-19;本轮)** —— **spec 轮内跨页边界快照**(生成相位逐页拍)。
   详见下文 **(8)**。
b. ✅ **已落(本节的容量定档)**。余:**池/份额/宿主梯的二次定档** ——
   单链 > 池时的复用面仍受 `nb` 约束(`13k 会话 / 16.9k 池` 已到 98%,再长就要加池或分块)。
   既有税**已结(2026-10-19)** —— 见下文 **(9)**。
c. **`[agent] implicit_session` 缺省 false**:pi 不带 `session` ⇒ 每轮新会话
   ⇒ **同会话路径对 pi 不适用**,只有跨会话块级缓存能救。是否改缺省 = 产品裁决
   (并发交错时频繁 reset,慢但不坏)。
d. `testdata/args/e2e-27b.toml` 被 `.gitignore` 的 `testdata/*` 吞掉,而
   `apps/server/tests/e2e.rs` 用 `include_str!` —— **干净检出无法构建 e2e**。
   修:给该夹具(与本专项的测装)加 `.gitignore` 例外。

### (7) **快照税的门(2026-10-19,prefill 提速会战副产品)**

**现象**:prefill 每**页对齐 chunk 末**拍一次 GDN 边界快照(`capture_gdn_snap` 第 ④ 步),
27B 上 = 192 片 `copy_batch`(48 层 × 4 状态 = **152 MiB**),槽满还要 DtoH 下沉。
实测代价 **67~77 ms/块** ⇒ 8k 请求 **≈ 1.0 s ≈ prefill 的 17%**
(`prefill.bnd` 3039 ms → 0 的 A/B:7962 tok 7.0 s → **6.0 s**,+17%)。

**为什么要门它**:快照的唯一消费者是**前缀复用**(`reuse_face` 要求边界快照在场;
`restore_snap` 只在 prefix 命中时调用)。`prefix_cache = false` ⇒ 缓存区份额 0
⇒ 复用面结构上恒 0 ⇒ 快照是**纯浪费**。

**修法**(`crates/engine/src/exec/spec.rs`;**2026-10-19 与下文 (8) 的轮内快照合流**:
函数已拆成 `capture_gdn_snap`(前缀段,Durable)→ `capture_gdn_snap_tail`(带 `tail` +
`durability`),本门与容量门在两处**共用同一条 `snap_on`**):
```rust
let snap_on = self.dev.pool.snap_capacity() > 0 && self.cfg.knobs.prefix_cache;
```
`prefix_cache = true`(生产形)行为**完全不变**(税照付、复用照得:同 prompt 第二发
0.14 s = 50×)。engine lib 53/53 绿。

**⚠️ 基准口径坑**:std 五点三轮的**第 2/3 轮看不到这笔税** —— 同 session 重复同一
prompt ⇒ 链游标已推进 ⇒ `start >= full_blocks` 早退(实测:第 1 轮 1106 → 门修后 1367;
稳态 ~1358 不变)。**该税只在"新内容首轮"付**,即生产真实形态。

**剩余头寸**(留案):生产形(prefix_cache=true)的快照税仍在,候选 = ①下沉 DtoH
异步/与下一块重叠 ②更粗的拍摄粒度(每 K 个边界)③扩设备槽(152 MiB/条,贵)。

### (8) ✅ 已结(2026-10-19):spec 轮内跨页边界快照(§9.1(6)a)

**问题**:`prefill_align_tail` 只管 **prefill 段**;生成相位的页边界从不拍快照
(spec 轮一次提交 `(pos, pos+m]` 多为**跨过**边界而非落在边界 ⇒ 状态已过界,
而捕获图定形 `T=depth+1` 不能靠减 depth 对齐)。⇒ 下一轮的复用面 = **prompt 段
最后一个整页**(实测 187 tok 档只复用 64;501 tok 生成长度 ⇒ 只用到 32/544 = 6%)。

**修法(三步,均已落)**:

1. **fold 族状态叶子改"写池"**(`graph_fleet::plan_fold_family`):旧形是"就地
   在 `spec_snap` 缓冲里重放 + restore 回写池" ⇒ 缓冲被写成 state@fed,**一轮只能折一次**。
   现改为叶子 = **池 GDN 态**(`slots` 输入 = 会话行号)—— 与 **eager 路径同形**
   (少一条"两套规范"旁路),且 `spec_snap` 在轮内**保持 pristine**(= state@pos)
   ⇒ 轮内可**多次**回卷。每轮拷贝次数不增(restore 从尾部调到首部)。
2. **回卷拍快照**(`RunningEngine::spec_fold_to` + `capture_gdn_snap_tail`):
   本轮提交跨过页边界 `b`(`Scheduler::decode_boundary_in`,纯函数,判据 = 
   `(pos, fed_new]` 内最大的 `stride` 倍数页边界)⇒ `restore → fold[b-pos]`
   得到 state@b ⇒ 拍快照 ⇒ `restore → fold[m+1]` 回到 state@fed。
   尾 token:提交在 emit **之前**,`act.out` 还差本轮那几枚 ⇒ `tokens_upto_with`
   接受显式 `tail`(越界仍响亮拒)。**`m == depth` 全接受轮不拍**(该桶无 fold 图;
   代价 = 少一个边界,均值 <16 tok 复用面,已记账)。
3. **快照重要性档**(`owl_store::SnapDurability`;store 定政策):
   | 档 | 来源 | 逐出时 |
   |---|---|---|
   | `Durable` | **前缀段**(prefill chunk 边界 / 行归还快照) | 可下沉宿主(跨会话共享前缀靠它) |
   | `Transient` | **生成段**(逐页边界;spec 轮内 + 裸 decode 步) | **丢** |
   逐出优先级 `rank` 传进 `StatePool::snap_pick_slot`(小者先出,同档 LRU)
   ⇒ 先丢生成段、后动前缀段 ⇒ **链头那条永远最新、永远是设备命中**。
   动机 = 实测:生成段每 32 tok 出一条,**下沉一条 159 MiB ≈ 208 ms**
   (旧档全下沉 = 每 token 数 MiB 级 PCIe + 每 32 tok 一个 200ms 级尖刺)。

**实测(27B/`spec-dflash`/单张 3090 Ti;同二进制 + 单键 A/B)**:

| 项 | `snap_decode_stride = 0`(旧行为) | **= 1(新)** |
|---|---|---|
| `prefix_reuse_probe` T2(187 tok 同会话) | 64/187 | **160/187**(上界 = ⌊165/32⌋·32 = 160 ✓) |
| 512 tok 生成长文本 → 下一轮 cached(572 tok prompt) | **32 / 544 = 5.9%** | **544 / 544 = 100%** |
| 该下一轮墙钟 | 1.39 s | **1.00 s** |
| 本轮生成开销(18 张边界) | — | fold 1.63ms + 拍 1.01ms = **~2.6ms/边界** |
| turn 1 墙钟(512 tok 生成) | 12.05 s | 12.26 s(+1.7%;均在噪声内) |

**验收(全绿)**:`prefix_reuse_probe` G4 ✅ / `lag_probe` 4/4 ✅ / `agent_wire_smoke` 全过 ✅ /
单测 store **81+3+3** · shared **81** · engine **51** · server **36** + ops_whitelist **2/2** ✅ /
**spec 恒等门**(`OWL_E2E_ONLY=spec_identity`)✅ —— 且输出 sha 与**改前基线逐字相同**
(`670671cd97404156` / `565339bc4d33d728`)⇒ fold 改叶子为纯搬移,零数值漂移。

**新键**:`[pool] snap_decode_stride`(缺省 **1** = 每页一拍;0 = 关;N = 每 N 页一节流)。
**新读数**:`spec.bnd.fold1` / `spec.bnd.snap`(轮内边界分相计时)、
`store.snap.drop` / `store.snap.out{,.bytes}`(逐出归向)。
**未做(另账)**:`m == depth` 轮的边界(代价 <16 tok 复用面);
池/份额/宿主梯的二次定档(G2/§9.1(6)b)。

### (9) ✅ 已结(2026-10-19):快照下沉的"冷写缺页税"(§9.1(6)b 立案)

**现象(立案数字)**:8417 tok 冷 prompt ⇒ 16 个 prefill chunk 边界 ⇒ 12 次状态腿快照
下沉(每次 159 MiB=1.91 GB)⇒ `prefill.bnd` avg **71.6ms**(p99 212ms)≈ 该请求
**TTFT 16%**。连发实测:第 3 发宿主梯满 ⇒ 0 次下沉 ⇒ `prefill.bnd` **1.0ms/边界**
⇒ 边际成本 ≈ **107 ms/条**(0.75 s/GB,1.6 GB/s 级)。

**定点取证**(新探针 `crates/backends/cuda/tests/swap_econ_probe.rs`,`--ignored`):

| 实验 | 结果 | 读法 |
|---|---|---|
| 片数扫描 1 / 4 / 24 / 192 片(同 159 MiB) | 293 / 186 / 107 / 106 ms | **片数越多越快** ⇒ 不是逐片开销 |
| host memcpy 到**冷** pageable 目的(159 MiB) | **95 ms = 1.67 GB/s** | ✅ 真凶 = 目的页**首次写缺页**(4 万次 4 KiB) |
| 同缓冲复拷(warm) | 7.5 ms = 21 GB/s | 热了就快 13× |
| 冷写 A/B:`MADV_HUGEPAGE`(2 MiB 页) | **12 ms = 13.6 GB/s** | 缺页数 ÷512 ⇒ 6.6× |

⇒ 100 ms ≈ **95 ms 冷写 + ~10 ms DMA**。根因:`HostBlock::zeroed` 的 `vec![0u8; n]`
= 新 mmap ⇒ 首次写逐页缺页(本机 THP = `[madvise]`,不申请就不给大页)。

**修法(与"宿主不 pin"形态律兼容 —— 不动 pin,只要大页)**
`store::host::HostBlock::zeroed` 分配后对 **2 MiB 对齐内区间** `madvise(MADV_HUGEPAGE)`
(`libc` 入 `owl-store` 依赖;区间不对齐会 EINVAL ⇒ 先对齐再呼)。
**不静默**:提示结果记在块上(`HostBlock::hugepage`),`HostTier::hint_failures`
汇总,引擎在下沉点打 `store.host.hint_fail` 指标 + 点名日志;
单测 `hugepage_hint_is_counted` 钉记账律(环境无关)。

**实测(同机同档,三连发 8424 tok 冷 prompt)**:

| 项 | 修前(4 KiB) | **修后(2 MiB)** |
|---|---|---|
| 第 1 发 `prefill.bnd` | 1181.9 ms(73.9/边界) | **445.8 ms(27.9)** |
| 第 2 发 | 880.8 ms(55.0) | **280.0 ms(17.5)** |
| 第 3 发(宿主满 ⇒ 零下沉) | 16.7 ms(1.0) | 18.7 ms(1.2) |
| 第 1/2 发墙钟 | 8.65 / 8.33 s | **7.89 / 7.67 s(−8.8% / −7.9%)** |
| 单次下沉边际成本 | 107 ms | **38.8 ms**(2.8×) |

**验收**:store 81+3+3 ✅ · e2e `gpu_08b_offload_probe` / `gpu_08b_state_leg_probe` ✅ ·
G4 探针 160/187 + 1728/1732(sha 与基线逐字同)✅ · lag 4/4 ✅ · wire 全过 ✅。
**未做(可选)**:①启动期预热宿主梯(19 槽 × 12 ms ≈ 0.23 s 启动税换请求内 −12 ms/槽)。
②逐项拆解**已做**,见 (10.2) —— 38.8 ms ≈ DMA 12.6 + 宿主拷贝 ~17(THP)+ 链路 ~9;
DMA 已到 PCIe 上限,要再降只剩"宿主内容 pin"(被形态律 1 否)。

### (10) ✅ 已结(2026-10-19):池/份额/宿主梯**二次定档**(§9.1(6)b;G3)

#### (10.1) 定档面事实(定档前必读)

| 键 | 现值 | 事实 |
|---|---|---|
| `pool_tokens` | 16896(528 MiB) | **预算上限是 85251 tok(4.23 GB)**([vram-gov] 行:卡 23.6G、fixed 1.12G、live 17.52G ⇒ budget 4.23G)⇒ 池只用了 1/8。**当前 16k ctx 下不是约束**(单会话 512 页 ≤ 528;实测 13.4k 命中 98%);改 ctx(32k/64k)才需同步扩 |
| `prefix_cache_share` | 已退役 | 2026-10-22 起缓存可占整池 |
| `snap_max` | 4(637 MiB 显存) | 设备槽 = 链头边界的常驻位;透明档逐出优先丢 Transient ⇒ **保留 4** |
| `snap_host_mb` | 3072(20 槽) | **原本是硬天花板**(满即拒 ⇒ 新快照被丢);修政策后 = 有界 LRU ⇒ **保留 3072** |
| `prefill_tail_merge` | 1024 | TTFT 赢(516 tok:651→406 ms);粒度代价见 (9.4)F1 |

#### (10.2) 下沉成本逐项实测(`swap_econ_probe` 扩项)

| 环节 | 耗时(159 MiB/条) | 读法 |
|---|---|---|
| 192 片 Out → **丢弃型**连接器(DMA + pinned staging) | **12.6 ms**(12.6 GB/s) | = PCIe 4.0 x16 实际上限,**不可再降** |
| 192 片 Out → 宿主块(**THP 冷写**) | **29.5 ms** | 宿主拷贝 ≈ 17 ms(THP;4 KiB 时 90 ms) |
| 生产实测单条 | 38.8 ms | 差额 ≈ 链路(块指针/校验/回执)~9 ms |

⇒ 再省只能靠"宿主内容 pin"(律 1 明拦)⇒ **换路线:降"次数"而不是"单价"**(下条)。

#### (10.3) 两个真 bug(G3 测量把他们拓出来了)

**bug①:AGT-12 真值帧被终态静默丢掉**(`apps/server/src/actor.rs`)。
旧序:终态分支先 `pending_cache.remove`,再轮"送校正帧" ⇒ **首 token 后即 EOS 的短 turn**
(实测 "只回答:OK" ⇒ completion=1)下一帧就落进了被删的槽 ⇒ 引擎日志说
`复用面 m=285/285 = 9120` 而客户端 `cached=0` ✗(pi 的 cacheRead 记账 → 误判需压缩)。
修:把送帧段提到终态清理**之前**;终态清理变兵底且**叫名日记**(不再静默)。
验:1-token EOS 形态三连会话 —— 修前会话 2/3 `cached=0`,修后 **9120** ✓。

**bug②:宿主梯从不驱逐**(`crates/store/src/rel.rs::plan_snap_offload`)。
旧行为:宿主满 ⇒ `acquire()` 直接 `Exhausted` ⇒ 引擎回落"丢**新**快照"
⇒ 开机做过几轮会话后,3 GB 宿主梯被旧链边界**永久占满** ⇒ 14k prompt 的 28 个边界里
**18 个被丢**。
修:**满 ⇒ 先用同一份 `EvictPolicy` 逐最冷的 `Tier::Host` 条目**(政策住 store,
与 KV 腿共用一本策略;逐出量经 `snap_host_evicted_bytes` 差分上报 + `store.snap.host_evict_bytes`
指标——store 零日志依赖)。store 门 `snap_leg.rs::③` 从"满即拒"改写为"满⇒逐最冷腾位"(含零副作用/un 断言)。

#### (10.4) 实测(同二进制 + 同"脏"启动状态,前置三连 9.1k 会话污染宿主梯)

| 场景 | 修前 | **修后** |
|---|---|---|
| 14k 冷 prompt(28 边界) | 8 下沉 + **18 丢** | **26 下沉 + 0 丢** |
| 后续新会话:共享 3.5k 前缀 | 2048(57.8%)· 1.60 s | **3072(86.7%)· 0.85 s** |
| 共享 7.0k 前缀 | **2048(29.1%)· 4.79 s** ✗ | **6656(94.6%)· 1.36 s** |
| 共享 10.5k 前缀 | 5632(53.4%)· 4.53 s | **10240(97.2%)· 0.90 s** |
| pi 型同会话历史重发(9.1k) | — | **9120/9148 = 99.7%**、0.64 s(1 分块) |

**F1(粒度上界,已记非 bug)**:跨会话**部分**前缀的复用面 = 最近一个**在场快照**
≤ 共享长度;而快照只在 **chunk 尾**(512,经 L4 合并可到 1024)出现 ⇒ 最多丢 ~1 个
合并块(实测:5.6k 共享拿 4608 = 82%,差 ~1k tok ≈ 1 s/新会话)。
pi 的**历史重发不受影响**(共享段尾 = 上轮生成末页,已由 (8) 的逐页快照覆盖 ✓)。
候选修法(未做):①降 chunk 宿主税(~250 ms/chunk ⇒ 只 32 tok 一块)——prefill-graph M0 的
“命令批/零回执”做完才划算;②导出 FLA chunked 核的**逐 chunk 状态**(核内已有,
导出即可得 64-128 tok 粒度)。

#### (10.5) 定档结论 + 未做

- **不动尺寸**:`snap_max=4` / `snap_host_mb=3072` / `pool_tokens=16896`
  —— 修政策后实测均充分(上表)。
- **改的是政策**(无新旋钮、无需配档):宿主梯**有界 LRU** + 真值帧**次序**。
- 未做:①F1 两条候选;②启动期预热宿主梯;③ctx 升档时池/份额的联动算式(现在写在
  `flavors/*.toml` 注释里:`pool_tokens ≈ ctx/page + 1 页余量`,升级时直接套)。




## §9.2 G5 服务面能力缺口实录(2026-10-19;server/engine 服务层专项)

立案 = 本地台账《agent-服务面-剩余工作项-2026-10-14》 §G 的 **G5** 行:
keep-alive ・`/v1/completions` ・`/tokenize` ・Responses API ・`/debug/agent` ・
undo/rollback ・`reasoning_tokens`。**施工序**(风险从低到高):
① `reasoning_tokens` ② `/debug/agent` ③ `/tokenize` ④ keep-alive(唯一动协议面 ——
连接级 permit / SSE chunked / 空闲超时)。

### (1) ✅ `reasoning_tokens` 分通道计数(2026-10-19)

**契约**见 §3.5。实现两处:
- `apps/server/src/toolcall.rs`:`ChannelSplitter` 加 `reasoning: usize` + 访问器
  `reasoning_steps()`;`push()` **入口**判相位(入口态 = Think ⇒ +1)。调用契约钉在
  方法 doc 上:**一个引擎 token 恰好调一次 `push`**(批量合并多个 token 会低估)。
- `apps/server/src/openai.rs`:`Usage.reasoning`;`Completed` 帧处取
  `splitter.reasoning_steps()` 终值;流式 usage 帧与 `completion_json` **两形态单源**。

**为什么不用 tokenizer 反数**:标记文本(`</think>` 等)被解析器吞掉,反数与会话真值
天然不等;且多一次 CAS 往返不値当 —— 引擎侧“1 token = 1 采样步”,而协议层每 token
调一次 `push`,已是**精确账**。

**边界口径(有意为之)**:`</think>` **标记本体计为思考 token** —— 它由思考段生成;
而“思考与答案同一 token”的病态边界(实机 0.8B 见过:1 token 同时关思考 + 出答案)
按**入口相位**判 ⇒ 计 1。这是“相位归属”而不是“文本归属”;口径稳、不需额外状态。

**真机验证(dev-08b / 3090 Ti@41:00 / 8135;非流式 usage 读数)**:

| 场景 | `completion_tokens` | `reasoning_tokens` |
|---|---|---|
| thinking 关(`enable_thinking=false`) | 1 | **0** |
| thinking 开,长思考 | 80 | **80** |
| thinking 开,单 token 关思考 + 出答案 | 1 | **1** |
| 工具历史 + 思考档(345 tok prompt) | 24 | **24** |

**门**:`agent_wire_smoke.py` **全 PASS**(新增两形态断言:流式 usage 帧与
非流式 usage 都有 `completion_tokens_details.reasoning_tokens`,且 `0 ≤ R ≤ completion`)、
`cargo test -p server --bins` **37/37**(新增
`toolcall::tests::reasoning_steps_are_phased_by_channel` 三档相位)、
`cargo check -p server --tests` 干净。

**未跑** `apps/server/tests/e2e.rs`(27B 黑盒门;新增断言 = 两形态透出 +
`reasoning ≤ completion`)—— 与后续 G5 项合并跑一轮。

### (2) ✅ `/debug/agent` 诊断环(2026-10-19)

**契约**见 §3.6。实现三块:
- `apps/server/src/agent_trace.rs`(**新模块**,486 行):五面事实 struct
  (`ReqFacts`/`PromptFacts`/`UsageFacts`/`OutcomeFacts`/`TurnTrace`)+ 环形仓
  `AgentTrace`(固定深 `TURNS = 8`)+ 施工期账本 `TurnLedger`。
- `apps/server/src/toolcall.rs`:`ChannelSplitter` 加**结构轨迹**
  (`Vec<StructEvent>`,`TRACE_CAP = 512` + `trace_dropped` 记账;`malformed()` 计数)。
- `apps/server/src/main.rs`:模块接入 + **`ConnCtx` 收拢**(serve 参数从 6 个涨到 7 个
  时收口 —— 再逐个下传就要靠注释对位了)+ 三条路由。
- `apps/server/src/openai.rs`:`chat(…, trace)` 开账 → 逐帧记账 → `commit`。

**几条设计选择(为什么这么写)**:
- **不做全局变量**:`main` 建 `Arc<AgentTrace>` → `ConnCtx` → `chat`(同 `/debug/blocks`
  的 face 律:显式依赖)。
- **不设新配置键**:深度固定 8、内存上界 ~256 KiB(截断 + 账号),不值得开档位旋钮
  (且 `crates/shared/src/config.rs` 正被 prefill-parallel 线占用 —— 不添交叉)。
- **不许静默丢**(律 #1):①轨迹超限计 `trace_dropped`(尾行 `…(+N dropped)`);
  ② prompt 截断计 `omitted_bytes`(账面平);③每轮**必留一条** —— `TurnLedger::drop`
  是兑底网(提前返回/`?` 传播也会显形为 `error = 未结算(提前返回;见服务端日志)`)。
- **只记结构不记 delta**:轨迹是排查骨架,不做第二份输出副本(不然内存/隐私两面都输)。
- **失败与拒绝同样留痕**:显式 `led.fail(...)`(Rejected / turn Failed / 协议序错乱 /
  客户端断开 / 引擎通道关闭)⇒ “短 turn 看不到真账”这类 A7 族盲区不再靠猜。

**真机验证(dev-08b / 3090 Ti@41:00 / 8135)**:
| 场景 | 读数 |
|---|---|
| 一轮正常生成后 `GET /debug/agent` | 5 轮留痕(cap 8);末轮 usage/outcome/events 全在场,`ttft_ms=140`/`elapsed_ms=327` |
| 超预算请求(prompt 24012 + 生成 8 > 4096) | **拒绝也留痕**:`error = "提交被拒:submit: prompt 24012 + 生成 8 超预算 4096"` |
| 大 prompt(120069 B) | 截断账面平:`head 8190 + tail 24576 + omitted 87303 = 120069` ✓ |
| `GET /debug/agent/1` / `POST /debug/agent/clear` / `GET /debug/agent/abc` | 1 轮 / `cleared:5` / **400**(不静默回缺省) |

**门**:`agent_wire_smoke.py` **全 PASS**(新增 4 项:有轮次记录 / 含请求事实+usage 真账 /
含结构轨迹 / prompt 快照含本轮用户文本;并断言非法段 400)、
`cargo test -p server --bins` **43/43**(新增 agent_trace 5 条 + 轨迹 1 条)。

**未做**:`e2e.rs`(27B 黑盒门;与后续 G5 项合并跑);预算/排队面的诊断(易做,
暂区——现有 outcome 已能看出异常)。

### (3) ✅ `POST /tokenize` 工具面(2026-10-19)

**契约**见 §3.7。实现四块:
- `crates/models/src/tokenizer.rs`:**`try_encode`(Result 版)**;旧 `encode`
  改为薄封装(保留吞错回退 + 日志)—— 新调用方走显式路。
- `crates/engine/src/running.rs`:**`RunningEngine::tokenize(text, chat_wrap)`**
  (+ 新类型 `TokenizeReply { tokens, max_seq_tokens }`,已从 crate 根导出);
  `chat_wrap` 分支复用 `Tokenizer::chat_wrap`(与裸文本提交**同一律**,不重写)。
- `apps/server/src/actor.rs`:`EngineReq::Tokenize { text, chat_wrap, reply(oneshot) }`
  —— **不落 turn、不碰池/图**;首收与积压两处共同收拢到新 `handle_req`
  (Abort / Tokenize / Chat 三路由,不再在两个地方写两份 if-let)。
- `apps/server/src/openai.rs`:`tokenize_plan`(**纯校验**,可单测)+ `tokenize`
  (oneshot 往返;投递失败/通道关都以 503 显式上抛);`main.rs` 一条路由。

**为什么不做 vLLM 全兼容**:vLLM 的 `add_special_tokens` 默认 true、`messages`
要 chat_template、还会回 `token_strs`。我们只做**能说真话**的子集:
`mode=raw|chat` 两模式显式化,`add_special_tokens=true` **响亮拒**
(而不是静默接受一个不生效的开关)。

**真机验证(dev-08b / 3090 Ti@41:00 / 8135,`agent_wire_smoke.py` 实测)**:

| 项 | 读数 |
|---|---|
| `raw`(“用一句话说明什么是目录。”) | `count 6 == len(tokens)`,`max_model_len 4096` |
| `chat`(同文本,家族包装) | **18** —— 与**同文本走 chat 面的 `prompt_tokens = 18` 逐字对上**(单轮下“模板渲染 ≈ chat_wrap”;两路各自独立算账) |
| 响亮拒 | 空 prompt / `mode=tokens` / `add_special_tokens=true` / 未知字段 → **全 400** |

**门**:`agent_wire_smoke.py` **全 PASS**(新增 6 项:`count==len(tokens)`、
`max_model_len`、`chat > raw`、四类 400);`cargo test -p server --bins` **44/44**
(新增 `tokenize_plan_modes_and_rejects`);`cargo check -p server --all-targets` 干净。

**未做**:`messages` 渲染(候选:把 chat 的渲染参数构造抽成公共面即可复用)、
`token_strs`(逐 token 文本,vLLM 有)、生成期间可查(见 §3.7 单源/延迟说明)。

### (4) ✅ keep-alive + SSE chunked 定界(2026-10-19)

**契约**见 §3.8。实现四块:
- `apps/server/src/http.rs`:请求行版本 + `Connection` 语义解析(`Request.keep_alive` /
  `.http_11`;列表形 `keep-alive, close` 也认);`respond_json`/`respond_text` 去掉
  `Connection: close`(自定界);`respond_sse_head` 改 `Transfer-Encoding: chunked`;
  `sse_data` 加 chunk 帧;新增 `sse_end`(`0\r\n\r\n`)。
- `apps/server/src/main.rs`:**serve = 请求循环**(空闲超时 / 请求数上限 /
  per-request permit / 非 1.1 ⇒ 505 + 关);路由拆到 `handle`(连接生命周期与
  单请求处置解耦)。
- `apps/server/src/openai.rs`:三处 SSE 收口都调 `sse_end`(正常完成 / 错误流 /
  环外 break);流式写入失败仍走 Err ⇒ 连接不重用(帧不可信)。
- `apps/server/tests/keep_alive_probe.py`(新):**裸 socket** 六项探针。

**战役教训(插桩定位,值得记)**:首版实现后连接仍一请求即死,表象是
“客户端复用时报 dead / 服务端读到 EOF”。真因 = `handle` 尾部残留旧时代的
`stream.shutdown().await`(一请求一连接时代的“发 FIN 收尾”)——
**keep-alive 下它把每条连接在首响应后直接杀掉**。定位路径:
① 裸 socket 复现(排除客户端);② `ss -tn` 看到 `FIN-WAIT-2`(服务端主动 FIN);
③ 在 serve 四个返回分支插 `eprintln` ⇒ 命中 `read=0(peer EOF)`;
④ 全文搜 `shutdown(` ⇒ 一行定罪。教训:**协议面改造要把“旧时代的收尾动作”
一起清点**(半关闭、flush、close 都算)。

**真机验证(dev-08b / 3090 Ti@41:00 / 8135)**:

| 项 | 读数 |
|---|---|
| 同连接两请求(JSON) | 两应答均完整,连接不断 |
| SSE | `Transfer-Encoding: chunked` + 终止块;**流后同连接再发 `/health` 仍 200** |
| `Connection: close` | 应答后关连接(尊重) |
| `HTTP/1.0` | **505 + 关** |
| curl 双 URL | “Connection #0 left intact” 后**不再 dead**(修复前必 dead) |

**门**:`keep_alive_probe.py` **6/6**;`agent_wire_smoke.py` 全 PASS(urllib 原生吃
chunked —— 这就是“真客户端形状”的验证);`e2e.rs` 27B **46.4s 全 PASS**
(**需同步改测试侧**:裸 socket SSE 读侧加 `dechunk` + 字节级累积;
否则 chunk 长度行会污染 usage 帧 JSON);单测 **46/46**(新增 keep-alive 判定 +
chunk 帧两项)。

**未做**:真 pi/undici 实测(需起真机客户端);`/debug/perf` 等大文本面的分块
(现仍 Content-Length,够用);HTTP/2 不在路线图。

### (5) ✅ pi 真机验证 + 日志面两修(2026-10-19)

**验证形状**:真 **pi(undici 连接池)** → 27B 生产档(`spec-dflash` + `prefix-reuse-27b.toml`,8137)
多轮 + 工具调用。

| 读数 | 值 |
|---|---|
| 会话 | 4 轮(第 2 轮带 13 tools 的 agent 形状) |
| 传输面 | **`[conn]` 警告 0 / abort·error·fail 0** —— keep-alive + chunked 在真客户端下零异常 |
| 工具 | 第 2 轮 `finish=tool_calls`,解析器轨迹 `tool #0 bash args=83B` |
| 前缀复用 | 第 3 轮 cached **13952/14112 = 98.9%**、第 4 轮 **14208/14229 = 99.85%**;TTFT 12.4s(冷 prefill 13.8k)→ **1.0-1.3s** |
| reasoning | 41 / 59 / 43(均 ≤ completion) |
| `/debug/agent` | 四轮全在场(usage/outcome/events),零 `error` |

**日志面两修(真机暴露,已修并复验)**:
1. `[WARN] [owl-gpu] free: N/M 块归池` **刷屏**(单会话 **49920** 行,把真信号淹了)
   —— 常规路径改 **DEBUG**;只有 `freed ≠ 请求数`(**账目不平** = 真异常)才升 WARN。
   实测(dev-08b):1500 tok 生成下 debug 档 546 行 DEBUG / warn 档 **0** 行 / 账目不平 **0**。
2. `[WARN] [store] 快照下沉失败: … 不可下沉(生成段边界…)`(单会话 9 条)——
   **设计行为被报成故障**:生成段快照(Transient)本就是“逐出即丢”,store 的不变量
   拒它下沉**是对的**,错在引擎“先试下沉再丢”。改:Victim 是 Transient ⇒ **不试下沉**
   (DEBUG 记账);只有 Durable 才走下沉重试。实测:1500 tok ⇒ **47** 条“逐出生成段快照”
   DEBUG(与 1500/32 ≈ 47 个页边界逐一对应)、**0** 条失败 WARN。

**未做(挂账)**:pi 发的 `store` 未知字段(无害,已记日志)。

### (6) ✅ pi 口径定档 64k(2026-10-19;上一节的后续)

**起因**:上节把 pi 的 `contextWindow` 对齐到当时跑的 **16k** 档 —— 但本工作区 pi 会话
(系统提示 + 工具 + AGENTS.md + 历史)远超 16k ⇒ 真机直接“Response was truncated”。
口径错在“拿小档当常态”:16k 是 **AGT-13 的 agent 定档**,不是能力上限。

**处置**:
- 新测装 `testdata/args/pi-27b-64k.toml`(gitignored;基座 = `spec-dflash`,
  只改 `max_seq=65536` + `pool_tokens=66048`,`prompt_dump=false`)——
  启动命令与口径注在文件头;
- pi 侧 `~/.pi/agent/models.json`:**contextWindow 65536 / maxTokens 8192**(备份
  `models.json.bak-agent-1019-pi-align`);
- 实测池:2064 页 = 66048 tok(治理器 cap 85251 ⇒ 未被夹),per_tok 53248B
  (主池 32768 + 草稿池 20480,DFlash2 仍在);**>16k 实测**:16064 tok prompt
  200 OK / prefill ≈1036 t/s / 墙钟 15.5s / **零“预算收缩”告警**。

**pi 真机第二轮(64k 档,7 轮)**:

| 轮 | prompt | cached | TTFT | 结局 |
|---|---|---|---|---|
| #2(冷) | 13857 | **0** | 12.25 s | length(1 tok) |
| #3 | 13857 | **13856 = 100.0%** | **0.42 s** | `tool_calls`(`tool #0 bash args=83B`) |
| #4 | 14112 | 13952 = 98.9% | 0.46 s | stop |
| #5 | 13858 | 12800 = 92.4% | 1.33 s | `tool_calls` |
| #6 | 14103 | 13920 = 98.7% | 0.76 s | stop |
| #7 | 14223 | 14176 = 99.7% | 0.77 s | stop(comp 331) |

**日志面(两修后的效果实证)**:整轮 pi 会话日志 **117 行**(修前同形状会话 50097 行);
`[conn]` 警告 **0** / abort·error·fail **0** / “预算收缩” **0**;仅 6 条
`未知字段 ["store"]`(pi 发的,无害)。

**待办**:若将来要更大,`flavors/awq27b-262k.toml` 是单卡 262144 超卖档
(设备 3 GiB + 宿主 9 GiB、fp8 KV、需虚拟化两扇门),代价 = **无 DFlash2 投机**。
## §9.3 agent workload e2e（2026-10-25）

真实 HTTP/SSE + 真图 + 真工具的执行面回归，入口在 `apps/cli`：

```bash
cargo run -p cli -- --workload core --url http://127.0.0.1:8135 \
  --server-log /tmp/owl-8135-image-evict.log
# 调试单场景:
cargo run -p cli -- --workload image-evict --only image_delete_middle \
  --url http://127.0.0.1:8135 --server-log /tmp/owl-8135-image-evict.log
```

| suite | 内容 |
|---|---|
| `core` | 文本多轮 / 工具闭环 / 视觉 / 图文混合 + GUI-agent 窗口 + 五项图像删除 |
| `gui` | **真实 GUI-agent 负载**：screenshot/click/type/key 工具，每个工具结果附当前截图；客户端永远只保留最新 3 张图 |
| `image-evict` | 只跑图像删除：首图、中间图、最后一张（无图后续）、工具轮后删图、滑动窗口 |
| `image-evict-rebase` | 叠转上限臂（需 `--set pool.image_evict_delta_limit=2`） |

**GUI-agent 场景**：系统提示词声明 `screenshot/click/type/key`；用户只发一次
6 步任务（截图 → 两次 click → type → ENTER → 截图）。**一个用户 turn 内**，
agent loop 可多次调用模型，**一个模型响应可返回多个 function call**（实测
`[click, click]`）；harness 执行全部调用并把每张结果截图回填，用户不再操作。
每个工具结果 = `[text, 当前屏幕截图, text]`（截图带 `owl_evict`），客户端每次
工具结果后只保留最新 3 张图。真机 27B：6 次工具 / 3 次 prune / 3 张图 /
末三轮 cached 88-92% / 终态 yellow；TP2 131k spec-on 的完整 `core` 10/10
通过（fresh server `workload_e2e` 35.00s；**FP8 e4m3 + spec-on**）。
运行条件：测试 server `snap_max≥2`
（3 次删除需要后缀快照），且每次 transcript 带唯一 run tag（避免相同前缀
跨会话共享带图块）。

判据不是模型文风，而是结构性事实：工具名与产物字节、`usage.cached_tokens`、
服务端 info 日志 `[img-evict] 前缀无感删除` / `叠转到顶`。release 构建里
`image_evict.*` counter 被编译期关闭，因此日志是吸收/重整的直接证据。

服务端要求：**分页 KV（f16 或 fp8-e4m3）+ `pool.image_evict=true` +
`pool.prefix_cache=true` + 视觉档**。FP8 生产档现已支持重转，但必须
`pool.image_evict_delta_limit=4`（1..=4；f16 可用 32）。自包含入口：

```bash
tools/run-agent-workload.sh gui
tools/run-agent-workload.sh image-evict
OWL_WORKLOAD_DELTA_LIMIT=2 tools/run-agent-workload.sh image-evict-rebase
```

脚本会先查端口占用（不抢占现有 server），再起隔离测试 server、等 health、
跑 workload，退出时只 kill 自己记录的 PID。

测试门：

```bash
OWL_E2E=1 OWL_SERVER_URL=http://127.0.0.1:8135 \
OWL_SERVER_LOG=/tmp/owl-8135-image-evict.log \
cargo test -p cli --test workload_e2e -- --nocapture
```

`OWL_WORKLOAD_SUITE` 选择 suite（默认 `core`）。图片删除需要稳定 session；
中间图删除曾因重建路径重复补 pad 被调度面拒收，已在
`pad_image_spans` 修复（见 `docs/devlog/memory/prefix-deletion.md` §3.0 补记）。
`gui` 档要求测试 server 至少 `snap_max=2`（3 次删图需要 3 个后缀快照）；
每次运行会加唯一 run tag，避免相同 transcript 跨会话共享带图块。
pi 形态无 session 时依赖 `runtime.implicit_session=true`。多段 text 的边界
由服务端 `normalize_multipart_text` 统一补 `\n`，客户端不要手写分隔符。

## §十 变更记录

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-10-25 | v0.16 | **agent workload e2e 回归面 + FP8 重转**：`core` / `gui` / `image-evict` / `image-evict-rebase`；真 HTTP/真图/真工具，覆盖 **GUI-agent 三图窗口（tool 结果附截图）**、首图/中间图/最后一张/工具轮后删图/滑动窗口 + 叠转重建；新增 `tools/run-agent-workload.sh` 隔离起停;同时修复中间图删除的前缀 pad 重复 —— 见 §9.3 |
| 2026-10-19 | v0.15 | **pi 口径定档 64k**(测装 `pi-27b-64k.toml` + pi `contextWindow 65536/maxTokens 8192`;>16k 实测 16064 tok OK;pi 真机 7 轮:冷 12.25s → cached 100.0%/98.9%/92.4%/98.7%/99.7%,TTFT 0.42-1.33s;日志 117 行/零 conn 警告/零收缩)—— 见 §9.2(6) |
| 2026-10-19 | v0.14 | **G5 真机收口**:pi(undici)真客户端验证——keep-alive/chunked 零 `[conn]` 警告,4 轮含工具调用;前缀复用 cached 98.9%→99.85%,TTFT 12.4s→1.0-1.3s;`/debug/agent` 四轮全在场。**日志面两修**:①`owl-gpu` free 刷屏(49920 行/会话)降 DEBUG,仅账目不平升 WARN;②生成段快照(Transient)不再试下沉(设计行为曾被报成故障 WARN)—— 见 §9.2(5) |
| 2026-10-19 | v0.13 | **G5(4) keep-alive 落地**(请求循环 + 每请求 permit + 空闲 30s + 请求数上限;SSE 改 **chunked 定界**;JSON/文本自带定界;非 1.1 ⇒ 505);战役教训:**旧 `stream.shutdown()` 残留会每响应发 FIN 掐死复用连接**(插桩定位);门 = 裸 socket 探针 6 项 + wire smoke + e2e 27B(46.4s)+ 单测 46/46 —— 见 §3.8 / §9.2(4) |
| 2026-10-19 | v0.12 | **G5(3)`POST /tokenize` 落地**(raw/chat 两模式;单一真源=引擎 tokenizer;`add_special_tokens=true` 等四类响亮拒;`Tokenizer::try_encode` 显式 Result 版);门 = wire smoke 6 项断言 + 单测 44/44;真机 raw 6 / chat 18(与 chat 面 prompt_tokens 对上)—— 见 §3.7 / §9.2(3) |
| 2026-10-19 | v0.11 | **G5(2)`/debug/agent` 诊断环落地**(最近 8 轮:请求事实 / prompt 头尾快照 / usage 真账 / 结局 / 解析器结构轨迹;`TurnLedger` Drop 兑底网;接 `ConnCtx`);门 = wire smoke 4 项断言 + 单测 43/43 + 真机四场景 —— 见 §3.6 / §9.2(2) |
| 2026-10-19 | v0.10 | **G5(1)`reasoning_tokens` 落地**(解析器相位归因;流式/非流式两形态单源;门 = wire smoke 两形态断言 + server 单测 37/37;真机 0.8B 四场景实测)—— 见 §3.5 / §9.2(1) |
| 2026-10-14 | v0.9 | **§7.11:A7/A8 深挖** —— 定谳到 **prefill 末行 logits 的布局相关错读**(布局运气律:所有变量都是扰动而非因;`prefill_split` 由核发射账证伪)⇒ 真凶 = 释放块**设备地址复用**(UAF 类),**未根治**,已落 plain-eval 缓解 + 立案 A13;**A11 spec 臂采样化落地**(verify 出 logits 槽 + 抽-比式接受,分布精确,单测+真机);G4 口径定谳(token 级只在 spec-off);pi 真机进步到"发 `<tool_call>` 未闭合"(残余归 D2) |\n| 2026-10-14 | v0.8 | **第二批交付**:AGT-06(per-turn 采样/stop/seed,引擎侧 `TurnOpts`)/07(reasoning_effort 直通)/09(implicit_session)/10(abort)/11(chunked)/12(cached_tokens 真值含块级命中)/13(定档维持 16k);B1-B4(`[agent]` 配置模块 + e2e agent L1 七段 + `agent_gates.sh`);A3/A4/A5 ✅;D1 ✅(**两层前缀复用 + G4 真值门**:同会话 88% / 跨会话 89%)、D3(agent 域 AL 2.68)、D4(4 并发无并行);**新增 §7.8 A8 = spec 臂 token 级非路径稳定**(与 A7 疑同源,恒等/真值门一律 spec-off 跑);实测数字全部入 §7.9。 |
| 2026-10-14 | v0.7 | 清 A 组引擎债三项:**A1 拒启可见**(四项真机实测)/ **A2 `snap_max=0` 不再 panic**(真机 4/4、零 503)/ **A6 think 别名**(+单测);生产档回归绿(wire 17/17、lag 4/4、pi 全链 1 次成功)。**新增 A7:工具调用遵循度实测 1/9**(模型倾向直接作答+幻觉完成,日志证明非解析丢失)= 现第一优先,台账见本地台账《agent-服务面-剩余工作项-2026-10-14》。 |
| 2026-10-14 | v0.6 | **T7 PASS + §7.7 结案**:病灶 = `rep_penalty` 逐元素施加 ⇒ `1.15^k` 指数累积(短输出无感、长思考必崩),已修(去重 + 默认 1.0,单测 7/7);真机生产形状(froggeric + spec d7 + fp8/XQA)pi↔owl 全链跑通,产物文件落地。**M-A 达成**。另记:本轮另修引擎 3 项(XQA 页表恒等链 × 多会话、配置错误静默 exit、`snap_max=0` panic 立案)。 |
| 2026-10-14 | v0.5 | **§7.6 结案并修复**:定谳更正为 **XQA 页表恒等链**(fp8 只是触发条件 —— XQA 前置为 fp8 池;V11 消歧);修法 = 页表改走图输入槽 + `xqa_pt_of` 会话真链,三处触点;验证 = 生产形状 lag 4/4 + 27B 冒烟 17/17 + XQA 恒等探针;agent 档收回生产形状;影响面(多请求接受率/质量读数需复核)已记账。 |
| 2026-10-14 | v0.4 | **V1–V8 二分定谳:T7 拦路虎 = `kv_fp8`(fp8 KV 池跨请求污染)**;f16 池规避后 `lag_probe` 4/4 + **27B 真机 wire 冒烟 17/17**。服务面侧 M-A 收口;新增 §7.7(模板/模型侧第二阻塞:froggeric 模板下工具提示复读塌陷,上游原版模板可出正确 tool_call)+ 两条连带立案(`snap_max=0` 运行期 panic、f16 池白名单缺口)。可用档 = `testdata/server/agent-27b.toml`。 |
| 2026-10-14 | v0.3.1 | **T7 真机冒烟暴露拦路虎(§7.6)**:27B spec-dflash 档多请求「一轮滞后」污染(0.8B 无 / prompt dump 证明服务端无罪 / lane C、GDN 格、session 均排除);另发现 `depth=0` 首请求 panic(`state.rs:859 池非空`)。**M-A 代码面(P0 四项)全部完成并 26 单测 + 17/17 wire 冒烟通过**;G3 待引擎侧修复。 |
| 2026-10-14 | v0.3 | **T3/T4/T6 完工**:`openai.rs` 全面重写(原样透传 + 结构化校验 + 三通道 SSE + 400 面 + `normalize_tool_calls`)、boot 期装载模板(失败拒启)、`render_prompt` 退役、`agent_wire_smoke.py` **17/17 全绿**;单测 26 全绿。**M-A 只剩 T7(pi 真机冒烟)**。 |
| 2026-10-14 | v0.2.1 | **T1/T2 完工**:`chat_tpl.rs`(9 例)+ `toolcall.rs`(12 例)= 21/21 全绿;G0 金标门入仓(`apps/server/tests/fixtures/chat_tpl/` + `gen_fixtures.py`);新增事实 5-8。代码在图 `feat/agent-serving`(worktree `工作树 agent-serving(旧位置)`)。 |
| 2026-10-14 | v0.2 | **T0(G0)实测 PASS**:minijinja 兼容层定案(trim/lstrip blocks + pycompat + HF `tojson` 语义 + raise_exception 注入);4 用例 + 0.8B 模板逐字节相等;踩坑入契约(`chat_template_kwargs` 展平、`<function=>` 5-token、单 token 标记面、渲染 <1ms);回退方案退役。 |
| 2026-10-14 | v0.1.1 | 名词澄清:`minijinja` = **Rust crate**(PyPI 同名包为 PyO3 绑定,不用);G0 oracle 改用本机 venv 内的 HF `apply_chat_template`(逐字节真值面);`tojson` 走 `json` feature,`raise_exception` 确认两边都需注入。 |
| 2026-10-14 | v0.1 | 立项:pi wire 契约取证(pi-ai v1.0.4)+ 现状对账(15 项)+ REQ-AGT-01..14 + §4.3 渲染对称律 + M-A 任务单 T0~T7。用户裁决:①M-A 最小闭环;②模板路线 = minijinja 直读快照;③与 `prefill-graph` 并行、用另一张 3090 Ti(CUDA ordinal 0)/端口 8137。 |

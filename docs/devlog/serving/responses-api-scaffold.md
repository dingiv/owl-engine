# Responses API 支持架子（未实现）

> 状态：**架子 / 未排期 / 零代码**（2026-10-25）
> 目的：把“以后要支持 OpenAI Responses API”这件事的接口、状态模型、
> 与现有 Chat Completions 面的差异、以及最容易踩的坑先钉在纸上，
> 避免真正开工时重新推一遍。
>
> **本文件不是实现承诺，也不代表任何配置键/模块已经存在。**
> 当前服务端仍然只有：
>
> ```text
> POST /v1/chat/completions
> GET  /v1/models
> GET  /health
> ```
>
> 路由入口见 `apps/server/src/main.rs`，Chat 协议面见
> `apps/server/src/openai.rs`。

---

## §0 一句话

Chat Completions 是**无状态**协议：客户端每轮重发完整 `messages`。
Responses API 的核心增量是**服务端保存 response / conversation items**：

```text
第一轮：POST /v1/responses
        input = 用户消息
        → 模型返回 function_call

第二轮：POST /v1/responses
        previous_response_id = 第一轮 id
        input = [function_call_output]
        → 服务端自动把第一轮历史接回来
```

所以 Responses 支持不是“给现有 handler 换一个 JSON 外壳”，而是新增一层
**response store + item 链 + 续写语义 + 新的事件流**。

---

## §1 范围

### P0：先做可用闭环

- `POST /v1/responses`
- `store=false` 的无状态模式：
  - 客户端每次发完整 `input`；
  - 不支持 `previous_response_id`；
  - 可复用现有 Chat 渲染/引擎路径。
- `store=true` + `previous_response_id`：
  - 服务端保存 response；
  - 下一轮只接收新增 item；
  - 支持 `function_call_output` 续写；
  - 内存存储即可，重启后返回 404/410。
- 非流式 JSON + 流式 SSE。
- 工具调用：
  - `function_call` 输出 item；
  - `function_call_output` 输入 item；
  - 同一 response 多个 tool call；
  - 客户端只发工具结果时，模型仍能看到完整历史。
- `GET /v1/responses/{id}`（读取已存 response）。
- usage 映射：
  - `input_tokens` / `output_tokens` / `total_tokens`；
  - `input_tokens_details.cached_tokens`；
  - `output_tokens_details.reasoning_tokens`（能算则算）。

### P1：工程完备

- 磁盘持久化 / TTL / LRU / 容量上限。
- `conversation` 对象（Conversations API 风格）。
- 分支：同一个 `previous_response_id` 创建多个子 response。
- `DELETE /v1/responses/{id}` / cancel。
- `include`、`metadata`、`truncation`、`parallel_tool_calls` 等参数。
- `instructions` / `tools` / `tool_choice` 的逐请求语义核对。

### 明确不在第一版

- OpenAI 内置工具（web search / file search / code interpreter / MCP）。
- background / async execution。
- 完整 Conversations API 管理面。
- 加密 reasoning item 的跨厂商兼容。
- 多租户鉴权与配额。
- 以 Responses 替代现有 Chat 面。

---

## §2 路由与模块架子

### 2.1 路由

```text
POST   /v1/responses              # 创建 response
GET    /v1/responses/{id}         # 读取 response
DELETE /v1/responses/{id}         # P1：删除存储项（非取消运行）
POST   /v1/responses/{id}/cancel  # P1：取消进行中的 response
```

第一版只注册前两个；未实现的路径必须响亮返回 404/501，不能静默落到
Chat handler。

### 2.2 建议模块拆分

```text
apps/server/src/responses/
  mod.rs        # route 分派 + 对外错误
  wire.rs       # 请求/响应/SSE 的 serde 结构（只做协议形状）
  convert.rs    # Responses items <-> 内部 ChatRequest/messages
  store.rs      # ResponseRecord、链查找、TTL/LRU（纯状态，不做 IO）
  stream.rs     # Responses 事件序列（复用 http.rs 的帧写回）
```

如果第一版想少开文件，可先落 `responses.rs`；但 **wire / convert / store
三种职责不要混在一个大函数里**。后续拆文件不改变语义。

### 2.3 与现有代码的接点

| 现有模块 | 复用方式 |
|---|---|
| `main.rs` | 在 `match (method, path)` 增加 `/v1/responses`；沿用 `ConnCtx` / permit / session gate |
| `http.rs` | 继续做 HTTP 帧解析与写回；不引入 axum |
| `openai.rs` | 可复用 SSE 帧工具、usage 计算、tool-call 解析；**不要复用 Chat 的请求结构当 wire 契约** |
| `chat_tpl.rs` | 继续作为唯一模板真源；Responses 只负责把 items 转成 messages |
| `actor.rs` | 提交同一个 `EngineReq`；Responses 不绕过 actor |
| `session_gate.rs` | 同一条 response 链必须串行；跨链可并行（受现有并发上限约束） |
| `agent_trace.rs` / `debug.rs` | 增加 `response_id` / `previous_response_id` 诊断字段（P1） |

---

## §3 Wire 形状（P0）

### 3.1 创建请求

```json
{
  "model": "qwen3.8-27b",
  "input": "你好",
  "instructions": "你是 ...",
  "tools": [
    {
      "type": "function",
      "name": "screenshot",
      "description": "Capture the screen",
      "parameters": {"type": "object", "properties": {}}
    }
  ],
  "tool_choice": "auto",
  "stream": false,
  "store": true,
  "previous_response_id": "resp_...",
  "max_output_tokens": 2048,
  "temperature": 0.7
}
```

`input` 既可以是字符串，也可以是 item 数组。P0 支持：

```json
[
  {"role": "user", "content": "你好"},
  {
    "type": "function_call_output",
    "call_id": "call_1",
    "output": "25C"
  }
]
```

### 3.2 输入 item

| item | P0 | 说明 |
|---|---|---|
| `message` / `{role, content}` | ✅ | user / assistant / system / developer |
| `function_call` | ✅ | 历史 assistant 工具调用，必须逐字段保留 |
| `function_call_output` | ✅ | 工具结果；`call_id` 必须匹配 |
| `reasoning` | ⚠️ | 先原样存储/回放；Qwen 没有 OpenAI 加密 reasoning |
| `item_reference` | ❌ | P1；不要把引用解析成隐式删除 |
| `input_image` | ⚠️ | 先支持标准 Responses 形状；`owl_evict` 语义见 §6 |

Chat 的 `tool_calls` / `role:"tool"` 与 Responses 的
`function_call` / `function_call_output` 是两套 wire 形状，转换层必须显式。

### 3.3 输出 item

P0 至少：

```json
{
  "type": "message",
  "id": "msg_...",
  "role": "assistant",
  "status": "completed",
  "content": [
    {"type": "output_text", "text": "最终回答", "annotations": []}
  ]
}
```

```json
{
  "type": "function_call",
  "id": "fc_...",
  "call_id": "call_...",
  "name": "screenshot",
  "arguments": "{}",
  "status": "completed"
}
```

注意：

- **`call_id` 与 `id` 不同**；客户端回传工具结果用 `call_id`。
- `arguments` 是 JSON **字符串**，不是对象。
- 工具调用参数必须原样保存；下一轮重新渲染时不能重新序列化出不同字节。
- `output_text` 是 SDK 便利字段；wire 的权威面是 `output` items。

### 3.4 Response 对象

```json
{
  "id": "resp_...",
  "object": "response",
  "created_at": 1760000000,
  "status": "completed",
  "model": "qwen3.8-27b",
  "previous_response_id": null,
  "output": [ ... ],
  "usage": {
    "input_tokens": 128,
    "input_tokens_details": {"cached_tokens": 96},
    "output_tokens": 16,
    "output_tokens_details": {"reasoning_tokens": 0},
    "total_tokens": 144
  },
  "error": null,
  "incomplete_details": null
}
```

状态机建议：

```text
in_progress -> completed
            -> incomplete
            -> failed
            -> cancelled (P1)
```

### 3.5 流式事件

Responses 的 SSE 是**命名事件**，不要照搬 Chat 的 `data: {...}` + `[DONE]`：

```text
event: response.created
data: {"type":"response.created", ...}

event: response.output_item.added
data: {"type":"response.output_item.added", ...}

event: response.output_text.delta
data: {"type":"response.output_text.delta", "delta":"...", "sequence_number":...}

event: response.function_call_arguments.delta
data: {"type":"response.function_call_arguments.delta", ...}

event: response.output_item.done
data: {"type":"response.output_item.done", ...}

event: response.completed
data: {"type":"response.completed", "response":{...}}
```

P0 只实现文本与 function_call 所需事件；每个事件带单调
`sequence_number`。完成判据是 `response.completed` / `response.failed`，
不是 `[DONE]`。具体字段以开工时的官方事件表为准，本文只钉分层。

---

## §4 Response store 与链

### 4.1 Record

```rust
struct ResponseRecord {
    id: String,                 // resp_...，不可猜测
    parent: Option<String>,     // previous_response_id
    conversation: Option<String>,
    created_at: u64,
    model: String,
    template_fingerprint: String,
    input_items: Vec<Item>,     // 本轮新增输入（不是全量）
    output_items: Vec<Item>,    // 模型产生的 items
    usage: Usage,
    status: ResponseStatus,
}
```

**不要只存渲染后的 token 串**：

- 需要支持 `GET /v1/responses/{id}` 返回原始 items；
- 需要支持 fork / 分支；
- 渲染模板或模型变化时要做 fingerprint 校验；
- 图像删除需要 item 级身份，token 串不够。

### 4.2 续写算法

```text
create(input, previous_response_id):
  if previous_response_id:
      parent = store.get(previous_response_id)?     # 404 if missing/expired
      items = parent.chain_items() ++ input          # 服务端重建
  else:
      items = input

  messages = convert_to_chat_messages(items)         # 单一转换入口
  prompt = render(messages)
  result = engine.submit(prompt)

  record = ResponseRecord(parent, input, result.output)
  store.put(record)
  return record
```

链的读取方向是**父到子**；存储可以只存 parent 指针 + 每轮 items，
不要每轮复制整个祖先数组（长链会平方膨胀）。P1 可做缓存快照。

### 4.3 分支

`previous_response_id` 天然允许一个父节点有多个子节点。P0 先支持
**读分支**，但要明确：

- 同一父节点 fork 后，两条链不能共享同一个可变引擎会话状态；
- 现有 `AgentSession` 的 KV/GDN 是单活链语义；
- R1 可以限制“同一条链只允许一个活跃 leaf”，分支创建新链并接受一次
  冷启动/重建；
- 真正的多分支 KV 隔离需要内容寻址前缀缓存 + 快照克隆，另立设计。

### 4.4 `store=false`

- 不写 store；
- 请求必须带完整 `input`；
- `previous_response_id` 与 `store=false` 同时出现 ⇒ 400（明确拒，
  不静默忽略）；
- `GET /v1/responses/{id}` 对未存储的 response ⇒ 404；
- 这是最接近现有 Chat 语义的模式，也是图像删除最容易支持的模式。

### 4.5 生命周期

建议配置（**尚未加入 `[responses]`，只是草案**）：

```toml
[responses]
enabled = false
store = true
max_records = 256
ttl_secs = 86400
max_chain_depth = 64
# store_dir = "..."   # P1 磁盘持久化才需要
```

- `enabled=false` 时路由不存在/404；
- 容量满 ⇒ 按 LRU 驱逐最老 **叶子**；驱逐父节点必须连带处理子链；
- 过期 ⇒ 404/410，错误体必须说明 `previous_response_id` 不存在；
- 重启后内存记录丢失是 P0 可接受行为，但必须在文档/错误面写清；
- 不做无界全局 map；store 必须是显式 struct，不引全局变量。

---

## §5 与引擎/会话的映射

### 5.1 复用 Chat 内部路径

Responses 的 `input` items 先转成现有内部 `ChatRequest` / messages，
再走同一个 `EngineReq`：

```text
Responses items
  -> convert::to_chat_messages
  -> ChatRequest / TurnOpts
  -> EngineReq
  -> EngineEvent
  -> convert::to_response_items
```

不要在 `responses` 里复制一套模板渲染、工具解析、采样参数处理。

### 5.2 工具调用

- 第一轮模型返回 `function_call`，保存 `call_id`、`name`、`arguments`；
- 客户端执行后只发 `function_call_output`；
- 服务端重建时，历史里的 `function_call` 必须与当轮返回逐字节一致；
- 多工具调用按原顺序保存，不能按名字排序；
- `tool_choice` / `parallel_tool_calls` 的语义按官方逐项核对，P0 可先
  只支持 `auto`。

### 5.3 Session / 前缀缓存

`previous_response_id` 只解决“消息历史”，不自动等于“引擎会话”：

- 一条 response 链应尽量映射到稳定的引擎 session，以复用 KV/GDN；
- 但分支会让同一祖先对应多个活跃状态；
- P0 建议：**response 链 -> 一个显式 session 映射；分支必须新会话**；
- 若映射失败，退回内容寻址前缀缓存，正确性优先；
- 不要让所有无 session 的 Responses 请求落现有全局 implicit session 0；
- 同链请求必须串行（复用 `session_gate` 思路）。

### 5.4 usage

| Responses | 内部来源 |
|---|---|
| `input_tokens` | prompt tokens（含 cached） |
| `input_tokens_details.cached_tokens` | 现有 cached 真账 |
| `output_tokens` | completion tokens |
| `output_tokens_details.reasoning_tokens` | 现有 reasoning 归因 |
| `total_tokens` | 两者之和 |

### 5.5 reasoning

Qwen 的 thinking 不是 OpenAI 的加密 reasoning item。P0 建议：

- 内部 `reasoning_content` 映射为 Responses 的 reasoning 摘要/文本时，
  先按官方 item 形状存储；
- 不承诺跨厂商 `encrypted_content`；
- 不把 reasoning 文本混进最终 `output_text`；
- 没有 reasoning 时返回空数组/省略，不伪造。

---

## §6 图像与无感删除：必须先裁决

这是 Responses API 与现有 Chat 面**最大的语义冲突**。

### 6.1 Chat 面为什么能删

Chat 是无状态：客户端每轮发全量 `messages`。客户端把旧图的
`image_url` part 拿掉，下一次请求的全量历史就少了一段；引擎用
`old prompt vs new prompt` 识别删除并吸收。

### 6.2 Responses 面为什么不能照搬

`store=true` 时，历史在服务端：

```text
第一轮 response 里已经有 image item
第二轮只发 function_call_output
```

即使客户端“这一轮没提那张图”，服务端历史里那张图仍然存在。
**“不发”不等于“删除”。**

### 6.3 可选方案

| 方案 | 语义 | 代价 |
|---|---|---|
| A. 只支持 `store=false` 的图像删除 | 客户端发全量 items，少了 image item | 与 Chat 同构；但失去 `previous_response_id` 的省传输 |
| B. 增加显式 item 删除 | `DELETE /conversations/{id}/items/{item_id}` 或等价物 | 需要新 API、引用计数、快照重挂 |
| C. fork 一条裁剪后的新链 | 从旧链复制 items，去掉图，创建新 response | 历史身份断，需一次冷启动 |
| D. Responses 不支持 `owl_evict` | 只在 Chat 面承诺无感删除 | 最安全，但功能不完整 |

### 6.4 建议默认

P0 先按 **D + A**：

- `store=true` 的 Responses 链**不承诺** `owl_evict` 无感删除；
- 客户端在 Responses 里带 `owl_evict` 时，要么响亮 400，要么明确返回
  “该模式不支持无感删除”，不能静默假装吸收；
- 只有 `store=false` + 全量 input 的模式，才复用 Chat 的删除识别；
- B/C 作为 P1 单独立项，不在协议适配里偷偷实现。

否则最容易出现的 bug 是：客户端以为删了图，服务端历史仍有图，模型继续
看见旧截图，而 `cached_tokens` 看起来还正常——静默错语义。

---

## §7 错误面

统一 OpenAI 风格：

```json
{
  "error": {
    "message": "...",
    "type": "invalid_request_error",
    "param": "previous_response_id",
    "code": "previous_response_not_found"
  }
}
```

| 情形 | HTTP | code |
|---|---:|---|
| JSON/字段非法 | 400 | `invalid_request_error` |
| `previous_response_id` 不存在/过期 | 404 | `previous_response_not_found` |
| `store=false` + `previous_response_id` | 400 | `invalid_request_error` |
| 同链并发冲突 | 409 | `response_busy` |
| 容量/并发满 | 429 | `rate_limit_exceeded` / `server_busy` |
| 未实现的路由 | 404/501 | `not_implemented` |
| 引擎失败 | 500 | `engine_error` |

未知字段策略要显式：Responses 面建议 `deny_unknown_fields`（与
`/tokenize` 一致），不要走 Chat 面“收下再 warning”的宽松路径。

---

## §8 测试门（未来开工时）

### 协议

- 非流式文本 response 形状；
- 流式事件顺序、`sequence_number`、`response.completed`；
- `GET /v1/responses/{id}` 与 404；
- `store=false` 全量 input；
- `store=false` + `previous_response_id` ⇒ 400；
- 未知字段/非法 `call_id` ⇒ 400。

### 有状态链

- 第一轮 `function_call`；第二轮只发 `function_call_output`；模型答对；
- 三轮链，历史逐轮增长；
- 同一父节点 fork 两个子链，互不污染；
- 重启后内存 store 丢失 ⇒ 明确 404（P0）；
- TTL/LRU 驱逐叶子，不误删父链。

### 工具/图片

- 单 response 多个 function calls；
- 工具结果顺序与 `call_id` 对齐；
- `tools` / `instructions` 是否随链继承，按官方语义核对；
- Responses `store=false` 的图像删除与 Chat 面行为一致；
- `store=true` 的 `owl_evict` 按 §6.4 明确拒/不支持，不得静默。

### 引擎

- 同链串行；
- 跨链并发不共享可变会话；
- cached_tokens / usage 与 Chat 面同口径；
- 分支/回退不产生 NaN、错值或跨链 KV 污染。

---

## §9 开工前必须拍板的问题

1. **图像删除**：§6 选 A/B/C/D 中哪条？`store=true` 是否允许 `owl_evict`？
2. **分支**：P0 是否只允许单活跃 leaf？分支是否强制新 session？
3. **持久化**：P0 内存够不够？TTL 默认多少？重启后 404 是否可接受？
4. **reasoning**：Qwen thinking 映射为哪种 item？是否保存 encrypted content？
5. **参数继承**：`tools` / `instructions` / `tool_choice` 是否每次重发？
6. **streaming 完整性**：P0 是否只实现文本 + function_call 事件？
7. **Conversations API**：是否与 `previous_response_id` 同批，还是 P1？
8. **兼容目标**：pi 目前是否消费 Responses？还是只为未来客户端预留？

---

## §10 参考

- OpenAI Conversation state:
  <https://developers.openai.com/api/docs/guides/conversation-state>
- OpenAI Function calling（Responses 模式）:
  <https://developers.openai.com/api/docs/guides/function-calling?api-mode=responses>
- OpenAI Responses API reference:
  <https://developers.openai.com/api/reference/resources/responses/methods/create>
- Gemini Interactions API（有状态 / `previous_interaction_id`）:
  <https://ai.google.dev/gemini-api/docs/interactions-overview>
- Gemini Function calling（`store=false` 时必须发全量历史）:
  <https://ai.google.dev/gemini-api/docs/function-calling>
- Anthropic Messages API（stateless，全量历史）:
  <https://platform.claude.com/docs/en/build-with-claude/working-with-messages>

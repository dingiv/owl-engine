# Chat Completions + 图片无感删除（TypeScript）

这是一份**请求侧组件**示例，不是完整 agent，也不是 GUI 自动化实现。

它只负责 agent 与 owl 服务端之间的 wire 契约：

- 每轮重发**完整 `messages`**；
- 图片 part 从第一次发送就带 `owl_evict: true`；
- 删除旧图时**只移除 image part**，保留文字、tool_call_id 和消息顺序；
- 工具结果使用稳定的文本边界；
- 自动维护“只看最新 3 张图”的滑动窗口；
- 调用 `POST /v1/chat/completions`。

工具执行、GUI 操作、模型循环都不在这里。真实 agent 只需要在拿到
`tool_calls` 后执行工具，再调用 `addToolResult()`。

---

## 1. 文件

```text
examples/chat-completions-image-evict/
  image-evict-chat.ts   # 可复用的请求组件
  demo.ts               # 确定性 wire 形状演示（不执行工具）
  README.md             # 本文
```

运行：

```bash
cd examples/chat-completions-image-evict

# 只打印请求体，不访问服务端
node demo.ts

# 真发两轮：先 4 图，再同 session 删到 3 图
node demo.ts --send

# 指定服务端
node demo.ts --base-url=http://127.0.0.1:8135 --send
```

`--send` 的第一轮故意用 `complete({prune:false})` 发送未裁剪的 4 图历史；
这样才能让第二轮成为服务端眼里的真实删除。生产代码使用默认的
`completeAndRecord()` 即可，不要关掉自动 prune。

Node 22.6+ 可直接运行 `.ts`（类型擦除）；也可以：

```bash
npm install
npm run typecheck
npx tsx demo.ts
```

---

## 2. 最小接入

```ts
import {
  ImageEvictChat,
  type ChatTool,
} from "./image-evict-chat.ts";

const tools: ChatTool[] = [
  {
    type: "function",
    function: {
      name: "screenshot",
      description: "Capture the current screen.",
      parameters: { type: "object", properties: {} },
    },
  },
];

const chat = new ImageEvictChat({
  baseUrl: "http://127.0.0.1:8134",
  model: "qwen3.8-27b",
  session: "my-gui-agent",     // 可选；同一对话保持稳定
  system: "You are a GUI agent.",
  imageLimit: 3,
  maxTokens: 2048,
  tools,
});

chat.addUser("Open the settings dialog and tell me what you see.");
```

### 第一轮：模型返回工具调用

```ts
const step = await chat.completeAndRecord({ tools });

for (const call of step.message.tool_calls ?? []) {
  // 这是你自己的工具执行；组件不关心它是不是 GUI。
  const result = await executeMyTool(call);

  chat.addToolResult(
    call.id,
    result.text,
    result.screenshotDataUrl,
  );
}
```

### 下一轮：全量历史再次发送

```ts
const next = await chat.completeAndRecord({ tools });

console.log(next.message.content);
console.log(
  "cached:",
  next.usage?.prompt_tokens_details?.cached_tokens,
  "/",
  next.usage?.prompt_tokens,
);
```

注意：**不是**只发最后一条 tool result。`completeAndRecord()` 每次都会把
完整 `messages` 发出去，服务端只 prefill 新增的尾巴。

---

## 3. 图片标记

图片 part 必须是：

```json
{
  "type": "image_url",
  "image_url": {
    "url": "data:image/png;base64,..."
  },
  "owl_evict": true
}
```

约束：

- `owl_evict` 与 `image_url` **同级**，不要塞进 `image_url` 对象；
- 只支持 `data:image/...;base64`（owl 当前不接受 http 图片 URL）；
- 一张图**第一次进入历史时**就要标记；
- 图片还在历史里时，标记必须一直保持 `true`；
- 不要等要删的那一轮才补标记：页对齐 pad 在首次发送时就定了，后补会破坏前缀身份；
- 不是所有图都必须标记；只有“将来可能被 prune”的图需要。永久保留的图标记会多付 pad 开销。

如果你用 OpenAI SDK 而不是本示例的 `fetch`，注意有些 SDK 的类型层会丢掉
未知字段；必须确认 `owl_evict` 原样出现在 wire JSON 里。必要时使用
`extra_body` / 自定义 content part / 直接构造 JSON。这是 owl 扩展字段，
不是 OpenAI 标准字段。

---

## 4. 删除：只删 image part

三图窗口的推荐策略：

```ts
// 新的工具结果已经 append；组件在下一次 complete() 前自动执行：
while (chat.imageCount > 3) {
  chat.pruneOldestImage();
}
```

`pruneOldestImage()` 的实际动作：

```text
找到最老的、带 owl_evict:true 的 image part
只 splice 掉这个 part
保留同一条消息里的所有 text part
保留 tool_call_id / assistant tool_calls / 消息顺序
```

例如工具结果原来是：

```json
[
  {"type": "text", "text": "Step 3: clicked.\n"},
  {"type": "image_url", "image_url": {"url": "data:..."}, "owl_evict": true},
  {"type": "text", "text": "End of tool result."}
]
```

删除后：

```json
[
  {"type": "text", "text": "Step 3: clicked.\n"},
  {"type": "text", "text": "End of tool result."}
]
```

**不要**：

- 删除整条 `role:"tool"` 消息；
- 删除 `assistant.tool_calls`；
- 把图片换成占位图 / 空 URL / `"[image deleted]"`；
- 改写图片前后的文本；
- 合并或重排消息；
- 一次请求删除多组不连续图片（可能退回全量重算）。

一次只删一组连续图片，最稳。

---

## 5. 文本边界

### user 消息

服务端会把 `role:"user"` 的多段 content 的 text part 自动补尾随 `\n`。
组件仍然采用：

```json
[
  {"type": "text", "text": "用户文本\n"},
  {"type": "image_url", "...": "...", "owl_evict": true},
  {"type": "text", "text": "Screenshot 1.\n"}
]
```

让删除点前后都有稳定文本。

### tool 消息

服务端**不会**自动规范化 `role:"tool"` 的 content 数组。组件固定生成：

```json
[
  {"type": "text", "text": "Step 3: clicked.\n"},
  {"type": "image_url", "...": "...", "owl_evict": true},
  {"type": "text", "text": "End of tool result."}
]
```

这能避免模板 `trim` 吃掉删除点前的换行，也保证删图后仍有文本后缀。

---

## 6. 会话

删除吸收依赖同一条 `AgentSession` 的 `img_spans/pads` 账本。

- 客户端能发 `session` 就发，同一对话保持不变；
- 不发 `session` 时，服务端需要：

```toml
[runtime]
implicit_session = true
```

当前 8134 / 8135 生产档已开启。

但隐式会话是全局会话 0；多个 GUI 会话不要无 session 交错发送。
多会话客户端必须显式带 `session`。

---

## 7. 服务端前提

客户端标记只是一半，服务端必须满足：

```toml
[pool]
image_evict = true
prefix_cache = true
image_evict_delta_limit = 4   # FP8；f16 可用 32
```

以及：

- 视觉档开启；
- 分页 KV（f16 或 fp8-e4m3）；
- 建议 `snap_max >= 2`（三图窗口连续 prune 需要后缀快照）。

当前 `qwen38_27b-3090.toml` 与 `qwen38_27b-3080_tp2.toml` 均已满足。
FP8 下连续删除到 `delta_limit=4` 后，下一轮会做一次整段重建，这是预期
行为；之后继续吸收。

---

## 8. 怎么确认删除被吸收

优先看 `usage`：

```ts
const cached = usage?.prompt_tokens_details?.cached_tokens ?? 0;
const prompt = usage?.prompt_tokens ?? 0;
```

删除旧图的轮次，理想情况是：

```text
cached_tokens 仍然很高
prompt_tokens 只增加新尾巴
```

服务端 info 日志会出现：

```text
[img-evict] 前缀无感删除
```

如果 `cached_tokens` 掉到 0，检查：

1. 图片首次发送时有没有 `owl_evict: true`；
2. 删除时是不是只移除了 image part；
3. 图片前后的文本是否被改写；
4. 是否换了 `session` / 模型 / 模板；
5. 是否一次删了多组不连续图片；
6. 服务端 `image_evict` / `prefix_cache` 是否真的开着。

release 构建里 `image_evict.*` counter 会被编译期关闭，日志才是稳定的
吸收证据；`cached_tokens` 是独立佐证。

---

## 9. 流式说明

本组件故意使用非流式请求，让图片删除/历史窗口逻辑保持清晰。
流式客户端只需要替换 `complete()` 的响应读取层；**历史管理、标记、
删除、全量重发逻辑完全一样**。

不要在流式增量里把 assistant 文本重新拼成一条新历史；仍然以最终
assistant message / tool_calls 为单位落历史。

---

## 10. 相关文档

- `docs/devlog/serving/agent-surface.md` §9.3 / §9.4
- `docs/devlog/memory/prefix-deletion.md` §3.0 / §3.2bis
- `docs/devlog/serving/responses-api-scaffold.md`（未来有状态 API 的架子）

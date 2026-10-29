# Frontend handoff —— 先减工具，不等 grammar

> 目的：在 constrained grammar 落地前，先用**减少工具暴露面**降低模型格式漂移与未知工具概率。
> 这是前端/业务层改动，不改 owl 引擎。

---

## 1. 目标

| 场景 | 现在 | 目标 |
|---|---:|---:|
| Pi coding agent | 32 tools | 8–12 个高频工具 |
| GUI agent | 32 或 4 tools | 只留 4–6 个 GUI 工具 |
| system prompt 里的 `<tools>` 文本列表 | 与 OpenAI `tools` 重复 | 删除，只保留一份 |

预期收益：

- prompt 更短，模板里的 Qwen 格式指令权重更高；
- 工具选择空间更小，`computer_use` 这类自造工具概率下降；
- grammar 未来编译的 tool 枚举更小、更快；
- 不改变引擎，不引入 decode 成本。

---

## 2. 保留哪些工具

### coding agent（建议 8–12 个）

```text
read
bash
edit
write
grep
find
ls
```

按需再加：

```text
send_message / ask / list_contacts   # 只有多 agent 协作才加
screenshot / click / type / key      # 只有 GUI 任务才加
serp_web_search / serp_fetch_page    # 只有调研任务才加
notebook / schedule                  # 只有长期会话/定时任务才加
```

### GUI agent（建议 4–6 个）

```text
screenshot
click
type
key
scroll
wait
```

不要同时暴露：

```text
read / bash / edit / write / grep / find / ls
send_message / ask / list_contacts / check_status
serp_web_search / serp_fetch_page / knowledge_base / video_collect
```

GUI 任务需要读文件时，由 GUI 工具链或后端服务完成，不要把 coding 工具全塞给 GUI 模型。

---

## 3. system prompt 修改

### 删除重复的 `<tools>` 文本列表

现在 prompt 里有两份工具声明：

```text
system: <tools> - read ... - bash ... </tools>
OpenAI request: tools: [ ... 32 个 JSON schema ... ]
```

模型会看到两套描述，格式指令被稀释。  
只保留 OpenAI `tools` 字段；system prompt 里不要再手写 `<tools>` 列表。

### 末尾加一条短硬规则

放在 system prompt **最后**，不要放中间：

```text
工具调用必须使用如下格式，禁止使用其它格式：

<tool_call>
<function=工具名>
<parameter=参数名>
参数值
</parameter>
</function>
</tool_call>

不要使用 <function_calls> / <invoke> / 裸 <command> / key="value" 等格式。
```

不要写长示例；一条规则 + 一个最小模板即可。

---

## 4. 验收

前端改完后，用同一组请求对比：

| 指标 | 采集方式 |
|---|---|
| tools 数量 | 请求 JSON |
| system prompt bytes | 请求 JSON |
| 首轮工具调用率 | `/debug/agent` 的 `outcome.tools_called` |
| 格式错误率 | 服务日志 `工具块畸形` / `repair` |
| 未知工具率 | `tool_call_unknown_tool` |
| TTFT / decode | 服务日志指标 |

目标：

```text
GUI 首轮 screenshot 调用率 ≥ 95%
格式错误 / 未知工具 ≤ 1%
普通聊天行为不变
```

---

## 5. 与 grammar 的关系

```text
前端减工具：降低错误率（软）
repair 层：接住近失（中）
grammar：结构硬保证（硬，后续项目）
```

先做前端减工具；如果错误率已经降到可接受，grammar 可以只做成 opt-in，不必全局常开。

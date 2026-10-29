# Constrained tool grammar（原型设计）

> 状态：**设计冻结，未实现**（2026-10-28）
> 定位：owl 服务面「工具调用格式硬保证」独立子项目
> 关联：`agent-surface.md` §7.3、`toolcall.rs` 近失修复层

## 这个项目解决什么

模型在长 system prompt / 多 tools / GUI agent 场景下，会概率性输出非模板要求的工具格式：

```text
<tool_call>
<function=bash>
<command>              # 少了 parameter=
ls ...
</parameter>
</function>
</tool_call>
```

或：

```text
<function_calls>
<invoke name="screenshot">
</invoke>
</function_calls>
```

现有服务层已经做了两件事：

1. 解析器兼容 Anthropic wrapper / 混合参数写法；
2. 对常见近失格式做**保守修复**，修不好才 502。

但修复层只能“事后救”，不能保证模型一定按 Qwen 格式生成。  
本项目要在**解码层**加一层结构约束（constrained decoding / grammar），把格式保证从“概率”提升到“结构确定”。

## 文档

- [`design.md`](design.md) —— 原型设计：目标、裁决、架构、接缝、性能、测试
- [`roadmap.md`](roadmap.md) —— 分阶段施工单、工时、验收门
- [`frontend-handoff.md`](frontend-handoff.md) —— 当前先做的前端减工具清单与验收

## 当前裁决（先记住这五条）

1. **不依赖客户端配置**：普通 agent 不会发 `tool_choice=required` / `strict`，服务端按 `tools` 是否存在决定策略。
2. **默认 `auto`，不是 `required`**：`auto` 允许文本回答，只在模型决定调工具时约束格式；`required` 只给 GUI/显式业务。
3. **grammar 只约束结构、工具名、参数名；参数值先保持自由文本**：类型/required 仍由现有 `ToolSchemas` 校验兜底，避免第一版把 JSON Schema 全量塞进 grammar。
4. **生产路径必须掩码在 top-32/argmax 之前**：host 采样后检查会导致合法 token 不在候选集，spec 收益崩掉。
5. **grammar 引擎优先 `llguidance`**：workspace `Cargo.toml` 已声明，纯 Rust；xgrammar 只作为对照/备选。

## 并行缓解（现在先做）

在 grammar 落地前，前端先减少工具暴露面：

- GUI 任务只暴露 GUI 工具（screenshot/click/type/key/scroll/wait），不要带 read/bash/edit/团队协作工具；
- coding 任务把 32 tools 压到 8–12 个高频工具；
- 去掉 system prompt 里重复的 `<tools>` 文本列表，只保留 OpenAI `tools` 字段；
- 末尾加一条短硬规则：工具调用必须使用 `<tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call>`。

这条线与 grammar 不冲突：前端减工具先降低错误率，grammar 再做结构硬保证。

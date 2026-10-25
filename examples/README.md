# owl client examples

面向客户端 agent 的可运行示例。每个目录自包含，不依赖 owl 仓库内的
Rust 构建。

| 示例 | 内容 |
|---|---|
| [`chat-completions-image-evict/`](chat-completions-image-evict/) | TypeScript Chat Completions 请求组件：全量历史、`owl_evict` 标记、工具结果截图、三图窗口无感删除 |

示例只演示**客户端如何发请求**，不实现完整 agent 循环或 GUI 自动化。

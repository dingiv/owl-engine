# server 分层（2026-11-17）

`apps/server` 是 owl 引擎的 **HTTP/OpenAI 适配层**，不是推理实现。
分层目标：协议（view）/ 业务编排（service）/ 后端端口与进程内状态（repo）
单向依赖；`main.rs` 只做组合根。

## 目录

```
src/
├── main.rs              # 组合根：CLI → config → EngineService::boot → accept → shutdown
├── view/                # 视图层：HTTP/1.1 帧、连接生命周期、路由、JSON/SSE 写回
│   ├── http.rs          #   帧解析 / 应答 / SSE chunked
│   ├── router.rs        #   ConnCtx + keep-alive 循环 + 路由表 + permit 策略
│   └── debug.rs         #   /debug/* 的 JSON 投影（纯读，来自 repo/engine）
├── service/             # 服务层：请求级业务编排
│   ├── chat.rs          #   /v1/chat/completions + /tokenize（协议 DTO 仍暂住本文件）
│   └── load.rs          #   /debug/load 内源灌流
├── repo/                # 持久层 / 后端端口
│   ├── engine.rs        #   EngineRepo：EngineClient 的唯一门面
│   ├── trace.rs         #   /debug/agent 环形诊断账（进程内持久）
│   └── vram.rs          #   互斥 VRAM 账（/debug/vram 的账本投影）
├── domain/              # 领域：OpenAI DTO、chat 模板、三通道流解析、视觉预处理
│   ├── openai.rs        #   ChatRequest / JSON 构造 / Usage / Agg / tokenize 校验
│   ├── chat_tpl.rs
│   ├── toolcall.rs
│   ├── repeat_guard.rs
│   └── vision.rs
└── infra/               # 基础设施：boot 配置
    └── config.rs
```

## 依赖方向

```
view ──► service ──► repo ──► owl_engine / owl_store
  │          │
  └──────────┴──► domain / infra（纯数据/配置）
```

- **view 不直接调 `owl_engine`**：只经 `repo::engine::EngineRepo`。
- **service 不解析 HTTP 帧**：输入是 body/DTO，输出是事件/应答；SSE 帧写回
  由 view 的 `http` 提供。
- **repo 不依赖 view**：只持有 `EngineClient`、诊断环、VRAM 账。
- **domain 不依赖 view/repo**：模板、协议 DTO、流式解析、图像预处理。

## 已知接缝（下一步）

`service/chat.rs` 目前仍把 **业务编排 + SSE 写回**放在一个文件里（历史原因：
SSE 与 turn 取消/背压强耦合）。DTO/JSON 已下沉 `domain/openai.rs`；下一步：

1. `service/chat.rs`：只留 turn 编排（提交/取消/聚合/背压）；
2. `view/chat.rs`：SSE 帧写回与 `TcpStream` 断流探测；
3. `domain/openai.rs` 的 `dump_request`（原始请求落盘）下沉 `service`/`infra`。

拆分时保持 `view/router.rs` 的调用面不变（`service::chat::chat` / `tokenize`）。

## 旧路径映射（历史文档用）

| 旧 | 新 |
|---|---|
| `src/openai.rs` | `src/service/chat.rs` |
| `src/http.rs` | `src/view/http.rs` |
| `src/debug.rs` | `src/view/debug.rs` |
| `src/load.rs` | `src/service/load.rs` |
| `src/config.rs` | `src/infra/config.rs` |
| `src/agent_trace.rs` | `src/repo/trace.rs` |
| `src/vram_ledger.rs` | `src/repo/vram.rs` |
| `src/chat_tpl.rs` | `src/domain/chat_tpl.rs` |
| `src/toolcall.rs` | `src/domain/toolcall.rs` |
| `src/repeat_guard.rs` | `src/domain/repeat_guard.rs` |
| `src/vision_processor.rs` | `src/domain/vision.rs` |

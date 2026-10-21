# chat_tpl 金标(HF oracle 逐字节面)

> 用途:`apps/server/src/chat_tpl.rs` 渲染器的 **G0 验收门**
> (设计:`docs/arch/agent-服务面-设计与施工.md` §5 G0 / §7.1)。

## 金标是什么

`*.txt` = **HF transformers** 用模型快照自带 `chat_template.jinja`
(`ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, …)` +
HF 版 `tojson` + pycompat 语义)渲染 `*.json` 的**逐字节输出**。
Rust 渲染器必须与之**逐字节相等** —— 这是"提示词面 = 上游真值"的判据。

| 文件 | 内容 |
|---|---|
| `ctx1_tools` | tools 块 + 单轮 user(基础路径) |
| `ctx2_tools_multiturn` | **工具多轮**:assistant `tool_calls` + `reasoning_content` 回填 + `role:"tool"`×2 + 续轮 |
| `ctx3_tools_notinking` | 同 ② 但 `enable_thinking=false`(模板 `_initial_thinking=false` 分支) |
| `ctx4_plain_notinking` | 无工具 + 关思考(模板自注入 reasoning 指令的分支) |
| `qwen35_0.8b_ctx1.txt` | 另一份快照(上游原版 153 行模板)= 家族通用性 |

## 怎么跑

```bash
# 单测(含金标;需指向模型快照 —— 机器资产,不在仓库内)
OWL_TPL_DIR=/path/to/models/cyankiwi/Qwen3.8-27B-AWQ-INT4 \
OWL_TPL_DIR_QWEN35=/path/to/models/Qwen/Qwen3.5-0.8B \
  cargo test --release -p server --bin server chat_tpl
```
未设 env = 自动跳过(与 GPU 用例同纪律)。

## 怎么重生成(**模板漂移报警器**)

```bash
# 需含 transformers+jinja2 的 venv(如 packages/vllm-test/.venv)
/path/to/venv/bin/python gen_fixtures.py
```

⚠️ **换模型 / 换模板 / 上游魔改模板后必须重跑**并复核 `*.txt` 的 diff:
模板是**快照资产**,不在本仓库版本控制内 —— 这张 diff 就是"提示词面变了"的唯一警报
(背景:同一家族的模板在本机已有 3 个版本:27B 魔改 461 行 / `.pre-froggeric` 175 行 /
0.8B 原版 153 行)。

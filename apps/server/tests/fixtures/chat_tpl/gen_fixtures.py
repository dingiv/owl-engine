#!/usr/bin/env python3
"""chat_tpl 金标生成器(HF transformers 真值面)。

口径(agent 服务面专项 §5 G0 / §7.1):
  参考实现 = HF `tokenizer.apply_chat_template(...)`,即模型快照自带
  `chat_template.jinja` 在 `transformers/utils/chat_template_utils.py` 的
  ImmutableSandboxedEnvironment 下的渲染结果(trim_blocks/lstrip_blocks +
  HF 版 tojson + pycompat 语义)。
  Rust 侧渲染器(`apps/server/src/chat_tpl.rs`)必须与这些 `.txt` **逐字节相等**。

用法(需含 transformers 的 venv;禁系统 python):
  /path/to/venv/bin/python gen_fixtures.py
产物:同目录 ctx*.json(输入)+ ctx*.txt(期望输出;均进版本 —— 资产无版本=哨兵失明)。

⚠️ 路径是本机资产(模型快照在 packages/owl 树外);换机改 MODEL/MODEL_QWEN35。
⚠️ 模板文件属于快照(可能被上游/本地魔改),换模型或模板后**必须重跑**本脚本
   并复核 diff —— 那正是"模板漂移"的报警器。
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "/home/div/Documents/codes/models/cyankiwi/Qwen3.8-27B-AWQ-INT4"
MODEL_QWEN35 = "/home/div/Documents/codes/models/Qwen/Qwen3.5-0.8B"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command. Use <cmd> carefully & wisely.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "命令,支持中文"},
                    "timeout": {"type": "number", "default": 30},
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
]

MULTITURN = [
    {"role": "system", "content": "You are pi, a coding agent."},
    {"role": "user", "content": "看看目录"},
    {
        "role": "assistant",
        "content": "",
        "reasoning_content": "我需要先列目录,再决定下一步。",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                # vLLM 归一后的形态:arguments 是**对象**而非 JSON 字符串
                # (vllm/entrypoints/chat_utils.py::_postprocess_messages;
                #  owl 侧同款归一在 apps/server/src/openai.rs)
                "function": {"name": "bash", "arguments": {"command": "ls -la"}},
            }
        ],
    },
    {
        "role": "tool",
        "content": "exit code: 0\ntotal 4\ndrwxr-xr-x 1 div div 4096 hello.txt",
        "tool_call_id": "call_1",
    },
    {"role": "user", "content": "现在把它改成 HELLO"},
    {
        "role": "assistant",
        "content": "我先读出来再写。",
        "reasoning_content": "改动前先读。",
        "tool_calls": [
            {
                "id": "call_2",
                "type": "function",
                "function": {"name": "read", "arguments": {"path": "hello.txt"}},
            },
            {
                "id": "call_3",
                "type": "function",
                "function": {"name": "bash", "arguments": {"command": "cat hello.txt"}},
            },
        ],
    },
    {"role": "tool", "content": "Exception: file not found: hello.txt", "tool_call_id": "call_2"},
    {"role": "tool", "content": "command not found: cat", "tool_call_id": "call_3"},
]

CASES = {
    # ① tools 块 + 单轮 user(基础路径)
    "ctx1_tools": {
        "messages": [
            {"role": "system", "content": "You are pi, a coding agent. Work in /home/div/proj."},
            {"role": "user", "content": "列出当前目录,然后创建 hello.txt"},
        ],
        "tools": TOOLS,
        "add_generation_prompt": True,
        "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True},
    },
    # ② 工具多轮(assistant tool_calls + reasoning_content 回填 + role=tool ×2 + 续轮)
    "ctx2_tools_multiturn": {
        "messages": MULTITURN,
        "tools": TOOLS,
        "add_generation_prompt": True,
        "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True},
    },
    # ③ 同 ② 但关思考(pi /thinking off;模板走 _initial_thinking=false 分支)
    "ctx3_tools_notinking": {
        "messages": MULTITURN,
        "tools": TOOLS,
        "add_generation_prompt": True,
        "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": True},
    },
    # ④ 无工具 + 关思考(纯闲聊路径;模板自注入 reasoning 指令的分支)
    "ctx4_plain_notinking": {
        "messages": [{"role": "system", "content": "hi"}, {"role": "user", "content": "1+1=?"}],
        "add_generation_prompt": True,
        "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": True},
    },
}


def render(model_dir: str, case: dict) -> str:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    tpl_path = os.path.join(model_dir, "chat_template.jinja")
    with open(tpl_path, encoding="utf-8") as f:
        tok.chat_template = f.read()
    ctx = dict(case)
    kw = ctx.pop("chat_template_kwargs", {}) or {}
    return tok.apply_chat_template(
        ctx["messages"],
        tools=ctx.get("tools"),
        tokenize=False,
        add_generation_prompt=ctx.get("add_generation_prompt", True),
        **kw,
    )


def main() -> None:
    for name, case in CASES.items():
        with open(os.path.join(HERE, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(case, f, ensure_ascii=False, indent=2)
            f.write("\n")
        out = render(MODEL, case)
        with open(os.path.join(HERE, f"{name}.txt"), "w", encoding="utf-8") as f:
            f.write(out)
        print(f"{name}: {len(out.encode())} B")
    # 家族通用性:另一份快照(上游原版 153 行模板)
    out = render(MODEL_QWEN35, CASES["ctx1_tools"])
    with open(os.path.join(HERE, "qwen35_0.8b_ctx1.txt"), "w", encoding="utf-8") as f:
        f.write(out)
    print(f"qwen35_0.8b_ctx1: {len(out.encode())} B")


if __name__ == "__main__":
    main()

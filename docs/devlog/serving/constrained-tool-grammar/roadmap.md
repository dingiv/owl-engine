# Constrained tool grammar —— 施工路线

> 状态：待排期（设计已冻结）
> 估算基准：1 名熟悉 owl 引擎/服务面的工程师；不含模型质量调优与 Anthropic `/v1/messages`

---

## 阶段总表

| 阶段 | 内容 | 工时 | 出口 |
|---|---|---:|---|
| P0 | 离线 spike：tokenizer env + Qwen grammar + mask | 2–3 人日 | 能证明 grammar 接受/拒绝正确 |
| P1 | MVP：required/GUI，host sampler，spec off | 3–5 人日 | GUI 真机格式 100%，普通聊天不变 |
| P2 | auto grammar + 3080 spec 保持 | 10–15 人日 | 普通 agent 零配置，decode 损失 ≤2% |
| P3 | 完整硬保证 + strict/required/named | 5–8 人日 | 全路径、全测试、文档收口 |

**只做 P0+P1：约 5–8 人日。**  
**做到 P2：约 15–23 人日。**  
**做到 P3：约 20–30 人日。**

---

## P0：离线 spike（2–3 人日）

### 任务

1. 用 `llguidance` 构建 `TokenizerEnv`：
   - 读模型 `tokenizer.json`；
   - `token_bytes_from_tokenizer_json`；
   - `TokTrie::from` + `TokenizerEnv` 实现；
   - 特殊 token / EOS 对齐。
2. 生成 Qwen XML grammar：
   - 32 tools 的工具名枚举；
   - 每个工具的 parameter 名枚举；
   - 多 tool call、自由文本参数值；
   - auto / required 两版。
3. 离线测试：
   - 合法 Qwen XML 接受；
   - `<command>` 拒绝；
   - Anthropic wrapper 拒绝（required）；
   - 未知工具/参数拒绝；
   - 32 tools 编译时间、mask 时间。
4. 记录：
   - compile ms；
   - compute_mask p50/p95/max；
   - commit 正确性；
   - 与 `ChannelSplitter` 的输出一致性。

### 出口门

```text
G0：20 个合法样本全接受；20 个非法样本全拒绝
G1：32 tools compile ≤ 2s；mask p95 ≤ 1ms（目标 ≤0.2ms）
G2：grammar 接受样本经 parser 得到同一 tool call
```

### 风险

- llguidance 的 Lark lexer 对自由文本 + trigger 的处理；
- 248k 词表下 mask 延迟；
- tokenizer 特殊 token 对齐。

---

## P1：MVP（3–5 人日）

### 范围

```text
tool_choice=required / GUI 模式
host sampler
spec decode 关闭
3080 优先，3090 可选
```

### 任务

1. `TurnOpts` 增加 `grammar: Option<TurnGrammarHandle>`。
2. `openai.rs`：
   - `tool_choice=required` 时构建 required grammar；
   - auto 暂不默认；
   - 失败回退 repair 层。
3. host sampler：
   - `read_full_logits` 后 `compute_mask`；
   - `apply_mask`；
   - `sample`；
   - `commit_token`。
4. 该 turn 关闭 spec（`spec.depth=0` 或 turn 级开关），避免候选 top-32 问题。
5. 观测：
   - compile/mask/fallback 日志；
   - `/debug/agent` grammar 字段。
6. 测试：
   - GUI 首轮 screenshot；
   - 合法/非法格式；
   - abort/断流；
   - 普通聊天零变化。

### 出口门

```text
GUI 真机 20 轮：工具格式 20/20
普通聊天：decode 不变
required turn：decode 允许下降（spec off），但工具轮总时长可接受
```

---

## P2：auto grammar + spec 保持（10–15 人日）

### 范围

```text
tools 非空 + tool_choice 缺省/auto
3080 TP2 + DFlash2 d7 保持开启
```

### 任务

1. 服务端策略：
   - `tool_grammar = auto` 默认；
   - `off/auto/required` 三档；
   - 与 `tool_choice=none/required` 合并。
2. Grammar cache：
   - key = model + format + thinking + tools schema hash；
   - LRU + 上限；
   - 编译与 prefill 重叠。
3. 3080 spec：
   - 方案 A（先做）：grammar turn 关闭 `use_cand`，回退 full logits；测损失；
   - 方案 B（目标）：bitmask 图输入槽 + mask 核；
   - 在 top-32/argmax 前应用；
   - draft token commit/rollback。
4. 3090：
   - 设备 mask 或 host fallback；
   - 允许该档默认 off。
5. 观测：
   - cache hit/miss；
   - mask p50/p95；
   - spec rollback；
   - grammar fallback。
6. 性能 A/B：
   - std 五点三轮；
   - 32 tools / 4 tools；
   - spec AL；
   - TTFT。

### 出口门

```text
32 tools 普通 agent：decode 损失 ≤2%
TTFT：缓存后 ≤ +5ms；首次 ≤ +1.5s（可与 prefill 重叠）
工具格式错误率：≤0.5%
spec AL 下降：≤5%
```

---

## P3：完整硬保证（5–8 人日）

### 任务

1. `tool_choice=required` / named tool 的完整语义；
2. `strict` 参数 schema（可选）；
3. 多工具、多调用、长参数、Unicode、转义；
4. grammar 与 `tool_call_format=json` 的兼容/拒绝；
5. 错误策略：
   - 编译失败 400 或 fallback（配置）；
   - mask/commit 失败 502/SSE error；
6. 文档：
   - `design.md` / `roadmap.md` 更新；
   - `agent-surface.md` 链接；
   - 配置说明；
7. 回归：
   - server 单测；
   - GUI/agent workload；
   - 3080/3090 A/B；
   - spec 等价门。

---

## 未来文件触点（预估）

```text
Cargo.toml                                  # llguidance 已声明，确认版本/feature
apps/server/src/openai.rs                   # tools -> GrammarPlan, tool_choice 策略
apps/server/src/toolgrammar.rs（新）        # grammar 生成/缓存
apps/server/src/config.rs                   # tool_grammar 配置
apps/server/src/agent_trace.rs              # /debug/agent grammar 字段
crates/engine/src/turn.rs                   # TurnOpts.grammar
crates/engine/src/exec/phases.rs            # host sampler 接缝
crates/engine/src/exec/spec.rs              # spec verify 接缝
crates/engine/src/graph_fleet.rs            # bitmask 图输入槽（P2）
crates/kernels/...                          # apply_token_bitmask（P2）
crates/engine/Cargo.toml                    # llguidance
```

---

## 决策待办

1. P0 实测后确认 `llguidance` vs `xgrammar`；
2. P1 实测 `required` 关闭 spec 的工具轮总时长是否可接受；
3. P2 选择“关闭候选路径”还是“图输入 bitmask”；
4. 3090 是否默认开 grammar；
5. auto grammar 是否真的能挡住 Anthropic wrapper，还是只挡 Qwen 内部近失；
6. 前端减工具后的错误率是否已降到不需要 P2。

---

## 当前不做

- 不实现 Anthropic `/v1/messages`；
- 不做全局 `required`；
- 不做参数值 JSON Schema 全约束；
- 不改模板默认内容；
- 不提交 git（等用户明确指示）。

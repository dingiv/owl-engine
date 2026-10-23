# KVMem query replay 最小闭环（2026-10-23）

> 论文依据：KVMem §4.2 step-level scheduling / §5.2
> “selected working set 固定后，current query 在更新视图上重新 prefill，再 decode”。

## 1. 已落地

```text
[kvmem].query_replay = true   # 当前仅 spec.depth=0 + 单卡
```

- 末个 prefill chunk 前，把会话 GDN 状态拍进专用 `replay_snap` 缓冲；
- 首次末 chunk 前向 + 最终 recall/视图装配后，恢复该 GDN 快照；
- 用**新视图**重跑末 chunk（`ChunkHost { sel: None, bank: None,
  select_decode=false }`），重新 harvest 末行 token；
- 再进入 decode。
- KV 槽/链不动：重跑只覆盖同位置 K/V；不重复推进 chain。
- `snap_max=0` 的档也可用（replay 缓冲独立于前缀快照池）。
- 当前限制：`spec.depth > 0` 与 TP 响亮拒（草稿环/双 rank replay 另立刀）。

日志：

```text
[kvmem-replay] query replay:ctx=... base=... t=... view=... 页(新视图上重跑末 chunk)
```

## 2. GGUF 60k 实测（Q4_K_M，单 3090 Ti）

coherent needle，code 在 token ~6.6k（逻辑块 207），问题在末尾：

| workspace | replay | 输出 |
|---:|:---:|---|
| 8192 | on | `:`（EOS） |
| 16384 | on | `:` |
| 32768 | on | `:` |
| 49152 | on | `:` + 解释，已看到 “The secret code is BLUEBIRD-42.” |
| 49152 | **off** | 与 on **逐字相同**（同样看到 code） |

块 207 在 8k/32k/48k 均被选中，target-hash `cfc5a2a59abf133f`。

## 3. 结论（负结果，重要）

1. **query replay 机制已执行**，但在这条 60k 单轮 prompt 上**没有改变输出**
   （48k on/off 逐字相同）。第一 token `:` 是模板/生成行为，不是 replay 能修的。
2. 60k 的瓶颈是 **execution view 预算 `Ba`**：
   - 8k/32k：选中针块但模型看不到足够上下文，只有 `:`；
   - 48k：模型明确复述出 `BLUEBIRD-42`；
   - 论文的 24GB 消费档用 **80K execution view**（NVFP4 + FP8 KV），
     我们 8k 远低于论文口径。
3. 因此 query replay 不是 60k 的充分修复；它是 step-level/长会话 replay 的
   必要基础件，但当前这条单轮测试不能证明收益。
4. 下一步应先把 `Ba` 提到论文量级（GGUF 可开 fp8 KV / 64k view），
   再做 step-level + compact re-RoPE；不要用 replay 掩盖预算问题。

## 4. 配置建议

```toml
[kvmem]
query_replay = true     # 保留；当前单轮 60k 不改变输出
workspace_per_req = 49152  # 该 60k 档至少需要 ~48k 才能看回 code
```

`profiles/qwen38_27b-3090-gguf-kvmem.toml` 暂用 8192 + replay；60k 质量档
需显式 `--set kvmem.workspace_per_req=49152`（f16 池 48k 已贴顶）。

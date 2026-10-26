# 2026-10-26 交接：leave-gap 前端联调、重复输出定位、视觉塔缓存

> 状态：**leave-gap 核心已落地并通过 0.8B 门；生产联调发现两个独立问题；视觉塔缓存尚未实现。**
>
> 本文是下一个会话的交接，不是结论记录。代码尚未 commit，当前 `main` 工作树有 21 个修改文件；
> 先读本文，再决定是否继续改代码或先做复现实验。

---

## 0. 一分钟摘要

### 已完成

- `image_evict_mode = "leave_gap"`：
  - 删除图片只摘页，不旋转 K；
  - 幸存 token 保留绝对 position，新 token 从 `max_pos + 1` 追加；
  - 删除段/旧链尾的缓存引用在宿主回灌完成后解除，槽何时 free 由 store 决定；
  - 支持宿主后缀回灌后再吸收；
  - target spec verify/replay 使用绝对 `next_rope_pos - 1`；
  - 保留 `rebase` 回退路径，默认仍是 `rebase`。
- 原 `gpu_image_evict_gate` 已直接改造成 leave-gap 版，并新增机械断言：
  - 删除前后保留后缀 K 逐字节不变；
  - `execute_begin` 后，被删图槽 `slot_refs == 0`；
  - 旧链止于删点；
  - 连续第二次删除仍吸收，`delta_limit=1` 不触发重建。
- 0.8B e2e、spec identity e2e、CPU 套件、release 构建均通过。
- 3080 TP2 生产联调跑过 23 次 leave-gap 删除，全部吸收；热轮 TTFT 约 1.7–1.8s，cached 95–97%。

### 未完成

1. **视觉塔重复计算历史图片**：每轮把 prompt 里 3 张图全部重新跑 tower，造成约 0.7–0.8s 固定开销。
   - 实测 1 图历史 + 34 新 token：TTFT 419ms；
   - 3 图历史 + 34 新 token：TTFT 1210ms；
   - 1.3k 纯文本冷 prefill：877ms。
   - 目标：只编码新增图片，缓存历史 `merged` 视觉特征。
2. **turn35 重复 token 现场**：模型在 raw JSON 截图后输出 8k 重复 token，根因尚未定谳。
3. **turn38 上下文压缩后的全量 prefill**：前端摘要后 prompt 重写，cache=0，TTFT 21.8s，前端断开。
4. **P4 consolidation**：position 上限/压缩仍未实现。
5. **删除段的 GDN 快照清理**：KV 链引用已解除，但删除段旧 snapshot 是否全部释放仍需核查。

---

## 1. 当前运行现场

### 8135（3080 TP2，leave-gap）

当前进程（2026-10-26 post-delete 缓存修复后）：

```text
server PID: 800821
port: 0.0.0.0:8135
profile: profiles/qwen38_27b-3080_tp2.toml
override: pool.image_evict_mode=leave_gap
request dump: /tmp/owl-requests-frontend
health: {"ok":true,"model":"qwen3.8-27b"}
```

历史进程：734556（事故后 fresh replay 进程，已停）。

启动命令：

```bash
cd /home/div/Documents/codes/packages/owl
./target/release/server \
  -p profiles/qwen38_27b-3080_tp2.toml \
  --set server.bind=0.0.0.0:8135 \
  --set pool.image_evict_mode=leave_gap \
  --set server.request_dump_dir=/tmp/owl-requests-frontend \
  --log-level info
```

当前日志：

```text
/tmp/owl-8135-fix-final.log
```

历史日志：

```text
/tmp/owl-8135-repro.log
```

事故现场日志（旧进程，保留）：

```text
/tmp/owl-8135-leave-gap.log
```

注意：

- 当前 8135 是在事故后**重启清缓存**用于 fresh replay 的进程，已经不是 turn35 原始 session 状态。
- 当前前端已断开；若要复现 turn35，需要从 request dump 重建到 turn34。
- 8134（3090 单卡）仍由 PID 466804 运行，未动。
- 8136（3090 测试 server）已停止。
- `/tmp/owl-8135.pid` 里是启动 shell 的 PID，不一定是 server；用 `nvidia-smi --query-compute-apps=pid` 或 `ss -ltnp` 确认。
- 3080 卡位置：GPU 0/3；3090：GPU 1/2。

### 代码状态

```text
HEAD: 4c227b9e
main ahead origin/main: 40
未 commit，未 push
```

修改文件：

```text
crates/engine/src/engine.rs
crates/engine/src/exec/phases.rs
crates/engine/src/exec/prefill_chunk.rs
crates/engine/src/exec/spec.rs
crates/engine/src/exec/turn_end.rs
crates/engine/src/img_evict.rs
crates/engine/src/running.rs
crates/engine/src/scheduler.rs
crates/engine/src/session.rs
crates/engine/src/tests.rs
crates/engine/src/turn.rs
crates/engine/tests/engine.e2e.rs
crates/models/src/vlm.rs
crates/shared/src/config/mod.rs
docs/README.md
docs/devlog/dev/config.md
docs/devlog/memory/image-evict-leave-gap-交接.md
docs/devlog/memory/prefix-deletion.md
profiles/README.md
profiles/full.toml
profiles/parts/feature/image-evict.toml
```

---

## 2. leave-gap 代码事实

### 2.1 配置

`crates/shared/src/config/mod.rs`：

```toml
[pool]
image_evict = true
image_evict_mode = "rebase"   # "rebase" | "leave_gap"
```

- `EngineKnobs.image_evict_mode`。
- `leave_gap` 不要求 FP8 `image_evict_delta_limit ∈ 1..=4`。
- 默认仍是 `rebase`；3080/3090 profile 文件没有切成 leave_gap，当前 8135 是 CLI override。

### 2.2 Session 位置账

`crates/engine/src/session.rs`：

```rust
pub positions: Vec<[i32; 3]>
pub max_pos: i32
pub positioned: bool
pub position_epoch: u32
```

- `commit_turn` 同时写 prompt + generated 的位置；
- `reset` 清位置；
- `rebase` 清并递增 epoch；
- `set_positions` 校验长度；
- `positions` 与 `tokens` 等长，unpadded 坐标不另存位置。

### 2.3 删除计划

`crates/engine/src/scheduler.rs`：

```text
plan_image_leave_gap
```

关键行为：

- 找旧链前缀/后缀；
- 后缀按页保留；
- 新表：前缀 ++ 后缀；
- 设备页直接 `incref`；
- 宿主页写 `PENDING_LOAD_PLACEHOLDER` + `pending_loads`；
- 旧链尾（删点起全部 matched 块）放 `release_chains`；
- 不做 K 旋转；
- 快照边界重挂；
- `apply_image_leave_gap` 写新表、tokens、positions、chain；`positioned=true`；`rot_rounds=0`。

`crates/engine/src/exec/phases.rs::execute_begin`：

```text
pending_loads 回灌（host -> device）
  ↓
release_chains：drop_chain（只解除缓存引用，不直接 free）
  ↓
write_bt / prefill
```

**顺序不能倒**：先 drop 会 evict 宿主内容，回灌失败。

### 2.4 前向位置

- `ChunkTables` 接收显式 positions；
- `Vlm::prepare_with_positions` 用位置账生成 m-RoPE cos/sin；
- `ActiveTurn.next_rope_pos` 是下一枚 token 的绝对位置；
- decode 用 `next_rope_pos - 1`（当前喂入 token 的位置）；
- spec target verify/replay 用 `rope_delta = next_rope_pos - 1 - pos`；
- 草稿仍走既有 token 位置（只影响 AL，不影响 greedy 恒等）。

### 2.5 缓存安全

- `positioned` 会话的跨会话内容命中要求 tail owner == 本会话；
- 同会话链继续复用；
- 纯追加的 positioned 会话保持 `positioned=true`；
- 删除段旧链身份释放后，旧历史不再完整命中，这是预期。

---

## 3. 已验收测试

### 3.1 0.8B e2e

```bash
OWL_TEST_DEVICE=42:00.0 \
OWL_E2E_ONLY=gpu_image_evict_gate \
cargo test -p owl-engine --features e2e --test engine_e2e -- --test-threads=1
```

通过。当前门是 **leave-gap 版**：

- 两图 → 删第一张；
- 保留后缀 K 逐字节不变；
- 删除前被删槽 `refs > 0`，`execute_begin` 后 `refs == 0`；
- 旧链止于删点；
- 连续第二次删除仍吸收；
- `delta_limit=1` 不触发 rebase。

### 3.2 spec identity

```bash
OWL_TEST_DEVICE=42:00.0 \
OWL_E2E_ONLY=gpu_image_evict_spec_identity_gate \
OWL_GATE_LEAVE_GAP=1 \
cargo test -p owl-engine --features e2e --test engine_e2e -- --test-threads=1
```

通过：leave-gap 下 spec on/off 贪心 token 逐位相同。

### 3.3 CPU / release

```text
cargo test -p owl-engine --lib          → 158 passed
cargo build --release -p server         → pass
```

`gpu_08b_chunk_invariance_gate` 因本地缺少 `dev-08b` profile 未跑，是 boot 配置问题，不是本改动。

---

## 4. TTFT / 删除数据

### 4.1 23 次吸收轮

前端会话 `0xc7a301a9fe5b2320`：

```text
吸收次数: 23
新增 prefill: 1036–1100 tok（1 次 1676）
中位数: 1068 tok
平均: 1093 tok
cached: 95–97%
TTFT 热轮: 约 1.7–1.8s
```

说明 leave-gap 没有把 896 token 的删除图再 prefill；新增主要是一张新截图（896）+ 文本尾巴（约 140–204）。

### 4.2 视觉塔开销 A/B

同一 8135，同模型，文本尾巴仅 34 token：

```text
1 张历史图: TTFT 419ms
3 张历史图: TTFT 1210ms
差: 约 791ms（约 0.4s/图，1024 图）
```

纯文本 1301 token 冷 prefill：877ms。

结论：turn 的 `prefill_tps` 被视觉塔时间污染；1k 新增 token 的纯文本 prefill 约 0.8–0.9s，
3 张历史图 tower 再增加约 0.7–0.8s。

### 4.3 27B 单 3090 leave-gap A/B

22k prompt，三图，删首图轮：

```text
leave-gap: TTFT 1342ms, cached 98.3%, prefill 368
--no-evict: TTFT 18070ms, cached 0%
```

另一次长上下文删除轮出现宿主回灌 1090 页，仍吸收，TTFT 约 2.1s。

---

## 5. 事故现场一：turn35 重复 token

### 5.1 事实

`/tmp/owl-8135-leave-gap.log`：

```text
turn 35:
prompt=38540
cached=37472 (97.2%)
prefill=1068 tok
ttft=1820ms
completion=8192
finish=length
spec rounds=1186
spec accepted=7005
spec_al=6.91
```

本 turn 的 leave-gap 删除正常，没有 rebase、OOM、NaN、报错。

请求 dump：

```text
/tmp/owl-requests-frontend/req-1792970929079652810-727608-0.json
```

- session `rover:01a1729b-0fac-771d-b453-6db15331eef3`；
- 88 messages，30 tools，3 images；
- 最后一张截图：`/tmp/turn35_img2_87.png`（从 request dump 解出）；
- 页面是 `https://s.search.bilibili.com/main/hotword` 的原始 JSON，含大量 mojibake（`â€`、`a4e3` 等）；
- 模型输出里的 `a4e3` / `Â€` 与截图内容对应。

输出里出现：

```text
（上下文压缩开始）
...
（上下文压缩结束）
```

**turn35 的 prompt 里没有“上下文压缩”字样**；`grep` request dump 为 0。
因此这段是**主模型自己生成的文本**，不是前端注入，也不是前端摘要请求的输出。

### 5.2 已做对照

重启 8135 清空缓存后，用同一 turn35 请求、fresh cache、`temperature=1.0` 重放：

```text
prompt=38092
cached=0
completion=677
finish=tool_calls
a4e3 出现 0 次
Â€ 出现 0 次
```

输出仍然读乱码 JSON，但正常发出 tool call，没有 8k 循环。

### 5.3 当前判断（未定谳）

强候选：

1. 病态 raw JSON / mojibake 截图诱导模型尝试逐字解析，进入重复采样；
2. `temperature=1.0 / topk=20` 放大了重复概率；
3. DFlash2 spec 高 AL（6.91）把重复流快速推进到 8192，属于放大器不是根因；
4. 原始 session 经过 23 次 leave-gap 删除后的 position 账/GDN 状态是否参与，**尚未复现**；
5. fresh replay 是 compact position，不等价于原始 position 洞，不能据此完全排除 leave-gap。

### 5.4 下一步最小实验

1. 从 request dump 重建原 session 到 turn34（保留原始 turn id / seed 更佳），再发 turn35；
2. 四臂：

```text
spec on  × temp 1.0
spec on  × temp 0
spec off × temp 1.0
spec off × temp 0
```

3. 对比前 512/1024 token 是否出现重复 n-gram；
4. 记录 `max_pos`、`position_epoch`、`max_pos - tokens.len()`（洞大小）、GDN 是否 reset；
5. 若只有 leave-gap 原始 session 复现，优先怀疑 position 洞/GDN；若 fresh 也复现，优先输入/采样；
6. 可加一个服务端 n-gram 重复诊断（只记录前 32 token / 重复周期，不改变生成）。

---

## 6. 事故现场二：turn38 上下文压缩后全量 prefill

### 6.1 前端确实调用了摘要模型

turn36/37 是独立请求，不是主 session：

```text
req-1792970982074521275-727608-0.json
req-1792970993711841440-727608-0.json

session = None
system = "You are a context summarization assistant. ... ONLY output the structured summary."
max_tokens = 13107 / 8192
```

日志：

```text
turn 36: prompt=958, session 0x0
turn 37: prompt=6850, session 0x0
```

这是前端/harness 在主模型重复后触发的**上下文压缩**，输出是摘要/checkpoint，不是主会话。

### 6.2 turn38

```text
session=0xc7a301a9fe5b2320
prompt=31436
cached=0
max_new=8192
ttft_ms=21858
elapsed_ms=62172
usage=0
finish_reason=None
error=未结算
```

日志：

```text
actor 事件投递失败(请求方已断开 ⇒ 本 turn 弃置): channel closed
actor 弃置 turn 0(客户端断流)
Broken pipe
```

原因链：

```text
turn35 重复输出
  → 前端调用 summarizer 压缩历史
  → 压缩后的 prompt 开头/历史被改写
  → 前缀缓存失配 cached=0
  → 31k 全量 prefill，TTFT 21.8s
  → 前端等待超时/主动断开
  → actor 正确弃置
```

这不是模型重复，也不是 leave-gap 删除失败；是**上下文压缩破坏前缀 + 客户端超时**。

### 6.3 下一步

- 前端：压缩后是否用新 session？若仍用旧 session，必须接受 prefix miss；
- 若要保留缓存，压缩策略必须保留稳定前缀（system + 前若干消息）或把摘要追加而不是重写；
- 服务端：可加请求/流式心跳/取消指标，但当前 `actor` 已能正确清理；
- 评估是否需要更长客户端超时，或压缩后异步预热。

---

## 7. 下一主任务：视觉塔缓存复用

### 7.1 当前调用链

```text
apps/server/src/openai.rs::images()
  → 收集 messages 中所有 image_url
  → 每轮把所有历史图送 TurnOpts.images

crates/engine/src/exec/phases.rs::prepare_vlm_rank()
  → vlm.prepare_with_positions(ids, grids, [], px)
  → crates/models/src/vlm.rs::prepare_from_inputs()
  → tower_inputs(grids, px)
  → tower_forward(...)          // 对全部历史图重新 forward
```

`Vlm::prepare_from_inputs` 的 `merged` 是图像特征；`Vlm::inputs_from_positions` 单独生成
m-RoPE cos/sin 和 `idx`。两者位置语义分离。

### 7.2 结论

视觉塔特征**不依赖文本 position**：

- 塔内部只有图像网格的 axial rope；
- 同一图像/同一预处理/同一 grid ⇒ 同一 `merged`；
- 删除/移动图像只改变 `VlmInputs.idx` 和 cos/sin 位置，不改变 `merged`。

所以缓存 key 不需要 position：

```text
image content hash
+ preprocessed grid
+ tower weights / model version
+ dtype / TP rank
```

### 7.3 建议实现

1. **图像身份**：
   - 在服务端 `vision_processor` 或 `TurnImage` 增加 `image_id/hash`；
   - hash 必须覆盖预处理后的像素 + grid，不能只 hash data URL（同一图可不同编码）。
2. **缓存值**：
   - `merged [N, D]` 的 host/device 副本；
   - TP2 每 rank 一份或 host 一份后上传；
   - 记录图像行数，构建 `VlmInputs.idx` 时按图像顺序拼接。
3. **缓存生命周期**：
   - session 内按图像内容；图片移出历史即淘汰；
   - LRU 容量按 3–8 张/会话或全局预算；
   - 删除图片时清理对应 cache；
   - 图像内容 hash 变化必须 miss。
4. **prepare 改造**：
   - 只对 cache miss 的图片跑 tower；
   - cache hit 直接复制/复用 merged 行；
   - 保持 `VlmInputs` 的 `idx_host/cos_host/sin_host` 全 prompt 语义；
   - 不能只把旧图 `idx=-1` 而仍让 chunk 引用它们（删除边界可能落在图像内）。
5. **验收**：
   - 3 图历史 + 只加 34 文本：TTFT 从 1210ms 降到约 400–600ms；
   - 1k 新增图 + 文本：从 1.7s 降到约 1.0–1.3s；
   - 删除/移动图片不触发 tower miss；
   - 改图内容/grid 必须 miss；
   - 27B TP2 + leave-gap 100 轮不退化。

### 7.4 不要做的事

- 不要把 position 放进 tower cache key；
- 不要只缓存 data URL 而不缓存预处理结果；
- 不要用图像下标（image index）作 key，删除后下标会移动；
- 不要在 cache 复用时跳过 `VlmInputs` 的 cos/sin 重算。

---

## 8. 还需核查/可能返工

1. **删除段 GDN 快照**：`release_chains` 只解除 KV 链身份，删除段旧 snapshot 是否已从
   `StateStore`/host/disk 释放，尚未核查；可能仍占 snapshot 池。
2. **P4 consolidation**：position 洞持续增长；当前没有 `max_position` 阈值/压缩。
3. **旧 rebase e2e**：原 `gpu_image_evict_gate` 已改成 leave-gap 版；rebase K 旋转的完整
   机械门不再由这条 e2e 覆盖，后续需要单独补一条 rebase 回归门。
4. **生产 profile**：`profiles/qwen38_27b-3080_tp2.toml` 仍是 `rebase`。当前 8135 只是 CLI
   `--set`；重启若不带 override 会回到 rebase。
5. **当前 8135 是 fresh replay 后状态**，不是 turn35 原始 session；不要拿当前缓存判断 turn35。
6. **不要主动 commit/push**：用户未指示；所有改动保持工作树。

---

## 9. 常用路径 / 复现命令

### 9.1 服务

```bash
# health
curl -s http://127.0.0.1:8135/health

# 当前日志
tail -f /tmp/owl-8135-repro.log

# 事故日志
less /tmp/owl-8135-leave-gap.log
```

### 9.2 请求 dump

```text
/tmp/owl-requests-frontend/
```

关键文件：

```text
turn35 主请求:
  req-1792970929079652810-727608-0.json   # 88 msgs, 3 images
turn36/37 摘要请求:
  req-1792970982074521275-727608-0.json   # 958 tok
  req-1792970993711841440-727608-0.json   # 6850 tok
turn38 压缩后请求:
  req-1792971019003918799-727608-0.json   # 41 msgs, 31436 tok
```

解出的 turn35 最后一张图：

```text
/tmp/turn35_img2_87.png
```

### 9.3 探针

```bash
# 三图滑动窗口，吸收/缓存
python3 apps/server/tests/image_evict_sliding_probe.py \
  --base http://127.0.0.1:8135 --model qwen3.8-27b \
  --session slide-demo --turns 6 --keep 3 --max-tokens 8 --size 256 --no-think

# 对照：不带 owl_evict
python3 apps/server/tests/image_evict_sliding_probe.py \
  --base http://127.0.0.1:8135 --model qwen3.8-27b \
  --session slide-noevict --turns 6 --keep 3 --max-tokens 8 --size 256 --no-think --no-evict
```

### 9.4 门

```bash
OWL_TEST_DEVICE=42:00.0 \
OWL_E2E_ONLY=gpu_image_evict_gate \
cargo test -p owl-engine --features e2e --test engine_e2e -- --test-threads=1

OWL_TEST_DEVICE=42:00.0 \
OWL_E2E_ONLY=gpu_image_evict_spec_identity_gate \
OWL_GATE_LEAVE_GAP=1 \
cargo test -p owl-engine --features e2e --test engine_e2e -- --test-threads=1
```

GPU 测试必须串行，先确认卡空闲；不要用 `pkill -f`，用 `nvidia-smi --query-compute-apps=pid` 列表确认。

---

## 10. 建议的下一会话顺序

1. **先读本文 + `docs/devlog/memory/image-evict-leave-gap-交接.md`**；
2. 确认 8135/8134 进程和卡位；
3. **优先实现视觉塔缓存复用**（收益最大、风险最低、与事故解耦）；
4. 用 A/B 证明 tower 只在首见图片时执行，TTFT 降到目标；
5. 再复现 turn35：重建 session 到 turn34，做 spec/temp/position 四臂；
6. 单独处理 turn38：压缩后 prefix miss / 客户端超时；
7. 最后核查删除段 snapshot 释放，再考虑 consolidation；
8. 所有改动先跑最小门，不跑全量套件；不要 commit/push，除非用户明确要求。

---

## 11. 关键结论一句话

> leave-gap 的 KV 删除、前缀命中、引用解除和 TTFT 收益已经成立；
> 当前联调暴露的主要性能问题是**视觉塔每轮重算全部历史图**，
> 当前稳定性事故是**病态 raw JSON 截图诱导的重复采样 + spec 放大**，
> 以及**前端上下文压缩重写前缀导致的 31k 全量 prefill/超时**。
> 下一会话先做视觉塔缓存，再用受控四臂复现 turn35，别把三件事混成一个根因。

---

## 12. 视觉塔缓存落地（2026-10-26 续）

### 12.1 实现

问题：`prepare_vlm_rank` 原来每轮把**全部历史图**重新送塔；删除/移动图片
只改变文本位置与 `VlmInputs.idx`，不会改变每张图自己的 `merged`。

实现：

- `crates/engine/src/vlm.rs`
  - `VlmImageKey`：像素 `f32.to_bits()` 双 64 位指纹 + 长度 + grid；
  - `VlmMergedCache`：每 rank 一份，LRU，上限 256 MiB；
  - `merged_rows`：按 `t · (h/merge) · (w/merge)` 校验行数；
  - `combine_merged`：用 `[N,D]` 零块 + 逐图 `image_scatter` 组合，
    支持不同截图行数不等；组合块归 turn，缓存块只作输入。
- `crates/engine/src/exec/phases.rs`
  - 逐图查缓存；命中不跑塔；未命中只跑该图；
  - 塔中间块立即 free；merged 根块归 `Owner::Cache`；
  - 组合求值完成后才 `evict_to_cap`，避免释放仍在输入的缓存块；
  - 指标：`vlm.cache.hit` / `vlm.cache.miss` / `vlm.cache.evict`。
- `crates/engine/src/running.rs`
  - `RunningEngine.vlm_cache: [VlmMergedCache; 2]`（rank0 / peer）。
- `crates/engine/src/tests.rs`
  - `gpu_image_evict_gate` 末尾断言同图跨 turn 至少命中一次。

### 12.2 已验证

```text
cargo test -p owl-engine --lib
→ 161 passed

gpu_vlm_image_baseline                OK
gpu_image_evict_gate                  OK（VLM 缓存 hit=3 miss=4）
gpu_image_evict_spec_identity_gate    OK（leave-gap spec on/off 逐位相同）
```

8135 合成 A/B（同一张 128×128 图，第二请求只换前置 nonce 避免 KV 前缀命中）：

```text
1 图冷塔: TTFT 330ms
1 图热缓存: TTFT 162ms
→ 单图塔耗时约 168ms

3 图冷塔（其中 1 张已缓存）: TTFT 434ms
3 图全热缓存: TTFT 230ms
→ 本轮节省约 204ms；metrics 计数与逐图 miss/hit 对得上
```

### 12.3 当前 8135

```text
PID: 747883
log: /tmp/owl-8135-vlm-cache.log
profile: profiles/qwen38_27b-3080_tp2.toml
override: pool.image_evict_mode=leave_gap
request dump: /tmp/owl-requests-frontend
health: ok
```

旧日志保留：

```text
/tmp/owl-8135-repro.log
/tmp/owl-8135-leave-gap.log
```

### 12.4 仍未做

- 缓存上限目前是**引擎内常量 256 MiB/rank**，尚未进 `[pool]` 配置；
- 内容指纹每轮仍要扫一遍 `pixel_values`；若历史图很多，可后续把指纹在
  服务面预处理时一次算好并随 `VlmImage` 下传；
- 视觉塔缓存**不改变** turn35 的重复采样结论，也不解决 turn38 的前缀重写；
- 未 commit / 未 push。


---

## 13. post-delete 缓存塌陷修复（2026-10-26 续）

### 13.1 现场

真实现场 turn36 → turn37：

```text
turn36 删除起点 37,056，cache 40,512
turn37 客户端 messages 逐条等于上一轮前缀 + 新 user
turn37 cache 22,560，prefill 22,474，TTFT 17.153s
```

在 3090 单卡隔离服（8136）用 turn36 的 pre-delete 状态复现：

```text
turn2 删除 [37,024, 37,920)，cache 41,344，prefill 1,004
turn3 prompt 43,914，cache 36,960，prefill 6,954，TTFT 5.8s
```

诊断日志（`leave-gap 复用`）：

```text
matched=1156 m=1155 tail_snap=None
```

`1156` 页 = 36,992 tok，正好是删点前最后一个 `pad_before` 块之前。
即：**内容树只匹配到删点前的孤儿 pad 块**，而不是 suffix 真的丢了。

### 13.2 根因

leave-gap 删除时，删除段前的 `pad_before` 块留在 padded 前缀里，但对应图像已经不存在。
下一轮是**普通追加**，`image_pad_law` 从客户端裸文本重算 pad，这个孤儿 pad 块
不会被重新插入 ⇒ 前缀在删点前一个 pad 块处分叉，cache 退到删点之前。

### 13.3 修复

`crates/engine/src/running.rs` + `crates/engine/src/img_evict.rs`：

- 新增 `padded_prefix_for_raw()`：把 unpadded 前缀长度映射回 padded 切点，
  **吞掉切点后的 pad 块**（孤儿 pad 必须留下）。
- `image_pad_law` 在无删除、客户端只是追加/改尾部时：
  - 求 `old_text` 与 `input` 的公共前缀 `common`；
  - 切点落在图像内部 ⇒ 放弃复用，走裸文本重算；
  - 映射 padded 切点 `p`，拼接 `s.tokens[..p] + input[common..]`；
  - 继承 `s.pads` 中 `hi <= p` 的 pad；
  - 拼接后图像游程数必须与当前 `marks` 一致，否则放弃复用。
- 单测：`padded_prefix_tests::raw_prefix_mapping_keeps_trailing_orphan_pad`。
- e2e 回归：`gpu_image_evict_gate` 在删除轮后断言
  `match_prefix(s.tokens).matched_blocks > del.a/page`（suffix 必须进树）。

### 13.4 修复后实测（8136，同构复现）

```text
turn2 删除 cache 41,344
turn3 cache 42,336（> 删点 37,024）
turn3 prefill 2,750（新增 assistant + user，非 suffix 重算）
turn3 TTFT 3.62s（修复前 5.8～9.2s）
```

### 13.5 验证

```text
cargo test -p owl-engine --lib
→ 162 passed

OWL_TEST_DEVICE=42:00.0 OWL_E2E_ONLY=gpu_image_evict_gate ...
→ OK（新链 matched=7 / 7 页，suffix 已覆盖）

OWL_TEST_DEVICE=42:00.0 OWL_E2E_ONLY=gpu_image_evict_spec_identity_gate OWL_GATE_LEAVE_GAP=1 ...
→ OK（spec on/off 逐位相同）
```

8136 已停。8135 已用最终 release 重启：

```text
PID: 800821
log: /tmp/owl-8135-fix-final.log
```

尚未 commit / push。

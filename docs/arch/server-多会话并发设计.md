# owl server 上层架构:本地多会话与并发设计

2026-10-13 立项落地(§三十五 server 上层;此前"上层未完整落地"的补全)。
定位纪律:**owl 只服务本地**(127.0.0.1 回环),不做对外高并发;最大并发
默认 8、启动可配。

## 一、定位与约束

| 约束 | 决策 |
|---|---|
| 本地单机服务 | bind 127.0.0.1(默认);无鉴权/无 TLS(回环信任)|
| 不搞高并发 | 并发上限默认 8,`server.max_concurrency` 启动可配 |
| 单引擎 actor(批=1) | 请求在 actor 队列**串行执行**;并发闸门限"在跑+排队"总量 |
| GDN 状态格 = 会话格 | 格数(knobs.gdn_slots)≥ 并发上限(启动 fail-fast 校验)|

## 二、会话模型

### 2.1 会话身份与状态

- **session 键**:OpenAI 请求的 `session` 字段(任意字符串)→ 哈希为
  u64 会话 id(进程内 DefaultHasher)。
- **AgentSession**(session.rs,已有):GDN 格号、KV block_table、
  cached_len(增量 prefill 指针)、前缀守卫、LRU last_use。
- **GDN 状态格分配**:get_or_create 扫描首个未占用格号;格耗尽 →
  LRU 逐出最久未用会话(逐出 = close + 页表释放)后重试。
- **§三十二 真凶修复的落点**:会话格隔离 = 跨 session 的 GDN 状态
  互不污染(此前同格残留 = 半崩根因)。同 session 连续请求共享格 =
  多轮连续对话的状态连续性(合法保留)。

### 2.2 多轮渲染(客户端拼历史)

owl 服务端**不存对话历史**(瘦服务端):多轮 = 客户端按 OpenAI 惯例
拼全量 `messages`;服务端 render_prompt 渲染 **Qwen chat 模板**
(`<|im_start|>role\n{text}<|im_end|>` × n + assistant 起手 + 空 think
预填——非思考模式,2026-10-13 §三十五 落地)。

- **前缀缓存红利**:同 session 连续请求的前缀逐字节相等 → 引擎前缀
  守卫命中 → 增量 prefill(模板落地前 naive 拼接跨 turn 失配回退全量)。
- 已知限制:role 不区分的 naive 拼接已废;正式 jinja 模板通用化挂账
  (现 Qwen 硬编码于 openai.rs,单家族现实)。

## 三、并发模型

### 3.1 闸门

- `tokio::sync::Semaphore(max_concurrency)`,HTTP accept 后每连接
  acquire 一 permit,**持有至连接服务完毕**(长连接 = 占一位)。
- 超出 max_concurrency 的连接**排队等待**(本地友好,不回 429);
  客户端超时自行管理。
- 启动 fail-fast 校验:`knobs.gdn_slots ≥ max_concurrency`(格不足 =
  活跃会话 LRU 互踩 → 状态污染,账外组合不静默)。

### 3.2 执行

- 引擎 actor 单线程串行:闸门放行的请求进 actor 队列逐个执行。
- **吞吐语义**:并发上限限制的是"进行中会话/排队"规模(资源面),
  不是并行加速(引擎批=1;真批处理 = B5 批桶,另案)。

### 3.3 会话格与驱逐

- 格数 = `knobs.gdn_slots`(≥ max_concurrency):每活跃会话占一格。
- LRU 逐出:格满时新会话逐出最久未用会话(页表释放 + 格回收);
  被逐出会话下次请求 = 全量重算(正确性保住,缓存失效)。

## 四、配置

```toml
[server]
max_concurrency = 8    # 最大并发(默认 8;须 ≤ knobs.gdn_slots)

[pool]
gdn_slots = 8          # GDN 状态格(= 活跃会话上限;生产 flavor 锁 1 = 单会话)
```

- 启动校验:`gdn_slots < max_concurrency` → 拒启(打印调参提示)。
- 生产 flavor(spec-dflash)锁 gdn_slots=1:单会话语义(并发闸门
  实际 = 1,串行);**多会话/8 并发 = `[pool] gdn_slots = 8` 一行**。

## 五、验证(E2E)

`apps/server/tests/e2e.rs`(OWL_E2E=1):

1. 会话隔离:独立 session 自含 codeword(PURPLE/ORANGE)各自应答正确
   —— 跨会话状态不串。
2. 多轮记忆:客户端全历史 3 轮,名字记忆(Alex)。
3. 并发:8 并发请求全 200(排队语义);语义判据全过。
4. L1 协议面强断言(回归防护网)持续绿。

## 六、已知边界(挂账)

- **真批处理**(B5 批桶):多请求 GPU 级合批 —— 另案(吞吐方向)。
- **正式 jinja chat template**(多家族/视觉宏):现 Qwen 硬编码于
  openai.rs(单家族现实);模板串数据化(tokenizer ChatFormat 多轮
  字段)已落,server 消费迁移挂账。
- **`</think>` 后答案段偶发空/N/A**(L3 known-bug):格式约束任务的
  模型行为,判据已分级(soft 断言);深挖另案。

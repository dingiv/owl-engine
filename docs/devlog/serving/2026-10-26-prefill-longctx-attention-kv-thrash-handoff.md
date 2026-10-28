# 长上下文 prefill 交接：FlashInfer attention 与 200k KV 重灌

> 2026-10-26。对象 = `qwen38_27b-3080_tp2`（2×RTX 3080 20G，TP2，27B AWQ，
> FP8 KV，DFlash2 d7，`prefill_chunk=2048`）。
>
> 结论先说：
>
> - **30k–100k 的 prefill 下降主要是 FlashInfer attention 的 O(ctx) 成本**，
>   Marlin GEMM / AR / GDN 基本是固定成本。
> - **200k 的 13 s TTFT 不是 attention 单独造成**：其中约 **9.3 s** 是
>   “整条前缀 KV 先下沉 host、再整条回灌设备”的抖动。
> - 两个问题必须分开修：
>   1. **准入/复用账**：不要再把马上要复用的 cache 链当成“额外 floor”先赶走；
>   2. **attention kernel**：在 100k→200k 上评估 FlashInfer 的 kernel 配置与
>      是否需要 KVMem 选择/稀疏化。
>
> 当前服务：
>
> ```text
> 8135 PID 852986
> spec=depth7 / snap_max=2 / leave_gap
> 非 perf 构建（生产基线）
> 日志: /tmp/owl-8135-final2.log
> ```
>
> 工作区：`main@9c2cf739`，clean，ahead 41，未 push。

---

## 1. 测量口径

### 1.1 文本增量协议

同一 session：

1. seed 一条长文本（冷 prefill）；
2. append 约 800 token 的新 user 文本；
3. `max_tokens=1`，`temperature=0`，`enable_thinking=false`；
4. 记录引擎每轮日志中的 `prefill_ms / prefill_tokens`。

这测的是“历史已缓存、只 prefill 新尾巴”的真实 agent 形状。

### 1.2 CUPTI

```bash
cargo build --release -p server --features perf,cupti
./target/release/server ... --perf-level gpu
curl -s 'http://127.0.0.1:8135/debug/perf/epoch/gpu.kern'
```

⚠️ CUPTI 必须在 **server 启动时** `gpu_armed`；启动后再
`/debug/perf/level/gpu` 只会改 host 级别，不会武装 CUPTI。

### 1.3 perf tag 现状

- 支持 `off / counters / host / gpu` 全局分级；
- 支持查询端按 tag 前缀过滤；
- **不支持**像 `/debug/log/tags` 那样在记录端按 tag allow/deny 动态开关。

本次定位不需要 per-tag gate；`gpu.kern.*` 已经够用。

---

## 2. 实测曲线

### 2.1 冷启动全量 prefill

| prompt | prefill | 吞吐 |
|---:|---:|---:|
| 3k | 1.88 s | 1638 t/s |
| 9k | 5.45 s | 1690 t/s |
| 18k | 11.27 s | 1636 t/s |
| 23k | 14.28 s | 1614 t/s |
| 30k | 19.81 s | 1543 t/s |
| 61k | 43.94 s | 1390 t/s |
| 99k | 81.26 s | 1222 t/s |
| 199k | 208.61 s | 953 t/s |

冷启动全量 prefill 仍然在 1–1.7k 量级，随 ctx 缓慢下降。

### 2.2 增量 ~800 token

| ctx | 新增 token | prefill | 吞吐 |
|---:|---:|---:|---:|
| 3.9k | 828 | 549 ms | **1508 t/s** |
| 10k | 831 | 605 ms | **1374 t/s** |
| 19k | 798 | 627 ms | **1273 t/s** |
| 23.8k | 809 | 671 ms | **1206 t/s** |
| 31.4k | 813 | 712 ms | **1142 t/s** |
| 61.9k | 796 | 834 ms | **954 t/s** |
| 100.1k | 807 | 1074 ms | **751 t/s** |
| 199.6k | 816 | 3739 ms | **218 t/s** |

100k 之前是平滑的 attention 曲线；200k 出现明显断层。

---

## 3. CUPTI：固定成本 vs attention

### 31k / 813 tok

```text
FlashInfer prefill   296 ms
Marlin W4A16 GEMM    654 ms
AR/barrier/stage     195 ms
GDN recurrence        86 ms
```

### 100k / 807 tok

```text
FlashInfer prefill   954 ms   ← 3.2×
Marlin W4A16 GEMM    658 ms   ← 不变
AR/barrier/stage     210 ms   ← 不变
GDN recurrence        86 ms   ← 不变
```

（以上为两 rank 的 kernel 时间合计。）

结论：

- **Marlin GEMM 是固定成本**：每轮都要把 27B W4A16 权重吃一遍；
- **AR / GDN 基本固定**；
- **100k 时 FlashInfer 已占 GPU 时间约 48%**；
- 200k 时 attention 会进一步放大，粗略外推 200k 增量 prefill
  本身约 3–4 s（实测 3.739 s）。

---

## 4. 200k 的真正大头：KV 下沉/回灌

### 4.1 现场

```text
prompt=199,568
cached=198,752（99.6%）
prefill=3.739 s
TTFT=13.014 s
```

即 **9.3 s 不在 prefill 计时内**。

turn3 用 debug 日志抓到：

```text
[DEBUG] [store] 钩子(Admit) 判 out=1 in=0
         ⇒ 搬:KV 下沉 6236 驱逐 0 溢盘 0 回灌 0 ...

[DEBUG] [store] 段边界回灌 6236 块
         (宿主→设备;一条命令; ...)
```

即：

```text
6236 块 × 0.5 MiB ≈ 3.1 GiB/rank
两 rank 往返 ≈ 6.2 GiB
```

### 4.2 为什么是 200k

设备池：

```text
8192 页 × 0.5 MiB = 4 GiB/rank
```

200k 上下文：

```text
199k / 32 ≈ 6236 页
```

但准入面按“整个 prompt 的 floor”再要一次：

```text
cache 链 6236 页
+ 新 turn floor 6238 页
= 12474 页 > 8192 页
```

于是准入先下沉 cache 链，复用阶段又整条回灌。

100k 不触发：

```text
3130 + 3130 = 6260 < 8192
```

这就是 200k 的悬崖。

> 这是**准入/复用账的重复计算**，不是“上下文太大装不下”。
> 链本身 6236 页是装得下的；问题是把同一条链同时算成 cache 和 active floor。

---

## 5. 代码定位

### 5.1 准入面

`crates/engine/src/running.rs::step_need`

```rust
if self.turns.active.is_empty() {
    if let Some(t) = self.turns.queue.front() {
        let (sid, plen) = (t.session_id, t.prompt_ids.len());
        let dev = {
            let (hit, recall) = self.bank.rel.match_prefix(&t.prompt_ids)?;
            Self::reuse_face(&self.bank.rel, &hit, &recall, plen, page).1
        };
        let have = session.block_table.len();
        return Need::kv(slots_for(plen) - dev - have, NeedWhy::Admit);
    }
}
```

疑点：

- `dev` 只是 **matched 的 device 驻留块数**；
- 若命中链在 host（或 `reuse_face` 因快照边界退回），`dev` 会偏小；
- `slots_for(plen)` 却按整个 prompt 申请 floor；
- 结果：准入认为“需要新页”，把命中的 cache 链下沉。

### 5.2 调度面

`crates/engine/src/scheduler.rs::plan_prefix_reuse`

```text
match_prefix(prompt)
  → reuse_face(...)
  → 保留 m 页
  → pending_loads（host → device）
  → session.cached_len = m * page
```

同一个前缀在 `step_need` 和 `plan_prefix_reuse` 里被算了两次，
而且两次的 `reuse_face` 结果不一定一致。

### 5.3 回灌执行面

`crates/engine/src/running.rs::store_load`

```text
ensure_host_resident
pool_housekeeping(Need::kv(Grow))
plan_onload
swap_all_ranks(In)
commit_onload
写回 session.block_table
```

这就是 6236 块回灌的落点。

### 5.4 200k 现场需要的下一层取证

在 `step_need` 入口打印：

```text
plen
matched_blocks
reuse_face m
dev (hit.slots[..m].is_some count)
have
free_pages
need
```

同一轮在 `plan_prefix_reuse` 再打印一次，比较：

```text
why step_need dev=0 / need=6238
why schedule m=6211 / cached=198752
```

这是下一步第一件事。

---

## 6. 建议的修复方向

### P0：禁止对“本轮将命中的前缀链”做预下沉

候选方案（按优先级）：

1. **复用同一个 reuse face**
   - `step_need` 与 `schedule` 共用一份匹配结果/快照判定；
   - 不要在准入时用一套“device 计数”，调度时再用另一套。

2. **准入扣除命中链的 floor**
   - 对 queued turn 先算 `matched` + `reuse_face m`；
   - 需要的新页 = `slots_for(plen) - m - have`；
   - 若链在 host，仍然要回灌，但**不能先把同一批 host 内容 evict 掉**。

3. **保护前缀链**
   - 准入/保洁的 `protect` 集合加入本轮 queued prompt 的命中链；
   - 禁止 N6/Admit 把即将复用的 cache leaves 当普通冷缓存下沉。

4. **重排准入与复用**
   - 如果可行，先做前缀匹配/复用决议，再做 capacity 预留；
   - 至少把“本轮会复用”的块标记为 reserved，而不是先 evict 再 load。

### P1：600–1000 行级 instrumentation

在 `step_need` / `plan_prefix_reuse` / `store_load` 加：

```text
prefill.reuse.matched
prefill.reuse.face
prefill.reuse.dev
prefill.reuse.need
store.load.blocks
store.load.bytes
```

只对 `plen > 128k` 或 `NeedWhy::Admit` 打，避免日志爆炸。

### P2：attention kernel 评估

CUPTI 已经证明 FlashInfer 是 100k 的主变量。下一步：

1. 用 CUPTI 在 100k/200k 逐层看 `BatchPrefillWithPagedKVCacheKernel`：
   - 每层耗时是否均匀；
   - page size / GQA / causal 配置是否退化；
   - 是否有 split-K / tile 配置可调。
2. 对比 `prefill` chunk = 512/1024/2048 在 100k 下的 attention kernel 时间
   （注意：总 FLOPs 不变，重点看 tile / occupancy）。
3. 评估 KVMem `select_decode` / 部分驻留是否能把 attention 从 O(ctx)
   降到 O(selected)；质量门另立。
4. 如果 kernel 已接近硬件上限，产品层只能选择：
   - 限制单轮新 token / 上下文；
   - 分层缓存 + 压缩；
   - 换更强算力或更稀疏的 attention。

---

## 7. 验收线

### 200k KV 抖动

```text
同一 200k 会话连续追加 3 轮：
  不应出现“整链 6236 块下沉 + 回灌”
  begin/backfill 时间 < 1 s
  TTFT ≈ prefill + 1s 以内
```

建议门：

```text
200k append TTFT ≤ 6s（修复前 13s）
200k prefill 本身不劣化
cached ≥ 99%
无 Xid / 无错误
```

### attention

先建立基线，不先定不切实际的目标：

```text
100k append ≥ 当前 751 t/s
200k append 的 FlashInfer GPU 时间可归因到 kernel 配置/算法
```

若 kernel 已到硬件上限，再决定是否走 KVMem/压缩/产品限制。

### 回归

```text
cargo test -p owl-engine --lib
gpu_image_evict_gate
gpu_image_evict_spec_identity_gate
长上下文 100k/200k 手工回放
```

---

## 8. 复现命令

### 8.1 长文本协议

```bash
# 30k
python3 /tmp/bench_prefill.py 40000

# 100k
python3 /tmp/bench_prefill.py 130000

# 200k
python3 /tmp/bench_prefill.py 260000
```

（`/tmp/bench_prefill.py` 是本次临时脚本；下个会话可重写进
`apps/server/tests/` 或 perf-runs。它按 1.4 tok/word 估长。）

### 8.2 200k 回灌取证

```bash
curl -s http://127.0.0.1:8135/debug/log/level/debug
# 发一轮 200k append
grep -E "钩子\\(Admit\\)|段边界回灌|turn Some" /tmp/owl-8135-*.log | tail -30
curl -s http://127.0.0.1:8135/debug/log/level/info
```

### 8.3 CUPTI

```bash
cargo build --release -p server --features perf,cupti
./target/release/server ... --perf-level gpu
curl -s 'http://127.0.0.1:8135/debug/perf/epoch/gpu.kern'
curl -s 'http://127.0.0.1:8135/debug/perf/epoch/prefill'
```

---

## 9. 当前遗留/未决

> 2026-10-26 追加：§9/§10 是初版交接；**下一会话的施工口径以 §11 为准**
> （只做 P0 + P2，P1 暂缓）。

1. `step_need` 与 `plan_prefix_reuse` 的 reuse face 不一致，**未修**。
2. 200k KV 下沉/回灌抖动，**未修**。
3. 200k 的 prefill 218 t/s 里 attention 与可能的 KV 访问放大各占多少，
   尚未做 200k CUPTI（本轮只做了 100k CUPTI）。
4. `prefill.bnd` 偶发 0.2–0.4 s（31k 观察到 371 ms，100k 为 30 ms；
   `snap_max=4` 不改善），暂列为次级问题，不是 200k 主因。
5. 视觉塔缓存已工作；长上下文问题与 VLM 塔重算无关。
6. 未新增/提交代码改动；本文件为交接文档，待下个会话施工。

---

## 10. 一句话交接

> **100k 以内慢在 FlashInfer attention（Marlin/AR/GDN 是固定成本）；
> 200k 的 13 s TTFT 主要是准入先把整条前缀 KV 下沉 host、再整条回灌的
> 重复算账，9 s 级。先修准入/复用账，再评估 attention kernel。**

---

# 11. 2026-10-26 追加交接：下一会话只做 P0 + P2，P1 暂缓

> 用户裁决（2026-10-26）：
>
> - **落地 P0（200k KV 下沉/回灌）与 P2（精确 attention 核）**；
> - **P1（capped f16 shadow）暂缓**：它会引入两遍 attention + LSE merge +
>   页表分叉，代码复杂度上升过快；且当前 f16 FI 承载核数值不一致，P1
>   没有可靠地基；
> - **不做滑动窗口/KVMem 组合**（质量不可接受；KVMem 留作以后长上下文路线）。

## 11.1 本轮新增的关键事实（先看这段）

### A. 200k TTFT 仍然被 KV 抖动支配

2026-10-26 在 8135（3080 TP2，leave_gap，spec d7）复测：

```text
prompt=198,743
cached=196,608 (98.9%)
TTFT=5.406 s
verify round=46.9 ms（decode burst 本身仍 47ms 量级）
```

之前交接的 199,568 / 99.6% / 13.014s 是同一形状的更重一档。  
**5.4–13s 的 TTFT 与 attention 无关；5.4s 这一档尚未抓 debug 日志，但
cached=98.9% + 同样在首 token 前发生，按已有现场判据仍是准入把 cache 链
先下沉、复用阶段再整链回灌。**

### B. 100k 的 attention 仍严格线性，单卡效率约 56–66% 墙

3090 单层 FI（T=807，fp8，kNHD）：

| ctx | ms/层 | 备注 |
|---:|---:|---|
| 32k | 12.6 | |
| 64k | 23.3 | |
| 100k | 36.3 | |
| 200k | 73.1 | |

T=2048 @100k：93.9 ms/层。  
即 O(ctx) 没有拐点；100k 的 prefill 下降主要是 attention，不是 Marlin/AR/GDN。

decode/verify 也走 FI（`fi-graph.toml`：decode/verify_arm="fi"），但形状是
T=1/8 + 2 rank 各半，关注点不同：

| ctx | spec verify round | 说明 |
|---:|---:|---|
| 30k | 38.4 ms | 基本平 |
| 100k | 42.3 ms | +10% |
| 200k | 46.9 ms | +22%；首 token 前 5.4s 仍是 KV 抖动 |

⇒ P2 的 f16 承载核只用于 **prefill**；decode/verify 保持 fp8，不能因为
P2 改动把它们的轮时/输出带崩。

### C. f16 承载核的原始性能很好，但数值不对

同机、同形状、同迭代，f16 shadow（dense kNHD，**含 dequant**）vs fp8：

| 形状 | FP8 | F16 shadow(+dq) | 原始加速 |
|---|---:|---:|---:|
| 32k / T=807 | 12.582 ms | 9.872 ms | 1.274× |
| 64k / T=807 | 23.330 | 18.598 | 1.254× |
| 100k / T=807 | 36.324 | 29.361 | 1.237× |
| 200k / T=807 | 73.115 | 59.139 | 1.236× |
| 100k / T=2048 | 93.875 | 71.993 | 1.304× |

dequant（fp8→f16，含 K+V）实测：

```text
32k   0.223 ms/层
64k   0.441 ms/层
100k  0.688 ms/层
200k  1.373 ms/层
```

即 dequant 只占 1–2%，**f16 shadow 的性能假设成立**。  
（纯 f16 核本身在 100k 约 28.9 ms/层；上表多出的就是 dequant。）

但 **f16 FI 输出与 fp8 不一致**：

```text
T=1  (cta=16): max|Δ| = 0
T=2  (cta=16): max|Δ| = 0
T=4  (cta=64): max|Δ| = 0.25
T=8  (cta=64): max|Δ| = 0.25
T=512(cta=64): max|Δ| = 0.53125
```

- 直接 f16 物理池（不经过 shadow）也差 0.25；
- identity 页表、输出 padding、独立 `kv_bytes=2` plan、强制关闭
  `USE_SOFTMAX_VO_SPLIT` 都试过，**无变化**；
- f16 主池的 `k_shape_kv = [nb,hkv,hd/x,page,x]` 是 classic HND，
  而 FI adapter 硬编码 `QKVLayout::kNHD` —— 这是一个独立的结构性不匹配；
  但即使喂 NHD f16 池，CTA64 仍然不等价。

⇒ **P2 的第一件事不是调优，而是审计/修正 f16 FI 的 CTA64 路径。**
在它修好之前，P1 不能落地。

### D. P1 的 shadow 实验资产（保留，不合并）

工作树：

```text
packages/owl-worktrees/f16-prefill-shadow
branch: feat/f16-prefill-shadow
base:   9c2cf739
状态:   已实现、dequant 全绿、f16 parity 红；未提交、未合并
```

已实现：

- `owl_kv_fp8_to_f16_kernel` / `owl_fi_dequant_fp8_to_f16`
  （fp8 物理页 → dense f16 逻辑页，4 元素/线程）；
- `owl_fi_prefill_plan2(..., kv_bytes)`（显式 f16 plan，不读 env）；
- `FiRuntime` 的 `fi.prefill_shadow` + identity 页表 + `PlanKey.kv_bytes`；
- `[kernel].prefill_shadow` 配置面；
- `crates/kernels/tests/fi_shadow_probe.rs`（dequant 对拍 + fp8/f16 parity
  + bench 模式）。

dequant 实测：

```text
bad=0/102,400,000  bad_v=0/102,400,000  worst=0
```

> 这些资产是 P1 的已验证部分；但 **不要合主干**，直到 P2 的 f16 核通过
> parity。

## 11.2 P0：200k KV 下沉/回灌（下一会话第一优先）

### 目标

同一 200k 会话连续追加 3 轮：

```text
不再出现“整链 6236 块下沉 + 整链回灌”
begin/backfill 时间 < 1 s
TTFT ≈ prefill + ≤1 s
200k append TTFT ≤ 6 s（修复前 13s）
200k prefill 本身不劣化
cached ≥ 99%
无 Xid / 无错误
```

### 已知现场

```text
prompt=199,568
cached=198,752 (99.6%)
prefill=3.739 s
TTFT=13.014 s
```

debug 日志：

```text
[DEBUG] [store] 钩子(Admit) 判 out=1 in=0
         ⇒ 搬:KV 下沉 6236 ...

[DEBUG] [store] 段边界回灌 6236 块
```

设备池：8192 页 × 0.5 MiB = 4 GiB/rank。  
200k 链 6236 页；准入按整个 prompt floor 又算 6238 页：

```text
6236 + 6238 = 12474 > 8192
```

100k 不触发：3130 + 3130 = 6260 < 8192。

### 关键代码

```text
crates/engine/src/running.rs::step_need
crates/engine/src/scheduler.rs::plan_prefix_reuse
crates/engine/src/scheduler.rs::reuse_face
crates/engine/src/running.rs::store_load
```

### 施工步骤

1. **先打同一把尺的 instrumentation**（不要先改逻辑）：
   - `step_need` 入口：`plen / matched_blocks / reuse_face m / dev_slots /
     have / free_pages / need / why=Admit`；
   - `plan_prefix_reuse`：同一组字段 + `m / cached_len / pending_loads`；
   - `store_load`：`blocks / bytes / 方向`；
   - 只对 `plen > 128k` 或 `NeedWhy::Admit` 打印/计数，避免日志爆炸。
2. **比较两处 reuse face**：
   - `step_need` 的 `dev` 只数 device 驻留块；
   - `schedule` 的 `m` 可能因快照边界不同；
   - 若 `m != dev`，先统一成同一份匹配结果/快照判定，禁止两套账。
3. **准入扣除命中链 floor**：
   - 对 queued turn 先算 `matched` + `reuse_face m`；
   - `need = slots_for(plen) - m - have`；
   - 不允许把“本轮将复用的 cache 链”当额外 floor 先 evict。
4. **保护前缀链**：
   - 准入/保洁的 `protect` 集合加入 queued prompt 的命中链；
   - 禁止 N6/Admit 把即将复用的 cache leaves 当普通冷缓存下沉。
5. **重排准入与复用**（若前两步不够）：
   - 先做前缀匹配/复用决议，再做 capacity 预留；
   - 至少把“本轮会复用”的块标记 reserved，而不是先 evict 再 load。

### 回归

```text
cargo test -p owl-engine --lib -- --test-threads=1
gpu_image_evict_gate
长上下文 100k / 200k 手工回放
```

### 复现命令

```bash
curl -s http://127.0.0.1:8135/debug/log/level/debug
# 发一轮 200k append
grep -E "钩子\\(Admit\\)|段边界回灌|store|turn Some" /tmp/owl-8135-*.log | tail -50
curl -s http://127.0.0.1:8135/debug/log/level/info
```

---

## 11.3 P2：精确 attention 核（P0 之后/并行）

### 目标

- **不改语义、不丢 token、不做窗口/KVMem**；
- 先修 f16 FI 的 CTA64 数值，或换一个已知正确的 f16 承载核；
- 目标 attention **1.2–1.4×**；
- 仍然不改变 O(ctx)，只压常数。

### P2-A：审计 f16 FI 的 CTA64（首选，成本最低）

复现入口：

```bash
cd packages/owl-worktrees/f16-prefill-shadow
OWL_TEST_DEVICE=42:00.0 FI_SHADOW_T=8 FI_SHADOW_CTX=256 \
  FI_SHADOW_IDX=1 FI_SHADOW_HOST=1 \
  cargo test --release -p owl-kernels --features device,flashinfer \
  --test fi_shadow_probe -- --test-threads=1 --nocapture
```

已知：

- T=1/2（cta16）逐位一致；
- T=4/8/512（cta64）开始差；
- 直接 f16 池也差 ⇒ 不是 shadow/dequant；
- f16 主池 classic HND vs adapter kNHD 是独立问题；
- 强制 `USE_SOFTMAX_VO_SPLIT=false` 无变化。

审计顺序：

1. 用 host 参考固定 T=8/ctx=256 的期望输出（不要只比 fp8，避免两臂一起错）；
2. 检查 `cta_tile_q=64` 下 f16 的：
   - `USE_VO_SPLIT` / `NUM_WARPS_KV`；
   - `USE_KV_SHARED_SMEM`；
   - `kRepackActive`（f16 为 false，走 `produce_kv` + native `ldmatrix`）；
   - `qo_upper_bound` / GQA packed 行掩码；
3. 用同一份 K/V、同一 q，分别跑 cta16 与 cta64，定位第一处行/head 分歧；
4. 若确认是 FlashInfer 上游 hd256+GQA6+CTA64 的 f16 缺陷：
   - 优先修 adapter/模板；
   - 修不动则换核（P2-B）。

### P2 负结果（别再重挖）

本轮已实测排除：

| 试验 | 结果 |
|---|---|
| 关闭 FP8→BF16 repack（hd256） | 40.36 vs 37.29 ms，**慢 8%** |
| page 32 → 64 | f16 29.16 / fp8 36.74 ms，**中性** |
| page 32 → 16 | f16 29.02 / fp8 37.70 ms，**更差** |
| KV 布局 NHD → HND | f16 28.91 vs 28.87；fp8 36.33 vs 36.22，**中性** |
| 强制 f16 关闭 `USE_SOFTMAX_VO_SPLIT` | 输出差不变 |

⇒ 不要从 page/layout/repack 开新战场；P2 的靶子是 **f16 CTA64 数值**，
或者换核。

### P2-B：换 f16 承载核（备选）

候选：

- vLLM FA2 sm80 hdim256（历史单测：f16 比旧 FI 快约 1.8×；
  **无 fp8 路径**，需要 f16 shadow/池）；
- 其他已知正确的 f16 paged prefill 核。

约束：

- 需要 paged KV / causal / GQA6 / hd256 / page32；
- 需要 LSE/输出与现有 FI 契约对齐；
- 需要 greedy 输出逐字门；
- 如果只能 f16，P1 的 capped shadow 才有意义。

### P2 验收

```text
f16 FI vs host / fp8：max|Δ| 与现有门同级（不得 0.25 级）
同 prompt greedy 输出逐字一致
100k / 200k attention A/B：目标 1.2–1.4×
decode/verify 不回归（它们仍走 fp8）
无 Xid
```

---

## 11.4 P1：capped f16 shadow（暂缓，设计留档）

### 为什么暂缓

1. **代码复杂度**：需要两遍 attention（f16 shadow 页 + fp8 其余页）+
   LSE merge + 两套页表 + 页选择/驱逐；
2. **承载核未过**：f16 FI CTA64 数值不对，P1 没有可靠地基；
3. **收益饱和**：即使 256MB 上限，1M 也只有 ~1.03×，不是改变斜率；
4. 当前 P0 的 TTFT 收益远大于 P1；先修 P0。

### 设计要点（将来直接照此接）

- **上限**：256 MiB/rank；
  - TP2（hkv=2, hd=256）：`256MiB / 2048B = 131,072 token`；
  - 单 3090（hkv=4）：`65,536 token`。
- **语义**：只 shadow 最多 cap 个 token 的 K/V；其余仍走 FP8，
  **两边都参与 attention，merge LSE**；不丢 token。
- **选页**：最近 cap 页（ring）；不需要打分/KVMem；
- **页表**：
  - shadow：dense 逻辑页 + identity；
  - fp8：原始物理页；
- **性能上限**（f16 1.24×，不含 merge）：
  - ≤128k：1.24×；
  - 200k：~1.14×；
  - 262k：~1.10×；
  - 1M：~1.03×；
- **实施顺序**：P2 f16 核通过 → dequant/页表 → 两遍 + merge → cap/驱逐。

> 工作树 `feat/f16-prefill-shadow` 已有 dequant、plan2、shadow 骨架；
> 不要重写，先复用；但先修 P2。

---

## 11.5 当前工作区状态（2026-10-26 追加时）

主树 `packages/owl`：

```text
M crates/kernels/tests/fi_prefill_probe.rs   # 长上下文/f16 探针（未提交）
M docs/README.md
?? docs/devlog/serving/2026-10-26-prefill-longctx-attention-kv-thrash-handoff.md
```

- `fi_prefill_probe.rs` 增加了 `fi_prefill_longctx_probe` 与
  f16 长上下文探针；用于 100k/200k 核级 A/B；
- 生产路径未改；
- 8135 仍运行（PID 852986，spec d7，leave_gap，非 perf 构建）。

工作树 `packages/owl-worktrees/f16-prefill-shadow`：

```text
branch: feat/f16-prefill-shadow
base:   9c2cf739
状态:   未提交、未合并；f16 parity 红
```

### 下次会话的起手顺序

1. 读本文 §11；
2. 做 P0（先 instrumentation，再改账）；
3. P0 验收通过后，做 P2-A（f16 CTA64 审计）；
4. P2 通过后，再决定 P1 是否重启；
5. 全程不引入滑动窗口/KVMem 组合。

### 一句话

> **下一会话只做 P0 + P2：先把 200k 的“下沉 6236 / 回灌 6236”账修掉；
> 再审计 f16 FI 的 CTA64 数值，或换一个正确的 f16 承载核。P1 的 capped
> shadow 暂缓——它需要两遍 attention + merge，且当前 f16 核不过门。**

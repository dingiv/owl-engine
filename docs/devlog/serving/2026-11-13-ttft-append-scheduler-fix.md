# append TTFT：先修“快照切碎”，再谈 prefill 图（2026-11-13）

> 关联：`2026-11-13-ttft-prefill-handoff.md`（上一轮把问题归为“prefill 固定
> host 税”，本轮把账拆细：**先被页对齐/消息边界切成了两块**）。
>
> worktree：`packages/owl-worktrees/session-kv-snap`
> 分支：`feat/session-kv-snap`（基于 `main@39ed54e5`，全部未提交）

## 0. 一句话

agent append（30k 文本 + ~100 字符）的 prefill **195ms → 105ms**，
client TTFT **235ms → 145ms**；带 3 图场景 prefill **542ms → 456ms**，
TTFT **794ms → 611ms**。缓存命中不变（99.7%/99.5%）。
仍未到用户要求的 <100ms：剩下的 105ms 是**单块固定税**（GPU attention
≈38ms + host ≈50ms），下一步才是 prefill 图 / split-K。

---

## 1. 病根：不是“92 tok 一块”，而是“80 + 2 两块”

`prefill_align_tail`（缺省 true）在**最后一段**把 chunk 切到最后一个整页
边界，目的是让该页有 GDN 快照。agent append 的 82 tok 尾巴被切成：

```text
30528 .. 30608 (80 tok)  ← 页对齐块
30608 .. 30610 ( 2 tok)  ← 纯固定税尾巴
```

`prefill.chunk` 的 n=4 = **2 块 × 2 rank**；每块固定税 ~100ms，于是一个
82-token 追加变成 ~195ms。`snap_msg_min_span=0` 还会让块内 2~3 个消息
边界继续切，最坏 3~4 块。

> 上一轮 `prefill_chunk=128` 实验“append 不变”也是同一原因：改 chunk 大小
> 不影响“最后一段被页对齐再切一刀”。

## 2. 修法（`crates/engine/src/sched/scheduler.rs`）

新增常量：

```rust
const ALIGN_TAIL_MIN_REMAINING: usize = 1024;
```

`prefill_step_chunk_msg` 里：

- `fed > 0 && step == remaining && remaining <= 1024` ⇒ **`small_tail`**；
- `small_tail` 时**跳过页对齐切分**（`fed` 是页边界时前缀已有锚点，
  再对齐只买到 `last_boundary - fed` 个 token 的下轮复用，不值一块固定税）；
- `small_tail` 时**跳过消息边界切分**（82 tok 尾巴里的 2~3 个边界不再
  把一块打碎）；
- `fed == 0`（冷 prompt）**保持旧行为**：仍切出第一个页边界快照；
- `step < remaining`（非末块）仍按消息边界切，不丢中段锚点；
- 剩余 >1024 时仍按旧行为对齐（下轮少重算 ~1k token，值得多一块）。

单测更新/新增在 `prefill_tail_merge_tests`（5/5 绿）。

## 3. 实测（3080 对，30k 文本 base + 100 字符 append，max_tokens=8）

| 轮次 | client TTFT | server prefill | cache |
|---|---:|---:|---:|
| 修前 append1 | 234 ms | 195 ms / 82 tok | 30528/30610 |
| 修前 append2 | 251 ms | 211 ms / 103 tok | 30592/30695 |
| **修后 append1** | **145 ms** | **105 ms / 82 tok** | 30528/30610 |
| **修后 append2** | **165 ms** | **126 ms / 167 tok** | 30528/30695 |
| **修后 append3** | **189 ms** | **150 ms / 252 tok** | 30528/30780 |
| 修后 append4 | 152 ms | 110 ms / 81 tok | 30784/30865 |

- 每轮都是**一块**（`prefill.chunk` n=2 = 1 块 × 2 rank）。
- 缓存命中率保持 99.2~99.7%；偶尔累计到 252 tok 仍是一块（<1024）。
- 3 图场景：`prefill 542 → 456ms`，client TTFT `794 → 611ms`（视觉
  attention ~350ms 仍在；切碎税已去掉）。

## 4. 单块 105ms 的拆解（CUPTI / 真机）

- GPU `owl_int8_prefill_paged`：**2.38ms/层/rank × 16 层 = 38ms**（两 rank
  并行；CUPTI sum 76ms 是两 rank 之和）。
- GDN chunked ~9ms、Marlin/cublas/elementwise ~15ms。
- 余下 ~45ms 是 host 发射/表上传/解释器 DAG 遍历。
- client TTFT 比 server prefill 多 ~40ms：tokenize 30k prompt + HTTP +
  首 token 写出；prefill 要进 <60ms 才能整体 <100ms。

### 已排除的廉价刀

| 刀 | 结果 |
|---|---|
| `pool.block_pool_bytes=256MiB`（累计批 + 块池） | append 141→142ms，无变化；host 不是客户端往返 |
| `pool.kv_dtype=fp8` | 104ms，与 int8 持平；不是量化核问题 |
| `runtime.int8_prefill_sync=false` | 106ms，无变化；末块前全设备 sync 不是主项（该键已于 2026-11-16 删除；同日根因修复后栅栏也已删除，见 1k 门 §6.6；此行保留当时记录） |
| `prefill_chunk=128` | 已证伪（上一轮）：append 不变，cold 30k 5× 慢 |

## 5. 下一步：单块 105ms 的硬仗

### P0 小 chunk prefill 图（首选）

目标：把 16 层 × ~500 条 host 发射压成 1 条 `graph_launch`，预计
105 → 55~65ms，client TTFT 进入 95~105ms 区间。

**int8 专属阻断（本轮新发现）**：

- `owl_int8_prefill_paged` 的 `L`（总 KV 长）与 `base_pos` 是**发射标量**，
  捕获时会被烘死；append 的 base 每轮变（30528、30695…），一张图不能
  服务不同 base。
- 老 `prefill_plan.rs` 的桶图只服务 FI（FI 的 `kv_len/indptr` 在设备表里，
  所以能跨 base 复用）；int8 不能直接套。
- 最小改法：给 int8 核加 **dyn 变体**（`const int* L_ptr, base_ptr`，
  核首读），桶图把这两个当 `InputSlot`；其余 `T=t_pad`、`bt/kv_slots/
  kv_lens/gdn_table` 仍走输入槽/持久槽。需要同时恢复 dump 页 + `t_pad/
  t_real` 分离（老桶代码里有现成实现）。
- 若不想动核：只对 **base==0 的冷首块**建图（append 不命中），收益不在
  agent 场景，不建议。

### P1 attention split-K

`owl_int8_prefill_paged` 在 L=30k、T=82 时只有 `ceil(82/64)×Hq` 个 CTA，
3080 对只有 ~24 CTA，远低于 68 SM。把 KV 维切 4 片 + partial/reduce，
预计 38 → 10~15ms；与 P0 叠加可把单块压到 40~50ms。参考现成
`cu/attn/prefill_split_f16.cu` 与 `owl_int8_decode_*_split` 的 partial/stat
格式。

### P2 图像 KV attention（~350ms）

3 图 append 的 456ms 里 ~350ms 是 92 q × 36k KV（含 6120 图像 token）的
int8 attention。与 P1 同源；split-K 对图像段同样有效。

## 6. 验证命令

```bash
# 调度纯函数
GDN_TRITON_BUILD=0 CARGO_TARGET_DIR=/home/div/Documents/codes/packages/owl/target \
  cargo test -p owl-engine --lib prefill_tail_merge_tests
# 5 passed

# 构建
GDN_TRITON_PY=/home/div/Documents/codes/packages/owl/crates/kernels/.venv/bin/python \
CARGO_TARGET_DIR=/home/div/Documents/codes/packages/owl/target \
  cargo build --release -p server

# 文本 append bench
python3 /tmp/ttft_text_bench.py /tmp/owl-text-36k.json 4 <tag>
```

运行态（本轮结束）：

| 项 | 值 |
|---|---|
| 8135 server | PID 见 `pgrep -af 'target/release/server -p profiles/qwen38_27b-3080'` |
| binary | `/home/div/Documents/codes/packages/owl/target/release/server` |
| profile | `profiles/qwen38_27b-3080_tp2.toml`（默认 int8 / sync 开） |
| log | `/tmp/owl-8135-final.log` |
| 3090 server | PID 201670，未动 |

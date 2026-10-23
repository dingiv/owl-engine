# TP2 allreduce：owl vs SGLang 实现差（2026-10-23）

> **落地修正（见 §6）**：上面的 9.3µs 是 **10KB** 口径；owl verify AR 实际是
> **80KB**（m=8 × hidden 5120 × f16）。AR-2 落地后 80KB 核时 **19.4µs**，
> 端到端 **−3.7~−4.5%**，不是按 10KB 估的 8%。但“固定成本高、协议该简化”
> 这个判断成立。
>
> 结论先行：**owl 的 AR 不是“数据腿走完了”，而是每调用固定成本太高。**
> 2×3080、10KB f16、同一对卡：
> - SGLang v1 `CustomAllreduce`（world=2 一阶段）：**9.3 µs/call**（含 D2D copy）
> - owl `ar_fused + push`：**26.3 µs/call**（nsys kernel 时长，不含 launch gap）
>
> AR 每 rank 每轮约 **162 次**（nsys：5208 kernel / 16 轮 / 2 rank）。
> 若固定成本从 26.3 降到 9.3 µs，理论省
> `162 × 17 µs ≈ 2.77 ms/rank/轮`，对 33.9ms round 约 **8%**。
> 这不是 Marlin/GDN 那种 0.2ms 级收益。

## 1. 基准口径

### SGLang v1

- 卡：2×3080（CUDA ordinal 2/3，`CUDA_VISIBLE_DEVICES=2,3`）。
- 后端：`gloo` 控制面 + `CustomAllreduce`（v1，CUDA IPC）。
- 调用：`ops.all_reduce(ar._ptr, inp, out, ar.buffer_ptrs[rank], ar.max_size)`，
  10KB f16，预热 20，测 1000 次，CUDA Event 计时；含 `cudaMemcpyAsync` 到注册 staging。
- 实测：rank0 **9.28 µs**，rank1 **9.29 µs**；5KB 9.24µs，20KB 10.09µs。
- SGLang v2 JIT 不适用：`can_use_custom_all_reduce_v2()` 要求 full NVLink，
  本机双 3080 是 PCIe P2P，会退回 v1。

### owl

- `tp_head_i8_gate`，K7，N=32，DFlash2 d7，`OWL_TP_PUSH=1 OWL_TP_AR_FUSED=1`，
  nsys `--cuda-graph-trace=node`；
- decode 窗口（最后一次 `vllm_chunked_prefill` 之后）：
  - `owl_comm_ar_fused_f16` + `bf16` = **137.07ms / 5208 次 = 26.32 µs/次**
  - 同窗口总 kernel 890.76ms。
- 对照：`ar_fused` 不 push 148.58ms（28.53µs/次）；
  两核 v1 165.44ms（31.77µs/AR，按 5208 次折算）。

> SGLang 的 9.3µs **包含**一次 D2D copy；owl 的 26.3µs **只是 kernel**。
> 所以差距是保守估计。

## 2. SGLang 到底怎么做的

### v1：world=2 固定一阶段

`python/sglang/srt/distributed/device_communicators/custom_all_reduce.py`
→ `custom_all_reduce_ops.py`
→ `python/sglang/kernels/aot/csrc/allreduce/custom_all_reduce.cu`
→ `custom_all_reduce.cuh`。

入口 `ops.all_reduce`：

1. 若输入不是注册 buffer：`cudaMemcpyAsync` 到本 rank 的注册 buffer；
2. 启动 `cross_device_reduce_1stage`：
   - `multi_gpu_barrier<ngpus, true>`：每个 block 的前 `ngpus` 个线程各推一个
     release flag，等对端 acquire flag；然后 `__syncthreads`；
   - 每线程直接读 **所有 rank 的注册 buffer**，reduce 后写 `out`；
   - `multi_gpu_barrier<ngpus, false>`：同样一次 flag 往返，保护 buffer 复用。
3. 没有 `produced/consumed` 信用、没有 `wait_done` 全局放行、没有
   “最后一块负责 rendezvous”。
4. `world_size_ == 2` 时**无条件走一阶段**（`custom_all_reduce.cuh` 的
   `REDUCE_CASE`），不看 `full_nvlink`。
5. 10KB 时 `blocks = min(36, ceil(640/512)) = 2`；5KB 是 1 block，
   实测仍是 9.24µs ⇒ **固定成本不在 block 数**。

### v2：更激进，但本机用不上

`custom_all_reduce_v2.py` 的 `1shot_push` 用 Lamport 逐向量标记，
`num_push_blocks = num_sm`，只支持 full NVLink。它是未来方向，但不是
当前 3080 P2P 的可用参照。

## 3. owl 的一核 fused 还在多做什么

`crates/kernels/cu/comm/ar_fused.cu`（push 档）：

```text
credit 等 peer.consumed
→ 写 peer 槽（10KB）
→ __threadfence_system() 全线程
→ __syncthreads()
→ 最后一块 atomicAdd(blocks_done)
→ 发布 own.produced
→ 自旋等 peer.produced（尾部 rendezvous）
→ 写 own.wait_done 放行全 block
→ 全 block 自旋等 wait_done
→ __threadfence_system()
→ __syncthreads()
→ 读 own 槽 + 加到 payload
→ __threadfence_system()
→ __syncthreads()
→ 最后一块 atomicAdd(blocks_done)
→ 发布 own.consumed
```

对比 SGLang：

| 项 | SGLang v1 一阶段 | owl `ar_fused` push |
|---|---|---|
| 数据阶段 | copy 到注册 buffer + 一次 reduce | 写 peer + 读 own + 加 |
| 同步 | 起/止两次 per-block barrier | credit + 尾部 rendezvous + wait_done + 起/止 |
| 全局 system fence | 只有 flag 线程 release/acquire | **全线程 `__threadfence_system()` ×3** |
| 最后一块职责 | 无 | `blocks_done`、`produced`、`wait_done`、`consumed` |
| 复用保护 | 起/止 barrier | `produced/consumed` 信用 + 双槽 |
| 固定成本 | ~9µs | ~26µs |

判断：**owl 的 `produced/consumed` 信用和尾部 rendezvous 是为“异步复用/多槽”
设计的；但当前 graph 回放是同流串行的，每个 AR 都有 SGLang 式起/止 barrier
就足够。** 这套协议在 10KB 小包上就是纯固定税。

## 4. 建议的 AR-2（小包专用）

不要直接改现役 `ar_fused`；新增一条小包路径，保留大包/预填充原路：

1. 只对 `bytes <= 256KiB`、graph 内、非 fp8 的 AR 生效；
2. 单 staging 槽；push 写 peer；
3. 起 barrier：`__syncthreads` → 线程 0 `st.release.sys` → 自旋 `ld.acquire.sys`；
4. 读 own 槽 + 加 payload；
5. 止 barrier：同样一次 release/acquire，保护下一轮覆盖；
6. **不要** `produced/consumed/wait_done/blocks_done`；
7. **不要**全线程 `__threadfence_system()`；按 SGLang 的 release/acquire flag 语义；
8. 先做单 block（10KB 够用）；SGLang 5KB 单 block 也 9.2µs；
9. 大包/prefill 继续走现役 `ar_stage/ar_reduce`（含 f8 腿），不混协议。

预期：AR 137ms → ~50ms/16 轮，round 约 −2.7ms/rank（≈8%）。
验收：token 逐位不变 + `tp.ar` 路径计数 + 3 次重复 A/B + 32K fixed-tail。

## 5. 下一步

1. 先写 AR-2 独立 kernel + 小包 dispatch（config opt-in，默认关）；
2. 与 `ar_fused+push` 同参 A/B；不对就回退；
3. AR-2 稳定后再谈 GDN fold/tap 输出槽。


## 6. 落地结果（2026-10-23 晚）

AR-2 最终采用 **v1 相对游标 + 双槽乒乓、单 block、无尾部 rendezvous**：

- header 48/56 = AR-2 专用 `produced`/`consumed`（与 v1/fused 的 0/8 分开；
  否则 f8 prefill 的 v1 每 AR 把 produced +2，AR2 奇偶立即错位）；
- graph 捕获成功后把本端 header 48..64 清零并 sync，消除 warmup 差拍；
- 数据腿固定写本端/读对端，不依赖 `tp_push`，peer 只读别名即可；
- 80KB verify AR **19.4µs**（nsys；不是 10KB 的 9.3µs）。

**独立算子台 `ar_channel_bench`（图内回放，N=64，f16，K=200）**：

| 载荷 | v1 图内 | **AR-2 图内** | 提升 |
|---:|---:|---:|---:|
| 10KB | 21.92µs | **10.81µs** | −50.7% |
| 80KB | 30.56µs | **20.29µs** | −33.6% |
| 160KB | 39.39µs | **29.49µs** | −25.1% |
| 256KB | 49.23µs | **41.71µs** | −15.3% |

> 直接算子台与 nsys 的 19.4µs 对得上（20.29 vs 19.4，差在图/量法）。

**1-stage 臂也已加**（对称 256KiB slot + peer 别名；slot 清零，只作计时/路径
存在性，不替代 L1 数值门）：

| 载荷 | v1 图内 | AR-2 图内 | **1-stage 图内** | 1-stage vs AR2 |
|---:|---:|---:|---:|---:|
| 10KB | 21.9µs | **10.8µs** | 12.5µs | AR2 更快 |
| 80KB | 30.6µs | 20.2µs | **18.9µs** | −6.6% |
| 160KB | 39.4µs | 29.6µs | **26.9µs** | −9.1% |
| 256KB | 49.2µs | 40.1µs | **34.7µs** | −13.6% |

> 结论：**1-stage 在中大包更快，10KB 反而输给 AR2**；而且它的前提是
> GEMM 直接写对称槽。27B 生产档历史上从未走到该路径（§60.2），所以
> “开 `tp_one_stage=true`”目前不是即插即用。真要做，需改 payload 分配/
> GEMM 输出到对称 arena，再把 1-stage 的旧协议换成 AR2 的轻协议。

### 与 SGLang 同尺寸对比（同一对 3080）

SGLang v1 eager（含 `cudaMemcpyAsync` 到注册 buffer）实测：

| 载荷 | SGLang v1 | owl AR2（图内） | owl 1-stage（图内） |
|---:|---:|---:|---:|
| 10KB | **9.4µs** | 10.8µs | 12.5µs |
| 80KB | **15.6µs** | 20.2µs | 18.9µs |
| 160KB | **23.1µs** | 29.6µs | 26.9µs |
| 256KB | **32.1µs** | 40.1µs | 34.7µs |

口径注意：SGLang 9.4/15.6 是 **eager + copy**，owl 是 **graph 设备时延**；
所以 SGLang graph 只会更快。**owl 已进入同一量级，但还没有完全追平**：
10KB 差 ~15%，80KB AR2 差 ~29%（1-stage 差 ~21%），256KB 1-stage 差 ~8%。

受控 A/B（K7，N=192，token 逐位相同）：

| 档 | round | verify | propose |
|---|---:|---:|---:|
| fused | 34.586ms | 27.002ms | 5.875ms |
| **AR-2** | **33.316ms（−3.7%）** | **25.975ms（−3.8%）** | **5.674ms（−3.4%）** |
| f8 fused | 34.392ms | 26.978ms | 5.685ms |
| **f8 + AR-2** | **32.968ms（−4.1%）** | **25.764ms（−4.5%）** | **5.533ms（−2.7%）** |

32K fixed-tail（f8，N=64）：AR-2 verify **67.399ms**、round **78.910ms**；
同二进制 fused+push 基线 verify **68.012ms**、round **80.193ms**，token
**逐位相同**。

生产档当前 `tp_push=true`（外部改动，本刀未动）：push+AR-2 K7 N=192
round **33.708ms**、verify **26.250ms**，token 逐位相同。push 与 AR2 的
收益部分重叠（fused+push 33.907/26.532）；最大 AR 收益仍是 AR2-only
（33.316/25.975），后续若要压 decode 应单列 push=false A/B。

> 结论修正：SGLang 9.3µs 是 **10KB** 口径；owl verify AR 是 **80KB**
> （m=8 × hidden 5120 × f16），所以 AR-2 只能做到 19.4µs，端到端约 **−4%**，
> 不是最初按 10KB 估的 8%。但相对 fused 仍是实打实的 3.7~4.5%。


## 7. 多 block AR-2 试刀：否决（2026-10-23）

试了“全局 `blocks_done` + 最后一块发布 + block0 单 poller + 本地 wait 令牌”：

| 载荷 | 单 block AR-2 | 2~16 block AR-2 |
|---:|---:|---:|
| 10KB | **10.8µs** | 12.7~13.4µs |
| 80KB | **20.2µs** | 28.5~31.6µs |
| 160KB | **29.6µs** | 42.9~48.3µs |
| 256KB | **40.2µs** | 50.8~52.2µs |

结论：**朴素多 block 反而更慢**。全局 `blocks_done`/最后一块发布/
单 poller 的同步税超过了多 block 数据面收益；SGLang 的 10 block 快，
靠的是 **per-block signal plane**（每 block 自己的一对 peer flag），
不是“把 grid 调大”。

要做对，需要：
- 独立 per-block signal buffer（例如 64 block × 16B）及其 peer 别名；
- 每 block 自己的起/止 barrier；
- 仍保持单 block 的 fast path。

这是**中等施工量**，不是一行 grid 改动；当前生产因此维持 **单 block AR-2**。

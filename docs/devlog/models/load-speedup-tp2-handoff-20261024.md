# 装载提速交接：27B 冷盘 / TP2 并行 / 全局 vram 账清理（2026-10-24）

> 下一会话入口。**P0(全局 vram)已完成**；本文件保留下半成施工记录与剩余刀。
>
> 工作树：`packages/owl-worktrees/metrics-optimize`
> 分支：`perf/load-speed`（已 fast-forward main `87eba470`）
> 状态：已提交到 `perf/load-speed`（未 push）；无运行中的 27B server；4 张卡空闲。

---

## 0. 一句话现状（2026-10-24 晚更新）

- 单卡 27B AWQ：热启动 **9.23 → 7.49s**；冷启动 **18.96 → 10.2–10.8s**。
- TP2 27B（2×3080，含 vision + DFlash2）：**热 21.52 → 12.92s**，
  **冷 24.58 → 14.86s**。
- 双 rank 装载已是两条 OS 线程；rank0 预取 19.25GB。
- **P0 已完成**：`VramAccount/VramSnapshot` per-device，`GpuServer/GpuClient`
  共持；`boot_rank` 已 `tokio::join!` 并行，池容量两 rank 各读本卡快照。
- 最新定谳 1：双 rank 旧 `ShardedSource` 对**单段键也做整键 `Vec` 拷贝**
  （单 rank 直通）；按段数分流后主载 12.8→9.96s。
- 最新定谳 2：DFlash2 草稿的 `W4A16 packed_raw` 也先整键 `to_vec` 再切半；
  改成 `Resolved::View` + I32→U32 `copy_par` 后，草稿 4.19→**1.2–1.6s**，
  总启动再降到热 **12.92s** / 冷 **14.86s**。

---

## 1. 已完成

### 1.1 单卡冷盘：预取 + 重叠

根因：冷 page cache 时 mmap 缺页串行等盘，27B 键循环 17.13s；CPU 物化
（lm_head int8 2.87s、scales 2.27s、zeros 0.72s、packed_raw 1.32s）被 I/O
串行放大。

实现：

| 文件 | 改动 |
|---|---|
| `crates/models/src/module.rs` | `WeightSource::{prefetch, prefetch_finish}` 默认 no-op |
| `crates/models/src/formats/mmap.rs` | `Mmap` 保留 fd；`readahead()` 用后台 `pread` 填页缓存 |
| `crates/models/src/formats/awq.rs` | `AwqSource::prefetch_ranges`：只读索引/`linear_form`，收集 19.25GB 源区间；**2 条读流**按文件轮转；`Drop` join |
| `crates/models/src/formats/load.rs` | 装载前 `src.prefetch(keys)`；键循环结束 `prefetch_finish()`；单卡/TP2-rank0 才预取 |
| `crates/models/src/formats/awq_build.rs` | `rtn_i8_row_sym_bf16`：BF16 两遍直读，不建 5.1GB f32 中间态 |
| `crates/models/src/formats/awq.rs` | 对应逐位门 `i8_head_bf16_fused_matches_f32_chain` |

调参结论（冷载键循环）：

```text
1 读流 8.93s（I/O 不饱和）
2 读流 8.30–8.99s  ← 采用
3 读流 10.31s（NVMe 队列争用）
5 读流 10.44s
按字节贪心重排 9.84s（打乱文件顺序）
```

### 1.2 TP2 双 rank 并行装载

原路径（`apps/server/src/main.rs`）：

```text
load_shard(rank0).await
→ engine_of_tp(rank1)
→ load_shard(rank1).await
```

现路径：

```text
engine_of_tp(rank1)                    // 账本在装载前合流
→ std::thread::scope
   ├─ 线程0: current_thread runtime → load_shard(rank0)
   └─ 线程1: current_thread runtime → load_shard(rank1)
→ join → TpEngine::assemble
```

前置改动：

- `Weight.dtype: Cell<Dtype>` → `AtomicUsize`（Relaxed），`Weight` 手写 `Clone`；
  `Model: Send + Sync`，门 `model_is_send_sync_for_tp_parallel_load`。
- TP2 只让 **rank0** 预取全量源区间（两 rank 切片并集 = 全量）；rank1 不重复预取。

### 1.3 实测（同机 release）

单卡 `qwen38_27b-3090-w4a16.toml`：

| 口径 | 修复前 | 修复后 |
|---|---:|---:|
| 冷盘键循环 | 17.13s | 8.30–8.99s |
| 冷盘权重装载 | 18.23s | 9.42–10.16s |
| 冷盘启动到就绪 | 18.96s | 10.2–10.8s（3 次） |
| 热启动到就绪 | 9.23s | 8.20s |

TP2 `qwen38_27b-3080_tp2-w4a16.toml`（2×3080；vision + DFlash2）：

| 口径 | 串行（修复前） | 并行（修复后） |
|---|---:|---:|
| 分片装载（冷） | 31.44s | 15.42s |
| 启动到就绪（冷） | 37.86s | 24.58s |
| 分片装载（热） | 22.24s | 13.52s |
| 启动到就绪（热） | 26.14s | 21.52s |

TP2 热载两 rank 各约 12.36/12.68s；图捕获/装配仍约 7s 串行。

### 1.4 验证

```text
cargo check -p server                              ✓
cargo test -p owl-models --lib                     308/308 ✓
cargo test -p owl-engine --lib -- --test-threads=1 156/156 ✓（含 TP2 GGUF 门）
TP2 真机 boot + 只回答:OK                           ✓
```

---

## 2. P0（已完成，保留设计记录）：清理 `owl_shared::vram` 全局变量

> ✅ 2026-10-24 已落地：见 §5 与 `load-speedup.md` §10。下面保留原设计与
> 施工顺序作复核材料；当前决策路径已不读全局三元组。

### 2.1 现状

`crates/shared/src/vram.rs`：

```rust
static TOTAL: AtomicU64;
static LIVE: AtomicI64;
static FREE: AtomicU64;

pub fn total_set/total_bytes/live_add/live_bytes/free_set/free_bytes
```

文件头自述：**“全进程单引擎假设”**。TP2 两 rank 共享一个 `FREE`：

- `boot_rank` 里 `free_set(free)` 写全局；
- `StatePool::alloc` 读 `free_bytes()` 算池容量；
- `assemble_with_staging` 读 `live_bytes()` 算 `baseline_per_card`/`weights_per_card`；
- `apps/server`/`running.rs`/`state.rs`/`backends/cuda` 共 30+ 处读写。

两 rank 并行 `boot_rank` 时：

```text
rank0 free_set(rank0 卡free)
rank1 free_set(rank1 卡free)   // 覆盖 rank0
rank0 StatePool::alloc 读到 rank1 的 free
⇒ 池容量错、可能 OOM / 预算拒
```

所以当前 `tp.rs::assemble_with_staging` 仍 **rank0 boot_rank → rank1 boot_rank** 串行；
这也是 TP2 热载 21.5s 中约 7s 的图捕获/装配来源。

### 2.2 目标设计（下一会话施工）

**把全局三元组改成每设备/每 rank 显式的 `VramAccount`，由 `GpuServer` 持有，
经 handle 查询；引擎 boot 只收快照，不读 static。**

建议形态：

```rust
// crates/shared/src/vram.rs（或 backends/cuda 内更贴近设备）
pub struct VramAccount {
    total: AtomicU64,
    live: AtomicI64,
    free: AtomicU64,
}

#[derive(Clone, Copy)]
pub struct VramSnapshot { pub total: u64, pub live: u64, pub free: u64 }

impl VramAccount {
    pub fn total_set(&self, b: u64);
    pub fn live_add(&self, d: i64);
    pub fn free_set(&self, b: u64);
    pub fn snapshot(&self) -> VramSnapshot;
}
```

归属：

- 每个 `GpuServer`/`GpuClient`（每设备）持 `Arc<VramAccount>`；
- alloc/free/harvest 更新**本设备**的 account；
- `DeviceHandle` 暴露 `vram_snapshot()`；
- `StatePool::alloc` / `boot_rank` / `Engine::new` 收 `VramSnapshot`（或 `&VramAccount`），
  不再读 `owl_shared::vram::*`；
- `TpEngine::assemble_with_staging` 的 `baseline_per_card`/`weights_per_card`
  改为逐 rank 快照；不再 `live_bytes()/world`；
- 删除 `TOTAL/LIVE/FREE` 三个 static 与全部 free 函数；
- `apps/server`/测试/诊断日志改读对应 rank 的 snapshot。

### 2.3 施工顺序（避免大爆炸）

1. **S1 立类型**：`VramAccount`/`VramSnapshot` + 单测（并发 add/sub、快照）；
2. **S2 设备侧接管**：`GpuServer` 持有 account；`state.rs`/`harvest.rs` 的
   `total_set/free_set/live_add` 改成本设备 account；`DeviceHandle::vram_snapshot()`；
3. **S3 引擎消费快照**：`boot_rank`/`StatePool::alloc`/`Engine::new`/
   `assemble_with_staging` 参数化；删全局读取；
4. **S4 删 static**：全仓 `rg "vram::(total|live|free|.*_set)"` 清零；
   CPU/mock 用显式测试 account；
5. **S5 图捕获并行**：S3 后 `boot_rank` 的池分配/图捕获可安全并行（先 `tokio::join!`
   两 rank；再评估是否拆 pool/capture 两段）；目标 TP2 热载 21.5 → ~14–15s，
   冷载 24.6 → ~17–18s；
6. **S6 TP2 派生键**：确认两 rank 是否各自构建全量 `.scales/.zeros`（当前
   热载各 12s 的主疑点）；做 sharded derived build 或每 rank rayon 限流。

### 2.4 判据

- `rg "static (TOTAL|LIVE|FREE)" crates/shared/src/vram.rs` = 0；
- 全仓 `owl_shared::vram::` 只剩新 account 类型（或零）；
- TP2 两 rank 并行 `boot_rank`：池容量与串行逐档一致（日志 `[ctx-max]`/`cap` 对比）；
- 单卡冷/热不回归（≤10.8 / ≤8.5）；
- TP2 冷 ≤18s、热 ≤15s 为第一阶段目标；再追 10s 需 S6 + 图捕获并行；
- 0.8B/TP2 e2e + 27B 短生成 + 逐位/质量门。

---

## 3. 本次提交范围

`perf/load-speed` 共 24 个文件（8 个 loader 文件 + P0/per-device/TP2 装配 +
AVX2 + W4A16 零拷贝 + 2 个 profile + 3 份文档 + 本交接文档）。
已提交到 `perf/load-speed`；未 push。

---

## 4. 下一会话起手式

```bash
cd /home/div/Documents/codes/packages/owl-worktrees/metrics-optimize
git status
git log --oneline -3          # branch perf/load-speed（已含本次提交）
# main 已到 87eba470；先同步 main（预计只涉及 kv pad token，需解冲突后重跑）
git merge main

export GDN_TRITON_PY=/home/div/Documents/codes/packages/owl/crates/kernels/.venv/bin/python
cargo check -p server
```

单卡冷盘复现（会清 27B 模型 page cache）：

```bash
python3 - <<'PY'
import os, pathlib
root=pathlib.Path('/home/div/Documents/codes/models/cyankiwi/Qwen3.8-27B-AWQ-INT4')
for p in root.rglob('*'):
    if p.is_file():
        fd=os.open(p, os.O_RDONLY); os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED); os.close(fd)
PY
./target/release/server -p profiles/qwen38_27b-3090-w4a16.toml \
  --set server.bind=127.0.0.1:8143 --set parallel.device='["41:00.0"]' --log-level info
```

TP2 复现（2×3080；冷盘时同时清 27B + DFlash2 两个目录）：

```bash
./target/release/server -p profiles/qwen38_27b-3080_tp2-w4a16.toml \
  --set server.bind=127.0.0.1:8143 --log-level info
```

看这些行：

```text
[load] 预取源区间: ... (冷盘 I/O 与物化重叠)
[load] eval_load 分相:键循环(host) ...
[boot] TP2 rank0 装载 ...s
[boot] TP2 rank1 装载 ...s
[boot] TP2 分片装载 ...s
[boot] TP2 装配完成 ...s
[boot] 引擎装配(run)总计 ...s
```

---

## 5. 2026-10-24 续：P0 落地与二次提速

### 5.1 已完成

- `owl-iface::contract::{VramAccount, VramSnapshot}`；`GpuServer`/`GpuClient`
  共持 `Arc<VramAccount>`；`GpuCtx` 的 alloc/free/slab/peer/mem_info 全落本卡账。
- `DeviceClient::vram_snapshot()` + `DeviceHandle` 转发；`BootVram` 改快照；
  `PoolPlan.vram` 供 `pool_cap`/`[vram-gov]`，不再读全局。
- `VramGuard` 改读本设备账；全局 `vram` 函数仅留诊断聚合镜像。
- `TpEngine::assemble_with_staging`：两 rank 各取本 face 的池前 live 基线，
  `boot_rank` 用 `tokio::join!` 并行。
- DFlash2 草稿：`W4A16Source` 补 prefetch；停用逐键 `DONTNEED`(TP2 共享页缓存)；
  修 `ShardedSource::full_bytes` 的 `Ok(None)` 被丢弃后 `expect` 的潜在 panic。
- `rtn_i8_row_sym_bf16`：行 amax 改位型整数 max + AVX2 第二遍(运行期检测,
  标量回落;逐位门 `bf16_i8_row_avx2_matches_scalar_including_ties`)。
- **`ShardedSource` 按段数分流**：≤8 段直接写 dst（与单 rank 同路），
  >8 段才整键缓存。旧实现对单段 `packed_raw` 也 `vec![0; 整键]` 再拷，
  双 rank 半量反慢 10–15×；修后热启动 19.17→**15.57s**。
- profile 补 main `87eba470` 新增必填 `pool.image_evict_pad = "。"`。

### 5.2 实测

| 口径 | 修复前 | 现在 |
|---|---:|---:|
| 单卡热启动 | 7.75s | **7.49s** |
| TP2 热载 / 启动 | 13.52 / 21.52s | **9.85 / 12.92s** |
| TP2 冷载 / 启动 | 15.42 / 24.58s | **11.80 / 14.86s** |
| 主键热载(每 rank) | 11.5s | **8.1–9.9s** |
| 草稿热载 | 5.05s | **1.2–1.6s** |

冷盘清理目录：`models/cyankiwi/Qwen3.8-27B-AWQ-INT4` +
`models/syvai/Qwen3.8-27B-DFlash2-W4A16`(逐文件 fadvise DONTNEED)。

### 5.3 仍未闭环（按收益排序）

1. **主模型双 rank host 键循环**仍是绝对主项（热 ~9.9s / 冷 ~11.8s）；
   剩余 CPU 派生：`lm_head` int8、scales/zeros、多段 k 切键整键缓存。
2. 草稿已不是主项（~1.2–1.6s），无需再为它单开并行装载。
3. **`boot_rank` 真 OS 线程并行**：前置 = `slab_hint` 的 `METER/HINT` 按
   rank/设备隔离(当前是进程级单值，直接并行会串 hint)。
4. **`lm_head` 设备 int8 量化**：`owl_quant_i8_row_w_{scale,q}_bf16` 已有
   bench，接装载路径可再省 1–2s；需新增 `i8q/i8s` 设备派生臂。
5. `into_running` 的 `assemble_budget` 仍用全局 live/world(此时 boot 已结束,
   无并行写;非 P0)。彻底去全局需把两 rank `VramSnapshot` 传入。

### 5.4 风险/边界

- `tokio::join!` 是同一 async task 的并发，不是 CPU 真并行；图装配 CPU 仍
  共享执行线程。
- `slab_hint` 仍是进程级单值；在改 per-rank 前**不要**把 `boot_rank` 改 OS 线程。
- W4A16 草稿源不再 `DONTNEED`：保留 ~1.28GB clean page，内核可按压力回收。
- 全局 `vram` 镜像仍在(诊断用)，不要新增决策读取；后续可删。

---

## 6. 明确不做/暂不做

- 不把 TP2 两个 rank 全量预取（会读 2×19GB）。
- ~~不在全局 `FREE` 未清理前强行并行 `boot_rank`~~ —— 2026-10-24 已完成 per-device 账；但 `slab_hint` 仍全局，OS 线程 `boot_rank` 前必须先隔离它。
- 不为追 10s 把 TP2 图捕获关掉（turns 首步会付捕获税，且破坏生产形态）。
- 不把 `LoadedModel` 用 `unsafe impl Send` 绕过；已用 `AtomicUsize` 正解。
- 本文件只记录施工，不重复 `docs/devlog/models/load-speedup.md` §9/§10 的全部实验细节。

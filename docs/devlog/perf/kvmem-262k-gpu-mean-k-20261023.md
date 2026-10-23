# KVMem 262K：GPU Mean-K + 索引 D2H 批化（P0/P1）

> 工作树：`packages/owl-worktrees/metrics-optimize`
> 分支：`feat/metrics-optimize`
> 日期：2026-10-23
> 结论：**索引相位 86.9s → 310ms**；262K prefill 墙钟 362.6s → 347.5s
> （B 轴 46.5s → 39.4s）。索引不再是主项；262K 质量仍另案。

---

## 1. 做了什么

### P0：GPU 块级 Mean-K

新增两个核（同一份源）：

```text
crates/kernels/cu/elems/mean_k.cu
  owl_mean_k_block_f16
  owl_mean_k_block_fp8
```

语义：

```text
输入：主池 K 页（kNHD：[nb,page,hkv,hd]）+ 该块 page 个位置的 cos/sin
处理：逐 token 逆 rotate-half RoPE → 逐 (layer, kv_head, dim) 求均值
输出：[block][layer][hkv][hd] f16
```

- 池内 K **不回读**；D2H 只有索引体（27B 几何 32 KiB/块）。
- `phys`/`cos`/`sin` 打成一个 U8 辅助块一次上传，发射时用
  `Arg::BlockSlice` 切给核；输出块固定末参（槽序契约 4）。
- 逐层发射（16 层 = 16 发/批）；grid `(blocks, hkv)`，block `(hd)`。
- 逆 RoPE 用 `__fmul_rn/__fadd_rn` **显式禁 FMA**，与 host
  `MeanKCollector` 的 f32 运算逐位配对（缺省会差 1 ulp f16）。
- host 旧路径保留：`[kvmem].index_gpu = false`（golden/排障）。

声明面：

```text
crates/kernels/src/ops/elems.rs
  OWL_MEAN_K_BLOCK_F16 / OWL_MEAN_K_BLOCK_FP8
  PoolAccess::read(KV_MAIN, Knhd, F16 / F8E4M3)
```

store 侧新增扁平登记（免 N×hkv×dim 个 `MeanKEntry` 堆分配）：

```text
MeanKIndex::register_flat(block, Vec<u16>)
Rel::index_register_flat(...)
```

### P1：廉价边界不 D2H，完整召回前一次补齐

`recall_every > 1` 时，廉价边界只把 `(内容身份, 逻辑块, 物理页)` 攒进
`SelectRuntime.index_pending`，**不触发同步 D2H**；下一次完整召回（或末边界）
一次送进 `collect_index_for_blocks`。暂存块在 `reconcile_residency` 的
`keep` 名单里**受保护**，不会在下一次采集前被沉走。turn 收口兜底排空。

---

## 2. 关键实测：87s 不是 Mean-K 数学，而是逐边界 D2H 栅栏

只上 P0（GPU 核，但仍在每个边界 D2H）：

```text
boundaries=511 full=128 cheap=383
index=86978.7 qface=31810.2 score=2839.5 select=4450.7
view=1579.0 reconcile=35512.7 | total=46515.0ms
wall(客户端脚本)=362.56s
```

服务端 metrics：

```text
srv.op.owl_mean_k_block_fp8   4096 次 / 36.9ms
srv.wait.dtoh                  550 次 / 86570.9ms
srv.wait.dtoh_batch            128 次 / 29386.0ms
prefill.chunk                  513 次 / 226840.0ms
prefill.bnd                    512 次 / 133596.5ms
```

结论：核本身只花 **36.9ms**；`index` 的 86.9s 几乎全是
**每个廉价边界一次同步 D2H** 的等待。P0 单独不会降墙钟。

上 P1（128 个完整边界才 D2H）：

```text
boundaries=511 full=128 cheap=383
index=310.1 qface=31738.1 score=2785.0 select=4686.7
view=1389.4 reconcile=28563.2 | total=39388.8ms
wall(客户端脚本)=347.47s
```

服务端 metrics：

```text
srv.op.owl_mean_k_block_fp8   2048 次 / 22.4ms
srv.wait.dtoh                  167 次 / 189.4ms
srv.wait.dtoh_batch            128 次 / 29492.0ms
prefill.chunk                  513 次 / 305695.8ms
prefill.bnd                    512 次 / 39786.6ms
```

账：

```text
index       86,978.7 → 310.1 ms   (-99.6%)
B 轴 total  46,515.0 → 39,388.8 ms (-15.3%)
墙钟        362.56 → 347.47 s     (-15.1s, -4.2%)
```

`prefill.chunk` 从 226.8s 升到 305.7s、`prefill.bnd` 从 133.6s 降到 39.8s：
D2H 等待从边界段挪进了 chunk 收割段，但总墙钟下降 —— 批化把 3/4 的
同步点消掉了。

---

## 3. 验收

- [x] `mean_k_block_parity`：f16/e4m3 与 host 独立参考**逐位一致**
      （page32 / hkv4 / hd256 / half32 / 16 层）；
- [x] `gpu_08b_index_probe`：GPU 路径真实产出非零索引，
      且 `recall_every=4` 覆盖廉价边界暂存/take/末边界排空；
- [x] `cargo test -p owl-store --lib` 118 绿；
- [x] `cargo test -p owl-shared --lib` 122 绿（新增 `index_gpu` 全量档字段）；
- [x] 262K 零 IMA / 零 Xid；
- [ ] 262K coherent needle 仍失败（本次输出 `:`；80K view 对 262K 仍不够）；
- [x] 90K coherent needle 回归：`: BLUEBIRD-42`（80K view / 85.5K 池；
      `index=231.9ms`，175 个边界都是完整召回）。

---

## 3.5 Q 面 D2H 实验（负结果，已回退）

90K/r1 实测 Q 面：

```text
qface=36.16s
  q_toh=32.64s  ← face.dtoh_batch(16 层 × 末 64 行)
  q_dec=2.33s   q_derope=0.54s
```

原以为 Q D2H 是带宽问题（12.6MB/边界 × 175 = 2.2GB）。但：

- 裸 D2H 微基准（torch，3080）：13.2GB/s；16 片 786KB 也 12.9GB/s；
- 把 Q 读到 **capture_gdn_snap 之后**（索引 D2H 已排空 COMPUTE 尾）：
  `q_toh` 32.6s → **0.5s**，但 `index` 从 0.23s → **32.4s**；
- 墙钟 96.5s → 96.7s（**不变**）。

⇒ 这不是 Q 传输慢，而是**块边界第一次 D2H 在等本 chunk 的 compute 尾**。
把 Q 挪到后面只是把等待挪给 index。该实验已回退（保留 `qface 拆` 日志）。

**结论**：要再降 Q/index 的“等待”，只能减少同步点/做 step-level 重叠，
不能靠换 D2H 路径。

## 3.6 错峰调度：选择 r16 / 索引 r4 / 视图每边界

不是把 Q/索引/视图都堆到末尾，而是**三条节奏错开**：

| 任务 | 节奏 | 说明 |
|---|---|---|
| Q + 打分 + 选择 | 每 16 chunk（`recall_every`） | 刷新 keep 集；末边界强制完整 |
| Mean-K 索引采集 | 每 4 chunk（`index_cadence` 上限） | 暂存最多 64 页，必须装进池 |
| 视图 + 驻留调和 | **每边界** | 用冻结选择写视图/下沉；视图尾只有 1 chunk |

关键修正：**视图不能只在末尾写**。`past_view_of` 依赖 `self.sel.view()`；
中间不写视图 ⇒ 链出洞后 prefill 仍读全量链 ⇒ FI 页链超池（90K 实测
`nb_used=2688 > 2676`）。正确做法 = 每边界用**冻结选择**写视图
（`recall_pos=当前边界`，尾 1 chunk），这样 prefill 子集和池下限都成立。

### 90K（80K view / 85.5K 池）

```text
r1 : 96.5s  full=175 index=0.23s qface=36.2s
r4 : 90.4s  full=44  index=0.12s qface=9.24s
r16: 88.3s  full=11  index=6.01s qface=2.47s reconcile=7.40s
输出均为 : BLUEBIRD-42
```

### 262K（80K view / 85.5K 池）

```text
P1 r4  : 347.5s  full=128 index=0.31s qface=31.7s reconcile=28.6s
r16    : 324.8s  full=32  index=21.98s qface=8.52s reconcile=30.29s
输出 `**`（262K 质量问题未变；零 IMA/Xid）
```

`index` 上升是预期的：索引从“每 4 个选择边界”变成“每 4 个 chunk”，
且 P1 里 Q 先吸收了 compute 屏障（index 看似 0.31s）；r16 错峰后 Q 只在
32 个边界出现，index D2H 成了廉价边界的第一个同步点 ⇒ 22s 仍是
**边界 D2H 等 compute 尾**，不是 Mean-K 数学（GPU op 仍是毫秒级）。
总墙钟仍降 **22.7s（-6.5%）**：Q 的 23s 屏障被摊薄，收益大于 index 新增的等待。
`recall_every=16` + `index_cadence=4` 已写进 80K/262K profiles。

## 3.7 reconcile 拆账：90% 是 full 边界的 stage-in

262K r16 的 `reconcile=30.3s` 不是廉价边界“沉太多”；90K r16 实测：

```text
full 边界 need=192 块 stage-in（一次搬）
cheap 边界 sinkable=60 但 free=52 ≥ want=18 ⇒ 实际不沉
```

试过“按需下沉”（只沉到够下一 chunk + 本次回灌）：

```text
90K r16 原版 : wall 88.30s reconcile 7.40s
按需下沉     : wall 88.19s reconcile 7.31s   ← 无收益
```

⇒ 已回退。**瓶颈是 full 边界一次 stage-in 190+ 块**，不是廉价边界 sink。
下一步若要降 reconcile，必须做 **stage-in 预算/滞回（少换选中集）** 或
**异步/分批回灌并与下一 chunk 重叠**；单纯少沉无效。

## 4. 当前剩余主项（262K）

```text
prefill.chunk 305.7s ← 主体是模型 compute（不是 B 轴）
reconcile    30.3s   ← 召回点下沉/回灌（同步；当前最大 B 轴项）
qface         8.5s   ← 32 个完整边界，Q 屏障已摊薄
index        22.0s   ← 128 次索引 D2H（每 4 chunk），仍是 compute 屏障
view          1.0s
```

下一刀优先级（按“真能降墙钟”排序）：

1. **reconcile 的 stage-in 分批/滞回**：30.3s 主要是 full 边界一次回灌 190+ 块；
   不是廉价 sink。需要少换选中集（滞回）或把回灌摊到多个 chunk；
2. **index_cadence 4→6**：暂存 96 页，`workspace + 96 = 2656 ≤ 2676`；
   索引 D2H 次数 128→86，预计再省数秒（需 262K 复测）；
3. **262K 质量**：更大 view / compact re-RoPE，与索引无关；
4. ~~Q D2H 换路径~~：负结果，已回退；
5. ~~把所有任务压到末尾~~：视图尾/索引暂存会把池钉死，已用错峰节奏替代。

---

## 5. 复现

```bash
# 构建
GDN_TRITON_PY=/home/div/Documents/codes/packages/vllm-test/.venv/bin/python \
  cargo build --release -p server --features perf,cupti

# 启动（3090 41:00）
./target/release/server \
  -p profiles/qwen38_27b-3090-gguf-kvmem-262k.toml \
  --set 'server.bind="127.0.0.1:8145"' \
  --set 'parallel.device=["41:00.0"]' \
  --set 'probes.dflash_probe=true' \
  --log-level info

# 客户端
python3 /tmp/needle_coherent10_64.py http://127.0.0.1:8145 /tmp/out262k.txt 262000
```

数据目录：

```text
perf-runs/metrics-optimize/gguf-kvmem-262k-gpu-mean-k-prof/   # P0-only
perf-runs/metrics-optimize/gguf-kvmem-262k-gpu-mean-k-p1/     # P0+P1
perf-runs/metrics-optimize/gguf-kvmem-90k-80k-p1/             # 90K 质量回归
perf-runs/metrics-optimize/gguf-kvmem-262k-k4-qskip2/         # 旧 host 路径
```

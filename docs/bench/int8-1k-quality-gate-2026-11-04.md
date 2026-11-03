# INT8 KV 1k 文本质量门（2026-11-04）

> **最终结论（2026-11-04 结案）：1k 门已过。** 根因 = INT8 prefill
> 末块前缺一次设备同步；修复见 §6。A（`select_partial`）不受影响；
> B（视觉/`image_evict`）也在同一 sync 修复后 36/36 全绿（见 B 定谳文档）。
>
> §2–§5 是定位过程（含中途误判），最终定谳以 §6 为准。

## 1. 门定义

刚启动的 INT8 端点（`profiles/measurement/qwen38_27b-3090-int8.toml`，
`prefix_cache=true`、`prefill_chunk=2048`）上，单请求：

```bash
python3 tools/fixture/int8_text_gate.py --base http://127.0.0.1:8141
```

`needle_eval` 的 `--tokens 1000`（实际 prompt 1040 tok，depth 50%）
必须 `RETRIEVED`。默认只跑这一档；多档加 `--sizes 1000,8000,32000`。

## 2. 实测

| 配置 | 结果 |
|---|---|
| 生产默认：`prefix_cache=true` + 末段页对齐 + chunk 2048 | **MISSED** `answer="ZXCVBNM"` |
| `prefix_cache=false`（单块 1040） | RETRIEVED |
| `pool.prefill_align_tail=false`（单块 1040） | RETRIEVED |
| `prefill_chunk=512`（512+512+16） | RETRIEVED |
| `prefill_chunk=96`（96×10 + 80） | RETRIEVED |
| `prefill_chunk=64`（64×16 + 16） | RETRIEVED |
| `prefill_chunk=1024`（1024+16） | **MISSED** |
| `prefill_chunk=1536`（1536+544，2k prompt） | **MISSED** |
| FP8 同形（chunk 2048） | RETRIEVED |

默认档的 1040 被末段页对齐切成 **1024 + 16**；**1024 这个 chunk 是触发点**。
同一 server 上连续跑不同长度会受前缀缓存影响，**门必须 fresh server**。

## 3. 定因

- 不是 `select_partial`：不关选择层也复现。
- 不是 prefill 核数学：`int8_prefill_probe` 的 T=1024 / T=69+L=1093 长尾
  对照均 `cos>0.999999`。
- 同一 token、同一 slot（layer-0 K，page 32）在
  full T=1040 vs chunk=64 T=16 下：32768 B 中 13594 B 不同（41.5%），
  最大码值差 8。差异来自 **W4A8 投影 GEMM 随 prefill 批 T 变化**，
  再被 INT8 per-32-group K 量化放大；FP8 的逐元素指数更鲁棒。
- 视觉 prefill 在 chunk 2048/1024 失败、chunk≤96 通过；chunk 512 又失败
  （T=512 对视觉数据仍不稳）——同一根因，不能靠“固定 512”通吃。

## 4. 决策

1. （历史）**1k 门当时保持失败**；最终已过，见 §6。
2. **不把 `prefill_chunk` 改成 512**：120k 实测 99.0s vs chunk 2048 的
   44.2s（同一 A 配置），长上下文不可接受；且视觉 chunk 512 仍失败。
3. （历史）当时不删 `image_evict` 的 INT8 拒启；最终 sync 后已删，见 §6.5。
4. 转正前置（按优先级）：
   - **W4A8 投影 GEMM 行分块与批 T 解耦**（固定 M tile / 禁 split-K），
     使同一 token 的 K 在任意 chunk 下逐位一致；
   - 视觉档 K/V 走 FP8/FP16 高精度旁路；
   - 再补 1k/2k/8k/32k/120k fresh-server 门。

## 5. 对 A 的影响

A 的 `select_partial` 修法（`dense_int8_bt_values`）与 1k 回归无关；
A 的 8k/32k/64k/120k 验收仍成立。**1k 门拦的是 INT8 整体转正，不是
`select_partial` 机制。**

## 6. 2026-11-04 结案：1k 门已过，根因是 INT8 prefill 末块前缺同步

### 6.1 最终结论

1k 质量门 **已通过**（fresh server 连续 3/3 RETRIEVED）。根因不是
INT8 prefill 核、不是 W4A8 批形状、不是 GDN chunked，而是：

> **INT8 prefill 的非末块把 KV/GDN 写留在 COMPUTE 流上；末块的
> host 表/调度与跨流复用之间存在一个漏 event 的窗口。末块前加一次
> 全设备栅栏即修复。**

修复位置：`crates/engine/src/exec/prefill_chunk.rs`
`prefill_chunk_enqueue` 入口：

```rust
if is_last && base > 0 && self.pool.geom.kv_int8 {
    let face = self.graphs.session.face_mut();
    face.sync().await?;
}
```

每请求只付一次（末块前），不是每 chunk 税；FP8/非 INT8 不走此路。

### 6.2 验收（fresh server，`prefill_chunk=2048`）

| 场景 | 结果 |
|---|---|
| 1k needle | **RETRIEVED**，0.7s，3/3 fresh server |
| 2k needle | **RETRIEVED**，1.3s |
| 8k needle | **RETRIEVED**，3.8s |
| 32k needle | **RETRIEVED**，15.9s |
| 120k needle（A 档：select/index/partial） | **RETRIEVED**，86.2s |
| 短问答 `17*23` | `391`（A 档机制未回归） |

### 6.3 中途误判：W4A8 批形状是“真实低比特差”，不是门因

定位过程中曾观察到：

- chunk64 vs 2048 的 `s0.mlp` / `s0.gg` / `s0.gbeta` FNV 不同；
- Marlin `parallel = prob_m/64` 使同一行的 K 归约顺序随批 M 变；
- 固定 M=512 + GDN b/a 强制 GEMV 后，1k 答案从 `"ZXCVBNM"` 变成
  `"ZXCVBN12345"`（仍少一个 `M`）。

但这些都**不是门因**：

- `probes.prefill_bisect` / `live_diag` 的 state dump 会 `dtoh`，顺带
  同步设备 ⇒ 探针路径把门“治好”，造成假阳性；
- 固定 M=512 + GEMV 单独仍 MISSED；加末块 sync 后，**基线 W4A8 原样**
  也 RETRIEVED；
- 固定 M=512 会显著增加权重重读，已**全部回滚**；`c_tmp` 清零实验也
  未修门，回滚。

保留的只有：末块前 sync + 可选诊断面 `probes.prefill_bisect`
（默认 false，层间 taps，非生产）。

### 6.4 `llm_speedtest` 复测：sync 税 ≈ 0

开关：`[runtime] int8_prefill_sync`（缺省 true；false 仅 A/B，关掉会
重新丢 `M`）。同一 prompt nonce、`rounds=1`、fresh server：

**标准矩阵（512/1024/2048/4096/8192，输出 256）**

| 指标 | sync on | sync off | Δ |
|---|---|---|---|
| prefill avg | 2467.92 t/s | 2468.51 t/s | **−0.02%** |
| TTFT avg | 1325.11 ms | 1325.26 ms | −0.01% |
| decode avg | 42.54 t/s | 42.73 t/s | −0.45% |

**长上下文（32k/65k，输出 16）**

| 长度 | sync on prefill | sync off prefill | Δ |
|---|---|---|---|
| 32k | 2066.35 t/s | 2057.78 t/s | +0.42% |
| 65k | 1781.13 t/s | 1739.68 t/s | +2.38% |
| avg | 1923.74 t/s | 1898.73 t/s | +1.32% |

结论：**每请求一次的末块栅栏没有可测开销**；32k/65k 的 on/off 差在
逐次运行方差内（on 甚至略快）。sync 不是吞吐税。

### 6.5 仍开放

- 末块前 sync 是**一次每请求的栅栏**，不是最小根因；后续应找出
  具体漏掉的 stream event（KV/GDN 写 → 末块 host/调度/跨流复用），
  用 event 替代全设备 sync。
- B（视觉 / `image_evict`）已过：sync 后无 evict 36/36、带 evict
  36/36，slope≈1.0、hit@16px 100%；`engine.rs` 的 INT8 拒启已删。
- 文本质量门与 sync 税已闭环；转正仍待 B/视觉与更广质量集。

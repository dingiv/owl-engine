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

---

## 7. **更正（2026-11-04 晚）**：§3 的「W4A8 投影 GEMM 随批 T 变化」被证伪

§3 把"同一 token 的 K 在不同 chunking 下 41.5% 字节不同"归因于
**W4A8 投影 GEMM 随 prefill 批 T 变化**，并据此把"GEMM 行分块与批 T 解耦"
排成 §4 的第一前置。**新探针实测该归因不成立**：

`crates/kernels/tests/marlin_m_determinism.rs`（新增，1.5 s）：

- 形状 n=5120/k=5120 与 n=34816/k=5120（27B 的真实投影面）；
- 同一份权重、同一批激活，M ∈ {1, 8, 16, 64, 128, 256, 320, 512, 1024}，
  以 M=2048 为基准**逐位**比对输出 half；
- **差异 0 / 全等**，max|Δ| = 0.000000 —— 含跨 `thread_m_blocks` 档位表
  （M≤16 走 `{128,128,256}`，M≥64 走 `{64,256,256}`，thread_k 128 vs 64）
  与 **K-slice 分支**（M<320 时 `global_mn_tiles < gridDim.x`，走
  `global_reduce` 部分和再相加）两条路径。

原因：W4A8 的累加器是 **int32**（int16 尺度当整数乘进去），**整数加法精确且
与顺序无关**；per-token 的 a_scale / b_scale 只在**每个输出元素**上乘一次。
⇒ 任何 tiling / split / segment 都不会改变结果位型（只要不溢出，而生产
a8 因子已保证不溢出）。

**连带结论**：
1. §4「W4A8 GEMM 行分块与批 T 解耦」**不必做**（做了也不会解锁什么）。
2. 因此 **per-token K scale（+12~15% 核级）的闸门是纯精度，不是确定性**：
   KG=1 打穿 1k 门、KG=4 打穿 8k 门（3/3 确定性），与 chunking 无关。
   详见 `docs/devlog/kernels/int8-prefill-opt-community-2026-11-04.md` §7。
3. §3 观察到的 41.5% K 字节差异，最可能的解释是**当时的漏 event 窗口**
   （§6 已用末块前全设备 sync 修掉）造成的陈旧/重复 KV 写，而不是 GEMM。
   若将来还要追这条线，先在**已修 sync 的树**上复测那份差异。

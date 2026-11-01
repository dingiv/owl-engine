# FlashInfer FP16-PV 预填核移植（2026-11-02）

> 源：FlashAmpere `repos/vllm-ampere-optimized/patches/flashinfer_fp16pv/`
> （`0007-fp16-accum-pv-gated-flashinfer.patch`，FI 0.6.12；本仓按 0.7.2 的
> `prefill.cuh` 重锚，不直接套补丁）。
>
> 结论：**核级 f16 +20~26%，fp8 +12~20%；3090 单卡 60k e2e 约 +6.6%
>（四 seed 均值）/ speedtest +7.5%，输出逐字节一致。** 默认关，Cargo
> feature `fi-fp16pv` 显式开。

---

## 1. 做了什么

PV 累加器从 fp32（8 regs/fragment）改成纯 fp16 `o_acc`（uint32[4]
half-packed，4 regs/fragment）：

- `compute_sfm_v` 的 PV HMMA 从 `mma_sync_m16n16k16_row_col_f16f16f32`
  改成 `mma_sync_m16n16k16_row_col_f16f16f16`；
- `update_mdo_states` 的在线 softmax 重缩放用 half2 `__hmul2`；
- KV loop 结束后 `owl_materialize_o_frag` 把 half 累加器还原成 float
  `o_frag[8]`，后续 `threadblock_sync_mdo_states` / `transform_output` /
  `write_o_reg_gmem` **逐字不动**；
- `init_states` / 三个 device 函数（Single / Ragged / Paged）的累加器类型
  统一走 `OwlOFragT<KTraits>`。

收益主因不是 2× MMA，而是**寄存器/溢出缓解**：hd256 的 fp32
`o_frag` 把核推到 255-reg 上限并产生 stack spill；half 累加器把 spill
压下去，整条 attention kernel 都变快。

### 1.1 硬门

- 只对 `DTypeQ == half` 且 `DTypeKV == half` 或 1-byte（fp8/fp4）生效；
  **bf16 必须拒绝**（`f16f16f16` 会把 bf16 位型当 fp16 读，输出 garbage）。
- `USE_VO_SPLIT`（fp4 hd≥512）保持原 fp32 路径；owl 生产 hd256
  f16/fp8 预填是非 VO-split 路径。
- 设备码硬门：`__CUDA_ARCH__ == 860`。A100/A40/A6000 的 fp32 累加全速，
  fp16 累加只会更差；sm_89 未验证。**不要用 `>=860` 当判据。**
- 编译期开关 `OWL_FP16_PV`，由 Cargo feature `fi-fp16pv` 派生；默认 0。

### 1.2 构建接线

```
crates/kernels/Cargo.toml       fi-fp16pv = ["flashinfer"]
crates/backends/cuda/Cargo.toml fi-fp16pv = ["owl-kernels/fi-fp16pv"]
apps/server/Cargo.toml          fi-fp16pv = ["owl-cuda/fi-fp16pv"]
```

`build.rs` 只给主 FI 两个 TU（f16 臂 + fp8 臂）加
`-DOWL_FP16_PV=1`；ragged 视觉臂不加。构建：

```bash
cargo build --release -p server --features fi-fp16pv   # 缺省 arch-sm86
```

启动日志会打印：

```
[boot] kernel: FlashInfer FP16-PV = on (sm_86 GeForce build; f16/fp8 KV prefill)
```

不要把 `fi-fp16pv` 与多代际 fatbin / sm89 一起开：C++ 侧
`static_assert(__CUDA_ARCH__ == 860)` 会直接编译失败（这是故意的）。

---

## 2. 核级 A/B

3090 Ti（`0000:41:00.0`），24 q / 4 kv，hd256，page32，
`FI_SPLIT_KV=1`，eager plan，`fi_prefill_probe`。

### 2.1 fp8 KV

| T | ctx | 基线 fp32 PV | FP16-PV | 增益 |
|---:|---:|---:|---:|---:|
| 512 | 65536 | 18.45 ms | 16.45 ms | **+12.2%** |
| 2048 | 65536 | 60.81 ms | 50.48 ms | **+20.5%** |
| 2048 | 139264 | 130.55 ms | 108.73 ms | **+20.1%** |
| 8192 | 65536 | 212.42 ms | 177.78 ms | **+19.5%** |

### 2.2 f16 KV（`fi_prefill_longctx_f16_probe`）

| T | ctx | 基线 fp32 PV | FP16-PV | 增益 |
|---:|---:|---:|---:|---:|
| 512 | 65536 | 15.43 ms | 12.37 ms | **+24.7%** |
| 2048 | 65536 | 48.30 ms | 38.33 ms | **+26.0%** |
| 2048 | 139264 | 104.34 ms | 83.19 ms | **+25.4%** |
| 8192 | 65536 | 186.88 ms | 156.05 ms | **+19.8%** |

f16 臂接近 FlashAmpere 报告的 +25% op；fp8 臂因 V 反量化/额外路径摊薄，
仍有 +12~20%。

### 2.3 资源

hd256 `CTA_TILE_Q=64, NUM_MMA_KV=8` 的 paged 核：
`STACK 32 → 16`（`cuobjdump -res-usage`）。0.7.2 本身已比 0.6.12 少 spill，
所以这里不是“9→0”的戏剧性数字，但寄存器压力确实下降。

---

## 3. 数值

host oracle（fp8 KV，`FI_LONG_HOST=1` / `FI_PF_HOST=1`；host 是 fp32 参考）：

| T | ctx | max\|Δ\| | cos | mean\|Δ\| |
|---:|---:|---:|---:|---:|
| 1 | 3072 | 5.76e-4 | — | — |
| 8 | 3072 | 1.44e-3 | — | — |
| 54 | 3072 | 3.43e-3 | — | — |
| 128 | 3072 | 6.56e-3 | — | — |
| 512 | 3072 | 1.55e-2 | 0.999983712 | 2.37e-3 |
| 2048 | 3072 | 1.55e-2 | 0.999989456 | 1.82e-3 |
| 1 | 65536 | 4.39e-3 | — | — |
| 8 | 65536 | 9.91e-3 | 0.999993530 | 1.46e-3 |

绝对误差比 fp32 PV 大一个量级（half 累加的正常代价），但整段余弦
≥0.99998。探针回归门在该 feature 下改为
`cos ≥ 0.99995 && mean|Δ| ≤ 5e-3`；非该 feature 仍用原 `max|Δ| < 1e-2`。

---

## 4. e2e（3090 单卡，60k 纯文本）

启动：

```bash
./target/release/server -p profiles/qwen38_27b-3090.toml --log-level info \
  --set runtime.max_seq=70000 --set kernel.prefill_shadow=true \
  --set model.vision=false --set spec.depth=0 --set spec.lookup=false \
  --set pool.image_evict=false
```

### 4.1 speedtest 60k（1 轮，输出 8）

| 版本 | prefill t/s | TTFT | decode t/s |
|---|---:|---:|---:|
| 基线（main 二进制） | 1704.32 | 35206 ms | 45.72 |
| FP16-PV（feature） | 1831.51 | 32761 ms | 45.55 |

**prefill +7.46%，TTFT −6.94%**；decode 不变（prefill-only）。

### 4.1b 不开影子的纯 fp8 路径（3090 档缺省）

`--set kernel.prefill_shadow=false`，同 60k speedtest：

| 版本 | prefill t/s | TTFT |
|---|---:|---:|
| 基线 | 1562.51 | 38401 ms |
| FP16-PV | 1663.96 | 36060 ms |

**prefill +6.49%，TTFT −6.10%**。所以 3090 档即使维持
`prefill_shadow=false`，fp8 臂本身也能吃到约 6.5% 的 e2e 收益。

### 4.2 四 seed 引文复制（59.7k prompt，64 max tok，greedy）

同一句 19-token 引文，baseline / FP16-PV **四组全部逐字节一致**。

| seed | 基线 | FP16-PV |
|---:|---:|---:|
| 20261102 | 35.307 s | 33.074 s |
| 20261103 | 37.331 s | 35.139 s |
| 20261104 | 36.779 s | 34.523 s |
| 20261105 | 36.922 s | 34.506 s |
| **均值** | **36.585 s** | **34.311 s** |

**+6.63%**（60k prefill，含 e2e 噪声）。

### 4.3 needle 59,720 tok

| 版本 | 耗时 | 输出 |
|---|---:|---|
| 基线 | 36.359 s | `7391` |
| FP16-PV | 32.578 s | `7391` |

两者都正确检索；PV 冷启动快 11.6%。

---

## 5. 未做 / 风险

- **3080 TP2 未测**：8135 被外部 3080 服务占用，未停服。TP2 all-reduce
  会稀释 e2e；按 FlashAmpere 经验可能接近 0，需在 3080 生产档单独验。
- **只验证 sm86 GeForce**；sm89 / 数据中心卡不要开。
- **bf16 硬拒**；未来若 bf16 需要，先做 P/V cast 到 half，不能直接开。
- **fp8 KV 的 e2e 增益小于 f16**；生产档若 `prefill_shadow=false`，
  预期收益按 fp8 核的 +12~20% × attention 占比折算，约 3~5%。
- **3090 档默认 `prefill_shadow=false`**；本报告 e2e 显式开了 shadow，
  与 3080 生产档一致。若要 3090 也吃满，需同时评估 shadow 的显存/带宽税。
- 当前是 **Cargo feature**，不是运行时配置键；同一二进制不能按 profile
  切换。若将来要 A/B 共存，需要把 `OWL_FP16_PV` 提升为模板参数并编译
  两套 cubin（本次未做）。

---

## 6. 复现

```bash
cd packages/owl-worktrees/fi-fp16pv

# 核级（feature on）
GDN_TRITON_PY=/home/div/Documents/codes/packages/owl/crates/kernels/.venv/bin/python \
OWL_TEST_DEVICE=0000:41:00.0 FI_LONG_T=2048 FI_LONG_CTX=65536 FI_LONG_ITERS=3 \
cargo test --release -p owl-kernels \
  --features device,flashinfer,arch-sm86,fi-fp16pv \
  --test fi_prefill_probe fi_prefill_longctx_probe -- --test-threads=1 --nocapture

# host 对拍（fp8）
... FI_LONG_T=8 FI_LONG_CTX=65536 FI_LONG_HOST=1 FI_SPLIT_KV=1 FI_CUDA_GRAPH=0 ...

# e2e
cargo build --release -p server --features fi-fp16pv
./target/release/server -p profiles/qwen38_27b-3090.toml ...  # 见 §4
cd ../llm_speedtest && uv run cli ... --lengths 60000 --rounds 1 \
  --output-length 8 --temperature 0
```

---

## 7. 文件

| 文件 | 变更 |
|---|---|
| `crates/kernels/cu/flashinfer/include/flashinfer/attention/prefill.cuh` | FP16-PV 类型/helper/三个 device 函数 |
| `crates/kernels/build.rs` | `CARGO_FEATURE_FI_FP16PV` → `-DOWL_FP16_PV=1`（仅 f16/fp8 两 TU） |
| `crates/kernels/Cargo.toml` / `crates/backends/cuda/Cargo.toml` / `apps/server/Cargo.toml` | feature 透传 |
| `apps/server/src/main.rs` | boot 日志显式打印 feature 生效 |
| `crates/kernels/tests/fi_prefill_probe.rs` | cos/mean 指标 + PV 档回归门 |

> 工作树同时带着 P0 的 eager split-KV mask 修复（`owl_fi_attention_common.h`
> + probe 的 `FI_LONG_HOST`）。提交时建议拆成两笔：P0 修复 / FP16-PV。

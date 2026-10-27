# CUTLASS 4.8.1 — shared header family（**最小 TU 闭包**）

**Upstream:** [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass), workspace
`repos/cutlass` commit `91843b71` (v4.8.1), BSD-3-Clause. See `LICENSE.txt`.

**Include 方式**（两个消费方共用）：

```text
-Icrates/kernels/cu/cutlass/include
```

## Scope —— 514 个头（**先跑通态**；瘦身待办）

| 消费方 | 头数 | 内容 |
|---|---:|---|
| FlashAttention-2 hdim96（`../flash_attn/owl_fa2.cu`） | 146 | 原闭包，未改动 |
| NVFP4 on sm_120（`../nvfp4/`） | +368 | SM120 块缩放 mainloop + cooperative kernel + 默认 epilogue + **device adapter** |

> **为什么现在是 514 而不是 162**：2026-10-27 用户裁决「先跑通再瘦身」。
> 少头路线（自写启动器）在 `device_kernel` 可见性上卡住，改用 `GemmUniversalAdapter` 后一次跑通
> （4B 真权重 **463 GFLOPS @ M=512**，3.19× 现役 MMQ2）。
> 瘦身路线与实测头数见 `.local-state/perf-sm120.md` §8/§9：514 → 353（自写启动器）→ ~223（SM120 专用 builder）。

NVFP4 那 16 个（合计 324 KiB）：

```
gemm/collective/sm120_blockscaled_mma_tma.hpp   ← mainloop 本体（唯一要的族）
gemm/collective/collective_mma_decl.hpp         ← 主模板声明（上面的特化要有它才编得过）
gemm/dispatch_policy.hpp                        ← MainloopSm120TmaWarpSpecializedBlockScaled 等标签
pipeline/{pipeline,sm100_pipeline,sm90_pipeline}.hpp  ← TMA 多级流水线
detail/sm100_blockscaled_layout.hpp             ← SF(e4m3/16) 布局助手（host 打包 + kernel 读缩放）
detail/{collective,layout}.hpp · detail/collective/sm103_kernel_type.hpp
arch/{arch,barrier,memory_sm75}.h
gemm/{gemm.h,gemm_enumerated_types.h} · gemm_coord.h
```

`closure.txt` = 并集（`nvcc -M` 生成，勿手工增减）。

## 纪律：不许走"便利入口"

**实测过两条路的代价差 10 倍**（同一个 NVFP4 GEMM，`nvcc -M` 实测）：

| 取法 | 头数 | 说明 |
|---|---:|---|
| `CollectiveBuilder` + `GemmUniversalAdapter`（官方样例写法） | **513** | 会打开 **所有** 架构的 collective、2.x 旧 API、全部 epilogue/fusion 变体（其中 sm90/sm100/sm103 共 121 个、2.x 旧层 104 个，真正给 sm120 的只有 20 个） |
| `CollectiveMma` mainloop + **自写 epilogue / tile 调度** | **+16** | 本目录采用 |

因此本族的做法是：**mainloop 用上游特化；epilogue、tile 调度、host 适配按上游源码抄进 `../nvfp4/` 再简化**（删掉 grouped / sparse / stream-K / fusion 分支），
不许为了省事把 `CollectiveBuilder` 拉进来。

## 维护纪律（A4）

- 新族要更多头 ⇒ 从**同一 pin** 取文件，用 `nvcc -M` 对**能编译的最小 TU**重算闭包，
  再重生成 `closure.txt`；**不要**把整棵 CUTLASS 拷进来（全量 838 头 / 28 MiB）。
- 全量 CUTLASS 的 tests / examples / tools **不入树**。
- 升级 pin = 重跑两个 TU 的闭包 + 重生成清单 + 更新 `THIRD_PARTY.md` 与本节。

## 版本注意

`repos/cutlass` 现为 **v4.8.1**（`91843b71`，原登记 v4.8.0 `78e46228`）。v4.8.1 起
`sm120_blockscaled_mma_tma.hpp` 与其依赖成熟可用；两个 TU 都在该 pin 上编过验证。

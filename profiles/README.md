# profiles —— 组件化配置

配置只有一个概念：**profile = 一份 TOML**。`full.toml` 是唯一全量 schema；
所有可启动档都是 `extends` 薄 leaf。

## 规则

- 新档只写差异，不要复制整份配置：

```toml
extends = [
  "full.toml",
  "parts/model/27b-awq.toml",
  "parts/hardware/3090-single.toml",
  "parts/kernel/fi-graph.toml",
  "parts/pool/fp8.toml",
]
```

- `extends` 路径相对**声明它的文件**；数组按序覆盖，后者胜。
- 合并后仍缺键 ⇒ 拒启；`--print-config` 是唯一放行半份档的迁移工具。
- 卡位只走 `[parallel]`；`device` 用 BDF / UUID / 名字子串，**禁 CUDA 序号**。
- **实验变体一律用 `--set`，不新建入口档**（见下"唯二常驻档门"）。

## 唯二常驻档

2026-10-24 用户裁决：入口档只留两个硬件形状，其余 27 个历史档删除
（收敛记录与找回方式：`roadmap.local/profiles-consolidation-2026-10-24.md`）。

| 文件 | 形状 |
|---|---|
| `qwen38_27b-3090.toml` | 单 3090 Ti / 27B AWQ-INT4 + 视觉 / DFlash2 d7 / FI+图 / fp8 KV / **max_seq 79872** |
| `qwen38_27b-3080_tp2.toml` | 2×3080 20G TP2 / 27B AWQ-INT4 + 视觉 / DFlash2 d7 / fp8 KV / **max_seq 262144** |

这条裁决被**测试钉住**：`crates/shared/src/profiles.rs::入口档面只允许唯二常驻档`
—— 往 `profiles/` 顶层再丢一个 `.toml` 就会红。

`max_seq=79872` 是 **2026-10-25 全特性拉满档**（视觉塔 + DFlash2 d7 + fp8 KV +
1 快照）的实测最大：1 快照下 boot cap=90025，但**可用上限由 runtime 压力地板
决定**（1GiB + 单块工作集 ≈64.6MB）：79872 时驱动余量 1106MiB（82MiB 余量），
70,250-token prompt 实测 200 OK（73.4s 冷），同 prompt 第二发
cached 70,240/70,250 = **99.99%**（1.52s）；再高到 80,896 就会
`PoolExhausted`。池不手写（自动派生）。4 快照时 75712 连第一个 turn 都过不了
（余量 786MiB < 1GiB）——所以本档把快照压到 1 槽（实测复用没丢）。
视觉塔让 live 权重 +0.86G，无视觉时上限 ≈104.4k；代价是驱动余量贴着压力
地板，会出现 `N6 压力节点` WARN（功能/缓存实测未坏；要零腾槽改回 65536）。
宿主 KV 腿 **10GiB**（≈4 条 79k 链）：4GiB 时两发长 prompt 后第三发的 27k 前缀
回灌会撞「槽池耗尽」；10GiB 让整链下沉、设备槽立刻回来（2026-10-25 修复配套）。
**旧注释里的 262k `--set` 配方已失效**：自动池在 overcommit 之前先判
`max_seq ≤ ctx_max`，262k 会 boot 响亮拒；要 262k 需先改引擎（逻辑 ctx 与
物理池解耦），不是配置能开出来的。历史实测表在该档文件头注释里。

```bash
# 临时变体（不再建文件；超过 ctx_max 会被 boot 响亮拒，不会静默夹）
--set spec.depth=0 --set spec.lookup=false     # 关投机对照臂
--set runtime.max_seq=65536                    # 回退到旧稳态点（压力更小）
--set model.vision=false                       # 关视觉可换回更大 ctx（~104k）
```

## 常用命令

```bash
# 预检（不装权重，<2s）
./target/release/server -p profiles/qwen38_27b-3090.toml --check
./target/release/server -p profiles/qwen38_27b-3080_tp2.toml --check

# 起服（档内已写端口：3090 = 127.0.0.1:8134，3080 TP2 = 127.0.0.1:8135）
cargo serve -p profiles/qwen38_27b-3090.toml
./target/release/server -p profiles/qwen38_27b-3080_tp2.toml \
  --set parallel.device='["GPU-…","GPU-…"]' --log-level info

# 临时变体（不再建文件）
--set runtime.max_seq=65536                    # 回退到无抖动稳态点
--set spec.depth=0 --set spec.lookup=false     # 关投机对照臂
--set pool.snap_max=2                          # 用前缀复用换更大 ctx
```

## 目录

```text
profiles/
  full.toml                    # 全量 schema + 注释（底座，不是档位）
  qwen38_27b-3090.toml         # 常驻档①：单 3090 满血
  qwen38_27b-3080_tp2.toml     # 常驻档②：双 3080 TP2 满血
  parts/
    model/                     # 模型资产（path/format/quant/head；含 0.8B/4B/27B GGUF 与 AWQ）
    hardware/                  # 卡位 + TP 开关（3090-single / 3080-tp2）
    kernel/                    # fi-graph / v2 臂
    pool/                      # fp8 / f16 池
    feature/                   # vision / dflash2 / mtp / kvmem / agent / w4a8 / image-evict
```

`parts/**` 是**构成件**，不是入口档；小模型（0.8B / 4B）与 GGUF 形状要用时
按上面的 `extends` 样板临时组合，或常驻档 + `--set`（例如
`--set model.path=../models/… --set model.format=gguf`）。

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
- 实验变体优先用 `--set`，不要为每个变体新建文件。

## 目录

```text
profiles/
  full.toml                 # 全量 schema + 注释
  parts/
    model/                  # 模型资产（path/format/quant/head）
    hardware/               # 卡位 + TP 开关
    kernel/                 # FI 图 / v2 臂
    pool/                   # fp8 / f16 池
    feature/                # vision / dflash2 / mtp / kvmem / agent / w4a8 / image-evict
  experiments/              # 历史实验档（路径已变，不作为生产入口）
  <canonical>.toml          # 生产/测试入口
```

## 保留入口

| 文件 | 说明 |
|---|---|
| `qwen38_27b-3090-w4a16.toml` | 单 3090 Ti / 27B AWQ-INT4 / 128k / fp8 KV |
| `qwen38_27b-3090-w4a16-dflash.toml` | 单 3090 / AWQ + DFlash2 d7 / 16k |
| `qwen38_27b-3090-w4a16-mtp.toml` | 单 3090 / AWQ + MTP d3 + window / 16k |
| `qwen38_27b-3080_tp2-w4a16.toml` | 2×3080 / AWQ + 视觉 + DFlash2 d7 / 262k |
| `qwen38_27b-3080_tp2-w4a8.toml` | 2×3080 / W4A8 + DFlash2 / 262k |
| `qwen38_27b-3090-gguf.toml` | 单 3090 / 27B GGUF Q4_K_M / 32k |
| `qwen38_27b-3090-gguf-agent.toml` | 上档 + pi/agent 10K 前缀复用 |
| `qwen38_27b-3090-gguf-kvmem.toml` | 27B GGUF + KVMem 131k |
| `qwen38_27b-3090-iq4xs.toml` | 27B GGUF UD-IQ4_XS / 12k |
| `qwen38_27b-3090-q4km-vlm.toml` | 27B Q4_K_M + Q8_0 视觉塔 / 8k |
| `qwen35_4b-3090-q4km-mtp.toml` | 4B Q4_K_M + NextN MTP / 16k |
| `qwen35_4b-3090-q4km-vlm.toml` | 4B Q4_K_M + mmproj / 8k |
| `qwen35_0_8b-3090-iq4xs.toml` | 0.8B IQ4_XS / 16k 开发档 |

## 常用命令

```bash
# 预检（不装权重）
./target/release/server -p profiles/qwen38_27b-3090-w4a16.toml --check

# 单卡 3090（pi 前端 3090_dual provider 指向 8133）
./target/release/server -p profiles/qwen38_27b-3090-w4a16.toml \
  --set server.bind=127.0.0.1:8133 --log-level info

# 双卡 3080（pi 前端 3080_dual provider 指向 8134）
./target/release/server -p profiles/qwen38_27b-3080_tp2-w4a16.toml \
  --set server.bind=127.0.0.1:8134 --log-level info

# 临时改上下文，不新建档
--set runtime.max_seq=65536
```

> `profiles/experiments/` 里的旧档仍可用（`extends` 已改为 `../` 路径），
> 但不再维护在顶层；新实验请用 canonical 档 + `--set`。

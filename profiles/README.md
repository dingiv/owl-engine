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

## 常驻档（三个）

2026-10-24 用户裁决：入口档只留硬件形状，其余 27 个历史档删除
（收敛记录与找回方式：`roadmap.local/profiles-consolidation-2026-10-24.md`）。
2026-10-27 增补第三档（GGUF 文本档，实测全面快于 AWQ 档）。

| 文件 | 端口 | 形状 |
|---|---|---|
| `qwen38_27b-3090.toml` | 8134 | 单 3090 Ti / 27B AWQ-INT4 + 视觉 / DFlash2 d7 / FI+图 / fp8 KV / 前缀缓存 / image_evict / **max_seq 63456（全特性稳定点）** / chunk 2048 |
| `qwen38_27b-3080_tp2.toml` | 8135 | 2×3080 20G TP2 / 27B AWQ-INT4 + 视觉 / DFlash2 d7 / fp8 KV / 前缀缓存 / image_evict / **max_seq 262144** / chunk 2048 |
| `qwen38_27b-gguf_3090.toml` | 8136 | 单 3090 Ti / 27B **GGUF UD-IQ4_XS**（14.3GB）+ **mmproj-F16 视觉塔** / DFlash2 d7 / fp8 KV / 前缀缓存 / **max_seq 172032（168k，本机单卡最长）** / chunk 2048。prefill 1633–1681 t/s、decode 127–135、128k prompt 零拒发；代价 = prefill 比 8134 的 a8 慢 ~24% |

这条裁决被**测试钉住**：`crates/shared/src/profiles.rs::入口档面只允许三个常驻档`
—— 往 `profiles/` 顶层再丢一个 `.toml` 就会红。置顶三档各占一张卡
（3090×2 在 8134/8136，3080 对在 8135），互不干扰。

`profiles/measurement/` = **测量档**（不计入入口档面；子目录不参与档名发现）：
`qwen38_27b-gguf.toml` / `qwen38_27b-gguf-iq4.toml` 是 GGUF 裸性能口径，
不是常驻形状。

2026-10-25 生产档拉满（两档同形，硬件差异只留 max_seq/TP）：

- `prefix_cache=true` + `snap_msg_boundary=true`；
- `implicit_session=true`（pi 不发 session，工具轮/图片删除有稳定会话）；
- `image_evict=true`（最终方案 = **leave-gap**：删后不转 K、保留位置洞，
  可无限次丢弃旧图片缓存；旧的 rebase/K 旋转路径已删除）；
- 3090：`max_seq=63456 / snap_max=2 / host_cache=10G / snap_host=3G`；
- 3080 TP2：`max_seq=262144 / snap_max=2 / host_cache=10G / snap_host=3G`。

pi 实测工具跟进 cached 95.6% / TTFT 149ms，手动新消息 96.2% / 97ms，
新会话 98.3% / 116ms；8135 的 `image-evict` 5/5、GUI 窗口 PASS。
详见 `docs/devlog/serving/agent-surface.md` §9.4。

`max_seq=63456 / snap_max=2` 是 **2026-10-25 单 3090 全特性稳定点**：
24G 上同时开视觉塔 + DFlash2 d7 + fp8 KV + 前缀缓存 + image_evict。
fresh boot 驱动余量高于 1GiB；前缀缓存饱和后可能短暂出现 N6 压力 WARN，
`image_delete_marked` 与 GUI 三图窗口实测仍通过。更大的 79872/snap1 虽然 ctx 更大，但驱动余量贴地板，
image_evict 的前缀块会被下沉到 host（`前缀有宿主占位 ⇒ 不吸收`），
不再作为生产默认；长上下文/大图窗口由 3080 TP2（262144/snap2）承担。

宿主 KV 腿 **10GiB**（≈4 条 63k 链）：4GiB 时两发长 prompt 后第三发的
27k 前缀回灌会撞「槽池耗尽」；10GiB 让整链下沉、设备槽立刻回来
（2026-10-25 修复配套）。自动池在 overcommit 之前先判 `max_seq ≤ ctx_max`，
262k 在单 3090 上会 boot 响亮拒；要 262k 需改引擎（逻辑 ctx 与物理池解耦），
不是配置能开出来的。

```bash
# 临时变体（不再建文件；超过 ctx_max 会被 boot 响亮拒，不会静默夹）
--set spec.depth=0 --set spec.lookup=false     # 关投机对照臂
--set runtime.max_seq=63456                    # 全特性稳定点（默认）
--set runtime.max_seq=79872 --set pool.snap_max=1  # 更大 ctx 但 image_evict 可能失效
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
--set spec.depth=0 --set spec.lookup=false     # 关投机对照臂
# 默认 63456 + snap2;要更大 ctx 见上面 79872 的取舍
```

## 目录

```text
profiles/
  full.toml                    # 全量 schema + 注释（底座，不是档位）
  qwen38_27b-3090.toml         # 常驻档①：单 3090 满血（AWQ + 视觉）
  qwen38_27b-3080_tp2.toml     # 常驻档②：双 3080 TP2 满血
  qwen38_27b-gguf_3090.toml    # 常驻档③：单 3090 / GGUF Q4_K_M（文本，更快）
  measurement/                 # 测量档（不算入口档面；GGUF 裸性能口径）
    qwen38_27b-gguf.toml
    qwen38_27b-gguf-iq4.toml
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

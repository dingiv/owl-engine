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
- **并发三键必须成组改**（boot 门会拒启，别指望缺省兜）：
  - `[runtime] max_concurrency = N`（在跑 + 排队的请求上限）；
  - `[pool] gdn_slots ≥ N`（每活跃 turn 一格 GDN 状态；`parts/pool/fp8.toml` 给的是 1，
    多并发档要在 leaf 里显式覆盖）—— 不足会拒启：
    `[pool] gdn_slots(1) < [runtime] max_concurrency(2)`；
  - `[pool] snap_max ≥ N + 1`（每个活跃 turn 的 `last_snap` 各占一格且**受保护**，
    再留一格给"刚收口那条链的尾边界"）。
  **并发 ≠ 设备并行**：本仓 `max_batch` 只能为 1（合批要求单卡 / spec off / 无选择层），
  所以 N>1 的效果是"第二路**进引擎队列**（`/debug/queue` 可见、有 turn id），
  而不是在 HTTP permit 上干等"；设备拍仍串行。

## 常驻档（入口面）

2026-10-24 用户裁决：入口档只留硬件形状，其余 27 个历史档删除
（收敛记录与找回方式：`roadmap.local/profiles-consolidation-2026-10-24.md`）。
2026-10-27 增补第三档（GGUF 文本档，实测全面快于 AWQ 档）。

| 文件 | 端口 | 形状 |
|---|---|---|
| `qwen38_27b-3090.toml` | 8137 | 单 3090 Ti / 27B AWQ-INT4 + 视觉 / DFlash2 d7 / FI+图 / fp8 KV / 前缀缓存 / image_evict / **max_seq 57344（a8 版全特性稳定点）** / chunk 2048 |
| `qwen38_27b-3080_tp2.toml` | 8135 | 2×3080 20G TP2 / 27B AWQ-INT4 + 视觉 / DFlash2 d7 / fp8 KV / 前缀缓存 / image_evict / **max_seq 262144** / chunk 2048 / **并发 2（`gdn_slots=2` · `snap_max=3`；2026-11-03 用户裁决）** |
| `qwen38_27b-gguf_3090.toml` | 8137 | 单 3090 Ti / 27B **GGUF UD-IQ4_XS**（14.3GB）+ **mmproj-F16 视觉塔** / DFlash2 d7 / fp8 KV / 前缀缓存 / **max_seq 172032（168k）** / chunk 2048。prefill 1633–1681 t/s、decode 127–135、128k prompt 零拒发；代价 = prefill 比 a8 档(同端口)慢 ~24% |
| `qwen38_27b-gguf-iq3_3090.toml` | 8137 | 单 3090 Ti / 27B **GGUF UD-IQ3_S**（12.0GB）+ mmproj-F16 / DFlash2 d7 / fp8 KV / **max_seq 229376（224k，逼近 native 262144）** / chunk 2048。prefill 1409–1509 t/s、decode 97–121；**与 8137 上的其它单卡 3090 档互斥**（同一张卡二选一），速度慢 ~13%、3bit 质量低一档 |
| `qwen38_27b-4080.toml` | 8138 | 单 **4080 SUPER（Ada sm_89 / 32G）** / 27B AWQ-INT4 **a8** + 视觉 / DFlash2 d7 / FI+图 / fp8 KV / 前缀缓存 / image_evict / **max_seq 262144（= 模型 native 上限，32 KiB/token ⇒ 池 8192MiB）** / chunk 2048。std 五点三轮（`perf-runs/std-4080-20261030/`）：**spec-off prefill 2392→3084 t/s、decode 35.9 t/s 长 ctx 平坦**；spec-d7 decode ×1.70（60.1@8k）；**同二进制 3090 Ti 对照：prefill ×1.21–1.37、decode ×0.81、TTFT ×0.73–0.83**。**需 sm_89 二进制**（`cargo build --release -p server --no-default-features --features arch-sm89`，落到独立 target 目录）——单份 `.a` 家族取最高代，30 系与 40 系不混在一份构建里 |

这条裁决被**测试钉住**：`crates/shared/src/profiles.rs::入口档面只允许三个常驻档`
—— 往 `profiles/` 顶层再丢一个 `.toml` 就会红。置顶三档各占一张卡
（**端口纪律 2026-11-03 用户裁决**）：

| 形状 | 端口 | 说明 |
|---|---|---|
| 双 3080 20G（TP2） | **8135** | `qwen38_27b-3080_tp2.toml` |
| 双 3090 Ti（TP2） | **8136** | `qwen38_27b-3090_tp2.toml`（本地变体见 `*.local.toml`） |
| **单卡 3090（任意配置）** | **8137** | AWQ / GGUF-IQ4 / GGUF-IQ3 / KVMem … **统一这一个端口**，同卡互斥 ⇒ 换配置 = 停一个起一个（`--set` 或换档文件，端口不变） |
| **单 4080 SUPER** | **8138** | 同上：任意单卡 4080 配置都用 8138 |

单卡的"换配置"不换端口：端口 = **硬件形状**，配置 = 档文件 / `--set`。四档各占各的卡（3090 对 8136 占两张 3090 Ti；单卡 3090 档 8137 与它互斥；3080 对 8135；4080 8138），互不干扰。

**绑定 = 全部网卡**（2026-11-03 用户裁决）：所有档 `[server] bind = "0.0.0.0:<port>"`，局域网客户端直连 `http://192.168.1.6:<port>/v1`（`~/.pi/agent/models.json` 已按此配置）。
⚠️ 服务**无鉴权** ⇒ 只在可信内网用，不要做公网端口映射。

`profiles/measurement/` = **测量/实验档**（不计入入口档面；子目录不参与档名发现）：

- `qwen38_27b-gguf.toml` / `qwen38_27b-gguf-iq4.toml` —— GGUF 裸性能口径；
- `qwen38_27b-3090-int8.toml` —— **INT8 KV 单卡档（8141）**：**w4a8**、
  max_seq 131072、K int8 + V packed + per-token/per-page scale；P8 prefill /
  P32 decode；spec off。3 次中位 65k 1691 t/s / TTFT 38.75s，
  100k 1488 / 67.19s；GSM8K 200 96.5%，needle 8k/32k/64k/100k 全过。
- `qwen38_27b-3090-fp8.toml` —— **FP8 KV 单卡同形对照档（8144）**：默认
  w4a16，A/B 时用 `--set model.quant=w4a8` 与 int8 档同形。
- `qwen38_27b-3090_tp2-int8.toml` / `qwen38_27b-3090_tp2-fp8.toml` —— **双 3090
  TP2 INT8/FP8 同形对照档（8142/8143）**：w4a8、262k、spec off。
  65k INT8 2483.8 t/s / decode 58.97；100k 2222.5 / 57.48。
- `qwen38_27b-3090-kvmem.toml` —— **KVMem 单卡档（8137，与单卡 3090 统一端口）**：池超卖 5.49× +
  选择层 + 块下沉。起机 `-p profiles/measurement/qwen38_27b-3090-kvmem.toml`
  （不需要 `--set`）。
  🔴 **真实可用上下文 = 池容量 47744 tok，不是 `max_seq`**：链越过池就有
  占位符 ⇒ prefill 被迫只读稀疏视图 ⇒ 实测 64k 起输出退到 1 token。
  长上下文请用 `qwen38_27b-3080_tp2.toml`。
  实测：prefill 4k/8k/16k/32k/64k = 2168/2116/2018/1795/1755 t/s；
  decode（真实文本 8k/512 tok）**83 t/s**、AL 2.83、verify 24.4 ms/轮；
  48k 文档增量对话 ≈ 9 s（修 GDN 快照锚点前是 28.7 s）。
  档内焊死三个"看着能开、实际会坏"的开关（`prefill_view=always` /
  `decode_arm=fi` / `spec.degrade_after>0`），档头有逐条实测依据。

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

# 起服（档内已写端口：单卡 3090 = 8137 · 3090 对 = 8136 · 3080 对 = 8135 · 4080 = 8138）
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
    hardware/                  # 卡位 + TP 开关（3090-single / 3080-tp2 / 4080-single / 5070ti-single）
    kernel/                    # fi-graph / v2 臂
    pool/                      # fp8 / f16 / int8 池
    feature/                   # vision / dflash2 / mtp / kvmem / agent / w4a8 / image-evict
```

`parts/**` 是**构成件**，不是入口档；小模型（0.8B / 4B）与 GGUF 形状要用时
按上面的 `extends` 样板临时组合，或常驻档 + `--set`（例如
`--set model.path=../models/… --set model.format=gguf`）。

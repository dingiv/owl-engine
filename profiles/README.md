# profiles —— 组件化配置

配置只有一个概念：**profile = 一份 TOML**。`full.toml` 是公开配置模板
（常用键全列；诊断/内部键默认不展示，必要时用 `--set`）；所有可启动档都是
`extends` 薄 leaf。

## 规则

- 新档只写差异，不要复制整份配置：

```toml
extends = [
  "full.toml",
  "parts/model/27b-awq.toml",
  "parts/feature/fi-graph.toml",
  "parts/pool/fp8.toml",   # 30 系换成 parts/pool/int8.toml；40/50 保持 fp8
]
```

- `extends` 路径相对**声明它的文件**；数组按序覆盖，后者胜。
- 合并后仍缺键 ⇒ 拒启；`--print-config` 是唯一放行半份档的迁移工具。
- 卡位只走 `[parallel]`；`device` 用 BDF / UUID / 名字子串，**禁 CUDA 序号**。
  硬件形状直接写进 leaf（例如 `mode = "single"; device = ["3090"]`）。
- **实验变体一律用 `--set`，不新建入口档**（见下"入口档面"）。
- **并发只有两个键，必须成组改**：
  - `[store.gdn] gdn_slots = N` —— **并发唯一真源**（2026-11-16：旧的
    `[runtime] max_concurrency` 已删；服务面 permit 数与引擎活跃槽位上限都从它来）。
    每活跃 turn 一格 GDN 状态；`parts/pool/fp8.toml` 给的是 1，多并发档要在 leaf
    里显式覆盖；**0 = 拒启**（旧的"0 = 自动"已退休）。
  - `[store.gdn] snap_max`：**N=2 时用 6**（2026-11-06 双会话门实测：3 会丢空闲会话的尾锚点，
    下一轮命中 71.6% vs 期望 86%；4 仍偶发，6 稳定）。单并发 N=1 用 2。
    语义 = 每个活跃 turn 的 `last_snap` 各占一格且**受保护**，
    再留一格给"刚收口那条链的尾边界"）。
  **并发 ≠ 设备并行**：本仓 `max_batch` 只能为 1（合批要求单卡 / spec off / 无选择层），
  所以 N>1 的效果是"第二路**进引擎队列**（`/debug/queue` 可见、有 turn id），
  而不是在 HTTP permit 上干等"；设备拍仍串行。

## KV 池代际默认（2026-11-04）

- **30 系（3090 / 3080）**：默认 `parts/pool/int8.toml`
  （唯一例外：GGUF IQ3_S 3-bit 档保持 FP8 —— INT8 下 1k needle 稳定拒答，
  FP8 同题通过；不是 INT8 机制错，是 3-bit 主干对 KV 量化误差更敏感）
  （K int8 + V packed + per-token/per-page scale）。
  INT8 文本 1k/2k/8k/32k/120k、视觉 grounding、`image_evict` 删除吸收
  均已过门；默认档不再走 FP8。
- **40 系 / 50 系**：继续 `parts/pool/fp8.toml`；`KernelPick` 在
  `Arch=Sm89/Sm120` 自动选**原生 FP8 MMA**（e4m3 Q×KV，不再反量化到
  half）。4080 实测 std 1024–8192 +2.8%、32k +19.6%、65k +16.0%；
  **50 系共用同核，待真机质量/性能验收**。

## 常驻档（入口面）

2026-10-24 用户裁决：入口档只留硬件形状，其余 27 个历史档删除
（收敛记录与找回方式：`roadmap.local/profiles-consolidation-2026-10-24.md`）。
2026-10-27 增补第三档（GGUF 文本档，实测全面快于 AWQ 档）。

| 文件 | 端口 | 形状 |
|---|---|---|
| `qwen38_27b-3090.toml` | 8137 | 单 3090 Ti / 27B AWQ-INT4 + 视觉 / DFlash2 d7 / FI+图 / **INT8 KV（30 系默认）** / 前缀缓存 / image_evict / **max_seq 57344（a8 版全特性稳定点）** / chunk 2048 |
| `qwen38_27b-3080_tp2.toml` | 8135 | 2×3080 20G TP2 / 27B AWQ-INT4 + 视觉 / DFlash2 d7 / **INT8 KV（30 系默认）** / 前缀缓存 / image_evict / **max_seq 262144** / chunk 1024（20G 稳定点）/ **满池双活跃（`gdn_slots=2` · `max_batch=2` · `snap_max=6` · `vram_activation_reserve=1024` · `snap_session_cap=20`/`structural_reserve=4`）** |
| `qwen38_27b-gguf_3090.toml` | 8137 | 单 3090 Ti / 27B **GGUF UD-IQ4_XS**（14.3GB）+ **mmproj-F16 视觉塔** / DFlash2 d7 / fp8 KV / 前缀缓存 / **max_seq 172032（168k）** / chunk 2048。prefill 1633–1681 t/s、decode 127–135、128k prompt 零拒发；代价 = prefill 比 a8 档(同端口)慢 ~24% |
| `qwen38_27b-gguf-iq3_3090.toml` | 8137 | 单 3090 Ti / 27B **GGUF UD-IQ3_S**（12.0GB）+ mmproj-F16 / DFlash2 d7 / **FP8 KV（例外：3-bit 主干对 INT8 KV 的 1k needle 敏感）** / **max_seq 229376（224k，逼近 native 262144）** / chunk 2048。prefill 1409–1509 t/s、decode 97–121；**与 8137 上的其它单卡 3090 档互斥**（同一张卡二选一），速度慢 ~13%、3bit 质量低一档 |
| `qwen38_27b-4080.toml` | 8138 | 单 **4080 SUPER（Ada sm_89 / 32G）** / 27B AWQ-INT4 **a8** + 视觉 / DFlash2 d7 / FI+图 / fp8 KV / 前缀缓存 / image_evict / **max_seq 262144（= 模型 native 上限，32 KiB/token ⇒ 池 8192MiB）** / chunk 2048。std 五点三轮（`perf-runs/std-4080-20261030/`）：**spec-off prefill 2392→3084 t/s、decode 35.9 t/s 长 ctx 平坦**；spec-d7 decode ×1.70（60.1@8k）；**同二进制 3090 Ti 对照：prefill ×1.21–1.37、decode ×0.81、TTFT ×0.73–0.83**。**需 sm_89 二进制**（`cargo build --release -p server --no-default-features --features arch-sm89`，落到独立 target 目录）——单份 `.a` 家族取最高代，30 系与 40 系不混在一份构建里。**2026-11-04 原生 FP8 MMA**（KernelPick 按运行期 Sm89/Sm120 选臂；e4m3 Q×KV，CTA16）：std 1024–8192 均值 3113 t/s（旧 3030，+2.8%）、32k 2844（旧 2379，+19.6%）、65k 2435（旧 2099，+16.0%），decode 不变；1k/8k/65k needle + 视觉 36/36 全绿（`docs/bench/fp8-sm89-native-mma-2026-11-04.md`） |

这条裁决被**测试钉住**：`crates/shared/src/profiles.rs::入口档面只允许五个常驻档`
—— 往 `profiles/` 顶层再丢一个 `.toml` 就会红。五个入口档按硬件形状分工
（**端口纪律 2026-11-03 用户裁决**）：

| 形状 | 端口 | 说明 |
|---|---|---|
| 双 3080 20G（TP2） | **8135** | `qwen38_27b-3080_tp2.toml` |
| 双 3090 Ti（TP2） | **8136** | 无公共入口档；本地档直接写 `[parallel]`（`mode="tensor"`, `device=["3090","3090"]`） |
| **单卡 3090（任意配置）** | **8137** | AWQ / GGUF-IQ4 / GGUF-IQ3 / KVMem … **统一这一个端口**，同卡互斥 ⇒ 换配置 = 停一个起一个（`--set` 或换档文件，端口不变） |
| **单 4080 SUPER** | **8138** | 同上：任意单卡 4080 配置都用 8138 |

单卡的"换配置"不换端口：端口 = **硬件形状**，配置 = 档文件 / `--set`。
3090 对 8136 占两张 3090 Ti；单卡 3090 的 AWQ/IQ4/IQ3 档共用 8137、彼此互斥；
3080 对 8135；4080 8138。

**绑定**：档里的 `[server] bind` 只是本机示例；部署到别的机器时改成该机的监听地址。
服务**无鉴权** ⇒ 不要直接做公网端口映射。

本机实验档放在 ignored 的 `profiles/.local/` 与 `*.local.toml`（站点私有，不入库、
不参与入口档面）；公共档不要引用这些文件。

2026-10-25 生产档拉满（两档同形，硬件差异只留 max_seq/TP）：

- `prefix_cache=true`（消息边界快照/删图锚点**恒开**，2026-11-16 键退休）；
- 会话身份 = **内容身份**（pi 不发 session 键 ⇒ 引擎按 prompt 前缀认会话；
  2026-11-16 原始算法；旧 `implicit_session` 键已删）；
- `image_evict=true`（最终方案 = **leave-gap**：删后不转 K、保留位置洞，
  可无限次丢弃旧图片缓存；旧的 rebase/K 旋转路径已删除）；
- 3090：`max_seq=57344 / snap_max=2 / host_cache=10G / snap_host=3G`；
- 3080 TP2：`max_seq=262144 / max_batch=2 / snap_max=6 / gdn_slots=2 / vram_reserve=1024M / host_cache=10G / snap_host=3G`。

pi 实测工具跟进 cached 95.6% / TTFT 149ms，手动新消息 96.2% / 97ms，
新会话 98.3% / 116ms；8135 的 `image-evict` 5/5、GUI 窗口 PASS。
详见 `docs/devlog/serving/agent-surface.md` §9.4。

`max_seq=57344 / snap_max=2` 是 **2026-10-27 a8 版单 3090 全特性稳定点**：
24G 上同时开视觉塔 + DFlash2 d7 + fp8 KV + 前缀缓存 + image_evict。
fresh boot 驱动余量高于 1GiB；前缀缓存饱和后可能短暂出现 N6 压力 WARN，
`image_delete_marked` 与 GUI 三图窗口实测仍通过。更大的 79872/snap1 虽然 ctx 更大，但驱动余量贴地板，
image_evict 的前缀块会被下沉到 host（`前缀有宿主占位 ⇒ 不吸收`），
不再作为生产默认；长上下文/大图窗口由 3080 TP2（262144/snap6/满池）承担。

宿主 KV 腿 **10GiB**（≈4 条 63k 链）：4GiB 时两发长 prompt 后第三发的
27k 前缀回灌会撞「槽池耗尽」；10GiB 让整链下沉、设备槽立刻回来
（2026-10-25 修复配套）。自动池在 overcommit 之前先判 `max_seq ≤ ctx_max`，
262k 在单 3090 上会 boot 响亮拒；要 262k 需改引擎（逻辑 ctx 与物理池解耦），
不是配置能开出来的。

```bash
# 临时变体（不再建文件；超过 ctx_max 会被 boot 响亮拒，不会静默夹）
--set spec.depth=0 --set spec.lookup=false     # 关投机对照臂
--set runtime.max_seq=63456                    # 全特性稳定点（默认）
--set runtime.max_seq=79872 --set store.gdn.snap_max=1  # 更大 ctx 但 image_evict 可能失效
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
# 无 P2P 驱动环境:切 pinned host-mapped 中转(当前需同时落回 v1 两核)
--set parallel.tp_comm=host --set parallel.tp_ar=v1
# 默认 57344 + snap2;要更大 ctx 见上面 79872 的取舍
```

## 目录

```text
profiles/
  full.toml                    # 公开配置模板（常用键；底座，不是档位）
  qwen38_27b-3090.toml         # 常驻档①：单 3090 / AWQ-INT4 + 视觉
  qwen38_27b-3080_tp2.toml     # 常驻档②：双 3080 TP2 / AWQ-INT4
  qwen38_27b-gguf_3090.toml    # 常驻档③：单 3090 / GGUF IQ4_XS + 视觉
  qwen38_27b-gguf-iq3_3090.toml# 常驻档④：单 3090 / GGUF IQ3_S（极限上下文）
  qwen38_27b-4080.toml         # 常驻档⑤：单 4080 SUPER / AWQ-INT4
  examples/                    # 模板变量示例（不入入口档面）
  parts/
    model/                     # 模型资产（path/format/quant/head；含 0.8B/4B/27B GGUF 与 AWQ）
    pool/                      # fp8 / f16 / int8 池
    feature/                   # vision / dflash2 / mtp / kvmem / fi-graph / w4a8 / image-evict
```

`parts/**` 是**构成件**，不是入口档；小模型（0.8B / 4B）与 GGUF 形状要用时
按上面的 `extends` 样板临时组合，或常驻档 + `--set`（例如
`--set model.path=../models/… --set model.format=gguf`）。

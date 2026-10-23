# 配置链:意图 → 派生 → 预检 → 解释(docs/devlog/dev)

> 动手改档/加实验档前先看这一页。**配置的定义面唯一在
> `crates/shared/src/config/`**(键的账本 + 类型 + 缺省);本页讲"怎么用"。
> 纪律:**引擎行为不许用 env 控**(env 只剩 `OWL_CONFIG` 路径兜底 + 测试域自读);
> 一切可配项走配置文件/CLI 显式声明。
>
> **2026-10-20 配置统一**:只有一个概念 —— **profile = 一份 TOML 配置文件**。
> 没有 "flavor"。用户启动用 `-p <profile_path>`;源码 `profiles/` 下是可直接
> 复制修改的常用档;**所有档都是完整档**(全字段显式,缺一拒启,见 §1.1)。

## 1. 一个概念:profile

| | profile |
|---|---|
| 文件 | 任意路径的 `.toml`;源码常用档在 `profiles/` |
| 内容 | `[intent]` 意图键,或低层键(同一 loader;高级覆盖) |
| 入口 | `-p/--profile <profile_path>` |
| 查找 | **不做名字查找**;路径直接给 |
| 预检 | `--check` 派生 + 内存账 + 模型几何 |

常用组合直接复制:

```bash
cp profiles/qwen38_27b-3080_tp2-w4a16.toml ~/my-27b.toml
$EDITOR ~/my-27b.toml          # 改 model.path / server.bind / [parallel] 卡位
./target/release/server -p ~/my-27b.toml
```

`profiles/full.toml` 是**唯一全量档**(全部键 + 注释);其余档都用
`extends = ["full.toml"]` 只写差异(见 §1.1)。

历史 `legacy/` 目录已于 2026-10-21 删除。

## 1.1 全字段显式 + `extends`（2026-10-21 裁决）

- 用 `-p <profile>` 给的档，**合并 `extends` 之后**每个键都必须有值；缺一个 ⇒
  **拒启**（一次列出全部缺项）。代码缺省不兜底 —— 旧行为让各会话跑在
  "半份配置 + 旧缺省"上，启动点不可复现。
- **继承**写在**根键**：

  ```toml
  extends = ["full.toml"]                 # 相对本文件;也接受绝对路径
  extends = ["base.toml", "site.toml"]    # 数组:按序覆盖,后者胜
  extends = "full.toml"                   # 单串也认
  ```

  被继承文件的**全部字段**先落地（可递归，带环检测/深度上限），本文件覆盖之；
  于是子档只需写"与 base 不同的项"。**完整性按合并结果判**（父辈字段算数）。
- 站点层（位置参数 / `--site`）与 CLI（`--set`）仍是**覆盖层**：只覆写，不补全。
- 生成全量档：`server -p <档> --print-config > <新档>.toml`
  （该路径**放行半份档** —— 它就是迁移工具）。
- **卡位唯一真源 = `[parallel]`**：

  ```toml
  [parallel]
  mode = "single"                       # single | tensor | layer(暂未支持)
  device = ["41:00.0", "GPU-…"]         # BDF / UUID / 名字子串;"3090" 亦可
  ```

  - 空列表 = 自动挑空闲卡（single 1 张 / tensor 2 张）；
  - 列表按**去重消费**：`["3090","3090"]` = 两张不同的 3090；
  - **禁 CUDA 序号**（0/1/… 这种数字会被拒；4 位型号名如 3090 是名字）。

## 2. 层序(后面的覆盖前面的)

```text
Default(代码缺省)
  → -p <profile_path>(配置档;意图或低层键)
  → 位置参数 CONFIG(可选覆盖层)
  → --site <PATH>(可选站点层)
  → CLI(--set / --log-* / --perf-level / --step-gate*)
  → [intent] 派生(只填空:显式低层键优先)
```

注意最后一步的顺序:**显式低层键覆盖意图派生值**(高级覆盖),派生只填
"没人写"的键。`--set`/CLI 旗标与文件同一条链,所以
`--print-config` 看到的**就是** boot 用的配置(不再有 print 之后二次覆盖)。

### 2.1 KernelSelectionEnv(boot 投影)

启动后由 `KernelSelectionEnv` 冻结选核环境,不再让 picker 读配置:

```text
EnvProvider → KernelSelectionEnv {
    HwCaps    arch
    QuantCaps plan / act dtype
    KvCaps    quant / dtype / page / hd / grp / paged
    FeatureCaps vision / spec_depth / tp_ranks
}
           → kernel_env.op_env() → driver::OpEnv
```

- `EnvProvider.kernel_env` 是唯一快照;`op_env()` 优先读它;
- target 与 draft 各一份(draft 的 hd/grp/池 dtype 不同);
- boot 日志 `[kernel-env] target: ... / draft: ...` 可直接审计;
- `KernelSelectionEnv::resolve(OpSpec)` / `collect_ops` 已提供派生入口;
  `TagFilter` 由配置事实派生（W4A16/GGUF/KV/vision/spec/TP/head），
  不再手抄 `allowed_ops`。

## 3. Intent schema(`[intent]`)

```toml
[intent]
model       = "qwen3.8-27b-awq"   # 模型注册表键
hardware    = "3090ti-24g"        # 或 3080-20g;不写 = 从 device.find_name 推断
goal        = "max-context"       # max-context | max-speed | balanced | low-memory
context     = "max"               # 数字(token)或 "max"(启动前求解)
concurrency = 1                   # gdn_slots = max_concurrency = max_batch
spec        = "off"               # off | dflash(需 [site].draft_dir)
kv          = "fp8"               # fp8 | f16
kvmem       = "off"               # off | overcommit(一个键替四键)
graphs      = "auto"              # auto | off;auto 不可用时自动降级并打印原因
reuse       = "off"               # off | session | global;不写 = goal 决定
```

| 意图 | 派生 |
|---|---|
| `model` | `[model] kind/quant/head_quant/embed_quant`(与显式值冲突 = 拒) |
| `goal=max-context` | 复用 off、`512+512`、`vram_target=0.99`、`reserve=256M` |
| `goal=max-speed` | 复用 global、`1024+0`、贴顶 |
| `goal=balanced` | 复用 session、`512+512`、0.97 |
| `goal=low-memory` | kvmem overcommit、复用 global、小激活面 |
| `context=N` | `max_seq=N`(物理池:无缓存→min(cap,N×并发+1页);缓存开→cap) |
| `context="max"` | 内存模型求解(§4) |
| `concurrency=C` | `gdn_slots=C`、`runtime.max_concurrency=C`、`max_batch=min(C,8)` |
| `spec=dflash` | `depth=7` + 草稿目录必填 + **草稿环固定 ~40MiB**(窗口 2048,见下) |
| `[spec] draft` | `auto`（缺省，DFlash2 目录优先 → 检查点 MTP）\| `mtp` \| `dflash2`；显式档缺资产 = boot 拒 |
| `kv=fp8/f16` | `pool.kv_fp8` |
| `kvmem=off` | 选择层/索引/超卖四键全关 + 宿主/磁盘腿清零 |
| `kvmem=overcommit` | 四键全开 + 宿主腿 + 强制 prefix_cache |
| `graphs=auto` | FI 单图 decode + `fi_plan_bake=0` + 窗刷表（prefill 桶已退役） |
| `graphs=off` | `no_graph=true`,FI 臂全关 |
| `reuse=off` | 前缀缓存/快照全关(单会话最大上下文) |
| `reuse=session` | 1 条快照 + 1G 状态腿 |
| `reuse=global` | 4 条快照 + 4G KV 宿主腿 + 3G 状态腿 |

**graphs=auto 的降级条件**(降级会写进 `--check` 的 WARN):
f16 池(没有 FP8 读核)、`kvmem=overcommit`(选择层视图与 FI 图不共戴)、
TP 档。降级后仍保留普通 decode 图,但 FI 臂/桶关。

## 4. 内存模型(`context = "max"`)

与引擎治理器 `[vram-gov]` **同一算式**,再补两项治理器不知道的:

```text
target_total = total × vram_target
fixed        = gdn_row × (gdn_slots + snap_max) + 图 slab 640M + warmup 138M
budget       = target_total − reserve − fixed − live(权重/运行底价)
usable       = budget − activation − stability_margin
max_seq      = usable / per_tok(向下取 4k)
```

- `activation` 按块档:`≤512 → 0.60G`、`≤1024 → 1.05G`、`>1024 → 1.70G`
  (最后一项 = 没有桶、eager 全开;`512+512` 按 512 档)。
- `stability_margin = 1.30G`(覆盖已知 turn 临时块泄漏 + 激活抖动;
  144k/160k 的实测曲线见设计卷 §7.2)。
- 3090 Ti + 27B AWQ-INT4 + FP8 KV + spec off + `512+512` ⇒ **128k / 池 4G**,
  与 `profiles/qwen38_27b-3090-w4a16.toml` 同档(实测 prefill 1436 / decode 47.55)。
- **并发 C**:无缓存→min(`[vram-gov]` cap, 上下文 × C + 1 页);缓存开→cap(整池给缓存);
  非超卖档只要求池装得下**单会话整条上下文**(Strict 准入),多会话共享池。
  `max_batch=C(≤8)` 自动启用合批;**≥2 活跃 turn 的批路走 v2 注意力**
  (FI 臂只服务 T=1 单行),`--check` 会写明这条告警。
- **2026-10-22 起池容量无配置项**:单卡口径 `budget = free(cuMemGetInfo)
  − reserve − fixed_after`;`fixed_after` 含 `post_pool_extra`(图/plan/
  cudaMallocAsync 保留/驱动碎片/首请求 Temp 的实测固定项)与 `guard slack`。
  `max_seq` 超过治理器 `cap − 1 页` 直接拒启,并报可服务的 `ctx_max`。
- **DFlash2 草稿环**:`[spec].draft_window`(0 = auto = 检查点
  `dflash_config.sliding_window`)。草稿 KV 不再按池容量全量物化,
  而是 **窗口大小的环形缓冲**(2048 → 固定 ~40MiB;写槽 = 逻辑位 % 窗口;
  propose 自块/被拒行写环外 scratch)。当前 NC 核只实例化 **2048**,
  写 4k/8k 会 boot 拒(等核加运行时 window);环容量本身已按检查点/配置取值,
  不写死。

**显式覆盖优先**:`--set runtime.max_seq=147456` 不再求解;若超过引擎
治理器 `cap − 1 页`,boot 硬拒并报 `ctx_max`;若只是 `--check` 估算余量
< 1.30G,`--check` **告警但不拒**(长跑泄漏是另一案)。

## 5. 站点层(`--site`)

站点档写本机事实(不入库),两种形态都认:

```toml
[site]
model_dir = "/…/Qwen3.8-27B-AWQ-INT4"          # → [model] path
draft_dir = "/…/Qwen3.8-27B-DFlash2-W4A16"     # → [spec] draft_path
bind      = "127.0.0.1:8139"                   # → [server] bind
device    = "42:00.0"                          # → [parallel] device = ["42:00.0"]
```

也接受普通配置表(直接写 `[model].path` / `[server].bind` / `[parallel].device`)。
卡位纪律不变:**用 BDF/UUID/名字,不用 CUDA 数字序**。

## 6. 常用命令

```bash
# 预检(不加载权重,~10–20ms):派生 + 内存账 + 模型/草稿几何
# 27B 必须给站点档(无缺省资产);有 FAIL ⇒ 退出码 1,报告一次列全
server -p profiles/qwen38_27b-3090-w4a16.toml --site testdata/args/base.toml --check

# 生效配置 + 逐键来源 / 内存估算(JSON)
server -p profiles/… --site … --print-config --format json | jq '.provenance["runtime.max_seq"]'

# 单键解释:值 / 来源 / 派生规则
server -p profiles/… --site … --explain runtime.max_seq

# 临时覆盖(替代启动脚本的 extends 拼档;可重复)
server -p profiles/… --site … --set runtime.max_seq=65536 --set 'parallel.device=["41:00.0"]'

# 查看源码里的常用档
ls profiles/
```

来源(`provenance`)五类:`default` / `file:<路径>` / `site` / `cli` /
`derived:<规则>`。`--print-config` 永远给**生效值**,不是意图原文。

## 7. 加新档

1. **新建一个只写差异的档**:`extends = ["full.toml"]` + 要改的键
   (或复制 `profiles/` 里最接近的档再改);
2. 先 `--check`(无 GPU 也能跑),看清派生值与内存账;
3. 真机 boot 后对照 `[vram-gov]` 与 `--print-config` 的估算;偏差大时先查
   `live`(权重是否换了家族)/ `fixed_after`(并发/快照);
4. 实验档用 `-p <path> <overlay.toml>` 或 `--set`(不再有 legacy 目录)。

## 7.1 视觉塔(`[model].vision`)

- **HF/AWQ 档**:`[model] vision = true` ⇒ 装载检查点 `config.json:vision_config`
  的 `model.visual.*`(27B AWQ 档;0.8B 响亮拒)。
- **GGUF 档**(2026-10-21):文本 `.gguf` 只带主干;`vision = true` 时用
  `[model] vision_path = "…/mmproj-*.gguf"` 指向视觉塔。缺省会在 `model.path`
  同目录找**唯一** `mmproj*.gguf`;多个候选/没有 = 装载期响亮拒并列出候选。
  视觉塔走 host 解量化(F16/F32/Q8_0… → F16),文本主干仍走 GGUF 零解量化
  通道;patch_embd 的两个时间片按 HF `[out][in][t][p][p]` **按 in 通道交错**
  还原(直接 concat = 静默错值)。
  - `profiles/qwen35_4b-3090-q4km-vlm.toml`(4B Q4_K_M + mmproj-F16,单卡 3090 Ti)
  - `profiles/qwen38_27b-3090-q4km-vlm.toml`(27B Q4_K_M + mmproj-Q8_0,单卡 3090 Ti)
  - 把 `vision_path` 写到 HF/AWQ 档 = 响亮拒(塔权重在检查点目录);
    `vision = false` 却写 `vision_path` 同样拒。
- 视觉塔装载 + 引擎图像 prefill + **OpenAI `image_url` / Rust 预处理**已接
  (data URL;单图;TP2 两 rank 各备塔输出)。
- **MTP 头投机**（`[spec] draft = "mtp"`，2026-10-21）：27B AWQ 检查点自带
  15 个 `mtp.*` 键；`depth ≤ 7`。MTP 链是独立 **F16** 页池（不是主池 fp8），
  标准 attention 读写走 `kv.main.f16`/KnHD；`--check` 会预检权重索引的
  `mtp.*`。验证档 `profiles/qwen38_27b-3090-w4a16-mtp.toml`；历史 DFlash2 档不受影响
  （`draft` 缺省 = auto，DFlash2 目录在场时仍优先 DFlash2）。
  - **MTP 窗口（2026-10-21，Phase 0）**：`[spec] mtp_window = N`（0 = 关，默认）
    只改 MTP **草稿读**的 KV 视图（最老 `mtp_sink` 个 token + 最近 N 个 token），
    target verify 始终全上下文。按池页对齐，窗口只开给 MTP；配到
    DFlash2/Off、缺 v2 scratch、或 `mtp_sink >= mtp_window` 都会 boot 拒。
    生产 MTP 档 `profiles/qwen38_27b-3090-w4a16-mtp.toml` 已写
    `mtp_window=4032` / `mtp_sink=64` / `mtp_window_min_ctx=12288`；A/B 回退用
    `--set spec.mtp_window=0`。`full.toml` 只保留 schema 缺省 0/64/0。
    `mtp_window_min_ctx` 是短 ctx 不压缩的门槛（8k prompt+输出会跨 8192，
    取 12288 才真正保 full）。
  - **自适应 depth（2026-10-21/22，T1a+T1b v2）**：`[spec] adaptive_depth = true`
    （默认 false）+ `depth_min = 3` + `depth_switch_ctx = 49152` +
    `adaptive_min_tpr = 3.0`。`depth` 为上限（建议 7）；`pos >= depth_switch_ctx`
    回下限。两档 verify/fold/propose 图族 boot 期全量预捕获（d3+d7 共 24 face），
    运行期只换引用；捕获不全 = boot 硬失败。T1b v2 = 每 8 轮窗口，连续 2 窗
    `tokens/round < adaptive_min_tpr` 才只降不升到 `depth_min`。
    A/B 命令见本地实测档(未入库) §8。
    `qwen38_27b-3090-w4a16-mtp.toml` 保持静态 d3 + window；
    `qwen38_27b-3090-iq4xs-mtp.toml`（GGUF 单卡）已开 window + T1b 自适应。
  - **DFlash2 lookup（2026-10-22，仅 DFlash2）**：`[spec] lookup = true`
    （默认 false）+ `lookup_nmin = 6` / `lookup_nmax = 12` /
    `lookup_nstrong = 6` / `lookup_agree = 0`。在当前请求的
    `prompt_ids + out + 本轮 verify 块` 里找当前后缀最近一次出现，用其后继
    token 覆盖草稿头（head-only；verify 图仍 depth+1 定形）。非 DFlash2 档
    写 `lookup=true` = boot 拒；`nmax < nmin` / `nmax > 128` /
    `nstrong > nmax` / `agree > SPEC_DEPTH_MAX` = 配置拒。
    `profiles/qwen38_27b-3080_tp2-w4a16.toml` 已转正 `lookup=true`
    （W4A8/VLM 继承）；回退 `--set spec.lookup=false`。
    实测与实现见 `docs/devlog/perf/tp2-decode-fast.md` §10。
- **图像 × 投机**(`spec.depth>0`)已接(V5):verify 的 rope 位置 =
  真实位置 + m-RoPE `delta`,KV slots/lens 仍用真实位置;带图请求仍限
  单图/非合批。无投机对照档 = `profiles/qwen38_27b-3080_tp2-w4a16-vlm.toml`;
  视觉+DFlash2 d7 = `profiles/qwen38_27b-3080_tp2-w4a16.toml`;
  （262k 全设备池档按新 schema 另行生成;视觉塔算子由 `feat.vision` tag + 配置派生 TagFilter 自动放行,
  不再手抄 `allowed_ops`/`VISION_OPS`)。
- `--check` 的 HF/AWQ 分支检查 `vision_config`/`image_token_id`;GGUF 分支改读
  文件 KV(arch/hidden/layers/kv_heads/head_dim/vocab/ctx)并打开 `vision_path`
  校验塔层数/out_hidden/`<|image_pad|>` id。详见
  `vision-wiring.md` 与
  `gguf-format.md` §23.14。

## 7.2 前缀无感删除(图像段;2026-10-23)

**两个面缺一不可**(引擎不自作主张):

| 面 | 形态 | 说明 |
|---|---|---|
| 引擎配置 | `[pool] image_evict = false` | 缺省关 = 逐字节旧行为。开 ⇒ boot 门要求**视觉档 + f16 分页 KV + 前缀缓存**,任缺**拒启** |
| 客户端标记 | 图像 content part `"owl_evict": true` | 「这张图将来可能从历史里删」。未标记 = 零行为差异;带标记但档位没开 = 提交**响亮拒** |

- **生效形态**:带标记的图像段被补 `\n` 到「段首页对齐 + 整段整页」;下一轮若
  历史里**正好删掉这一段**,引擎保留后缀页 + 只把后缀 K 乘 `R(−S)`(位置帧平移),
  **不重算删点之后的 token**。
- **标记冻结律**:一张图打了标记后,留在历史期间标记必须保持不变(pad 数由标记
  决定,改标记 ⇒ 前缀身份断 ⇒ 退化为重算,不会错值)。
- **不适用**:`kv_fp8 = true` 档(boot 拒;fp8 的 dequant→rotate→requant 另案)。
- 语义:删点**之前**逐位等价;删点**之后**是**搬运记忆**(后缀 KV/GDN 仍带着被删
  图的上下文)。设计与门:`memory/prefix-deletion.md`。

## 8. 兼容与挂账

- **`profiles/` 全部档 = `extends = ["full.toml"]` + 差异**(无 `[intent]` ⇒ 派生器不介入);
- **schema v2(2026-10-21)退役键**会给迁移提示:`[dispatch]`→`[kernel]`、
  `[agent]` 解散、`[graph] no_graph`→`[cuda] graph`、`model.kind/dir`→`model.format/path`、
  `model.dflash2_dir`→`spec.draft_path`、`runtime.tp_*`/`device*`→`[parallel]`、
  `server.max_concurrency`→`runtime.max_concurrency`;
- `OWL_MAX_SEQ` / `OWL_PREFILL_CHUNK` **已删除**(2026-10-20):它们会在
  意图派生后再改一次,造成 `max_seq` 双源。临时调压请
  `--set runtime.max_seq=…`(池自动跟随剩余显存)。
- **allowed_ops 已退役**(2026-10-20):配置按 `model.format/quant/head_quant`、
  `pool.kv_fp8`、`spec.depth`、`model.vision`、`parallel.mode` 自动派生
  tag filter;`[kernel] tag_wildcard = true` 是诊断 `*` 逃生门。
- **P2 未做**:旧档迁移器、`context=max` 校准探测、长跑泄漏余量曲线。
- 配置层 `deny_unknown_fields` 仍全 module 生效;退役键有专门迁移提示。
- 测试链(`#[owl_test]` / `OWL_TEST_CONF`)不变:同一 loader;
  环境档基准用 `extends = "<path>"`,不再有 `flavor` 名字键。

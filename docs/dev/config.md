# 配置链:意图 → 派生 → 预检 → 解释(docs/dev)

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
cp profiles/tp27b-3080-184k.toml ~/my-27b.toml
$EDITOR ~/my-27b.toml          # 改 model.dir / runtime.bind / tp_ordinals
./target/release/server -p ~/my-27b.toml
```

`profiles/` 下全部是可直接用的**完整档**。历史 `legacy/` 目录已于
2026-10-21 删除,`tp27b-3080-262k.toml`(引用已退役的顶层 `kv_fp8`)一并删除。

## 1.1 配置档必须**全字段显式**（2026-10-21 裁决）

- 用 `-p <profile>` 给的档，**每一个键都必须在文件里写出来**；缺一个 ⇒ **拒启**
  （错误里一次列出全部缺项）。代码缺省不再兜底 —— 旧行为让各会话跑在
  "半份配置 + 旧缺省"上（4k / F16 KV / 无投机 / 无图），启动点不可复现。
- 站点层（位置参数 / `--site`）与 CLI（`--set`）仍是**覆盖层**：它们只覆写，
  不负责补全。
- 生成完整档的最快方式：

  ```bash
  server -p <旧档> --print-config > <新档>.toml
  ```

- 卡位**必须显式**：`runtime.tp_ordinals = [a, b]`（TP 双卡）或
  `[runtime.device] bdf/uuid/find_name`（单卡）。不再自动挑"任意空闲卡"。

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
| `model` | `[model] kind/quant/head_quant`(与显式值冲突 = 拒) |
| `goal=max-context` | 复用 off、`512+512`、`vram_target=0.99`、`reserve=256M` |
| `goal=max-speed` | 复用 global、`1024+0`、贴顶 |
| `goal=balanced` | 复用 session、`512+512`、0.97 |
| `goal=low-memory` | kvmem overcommit、复用 global、小激活面 |
| `context=N` | `max_seq=N`;非超卖档 `pool_tokens=N` |
| `context="max"` | 内存模型求解(§4) |
| `concurrency=C` | `gdn_slots=C`、`server.max_concurrency=C`、`max_batch=min(C,8)` |
| `spec=dflash` | `depth=7` + 草稿目录必填 + 草稿池 20KiB/token 进内存账 |
| `kv=fp8/f16` | `dispatch.kv_fp8` |
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
  与 `profiles/awq27b-3090-max.toml` 同档(实测 prefill 1436 / decode 47.55)。
- **并发 C**:`pool_tokens` 是**全局**容量 ⇒ 非超卖档 `pool = 上下文 × C`,
  单会话上下文按 `usable / (per_tok × C)` 求解(例:C=2 ⇒ 每会话 64k / 池 128k)。
  `max_batch=C(≤8)` 自动启用合批;**≥2 活跃 turn 的批路走 v2 注意力**
  (FI 臂只服务 T=1 单行),`--check` 会写明这条告警。

**显式覆盖优先**:`--set runtime.max_seq=147456`(或 `--set pool.pool_tokens=…`)
不再求解;若超过治理器 `cap`,`--check` 硬拒;若只是余量 < 1.30G,`--check`
**告警但不拒**(144k 实测可跑,长跑泄漏是另一案)。

## 5. 站点层(`--site`)

站点档写本机事实(不入库),两种形态都认:

```toml
[site]
model_dir = "/…/Qwen3.8-27B-AWQ-INT4"
draft_dir = "/…/Qwen3.8-27B-DFlash2-W4A16"   # spec=dflash 时必填
bind      = "127.0.0.1:8139"
device    = { bdf = "42:00.0" }              # 或 uuid / find_name
```

也接受普通配置表(`testdata/args/base.toml`:直接写 `[model].dir`/`[runtime].bind`)。
卡位纪律不变:**用 BDF/UUID,不用 CUDA 数字序**。

## 6. 常用命令

```bash
# 预检(不加载权重,~10–20ms):派生 + 内存账 + 模型/草稿几何
# 27B 必须给站点档(无缺省资产);有 FAIL ⇒ 退出码 1,报告一次列全
server -p profiles/awq27b-3090-max.toml --site testdata/args/base.toml --check

# 生效配置 + 逐键来源 / 内存估算(JSON)
server -p profiles/… --site … --print-config --format json | jq '.provenance["pool.pool_tokens"]'

# 单键解释:值 / 来源 / 派生规则
server -p profiles/… --site … --explain runtime.max_seq

# 临时覆盖(替代启动脚本的 extends 拼档;可重复)
server -p profiles/… --site … --set runtime.max_seq=65536 --set runtime.device.bdf=41:00.0

# 查看源码里的常用档
ls profiles/
```

来源(`provenance`)五类:`default` / `file:<路径>` / `site` / `cli` /
`derived:<规则>`。`--print-config` 永远给**生效值**,不是意图原文。

## 7. 加新档

1. **复制 `profiles/` 里最接近的档**到任意路径,再改:优先意图键,
   必要的低层键作覆盖;
2. 先 `--check`(无 GPU 也能跑),看清派生值与内存账;
3. 真机 boot 后对照 `[vram-gov]` 与 `--print-config` 的估算;偏差大时先查
   `live`(权重是否换了家族)/ `fixed_after`(并发/快照);
4. 实验档用 `-p <path> <overlay.toml>` 或 `--set`(不再有 legacy 目录)。

## 7.1 视觉塔(`[model].vision`)

- `[model] vision = true` ⇒ 装载检查点 `config.json:vision_config` 的
  `model.visual.*`(27B AWQ 档;0.8B 响亮拒)。
- 视觉塔装载 + 引擎图像 prefill + **OpenAI `image_url` / Rust 预处理**已接
  (data URL;单图;TP2 两 rank 各备塔输出)。
- **图像 × 投机**(`spec.depth>0`)已接(V5):verify 的 rope 位置 =
  真实位置 + m-RoPE `delta`,KV slots/lens 仍用真实位置;带图请求仍限
  单图/非合批。无投机对照档 = `profiles/tp27b-3080-vlm-184k.toml`;
  视觉+DFlash2 d7 = `profiles/tp27b-3080-184k.toml`;
  **262k 全设备池 + 视觉 + d7** = `profiles/tp27b-3080-262k-vlm.toml`
  (单文件完整档;视觉塔算子由 `feat.vision` tag + 配置派生 TagFilter 自动放行,
  不再手抄 `allowed_ops`/`VISION_OPS`)。
- `--check` 会检查 `vision_config`/`image_token_id` 并打 WARN;详见
  `docs/arch/vision-引擎与服务面-接线-设计与施工.md`。

## 8. 兼容与挂账

- **低层档(`profiles/awq27b.toml`、`spec-dflash.toml`)/ testdata 原样可跑**(无 `[intent]` ⇒ 解析器不介入);
- `OWL_MAX_SEQ` / `OWL_PREFILL_CHUNK` **已删除**(2026-10-20):它们会在
  意图派生后再改一次,造成 `max_seq` 与 `pool_tokens` 双源。临时调压请
  `--set runtime.max_seq=…`(池会跟随,除非显式给 `pool.pool_tokens`)。
- **allowed_ops 已退役**(2026-10-20):配置按 `model.kind/quant/head_quant`、
  `dispatch.kv_fp8`、`spec.depth`、`model.vision`、`runtime.tp_ordinals` 自动派生
  tag filter;`[dispatch] tag_wildcard = true` 是诊断 `*` 逃生门。
- **P2 未做**:旧档迁移器、`context=max` 校准探测、长跑泄漏余量曲线。
- 配置层 `deny_unknown_fields` 仍全 module 生效;退役键有专门迁移提示。
- 测试链(`#[owl_test]` / `OWL_TEST_CONF`)不变:同一 loader;
  环境档基准用 `extends = "<path>"`,不再有 `flavor` 名字键。

# 配置链:意图 → 派生 → 预检 → 解释(docs/dev)

> 动手改档/加实验档前先看这一页。**配置的定义面唯一在
> `crates/shared/src/config/`**(键的账本 + 类型 + 缺省);本页讲"怎么用"。
> 纪律:**引擎行为不许用 env 控**(env 只剩 `OWL_CONFIG` 路径兜底 + 测试域自读);
> 一切可配项走配置文件/CLI 显式声明。
>
> 2026-10-20 起有两条档位路径:**旧 flavor(低层键)** 与 **新 profile(意图)**。
> 旧档原样可跑;新档只写 6–10 个意图键,低层键由解析器派生。

## 1. 两条路径

| | 旧:flavor | 新:profile(意图) |
|---|---|---|
| 文件 | `flavors/*.toml` | `profiles/*.toml` |
| 内容 | 低层键(池/上下文/图臂/白名单…) | `[intent]` 6–10 键 |
| 入口 | `--flavor <名>` | `--profile <名>` |
| 预检 | 只做配置校验 | `--check` 派生 + 内存账 + 模型几何 |
| 白名单 | 手写 `allowed_ops` | 不写(P1 才从臂集合派生;当前 = 门关) |

两条路径共用同一个 loader/语义门;`--profile` 与 `--flavor` **互斥**。
旧档迁移(P2)之前,行为面临时实验仍可写低层键(profile 里也允许,见 §6)。

## 2. 层序(后面的覆盖前面的)

```text
Default(代码缺省)
  → Flavor 或 Profile(基准档;--flavor/--profile)
  → 位置参数 CONFIG(历史用法:站点/实验覆盖)
  → --site <PATH>(站点层:路径/卡位/端口)
  → CLI(--set / --log-* / --perf-level / --step-gate*)
  → [intent] 派生(只填空:显式低层键优先)
```

注意最后一步的顺序:**显式低层键覆盖意图派生值**(高级覆盖),派生只填
"没人写"的键。`--set`/CLI 旗标与文件同一条链,所以
`--print-config` 看到的**就是** boot 用的配置(不再有 print 之后二次覆盖)。

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
| `graphs=auto` | FI 单图 decode + prefill 桶族 + `fi_plan_bake=0` + 窗刷表 |
| `graphs=off` | `no_graph=true`,FI 臂/桶全关 |
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
  与 `flavors/awq27b-3090-max.toml` 同档(实测 prefill 1436 / decode 47.55)。
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
server --profile awq27b-3090-max --site testdata/args/base.toml --check

# 生效配置 + 逐键来源 / 内存估算(JSON)
server --profile … --site … --print-config --format json | jq '.provenance["pool.pool_tokens"]'

# 单键解释:值 / 来源 / 派生规则
server --profile … --site … --explain runtime.max_seq

# 临时覆盖(替代启动脚本的 extends 拼档;可重复)
server --profile … --site … --set runtime.max_seq=65536 --set runtime.device.bdf=41:00.0

# 列表
server --list-profiles
server --list-flavors
```

来源(`provenance`)五类:`default` / `file:<路径>` / `site` / `cli` /
`derived:<规则>`。`--print-config` 永远给**生效值**,不是意图原文。

## 7. 加新档

1. **优先写意图档**(`profiles/<名>.toml`):只写意图 + 必要的低层覆盖;
2. 先 `--check`(无 GPU 也能跑),看清派生值与内存账;
3. 真机 boot 后对照 `[vram-gov]` 与 `--print-config` 的估算;偏差大时先查
   `live`(权重是否换了家族)/ `fixed_after`(并发/快照);
4. 旧档要实验:仍可用 `--flavor … <overlay.toml>` 或 `--set`。

## 8. 兼容与挂账

- **旧 flavor / testdata 原样可跑**(无 `[intent]` ⇒ 解析器不介入);
- `OWL_MAX_SEQ` / `OWL_PREFILL_CHUNK` **已删除**(2026-10-20):它们会在
  意图派生后再改一次,造成 `max_seq` 与 `pool_tokens` 双源。临时调压请
  `--set runtime.max_seq=…`(池会跟随,除非显式给 `pool.pool_tokens`)。
- **P1 未做**:`allowed_ops` 从臂集合派生、启动一行摘要、`/debug/config`、
  `--list-profiles --verbose`;**P2 未做**:旧档迁移器、`context=max`
  校准探测、长跑泄漏余量曲线。
- 配置层 `deny_unknown_fields` 仍全 module 生效;退役键有专门迁移提示。
- 测试链(`#[owl_test]` / `OWL_TEST_CONF`)不变:同一 loader,
  `flavor = "<名>"` 仍是测试环境档的基准写法。

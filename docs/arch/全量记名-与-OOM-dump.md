# 全量记名 与 OOM dump(设计与施工)

> 立项 2026-10-16。**前因**:`a374343` 已经给 OOM 现场加了三笔**总量**账
> (池 used/reserved、驱动 `mem_get_info`、引擎 `vram::live`)—— 它们能回答
> "多满",**回答不了"谁占着"**。用户令:**先全量记名,再做 OOM dump**。
> 分支 `feat/alloc-naming`(从 `c54ff16`)。
> 相关:`piecewise-断点图-设计与实现.md` §12(显存账/对拍/结案/门纪律)、
> `charter.md` A5(严格显存预算)、`common-dev-pitfall.md`(门纪律)。

## 1. 一句话

**每个显存分配点必须报出自己的语义名**(核名/层名/装载键名);名字随命令面
走到**块账房**,成为块的身份字段;OOM 与查询面按名字聚合,一行给出
"谁占了多少"。

## 2. 为什么必须"全量"而不是"抽样"

OOM 现场最需要回答的是**这个正在失败的分配属于谁**、以及**谁把显存吃光了**。
任何"部分记名"都会退化成:OOM 现场一半条目是 `(未命名)`,归因重新变成猜。
所以本设计的硬要求是 **零未命名** —— 并且这个不变式必须**可判定**:

| 门 | 判据 | 失败长相 |
|---|---|---|
| `debug_assert!(!name.is_empty())` | `new_block`/`carve_block` 入口 | debug 构建当场炸(带 stack 指到调用点) |
| 按名清单尾 `⚠ 未命名块 N` | 渲染面把空名单独计数并上屏 | OOM 行里出现 `(未命名!)` 或计数 > 0 |
| `gpu_server` 步骤 13 | 真机:全带名 → 断言 `未命名块` 计数为 0 | 端到端门红 |
| 编译期 | `Command::Alloc/Htod/AllocPinned` 与 `DeviceClient::*` 都是**必填参数**(无默认值/无 `Option`) | 漏传 = 编译不过 |

**为什么不用默认值兜底**(否决方案):给了默认名(如 `"unknown"`)就等于允许
漏报,而漏报恰好是本设计要消灭的东西 —— 家规"拒绝静默兜底"在此的直接应用。

## 3. 名字怎么走(四层贯通)

```
调用点(语义名:核名/层名/装载键名)
   ├─ models / engine  → DeviceClient::{alloc, alloc_uninit, htod, htod_f32, alloc_pinned}(name)
   └─ kernels 家族核    → DeviceRes::{alloc, alloc_block}(tag/name)
                       ↓
              Command::{Alloc, Htod, AllocPinned} { name: String }
                       ↓  (actor 信封;捕获期与图外两条派发臂都已接通)
        ┌──────────────┴───────────────────────────┐
        │ 块块房 BlockEntry.name                    │  scratch 账本
        │ (GpuCtx.blocks:图外 Owned + 捕获 Carved)   │  (u64 裸指针, 名字同行)
        └──────────────┬───────────────────────────┘
                       ↓
        render_block_report(按名聚合) ← 单源渲染
                       ↓
   OOM WARN(`vram-ledger`,三账 + 按名清单)
   Command::LedgerReport(查询面:排障/回归门)
```

**要点**:
- 名字是**分配点的语义**,不是尺寸/相位(那些账上本来就有)。
- **池命中(复用)不改名**:块 id 不变 ⇒ 账房保留**首次出生**那次的名字。
  语义正确性:块身份(= 图/槽的同一个位置)没变,同一块被复用说明"这个槽
  就是干这个的"。(若将来需要精确到"本轮用途",再加 rename 面 —— 现在不做。)
- **scratch 也要记名**:家族 runtime 的 `res.alloc` 分配是**裸指针**,不进块
  账房 ⇒ 单独记 `(ptr, slice, name)` 三元组,否则家族开销在 dump 里隐身。

## 4. 渲染口径(`render_block_report`)

纯函数 `Iterator<(&str, usize)> → String`(与显存账本体解耦 ⇒ 单测零 CUDA):

- **按名聚合**:同名合并(字节求和 + 块计数);
- **字节降序**:谁占大头在前;
- **`top-N` 截断**:其余折入 `…另 N 类`(⚠️ 排障时要给够 top —— 小块的
  名字会被大块挤出去,`gpu_server` 步骤 13 首版就因此假红过一次);
- **单位律**:≥1MiB 用 MiB,否则 KiB(一刀切 MiB 会把 KB 级表件渲染成
  `0.0MiB` 而丢失量级);
- **空名响亮**:渲染成 `(未命名!)` **且**单独计数 —— 漏网不能静默混入正常名字。

## 5. 命名规约(实测落地的一套;新调用点照此)

| 层 | 前缀 | 例 |
|---|---|---|
| 装载域 | `wt.<检出键名>` | `wt.model.layers.12.mlp.down_proj.weight` |
| 装载暂存 | `staging.<臂tag>` / `pin.<键名>` / `repack.<键名>` | `staging.cast_bf16_in` |
| KV 池 | `kv.{k,v}.<层号>` | `kv.k.0` |
| 草稿池 | `dfk.{k,v}.<层号>` | `dfk.k.3` |
| MTP | `mtp.kv.{k,v}` | |
| 解释器输出 | **核名 / 内建类名**(`class`:即 `probe_name` 那一套) | `marlin.w4a16` / `owl_norm_rope_f16` / `gdn_chunked` / `gemm` |
| 图槽 | `plan.in.<槽名>` | `plan.in.frontier` |
| 断点钉块 | `piecewise.bp.<节点id>:<dtype>` | |
| GDN | `gdn.{table,conv_*,recurrence,slot.*,ring.*}` | |
| 注意力 v2 | `attn2.{exp_sums,max_logits,tmp_out}` / `vattn2.*` | |
| XQA / FI 表件 | `xqa.{pt,mask,sem,scratch}` / `fi.{q_cu,indices,indptr,last_len}` | |
| 探针临时块 | `probe.kv_page.<tag>` | |

**两条命名路径**(2026-10-16 二次落):

| 路径 | 用在哪 | 名字长什么样 | 为什么 |
|---|---|---|---|
| **语义名**(手写) | **生产代码** | `marlin.w4a16` / `wt.<键名>` / `plan.in.frontier` | OOM 现场要回答"谁占着" ⇒ 名字必须是**语义**(核名/层名/键名) |
| **位置名**(宏) | 测试 / 探针 / example | `alloc@crates/backends/cuda/tests/gpu_server.rs:313:17` | 这些地方没有语义可得;要的是"**哪一处**分配的" |

宏在 `owl-iface::contract`(`#[macro_export]`,5 个方法各一臂)。

**为什么敢叫 `alloc!` 这种"通名"**(2026-10-16 复核,四条都查过):
调用点**一律带 crate 路径**(`owl_iface::alloc!`)⇒ 后缀是冗余的;
全仓无 `macro_rules! alloc` 同名定义、无以 `alloc` 为名的模块/函数、
无 `extern crate alloc`/`use alloc::`(Rust 的 `alloc` crate 未被使用 ⇒
不存在类型/宏命名空间歧义)、std 也没有 `alloc!` 宏。改名只是把冗余后缀去掉。

```rust
owl_iface::alloc!(client, Dtype::F32, 8)          // 若有需额外语义后缀 …见下
owl_iface::alloc_uninit!(client, Dtype::U32, n)
owl_iface::htod!(gpu, Dtype::F16, &shape, &bytes)
owl_iface::htod_f32!(gpu, &shape, data)
owl_iface::alloc_pinned!(gpu, len_bytes)
```

**它解决的三个具体毛病**:
1. **行号会漂** —— 手写 `"文件::函数#N.方法"` 里若含行号,改一行就指错地方
   (比没有名字更坏:它会误导排障)。宏的 `file!()`/`line!()`/`column!()`
   由**编译器**在调用点展开,永不漂。
2. **名字要唯一** —— 五臂宏各自用 `方法@位置` 形态 ⇒ 天然唯一,不必手写编号。
3. **调用点变短** —— `client.htod(Dtype::F16, &sh, &b, "gpu_server::step_x#1.htod")`
   → `owl_iface::htod!(client, Dtype::F16, &sh, &b)`(160 处迁移)。

**纪律**:两条路都落进同一个 `BlockEntry.name`,口径不分裂;生产点**仍写语义名**
(位置名对"谁占着"没有信息量)。`gpu_server` 同时保留一处**显式名**调用
(`ledgerprobe.*`)作正向对照 ⇒ 两条路各有门覆盖。

### 5.1 迁移的一个实测坑(角度括号)

扫描器要把"最后一个实参 = 名字"切出来,得按**顶层逗号**切分;而深度跟踪若把
`<<`(位移)当泛型尖括号,`alloc_pinned(64 << 20, "…")` 的逗号就落在"深度 2"上
⇒ 那处**静默漏迁**(全仓最后剩的 2 处就是这么来的:第二轮 0 edit 才暴露)。
修法:`<<`/`>>` 成对跳过,不涨深度。

## 6. 落地清单(2026-10-16;`feat/alloc-naming`)

| # | 落点 | 内容 |
|---|---|---|
| 1 | `iface/src/contract.rs` | `DeviceClient::{alloc, alloc_uninit, htod, htod_f32, alloc_pinned}` 全部收 `name: &str`(必填);缺省实现一并带名 |
| 2 | `cuda/src/command.rs` | `Command::{Alloc, Htod, AllocPinned}` 带 `name: String`;新增 **`LedgerReport { top, ack }`** 查询变体;`kind()`/`created_at()` 同步 |
| 3 | `cuda/src/gpu_client.rs` | 三处客户端实现透传名字;新增 `pub async fn ledger_report(top)` |
| 4 | `cuda/src/state.rs` | `BlockEntry.name`;`new_block(slice, name)` / `carve_block(n, name)`(**debug_assert 空名拒**);`render_block_report` / `count_unnamed`(纯函数) |
| 5 | `cuda/src/server/mod.rs` | 两条派发臂 + Closing/拒绝臂 + 发射账条目;**`ledger_report()` 单源渲染**;OOM WARN 追加"按名清单";`scratch_report_by_name`(家族裸指针账) |
| 6 | `cuda/src/server/harvest.rs` | `handle_htod/try_htod/handle_alloc_pinned` 透传名字;`Staging::alloc(label)` 收 `&str` |
| 7 | `kernels/src/device.rs` | `DeviceRes::{alloc, alloc_block}` 的 `tag: &'static str` → `name: &str`(家族可拼动态名) |
| 8 | models(11 文件)/ engine(4 文件) | 全部生产分配点填语义名(见 §5 规约) |
| 9 | `cuda/tests/gpu_server.rs` | **步骤 13**:真机端到端——名字进账房 + 零未命名;步骤 10 加**窗内(carve)读账**断言 |
| 10 | 本文 | 设计/口径/规约/门 |

## 7. 验收(真机,2026-10-16)

| 门 | 结果 |
|---|---|
| `cargo check --workspace --all-targets --exclude server` | **0 error** |
| `owl-cuda --lib` 单测 | **15/15**(新增 5 条:聚合降序 / 空名响亮 / 空账良构 / JSON 形状+转义 / 落盘路径三态) |
| `owl-cuda --test gpu_server`(真机) | **2/2**(步骤 13:记名端到端 + 窗内读账 + JSON 真解析;步骤 14:峰值快照门 —— 门关 skip/门开落 5 份阶梯) |
| `owl-models --lib` | 160/160 |
| `owl-kernels --lib` | 72/72 |
| `owl-engine --lib` | 42/42 |
| e2e(真机,**全场景**) | **全绿,86.7s**:`two_turns` / `schedule_decision_surface` / `gdn_slot_cross_probe` / `08b_fuse_text_parity` / `longctx_fi_direct_gate` / `w4a16_marlin_e2e` / `spec_identity_gate` / `piecewise_segments` / `27b_raw_completion_diag` / `decode_marginal_bench` / `spec_identity_gate_27b` / `dflash2_identity_gate_27b`(仅 `27b_chat_inference`+`awq27b_marlin_e2e` 因缺 `OWL_AWQ27B_DIR` 整组 skip) |

**实测按名清单样例**(真机 0.8B 窗内读账):

```
在册 26 块 / 合计 16.0MiB(按名 top64): gpu_server.L320.alloc=16.0MiB×1
 gpu_server.L151.htod=16.0KiB×1 … ‖ 家族 scratch 7 笔/96.0MiB(按名 top8):
 fi.float_ws=64.0MiB×1 fi.int_ws=32.0MiB×1 xqa.consts=0.0MiB×3 …
```

## 8. 结构化落盘(OOM dump 的另一半;同日落地)

人读的一行(`ledger_report`)够排障,但**跨时间比对**与**脚本消费**(启动前期 vs
稳态、OOM 前 vs OOM 时)需要机器可读 ⇒ 同一现场另给 JSON。

**形状(嵌套表达归属)**

```json
{
  "blocks":  {"blocks":[{"name":"kv.k.0","bytes":16777216,"blocks":1}],
              "total_bytes":…,"total_blocks":…},
  "scratch": {"scratch":[{"name":"fi.float_ws","bytes":67108864,"blocks":1}],
              "total_bytes":…,"total_blocks":…}
}
```

| 面 | 用途 |
|---|---|
| `Command::LedgerJson` / `GpuClient::ledger_json()` | 查询(纯读,捕获期放行;脚本直接消费) |
| `OWL_VRAM_DUMP=1`（或给路径） | **OOM 时自动落盘** `/tmp/owl-vram-{ms}.json`(家法同 `prompt_dump`;失败只记一行 log,不覆盖原 OOM 错) |

**实测样例**(真机 0.8B,窗内查):`blocks 36 块/16904644B ‖ scratch 22 笔/100844840B`
—— 家族 scratch(fi/xqa 工作区)占了现场的 **86%**,没有它的一半,现场会严重误导。

### 8.0 峰值快照(门 `OWL_VRAM_PEAK_MB`;默认关)

OOM 那份是"**现场**",峰值快照是"**爬到现场的阶梯**" —— 只看 OOM 那一刻
回答不了"是稳态就高,还是某一步突然涨上去的"。

| 旋钮 | 语义 | 缺省 |
|---|---|---|
| `OWL_VRAM_PEAK_MB` | live 越过该水位(与**上次水位 ×1.05** 比较)就落一份 | **未设 = 关**(零开销零文件) |
| `OWL_VRAM_PEAK_MAX` | 本次进程最多落几份(有界,防刷 `/tmp`) | 8 |

落点 `/tmp/owl-vram-peak-{ms}-{MiB}MiB.json`;判定在 state 侧纯函数
`peak_should_dump(live, threshold, watermark, written, max)`(离线单测覆盖
未到门/步进 4% 不写/步进 6% 写/写满不写/门值 0 关)。

**真机实测阶梯**(`OWL_VRAM_PEAK_MB=1`,gpu_server 合同测试):

| 文件 | live | phase | blocks | scratch |
|---|---|---|---|---|
| `…-64MiB.json` | 64MiB | graph | 14 | 7 |
| `…-384MiB.json` | 384MiB | graph | 22 | 7 |
| `…-704MiB.json` | 704MiB | graph | 32 | 7 |
| `…-768MiB.json` | 768MiB | graph | 35 | 22 |
| `…-832MiB.json` | 832MiB | graph | 37 | 22 |

这份阶梯把两件事直接暴露出来:①块数随水位单调增(粒度可追);②scratch 在
**倒数第二份**才从 7 跳到 22 —— 家族 runtime 是**按需**首次分配(不是启动即占)。

### 8.1 `meta` 段(每份现场自带上下文)

```json
{"meta":{"ts_ms":1792124456019,"phase":"graph","live_bytes":805392960}, …}
```

- `ts_ms` 墙钟(时钟回拨 = 0,非失败路径);
- `phase` ∈ {`eager`,`graph`,`capture`}(有界词表:`capture_stream()` 优先,
  其次已登记图数 > 0 ⇒ `graph`,否则 `eager`);
- `live_bytes` = 引擎账(`vram::live`)当刻值 —— 与文件名里的 MiB 同源互证。

### 8.2 三个真踩过的坑(都由门抓出,不是推理)

1. **JSON 的"配对"靠嵌套,不靠顺序**。首版拼成
   `{"blocks":[…],"total_bytes":B,"total_blocks":N,"scratch":[…]}`——
   `block_json` 自带的 totals 落在了 `scratch` **之前** ⇒ 读者(和 Python
   `json.load`)会把它当成 scratch 的汇总。修法:每段自带外壳
   (`{"blocks":{…},"scratch":{…}}`)。
2. **别用尾字符判据**。测试首版写 `js.ends_with("}}")` 并据此判红过一次 ——
   而服务端实测同一串 `open/close` 各 56 **完全配平**(拼装是对的),错的是
   "数尾巴"这种脆弱判据。结构化数据一律交给**真解析**(见下)。
3. **"读最近文件"必须与触发条件对齐**(峰值门首版假红):测试只分 8MiB,而
   水位已到 768MiB ⇒ **没破 5% 步进** ⇒ 读到的是**上一次**快照(不含本块名),
   看起来像"触发面断了"。修法 ①分配量给够(64MiB)②判据改"**新文件数 > 前值**"
   (race-free,不猜哪个是"最近")。

### 8.3 真解析门(不引依赖)

`gpu_server` 步骤 13 自带一个 ~80 行的**递归下降 JSON 校验器**
(`assert_valid_json`):覆盖对象/数组/字符串转义/数字/字面量/空白,
失败指向第一个坏字符。配一条离线的 **Python 往返校验**作独立第二意见:

```bash
# 落盘后(或把 ledger_json 的返回存文件)用标准解析器验:
python3 - <<'EOF'
import json
d = json.load(open('/tmp/owl-vram-XXXX.json'))
for k in ('blocks', 'scratch'):
    inner = d[k][k]
    assert d[k]['total_bytes'] == sum(x['bytes'] for x in inner), k
    assert d[k]['total_blocks'] == sum(x['blocks'] for x in inner), k
print('ok', {k: d[k]['total_blocks'] for k in d})
EOF
```

（另:落盘路径解析抽成纯函数 `resolve_vram_dump_path(target, ms)`,三态
`"1"`/`""`/显式路径各有单测 —— 免“写盘那段只能等 OOM 才跑到”。）

## 8.4 "它凭什么是活的"—— 引用计数视图(2026-10-16 补充)

§12.5 留下的那条追问("活着一大片,凭什么?")在本轮补齐:**求值器在引用计数
改动点把 `(users, pending)` 推给探针**,渲染进 `slab 峰值归因` 报告。

**数据流**:`EvalCtx::track_new` / `track_alias` / 死亡减一(三处)→
`slab_hint::probe_refs(id, users, pending)`(探针关 = 一次 relaxed 读,零动作)→
`ProbeState.refs`(插入即留,与 `bytes`/`names` 同律)→ `render_refs`。

**三条判读**(报告尾部一段):

| 视图 | 回答什么 |
|---|---|
| **users 直方图**(u=1 / u=2–3 / u≥4 的块数与字节) | 内存主要被**独占**还是被**一堆共享**抬着 |
| **users 前 3**(名字 + 大小 + `u{u}/p{p}`) | 缩峰值该去找哪个节点(被最多消费者引用者) |
| **⚠ 幽灵块**(`users == 0` 却仍在活跃集) | **回收记账漏口**信号(归零那刻就该走出口);非 0 即 bug |

**可判定性**(为什么不信日志):新增计数器 `slab_hint::refs_pushed()`
(`probe_on` 清零)。门两条:

1. `models::eval_pushes_ref_counts_to_probe`(离线):菱形图推送后计数 > 0;
   **探针关时二次求值零推送**(反向对照)。
2. `engine::gpu_piecewise_segments`(真机):两段计划跑完 `refs_pushed > 0`
   —— 实测 **8 次**(小图),证明真实图路径上数据源活着。

**为什么不做"逐块全表落盘"**:报告按 users 降序取前 3 + 直方图已够定位;
全表(每节点一块)会把日志/JSON 撑到 MB 级,而信息密度极低。要全表用
`OWL_ALLOC_TRACE` 那类逐条探针。

## 9. 未做 / 已知取舍(诚实账)

1. **池命中不改名**(见 §3):复用块显示首次出生的名字。对"谁占着"够用;
   若要"本轮用途"需加 rename 面(热路径每节点一条消息 —— 本轮不做)。
1c. **`pending` 只在死亡点推**(不逐节点推):`pending` 是预波计数,逐节点推会
   把热路径变成 O(节点) 哈希写。因此幽灵块判据用 `users`(它才是"谁抬着"的
   主判据);`pending` 只在块死亡那一刻的推送里可见(足够回答"还有没有消费者")。
1b. **峰值快照的"阶梯"是诊断,不是回归门**:只有设了门才有文件;门关时
   该路径零开销(实测文件数 0)。
2. **pinned 租约只带名不建账**:`handle_alloc_pinned` 的名字只进错误上下文,
   池命中分支无账目(host 侧泄漏的归因价值低;要做需给 Staging 带标签)。
3. **未做名字的 `top-N` 之外审计**:OOM 行默认 top12;要全量需调 `top`。
4. **存量红**(非本轮引入,阻断 `--all-targets` 全绿):`apps/server/tests/ops_whitelist.rs`
   调 `OwlConfig::load(Some(&str), None)`,而该函数已改收 `Option<&Path>`
   (main 会话的 config 重构只改了签名,测试没跟上);kernels 若干集成测试
   缺 `cubins` feature 门(`family::cubins` 不存在)—— 两者都在基线 `c54ff16`
   上同样失败,与本轮无关。

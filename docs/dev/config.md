# 配置链与 `extends`(docs/dev)

> 动手改档/加实验档前先看这一页。**配置的定义面唯一在
> `crates/shared/src/config.rs`**(键的账本 + 类型 + 缺省);本页只讲"怎么用"。
> 纪律:**引擎行为不许用 env 控**(env 只剩 `OWL_CONFIG` 路径兜底 + 测试域自读);
> 一切可配项走配置文件显式声明。

## 1. 链(四段,后面的覆盖前面的)

```text
Default(代码缺省) → Flavor(内置预设 --flavor) → 用户文件(位置参数 CONFIG) → CLI 旗标
```

启动形态(位置参数 = 配置文件;`--flavor` 可省):

```bash
CUDA_VISIBLE_DEVICES=<GPU UUID> packages/owl/target/release/server \
    --flavor spec-dflash packages/owl/testdata/args/perf-ctxladder-27b.toml --log-level info
```

- **Flavor = 预先写好的一份配置**(仓库顶层 `flavors/*.toml`:行为面最优组合;
  2026-10-18 自 `assets/flavors/` 提到顶层);
  **站点面**(`model.dir`/`dflash2_dir`、`bind`、几何)放用户文件。
- `--list-flavors` 列全部预设;未知键 / 非法枚举 / 退役键 = **拒启**(带键名)。
- 语义门在装载末尾统一跑(如 `gdn_slots ≥ max_concurrency`)—— 非法组合也是拒启。

## 2. `extends`:文件继承(2026-10-18)

任一被加载的文件可在**顶层**写:

```toml
extends = "base.toml"     # 相对路径 = 相对本文件目录;也可绝对路径
```

语义:**被继承者先合并,本文件覆盖之**,链可嵌套(孙 → 子 → 父);
`--flavor` 仍在整条链最底层。母档自己也可以再 `extends`。

本地档组织(2026-10-18 起):

| 档 | 内容 |
|---|---|
| `testdata/args/base.toml` | **站点档**(本地,未入库):model 路径 + bind |
| `testdata/args/perf-ctxladder-27b.toml` | `extends = "base.toml"` + 32k 几何(40960/41216) |
| `testdata/args/prefill-prof-bare-27b.toml` | 同上 + `[spec] depth = 0`(裸臂) |

实验档 = 复制 ladder 改一行,不再整档复述。

**错误面(全部响亮拒,不静默)**:

- 成环:`extends 成环:a.toml → b.toml → a.toml`;
- 超深(>8 层)、父档缺失、值为空串/非字符串(报出得到什么类型);
- 字符串直载 `OwlConfig::from_toml_str` 无文件上下文 ⇒ 含 `extends` 即拒
  (测试请用文件加载或 `testkit`)。

## 3. 加实验档的最短姿势

```toml
# 与 base 的差异项;行为面缺省全取 flavor spec-dflash
extends = "base.toml"

[runtime]
max_seq = 40960
```

诊断面(探针/发射账)优先 CLI(`--perf-level gpu`);要写进档的只有
`[cuda] trace_ring = 65536` 这类没有 CLI 的键,做完即删。

## 4. 测试配置链(`#[owl_test]`;见 `docs/arch/owl-test-macro.md` §2.2)

同一套 loader,测试域的层链是:

```
flavor(可选) → OWL_TEST_CONF(整轮环境档) → testdata/cases/<crate>/<模块>/<用例>.toml
```

- 任何配置档都可写顶层 `flavor = "<预设名>"` 自带基准(等效 `extends` 到
  `flavors/<名>.toml`);
- `OWL_TEST_CONF` 是测试域**唯一** env(整轮口径);用例 overlay 走约定路径
  (**零 env**,删文件即还原);
- 速查:`testdata/README.md`。

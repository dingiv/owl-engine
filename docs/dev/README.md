# owl 开发者文档(docs/dev)

**这个目录是给"动手改 owl"的人看的**(新人上手、日常排障、加夹具、做 profiler 与性能验收);
架构级设计仍在 `docs/arch/`(charter/各类设计与施工文档),工作清单在 `roadmap.local/`,
包内工作约定在包根 `AGENTS.md` —— 本目录**不复制**它们,只做"怎么用"的导引。

| 文档 | 讲什么 | 什么时候看 |
|---|---|---|
| [`config.md`](config.md) | 配置链(Default→Flavor→文件→CLI)、`extends` 继承、站点档/实验档组织、拒启错误面 | 起服务、加实验档、改档位 |
| [`testing.md`](testing.md) | **测试工作流 v2**:用例 overlay 自动解析、环境档、资源清单、`#[owl_test]` 选项、决策表与反模式 | 写/改测试、加检查点、调测试口径 |
| [`debug-tools.md`](debug-tools.md) | 诊断面速查:`/debug/*` 端点、日志分级、metrics 各族、探针、排障顺序、杀进程纪律 | 服务行为可疑、要量分相、要抓现场 |
| [`fixtures.md`](fixtures.md) | 夹具与门控:testkit / 金标资产 / 探针测试 / e2e 两入口 / lint 门 / 怎么加新场景 | 写测试、改算子、跑回归 |
| [`profiling.md`](profiling.md) | nsys 用法 + 统一测速口径(`llm_speedtest`)+ A/B 方法论与常见误读 | 调性能、判收益、写验收表 |

## 三条先记

1. **量真值优先用引擎内 metrics**(`/debug/metrics`),不要用客户端逐帧计时 —— 客户端消费
   SSE 的开销会混进 decode 单步耗时(历史上把 28.5ms/步量成 51.3ms)。
2. **GPU 上的一切测试串行**:并行 = 多 CUDA 上下文同卡并发图捕获 ⇒ 假挂
   (`cargo test ... -- --test-threads=1`)。
3. **卡是共享资源,本工作区多会话**:起机前查卡占位、`/proc/vmstat` 看 `oom_kill`、
   被外部 kill 是常态(判定顺序见 `debug-tools.md` §5)。

## 新工作树开箱(三条隐性前置 + 冷建成本)

`git worktree add` 出来的新树**开箱不能构建/跑测试** —— 三件前置资产在 git 里没有
(2026-10-17 审查实录:一位 agent 在这上面花了十几分钟;2026-10-19 gguf 轮补第 ③ 条)。

```bash
# ① triton venv(不建 = `kernels/build.rs` 直接 panic:"gdn_chunked 六核 AOT
#    需要 Python+triton 环境")。两条路,选一:
uv sync --project /home/div/Documents/codes/packages/owl-worktrees/<新树>/crates/kernels
# 或(更快)把构建指到主树 venv —— build.rs 自带旋钮,不必复制 venv:
export GDN_TRITON_PY=/home/div/Documents/codes/packages/owl/crates/kernels/.venv/bin/python

# ② 0.8B 测试资产(不建 = `engine --lib` 3 项红:"models/assets/Qwen3.5-0.8B/
#    config.json: No such file"。2.5G,勿复制,软链即可):
mkdir -p /home/div/Documents/codes/packages/owl-worktrees/<新树>/crates/models/assets
for d in /home/div/Documents/codes/packages/owl/crates/models/assets/*; do
  ln -sfn "$d" "$(dirname "$d")/../.."/crates/models/assets/$(basename "$d") 2>/dev/null || true
done
```

冷建一次 **~9 分钟**(44 源 cubin 全编;之后增量 10-20s)。工作树位置与清理纪律见包根 `AGENTS.md`。

```bash
# ③ testdata/ 的**本地面**(资源清单 / args / 用例 overlay / 金标)。
#    ⚠️ 不要 `cp -r <主树>/testdata <新树>/testdata` —— 新树**已经**有 testdata/
#    (被 git 追踪的那几个文件)⇒ cp 会嵌成 `testdata/testdata/`,顶层
#    `resources.toml` 依然缺 ⇒ 需要资源清单的用例**静默 skip**(看起来全绿,
#    实则少跑)。用 `-a ./.` 形式覆盖式拷:
SRC=/home/div/Documents/codes/packages/owl
DST=/home/div/Documents/codes/packages/owl-worktrees/<新树>
cp -a "$SRC/testdata/." "$DST/testdata/"
# 验证(两行都必须有内容):
#   ls "$DST/testdata/resources.toml"          # 资源清单(本机路径)
#   ls "$DST/testdata/args" | wc -l            # 预期 ~17(只留 git 追踪的 = 3)
```

**静态资产面自查**:跑完一轮 GPU 套件后 `grep -i "skip:"` 一次 —— 若出现
`skip: 资源清单缺 …` 或 `skip: 资源 "<key>" 的路径不在 …`,就是前置没铺全
(其余 `skip: OWL_*=1 未开` 是有意门控,正常)。

## 维护约定

- 改了**接口/旋钮/端点/门控** ⇒ 顺手改本目录对应段;**文档里出现的命令必须可直接复制执行**。
- 命令一律给**绝对仓库路径前缀**(`packages/owl/...`),避免"在哪个目录跑"歧义。

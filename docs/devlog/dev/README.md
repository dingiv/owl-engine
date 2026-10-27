# owl 开发者文档(docs/devlog/dev)

**这个目录是给"动手改 owl"的人看的**(新人上手、日常排障、加夹具、做 profiler 与性能验收);
架构级设计仍在 `docs/devlog/`(charter/各类设计与施工文档),工作清单在本地台账区,
包内工作约定在包根 `AGENTS.md` —— 本目录**不复制**它们,只做"怎么用"的导引。

| 文档 | 讲什么 | 什么时候看 |
|---|---|---|
| `config.md` | 配置链(Default→Flavor/Profile→文件→站点→CLI→意图派生)、`extends` 继承、`profiles/` 意图档、`--check/--print-config/--explain/--set`、拒启错误面 | 起服务、加实验档、改档位、预检 |
| `testing.md` | **测试工作流 v2**:用例 overlay 自动解析、环境档、资源清单、`#[owl_test]` 选项、决策表与反模式 | 写/改测试、加检查点、调测试口径 |
| `debug-tools.md` | 诊断面速查:`/debug/*` 端点、日志分级、metrics 各族、探针、排障顺序、杀进程纪律 | 服务行为可疑、要量分相、要抓现场 |
| `fixtures.md` | 夹具与门控:testkit / 金标资产 / 探针测试 / e2e 两入口 / lint 门 / 怎么加新场景 | 写测试、改算子、跑回归 |
| `profiling.md` | nsys 用法 + 统一测速口径(`llm_speedtest`)+ A/B 方法论与常见误读 | 调性能、判收益、写验收表 |

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
uv sync --project /path/to/worktrees/<新树>/crates/kernels
# 或(更快)把构建指到主树 venv —— build.rs 自带旋钮,不必复制 venv:
export GDN_TRITON_PY=/path/to/crates/kernels/.venv/bin/python

# ② 0.8B 测试资产(不建 = `engine --lib` 3 项红:"models/assets/Qwen3.5-0.8B/
#    config.json: No such file"。2.5G,勿复制,软链即可):
mkdir -p /path/to/worktrees/<新树>/crates/models/assets
for d in /path/to/crates/models/assets/*; do
  ln -sfn "$d" "$(dirname "$d")/../.."/crates/models/assets/$(basename "$d") 2>/dev/null || true
done
```

冷建一次 **~9 分钟**(44 源 cubin 全编;之后增量 10-20s)。`build.rs` 现在
自带 CUDA 编译线程池：`OWL_BUILD_JOBS`（缺省 `min(nproc,8)`）可并行
native/Marlin/FlashInfer/XQA 的独立 TU；48 核全量强制重编实测 ~12m → ~4m28s。
工作树位置与清理纪律见包根 `AGENTS.md`。

```bash
# ③ testdata/ 的**本地面**(资源清单 / args / 用例 overlay / 金标)。
#    ⚠️ 不要 `cp -r <主树>/testdata <新树>/testdata` —— 新树**已经**有 testdata/
#    (被 git 追踪的那几个文件)⇒ cp 会嵌成 `testdata/testdata/`,顶层
#    `resources.toml` 依然缺 ⇒ 需要资源清单的用例**静默 skip**(看起来全绿,
#    实则少跑)。用 `-a ./.` 形式覆盖式拷:
SRC=/path/to/owl
DST=/path/to/worktrees/<新树>
cp -a "$SRC/testdata/." "$DST/testdata/"
# 验证(两行都必须有内容):
#   ls "$DST/testdata/resources.toml"          # 资源清单(本机路径)
#   ls "$DST/testdata/args" | wc -l            # 预期 ~17(只留 git 追踪的 = 3)
```

**静态资产面自查**:跑完一轮 GPU 套件后 `grep -i "skip:"` 一次 —— 若出现
`skip: 资源清单缺 …` 或 `skip: 资源 "<key>" 的路径不在 …`,就是前置没铺全
(其余 `skip: OWL_*=1 未开` 是有意门控,正常)。

## 换卡代际:代际表是**一等公民**(`crates/kernels/src/arch.rs`)

代际不是一个环境变量,是**一张表 + 两个消费者**:

```
crates/kernels/src/arch.rs        ← 唯一代际表(Arch / Family / Caps / parse)
   ├── #[path] 引入 → crates/kernels/build.rs   （构建期:四张设备码面都从它派生）
   └── mod arch     → owl_kernels::Arch          （运行期:探测/选核/能力面）
```

四张设备码面 = native cubin / vLLM-Marlin / FlashInfer+FA2 / GDN chunked(Triton AOT);
以前 FI 的 `-arch=sm_86`、Marlin 缺省 `86`、Triton `GPUTarget("cuda", 86, 32)` 是**三处各自写死**的
⇒ 换卡漏一处就是"编过了但设备码是上一代"。

| 旋钮 | 语义 |
|---|---|
| **`arch-sm75 / sm80 / sm86 / sm89 / sm90 / sm100 / sm120`** | **Cargo feature** = 构建期代际。可叠加(多代际 fatbin)。缺省由入口包给: `apps/server` 的 `default = ["arch-sm86"]` |
| —(已废) | 旧的环境变量 `OWL_TARGET_ARCH` / `OWL_CUBIN_ARCHS` / `MARLIN_CUDA_ARCH` / `FLASHINFER_ARCH` / `GDN_TRITON_ARCH` **已删除**。理由:env 不进 cargo 缓存键 ⇒ 同一份源码 + 另一个 env 在 `target/` 里不是一个东西,旧产物会被静默留用 |
| `OWL_NVCC` / `OWL_BUILD_JOBS` / `OWL_CUBIN_SKIP` | 工具链旋钮(非代际),仍走 env |

**为什么是 feature 而不是 env**:feature 由 cargo 解析 ⇒ 天然进缓存键/指纹(实测:sm_86 与
sm_120 两套产物在 `target/` 里 **各占一个 build 目录**,互不覆盖)、可复现、可叠加;
env 那个路子的后果是“换卡后编了一遍,用的还是上一代的 fatbin”(本仓真踩过)。

```bash
export CUDA_HOME=/path/to/cuda-13.2      # 必须有 nvcc(sm_120 需 ≥ CUDA 12.8)
export PATH="$CUDA_HOME/bin:$PATH"

# 单代际(本机 5070 Ti):
cargo build --release -p server --no-default-features --features arch-sm120

# 多代际 fatbin(一包两代;驱动运行期挑;单份 .a 家族取最高代):
cargo build --release -p server --no-default-features --features arch-sm86,arch-sm120

# 缺省 = sm_86(可复现基线;在 5070 上会被装配期架构门拒启并告诉你换哪个 feature)
cargo build --release -p server
```

**多选语义**:native cubin 出多代际 fatbin;单份 `.a` 家族(Marlin / FlashInfer+FA2 /
GDN AOT)取**最高**代 —— 一份 `.a` 只能是某一代的 SASS。

**零 feature**(只跑 CPU 单测 / 依赖方没转发) ⇒ 落 `Arch::DEFAULT` 并**响亮告警**,
绝不假装支持别的代际。

**运行期真值不由构建设定**,而由设备自述:`DeviceClient::device_arch()`
(`COMPUTE_CAPABILITY_*` → `Arch::from_cc`)在 boot 期问一次,写进 `EnvProvider::hw` ——
`[kernel-env] arch=` 现在是**真代际**,不再是占位的 `Sm86`;GPU 面答不出代际 = **拒启**
(纯 host face 没有硬件档,自然跳过)。装配期架构门把两真值对齐:设备 sm ∉ `CUBIN_ARCHS` ⇒ 拒启。

**四条判据**(缺一条就是"编过了但设备码是上一代"):

```bash
# ① 构建日志:代际集 + 单份 .a 的取优结果
#    owl-kernels: 构建代际集 = [sm_120]（Cargo feature arch-smXX）· native 出 fatbin 多代 · 单份 .a 取最高代 sm_120(世代 Blackwell · …)
#    owl-kernels: xqa_decode.fatbin [sm_120] / built libmarlin.a (sm_120)
# ② 产物自检(cuobjdump 随 triton venv 一起来)
CUOBJ=crates/kernels/.venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/cuobjdump
OUT=$(ls -dt target/release/build/owl-kernels-*/out | head -1)
$CUOBJ --list-elf $OUT/cubins/$(ls $OUT/cubins | head -1)     # native → sm_120
$CUOBJ --list-elf $OUT/marlin/*.o | head -2                    # marlin → sm_120
$CUOBJ --list-elf $OUT/xqa/xqa_decode.fatbin                   # xqa    → sm_120
$CUOBJ --list-elf $OUT/gdn/o.cubin                             # gdn    → sm_120a
# ③ 启动日志:`[aot] 构建期 cubin 代际 ["sm_120"]` + `[boot] 设备代际 = sm_120(Blackwell · …)`
# ④ `[kernel-env] target: arch=sm_120 …`(两真值对上)
```

**新增一个代际的操作面**(只五处,编译器会逼你改):

1. `arch.rs` 加变体(`Sm121`)或直接靠 `Other` 兜底;
2. `Arch::caps` 写它的 ISA 事实(按 CC 阈值推导,表外代际自动获得合理答案);
3. `crates/kernels/Cargo.toml` 加 `arch-sm121 = []`,`owl-cuda` / `apps/server` 各加一行转发;
4. `marlin` 的 `QuantGemm`(无通配 match)+ `dispatch()`(无通配 match)—— 不表态就编不过;
5. 跑一遍四条判据。

坑(2026-10-27 实测):

- **CUDA toolkit 版本必须自洽**。pip 拆包(`nvidia-cuda-nvcc` / `-runtime` / `-crt` / `-cccl`)
  装出"nvcc 13.2 + 头 13.0"就会在 `cuda/std/__cccl/cuda_toolkit.h` 直接 `#error`(CCCL
  自己比 `__CUDACC_VER_*` 与 `CUDART_VERSION`)。四件套**钉同一个版本**即可;pip 布局
  的 CCCL 在 `include/cccl/`,只有 `include/` 下缺 `cuda/`、`cub/`、`thrust/` 时
  才需要额外 `-I`。
- **Triton 在 Blackwell 上产出 `sm_120a`**(实测 `cuobjdump`: `wu.sm_120a.cubin`)。
  `a` = arch-specific,同代设备照跑;装配期架构门只看 `sm_120`。
- **GDN AOT 的 `shared` 自校验**按代际收录(`triton/gdn_chunked_aot.json` 的
  `shared_by_arch`):sm_120 上 `wu` 的 smem 是 28672(≠ sm_86 的 32768,
  Blackwell codegen 少一个 stage)。未收录的代际只警告不拒 —— 表不是在那个代际采集的。
- 换卡不改配置面:cubin 是**构建期**产物,卡位仍只走 `[parallel].device`。

## 维护约定

- 改了**接口/旋钮/端点/门控** ⇒ 顺手改本目录对应段;**文档里出现的命令必须可直接复制执行**。
- 命令一律给**绝对仓库路径前缀**(`...`),避免"在哪个目录跑"歧义。

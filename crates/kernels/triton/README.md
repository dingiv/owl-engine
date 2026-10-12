# Triton 算子源与 AOT 工具(owl 树内,2026-10-13)

本目录 = **所有 Triton 相关的东西**:gdn_chunked 六核的源、编译配置、AOT 编译器。

```
triton/
├── README.md                 ← 本文件
├── gdn_chunked_aot.json      ← **AOT 编译配置表**(烘焙的 autotune 决策;见下)
├── gdn_aot_compile.py        ← AOT 编译器(只编译不发射;build.rs 默认调它)
└── fla/                      ← vendor 的 FLA 六核 import 闭包(上游源)
    ├── triton_shim.py        ← owl 加的 import 垫片(见下)
    └── ops/*.py
```

**为什么源要 vendor 进树**(设计律 §15.1.1 律 1"所有族的源必须在树"):原采集配方
(`testdata/gdn_fla/harvest_fork.py`)从 vllm 检出取源 + 跑整条流水线 + 让 autotune
**现场选配置** ⇒ 产物不确定、且 owl 树内无法重建。现在:**源在树 + 配置烘焙 +
AOT 编译** ⇒ 实测**逐字节复现**原入库件(设计 §15.11)。

**`gdn_chunked_aot.json` 是什么、为什么需要**:Triton 核的编译选项空间是
**autotune 搜出来的**(不从源码唯一导出;kkt 有 3 个、h 有 2 个、o 有 4 个合法候选),
所以必须记下"选中哪组"(`num_warps`/`num_stages`/`kwargs(BK,BV)` + `shared` 期望值)。
脚本**按表直接编译**,并把产物回读的 `shared` 与表值比对 —— 不等即拒用。
⇒ 它是一份 **lockfile**,不是 per-binary 元数据;**native 核没有也不需要有**
(它们的编译输入 = `.cu` + 架构)。

## fla/ 的 owl 本地改动(两处,语义不变)

1. **import 重写**:原文件 `from vllm.triton_utils import tl, triton` /
   `from vllm.platforms import current_platform` /
   `from vllm.platforms.rocm import on_gfx1x` → `from ..triton_shim import …`。
   垫片给出等价最小实现(`import triton` / `import triton.language as tl` /
   `triton.language.extra.libdevice as tldevice`;平台判定恒 CUDA)。
   ⇒ **vendor 副本不依赖 vllm**。
2. **`ops/__init__.py` 清空**:上游把全部 ops 子模块 import 进来;本副本只含
   六核闭包,未 vendor 的模块会 ModuleNotFoundError ⇒ 改为不导入
   (需要哪个核就 `from fla.ops import <模块>`)。

**其余逐字未改** —— 升级 = 换 vllm 检出后重取闭包 + 重跑
`triton/gdn_aot_compile.py`(产物应与上一代**逐字节一致**;不一致 = 上游语义变了,
`gdn_chunked_golden` 会拦)。

## 资产现状(2026-10-13 起:资产 = 构建产物)

- **入库采集件已整体删除**(`crates/kernels/assets/` 退役,~1.8 MB:6 生产 cubin +
  `h_noh0` + pipfla_f32 退役备份 + manifest)。六核改为**构建期 AOT 自编**,与
  native cubin / marlin / flashinfer 同形态。
- 两个未注册资产(`h_noh0` 无初态变体、pipfla_f32 退役变体)随之一并删除 ——
  它们是设备码但**没有任何装载路径**;将来要用 = 配置表加一行 +
  `initial_state=None` 重编即可(一行改动)。
- **实测**:自编六核与原入库件**逐字节一致**(§15.11);`gdn_chunked_golden`
  c5(同源同核紧容差)max|Δ| = 0.000e0。

## 构建环境与开关(uv)

构建期需要 Python + triton。环境由 **`crates/kernels/pyproject.toml`(uv 项目)**
声明 —— 含**源目录声明**:

```toml
[build-system]
requires = ["uv_build>=0.12,<0.13"]
build-backend = "uv_build"

[tool.uv.build-backend]
module-root = "triton"      # ← 源根 = crates/kernels/triton/
module-name = "fla"         # ← 顶层模块 = triton/fla(vendor 的 FLA 闭包)
```

⇒ `uv sync --project crates/kernels` 会装 deps **并把本包 editable 装进 venv**
(venv 里的 `fla` 指回树内源码,不留第二份拷贝),于是脚本 `import fla.ops` 直接可用。

```bash
uv sync --project crates/kernels      # → crates/kernels/.venv
cargo build                           # build.rs 自动用这个 venv
uv run --project crates/kernels triton/gdn_aot_compile.py --out /tmp/gdn   # 手动跑
```

| 依赖 | 为什么是它 |
|---|---|
| **`triton==3.7.1`(钉死)** | 它是**编译器** —— 版本变 SASS 就变,资产不再"逐字节可复现"。这一条不能松 |
| **`torch`(CPU 轮子)** | FLA 的 wrapper 用 torch 做张量簿记(`torch.empty_like`/`len()`/`.to()`);AOT 只编译不发射 ⇒ **不需要 CUDA 版**(CPU 轮子,省 ~2.5 GB) |

> 试过"纯 torch-free"(Triton 的 `MockTensor` 描述符)路线:在 wrapper 的
> `len(chunk_indices)` 处就破 —— wrapper 要的是**真张量**,不是描述符。
> 故保留 torch,但**只用 CPU**、不碰 CUDA。

> 若哪天想改成"纯工具环境、不装本包":去掉 `[build-system]` 与
> `[tool.uv.build-backend]`,加 `[tool.uv] package = false` —— 那样 `fla` 不被装,
> 脚本会退回 `sys.path.insert(triton/)` 兜底(两条路都在脚本里,见其 import 段)。

| 开关 | 语义 |
|---|---|
| (缺省) | 自编。解释器探测序:`.venv/bin/python`(本 crate 的 uv 环境)→ 工作区 `vllm-test/.venv`;可用 `GDN_TRITON_PY` 指定 |
| `GDN_TRITON_DEVICE` | 缺省 `none` = **零 GPU**(注入 target 桩 sm86;AOT 只编译不发射,`CompiledKernel` 的设备句柄惰性)⇒ **构建不需要 GPU**。`cuda:N` 切真实设备(排障/对照) |
| `GDN_TRITON_BUILD=0` | 显式跳过自编 ⇒ 资产表为空 ⇒ 该族**装配期拒启**(无 triton 环境仍能编过 crate) |

## 加一个新的 Triton 算子族(2026-10-13 用户追问)

uv_build 的 `module-name` **支持列表**(官方文档:`module-name = ["foo", "bar"]`)。
本目录就是**多族共用的源根**,所以加族的动作是固定的五步:

| # | 动作 | 落在哪 |
|---|---|---|
| 1 | 把上游包 vendor 到 `triton/<name>/`(**保持上游包名** —— 其内部相对 import 依赖它) | `triton/<name>/` |
| 2 | 往 `crates/kernels/pyproject.toml` 的 `module-name` 列表**加一个名字** | 一行 |
| 3 | 该族的 AOT 编译器 + 配置表:`triton/<name>_aot.py` + `triton/<name>_aot.json`(配置表 = 烘焙的 autotune 决策,见下) | `triton/` |
| 4 | `build.rs` 加一段:调脚本 → `$OUT_DIR/<name>/*.cubin` → 生成资产表 + 增量闸(照 `build_gdn_assets` 抄) | `build.rs` |
| 5 | family 侧 `resolve()` 取资产(缺 = 装配期拒启)+ 数值判据(golden/探针) | `src/family/<name>.rs` |

**实测**(2026-10-13 用一个 dummy 族验过):建 `triton/_probe_family/__init__.py`
→ `module-name` 加名 → `uv sync` → `import fla, _probe_family` 两个都通、
都指向树内源码(editable)。⇒ "加族 = 放目录 + 一行"。

**不推荐 `namespace = true`**(官方注明它会**关掉安全检查**),也**不要**为了省事把
`crates/kernels` 塞进 `sys.path`(会遮蔽真正的 `triton` 包)。

> 若某族的源**不能** vendor(只有二进制/上游闭源):那就该走"入库资产 + 指纹清单"
> 的老路,而不是塞进本目录 —— 本目录的语义是"**源在树**"。

**零 GPU 是怎么成立的**:`warmup` 只走到 `compile()`,不 launch ⇒ `CompiledKernel._init_handles()`
(它才需要 `driver.utils.load_binary` + 设备)从不被调用;`run()` 起手只问
`get_current_device/get_current_stream`(缓存键)与 `get_current_target()`,
桩驱动给这三个即可,其余方法一被调用就报错(等于"设备触达哨兵")。
实测:零 GPU 产物与带 GPU 产物**逐字节一致**。

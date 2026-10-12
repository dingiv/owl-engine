#!/usr/bin/env python3
"""gdn_chunked 六核 **AOT 编译**(2026-10-13 C4b-β;opt-in,`GDN_TRITON_BUILD=1`)。

# 为什么是 AOT,不是"重跑采集脚本"

原 `testdata/gdn_fla/harvest_fork.py` 跑整条 FLA 流水线 + 让 **autotuner 现场选
配置**再读 Triton 磁盘缓存取 cubin。两个问题:
1. 现场 autotune = **产物不确定**(换个负载/机器就可能选到别的配置);
2. 它还要 GPU 跑完整流水线。

本脚本改走 **AOT**:内核的 `kernel[grid](...)` 被换成 `jit_fn.warmup(...)`
(**只编译不发射**),配置**烘焙**自配置表 `triton/gdn_chunked_aot.json`。

# 配置从哪来(自校验)

Triton 核的编译选项空间是 **autotune 选出来的**,不是从源码唯一确定 ⇒ 必须把
选中那组记下来:配置表 = `triton/gdn_chunked_aot.json`(`num_warps`/`num_stages`/
`kwargs`(BK/BV)/`shared`)。本脚本按表**直接编译**(不再搜候选),并把编译产物
回读的 `shared` 与表中 `shared` 比对 —— 不等即报错退出(源/配置漂移的机器判据)。

> native 核没有这类表:它们的编译输入就是 `.cu` + 架构(见 build.rs)。

# 用法

    uv run --project crates/kernels triton/gdn_aot_compile.py --out <目录> [--device none|cuda:N]
    #   (或直接 crates/kernels/.venv/bin/python triton/gdn_aot_compile.py ...)
    # triton 环境 = 只需 **triton**(钉版本;见 crates/kernels/pyproject.toml,`uv sync` 生成)
    # --device none(缺省)= **零 GPU**:注入 target 桩(sm86),不碰设备 —— AOT 只编译
    #   不发射,`CompiledKernel` 的设备句柄是惰性的(不 launch 就不 _init_handles)
    # --device cuda:N    = 用真实设备取 target(排障/对照)

产物:`<目录>/{cumsum,kkt,merge,wu,h,o}.cubin` + 同名 `.json`(本次编译选项)。

# 判据

产物**不保证与入库 cubin 逐字节一致**(当年采集的源修订/环境未钉死)——
忠实性由 `gdn_chunked_golden`(c1-c5;c5 = 同源同核紧容差)判定,不看字节。
"""

import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))  # crates/kernels/triton
KERNELS_DIR = os.path.dirname(HERE)  # crates/kernels(仅用于文档/相对定位)
CFG_PATH = os.path.join(HERE, "gdn_chunked_aot.json")
# ⚠️ sys.path 放的是 **triton/ 这一层**(不是 crates/kernels):这样 `import fla`
# 命中 `triton/fla`,而 `import triton` 仍命中**真正的 triton 库**。若把
# crates/kernels 放进 sys.path,我们的 `triton/` 目录会遮蔽 triton 包
# (PEP 420 命名空间包,不需要 __init__.py 也能遮蔽)⇒ 后果灾难性。
PKG_ROOT = HERE


def _jit_of(obj):
    """剥掉 @triton.heuristics / @triton.autotune,取 JITFunction。"""
    while type(obj).__name__ in ("Heuristics", "Autotuner"):
        obj = obj.fn
    return obj


def _autotuner_of(obj):
    while type(obj).__name__ != "Autotuner" and hasattr(obj, "fn"):
        obj = obj.fn
    return obj


def _install(mod, sym, original, proxy) -> None:
    """把代理装进调用链(**每次都从 `original` 重建**,不用模块当前值)。

    ⚠️ 两条实测坑:
    1. 必须**保留外层 Heuristics**(它注入 IS_VARLEN 等):直接换掉模块符号会让
       `Heuristics.run` 被绕过 ⇒ `dynamic_func() missing IS_VARLEN`;
    2. 候选轮询时**不能** `getattr(mod, sym)` 取"当前值" —— 上一轮已把它换成
       代理,再包一层就得到"代理的代理" ⇒ `'_WarmupProxy' object has no
       attribute 'warmup'`。
    """
    if type(original).__name__ == "Heuristics":
        original.fn = proxy
    else:
        setattr(mod, sym, proxy)


class _WarmupProxy:
    """顶替 `kernel[grid](...)` → `jit_fn.warmup(...)`(只编译)。

    Triton 调用链:`Heuristics.__getitem__` → `Heuristics.run`(注入 heuristics
    值,如 IS_VARLEN)→ `Autotuner.run` → `JITFunction.run`。本代理**顶替
    Autotuner 那一层**,保留外层 Heuristics ⇒ wrapper 的 constexpr/heuristics
    计算逐字不变,只是最后不发射,且额外并入本候选的 config kwargs。

    grid 可以是元组,也可以是 wrapper 的 `def grid(meta)` 回调(用本次 kwargs 求值)。
    """

    def __init__(self, obj, num_warps, num_stages, extra_kwargs):
        self._jit = _jit_of(obj)
        self._w, self._s, self._extra = num_warps, num_stages, dict(extra_kwargs)
        self.compiled = None

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.run(*args, grid=grid, **kwargs)

    def run(self, *args, **kwargs):
        kwargs.pop("warmup", None)
        grid = kwargs.pop("grid", None)
        merged = {**self._extra, **kwargs}
        # ⚠️ grid 回调要吃**并入候选 kwargs 之后**的 meta:h/o 的 `def grid(meta)`
        # 依赖 config 里的 BV/BK(实测 KeyError: 'BV')
        g = grid(merged) if callable(grid) else grid
        self.compiled = self._jit.warmup(
            *args, grid=g, num_warps=self._w, num_stages=self._s, **merged
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="cubin 输出目录")
    ap.add_argument("--device", default="none",
                    help="`none`(缺省)= 零 GPU(target 桩);`cuda:N` = 真实设备取 target")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    with open(CFG_PATH) as f:
        CFG = json.load(f)
    KCFG = CFG["kernels"]
    sh = CFG["shape"]
    NK, NV, KD, VD, T, BT = sh["NK"], sh["NV"], sh["KD"], sh["VD"], sh["T"], sh["BT"]

    # 张量簿记用 **CPU torch**(`uv sync` 装的是 CPU 轮子,见 pyproject.toml):
    # FLA 的 wrapper 会做 `len()`/`.to()`/`torch.empty_like()` 等真张量操作,
    # 不是 MockTensor 能糊过去的 ⇒ 保留 torch,但**只用 CPU**、不碰 CUDA。
    # (试过 MockTensor 纯 torch-free 路线:在 `len(chunk_indices)` 处就破 —— 见
    #  triton/README.md 的"依赖为什么是这两个"。)
    import torch

    DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16,
              "fp32": torch.float32, "i64": torch.int64}

    dev = args.device
    if dev != "none":
        # 排障路径:真实设备取 target。⚠️ 默认设备必须 = 张量设备(否则 autotuner/
        # do_bench 按默认设备分配缓冲,跨设备报 "Pointer argument cannot be
        # accessed from Triton";本脚本走 warmup 不经 bench,保持纪律免将来踩)
        torch.cuda.set_device(dev)
    else:
        # **零 GPU**:注入 target 桩 —— AOT 只编译不发射,`CompiledKernel` 的设备
        # 句柄是惰性的(`.run` 才 `_init_handles`)⇒ 全程不碰设备。
        from triton.backends.compiler import GPUTarget
        from triton.runtime.driver import driver as _drv

        class _StubDriver:
            """target 桩:`run()` 起手要 `get_current_device/stream`(仅用于缓存键),
            其余驱动方法一旦被调用就报错(说明真的触到了设备)。"""

            def get_current_target(self):
                return GPUTarget("cuda", 86, 32)

            @staticmethod
            def get_current_device():
                return 0

            @staticmethod
            def get_current_stream(_device):
                return None

            def __getattr__(self, k):
                raise RuntimeError(
                    f"AOT 无设备模式(--device none):驱动方法 `{k}` 被调用 —— "
                    f"说明该核触到了真实设备;改用 `--device cuda:N`"
                )

        _drv.set_active(_StubDriver())

    # 源来自 `triton/fla`(顶层模块 `fla`)。uv 项目(`crates/kernels/pyproject.toml`)
    # 声明了 `module-root = "triton"` / `module-name = "fla"` ⇒ `uv sync` 把它
    # **editable 装进 venv**(指回树内源码)⇒ 直接 import 即可。
    # 兜底:若 python 来自没装本包的 venv(`GDN_TRITON_PY` 指向别处),退回手插路径。
    try:
        from fla.ops import (  # noqa: E402
            chunk_delta_h,
            chunk_o,
            chunk_scaled_dot_kkt,
            cumsum,
            solve_tril,
            wy_fast,
        )
    except ImportError:
        sys.path.insert(0, PKG_ROOT)
        from fla.ops import (  # noqa: E402
            chunk_delta_h,
            chunk_o,
            chunk_scaled_dot_kkt,
            cumsum,
            solve_tril,
            wy_fast,
        )

    # dtype 配方 = 引擎实参(family/gdn_chunked.rs + server 的 cast-in 链):
    #   handler 先 owl_cast_f16_bf16 把 f16 输入转 **bf16**,再喂 Triton 核
    #   ⇒ 激活域 = bf16;g / g_cum / A / state(h0,ht) = f32
    #   (资产侧 `origin` 亦记 "bf16+g_f32";harvest 脚本里的 DT=f16 是历史残留)
    def z(*shape, dtype="bf16"):
        return torch.zeros(*shape, dtype=DTYPES[dtype])  # CPU;值无关(warmup 只取 dtype/shape)

    # stub 实参(**按名**给,不用位置元组 —— 位置元组极易与闭包的取值下标错位,
    # 实测踩过:kkt 的 beta/g 互换 ⇒ 内核拿到 f16 的 g ⇒ exp 的 fp32 检查炸)。
    # dtype 配方 = family/gdn_chunked.rs ABI 注记:
    #   q/k/v/beta = f16;g / g_cum / A / h 缓冲 / h0 = **f32**(核内 exp/log 只吃 fp32)
    S = {
        "q": z(1, T, NK, KD),
        "k": z(1, T, NK, KD),
        "v": z(1, T, NV, VD),
        "beta": z(1, T, NV),
        "g32": z(1, T, NV, dtype="fp32"),
        "A32": z(1, T, NV, BT, dtype="fp32"),
        "Ai": z(1, T, NV, BT),
        "w": z(1, T, NV, KD),
        "u": z(1, T, NV, VD),
        "h32": z(T // BT, NV, KD, VD),          # h_buf:bf16(引擎 h_buf 2B/元素)
        "h0": z(1, NV, KD, VD, dtype="fp32"),
        "cu": z(2, dtype="i64"),
        "ci": z(T // BT, 2, dtype="i64"),
    }

    def call_cumsum(proxy, original):
        _install(cumsum, "chunk_local_cumsum_scalar_kernel", original, proxy)
        # 引擎:raw g(f16 镜像)→ cast f32 → cumsum → g_cum(f32)
        return cumsum.chunk_local_cumsum_scalar(
            S["g32"], chunk_size=BT, cu_seqlens=S["cu"], chunk_indices=S["ci"])

    def call_kkt(proxy, original):
        _install(chunk_scaled_dot_kkt, "chunk_scaled_dot_kkt_fwd_kernel", original, proxy)
        return chunk_scaled_dot_kkt.chunk_scaled_dot_kkt_fwd(
            k=S["k"], beta=S["beta"], g=S["g32"], cu_seqlens=S["cu"],
            chunk_indices=S["ci"], output_dtype=torch.float32)

    def call_merge(proxy, original):
        _install(solve_tril, "merge_16x16_to_64x64_inverse_kernel", original, proxy)
        return solve_tril.solve_tril(A=S["A32"], cu_seqlens=S["cu"],
                                     chunk_indices=S["ci"], output_dtype=torch.bfloat16)

    def call_wu(proxy, original):
        _install(wy_fast, "recompute_w_u_fwd_kernel", original, proxy)
        return wy_fast.recompute_w_u_fwd(
            k=S["k"], v=S["v"], beta=S["beta"], A=S["Ai"], g_cumsum=S["g32"],
            cu_seqlens=S["cu"], chunk_indices=S["ci"])

    def call_h(proxy, original):
        _install(chunk_delta_h, "chunk_gated_delta_rule_fwd_kernel_h_blockdim64", original, proxy)
        return chunk_delta_h.chunk_gated_delta_rule_fwd_h(
            k=S["k"], w=S["w"], u=S["u"], g=S["g32"], initial_state=S["h0"],
            output_final_state=True, cu_seqlens=S["cu"], chunk_indices=S["ci"])

    def call_o(proxy, original):
        _install(chunk_o, "chunk_fwd_kernel_o", original, proxy)
        return chunk_o.chunk_fwd_o(q=S["q"], k=S["k"], v=S["v"], h=S["h32"], g=S["g32"],
                                   scale=KD**-0.5, cu_seqlens=S["cu"], chunk_indices=S["ci"])

    specs = [
        ("cumsum", cumsum, "chunk_local_cumsum_scalar_kernel", call_cumsum),
        ("kkt", chunk_scaled_dot_kkt, "chunk_scaled_dot_kkt_fwd_kernel", call_kkt),
        ("merge", solve_tril, "merge_16x16_to_64x64_inverse_kernel", call_merge),
        ("wu", wy_fast, "recompute_w_u_fwd_kernel", call_wu),
        ("h", chunk_delta_h, "chunk_gated_delta_rule_fwd_kernel_h_blockdim64", call_h),
        ("o", chunk_o, "chunk_fwd_kernel_o", call_o),
    ]

    print(f"[aot] 形状 T={T} NK={NK} NV={NV} KD={KD} VD={VD} BT={BT} dev={dev}")
    for asset, mod, sym, call in specs:
        cfg = KCFG[asset]
        w, st, kw = cfg["num_warps"], cfg["num_stages"], cfg.get("kwargs", {})
        original = getattr(mod, sym)
        pristine = _jit_of(original)
        proxy = _WarmupProxy(pristine, w, st, kw)
        call(proxy, original)
        shared = getattr(proxy.compiled.metadata, "shared", None)
        if shared != cfg["shared"]:
            print(f"[aot] ❌ {asset}: 编译产物 shared={shared} ≠ 表值 {cfg['shared']}"
                  f"(配置 {kw} w{w} s{st})—— 源/配置漂移,拒用", file=sys.stderr)
            return 1
        print(f"[aot] ✓ {asset}: w{w} s{st} {kw} shared={shared}(= 表值)")
        cubin = proxy.compiled.asm["cubin"]
        dst = os.path.join(args.out, f"{asset}.cubin")
        with open(dst, "wb") as f:
            f.write(cubin)
        print(f"[aot] → {dst} {len(cubin)}B")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""CUTLASS / cute 头树的**瘦身探针**：文件粒度试删 + 符号粒度“到底用了什么”。

## 为什么要它

头文件树里“被 include”≠“被需要”：一个 300 KB 的头可能只为 3 个空标签而被
拉进来（第 3 轮删掉 `cutlass/arch/mma.h` 那条链就是这么发现的）。人肉翻文件
既慢又容易漏，而**唯一的权威判据是编译器**：

> 把这个文件清空 → 它还编得过吗？编出来的 PTX 变了吗？

本脚本把这件事自动化，两种模式：

    # ① 文件粒度：逐个候选头试删（并行、每个 worker 一份树的副本）
    ./cu_trim.py files -n 8
    # 输出 report.csv：path,size,verdict,errs —— verdict ∈ 可删 / 需要(编不过) / 变了(PTX 变)

    # ② 符号粒度：只看某一个头“我们用了它里面的哪几个符号”
    ./cu_trim.py symbols most-of-it
    # 输出：清空该头后的全部编译错误里出现的**符号名**（去重、按出现次数排序）

② 的意义：错误列表 = “我们真正需要它提供的东西”的最小证据。把这些符号抄进
一个 owl 侧小头（本轮的第 2/3/4 刀都是这么做的），原文件就能整块删。

## 判据与纪律

- **PTX 判据**：`nvcc -ptx` 去注释行与 `_INTERNAL_<hash>_`（随 include 集合变化的
  TU id）后逐行相同 ⇒ 代码生成未变。这是比跑分更硬的判据（参考 `TRIMMED.md` §5.1）。
- **每个 worker 一份树副本**：试删是全局变异，并行跑同一棵树会互相污染；副本只有
  5 MB，代价忽略。注意 `__FILE__` 会把副本路径编进 PTX ⇒ 归一化时把副本根抹掉
  （baseline 也用副本编，两边同样处理）。
- **文件粒度可删 ≠ 结构上没用**：它只说明“当前这个 TU 不需要”。所以脚本只报
  判决，**永远自己决定删不删**（有些是给别的 TU/别的代际留的）。

## 用法前提

在本仓 `crates/kernels/` 下跑（`build.rs::build_nvfp4` 的那套 nvcc 旗标硬编码在
`CFG` 里，改旗标就改它）。
"""
import argparse
import concurrent.futures as cf
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

# ---- 与 build.rs::build_nvfp4 对齐的编译面 ----------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # crates/kernels
TREE = os.path.join(ROOT, "cu/cutlass/include")
TU = "cu/nvfp4/nvfp4_gemm.cu"
BASE = [
    "-O3",
    "-std=c++17",
    "--expt-relaxed-constexpr",
    "-gencode",
    "arch=compute_120a,code=sm_120a",
]
EXTRA_INC = ["-Icu/nvfp4", "-Icu"]

NVCC = os.environ.get("NVCC", "nvcc")


def norm_ptx(text: str, copy_root: str) -> list[str]:
    """去注释行 / TU-id / 副本路径后的可比较行序列。"""
    out = []
    for ln in text.splitlines():
        if ln.lstrip().startswith("//"):
            continue
        ln = re.sub(r"_INTERNAL_[0-9a-f]+_", "", ln)
        ln = ln.replace(copy_root + "/", "").replace(copy_root, "")
        out.append(ln)
    return out


def compile_ptx(tree_root: str, out_path: str) -> tuple[bool, str]:
    """用给定树根编一次 TU，出 PTX。返回 (成功?, stderr)。"""
    cmd = [NVCC, *BASE, f"-I{tree_root}/include", *EXTRA_INC, "-ptx", TU, "-o", out_path]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return p.returncode == 0, p.stderr


def make_copy(dst: str) -> str:
    """复制一份树（只说 include 目录），返回 dst。"""
    shutil.copytree(TREE, os.path.join(dst, "include"))
    return dst


def trial_one(path: str, base_lines: list[str]) -> tuple[str, str, int, str]:
    """在**私有副本**里清空 path 后编译，返回 (path, verdict, size, errs)。"""
    rel = os.path.relpath(path, TREE)
    size = os.path.getsize(path)
    with tempfile.TemporaryDirectory(prefix="cutrim_") as tmp:
        copy = make_copy(os.path.join(tmp, "t"))
        target = os.path.join(copy, "include", rel)
        # 清空但保留 #pragma once（避免同一头被二次 include 时炸出无关错误）
        with open(target, "w") as f:
            f.write("#pragma once\n")
        ptx = os.path.join(tmp, "x.ptx")
        ok, err = compile_ptx(copy, ptx)
        if not ok:
            return (rel, "需要", size, first_names(err))
        got = norm_ptx(open(ptx).read(), copy)
        if got != base_lines:
            return (rel, "变了", size, "PTX 与基线不同")
        return (rel, "可删", size, "")


UNDEF = re.compile(r"identifier\s+\"([^\"]+)\"\s+is undefined|"
                   r"namespace\s+\"([^\"]+)\"\s+has no member\s+\"([^\"]+)\"|"
                   r"\"([^\"]+)\"\s+is not a|"
                   r"name followed by.*must be|"
                   r"undefined reference to\s+`([^']+)'")


def err_name(m: "re.Match") -> str:
    """从一条 UNDEF 匹配里取出人类可读的符号名（命名空间形式拼成 `a::b`）。"""
    if m.group(1) or m.group(4):
        return m.group(1) or m.group(4)
    if m.group(2) and m.group(3):
        return f"{m.group(2)}::{m.group(3)}"
    return m.group(5) or ""


def first_names(err: str, limit: int = 6) -> str:
    names = []
    for m in UNDEF.finditer(err):
        nm = err_name(m)
        if nm and nm not in names:
            names.append(nm)
        if len(names) >= limit:
            break
    return " | ".join(names)


def mode_files(workers: int, only: str | None, top: int | None):
    with tempfile.TemporaryDirectory(prefix="cutrim_base_") as tmp:
        base_copy = make_copy(os.path.join(tmp, "b"))
        ptx = os.path.join(tmp, "b.ptx")
        ok, err = compile_ptx(base_copy, ptx)
        assert ok, f"基线编译失败（旗标/树不对）:\n{err[-2000:]}"
        base_lines = norm_ptx(open(ptx).read(), base_copy)
    print(f"基线 PTX {len(base_lines)} 行", flush=True)

    files = []
    for d, _, fs in os.walk(TREE):
        for f in fs:
            p = os.path.join(d, f)
            if only and not re.search(only, p):
                continue
            files.append(p)
    files.sort(key=lambda p: -os.path.getsize(p))
    if top:
        files = files[:top]
    print(f"候选 {len(files)} 个（最大 {os.path.getsize(files[0]) // 1024} KB）", flush=True)

    t0 = time.time()
    rows = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(trial_one, p, base_lines): p for p in files}
        for i, fu in enumerate(cf.as_completed(futs), 1):
            rows.append(fu.result())
            if i % 10 == 0:
                print(f"  …{i}/{len(files)}  ({time.time() - t0:.0f}s)", flush=True)
    rows.sort(key=lambda r: (r[1] != "可删", -r[2]))
    out = os.path.join(ROOT, "../../.local-state/cu_trim_report.csv")
    with open(out, "w") as f:
        f.write("path,size,verdict,detail\n")
        for rel, verdict, size, errs in rows:
            f.write(f"{rel},{size},{verdict},{errs}\n")
    ndel = sum(1 for r in rows if r[1] == "可删")
    nbytes = sum(r[2] for r in rows if r[1] == "可删")
    print(f"\n可整文件删: {ndel} 个 / {nbytes / 1024:.0f} KB → {out}")
    for rel, verdict, size, _ in rows[:40]:
        print(f"  {verdict:4s} {size / 1024:7.1f} KB  {rel}")


def mode_symbols(match: str):
    """清空匹配到的那个头，把编译错误里出现的符号名去重列出。"""
    hits = [p for p in (os.path.join(d, f) for d, _, fs in os.walk(TREE) for f in fs)
            if re.search(match, p)]
    assert len(hits) == 1, f"匹配到 {len(hits)} 个文件，请给更精确的路径片段"
    path = hits[0]
    rel = os.path.relpath(path, TREE)
    print(f"目标: {rel} ({os.path.getsize(path) / 1024:.1f} KB)")
    with tempfile.TemporaryDirectory(prefix="cutrim_sym_") as tmp:
        copy = make_copy(os.path.join(tmp, "t"))
        open(os.path.join(copy, "include", rel), "w").write("#pragma once\n")
        ptx = os.path.join(tmp, "x.ptx")
        ok, err = compile_ptx(copy, ptx)
        if ok:
            print("清空后**照样编过** ⇒ 这整个文件当前无人需要（用 files 模式确认 PTX 未变）")
            return
        names = {}
        for m in UNDEF.finditer(err):
            nm = err_name(m)
            if nm:
                names[nm] = names.get(nm, 0) + 1
        for nm, c in sorted(names.items(), key=lambda kv: -kv[1]):
            print(f"  ×{c:3d}  {nm}")
        print(f"\n共 {len(names)} 个符号名（这些就是“我们要抄过来的东西”）")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    f = sub.add_parser("files")
    f.add_argument("-n", "--workers", type=int, default=8)
    f.add_argument("--only", default=None, help="只试匹配这些路径片段（正则）")
    f.add_argument("--top", type=int, default=None, help="只试最大的 N 个")
    s = sub.add_parser("symbols")
    s.add_argument("match", help="唯一匹配的文件路径片段（正则）")
    a = ap.parse_args()
    if a.mode == "files":
        mode_files(a.workers, a.only, a.top)
    else:
        mode_symbols(a.match)


if __name__ == "__main__":
    sys.exit(main())

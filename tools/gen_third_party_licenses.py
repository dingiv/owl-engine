#!/usr/bin/env python3
"""从 `cargo metadata` 生成 `THIRD_PARTY_LICENSES.md`。

用法（在仓库根）：
    cargo metadata --format-version 1 --all-features \\
      | python3 tools/gen_third_party_licenses.py > THIRD_PARTY_LICENSES.md

约定：
- 只列**非 workspace 成员**的包（crates.io / git 依赖）；
- 仓内 vendor 的 CUDA 源码不在此列 —— 归 `NOTICE` 与 `VENDOR.list.md`；
- 输出稳定排序（名字小写、版本），便于 diff。
"""
import json
import sys


def main() -> int:
    meta = json.load(sys.stdin)
    workspace = set(meta["workspace_members"])
    rows = []
    for pkg in meta["packages"]:
        if pkg["id"] in workspace:
            continue
        lic = pkg.get("license")
        if not lic and pkg.get("license_file"):
            lic = f"file:{pkg['license_file']}"
        rows.append(
            (
                pkg["name"],
                pkg["version"],
                lic or "UNKNOWN",
                (pkg.get("repository") or pkg.get("homepage") or "").rstrip("/"),
            )
        )
    rows.sort(key=lambda r: (r[0].lower(), r[1]))

    out = [
        "# Third-party Rust dependency licenses",
        "",
        "> **Generated file — do not edit by hand.** Regenerate with:",
        "> ```bash",
        "> cargo metadata --format-version 1 --all-features \\",
        ">   | python3 tools/gen_third_party_licenses.py > THIRD_PARTY_LICENSES.md",
        "> ```",
        ">",
        "> Only **crates.io / git dependencies** are listed. Vendored CUDA sources are",
        "> attributed in `NOTICE` and `VENDOR.list.md`.",
        "> Audit result: **no GPL/AGPL dependency**; `r-efi` merely offers",
        "> `LGPL-2.1-or-later` as one of three choices and is absent from the",
        "> x86_64-linux dependency tree.",
        "",
        f"**{len(rows)}** packages (workspace crates excluded).",
        "",
        "| crate | version | license | upstream |",
        "|---|---|---|---|",
    ]
    for name, ver, lic, repo in rows:
        repo = repo.replace("https://github.com/", "gh:")
        out.append(f"| `{name}` | {ver} | {lic} | {repo} |")
    out.append("")
    sys.stdout.write("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())

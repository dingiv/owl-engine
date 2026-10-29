#!/usr/bin/env python3
"""按需打印第三方 Rust 依赖的许可表(审计用;**仓内不落副本**)。

用法(在仓库根):

    cargo metadata --format-version 1 --all-features \
      | python3 tools/gen_third_party_licenses.py

为什么不留副本:`Cargo.toml` / `Cargo.lock` 已经是依赖的**唯一真源**,抄一份到
文档里 = 双账本(必然漂移)。许可证/来源门走 `cargo deny check licenses bans
sources`(`deny.toml` 是 allow-list,CI 每次跑);本脚本只在需要"逐包看"时跑。

约定:
- 只列**非 workspace 成员**的包(crates.io / git 依赖);
- 仓内 vendor / 移植 / 引用的第三方代码**不在**此列 —— 见 `THIRD_PARTY.md` §2/§3/§5;
- 输出稳定排序(名字小写、版本),便于 diff。
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

    out = [f"**{len(rows)}** packages (workspace crates excluded).", ""]
    out += ["| crate | version | license | upstream |", "|---|---|---|---|"]
    for name, ver, lic, repo in rows:
        repo = repo.replace("https://github.com/", "gh:")
        out.append(f"| `{name}` | {ver} | {lic} | {repo} |")
    sys.stdout.write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

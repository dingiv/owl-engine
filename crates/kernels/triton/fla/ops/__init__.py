# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 The vLLM team
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# ⚠️ owl vendor 注记(2026-10-13):上游此文件把 **全部** ops 子模块 import 进来;
# owl 只 vendor 了 **gdn_chunked 六核的 import 闭包**(见 ../README.md),
# 故此处**不导入任何子模块**(未 vendor 的模块会 ModuleNotFoundError)。
# 需要哪个核就 `from fla.ops import <模块>` 显式导入。

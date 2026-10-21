# profiles —— 唯一配置档目录

配置层只有一个概念:**profile = 一份 TOML 配置文件**。没有 "flavor"。

## 规则(2026-10-21 · schema v2)

- **`full.toml` = 唯一全量档**:所有键 + 注释。新档一律**继承**它:

  ```toml
  extends = ["full.toml"]        # 根键;相对本文件;也接受绝对路径
  # extends = ["a.toml", "b.toml"]   # 数组:按序覆盖,后者胜
  ```

  被继承文件先全部落地(可递归,带环检测),本文件覆盖之 ⇒ 子档**只写差异**。
- **完整性按合并结果判**:合并后仍缺键 ⇒ 拒启(一次列出全部缺项)。
  `--print-config` 是唯一放行半份档的路径(迁移工具)。
- **卡位只走 `[parallel]`**:

  ```toml
  [parallel]
  mode = "single"                    # single | tensor | layer(暂未支持)
  device = []                        # 空 = 自动挑空闲卡
  # device = ["41:00.0", "42:00.0"]  # BDF / UUID / 名字子串("3090" 也行)
  ```

  列表按**去重消费**;`["3090","3090"]` = 两张不同的 3090;**禁 CUDA 序号**。
- 站点层(`--site`)与 `--set` 仍可覆盖(覆盖层不负责补全)。

## 用法

```bash
# 新档:最小差异写法
cat > ~/my-27b.toml <<'EOF'
extends = ["/abs/path/to/profiles/full.toml"]

[model]
path = "/data/Qwen3.8-27B-AWQ-INT4"

[parallel]
mode = "tensor"
device = ["3090", "3090"]
EOF
./target/release/server -p ~/my-27b.toml --check
```

也可以直接复制现成的档再改(它本来就是 `extends` + 差异):

```bash
cp profiles/qwen38_27b-3080_tp2-w4a16.toml ~/my-27b.toml
```

`-p/--profile` 只接受**文件路径**,不做目录名查找;`--flavor` /
`--list-flavors` / `--list-profiles` 已移除。

## 目录

| 文件 | 说明 |
|---|---|
| `full.toml` | **全量模板**(所有键 + 注释;新档的 `extends` 基) |
| `qwen38_27b-3090-w4a16.toml` | 3090 Ti 单卡 27B AWQ / 128k / fp8 KV |
| `qwen38_27b-3090-w4a16-dflash.toml` | 3090 Ti 单卡 27B AWQ + DFlash2 d7 / 16k |
| `qwen38_27b-3080_tp2-w4a16.toml` | 2×3080 27B AWQ + 视觉 + DFlash2 d7 / 184k |
| `qwen38_27b-3080_tp2-w4a16-vlm.toml` | 同上但 spec off(视觉对照) |
| `qwen38_27b-3090-w4a16-mtp.toml` | 3090 Ti 单卡 27B AWQ + 检查点 MTP 头 d3 / 16k |
| `qwen38_27b-3090-iq4-mtp.toml` | 27B AWQ + 外部 GGUF MTP 头 d3 / 16k |
| `qwen35_4b-3090-q4km-mtp.toml` | 4B Q4_K_M 自带 NextN MTP 头 d3 / 16k |
| `qwen35_4b-3090-q4km-vlm.toml` | 4B Q4_K_M + mmproj 视觉塔 / 8k |
| `qwen38_27b-3090-q4km-vlm.toml` | 27B Q4_K_M + mmproj-Q8_0 视觉塔 / 8k |
| `qwen38_27b-3090-iq4xs-mtp.toml` | 27B UD-IQ4_XS + 单文件 NextN MTP 头 d7 / 12k；window 4032 + T1b 自适应 |
| `qwen38_27b-3090-iq4xs.toml` | 27B UD-IQ4_XS 纯块型 / 12k |
| `qwen38_27b-3090-iq4xs-dflash.toml` | 27B UD-IQ4_XS + DFlash2-GGUF d7 / 12k |
| `qwen35_0_8b-3090-iq4xs.toml` | 0.8B GGUF IQ4_XS / 16k(开发档) |

## MTP 窗口（Phase 0，生产 MTP 档已开）

`qwen38_27b-3090-w4a16-mtp.toml` 与 `qwen38_27b-3090-iq4xs-mtp.toml` 已写
`mtp_window=4032` / `mtp_sink=64` / `mtp_window_min_ctx=12288`；后者另开 T1b
自适应 depth。`full.toml` 只保留 schema 缺省 0/64/0，不在模板里做调优。
A/B 回退：

```bash
--set spec.mtp_window=0
```

只作用于 MTP 草稿读；target verify 不动。`mtp_sink` 按池页向上取整；
非 MTP 档/缺 v2 scratch/`sink >= window` 都会 boot 拒。

## 自适应 depth（T1a，缺省关）

`[spec] adaptive_depth = true` + `depth_min = 3` + `depth_switch_ctx = 49152`
+ `adaptive_min_tpr = 3.0`（`depth` = 上限，建议 7）。两档 verify/fold/propose
图族 boot 期全量预捕获（d3+d7 共 24 face），运行期只换图引用；捕获不全 =
boot 硬失败。T1b v2：每 8 轮窗口，连续 2 窗 `tokens/round < adaptive_min_tpr`
才只降不升到 `depth_min`。生产 MTP 档暂未转正，A/B 用 `--set`（见
`perf-runs/mtp-longctx-2026-10-21/REPORT.md` §8）。

## 站点事实(复制后按机器改)

```toml
[model]
path = "/path/to/Qwen3.8-27B-AWQ-INT4"
format = "awq"                  # safetensors | awq | gguf
quant = "w4a16"                 # gguf 档写 q4k / iq4 / iq3
vision_path = ""                # 留空 = 检查点内置视觉塔

[spec]
draft_path = "/path/to/Qwen3.8-27B-DFlash2-W4A16"   # DFlash2;MTP 用 model.mtp_dir/检查点自带

[server]
bind = "127.0.0.1:8138"

[parallel]
mode = "tensor"
device = ["3080", "3080"]       # 或 BDF / UUID
```

`--site` / 位置参数只是可选的覆盖层;不写也能用(只要 profile 里已含站点事实)。

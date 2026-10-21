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
cp profiles/tp27b-3080-184k.toml ~/my-27b.toml
```

`-p/--profile` 只接受**文件路径**,不做目录名查找;`--flavor` /
`--list-flavors` / `--list-profiles` 已移除。

## 目录

| 文件 | 说明 |
|---|---|
| `full.toml` | **全量模板**(所有键 + 注释;新档的 `extends` 基) |
| `awq27b-3090-max.toml` | 3090 Ti 单卡 27B / 128k / fp8 KV(最大上下文档) |
| `awq27b.toml` | 27B 单卡基线(4k;回归/取证与测试用) |
| `spec-dflash.toml` | 27B + DFlash2 d7(16k;测试/取证用) |
| `tp27b-3080-184k.toml` | 2×3080 27B + 视觉 + DFlash2 d7 / 184k |
| `tp27b-3080-vlm-184k.toml` | 同上但 spec off(视觉对照) |
| `tp27b-3080-262k-vlm.toml` | 2×3080 27B + 视觉 + d7 / **262k 全设备池** |

## 站点事实(复制后按机器改)

```toml
[model]
path = "/path/to/Qwen3.8-27B-AWQ-INT4"
format = "awq"                  # safetensors | awq | gguf
quant = "w4a16"                 # gguf 档写 q4k / iq4 / iq3
vision_path = ""                # 留空 = 检查点内置视觉塔

[spec]
draft_path = "/path/to/Qwen3.8-27B-DFlash2-W4A16"   # depth > 0 时必填

[server]
bind = "127.0.0.1:8138"

[parallel]
mode = "tensor"
device = ["3080", "3080"]       # 或 BDF / UUID
```

`--site` / 位置参数只是可选的覆盖层;不写也能用(只要 profile 里已含站点事实)。

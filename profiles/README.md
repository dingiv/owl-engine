# profiles —— 唯一配置档目录

配置层只有一个概念:**profile = 一份 TOML 配置文件**。没有 "flavor"。

## 用法

```bash
# 常用档:复制到任意路径,改 model.dir / runtime.bind / tp_ordinals
cp profiles/tp27b-3080-184k.toml ~/my-27b.toml
$EDITOR ~/my-27b.toml
./target/release/server -p ~/my-27b.toml
```

`-p/--profile` 只接受**文件路径**,不再做目录名查找;`--flavor` /
`--list-flavors` / `--list-profiles` 已移除。

## 目录

| 文件 | 说明 |
|---|---|
| `awq27b-3090-max.toml` | 3090 Ti 单卡 27B / 128k(意图档) |
| `tp27b-3080-184k.toml` | 2×3080 27B + 视觉 + DFlash2 d7 / 184k(意图档) |
| `tp27b-3080-vlm-184k.toml` | 同上但 spec off(视觉对照) |
| `tp27b-3080-262k.toml` | 2×3080 27B / 262k(低层档,小池 + 宿主腿) |
| `tp27b-3080-262k-vlm.toml` | 2×3080 27B + 视觉 + d7 / **262k 全设备池**(单文件完整档) |
| `legacy/` | 历史低层档(回归/取证用,不再是用户主入口) |

常用档都只写"目标 + 必要覆盖";站点事实请复制后按机器补/改:

```toml
[model]
dir = "/path/to/Qwen3.8-27B-AWQ-INT4"
dflash2_dir = "/path/to/Qwen3.8-27B-DFlash2-W4A16"  # spec=dflash 时

[runtime]
bind = "127.0.0.1:8138"
tp_ordinals = [2, 3]        # 2×3080 档;单卡用 device.bdf / UUID
```

`--site` / 位置参数只是可选的覆盖层;不写也能用(只要 profile 里已含站点事实)。

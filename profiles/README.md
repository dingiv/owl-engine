# profiles —— 唯一配置档目录

配置层只有一个概念:**profile = 一份 TOML 配置文件**。没有 "flavor"。

## 规则(2026-10-21)

- **全字段显式**:档里每个键都要写;缺一个 ⇒ 拒启(错误列出全部缺项)。
  代码缺省不再兜底。生成完整档:`server -p <旧档> --print-config > <新档>.toml`。
- **卡位必须显式**:`runtime.tp_ordinals = [a,b]`(TP)或 `[runtime.device] bdf/uuid/find_name`(单卡)。
- 站点层(`--site`)与 `--set` 仍可覆盖(覆盖层不负责补全)。

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
| `awq27b-3090-max.toml` | 3090 Ti 单卡 27B / 128k / fp8 KV(最大上下文档) |
| `awq27b.toml` | 27B 单卡基线(4k;回归/取证与测试用) |
| `spec-dflash.toml` | 27B + DFlash2 d7(16k;测试/取证用) |
| `tp27b-3080-184k.toml` | 2×3080 27B + 视觉 + DFlash2 d7 / 184k |
| `tp27b-3080-vlm-184k.toml` | 同上但 spec off(视觉对照) |
| `tp27b-3080-262k-vlm.toml` | 2×3080 27B + 视觉 + d7 / **262k 全设备池** |
| `full.toml` | 全字段模板(新档起点;`--print-config` 亦可生成) |

⚠️ 每档都是**全量档**(全字段显式);`--print-config` 是唯一的
"半份旧档 → 全量档"迁移工具(它放行缺键,输出含全部缺省)。

站点事实请复制后按机器补/改:

```toml
[model]
dir = "/path/to/Qwen3.8-27B-AWQ-INT4"
dflash2_dir = "/path/to/Qwen3.8-27B-DFlash2-W4A16"  # spec=dflash 时

[runtime]
bind = "127.0.0.1:8138"
tp_ordinals = [2, 3]        # 2×3080 档;单卡用 device.bdf / UUID
```

`--site` / 位置参数只是可选的覆盖层;不写也能用(只要 profile 里已含站点事实)。

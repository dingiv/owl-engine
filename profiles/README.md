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
| `qwen38_27b-3080_tp2-w4a16.toml` | 2×3080 27B AWQ + 视觉 + DFlash2 d7 / **262k**；**FI 单图快档 + `tp_draft` + 草稿池 ring**(std decode 139.8 t/s) |
| `qwen38_27b-3080_tp2-w4a16-vlm.toml` | **兼容名**：与主档同配方(262k 视觉档；`extends` 主档) |
| `qwen38_27b-3080_tp2-w4a8.toml` | 2×3080 27B **W4A8** + DFlash2 d7 / 262k；prefill **+45%** / decode −41%(长 prompt、RAG 档) |
| `qwen38_27b-3090-w4a16-mtp.toml` | 3090 Ti 单卡 27B AWQ + 检查点 MTP 头 d3 / 16k |
| `qwen38_27b-3090-iq4-mtp.toml` | 27B AWQ + 外部 GGUF MTP 头 d3 / 16k |
| `qwen35_4b-3090-q4km-mtp.toml` | 4B Q4_K_M 自带 NextN MTP 头 d3 / 16k |
| `qwen35_4b-3090-q4km-vlm.toml` | 4B Q4_K_M + mmproj 视觉塔 / 8k |
| `qwen38_27b-3090-q4km-vlm.toml` | 27B Q4_K_M + mmproj-Q8_0 视觉塔 / 8k |
| `qwen38_27b-3090-iq4xs-mtp.toml` | 27B UD-IQ4_XS + 单文件 NextN MTP 头 d7 / 12k；window 4032 + T1b 自适应 |
| `qwen38_27b-3090-iq4xs.toml` | 27B UD-IQ4_XS 纯块型 / 12k |
| `qwen38_27b-3090-iq4xs-dflash.toml` | 27B UD-IQ4_XS + DFlash2-GGUF d7 / 12k |
| `qwen35_0_8b-3090-iq4xs.toml` | 0.8B GGUF IQ4_XS / 16k(开发档) |
| `qwen38_27b-3090-gguf-agent.toml` | 27B GGUF Q4_K_M / 32k；agent 10K 多轮：`prefix_cache` + `snap_max=4` + 快照宿主 + 消息边界快照，命中 99%+ |

> **TP2 快档口径(2026-10-22)**：`…-tp2-w4a16` / `…-tp2-w4a8` 都走
> `decode_arm=verify_arm="fi"` + `fi_plan_bake=true` + `tp_draft=true`，
> 并且**草稿池 ring**(草稿 config sliding_window=2048 ⇒ 池只留 2080 slots/20MiB)。
> 实测(262k std)：W4A16 decode **138.1** / prefill 1616.8；W4A8 prefill
> **2342.7(+44.9%)** / decode 82.1(−40.5%)。
> **`tp_vocab_split` 仍不要开** —— 半表草稿 host 合并的负收益，与 head 精度无关。
>
> **head_quant 默认 int8(2026-10-22 用户裁决)**：`rtn-int8-row`(int4 头太小,
> 质量优先)。**所有 27B AWQ 档**（full.toml 默认 + 3080 TP2 + 3090 w4a16/
> dflash/mtp/iq4-mtp）已切 int8；GGUF 档保持 `off`。
> **W8A16 多 token 快路已转正**（spec m=7/8 权重只读一遍）：同 prompt
> round **33.4ms vs int4 29.2ms**、propose **6.7ms vs 5.1ms**；旧逐行
> GEMV 的 57.0/18.7ms 已退役（mt7 专用 7 行）。速度档仍可 `--set model.head_quant=rtn-int4-g32`。
>
> **W4A8 Marlin int8 m8（2026-10-22）**：famp-marlin 的 kS8 8-row 已 port，
> 3080 A/B 无收益 ⇒ `[kernel] marlin_w4a8_m8 = false` 默认；复验显式
> `--set kernel.marlin_w4a8_m8=true`（详见 `kernels/w4a8-family.md` §20）。
>
> **W8A16 head 已上 f16 Tensor Core（2026-10-22 深夜）**：默认
> `rtn-int8-row` 自动走 `gemm_i8w_a16_tc`：HBM 读 int8 权重、shared 仍
> int8、MMA fragment 在寄存器里 `int8→half`、f16 MMA；激活保持 f16。
> kernel 级 7/8 token **980.5/983.8µs**（旧 FMA 1,451/2,070；
> 646–648 GB/s）。引擎 spec 同门 round **36.65→35.28ms**、propose
> **6.26→5.97ms**、verify **28.62→27.45ms**；f16 vs i8 仍 **0.9896
> （过门）**。无需新配置键。
>
> **W8A8 head 退回诊断档（2026-10-22 深夜）**：
> `profiles/qwen38_27b-3080_tp2-w4a16-a8.toml` 仍可跑，但 W8A16 TC
> 已同速（35.28 vs 35.31ms/round）且质量门 0.9896；W8A8 的 0.6962
> 没有速度补偿 ⇒ **不推荐**。保留作 a8/int8-TC 对照。
> 详见 `docs/devlog/perf/tp2-decode-fast.md` §10.13/§10.14。
>
> **DFlash2 lookup(2026-10-22 转正)**：主档 `[spec] lookup = true`。
> copy 任务 AL 5.91→7.00 / decode ~238→286 t/s；32k 固定尾部 prose
> AL 1.69→1.87、轮时持平；greedy 恒等门(d7,lookup on)已过。
> 回退 `--set spec.lookup=false`；阈值 `lookup_nmin/nmax/nstrong/agree`
> 见 `full.toml`。详见 `docs/devlog/perf/tp2-decode-fast.md` §10。

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
本地实测档(未入库) §8）。

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

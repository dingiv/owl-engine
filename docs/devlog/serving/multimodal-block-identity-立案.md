# 多模态块身份缺失：同文本换截图 ⇒ 复用陈旧图像 KV（2026-11-03 立案即结案）

> 症状入口：GUI agent 在 **2× 缩放**下点不中 UI 元素，坐标偏 30–40 px。
> 定谳：**不是缩放、不是图像无感删除**，是块身份只看 token —— 图像在 token 流里
> 只是字面相同的占位符，换一张截图不换一个 token ⇒ 整条链命中 ⇒ 复用上一条
> 请求的**图像 KV**，模型对着旧截图答。

## §0 一句话

`hash_block(parent, tokens)` / `content_fingerprint(tokens)` 里没有图像内容
⇒ 「同提示词 + 不同截图」判同 ⇒ `cached_tokens` 满命中，新截图根本没编码。

修复 = **逐块多模态指纹**（`owl_store::BlockTag`）：块身份混入该块覆盖的图像内容
指纹。纯文本块 `tag = 0`，身份与旧实现**逐位相同**（缓存面零回归）。

## §1 真机复现（4 点交叉验证）

工具：`tools/vision_grounding.py`（合成图 + 0~1000 归一化网格坐标 + GT 对拍）。
图 A / 图 B = 同尺寸（1280×720）、同 12 个彩色方块、位置随机不同；
提示词只有末尾 `Run tag: n1|n2` 不同 ⇒ 换图**不改 token**。

| 请求 | 图 | nonce | cached_tokens | 修复前 | 修复后 |
|---|---|---|---|---|---|
| An1 | A | 1 | 96 → 1088 | ✅ A | ✅ A |
| **Bn1** | **B** | **1** | **1088/1100** | ❌ **逐点等于 A** | ✅ B（rmse 1.06px） |
| Bn2 | B | 2 | 96 | ✅ B | ✅ B |
| **An2** | **A** | **2** | **1088/1100** | ❌ **逐点等于 B** | ✅ A（rmse 1.07px） |

修复前"换图"那两条的答复与**另一张图**的回答逐位相同（12 个方块坐标全等），
即模型看的是**上一条请求的截图**。

## §2 为什么 GUI agent 表现为"偏 30–40 px 而不是崩"

陈旧 KV 的语义 = 「按上个屏幕答」。元素在两次截图之间移动几十像素是常态
（占位符消失、下拉展开、滚动半行、光标后移）⇒ 落点差几十像素、点不中，
但页面结构与画风都对得上 —— 这正是线上看到的样子。
若元素位置差得远，症状会退化成"点到了完全无关的地方"。

## §3 为什么现有门一直绿

`apps/cli/src/workload/scenarios.rs::gui_agent_screenshots` **故意**在 system
prompt 里加 `Run tag: {nonce(pid)}`（注释写明"否则内容寻址会与旧会话共享带图块"，
本意是避免 `refs>1` 拒吸收）。副作用：整条链身份每次都不一样 ⇒ 陈旧 KV 命中
被完美遮住。真机 harness 不加 tag，于是线上必踩。

⇒ 纪律：**"带图请求的缓存正确性"必须用"同文本换图"的形状做门**，
不能用"每条请求都换 tag"的负载来验。

## §4 修法

块身份 = 原有 `(parent, tokens)` **再混入该块的图像内容指纹**：

```text
纯文本块:  hash(parent, tokens, tag=0)         ← 与旧实现逐位相同
图像块:    hash(parent, tokens, tag=img_tag)   ← 图变则块身份变
```

指纹产地（唯一）：
`VlmImageKey::of(&img.image)`（预处理后 `pixel_values` 的逐位双哈希，
与视觉塔输出缓存**同源**）→ `content_tag()` → `ImgSpan::img_tag`
（submit 侧 `RunningEngine::image_pad_law` 盖章）→
`img_evict::block_tags(len, page, spans)` 铺到每个块（同块多图用**可交换折叠**）。

落地面：

| 层 | 改动 |
|---|---|
| `owl-store` | `hash_block/content_fingerprint/tail_hash/ChainCursor::advance` 收 `tag/tags`；`KvStore`/`Leg`/`Rel` 每个入口加 `*_tagged` 变体（旧签名为薄包装，测试面零改动） |
| `owl-engine` | `ImgSpan::img_tag` + `block_tags()`；`turn_block_tags(sid, page)` 单一入口 |
| 调用点 | 调度命中面 ×2、leave-gap 新链游标、turn 收口登记、轮内增量登记、快照游标、覆盖率候选、盘索引/checkpoint（指纹**随令牌**带过空转拍，会话出表后仍可算） |

### 4.1 唯一产地律（踩过的坑）

轮内捕获（`capture_gdn_snap_tail`）原本从**会话账**取 spans —— 但会话账在 turn
收口才写回，轮内读它拿到的是**上一轮**的 spans（首轮为空）⇒ 游标/快照键用
tag=0 身份，命中面用真指纹 ⇒ 边界快照对不上 ⇒ `reuse_face` 把复用面腰斩到
**3 块**（实测 `cached_tokens 96/1100`，同图重发也只 96）。
修法 = `turn_block_tags()`：**活跃 turn 优先**，会话账只当退路。

## §5 验证

| 门 | 位置 | 结果 |
|---|---|---|
| 身份原语（红→绿） | `owl-store::kv::chain::identical_tokens_different_media_tag_do_not_alias` | 修复前红（同 hash）／修复后绿 |
| 逐块 tag + store 端到端 | `owl-engine::img_evict::different_image_content_never_shares_chain` | 换第二张图的复用面截到块 2 为止；空 tags 只命中纯文本块（**宁少不错**） |
| 真机形状 | `cli --workload vision-identity` | A=(200,200) B=(795,795)，同图重发 90% 命中 |
| 回归（6 个图像删除形状） | `cli --workload image-evict` | 6/6 PASS（删首/中/末图、工具轮后删、滑动窗口） |
| 单测 | `cargo test -p owl-store` / `-p owl-engine --lib` | 163 / 48+10 全绿 |

## §6 代价与已知限制

1. **submit 多一趟像素哈希**：1× 截图 ≈ 1 ms、2× ≈ 8 ms（相对 prefill 秒级可忽略）；
2. **逐块 tags = 1 MiB / 130k prompt**（u64 × 块数，host 侧、随 turn 释放）；
3. 同图跨会话共享**保留**（指纹相同 ⇒ 身份相同），实测 1088/1100；
4. **盘上重建（`adopt_disk_chains`）不知道图像内容** ⇒ 含图链以"纯文本身份"入树，
   永远不会被带指纹的活请求命中。语义 = 宁可少用，不得错用；
5. `vision_max_pixels` 一开（0.5 MP 等）会让像素指纹随封顶变化 —— 这是对的
   （模型看到的内容确实变了），但会让同一张原图在不同档位间不共享缓存。

## §7 对 GUI agent 的运维结论

- **立即止血（不用改码）**：给每张截图的**相邻文本**加一个短 id（png sha1 前 8 位）。
  含图块身份随之改变、其后块照旧命中 ⇒ 缓存收益几乎不损失。
  ⚠️ 加在 system prompt 上（现状 e2e 的做法）会让**整条链**失效，0% 命中。
- 修后可以直接删掉 harness 里的 run tag（缓存正确性与它无关了）。

## §8 顺带排除的假说（同一轮实测）

| 假说 | 实测 | 结论 |
|---|---|---|
| 2× 缩放 + `smart_resize` 32 对齐（1280×720 → 1280×**704**）压扁 | 模型输出 0~1000 **归一化**坐标 ⇒ 归一化天然吸收缩放；48 px 方块臂 slope=0.9983/1.0004、bias −2px@sent | 不是原因 |
| `vision_max_pixels` 静默降采样 | 本档 = 0（不缩）；2× 图 2560×1440 两轴均 32 整除 ⇒ 完全不缩 | 不是原因 |
| 图像无感删除（pad/leave-gap）污染图像内容 | pad 是**文本** token 且落在图段之外；不改像素/网格/图像 token 数 | 不是原因（但**同一层**：内容寻址只认 token） |
| ~~感知地板~~ | **本条最初写错了**：修复前测到的"10/24/32 px 方块全错"其实是**陈旧 KV 的同一批答案**（所有臂都返回第一张图的方块位置），而我误读成"模型看不见" | **不成立**。见 §8bis：修复后 1280×720 / 880 token 下 **5 px** 方块也能定位到 ±1.3 屏幕像素，两引擎一致 |

## §8bis 跨引擎对照（vLLM 8133 vs owl 8135，2026-11-03）

同一 checkpoint（`bytkim/Qwen3.8-27B-pi-AWQ-INT4`）、同一批 PNG、同样 **880 图像
token**（1280×720 → smart_resize 1280×704）、同样 fp8 e4m3 KV、temperature 0。
vLLM = 双 3080 TP2（`--flavor qwen38_3090 --context 131072` + 3080 UUID 逃生门）。

| 方块边长（送出的图里） | 等价屏幕元素（×2） | vLLM dx/dy rmse（sent px） | owl dx/dy rmse（sent px） |
|---|---|---|---|
| 48 px | 96 px | 0.90 / 0.49 | 0.92 / 0.96 |
| 16 px | 32 px | 0.74 / 1.70 | 0.74 / 1.83 |
| 10 px | 20 px | 1.50 / 1.84 | 1.27 / 1.98 |
| **5 px** | **10 px** | **0.59 / 0.86** | **0.59 / 1.19** |

结论：
1. **两引擎等价**（逐点数值贴到 0.1 px 级）⇒ 引擎侧没有额外精度税；
2. **1280×720 / 880 token 对 5 px 元素毫无压力**（0~1000 网格 1 个单位 =
   1.28 送图 px，模型几乎命中整数单位）；
3. ⇒ 线上"偏 30–40 px"不是分辨率、不是感知地板、也不是归一化映射，
   而是 §1–§4 的**陈旧图像 KV**；
4. ⇒ ROI/裁剪方案从"必须"降级为**可选的成本优化**（同样的精度可以用更少
   token 买到），不是精度补救措施。

## §9 复现/验收命令

```bash
# 1) 合成图 + GT（纯 stdlib，无 PIL）
python3 tools/vision_grounding.py make --out /tmp/vg --size 1280x720 --sq 10,16,48

# 2) 真机坐标对拍（0~1000 归一化网格；--screen-scale 2 = 屏幕 2560×1440/发出 1280×720）
python3 tools/vision_grounding.py probe --url http://127.0.0.1:8135/v1/chat/completions \
    --model qwen3.8-27b --fixture /tmp/vg/1280x720_sq48 --repeat 3 --screen-scale 2

# 3) 陈旧图像 KV 门（同文本换图；修复前必红）
./target/release/cli --workload vision-identity --url http://127.0.0.1:8135

# 4) 图像删除回归
./target/release/cli --workload image-evict --url http://127.0.0.1:8135
```

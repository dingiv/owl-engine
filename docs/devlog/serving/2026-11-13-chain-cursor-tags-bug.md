# 链哈希游标 `advance` 因 tags 少一块被拒（2026-11-13）

## 现象

```text
engine: 链哈希游标推进失败:sid=14196098734897287430 fed=36704 start=1115 新增 32 块
这轮没跑起来（0 输出）—— 原消息已排回队列（followUp）
```

- session 带 3 图（6120 image token），请求带 `owl_evict=true` ×3；
- `prompt=36685`，`fed=36704`（= prompt + 19 个已生成/尾部 token），
  `start=1115`、`full_blocks=1147`、新增 32 块；
- 报错在 `capture_gdn_snap_tail` 的 `ChainCursor::advance`，不是链内容坏。

## 根因

`turn_block_tags` 对**活跃 turn** 只按 `act.prompt_ids.len()` 生成 tags：

```rust
block_tags(act.prompt_ids.len(), page, &act.img_spans)
```

但 `capture_gdn_snap_tail` 的 `fed` 是**完整 token 链**（`prompt + out + tail`）。
当生成/spec 尾部跨过一个 32-token 页边界时：

- `full_blocks = fed / 32` 比 prompt 块数多 1；
- `tags` 仍只有 prompt 块数；
- `advance` 的 `seg_tags.len() < n` 检查**正确拒**（不静默按 0），
  整个 turn 被弃置。

这不是 leave-gap 专属：任何“生成 token 跨页边界”的 decode 快照都可能触发；
带图 + `owl_evict` 时 `fed=36685+19` 正好命中。

## 修法（两处）

### 1. `running.rs::turn_block_tags`

活跃 turn 的 tags 覆盖 `prompt + out`：

```rust
let total = act.prompt_ids.len() + act.out.len();
crate::img_evict::block_tags(total, page, &act.img_spans)
```

`img_spans` 只覆盖 prompt 区，`block_tags` 对超出部分自然给 0（生成 token
无图像指纹）；跨 prompt/生成边界的那个块仍按 prompt 的图像 span 盖章，
不会把图像块误判成纯文本。

### 2. `exec/spec.rs::capture_gdn_snap_tail`

`tail`（spec 轮内已提交、尚未入 `out` 的 token）也属于链，但 tags 只到
`out`；把 tags 补到 `full_blocks`（尾部一律 0）：

```rust
let mut tags = self.turn_block_tags(sid, page);
if tags.len() < full_blocks {
    tags.resize(full_blocks, 0);
}
```

leave-gap 的位置洞/尾部同样走这条补齐；缺的块一定是生成/尾部文本块，
不是图像块。

## 验证

- `cargo build --release -p server` 通过。
- 真机复现路径：先 warm 一个 3 图 session，再发同一请求（3 图
  `owl_evict=true`），`max_tokens=64`；带 `fed=36697+...` 的跨页边界不再
  报错，turn 正常产出 64 token。
- 失败日志中的 `fed=36704 = 36685 + 19` 与上述“prompt + out/tail”完全吻合。

## 关联

- `2026-11-13-image-append-ttft.md`（图场景成本与同轮优化）
- `2026-11-13-ttft-append-scheduler-fix.md`（小尾块/页对齐）

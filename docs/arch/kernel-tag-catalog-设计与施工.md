# 算子 tag 目录 + 构建期总表（2026-10-20）

> 用户指令：给所有算子打 tag；用宏标记；build.rs 扫描并生成表；运行时使用。
> 本文记录落地口径、为什么这样切、当前验收与下一步。

## 1. 硬约束：build.rs 看不到宏展开

`crates/kernels` 现有 **212 个生产算子**。源文本里只有 166 个字面
`impl KernelSpec`；其余 **46 个**（GGUF 33 + paged-attn v2 13）由 `macro_rules!` 生成：

```rust
impl KernelSpec for $t { const ENTRY = $entry; ... }
```

build.rs 只能读源文本，看不到 `$t` / `$entry` 的展开结果。直接扫
`impl KernelSpec for T` 会：
- 把宏模板本身当成算子（`$t` 不是类型）；
- 解析不了 `const ENTRY = crate::family::...::CONST`；
- 新增宏形态就要改 build.rs 解析器 —— 这是“脆弱”的真实来源。

**解法**：扫 **static 提升点**，不是扫 impl。

```rust
pub static LAYERNORM_F16: KernelMeta = meta_of::<LayernormF16>();
```

每个生产算子都显式写这一行（宏生成的 impl 也一样），它是源级、每算子一行、
且 build.rs 能稳定解析的锚点。tag 不需要在 build.rs 再抄一遍：它随
`KernelMeta` 走。

## 2. 四层结构

```text
tags.toml
  └─ build.rs ──→ $OUT_DIR/kernel_tags.rs
                   KernelTag / KERNEL_TAGS / tag_mask
                        │
#[kernel_tags("dom.attn", ...)]   ← owl-macros 属性宏
  └─ impl KernelSpec ──→ const TAGS
                        │
                 meta_of::<T>() ──→ KernelMeta.tags
                        │
build.rs 扫 `pub static ... meta_of::<T>();`
  └─ $OUT_DIR/ops_table.rs：TABLES / OP_COUNT
                        │
              ops::all() ──→ KernelTagIndex（运行时显式构造）
```

- **单一真源**：tag 在算子声明处；算子清单在 static 提升点。
- **无第二份手抄表**：`ops/mod.rs` 的 `TABLES` 与 16 个 `METAS` 数组已删除。
- **无全局**：`KernelTagIndex` 由 boot/工具面显式构造，作为
  `OpRegistry.tag_index` 持有，不落 static。

## 3. Tag 词表

分面命名空间（`crates/kernels/tags.toml`，47 项）：

| 面 | tag |
|---|---|
| 域 | `dom.attn` `dom.gdn` `dom.gguf` `dom.marlin` `dom.vis` `dom.elem` `dom.comm` `dom.load` `dom.cublas` `dom.flashinfer` `dom.xqa` `dom.fused` `dom.dflash` `dom.norm` |
| 量化 | `quant.w4a16` `quant.w4a8` `quant.gguf` `quant.int8` `quant.awq` |
| KV 主池 | `kv.main.fp8` `kv.main.f16` `kv.main.bf16` |
| KV 草稿池 | `kv.draft.fp8` `kv.draft.f16` `kv.draft.bf16` |
| 阶段 | `phase.prefill` `phase.decode` `phase.verify` `phase.propose` `phase.load` |
| 角色 | `role.attn` `role.gemm` `role.gemv` `role.norm` `role.rope` `role.kv_write` `role.kv_read` `role.sample` `role.elem` `role.comm` `role.embed` |
| feature | `feat.vision` `feat.dflash` `feat.tp` `feat.spec` `feat.head` `feat.graph` |

宏把 `"dom.attn"` 映射成 `KernelTag::DOM_ATTN`；tag 写错 = 编译期找不到
关联常量。`KernelTagIndex` 构造时还会校验：
- 每个生产算子至少 1 个 tag；
- 每个 tag 都在词表里；
- entry 不重复。

## 4. 运行时入口

```rust
let idx = owl_kernels::tag_index::KernelTagIndex::build()?;
let vision = idx.tagged_id("feat.vision").unwrap_or(&[]);
let fp8_kv = idx.tagged(KernelTag::KV_MAIN_FP8);
let names  = idx.entries_for(&[KernelTag::FEAT_VISION, KernelTag::KV_MAIN_FP8]);
```

`OpRegistry` boot 时已建好同一份索引：

```rust
registry.tagged("feat.vision");
registry.tag_index().counts();
```

## 5. 当前状态

- `tags.toml` 47 项；212/212 算子已打 tag（`tag_index` 测试锁死未打 = 0）。
- `build.rs` 生成 `ops_table.rs`（212 张 / 16 域）与 `kernel_tags.rs`。
- `crates/kernels/src/tag_index.rs`：倒排索引 + 结构化错误。
- `OpRegistry` 持有 `tag_index`，boot 日志打印词表/算子数。
- `cargo test -p owl-kernels --lib`：**110 passed / 0 failed**；
- `cargo test -p owl-kernels --test op_tables`：**10 passed / 0 failed**；
- `cargo test -p owl-shared`：104 passed / 0 failed；
- `cargo test -p owl-models --lib kernel_env`：5/5；
- `cargo test -p owl-cuda --lib tag_filter_tests`：3/3；
- `cargo test -p server --test config_check`：6/6。
- 顺手修了三处存量门：`cublas_gemm_f16_dst` 漏登两处锁、GGUF q8_1 静态断言
  读错文件（`gguf_quant.cu` → `gguf.cuh`）。


## 5. 配置派生 TagFilter（2026-10-20 第二轮）

用户裁决：**不显式写 `allow_tags`**。配置本身就是过滤条件：

```text
model.kind=gguf                  → quant.gguf
model.quant=w4a16                → quant.w4a16
model.quant=w4a8                 → quant.w4a8
model.head_quant=rtn-int8-row    → quant.int8 + feat.head
dispatch.kv_fp8=true             → kv.main.fp8（false → kv.main.f16）
spec.depth>0                     → feat.dflash + feat.spec（草稿池 → kv.draft.fp8/bf16）
model.vision=true                → feat.vision
runtime.tp_ordinals 双卡          → feat.tp
```

`TagFilter` 的匹配语义（`crates/kernels/src/tag_filter.rs`）：

- tag 按 **互斥 facet** 分组（`quant.weight` / `quant.head` / `kv` /
  `feature.vision` / …；见 `tags.toml` 的 `facet` 字段）；
- 同一 facet 内**单选**：算子的该组 tag 必须全部被配置选中；
- 算子在某 facet **无 tag** ⇒ 该 facet 视为通用，放行；
  （`quant.w4a16` 不会误杀 rmsnorm/add 这类与量化无关的核）
- facet 之间 AND；
- `*` / `TagFilter::wildcard()` = 全放行（诊断/兜底）。

落地两处：
1. `KernelSelectionEnv::resolve`：pick 后按 `tag_filter` 校验，不符返回
   `ModelError::Msg`（不 panic）；
2. `owl-cuda` server 的 `admission_check`：所有发射必经的最终门；
   `DiagOpts::from_config` 从 `OwlConfig` 自动派生，不读 env。

典型结果：W4A16 档拒 `marlin_gemm_w4a8_awq` / `owl_gguf_gemm_q4k`；
GGUF 档拒 marlin；fp8 KV 档拒 `vllm_paged_attention_v2_f16_*`；
vision 关时拒视觉核；spec 关时拒草稿核；通用 rmsnorm/add 始终放行。

## 6. 已收尾项

- `allowed_ops` 配置键/profile 大清单已删除；server boot 的激活集从
  `TagFilter` 派生，池审计/自检只跑放行算子。
- `VISION_OPS` 手抄表已删除；视觉由 `feat.vision` 放行。
- `kv` 拆成 `kv.main.*` / `kv.draft.*` 两个独立 facet；带 tag 必须有对应
  `pools::KV_MAIN` / `pools::KV_DRAFT` POOL 声明（机器门）。
- `feat.graph` 由宏从 `CAPTURE = NoHostWork` 自动注入，`tag_index` 逐核双向锁。
- `[dispatch] tag_wildcard = true` 提供 `*` 诊断逃生门。

## 7. 下一步（未做）

1. 清理 `profiles/legacy/*` 的历史白名单注释（键已删，注释仍提到旧机制）。
2. 输出 `--print-tags` / boot 摘要的 tag 命中表，供运维审计。
3. 若需要“一个语义多候选实现按 tag 打分”，再把 `KernelTagIndex` 的候选集
   接进 picker；当前 picker 是 op_id 单候选 + 准入校验。

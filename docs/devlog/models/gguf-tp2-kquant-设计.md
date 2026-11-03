# GGUF TP2 放开 K/I 量子 —— 立项与设计（2026-11-03）

> **一句话**：`ssm_out` 的 v-head 宽 128 小于量化原子 256，现行分片几何按"逐头取原子"
> 必然为 0 ⇒ TP 拒。本项目把这条**从"逐头原子"改成"原子粒度逐段聚合"**，并为
> `ssm_out` 重新引入 **per-rank 激活置换表**，从而让 **Q4_K / Q5_K / Q3_K / IQ4_XS
> 这四类 27B GGUF 也能在双卡上跑**。
>
> 交接对象：下一会话（本文件 = 设计 + 施工工单，可直接照 §8 开工）。
> 关联：`docs/devlog/models/gguf-format.md` §14.6（原侦察）/ §15（W-1 落地，含不变式）；
> `docs/bench/gguf-owl-vs-llamacpp-2026-11-03.md`（本轮对抗，单卡基线）。

---

## §0 现状（本轮实测钉死，不要重测）

### 0.1 现象

```
GGUF 分片几何: [blk.0.ssm_out.weight] v-head 宽 128 不是原子 256 的整数倍
⇒ 块粒度聚合不成立(该型按头不可分 ⇒ TP 拒;设计 §14.6 ②)
```

### 0.2 范围（GGUF 头实测：866 张量逐型扫过）

**只有 `ssm_out` 一个角色被卡**，其余全部合法：

| 张量 | 本仓 27B 实际类型 | 角色 | atom | 判定 |
|---|---|---|---|---|
| `blk.N.ssm_out.weight` (6144×5120) | **Q5_K / Q4_K / IQ4_XS / Q3_K** | `InHead` | **256** | ✗ |
| `blk.N.attn_qkv` (5120×10240) / `attn_gate` | IQ4_XS | `In`（连续列切） | 256 | ✓ 切点 5120 |
| `ssm_alpha` / `ssm_beta` | Q8_0 | `OutHead` | 32 | ✓ |
| `ssm_a` / `ssm_dt.bias` | F32 | `OutHead` | 1 | ✓ |
| `ssm_conv1d` / 各 norm | F32 | `Segs`/`Replicate` | 1 | ✓ |

⇒ **"以前能跑"是真的**：32 块量子（`Q4_0/Q5_0/Q8_0/IQ4_NL`，atom=32）满足
`hv=128 % 32 == 0`，TP2 现在就能开；本仓 27B 的三个 GGUF 恰好全是 K/I（atom=256）。

### 0.3 硬件/模型常量（27B，单一真源 `GgufDims`）

```
nk=16  hk=128  nv=48  hv=128  vpr=nv/nk=3  hidden=5120  value_dim=nv·hv=6144
atom(QK_K)=256  ⇒  hpa = atom/hv = 2（一原子跨 2 个文件头）
```

---

## §1 目标与非目标

### 1.1 目标（验收线）

| 编号 | 判据 | 级别 |
|---|---|---|
| **G1** | `Q4_K_M / UD-IQ4_XS / UD-IQ3_S` 三个 27B GGUF 文件在 **双 3090 Ti（TP2）** 上 boot：权重每 rank 只驻留自己那一半，零 Xid | P0 |
| **G2** | **几何门**：`ssm_out` 两 rank 的原子集**互不重叠且并集 = 全量 24 原子**；每 rank 交付的字节 == 全量张量在**给定头序**下的窗口 | P0 |
| **G3** | **数值门**：首 token ≥ 31/32、前 16 token ≥ 29/32 与单卡逐字一致（**原写「全输出 ≥28/32」不可达**：AR 的 f16 半和语义是 TP 固有浮点语义，`tp2-log.md` §18.5 定谳；长输出按混沌放大预期） | P0 |
| **G4** | **恒等门**：`gpu_dflash2_identity_gate_27b`（greedy spec ≡ 无 spec）绿 | P0 |
| **G5** | 性能：TP2 相对单卡 decode ≥ 1.6×、prefill ≥ 1.3×（参考量级，不设卡点） | P1 |

### 1.2 非目标

- 不改 GGUF 文件格式、不重转模型（那是 §7 的 B 路，仅作对照/过渡）。
- 不支持 `world > 2`（`nv=48` 只有 2 的干净分法；`world=3` 需另算窗口，本期不做，
  保持响亮拒）。
- 不动 K-quant 的解量化/GEMM 核（纯几何 + 表）。

---

## §2 几何推导（本项目的核心论证，先把它钉死再写代码）

### 2.1 为什么"逐头取原子"必然失败

现行 `shard.rs::geom_of` 的 `InHead` 分支：

```rust
if !hv.is_multiple_of(elem_per_atom) { return Err(...) }   // hv=128, atom=256 ⇒ Err
let bph = hv / elem_per_atom;                             // = 0
segs.push((g * bph, bph));                                // 段长 0
```

文件布局是 **tiled**（`g = (h % vpr)*nk + h/vpr`），一个 256 原子 = **两个连续文件头**
（`g`, `g+1`）。所以"一个头一段、每段整数个原子"在 atom>hv 时不存在。

### 2.2 但"每 rank 的原子集"是完整的

rank r 拿 grouped 头窗口 `[h0, h0+hn)`（`hn = nv/world = 24`）：

| rank | grouped 头 | 映射到的**文件头** | **原子** |
|---|---|---|---|
| 0 | 0..23 | `{0..7} ∪ {16..23} ∪ {32..39}` | `{0..3} ∪ {8..11} ∪ {16..19}` = **12 原子** |
| 1 | 24..47 | `{8..15} ∪ {24..31} ∪ {40..47}` | `{4..7} ∪ {12..15} ∪ {20..23}` = **12 原子** |

- 每段起点（文件头 0/16/32）都是偶数 ⇒ **段内原子完整**；
- 两 rank 原子集**不相交**、并集 = 24 原子 = `value_dim/atom` ✓。

**⇒ 约束不是数学的，是"逐头"这个实现选择的。** 改成"逐段（原子）"即可。

### 2.3 代价：rank 内的头序不再是 grouped 序

dest 的 24 个头按**原子序**排列，即文件头 `[0..7, 16..23, 32..39]`，对应 grouped 头：

```
[0,3,6,9,12,15,18,21,  1,4,7,10,13,16,19,22,  2,5,8,11,14,17,20,23]
```

而激活（GDN 递推输出）是 **grouped 序的连续 24 头**。两者差一张**置换表** ——
这正是单卡档的 A2 表（`ssm_out_perm`），而 TP 档在 W-1 里被**显式禁掉**了
（`SourceFacts::GGUF_TP.vhead_tiled_columns = false`，`source.rs:190` 响亮拒虚拟键）。

**本项目的关键裁决：这句"TP 不需要 A2 表"只在 `atom | hv` 时成立。**
atom>hv 时必须**条件化**地把它加回来，且表是 **per-rank** 的（长度 `value_dim/world/32`）。

---

## §3 不变式的改写（唯一需要动脑的地方）

### 3.1 旧不变式（`gguf-format.md` §15.2）

> 每一 rank 装载后的张量 == 全量**逆变换（grouped 序）**张量的**连续块**。

### 3.2 新不变式（本项目）

> 每一 rank 装载后的张量 == 全量 grouped 序张量在**该 rank 的头集**上的
> **指定头序（`HeadOrder`）**窗口；`HeadOrder = Grouped` 时退化为旧不变式，
> `HeadOrder = AtomTiled` 时即 §2.3 的原子序。

- `HeadOrder` 由 `shard` 在几何阶段**算出并上报**（不是全局常量）；
- 消费方（`gdn.rs` / `linear.rs`）按 `HeadOrder` 决定要不要 `col_index` 表；
- 表的构造与 `HeadOrder` **同源**（`shard` 产出头序 → `transform` 生成表 → 装载期校验
  表长 == 本 rank 列宽/32）。

### 3.3 自检等式（写进单测，防"错序不报错只错值"）

```
∀ rank r:  assemble(full_grouped) == concat_r( rank_r_shard )    // 由头序映射拼回
          且  act_perm_r ∘ row_perm == 恒等（本 rank 窗口内）
```

---

## §4 方案选择

| 路 | 内容 | 工期 | 评价 |
|---|---|---|---|
| **A** | 换 32 块模型（Q4_0/Q5_0/Q8_0） | 0 | 质量/体积受限，**不作为主线**；但它是**现成的对照臂**（见 §7） |
| **B** | 只把 48 个 `ssm_out` 重转成 Q5_0/Q8_0，其余保持 IQ4_XS | 半天 | 零引擎改动，**可当天拿到双卡数字**；代价 +0.7GB、需 `llama-quantize --tensor-type`（本仓 build 未见该 flag，可能要先编译新工具） |
| **C** | 改引擎（本设计） | 1–2 d | **根治**：任何 K/I 量子 GGUF 直接可 TP2，不需要重转模型 |

**决策：C 为主线，B 只作为 §7.3 的"快速对照"。** 理由：GGUF 档案会持续换量子，
每次为了 TP 重转模型不可接受；而 C 的改动面被 §2.2 压到"一个角色 + 一张表"。

---

## §5 详细设计

### 5.1 `shard.rs`（几何；纯函数，零 I/O）

**改动点 `InHead` 分支**（`crates/models/src/formats/gguf/shard.rs:322-365`）：

```
输入: nk, nv, hv, atom(=elem_per_atom), world, h0, hn, vpr = nv/nk
1. hpa = atom / hv            // atom>hv 时 = 2；hv|atom 必须整除，否则仍拒
2. 若 atom <= hv 且 hv % atom == 0  ⇒ 走旧路（bph = hv/atom，逐头段）
3. 否则（atom > hv）:
   a. 把 rank 的 grouped 头窗口 [h0,h0+hn) 按文件头 g(h) 升序排序，切成**连续段**
      （rank0 = 3 段，每段 8 个文件头）;
   b. 每段起点 g0 必须满足 g0 % hpa == 0、段长 % hpa == 0（原子对齐），
      否则 ⇒ Err("该 world 下窗口跨原子 ⇒ TP 拒"，仍是响亮拒);
   c. segs.push((g0 / hpa, seg_len / hpa))   // 单位 = 原子（既有 Cols::Segs 语义）
   d. row_atoms = hn * hv / atom
4. **上报 HeadOrder**：`AtomTiled`（新枚举，随 KeyGeom 返回）
```

**新增纯函数（供门与表共用）**：

```rust
/// 本 rank 的目标头序（dest 位置 → grouped 头号）。
pub fn rank_head_order(nk, nv, hv, atom, world, rank) -> Result<Vec<usize>, ModelError>;
```

- `atom | hv` ⇒ 返回 `[h0..h0+hn)`；
- `atom > hv` ⇒ 返回 §2.3 的原子序；
- 单测：两 rank 的并集 = `0..nv`、无重复、每个原子在**唯一** rank 里完整出现。

### 5.2 `transform.rs`（per-rank 表）

新增：

```rust
/// **per-rank 激活列序表**（q8_1 块粒度）：dest 块 p → 源（grouped）块。
/// 与全局 `vhead_q8_blocks` 同方向、同公式，只是**窗口化 + 原子序化**。
pub fn vhead_q8_blocks_rank(
    nk, nv, hv, atom, world, rank,
) -> Result<Vec<u32>, ModelError>;
```

构造：`order = rank_head_order(...)`；对 `dest_head = order[p/bph]`、
`b = p % bph`（`bph = hv/32 = 4`）⇒ `src = (dest_head - h0) * bph + b`。

门：`atom|hv` 时 `vhead_q8_blocks_rank(..., world=1, rank=0) == vhead_q8_blocks(...)`
（**旧表是新表的特例** —— 这条保证不回归单卡）。

### 5.3 `source.rs`（虚拟键与装配）

- `source.rs:190` 的 TP 拒虚拟键分支改成：**若 shard 上报 `HeadOrder::AtomTiled`
  则允许**，并改用 `vhead_q8_blocks_rank(...)` 造表（长度 = `row_atoms * 32`? 按块算：
  `hn*hv/32 = 96`）；`HeadOrder::Grouped` 仍拒（防两条规范并存）。
- `SourceFacts::GGUF_TP.vhead_tiled_columns`：从常量 `false` 改成 **`shard.head_order == AtomTiled`**
  （在 `open_shard` 后回填；`gdn.rs:585` 的建表条件自然跟着走，无需改它）。

### 5.4 `gdn.rs` / `linear.rs`（消费侧，预期零改动）

- `gdn.rs:585` 的条件是 `facts.vhead_tiled_columns && nk != nv && 量化臂` ⇒ 自动成立；
  表长必须等于 `value_dim / world / 32` —— **需要把 `value_dim` 改成"本 rank 的列宽"**
  （唯一必改点：`vec![value_dim / 32]` → `vec![value_dim / world / 32]`）。
- `linear.rs:121` 的 `GgufGatherCols` 与 `:318` 的 tiled 量化臂**逐字复用**（表长变了，
  内核按 `k` 参数走，天然适配）。

### 5.5 端到端数据流（calltree）

```
boot
 └ GgufSource::open_shard(world,rank)              source.rs
    ├ shard::geom_of(key)  ── InHead: 原子段 + HeadOrder  ◀ 改动①
    └ virtual_key("ssm_out_perm")
        └ vhead_q8_blocks_rank(...)                ◀ 改动②（新）
 └ GatedDeltaNet::new(...)
    └ if facts.vhead_tiled_columns ⇒ out_proj.set_gguf_col_index(len = value_dim/world/32)  ◀ 改动③
 forward
 └ out_proj.forward(xs)
    └ hpa; GgufGatherCols / QuantActTiled（表 = per-rank）   // 零改动
```

---

## §6 风险（按严重度）

| # | 风险 | 判据/对策 |
|---|---|---|
| 1 | **错序不报错，只错值**（§5.2 表写反 = 整数列错位） | §3.3 互逆单测 + §7.2 的"两 rank 拼回 == 全量"逐字节门；**先跑纯 CPU 门再上机** |
| 2 | TP 数值非逐位（归约序变） | G3 用 N=32 prompt 的 ≥28 一致率（照 `tp2-log.md` 既有 TP 口径），不做逐位断言 |
| 3 | `atom` 不是 `hv` 的整数倍（如 hv=96） | 保持响亮拒（现状），不静默 |
| 4 | `world=3` 的窗口跨原子 | 保持响亮拒；本期只放 world=2 |
| 5 | 回退 | 新增 `[parallel] gguf_inhead = "auto"|"grouped"|"atom_tiled"`（缺省 auto）；`grouped` = 强制旧行为（K/I 仍拒），用于 A/B 与事故回退 |

---

## §7 测试矩阵

### 7.1 纯 CPU（先跑，不占卡）

| 门 | 判据 | 落点 |
|---|---|---|
| 头序合法性 | 两 rank 并集 = `0..nv`、无重、原子不跨 rank | `transform::tests` |
| 表是旧表的特例 | `atom|hv` 且 world=1 ⇒ `vhead_q8_blocks_rank == vhead_q8_blocks` | 同上 |
| 互逆 | 按 rank 窗口：`perm ∘ deperm == id` | 同上 |
| 几何 | 27B：rank0 = 3 段×4 原子、rank1 同；两 rank 原子并集 = 24 | `shard::tests::key_geometry_27b_is_pinned` 扩表 |
| 拼回 | 两 rank 的 `ssm_out` 按头序拼回 == 全量 grouped（逐字节） | 扩 `tests/tp_load_gate_gguf.rs` |

### 7.2 真机门

| 门 | 判据 |
|---|---|
| boot | IQ4_XS / Q4_K_M / IQ3_S 三档在 3090 对上 boot，两 rank 显存对称（±300MiB） |
| 数值 | 单卡 vs TP2：N=32 prompt，≥28 逐字一致（差异必须只出现在近并列处，人工抽检 3 条） |
| 恒等 | `gpu_dflash2_identity_gate_27b` 绿（greedy spec ≡ 无 spec） |
| 性能 | decode ≥ 1.6× 单卡、prefill ≥ 1.3×；记录 AL（应与单卡同量级） |

### 7.3 快速对照（B 路，可与 C 并行）

把 48 个 `ssm_out` 重转 Q5_0/Q8_0（`llama-quantize --tensor-type`），
在**不改引擎**的情况下上双卡 —— 用来先拿到"GGUF 双卡能到多少"的**上界**，
并作为 C 路数值门的对照基准（同模型、只差一个张量的量化）。

---

## §8 施工分解（每步独立可验，不许跳步）

| M | 内容 | 验证 | 预估 |
|---|---|---|---|
| **M0** | 写 `rank_head_order` + `vhead_q8_blocks_rank` + 3 条纯 CPU 门（含"旧表是新表特例"） | `cargo test -p owl-models`，零设备 | 2 h |
| **M1** | `shard.rs` InHead 改原子段 + `HeadOrder` 上报；扩 pinned 几何表 | `shard::tests` 全绿；**boot 仍拒**（表未接） | 3 h |
| **M2** | `source.rs` 放开虚拟键 + per-rank 表；`gdn.rs` 表长改 `value_dim/world/32`；`vhead_tiled_columns` 条件化 | 纯 CPU 拼回门绿 + boot 双卡成功 | 3 h |
| **M3** | 数值门 + 恒等门 + 性能三点（512/8k/32k） | §7.2 全绿；产出 bench 表 | 2 h |
| **M4** | 文档回写（`gguf-format.md` §14.6/§15 的"TP 拒"改成"条件化"）+ 配置旋钮 | 文档与代码一致 | 1 h |

**开工第一步**（照抄）：
```bash
cd packages/owl && git status          # 工作树应干净（87e5e9f5 之后）
cargo test -p owl-models --lib formats::gguf -- --test-threads=1   # 基线
# 然后用 3090 对（8136 端口纪律）起双卡；单卡对照用 8137
```

---

## §9 代码坐标（改哪里，一目了然）

| 件 | 位置 | 动作 |
|---|---|---|
| InHead 几何 + 拒门 | `crates/models/src/formats/gguf/shard.rs:322-365` | 改（原子段 + HeadOrder） |
| 头序/表 | `crates/models/src/formats/gguf/transform.rs:313` 附近 | 新增 `rank_head_order` / `vhead_q8_blocks_rank` |
| 虚拟键 + 装配 | `crates/models/src/formats/gguf/source.rs:190-205`、`:415`（Segs 消费） | 改（条件放行 + per-rank 表） |
| 建表条件/表长 | `crates/models/src/layers/gdn.rs:585` | 改（`value_dim/world/32`） |
| 消费侧 | `crates/models/src/layers/linear.rs:121`（gather）、`:318`（tiled 量化） | **零改动**（复用） |
| Facts | `SourceFacts::GGUF_TP.vhead_tiled_columns` | 常量 → 由 shard 回填 |
| 既有门 | `shard::tests::key_geometry_27b_is_pinned`、`geometry_tiles_full_tensor_27b`、`tests/tp_load_gate_gguf.rs` | 扩/改（不变式变了） |

---

## §10 待决问题（开工前先答）

1. **`world=3` 要不要现在支持**？`nv=48 / 3 = 16` 头 = 2048 元素 = 8 原子 ✓ 可能成立
   （需验窗口原子对齐）。本期默认不做，但几何函数建议写成通用的 `world`。
2. **Q4_K_M（llama.cpp 生产档）的 `ssm_out` 是 Q4_K 还是混合**？本项目只按 atom 判定，
   与具体类型无关；但 §7.3 的 B 路要按实际类型重转。
3. **性能上限**：TP2 后 GGUF 的 decode 会不会被 AR（fp8 两核）拖住？参考 AWQ TP2 的
   实测（AR 占轮时 ~11%），预计 GGUF TP2 decode ≈ 1.7–1.9× 单卡（单卡 147.7 ⇒ 250–280）。
4. **是否顺手把 32 块档的 `ssm_out`（Q8_0 等）也走新路**？建议：`atom <= hv` 保持旧路
   （不变式简单、已验证），只在 `atom > hv` 时启用新路 —— 减少回归面。

---

## §11 结案（2026-11-03 当晚；M0–M3 实做，M4 部分）

### 11.1 落地形态（与 §5 设计的差异）

| 件 | 实际实现 |
|---|---|
| `shard.rs::InHead` | 分两支：`hv % atom == 0` 走旧路（逐头一段，零行为变化）；`atom % hv == 0 && atom > hv` 走**整原子段**（文件头升序，段起点/段长都 `% hpa == 0`，否则响亮拒）；`row_atoms = hn·hv/atom` |
| `transform.rs` | 新增 `rank_head_order`（dest 位置 → grouped 头号）与 `vhead_q8_blocks_rank`（per-rank 32 元素块表） |
| `source.rs::virtual_perm` | TP 档**重新放行**虚拟键；表按**同层** `ssm_out 物理张量`的实际原子宽生成（27B 的 ssm_out 逐层混型，全局常量不可用） |
| `gdn.rs` | 判据改为**逐层**：`vhead_tiled_columns && nk != nv && atom.is_some_and(|a| !tp_rank_shard || a > hv)`。表长仍 `value_dim/32`（**spec 已按 rank 切过** ⇒ `value_dim` 就是本 rank 值，设计 §5.4 说的 `/world` 是多余担忧） |
| `module.rs` | 新增事实 `SourceFacts::tp_rank_shard`；`GGUF_TP` 的 `vhead_tiled_columns` 回到 `true`（语义 = 文件列序是 tiled），由 gdn 逐层决定要不要表 |
| 旋钮 `[parallel] gguf_inhead` | **未做**（§6 风险 5 的回退项）—— 不需要：新路只在旧路必定拒启的形态下启用，旧档零变化；回滚 = 换回旧二进制 |

### 11.2 门（全绿）

- 零设备：`shard::ssm_out_atom_tiled_geometry_matches_rank_table`、
  `transform::rank_head_order_27b_two_atom_regimes`、`vhead_q8_blocks_rank_is_window_local`，
  以及**旧门逐位不变**（`gguf:27b` Q8_0 档的 36 个 (键,rank) 组合）。
- `gguf:27b-iq4xs`：`open_shard` 逐键过几何（866 张量）；blk.0+blk.3 **36 个 (键,rank)**
  分片读 == grouped 全量对应窗口（列序按 `rank_head_order` 换算）；装载账
  rank0/1 各 **899 件**（851 + 48 张 per-rank 置换表），量化字节 **7.12 GiB/rank**。
- 真机（双 3090 Ti，8136，`tp_vocab_split=false`，spec off）：boot ✓（权重 7.7 GiB/rank，
  两 rank 对称，零 Xid）；冒烟中文回答连贯；首 token **31/32**、前 16 token **29/32**
  与单卡逐字一致，prompt_tokens 32/32 一致。

### 11.3 性能（`llm_speedtest`，spec off，1 轮，nonce 固定）

| 点 | 单卡 3090 Ti | TP2 3090 对 | 比 |
|---|---:|---:|---:|
| prefill 512 | 1768.9 | 2186.3 | 1.24× |
| prefill 8192 | 1855.1 | 2520.6 | 1.36× |
| decode 512 | 42.06 | 62.00 | 1.47× |
| decode 8192 | 41.52 | 61.43 | 1.48× |

spec-on（DFlash2 d7）单卡 3090：decode **156.45 / 119.85**（512/8k）；3090 对 spec-on
被占卡未测，**改用双 3080 20G 补测**（同文件、同 nonce、`tp_vocab_split=true`）：

| 点 | 单 3080 | 2×3080 TP2 | 比 |
|---|---:|---:|---:|
| prefill 512/8k（spec off） | 1409.7 / 1493.3 | 1889.6 / 2142.6 | 1.34× / 1.43× |
| decode 512/8k（spec off） | 33.91 / 34.62 | **55.67 / 56.92** | 1.64× / 1.64× |
| decode 512/8k（d7） | 109.03 / 145.93 | **181.06 / 221.20** | 1.66× / 1.52× |
| AL（d7） | 6.82 | 6.32 | −7% |

**长上下文**（TP 的真正收益，20G 卡）：131k needle RETRIEVED（93.7s）、
**260k needle RETRIEVED**（238.4s，≈1115 t/s prefill，262k 池总 **11.7 GiB/rank**）；
IQ3_S（12GB）TP2 boot ✓（7.0 GiB/rank、cap ≈ 819k、decode 46.9/47.7 spec off）。

### 11.4 产物

原始数据在 `perf-runs/gguf-tp2-kquant-20261103/`（gitignored）：
boot 日志、三份 `llm_speedtest` JSON、数值门参考 + 脚本、复现命令。

### 11.5 未完成 / 下一步

1. ~~IQ4_XS 双卡 boot~~ ✓（3090 对 + **3080 对**）；~~IQ3_S 双卡 boot~~ ✓（3080 对）；
2. ~~TP2 spec-on~~ ✓（3080 对：decode 181.06/221.20，AL 6.32）；3090 对的 spec-on 未测
   （卡被其它会话占用，非阻塞）；
3. 上下文：**262144（模型 native）实测可达**（260k needle RETRIEVED；3080 对
   11.7 GiB/rank）；boot 报的 cap ≈ 750k（spec off）/ 687k（d7）—— 再高要 YaRN；
4. 文档回写已完成（`gguf-format.md` §14.5/§14.6/§15.2/§15.3/§15.4 + 新 §15.8）；
5. 可选的后续刀：`world=3` 不做（`nk=16` 不被 3 整除，模型层就拒）；32 块档保持旧路。

---

## §12 prefill 扩展比归因（2026-11-04；nsys 逐核账）

> 用户问：AWQ-INT4 的 TP2 prefill 扩展比 ~1.5×，GGUF 只有 1.3–1.4×，为什么？
> **答案：GGUF 的 GEMM 缩放是完美的（2.05–2.09×），差距来自两项"不缩放"的开销 ——
> 其一是 TP 档 GDN 掉进了**串行 recurrence** 臂（9%），其二是 AR（19%）。**

### 12.1 量法（同二进制 · 同卡对 · 同 prompt · 缓存关）

`tools/fixture/prefill-account.sh`（nsys，`prefill` 为 eager ⇒ `graphId is null` 的核），
prompt ≈ **7845 tok**、`max_tokens=1`、`prefill_chunk=2048`、prefix cache 关：

- TP1：单 3080，设备 Σ = **5282 ms**（window 5342 ms，GPU 忙 99%）；
- TP2：双 3080，**每 device** Σ = **3630 ms**（window 3682 ms）⇒ 比 **1.455×**（与
  `llm_speedtest` 的 1.34–1.43× 同量级）。

### 12.2 逐类账（每 device）

| 类别 | TP1 ms | TP2/dev ms | 比（理想 2.0） | 判读 |
|---|---:|---:|---:|---|
| GGUF GEMM（`owl_mmq2_*`） | 4384.9 | 2104.9 | **2.08** | **完美缩放** |
| attention（FlashInfer） | 225.1 | 109.7 | 2.05 | ✓ |
| 其它 GEMM（cutlass/xmma/cuBLAS） | 56.2 | 30.4 | 1.85 | ✓ |
| 其它 | 80.5 | 38.3 | 2.10 | ✓ |
| **AR（`owl_comm_ar_stage/reduce_f8`）** | 0 | **695.6** | — | **19.2%** |
| **GDN** | 263.0 | **453.0** | **0.58** | **唯一变慢** |
| norm/add/silu | 165.3 | 117.8 | 1.40 | Replicate 复制 |
| 激活量化 q8_1 | 106.8 | 79.9 | 1.34 | 两 rank 各量化一次 |
| **合计** | **5281.8** | **3629.6** | **1.455** | |

### 12.3 两项主因

**① GDN：TP 档掉进 recurrence 臂（+321 ms/device = 8.9%）**

`crates/models/src/layers/gdn.rs:823` 的臂门是**全模型形状**硬约束：

```rust
let chunked_shape_ok = self.nv == 48 && self.nk == 16 && self.hk_dim == 128 && self.hv_dim == 128;
```

AOT cubin 把 `NV=48 / NK=16` **烘成 constexpr**（`triton/gdn_chunked_aot.json` 的
`shape`），而 TP 档 spec 已按 rank 切头（`nv=24, nk=8`）⇒ 门不过 ⇒ 退到
`owl_gdn_recurrence_varlen_gqa_f16`（**按 token 串行**）。

实测（同一 7845 tok）：

| 臂 | TP1 | TP2/dev |
|---|---:|---:|
| chunked（FLA 六核） | 263 ms（其中六核 ≈185） | **不可用**（guard 拒） |
| recurrence（`owl_gdn_recurrence_varlen_gqa_f16`） | 850 ms（强制 `gdn_chunked=false`） | **419 ms** |

⇒ recurrence 在两侧都随头数减半（850→419 ✓），但仍比 chunked 的 TP 理想值
（≈ 185/2 + 其它 ≈ 131）慢 **3.4×**。

**② AR：695 ms/device（19.2%）**

128 AR/块（64 层 × 2 行并行层）× 4 块 = 512 次/rank；fp8 载荷（10.49 MB/次）。
`stage` 34 ms + `reduce` 139 ms 每块 ≈ 174 ms/块。A/B（`tp_ar_f8=false` ⇒ f16）：
块时 757 → 831 ms ⇒ **payload 减半省 74 ms/块（9.8%）**，其余约 100 ms/块是
rendezvous/偏斜等待（单流下不可省）。

### 12.4 与 AWQ 的对比（为什么"比例"更低、但绝对值更高）

| 档 | TP1 µs/token | TP2 µs/token | 比例 | TP2 绝对 t/s |
|---|---:|---:|---:|---:|
| AWQ-INT4（`tp2-roadmap.md` §1，2026-10-19 旧二进制） | 1030 | 677 | 1.522× | 1478 |
| AWQ-INT4（**2026-11-04 复测**，w4a8，3080 对） | ~535(估) | 341(8k) | **~1.57×** | **2934** |
| **GGUF IQ4_XS（本节）** | **673** | **463** | **1.455×**（GDN 修复后 1.589×） | **2160**（修复后 2353） |

> ⚠️ **2026-11-04 更正**：上表早期结论（"GGUF TP2 绝对吞吐比 AWQ 高 44%"）是拿
> **旧二进制**的 AWQ 读数（1478）比的，**不成立**。用同一二进制复测（spec off、
> 同 nonce、同卡对）：AWQ-INT4(w4a8) TP2 prefill **1942/2934**（512/8k）、
> decode 57.7/56.6；GGUF TP2 prefill 2037/2353、decode 55.7/56.9。
> ⇒ **比例两者相当（~1.57）**；绝对 prefill 在 **8k 上 AWQ 快 25%**（w4a8 的
> int8 张量核 vs GGUF 的 dequant-in-kernel），512 上 GGUF 略快（噪声级）。
> GGUF 的卖点是**显存/格式**（14.25GB、K/I 量子、20G 卡上 262k），不是速度。

### 12.5 杠杆（按性价比）

| # | 刀 | 预期 | 代价 |
|---|---|---|---|
| 1 | **chunked GDN 的 TP2 AOT 形状**（NK=8/NV=24 再编一套 6 核；guard 改双形状；runtime 按 `GdnShape` 选模块） | **1.455 → ~1.60×**（−321 ms/device） | 半天；Triton AOT 编译可复用同一组 warps/stages（smem 与 NV 无关），加一条 TP 形状的金标（对拍 recurrence） |
| 2 | **AR 与计算重叠**（四流对班；tp2-roadmap 既有下一刀） | 全藏 ⇒ **~2.0×**；只藏 payload 部分 ⇒ ~1.7× | 1–2 天（多流 + 捕获面） |
| 3 | 序列并行（row-parallel 改 reduce-scatter） | AR payload 减半 ⇒ **+0.05–0.10** | 1–2 天（层间激活按 seq 分片，LN/MLP 局部化） |
| 4 | norm/add 复制件、q8_1 重复量化 | +1–1.5% | 低（部分不可去） |

**注**：本轮机器上还有别的会话在跑（3090 单卡实验 + `nvidia-vbios` 高频占用），
绝对读数有 ±10% 漂移；但 12.2 的**逐类比**来自同一次成对 nsys 采集（TP1/TP2 相邻），
内部自洽（合计比 1.455 与 llm_speedtest 同档）。

---

## §13 chunked GDN 的 TP2 形状（2026-11-04 落地；§12 杠杆 ①）

### 13.1 问题复述

FLA 六核把 `NK/NV` **烘成 constexpr**（`triton/gdn_chunked_aot.json::shape`，
`NK=16/NV=48`）；`gdn.rs` 的臂门是硬相等 ⇒ TP 档（`nk=8/nv=24`）掉进
`owl_gdn_recurrence_varlen_gqa_f16`（按 token 串行）。实测 per-device：
**453 ms（recurrence）vs 263/2 ≈ 132 ms（chunked 理想值）**。

### 13.2 改动（内核 + 模型面，无算法改动）

| 件 | 改动 |
|---|---|
| `triton/gdn_chunked_aot.json` | 新增 `extra_shapes = [{tag: "nk8nv24", NK:8, NV:24, KD/VD:128, BT:64}]`；`shared` 与主形状**同值**（编译期严格校验，smem 与 NV 无关） |
| `triton/gdn_aot_compile.py` | 形状循环：产物 = `<核>.cubin` + `<核>_<tag>.cubin`；**pristine 符号快照**（`ORIG`/`JIT`）—— 否则第二个形状会拿到"代理的代理"（`_WarmupProxy` 无 `warmup`，实测） |
| `kernels/build.rs` | 读 `extra_shapes`（`serde_json` 进 build-deps）→ 校验 12 个产物 → 资产表加 `_<tag>` 行 |
| `family/gdn_chunked.rs` | `TP_TAG = "nk8nv24"`、`cubins::for_shape(name, tp)`；两条新单测（tag 与配置同源 / TP 六核齐全） |
| `server/gdn_chunked.rs` | 第二张函数表 `fns_tp`；`run` 按 `(nk,nv)==(8,24)` 选表；**cast/state_transpose 回落主表**（形状无关符号只装一次） |
| `models/layers/gdn.rs` | 臂门白名单 `(16,48) \| (8,24)`（KD=VD=128 共用）；其它形状继续 varlen |

### 13.3 验收

**nsys（同 §12 量法，7845-token prompt，3080 对）**：

| 类别（每 device） | 修复前 | **修复后** | 比 |
|---|---:|---:|---:|
| GDN | 453.0 ms | **132.0 ms** | **1.99×** ✓ |
| GGUF GEMM | 2104.9 | 2116.1 | 2.07× |
| attention | 109.7 | 110.3 | 2.04× |
| AR | 695.6 | 694.4 | — |
| 合计 | 3629.6 | **3324.3** | **1.455 → 1.589×** |

**端到端（`llm_speedtest`，same nonce，3080 对，spec off）**：

| 点 | 修复前 | **修复后** | 变化 |
|---|---:|---:|---:|
| prefill 512 | 1889.6 | **2036.7** | **+7.8%** |
| prefill 8192 | 2142.6 | **2353.3** | **+9.8%** |
| decode 512/8k | 55.67/56.92 | 55.34/56.30 | 持平（decode 不走 chunked） |
| TP2/TP1（8k） | 1.43× | **1.576×** | +0.15 |

**数值**（32 prompt，thinking off，与同一单卡参考比）：
`tp_vocab_split=false` 档 **L1 31/32 · L16 29/32 · L64 22/32**，
与修复前（recurrence）**逐项相同** ⇒ chunked TP 与 recurrence 在端到端不可区分，
无回归。

### 13.4 下一个天花板

AR 现在占 per-device 的 **20.9%**（694 ms）；norm/add 与 q8_1 量化复制件合计
约 +62 ms（1.9%）。若 AR 完全重叠 ⇒ 比可达 ~2.0×；若只做序列并行（payload 减半）
⇒ ~1.77×。见 §12.5 杠杆 2/3。

### 13.5 顺带发现的独立缺陷（另案）

`parallel.tp_vocab_split=true` + **greedy**（`temperature=0` / `[sampling] mode=greedy`）
⇒ **EOS 永不命中**，请求跑到 `max_tokens`。同 prompt 在 `tp_vocab_split=false` 下
3 token 正常 `stop`。立案见 `docs/devlog/serving/vocab-split-greedy-eos-立案.md`。
（生产 3080 档是 sampling 档，`cand` 路径带 `half` 偏移，未见此症状；greedy 客户端
会撞上。）

**✅ 2026-11-04 当晚已修（根因 = GGUF 装载面）**：`load_gguf` 构造 `LoadOpts` 时
漏传 `head_split` ⇒ `load_from_dir` 内的 `GgufSource` 以 `head_split=false` 打开
⇒ `output.weight` 走 Replicate，**两 rank 都拿"全表的前半"**（形状恰好相等 ⇒
静默错值）。所以只有 **GGUF** 档中招，AWQ 档（`shard_opts` 传了 head_split）正常。
修法两处：`opts.head_split = self.vocab_split` + 新增"源自担分片对账门"
（`load_group` 对 `owns_shard` 源断言 `elem_len == shape.product()`，这类漏传今后
在装载期响亮拒）。验收：split=true 的 greedy 正常 stop，且 logits top-8 与单卡
**逐值相同**；32-prompt 门回到 **31/29/22**（与未切档一致）。

---

## §14 AR 天花板定谳（2026-11-04；实测 + 历史证据）

> 结论先行：**AR 已经贴着 P2P 拷贝墙，且被数据依赖串行化。**
> 在"单序列 + GDN 递推 + 单流"的形态下，**没有可重叠的独立计算**；
> 剩下的只有"少搬字节"（序列并行/fp4）这类改布局的刀。

### 14.1 两点拟合（同一次 nsys 采集，两种 chunk）

| 请求 | chunk 数 | per-AR 载荷(fp8) | AR 总时/device | **per-AR** |
|---|---:|---:|---:|---:|
| 657 tok | 1 | 3.36 MB | 64 ms(128 AR) | **499 µs** |
| 7845 tok | 4 | 10.49 MB(尾 8.7) | 694 ms(512 AR) | **1355 µs** |

**同一请求的 f16 对照（`tp_ar_f8=false` ⇒ 走 `owl_comm_ar_fused_f16` 单核）**：

| 档 | 载荷/AR | AR 总时/device(512 AR) | per-AR | 有效带宽 |
|---|---:|---:|---:|---:|
| **fp8 两核**（生产） | 10.49 MB | 694 ms（stage 135 + reduce 560） | **1356 µs** | ~9.6 GB/s |
| f16 融合 | 20.97 MB | **1118 ms** | 2183 µs | **~12.7 GB/s** |

⇒ **f16 腿已经贴着卡对 P2P 参考 13.2 GB/s**；**fp8 腿的 reduce 只有 9.6 GB/s**，
另外**多了 stage 这一段 264 µs/AR**（量化 21MB→10.5MB 并写本端槽）。

stage 的账面：单 AR 本地流量 = 读 20.97 MB + 写 10.49 MB ≈ 45 µs（@700 GB/s），
实测 264 µs ⇒ **~5× 的本地低效**（尺度归约/同步/在飞深度），不是带宽。
reduce 侧同类（9.6 vs 12.7）。也就是说 **fp8 两核的"非传输"部分还有 ~7% 的整机账**。

- 固定项 ≈ 96 µs/AR × 512 ≈ 49 ms（1.5%），压不出东西；
- **AR 份额与 chunk 大小弱相关**（657 tok 19.0% vs 7845 tok 20.9%）——每 token 通信字节
  恒定，加 chunk 只减次数不减字节。

### 14.2 为什么"重叠"在本形态下不成立

1. **依赖链是严格串行的**：`row-parallel GEMM → AR → 下一层的 norm/column-GEMM`。
   同一序列里，AR 之前的所有算完才轮到它，AR 之后的一切都要等它 ⇒ **没有独立计算**。
2. **微批交错被 GDN 递推封死**：48/64 层是 GDN，同一请求的两个微批在 GDN 上是
   **前后依赖**（后批要前批的终态）。把 chunk 切两半让 A 算、B 搬，B 会卡在 GDN 上。
3. **CE 不占 SM，但把"读对端与本地 RMW 的重叠"变成串行四段**（§32.5 定谳）——
   实测净收益 0（prefill）/ 负（decode）。
4. **单流 + `max_batch=1`**：引擎的图/派发是单流顺序；第二请求在队列里等设备。

### 14.3 仅剩的结构性选项（都需要改布局，不是"调 AR"）

| 选项 | 省什么 | 预期比例 | 代价/风险 |
|---|---|---:|---|
| **fp8 reduce 提带宽**（9.6 → 12.7 GB/s：grid/在飞深度/访问模式微调，对照 §27.9 的 f16 融合路径） | ~250 µs/AR ≈ 128 ms | **1.59 → ~1.66** | 内核调优 + 运输门（`ar_stress_gate`/`ar_transport_gate`）；不改协议 |
| **stage 折进 row-parallel GEMM epilogue**（GEMM 直接产出 fp8 到本端槽，省 264 µs/AR） | ~135 ms | **1.59 → ~1.63** | 改 MMQ2/Marlin epilogue + 尺度布局对齐；面较宽 |
| 序列并行（Megatron SP：1 all-gather + 2 reduce-scatter / 层；4× → 3×） | ~174 ms | **~1.68** | 1–2 天；三路（prefill/verify/decode）布局改动；数值等价 |
| AR 载荷 fp4（10.49 → 5.24 MB） | ~350 ms | **~1.75** | 精度风险（每层残差加 4bit 噪声）+ 新内核 + 质量门 |
| 跨请求重叠（两流、两请求层间交错；GDN 槽按请求隔离 ⇒ 合法） | 全部 AR（有并发时） | 负载相关 | 1–2 天；调度器/执行器；单会话无效 |
| 接受 1.59 | — | — | 零成本 |

**前两项合计上限 ≈ 8%（1.59 → ~1.72），且都是内核级手术；没有"调参"空间。**

### 14.4 本轮结论

- **AR 本身不再投刀**：per-AR 已 1.36 ms（10.49 MB 单向 ≈ 8.3 GB/s），CE 已被实证否掉；
- 若要继续提比例，应选 **序列并行**（唯一不赌精度、且能落到 decode 也受益的路线），
  或 **跨请求重叠**（对 agent 的并发场景有效，对单会话无效）。

### 14.5 实证负结果：内核"加深在飞"无效（2026-11-04）

按 §14.3 表的第一行（fp8 内核提效）做了一次受控实验：把 `ar_stage` 的量化循环与
`ar_reduce` 的 QDQ 循环从 **1 组/lane** 改成 **2 组/lane**（先把两组的对端/本地
load 全部发出，再算 —— 与 `ar_reduce` 的 f16 8 路展开同一手法）。

| 档（同 prompt / 同卡对 / nsys） | reduce | stage | AR 合计 | 总时 |
|---|---:|---:|---:|---:|
| 基线（1 组/lane） | 560 ms | 135 ms | **694 ms** | **3324 ms** |
| 2 组/lane | **608 ms**(+8.6%) | 133 ms | 741 ms | 3369 ms(+1.4%) |
| 回退复测（还原） | 559 | 136 | **695** | **3321** |

**判读**：加深在飞**没有收益**（reduce 反而慢 8.6%，寄存器压力/占用下降）。
⇒ 两段都**不是延迟受限**：

- **reduce**：9.6 GB/s 已接近该核形状（v1 远端读 + 本地 RMW + 尺度解码）的带宽底；
  f16 融合腿的 12.7 GB/s 来自**不同形状**（无尺度区、单核、8 路展开的纯 add）；
- **stage**：264 µs/AR 与 **fp8 转换指令吞吐**吻合 —— 每 lane 每组 16 次
  `cvt.e4m3`、80 轮迭代、68 SM（sm_86 的 cvt 吞吐 ~1/4 clk)≈ 280 µs。
  即 **指令受限**，不是内存/延迟；除非换量化公式（改数值），否则压不动。

⇒ **§14.3 表的第一行应当划掉**：AR 内核本身没有可挖余量（用户判断成立）。
剩余路径只剩"少搬字节"（序列并行 / fp4 / 跨请求重叠）。

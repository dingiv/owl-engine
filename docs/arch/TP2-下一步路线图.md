# TP2 下一步路线图 —— prefill / decode 提速(2026-10-19 立)

> **状态:待执行** · 本卷是"接下来怎么干"的**唯一入口**(已入库:`docs/arch/`);历史逐轮账 → 同目录《TP2-实录卷.md》;
> 设计/契约/门表 → `docs/arch/TP2-施工设计.md`(入库面)。
> **用法**:每做完一刀,在 §7 进度表里打勾并补一行读数;§1 的两把尺子 + §5 的验收模板是硬口径,
> 不需要另翻文档就能开工。

---

## 0. 现状一句话(2026-10-19,全部本轮实测)

```
prefill:TP2 1489.5 t/s / TP1 973.7 t/s = **1.530×**(AR 占 18.3%)
decode :TP2   45.1 t/s / TP1  33.6 t/s = **1.342×**
decode 权重地板 = **79 t/s**(每卡 9.62 GB ÷ 760 GB/s = 12.7ms)⇒ 离地板还有 35%(都是"非权重"成本)
decode 已落刀:③④ 小核治理(−1.41ms/步,已验)
```

---

## 1. 尺子(唯一口径,任何新刀都用这三把)

| 面 | 命令 | 口径 |
|---|---|---|
| **排名词**(端到端) | `PORT=8141 TP1_CFG=/tmp/diag/tp1-8141.toml TP2_CFG=/tmp/diag/tp2-8141.toml OUT_DIR=/tmp/owl-ab-<tag> tools/tp-ab-std.sh ab` | std 五点三轮;两臂**同二进制/同卡对(ord 2/3)/同会话**;唯一差别 = rank 数 |
| **单步核账**(图内真设备时) | `tools/nsys-decode-account.sh <tag> <flavor> <cfg>` → `python3 tools/nsys_decode_account.py <tag> [<tag2>]` | nsys `--cuda-graph-trace=node`;>200µs 切段 ⇒ 段 = 一步;取段内中位 |
| **单核微基准** | `OWL_TEST_DEVICE=2 cargo test -p owl-kernels --features device --test ln_elems_bench -- --nocapture --test-threads=1` | 数值对拍 + µs(M=1/M=32 双档分辨 host/设备受限) |
| (映射路径) | `nvcc -O3 -arch=sm_86 -o /tmp/vmm-tax tools/vmm-mapping-tax-probe.cu -lcuda && /tmp/vmm-tax 2 3` | API 一次性 + 访问税阶梯(读/写 × RT/VMM × READ/RW) |

### 1.1 开工检查单(每条都有血的代价)

1. `nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid --format=csv` —— **本机有别的 agent 会话**
   (兄弟 worktree `owl-worktrees/server-engine` 也跑 owl server)⇒ 读数必须标机器状态(律㉘);
2. **专用端口**:别的会话默认绑 **8137**,共用会被彼此的清场脚本互杀(客户端 `ConnectError`
   + 服务端零错 + 进程消失,像"引擎静默崩";㉜)⇒ 用 **8141**,且运行档里 `[runtime] bind` 同端口;
3. 改代码前 `git status --short` —— 树里有 **19 项未提交的测试/工具面**(按口径留着),别误提交;
4. 任何"结构更优"的改动,**先在隔离台子上量**(律⑳),再上生产 std。

---

## 2. 现状账(TP2 单步;nsys 图内,2026-10-19)

### 2.1 decode = 19.6ms 设备 + 2.3ms host(每 token)

| 项 | µs | 占比 | 判语 |
|---|---:|---:|---|
| marlin(量化权重;352 核) | 11473 | 58% | **617 GB/s(81% 峰值)** —— 已近墙 |
| **`gemvx` = lm_head(BF16,复制)** | **3506** | **18%** | ← **刀 P1.1** |
| **AR(stage+reduce;128 对)** | **2860** | **15%** | 22.3µs/对 ← **刀 P1.2** |
| norm 两族(128 核) | 364 | 2% | ✅ ③④ 已落(旧 1547) |
| paged attention | 397 | 2% | |
| GDN 族 | ~730 | 4% | |
| embed 1 核 | 2.1 | 0% | ✅ ④ 已落(旧 288) |
| 其余小核 | ~270 | 1% | |
| **host 尾巴** | **~2300** | (10%) | fill+launch 仅 200 ← **刀 P0.3** |

### 2.2 prefill = 672 µs/token(TP2)

| 项 | µs/token | 占比 | 判语 |
|---|---:|---:|---|
| AR(491.5µs/次 × 128 / 512 tok) | 122.9 | **18.3%** | 地板 411µs ⇒ AR 本身还有 ~17% 可挖 ← **P1.3** |
| **其余(算)** | **549** | 81.7% | ⚠️ **从没做过逐核账** ⇒ 黑的 ← **P0.1** |

---

## 3. 刀(P0 → P3;每刀自带判据/量法/预期/回退)

### P0 补测量缺口 + **上主干对账**(便宜;每件 ≤ 1 小时,不动生产代码)

| 刀 | 做什么 | 判据/量法 | 预期 |
|---|---|---|---|
| **P0.0** | **RankDev × 主线骶架对账(TP 上主干的前置)**—— 见下方专节 | 28 文件 / 94 冲突块的干跑已做;验收 = 构建 + 五道门 + 排名词复现 | TP 上主干 |
| **P0.1** | **prefill 逐核账** ✅ **已完成(2026-10-19)**:新工具 `tools/prefill-account.sh` + `tools/prefill_account.py`(8k prompt + 出 1 token ⇒ eager 段逐核) | 见下表 | ✅ **结论见 P1.4** |
| **P0.2** | **W4A8 A/B** ✅ **已完成(2026-10-19 两次独立运行,合并前后各一次,逐点吻合)**:TP2 下 **prefill +45.7%**(1430–1511 → 2017–2203 t/s)、**TTFT@8k −31%**(5502 → 3765 ms)、**decode −12.4%**(44.0–44.9 → 38.3–39.6)| 配置级(`[model] quant = "w4a8"`);数据 `docs/bench/data/owl-w4a8-2026-10-19/`(base/a8 + merge-rerun)| ✅ **比 AR 大得多的 prefill 杠杆**(M2 ≥1620 直接跨过,M3 1750 也已过);⚠️ **decode 退化 12.4% 是 TP2 特有**(单卡实测"decode 不变")⇒ **待排查**(候选:量化核的发起开销/小 m 效率/图外每步量化,见 P0.1 的逐核账) |
| **P0.3** | **host 尾巴 2.1ms 定谳**:`[probes] step_profile = true` + 在 `phases.rs` 的采样收口段加临时计时(记了要拆)| 每步 `fill+launch` vs 尾巴的分布 | 上限 **decode +10%** |

#### P0.0 详案:RankDev × 主线骶架对账(2026-10-19 立案;干跑已做)

**背景**:`feat/w4a8-p1` 与 `main` 的分叉点 `fc2737db` 之后,主线走了 **80 刀**(store 状态存储 /
`#[owl_test]` 宏 / prefill 宿主税 / **`model.quant` 方案枚举** / 显式 Result 上抛 / piecewise 退役 /
B 轴覆盖率与选择层)。干跑合并 = **28 文件 / 94 冲突块**,其中三处是**结构性**的(不是文本冲突):

| # | 面 | 我方(TP 分支)| 主线 | 对账动作 |
|---|---|---|---|---|
| 1 | **每 rank 状态** | `RunningEngine { dev: RankDev<D>, peer: Option<RankDev<D>> }`(**TP 镜像设计的地基**)| 平铺 `pool/graphs/spec/env/model/rope` + **新增 9 个字段**(`cov_pending/cov_last/sel_view/select_view_steps/sel_view_fallback/stage/partial/prof/sel_keep`)| 主线的 9 个字段**收进 `RankDev`**;全仓 `self.pool/graphs/spec/env/model/rope` **≈274 处**重指 `self.dev.*`(按文件:`exec/prefill_chunk.rs` 83 · `exec/draft.rs` 68 · `exec/spec.rs` 33 · `exec/phases.rs` 29 · `scheduler.rs` 13 · `running.rs` 10 · `interpreters/eval.rs` 4 · `graph_plan.rs` 4 · `engine.rs` 4 · `backends/cpu/face.rs` 2)|
| 2 | **快照 API** | `RankDev::snap_dev(gdn_slot, key)` | **`capture_gdn_snap(sid, fed)`**(按 sid/fed 算 —— **语义不同**,不是纯改名)| 我方 4 处调用点(`exec/phases.rs` 块边界/跳页)改签名;删我方旧定义 |
| 3 | **量化档** | `model.w4a8: bool` | **`model.quant: QuantScheme`**(`w4a16`/`w4a8`) | 采用枚举、删 bool;`engine::quant_plan_of` 是唯一映射点(主线已建)|

**另两条"换风"要跟着走**(散在 21 个待解文件里):
- **显式 Result 上抛**(主线裁决,panic 面 115→19):`.expect("active")` →
  `ok_or_else(|| ModelError::EngineInvariant { what: .. })?` —— 我方新代码(tp/phases/prefill_chunk/spec/draft)
  内所有 `expect/unwrap` 按此收口;
- **store 块腿迁移**:`bank.blocks_m.cache_seq(..)` → `bank.rel.leg.register_chain(..)`(返回 up/evicted 账),
  以及主线新的池/图访问面。

**验收(与主线自己的大重构同级)**:
① `cargo build --release --bin server` + `cargo test -p owl-kernels --lib`(**全特性跑法**);
② 五道门复跑:`tp_g8_gate`(**L1 = 0 项不同**)· `tp_determinism_gate` · `tp_transport_bits_gate` ·
   `tp_assemble_gate` · `ar_*`;
③ **排名词复现**:TP1 973.7 / 33.6 · TP2 1489.5 / 45.1(±噪声);
④ 顺带验收 **`model.quant = "w4a8"` 在 TP2 下的 prefill**(主线单卡实测 +48~50%)—— 若成立,
   它比 AR 那 18.3% 大得多;
⑤ 入档:实录卷新 §N + 施工设计 §10 + 本路线图 §7 勾选。

**已完成的前期工作(重做很快)**:干跑时已解过八个文件的冲突并过 `cargo check -p owl-shared`:
`Cargo.toml`(取主线:删 `crates/graph`、加 `crates/store`)· `engine/Cargo.toml`(取主线:
`owl-store`/`owl-cpu`)· `charter.md`(我方:`comm` 已并入 `cuda/src/comm`)· `sampler.rs`(取主线)·
`exec/turn_end.rs`(取主线:store 块腿迁移)· `models/kernel.rs`(**两边都要**)·
`kernels/client/mod.rs`(**两边都要**:视觉塔 ragged 锁表)· `shared/device.rs`(**两边都要**:
BAR1 详查 + `test_device_bdf`;注意 git 会把我方函数的闭合括号当**公共行**)。

### P1 账已钉死的刀(收益确定;按值排序)

| 刀 | 目标 | 预期 | 判据 | 风险/回退 |
|---|---|---|---|---|
| **P1.1** | **decode:lm_head 按词表切**(`TpSplit::ColParallel`,vocab 248320 分半)| **−1.76ms ⇒ +9%**(45.1→49);**数值中性**;副产品每卡省 **1.18 GiB** | ①greedy 档:各 rank 算半表 + **8B 分布式 argmax**;②采样档:两半 logits allgather(≈0.08ms);③`tp_g8_gate` 的 L1 判据要改(两 rank 不再持有同一份 logits ⇒ 改为"拼起来 == 全表")| 动面宽(TpPlan/装载行切/采样收口/门);落地后 OWL_TP_VOCAB_SPLIT 开关,缺省关 |
| **P1.2** | **decode:AR 同步面瘦身** ⚠️ **三段结论(2026-10-19,§59+§60)**:①§59.1 图内实测 **协议钱 −0.53µs(−2.4%)≈0**、**单核 11.13µs vs 两核 22.13µs** ⇒ 靶 = 少发核;②1-stage 竞态**已修**(核改独立输出块 + 五处接线 + 门**连跑 5 次全绿**,§59.2/§60.1);③**但 §60.2 定谳:1-stage 在 27B 生产档从未生效** —— `tp_role` 标在最外层 `SliceView` 而判据在 `Op::Call` 臂 ⇒ 恒走 v1 ⇒ **+7% 落空** | **新路线 = v1 两核融合**(见 §10)= 在 AR 注入点(窄视图,10KB)用一个核做"写对端槽 → 等对端 → 读+加";**不需要对称槽** ⇒ 绕开结构错位 | **decode +4%**(省一个核启动 ~7µs/次 ×128 = 0.9ms/步)|
| **P1.3** | **prefill:AR 载荷 fp8** ✅ **已落地(2026-10-19)**:§47 数值件(0.8B 门 L1 max\|Δ\|=0 / L3 同包络)+ §53 收益实测(−46.2%)之上,做完**第二步(真字节)**:`cu/comm/ar_stage.cu` 的 `owl_ar_quant_copy_fp8`(64 元素区间分组 + 8 线程/组 shuffle 归约 + 8B 向量写)+ `ar_reduce.cu` 的 `owl_ar_reduce_f8`(对端反量化 + **本端同式量化**,对称律)+ 4 个新核/声明/白名单 + `runtime.tp_ar_f8` 配置面 + **载荷分档**(≥256KB 才走 fp8)| 见下 | ✅ **实测(27B TP2 双 3080)**:fp8 全档 ⇒ prefill 2155→**2312(+7.3%)**/TTFT −6.7%,**但 decode 41.35→38.69(−6.4%)**;加**载荷分档**(≥256KB)后 ⇒ prefill **2302.8(+6.8% vs W4A8)**、decode **41.24(持平)**、TTFT 1375ms ⚠️ 收益只有理论的一半(量化开销吃掉 ~0.9s/8k)⇒ 后续可优化 stage 量化核或 reduce 的本端量化 |
| **P1.4** | **prefill 下一刀 = AR**(P0.1 定谳) | 见下 | 待定 | 见下 |

### P0.1/P0.2 实测账(2026-10-19;TP2 双 3080;prompt 7840 tok;两卡合计)

| 项 | w4a16 (base) | W4A8 | Δ |
|---|---|---|---|
| **marlin(权重 GEMM)** | 6119685 µs (59%) | **2582853 µs (36%)** | **−58%**(int8 算力翻倍生效) |
| **AR(通信)** | 2275596 µs (22%) | 2285390 µs (32%) | 0 ⇒ **占比升为头号可挖项** |
| attn/GDN(几乎全是 `gdn_recurrence_varlen_gqa`) | 1359912 µs (13%) | 1371141 µs (19%) | 0 |
| quant_int8(W4A8 新增) | — | 204228 µs (3%) | prefill 下便宜(摊薄) |
| norm / 其它 | 651463 µs (6%) | 655869 µs (9%) | 0 |
| **设备合计** | **10445145 µs** | **7137966 µs** | **−31.7%** |
| 每 token(两卡) | 1332.3 µs | **910.5 µs** | ⇒ std prefill 1483 → **2135 t/s** ✓ |

**AR 明细(base)**:`ar_reduce` 2048 次 × **1037.5 µs/次**(2.12 s)· `ar_stage` 2048 次 × 73.6 µs(0.15 s)
⇒ **reduce 比 stage 贵 14×** ⇒ 它不是在传数据,是在**等对端**(spin-wait)⇒ 有效带宽 ≈ 20% 理论值。
⇒ **AR 的杠杆在"重叠"(S5 pipelined ring),不在"少传"(fp8 只对传输那 20% 有效)**。

**decode 侧破案 + 已修(W4A8 的 −12.4% → −6.6%;2026-10-19)**:`
- 旧:`owl_quant_int8_row_f16` 每次 **6.45 µs**(5120 宽、256 线程、**标量 2B 装载** 20 次/线程)
  × 345 次/步 = 2228 µs(占退化 2496 µs 的 89%);
- **修法 = §58 刀③④ 同配方**(16B `uint4` 向量装载 + 8B 向量写;**数值逐位不变** ——
  max 归约可交换、`cvt.rni.sat.s8` 逐元素 ⇒ 门 `quant_int8_row_probe` 3/3 逐元素一致):
  **6.45 → 2.73 µs/次**(-58%),每步 2228 → 951 µs ⇒ 设备 Σ 22027 → **20779 µs**;
- 端到端(同条件对照,均带别的会话 vLLM 干扰):`base 1475.5/44.30` vs `a8-v2 **2155.9/41.38**`
  ⇒ **prefill +46.1% / decode −6.6%**(原 −12.4%)。数据 `docs/bench/data/owl-w4a8-2026-10-19/quant-kernel-v2/`;
- ⚠️ **W4A8 在 m=1 的 marlin 无收益**(35.48 vs 34.84 µs/次,基础事实)⇒ decode 的**天花板 = base**;
  剩余 −6.6% = quant 950 µs(76%) + marlin +189 + AR +79 ⇒ 再压只能靠"消核"(融合 norm/silu,
  预计再拿 ~+1.8%,边际递减;或按负载选档 —— `[model] quant` 已是零成本开关)。

**decode 侧旧记录(W4A8 的 −12.4%)**:逐核对照 = `owl_quant_int8_row_f16`
**每步 ~345 次**(每 Linear 一次)× 6.46 µs = **2.23 ms/步**(占退化量 2496 µs 的 89%);
而 W4A8 的 marlin 在 m=1 时**无收益**(35.48 vs 34.84 µs/次)⇒ decode 下 W4A8 = **纯付成本**。
⇒ 修法(待做):①同一 hidden 的多次量化**复用**(q/k/v、gate/up 共享输入 ⇒ 3+2 → 2 次/层);
②把量化**融进 norm 的 epilogue**(消掉全部独立量化核)。

#### P1.1 详案:**lm_head 词表切**(2026-10-19 立案;第一步已落)

**目标**:decode 的 lm_head(BF16,复制态 3.51ms = 18%)按词表行切半 ——
每 rank 只算/只持自己那半 ⇒ **−1.76ms/步 ≈ decode +9%**,副产品**每卡省 1.18 GiB**;
数值中性(词表行独立,采样面等价)。

**现状锚点(改动面清单;行号 = 立案时)**:

| 面 | 位置 | 现状 |
|---|---|---|
| 生产 | `model.rs` `Model::forward_last` → `embed.lm_head_matmul`(tag `logits_last`) | 全量 [1,V] |
| 图槽 | `graph_fleet.rs:557` decode `OutputSlot{"logits",[1,vocab]}`;`:667` verify `[depth1,vocab_size()]`;`:843/:1193` | 槽宽 = `vocab_size()` |
| 消费 | `phases.rs:462` 采样 dtoh 全量 → `sampler::sample`;`:565` greedy host argmax;`:579` 诊断;`:598` device token 回读;`spec.rs:201` verify;`prefill_chunk.rs:1267` `output_block("logits")` | 每处都假定"本 rank 持全量" |
| 装载 | `plan/rules.rs::rule_of`(`lm_head` = `Replicate`);`formats/shard.rs::build_shard_pass`(强校验:反推分片形状必须逐字 == `want.shape`) | 头不切 |
| 门 | `tp_g8_gate` L1(两 rank 末步 logits **逐位相同**)+ L3(top-N 互含);`tp_load_gate_27b`(装载分片) | 判据 = "两 rank 同持全量" |

**✅ 第一步已落(2026-10-19;缺省关 ⇒ 零行为变化,models 220 绿 / engine 54 绿 / 0.8B 门绿)**:
`ModelSpec.head_split`(意图位)+ `head_vocab`(派生,`sharded()` 算;单卡退化 = 全量);
`Embedding::new_untied_head`(embed 表**保持全量**,头按 `head_vocab` 建;`head_vocab < vocab` ⇒
`Linear::with_tp_rule(UniformCol)`);`Linear::{out_dim,tp_rule}` / `Embedding::head_rule` /
`Model::logits_vocab()`(**本 rank logits 宽度的唯一真源**);3 个单测(切半/覆盖/整除硬门/规则)。

**✅ 第二步已落(2026-10-19;见实录卷 §63)—— 上列五步全部完成**:
配置位/槽宽/拼接面/门/验收均按本详案落地;实测 decode **+7.9%**(五点一致)、
显存 **−1.18 GiB/卡**、27B 切 vs 不切 logits **逐位相同**;新门
`tp_vocab_split_gate`;`tp_g8_gate` 加 `OWL_TP_VOCAB_SPLIT` 档(tied 拒)。

**遗留(本轮新增定谳)**:
- ★ **切档 × 草稿档(DFlash2/MTP)不兼容** —— 两族草稿头**共享 target lm_head**,
  切档后草稿 logits 只剩半表、选择面还按全量 vocab 索引 ⇒ 候选塔到半词表
  (**实测 AL 2.4→1.4 ⇒ 降级裸 decode,104→40 t/s**)。已收口为 **boot 响亮拒**;
  解 = 草稿选择面半表化(两 rank 各出候选 + host 合并;DFlash2 另需码本行偏移),
  **未施工**(估半天)。⇒ **生产 spec 档暂不能开词表切**;非投机档可用。
- 半表档 greedy 走 host argmax(每步 ~1MB D2H)——“8B 分布式 argmax”仍是后续优化。

**待做(按顺序;每步自带判据)**:

1. **配置位**:`[runtime] tp_vocab_split`(仿 `tp_ar_f8`/`tp_push`;配置显式,不读 env)→ `ModelSpec.head_split`;
2. **槽宽**:`graph_fleet` 的 logits 槽改用 `model.logits_vocab()`(decode/verify/prefill 三处);
3. **拼接面**:引擎在 logits 读回后把**对端那半**写进全量表(host 侧 0.5MB 拼接,零设备工作)
   —— 采样/argmax/诊断/checksum 全部只认"拼好的全量"⇒ 消费面**零改动**;
   (greedy 的 8B 分布式 argmax 是**后续**优化,不在第一步);
4. **门**:`tp_g8_gate` 的 L1 判据改"**两半拼起来 == TP1 全表**(逐位)";`tp_load_gate_27b` 加 head_split 档;
5. **验收**:std(decode 目标 +9% / 显存 −1.18 GiB/卡)+ 五门 + 0.8B 门连跑。

**风险 / 响亮拒**:
① 与 **spec/draft 面互斥**(草稿族双 rank 未接线 ⇒ `assemble` 里 `draft.depth = 0` 已是 v1 约束);
② 头的行数必须满足 marlin tile 约束(`124160 % 256 = 0` ✓);不满足时 `Linear` 自动退 f16 直读
(既有 `marlin_eligible` 门,非新增行为);
③ 任何"读全量 logits"的**新**调用点都必须走拼接面 —— 建议在 `read_output_f32("logits")` 的**唯一出口**
做拼接(而不是在每个消费者),否则就是"两套规范"。

#### P1.4 详案:**GDN prefill 递推核布局重写**(2026-10-19 立案并当日完工;实录 §68)

**为什么选它**:P0.1 账 `gdn_recurrence` = 664 ms/卡(19%)= prefill 唯一未调研的大块;
而原计划里的 S5(AR 藏进算里)被 §49.3/§49.4 实测挡住(链式形状串行度 ≈ 1.000)。

**形状校准(硬纪律)**:TP 把 linear-attention 的 **value 头也切半** ⇒ 生产 grid `(16, 24, 1)`
= 每 rank **nv=24 / nk=8**;按 config.json 的 `nv=48` 建台会对不上 2×。

**归因(核台 `tools/gdn-rec-probe.cu`,1.74 µs/token 的构成)**:两次 32-lane 归约 0.42 ·
q/k 标量装载 0.45(块内 8× 冗余)· 单 lane 2B 写 out 0.18 · `__syncthreads` 0.24 ·
纯算术 **0.33(19%)**。

**新形**:C=2 列/warp(归约共用)+ 无 shared/无 barrier + g/beta 每 head 只取一次 +
2 列合写 4B + 寄存器预取 2(**槽号必须编译期常量**,否则退 local memory);
**算术序未变 ⇒ 逐字节相同**。

**验收**:核 1.61× / 端到端 prefill +5.7% / 全门绿。**遗留**:q/k 仍 2 列冗余(下一刀 =
lane 持连续 4 行 + `uint2` 装载,但**求和序会变**、不再逐位);FLA chunked 版需按 (8,24)
重编 cubin 才能比较。

#### P1.5 详案:**decode 权重侧两刀**(lm_head→int8 / marlin m=1 GEMV;2026-10-19 立案)

**立案账(§71 更正后)**:每卡每 token 真读 **8.29 GB**,13.22 ms ⇒ **627 GB/s = 82% 峰值**;
拆开 **lm_head 724 GB/s(95%)** vs **marlin 613 GB/s(81%)**。m=1 配置空间**已穷尽**
(§71.3:auto 档唯一可用,`MARLIN_NO_M8` 更慢,其它 tile 未实例化)⇒ 只能自己写核。

##### P1.5-a lm_head → int8(**+4.3%**:1.76 → ~0.88 ms;用户裁决 2026-10-19)

- **权重来源 = 装载期现算**(不新增产物、不污染模型目录):
  `lm_head.weight`(bf16)在读入后按**行**对称量化 `s_row = max|w_row|/127`,`q = round(w/s)`,
  只保留 `i8 [rows, hidden]` + `f32 [rows]` 尺度;TP 词表切 ⇒ **只量化本 rank 那半**
  (它与 §63 的 `head_vocab` 同源);tied 档(embed 复用)按 **响亮拒**处理(27B 是 untied)。
  判据:量化后再反量化 vs 原 bf16 的**逐行相对误差**写进装载报告;装载耗时 +0.3~0.5 s。
- **核**:新 `cu/elems/gemv_i8w_f16.cu` —— `out[n] = Σ_k (w_i8[n,k]·s[n]) · x_dtype[k]`,
  warp 一行、`uint4`(16 int8)直读、x 先搬进 shared(10 KB/块)、8 路展开、
  warp 归约 5 步。靶 **≥700 GB/s**(= 0.9 ms),判据台 = 复用 `marlin_bw_bench` 的计时骨架
  + 逐元素对拍 bf16 参考。
- **数值门(必须先过)**:真实 hidden 上比 bf16 / int8 两套 logits ——
  **top-1 ≥ 99.9%** 且 **贪心 256 token 逐字一致**;不达标 ⇒ 停在台子上,不进生产。
- ✅ **落地实录见 §75**(2026-10-20):两处必须一起改的暗礁 —— ①**两档 dtype 各要
  一个 op_id**(f16 执行档拿 bf16 核 = 装配期 dtype 守门当场红)②派生键
  (`.i8q/.i8s`)必须进 `DERIVED_SUFFIXES` **且**层侧挂 `UniformCol` 覆盖
  (漏了 = 两 rank 读同一段 ⇒ 高半 = 低半复制,**静默错值**);
  精度 TP1 100% / TP2 96.9%(唯一分歧 near-tie 1.6e-2);
  生产 std decode **+9.5%**(prefill/TTFT 不动 —— prefill 也在末行窄切后调头,m 恒 1)。
- 落地面(九处同改的老清单):ops 表 + family 常量 + client/server + `driver` 几何 +
  `embeddings.rs` 的 `lm_head_matmul` 分派 + 装载器 + flavor 白名单 + 槽宽(graph 的 logits 槽不变)。

##### P1.5-b marlin → **m=1 int4 GEMV**(+8.5% 上限;7.02 GB 按 720 GB/s ⇒ 9.75 ms)

- **不改量化、不改布局**:直接读**同一份** marlin 打包 B(`(k/16, n·16/8)` i32 tile-perm),
  数值同式(同 scale/zp、同 dequant),只差 fp32 累加序(~1e-7)⇒ **不会新增精度风险**。
- 靶三件事(§71.4):①线性化/瓦片化搬运(每个 tile 的 128 B 一次读满、16 行共用)
  ②解包与 FMA 融合(`shift/or` → `cvt.f32.s8` → FMA,或 `dp4a` 变体的 int 累加)
  ③每 warp 多行以摊薄归约。
- **分两步走(重要)**:先把 P1.5-a 的 GEMV 骨架跑通(它没有 tile-perm 的复杂度),
  再把 B 的读取换成 marlin 布局 —— 一次只引入一个变量。
- 判据:同一形状 ≥ 700 GB/s(`marlin_bw_bench` 现成),然后 `tp_g8_gate` L1 + std 五点三轮。
- 回退:P1.5-a 与 b 都是**独立开关**(decode 走 GEMV / prefill 继续 marlin),可单独 revert。

##### 不要做的事(已封死,留证)
- lm_head **int4**:用户裁决不做(误差 2~3%,near-tie 翻转会积累);
- 靠 marlin 的 `thread_k/n` / `MARLIN_NO_M8` / override 白捡带宽:**已实测无解**(§71.3)。

#### P2.1 详案:**TP2 × 投机解码 —— 草稿族双 rank 化**(2026-10-19 立案)

**为什么走"双 rank 草稿"而不是"草稿单卡"**:草稿的 hidden 来自 target 的 taps
(TP 下 AR 之后两 rank 各持完整 hidden),单卡草稿能省一半工作,但**草稿侧的张量
并行收益(草稿本身也有 GEMM/权重遍)会丢**,且形态与主干不对称(将来 EP/PP 复用
不了)。双 rank 草稿 = 与主干同构,是**可推广形态**。

**设计参照(用户点名 vLLM / SGLang)**:

| 引擎 | 形态 | 借鉴点 |
|---|---|---|
| **vLLM** | draft model(EAGLE/MTP)与 target **同 TP group、同 world size**;draft 权重按同一套切分规则分片;通信复用同一个 group;verify 把 k+1 位置的 draft 与 target logits 一起算,接受/拒绝按 `p_target/p_draft` 判 | **草稿不另立通信域** —— 复用主干 AR 通道(我们的 AR 是"按注入点定序的单流",两 rank 同图同序天然对齐) |
| **SGLang** | `spec_worker` 与 target 共享 `tp_size`/通信组;draft KV 池独立但**同分片几何** | 草稿池按**分片后的 hkv** 建(不是全量) |

⇒ **结论:"双 rank 草稿"在本仓骨架里不是新协议,而是"同一套 TpPlan 再走一遍"**
(权重分片规则表 / 装载分片 pass / AR 注入点 / 每 rank 建图 —— 全是既有件)。

**现状锚点(立案时)**:

| 面 | 位置 | 现状 |
|---|---|---|
| 响亮拒 | `engine/src/tp.rs:225` | `k.draft.depth != 0` ⇒ **拒**(v1 约束) |
| 草稿装载 | `engine/src/spec_runtime.rs:156-172` | `LoadOpts::new(...)`(shard=1/slice_id=0 **硬编码**)+ 草稿 rope 同样单 rank |
| 草稿形状 | `models/src/formats/qwen35.rs::dflash2_draft(dir, kv_fp8)` | 全量 dims(无 sharded 变体) |
| 草稿权重 | `models/src/layers/dflash2.rs`(2556 行)/ `mtp.rs` | `Linear` 键尾 `q/k/v/o_proj`/`gate/up/down_proj` ⇒ **规则表已能认**(`rule_of` 按键尾匹配) |
| 共享件 | `spec_runtime.rs:31` | 草稿 **embed/lm_head 共享 target 不持有** ⇒ 头/表的切分跟着 target 走(P1.1 的词表切对它自动生效) |
| 池/图 | `state.rs` `draft_pool` · `graph_fleet.rs:1034/1168`(draft/fold/verify 三族) | 单 rank 几何 |

**分相(每相自带判据;建议一相一提交)**:

| 相 | 做什么 | 判据 |
|---|---|---|
| **a 形状/装载** | `dflash2_draft_sharded(dir, kv_fp8, plan)`(inter/heads 切;hidden 不切)+ `spec_runtime` 的草稿装载与草稿 rope 改 `LoadOpts{shard: world, slice_id: rank}` | 装载期强校验(反推分片形状 == `want.shape`);`tp_load_gate_27b` 加 draft 腿 |
| **b 池/图** | 草稿 KV 池按**分片 hkv** 建;draft/fold/verify 三族图在每 rank 各建;图槽 vocab 用 `Model::logits_vocab()`(与 P1.1 对齐) | 池几何单测 + 图槽形状断言 |
| **c 协议** | 草稿层的 RowParallel AR 复用**同一条 AR 序列**(两 rank 同图同序);verify 轮 k+1 深度图 + AR 交错;`tp.rs:225` 的拒改为"**仅双 rank 草稿档开启时放行**"(缺省仍拒) | `tp_g8_gate` 的 spec 档 + `ar_stress_gate` |
| **d 采样/接受** | 接受判据(`exec/spec.rs`)走**拼接后的全量 logits**(与 P1.1 的拼接面同一出口);draft logits 走共享头 | 接受账闭合(rounds/sum_m/full/partial/zero) |
| **e 门** | **TP2-spec 恒等门**:同 prompt/同 seed,TP1+spec 与 TP2+spec 的 token 序列 + 接受账**逐位一致**(TP 不该改变投机语义) | 新门 + 0.8B spec 档门(小草稿)+ 压力门 |
| **f 验收** | std(目标 **decode ≥ 90 t/s** = M4)+ 显存账 | 五点三轮 + 每卡显存 |

**风险 / 坑(预判)**:
① **verify 轮的 FI 表上界**:`fi_plan_bake` 的上界必须 ≥ 任何可能 ctx(k+1 深度图 + 长 ctx)⇒ 现有 fail-fast 门会替我们挡;
② **AR 序列同构**:草稿 AR 与主干 AR 在同一条通道交错 ⇒ 两 rank 必须**逐节点同序**(v1 已保证:同图同序);任何"单 rank 条件分支"都是雷;
③ **显存**:草稿 1.92B BF16 分片后 ~1.9GB/卡 + 草稿池(按分片 hkv 减半)+ 主干池 ⇒ 20G 卡紧,可能要 `draft_kv_fp8` 或降 `depth`;
④ **草稿 rope/标量表**等非权重件:`slice_id` 只作用于**分片维**,标量/表类必须 Replicate(规则表已按 `is_scratch_key` 兜住)。

#### 装载面参数化(用户裁决 2026-10-19:参数从**配置文件**来,不用 env;布局事实从 config.json 读)

**用户三点(原话摘要)**:①给 loader 提供**足够的参数**,从配置文件来,**不要 env**;
②checkpoint 的 `config.json` **本来就有布局信息**,不许写死;③(待补)。

**审计(env 依赖清单;`grep env_reader` 于装载面)**:

| 位置 | env | 性质 | 目标(配置键) |
|---|---|---|---|
| `engine/src/loader.rs:222` | `OWL_LOAD_DEBUG` | **行为**(逐键分相账) | `[load] debug_tap` |
| `engine/src/loader.rs:224` | `OWL_LOAD_VERIFY` | **行为**(整块回读校验) | `[load] verify` |
| `models/src/formats/load.rs:456` | `OWL_LOAD_STAGING_MB` | **行为**(暂存槽上限) | `[load] staging_mb` |
| `models/src/formats/load.rs:1614` | `OWL_L2_AWQ` | **行为**(L2 臂选择) | `[load] l2_awq` |
| `models/src/formats/qwen35.rs:461` | `OWL_LOAD_DEBUG` | 行为(同上) | 同上 |
| `models/src/formats/qwen35.rs:512/546/655` | `OWL_TEST_DEVICE` / `OWL_DRAFT_AB` | **测试面**(可留) | — |
| `models/src/formats/awq.rs:998/1031/1039` | `OWL_AWQ_IOBENCH*` | **bench 面**(可留) | — |
| `models/src/formats/load.rs:811` | `OWL_LOAD_DUMP` | 诊断面 | `[load] dump` |

**配置面现状**:`shared/src/config.rs` **没有任何 loader 旋钮**(grep `repack` = 0)
⇒ 装载参数全在 env/硬编码里 —— 这正是要收的那条缝。

**施工清单(建议一相一提交)**:

1. `shared/config.rs` 新增 `[load]` 段:`staging_mb`(缺省 128)/`verify`/`debug_tap`/`dump`/
   `repack = "auto"|"device"|"cpu"`(**设备重排裁决**:`auto` = 现状能力自述)/`l2_awq`;
2. `LoadOpts += staging_mb`(缺省 128)⇒ `load.rs` 的 `cap_from_env()` 改 `cap_bytes(opts.staging_mb)`
   (**删 env 读取**);`verify`/`debug_tap` 已有字段,只是来源换成配置;
3. `ModelLoader` 带 `knobs: &LoadKnobs`(现在只有 `face`/`slice`)⇒ `loader.rs` 三处 env 读取
   改读配置;`spec_runtime.rs` 的草稿装载同源(**一处配置,两处消费**);
4. **布局事实从 config.json 读**(用户第②点):已读 = `quantization_config.config_groups.group_0.weights`
   (`group_size` / `symmetric` / `num_bits`)与维度/`tie_word_embeddings`;新增:
   - `architectures`(**`DFlash2DraftModel` vs 主模型**)⇒ 同名键(`q_proj`)的**语义裁决依据**
     (草稿无门控 ⇒ `UniformCol`;主模型 gated ⇒ `SegmentedCol`),比"按键尾猜"硬;
   - `num_bits` ⇒ 打包字宽(`/8` 现在写死)⇒ `num_bits != 4` **响亮拒**;
   - `block_structure` / `actorder` ⇒ 将来 k-major 轴选择的依据(`want_axis` 现为键名启发式)。

**与 P2.1-a 的关系**:草稿分片挂住的根因就在"**同名键语义**"这一处;本相把裁决从
"键名启发式"挪到"**声明事实**"(`architectures`/`segments`/`num_bits`)之后,
这类挂住/拒启会变成**装载期的结构化错误**,而不是运行期的自旋。

#### P2.1-a 挂住立案的**当前诊断**(2026-10-19;下一会话从这里接)

**现象**:草稿分片装载(`tp_load_gate_27b_draft`,shard=2)在 slice0 的
`layers.0.self_attn.q_proj` 之后**挂住**;同键在 **CPU 路**(`OWL_TP_LOAD_REPACK=0`)
**响亮拒**:`qweight 是 CPU marlin 物化键,不可矩形切(shard=2)—— 分片档请开
device_repack`。

**收窄结论**:病灶 = **device rearrange 臂 × 分片几何**(`packed_raw` 按行切)。

**最强嫌疑(与旧现场账同向)**:
- `shard.rs:181-190` 的 2026-10-17 注:"`DeviceRearrange{rows,cols}` 由**层**按
  **分片 spec** 声明 ⇒ 分片 pass **不得再切一次**(原实现双切)"⇒ 说明该臂的
  **源侧读窗口**与**目标矩形**曾经对不上;
- `load.rs` 的 `[load-verify]` 现场账:"分片档实测 **GPU 块字数 n ≠ CPU 参考
  字数 b_ref.len()** ⇒ 索引表/字数口径在分片维上错位";
- 设备臂的填源走 `fill_packed_chunks(face, src, key, raw_elems, …)` ——
  **若源供给的字节数 < 装载面期望的 `raw_elems`,该路会等不满而自旋**(与
  "CPU 路拒、设备路挂"的对比一致)。

**★ 几何账已上屏(2026-10-19;`LogConfig{level:debug}` 初始化 logger 后,判据面通了)**:

```text
全量档(85 键)  layers.1.mlp.gate_proj.packed_raw  rows=17408 cols=640  n=11141120 = raw_elems ✓
               layers.1.mlp.down_proj.packed_raw  rows= 5120 cols=2176 n=11141120 = raw_elems ✓
分片档(slice0) fc_0/fc_1                        rows= 5120 cols= 640 n= 3276800 ✓
               layers.0.self_attn.q_proj           rows= 2048 cols= 640 n= 1310720 ✓(out 切半)
               layers.0.self_attn.o_proj           rows= 5120 cols= 256 n= 1310720 ✓(k 切半)
               layers.0.mlp.gate_proj / up_proj    rows= 8704 cols= 640 n= 5570560 ✓(out 切半)
               ← 打印到此为止;挂住的键 = **下一个 = `layers.0.mlp.down_proj`**
```

⇒ **几何全对**(rows/cols/n 都是分片后的值,`n == raw_elems` 一致)⇒ 病灶**不在几何**,
而在**该臂对 k 切键的源读窗口**:

- `down_proj` 是 **UniformRow(k 切)**:分片后 `cols = 1088`(= 17408/8/2),而源里它是
  **k-major 打包**张量 ⇒ 本 slice 要的是**列窗口**(k 方向的连续半段),
  在源物理布局里**不是矩形行窗口**;
- 设备臂的填源 `fill_packed_chunks(face, src, key, raw_elems, …)` 若只实现了
  **矩形(行)窗口**,k 切键就会**等不满而自旋**(与"CPU 路拒、设备路挂"完全一致);
- 而 `q_proj/o_proj/gate/up` 能过 ⇒ 它们的窗口在源布局里恰是矩形(k 方向整段、
  out 方向连续行)—— **这解释了为什么挂在 down_proj 而不是第一个键**。

**★★ 读窗口面已核(2026-10-19 同轮):span 机制**在**,挂点在下游**

读了实现,推翻了"源读窗口只支持行窗口"的第一猜想:

- `build_shard_pass` **为所有键**(含 `DeviceRearrange`)登记 `SpanEntry`
  (`shard.rs` 末尾 `spans.insert` 无条件执行),并显式钉死不变量
  **`rows/cols == w.shape`(分片形状)** —— 即 2026-10-17 那次"双切 ⇒ 半截
  marlin B ⇒ NaN"的定谳修正仍在位;DeviceRearrange 还要求 `axis == Logical`
  (否则响亮拒,不静默);
- `ShardedSource::convert_chunk_into_bytes` 对**有 span 的键**走
  `self.map(key, off, len)` → 逐段 `(src_off, n)` 读内层源;**`map` 返回 `None`
  时上抛 `Ok(None)` ⇒ `fill_packed_chunks` 会 `ok_or_else(attribution)` 报错,
  *不会*自旋**。

⇒ **挂点不在"取数"(取数失败会响亮报错),而在下游**:
① 设备侧**复合口(上传 + 发射)**的事件等待(写口在搬运前等"消费者读毕"的设备端
事件 ⇒ 若该信号没被发出,就等不满);② 或 CT repack 的 **grid/block 契约公式**
对 **k 切键**算出的发射与 staged 矩形不匹配(driver 的 `load::ct_repack` 公式)。

**★★★ 再收窄(同轮):挂点在"发射之前"(fill/暂存路),不在发射口**

`load-rearrange` 的日志打在 `upload_and_launch` **之前一行**,而分片档**连
`down_proj` 这一行都没打出来** ⇒ **挂点在到达发射之前**,即:

- `out_block` 分配(`alloc_uninit`)/ `fill_packed_chunks` 的**分块取数循环**
  (`alloc_pinned` → `src.convert_chunk_into_bytes` → push)/ 或暂存槽的借还;
- 而**不是** `upload_and_launch` 的事件等待,也不是 grid/block 契约(那两处都在
  日志之后)。

**这一格把范围锁死到三处**(按可疑度):
1. **`alloc_pinned` 的槽等待**(分块循环每轮借一个 pinned 租约;槽归还若被
   "消费者读毕"事件卡住 ⇒ 下一轮借不满 ⇒ 挂);
2. **内层源 `W4A16Source::convert_chunk_into_bytes`** 对 **k 切窗口**的子区间
   请求:懒物化缓存/索引在该窗口下可能**等不满**(注意 `map` 返回 None 只是
   `Ok(None)` 上抛 —— 若**内层**源对某段永远返回 `Ok(None)` 之前先阻塞,就挂);
3. 暂存槽 `plan`(本键 `raw_elems` 超槽 ⇒ 流式段数/环轮转)。

**下一步(一条命令级判据)**:在这三处各打一行 `log_debug!("load-rearrange", …)`
(键 + 段号 + 字节 + 槽号),跑一次就知道停在哪一处 —— 判据面已通(`LogConfig{level:3}`),
零额外成本。

**旧下一步(判据就位,按序)**:
0. **打发射账**(同一 `log_debug!` tag):对 `layers.0.mlp.down_proj` 在**全量档**
   与**分片档**各打一次 —— `staged 矩形 / grid / block / 期望事件数 / 实际发事件数`;
   两档一 diff,"第一处不等"即根因(这是唯一还没量的一格);

**旧下一步(按序,每步一条判据)**:
1. 在设备臂入口打全 `rows/cols/raw_elems/n`(**已落**,`log_debug!("load-rearrange", …)`)
   ⇒ 与源侧实际供给字节数对账(第一处不等即根因);
2. 若确认"期望 = 全量、源 = 分片"(或反之)⇒ 把该臂的**源读窗口**接上
   `SourceSpan`(分片 pass 已算出,只是设备臂没消费);
3. 顺手把 `OWL_LOAD_VERIFY=1` 的端到端对拍臂**在分片档跑一遍**(它正是为这类
   口径错位准备的;此前只在单卡档用过)—— 判据 = GPU 重排输出 vs CPU fused 逐位。

#### P2.1-c 首跑实录(2026-10-19):**TP2+投机已能装配并起跑,卡在"轮内 peer 镜像"**

**已达(全部实测)**:

| 步 | 证据 |
|---|---|
| 装配 | `TP2 装配完成(分档 + staging 握手 + 双 rank 图捕获):22.20s` → `引擎就绪(捕获态 Captured)` |
| **草稿双 rank 装载** | `DFlash2 草稿装载 1.02s`;分片几何 `q_proj rows=2048 / down_proj rows=5120 cols=1088 / gate,up rows=8704`(全 = 分片后) |
| **草稿池按分片几何** | `草稿池 10240B/tok`(单卡 20480 的一半 ✓ hkv 8→4);fp8 档 `5120B/tok` |
| 三族图 | `verify 桶形图捕获(T=5)` + `fold 桶形图捕获(m=0..3)` + `dflash propose 图捕获(encode+propose 单图,固定 8 行)` |
| 请求受理 | `[chat] 受理 … prompt=55` → prefill ✓ → `[spec-seed] 裸步成本 = 24.5ms(pos=48)`(42 t/s,与 TP2 非投机一致) |

**卡点(定谳)**:第一个投机轮里,**rank0 注入了草稿 propose 的 AR**(`bytes=81920` = 8 行草稿块 ×5120),
**rank1 一条都没有** ⇒ 两 rank 失步 ⇒ rank0 的 AR 等不满。

**根因(结构)**:`exec/spec.rs::execute_spec_round` 与 `exec/draft.rs::{dflash_propose,propose_first}`
**只打 `self.dev`(rank0 束)**;而主干路径(phases/prefill_chunk/fold)都是
`join2(dev.op(..), peer.op(..))` 的**双 rank 镜像**形态(`spec.rs:962` 的 fold 已是)。

**下一步(逐点定谳后更新;每步都有现场证据)**:

1. ✅ **草稿 propose 体镜像**(已落,`draft.rs::dflash_propose`):两 rank `join2`,
   **peer 树按 peer 自己的块重建**(`peer_tree` 闭包:用 `p.pool.dflash_kv_leaves()`
   + `p.model.embed` + **`p.spec_rt.drafter`/`draft_rope`**)—— 用 rank0 的 `d2`/rope
   建树会被算子准入当场拒(`owl_embed_f16/owl_rmsnorm_bf16: 槽 0/1 的块不在本账本`)。
   ⚠️ 该路**尚未被走到**(首轮草稿来自 prefill seed)⇒ 属"已写未验";
2. ✅ **verify 轮镜像**(已落,`RankDev::verify_dev`):FI 表同步 + step + 回读全在
   `join2`(单卡 = 原路;peer 不付 logits 的 dtoh;两半都校验)。
   证据:debug 账 `verify 镜像:ins=13 peer=true` → **`verify 完成:tok=5`** ✓
   (此前的"轮内挂住"由此解开);
3. ✅ **fold 镜像**(已落,`RankDev::fold_step_dev`):快照恢复 + fold 图入队两 rank 同跑
   —— 该处代码注释本就自认"fold 需同样补 peer 臂",现补齐(单卡 = 原路);
4. ✅ **fold 两 rank 跑通**(`fold:图入队完 ×2`);
5. ✅ **`verify_forward` 镜像**(回滚/重放路的 eager 前向):整段搬到
   `RankDev::verify_forward_dev`(`self.dev.` → `self.`,宿主 `bt_chain` 传入),
   原处留 `join2` 包装 —— 单卡档逐字旧行为;
6. ✅ **`dflash_graph_round` 镜像**(图态 propose:捕获图里烘焙着草稿 AR)——
   `join2` 两 rank 各自回放;`write_bt` 也改双 rank(页表是每 rank 独立的设备块);
7. ✅✅ **端到端跑通 + M4 达成**(2026-10-19;`max_seq = 8704`,depth=4):

| 臂(std 五点三轮,双 3080) | prefill 中位 | decode 中位 | TTFT |
|---|---:|---:|---:|
| TP2 非投机(W4A16) | 1475.5 | 44.30 | ~5500ms |
| **TP2 + DFlash2 投机(depth=4)** | **1473.0** | **88.90**(峰值 **119.4**)| 2149ms |

   逐档 decode:512 → **110.9** / 1024 → 106.5 / 2048 → 80.8 / 4096 → 78.3 / 8192 → **84.4**
   ⇒ **decode 2.0×**(prefill 持平,零税);15/15 点全绿。

8. 🔴 **遗留(质量面)**:本轮 `m` 多为 0(草稿常全拒;TP1 同档 AL≈2.4)⇒ 2.0× 里
   还有一半没吃到;下一步建 **TP2-spec 恒等门**(TP1+spec vs TP2+spec 的 token 序列
   + 接受账逐位一致)并查草稿 KV/位置/掩码面在双 rank 下是否对齐。
4. 判据:**TP2-spec 恒等门** —— 同 prompt/同 seed,TP1+spec 与 TP2+spec 的 token 序列
   + 接受账(`rounds/sum_m/AL`)**逐位一致**;
5. 白名单已补;`[spec] tp_draft = true` 是**显式闸门**(缺省仍拒)。

**方法论沉淀(本轮第三次同族)**:`join2` 里**任何一半的错误都必须立刻上屏**
—— `join2` 会等另一半到底 ⇒ 被吞的错误表现为"**挂住**"(现场:peer 的
`owl_embed_f16: 块不在本账本` 被掩了两轮)。凡并发两半,**两半都校验 + 两半都出声**。

**白名单(P2.1-c 附带)**:`tp27b` 档补了草稿族核(BF16 全程 + 选择器/卷积 + 草稿池
Classic 岛写读者),并**剔掉**本档用不到的 Classic 主池/fp8-Knhd 条目
(池一致性门 E10/E14 会把"不成套/双寻址"的条目响亮拒 —— 这次现场教了两遍)。

### P2 结构性(大;决定"能不能上一个量级")

| 刀 | 目标 | 预期 | 说明 |
|---|---|---|---|
| **P2.1** | **🔴 TP2 × 投机解码(DFlash2 / MTP)接线** | decode **2~4×** 量级(唯一能突破权重字节墙的路:每字节算更多)| TP1 侧已跑通(`spec-dflash` 档);**TP2 明确未接线**(`tp27b` 的 v1 约束:"投机草稿 draft.depth = 0(草稿族双 rank 未接线)",越界 `TpEngine::assemble` 响亮拒)⇒ 草稿族的双 rank 化 + 验证/接受路径 |
| **P2.2** | host 尾巴治理(承 P0.3)| +5~10% | 若指向"每步 logits dtoh + host 采样":改设备采样 + 4B token 回读(device argmax 核已存在);若指向簿记:从关键路径摘 |
| **P2.3** | 1-stage 融合核的 **L1 竞态**(§57.4)| ✅ **已修(2026-10-19,§59.2/§60.1)**:根因 = `out[i] += peer[i]` **原地加**与"对端可读"语义冲突(vLLM 用独立输出块)⇒ 核改独立块 + `src` 槽 + 五处接线 + 门**连跑 5 次 L1 max\|Δ\|=0** | ⚠️ 但 §60.2:该路径在 27B **从不触发**(判据错位)⇒ 修复保留(代码正确),收益另走"两核融合" |
| **P2.4** | **AR 分块流水(S5 pipelined ring)** ⚠️ **前置判据不过**(§49.3 链式形状吞吐比卡死 1.21–1.28×、§49.4 串行度 1.000;§48.3 的 97% 是单发形状不可移植)⇒ 动 S5 前必须先解决**核级重叠**(见 §9 #4/#8 的翻案条件) —— 把 AR 从"整段屏障"改成"按 slab 渐进" | prefill 上限 **1.64× → ~1.87×**(AR 完全藏进算里 = 672→549µs/token)| **只设计了没施工**(施工设计 §1 S5 / §8 刀 C6);第一步先在 `ar_channel_bench` 上验"分块 AR 能否与算重叠" | 改协议(分块 + 每 chunk 一对 produced/consumed);风险:图内节点数与 chunk 数成正比 |

### P3 人工/硬件(需要你动手;与 P1/P2 可叠加)

| 刀 | 收益 | 动作 |
|---|---|---|
| **P3.1** | prefill 上限 **1.53 → ~1.71×**;MPS 再加 **+1.6%** | ①iKVM/BIOS 查 PCIe 档(现跑 **Gen3 x16**;主板标 Gen4);②`sudo lspci -vv` 看 MPS(128B → 256B);③若开成:重跑 §1 排名词 |
| **P3.2** | ~**1.80×**(仅 3090 对)| NVLink 桥采购 + 装车(NVLink 下 AR 411µs → 93.6µs)|

---

## 4. 已封死的路(**不要重开**;每条都有实测出处)

> ⚠️ **本表是"当前判据"而不是"永久判决"**:其中若干条是**单轴 A/B × 单一形状 × 单一机器状态**下的结果
> —— 换组合就可能借尸还魂。**逐条的范围与翻案条件 → §9 待翻案清单**(12 条)。

| 路 | 判决 | 出处 |
|---|---|---|
| **绕开 VMM(换 runtime 映射)** | ❌ 现役是**写腿**:写腿税 0.4–1.2% 全档免疫;读腿才有 14–24%(10KB 档≈0)⇒ decode ≈0.01% / prefill ≈0.2% | §60 |
| "API 成本贵" | ❌ VMM 一次性 1.47–1.89ms(装配期;`cuMemSetAccess(peer)` 一发占 58%)与 runtime 1.73ms 同量级 | §60.3 |
| CE 换通道 | 三段串行,净负 ⇒ 回退 | §32 |
| 图内跨流藏写腿 | PCIe 写与本地访存完全串行 | §41 |
| 读腿 97% 重叠可移植 | 形状特有(长同质链) | §48/§49 |
| TBO / 微批 | 与 N 无关,零重叠 | §49.3 |
| vLLM 的 D 结构照搬 | 它的票写向对端;本引擎 header 留本端,且载荷由上游 GEMM 写 | §57.3 |
| 1-stage 融合核(直接转正)| L1 间歇性失败 ⇒ 必须先修竞态(P2.3)| §57.4 |
| "VMM 读腿税 17–20%" / "3.8–5.9%" | 旧读数:前者是跨程序伪差;后者低估(本会话同口径 13.9–20.7%)⇒ 以 §60 阶梯为准 | §50.8/§60 |
| "decode 换结构可得 5.8%" | 修正为 +1.2%;差额在核的同步面 | §56.4/§56.5 |
| "TP 对 decode 零收益" | ❌ 错标签(42.14 是 TP2-v1 臂;真 TP1=32.02) | §58.2 |
| fp8 用在 decode | 载荷才 10KB ⇒ 减半只省 0.4µs | §53 |

---

## 5. 每刀的验收模板(照这个填,缺一不算完)

```
刀名:
① 判据先行:这条改动"怎么证明对"(数值判据/门/计数器的**路径存在性**)—— 律㉗/⑱
② 单核台子:ln_elems_bench / ar_channel_bench 的 µs(隔离读数)
③ 图内读数:nsys 逐核(这一步的核时延真降了吗)
④ 端到端:tools/tp-ab-std.sh ab(TP1/TP2 两臂,C SV 存档)
⑤ 门:tp_g8_gate(L1=0)· tp_determinism_gate · tp_transport_bits_gate · tp_assemble_gate
    · cargo test -p owl-kernels --lib(**全特性**跑法!)· 涉及核:marlin/ln 家族门
⑥ 入档:实录卷新 §N(逐轮账 + 教训)· 施工设计 §10 表更新(当前形态)
⑦ 回退:开关缺省关 / 独立提交(可 revert)
```

---

## 6. 里程碑(拿尺子说话)

| 里程碑 | decode(TP2)| prefill(TP2)| 比值 | 构成 |
|---|---|---|---|---|
| 现在 | **45.1** | **1489.5** | 1.342× / 1.530× | ③④ 已落 |
| **M1**(P1.1+P1.2+P0.3)| **≥ 52** | ≥ 1500 | ≥1.45× | 三个"账已钉死"的刀 |
| **M2**(+P1.3/P1.4)| ≥ 52 | **≥ 1620**(1.66×)| prefill 比值 1.66× | AR fp8 + prefill 逐核账结论 |
| **M3**(+P3.1)| ≥ 52 | **≥ 1750**(~1.75×)| — | BIOS Gen4 + MPS |
| **M4**(P2.1 投机)| **≥ 90~120** | — | — | TP2 投机接线(量级跳) |
| **M5**(P2.4 AR 分块流水;上限模型)| ≥ 60 | **≥ 1820**(~1.87×)| — | 把 AR 藏进算里 —— **1.64× 这个上限本身被抬起来** |

---

## 7. 进度表(做完就勾;读数写实测值)

| 刀 | 状态 | 读数(实测)| 门 | 入档 |
|---|---|---|---|---|
| ③④ 小核治理 | ✅ 已落(2026-10-19)| decode 42.1→**45.07**;设备 −1412µs/步 | 全绿 | §58.10 |
| P0.0 RankDev × 主线骨架对账 | ✅ **已完成并合入主干**(2026-10-19;`1d0ed15` 快进 main) | 28 冲突文件 · 双向保留(store 门面 × TP 骨架)· 0 errors · 三门全绿 · 0.8B 冒烟 | 全绿 | — |
| P0.1 prefill 逐核账 | ✅ **已完成**(2026-10-19;新工具 `tools/prefill-account.{sh,py}`) | w4a16: marlin 59%/AR 22%/GDN 13%;W4A8: marlin 36%/**AR 32%**/GDN 19% | 全绿 | §P0.1 账表 |
| P0.2 W4A8 A/B | ✅ **已完成**(2026-10-19,两次独立运行吻合) | prefill **+45.7%**(1430→2100+)/TTFT −31%/decode −12.4% | 全绿 | §P0.1/P0.2 账表 |
| **P0.3 host 尾巴定谳** | ✅ **已完成 + 定谳修正**(2026-10-19;§66.5)| **真实 host 暴露 ≈0.6ms/步(~3%)** = fill+launch 0.25 + host-sample 0.35(旧账 2.1~2.3ms 是**探针污染** —— `query_metrics` 每步 1.63ms,已降频修)| — | §66 |
| **P0.4 第二次对账 + 重基线** | ✅ **已完成**(2026-10-19;`main@c0e38ae` → 分支 = `20cd14a`,**0 冲突**)| 合并持平略好(v1+f8 中位 2351.2/decode 41.67);带进桶图/快照税/chunk1024 | 全绿(11 门 + 1-stage 5/5) | §61 |
| **P0.5 fp8 载荷路治理(顺路)** | ✅ **已完成**(2026-10-19;三个真 bug:逐字节粒度 / 对称律从未成立 / 奇数 n 未对齐 + shuffle 发散)| **prefill +3.9%(五点一致)**;push×fp8 从 **711 → 2387**(修好但比 v1 低 1% ⇒ 不用)| `ar_transport_gate` **+G1b f8 腿** + 11 门全绿 | §61.3/61.4 |
| **P1.1 lm_head 词表切** | ✅ **已落(三步;2026-10-19)** | 非投机 decode **+7.9%**(44.55→48.06)/**spec 档 +5.5%**(104.05→109.74,五点全正,0 降级)/prefill 持平/**显存 −1.18 GiB/卡**;27B 切 vs 不切 logits 逐位相同 | 四门 + 新门全绿 | §63/§64 |
| **P1.2 AR 同步面瘦身** | ⚠️ 靶被实测改写(§59.1:协议钱≈0,**少发核才是杠杆**)⇒ **v1 两核融合已落**(2026-10-19;§11.2 细案) | **非投机 +3.8%**(47.44→49.25);0.8B 门 5/5 绿;**spec 档待更多样本**;⚠️ 生产档需 `[cuda] capture_slab_mb` 解封 | §65 |
| P1.3 AR 载荷 fp8 | ✅ **已完成**(2026-10-19;§47 数值件 + §53 收益 + **第二步真字节**) | prefill **+6.8%**(分档后)/decode 持平/0.8B 门 L1 max\|Δ\|=0 | 全绿 | §59 前段 |
| **P1.4 GDN prefill 布局重写** | ✅ **已完成(2026-10-19;§68)** | 核台 1.75→**1.08 µs/token(1.61×;out/state 逐字节相同)**;端到端 std(TP2/双 3080)prefill **1486→1570(+5.7%)**、decode 49.41(未动;GDN decode 走另一核);归因:81% 是布局税 | `gdn_*_golden` 三门 + `tp_g8_gate`(L1)+ determinism/transport/assemble 全绿 | §68 |
| P2.1 TP2 投机接线 | ✅ **首跑跑通 + M4 达成**(2026-10-19;草稿双 rank:装载/池/图/四类镜像)| prefill **1473.0** / decode **88.90**(峰值 119.4)⇒ **2.0×**;15/15 全绿 | 草稿装载门 + 六门全绿 | §62 + §3 P2.1 详案 |
| ~~P2.2 host 尾巴治理~~ | ✅ **结案(无可修)**(2026-10-19;§66.5:靶不存在 —— 循环尾=探针假象;真实 0.6ms 全是必要工作)| | | §66.5 |
| P2.3 1-stage L1 竞态 | ✅ **已修**(2026-10-19,门 5/5 绿);⚠️ 路径不触发(§60.2) | | | |
| **P1.5-a lm_head → int8** | ✅ **已完成(TP1 + TP2 词表切;2026-10-20)** | 核 **718.8 GB/s**;生产 std(3080 对 @1024/256)decode **45.09 → 49.37(+9.5%)**、prefill/TTFT 不变;精度 TP1 **344/344**、TP2 **744/768(唯一分歧 = 1.6e-2 near-tie)** | 核带宽台 ✅;`gemv_i8w` 双 dtype 选择门 ✅;派生键切分 CPU 门 ✅;`tp_head_i8_gate`(两进程)✅;TP2 0.8B 装配/确定性门 ✅ | §72/§75 |
| **AR 融合转正(§11.2 → 生产)** | ✅ **已完成(2026-10-20;§76.3)** | 两核 → 一核:decode **48.87 → 50.23(+2.4%)**(2 冷启 × 3 轮,6/6 配对全正);0.8B 门融合档 L1 逐位相同 | `tp_g8_gate`(融合档)✅ + std A/B | §76 |
| **词表切转正(P1.1 → 生产)** | ✅ **已完成(2026-10-20;§76.4)** | 生产档此前一直**全表头**(每 rank 2.54 GB bf16 / 1.27 GB i8)⇒ 开半表:decode **50.23 → 51.92(+3.4%)**;数值 **max\|Δ\|=0.000000** | `tp_vocab_split_gate`(27B 双卡)✅ + std A/B | §76 |
| **decode「15% 未归因空隙」(L2)** | ✅ **结案:幽灵(2026-10-20;§76.1)** | 设备 Σ **17.26 ms** = span 17.25(**核背靠背,图内零空隙**);ITL 与 ctx 无关(209→1249 全 18.6~18.7);真账 = 设备 + **1.4 ms**(launch 0.33 + emit 0.38 + HTTP/客户端 ~0.6) | node 模式账(**只信核内时长**)+ 干净客户端 ITL + `step_profile` | §76 |
| **AR 单次 18 µs 的剩余面(L3 余项)** | ⏸ **主动放弃(§76.3)** | 核 = 1 block × 512 thr,时长 min 16.4/p90 19.2(**无偏斜、无逐层差**)⇒ 固定链开销(4× sys fence + 2 远端读 + 1 远端写);可安全回收 ~1-2 µs/对(**+1%**),但动系统 fence 语义风险不成比例 | 已量到闭账 | §76 |
| **P1.5-b marlin → m=1 int4 GEMV** | ⚠️ **路线改写(§73)**:读 marlin 布局要付 **perm 查表税** ⇒ 不划算;唯一可行 = **双布局**(AWQ 原生给 decode) | 目标 +8.5%,代价 **+7 GB/卡 + 装载×2 + 双真相** | 需先裁决「8.5% vs 262k 弹性」;量 P1.5-a 落地后的 `live` | §73 |
| **slab 定尺根治(保守版)** | ✅ **已完成**(2026-10-19;§67) | 余量 `+12.5%+1MiB`→**`2×+2MiB`**(源头)+回退二次 ×2;融合/非融档均无固定 slab 启动;spec 融合 std **109.96** | — | §67 |
| P3.1 BIOS/MPS | ☐ 待人工 | | | |
| P3.2 NVLink 桥 | ☐ 待人工 | | | |

---

## 8. 风险与依赖

1. **别的 agent 会话**:兄弟 worktree(`server-engine`)随时起停 server,会抢卡 + 清杀进程(㉜)
   ⇒ 开工先查卡、用专属端口、读数标机器状态;若发现"server 无声消失",先怀疑它,别怀疑自己的改动。
2. **树里 19 项未提交的测试/工具面**:改动前确认没有别的会话在动同一批文件;
   `ln_elems_bench.rs`/`nsys_*`/探针 `.cu` 都是**未跟踪**文件,`git add -A` 会静默扫进去(仓史踩过)。
3. **数值面**:P1.1 数值中性(词表行独立);P1.3 有精度代价(★必须先立判据);
   P2.1 改变 token 序列(投机接受律)⇒ 需"与 greedy 一致性"门。
4. **权重字节型杠杆的前提**:减字节(decode)要么数值换(lm_head 量化)、要么换结构(P2.1 投机);
   单纯"优化内核"最多吃掉权重地板上方那 35% 的一半 —— 别指望它翻倍。
5. **文档面**:`docs/arch/TP2-施工设计.md` 是**入库面**(只留 §0–§9 + §10);
   逐轮账与作废结论一律进同目录的实录卷(《TP2-实录卷.md》,同样已入库 —— 两卷同仓同行)。

---

## 9. 待翻案清单(已封死的路里,**范围没写清的那些**)

> **为什么单列**:本项目的"封死"结论大多是**单轴 A/B**(换一个变量、其它全不动)在**一个形状 + 一台机器状态**
> 下量的。这种结论**没有错**,但它默认了"其它维度不动" —— 一旦换组合(形状/协议/链路/并发),
> 它就会**借尸还魂**。逐条附:①当时怎么封的(**判据 + 范围**)②草率点 ③翻案条件 ④便宜重测。

| # | 当时怎么封的(判据 + **范围**)| 草率点(为什么会借尸还魂)| 翻案条件 | 便宜重测 |
|---|---|---|---|---|
| 1 | **CE 换通道**(§32):"三段串行,净负";范围 = **CE + 两个分离核 + 跨流事件**那个形状 | 它否的是"CE 单独当搬运腿";vLLM 的形态是 **CE 搬载荷 + 紧邻一个融合核**(票写对端)—— **那是另一个形状**;§56.2 测到 D 臂 +1.2%(结构面),真值被同步面盖着(§56.5) | ① 上 D 结构(CE + 融合核)② 或 AR 与别的核**并发**(CE 不吃 SM 槽,§56.3) | `tools/ar-struct-probe.cu` 的 D 臂 + 生产图内 `ar_channel_bench` |
| 2 | **图内跨流藏写腿**(§41):"PCIe 写与本地访存完全串行";范围 = **重本地负载 + PCIe Gen3**(★ **§44 已自行翻过一次**:轻负载**全免费**) | 结论本是**负载相关**的,却被当成了通性;§42.3 自认 **NVLink 下这条未验**(机制是"PCIe 写独占 L2 未完成缓冲"⇒ NVLink 是另一条通路) | ① NVLink 装车 ② 轻本地负载形状 ③ 多会话并发(AR 与别的核交错) | `tools/p2p-overlap-probe.cu` 的 X/Y/V 臂(带负载扫描) |
| 3 | **读腿 97% 重叠不可移植**(§48/§49):"形状特有(长同质链)";范围 = **现役 27B decode 形状** | 形状一变(batch>1 / 投机 verify 轮 k token / 不同层混合)比率就变 —— 而 **P2.1 投机正要改形状** | 投机接线后重量;或按层/按 batch 扫形状 | 同探针换形状参数 |
| 4 | **TBO/微批**(§49.3/§68.0):"与 N 无关,零重叠 ⇒ 无地基";范围 = **当前引擎串行调度** + **核级链式形状**(2026-10-19 复核:`G→AR→G` 链 N=2/3/4 吞吐比 1.21–1.28×,**串行度 ≈ 1.000**)| 它封的是"**当前引擎**没有交错",不是"TBO 无效";引擎侧交错是**未建的能力**(探针 `OWL_PERF_MODE=concurrent` 就是为它留的) | ① 调度层支持前向交错 ② **投机轮天然多 token = 交错**(P2.1) | `tp_perf_split_probe` 的 concurrent 档 |
| 5 | **vLLM 的 D 结构照搬**(§57.3):封的是"**载荷由上游 SM 核多线程写** ⇒ 单线程 fence 背不住可见性" | **前提可消除**:载荷改由 **CE** 搬 ⇒ 流序背书可见性 ⇒ vLLM 的做法就成立(**与 #1 合流**) | 同 #1 | 同 #1 |
| 6 | **1-stage 融合核**(§57.4/§60.2):L1 间歇失败 **+ 从来未触发** | ①竞态**已修**(§60.1,门 5/5 绿);②**但 27B 生产档从未走到该路径**(§60.2 判据错位:标记在窄视图、判据在 Call 臂)⇒ §51 的 −4.7% **既不可靠也不存在** | ①**两核融合**(§10,不需要对称槽)②或修判据错位 | `tp_g8_gate`(L1)+ 生产 nsys 看核名 |
| 7 | **"decode 换结构 5.8%" → 修正 1.2%**(§56.4/56.5) | 修正的是"**结构面**的净差",不等于"把生产 20.58µs 瘦到 9µs"的收益 —— 后者上限 **11%**(§56.5),必须在**生产上**量 | 融合/瘦核落生产 | 生产图内 nsys(§1 第二把尺) |
| 8 | **"AR 全串行上限 1.64×"**(§39.4) | 它是**算术模型**:假设 AR 是**整段屏障**;**S5 "pipelined ring" 只设计了没施工**(施工设计 §1 S5 / §8 刀 C6)⇒ 若 AR 按 slab 渐进,上限抬到 ~**1.87×** | 施工 S5(**已列入 §3 P2.4**) | 先在 `ar_channel_bench` 上验"分块 AR 能否与算重叠" |
| 9 | **"VMM 读腿税 3.8–5.9%"**(§50.8) | **已被 §60 翻案一次**(同口径 13.9–20.7%,差 3 倍);两版量法口径未定谳;且 **READ vs RW 差 3–6 点** ⇒ 授予位的旋钮**没扫完**(alloc props / handle 类型 / 压缩位) | 若要保留跨卡读:扫授予位 | `tools/vmm-mapping-tax-probe.cu` |
| 10 | **"绕开 VMM 不值得"**(§60;**我自己的结论**) | 范围 = **现役写腿 + 10KB/5MB 两档**。一旦引入**跨卡读**形态(allgather / 1-stage 的 `peer_data` 读),那 14–24% 立刻相关 ⇒ 别读成"VMM 永远没问题" | 任何新的跨卡读路径 | 同探针(读腿档) |
| 11 | **"MPS 128→256 ⇒ +1.6%"** | **从未实测** —— 是从 128B TLP 天花板模型**估**的(与 Gen4 的 1.71× 同属推算) | 拿到 root | `sudo lspci -vv` 看 MPS + 复跑排名词(一行命令就能把"估"变成"测") |
| 12 | **"权重地板 79 t/s"**(§0) | 它假设"每 token 必须把每卡权重**读满一遍**" ⇒ **减字节**(lm_head 量化)**或**改结构(投机:一次权重读算 k 个 token)都会把地板本身移动 | 任何减字节 / 多 token 方案 | 由 P1.1 / P2.1 直接反映 |

### 9.1 由此得一条纪律(**㊱**)

**任何写进卷里的"封死/作废"结论,必须带三个范围字段:①形状 ②机器状态 ③组合(其它变量当时的值)。**
缺任何一项,它就只能当"当前配置下的读数",不能当判决 —— 否则后人(包括自己)会在换组合时,
把一条"范围外的旧读数"当成依据:要么错杀一条好路,要么盲信一条已经不成立的路。

> 卷里的三个现成反例(都在本表里):**§41 被写成通性 ⇒ §44 自己翻了自己**;
> **"VMM 读腿税 3.8–5.9%" ⇒ §60 翻了 3 倍**;**"1.64× 上限"是模型不是实测**(能把上限抬起来的 S5 从未施工)。
> ⇒ 本表就是这类"范围内读数"的收容所:**共 12 条;每条都带"翻案条件 + 便宜重测"**。


---

## 11. ★ 下一轮起点(会话交接面;2026-10-19 深夜)

> 目的:让下一次接手(人或压缩后的会话)**不必重新侦察**。所有坐标都是文件级/行号级,
> 判据与踩过的坑都写在这里。

### 11.1 代码与提交状态(2026-10-19 P1.1 轮后)

| 项 | 值 |
|---|---|
| worktree | `packages/owl-worktrees/w4a8-p1`(分支 `feat/w4a8-p1`)|
| 主干 | `main` = `c41b308`(**= 本分支头;前轮 P0.4/P0.5/P1.1-a/P2.1 已合主干**)|
| 工作区 | **本轮 P1.1 第二步 16 文件未提交**(`models`:module/linear/shard/model;`shared`:config;`engine`:loader/graph_fleet/running/engine/exec×4/`tests/tp_g8_gate`;`apps/server`:config/main)+ **新门 1 个未跟踪**(`tests/tp_vocab_split_gate.rs`)—— 已全门验证 + std 实测 + 27B 对拍 |
| ⚠️ 别的会话 | 本机**3090 对(ord 0/1)**被占(别的 agent 在跑 server)—— 双 3080(ord **2/3**)= 本线基准对 |

### 11.2 ~~下一刀~~ **已落:v1 两核融合**(把 stage+reduce 合成一个核;2026-10-19)

> **✅ 战果见实录卷 §65**:非投机档 **+3.8%**、0.8B 门 5/5 连跑绿、融合核
> `cu/comm/ar_fused.cu` 落地;**遗留 = 图捕获 slab 定尺要覆盖融合形态**(§65.3-2)。
> 以下为原始细案(存档)。

**为什么是这个**:§59.1 图内实测 —— **单核 11.13µs vs 两核 22.13µs**;而"协议钱"≈0
(砍 credit/rendezvous 无用)。1-stage(对称槽)那条路**判据错位走不通**(§60.2),
但"**少发一个核**"这个杠杆本身成立 ⇒ 把它落在 **v1 的真实形状**上。

**设计(不用对称槽,绕开 §60.2)**:

```text
入口 = eval.rs 的 AR 注入点(载荷 = 窄视图私有块,10KB;v1 两段路径)
        ↓ 现在是两核: ArStageCall(out→staging) → ArReduceCall(staging→out)
        ↓ 改为一核: ArFusedCall(out, own, peer, ...)
核内(单核两阶段;block 内先写后等再读):
  ① 本 block 群把 payload 拷进"数据槽"(与 stage 同布局:own 或 peer,按 v1/push)
  ② 全 block 到齐 → 发布 produced → 尾部 rendezvous(等对端 produced)
  ③ 读对端槽 → out[i] += peer[i](或按 push 档对调)
  ④ 发布 consumed + credit(与现 reduce 同律)
```
- **关键**:载荷仍是**私有块**(`out`),不依赖 `symmetric_out_slot` ⇒ 与 §60.2 的
  "判据在 Call 臂"无关 ✓
- **协议复用**:staging 布局 / header 四格 / credit / 两级等待**逐字沿用**(不改协议,
  只把两个核的体合并)⇒ 门(`tp_g8_gate` L1、`ar_transport_gate`、压力门)可直接复用;
- ⚠️ **核内两阶段 = 同一核里"先写后读"** ⇒ 必须沿用 §56.5 的**全线程 acquire**
  (生产已验证的那条),不要退回单线程 fence。

**收益账**:省一个核的启动(§59.1 的 11.13µs 是"单核落地"成本;融合核估 ~13–15µs)
⇒ ~7µs/次 × 128 次/步 = **0.9ms/步 ⇒ decode +4%**。
**验证**:生产 nsys 看"每步 AR 核数"(应从 2 降到 1:核名 `owl_comm_ar_fused_*`)。

**改动清单(照 1-stage 那次的经验,参数表是**五处**同改)**:
1. `crates/kernels/cu/comm/ar_reduce.cu`(或新 .cu):融合核 + 实例(f16/bf16/f32);
2. `crates/kernels/src/family/comm.rs`:常量 + `entry` 函数;
3. `crates/kernels/src/ops/comm.rs`:**KernelSpec + `METAS` 登记**(漏 = "无构建期 cubin 资产");
4. `crates/kernels/src/ops/mod.rs`:槽序锁表 `EXPECT`;
5. `crates/kernels/src/client/mod.rs`:两张清单(`cases`/`want`);
6. `crates/kernels/src/client/comm.rs`:新 `ArFusedCall`(字段/new/parse/to_launch);
7. `crates/kernels/src/server/comm.rs`:**发射面**(`launch_*` 的参数表 —— **本轮漏这里 = SIGSEGV**);
8. `crates/models/src/interpreters/eval.rs`:注入点改发一核(保留 v1 两段的开关以便 A/B);
9. `flavors/tp27b.toml` 的 `allowed_ops` +4;
10. 门:`tp_g8_gate` 连跑 ≥5 次 + `ar_transport_gate`。

### 11.3 本轮学到的四条(写下来防再踩)

1. **参数表是五处同改**:`KernelSpec::PARAMS` / `ops::EXPECT` / `client::PARSE_SLOTS`+`want` /
   `client 的 to_launch` / **`server 的 launch_*`** —— 漏任何一处不是编译错(前四处有锁),
   而是 **SIGSEGV(发射面参数错位)**;
2. **"核写好了"≠"路径被走到"**(§57 第二次):判据面(门)与生产**形态不同**时可以完全不触发;
   ⇒ 任何"性能中性/有收益"的结论,必须先用 **debug 日志/nsys 核名**证明**路径被走到**;
3. **判据点与接线点必须在同一节点**:`tp_role` 在**最外层视图**(`SliceView`),
   而"落对称槽"的决策在**更早的 `Op::Call`** ⇒ 两者差一层结构 = 永不命中(§60.2);
4. **`cargo build --bin server` 不含测试目标**:改了签名/字段后必须跑
   `cargo check --workspace --all-targets`(本轮漏跑 ⇒ 提交后才发现 `eval.rs` 两处测试档缺字段)。

### 11.4 复现/验证清单(命令级)

```bash
cd packages/owl-worktrees/w4a8-p1
# ① 全量编译(含测试目标!)+ 三门
cargo check --workspace --all-targets
cargo test -p owl-kernels --lib --all-features   # 92
cargo test -p owl-models  --lib                  # 217
cargo test -p owl-engine  --lib                  # 53
# ② 1-stage 档的 L1 门(连跑,验证竞态修复;应 5/5 全绿)
OWL_TP_TEST_DEV_A=2 OWL_TP_TEST_DEV_B=3 \
  OWL_0_8B_DIR=/home/div/Documents/codes/models/Qwen/Qwen3.5-0.8B \
  OWL_TP_1STAGE=1 cargo test -p owl-engine --test tp_g8_gate -- --nocapture
# ③ 生产 A/B(std 五点三轮;端口须避开别的会话 —— 纪律㉜)
PORT=8143 TP2_CFG=/tmp/bench8141/tp2-w4a8-arf8.toml OUT_DIR=/tmp/owl-bench-x \
  tools/tp-ab-std.sh tp2
# ④ 逐核账(两把尺;报告在 /tmp/diag/<tag>/)
PORT=8143 tools/nsys-decode-account.sh  <tag> tp27b <cfg>     # decode(图模式)
PORT=8143 tools/prefill-account.sh      <tag> tp27b <cfg>     # prefill(eager)
python3 tools/nsys_decode_account.py <tag> [...]
```
**基准读数(28B TP2,双 3080,std 五点三轮;供回归对照)**:
| 档 | prefill | decode | TTFT@8k |
|---|---|---|---|
| base(w4a16)| 1475.5 | **44.30** | ~5500ms |
| W4A8 | 2155.3 | 41.35 | 3820ms |
| W4A8 + fp8 AR(v1,合并后) | 2351.2(中位)| 41.67 | — |
| **W4A8 + fp8 AR(v1,fp8 核修后)** | **2441.4**(中位;均值 2343.6)| **41.72** | ~3450ms |
| W4A8 + push + fp8(修后)| 2418.6 | 41.80 | — |
| W4A8 + fp8 AR + 1-stage | 2316.8 | 41.36 | 1367ms(⚠️ 1-stage 实际未生效,见 §60.2)|

**新增仪器/判据(本轮)**:`ar_transport_gate` 的 **G1b f8 载荷腿**(精确 host 预言机 + 两 rank 逐位 + 尾组/奇数 n);
`OWL_AR_FP8_SC_OFF` 线格式宏(尺度区偶对齐,两 .cu 逐字一致)。

### 11.5 本轮追加(P0.4/P0.5;2026-10-19 深夜)

**已完成**:① P0.4 对账(merge main → 分支 = `20cd14a`,0 冲突;11 门全绿);② **P0.5 fp8 载荷路治理**
(prefill **+3.9%** 五点一致;三个真 bug:逐字节粒度 / 对称律从未成立 / 奇数 n 未对齐 + shuffle 发散;
新判据 = `ar_transport_gate` 的 **G1b f8 腿**)。详见实录卷 **§61**;std 数据 `docs/bench/data/owl-tp-2026-10-19-p04/`。

**下一刀不变**:**v1 两核融合**(§11.2 的细案)—— 但**有一条新情报**:
fp8 路的 stage/reduce 现在都是 16 B 粒度,而 §59.1 的"单核 11.13µs"是在 **f16** 档量的;
融合核必须**同时**保留 f8 的量化/反量化与 `OWL_AR_FP8_SC_OFF` 布局,并把 reduce 侧的
"读对端 → 解码 → 加"与 stage 侧的"量化 → 写"接在同一核内(**核内两阶段**,沿用 §56.5 的全线程 acquire)。

**备选(值更大)**:~~P1.1 lm_head 词表切~~(✅ 已落:非投机档 decode +7.9% /
−1.18 GiB/卡;**遗留 = 草稿选择面半表化**,见 §11.6);GDN prefill 调研(W4A8 里 19%)。

### 11.6 ★ P1.1 词表切已落(2026-10-19;本条为**未提交改动**的交接面)

**战绩**(std 五点三轮;双 3080;`tp2`;w4a16;非投机):decode **44.55 → 48.06**(+7.9%,
五点一致)/ prefill 持平 / **显存 live 11.42 → 10.24 G(每卡)**;27B 切 vs 不切
logits **逐位相同**(`tp_vocab_split_gate`)。

**改动面(16 文件 + 1 新门;全部未提交)**:

| 面 | 文件 | 要点 |
|---|---|---|
| 装载规则声明 | `models/src/module.rs`(Want.rule + `with_rule`)/ `layers/linear.rs`(layout 收口,仅真覆盖打标)/ `formats/shard.rs`(`want.rule` 优先) | 层声明 > 键表;`declared_rule_overrides_keytable` 单测 |
| tied 拒 | `models/src/model.rs::sharded` | `head_split && tied` ⇒ 拒(embed 表切不了) |
| 配置位 | `shared/src/config.rs`(`tp_vocab_split`)/ `apps/server/src/{config,main}.rs` / `engine/src/loader.rs`(`ModelLoader::vocab_split`) | 单卡给值 ⇒ 拒 |
| 槽宽/stride | `engine/src/graph_fleet.rs`(decode/verify 槽 + 两处 argmax stride) | 全用 `logits_vocab()` |
| 拼接面 | `engine/src/running.rs`(`read_full_logits` + `concat_head_rows`)/ `exec/{phases,spec,draft,prefill_chunk}.rs` | **唯一出口**;判据 = `head_split_active()`(事实,不读意图位) |
| 草稿档护栏 | `engine/src/engine.rs::boot_rank` | DFlash2/MTP × 切档 ⇒ **响亮拒**(见下) |
| 门 | `engine/tests/tp_vocab_split_gate.rs`(新,27B)/(`tp_g8_gate` 加 tied-拒档) | L1 结构 / L2 数值 / L3 功能 |

**遗留 1(★ 下一刀候选):草稿选择面半表化** —— 切档 × 草稿档现为**响亮拒**:
DFlash2(`dflash2.rs` propose 的 `embed.lm_head_matmul` + 码本 `[vocab, rank]`)、
MTP(`mtp.rs` 同构)的**选择面按全量 vocab 索引**,切档后候选塔到半词表
(实测 AL 2.4→1.4 ⇒ B4 降级 ⇒ **104→40 t/s**)。解:两 rank 各出 top-k
(DFlash2 的码本行加偏移)→ host 合并;估半天。

**遗留 2(存量账)**:`tp_load_gate`(0.8B CpuFace)失败 = **既有**(stash 复现同错):
`dtoh` 按 F32 读而块 dtype = F16;修法 = 门按 `LoadEntry.dtype` 读。另案。

**复现命令**:

```bash
# 性能 A/B(非投机;五点三轮)
PORT=8143 TP2_CFG=/tmp/bench8143/tp2-nospec.toml        OUT_DIR=/tmp/ab/base   tools/tp-ab-std.sh tp2
PORT=8143 TP2_CFG=/tmp/bench8143/tp2-nospec-vsplit.toml OUT_DIR=/tmp/ab/vsplit tools/tp-ab-std.sh tp2
# 切档正判据(27B;两臂对拍)
OWL_TP_TEST_DEV_A=2 OWL_TP_TEST_DEV_B=3 \
  OWL_27B_DIR=/home/div/Documents/codes/models/cyankiwi/Qwen3.8-27B-AWQ-INT4 \
  cargo test -p owl-engine --test tp_vocab_split_gate -- --nocapture --test-threads=1
# tied 拒判据(0.8B)
OWL_TP_TEST_DEV_A=2 OWL_TP_TEST_DEV_B=3 OWL_TP_VOCAB_SPLIT=1 \
  OWL_0_8B_DIR=/home/div/Documents/codes/models/Qwen/Qwen3.5-0.8B \
  cargo test -p owl-engine --test tp_g8_gate -- --nocapture --test-threads=1
```

**运行档**(本机):`/tmp/bench8143/tp2-nospec-vsplit.toml`(= 生产切档档)、
`tp2-spec-d7-vsplit.toml`(**应被拒启** —— 草稿档护栏)。

### 11.7 ★ 在飞的一刀:**草稿选择面半表化**(P1.1 遗留 1;2026-10-19 深夜立案)

**目标**:解锁 **spec 档 × 词表切**(现为 boot 响亮拒)。收益 = 显存 **−1.18 GiB/卡**
(生产档就是 spec 档 ⇒ 池 cap +20%)+ decode 预估 **+2~4%**
(每轮 lm_head 权重读两遍:verify 8 行 + propose 7 行 ≈ 6.6ms;verify 侧已省,
propose 侧待省 ~1.6ms)。

**方案(已定;零新核、零跨 rank 设备设施)**:

```text
propose 图（图内,每 rank 半表）:
  7 层草稿 → lm_head（半表）→ logits [7, half]
  → topk_of_logits（半表域;stride = 张量形状）→ cand/vals [7,16]
  → 图输出三槽:topk_vals / topk_ids（各 448B）+ sel_proj [7,256]（~3.5KB）
host（engine）:
  读两 rank 候选 → rank1 的 id += half → 按值归并取全局 top-16/行
  → H2D 回设备 → rank0 跑 select_from_candidates（全表域,码本**不切**）
```

**为什么对**:全局 top-16 ⊆ 两半各自 top-16 的并集(截断性质)⇒ 候选合并**无损**;
传输 896B/rank(对比"交换半表 logits"3.5MB);select 核与码本零改动。

**✅ 已完成(本轮;未提交,零行为变化)**:
- `models/src/layers/dflash2.rs`:selector 拆三件 —— `topk_of_logits`(stride 从
  `logits.shape()[1]` 取,唯一真源)/ `selector_proj` / `select_from_candidates`;
  `propose_block` 抽 `forward_hidden`(层循环不复制);新增 `propose_block_split`
  (出 cand/vals/proj,不出 drafts)。`select_with_logits` = 三件组合(**逐字节原路**)。
- 验证:models 编译 ✓;生产 spec 档 smoke(27B+DFlash2,不切档)**逐字一致** ✓。

**✅ 剩余步骤已全部完成(2026-10-19 同夜;见实录卷 §64)** —— 战绩:spec 档
**104.05 → 109.74(+5.5%,五点全正,0 降级)** / 显存 **−1.18 GiB/卡** /
两臂逐字一致。落点:图两形态(`plan_dflash`)+ 收割面(`dflash_select_merged`)
+ 合并纯函数 + `read_output_f32` BF16 分支 + 护栏收窄(DFlash2 图档放行)。
**残留**:MTP × 切档 / DFlash2 eager 降级档 ⇒ 仍**响亮拒**(§64.6)。

<details><summary>原始步骤清单(存档)</summary>

1. `graph_fleet::plan_dflash`:切档形态加三输出槽(`OutputSlot::f32("topk_vals", &[7*16])`
   / `topk_ids` / `OutputSlot{name:"sel_proj", shape:[7,RANK], dtype: d2g.dtype()}`)
   + 闭包按 `boot.model.logits_vocab() != boot.model.vocab_size()` 分派
   (`propose_block_split` vs 原路);
2. engine 图臂(`exec/draft.rs::propose_graph_step`);
   图输出槽读取 + **候选合并纯函数**(放 `running.rs`,可单测:
   `merge_topk_two_halves(a_vals,a_ids,b_vals,b_ids,half) -> (vals,ids)`)
   + rank0 `select_from_candidates`(eager:eval 核);
3. 判据:`spec_degrade` 计数 = 0;AL 与不切档档一致(±0);生成逐字一致
   (同 prompt greedy;与 `tp_vocab_split_gate` 同手法);
4. 生产 A/B:`tp2-spec-d7.toml` vs `tp2-spec-d7-vsplit.toml`(std 五点;
   基线 = 104.05),验收线 = 不退化 + 显存 −1.18G;
5. 完成后:移除 `engine.rs::boot_rank` 的"切档 × 草稿档"拒启(改为放行);
   门补 `tp_vocab_split_gate` 的 spec 档。

</details>

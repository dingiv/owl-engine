# 优化候选总表 —— 性能工作清单(prefill / decode)

> **来源与定位**:本卷是本地台账《优化候选总表-2026-10-14》(453 行,
> 「下一会话的直接入口」)与本地台账《perf-roadmap-2026-10-08》(2079 行,
> 逐轮补录与已关闭项)的**摘要收编**(2026-10-22 入库)。
> **本地全本仍是活台账**(不入 git,持续更新);本卷保留的是**判据与定谳**——
> 尤其是「已关死、禁止重挖」清单与测量纪律。新读数先落本地全本,定谳再回写本卷。
>
> 配套:仪器全录 = `perf-framework.md`;decode 结构账 = `decode-cycle.md`;
> prefill 图 = `prefill-graph.md`;换核全案 = `prefill-attention-swap.md`;
> 单卡会战读数 = `qwen38-single-3090.md`;10-14 结案 = `session-2026-10-14.md`。

---

## 0. 判读纪律(先看这三条,能省一轮返工)

1. **一律用 `epoch` 视图**(`/debug/perf/epoch…`):since-boot 视图被首核/捕获暖机
   outlier 独占;
2. **每 tag 样本 cap = 4096**:count 会**饱和**,判读看 `drop` 字段;
   per-sample 均值仍有效,总数/会话级计数无效;
3. **图态 vs eager 语义相反**:`srv.seg.dtoh` 只在**图态**是 GPU 时代理;
   `gpu_prof` 只在 **eager** 有意义(且每发 sync 税实测 **131 µs**,只作相对);
   `interp.host.*` 在**图回放期结构性为零**。

**设备时间判读铁律**(2026-10-14 血换来的,§4.0 的副产品):

- 设备时间只看 **nsys 窗口**或 **CUPTI `gpu.kern.*`**;
  `srv.op.*` = **下发耗时**,`srv.wait.*` = **尾视图**(含等 GPU)——都不是设备账;
- 测 prefill 一律 `prefill_graph=false`(生产默认)+ `step_profile=false`;
- 先数**内核实例数**(`owl_gdn_conv_fwd_f16` 144 / `owl_silu` 64 / `FI` 16 = 一个 chunk
  的指纹)再判时长,否则会把两个 chunk 当一个。

**工具/构建**:

```bash
# 宿主分相
cargo build --release -p server --features perf
# + CUPTI 图内逐核
cargo build --release -p server --features perf,cupti
./target/release/server --flavor spec-dflash <cfg> --log-level warn --perf-level gpu
# 统一口径测速(五点三轮)
llm_speedtest(外部存档): uv run cli --api-url <端点> --model <名> --flavor std
```

**口径警告**:① 测轮时/decode 必须用 `testdata/args/spec-27b-prod.toml`
(`step_profile` 关)——诊断档的 `probes.step_profile=true` 会在 spec 路径加
**~34 ms/轮(74.5 vs 40.6,−46%)**,其轮时数**不可引用**;② std decode **跨轮不可比**
——随机词池 AL 逐次 4.6~10 跳,必须比 AL/轮时或固定 prompt 集;
③ **flavor + 文件 = 完整启动参数**(files 不含 flavor):spec-* 要 `spec-dflash`,
std/decode 要 `awq27b`;配错 = 静默换引擎。

---

## 1. 基线快照(2026-10-14;27B AWQ / 单 3090 Ti / 生产档 spec-dflash)

| 口径 | 值 | 备注 |
|---|---:|---|
| prefill 吞吐 | ~1200 t/s | 生产档(fp8 池+FI);per-chunk 300 ms@512 → 413 ms@9010 |
| 裸 decode 步时 | **25.3 ms** | f16 池平坦;fp8 池另有 −27% 长 ctx 税(旧测) |
| spec 轮时 | 35.2 ms@512 → 43.9 ms@8k | `spec.round` |
| ├ verify | 27.2 → **31.5 ms/轮** | `spec.verify.read` |
| └ propose | 6.9 → 11.8 ms/轮 | >2048 后平坦(滑窗生效) |
| spec decode | 128 tok/19 轮 ≈ **155 t/s** | 7870 ctx;vs 裸档同 ctx ~38 t/s(=4×) |
| 解码图材料单 | **836 节点** | marlin 352 / glue 257 / GDN 144 / attn 32 / … |
| TTFT@512 | ~430 ms | prefill 610 tok ≈ 0.60 s |

**CUPTI 首战(8k ctx,图内逐核;两套仪器互证:累加 30.3 ms/轮 ≈ `spec.verify.read` 31.1 ms)**

| verify 图内 | ms/轮 | 占比 |
|---|---:|---:|
| marlin GEMM | **17.8** | **59%** |
| `kernel_mha`(XQA,469 µs/层 × 16 层) | **7.5** | **25%** |
| `owl_gdn_recurrence_varlen_gqa` | 2.0 | 7% |
| glue(farn/rmsnorm/conv/argmax/narrow/embed) | ~2.5 | 8% |

| draft 图内(graph 29) | 每次 | 量级 |
|---|---:|---:|
| `owl_naive_attn_nc_bf16_v2` | 1127.8 µs/层调用 | 5 层 = **5.6 ms/轮** |
| `owl_topk16_f16` | 1067.9 µs/次 | 每轮 1 次(✅ 已结案) |
| `owl_dflash_conv_bf16` | 442.7 µs | 5 层 |
| draft marlin(小形状) | 25.1 µs/次 | 800 次/两轮 |

---

## 2. 优先序(一行版)

- **decode**:D1(draft NC 核 5.6 ms/轮)> D3(XQA 长 ctx 税)> D4(glue/节点数)
  ≈ D5(采样链)> D6(批桶)。
- **prefill**:N1 ✅ 已落地(−85% 核时);N2/N3 判死/挂账;剩 M1/M2/M3(小肉)
  与 **M4 = W4A8(int8 张量核,−45%,立项级)**。
- **图 / 零回执 / padding 全部关闭**:真设备账 = marlin **84.7%**、其余核 9.8%、
  空转 5.5%(其中 4% 是每进程一次性);GPU 94.6% 忙、host 31% busy
  ⇒ **没有"被抹掉的墙"**。
- **墙的时钟判据**:负载下 SM 1965-2010 MHz ⇒ fp32-累加 fp16 墙 = **84.5-86.4 TFLOPS**;
  marlin 主发 ≈ **83 TFLOPS = 96-98% 墙**。

---

## 3. decode 候选

### D1 draft 注意力核(`owl_naive_attn_nc_bf16_v2`)—— 1.13 ms/层调用

- **证据**:CUPTI `gpu.graph.29.owl_naive_attn_nc_bf16_v2` n=80 → 1127.8 µs/次;
  5 层 ⇒ **5.6 ms/轮**,占 propose(~11 ms)一半。
- **已知头寸**(旧勘 `decode-cycle.md`):V 读合曲需 smem 转置分块(0.36 ms/层)
  + shuffle/softmax(≈0.5 ms/层);当年"收益递减/风险上升"暂止,现可复开。
- **预期**:1.13 → 0.7 ms/层 ⇒ **−2.2 ms/轮**(≈轮时 −5%)。成本中;
  必须过**恒等门**(AL 不降)与长 ctx 受控 A/B。
- ⚠️ **D1′ 草稿 NC 核 v3(行共享)已试过:单测 5.6×,但引擎里 AL 塌**
  (`[spec-degrade] m=0×6` → 降级裸 decode,t/s 63→26)。已排除 T 值/图捕获/gridDim.z/
  运行期 T 破坏展开;**核+表项+探针留树但不接线**(`driver.rs: let v3 = false;`)。
  续查入口 = dump 引擎真实张量指针/步长与探针对拍(怀疑引擎 q/k/v 为非稠密视图)。

### D2 `owl_topk16_f16` —— ✅ 已结案(2026-10-14)

定谳 = **grid=(rows,1,1) 只用 7/132 SM** 的结构性慢核(1067.9 → **45.7 µs/轮**)
⇒ 转正为 **split-K 两段**(`owl_topk16_split_f16` + `owl_topk16_merge_f16`,
`TOPK_NPARTS=16`)。三段真凶:①旧插入法**动态索引 → 局部内存**(ptxas 128B 栈帧,6.7×)
②每线程 1 个在飞载入 ③16 轮 smem argmax = **144 次 `__syncthreads`/块**。
**记住的坑**:**插入必须 (值降,索引升) 严格全序** —— 只比值 ⇒ 同值乱序 ⇒
锦标赛"winner 在 slot 0"破裂 ⇒ 同值重复选中(值全对、只索引错)。
探针 = `crates/kernels/tests/topk16_probe.rs`。

### D3 XQA 的 ctx 项(f8 读 + GQA 展开)—— 3.1 ms/轮 @8k

- **证据**:`kernel_mha` 469 µs/层 @8k vs 512 档 ≈275 µs/层 ⇒ +70%;轮内绝对头寸 3.1 ms。
- **注意**:**verify 大头是 GEMM(59%)不是 attention(25%)** ——
  旧案「8 伪序列单遍化」**已由 XQA 完成**,不要再按旧案动刀。
- **候选**:fp8 读路径向量化(`uint4` 载入 + 手写位拼装)、减 GQA 6× 重复反量化。
  预期 −1~2 ms/轮 @8k。成本中高(改核,须 fp8 数值门 + 恒等门)。

### D4 裸 decode 地板与 glue —— 节点 836,其中 257 是 norm/add/silu

- 候选:跨层融合(`add + 次层 rmsnorm` → `fused_add_rmsnorm` 已是既有先例)、
  glue 波次合并。预期 −0.5~1 ms/步(3-5%)。成本中(跨层接口重构)。

### D5 采样链 —— 1.7 ms/步 = 6.7%(图态)

- **证据**:图态轮时 25.37 vs `srv.seg.dtoh` 23.34 ⇒ ~1.7 ms/步 = 1 MB logits D2H
  + host 惩罚链/采样。
- 候选:设备侧 argmax/top-k(只回 4 B)+ 惩罚链下沉设备核。预期 −1.5 ms/步(≈+6%);
  **须过质量门**(greedy 逐位恒等可 A/B)。成本低-中。

### D6 并发批桶(B5.1)—— 3-5 天,std 聚合 ≥2×

- 立项已在本地台账《b5-concurrent-batch-立项》;B5.0 核级证可已完成;
  设计与裁决见 `cross-session-batch.md`。**2026-10-22 状态:施工中**
  (worktree `工作树 server-engine` / 分支 `feat/b5-batch`)。

### D7 B4 域自适应

- 判据已修,可再做"更好"而不是"更快降级";可选:域画像(AL 直方图)+ 预设档位。
  **AL 是域属性、与 ctx 正交**(长 ctx 塌 = 假案,勿再追)。

---

## 4. prefill 候选

### 4.0 归因定谳(2026-10-14 下半场;读本节前必读)

**方法**:worktree + `prefill_graph=false`(生产默认)+ **nsys 单 chunk 抓取**
(`ptok=461 max_tokens=1`)⇒ 逐核设备时间(窗口 2552 核 / 385.1 ms)。

| 项 | ms | 占比 |
|---|---:|---:|
| Marlin 主 GEMM(326 发,75-77 TFLOPS) | 290.3 | **75.4%** |
| Marlin 附发(64 行分块 **M 尾** + 小形状) | 22.0 | 5.7% |
| `owl_gdn_conv_fwd_f16`(grid 只 8/8/24 CTA) | 17.6 | 4.6% |
| 其余 61 个非 GEMM 核 | 20.7 | 5.4% |
| **GPU 空转**(9.65 FI 首调 + 4.43 cublas 首调) | **20.9** | **5.4%** |

**主机对照**:CUDA API 126.1 ms busy / 401 ms(31%)⇒ host 有余量,**GPU 是瓶颈(94.6% 忙)**。
**墙**:GEMM 总功 24.8 TFLOP / 290.3 ms = **76.5 TFLOPS = fp32 累加半率墙(80)的 96%**
⇒ GEMM 面无工程余量。

**老候选全线关闭**:图(S1/S2/S3/S4/S5)、零回执、块池/arena 都**不被设备账支持**;
"设备核只占 8%" = `srv.op.*`(下发耗时)误读;"encode 157 ms" = host 等 GPU。

### N1 ✅ 已落地(2026-10-14 晚)—— 实测核时 −85%,chunk wall −3.9%

- 落地面:`cu/gdn/gdn_conv_fwd.cu` 重写 + `driver.rs::gdn::conv_fwd(dt,d,tokens,batch)`
  (grid `(ceil(d/256), tiles, batch)`,`tiles=ceil(tokens/32)`)+ 新探针
  `crates/kernels/tests/gdn_conv_fwd_probe.rs`(4 关:逐位/串联/边界/延迟)。
- 实测(t=461,nsys 窗口 A/B):conv **17.51 → 2.00 ms(−88.6%)**,
  chunk wall **384.3 → 369.7 ms(−3.78%)**;引擎相位口径 `engine.phase.prefill`
  **359.8 → 346.9 ms(−3.59%)** ⇒ 吞吐 **1281 → 1329 t/s(+3.7%)**。
- 引擎级恒等门:4 类固定题 × 96 tok greedy **逐字一致(4/4)**。

### N2 ❌ 判死(代码级证明)

- `marlin_template.h:372` `parallel = prob_m / m_block_size`(**整除截断**)⇒ 内核只吃整块 M,
  余数必须调用侧另发。补到 16(461→464)后 `464/64=7 余 16` ⇒ 仍是两发;
  要单发得补到 64 的倍数(461→512 = **多算 11% ≈ +32 ms**)。
- 尾发本身**已近带宽极限**:13 行 = 重流权重 44.6 MB / 64 µs ≈ **697 GB/s**(带宽 74%)。
- 账:尾发 22.0 ms 里合法计算 8.2 ms,纯损失 6.5 ms(1.7%),而修法 ≥+32 ms ⇒ **净亏**。

### N3 ❌ 降级为挂账(两个 stall 是每进程一次性)

- 取证(同 prompt 两请求):首个 FI attention 前 host 静默 9.51 ms → 0.00 ms;
  `cuKernelGetFunction`(cublas prefill 形状)4.48 ms 只现一次 ⇒ 稳态零收益。
- 只值"进程内第一个请求 TTFT −13.6 ms";修法 = boot 后跑一发 prefill 形状 dummy chunk
  (~+150 ms boot)⇒ **挂账**。真因:boot warmup 只覆盖 verify/decode 形状。

### N4 / M4 **W4A8(激活 int8 × 权重 int4,int8 张量核)** —— 立项级

- **判据**:GA102 int8 张量核 = **160 TOPS = 2× fp16 半率墙**;现 GEMM 71% 的墙内时间
  是纯 fp32-累加 fp16 张量核,**换算术才动得了**。
- 代价:激活在线量化核 + per-token scale + 精度门(长 ctx 累积)⇒ 立项 + 金标 + 抽检。
- 预期:prefill 385 → ~200-230 ms;1297 → **~2200-2500 t/s(过 2178 及格线)**。
- **2026-10-22 状态:✅ 已交付** —— 算子家族与接线见 `w4a8-family.md`;
  TP2 上 prefill **+47%**(2026-10-22 TP2 快档实测,数据本地原始数据(未入库));单独成档(非默认,
  因 decode 回退)。

### M1–M3 小肉(2026-10-19 账)

- M1 `gdn_norm_act`(warp-per-unit,估 −2 ms/块):**✅ 已落**(`f1f4c6b`);
- M2 `owl_concat_rows`(48 GB/s 跨步读,估 ~0.3%):**挂账**(收益小);
- M3 FLA(8 ms):见本地台账《26-10-19-prefill-8192-头寸分析》。

### P1 ❌ 作废(归因定谳;下列数字是 `srv.op.*` 下发耗时,非设备时间)

实测(510 tok/1 chunk,`--perf-level host`):`engine.phase.prefill` 404.8 ms,
其中 `srv.wait.htod`(62 次,avg 3.5 ms)= **270.0 ms(67%)**、`srv.wait.launch`
47.2 ms(12%)、alloc/free/memset 65 ms(16%)、**设备核 `srv.op.*` 全部 30.7 ms(7.6%)**。
同一个 htod:**issue 侧 `srv.seg.htod` 仅 16 µs**(220× 差)⇒ **ack 交付延迟**。
全节保留为考古,**不要再施工**。

### ✅ 前置件已合入(2026-10-14 双 lane):同步面定向化 + 块池(opt-in)

- **同步面**:逐 free 跨流栅栏 → 按块定向;dtoh 整流 sync → 事件等待。
  实测 `srv.wait.htod` **p50 4.6 ms → 70 µs**、栅栏 1434→5 道/请求;**wall 中性**。
- **块池**(`OWL_BLOCK_POOL=<bytes>`,默认关):best-fit 块复用;27B 实测 alloc 宿主 −94%、
  真回收 −51%、命中率 53.7%(0.8B 91.9%);**wall −1~2%**。
  集成雷已修:池命中必须计入 `slab_hint` 计量(否则捕获 slab 偏小 ⇒ 27B 开池 boot 失败)。
- 两条都证明:单点拆掉不撼动 wall ⇒ 自变量是**逐命令回执链** ⇒ 下一刀 = P1‴。

### P1‴ ❌ 作废(图真回放后墙时纹丝不动:384/380/390 vs eager 382/380/402 ms)

### P1″ ❌ 作废(arena 实测墙时零变化;`OWL_NO_RECLAIM` 的 5× 是 OOM 伪影)

### P2 ❌ 判死(prefill attention)

真值 = FI 16 发 **2.2 ms / 385 ms = 0.6%** ⇒ **FI 归零也只值 <1%**。

### P2′ ✅ 已落地:换上游 FlashInfer 0.6.18(1.4×);XQA 路线否决

- CUPTI 逐核同请求 A/B(3038 tok/6 chunk):XQA `kernel_mha` **258.5 ms** vs
  FI `BatchPrefill` **63.0 ms** ⇒ **FI 快 4.1×**。根因:XQA 是 **decode 形状**核
  (q=T 只 ~9 TFLOPS),FI(FA2 瓦片)40+。同一个 `kernel_mha`,**q=8 是 5.4× 的赢家、
  q=512 是 4× 的输家**。
- ✅ 顺手:把臂挪到 FI 之后只接 `t<128` 尾块:3126 tok 实测 **13.6 vs 127.6 ms = 9.4×**,
  prefill 相位 −75 ms(生产档缺省 `xqa_prefill=false`)。
- ✅ 落地:FI 从 guoqingbao fork **换成上游 nightly-0.6.18**(`fdc56343`),ABI 未变 ⇒
  Rust 侧零改动;单算子(T=512,fp8 池)旧 fork 的 **1.36–1.41×**;CUPTI 引擎内注意力
  **49.6 vs 63.0 ms(−21%)**;隐层校验和相对差 **2.8e-5**。
- **别再挖**:① 拿 XQA 打 prefill 主路径;② 拿 `chunked knhd` 当"生产基线"做选型;
  ③ 用 nsys 作 FI 侧判据(本机对 FI 档稳定丢 2/3 核;**CUTPI in-engine 才是可靠尺子**)。

### P2″ ✅ 已结案(FI 族收口)

- ✅ `dispatch.fi_min_tokens` 128 → 1:FI 在 T=1/8/54/128 实测 0.287–0.438 ms/层,
  而回落核同形状 **6.7 ms/层** ⇒ **FI 快 15×**;引擎 A/B −73.8 ms@3.1k(−3.0%)、
  −272.2 ms@8.2k(−4.1%)。
- ❌ FI 的 split-kv 旋钮关死:该 fork 的分块 merge 本就是坏的
  (`FI_SPLIT_KV=1` + 单测:T=8 数值 max|FI−host| = **1.39**,门 1e-2)⇒ 写死是对的。
- ⚠️ **FI 战线结案**:FI 本体 33 TFLOPS(38% 墙)已是最好;prefill 注意力只占
  prefill 相位 **2.6%**,到此为止。

### P3 host 命令管线(29 ms/chunk)—— 已判"收益不明显",封档

- 每命令 `tx 1.84 + queue 6.67 + op 10.86 + 回执/poll 1.44 = 20.8 µs`;
  占 8192 档仅 ~5%、TTFT@512 ~6%。
- **复开条件**:目标变成 TTFT 极致,或出现"小 chunk + 多层"新负载。
  届时顺序:①入向唤醒 6.67 µs ②服务端记账 ~6.3 µs ③合批(税 ÷ N)。
  **免回执收益仅 1.4 µs,勿再当候选。**

### P4 piecewise prefill graph —— ❌ 关闭(S1/S1.5 已证无收益)

抹掉 29 ms/chunk 的 host 税,但换成设备侧节点派发 4-15 ms/chunk;
需形状分桶 × 输入槽化 × slab 预算。**与 P3 是替代关系,不要都做。**

---

## 5. 已关死(禁止重挖)

| 项 | 定谳 |
|---|---|
| **AL 在长 ctx 塌** | **假案** —— 内容方差(两轮受控扫描 AL 非单调,长 ctx 点反而最高) |
| verify "8 伪序列单遍化" | **已由 XQA verify 完成**;verify 大头是 GEMM 不是 attention |
| prefill 宿主链三刀(P1/P1′/P1‴/P1″) | 已定谳并**封档** |
| 免回执 | 收益仅 1.4 µs,降级 |
| 多块链式起草 | **架构性不可行**:草稿 memory 源 = verify taps,推理视界恒滞后目标一个 block |
| 深度 >7 | min(7) 架构钳(block=8 检查点家族定形),depth10 实测 AL/吞吐逐位不变 |
| verify 骨干压缩 | GEMM 19.7 ms 在 83-93% 峰值带宽;T=8 与 T=1 同价 |
| marlin 配置调优 | 无增益(NO_M8/NO_OVERRIDE 扫参;override 反而在帮) |
| schedule/emit 修剪 | schedule 实测 0.00 ms、emit 0.2 ms —— 免费 |
| 装载线 ct_repack | 124 ms 属装载期,非轮内(收编候选 = s3-load 式) |
| **prefill per-chunk +36%** | **不可复现**(同档同口径 411 vs 395 ms/chunk,持平)⇒ 命题撤销 |
| **P2 prefill 注意力读核** | **判死**:FI 16 发 = 2.2 ms / 385 ms;归零也只值 0.6% |
| **P2′ XQA 换 prefill** | **主路径实测否决**(FI 快 4.1×);已改造成 FI 缺口交接臂 |
| **prefill "逐命令回执链"** | **假案**:GPU 94.6% 忙 / host 31% busy;图真回放 0% 收益 |
| **prefill 图 / 断点 / padding / 草稿面图** | **关闭**(S1-S5):无墙可抹;剩余价值 = 捕获可行性证明 |
| **`[dflash] prefill encode` = 157 ms** | **误读** —— host 等 GPU;真设备时间 **5.2 ms** |
| fp8 池 dtype | **产品选择不换**(只能在核内消化反量化税) |
| 历史回退假象("45→31.8") | 口径像差;ITL 23.9 vs 差分 24.8 |

---

## 6. 挂账 bug / 卫生项(与性能无关但须在册)

| # | 项 | 说明 |
|---|---|---|
| H1 | kernels 13 处打印未收编 | lib 零依赖律禁它依赖 `owl-shared`;正解 = **注入 `LogSink`**(CUPTI shim 同款先例) |
| H2 | logger 收编批②③ | engine 探针 ≈75 / models ≈165 |
| H3 | `gpu_prof` 每发 sync 税 **131 µs** | 与文档口径(~10 µs)差 13×;只作同核相对对比 |
| H4 | CUPTI 必须用 `CONCURRENT_KERNEL`(kind 10) | legacy `KERNEL`(3)对图内节点不报时间戳 |
| H5 | `spec + no_graph` 已改为装配期拒启 | 原为首轮 panic(引擎线程退役) |
| H6 | metrics cap 4096 计数饱和 | 判读看 `drop` |
| H7 | 会话压缩前未提交面 | engine(B4+拒启)+ cuda(sink/CUPTI)+ shared(LogSink) |

---

## 7. 时间线与现状对照

| 日期 | 事件 |
|---|---|
| 2026-10-08 | 性能会战收口:本文源头 `perf-roadmap` 立案(逐轮补录 + 已关闭项) |
| 2026-10-10 | B4 自适应降级 / B6 fp8 KV 全家桶 / B6.3 双 bug 破案 / B6.5 显存硬预算 |
| 2026-10-11 | FLA fork-bf16 全家桶转正(g 语义律)+ lm_head int4 结案 |
| 2026-10-12 | 算子契约权威 M1-M5 一口气完工 |
| 2026-10-14 | 性能框架 P1-P4 + logger 第一批;**prefill 归因定谳**;N1/D2/P2′/P2″ 落地;本文源头立案 |
| 2026-10-17 | perf-review 合并:a8 装载 −12.3 s / 算子预加载 / 投机盈亏线;**32k 墙头号事实 = GPU 只忙 51.5%** |
| 2026-10-19 | TP2 AR 通道提速(prefill 0.244× → 1.396×);prefill 8192 头寸分析 |
| 2026-10-20/21 | B5.1 并发批立项与施工;prefill 桶转正后**按裁决退役**(Q 槽阻断);MTP Windowed |
| 2026-10-22 | TP2 decode 快档(FI 单图 +21.7%);DFlash2 草稿池 ring;W4A8 定档 |

**截至 2026-10-22 仍开着的门**(本地全本为准):

1. **D1 draft NC 核**(v2 1.13 ms/层;v3 行共享单测 5.6× 但 AL 塌,未接线);
2. **D3 XQA 长 ctx 税**、**D4 glue**、**D5 采样链**(均未动);
3. **D6 B5.1 并发批**——施工中;
4. **M2 `concat_rows`**、**M3 FLA**(小肉);
5. **32k 墙的非 GPU 空窗 48.5%** —— 2026-10-17 定为首要下一刀
   (与 decode 的 2.5× 结构空档同族;见 `decode-cycle.md`);
6. **int8-QK prefill**:判决暂不做(天花板 +4.1%),卷 =
   本地台账《26-10-17-int8-QK-prefill-调研》(触发条件已写)。

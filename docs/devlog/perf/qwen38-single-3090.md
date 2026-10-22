
## 补录Ⅴ:B7 server spec 衰减复测(2026-10-10,结案)

协议:server 单 3090,27B AWQ+DFlash2,池 4352,spec 强制开
(DEGRADE_AFTER=0);speedtest --lengths 钉长 ×2 轮 ×256 out;
metrics counter 差分(闭合模型逐桶校验)。随机词池域。

| ctx | 轮时 ms | verify | propose | AL | 模型 tok/s | 实测 |
|---|---:|---:|---:|---:|---:|---:|
| 512 (slots8) | 52.8 | 40.9 | 10.6 | 0.14 | 21.6 | 23.1 |
| 1024 (slots8) | 67.8 | 53.2 | 13.3 | 0.16 | 17.2 | 17.9 |
| 2048 (slots8) | 91.2 | 72.2 | 17.7 | 0.47 | 16.1 | 15.5 |
| 2048 (slots2) | 96.7 | 76.9 | 18.4 | 0.17 | 12.1 | 12.7 |
| 4096 (slots2)* | 152.9 | 122.8 | 28.6 | 0.78 | 11.6 | 11.8 |

*4096 桶需 PREFIX_CACHE=0 + MAX_SEQ=4352(前缀缓存占 68 页 vs 单会话
需 136 页,块级缓存无压力驱逐 —— "E2c 驱逐另接"挂账仍在)。

**判决:有衰减,结构成本,非 bug,销案。**
- 衰减全部在轮时(verify 512→4096:+200%,KV 读 T=8 行 × ctx 线性项;
  propose 同步涨 = 草稿读窗全历史),**AL 不随 ctx 衰减**(0.14-0.78
  噪声级;域归属 = 接受率,ctx 归属 = 轮时,正交 —— 补录Ⅳ同源)。
- 模型/实测全程闭合(±6%)—— 无隐藏税。
- GDN slot 数无关轮时(72.2 vs 76.9 @2048,噪声级)。
- 运营结论:随机域 spec 任何 ctx 都净亏(512 已 21.6 vs 底线 40)→
  B4 降级是唯一正解且已落地;健康域(math AL≈4)衰减不翻盘
  (轮时 +100ms/轮 vs 5 tok/轮收益,4096 仍 ~2.2× 于裸 decode)。
- 协议勘误:speedtest math 域 = 随机出题(AL 0.25,非健康推理域),
  且 len 参数失真(请求 2048 实发 7892)—— 健康域曲线须用固定
  数学题(gate 式),不能依赖 speedtest 域标签。

## 补录Ⅵ:C1 lm_head int4 当日结案(2026-10-11,server 差分协议)

spec-dflash flavor(slots1+fp8+d7+syvai 草稿),3090Ti 单卡,greedy,
`model.head_quant=true` = 装载期 RTN int4 g32 sym(检查点不动,AwqSource
F16 兜底臂;marlin kU4)。全案卷 = 本地台账《perf-roadmap-2026-10-08》
2026-10-11 会话补录。

| 域 | f16 头 | int4 头 | Δ |
|---|---:|---:|---:|
| 多步数学 | 159.06 | **177.35** | +11.5%(**超冲刺线 165**) |
| 竖式乘法 | 102.33 | 117.81 | +15.1% |
| prose 长城 | 45.15 | 54.24 | +20.1% |
| random 词池 | 47.17 | 52.28 | +10.8% |

质量:prose/数学 A/B 逐字一致;竖式同错(模型固有,47×893=42149 两边
一致);random 域贪心分叉皆连贯。显存 live −1.7G(19.2→17.5G),自动池
+34k tok;装载 +4.4s。fast-variant GPTQ 移植血统红灯(幅值差 25×,
不同源);GPTQ 升级 = 自采 hidden states 另案。server 税终审 ~1%(159.06
vs 引擎 157.5)销案。

## 补录Ⅶ:M4+多轮模板后 std 全档复测 + 裸档 A/B + 时钟假象立案(2026-10-12 续)

> 协议:llm_speedtest std 五点三轮(512/1024/2048/4096/8192 × 3,out 256,单并发,
> 随机词池);owl spec-dflash 生产档(27B AWQ + int4 头 + DFlash2 d7 + fp8 池 +
> gdn_chunked),**池放宽 4352→10240、max_seq 2560→9216**(为覆盖 4096/8192 两点,
> 本轮首次全档 15/15;上轮 M4 验收仅 9/15)。3090 Ti 单卡(CUDA ordinal 1)。
> 裸档 A/B = 同配置仅 spec.depth=0。数据 `data/qwen38-3090单卡/owl-{specdflash,bare}-std-2026.{csv,json}`。

### 生产档(spec-dflash)15/15 全绿

| ctx | prefill t/s(三轮) | decode t/s(三轮) | TTFT ms |
|---:|---|---|---:|
| 512 | 1174/1191/1190(avg 1185) | 189/61.9/86.8 | 432-438 |
| 1024 | 1112/1144/1136(avg 1131) | 52.3/63.4/55.1 | 897-923 |
| 2048 | 1049/1038/1049(avg 1045) | 55.6/51.0/67.6 | 1955-1975 |
| 4096 | 914/901/907(avg 907) | 36.0/33.6/38.7 | 4483-4547 |
| 8192 | 727/720/717(avg 721) | 21.8/27.6/20.3 | 11275-11426 |

- **512-2048 窗口 avg prefill 1120 vs M4 验收 1153.6(−3%,同带零回归)**;
  decode 窗口 avg 75.9(spec 域归属正常波动);TTFT@512 与 M4 验收一致(427→432ms)。
- 多轮 chat 模板修复(openai.rs)后首测,冒烟+15 点全连贯,质量面无回退。

### 发现 1·终审(同日补):长 ctx prefill 衰减 = FI 未开,非新病(历史修复实证仍成立)

用户追问"此题上轮已解,为何复现"——文档考古 + 同日 FI-on A/B 定谳:

- **上轮修复在案**(结构检视 §三十·补二,2026-10-12):衰减真凶 = 自研
  split/fp8 attention 核(~118ms/chunk 随 ctx 线性涨);修法 = **FI 门槛通道**,
  run-8k.toml(flashinfer=true)std 15/15 平坦(8192 仅 −7%,TTFT@8192
  10.96→6.64s),**行动项"flashinfer=true 转正生产默认"已写但从未落地**;
- **未落地的原因**(perf-roadmap #300/#380/#637):**FI 影子池 decode 缺写**
  架构项 —— FI 模式 decode 段融合插池核只写 classic 池,多 turn 跨 turn
  质量隐患(单 turn 不可见)→ FI 转正被锁,前置 = 融合插池核写影子变体
  (owl_qknorm_rope_kv_insert 增 kNHD 影子输出,解锁 prefill +27%)。
  spec-dflash flavor 因此不含 flashinfer=true;
- **今日数字非回退**:FI-off 生产档 997.9 avg / 721@8k = 文档"生产实走
  chunked(976 avg/700@8k)"的忠实复现(±2%);
- **同日实证(本机,FI-on 全同配置)**:8192 → **1234/1237 t/s,TTFT
  6.62-6.64s**(与历史 run-8k 的 1238/6.64s 逐位吻合);512→8192 曲线平坦
  (1158/1285/1234 与 1327/1300/1237)。历史修复在现役代码上完好;
- **决策挂账**:①融合插池核写影子变体 → FI 生产转正(根治);②bench/
  单 turn 场景可先行 opt-in FI(无跨 turn 风险);③131k 档 FI 双影子池
  超预算(8.4G>5.9G),超长 ctx 维持 chunked(结构衰减已知)。

### 发现 1:长 ctx prefill 衰减与 spec 无关(裸档 A/B 定谳)

裸档 TTFT@8192 = 10948/10968ms ≈ spec 档 11275-11426ms(+3%,草稿 encode 税可忽略)
→ **8192 点 prefill 721 t/s 的衰减是主模型 prefill 路径自身**(512→8192 逐点
1185→721,−39% 单调下行)。对照 10-04 裸档(FI+gdn_scalar,f16 KV)TTFT@8192
6511ms → 现在 10948ms(+68%)。注意**非同口径回退**:现役 = FI 关(影子池 decode
缺写锁死)+ gdn_chunked + fp8 KV。归因候选:①FI prefill 面缺席;②gdn_chunked
prefill 臂长 ctx 行为;③fp8 KV 写路径。**长 ctx prefill = 当前对 vLLM 的最大差距面**
(8192 点 owl 721 vs vLLM ~1450 级,短 ctx 仅差 20%),立为下一杠杆候选。

### 发现 2:GPU 空闲掉钟测量假象(裸档短 ctx 假病,立案销案)

- 现象:裸档 512/2048 行 decode 3.5-20 t/s(病值 75ms/50ms = 25ms 的整数倍),
  4096/8192 行正常(26-28ms);手动流式复测同点全速 24ms 平坦。
- 定谳:**persistence mode = Disabled,空闲 SM 210MHz(满 2100)**。请求间隙掉钟,
  早 EOS 短生成轮(50-75 tok,窗口 1.2-1.8s)在时钟爬坡完成前结束 → 均值被砸;
  长窗口轮(7s+)与 spec 轮(242-309 tok)摊薄爬坡 → 显得健康。连打 3 发:
  3.49 → 40.37 → 40.60 t/s,第一发吃满冷钟。10-04"SM 时钟 ramp"病史同源。
- **测量纪律增补:测速前暖机(连打≥1 发弃样),或 sudo nvidia-smi -pm 1**
  (本机 sudo 需密码,未开)。裸档 512/2048 行作废,4096/8192 行有效。

### 发现 3:流式 SSE 无 usage 尾帧(测速口径缺口,挂账)

owl 流式响应不实现 `stream_options.include_usage` 尾帧(非流式 usage 正确),
speedtest 退化 `token_source: Local Estimation`(字符估算):样本 309 est vs
257 chunk 实际(+20% 漂移,依输出风格浮动)。**现役所有 std decode t/s =
估算口径**,与 usage 口径存在 ±20% 级系统性偏差(AGENTS 测量纪律 #1 被绕过)。
修复 = openai.rs 流式末尾补 usage 帧(挂账,半小时级);修后历史数据不可直接
与 usage 口径横向比。

### 附:spec 节奏白送样本(token_timestamps)

verify 批突发 ~8 tok/2.3ms + 37-39ms 轮间隙 → AL 高时引擎率 ~194 t/s@512;
max_tokens 上限执行正确(255/256 chunk)。

---

## 2026-10-21 补录:AWQ 档 vs **GGUF 档**同机对照(投机 d7,std 256tok,`--ignore-eos`)

**背景**:GGUF 线(27B UD-IQ4_XS)在 8192 只有 33.5 t/s,而本卷补录Ⅲ 的 AWQ 档
(同引擎)是 76.08 —— 先把**基准尺**在本机重建出来(intent 档
`profiles/awq27b-3090-dflash.toml`,AWQ-INT4 + syvai W4A16 draft + fp8 + FI 三面),
再逐项对账。

### A. 基准(本机重测,2026-10-21)

| 点 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---:|---:|---:|---:|---:|
| decode t/s(三轮) | 134.3 / 107.8 / 178.6 | 118.7 / 117.2 / 107.4 | 102.9 / 104.9 / 140.6 | 103.5 / 109.1 / 67.6 | **114.1 / 105.5 / 65.9** |
| prefill t/s | ~1380 | ~1410 | ~1405 | ~1398 | ~1372 |

⇒ 8192 **中位 ≈ 105**(优于 10-13 补录的 76.08 —— 其间的轮内优化都算进去了);
**全程零降级**。

### B. 同点收口(8192,`probes.step_profile=true`,同一发请求)

| 项 | **AWQ 档** | **GGUF 档** | 比 |
|---|---:|---:|---:|
| verify pure(图 GPU) | **23.7 ms** | **43.9 ms** | **1.85×** |
| propose pure_gpu | 6.26 ms | 9.1 ms | 1.45× |
| rollback+snap | ≈2.3 ms | ≈2 | — |
| **round** | **≈32.5 ms** | **≈55 ms** | 1.7× |
| 裸步 | 28.2 ms(fp8 税后) | 29.4 ms | 1.04× |
| **窗口 AL** | **≈3.6**(反推 32.5/8.8) | **≈1.84** | **2.0×** |
| decode | ~105 t/s | 33.5 t/s | **3.1×** |

### C. 两个等量级的根因

1. **verify 的 m=8 量化 GEMM**:两者都读 ~14.24 GB 权重 ——
   Marlin(W4A16)23.7 ms ⇒ **600 GB/s**;GGUF MMQ2 43.9 ms ⇒ **324 GB/s**。
   差在**并行度**:`gx = n/128` ⇒ n=5120/6144/10240 只有 40/48/80 个 CTA(84 SM)。
   ⇒ 对应工作项 **MMQ2 stream-k**(k 向切分;路线图 §9.8)。
2. **AL 差一倍(1.84 vs 3.6)**,且已排除:
   - 草稿量化:GGUF target 换 HF bf16 草稿 vs GGUF Q4_K_M 草稿 ⇒ 同速(31.0 vs 32.2);
   - verify 臂:v2 / XQA / FI 三臂 AL 都是 1.67~1.84;
   - fp8 池:GGUF 用 f16 池,AL 不变。
   ⇒ 嫌疑集中在 **GGUF 目标侧的 draft 输入链(taps)**:待查(见 §D)。

### D. 下一步(按 ROI)

1. **AL 定谳**:固定 prompt 长生成,打 AL-vs-position 曲线;并把 GGUF 的 5 个 tap
   与 AWQ 的逐层 hidden 做数值对拍(若 tap 层级错位,AL 会系统性腰斩而非归零 ——
   与 1.84 的现象吻合)。
2. **MMQ2 stream-k**:verify 43.9 → ≤25 ms(对齐 Marlin 的 600 GB/s)⇒ round ≈35,
   8192 由 33.5 → **≈52 t/s**(1.57×)。
3. FI `fixed_split_size`(split 在 graph 档被 planner 强制 1):8k 无收益(verify 里
   FI 只占 0.69 ms/轮),**≥16k 才值 1.6×**。

### E. AL 定谳(直读 `[spec-prof] m=`,不靠反推)

同一点(pos≈8240,depth 7):**AWQ 档 mean m = 2.85 / GGUF 档 mean m = 1.33**(2.14×)。
GGUF 档的 AL **随位置塌陷**,拐点正好在草稿滑窗(2048)开始截断处:

| GGUF pos | ~512 | ~2048 | ~8240 |
|---|---:|---:|---:|
| mean m(接受/轮) | **4.75** | **4.31** | **1.33** |

⇒ 短 ctx 我们的草稿**比 AWQ 档还好**(4.75 vs ~4.5),只有长 ctx 崩 —— 这不是
"草稿质量差",而是**长 ctx 的 draft 输入链**问题(待查:taps 数值 / 草稿池页表
在 `prefix_len > 2048` 后的窗口读)。

### F. stream-k 的**计时探针**(gx=84 k 切分,结果故意无效)

把 `mmq2_geom` 的 `gx` 硬拉到 84(SM 数)、`tmp_fixup` 指到权重缓冲(只为过空指针
检查;结果作废),同发 8k 请求:

| 项 | gx=ntiles(现役) | gx=84(k 切分) | 收益 |
|---|---:|---:|---:|
| verify pure | 43.9 ms | **36.0 ms** | **1.22×** |
| propose pure_gpu | 9.1 ms | 7.4 ms | 1.23× |

⇒ 比按 CTA 数线性外推的 1.75× 低(40→84 CTA 时**单 CTA 吞吐会掉**:40 CTA 时
11.5 GB/s/CTA,136 CTA 时只有 4.2)。**stream-k 的真收益 ≈ 1.2×**(round 55→45,
8192 由 33.5→~40),实现成本(固定工作区 + fixup 核改 half + 第二发)不小。
⇒ **优先级下调**:AL 的 2.14× 才是主战場。

### G. AL 塌陷的三条隔离试验(2026-10-21 深夜)

| 试验 | 改动 | mean_m@512 | mean_m@8k | 判定 |
|---|---|---:|---:|---|
| 基线 | 窗口 2048 / rot=128 | 4.75 | **1.33** | — |
| **窗口关** | `OWL_NC_SLIDING_WINDOW = 1e9` | 4.75 | **0.00** | **窗口是必需件**:草稿在 >2048 的全 ctx 上完全废掉(训练分布);propose 也从 9.1 → 19.0 ms |
| **rot=64** | 按 GGUF 元数据 `dflash.rope.dimension_sections=[64,0,0,0]` | **0.56** | 0.00 | **rot=128 才对**(HF config 无 `partial_rotary_factor` ⇒ 全维旋转);GGUF 里的 64 是转换器残留,不采信是对的 |
| **spec-on ≡ spec-off** | 8k 首 8 token | — | **逐字相同** | **verify 的 8k 前向与 decode 一致**(先前 v2 在 7.7k 的"第 10 字符分叉"在 FI 臂不成立)⇒ 不是 verify 数值拖的 |

⇒ 排除三条后,AL 1.33 只剩**目标模型侧的 draft 输入**(taps)这一条差异通道:
同一草稿(HF bf16 与 GGUF Q4_K_M 都试过,AL 不变)+ 同一窗口 + 同一 verify 数值,
**换目标(AWQ 2.85 vs GGUF 1.33)才变**。
待办:把两档在 pos=8192 的 5 个 tap(`tap0..4` 输出槽)抽出来逐层对 cos;
若 GGUF 侧 tap 层号/数值有系统偏差,即为 AL 腰斩的直接原因。

### H. taps 对拍的**失败记录**(重要,别再走这条)

想用 verify 图的 `tap0..4` 输出槽直接对拍两档在 pos≈8182 的 draft 输入 ——
`OWL_TAP_DUMP`(临时探针)落盘后:

| 档 | tap 范数(行 0, 5 个 tap) | 判定 |
|---|---|---|
| GGUF | 55.8 / 183.4 / 142.8 / 50.8 / 2.7(有限、量级正常) | 槽**有写** |
| **AWQ** | **全 NaN** | 槽**未写**(声明了但没节点写它) |

⇒ 两档的 taps **供给路径不同**(AWQ 由图内内部节点直喂草稿,输出槽是死槽),
**槽面对拍不成立**;要真对拍必须在「图内 `project_memory(tap_views)` 的输入处」插桩
(改 `plan_dflash` 的图拓扑或加一次性 dtoh 节点),不能走输出槽。

**但这一轮仍有一个决定性结论**:

| 位置 | GGUF AL | AWQ AL |
|---|---:|---:|
| ~512 | **4.75** | ~4.5 |
| ~8240 | 1.33 | **2.85** |

⇒ **短 ctx 我们对齐甚至更好** ⇒ 可以排除「GGUF 量化质量差 ⇒ 目标更难猜」这一假设
(那样在 512 就该显现)。差异是**长 ctx 专属**的,且已排除:草稿权重、窗口、rotary、
verify 数值。剩下唯一通道仍是「长 ctx 的 draft 输入」,只是取证要进图内。

### I. taps 真·对拍(**结论:taps 没问题,AL 差是模型侧**)—— 2026-10-21

上一节的失败是探针位置错(在 `vg.replay()` **之前**读,读到上一轮/未初始化 ⇒ NaN)。
本轮回溯到**正确位置**(replay 之后、同一个 `vg` 借用内),并且落**原始字节**
(离线用 f16/bf16/f32 三种解释各试;实测槽 = **F16**,f16 解出有限值且量级正常)。

**对拍结果(pos≈8182,同 prompt,两档各一发)**:

| | tap0 | tap1 | tap2 | tap3 | tap4 |
|---|---:|---:|---:|---:|---:|
| 跨档 cos(GGUF, AWQ)行对齐 | 0.949 | 0.801 | 0.639 | 0.441 | 0.378 |
| **同档内相邻行 cos(GGUF)** | 0.957 | 0.837 | 0.716 | 0.478 | 0.542 |
| 同档内相邻行 cos(AWQ) | 0.970 | 0.908 | 0.807 | 0.755 | 0.756 |

**判据 = 与"同一模型相邻位置(1 token)"的漂移同量级** ⇒ 两档的 hidden 之差
≈ 目标**多走了一个 token**,不是错位/错层/量纲错(错位会让某一层掉到 0.0x,
量纲错会让 cos 与范数同时崩;这里范数 39~266 vs 36~344 同量级)。

⇒ **taps 假设排除**。AL 1.33 vs 2.85 的成因只剩"目标量化离 fp16 的距离"
(AWQ-INT4 g128 vs IQ4_XS 混型):**短 ctx 两者都贴近 fp16(AL 4.75 vs 4.5),
长 ctx 误差沿 64 层 × 8k 位置累积后分道,接受率的边际敏感性把这点差放大成 2×**。
⇒ 这是**模型侧**属性,不是引擎 bug;GGUF 档要 AL 就得换目标量化(Q4_K_M/AWQ),
引擎侧能做的是**速度**(verify 43.9 → Marlin 级 ~25)。

### J. 文献对照:长 ctx 接受率衰减是**已知问题**,且病因已定位到架构

查到四份直接对口的材料(2026-10-21):

| 来源 | 关键结论(带数) |
|---|---|
| **OWL / LongSpecBench**(arXiv 2510.07535) | 树式**transformer 草稿在其训练窗口外失效**:EAGLE3 长 ctx AL 只有 **1.28**;换 **LSTM(只看 last-token state)** 的长度无关草稿 ⇒ AL **~5×**;而"把 EAGLE3 拿去长 ctx 重训"(EAGLE3-L)**仍然更差** |
| **EAGLE-3.1 / Attention Drift**(arXiv 2605.09992 + vLLM blog 2026-05-26) | 病因两条:①**融合输入失衡**(高层 hidden 主导草稿输入)②**hidden 量级随步增长**(未归一化残差)。修法 = **fc 前对每个 target hidden 做 norm + 回灌 post-norm hidden** ⇒ 长 ctx **AL 最高 2×** |
| **DSpark 1M 实测**(botAGI LONG-CONTEXT-FINDINGS) | mean-accept:16K **3.30** → 64K 2.23 → 512K **1.98**;**逐位置**:pos0 0.80→0.54、pos4 0.18→**0.02**;同 ctx 换 KV 格式(NVFP4 vs fp8)mean-accept **2.1 → 3.0–4.2**;无投机对照证明差异**全在接受率、不是带宽** |
| **vLLM PR #53827** | EAGLE3 草稿在长 ctx 应**"限窗注意力 + 保留全量 KV"**(驱逐式滑窗对"续用长前缀"更差);限窗 vs 全 ctx 对照只差 **0.76%** |

**我们的实测逐条对上**:

| 文献条目 | 我们的证据 |
|---|---|
| 草稿超训练窗口即失效 | 窗口 2048:AL 4.75@512 / 4.31@2048 / **1.33@8240**;**关窗(全 ctx)→ AL 0.00** —— 就是这个论断的极端版 |
| "限窗 + 保留全 KV" ✓ | 我们的 `naive_attn_nc.cu` 正是掩码不驱逐 ✓ |
| **fc 前无 norm**(EAGLE-3.1 病因①) | 我们代码里明写「fc 的输入是**未归一化的 target taps**」(`dflash2.rs::project_memory`)⇒ **DFlash2 原设计就缺这一层** |
| KV 格式影响 AL(2× 级) | 我们在 GGUF 档实测 **fp8 KV:4.75→2.93 / 1.33→0.89(更差)** ⇒ 我们这侧 KV 精度**不是**杠杆(排除) |
| 块内冷启动(边界瞬态) | 我们统计的是 8k 处第 2 轮起的 m,不含边界瞬态(排除) |

**⇒ 结论**:AL 1.33 属"transformer 草稿 + 无 pre-FC norm + 目标 4-bit k-quant"这一类**已知模型侧缺陷**,
不是引擎 bug;文献给出的两个真正杠杆都在**模型侧**:
① **fc 前 tap norm / post-norm 回灌**(EAGLE-3.1,长 ctx 可达 2×,可先做**推理期 A/B**);
② 换目标量化(AWQ 档 AL 2.85 已实测)。
**引擎侧唯一杠杆仍是 verify 的 m=8 GEMM 效率**(43.9 → Marlin 级 25 ms),
而"按 ctx 降投机深度"在我们这里**无收益**(verify 是权重流受限,m=8 与 m=4 读同样多权重)。

### K. 速度三刀(2026-10-21 下午):轮内账 + P1/P3 定谳 + P2 证伪

#### K.1 一轮的**精确**账(nsys,512 ctx,spec d7,v2 臂,GPU busy 53.5 / wall 56.0 ms)

段界 = 草稿域的 bf16 核(`scale_bf16`/`mmq2_quant_*_bf16`);verify 47.28 ms(2189 核)、
propose 6.57 ms(384 核,1.08 GiB ⇒ **176 GB/s**,同病)。

| verify 内 | ms | 发数 | 每发 | 备注 |
|---|---:|---:|---:|---|
| MMQ2 GEMM(j16) | **35.0** | 459 | — | 14.24 GB ⇒ **407 GB/s** |
| gdn_recurrence | 3.65 | 96 | 38 µs | |
| **tile_q8_0(ssm_alpha/beta)** | **3.28** | 96 | **34.2 µs** | n=48 ⇒ tile 网格只有 12 CTA |
| paged_attention(v2) | 1.89 | 16 | 118 µs | **512→8k 变 ~15 ms(verify 的 26%)** |
| gdn_conv | 1.21 | 288 | 4.2 µs | |
| quant_q8_1 | 0.41 | 293 | 1.4 µs | |
| 其余(norm/silu/argmax/iq4nl/q2k) | ~1.8 | — | — | |

⇒ 判决:**唯一的钱在 GEMM(74%)与长 ctx 的注意力(26%)**;主机侧零空洞(2.5 ms)。

#### K.2 P1 定谳:`ssm_alpha/beta` 小 n 改走 MMVQ(**−2.8 ms/轮**)

同一张量(n=48,k=5120,Q8_0,260 KB)两条臂的实测:tile 34.2 µs(m=8,12 CTA)vs
MMVQ **5.1 µs**(m=1,96 CTA)。判据 = 「`m ≤ TILE_MT` 且整张量 ≤ 4 MiB(装得进 L2)」
⇒ MMVQ 的按 token 重读落在 L2 上,近乎免费(`layers/linear.rs::gguf_prefers_mmvq_small_n`,
两臂位级相同,已有门)。

| 512 ctx | verify | round | decode |
|---|---:|---:|---:|
| 改前 | 45.20 | 56.45 | 81.5 t/s |
| 改后 | **42.39** | **53.79** | **85.4 t/s** |

#### K.3 P3 定谳:注意力臂换 **FI 双面**(长 ctx 的 26% 变 3%)

`[dispatch] verify_arm = decode_arm = "fi"` + `fi_plan_bake = 0`(取 max_seq)+
`fi_plan_restore = false` + `fi_cuda_graph = true`(三项互相约束,缺一拒启)。

| 点 | v2 verify | **FI verify** | 备注 |
|---|---:|---:|---|
| 512 | 43.4 | **41.2** | 短 ctx 也小胜 |
| 8192 | 56.5 | **41.1** | **−27%**;decode 28.1→39.9 t/s(裸步同吃 FI) |
| XQA | 56.7 | — | 本档 f16 池 ⇒ **静默回落 v2**(别指望) |

`fi_cuda_graph` 让 FI 核留在捕获图内(plan 表由 host 每步刷 `fi_plan_sync`),
`profile` 里已落这五行 + 回退说明。

#### K.4 P2 证伪:**算子级 k 向分段是个陷阱**(同流串行,越切越慢)

想法:verify m=8 ⇒ J=16 ⇒ `ntx=1` ⇒ CTA 数 = n/128(40/48/80/136),而 j16 族寄存器
96–128 ⇒ occupancy 2 ⇒ 机器有 168 slot。于是把 k 轴切 S 段、**同核连发** S 次
(chunk0 store、后续 `dst +=`,零新缓冲)→ 期望 CTA 数 ×S。

实测(同二进制,`OWL_MMQ2_KSPLIT` 强制,512 ctx,verify 均值):

| S | verify | round | decode | Δround |
|---:|---:|---:|---:|---:|
| 1 | **40.89** | **52.68** | **82.0** | — |
| 2 | 42.43 | 54.11 | 79.4 | +1.4 |
| 4 | 45.46 | 58.08 | 79.8 | +5.4 |
| 8 | 50.74 | 64.11 | 74.7 | +11.4 |

⇒ 每多一发 ≈ **+3.2 µs**,线性叠加(459 发/S),**带宽一分没涨**。根因:
同流内的 S 个 chunk 是**依赖链**(同一 dst),既不能并发、每发又各自吃满一次
launch/teardown —— "更多 CTA 同飞"根本不发生。
**教训**:k 向分块必须是**单次发射内的** stream-k(`gx = nsm`),不能靠算子层连发;
而 §F 的 stream-k 计时探针只值 1.22×、还要 fixup 第二发(+459 发 ≈ +1.5 ms)⇒
净值 ≈ −4 ms(−7%),优先级排在模型侧之后。

数值面(门 `gpu_gguf_mmq2_small_m` 扩了「分段链 vs 单发」):S=3/4/8 段链相对最大差
**5.4e-4 ~ 1.2e-3**(门 5e-2)⇒ 机制本身没错,是调度模型不成立。

#### K.5 三刀后的 std 五点三轮(out 256,`data/gguf27b-iq4xs-std-spec-2026-10-21c-p1fi.json`)

| 点 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---:|---:|---:|---:|---:|
| decode t/s(A/B 两轮取样) | 88.2 / 88.8 / 82.9 | 86.8 / 78.5 / 77.8 | 62.4 / 58.6 / 58.7 | **38.5 / 39.0 / 39.4** | **37.6 / 38.9 / 35.0** |
| prefill | 1618–1645 | 1767–1781 | 1761–1768 | 1748–1753 | 1720–1731 |

与**本日改前**(v2 两臂 + tile 小 n;同协议 out=256 两轮)逐点对比:

| 点 | 改前 | 改后 | 比 |
|---|---:|---:|---:|
| 4096 | 34.49 / 33.75 | **38.46 / 39.01 / 39.38** | **+14%** |
| 8192 | 33.57 / 33.02 | **37.57 / 38.85 / 34.99** | **+12%** |
| 512(out=128 单点口径) | 81.5 | **85.4** | +4.8%(**仅 P1**;FI 的 512 增益另算) |

⇒ **4096/8192 从"投机净亏(裸步 38.8/40.6 t/s)"翻到打平/微赚** —— 主要靠 FI 把
长 ctx 的注意力从 26% 打到 3%,P1 是全员普涨的那一刀。

#### K.6 P6 定谳:J=16 档 **I 减半**(tile 数 ×2 ⇒ SM 覆盖 ×2)—— 全线 +13~17%

**怎么找对的**:P2 的证伪把问题从"CTA 不够"精化成"**SM 不够**"——
40 CTA 时每 SM 只有 **10.2 GB/s**,而机器的 per-SM 份额 = 936/84 = **11.1 GB/s**
⇒ 单 SM 已近饱合,**加深流水/多切块都没用,只能铺开 SM**。
I=128 时 `ntx = 1` ⇒ tile 数 = `n/128`(40–136),一半 SM 在空转。

**做法**(不动算核逻辑,只动 tile 宽):
`mmq-config-ampere` 里 J=16 的 44 行改成 `I=64 / nthreads=128 / occupancy 2`,
Rust 侧 `mmq2_i(m)`/`mmq2_threads(m)` 按 J 档取(64/128 与 128/256),几何与 smem
同源(`smem = ids + I·stride·4 + PAD(J·144, nthreads·4)` ⇒ 41 KB → **21.8 KB**,
2 CTA/SM 反而更宽松)。不变量:`I = rows_per_warp(16) × nwarps`。

| 512 ctx | verify | round | decode |
|---|---:|---:|---:|
| P1+FI(P6 前) | 41.18 | 52.5 | 88 t/s |
| **P6** | **35.27** | **45.7** | **93.8 t/s** |

**std 五点三轮**(out 256,`data/gguf27b-iq4xs-std-spec-2026-10-21d-p6.json`):

| 点 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---:|---:|---:|---:|---:|
| P1+FI | 88.2 / 88.8 / 82.9 | 86.8 / 78.5 / 77.8 | 62.4 / 58.6 / 58.7 | 38.5 / 39.0 / 39.4 | 37.6 / 38.9 / 35.0 |
| **+P6** | **103.5 / 102.9 / 94.6** | **98.1 / 90.5 / 86.4** | **71.4 / 66.8 / 66.8** | **44.6 / 44.3 / 45.2** | **42.1 / 44.1 / 39.9** |
| 中位比 | **+17%** | **+15%** | **+14%** | **+14%** | **+12%** |

prefill 不变(1717–1778 t/s;预填走 J=64/128 的 I=128 档,零改动)。

**两条被证伪的方向(留档,别再走)**:

1. **核内寄存器预取流水(P5)**:把下一 k 块的 16+4 个 global 读提到本块两次
   `vec_dot` 之前(单 smem 缓冲、occupancy 保留)——实测 verify 41.18→**41.45**、
   iq4xs 每发 76.7→**79.7 µs**,**零收益**。原因:装载器的 `#pragma unroll` 16 发
   本来就已经全在飞(预取窗口已 ≈ 一整个 k 块),且 40 CTA 时每 SM 已近份额上限。
2. **I=32(再切一半)**:`nthreads=64` / 2 warp 时某型的 `load_tiles` 在 I<64 上
   越界 —— 起机 warmup 即 CUDA 700(报在随后的 FI 调用点)。⇒ **I ≥ 64 是下界**,
   而 64 正好给出 80 tile / 80 SM。

#### K.7 GGUF 的 MTP 头:一直在文件里,没接(2026-10-21 核对)

另一条线(`feat/mtp-spec`,AWQ-INT4 + 检查点自带 MTP)8k 已到 **108–116 t/s**,
差的不是调优而是**两件结构**:

| 8k | AWQ+MTP | GGUF+DFlash2(P6 后) | 说明 |
|---|---:|---:|---|
| target verify | 22.9 ms / 625 GB/s | 35.3 ms / 403 GB/s | Marlin W4A16 vs MMQ2 k-quant 核级差 |
| propose | 7.74 ms | ~9–12 ms | MTP 头 0.35 GB(Q6_K)vs DFlash2 草稿 1.08 GiB |
| **AL(tokens/轮)** | **≈2.5** | **≈1.1** | **主因** |
| decode @8k | 108–116 | 40–44 | 2.5× |

**本档 GGUF 自带 MTP 块**(`blk.64.*`,共 15 张;`meta.rs` 目前按"M1 不载 MTP"
剔掉):

| 张量 | 型 | 形状 [in, out] | 对应 HF |
|---|---|---|---|
| `blk.64.attn_q` | **Q6_K** | [5120, 12288] | `mtp.layers.0.self_attn.q_proj` |
| `blk.64.attn_k` | **Q8_0** | [5120, 1024] | …`k_proj` |
| `blk.64.attn_v` | **Q8_0** | [5120, 1024] | …`v_proj` |
| `blk.64.attn_output` | Q6_K | [6144, 5120] | …`o_proj` |
| `blk.64.ffn_{gate,up}` | Q6_K | [5120, 17408] | …`gate/up_proj` |
| `blk.64.ffn_down` | Q6_K | [17408, 5120] | …`down_proj` |
| `blk.64.nextn.eh_proj` | Q6_K | [10240, 5120] | `mtp.fc`(in 两半 = fc_e/fc_h) |
| `blk.64.nextn.{enorm,hnorm,shared_head_norm}` | F32 | [5120] | `pre_fc_norm_embedding/hidden`、`norm` |
| `blk.64.{attn_norm,post_attention_norm,attn_q_norm,attn_k_norm}` | F32 | [5120]/[256] | 同 HF |

⇒ MTP 头驻留 **0.35 GB**(Q6_K 原样)/ ≈0.86 GB(解成 F16);
本档 live 17.5 GB + 0.86 ⇒ 仍装得下。**型是混的(Q8_0 + Q6_K)**,不是单一 Q6_K。

**预期**(本档实测量级):8k round ≈ verify 35.3 + propose ~5(MTP 层 + 3 发共享
lm_head)+ 杂 1 ≈ 41 ms;AL 若真到 2.5 ⇒ 3.5 token/轮 ⇒ **~85 t/s(+100%)**。
512 档 DFlash2 的 AL(4.7)仍更强(100 t/s)⇒ 短/长分档或按 AL 选草稿最优。

> 状态:**这条已由 `feat/mtp-spec` 那条线在实现**(未提交 11 文件:
> `MtpPredictor::new_gguf` / `GgufDequantSource` / `qwen35` 键映射 /
> `spec_runtime` / `loader` / `exec/draft` / `graph_fleet` 等),
> 本节只作**契约与数值底账**,不在 gguf 分支重复实现。

#### K.8 GGUF-MTP 通路验证(4B 打洞;27B 卡在分支)

**验证对象**:`feat/mtp-spec` 那条线的未提交实现(`MtpPredictor::new_gguf` +
`GgufDequantSource` + `qwen35` 的 `blk.N.nextn.*` 键映射 + `spec_runtime`/`loader`)。
我用它的二进制**只读**跑,不改它的树。

① **通路**(Qwen3.5-4B GGUF,自带 `blk.32.nextn.*`):

```text
[gguf] · 块 32 带 `nextn.*` = MTP/NextN 头(非解码层)⇒ 解码层数 = 32
GGUF 自带 MTP/NextN 块:blk.32.nextn.*（spec.draft=mtp 时由同一文件装载）
MTP 草稿头装载(spec depth=3;GGUF)   +   mtp.kv.k/v 各 32 MiB
```

② **数值恒等(门)**:同一固定 prompt(120 个随机整数,贪心,256 tok)——

| | 输出 |
|---|---|
| spec off(depth 0) | 464 字符 |
| **spec on(depth 3)** | **464 字符,与 bare 逐字节相同** |

⇒ 解量化源 + 键映射 + 伪序列 extend 臂**与裸 decode 同口径**(GGUF 侧的 MTP 通路正确)。

③ **4B 上的频谱**(math 域,out 256):

| prompt | spec-on MTP | spec-off |
|---|---:|---:|
| 512 | 134.6 t/s | **192.2** |
| 8192 | 159.7 t/s | **184.1** |

分相:round 22.0 ms = verify 15.9 + propose 5.3;AL = 533/237 = **2.25/3**;
**长 ctx 零衰减**(512→8192 反而 +19%)。

⚠️ **但 4B 上 MTP 是净亏**(160 < 184):盈亏线 = round/bare = 22.0/5.43 = **4.05**,
而 AL 只有 3.25 token/轮 ⇒ 小模型的 verify(4 行 × 2.7 GB)摊不开。
这与 27B 完全相反——27B 裸步 28 ms/token,MTP 轮 ≈ 42~45 ms 换 3.25 token
⇒ **13~14 ms/token**,所以 MTP 的收益是**大模型专属**(AWQ-27B 实测 108–116 t/s 印证)。

④ **27B GGUF 装载被分支挡住**(关键):

`feat/mtp-spec` 的 `family::gguf::PORTED` 只有 `{Q8_0, Q4_K, Q5_K, Q6_K}`,
而 27B UD-IQ4_XS 混了 **IQ4_XS / IQ3_S / IQ3_XXS / IQ2_S / IQ2_XS / Q3_K / IQ4_NL / Q2_K**
(本仓 `feat/gguf` 才有这些臂,两分支的 `family/gguf.rs` 差 **447 行**)⇒ 起机
**293 个键"型未移植"拒建**。4B 能跑只是因为它是 Q4_K_M。

⇒ **27B GGUF + MTP 必须先合分支**;合并时有两处硬冲突要裁决:

1. **注意力臂配置面**:本分支是单键枚举(`verify_arm`/`decode_arm`,2026-10-21 重构),
   对方是全字段布尔(`verify_v2`/`verify_fi`/`decode_fi`/`decode_xqa`)+ `e8f22e3`
   的"缺一拒启"schema ⇒ 取单键枚举 + 把全字段补齐;
2. **算核**:P6(J=16 档 I=64)只在 `feat/gguf` ⇒ 合并后 27B verify 才是 35.3 ms
   (不合并是 ~41)。

**合并后的 27B 预期**:round ≈ verify 35.3 + propose ~6~8(MTP 头 0.35 GB Q6_K /
F16 解到 0.86 GB + 3 发共享 lm_head)+ 杂 1 ≈ **43 ms**;AL 若 2.5 ⇒ 3.5 token/轮
⇒ **≈82 t/s**(对比:本档 DFlash2 现役 42 t/s;AWQ+MTP 108–116)。

#### K.9 合并 `feat/mtp-spec` + 27B GGUF-MTP 实测(**8k 42→81~86 t/s**)

**合并**(`5f73ddc`):`feat/mtp-spec` 的 5 笔 commit + 全字段配置 schema 并入 `feat/gguf`。
10 处冲突的裁决:

| 冲突 | 裁决 |
|---|---|
| 注意力臂 schema(我单键枚举 / 他五布尔) | **取单键**;9 个 profile 批量改写为 `verify_arm`/`decode_arm`;engine 里 4 处 `*_xqa/*_fi` 引用改 `matches!(…VerifyArm::Xqa)` |
| prefill 桶族(他退役 / 我改过) | 取退役;`prefill_plan.rs` 删除 |
| `graph.rs`/`phases.rs`/`spec.rs`/`engine.rs` 四文件大重构 | 以**他**为基(抽成 `verify_round`/`decode_step_inputs`/`boot_env_and_spec`),把我的两个**守卫**手工搬回新位置(部分驻留页占位符 / 视图页表不变量) |
| `gguf/mod.rs` | 并集(加 `dequant_source`) |
| `spec_runtime.rs` | 并集:他的 MTP 臂 + 我的 DFlash2-GGUF 几何不重开容器 |

**27B GGUF-MTP(depth=3,PROFILE 首版)**:

| 点 | decode | 备注 |
|---|---:|---|
| 512 | 81.4 / 85.2 | |
| 2048 | 85.1 / 82.3 | |
| 8192 | **77.5 / 81.1** | round 44.1 = verify 34.3 + propose 8.4;AL 2.63/3 |

⇒ 对比本档 DFlash2(42 t/s @8k):**+85%**,且**长 ctx 不衰减**(512 81 → 8k 78)。

**★ depth 契约翻案**:`spec_runtime` 里 MTP 分支原本硬钳 `depth.min(3)`
(对方文档记为"契约上限",代码无强制理由)。实测 **m3 = 28/35 轮满接受**,
第 4/5 步仍有 **70–80%** 接受 ⇒ 钳制解除(放到 7,与 DFlash2 同档):

| depth | 接受/轮 | round | tokens/轮 | ms/token | 8192 decode |
|---:|---:|---:|---:|---:|---:|
| 3 | 2.61 | 44.4 | 3.61 | 12.31 | 77–81 |
| 5 | 3.58 | 50.8 | 4.58 | 11.10 | 84.7 |
| 6 | — | — | — | — | 88.0 |
| **7** | **4.48** | **56.3** | **5.48** | **10.28** | **74.8 / 81.4 / 86.1** |

**std 五点三轮(depth=7,out 256;`data/gguf27b-iq4xs-mtp-d7-2026-10-21.json`)**:

| 点 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|
| decode | 100.7 / 109.5 / 100.1 | 100.0 / 99.6 / 95.5 | 106.7 / 95.6 / 109.6 | 91.1 / 106.1 / 92.7 | **74.8 / 81.4 / 86.1** |

分相(全矩阵):round 56.3 = **verify 35.5** + **propose 18.65** + 2.1;AL 4.48/7。

**与 DFlash2 档逐点比**(同为 P1+FI+P6 的 target 侧):

| 点 | DFlash2 | **MTP d7** | 比 |
|---|---:|---:|---:|
| 512 | 100.3 | 101 | 持平 |
| 2048 | 68.0 | **104** | +53% |
| 4096 | 44.6 | **96.6** | +117% |
| 8192 | 42.0 | **80.8** | **+92%** |

**数值**:贪心 256 tok,spec-on(d7) 与 spec-off 在 **240/450 字符**处分叉 —— 与
DFlash2 档(139 字符)同类(verify(m=8) 与 decode(m=1) 的核级累加序不同),
**不是 MTP 通路 bug**;MTP 头本身经 GGUF 块 → f32 → F16 装载(块内值不变)。

**剩余空间(下一刀)**:

| 项 | ms/轮 | 占比 | 杠杆 |
|---|---:|---:|---|
| verify | 35.5 | 63% | GEMM 已 501 GB/s(Marlin 625);stream-k(I=64 之上 k 切)上限 ~1.25× |
| propose | 18.65 | 33% | GPU ~10(7 发共享 lm_head GEMV 0.92 + MTP 层)+ **~8 host/回读**(`propose.read` 同步等图) |
| 其余 | 2.1 | 4% | |

#### K.10 写死项审计:depth 与「配置 vs 结构常量」的分界(2026-10-21)

用户裁决:**MTP 的 depth 是核心参数,不许写死**。审计结果(本会话新增/触碰的全部硬编码):

| 位置 | 原状 | 处置 |
|---|---|---|
| `spec_runtime.rs` 深度 | `depth_raw.min(3)`(MTP)/`.min(7)`(DFlash2) —— **静默截断** | **删掉截断**:`[spec] depth` 为唯一真源;上限由 `validate()` **响亮拒启** |
| `fold`/快照行容量 | 四处裸 `* 8` | 命名 `SPEC_VERIFY_ROWS_MAX = 8`(单一产地,`owl_shared::config`),四处改引用 |
| `SPEC_DEPTH_MAX` | 无(隐含) | 新增 = `SPEC_VERIFY_ROWS_MAX - 1 = 7`,并在 `validate()` 里挡 `depth > 7`(附单测 `spec_depth_over_structural_max_is_refused`) |
| `linear.rs` 小 n 判据 | `GGUF_MMVQ_L2_BYTES = 4 MiB` 常量 | **进配置** `[dispatch] gguf_mmvq_l2_mib`(缺省 4,`0` = 关);经 `EnvProvider::gguf` 下发,14 个档补齐字段 |
| `family::gguf` P6 `I=64` | `MMQ2_SMALL_I` 常量 | **保留为核常量**(与 `TILE_MT`/`MMQ2_I`/`MMQ2_THREADS` 同族:CUDA config 表 = 编译期实例,运行期换宽要双实例化;不是行为参数) |
| `MMQ2_SMALL_M = 4` | 常量 | 同上(臂选择 = 核几何,已有 `dispatch.verify_arm` 级开关作总闸) |
| profile 面 | — | `depth=7` / `kv_fp8=false` / `gdn_slots=4` 全部显式在档里 |

**分界律**(本仓既有约定,本节显式写下来):

- **行为/策略参数** ⇒ `[spec]`/`[dispatch]` 配置项,缺省值写在 `DispatchCfg`/`SpecCfg` 的 `Default` 里;
- **结构性容量**(sizing 同时被多处分配面消费) ⇒ 命名常量 + `validate()` 拒启,不许静默 clamp;
- **核几何**(tile 宽/线程数/MMA 档) ⇒ 常量(编译期实例表),换档 = 重编,不作为配置项。

#### K.11 配置面收口(承 K.10 的写死项审计):并顺手修了合并带出来的 5 处红灯

K.10 把三条律写下来后,按律把配置面收口,**期间发现合并(`5f73ddc`)带进来的一批
"制度 vs 夹具"冲突**,一并修掉(否则 `cargo test` 有 8 项红):

| 问题 | 根因 | 处置 |
|---|---|---|
| `--print-config` **被自己的严格门挡死** | 全字段门的报错文案让人用 `--print-config` 生成完整档,而生成器先被同一道门拒 | 新增 `load_resolved_for_print`(**宽松档**,只服务生成器);`--check`/启动仍严格 |
| 4 个档(含 `qwen38_27b-3090-iq4xs-dflash`)在新 schema 下**不可解析** | 它们是"只写差异"的半份档 | 用 `--print-config` 重新生成成完整档(保留 dir/draft/depth/FI 臂语义)+ 头注 |
| `profiles/legacy/` 被 `777ba69` 删了,**但 4 处调用面还在用** | 删除未同步收编(`legacy_path` 在 `testkit::case`/`engine/tests`/两个单测里) | 重建**最小两份**(`awq27b-3090-max`/`spec-dflash`,完整档无 intent) |
| `[intent]` 机制与"全字段门"**互斥** | 全字段门在 intent 解析**之前**跑 ⇒ 意图档必然缺键;而意图派生 `[model]` 又与显式 `[model]` 撞真源 | 两档并立:**有 `[intent]` 的档豁免**全字段门(意图 = 显式声明目标,派生确定);完整档仍缺一拒启 |
| `OwlConfig` 直读带 `[intent]` 的档会报 unknown field | 测试用 `toml::from_str::<OwlConfig>` 直读 | 意图头放在**完整 dump 之后**(两档并立),测试改读 legacy 里的纯完整档 |

测试面(本机实跑,全绿):`owl-shared` **108** · `owl-kernels` **116** ·
`owl-models` **296**(+ `gpu_gguf_mmq2_small_m` GPU 门)· `owl-engine` **116**。
## 2026-10-21 补录:**上下文阶梯 / fp8 全链 / 216k 定档**(目标函数裁定后)

> 本节从 `feat/gguf` 线移植到 main(该线基于较早的 tip)。迁移差异:
> - 配置组已改名 `[dispatch]` → **`[kernel]`**;臂开关在 main 上是
>   `gguf_tile / gguf_cublas / gguf_mmq2 / gguf_mmq / gguf_smem`(**全 true 缺省**)
>   与 `gguf_cublas_tile: Option<usize>` —— 与本节"env ⇒ 配置"同终局,只是键名不同;
> - `[pool].pool_tokens` 因档位走 `extends` 继承,自动档用**显式 `0`** 表达;
> - `fi_plan_bake` 在 main 上是 `bool`(上界恒 = max_seq)。

> 用户裁定(2026-10-21)的**常设目标函数**:
> **① 上下文长 > ② decode 速度 > ③ prefill 速度,但三者必须均衡。**
> 本节给出 27B GGUF + MTP d7(单 3090 Ti)在这三轴上的**实测阶梯**,据此把
> `gguf27b-mtp` 从 12288 定到 **32768(f16 池)**。

### M.1 四点曲线(f16 池,单次启动,std 256tok,三轮)

档 = `profiles/gguf27b-mtp.toml` + `--set runtime.max_seq/pool.pool_tokens`:

| ctx | pref 池 | 池大小 | prefill t/s | decode t/s(三轮中位) | TTFT |
|---:|---|---:|---:|---:|---:|
| 512 | f16 | 32 MiB | 1622 | **87.9** | 0.32 s |
| 8k | f16 | 0.5 GiB | 1708 | **78.2** | 4.8 s |
| 32k | f16 | 2 GiB | 1387 | **67.9** | 23.6 s |
| 64k | f16 | 4 GiB | 1099 | **57.5** | 59 s |
| 8k | **fp8** | 0.25 GiB | 1700 | 63.9 | 4.8 s |
| 96k | f16 | 6.0 GiB | 801 | **51.8** | 100–120 s |
| 100k | f16 | 6.4 GiB | 949※ | 58.0※ | 105 s※ | ⚠ 仅 **1/3** 成功 |
| 108k | f16 | 6.8 GiB | — | — | — | ⚠ **0/3**(per-turn 34–74 MiB 拿不到) |
| 112k | f16 | 7.0 GiB | — | — | — | ⚠ boot OK、**每 turn 必败**(free 114 < reserve) |
| 120k | f16 | 7.5 GiB | — | — | — | ⚠ 装配失败(末 258 MiB) |
| 128k | **fp8** | 4 GiB | 663 | **38.4** | 172–214 s |

※ 100k 那 1 次成功是唯一有效样本(另 2 次被 `vram-guard` 拒),**不作数**;
   96k 是 3/3 稳的上限。

原始:512 `87.9/96.7/84.0` · 8k `78.2/82.4/72.2` · 32k `74.7/67.9/62.2`(定档档
`max_seq=pool=34816`,prefill 1275–1387)· 64k `48.8/57.5/58.2` ·
128k(fp8)`37.1/38.4/42.9`。

### M.2 读法(为什么 32k 是拐点)

1. **decode 衰减是温和的**:512→32k 掉 23%(87.9→67.9),32k→64k 再掉 15%。
   这和"AL 在长 ctx 塌到 1.33"的旧账**不冲突**:MTP d7 的 AL 塌陷主要发生在
   DFlash2 草稿那条线;MTP 头在 32k 仍稳(否则 decode 会成倍掉)。
2. **prefill 衰减才是长 ctx 的主痛**:8k 1708 → 32k 1415(−17%)→ 64k 1099(−36%)
   → 128k 663(fp8,−61%)。TTFT 因此从 4.8 s 涨到 59 s / 3 min。
   ⇒ 第三优先级,但它是 **TTFT 的唯一支配项**(长 ctx 的"体感"就是它)。
3. **fp8 池在 8k 就亏 18%**(63.9 vs 78.2),而它换来的只是容量 —— 与
   §23.6 的定档实录同向(f16 核读 2× 字节仍更快:fp8 要逐元素反量化,compute-bound)。
   ⇒ **fp8 只在"f16 装不下"时才是正解**(本机 f16 能到 64k/4 GiB,够用)。
4. **两个边界别看错**(实测踩了两次):
   - **`max_seq` = 「prompt + 生成」合计**。`max_seq = pool = 32768` 时,一个
     32768 的 prompt 会被夹紧到 **0 生成**(`finish_reason=length`,CLI 报 FAILED);
   - **池 ≥ `max_seq`**,且 `Strict` 准入是整 ctx 常驻 ⇒ `131072` 的 prompt 配
     `pool=131328` 差 172 tok 被 `no_memory` 拒。
   ⇒ 本档取 **`max_seq = pool = 34816`**:32k prompt + 256 生成 + 余量全装得下。
5. **硬预算的实际上限 = 96k**(`vram_target=1.0` / `vram_reserve_mb=128`):
   权重 live 12.96 GiB + 固定 1.17 GiB + 池 6.0 GiB ≈ 20.1 GiB,其余全是
   **per-turn scratch**(实测 34–74 MiB,随 prompt 长度变)与 boot 期 258 MiB。
   池再大就"能启动、不能服务":100k ⇒ 1/3 成功、108k ⇒ 0/3、112k ⇒ 每 turn 必败、
   120k ⇒ 连装配都过不去。⇒ 想上 128k 只能 **fp8**(4 GiB)或 **TP2**(权重分半)。
   ⚠️ 早先"f16 只能到 64k"的判断是**没顶满预算**时的错判(当时 target=0.99、
   reserve=512,白留 ~6.6 GiB),已按本条修正。

### M.2b `prefill_split`(v2 prefill 的 nparts 分片 + LSE 归并):开,但只值 2~4%

同档 A/B(池 66048,8k 三轮):

| `dispatch.prefill_split` | decode(三轮) | prefill(三轮) |
|---|---|---|
| false(原缺省) | 69.7 / 78.6 / 73.2 | 1625 / 1714 / 1714 |
| **true** | 70.2 / 78.4 / 72.4 | 1656 / 1711 / 1708 |

⇒ 8k 无差;32k `1387→1444`(+4%)、64k `1099→1117`(+1.6%)。**decode 零代价**
⇒ 定为开。注:上游 FlashInfer 的 `fixed_split_size`(路线图里记的"≥16k 1.6×")
**本仓未实现(挂账)**,而且 prefill 桶族已退役(prefill 恒 eager)⇒ 这份收益拿不到。

### M.3 定档

| 档 | 配置 | 用途 |
|---|---|---|
| **`gguf27b-mtp`(默认)** | `max_seq = pool = 98304`,f16,`vram_target=1.0`/`reserve=128` | **ctx 顶满**(硬预算上限):96k / 51.8 t/s / 100–120 s TTFT / 6.0 GiB 池 |
| 64k(临时) | `--set runtime.max_seq=66048 --set pool.pool_tokens=66048` | 更"均衡"的档:57.5 t/s / 59 s TTFT |
| 32k(临时) | `--set runtime.max_seq=34816 --set pool.pool_tokens=34816` | 交互优先:67.9 t/s / 24 s TTFT |
| 128k(极端) | `--set dispatch.kv_fp8=true --set runtime.max_seq=140000 --set pool.pool_tokens=140000` | 批处理/离线;38.4 t/s / 3 min TTFT(**f16 装不下**) |

> TP2(双 3090)随手可翻倍:`max_seq` 到 131072 用 **f16**(8 GiB 池,fp8 不收),
> 且 prefill 约 ×2 ⇒ 128k 的 TTFT 会从 3 min 掉到 ~1.5 min。

### M.4 fp8 KV + **原生块草稿**(MTP 头量化)+ 192k —— 定档(2026-10-21 晚)

用户令:**用 fp8 KV、把草稿模型量化、草稿 KV 也 fp8、上下文拉到最大**。

#### (1) 草稿头从 F16 解量化改**原生块**(省 0.5 GB,且更快)

27B GGUF 的 MTP 头是 **Q6_K 12 张 + Q8_0 3 张**混型(§K.7)。原来 `opts.gguf_dequant = true`
⇒ `GgufDequantSource` 在 host 解到 F16(0.86 GB)。现在改走 `GgufSource` 原生块(0.35 GB):

| 件 | 改动 |
|---|---|
| `MtpPredictor::new_gguf(.., plan)` | 收 `PlanSource`(逐键表):`fc_e/fc_h` 用 `plan.at(虚拟键)`,内层用 `at_layer(block)` |
| `GgufSource` | 新虚拟键 `nextn.eh_proj.{e,h}`:**块对齐 strided gather**(半行 = 10 个 Q6_K 块),字节通道按块搬 |
| `DraftGgufPlan::plan_at` | 虚拟键按**物理键**取型(形状核对换 `inner == in*2`) |
| `MtpPredictor::fc_path` | 量化块档走 `Linear::forward`(统一分派 ⇒ GGUF 臂);裸 `matmul_nt` 只认 F16 行主序 |
| `Linear::is_gguf()` | 新增访问器(上面那条的判据) |
| `spec_runtime` | `native_blocks` 只在**单卡**开(TP 的 `eh_proj` 半拆是另一套块算术,暂走 F16) |

实测(512 ctx,fp8 池):**decode 87.9 → 101.9 t/s(+16%)**、prefill 不变。
⇒ 草稿不再付解量化,且省下的 0.5 GB 直接变成上下文。

#### (2) fp8 主池 + fp8 草稿池 + 192k(硬预算顶满)

| ctx | 池 | prefill | decode | TTFT | 备注 |
|---:|---|---:|---:|---:|---|
| 512 | fp8 | 1620 | **101.9** | 0.32 s | 原生块草稿 +16% |
| 100k | fp8 | 861 | **51.2** | 94–134 s | 三轮中位 |
| 184.8k | fp8 | 623 | — | **297 s** | 单发(池上限内) |

池上限(实测,`vram_target=1.0` / `reserve=128`):**192k 稳**(6.0 GiB);
216k 能启动但每 turn 差 **34 MiB** 被 `vram-hard-budget` 拒;235k 起不来
(`mtp.kv.k/v` 各 432 MiB + 权重 12.96 GiB + 固定 1.17 GiB)。

#### (3) 下一个杠杆(挂账)

**MTP 草稿 KV 池按 `pool_tokens` 分配**(`state.rs::mtp_kvs`,容量同 pool_tokens),
而草稿链每轮只用 **depth+1 = 8 个页** ⇒ 216k 时白占 864 MiB。砍到 `depth+1` 页
即可再 +26k ctx(或把 decode 的头寸让给 f16)。判据:草稿链 KV 无历史(每轮重算
hidden ⇒ 槽位可复用),改动面 = `state.rs` 一处 + 一轮 smoke。

#### M.4b MTP 草稿链 KV 也 fp8(此前 `draft_kv_fp8` 对 MTP **无效**)+ 预算项补账

立案时以为"MTP 池可以砍到 depth+1 页" —— **查代码后推翻**:MTP 链的 KV 是
**按绝对位置寻址**的(`exec/draft.rs::enc_slots = slot_at(pos+i)`、`kv_lens = pos+i+1`),
必须与 ctx 同尺寸。但查出两个**真问题**:

1. **`[spec] draft_kv_fp8` 只管 DFlash2**:`state.rs::DraftBank::alloc` 的 MTP 池恒用
   `dims.dtype`(F16)⇒ 2 KiB/token 的差在 192k 就是 **432 MiB**(且 `spec_runtime`
   的 MTP `draft_env` 硬把 `kv.quant` 设成 `None` 与之配套)。现改为按 `plan.draft_kv`:
   存储 = U32 扁平字节账(1 B/elem),叶子注解跟着走,`draft_env.kv.quant = Fp8E4M3`。
2. **MTP 池根本没进预算**:`PerTokBytes` 只有 `main_kv + draft_pool(dflash)`,
   MTP 链那 2~4 KiB/token 在预算外 ⇒ 手写 `pool_tokens` 时主池吃满、MTP 池再来抢
   ⇒ turn 级 `vram-hard-budget` 拒发。现新增 `mtp_chain` 一项(单测钉住 2 KiB/4 KiB)。

实测(`profiles/gguf27b-mtp.toml = fp8 主池 + fp8 MTP 池 + 原生块草稿`):

| 池档 | 之前 | 之后 |
|---|---|---|
| 192k | 稳(964 MiB 池外占用) | 稳 |
| 216k | ✗ 每 turn 差 34 MiB | ✅ **198.9k prompt 实测 451 s / prefill 441 t/s** |
| 224k | ✗ | ✗ 仍差 10 MiB(`fi_plan_bake` 的上界恒 = `max_seq`,挂账) |

⇒ 定档 **216k**(`max_seq = pool_tokens = 221184`)。

**更正(2026-10-21 实测)**:原先挂的"FI plan 表按桶 max_seq 烘可省几百 MiB"是**错的** ——
FI 的 workspace 是**定尺**的(`FI_FLOAT_WS_BYTES` 64 MiB + `FI_INT_WS_BYTES` 32 MiB +
host staging 32 MiB,构造期预钉),plan 表只是其中一个按键缓存的分片,**不随 ctx 增**;
而且 `validate` 明确禁止 `fi_plan_bake < max_seq`(表的工作项数 < 运行期 chunk 数 ⇒
partial 行 stride 错)。224k 只差 10 MiB 的真凶 = **每 turn 的 prefill 瞬态**
(`owl_silu_and_mul_f16` 36~50 MiB / `owl_add_f16` 10 MiB / `owl_fused_add_rmsnorm` 10 MiB /
`owl_gguf_dequant_rows_q3k` 14 MiB,随 `prefill_chunk=1024` 走),不是 plan 表。
⇒ 若要 224k:降 `vram_reserve_mb` 到 64、或 `prefill_chunk` 减半(各换 ~64 MiB),
而不是动 plan。

#### M.5 池容量改为**自动**(2026-10-21 用户令:"池子一般不需要手动指定 —— 算出来、把最大 ctx 显示出来")

| 件 | 改动 |
|---|---|
| `[pool].pool_tokens` | **不写 = 自动**:池 = 本档 `max_seq`(页对齐);手写值仍可钉(多会话/省显存),但**不再用来"开大 ctx"** |
| boot 报告 | 新增两行 `[ctx-max]`:① `per_tok` 分解 + 预算 `cap` + **算式上限 ctx**(`cap − 1 页`)+ 当前 `max_seq` 占比;② 池落账(页数 × 页宽、工作集常驻口径) |
| 拒启 | 自动档下 `max_seq > 算式上限` ⇒ 响亮拒并给出可服务上限(不再靠"池静默夹紧") |
| **治理器 bug** | `pool_cap` 的启用条件写成 `target > 0 && target < 1.0` ⇒ **`vram_target = 1.0` 时治理静默失效**(boot 里根本没有 cap 行,池怎么设都不校验,直到驱动 OOM —— 上一轮 224k 的"boot OK / turn 全挂"就是这么来的)。改为 `target > 0` |

**算式上限 vs 真机上限**(本档 `per_tok = 34816 B`:主池 32768 + MTP 链 2048):

```
[ctx-max] per_tok=34816B(主池 32768 + 草稿 0 + MTP 链 2048) ⇒ 预算 cap=280278 tok
          ⇒ 本档最大 ctx ≈ 280246 tok;当前 max_seq=221184(占 79%)⇒ 池 221184 tok / 7344 MiB(自动)
[ctx-max] 池落账 6912 MiB(221184 tok × 32 页宽;工作集 2240 tok 常驻,Strict 准入)
```

实测边界(同档,只改 `max_seq`):**216k ✓**(198.9k prompt,prefill 441 t/s / 451 s)、
220k ✗(差 17 MiB)、224k ✗。差额 = 未入账固定开销 ~1.6 GiB + 每 turn 瞬态。
⇒ **算式值只是上界,档位按实测取 216k**;`[ctx-max]` 两行正是为了让这件事一眼可见。


# HyperQwen 对标调研 —— 单 24G 卡 Qwen3.8-27B 服务的可摘清单

> 2026-10-10。对象:`repos/HyperQwen`(syv-ai,Apache-2.0)= vLLM 0.30.0 +
> ~50 补丁系列 + 模型准备管线,单 RTX 3090(250W)服务 Qwen3.8-27B。
> 作者与 syvai(Qwen3.8-27B-DFlash2-W4A16 草稿检查点来源)同源,
> **fast-variant 模型在 Hub 公开**(`syvai/qwen3.8-27b-3090-fast-variant`)。
> 性质:对标调研 + 可摘杠杆清单,映射 owl roadmap。

## 一、他们的数字(vs owl 同口径)

| 口径 | HyperQwen(3090 单卡) | owl(3090 Ti 单卡) | 判决 |
|---|---:|---:|---|
| 裸 decode 单流 | 45-46 t/s(batch 档单流) | **42.7**(marginal bench) | **打平** ✓ |
| spec decode 单流 | 121.8-131.2 t/s(DFlash2 C1 greedy,3.1-3.3 tok/step,步 39ms) | 引擎级 **117**(AL 4.10,轮 43.1ms) | **引擎级同量级**(89%)|
| spec decode **server** | 131(server 即引擎,V2 runner) | **36.8** | **3.6× 差距 = 服务层** |
| prefill | 未报直接 t/s(TTFT 口径) | FI **1102@8192** / 1185@512 | owl 口径更细 |
| ctx | 64k(默认)/150k(fp8+FI)/**240k(KVarN)** | **128k**(fp8 墙) | KVarN 是他们的 ctx 来源 |
| 引用/复现场景 | **381 t/s**(DFLASH_TOKENS=15 + lookup) | 无此能力 | 新杠杆 |
| 多轮前缀 | 状态精确恢复 23.5s→**0.85s** | prefix cache 有(实现不同) | 体验项 |

**关键重构认知**:我们与他们的 decode 差距**不在内核**(引擎级 117 vs 131,步时
39 vs 43ms,裸档打平)——**在服务层**(server 轮外开销把 117 烧成 36.8)。
prefill 我们 FI 修复后 1102 已是可战之数。ctx 差距 = KVarN 一个组件。

## 二、八个可摘杠杆(按 owl ROI 排序)

### ① server 轮外开销压缩(decode 战线的主矛盾)
server spec math 36.8 vs 引擎 117:每轮 ~104ms 中 ~70ms 是 actor/pump/
submit/emit 往返(引擎内轮账 43ms)。他们没有这层(V2 runner 直接服务)。
- owl 打法:轮内批量化(submit 连续轮)、read dtoh 与 propose 重叠、
  emit 出 actor;或 spec 轮循环下沉引擎连续泵(server 只交边界)。
- 收益:decode server 口径 36.8 → ~100+(3×),**不动内核**。

### ② KVarN KV 压缩(ctx 战线主矛盾)
华为 CSL,Apache-2.0,**参考实现就在 repos/HyperQwen/kvarn/files/**
(Triton 核 + backend):Hadamard 旋转 + 迭代方差归一 + **4-bit K / 2-bit V,
128-token tile**。他们的数:**240k ctx @ 单 3090**(E 档 268k 池 /
245760 max_len,decode 代价 ~47%)。
- owl 打法:移植**方案**(非 vLLM 集成)到 owl paged 池 + v2 读:
  K e4m3→int4、V e4m3→int2/4,Hadamard 预旋转,K0 写/读核改造。
  参考核 = kvarn/files 的 fused decode kernels。
- 收益:ctx 128k → **240k 级**(KV 面 32KB→~12KB/tok);及格 131k、
  达标 262k 一次覆盖。替代/合并 B8 三级缓存的 GPU 侧。

### ③ DFlash2 lookup drafting(n-gram 上下文起草)
`dflash2-lookup-drafting.patch`:扫描请求自身 token 历史,找"已生成序列
最长后缀"的最近出现,取其后继为草稿——**草稿免费,verify 块不受草稿器
block 限制**。复现模式 DFLASH_TOKENS=15(verify 16/step,草稿器仍产 7,
lookup 填余):**逐字复现 381 t/s(+47%)、quote+explain +12%、自由文本
不受影响(入口条件保证)**。入口/粘滞状态机(两连饱和步进入,粘 3 步,
单请求限)设计直接可抄;CUDA graph 双块长捕获的坑他们也踩平了。
- owl 打法:execute_spec_round 加 lookup 臂 + verify 块长动态(图双 shape)。
- 收益:RAG/复现/代码编辑场景 AL→15;自由域零损。

### ④ draft vocab 校准(DFlash2 selector 码本裁剪)
他们:lm_head 草稿头截 40,960 行(按**模型自身输出**计数,非通用语料),
覆盖率 92.1%→97.5%,**98→108.6 t/s(+10.7%)零其他改动**;49k 行封顶
(模型只发 ~54k distinct token)。
- owl 打法:DFlash2 candidate_selector 的前/后继码本 [248320, 256] 同构
  可裁:码本行裁剪 → selector 打分面缩小(算力)+ 码本显存 127MB→21MB
  + guaranteed-rejection 链截断消失。需要自采 owl 输出语料计数。
- 收益:AL 全域 +5-10%,零风险(覆盖率数据说话)。

### ⑤ embed/lm_head GPTQ int4(= owl C1,工具全套现成)
`drafter/gptq_lm_head.py`:Hessian 取自 300k 捕获 hidden states。
**RTN int4 = +1.5% PPL;GPTQ = KL 减半(+0.6% PPL,GSM8K 96.5% 不变)**,
lm_head+MTP 合计 **−1.8ms/step(108.6→118.8)**。
- owl 打法:C1 从"挂裁决"转正:借他们的 Hessian 管线(gptq_lm_head.py
  可直接跑 owl 同款检查点);顺带解决 ctx 审计面里 embed/lm_head f16
  ~5G 的大头(int4 后 ~1.2G)。
- 收益:decode +5-7%;ctx 审计面 −3.8G。

### ⑥ fast-variant 模型直接可用
`syvai/qwen3.8-27b-3090-fast-variant`(Hub 预建):校准草稿词表 + GPTQ
int4 heads + requant。owl AWQ 装载线(cyankiwi 同构)大概率直载。
- 零代码收益:换检查点即吃 ④⑤ 的成果(装载冒烟 + 恒等门/质量门即可)。

### ⑦ split-KV verify attention(`spec-decode-attn` + `spec-attn-smem-fit`)
verify 注意力 FLASH_ATTN split-KV + query-row tiling + smem 自适应。
owl verify 图的 paged attn 未专项优化(§6.21: verify 27.1ms 贴地板,
但那是 AL=4 语境;T=8 attention 1.95ms 占比小)——**低优先**,记录。

### ⑧ 负结果(省钱):MTP 头自蒸馏微调**无效**
KL 减半但 response-token top-1 agreement 不变(0.685→0.685),vLLM 接受
率噪声级。**验证 owl"草稿权重定,不可训练"的判断**——省一个训练项目。
另:mamba-chunked-prefill-align(GDN chunked prefill 状态丢失/NaN)、
pinned-kv-empty-cache(compile cache 1.2GiB 藏在 KV 下顶爆卡)——
两个我们同款踩过的坑,他们的修法可参考。

## 三、工程文化参照(可抄的坏习惯预防)

- **patch series 管理**:fork branch 一补丁一 commit,`patches/series` 顺序
  应用,`verify.sh --fuzz 0` 校验,过期补丁显式"retire when"列——owl 的
  补丁/内核变体矩阵(本轮 fp8 不一致案的根)值得借鉴此纪律。
- **benchmark 工具箱入库**:bench/ 每个测法一个脚本(accept 计数、soak、
  verbatim、conc_ladder、seat_ttft)——对应我们的"测量四把尺"但更成体系。
- **负结果入档**:MTP 微调无效写满一节——与我们 pitfall 文化同源。

## 四、owe 立项建议(更新 roadmap 排序)

| # | 项 | 战线 | 收益 | 量级 |
|---|---|---|---|---|
| 1 | server 轮外开销压缩 | decode | 36.8→100+(3×)| 1-2 天 |
| 2 | KVarN 方案移植(K int4/V int2 + Hadamard) | ctx | 128k→240k | 2-4 天 |
| 3 | DFlash2 lookup drafting | decode(复现场景)| AL→15,381 t/s | 1-2 天 |
| 4 | fast-variant 检查点直载 + selector 码本校准 | decode | +10% | 0.5 天 |
| 5 | embed/lm_head GPTQ int4(借 drafter/ 管线)| decode+ctx | −1.8ms/step,−3.8G | 1 天 |
| 6 | piecewise graph / FI-decode 后续 | prefill | 见上轮 | 另案 |

**与既有立项关系**:② 吸收 B8 的 GPU 侧(B8 的 RAM 卸载仍是 1M 冲刺的
另一半);③ 与"多块链式起草"互补(链式=草稿器加深,lookup=上下文白拿,
可叠加);④⑤ 合并进 C1/新② 旧账注销。
EOF
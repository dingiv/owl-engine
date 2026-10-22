# TP2 通信 —— 预研(现实约束 × 设计契约 × M2 施工序列)

> 立项:2026-10-16(与 W4A8 合入同窗) · 状态:**预研**(零代码改动) · 定位:
> `charter.md` **M2** 的开工单 —— "comm 尺寸路由 + one-shot AR;UUID 钉卡;
> TP2 数值对拍 TP1;**验收:对照 vLLM 同机 TP2 prefill/decode 三线表**"(charter:220-222)
> 前置阅读:charter §二 A2(六条通信公理)/ `crates/backends/comm/src/lib.rs`(91 行策略骨架)
> 配套读物:《tp-overview.md》(行业怎么做:数学骨架 / 各引擎实况 / 陷阱清单)
> 下游:《tp2-design.md》(具体怎么实现:落点文件 / 命令与核 / 协议不变量 / 门 / 提交切分)
> 本篇只做两件事:① 把 A2 公理与本机**魔改驱动现实**对齐(发现一处必须修正的冲突);
> ② 把 27B 的 TP 切分面/通信量/验收线算清楚,给出 M2 施工序列。

---

## 0. 结论速览

1. **A2.6 有一处必须修正**:公理写的"跨卡通信走 `enablePeerAccess` 直写"在
   **3080 类对(BAR1 256MiB)上会触发进程级急切映射 → 192MB 预算墙** —— llama.cpp
   单进程引擎就是这么被打死的(llama 卷《p2p-allreduce-后端-施工文档》§一:三味全灭)。
   正确姿势 = **显式 VMM 授予**(`cuMemCreate/Map/SetAccess`,≤192MB 分块),
   **全程不调 `cudaDeviceEnablePeerAccess`**。owl 的 A6 池契约
   (`Pool::malloc_peer_shared` = "VMM(cuMemCreate,粒度取设备最小值)+ PeerShared 池")
   恰好就是这个形状 ⇒ **owl 结构性免疫**,但**必须把"禁 enable"写进能力探测与契约**,
   否则照 A2.6 字面实现就会踩同一雷。
2. **两档机制必须分档**:3090 类对(BAR1 32G,静态映射)= enable 可用;
   3080 类对(BAR1 256MiB,动态窗口)= 只准 VMM 显式授予。分档信息现在**不在**
   `DeviceDesc` 里(`{uuid, arch, total_bytes}`,iface:64-70)⇒ 预研建议加 **peer 能力描述**。
3. **owl 的 TP2 红利预期要压**:本机 27B decode 的逐核归因是 **GEMM 只占 ~35%、
   胶水核 ~36% + 调度 ~20%**(代差分析卷 2026-10-01 实测),而胶水/调度在 TP 下
   **不切分(复制)** ⇒ decode 提速远达不到 vLLM 的 2.32×(那是 GEMM-bound 引擎的红利);
   粗算 **1.2~1.4×**,真正的杠杆回到 A2.3 的融合口(AR+残差+norm)。
   **3080 对的价值主要是容量(27B 装进 20G×2),不是速度**。

---

## 1. 本机现实约束表(全部实测/带出处)

| 项 | 现场值 | 出处 |
|---|---|---|
| 卡 | 3090Ti@41:00/42:00(BAR1 32G)· 3080@01:00/C1:00(BAR1 **256MiB**,无 ReBAR) | `nvidia-smi -q` 本机实测 |
| 驱动 | **615.71.09**(aikitoria fork 自编,`div@s8030` Release Build) | `/proc/driver/nvidia/version` |
| 在位模块 | `modules-615-reserve-param` md5 **a8e5167d**(混插+探针+预算保留参数化),运行时 `nv_dynbar1_reserve_mb=48` | `p2p-build/README.md` §现役快照 |
| 主干分支 | `615.71.09-p2p-3080-20g` @ `e0609f7d`(5 笔:duanyll×2 + 律v3 + 探针 + 混插;已推 fork) | 同上 |
| 六对带宽 | **13.2 GB/s 双向零 mismatch**(64MB/128MB;含混插四对) | v3基线§十 / 混插§14.9 |
| 律 v3(窗口) | GSP 按 **2MB 粒度**绑定 aperture ⇒ 死区 = `range0 mod 2M`;修法 = 2M VA placement floor + 64K PTE 解耦(align3) | v3基线 §10.1 |
| 窗口语义 | 懒创建/refcount/随分配销毁,稳态零映射开销;"窗口大小 ≠ 管径"(TLP ≤512B) | method3 施工文档 §一 |
| **192MB 预算墙** | peer-enabled 时**单笔分配 > (BAR1 − 64MB 预留) ≈ 192MB** 被硬拒(`cudaMalloc OOM`;`cuMemSetAccess` 整段授权撞同一墙) | vmm-p2p 报告:164-168 / v3基线:288-298 |
| 3090 对 | **无墙**(32G BAR1 不走动态窗口) | vmm-p2p 报告:182 |
| 硬重置红线 | **595/v1 时代**"3080 发起端"邮箱路径专属(method3:528);615 align3+混插后六对双向全绿 ⇒ **红线已被现役驱动解除**,但保留能力探测黑名单开关(vLLM/mistral"合成流量全绿≠引擎可用"教训) | gpu2-p2p悬案 / 混插§14.9 |
| 混插两路 | 3090→3080 = **窗口**(residue=0)· 3080→3090 = **静态**(原路) | AGENTS.md / 混插§14.5 |
| 驱动升级风险 | apt/unattended-upgrade 会自动顶内核+装官方驱动包 ⇒ vermagic 失配失联(**已上演四次**);换内核/驱动后需 `make clean` 重编同 commit 弹药 | p2p-build README(09-26/10-12 两条) |
| NCCL | 已除名(A2.5);但**读数可参考**:vLLM 3080 对 P2P(NCCL IPC 形态)prefill +28.5% | nex-3080 会战:120-140 |
| SHM 兜底 | host 中转有效带宽 ≈ 6-12 GB/s;llama.cpp 实测 prefill/dec 2538/**131.7** | p2p-allreduce 文档头 |

**一句话**:现役驱动下"六对全通 + 13.2GB/s"是既成事实;真正的工程约束是
**① 3080 类的单笔 ≤192MB + 禁 process-wide enable;② 驱动/内核升级会掀桌。**

---

## 2. 27B → TP 切分面(精确几何)

`models/cyankiwi/Qwen3.8-27B-AWQ-INT4/config.json`(text_config)实测:

```
hidden 5120 · layers 64 · inter 17408 · vocab 248320
全注意力层(16 层,full_attention_interval=4):
  Q 头 24 / KV 头 4 / head_dim 256   ⇒ Q 6144 / K,V 1024 每 rank 512/512
GDN 线性注意力层(48 层):
  NK 16 / NV 48 / KD 128 / VD 128 / conv_kernel 4
  ⇒ K 2048 · V 6144 · conv 通道 10240(= 2K+V,与 kernels 测试常量一致)
```

| 张量/状态 | TP2 切法 | 每 rank 形状 | 备注 |
|---|---|---|---|
| q_proj [6144,5120] | 列切(按 Q 头) | [3072,5120] | 12 Q 头 |
| k/v_proj [1024,5120] | 列切(**按 KV 头**) | [512,5120] | 2 KV 头,GQA 比 6 保持 |
| o_proj [5120,6144] | 行切 | [5120,3072] | **输出需 AR** → [T,5120] |
| GDN in_proj_qkvz | 列切(按 NK/NV 头) | K 1024 / V 3072 … | NK 8/NV 24,GQA 比 3 保持 |
| GDN conv1d(10240 通道) | 通道切 | 5120 | 每 rank 独立卷积 |
| GDN 递推态 `S` | **按 V 头切**(head-parallel) | NV 24/rank | A_log/dt_bias 随头切;状态池按 rank 减半 |
| mlp gate/up [17408,5120] | 列切 | [8704,5120] | SwiGLU |
| mlp down [5120,17408] | 行切 | [5120,8704] | **输出需 AR** → [T,5120] |
| embed [248320,5120] | 行切 | [124160,5120] | |
| lm_head [248320,5120] | **列切(vocab-parallel)** | [124160,5120] | logits 需 gather 或 argmax 归约 |

**结论**:27B 是**混合架构**,但 GDN 的 TP 语义天然干净 —— **head-parallel**
(NK/NV 按头切、GQA 比保持、逐头状态 `S`/A_log/dt_bias 随头走、conv 按通道切)。
不需要为 GDN 发明新并行语义;要小心的只有两处:
**① conv 的 4-tap 窗口跨 chunk 边界(与 TP 无关,是 chunked 语义既有难点);
② 状态池 `gdn_slots` 的每 rank 副本(nv 头 24 vs 48 ⇒ 池减半,账目要重算)。**

---

## 3. 通信量账(为什么路由必须是架构对象)

每层有 **2 次输出 AR**(attention/GDN 的 o_proj + MLP down),64 层 ⇒ **128 次 AR/步**,
每次载荷 = `[T, 5120] × dtype`。

| 相位 | T | 单次载荷(f16) | 每 rank 总量 | @13.2GB/s | 对照 |
|---|---|---|---|---|---|
| decode | 1 | 10 KB | **1.31 MB/步** | **99 µs** | 现役步时 ~22-27ms ⇒ 带宽税 ~0.4% |
| prefill | 8192 | 80 MB | **10.7 GB** | **0.81 s** | TP1 prefill 该点 6.1s ⇒ 串行 AR 吃 13% |

**推论**:
- **decode 的敌人是发射数不是带宽**(128 发 × 3-5µs ≈ 0.4-0.6ms,比 99µs 大 5 倍)
  ⇒ decode 路必须 **one-shot(单 kernel 内完成二卡交换与累加)+ 融合**
  (A2.3 的 AR+残差+norm 直接把 128 发收敛成 64 发甚至更少)。
- **prefill 的敌人是带宽**(0.81s 串行)⇒ 必须 **two-shot/ring 且与 GEMM 重叠**;
  若不重叠,TP2 的 prefill 只有 ~1.2-1.5×(vLLM 实测 1.48×,与账吻合)。
- **owl 特有的坏消息**:逐核归因(代差分析 2026-10-01)显示 decode 步里
  **marlin 30% + cublas 5% = GEMM 仅 35%**,胶水核 36% + 调度 ~20% **不随 TP 切分**
  ⇒ 即便 GEMM 完全减半,步时也只降 ~17% ⇒ **decode 预期 1.2× 量级**,
  而非 vLLM 的 2.32×(vLLM 是 GEMM-bound)。**这条要在 M2 立项时先和用户对齐预期**,
  否则"TP2 只快 20%"会被误读成工程失败。
  (修正杠杆:胶水核里 norm/rope 可以在 AR 融合里吃掉一部分 —— 正是 A2.3 的设计意图。)

---

## 4. 后端策略谱(落 A2.1/A2.2/A2.5;本机可用性矩阵)

| 后端 | CommCap | sweet spot | 本机姿势 | 出处/依据 |
|---|---|---|---|---|
| **one-shot AR**(自研裸 kernel,VMM 显式授予缓冲) | InGraph | 0 … ~1MB | 六对皆可;3080 类**禁 enable**,缓冲 ≤192MB 分块授予 | p2p-allreduce 设计(2×32MB 常驻授予) |
| **two-shot / 分块 ring** | InGraph | ≥1MB | 同上(prefill 大消息) | A2.1 公理 |
| **SHM host 中转**(兜底保险丝) | SegmentBoundary | 任意(慢) | 永远可用;6-12GB/s 有效 | llama.cpp internal 后端 |
| ~~NCCL~~ | 除名 | — | 账外缓冲违 A5.2;capture 姿势苛刻 | A2.5(2026-09-22 裁决) |
| ~~process-wide `enablePeerAccess`~~ | **黑名单(3080 类)** | — | 触发进程级急切映射 ⇒ 192MB 墙逐笔撞死 | llama.cpp 三味全灭定谳 |

**路由断点建议(待实测校准)**:decode 载荷 1.31MB/步 落在 one-shot 的甜区上界附近
⇒ 断点取 **≥2MB 走 two-shot**;融合口(AR+add+norm)让路由粒度从"张量"升级为"层相位"。

---

## 5. 硬性设计要点(现实约束 → 契约)

1. **能力探测与分档(A2.4 扩展)**:`DeviceDesc{uuid, arch, total_bytes}` 不足以表达
   "这卡能不能 enable / 单笔上限多少" ⇒ 建议加 `PeerCaps { bar1_bytes, mechanism: Static|Window, per_alloc_limit }`,
   探测在 comm 初始化完成、热路径零探测(A2.4 原律)。
2. **3080 类对契约**:`enable_peer_access` 对该档 **返回结构化拒绝**(不是静默降级);
   `malloc_peer_shared` + `map_remote` 是唯一合法通道;授予 ≤ min(192MB, 声明上限) 分块
   (与 `cuMemCreate` 块粒度对齐,窗口 refcount 随映射生命周期)。
3. **通信缓冲入账(A5.2)**:one-shot staging 常驻(参考 2×32MB/对)+ per-AR 零分配;
   `MemStats.peer_mappings / peer_mapped_bytes` 已在 iface 记账面 ⇒ 复用。
4. **可捕获性(A2.2)**:one-shot 是裸 kernel ⇒ InGraph 天生成立;SHM 后端声明
   SegmentBoundary,编排层据此排段(禁止捕获期撞上)。
5. **故障语义(A2.7)**:P2P 中途失败 ⇒ 全有或全无(不得半可用);降级路径 =
   切 SHM 后端 + 记事件,**不静默**。
6. **进程拓扑(A2.6,保留)**:单进程 N 后端实例 / 每卡一 runner 线程 —— 这条**不改**,
   改的只是"跨卡写"的机制描述(见 §0.1)。

---

## 6. 对标读数与 M2 验收线

**vLLM TP2(3090 对,27B AWQ + DFlash2 d7 + fp8 KV + 262K + 前缀缓存,std 五点)** —— 对照组主锚:

| 长度 | prefill | decode | TTFT |
|---|---|---|---|
| 512 | 1971~1978 | 94.8~158.3 | 261~281 ms |
| 1024 | 2099~2115 | 97.4~103.9 | 487~490 ms |
| 2048 | 2011~2047 | 92.0~122.0 | 1003~1021 ms |
| 4096 | 2052~2083 | 95.0~136.2 | 1969~1999 ms |
| 8192 | 2062~2075 | 89.9~113.5 | 3950~3975 ms |
| **avg** | **2040.6** | **108.9** | 1542.9 ms |

同卡单卡对照(vLLM TP1):prefill ≈1380(由 owl 932.5 差 1.48× 反推)、decode **46.86**
⇒ **TP2 增益:prefill 1.48× / decode 2.32×**。

其它引擎的"TP2 形状律"(用于选后端,不用于绝对比对):

| 引擎/对 | 配方 A | 配方 B | 律 |
|---|---|---|---|
| exl3(3090 对,EXL3 4bpw) | native(host SHM)+MTP:prefill 1307@4k / decode ~91 | nccl+P2P:1809@4k / ~75 | **prefill 要 P2P(+38%),decode 要 host(+22%)** |
| vLLM(3080 对,Nex INT4) | 5936 / 153 | 7656(+28.5%) / 157.2(+2.4%) | 大消息带宽敏感,小消息延迟主导 |
| llama.cpp(3080 对) | P2P **结构性封死**(legacy 857MB 段撞 192MB 墙) | SHM 2538 / 131.7 | 单进程 + 大段分配 = 撞墙 |

**owl 现状(TP1,今日回归实测)**:prefill **1340.8** · decode **36.5**(spec-off,池满)
⇒ M2 的三线表(TP1 → TP2):

| 指标 | owl TP1(实测) | owl TP2(目标,粗估) | vLLM TP2(参照) |
|---|---|---|---|
| prefill avg | 1340.8 | 1.2~1.5× ⇒ **1600~2000** | 2040.6 |
| decode avg | 36.5 | 1.2~1.4× ⇒ **44~51** | 108.9 |
| TTFT@512 | (待同口径实测) | — | 261~281 ms |

**M2 验收门(建议写进工单)**:
1. **TP2 ≡ TP1 数值对**拍门:同权重同 prompt,greedy 逐 token 一致(参照
   `gpu_spec_identity_gate` 的做法:token 级硬判,不用"看起来对");
2. **三线表**:prefill / decode / TTFT,五点三轮,与 TP1 同机同口径;
   ⚠️ **口径对齐是第一件事** —— vLLM 表含投机+fp8 KV+前缀缓存,owl 的对照必须
   同档或显式声明差异(否则又是一次"数字跨口径引用");
3. **零 Xid** + `NCCL_DEBUG` 式通道取证(自研路则用 P3 探针/自身日志);
4. **账目闭合**:TP2 下两 rank 的 `bytes_alive` 稳态无漂移(A5 硬合同)。

---

## 7. 未决清单(现状:4 条已裁决 / 3 条待定)

### 7.0 用户裁决(2026-10-16)

| # | 裁决 | 含义(写进设计) |
|---|---|---|
| 1 | **验收口径不补对照臂** | 先把 TP2 做出来,整机跑一遍总对比即可(不单独立一条 spec-on 对照门) |
| 2 | **直奔 3080 对形态** | 不先做 3090 对的"容易版";**从第一天就按 3080 档写**(VMM 显式授予、≤192MB、禁 enable) |
| 3 | **通信全自研,不用 NCCL** | 不是"先 NCCL 顶一下"——集合库自己写,只满足基本需求 |
| 4 | **投机(DFlash2)拆开** | 草稿随主模型一起按 head/维度切,不复制 |
| 5 | **前缀缓存后议** | TP 先跑通,块表/缓存面留到 TP 之后 |

**用户对魔改驱动的简化判词**:"它只有一个硬要求 —— 不能分配太大的显存;
其他的照着正常写就 OK。" ⇒ 驱动侧的全部复杂度收敛为**一条纪律**:
peer 可映射的单笔缓冲 ≤192MB(超了响亮拒),别的一律按标准 VMM 写。

### 7.1 剩余待定

| # | 问题 | 状态 |
|---|---|---|
| 4 | 图预算 ×N(A5.1):两 rank 各自捕获图形 | 待实测(账在 §5.3) |
| 5 | ~~前缀缓存块表一致性~~ | **裁决:后议** |
| 6 | 3080 对 20G 预算账(权重 7.6G + KV + 图 + 通信能否容 262k ctx) | 待算/待实测(可能否决长 ctx) |
| 7 | 混插对(3090+3080)有无意义 | 低优先 |

---

## 8. 施工序列(按用户裁决重排:**3080 档优先 + 自研集合库**)

| 步 | 内容 | 验收 |
|---|---|---|
| **S1** | **最小集合库骨架**(见 §10):构建期能力探测/分档 + VMM 显式授予
  (peer alias + ≤192MB 分块)+ 账本入账 | 单测 + 六对分类正确;超限响亮拒 |
| **S2** | **allreduce(one-shot,InGraph)**:本地 D2D → staging → flag → peer 读回累加 | 与 host 参考逐位对拍;图捕获 ≡ eager |
| **S3** | **allgather**(vocab-parallel logits 取回;双 rank 拼接) | 同上 |
| **S4** | **接引擎**:TP2 双实例装载 → 层内 AR 接线 → **TP2 ≡ TP1 数值对拍门** | 逐 token 一致 |
| **S5** | **prefill 分块 ring + 与 GEMM 重叠**(大消息) | prefill 增益 ≥1.3×;三线表出表 |
| **S6** | **3080 对整机装车**(20G 预算账闭合 + 长跑零 Xid) | 三线表 + 总对比 vLLM |
| 后置 | SHM host 中转保险丝(SegmentBoundary 后端) | 非阻塞项 |

---

## 9. 最小集合通讯库 —— 设计草案(自研;无 NCCL)

> 定位:满足 owl TP2 的**基本需求**;复杂度全部让位于"3080 档从第一天就跑得起来"。
> 依赖:`owl-iface` 既有窄口(`malloc_peer_shared` / `map_remote` / `DeviceDesc`)+
> `owl-cuda` 池层 + `owl-graph` 的 GraphPhase。

### 9.1 范围(必须有 / 明确不做)

| 必须有 | 说明 |
|---|---|
| **allreduce** | row-parallel 输出合并(o_proj/MLP down),f16/bf16/f32,任意字节长度,chunk 循环 |
| ~~allgather~~ → **留位不写** | **v1 裁决(2026-10-16)**:lm_head 两 rank 各算全词表(复制)⇒ 不需 AG;
但判词同时是“反正最终都要写完”⇒ **staging/flag 层必须把 AG 的扩展点留出来**(见 9.7) |
| 2 rank / 单进程 / 每卡一 context | A2.6 拓扑不变 |
| VMM 显式授予 + peer alias | **禁 `enablePeerAccess`**;单笔 ≤192MB(超限响亮拒) |
| InGraph(图内安全) | flag 协议 + 自旋;数据面无 host 同步/无 D2H |
| 账本 | staging 走 PeerShared 池;`peer_mapped_bytes` 如实 |

**明确不做(v1)**:NCCL、多 rank(>2)、reduce_scatter/all_to_all、
跨进程 IPC、SHM 兜底(留 trait 位,二期)、融合 AR+add+norm(口子在 A2.3,二期)
—— 以及 **allgather(v1 用 lm_head 复制替代;但接口/协议为其留位)**。

**v1 代价记账(lm_head 复制)**:每 rank 多 ~0.64GB(int4)权重;
prefill 末层 lm_head 算力不摊半(约总墙时 5%);两者都记在 20G 预算账里(§7.1 #6)。

### 9.2 API 草案(`crates/backends/comm`)

```rust
pub struct CommGroup { /* 2 rank 的上下文 + staging 账 */ }

impl CommGroup {
    /// 构建期:能力探测 → 分档(Static|Window)→ VMM 授予(≤192MB 分块)→ 账本登记。
    /// Window 档(3080)拿到 enable 请求 = 结构化拒绝(不静默降级)。
    pub fn new(plan: &TopologyPlan, ranks: &[DeviceHandle]) -> Result<Self, CommError>;

    /// allreduce(in-place)。payload 任意字节;内部按 chunk 分块走 staging。
    pub fn allreduce(&self, buf: DevBufRef, bytes: u64, dt: Dtype, phase: GraphPhase)
        -> Result<(), CommError>;

    /// allgather:各 rank 一片 src → 全量 dst(rank 序拼接)。
    pub fn allgather(&self, src: DevBufRef, dst: DevBufRef, bytes: u64, dt: Dtype)
        -> Result<(), CommError>;
}
```

### 9.3 数据面与协议(one-shot;chunk ring)

```
staging:每 rank 16~32MB × 2 块(chunk ring;≤192MB 单笔授予,常驻)
AR(payload P,chunk C):
  1) 本地 D2D:P → 自己 staging[k]          (局域写,全速,零穿越)
  2) fence + flag[k] = seq(单调计数)
  3) 归约核:P += peer_alias 的 staging[k]   (对端读 = 窗口直读,P2P 单程)
     自旋上限 = 响亮失败(不静默错值)
AG:同形,只是第 3 步改为按 rank 偏移写入 dst
```

**性质**:每 rank 每 AR = 1 次本地写 + 1 次远端读(单程穿越);
与 llama.cpp 已验证的 p2p 设计同构(其 §3.1),但**换成 owl 的池/账本语义**。

### 9.4 要写的核(尽量少)

| 核 | 作用 |
|---|---|
| `owl_ar_stage_copy_f16/bf16/f32`(或直接 cudaMemcpyAsync D2D) | 本地 → staging |
| `owl_ar_reduce_add_{f16,bf16,f32}` | P += peer staging(16B 向量化 + flag 等待合并) |
| `owl_ag_scatter` / `owl_ag_gather` | allgather 两半 |
| (可选)`owl_ar_fused_add_rmsnorm_f16` | 二期,融合口 |

### 9.5 门(设计时就定,别事后补)

1. **host 参考对拍**:随机数据 + 边界尺寸(0/1B/10KB/1MB/32MB±1/100MB)、非对齐长度,
   f16/bf16/f32 三 dtype;
2. **确定性**:同输入两跑逐位相同;
3. **图捕获门**:捕获后 replay ≡ eager(owl 既有对照法);
4. **墙门**:≤192MB 授予成功;故意 >192MB 单笔 → 响亮拒;
5. **账本门**:`peer_mapped_bytes` 生命周期闭合、无漂移(A5);
6. **长跑门**:≥1000 次 AR 零 Xid(驱动稳定性);
7. **性能门**:64MB 载荷带宽 ≥12GB/s;decode 尺寸(10KB)单次 ≤10µs。

### 9.6 预期读数(据 §3 通信量账)

decode 步 1.31MB(128 次 AR)→ 带宽税 ~99µs、发射 ~0.4-0.6ms;
prefill@8192 每 rank 10.7GB → 串行 0.81s(必须与 GEMM 重叠)。

### 9.7 扩展点(为“最终都要写完”留位;v1 不实现)

| 扩展 | 为什么现在就要留位 | 怎么留 |
|---|---|---|
| **allgather** | lm_head 词表并行是必然的下一步(省 0.64GB/rank + prefill 减半) | staging 的块/flag 表做成**数组**(per-collective kind × per-chunk),别把 AR 写死为“只有一个环”;`CommGroup::allgather` 先留签名 + `unimplemented!` 叶 |
| **融合 AR+add+rmsnorm** | A2.3 明写的 decode 头号杠杆(owl 胶水占 36%) | 归约核签名带 `fuse: FusePlan` 参(常量泛型/编译期),v1 传 `None` |
| **SHM 兜底** | P2P 失败时的保险丝(A2.7 全有或全无) | `CommBackend` trait 已有的 `SegmentBoundary` 位——不动 |
| **two-shot/ring 大消息** | prefill 与 GEMM 重叠 | `CommRouter.sweet_spot` 已按尺寸路由——不动 |
| **多 rank(>2)** | 本机只有两对双卡;暂不需要 | 协议里的 `rank/peer` 用变量而非硬编码 2 |

**本轮边界(用户裁决 2026-10-16)**:本轮**只到设计定稿**,不动代码;
下一轮从 **S1** 开工。


---

## 10. 出处索引

- 公理与里程碑:`charter.md` §二(A2)/ §五(M2)/ §六.2(自研 AR 风险)
- 骨架:`crates/backends/comm/src/lib.rs`(CommCap/CommBackend/CommRouter/Topology)
- iface 窄口:`crates/backends/iface/src/lib.rs`(MemStats 107-108 / malloc_peer_shared 259 /
  enable_peer_access 285 / map_remote 290 / DeviceDesc 64)
- 驱动与窗口机制:`p2p-build/README.md`、`docs/llama.cpp/v3基线不通-立案排查.md` §十、
  `docs/llama.cpp/method3-动态BAR1-P2P-施工文档.md`(§一 原理 / §十四 混插 / §14.7 红线)、
  `docs/llama.cpp/vmm-p2p-3080会战-测试报告.md`(§七/§八 十坑)、
  `docs/llama.cpp/3080-p2p驱动攻坚-setAccess205-定位实录.md`(F0/F1/F2)
- 单进程引擎封死机理:`docs/llama.cpp/p2p-allreduce-后端-施工文档.md`(设计完成待施工)、
  `docs/llama.cpp/patch3-besteffort-cudaMalloc-施工文档.md`
- 对标读数:`vllm.md`(TP2 三线表)、`qwen38-single-3090.md`(单卡)、
  `nex-3080对-三引擎会战.md`(+28.5%)、`docs/exllamav3/*.md`(A/B 后端律)
- owl 自身归因:`roadmap.local/代差分析与演进路线.md`(2026-10-01 逐核分账)、
  回归读数见 `docs/bench/data/owl-main-regress-2026-10-16/`

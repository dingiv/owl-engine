# owl 通用开发坑册(common-dev-pitfall)

> E6 立项补全(2026-10-10)。收录跨会话反复踩的工程坑;每条 = 症状 /
> 根因 / 纪律。性能与 CUDA 语义级坑另见 cuda-pitfalls.md;模型/引擎级
> 结案见 docs/arch/dflash2-施工方案.md §6.20-6.23 与
> docs/bench/qwen38-3090单卡-引擎会战.md。

## 1. GPU 卡序映射(本机常态化)

- **症状**:`OWL_DEVICE=0` 启动 27B 直接 OOM 或打到生产引擎的卡。
- **根因**:本 boot CUDA ordinal ≠ nvidia-smi 物理序。现行映射:
  **CUDA ordinal 0,2 = 生产 3080 对(vLLM 占用)/ 1,3 = 空闲 3090 Ti**。
  vLLM 内部还有 `CUDA_DEVICE_ORDER=PCI_BUS_ID` 重解释(数字 CVD 会被
  重排成混插对,见会战补录钉卡事故)。
- **纪律**:owl 测试/服务一律 `OWL_DEVICE=1`(或 3);启动后用
  `nvidia-smi --query-compute-apps=pid,used_memory` 复核落卡;
  判据 = 双 3090(24G)被占而非 3080(20G)。

## 2. 杀进程:PID 列表纪律(pkill 自匹配事故 ×4)

- **症状**:`pkill -f server` 把自己的 shell/父会话一起杀掉(模式匹配
  自身命令行);或 `kill $(cat pid文件)` 杀错(见 §3)。
- **纪律**:
  1. 永远经 `nvidia-smi --query-compute-apps=pid` 取真 PID 再 kill;
  2. 显存 <300MiB / compute-apps 清零后再启动下一个(旧进程 kill 后
     显存释放有 2-5s 延迟,抢跑 = 新 boot 绑卡 OOM);
  3. owl server(及测试 boot)用 `kill -9`(SIGTERM 不一定生效)。

## 3. `nohup cmd &` 的 `$!` 不是真 server PID

- **症状**:pid 文件里记的 PID kill 不掉真进程(显存照占)。
- **根因**:部分 shell 环境下 `$!` 捕到 nohup/包装层;真进程 PID 偏移
  (实测 299945 vs 299947)。
- **纪律**:启动后以 `nvidia-smi --query-compute-apps=pid` 反查写 pid
  文件;`$!` 只作参考。

## 4. 捕获 slab:满额预留 × 图数 = VRAM 爆(A1 复测实录 2026-10-10)

- **症状**:boot 反复报"捕获 slab 耗尽"(used ≈ cap 差一口气),调大
  `OWL_CAPTURE_SLAB_MB` 后死循环(64→96→128→160 逐级抬高仍差 /
  最终 slab 本体 OOM),propose 图降级 eager(**静默**:只有一行
  "图降级",性能立刻掉 ~19ms/轮,极易当真实数据测下去)。
- **根因**:`graph_begin` 每图预留**满额 slab**,slab 经 carve 输出块被
  图持久引用 → 常驻。图族 = verify + fold×7 + propose ≈ 10 图 × 满额。
  carve 是"填满才报错",used≈cap−ε 是必然现象而非"差一点"。
- **修**(已落,2026-10-10):warmup 计量定量 —— GraphPlan warmup 的
  eager dry 真分配字节 ×1.15 + 1MiB 作 hint,`graph_begin` 消费
  (`owl_shared::slab_hint`;env 显式设 `OWL_CAPTURE_SLAB_MB` = 旧固定
  档兼容)。10 图总预留 1296MiB → ~260MiB。
- **纪律**:spec 服务 boot 后必须核对日志逐图"捕获"行(verify/fold×7/
  propose 全 Captured 才是生产形态;"图降级 EagerFallback" = 数据作废)。

## 5. metrics 动态 tag 无限增殖(mfact 无门控)

- **症状**:`/debug/metrics` 快照里 spec.* 计数器消失 / 前后差分为负;
  512 tag 上限总被占满。
- **根因**:UAF 案取证打点 `mfact(r{n}.pos/m/bonus/d{i}/i{i})` 每轮
  ×9 个**动态 tag** 无限增殖(917 轮 = 8000+ tag),把真账挤出 collect
  上限。
- **修**(已落):`OWL_FACT_PROBE` 门控(案结默认关)+ endpoint
  `/debug/metrics/<prefix>` 前缀过滤 + 投影补 counter 字段。
- **纪律**:metrics tag 必须有界(字面量 tag / 有门控的动态 tag);
  快照差分前先核对 tag 数恒定。

## 6. metrics counter ≠ timer count(TagReport 双字段)

- **症状**:想看 AL(`spec.accepted/spec.rounds` counter)却只读
  `count`,永远 0。
- **根因**:`TagReport.count` = timer 样本数;counter 计数在
  `TagReport.counter` 独立字段(/debug/metrics 投影 2026-10-10 已补)。
- **纪律**:AL 提取 = counter 字段;timer tag 没有 spec.accepted。

## 7. 测量四把尺(全项目统一,违者结论作废)

1. **decode 边际** = 差分法(`gpu_decode_marginal_bench`,Δwall/Δtokens)
   —— 唯一定谳口径;
2. **ITL** = speedtest 明细 itl_stats.mean —— 逐 token 延迟真相;
3. **禁止**裸用 wall÷max_new 或 gate 生成段 wall÷实际 tok 下 decode
   结论(口径像差来源,"45→31.8" 假回退案);
4. 所有 bench 记录带协议标签(长度/prompt 域/采样/spec 开关/池配置/
   **图形态**——eager 降级必须标注)。

## 8. AL 与 tok/s 的口径错配("矛盾"假案)

- **症状**:"AL=4.76 但 tok/s 只有 17-20"式矛盾(A1 立案)。
- **根因**:AL 取自 A 域(gate math 提示词),tok/s 取自 B 域(server
  std 随机词池)——**不同 prompt 域的 AL 不可比**。随机词池 = 最大熵,
  DFlash2 接受率结构性坍塌(真账 AL=0.22,tok/round 1.22),GPU 轮账
  与墙钟闭合(模型 17.2 vs 实测 19.8),无隐藏税。
- **纪律**:AL 与 tok/s 必须同域同请求同 boot 计量(counter 面直读);
  跨域对照先声明域标签。

## 9. nsys 图追踪三模式

- 默认 `--trace=cuda`:图内节点不可见(decode 内核"消失",11393 核全
  Regular,险些误判图回退 eager);
- `--cuda-graph-trace=graph`:聚合单记录(更不可见);
- **`--cuda-graph-trace=node`**:逐节点展开(70460 核 / 59067 图节点,
  §6.21 的 1436 核零空隙分解即此姿势)。
- 附加:`--sample=none --cpuctxsw=none` 降噪音;owl 测试内嵌
  OWL_TEST_DEVICE=1 钉空闲卡。

## 10. worktree 补丁污染

- **症状**:worktree 里的构建产物/补丁改动混回主树,或共享 target 目录
  导致孩子构建互相覆盖二进制。
- **纪律**(承 xinfer s3 教训):每个 worktree 独立 `target/`;补丁只落
  分支不落工作树;审 diff 前先 `git status` 确认干净基线。

## 11. 键控会话无逐出 → GDN 格耗尽(已修,2026-10-10 实录)

- **症状**:家族两症 —— ①键控会话(session id)累积占死 GDN 格
  (8 格),后续 ephemeral 请求(sweep/新客户端)全 400 "GDN 状态格
  耗尽";②格满时**重提交既有会话**也被误拒(slot 扫描在存在性检查
  之前)。曾误诊为"前缀缓存无逐出 → 池满 OOM"(设备侧 alloc OOM
  是另一现象:boot 后首 turn 一次性 +100MiB 懒加载,随后平坦,非泄漏)。
- **根因**:SessionTable 只增不减;get_or_create 先扫格后查存在。
- **修**:①`lru_victim()` + 引擎层 `get_or_create_evicting`(两臂
  共用:键控与 ephemeral 都可逐最老闲置会话,块链归账房,格随 close
  回收);②存在性检查前置(复用不占新格)。metrics = `session.evict`。
- **纪律**:压测脚本若用 session id,格上限 = 并发会话数(GDN_SLOTS),
  超限走逐出属正常;OOM 类报错先分清 块账(格/池)vs 设备 VRAM。

## 12. boot 打印语义坑

- "状态块分配 …槽位 N":N = **单会话容量**(max_seq);池真值 =
  `nb × page`(136 × 32 = 4352)。读账先对齐字段语义(E2 同族)。
- "引擎就绪(捕获态 Captured)":只保证**部分**图 Captured;逐图捕获行
  (verify / fold m=0..6 / propose)须全数在场。

## 13. 内存分类账(27B AWQ 单卡满配,2026-10-10 定账)

| 面 | MiB | 备注 |
|---|---:|---|
| CUDA context/driver | ~430 | 0.8b boot 基线差分 |
| 目标权重(AWQ INT4) | ~14029 | |
| **GDN 状态格** | **~6010** | 48 层 × slots(8)× 3.32MB,**f32 存储** |
| KV 池(4352 tok,16 full 层) | 272 | |
| GDN 快照(SNAP_MAX=4) | ~318 | |
| draft 权重(打包态) | 1321 | U32 0.87G + BF16 0.39G + F16 scales |
| spec 图 slab(计量定量) | ~246 | 修复前 1296 |
| dflash 草稿池 | 92 | |
| blas ws + 杂项 | ~150 | |
| **合计** | **~22882** | nvidia-smi 实测闭合,残差 ~40 |

- 可调杠杆:`OWL_GDN_SLOTS=2` 实测 **−928MiB**(全图 Captured + 冒烟绿,
  slots=2 够 2 并发);f16 化 GDN 状态 = −3GB 候选(需恒等门,未做)。
- 旧账勘误:昨日"杂项 ~1.9G"实为 snap 318 + slab 246 + dflash 92 +
  misc 之和;"draft + 8704 池共存 23.2G"含满额 slab 浪费。

## 14. 池 dtype 改造的写者穷举律(B6.3 双 bug,2026-10-10)

- **律**:把池从 f16 改 1B/elem(e4m3 等)前,先列出该池的**全部写者与
  全部读者**,逐一对账字节宽度。B6.2 改造覆盖了 K0/chunked 读/v2 读三路,
  漏了 **decode 融合插池核 owl_qknorm_rope_kv_insert**(f16 2B 池写落
  2× 字节偏移 = 真槽区永不落笔 + 毒液洒 2s 槽区;E2E 表现为 decode 崩坏
  而 prefill 正常,kv-dump 探针实证)。写者清单:K0 写(三条分派)/
  chunked 读 / v2 读 / FI dual / 融合插池 / NC 自块直读。
- **`__half` 赋 uint16_t 槽 = 数值截断不是位拷贝**(`k_smem[i] = __half(x)`
  隐式走 float 截断;值<1→0、负→UB)。CUDAB 侧位拷贝一律
  `__half_as_ushort/__ushort_as_half`,或经 `__half*` 视图存取。
  探针层面:f16 vs fp8 的 A/B 对拍验不出转换/布局错误(两边同错抵消),
  **核正确性门必须 host 参考**(gpu vs host 同输入同布局逐式)。
- **探针 debug 溢出假失败**:`i as u32 * 常数` 在 debug profile overflow
  panic → 测试从未真正运行("37/37 绿"可能是套件根本没跑到)。
  hash 类 lcg 一律 wrapping_mul。本次两探针(B6.1 v2 fp8 / B5.0 gdn 批)
  修后首次真跑即绿 —— 假失败掩蔽真覆盖。
- **改名 = 删旧符号**:CUDA 核改名时,分派表 match 臂 / registry Entry /
  引用它的测试三处必须同轮清点(fb1e453 改名 owl_naive_attn_nc_f16
  留悬空臂,spec 默认档首轮必炸,潜伏两轮才被抓)。
- **真 PID 判据(运维)**:`$!` 与 `pgrep -f | head -1` 都可能是包装进程;
  唯一可靠 = `nvidia-smi --query-compute-apps` × `/proc/pid/exe` 双验。
  僵尸 server 占端口会让 A/B 测试打到旧二进制(f16 查询打到 fp8 僵尸
  = "fp8 与 f16 同崩"假象,白查一小时)。

## 15. 僵尸 server 的 "(deleted)" 隐身(2026-10-10)

- **律**:kill 循环按 `/proc/pid/exe` 精确匹配二进制路径时,必须剥掉
  ` (deleted)` 后缀(rebuild 后在跑的老进程 exe 全带此后缀)——否则
  永远漏杀,僵尸占卡占端口,A/B 测试打到旧二进制出假数。
- 事故链:rebuild → 老 server 变 "(deleted)" → 精确匹配漏杀 → 新 boot
  OOM 或打到旧代码 → 假信号(f16 查询打到 fp8 僵尸 = "f16 也崩")。
- **正解**:`case "$exe" in *owl/target/release/server*) kill -9 $p;; esac`
  (glob 前缀匹配,天然兼容 deleted 后缀);杀完 `ss -tlnp` 验端口、
  `nvidia-smi` 验卡。
- **套件级联 OOM 假回归**:单跑全绿、连跑 OOM = 前一测试进程的
  设备内存未释放(B6.5 治理下 27B gate 驻留 ~23G,teardown 异步赛跑)。
  不是被测代码回归;单跑逐个验证 + 清卡后再下判决。

## 16. 批量 GPU 测试必须串行(并行 = 多上下文捕获竞争,2026-10-10)

- **律**:同一卡上的 GPU 测试套件(`cargo test -p owl-engine --lib` 等)
  必须 `-- --test-threads=1`;默认并行 = 6+ 个 CUDA 上下文同卡并发图
  捕获 → `CUDA_ERROR_STREAM_CAPTURE_INVALIDATED` / ILLEGAL_ADDRESS 级
  连环假挂(失败集逐轮漂移、单跑全绿)。
- **归因纪律**:批量挂先 `git stash` 在干净树复跑同门 —— 本日实测
  HEAD 同样 31/7 挂 = 存量竞争,非工作树回归;串行 38/38 绿。
  **先定存量/回归再改代码。**
- 孪生坑:pitfall #15 的 teardown 异步赛跑(串行下仍会踩,清卡解);
  本条是并行调度层(串行解)。两坑同象不同根,判据 = 失败集是否漂移。

## 17. 抄编排要审计变量的重绑/别名(g 语义律,2026-10-11 终案)

- **案**:FLA fork orchestrator `chunk_gated_delta_rule_fwd` **首行**
  `g = chunk_local_cumsum(g)` 就地重绑 —— 下游 kkt/h/o 的形参名叫 "g",
  吃的却是 **cumsum 产物**;owl 移植对着核签名接线,喂了 raw g,
  per-token 深谷 −24 经 `exp(g_i−g_j)` 放大成 e²⁴,f16 管线溢出染毒 60 层。
- **律**:移植多核编排时,**只对齐核签名不够** —— 必须通读 orchestrator
  全文,凡"入参在函数体内被重新赋值/别名"的变量,下游每一处消费都要按
  **重绑后的语义**接线。形参名是不可信文档(同名不同物;wu 的
  `g_cumsum=` 才是唯一诚实命名)。
- SASS/ABI 级审计抓不住这一类:参数序、宽度、dtype 全对,只有**数学语义**
  错。防线 = 不变量闸门( cumsum 后 g_cum ≤ 0 全负律)+ 第三方真值(§18)。

## 18. 同源对拍对不出双错 —— 唯一裁判是独立实现的真值(2026-10-11)

- **案**:g 语义案排查五幕,handler 与 venv 探针**同错**(都喂 raw g),
  层 0-2 温和域(g ∈ −8~−12,e⁸≈3000)双方一致地错,"对拍全绿"是双错
  互证;直到 f64 朴素递推真值(独立实现)才把爆炸改判为"输入越域"。
- **律**:对拍参照系必须与被测物**无共享错误源**。引擎 vs 移植来源的
  同构探针 = 同源对拍,只能证一致性,不能证正确性。仲裁序:
  ①独立朴素实现(f64 递推/naive 参照)②金标(§19)③同源对拍(仅回归用)。
- 副律:**数学不变量是免费的裁判** —— "g 全负则 cumsum ≤ 0"、
  "exp(Δ≤0) ≤ 1" 这类恒等式一旦违反即为铁证(本案 +142 冒烟枪出现过,
  被无关矛盾抢走注意力;应升级为机器门)。

## 19. 金标纪律:输入域极值 + 布局代际(gdn 金标修活,2026-10-11 M1)

- **输入域极值**:c1-c4 金标 g ∈ [−0.51,−0.01],引擎真实域深谷 −24;
  "温和域全绿"≠正确(层 4 的 −24 才引爆,教训见补录Ⅷ)。**对拍集必须
  覆盖输入域极值** —— c5_t128_deep(g 至 −23.98,fork 权威,紧容差)
  已入库;raw-g 类回归在 c5 上逐位爆炸、在 c1-c4 上 A/w 静默偏离,双拦截。
- **布局代际**:两代金标布局约定不同 —— pip-fla(c1-c4)h/final = **[K,V]**
  (FLA 习惯);fork(c5)wrapper `new_empty(B,NT,H,V,K)` = **[V,K]**;
  fork h 核 h0/ht/h_buf 全 [V,K](引擎 kv_to_vk 转置即此)。对拍前先核
  金标代际,比较方式按代际分层(测试内 `needs_transpose(tight)`)。
- **换算律**:g_cum 域 —— fork cumsum 输出 **nats**(RCP_LN2 烘核内);
  pip-fla 金标为 **log2**(gen 显式 scale=RCP_LN2)。下游 exp(Δ) 两域
  同值,唯 g_cum 对拍需 `/RCP_LN2` 换算。
- **金标即生产镜像**:金标测试的发射序/dtype 配方必须与 handler 逐一对齐
  (fork ABI:q/k/v/beta=bf16,g/state/A=f32,Ai/w/u/h/v_new/o=bf16 字节);
  两套手写编排必然漂移(本案 golden 喂 gcum、handler 喂 raw g 能各自为政
  正是平行实现的锅)→ 终态 = 金标跑**真实 runtime**(operator-contract
  设计 §4.10)。

## 20. `#[ignore]` 不跳编译 —— feature 门控模块的测试文件要文件级 `#![cfg]`(2026-10-13)

- **案**:`crates/kernels/tests/marlin_bw_bench.rs` 引 `family::marlin`,而该
  模块受 `feature = "marlin"` 门控;`owl-kernels` 默认 features = `[]` →
  **整个 `cargo test -p owl-kernels` 在编译期 E0433**(与文件里那行
  `#[ignore = "bench: …"]` 无关)。
- **律**:`#[ignore]` 只影响**运行**筛选,文件仍进编译面;测试文件引
  feature 门控模块时,首行必须 `#![cfg(feature = "…")]`(先例:
  `ops_cublas_isolation.rs` / `ops_gdn_chunked_chain.rs` 的 `feature = "device"`)。
- 危害等级高:带 `--features X` 跑的人看不到,只有默认档整包红。

## 21. 测试构造的 ctx ≠ 生产 ctx = 静默错值高危区(2026-10-13,kNHD §11.8)

- **案**:`gpu_real_weights_prefill` 用 `ForwardCtx::model_decode(…)`(即
  `EnvProvider::default()` = **全 opt-in 关** 基线)当"参考臂",而生产
  `DispatchCfg` 默认 `qkv_fuse / gdn_fused_decode / gdn_fused_decode_v2` **全真**。
  于是测试的 decode 落到防御臂 `paged_decode_v1_k0`,而 **v1 分页核没有
  kNHD 变体**(4 个 v1 名全是 classic 取址;8 个 `_knhd` 全在 v2)→ 拿 classic
  交错地址读 kNHD 池 = 静默读错元素(logits 差 ~2×),一整个轮次被当成
  "prefill 路径不等价"立案。
- **律**:凡「测试跑 A 臂、生产跑 B 臂」的断言,先核**参考臂是不是生产臂**;
  ctx 装配应走与 engine 同形的构造器/helper(本仓 = `prod_decode_attn`:
  `attn_v2` + `qkv_fuse`),不要把 `EnvProvider::default()` 当生产基线。
- **配套**:层侧对「已无变体的臂」加 fail-fast 断言(kNHD + v1 → panic),
  把静默错值升级为响错 —— 不能被静默炸第二回。

## 22. 归因二选一时,"参考臂自身可疑"必须是候选(2026-10-13;与 §18 同源)

- **案**:§11.8 观测「classic 绿 / knhd 红」,当时只在**被测侧**(prefill 臂)
  找原因;而该观测同样兼容「参考侧(decode 臂)在 knhd 下坏掉」。实验一击
  定谳:注入生产 `AttnV2Scratch` 后本门转绿 → 参考才是真凶。
- **律**:A/B 判别出「布局相关」之后,先问**两侧谁在布局下会变**;被测物与
  参考物都可能布局敏感(尤其参考物复用了同一条产品代码路径)。
- 注意区分 §18(同源对拍对不出双错):本条不是"双错互证",而是**单侧错、
  且错在参考侧** —— 症状与"被测错"完全同形。对拍前的第一句永远是
  **「先证参考清白」**(2026-10-02 k-probe 头注已立此律,本轮二度应验)。

## 23. 跨实现"逐字节等价门"不能建在 greedy 生成上(2026-10-13,FI 直读验证踩坑)

- **案**:验 FI 直读是否健康,设计"同池同写核、只换读核(FI vs 自研
  chunked_opt/split)+ greedy → 输出逐字节一致"。结果:
  - 两个 flash-attention 实现本就有 ulp 级差异 → 不可能逐位等价;
  - greedy 在 OOD 长 prompt 上把这点差异**放大成完全不同的文本**(同一档
    494 tok 天差地别、3722 tok 逐字相同 —— 混沌);
  - 更要命的是**模型自身在长文 prompt 上退化**(27B AWQ-INT4 + 空
    `<think>` 预填,经 HTTP 喂重复填充文本 → 输出 `'a a a a'`/`'?'`/
    词汤复读),两臂**同时**退化 → 判据完全没有分辨率。
- **律**:
  1. **跨实现比较用白盒**:logits/中间张量 + 容差(`[logits-dump]` top-8
     带值可作粗判);greedy 文本只在**同实现**回归里当判据。
  2. **长上下文探针用补全式、且走裸 LM**(引擎 e2e `longctx_4k_prefix_qa`
     头注早就写了:"问答式会退化成文本续写 → 改为让模型续写暗号本体");
     经 chat 模板问"是什么"这类 Q&A 在 OOD 长文上必然退化。
  3. **判据先验分辨率先行**:设计门之前先问"模型在这个输入上还清醒吗";
     否则拿到的红/绿都是噪声(§5/§7 的同类教训)。
  4. 归因警惕:曾据一次 1944 tok 的对比得出"FI 对、自研读核错"的结论 ——
     换长度后两臂行为互换,证明那是**退化噪声**,不是读核差异(已收回)。

## 24. 长 ctx 性能定位:三条工具纪律(2026-10-13,spec 长 ctx 衰减立案)

- **症状**:同一批"慢"结论互相矛盾 —— 分相探针说 verify 197ms,逐核说同一段
  只花 99ms;spec 档吞吐时而 19 t/s 时而 5 t/s。
- **根因三件**:
  1. **分相探针自带 2× 自放大**:`[verify-prof]`/`[dflash-prof]` 的"纯 GPU"判别
     会**同步后再裸 replay 一次**并计入 `[spec-prof]` 的 `verify=`/`propose=` →
     行值 ≈ 真实值 + 一次完整图。判据:`[spec-prof] verify ≈ step + read + pure`,
     read ≈ pure ≈ 单图时长(实测 0.22+98.92+98.86 = 198.0 vs 行值 198.1)。
  2. **B4 降级会掩盖退化**:spec 净亏时自动回落裸档,吞吐被"救"回裸档量级 ——
     不带**同池裸档对照尺**的 spec 读数会把"降级救场"误读成"spec 可用"。
  3. **池 dtype 是隐藏自变量**:同 flavor 同几何,仅 `kv_fp8` 之差,
     ctx 8192 裸档 27.7 vs 41.9 t/s(**+52%**)、verify 注意力核
     4.63ms vs 0.93ms/次(**5.0×**)—— 任何跨会话性能对比先核池 dtype。
- **律**:
  1. **逐核定谳用 nsys,不信相计时**:
     `nsys profile -t cuda --cuda-graph-trace=node`(图回放要看节点)
     → `nsys export --type sqlite` → 按**解码窗** SQL 聚合
     (`CUPTI_ACTIVITY_KIND_KERNEL` + `StringIds`);相探针只作内部相对比较。
  2. **长 ctx 三方对照**:spec / 裸档(同池)/ 逐核账,**三者必须自洽**
     (逐核账 = 分相账 = 实测时长);不自洽就先查探针,别查核。
  3. **Ampere(sm86)上 fp8 KV 是净亏**:无硬件 fp8 cvt → 软件反量化 +
     逐字节标量读,每元素 ~4-6 ALU;且被 (Hq/Hkv)×(伪序列数)×层数 乘性放大。
     fp8 只省流量,而流量从来不是瓶颈。

## 25. fp8 KV 长 ctx 慢 5×:先查**标量读的 sector 放大**(2026-10-13,D3 定谳)

- **症状**:同一条 v2 paged attention 核、同一形状,只把 KV 池从 fp8 换 f16 →
  **4.63ms/次 → 0.93ms/次(5.0×)**;裸 decode ctx 8192 = 27.6 vs f16 42.0 t/s。
- **根因**:不是"fp8 省流量所以不该慢",而是读**宽**:
  - 旧写法每元素一次 1B 标量读 `e.__x = k_ptr8[t]` → 32 lane 落在
    **32 个不同 32B sector**(token 维 stride = hkv·hd = 1KB)→ 取回 32B 只用 1B,
    **sector 放大 32×**(DRAM/L2 侧的成本标签:每次 load 32 sector,而非 1 次 32B);
  - 且 kNHd 页内 dim 维连续,**本可以** 8B/16B 合读 —— 只是代码写成了逐字节。
- **修**(已落,形状不动):
  1. K:1B → **8B 向量读**(vec_idx·VEC_SIZE 恒 8B 对齐)→ 放大 32×→4×,
     verify 98.2→79.3ms;
  2. V:V 的 gather 轴是 **token 维**(跨 hkv·hd,**lane 间不可合并**)→ 把映射换成
     **lane→连续 dim**(每 lane 持 hd/32 个连续 dim,逐 token 读连续 D 字节,
     warp 内 lane 相邻=字节相邻=完全合曲)→ verify 79.3→65.2ms;
  3. 试过 K 再升 16B:**无收益**(→K 已不在带宽墙),按"不留无收益分支"回退。
- **律**:
  1. **同一核换 dtype 快慢反常 = 先数"每次 load 取回几个字节"**,不要先猜 cvt ALU;
  2. **对齐律**:向量读前先算"偏移 ≡ 常数 × lane"的对齐余数(kNHD 的
     dim 维连续是 fp8 唯一可合读轴;V 的 token 轴永远合不了,只能换映射);
  3. **归约守卫**:改 lane→dim 映射后,跨 warp 树归并只有 **warp 0 持全和**,
     写回必须 `if (warp_idx == 0)`(漏了 = 各 warp 部分和互相覆盖,探针立刻抓);
  4. 收益必须回落到**同池裸档对照尺**(裸 decode 27.6→36.4 t/s 才是净收益)。

## 26. 移植第三方核:先查"装载期 static 初始化"与"参数语义"两处(2026-10-13,XQA 移植)

- **症状 A(幽灵版)**:改了 vendored 核的 .cu 并重建 `.a`,二进制行为**不变**、
  `strings` 查不到新串;或**同一二进制 device 0 通过 / device 1 崩**。
  - 根因一:**`ar rcs` 是按名替换** —— 对象文件名换过(`xqa_launch.o` → `launch.o`)时
    旧成员**留在归档里且排在前**,链接器取第一个定义 = 永远链到旧版。
    **律:重建归档前先 `rm`**。
  - 根因二:上游把 `hostSmemSize` 写成**文件作用域 static** ⇒ 在**库装载期**(还没有
    CUDA 上下文)就 `cudaMemcpyFromSymbol` → 失败后 `size` 是**未初始化栈垃圾**,被当
    `dynamicSmemBytes` 发射 ⇒ `cudaLaunchKernelEx` 报 `cudaErrorInvalidValue`。
    栈垃圾逐进程不同 → 表现为"设备相关"的伪规律。
    **律:凡"装载期求值的 CUDA 状态"必须惰性化到首次发射 + 逐步 `checkCuda`**。
- **症状 B(假误差)**:新核 vs host 参考差 ~1e-2,而老核只差 ~5e-4 → 像"精度不够"。
  - 根因:**参数语义错**,不是精度。spec-dec 注意力的 `seq_lens` 是**整请求总 KV 长**
    (batch 维),块内各 query 行的因果结构**只由位掩码表达**;误传逐行前缀长 ⇒ 行 1..K
    各少 1..K 个 key。且"因果"是**闭区间**(含自身位置),host 参考按开区间写就差 1 个 key。
  - **律:分离判据先做"退化输入对照"** —— 把掩码全开 + host 同口径,误差塌到 3e-4 就
    证明"核是对的,参数/口径错";再用 2-4 路变体(位序/极性/行序)A/B 唯一确定语义。
- **通用移植纪律(本次实证)**:①单算子先在 **kernels crate 的 tests/** 里测,不拉上层
  ②先"全开/退化输入"对照分离"语义错 vs 精度错" ③把踩到的契约(布局/strides/掩码/
  尺寸)写进 vendored 目录的 README,别只留在聊天里。

## 27. 跨账本命令 = 块 id 被拿到别家账本解析(2026-10-13,XQA 转正前的最后一雷)

- **症状**:`pump: DeadBlock { id: 11 }`,且**同参同环境时有时无**(改一行配置文本、
  改个缓冲尺寸就换结果)→ 看着像"内存治理太硬/回收器误杀"。
- **定谳路径(可复用)**:①给块账加**分配/切块/释放探针** → 发现 `id=11` 是
  **42.5MiB 权重块、只分配一次、从未 free** ⇒ 不是被回收;②给 `GpuRes::resolve`、
  原生 launch 实参解析、`memset_zero` 三处**各带标记**的报错 → 命中
  `[memset-arg] id=11 len=48`(48B = 我的 semaphores 尺寸)⇒ **id 撞号**:
  同一个 id 在**另一个账本**里是权重块 ⇒ **跨账本解析**。
- **根因**:引擎里有**多个 GPU actor = 多份账本**(`GpuClient::spawn` 各起一个
  `GpuServer`,`face_clone` 才共享同一份);块 id 是**每账本各自从 1 单调计数**。
  我为了"每轮把 semaphores 清零"发了一条 **`memset_zero` 命令**,而这条命令走的
  face 的账本**并不持有**那些池块 ⇒ 解析到别家账本 ⇒ DeadBlock。
- **修法(正解)**:别发跨账本命令 —— 把要清零的缓冲**改成图输入槽**,每轮用
  `step()` 写零(与其它图输入同一条**已验证通路**,且天然捕获安全)。
- **律**:
  1. **一个块只能被它所属账本的 face 操作**:要动池块,先确认 `face_mut()` 拿到的是
     分配它的那个 client(不确定就用**分配时的那个 client**);
  2. **优先用"图输入槽"承载每轮要重置的量**(step 写),而不是发设备命令 —— 槽路径
     与算子实参同源,不会跨账本;
  3. 定位 `DeadBlock` 三步:探针看该 id 是否被 free → 三处解析点各带标记看**谁在解析**
     → 用 id 撞号(尺寸/内容对不上)判定跨账本;
  4. 报错信息里带 **actor 编号 + 核名 + 参序号**(本次已落,零热路径成本)。

## 28. 同一份第三方核编两个实例化:符号要改名,但**别默认"改编译期旋钮"这条臂能用**(2026-10-13,XQA D1)

- **症状 A(幽灵宏名)**:给第二份实例化传 `-Down_xqa_launch=owl_xqa_launch_decode`,
  以为把 `owl_xqa_launch` 改名了 —— 实际 GCC 把 `-Down_xqa_launch` 解析成
  `-D` + **`own_xqa_launch`**(宏名少了首字母 `l`)。两份归档于是**都导出
  `owl_xqa_launch`**,链接器只取一个 ⇒ 第二份实例化悄悄跑到第一份的核上。
  **律:改名前先 `nm -C --defined-only` 对账两份归档的 `T` 符号集合**;
  正确写法 = `-Dowl_xqa_launch=owl_xqa_launch_decode`。
- **症状 B(SPEC_DEC=0 实例化 SIGSEGV)**:上游/fork 的 JIT 对 q=1 走
  `-DSPEC_DEC=0`,我们编出来同一份源码的该实例化在
  `cuLaunchKernelEx` **内部 libcuda 段错误**(`owl_xqa_launch_decode →
  launchMHAFlashInferDec → cudaLaunchKernelExC → cuLaunchKernelEx`)。
  排查顺序(可复用):①gdb 拿栈 → 定位是"host 侧 libcuda 崩"而非核内错;
  ②打印 launch 契约(`hostSmem/grid/cta/ctaTile/launchCfg.dynamicSmemBytes`)
  → **全部正常**(`smem=67200 grid=(8,4,1) cta=(128,1,2) ctaTile=(256,16)`);
  ③`compute-sanitizer --tool memcheck` → 0 error(说不上话);
  ④摘掉另一份实例化单独跑 → **照样崩** ⇒ 排除双 fatbin 同进程干扰;
  ⑤对账符号 → 无重名。
  **结论:不是我们的用例错,是该配置在"我们的编译方式(单 `-arch=sm_86` /
  普通 nvcc)"下落地产物不可用**;fork 用 multi-arch gencode + torch JIT,
  可能压根没在这条路上跑过 sm86。
- **正解(用它)**:保留已验证的 `SPEC_DEC=1` **掩码路径**,只把 tile 行数收窄
  —— `-DM_TILESIZE=16`(`mha.cu:84` 在 SPEC_DEC 下 `nbValidRows = rowsPerBlock
  = M_TILESIZE`;`mha.cu:2907` 有 `M_TILESIZE == 16` 的上游支持痕迹)。
  q=1 只需 6 行 ⇒ `warpTile.y` 从 32 降到 16,MMA 浪费减半,
  **掩码/参数语义与 verify 逐字同一套实现 ⇒ 零新风险面**。
- **律**:
  1. **双实例化优先"同一份已验证语义 + 收窄几何参数",而不是"切换另一条代码臂"**;
     切换代码臂的验证成本高一个量级(本次就撞上不可用的臂)。
  2. **两份实例化必须 `nm` 对账 `T` 符号**(改名前/后各一次),并**实测两份的
     `hostSmem` 等编译期量确实不同**(证明没跨解析 —— 本次:83584 vs 67200 ✓)。
  3. **"调编译期旋钮"这条路要先用最小独立测试证伪/证实**,不要直接接进引擎
     (§26 同源:先 kernels crate 单算子,不拉上层)。
- **另记(负结果,别重复试)**:XQA 的 `nbSubSeqPerSeq = min(SM/(batch·hkv), ·)`
  给 1 CTA/SM;把 `sm_count` 放大 2/3/4×(= 2-4 CTA/SM)**反而更慢**
  (`0.283 → 0.396 ms`,多 split 的归并+尾浪成本 > 占用率收益)。q=1 时 XQA
  对 v2 只有 **1.46×**(对比 q=8 的 4.72×)—— 因为 v2 以 384 CTA 超高占用把延迟
  藏住了,XQA 的"元素数 ÷6"优势被每 split 固定开销摊薄。

## 29. 「按 exe 路径认进程」在重建后会静默失效(` (deleted)` 后缀)——A/B 被旧 server 污染(2026-10-13,② 验收)

- **症状**:同一个 A/B 序列跑出互相矛盾的结论(某次 B 的读数与 A 完全一致),且
  重跑结果漂移;事后 `pgrep` 才发现**上一轮的 server 一直活着占着 18G 显存**。
- **根因**:清理脚本用 `readlink -f /proc/$pid/exe` 与目标二进制路径做**字符串相等**
  比较 —— 只要该二进制在进程存活期间被 `cargo build` 覆盖过,`readlink` 就变成
  `/path/server (deleted)`,比较**永远为假** ⇒ 循环静默跳过 ⇒ 旧 server 不死。
  更毒的是:它不报错、不崩溃,只是让后续每个测量都跑在**被占用的卡**上
  (起不来的会 OOM 报错,起得来的会给出貌似正常的读数)。
- **律**:
  1. 认进程时先剥后缀:`readlink -f /proc/$p/exe | sed 's/ (deleted)$//'` 再比较;
  2. **每次起 server 前先核显存**(`nvidia-smi --query-gpu=index,memory.used`),
     等到目标卡 <100MiB 才继续 —— 显存是唯一可信的"卡是否空"判据;
  3. A/B 脚本必须**自证环境干净**(打印起跑时的显存),否则结论作废。
- **附带**:`[boot] vram 治理: ... free=` 里的 `free` 是**设备线程缓存值**
  (`owl_shared::vram::FREE`,`cuMemGetInfo` 落账后覆盖),**不是实时查询** ——
  同一配置两次 boot 可差 17GB(实测 A=23.32G / B=5.56G 而真实分配逐位相同)。
  **别用这一行做 A/B 的内存差异判据**,要看 `live=`/`fixed_after=`/`per_tok=`。

## 30. 测速工具:`usage` 拒用陷阱 —— 启发式估算**覆盖**了正确的真账(2026-10-13,② 验收)

- **症状**:`llm_speedtest` 报 `out 1 / 8 / 110`(期望 256)+ `⚠token统计异常`,看着像
  "引擎早停"。但那些行的**墙钟时间与正常行完全一样**(6247ms vs 6209ms),ITL 也是
  正常量级(68.8ms)⇒ 生成显然跑满了。
- **定谳**(`-v` 透传内核日志,同一次运行两行对照):
  ```
  行1(异常): API usage completion_tokens=256 ✅ / 工具累计内容=117 字符 / 本地估算=8  → 报 8
  行2(正常): API usage completion_tokens=256 ✅ / 工具累计内容=942 字符 / 本地估算≈256 → 报 256
  [TokenWarning] Decode API stats look abnormal; api=256, local=8, diff=3100.0%. Using local estimate.
  ```
  **服务端没错**(curl 原始 SSE:10/10 次 `completion_tokens == max_tokens`、
  `finish_reason=length`、`ignore_eos` 生效)。错的是工具的 sanity 检查——
  它拿一个**字符/词启发式估算器**去"校验"引擎的 usage,不一致时**用估算覆盖真账**,
  与工作区自己的口径律(`token 计数必须用 usage.completion_tokens`)直接冲突。
- **触发条件**:随机词池域让模型整段**退化成复读/空白** ⇒ 字符/token 骤降
  (256 token 只产出 117 字符)⇒ 启发式(忽略空白、按字/词计数)严重低估 ⇒ 触发
  偏差判据。**这不是引擎行为,是"域退化"在输出侧的表现**,此前只记录了域退化对 AL
  的影响(§2.3/§B4),没记它对**输出文本**的影响。
- **修**:`resolve_token_count_with_sanity_check` 增加 `hard_upper`(单响应物理上界
  = 请求的 `max_tokens`),判据从"统计"改"结构":
  - `api <= hard_upper` ⇒ **以 usage 为准**(不符只告警,不改值);
  - `api > hard_upper` ⇒ 物理不可能(累计总量型服务端,llama.cpp 并发)⇒ 才回落本地估算。
  另外把两处告警分开计数/显示(`token记账异常` vs `输出与估算不符(以usage为准)`),
  并把 prompt/out 的"来源"标签改成**逐字段**(此前 decode 一告警,连 prompt 列也显示
  "本地估算",误导判读)。
- **律**:
  1. **真账(服务端 usage)永远优先于本地启发式**;启发式只能在真账"结构性不可能"时兜底;
  2. 看到 `out`/`decode 速度` 反常,先核**墙钟时间与 ITL**——时间正常 = 生成正常,
     问题在记账,不在引擎;
  3. 测速口径的三件套一起看:`usage` 真账 + 累计内容长度 + 墙钟时间,任一不符即先怀疑工具。

## 31. 「用 PID 列表杀进程」= 杀**你自己的** PID,不是枚举全卡 compute-apps(2026-10-13,审计事故)

- **现场**:跑生产算子审计后清场,顺手写了
  `nvidia-smi --query-compute-apps=pid --format=csv,noheader | while read p; do kill $p; done`
  —— 这条命令把**整机所有 GPU 计算进程**都算进了"我的进程",于是把另一个引擎
  (vLLM,3080 对)的 TP worker 一起杀了:生产服务当场下线(GPU 0/3 显存归零、
  8134 端口无监听)。
- **根因**:工作区纪律写的是"杀进程用 **PID 列表**(别用 `pkill -f`,它会匹配自身命令行)",
  而我把"PID 列表"误读成"全卡枚举出来的 PID 列表"。`--query-compute-apps` 是**机器视角**,
  不是**本次任务的视角**——同一台机器上永远可能有别的引擎(本工作区常态:vLLM 占 3080 对)。
- **正解**:
  1. 只杀**本次会话自己拉起的** PID(`$!` / 明确记下的进程号),杀前先 `ls -l /proc/<pid>/exe`
     确认 exe 路径 + `ps -o pid,etime,cmd -p <pid>` 确认是自己的;
  2. 需要"确认自己进程真的死了"时,枚举 compute-apps 只用来**核对**、绝不用来**下刀**;
  3. 必须清场(如换档)时,先 `ps` 看 cmdline,按 **exe 路径前缀**过滤(如 `target/release/server`),
     再逐个核对;
  4. `nohup cmd &` 的 `$!` 是**子 shell** 的 PID,不是 `cmd` 的 PID —— `kill $!` 杀不掉
     `cmd`(本次审计里 `kill 2261585` 就是这个坑,真进程是 2261586,结果"以为杀了"的
     服务器还占着 20.5G 显存,下一轮启动 OOM)。**拉起后立刻用 `/proc/*/exe` 反查真 PID**。
- **律**:**"清场"必须限定在自己的进程树内**;跨引擎共享的机器上,"全卡枚举杀"= 越权操作。
  破坏生产服务后第一时间**恢复原状**(本次:原命令 `uv run qwen-server --flavor <档>` 重启 +
  `/health` 200 + 真实请求冒烟),再继续手头工作。

## 32. 族的"名字"与"算子表"是一对**方向性**约束,写成相等会炸在正常路径(2026-10-13,P3 B2)

- **现场**:给 `KernelFamily` 加注册语义 `ops()`(本族已建表的算子)后,把
  `names()` 改成"由 `ops()` 派生",并在 boot 里加了"两处必须同票"的门
  (`names.len() == ops.len() && 双向包含`)。结果**生产档直接拒启**:
  flashinfer 族有 2 个名字(`PREFILL_FI` f16 + `PREFILL_FI_FP8KV` fp8),
  但只有 fp8 建了表 ⇒ `ops`(1) ≠ `names`(2) ⇒ Err。拒启日志还被
  5 条"未经 close 即 drop"的兜底告警淹没(见 §33)。
- **第二层**(改完方向后仍踩):把 `names()` 默认实现写成"`ops()` 非空则
  **纯派生**" ⇒ **部分建表**的族丢掉主名:gdn_scalar(`ops` 空)与 FI 的
  f16 主名从路由表消失 ⇒ 该臂会回落原生 NVRTC 路径后必然失败。
  实测:家族臂数从 11 静默变 10(不报错、不告警,只少一个名字)。
- **正解**:
  1. 门的方向 = **`ops() ⊆ names()`**(已建表的算子必须能被路由到),
     绝不写成相等 —— "族内还有未建表算子"是**正常状态**(P3/P4 并存期常态);
  2. `names()` 默认实现 = **族主名 `id()` ∪ `ops()` 各算子名(去重)**:
     全量建表的族零手列、部分建表的族自动"主名 + 已建表项"、未建表的族只剩主名;
  3. 两处都加**回归锁测试**(`names_is_id_union_ops` / boot 期门),因为这两类
     错误都是"静默少名字",不写门就等着在某个不常跑的臂上炸。
- **律**:**注册表类的"两处声明"必须明确方向**,而且"少一项"必须能红 ——
  `⊆` 是真不变量,`==` 只是此刻的巧合。

## 33. 拒启(错误)路径同样要收尾已建成的资源(2026-10-13,同案)

- **现场**:`OpRegistry::init_all` 逐个装配族;第 6 个族拒启时,**前 5 个已建成的族
  被裸 drop** ⇒ 每个都刷一条"未经 close() 即被 drop"的兜底告警(§4.4 的同律反例:
  那次修的是**正常退出**路径,这次漏的是**错误退出**路径)。
- **正解**:`init_all` 里每条 `return Err(..)` 之前都 `close_all(families)`
  (含正在被拒的那个族:`let _ = f.close()`)。teardown 无法上抛 ⇒ best-effort,
  但**噪声归零**。
- **律**:**兜底告警的静默性要在"正常 + 拒启"两条路径上都验证**;
  只测正常路径 = 拒启时反而最吵(那时正是最需要看清原因的时刻)。

## 34. "生产里到底在跑哪些算子"此前无人知道 —— 白名单硬门把这件事变成可判定的(2026-10-13,P4)

- **病灶**:P3 审查生产算子时,唯一可靠手段是**真跑一遍 + `[cuda] launch_time`
  逐发射记账**(46 个算子、13108 次发射)。也就是说:"本档位会发射哪些算子"
  这件事**没有任何静态声明**,只能靠日志数。后果有两个:
  ① 未建表(仅引入)的算子可以随时进生产,没人拦;
  ② 池写集穷举(E10"漏写者 ⇒ 槽区全零")**无法审计** —— 不知道谁在跑,
  就不知道"谁该写这个池"。
- **正解**:`dispatch.allowed_ops`(算子名白名单;非空 = 门开)三处消费:
  ① **装配期**:列出的算子必须都已建表(否则拒启)+ 用生效集合跑池写集审计;
  ② **发射期**:未列出的算子发射即结构化拒(deny-by-default);
  ③ **测试**:flavor 的清单与实测台账常量(`ops::PROD_*_2026_10_13`)逐项
  一致 —— 两处声明不许漂移。
- **实测**:去掉清单里的一个算子(`owl_sigmoid_gate_mul_f16`)⇒ boot 的
  capture warmup 阶段当场拒启:
  `[算子准入] contract owl_sigmoid_gate_mul_f16: allowed_ops 期望 ∈ 本档白名单(43 项)实得 未列入`。
  这正是"拒启"的落地形态 —— **boot warmup 就是预演**,凡是模型真正需要的算子
  都必然在 boot 期被发射一次。
- **律**:**"哪些东西被允许出现在生产"必须是一份可判定的静态清单**,
  而不是日志里数出来的事实;清单本身还要有门锁(与权威台账逐项一致),
  否则它会随人手改动而静默失效。

## 35. 声明驱动 + 未知容忍:几何门不能因为"不知道"而误伤(2026-10-13,P4)

- **现场**:给算子加 `ShapeFamily`(头维/页/分组比/实例化上界)声明后,门要在
  `driver::resolve` 里判"当前模型几何是否被接受"。踩到两个设计点:
  1. **未知必须放行**:单测/CPU 面/后端自述面拿不到模型几何(0/`None`),
     若把 0 当成"不匹配"⇒ 全库误伤。正解 = **`0`/`None` = 未知 ⇒ 跳过该项**,
     由真正知道事实的那一层(引擎 boot 的 `ModelDims`)填值。
  2. **池身份不是全局几何**:主池 fp8 + 草稿池 bf16 并存 ⇒ "池 dtype/布局"
     无法用**一个**全局字段表达(写谁才知道)。若硬塞进 `OpEnv`,草稿池写者
     会被主池事实判死。正解 = 池 dtype/布局归 `PoolAccess`(按池实例声明),
     由**写者/读者互证审计**管;`ShapeFamily` 只管模型级几何。
- **律**:**门的输入必须是"确实知道的事实"**;把"不知道"当"不匹配"会把门
  变成误伤源,把"多值事实"塞进单值字段则会把门变成摆设。

## 36. 自检(oracle)必须有"判据敏感性负例" —— 否则"通过"可能只是输入全零(2026-10-13,P5)

- **现场**:P5 的装配期 oracle 写完后,写了三条 GPU 测试:正例(oracle 通过)
  + 两条**判据敏感性负例**(把参考故意做错,要求判据能红)。负例当场抓出三件事:
  1. **测试 harness 自身坏了**:测试用 `DeviceRes` 实现里图省事 `clone()` 了
     `CudaSlice` 再 `memcpy_htod` —— cudarc 的 clone 是**深拷贝**,数据写进了副本,
     设备缓冲永远是零。于是 oracle 拿到**全零输入**,正例"通过"(0 vs 0),
     NC 的两条参考也都算出 0 ⇒ 敏感性负例显示"关滑窗也不敏感" ⇒ 破案。
     **若只写正例,这里会得到一个全零假绿的自检门**。
  2. **参考的语义细节**:e4m3 的 **±0.0 符号**必须保留(硬件 `__nv_fp8_e4m3(-0.0)`
     = `0x80`;负次正规量下溢到 `-0.0`)。逐字节判据立刻报出 1 处 `0x00 vs 0x80`
     —— 既有探针的输入刻意避开微小量,**从没碰到过这个边界**。
  3. **判据的分辨率**:NC 的滑窗只咬掉 8/2056 行时,关窗参考的偏差 `8.8e-3`
     **落在 bf16 噪声(3.9e-3)量级之内** ⇒ 该 oracle 结构上分辨不出这类小窗
     偏差。把算例改成咬掉 512 行后偏差 `1.119e-1`(28× 噪声)⇒ 判据这才真的有效。
- **律**:
  1. **每个 oracle 都要配一条"故意做错"的负例**,并断言判据**会红**;
  2. 负例要同时验证"正参考不红"(否则可能是输入退化,如全零);
  3. 判据的**分辨率**要写进文档:能区分多大的偏差、不能区分什么
     —— 别把"通过"当成"绝对正确"。

## 37. 独立参考最容易错的地方:**写者分支结构**(2026-10-13,P5 融合插池 oracle)

- **现场**:给 `owl_qknorm_rope_kv_insert_f16_fp8kv_knhd` 写 host 参考时,按"每个 d
  各自算 rope"直写(看起来最自然),实测 `max|err| = 3.914`(值量级才 ~1.15)。
- **根因**:核内 q 路径**只有两个写者分支**:
  `d < half` 的线程**一次写两个位置**(`base+d` 与 `base+d+half`),
  `d ≥ 2·half` 直通;`[half, 2·half)` 自己**不写**(由配对线程代写)。
  参考按"每 d 自算自写"就把伙伴位写成了自己的值 ⇒ 结构性错位。
- **连带**:修好之后立刻暴露第二个问题 —— 容差口径。核内用 `rsqrtf` + 树形归约,
  host 用 `1/sqrt` + 顺序求和,**同一算式的 f32 结果天然差 ~1%**(f16 域下更明显),
  绝对阈值(`2e-3`)会把噪声当错值;改**相对阈值**(`|g−w|/|w| ≤ 1e-2`)才对。
- **律**:
  1. 写独立参考时,**先逐字读核的"谁写哪个位置"分支结构**,再写数学 ——
     数学对而写者结构错 = 参考系统性错位(而且看起来"差不多",极难发现);
  2. 参考容差要按**数值路径**定(是否含近似指令/归约序),别用一把绝对尺量所有算子;
  3. 这两点都由"判据敏感性负例"兜底:参考换历史 bug 形态(rope 配对互换)后
     `max|err|` 从 `1.95e-3` 跳到 `4.478`(2300×)—— 判据确实在测真东西。

## 38. 拒启被吞 = 只剩"族被 drop"的兜底告警;两条 upload 语义不能混(2026-10-13,P5c)

- **现场**:把装配期自检的调用点从 `init_all` 内部搬到注册表建成之后(P5c:foreign
  算子 oracle 要经注册表 `dispatch`),生产 boot 突然**只打两行自检就停住**,后面是
  7 条"族未经 close 即被 drop"的兜底告警,**没有一行错误原因**。
- **两个根因**(都值得永久记下):
  1. **拒启不可见**:`GpuServer::run` 里注册表 boot 的 `Err` 只 `return Err(...)`,
     而设备 boot 通道**此前已经 `send(Ok)`** ⇒ 错误既没进通道也没打印,进程挂在
     "等就绪"上。**修**:该错误路径加 `eprintln!`(拒启必须可见);
     同时自检**跑之前先打算子名**(失败/挂起时一眼定位)。
  2. **`DeviceRes::upload` 有 scratch-only 白名单**:它的 dst 是裸指针,服务端
     `find(|(p,_)| *p == dst)` 只在 scratch 账本里找 ⇒ 往**账本块**灌输入时
     报 `upload 期望 指针在 scratch 账本 实得 0x…`。**修**:新增
     `DeviceRes::upload_block(&Bytes, &[u8])`(块句柄 → `resolve` → 同步 htod),
     与 `upload` 并列而非合并 —— 两者语义不同(scratch 写者白名单 vs 账房在册块)。
- **律**:
  1. **拒启路径必须有可见输出**(§33 同族:那次修的是"噪声",这次修的是"沉默");
     启动门失败时,人看到的第一条信息必须就是原因;
  2. 加"跨层新原语"时先读**旧原语的校验白名单** —— 复用旧原语而不知其隐含
     约束,会把约束错误伪装成业务错误(`upload` 的 scratch 白名单就是这样撞上的)。

## 39. 补丁脚本静默不命中 = 门静默缺失(2026-10-13,P5d)

- **现场**:给 marlin 算子挂装配期 oracle 时,用脚本把 `SELFCHECK` 常量插进
  `ops/marlin.rs`。脚本里那一条 `replace` **没有断言命中**(前几条有,这条漏了),
  实际没改到文件。结果:oracle 代码、测试、`cfg` 模块全都就位,**生产 boot 里
  就是不跑 marlin 自检** —— 日志看着全绿(其余 4 条自检照跑),没有任何异常信号。
  是"数自检行数发现只有 4 条(应为 5 条)"才抓到的。
- **律**:
  1. **任何文本补丁脚本的每一处替换都要 `assert` 命中**(命中数为 0 时必须报错);
     静默 no-op 的破坏力远大于报错 —— 它让**门静默缺失**;
  2. 门类改动**要数数**:"应当跑 N 条自检/应当有 M 条断言",数字对不上就是漏了
     (本棒靠这一条抓到);
  3. 同类:**门的清单/覆盖数**最好由机器从权威表算出(如本项目的
     `PRODUCTION_UNION ⊆ all()`、`41 张表互证`),而不是人眼数日志。

## 40. 声扫/判定口里调"会物化的解析" = 懒物化被提前成全量物化(2026-10-16,装载提速)

- **现场**:给 `AwqSource` 补 `native_dtype`(设备 cast 臂判定口)时,为了"与
  `copy_native_into` 同源",实现里写了 `self.resolve(key)`。而 `resolve` 对派生
  键(`.scales`/`.zeros`/`.qweight`)会**当场物化整键**。`load` 的暂存区声扫
  (`LoadStaging::plan`)对**每个** Want 问一次 `native_dtype` ⇒ 装载一开始就把
  352 个 base 的派生键全建了(27B 档 ≈14GB 主机内存)。
- **症状**:不是错值,而是**慢** —— 键循环 2.82s 被拉到 6.31s(3.5s 花在声扫里),
  且 `resolve` 的懒物化语义被破坏(内存峰值前移)。
- **律**:
  1. **"廉价判定口"(声扫/类型探测/能力自述)禁止触发副作用**(物化/分配/IO):
     要一致性就写**谓词镜像**(查索引 + 排派生键名),不调执行器;
  2. 判断"哪一段慢"用**分账**:本棒是 `eval_load 分相`(键循环 vs 总栅栏)+
     `键循环分账`(总时长 vs `load_group` 累计)两步把 3.5s 钉在声扫上;
  3. 任何"看起来只是判定"的新口,先问一句:**调用点是谁、每载跑几次**
     (本例:声扫 = 每 Want 一次)。

## 41. 逐键 host 排空 = 把设备管线斩成 N 段;"每键"与"每 N 块"要分清(2026-10-16,装载提速)

- **现场**:直接臂的排空判据原为 `off.is_multiple_of(chunk_elems*4) || off >= n`。
  第二项 = **每个键**都排空一次(27B ≈1050 个小键)。CPU 慢时它是免费的(设备早已
  追平),CPU 一快(本轮把 convert 从 16.7s 压到 1.7s)它立刻变成主项:**键循环
  host 8.3s 里 5.9s 是等设备追平**。
- **修法**:排空判据改成**跨键累计在飞字节**(`note_inflight`,阈值 4×CHUNK_BYTES),
  小键不再各排一次;正确性由 ①server 侧 pinned 租约**流完成回调才还池** ②复合口的
  设备端事件序 ③`eval_load` 末尾的总栅栏 三层兜住(逐键 `sync` 从来不是语义必需)。
- **律**:
  1. **"每键一次"的同步/分配/往返,在键数上千时就是 O(N) 的墙**;判据要写成
     资源量口径(字节/在飞数),不要写成"每个键";
  2. **优化 CPU 会把下一堵墙从设备侧搬到 host 侧**(反之亦然):提速后必须重跑
     分相账,别拿旧结论继续调参(本棒四轮:convert → 声扫 → 排空 → 段规);
  3. 段规(流式分段大小)存在**拐点**:16.8MB 段 host 拷贝 5.6GB/s 且抖动大,
     33.6MB 段 10GB/s 且抖动归零 —— 小段不是"更省内存就更安全",它同时变慢、变抖。

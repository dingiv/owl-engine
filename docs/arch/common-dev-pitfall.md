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

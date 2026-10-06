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

## 11. 前缀缓存无逐出 → 池满 OOM(待修,2026-10-10 实录)

- **症状**:std 扫长跑完后,后续任意小请求 400 `alloc(0.2MiB) OOM`。
- **根因**:前缀缓存块常驻池(4352 池),多请求后缓存占满,新 turn
  分配失败;引擎未崩(abandon 可恢复)但服务面拒绝。
- **纪律**(暂):长扫压测后重启 server 再做精度/功能验证;
  正解 = 缓存逐出(LRU/压力驱逐),挂 B4/B7 前置。

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

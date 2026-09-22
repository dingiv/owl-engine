# backends —— 后端 HAL 与准入纪律

```text
iface/   统一契约:Backend(厂商栈:枚举/打开) + Device(卡:账本/池/相位)
         + DevBuf/MemPhase/MemStats/Pool* —— 上层唯一入口
cuda/    CUDA 栈(cudarc 官方封装):CudaBackend + OwlCuda(Device),第一实现
rocm/…   未来厂商栈:实现同一 iface,注册即可,上层零改动

三元关系:Backend(厂商 API 家族) 1:N Device(具体卡);
         Device = 账本 + 池组 + 相位机的唯一宿主;Pool = Device 账本的红线。
```

依赖方向:iface 不依赖任何后端;后端只依赖 iface(与 kernels/graph/comm 平级)。
上层(core/server)禁止直接 include 任何后端内部类型。

---

## ⚠️ 第三方库 CUDA 准入铁律(A5.2 的执行细则)

**任何能够直接触达 CUDA 的第三方库,引入前必须源码审核以下问题:它是否绕过
后端账本私自调用 `cudaMalloc` / `cudaMallocAsync` / `cuMemAlloc*` / 池分配 /
cuBLAS 懒 workspace 等,自行产生"账外显存"。**

- 审核结论为**会** → 二选一:
  1. **不用**这个库;
  2. **copy 其代码进我们自己的 crate 改造**,把分配改道后端账本
     (所有权按 charter A4 显式认领:parity 硬门 + 差异清单 + 只 cherry-pick)。
- 审核结论为**不会**(分配全部经我们出口,或可预钉)→ 允许依赖,登记进
  `VENDOR.list.md` 的审核栏(结论 + 审核日期 + 关键证据行号)。
- 审核方法(最小清单):
  ```
  grep -rn "cudaMalloc\|cuMemAlloc\|cudaFreeAsync\|cublasSetWorkspace\|cudnnSetWorkspace\|malloc_async\|alloc_zeros\|htod" <crate>/src
  ```
  重点:构造函数/首次调用路径上的**懒分配**;Drop 路径上的**自由释放**;
  第三方 FFI 里打包的静态分配。A5 下没有"库自己管自己的内存"这回事。

### 已审核结论(持续维护)

| 库 | 结论 | 证据/说明 |
|---|---|---|
| cudarc(官方 0.19) | ✅ 通过 | 分配仅在显式调用 `alloc*` 时发生,无懒分配;context 常驻底价已计入 `Budget.runtime_floor_bytes` |
| candle fork(candle-gb @ 23a6f38) | ✅ 通过(带焊缝) | 账外裸分配:无;BLAS workspace 32MiB 池外预钉 = "预登记"正面教材(device.rs:29) |
| attention-rs(@ c0f19f2) | ⚠️ 有条件通过 | ① moe.rs:2602 **自曝**:MXFP4 prefill 路径经 cudaMallocAsync 分配 scratch 且**与图捕获不兼容**——我们不用 MXFP4;该路径若启用必须改道账本+出图;② workspace.rs:151 cuMemAllocHost_v2 是 pinned **host** 内存,不违 A5(设备预算),但需 host 账本另记;③ 其余 device 分配走 candle/图池。M1 接入时按清单复审 |

新库不登记不依赖;结论变更(升级快照/上游行为变化)必须重审并更新本表。

---

## ☠️ CUDA Graph 死亡姿势图鉴(判例汇编,2026-09-22 定稿)

> 来源 = 本工作区全部实战卷宗(mistral UAF 案、xinfer 456524b 泄露案、
> candle/cudarc fork 焊缝、owl T4 实测)。每条姿势:症状 → 根因 → 判例
> → 防线(回链 charter 公理)。**新 graph 相关 bug 先来这里对号,再动手。**
>
> **本质分类(2026-09-22 归约,经 NVIDIA DL CUDA Graph troubleshooting
> 官方文档交叉验证——memory-issues / capture-failures / dynamic-patterns
> / numerical-errors 四板块与我们一一对应)**:graph 把程序四个维度永久
> 冻结(地址/控制流/资源承诺/形状),每类死亡 = 某个冻结维度被解冻:
>
> | 类 | 冻结 vs 解冻 | 姿势 | 典型症状 |
> |---|---|---|---|
> | Ⅰ 地址生命周期违约 | 冻结地址 vs 自动内存管理 | 2,3,4,10 | UAF/Xid/静默损坏 |
> | Ⅱ 捕获期合法性违规 | 冻结执行序列 vs host 逻辑 | 1,9 | 捕获作废/错序 |
> | Ⅲ 资源承诺违约 | 冻结预留 vs 事后伸手 | 5,7 | OOM/互踩 |
> | Ⅳ 静态性 vs 动态性 | 冻结形状 vs 动态形状/懒初始化 | 6,8 | 首跑异常/旧形状 |
>
> 一元本质:**录制时代的地址承诺,必须在回放时代依然有效**。
> 四类 = 按"谁破坏了承诺"分:内存系统破坏(Ⅰ)、录制期夹带私货(Ⅱ)、
> 预算没留够(Ⅲ)、世界变了承诺没更新(Ⅳ)。

### 姿势 1:捕获期隐式同步/回读 → 捕获作废
- **症状**:`end_capture` 返回 invalidated/`CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`;
  或 legacy NULL stream 上 begin 直接不支持
- **根因**:捕获段内出现同步点(D2H 回读、`item()`/`to_vec0`、事件等待)
  或捕获了 legacy stream
- **判例**:owl T4 实测 `default_stream()` = legacy,begin 即拒;candle 的
  `to_vec0` 一进段即废
- **防线**:A1.5 kernel 契约;ops 只接 `KernelCtx`;显式 non-blocking
  stream(owl.md §四-M1①);`EagerOnly` 类型标记禁入捕获段

### 姿势 2:replay 悬空地址(最经典的 UAF/Xid 族)
- **症状**:replay 时 `CUDA_ERROR_ILLEGAL_ADDRESS` / **Xid 31 MMU Fault
  (FAULT_PDE VIRT_READ,可双卡连环)**;或更毒的——无崩溃、静默写坏
- **根因**:捕获期烘进图的地址(参数缓冲、中间张量、host 暂存),
  之后被 free/复用
- **判例**:candle 每次 launch htod 新参数(其 fork 的 `CUDA_PARAM_CACHE`
  就是补这个);mistral.rs "CUDA graph 烤死指针 × cuMemPoolTrimTo" 案
  (插桩 R1 拍到全程,R2 禁 trim 后 40/40 零崩)
- **防线**:A1.5 + cudarc/candle 焊缝(捕获期池分配 + Drop 禁 free)+
  A5.2 一切分配过账本

### 姿势 3:图存活期 free/trim(账外释放)
- **症状**:replay OOM(物理页被撕)或间歇性非法地址;特征是
  "启动正常、跑一阵子炸、且与分配时机强相关"
- **根因**:图实例化后对其依赖的池做 `cuMemPoolTrimTo`/`empty_cache`,
  AUTO_FREE_ON_LAUNCH 延迟归还的块被物理回收,下一档 replay 要原地址
- **判例**:xinfer 456524b(DEDICATED_POOL + trim博弈,verify 图懒提交
  4.2GB 窗口);mistral cuMemPoolTrimTo(同族)
- **防线**:**A1.2 生命周期律**(trim 只在净空窗口/全销毁后),
  owl-cuda `GraphGovernor` 已把非法迁移做成类型错误

### 姿势 4:库懒分配进图 → 静默数值损坏(最毒,无崩溃)
- **症状**:输出偶发错误但零报错;特定形状触发(split-K)
- **根因**:cuBLAS 等库首次调用时自己内部 alloc workspace → 恰在
  捕获期发生 → workspace 变 graph-owned,地址进 handle;池回收后
  split-K 写坏无关张量
- **判例**:candle fork `BlasWorkspace` 注释(原文"corrupt unrelated
  tensors on replay",mirrors PyTorch)
- **防线**:启动时池外预钉 32MiB + `cublasSetWorkspace`
  (owl `cublas.rs` 已移植);一切库句柄创建时点纳入 A5.2 登记

### 姿势 5:图内存无预算 → 捕获 OOM / 懒提交博弈
- **症状**:捕获时 OOM;或首次 replay 懒提交巨量物理页(实测 ≥4.2GB)
  与运行时分配互踩
- **根因**:池计算不知道图的存在,图内存"事后伸手"
- **判例**:xinfer 池计算器把省下的显存全喂 KV(0.92→0.89 事故);
  vLLM 池 453k→489k 后 draft 捕获 20MB 即 OOM 僵死
- **防线**:**A1.1 预算前置**(graph_allowance 是规划输入)+
  A1.4 捕获预检降级;owl 已由 `GraphAllowance` + `validate_budget` 落地

### 姿势 6:缺 warmup → 捕获失败或首跑异常
- **症状**:捕获段首 replay 慢(PTX JIT/module load 混入)或捕获失败
  (懒初始化含非法操作)
- **根因**:kernel module 加载、cudnn/flashinfer plan、cublas 句柄
  初始化等懒状态未在捕获前逼出
- **判例**:owl T4(warmup 制度化:捕获前 eager 全链先跑);
  candle/vLLM 的多 phase 捕获(warmup→prewarm→真捕获)同源
- **防线**:捕获三阶段制度(warmup→prewarm→capture),M1 写进
  GraphGovernor 流程

### 姿势 7:多图/多池互踩
- **症状**:A 图 replay 把 B 图的页顶掉;`cuDeviceSetMemPool` 切换后
  别人的分配进错池
- **根因**:每图独立池互相博弈提交窗口;或全局池切换影响所有 context
  分配
- **判例**:xinfer 456524b(DEDICATED_POOL 全局切换 + verify/decode
  双图并存)
- **防线**:**A1.3 单共享捕获池**(峰值≈最大一图);池切换禁止

### 姿势 8:形状/档位热更未走 ExecUpdate → 旧形状重放
- **症状**:replay 输出形状/内容错;或为多档各存一份 exec 撑爆显存
- **根因**:图把 kernel 配置焊死,运行时换形状/换 kernel 选择必须
  重捕获或 ExecUpdate
- **判例**:ninfer 的 profile/topology 分档 + `cudaGraphExecUpdate`
  热切换(摘樱清单在案)
- **防线**:档位预捕获(owl `GraphProfile`);ExecUpdate 列 M1+ 候选

### 姿势 9:跨流捕获边界未 join → 部分捕获/错序
- **症状**:end_capture 报 "operating on captured stream" 或依赖丢失
- **根因**:多流依赖未用事件 join 进捕获子图;或在捕获中触碰
  未捕获流
- **判例**:xfer owl T4(legacy→non-blocking 切换必须显式栅栏,
  否则正确性依赖隐式同步)
- **防线**:capture mode Relaxed + 显式事件 join;单流先行(M1)

### 姿势 10:capture 期 host 暂存对象被回收 → DMA 悬空
- **症状**:replay 时 H2D 拷到的是已回收的 host 页(垃圾数据/崩溃)
- **根因**:candle 原版 `htod_copy` 把 src Pin 进 `dst.host_buf`,
  捕获场景下所有权/生命周期悬空
- **判例**:cudarc fork `htod_copy_into` 专门改写(WRITE_COMBINED
  临时页锁缓冲 + 异步直拷)
- **防线**:owl `htod_persistent` 同语义(装数在 P 阶段完成);
  捕获段内禁止任何 host 缓冲所有权转移

### 使用规则
1. graph 相关新 bug:**先对号入座**(90% 是姿势 2/3/5 的变体),
   对不上号新开判例;
2. 每条防线在 owl 中必须有对应代码位(公理/类型/测试),对不出的
   防线视为待办;
3. 本图鉴随判例库滚动更新,新姿势需带卷宗链接。

### 审核发现的两个延伸议题(登记备忘)

1. **host pinned 内存也要账本**:A5 只约束设备显存,但 cuMemAllocHost_v2 /
   cudaMallocHost 类分配若不加节制同样会 OOM 主机(attention-rs workspace.rs
   即在用)。M2 前给 iface 加 host 账本或明确豁免清单。
2. **"库内注释自曝不兼容图"是好信号**:attention.rs 明说 MXFP4 prefill
   不兼容捕获——这类注释是准入审核的第一线索,审库时先 grep
   `graph.*[Ii]ncompat|not.*capture|cudaMallocAsync`。

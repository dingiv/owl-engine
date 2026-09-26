# cuda backend

cudarc 0.19(std/driver/cublas/nvrtc/dynamic-loading;cuda-version-from-build-system)。
官方 crates.io 版,非 fork;图语义由本封装层自持(A4:语义分歧不进第三方库)。

## 显存记账(硬预算约束;2026-09-26 f16 基线时定稿)

块账本只记**我们主动发起**的分配(Alloc/Htod 的块,`new_block` 登记,
Bytes.len 元素口径)。三类**账外占用**必须从硬预算里扣:

| 项 | 量级 | 性质 |
|---|---|---|
| CUDA context(首分配惰性建立) | **~230-250 MiB** | 驱动级,不可避免,boot 后首个 alloc 时一次性落定 |
| **第三方库自分配预留** | **预留 200 MiB**(硬预算约束) | 见下;当前实测 cuBLAS 热稳态 ≈ 50 MiB |
| kernel module 代码(cuModuleLoadData) | KB~MB 级/核 | nvrtc 编译产物装载,小但账外 |

**第三方库自分配清单(现状与预告)**:

- **cuBLAS(已接,F1)**:内部 workspace 池走裸 cudaMalloc,绕过块账本。
  实测(`tests/gemm_workspace_probe.rs` 永久探针):首个 gemm +18 MiB、
  换形状 +32 MiB、复跑同形状零增长 —— 热稳态 ≈ 50 MiB。治理选项:
  `cublasSetWorkspace` 预分配固定块 / `CUBLAS_WORKSPACE_CONFIG`;
  **接 graph capture 前必须先 warmup(handle + 一次各形状 gemm)再
  graph_begin**(捕获窗内 cudaMalloc 非法);
- **将来同族**:FlashInfer(paged/prefill 引入时有自己的 workspace 语义)、
  cublasLt(batched/bias-fused GEMM,workspace 更大)、NCCL(若引入,
  缓冲自管)—— 全部计入这 200 MiB 预留,不逐项追账;
- **cudarc driver 本身不背账**:所有分配都是我们显式调 `stream.alloc`
  进 handle_alloc,无隐藏池(未用 cudarc 的 arena/allocator feature)。

静态公式(无第三方时的口径):

```text
可用块预算 = 显存总量 − context(~250 MiB) − 第三方预留(200 MiB) − 余量
```

有第三方后不用静态公式,走下面的**实测制**:启动预热后量真实水位。
探针:gemm_workspace_probe.rs(升级驱动/换卡后重跑一次)。

## 硬预算约束(A1 预算前置;实现业务,2026-09-26 定稿)

原则:**进程永不主动 OOM**。所有门槛在启动期一锤定音,服务期只做
准入控制,不在运行中碰预算边界。四阶段生命周期:

### 阶段一:启动预热(模型加载前)⏳

1. server boot(上下文落地);
2. **第三方组件逐个预热**:cuBLAS 懒句柄提前建 + 一次各形状 gemm
   (workspace 落定)、将来 FlashInfer/其他同款 —— 每家一个 warmup
   动作,列入组件注册表;
3. **计量**:`cudaMemGetInfo`(ffi 新原语 → client `mem_info()`)
   取 `(free, total)`,记 `base_used = total − free` —— 这就是 context
   + 全部第三方库的真实账外水位(实测制,替代静态公式);
4. 自此之后,我们自己的每次分配都进块账本(ledger_bytes),任意时刻
   恒等式:`total − free ≈ base_used + ledger_bytes + 小误差`。

### 阶段二:模型装载,量实占 ⏳(装载面已有,门未挂)

- **不做 manifest 预算**(估算不准:激活/碎片/装载期临时峰值都算不齐)
  —— 直接装载,看真实占用;
- 装载中任一分配 OOM → 结构化报错,**拒绝启动**(清理退出,不 panic);
- 装载完成后再量一次 `(free, total)`:模型实占 = base_used 增量 +
  块账本增量,进阶段三做能力验证(那里才是真正的门)。

### 阶段三:warmup 门(最低能力验证)⏳

模型就位后,验证「**最低空余要求**」——不是看剩余多少 MiB,而是
**实际做一遍最低能力动作**:

1. **最低 KV 池预分配**:按 `min_kv_tokens`(默认 1K)算 KV 字节,
   真分配(分配失败 = 拒绝启动);
2. **一次 CUDA graph 捕获**:decode 步走 session 捕获路径(slab + 实例化
   —— cublas/后续第三方已预热,捕获窗内无账外申请);
3. 任一步失败 → 拒绝启动;全过 → 进入服务。

### 阶段四:服务期(池内纪律 + 准入控制)✅/⏳

- **KV 池封顶 by construction**:池 = 预分配常驻块 `[max_slots, …]`,
  运行期零增长 —— 结构上不可能 OOM;
- **准入控制(✅ 已实现)**:`submit` 时 `prompt + max_new >
  max_seq_tokens` 直接拒绝(fail-fast,engine.rs);
- **池满行为(⏳)**:新请求无空位 → ①排队等待 **三级 KV 调度**
  (GPU → host → disk 卸载腾位,照 sglang HiCache/vLLM 先例,另行立项)
  或 ②直接回「当前繁忙」拒绝 —— 策略旋钮,**永不超池**;
- 长上下文请求超池上限 → 同准入拒绝(不截断不挤兑)。

### 实现位清单

| 项 | 状态 | 位置 |
|---|---|---|
| `mem_info()` 原语(ffi cudaMemGetInfo) | ⏳ | cuda ffi + client + iface |
| 第三方组件预热注册表(cublas 先行) | ⏳ | server boot 序 |
| 装载 OOM 结构化报错 + 装后实占计量门 | ⏳ | engine run() 装载路径 |
| 最低 KV + 图捕获 warmup 门 | ⏳ | engine run()(捕获路径已有) |
| submit 准入 fail-fast | ✅ | engine.rs |
| 三级 KV 调度 | ⏳(另案) | 参照 sglang HiCache 实录 |

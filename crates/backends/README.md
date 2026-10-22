# backends —— 后端 HAL 与准入纪律

```text
iface/   HAL 契约:contract::DeviceClient(现役唯一入口)+ DeviceHandle
         (分配账 + 命令面)+ Backend/Device/Pool/Buf 声明面(A6 设计出处,
         M2/TP 面桩,当前零实现体 —— 见 crates/backends/iface/src/lib.rs 头注)
cuda/    CUDA 栈(官方 cudarc 封装):GpuServer(actor 事件循环)/ GpuClient
cpu/     CpuFace —— DeviceClient 的进程内同步实现(测试 / 对拍 / 无卡开发)
rocm/    占位(未来厂商栈)
```

依赖方向:iface 不依赖任何后端;后端只依赖 iface。**上层(engine / models /
server)禁止直接 include 任何后端内部类型**,只 import `owl-iface`。

> 显存记账与四阶段硬预算 = [`cuda/README.md`](cuda/README.md);
> CUDA Graph 死亡姿势图鉴(判例汇编)= `docs/devlog/perf/pitfalls-cuda.md`;
> 所有权四分类(FROZEN / OWNED / REFERENCE / BINARY)= `VENDOR.list.md`。

## ⚠️ 第三方库 CUDA 准入铁律(A5.2 执行细则)

**任何能直接触达 CUDA 的第三方库,引入前必须源码审核:它是否绕过后端账本
私自调用 `cudaMalloc` / `cudaMallocAsync` / `cuMemAlloc*` / 池分配 / cuBLAS
懒 workspace 等,自行产生"账外显存"。**

- 审核结论为**会** → 二选一:
  1. **不用**这个库;
  2. **copy 其代码进我们自己的 crate 改造**,把分配改道后端账本
     (所有权按 charter A4 显式认领:parity 硬门 + 差异清单 + 只 cherry-pick)。
- 审核结论为**不会**(分配全部经我们出口,或可预钉)→ 允许依赖,
  登记进 `VENDOR.list.md` 的审核栏(结论 + 审核日期 + 关键证据行号)。
- 审核方法(最小清单):

  ```bash
  grep -rn "cudaMalloc\|cuMemAlloc\|cudaFreeAsync\|cublasSetWorkspace\|cudnnSetWorkspace\|malloc_async\|alloc_zeros\|htod" <crate>/src
  ```

  重点:构造函数 / 首次调用路径上的**懒分配**;Drop 路径上的**自由释放**;
  第三方 FFI 里打包的静态分配。**A5 下没有"库自己管自己的内存"这回事。**

### 已审核结论(持续维护)

| 库 | 结论 | 证据 / 说明 |
|---|---|---|
| cudarc(官方 0.19) | ✅ 通过 | 分配仅在显式调用 `alloc*` 时发生,无懒分配;context 常驻底价已计入硬预算 |
| candle fork(candle-gb) | ✅ 通过(带焊缝) | 账外裸分配无;BLAS workspace 32 MiB 池外**预钉** = "预登记"正面教材 |
| attention-rs | ⚠️ 有条件通过 | ① MXFP4 prefill 路径经 `cudaMallocAsync` 分配 scratch 且**与图捕获不兼容** —— 该路径我们不启用;启用前必须改道账本 + 出图;② `cuMemAllocHost_v2` 是 pinned **host** 内存(不违 A5 设备预算,host 账本另记) |
| cuBLAS | ✅ 通过(实测制) | 内部 workspace 走裸 `cudaMalloc`,热稳态 ≈ 50 MiB,计入"第三方预留";**接图捕获前必须先 warmup**(handle + 一次各形状 gemm),见 `cuda/README.md` |
| FlashInfer / vLLM-marlin / XQA | ✅ 通过(源码进树) | 三者均 **vendor 进 `crates/kernels/cu/`、构建期从源编**(非二进制依赖);分配面按上表清单复审,登记见 `VENDOR.list.md` §2 |
| NCCL | ❌ 除名 | 内部自建通信缓冲不透明、不可登记,违反 A5.2 + A2.2 ⇒ **不引入**(charter A2.5) |

新库不登记不依赖;结论变更(升级快照 / 上游行为变化)必须**重审并更新本表**。

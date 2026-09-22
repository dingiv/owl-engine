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

### 审核发现的两个延伸议题(登记备忘)

1. **host pinned 内存也要账本**:A5 只约束设备显存,但 cuMemAllocHost_v2 /
   cudaMallocHost 类分配若不加节制同样会 OOM 主机(attention-rs workspace.rs
   即在用)。M2 前给 iface 加 host 账本或明确豁免清单。
2. **"库内注释自曝不兼容图"是好信号**:attention.rs 明说 MXFP4 prefill
   不兼容捕获——这类注释是准入审核的第一线索,审库时先 grep
   `graph.*[Ii]ncompat|not.*capture|cudaMallocAsync`。

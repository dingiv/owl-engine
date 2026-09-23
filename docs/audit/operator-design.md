# 算子设计审查清单(crates/nn + engine OwlTensor/dry_kernels)

> 审查域:tensor.rs / dyn_tensor.rs / erased.rs / kernels/ / ops.rs / cublas.rs
> + engine models/layers/mod.rs(OwlTensor)/ dry_kernels。
> 校准案例:cat/stack `*mut u8 .add(元素数)` 字节步长错位(已实锤)、
> varlen scratch 视图逃逸(已实锤)、narrow 中间维丢 after 因子(已修)、
> device_ptr() 契约"只许传地址不许算术"(dyn_tensor.rs:140 注释)被多处违反。
> 风险级:P0 = 已知能产出错误数值/UB;P1 = 特定条件触发;P2 = 债/可维护性。

## 危险清单

### A. 指针算术(元素 vs 字节语义)

- **A1** `nn/src/erased.rs:472,487-488,530` — cat/stack 对 `*mut u8` 做 `.add(元素数)`
  (已修复,本清单保留为族首;教训 = 擦除层指针算术必须先 cast 具体类型)。
- **A2** `engine/src/models/layers/attention.rs:309,315` — `kc.device_ptr().add(slot_row * row * 4)`
  字节意图正确(`*mut u8` × 字节偏移),但 **硬编码 4** 与 dtype 耦合;
  fp8/bf16 KV 落地时静默错位。风险 **P1**。修法:常量 `ELEM_W = size_of::<T>()` 或泛型化。
- **A3** `nn/src/kernels/attention.rs` 14 处 unsafe —— 逐条标注见 §A.3 表(P0/P1 混合,
  详见下文"launch 参数"节);核心风险 = 指针从 DynTensor 擦除层 `device_ptr() as *mut T`
  进入 kernel,元素/字节语义靠调用点自律,无类型系统保护。
- **A4** `nn/src/erased.rs` index_select/gather/scatter_add 族:`src[(idx[k]) * inner + c]`
  在 .cu 内做 `unsigned long long` 乘法,**src 长度 > 2^31 时 u32 idx 乘 inner 溢出路径**
  未审计(0.8B vocab 248320 × 1024 = 2.5e8,f32 边缘安全;长上下文 KV 需复核)。
  风险 **P1**。

### B. 双类型边界(DynTensor/Tensor)

- **B1** `dyn_tensor.rs from_raw(Raw 变体)` — 无保活视图构造。调用点审计:
  仅 graphplan.rs bindings 路径使用(注释声称 GraphPlan 持有租约);
  **graphplan 已 deprecated(挂账 §五·5),接手者 Session 不持有同租约** → 迁移时
  Raw 视图可能悬空。风险 **P0(迁移时)**。修法:Session 化时 Raw 变体改
  结构化报错或强制携带 keepalive。
- **B2** `downcast` 纪律:全调用点 dtype 检查覆盖良好(不匹配 = 结构化报错),
  无静默位型重解释 —— **安全项**。
- **B3** `DynTensor::device_ptr()` 契约("只许传地址不许算术")与实际使用矛盾:
  全仓 20+ 处算术。契约已名存实亡。风险 **P2(文档债)**。修法:
  提供类型化视图 API(`f32_ptr_at(off)`),契约改为"算术必须在具体类型指针上"。

### C. OwlTensor trait 设计

- **C1** `&self -> Result<Tensor>` 共享语义未文档化(Arc 租约,非拷贝);
  candle 形态照抄,engine 侧读者无从分辨副本/视图。风险 **P2**。修法:trait 方法
  逐个补语义 doc;`reshape` 补 contiguous 断言(S 恒 contiguous 律的强制点缺失)。
- **C2** `reshape` 内残留调试代码:`Backtrace::force_capture()` 抓后即弃,
  且 n≠have 分支未 bail、继续执行 reshape —— **P1**(错误形状静默通过,
  reshape 本体可能 panic 或产出错布局)。
- **C3** TLS ctx_scope 桥调用点清单(拆桥准备):
  erased 全部 25 个算子、ops 面全部、ctor 6 个、mask/narrow/sigmoid 垫片;
  汇总 ~45 个调用点,全部经 `ctx_scope::{with,with_dry,with_blas}` 三个入口。
  拆桥建议:三入口改为接收 `&KernelCtx` 参数,调用点机械替换(单 commit 可完成)。

### D. erased 输入校验缺口(对照 S1-S6)

- **D1** `sub/div/broadcast_sub/broadcast_div` 均以 mul(−1/inv) 合成 —— 语义等价但
  额外一次 scratch 分配 + kernel(热路径债,P2)。
- **D2** `transpose/t2`:>2D 非 size-1 换位直接结构化报错(一期限制),
  调用点已绕行(transpose01/12 dry 核)——**安全项,限制需写进 S 律文档**。
- **D3** `cat` 对非 f32 输入无 early check(依赖 same_shape 后 `scratch_tensor::<f32>`
  的 dtype 假设;bf16 cat 会产出位型垃圾)。风险 **P1**。修法:入口 require_f32
  或按 dtype 分派。

### E. cublas.rs

- **E1** `unsafe impl Send/Sync for NnBlas`:handle 经 cublasSetStream 换流使用,
  Send/Sync 担保成立的前提 = **所有调用点持 rig.ops 锁**(ctx_scope::with_blas)。
  审计确认锁覆盖完整 —— **安全项,但担保依据必须写成注释**(目前只有一句)。
- **E2** workspace:`cublasSetWorkspace` 未设,cublas 自建 workspace 与图捕获的
  相互作用未验证(捕获期 cublas 可能内部分配 → 图外分配 → replay 悬空)。
  风险 **P1(图路径)**。修法:P 阶段显式 SetWorkspace(池内)。

### F. dry_kernels / launch 封送

- **F1** usize→u64 as 转换:64 位平台无截断;`total as u32` 出现两处
  (narrow_strided/dry attention),>4G 元素时截断。当前形状远小于界限。
  风险 **P2**。修法:grid 維拆分或 debug_assert!(total < u32::MAX)。

## 设计问题(非条目)

1. erased 层"指针进/指针出 + stream"的纪律正确,但缺一层**类型化发射器**
   (`src.ptr::<f32>(off)`),把元素/字节语义收敛到一处 —— A 族全部根治。
2. OwlTensor 与 erased 双面(签名翻译 + 实现)使每个算子的审查面 ×2;
   ctx 拆桥(C3)后可合并为单面。

## 安全项(审查通过)

- downcast dtype 检查(B2);erased same_shape/require_f32 覆盖 add/mul/matmul;
  signal emit 全调用点覆盖(借/写/出);scratch_tensor 与 zeros_tensor 的
  P 阶段分离正确;testkit rig 的 htod 持有缓冲防悬空(案例⑥反面教材已吸收)。

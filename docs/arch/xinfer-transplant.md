# xinfer 换头移植总设计 —— owl 仓库收编新顶层 crate

> 版本:v0.1(2026-09-22 立项)。裁决人:用户。
> 关系修订:本设计**部分取代** charter §一"对照与判例库 = xinfer(冻结存档)"的
> 隔离姿态——xinfer 的上层实现收编进 owl,作为顶层 crate 长期演进;
> `packages/xinfer` 仓库本体维持冻结不动(只读参照源)。
> **验收口径(用户裁决):先编译,后运行。** M 阶段全部以
> `cargo build --workspace` 全绿为准;运行/指标验收另立里程碑。

---

## 一、手术定义(换头,不是换脚)

- **owl = 身体**:iface/owl-cuda(governor/pool/graph/signal)/owl-nn 是基座,
  graph 治理是第一天就有的架构位;
- **xinfer 上层 = 头**:models / scheduler / runner / speculative / utils /
  transfer / server 的**逻辑**搬进 owl,搬运时算子层 API 从
  `candle_core::Tensor` 统一翻译为 owl API;
- **packages/xinfer 不动**:不升 cudarc、不解冻、不加分支——它是 96k 行的
  只读参照源(避免此前评估中"candle fork 升 0.19"的 2-4 天迁移成本:
  搬运重写天然落在新 API 上,迁移成本被翻译成本吸收)。

## 二、普查结论(2026-09-22 实测)

| 维度 | 数字 |
|---|---|
| candle_core 触点 | ~880 处(models 437/58 文件;utils 255/25;core 103/7;transfer 46/3;speculative 31/6;runner 9/1) |
| 最热方法 | to_dtype 420 / contiguous 288 / narrow 249 / reshape 239 / get_with_hints_dtype 202(权重装载)/ index_select 29 / gather 12 |
| dtype 面 | F32(451)/ BF16(68)/ U8(63)/ U32(62)/ Q8(35)/ F16(28)/ I64(15)/ F8E8M0(12) |
| owl 现状 | Tensor 仅 f32;算子仅 matmul/add/silu/rmsnorm/rope/embedding + f16 kernel 雏形 |

差距 = owl-nn 的 dtype × 算子矩阵扩面,这正是移植的主体工作量。

## 三、crate 布局

单 crate `owl-engine`(顶层),模块树镜像 xinfer `crates/core/src`,避免
workspace 碎片化、允许逐模块搬运:

```text
crates/engine/
  src/
    config.rs        <- utils/config.rs
    scheduler/       <- core/scheduler.rs(+sequence)
    runner/          <- core/runner.rs(GraphCapturer 改驱动 owl CaptureSession)
    kvcache/         <- utils/kvcache_allocator.rs + kv_backend.rs
    sampler.rs       <- logits_processor 采样链(P4 已定谳 radix 直吃 logits)
    models/          <- models/*(qwen3_5 先行,其余按需)
    speculative/     <- dflash2 等
    server/          <- HTTP 入口(bin)
```

依赖向:owl-engine → owl-nn → owl-cuda → owl-iface/owl-signal。
**owl-engine 禁止依赖 candle/cudarc**(A4 vendor 纪律:FFI 只存在于 owl-cuda)。

## 四、API 翻译原则

1. **语义直译,不复刻 candle**:搬运点按 owl 的 Tensor/Device/Pool API 重写
   调用点;不为兼容 candle 造影子层(影子层=把旧基座的假设偷渡进来);
2. **分配语义重写**:candle 的 `Tensor::new/arange/...`(隐式分配)一律改为
   owl 的**池直连工厂**(`pool.zeros_tensor/scratch_tensor/from_vec_tensor`,
   裁决 5:分配入口在 Pool);
3. **捕获语义重写**:runner 的 GraphCapturer/planned_graph_capture_batches
   改驱动 owl `CaptureSession`(warmup 门禁/Capturing 相位/哨兵③/强租约
   自动生效);xinfer 的 utils/graph.rs 裸 FFI **不搬运,直接消失**;
4. **dtype 扩面在 owl-nn**:新 dtype(F16/BF16/U32/I64/U8)与缺失算子
   (softmax/narrow/cat/index_select/gather/slice-scatter/...)落在
   owl-nn 的 `Tensor`/`ops`,kernel 主体可先 `unimplemented!()`
   (编译先行口径),按运行里程碑回填;
5. **量化/Q8/F8E8M0**:Q8/F8E8M0 触点(35+12 处)登记进缺口清单,
   对应 marlin-ffi/dflash2 量化路径,搬运期以类型占位、实现延后。

## 五、里程碑(全部以编译全绿为验收)

| 阶段 | 内容 | 编译验收 |
|---|---|---|
| T0 | 设计文档 + crate 骨架 + 翻译映射表 | 本文档 |
| T1 | owl-nn dtype 泛型化 + 翻译映射表覆盖的算子签名层(体可 stub) | workspace 全绿 |
| T2 | config/scheduler/sequence/kvcache/sampler 搬运 | 同上 |
| T3 | runner 搬运(GraphCapturer → CaptureSession)+ models/qwen3_5 + layers | 同上 |
| T4 | speculative + transfer + server bin | 同上 |
| M-运行 | 真机冒烟 → std 五点三轮 → 对 xinfer requirements 三线 | 另案 |

## 六、风险登记

1. **翻译不是转写**:xinfer 若干正确性依赖 candle 语义(broadcast 规则、
   dtype 提升表、contiguous 惰性),owl Tensor 的语义表必须在 T1 前写死,
   否则运行期炸在静默错误上;
2. **量化层空白**:Q8/F8E8M0/GGUF 装载路径(gguf_varbuilder)在 owl 无对应,
   T3 搬运 qwen3_5 时按 bf16/f32 路径先走通类型层;
3. **mamba/GDN 特化**:qwen3_5 的 mamba_slot/mamba 网格依赖深度绑定
   attention-rs 语义,搬运时按 owl-iface 的 DevBuf 契约重表达;
4. **多卡**:xinfer 多进程 rank 拓扑与 owl A2.6 单进程多实例冲突,
   多卡语义延后到 M-运行之后的独立里程碑,先收编单卡路径。

---

## 七、施工日志

- **2026-09-22 T1 第一刀**(`292a733`):dtype 面/语义表/视图算子签名层;
  测试设备环境变量化(OWL_TEST_DEVICE)——根因修复与 8133 生产引擎同卡互踩;
  千问普查产物 candle-api-mapping.md 入库。
- **2026-09-22 T1 第二刀**(本提交,千问②施工+主 agent 复验):
  contiguous/is_contiguous 恒等、to_dtype 签名锁定、sum/max/min/cat/stack
  shape 校验真实+设备体 stub、TensorPoolOps 增 full_tensor/arange_tensor
  (HostArith trait:f32/u8/u32/i64;Bf16/F16 无算术=无 host 模拟,S5)。
- **观察项 O-1**:owl-cuda 图租约测试在多线程并发 capture/launch 时偶发
  (~1/8 轮)失败,单测试二进制 10 轮 + 全 workspace 8 轮未能复现;
  定性 = 双 context 并发图操作的 GPU 级竞态(测试环境,非引擎逻辑);
  M-运行里程碑前须转单线程化验收或 context 级隔离。

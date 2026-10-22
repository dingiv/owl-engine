# 解释器执行环境 —— EnvProvider 设计与参数枚举

> 版本:v1.0(2026-10-03)。代码:`crates/models/src/env.rs`(本文件是它的
> 设计文档与参数总表)。抽象定位:解释器执行环境的**完备描述** —— 硬件档、
> 量化方案、KV 策略、分派旋钮、诊断面,一等公民化;默认值齐备,engine /
> 测试构造后按需定制,解释器按它执行。
> 关系:charter A(被动律:环境必传参,kernels/解释器零探测)的**载体实现**;
> requirements REQ-HW-01(arch 分发)、REQ-CTX-01(量化)、REQ-CTX-03(KV 量化)、
> REQ-DEC-02(投机)的**参数面表达实体**。

---

## 一、为什么需要这个抽象(动机)

EnvProvider 之前,解释器的影响面散落三处:

1. **iface `op_env()`**(hw):后端 trait 方法,解释器逐 Call 探测;
2. **`kv_paged_policy(dtype)`**(页/向量宽):dtype 键表的自由函数;
3. **层/解释器内散装 `std::env` 直读**(qkv_fuse / prefill_split /
   force_naive / fi / resolve_trace / host_argmax):测试与运维被全局
   env-var 耦合,热路径逐次读环境,"当前环境是什么"无处可查。

三处合一后:**环境是什么 = 一个对象的字段**;测试定制 = 改一个字段;
新硬件/新量化/新 KV dtype = 加一个枚举臂 + 对应实现,调用点零改动
(REQ-HW-01 的 arch 分发表达在此收敛)。

## 二、对象结构(现役代码)

```rust
pub struct EnvProvider {
    pub hw:   HwEnv,     // Cpu | Cuda(Arch)
    pub quant: QuantPlan,// F16 | W4A16 | W4A16Awq(W4A8 留位)
    pub kv:   KvEnv,     // { dtype, page, x, paged }
    pub attn: AttnEnv,   // { qkv_fuse, prefill_split, force_naive_prefill, fi }
    pub diag: DiagEnv,   // { resolve_trace, host_argmax }
}
```

- `Default` = **生产基线**:全 opt-in 关、f16 paged 页 32、`HwEnv::Cpu`;
- `from_env()` = env-var 兼容面(env 读取**只**在此一处;组件内部直读绝迹);
- `op_env()` = env → kernels `OpEnv { hw, page }` 的唯一投影(kernels 零依赖
  律:契约类型不过 kernels);
- 供给链:engine boot 构造(`from_env` + iface `op_env()` 合入硬件 +
  `LoadedModel.quant_plan` 注入量化)→ 逐 chunk/step `ctx.env = self.env`
  (捕获期烘焙)→ 层声明 / 解释器按它执行。测试:`ctx.env.attn.qkv_fuse = true`
  式字段定制,不碰全局态。

## 三、参数枚举总表

> 状态标记:**现役**(生产在用)/ **已验证**(测试/实验全绿,未默认)/
> **留位**(枚举/文档就位,实现缺)/ **未开工**(需求在案,无代码)。

### 3.1 硬件(hw)

| 选项 | 状态 | 说明 / 前置件 |
|---|---|---|
| `Cuda(Sm86)` | **现役** | 30 系一等公民(REQ-HW-01)。marlin .a(sm_86)、FA2 非 hopper 分支、nvrtc 通道、GDN port、所有现役核 |
| `Cuda(Sm89)` | 留位 | 40 系顺带兼容(REQ-HW-01)。枚举已有;kernel 实例化需 `sm_89` 编译档 + FI/FA2 路径自证 |
| `Cuda(Sm90)` | 留位(未开工) | Hopper。FI adapter 的 `SM_90_PASS` 分支(FA3 + cutlass 内部头,csrc/nv_internal 已随 fork 在库)+ TMA/warp-spec 内核族 + TP2 NVLink 面。**头号价值:FA3 prefill / Hopper tensor core** |
| `Cpu` | **现役**(对拍锚) | reference 解释器(CpuInterpreter/reduce 同步参照)+ CPU face 装载路径;Call op 在 Cpu = 结构化拒绝。**不是推理目标,是数值锚与降级路径** |
| `Rocm(GfxN)` | 未开工(远景留位) | AMD RDNA(gfx1100/ gfx1151 等)。**不是枚举加一行**:需要 HIP backend(DeviceClient 第二实现)+ kernel 族重编(HIP 构建通道)+ marlin/FA2 替代品(ROCm 的 aotriton/ComposableKernel)。抽象上 `HwEnv` 预留 `Rocm` 臂位,driver 分发表按 arch 键扩展 |

**演进规则(REQ-HW-01)**:新硬件 = ① `HwEnv`/`Arch` 加臂 → ② driver
resolve 分发表加行 → ③ 该 arch 的 kernel 实例化/预编资产 → ④ FI adapter
或等价物 gating → ⑤ 对照矩阵全绿。禁止在业务代码写死 arch。

### 3.2 量化方案(quant)

| 选项 | 状态 | 说明 / 现役检查点 |
|---|---|---|
| `F16` | **现役** | 0.8B 基线(bf16 检查点 f16 直转);cublas/marlin-free 全 f16 面 |
| `W4A16`(g128-sym) | **现役** | llm-compressor RTN / compressed-tensors U4B8;marlin kU4B8 内核;0.8B-W4A16 装载线(E3) |
| `W4A16Awq`(g32-asym) | **现役** | cyankiwi/Qwen3.8-27B-AWQ-INT4;kU4(has_zp)+ zp 烘焙;27B 主力档 |
| `W4A8` | 留位 | kS8 内核族在 marlin .a 内固化(fp32_reduce 契约已修);缺激活侧动态量化发射线;REQ-CTX-01"一 flag 切换"宿主 = EnvProvider.quant |
| GGUF k-quant / i-quant | 未开工 | 双生态需求(requirements §其他);需 GGUF dequant 发射线 + marlin repack 桥 |
| ISQ 在线量化 | 未开工 | mistral.rs `--isq` 先例;装载期现场量化(与 GPU repack 同窗口) |

**演进规则**:新量化 = ① `QuantPlan` 加臂 → ② Linear 构造臂(marlin_eligible
谓词扩展)→ ③ 装载源(formats/ 加 Source)→ ④ GEMM foreign 臂 → ⑤ parity
对拍锚。量化是**构造期定形**(REQ-PRE-01:禁 enable_* 可变后置)——provider
携带的是环境真值,装载入口显式选型。

### 3.3 KV 策略(kv)

| 选项 | 状态 | 说明 |
|---|---|---|
| `dtype=F16`,paged 页 32 | **现役** | 页 32 = 配对律定谳(decode v1 bs32 × prefill bs32 交集);x=8 |
| `dtype=FP8` | 留位(REQ-CTX-03,P2) | FI adapter 的 fp8 分支现成(裁剪面外,加臂即可);**税收在案**:xinfer sm86 native flash 路径 32k 仅 0.42×;走 FI 需实测,C7 净增益护栏 |
| `dtype=INT8` | 留位 | 速度近无损口径;xinfer 机制沿用 |
| `dtype=Q4`(turbo) | 留位 | 精度损伤待评(ppl/任务双口径) |
| paged=false(legacy 直排) | **现役**(回退) | naive 核族对拍/测试用 |

**演进规则**:新 KV dtype = ① `KvEnv` 构造行 → ② `kv_paged_policy` 键表行
(或 provider 直接携带)→ ③ K0 核 dtype/布局变体 → ④ 注意力核读臂 → ⑤ 影子池
几何(若 FI)→ ⑥ 速度/精度二维数据表(REQ-CTX-03 验收口径)。
**容量红利**:fp8 = 池×2(max_seq/并发格翻倍),对长 ctx 是白赚项。

### 3.4 注意力分派(attn)

| 旋钮 | 默认 | 状态 | 说明 |
|---|---|---|---|
| `fi` | false | **现役**(生产建议开) | FlashInfer FA2 paged prefill(K/V 影子池 + plan 缓存);prefill +23% / @8192 +55.7% |
| `qkv_fuse` | false | 已验证(opt-in) | C1-W2 qknorm_rope_kv_insert 三发合一;k 支转置已结案;release 预期 +2-4% |
| `prefill_split` | false | 已验证(对照臂) | 自研 flash-decoding;FI 落地后退役为对照/兜底 |
| `force_naive_prefill` | false | **现役**(诊断) | 朴素逐 token 回退;数值质量 A/B 的二分开关 |

**分派优先级**(层侧谓词序):FI(表在场)→ split(门开且形状合)→ paged
(配对律谓词)→ naive(force_naive 或谓词不过)。

### 3.5 诊断面(diag)

| 旋钮 | 默认 | 说明 |
|---|---|---|
| `resolve_trace` | false | driver 逐 Call 拾取追踪(OWL_RESOLVE_TRACE) |
| `host_argmax` | false | host argmax 对照(校准设备采样;OWL_HOST_ARGMAX) |

### 3.6 投机解码(spec;E5 立项预留,未开工)

> REQ-DEC-02(P1)+ C7 护栏(净增益 >0 才开)。provider 预留 `spec: SpecEnv`,
> 建议形态(立项时定稿):

```rust
pub struct SpecEnv {
    pub method: SpecMethod,   // None | DFlash2 | Mtp | DSpark(DDTree 待调研)
    pub draft: DraftRef,      // 草稿模型路径 + 量化档(REQ-CTX-04)
    pub depth: usize,         // DFlash2 n(4=单卡甜区 / 7=双卡)、MTP3
    pub graph_verify: bool,   // spec-aware graph(verify 形状入图;S3 档位复用)
    pub log_accept: bool,     // 接受率/接受长度分布(REQ-DEC-04;C7 数据源)
}
```

- 现役参照:llama.cpp DFlash2 n=4 单卡 76.25 t/s;ninfer MTP3 85.34;
  xinfer spec-aware graph + 锚复用配方(净增益 +33.8%)
- **前置**:调度层每步开销清零(S1 ✓)与胶水收敛(C1)——否则净增益测不准
- 验收:C7 净增益 >0 才翻默认;接受率日志可视(REQ-DEC-02)

## 四、现役组合矩阵(已验证)

| 组合 | 硬件 | 量化 | KV | 分派 | 验证面 |
|---|---|---|---|---|---|
| 0.8B 基线 | Sm86 | F16 | f16 paged32 | paged | models/engine 全量 + 4k QA |
| 0.8B W4A16 | Sm86 | W4A16 | f16 paged32 | paged | w4a16 e2e + shape_sweep |
| **27B 主力** | Sm86 | W4A16Awq | f16 paged32 | **FI** | 104+29 全量 + 会战 15/15 + 召回梯度 |
| 27B 对照 | Sm86 | W4A16Awq | f16 paged32 | paged / split / naive | A/B 对照锚 |
| 27B 融合 | Sm86 | W4A16Awq | f16 paged32 | FI + qkv_fuse | k-probe 三臂硬门 |
| CPU 锚 | Cpu | F16 | — | — | reference 对拍(非推理) |

## 五、开放问题 / 挂账

1. **装载域 env**:OWL_LOAD_CPU_REPACK / LOAD_VERIFY / LOAD_DEBUG 迁
   `LoadEnv`(LoaderCtx 携带)——装载是独立执行域,provider 是否统辖待裁决;
2. **kv_paged_policy 与 KvEnv 并峙期**:现 dtype 键表函数与 KvEnv 并存,
   新 KV dtype 只走 KvEnv 构造行,最终收敛删函数;
3. `Hw::detect` 永久缺席(被动律)——多 GPU 异构(heterogeneous arch)时
   provider 需按 device ordinal 分发(EngineConfig 扩展,TP2 前裁决);
4. spec 融入后 `AttnEnv` 是否升格 `ExecEnv`(分派旋钮域扩展的命名裁决);
5. AMD RDNA:需求 §二"非 CUDA 不维护"与本文件 Rocrm 留位的张力 ——
   留位是**抽象预留**(枚举臂位),不承诺排期;立项需先过 backend 成本盘点
   (HIP DeviceClient + kernel 族),参照 FI 预编先例评估。

## 六、变更记录

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-10-03 | v1.0 | EnvProvider 抽象落地(代码 env.rs)+ 参数总表定稿;硬件四档 / 量化六档 / KV 四档 / 分派四旋钮 / spec 预留形态 |

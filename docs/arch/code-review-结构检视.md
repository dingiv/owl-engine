# owl 代码结构检视(2026-10-03 深夜Ⅳ)

> 标尺(用户裁决):**模块明确拆分;纯函数/不可变路径为主体;可变状态函数收敛到
> 顶层 mutable 函数与 I/O 边界**。本文件 = 全景解释 + 按此标尺的逐条检视 + 重构路线。

---

## 一、全景:数据如何流成一个 token

```text
[启动]
Engine::new ──► ModelLoader::load(格式源合成/懒物化,字节→设备块)
            ──► StatePool(状态块池:KV/GDN 格/快照/块表)
            ──► GraphPlan::plan(闭包=「画一步」的纯声明)
                  ├─ warmup:  init 数据 eager 干跑一遍(声明违约在此暴露)
                  └─ capture: graph_begin → eval_ops_multi(多根共享 memo)
                              → 逐节点翻译 launch/alloc → graph_end 实例化

[每 token]
scheduler(决策面:纯函数,产 SchedulerOutput)
  └─ exec(执行面:唯一可变编排)
       ├─ fill_slots(5 标量 H2D 入持久槽,指针稳定)
       ├─ graph_launch(单次 cuGraphLaunch,~1044 节点)
       ├─ token 回读(4B D2H,同步点)
       └─ sample_and_emit(采样/EOS/事件)

[图内一步](约 1044 内核,23.6ms)
embed → 48×(GDN|attn 层:投影 marlin → conv → 门控/递推/归一 → out_proj
        ;attn: norm_rope → paged attn → gate_mul → o_proj;mlp: gate/up →
        silu → down)+ final_norm → lm_head → argmax
```

**分层与职责**(自底向上,依赖单向):

| crate | 职责 | 纯度 |
|---|---|---|
| `backends/iface` | 设备契约(Arg/Bytes/LaunchMsg/DeviceClient 五原语+图三原语) | 纯类型 |
| `kernels` | CUDA 源码之家 + 注册表 + driver 拾取(OpReq→KernelPick)+ repack/cublas/flashinfer 绑定 | 纯值+FFI 边界 |
| `models` | **声明层**:TensorOps 纯值 DAG、层容器(零执行)、ops lower、格式源、解释器(唯一发射者)、EnvProvider | **纯函数核心** |
| `engine` | 调度(决策)+ 执行(可变编排)+ 采样 + 状态池治理 | 可变边界(应收敛处) |
| `backends/cuda,cpu` | 哑执行器(参数槽→设备指针;同流保序;图捕获) | I/O 边界 |
| `shared` | metrics(timers,OnceLock 全局 store) | 可观测 |

**声明/执行分离**(本项目最重要的既有资产,与用户纯函数原则同构):
- `TensorOps` = 纯值 SSA DAG(clone=O(1) Arc 共享;无 Arc 共享可变;毒值结构化)
- 层 = `new(容器)/load_ops(装载描述)/forward(纯声明)` 三段,层内零执行
- 解释器 = 唯一把声明翻译为 alloc/launch 的地方(CSE 备忘录 + 竞技场 + 毒值落地)
- server = 哑执行器(不做算子语义判断,按类型化槽序对位)

---

## 二、检视发现(按严重度)

### 🔴 R1 热路径逐次读环境变量(违:配置单点解析)
- `exec.rs` decode 每步 7 次 `var_os`(OWL_STEP_PROFILE/GDN_DUMP/DEBUG×3/REP_PENALTY…)
- `server.rs::handle_launch` 每次发射 1 次 `var_os`(OWL_GPU_PROF)——**每步 1044 次发射 = 1044 次 env 查询**
- 全仓 **59 个 OWL_* 变量**,读取点散布 engine/models/backends 三层,有的 boot 读、有的每步读、有的每发射读
- **改法**:进程入口一次性解析为不可变 `RuntimeConfig`(分段:EngineCfg/ProbeCfg/BackendCfg),
  Arc 贯穿传递;`var_os` 只允许出现在 `main`/`EnvProvider::from_env`/测试三处。

### 🔴 R2 捕获期诊断注入声明层(违:声明纯度)
- `model.rs::last_hidden` 现含 `ts_stamp` 注入分支(env 门控)——**取证探针长进了纯声明树**
- `exec.rs` kv-dump/logits-dump/GDN-dump/sample-dump/step-prof 五套手搓仪表混在 execute_decode
- **改法**:探针统一为「声明期 tap」机制(interpreter-tap 已有雏形):tap 由 ctx 携带、
  声明层只 `tag()` 打标,插桩与收割全部在解释器/观测面。历史专项探针(kv-dump/
  logits-dump/GDN_DUMP/TS_PROBE)迁入 tap 或降级为测试。

### 🟠 R3 server.rs 巨石 + 外来核槽序解析 ×12 重复
- 2114 行:dispatch + 7 个 foreign handler(**blocks/scalars 收集循环逐字重复 12 次**)
  + DMA/码头池 + 图处理 + 三套探针
- **改法**:`ForeignSlots::parse(&[Arg]) -> Result<(Vec<(id,off)>, Vec<u64>)>` 单点解析
  (Block/BlockSlice/U64 通吃),7 个 handler 各剩业务行;再按 `foreign/{cublas,
  marlin,flashinfer,gdn}.rs` 拆文件;DMA 收割拆 `harvest.rs`。

### 🟠 R4 ForwardCtx 上帝对象(16 字段 Option 大杂烩)
- tokens/kind/pos/kv/rope/gdn/kvs/gdns/kv_slots/kv_lens/gdn_slot/ctx_base/fi/fi_kvi/
  env/gdn_slot_host/ts_buf —— 每层 clone 全量传递,层实际只读其中 3-5 个
- **改法**:`StepCtx`(全树共享:env/kvs/gdns/rope/ts_buf)+ `LayerView`(每层派生:
  kv 或 gdn 之二 + slots)两层拆分;`layer_ctx` 已是雏形,补成类型即收敛。

### 🟠 R5 32k 行 bak 死代码驻留 crate 树
- `engine-bak`(1.1M/19k 行)+ `nn-bak`(496K/13k 行)+ `cuda_bak`——grep/wc/审计
  全被污染(本检视的 top 文件榜一半是 bak)
- **改法**:git 历史已存档,直接删除目录(或移 docs/archive 外部);`engine-bak`
  中仍有参考价值的 GGUF/调度器逻辑先摘录进 docs。

### 🟡 R6 层文件巨石化(gdn.rs 2594 行 / attention.rs 2066 行)
- 单文件混居:层容器 + 内核声明 helper + host 对拍参考 + 单测 + 取证探针
- **改法**:gdn.rs 拆 `gdn/mod.rs(层)+ gdn/host.rs(HF 参考)+ gdn/tests.rs`;
  attention 同。布局账(sizes 谓词)下沉 specs。

### 🟡 R7 测试单文件与探针残留
- `engine/tests.rs` 845 行 30 测单文件(应按主题拆:capture/quant/session/e2e)
- 案卷期插桩(GDN_DUMP/TS_PROBE/CAP_PROF/D2H_PROF)散留 5 文件——统一后归 tap/测试

### 🟡 R8 `static` 与全局面(可接受,登记即可)
- `NEXT_ID`(AtomicU64,节点身份证)✓;`metrics::STORE`(OnceLock)✓;
  `BLAS_WS` 已改 server 私有 ✓。**无 static mut** ✓。

### ✅ 已达标项(点名表扬)
- TensorOps 纯值世界 / 层零执行 / 解释器单发射者 —— 与纯函数原则同构
- 毒值结构化(LazyError 随树透传,错误现场可重放)
- 格式源 trait(键→字节)与装载域(字节→设备)分离;源不懂设备
- server 哑执行器纪律(handle_launch 不做语义判断)
- C2 槽序签名机器校验(组合期拦截 u32/sz 错位)
- capture-slab(捕获期零 cudaMalloc)与 pinned 码头池(S1)

---

## 三、重构路线(按 ROI)

| 批 | 内容 | 工作量 | 风险 |
|---|---|---|---|
| **P0** | 删三 bak(先摘 GGUF/调度器笔记进 docs);ForeignSlots 去重 ×12 | 0.5 天 | 低 |
| **P0** | RuntimeConfig 单点解析(59 env → 一结构),热路径零 var_os | 1 天 | 低 |
| **P1** | server.rs 拆 foreign/ + harvest/;exec.rs 仪表迁 tap | 1-2 天 | 中 |
| **P1** | ForwardCtx → StepCtx + LayerView | 1 天 | 中(全层触点) |
| **P2** | gdn/attention 巨石拆分;tests.rs 分主题 | 1-2 天 | 低 |

**纪律固化**(写进 AGENTS/README):
1. `std::env` 只允许出现在 `EnvProvider::from_env` / `main` / `#[cfg(test)]`
2. 声明层(model/layers)禁 eprintln 与 env 直读;诊断走 tap/metrics
3. server 新 foreign 核必须走 ForeignSlots::parse,禁手写槽序循环
4. 新探针必须以 tap 形式进观测面,不得在 exec/model 加特区分支

---

## 附:删除 bak 前摘录(engine-bak 有参考价值的要点,2026-10-03)

- `engine-bak/src/loader/gguf.rs`:GGUF 解析范式 = vendor candle gguf_file.rs 搬运
  (magic/version/tensor-info 读表 + 量化反解 Q4K 等);将来若支持 GGUF 权重从此 resume。
- `engine-bak/src/scheduler/`:continuous-batching 决策参考(join/switch/preempt 语义)。
- `engine-bak/src/kvcache/allocator.rs`:页分配器Alternate实现(与现 blocks.rs 对照用)。
- 以上均已存 git 历史(594bee1 之前的提交链),需要时 `git log -- crates/engine-bak`。


---

## 四、执行实录(2026-10-03 深夜Ⅳ,P0+P1)

| 项 | 状态 | 结果 |
|---|---|---|
| P0.1 删三 bak | ✅ | engine-bak/nn-bak/cuda_bak 移除(GGUF/调度器笔记先摘录);workspace 全绿 |
| P0.2 ForeignSlots | ✅ | `parse_foreign_slots` 单点;6 handler 循环删除;awq 臂修复中被误伤的残留括号一并清理 |
| P0.3 热路径 env 清零 | ✅ | exec.rs 7 步读 → `StepProbes`(boot 解析);server 每发射 2 次 → `ServerProbes`;GpuCtx 持 srv_timing/cap_prof;剩余 var_os 仅 boot/实例化点 |
| P1.4 server 拆分 | ✅ | 2114 行 → `server/mod.rs` 806 + `server/foreign.rs` 933 + `server/harvest.rs` 350(impl GpuServer 跨文件,子模块可及父私有) |
| P1.5 exec 仪表抽方法 | ◐ | gdn_dump 已抽;kv/logits dump 与采样回读交织保持内联(已 probes 门控) |
| P1.6 ForwardCtx 拆分 | ⏳ | 触点全层,下一会话单独手术(设计:StepCtx+LayerView) |
| P2 巨石层文件 | ⏳ | 同上 |

**回归**:models 104/104 ✓;engine 30/30 ✓;27B e2e 文本连贯稳定,731ms/16tok。

**过程教训**:a. bash 正则改码先打印后落盘(awq 臂被误伤残留括号,编译期即拦);b.
`| head` 管道会在 --nocapture 大模型测试中途 SIGPIPE 杀进程 → server 线程带 18.8GB 成
僵尸 → 后续测试 ALLOC/EXECUTION 假回归——**先查 compute-apps 僵尸再怀疑代码**。

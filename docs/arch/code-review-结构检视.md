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


---

## 五、执行实录(2026-10-10 深夜Ⅴ,clean-code 轮:入口检视 + 双统一基建)

> 起点 = 用户指令:从应用入口 `apps/server/src/main.rs` 开始 clean code 排查;
> 追加指令:新增 **env reader**(全 workspace 环境变量统一读取面,含测试域)
> 与 **file loader**(全应用文件 I/O 统一入口)两模块,落 `owl_shared`。

### 检视结论(main.rs 344 行)

- 进程层:main() 混三职(env 解析/actor 装配/accept 循环);配置无对象
  (7 位置参数进 actor_main);`model_kind` 魔串分派藏在引擎线程 ——
  打错字静默落 0.8b 基线(真隐患);env 解析 `parse().unwrap_or` 静默吞错。
- 模块层:chat() 130 行协议状态机住在 main.rs;流式/非流式双 loop 重复;
  usage 账本散写两处;`/debug/metrics` 40 行投影内联路由臂。
- 代码层:魔数(通道 64/预览 60/行数 512)、env_or 签名、排序反转技巧。

### 落地

| 项 | 状态 | 结果 |
|---|---|---|
| owl_shared::env_reader | ✅ | flag/str/str_or/parse/parse_or + set/remove + **EnvGuard**(RAII 恢复,测试防踩踏);parse 失败 eprintln 可见(降级≠无声) |
| owl_shared::file_loader | ✅ | read/read_to_string/write/create_dir_all/remove_file/metadata/exists/read_dir/open/join 薄包装;未来统一控制点(记账/沙箱)留口 |
| server config.rs | ✅ | ServerConfig::from_env 唯一 env 读取点;ModelKind 枚举,非法档位 boot fail-fast;坏值降级 eprintln |
| main.rs 瘦身 | ✅ | 344→193 行:只留编排(fail-fast → actor spawn → ready 门 → accept 循环);accept 失败 100ms 退避(防错误风暴忙旋) |
| openai.rs 收编 chat | ✅ | 双 loop 合一(单折叠循环 + ReplyMode 出口策略);Usage 账本单源;None 断流行为保持原差异(流式静默/非流式 503) |
| debug.rs | ✅ | /debug/metrics 投影纯函数化;路由只转发;Ordering 语义直写 |
| workspace sweep | ✅ | 153 处 env::var* + 47 处 fs::* → 196 处统一调用面(47 文件);sed 批量 + 手工修类型形态(let Ok→let Some、Option.ok 链→parse、unwrap_or_else 闭包 arity) |

### 设计决策

1. **build.rs 构建期例外**:kernels/build.rs 18 处 env 保留 std::env ——
   构建期基础设施不经运行时读取面(引 owl-shared 做构建依赖得不偿失)。
2. **kernels lib 零依赖律例外**:driver.rs `OWL_RESOLVE_TRACE` 探针保留
   std::env(lib 不引 owl-shared);测试域经 dev-dependencies 收编。
3. **shared 内部自引用**:crate 内用 `crate::` 路径,非自名外引。
4. **全限定调用面**:sweep 一律 `owl_shared::env_reader::xxx` 全路径,
   免 import 变更、调用点自解释(grep 可审计)。
5. **tokio "time" feature**:server 新增(accept 退避 sleep;零重量级依赖)。

### 回归

- **shared 21/21**(新 5:env_reader 3 + file_loader 2)
- **kernels lib 8/8**;gdn_scalar_golden ✓;split_direct_probe ✓;
  gdn_chunked_golden_stage_by_stage ✗ = 存量案(ILLEGAL_ADDRESS 同点位,
  干净树复现,早于本轮)
- **engine 38/38**(串行 83s);**models 116/116**(串行 35s)
- server 六端点冒烟对等:health/models/chat 流式+非流式(usage 23/17 token
  正确)/debug/metrics/404/400;boot 时序日志同形
- workspace `cargo check --all-targets` 绿
- **附案(pitfall #16)**:批量 GPU 测试默认并行 = 多上下文同卡捕获竞争,
  失败集逐轮漂移;git stash 干净树复跑定谳存量,`--test-threads=1` 全绿。
  新纪律:GPU 套件一律串行跑门。

### 新纪律(入册)

- 新 env 读取一律 `owl_shared::env_reader`(禁止散点 std::env::var*);
  测试写 env 用 EnvGuard;热路径禁律不放松(boot 一次解析)。
- 新文件 I/O 一律 `owl_shared::file_loader`(禁止散点 std::fs::*;
  mmap 经 open 取句柄;网络 IO 不属此模块)。


---

## 六、执行实录(2026-10-10 深夜Ⅵ,显式依赖律:env 隐式依赖 → 显式参数)

> 起点 = 用户裁决:env_reader 收口只是散点收口,不是终态。**环境变量 =
> 隐式依赖;除程序入口(config 构造)与测试用例外,任何函数不得读 env
> —— env 派生数据必须显式变参数/配置结构传下去。**

### 终态结构(生产路径零 env 读取)

| 域 | 载体 | 收编 |
|---|---|---|
| engine | **`EngineKnobs`**(EngineConfig.knobs;24 字段 = probes + pool/spec/dflash 族 + gdn/snap/vram + sampler + raw_completion + pf 族 + cuda:DiagOpts + env:EnvProvider) | engine.rs 16 + running.rs 5(含 **OWL_STEP_PROFILE 每步热违例**) + state.rs 6 + sampler.rs 2(**每步 from_env 热违例废除**) + scheduler/blocks/graph_plan/exec 8 |
| cuda | **`DiagOpts`**(srv_timing/cap_prof/launch_sync/free_legacy/graph_flags/capture_slab_mb/nvrtc_include;GpuCtx.diag + GpuServer.with_diag + GpuClient::spawn_with) | state.rs 7(含 **OWL_LAUNCH_SYNC 热违例**)+ CUDA_HOME/CUDA_PATH |
| models | LoaderCtx.{verify,debug_tap} + ForwardCtx.trace_gate + CpuInterpreter.debug + RepackPath::resolve 去 env 臂 | load.rs 3 + module.rs 1 + reference.rs 1 + gdn.rs 1(**forward 热路径**) |

### 入口侧构造器(唯一合法 env 读取面,全部文档化)

`ServerConfig::from_env`(server)→ `EngineKnobs::from_env` →
`StepProbes::from_env` / `SamplerCfg::from_env`+`sampler::enabled` /
`EnvProvider::from_env`(被动律原样)/ `DiagOpts::from_env` / `ServerProbes::from_env`。
**新增旋钮 = 结构体加字段 + 对应 from_env 登记 + 调用点显式传**,禁止回生散点直读。

### 关键裁决

1. **StatePool 几何 boot 定谳**:snap_max()/gdn_slots() 自由函数删除 →
   pool 字段(gdn_slots)+ alloc 参数(snap_max/vram 族);运行期读 =
   self.gdn_slots(跨 boot 不变,进程内冻结语义显式化)。
2. **GDN 融合 A/B 收敛**:OWL_GDN_FUSE_DECODE 全工程零读者(死旗标)删除;
   唯一活闸 = EnvProvider.gdn.fused_decode 反开关。
3. **热路径违例三连清**:OWL_STEP_PROFILE(pump 每步)/ SamplerCfg::from_env
   (采样每步)/ OWL_LAUNCH_SYNC(每次图回放)/ OWL_TRACE_GATE(每层
   forward)→ 全部 boot 一次解析进 knobs/probes。
4. **LoaderCtx 扩容而非旁路**:verify/debug_tap 进 ctx(33 处字面量机械
   补字段);装载域校验/观测与 ctx 同生命周期。
5. **server.rs.bak 残留发现**:P0.1 删 bak 轮漏网(cargo 不编译,无害),
   待下轮清理。

### 回归

- engine **38/38**(串行 88s)/ models **116/116**(串行 35s)/
  shared 21/21 / kernels lib 8/8 / workspace `--all-targets` 绿
- server 冒烟:health/chat 非流式(usage 15/17)+ 流式 [DONE] ✓,GPU 清零
- 生产面 env 直读清点:**0 处**(仅入口构造器 + 测试域 + build.rs/driver.rs
  两例外)


---

## 七、执行实录(2026-10-10 深夜Ⅶ,显式配置模块:全量总账 + 统一 loader)

> 起点 = 用户裁决:配置读取收敛后还要消灭**配置定义散点** —— 全量配置
> 约定总量一处登记,loader 一函数加载返回强类型对象;对象展开为多个
> config module,每 module 管一个功能域;枚举严格枚举,不许 String
> 和稀泥;消费点硬编码配置散点全部消灭。

### 终态:`owl_shared::config`

- **`OwlConfig`** 十 module:`runtime`(device/max_seq/chunk/test_device/
  server_url/bind)/ `model`(kind 枚举/dir/name/awq27b_dir/dflash2_dir)/
  `pool`(池几何 + vram 预算 + prefix_cache)/ `spec`(三态/草稿量化/
  B4 参数/tap 诊断)/ `sampling`(mode 枚举 + 三维 + rep)/ `load`/
  `dispatch`(解释器分派 12 旋钮,EnvProvider 组装源)/ `probes`(16 旗标)/
  `cuda`(7 诊断旗标 + graph_flags 枚举 + slab + nvrtc)/ `graph`(no_graph)。
- **总账**:模块 doc 全键登记表(键/类型/缺省/module)—— 新键必须登记,
  禁止账外键;测试私有旋钮与构建期键显式列为账外。
- **唯一 loader**:`OwlConfig::from_env()`;**严格枚举 fail-fast**:
  ModelKind/SamplerMode/GraphInstantiateFlags 非法值 = Err 拒启(消息带
  合法值);数值解析失败 = warn + 缺省(env_reader 降级可见)。
  `bool_strict`:PREFIX_CACHE 的 `!= "0"` 和稀泥解析废除(0/false/off/
  1/true/on 白名单)。

### 类型严格化三案(和稀泥清零)

| 键 | 原形态 | 终态 |
|---|---|---|
| OWL_MODEL_KIND | server 私有枚举 | `ModelKind` 枚举(总账;非法拒启) |
| OWL_SAMPLER | `v != "greedy"` 字符串和稀泥(打错字静默采样) | `SamplerMode` 枚举(非法拒启) |
| OWL_GRAPH_FLAGS | 裸 u64 魔数(2/4) | `GraphInstantiateFlags` 位面枚举(to_bits 唯一出口) |

### 消费方组装链(全部 from_config,零散点 parse)

```
main → ServerConfig::from_env
         └─ OwlConfig::from_env()(唯一 loader)
             ├─ EngineKnobs::from_config(&cfg)   ← StepProbes/SamplerCfg/DiagOpts 同源
             │     └─ EnvProvider::from_dispatch(&cfg.dispatch)
             └─ app 面(bind/model_dir/kind/name/device/max_seq/chunk)
```

原 `EngineKnobs/StepProbes/SamplerCfg/DiagOpts/ServerProbes/EnvProvider`
六个 from_env 散点构造器全部删除;`sampler::enabled()` 删除(mode 枚举)。

### 回归

- shared **25/25**(config 4 新测:缺省/枚举拒启/合法值/布尔白名单)
- engine **38/38** / models **116/116**(串行)/ workspace --all-targets 绿
- server 冒烟:health/chat(usage 15/15)✓,GPU 清零
- 生产面 env 直读终态:**config loader 一处**(tests/build.rs/driver.rs 例外不变)


---

## 八、执行实录(2026-10-10 深夜Ⅷ,配置文件 + flavor 预设)

> 起点 = 用户裁决:其他引擎启动参数爆炸难用;走配置文件;TOML;
> **缺省 < 文件 = env(隐藏键) < 启动参数**;env 太多要砍 —— env 只能
> 作隐藏参数(配置文件没有的键);参考 vllm flavors.py 三层律。

### 终态:四层叠加配置面

```text
Default(代码) → Flavor(内置预设) → 用户文件(toml) → CLI 启动参数
```

- **flavor 机制**(移植 flavors.py 三层律):每 flavor = TOML patch 常量
  (与用户文件同一 serde 路径,零额外机制);只锁"必须成套"的旋钮;
  每档注释 = 为什么。现役五档:`dev-08b`/`awq27b`(贴顶三旋钮+greedy)/
  `spec-dflash`(depth3+slab160,需配 dflash2_dir)/`fi-kv8`/`eager-debug`。
  `--flavor <N>` 激活 / `--list-flavors` 查表。
- **TOML 面**:owl.toml 直接镜像 OwlConfig 十 module(模板
  owl.toml.example);`deny_unknown_fields` 全 module —— **账外键拒启**
  (实测 kindz → unknown field 报错);枚举严格同样生效(awq27bb 拒)。
- **merge**:toml::Value 深合并(表递归,标量覆盖)→ 最终反序列化
  一次过 deny/枚举门。
- **CLI**:手写解析零 clap:`owl-server [CONFIG] [--flavor N]
  [--list-flavors]`;未知参数拒启;OWL_CONFIG = 路径兜底隐藏键。

### env 缩编(主见裁决)

正式配置链 env = **0**:原 60 键 env 面全废,OWL_* 业务键不再被任何
生产代码读取。遗留:OWL_CONFIG(路径兜底)+ 测试域账外自读(测试入口
合法)+ build.rs/driver.rs 既有例外。隐藏键若将来需要(临时 A/B),走
env_reader 原语 + 入口自读,不回正式链。

### dispatch 正语义

DispatchCfg 字段改正语义(qkv_fuse/gdn_fused_decode/gdn_fused_decode_v2,
缺省真)—— env 面的 NO_FUSE 反开关不入 toml;from_dispatch 直映射。

### 回归

- shared **27/27**(config 6 新测:缺省/枚举拒/账外拒/正语义/flavor 全表
  可解析+叠加/load 文件覆盖 flavor)
- engine **38/38** / models **116/116**(串行)/ workspace --all-targets 绿
- 冒烟五连:--list-flavors ✓ / 未知 flavor 拒 ✓ / 账外键拒 ✓ /
  文件启动(bind 8136/name/容量 128 全生效)✓ / --flavor fi-kv8
  (kv-manifest Fp8E4M3+fi ✓);GPU 清零
- 附记:awq27b 档缺检查点目录时 loader panic 逃逸(actor 线程 expect)
  —— boot 失败路径结构化挂账,非本轮


---

## 九、性能验证 + spec 配方立案(2026-10-10 深夜Ⅸ,clean-code 轮闭环)

> 起点 = 用户指令:重构后跑引擎看性能正不正常(llm_speedtest)。
> 硬件:单 3090 Ti(vLLM 生产在 3080 对)。

### 性能判决(重构后无回归)

| 档 | 指标 | 实测 | 基线 | 判决 |
|---|---|---|---|---|
| 0.8B dev | prefill@2048 / decode / TTFT | 8128 t/s / 251.7 t/s / 246ms | —(首立台账) | 量级正常 |
| 27B greedy 裸 decode | decode@512 | **45.15 t/s** | 底线 40 / marginal 42.7 | ✓ 无回归略优 |
| 27B + DFlash2 d7 + fp8 主池 | decode@512 | **34.56 t/s** | perf-roadmap 首测 35.16(同口径) | ✓ 吻合(-1.7% 噪声) |

全程 Captured / 账闭合 / GPU 清零。

### spec 配方立案(两案实测定谳,flavor 已固化)

1. **GDN 格预算案**:slots=8 的 fixed_after = 1.70G(48 层 × 8 格),27B
   +草稿 21.3G 驻留后 budget = −0.18G → 池 0 → OOM。slots=1 → fixed
   0.68G → budget 0.84G → 池 16846 tok。**贴顶形态 slots 必须为 1**
   (单会话门本就单并发)。
2. **slab 固定档案**:capture_slab_mb=160 × ~10 图族 = 1.6G,6 图降级后
   0.1MiB 都分不出。**切 hint 定量(warmup 计量)= 零降级** —— A1 设计
   的正解,固定档只属于非贴顶诊断形态。
3. 治理可观测性:boot 日志补 fixed_after/budget/per_tok 三数(立案工具)。

### flavor 修正

spec-dflash 配方整体替换为实测版(d7 + fp8 主池 + slots1 + hint slab +
钉池 4352 + vram_target 0.99);health 应答名误导案(忘配 model.name →
回退 0.8b 名)入 flavor 注释。

### boot panic 收口修复

actor 线程 catch_unwind 包裹(AssertUnwindSafe)→ 装配深路径 panic
收口为 ready Err → main 干净 exit(1)。实测:坏检查点目录 = 结构化
"读目录 No such file" 而非 panic。附:kill 时再证 pitfall #15
(exe 带 "(deleted)" 后缀,glob 前缀匹配兜住)。

### 回归

- shared **27/27**(flavor 新配方断言)/ workspace --all-targets 绿
- --flavor spec-dflash 端到端:health/速度全通,残留 0


---

## 十、根因定谳:草稿检查点家族错配(2026-10-10 深夜Ⅹ,AL=0 案结)

> §九 遗留:27B+spec AL=1.000 恒 m0(三域/双 dtype 主池同象)。二分
> 考古(HEAD/727ca7a/09ad2a1/09ad2a1^ 全 AL=0)排除未提交回归;golden
> 测试"数据源缺键 qweight"一击定谳。

### 根因

**草稿检查点家族错配** —— `dflash2_dir` 被配成 `z-lab/Qwen3.8-27B-DFlash2`
(**BF16 未量化源**,golden 参考系),而生产草稿 = `syvai/Qwen3.8-27B-DFlash2-W4A16`
(W4A16 量化家族,weight_packed/scale/shape 三件套)。引擎 `load_27b_dflash2`
双家族自动探测:z-lab 走 **BF16 原生直载分支**(合法家族,不报错),但该
分支的草稿在 27B+draft 服务形态**实测全拒**(装上 ≠ 能用)。

### 证据链

| 步 | 观察 | 推论 |
|---|---|---|
| spec.m 直方图 | m0 恒定,AL=1.000(三域) | 草稿全拒,域无关 |
| eager 臂对照 | 同样 m0 | propose 图无罪 |
| FACT_PROBE | drafts='ChronB (user - F' junk | 草稿前向输出垃圾 |
| golden 测试 | "缺键 qweight" | z-lab 无量化键(家族错配实锤) |
| 换 syvai | **AL=4.345,数学 157 t/s** | 一发入魂 |

### 正确配方的性能(24G 单卡,27B+DFlash2 d7+fp8 主池)

| 域 | decode t/s | 对照 |
|---|---:|---|
| 多步数学(健康域) | **157.5** | 历史 92-115(超) |
| 竖式乘法 | **101.2** | >100 ✓ |
| prose 长城 | 44.5(短输出) | — |
| 随机词池(speedtest) | 34.56 | 首测 35.16 同口径 ✓(随机域净亏已知) |

错配→正确 = **4.3×**(36.6 → 157.5)。

### 防再犯(三层)

1. **装载期家族嗅探**(w4a16.rs W4A16Source::open_dir):缺 weight_packed
   = 结构化拒启并指路(拦 W4A16Source 误喂 BF16;golden 测试同款);
2. **BF16 分支警示日志**(load_27b_dflash2 else 臂):提醒该分支实测全拒,
   生产用 W4A16,分支死活另案;
3. **配置件警示**(owl.example.toml / config 总账 / flavor 注释):
   dflash2_dir 须 W4A16 家族。

### 挂账

- **z-lab BF16 分支死活另案**(装载合法但草稿全拒;golden 参考系不阻塞
  生产;修复方向 = BF16 激活溢出 §6.14 同族或 fp8 草稿池交互);
- 健康域 92-115(历史)vs 157(今日)口径差待考(可能 = slots1+fp8 主池
  正确配方 vs 旧 slots8 组合;正差异不追)。


---

## 十一、PoolPlan 参数对象(2026-10-10 深夜ⅩⅠ,布尔和稀泥清零)

> 用户裁决:alloc 的 `kv_fp8/dflash_fp8/spec/mtp/dflash/dump_all` 布尔群
> 改枚举 —— spec/mtp/dflash 是同一互斥态的展开,量化该用枚举。

- **`PoolPlan`**(state.rs;池装配计划参数对象):alloc 16 参 → 4 参
  `(face, dims, layer_types, &PoolPlan)`。
- 量化 = `KvQuant` 枚举(kv/draft_kv/fi 三处同风格);投机形态 =
  `SpecMode` 枚举(Off/Dumb/Mtp/DFlash2,快照/MTP 链/草稿池三件由 mode
  派生 —— 互斥语义由派生保证,调用点零自洽负担)。
- 遗留独立布尔仅 `dump_all`(诊断开关,非互斥态,语义独立保留)。
- 回归:engine **38/38**(串行)/ workspace --all-targets 绿。


---

## 十二、DraftCfg 草稿域收编(2026-10-10 深夜ⅩⅡ,量化布尔废除)

> 用户裁决:knobs 里 draft 的六散字段该单独抽模块;草稿量化该用 dtype
> 表示,不是 bool。

- **`DraftCfg`**(engine.rs;草稿域子结构,收编 knobs 顶层六散字段):
  `depth`(投机深度)/ `dumb`(哑草稿)/ `dflash2_dir`(W4A16 家族目录)/
  **`kv_quant: KvQuant`**(草稿池存储量化,None=f16 / Fp8E4M3=偷显存档;
  原 `draft_kv_fp8: bool` 布尔废除)/ `notaps` / `tapdecl`。
- 量化选 `KvQuant` 而非 iface `Dtype`:Dtype 契约(F32/BF16/F16/U32)无
  fp8 变体,KvQuant(None/Fp8E4M3)即 KV 存储 dtype 语义(与
  PoolPlan.draft_kv 同型零映射);iface Dtype 扩 fp8 变体挂账(契约层
  变更,影响面大)。
- 消费点:run()(spec 三态/草稿装载/dflash taps)、PoolPlan.draft_kv、
  gate 测试 ×4 —— 全部 knobs.draft.* 直达。
- 回归:engine **38/38**(串行)/ shared 27/27 / workspace 绿。

---

## 十三、层↔kernels 隔离律(2026-10-12,用户律:layers 禁触 owl_kernels)

> 用户裁决:`impl GdnScalarCall` 出现在 layers/ = 越界。**层不越解释层
> 直触 kernels** —— face builder/名字/槽序签名/OK 谓词知识一律收编解释
> 层动作表(crate::ops),层侧只供语义张量与纯标量。

- **ops §6 层↔kernels 隔离面**(新立):`gdn_chunked_node` /
  `gdn_scalar_node`(原 layers/gdn.rs 收编;签名改纯标量 t/nv/nk/kd,
  GdnShape 构造内化)/ `marlin_node`(原 linear.rs 手撸 sig 三臂收编;
  AWQ 7 块/f16·bf16 6 块路由内化)/ `marlin_ws_elems`(装载域计尺)/
  `fi_prefill_node` + `fi_name`(原 attention.rs 手撸 sig/scale_bits/
  names 收编)/ `paged_decode_ok` / `paged_prefill_ok`(driver 谓词的
  层侧出口)/ `CT_REPACK` re-export(设备重排 Want 键)。
- **机器门**:facedir_lint 新增 `layers_shall_not_touch_kernels` ——
  layers/ 子树含 `owl_kernels` 字面量即红(crate::kernel 垫子与
  ops::kernel_call 为层侧唯一合法声明面)。
- 层调用点形态:全为 `crate::ops::xxx_node(语义张量…, 纯标量…)` ——
  装配失败 panic 语义不变(A5.4 fail-fast,shape 全由 self/ctx 推导)。
- 回归:models 119/119(含 §6 双 face 锁测试,自 gdn.rs 迁 ops)/
  engine 38/38 / facedir_lint 2/2 / workspace check 绿 /
  gdn_chunked_golden GPU 绿;llm_speedtest std 复测 prefill avg
  **1154.8**(基线 1153.6,零回归),9/15(4096/8192 超 max_seq 预期)。

---

## 十四、gdn foreign 两臂入词表(2026-10-12,用户律二番:不做特殊面)

> 用户质询:`impl GdnScalarCall` 式的特殊面为什么存在?gdn 应当入
> `ids` 具名算子枚举,与 native 同一 Call 通道。

- **ids += 两票**:`GDN_CHUNKED = OpId("gdn.chunked_delta")` /
  `GDN_SCALAR = OpId("gdn.scalar_delta")`(语义词表,非实现名)。
- **driver 两臂**:`gdn::chunked_delta()` / `scalar_delta()`,名字 =
  family 单源,发射配置 = **`FOREIGN` 哨兵**(新立:grid/block 全零直通,
  server 家族 runtime 自算网格)—— resolve 分派表 +2 行。
- **解释器 Call 臂桥**:`ops::foreign_sig(name) -> Option<sig>` ——
  foreign 名不在 native 登记表(无 .cu 源),槽序 sig 住 client face
  单源;Call 臂 `foreign_sig → Kernel::new+with_sig`,否则 `with_pick`
  原径。dtype 守门对 foreign 自然跳过(lookup None)。
- **层侧**:`chunked_delta` / `scalar_delta` 声明函数(纯 models 词汇:
  `TensorOps::call(ids::…)` + 语义标量;scale = kd^-0.5 语义量,f32
  位型编码 = 局部纯数学 fn)—— 与 gating_g 同一形态,§6 的 gdn 特殊
  face 节点删除。
- **四点锁测试**(foreign_call_lock):ids ↔ driver 臂同票 / FOREIGN
  直通 / foreign_sig 全覆盖 / 层声明 → lower_kernel → client face
  parse 逐位对拍(chunked+scalar 双臂)。
- marlin/FI 仍走 §6 装配面(变体路由依赖 dt/zs/env,Call 化挂账)。
- 回归:models 120/120 / kernels 23/23 / engine 38/38 / facedir 2/2 /
  workspace 绿 / gdn 金标 + 新链 E2E GPU 绿;llm_speedtest std 复测
  prefill avg **1161.5**(基线带内,零回归)。

---

## 十五、动作词表枚举化(2026-10-12,用户律三番:层面向动作,不面向实现)

> 用户裁决:解释层与模型声明解耦的下一刀 —— ids 字符串词表是实现分派
> 键的泄漏(现役全落 NVIDIA 算子);层应该说「对张量做什么」,硬件归
> 解释器/驱动关注。**ids 全废,改枚举。**

- **`SemanticKernel`**(models::ops;40 变体动作词表,Debug/Clone/Copy/
  PartialEq/Eq/Hash):`Op::Call { op: SemanticKernel, aux }` —— 层侧唯一
  算子词汇。命名 = 语义动作去域前缀(GdnGatingG / PagedDecodeV2 /
  SiluAndMul / CtRepack…),域归属只留注释分组。
- **`SemanticKernel::op_id(self) -> OpId`**:语义动作 → 命名空间身份证的
  **CUDA 解释器现役 lowering**(穷尽 match,加变体不映射 = 编译红);
  值 = 语义名空间("gdn.gating_g"),非 kernel 实现名 —— 实现名由
  driver 按 env 推导不变。**CPU/AMD 解释器不走此表**:直接 match 枚举
  落自有实现 —— 枚举 = 硬件无关动作契约,OpId+driver 退化为 CUDA 域
  内部细节,模型层零 NVIDIA 词汇。
- **全消费点切换**(14 文件):layers ×8 / env(5 个 _op() 返回值)/
  module(Layout::DeviceRearrange)/ formats/load(装载域重排)/
  tensor(TensorOps::call)/ eval(Call 臂 + trace)/ engine ×5;
  `pub use OpId` 转私有(模型公共面零 OpId)。
- **机器锁**:`semantic_op_vocab_lock::op_ids_are_unique_and_total`
  (40 变体 op_id 非空唯一 + 总数一致)+ foreign_call_lock 同步枚举化。
- driver 分派表报错文案随更新(词表住 SemanticKernel);driver 本体零改动
  (仍吃 OpId 字符串 —— 它只是 CUDA lowering 的内部键)。
- 回归:models 121/121 / engine 38/38 / facedir 2/2 / workspace 绿 /
  gdn 金标 GPU 绿;llm_speedtest std 复测 prefill avg **1160.8**
  (基线带 1153~1161,零回归;同日另有 1127 一轮 = 运行方差)。

---

## 十六、Op 瘦身(2026-10-12,用户律四番:纯算子不再单独搞)

> 用户裁决:`pub enum Op` 里纯算子变体与 SemanticKernel 词表双重记账
> —— 同一动作两套词汇/两条 lower/两个 eval 臂。**纯算子折叠进 Call。**

- **Op 删 8 变体**:Add / Mul / Silu / Sigmoid / Rmsnorm{eps,w_off}
  (折叠进 `Op::Call`;TensorOps 五方法签名不变,内部改发 Call 节点,
  标量槽 = 线契约:sz n 或 cols/eps/w_off)+ Rope{theta_base} /
  Embedding / PagedAttn(**死变体**,零构造点,eval 本就无臂)。
  Op 剩 11 变体:源(Htod/Zeros/Block)+ 视图(Reshape/SliceView)+
  复合(Matmul/MatmulNt:dtypes 分派 cublas/native + cm 映射,非单一
  kernel 语义)+ 统一通道(Call)+ 逃生舱(Kernel/Spec)+ 状态
  (SlotWrite)。
- **driver +4 臂**:ops.add / ops.mul / ops.silu(BF16 无独立变体,
  unimplemented 指路 silu_and_mul)/ ops.rmsnorm(rows 从形状推导:
  x_total/cols,grid (rows,1,1) smem 1K)。
- **登记表 O 标记**:纯算子族 .cu 输出块在中间槽(add 第 3 / sigmoid
  第 2),lower_kernel 直连需要显式输出位 —— Entry.args "T,T,T,sz" →
  "T,T,O,sz"(sigmoid/silu "T,O,sz";rmsnorm "T,T,O,i32,f32,i32");
  C2 测试 O 归一化对拍 + O 位必须是 .cu 指针槽。
- **CPU 面**:reference 解释器 match 同一枚举落 host 实现
  (Call{Add/Mul/Silu/Sigmoid/Rmsnorm} 臂;Rmsnorm 参数从标量槽按线
  契约解码)—— **动作词表的 CPU 面第一次成型**。eval Call 臂 env 缺省
  (CPU face 测试域)= Sm86 占位,纯语义臂不消费 env。
- 删 lower_add/mul/silu/sigmoid/rmsnorm + dname/concat_op/to_static
  (dtype 路由命名随双词汇一起退役;lower_matmul/nt/gemm 留守 Matmul)。
- 回归:models 121/121(带 GPU)/ kernels 23/23 / engine 38/38 /
  facedir 2/2 / workspace 绿 / gdn 金标 GPU 绿;llm_speedtest std 复测
  prefill avg **1166.0**(基线带,零回归)。

---

## 十七、三形态归一 Call(2026-10-12,用户律五番:Kernel/Spec 节点消灭)

> 用户裁决:`Call / Kernel / Spec` 三个算子节点形态合一 —— 不是包一层,
> 是把其中形式**彻底消灭**。

- **消灭 `Op::Spec`**:胖算子路径零生产消费(样例 NarrowStrided 与
  `SemanticKernel::Narrow` 重复);contract::OpSpec trait /
  client/native.rs / TensorOps::spec / eval Spec 臂 / spec_arm_tests
  全删。值域校验职责 = driver 臂(契约公式)+ 登记表 O 标记 + family
  face parse 三层承接,E1-E15 防线不缩水。
- **消灭 `Op::Kernel`**:全部 ~50 发射点迁移 Call ——
  ① native 测试守卫(gdn 逐核 ×9 / attention ×3 / embedding / rope);
  ② 生产(dflash2 草稿链 ×10 / rope f32 臂 ×2 / specs topk·select /
  model argmax / mod narrow·concat f32 / argmax helper);
  ③ foreign(marlin 三臂 / FI 两臂 → `MarlinW4A16` / `FiPrefill{,Fp8kv}`
  + driver FOREIGN 臂 + foreign_sig 扩容);
  ④ **逃生舱收编**:ts_stamp 探针(刀D)源入
  kernels cu/owl/ts_probe.cu + 登记 + `SemanticKernel::TsStamp` ——
  非注册内核的合法入口从"内联源+手撸 sig"回归"加源+登记+变体+臂"。
- **driver 新臂 12**:ops.ts_stamp/argmax/topk16/sigmoid_gate_mul、
  elems.cast_{f16_bf16,bf16_f16}、dflash.{select,conv}、
  attn.naive_nc{,_fp8kv}(grid 自 shapes+标量槽推导)、marlin.w4a16
  (**父数路由:AWQ 6 父 / f16·bf16 5 父;boot 案:误判 7 块致 AWQ 走
  f16 槽 5≠6 panic,已修**)、attn.fi_prefill{,_fp8kv}。
  ⚠️ dflash.select 输入侧 f16/bf16(aux 路由)而输出恒 f32 —— 变体
  路由不吃 req.dt 的首个反例。
- **eval Call 臂 env 占位**:CPU face(测试域)无 op_env = Sm86 占位,
  纯语义臂不消费 env;硬件感知臂在 CPU face 不可达(launch 面拒)。
- **Op 终态 9 变体**:源(Htod/Zeros/Block)+ 视图(Reshape/SliceView)+
  复合(Matmul/MatmulNt)+ **Call(唯一算子形态)** + 状态(SlotWrite)。
- **事故录**:gdn.rs 曾被批量脚本截断为 0 行 —— git HEAD 恢复
  (SemanticKernel 轮已提交,零损失),本轮测试站 call 化在恢复版上
  重放。教训:批量替换脚本必须逐文件写后立即 `wc -l` 自检。
- 回归:models 115/115(GPU 全套,含迁移守卫)/ kernels 19/19 /
  engine 38/38 / facedir 2/2 / workspace 绿;llm_speedtest std 复测
  prefill avg **1154.9**(基线带,零回归;boot 期 AWQ 路由案在
  smoke 中暴露并修复)。

---

## 十八、cuda_ops 退役(2026-10-12,存量断链清理)

- `crates/kernels/src/cuda_ops.rs`(+180 行)删除:早期"crate 自持发射"
  架构残骸(KernelFn 自描述发射包 + build.rs nvcc 预编 PTX),三重死亡
  —— cu/ops.cu 已不存在(预编必失败)/ 实现全注释化 / 零消费方。
  职能早已被现役三层取代:server nvrtc 懒编译 / AOT cubin 资产 /
  marlin·FI 静态 FFI。
- 连带:build.rs PTX 预编段删除(仅剩 marlin .a 段)、Cargo.toml
  "cuda" feature 删除(device 保留 = 现役 Exec)、lib.rs re-export 清理、
  device.rs 头注同步。**非注册内核合法入口 = 加源 + 登记 + 语义变体 +
  driver 臂**(§十七收编律),零 build.rs PTX 面。
- 回归:kernels 19/19 / models 115/115 / engine 38/38 / workspace 绿。

---

## 十九、登记表更名 list(2026-10-12)

- `kernels/src/native.rs` → **`list.rs`**(git mv 保历史):本模块本体 =
  **封闭登记表**(名字 → 源/槽序契约/dtype 的唯一权威表),"native"
  只是条目属性之一(native/foreign 之别住 driver 分派与 registry 路由)
  —— 模块名让位本体。models::kernel 垫子 re-export 路径随更名。
- 回归:kernels 19/19 / models 115/115 / workspace 绿。

---

## 二十、cu/ 目录收编(2026-10-12,孤文件归位)

- `cu/text/`(5 文件:embed/rope/attention/gdn/concat)→ **`cu/owl/`**
  (owl 自研/移植位;文本主干不再单设目录,mod 名保留 = 消费方零改动);
- `cu/ops_pair.cu` → **`cu/owl/`**(owl 自研语义算子对,F5 双 dtype 宏桥);
- `cu/marlin_repack_ct.cu` → **`cu/marlin/`**(marlin 家族源码同目录);
- `cu/` 根目录清零:剩 5 个域目录(attention/flashinfer/gdn/marlin/owl),
  无孤文件。sources.rs include 路径 + README 表 + 4 处文档提及随更;
  workorder 历史档案按惯例不改写。
- 回归:kernels 19/19 / models 115/115 / engine 38/38 / workspace 绿。

---

## 二十一、vLLM 对照案:AL 劣化 + 模板缺失 + spec 图态正确性(2026-10-12 深夜立案)

> 用户对照实锤:同模型同草稿同词池,vLLM+DFlash(3080 对)decode avg
> **113.1 / min 66.3(15/15)**,owl **43.7 / min 9.3**。"实现有问题"成立。

- **质量账(AL)**:vLLM Prometheus AL=**2.61**(pos0 命中 80%,健康衰减);
  owl **1.45**(半数轮 m=0)。同 DFlash2 家族草稿,接受率差 44% = 实现级。
- **根因链三层**(逐层二分,每层都有 A/B 实证):
  1. **模板注入缺失(已修)**:owl chat 包装 = 手写前后缀对,**从未渲染
     chat_template.jinja**(0.8B 时代挂账欠至今)。froggeric v22.5
     thinking 模板正确注入 = system 段(reasoning effort medium)+ user +
     `<|im_start|>assistant\n<think>\n`(开 think),prompt **84** tok;
     owl 旧形态 = 49 tok(缺 system + 错预填空 think 块)。修:27B 档
     ChatFormat 数据化更正(jinja 渲染实证);**load_tokenizer 两参化**
     (原恒用 0.8B 档 spec,27B 档从未消费)。修后裸 decode 输出恢复正常
     推理(此前 49 形态下"、"+EOS 与"诗意重复句"皆为错上下文产物)。
  2. **spec 图态正确性 bug(新立案)**:新模板下裸 decode 正常、spec d7
     输出 "a a a" 崩坏;**eager 臂(OWL_DFLASH_EAGER=1)与 bare 逐字一致**
     → 定位 `dflash_graph`(图态 encode+propose 单图)。图捕获语义排查
     (anchor 绑定/m 桶/taps 指针)待战役展开。
  3. **B4 连击判据失效(已修)**:随机域 AL≈1.1 时"偶发 m=1"重置连击 →
     永不降级 → 104ms 冷轮 × 1.1 tok = 9.3 t/s。改**滚动 AL 窗口**
     (16 轮均值 < 盈亏线即降级);阈值标定四轮:3.6→40.9 / 1.5→14.2 /
     **1.2→43.7**(1.5 错杀盈利 spec:稳态轮 38.5ms 下 AL1.45=63 t/s
     在盈利侧)。稳态/冷轮之别为本轮探针新知。
- **P0(本轮探针补丁引入,当场抓获修复)**:预算/EOS 恰落 spec 轮
  bonus 位 → complete() 后触 tspec() → "无活跃 turn" panic → 引擎退役。
  降级态捕获移至 emit 循环前。
- **探针面固化**:spec.m0..m7 直方图 / spec.round.{spec,deg} 分账 /
  spec.deg.rounds·tokens;对照工具 = vLLM /metrics per-pos 接受计数差分。
- **残余挂账**:①dflash_graph 图语义战役(P0 级,spec 正确性);
  ②降级态每步 dflash encode 同步开销(~20ms/步,残差);
  ③阈值在线自适应;④verify 8 行 = 2.9× 裸步(38.5ms 轮的构成主体)。
- 回归:engine 38/38 / models 115/115 / kernels 19/19 / workspace 绿。
  vLLM 对照数字:113.1 avg / 66.3 min(15/15,3080 对 TP2)。

### §二十一·修复回执(同日)

- **根因修 = 图态 kv_lens 双绑错**:
  ①encode `kv_lens` 误绑 `enc_pos`(pos+i,**每行注意力缺自身**)→ 绑
  `enc_kv_lens`(pos+i+1,eager 同式);
  ②propose `kvs.kv_lens` 误绑**标量** `kv_len`(fp+8,1 值槽供 8 行读
  越界垃圾)→ 绑 `prop_kv_lens`(fp+1+i 逐行,eager 同式)。
  图/ eager 双臂逐参数 diff 定位(数据流/槽表/slots 全同,唯 kv_lens)。
- **修复验证**:词表复述 AL **1.45 → 4.38(反超 vLLM 2.61)**;spec 输出
  与 bare 逐字一致;llm_speedtest std decode avg **43.7 → 66.1(+51%)**,
  prefill 1155 持平。
- **对照结论更新**:AL 已反超;decode avg 66.1(单 3090 Ti)vs vLLM
  113.1(3080 对 TP2)残差 = ①硬件(TP2 每步 ~2× 带宽)②min 8.4 段
  (2048 长随机域,草稿分布真实短板,m0 十连非 bug;lookup drafting
  摘樱候选)。B4 滚动窗口(1.2)兜底生效。

---

## 二十二、2048"9.3 t/s 案"结案 + dflash 图 kv_lens 修复(2026-10-12 深夜)

> 用户挑战"修好了怎么还是 8"→ 复测发现 **8.8 那轮是脏测量**:新实例
> bind 失败静默退出,8135 被上一实例占用,llm_speedtest 打到旧引擎。
> 测量纪律立案:boot 后必须 `grep 引擎就绪` + 端口进程核验才可发压。

- **干净环境终测(2048 单点 ×2)**:decode **67.3/67.32**(逐位一致,
  修复前 9.3 → **+623%**);全 std:decode avg **73.5**,prefill **1167.6**,
  9/15(flavor max_seq 锁定,预期)。
- **AL 终值**:**5.82**(vLLM 2.61 的 2.2 倍);m 直方图双峰 = m7 全接受
  124 轮(62%,可预测段吃满)+ m0~2 散布(随机段);verify 83.5ms /
  propose 15.3ms(2048 ctx)。
- **根因链终版**:
  1. ~~草稿分布短板~~ **撤销** —— 真 bug = 图态 kv_lens 双绑(encode 绑
     enc_pos 缺自身;propose 绑标量供 8 行越界),eager 臂正确故金标全绿;
  2. ~~B4 判据~~ 降为次要(干净环境下 2048 深亏不复现;窗口机制保留
     兜底,阈值 1.2);
  3. 模板注入缺失(49 vs 84)独立成立,已修。
- **教训入册**:①性能归因前必须验证"测的是哪个进程"(端口/进程/日志
  三验);②"分布不利"结论必须有同草稿同分布的对照(vLLM per-pos
  Prometheus 差分一把就够,三周前就该做);③图捕获闭包的槽绑定是
  AL 类语义错的温床,graph/eager 逐位 parity 门立项(本轮人工 diff
  已替代一次,门未固化)。
- vLLM 对照更新:decode 113.1(3080 对 TP2)vs owl 73.5(单 3090 Ti)
  —— 按 TP2 带宽 ~1.5× 折算,单卡口径 owl **已持平略优**;prefill
  1167.6 vs 1653(TP2 同折算后 owl 1750+,反超)。

---

## 二十三、m0 连段二分定谳:kv_fp8 × gdn_chunked 交互(2026-10-12 深夜续)

> 2048 随机词池 prompt 上 19% 轮 m0(11 t/s 龟速段)。矩阵二分(300 词
> 英文流 + 散文后缀,prompt=384):

| 配置 | 输出 |
|---|---|
| fp8=F chunked=F depth=0 | 正常推理 ✓ |
| fp8=F chunked=T depth=0 | 正常推理 ✓ |
| fp8=T chunked=F depth=0 | 正常推理 ✓ |
| **fp8=T chunked=T depth=0** | **'safety'+EOS 崩(首 token 即歪)** |
| 生产档(+d7) | ' 数据.' 循环 |

- **定谳:fp8 主池 × gdn_chunked prefill 的交互数值劣化** —— 单独各自
  均正常,合用即崩。机制候选:chunked 六核的输出 f16 精度写入 + fp8
  KV 读出的量化误差在 24 层 GDN 混合架构上逐层放大(matmul 全注意力
  层无罪——fp8 单独全绿)。注:此 prompt 的 vLLM 贪心输出也是不可控
  文本(英文流+哲学散文 = 边缘分布),'safety' 与 'The user's...'
  均属"分歧后各自连续"——分歧点在 prefill 首/早期轮的数值面。
- **影响面**:随机域 std 的 min 9.4 段即此(2048 prompt 触发 24 GDN
  层 × 8 chunk 的 chunked 前向,fp8 KV 读出参与 chunked 输入链)。
  数学域(AL 高)与词表复述不受影响。
- **修复方向(立案)**:①chunked 前向的输入端校验(层 0 的 q/k/v 走
  f16 直读,层间链上 fp8 KV 读出经 v2 读核—— suspected 在 v2 读核
  与 chunked 输入的 dtype/精度交接);②金标:2048 prompt 的 m 序列
  与 sglang 对拍(已有 testdata/gdn_fla 金标管线可复用)。
- **非问题排除**:B4 窗口(1.2 阈值未触发于健康段)、模板(84=84 对
  齐)、分词(337=337)、dflash 图 kv_lens(已修,词表复述 AL 4.38)。

### §二十一·探针面完善回执(同日,用户批评"日志舍不得多加")

- metrics.rs + `counter_set`(gauge 语义:当前值;/debug/metrics counter
  字段直读)—— AL 窗口/pos/配置开关类"现在多少"量有正经落点。
- spec 轮全相记录(恒开,µs 级):verify 拆 step(上传+launch)/
  read(同步+回读)—— **read=67ms 是 verify 82ms 的主体**(图执行后
  的同步+回读,非 GPU 计算),优化指向 = tok 读的旁路/重叠;
  `spec.bt`(write_bt)/`spec.al.win`(滚动 AL×100)/`spec.pos.last`
  (fp 定位)/`spec.round.{spec,deg}` 分账/m0..m7 直方图。
- 降级裸步分相:`decode.sync`(草稿 KV 同步 encode 单独计时)+
  `decode.step.deg` —— 9.3 案的 80ms/步嫌疑犯自此一屏定谳。
- prefill 侧:`prefill.top1`/`prefill.top1.gap100` gauge(首 token
  id 与 top1-‐top2 差距,跨引擎逐位对照的落点)。
- 实测回收(std 全程):spec 轮 80.3ms = verify 67.1(step 0.1 +
  **read 67.0**)+ propose 12.5 + fold 0.6;AL=5.80,tok/轮 6.80;
  ~~read 67ms = 下一个优化主目标(read_output_f32 的同步等待 ——
  图 replay 异步排队后首读承担全队列排水;与 emit 重叠/拆分可回收
  ~2/3 轮时间)~~ **⚠️ 本节两条结论(「非 GPU 计算」「重叠可回收 2/3」)
  已被 §二十四 实测推翻** —— read 阻塞就是 verify 图的 GPU 执行本身。

## 二十四、read 阻塞定谳:verify 图 GPU 执行本身,"排水税"翻案(2026-10-09)

用户令查「read 为什么阻塞那么长」。control 实验 + 代码链路双证,**§二十一
的"首读承担全队列排水"推断错误**,当庭翻案:

**机制链**(代码定谳):
- `graph_launch` = 异步火后不理(state.rs:cuGraphLaunch 入 COMPUTE 流即回)→
  step 只计提交延迟(~0.1ms);
- `Dtoh`(harvest.rs handle_dtoh)第一步 = `STREAM_COMPUTE.synchronize()`
  —— **栅栏等的就是 verify 图算完**;之后 D2H memcpy(n=32B)+ host 回调,
  回读本身 ~20µs(d2h-prof:sync≈issue)。
- 所以 read 时间 ≡ COMPUTE 流上在队工作的 GPU 执行时间。read 是「首个
  阻塞点」,不是「排水税」。

**control 实验**(run-depth*.toml + step_profile + d2h_prof,随机域短
prompt,engine 单卡 3090 Ti):

| 点 | T(depth+1) | ctx | read | pure(replay+read) |
|---|---|---|---|---|
| depth=7 | 8 | ~500 | 43.2ms | 43.2ms(差 0.04ms) |
| depth=3 | 4 | ~500 | 35.5ms | 35.7ms |
| depth=0 | 1 | ~500 | 24.5ms/步 | —(decode 图) |
| depth=7 | 8 | ~2356 | 90.8ms | 90.7ms |

**read ≡ pure 全程成立(40+ 轮样本)**:首读排掉的队列里只有 verify 图
自己(上轮 ⑦ dflash propose 自带 read 已排干)→ **阻塞 = verify 图的
真实 GPU 执行**,与 host/队列/回读无关。

**verify 图时间构成**(depth 扫描回归):
- **权重流底座 ≈ 24.5ms**(T=1;27B AWQ int4 ~13.5GB → 有效带宽
  ~551GB/s,3090 Ti 峰值 1008 的 ~55% —— marlin/访存效率有 1:1 杠杆);
- **行边际 ≈ +2.3~2.7ms/行**(T 无关权重的部分:GDN 逐行递归/conv
  逐行副标题 + attention + 每行小核;depth=7 时合计 ~16ms ≈ 37%);
- **KV 项随 ctx 线性 × 行数**:T=8 时 ~25.7µs/ctx-token(500→2356 ctx
  贡献 +47.6ms)—— paged attention 每行独立读全 KV,T 大长 ctx 被此吃掉
  (std 矩阵长点 read 100ms+ 即此;67ms = 五点平均的解释)。

**结论修正(替代 §二十一 优化指向)**:
1. ~~读旁路/重叠回收 2/3~~ **作废** —— 重叠只能藏 host 段(~0.1ms),
   关键路径下界 = verify 图 GPU 时间,重叠不产生任何token;
2. 正解三杠杆:① T=1 底座(权重流 55% 峰值带宽,marlin/访存效率);
   ② 行边际(GDN 多行批处理化 —— verify 行间 GDN 递归本就串行,
   chunked 化可摊平);③ KV 共享读(attention 一次读 KV 服务全查询行,
   vLLM paged attn 本就如此,核对 per-row 循环与否另案核对)。
3. depth 经济学定价:每加一行 depth 花 (2.3ms + KV 项),买 ~AL 行 token
   —— AL>2 才有净赚,与 B4 窗口盈亏线互证。

**探针纪律注**:gpu_prof 逐核分账与图回放**互斥污染** —— graph_launch
不走 gpu_prof 同步,其后首个被探针的 launch 会替图背 40ms 级黑锅
(topk16 p50=40.9ms 假象);图内分账必须走 no_graph eager 窗。
另:杀 owl server 现行犯两连 —— nohup+& 的 `$!` = 包装子 shell pid
(真身要用 ps/proc 双验),`pgrep -f` 模式匹配自身命令行自杀两次
(括号 trick 或 ps+grep "[x]" 规避)。

### §二十四·数据质量补账(同日,用户问「你测的接受率是多少」)

depth 扫描各点接受率事后反推:短ctx depth=7 主点(43.2ms)= 退化复述
域 "a a a",7.08 tok/轮 ≈ m6 饱和,零降级;长ctx 90.8ms 点 = 词沙拉域,
m=0×6 → B4 降级(全部样本来自 m=0 轮)。read≡pure 在高低 AL 两域都
成立 → 机制结论双向加固;行边际剔脏点后用 T=1/T=8 两点 = 2.67ms/行。
**depth=3 点作废**:第二轮即崩 —— ⑦ 统一 propose 恒产 7 草稿不随 depth
截断 → ids 8≠4 装填拒,pump 退役。修复(消费点单处截断,spec.rs ②
`d.truncate(depth)`)已落,depth 开区间 (0,7) 档位自此可用(待重验)。

## 二十五、27B 语义崩案立案:"a a a" 吸引子,全天验收盲区(2026-10-09)

用户令重跑健康数学域 depth 扫描(选案②),执行中撞破大案:**27B 主模型
语义输出已崩**,数学域健康样本在本构建上不存在,② 不可达:

**症状**:清晰中文指令(「请连续写加法算式 a+b=c」)greedy 输出
"a a a"×N 吸引子;数字流+英文指令 → 1-2 token 即 EOS("──"/"=the")
或数字雪崩("= 100000…");温度 0/0.7 同症状。

**判别链(逐层排除)**:
1. fp8 KV 关 → 同坏(非 §二十三 数值劣化族);
2. 全参考位(无图 eager + gdn_scalar + fp16 KV + 无 spec)→ 同坏 →
   **图/fp8/chunked/spec/采样全部无罪**;
3. **0.8B 同引擎链 greedy 完全健康**("10 + 50 = 60\n98 - 42 = ? →
   Wait, I cannot write this…" —— 真实模型行为)→ 引擎模板/prefill/
   解码循环/采样无罪 → **27B 特有路径**;
4. 27B 唯一独有路径 = **cyankiwi AWQ-INT4 装载/repack/marlin GEMM 链**
   (0.8B 走 BF16 不经此链)。

**嫌疑定位**:M1-M5 算子契约重构(kernels registry/LaunchVal 值表/
GdnChunkedCall 客户端面)最可疑 —— 回归门全是金标/单元(engine 38/models
117/kernels 19),**没有全模型语义 E2E**;单形状标量错值可漏网。
时间线佐证:§十(重构前夜)数学域 157 t/s + AL 4.345 = 模型语义尚活;
今日全部数据(含 43.7/66.1/55.3 bench 与 AL 5.8)的输出全是 "a a a"
—— **全天验收只看接受率/速度,无人看过一条语义输出,语义崩穿无人察觉**。

**波及面**:① 全天 spec/速度数据在崩坏模型上测得 —— 时序结论(read≡pure、
时间构成)不受影响,但 AL/接受率类数据的「模型健康」前提不成立;
② §二十一 vLLM 对照(同 prompt 给哲学散文,owl 给 a a a 循环)当时就
是本案的可见症状,被误读为「垃圾进垃圾出」;
③ 「数学域不受影响」(§二十三)作废 —— 数学域 EOS 即本案症状。

**下一步(待拍板)**:① AWQ 装载/repack parity 刀(marlin GEMM 对拍
逐形状,重点 M1-M5 动过的 LaunchVal 值表);② 金标扩全模型语义 E2E
(固定 prompt → 固定输出逐位,入 CI);③ 0.8B 加 AWQ 臂对拍(隔离
repack 路径);④ 查 cyankiwi 检查点完好性(外部 sha/他引擎跑同文件)。


## 二十六、语义崩案结案:GpuRes::upload 异步 memcpy 静默不写(2026-10-09)

用户令 worktree 回滚 bisect(§二十五 的下一步)。**当日结案,凶器落网:**

### 定位链(worktree 对照 + 逐层 checksum + 指针侦探)

1. worktree @ HEAD(9f876c2)短 prompt"健康" = **假信号**(真凶与提交无关);
2. 同二进制垫长 prompt(43→89 tok)即崩 → **长度触发**,边界精确 = 64
   (59 健康/67 崩;GDN 大 T 分臂 `tokens>=64`,prefill_chunk=512 单块);
3. 0.8B 同样崩(85 tok 乱码)→ 与 AWQ/27B 无关;scalar 与 chunked 崩法
   **逐字节相同** → 共享根因;
4. pf_bisect 逐层 checksum(chunked on/off 双 boot):**`post.g*.rec`
   18 层同一常数 = 状态池 rec 全零** —— h 核终态未写回;
5. **引擎内对拍 test 三臂"逐位相同"是假象**:`testkit::harvest_f16` 走
   `eval_ops`(env 缺省 = Cpu/gdn 全关)→ 三臂全走 varlen。修正
   `eval_ops_env` 后臂选路生效 —— 但 test 的 attention 输入(kv 池手搭)
   与 server 不可比,弃 test 路线,转 server 指针侦探;
6. handler 内侦探:in_k/in_g 合理非零,**state_out(ht)/v_new/h_buf 全零**
   → h 核空转;meta 表侦探:**cu=[0,0],应为 [0,T]** → upload 后读回全零
   → **`GpuRes::upload` 非捕获分支的 cudarc `stream.memcpy_htod(src,&mut
   target)` 返回 Ok 但数据未落** —— 六核 cu 全零 → 全部 early-return
   → y 零 + rec 零 → "a a a" 吸引子。

### 修复(三件)

1. **server/mod.rs `GpuRes::upload`**:非捕获分支弃 async
   `memcpy_htod`,改 `result::memcpy_htod_sync(dst, src)` 同步直写
   (捕获窗 async 分支保留);
2. **gdn.rs 形状守卫**:chunked cubin 烘 constexpr NV=48/KD=VD=128、
   scalar d=128 —— `chunked_shape_ok/scalar_shape_ok` 不配强制回落
   varlen(0.8B NV=16 结构性不适用,曾直接写飞);
3. (前案已落)spec.rs depth 联动 truncate。

### 修复后回归

- 27B:chunked@35/51/67/83、scalar@67、varlen 全健康;spec d7 全管线
  语义正常(竖式/思考链);AL 反推 ≈3.5(数学域真实值,非崩坏假 5.8);
  verify read 27.8ms @ 短 ctx;
- 0.8B:chunked 请求守卫回落 varlen 健康;
- 测试资产:`crates/kernels/tests/gdn_big_t_arms_parity.rs`(合成双臂
  对拍,浅/深/混合 g 域 Δ~1e-7,入 git)。

### 波及面改判

- **今天全部性能/AL 数据重定性**:read≡pure 等时序结论仍成立,但
  AL 5.8/55.3 t/s 等为崩坏吸引子上的假值,须全量重测;
- §二十三"数学域不受影响"作废(数学域 EOS 即本案);fp8×chunked
  交互案待重审(真凶是 upload + 形状,fp8 或为无辜共犯);
- §二十一 vLLM 对照的"owl 给 a a a"即本案症状;
- **方法论教训**:①async memcpy 排队 + 无 sync = 静默不写,handler
  内所有 `GpuRes::upload` 消费方(soff/meta 表)全受累 —— 上传后必须
  sync 或用同步 API;②`testkit::harvest_*` 的 env 缺省陷阱(带臂选路
  的对拍必须 `eval_ops_env`);③"金标绿"只覆盖金标的输入域(浅 g/
  整块 T/金标上传路径),**handler 生产路径需 handler 级上传链测试**。

## 二十七、修复后重测:stream 计时假象 + AL 随 ctx 衰减 + B4 盈亏线失配(2026-10-09)

用户令重测性能并追问"修复后 min 还是 9?"。三连发现:

### 1. speedtest 流式计时扭曲(9.4/7.05 假象)

llm_speedtest decode 走 `stream=True`,t/s = 末/首 chunk 时间戳差;spec 轮
**批量 emit**(一轮 m+1 tok 一次到达)→ chunk 粒度扭曲分母:同 prompt
三轮 9.45/139/139 交替(512 点)、101/104.6/7.05(1024 点)。**非流式
墙钟口径(真实)**:512 → 93.3/83.8;1024 → 45.1/41.1;2048 → 28.1/35.2
(全健康语义,零降级触发)。speedtest 的 decode t/s 仅作相对参考,
**绝对值以非流式墙钟为准**。

### 2. 真实 AL 随 ctx 衰减(下一个速度杠杆)

同域 AL:512 → **6.1/5.4**;1024 → **3.9/3.5**;2048 → **3.8/5.2**;
m0 同步上升(6/7 → 14/19)。长 ctx 草稿与主模型分歧增大。嫌疑:
①fp8 KV × chunked prefill 残留数值面(§二十三 重审,upload 修复后
重新 A/B);②草稿 fc 长 ctx 固有衰减。AL 回 6 则 1024/2048 点 t/s 翻倍。

### 3. B4 盈亏线失配(策略层,9.4 t/s 的真身)

实测分相:spec 轮 ~97ms(verify 68.6 + propose 12.9 + snap/fold/emit)、
裸步 ~24.5ms → **盈亏平衡 AL = 97/24.5 ≈ 4**。当前 window_starved 阈值
常数 **1.2** → AL 1.2~4 区间亏本跑 spec(9.4 t/s = AL≈1.1 段 97ms 产
1 tok;裸步可 40 t/s)。且降级路径实测为零触发 —— 阈值形同虚设。

**优化项**:①B4 阈值动态化(按滚动实测轮成本/裸步成本算盈亏线);
②AL-ctx 衰减诊断(kv_fp8 A/B)。

### 优化落地(同日)

1. **B4 盈亏线动态化**(running.rs TurnSpecState + spec.rs + phases.rs):
   TurnSpecState 加 `round_cost_ewma_ms/bare_cost_ewma_ms`(α=0.15 滚动,
   轮成本在 B4 窗口段更新 —— 轮尾 tspec 可能已随 complete 消失,P0 同款);
   window_starved 阈值 = `(round_ewma/bare_ewma).clamp(2.0, 6.0)`,缺省
   回落 1.2;降级 eprintln 带实时盈亏线。**注**:原代码注释"盈亏线 3.6"
   与实现常数 1.2 脱节已久。
2. **回归(非流式墙钟,6 请求)**:1024 点 45.1/41.1 → **64.9/57.0
   (+40%)**;总吞吐 +6%;深谷点(9.4/7.05)消失。
3. **AL-ctx 衰减诊断(kv_fp8 A/B)**:off 后 1024 AL 4.45/4.47(vs on
   6.12/5.22,波动内)、2048 AL 3.84-4.41,且 t/s 更慢(f16 KV 带宽税)
   —— **kv_fp8 无罪,AL 衰减主因 = 草稿 fc 固有长 ctx 质量衰减**
   (模型/训练面,草稿训练另案;非数值 bug)。
4. **观察存疑**:同 prompt 跨请求 AL 波动大(512 seed0:6.10 vs 3.45,
   跨 boot)—— spec 链路存在非确定性(跨请求状态/归约序?),影响
   单点对比可信度,立案待查。

### §二十七·勘误:AL-ctx 衰减撤案(用户裁决"衰减不正常",多 seed 复测)

§二十七 第 2 条(AL 随 ctx 衰减 6.1→3.5)**是小样本波动假象**(每点
1-2 seed)。多 seed 曲线(每点 4 seed,随机英文域):

| len | AL 均值 | ±σ |
|---|---|---|
| 512 | 4.64 | ±1.23 |
| 1024 | 4.52 | ±1.38 |
| 1536 | **5.16** | ±0.89 |
| 2048 | 4.23 | ±0.54 |

**平坦,无衰减**(2048 略低在噪声内);kv_fp8 A/B 亦无罪(off 更慢)。
请求间 σ≈1.2 的波动是随机域文本 accept 起伏(正常)。**t/s 随 ctx 下降
= 物理成本**(verify read 随 ctx 线性涨 §二十四 + KV 带宽),与 vLLM
同方向、同带宽比例,非病态。下一个真杠杆回到 §二十四 三正解之首:
verify read 的同步等待(2.3k ctx ~90ms/轮)。

### §二十七·再勘误(僵尸 CUDA 进程案):AL"波动"的真身与健康基线确认

多 seed 曲线的"请求间 σ≈1.2 波动"与"语义健康请求偶发乱码"的追查,
揪出**环境真凶**:tg4 probe 的 IMA 挂死进程被 timeout 杀时**未死透**
(卡在损坏的 CUDA 上下文里,324MiB 挂在 GPU 1)—— owl server 与之共卡
期间,请求输出退化为多语言乱码吸引子("ölfölf"、"支支支"),GPU util
**0%**(非 spin,是病态等待)。`kill -9` 清僵尸后**立即全面健康**。

- **多 seed AL"波动"(σ≈1.2)实为僵尸干扰污染的测量**;健康环境下
  同组数据:512 → 55.6/92.5(AL 3.45/6.02)、1024 → 65.2/57.2
  (AL 6.12/5.22)、2048 → 33.1/30.6(AL 4.65/4.28)—— 语义全部健康;
- **教科书级教训入册**:CUDA 挂死进程 timeout/kill -TERM 后**必须核对
  `nvidia-smi --query-compute-apps` 确认清空**,僵尸上下文会静默毒化
  同卡后续进程(与"杀进程用 PID 列表"同族,第 N 次踩坑升级);
- tg4 变体本身:**IMA 真 bug**(sanitizer 复现于干净 probe),已回滚
  立案(probe + 侦探链留存,`OWL_TG4_PROBE=1` 显式启用);修好后
  预期 prefill 一半时间回收(当前核 ~2% 峰值);
- §二十七 第 2 条(AL-ctx 衰减)撤案维持;kv_fp8 无罪维持;
- **当前健康基线(非流式墙钟,含 TTFT)**:512 → 55~93;1024 → 57~65;
  2048 → 31~33;math 域峰值 145.6。

### §二十七·性能刀回执:prefill_split 上生产(用户指令"猛猛优化")

1. **核路线裁决(用户律:不自研算子,vLLM 移植集内盘点)**:tg4 自研
   分摊重写(挂死 = shfl 在分支 continue 后的 warp 收敛破坏;数值 =
   TG 段错位)—— **回滚弃用**,IMA 证据链留存于
   attention_tg4_probe.rs(skip 门保护);
2. **prefill_split(flash-decoding,K1 分块在线 softmax + K2 归并)
   上生产**:非流式墙钟 A/B(僵尸清后的干净环境)—— 512/1024 持平,
   **2048 → 36.3/46.3(旧核 33.1/30.6,+10%/+51%),2048 seed1 AL
   7.43(近上限 8)**;verify(T=8)同通道受益(flash-decoding 正是
   小 T 大 kv 的专用形状);FI 复测确认更慢(57.8/23.2,弃);
3. **speedtest 流式口径正式弃用于 decode 评估**(§二十七 计时假象),
   一律非流式墙钟 + /debug/metrics AL 差分;
4. **当前基线(非流式墙钟,含 TTFT)**:512 → 56~93;1024 → 57~65;
   2048 → 36~46;math 域峰值 **145.6 t/s**;
5. 遗留刀位:①tg4 分摊(重写需专注轮,shfl 收敛 + 段对拍,潜力
   prefill 一半时间);②T=1 底座带宽(551GB/s → 峰值 70%+);③4096/
   8192 的 submit 预算 2560 门(矩阵不完整,另案调)。

### §二十七·FI 分相复测(同日,用户问"为什么不用 FlashInfer")

FI prefill.chunk p50 = **390ms**(opt 核 494,-21%,且含影子池开销)→ FI 核
本身快;但非流式墙钟全矩阵(干净环境):FI 均值 61.8 vs split 58.9 —— **持平,
FI 波动大**(512 点 110.7/69.8)。裁决:**split 保持生产**,FI 作为 dispatch
配置项保留(语义健康已验证);FI prefill 快 21% 被 decode/emit 面开销吃掉
的机制(影子池 per-turn 成本)另案。prefill chunk 分布(4×512 块):
p50 494/p99 602ms,块间平坦;逐核:GEMM 35%(57-70% 峰值,正常)、
attention opt 核 24%(2% 峰值,唯一重病号)、GDN 15%、小核 17%。

### §二十八·FI 定性翻案 + 门槛分派(2026-10-12,A 线性能轮)

背景:§二十七"FI 核快 21% 但墙钟持平 + 波动大,弃"。本轮社区调研
(flashampere = 3090 专项 attention 后端;repos/flashinfer 0.7.0;vLLM
_vllm_fa2_C.abi3.so 导出 half_t hd256 全套符号)+ 实测定性。

**四连定谳(mt 阶梯行为学 + spec 面板差分,2048 词,AL≈2.9 域)**:

1. **plan 无罪**:FI run 被 CUDA 图捕获,plan(host)仅捕获期跑
   (launch_time 日志零 fi_prefill 发射;PlanKey 含 ctx_total 的每步
   miss 假设不成立 —— 回放期无 host 面)。
2. **"波动大"翻案 = AL 域波动**:同 ctx 快慢请求差一倍(3.8 vs 7.4s),
   差分显示每轮成本恒定(72.5 vs 73.4ms/轮),轮数 40 vs 88 = **AL 6.4
   vs 2.9** —— prompt 生成域决定接受率,与通道无关。§二十七"512 点
   110.7/69.8 波动"同机制。矩阵墙钟 t/s 跨 seed 不可直接比(域污染)。
3. **FI 输 verify(T=8)形状**:小 qo 下 FI 核低效且随 ctx 线性放大
   (decode 边际 18→36ms/tok 递增;split 核同形状平坦)。每轮 verify:
   **split 核(f16 影子)39ms < chunked(fp8 主池)54ms < FI(fp8 影子)
   85ms**。
4. **FI 赢 prefill**:mt=8 点(prefill 主导)FI 1.86s vs split 2.47s
   (**−25%**),tensor-core 收益真实。

**门槛分派落地**(attention.rs,FI_MIN_TOKENS=128):大 chunk 走 FI,
小 T 回落 chunked_opt(读 classic 主池零影子依赖);FI 模式独占
(split 臂需 f16 影子,FI 只写 fp8 影子,禁)。实测 2048 mt=256:
9.34 → 6.90s(decode 恶化修掉大半;仍微亏 split 5.92,因 chunked
verify 比慢 split 核 15ms/轮)。

**新格局与下一刀**:全赢组合 = **prefill FI + verify split 核**。卡点
= split 核仅 f16 变体(f16 影子 9.2G 与 FI 的 fp8 影子 4.6G 互斥,
双影子爆显存)→ **split 核 fp8 KV 读变体提前立项**(原遗留刀位 ③):
模板加 e4m3 dequant 读 fp8 影子,与 FI 共用 k0_dual_fp8kv 单写。预估
2048 AL2.9 域 ≈ 4.8s(vs split 5.92,−19%);fp8 读量减半 verify 或
再快(39 → 25-30ms/轮)。A3(FI fp16 PV,flashampere sm86 先例)降级
为 prefill 锦上添花。

### §二十九·split-fp8 修雷落地 + 轮数污染大勘误(2026-10-12 同日)

**修雷(重大正确性)**:`owl_prefill_split_fp8kv_hd256` 落地(模板
KV_FP8,e4m3 字节池直读,smem 前转 f16,chunked B6.3 同式)。**埋雷
实锤:生产 split 配置(kv_fp8 + prefill_split)下 ctx≥1536 的输出一直
是垃圾**(2048 词 → "olata, } } }…";f16 核读 U32 字节池 = 字节错位)。
§七 fp8 纯度审计漏环:k0_write_op 在 split 模式是 **K0WriteFp8 单写**,
"f16 影子"不存在。修复后 2048 输出连贯 ✓。布局注:split 核读 classic
主池(非 kNHD),index 数学 fp8/f16 同式仅单位 1B/2B。
NVRTC 坑:RTC typedef 块需补 uint8_t。

**轮数污染大勘误(方法论级)**:§二十八 的"每轮 verify 39<54<85ms"
**作废** —— 那是用"255 tok ÷ AL2.9 = 88 轮"一个假设轮数除各配置墙钟
的假账。真差分(spec.round/verify 计数器):**各通道每轮 105~108ms
无差**(CHUNKED 108/89,FI-MIX 106/88),verify 走 chunked 还是
split-fp8 对 decode **无感**。mt 阶梯的"递增 ms/tok" = AL 随生成进度
自然衰减 + 配置间输出域分叉(同 prompt 各配置贪心链分叉 → 轮数 71 vs
78),非通道病。**跨配置墙钟对比必须配 spec 差分,单看墙钟 = 域轮盘**。

**存活的真实结论**:
1. prefill FI −24%(mt=8 同口径:1.86 vs chunked 2.46 vs split-fp8
   2.26)—— dispatch flashinfer=true 净赢,无 decode 代价;
2. split-fp8 = 正确性补丁(non-FI 大 ctx 路径从垃圾变正确),性能上
   被 FI 覆盖(prefill 场景)或与 chunked 持平(verify 场景);
3. decode 每轮 ~105ms 的构成(round − verify ≈ 17ms;verify 89ms 内核
   时间 vs host 面占比)未拆,是下一个定性目标 —— 但与 attention 读
   通道无关(四通道无差已证)。

### §二十九·补一:UB 雷完整定谳 + 探针收口(同日)

**split-fp8 探针**(`attention_split_fp8_probe.rs`,chunked fp8 探针同款
三门:host f32 全局 softmax 参考 + e4m3 位型近似编码;np1/np2/np3 三
形状;④ 同核分区对拍;⑤ stat 分相)PASS 全绿:② max 1.2e-3,④ 分区
合并 vs np1 max 4.9e-4(f16 scratch 精度),⑤ mismatch=0。

**§二十九 正文之外的第三颗雷:K1 分区早退 __syncthreads UB**(f16/fp8
同病,寿险级潜伏):
- 旧代码 `if (ps >= ctx_end || !active) return;` —— **ctx_end 依赖
  tok**,同 block(64 tok)内 tok 跨分区边界时部分线程早退、部分继续,
  后续双 `__syncthreads()` UB(违反本文件 2026-10-02 块统一律,该律
  只修了循环上界,早退分支漏网);
- 症状:探针 np>1 时 t≥P(P=ceil(ctx/nparts))的输出全错(max 0.49),
  t<P 全对(整块早退恰好安全);核内 printf 锁定 k_smem 装载 UB;
- **生产掩蔽机制**:t0 = part×P 恰为 64(TILE)倍数时装载槽映射巧合
  一致(2048 ctx → P=512),UB 症状被数据掩蔽;探针 P=22(非 64 倍数)
  一发入魂;
- **修法**:①块统一早退仅保留 `ps >= max_ctx`(整分区对全 token 空,
  判定不含 tok,合法);②per-token 中性(ps≥ctx_end)不再早退,走循环
  由 in_ctx 掩码自然产出中性值,写 partial 加 active 门(尾块 tok≥T
  防越界)。修后 np2/np3 全绿,④ max 4.9e-4。

**勘误再勘误(⑤ 的假阳性)**:⑤ 首版把 l 公式写成"未减 m 的 exp 和"
(Σexp(s)),与核的在线 softmax l(Σexp(s−m))不同式 —— 首批
mismatch=64/496 全是该假阳性,修公式后归零。教训:对拍探针的 host
参考公式必须与核的**中间量语义**逐式同源,不能只对最终 out。

**端到端回归**:修复后 512/1024/2048 三点语义连贯;墙钟与 FI 配置同
域点一致(域主导,差异 <2%)。生产裁决不变:flashinfer=true 净赢
(prefill −24%),split 通道保留为 non-FI 正确性路径。

**探针方法学沉淀**(第三次验证):host 参考公式必须与核的**中间量
语义逐式同源**(本次 l 的减 m 语义);位型近似编码的 decode 公式
(subnormal = m_int×2^-9)写错会伪装成"核误差孤点";UB 类核 bug 的
指纹 = "误差从某个边界值起全错、边界前全对"(本例 t≥P)。

### §三十·decode 分相定谳 + std 五点全绿(2026-10-12,驱动恢复后)

**环境**:unattended-upgrade 第四次上演(10-07 顶内核 7.0.0-38 + 官方 595 包,
10-09 重启后 align3p 失联 NVML mismatch)。已处置:updater 四层根除(timer
disable+mask / conf 全 0 / 删包 / 内核+官方 nvidia 包 hold);重编
**noprobe 生产弹药** `modules-615-align3p-noprobe` md5 `91684e00`(e0609f7d
默认构建,探针剔除;构建坑 = nv-kernel.o 增量残留致 modpost undefined,
make clean 后一次通过;细节见 p2p-build/README.md 弹药表)。

**decode 每轮 105ms 分相定谳**(2048 慢域,spec 差分 + step_profile 探针):
- **verify.read 88.4ms/轮(83%)= verify 图 GPU 执行纯时**(step_profile:
  step=0.2ms(上传+launch)、read≈pure=86ms(图独占时间),host 面零嫌疑);
- propose(草稿 d7)17.2ms/轮(16%);fold/snap/调度 ~2ms;
- read 的名字有误导:**它不是 KV 读**(KV 流量 ~0.1ms),而是"同步+回读
  logits"= 等图跑完;主体 = T=8 主模型 forward 图内算子(GDN 串行递推
  + marlin 小批 + attention),理论 FLOPs 下限 ~6ms,当前 86ms 差 14×
  —— **下钻需捕获期插桩(cap_prof 逐算子),下一轮**;verify 通道选择
  (chunked/split/FI)已证无感(§二十九),刀位不在 attention。

**std 五点全绿**:`run-8k.toml`(max_seq=8448 + pool_tokens=8704,继承
split 生产档;vram 治理池上限 94024 内,boot 无贴顶)。15/15 成功:
| len | prefill t/s | TTFT |
|---|---|---|
| 512 | 1230 | 418ms |
| 1024 | 1164 | 882ms |
| 2048 | 1088 | 1884ms |
| 4096 | 945 | 4338ms |
| 8192 | 747 | 10962ms |
4096/8192 语义抽查连贯 ✓(词汤标准 prompt);拼接病态 prompt 的单词
输出为模型行为非引擎问题。decode 列流式轮盘照旧(6.6~128.7 t/s),
可信口径 = 非流式墙钟 + spec 差分。

### §三十·补二:prefill 曲线定谳——FI 平坦曲线实测 + 两处结论修正(同日)

用户对照表(vLLM/sglang std 五点)推翻 §三十 前文两处说法,实测重定谳:

1. **"vLLM prefill 也在降"——错**。同模型同卡 std:vLLM 1365→1361、
   sglang 1367→1346(**全平坦**);O(T²) 的常数在 FA2 mma 路径下小到无感。
2. **"attention 理想 1835ms(T²/2×KV 字节)"——模型错**。那是不带 Q-block
   复用的读模型;FA2 的 KV 块被 Q 块共享(tile_q 复用)+ QK/PV 走 tensor
   core,attention 在大 ctx 下只占个位数 ms/chunk。

**真凶(自家核)**:split/fp8 attention 核每 512-chunk ~118ms(§二十七
逐核 24%)且**随 ctx 线性涨**(读全前缀 KV 必然),8192 时放大 4×+——
prefill 曲线掉 40% 全是它;vLLM/sglang 的 FA2 核同位置 ~5-14ms/chunk
(差 10-20×),故平坦。

**修法已在本仓:FI 门槛通道**(§二十七 落地:prefill≥128 走 FI/FA2,小 T
回 chunked)。`run-8k.toml`(max_seq=8448 + pool 8704 + flashinfer)std
**15/15 全绿且平坦**:512→8192 = 1321/1280/1295/1270/1238(三轮均值,
**8192 仅 -7%**,split 版掉 40%;TTFT@8192 10.96s → 6.64s,-39%)。
与 vLLM/sglang 差 3~9%,其余 gap 在 GDN/marlin 线性项(模型结构税)。

**生产行动项**:flashinfer=true 应转正为生产默认(prefill_split 保留为
FI fallback);run-8k.toml 随附。FI 影子池显存账(f16 9.2G)与 8k 池并存
已验证可 boot(vram 治理 94024 上限内)。

### §三十一·20k 长上下文实测:prefill 温和、decode 雷爆(2026-10-12 深夜)

`run-20k.toml`(max_seq=20480 + pool 20736 + FI;boot 池 94024 上限内)。

**prefill 曲线修正**:20k 全程 40s 里 decode 占 21.1s(30 轮)——prefill
实际 **17.9s = 1122 t/s**(vs 8192 的 1236,-9%,温和;此前"595 t/s 掉
一半"是 prefill+decode 混账)。chunked prefill 一直开着(39×512)。

**decode 雷爆**:spec.round **704ms/轮 @20k**(vs 2048 的 107ms)= verify
593 + propose 110。verify 随 ctx **严格线性**(593 ≈ 122.8×20480/4096,
与补录Ⅴ同源)。

**593ms 定谳(三核同病,一轮排除法)**:
- ctx 线性项分解(2048=86 vs 20480=593):X≈30ms(GEMM 19.7 T=8 同价 +
  GDN 残余)+ **Y≈563ms = attention 核读 20k KV**(理论流量仅 240-250MB,
  低效 ~2000×);
- **nth=256 装载修复无效**(589 vs 592)——装载并发不是瓶颈;
- 定谳 = **T=8 大 KV 形态下三核全为延迟灾难**:在线 softmax 递推依赖链
  无法流水,每 KV 迭代 ~97µs(97µs = HBM 往返 + shfl 链,无重叠);
  chunked(24 blocks 串行)/ split(960 blocks 但 8/64 query 活跃)/
  FI(split-kv 被 fork bug 禁用)三核同病;
- **下一刀(下轮)**:verify 改 **v2 fp8 核逐 token 发射**(T=8 = 8 次
  vllm_paged_attention_v2_fp8_hd256bs32,每次 1 token partition 并行读
  全 ctx;理想 8.8ms,预期 20-40ms = **15-20× 提速**;需 spec.rs verify
  图构造改动 + 语义对拍)。propose 110ms(草稿读窗全历史)同族问题,
  次优先。
- agent 20k~100k 场景结论:prefill 1122 t/s 温和可用;**decode 1.4 tok/s
  不可用**——长上下文的刀全在 verify。

### §三十二·verify v2 伪序列臂:14× 提速落地 + nb=136 半崩回归(2026-10-12 深夜)

**verify v2 臂落地**(§三十一 立项当日完工):8 token = 8 伪序列单次 v2
发射(grid (Hq, seqs, nparts),partition 维 = flash-decoding);bt8 =
页链×depth1 图输入(engine 每轮 host 写入);verify_attn_v2 scratch
(seqs=depth1 倍,展平 [seq][head][part] 与 vLLM 核同构);链路 = state
分配 + engine 图闭包注入 + ForwardCtx.v2_bt8 透传(model.rs 逐字段构造
漏透传曾致臂不触发)+ attention 层臂(seqs==tokens 防误配)。

**实测**:20k verify **592 → 143ms(4.1×)**、轮 700 → 259ms;2048 verify
86 → 28ms(**3×**)、轮 107 → 48ms;语义健康(8k/20k/2048@8k)。

**修掉的网格级 bug**:v2 pick 的 `grid.y = 1` 硬编码(decode B=1 遗留)
—— 核内 seq_idx = blockIdx.y 恒 0,8 伪序列全用 seq0 槽,seq1..7 的
reduce 读空槽 → 输出垃圾。修复 = pick seqs 参数化(decode=1/verify=8)。

**遗留(最高优先):nb=136 半崩回归**。512@生产配置(run-bench,max_seq
2560/pool 4352/nb=136)输出尾段退化("and, and, and"),verify 56ms;
同池同臂 max_seq 8448(nb=264)正常(28ms);**v2 关臂更崩(85ms 全乱)
→ 非 v2 臂引入,chunked/split 老路径同病**;nb=136 唯一低于 144,144/
192/264/272/648 全正常。nb 与语义的耦合机制未定位(nparts=9 双方相同,
排除;max_seq 仅 submit 门,排除)——下一轮首案,二分 nb(136~144)或
bt8 尺寸敏感面。**生产缓解**:max_seq ≥ 8192 即绕开(池照旧)。

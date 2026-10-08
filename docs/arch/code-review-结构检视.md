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

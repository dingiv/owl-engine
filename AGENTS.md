# owl 包内工作约定(AGENTS.md)

> 本文件是 **owl 包内的操作手册**(不是历史)。
> 架构事实 = `docs/devlog/canon/charter.md`;文档总索引 = `docs/README.md`;
> 依赖地图 = `VENDOR.list.md`;逐轮施工台账保存在本地工作区(不入库)。
> 本文件旧版全文(含 2026-09-22 → 10-16 压缩历史)归档在本地台账区。

## 一、本包是什么

owl = graph 与 TP 一等公民的**从零架构推理引擎**(立项 2026-09-22)。
生产引擎仍是 vLLM;xinfer 冻结存档(判例库)。

**里程碑**:M0 骨架 ✅ → M1 图治理最小闭环 ✅ → M2 TP2 comm ✅(双 3080 生产档)
→ M3 模型 + DFlash2 ✅(27B 数学域 175-177,超冲刺线 165)→ **M4 冲刺线(进行中)**。

## 二、必读(按序,10 分钟)

| 文档 | 讲什么 |
|---|---|
| `docs/README.md` | 文档总索引(10 主题 + 逐篇审阅台账) |
| `docs/devlog/canon/charter.md` | 设计公理 A1 图治理 / A2 通信 / A3 同步隔离 / A4 vendor / A5 严格显存预算 / A6 池契约 |
| `docs/devlog/canon/requirements.md` | 需求基线(xinfer v1.0 全量继承;三大指标三线) |
| `docs/devlog/dev/README.md` | 开发者入口:配置 / 测试 / 调试 / 性能 四件套 + 新工作树开箱 |
| `docs/devlog/graph/graph-model-seam.md` | 图 × 模型层接缝(改执行路径前先看) |
| `docs/devlog/memory/store.md` | 状态存储三层(GPU 池 → host pinned → 磁盘) |
| `crates/backends/README.md` | **第三方 CUDA 准入铁律**:能触 CUDA 的库必须源码审核账外分配;会私自 cudaMalloc 的库不用,或 copy 进来改造 |
| `VENDOR.list.md` | 依赖 / 参考源地图(FROZEN / OWNED / REFERENCE / BINARY 四分类 + vendor 树索引) |

## 三、工作纪律(硬约束,违反即返工)

### A. 代码与架构

1. 上层代码只 import `owl-iface`,禁止直接 include 后端内部类型;
2. **一切设备分配过 `owl-cuda` Governor 账本**(A5.2),且**每个分配点必须报语义名**
   (零未命名是硬不变式,`debug_assert` 拒空名);池外原语(cuBLAS workspace 等)必须预分配并登记;
3. FROZEN 快照(`candle-gb`(外部存档)、`attention.rs`、`cudarc-gb`)只读;
   要改语义 = 走 A4 显式认领,不许顺手 patch;
4. 需求基线 = `canon/requirements.md`;加格式走 `models/new-format-checklist.md`,
   加/换算子走 `kernels/kernel-admission.md` + `kernels/operator-contract.md`;
5. 文件及时拆分:原则上 **≤1000 行**,核心复杂模块 ≤2000 行;
6. **禁全局变量与线程变量**:变量构造写成构造函数导出,由调用者创建与维护;
   依赖不足就写进函数参数,不要嫌麻烦 —— 变量从哪来由调用者负责;
7. 新增配置旋钮 = 结构体加字段 + `from_env` 登记 + 调用点显式传参;
   **生产路径零 env 直读**(唯一合法读取面 = 入口侧构造器族);
8. env / fs 调用一律走 `owl_shared::{env_reader, file_loader}`;
   `owl-kernels` 保持 lib 零依赖(需要日志走 `LogSink` **注入**);
9. **拒启必须响亮**:配置错误 / 资源不足一律结构化 `Err` + 非零退出;
   禁止静默 exit、禁止静默降级(静默拒启 = 运维事故,拒启类错误用裸 eprintln 恒输出);

### B. 测试与门

10. 测试用例用 `#[owl_test]` 宏(`crates/macros`);资源清单 `testdata/resources.toml`;
11. **GPU 套件必须 `--test-threads=1`**(并行 = 多 CUDA 上下文捕获竞争 ⇒ 假挂);
12. 跑门前**轮询等空卡**(`memory.used < 500MiB`)。未守护的 e2e 会**伪造"正确性回归"**,
    三签名:①失败断言每次不同 ②伴随 `[vram-ledger] alloc 失败…驱动 free≈0` ③守护空卡复跑即全绿;
13. **恒等 / 真值类门必须 spec-off 跑**(spec 臂 token 级非路径稳定);
    spec 改动的硬门 = greedy spec ≡ no-spec **逐位一致**,任何时候不许豁免;
14. 只在长输出显形的缺陷必须用**真机长回合**做门,短探针会给假绿;
15. **args 档覆盖 profile**:凡"改了 flavor 就生效"的直觉都是错的(深合并只覆盖列出的键);
    参考臂要**显式写全**相关键,否则静默继承(踩过三次);

### C. 设备与卡

16. **UUID / BDF 钉卡,禁 CUDA 序号**(数字序会被重解释成混插对);
    卡位只走 `profiles/*.toml` 的 `[parallel].device`;
17. 杀进程用 `nvidia-smi --query-compute-apps=pid` 列表 + `/proc/pid/exe` 双验;
    **禁用 `pkill -f`**(模式会匹配自身命令行);
18. 本工作区多会话共用卡:起机前查卡、`/proc/vmstat` 看 `oom_kill`、被外部 kill 是常态
    (判定顺序 = oom_kill 计数 → 同序列复跑 → 问人);
19. **owl 启动 <30s**,超过即有故障;不要 sleep 超过 30s 去等启动;
20. 测量口径:`llm_speedtest`(外部存档) 统一尺(`--flavor std` 五点三轮)+ 引擎内 metrics 优先
    (客户端逐帧计时会把 SSE 开销混进单步);**禁裸用 wall ÷ tokens**;±20% 逐次方差警觉。

### D. 协作与流程

21. **git 提交须等用户明确指示**,不自行 commit / push;
22. 新 worktree 一律建在**包外同盘**的 worktree 根下:`<worktree-root>/<名字>/`;移动 / 清理只经 git 原生命令
    (`worktree move` / `remove --force` + `prune`;分支 tip 已并入 main 时 `git branch -d`);
    删前先归档独有物(台账 / 本地配置 → 本地台账区);
23. 近期任务与计划写本地台账(不入库),做完写一句话压缩;
24. 改了接口 / 旋钮 / 端点 / 门控 ⇒ 顺手改 `docs/devlog/dev/` 对应段与 `docs/README.md` 索引。

## 四、常用入口

```bash
# 构建
cargo build --release -p server                  # 生产(release 零计时税)
cargo build --release -p server --features perf  # 基准(计时器保留)

# 预检 / 生效配置 / 单键解释(不加载权重,~10–20ms)
./target/release/server -p profiles/<档>.toml --check
./target/release/server -p profiles/<档>.toml --print-config
./target/release/server -p profiles/<档>.toml --explain runtime.max_seq

# 起服务 + 诊断(默认 127.0.0.1:8135)
./target/release/server -p profiles/<档>.toml --log-level info
curl 127.0.0.1:8135/debug/log            # 在线调级 / 换 tag 过滤
curl 127.0.0.1:8135/debug/metrics/<族>   # 引擎内真值(优先用这个)
curl 127.0.0.1:8135/debug/perf/epoch/json

# 统一测速(与 Web 前端同内核)
cd llm_speedtest(外部存档) && uv run cli --api-url http://127.0.0.1:8135/v1/chat/completions \
  --model <名> --flavor std

# 测试
cargo test --workspace --lib                          # CPU 面
cargo test -p owl-engine --lib -- --test-threads=1    # GPU 面(必须串行)
```

**新工作树开箱**(三条隐性前置:triton venv / 0.8B 资产软链 / testdata 本地面 + 冷建 ~9 分钟)
= `docs/devlog/dev/README.md` 同名章节,照抄即可。

**配置面**(schema v2,2026-10-21 起):`profiles/full.toml` 是唯一全量档,
新档 `extends = ["full.toml"]` 只写差异;**合并后缺键 = 拒启**;卡位只走 `[parallel]`。
详见 `docs/devlog/dev/config.md` 与 `profiles/README.md`。

## 五、文档地图(devlog 10 主题)

| 主题 | 什么时候看 | 代表卷 |
|---|---|---|
| `canon/` | 定架构 / 评审 | charter · requirements · async-runtime |
| `graph/` | 改执行路径 / 图捕获 | graph-model-seam · prefill-graph · piecewise-graph · fi-single-graph · decode-out-of-graph |
| `comm/` | 动 TP / 通信 | tp-overview · tp2-design · **tp2-log(全账)** · tp2-roadmap |
| `kernels/` | 加 / 换算子 | kernel-admission · operator-contract · w4a8-family · kernel-error-ledger |
| `models/` | 加模型 / 格式 / 装载 | loader-interpreter · gguf-format · new-format-checklist · vision-wiring |
| `spec/` | 投机解码 | dflash2-design · mtp-design · mtp-longctx-* · spec-decode-survey |
| `memory/` | KV / 存储 / 长 ctx | store · kvmem-selection · longctx-attribution · prefix-deletion |
| `serving/` | 服务面 / 会话 / 配置 | agent-surface · multi-session · cross-session-batch · config-ux |
| `perf/` | **调性能 / 工程纪律 / 看实测数字**(先看这张表) | **optimization-candidates(工作清单)** · perf-framework · decode-cycle · device-command-ledger · stream-ordering · log-framework · pitfalls-general · pitfalls-cuda · qwen38-single-3090 · v1-fi-switch |
| `dev/` | 上手干活 | config · testing · debug-tools · profiling · fixtures |

## 六、现状快照(2026-10-22)

**验收线**(xinfer requirements v1.0 继承):prefill 2178 / 7656 / 9k+ ·
decode 133 / 157 / 165 · ctx 131k / 262k / 1M(及格 / 达标 / 冲刺)。

| 面 | 现状 | 出处 |
|---|---|---|
| 单 3090 Ti · 27B AWQ | 裸 decode **48.8**(超 vLLM 46.86);+DFlash2 数学域 **175-177**(超冲刺线 165)/ 竖式 113-118 / 长城 53-54 | `perf/qwen38-single-3090.md` |
| 三段换装(V1) | prefill / verify / decode 全部 FI+split(piecewise);verify −30% / decode −39%,文本逐位一致 | `perf/v1-fi-switch.md` |
| TP2 · 2×3080 | AR 通道提速后 prefill 1.396×;FI 单图 + spec d7 std decode **139.8**;W4A8 档 prefill **2386(+47%)** | `comm/tp2-roadmap.md` · `perf/tp2-decode-fast.md` |
| 上下文 | 132k 单请求过及格线;262k 需三级缓存 / int8 KV | `perf/optimization-candidates.md` |
| 服务面 | agent 面(pi 对接)闭环:前缀复用同会话 88% / 跨会话块级 89% | `serving/agent-surface.md` |
| 性能下一刀 | 32k 墙的非 GPU 空窗 48.5%(GPU 只忙 51.5%) | `perf/optimization-candidates.md` §7 |
| 当前第一优先 | A7 / A8 同源排查(spec 臂 token 级不稳定 + 工具遵循度 1/9) | 本地台账 · A 组 |

## 七、历史与台账在哪(本文件不装历史)

- **逐轮施工 / 战役实录**:本地台账(92 篇,不入库)。重点:
  `优化候选总表-2026-10-14.md`(已摘要收编 `perf/optimization-candidates.md`)、
  `code-review-结构检视.md`(clean-code 十轮实录)、
  `26-10-21-源码公开准备.md`(公开计划与进度)、
  `agent-服务面-剩余工作项-2026-10-14.md`(专项待办)
- **已定谳的判例 / 坑**:`perf/pitfalls-general.md`(通用)· `perf/pitfalls-cuda.md`(CUDA / 图)
  · `kernels/kernel-error-ledger.md`(算子)· `perf/optimization-candidates.md` §5(禁止重挖清单)
- **旧版 AGENTS.md 全文**:本地台账区(2026-10-22 归档)

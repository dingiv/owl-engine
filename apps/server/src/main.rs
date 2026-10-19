//! owl 引擎服务架子(2026-09-26):OpenAI 兼容 `/v1/chat/completions`
//! (stream SSE / 非 stream JSON)+ `/health` + `/v1/models`。
//!
//! 形态(charter A3 同步隔离 / A2.6 每卡一线程先声):**engine actor 独占
//! 专属 OS 线程** —— `RunningEngine` 非 Send(权重 Cell 内变异性 + 图闭包),
//! 构造/装载/泵全部在该线程内闭合,HTTP 层(多线程 tokio)只持提交通道;
//! 装配完成经 ready 信号回主线程,之后才开始监听(架子:引擎就绪前无
//! HTTP 面)。
//!
//! 模块布局(2026-10-10 clean-code 轮):
//! - [`config`]:`ServerConfig`(boot 边界唯一 env 读取点,档位分派);
//! - [`actor`]:engine actor(收件/泵不相交纪律 + 事件路由);
//! - [`openai`]:OpenAI 协议面(请求解析/应答 JSON/chat 事件流消费);
//! - [`debug`]:`/debug/metrics` 投影(纯函数,路由只转发);
//! - [`agent_trace`]:`/debug/agent` 诊断环(最近 N 轮结构现场;G5②);
//! - [`http`]:HTTP 帧解析/回写(协议无关)。
//!
//! 本文件只留编排:CLI 解析 → config 四层叠加(Default → Flavor →
//! 文件 → CLI)fail-fast → actor 线程 spawn → ready 门 → accept 循环。
//! env 已退出正式配置链(唯一遗留 = OWL_CONFIG 路径兜底);文件 I/O
//! 一律经 `owl_shared::file_loader`。
//!
//! 用法:
//! ```text
//! cargo run -p server --release          # 或 OWL_DEVICE=3 cargo run -p server
//! curl 127.0.0.1:8135/health
//! curl 127.0.0.1:8135/v1/chat/completions -H 'content-type: application/json' \
//!   -d '{"messages":[{"role":"user","content":"你好"}],"max_tokens":24,"stream":true,"session":"demo"}'
//! ```
//!
//! 架子边界(挂账):chat 模板 naive 渲染(openai.rs)、turn abort
//! (actor.rs pump Err 处置)、引擎就绪前的 503 健康面、usage 模型侧真账
//! (现 = 事件面计数)。keep-alive/chunked 已于 G5④ 落地(见 http.rs 头注)。

mod actor;
mod shutdown;
// `/debug/agent` 诊断环(最近 N 轮:请求事实 / 渲染 prompt / usage / 轨迹;G5②)
mod agent_trace;
// chat template 渲染器(HF 语义;agent 服务面 §3.2/§7.1)
mod chat_tpl;
mod config;
mod debug;
mod http;
mod openai;
// 三通道流式解析器(增量文本 → 思考/正文/工具调用;agent 服务面 §4.5)
mod toolcall;

use std::time::Duration;

use serde_json::json;
use tokio::net::TcpListener;
use tokio::sync::mpsc;

use crate::actor::EngineReq;
use crate::config::{ModelKind, ServerConfig};

/// 引擎请求通道深度(pump 期间的提交在缓冲;收件与泵不相交纪律)
const CHANNEL_CAPACITY: usize = 64;

/// accept 失败退避(防 fd 耗尽等错误风暴下忙旋)
const ACCEPT_BACKOFF: Duration = Duration::from_millis(100);

/// **keep-alive 空闲超时**(G5④):两次请求之间静默超过它 ⇒ 服务端主动关。
///
/// 设计选择:固定常量,不添配置键 —— ①它不影响引擎行为/测量口径(纯传输层);
/// ②值域没过窄到需要现场调(pi/undici 的连接池空闲回收 ≪ 30s);
/// ③真要调时的落点已定:`[server] keep_alive_secs`(添键即插,现不添)。
const KEEP_ALIVE_IDLE: Duration = Duration::from_secs(30);

/// **单连接请求数上限**(G5④):防“一个连接永远不换”在长时运行里变成 fd/任务的
/// 无限累积;达上限主动关(客户端重建连接,行为无感)。
const MAX_REQUESTS_PER_CONN: usize = 1024;

/// **引擎 actor 线程栈**(2026-10-17):缺省 2MiB 对 release 够、对 debug 不够
/// (见 `owl-engine` 线程处的长注)。32MiB 是余量档:虚拟内存代价可忽略
/// (进程只留 1 个),只买“深链不崩”。
const ENGINE_STACK_BYTES: usize = 32 << 20;

/// **诊断投影面共享单元**(`/debug/blocks` 的 GPU 命令出口)。
/// 显式依赖律:不做全局变量 —— `main` 建、actor 装、HTTP 读,全走参数。
/// 类型 = **设备句柄**(`DeviceHandle<GpuClient>`;2026-10-19 步 3:face 收编到句柄)。
/// 注意:句柄的账本单元是 `Arc<Mutex<..>>`(真 Send)⇒ 送到 HTTP 线程是安全的
/// ——这正是选 `Arc<Mutex>` 而不是 `Rc<RefCell>` 的实证理由(设计卷 §十二)。
type DbgFace = std::sync::Arc<
    std::sync::OnceLock<std::sync::Arc<tokio::sync::Mutex<owl_engine::GpuFace>>>,
>;

#[tokio::main]
async fn main() {
    // CLI:唯一参数面(配置文件路径 + flavor;手写解析,未知参数拒启)
    let args = match crate::config::CliArgs::parse(std::env::args().skip(1)) {
        Ok(a) => a,
        Err(e) => {
            // **拒启必须可见**(A1):此刻 logger 尚未初始化,`log_info!` 会被丢弃
            // ⇒ 早期失败一律裸 eprintln(与 `[boot] 引擎装配失败` 同款)
            eprintln!("[boot] 参数非法: {e}");
            std::process::exit(1);
        }
    };
    if args.help {
        eprintln!("{}", crate::config::HELP);
        return;
    }
    if args.list_flavors {
        eprintln!("可用 flavor:");
        for n in owl_shared::flavors::preset_names() {
            eprintln!("  {n}");
        }
        return;
    }
    if args.list_profiles {
        eprintln!("可用意图档(--profile):");
        for n in owl_shared::flavors::profile_names() {
            eprintln!("  {n}");
        }
        return;
    }
    // 配置:意图/预设 → 位置参数 → 站点 → CLI(非法即拒;**预检不解意图不选卡**)
    let resolved = match ServerConfig::load_resolved(&args) {
        Ok(r) => r,
        Err(e) => {
            // 拒启必须可见(A1):此刻 logger 尚未初始化,`log_info!` 会被丢弃
            eprintln!("[boot] 配置非法: {e}");
            std::process::exit(1);
        }
    };
    // `--check` / `--print-config` / `--explain`:不加载权重、不选卡、不起引擎
    if args.check || args.print_config || !args.explain.is_empty() {
        match crate::config::inspect(&args, &resolved) {
            Ok(()) => return,
            // 预检报告已把 FAIL 行打在 stdout/stderr,这里只置退出码(不重复刷屏)
            Err(e) if e.contains("预检未通过") => std::process::exit(1),
            Err(e) => {
                eprintln!("[config] FAIL: {e}");
                std::process::exit(1);
            }
        }
    }
    // boot 一行摘要(意图档;详细依据用 --check / --print-config)
    if let Some(line) = crate::config::summary_line(&resolved) {
        eprintln!("{line}");
    }
    // 完整解析 → server 形态(选卡在这里;预检已提前返回)
    let config = match ServerConfig::from_resolved(resolved, &args) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[boot] 配置非法: {e}");
            std::process::exit(1);
        }
    };

    // **FI plan 档位投递**(配置 → 核):C++ 适配器与 kernels 的 FI server 都用
    // `getenv("FI_CUDA_GRAPH")` 判档(上游 ABI 不动),而档位必须来自配置面
    // ⇒ 入口 boot 期**唯一写者**。读侧唯一 = `kernels/src/server/flashinfer.rs`
    // 的 `fi_cuda_graph()`(同键、同口径);此处零第二套判据。
    // 必须在任何图捕获/plan 之前(boot 期一次)。
    std::env::set_var(
        "FI_CUDA_GRAPH",
        if config.knobs.fi_cuda_graph { "1" } else { "0" },
    );

    // 注:**图自携 plan restore** 不再是 env(2026-10-18 用户裁决)。它现在走
    // 配置 `[dispatch] fi_plan_restore` → `DiagOpts` → 后端 boot 期
    // `set_plan_restore`(见 cuda `state.rs`/`server/mod.rs`);单图档必须配
    // `fi_plan_restore = false`,由 `OwlConfig::validate` 拒启兜底。

    // 调试门三件套先取局部:下面 `config` 会被 move 进 actor 线程,
    // 而武装点必须在 boot 完成之后(见下方注释)
    let (sg_mode, sg_ms, sg_us, sg_every) = (
        config.step_gate,
        config.step_gate_ms,
        config.step_gate_us,
        config.step_gate_every,
    );

    // logger 全局仓(框架;五级宏的编译期门 = feature `log` ——
    // 不开时宏全是空语句,本 init 仍无害且让直接 API 可用)
    {
        let lc = &config.log;
        let log_cfg = owl_shared::log::LogConfig {
            level: lc.level.to_u8(),
            tags_allow: lc.tags_allow.clone(),
            tags_deny: lc.tags_deny.clone(),
            target: match lc.file.clone() {
                Some(p) => owl_shared::log::Target::Tee(p),
                None => owl_shared::log::Target::Stderr,
            },
            max_bytes: lc.max_mb.saturating_mul(1024 * 1024),
            keep: lc.keep.max(1),
            timestamp: lc.timestamp,
        };
        match owl_shared::log::init_logger(log_cfg) {
            Ok(()) => owl_shared::log_info!("boot", "log level = {}(log_build={})",
                owl_shared::log::level_name(),
                owl_shared::log::LOG_BUILD,
            ),
            // logger 自身失败 ⇒ 只能用 stderr(A1)
            Err(e) => eprintln!("[boot] logger 初始化失败: {e}"),
        }
    }

    // metrics 全局仓(E5-M5 spec 分相探针消费面;/debug/metrics 查询;
    // 重复 init = AlreadyInit 保首个,loader 侧 init 幂等兼容)
    let _ = owl_shared::metrics::init_metrics(owl_shared::metrics::MetricsStore::new());

    // ── 礼仪启动 + 优雅关停(2026-10-16)────────────────────────────────
    // 一卡一锁(文件名 = 卡 UUID):被占 ⇒ 立即响亮拒绝(不抢占、不 OOM);
    // 退出路径双保险:① 正常返回 ⇒ LockGuard::Drop ② 信号 ⇒ release(见下)。
    let _gpu_locks = {
        let mut guards = Vec::with_capacity(config.devices.world());
        let port = config
            .bind
            .rsplit(':')
            .next()
            .and_then(|p| p.parse::<u16>().ok())
            .unwrap_or(0);
        for uuid in config.devices.uuids() {
            let info = owl_shared::gpu_lock::LockInfo::new(uuid, port, None);
            match owl_shared::gpu_lock::LockGuard::acquire(&info) {
                Ok(g) => guards.push(g),
                Err(e) => {
                    // 已取到的锁显式还(`process::exit` 不走 Drop)
                    for g in &guards {
                        owl_shared::gpu_lock::release(g.path());
                    }
                    owl_shared::log_error!("boot", "卡锁失败(拒绝启动): {e}");
                    eprintln!("[boot] 卡锁失败(拒绝启动): {e}");
                    std::process::exit(2);
                }
            }
        }
        guards
    };
    owl_shared::log_info!(
        "boot",
        "已锁卡 {} (pid {};停服 = kill -TERM <pid>,自动清锁)",
        config.devices.uuids().join(" + "),
        std::process::id()
    );
    // **收摊上下文**(SIGTERM 与"引擎不可恢复错误"共用;见 shutdown.rs)
    // TP 档可能多把卡锁;`Shutdown` 带 rank0 那把(其余由信号处理统一释放)
    let shutdown = shutdown::Shutdown {
        lock_path: _gpu_locks[0].path().to_path_buf(),
        uuid: config.devices.uuids().join(" + "),
    };
    // 信号:SIGTERM/SIGINT ⇒ 清锁 + 有序退出(`process::exit` 不跑 Drop ⇒ 显式 release)
    {
        let ctx = shutdown.clone();
        // TP 档:每 rank 一把卡锁 ⇒ 收集**全部**(停服统一释放;`graceful` 只带一把)
        let lock_paths: Vec<std::path::PathBuf> =
            _gpu_locks.iter().map(|g| g.path().to_path_buf()).collect();
        let uuids = config.devices.uuids().join(" + ");
        tokio::spawn(async move {
            let mut sigterm = match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
                Ok(s) => s,
                Err(e) => {
                    owl_shared::log_warn!("boot", "SIGTERM 监听失败(优雅关停降级): {e}");
                    return;
                }
            };
            let mut sigint = match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt()) {
                Ok(s) => s,
                Err(e) => {
                    owl_shared::log_warn!("boot", "SIGINT 监听失败(优雅关停降级): {e}");
                    return;
                }
            };
            tokio::select! { _ = sigterm.recv() => {}, _ = sigint.recv() => {} }
            owl_shared::log_info!("boot", "收到停止信号 ⇒ 优雅关停(释放卡锁 {uuids} + 显存)");
            for p in &lock_paths {
                owl_shared::gpu_lock::release(p);
            }
            ctx.graceful("收到停止信号(SIGTERM/SIGINT)");
        });
    }
    // 性能框架分级(docs/arch/perf-framework.md §一;off = 零成本基线臂)
    owl_shared::metrics::set_level(config.perf_level);
    owl_shared::log_info!("boot", "perf level = {}(perf_build={};无该 feature 的 release 里钩子空展开)",
        owl_shared::metrics::level_name(),
        owl_shared::metrics::PERF_BUILD,
    );
    if let Some(p) = &config.perf_dump {
        owl_shared::log_info!("boot", "perf dump → {}(每 turn 终了覆写 CSV + .json)", p.display());
    }

    // ── chat 模板(agent 服务面 §3.2):**快照资产**,装配期编译一次 ──
    // 渲染唯一权威 = 模型目录里的 chat_template.jinja(新建/重命名为内置模板
    // 串会直接打碎前缀复用,见 §4.3 R2);装载失败 = 拒启(渲染不出来 = 服务无意义)
    let tpl = match chat_tpl::ChatTpl::load_with(
        &config.model_dir,
        config.agent.template_path.as_deref(),
        config.agent_template_defaults(),
    ) {
        Ok(t) => {
            owl_shared::log_info!("boot", "chat 模板已装载({})", t.source());
            std::sync::Arc::new(t)
        }
        Err(e) => {
            eprintln!("[boot] chat 模板装载失败: {e}");
            std::process::exit(1);
        }
    };

    // ── engine actor:专属线程(构造+装载+泵全在内;RunningEngine 非 Send)──
    // HTTP 侧留 bind/model_name 副本(model_name 回退服务端缺省)
    // 并发上限守卫(§三十五)**已下沉**:`gdn_slots ≥ max_concurrency` 是
    // **跨字段配置不变量**,现由 `OwlConfig::validate` 在**配置装载**时拒
    // (D1 2026-10-17:原写法在本文件里读引擎内部容量 + 直接 exit(1),
    // 且只有 server 一条入口守;tests/cli 直构 engine 的路径裸奔)。
    // 此处只留信息行。
    let gdn_slots = config.knobs.gdn_slots;
    let max_concurrency = config.max_concurrency;
    owl_shared::log_info!("boot", "并发上限 {max_concurrency}(GDN 格 {gdn_slots};本地服务,排队超出请求)"
    );

    let (bind, model_name) = (config.bind.clone(), config.model_name.clone());
    // agent 服务面旋钮(模板已装载用于 log;此处交协议面消费)
    let agent = std::sync::Arc::new(config.agent.clone());
    // **诊断投影面共享单元**(见 `DbgFace` 注):actor 装、HTTP 读
    let dbg_face: DbgFace = std::sync::Arc::new(std::sync::OnceLock::new());
    let dbg_face_actor = dbg_face.clone();
    // **agent 诊断环**(G5②):固定深 `agent_trace::TURNS`,容量与 ctx 无关
    // (prompt 快照截断;内存上界见该模块头注)。显式依赖:main 建 → serve → chat
    let agent_trace = std::sync::Arc::new(agent_trace::AgentTrace::default());
    let (tx, rx) = mpsc::channel::<EngineReq>(CHANNEL_CAPACITY);
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<String, String>>();
    let shutdown_actor = shutdown.clone();
    // 兜底网的**独立句柄**(闭包会把 shutdown_actor move 走 ⇒ 网里要用另一个)
    let shutdown_net = shutdown.clone();
    std::thread::Builder::new()
        .name("owl-engine".into())
        // **显式栈**(2026-10-17):缺省 2MiB 是 debug 档的暗雷 —— 27B + DFlash2
        // 草稿的装载/装配链在 debug 下帧深远超 release(实测
        // `--flavor spec-dflash testdata/args/e2e-27b.toml` 的 debug 二进制
        // 在 DFlash2 草稿初始化处栈溢出 SIGABRT;`RUST_MIN_STACK=64M` 即复活)
        // ⇒ 本线程是**唯一**背全模型生命周期的线程,给足栈,别赌优化器。
        .stack_size(ENGINE_STACK_BYTES)
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("actor runtime");
            // ⚠️ **panic 兜底**(2026-10-17 审查 J2):本线程是全进程**唯一**背引擎
            // 状态的线程,而引擎热路径以 `expect` 表达状态不变量(`exec/draft.rs` 31 处、
            // `exec/phases.rs` 8 处…)⇒ 一旦失守,旧行为 = 线程**静默死亡**、
            // 之后**所有请求悬挂**(HTTP 面 catch 不到,只能等客户端超时;
            // 运维在日志里也看不到"为什么不再应答")。
            // 现:捕获 → 响亮日志(带 panic 载荷)→ **exit(70)**(EX_SOFTWARE,
            // 与"配置错 exit(1)"分开 ⇒ supervisor 能分辨"该不该重启")。
            // **不续跑**的理由:panic 可能落在锁 / 块账 / 设备状态中途 ⇒ 续跑等于
            // 拿已损坏的状态继续服务(静默错值,比崩溃坏)。GPU 卡锁按死 pid 自动回收
            // (`gpu_lock::live_locks_in`),不会挡下次启动。
            // **最后一道网,不是主路径**(2026-10-17 用户裁决):引擎里的**可失败逻辑
            // 一律用显式 Result 上抛** —— 泵循环收到 `Err` 就①打印 ②丢该 turn
            // ③致命者(`is_fatal`,含 `EngineInvariant`)调 `shutdown.graceful()`
            // 收摊(释放卡锁 + 有序退出)。本网只兜**设计外**的恐慌(索引越界、
            // 第三方库 unwrap 之类),且**走同一条优雅收摊**,不硬退、不冒烟退出。
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                rt.block_on(actor_main(config, rx, &ready_tx, dbg_face_actor, shutdown_actor));
            }));
            if let Err(p) = r {
                let what = panic_payload_str(&p);
                owl_shared::log_error!(
                    "boot",
                    "owl-engine 线程 panic(设计外 —— 应转显式 Result):{what}"
                );
                shutdown_net.graceful(&format!("引擎线程 panic(设计外):{what}"));
            }
        })
        .expect("actor 线程");

    // 装配结果(引擎就绪前不开 HTTP 面;架子边界)
    match ready_rx.recv() {
        Ok(Ok(capture)) => owl_shared::log_info!("boot", "引擎就绪(捕获态 {capture})"),
        Ok(Err(e)) => {
            // 拒启是运维必须看见的事件 ⇒ 恒输出(不经级别门)
            eprintln!("[boot] 引擎装配失败: {e}");
            std::process::exit(1);
        }
        // 就绪前通道断开 = actor 线程已死(panic)。旧写法 `.expect("actor 线程存活")`
        // 会在这里再 panic 一次,把真正的现场(上方 panic 日志)埋掉
        // ⇒ 走**同一条优雅收摊**(2026-10-17 用户裁决:不硬退)。
        Err(_) => {
            shutdown.graceful("actor 线程在就绪前退出(现场见上方 panic 日志)");
        }
    }

    // **单步/粘滞调试门武装**(2026-10-14):**必须放在 boot 完成之后** ——
    // boot 期的 warmup/图捕获也会跑解释器,而此刻 HTTP 面还没起来;若一开始
    // 就 Single 会自锁(见 stepgate.rs 模块头纪律 1)。非 off 档在未编入
    // `stepgate` feature 的二进制里会**可见告警**(不静默失效)。
    owl_shared::stepgate::configure_ex(sg_mode, sg_ms, sg_us, sg_every);
    if sg_mode != owl_shared::stepgate::StepMode::Off {
        eprintln!(
            "[boot] 调试门已武装:mode={} interval={}(ms={} us={}) every={}(状态面 GET /debug/step;发步 POST /debug/step/release/<n>)",
            sg_mode.name(),
            if sg_us > 0 { format!("{}us", sg_us) } else { format!("{}ms", sg_ms) },
            sg_ms,
            sg_us,
            sg_every
        );
    }

    // 并发上限守卫(§三十五 server 上层):GDN 行 = 每并发会话一格,
    // max_concurrency 超格数 = 第 N+1 会话状态格耗尽(LRU 逐出活跃会话)
    // → 跨会话状态污染。不足即拒启(账外组合不静默)。
    // 并发闸门(permit = max_concurrency;在跑 + 排队 ≤ 上限,超出排队等待)
    let concurrency_gate = std::sync::Arc::new(tokio::sync::Semaphore::new(max_concurrency));

    let listener = TcpListener::bind(&bind).await.expect("bind");
    // 就绪横幅**恒输出**(不经级别门):默认级别 = warn 时前面一行不出,
    // 运维至少需要知道“起来了 + 端口 + 如何开详细日志”。
    eprintln!(
        "[boot] 就绪 http://{bind}(model={model_name};日志级别 {},详细装配报告用 --log-level info)",
        owl_shared::log::level_name(),
    );
    loop {
        match listener.accept().await {
            Ok((stream, _peer)) => {
                let ctx = ConnCtx {
                    tx: tx.clone(),
                    model_name: model_name.clone(),
                    tpl: tpl.clone(),
                    agent: agent.clone(),
                    dbg_face: dbg_face.clone(),
                    trace: agent_trace.clone(),
                    // 并发闸门(**每请求**获取,不是每连接 —— G5④:
                    // keep-alive 下连接会长时间存活,连接级 permit =
                    // 空闲连接占死并发位)
                    gate: concurrency_gate.clone(),
                };
                tokio::spawn(async move {
                    if let Err(e) = serve(stream, ctx).await {
                        owl_shared::log_warn!("conn", "{e}");
                    }
                });
            }
            Err(e) => {
                owl_shared::log_warn!("conn", "accept 失败: {e}(退避 {ACCEPT_BACKOFF:?})");
                tokio::time::sleep(ACCEPT_BACKOFF).await;
            }
        }
    }
}

/// 单 rank 引擎壳构造(设备绑定;装载走 [`load_shard`])。
fn engine_of(
    config: &ServerConfig,
    ordinal: usize,
) -> owl_engine::Result<owl_engine::Engine<owl_cuda::GpuClient>> {
    owl_engine::Engine::new(owl_engine::EngineConfig {
        device_ordinal: ordinal,
        max_seq_tokens: config.max_seq,
        prefill_chunk: config.prefill_chunk,
        knobs: config.knobs.clone(),
    })
}

/// 单 rank 分片装载(**单卡档 `world = 1` = 恒等分片**,与旧路径字节同源)。
/// TP 档由 [`actor_main`] 逐 rank 调用(rank 序 = `devices` 序)—— 确定性分片
/// 协议:两侧零协调,各自按同一张规则表切(切分见 `crates/engine/src/loader.rs`)。
async fn load_shard(
    config: &ServerConfig,
    engine: &mut owl_engine::Engine<owl_cuda::GpuClient>,
    rank: usize,
    world: usize,
) -> owl_engine::Result<owl_engine::LoadedModel> {
    let mut loader = engine.loader().slice(rank, world).vocab_split(config.tp_vocab_split);
    match config.model_kind {
        ModelKind::Awq27b => {
            // 量化方案映射(唯一点 owl_engine::{head_plan_of, quant_plan_of})
            let head_plan = owl_engine::head_plan_of(config.head_quant);
            let quant = owl_engine::quant_plan_of(config.quant);
            loader
                .load_qwen38_27b_awq(&config.model_dir, &config.model_dir, head_plan, quant)
                .await
        }
        ModelKind::Qwen35_08b => loader.load_qwen35_0_8b(&config.model_dir).await,
    }
}

/// actor 线程主:引擎生命周期 facade(装配)→ 事件泵(actor::run_actor)。
/// 装配决策已在 config 落地 —— 本函数零 env 读取,档位分派 = 枚举查表。
async fn actor_main(
    config: ServerConfig,
    rx: mpsc::Receiver<EngineReq>,
    ready: &std::sync::mpsc::Sender<Result<String, String>>,
    dbg_face: DbgFace,
    shutdown: shutdown::Shutdown,
) {
    let built: owl_engine::Result<_> = async {
        let t0 = std::time::Instant::now();
        let world = config.devices.world();
        let mut engine = engine_of(&config, config.devices.device())?;
        owl_shared::log_info!("boot", "设备绑定 {:.2}s", t0.elapsed().as_secs_f32());
        match world {
            // ── 单卡档(原路径;分片 = 恒等)─────────────────────────────
            1 => {
                // 装载 = 模型档分发(唯一点 owl_engine::{head_plan_of, quant_plan_of})
                let model = match config.model_kind {
                    // 量化方案映射在 `load_shard` 内部(唯一映射点)
                    ModelKind::Awq27b => load_shard(&config, &mut engine, 0, 1).await?,
                    ModelKind::Qwen35_08b => {
                        engine.loader().load_qwen35_0_8b(&config.model_dir).await?
                    }
                };
                owl_shared::log_info!("boot", "权重/tokenizer/rope 装载 {:.2}s", t0.elapsed().as_secs_f32());
                let running = engine.run(model).await?;
                owl_shared::log_info!("boot", "引擎装配(run)总计 {:.2}s", t0.elapsed().as_secs_f32());
                Ok(running)
            }
            // ── TP 档(S6 2026-10-17;v1 恒 world = 2)────────────────────
            // 装配顺序有语义(见 `crates/engine/src/tp.rs` 模块头):
            // 分片装载 → `TpEngine::assemble`(分档 / staging 对称握手**在建图
            // 之前** / 每 rank `boot_rank`)→ `run`(rank0 束 = 主设备面,
            // rank1 束 = peer)。**actor/HTTP 面零改动** —— `run()` 返回同型
            // `RunningEngine<owl_engine::GpuFace>`。
            2 => {
                let ord = config.devices.ordinals();
                let model0 = load_shard(&config, &mut engine, 0, 2).await?;
                let mut engine1 = engine_of(&config, ord[1])?;
                let model1 = load_shard(&config, &mut engine1, 1, 2).await?;
                owl_shared::log_info!(
                    "boot",
                    "TP2 分片装载(rank0 ordinal {} / rank1 ordinal {}):{:.2}s",
                    ord[0],
                    ord[1],
                    t0.elapsed().as_secs_f32()
                );
                let tp = owl_engine::TpEngine::assemble_with(
                    vec![engine, engine1],
                    vec![model0, model1],
                    owl_engine::TpOpts {
                        push: config.tp_push,
                        ar_f8: config.tp_ar_f8,
                        ar_fused: config.tp_ar_fused,
                    },
                )
                .await?;
                if config.tp_ar_f8 {
                    owl_shared::log_info!(
                        "boot",
                        "TP2 **AR fp8 载荷档**已开(runtime.tp_ar_f8):载荷 e4m3(1.031×)"
                    );
                }
                if config.tp_push {
                    owl_shared::log_info!(
                        "boot",
                        "TP2 **push 档**已开(runtime.tp_push):数据腿 = 写对端槽(RW 别名)"
                    );
                }
                owl_shared::log_info!(
                    "boot",
                    "TP2 装配完成(分档 + staging 握手 + 双 rank 图捕获):{:.2}s",
                    t0.elapsed().as_secs_f32()
                );
                let running = tp.run()?;
                owl_shared::log_info!("boot", "引擎装配(run)总计 {:.2}s", t0.elapsed().as_secs_f32());
                Ok(running)
            }
            // world 的合法值由 config 层守(`resolve_tp_devices` 只放 2 项);
            // 到这里 = 内部不变量破了 ⇒ 响亮拒,不静默降级成单卡
            n => Err(owl_iface::contract::ModelError::Msg(format!(
                "TP 档只支持 world = 2(设备束给了 {n} 项;config 层守门漏了)"
            ))),
        }
    }
    .await;
    let mut running = match built {
        Ok(r) => {
            let _ = ready.send(Ok(format!("{:?}", r.capture_outcome())));
            r
        }
        Err(e) => {
            let _ = ready.send(Err(e.to_string()));
            return;
        }
    };
    // **调试投影面出口**(`GET /debug/blocks`):引擎就绪后把 face 句柄克隆
    // 放进共享单元(HTTP 侧只读到同一个 actor 的另一个发送端;不阻塞引擎)。
    if let Some(c) = running.face_clone() {
        let _ = dbg_face.set(std::sync::Arc::new(tokio::sync::Mutex::new(c)));
    } else {
        owl_shared::log_info!("boot", "face 克隆不可用 ⇒ /debug/blocks 回 503(该后端不支持)");
    }
    actor::run_actor(&mut running, rx, config.perf_dump.clone(), shutdown).await;
}

/// panic 载荷 → 可读串(`&str` / `String` 两型覆盖 std 惯例;其余给个兜底)
fn panic_payload_str(p: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = p.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = p.downcast_ref::<String>() {
        s.clone()
    } else {
        "(非字符串 panic 载荷)".to_string()
    }
}

/// HTTP 面共享句柄(2026-10-19,G5② 收拢:参数逐个下传已到临界;
/// `/debug/agent` 是接入的第四方依赖 —— 再堆参数就要靠注释对位了)。
#[derive(Clone)]
struct ConnCtx {
    tx: mpsc::Sender<EngineReq>,
    model_name: String,
    tpl: std::sync::Arc<chat_tpl::ChatTpl>,
    agent: std::sync::Arc<owl_shared::config::AgentCfg>,
    /// `/debug/blocks` 的 GPU 命令出口(engine `face_clone()`;None = 不支持)
    dbg_face: DbgFace,
    /// `/debug/agent` 诊断环(G5②;显式依赖,不做全局变量)
    trace: std::sync::Arc<agent_trace::AgentTrace>,
    /// 并发闸门(G5④:**每请求**获取;在跑 + 排队 ≤ max_concurrency)
    gate: std::sync::Arc<tokio::sync::Semaphore>,
}

/// 单连接服务(**keep-alive 请求循环**;G5④)。
///
/// 连接生命周期纪律:
/// - **permit 每请求获取**:空闲等待(读下一请求/排队)不占并发位;
/// - **空闲超时** [`KEEP_ALIVE_IDLE`] / **请求数上限** [`MAX_REQUESTS_PER_CONN`];
/// - **非 HTTP/1.1 请求响亮拒**(505 + 关)—— chunked 应答是 1.1 语法,
///   静默降级成 close 定界会让“复用判定”与“帧语法”两套律打架;
/// - 处理函数返 Err(含 SSE 中途写失败)= 帧不可信 ⇒ 关连接不重用。
async fn serve(mut stream: tokio::net::TcpStream, ctx: ConnCtx) -> std::io::Result<()> {
    let mut served = 0usize;
    loop {
        // 等下一个请求(空闲超时 = 正常收摊,不报错不静默丢日志)
        let req = match tokio::time::timeout(KEEP_ALIVE_IDLE, http::read_request(&mut stream)).await
        {
            Ok(Ok(Some(r))) => r,
            Ok(Ok(None)) => return Ok(()), // 对端关
            Ok(Err(e)) => return Err(e),
            Err(_) => {
                owl_shared::log_debug!(
                    "conn",
                    "keep-alive 空闲 {:?} 无请求 ⇒ 关连接(已服务 {served} 个)",
                    KEEP_ALIVE_IDLE
                );
                return Ok(());
            }
        };
        let close_after = !req.keep_alive;
        if !req.http_11 {
            http::respond_json(
                &mut stream,
                505,
                &openai::error_json(
                    "只接 HTTP/1.1(chunked 应答是 1.1 语法;HTTP/1.0 请先升版)",
                ),
            )
            .await?;
            return Ok(());
        }
        // 并发闸门:permit 在本请求处理期间持有(G5④:从连接级改成请求级)
        let _permit = ctx.gate.acquire().await;
        handle(&mut stream, &req, &ctx).await?;
        served += 1;
        if close_after || served >= MAX_REQUESTS_PER_CONN {
            return Ok(());
        }
    }
}

/// 单请求路由(与连接生命周期解耦:serve 只管读/写与何时关)
async fn handle(
    stream: &mut tokio::net::TcpStream,
    req: &http::Request,
    ctx: &ConnCtx,
) -> std::io::Result<()> {
    let mut stream = stream;
    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/health") => {
            http::respond_json(&mut stream, 200, &json!({"ok": true, "model": ctx.model_name})).await?;
        }
        ("GET", "/v1/models") => {
            http::respond_json(
                &mut stream,
                200,
                &json!({"object": "list", "data": [{
                    "id": ctx.model_name, "object": "model", "owned_by": "owl",
                }]}),
            )
            .await?;
        }
        ("POST", "/v1/chat/completions") => {
            openai::chat(
                &mut stream,
                &req.body,
                ctx.tx.clone(),
                &ctx.model_name,
                &ctx.tpl,
                &ctx.agent,
                &ctx.trace,
            )
            .await?;
        }
        // **tokenize 面**(G5③):纯文本/家族包装两模式;
        // 延迟说明见 `openai::tokenize` 头注(actor 单线程)
        ("POST", "/tokenize") => {
            openai::tokenize(&mut stream, &req.body, ctx.tx.clone()).await?;
        }
        // 日志面入口(/debug/log[/level/<name>])
        ("GET", p) if p == "/debug/log" || p.starts_with("/debug/log/") => {
            let rest = p.strip_prefix("/debug/log").unwrap_or("").trim_start_matches('/');
            if let Some(name) = rest.strip_prefix("level/") {
                match owl_shared::log::Level::parse_threshold(name) {
                    Some(l) => {
                        let cur = owl_shared::log::set_level(l);
                        http::respond_json(&mut stream, 200, &json!({"ok": true, "level": cur}))
                            .await?
                    }
                    None => {
                        http::respond_json(
                            &mut stream,
                            400,
                            &openai::error_json(format!(
                                "未知 log level {name:?}(off|error|warn|info|debug|trace)"
                            )),
                        )
                        .await?
                    }
                }
            } else if let Some(spec) = rest.strip_prefix("tags/") {
                // /debug/log/tags/<allow_csv>/<deny_csv>("-" = 空表;
                // 逗号分隔的 tag 前缀;allow 空 = 全放行,deny 优先)
                let (a, d) = spec.split_once('/').unwrap_or((spec, "-"));
                let parse = |s: &str| -> Vec<String> {
                    if s == "-" || s.is_empty() {
                        Vec::new()
                    } else {
                        s.split(',').map(|x| x.trim().to_string()).collect()
                    }
                };
                let (allow, deny) = (parse(a), parse(d));
                let ok = owl_shared::log::set_tags(allow.clone(), deny.clone());
                http::respond_json(
                    &mut stream,
                    200,
                    &json!({"ok": ok, "tags_allow": allow, "tags_deny": deny}),
                )
                .await?;
            } else {
                let (allow, deny) = owl_shared::log::with_logger(|g| g.tags()).unwrap_or_default();
                http::respond_json(
                    &mut stream,
                    200,
                    &json!({
                        "level": owl_shared::log::level_name(),
                        "log_build": owl_shared::log::LOG_BUILD,
                        "tags_allow": allow,
                        "tags_deny": deny,
                    }),
                )
                .await?;
            }
        }
        // **单步/粘滞调试门**(docs/arch/perf-framework.md §五;投影在
        // `owl_shared::stepgate`,路由只转发):
        //   GET  /debug/step                     状态快照
        //   *    /debug/step/mode/<off|sticky|single>   切档
        //   *    /debug/step/release/<n>         发 step 信号(n=1 走一步;
        //                                        n=0 或 inf = 不限步)
        //   *    /debug/step/ms/<n>             粘滞间隔
        //   *    /debug/step/every/<n>          每 N 点停一次
        //   POST /debug/step                     便捷:放行 1 步
        ("GET", p) if p == "/debug/step" => {
            http::respond_json(&mut stream, 200, &debug::step_json()).await?;
        }
        (m, p) if p == "/debug/step" || p.starts_with("/debug/step/") => {
            let rest = p.strip_prefix("/debug/step").unwrap_or("").trim_start_matches('/');
            let (what, arg) = rest.split_once('/').unwrap_or((rest, ""));
            let arg = arg.trim();
            let (ok, msg) = match what {
                "" => {
                    if m == "POST" {
                        (owl_shared::stepgate::release(1), "release 1".to_string())
                    } else {
                        (true, "GET /debug/step = 状态;POST = 放行 1 步".to_string())
                    }
                }
                "mode" => match owl_shared::stepgate::StepMode::parse(arg) {
                    Some(mm) => (
                        owl_shared::stepgate::set_mode(mm),
                        format!("mode={}", mm.name()),
                    ),
                    None => {
                        http::respond_json(
                            &mut stream,
                            400,
                            &openai::error_json(format!(
                                "未知调试门档位 {arg:?}(off|sticky|single)"
                            )),
                        )
                        .await?;
                        return Ok(());
                    }
                },
                "release" => {
                    let n = match arg {
                        "inf" | "all" | "continue" | "0" => 0u64,
                        s => match s.parse::<u64>() {
                            Ok(v) => v,
                            Err(_) => {
                                http::respond_json(
                                    &mut stream,
                                    400,
                                    &openai::error_json(format!(
                                        "release 需要整数或 inf,得 {s:?}"
                                    )),
                                )
                                .await?;
                                return Ok(());
                            }
                        },
                    };
                    (
                        owl_shared::stepgate::release(n),
                        format!("release {n}{}", if n == 0 { "(不限步)" } else { "" }),
                    )
                }
                "us" => match arg.parse::<u64>() {
                    Ok(v) => (owl_shared::stepgate::set_interval_us(v), format!("us={v}")),
                    Err(_) => {
                        http::respond_json(
                            &mut stream,
                            400,
                            &openai::error_json(format!("us 需要整数,得 {arg:?}")),
                        )
                        .await?;
                        return Ok(());
                    }
                },
                "ms" => match arg.parse::<u64>() {
                    Ok(v) => (owl_shared::stepgate::set_interval_ms(v), format!("ms={v}")),
                    Err(_) => {
                        http::respond_json(
                            &mut stream,
                            400,
                            &openai::error_json(format!("ms 需要整数,得 {arg:?}")),
                        )
                        .await?;
                        return Ok(());
                    }
                },
                "every" => match arg.parse::<u64>() {
                    Ok(v) => (owl_shared::stepgate::set_every(v), format!("every={v}")),
                    Err(_) => {
                        http::respond_json(
                            &mut stream,
                            400,
                            &openai::error_json(format!("every 需要整数,得 {arg:?}")),
                        )
                        .await?;
                        return Ok(());
                    }
                },
                other => {
                    http::respond_json(
                        &mut stream,
                        400,
                        &openai::error_json(format!(
                            "未知调试门路径 {other:?}(mode/release/ms/every)"
                        )),
                    )
                    .await?;
                    return Ok(());
                }
            };
            // 控制通道没消费者(引擎侧未编入 feature / 已退出)= 如实报,不谎报 ok
            let mut v = debug::step_json();
            v["applied"] = json!(ok);
            v["action"] = json!(msg);
            http::respond_json(&mut stream, 200, &v).await?;
        }
        // **store 账查询面**(第 0 刀;`/debug/vram`):三层账 + **owner 维明细**。
        // 纯读账(零副作用、零搬运)—— 泄漏判据面:同一 owner 字节逐轮爬坡 = 漏。
        ("GET", "/debug/vram") => {
            let (rtx, rrx) = tokio::sync::oneshot::channel();
            if ctx.tx.clone().send(EngineReq::VramReport { reply: rtx }).await.is_err() {
                http::respond_json(&mut stream, 503, &openai::error_json("引擎退役")).await?;
                return Ok(());
            }
            match rrx.await {
                Ok(rep) => http::respond_json(&mut stream, 200, &debug::vram_json(&rep)).await?,
                Err(_) => {
                    http::respond_json(
                        &mut stream,
                        503,
                        &openai::error_json("引擎连接中断(vram 账应答通道关闭)"),
                    )
                    .await?
                }
            }
        }
        // **块账本投影**(判"同 id 两地址 / 两活块区间相交";与发射账互补:
        // 那张账 = 每次命令的块 id+地址,本投影 = 当前账本的权威快照)
        ("GET", "/debug/blocks") => {
            use owl_cuda::DeviceClient as _;
            match ctx.dbg_face.get() {
                Some(f) => match f.lock().await.block_ledger_dump().await {
                    Ok(rows) => {
                        http::respond_json(&mut stream, 200, &debug::blocks_json(&rows)).await?
                    }
                    Err(e) => {
                        http::respond_json(
                            &mut stream,
                            500,
                            &json!({"ok": false, "error": e.to_string()}),
                        )
                        .await?
                    }
                },
                None => {
                    http::respond_json(
                        &mut stream,
                        503,
                        &json!({"ok": false, "error": "face 克隆不可用(此后端不支持)"}),
                    )
                    .await?
                }
            }
        }
        // **agent 诊断面**(G5②;进程内环形,不落盘):
        //   GET  /debug/agent             最近全仓(旧→新;含 prompt 头/尾与解析器轨迹)
        //   GET  /debug/agent/<n>         最近 n 轮(夹到仓深)
        //   POST /debug/agent/clear       清仓(返回清掉轮数)
        ("POST", "/debug/agent/clear") => {
            let n = ctx.trace.clear();
            http::respond_json(&mut stream, 200, &json!({"ok": true, "cleared": n})).await?;
        }
        ("GET", p) if p == "/debug/agent" || p.starts_with("/debug/agent/") => {
            let rest = p.strip_prefix("/debug/agent").unwrap_or("").trim_start_matches('/');
            // 空段 = 全仓(0 与 `snapshot` 的取值约定一致);非法段拒绝而不静默回缺省
            let n = if rest.is_empty() {
                0
            } else {
                match rest.parse::<usize>() {
                    Ok(v) => v,
                    Err(_) => {
                        http::respond_json(
                            &mut stream,
                            400,
                            &openai::error_json(format!(
                                "/debug/agent 路径段只接轮数(或 clear),得到 {rest:?}"
                            )),
                        )
                        .await?;
                        return Ok(());
                    }
                }
            };
            http::respond_json(&mut stream, 200, &ctx.trace.snapshot(n)).await?;
        }
        // **发射账查询面**(cuda::trace;`[cuda] trace_ring = <N>` 开启):
        //   GET /debug/launches[/<n>[/<kind>]]  —— 最近 n 条(缺省 200)按发射序;
        //   kind = 命令类(launch/htod/dtoh/free…)/流名(compute/h2d/d2h)/核名子串
        ("GET", p) if p == "/debug/launches" || p.starts_with("/debug/launches/") => {
            let rest = p.strip_prefix("/debug/launches").unwrap_or("").trim_start_matches('/');
            let (ns, kind) = rest.split_once('/').unwrap_or((rest, ""));
            let n = if ns.is_empty() {
                200
            } else {
                ns.parse::<usize>().unwrap_or(200).min(100_000)
            };
            let kind = (!kind.is_empty()).then_some(kind);
            http::respond_json(&mut stream, 200, &debug::launches_json(n, kind)).await?;
        }
        // 诊断分账出口(投影在 debug 模块;路由只转发)
        ("GET", p) if p == "/debug/metrics" || p.starts_with("/debug/metrics/") => {
            let prefix = p.strip_prefix("/debug/metrics/").unwrap_or("");
            http::respond_json(&mut stream, 200, &debug::metrics_json(prefix)).await?;
        }
        // 性能框架出口(perf-framework.md §三):树状人读 / JSON / 分级切换
        // 路径语:[/debug/perf[/epoch][/json|/<prefix>]] + /debug/perf/level/<name>
        // (http 层不保留 query,故用路径段而非 ?epoch=1)
        ("GET", p) if p == "/debug/perf" || p.starts_with("/debug/perf/") => {
            let rest = p.strip_prefix("/debug/perf").unwrap_or("").trim_start_matches('/');
            if let Some(name) = rest.strip_prefix("level/") {
                match debug::perf_set_level(name) {
                    Ok(l) => {
                        http::respond_json(&mut stream, 200, &json!({"ok": true, "level": l}))
                            .await?
                    }
                    Err(e) => {
                        http::respond_json(&mut stream, 400, &openai::error_json(e)).await?
                    }
                }
            } else {
                let (epoch, rest) = match rest.strip_prefix("epoch") {
                    Some(r) => (true, r.trim_start_matches('/')),
                    None => (false, rest),
                };
                if rest == "json" {
                    http::respond_json(&mut stream, 200, &debug::perf_json("", epoch)).await?;
                } else {
                    http::respond_text(&mut stream, 200, &debug::perf_text(rest, epoch)).await?;
                }
            }
        }
        _ => {
            http::respond_json(
                &mut stream,
                404,
                &openai::error_json(format!("未知路径 {} {}", req.method, req.path)),
            )
            .await?;
        }
    }
    // 注:**不调 `stream.shutdown()`** —— 那是“一请求一连接”时代的收尾(发 FIN),
    // keep-alive 下会把连接直接杀掉(G5④ 实测踩过:客户端复用时报 dead)
    Ok(())
}

//! owl 引擎服务架子(2026-09-26):OpenAI 兼容 `/v1/chat/completions`
//! (stream SSE / 非 stream JSON)+ `/health` + `/v1/models`。
//!
//! 形态(2026-11-15 业务下沉 P1):**引擎自管线程**(与 `owl-cuda` 的
//! `GpuServer` 同形)—— `owl_engine::EngineService::boot(spec)` 在引擎自己的
//! OS 线程里完成 选卡/卡锁/装配/装载/run/泵;本进程只持 Send 的
//! `EngineClient`(chat/cancel/tokenize/diag/queue_facts)。`RunningEngine`
//! 非 Send 的事实被完全封装在引擎侧,server 不再有 executor 线程/ready 门。
//!
//! 模块布局:
//! - [`config`]:`ServerConfig`(boot 边界配置组装;选卡已下沉 engine);
//! - [`openai`]:OpenAI 协议面(请求解析/应答 JSON/chat 事件流消费);
//! - [`debug`]:`/debug/*` 投影(纯函数,数据来自 `EngineClient::diag`);
//! - [`agent_trace`]:`/debug/agent` 诊断环(最近 N 轮结构现场;G5②);
//! - [`http`]:HTTP 帧解析/回写(协议无关)。
//!
//! 本文件只留编排:CLI 解析 → config 四层叠加(Default → Flavor →
//! 文件 → CLI)fail-fast → `EngineService::boot` → accept 循环 → shutdown。
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
//! 架子边界(挂账):chat 模板 naive 渲染(openai.rs)、引擎就绪前的 503
//! 健康面、usage 模型侧真账(现 = 事件面计数)。keep-alive/chunked 已于 G5④
//! 落地(见 http.rs 头注);泵循环/取消/压力收缩/遥测已下沉 `owl_engine::service`。

// `/debug/agent` 诊断环(最近 N 轮:请求事实 / 渲染 prompt / usage / 轨迹;G5②)
mod agent_trace;
// chat template 渲染器(HF 语义;agent 服务面 §3.2/§7.1)
mod chat_tpl;
mod config;
mod debug;
mod http;
// `/debug/load` 内源灌流(服务面并发现场测量)
mod load;
mod openai;
// 跳轮重复探测(只可见不改生成;2026-10-31 立案)
mod repeat_guard;
// 视觉输入(主分支 V1–V5)
mod vision_processor;
// 三通道流式解析器(增量文本 → 思考/正文/工具调用;agent 服务面 §4.5)
mod toolcall;

use std::time::Duration;

use serde_json::json;
use tokio::net::TcpListener;

use crate::config::ServerConfig;
use owl_engine::{EngineClient, EngineService, ServiceExit};

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

/// **HTTP 面 worker 上限**(2026-11-03 用户裁决:最多 16 条)。
///
/// tokio 的缺省是 `available_parallelism()`(本机 48 核 ⇒ 48 条 worker 线程),
/// 而 HTTP 面是**纯 I/O 等待**(写 socket、等引擎事件),真算力全在
/// `engine-service` 与设备 server 线程里 ⇒ 48 条 worker 只是多占栈和调度器。
/// 取 `min(nproc, 16)`:`<=16` 核的机器按核数走,大机器封顶 16。
///
/// 为什么是编译期常量而不是 profile 旋钮:worker 数必须在**读到 profile 之前**
/// 定下来(建 runtime 在最前),而 profile 的解析/完整度门是 async 面的一部分;
/// 要变成可配就得把 args/config 解析整体前移到 `fn main()`(boot 次序重构),
/// 目前不值得 —— 这里显式、可查、有上界。
const MAX_HTTP_WORKERS: usize = 16;

fn main() -> std::process::ExitCode {
    let nproc = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let workers = nproc.min(MAX_HTTP_WORKERS);
    let rt = match tokio::runtime::Builder::new_multi_thread()
        .worker_threads(workers)
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("[boot] tokio runtime 构建失败: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    rt.block_on(async_main(workers))
}

async fn async_main(workers: usize) -> std::process::ExitCode {
    // 裸 eprintln:此刻 logger 还没初始化(`log_info!` 会被丢弃;与其它 boot 早期行同款)
    eprintln!(
        "[boot] tokio HTTP runtime:worker={workers}(nproc={}, 上限 {MAX_HTTP_WORKERS})",
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0)
    );
    // CLI:唯一参数面(-p <profile_path> + 显式覆盖;手写解析,未知参数拒启)
    let args = match crate::config::CliArgs::parse(std::env::args().skip(1)) {
        Ok(a) => a,
        Err(e) => {
            // **拒启必须可见**(A1):此刻 logger 尚未初始化,`log_info!` 会被丢弃
            // ⇒ 早期失败一律裸 eprintln(与 `[boot] 引擎装配失败` 同款)
            eprintln!("[boot] 参数非法: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    if args.help {
        eprintln!("{}", crate::config::HELP);
        return std::process::ExitCode::SUCCESS;
    }
    // 配置:意图/预设 → 位置参数 → 站点 → CLI(非法即拒;**预检不解意图不选卡**)
    let resolved = match ServerConfig::load_resolved(&args) {
        Ok(r) => r,
        Err(e) => {
            // 拒启必须可见(A1):此刻 logger 尚未初始化,`log_info!` 会被丢弃
            eprintln!("[boot] 配置非法: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    // `--check` / `--print-config` / `--explain`:不加载权重、不选卡、不起引擎
    if args.check || args.print_config || !args.explain.is_empty() {
        match crate::config::inspect(&args, &resolved) {
            Ok(()) => return std::process::ExitCode::SUCCESS,
            // 预检报告已把 FAIL 行打在 stdout/stderr,这里只置退出码(不重复刷屏)
            Err(e) if e.contains("预检未通过") => return std::process::ExitCode::FAILURE,
            Err(e) => {
                eprintln!("[config] FAIL: {e}");
                return std::process::ExitCode::FAILURE;
            }
        }
    }
    // boot 一行摘要(意图档;详细依据用 --check / --print-config)
    if let Some(line) = crate::config::summary_line(&resolved) {
        eprintln!("{line}");
    }
    // 构建期实验档必须让运维看得见(profile 本身不携带 feature)。
    #[cfg(feature = "fi-fp16pv")]
    eprintln!("[boot] kernel: FlashInfer FP16-PV = on (sm_86 GeForce build; f16/fp8 KV prefill)");
    #[cfg(feature = "fi-fp16acc")]
    eprintln!("[boot] kernel: FlashInfer FP16-QK-acc = on (QK half accumulate + half2 softmax)");
    // 完整解析 → server 形态(选卡在这里;预检已提前返回)
    let config = match ServerConfig::from_resolved(resolved, &args) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[boot] 配置非法: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };

    // **FI 族装配参数不在入口写**(2026-11-01 分层修正):`FI_CUDA_GRAPH` env
    // 与 Rust 侧原子由**同一处**一次落 —— `owl_kernels::family::flashinfer::
    // configure`(写者唯一 = 引擎装配期 `owl_engine::kernel_cfg::apply`)。
    // 入口不再设 env:两份写者 = 迟早漂移(此前的两处同值写入已合并)。
    //
    // 注:`[dispatch] fi_plan_restore` / `fi_cuda_graph` / `[kernel] prefill_shadow`
    // 都是配置键 → `EngineKnobs` → 族 `configure`;单图档必须配
    // `fi_plan_restore = false`,由 `OwlConfig::validate` 拒启兜底。

    // 调试门三件套先取局部(交给 `BootSpec`;武装点在引擎 run 之后,
    // 由 `owl_engine::service` 在引擎线程内完成)。
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
            Ok(()) => owl_shared::log_info!(
                "boot",
                "log level = {}(log_build={})",
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

    // ── 停服信号（单源）────────────────────────────────────────────
    // SIGTERM/SIGINT ⇒ `watch(true)`。两处消费者：
    // ① accept 循环（不再接新连接）；② 引擎泵循环的**步边界**（不再推进任何一拍，
    // 见 `owl_engine::service`）。
    let (stop_tx, mut stop_rx) = tokio::sync::watch::channel(false);
    tokio::spawn(async move {
        let mut sigterm =
            match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
                Ok(s) => s,
                Err(e) => {
                    owl_shared::log_warn!("boot", "SIGTERM 监听失败(优雅关停降级): {e}");
                    return;
                }
            };
        let mut sigint =
            match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt()) {
                Ok(s) => s,
                Err(e) => {
                    owl_shared::log_warn!("boot", "SIGINT 监听失败(优雅关停降级): {e}");
                    return;
                }
            };
        tokio::select! { _ = sigterm.recv() => {}, _ = sigint.recv() => {} }
        owl_shared::log_info!(
            "boot",
            "收到停止信号 ⇒ 优雅关停:停接新连接 → 引擎步边界收摊 → 释放卡锁"
        );
        let _ = stop_tx.send(true);
    });

    // 性能框架分级(docs/arch/perf-framework.md §一;off = 零成本基线臂)
    owl_shared::metrics::set_level(config.perf_level);
    owl_shared::log_info!(
        "boot",
        "perf level = {}(perf_build={};无该 feature 的 release 里钩子空展开)",
        owl_shared::metrics::level_name(),
        owl_shared::metrics::PERF_BUILD,
    );
    if let Some(p) = &config.perf_dump {
        owl_shared::log_info!(
            "boot",
            "perf dump → {}(每 turn 终了覆写 CSV + .json)",
            p.display()
        );
    }

    // ── chat 模板(agent 服务面 §3.2):**快照资产**,装配期编译一次 ──
    // 渲染唯一权威 = 模型目录里的 chat_template.jinja(新建/重命名为内置模板
    // 串会直接打碎前缀复用,见 §4.3 R2);装载失败 = 拒启(渲染不出来 = 服务无意义)
    //
    // **2026-10-19 GGUF 轮**:`model.dir` 指向 **`.gguf` 单文件**时,模板改从
    // `.gguf` 的 `tokenizer.chat_template` KV 读(设计 REQ-GGUF-02「单文件自给」)
    // —— 与快照路径**同一个下游**(`ChatTpl::from_str`,环境/兼容层单源)。
    // 显式 `agent.template_path` 覆盖仍然优先(运维 A/B)。
    let tpl = match chat_tpl::ChatTpl::load_snapshot_or_gguf(
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
            return std::process::ExitCode::FAILURE;
        }
    };

    // ── 引擎：init + 泵 + 句柄全部在 engine 内（server 只见 Send 客户端）──
    //
    // 卡锁 / 选卡 / 单卡与 TP 双 rank 并行装载 / 图捕获 / 泵循环 / 卡锁释放
    // 全部由 `owl_engine::EngineService::boot` 在**引擎自己的 OS 线程**内完成
    //（与 `owl-cuda` 的 `GpuServer` 同形）。server 这一步只搬配置字段。
    let gdn_slots = config.knobs.gdn_slots;
    let max_concurrency = config.max_concurrency;
    let max_batch = config.max_batch;
    owl_shared::log_info!(
        "boot",
        "并发上限 {max_concurrency}(GDN 格 {gdn_slots};本地服务,排队超出请求)"
    );
    let (bind, model_name) = (config.bind.clone(), config.model_name.clone());
    // 请求原文落盘目录（诊断开关；空 = 关）。写入失败 = 启动失败，
    // 不允许“日志开了但其实没落盘”的静默失约。
    let request_dump_dir = config.request_dump_dir.clone();
    let vision_max_pixels = config.vision_max_pixels;
    if let Some(dir) = &request_dump_dir {
        if let Err(e) = std::fs::create_dir_all(dir) {
            eprintln!("[boot] request_dump_dir 创建失败 {}: {e}", dir.display());
            return std::process::ExitCode::FAILURE;
        }
        owl_shared::log_info!("boot", "请求原文落盘:{}", dir.display());
    }
    // agent 服务面旋钮（协议面消费）
    let agent = std::sync::Arc::new(config.agent.clone());
    // **agent 诊断环**（G5②）：固定深 `agent_trace::TURNS`，容量与 ctx 无关
    let agent_trace = std::sync::Arc::new(agent_trace::AgentTrace::default());
    // 引擎 boot（选卡/卡锁/装配/装载/就绪 + 泵线程都在里面）
    let lock_port = bind
        .rsplit(':')
        .next()
        .and_then(|p| p.parse::<u16>().ok())
        .unwrap_or(0);
    let boot = owl_engine::BootSpec {
        devices: owl_engine::DeviceOpts {
            parallel: config.device_opts.parallel.clone(),
            gpu_min_free_ratio: config.device_opts.gpu_min_free_ratio,
            lock_port,
            lock_role: config.device_opts.lock_role.clone(),
        },
        engine: owl_engine::EngineConfig {
            device_ordinal: 0,
            max_seq_tokens: config.max_seq,
            prefill_chunk: config.prefill_chunk,
            knobs: config.knobs.clone(),
        },
        model: config.model_dir.clone(),
        format: config.model_format,
        quant: config.quant,
        head_quant: config.head_quant,
        embed_quant: config.embed_quant,
        tp_ar: config.tp_ar,
        tp_comm: config.tp_comm,
        tp_ar_f8: config.tp_ar_f8,
        tp_vocab_split: config.tp_vocab_split,
        step_gate: owl_engine::StepGate {
            mode: sg_mode,
            ms: sg_ms,
            us: sg_us,
            every: sg_every,
        },
        perf_dump: config.perf_dump.clone(),
        channel_capacity: CHANNEL_CAPACITY,
    };
    let service = match EngineService::boot(boot) {
        Ok(s) => s,
        Err(e) => {
            // 拒启是运维必须看见的事件 ⇒ 恒输出(不经级别门)
            eprintln!("[boot] 引擎装配失败: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    let client = service.client();

    // 并发上限守卫(§三十五 server 上层):GDN 行 = 每并发会话一格,
    // max_concurrency 超格数 = 第 N+1 会话状态格耗尽(LRU 逐出活跃会话)
    // → 跨会话状态污染。不足即拒启(账外组合不静默)。
    // 并发闸门(permit = max_concurrency;在跑 + 排队 ≤ 上限,超出排队等待)
    let concurrency_gate = std::sync::Arc::new(tokio::sync::Semaphore::new(max_concurrency));
    // **会话串行已下沉引擎**（2026-11-16 P4）：同会话后继 turn 在引擎队列里
    // 等前一个终了（`startable_queue_idx`），服务面不再需要 per-session 闸。
    // 口径告警(不拒启:语义仍然正确,只是这类客户端拿不到合批收益)
    if max_batch > 1 && !agent.implicit_session {
        if max_batch < max_concurrency {
            owl_shared::log_info!(
                "boot",
                "并发上限 {max_concurrency} > 合批上限 {max_batch} ⇒ 同时最多 {max_batch} 个 turn \
                 在批内,其余在**引擎队列**排队(TTFT 拉长,吞吐不塌;用 /debug/load 可现场量)"
            );
        } else if max_batch > max_concurrency {
            owl_shared::log_warn!(
                "boot",
                "合批上限 {max_batch} > 并发上限 {max_concurrency} ⇒ 批位闲置(服务面进不来那么多) \
                 —— 想让批吃满,`server.max_concurrency` 应 ≥ `runtime.max_batch`"
            );
        }
    }
    if max_batch > 1 && agent.implicit_session {
        owl_shared::log_warn!(
            "boot",
            "隐式会话开着(implicit_session=true)且 max_batch={} —— 无 session 键的请求\
             全落会话 0 ⇒ 会被引擎按会话串行(合批无收益)。要合批:客户端带 session 键,\
             或关 implicit_session",
            max_batch
        );
    }

    let listener = TcpListener::bind(&bind).await.expect("bind");
    // 就绪横幅**恒输出**(不经级别门):默认级别 = warn 时前面一行不出,
    // 运维至少需要知道“起来了 + 端口 + 如何开详细日志”。
    eprintln!(
        "[boot] 就绪 http://{bind}(model={model_name};日志级别 {},详细装配报告用 --log-level info)",
        owl_shared::log::level_name(),
    );
    let mut stop = false;
    while !stop {
        let accepted = tokio::select! {
            biased;
            // 停服优先:不再接新连接(在飞请求由引擎在**步边界**收摊)
            _ = stop_rx.changed() => {
                stop = true;
                continue;
            }
            r = listener.accept() => r,
        };
        match accepted {
            Ok((stream, _peer)) => {
                let ctx = ConnCtx {
                    client: client.clone(),
                    model_name: model_name.clone(),
                    request_dump_dir: request_dump_dir.clone(),
                    vision_max_pixels,
                    tpl: tpl.clone(),
                    agent: agent.clone(),
                    trace: agent_trace.clone(),
                    // 并发闸门(**每请求**获取,不是每连接 —— G5④:
                    // keep-alive 下连接会长时间存活,连接级 permit =
                    // 空闲连接占死并发位)
                    gate: concurrency_gate.clone(),
                    max_concurrency,
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

    // ── **收摊(单条路径;2026-11-03:不再 `process::exit`)** ────────────
    //
    // 次序即语义:
    // ① 停接新连接(上面已 break;listener 在这里 drop);
    // ② `service.shutdown()`:置停服信号 ⇒ 引擎在**步边界**收摊、TP 收尾、
    //    释放卡锁、join 引擎线程(线程与锁全在 engine 内,server 不再手工参与);
    // ③ 返回 `ExitCode`(正常 0 / 致命 1 / 设计外 panic 70)。
    drop(listener);
    let reason: ServiceExit = service.shutdown();
    if !matches!(reason, ServiceExit::Normal) {
        owl_shared::log_warn!("boot", "收摊:{}", reason.describe());
    } else {
        owl_shared::log_info!("boot", "收摊:{}", reason.describe());
    }
    std::process::ExitCode::from(reason.code())
}

/// HTTP 面共享句柄(2026-10-19,G5② 收拢:参数逐个下传已到临界;
/// `/debug/agent` 是接入的第四方依赖 —— 再堆参数就要靠注释对位了)。
#[derive(Clone)]
struct ConnCtx {
    /// 引擎客户端（Send + Clone；init/泵/线程全在 engine 内）
    client: EngineClient,
    model_name: String,
    /// 原始请求 JSON 落盘目录（None = 关）
    request_dump_dir: Option<std::path::PathBuf>,
    /// 单图像素上限(视觉预处理封顶;0 = 内置 4096² = 不缩)
    vision_max_pixels: usize,
    tpl: std::sync::Arc<chat_tpl::ChatTpl>,
    agent: std::sync::Arc<crate::config::AgentCfg>,
    /// `/debug/agent` 诊断环(G5②;显式依赖,不做全局变量)
    trace: std::sync::Arc<agent_trace::AgentTrace>,
    /// 并发闸门(G5④:**每请求**获取;在跑 + 排队 ≤ max_concurrency)
    gate: std::sync::Arc<tokio::sync::Semaphore>,
    /// 服务面并发上限(`server.max_concurrency`;`/debug/load` 按它校验)
    max_concurrency: usize,
}

/// 单连接服务(**keep-alive 请求循环**;G5④)。
///
/// 连接生命周期纪律:
/// - **permit 每请求获取**(按 [`permit_scope`] 定档):空闲等待不占并发位;
/// - **空闲超时** [`KEEP_ALIVE_IDLE`] / **请求数上限** [`MAX_REQUESTS_PER_CONN`];
/// - **非 HTTP/1.1 请求响亮拒**(505 + 关)—— chunked 应答是 1.1 语法,
///   静默降级成 close 定界会让“复用判定”与“帧语法”两套律打架;
/// - **帧非法**(定界/长度/trailer 残留)先回 400 再关 —— 只 reset 不回话会把
///   真因埋进客户端超时里;
/// - 处理函数返 Err(含 SSE 中途写失败)= 帧不可信 ⇒ 关连接不重用。
async fn serve(mut stream: tokio::net::TcpStream, ctx: ConnCtx) -> std::io::Result<()> {
    let mut served = 0usize;
    loop {
        // 等下一个请求(空闲超时 = 正常收摊,不报错不静默丢日志)
        let req = match tokio::time::timeout(KEEP_ALIVE_IDLE, http::read_request(&mut stream)).await
        {
            Ok(Ok(Some(r))) => r,
            Ok(Ok(None)) => return Ok(()), // 对端关
            Ok(Err(e)) => {
                // 客户端帧错误 ≠ 连接噪声:回 400 让对端看见原因,再关(不重用)
                if e.kind() == std::io::ErrorKind::InvalidData
                    || e.kind() == std::io::ErrorKind::UnexpectedEof
                {
                    let _ = http::respond_json(
                        &mut stream,
                        400,
                        &openai::error_json(format!("请求帧非法: {e}")),
                    )
                    .await;
                }
                return Err(e);
            }
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
                &openai::error_json("只接 HTTP/1.1(chunked 应答是 1.1 语法;HTTP/1.0 请先升版)"),
            )
            .await?;
            return Ok(());
        }
        // 并发闸门:按 **permit 策略表**取(2026-11-16 收拢;
        // 旧版 = `chat_route || cheap` 两条件 —— 诊断面被顺手挡在门外)
        let _permit = match permit_scope(req.method.as_str(), req.path.as_str()) {
            PermitScope::Engine => Some(ctx.gate.acquire().await),
            // Turn 的位在 `openai::chat` 里取(会话解析之后);Free 不占位
            PermitScope::Free | PermitScope::Turn => None,
        };
        handle(&mut stream, &req, &ctx).await?;
        served += 1;
        if close_after || served >= MAX_REQUESTS_PER_CONN {
            return Ok(());
        }
    }
}

/// **并发位策略**（permit = 一个 **turn** 位，`server.max_concurrency`）。
///
/// 2026-11-16 定策。旧版是 `serve` 里的 `chat_route || cheap` 两条条件，白名单
/// 只收了当时想到的三条路径 ⇒ `/debug/metrics`、`/debug/launches`、`/debug/perf`、
/// `/debug/log` 这些**纯进程内读**也得排队等一个 turn 让出 permit，满并发时
/// 排障入口被并发位自己锁死（`/debug/vram`、`/debug/blocks` 还要绕一圈引擎）。
///
/// 三档语义：
/// - [`Free`](PermitScope::Free)：不占 turn 位。纯进程内读/控制面
///   （`/health`、`/v1/models`、`/debug/*`）与**自限流**的 `/debug/load`
///   （它自己校验 `concurrency <= max_concurrency` 且绕过 permit 直进引擎通道，
///   再占一位只会让读数少一档）；未知路径 404 也走这档。
/// - [`Engine`](PermitScope::Engine)：进 handler 前占位 —— 消耗引擎时间但不产生
///   turn 的客户端面（`/tokenize`）。
/// - [`Turn`](PermitScope::Turn)：**不在此处取**，由 `openai::chat` 在解析会话后、
///   提交之前取 —— 同会话排队不占位，否则会把别家会话饿住。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PermitScope {
    Free,
    Engine,
    Turn,
}

/// 路由 → 并发位档位（新增引擎面路由时必须在这里表态）。
fn permit_scope(method: &str, path: &str) -> PermitScope {
    match (method, path) {
        ("POST", "/v1/chat/completions") => PermitScope::Turn,
        ("POST", "/tokenize") => PermitScope::Engine,
        _ => PermitScope::Free,
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
            http::respond_json(
                &mut stream,
                200,
                &json!({"ok": true, "model": ctx.model_name}),
            )
            .await?;
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
                stream,
                &req.body,
                ctx.client.clone(),
                &ctx.model_name,
                &ctx.tpl,
                &ctx.agent,
                &ctx.trace,
                &ctx.gate,
                ctx.request_dump_dir.as_deref(),
                ctx.vision_max_pixels,
            )
            .await?;
        }
        // **队列/容量快照**(忙时也能读;零等待 —— 每拍事实)
        ("GET", "/debug/queue") => {
            let snap = ctx.client.queue_facts();
            http::respond_json(&mut stream, 200, &debug::queue_json(&snap)).await?;
        }
        // **内源灌流面**(服务面并发能力的现场测量;见 `load` 模块头)
        ("POST", "/debug/load") => {
            load::load(
                &mut stream,
                &req.body,
                ctx.client.clone(),
                &ctx.tpl,
                &ctx.agent,
                ctx.max_concurrency,
            )
            .await?;
        }
        // **tokenize 面**(G5③):纯文本/家族包装两模式;
        // 延迟说明见 `openai::tokenize` 头注(引擎单线程)
        ("POST", "/tokenize") => {
            openai::tokenize(&mut stream, &req.body, ctx.client.clone()).await?;
        }
        // 日志面入口(/debug/log[/level/<name>])
        ("GET", p) if p == "/debug/log" || p.starts_with("/debug/log/") => {
            let rest = p
                .strip_prefix("/debug/log")
                .unwrap_or("")
                .trim_start_matches('/');
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
                http::respond_json(&mut stream, 200, &debug::log_state_json()).await?;
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
        ("GET", "/debug/step") => {
            http::respond_json(&mut stream, 200, &debug::step_json()).await?;
        }
        (m, p) if p == "/debug/step" || p.starts_with("/debug/step/") => {
            let rest = p
                .strip_prefix("/debug/step")
                .unwrap_or("")
                .trim_start_matches('/');
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
                                    &openai::error_json(format!("release 需要整数或 inf,得 {s:?}")),
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
            http::respond_json(&mut stream, 200, &debug::step_action_json(ok, &msg)).await?;
        }
        // **store 账查询面**(第 0 刀;`/debug/vram`):三层账 + **owner 维明细**。
        // 纯读账(零副作用、零搬运)—— 泄漏判据面:同一 owner 字节逐轮爬坡 = 漏。
        ("GET", "/debug/vram") => {
            match ctx.client.diag().await {
                Ok(rep) => http::respond_json(&mut stream, 200, &debug::vram_json(&rep)).await?,
                Err(e) => {
                    http::respond_json(
                        &mut stream,
                        503,
                        &openai::error_json(format!("引擎连接中断({e})")),
                    )
                    .await?
                }
            }
        }
        // **块账本投影**(判"同 id 两地址 / 两活块区间相交";与发射账互补:
        // 那张账 = 每次命令的块 id+地址,本投影 = 当前账本的权威快照)
        ("GET", "/debug/blocks") => match ctx.client.diag().await {
            Ok(snap) => http::respond_json(&mut stream, 200, &debug::blocks_ranks_json(&snap.blocks)).await?,
            Err(e) => {
                http::respond_json(
                    &mut stream,
                    503,
                    &json!({"ok": false, "error": format!("引擎连接中断({e})")}),
                )
                .await?
            }
        },
        // **agent 诊断面**(G5②;进程内环形,不落盘):
        //   GET  /debug/agent             最近全仓(旧→新;含 prompt 头/尾与解析器轨迹)
        //   GET  /debug/agent/<n>         最近 n 轮(夹到仓深)
        //   POST /debug/agent/clear       清仓(返回清掉轮数)
        ("POST", "/debug/agent/clear") => {
            let n = ctx.trace.clear();
            http::respond_json(&mut stream, 200, &json!({"ok": true, "cleared": n})).await?;
        }
        ("GET", p) if p == "/debug/agent" || p.starts_with("/debug/agent/") => {
            let rest = p
                .strip_prefix("/debug/agent")
                .unwrap_or("")
                .trim_start_matches('/');
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
            let rest = p
                .strip_prefix("/debug/launches")
                .unwrap_or("")
                .trim_start_matches('/');
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
            let rest = p
                .strip_prefix("/debug/perf")
                .unwrap_or("")
                .trim_start_matches('/');
            if let Some(name) = rest.strip_prefix("level/") {
                match debug::perf_set_level(name) {
                    Ok(l) => {
                        http::respond_json(&mut stream, 200, &json!({"ok": true, "level": l}))
                            .await?
                    }
                    Err(e) => http::respond_json(&mut stream, 400, &openai::error_json(e)).await?,
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

#[cfg(test)]
mod permit_tests {
    use super::{permit_scope, PermitScope};

    /// **permit 策略表**（2026-11-16）：turn 位只跟"会产生 turn 的请求"挂钩。
    /// 诊断/控制面必须随时可进 —— 旧版白名单只放三条路径，满并发时排障入口
    /// 被并发位自己锁死（`/debug/metrics` 要等一个 turn 跑完才返回）。
    #[test]
    fn turn_slots_only_for_turns() {
        assert_eq!(
            permit_scope("POST", "/v1/chat/completions"),
            PermitScope::Turn
        );
        assert_eq!(permit_scope("POST", "/tokenize"), PermitScope::Engine);
        for (m, p) in [
            ("GET", "/health"),
            ("GET", "/v1/models"),
            ("GET", "/debug/queue"),
            ("GET", "/debug/metrics"),
            ("GET", "/debug/launches/50/launch"),
            ("GET", "/debug/vram"),
            ("GET", "/debug/blocks"),
            ("GET", "/debug/perf/json"),
            ("GET", "/debug/agent/8"),
            ("GET", "/debug/log"),
            ("POST", "/debug/log/level/info"),
            ("GET", "/debug/step"),
            ("POST", "/debug/step"),
            ("POST", "/debug/load"),
            ("POST", "/nope"),
        ] {
            assert_eq!(permit_scope(m, p), PermitScope::Free, "{m} {p}");
        }
    }
}

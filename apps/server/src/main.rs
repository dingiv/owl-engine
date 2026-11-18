//! owl 引擎服务架子(2026-09-26):OpenAI 兼容 `/v1/chat/completions`
//! (stream SSE / 非 stream JSON)+ `/health` + `/v1/models`。
//!
//! 形态(2026-11-15 业务下沉 P1):**引擎自管线程**(与 `owl-cuda` 的
//! `GpuServer` 同形)—— `owl_engine::EngineService::boot(spec)` 在引擎自己的
//! OS 线程里完成 选卡/卡锁/装配/装载/run/泵;本进程只持 Send 的
//! `EngineClient`(chat/cancel/tokenize/diag/queue_facts)。`RunningEngine`
//! 非 Send 的事实被完全封装在引擎侧,server 不再有 executor 线程/ready 门。
//!
//! 模块布局（2026-11-17 分层收拢）:
//! - `view/`   ：HTTP 帧、连接循环、路由（`router`）、响应写回；
//! - `service/`：请求级业务编排（`chat` / `tokenize` / `load` / `debug` 投影）；
//! - `repo/`   ：后端端口与进程内状态（`engine` 门面 / `trace` 诊断环 / `vram` 账）；
//! - `domain/` ：OpenAI DTO、chat 模板、三通道流解析、视觉预处理；
//! - `infra/`  ：boot 配置（`ServerConfig`）与进程级设施。
//!
//! 本文件只留**组合根**:CLI → config 四层叠加 → `EngineService::boot` →
//! 构造 `view::router::ConnCtx` → accept 循环 → shutdown。
//! env 已退出正式配置链（唯一遗留 = OWL_CONFIG 路径兜底）；文件 I/O
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

mod domain;
mod infra;
mod repo;
mod service;
mod view;

// 根级别名（分层后保留旧 `crate::<模块>` 调用面，避免一次性重写 11k 行）
pub(crate) use domain::{chat_tpl, repeat_guard, toolcall, vision as vision_processor};
pub(crate) use infra::config;
pub(crate) use repo::{trace as agent_trace, vram as vram_ledger};
pub(crate) use view::{http, router};

use std::time::Duration;

use tokio::net::TcpListener;

use crate::config::ServerConfig;
use owl_engine::{EngineService, ServiceExit};

/// 引擎请求通道深度(pump 期间的提交在缓冲;收件与泵不相交纪律)
const CHANNEL_CAPACITY: usize = 64;

/// accept 失败退避(防 fd 耗尽等错误风暴下忙旋)
const ACCEPT_BACKOFF: Duration = Duration::from_millis(100);

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
            // 模板变量（2026-11-16 起独立文件，不属于引擎配置）
            match &config.agent.vars_source {
                Some(p) => owl_shared::log_info!(
                    "boot",
                    "chat 模板变量:{} ({} 项)",
                    p.display(),
                    config.agent.vars.len()
                ),
                None => owl_shared::log_info!("boot", "chat 模板变量:无文件(模板取自身缺省)"),
            }
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
    // 并发上限的**唯一真源 = `store.gdn.gdn_slots`**（2026-11-16：`[runtime]
    // max_concurrency` 已删，permit 数与引擎槽位上限都从这里出）。
    let max_concurrency = config.max_concurrency;
    let max_batch = config.max_batch;
    owl_shared::log_info!(
        "boot",
        "并发上限 {max_concurrency}(= store.gdn.gdn_slots;本地服务,排队超出请求)"
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
            max_context_length_tokens: config.max_context_length,
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
    let engine = crate::repo::engine::EngineRepo::new(client);

    // 并发闸门（permit = `store.gdn.gdn_slots`；在跑 + 排队 ≤ 它）。
    // 两者同一口径是**类型级**保证：`max_concurrency` 就是 `gdn_slots` 的投影，
    // 不再有"配错会污染状态格"的组合（旧的 `gdn_slots ≥ max_concurrency` 门已删）。
    let concurrency_gate = std::sync::Arc::new(tokio::sync::Semaphore::new(max_concurrency));
    // **会话串行已下沉引擎**（2026-11-16 P4）：同会话后继 turn 在引擎队列里
    // 等前一个终了（`startable_queue_idx`），服务面不再需要 per-session 闸。
    // 口径告警(不拒启:语义仍然正确,只是这类客户端拿不到合批收益)
    if max_batch > 1 {
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
                 —— 想让批吃满,`store.gdn.gdn_slots` 应 ≥ `runtime.max_batch`"
            );
        }
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
                let ctx = router::ConnCtx {
                    engine: engine.clone(),
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
                    if let Err(e) = router::serve(stream, ctx).await {
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

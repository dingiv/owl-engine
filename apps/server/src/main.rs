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
//! 架子边界(挂账):chat 模板 naive 渲染(openai.rs)、keep-alive/chunked
//! (http.rs)、turn abort(actor.rs pump Err 处置)、引擎就绪前的 503 健康
//! 面、usage 模型侧真账(现 = 事件面计数)。

mod actor;
mod config;
mod debug;
mod http;
mod openai;

use std::time::Duration;

use serde_json::json;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpListener;
use tokio::sync::mpsc;

use crate::actor::EngineReq;
use crate::config::{ModelKind, ServerConfig};

/// 引擎请求通道深度(pump 期间的提交在缓冲;收件与泵不相交纪律)
const CHANNEL_CAPACITY: usize = 64;

/// accept 失败退避(防 fd 耗尽等错误风暴下忙旋)
const ACCEPT_BACKOFF: Duration = Duration::from_millis(100);

#[tokio::main]
async fn main() {
    // CLI:唯一参数面(配置文件路径 + flavor;手写解析,未知参数拒启)
    let args = match crate::config::CliArgs::parse(std::env::args().skip(1)) {
        Ok(a) => a,
        Err(e) => {
            owl_shared::log_info!("boot", "{e}");
            std::process::exit(1);
        }
    };
    if args.list_flavors {
        eprintln!("可用 flavor:");
        for n in owl_shared::config::flavor_names() {
            eprintln!("  {n}");
        }
        return;
    }
    // 配置:四层叠加(Default → Flavor → 文件 → CLI),非法即拒
    let config = match ServerConfig::load(args.flavor.as_deref(), args.config.as_deref(), &args) {
        Ok(c) => c,
        Err(e) => {
            owl_shared::log_info!("boot", "配置非法: {e}");
            std::process::exit(1);
        }
    };

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
            Err(e) => owl_shared::log_info!("boot", "{e}"),
        }
    }

    // metrics 全局仓(E5-M5 spec 分相探针消费面;/debug/metrics 查询;
    // 重复 init = AlreadyInit 保首个,loader 侧 init 幂等兼容)
    let _ = owl_shared::metrics::init_metrics(owl_shared::metrics::MetricsStore::new());
    // 性能框架分级(docs/arch/perf-framework.md §一;off = 零成本基线臂)
    owl_shared::metrics::set_level(config.perf_level);
    owl_shared::log_info!("boot", "perf level = {}(perf_build={};无该 feature 的 release 里钩子空展开)",
        owl_shared::metrics::level_name(),
        owl_shared::metrics::PERF_BUILD,
    );
    if let Some(p) = &config.perf_dump {
        owl_shared::log_info!("boot", "perf dump → {}(每 turn 终了覆写 CSV + .json)", p.display());
    }

    // ── engine actor:专属线程(构造+装载+泵全在内;RunningEngine 非 Send)──
    // HTTP 侧留 bind/model_name 副本(model_name 回退服务端缺省)
    // 并发上限守卫(§三十五 server 上层):GDN 状态格 = 每并发会话一格,
    // max_concurrency 超格数 = 第 N+1 会话状态格耗尽(LRU 逐出活跃会话)
    // → 跨会话状态污染。不足即拒启(账外组合不静默)。
    let gdn_slots = config.knobs.gdn_slots;
    let max_concurrency = config.max_concurrency;
    if gdn_slots < max_concurrency {
        owl_shared::log_info!("boot", "❌ max_concurrency({max_concurrency}) > GDN 状态格数({gdn_slots})—— 每并发会话占一格,请调 knobs.gdn_slots ≥ max_concurrency 或降低并发"
        );
        std::process::exit(1);
    }
    owl_shared::log_info!("boot", "并发上限 {max_concurrency}(GDN 格 {gdn_slots};本地服务,排队超出请求)"
    );

    let (bind, model_name) = (config.bind.clone(), config.model_name.clone());
    let (tx, rx) = mpsc::channel::<EngineReq>(CHANNEL_CAPACITY);
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<String, String>>();
    std::thread::Builder::new()
        .name("owl-engine".into())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("actor runtime");
            rt.block_on(actor_main(config, rx, &ready_tx));
        })
        .expect("actor 线程");

    // 装配结果(引擎就绪前不开 HTTP 面;架子边界)
    match ready_rx.recv().expect("actor 线程存活") {
        Ok(capture) => owl_shared::log_info!("boot", "引擎就绪(捕获态 {capture})"),
        Err(e) => {
            // 拒启是运维必须看见的事件 ⇒ 恒输出(不经级别门)
            eprintln!("[boot] 引擎装配失败: {e}");
            std::process::exit(1);
        }
    }

    // 并发上限守卫(§三十五 server 上层):GDN 状态格 = 每并发会话一格,
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
                let tx = tx.clone();
                let model_name = model_name.clone();
                let gate = concurrency_gate.clone();
                tokio::spawn(async move {
                    // 并发闸门:permit 获取排队(超出 max_concurrency 等待);
                    // 获取后持有至该连接服务完毕(长连接 = 占一并发位)
                    let _permit = gate.acquire().await;
                    if let Err(e) = serve(stream, tx, model_name).await {
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

/// actor 线程主:引擎生命周期 facade(装配)→ 事件泵(actor::run_actor)。
/// 装配决策已在 config 落地 —— 本函数零 env 读取,档位分派 = 枚举查表。
async fn actor_main(
    config: ServerConfig,
    rx: mpsc::Receiver<EngineReq>,
    ready: &std::sync::mpsc::Sender<Result<String, String>>,
) {
    let built: owl_engine::Result<_> = async {
        let t0 = std::time::Instant::now();
        let mut engine = owl_engine::Engine::new(owl_engine::EngineConfig {
            device_ordinal: config.device,
            max_seq_tokens: config.max_seq,
            prefill_chunk: config.prefill_chunk,
            knobs: config.knobs,
        })?;
        owl_shared::log_info!("boot", "设备绑定 {:.2}s", t0.elapsed().as_secs_f32());
        let model = match config.model_kind {
            ModelKind::Awq27b => {
                // 量化方案映射(唯一点 owl_engine::head_plan_of)
                let head_plan = owl_engine::head_plan_of(config.head_quant);
                engine
                    .loader()
                    .load_qwen38_27b_awq(&config.model_dir, &config.model_dir, head_plan)
                    .await?
            }
            ModelKind::Qwen35_08b => engine.loader().load_qwen35_0_8b(&config.model_dir).await?,
        };
        owl_shared::log_info!("boot", "权重/tokenizer/rope 装载 {:.2}s", t0.elapsed().as_secs_f32());
        let running = engine.run(model).await?;
        owl_shared::log_info!("boot", "引擎装配(run)总计 {:.2}s", t0.elapsed().as_secs_f32());
        Ok(running)
    }
    .await;
    let mut running = match built {
        Ok(r) => {
            let _ = ready.send(Ok(format!("{:?}", r.capture_outcome)));
            r
        }
        Err(e) => {
            let _ = ready.send(Err(e.to_string()));
            return;
        }
    };
    actor::run_actor(&mut running, rx, config.perf_dump.clone()).await;
}

/// 单连接服务(Connection: close;架子一请求一连接)
async fn serve(
    mut stream: tokio::net::TcpStream,
    tx: mpsc::Sender<EngineReq>,
    model_name: String,
) -> std::io::Result<()> {
    let Some(req) = http::read_request(&mut stream).await? else {
        return Ok(());
    };
    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/health") => {
            http::respond_json(&mut stream, 200, &json!({"ok": true, "model": model_name})).await?;
        }
        ("GET", "/v1/models") => {
            http::respond_json(
                &mut stream,
                200,
                &json!({"object": "list", "data": [{
                    "id": model_name, "object": "model", "owned_by": "owl",
                }]}),
            )
            .await?;
        }
        ("POST", "/v1/chat/completions") => {
            openai::chat(&mut stream, &req.body, tx, &model_name).await?;
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
    stream.shutdown().await
}

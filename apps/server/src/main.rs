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
use tokio::io::AsyncWriteExt;
use tokio::net::TcpListener;
use tokio::sync::mpsc;

use crate::actor::EngineReq;
use crate::config::{ModelKind, ServerConfig};

/// 引擎请求通道深度(pump 期间的提交在缓冲;收件与泵不相交纪律)
const CHANNEL_CAPACITY: usize = 64;

/// accept 失败退避(防 fd 耗尽等错误风暴下忙旋)
const ACCEPT_BACKOFF: Duration = Duration::from_millis(100);

/// **诊断投影面共享单元**(`/debug/blocks` 的 GPU 命令出口)。
/// 显式依赖律:不做全局变量 —— `main` 建、actor 装、HTTP 读,全走参数。
type DbgFace =
    std::sync::Arc<std::sync::OnceLock<std::sync::Arc<tokio::sync::Mutex<owl_cuda::GpuClient>>>>;

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
    if args.list_flavors {
        eprintln!("可用 flavor:");
        for n in owl_shared::assets::preset_names() {
            eprintln!("  {n}");
        }
        return;
    }
    // 配置:四层叠加(Default → Flavor → 文件 → CLI),非法即拒
    let config = match ServerConfig::load(args.flavor.as_deref(), args.config.as_deref(), &args) {
        Ok(c) => c,
        Err(e) => {
            // 拒启必须可见(A1):此刻 logger 尚未初始化,`log_info!` 会被丢弃
            eprintln!("[boot] 配置非法: {e}");
            std::process::exit(1);
        }
    };

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
    // 信号:SIGTERM/SIGINT ⇒ 清锁 + 有序退出(`process::exit` 不跑 Drop ⇒ 显式 release)
    {
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
            std::process::exit(0);
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
    // 并发上限守卫(§三十五 server 上层):GDN 状态格 = 每并发会话一格,
    // max_concurrency 超格数 = 第 N+1 会话状态格耗尽(LRU 逐出活跃会话)
    // → 跨会话状态污染。不足即拒启(账外组合不静默)。
    // FIXME: ??? 你server层的东西，你来感知引擎内部的逻辑???
    let gdn_slots = config.knobs.gdn_slots;
    let max_concurrency = config.max_concurrency;
    if gdn_slots < max_concurrency {
        // 拒启必须可见(A1):默认日志级别 warn 下 log_info 不出 ⇒ 裸 eprintln
        eprintln!(
            "[boot] ❌ max_concurrency({max_concurrency}) > GDN 状态格数({gdn_slots})—— 每并发会话占一格,请调 knobs.gdn_slots ≥ max_concurrency 或降低并发"
        );
        std::process::exit(1);
    }
    owl_shared::log_info!("boot", "并发上限 {max_concurrency}(GDN 格 {gdn_slots};本地服务,排队超出请求)"
    );

    let (bind, model_name) = (config.bind.clone(), config.model_name.clone());
    // agent 服务面旋钮(模板已装载用于 log;此处交协议面消费)
    let agent = std::sync::Arc::new(config.agent.clone());
    // **诊断投影面共享单元**(见 `DbgFace` 注):actor 装、HTTP 读
    let dbg_face: DbgFace = std::sync::Arc::new(std::sync::OnceLock::new());
    let dbg_face_actor = dbg_face.clone();
    let (tx, rx) = mpsc::channel::<EngineReq>(CHANNEL_CAPACITY);
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<String, String>>();
    std::thread::Builder::new()
        .name("owl-engine".into())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("actor runtime");
            rt.block_on(actor_main(config, rx, &ready_tx, dbg_face_actor));
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
                let dbg_face = dbg_face.clone();
                let model_name = model_name.clone();
                let tpl = tpl.clone();
                let agent = agent.clone();
                let gate = concurrency_gate.clone();
                tokio::spawn(async move {
                    // 并发闸门:permit 获取排队(超出 max_concurrency 等待);
                    // 获取后持有至该连接服务完毕(长连接 = 占一并发位)
                    let _permit = gate.acquire().await;
                    if let Err(e) = serve(stream, tx, model_name, tpl, agent, dbg_face.get().cloned()).await {
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
    let mut loader = engine.loader().slice(rank, world);
    match config.model_kind {
        ModelKind::Awq27b => {
            // 量化方案映射(唯一点 owl_engine::head_plan_of)
            let head_plan = owl_engine::head_plan_of(config.head_quant);
            loader
                .load_qwen38_27b_awq(&config.model_dir, &config.model_dir, head_plan, config.w4a8)
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
) {
    let built: owl_engine::Result<_> = async {
        let t0 = std::time::Instant::now();
        let world = config.devices.world();
        let mut engine = engine_of(&config, config.devices.device())?;
        owl_shared::log_info!("boot", "设备绑定 {:.2}s", t0.elapsed().as_secs_f32());
        match world {
            // ── 单卡档(原路径;分片 = 恒等)─────────────────────────────
            1 => {
                let model = load_shard(&config, &mut engine, 0, 1).await?;
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
            // `RunningEngine<GpuClient>`。
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
                    owl_engine::TpOpts { push: config.tp_push },
                )
                .await?;
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
    actor::run_actor(&mut running, rx, config.perf_dump.clone()).await;
}

/// 单连接服务(Connection: close;架子一请求一连接)
async fn serve(
    mut stream: tokio::net::TcpStream,
    tx: mpsc::Sender<EngineReq>,
    model_name: String,
    tpl: std::sync::Arc<chat_tpl::ChatTpl>,
    agent: std::sync::Arc<owl_shared::config::AgentCfg>,
    // 诊断投影用的 face 句柄(`GET /debug/blocks`):engine 侧 `face_clone()`
    // 得到的同 actor 发送端;None = 该后端不支持克隆(端点回 503)
    dbg_face: Option<std::sync::Arc<tokio::sync::Mutex<owl_cuda::GpuClient>>>,
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
            openai::chat(&mut stream, &req.body, tx, &model_name, &tpl, &agent).await?;
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
        // **块账本投影**(判"同 id 两地址 / 两活块区间相交";与发射账互补:
        // 那张账 = 每次命令的块 id+地址,本投影 = 当前账本的权威快照)
        ("GET", "/debug/blocks") => {
            use owl_cuda::DeviceClient as _;
            match dbg_face.as_ref() {
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
    stream.shutdown().await
}

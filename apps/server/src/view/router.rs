//! **视图层：连接循环 + 路由 + 响应写回**（从 `main.rs` 收拢，2026-11-17）。
//!
//! 本文件只做 HTTP 协议适配：读帧 → 取并发位 → 按路径分发 → 调 service/repo
//! → 写 JSON/SSE。业务编排在 `service`，引擎/账本在 `repo`。

use serde_json::json;

use std::time::Duration;

use crate::config::AgentCfg;
use crate::domain::chat_tpl::ChatTpl;
use crate::repo::engine::EngineRepo;
use crate::repo::trace::AgentTrace;
use crate::service::{chat as openai, load};
use crate::view::debug;
use crate::view::http;

/// **keep-alive 空闲超时**(G5④):两次请求之间静默超过它 ⇒ 服务端主动关。
const KEEP_ALIVE_IDLE: Duration = Duration::from_secs(30);

/// **单连接请求数上限**(G5④):防“一个连接永远不换”在长时运行里变成 fd/任务的
/// 无限累积;达上限主动关(客户端重建连接,行为无感)。
const MAX_REQUESTS_PER_CONN: usize = 1024;

/// HTTP 面共享句柄(2026-10-19,G5② 收拢:参数逐个下传已到临界;
/// `/debug/agent` 是接入的第四方依赖 —— 再堆参数就要靠注释对位了)。
#[derive(Clone)]
pub(crate) struct ConnCtx {
    /// 引擎仓库门面（repo 层；view 不直接认 `owl_engine` 句柄）
    pub(crate) engine: EngineRepo,
    pub(crate) model_name: String,
    /// 原始请求 JSON 落盘目录（None = 关）
    pub(crate) request_dump_dir: Option<std::path::PathBuf>,
    /// 单图像素上限(视觉预处理封顶;0 = 内置 4096² = 不缩)
    pub(crate) vision_max_pixels: usize,
    pub(crate) tpl: std::sync::Arc<ChatTpl>,
    pub(crate) agent: std::sync::Arc<AgentCfg>,
    /// `/debug/agent` 诊断环(G5②;显式依赖,不做全局变量)
    pub(crate) trace: std::sync::Arc<AgentTrace>,
    /// 并发闸门(G5④:**每请求**获取;在跑 + 排队 ≤ max_concurrency)
    pub(crate) gate: std::sync::Arc<tokio::sync::Semaphore>,
    /// 服务面并发上限(`server.max_concurrency`;`/debug/load` 按它校验)
    pub(crate) max_concurrency: usize,
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
pub(crate) async fn serve(mut stream: tokio::net::TcpStream, ctx: ConnCtx) -> std::io::Result<()> {
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
                ctx.engine.client().clone(),
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
            let snap = ctx.engine.client().queue_facts();
            http::respond_json(&mut stream, 200, &debug::queue_json(&snap)).await?;
        }
        // **内源灌流面**(服务面并发能力的现场测量;见 `load` 模块头)
        ("POST", "/debug/load") => {
            load::load(
                &mut stream,
                &req.body,
                ctx.engine.client().clone(),
                &ctx.tpl,
                &ctx.agent,
                ctx.max_concurrency,
            )
            .await?;
        }
        // **tokenize 面**(G5③):纯文本/家族包装两模式;
        // 延迟说明见 `openai::tokenize` 头注(引擎单线程)
        ("POST", "/tokenize") => {
            openai::tokenize(&mut stream, &req.body, ctx.engine.client().clone()).await?;
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
            match ctx.engine.client().diag().await {
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
        ("GET", "/debug/blocks") => match ctx.engine.client().diag().await {
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

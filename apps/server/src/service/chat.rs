//! OpenAI 兼容面(chat completions;**agent 服务面专项 §3 契约**)。
//!
//! 会话映射(引擎 S0 接线):OpenAI 协议无状态、客户端全量重发历史 ——
//! 与引擎会话模型(全量 prompt + 前缀增量)天然同构。请求扩展字段
//! `"session"`(任意字符串)为会话键:同键跨请求 = 引擎侧同一会话。
//!
//! 模块布局:§1 协议数据 + 校验(不静默丢弃)/ §2 JSON 构造 /
//! §3 三通道事件流消费(思考 / 正文 / 工具调用)。
//!
//! **旧 `render_prompt` 已退役**(§4.3 渲染对称律 R2):prompt 一律经
//! [`ChatTpl`] 从模型快照 `chat_template.jinja` 渲染 —— 手写模板串
//! = 前缀复用零命中的根因。
//!
//! 挂账(M-C):取消/abort 的上游用法(`EngineReq::Abort` 已就位,pi 侧
//! 断流由引擎泵循环自动弃置)。

use std::sync::Arc;

use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::mpsc;

use owl_engine::{EngineClient, StreamEvent, TurnEvent, TurnOpts};
use crate::config::AgentCfg;
use crate::agent_trace::{AgentTrace, OutcomeFacts, ReqFacts, TurnLedger, UsageFacts};
use crate::chat_tpl::{ChatTpl, RenderArgs};
use crate::http;
use crate::toolcall::{raw_for_log, Chan, ChannelSplitter, Item, ToolSchemas};

// ============================================================================
// §1 协议数据与校验
pub(crate) use crate::domain::openai::*;

/// chat completions:提交引擎 → 首笔 Accepted/Rejected → 三通道折叠循环。
// 结构性豁免(2026-11-02 clippy 清理):派发/契约 ABI:参数表就是调用面,拆结构体会把每个调用点变成二次装箱
#[allow(clippy::too_many_arguments)]
/// **`w` 是具体 `TcpStream` 而不是泛型 `AsyncWrite`**(2026-11-03):流式循环要多
/// 一路事件源 —— **读半 FIN**(`peek`,非消费)。只有 `TcpStream` 能 `peek`;调用点
/// 只有 `main::handle` 一处 ⇒ 具体化零代价。
pub(crate) async fn chat(
    w: &mut tokio::net::TcpStream,
    body: &[u8],
    tx: EngineClient,
    model_name: &str,
    tpl: &Arc<ChatTpl>,
    agent: &AgentCfg,
    // `/debug/agent` 诊断环(G5②):每轮必落一条(`TurnLedger` 的 Drop 是兜底网)
    trace: &Arc<AgentTrace>,
    // **并发 permit**：在会话解析之后取（同会话的排队交给引擎，不再占位）
    gate: &Arc<tokio::sync::Semaphore>,
    // 原始请求 JSON 落盘目录（None = 关）
    request_dump_dir: Option<&std::path::Path>,
    // 单图像素上限（视觉预处理封顶；0 = 内置 4096² = 不缩）
    vision_max_pixels: usize,
) -> std::io::Result<()> {
    if let Some(dir) = request_dump_dir {
        dump_request(dir, body);
    }
    let mut req: ChatRequest = match serde_json::from_slice::<ChatRequest>(body) {
        Ok(r) => r,
        Err(e) => {
            return http::respond_json(w, 400, &error_json(format!("请求体非法: {e}"))).await
        }
    };
    if let Err(e) = req
        .validate()
        .and_then(|_| req.stop_strings().map(|_| ()))
        .and_then(|_| req.normalize_tool_calls())
    {
        return http::respond_json(w, 400, &error_json(e)).await;
    }
    // 图片删除的文本边界规范化(渲染前;plain string / 单 text 数组零变化)。
    let _ = req.normalize_multipart_text();
    for wmsg in req.warnings() {
        owl_shared::log_warn!("chat", "收下但暂未生效(AGT-06 后仅余未知字段):{wmsg}");
    }
    let stop = match req.stop_strings() {
        Ok(s) => s,
        Err(e) => return http::respond_json(w, 400, &error_json(e)).await,
    };
    let mode = if req.stream { ReplyMode::Stream } else { ReplyMode::Aggregate };
    let model_name = req.model.as_deref().unwrap_or(model_name);
    let (max_new, clamped) = req.max_new(agent);
    if clamped {
        owl_shared::log_warn!(
            "chat",
            "max_tokens={:?} 超出本档上限 {},已夹紧(改 [agent] max_new_cap 随池定档放开)",
            req.max_tokens,
            agent.max_new_cap
        );
    }
    // 思考档位(AGT-07):OpenAI `reasoning_effort` **直通模板变量**
    // (模板自持值域: none|low|medium|high|xhigh|…)。请求 > 配置缺省
    let kw = req.effective_template_kwargs(&agent.vars);
    let kw_ref = if kw.is_empty() { None } else { Some(&kw) };
    // 模板变量 + 请求 kwargs 合并后的思考态（splitter / 日志 / 诊断同一份）
    let thinking = req.enable_thinking(
        &agent.vars,
        req.tools.as_ref().is_some_and(|t| !t.is_empty()),
    );
    // ── 会话身份 ──
    //
    // 会话 id 只有两个消费者：前缀复用（同会话增量）与断流取消点名。
    // **会话串行已下沉引擎**（2026-11-16 P4）：同会话第二个 turn 由引擎排队，
    // 在会话空闲前不会起跑（`startable_queue_idx`）—— 服务面不再有闸，
    // 也不再需要"闸之后才取 permit"的次序纪律（合并批档 SSE 语义不变）。
    let session = req.session.as_deref().map(session_id);
    // 渲染:唯一权威 = 快照模板(§4.3 R2);模板拒(如 system 含图像)= 400
    let mut prompt = match tpl.render(RenderArgs {
        messages: &req.messages,
        tools: req.tools.as_deref(),
        add_generation_prompt: true,
        template_kwargs: kw_ref,
    }) {
        Ok(p) => p,
        Err(e) => return http::respond_json(w, 400, &error_json(e.to_string())).await,
    };
    // 图像:预处理(与模板顺序一致)+ **占位 token 展开**:
    // 模板对每张图只发一个 `<|image_pad|>`;视觉塔要求 `grid_h*grid_w/4` 个。
    let images = match req.images(vision_max_pixels) {
        Ok(v) => v,
        Err(e) => return http::respond_json(w, 400, &error_json(e)).await,
    };
    if !images.is_empty() {
        prompt = match expand_image_pads(&prompt, &images) {
            Ok(p) => p,
            Err(e) => return http::respond_json(w, 400, &error_json(e)).await,
        };
    }
    let (sink_tx, mut sink_rx) = mpsc::channel::<StreamEvent>(SINK_CAPACITY);
    // 请求级图像事实（日志/诊断；图像 token = 视觉占位展开量的真源）。
    let image_count = images.len();
    let image_tokens = images
        .iter()
        .map(|i| (i.image.grid.1 * i.image.grid.2) / 4)
        .sum::<usize>();
    let evict_images = images.iter().filter(|i| i.evict).count();
    // **并发位**：闸之后再占（同会话排队不占 permit；跨会话才竞争）
    let _permit = gate.acquire().await;
    // G5②:诊断环开账(此后的每一条返回路径都可见 —— 显式 fail 或 Drop 兜底)
    let mut led = TurnLedger::new(trace);
    // **跳轮重复探测**(2026-10-31;只可见):模型反复发同一个 tool_call 时
    // 上下文尾部同形 ⇒ 输出逐字复现的固定点(生产实录 17 轮)。这里只记账 +
    // WARN,不改生成 —— 升级成提示注入/截断是另一道裁决。
    let mut repeat = crate::repeat_guard::scan(&req.messages);
    if let Some(f) = repeat.as_ref() {
        if f.triggered() {
            owl_shared::log_warn!(
                "chat",
                "跳轮重复(可见态):{} —— 客户端连续重发同一调用;本轮未做干预",
                f.summary()
            );
        }
    }
    led.set_req(ReqFacts {
        model: model_name.to_string(),
        stream: mode == ReplyMode::Stream,
        messages: req.messages.len(),
        roles: req
            .messages
            .iter()
            .filter_map(|m| m.get("role").and_then(|r| r.as_str()))
            .collect::<Vec<_>>()
            .join(","),
        tools: req.tools.as_ref().map(|t| t.len()).unwrap_or(0),
        images: image_count,
        image_tokens,
        evict_images,
        thinking,
        reasoning_effort: req.reasoning_effort.clone(),
        session,
        max_new,
        ignore_eos: req.ignore_eos,
        stop: stop.len(),
        repeat: repeat.take(),
    });
    led.set_prompt(&prompt);
    let chat_opts = TurnOpts {
            ignore_eos: req.ignore_eos,
            sampler: req.sampler_override(),
            stop,
            seed: req.seed.map(|s| s as u64),
            // 直接显存地板(**per-turn**,设计卷 §2.4/§23.5):OpenAI 面暂无
            // 扩展字段 —— 走引擎 `knobs.dense_floor`(缺省 `Strict`)。
            // 将来加请求字段时在这里映射;非 `Strict` 由引擎**响亮拒**
            // (选择层 B 轴未落地,地板语义无从兼现)。
            dense_floor: None,
            images,
    };
    if tx
        .chat(session, prompt, max_new, chat_opts, sink_tx)
        .await
        .is_err()
    {
        return http::respond_json(w, 503, &error_json("引擎退役")).await;
    }

    // 首笔:受理确认 / 提交拒绝(预算越界等,submit 边界落地)
    let mut usage = Usage::default();
    // 预算被夹紧 ⇒ 响应头如实带出（`X-Owl-Max-New`；非私有 body 字段）
    let mut clamped_max_new: Option<usize> = None;
    let turn = match sink_rx.recv().await {
        Some(StreamEvent::Accepted { session, turn, prompt_tokens, cached_tokens, max_new }) => {
            usage.prompt = prompt_tokens;
            usage.cached = cached_tokens;
            // 预算被准入夹紧 ⇒ 如实留痕（客户端会看到 finish_reason=length）
            if max_new < req.max_new(agent).0 {
                clamped_max_new = Some(max_new);
                owl_shared::log_warn!(
                    "chat",
                    "生成预算被夹紧:请求 {} → 实际 {max_new}(session {session:#x} turn {turn})",
                    req.max_new(agent).0
                );
            }
            owl_shared::log_info!(
                "chat",
                "受理 session {session:#x} turn {turn}(stream={};tools={};images={image_count}/tokens={image_tokens}/evict={evict_images};think={};prompt={prompt_tokens};cached={cached_tokens};max_new={max_new})",
                mode == ReplyMode::Stream,
                req.tools.as_ref().map(|t| t.len()).unwrap_or(0),
                thinking,
            );
            Some((session, turn))
        }
        Some(StreamEvent::Rejected { err }) => {
            led.fail(format!("提交被拒:{err}"));
            // **容量拒绝**（引擎 submit 边界的结构化 `AdmissionReject`）⇒ 429；
            // 其余（预算越界）仍走 pi 能认的 400 context-overflow 报文。
            if let Some(reject) = err.reject() {
                let retry = retry_after_secs(reject).to_string();
                return http::respond_json_hdr(
                    w,
                    429,
                    &error_json_rate_limit(
                        reject.code(),
                        "服务端容量不足（排队已满）—— 请按 Retry-After 退避重试",
                    ),
                    &[("Retry-After", retry.as_str())],
                )
                .await;
            }
            return http::respond_json(w, 400, &error_json_for_reject(err.message().to_string()))
                .await;
        }
        // 契约:首笔必为 Accepted/Rejected(executor 在 pump 前投递)⇒ 不会走到
        // 这里(真走到 = 协议序错乱,不静默吞)
        Some(StreamEvent::CachedTokens { .. }) => {
            return http::respond_json(
                w,
                503,
                &error_json("协议序错乱:缓存校正帧先于受理确认"),
            )
            .await
        }
        _ => return http::respond_json(w, 503, &error_json("引擎退役")).await,
    };

    let id = completion_id();
    let mut splitter = ChannelSplitter::new(
        thinking,
        ToolSchemas::from_tools(req.tools.as_deref().unwrap_or(&[])),
    )
    // 工具解析**恒开**（2026-11-16 认领 `[server] parse_tool_calls`：
    // 没有生产档关过它；唯一用途是"工具标记按普通文本放行"的诊断臂，
    // 该需求由 `/debug/agent` 的结构轨迹覆盖。旧键已删 = 旧档里写它会被拒）。
    .with_tool_parsing(true);
    let mut acc = Agg {
        seed: unix().as_nanos() as u64,
        ..Default::default()
    };
    // role 首帧延到首个内容帧(2026-10-02:提前发会让测速端把 HTTP 往返
    // 当 TTFT,长 prompt 的 prefill 全被藏进首帧延迟)
    let mut role_sent = false;
    // 工具协议错误(畸形/缺 required/未知工具):一旦置位,后续通道全部静音,
    // 终态按错误收口 —— 绝不再把半截工具块或空 stop 当正常结果返回。
    let mut protocol_error: Option<&'static str> = None;
    // 写失败 ⇒ 显式弃置引擎 turn(AGT-10):executor 侧虽会在下次投递失败时
    // 发现,但显式 abort 立刻停(长思考/长输出下差几十秒 GPU)。
    // 注:宏必须在 `acc`/`role_sent` 作用域内定义 —— macro_rules 卫生:
    // 宏体内标识符按**定义点**解析,定义在前就看不到(实测踩过)
    macro_rules! emit {
        ($items:expr) => {
            if let Err(e) = emit_items(
                w,
                mode,
                &id,
                model_name,
                &mut acc,
                &mut role_sent,
                &mut protocol_error,
                $items,
            )
            .await
            {
                // **先关 sink 再等回执**(2026-10-20 真机定谳):本任务若还持有
                // `sink_rx` 等 abort 回执,executor 的投递就一直成功 ⇒ 它**察觉不到**
                // 断流,会把整个 turn 跑完(实测:256 tok 的流断掉后仍烧满 5s+,
                // 回执超时、队列位也被拖住)。`close()` 让 executor 下一笔投递失败
                // ⇒ 走既有 "断流 ⇒ 弃置"(≤1 采样步)。
                sink_rx.close();
                abort_turn(&tx, session, turn.map(|(_, t)| t)).await;
                return Err(e);
            }
        };
    }
    // **断流收尾单一路径**:日志 → 关 sink → 指名取消 → 等回执 → 返回。
    // 三处断流源(保活写失败 / 对端 FIN / select 里的保活到点)共用它 ——
    // 旧版三份手写序列,改一处漏两处。
    macro_rules! drop_turn {
        ($fmt:literal $(, $arg:expr)*) => {{
            owl_shared::log_info!("chat", $fmt, $($arg),*);
            close_and_abort(&mut sink_rx, &tx, session, turn).await;
            return Ok(());
        }};
    }
    if mode == ReplyMode::Stream {
        match clamped_max_new {
            Some(m) => {
                let v = m.to_string();
                http::respond_sse_head_hdr(w, &[("X-Owl-Max-New", v.as_str())]).await?
            }
            None => http::respond_sse_head(w).await?,
        }
    }
    // **排队超时**(`[runtime] queue_wait_ms`;0 = 关):提交后迟迟不见实质进展 ⇒
    // 取消该 turn + 429（D2 的 R 档）。`started` = 已见到 prefill/token/终态。
    //
    // **静默保活**(传输层定值,见 `SSE_KEEPALIVE_MS`;2026-11-03):长 prefill 期间流上
    // 一个字节都不写 ⇒ chunked 写永远不会失败 ⇒ 对端 FIN 无人察觉 ⇒ **引擎把
    // 整段 prefill 白跑完**(真机:前端在 100k prompt 的 prefill 中断开,引擎仍
    // 跑满 50s)。这里按周期写一帧 SSE **注释**(`: keepalive`;解析器按规范忽略),
    // 写失败 = 连接已死 ⇒ 立刻弃流 + 弃置 turn。
    let ka_wait = std::time::Duration::from_millis(SSE_KEEPALIVE_MS);
    let mut ka_at = if mode == ReplyMode::Stream {
        Some(tokio::time::Instant::now() + ka_wait)
    } else {
        None
    };
    let mut started = false;
    loop {
        // **保活到点即发**(必须在等事件之前判):`Prefill` 事件本身**不写字节**,
        // 而它每 chunk 就来一笔(0.7~1.2s);若把保活判断放在 select 分支里,
        // 密集事件会把它永久饿死(实测:28k prompt 的 prefill 期间一个保活帧都没发,
        // 断流照样白跑到尾)。
        if ka_at.is_some_and(|t| tokio::time::Instant::now() >= t) {
            if let Err(e) = http::sse_comment(w, "keepalive").await {
                drop_turn!(
                    "保活帧写失败({e}) ⇒ 客户端已断流,弃置 turn {:?}(prefill 阶段也可停)",
                    turn.map(|(_, t)| t)
                );
            }
            ka_at = Some(tokio::time::Instant::now() + ka_wait);
            continue;
        }
        // 排队超时只对"尚未起跑"生效(每拍重算 —— 起跑后即失效)
        let queue_at = if !started && agent.queue_wait_ms > 0 {
            Some(
                tokio::time::Instant::now()
                    + std::time::Duration::from_millis(agent.queue_wait_ms as u64),
            )
        } else {
            None
        };
        let next = tokio::select! {
            biased;
            // ① 引擎事件优先
            v = sink_rx.recv() => v,
            // ①' **对端 FIN(读半 EOF)—— 真正的断流事件源**
            //
            // 为什么要这一路:`peek` 非消费、**不需要先写**。客户端正常关闭
            // (浏览器/前端 abort/SDK close)会送 FIN ⇒ 这里**立刻**醒,不必等
            // 保活帧到点(那一档是给"没有 FIN 的掉线"兜底的)。
            // 有数据(HTTP 管线化)= 无法区分后续 EOF ⇒ 本请求内不再盯(还在
            // `peer_fin` 里永久挂起),由保活帧兜底。
            _ = peer_fin(w) => {
                drop_turn!(
                    "读半就绪=对端已关(FIN 或 RST)⇒ 客户端已断流,弃置 turn {:?}(prefill 阶段也可停)",
                    turn.map(|(_, t)| t)
                );
            }
            // ② 静默到点 ⇒ 写一帧注释(失败 = 断流)
            _ = sleep_until_opt(ka_at) => {
                match http::sse_comment(w, "keepalive").await {
                    Ok(()) => {
                        ka_at = Some(tokio::time::Instant::now() + ka_wait);
                        continue;
                    }
                    Err(e) => {
                        // 与 `emit!` 同一处置:先关 sink(executor 下一笔投递失败 ⇒
                        // 走"断流 ⇒ 弃置"),再指名取消 + 等回执
                        drop_turn!(
                            "保活帧写失败({e}) ⇒ 客户端已断流,弃置 turn {:?}(prefill 阶段也可停)",
                            turn.map(|(_, t)| t)
                        );
                    }
                }
            }
            // ③ 排队超时
            _ = sleep_until_opt(queue_at) => {
                owl_shared::log_warn!(
                    "chat",
                    "排队超时 {}ms ⇒ 取消 turn 并回 429（session {:?}）",
                    agent.queue_wait_ms,
                    session
                );
                // 先关 sink（executor 下一笔投递失败 ⇒ 按断流弃置），再指名取消 + 等回执
                close_and_abort(&mut sink_rx, &tx, session, turn).await;
                return respond_queue_timeout(w, mode).await;
            }
        };
        // 只有**真的写了字节**的事件才重置保活计时:`Token` 的处理会写 SSE 帧
        // 与 flush;`Prefill`/`Session` 只改账本、一字节不写 ⇒ 不能重置(否则
        // 每 chunk 一笔 Prefill 就把保活推到 1s 之外,断流永远探不到)。
        if ka_at.is_some()
            && matches!(
                next,
                Some(StreamEvent::Turn(
                    TurnEvent::Token { .. }
                        | TurnEvent::Completed { .. }
                        | TurnEvent::Failed { .. }
                ))
            )
        {
            ka_at = Some(tokio::time::Instant::now() + ka_wait);
        }
        if matches!(
            next,
            Some(StreamEvent::Turn(
                TurnEvent::Prefill { .. }
                    | TurnEvent::Token { .. }
                    | TurnEvent::Completed { .. }
                    | TurnEvent::Failed { .. }
            ))
        ) {
            started = true;
        }
        match next {
            Some(StreamEvent::Turn(TurnEvent::Token { delta, .. })) => {
                usage.completion += 1;
                led.first_token(); // TTFT(幂等)
                let items = splitter.push(&delta);
                emit!(items);
            }
            // Prefill 进度帧(末笔 total = prompt token 数,入 usage)
            Some(StreamEvent::Turn(TurnEvent::Prefill { total, .. })) => {
                usage.prompt = total;
            }
            // 缓存校正帧(AGT-12):块级前缀缓存命中的真值(覆盖 submit 时估值)
            Some(StreamEvent::CachedTokens { turn: t, cached_tokens }) => {
                if turn.map(|(_, x)| x) == Some(t) && cached_tokens != usage.cached {
                    owl_shared::log_info!(
                        "chat",
                        "turn {t} 前缀命中校正 {}(submit 时估值 {})",
                        cached_tokens,
                        usage.cached
                    );
                    usage.cached = cached_tokens;
                }
            }
            Some(StreamEvent::Turn(TurnEvent::Completed {
                text, finish, stats, ..
            })) => {
                let items = splitter.finish();
                emit!(items);
                // **部分可救**(2026-10-31):协议错误只在"一个调用都没拿到"时才
                // 让整轮失败。已有合法调用 ⇒ 照常返回,畸形块只留 WARN ——
                // 把整轮换成错误会让客户端拿同一上下文重放(模型原地复读)。
                if let Some(kind) = protocol_error {
                    if !acc.tools.is_empty() {
                        owl_shared::log_warn!(
                            "chat",
                            "turn {:?} 含畸形工具块({kind}),但已取得 {} 个合法调用 ⇒ 照常返回",
                            turn.map(|(_, t)| t),
                            acc.tools.len()
                        );
                    }
                }
                // **长度截断 ≠ 协议错误**(2026-10-31 实测):`max_tokens` 用完导致
                // 工具块半截(`<function=bash><parameter=command>curl -s "http://`)
                // 时,旧行为 = 502 ⇒ 客户端拿同一请求重放 ⇒ 同样截断 ⇒ 复读。
                // 长度截断是**客户端的预算问题**,如实回 `finish_reason="length"`
                // (上层雷同 OpenAI/vLLM),由客户端决定加预算或换姿势。
                let len_cut = matches!(finish, owl_engine::turn::FinishReason::Length);
                if protocol_error.is_some() && len_cut {
                    owl_shared::log_warn!(
                        "chat",
                        "turn {:?} 因 **length 截断**导致工具块不完整 ⇒ 按 length 返回(不判协议错误)",
                        turn.map(|(_, t)| t)
                    );
                }
                if let Some(kind) = protocol_error.filter(|_| acc.tools.is_empty() && !len_cut) {
                    let msg = format!(
                        "模型工具调用不符合 OpenAI 协议({kind});服务端已拒绝返回无效结果"
                    );
                    led.fail(msg.clone());
                    owl_shared::log_warn!(
                        "chat",
                        "turn {:?} 工具协议错误({kind}) ⇒ 拒绝正常完成",
                        turn.map(|(_, t)| t)
                    );
                    return match mode {
                        ReplyMode::Stream => {
                            let mut chunk =
                                chunk_json(&id, model_name, json!({}), Some("error"));
                            chunk["error"] = json!({
                                "message": msg,
                                "type": "tool_call_error",
                                "code": kind,
                            });
                            http::sse_data(w, &chunk.to_string()).await?;
                            http::sse_data(w, "[DONE]").await?;
                            http::sse_end(w).await?;
                            Ok(())
                        }
                        ReplyMode::Aggregate => {
                            http::respond_json(
                                w,
                                502,
                                &json!({"error": {
                                    "message": msg,
                                    "type": "tool_call_error",
                                    "code": kind,
                                }}),
                            )
                            .await
                        }
                    };
                }
                // 思考相位真账(在 finish 之后取:本 turn 再无 push)
                usage.reasoning = splitter.reasoning_steps();
                // 有工具调用 ⇒ finish_reason = tool_calls(pi 的 mapStopReason 只认
                // stop/length/tool_calls 三者);strays 非零 = 值得一条日志
                if splitter.strays() > 0 {
                    owl_shared::log_warn!(
                        "chat",
                        "turn {:?} 出现 {} 个游标标记(已吞掉不泄漏)",
                        turn.map(|(_, t)| t),
                        splitter.strays()
                    );
                }
                let reason = if splitter.tool_calls() > 0 {
                    "tool_calls"
                } else {
                    match finish {
                        owl_engine::turn::FinishReason::Stop => "stop",
                        owl_engine::turn::FinishReason::Length => "length",
                    }
                };
                // **单轮诊断主日志**(2026-10-25):TTFT / prefill / decode /
                // cache / spec AL 同一条,服务面日志不再只有"完成"。
                // TTFT 含排队与 HTTP;prefill/decode 来自引擎侧 TurnStats
                // (起跑→首 token / 首 token→终态),两者口径分开。
                let ttft_ms = led.ttft_ms().unwrap_or(0);
                let cache_pct = if usage.prompt > 0 {
                    usage.cached as f64 * 100.0 / usage.prompt as f64
                } else {
                    0.0
                };
                owl_shared::log_info!(
                    "chat",
                    "turn {:?} 指标: ttft={}ms; prefill={}ms/{}tok={:.1}t/s; decode={}ms/{}tok={:.1}t/s; cache={}/{} ({:.1}%); spec rounds={} accepted={} al={:.2} deg={}/{}",
                    turn.map(|(_, t)| t),
                    ttft_ms,
                    stats.prefill_ms,
                    stats.prefill_tokens,
                    stats.prefill_tps(),
                    stats.decode_ms,
                    stats.decode_tokens,
                    stats.decode_tps(),
                    usage.cached,
                    usage.prompt,
                    cache_pct,
                    stats.spec_rounds,
                    stats.spec_accepted,
                    stats.spec_al(),
                    stats.spec_degraded_steps,
                    stats.decode_tokens,
                );
                // G5②:结算诊断环(结构轨迹 + usage + 结局;与 wire 同一份账)
                led.set_events(splitter.structural_trace());
                led.commit(
                    UsageFacts {
                        prompt: usage.prompt,
                        completion: usage.completion,
                        cached: usage.cached,
                        reasoning: usage.reasoning,
                    },
                    OutcomeFacts {
                        finish: Some(reason),
                        tools_called: splitter.tool_calls(),
                        strays: splitter.strays(),
                        malformed: splitter.malformed(),
                        prefill_ms: stats.prefill_ms,
                        decode_ms: stats.decode_ms,
                        prefill_tps: stats.prefill_tps(),
                        decode_tps: stats.decode_tps(),
                        spec_rounds: stats.spec_rounds,
                        spec_accepted: stats.spec_accepted,
                        spec_al: stats.spec_al(),
                        ..Default::default()
                    },
                );
                return match mode {
                    ReplyMode::Stream => {
                        let chunk = chunk_json(&id, model_name, json!({}), Some(reason));
                        http::sse_data(w, &chunk.to_string()).await?;
                        // OpenAI 约定:请求 include_usage 时,末帧(DONE 之前)
                        // 发 choices=[] + usage 真账
                        if req.stream_options.as_ref().is_some_and(|o| o.include_usage) {
                            let mut u = chunk_json(&id, model_name, json!({}), None);
                            u["choices"] = json!([]);
                            u["usage"] = json!({
                                "prompt_tokens": usage.prompt,
                                "completion_tokens": usage.completion,
                                "total_tokens": usage.prompt + usage.completion,
                                "prompt_tokens_details": {"cached_tokens": usage.cached},
                                "completion_tokens_details": {"reasoning_tokens": usage.reasoning},
                            });
                            http::sse_data(w, &u.to_string()).await?;
                        }
                        http::sse_data(w, "[DONE]").await?;
                        // chunked 终止块(不调 = 复用连接上的客户端永远等下一 chunk)
                        http::sse_end(w).await?;
                        owl_shared::log_info!(
                            "chat",
                            "turn {:?} 完成({reason};prompt={};cached={};completion={};tools={};think={}B/text={}B):{}",
                            turn.map(|(_, t)| t),
                            usage.prompt,
                            usage.cached,
                            usage.completion,
                            splitter.tool_calls(),
                            acc.reason.len(),
                            acc.text.len(),
                            preview(&text)
                        );
                        Ok(())
                    }
                    ReplyMode::Aggregate => {
                        let msg = agg_message(&acc, splitter.tool_calls());
                        owl_shared::log_info!(
                            "chat",
                            "turn {:?} 完成({reason};prompt={};cached={};completion={};tools={};think={}B/text={}B)",
                            turn.map(|(_, t)| t),
                            usage.prompt,
                            usage.cached,
                            usage.completion,
                            splitter.tool_calls(),
                            acc.reason.len(),
                            acc.text.len()
                        );
                        let body = completion_json(
                            &id,
                            model_name,
                            &msg,
                            reason,
                            usage.prompt,
                            usage.completion,
                            usage.cached,
                            usage.reasoning,
                        );
                        match clamped_max_new {
                            Some(m) => {
                                let v = m.to_string();
                                http::respond_json_hdr(
                                    w,
                                    200,
                                    &body,
                                    &[("X-Owl-Max-New", v.as_str())],
                                )
                                .await
                            }
                            None => http::respond_json(w, 200, &body).await,
                        }
                    }
                };
            }
            Some(StreamEvent::Turn(TurnEvent::Failed { err, .. })) => {
                led.fail(format!("turn 失败:{err}"));
                // **准入拒绝**(引擎结构化 `AdmissionReject`)：用 OpenAI 协议内的
                // 标准手段表达 —— 非流式 `429 + Retry-After`;流式头已发,走既有
                // 错误帧收口(不自造 SSE 事件类型)。
                if let Some(reject) = err.reject() {
                    let retry = retry_after_secs(reject).to_string();
                    let code = reject.code();
                    let msg = format!(
                        "服务端容量不足（{code}）—— 在跑请求优先;请按 Retry-After 退避重试"
                    );
                    return match mode {
                        ReplyMode::Aggregate => {
                            http::respond_json_hdr(
                                w,
                                429,
                                &error_json_rate_limit(code, &msg),
                                &[("Retry-After", retry.as_str())],
                            )
                            .await
                        }
                        ReplyMode::Stream => {
                            if let Err(e) = http::sse_data(w, &error_json(msg).to_string()).await {
                                owl_shared::log_info!("chat", "准入错误帧写失败(客户端可能已断开): {e}");
                            }
                            if let Err(e) = http::sse_data(w, "[DONE]").await {
                                owl_shared::log_info!("chat", "准入收尾帧写失败(客户端可能已断开): {e}");
                            }
                            if let Err(e) = http::sse_end(w).await {
                                owl_shared::log_info!("chat", "准入终止块写失败(客户端可能已断开): {e}");
                            }
                            Ok(())
                        }
                    };
                }
                let msg = err.message().to_string();
                return match mode {
                    // SSE 中途错误:协议无标准错误帧,error json + DONE 收口
                    ReplyMode::Stream => {
                        // 客户端可能已断开(写失败属正常),但不许静默(律 #1)
                        if let Err(e) = http::sse_data(w, &error_json(&msg).to_string()).await {
                            owl_shared::log_info!("chat", "流式错误帧写失败(客户端可能已断开): {e}");
                        }
                        if let Err(e) = http::sse_data(w, "[DONE]").await {
                            owl_shared::log_info!("chat", "流式收尾帧写失败(客户端可能已断开): {e}");
                        }
                        // 错误流也要收口(chunked 终止块;失败 = 客户端已走)
                        if let Err(e) = http::sse_end(w).await {
                            owl_shared::log_info!("chat", "流式终止块写失败(客户端可能已断开): {e}");
                        }
                        Ok(())
                    }
                    ReplyMode::Aggregate => http::respond_json(w, 500, &error_json(&msg)).await,
                };
            }
            Some(StreamEvent::Turn(_)) => {}
            // 协议不变式:首笔之后不再有 Accepted/Rejected(与断流同处置)
            Some(StreamEvent::Accepted { .. }) | Some(StreamEvent::Rejected { .. })
                if mode == ReplyMode::Stream =>
            {
                led.fail("协议序错乱:受理帧之后再现受理/拒绝帧");
                break
            }
            Some(StreamEvent::Accepted { .. }) | Some(StreamEvent::Rejected { .. }) => {
                led.fail("引擎连接中断(受理帧之后)");
                return http::respond_json(w, 503, &error_json("引擎连接中断")).await;
            }
            None if mode == ReplyMode::Stream => {
                led.fail("客户端断开(流式;引擎 turn 由 executor 侧弃置)");
                break
            }
            None => {
                led.fail("引擎连接中断(通道关闭)");
                return http::respond_json(w, 503, &error_json("引擎连接中断")).await;
            }
        }
    }
    // 环外到达 = 流式被 break(协议序错乱 / 客户端断开):补终止块,让复用连接
    // 上的下一个请求有干净帧界(写失败 = 客户端已走,记一条但不改结局)
    if mode == ReplyMode::Stream {
        if let Err(e) = http::sse_end(w).await {
            owl_shared::log_debug!("chat", "SSE 终止块写失败(客户端可能已断开): {e}");
        }
    }
    Ok(())
}

/// **429 错误体**（OpenAI 错误 schema：`type = rate_limit_error` + 稳定 `code`）。
/// 准入拒绝用它 —— 客户端按 `Retry-After` 退避，不需要任何私有字段。
pub fn error_json_rate_limit(code: &str, msg: &str) -> Value {
    json!({
        "error": {
            "message": msg,
            "type": "rate_limit_error",
            "param": Value::Null,
            "code": code,
        }
    })
}

/// **排队超时 → 429**（`[agent] queue_wait_ms`;D2 的 R 档）。
///
/// 非流式 = `429 + Retry-After`;流式头已发 ⇒ 既有错误帧收口（不自造事件）。
async fn respond_queue_timeout<W: tokio::io::AsyncWrite + Unpin>(
    w: &mut W,
    mode: ReplyMode,
) -> std::io::Result<()> {
    let msg = "排队超时:服务端容量不足 —— 请按 Retry-After 退避重试";
    let body = error_json_rate_limit("queue_timeout", msg);
    match mode {
        ReplyMode::Aggregate => {
            http::respond_json_hdr(w, 429, &body, &[("Retry-After", "5")]).await
        }
        ReplyMode::Stream => {
            if let Err(e) = http::sse_data(w, &body.to_string()).await {
                owl_shared::log_info!("chat", "排队超时帧写失败(客户端可能已断开): {e}");
            }
            if let Err(e) = http::sse_data(w, "[DONE]").await {
                owl_shared::log_info!("chat", "排队超时收尾帧写失败(客户端可能已断开): {e}");
            }
            http::sse_end(w).await
        }
    }
}

/// **退避秒数**（引擎结构化拒绝的 `retry_in_ms` → `Retry-After`）。
///
/// 口径与旧版一致：`ceil(ms/1000)` 夹在 1..=30（客户端不该被估成分钟级）；
/// 引擎没给估计 = 5s 保守值。**2026-11-16 P3**：输入从错误串改为结构化字段
/// （`AdmissionReject`），服务面不再解析人读文本。
pub fn retry_after_secs(reject: &owl_engine::turn::AdmissionReject) -> u64 {
    reject
        .retry_in_ms()
        .map(|ms| ms.div_ceil(1000).clamp(1, 30))
        .unwrap_or(5)
}

/// 客户端写失败 ⇒ 请求引擎弃置**该 turn**(AGT-10;B5.1 起必须指名)。
///
/// 语义要点(合批档):
/// - `turn` 已知(Accepted 之后)= 精确取消该 turn —— **不许**走"弃当前执行 turn"
///   (那会杀掉同批别家会话的 turn);
/// - `turn` 未知(断在受理帧之前)= 按会话找（P4 后同会话可能有排队 turn，
///   引擎按会话取消最早的那个）；
/// - **等 executor 回执再返回**:让取消在步边界尽快落地（P4 后不再是闸的正确性
///   依赖）。回执超时只告警(不静默)。
async fn abort_turn(tx: &EngineClient, session: Option<u64>, turn: Option<u64>) {
    // 临时会话(None)= 无键可寻;引擎会在 sink drop 后自弃(executor 下一个投递点)
    let Some(session) = session else { return };
    let cancel = tx.cancel(session, turn);
    match tokio::time::timeout(ABORT_ACK_TIMEOUT, cancel).await {
        Ok(Ok(())) | Ok(Err(_)) => {}
        Err(_) => owl_shared::log_warn!(
            "chat",
            "abort 回执超时({ABORT_ACK_TIMEOUT:?};引擎可能正忙) "
        ),
    }
}

/// 断流/取消收尾的**唯一路径**：关 sink → 指名取消 → 等回执。
///
/// 次序不可互换（2026-10-20 真机定谳）：本任务若还持有 `sink_rx` 等 abort 回执，
/// 引擎的投递就一直成功 ⇒ 它**察觉不到**断流，会把整个 turn 跑完（实测：256 tok
/// 的流断掉后仍烧满 5s+）。`close()` 让引擎下一笔投递失败 ⇒ 走既有「断流 ⇒ 弃置」
/// （≤1 采样步），再叠加一次指名取消。
async fn close_and_abort(
    sink: &mut mpsc::Receiver<StreamEvent>,
    tx: &EngineClient,
    session: Option<u64>,
    turn: Option<(u64, u64)>,
) {
    sink.close();
    abort_turn(tx, session, turn.map(|(_, t)| t)).await;
}

/// 三通道产出 → SSE 帧(流式)或聚合器(非流式);role 首帧在此处懒发
// 结构性豁免(2026-11-02 clippy 清理):派发/契约 ABI:参数表就是调用面,拆结构体会把每个调用点变成二次装箱
#[allow(clippy::too_many_arguments)]
/// **等对端关闭(FIN)**:`peek` 不消费任何字节。
///
/// 判据:
/// - `Ok(0)` ⇒ EOF(对端发了 FIN)⇒ 本 future 返回 ⇒ 断流;
/// - `Ok(n>0)` ⇒ 有数据(HTTP 管线化的下一个请求 / 非法提前写)⇒ **永久挂起**:
///   本请求内不再用这条路探测(数据必须留给下一轮 `read_request` 消费);
/// - `Err(_)` ⇒ 连接已坏 ⇒ 同样算断流。
async fn peer_fin(s: &tokio::net::TcpStream) {
    let mut b = [0u8; 1];
    loop {
        match s.peek(&mut b).await {
            Ok(0) | Err(_) => return,
            Ok(_) => std::future::pending::<()>().await,
        }
    }
}

/// `select!` 分支用的"到点唤醒":`None` ⇒ 永不就绪(`select!` 的分支表达式
/// 即使被 `if` 关掉也会被**构造**,所以不能用 `Option` 直接 await)。
async fn sleep_until_opt(at: Option<tokio::time::Instant>) {
    match at {
        Some(t) => tokio::time::sleep_until(t).await,
        None => std::future::pending().await,
    }
}

// 结构性豁免(2026-11-03 clippy 收口):SSE 帧的**全部**上下文(写口/模式/会话 id/
// 模型名/聚合账/role 首帧位/协议错位/待发项)都是逐帧必需;打包成 struct 只会在
// 调用点重复同一组字段名(宏 `emit!` 的定义点还要再抖一遍)。
#[allow(clippy::too_many_arguments)]
async fn emit_items<W: tokio::io::AsyncWrite + Unpin>(
    w: &mut W,
    mode: ReplyMode,
    id: &str,
    model: &str,
    acc: &mut Agg,
    role_sent: &mut bool,
    protocol_error: &mut Option<&'static str>,
    items: Vec<Item>,
) -> std::io::Result<()> {
    if protocol_error.is_some() {
        return Ok(());
    }
    for it in items {
        if protocol_error.is_some() {
            break;
        }
        match it {
            Item::Delta { chan, text } => {
                if mode == ReplyMode::Stream {
                    if !*role_sent {
                        *role_sent = true;
                        let first =
                            chunk_json(id, model, json!({"role": "assistant", "content": ""}), None);
                        http::sse_data(w, &first.to_string()).await?;
                    }
                    let delta = match chan {
                        Chan::Think => json!({"reasoning_content": text}),
                        Chan::Text => json!({"content": text}),
                    };
                    let chunk = chunk_json(id, model, delta, None);
                    http::sse_data(w, &chunk.to_string()).await?;
                } else {
                    match chan {
                        Chan::Think => acc.reason.push_str(&text),
                        Chan::Text => acc.text.push_str(&text),
                    }
                }
            }
            Item::ToolCall { index, name, arguments } => {
                let cid = acc.tool_id(index);
                if mode == ReplyMode::Stream {
                    if !*role_sent {
                        *role_sent = true;
                        let first =
                            chunk_json(id, model, json!({"role": "assistant", "content": ""}), None);
                        http::sse_data(w, &first.to_string()).await?;
                    }
                    // pi 按 `index` 归并工具块(id 只作标识);一次性给全 name+arguments
                    // —— arguments 恒为合法 JSON 字符串(§3.3 约束 1)
                    let delta = json!({"tool_calls": [{
                        "index": index,
                        "id": cid,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }]});
                    let chunk = chunk_json(id, model, delta, None);
                    http::sse_data(w, &chunk.to_string()).await?;
                } else {
                    acc.tools.push((cid, name, arguments));
                }
            }
            // 畸形工具块:不发射工具调用(§3.3 约束 4),但**不静默**
            Item::Malformed { kind, raw } => {
                if protocol_error.is_none() {
                    *protocol_error = Some(kind);
                }
                owl_shared::log_warn!(
                    "chat",
                    "工具块畸形({kind};{}B) ⇒ 拒绝正常返回:{}",
                    raw.len(),
                    raw_for_log(&raw)
                );
            }
        }
    }
    Ok(())
}

// ============================================================================
// §3 `/tokenize`(G5③;契约见《设计》§3.7)
// ============================================================================

/// `/tokenize` 请求体(**最小面**)。
///
/// 与 `/v1/chat/completions` 不同,本端点 `deny_unknown_fields` **开着**:
/// 它是 owl 自有面(无 pi 兼容史要背),多传字段 = 调用方预期落空 ⇒ 响亮拒。
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TokenizeRequest {
    /// 回显用(不校验与本进程模型名一致 —— 单模型服务)
    model: Option<String>,
    /// 待编码文本(**必填**;本端点不做 messages 渲染)
    prompt: Option<String>,
    /// `raw`(缺省)= 纯文本编码;`chat` = 先套家族单轮包装
    mode: Option<String>,
    /// vLLM 同名字段:**只接 false/缺省** —— true = 响亮拒
    /// (本引擎编码恒 `add_special_tokens=false`;要家族包装用 `mode="chat"`)
    add_special_tokens: Option<bool>,
}

/// 请求 → `(text, chat_wrap, mode)` 的**纯校验**(不碰网络/引擎;可单测)
fn tokenize_plan(req: &TokenizeRequest) -> Result<(String, bool, &'static str), String> {
    let text = match req.prompt.as_deref() {
        Some(t) if !t.is_empty() => t.to_string(),
        _ => {
            return Err(
                "prompt 必填且非空(messages 渲染未做 —— 需要时先自己渲染再送本端点)".into(),
            )
        }
    };
    let mode = req.mode.as_deref().unwrap_or("raw");
    let (chat_wrap, mode_canon) = match mode {
        "raw" => (false, "raw"),
        // 家族单轮包装(与裸文本提交同一律):问“这轮用户内容实际花多少 token”
        "chat" => (true, "chat"),
        other => return Err(format!("mode 只接 raw|chat,得到 {other:?}")),
    };
    if req.add_special_tokens == Some(true) {
        return Err(
            "add_special_tokens=true 未支持(本引擎编码恒不加特殊 token;要家族单轮包装用 mode=\"chat\")"
                .into(),
        );
    }
    Ok((text, chat_wrap, mode_canon))
}

/// `POST /tokenize`:校验 → 旁路引擎 tokenize → 回兼容形状的 JSON。
///
/// ⚠️ **延迟说明**:引擎 executor 单线程且“收件与泵不相交” ⇒ 本请求排在当前
/// turn 之后(生成期间调用会等到本轮结束)。毫秒级 CPU 工,单源真理优先。
pub(crate) async fn tokenize<W: tokio::io::AsyncWrite + Unpin>(
    w: &mut W,
    body: &[u8],
    tx: EngineClient,
) -> std::io::Result<()> {
    let req: TokenizeRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => return http::respond_json(w, 400, &error_json(format!("请求体非法: {e}"))).await,
    };
    let (text, chat_wrap, mode) = match tokenize_plan(&req) {
        Ok(v) => v,
        Err(e) => return http::respond_json(w, 400, &error_json(e)).await,
    };
    match tx.tokenize(text, chat_wrap).await {
        Ok(rep) => {
            http::respond_json(
                w,
                200,
                &json!({
                    "count": rep.tokens.len(),
                    "max_model_len": rep.max_seq_tokens,
                    "tokens": rep.tokens,
                    "mode": mode,
                    "model": req.model,
                }),
            )
            .await
        }
        // 编码失败(显式上抛;不混淆“文本是空的”)
        Err(e) => http::respond_json(w, 400, &error_json(e)).await,
    }
}

/// 非流式 message(三通道聚合;pi 的 AssistantMessage 同形)
fn agg_message(acc: &Agg, tool_calls: usize) -> Value {
    let mut msg = json!({"role": "assistant", "content": acc.text});
    if !acc.reason.is_empty() {
        msg["reasoning_content"] = json!(acc.reason);
    }
    if tool_calls > 0 {
        let tcs: Vec<Value> = acc
            .tools
            .iter()
            .map(|(id, name, args)| {
                json!({"id": id, "type": "function",
                       "function": {"name": name, "arguments": args}})
            })
            .collect();
        msg["tool_calls"] = json!(tcs);
    }
    msg
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(msgs: Value) -> ChatRequest {
        serde_json::from_value(json!({"messages": msgs})).expect("parse")
    }

    /// **可删标记只认布尔**(前缀无感删除):非布尔 = 400,不许静默当真/当假
    #[test]
    fn owl_evict_must_be_bool() {
        let base = json!({"role":"user","content":[
            {"type":"text","text":"hi"},
            {"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}
        ]});
        let plain = req(json!([base.clone()]));
        assert!(plain.validate().is_ok(), "无标记应通过");
        let mut with_flag = base.clone();
        with_flag["content"][1]["owl_evict"] = json!(true);
        let flagged = req(json!([with_flag]));
        assert!(flagged.validate().is_ok(), "布尔标记应通过");
        let mut bad = base.clone();
        bad["content"][1]["owl_evict"] = json!("yes");
        let bad = req(json!([bad]));
        let e = bad.validate().expect_err("非布尔必须结构化拒");
        assert!(e.contains("owl_evict"), "错误必须点名字段(得 {e})");
    }

    /// vLLM 同款:字符串 arguments → 对象;dict/array 原样;空串 → {}
    #[test]
    fn normalize_string_arguments_to_object() {
        let mut r = req(json!([{"role":"assistant","content":"","tool_calls":[
            {"id":"c1","type":"function","function":{"name":"bash","arguments":"{\"command\":\"ls\"}"}},
            {"id":"c2","type":"function","function":{"name":"f2","arguments":""}},
            {"id":"c3","type":"function","function":{"name":"f3","arguments":{"a":1}}}
        ]}]));
        r.validate().unwrap();
        r.normalize_tool_calls().unwrap();
        assert_eq!(r.messages[0]["tool_calls"][0]["function"]["arguments"]["command"], "ls");
        assert_eq!(r.messages[0]["tool_calls"][1]["function"]["arguments"], json!({}));
        assert_eq!(r.messages[0]["tool_calls"][2]["function"]["arguments"], json!({"a": 1}));
    }

    /// 空 tool_calls 数组剔除(模板回普通 assistant 路径)
    #[test]
    fn normalize_drops_empty_tool_calls() {
        let mut r = req(json!([{"role":"assistant","content":"hi","tool_calls":[]}]));
        r.validate().unwrap();
        r.normalize_tool_calls().unwrap();
        assert!(r.messages[0].get("tool_calls").is_none(), "空数组应被剔除");
    }

    /// 坏 JSON / 非 function 类型 → 结构化拒绝(不静默)
    #[test]
    fn normalize_rejects_bad_shapes() {
        let r = req(json!([{"role":"assistant","content":"","tool_calls":[
            {"id":"c1","type":"function","function":{"name":"f","arguments":"{oops"}}]}]));
        assert!(r.validate().is_err(), "坏 JSON 应被 validate 拒");

        let mut r = req(json!([{"role":"assistant","content":"","tool_calls":[
            {"id":"c1","type":"custom","function":{"name":"f","arguments":{}}}]}]));
        assert!(r.normalize_tool_calls().is_err(), "非 function 类型应被拒");
    }

    /// 采样参数/未知字段 → 告警条目(AGT-06 后温度/停止串已真生效;
    /// 告警只余**未知字段**,不许静默丢弃)
    #[test]
    fn warnings_surface_only_unknown_fields() {
        let r: ChatRequest = serde_json::from_value(json!({
            "messages":[{"role":"user","content":"hi"}],
            "temperature": 0.7, "stop": ["x"], "seed": 1})).expect("parse");
        r.validate().unwrap();
        assert!(r.warnings().is_empty(), "已生效参数不该再告警:{:?}", r.warnings());
        let r: ChatRequest = serde_json::from_value(json!({
            "messages":[{"role":"user","content":"hi"}],
            "presence_penalty": 0.5})).expect("parse");
        assert_eq!(r.warnings().len(), 1, "未知字段必须可见");
    }

    /// AGT-06:采样覆写只带请求出现的字段(未带 = 继承引擎档位)
    #[test]
    fn sampler_override_partial() {
        let r: ChatRequest = serde_json::from_value(json!({
            "messages":[{"role":"user","content":"hi"}]})).expect("parse");
        assert!(r.sampler_override().is_none(), "未带采样参数 ⇒ 无覆写");
        let r: ChatRequest = serde_json::from_value(json!({
            "messages":[{"role":"user","content":"hi"}], "temperature": 0.0})).expect("parse");
        let o = r.sampler_override().expect("有覆写");
        assert_eq!(o.temp, Some(0.0));
        assert_eq!(o.topk, None);
        assert_eq!(o.topp, None);
    }

    /// AGT-06:stop 两形态(string / array)+ 非法形态结构化拒
    #[test]
    fn stop_strings_shapes() {
        let r: ChatRequest = serde_json::from_value(json!({
            "messages":[{"role":"user","content":"hi"}], "stop": "\n\n"})).expect("parse");
        assert_eq!(r.stop_strings().unwrap(), vec!["\n\n".to_string()]);
        let r: ChatRequest = serde_json::from_value(json!({
            "messages":[{"role":"user","content":"hi"}], "stop": ["a", "b"]})).expect("parse");
        assert_eq!(r.stop_strings().unwrap(), vec!["a".to_string(), "b".to_string()]);
        let r: ChatRequest = serde_json::from_value(json!({
            "messages":[{"role":"user","content":"hi"}], "stop": [1, 2]})).expect("parse");
        assert!(r.stop_strings().is_err(), "非字符串元素应被拒");
        let r: ChatRequest = serde_json::from_value(json!({
            "messages":[{"role":"user","content":"hi"}], "stop": 7})).expect("parse");
        assert!(r.stop_strings().is_err(), "非 string/array 应被拒");
    }

    /// max_tokens 夹紧(上限来自 [agent]) + enable_thinking 取值
    #[test]
    fn max_new_clamp_and_thinking_flag() {
        let agent = AgentCfg::default();
        let r: ChatRequest = serde_json::from_value(json!({
            "messages":[{"role":"user","content":"x"}], "max_tokens": 32000})).expect("parse");
        assert_eq!(r.max_new(&agent), (agent.max_new_cap, true));
        let r: ChatRequest = serde_json::from_value(json!({
            "messages":[{"role":"user","content":"x"}]})).expect("parse");
        assert_eq!(r.max_new(&agent), (agent.default_max_new, false));
        // `defaults` = `chat_template.toml`（模板变量；空表 = 无文件）
        let none = serde_json::Map::new();
        let r: ChatRequest = serde_json::from_value(json!({
            "messages":[{"role":"user","content":"x"}],
            "chat_template_kwargs": {"enable_thinking": false}})).expect("parse");
        assert!(!r.enable_thinking(&none, false));
        // `reasoning_effort=none/off` 与模板同口径：splitter 也必须按关思考启动
        for off in ["none", "off", "NONE", "Off"] {
            let r: ChatRequest = serde_json::from_value(json!({
                "messages":[{"role":"user","content":"x"}],
                "reasoning_effort": off})).expect("parse");
            assert!(!r.enable_thinking(&none, false), "reasoning_effort={off} 应关思考");
        }
        let r: ChatRequest = serde_json::from_value(json!({
            "messages":[{"role":"user","content":"x"}],
            "reasoning_effort": "high"})).expect("parse");
        assert!(r.enable_thinking(&none, false), "reasoning_effort=high 应开思考");
        // 显式 enable_thinking 覆盖 reasoning_effort
        let r: ChatRequest = serde_json::from_value(json!({
            "messages":[{"role":"user","content":"x"}],
            "reasoning_effort": "none",
            "chat_template_kwargs": {"enable_thinking": true}})).expect("parse");
        assert!(r.enable_thinking(&none, false), "显式 enable_thinking=true 应优先");

        // **模板变量文件**（2026-11-16）：文件缺省 < 请求
        let mut defaults = serde_json::Map::new();
        defaults.insert("auto_disable_thinking_with_tools".into(), json!(true));
        defaults.insert("reasoning_effort".into(), json!("none"));
        let r: ChatRequest = serde_json::from_value(json!({
            "messages":[{"role":"user","content":"x"}], "tools": [{"type":"function","function":{"name":"f"}}]
        })).expect("parse");
        assert!(
            !r.enable_thinking(&defaults, true),
            "文件里的 reasoning_effort=none / auto_disable+tools 都要落到 splitter"
        );
        let r: ChatRequest = serde_json::from_value(json!({
            "messages":[{"role":"user","content":"x"}],
            "reasoning_effort": "high"})).expect("parse");
        assert!(
            r.enable_thinking(&defaults, false),
            "请求档位覆盖文件缺省"
        );

        // vLLM 同款归一化：reasoning_effort 必须同步派生 enable_thinking
        let r: ChatRequest = serde_json::from_value(json!({
            "messages":[{"role":"user","content":"x"}],
            "reasoning_effort": "none"})).expect("parse");
        let kw = r.effective_template_kwargs(&none);
        assert_eq!(kw.get("reasoning_effort"), Some(&json!("none")));
        assert_eq!(kw.get("enable_thinking"), Some(&json!(false)));
        let r: ChatRequest = serde_json::from_value(json!({
            "messages":[{"role":"user","content":"x"}],
            "reasoning_effort": "high"})).expect("parse");
        assert_eq!(
            r.effective_template_kwargs(&none).get("enable_thinking"),
            Some(&json!(true))
        );
        // 显式 enable_thinking 不被派生值覆盖(与 vLLM build_chat_params 一致)
        let r: ChatRequest = serde_json::from_value(json!({
            "messages":[{"role":"user","content":"x"}],
            "reasoning_effort": "none",
            "chat_template_kwargs": {"enable_thinking": true}})).expect("parse");
        assert_eq!(
            r.effective_template_kwargs(&none).get("enable_thinking"),
            Some(&json!(true))
        );
        // 文件缺省的 merge：请求 kwargs 覆盖同名键
        let mut file_high = serde_json::Map::new();
        file_high.insert("reasoning_effort".into(), json!("high"));
        assert_eq!(
            r.effective_template_kwargs(&file_high).get("reasoning_effort"),
            Some(&json!("none")),
            "请求里的 reasoning_effort 覆盖文件值"
        );
        assert_eq!(
            r.effective_template_kwargs(&file_high).get("enable_thinking"),
            Some(&json!(true)),
            "请求显式 enable_thinking 覆盖文件"
        );

        // [agent] 定制上限生效(配置面唯一旋钮)
        let agent = AgentCfg { max_new_cap: 32768, default_max_new: 2048, ..AgentCfg::default() };
        let r: ChatRequest = serde_json::from_value(json!({
            "messages":[{"role":"user","content":"x"}], "max_tokens": 40000})).expect("parse");
        assert_eq!(r.max_new(&agent), (32768, true));
        let r: ChatRequest = serde_json::from_value(json!({
            "messages":[{"role":"user","content":"x"}]})).expect("parse");
        assert_eq!(r.max_new(&agent), (2048, false));
    }

    /// **上下文溢出归一化**(wire 契约;设计 §3.6 挂账项)。
    ///
    /// pi 的 `isContextOverflow()` **只看报文文本**与它的 `OVERFLOW_PATTERNS`
    /// 做正则匹配(两轮门 `explicitOverflow`/`assistantUsageMatchesProjection`
    /// 都只要报文命中)⇒ 引擎的中文"超预算"必须带上词表里的规范短语,否则
    /// pi 把它当**普通 400** 原样展示、**不触发自动 compaction**。
    /// 词表 = pi-ai bundle 实测(2026-10-15;改短语就同步改这里 + 复核 bundle)。
    #[test]
    fn budget_reject_is_normalized_for_pi_compaction() {
        const PI_OVERFLOW_PATTERNS: &[&str] = &[
            "prompt (is )?too long",
            "prompt exceeds max length",
            "request_too_large",
            "input is too long for requested model",
            "exceeds the context window",
            "exceeds .* maximum context length",
        ];
        // ① 短语必须**逐字**在词表里
        assert!(
            PI_OVERFLOW_PATTERNS.contains(&OVERFLOW_PHRASE),
            "OVERFLOW_PHRASE {OVERFLOW_PHRASE:?} 不在 pi 词表里"
        );
        // ② 引擎原诊断保留 + code 落位(绝不静默改写)
        let raw = "submit: prompt 27584 + 生成 8192 超预算 32768";
        let body = error_json_for_reject(raw.to_string());
        let msg = body["error"]["message"].as_str().expect("message");
        assert!(msg.starts_with(raw), "原诊断须保留: {msg}");
        assert!(msg.contains(OVERFLOW_PHRASE), "{msg}");
        assert_eq!(body["error"]["code"], "context_length_exceeded");
        // ③ 非预算拒绝不动(无 code、报文原样)
        let plain = error_json_for_reject("messages 为空".into());
        assert!(plain["error"].get("code").is_none());
        assert_eq!(plain["error"]["message"], "messages 为空");
    }

    /// **G5③ `/tokenize` 纯校验面**:两模式 + 三类响亮拒(不静默回缺省)
    #[test]
    fn tokenize_plan_modes_and_rejects() {
        let plan = |j: &str| {
            let req: TokenizeRequest = serde_json::from_str(j).expect("请求体应可解析");
            tokenize_plan(&req)
        };
        assert_eq!(plan(r#"{"prompt":"你好"}"#).unwrap(), ("你好".into(), false, "raw"));
        assert_eq!(
            plan(r#"{"prompt":"你好","mode":"chat"}"#).unwrap(),
            ("你好".into(), true, "chat")
        );
        assert!(plan(r#"{"prompt":"你好","add_special_tokens":false}"#).is_ok(), "false 可接");
        assert!(plan(r#"{"prompt":""}"#).unwrap_err().contains("prompt 必填"), "空 prompt 拒");
        assert!(plan(r#"{"mode":"chat"}"#).unwrap_err().contains("prompt 必填"), "缺 prompt 拒");
        assert!(plan(r#"{"prompt":"x","mode":"tokens"}"#).unwrap_err().contains("raw|chat"));
        assert!(
            plan(r#"{"prompt":"x","add_special_tokens":true}"#)
                .unwrap_err()
                .contains("add_special_tokens=true 未支持")
        );
        // 未知字段:本端点开着 deny_unknown_fields(自有面,不背兼容史)
        assert!(serde_json::from_str::<TokenizeRequest>(r#"{"prompt":"x","nope":1}"#).is_err());
    }
}


#[cfg(test)]
mod admission_tests {
    use super::retry_after_secs;
    use owl_engine::turn::AdmissionReject;

    /// **协议面只认结构**（2026-11-16 P3）：退避秒数来自 `retry_in_ms`，
    /// 与错误串的写法无关 —— 以前解析 `admission:<code> retry_ms=<n>`，改一个
    /// 字就静默改变 HTTP 语义。
    #[test]
    fn retry_after_comes_from_structured_estimate() {
        let r = |ms| AdmissionReject::new("no_memory", ms, "在飞 8 已达上限 8");
        assert_eq!(retry_after_secs(&r(Some(12_000))), 12);
        assert_eq!(retry_after_secs(&r(Some(200))), 1, "不足 1s 夹到 1");
        assert_eq!(retry_after_secs(&r(Some(999_999))), 30, "封顶 30");
        assert_eq!(retry_after_secs(&r(None)), 5, "无建议值 = 5s 保守");
    }

    /// 结构化拒绝的显示串保持旧口径（日志/既有断言不动），服务面不再解析它。
    #[test]
    fn display_keeps_the_legacy_wire_for_logs() {
        assert_eq!(
            AdmissionReject::new("no_slot", Some(137), "在飞 8 已达上限 8").to_string(),
            "admission:no_slot retry_ms=137 —— 在飞 8 已达上限 8"
        );
        assert_eq!(
            AdmissionReject::new("no_memory", None, "工作集不够").to_string(),
            "admission:no_memory 工作集不够"
        );
        assert_eq!(
            AdmissionReject::new("queue_timeout", Some(900), "").to_string(),
            "admission:queue_timeout retry_ms=900"
        );
    }

    /// `TurnError` 只在**产生点**决定"是不是准入拒绝"；显示串同源。
    #[test]
    fn turn_error_carries_reject_only_when_structured() {
        let plain = owl_engine::turn::TurnError::new("submit: 空 prompt");
        assert!(plain.reject().is_none());
        assert_eq!(plain.to_string(), "submit: 空 prompt");
        let rejected = owl_engine::turn::TurnError::rejected(AdmissionReject::new(
            "no_memory",
            Some(1000),
            "池满",
        ));
        assert_eq!(rejected.reject().map(|r| r.code()), Some("no_memory"));
        assert!(rejected.to_string().starts_with("admission:no_memory"));
    }
}

#[cfg(test)]
mod multipart_text_tests {
    use super::ChatRequest;
    use serde_json::{json, Value};

    fn req(messages: Value) -> ChatRequest {
        serde_json::from_value(json!({"messages": messages})).expect("请求解析")
    }

    /// 最小影响面:plain string / 单 text 数组 / system / assistant 一律不动;
    /// 带图多段只给 text 补 `\n`,image 顺序与 owl_evict 原样;幂等。
    #[test]
    fn multipart_text_normalization_is_minimal_and_idempotent() {
        let mut r = req(json!([
            {"role": "system", "content": [{"type": "text", "text": "sys"}]},
            {"role": "user", "content": "plain string"},
            {"role": "user", "content": [{"type": "text", "text": "single"}]},
            {"role": "user", "content": [
                {"type": "text", "text": "a"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}, "owl_evict": true},
                {"type": "text", "text": "b\n"}
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "asst"}]}
        ]));
        let changed = r.normalize_multipart_text();
        assert_eq!(changed, 1, "只有 a 需要补;b 已带换行");
        assert_eq!(r.messages[0]["content"][0]["text"], "sys", "system 不动");
        assert_eq!(r.messages[1]["content"], "plain string", "string 逐字节不动");
        assert_eq!(r.messages[2]["content"][0]["text"], "single", "单 text 数组不动");
        assert_eq!(r.messages[3]["content"][0]["text"], "a\n");
        assert_eq!(r.messages[3]["content"][2]["text"], "b\n");
        assert_eq!(r.messages[3]["content"][1]["owl_evict"], true, "标记保留");
        assert_eq!(
            r.messages[3]["content"][1]["image_url"]["url"],
            "data:image/png;base64,AA",
            "image 原样"
        );
        assert_eq!(r.messages[4]["content"][0]["text"], "asst", "assistant 不动");
        let snapshot = r.messages.clone();
        assert_eq!(r.normalize_multipart_text(), 0, "幂等");
        assert_eq!(snapshot, r.messages, "第二次调用零变化");
    }

    /// 删图后的 user 消息形状:text1 + text2 相邻(无图)也必须补换行,
    /// 否则引擎的删点前缀/后缀无法对齐。
    #[test]
    fn adjacent_user_texts_after_image_removal_get_newlines() {
        let mut r = req(json!([
            {"role": "user", "content": [
                {"type": "text", "text": "before"},
                {"type": "text", "text": "after"}
            ]}
        ]));
        assert_eq!(r.normalize_multipart_text(), 2);
        assert_eq!(r.messages[0]["content"][0]["text"], "before\n");
        assert_eq!(r.messages[0]["content"][1]["text"], "after\n");
    }

    /// 纯文本多段 content 的规范化是确定性的:同输入两次渲染结果一致。
    #[test]
    fn normalization_is_deterministic_for_cache() {
        let msg = json!([{"role": "user", "content": [
            {"type": "text", "text": "x"},
            {"type": "text", "text": "y"}
        ]}]);
        let mut a = req(msg.clone());
        let mut b = req(msg);
        a.normalize_multipart_text();
        b.normalize_multipart_text();
        assert_eq!(a.messages, b.messages);
    }
}

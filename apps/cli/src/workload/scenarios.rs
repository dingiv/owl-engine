//! One scenario per feature family. Assertions are structural (tools, files,
//! usage, engine counters) or simple colour/fact checks — never model prose.

use serde_json::{json, Value};

use crate::client::{ChatReply, Client};
use crate::workload::png::{data_url_png, BLUE, CYAN, GREEN, PURPLE, RED, YELLOW};
use crate::workload::report::{ScenarioReport, TurnRecord};
use crate::workload::{gui, image_count, remove_oldest_image, report, Args, Ctx, Workspace};

const SYS_TEXT: &str =
    "You are a careful assistant. Answer concisely. Use the conversation history; never invent facts.";
const SYS_TOOLS: &str = "You are a coding agent. You have tools: read, write, edit, bash. \
You MUST use the provided tools for file operations; never invent file contents. \
When asked to create a file, call write. When asked to verify, call read or bash. \
Keep answers short and only answer after the tool results.";
const SYS_VISION: &str = "You are a vision-capable assistant. Describe images accurately. \
When asked for a colour, answer with one lowercase English colour word.";
const SYS_MIXED: &str = "You are a coding agent with vision. Use the provided tools for file \
operations; never invent file contents. Describe the image accurately. Keep answers short.";

const IMG_W: u32 = 256;
const IMG_H: u32 = 256;

fn session(pid: u32, tag: &str) -> String {
    format!("wl-{tag}-{pid}")
}

fn ensure(cond: bool, msg: impl Into<String>) -> Result<(), String> {
    if cond { Ok(()) } else { Err(msg.into()) }
}

fn lower(s: &str) -> String {
    s.to_lowercase()
}

fn has(r: &ChatReply, needle: &str) -> bool {
    lower(r.text()).contains(&lower(needle))
}

fn nonce(pid: u32) -> String {
    let ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    format!("{pid}-{ms}")
}

fn image_part(url: &str, evict: bool) -> Value {
    let mut part = json!({"type": "image_url", "image_url": {"url": url}});
    if evict {
        part["owl_evict"] = Value::Bool(true);
    }
    part
}

fn remove_image(history: &mut [Value], url: &str) -> bool {
    for msg in history.iter_mut() {
        let Some(arr) = msg.get_mut("content").and_then(Value::as_array_mut) else {
            continue;
        };
        if let Some(pos) = arr
            .iter()
            .position(|p| p["image_url"]["url"].as_str() == Some(url))
        {
            arr.remove(pos);
            return true;
        }
    }
    false
}

fn tool_names(ctx: &Ctx<'_>) -> Vec<String> {
    ctx.turns.iter().flat_map(|t| t.tools.clone()).collect()
}

const SYS_GUI: &str = "You are a GUI agent controlling a desktop. You have exactly these tools: \
screenshot, click, type, key. Every tool call returns the current screen as an image. \
The harness executes every tool call, attaches the resulting screen, and keeps only the three \
newest screenshots in the conversation; older images are removed automatically. \
A single assistant response may contain several tool calls. The user does not act between them; \
the agent loop executes all calls and sends the tool results back to you. \
Follow the user's step list in order and use the attached images.";

// ── 0. real GUI-agent load: tools return screenshots, keep newest three ───
pub(crate) fn gui_agent_screenshots(
    client: &Client,
    args: &Args,
    pid: u32,
    max_tokens: usize,
) -> Result<ScenarioReport, String> {
    let ws = Workspace::new("gui_agent_screenshots", pid, args.keep)?;
    let mut ctx = Ctx::new(
        client,
        "gui_agent_screenshots",
        Some(session(pid, "gui")),
        true,
        max_tokens,
        ws.path(),
    );
    ctx.log_path = args.server_log.as_ref().map(std::path::PathBuf::from);
    // Unique run tag: the prefix tree is content-addressed, so two identical
    // GUI transcripts would share image-bearing blocks and the first delete
    // would be refused as "shared suffix" (refs > 1) instead of absorbed.
    ctx.system(&format!("{SYS_GUI} Run tag: {}.", nonce(pid)));
    ctx.gui = Some(gui::GuiAgent::new());
    ctx.image_limit = Some(3);
    let log_mark = ctx.log_mark();
    let result = (|| -> Result<String, String> {
        let reply = ctx.send(json!(
            "Complete these six steps in order. The user does not act between steps; the \
             agent loop executes every tool call and returns the screen after each action.\n\
             1. screenshot\n\
             2. In one assistant response, emit two separate function calls: \
                click(64, 64) and click(192, 192). Do not split them across responses.\n\
             3. type the text \"owl\" into the focused field\n\
             4. press the ENTER key\n\
             5. screenshot\n\
             6. After the final screenshot, answer with the dominant colour of the latest \
                screenshot as one lowercase English colour word. Do not answer earlier."
        ))?;
        let names = tool_names(&ctx);
        ensure(
            names.iter().filter(|n| *n == "screenshot").count() >= 2,
            format!("截图次数不足: {names:?}"),
        )?;
        ensure(
            names.iter().filter(|n| *n == "click").count() >= 2,
            format!("点击次数不足: {names:?}"),
        )?;
        ensure(names.iter().any(|n| n == "type"), format!("没有 type: {names:?}"))?;
        ensure(names.iter().any(|n| n == "key"), format!("没有 key: {names:?}"))?;
        ensure(
            names.len() >= 6,
            format!("工具调用少于 6 次,不足以触发图片窗口: {names:?}"),
        )?;
        // A single assistant response may return several function calls; the
        // agent loop executes all of them. Do not assert one tool per round.
        let max_images = ctx.turns.iter().map(|t| t.images).max().unwrap_or(0);
        ensure(
            max_images <= 3,
            format!("请求里图片数超过 3: {max_images}(turns={:?})",
                ctx.turns.iter().map(|t| t.images).collect::<Vec<_>>()),
        )?;
        ensure(
            ctx.image_prunes.len() >= 3,
            format!("只删除了 {} 张旧图(期望 ≥3)", ctx.image_prunes.len()),
        )?;
        ensure(
            image_count(&ctx.history) == 3,
            format!("结束时历史图片数 = {}(期望 3)", image_count(&ctx.history)),
        )?;
        let log = ctx.log_since(log_mark);
        if ctx.log_path.is_some() {
            let absorbed = log.matches("前缀无感删除").count();
            ensure(
                absorbed >= 3,
                format!(
                    "服务端只吸收 {absorbed} 次旧图删除(期望 ≥3);img-evict={:?}",
                    log.lines().filter(|l| l.contains("img-evict")).collect::<Vec<_>>()
                ),
            )?;
        }
        // 最后三轮都在首次 prune 之后;不能把 t4(刚好 3 图、尚未删图)
        // 的 cached 当成删除吸收的证据。
        let warm = ctx
            .turns
            .iter()
            .rev()
            .take(3)
            .map(|t| t.cached_pct())
            .fold(0.0f64, f64::max);
        ensure(
            warm >= 30.0,
            format!("三图窗口下 cached 最高只有 {warm:.1}%(前缀/删图吸收失败)"),
        )?;
        ensure(has(&reply, "yellow"), format!("最终颜色不是 yellow: {:?}", reply.text()))?;
        let screen = ctx.gui.as_ref().map(gui::GuiAgent::screen).unwrap_or((0, 0, 0));
        ensure(
            screen == crate::workload::png::YELLOW,
            format!("GUI 终态颜色 {screen:?} != yellow"),
        )?;
        Ok(format!(
            "tools={} prunes={} images={} max_images/round={} warm_cached={warm:.1}% final=yellow",
            names.len(),
            ctx.image_prunes.len(),
            image_count(&ctx.history),
            max_images,
        ))
    })();
    Ok(report("gui_agent_screenshots", ctx, result))
}

// ── 1. text multi-turn ────────────────────────────────────────────────────
pub(crate) fn text_multiturn(
    client: &Client,
    args: &Args,
    pid: u32,
    max_tokens: usize,
) -> Result<ScenarioReport, String> {
    let ws = Workspace::new("text_multiturn", pid, args.keep)?;
    let mut ctx = Ctx::new(
        client,
        "text_multiturn",
        Some(session(pid, "text")),
        false,
        max_tokens,
        ws.path(),
    );
    ctx.system(SYS_TEXT);
    let result = (|| -> Result<String, String> {
        // No-image multipart content: exercises the server-side text boundary
        // normalization and proves the next turn still hits the prefix cache.
        ctx.send(json!([
            {"type": "text", "text": "Remember this codeword: BLUE-OTTER-42."},
            {"type": "text", "text": "Reply with OK only."}
        ]))?;
        let r2 = ctx.send(json!(
            "What is the codeword? Answer with the codeword only."
        ))?;
        ensure(has(&r2, "BLUE-OTTER-42"), format!("第二轮丢码字: {:?}", r2.text()))?;
        let r3 = ctx.send(json!(
            "Write one short sentence that contains the codeword and the word owl."
        ))?;
        ensure(has(&r3, "BLUE-OTTER-42"), "第三轮丢码字")?;
        ensure(has(&r3, "owl"), "第三轮没有 owl")?;
        let t2 = &ctx.turns[1];
        let t3 = &ctx.turns[2];
        ensure(t2.cached > 0, format!("第二轮 cached=0(前缀守卫失败): {t2:?}"))?;
        ensure(
            t2.prompt > 0 && t3.prompt > 0,
            "usage.prompt_tokens 未上报",
        )?;
        Ok(format!(
            "2nd cached {:.1}% / 3rd cached {:.1}%",
            t2.cached_pct(),
            t3.cached_pct()
        ))
    })();
    Ok(report("text_multiturn", ctx, result))
}

// ── 2. tool roundtrip: write -> read -> bash ──────────────────────────────
pub(crate) fn tool_roundtrip(
    client: &Client,
    args: &Args,
    pid: u32,
    max_tokens: usize,
) -> Result<ScenarioReport, String> {
    let ws = Workspace::new("tool_roundtrip", pid, args.keep)?;
    let mut ctx = Ctx::new(
        client,
        "tool_roundtrip",
        Some(session(pid, "tools")),
        true,
        max_tokens,
        ws.path(),
    );
    ctx.system(SYS_TOOLS);
    let result = (|| -> Result<String, String> {
        let reply = ctx.send(json!(
            "Create a file named hello.txt in the workspace with exactly 'owl-tool-ok' \
             (no trailing newline). Then read it back with the read tool and run \
             `wc -c hello.txt` with bash. Report the byte count."
        ))?;
        let names = tool_names(&ctx);
        ensure(!names.is_empty(), "模型没有调用任何工具")?;
        ensure(names.iter().any(|n| n == "write"), format!("没有 write 调用: {names:?}"))?;
        ensure(
            names.iter().any(|n| n == "read" || n == "bash"),
            format!("没有 read/bash 验证: {names:?}"),
        )?;
        let path = ws.path().join("hello.txt");
        let content = std::fs::read_to_string(&path).map_err(|e| format!("hello.txt: {e}"))?;
        ensure(
            content == "owl-tool-ok",
            format!("文件内容不符: {content:?}"),
        )?;
        ensure(has(&reply, "11"), format!("最终答复没报 11: {:?}", reply.text()))?;
        Ok(format!("tools={names:?}, bytes=11"))
    })();
    Ok(report("tool_roundtrip", ctx, result))
}

// ── 3. vision chat: two images in one request ────────────────────────────
pub(crate) fn vision_chat(
    client: &Client,
    args: &Args,
    pid: u32,
    max_tokens: usize,
) -> Result<ScenarioReport, String> {
    let ws = Workspace::new("vision_chat", pid, args.keep)?;
    let mut ctx = Ctx::new(
        client,
        "vision_chat",
        Some(session(pid, "vision")),
        false,
        max_tokens,
        ws.path(),
    );
    ctx.system(SYS_VISION);
    let red = data_url_png(IMG_W, IMG_H, RED);
    let blue = data_url_png(IMG_W, IMG_H, BLUE);
    let result = (|| -> Result<String, String> {
        let reply = ctx.send(json!([
            {"type": "text", "text": "Name the dominant colour of each of the two images, in order, separated by a comma. Use one lowercase English colour word per image."},
            image_part(&red, false),
            image_part(&blue, false),
        ]))?;
        ensure(has(&reply, "red"), format!("没识别红色: {:?}", reply.text()))?;
        ensure(has(&reply, "blue"), format!("没识别蓝色: {:?}", reply.text()))?;
        Ok("red + blue".to_string())
    })();
    Ok(report("vision_chat", ctx, result))
}

// ── 4. mixed image + tool call ───────────────────────────────────────────
pub(crate) fn mixed_tool_image(
    client: &Client,
    args: &Args,
    pid: u32,
    max_tokens: usize,
) -> Result<ScenarioReport, String> {
    let ws = Workspace::new("mixed_tool_image", pid, args.keep)?;
    let mut ctx = Ctx::new(
        client,
        "mixed_tool_image",
        Some(session(pid, "mixed")),
        true,
        max_tokens,
        ws.path(),
    );
    ctx.system(SYS_MIXED);
    let red = data_url_png(IMG_W, IMG_H, RED);
    let result = (|| -> Result<String, String> {
        let reply = ctx.send(json!([
            image_part(&red, false),
            {"type": "text", "text": "This is a screenshot. Create a file named color.txt containing the dominant colour of the image as one lowercase English word (no newline). Then read it back. Finally answer with the colour."},
        ]))?;
        let names = tool_names(&ctx);
        ensure(names.iter().any(|n| n == "write"), format!("没有 write: {names:?}"))?;
        let path = ws.path().join("color.txt");
        let color = std::fs::read_to_string(&path)
            .map_err(|e| format!("color.txt: {e}"))?
            .trim()
            .to_lowercase();
        ensure(color == "red", format!("color.txt={color:?}"))?;
        ensure(has(&reply, "red"), format!("最终答复没红: {:?}", reply.text()))?;
        Ok(format!("tools={names:?}, color=red"))
    })();
    Ok(report("mixed_tool_image", ctx, result))
}

fn assert_image_delete_absorbed(
    label: &str,
    last: &TurnRecord,
    log: &str,
    has_log: bool,
    min_cached_pct: f64,
) -> Result<(), String> {
    if has_log {
        ensure(
            log.contains("前缀无感删除"),
            format!(
                "{label}: server log 未出现吸收;img-evict 行: {:?}",
                log.lines().filter(|l| l.contains("img-evict")).collect::<Vec<_>>()
            ),
        )?;
    }
    ensure(
        last.cached > 0,
        format!("{label}: cached=0(删点之后重算;log 吸收证据={})", log.contains("前缀无感删除")),
    )?;
    ensure(
        last.cached_pct() >= min_cached_pct,
        format!(
            "{label}: cached={:.1}%(期望 ≥{min_cached_pct:.0}%)",
            last.cached_pct()
        ),
    )?;
    Ok(())
}

// ── 5. image prefix-transparent deletion (marked + control + implicit) ───
fn delete_arm(
    client: &Client,
    max_tokens: usize,
    root: &std::path::Path,
    tag: &str,
    nonce: &str,
    session: Option<String>,
    marked: bool,
    log_path: Option<&std::path::Path>,
) -> Result<(Vec<TurnRecord>, String), String> {
    let mut ctx = Ctx::new(
        client,
        &format!("image_delete_marked/{tag}"),
        session,
        false,
        max_tokens,
        root,
    );
    ctx.log_path = log_path.map(std::path::Path::to_path_buf);
    ctx.system(SYS_VISION);
    let red = data_url_png(IMG_W, IMG_H, RED);
    let blue = data_url_png(IMG_W, IMG_H, BLUE);
    ctx.send(json!([
        {"type": "text", "text": format!("History starts here [{nonce}-{tag}].")},
        image_part(&red, marked),
        {"type": "text", "text": format!("Keep this middle text [{nonce}-{tag}].")},
        image_part(&blue, marked),
        {"type": "text", "text": format!("History ends here [{nonce}-{tag}]. Reply with OK.")},
    ]))?;
    ensure(
        remove_image(&mut ctx.history, &red),
        "第一轮的 red 图没找到",
    )?;
    let log_mark = ctx.log_mark();
    let reply = ctx.send(json!(
        "The first image was removed. Which colour is the remaining image? \
         Answer with one lowercase English colour word."
    ))?;
    let log = ctx.log_since(log_mark);
    let log_absorbed = log.contains("前缀无感删除");
    ensure(has(&reply, "blue"), format!("保留图没答蓝: {:?}", reply.text()))?;
    let last = ctx.last();
    if marked {
        // Release builds compile counter_add! out (zero overhead); the info log is
        // the stable observable. cached_tokens independently proves the suffix hit.
        if log_path.is_some() {
            ensure(
                log_absorbed,
                format!("server log 未出现吸收(marked);img-evict 行: {:?}",
                    log.lines().filter(|l| l.contains("img-evict")).collect::<Vec<_>>()),
            )?;
        }
        ensure(
            last.cached > 0,
            format!("marked 臂 cached=0(删点之后重算;log_absorbed={log_absorbed})"),
        )?;
        ensure(
            last.cached_pct() >= 30.0,
            format!("marked 臂 cached={:.1}%(期望 ≥30%)", last.cached_pct()),
        )?;
    } else {
        if log_path.is_some() {
            ensure(
                !log_absorbed,
                format!("control 臂不应吸收,但日志出现前缀无感删除: {:?}",
                    log.lines().filter(|l| l.contains("img-evict")).collect::<Vec<_>>()),
            )?;
        }
        // 无日志时的弱判据:只靠前缀(删点之前)命中。不同 prompt 形状下
        // 这份前缀可以到 30%+;有日志时上面的 `!log_absorbed` 才是硬判据。
        ensure(
            last.cached_pct() < 50.0,
            format!("control 臂 cached={:.1}%(删点之后不应命中)", last.cached_pct()),
        )?;
    }
    let detail = format!(
        "{tag}: cached={:.1}% log_absorbed={log_absorbed} absorbed_metric={} rotate_metric={}",
        last.cached_pct(),
        last.absorbed,
        last.rotate_bytes
    );
    Ok((ctx.turns, detail))
}

// NOTE: the two-image arm keeps a token-stable text boundary. The sliding
// scenario puts a newline before each screenshot (a realistic GUI block); with
// no separator, BPE can merge the text before/after the removed image, so the
// token-level delete detector cannot align (observed on 27B). A real harness
// should keep images on their own line/block.
pub(crate) fn image_delete_marked(
    client: &Client,
    args: &Args,
    pid: u32,
    max_tokens: usize,
) -> Result<ScenarioReport, String> {
    let ws = Workspace::new("image_delete_marked", pid, args.keep)?;
    let mut turns = Vec::new();
    let mut details = Vec::new();
    let mut pass = true;
    // pi-shaped no-session arm first: implicit session 0 must be fresh.
    let arms = [
        ("implicit", None, true),
        ("marked", Some(session(pid, "img-marked")), true),
        ("control", Some(session(pid, "img-control")), false),
    ];
    let log_path = args.server_log.as_ref().map(std::path::Path::new);
    let nonce = nonce(pid);
    for (tag, sess, marked) in arms {
        match delete_arm(client, max_tokens, ws.path(), tag, &nonce, sess, marked, log_path) {
            Ok((t, d)) => {
                turns.extend(t);
                details.push(d);
            }
            Err(e) => {
                pass = false;
                details.push(format!("{tag}: FAIL {e}"));
            }
        }
    }
    Ok(ScenarioReport {
        name: "image_delete_marked".to_string(),
        pass,
        detail: details.join(" | "),
        turns,
    })
}

// ── 6. delete a middle image from a three-image history ─────────────────
pub(crate) fn image_delete_middle(
    client: &Client,
    args: &Args,
    pid: u32,
    max_tokens: usize,
) -> Result<ScenarioReport, String> {
    let ws = Workspace::new("image_delete_middle", pid, args.keep)?;
    let mut ctx = Ctx::new(
        client,
        "image_delete_middle",
        Some(session(pid, "img-middle")),
        false,
        max_tokens,
        ws.path(),
    );
    ctx.log_path = args.server_log.as_ref().map(std::path::PathBuf::from);
    ctx.system(SYS_VISION);
    let red = data_url_png(IMG_W, IMG_H, RED);
    let green = data_url_png(IMG_W, IMG_H, GREEN);
    let blue = data_url_png(IMG_W, IMG_H, BLUE);
    let nonce = nonce(pid);
    let result = (|| -> Result<String, String> {
        ctx.send(json!([
            {"type": "text", "text": format!("History starts here [{nonce}-middle].")},
            image_part(&red, true),
            {"type": "text", "text": "First image."},
            image_part(&green, true),
            {"type": "text", "text": "Middle image."},
            image_part(&blue, true),
            {"type": "text", "text": format!("History ends here [{nonce}-middle]. Reply with OK.")},
        ]))?;
        ensure(
            remove_image(&mut ctx.history, &green),
            "第一轮的 middle 图没找到",
        )?;
        let log_mark = ctx.log_mark();
        let reply = ctx.send(json!(
            "The middle image was removed. Name the colours of the remaining images \
             in order, separated by a comma. Use one lowercase English colour word per image."
        ))?;
        let log = ctx.log_since(log_mark);
        ensure(has(&reply, "red"), format!("剩余图没答红: {:?}", reply.text()))?;
        ensure(has(&reply, "blue"), format!("剩余图没答蓝: {:?}", reply.text()))?;
        ensure(!has(&reply, "green"), format!("删掉的绿图又被提到: {:?}", reply.text()))?;
        assert_image_delete_absorbed("middle", ctx.last(), &log, ctx.log_path.is_some(), 55.0)?;
        Ok(format!(
            "middle removed; remaining=red+blue; cached={:.1}%",
            ctx.last().cached_pct()
        ))
    })();
    Ok(report("image_delete_middle", ctx, result))
}

// ── 7. delete the only/last image => next prompt is pure text ───────────
pub(crate) fn image_delete_last_image(
    client: &Client,
    args: &Args,
    pid: u32,
    max_tokens: usize,
) -> Result<ScenarioReport, String> {
    let ws = Workspace::new("image_delete_last_image", pid, args.keep)?;
    let mut ctx = Ctx::new(
        client,
        "image_delete_last_image",
        Some(session(pid, "img-last")),
        false,
        max_tokens,
        ws.path(),
    );
    ctx.log_path = args.server_log.as_ref().map(std::path::PathBuf::from);
    ctx.system(SYS_VISION);
    // This arm only has one image and one suffix snapshot; a long free-form
    // think can evict that snapshot and force a full recompute. Thinking is
    // orthogonal to the deletion assertion, so pin it off for determinism.
    ctx.thinking(false);
    let red = data_url_png(IMG_W, IMG_H, RED);
    let nonce = nonce(pid);
    let result = (|| -> Result<String, String> {
        ctx.send(json!([
            {"type": "text", "text": format!("History starts here [{nonce}-last].")},
            image_part(&red, true),
            {"type": "text", "text": format!("History ends here [{nonce}-last]. Reply with OK.")},
        ]))?;
        ensure(
            remove_image(&mut ctx.history, &red),
            "第一轮的 last 图没找到",
        )?;
        ensure(
            image_count(&ctx.history) == 0,
            "删掉最后一张图后历史里仍有图",
        )?;
        let log_mark = ctx.log_mark();
        let reply = ctx.send(json!(
            "The only image was removed from the prompt. Recall its colour and \
             answer with one lowercase English colour word."
        ))?;
        let log = ctx.log_since(log_mark);
        ensure(has(&reply, "red"), format!("没回忆出红色: {:?}", reply.text()))?;
        assert_image_delete_absorbed("last", ctx.last(), &log, ctx.log_path.is_some(), 55.0)?;
        Ok(format!(
            "only image removed; history has no images; recalled=red; cached={:.1}%",
            ctx.last().cached_pct()
        ))
    })();
    Ok(report("image_delete_last_image", ctx, result))
}

// ── 8. delete an old image after a real tool round ──────────────────────
pub(crate) fn image_delete_with_tool_round(
    client: &Client,
    args: &Args,
    pid: u32,
    max_tokens: usize,
) -> Result<ScenarioReport, String> {
    let ws = Workspace::new("image_delete_with_tool_round", pid, args.keep)?;
    let mut ctx = Ctx::new(
        client,
        "image_delete_with_tool_round",
        Some(session(pid, "img-tools")),
        true,
        max_tokens,
        ws.path(),
    );
    ctx.log_path = args.server_log.as_ref().map(std::path::PathBuf::from);
    ctx.system(SYS_MIXED);
    let red = data_url_png(IMG_W, IMG_H, RED);
    let nonce = nonce(pid);
    let result = (|| -> Result<String, String> {
        ctx.send(json!([
            {"type": "text", "text": format!("History starts here [{nonce}-tools].")},
            image_part(&red, true),
            {"type": "text", "text": format!(
                "This is the current screenshot. Create a file named color.txt containing \
                 the dominant colour of the image as one lowercase English word (no newline). \
                 Then answer with the colour. [{nonce}-tools]"
            )},
        ]))?;
        let first_tools = tool_names(&ctx);
        ensure(
            first_tools.iter().any(|n| n == "write"),
            format!("工具轮没有 write: {first_tools:?}"),
        )?;
        let path = ws.path().join("color.txt");
        let color = std::fs::read_to_string(&path)
            .map_err(|e| format!("color.txt: {e}"))?
            .trim()
            .to_lowercase();
        ensure(color == "red", format!("color.txt={color:?}"))?;

        ensure(remove_image(&mut ctx.history, &red), "历史里的图没找到")?;
        let log_mark = ctx.log_mark();
        let post_start = ctx.turns.len();
        let reply = ctx.send(json!(
            "The screenshot was removed from the prompt. Use the read tool or bash to read \
             color.txt and answer with the colour it contains. Do not guess."
        ))?;
        let log = ctx.log_since(log_mark);
        let post_tools: Vec<String> = ctx.turns[post_start..]
            .iter()
            .flat_map(|t| t.tools.clone())
            .collect();
        ensure(
            post_tools.iter().any(|n| n == "read" || n == "bash"),
            format!("删图后的工具轮没有 read/bash: {post_tools:?}"),
        )?;
        ensure(has(&reply, "red"), format!("最终答复没红: {:?}", reply.text()))?;
        ensure(
            std::fs::read_to_string(&path).is_ok_and(|c| c.trim() == "red"),
            "color.txt 在删图后被改坏",
        )?;
        // 吸收发生在删图后的**首个**模型请求(可能是 tool_calls);
        // 后续 tool 结果会改写尾巴,最后一轮 cached 不能作为删除证据。
        let absorbed_turn = ctx
            .turns
            .get(post_start)
            .ok_or_else(|| "工具轮没有记录删图后的首个 turn".to_string())?;
        assert_image_delete_absorbed(
            "tool-round",
            absorbed_turn,
            &log,
            ctx.log_path.is_some(),
            50.0,
        )?;
        Ok(format!(
            "write before delete; read/bash after; cached={:.1}%",
            absorbed_turn.cached_pct()
        ))
    })();
    Ok(report("image_delete_with_tool_round", ctx, result))
}

// ── 10. sliding screenshot window (6 turns, keep 3, delete oldest) ───────
pub(crate) fn image_sliding_window(
    client: &Client,
    args: &Args,
    pid: u32,
    max_tokens: usize,
) -> Result<ScenarioReport, String> {
    let ws = Workspace::new("image_sliding_window", pid, args.keep)?;
    let mut ctx = Ctx::new(
        client,
        "image_sliding_window",
        Some(session(pid, "sliding")),
        false,
        max_tokens,
        ws.path(),
    );
    ctx.log_path = args.server_log.as_ref().map(std::path::PathBuf::from);
    ctx.system(SYS_VISION);
    let colors = [RED, BLUE, GREEN, YELLOW, PURPLE, CYAN];
    let nonce = nonce(pid);
    let result = (|| -> Result<String, String> {
        let mut cached = Vec::new();
        for (k, rgb) in colors.iter().enumerate() {
            if image_count(&ctx.history) >= 3 {
                let removed = remove_oldest_image(&mut ctx.history);
                ensure(removed.is_some(), format!("turn {} 应删图但没找到", k + 1))?;
            }
            let url = data_url_png(IMG_W, IMG_H, *rgb);
            let log_mark = ctx.log_mark();
            // No client-side newline: the server normalizes multipart text parts
            // (text parts end with '\n') before rendering, so the token boundary
            // around the image stays stable when the image is removed later.
            let reply = ctx.send(json!([
                {"type": "text", "text": format!("Step {} [{nonce}]: this is the current screenshot.", k + 1)},
                image_part(&url, true),
                {"type": "text", "text": "Describe the screenshot in one short sentence."},
            ]))?;
            let log = ctx.log_since(log_mark);
            ensure(!reply.text().trim().is_empty(), format!("turn {} 回答为空", k + 1))?;
            let last = ctx.last();
            if k < 3 {
                ensure(
                    last.cached_pct() == 0.0 || last.absorbed == 0,
                    format!("turn {} 还没删图却命中了删除路径", k + 1),
                )?;
            } else {
                if ctx.log_path.is_some() {
                    ensure(
                        log.contains("前缀无感删除"),
                        format!(
                            "turn {} 删图未吸收(server log);cached={:.1}% img-evict={:?}",
                            k + 1,
                            last.cached_pct(),
                            log.lines().filter(|l| l.contains("img-evict")).collect::<Vec<_>>()
                        ),
                    )?;
                }
                ensure(
                    last.cached_pct() >= 30.0,
                    format!(
                        "turn {} cached 只有 {:.1}%(删点之后可能重算)",
                        k + 1,
                        last.cached_pct()
                    ),
                )?;
            }
            cached.push(format!("{:.0}%", last.cached_pct()));
        }
        Ok(format!("cached by turn: {}", cached.join(" / ")))
    })();
    Ok(report("image_sliding_window", ctx, result))
}

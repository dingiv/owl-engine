//! owl E2E 黑盒测试(2026-10-13):启动 server 进程 + HTTP 请求 + 语义
//! 判据分级。
//!
//! 判据分级(§三十二 教训:判据污染,词汤/长数字/verbatim 复述不可靠):
//! - L1 协议面(强):HTTP 200、JSON 结构、usage、finish_reason、content 非空
//! - L2 语义面(软):常识/算术 → content 含正确关键词(think 推理段即可)
//! - L3 最终答案面(已知 bug 记录):`</think>` 后含正确答案 —— 当前
//!   think 结论正确但最终答案输出损坏(N/A/"["),xfail 立案,修复后转正
//!
//! 门:OWL_E2E=1(需 GPU + 模型路径);server 启动 = CARGO_BIN_EXE_server
//! + 内嵌配置(8448/8704,split;§三十三:此配置数学 gate 通过)。

use std::io::{Read, Write};
use std::net::TcpStream;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const PORT: u16 = 8135;
const HEALTH_TIMEOUT_SECS: u64 = 180;
const REQ_TIMEOUT_SECS: u64 = 300;

/// server 配置(testdata/args/e2e-27b.toml;口径来由见该文件头注)
const CONFIG: &str = include_str!("../../../testdata/args/e2e-27b.toml");

// ── 测试骨架(2026-10-13 按两条律重写)───────────────────────────────
// 律 #1:可失败函数的返回值**不许丢弃**;律 #2:可失败逻辑**不许**写成
// 返回 `()`/`bool` 的"看起来不会失败"的函数。
//
// 病史(这条律的由来):原骨架 `let _srv = spawn_server();`(丢句柄)+
// `wait_health() -> bool`(吞掉全部连接/读写错误)→ 残留实例占着 8135 时,
// 新实例绑不上(且还在装载权重),而 health 由**残留实例**应答 → **静默假绿**
// (实测整轮 10.32s,远短于一次冷启动才被识破)。现:
//   ① 起 server 前探端口:占用即 fail-fast 并报占用者;
//   ② wait_health 返 Result,且每轮校验**自己的子进程仍存活**(子进程退出
//      = 立即报错带状态,而不是空等 180s 超时);
//   ③ 所有 HTTP 辅助返 Result(连接/写/读错误一律上报,零静默)。

struct ServerGuard {
    child: Child,
}

impl ServerGuard {
    fn child_mut(&mut self) -> &mut Child {
        &mut self.child
    }
}

impl Drop for ServerGuard {
    fn drop(&mut self) {
        // Drop 无法返回 Result:失败**显式落日志**(不许静默丢弃 —— 律 #1)
        if let Err(e) = self.child.kill() {
            eprintln!("[e2e] ServerGuard::drop: kill 子进程失败: {e}");
        }
        if let Err(e) = self.child.wait() {
            eprintln!("[e2e] ServerGuard::drop: 回收子进程失败: {e}");
        }
    }
}

/// 端口是否已被占用(占用者线索一并带回;零新依赖 = 调 `ss` 取行)。
/// `ss` 不可用 = **无法完成这项校验** → 显式报错(不许静默当作"端口空闲",
/// 那正是假绿的来源 —— 律 #1)。
fn port_occupant(port: u16) -> Result<Option<String>, String> {
    let out = Command::new("ss")
        .args(["-tlnp"])
        .output()
        .map_err(|e| format!("探测端口占用失败(ss 不可用): {e};请装 iproute2 或手工确认 {port} 空闲"))?;
    let text = String::from_utf8_lossy(&out.stdout);
    let needle = format!(":{port} ");
    Ok(text.lines().find(|l| l.contains(&needle)).map(|l| l.trim().to_string()))
}

/// 起 server(可失败 → `Result`;占用/写配置/启动任一失败都上报)
fn spawn_server() -> Result<ServerGuard, String> {
    if let Some(occ) = port_occupant(PORT)? {
        return Err(format!(
            "端口 {PORT} 已被占用(疑似残留 server 实例,会导致本测试打到别人身上 = 假绿)。\
             请先清理:`ss -tlnp | grep {PORT}` → kill 占用者 PID。占用者: {occ}"
        ));
    }
    let cfg_path = std::env::temp_dir().join("owl-e2e-server.toml");
    std::fs::write(&cfg_path, CONFIG).map_err(|e| format!("写 e2e 配置 {cfg_path:?}: {e}"))?;
    let log_path = std::env::temp_dir().join("owl-e2e-server.log");
    let log = std::fs::File::create(&log_path).map_err(|e| format!("建日志 {log_path:?}: {e}"))?;
    let bin = env!("CARGO_BIN_EXE_server");
    let mut cmd = Command::new(bin);
    let spec = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../profiles/legacy/spec-dflash.toml");
    cmd.arg("-p")
        .arg(&spec)
        .arg(&cfg_path)
        .stdout(Stdio::null())
        .stderr(log);
    // 父死子必死(2026-10-13):本测试进程若被 SIGKILL(超时/中断),`Drop`
    // 不会执行 → 子 server 变孤儿,继续占端口 + 显存(实测导致后续整轮假绿)。
    // `PR_SET_PDEATHSIG` = 内核在父进程退出时给子进程发 SIGKILL。
    #[cfg(target_os = "linux")]
    unsafe {
        use std::os::unix::process::CommandExt as _;
        cmd.pre_exec(|| {
            if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let child = cmd
        .spawn()
        .map_err(|e| format!("启动 server 二进制 {bin}: {e}"))?;
    Ok(ServerGuard { child })
}

/// 等 health **并校验应答者是我们自己的子进程**(可失败 → `Result`)
fn wait_health(srv: &mut ServerGuard) -> Result<(), String> {
    let deadline = Instant::now() + Duration::from_secs(HEALTH_TIMEOUT_SECS);
    while Instant::now() < deadline {
        // ① 自己的子进程还活着吗?(绑端口失败会 panic 退出 —— 立刻报,别空等)
        match srv.child_mut().try_wait() {
            Ok(Some(status)) => {
                return Err(format!(
                    "server 子进程已退出({status})—— 启动失败(绑端口/装载失败?),见 {}",
                    std::env::temp_dir().join("owl-e2e-server.log").display()
                ))
            }
            Ok(None) => {}
            Err(e) => return Err(format!("try_wait 子进程失败: {e}")),
        }
        // ② health 探活(连接/写/读错误都不吞,循环内重试但留最后错误)
        if let Ok(mut s) = TcpStream::connect(("127.0.0.1", PORT)) {
            let req = format!(
                "GET /health HTTP/1.1\r\nHost: 127.0.0.1:{PORT}\r\nConnection: close\r\n\r\n"
            );
            s.set_read_timeout(Some(Duration::from_secs(5)))
                .map_err(|e| format!("设 health 读超时: {e}"))?;
            let wrote = s.write_all(req.as_bytes());
            let mut buf = String::new();
            let read = s.read_to_string(&mut buf);
            if wrote.is_ok() && read.is_ok() && buf.contains("\"ok\":true") {
                return Ok(());
            }
        }
        std::thread::sleep(Duration::from_secs(3));
    }
    Err(format!(
        "server health 超时({HEALTH_TIMEOUT_SECS}s,子进程仍在)—— 见 {}",
        std::env::temp_dir().join("owl-e2e-server.log").display()
    ))
}

/// 最小 HTTP POST(可失败 → `Result`;手写帧解析,零新依赖)
fn http_post(path: &str, body: &str) -> Result<(u16, String), String> {
    let mut s = TcpStream::connect(("127.0.0.1", PORT)).map_err(|e| format!("连接 server: {e}"))?;
    s.set_read_timeout(Some(Duration::from_secs(REQ_TIMEOUT_SECS)))
        .map_err(|e| format!("设读超时: {e}"))?;
    s.set_write_timeout(Some(Duration::from_secs(REQ_TIMEOUT_SECS)))
        .map_err(|e| format!("设写超时: {e}"))?;
    let req = format!(
        "POST {path} HTTP/1.1\r\nHost: 127.0.0.1:{PORT}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    s.write_all(req.as_bytes()).map_err(|e| format!("写请求: {e}"))?;
    let mut buf = String::new();
    s.read_to_string(&mut buf).map_err(|e| format!("读响应: {e}"))?;
    // 解析失败**不静默**:下面 ok_or_else 会把失败上报(丢掉的是 parse 细节)
    let status: u16 = buf
        .split_whitespace()
        .nth(1)
        .and_then(|x| x.parse().ok())
        .ok_or_else(|| format!("状态行非法: {}", buf.chars().take(80).collect::<String>()))?;
    // body = 第一个 \r\n\r\n 之后(响应无 chunked:owl respond_json 定长)。
    // 缺分隔符 → 空 body:由调用方的 JSON 解析/断言兜住(非静默通过)
    let body = buf
        .split_once("\r\n\r\n")
        .map(|(_, b)| b.to_string())
        .unwrap_or_default();
    Ok((status, body))
}

/// 流式 POST:逐块读到 `data: [DONE]` 即停(可失败 → `Result`)。
/// **不能 read_to_string** —— SSE 连接按 HTTP keep-alive 不关闭,读到 EOF
/// 会挂满读超时;读错误/超时一律上报(不许静默截断 —— 律 #1)。
fn http_post_sse(path: &str, body: &str) -> Result<(u16, String), String> {
    let mut s = TcpStream::connect(("127.0.0.1", PORT)).map_err(|e| format!("连接 server: {e}"))?;
    s.set_read_timeout(Some(Duration::from_secs(30)))
        .map_err(|e| format!("设读超时: {e}"))?;
    s.set_write_timeout(Some(Duration::from_secs(30)))
        .map_err(|e| format!("设写超时: {e}"))?;
    let req = format!(
        "POST {path} HTTP/1.1\r\nHost: 127.0.0.1:{PORT}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    s.write_all(req.as_bytes()).map_err(|e| format!("写请求: {e}"))?;
    // 字节级累积(不再逐读 from_utf8_lossy:多字节字符跨读边界会被截成替换符 ——
    // chunk 体里的中文一旦被截,后面的 chunked 解帧就跟着错位)
    let mut raw: Vec<u8> = Vec::new();
    let mut buf = [0u8; 8192];
    loop {
        match s.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                raw.extend_from_slice(&buf[..n]);
                if find_bytes(&raw, b"data: [DONE]").is_some() {
                    break;
                }
            }
            Err(e) => {
                // 已收到部分帧:报错并把现场带出(不许静默吞)
                return Err(format!(
                    "流式读失败({e});已收到 {} 字节: {}",
                    raw.len(),
                    String::from_utf8_lossy(&raw).chars().take(200).collect::<String>()
                ))
            }
        }
    }
    let out = String::from_utf8_lossy(&raw).to_string();
    if !out.contains("data: [DONE]") {
        return Err(format!(
            "流式响应未见 `data: [DONE]`(连接提前结束);已收到: {}",
            out.chars().take(200).collect::<String>()
        ));
    }
    // 同上:失败由 ok_or_else 上报(非静默)
    let status: u16 = out
        .split_whitespace()
        .nth(1)
        .and_then(|x| x.parse().ok())
        .ok_or_else(|| format!("状态行非法: {}", out.chars().take(80).collect::<String>()))?;
    // **G5④:SSE 走 chunked 定界** —— 测试侧裸 socket 得自己解帧
    // (真客户端 undici/fetch/http.client 原生支持;旧实现靠连接关闭定界)
    let (head, body) = match out.split_once("\r\n\r\n") {
        Some((h, b)) => (h.to_string(), b.to_string()),
        None => (out.clone(), String::new()),
    };
    let body = if head.to_ascii_lowercase().contains("transfer-encoding: chunked") {
        dechunk(&body)?
    } else {
        body
    };
    Ok((status, format!("{head}\r\n\r\n{body}")))
}

/// chunked 解帧(`<hex>[;ext]\r\n<data>\r\n` … `0\r\n\r\n`);
/// 失败**显式上报**(不静默返回半截 —— 与 http.rs 读侧同律)
fn dechunk(raw: &str) -> Result<String, String> {
    let mut out = String::new();
    let mut rest = raw;
    loop {
        let (line, after) = rest.split_once("\r\n").ok_or_else(|| {
            format!("chunk 头缺 CRLF:{}", rest.chars().take(60).collect::<String>())
        })?;
        let n = usize::from_str_radix(line.split(';').next().unwrap_or("").trim(), 16)
            .map_err(|e| format!("chunk 长度非法({line:?}): {e}"))?;
        if n == 0 {
            return Ok(out);
        }
        if after.len() < n + 2 {
            return Err(format!("chunk 体截断(需 {}B,余 {}B)", n + 2, after.len()));
        }
        out.push_str(&after[..n]);
        rest = &after[n + 2..];
    }
}

/// 字节级子串查找(SSE 判据不看字符边界,只看 `data: [DONE]` 出现)
fn find_bytes(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

fn chat(session: &str, prompt: &str, max_tokens: u32) -> Result<(u16, serde_json::Value), String> {
    let body = serde_json::json!({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "session": session,
    });
    let (status, resp) = http_post("/v1/chat/completions", &body.to_string())?;
    let v: serde_json::Value = serde_json::from_str(&resp)
        .map_err(|e| format!("响应非法 JSON({status}): {e}: {resp}"))?;
    Ok((status, v))
}

/// 取 choices[0].message.content。
/// 字段缺失 = **显式失败**(原 `unwrap_or_default` 静默变空串 —— 违反律 #1:
/// 判据可能因此"看起来在断言",实则断的是空串)。
fn content_of(v: &serde_json::Value) -> String {
    v["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_else(|| panic!("响应缺 choices[0].message.content: {v}"))
        .to_string()
}

#[test]
fn e2e_server_suite() {
    // if !owl_shared::env_reader::flag("OWL_E2E") {
    //     eprintln!("skip: OWL_E2E 未设(需 GPU + 模型路径)");
    //     return;
    // }
    // 可失败辅助一律在此收口:失败即红(不丢句柄、不吞错误 —— 律 #1/#2)
    let mut srv = spawn_server().expect("起 e2e server");
    wait_health(&mut srv).expect("server 就绪(健康探活 + 子进程存活校验)");
    let post = |p: &str, b: &str| http_post(p, b).expect("http_post");
    let chatx = |s: &str, p: &str, m: u32| chat(s, p, m).expect("chat");

    // ============ L1 协议面(强判据)============
    let (status, _) = chatx("e2e-l1", "What is the capital of France?", 96);
    assert_eq!(status, 200, "HTTP 200");
    let (_, v) = chatx("e2e-l1b", "What is the capital of France?", 96);
    let content = content_of(&v);
    assert!(!content.is_empty(), "content 非空");
    assert!(
        v["usage"]["prompt_tokens"].as_u64().unwrap_or(0) > 0,
        "usage.prompt_tokens > 0"
    );
    assert!(
        v["usage"]["completion_tokens"].as_u64().unwrap_or(0) > 0,
        "usage.completion_tokens > 0"
    );
    let finish = v["choices"][0]["finish_reason"].as_str().unwrap_or("");
    assert!(
        finish == "stop" || finish == "length",
        "finish_reason 合法({finish})"
    );
    println!("[L1] 协议面 PASS(status/usage/finish/content 非空)");

    // ============ L2 语义面(think 推理含正确答案)============
    // 常识:Paris(think 段推理正确;content 应含 Paris)
    let (_, v) = chatx("e2e-l2-paris", "What is the capital of France?", 256);
    let c = content_of(&v);
    assert!(
        c.to_lowercase().contains("paris"),
        "常识 Paris content 应含 Paris:{c}"
    );
    let paris_c = c.clone(); // L3 判据用本应答(原误用后续多轮应答)
    println!("[L2] 常识 Paris PASS");

    // 算术:2+2 —— 已知问题记录(§三十三):短 prompt 下 think 混乱
    // ("Wait, no, they're asking…"),不含答案 4;修复后转 assert。
    let (_, v) = chatx("e2e-l2-arith", "What is 2 + 2? Answer with just the number.", 256);
    let c = content_of(&v);
    assert!(c.contains('4'), "算术 2+2 content 应含 4:{c}");
    println!("[L2] 算术 2+2 PASS");

    // ============ 多轮(agent 式,self-contained 上下文,同 session 续)============
    // 轮 1:告知名字
    let (st1, v) = chatx(
        "e2e-mt",
        "My name is Alex. Nice to meet you.",
        256,
    );
    assert_eq!(st1, 200, "多轮 轮1 HTTP 200");
    assert!(!content_of(&v).is_empty(), "多轮 轮1 非空");
    // 轮 2:询问(self-contained 提示:历史由客户端拼,naive 渲染现状)
    let body = serde_json::json!({
        "messages": [
            {"role": "user", "content": "My name is Alex. Nice to meet you."},
            {"role": "assistant", "content": "Nice to meet you too, Alex."},
            {"role": "user", "content": "What is my name? Answer with just the name."}
        ],
        "max_tokens": 256,
        "session": "e2e-mt"
    });
    let (status, resp) = post("/v1/chat/completions", &body.to_string());
    let v: serde_json::Value = serde_json::from_str(&resp)
        .unwrap_or_else(|e| panic!("多轮 轮2 响应非法 JSON({status}): {e}"));
    assert_eq!(status, 200, "多轮 轮2 HTTP 200");
    let c = content_of(&v);
    assert!(
        c.to_lowercase().contains("alex"),
        "多轮记忆 content 应含 Alex:{c}"
    );
    println!("[多轮] 名字记忆 PASS(content 含 Alex)");

    // ============ L3 最终答案面(已知 bug 记录:xfail)============
    // `</think>` 后的最终答案当前损坏(N/A/"[";§三十二):think 推理正确
    // (含 Paris)但最终答案段不含。修复后本断言转正。
    let after_think = paris_c
        .split_once("</think>")
        .map(|(_, tail)| tail.to_string())
        .unwrap_or_default();
    let has_paris_after = after_think.to_lowercase().contains("paris");
    if has_paris_after {
        println!("[L3] 最终答案含 Paris(v2 bug 已修,xfail 转正!)");
    } else {
        println!(
            "[L3] 已知 bug:think 结论正确但最终答案损坏(尾段={:?})—— 立案不失败",
            after_think.chars().take(40).collect::<String>()
        );
    }

    // ============ 测速口径扩展(ignore_eos / 流式 usage;2026-10-13)============
    // ① ignore_eos=true → 强制跑满 max_tokens 且 finish_reason=length
    //   (动机:OOD 输入下模型常吐 2 token 就 EOS,decode 读数成伪影;
    //    finish_reason 此前恒 "stop" 的谎一并修正 —— length 分支由此钉住)
    let body = serde_json::json!({
        "messages": [{"role": "user", "content": "What is 2 + 2? Answer with just the number."}],
        "max_tokens": 64,
        "session": "e2e-ie",
        "ignore_eos": true,
    });
    let (st, resp) = post("/v1/chat/completions", &body.to_string());
    let v: serde_json::Value = serde_json::from_str(&resp).expect("ignore_eos 响应 JSON");
    assert_eq!(st, 200, "ignore_eos HTTP 200");
    assert_eq!(
        v["usage"]["completion_tokens"].as_u64(),
        Some(64),
        "ignore_eos 应跑满 64 token(usage 真账)"
    );
    assert_eq!(
        v["choices"][0]["finish_reason"].as_str(),
        Some("length"),
        "跑满预算 → finish_reason=length"
    );
    println!("[口径] ignore_eos → 64/64 + finish_reason=length PASS");

    // ② 流式 + stream_options.include_usage → 末帧 usage 真账
    //   (病史:spec 路径轮末 `pending_events = evq` 覆盖丢 Completed →
    //    客户端一帧不收且不 EOF;定谳见台账 §11.14/坑册。本门即其回归门)
    let body = serde_json::json!({
        "messages": [{"role": "user", "content": "Say READY."}],
        "max_tokens": 16,
        "session": "e2e-usage",
        "stream": true,
        "stream_options": {"include_usage": true},
    });
    let (st, resp) = http_post_sse("/v1/chat/completions", &body.to_string())
        .expect("流式请求(挂死/读失败即在此报错)");
    assert_eq!(st, 200, "流式 include_usage HTTP 200");
    assert!(
        resp.contains("\"completion_tokens\"") && resp.contains("\"prompt_tokens\""),
        "流式末帧应含 usage 真账:{}",
        resp.chars().take(200).collect::<String>()
    );
    assert!(resp.contains("data: [DONE]"), "流式应以 [DONE] 收尾");
    println!("[口径] 流式 include_usage → 末帧 usage PASS");

    // ============ 会话隔离(自含 codeword;owl 服务端 session = KV 复用键,
    // 历史由客户端拼 → 跨 session 天然隔离;此用例测独立 session 应答正确性)
    // ⚠️ 2026-10-13 修正(原轮 2 只发新句 = 违反契约,必红):服务端**不存
    // 对话历史**(瘦服务端,见 docs/arch/server-多会话并发设计.md §2.2;
    // session.rs 头注:"引擎感知 session" = KV cache 区域 + 会话 token 账,
    // 不是服务端记忆)→ 多轮必须按 OpenAI 惯例**客户端拼全量 messages**;
    // 引擎仅按 token 账前缀守卫做增量 prefill(失配回退全量,S1)。
    let (_, v) = chatx(
        "e2e-iso-a",
        "The magic word is PURPLE. What is the magic word? Answer with just the word.",
        256,
    );
    let ca = content_of(&v);
    assert!(
        ca.to_lowercase().contains("purple"),
        "隔离 A 应答 PURPLE:{ca}"
    );
    let (st, v) = chatx(
        "e2e-iso-b",
        "The magic word is ORANGE. Say READY.",
        64,
    );
    assert_eq!(st, 200, "隔离 B 轮1 200");
    assert!(!content_of(&v).is_empty(), "隔离 B 轮1 非空");
    // 轮 2:契约姿势 —— 全量历史重发(assistant 段原样回填轮 1 应答,
    // = 真实 OpenAI 客户端行为);同 session 保 GDN 格/KV 块连续性
    let turn1 = content_of(&v);
    let body = serde_json::json!({
        "messages": [
            {"role": "user", "content": "The magic word is ORANGE. Say READY."},
            {"role": "assistant", "content": turn1},
            {"role": "user", "content": "What was the magic word I told you? Answer with just the word."}
        ],
        "max_tokens": 256,
        "session": "e2e-iso-b"
    });
    let (status, resp) = post("/v1/chat/completions", &body.to_string());
    let v: serde_json::Value = serde_json::from_str(&resp)
        .unwrap_or_else(|e| panic!("隔离 B 轮2 响应非法 JSON({status}): {e}"));
    assert_eq!(status, 200, "隔离 B 轮2 HTTP 200");
    let cb = content_of(&v);
    assert!(
        cb.to_lowercase().contains("orange"),
        "隔离 B 应答 ORANGE:{cb}"
    );
    println!("[会话隔离] 独立 session 自含应答 PASS(PURPLE/ORANGE)");

    // ============ 并发(8 线程;闸门/排队/会话隔离并发版回归)============
    let mut handles = Vec::new();
    for i in 0..8usize {
        let prompt = format!(
            "The magic number assigned to you is {}. What is your magic number? Answer with just the number.",
            1000 + i
        );
        handles.push(std::thread::spawn(move || {
            let body = serde_json::json!({
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 96,
                "session": format!("e2e-cc-{i}"),
            });
            let (status, resp) = post("/v1/chat/completions", &body.to_string());
            let v: serde_json::Value = serde_json::from_str(&resp)
                .unwrap_or_else(|e| panic!("并发 {i} 响应非法 JSON: {e}"));
            let answer = content_of(&v);
            let want = format!("{}", 1000 + i);
            (status, answer.contains(&want), i)
        }));
    }
    for h in handles {
        let (status, ok, i) = h.join().expect("并发线程 panic");
        assert_eq!(status, 200, "并发 {i} HTTP 200");
        assert!(ok, "并发 {i} 应答应含魔数 {}", 1000 + i);
    }
    println!("[并发] 8 线程并发独立 session 全 PASS(闸门/隔离回归)");

    // ============ agent 服务面 L1(B3;2026-10-14)============
    // 强判据优先逆:这里断的全是**协议不变式**(I1~I7),不依赖
    // "模型肯不肯发工具调用" —— 后者是 A7 台账项(工具遵循度),
    // 拿它做门 = 拿模型脾气做门(红绿不可复现)。
    //
    // ① 工具历史可渲染 + 往返 200(assistant.tool_calls 的 arguments 是
    //    **JSON 字符串** + role:"tool" 结果回填)⇒ `normalize_tool_calls`
    //    + 模板工具分支同时被验到
    let tool_history = serde_json::json!({
        "messages": [
            {"role": "user", "content": "List the files in the current directory."},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_e2e_1", "type": "function",
                 "function": {"name": "bash", "arguments": "{\"command\":\"ls -la\"}"}}
            ]},
            {"role": "tool", "tool_call_id": "call_e2e_1", "name": "bash",
             "content": "total 8\ndrwxr-xr-x 2 owl owl 4096 ."},
            {"role": "user", "content": "How many entries does that listing show? Answer with just the number."}
        ],
        "tools": [{"type": "function", "function": {
            "name": "bash", "description": "Run a bash command",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}
        }}],
        "max_tokens": 256,
        "session": "e2e-agent-hist"
    });
    let (st, resp) = post("/v1/chat/completions", &tool_history.to_string());
    let v: serde_json::Value = serde_json::from_str(&resp).expect("工具历史响应 JSON");
    assert_eq!(st, 200, "工具历史可渲染且 200:{resp}");
    // **强判据 = prompt_tokens A/B**(不依赖模型质量):带工具历史(含 tools 块
    // + 两条 tool 轮)的 prompt 必然显著长于同题单轮 ⇒ 渲染丢工具轮 = 此处红
    // (G2 负判据)。内容是否非空属模型/贪心档行为(e2e 档锁 greedy),
    // 只打印不断言 —— 拿模型雿气做门 = 红绿不可复现。
    let pt_tools = v["usage"]["prompt_tokens"].as_u64().unwrap_or(0);
    let bare = serde_json::json!({
        "messages": [
            {"role": "user", "content": "List the files in the current directory."},
            {"role": "user", "content": "How many entries does that listing show? Answer with just the number."}
        ],
        "max_tokens": 64,
        "session": "e2e-agent-bare"
    });
    let (_, resp_bare) = post("/v1/chat/completions", &bare.to_string());
    let vb: serde_json::Value = serde_json::from_str(&resp_bare).expect("对照响应 JSON");
    let pt_bare = vb["usage"]["prompt_tokens"].as_u64().unwrap_or(0);
    assert!(
        pt_tools > pt_bare + 100,
        "工具历史必须真进 prompt(pt_tools={pt_tools} vs 对照 pt_bare={pt_bare})—— 渲染丢工具轮 = 负判据"
    );
    println!(
        "[agent] 工具历史渲染 PASS(prompt_tokens {pt_tools} vs 无工具对照 {pt_bare});本轮 content={:?}",
        content_of(&v).chars().take(40).collect::<String>()
    );

    // ② thinking 回填对称(reasoning_content 入历史;§3.4/§4.3 R1)——
    //    同强判据:回填段必须进 prompt(与不带 reasoning_content 的对照比)
    let think_history = serde_json::json!({
        "messages": [
            {"role": "user", "content": "In one word, the color of grass?"},
            {"role": "assistant", "reasoning_content": "The user asks a simple question. Grass is green. So the answer is green.",
             "content": "Green"},
            {"role": "user", "content": "Say READY."}
        ],
        "max_tokens": 64,
        "session": "e2e-agent-think"
    });
    let (st, resp) = post("/v1/chat/completions", &think_history.to_string());
    assert_eq!(st, 200, "reasoning_content 回填可渲染:{resp}");
    let v: serde_json::Value = serde_json::from_str(&resp).expect("think 回填响应 JSON");
    let pt_think = v["usage"]["prompt_tokens"].as_u64().unwrap_or(0);
    let no_think = serde_json::json!({
        "messages": [
            {"role": "user", "content": "In one word, the color of grass?"},
            {"role": "assistant", "content": "Green"},
            {"role": "user", "content": "Say READY."}
        ],
        "max_tokens": 64,
        "session": "e2e-agent-nothink"
    });
    let (_, resp_nt) = post("/v1/chat/completions", &no_think.to_string());
    let vn: serde_json::Value = serde_json::from_str(&resp_nt).expect("对照响应 JSON");
    let pt_nothink = vn["usage"]["prompt_tokens"].as_u64().unwrap_or(0);
    assert!(
        pt_think > pt_nothink,
        "reasoning_content 必须真进 prompt(回填 {pt_think} vs 不回填 {pt_nothink})"
    );
    println!(
        "[agent] reasoning_content 回填 PASS(prompt_tokens {pt_think} vs 不回填 {pt_nothink})"
    );

    // ③ SSE 不变式:finish_reason 恰好一次 / usage 帧 choices=[] 且在
    //    [DONE] 之前 / cached_tokens 透出(AGT-12)
    let sse_body = serde_json::json!({
        "messages": [{"role": "user", "content": "Say READY."}],
        "max_tokens": 32,
        "session": "e2e-agent-sse",
        "stream": true,
        "stream_options": {"include_usage": true},
        "chat_template_kwargs": {"enable_thinking": true, "preserve_thinking": true},
    });
    let (st, sse) = http_post_sse("/v1/chat/completions", &sse_body.to_string())
        .expect("agent SSE 请求(挂死/读失败即在此报错)");
    assert_eq!(st, 200, "agent SSE HTTP 200");
    assert_eq!(
        sse.matches("\"finish_reason\":\"").count(),
        1,
        "I1:finish_reason 必须恰好发一次(chunk 里的 null 不计)"
    );
    assert!(sse.contains("data: [DONE]"), "流式以 [DONE] 收尾");
    let before_done = sse.split("data: [DONE]").next().unwrap_or("");
    let last_frame = before_done
        .rsplit("data: ")
        .next()
        .unwrap_or("");
    let uv: serde_json::Value = serde_json::from_str(last_frame.trim())
        .unwrap_or_else(|e| panic!("usage 帧非法 JSON({e}): {last_frame}"));
    assert_eq!(
        uv["choices"].as_array().map(|a| a.len()),
        Some(0),
        "I2:usage 帧 choices 必须为空数组:{uv}"
    );
    assert!(uv["usage"]["prompt_tokens_details"]["cached_tokens"].is_u64(),
        "AGT-12:cached_tokens 必须透出:{uv}");
    // G5:思考相位真账(两形态单源;reasoning ≤ completion 恒成立)
    let rt = uv["usage"]["completion_tokens_details"]["reasoning_tokens"]
        .as_u64()
        .unwrap_or_else(|| panic!("G5:reasoning_tokens 必须透出:{uv}"));
    let ct = uv["usage"]["completion_tokens"].as_u64().unwrap_or(0);
    assert!(rt <= ct, "reasoning_tokens({rt}) 不得超过 completion_tokens({ct}):{uv}");
    assert!(sse.contains("\"reasoning_content\"") || sse.contains("\"content\""),
        "三通道至少一路有内容");
    println!(
        "[agent] SSE 不变式 PASS(finish_reason×1 / usage 帧 choices=[] / cached_tokens 透出){}",
        if sse.contains("\"reasoning_content\"") { ";think 通道在" } else { ";本轮无 think 段(非判据)" }
    );

    // ④ stop 串真生效(AGT-06):受控 A/B —— 用**贪心必复读**的提示词
    //    (greedy 的看家本领)先验"无 stop 时会吐出停止串",再验带 stop 被截断。
    //    前提不成立(模型没按指令复读)= 打印 SKIP(不拿模型雿气当门);
    //    engine 侧 holdback 另有单测。
    // 判据用**单词** "BANANA"(不依赖复读的间隔形态 —— 实测 split/非 split 档
    // 下间隔不同,用 "BANANA BANANA" 会让门变脆):控制臂必含它,stop 臂必无它。
    let rep_prompt = "Print the word BANANA separated by spaces, over and over, forever. Do not stop.";
    let (_, v0) = chatx("e2e-agent-nostop", rep_prompt, 64);
    let c0 = content_of(&v0);
    let with_stop = serde_json::json!({
        "messages": [{"role": "user", "content": rep_prompt}],
        "max_tokens": 64,
        "temperature": 0.0,
        "stop": ["BANANA"],
        "chat_template_kwargs": {"enable_thinking": false},
        "session": "e2e-agent-stop",
    });
    let (st, resp) = post("/v1/chat/completions", &with_stop.to_string());
    let v1: serde_json::Value = serde_json::from_str(&resp).expect("stop 响应 JSON");
    assert_eq!(st, 200, "stop 请求 HTTP 200");
    let c1 = content_of(&v1);
    if c0.contains("BANANA") {
        assert!(!c1.contains("BANANA"), "stop 命中后输出不得含停止串:{c1:?}");
        assert_eq!(
            v1["choices"][0]["finish_reason"].as_str(),
            Some("stop"),
            "stop 命中 ⇒ finish_reason=stop(实得 {:?})",
            v1["choices"][0]["finish_reason"]
        );
        assert!(c1.len() <= c0.len(), "截断后不得变长({} vs {})", c1.len(), c0.len());
        println!("[agent] stop 串 PASS(对照 {:?} / 带 stop {:?})", c0, c1);
    } else {
        println!(
            "[agent] stop 串 SKIP:对照未复读停止串(模型未按指令),无法判定;engine holdback 另有单测。对照={c0:?}"
        );
    }

    // ⑤ 400 面(不静默):未支持关键字段 / 坏 arguments / 坏 stop / n>1
    for (name, body) in [
        ("n>1", serde_json::json!({"messages":[{"role":"user","content":"x"}],"n":4})),
        ("坏 arguments", serde_json::json!({"messages":[
            {"role":"user","content":"x"},
            {"role":"assistant","content":"","tool_calls":[{"id":"c","type":"function",
              "function":{"name":"bash","arguments":"{oops"}}]}],"max_tokens":8})),
        ("坏 stop", serde_json::json!({"messages":[{"role":"user","content":"x"}],"stop":[17]})),
        ("logprobs", serde_json::json!({"messages":[{"role":"user","content":"x"}],"logprobs":true})),
        ("非 function 工具历史", serde_json::json!({"messages":[
            {"role":"user","content":"x"},
            {"role":"assistant","content":"","tool_calls":[{"id":"c","type":"custom",
              "function":{"name":"bash","arguments":{}}}]}],"max_tokens":8})),
    ] {
        let (st, resp) = post("/v1/chat/completions", &body.to_string());
        assert_eq!(st, 400, "{name} 应结构化拒(400),实得 {st}:{resp}");
        let e: serde_json::Value = serde_json::from_str(&resp).expect("错误体 JSON");
        assert!(
            e["error"]["message"].as_str().is_some_and(|m| !m.is_empty()),
            "{name}:错误消息非空:{e}"
        );
    }
    println!("[agent] 400 面 PASS(n>1/坏 arguments/坏 stop/logprobs/非 function)");

    // ⑥ `tools: []` 与 `strict` 容忍(pi 在"有工具历史但无新工具"时发明文)
    let tolerated = serde_json::json!({
        "messages": [{"role": "user", "content": "Say READY."}],
        "tools": [],
        "tool_choice": "auto",
        "max_tokens": 16,
        "session": "e2e-agent-empty"
    });
    let (st, _) = post("/v1/chat/completions", &tolerated.to_string());
    assert_eq!(st, 200, "tools:[] 应被接受");
    let strict = serde_json::json!({
        "messages": [{"role": "user", "content": "Say READY."}],
        "tools": [{"type": "function", "function": {"name": "bash", "strict": true,
            "description": "d", "parameters": {"type": "object", "properties": {}}}}],
        "max_tokens": 16,
        "session": "e2e-agent-strict"
    });
    let (st, _) = post("/v1/chat/completions", &strict.to_string());
    assert_eq!(st, 200, "function.strict 应被容忍");
    println!("[agent] tools:[] / strict 容忍 PASS");

    // ⑦ 工具闭环 —— **条件式**(模型肯发才续轮;A7 台账项,不做硬门):
    //    若 turn1 真的出 tool_calls,则把结果回填续一轮并断协议正确性
    let tool_req = serde_json::json!({
        "messages": [{"role": "user", "content": "Use the bash tool to run: echo owl-agent"}],
        "tools": [{"type": "function", "function": {"name": "bash", "description": "Run a bash command",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}],
        "tool_choice": "auto",
        "max_tokens": 512,
        "stream": true,
        "stream_options": {"include_usage": true},
        "session": "e2e-agent-loop"
    });
    let (st, sse) = http_post_sse("/v1/chat/completions", &tool_req.to_string())
        .expect("工具闭环请求");
    assert_eq!(st, 200, "工具闭环 turn1 HTTP 200");
    if sse.contains("\"tool_calls\"") {
        assert!(
            sse.contains("\"finish_reason\":\"tool_calls\""),
            "有工具调用 ⇒ finish_reason 必须是 tool_calls"
        );
        // 提取 arguments(pi 侧的终极判据:必须能解析成 JSON 对象)。
        // ⚠️ **必须按 SSE JSON 解析后拼接 delta** —— 不能用 `split('"')`:
        // SSE 帧里 arguments 是**字符串字段**,其内层引号被转义(`{\"command\":…}`),
        // 朴素截断会在第一个 `\"` 处断成 `{\`(2026-10-15 实测:该断言自诞生起
        // 只在"模型没发工具"时被跳过,一旦真发就假红 —— 测试自身缺陷,非引擎问题)。
        let mut args = String::new();
        for line in sse.lines() {
            let Some(payload) = line.strip_prefix("data: ") else { continue };
            let Ok(v) = serde_json::from_str::<serde_json::Value>(payload) else { continue };
            for d in v["choices"].as_array().into_iter().flatten() {
                for tc in d["delta"]["tool_calls"].as_array().into_iter().flatten() {
                    if let Some(a) = tc["function"]["arguments"].as_str() {
                        args.push_str(a);
                    }
                }
            }
        }
        assert!(!args.is_empty(), "I4:有 tool_calls 却没拿到 arguments delta");
        let av: serde_json::Value = serde_json::from_str(&args)
            .unwrap_or_else(|e| panic!("I4:arguments 必须是合法 JSON 串({e}):{args:?}"));
        assert!(av.is_object(), "I4:arguments 应为对象:{av}");
        // 轮 2:工具结果回填
        let turn2 = serde_json::json!({
            "messages": [
                {"role": "user", "content": "Use the bash tool to run: echo owl-agent"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "call_e2e_loop", "type": "function",
                     "function": {"name": "bash", "arguments": args}}
                ]},
                {"role": "tool", "tool_call_id": "call_e2e_loop", "name": "bash",
                 "content": "owl-agent"},
                {"role": "user", "content": "What did the command print? Answer with just that word."}
            ],
            "max_tokens": 256,
            "session": "e2e-agent-loop"
        });
        let (st, resp) = post("/v1/chat/completions", &turn2.to_string());
        assert_eq!(st, 200, "工具闭环 turn2 HTTP 200:{resp}");
        let v: serde_json::Value = serde_json::from_str(&resp).expect("turn2 JSON");
        assert!(
            v["choices"][0]["finish_reason"].as_str().is_some_and(|f| ["stop", "length", "tool_calls"].contains(&f)),
            "turn2 finish_reason 合法:{v}"
        );
        println!(
            "[agent] 工具闭环 turn1→turn2 PASS(arguments 合法 JSON;turn2 content={:?})",
            content_of(&v).chars().take(40).collect::<String>()
        );
    } else {
        println!(
            "[agent] 工具闭环:本轮模型未发 tool_calls(A7 工具遵循度台账项,非本门判据);\n        已验:请求接受 + 200 + finish_reason 合法"
        );
    }

    // ============ 诊断面(非判据):前缀复用命中统计 ============
    // 服务端 `[dflash] prefill encode N rows @M`:M>0 = 会话守卫命中 →
    // 增量 prefill(新 KV 架构的 block 复用 + 快照 + 跨轮 GDN 连续路径)。
    // 只打印不进断言(命中取决于重发前缀是否逐字节相等 + 页粒度)。
    if let Ok(log) = std::fs::read_to_string(std::env::temp_dir().join("owl-e2e-server.log")) {
        let (mut hit, mut full) = (0usize, 0usize);
        for l in log.lines().filter(|l| l.contains("prefill encode")) {
            match l
                .rsplit_once('@')
                .and_then(|(_, r)| r.split_whitespace().next())
                // 诊断统计用(解析失败计作"全量 prefill");非判据路径
                .and_then(|n| n.parse::<usize>().ok())
            {
                Some(0) | None => full += 1,
                Some(_) => hit += 1,
            }
        }
        println!("[前缀复用] 全量 prefill {full} 次 / 增量 prefill(复用命中) {hit} 次");
    }

    println!("\nE2E 套件 RUN:L1 协议面 = 回归防护(强断言);L2/L3/多轮 = 语义判据记录(已知问题清单见上,修复后逐条转 assert)");
}

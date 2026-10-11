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

/// server 配置(testdata/server/e2e-27b.toml;口径来由见该文件头注)
const CONFIG: &str = include_str!("../../../testdata/server/e2e-27b.toml");

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
    cmd.args(["--flavor", "spec-dflash"])
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
    let mut out = String::new();
    let mut buf = [0u8; 8192];
    loop {
        match s.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                out.push_str(&String::from_utf8_lossy(&buf[..n]));
                if out.contains("data: [DONE]") {
                    break;
                }
            }
            Err(e) => {
                // 已收到部分帧:报错并把现场带出(不许静默吞)
                return Err(format!(
                    "流式读失败({e});已收到 {} 字节: {}",
                    out.len(),
                    out.chars().take(200).collect::<String>()
                ))
            }
        }
    }
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
    Ok((status, out))
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

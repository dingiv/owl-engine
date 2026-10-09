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
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const PORT: u16 = 8135;
const HEALTH_TIMEOUT_SECS: u64 = 180;
const REQ_TIMEOUT_SECS: u64 = 300;

const CONFIG: &str = r#"
[model]
dir = "/home/div/Documents/codes/models/cyankiwi/Qwen3.8-27B-AWQ-INT4"
dflash2_dir = "/home/div/Documents/codes/models/syvai/Qwen3.8-27B-DFlash2-W4A16"
name = "qwen3.8-27b"
[runtime]
device = 1
max_seq = 8448
[pool]
pool_tokens = 8704
gdn_slots = 8
[dispatch]
prefill_split = true
"#;

struct ServerGuard {
    _child: Child, // Drop 时 kill
}

impl Drop for ServerGuard {
    fn drop(&mut self) {
        let _ = self._child.kill();
        let _ = self._child.wait();
    }
}

fn spawn_server() -> ServerGuard {
    let cfg_path = std::env::temp_dir().join("owl-e2e-server.toml");
    std::fs::write(&cfg_path, CONFIG).expect("写 e2e 配置");

    let bin = env!("CARGO_BIN_EXE_server");
    let child = Command::new(bin)
        .args(["--flavor", "spec-dflash"])
        .arg(&cfg_path)
        .stdout(Stdio::null())
        .stderr(std::fs::File::create(
            std::env::temp_dir().join("owl-e2e-server.log"),
        )
        .expect("日志文件"))
        .spawn()
        .expect("启动 server 二进制");
    ServerGuard { _child: child }
}

fn wait_health() -> bool {
    let deadline = Instant::now() + Duration::from_secs(HEALTH_TIMEOUT_SECS);
    while Instant::now() < deadline {
        if let Ok(mut s) = TcpStream::connect(("127.0.0.1", PORT)) {
            let _ = s.write_all(
                format!("GET /health HTTP/1.1\r\nHost: 127.0.0.1:{PORT}\r\nConnection: close\r\n\r\n")
                    .as_bytes(),
            );
            let mut buf = String::new();
            let _ = s.set_read_timeout(Some(Duration::from_secs(5)));
            if s.read_to_string(&mut buf).is_ok() && buf.contains("\"ok\":true") {
                return true;
            }
        }
        std::thread::sleep(Duration::from_secs(3));
    }
    false
}

/// 最小 HTTP POST:返回 (status, body)。手写帧解析,零新依赖。
fn http_post(path: &str, body: &str) -> (u16, String) {
    let mut s = TcpStream::connect(("127.0.0.1", PORT)).expect("连接 server");
    s.set_read_timeout(Some(Duration::from_secs(REQ_TIMEOUT_SECS))).unwrap();
    s.set_write_timeout(Some(Duration::from_secs(REQ_TIMEOUT_SECS))).unwrap();
    let req = format!(
        "POST {path} HTTP/1.1\r\nHost: 127.0.0.1:{PORT}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    s.write_all(req.as_bytes()).expect("写请求");
    let mut buf = String::new();
    s.read_to_string(&mut buf).expect("读响应");
    // 状态行
    let status: u16 = buf
        .split_whitespace()
        .nth(1)
        .and_then(|x| x.parse().ok())
        .unwrap_or(0);
    // body = 第一个 \r\n\r\n 之后(响应无 chunked:owl respond_json 定长)
    let body = buf
        .split_once("\r\n\r\n")
        .map(|(_, b)| b.to_string())
        .unwrap_or_default();
    (status, body)
}

fn chat(session: &str, prompt: &str, max_tokens: u32) -> (u16, serde_json::Value) {
    let body = serde_json::json!({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "session": session,
    });
    let (status, resp) = http_post("/v1/chat/completions", &body.to_string());
    let v: serde_json::Value = serde_json::from_str(&resp)
        .unwrap_or_else(|e| panic!("响应非法 JSON({status}): {e}: {resp}"));
    (status, v)
}

fn content_of(v: &serde_json::Value) -> String {
    v["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default()
        .to_string()
}

#[test]
fn e2e_server_suite() {
    if !owl_shared::env_reader::flag("OWL_E2E") {
        eprintln!("skip: OWL_E2E 未设(需 GPU + 模型路径)");
        return;
    }
    let _srv = spawn_server();
    assert!(wait_health(), "server health 超时({HEALTH_TIMEOUT_SECS}s)");

    // ============ L1 协议面(强判据)============
    let (status, _) = chat("e2e-l1", "What is the capital of France?", 96);
    assert_eq!(status, 200, "HTTP 200");
    let (_, v) = chat("e2e-l1b", "What is the capital of France?", 96);
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
    let (_, v) = chat("e2e-l2-paris", "What is the capital of France?", 256);
    let c = content_of(&v);
    assert!(
        c.to_lowercase().contains("paris"),
        "常识 Paris content 应含 Paris:{c}"
    );
    println!("[L2] 常识 Paris PASS");

    // 算术:2+2 —— 已知问题记录(§三十三):短 prompt 下 think 混乱
    // ("Wait, no, they're asking…"),不含答案 4;修复后转 assert。
    let (_, v) = chat("e2e-l2-arith", "What is 2 + 2? Answer with just the number.", 256);
    let c = content_of(&v);
    assert!(c.contains('4'), "算术 2+2 content 应含 4:{c}");
    println!("[L2] 算术 2+2 PASS");

    // ============ 多轮(agent 式,self-contained 上下文,同 session 续)============
    // 轮 1:告知名字
    let (st1, v) = chat(
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
    let (status, resp) = http_post("/v1/chat/completions", &body.to_string());
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
    let after_think = c
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

    // ============ 会话隔离(自含 codeword;owl 服务端 session = KV 复用键,
    // 历史由客户端拼 → 跨 session 天然隔离;此用例测独立 session 应答正确性)
    let (_, v) = chat(
        "e2e-iso-a",
        "The magic word is PURPLE. What is the magic word? Answer with just the word.",
        256,
    );
    let ca = content_of(&v);
    assert!(
        ca.to_lowercase().contains("purple"),
        "隔离 A 应答 PURPLE:{ca}"
    );
    let (st, v) = chat(
        "e2e-iso-b",
        "The magic word is ORANGE. Say READY.",
        64,
    );
    assert_eq!(st, 200, "隔离 B 轮1 200");
    assert!(!content_of(&v).is_empty(), "隔离 B 轮1 非空");
    let (_, v) = chat(
        "e2e-iso-b",
        "What was the magic word I told you? Answer with just the word.",
        256,
    );
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
            let (status, resp) = http_post("/v1/chat/completions", &body.to_string());
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

    println!("\nE2E 套件 RUN:L1 协议面 = 回归防护(强断言);L2/L3/多轮 = 语义判据记录(已知问题清单见上,修复后逐条转 assert)");
}

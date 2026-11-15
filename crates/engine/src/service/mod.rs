//! **引擎服务面（engine init + 泵 + Send 句柄）** —— 2026-11-15 P1。
//!
//! 用户裁决（2026-11-15）：
//!
//! > 初始化函数初始化之后，直接把一个 engine 实例给它改成 running engine；
//! > 线程由 GPU engine 内部管理，和外面的 server 没有关系。
//!
//! 所以本模块与 `owl-cuda` 的 `GpuServer`/`GpuClient` 同形：
//!
//! ```text
//! EngineService::boot(spec)                  ← 一次调用：选卡/卡锁/装配/装载/run/就绪
//!        │  内部 spawn 一条 OS 线程（RunningEngine 非 Send，建/泵/销毁同线程）
//!        ├── EngineClient（Send + Clone）    ← server 只持它：chat/cancel/tokenize/diag/queue_facts
//!        └── shutdown() → ServiceExit        ← 停通道 + 步边界收摊 + 释放卡锁 + join
//! ```
//!
//! `apps/server` 不再持有：卡锁、engine 线程、ready 信号、单卡/TP 装配、
//! 泵循环策略、每拍遥测。它只剩 HTTP/SSE/协议与进程生命周期。

mod boot;
mod devices;
pub mod diag;
mod pump;

use std::sync::Arc;

use owl_iface::contract::ModelError;
use tokio::sync::mpsc;

use crate::TokenizeReply;
use crate::TurnEvent;
use crate::Result;

pub use boot::{BootSpec, Booted, DeviceOpts, StepGate};
pub use devices::{resolve_devices, Devices};
pub use diag::{BatchStatsSnapshot, CapacityFacts, DiagSnapshot, QueueFacts};

/// 引擎线程栈（与 server 原常量同值：唯一背全模型生命周期的线程，debug 深链
/// 缺省 2MiB 会栈溢出）。
const SERVICE_STACK_BYTES: usize = 32 << 20;

/// **服务端请求**（server → 引擎线程）。
pub enum Request {
    Chat {
        /// None = 临时会话（每请求独立）；Some = 连续会话（增量 prefill）
        session: Option<u64>,
        prompt: String,
        max_new: usize,
        opts: crate::TurnOpts,
        sink: mpsc::Sender<StreamEvent>,
    },
    /// 取消 turn（客户端断流 / 显式 abort）。
    ///
    /// `turn` 已知 = 精确终结；未知 = 按会话找。回执 = 引擎已清完
    /// （服务面据此释放会话闸）。
    Cancel {
        session: u64,
        turn: Option<u64>,
        ack: tokio::sync::oneshot::Sender<()>,
    },
    /// tokenize 旁路（不落 turn、不碰池/图）。
    Tokenize {
        text: String,
        chat_wrap: bool,
        reply: tokio::sync::oneshot::Sender<std::result::Result<TokenizeReply, String>>,
    },
    /// 合批事实旁路（灌流测量前后取差值；零成本）。
    BatchStats {
        reply: tokio::sync::oneshot::Sender<BatchStatsSnapshot>,
    },
    /// 诊断快照（`/debug/vram`；引擎空闲期取，零副作用）。
    Diag {
        reply: tokio::sync::oneshot::Sender<DiagSnapshot>,
    },
}

/// **引擎 → 请求方事件**（首笔必为 Accepted/Rejected，其后 Turn 流）。
#[derive(Clone, Debug)]
pub enum StreamEvent {
    Accepted {
        session: u64,
        turn: u64,
        prompt_tokens: usize,
        /// 前缀命中量（usage `prompt_tokens_details.cached_tokens` 真源）
        cached_tokens: usize,
        /// 实际生成预算（准入夹紧后的值）
        max_new: usize,
    },
    /// 提交边界拒绝（预算/准入）；`err.reject()` 非空 = 准入拒绝，服务面翻 429
    Rejected {
        err: crate::turn::TurnError,
    },
    /// 定相后的真实前缀命中量校正帧（块级命中在 submit 时尚未决定）。
    CachedTokens {
        turn: u64,
        cached_tokens: usize,
    },
    Turn(TurnEvent),
}

/// **服务收摊原因**（决定进程退出码；进程策略仍在 server）。
#[derive(Clone, Debug)]
pub enum ServiceExit {
    /// 请求通道关闭 / 停服信号 ⇒ 正常
    Normal,
    /// 引擎不可恢复错误（`ModelError::is_fatal`）
    Fatal(String),
    /// 设计外 panic（catch_unwind 兜到）
    Panic(String),
}

impl ServiceExit {
    /// `sysexits` 习惯：0 正常 / 1 致命 / 70 软件缺陷
    pub fn code(&self) -> u8 {
        match self {
            Self::Normal => 0,
            Self::Fatal(_) => 1,
            Self::Panic(_) => 70,
        }
    }

    pub fn describe(&self) -> String {
        match self {
            Self::Normal => "正常停服(请求通道关闭)".to_string(),
            Self::Fatal(e) => format!("引擎不可恢复错误:{e}"),
            Self::Panic(e) => format!("owl-engine 线程 panic(设计外 —— 应转显式 Result):{e}"),
        }
    }
}

/// **引擎服务句柄**（server 持它；Drop 不隐式停服 —— 显式 `shutdown`）。
pub struct EngineService {
    client: EngineClient,
    stop: tokio::sync::watch::Sender<bool>,
    join: Option<std::thread::JoinHandle<ServiceExit>>,
}

impl EngineService {
    /// **一步完成 init**：选卡 → 卡锁 → 单卡/TP 装配 → 装载 → run → 就绪。
    ///
    /// 返回即"引擎已就绪"（与 `GpuServer` 的 spawn 同义）；失败 = 结构化 Err
    /// （线程已收摊、卡锁已释放）。线程与泵全部在引擎内部，server 零感知。
    pub fn boot(spec: BootSpec) -> Result<Self> {
        let (req_tx, req_rx) = mpsc::channel::<Request>(spec.channel_capacity.max(1));
        let (stop_tx, stop_rx) = tokio::sync::watch::channel(false);
        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<std::result::Result<Booted, String>>();
        let facts = diag::FactsCell::new();
        let facts_thread = facts.clone();
        let join = std::thread::Builder::new()
            .name("engine-service".into())
            .stack_size(SERVICE_STACK_BYTES)
            .spawn(move || {
                let rt = match tokio::runtime::Builder::new_current_thread().enable_all().build() {
                    Ok(rt) => rt,
                    Err(e) => {
                        let _ = ready_tx.send(Err(format!("engine service runtime: {e}")));
                        return ServiceExit::Fatal(format!("engine service runtime: {e}"));
                    }
                };
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    rt.block_on(service_main(spec, req_rx, stop_rx, &ready_tx, facts_thread))
                }));
                match r {
                    Ok(reason) => reason,
                    Err(p) => ServiceExit::Panic(panic_payload_str(&p)),
                }
            })
            .map_err(|e| ModelError::Msg(format!("engine service 线程创建失败: {e}")))?;
        let client = EngineClient { tx: req_tx, facts };
        match ready_rx.recv() {
            Ok(Ok(_booted)) => Ok(Self { client, stop: stop_tx, join: Some(join) }),
            Ok(Err(e)) => {
                let _ = join.join();
                Err(ModelError::Msg(e))
            }
            // 就绪前通道断开 = 引擎线程已死（panic 已被 catch 成 ServiceExit）
            Err(_) => {
                let reason = join.join().unwrap_or_else(|_| {
                    ServiceExit::Panic("engine service 线程 join 失败".into())
                });
                Err(ModelError::Msg(reason.describe()))
            }
        }
    }

    /// Send + Clone 的客户端（server 的 HTTP 面只持它）。
    pub fn client(&self) -> EngineClient {
        self.client.clone()
    }

    /// **停服**：置停服信号 → 等引擎在**步边界**收摊 → join →（线程内已释放卡锁）。
    pub fn shutdown(mut self) -> ServiceExit {
        let _ = self.stop.send(true);
        match self.join.take() {
            Some(j) => j.join().unwrap_or_else(|_| {
                ServiceExit::Panic("engine-service 线程 join 失败".into())
            }),
            None => ServiceExit::Normal,
        }
    }
}

impl Drop for EngineService {
    /// 兜底：未显式 `shutdown` 就 drop（例如测试提前返回）⇒ **尽快**置停服信号。
    /// 不 join（Drop 里等线程可能死锁在 channel 上）；孤儿线程随进程退出。
    fn drop(&mut self) {
        let _ = self.stop.send(true);
    }
}

/// **请求侧句柄**（Send + Clone；server 的 HTTP worker 人手一份）。
#[derive(Clone)]
pub struct EngineClient {
    tx: mpsc::Sender<Request>,
    facts: Arc<diag::FactsCell>,
}

impl EngineClient {
    /// 提交一个 chat turn（受理结果走 `sink` 的 Accepted/Rejected 帧）。
    pub async fn chat(
        &self,
        session: Option<u64>,
        prompt: String,
        max_new: usize,
        opts: crate::TurnOpts,
        sink: mpsc::Sender<StreamEvent>,
    ) -> Result<()> {
        self.tx
            .send(Request::Chat { session, prompt, max_new, opts, sink })
            .await
            .map_err(|_| ModelError::ServerClosed)
    }

    /// 取消（`turn` 已知 = 精确；未知 = 按会话）。返回时引擎已清完该 turn。
    pub async fn cancel(&self, session: u64, turn: Option<u64>) -> Result<()> {
        let (ack, rx) = tokio::sync::oneshot::channel();
        self.tx
            .send(Request::Cancel { session, turn, ack })
            .await
            .map_err(|_| ModelError::ServerClosed)?;
        rx.await.map_err(|_| ModelError::ServerClosed)
    }

    /// tokenize 旁路（不落 turn）。
    pub async fn tokenize(
        &self,
        text: String,
        chat_wrap: bool,
    ) -> std::result::Result<TokenizeReply, String> {
        let (reply, rx) = tokio::sync::oneshot::channel();
        self.tx
            .send(Request::Tokenize { text, chat_wrap, reply })
            .await
            .map_err(|_| "引擎已退役".to_string())?;
        rx.await.map_err(|_| "引擎连接中断(tokenize 应答通道关闭)".to_string())?
    }

    /// 合批事实（灌流测量）。
    pub async fn batch_stats(&self) -> std::result::Result<BatchStatsSnapshot, String> {
        let (reply, rx) = tokio::sync::oneshot::channel();
        self.tx
            .send(Request::BatchStats { reply })
            .await
            .map_err(|_| "引擎已退役".to_string())?;
        rx.await.map_err(|_| "引擎连接中断(batch stats 应答通道关闭)".to_string())
    }

    /// 诊断快照（`/debug/vram`；引擎空闲期取）。
    pub async fn diag(&self) -> std::result::Result<DiagSnapshot, String> {
        let (reply, rx) = tokio::sync::oneshot::channel();
        self.tx
            .send(Request::Diag { reply })
            .await
            .map_err(|_| "引擎已退役".to_string())?;
        rx.await
            .map_err(|_| "引擎连接中断(vram 账应答通道关闭)".to_string())
    }

    /// 每拍队列事实（零等待；泵循环写）。
    pub fn queue_facts(&self) -> QueueFacts {
        self.facts.read()
    }
}

/// 引擎线程主：选卡/锁/装配/就绪/泵/收尾（全部落在这条线程里）。
async fn service_main(
    spec: BootSpec,
    rx: mpsc::Receiver<Request>,
    stop: tokio::sync::watch::Receiver<bool>,
    ready: &std::sync::mpsc::Sender<std::result::Result<Booted, String>>,
    facts: Arc<diag::FactsCell>,
) -> ServiceExit {
    // ① 选卡（失败 = 结构化 Err 回 boot）
    let devices = match boot::resolve(&spec) {
        Ok(d) => d,
        Err(e) => {
            let _ = ready.send(Err(e.to_string()));
            return ServiceExit::Fatal(e.to_string());
        }
    };
    // ② 卡锁（每 rank 一把；进程 pid 判活，停服显式释放）
    let mut locks = Vec::with_capacity(devices.world());
    for uuid in devices.uuids() {
        let info = owl_shared::gpu_lock::LockInfo::new(
            uuid,
            spec.devices.lock_port,
            spec.devices.lock_role.clone(),
        );
        match owl_shared::gpu_lock::LockGuard::acquire(&info) {
            Ok(g) => locks.push(g),
            Err(e) => {
                release_locks(&locks);
                let _ = ready.send(Err(format!("卡锁失败(拒绝启动): {e}")));
                return ServiceExit::Fatal(format!("卡锁失败(拒绝启动): {e}"));
            }
        }
    }
    owl_shared::log_info!(
        "boot",
        "已锁卡 {} (pid {};停服 = kill -TERM <pid>,自动清锁)",
        devices.uuids().join(" + "),
        std::process::id()
    );
    // ③ 装配 + 装载 + run（单卡/TP 全部在 engine 内）
    let (mut running, booted) = match boot::assemble(&spec, &devices).await {
        Ok(v) => v,
        Err(e) => {
            release_locks(&locks);
            let _ = ready.send(Err(e.to_string()));
            return ServiceExit::Fatal(e.to_string());
        }
    };
    owl_shared::log_info!("boot", "引擎就绪(捕获态 {})", booted.capture);
    let _ = ready.send(Ok(booted));
    // ④ 泵循环（收件/注入/取消/事件路由/压力收缩/每拍事实）
    let mut reason = pump::run(&mut running, rx, spec.perf_dump.clone(), facts, stop).await;
    // ⑤ TP 收尾显式化：peer/host staging 的正规释放口（Drop 只是兜底）
    if running.has_peer() {
        if let Err(e) = running.shutdown_tp().await {
            owl_shared::log_warn!("boot", "TP 收尾失败:{e}");
            if matches!(reason, ServiceExit::Normal) {
                reason = ServiceExit::Fatal(format!("TP 收尾失败:{e}"));
            }
        }
    }
    release_locks(&locks);
    reason
}

fn release_locks(locks: &[owl_shared::gpu_lock::LockGuard]) {
    for g in locks {
        owl_shared::gpu_lock::release(g.path());
    }
    if !locks.is_empty() {
        owl_shared::log_info!("gpu-lock", "卡锁已全部释放({} 把)", locks.len());
    }
}

fn panic_payload_str(p: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = p.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = p.downcast_ref::<String>() {
        s.clone()
    } else {
        "(非字符串 panic 载荷)".to_string()
    }
}

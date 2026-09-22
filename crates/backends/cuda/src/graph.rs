//! GraphLease:捕获会话 + 定影后的图(A1.2 生命周期律代码化)。
//!
//! M1:`capture()` 把四步合一——
//! 1. **姿势 6 门禁**:本设备须先完成过 eager 发射(warmup 制度化);
//! 2. **signal 自动租约**:闭包内 ops 读过的缓冲自动成为强租约
//!    (设计稿 docs/arch/signal-dependency-design.md §2.1,免手工登记);
//! 3. **哨兵③ 审计**:instantiate 前对 driver 真相对账(见 `audit.rs`);
//! 4. 定影:租约 + keepalive 进 [`DeviceGraph`],replay 前逐租约世代校验。
//!
//! DeviceGraph 持**裸句柄**(exec + 源图):源图留作姿势 8
//! (ExecUpdate 热切档位)的 M1+ 弹药。

use super::audit::{audit_graph, AuditReport};
use super::governor::Governor;
use super::pool::PoolBufInner;
use crate::buffers::Persistent;
use crate::ffi::{sys, graph_destroy, graph_exec_destroy, graph_instantiate, graph_launch, stream_end_capture};
use cudarc::driver::CudaStream;
use owl_iface::{BackendError, BufToken, MemValue};
use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

/// 捕获帧:捕获闭包接收的执行上下文——捕获流(E 阶段发射目标)+
/// 会话租约 sink(signal 自动登记通道)。
pub struct CaptureFrame<'s> {
    pub stream: &'s Arc<CudaStream>,
    pub lease_sink: owl_signal::Sink,
}

/// 捕获会话:强租约 keepalive + 会话流 + 治理句柄。
pub struct CaptureSession {
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) leases: Vec<BufToken>,
    pub(crate) keepalive: Vec<Arc<PoolBufInner>>,
    pub(crate) gov: Arc<Governor>,
}

impl CaptureSession {
    /// 手工登记依赖缓冲(自动租约的逃生口:非 ops 触碰的缓冲,如
    /// 仅被原始 FFI 引用的 VMM 缓冲)。Arc 克隆入 keepalive(强租约,
    /// 用户侧句柄先 drop 也不回收)。
    pub fn lease<T: MemValue>(&mut self, t: &Persistent<T>) {
        let (buf, token) = t.lease_parts();
        if let Some(tok) = token {
            if !self.leases.iter().any(|l| l.id == tok.id) {
                self.leases.push(tok);
            }
        }
        // 从带 Drop 的类型不能移动字段:克隆 Arc 后弃外壳
        let inner = buf.inner.clone();
        drop(buf);
        self.keepalive.push(inner);
    }

    /// 捕获用流(non-blocking;kernel 直发目标)
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// 一次捕获(四步合一,见模块头)。闭包内经 [`CaptureFrame`] 发射的
    /// 全部 kernel 参数引用自动对账;读过的缓冲自动租约。
    pub fn capture<R, F>(
        &mut self,
        flags: sys::CUgraphInstantiate_flags,
        f: F,
    ) -> Result<(R, DeviceGraph), BackendError>
    where
        F: FnOnce(&CaptureFrame<'_>) -> Result<R, BackendError>,
    {
        // 姿势 6:effect 未完整执行过一次不可 seal(warmup 制度化)
        if self.gov.eager_launches() == 0 {
            return Err(BackendError::LawViolation(
                "姿势 6:捕获前本设备须完成至少一次 eager 发射(warmup;JIT/懒状态清理)",
            ));
        }

        self.stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            .map_err(|e| BackendError::Init(format!("begin_capture: {e:?}")))?;

        // signal 作用域:闭包内 owl_signal::emit 自动汇聚为依赖集(哨兵①自动化)
        let toks: Arc<Mutex<Vec<BufToken>>> = Arc::new(Mutex::new(Vec::new()));
        let sink: owl_signal::Sink = Arc::new({
            let t = Arc::clone(&toks);
            move |tok: owl_signal::Token| {
                t.lock().expect("lease sink 中毒").push(tok);
            }
        });
        let _guard = owl_signal::enter(sink.clone());
        let frame = CaptureFrame {
            stream: &self.stream,
            lease_sink: sink,
        };
        let r = f(&frame);
        drop(_guard);

        let ctx = self.gov.ctx.as_ref().expect("ctx 缺失");
        ctx.bind_to_thread()
            .map_err(|e| BackendError::Init(format!("bind_to_thread: {e:?}")))?;

        let result: Result<(R, DeviceGraph), BackendError> = match r {
            Ok(r) => {
                let cu_graph = unsafe { stream_end_capture(self.stream.cu_stream()) }
                    .map_err(|e| BackendError::Init(format!("end_capture: {e:?}")))?;

                // 自动租约:闭包读过的每个令牌 → 强租约(Weak 登记表升级)
                let dead = {
                    let live = self.gov.live_bufs.lock();
                    let collected = std::mem::take(&mut *toks.lock().expect("toks 中毒"));
                    let mut dead = Vec::new();
                    for t in collected {
                        if self.leases.iter().any(|l| l.id == t.id) {
                            continue;
                        }
                        if !self.gov.validate(&t) {
                            dead.push(t);
                            continue;
                        }
                        match live.get(&t.id).and_then(|w| w.upgrade()) {
                            Some(inner) => {
                                self.keepalive.push(inner);
                                self.leases.push(t);
                            }
                            None => dead.push(t),
                        }
                    }
                    dead
                };
                if !dead.is_empty() {
                    unsafe {
                        let _ = graph_destroy(cu_graph);
                    }
                    eprintln!(
                        "[owl-graph] A1.7 违约细节:定影前死亡的令牌 = {:?} (P 阶段缓冲生命周期必须 ≥ 捕获窗口)",
                        dead
                    );
                    return Err(BackendError::LawViolation(
                        "A1.7:捕获依赖缓冲已在定影前死亡(P 阶段缓冲生命周期必须 ≥ 捕获窗口;细节见日志)",
                    ));
                }

                // 哨兵③:instantiate 前对 driver 真相对账(违约即毁图报错)
                let lease_ids: BTreeSet<u64> = self.leases.iter().map(|t| t.id).collect();
                let audit = match audit_graph(cu_graph, &self.gov, &lease_ids) {
                    Ok(a) => a,
                    Err(e) => {
                        unsafe {
                            let _ = graph_destroy(cu_graph);
                        }
                        return Err(e);
                    }
                };
                if !audit.suspicious.is_empty() {
                    eprintln!(
                        "[owl-graph] WARN: 嫌疑 {:?} | 指针引用 {:?}",
                        audit.suspicious.iter().map(|v| format!("0x{:x}", v)).collect::<Vec<_>>(),
                        audit.ptr_refs.iter().map(|v| format!("0x{:x}", v)).collect::<Vec<_>>(),
                    );
                }

                let exec = unsafe { graph_instantiate(cu_graph, flags) }
                    .map_err(|e| BackendError::Init(format!("instantiate: {e:?}")))?;
                Ok((
                    r,
                    DeviceGraph {
                        cu_graph,
                        exec,
                        stream: Arc::clone(&self.stream),
                        leases: std::mem::take(&mut self.leases),
                        keepalive: std::mem::take(&mut self.keepalive),
                        gov: Arc::clone(&self.gov),
                        audit,
                    },
                ))
            }
            Err(e) => {
                // 闭包失败:结束并丢弃捕获(取回图即销毁)
                if let Ok(g) = unsafe { stream_end_capture(self.stream.cu_stream()) } {
                    unsafe {
                        let _ = graph_destroy(g);
                    }
                }
                Err(e)
            }
        };
        result
    }
}

/// 定影后的设备图:持有全部依赖租约;replay(debug 构建)前逐租约校验,
/// 失效 = 结构化报错而非 Xid 盲死。
pub struct DeviceGraph {
    cu_graph: sys::CUgraph,
    exec: sys::CUgraphExec,
    stream: Arc<CudaStream>,
    leases: Vec<BufToken>,
    /// 强租约:持有即语义——图存活期钉住依赖缓冲的物理页,无需读取;
    /// drop 时 Arc 计数回落,回收流程自然解封。
    #[allow(dead_code)]
    keepalive: Vec<Arc<PoolBufInner>>,
    gov: Arc<Governor>,
    audit: AuditReport,
}

impl DeviceGraph {
    /// 预上传到设备(可选优化;launch 前一次)
    pub fn upload(&self) -> Result<(), BackendError> {
        unsafe {
            use sys::cuGraphUpload;
            if cuGraphUpload(self.exec, self.stream.cu_stream()) != sys::CUresult::CUDA_SUCCESS {
                return Err(BackendError::Init("graph upload".into()));
            }
        }
        Ok(())
    }

    /// 重放(租约校验 → cuGraphLaunch;E 阶段零抽象,一次 FFI)
    pub fn launch(&self) -> Result<(), BackendError> {
        #[cfg(debug_assertions)]
        for t in &self.leases {
            if !self.gov.validate(t) {
                return Err(BackendError::LawViolation(
                    "A1.2 图依赖令牌失效(replay 前校验;详见租约表)",
                ));
            }
        }
        unsafe { graph_launch(self.exec, self.stream.cu_stream()) }
            .map_err(|e| BackendError::Init(format!("graph launch: {e:?}")))
    }

    /// 租约表(依赖令牌;哨兵③/测试对账用)
    pub fn leases(&self) -> &[BufToken] {
        &self.leases
    }

    /// 定影时的 driver 自审报告(哨兵③)
    pub fn audit(&self) -> &AuditReport {
        &self.audit
    }
}

impl Drop for DeviceGraph {
    fn drop(&mut self) {
        unsafe {
            if !self.exec.is_null() {
                let _ = graph_exec_destroy(self.exec);
            }
            if !self.cu_graph.is_null() {
                let _ = graph_destroy(self.cu_graph);
            }
        }
    }
}

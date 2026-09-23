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
use crate::buffers::{Persistent, Scratch};
use crate::ffi::{sys, graph_destroy, graph_exec_destroy, graph_instantiate, graph_launch, stream_end_capture};
use cudarc::driver::CudaStream;
use owl_iface::{BackendError, BufToken, MemPhase, MemValue};
use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

/// 捕获帧:捕获闭包接收的执行上下文——捕获流(E 阶段发射目标)+
/// 会话租约 sink(signal 自动登记通道)。
pub struct CaptureFrame<'s> {
    pub stream: &'s Arc<CudaStream>,
    pub lease_sink: owl_iface::signal::Sink,
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
        // 从带 Drop 的类型不能移动字段：克隆 Arc 后弃外壳
        let inner = buf.inner.clone();
        drop(buf);
        self.keepalive.push(inner);
    }

    /// 手工登记暂存域依赖缓冲(与 [`CaptureSession::lease`] 对称；原始 FFI
    /// 引用的 pre-window scratch 块用，哨兵③对账需要它在租约表)
    pub fn lease_scratch<T: MemValue>(&mut self, t: &Scratch<T>) {
        let (buf, token) = t.lease_parts();
        if let Some(tok) = token {
            if !self.leases.iter().any(|l| l.id == tok.id) {
                self.leases.push(tok);
            }
        }
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

        // P2 接线:捕获窗口 = Capturing 相(延迟释放生效)。未 lease 的缓冲
        // 在窗口内 drop 不再即时 free/unmap,图指针不会悬空;窗口结束恢复原相。
        struct PhaseGuard<'a>(&'a Governor);
        impl Drop for PhaseGuard<'_> {
            fn drop(&mut self) {
                self.0.restore_phase();
            }
        }
        let _phase = {
            self.gov.push_phase(MemPhase::Capturing);
            PhaseGuard(&self.gov)
        };
        // P1-2:出生钉住账水位线(本窗口收编 [mark..);失败时从 mark 弃置
        let born_mark = self.gov.capture_born_mark();

        self.stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            .map_err(|e| BackendError::Init(format!("begin_capture: {e:?}")))?;

        // signal 作用域:闭包内 owl_iface::signal::emit 自动汇聚为依赖集(哨兵①自动化)
        //
        // 关键时序修正(2026-09-22):强租约升级必须在 **emit 时刻**完成——
        // functional 风格的 forward 中间量是语句级瞬态,forward 返回时早已
        // drop(retire)。"f 结束后统一升级"永远只见到尸体(A1.7 假违约)。
        // emit 时缓冲必然存活 → 当场 Weak 升级 Arc 入 keepalive 暂存。
        let toks: Arc<Mutex<Vec<BufToken>>> = Arc::new(Mutex::new(Vec::new()));
        let keepalive_new: Arc<Mutex<Vec<Arc<PoolBufInner>>>> =
            Arc::new(Mutex::new(Vec::new()));
        let sink: owl_iface::signal::Sink = Arc::new({
            let t = Arc::clone(&toks);
            let ka = Arc::clone(&keepalive_new);
            let gov = std::sync::Arc::clone(&self.gov);
            move |tok: owl_iface::signal::Token| {
                t.lock().expect("lease sink 中毒").push(tok);
                // 当场升级(emit 时缓冲必活);重复 emit 幂等(dedup 在合并段)
                if let Some(inner) = gov.live_bufs.lock().get(&tok.id).and_then(|w| w.upgrade()) {
                    ka.lock().expect("ka 中毒").push(inner);
                }
            }
        });
        let _guard = owl_iface::signal::enter(sink.clone());
        let frame = CaptureFrame {
            stream: &self.stream,
            lease_sink: sink,
        };
        // P0-1(audit 2026-09-23):闭包 panic 原本直接跳栈,捕获流永久悬死
        // (未 end_capture → 设备后续发射全失效)。兜底:panic 转结构化
        // Error,end_capture + 毁图,捕获流复位。
        let r = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(&frame))) {
            Ok(r) => r,
            Err(payload) => {
                let msg = if let Some(s) = payload.downcast_ref::<&str>() {
                    (*s).to_string()
                } else if let Some(s) = payload.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "<非字符串 panic payload>".to_string()
                };
                if let Ok(g) = unsafe { stream_end_capture(self.stream.cu_stream()) } {
                    unsafe {
                        let _ = graph_destroy(g);
                    }
                }
                // P1:捕获失败,本轮租约/保活全部作废(它们只服务于本次捕获);
                // P1-2:窗口钉住账同步清理(borns 归零落 park,统一执行释放)
                self.leases.clear();
                self.keepalive.clear();
                self.gov.discard_capture_state(born_mark);
                return Err(BackendError::Init(format!(
                    "capture: 闭包 panic(已兜底:捕获流已复位、图已销毁): {msg}"
                )));
            }
        };
        drop(_guard);

        let ctx = self.gov.ctx.as_ref().expect("ctx 缺失");
        ctx.bind_to_thread()
            .map_err(|e| BackendError::Init(format!("bind_to_thread: {e:?}")))?;

        let result: Result<(R, DeviceGraph), BackendError> = match r {
            Ok(r) => {
                let cu_graph = unsafe { stream_end_capture(self.stream.cu_stream()) }
                    .map_err(|e| BackendError::Init(format!("end_capture: {e:?}")))?;

                // P1-2 收编①:窗口内出生的块(birth-pin 的强引用)——它们的地址
                // 可能已烙进图,统一入图 keepalive(图存活期钉住物理页);
                // 令牌同步入租约表(哨兵③把它们当合法依赖对账 + replay 校验)
                let borns = self.gov.take_capture_borns_from(born_mark);
                for inner in borns {
                    let tok = inner.token.clone();
                    if !self.leases.iter().any(|l| l.id == tok.id) {
                        self.leases.push(tok);
                    }
                    if !self.keepalive.iter().any(|k| Arc::ptr_eq(k, &inner)) {
                        self.keepalive.push(inner);
                    }
                }

                // 自动租约:emit 期已升级的强引用合并进会话 keepalive;
                // 令牌在 emit 后、定影前死亡的(真 P 阶段违规)仍被 validate 拦截
                let dead = {
                    let live = self.gov.live_bufs.lock();
                    let collected = std::mem::take(&mut *toks.lock().expect("toks 中毒"));
                    let upgraded: Vec<Arc<PoolBufInner>> =
                        std::mem::take(&mut *keepalive_new.lock().expect("ka 中毒"));
                    self.keepalive.extend(upgraded);
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
                    self.gov.discard_capture_state(born_mark);
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
                        self.gov.discard_capture_state(born_mark);
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

                // P0-2(audit 2026-09-23,口径修正):拒绝的是 **leaked**
                // (命中池区间但 ∉ 租约 = 依赖泄漏,回放必读悬空);
                // suspicious(像地址的标量/host 指针,首跑实锤 0x3f800000=1.0f)
                // 保持告警——误报面在 kernel 参数含立即数,无法与指针盲区分。
                let leaked_unlisted: Vec<u64> = audit
                    .leaked
                    .iter()
                    .filter(|v| !audit.allowlist.contains(v))
                    .copied()
                    .collect();
                if !leaked_unlisted.is_empty() {
                    unsafe {
                        let _ = graph_destroy(cu_graph);
                    }
                    let detail = format!(
                        "哨兵③:图引用未租约池指针,拒绝定影(泄漏指针 = {:?};指针引用 = {:?};豁免请填 AuditReport.allowlist)",
                        leaked_unlisted
                            .iter()
                            .map(|v| format!("0x{v:x}"))
                            .collect::<Vec<_>>(),
                        audit.ptr_refs.iter().map(|v| format!("0x{v:x}")).collect::<Vec<_>>(),
                    );
                    self.gov.discard_capture_state(born_mark);
                    return Err(BackendError::LawViolation(Box::leak(detail.into_boxed_str())));
                }
                if !audit.suspicious.is_empty() {
                    eprintln!(
                        "[owl-graph] WARN(嫌疑,非违约): {:?}",
                        audit.suspicious.iter().map(|v| format!("0x{v:x}")).collect::<Vec<_>>(),
                    );
                }

                let exec = unsafe { graph_instantiate(cu_graph, flags) }
                    .map_err(|e| BackendError::Init(format!("instantiate: {e:?}")))?;
                // P1-2 收编②:窗口内死亡的块(pre-window 出生、释放闭包停放)——
                // 图 drop 时才真正释放(销毁 exec 之后);取出时机 = 定影
                // 在手,失败路径已全部覆盖 discard
                let parked = self.gov.take_capture_parked();
                Ok((
                    r,
                    DeviceGraph {
                        cu_graph,
                        exec,
                        stream: Arc::clone(&self.stream),
                        leases: std::mem::take(&mut self.leases),
                        keepalive: std::mem::take(&mut self.keepalive),
                        parked,
                        gov: Arc::clone(&self.gov),
                        audit,
                    },
                ))
            }
            Err(e) => {
                // 闭包失败:结束并丢弃捕获(取回图即销毁);P1:同时作废本轮租约;
                // P1-2:窗口钉住账清理(borns 归零落 park,统一执行释放)
                if let Ok(g) = unsafe { stream_end_capture(self.stream.cu_stream()) } {
                    unsafe {
                        let _ = graph_destroy(g);
                    }
                }
                self.leases.clear();
                self.keepalive.clear();
                self.gov.discard_capture_state(born_mark);
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
    /// drop 时 Arc 计数回落,回收流程自然解封。含 P1-2 窗口出生块。
    #[allow(dead_code)]
    keepalive: Vec<Arc<PoolBufInner>>,
    /// P1-2:窗口内死亡块的停放释放闭包——图 drop(销毁 exec 之后)才
    /// 执行;图存活期这些地址被图引用,提前释放 = 回放悬空
    parked: Vec<super::governor::DeferredFree>,
    /// 治理句柄(replay 前租约校验通道;P1 后全构建生效)
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
        // P1(audit):租约校验不再限 debug 构建——validate 是纯指针账本
        // 查询,成本可忽略;失效 = 结构化报错而非 Xid 盲死(release 更需要)
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
        // P1-2:窗口内死亡块的停放释放(exec 已销毁,图不再引用;先同步
        // 防末次 replay 在飞——release_backing 对 VMM 自带 ctx sync)
        let _ = self.gov.ctx.as_ref().map(|c| c.synchronize());
        for f in std::mem::take(&mut self.parked) {
            f();
        }
    }
}

//! DMA 收割臂(htod/dtoh/pinned 码头/分块流水)。码头池化(S1)、
//! COMPUTE 排空纪律(塔零案定谳)集中于此。`impl GpuServer` 跨文件
//! (server 巨石拆分 P1.4,2026-10-03)。

use super::*;

impl GpuServer {
    /// **notify 失败留痕**(2026-10-15 审查补):`launch_host_function` 失败 =
    /// 回执断链 —— 旧实现三处 `ack.take().unwrap()` 必 panic(server 线程猝死
    /// ⇒ 引擎死),一处 `let _ = e` 让客户端 future **永挂**。现在统一:失败
    /// 回执走 `Arc<Mutex<Option<Ack>>>` cell(与闭包二选一原子成立),并在此
    /// 留 counter + LVL_ERROR(不再静默)。
    fn notify_failed(&self, what: &str, e: &ModelError) {
        owl_shared::counter_inc!("srv.notify_fail");
        self.sink.logf(
            owl_shared::log::LVL_ERROR,
            "srv",
            format_args!("{what} notify 失败(回执断链): {e:?}"),
        );
    }


    /// srv.seg.htod 分相(与客户端同名段对账;差值 = 管道/等待)
    pub(super) fn handle_htod(&mut self, data: Vec<u8>, name: String, ack: Ack<Result<Bytes, ModelError>>) {
        let t0 = std::time::Instant::now();
        self.handle_htod_inner(data, &name, ack);
        self.sink.timer("srv.seg.htod", t0.elapsed());
    }

    pub(super) fn handle_htod_inner(&mut self, data: Vec<u8>, name: &str, ack: Ack<Result<Bytes, ModelError>>) {
        let mut ack = Some(ack);
        let n = data.len();
        let stream = match self.ctx().stream(STREAM_H2D) {
            Ok(s) => s.clone(),
            Err(e) => return ack.take().unwrap().send(Err(e)),
        };
        match self.try_htod(&stream, &data, name) {
            Ok((id, buf)) => {
                // 发射账:上传**落定**记 id + 设备地址(A13 类"跨流 UAF"的钥匙:
                // 该地址的 COMPUTE 读者是否在本次落定**之前**就已发射)
                if let Some(tr) = self.trace.as_ref() {
                    let ptr = self.ctx().block_ptr(id, &stream).map(|(p, _)| p).unwrap_or(0);
                    tr.push(
                        crate::trace::TraceEntry::new("htod_landed", "h2d")
                            .bytes(n)
                            .id(id)
                            .ptrs([ptr]),
                    );
                }
                // 码头随 finish 存活至搬运完成后归还池(派发线程执行)。
                // ack 住 cell:闭包投递与失败回执**二选一**原子成立
                // (旧 `let mut cb_ack = ack.take()` + 错误分支 `ack.take().unwrap()`
                //  = 必 panic;2026-10-15 审查定谳)
                let ack_cell = std::sync::Arc::new(std::sync::Mutex::new(ack.take()));
                let pool = self.pool.clone();
                let cell = ack_cell.clone();
                let finish: Finish = Box::new(move || {
                    pool.put(buf);
                    if let Some(a) = cell.lock().unwrap_or_else(|p| p.into_inner()).take() {
                        // 事实面由客户端按 (dtype, shape) 定形(`htod` 回执后 patch);
                        // 服务端此处只给**句柄**(字节口径 U8 + rank 0 = 未定形)
                        a.send(Ok(Bytes::new(id, n, Dtype::U8, 0)));
                    }
                });
                if let Err(e) = self.notify(&stream, finish) {
                    if let Some(a) = ack_cell.lock().unwrap_or_else(|p| p.into_inner()).take() {
                        a.send(Err(e.clone()));
                    }
                    self.notify_failed("htod", &e);
                }
            }
            Err(e) => ack.take().unwrap().send(Err(e)),
        }
    }

    /// htod issue 相(非阻塞):设备块 + pinned 码头 + 异步 memcpy。
    /// 返回 (块 id, 码头) —— 码头随 finish 存活至搬运完成。
    pub(super) fn try_htod(
        &mut self,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        data: &[u8],
        name: &str,
    ) -> Result<(u64, Box<dyn owl_iface::contract::PinnedRegion + Send>), ModelError> {
        let n = data.len();
        // 1. 设备块(流序 malloc;未初始化;字节口径)
        let dst = unsafe { stream.alloc::<u8>(n) }
            .map_err(|e| ModelError::Msg(format!("htod alloc: {e:?}")))?;
        let id = self.ctx_mut().new_block(dst, name);
        // 2. pinned 码头 + 异步 memcpy(流序;host 侧必须 pinned 才真异步)。
        //    码头池化(2026-09-26 尾批:原每次 malloc_host 新分配,大权重
        //    锁页分配是装载耗时大头;finish 回调归还池)
        let mut buf: Box<dyn owl_iface::contract::PinnedRegion + Send> =
            match self.pool.take(n) {
                Some(b) => b,
                None => Box::new(Staging::alloc(n, "try_htod")?),
            };
        buf.slice_bytes_mut()[..n].copy_from_slice(data);
        // **清零序**(2026-10-16):H2D 写入设备块前,先等该块的 `alloc{zero}` memset
        // 落地 —— 两流无序时晚到的 memset 会抹掉刚落地的搬运(实测 A_log 32B)。
        self.ctx_mut().wait_zeroing(id, STREAM_H2D)?;
        let (dptr, _) = self.ctx().block_ptr(id, stream)?;
        // 池块容量 ≥ 请求:DMA 长度必须截到 n(越界写设备块 = INVALID_VALUE)
        unsafe { memcpy_htod_async(dptr, &buf.as_bytes()[..n], stream.cu_stream()) }
            .map_err(|e| ModelError::Msg(format!("htod async: {e:?}")))?;
        // 诊断开关(OWL_LAUNCH_SYNC=1):DMA 后同步归因(H2D 错误平时
        // 无条件 Ok 回执,sticky 晚冒 —— 2026-09-27 排查补)
        if self.diag.sync_at(owl_shared::config::sync_bit::HTOD) {
            stream
                .synchronize()
                .map_err(|e| ModelError::Msg(format!("htod-sync(n={n}): {e:?}")))?;
        }
        Ok((id, buf))
    }

    /// 分配 pinned 租约：池优先，miss 才 cudaHostAlloc。
    /// 不填零（2026-09-26 装载定谳）：租约语义 = 调用方整块覆写后再按
    /// len_bytes DMA，填零纯属 server 单线程上的双倍带宽税。
    pub(super) fn handle_alloc_pinned(
        &mut self,
        bytes: usize,
        name: String,
        ack: Ack<Result<Box<dyn owl_iface::contract::PinnedRegion + Send>, ModelError>>,
    ) {
        // 池取（内部单次加锁；容量 ≥ 请求即命中，逻辑长度由调用方界定）
        if let Some(b) = self.pool.take(bytes) {
            // 容量不可截，整块交出（调用方只用前 bytes）
            owl_shared::counter_inc!("srv.pinned_hit");
            ack.send(Ok(b));
            return;
        }
        // miss 才 cudaHostAlloc（页锁定；大块 ~300ms —— 装载期主要成本之一）
        owl_shared::counter_inc!("srv.pinned_miss");
        owl_shared::counter_add!("srv.pinned_miss_bytes", bytes as u64);
        match owl_shared::timer_scope!("srv.pinned_alloc", Staging::alloc(bytes, &name)) {
            Ok(s) => ack.send(Ok(Box::new(s))),
            Err(e) => ack.send(Err(e)),
        }
    }

    /// 上传租约(命令壳):`enqueue_upload_pinned` + 回执
    pub(super) fn handle_upload_pinned(
        &mut self,
        buf: Box<dyn owl_iface::contract::PinnedRegion + Send>,
        dst: Bytes,
        offset_bytes: usize,
        len_bytes: usize,
        ack: Ack<Result<(), ModelError>>,
    ) {
        ack.send(self.enqueue_upload_pinned(buf, dst, offset_bytes, len_bytes));
    }

    /// **上传核心**(只入队 DMA + 登记租约归还;不 ack)—— 复合命令
    /// `UploadAndLaunch` 与命令壳共用(mutual: 单机制,无第二套上传路径)
    pub(super) fn enqueue_upload_pinned(
        &mut self,
        buf: Box<dyn owl_iface::contract::PinnedRegion + Send>,
        dst: Bytes,
        offset_bytes: usize,
        len_bytes: usize,
    ) -> Result<(), ModelError> {
        self.enqueue_upload_pinned_on(STREAM_H2D, buf, dst, offset_bytes, len_bytes)
    }

    /// 同上,但**指定流**(复合命令 `UploadAndLaunch` 用:把搬运放在**消费它的
    /// 那条流**上 ⇒ 单流硬件保序,零事件零栅栏 —— 见 server 侧复合处理体)
    pub(super) fn enqueue_upload_pinned_on(
        &mut self,
        stream_id: crate::state::StreamId,
        buf: Box<dyn owl_iface::contract::PinnedRegion + Send>,
        dst: Bytes,
        offset_bytes: usize,
        len_bytes: usize,
    ) -> Result<(), ModelError> {
        let stream = match self.ctx().stream(stream_id) {
            Ok(s) => s.clone(),
            Err(e) => return Err(e),
        };
        // **清零序**(同 try_htod 注):目标块可能是 alloc{zero} 刚给的
        self.ctx_mut().wait_zeroing(dst.id, stream_id)?;
        let dptr_base = match self.ctx().block_ptr(dst.id, &stream) {
            Ok((p, _)) => p,
            Err(e) => return Err(e),
        };
        // 边界校验(2026-10-15 补口;装载期流式写权重走本路)
        if let Err(e) = self
            .ctx()
            .check_range(dst.id, offset_bytes, len_bytes, "upload-pinned")
        {
            self.pool.put(buf);
            return Err(e);
        }
        let t_dma = std::time::Instant::now();
        let dma = unsafe {
            memcpy_htod_async(
                dptr_base + (offset_bytes as u64),
                &buf.as_bytes()[..len_bytes],
                stream.cu_stream(),
            )
        };
        owl_shared::timer_record!("srv.dma_enqueue", t_dma.elapsed());
        owl_shared::counter_add!("srv.dma_bytes", len_bytes as u64);
        if let Err(e) = dma {
            // 建场失败:未入队任何 DMA,finish 不得投递 —— 租约回池 + 返回错误
            let e = ModelError::Msg(format!("upload_pinned async: {e:?}"));
            self.pool.put(buf);
            return Err(e);
        }
        // 异步语义(F5 尾批,用户裁决):入队即回执 —— DMA 完成由调用方
        // 的显式 sync 栅栏兜底;finish 只归还租约入池(页锁定代价不再
        // 逐块支付,池真正转起来)。
        let cell = std::sync::Arc::new(std::sync::Mutex::new(Some(buf)));
        let cell2 = cell.clone();
        let pool = self.pool.clone();
        let pool_err = self.pool.clone();
        let finish: Finish = Box::new(move || {
            let buf = cell2.lock().unwrap().take().unwrap();
            pool.put(buf);
        });
        if let Err(e) = self.notify(&stream, finish) {
            // notify 失败:finish 未投递,租约回收入池(ack 已回,
            // DMA 未发生 —— 块内容缺失由调用方末端 sync 后自查/重试兜底)
            if let Some(buf) = cell.lock().unwrap().take() {
                pool_err.put(buf);
                self.sink.logf(owl_shared::log::LVL_DEBUG, "srv", format_args!("upload notify 失败: {e:?}"));
            }
        }
        Ok(()) // 入队即返回(DMA 完成由调用方栅栏/事件序兜底)
    }

    /// 分块写入已 alloc 的块(流式装载):pinned 码头 + memcpyAsync
    /// 到 dst+offset;完成回调回执
    /// 批量分块写入(S1):逐写池化码头 + H2D 流序 memcpy,末尾单
    /// notify —— 一命令一往返,保序协议不变(ack = 全部写完)
    pub(super) fn handle_htod_chunks(
        &mut self,
        writes: Vec<(u64, usize, Vec<u8>)>,
        ack: Ack<Result<(), ModelError>>,
    ) {
        let mut ack = Some(ack);
        let stream = match self.ctx().stream(STREAM_H2D) {
            Ok(s) => s.clone(),
            Err(e) => return ack.take().unwrap().send(Err(e)),
        };
        let mut pending: Vec<Box<dyn owl_iface::contract::PinnedRegion + Send>> =
            Vec::with_capacity(writes.len());
        for (block, offset_bytes, data) in &writes {
            // 边界校验(2026-10-15 补口;S1 批量分块写:decode 每步装填)
            if let Err(e) = self
                .ctx()
                .check_range(*block, *offset_bytes, data.len(), "htod-chunks")
            {
                for b in pending {
                    self.pool.put(b);
                }
                return ack.take().unwrap().send(Err(e));
            }
            // **清零序**(2026-10-16):见 try_htod 注(H2D 写前等该块 memset)
            if let Err(e) = self.ctx_mut().wait_zeroing(*block, STREAM_H2D) {
                for b in pending {
                    self.pool.put(b);
                }
                return ack.take().unwrap().send(Err(e));
            }
            let dptr_base = match self.ctx().block_ptr(*block, &stream) {
                Ok((p, _)) => p,
                Err(e) => {
                    for b in pending {
                        self.pool.put(b);
                    }
                    return ack.take().unwrap().send(Err(e));
                }
            };
            let mut staging: Box<dyn owl_iface::contract::PinnedRegion + Send> =
                match self.pool.take(data.len()) {
                    Some(b) => b,
                    None => match Staging::alloc(data.len(), "htod_chunks") {
                        Ok(s) => Box::new(s),
                        Err(e) => {
                            for b in pending {
                                self.pool.put(b);
                            }
                            return ack.take().unwrap().send(Err(e));
                        }
                    },
                };
            staging.slice_bytes_mut()[..data.len()].copy_from_slice(data);
            if let Err(e) = unsafe {
                memcpy_htod_async(
                    dptr_base + (*offset_bytes as u64),
                    &staging.as_bytes()[..data.len()],
                    stream.cu_stream(),
                )
            }
            .map_err(|e| ModelError::Msg(format!("htod chunks async: {e:?}")))
            {
                pending.push(staging);
                for b in pending {
                    self.pool.put(b);
                }
                return ack.take().unwrap().send(Err(e));
            }
            pending.push(staging);
        }
        // 单 notify:全部写到位后一次性回执(保序 = H2D 流序)。
        // ⚠️ 旧实现在此 **`let _ = e;`** —— ack 已随闭包移交 ⇒ 客户端 future
        // **永远 pending**(无 panic、无日志)。现统一走 cell + 留痕。
        let ack_cell = std::sync::Arc::new(std::sync::Mutex::new(ack.take()));
        let pool = self.pool.clone();
        let cell = ack_cell.clone();
        let finish: Finish = Box::new(move || {
            for b in pending {
                pool.put(b);
            }
            if let Some(a) = cell.lock().unwrap_or_else(|p| p.into_inner()).take() {
                a.send(Ok(()));
            }
        });
        if let Err(e) = self.notify(&stream, finish) {
            if let Some(a) = ack_cell.lock().unwrap_or_else(|p| p.into_inner()).take() {
                a.send(Err(e.clone()));
            }
            self.notify_failed("htod_chunks", &e);
        }
    }

    pub(super) fn handle_htod_chunk(
        &mut self,
        block: u64,
        offset_bytes: usize,
        data: Vec<u8>,
        ack: Ack<Result<(), ModelError>>,
    ) {
        let stream = match self.ctx().stream(STREAM_H2D) {
            Ok(s) => s.clone(),
            Err(e) => return ack.send(Err(e)),
        };
        let mut ack = Some(ack);
        // 边界校验(2026-10-15 补口):write_block 是 decode 每步高频小写路径,
        // 越界会静默落邻块(设备内无保护)
        if let Err(e) = self
            .ctx()
            .check_range(block, offset_bytes, data.len(), "htod-chunk")
        {
            return ack.take().unwrap().send(Err(e));
        }
        // **清零序**(2026-10-16):见 try_htod 注(H2D 写前等该块 memset)
        if let Err(e) = self.ctx_mut().wait_zeroing(block, STREAM_H2D) {
            return ack.take().unwrap().send(Err(e));
        }
        let dptr_base = match self.ctx().block_ptr(block, &stream) {
            Ok((p, _)) => p,
            Err(e) => return ack.take().unwrap().send(Err(e)),
        };
        // 码头池化(S1,2026-09-30):write_block 是 decode 每步 5 连发的
        // 高频小写路径,per-call malloc_host/Free 实测 ~3.3ms/次(与
        // try_htod 09-26 同病史;池 = PinnedPool,take>=len / finish 归还)
        let mut staging: Box<dyn owl_iface::contract::PinnedRegion + Send> =
            match self.pool.take(data.len()) {
                Some(b) => b,
                None => match Staging::alloc(data.len(), "htod_chunk") {
                    Ok(s) => Box::new(s),
                    Err(e) => return ack.take().unwrap().send(Err(e)),
                },
            };
        staging.slice_bytes_mut()[..data.len()].copy_from_slice(&data);
        unsafe {
            memcpy_htod_async(
                dptr_base + (offset_bytes as u64),
                &staging.as_bytes()[..data.len()],
                stream.cu_stream(),
            )
        }
        .map_err(|e| ModelError::Msg(format!("htod chunk async: {e:?}")))
        .map(|_| ())
        .map_err(|e| {
            if let Some(a) = ack.take() {
                a.send(Err(e));
            }
        })
        .err();
        // 码头随 finish 存活至搬运完成后归还池(派发线程执行);ack 住 cell
        // (同上:旧 `ack.take().unwrap()` 在 notify 失败路径必 panic)
        let ack_cell = std::sync::Arc::new(std::sync::Mutex::new(ack.take()));
        let pool = self.pool.clone();
        let cell = ack_cell.clone();
        let finish: Finish = Box::new(move || {
            pool.put(staging);
            if let Some(a) = cell.lock().unwrap_or_else(|p| p.into_inner()).take() {
                a.send(Ok(()));
            }
        });
        if let Err(e) = self.notify(&stream, finish) {
            if let Some(a) = ack_cell.lock().unwrap_or_else(|p| p.into_inner()).take() {
                a.send(Err(e.clone()));
            }
            self.notify_failed("htod_chunk", &e);
        }
    }

    /// srv.seg.dtoh 分相(与客户端同名段对账;差值 = 管道/等待)
    pub(super) fn handle_dtoh(
        &mut self,
        id: u64,
        want_bytes: usize,
        ack: Ack<Result<Vec<u8>, ModelError>>,
    ) {
        let t0 = std::time::Instant::now();
        self.handle_dtoh_inner(id, want_bytes, ack);
        self.sink.timer("srv.seg.dtoh", t0.elapsed());
    }

    pub(super) fn handle_dtoh_inner(
        &mut self,
        id: u64,
        want_bytes: usize,
        ack: Ack<Result<Vec<u8>, ModelError>>,
    ) {
        // 路由:D2H 流(结果收割维)。
        //
        // **跨流保序(2026-09-26 塔零案 / 2026-10-14 P1′-laneC 定向化)**:kernel
        // 在 COMPUTE 流, 本 memcpy 在 D2H 流 —— 跨流无保序,不排空则读块与产出
        // kernel 竞速(实测:真模型单步中途 dtoh 读到全零块)。
        // 旧修法 = `STREAM_COMPUTE.synchronize()`(整流排空,阻塞**服务端线程**
        // ⇒ 整条命令管道被钉死)。新修法 = **事件定向等待**:在 COMPUTE 当前尾记
        // 一个事件,D2H 流 wait 它 ⇒ 拷贝必然排在“本 dtoh 之前已入队”的全部产出
        // 核之后 —— **保序强度与整流 sync 同源**(事件记在尾上 = 等它 = 等此前全部),
        // 但服务端不等,后续命令照跑。
        // `sync_bit::DTOH`(全同步归因档的 dtoh 半边)= 退回整流 sync(A/B 与回退)。
        let compute = match self.ctx().stream(STREAM_COMPUTE) {
            Ok(s) => s.clone(),
            Err(e) => return ack.send(Err(e)),
        };
        let stream = match self.ctx().stream(STREAM_D2H) {
            Ok(s) => s.clone(),
            Err(e) => return ack.send(Err(e)),
        };
        let t_d2h = (want_bytes <= 64 && self.probes.d2h_prof).then(std::time::Instant::now);
        if self.diag.sync_at(owl_shared::config::sync_bit::DTOH) {
            if let Err(e) = compute.synchronize().map_err(|e| ModelError::Msg(format!("sync: {e:?}"))) {
                return ack.send(Err(e));
            }
        } else {
            match compute.record_event(None) {
                Ok(ev) => {
                    if let Err(e) = stream.wait(&ev) {
                        return ack.send(Err(ModelError::Msg(format!("dtoh 跨流事件等待: {e:?}"))));
                    }
                }
                Err(e) => {
                    return ack.send(Err(ModelError::Msg(format!("dtoh 跨流事件记入: {e:?}"))));
                }
            }
        }
        // 取证(B6.3 后泄漏案):每次收割刷新设备余量账(engine 侧逐 turn 打印;
        // cuMemGetInfo µs 级,收割频度下零税)
        if let Ok((free, _)) = cudarc::driver::result::mem_get_info() {
            owl_shared::vram::free_set(free as u64);
        }
        if let Some(t) = t_d2h {
            self.sink.logf(owl_shared::log::LVL_DEBUG, "d2h-prof", format_args!("序={:?} n={want_bytes}", t.elapsed()));
        }
        let t_d2h2 = t_d2h;
        let mut ack = Some(ack);
        match self.try_dtoh(&stream, id, want_bytes) {
            Ok(staging) => {
                if let Some(t) = t_d2h2 {
                    self.sink.logf(owl_shared::log::LVL_DEBUG, "d2h-prof", format_args!("issue={:?}", t.elapsed()));
                }
                // 完成 → 码头字节直出(传输面字节口径)→ 回执 → 码头归还池;ack 住 cell
                let ack_cell = std::sync::Arc::new(std::sync::Mutex::new(ack.take()));
                let pool = self.pool.clone();
                let cell = ack_cell.clone();
                // 字节口径 = want_bytes(池码头容量 ≥ n,直出整段会超发)
                let finish: Finish = Box::new(move || {
                    let bytes = staging.as_bytes()[..want_bytes].to_vec();
                    pool.put(staging);
                    if let Some(a) = cell.lock().unwrap_or_else(|p| p.into_inner()).take() {
                        a.send(Ok(bytes));
                    }
                });
                if let Err(e) = self.notify(&stream, finish) {
                    if let Some(a) = ack_cell.lock().unwrap_or_else(|p| p.into_inner()).take() {
                        a.send(Err(e.clone()));
                    }
                    self.notify_failed("dtoh", &e);
                }
            }
            Err(e) => ack.take().unwrap().send(Err(e)),
        }
    }

    /// dtoh issue 相(非阻塞):账长校验 + 异步 memcpy 到 pinned 码头。
    /// 返回码头(随 finish 存活至搬运完成后由派发线程释放)。
    pub(super) fn try_dtoh(
        &self,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        id: u64,
        want_bytes: usize,
    ) -> Result<Box<dyn owl_iface::contract::PinnedRegion + Send>, ModelError> {
        let n = self.ctx().block_len(id)?; // 块账本 = 字节(2026-09-26 f16 基线)
        // 前缀读取放行(E5-DF4:池复用块 > 声明尺寸,诊断收割取前缀;
        // 超读仍拒 —— 真错)。等大读路径不变。
        if want_bytes > n {
            return Err(ModelError::Msg(format!(
                "dtoh: 块 {id} 字节 {n} < 收割 {want_bytes}"
            )));
        }
        let (dptr, _) = self.ctx().block_ptr(id, stream)?;
        // 码头池化(S1):收割路径同样高频(token 4B/步;logits 大块另计)
        let mut staging: Box<dyn owl_iface::contract::PinnedRegion + Send> =
            match self.pool.take(n) {
                Some(b) => b,
                None => Box::new(Staging::alloc(n, "dtoh")?),
            };
        // 拷贝长度钉死 n(池码头容量 ≥ n,整段拷会越读设备块)
        unsafe {
            memcpy_dtoh_async(&mut staging.slice_bytes_mut()[..n], dptr, stream.cu_stream())
        }
        .map_err(|e| ModelError::Msg(format!("dtoh async: {e:?}")))?;
        Ok(staging)
    }



}

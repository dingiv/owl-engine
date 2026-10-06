//! DMA 收割臂(htod/dtoh/pinned 码头/分块流水)。码头池化(S1)、
//! COMPUTE 排空纪律(塔零案定谳)集中于此。`impl GpuServer` 跨文件
//! (server 巨石拆分 P1.4,2026-10-03)。

use super::*;

impl GpuServer {

    pub(super) fn handle_htod(&mut self, data: Vec<u8>, ack: Ack<Result<Bytes, ModelError>>) {
        let mut ack = Some(ack);
        let n = data.len();
        let stream = match self.ctx().stream(STREAM_H2D) {
            Ok(s) => s.clone(),
            Err(e) => return ack.take().unwrap().send(Err(e)),
        };
        match self.try_htod(&stream, &data) {
            Ok((id, buf)) => {
                // 码头随 finish 存活至搬运完成后归还池(派发线程执行)
                let mut cb_ack = ack.take();
                let pool = self.pool.clone();
                let finish: Finish = Box::new(move || {
                    pool.put(buf);
                    cb_ack.take().unwrap().send(Ok(Bytes::new(id, n)));
                });
                if let Err(e) = self.notify(&stream, finish) {
                    ack.take().unwrap().send(Err(e));
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
    ) -> Result<(u64, Box<dyn owl_iface::contract::PinnedRegion + Send>), ModelError> {
        let n = data.len();
        // 1. 设备块(流序 malloc;未初始化;字节口径)
        let dst = unsafe { stream.alloc::<u8>(n) }
            .map_err(|e| ModelError::Msg(format!("htod alloc: {e:?}")))?;
        let id = self.ctx_mut().new_block(dst);
        // 2. pinned 码头 + 异步 memcpy(流序;host 侧必须 pinned 才真异步)。
        //    码头池化(2026-09-26 尾批:原每次 malloc_host 新分配,大权重
        //    锁页分配是装载耗时大头;finish 回调归还池)
        let mut buf: Box<dyn owl_iface::contract::PinnedRegion + Send> =
            match self.pool.take(n) {
                Some(b) => b,
                None => Box::new(Staging::alloc(n)?),
            };
        buf.slice_bytes_mut()[..n].copy_from_slice(data);
        let (dptr, _) = self.ctx().block_ptr(id, stream)?;
        // 池块容量 ≥ 请求:DMA 长度必须截到 n(越界写设备块 = INVALID_VALUE)
        unsafe { memcpy_htod_async(dptr, &buf.as_bytes()[..n], stream.cu_stream()) }
            .map_err(|e| ModelError::Msg(format!("htod async: {e:?}")))?;
        // 诊断开关(OWL_LAUNCH_SYNC=1):DMA 后同步归因(H2D 错误平时
        // 无条件 Ok 回执,sticky 晚冒 —— 2026-09-27 排查补)
        if self.probes.launch_sync {
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
        ack: Ack<Result<Box<dyn owl_iface::contract::PinnedRegion + Send>, ModelError>>,
    ) {
        // 池取（内部单次加锁；容量 ≥ 请求即命中，逻辑长度由调用方界定）
        if let Some(b) = self.pool.take(bytes) {
            // 容量不可截，整块交出（调用方只用前 bytes）
            ack.send(Ok(b));
            return;
        }
        match Staging::alloc(bytes) {
            Ok(s) => ack.send(Ok(Box::new(s))),
            Err(e) => ack.send(Err(e)),
        }
    }

    /// 上传租约:buf 所有权移入,DMA 到 dst+offset;完成回调把 buf 归还池
    pub(super) fn handle_upload_pinned(
        &mut self,
        buf: Box<dyn owl_iface::contract::PinnedRegion + Send>,
        dst: Bytes,
        offset_bytes: usize,
        len_bytes: usize,
        ack: Ack<Result<(), ModelError>>,
    ) {
        let stream = match self.ctx().stream(STREAM_H2D) {
            Ok(s) => s.clone(),
            Err(e) => return ack.send(Err(e)),
        };
        let mut ack = Some(ack);
        let dptr_base = match self.ctx().block_ptr(dst.id, &stream) {
            Ok((p, _)) => p,
            Err(e) => return ack.take().unwrap().send(Err(e)),
        };
        if let Err(e) = unsafe {
            memcpy_htod_async(
                dptr_base + (offset_bytes as u64),
                &buf.as_bytes()[..len_bytes],
                stream.cu_stream(),
            )
        } {
            // 建场失败:未入队任何 DMA,finish 不得投递 —— 租约回池 + 回执错误
            let e = ModelError::Msg(format!("upload_pinned async: {e:?}"));
            self.pool.put(buf);
            ack.take().unwrap().send(Err(e));
            return;
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
        ack.take().unwrap().send(Ok(())); // 入队即回执(非阻塞)
        if let Err(e) = self.notify(&stream, finish) {
            // notify 失败:finish 未投递,租约回收入池(ack 已回,
            // DMA 未发生 —— 块内容缺失由调用方末端 sync 后自查/重试兜底)
            if let Some(buf) = cell.lock().unwrap().take() {
                pool_err.put(buf);
                eprintln!("[srv] upload notify 失败: {e:?}");
            }
        }
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
                    None => match Staging::alloc(data.len()) {
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
        // 单 notify:全部写到位后一次性回执(保序 = H2D 流序)
        let mut cb_ack = ack.take();
        let pool = self.pool.clone();
        let finish: Finish = Box::new(move || {
            for b in pending {
                pool.put(b);
            }
            cb_ack.take().unwrap().send(Ok(()));
        });
        if let Err(e) = self.notify(&stream, finish) {
            // notify 失败 = 断链(ack 已随闭包移交;与 handle_htod_chunk 同语义)
            let _ = e;
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
                None => match Staging::alloc(data.len()) {
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
        // 码头随 finish 存活至搬运完成后归还池(派发线程执行)
        let mut cb_ack = ack.take();
        let pool = self.pool.clone();
        let finish: Finish = Box::new(move || {
            pool.put(staging);
            cb_ack.take().unwrap().send(Ok(()));
        });
        if let Err(e) = self.notify(&stream, finish) {
            ack.take().unwrap().send(Err(e));
        }
    }

    pub(super) fn handle_dtoh(
        &mut self,
        id: u64,
        want_bytes: usize,
        ack: Ack<Result<Vec<u8>, ModelError>>,
    ) {
        // 路由:D2H 流(结果收割维)。先排空 COMPUTE:kernel 在 COMPUTE 流,
        // 本 memcpy 在 D2H 流 —— 跨流无保序,不排空则读块与产出 kernel 竞速
        // (实测:真模型单步中途 dtoh 读到全零块;sync 后重读同块数据完好

        // —— 2026-09-26 塔零案定谳,interpreter-tap.md)。收割路径本就阻塞
        // 等回调,排空无额外代价;htod 侧 async+sync 已内建,无对称问题。
        let t_d2h = (want_bytes <= 64 && self.probes.d2h_prof).then(std::time::Instant::now);
        if let Err(e) = self
            .ctx()
            .stream(STREAM_COMPUTE)
            .and_then(|s| s.synchronize().map_err(|e| ModelError::Msg(format!("sync: {e:?}"))))
        {
            return ack.send(Err(e));
        }
        if let Some(t) = t_d2h {
            eprintln!("[d2h-prof] sync={:?} n={want_bytes}", t.elapsed());
        }
        let stream = match self.ctx().stream(STREAM_D2H) {
            Ok(s) => s.clone(),
            Err(e) => return ack.send(Err(e)),
        };
        let t_d2h2 = t_d2h;
        let mut ack = Some(ack);
        match self.try_dtoh(&stream, id, want_bytes) {
            Ok(staging) => {
                if let Some(t) = t_d2h2 {
                    eprintln!("[d2h-prof] issue={:?}", t.elapsed());
                }
                // 完成 → 码头字节直出(传输面字节口径)→ 回执 → 码头归还池
                let mut cb_ack = ack.take();
                let pool = self.pool.clone();
                // 字节口径 = want_bytes(池码头容量 ≥ n,直出整段会超发)
                let finish: Finish = Box::new(move || {
                    let bytes = staging.as_bytes()[..want_bytes].to_vec();
                    pool.put(staging);
                    cb_ack.take().unwrap().send(Ok(bytes));
                });
                if let Err(e) = self.notify(&stream, finish) {
                    ack.take().unwrap().send(Err(e));
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
                None => Box::new(Staging::alloc(n)?),
            };
        // 拷贝长度钉死 n(池码头容量 ≥ n,整段拷会越读设备块)
        unsafe {
            memcpy_dtoh_async(&mut staging.slice_bytes_mut()[..n], dptr, stream.cu_stream())
        }
        .map_err(|e| ModelError::Msg(format!("dtoh async: {e:?}")))?;
        Ok(staging)
    }



}

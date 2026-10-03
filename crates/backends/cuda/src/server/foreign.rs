//! 外来核执行臂(cuBLAS/marlin W4A16+AWQ/FlashInfer/GDN chunked+scalar)。
//! 槽序解析单点化(`parse_foreign_slots`);各 handler 只做业务
//! (块→指针偏移、库调用、ack)。`impl GpuServer` 跨文件:子模块可及
//! 父模块私有字段(server 巨石拆分 P1.4,2026-10-03)。

use super::*;

impl GpuServer {
    /// 外来核槽序单点解析(刀1.6 结构清理:七臂重复循环收敛于此)。
    /// Block → (id, 0);BlockSlice → (id, byte_offset);标量 → u64 槽。
    /// 期望槽数不符 = 结构化违约(带核名)。
    pub(super) fn parse_foreign_slots(
        msg: &LaunchMsg,
        want_blocks: usize,
        want_scalars: usize,
    ) -> Result<(Vec<(u64, u64)>, Vec<u64>), ModelError> {
        let mut blocks: Vec<(u64, u64)> = Vec::with_capacity(want_blocks); // (id, byte_offset)
        let mut scalars: Vec<u64> = Vec::with_capacity(want_scalars);
        for a in &msg.args {
            match a {
                Arg::Block { id } => blocks.push((*id, 0)),
                Arg::BlockSlice { id, byte_offset, .. } => blocks.push((*id, *byte_offset)),
                Arg::U64(v) => scalars.push(*v),
                other => {
                    return Err(ModelError::Msg(format!(
                        "foreign kernel {} 槽序违约:仅 Block/BlockSlice/U64,得 {other:?}",
                        msg.kernel.name
                    )))
                }
            }
        }
        if blocks.len() != want_blocks || scalars.len() != want_scalars {
            return Err(ModelError::Msg(format!(
                "foreign kernel {} 槽序违约:{want_blocks} Block + {want_scalars} U64,得 {}B/{}S",
                msg.kernel.name,
                blocks.len(),
                scalars.len()
            )));
        }
        Ok((blocks, scalars))
    }

    /// 外部核执行(cuBLAS 先行;marlin/FlashInfer 同通道后续接入)。
    /// 槽序契约见 owl_kernels::cublas::GEMM_F16 文档。
    pub(super) fn handle_foreign_launch(&mut self, msg: LaunchMsg, ack: Ack<Result<Bytes, ModelError>>) {
        // 捕获期策略(2026-09-26 修订):cublasGemmEx/外部核本身可捕获
        // (kernel 入捕获流;workspace 需捕获前已分配 —— session warmup
        // 先于 graph_begin,句柄已建即满足)。唯一拒绝态 = 捕获内首次
        // 初始化(句柄创建含分配,捕获窗内非法)→ 结构化拒绝。
        if self
            .ctx
            .as_ref()
            .map(|c| c.capture_stream())
            .unwrap_or(false)
            && self.blas.is_none()
        {
            return ack.send(Err(ModelError::Msg(format!(
                "foreign kernel {} 捕获内首次初始化(需先 warmup 建句柄)",
                msg.kernel.name
            ))));
        }
        match msg.kernel.name.as_str() {
            name if name == owl_kernels::cublas::GEMM_F16 => self.handle_cublas_gemm(msg, ack),
            name if name == owl_kernels::marlin::GEMM_W4A16 => self.handle_marlin_gemm(msg, ack),
            name if name == owl_kernels::marlin::GEMM_W4A16_AWQ => {
                self.handle_marlin_gemm_awq(msg, ack)
            }
            name if name == owl_kernels::flashinfer::PREFILL_FI
                || name == owl_kernels::flashinfer::PREFILL_FI_FP8KV =>
            {
                self.handle_fi_prefill(msg, ack)
            }
            name if name == owl_kernels::gdn_chunked::GDN_CHUNKED_FWD => {
                self.handle_gdn_chunked(msg, ack)
            }
            name if name == owl_kernels::gdn_scalar::GDN_SCALAR_FWD => {
                self.handle_gdn_scalar(msg, ack)
            }
            other => ack.send(Err(ModelError::Msg(format!(
                "foreign kernel {other}: 无执行臂(owl_kernels::is_foreign_op 与分派表失配)"
            )))),
        }
    }

    /// cuBLAS f16 GEMM 臂(槽序:[T a, T b, T out, sz m, sz k, sz n, sz nt])
    pub(super) fn handle_cublas_gemm(&mut self, msg: LaunchMsg, ack: Ack<Result<Bytes, ModelError>>) {
        if self.blas.is_none() {
            let stream = match self.ctx().stream(STREAM_COMPUTE) {
                Ok(s) => s.clone(),
                Err(e) => return ack.send(Err(e)),
            };
            match owl_kernels::cublas::OwlCublas::new(stream.clone()) {
                Ok(h) => self.blas = Some(h),
                Err(e) => return ack.send(Err(ModelError::Msg(e))),
            }
            // 刀1.5:server 私有 4MB 工作区(捕获前分配;SetWorkspace 预绑
            // —— 捕获期 gemv splitK 不走池分配,免 MEM_ALLOC/FREE 节点)。
            // 并行 server 各享各的,零共享零竞争(全局单例实证会炸)。
            const BLAS_WS_BYTES: usize = 4 << 20;
            match unsafe { stream.alloc::<u8>(BLAS_WS_BYTES) } {
                Ok(buf) => {
                    use cudarc::driver::DevicePtr;
                    let arc = std::sync::Arc::new(buf);
                    let ptr = match arc.device_ptr(&stream) {
                        (p, _sync) => p as u64,
                    };
                    let handle = *self.blas.as_ref().unwrap().sys_handle();
                    let r = unsafe {
                        crate::ffi::sys::cublas::cublasSetWorkspace_v2(handle, ptr as *mut _, BLAS_WS_BYTES)
                    };
                    if r == crate::ffi::sys::cublas::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                        eprintln!("[blas-ws] 私有工作区 {BLAS_WS_BYTES}B @ {ptr:#x}");
                        self.blas_ws = Some(arc);
                    } else {
                        eprintln!("[blas-ws] SetWorkspace 失败 r={r:?}(启发式回退池分配)");
                    }
                }
                Err(e) => eprintln!("[blas-ws] 分配失败 {e:?}(回退池分配)"),
            }
        }
        // 槽序:[T a, T b, T out, sz m, sz k, sz n, sz nt]
        let (blocks, scalars) = match Self::parse_foreign_slots(&msg, 3, 4) {
            Ok(v) => v,
            Err(e) => return ack.send(Err(e)),
        };
      let (m, k, n, nt) =
            (scalars[0] as usize, scalars[1] as usize, scalars[2] as usize, scalars[3] != 0);
        if self.probes.debug {
            eprintln!(
                "[dbg foreign] {} a=Block({}) b=Block({}) out=Block({}) m={n_out} k={k} n={n_tok} nt={nt}",
                msg.kernel.name, blocks[0].0, blocks[1].0, blocks[2].0, n_out = m, n_tok = n
            );
        }
        let blas = self.blas.as_ref().unwrap();
        let stream = match self.ctx().stream(STREAM_COMPUTE) {
            Ok(s) => s.clone(),
            Err(e) => return ack.send(Err(e)),
        };
        let (a_ptr, _) = match self.ctx().block_ptr(blocks[0].0, &stream) {
            Ok(p) => (p.0 + blocks[0].1, p.1),
            Err(e) => return ack.send(Err(e)),
        };
        let (b_ptr, _) = match self.ctx().block_ptr(blocks[1].0, &stream) {
            Ok(p) => (p.0 + blocks[1].1, p.1),
            Err(e) => return ack.send(Err(e)),
        };
        let (out_ptr, _) = match self.ctx().block_ptr(blocks[2].0, &stream) {
            Ok(p) => (p.0 + blocks[2].1, p.1),
            Err(e) => return ack.send(Err(e)),
        };
        let capturing = self.ctx().capture_stream();
        match blas.gemm_f16(a_ptr, b_ptr, out_ptr, m, k, n, nt) {
            Ok(()) => {
                // 诊断开关(同 handle_launch;捕获期跳过)
                if self.probes.launch_sync && !capturing {
                    if let Err(e) = stream.synchronize() {
                        return ack.send(Err(ModelError::Msg(format!(
                            "gemm-sync(m={m},k={k},n={n},nt={nt}): {e:?}"
                        ))));
                    }
                }
                ack.send(Ok(Bytes::new(blocks[2].0, msg.out_elems)))
            }
            Err(e) => ack.send(Err(ModelError::Msg(e))),
        }
    }

    /// Marlin W4A16 臂(槽序:[T a, T b, T out, T scales, T ws, T c_tmp,
    /// sz m, sz k, sz n, sz groupsize];契约见 owl_kernels::marlin)。
    /// 无句柄状态(纯 FFI);stream/dev 由本 server 注入。
    pub(super) fn handle_marlin_gemm(&mut self, msg: LaunchMsg, ack: Ack<Result<Bytes, ModelError>>) {
        let (blocks, scalars) = match Self::parse_foreign_slots(&msg, 6, 4) {
            Ok(v) => v,
            Err(e) => return ack.send(Err(e)),
        };
  let (m, k, n, groupsize) =
            (scalars[0] as usize, scalars[1] as usize, scalars[2] as usize, scalars[3] as i32);
        let stream = match self.ctx().stream(STREAM_COMPUTE) {
            Ok(s) => s.clone(),
            Err(e) => return ack.send(Err(e)),
        };
        let mut ptrs = Vec::with_capacity(6);
        for (b, off) in &blocks {
            match self.ctx().block_ptr(*b, &stream) {
                Ok((p, _)) => ptrs.push(p + off),
                Err(e) => return ack.send(Err(e)),
            }
        }
        let dev = self.ctx().device_ordinal() as i32;
        // 排队即回执(fire-and-forget;marlin host launcher 入 COMPUTE 流)
        let r = unsafe {
            owl_kernels::marlin::gemm_v2_raw(
                ptrs[0] as *const u16,
                ptrs[1] as *const i32,
                ptrs[2] as *mut u16,
                ptrs[3] as *const u16,
                ptrs[5] as *const c_void,
                m as i32,
                n as i32,
                k as i32,
                ptrs[4] as *mut i32,
                groupsize,
                dev,
                stream.cu_stream() as usize,
            )
        };
        match r {
            Ok(()) => ack.send(Ok(Bytes::new(blocks[2].0, msg.out_elems))),
            Err(e) => ack.send(Err(ModelError::Msg(format!(
                "marlin gemm err {e}: {}",
                owl_kernels::marlin::v2_err_str(e)
            )))),
        }
    }


    /// FlashInfer paged prefill 臂(E1.5)。槽序契约:
    /// [T q, T kc_fi, T vc, T q_cu, T indices, T indptr, T last_len, T wr,
    ///  O out, sz total_rows, sz ctx_total, sz T, sz hq, sz hkv, sz hd,
    ///  sz page, sz sm_scale_bits, sz nb](8 Block + 8 sz;wr 依赖边)。
    /// plan(host,每 chunk 一次)1 项缓存;run 每层一次。批 = 1。
    /// 捕获窗拒绝态同 cublas(FI prefill 现役仅 eager;解码图不含它)。
    pub(super) fn handle_fi_prefill(&mut self, msg: LaunchMsg, ack: Ack<Result<Bytes, ModelError>>) {
        if self.fi.is_none() {
            if self.ctx.as_ref().map(|c| c.capture_stream()).unwrap_or(false) {
                return ack.send(Err(ModelError::Msg(
                    "flashinfer_prefill 捕获内首次初始化(需先 eager warmup)".into(),
                )));
            }
            let stream = match self.ctx().stream(STREAM_COMPUTE) {
                Ok(s) => s.clone(),
                Err(e) => return ack.send(Err(e)),
            };
            let float_ws = stream
                .alloc_zeros::<u8>(owl_kernels::flashinfer::FI_FLOAT_WS_BYTES)
                .map_err(|e| ModelError::Msg(format!("fi ws alloc: {e:?}")));
            let int_ws = stream
                .alloc_zeros::<u8>(owl_kernels::flashinfer::FI_INT_WS_BYTES)
                .map_err(|e| ModelError::Msg(format!("fi ws alloc: {e:?}")));
            let (float_ws, int_ws) = match (float_ws, int_ws) {
                (Ok(a), Ok(b)) => (a, b),
                (Err(e), _) | (_, Err(e)) => return ack.send(Err(e)),
            };
            use cudarc::driver::DevicePtr;
            // 空流快照:此时无排队内核,SyncOnDrop 守卫就地丢弃零成本
            let float_ws_ptr = {
                let (p, _g) = DevicePtr::<u8>::device_ptr(&float_ws, &stream);
                p
            };
            let int_ws_ptr = {
                let (p, _g) = DevicePtr::<u8>::device_ptr(&int_ws, &stream);
                p
            };
            self.fi = Some(FiState {
                float_ws,
                int_ws,
                float_ws_ptr,
                int_ws_ptr,
                host_staging: vec![0u8; owl_kernels::flashinfer::FI_HOST_STAGING_BYTES],
                plan: None,
            });
        }
        let (blocks, scalars) = match Self::parse_foreign_slots(&msg, 9, 8) {
            Ok(v) => v,
            Err(e) => return ack.send(Err(e)),
        };
 let (total_rows, ctx_total, t, hq, hkv, hd, page, sm_bits) = (
            scalars[0] as usize, scalars[1] as usize, scalars[2] as usize,
            scalars[3] as usize, scalars[4] as usize, scalars[5] as usize,
            scalars[6] as usize, scalars[7] as u32,
        );
        let sm_scale = f32::from_bits(sm_bits);
        let stream = match self.ctx().stream(STREAM_COMPUTE) {
            Ok(s) => s.clone(),
            Err(e) => return ack.send(Err(e)),
        };
        let mut ptrs = Vec::with_capacity(9);
        for (b, off) in &blocks {
            match self.ctx().block_ptr(*b, &stream) {
                Ok((p, _)) => ptrs.push(p + off),
                Err(e) => return ack.send(Err(e)),
            }
        }
        // workspace 指针/长度(init 时缓存的裸指针;发射期零同步)
        let (int_ws_ptr, int_ws_len, float_ws_ptr, float_ws_len, staging_ptr, staging_len) = {
            let fi = self.fi.as_mut().unwrap();
            (
                fi.int_ws_ptr as *mut c_void,
                fi.int_ws.len(),
                fi.float_ws_ptr as *mut c_void,
                fi.float_ws.len(),
                fi.host_staging.as_mut_ptr() as *mut c_void,
                fi.host_staging.len(),
            )
        };
        // plan 缓存(键 = 形状七元组 + total_rows)
        let key = (total_rows, ctx_total, t, hq, hkv, hd, page);
        let need_plan = self.fi.as_ref().unwrap().plan.as_ref().map(|c| c.key != key).unwrap_or(true);
        let fi_plan_t0 = std::time::Instant::now();
        let plan_result: Result<_, String> = (|| {
            let qo_indptr = [0i32, t as i32];
            let kv_indptr = [0i32, ctx_total as i32];
            let (mut plan15, mut cta_tile_q, mut split_kv) = ([0i64; 15], 0i32, 0i32);
            let r = unsafe {
                owl_kernels::flashinfer::owl_fi_prefill_plan(
                    float_ws_ptr,
                    float_ws_len,
                    int_ws_ptr,
                    int_ws_len,
                    staging_ptr,
                    staging_len,
                    plan15.as_mut_ptr(),
                    &mut cta_tile_q,
                    &mut split_kv,
                    qo_indptr.as_ptr(),
                    kv_indptr.as_ptr(),
                    total_rows as i32,
                    1, // batch = 1(owl 单会话)
                    hq as i32, hkv as i32, hd as i32, page as i32,
                    stream.cu_stream() as *mut c_void,
                )
            };
            if r != 0 {
                return Err(format!(
                    "flashinfer_prefill_plan err {r}(hd={hd} page={page} T={t} ctx={ctx_total})"
                ));
            }
            Ok((plan15, cta_tile_q, split_kv))
        })();
        owl_shared::metrics::with_metrics_store(|m| {
            m.timer_record_tag("fi.plan", fi_plan_t0.elapsed(), file!(), line!())
        });
        let (mut plan15, _cta, _split) = match plan_result {
            Ok(v) => v,
            Err(e) => return ack.send(Err(ModelError::Msg(e))),
        };
        if let Some(fi) = self.fi.as_mut() {
            fi.plan = Some(FiPlanCache { key, plan15, cta_tile_q: _cta, split_kv: _split });
        }
        let _ = (_cta, _split);
        let plan15 = self.fi.as_ref().unwrap().plan.as_ref().unwrap().plan15;
        let fi_run_t0 = std::time::Instant::now();
        let fp8kv = msg.kernel.name == owl_kernels::flashinfer::PREFILL_FI_FP8KV;
        let r = unsafe {
            if fp8kv {
                owl_kernels::flashinfer::owl_fi_prefill_run_fp8kv(
                    ptrs[0] as *const c_void,
                    ptrs[1] as *const c_void,
                    ptrs[2] as *const c_void,
                    ptrs[8] as *mut c_void,
                    ptrs[3] as *mut i32,
                    ptrs[4] as *mut i32,
                    ptrs[5] as *mut i32,
                    ptrs[6] as *mut i32,
                    plan15.as_ptr(),
                    int_ws_ptr,
                    int_ws_len,
                    float_ws_ptr,
                    float_ws_len,
                    1, // batch
                    hq as i32, hkv as i32, hd as i32, page as i32,
                    total_rows as i32,
                    sm_scale,
                    stream.cu_stream() as *mut c_void,
                )
            } else {
                owl_kernels::flashinfer::owl_fi_prefill_run(
                    ptrs[0] as *const c_void,
                    ptrs[1] as *const c_void,
                    ptrs[2] as *const c_void,
                    ptrs[8] as *mut c_void,
                    ptrs[3] as *mut i32,
                    ptrs[4] as *mut i32,
                    ptrs[5] as *mut i32,
                    ptrs[6] as *mut i32,
                    plan15.as_ptr(),
                    int_ws_ptr,
                    int_ws_len,
                    float_ws_ptr,
                    float_ws_len,
                    1, // batch
                    hq as i32, hkv as i32, hd as i32, page as i32,
                    total_rows as i32,
                    sm_scale,
                    stream.cu_stream() as *mut c_void,
                )
            }
        };
        owl_shared::metrics::with_metrics_store(|m| {
            m.timer_record_tag("fi.run", fi_run_t0.elapsed(), file!(), line!())
        });
        match r {
            0 => ack.send(Ok(Bytes::new(blocks[8].0, msg.out_elems))),
            e => ack.send(Err(ModelError::Msg(format!("flashinfer_prefill_run err {e}")))),
        }
    }


    /// GDN chunked delta rule 臂(FLA AOT cubin 五核编排;2026-10-03)。
    /// 槽序:[T q, T k, T v, T g, T beta, T state, O out, sz T, sz slot,
    /// sz hv, sz nk, sz kd, sz scale_bits](6 Block + 1 O + 6 sz)。
    /// state = 槽寻址 f32 [slots, HV, KD, VD];h0/ht 同槽原地。
    /// 编排 = FLA chunk.py fwd 序:cumsum → kkt → wu → h → o → cast。
    /// GDN chunked delta rule 臂(FLA AOT cubin 五核编排;2026-10-03)。
    /// 槽序:[T q, T k, T v, T g, T beta, T state, O out,
    /// sz T, sz slot, sz hv, sz nk, sz kd, sz scale_bits]
    /// = 6 Block + 1 O + 6 sz。state = 槽寻址 f32 [slots, HV, KD, VD]。
    /// 编排 = FLA chunk.py fwd 序:cumsum → kkt → wu → h → o → cast。
    pub(super) fn handle_gdn_chunked(&mut self, msg: LaunchMsg, ack: Ack<Result<Bytes, ModelError>>) {
        let init = || -> Result<GdnChunkedState, ModelError> {
            if self.ctx.as_ref().map(|c| c.capture_stream()).unwrap_or(false) {
                return Err(ModelError::Msg(
                    "gdn_chunked 捕获内首次初始化(需先 eager warmup)".into(),
                ));
            }
            let stream = self.ctx().stream(STREAM_COMPUTE)?.clone();
            let load = |bytes: &[u8], fname: &str| -> Result<std::sync::Arc<cudarc::driver::CudaFunction>, ModelError> {
                let m = self
                    .ctx()
                    .ctx
                    .load_module(cudarc::nvrtc::Ptx::from_binary(bytes.to_vec()))
                    .map_err(|e| ModelError::Msg(format!("gdn_chunked load: {e:?}")))?;
                let f = m
                    .load_function(fname)
                    .map_err(|e| ModelError::Msg(format!("gdn_chunked fn {fname}: {e:?}")))?;
                Ok(std::sync::Arc::new(f))
            };
            use owl_kernels::gdn_chunked::cubins::launch as LC;
            let cumsum = load(owl_kernels::gdn_chunked::cubins::CUMSUM, "chunk_local_cumsum_scalar_kernel")?;
            let kkt = load(owl_kernels::gdn_chunked::cubins::KKT, "chunk_gated_delta_rule_fwd_kkt_solve_kernel")?;
            let wu = load(owl_kernels::gdn_chunked::cubins::WU, "recompute_w_u_fwd_kernel")?;
            let h = load(owl_kernels::gdn_chunked::cubins::H, "chunk_gated_delta_rule_fwd_kernel_h_blockdim64")?;
            let o = load(owl_kernels::gdn_chunked::cubins::O, "chunk_fwd_kernel_o")?;
            // shared opt-in(wu 81920 / o 98304 > 48KB)
            for (f, sz) in [(&wu, LC::WU_SHARED as i32), (&o, LC::O_SHARED as i32)] {
                use cudarc::driver::sys::CUfunction_attribute_enum as Attr;
                f.set_attribute(Attr::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, sz)
                    .map_err(|e| ModelError::Msg(format!("gdn_chunked setattr: {e:?}")))?;
            }
            // cast 模块(nvrtc;f16<->f32 双向)
            use cudarc::nvrtc::CompileOptions;
            let ptx = cudarc::nvrtc::compile_ptx_with_opts(
                owl_kernels::sources::attention::CAST,
                CompileOptions {
                    include_paths: vec!["/usr/local/cuda/include".to_string()],
                    ..Default::default()
                },
            )
            .map_err(|e| ModelError::Msg(format!("gdn_chunked cast nvrtc: {e:?}")))?;
            let cast_m = self
                .ctx()
                .ctx
                .load_module(cudarc::nvrtc::Ptx::from_src(ptx.to_src()))
                .map_err(|e| ModelError::Msg(format!("gdn_chunked cast load: {e:?}")))?;
            let cast_f16_f32 = std::sync::Arc::new(
                cast_m
                    .load_function("owl_cast_f16_f32")
                    .map_err(|e| ModelError::Msg(format!("gdn_chunked cast fn: {e:?}")))?,
            );
            let cast_f32_f16 = std::sync::Arc::new(
                cast_m
                    .load_function("owl_cast_f32_f16")
                    .map_err(|e| ModelError::Msg(format!("gdn_chunked cast fn: {e:?}")))?,
            );
            Ok(GdnChunkedState {
                cumsum,
                kkt,
                wu,
                h,
                o,
                cast_f16_f32,
                cast_f32_f16,
                q_f: stream.alloc_zeros::<f32>(1).map_err(|e| ModelError::Msg(format!("{e:?}")))?,
                k_f: stream.alloc_zeros::<f32>(1).map_err(|e| ModelError::Msg(format!("{e:?}")))?,
                v_f: stream.alloc_zeros::<f32>(1).map_err(|e| ModelError::Msg(format!("{e:?}")))?,
                g_f: stream.alloc_zeros::<f32>(1).map_err(|e| ModelError::Msg(format!("{e:?}")))?,
                beta_f: stream.alloc_zeros::<f32>(1).map_err(|e| ModelError::Msg(format!("{e:?}")))?,
                g_cum: stream.alloc_zeros::<f32>(1).map_err(|e| ModelError::Msg(format!("{e:?}")))?,
                a: stream.alloc_zeros::<f32>(1).map_err(|e| ModelError::Msg(format!("{e:?}")))?,
                w: stream.alloc_zeros::<f32>(1).map_err(|e| ModelError::Msg(format!("{e:?}")))?,
                u: stream.alloc_zeros::<f32>(1).map_err(|e| ModelError::Msg(format!("{e:?}")))?,
                h_buf: stream.alloc_zeros::<f32>(1).map_err(|e| ModelError::Msg(format!("{e:?}")))?,
                v_new: stream.alloc_zeros::<f32>(1).map_err(|e| ModelError::Msg(format!("{e:?}")))?,
                o_f32: stream.alloc_zeros::<f32>(1).map_err(|e| ModelError::Msg(format!("{e:?}")))?,
                idx: stream.alloc_zeros::<i64>(1).map_err(|e| ModelError::Msg(format!("{e:?}")))?,
                coff: stream.alloc_zeros::<i64>(1).map_err(|e| ModelError::Msg(format!("{e:?}")))?,
                cu: stream.alloc_zeros::<i64>(1).map_err(|e| ModelError::Msg(format!("{e:?}")))?,
                t_cap: 0,
            })
        };
        if self.gdn_chunked.is_none() {
            self.gdn_chunked = match init() {
                Ok(v) => Some(v),
                Err(e) => return ack.send(Err(e)),
            };
        }
        // 槽序解析
        let (blocks, scalars) = match Self::parse_foreign_slots(&msg, 7, 6) {
            Ok(v) => v,
            Err(e) => return ack.send(Err(e)),
        };
 let (t, slot, hv, nk, kd, scale_bits) = (
            scalars[0] as usize,
            scalars[1] as usize,
            scalars[2] as usize,
            scalars[3] as usize,
            scalars[4] as usize,
            scalars[5] as u32,
        );
        let scale = f32::from_bits(scale_bits);
        let nt = t.div_ceil(64);
        let vd = kd; // GDN: KD == VD(27B = 128)
        let stream = match self.ctx().stream(STREAM_COMPUTE) {
            Ok(s) => s.clone(),
            Err(e) => return ack.send(Err(e)),
        };
        let mut ptrs = Vec::with_capacity(7);
        for (b, off) in &blocks {
            match self.ctx().block_ptr(*b, &stream) {
                Ok((p, _)) => ptrs.push(p + off),
                Err(e) => return ack.send(Err(e)),
            }
        }
        let st = self.gdn_chunked.as_mut().unwrap();
        // 扩容(T 超容量)
        if t > st.t_cap {
            if let Err(e) = gdn_chunked_realloc(st, &stream, t, nt, hv, nk, kd, vd) {
                return ack.send(Err(e));
            }
        }
        // (层侧已 emit owl_cast_f16_f32 ×5 SSA 节点 —— q/k/v/g/beta 进来即 f32;
        //  此处不再二次 cast。)
        let (q_p, k_p, v_p, g_p, beta_p) = (
            gdn_dptr(&mut st.q_f, &stream),
            gdn_dptr(&mut st.k_f, &stream),
            gdn_dptr(&mut st.v_f, &stream),
            gdn_dptr(&mut st.g_f, &stream),
            gdn_dptr(&mut st.beta_f, &stream),
        );
        let (gcum_p, a_p, w_p, u_p, h_p) = (
            gdn_dptr(&mut st.g_cum, &stream),
            gdn_dptr(&mut st.a, &stream),
            gdn_dptr(&mut st.w, &stream),
            gdn_dptr(&mut st.u, &stream),
            gdn_dptr(&mut st.h_buf, &stream),
        );
        let (vnew_p, of32_p) = (
            gdn_dptr(&mut st.v_new, &stream),
            gdn_dptr(&mut st.o_f32, &stream),
        );
        let (idx_p, coff_p, cu_p) = (
            gdn_dptr(&mut st.idx, &stream),
            gdn_dptr(&mut st.coff, &stream),
            gdn_dptr(&mut st.cu, &stream),
        );
        let state_p = ptrs[5] + (slot * hv * kd * vd * 4) as u64; // f32 → 字节
        // host 表 htod(COMPUTE 流,先于 FLA 核):cu=[0,T] / idx=[(0,i)×NT] / coff=[0,NT]
        let idx_host: Vec<i64> = (0..nt).flat_map(|i| [0i64, i as i64]).collect();
        if self.probes.cap_prof {
            eprintln!("[cap-prof] gdn-chunked cu/idx/coff htod x3 (T={t})");
        }
        if let Err(e) = unsafe { memcpy_htod_async(cu_p, &[0i64, t as i64], stream.cu_stream()) } {
            return ack.send(Err(ModelError::Msg(format!("gdn cu htod: {e:?}"))));
        }
        if let Err(e) = unsafe { memcpy_htod_async(idx_p, &idx_host, stream.cu_stream()) } {
            return ack.send(Err(ModelError::Msg(format!("gdn idx htod: {e:?}"))));
        }
        if let Err(e) = unsafe { memcpy_htod_async(coff_p, &[0i64, nt as i64], stream.cu_stream()) } {
            return ack.send(Err(ModelError::Msg(format!("gdn coff htod: {e:?}"))));
        }
        let hv_u = hv as u32;
        // 1. cumsum:grid (NT, HV)
        {
            let mut b = stream.launch_builder(&st.cumsum);
            let ti = t as i32;
            let ln2 = 1.4426950408889634f32;
            let scratch: u64 = 0;
            b.arg(&g_p)
                .arg(&gcum_p)
                .arg(&ln2)
                .arg(&cu_p)
                .arg(&idx_p)
                .arg(&ti)
                .arg(&scratch)
                .arg(&scratch);
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (nt as u32, hv_u, 1),
                block_dim: (32 * owl_kernels::gdn_chunked::cubins::launch::CUMSUM_WARPS, 1, 1),
                shared_mem_bytes: owl_kernels::gdn_chunked::cubins::launch::CUMSUM_SHARED,
            };
            if let Err(e) = unsafe { b.launch(cfg) }.map_err(|e| ModelError::Msg(format!("gdn cumsum: {e:?}"))) {
                return ack.send(Err(e));
            }
        }
        // 2. kkt:grid (NT, HV)
        {
            let mut b = stream.launch_builder(&st.kkt);
            let ti = t as i32;
            let scratch: u64 = 0;
            b.arg(&k_p)
                .arg(&gcum_p)
                .arg(&beta_p)
                .arg(&a_p)
                .arg(&cu_p)
                .arg(&idx_p)
                .arg(&ti)
                .arg(&scratch)
                .arg(&scratch);
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (nt as u32, hv_u, 1),
                block_dim: (32 * owl_kernels::gdn_chunked::cubins::launch::KKT_WARPS, 1, 1),
                shared_mem_bytes: owl_kernels::gdn_chunked::cubins::launch::KKT_SHARED,
            };
            if let Err(e) = unsafe { b.launch(cfg) }.map_err(|e| ModelError::Msg(format!("gdn kkt: {e:?}"))) {
                return ack.send(Err(e));
            }
        }
        // 3. wu:grid (NT, HV)
        {
            let mut b = stream.launch_builder(&st.wu);
            let ti = t as i32;
            let scratch: u64 = 0;
            b.arg(&k_p)
                .arg(&v_p)
                .arg(&beta_p)
                .arg(&w_p)
                .arg(&u_p)
                .arg(&a_p)
                .arg(&gcum_p)
                .arg(&cu_p)
                .arg(&idx_p)
                .arg(&ti)
                .arg(&scratch)
                .arg(&scratch);
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (nt as u32, hv_u, 1),
                block_dim: (32 * owl_kernels::gdn_chunked::cubins::launch::WU_WARPS, 1, 1),
                shared_mem_bytes: owl_kernels::gdn_chunked::cubins::launch::WU_SHARED,
            };
            if let Err(e) = unsafe { b.launch(cfg) }.map_err(|e| ModelError::Msg(format!("gdn wu: {e:?}"))) {
                return ack.send(Err(e));
            }
        }
        // 4. h:grid (cdiv(VD,BV) × HV)(BV 烘焙;探针迭代定值,起步 32)
        {
            const H_BV: u32 = 64;
            let mut b = stream.launch_builder(&st.h);
            let ti = t as i32;
            let scratch: u64 = 0;
            b.arg(&k_p)
                .arg(&v_p)
                .arg(&w_p)
                .arg(&vnew_p)
                .arg(&gcum_p)
                .arg(&h_p)
                .arg(&state_p)
                .arg(&state_p)
                .arg(&cu_p)
                .arg(&coff_p)
                .arg(&ti)
                .arg(&scratch)
                .arg(&scratch);
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: ((vd as u32).div_ceil(H_BV) * hv_u, 1, 1),
                block_dim: (32 * owl_kernels::gdn_chunked::cubins::launch::H_WARPS, 1, 1),
                shared_mem_bytes: owl_kernels::gdn_chunked::cubins::launch::H_SHARED,
            };
            if let Err(e) = unsafe { b.launch(cfg) }.map_err(|e| ModelError::Msg(format!("gdn h: {e:?}"))) {
                return ack.send(Err(e));
            }
        }
        // 5. o:grid (cdiv(VD,BV2), NT, HV)(BV2 起步 32,探针迭代)
        {
            const O_BV: u32 = 64;
            let mut b = stream.launch_builder(&st.o);
            let ti = t as i32;
            let scratch: u64 = 0;
            b.arg(&q_p)
                .arg(&k_p)
                .arg(&vnew_p)
                .arg(&h_p)
                .arg(&gcum_p)
                .arg(&of32_p)
                .arg(&cu_p)
                .arg(&idx_p)
                .arg(&scale)
                .arg(&ti)
                .arg(&scratch)
                .arg(&scratch);
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: ((vd as u32).div_ceil(O_BV), nt as u32, hv_u),
                block_dim: (32 * owl_kernels::gdn_chunked::cubins::launch::O_WARPS, 1, 1),
                shared_mem_bytes: owl_kernels::gdn_chunked::cubins::launch::O_SHARED,
            };
            if let Err(e) = unsafe { b.launch(cfg) }.map_err(|e| ModelError::Msg(format!("gdn o: {e:?}"))) {
                return ack.send(Err(e));
            }
        }
        // 6. cast o_f32 → out(f16)
        {
            let n_out = t * hv * vd;
            let out_p = ptrs[6] as u64;
            let n_out_i = n_out as i32;
            let mut b = stream.launch_builder(&st.cast_f32_f16);
            b.arg(&of32_p).arg(&n_out_i).arg(&out_p);
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (n_out.div_ceil(256) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            if let Err(e) = unsafe { b.launch(cfg) }.map_err(|e| ModelError::Msg(format!("gdn o cast: {e:?}"))) {
                return ack.send(Err(e));
            }
        }
        ack.send(Ok(Bytes::new(blocks[6].0, msg.out_elems)))
    }

    /// GDN scalar 臂(lmdeploy pre_sm90 port 单核;2026-10-03)。
    /// 槽序:[T q, T k, T v, T g, T beta, T state, O out,
    /// sz T, sz slot, sz ns, sz hv, sz nk, sz kd, sz scale_bits]
    /// = 7 Block + 7 sz。q/k/v/g/beta = 层侧 CAST_F16_F32 产物(f32 块),
    /// 内核直接在块指针上发射(零中间拷贝);state 槽寻址同 chunked 臂;
    /// o_f32 私有 scratch → cast → out(f16)。当前单序列(ns=1,seq_off=[0,T])。
    pub(super) fn handle_gdn_scalar(&mut self, msg: LaunchMsg, ack: Ack<Result<Bytes, ModelError>>) {
        let init = || -> Result<GdnScalarState, ModelError> {
            if self.ctx.as_ref().map(|c| c.capture_stream()).unwrap_or(false) {
                return Err(ModelError::Msg(
                    "gdn_scalar 捕获内首次初始化(需先 eager warmup)".into(),
                ));
            }
            let stream = self.ctx().stream(STREAM_COMPUTE)?.clone();
            let m = self
                .ctx()
                .ctx
                .load_module(cudarc::nvrtc::Ptx::from_binary(
                    owl_kernels::gdn_scalar::cubin::CHUNK_SCALAR_F32.to_vec(),
                ))
                .map_err(|e| ModelError::Msg(format!("gdn_scalar load: {e:?}")))?;
            let k = std::sync::Arc::new(
                m.load_function(owl_kernels::gdn_scalar::cubin::KERNEL_F32)
                    .map_err(|e| ModelError::Msg(format!("gdn_scalar fn: {e:?}")))?,
            );
            // cast 模块(nvrtc;与 gdn_chunked 臂同源 CAST)
            use cudarc::nvrtc::CompileOptions;
            let ptx = cudarc::nvrtc::compile_ptx_with_opts(
                owl_kernels::sources::attention::CAST,
                CompileOptions {
                    include_paths: vec!["/usr/local/cuda/include".to_string()],
                    ..Default::default()
                },
            )
            .map_err(|e| ModelError::Msg(format!("gdn_scalar cast nvrtc: {e:?}")))?;
            let cast_m = self
                .ctx()
                .ctx
                .load_module(cudarc::nvrtc::Ptx::from_src(ptx.to_src()))
                .map_err(|e| ModelError::Msg(format!("gdn_scalar cast load: {e:?}")))?;
            let cast_f32_f16 = std::sync::Arc::new(
                cast_m
                    .load_function("owl_cast_f32_f16")
                    .map_err(|e| ModelError::Msg(format!("gdn_scalar cast fn: {e:?}")))?,
            );
            Ok(GdnScalarState {
                k,
                cast_f32_f16,
                o_f32: stream
                    .alloc_zeros::<f32>(1)
                    .map_err(|e| ModelError::Msg(format!("{e:?}")))?,
                soff: stream
                    .alloc_zeros::<i32>(2)
                    .map_err(|e| ModelError::Msg(format!("{e:?}")))?,
                t_cap: 0,
            })
        };
        if self.gdn_scalar.is_none() {
            self.gdn_scalar = match init() {
                Ok(v) => Some(v),
                Err(e) => return ack.send(Err(e)),
            };
        }
        // 槽序解析:7 Block + 7 sz(q/k/v/g/beta/state/out + T/slot/ns/hv/nk/kd/scale_bits)
        let (blocks, scalars) = match Self::parse_foreign_slots(&msg, 7, 7) {
            Ok(v) => v,
            Err(e) => return ack.send(Err(e)),
        };
 let (t, slot, ns, hv, nk, kd, scale_bits) = (
            scalars[0] as usize,
            scalars[1] as usize,
            scalars[2] as usize,
            scalars[3] as usize,
            scalars[4] as usize,
            scalars[5] as usize,
            scalars[6] as u32,
        );
        if ns != 1 {
            return ack.send(Err(ModelError::Msg(
                "gdn_scalar 当前单序列(ns=1;varlen 并发批待 M2)".into(),
            )));
        }
        let scale = f32::from_bits(scale_bits);
        let stream = match self.ctx().stream(STREAM_COMPUTE) {
            Ok(s) => s.clone(),
            Err(e) => return ack.send(Err(e)),
        };
        let mut ptrs = Vec::with_capacity(7);
        for (b, off) in &blocks {
            match self.ctx().block_ptr(*b, &stream) {
                Ok((p, _)) => ptrs.push(p + off),
                Err(e) => return ack.send(Err(e)),
            }
        }
        let st = self.gdn_scalar.as_mut().unwrap();
        if t > st.t_cap {
            match stream.alloc_zeros::<f32>(t * hv * kd) {
                Ok(v) => {
                    st.o_f32 = v;
                    st.t_cap = t;
                }
                Err(e) => return ack.send(Err(ModelError::Msg(format!("{e:?}")))),
            }
        }
        // seq_off htod(COMPUTE 流,先于内核):单序列 [0, T]
        let soff_p = gdn_dptr(&mut st.soff, &stream);
        if let Err(e) = unsafe { memcpy_htod_async(soff_p, &[0i32, t as i32], stream.cu_stream()) } {
            return ack.send(Err(ModelError::Msg(format!("gdn_scalar soff htod: {e:?}"))));
        }
        // 内核发射:直接在层侧 f32 块指针上(q/k/v/g/beta = ptrs[0..5],
        // state = ptrs[5] + slot 解址;内参序 = extern C 声明序)
        let state_p = ptrs[5] + (slot * hv * kd * kd * 4) as u64; // f32 → 字节
        let (q_p, k_p, v_p, g_p, beta_p) =
            (ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4]);
        let o32_p = gdn_dptr(&mut st.o_f32, &stream);
        {
            let mut b = stream.launch_builder(&st.k);
            let ti = t as i32;
            let nk_i = nk as i32;
            let hv_i = hv as i32;
            let kd_i = kd as i32;
            b.arg(&o32_p)
                .arg(&q_p)
                .arg(&k_p)
                .arg(&v_p)
                .arg(&beta_p)
                .arg(&g_p)
                .arg(&state_p)
                .arg(&soff_p)
                .arg(&nk_i)
                .arg(&hv_i)
                .arg(&kd_i)
                .arg(&scale);
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (ns as u32, hv as u32, 1),
                block_dim: (owl_kernels::gdn_scalar::cubin::BLOCK, 1, 1),
                shared_mem_bytes: owl_kernels::gdn_scalar::cubin::SMEM_D128,
            };
            if let Err(e) =
                unsafe { b.launch(cfg) }.map_err(|e| ModelError::Msg(format!("gdn_scalar: {e:?}")))
            {
                return ack.send(Err(e));
            }
        }
        // cast o_f32 → out(f16)
        {
            let n_out = t * hv * kd;
            let out_p = ptrs[6];
            let n_out_i = n_out as i32;
            let mut b = stream.launch_builder(&st.cast_f32_f16);
            b.arg(&o32_p).arg(&n_out_i).arg(&out_p);
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (n_out.div_ceil(256) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            if let Err(e) = unsafe { b.launch(cfg) }
                .map_err(|e| ModelError::Msg(format!("gdn_scalar cast: {e:?}")))
            {
                return ack.send(Err(e));
            }
        }
        ack.send(Ok(Bytes::new(blocks[6].0, msg.out_elems)))
    }

    /// Marlin AWQ(kU4 非对称)臂(2026-10-01 cyankiwi g32 装载线)。
    /// 槽序:[T a, T b, T out, T scales, T zeros, T ws, T ctmp,
    /// sz m, sz k, sz n, sz groupsize](7 Block + 4 sz);
    /// zeros = pack_marlin_z 产物((k/g, n/8) i32)。
    pub(super) fn handle_marlin_gemm_awq(&mut self, msg: LaunchMsg, ack: Ack<Result<Bytes, ModelError>>) {
        let (blocks, scalars) = match Self::parse_foreign_slots(&msg, 7, 4) {
            Ok(v) => v,
            Err(e) => return ack.send(Err(e)),
        };
        let (m, k, n, groupsize) =
            (scalars[0] as usize, scalars[1] as usize, scalars[2] as usize, scalars[3] as i32);
        let stream = match self.ctx().stream(STREAM_COMPUTE) {
            Ok(s) => s.clone(),
            Err(e) => return ack.send(Err(e)),
        };
        let mut ptrs = Vec::with_capacity(7);
        for (b, off) in &blocks {
            match self.ctx().block_ptr(*b, &stream) {
                Ok((p, _)) => ptrs.push(p + off),
                Err(e) => return ack.send(Err(e)),
            }
        }
        let dev = self.ctx().device_ordinal() as i32;
        let r = unsafe {
            owl_kernels::marlin::gemm_v2_awq_raw(
                ptrs[0] as *const u16,
                ptrs[1] as *const i32,
                ptrs[2] as *mut u16,
                ptrs[3] as *const u16,
                ptrs[4] as *const i32,
                ptrs[6] as *const c_void,
                m as i32,
                n as i32,
                k as i32,
                ptrs[5] as *mut i32,
                groupsize,
                dev,
                stream.cu_stream() as usize,
            )
        };
        match r {
            Ok(()) => ack.send(Ok(Bytes::new(blocks[2].0, msg.out_elems))),
            Err(e) => ack.send(Err(ModelError::Msg(format!(
                "marlin awq gemm err {e}: {}",
                owl_kernels::marlin::v2_err_str(e)
            )))),
        }
    }


}

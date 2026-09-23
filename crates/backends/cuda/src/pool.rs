//! 显存池:**分配者**。持有 driver 原语句柄,按 PoolKind 路由物理路径
//! (PeerShared → VMM 2MiB 粒度,其余 → stream-ordered)。

use super::governor::Governor;
use crate::device::CudaDevice;
use cudarc::driver::sys;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use owl_iface::{
    BackendError, BufToken, DevBuf, Device, MemPhase, MemValue, OpaqueDevBuf, Pool, PoolBuf,
    PoolId, PoolKind, PoolUsage,
};
use parking_lot::Mutex;
use std::sync::Arc;

/// 池账目记录(Governor 持有;CudaPool 是它的对外视图)
pub(crate) struct PoolLedger {
    pub(crate) name: String,
    pub(crate) kind: PoolKind,
    pub(crate) capacity: u64,
    pub(crate) used: u64,
    pub(crate) peak: u64,
}

/// 显存池:**分配者**。持有 driver 原语句柄,按 PoolKind 路由物理路径。
#[derive(Clone)]
pub struct CudaPool {
    pub(crate) id: PoolId,
    pub(crate) name: String,
    pub(crate) kind: PoolKind,
    pub(crate) capacity: u64,
    pub(crate) ctx: Arc<CudaContext>,
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) gov: Arc<Governor>,
    pub(crate) dev: CudaDevice,
}

impl Pool for CudaPool {
    type Dev = CudaDevice;

    fn device(&self) -> Self::Dev {
        self.dev.clone()
    }

    fn id(&self) -> PoolId {
        self.id
    }
    fn name(&self) -> &str {
        &self.name
    }
    fn kind(&self) -> PoolKind {
        self.kind
    }
    fn capacity(&self) -> u64 {
        self.capacity
    }
    fn usage(&self) -> PoolUsage {
        let pools = self.gov.pools.lock();
        let led = pools.get(&self.id.0).expect("池账目丢失");
        PoolUsage {
            capacity: led.capacity,
            used: led.used,
            peak: led.peak,
        }
    }

    fn malloc_scratch(&self, bytes: u64) -> Result<PoolBuf, BackendError> {
        Ok(PoolBuf::wrap(Box::new(self.malloc_scratch_buf(bytes)?)))
    }

    fn malloc_persistent(&self, bytes: u64) -> Result<PoolBuf, BackendError> {
        self.kind_check_any(&[PoolKind::Weights, PoolKind::KvCache, PoolKind::Workspace])?;
        Ok(PoolBuf::wrap(Box::new(self.malloc_inner(bytes)?)))
    }

    fn malloc_peer_shared(&self, bytes: u64) -> Result<PoolBuf, BackendError> {
        self.kind_check(PoolKind::PeerShared)?;
        Ok(PoolBuf::wrap(Box::new(self.malloc_inner(bytes)?)))
    }

    fn alloc_persistent_in<T: MemValue>(
        &self,
        len: usize,
    ) -> Result<<CudaDevice as Device>::Persistent<T>, BackendError> {
        let bytes = (len * std::mem::size_of::<T>()) as u64;
        let buf = self.malloc_persistent_buf(bytes)?;
        self.gov.stats.lock().persistent_allocs += 1;
        Ok(crate::buffers::Persistent::new(buf, len))
    }

    fn htod_persistent_in<T: MemValue>(
        &self,
        src: Vec<T>,
    ) -> Result<<CudaDevice as Device>::Persistent<T>, BackendError> {
        let len = src.len();
        let bytes = (len * std::mem::size_of::<T>()) as u64;
        let buf = self.malloc_persistent_buf(bytes)?;
        self.ctx.bind_to_thread().map_err(|e| BackendError::Init(format!("{e:?}")))?;
        // P0-3 口径:H2D 必须走池流(async + sync),NULL 流与主流无序
        // (审计同族案:cu_seqlens 清零/cat 脏数据)
        unsafe {
            use cudarc::driver::sys;
            sys::cuMemcpyHtoDAsync_v2(
                buf.device_ptr() as sys::CUdeviceptr,
                src.as_ptr() as *const core::ffi::c_void,
                bytes as usize,
                self.stream.cu_stream(),
            )
            .result()
            .map_err(|e| BackendError::CopyFailed { dir: "htod", detail: format!("{e:?}") })?;
            self.stream.synchronize().map_err(|e| BackendError::CopyFailed {
                dir: "htod",
                detail: format!("stream sync: {e:?}"),
            })?;
        }
        self.gov.stats.lock().persistent_allocs += 1;
        Ok(crate::buffers::Persistent::new(buf, len))
    }

    fn alloc_scratch_in<T: MemValue>(
        &self,
        len: usize,
    ) -> Result<<CudaDevice as Device>::Scratch<T>, BackendError> {
        let bytes = (len * std::mem::size_of::<T>()) as u64;
        let buf = self.malloc_scratch_buf(bytes)?;
        self.gov.stats.lock().persistent_allocs += 1;
        Ok(crate::buffers::Scratch::new(buf, len))
    }
}

impl CudaPool {
    fn kind_check_any(&self, kinds: &[PoolKind]) -> Result<(), BackendError> {
        if kinds.contains(&self.kind) {
            Ok(())
        } else {
            Err(BackendError::LawViolation(
                "分配语义与池类型不匹配(语义错配是架构错误)",
            ))
        }
    }

    fn kind_check(&self, expect: PoolKind) -> Result<(), BackendError> {
        if self.kind != expect {
            return Err(BackendError::LawViolation(
                "分配语义与池类型不匹配(语义错配是架构错误)",
            ));
        }
        Ok(())
    }

    /// 归还全局账本
    pub(crate) fn uncharge_global(&self, bytes: u64) {
        self.gov.uncharge(bytes);
    }

    /// 归还池账本
    pub(crate) fn uncharge_pool(&self, bytes: u64) {
        self.gov.uncharge_pool(self.id.0, bytes);
    }

    /// 内部:持久域具体分配(语义校验 + Arc 租约缓冲)
    pub(crate) fn malloc_persistent_buf(&self, bytes: u64) -> Result<CudaPoolBuf, BackendError> {
        self.kind_check_any(&[PoolKind::Weights, PoolKind::KvCache, PoolKind::Workspace])?;
        self.malloc_inner(bytes)
    }

    /// 内部:暂存域具体分配
    pub(crate) fn malloc_scratch_buf(&self, bytes: u64) -> Result<CudaPoolBuf, BackendError> {
        self.kind_check(PoolKind::Scratch)?;
        self.malloc_inner(bytes)
    }

    /// 内部统一分配路径:校验链(kind 可选)→ 池账 → 全局账 → 物理 →
    /// 签发租约型 CudaPoolBuf(Arc;Clone 即租约)。Persistent/Scratch/
    /// CaptureSession 全部经由这里。
    fn malloc_inner(&self, bytes: u64) -> Result<CudaPoolBuf, BackendError> {
        // A2.8:PeerShared 走 VMM,粒度取整后的真实占用必须先入账
        let effective = match self.kind {
            PoolKind::PeerShared => {
                let gran = self.vmm_granularity()? as u64;
                bytes.div_ceil(gran) * gran
            }
            _ => bytes,
        };
        self.gov.charge_pool(self, effective)?;
        // charge 校验先行(失败不触账本)→ 此处失败无需回滚
        self.gov.charge(effective)?;
        let backing = match self.kind {
            PoolKind::PeerShared => match self.vmm_alloc_raw(effective as usize) {
                Ok((b, r)) => {
                    debug_assert_eq!(r as u64, effective);
                    b
                }
                Err(e) => {
                    // 物理分配失败:两本账都已入账,回滚
                    self.uncharge_global(effective);
                    self.uncharge_pool(effective);
                    return Err(e);
                }
            },
            _ => match self.stream.alloc_zeros::<u8>(effective as usize) {
                Ok(slice) => PoolBacking::Slice(slice),
                Err(e) => {
                    self.uncharge_global(effective);
                    self.uncharge_pool(effective);
                    return Err(BackendError::AllocFailed {
                        kind: "pool-malloc",
                        len: effective as usize,
                        detail: e.to_string(),
                    });
                }
            },
        };
        let token = self.gov.issue_token();
        // 哨兵①:地址区间登记用 base;先取基址再装配
        let base = match &backing {
            PoolBacking::Slice(slice) => {
                use cudarc::driver::DevicePtr;
                let (p, _sync) = slice.device_ptr(&self.stream);
                p as u64
            }
            PoolBacking::Vmm { ptr, .. } => *ptr,
            PoolBacking::Empty => unreachable!("PoolBuf 已被消费"),
        };
        let buf = CudaPoolBuf {
            inner: Arc::new(PoolBufInner {
                backing: Mutex::new(backing),
                bytes: effective,
                base,
                token: token.clone(),
                pool: Arc::new(self.clone_account()),
                gov: Arc::clone(&self.gov),
            }),
            token,
        };
        // 哨兵①:区间索引(emit_by_ptr 反查) + 存活登记表(Weak 反查升级——
        // 捕获自动租约的通道;只 Weak 不强持,强保活归租约持有方,
        // 2026-09-22 判例:登记表持强引用 → "最后句柄"永不成立 → 账本泄漏)
        self.gov
            .intervals
            .lock()
            .insert(base, (effective, token.id));
        self.gov
            .live_bufs
            .lock()
            .insert(token.id, Arc::downgrade(&buf.inner));
        Ok(buf)
    }

    fn clone_account(&self) -> CudaPool {
        self.clone()
    }

    /// A2.8 设备最小分配粒度(sm86 实测 2MiB)
    fn vmm_granularity(&self) -> Result<usize, BackendError> {
        use sys::{cuMemGetAllocationGranularity, CUmemAllocationProp, CUmemLocation};
        unsafe {
            let mut props = CUmemAllocationProp {
                type_: sys::CUmemAllocationType_enum::CU_MEM_ALLOCATION_TYPE_PINNED,
                requestedHandleTypes:
                    sys::CUmemAllocationHandleType_enum::CU_MEM_HANDLE_TYPE_NONE,
                location: CUmemLocation {
                    type_: sys::CUmemLocationType_enum::CU_MEM_LOCATION_TYPE_DEVICE,
                    __bindgen_anon_1: sys::CUmemLocation_st__bindgen_ty_1 {
                        id: self.ctx.cu_device() as i32,
                    },
                },
                win32HandleMetaData: std::ptr::null_mut(),
                allocFlags: sys::CUmemAllocationProp_st__bindgen_ty_1 {
                    compressionType: 0,
                    gpuDirectRDMACapable: 0,
                    usage: 0,
                    reserved: [0; 4],
                },
            };
            props.location.__bindgen_anon_1.id = self.ctx.cu_device() as i32;
            let mut gran: usize = 0;
            cuMemGetAllocationGranularity(
                &mut gran,
                &props,
                sys::CUmemAllocationGranularity_flags_enum::CU_MEM_ALLOC_GRANULARITY_MINIMUM,
            )
            .result()
            .map_err(|e| BackendError::Init(format!("granularity: {e:?}")))?;
            Ok(gran)
        }
    }

    /// A2.8:VMM 物理分配(reserve→create→map→setAccess RW)。
    /// 返回未封装句柄;调用方负责包装与销账。
    pub(crate) fn vmm_alloc_raw(
        &self,
        bytes: usize,
    ) -> Result<(PoolBacking, usize), BackendError> {
        use sys::{
            cuMemAddressFree, cuMemAddressReserve, cuMemCreate, cuMemMap, cuMemSetAccess,
            cuMemUnmap, cuMemRelease, cuMemGetAllocationGranularity, CUmemAccessDesc,
            CUmemAllocationProp, CUmemLocation,
        };
        unsafe {
            self.ctx
                .bind_to_thread()
                .map_err(|e| BackendError::Init(format!("{e:?}")))?;
            let mut gran: usize = 0;
            let mut props = CUmemAllocationProp {
                type_: sys::CUmemAllocationType_enum::CU_MEM_ALLOCATION_TYPE_PINNED,
                requestedHandleTypes:
                    sys::CUmemAllocationHandleType_enum::CU_MEM_HANDLE_TYPE_NONE,
                location: CUmemLocation {
                    type_: sys::CUmemLocationType_enum::CU_MEM_LOCATION_TYPE_DEVICE,
                    __bindgen_anon_1: sys::CUmemLocation_st__bindgen_ty_1 {
                        id: self.ctx.cu_device() as i32,
                    },
                },
                win32HandleMetaData: std::ptr::null_mut(),
                allocFlags: sys::CUmemAllocationProp_st__bindgen_ty_1 {
                    compressionType: 0,
                    gpuDirectRDMACapable: 0,
                    usage: 0,
                    reserved: [0; 4],
                },
            };
            props.location.__bindgen_anon_1.id = self.ctx.cu_device() as i32;
            cuMemGetAllocationGranularity(
                &mut gran,
                &props,
                sys::CUmemAllocationGranularity_flags_enum::CU_MEM_ALLOC_GRANULARITY_MINIMUM,
            )
            .result()
            .map_err(|e| BackendError::Init(format!("granularity: {e:?}")))?;
            let rounded = bytes.div_ceil(gran) * gran;

            let mut ptr: sys::CUdeviceptr = 0;
            if cuMemAddressReserve(&mut ptr, rounded, gran, 0, 0) != sys::CUresult::CUDA_SUCCESS {
                return Err(BackendError::AllocFailed {
                    kind: "vmm-reserve",
                    len: rounded,
                    detail: "cuMemAddressReserve".into(),
                });
            }
            let mut chunk: sys::CUmemGenericAllocationHandle = Default::default();
            if cuMemCreate(&mut chunk, rounded, &props, 0) != sys::CUresult::CUDA_SUCCESS {
                let _ = cuMemAddressFree(ptr, rounded);
                return Err(BackendError::AllocFailed {
                    kind: "vmm-create",
                    len: rounded,
                    detail: "cuMemCreate".into(),
                });
            }
            if cuMemMap(ptr, rounded, 0, chunk, 0) != sys::CUresult::CUDA_SUCCESS {
                let _ = cuMemRelease(chunk);
                let _ = cuMemAddressFree(ptr, rounded);
                return Err(BackendError::AllocFailed {
                    kind: "vmm-map",
                    len: rounded,
                    detail: "cuMemMap".into(),
                });
            }
            let access = [CUmemAccessDesc {
                location: props.location,
                flags: sys::CUmemAccess_flags_enum::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
            }];
            if cuMemSetAccess(ptr, rounded, access.as_ptr(), 1) != sys::CUresult::CUDA_SUCCESS {
                let _ = cuMemUnmap(ptr, rounded);
                let _ = cuMemRelease(chunk);
                let _ = cuMemAddressFree(ptr, rounded);
                return Err(BackendError::AllocFailed {
                    kind: "vmm-set-access",
                    len: rounded,
                    detail: "cuMemSetAccess".into(),
                });
            }
            Ok((
                PoolBacking::Vmm {
                    ptr,
                    bytes: rounded,
                    chunk,
                    _ctx: Arc::clone(&self.ctx),
                },
                rounded,
            ))
        }
    }
}

/// 池缓冲的物理背面:stream-ordered 切片或 VMM 映射
///
/// Drop 统一收口:Slice 交还 cudarc 的 stream-ordered 生命周期;
/// VMM 走 unmap → release → address_free 三步(无此 Drop 会物理泄漏)。
pub(crate) enum PoolBacking {
    Empty,
    Slice(CudaSlice<u8>),
    Vmm {
        ptr: sys::CUdeviceptr,
        bytes: usize,
        chunk: sys::CUmemGenericAllocationHandle,
        /// 保活分配时的主上下文(防 ctx 先于缓冲销毁)
        _ctx: Arc<CudaContext>,
    },
}

impl Drop for PoolBacking {
    fn drop(&mut self) {
        if let PoolBacking::Vmm { ptr, bytes, chunk, _ctx } = self {
            let _ctx = Arc::clone(_ctx); // 保活至清理完成
            let _ = _ctx.bind_to_thread(); // P1-A
            unsafe {
                use sys::{cuMemAddressFree, cuMemRelease, cuMemUnmap};
                if cuMemUnmap(*ptr, *bytes) != sys::CUresult::CUDA_SUCCESS {
                    // Drop 上下文不可 panic(析构违约);统一 [owl-vmm-leak]
                    // 前缀供运维 grep,ptr/bytes 供泄漏追责。
                    eprintln!("[owl-vmm-leak] cuMemUnmap 失败 ptr={:p} bytes={bytes}", *ptr as *const core::ffi::c_void);
                }
                if cuMemRelease(*chunk) != sys::CUresult::CUDA_SUCCESS {
                    eprintln!("[owl-vmm-leak] cuMemRelease 失败 chunk={chunk:?}");
                }
                if cuMemAddressFree(*ptr, *bytes) != sys::CUresult::CUDA_SUCCESS {
                    eprintln!("[owl-vmm-leak] cuMemAddressFree 失败 ptr={:p} bytes={bytes}", *ptr as *const core::ffi::c_void);
                }
            }
        }
    }
}

/// 池缓冲句柄:**Arc 共享所有权 = 租约系统**。
/// Clone(租约)把 Arc 计数 +1——图捕获期 lease 进 CapturedGraph 的
/// keepalive,用户侧句柄先 drop 也不会进入回收流程(强租约保证)。
/// 身份(token retire)只随**最后一个**句柄的 drop 消亡。
#[derive(Clone)]
pub struct CudaPoolBuf {
    pub(crate) inner: Arc<PoolBufInner>,
    pub(crate) token: BufToken,
}

pub(crate) struct PoolBufInner {
    pub(crate) backing: Mutex<PoolBacking>,
    pub(crate) bytes: u64,
    /// 设备基址(A2.6 窄口/哨兵①:区间索引键;Slice 于创建时取,VMM 为 ptr)
    pub(crate) base: u64,
    /// 缓冲身份(P1-C:retire 落点在本结构 drop,需随行)
    pub(crate) token: BufToken,
    pub(crate) pool: Arc<CudaPool>,
    pub(crate) gov: Arc<Governor>,
}

impl Drop for PoolBufInner {
    fn drop(&mut self) {
        // P1-C:身份注销落点 = Arc 计数真正归零处。原先放在 CudaPoolBuf::drop
        // 的 count==1 判定,在"keepalive 裸 Arc 是最后一个引用"的 lease-last
        // 路径永不触发 → 死令牌 validate 为活 + live_bufs 泄漏。
        self.gov.retire(&self.token);
        self.gov.intervals.lock().remove(&self.base);
        // 仅最后一个句柄消亡时到达此处(Arc 计数归零)
        let bytes = self.bytes;
        let pool = Arc::clone(&self.pool);
        let gov = Arc::clone(&self.gov);
        let backing = std::mem::replace(self.backing.get_mut(), PoolBacking::Empty);
        let idle = gov.phase() == MemPhase::Idle;
        let gov_for_closure = Arc::clone(&gov);
        if idle {
            release_backing(backing, &pool);
            gov.uncharge(bytes);
            pool.uncharge_pool(bytes);
        } else {
            gov.defer(Box::new(move || {
                release_backing(backing, &pool);
                gov_for_closure.uncharge(bytes);
                pool.uncharge_pool(bytes);
            }));
        }
    }
}

/// 物理回收统一收口。
///
/// P1-A:bind_to_thread(A3)前置——释放线程可能是协调/测试线程,
/// 当前 ctx 未必是本池 ctx(或无当前 ctx),裸调会被驱动静默拒绝。
/// P1-B:VMM unmap 即时生效(非 stream-ordered),先排空上下文所有流,
/// 防净空窗口 in-flight 访问竞态;Slice 走 free_async 自带排序,无此险。
fn release_backing(backing: PoolBacking, pool: &CudaPool) {
    let is_vmm = matches!(backing, PoolBacking::Vmm { .. });
    let _ = pool.ctx.bind_to_thread();

    // P0-4(debug 归还哨兵;audit/memory-backend.md):归还即整块填 0xFF
    // ——f32 位型 = NaN、u32 位型 = u32::MAX(兼容 S4 哨兵)。任何悬空视图
    // (租约归还后仍持有的裸指针)再读立即现形,而非静默读旧值/脏值。
    // 实现:主流 async memset + 流同步。选同步而非纯 async:归还发生在
    // 主机侧任意线程,纯 async 无法保证 drop 返回后哨兵已生效(host 侧
    // 悬空读会与 memset 竞速,探测器自身变成非确定性);debug-only,
    // 同步等待的开销可接受。release 构建零开销(整段 cfg 掉)。
    // 探测器门控:OWL_POISON_FREED=1 时启用(默认关)。
    // 已验证:开启后 dry_run 判别③红(P1-2 立案证据,见 docs/audit/README.md);
    // P1-2(捕获窗口 scratch 租约缺口)修复后应默认开启。
    #[cfg(any())] // 探测器门控见上注(OWL_POISON_FREED 方案待 P1-2 修复后接线)
    {
        use cudarc::driver::{DevicePtr};
        match &backing {
            PoolBacking::Slice(s) => {
                let (p, _sync) = s.device_ptr(&pool.stream);
                unsafe {
                    cudarc::driver::result::memset_d8_async(
                        p as sys::CUdeviceptr,
                        0xFF,
                        s.len(),
                        pool.stream.cu_stream(),
                    )
                }
                .expect("P0-4 归还哨兵:memset_d8_async 失败");
            }
            PoolBacking::Vmm { ptr, bytes, .. } => unsafe {
                cudarc::driver::result::memset_d8_async(
                    *ptr as sys::CUdeviceptr,
                    0xFF,
                    *bytes,
                    pool.stream.cu_stream(),
                )
            }
            .expect("P0-4 归还哨兵:VMM memset_d8_async 失败"),
            PoolBacking::Empty => {}
        }
        pool.stream.synchronize().expect("P0-4 归还哨兵:流同步失败");
    }

    if is_vmm {
        let _ = pool.ctx.synchronize();
    }
    drop(backing);
}

impl CudaPoolBuf {
    /// 令牌读取(哨兵①登记/校验用)
    pub(crate) fn token(&self) -> BufToken {
        self.token
    }
}

// 身份注销 retire 已移入 PoolBufInner::drop(P1-C:唯一真正的归零判定点);
impl OpaqueDevBuf for CudaPoolBuf {
    fn token(&self) -> BufToken {
        self.token
    }
}

impl DevBuf<u8> for CudaPoolBuf {
    fn len(&self) -> usize {
        self.inner.bytes as usize
    }
    fn device_ptr(&self) -> *mut u8 {
        use cudarc::driver::DevicePtr;
        let backing = self.inner.backing.lock();
        match &*backing {
            PoolBacking::Slice(s) => {
                let (ptr, _sync) = s.device_ptr(&self.inner.pool.stream);
                ptr as *mut u8
            }
            PoolBacking::Vmm { ptr, .. } => *ptr as *mut u8,
            PoolBacking::Empty => unreachable!("PoolBuf 已被消费"),
        }
    }
}

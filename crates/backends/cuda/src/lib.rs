//! owl-cuda —— cudarc **官方版**(crates.io)的治理封装层,实现
//! `owl_iface` 的 `Backend`/`Device`/`Pool` 三层契约(charter A1/A4/A5)。
//!
//! 选型注记(2026-09-22):官方 cudarc 0.19 已原生覆盖 fork 的图安全增量
//! (CudaGraph/stream-ordered 分配/13.x 绑定),按 A4"语义分歧不进第三方
//! 库"选官方。cudarc 仅在本 crate 导入;上层经 `ffi` 模块的精确白名单
//! 使用(整库 re-export 禁止)。
//!
//! 分配架构(裁决 5 + A5.2):
//! - **Pool 是分配者**:`CudaPool::malloc` 是 P 阶段唯一分配入口,
//!   校验链 = 池余量(A5.4)→ 全局预算(A5.4)→ 物理分配;
//! - 物理路径按池类型路由:`PeerShared` → VMM(cuMemCreate,2MiB 粒度,
//!   A2.8);其余 → stream-ordered(捕获安全);
//! - 所有缓冲 drop 时自动归池账 + 全局账(非 Idle 相延迟到净空窗口,
//!   A1.2);
//! - Device 的 `alloc_*_in` 只是 `pool.malloc` 的类型化薄封装。

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use cudarc::driver::{DevicePtr, sys};
use owl_iface::{MemPhase as _, Pool as _, PoolBuf as _};
use owl_iface::{
    Arch, Backend, BackendError, BackendFamily, BufToken, Device, DeviceDesc, DevBuf, MemValue,
    OpaqueDevBuf, Pool, PoolBuf, PoolConfig, PoolId, PoolKind, PoolUsage,
};
use parking_lot::{Mutex, RwLock};
use std::sync::Arc;

pub use owl_iface::{MemPhase, MemStats};

/// cudarc 受控再导出(A4:精确到接口粒度,禁止整库 re-export)。
/// 这是上层(nn/未来的 rocnn)唯一可见的 cudarc 表面。
/// 扩充本清单 = 扩大 driver 依赖面,须过 backends/README 准入审核并登记。
pub mod ffi {
    // driver safe 层(逐项)
    pub use cudarc::driver::{
        CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr,
        LaunchConfig, PushKernelArg,
    };
    /// result 同步原语(逐项;当前仅 D2H 回读)
    pub use cudarc::driver::result::memcpy_dtoh_sync;

    /// 裸 FFI(唯一低层出口;逐项白名单)
    pub mod sys {
        // driver 侧:指针类型 + 显式拷贝(tensor 装载/回读)
        pub use cudarc::driver::sys::{
            CUdeviceptr, cuMemcpyDtoH_v2, cuMemcpyHtoD_v2,
            // graph 捕获(M1 归入 graph 治理层)
            CUgraphInstantiate_flags, CUstreamCaptureMode,
        };

        /// cuBLAS FFI(matmul/workspace)
        pub mod cublas {
            pub use cudarc::cublas::sys::{
                cublasCreate_v2, cublasDestroy_v2, cublasHandle_t,
                cublasOperation_t, cublasSetStream_v2, cublasSetWorkspace_v2,
                cublasSgemm_v2, cublasStatus_t,
            };
        }
    }

    /// nvrtc 运行时编译(kernel 加载)
    pub mod nvrtc {
        pub use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
    }
}

// ---- 治理句柄:相位机 + 双账本 + 延迟队列 ----

type DeferredFree = Box<dyn FnOnce() + Send>;

#[derive(Default)]
struct Governor {
    ctx: Option<Arc<CudaContext>>,
    phase: RwLock<MemPhase>,
    deferred: Mutex<Vec<DeferredFree>>,
    stats: Mutex<MemStats>,
    /// A5.2 全局账本:存活字节 / 累计分配字节
    bytes_alive: Mutex<u64>,
    bytes_allocated_total: Mutex<u64>,
    /// A5.1 预算(None = 未设;设置后每次分配断言不超)
    budget: Mutex<Option<Budget>>,
    /// 显存池账本(A5.2);Arc 供池对象跨线程归账
    pools: Arc<Mutex<std::collections::HashMap<u64, PoolLedger>>>,
    next_pool_id: std::sync::atomic::AtomicU64,
    /// 哨兵①词汇:缓冲令牌发放与存活登记(id → gen)
    next_buf_id: std::sync::atomic::AtomicU64,
    alive: Mutex<std::collections::HashMap<u64, u64>>,
}

impl Governor {
    fn phase(&self) -> MemPhase {
        *self.phase.read()
    }

    fn set_phase(&self, phase: MemPhase) {
        *self.phase.write() = phase;
        if phase == MemPhase::Idle {
            // 净空窗口:统一归还延迟队列(A1.2);归还动作由各缓冲自己的
            // 闭包完成(含池账 + 全局账)
            let mut q = self.deferred.lock();
            let n = q.len();
            for free in q.drain(..) {
                free();
            }
            self.stats.lock().drained_frees += n as u64;
        }
    }

    fn defer(&self, free: DeferredFree) {
        self.stats.lock().deferred_frees += 1;
        self.deferred.lock().push(free);
    }

    fn charge(&self, bytes: u64) -> Result<(), BackendError> {
        let mut alive = self.bytes_alive.lock();
        *alive += bytes;
        *self.bytes_allocated_total.lock() += bytes;
        if let Some(b) = *self.budget.lock() {
            // A5 判据:引擎自身账本(含延迟滞留)≤ 预算。整卡 used 含
            // 其他进程/上下文,不能作为本引擎预算的判据
            if *alive > b.bytes {
                *alive -= bytes; // 违约:预记回滚
                return Err(BackendError::LawViolation(
                    "A5.4 显存超支(引擎账本超预算,详见账本快照日志)",
                ));
            }
            // A5.3 第三道闸:driver 侧对账,free 低于警线即告警
            if let Some(ctx) = &self.ctx {
                let (free, _total) = ctx
                    .mem_get_info()
                    .map_err(|e| BackendError::Init(format!("{e:?}")))?;
                if (free as u64) < b.reserve_floor {
                    eprintln!(
                        "[owl-mem] WARN: free {free}B < reserve_floor {}B(A5 警线,账本 alive={})",
                        b.reserve_floor, *alive
                    );
                }
            }
        }
        Ok(())
    }

    fn uncharge(&self, bytes: u64) {
        *self.bytes_alive.lock() -= bytes;
    }

    fn uncharge_pool(&self, id: u64, bytes: u64) {
        if let Some(led) = self.pools.lock().get_mut(&id) {
            led.used = led.used.saturating_sub(bytes);
        }
    }

    /// 哨兵①:签发缓冲令牌并登记存活
    fn issue_token(&self) -> BufToken {
        let id = self
            .next_buf_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let gen = id.wrapping_mul(2654435761) | 1;
        self.alive.lock().insert(id, gen);
        BufToken { id, gen }
    }

    /// 哨兵①:世代校验(replay 前检查;死亡令牌 = 结构化报错的依据)
    fn validate(&self, t: &BufToken) -> bool {
        self.alive.lock().get(&t.id).is_some_and(|&g| g == t.gen)
    }

    /// 哨兵①:令牌注销(drop 即死;延迟的只是物理回收,不是身份)
    fn retire(&self, t: &BufToken) {
        self.alive.lock().remove(&t.id);
    }

    /// 池余量校验(A5.4 第一道):池耗尽即违约。通过后调用方已占池账。
    fn charge_pool(&self, pool: &CudaPool, bytes: u64) -> Result<(), BackendError> {
        let mut pools = self.pools.lock();
        let led = pools
            .get_mut(&pool.id.0)
            .ok_or(BackendError::UnknownPool(pool.id.0))?;
        let available = led.capacity - led.used;
        if bytes > available {
            return Err(BackendError::PoolExhausted {
                pool: led.name.clone(),
                needed: bytes,
                available,
                capacity: led.capacity,
            });
        }
        led.used += bytes;
        led.peak = led.peak.max(led.used);
        Ok(())
    }
}

/// A5 硬预算:启动时声明,生命周期恒不超(见 charter 公理 A5)。
#[derive(Debug, Clone, Copy)]
pub struct Budget {
    pub bytes: u64,
    pub reserve_floor: u64,
}

/// A5.4 账本快照:违约/诊断时的完整内存叙事
#[derive(Debug, Clone, Copy)]
pub struct LedgerSnapshot {
    pub bytes_alive: u64,
    pub bytes_allocated_total: u64,
    pub stats: MemStats,
    pub phase: MemPhase,
}


// ---- 池:iface Pool 的实现,也是分配者 ----

/// 池账目记录(Governor 持有;CudaPool 是它的对外视图)
struct PoolLedger {
    name: String,
    kind: PoolKind,
    capacity: u64,
    used: u64,
    peak: u64,
}

/// 显存池:**分配者**。持有 driver 原语句柄,按 PoolKind 路由物理路径。
#[derive(Clone)]
pub struct CudaPool {
    id: PoolId,
    name: String,
    kind: PoolKind,
    capacity: u64,
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    gov: Arc<Governor>,
    dev: CudaDevice,
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
        self.kind_check(PoolKind::Scratch)?;
        Ok(PoolBuf::wrap(Box::new(self.malloc_inner(bytes)?)))
    }

    fn malloc_persistent(&self, bytes: u64) -> Result<PoolBuf, BackendError> {
        match self.kind {
            PoolKind::Weights | PoolKind::KvCache | PoolKind::Workspace => {}
            _ => {
                return Err(BackendError::LawViolation(
                    "持久分配语义与池类型不匹配(需 Weights/KvCache/Workspace)",
                ));
            }
        }
        Ok(PoolBuf::wrap(Box::new(self.malloc_inner(bytes)?)))
    }

    fn malloc_peer_shared(&self, bytes: u64) -> Result<PoolBuf, BackendError> {
        self.kind_check(PoolKind::PeerShared)?;
        Ok(PoolBuf::wrap(Box::new(self.malloc_inner(bytes)?)))
    }
}

impl CudaPool {
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
        if let Err(e) = self.gov.charge(effective) {
            self.uncharge_pool(effective);
            return Err(e);
        }
        let backing = match self.kind {
            PoolKind::PeerShared => match self.vmm_alloc_raw(effective as usize) {
                Ok((b, r)) => {
                    debug_assert_eq!(r as u64, effective);
                    b
                }
                Err(e) => {
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
        Ok(CudaPoolBuf {
            inner: Arc::new(PoolBufInner {
                backing: Mutex::new(backing),
                bytes: effective,
                pool: Arc::new(self.clone_account()),
                gov: Arc::clone(&self.gov),
            }),
            token,
        })
    }

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
    fn uncharge_global(&self, bytes: u64) {
        self.gov.uncharge(bytes);
    }

    /// 归还池账本
    fn uncharge_pool(&self, bytes: u64) {
        self.gov.uncharge_pool(self.id.0, bytes);
    }

    fn clone_account(&self) -> CudaPool {
        self.clone()
    }

    /// A2.8:VMM 物理分配(reserve→create→map→setAccess RW)。
    /// 返回未封装句柄;调用方负责包装与销账。
    fn vmm_alloc_raw(
        &self,
        bytes: usize,
    ) -> Result<(PoolBacking, usize), BackendError> {
        use sys::{
            cuMemAddressFree, cuMemAddressReserve, cuMemCreate, cuMemMap, cuMemSetAccess,
            cuMemUnmap, cuMemRelease, cuMemGetAllocationGranularity, CUmemAccessDesc,
            CUmemAllocationProp, CUmemLocation, CUresult::CUDA_SUCCESS,
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
            if cuMemAddressReserve(&mut ptr, rounded, gran, 0, 0) != CUDA_SUCCESS {
                return Err(BackendError::AllocFailed {
                    kind: "vmm-reserve",
                    len: rounded,
                    detail: "cuMemAddressReserve".into(),
                });
            }
            let mut chunk: sys::CUmemGenericAllocationHandle = Default::default();
            if cuMemCreate(&mut chunk, rounded, &props, 0) != CUDA_SUCCESS {
                let _ = cuMemAddressFree(ptr, rounded);
                return Err(BackendError::AllocFailed {
                    kind: "vmm-create",
                    len: rounded,
                    detail: "cuMemCreate".into(),
                });
            }
            if cuMemMap(ptr, rounded, 0, chunk, 0) != CUDA_SUCCESS {
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
            if cuMemSetAccess(ptr, rounded, access.as_ptr(), 1) != CUDA_SUCCESS {
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
                    ctx: Arc::clone(&self.ctx),
                },
                rounded,
            ))
        }
    }
}

/// 池缓冲的物理背面:stream-ordered 切片或 VMM 映射
enum PoolBacking {
    Empty,
    Slice(CudaSlice<u8>),
    Vmm {
        ptr: sys::CUdeviceptr,
        bytes: usize,
        chunk: sys::CUmemGenericAllocationHandle,
        ctx: Arc<CudaContext>,
    },
}

/// 池缓冲句柄:**Arc 共享所有权 = 租约系统**。
/// Clone(租约)把 Arc 计数 +1——图捕获期 lease 进 CapturedGraph 的
/// keepalive,用户侧句柄先 drop 也不会进入回收流程(强租约保证)。
/// 身份(token retire)只随**最后一个**句柄的 drop 消亡。
#[derive(Clone)]
pub struct CudaPoolBuf {
    inner: Arc<PoolBufInner>,
    token: BufToken,
}

pub struct PoolBufInner {
    backing: Mutex<PoolBacking>,
    bytes: u64,
    pool: Arc<CudaPool>,
    gov: Arc<Governor>,
}

impl Drop for PoolBufInner {
    fn drop(&mut self) {
        // 仅最后一个句柄消亡时到达此处(Arc 计数归零)
        let bytes = self.bytes;
        let pool = Arc::clone(&self.pool);
        let gov = Arc::clone(&self.gov);
        let backing = std::mem::replace(self.backing.get_mut(), PoolBacking::Empty);
        let idle = gov.phase() == MemPhase::Idle;
        let gov_for_closure = Arc::clone(&gov);
        if idle {
            drop(backing);
            gov.uncharge(bytes);
            pool.uncharge_pool(bytes);
        } else {
            gov.defer(Box::new(move || {
                drop(backing); // Slice 释放切片 / Vmm unmap+release
                gov_for_closure.uncharge(bytes);
                pool.uncharge_pool(bytes);
            }));
        }
    }
}

impl CudaPoolBuf {
    /// 最后一个句柄消亡 = 身份消亡 + 物理回收排队
    pub(crate) fn retire_if_last(&self) {
        if Arc::strong_count(&self.inner) == 1 {
            self.inner.gov.retire(&self.token);
        }
    }
}

impl Drop for CudaPoolBuf {
    fn drop(&mut self) {
        let is_last = Arc::strong_count(&self.inner) == 1;
        if is_last {
            self.inner.gov.retire(&self.token); // 身份随内存消亡
        }
        // PoolBufInner::drop 负责物理回收(Idle 立即 / 非 Idle 延迟)
    }
}

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

// ---- Device:一张卡,账本 + 池组 + 相位机的宿主 ----

/// CUDA 设备实例:一个 CudaDevice 绑定一张卡(UUID 钉),账本独立。
#[derive(Clone)]
pub struct CudaDevice {
    ordinal: usize,
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    gov: Arc<Governor>,
    desc: DeviceDesc,
}

impl CudaDevice {
    /// 按 UUID 钉卡(唯一合法入口;数字序禁用,09-19 事故判例)。
    pub fn new_by_uuid(uuid: &str) -> Result<Self, BackendError> {
        let ordinal = resolve_uuid_ordinal(uuid)?;
        let s = Self::new(ordinal)?;
        if s.desc.uuid != uuid {
            return Err(BackendError::Init(format!(
                "UUID 不匹配:请求 {uuid},ordinal {ordinal} 实际 {}",
                s.desc.uuid
            )));
        }
        Ok(s)
    }

    pub fn new(ordinal: usize) -> Result<Self, BackendError> {
        let ctx = CudaContext::new(ordinal)
            .map_err(|e| BackendError::Init(format!("cuda:{ordinal}: {e}")))?;
        let stream = ctx.default_stream();
        let uuid = cuda_uuid(ordinal)?;
        let (_free, total) = ctx
            .mem_get_info()
            .map_err(|e| BackendError::Init(format!("mem_get_info: {e}")))?;
        Ok(Self {
            ordinal,
            ctx: Arc::clone(&ctx),
            stream,
            gov: Arc::new(Governor {
                ctx: Some(ctx),
                ..Default::default()
            }),
            desc: DeviceDesc {
                uuid,
                arch: Arch::Sm86, // TODO(M1): compute capability 运行时读取
                total_bytes: total as u64,
            },
        })
    }

    pub fn ctx(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// 图状态机迁移(GraphGovernor 持真实状态机,这里镜像其 phase)。
    pub fn set_phase(&self, phase: MemPhase) {
        self.gov.set_phase(phase);
    }

    pub fn phase(&self) -> MemPhase {
        self.gov.phase()
    }

    pub fn stats(&self) -> MemStats {
        *self.gov.stats.lock()
    }

    /// **A5 硬预算**:声明预算后,每次分配即时校验(超支 fail-fast)。
    pub fn set_budget(&self, budget: Budget) {
        *self.gov.budget.lock() = Some(budget);
    }

    /// A5.3 第三道闸:driver 侧对账(free,total)。
    pub fn mem_get_info(&self) -> Result<(usize, usize), BackendError> {
        self.ctx
            .mem_get_info()
            .map_err(|e| BackendError::Init(format!("mem_get_info: {e}")))
    }

    /// **A2.8**:VMM 分配(2MiB 粒度;PeerShared 池语义的独立入口,
    /// 测试/诊断用;正式路径 = create_pool(PeerShared) + pool.malloc)。
    pub fn vmm_alloc(&self, bytes: usize) -> Result<VmmBuf, BackendError> {
        use sys::{
            cuMemAddressFree, cuMemAddressReserve, cuMemCreate, cuMemMap, cuMemSetAccess,
            cuMemUnmap, cuMemRelease, cuMemGetAllocationGranularity, CUmemAccessDesc,
            CUmemAllocationProp, CUmemLocation, CUresult::CUDA_SUCCESS,
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
            self.gov.charge(rounded as u64)?;
            let mut ptr: sys::CUdeviceptr = 0;
            if cuMemAddressReserve(&mut ptr, rounded, gran, 0, 0) != CUDA_SUCCESS {
                self.gov.uncharge(rounded as u64);
                return Err(BackendError::AllocFailed {
                    kind: "vmm-reserve",
                    len: rounded,
                    detail: "cuMemAddressReserve".into(),
                });
            }
            let mut chunk: sys::CUmemGenericAllocationHandle = Default::default();
            if cuMemCreate(&mut chunk, rounded, &props, 0) != CUDA_SUCCESS {
                let _ = cuMemAddressFree(ptr, rounded);
                self.gov.uncharge(rounded as u64);
                return Err(BackendError::AllocFailed {
                    kind: "vmm-create",
                    len: rounded,
                    detail: "cuMemCreate".into(),
                });
            }
            if cuMemMap(ptr, rounded, 0, chunk, 0) != CUDA_SUCCESS {
                let _ = cuMemRelease(chunk);
                let _ = cuMemAddressFree(ptr, rounded);
                self.gov.uncharge(rounded as u64);
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
            if cuMemSetAccess(ptr, rounded, access.as_ptr(), 1) != CUDA_SUCCESS {
                let _ = cuMemUnmap(ptr, rounded);
                let _ = cuMemRelease(chunk);
                let _ = cuMemAddressFree(ptr, rounded);
                self.gov.uncharge(rounded as u64);
                return Err(BackendError::AllocFailed {
                    kind: "vmm-set-access",
                    len: rounded,
                    detail: "cuMemSetAccess".into(),
                });
            }
            Ok(VmmBuf {
                ptr,
                bytes: rounded,
                chunk,
                ctx: Arc::clone(&self.ctx),
                gov: Arc::clone(&self.gov),
            })
        }
    }

    /// 哨兵①:缓冲令牌存活校验(replay 前世代检查;未来 M1 用)
    pub fn validate_token(&self, t: &BufToken) -> bool {
        self.gov.validate(t)
    }

    /// 账本快照(A5.4 违约时打印)
    pub fn ledger(&self) -> LedgerSnapshot {
        LedgerSnapshot {
            bytes_alive: *self.gov.bytes_alive.lock(),
            bytes_allocated_total: *self.gov.bytes_allocated_total.lock(),
            stats: self.stats(),
            phase: self.phase(),
        }
    }
}

// ---- UUID 钉卡:数字序禁用的唯一替代 ----

fn cuda_uuid(ordinal: usize) -> Result<String, BackendError> {
    use cudarc::driver::result;
    let count = result::device::get_count().map_err(|e| BackendError::Init(format!("{e:?}")))?;
    if ordinal >= count as usize {
        return Err(BackendError::Init(format!(
            "ordinal {ordinal} 越界({count} 卡)"
        )));
    }
    let dev = result::device::get(ordinal as i32)
        .map_err(|e| BackendError::Init(format!("cuDeviceGet: {e:?}")))?;
    let uuid = result::device::get_uuid(dev)
        .map_err(|e| BackendError::Init(format!("cuDeviceGetUuid: {e:?}")))?;
    let hex: String = uuid
        .bytes
        .iter()
        .map(|&b| format!("{:02x}", b as u8))
        .collect();
    Ok(format!(
        "GPU-{}-{}-{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    ))
}

fn resolve_uuid_ordinal(uuid: &str) -> Result<usize, BackendError> {
    use cudarc::driver::result;
    let count = result::device::get_count().map_err(|e| BackendError::Init(format!("{e:?}")))?;
    for ord in 0..count as usize {
        if cuda_uuid(ord)? == uuid {
            return Ok(ord);
        }
    }
    Err(BackendError::Init(format!("UUID 未找到: {uuid}")))
}

// ---- P2P 窄口(A2.6 例外通道)----

impl CudaDevice {
    /// P2P 授权(经 sys::culib 直接驱动 API;cudarc 0.19 无 safe 封装)。
    pub fn enable_peer_access(&self, peer: &CudaDevice) -> Result<(), BackendError> {
        unsafe {
            let can: unsafe extern "C" fn(
                *mut i32,
                sys::CUdevice,
                sys::CUdevice,
            ) -> sys::CUresult = *sys::culib()
                .get(b"cuDeviceCanAccessPeer\0")
                .map_err(|e| BackendError::Init(format!("symbol: {e}")))?;
            let mut ok: i32 = 0;
            can(&mut ok, self.ctx.cu_device(), peer.ctx.cu_device())
                .result()
                .map_err(|e| BackendError::Init(format!("{e:?}")))?;
            if ok != 1 {
                return Err(BackendError::LawViolation(
                    "P2P 不可达(cuDeviceCanAccessPeer=0):检查驱动 P2P 支持与黑名单",
                ));
            }
            self.ctx
                .bind_to_thread()
                .map_err(|e| BackendError::Init(format!("{e:?}")))?;
            let en: unsafe extern "C" fn(sys::CUcontext, u32) -> sys::CUresult =
                *sys::culib()
                    .get(b"cuCtxEnablePeerAccess\0")
                    .map_err(|e| BackendError::Init(format!("symbol: {e}")))?;
            en(peer.ctx.cu_ctx(), 0)
                .result()
                .map_err(|e| BackendError::Init(format!("{e:?}")))?;
        }
        Ok(())
    }
}

// ---- Device / Backend 契约实现 ----

impl Backend for CudaBackend {
    type Device = CudaDevice;

    fn family(&self) -> BackendFamily {
        BackendFamily::Cuda
    }

    fn enumerate(&self) -> Result<Vec<DeviceDesc>, BackendError> {
        use cudarc::driver::result;
        result::init().map_err(|e| BackendError::Init(format!("cuInit: {e:?}")))?;
        let count = result::device::get_count().map_err(|e| BackendError::Init(format!("{e:?}")))?;
        (0..count as usize)
            .map(|ord| {
                let dev = result::device::get(ord as i32)
                    .map_err(|e| BackendError::Init(format!("{e:?}")))?;
                let uuid = result::device::get_uuid(dev)
                    .map_err(|e| BackendError::Init(format!("{e:?}")))?;
                let hex: String = uuid
                    .bytes
                    .iter()
                    .map(|&b| format!("{:02x}", b as u8))
                    .collect();
                let total = unsafe { result::device::total_mem(dev) }
                    .map_err(|e| BackendError::Init(format!("{e:?}")))?;
                Ok(DeviceDesc {
                    uuid: format!(
                        "GPU-{}-{}-{}-{}-{}",
                        &hex[0..8],
                        &hex[8..12],
                        &hex[12..16],
                        &hex[16..20],
                        &hex[20..32]
                    ),
                    arch: Arch::Sm86, // TODO(M1): 运行时读取
                    total_bytes: total as u64,
                })
            })
            .collect()
    }

    fn open(&self, uuid: &str) -> Result<Self::Device, BackendError> {
        CudaDevice::new_by_uuid(uuid)
    }
}

pub struct CudaBackend;

/// 持久域缓冲:P 阶段经池分配;drop 归账由 PoolBuf/CudaPoolBuf 负责。
pub struct Persistent<T: MemValue> {
    buf: Option<PoolBuf>,
    len: usize,
    token: Option<BufToken>,
    _marker: std::marker::PhantomData<fn() -> T>,
}

impl<T: MemValue> Persistent<T> {
    fn new(buf: PoolBuf, len: usize) -> Self {
        let token = buf.token();
        Self {
            buf: Some(buf),
            len,
            token: Some(token),
            _marker: std::marker::PhantomData,
        }
    }

    fn token(&self) -> Option<BufToken> {
        self.token
    }
}

impl<T: MemValue> DevBuf<T> for Persistent<T> {
    fn len(&self) -> usize {
        self.len
    }
    fn device_ptr(&self) -> *mut T {
        self.buf
            .as_ref()
            .expect("Persistent 未被 drop")
            .device_ptr() as *mut T
    }
}

/// 暂存域缓冲(允许 Capturing 相创建;归账同 Persistent)。
pub struct Scratch<T: MemValue> {
    buf: Option<PoolBuf>,
    len: usize,
    token: Option<BufToken>,
    _marker: std::marker::PhantomData<fn() -> T>,
}

impl<T: MemValue> Scratch<T> {
    fn token(&self) -> Option<BufToken> {
        self.token
    }
}

impl<T: MemValue> DevBuf<T> for Scratch<T> {
    fn len(&self) -> usize {
        self.len
    }
    fn device_ptr(&self) -> *mut T {
        self.buf
            .as_ref()
            .expect("Scratch 未被 drop")
            .device_ptr() as *mut T
    }
}

impl Device for CudaDevice {
    type Persistent<T: MemValue> = Persistent<T>;
    type Scratch<T: MemValue> = Scratch<T>;
    type Remote<T: MemValue> = RemoteBuf<T>;
    type Pool = CudaPool;

    fn desc(&self) -> &DeviceDesc {
        &self.desc
    }

    fn arch(&self) -> Arch {
        self.desc().arch
    }

    fn persistent_token<T: MemValue>(&self, p: &Self::Persistent<T>) -> Option<BufToken> {
        p.token()
    }

    fn scratch_token<T: MemValue>(&self, s: &Self::Scratch<T>) -> Option<BufToken> {
        s.token()
    }

    fn phase(&self) -> MemPhase {
        Governor::phase(&self.gov)
    }

    fn set_phase(&self, phase: MemPhase) {
        CudaDevice::set_phase(self, phase)
    }

    fn stats(&self) -> MemStats {
        CudaDevice::stats(self)
    }

    fn enable_peer_access(&self, peer: &Self) -> Result<(), BackendError> {
        CudaDevice::enable_peer_access(self, peer)
    }

    fn map_remote<T: MemValue>(
        &self,
        peer: &DeviceDesc,
        ptr: *mut T,
        len: usize,
    ) -> Result<Self::Remote<T>, BackendError> {
        let mut st = self.gov.stats.lock();
        st.peer_mappings += 1;
        st.peer_mapped_bytes += (len * std::mem::size_of::<T>()) as u64;
        Ok(RemoteBuf {
            ptr,
            len,
            peer_uuid: peer.uuid.clone(),
        })
    }

    fn create_pool(&self, cfg: PoolConfig) -> Result<Self::Pool, BackendError> {
        let mut pools = self.gov.pools.lock();
        if pools.values().any(|p| p.name == cfg.name) {
            return Err(BackendError::Init(format!(
                "A5.1: 重复池名 '{}'——分解表不应有两行同名账",
                cfg.name
            )));
        }
        let id = self
            .gov
            .next_pool_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        pools.insert(
            id,
            PoolLedger {
                name: cfg.name.clone(),
                kind: cfg.kind,
                capacity: cfg.bytes,
                used: 0,
                peak: 0,
            },
        );
        Ok(CudaPool {
            id: PoolId(id),
            name: cfg.name.clone(),
            kind: cfg.kind,
            capacity: cfg.bytes,
            ctx: Arc::clone(&self.ctx),
            stream: Arc::clone(&self.stream),
            gov: Arc::clone(&self.gov),
            dev: self.clone(),
        })
    }

    fn pool(&self, id: PoolId) -> Result<Self::Pool, BackendError> {
        let pools = self.gov.pools.lock();
        let led = pools.get(&id.0).ok_or(BackendError::UnknownPool(id.0))?;
        Ok(CudaPool {
            id: PoolId(id.0),
            name: led.name.clone(),
            kind: led.kind,
            capacity: led.capacity,
            ctx: Arc::clone(&self.ctx),
            stream: Arc::clone(&self.stream),
            gov: Arc::clone(&self.gov),
            dev: self.clone(),
        })
    }

    fn alloc_persistent_in<T: MemValue>(
        &self,
        pool: &Self::Pool,
        len: usize,
    ) -> Result<Self::Persistent<T>, BackendError> {
        let bytes = (len * std::mem::size_of::<T>()) as u64;
        let buf = pool.malloc_persistent(bytes)?;
        self.gov.stats.lock().persistent_allocs += 1;
        Ok(Persistent::new(buf, len))
    }

    fn htod_persistent_in<T: MemValue>(
        &self,
        pool: &Self::Pool,
        src: Vec<T>,
    ) -> Result<Self::Persistent<T>, BackendError> {
        let len = src.len();
        let bytes = (len * std::mem::size_of::<T>()) as u64;
        let buf = pool.malloc_persistent(bytes)?;
        // MemValue = Send + 'static 的 plain-old-data 按字节搬运;
        // VMM/Slice 两种背面都是连续设备内存,统一走裸指针拷贝
        let host_bytes =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u8, bytes as usize) };
        self.ctx
            .bind_to_thread()
            .map_err(|e| BackendError::Init(format!("{e:?}")))?;
        unsafe {
            use sys::{cuMemcpyHtoD_v2, CUresult::CUDA_SUCCESS};
            if cuMemcpyHtoD_v2(
                buf.device_ptr() as sys::CUdeviceptr,
                host_bytes.as_ptr() as *const core::ffi::c_void,
                bytes as usize,
            ) != CUDA_SUCCESS
            {
                // buf 归账由其 Drop 完成(路径治理不变)
                return Err(BackendError::CopyFailed {
                    dir: "htod",
                    detail: "cuMemcpyHtoD_v2".into(),
                });
            }
        }
        self.gov.stats.lock().persistent_allocs += 1;
        Ok(Persistent::new(buf, len))
    }

    fn alloc_scratch_in<T: MemValue>(
        &self,
        pool: &Self::Pool,
        len: usize,
    ) -> Result<Self::Scratch<T>, BackendError> {
        let bytes = (len * std::mem::size_of::<T>()) as u64;
        let buf = pool.malloc_scratch(bytes)?;
        self.gov.stats.lock().scratch_allocs += 1;
        let token = buf.token();
        Ok(Scratch {
            buf: Some(buf),
            len,
            token: Some(token),
            _marker: std::marker::PhantomData,
        })
    }
}

// ---- A2.8 VMM 独立缓冲 ----

/// A2.8 VMM 缓冲(独立入口;PeerShared 池内路径用 PoolBacking::Vmm)
pub struct VmmBuf {
    ptr: sys::CUdeviceptr,
    bytes: usize,
    chunk: sys::CUmemGenericAllocationHandle,
    ctx: Arc<CudaContext>,
    gov: Arc<Governor>,
}

impl VmmBuf {
    pub fn device_ptr(&self) -> sys::CUdeviceptr {
        self.ptr
    }
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

impl Drop for VmmBuf {
    fn drop(&mut self) {
        unsafe {
            use sys::{cuMemAddressFree, cuMemRelease, cuMemUnmap};
            let _ = cuMemUnmap(self.ptr, self.bytes);
            let _ = cuMemRelease(self.chunk);
            let _ = cuMemAddressFree(self.ptr, self.bytes);
        }
        self.gov.uncharge(self.bytes as u64);
    }
}

// ---- P2P 远端映射视图 ----

pub struct RemoteBuf<T: MemValue> {
    ptr: *mut T,
    len: usize,
    peer_uuid: String,
}

unsafe impl<T: Send + 'static> Send for RemoteBuf<T> {}

impl<T: MemValue> DevBuf<T> for RemoteBuf<T> {
    fn len(&self) -> usize {
        self.len
    }
    fn device_ptr(&self) -> *mut T {
        self.ptr
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use owl_iface::Pool as _;

    fn make() -> CudaDevice {
        CudaDevice::new(0).expect("需要 CUDA 设备")
    }

    fn scratch_pool(b: &CudaDevice, name: &str, bytes: u64) -> CudaPool {
        b.create_pool(PoolConfig {
            name: name.into(),
            kind: PoolKind::Scratch,
            bytes,
        })
        .unwrap()
    }

    fn persistent_pool(b: &CudaDevice, name: &str, bytes: u64) -> CudaPool {
        b.create_pool(PoolConfig {
            name: name.into(),
            kind: PoolKind::Weights,
            bytes,
        })
        .unwrap()
    }

    /// 契约:池化分配 + A1.2 延迟归还 + 净空窗口清账
    #[test]
    fn pool_malloc_contract_and_deferred_free() {
        let dev = make();
        let pool = scratch_pool(&dev, "scratch", 64 << 10);

        let scratch = dev.alloc_scratch_in::<f32>(&pool, 4096).unwrap();
        assert_eq!(DevBuf::<f32>::len(&scratch), 4096);
        assert_eq!(pool.usage().used, 16 * 1024);

        // 池耗尽 → A5.4 PoolExhausted
        let over = dev.alloc_scratch_in::<f32>(&pool, 1024 * 1024);
        assert!(matches!(over, Err(BackendError::PoolExhausted { .. })));

        // Live 相 drop → 延迟(池账 + 全局账都不动)
        dev.set_phase(MemPhase::Live);
        let victim = dev.alloc_scratch_in::<f32>(&pool, 16).unwrap();
        drop(victim);
        assert_eq!(dev.stats().deferred_frees, 1);
        assert_eq!(dev.stats().drained_frees, 0);

        // Idle(净空窗口)→ 延迟队列统一归还
        dev.set_phase(MemPhase::Idle);
        assert_eq!(dev.stats().drained_frees, 1);
        assert_eq!(pool.usage().used, 16 * 1024);
    }

    /// A5:预算合同——超支 fail-fast,账本可归因
    #[test]
    fn budget_violation_fails_fast() {
        let dev = make();
        dev.set_budget(Budget {
            bytes: 1024 * 1024,
            reserve_floor: 512 * 1024,
        });
        let pool = persistent_pool(&dev, "b", 8 << 20);

        let ok = dev.alloc_persistent_in::<u8>(&pool, 1024).unwrap();
        drop(ok);
        dev.set_phase(MemPhase::Idle);

        let huge = dev.alloc_persistent_in::<u8>(&pool, 8 * 1024 * 1024);
        assert!(matches!(
            huge,
            Err(BackendError::LawViolation(msg)) if msg.contains("A5.4")
        ));
        assert!(dev.ledger().bytes_alive < 1024 * 1024);
        let (free, _total) = dev.mem_get_info().unwrap();
        assert!(free > 0);
    }

    /// A2.8:VMM 分配(2MiB 粒度,可被对端 P2P 映射的唯一合法路径)
    #[test]
    fn vmm_alloc_granularity_and_ledger() {
        let dev = make();
        let buf = dev.vmm_alloc(1024).expect("vmm_alloc");
        assert!(buf.bytes() >= 2 * 1024 * 1024);
        assert!(buf.bytes() % (2 * 1024 * 1024) == 0);
        assert!(buf.device_ptr() != 0);
        assert_eq!(dev.ledger().bytes_alive, buf.bytes() as u64);
        drop(buf);
        assert_eq!(dev.ledger().bytes_alive, 0);
    }

    /// 语义分立:池类型与分配性质错配 = LawViolation
    #[test]
    fn pool_kind_mismatch_is_law_violation() {
        let dev = make();
        let scratch = scratch_pool(&dev, "s", 1 << 20);
        let weights = persistent_pool(&dev, "w", 1 << 20);
        // Scratch 池上做持久分配 → 拒
        assert!(matches!(
            dev.alloc_persistent_in::<f32>(&scratch, 16),
            Err(BackendError::LawViolation(msg)) if msg.contains("不匹配")
        ));
        // Weights 池上做跨卡共享分配 → 拒
        assert!(matches!(
            weights.malloc_peer_shared(16),
            Err(BackendError::LawViolation(msg)) if msg.contains("不匹配")
        ));
        // 正确配对可用
        assert!(weights.malloc_persistent(256).is_ok());
        let peer = dev
            .create_pool(PoolConfig {
                name: "p".into(),
                kind: PoolKind::PeerShared,
                bytes: 8 << 20,
            })
            .unwrap();
        assert!(peer.malloc_peer_shared(4096).is_ok());
    }

    /// A2.8:PeerShared 池 malloc_peer_shared 真机验证
    #[test]
    fn peer_shared_malloc_uses_vmm() {
        let dev = make();
        let peer = dev
            .create_pool(PoolConfig {
                name: "ps".into(),
                kind: PoolKind::PeerShared,
                bytes: 8 << 20,
            })
            .unwrap();
        let buf = peer.malloc_peer_shared(1024).unwrap();
        // 粒度 ≥2MiB 对齐(经 DevBuf<u8> len 查询)
        assert!(buf.len() >= 2 * 1024 * 1024);
        assert!(buf.len() % (2 * 1024 * 1024) == 0);
        assert!(!buf.device_ptr().is_null());
        assert_eq!(dev.ledger().bytes_alive, buf.len() as u64);
        drop(buf);
        assert_eq!(dev.ledger().bytes_alive, 0);
    }

    /// 设备隔离律:UUID 钉卡回环 + Backend 枚举
    #[test]
    fn uuid_pinning_roundtrip() {
        let backend = CudaBackend;
        let devs = backend.enumerate().expect("需要 CUDA 设备");
        assert!(!devs.is_empty());
        let mut dev = backend.open(&devs[0].uuid).expect("open by uuid");
        assert_eq!(dev.desc().uuid, devs[0].uuid);
        assert!(dev.desc().total_bytes > 0);

        // 池化契约走 Device trait 静态分发
        let pool = dev
            .create_pool(PoolConfig {
                name: "t".into(),
                kind: PoolKind::Weights,
                bytes: 1 << 20,
            })
            .unwrap();
        let buf = dev.alloc_persistent_in::<f32>(&pool, 64).unwrap();
        assert_eq!(buf.len(), 64);
        drop(buf);
        dev.set_phase(MemPhase::Idle);
    }
}

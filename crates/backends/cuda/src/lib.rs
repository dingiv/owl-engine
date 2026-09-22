//! owl-cuda —— cudarc **官方版**(crates.io,非 guoqingbao fork)的治理
//! 封装层,实现 `owl_iface::GpuBackend`(charter A1/A4 落点)。
//!
//! 选型注记(2026-09-22):fork(cudarc-gb @ 2e81793)的图安全增量
//! (capture_status/捕获期池分配/Drop 禁 free)在官方 0.19 已被原生
//! API 覆盖(CudaGraph/begin_capture/stream-ordered 分配),且 fork 的
//! 13.x sys 绑定官方也已支持——按 A4"语义分歧不进第三方库",选官方。
//!
//! 治理语义(实现 GpuBackend 的方式):
//! 1. **分配分域**:`Persistent`/`Scratch` 两种生命周期,类型系统强制;
//! 2. **A1.2 执行**:非 Idle 相的 drop 一律延迟,回 Idle(净空窗口)归还;
//! 3. **字节级原语**:iface 的 MemValue 是抽象标记,后端不得要求它携带
//!    driver 私有 bound(如 DeviceRepr)——所以分配按 `len * size_of::<T>`
//!    字节进行,类型解释由 `DevBuf<T>` 的视图承担。这保证 iface 契约
//!    不泄漏任何 cudarc 类型到上层。
//!
//! 捕获期语义:分配走 stream-ordered allocator(可捕获);非 Idle 相
//! drop 延迟;专属捕获池路由在 M1 由 graph 治理层实现。

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr};
use cudarc::driver::sys;
use owl_iface::{Arch, Backend, BackendError, BackendFamily, Device, DeviceDesc, DevBuf, MemValue, Pool, PoolConfig, PoolId, PoolKind, PoolUsage};
use parking_lot::{Mutex, RwLock};
use std::sync::Arc;

pub use owl_iface::{MemPhase, MemStats};

type DeferredFree = Box<dyn FnOnce(&mut u64) + Send>;

/// A5.4 账本快照:违约/诊断时的完整内存叙事
#[derive(Debug, Clone, Copy)]
pub struct LedgerSnapshot {
    pub bytes_alive: u64,
    pub bytes_allocated_total: u64,
    pub stats: MemStats,
    pub phase: MemPhase,
}

/// 共享治理句柄(phase + 延迟释放队列 + 审计)。
#[derive(Default)]
struct Governor {
    ctx: Option<Arc<CudaContext>>,
    phase: RwLock<MemPhase>,
    deferred: Mutex<Vec<DeferredFree>>,
    stats: Mutex<MemStats>,
    /// A5.2 账本:当前存活字节(persistent 存量 + 未归还延迟释放)
    bytes_alive: Mutex<u64>,
    /// A5.2 账本:生命周期累计分配字节(单调增,用于漂移归因)
    bytes_allocated_total: Mutex<u64>,
    /// A5.1 预算(None = 未设;设置后每次分配断言不超)
    budget: Mutex<Option<Budget>>,
    /// 显存池账本(A5.2:一切分配归属具名池);Arc 供池对象跨线程归账
    pools: Arc<Mutex<std::collections::HashMap<u64, PoolLedger>>>,
    next_pool_id: std::sync::atomic::AtomicU64,
}

/// 池账本:容量承诺 + 存量 + 峰值
/// 池账目记录(Governor 持有;CudaPool 是它的对外视图)
struct PoolLedger {
    name: String,
    kind: PoolKind,
    capacity: u64,
    used: u64,
    peak: u64,
}

/// 池对象:iface Pool 的 cuda 实现。缓冲持有它的 Arc,drop 时归账。
pub struct CudaPool {
    id: PoolId,
    name: String,
    kind: PoolKind,
    capacity: u64,
    gov: Arc<Governor>,
}

impl Pool for CudaPool {
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
}

impl CudaPool {
    fn uncharge(&self, bytes: u64) {
        if let Some(led) = self.gov.pools.lock().get_mut(&self.id.0) {
            led.used = led.used.saturating_sub(bytes);
        }
    }
}

/// A5 硬预算:启动时声明,生命周期恒不超(见 charter 公理 A5)。
#[derive(Debug, Clone, Copy)]
pub struct Budget {
    /// 预算上限(bytes/卡)。含运行时底价与安全余量,由内存规划器分解而来。
    pub bytes: u64,
    /// 对账时的最低 free 余量警线(低于即告警,高于即违约)
    pub reserve_floor: u64,
}

impl Governor {
    /// 池余量校验(A5.4 第一道):池耗尽即违约。返回后调用方已占池账。
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

    fn charge(&self, bytes: u64) -> Result<(), BackendError> {
        let mut alive = self.bytes_alive.lock();
        *alive += bytes;
        *self.bytes_allocated_total.lock() += bytes;
        if let Some(b) = *self.budget.lock() {
            // A5 判据:引擎自身账本(含延迟滞留)≤ 预算。整卡 used 含
            // 其他进程/上下文,不能作为本引擎预算的判据
            if *alive > b.bytes {
                *alive -= bytes; // 违约:预记回滚,账本如实反映"未发生"
                return Err(BackendError::LawViolation(
                    "A5.4 显存超支(引擎账本超预算,详见账本快照日志)",
                ));
            }
            // A5.3 第三道闸:driver 侧对账,free 低于警线即告警
            // (可能是本引擎超支,也可能是同卡邻居;归因走 ledger())
            let (free, _total) = self.mem_get_info()?;
            if (free as u64) < b.reserve_floor {
                eprintln!(
                    "[owl-mem] WARN: free {free}B < reserve_floor {}B(A5 警线,账本 alive={})",
                    b.reserve_floor,
                    *alive
                );
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

    fn mem_get_info(&self) -> Result<(usize, usize), BackendError> {
        self.ctx
            .as_ref()
            .expect("Governor.ctx")
            .mem_get_info()
            .map_err(|e| BackendError::Init(format!("mem_get_info: {e}")))
    }
}

impl Governor {
    fn phase(&self) -> MemPhase {
        *self.phase.read()
    }

    fn set_phase(&self, phase: MemPhase) {
        *self.phase.write() = phase;
        if phase == MemPhase::Idle {
            // 净空窗口:统一归还延迟队列(A1.2)
            let mut q = self.deferred.lock();
            let n = q.len();
            let mut bytes = 0u64;
            for free in q.drain(..) {
                free(&mut bytes);
            }
            *self.bytes_alive.lock() -= bytes;
            self.stats.lock().drained_frees += n as u64;
        }
    }

    fn defer(&self, free: DeferredFree) {
        self.stats.lock().deferred_frees += 1;
        self.deferred.lock().push(free);
    }

    /// 非 Idle 相 → 延迟;Idle → 立即。所有缓冲域 drop 的公共出口。
    fn release_slice<T: Send + 'static>(&self, slice: CudaSlice<T>, bytes: u64) {
        if self.phase() == MemPhase::Idle {
            drop(slice);
            self.uncharge(bytes);
        } else {
            self.defer(Box::new(move |out| {
                drop(slice);
                *out += bytes;
            }));
        }
    }
}

// ---- A2.8:VMM 分配器(cuMemCreate/MAP/SetAccess)----
// 魔改驱动 BAR1=256MiB 约束下,跨卡共享缓冲的唯一合法分配路径。
// legacy cudaMalloc/池分配的物理段无法通过 BAR1 窗口(857MB 段判例)。

/// VMM 缓冲:物理 chunk(2MiB 粒度)+ 本地虚拟映射,可导出/可被对端
/// cuMemMap。Drop = unmap + release,账本同步销账。
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
            use sys::{cuMemAddressFree, cuMemMap, cuMemRelease, cuMemUnmap};
            let _ = cuMemUnmap(self.ptr, self.bytes);
            let _ = cuMemRelease(self.chunk);
            let _ = cuMemAddressFree(self.ptr, self.bytes);
        }
        self.gov.uncharge(self.bytes as u64);
    }
}

/// CUDA 设备上下文:owl 的 cuda 后端入口。
/// 设备隔离律:一个 OwlCuda 实例绑定一张卡(UUID 钉),账本独立。
pub struct OwlCuda {
    ctx: Arc<CudaContext>,
    ordinal: usize,
    /// 分配/拷贝走 stream-ordered 语义(捕获安全)
    stream: Arc<CudaStream>,
    gov: Arc<Governor>,
    desc: DeviceDesc,
}

impl OwlCuda {
    /// 按 UUID 钉卡(唯一合法入口;数字序禁用,09-19 事故判例)。
    /// UUID 形如 "GPU-e565c505-4921-9979-2e0f-2b83490b7aed"。
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
        let (free, total) = ctx
            .mem_get_info()
            .map_err(|e| BackendError::Init(format!("mem_get_info: {e}")))?;
        let _ = free;
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

    /// **A5 硬预算**:声明预算后,每次分配即时校验(超支 fail-fast),
    /// 且 reserve_floor 触发告警。规划器(core)负责在启动前完成分解封闭性
    /// 静态证明;这里是运行时第二道闸。
    pub fn set_budget(&self, budget: Budget) {
        *self.gov.budget.lock() = Some(budget);
    }

    /// A5.3 第三道闸:driver 侧对账(free,total)。返回 (free, total)。
    pub fn mem_get_info(&self) -> Result<(usize, usize), BackendError> {
        self.ctx
            .mem_get_info()
            .map_err(|e| BackendError::Init(format!("mem_get_info: {e}")))
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

    fn alloc_zeroed_bytes(&self, bytes: usize) -> Result<CudaSlice<u8>, BackendError> {
        // A5.2:先过账本校验(含预算断言),再分配
        self.gov.charge(bytes as u64)?;
        self.stream
            .alloc_zeros::<u8>(bytes)
            .map_err(|e| {
                self.gov.uncharge(bytes as u64); // 分配失败,账本回滚
                BackendError::AllocFailed {
                    kind: "zeroed",
                    len: bytes,
                    detail: e.to_string(),
                }
            })
    }

    /// **A2.8**:VMM 分配(PeerShared 缓冲的唯一合法路径)。
    /// 物理粒度取设备最小值(sm86 实测 2MiB),容量向上取整;
    /// 本地 RW 映射;经账本 charge(A5.2);可被对端 P2P 映射/导出。
    pub fn vmm_alloc(&self, bytes: usize) -> Result<VmmBuf, BackendError> {
        use sys::{
            cuMemAddressFree, cuMemAddressReserve, cuMemCreate, cuMemUnmap,
            cuMemGetAllocationGranularity, cuMemMap, cuMemRelease, cuMemSetAccess,
            CUmemAccessDesc, CUmemAllocationProp, CUmemLocation, CUresult::CUDA_SUCCESS,
        };
        unsafe {
            self.ctx
                .bind_to_thread()
                .map_err(|e| BackendError::Init(format!("{e:?}")))?;

            let mut gran: usize = 0;
            let mut location = CUmemLocation {
                type_: sys::CUmemLocationType_enum::CU_MEM_LOCATION_TYPE_DEVICE,
                __bindgen_anon_1: sys::CUmemLocation_st__bindgen_ty_1 {
                    id: self.ctx.cu_device() as i32,
                },
            };
            let mut props = CUmemAllocationProp {
                type_: sys::CUmemAllocationType_enum::CU_MEM_ALLOCATION_TYPE_PINNED,
                requestedHandleTypes:
                    sys::CUmemAllocationHandleType_enum::CU_MEM_HANDLE_TYPE_NONE,
                location,
                win32HandleMetaData: std::ptr::null_mut(),
                allocFlags: sys::CUmemAllocationProp_st__bindgen_ty_1 {
                    compressionType: 0,
                    gpuDirectRDMACapable: 0,
                    usage: 0,
                    reserved: [0; 4],
                },
            };
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
}

/// 持久域缓冲:图存活期 drop = 延迟,不丢数据也不撕图。
/// 内部持字节缓冲,按 `T` 视图暴露(DevBuf)。
pub struct Persistent<T: MemValue> {
    inner: Option<CudaSlice<u8>>,
    len: usize,
    stream: Arc<CudaStream>,
    gov: Arc<Governor>,
    /// 归属池(None = 历史无池分配;M1 起强制 Some)
    pool: Option<PoolId>,
    bytes: u64,
    _marker: std::marker::PhantomData<fn() -> T>,
}

impl<T: MemValue> Persistent<T> {
    fn new(
        slice: CudaSlice<u8>,
        len: usize,
        bytes: u64,
        pool: Option<PoolId>,
        ctx: &OwlCuda,
    ) -> Self {
        Self {
            inner: Some(slice),
            len,
            bytes,
            stream: Arc::clone(&ctx.stream),
            pool,
            gov: Arc::clone(&ctx.gov),
            _marker: std::marker::PhantomData,
        }
    }

    /// 主动提前归还(仅 Idle 合法)。拒绝时所有权原样带回 Err,数据不丢。
    pub fn release(mut self) -> Result<(), (Self, BackendError)> {
        if self.gov.phase() != MemPhase::Idle {
            return Err((
                self,
                BackendError::LawViolation("A1.2: 图存活期禁止提前释放持久缓冲"),
            ));
        }
        drop(self.inner.take());
        Ok(())
    }
}

impl<T: MemValue> Drop for Persistent<T> {
    fn drop(&mut self) {
        if let Some(slice) = self.inner.take() {
            if let Some(pid) = &self.pool {
                self.gov.uncharge_pool(pid.0, self.bytes);
            }
            self.gov.release_slice(slice, self.bytes);
        }
    }
}

impl<T: MemValue> DevBuf<T> for Persistent<T> {
    fn len(&self) -> usize {
        self.len
    }
    fn device_ptr(&self) -> *mut T {
        let (ptr, _sync) = self
            .inner
            .as_ref()
            .expect("Persistent 未被 drop")
            .device_ptr(&self.stream);
        ptr as *mut T
    }
}

/// 暂存域缓冲:允许 Capturing 相创建(stream-ordered 分配可捕获);
/// 非 Idle 相 drop 延迟(与 Persistent 同律)。
pub struct Scratch<T: MemValue> {
    inner: Option<CudaSlice<u8>>,
    len: usize,
    stream: Arc<CudaStream>,
    gov: Arc<Governor>,
    pool: Option<PoolId>,
    bytes: u64,
    _marker: std::marker::PhantomData<fn() -> T>,
}

impl<T: MemValue> Drop for Scratch<T> {
    fn drop(&mut self) {
        if let Some(slice) = self.inner.take() {
            if let Some(pid) = &self.pool {
                self.gov.uncharge_pool(pid.0, self.bytes);
            }
            self.gov.release_slice(slice, self.bytes);
        }
    }
}

impl<T: MemValue> DevBuf<T> for Scratch<T> {
    fn len(&self) -> usize {
        self.len
    }
    fn device_ptr(&self) -> *mut T {
        let (ptr, _sync) = self
            .inner
            .as_ref()
            .expect("Scratch 未被 drop")
            .device_ptr(&self.stream);
        ptr as *mut T
    }
}

// ---- iface 接线:owl-cuda 是 GpuBackend 的第一个实现 ----

/// 跨卡远端映射视图:指针在本地可读写(同一统一地址空间),
/// 生命周期内占用本卡 `peer_mapped_bytes` 账本。
pub struct RemoteBuf<T: MemValue> {
    ptr: *mut T,
    len: usize,
    peer_uuid: String,
    gov: Arc<Governor>,
}

unsafe impl<T: Send + 'static> Send for RemoteBuf<T> {}

impl<T: MemValue> Drop for RemoteBuf<T> {
    fn drop(&mut self) {
        let mut st = self.gov.stats.lock();
        st.peer_mapped_bytes -= (self.len * std::mem::size_of::<T>()) as u64;
        st.peer_mappings -= 1;
    }
}

impl<T: MemValue> DevBuf<T> for RemoteBuf<T> {
    fn len(&self) -> usize {
        self.len
    }
    fn device_ptr(&self) -> *mut T {
        self.ptr
    }
}

impl OwlCuda {
    /// P2P 授权 + 对端映射的全套窄口操作(经 sys::culib 直接驱动 API;
    /// cudarc 0.19 尚无 safe 封装)。
    pub fn enable_peer_access(&self, peer: &OwlCuda) -> Result<(), BackendError> {
        unsafe {
            let can: unsafe extern "C" fn(*mut i32, sys::CUdevice, sys::CUdevice) -> sys::CUresult =
                *sys::culib()
                .get(b"cuDeviceCanAccessPeer\0")
                .map_err(|e| BackendError::Init(format!("symbol: {e}")))?;
            let mut ok: i32 = 0;
            can(&mut ok, self.ctx.cu_device(), peer.ctx.cu_device()).result().map_err(|e| BackendError::Init(format!("{e:?}")))?;
            if ok != 1 {
                return Err(BackendError::LawViolation(
                    "P2P 不可达(cuDeviceCanAccessPeer=0):检查驱动 P2P 支持与黑名单",
                ));
            }
            self.ctx.bind_to_thread().map_err(|e| BackendError::Init(format!("{e:?}")))?;
            let en: unsafe extern "C" fn(sys::CUcontext, u32) -> sys::CUresult =
                *sys::culib()
                .get(b"cuCtxEnablePeerAccess\0")
                .map_err(|e| BackendError::Init(format!("symbol: {e}")))?;
            en(peer.ctx.cu_ctx(), 0).result().map_err(|e| BackendError::Init(format!("{e:?}")))?;
        }
        Ok(())
    }

    fn map_remote_internal<T: MemValue>(
        &self,
        peer: &DeviceDesc,
        ptr: *mut T,
        len: usize,
    ) -> Result<RemoteBuf<T>, BackendError> {
        let mut st = self.gov.stats.lock();
        st.peer_mappings += 1;
        st.peer_mapped_bytes += (len * std::mem::size_of::<T>()) as u64;
        Ok(RemoteBuf {
            ptr,
            len,
            peer_uuid: peer.uuid.clone(),
            gov: Arc::clone(&self.gov),
        })
    }
}

impl Device for OwlCuda {
    type Pool = CudaPool;

    type Persistent<T: MemValue> = Persistent<T>;
    type Scratch<T: MemValue> = Scratch<T>;

    fn desc(&self) -> &DeviceDesc {
        &self.desc
    }

    fn arch(&self) -> Arch {
        self.desc().arch
    }

    fn phase(&self) -> MemPhase {
        Governor::phase(&self.gov)
    }

    fn set_phase(&self, phase: MemPhase) {
        OwlCuda::set_phase(self, phase)
    }

    fn stats(&self) -> MemStats {
        OwlCuda::stats(self)
    }

    fn enable_peer_access(&self, peer: &Self) -> Result<(), BackendError> {
        OwlCuda::enable_peer_access(self, peer)
    }

    fn map_remote<T: MemValue>(
        &self,
        peer: &DeviceDesc,
        ptr: *mut T,
        len: usize,
    ) -> Result<Self::Remote<T>, BackendError> {
        OwlCuda::map_remote_internal(self, peer, ptr, len)
    }

    type Remote<T: MemValue> = RemoteBuf<T>;

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
            gov: Arc::clone(&self.gov),
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
            gov: Arc::clone(&self.gov),
        })
    }

    fn alloc_persistent_in<T: MemValue>(
        &self,
        pool: &CudaPool,
        len: usize,
    ) -> Result<Self::Persistent<T>, BackendError> {
        let bytes = (len * std::mem::size_of::<T>()) as u64;
        self.gov.charge_pool(pool, bytes)?; // A5.4 池校验
        let slice = self.alloc_zeroed_bytes(bytes as usize)?; // 全局预算 + 账本
        self.gov.stats.lock().persistent_allocs += 1;
        Ok(Persistent::new(slice, len, bytes, Some(pool.id), self))
    }

    fn htod_persistent_in<T: MemValue>(
        &self,
        pool: &CudaPool,
        src: Vec<T>,
    ) -> Result<Self::Persistent<T>, BackendError> {
        let len = src.len();
        let bytes = (len * std::mem::size_of::<T>()) as u64;
        self.gov.charge_pool(pool, bytes)?;
        let mut slice = self.alloc_zeroed_bytes(bytes as usize)?;
        let host_bytes =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u8, bytes as usize) };
        if let Err(e) = self.stream.memcpy_htod(host_bytes, &mut slice) {
            self.gov.uncharge(bytes);
            self.gov.uncharge_pool(pool.id.0, bytes);
            return Err(BackendError::CopyFailed {
                dir: "htod",
                detail: e.to_string(),
            });
        }
        self.gov.stats.lock().persistent_allocs += 1;
        Ok(Persistent::new(slice, len, bytes, Some(pool.id), self))
    }

    fn alloc_scratch_in<T: MemValue>(
        &self,
        pool: &CudaPool,
        len: usize,
    ) -> Result<Self::Scratch<T>, BackendError> {
        let bytes = (len * std::mem::size_of::<T>()) as u64;
        self.gov.charge_pool(pool, bytes)?;
        let slice = self.alloc_zeroed_bytes(bytes as usize)?;
        self.gov.stats.lock().scratch_allocs += 1;
        Ok(Scratch {
            inner: Some(slice),
            len,
            stream: Arc::clone(&self.stream),
            pool: Some(pool.id),
            bytes,
            gov: Arc::clone(&self.gov),
            _marker: std::marker::PhantomData,
        })
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

/// CUDA 厂商栈:枚举 + 按 UUID 打开设备。不记账(账本在 OwlCuda/Device)。
pub struct CudaBackend;

impl Backend for CudaBackend {
    type Device = OwlCuda;

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
        OwlCuda::new_by_uuid(uuid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 真机冒烟:iface 契约 + 分配分域 + A1.2 延迟释放 + 净空窗口归还。
    /// 需要至少一块 CUDA 设备。
    #[test]
    fn backend_contract_defers_free_during_live_phase() {
        fn use_backend<B: Device>(b: &B) {
            // A5.1:建池 = 分解表逐行实体化
            let weights = b
                .create_pool(PoolConfig {
                    name: "weights".into(),
                    kind: PoolKind::Weights,
                    bytes: 1 << 20,
                })
                .unwrap();
            let scratch_pool = b
                .create_pool(PoolConfig {
                    name: "scratch".into(),
                    kind: PoolKind::Scratch,
                    bytes: 64 << 10,
                })
                .unwrap();

            // 示例 API ①:池内持久分配(上层无 cudarc 类型)
            let buf = b.alloc_persistent_in::<f32>(&weights, 1024).expect("alloc");
            assert_eq!(DevBuf::<f32>::len(&buf), 1024);
            assert!(!buf.device_ptr().is_null());

            // 示例 API ②:池内暂存分配
            let scratch = b.alloc_scratch_in::<f32>(&scratch_pool, 4096).expect("scratch");
            assert_eq!(DevBuf::<f32>::len(&scratch), 4096);

            // htod 持久写入(入池)
            let hb = b.htod_persistent_in::<u32>(&weights, (0..64).collect()).unwrap();
            assert_eq!(DevBuf::<u32>::len(&hb), 64);

            // 池校验:scratch 池(64KiB)装不下 1MiB → A5.4 PoolExhausted
            let over = b.alloc_scratch_in::<f32>(&scratch_pool, 1024 * 1024);
            assert!(matches!(over, Err(BackendError::PoolExhausted { .. })));

            // 用量快照:weights 池 = 1024*4 + 64*4 字节
            let u = weights.usage();
            assert_eq!(u.used, (1024 + 64) * 4);

            // 图存活期 drop → 延迟(全局 + 池账都延迟归还)
            b.set_phase(MemPhase::Live);
            let victim = b.alloc_persistent_in::<f32>(&weights, 16).unwrap();
            drop(victim);
            assert_eq!(b.stats().deferred_frees, 1);
            assert_eq!(b.stats().drained_frees, 0);

            // 回 Idle → 延迟队列统一归还,池账同步回落
            b.set_phase(MemPhase::Idle);
            let s = b.stats();
            assert_eq!(s.deferred_frees, 1);
            assert_eq!(s.drained_frees, 1);
            let u = weights.usage();
            assert_eq!(u.used, (1024 + 64) * 4); // victim 的 64B 已还
            assert_eq!(b.phase(), MemPhase::Idle);
        }

        // Backend 枚举 → 按 UUID 打开(上层全程只见 iface 的 Backend/Device)
        let backend = CudaBackend;
        let devs = backend.enumerate().expect("需要 CUDA 设备");
        assert!(!devs.is_empty());
        let mut cuda = backend.open(&devs[0].uuid).expect("open by uuid");
        assert_eq!(cuda.desc().uuid, devs[0].uuid);
        assert!(cuda.desc().total_bytes > 0);
        use_backend(&mut cuda); // 静态分发:上层全程只见 Device trait
    }

    /// A5:预算合同——超支 fail-fast,账本可归因
    #[test]
    fn budget_violation_fails_fast() {
        let ctx = OwlCuda::new(0).expect("需要 CUDA 设备");
        ctx.set_budget(Budget {
            bytes: 1024 * 1024,
            reserve_floor: 512 * 1024,
        });

        // 预算内分配正常(经池)
        let pool = ctx
            .create_pool(PoolConfig {
                name: "test".into(),
                kind: PoolKind::Scratch,
                bytes: 8 << 20,
            })
            .unwrap();
        let ok = ctx.alloc_persistent_in::<u8>(&pool, 1024).unwrap();
        drop(ok);
        ctx.set_phase(MemPhase::Idle); // 归还,账本归零基线

        // 超支分配 → LawViolation(A5.4 fail-fast)
        let huge = ctx.alloc_persistent_in::<u8>(&pool, 8 * 1024 * 1024);
        assert!(matches!(
            huge,
            Err(BackendError::LawViolation(msg)) if msg.contains("A5.4")
        ));
        // 失败分配已回滚账本
        let led = ctx.ledger();
        assert!(led.bytes_alive < 1024 * 1024);

        // A5.3 对账原语可用
        let (free, _total) = ctx.mem_get_info().unwrap();
        assert!(free > 0);
    }

    /// A2.8:VMM 分配(2MiB 粒度,可被对端 P2P 映射的唯一合法路径)
    #[test]
    fn vmm_alloc_granularity_and_ledger() {
        let ctx = OwlCuda::new(0).expect("需要 CUDA 设备");
        // 1KB 请求 → 粒度向上取整(≥2MiB)
        let buf = ctx.vmm_alloc(1024).expect("vmm_alloc");
        assert!(buf.bytes() >= 2 * 1024 * 1024);
        assert!(buf.bytes() % (2 * 1024 * 1024) == 0);
        assert!(buf.device_ptr() != 0);
        // 账本:粒度取整后的字节已入账
        assert_eq!(ctx.ledger().bytes_alive, buf.bytes() as u64);
        drop(buf);
        assert_eq!(ctx.ledger().bytes_alive, 0);
    }
}

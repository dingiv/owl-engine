//! CUDA 设备实例:一张卡,账本 + 池组 + 相位机的宿主。
//! UUID 钉卡 + P2P 窄口 + `Backend`/`Device` 契约实现。

use super::buffers::{Persistent, RemoteBuf, Scratch, VmmBuf};
use super::governor::{Budget, Governor, LedgerSnapshot};
use super::graph::CaptureSession;
use super::pool::CudaPool;
use cudarc::driver::{CudaContext, CudaStream, result, sys};
use owl_iface::{
    Arch, Backend, BackendError, BackendFamily, BufToken, Device, DeviceDesc, MemPhase,
    MemStats, MemValue, PoolConfig, PoolId,
 PoolKind, };
use std::sync::Arc;

/// CUDA 设备实例:一个 CudaDevice 绑定一张卡(UUID 钉),账本独立。
#[derive(Clone)]
pub struct CudaDevice {
    pub(crate) inner: Arc<DeviceInner>,
}

/// 设备内核数据(Arc 共享;命名池挂这里,权重/暂存各一,env 可调大小)
pub struct DeviceInner {
    pub(crate) ctx: Arc<CudaContext>,
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) gov: Arc<Governor>,
    pub(crate) desc: DeviceDesc,
    /// 命名池实例(2026-09-23 用户裁决:分配只经池,Device 不分担)
    pub(crate) weights_pool: std::sync::OnceLock<Arc<CudaPool>>,
    pub(crate) scratch_pool: std::sync::OnceLock<Arc<CudaPool>>,
}

impl std::ops::Deref for CudaDevice {
    type Target = DeviceInner;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
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
        // M1②:主显存流 = 显式 non-blocking(legacy NULL 流不可捕获,
        // T4 判例;OpsCtx/池分配/重放全部落在这条流上)
        let stream = ctx
            .new_stream()
            .map_err(|e| BackendError::Init(format!("main stream: {e:?}")))?;
        let uuid = cuda_uuid(ordinal)?;
        let (_free, total) = ctx
            .mem_get_info()
            .map_err(|e| BackendError::Init(format!("mem_get_info: {e}")))?;
        let inner = Arc::new(DeviceInner {
            ctx: Arc::clone(&ctx),
            stream: Arc::clone(&stream),
            gov: Arc::new(Governor {
                ctx: Some(ctx),
                ..Default::default()
            }),
            desc: DeviceDesc {
                uuid,
                arch: Arch::Sm86, // TODO(M1): compute capability 运行时读取
                total_bytes: total as u64,
            },
            weights_pool: std::sync::OnceLock::new(),
            scratch_pool: std::sync::OnceLock::new(),
        });
        let dev = Self { inner };
        // 命名池初始化(用户裁决:分配只经池;大小 env 可调,默认 2G/512M)
        let wbytes = std::env::var("OWL_WEIGHTS_POOL_BYTES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2 << 30);
        let sbytes = std::env::var("OWL_SCRATCH_POOL_BYTES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(512 << 20);
        let wp = create_pool_for(&dev, PoolConfig {
            name: format!("weights-{}", std::process::id()),
            kind: PoolKind::Weights,
            bytes: wbytes,
        })?;
        let sp = create_pool_for(&dev, PoolConfig {
            name: format!("scratch-{}", std::process::id()),
            kind: PoolKind::Scratch,
            bytes: sbytes,
        })?;
        let _ = dev.inner.weights_pool.set(wp);
        let _ = dev.inner.scratch_pool.set(sp);
        Ok(dev)
    }

    /// 命名池:权重(常驻,分配走它)
    pub fn weights_pool(&self) -> Arc<CudaPool> {
        Arc::clone(self.inner.weights_pool.get().expect("weights 池未初始化"))
    }

    /// 命名池:暂存(捕获期可借,用完还)
    pub fn scratch_pool(&self) -> Arc<CudaPool> {
        Arc::clone(self.inner.scratch_pool.get().expect("scratch 池未初始化"))
    }

    pub fn ctx(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// 设备主显存流(非阻塞;E 阶段 kernel/cublas/图重放的默认目标)
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    // ==================== memx:流序 memcpy 收口(P0-3 修复) ====================
    // 审计裁定(docs/audit/README.md P0-3):NULL 流 cuMemcpy*_v2 与
    // non-blocking 主流互不同步(safetensors 零读/rotary 零读/cu_seqlens
    // 清零四案根因)。全部 H2D/D2H 统一走主显存流:async 发射 + synchronize,
    // 一次保证「与在飞 kernel 有序 + host 立即可读」。

    /// 流序 D2H(f32):src 在主流上与在飞 kernel 有序,返回时 host 可读。
    pub fn memcpy_dtoh_f32(
        &self,
        stream: &Arc<CudaStream>,
        src: *const f32,
        out: &mut [f32],
    ) -> Result<(), String> {
        self.ctx.bind_to_thread().map_err(|e| format!("{e:?}"))?;
        unsafe {
            result::memcpy_dtoh_async(out, src as sys::CUdeviceptr, stream.cu_stream())
        }
        .map_err(|e| format!("memcpy_dtoh_async: {e:?}"))?;
        stream.synchronize().map_err(|e| format!("stream sync: {e:?}"))
    }

    /// 流序 H2D(f32):host 数据经主流写入,后续主流 kernel 天然有序。
    pub fn memcpy_htod_f32(
        &self,
        stream: &Arc<CudaStream>,
        dst: *mut f32,
        src: &[f32],
    ) -> Result<(), String> {
        self.ctx.bind_to_thread().map_err(|e| format!("{e:?}"))?;
        unsafe { result::memcpy_htod_async(dst as sys::CUdeviceptr, src, stream.cu_stream()) }
            .map_err(|e| format!("memcpy_htod_async: {e:?}"))?;
        Ok(())
    }

    /// 流序 D2H(u32):槽位/索引读回。
    pub fn memcpy_dtoh_u32(
        &self,
        stream: &Arc<CudaStream>,
        src: *const u32,
        out: &mut [u32],
    ) -> Result<(), String> {
        self.ctx.bind_to_thread().map_err(|e| format!("{e:?}"))?;
        unsafe { result::memcpy_dtoh_async(out, src as sys::CUdeviceptr, stream.cu_stream()) }
            .map_err(|e| format!("memcpy_dtoh_async: {e:?}"))?;
        stream.synchronize().map_err(|e| format!("stream sync: {e:?}"))
    }

    /// 流序 H2D(u32)。
    pub fn memcpy_htod_u32(
        &self,
        stream: &Arc<CudaStream>,
        dst: *mut u32,
        src: &[u32],
    ) -> Result<(), String> {
        self.ctx.bind_to_thread().map_err(|e| format!("{e:?}"))?;
        unsafe { result::memcpy_htod_async(dst as sys::CUdeviceptr, src, stream.cu_stream()) }
            .map_err(|e| format!("memcpy_htod_async: {e:?}"))?;
        Ok(())
    }

    /// 流序 H2D(字节级;泛型 T 位型调用点统一入口)
    pub fn memcpy_htod_bytes(
        &self,
        stream: &Arc<CudaStream>,
        dst: *mut u8,
        src: &[u8],
    ) -> Result<(), String> {
        self.ctx.bind_to_thread().map_err(|e| format!("{e:?}"))?;
        unsafe {
            sys::cuMemcpyHtoDAsync_v2(
                dst as sys::CUdeviceptr,
                src.as_ptr() as *const _,
                src.len(),
                stream.cu_stream(),
            )
        }
        .result()
        .map_err(|e| format!("htod_async: {e:?}"))
    }

    /// 流序 D2H(字节级;泛型 T 位型调用点统一入口)。返回时 host 可读。
    pub fn memcpy_dtoh_bytes(
        &self,
        stream: &Arc<CudaStream>,
        src: *const u8,
        out: &mut [u8],
    ) -> Result<(), String> {
        self.ctx.bind_to_thread().map_err(|e| format!("{e:?}"))?;
        unsafe {
            sys::cuMemcpyDtoHAsync_v2(
                out.as_mut_ptr() as *mut _,
                src as sys::CUdeviceptr,
                out.len(),
                stream.cu_stream(),
            )
        }
        .result()
        .map_err(|e| format!("dtoh_async: {e:?}"))?;
        stream.synchronize().map_err(|e| format!("stream sync: {e:?}"))
    }

    /// 姿势 6(warmup 门禁):记一次 kernel 发射(算子层/cublas 每次发射调用)
    pub fn note_launch(&self) {
        self.gov.note_launch();
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

    /// 开启捕获会话(GraphLease 入口)。
    /// 会话自带一条 non-blocking 捕获流(legacy 流不可捕获,T4 判例)。
    pub fn capture_session(&self) -> Result<CaptureSession, BackendError> {
        let stream = self
            .ctx
            .new_stream()
            .map_err(|e| BackendError::Init(format!("capture stream: {e:?}")))?;
        Ok(CaptureSession {
            stream,
            leases: Vec::new(),
            keepalive: Vec::new(),
            gov: Arc::clone(&self.gov),
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

// ---- Backend / Device 契约实现 ----

/// CUDA 厂商栈:枚举 + 按 UUID 打开设备。不记账(账本在 Device 上)。
pub struct CudaBackend;

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
        Ok((*create_pool_for(self, cfg)?).clone())
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

}

/// 池构造(部件注入;`Device::create_pool` 与命名池初始化共用)。
/// 注:CudaPool 内嵌 `dev: CudaDevice` 与 inner 的池槽构成有界 Arc 环
/// (命名池与设备同生命周期,进程级常驻,不计泄漏)。
pub(crate) fn create_pool_for(
    dev: &CudaDevice,
    cfg: PoolConfig,
) -> Result<Arc<CudaPool>, BackendError> {
    let (ctx, stream, gov) = (&dev.inner.ctx, &dev.inner.stream, &dev.inner.gov);
    let mut pools = gov.pools.lock();
    if pools.values().any(|p| p.name == cfg.name) {
        return Err(BackendError::Init(format!(
            "A5.1: 重复池名 '{}'——分解表不应有两行同名账",
            cfg.name
        )));
    }
    let id = gov
        .next_pool_id
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    pools.insert(
        id,
        crate::pool::PoolLedger {
            name: cfg.name.clone(),
            kind: cfg.kind,
            capacity: cfg.bytes,
            used: 0,
            peak: 0,
        },
    );
    Ok(Arc::new(CudaPool {
        id: PoolId(id),
        name: cfg.name.clone(),
        kind: cfg.kind,
        capacity: cfg.bytes,
        ctx: Arc::clone(ctx),
        stream: Arc::clone(stream),
        gov: Arc::clone(gov),
        dev: dev.clone(),
    }))
}

//! CUDA 设备实例:一张卡,相位机/BufToken 台账 + 默认池的宿主(账本在池上)。
//! UUID 钉卡 + P2P 窄口 + `Backend`/`Device` 契约实现。

/// `Backend::open` 无容量入参时的默认池容量;需要定制容量请走
/// `CudaDevice::new/new_by_uuid` 构造参数(2026-09-23 裁决:容量构造指定)。
pub const DEFAULT_POOL_BYTES: u64 = 4 << 30;

use super::buffers::{RemoteBuf, VmmBuf};
use super::governor::{Budget, Governor, LedgerSnapshot};
use super::graph::CaptureSession;
use super::pool::{CudaPool, CudaPoolBuf};
use cudarc::driver::{CudaContext, CudaStream, result, sys};
use owl_iface::{
    Arch, Backend, BackendError, BackendFamily, BufToken, Device, DeviceDesc, MemPhase,
    MemStats, MemValue, PoolConfig, PoolId,
 PoolKind, };
use std::collections::HashMap;
use std::sync::Arc;

/// CUDA 设备实例:一个 CudaDevice 绑定一张卡(UUID 钉),账本独立。
#[derive(Clone)]
pub struct CudaDevice {
    pub(crate) inner: Arc<DeviceInner>,
}

/// 设备内核数据(Arc 共享)。2026-09-23 用户裁决:单动态池挂这里
/// (权重/暂存合一,General 档);账本归池持有,Device 不持账本对象。
pub struct DeviceInner {
    pub(crate) ctx: Arc<CudaContext>,
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) gov: Arc<Governor>, // 相位机/BufToken 台账/延迟队列(非账本)
    pub(crate) desc: DeviceDesc,
    /// 池注册表(强持;2026-09-24 破环:原在 Governor 且与 CudaPool.gov
    /// 至成强-强环,池/账本永不可回收——根 = 设备,治理不再回指池)
    pub(crate) pools: parking_lot::Mutex<HashMap<u64, Arc<CudaPool>>>,
    /// 默认动态池(General 档;容量由构造函数参数指定;与设备同生命周期,
    /// 初始化期未装填前访问 = 编序违约)
    pool: std::sync::OnceLock<Arc<CudaPool>>,
}

impl std::ops::Deref for CudaDevice {
    type Target = DeviceInner;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl CudaDevice {
    /// 按 UUID 钉卡(唯一合法入口;数字序禁用,09-19 事故判例)。
    /// `pool_bytes` = 默认动态池容量(构造函数参数指定,无 env 魔法)。
    pub fn new_by_uuid(uuid: &str, pool_bytes: u64) -> Result<Self, BackendError> {
        let ordinal = resolve_uuid_ordinal(uuid)?;
        let s = Self::new(ordinal, pool_bytes)?;
        if s.desc.uuid != uuid {
            return Err(BackendError::Init(format!(
                "UUID 不匹配:请求 {uuid},ordinal {ordinal} 实际 {}",
                s.desc.uuid
            )));
        }
        Ok(s)
    }

    pub fn new(ordinal: usize, pool_bytes: u64) -> Result<Self, BackendError> {
        Self::with_pools(ordinal, pool_bytes, &[])
    }

    /// 建设备 + 随设备出生的全部池(2026-09-24 用户裁决:create_pool 退役,
    /// 池是设备的出生属性,不是运行期动态产物)。默认 General 池恒在,
    /// `extra` 声明额外池(如 PeerShared);重名在出生时即拒。
    pub fn with_pools(
        ordinal: usize,
        default_pool_bytes: u64,
        extra: &[PoolConfig],
    ) -> Result<Self, BackendError> {
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
            pools: parking_lot::Mutex::new(HashMap::new()),
            pool: std::sync::OnceLock::new(),
        });
        // P0-4 毒化探测器:环境初始化(测试可用 set_poison_freed 覆写)
        if std::env::var_os("OWL_POISON_FREED").is_some_and(|v| v != "0") {
            inner.gov.set_poison_freed(true);
        }
        let dev = Self { inner };
        // 池随设备出生(默认 General 池 + extra 声明池;重名即拒)
        let p = create_pool_for(&dev, PoolConfig {
            name: format!("owl-default-{}", std::process::id()),
            kind: PoolKind::General,
            bytes: default_pool_bytes,
        })?;
        let _ = dev.inner.pool.set(p);
        for cfg in extra {
            create_pool_for(&dev, cfg.clone())?;
        }
        Ok(dev)
    }

    /// 按名取池(出生声明池的取回口)
    pub fn pool_by_name(&self, name: &str) -> Option<CudaPool> {
        self.inner
            .pools
            .lock()
            .values()
            .find(|p| p.name == name)
            .map(|p| (**p).clone())
    }

    /// 默认动态池(权重/暂存合一;唯一随设备附赠的池)
    pub fn default_pool(&self) -> Arc<CudaPool> {
        Arc::clone(self.inner.pool.get().expect("默认池未初始化"))
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
    /// 设备级策略:广播到全部池账本,新建池建账时继承。
    pub fn set_budget(&self, budget: Budget) {
        *self.gov.budget_seed.lock() = Some(budget);
        for p in self.inner.pools.lock().values() {
            p.set_budget(budget);
        }
    }

    /// P0-4 毒化探测器开关(测试/诊断覆写;环境默认 OWL_POISON_FREED=1)
    pub fn set_poison_freed(&self, on: bool) {
        self.gov.set_poison_freed(on);
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
            // VMM 独立入口非池路径,字节账记在默认池账本上
            self.default_pool().charge(rounded as u64)?;
            let mut ptr: sys::CUdeviceptr = 0;
            if cuMemAddressReserve(&mut ptr, rounded, gran, 0, 0) != CUDA_SUCCESS {
                self.default_pool().uncharge(rounded as u64);
                return Err(BackendError::AllocFailed {
                    kind: "vmm-reserve",
                    len: rounded,
                    detail: "cuMemAddressReserve".into(),
                });
            }
            let mut chunk: sys::CUmemGenericAllocationHandle = Default::default();
            if cuMemCreate(&mut chunk, rounded, &props, 0) != CUDA_SUCCESS {
                let _ = cuMemAddressFree(ptr, rounded);
                self.default_pool().uncharge(rounded as u64);
                return Err(BackendError::AllocFailed {
                    kind: "vmm-create",
                    len: rounded,
                    detail: "cuMemCreate".into(),
                });
            }
            if cuMemMap(ptr, rounded, 0, chunk, 0) != CUDA_SUCCESS {
                let _ = cuMemRelease(chunk);
                let _ = cuMemAddressFree(ptr, rounded);
                self.default_pool().uncharge(rounded as u64);
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
                self.default_pool().uncharge(rounded as u64);
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
                pool: Arc::clone(&self.default_pool()),
            })
        }
    }

    /// 哨兵①:缓冲令牌存活校验(replay 前世代检查;未来 M1 用)
    pub fn validate_token(&self, t: &BufToken) -> bool {
        self.gov.validate(t)
    }

/// 账本快照(A5.4 违约时打印;字节账在池上,跨池聚合 + 设备侧计数器)
pub fn ledger(&self) -> LedgerSnapshot {
    let mut bytes_alive = 0u64;
    let mut bytes_allocated_total = 0u64;
    for p in self.inner.pools.lock().values() {
        let (a, t) = p.alive_and_total();
        bytes_alive += a;
        bytes_allocated_total += t;
    }
    LedgerSnapshot {
        bytes_alive,
        bytes_allocated_total,
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
        CudaDevice::new_by_uuid(uuid, DEFAULT_POOL_BYTES)
    }
}

impl Device for CudaDevice {
    type Remote<T: MemValue> = RemoteBuf<T>;
    type Pool = CudaPool;
    /// 裸池块(u8 字节面)
    type Bytes = CudaPoolBuf;

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

    // FIXME: 为什么这里面还会有 create_pool 呢
    fn pool(&self, id: PoolId) -> Result<Self::Pool, BackendError> {
        let pools = self.inner.pools.lock();
        let p = pools.get(&id.0).ok_or(BackendError::UnknownPool(id.0))?;
        Ok((**p).clone())
    }
}

/// 池构造(部件注入;`Device::create_pool` 与默认池初始化共用)。
/// 池构造(部件注入;账本是池的字段,2026-09-24 内联裁决)。
/// 注:CudaPool 内嵌 `dev: CudaDevice` 与 inner 的池槽构成有界 Arc 环
/// (默认池与设备同生命周期,进程级常驻,不计泄漏)。
pub(crate) fn create_pool_for(
    dev: &CudaDevice,
    cfg: PoolConfig,
) -> Result<Arc<CudaPool>, BackendError> {
    let (ctx, stream, gov) = (&dev.inner.ctx, &dev.inner.stream, &dev.inner.gov);
    let mut pools = dev.inner.pools.lock();
    if pools.values().any(|p| p.name == cfg.name) {
        return Err(BackendError::Init(format!(
            "A5.1: 重复池名 '{}'——分解表不应有两行同名账",
            cfg.name
        )));
    }
    let id = gov
        .next_pool_id
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let pool = Arc::new(CudaPool {
        id: PoolId(id),
        name: cfg.name.clone(),
        kind: cfg.kind,
        ctx: Arc::clone(ctx),
        stream: Arc::clone(stream),
        gov: Arc::clone(gov),
        capacity: cfg.bytes,
        account: std::sync::Arc::new(parking_lot::Mutex::new(Default::default())),
        budget: Arc::new(parking_lot::Mutex::new(*gov.budget_seed.lock())),
        dev: dev.clone(),
    });
    pools.insert(id, Arc::clone(&pool));
    Ok(pool)
}

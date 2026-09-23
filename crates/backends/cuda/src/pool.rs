//! 显存池:**分配者**。持有 driver 原语句柄,按 PoolKind 路由物理路径
//! (PeerShared → VMM 2MiB 粒度,其余 → stream-ordered)。

use super::governor::{Budget, Governor};
use crate::device::CudaDevice;
use cudarc::driver::sys;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use owl_iface::{
    BackendError, BufToken, DevBuf, MemPhase, OpaqueDevBuf, Pool, PoolBuf,
    PoolId, PoolKind, PoolUsage,
};
use parking_lot::Mutex;
use std::sync::Arc;

/// 池内账本(2026-09-23 用户裁决:内存记账归池，Device 不持账本对象)。
/// 记账三件套:字节存活账 + 池容量账 + A5 预算。相位机/BufToken 台账
/// 不在此(仍归 Device 侧 Governor)。
/// 2026-09-24:四项字节账收进单一 Mutex(快照不可撕裂、入/归账原子；
/// 旧五锁形态 usage() 可读到 used/alive 分属不同时刻)。
pub(crate) struct Ledger {
    /// A5.3 driver 侧对账用(池构造时注入)
    pub(crate) ctx: Option<Arc<CudaContext>>,
    /// 池容量上限(bytes)
    pub(crate) capacity: u64,
    /// 字节账单锁四联(used/peak/alive/allocated_total)
    state: parking_lot::Mutex<LedgerState>,
    /// A5.1 预算(None = 未设;设置后每次分配断言不超)
    pub(crate) budget: Mutex<Option<Budget>>,
}

#[derive(Default)]
pub(crate) struct LedgerState {
    /// 池内在用字节
    pub(crate) used: u64,
    /// 峰值(诊断)
    pub(crate) peak: u64,
    /// A5.2 存活字节(本池)
    pub(crate) bytes_alive: u64,
    /// A5.2 累计分配字节(本池)
    pub(crate) bytes_allocated_total: u64,
}

impl Ledger {
    /// 建账(device.rs 构造池时用)
    pub(crate) fn new(
        ctx: Option<Arc<CudaContext>>,
        capacity: u64,
        budget: Option<Budget>,
    ) -> Self {
        Self {
            ctx,
            capacity,
            state: parking_lot::Mutex::new(LedgerState::default()),
            budget: Mutex::new(budget),
        }
    }
}

impl Ledger {
    /// 统一入账:预算闸(A5.4)→ 容量闸(池耗尽)→ 记账。
    /// 校验先行，失败不触账(调用方失败路径无需回滚——2026-09-22
    /// 双重扣减下溢判例)。入/归账在单一 state 锁内原子完成。
    /// A5.3 driver 对账降频:仅在逼近预算(剩余 <10%)时探
    /// mem_get_info——旧行为每次分配都同步驱动调用，热路径不可持续。
    pub(crate) fn charge(&self, pool_name: &str, bytes: u64) -> Result<(), BackendError> {
        if let Some(b) = *self.budget.lock() {
            let alive = self.state.lock().bytes_alive;
            if alive + bytes > b.bytes {
                return Err(BackendError::LawViolation(
                    "A5.4 显存超支(引擎账本超预算,详见账本快照日志)",
                ));
            }
            // A5.3 第三道闸(降频):逼近预算才 driver 对账
            if alive + bytes >= b.bytes - b.bytes / 10 {
                if let Some(ctx) = &self.ctx {
                    if let Ok((free, _total)) = ctx.mem_get_info() {
                        if (free as u64) < b.reserve_floor {
                            eprintln!(
                                "[owl-mem] WARN: free {free}B < reserve_floor {}B(A5 警线,账本 alive={alive})",
                                b.reserve_floor,
                            );
                        }
                    }
                }
            }
        }
        let mut st = self.state.lock();
        let available = self.capacity - st.used;
        if bytes > available {
            return Err(BackendError::PoolExhausted {
                pool: pool_name.to_string(),
                needed: bytes,
                available,
                capacity: self.capacity,
            });
        }
        st.used += bytes;
        st.peak = st.peak.max(st.used);
        st.bytes_alive += bytes;
        st.bytes_allocated_total += bytes;
        Ok(())
    }

    /// 归账(单锁原子;下溢饱和 + 告警)
    pub(crate) fn uncharge(&self, bytes: u64) {
        let mut st = self.state.lock();
        if st.bytes_alive < bytes {
            eprintln!(
                "[owl-mem][BUG] uncharge underflow: alive={} bytes={bytes}",
                st.bytes_alive,
            );
        }
        st.bytes_alive = st.bytes_alive.saturating_sub(bytes);
        st.used = st.used.saturating_sub(bytes);
    }

    pub(crate) fn set_budget(&self, budget: Budget) {
        *self.budget.lock() = Some(budget);
    }

    /// 原子快照(单锁;跨项无撕裂)
    pub(crate) fn alive_and_total(&self) -> (u64, u64) {
        let st = self.state.lock();
        (st.bytes_alive, st.bytes_allocated_total)
    }

    pub(crate) fn usage(&self) -> PoolUsage {
        let st = self.state.lock();
        PoolUsage {
            capacity: self.capacity,
            used: st.used,
            peak: st.peak,
        }
    }
}

/// 显存池:**分配者**。持有 driver 原语句柄,按 PoolKind 路由物理路径。
#[derive(Clone)]
pub struct CudaPool {
    pub(crate) id: PoolId,
    pub(crate) name: String,
    pub(crate) kind: PoolKind,
    pub(crate) ctx: Arc<CudaContext>,
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) gov: Arc<Governor>,
    /// 本池账本(字节账 + 容量账 + 预算;2026-09-23 裁决)
    pub(crate) ledger: Arc<Ledger>,
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
        self.ledger.capacity
    }
    fn usage(&self) -> PoolUsage {
        self.ledger.usage()
    }

    fn malloc_scratch(&self, bytes: u64) -> Result<PoolBuf, BackendError> {
        self.kind_check_any(&[PoolKind::Scratch, PoolKind::General])?;
        Ok(PoolBuf::wrap(Box::new(self.malloc_scratch_buf(bytes)?)))
    }

    fn malloc_persistent(&self, bytes: u64) -> Result<PoolBuf, BackendError> {
        self.kind_check_any(&[
            PoolKind::Weights,
            PoolKind::KvCache,
            PoolKind::Workspace,
            PoolKind::General,
        ])?;
        Ok(PoolBuf::wrap(Box::new(self.malloc_inner(bytes)?)))
    }

    fn malloc_peer_shared(&self, bytes: u64) -> Result<PoolBuf, BackendError> {
        self.kind_check(PoolKind::PeerShared)?;
        Ok(PoolBuf::wrap(Box::new(self.malloc_inner(bytes)?)))
    }

    // ---- 字节域第一公民(iface 字节化面;泛型方法由默认转发承载)----

    fn alloc_bytes_persistent(&self, n_bytes: usize) -> Result<CudaPoolBuf, BackendError> {
        let buf = self.malloc_persistent_buf(n_bytes as u64)?;
        self.gov.stats.lock().persistent_allocs += 1;
        Ok(buf)
    }

    fn htod_bytes_persistent(&self, src: &[u8]) -> Result<CudaPoolBuf, BackendError> {
        let bytes = src.len();
        let buf = self.malloc_persistent_buf(bytes as u64)?;
        self.ctx.bind_to_thread().map_err(|e| BackendError::Init(format!("{e:?}")))?;
        // P0-3 口径:H2D 必须走池流(async + sync),NULL 流与主流无序
        // (审计同族案:cu_seqlens 清零/cat 脏数据)
        unsafe {
            use cudarc::driver::sys;
            sys::cuMemcpyHtoDAsync_v2(
                buf.device_ptr() as sys::CUdeviceptr,
                src.as_ptr() as *const core::ffi::c_void,
                bytes,
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
        Ok(buf)
    }

    fn alloc_bytes_scratch(&self, n_bytes: usize) -> Result<CudaPoolBuf, BackendError> {
        let buf = self.malloc_scratch_buf(n_bytes as u64)?;
        self.gov.stats.lock().scratch_allocs += 1;
        Ok(buf)
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

    /// 归还本池账本(2026-09-23 裁决:账本归池)
    pub(crate) fn uncharge(&self, bytes: u64) {
        self.ledger.uncharge(bytes);
    }

    /// 内部:持久域具体分配(语义校验 + Arc 租约缓冲)
    pub(crate) fn malloc_persistent_buf(&self, bytes: u64) -> Result<CudaPoolBuf, BackendError> {
        self.kind_check_any(&[
            PoolKind::Weights,
            PoolKind::KvCache,
            PoolKind::Workspace,
            PoolKind::General,
        ])?;
        self.malloc_inner(bytes)
    }

    /// 内部:暂存域具体分配
    pub(crate) fn malloc_scratch_buf(&self, bytes: u64) -> Result<CudaPoolBuf, BackendError> {
        self.kind_check_any(&[PoolKind::Scratch, PoolKind::General])?;
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
        self.ledger.charge(&self.name, effective)?;
        // charge 校验先行(失败不触账本)→ 此处失败无需回滚
        let backing = match self.kind {
            PoolKind::PeerShared => match self.vmm_alloc_raw(effective as usize) {
                Ok((b, r)) => {
                    debug_assert_eq!(r as u64, effective);
                    b
                }
                Err(e) => {
                    // 物理分配失败:账已入,回滚
                    self.uncharge(effective);
                    return Err(e);
                }
            },
            _ => match self.stream.alloc_zeros::<u8>(effective as usize) {
                Ok(slice) => PoolBacking::Slice(slice),
                Err(e) => {
                    self.uncharge(effective);
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
        // P1-2:捕获窗口内出生的块,出生即钉住(强引用)——防 Arc 在窗口内
        // 归零→释放→图内地址悬空;CaptureSession 定影时按水位线收编进图
        if self.gov.is_capturing() {
            self.gov.pin_capture_born(Arc::clone(&buf.inner));
        }
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
        let backing = std::mem::replace(self.backing.get_mut(), PoolBacking::Empty);
        let phase = self.gov.phase();
        if phase == MemPhase::Idle {
            release_backing(backing, &pool);
            pool.uncharge(bytes);
        } else if self.gov.is_capturing() {
            // P1-2:捕获窗口内死亡(pre-window 出生、窗口内 Arc 归零)——
            // 地址可能已烙进图,释放闭包停放至图 drop;旧路径 defer 到
            // Idle drain → 图存活期回放读悬空(毒化探测器实锤的缺口)
            self.gov.park_capture(Box::new(move || {
                release_backing(backing, &pool);
                pool.uncharge(bytes);
            }));
        } else {
            self.gov.defer(Box::new(move || {
                release_backing(backing, &pool);
                pool.uncharge(bytes);
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
    // 实现:池流 async memset + 流同步(同步而非纯 async:drop 返回后
    // 哨兵必须已生效,否则探测器自身与 host 侧悬空读竞速)。
    // 开关:Governor.poison_freed(OWL_POISON_FREED=1 初始化,测试可覆写)。
    // P1-2 修复(捕获窗口钉住)后接线——dry_run 判别③曾复现悬空。
    if pool.gov.poison_freed.load(std::sync::atomic::Ordering::Relaxed) {
        match &backing {
            PoolBacking::Slice(s) => {
                use cudarc::driver::DevicePtr;
                let (p, _sync) = s.device_ptr(&pool.stream);
                unsafe {
                    cudarc::driver::result::memset_d8_async(
                        p as sys::CUdeviceptr,
                        0xFF,
                        s.len(),
                        pool.stream.cu_stream(),
                    )
                }
                .unwrap_or_else(|e| eprintln!("[owl-mem] P0-4 归还哨兵 memset 失败: {e:?}"));
            }
            PoolBacking::Vmm { ptr, bytes, .. } => unsafe {
                cudarc::driver::result::memset_d8_async(
                    *ptr as sys::CUdeviceptr,
                    0xFF,
                    *bytes,
                    pool.stream.cu_stream(),
                )
            }
            .unwrap_or_else(|e| eprintln!("[owl-mem] P0-4 归还哨兵 VMM memset 失败: {e:?}")),
            PoolBacking::Empty => {}
        }
        pool.stream
            .synchronize()
            .unwrap_or_else(|e| eprintln!("[owl-mem] P0-4 归还哨兵流同步失败: {e:?}"));
    }

    if is_vmm {
        let _ = pool.ctx.synchronize();
    }
    drop(backing);
}

impl CudaPoolBuf {
    /// 令牌读取(哨兵①登记/校验用;pub = 跨 crate 的合并 Tensor 词汇面)
    pub fn token(&self) -> BufToken {
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

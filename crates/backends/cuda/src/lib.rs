//! owl-cuda —— cudarc **官方版**(crates.io)的治理封装层,实现
//! `owl_iface` 的 `Backend`/`Device`/`Pool` 三层契约(charter A1/A4/A5)。
//!
//! 选型注记(2026-09-22):官方 cudarc 0.19 已原生覆盖 fork 的图安全增量
//! (CudaGraph/stream-ordered 分配/13.x 绑定),按 A4"语义分歧不进第三方
//! 库"选官方。cudarc 仅在本 crate 导入;上层经 `ffi` 模块的精确白名单
//! 使用(整库 re-export 禁止)。
//!
//! 分配架构(裁决 5 + A5.2):
//! - **Pool 是分配者**:`CudaPool::malloc_*` 是 P 阶段唯一分配入口,
//!   校验链 = kind 语义 → 池余量(A5.4)→ 全局预算(A5.4)→ 物理分配;
//! - 物理路径按池类型路由:`PeerShared` → VMM(cuMemCreate,2MiB 粒度,
//!   A2.8);其余 → stream-ordered(捕获安全);
//! - 所有缓冲 drop 时自动归池账 + 全局账(非 Idle 相延迟到净空窗口,
//!   A1.2);BufToken/世代校验 = 哨兵①(结构化报错替代 Xid 盲死);
//! - Device 的 `alloc_*_in` 只是 `pool.malloc_*` 的类型化薄封装。

pub mod audit;
pub mod ffi;
pub(crate) mod governor;
pub(crate) mod graph;
pub(crate) mod pool;
pub(crate) mod device;
pub(crate) mod buffers;

/// 测试/示例的设备序号(环境变量 `OWL_TEST_DEVICE`,默认 0)。
/// 生产引擎与测试同卡会互相踩(kv cache 满载时测试抖动),
/// CI/真机验证用 `OWL_TEST_DEVICE=2` 指到空闲卡。
pub fn test_device_ordinal() -> usize {
    std::env::var("OWL_TEST_DEVICE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

pub use buffers::{Persistent, RemoteBuf, Scratch, VmmBuf};
pub use device::{CudaBackend, CudaDevice};
pub use governor::{Budget, LedgerSnapshot};
pub use graph::{CaptureFrame, CaptureSession, DeviceGraph};
pub use pool::CudaPool;
pub use owl_iface::{BackendError, MemPhase, MemStats};

#[cfg(test)]
mod tests {
    /// GPU 测试互斥:同一张卡上的测试串行化(双 context 并发图操作有
    /// GPU 级竞态,观察项 O-1;锁是测试卫生,不是引擎语义)
    static GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    fn gpu() -> std::sync::MutexGuard<'static, ()> {
        GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }


    use super::ffi::sys;
    use super::{BackendError, Budget, CudaBackend, CudaDevice, CudaPool};
    use owl_iface::{Backend as _, DevBuf, Device as _, MemPhase, PoolConfig, PoolKind, Pool as _};

    fn make() -> CudaDevice {
        CudaDevice::new(super::test_device_ordinal()).expect("需要 CUDA 设备")
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
    let _g = gpu();
        let dev = make();
        let pool = scratch_pool(&dev, "scratch", 64 << 10);

        let scratch = pool.alloc_scratch_in::<f32>(4096).unwrap();
        assert_eq!(DevBuf::<f32>::len(&scratch), 4096);
        assert_eq!(pool.usage().used, 16 * 1024);

        // 池耗尽 → A5.4 PoolExhausted
        let over = pool.alloc_scratch_in::<f32>(1024 * 1024);
        assert!(matches!(over, Err(BackendError::PoolExhausted { .. })));

        // Live 相 drop → 延迟(池账 + 全局账都不动)
        dev.set_phase(MemPhase::Live);
        let victim = pool.alloc_scratch_in::<f32>(16).unwrap();
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
    let _g = gpu();
        let dev = make();
        dev.set_budget(Budget {
            bytes: 1024 * 1024,
            reserve_floor: 512 * 1024,
        });
        let pool = persistent_pool(&dev, "b", 8 << 20);

        let ok = pool.alloc_persistent_in::<u8>(1024).unwrap();
        drop(ok);
        dev.set_phase(MemPhase::Idle);

        let huge = pool.alloc_persistent_in::<u8>(8 * 1024 * 1024);
        assert!(matches!(
            huge,
            Err(BackendError::LawViolation(msg)) if msg.contains("A5.4")
        ));
        assert!(dev.ledger().bytes_alive < 1024 * 1024);
        let (free, _total) = dev.mem_get_info().unwrap();
        assert!(free > 0);
    }

    /// GraphLease 核心验收:图存活期用户 drop 依赖缓冲,
    /// 强租约保证 replay 仍正确;图销毁后租约解,账本回基线。
    #[test]
    fn graph_lease_keeps_memory_alive_across_user_drop() {
    let _g = gpu();
        let dev = CudaDevice::new(super::test_device_ordinal()).expect("需要 CUDA 设备");
        let pool = dev
            .create_pool(PoolConfig {
                name: "lease-t".into(),
                kind: PoolKind::Weights,
                bytes: 1 << 20,
            })
            .unwrap();
        let t = pool.alloc_persistent_in::<u8>(4096).unwrap();
        let token = t.token.unwrap();
        let survivor = t.clone(); // 租约克隆(Graph keepalive 之外的第二证明)

        // 先填 0xFF(eager 发射,兼 warmup;必须在捕获开始前)
        unsafe {
            sys::cuMemsetD8Async(
                t.device_ptr() as sys::CUdeviceptr,
                0xFF,
                4096,
                dev.stream().cu_stream(),
            )
            .result()
            .unwrap();
        }
        dev.ctx().synchronize().unwrap();
        dev.note_launch(); // 姿势 6:上述 eager 发射计入 warmup 门禁

        // 捕获一个 memset(0) 节点,触碰缓冲地址(原始 FFI 不 emit → 手工租约)
        let mut session = dev.capture_session().unwrap();
        session.lease(&t);
        let (_, graph) = session
            .capture(
                sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
                |frame| {
                    unsafe {
                        sys::cuMemsetD8Async(
                            t.device_ptr() as sys::CUdeviceptr,
                            0,
                            4096,
                            frame.stream.cu_stream(),
                        )
                        .result()
                        .unwrap();
                    }
                    Ok(())
                },
            )
            .unwrap();
        graph.upload().unwrap();

        // 图存活期:令牌有效(强租约)
        assert!(dev.validate_token(&token));

        // 用户侧先 drop(强租约:物理与身份都不动)
        drop(t);
        assert!(dev.validate_token(&token), "租约存活期令牌必须有效");

        // replay:memset(0) 重放,依赖显存必然有效
        graph.launch().unwrap();
        dev.ctx().synchronize().unwrap();

        // 读回:应为 0(replay 生效于幸存显存;survivor 租约克隆提供指针)
        let mut host = [0xAAu8; 4096];
        unsafe {
            sys::cuMemcpyDtoH_v2(
                host.as_mut_ptr() as *mut std::ffi::c_void,
                survivor.device_ptr() as sys::CUdeviceptr,
                4096,
            )
            .result()
            .unwrap();
        }
        assert!(host.iter().all(|&b| b == 0), "replay 应把缓冲清零");

        // 图销毁 → 租约解 → 身份注销;净空窗口归账,账本回基线
        drop(graph);
        drop(survivor);
        assert!(!dev.validate_token(&token), "图销毁后令牌应注销");
        dev.set_phase(MemPhase::Idle); // 净空窗口:延迟回收落地
        assert_eq!(dev.ledger().bytes_alive, 0);
    }

    /// A2.8:VMM 分配(2MiB 粒度,可被对端 P2P 映射的唯一合法路径)
    #[test]
    fn vmm_alloc_granularity_and_ledger() {
    let _g = gpu();
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
    let _g = gpu();
        let dev = make();
        let scratch = scratch_pool(&dev, "s", 1 << 20);
        let weights = persistent_pool(&dev, "w", 1 << 20);
        // Scratch 池上做持久分配 → 拒
        assert!(matches!(
            scratch.alloc_persistent_in::<f32>(16),
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
    let _g = gpu();
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

    /// P1-C:lease-last 路径的 drop 顺序无关性——用户句柄先灭时,
    /// 身份由 keepalive(而非用户句柄)钉住;图销毁后才 retire + 回收。
    /// 旧实现 retire 挂在用户句柄 drop 的 count==1 判定上,本顺序必挂。
    #[test]
    fn lease_survivor_drops_before_graph_token_stays_valid() {
    let _g = gpu();
        let dev = CudaDevice::new(super::test_device_ordinal()).expect("需要 CUDA 设备");
        let pool = dev
            .create_pool(PoolConfig {
                name: "lease-order".into(),
                kind: PoolKind::Weights,
                bytes: 1 << 20,
            })
            .unwrap();
        let t = pool.alloc_persistent_in::<u8>(4096).unwrap();
        let token = t.token.unwrap();

        unsafe {
            sys::cuMemsetD8Async(
                t.device_ptr() as sys::CUdeviceptr,
                0xFF,
                4096,
                dev.stream().cu_stream(),
            )
            .result()
            .unwrap();
        }
        dev.ctx().synchronize().unwrap();
        dev.note_launch();

        let mut session = dev.capture_session().unwrap();
        session.lease(&t);
        let (_, graph) = session
            .capture(
                sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
                |frame| unsafe {
                    sys::cuMemsetD8Async(
                        t.device_ptr() as sys::CUdeviceptr,
                        0,
                        4096,
                        frame.stream.cu_stream(),
                    )
                    .result()
                    .unwrap();
                    Ok(())
                },
            )
            .unwrap();
        graph.upload().unwrap();

        // 用户句柄先灭:keepalive 钉住,令牌必须仍活,replay 仍正确
        drop(t);
        assert!(dev.validate_token(&token), "keepalive 存活期令牌必须有效");
        graph.launch().unwrap();
        dev.ctx().synchronize().unwrap();

        // 图销毁 → keepalive 归零 → PoolBufInner::drop 统一 retire+回收
        drop(graph);
        assert!(!dev.validate_token(&token), "图销毁后令牌应注销(P1-C)");
        dev.set_phase(MemPhase::Idle);
        assert_eq!(dev.ledger().bytes_alive, 0);
    }

    /// 设备隔离律:UUID 钉卡回环 + Backend 枚举
    #[test]
    fn uuid_pinning_roundtrip() {
    let _g = gpu();
        let backend = CudaBackend;
        let devs = backend.enumerate().expect("需要 CUDA 设备");
        assert!(!devs.is_empty());
        let dev = backend.open(&devs[0].uuid).expect("open by uuid");
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
        let buf = pool.alloc_persistent_in::<f32>(64).unwrap();
        assert_eq!(buf.len(), 64);
        drop(buf);
        dev.set_phase(MemPhase::Idle);
    }
}

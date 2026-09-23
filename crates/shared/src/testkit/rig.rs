//! 进程级测试装置:设备 + Weights 池(OnceLock,进程一次)+ 搬运一条龙。
//!
//! 教训内建:
//! - DevBuf 持有缓冲(活性随行;裸指针即刻回收 = 悬空,gdn 测试踩过);
//! - dtoh 前强制 synchronize(非默认流 kernel 写与同步拷贝竞速,
//!   2026-09-23 safetensors flake 与 M-Ⅰ cu_seqlens 清零同根);
//! - 设备序号读 OWL_TEST_DEVICE(与全仓测试纪律一致,避免与生产引擎抢卡)。

use std::sync::Arc;

use owl_cuda::{CudaDevice, CudaPool};
use owl_iface::{Device as _, PoolConfig, PoolKind};

/// 设备 + 池的进程级单例(测试内直接 `Rig::acquire()`)。
pub struct Rig {
    dev: Arc<CudaDevice>,
    pool: Arc<CudaPool>,
    pool_bytes: usize,
}

static RIG: std::sync::OnceLock<Arc<Rig>> = std::sync::OnceLock::new();

impl Rig {
    /// 进程单例(默认池 64 MiB;需更大用 [`Rig::acquire_sized`],二者同进程
    /// 只装一次——先用小的会顶死,框架用户按算子族选一个口径)。
    pub fn acquire() -> Arc<Rig> {
        Self::acquire_sized(64 << 20)
    }

    /// 进程单例(指定池大小;OnceLock 语义 = 首次调用定大小)。
    pub fn acquire_sized(pool_bytes: usize) -> Arc<Rig> {
        RIG
            .get_or_init(|| {
                let dev =
                    Arc::new(CudaDevice::new(owl_cuda::test_device_ordinal()).expect("需要 CUDA 设备"));
                dev.ctx().bind_to_thread().expect("bind_to_thread");
                let pool = Arc::new(
                    dev.create_pool(PoolConfig {
                        name: format!("testkit-{}", std::process::id()),
                        kind: PoolKind::Weights,
                        bytes: pool_bytes as u64,
                    })
                    .expect("testkit 池创建"),
                );
                Arc::new(Rig {
                    dev,
                    pool,
                    pool_bytes,
                })
            })
            .clone()
    }

    pub fn device(&self) -> &CudaDevice {
        &self.dev
    }

    pub fn stream(&self) -> &Arc<owl_cuda::ffi::CudaStream> {
        self.dev.stream()
    }

    pub fn pool_bytes(&self) -> usize {
        self.pool_bytes
    }

    /// H2D(f32;返回持有缓冲)。
    pub fn htod_f32(&self, v: &[f32]) -> DevBuf {
        self.htod(v.to_vec())
    }

    /// H2D(u32)。
    pub fn htod_u32(&self, v: &[u32]) -> DevBuf {
        self.htod(v.to_vec())
    }

    /// H2D(通用;T: MemValue)。
    pub fn htod<T: owl_iface::MemValue>(&self, v: Vec<T>) -> DevBuf {
        let n = v.len();
        let buf = self
            .dev
            .htod_persistent_in(&self.pool, v)
            .expect("testkit htod");
        DevBuf {
            inner: Box::new(DevBufT { buf }),
            n,
            shape: vec![n],
        }
    }

    /// 池内输出缓冲(预清零;f32)。
    pub fn out_f32(&self, n: usize) -> DevBuf {
        self.htod(vec![0f32; n])
    }

    /// 池内输出缓冲(预清零;u32)。
    pub fn out_u32(&self, n: usize) -> DevBuf {
        self.htod(vec![0u32; n])
    }

    /// D2H(DevBuf 全量;内部 synchronize)。
    pub fn dtoh_f32(&self, buf: &DevBuf) -> Vec<f32> {
        self.dtoh_ptr_f32(buf.f32_ptr(), buf.n)
    }

    /// D2H(裸指针版;内部 synchronize)。
    pub fn dtoh_ptr_f32(&self, ptr: *const f32, n: usize) -> Vec<f32> {
        self.sync();
        let mut out = vec![0f32; n];
        unsafe {
            use owl_cuda::ffi::sys;
            sys::cuMemcpyDtoH_v2(
                out.as_mut_ptr() as *mut std::ffi::c_void,
                ptr as sys::CUdeviceptr,
                n * 4,
            )
            .result()
            .expect("testkit dtoh");
        }
        out
    }

    /// D2H(u32 裸指针)。
    pub fn dtoh_ptr_u32(&self, ptr: *const u32, n: usize) -> Vec<u32> {
        self.sync();
        let mut out = vec![0u32; n];
        unsafe {
            use owl_cuda::ffi::sys;
            sys::cuMemcpyDtoH_v2(
                out.as_mut_ptr() as *mut std::ffi::c_void,
                ptr as sys::CUdeviceptr,
                n * 4,
            )
            .result()
            .expect("testkit dtoh u32");
        }
        out
    }

    /// 设备同步(流序屏障;kernel 写 → host 读之间必须)。
    pub fn sync(&self) {
        self.dev.ctx().synchronize().expect("testkit sync");
    }
}

/// 持有缓冲的设备张量句柄(位型视图按需取;活性随行)。
pub struct DevBuf {
    inner: Box<dyn DevBufErased>,
    pub(crate) n: usize,
    pub(crate) shape: Vec<usize>,
}

trait DevBufErased: Send {
    fn ptr_u8(&self) -> *mut u8;
}

struct DevBufT<T: owl_iface::MemValue> {
    buf: owl_cuda::Persistent<T>,
}

impl<T: owl_iface::MemValue> DevBufErased for DevBufT<T> {
    fn ptr_u8(&self) -> *mut u8 {
        owl_iface::DevBuf::<T>::device_ptr(&self.buf) as *mut u8
    }
}

impl DevBuf {
    /// 元素数。
    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// 逻辑形状(声明面;契约检查用)。
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn with_shape(mut self, shape: &[usize]) -> Self {
        assert_eq!(
            shape.iter().product::<usize>(),
            self.n,
            "DevBuf::with_shape: 形状积 != 元素数"
        );
        self.shape = shape.to_vec();
        self
    }

    /// f32 指针视图。
    pub fn f32_ptr(&self) -> *const f32 {
        self.inner.ptr_u8() as *const f32
    }

    pub fn f32_ptr_mut(&self) -> *mut f32 {
        self.inner.ptr_u8() as *mut f32
    }

    /// u8 指针视图(kernel 通配位型入口)。
    pub fn u8_ptr(&self) -> *const u8 {
        self.inner.ptr_u8() as *const u8
    }

    pub fn u8_ptr_mut(&self) -> *mut u8 {
        self.inner.ptr_u8()
    }

    /// u32 指针视图。
    pub fn u32_ptr(&self) -> *const u32 {
        self.inner.ptr_u8() as *const u32
    }

    pub fn u32_ptr_mut(&self) -> *mut u32 {
        self.inner.ptr_u8() as *mut u32
    }
}

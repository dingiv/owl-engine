//! 后端工具层:测试/工具用的语法糖(开设备 → 执行闭包 → 关设备)。
//!
//! 依赖向:owl-backend-utils → { owl-cuda, owl-iface }(rocm 就绪后接入)。
//! 后端开/关设备的全部语法糖都收在本 crate:泛型 [`run_with_device`]
//! (任意 Backend)+ CUDA 零仪式版 [`run_with_cuda`]。

use owl_cuda::{BackendError, CudaDevice};

pub use owl_cuda::{test_device_ordinal, TEST_POOL_BYTES};
use owl_iface::Backend;

/// 泛型语法糖:打开设备 → 执行闭包 → 自动关设备(测试/工具用)。
/// "关" = Device drop:池缓冲全部归还,身份台账清零,无残留显存。
/// 闭包内拿到的 `&B::Device` 与常规用法完全同型,无 Any/向下转型。
/// CUDA 零仪式版见 [`run_with_cuda`](免 UUID)。
pub fn run_with_device<B, F, R>(backend: &B, uuid: &str, f: F) -> Result<R, BackendError>
where
    B: Backend,
    F: FnOnce(&B::Device) -> R,
{
    let dev = backend.open(uuid)?;
    let r = f(&dev);
    drop(dev);
    Ok(r)
}

/// 语法糖:打开 CUDA 设备 → 执行闭包 → 自动关设备(测试/工具用)。
/// 设备序号走 `OWL_TEST_DEVICE`(默认 0),池容量由入参指定;
/// 返回时设备已 drop:池缓冲归还、身份台账清零、显存无残留。
/// 闭包内拿 `&CudaDevice`,与常规用法完全同型。
/// 需要指定 UUID/序号请用 iface 的 `run_with_device` + `CudaBackend`。
///
/// 不传容量时的默认口径 = [`TEST_POOL_BYTES`]。
pub fn run_with_cuda<F, R>(pool_bytes: u64, f: F) -> Result<R, BackendError>
where
    F: FnOnce(&CudaDevice) -> R,
{
    let dev = CudaDevice::new(test_device_ordinal(), pool_bytes)?;
    let r = f(&dev);
    drop(dev);
    Ok(r)
}

#[cfg(test)]
mod tests {
    /// GPU 测试互斥(与 owl-cuda 同纪律:双 context 并发图操作竞态)
    static GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn run_with_cuda_sugar() {
        use owl_iface::{DevBuf, Pool as _};
        let _g = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let r = crate::run_with_cuda(super::TEST_POOL_BYTES, |dev| {
            let pool = dev.default_pool();
            let buf = pool.alloc_persistent_in::<u32>(16).expect("alloc");
            DevBuf::<u32>::len(&buf)
        })
        .expect("run_with_cuda");
        assert_eq!(r, 16);
    }
}

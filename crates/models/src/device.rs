//! 设备抽象:跨设备统一表达的契约(CPU 与 GPU server 同一形状)。
//!
//! 这是 models crate 对"任何设备"的最小期望:
//! - 分配(清零)/ 装载(htod)/ 收割(dtoh)/ 容量查询;
//! - 池块句柄 `Bytes` 由各设备自定义(CPU = Arc 内存;GPU = 池块);
//! - owl-cuda server 将来实现本 trait(经 async 面桥接,见 async-runtime.md)。

use crate::contract::ModelError;
use std::sync::Arc;

/// 设备种类(标识/日志/归因用;路由逻辑不在此)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceKind {
    Cpu,
    Cuda { uuid: String },
}

/// 设备契约:字节世界的分配与搬运。
/// 泛型参数零成本——句柄 `Bytes` 由设备自定义,上层经 Tensor<D> 统一表达。
pub trait Device: Clone + Send + Sync + 'static {
    /// 池块句柄(字节面;不透明)
    type Bytes: Clone + Send + Sync;

    /// 设备种类
    fn kind(&self) -> DeviceKind;

    /// 分配(字节;清零)
    fn alloc(&self, bytes: usize) -> Result<Self::Bytes, ModelError>;

    /// host → device 写入式分配(装载路径)
    fn htod(&self, src: &[u8]) -> Result<Self::Bytes, ModelError>;

    /// device → host 收割(阻塞等待 + 拷贝)
    fn dtoh(&self, b: &Self::Bytes, out: &mut [u8]) -> Result<(), ModelError>;

    /// 块容量(字节;Tensor 装配时的容量守卫)
    fn capacity(&self, b: &Self::Bytes) -> usize;
}

// ============================================================================
// CPU 参考设备(样板即测即用;GPU 版在 owl-cuda 侧实现同一 trait)
// ============================================================================

/// CPU 设备。Bytes = Arc 页内存(支持多 Tensor 共享底仓 + 偏移视图)。
#[derive(Debug, Clone, Default)]
pub struct Cpu;

/// CPU 池块:共享底仓(引用计数),容量即 len
#[derive(Clone)]
pub struct CpuBytes(pub(crate) Arc<Vec<u8>>);

impl Device for Cpu {
    type Bytes = CpuBytes;

    fn kind(&self) -> DeviceKind {
        DeviceKind::Cpu
    }

    fn alloc(&self, bytes: usize) -> Result<Self::Bytes, ModelError> {
        Ok(CpuBytes(Arc::new(vec![0u8; bytes])))
    }

    fn htod(&self, src: &[u8]) -> Result<Self::Bytes, ModelError> {
        Ok(CpuBytes(Arc::new(src.to_vec())))
    }

    fn dtoh(&self, b: &Self::Bytes, out: &mut [u8]) -> Result<(), ModelError> {
        let inner = b.0.as_slice();
        if out.len() > inner.len() {
            return Err(ModelError::Msg(format!(
                "cpu dtoh: 收割 {}B > 块 {}B",
                out.len(),
                inner.len()
            )));
        }
        out.copy_from_slice(&inner[..out.len()]);
        Ok(())
    }

    fn capacity(&self, b: &Self::Bytes) -> usize {
        b.0.len()
    }
}

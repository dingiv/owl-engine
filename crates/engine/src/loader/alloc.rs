//! WeightAllocator:装载目标的抽象(显存 / 内存双指向)。
//!
//! 设计:allocator 是 loader 的构造参数(注入),不是全局单例——
//! 同一个 loader 实例可以装到显存也可以装到内存(测试),
//! 装载逻辑零分支。

use owl_nn::Dtype;
use crate::error::Result;
use owl_iface::Device;

/// host 侧张量(内存 allocator 的产出;测试/传输层通用)。
#[derive(Clone)]
pub struct HostTensor {
    pub shape: Vec<usize>,
    pub dtype: Dtype,
    /// 原始字节(row-major,元素序 = dtype 布局)
    pub data: Vec<u8>,
}

impl HostTensor {
    pub fn elems(&self) -> usize {
        self.shape.iter().product()
    }
    /// 元素数 × 字节宽一致性断言(装载正确性的最后防线)
    pub fn assert_layout(&self) -> Result<()> {
        let expect = self.elems() * self.dtype.size_bytes();
        if self.data.len() != expect {
            crate::bail!(
                "HostTensor 布局不符: data {} B != elems×width {expect} B",
                self.data.len()
            );
        }
        Ok(())
    }
}

/// 装载目标抽象。实现者二选一:
/// - [`DeviceWeightAllocator`]:owl Weights 池(账本化,裁决 5);
/// - [`HostWeightAllocator`]:host Vec(测试)。
pub trait WeightAllocator: Send + Sync {
    /// 分配 shape/dtype 的零化缓冲
    fn alloc(&self, shape: &[usize], dtype: Dtype) -> Result<HostTensor>;
    /// 写入原始字节(row-major;长度必须与 alloc 的布局严格一致)
    fn upload(&self, buf: &mut HostTensor, bytes: &[u8]) -> Result<()>;
    /// 分配 + 写入一步
    fn materialize(&self, shape: &[usize], dtype: Dtype, bytes: &[u8]) -> Result<HostTensor> {
        let mut buf = self.alloc(shape, dtype)?;
        self.upload(&mut buf, bytes)?;
        Ok(buf)
    }
}

/// 显存 allocator:owl Weights 池直连。
///
/// 泛型 over owliface Pool 闭环(与 image.rs 的 to_tensor_f32 同型约束)。
/// 产出经 `TensorPoolOps::from_vec_tensor` 落池——字节流 host 侧组装后
/// 一次 H2D,装载路径零分配出口泄漏。
pub struct DeviceWeightAllocator<'a, P> {
    pool: &'a P,
}

impl<'a, P> DeviceWeightAllocator<'a, P>
where
    P: owl_nn::TensorPoolOps,
    P::Dev: Device,
{
    pub fn new(pool: &'a P) -> Self {
        Self { pool }
    }
}

macro_rules! dev_materialize {
    ($method:ident, $t:ty, $bytes:expr, $n:expr) => {{
        let mut v: Vec<$t> = Vec::with_capacity($n);
        // 字节流按目标端序直拷(row-major;host 与设备同为小端)
        v.resize($n, 0 as $t);
        // SAFETY: $t 全部为 POD(u8/u32/i64/f32),按字节覆写合法
        unsafe {
            std::ptr::copy_nonoverlapping($bytes.as_ptr(), v.as_mut_ptr() as *mut u8, $bytes.len());
        }
        v
    }};
}

impl<'a, P> WeightAllocator for DeviceWeightAllocator<'a, P>
where
    P: owl_nn::TensorPoolOps,
    P::Dev: Device,
{
    fn alloc(&self, _shape: &[usize], _dtype: Dtype) -> Result<HostTensor> {
        // 显存直分不走 HostTensor;materialize 全量覆写,统一走 materialize
        crate::bail!("DeviceWeightAllocator: 直接 alloc 无意义,用 materialize(装载一次成)"
        )
    }

    fn upload(&self, _buf: &mut HostTensor, _bytes: &[u8]) -> Result<()> {
        crate::bail!("DeviceWeightAllocator: upload 无意义,用 materialize")
    }

    fn materialize(&self, shape: &[usize], dtype: Dtype, bytes: &[u8]) -> Result<HostTensor> {
        // 先组装 host 侧校验,再落池(from_vec_tensor 内部做 H2D + 账本)
        let n: usize = shape.iter().product();
        if bytes.len() != n * dtype.size_bytes() {
            crate::bail!(
                "materialize: 字节数 {} != elems×width {}",
                bytes.len(),
                n * dtype.size_bytes()
            );
        }
        match dtype {
            Dtype::F32 => {
                let v = dev_materialize!(x, f32, bytes, n);
                self.pool.from_vec_tensor(shape, v)?;
            }
            Dtype::U32 => {
                let v = dev_materialize!(x, u32, bytes, n);
                self.pool.from_vec_tensor(shape, v)?;
            }
            Dtype::U8 => {
                let v = dev_materialize!(x, u8, bytes, n);
                self.pool.from_vec_tensor(shape, v)?;
            }
            Dtype::I64 => {
                let v = dev_materialize!(x, i64, bytes, n);
                self.pool.from_vec_tensor(shape, v)?;
            }
            _ => {
                crate::bail!("DeviceWeightAllocator: dtype {dtype} 装载路径待回填(F16/BF16 设备侧直拷,运行里程碑)");
            }
        }
        // 显存产物不回传 host 块;返回空 HostTensor 只携带元数据(供调用方登记形状)
        Ok(HostTensor {
            shape: shape.to_vec(),
            dtype,
            data: Vec::new(),
        })
    }
}

/// 内存 allocator:纯 host(测试用;零设备依赖)。
pub struct HostWeightAllocator;

impl WeightAllocator for HostWeightAllocator {
    fn alloc(&self, shape: &[usize], dtype: Dtype) -> Result<HostTensor> {
        let n: usize = shape.iter().product();
        Ok(HostTensor {
            shape: shape.to_vec(),
            dtype,
            data: vec![0u8; n * dtype.size_bytes()],
        })
    }

    fn upload(&self, buf: &mut HostTensor, bytes: &[u8]) -> Result<()> {
        if bytes.len() != buf.data.len() {
            crate::bail!(
                "upload: 字节数 {} != 缓冲 {}",
                bytes.len(),
                buf.data.len()
            );
        }
        buf.data.copy_from_slice(bytes);
        Ok(())
    }
}

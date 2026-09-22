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

macro_rules! dev_materialize {
    ($t:ty, $bytes:expr, $n:expr) => {{
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

/// 显存 allocator:owl Weights 池直连。
///
/// 泛型 over owliface Pool 闭环(与 image.rs 的 to_tensor_f32 同型约束)。
/// 产出经 `TensorPoolOps::from_vec_tensor` 落池——字节流 host 侧组装后
/// 一次 H2D,装载路径零分配出口泄漏。Arc 持有(可被 VarBuilderX 等长寿
/// 结构注入;R3 统一装载口)。
pub struct DeviceWeightAllocator<P> {
    pool: std::sync::Arc<P>,
}

impl<P> DeviceWeightAllocator<P>
where
    P: owl_nn::TensorPoolOps,
    P::Dev: Device,
{
    pub fn new(pool: std::sync::Arc<P>) -> Self {
        Self { pool }
    }

    /// R3:装载到设备并返回擦除句柄(loader→VarBuilderX 统一入口)。
    /// F16/BF16 = f32→位型(RNE)后 from_vec_tensor;返回 DynTensor。
    pub fn materialize_dyn(
        &self,
        shape: &[usize],
        dtype: Dtype,
        bytes: &[u8],
    ) -> Result<owl_nn::DynTensor<P::Dev>> {
        let n: usize = shape.iter().product();
        if bytes.len() != n * dtype.size_bytes() {
            crate::bail!(
                "materialize_dyn: 字节数 {} != elems×width {}",
                bytes.len(),
                n * dtype.size_bytes()
            );
        }
        match dtype {
            Dtype::F32 => {
                let v = dev_materialize!(f32, bytes, n);
                Ok(owl_nn::DynTensor::from_f32(
                    &self.pool.from_vec_tensor::<f32>(shape, v)?,
                ))
            }
            Dtype::F16 => {
                let v: Vec<owl_nn::F16> = bytes
                    .chunks_exact(2)
                    .map(|c| owl_nn::F16(u16::from_le_bytes([c[0], c[1]])))
                    .collect();
                Ok(owl_nn::DynTensor::from_f16(
                    &self.pool.from_vec_tensor::<owl_nn::F16>(shape, v)?,
                ))
            }
            Dtype::BF16 => {
                let v: Vec<owl_nn::Bf16> = bytes
                    .chunks_exact(2)
                    .map(|c| owl_nn::Bf16(u16::from_le_bytes([c[0], c[1]])))
                    .collect();
                Ok(owl_nn::DynTensor::from_bf16(
                    &self.pool.from_vec_tensor::<owl_nn::Bf16>(shape, v)?,
                ))
            }
            Dtype::U32 => {
                let v = dev_materialize!(u32, bytes, n);
                Ok(owl_nn::DynTensor::from_u32(
                    &self.pool.from_vec_tensor::<u32>(shape, v)?,
                ))
            }
            Dtype::U8 => {
                let v = dev_materialize!(u8, bytes, n);
                Ok(owl_nn::DynTensor::from_u8(
                    &self.pool.from_vec_tensor::<u8>(shape, v)?,
                ))
            }
            Dtype::I64 => {
                let v = dev_materialize!(i64, bytes, n);
                Ok(owl_nn::DynTensor::from_i64(
                    &self.pool.from_vec_tensor::<i64>(shape, v)?,
                ))
            }
            _ => {
                crate::bail!("DeviceWeightAllocator: dtype {dtype} 装载路径未开(Q8/F8 = marlin-ffi 路线)")
            }
        }
    }

}

/// f32 host 值 → 目标 dtype 字节流(F16/BF16 = RNE 位型),
/// 供调用方在反解后统一走 [`DeviceWeightAllocator::materialize_dyn`]。
pub fn f32_vec_to_dtype_bytes(dtype: Dtype, f32s: &[f32]) -> Result<Vec<u8>> {
    match dtype {
        Dtype::F32 => Ok(f32s.iter().flat_map(|f| f.to_le_bytes()).collect()),
        Dtype::BF16 => Ok(f32s
            .iter()
            .flat_map(|&f| owl_nn::Bf16(f32_to_bf16_rne(f)).0.to_le_bytes())
            .collect()),
        Dtype::F16 => Ok(f32s
            .iter()
            .flat_map(|&f| owl_nn::F16(f32_to_f16_rne(f)).0.to_le_bytes())
            .collect()),
        _ => crate::bail!("f32_vec_to_dtype_bytes: 目标 {dtype} 非 float 家族"),
    }
}

/// f32 → BF16(IEEE 模型位型,RNE 舍入)
fn f32_to_bf16_rne(f: f32) -> u16 {
    let b = f.to_bits();
    let lsb = (b >> 16) & 1;
    ((b + 0x7FFF + lsb) >> 16) as u16
}

/// f32 → F16(IEEE 754 half,RNE 舍入;含非规格化与溢出饱和)
fn f32_to_f16_rne(f: f32) -> u16 {
    let b = f.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xFF) as i32;
    let frac = b & 0x007F_FFFF;
    if exp == 0xFF {
        // Inf/NaN
        return sign | 0x7C00 | if frac != 0 { 0x0200 } else { 0 };
    }
    // 重中心化到 half 域(e = exp - 127 + 15)
    let e = exp - 127 + 15;
    if e >= 0x1F {
        return sign | 0x7C00; // 溢出 → Inf
    }
    if e <= 0 {
        // 非规格化(half):右移 + RNE;过深 → 零
        if e < -10 {
            return sign;
        }
        let frac16 = frac | 0x0080_0000; // 补隐含位
        let shift = (14 - e) as u32;
        let mut m = frac16 >> shift;
        let rem = frac16 & ((1u32 << shift) - 1);
        let half = 1u32 << (shift - 1);
        if rem > half || (rem == half && (m & 1) == 1) {
            m += 1;
        }
        return sign | m as u16;
    }
    // 规格化:frac10 = frac >> 13 + RNE
    let mut m = frac >> 13;
    let rem = frac & 0x1FFF;
    if rem > 0x1000 || (rem == 0x1000 && (m & 1) == 1) {
        m += 1;
        if m == 0x0400 {
            return sign | (((e + 1) as u16) << 10);
        }
    }
    sign | ((e as u16) << 10) | m as u16
}

impl<P> WeightAllocator for DeviceWeightAllocator<P>
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
        // R3 统一装载口:委托 materialize_dyn(真句柄在此);
        // trait 面只回元数据(与历史契约一致)
        self.materialize_dyn(shape, dtype, bytes)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use owl_cuda::test_device_ordinal;
    use owl_iface::{Device, PoolConfig, PoolKind};

    /// R3:F16/BF16 臂真机往返(f16/bf16 精确可表示值,位型应无损)
    #[test]
    fn materialize_dyn_f16_bf16_roundtrip() {
        let dev = owl_cuda::CudaDevice::new(test_device_ordinal()).expect("需要 CUDA 设备");
        let pool = dev
            .create_pool(PoolConfig {
                name: format!("alloc-dyn-t-{}", std::process::id()),
                kind: PoolKind::Weights,
                bytes: 1 << 20,
            })
            .unwrap();
        let alloc = DeviceWeightAllocator::new(std::sync::Arc::new(pool));

        // 全部为 F16/BF16 精确可表示值 → 位型往返应零误差
        let f32s = vec![1.0f32, -2.0, 3.5, 0.25];
        let to_f32 = |dtype: Dtype, b: u16| -> f32 {
            match dtype {
                Dtype::BF16 => f32::from_bits((b as u32) << 16),
                _ => {
                    // F16 位型 → f32:符号/指数/尾数重中心化(测试值精确路径)
                    let sign = ((b & 0x8000) as u32) << 16;
                    let exp = ((b & 0x7C00) >> 10) as u32;
                    let man = (b & 0x03FF) as u32;
                    let bits = if exp == 0 {
                        sign | (man << 13)
                    } else if exp == 0x1F {
                        sign | 0x7F80_0000 | (man << 13)
                    } else {
                        sign | ((((exp as i32) - 15 + 127) as u32) << 23) | (man << 13)
                    };
                    f32::from_bits(bits)
                }
            }
        };
        for dtype in [Dtype::F16, Dtype::BF16] {
            let bytes = f32_vec_to_dtype_bytes(dtype, &f32s).unwrap();
            assert_eq!(bytes.len(), 8);
            let dt = alloc.materialize_dyn(&[2, 2], dtype, &bytes).unwrap();
            assert_eq!(dt.dtype(), dtype);
            assert_eq!(dt.shape(), &[2, 2]);
            let got: Vec<f32> = match dtype {
                Dtype::F16 => dt
                    .typed_f16()
                    .unwrap()
                    .to_vec()
                    .unwrap()
                    .into_iter()
                    .map(|h| to_f32(dtype, h.0))
                    .collect(),
                _ => dt
                    .typed_bf16()
                    .unwrap()
                    .to_vec()
                    .unwrap()
                    .into_iter()
                    .map(|h| to_f32(dtype, h.0))
                    .collect(),
            };
            for (g, &w) in got.iter().zip(&f32s) {
                assert!((g - w).abs() < 1e-6, "{dtype}: got {g} want {w}");
            }
        }
    }
}

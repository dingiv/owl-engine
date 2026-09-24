//! Kernel 值:核函数的身份证 + 源码(模型层直接导入;无 kernel 表)。
//!
//! - `name`:kernel 入口名(后端编译缓存的键成分);
//! - `source`:.cu 源码(include_str 内嵌;模型层旁)。
//! 后端懒注册:归约遇到携带 Kernel 的节点时,按源码哈希查缓存,
//! 未编译过 → load_kernel(ptx, name) → KernelId;随后 LaunchMsg 发射。

#[derive(Clone, Debug)]
pub struct Kernel {
    pub name: &'static str,
    pub source: &'static str,
}

impl Kernel {
    pub fn new(name: &'static str, source: &'static str) -> Self {
        Self { name, source }
    }
}

/// 标量参数(位型打包;类型由 kernel 签名约定,server 解包按序对位)
#[derive(Clone, Copy, Debug)]
pub struct Scalar(pub u64);

impl From<f32> for Scalar {
    fn from(v: f32) -> Self {
        Scalar(v.to_bits() as u64)
    }
}
impl From<usize> for Scalar {
    fn from(v: usize) -> Self {
        Scalar(v as u64)
    }
}
impl From<u32> for Scalar {
    fn from(v: u32) -> Self {
        Scalar(v as u64)
    }
}
impl From<i32> for Scalar {
    fn from(v: i32) -> Self {
        Scalar(v as u64)
    }
}
impl From<bool> for Scalar {
    fn from(v: bool) -> Self {
        Scalar(v as u64)
    }
}

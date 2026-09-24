//! Kernel 值:核函数的身份证 + 源码(模型层直接导入;无 kernel 表)。
//!
//! - `name`:kernel 入口名(后端编译缓存的键成分);
//! - `source`:.cu 源码(include_str 内嵌;模型层旁)。
//! - `grid` / `block` / `shared_mem`:发射配置(grid = (0,0,0) 哨兵 =
//!   解释层按输出元素数自动取 1D ceil/256)。
//!
//! 后端懒注册:解释层 lower 为 LaunchMsg,server 按 (name, source) 懒编译
//! (nvrtc → PTX → module),缓存后逐次直发。

/// 发射配置(grid 哨兵 (0,0,0) = 自动 1D)
#[derive(Clone, Debug)]
pub struct LaunchShape {
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub shared_mem: u32,
}

impl Default for LaunchShape {
    fn default() -> Self {
        Self { grid: (0, 0, 0), block: (256, 1, 1), shared_mem: 0 }
    }
}

#[derive(Clone, Debug)]
pub struct Kernel {
    pub name: &'static str,
    pub source: &'static str,
    pub launch: LaunchShape,
}

impl Kernel {
    pub fn new(name: &'static str, source: &'static str) -> Self {
        Self { name, source, launch: LaunchShape::default() }
    }

    /// 显式发射配置(覆盖自动 1D)
    pub fn with_launch(mut self, grid: (u32, u32, u32), block: (u32, u32, u32), shared_mem: u32) -> Self {
        self.launch = LaunchShape { grid, block, shared_mem };
        self
    }
}

/// 标量参数(位型打包;类型由 kernel 签名约定,server 解包按序对位)
///
/// 注:预定义算子动作表已改用类型化 `client::Arg`(I32/F32/U64,与
/// kernel 形参宽度严格对位);`Scalar` 仅服务 Kernel 节点 Bits 槽的
/// 便利构造,异宽形参的 kernel 请将来扩 KernelArg 类型化变体。
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

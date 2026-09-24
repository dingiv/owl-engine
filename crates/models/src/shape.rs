//! 标注词汇:dtype + shape。client 侧元数据;server(字节世界)不感知。

/// 数据类型(S4 语义表的 dtype 维;起步面,按需扩)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F32,
    BF16,
    F16,
    U32,
}

impl Dtype {
    /// 字节宽(server 的 Malloc 只认字节;宽是 client 侧换算用的)
    pub fn size_bytes(self) -> usize {
        match self {
            Dtype::F32 => 4,
            Dtype::BF16 | Dtype::F16 => 2,
            Dtype::U32 => 4,
        }
    }
}

/// 形状(行主序;一维 = vec![n])
pub type Shape = Vec<usize>;

/// 元素总数
pub fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

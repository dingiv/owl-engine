//! 维度索引类型:candle `D::Minus1` 负维枚举的直译(27 处触点)。
//!
//! 搬运规则:`Dim::From(3)` ↔ candle `3`;`Dim::From(-1)` ↔ candle
//! `D::Minus1`。[`Dim::resolve`] 在调用点归一化为正索引(引擎内
//! 统一禁止负数穿透到算子层)。

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dim(i32);

impl Dim {
    /// 归一化:负数按 rank 回绕(candle 语义:`-1` = 最后一维)。
    pub fn resolve(self, rank: usize) -> Result<usize, crate::error::Error> {
        let r = rank as i32;
        let d = self.0;
        let resolved = if d < 0 { r + d } else { d };
        if resolved < 0 || resolved >= r {
            crate::bail!("dim {d} 越界(rank={rank})");
        }
        Ok(resolved as usize)
    }
}

impl From<usize> for Dim {
    fn from(v: usize) -> Self {
        Dim(v as i32)
    }
}

impl From<i32> for Dim {
    fn from(v: i32) -> Self {
        Dim(v)
    }
}

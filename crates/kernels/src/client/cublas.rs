//! cublas GEMM 解析面(registry 服务端面;models 调用点暂走 lower_kernel
//! sig 路径——cublas 家族零事故面,面成熟度按需补)。
//!
//! 线格式:`[T a, T b, T out, sz m, sz k, sz n, sz nt]`(3 Block + 4 sz;
//! a/b/out 允许 BlockSlice —— linear 行切片直喂)。nt 非零 = bf16 三参。

use crate::contract::{Arg, LaunchMsg, OpError};

/// 块引用(id + 字节偏移;BlockSlice 兼容)
#[derive(Clone, Copy, Debug)]
pub struct BlockRef {
    pub id: u64,
    pub byte_offset: u64,
}

pub const GEMM_F16: &str = crate::family::cublas::GEMM_F16;
pub const GEMM_BF16: &str = crate::family::cublas::GEMM_BF16;
#[derive(Clone, Copy, Debug)]
pub struct GemmCall {
    pub a: BlockRef,
    pub b: BlockRef,
    pub out: BlockRef,
    pub m: usize,
    pub k: usize,
    pub n: usize,
    /// cublas 转置标志(gemm_f16/bf16 的 nt 形参;非变体选择!)
    pub nt: bool,
}

/// 解析(3 Block + 4 sz;F16/BF16 两名同构)
pub fn parse_gemm(msg: &LaunchMsg) -> Result<GemmCall, OpError> {
    let mut blocks: Vec<BlockRef> = Vec::new();
    let mut scalars: Vec<u64> = Vec::new();
    for a in &msg.args {
        match a {
            Arg::Block { id } => blocks.push(BlockRef { id: *id, byte_offset: 0 }),
            Arg::BlockSlice { id, byte_offset, .. } => {
                blocks.push(BlockRef { id: *id, byte_offset: *byte_offset })
            }
            Arg::U64(v) => scalars.push(*v),
            _ => {
                return Err(OpError::Contract {
                    op: msg.kernel.name.clone(),
                    field: "args",
                    expect: "Block/BlockSlice/U64".into(),
                    got: "其他".into(),
                })
            }
        }
    }
    if blocks.len() != 3 || scalars.len() != 4 {
        return Err(OpError::Contract {
            op: msg.kernel.name.clone(),
            field: "槽序",
            expect: "3 Block + 4 sz".into(),
            got: format!("{}B/{}S", blocks.len(), scalars.len()),
        });
    }
    let [a, b, out] = [blocks[0], blocks[1], blocks[2]];
    Ok(GemmCall {
        a,
        b,
        out,
        m: scalars[0] as usize,
        k: scalars[1] as usize,
        n: scalars[2] as usize,
        nt: scalars[3] != 0,
    })
}

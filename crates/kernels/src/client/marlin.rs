//! Marlin W4A16 解析面(registry 服务端面;三名同构:f16 / AWQ / bf16)。
//!
//! 线格式:`[T a, T b(qpack), T out, T scales, T ws, T c_tmp,
//! sz m, sz k, sz n, sz groupsize]`(6 Block + 4 sz;BlockSlice 兼容)。

use crate::contract::{Arg, LaunchMsg, OpError};

pub use crate::family::marlin::GEMM_W4A16;
pub use crate::family::marlin::GEMM_W4A16_AWQ;
pub use crate::family::marlin::GEMM_W4A16_BF16;

/// 槽序签名(f16/bf16 通用臂;6 Block + 4 sz,O = 输出槽。
/// 槽序单源 —— models 调用点经本常量声明,禁止手撸 —— 2026-10-12 review H 案)
pub const SIG: &str = "T,T,O,T,T,T,sz,sz,sz,sz";

/// AWQ 臂签名(kU4 has_zp:scales 后插 zeros,7 Block + 4 sz)
pub const SIG_AWQ: &str = "T,T,O,T,T,T,T,sz,sz,sz,sz";

/// 块引用(id + 字节偏移;BlockSlice 兼容;与 cublas 面同构)
#[derive(Clone, Copy, Debug)]
pub struct BlockRef {
    pub id: u64,
    pub byte_offset: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct MarlinCall {
    pub a: BlockRef,
    pub b: BlockRef,
    pub out: BlockRef,
    pub scales: BlockRef,
    /// AWQ zeros 槽(kU4 has_zp;仅 AWQ 臂)
    pub zeros: Option<BlockRef>,
    /// workspace(i32;marlin 排队工作区)
    pub ws: BlockRef,
    /// c_tmp(FFI 透传,内核不解引用的依赖边)
    pub c_tmp: BlockRef,
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub groupsize: i32,
    /// bf16 三族实例(E5-DF3)
    pub bf16: bool,
}

/// 解析(f16/bf16:6 Block + 4 sz;bf16 标志由 registry 按名字路由)
pub fn parse_gemm(msg: &LaunchMsg, bf16: bool) -> Result<MarlinCall, OpError> {
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
    if blocks.len() != 6 || scalars.len() != 4 {
        return Err(OpError::Contract {
            op: msg.kernel.name.clone(),
            field: "槽序",
            expect: "6 Block + 4 sz".into(),
            got: format!("{}B/{}S", blocks.len(), scalars.len()),
        });
    }
    let [a, b, out, scales, ws, c_tmp] = [blocks[0], blocks[1], blocks[2], blocks[3], blocks[4], blocks[5]];
    Ok(MarlinCall {
        a,
        b,
        out,
        scales,
        zeros: None,
        ws,
        c_tmp,
        m: scalars[0] as usize,
        k: scalars[1] as usize,
        n: scalars[2] as usize,
        groupsize: scalars[3] as i32,
        bf16,
    })
}

/// AWQ 臂解析(kU4 has_zp:**7 Block** = scales 后插 zeros 槽 + 4 sz;
/// FFI 实参序 = scales(3), zeros(4), ws(5), c_tmp(6))
pub fn parse_gemm_awq(msg: &LaunchMsg) -> Result<MarlinCall, OpError> {
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
    if blocks.len() != 7 || scalars.len() != 4 {
        return Err(OpError::Contract {
            op: msg.kernel.name.clone(),
            field: "槽序",
            expect: "7 Block + 4 sz(AWQ:scales 后插 zeros)".into(),
            got: format!("{}B/{}S", blocks.len(), scalars.len()),
        });
    }
    let [a, b, out, scales, zeros, ws, c_tmp] =
        [blocks[0], blocks[1], blocks[2], blocks[3], blocks[4], blocks[5], blocks[6]];
    Ok(MarlinCall {
        a,
        b,
        out,
        scales,
        zeros: Some(zeros),
        ws,
        c_tmp,
        m: scalars[0] as usize,
        k: scalars[1] as usize,
        n: scalars[2] as usize,
        groupsize: scalars[3] as i32,
        bf16: false,
    })
}

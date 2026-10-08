//! FlashInfer paged prefill 解析面(registry 服务端面)。
//!
//! 线格式(handler 权威;8 sz):`[T q, T kc_fi, T vc, T q_cu, T indices,
//! T indptr, T last_len, T wr, O out, sz total_rows, sz ctx_total, sz T,
//! sz hq, sz hkv, sz hd, sz page, sz sm_scale_bits]`(9 Block + 8 sz;
//! wr = 依赖边;sm_scale 以 f32 位型过线,server 面 from_bits 还原;
//! nb 由页表推导,非线格式字段)。

use crate::contract::{Arg, Bytes, LaunchMsg, OpError};

pub const PREFILL_FI: &str = crate::family::flashinfer::PREFILL_FI;
pub const PREFILL_FI_FP8KV: &str = crate::family::flashinfer::PREFILL_FI_FP8KV;

#[derive(Clone, Copy, Debug)]
pub struct BlockRef {
    pub id: u64,
    pub byte_offset: u64,
}

#[derive(Clone, Debug)]
pub struct FiPrefillCall {
    pub q: BlockRef,
    pub kc_fi: BlockRef,
    pub vc: BlockRef,
    pub q_cu: BlockRef,
    pub indices: BlockRef,
    pub indptr: BlockRef,
    pub last_len: BlockRef,
    /// 依赖边(树序;FI 不解引用)
    pub wr: BlockRef,
    pub out: Bytes,
    pub total_rows: usize,
    pub ctx_total: usize,
    pub t: usize,
    pub hq: usize,
    pub hkv: usize,
    pub hd: usize,
    pub page: usize,
    pub sm_scale: f32,
    /// fp8kv 变体(按名字路由)
    pub fp8kv: bool,
}

/// 解析(9 Block + 8 sz)
pub fn parse_prefill(msg: &LaunchMsg) -> Result<FiPrefillCall, OpError> {
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
    if blocks.len() != 9 || scalars.len() != 8 {
        return Err(OpError::Contract {
            op: msg.kernel.name.clone(),
            field: "槽序",
            expect: "9 Block + 8 sz".into(),
            got: format!("{}B/{}S", blocks.len(), scalars.len()),
        });
    }
    let q = blocks[0];
    let kc_fi = blocks[1];
    let vc = blocks[2];
    let q_cu = blocks[3];
    let indices = blocks[4];
    let indptr = blocks[5];
    let last_len = blocks[6];
    let wr = blocks[7];
    Ok(FiPrefillCall {
        q,
        kc_fi,
        vc,
        q_cu,
        indices,
        indptr,
        last_len,
        wr,
        out: Bytes { id: blocks[8].id, len: 0 },
        total_rows: scalars[0] as usize,
        ctx_total: scalars[1] as usize,
        t: scalars[2] as usize,
        hq: scalars[3] as usize,
        hkv: scalars[4] as usize,
        hd: scalars[5] as usize,
        page: scalars[6] as usize,
        sm_scale: f32::from_bits(scalars[7] as u32),
        fp8kv: msg.kernel.name == PREFILL_FI_FP8KV,
    })
}

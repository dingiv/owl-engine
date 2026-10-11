//! XQA verify/decode 解析面(registry 服务端面)。
//!
//! 线格式(handler 权威;8 Block + 10 sz,见 family::xqa::VERIFY_XQA_SLOTS):
//! `[T q, T k_cache, T v_cache, T page_table(i32), T seq_lens(u32),
//!   T mask(u32), T semaphores(u32), T scratch, T wr, O out, sz q_seq_len, sz hkv, sz hd,
//!   sz page, sz max_seq_len, sz q_scale_bits, sz sm_count, sz stride_page,
//!   sz stride_token, sz stride_head]`
//!
//! ⚠️ semaphores 必须**每次发射前为零**(KV-split 归约计数;参考实现
//! `sem.zero_()`)。图内路线 = 图里挂一个零写节点(引擎接线时落;本步
//! 层测试由调用方直接给零张量)。

use crate::contract::{Arg, Bytes, LaunchMsg, OpError};

pub const VERIFY_XQA: &str = crate::family::xqa::VERIFY_XQA;

#[derive(Clone, Copy, Debug)]
pub struct BlockRef {
    pub id: u64,
    pub byte_offset: u64,
}

#[derive(Clone, Debug)]
pub struct XqaVerifyCall {
    pub q: BlockRef,
    pub out: Bytes,
    pub k_cache: BlockRef,
    pub v_cache: BlockRef,
    pub page_table: BlockRef,
    pub seq_lens: BlockRef,
    pub mask: BlockRef,
    pub semaphores: BlockRef,
    pub scratch: BlockRef,
    /// 依赖边(树序;K0 写池先于 XQA 读池 —— XQA 不解引用,仅保序)
    pub wr: BlockRef,
    pub q_seq_len: usize,
    pub hkv: usize,
    pub hd: usize,
    pub page: usize,
    pub max_seq_len: usize,
    pub q_scale: f32,
    pub sm_count: usize,
    pub stride_page: usize,
    pub stride_token: usize,
    pub stride_head: usize,
}

pub fn parse_verify(msg: &LaunchMsg) -> Result<XqaVerifyCall, OpError> {
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
                    expect: "Block/BlockSlice/U64(9 Block + 10 sz)".into(),
                    got: format!("{a:?}"),
                })
            }
        }
    }
    let need = (10, 10);
    if blocks.len() != need.0 || scalars.len() != need.1 {
        return Err(OpError::Contract {
            op: msg.kernel.name.clone(),
            field: "argc",
            expect: format!("{} Block + {} sz", need.0, need.1),
            got: format!("{} Block + {} sz", blocks.len(), scalars.len()),
        });
    }
    Ok(XqaVerifyCall {
        q: blocks[0],
        // 槽序:[q, k, v, page_table, seq_lens, mask, semaphores, scratch, wr]
        //      + out(解释器追加)
        out: Bytes { id: blocks[9].id, len: 0 },
        k_cache: blocks[1],
        v_cache: blocks[2],
        page_table: blocks[3],
        seq_lens: blocks[4],
        mask: blocks[5],
        semaphores: blocks[6],
        scratch: blocks[7],
        wr: blocks[8],
        q_seq_len: scalars[0] as usize,
        hkv: scalars[1] as usize,
        hd: scalars[2] as usize,
        page: scalars[3] as usize,
        max_seq_len: scalars[4] as usize,
        q_scale: f32::from_bits(scalars[5] as u32),
        sm_count: scalars[6] as usize,
        stride_page: scalars[7] as usize,
        stride_token: scalars[8] as usize,
        stride_head: scalars[9] as usize,
    })
}

/// q_scale 位型过线(层侧 builder 与 server 侧还原共用一个口径)
pub fn q_scale_bits(attn_scale: f32, hd: usize) -> u64 {
    (attn_scale * (hd as f32).sqrt()).to_bits() as u64
}

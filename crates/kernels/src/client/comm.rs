//! 通信算子族客户端面 —— allreduce stage/reduce(TP2 S2;models 视角)。
//!
//! 线格式(与 `ops::comm` 声明表同源;按名取址 —— F3 成对律):
//!
//! ```text
//! stage : [T local, T own, T peer, sz bytes, sz slot_bytes, sz slab_off]
//! reduce: [T out,   T own, T peer, sz bytes, sz slot_bytes, sz slab_off]
//! ```
//!
//! **发射名 = 声明表 entry(三 dtype 三张表)**:`owl_comm_ar_stage_{f32,f16,bf16}`
//! —— `NamedSlots` 按 `kernel.name` 回声明表取址,故名字必须**逐字**是
//! 声明 entry(族主名 `comm_ar_stage` 不是发射名,它是路由 id 的部分)。
//!
//! 语义:
//! - **stage** = 本端 payload(`local`)第 `slab_off` 起 `bytes` 字节 → 本端
//!   staging 的 slot[produced % n_slots],系统 fence 后发布 produced++;
//! - **reduce** = 本端 payload(`out`)第 `slab_off` 起 `bytes` 字节**原地累加**
//!   对端 staging 的 slot[consumed % n_slots],随后发布 consumed++。
//!
//! `slot_bytes` = staging slot 大小(comm::ar::StagingLayout 的布局值,两侧
//! 必须同参);第 i 块的 `slab_off = i × slot_bytes`。

use crate::contract::{Arg, Bytes, Dtype, LaunchMsg, OpError};

/// stage 臂的族名(server 路由 id;⚠️ **不是发射名** —— 发射名 = 声明 entry)
pub const AR_STAGE: &str = crate::family::comm::COMM_AR_STAGE;
/// reduce 臂的族名
pub const AR_REDUCE: &str = crate::family::comm::COMM_AR_REDUCE;

/// stage 线格式 sig(声明表派生 —— `T,O,T,sz,sz,sz`;与 `ops::comm` 同票,
/// `client::tests::parse_slots_are_exactly_declared` 与 ops 槽序锁双锁)
pub const SIG: &str = "T,O,T,sz,sz,sz";
/// reduce 线格式 sig(`O,T,T,sz,sz,sz`)
pub const SIG_REDUCE: &str = "O,T,T,sz,sz,sz";

/// payload dtype → stage 发射名(**单源在 `family::comm`** —— driver 同读)
///
/// 两个 dtype 词汇(contract vs driver)的桥在此显式;非法 dtype **响亮
/// panic**(发射名不是可以静默降级的东西 —— 与 driver 其他档同律)。
fn stage_entry(dt: Dtype) -> &'static str {
    use crate::driver::DType as D;
    match dt {
        Dtype::F32 => crate::family::comm::stage_entry(D::F32),
        Dtype::F16 => crate::family::comm::stage_entry(D::F16),
        Dtype::BF16 => crate::family::comm::stage_entry(D::BF16),
        other => panic!("comm_ar_stage: 非法 payload dtype {other:?}(f32/f16/bf16)"),
    }
}

/// payload dtype → reduce 发射名(单源同上)
fn reduce_entry(dt: Dtype) -> &'static str {
    use crate::driver::DType as D;
    match dt {
        Dtype::F32 => crate::family::comm::reduce_entry(D::F32),
        Dtype::F16 => crate::family::comm::reduce_entry(D::F16),
        Dtype::BF16 => crate::family::comm::reduce_entry(D::BF16),
        other => panic!("comm_ar_reduce: 非法 payload dtype {other:?}(f32/f16/bf16)"),
    }
}

/// stage 的 parse 槽名(声明序;F3 成对律 —— `client::tests::
/// parse_slots_are_exactly_declared` 与 `ops` 的槽序锁双双锁死)
pub const PARSE_SLOTS: &[&str] = &["local", "own", "peer", "bytes", "slot_bytes", "slab_off"];
/// reduce 的 parse 槽名(声明序)
pub const PARSE_SLOTS_REDUCE: &[&str] = &["out", "own", "peer", "bytes", "slot_bytes", "slab_off"];

/// stage 调用(解析/组装双向同型)。
#[derive(Clone, Debug)]
pub struct ArStageCall {
    local: Bytes,
    own: Bytes,
    peer: Bytes,
    bytes: u64,
    slot_bytes: u64,
    slab_off: u64,
}

impl ArStageCall {
    /// 直接构造(调用方已备好三块:payload / 本端 staging / 对端别名)。
    pub fn new(
        local: Bytes,
        own: Bytes,
        peer: Bytes,
        bytes: u64,
        slot_bytes: u64,
        slab_off: u64,
    ) -> Self {
        Self { local, own, peer, bytes, slot_bytes, slab_off }
    }

    /// 反序列化(6 槽;按名取址)。
    pub fn parse(msg: &LaunchMsg) -> Result<(Self, Bytes), OpError> {
        let s = crate::client::NamedSlots::new(msg)?;
        let local = s.flat_block("local")?;
        // stage 的输出槽 = own(被写入的 staging;`out_elems` 由调用方按字节口径给)
        let own = s.out_block("own", msg.out_elems)?;
        let peer = s.flat_block("peer")?;
        let me = Self {
            local,
            own: own.clone(),
            peer,
            bytes: s.u64("bytes")?,
            slot_bytes: s.u64("slot_bytes")?,
            slab_off: s.u64("slab_off")?,
        };
        Ok((me, own))
    }

    /// 组装发射消息(几何由 runtime 自持 ⇒ grid/block 全零哨兵;名字按 dtype 选 entry)
    pub fn to_launch(&self) -> LaunchMsg {
        LaunchMsg {
            kernel: crate::contract::KernelSource {
                name: stage_entry(self.local.dtype).to_string(),
                source: String::new(),
            },
            args: vec![
                Arg::block(self.local.id, self.local.dtype, self.local.rank),
                Arg::block(self.own.id, self.own.dtype, self.own.rank),
                Arg::block(self.peer.id, self.peer.dtype, self.peer.rank),
                Arg::U64(self.bytes),
                Arg::U64(self.slot_bytes),
                Arg::U64(self.slab_off),
            ],
            grid: (0, 0, 0),
            block: (0, 0, 0),
            shared_mem: 0,
            out_elems: self.bytes as usize,
        }
    }

    pub fn local(&self) -> &Bytes {
        &self.local
    }
    pub fn own(&self) -> &Bytes {
        &self.own
    }
    pub fn peer(&self) -> &Bytes {
        &self.peer
    }
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
    pub fn slot_bytes(&self) -> u64 {
        self.slot_bytes
    }
    pub fn slab_off(&self) -> u64 {
        self.slab_off
    }
}

/// reduce 调用(解析/组装双向同型)。
#[derive(Clone, Debug)]
pub struct ArReduceCall {
    out: Bytes,
    own: Bytes,
    peer: Bytes,
    bytes: u64,
    slot_bytes: u64,
    slab_off: u64,
}

impl ArReduceCall {
    pub fn new(
        out: Bytes,
        own: Bytes,
        peer: Bytes,
        bytes: u64,
        slot_bytes: u64,
        slab_off: u64,
    ) -> Self {
        Self { out, own, peer, bytes, slot_bytes, slab_off }
    }

    pub fn parse(msg: &LaunchMsg) -> Result<(Self, Bytes), OpError> {
        let s = crate::client::NamedSlots::new(msg)?;
        let out = s.out_block("out", msg.out_elems)?;
        let own = s.flat_block("own")?;
        let peer = s.flat_block("peer")?;
        let me = Self {
            out: out.clone(),
            own,
            peer,
            bytes: s.u64("bytes")?,
            slot_bytes: s.u64("slot_bytes")?,
            slab_off: s.u64("slab_off")?,
        };
        Ok((me, out))
    }

    pub fn to_launch(&self) -> LaunchMsg {
        LaunchMsg {
            kernel: crate::contract::KernelSource {
                name: reduce_entry(self.out.dtype).to_string(),
                source: String::new(),
            },
            args: vec![
                Arg::block(self.out.id, self.out.dtype, self.out.rank),
                Arg::block(self.own.id, self.own.dtype, self.own.rank),
                Arg::block(self.peer.id, self.peer.dtype, self.peer.rank),
                Arg::U64(self.bytes),
                Arg::U64(self.slot_bytes),
                Arg::U64(self.slab_off),
            ],
            grid: (0, 0, 0),
            block: (0, 0, 0),
            shared_mem: 0,
            out_elems: self.bytes as usize,
        }
    }

    pub fn out(&self) -> &Bytes {
        &self.out
    }
    pub fn own(&self) -> &Bytes {
        &self.own
    }
    pub fn peer(&self) -> &Bytes {
        &self.peer
    }
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
    pub fn slot_bytes(&self) -> u64 {
        self.slot_bytes
    }
    pub fn slab_off(&self) -> u64 {
        self.slab_off
    }
}

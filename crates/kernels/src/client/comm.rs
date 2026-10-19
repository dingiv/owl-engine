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
fn stage_entry_dt(dt: Dtype, f8: bool) -> &'static str {
    use crate::driver::DType as D;
    let d = match dt {
        Dtype::F32 => D::F32,
        Dtype::F16 => D::F16,
        Dtype::BF16 => D::BF16,
        other => panic!("comm_ar_stage: 非法 payload dtype {other:?}(f32/f16/bf16)"),
    };
    crate::family::comm::stage_entry_f8(d, f8)
}

/// payload dtype → reduce 发射名(单源同上)
fn reduce_entry_dt(dt: Dtype, f8: bool) -> &'static str {
    use crate::driver::DType as D;
    let d = match dt {
        Dtype::F32 => D::F32,
        Dtype::F16 => D::F16,
        Dtype::BF16 => D::BF16,
        other => panic!("comm_ar_reduce: 非法 payload dtype {other:?}(f32/f16/bf16)"),
    };
    crate::family::comm::reduce_entry_f8(d, f8)
}

/// payload dtype → **1-stage** 发射名(单源同上;2026-10-18 §35)
fn one_stage_entry(dt: Dtype) -> &'static str {
    use crate::driver::DType as D;
    match dt {
        Dtype::F32 => crate::family::comm::one_stage_entry(D::F32),
        Dtype::F16 => crate::family::comm::one_stage_entry(D::F16),
        Dtype::BF16 => crate::family::comm::one_stage_entry(D::BF16),
        other => panic!("comm_ar_1stage: 非法 payload dtype {other:?}(f32/f16/bf16)"),
    }
}

/// stage 的 parse 槽名(声明序;F3 成对律 —— `client::tests::
/// parse_slots_are_exactly_declared` 与 `ops` 的槽序锁双双锁死)
///
/// ⚠️ **push 档(§36.7)第 4 槽 = `data`**(slot 数据写哪里:v1 = `own` / push = `peer`)。
/// 2026-10-19 修复:上一轮只给 `ArStageCall` 加了 `data` 字段与 `parse`/`to_launch`,
/// **漏了这张常量表** ⇒ `cargo test -p owl-kernels --lib` 两条槽序锁红(F3 铁律现场:
/// 真源有两处就会漂 —— 两把锁存在的理由就是把这类漏跟当场抹出来)。
pub const PARSE_SLOTS: &[&str] = &["local", "own", "peer", "data", "bytes", "slot_bytes", "slab_off"];
/// reduce 的 parse 槽名(声明序;push 档同律 —— 第 4 槽 = `data`)
pub const PARSE_SLOTS_REDUCE: &[&str] = &["out", "own", "peer", "data", "bytes", "slot_bytes", "slab_off"];

/// stage 调用(解析/组装双向同型)。
#[derive(Clone, Debug)]
pub struct ArStageCall {
    local: Bytes,
    own: Bytes,
    peer: Bytes,
    /// ★ **数据块**(2026-10-18;push 档 §36.7):slot 数据写哪里。
    /// v1 = `own`(写本端槽);push = `peer`(写对端槽)。header 两档都是 `own`/`peer`。
    data: Bytes,
    bytes: u64,
    slot_bytes: u64,
    slab_off: u64,
    /// ★ **fp8 载荷档**(§47 第二步):`true` ⇒ 发射名走 `*_f8_*`(量化写 +
    /// 反量化读)。缺省 false ⇒ 逐字走常规档(零行为变化)。
    f8: bool,
}

impl ArStageCall {
    /// 直接构造(调用方已备好四块:payload / 本端 staging / 对端别名 / **数据块**)。
    ///
    /// `data` = slot 数据写哪里:v1 传 `own`(写本端槽),push 传 `peer`(写对端槽)。
    pub fn new(
        local: Bytes,
        own: Bytes,
        peer: Bytes,
        data: Bytes,
        bytes: u64,
        slot_bytes: u64,
        slab_off: u64,
    ) -> Self {
        Self { local, own, peer, data, bytes, slot_bytes, slab_off, f8: false }
    }

    /// 切到 **fp8 载荷档**(§47 第二步;builder —— 存量调用点零改)
    pub fn with_f8(mut self, f8: bool) -> Self {
        self.f8 = f8;
        self
    }

    /// 反序列化(7 槽;按名取址)。
    pub fn parse(msg: &LaunchMsg) -> Result<(Self, Bytes), OpError> {
        let s = crate::client::NamedSlots::new(msg)?;
        let local = s.flat_block("local")?;
        // stage 的输出槽 = own(被写入的 staging;`out_elems` 由调用方按字节口径给)
        let own = s.out_block("own", msg.out_elems)?;
        let peer = s.flat_block("peer")?;
        let data = s.flat_block("data")?;
        let me = Self {
            local,
            own: own.clone(),
            peer,
            data,
            bytes: s.u64("bytes")?,
            slot_bytes: s.u64("slot_bytes")?,
            slab_off: s.u64("slab_off")?,
            f8: false, // 反序列化面不决定发射名(名字由 msg.kernel.name 带)
        };
        Ok((me, own))
    }

    /// 组装发射消息(几何由 runtime 自持 ⇒ grid/block 全零哨兵;名字按 dtype 选 entry)
    pub fn to_launch(&self) -> LaunchMsg {
        LaunchMsg {
            kernel: crate::contract::KernelSource {
                name: stage_entry_dt(self.local.dtype, self.f8).to_string(),
                source: String::new(),
            },
            args: vec![
                Arg::block(self.local.id, self.local.dtype, self.local.rank),
                Arg::block(self.own.id, self.own.dtype, self.own.rank),
                Arg::block(self.peer.id, self.peer.dtype, self.peer.rank),
                Arg::block(self.data.id, self.data.dtype, self.data.rank),
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
    /// ★ **数据块**(v1 = own / push = peer;见 `new` 的说明)
    pub fn data(&self) -> &Bytes {
        &self.data
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
    /// ★ **数据块**(2026-10-18;push 档 §36.7):slot 数据读哪里。
    /// v1 = `peer`(读对端槽);push = `own`(读本端槽 —— 对端已写进来)。
    data: Bytes,
    bytes: u64,
    slot_bytes: u64,
    slab_off: u64,
    /// ★ **fp8 载荷档**(与 `ArStageCall::f8` 同义;两侧必须同档 —— 否则
    /// 收端按 f16 解读 fp8 载荷 = 静默错值)
    f8: bool,
}

impl ArReduceCall {
    /// `data` = slot 数据读哪里:v1 传 `peer`(读对端槽),push 传 `own`(读本端槽)。
    pub fn new(
        out: Bytes,
        own: Bytes,
        peer: Bytes,
        data: Bytes,
        bytes: u64,
        slot_bytes: u64,
        slab_off: u64,
    ) -> Self {
        Self { out, own, peer, data, bytes, slot_bytes, slab_off, f8: false }
    }

    /// 切到 **fp8 载荷档**(与 `ArStageCall::with_f8` 成对使用)
    pub fn with_f8(mut self, f8: bool) -> Self {
        self.f8 = f8;
        self
    }

    pub fn parse(msg: &LaunchMsg) -> Result<(Self, Bytes), OpError> {
        let s = crate::client::NamedSlots::new(msg)?;
        let out = s.out_block("out", msg.out_elems)?;
        let own = s.flat_block("own")?;
        let peer = s.flat_block("peer")?;
        let data = s.flat_block("data")?;
        let me = Self {
            out: out.clone(),
            own,
            peer,
            data,
            bytes: s.u64("bytes")?,
            slot_bytes: s.u64("slot_bytes")?,
            slab_off: s.u64("slab_off")?,
            f8: false, // 反序列化面不决定发射名(名字由 msg.kernel.name 带)
        };
        Ok((me, out))
    }

    pub fn to_launch(&self) -> LaunchMsg {
        LaunchMsg {
            kernel: crate::contract::KernelSource {
                name: reduce_entry_dt(self.out.dtype, self.f8).to_string(),
                source: String::new(),
            },
            args: vec![
                Arg::block(self.out.id, self.out.dtype, self.out.rank),
                Arg::block(self.own.id, self.own.dtype, self.own.rank),
                Arg::block(self.peer.id, self.peer.dtype, self.peer.rank),
                Arg::block(self.data.id, self.data.dtype, self.data.rank),
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
    /// ★ **数据块**(v1 = peer / push = own;见 `new` 的说明)
    pub fn data(&self) -> &Bytes {
        &self.data
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

// ============================================================================
// 1-stage(对称缓冲 ⇒ 融合核;`TP2-施工设计.md` §35)
// ============================================================================

/// 1-stage 线格式(`O,T,T,T,sz`):`out`(= **本端槽**)· `own` hdr · `peer` hdr ·
/// `peer_data`(= **对端槽的跨卡别名**)· `bytes`
pub const SIG_1STAGE: &str = "O,T,T,T,sz";
/// `SIG_1STAGE` 的槽名表(与 `ops::comm::CommAr1Stage*::PARAMS` **逐位对应**;
/// `client::tests::parse_slots_are_exactly_declared` 锁)
pub const PARSE_SLOTS_1STAGE: &[&str] = &["out", "src", "own", "peer", "peer_data", "bytes"];

/// **1-stage allreduce 调用**(对称缓冲;融合核 —— 没有 stage 段、零拷贝)。
///
/// 语义:`out += peer_data`(**原地**),同时由**同一个核**维护 header 的
/// `produced`/`consumed` 与 credit(`cu/comm/ar_reduce.cu` 的 1-stage 头注)。
///
/// ⚠️ **`out` 就是本端那个对称槽本身** —— 上游算子(GEMM)必须**直接写它**,
/// 否则"发布 produced = 数据就绪"就是谎话(前置清单 ④)。
#[derive(Clone, Debug)]
pub struct Ar1StageCall {
    out: Bytes,
    /// ★ **本端对称槽**(只读;§59.2 修法,2026-10-19):GEMM 直写它 ⇒ 对端可读。
    /// 旧实现把 `out` 指向它并**原地累加** ⇒ 与"对端可读"语义冲突 ⇒ 间歇竞态
    /// (对端读到半加状态)。现在合成结果写 `out`(独立块),槽全程只读。
    src: Bytes,
    own: Bytes,
    peer: Bytes,
    peer_data: Bytes,
    bytes: u64,
}

impl Ar1StageCall {
    pub fn new(out: Bytes, src: Bytes, own: Bytes, peer: Bytes, peer_data: Bytes, bytes: u64) -> Self {
        Self { out, src, own, peer, peer_data, bytes }
    }

    pub fn parse(msg: &LaunchMsg) -> Result<(Self, Bytes), OpError> {
        let s = crate::client::NamedSlots::new(msg)?;
        let out = s.out_block("out", msg.out_elems)?;
        let me = Self {
            out: out.clone(),
            src: s.flat_block("src")?,
            own: s.flat_block("own")?,
            peer: s.flat_block("peer")?,
            peer_data: s.flat_block("peer_data")?,
            bytes: s.u64("bytes")?,
        };
        Ok((me, out))
    }

    pub fn to_launch(&self) -> LaunchMsg {
        LaunchMsg {
            kernel: crate::contract::KernelSource {
                name: one_stage_entry(self.out.dtype).to_string(),
                source: String::new(),
            },
            args: vec![
                Arg::block(self.out.id, self.out.dtype, self.out.rank),
                Arg::block(self.src.id, self.src.dtype, self.src.rank),
                Arg::block(self.own.id, self.own.dtype, self.own.rank),
                Arg::block(self.peer.id, self.peer.dtype, self.peer.rank),
                Arg::block(self.peer_data.id, self.peer_data.dtype, self.peer_data.rank),
                Arg::U64(self.bytes),
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

    /// **本端对称槽**(只读;§59.2 修法:输出写独立块,槽全程只读)
    pub fn src(&self) -> &Bytes {
        &self.src
    }
    pub fn own(&self) -> &Bytes {
        &self.own
    }
    pub fn peer(&self) -> &Bytes {
        &self.peer
    }
    pub fn peer_data(&self) -> &Bytes {
        &self.peer_data
    }
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

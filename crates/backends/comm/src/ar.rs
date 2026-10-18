//! allreduce 协议:staging 布局 + chunk 计划 —— **TP2 S2 的协议面**。
//!
//! 设计出处:`docs/arch/TP2-施工设计.md` §3(布局 / 两核 / 不变量 I1-I5 / chunk 计划)。
//!
//! # 协议速览(两个核,两条不变量)
//!
//! 每 rank 一块 staging(VMM 显式授予,对端经 alias 可读),布局见 [`StagingLayout`]:
//!
//! ```text
//! header: { produced: u64, consumed: u64, slot_bytes: u32, n_slots: u32, magic: u32, rsv }
//! slot[0..N_SLOTS]
//! ```
//!
//! - `produced` = 本端已写入的调用数(stage 核递增);
//! - `consumed` = 本端已消费的**对端**调用数(reduce 核递增;对端 stage 核读它当 credit);
//! - **I1 不覆盖未读**:stage 等 `peer.consumed + n_slots > own.produced`;
//! - **I2 不读未写**:reduce 等 `peer.produced > own.consumed`;
//! - 两侧只做**相对比较** + 设备内自增游标 ⇒ 图回放安全(不烘绝对值)。
//!
//! # 本文件现状(C2/C3)
//!
//! 纯逻辑:布局算术 + chunk 计划 + **调用序列展开**([`AllReduce::steps`])。
//! 发射是设备面(async 命令/流序)—— 调用方(引擎/门测试)把每个
//! `(phase, chunk)` 转成 LaunchMsg;两个核与家族在 `owl-kernels`
//! (`cu/comm/ar_*.cu` + `server::comm`)。
//!
//! # 纪律:归约序必须与 shape 解耦(`TP2-施工设计.md` §19.3-①;S5 前置)
//!
//! v1 的"固定核 + 相对游标 + **单 chunk**"是**确定性资产**:载荷 > slot 时
//! **响亮拒**(绝不静默分块)⇒ **归约块数/树形与 batch 形状无关** —— 这正是 SGLang
//! 给 TP=2 的 "consistent floating-point addition order"。
//!
//! S5 放开多块/流水时**必须**守住:(a) 切法固定(块数与 shape 无关)或
//! (b) 显式"确定性档"固定单一路径。**验收 = `tp_determinism_gate`(G8-a)在 S5 之后
//! 仍绿**。反例参照:SGLang TP4-prefix 模式下"切分随 shape 变"导致的非确定(issue #10785)。

/// staging header 字节数(64 B:两条 cache line,避免与 slot 数据同线伪共享)。
pub const HEADER_BYTES: u64 = 64;

/// 每个 slot 的字节数(设计值:32 MiB × 2 slot = 64 MiB/rank;见施工设计 §6 预算账)。
pub const SLOT_BYTES: u64 = 32 << 20;

/// slot 环深度(2 = 乒乓;credit 上限由它决定)。
pub const N_SLOTS: u32 = 2;

/// header 字段偏移(与 `.cu` 侧必须逐字节一致 —— 改动 = 双侧同改 + 门重跑)。
///
/// 权威布局注记住 `crates/kernels/cu/comm/ar_stage.cu` 头注;此处是 Rust 侧
/// 镜像(`kernels::family::comm::asset::AR_HDR_BYTES` 是第三处——三处同值)。
pub mod hdr {
    /// `u64` 本端已写入调用数(stage 核递增)
    pub const PRODUCED: u64 = 0;
    /// `u64` 本端已消费对端调用数(reduce 核递增)
    pub const CONSUMED: u64 = 8;
    /// `u32` slot 字节数
    pub const SLOT_BYTES: u64 = 16;
    /// `u32` slot 数量
    pub const N_SLOTS: u64 = 20;
    /// `u32` 魔数(`0x4F57_4C41` = "OWLA";防串块)
    pub const MAGIC: u64 = 24;
    /// `u32` **错误码**(核写;host 在下次同步点读 —— D4 "全有或全无"):
    /// 0 正常 · 1 stage credit 超时 · 2 reduce 数据超时 · 3 header 校验失败
    pub const ERR: u64 = 28;
    /// `u64` **多 block 的两级等待令牌**(2026-10-17;32..40)。
    ///
    /// 语义:**单调递增** = `本端 consumed + 1`(不是布尔!)⇒ 与游标同源、
    /// **图回放安全**(不需要每次重置,也就不存在"重置与下一次读的竞态")。
    /// 写者 = block 0(它去 PCIe 上轮询对端 `produced`);读者 = 其余 block
    /// (**读本端 header = 本地 L2**,避免 N 个 block 同时用 PCIe 轮询对端
    /// 同一行 —— 那是"多 block 化"最容易顺手引入的坏账)。
    pub const WAIT_DONE: u64 = 32;
    /// `u32` **完成块计数**(40..44;reduce 尾"最后一块发布 consumed")。
    /// 每 block 数据收尾后 `atomicAdd(+1)`;计数达 `gridDim.x` 的那一块
    /// **回零**并发布 `consumed`(对端 stage 的 credit)。回零发生在同一核内、
    /// 且在所有块都加过之后 ⇒ 下一次调用(同流保序)必定看到 0。
    pub const BLOCKS_DONE: u64 = 40;
}

/// 错误码(与 `.cu` 的 `*own_err = N` 同值;host 侧判据面)
pub mod err_code {
    /// stage 等 peer.consumed(credit)超时
    pub const STAGE_TIMEOUT: u32 = 1;
    /// reduce 等 peer.produced 超时
    pub const REDUCE_TIMEOUT: u32 = 2;
    /// header 校验失败(magic / slot_bytes 漂移)
    pub const HDR_INVALID: u32 = 3;
}

/// 魔数("OWLA";串块判据)。
pub const MAGIC: u32 = 0x4F57_4C41;

/// staging 布局(纯算术;两侧 rank 必须同参 —— 协议对称性前提)。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StagingLayout {
    pub slot_bytes: u64,
    pub slots: u32,
}

impl StagingLayout {
    /// 默认布局(32 MiB × 2)。
    pub const fn default_layout() -> Self {
        Self { slot_bytes: SLOT_BYTES, slots: N_SLOTS }
    }

    pub const fn new(slot_bytes: u64, slots: u32) -> Self {
        Self { slot_bytes, slots }
    }

    /// 整块字节数(header + 全部 slot)= 需要 VMM 授予的额度。
    pub const fn total_bytes(&self) -> u64 {
        HEADER_BYTES + self.slot_bytes * self.slots as u64
    }

    /// 第 `idx` 个 slot 的块内偏移。`idx` 必须 < `slots`。
    pub const fn slot_offset(&self, idx: u32) -> Option<u64> {
        if idx < self.slots {
            Some(HEADER_BYTES + self.slot_bytes * idx as u64)
        } else {
            None
        }
    }

    /// slot 选择(第 `call` 次调用用哪个 slot = `call % slots`;两侧同式)。
    pub const fn slot_of(&self, call: u32) -> u32 {
        call % self.slots
    }
}

/// 一次 allreduce 的分块计划(host 侧在**建图前**算好,偏移量烘进图节点)。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArPlan {
    /// 载荷总字节数
    pub bytes: u64,
    /// slot 字节数
    pub slot_bytes: u64,
    /// 分块数 = ceil(bytes / slot_bytes);0 字节载荷 ⇒ 0 次
    pub calls: u32,
}

impl ArPlan {
    /// 第 `i` 块的载荷偏移与字节数(`i` 必须 < `calls`)。
    pub fn chunk(&self, i: u32) -> Option<(u64, u64)> {
        if i >= self.calls {
            return None;
        }
        let off = self.slot_bytes * i as u64;
        Some((off, (self.bytes - off).min(self.slot_bytes)))
    }
}

/// 分块计划(纯函数)。
pub fn plan_ar(bytes: u64, layout: &StagingLayout) -> ArPlan {
    let slot = layout.slot_bytes.max(1);
    let calls = bytes.div_ceil(slot) as u32;
    ArPlan { bytes, slot_bytes: layout.slot_bytes, calls }
}

/// allreduce 的调用阶段(两核交替:I5 两侧序列同构的载体)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArPhase {
    /// 本地 payload → 本端 staging slot(发布 produced)
    Stage,
    /// 原地累加对端 slot → 本端 payload(发布 consumed)
    Reduce,
}

/// 一次核调用的计划(发射端按此构造 LaunchMsg 的 `(bytes, slab_off)`)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArChunk {
    /// 第几块(0 起;= 调用序)
    pub index: u32,
    /// payload 内字节偏移(= `index × slot_bytes`)
    pub offset: u64,
    /// 本块字节数(末块可短)
    pub bytes: u64,
}

/// allreduce 调用门面(S2;计划面)。
///
/// 语义:对 `buf`(本端部分和)做 in-place allreduce —— `buf += Σ_{peer≠me} peer_partial`。
#[derive(Debug, Clone, Copy)]
pub struct AllReduce {
    pub plan: ArPlan,
}

impl AllReduce {
    pub fn new(plan: ArPlan) -> Self {
        Self { plan }
    }

    /// **展开调用序列**(v1 顺序:第 i 块 = `stage_i → reduce_i`)。
    ///
    /// 为什么是**纯数据展开**而不是直接发射:发射是设备面(async 命令/
    /// 流序,见 `GpuClient::launch`),归引擎/后端;本层只对"发什么、按什么
    /// 序"负责 —— 而**两侧序列同构**(I5)正是由本函数保证的(两侧同 plan
    /// ⇒ 同序列)。调用方把每个 `(phase, chunk)` 转成 LaunchMsg:
    /// `stage` 用 [`crate::ar::ArChunk`] 的 `(offset, bytes)` 填 slab_off/bytes。
    pub fn steps(&self) -> Vec<(ArPhase, ArChunk)> {
        let mut out = Vec::with_capacity(self.plan.calls as usize * 2);
        for i in 0..self.plan.calls {
            let Some((offset, bytes)) = self.plan.chunk(i) else { break };
            let c = ArChunk { index: i, offset, bytes };
            out.push((ArPhase::Stage, c));
            out.push((ArPhase::Reduce, c));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_layout_is_64mib_plus_header() {
        let l = StagingLayout::default_layout();
        assert_eq!(l.total_bytes(), HEADER_BYTES + 2 * (32 << 20));
        // 预算账:64MiB + 64B ≤ 硬上限 192MiB(施工设计 §6)
        assert!(l.total_bytes() < crate::peer::PEER_ALLOC_HARD_CAP);
    }

    #[test]
    fn slot_offsets_and_bounds() {
        let l = StagingLayout::default_layout();
        assert_eq!(l.slot_offset(0), Some(HEADER_BYTES));
        assert_eq!(l.slot_offset(1), Some(HEADER_BYTES + (32 << 20)));
        assert_eq!(l.slot_offset(2), None, "越界 slot 必须返回 None(不许静默钳位)");
        assert_eq!(l.slot_of(0), 0);
        assert_eq!(l.slot_of(1), 1);
        assert_eq!(l.slot_of(2), 0, "乒乓:第 2 次回到 slot 0");
    }

    #[test]
    fn plan_covers_boundaries() {
        let l = StagingLayout::default_layout();
        // decode 尺寸:1 块
        assert_eq!(plan_ar(10 << 10, &l).calls, 1);
        // 恰好一块
        assert_eq!(plan_ar(32 << 20, &l).calls, 1);
        // 一块 + 1 字节
        assert_eq!(plan_ar((32 << 20) + 1, &l).calls, 2);
        // 零载荷:0 次(调用方应跳过,不该产生空核)
        assert_eq!(plan_ar(0, &l).calls, 0);
        // 80MB(prefill@8192 单层载荷)= 3 块
        assert_eq!(plan_ar(80 << 20, &l).calls, 3);
    }

    #[test]
    fn chunk_sizes_are_exact_partition() {
        let l = StagingLayout::default_layout();
        let p = plan_ar(80 << 20, &l);
        let mut sum = 0u64;
        for i in 0..p.calls {
            let (off, len) = p.chunk(i).expect("in-range");
            assert_eq!(off, (32 << 20) * i as u64);
            sum += len;
        }
        assert_eq!(sum, p.bytes, "分块必须恰好覆盖载荷(不重不漏)");
        assert!(p.chunk(p.calls).is_none());
        // 末块 ≤ slot
        let (_, last) = p.chunk(p.calls - 1).unwrap();
        assert!(last <= 32 << 20);
    }

    #[test]
    fn steps_expand_stage_reduce_pairs() {
        let l = StagingLayout::default_layout();
        // 零载荷 ⇒ 零步(调用方应跳过)
        assert!(AllReduce::new(plan_ar(0, &l)).steps().is_empty());
        // 单块 ⇒ stage + reduce
        let one = AllReduce::new(plan_ar(10 << 10, &l)).steps();
        assert_eq!(one.len(), 2);
        assert_eq!(one[0].0, ArPhase::Stage);
        assert_eq!(one[1].0, ArPhase::Reduce);
        assert_eq!(one[0].1, one[1].1, "同一块的两相用同一 chunk 计划");
        assert_eq!(one[0].1.offset, 0);
        assert_eq!(one[0].1.bytes, 10 << 10);
        // 三块 ⇒ 六步,序 = S0 R0 S1 R1 S2 R2
        let three = AllReduce::new(plan_ar(80 << 20, &l)).steps();
        assert_eq!(three.len(), 6);
        let want: Vec<ArPhase> = vec![
            ArPhase::Stage,
            ArPhase::Reduce,
            ArPhase::Stage,
            ArPhase::Reduce,
            ArPhase::Stage,
            ArPhase::Reduce,
        ];
        assert_eq!(three.iter().map(|(p, _)| *p).collect::<Vec<_>>(), want);
        // 偏移逐块推进、末块短
        assert_eq!(three[4].1.offset, 64 << 20);
        assert_eq!(three[5].1.bytes, (80 << 20) - (64 << 20));
        assert_eq!(three[5].1.bytes, 16 << 20);
        for (i, (_, c)) in three.iter().enumerate() {
            assert_eq!(c.index as usize, i / 2, "index = 调用序");
        }
    }
}

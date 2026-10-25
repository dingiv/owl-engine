//! 宿主层(vLLM 式 CPU 卸载层的形态):**页可换出的宿主存储 + 有界池**。
//!
//! 三条形态律(对齐 vLLM 的 CPU KV cache):
//! 1. **宿主内容不 pin**(pageable `Vec<u8>`):pin 只用在*搬运途中*的暂存租约上
//!    (vLLM 同款:CPU cache 是普通内存,传输走 pinned staging)⇒ RAM 可被内核回收/换出;
//!    **但分配带大页提示**(2026-10-19):pageable 的代价是首次写逐页缺页 ——
//!    159 MiB 的 4 KiB 缺页 ≈ 80 ms(≈2 GB/s)而 2 MiB 大页 ≈ 12 ms ⇒ 这是"快照
//!    下沉一条 ≈100 ms"的主因。提示不是 pin(仍可换出/回收),见 `HostBlock::zeroed`。
//! 2. **容量显式有界**(`budget_bytes`,0 = 关层):越界 = **响亮拒**,
//!    由驱逐序(策略对象)决定谁先让位 —— 不许悄悄超发(C4 风险条款);
//! 3. **缓冲复用**:小块池(free list)吸收 churn;`evict` 真的把内容还掉
//!    (超出 `free_cap` 的缓冲直接 drop ⇒ RAM 真释放,不留影)。
//!
//! 本文件只做**账与容器**(方法自肃);真搬运(`dtoh`/`upload_pinned`)在
//! [`crate::moves`] 的自由函数里 —— 见 `arch.md` 律⑦。

use std::collections::BTreeMap;

use owl_iface::contract::{ConnSlot, Connector, ModelError};

use crate::error::{StoreError, StoreResult};
use crate::id::ChainHash;

/// 宿主层的一块存储(pageable;大小 = 逻辑块字节)。
///
/// **分配带大页提示**(2026-10-19;§9.1(6)b):块第一次被**写**时内核逐页缺页,
/// 159 MiB 的 4 KiB 缺页(4 万次)≈ **80 ms**(≈2 GB/s,实测),而 2 MiB 大页 ≈ **12 ms**
/// —— 这就是"快照下沉一条 ≈100 ms"的主因(实测 = 大页冷写 + DMA)。
/// 内核 THP 姿态在本机 = `[madvise]` ⇒ 必须显式申请;提示失败**不影响正确性**
/// (只是回到 4 KiB 缺页速度),故不阻断、只一次性可见告知。
#[derive(Debug)]
pub struct HostBlock {
    bytes: Vec<u8>,
    /// 大页提示是否生效(可观测面:失败 = 冷写回到 4 KiB 缺页速度;
    /// 纠错/报告靠它 —— 不许静默吞掉)。
    hugepage: bool,
}

/// 大页提示:把 `bytes` 的 **2 MiB 对齐内区间**申请为 THP(边缘不足一页的段
/// 不碰 —— `madvise` 要求区间对齐,部分页会 EINVAL)。
/// 返回 `Err(os error)` = 提示没生效(调用方自行决定报不报)。
#[cfg(target_os = "linux")]
fn hint_hugepage(bytes: &mut [u8]) -> std::io::Result<()> {
    const HP: usize = 2 << 20;
    let base = bytes.as_mut_ptr() as usize;
    let lo = (base + HP - 1) & !(HP - 1);
    let hi = (base + bytes.len()) & !(HP - 1);
    if hi <= lo {
        return Ok(()); // 块 < 一页大页:没东西可提示(小 KV 块走这条路)
    }
    let r = unsafe {
        libc::madvise(lo as *mut libc::c_void, hi - lo, libc::MADV_HUGEPAGE)
    };
    if r == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

#[cfg(not(target_os = "linux"))]
fn hint_hugepage(_bytes: &mut [u8]) -> std::io::Result<()> {
    Ok(())
}

impl HostBlock {
    /// 零填构造(大小 = 逻辑块字节)。大页提示失败 ⇒ **记在块上**
    /// ([`Self::hugepage`]),由 [`HostTier::hint_failures`] 汇总上报 ——
    /// 不阻断(提示只影响速度,不影响内容/正确性),但**不静默**。
    pub fn zeroed(len: usize) -> Self {
        let mut bytes = vec![0u8; len];
        let hugepage = hint_hugepage(&mut bytes).is_ok();
        Self { bytes, hugepage }
    }

    /// **从已有字节构造**(磁盘读回 / 溢写回滚用:内容已在,无需再零填)。
    ///
    /// 大页提示不适用(内存来自 `Vec` 的既有分配);`hugepage` 记 `false`
    /// ⇒ 报告面的 `hint_failures` 不该被这条计入,故本构造**不**自报失败。
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Self { bytes, hugepage: false }
    }

    /// 本块的大页提示是否生效。
    pub fn hugepage(&self) -> bool {
        self.hugepage
    }

    /// 字节数。
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// 是否空块。
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// 只读字节面。
    pub fn as_slice(&self) -> &[u8] {
        &self.bytes
    }

    /// 可写字节面(搬运的落地口)。
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.bytes
    }

    /// **取出内部字节**(零拷贝;磁盘溢写把内容交给 I/O 任务时用 ——
    /// 159 MiB 的快照不该为了搬运再拷一份)。
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

/// 宿主层(有界 + 可驱逐 + 缓冲复用)。`free_cap` = 空转缓冲保留上限。
///
/// **键泛型**(S2):KV 腿按 `ChainHash`(内容身份)驻留,状态腿按 [`SnapKey`]
/// (块边界身份)—— 两腿各自一本账,键类型把"混用"挡在类型上。
#[derive(Debug)]
pub struct HostTier<K = ChainHash> {
    block_bytes: usize,
    budget_bytes: usize,
    free_cap: usize,
    resident: BTreeMap<K, HostBlock>,
    free: Vec<HostBlock>,
    /// 因容量驱逐而累计释放的字节(报告面)。
    evicted_bytes: u64,
    /// **同身份重复落地**次数(账漂指纹;2026-10-24 粘性毒化案新增)。
    ///
    /// 判据：`K` 是内容寻址链哈希 ⇒ 同身份 = 同内容，落地幂等可合并；
    /// 但这个计数非零意味着**上游两条腿账对不上**(典型成因：重绑设备时
    /// 没丢旧宿主副本)，必须当告警看，不是当正常路径看。
    duplicate_landings: u64,
    /// 大页提示未生效的块数(可观测面;0 = 全部生效)。
    /// 它不等于错误:提示只影响"冷写缺页速度"(159 MiB: 12ms vs 80ms),不影响内容。
    hint_failures: usize,
}

/// 宿主层键(可寻址 + 可打印)。
pub trait HostKey: Ord + Copy + std::fmt::Debug {}
impl HostKey for ChainHash {}
impl HostKey for crate::id::SnapKey {}

/// 空转缓冲默认保留上限(块数)。
const DEFAULT_FREE_CAP: usize = 4;

impl<K: HostKey> HostTier<K> {
    /// 构造。`budget_bytes = 0` = **关层**(任何落位都拒)。
    pub fn new(block_bytes: usize, budget_bytes: usize) -> StoreResult<Self> {
        if block_bytes == 0 {
            return Err(StoreError::Invariant("宿主层块字节为 0".into()));
        }
        if budget_bytes != 0 && budget_bytes < block_bytes {
            return Err(StoreError::Budget(format!(
                "宿主层预算 {budget_bytes} 装不下单块 {block_bytes}(按块粒度对齐)"
            )));
        }
        Ok(Self {
            block_bytes,
            budget_bytes,
            free_cap: DEFAULT_FREE_CAP,
            resident: BTreeMap::new(),
            free: Vec::new(),
            evicted_bytes: 0,
            duplicate_landings: 0,
            hint_failures: 0,
        })
    }

    /// 单块字节。
    pub fn block_bytes(&self) -> usize {
        self.block_bytes
    }

    /// 本层上限(0 = 关层)。
    pub fn budget_bytes(&self) -> usize {
        self.budget_bytes
    }

    /// 是否关层。
    pub fn is_disabled(&self) -> bool {
        self.budget_bytes == 0
    }

    /// 取一块空缓冲(优先复用;新建时**先判预算** —— 超界响亮拒)。
    pub fn acquire(&mut self) -> StoreResult<HostBlock> {
        if let Some(b) = self.free.pop() {
            return Ok(b);
        }
        if self.is_disabled() {
            return Err(StoreError::Exhausted("宿主层已关闭(host_bytes = 0)".into()));
        }
        let after = self.used_bytes() + self.free_bytes() + self.block_bytes;
        if after > self.budget_bytes {
            return Err(StoreError::Exhausted(format!(
                "宿主层越界:{after} > {}(关层请置 0,放量请调大 host_bytes)",
                self.budget_bytes
            )));
        }
        let b = HostBlock::zeroed(self.block_bytes);
        if !b.hugepage() {
            self.hint_failures += 1;
        }
        Ok(b)
    }

    /// 大页提示未生效的块数(0 = 全部生效;>0 = 冷写回到 4 KiB 缺页速度 ——
    /// 159 MiB 约 80ms vs 12ms)。上报方:引擎在快照下沉点打点。
    pub fn hint_failures(&self) -> usize {
        self.hint_failures
    }

    /// **借一批空槽**(swap 命令的台子;§15)。
    ///
    /// 预算检查逐块走 [`Self::acquire`] ⇒ 中途越界 = 已借出的缓冲回小块池
    /// (**不留半台**),错误上抛。
    pub fn lend_slots(&mut self, n: usize) -> StoreResult<Vec<HostBlock>> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            match self.acquire() {
                Ok(b) => out.push(b),
                Err(e) => {
                    self.recycle_many(out);
                    return Err(e);
                }
            }
        }
        Ok(out)
    }

    /// **批量落账**(命令回执后:内容已写好,装回驻留表)。**两趟、零半途态**。
    ///
    /// 为什么要两趟(2026-10-24 粘性毒化案)：旧版逐项 `put` 边落边校验，
    /// 中途一撞重复/预算 ⇒ 前几块已落宿主、后面的没落，而调用方
    /// (`commit_offload`)的 `unbind` 整个不执行 ⇒ 形成“设备还绑着 + 宿主已有几块”
    /// 的**半半账**，下一次下沉同一块必再撞 ⇒ 每个后续 turn 全灭(须重启才解)。
    ///
    /// 现口径：
    /// ① 预检(**零副作用**)：尺寸不符 / 预算越界 ⇒ **一块都不落**直接 `Err`；
    /// ② 落地：同身份重复 = **同一内容**(K 内容寻址)⇒ 幂等换新，留本次刚从
    ///    设备搬来的那份(它对应设备现状)，旧缓冲回收，并计入
    ///    [`HostTier::duplicate_landings`](响亮，不静默)。
    pub fn land(&mut self, items: Vec<(K, HostBlock)>) -> StoreResult<()> {
        // ① 预检
        let mut fresh = 0usize;
        for (chain, buf) in &items {
            if buf.len() != self.block_bytes {
                return Err(StoreError::Invariant(format!(
                    "宿主块大小不符:{} ≠ {}",
                    buf.len(),
                    self.block_bytes
                )));
            }
            if !self.resident.contains_key(chain) {
                fresh += 1;
            }
        }
        let after = self.used_bytes() + fresh * self.block_bytes;
        if self.budget_bytes != 0 && after > self.budget_bytes {
            return Err(StoreError::Exhausted(format!(
                "宿主层越界:{after} > {}(需先驱逐)",
                self.budget_bytes
            )));
        }
        // ② 落地
        for (chain, buf) in items {
            if let Some(stale) = self.resident.remove(&chain) {
                self.recycle(stale);
                self.duplicate_landings += 1;
            }
            self.resident.insert(chain, buf);
        }
        Ok(())
    }

    /// 同身份重复落地次数(账漂指纹;报告面用)。
    pub fn duplicate_landings(&self) -> u64 {
        self.duplicate_landings
    }

    /// **批量回收空转缓冲**(换入后内容已上行;超 `free_cap` 的真释放)。
    pub fn recycle_many(&mut self, bufs: Vec<HostBlock>) {
        for b in bufs {
            self.recycle(b);
        }
    }

    /// 登记驻留(内容落到宿主层)。
    pub fn put(&mut self, block: K, buf: HostBlock) -> StoreResult<()> {
        if buf.len() != self.block_bytes {
            return Err(StoreError::Invariant(format!(
                "宿主块大小不符:{} ≠ {}",
                buf.len(),
                self.block_bytes
            )));
        }
        if self.resident.contains_key(&block) {
            return Err(StoreError::Invariant(format!("块 {block:?} 已在宿主层")));
        }
        let after = self.used_bytes() + self.block_bytes;
        if self.budget_bytes != 0 && after > self.budget_bytes {
            return Err(StoreError::Exhausted(format!(
                "宿主层越界:{after} > {}(需先驱逐)",
                self.budget_bytes
            )));
        }
        self.resident.insert(block, buf);
        Ok(())
    }

    /// 取走驻留内容(回灌设备时用;缓冲所有权移出,由调用方决定复用或释放)。
    pub fn take(&mut self, block: K) -> StoreResult<HostBlock> {
        self.resident
            .remove(&block)
            .ok_or_else(|| StoreError::Unknown(format!("块 {block:?} 不在宿主层")))
    }

    /// 只读查看。
    pub fn get(&self, block: K) -> Option<&HostBlock> {
        self.resident.get(&block)
    }

    /// 驱逐(内容丢弃;缓冲回小块池,超上限则真释放)。
    pub fn evict(&mut self, block: K) -> StoreResult<usize> {
        let buf = self
            .resident
            .remove(&block)
            .ok_or_else(|| StoreError::Unknown(format!("块 {block:?} 不在宿主层")))?;
        let bytes = buf.len();
        self.evicted_bytes += bytes as u64;
        self.recycle(buf);
        Ok(bytes)
    }

    /// **键重挂**(前缀无感删除的宿主半;2026-10-23):驻留内容换一个身份。
    ///
    /// 只动账 —— 内容零搬运(删除是"搬运记忆":状态内容对平移后的位置帧依然
    /// 成立,变的只是它的**身份** = 块边界哈希)。两条响亮拒与
    /// [`crate::state::StateStore::rekey_snap`] 同律:`old` 不在账 / `new` 已在账
    /// (覆盖会让两处身份指向同一份内容,账与槽位映射随即分叉)。
    pub fn rekey(&mut self, old: K, new: K) -> StoreResult<()> {
        if self.resident.contains_key(&new) {
            return Err(StoreError::Invariant(format!(
                "宿主层重挂:{new:?} 已在账(不许覆盖)"
            )));
        }
        let buf = self
            .resident
            .remove(&old)
            .ok_or_else(|| StoreError::Unknown(format!("宿主层重挂:{old:?} 不在账")))?;
        self.resident.insert(new, buf);
        Ok(())
    }

    /// 驻留块号(驱逐序输入;调用方按策略排序)。
    pub fn resident_blocks(&self) -> Vec<K> {
        self.resident.keys().copied().collect()
    }

    /// 驻留内容字节。
    pub fn used_bytes(&self) -> usize {
        self.resident.len() * self.block_bytes
    }

    /// 空转缓冲字节(真实占用的 RAM,报告面要算上)。
    pub fn free_bytes(&self) -> usize {
        self.free.len() * self.block_bytes
    }

    /// 驱逐累计字节。
    pub fn evicted_bytes(&self) -> u64 {
        self.evicted_bytes
    }

    fn recycle(&mut self, buf: HostBlock) {
        if self.free.len() < self.free_cap {
            self.free.push(buf);
        }
        // 超出上限:drop ⇒ 真释放
    }
}

// ─────────────────── 搬运期槽台(§15 swap 的 Connector) ───────────────────

/// **搬运期的槽台**(设计卷 §15):一块逻辑块 = 一个槽(连续 `slot_bytes`)。
///
/// 生命周期:store 建台(借空缓冲)→ 随 `swap` 命令**移入** server(server 按
/// 槽号 + 槽内偏移逐片读写)→ 回执**原箱奉还** → store 拆台(`HostTier::land`
/// 落账,或 `recycle_many` 回收)。
///
/// **单主零锁**:命令期间台子的所有权在 server(无共享、无 `Mutex`);
/// **层级知识不外泄** —— server 只看见 [`Connector`] 三件(put/get/tier_name)。
///
/// 为什么台子不借自 `HostTier` 本身:宿主层是常驻账(驻留表 + 预算 + LRU),
/// 而台子是**一次命令的视图** —— 两者生命周期不同,拆开才便于「命令进行中
/// 宿主账不动」这条审计。
pub struct SlotTable {
    slots: Vec<HostBlock>,
}

impl SlotTable {
    /// 建台(槽号 = 入参顺序)。
    pub fn new(slots: Vec<HostBlock>) -> Self {
        Self { slots }
    }

    /// 单槽台(单块搬运)。
    pub fn one(buf: HostBlock) -> Self {
        Self { slots: vec![buf] }
    }

    /// 槽数。
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// 是否空台。
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// 拆台取回缓冲(顺序 = 槽号序)。
    pub fn into_slots(self) -> Vec<HostBlock> {
        self.slots
    }

    /// 槽号 → 缓冲(诊断/测试面)。
    pub fn slot(&self, slot: ConnSlot) -> Option<&HostBlock> {
        self.slots.get(slot.0 as usize)
    }

    /// 片窗口(统一越界/溢出检查 —— put/get 同一条口径)。
    fn window(&self, slot: ConnSlot, off: usize, len: usize) -> Result<std::ops::Range<usize>, ModelError> {
        let i = slot.0 as usize;
        let n = self.slots.len();
        let buf = self
            .slots
            .get(i)
            .ok_or_else(|| ModelError::Msg(format!("槽台越界:槽 {i} ≥ 台面 {n} 槽")))?;
        let end = off
            .checked_add(len)
            .ok_or_else(|| ModelError::Msg(format!("槽 {i} 偏移溢出:{off}+{len}")))?;
        if end > buf.len() {
            return Err(ModelError::Msg(format!(
                "槽 {i} 越界:{off}+{len} > 槽字节 {}",
                buf.len()
            )));
        }
        Ok(off..end)
    }
}

impl Connector for SlotTable {
    fn put(&mut self, slot: ConnSlot, off: usize, bytes: &[u8]) -> Result<(), ModelError> {
        let r = self.window(slot, off, bytes.len())?;
        let i = slot.0 as usize;
        let buf = self.slots.get_mut(i).expect("window 刚验过");
        buf.as_mut_slice()[r].copy_from_slice(bytes);
        Ok(())
    }

    fn get(&self, slot: ConnSlot, off: usize, out: &mut [u8]) -> Result<(), ModelError> {
        let r = self.window(slot, off, out.len())?;
        let buf = self.slots.get(slot.0 as usize).expect("window 刚验过");
        out.copy_from_slice(&buf.as_slice()[r]);
        Ok(())
    }

    fn tier_name(&self) -> &'static str {
        "ram"
    }

    fn unbox(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `land` 预检失败 ⇒ **零副作用**(旧版逐项 `put` 边落边校验,会在
    /// 撞错前把前几块落进去 ⇒ “设备还绑 + 宿主已有几块”的半半账;
    /// 2026-10-24 粘性毒化案的放大器)。
    #[test]
    fn land_rejects_without_half_landed_state() {
        let mut t: HostTier = HostTier::new(1024, 2048).expect("构造");
        let items = vec![
            (ChainHash::new(1), HostBlock::from_bytes(vec![7u8; 1024])),
            (ChainHash::new(2), HostBlock::from_bytes(vec![7u8; 512])), // 尺寸不符
        ];
        let err = t.land(items).expect_err("尺寸不符必须响亮拒");
        assert!(matches!(err, StoreError::Invariant(_)), "得 {err:?}");
        assert_eq!(t.used_bytes(), 0, "预检失败 ⇒ 一块都不落");
        assert!(t.resident_blocks().is_empty(), "驻留表不得动");
    }

    /// 超预算同样零副作用(预算在预检统一校,不靠逐块撞墙)。
    #[test]
    fn land_over_budget_lands_nothing() {
        let mut t: HostTier = HostTier::new(1024, 2048).expect("构造");
        let items = vec![
            (ChainHash::new(1), HostBlock::from_bytes(vec![1u8; 1024])),
            (ChainHash::new(2), HostBlock::from_bytes(vec![2u8; 1024])),
            (ChainHash::new(3), HostBlock::from_bytes(vec![3u8; 1024])),
        ];
        assert!(matches!(t.land(items).expect_err("3 块超 2 块预算"), StoreError::Exhausted(_)));
        assert_eq!(t.used_bytes(), 0, "超预算 ⇒ 零副作用");
    }

    /// 同身份重复落地 = 同内容(K 是内容寻址链哈希)⇒ **幂等换新**,
    /// 但必须**留下指纹**(`duplicate_landings`)供报面追问,不静默吞过。
    #[test]
    fn land_merges_duplicate_identity_loudly() {
        let mut t: HostTier = HostTier::new(1024, 2048).expect("构造");
        let k = ChainHash::new(1);
        t.put(k, HostBlock::from_bytes(vec![1u8; 1024])).expect("先落");
        t.land(vec![(k, HostBlock::from_bytes(vec![2u8; 1024]))])
            .expect("同身份 ⇒ 不该炸 turn");
        assert_eq!(t.used_bytes(), 1024, "不重复占预算(驻留只一份)");
        assert_eq!(t.resident_blocks().len(), 1, "驻留表不涨");
        assert_eq!(t.free_bytes(), 1024, "旧副本进空转缓冲(RAM 复用,不白还又要)");
        assert_eq!(t.get(k).map(|b| b.as_slice()[0]), Some(2), "留本次从设备搬来的那份");
        assert_eq!(t.duplicate_landings(), 1, "账漂指纹必须记账");
    }

    #[test]
    fn disabled_tier_rejects_loudly() {
        let mut t: HostTier = HostTier::new(1024, 0).expect("构造");
        assert!(t.is_disabled());
        let err = t.acquire().expect_err("关层必须拒");
        assert!(matches!(err, StoreError::Exhausted(_)), "得 {err:?}");
    }

    /// **大页提示的记账律**(2026-10-19;§9.1(6)b):
    /// ① 块自述提示结果(`HostBlock::hugepage`);
    /// ② `HostTier::hint_failures` = 借出块里提示未生效的**块数**(可观测面,不许静默);
    /// ③ 环境无关:本机 THP 姿态若不开,断言仍然成立(只声明二者一致)。
    #[test]
    fn hugepage_hint_is_counted() {
        let mut t: HostTier = HostTier::new(1024, 4096).expect("构造");
        let mut ok = 0usize;
        let mut fail = 0usize;
        for _ in 0..3 {
            let b = t.acquire().expect("借块");
            if b.hugepage() {
                ok += 1;
            } else {
                fail += 1;
            }
        }
        assert_eq!(ok + fail, 3);
        assert_eq!(t.hint_failures(), fail, "上报计数须与块自述一致(ok={ok} fail={fail})");
        // 大块 ≥ 2MiB 在 Linux+THP[madvise] 上应拿到提示(拿不到也不断言,只打印事实)
        let big = HostBlock::zeroed(4 << 20);
        eprintln!("[host] 4MiB 块大页提示生效 = {}(本机 THP 姿态见 /sys/kernel/mm/transparent_hugepage/enabled)", big.hugepage());
        assert_eq!(big.len(), 4 << 20);
    }

    /// 槽台:越界/溢出必须**响亮拒**(不静默截断);多槽按号各就各位。
    #[test]
    fn slot_table_is_bounds_checked_and_addressed_by_slot() {
        let mut t = SlotTable::new(vec![HostBlock::zeroed(16), HostBlock::zeroed(16)]);
        t.put(ConnSlot(1), 4, &[7u8; 8]).expect("写槽 1");
        let mut out = [0u8; 8];
        t.get(ConnSlot(1), 4, &mut out).expect("读槽 1");
        assert_eq!(out, [7u8; 8]);
        let mut zero = [9u8; 8];
        t.get(ConnSlot(0), 4, &mut zero).expect("读槽 0");
        assert_eq!(zero, [0u8; 8], "槽 0 未被写");
        assert!(matches!(t.put(ConnSlot(2), 0, &[0u8; 4]), Err(ModelError::Msg(_))), "槽号越界");
        assert!(matches!(t.put(ConnSlot(0), 12, &[0u8; 8]), Err(ModelError::Msg(_))), "槽内越界");
        assert!(matches!(t.get(ConnSlot(0), usize::MAX, &mut [0u8; 4]), Err(ModelError::Msg(_))), "偏移溢出");
        assert_eq!(t.tier_name(), "ram");
    }

    #[test]
    fn budget_is_enforced_and_content_can_be_evicted() {
        let mut t: HostTier = HostTier::new(1024, 2048).expect("构造");
        let a = t.acquire().expect("块 A");
        let b = t.acquire().expect("块 B");
        t.put(ChainHash::new(1), a).expect("驻留 A");
        t.put(ChainHash::new(2), b).expect("驻留 B");
        assert_eq!(t.used_bytes(), 2048);
        // 第三块:预算满 ⇒ 响亮拒(必须先驱逐)
        assert!(matches!(t.acquire(), Err(StoreError::Exhausted(_))));
        // 驱逐一块 ⇒ 有位置了
        let freed = t.evict(ChainHash::new(1)).expect("驱逐");
        assert_eq!(freed, 1024);
        assert_eq!(t.used_bytes(), 1024);
        assert_eq!(t.free_bytes(), 1024, "缓冲进小块池(可复用)");
        assert_eq!(t.evicted_bytes(), 1024);
        let c = t.acquire().expect("复用小块池");
        t.put(ChainHash::new(3), c).expect("驻留 C");
        assert_eq!(t.resident_blocks(), vec![ChainHash::new(2), ChainHash::new(3)]);
    }

    #[test]
    fn free_list_is_capped_so_ram_is_really_returned() {
        let mut t: HostTier = HostTier::new(512, 64 * 1024).expect("构造");
        for i in 0..32u64 {
            let b = t.acquire().expect("块");
            t.put(ChainHash::new(i), b).expect("驻留");
        }
        for i in 0..32u64 {
            t.evict(ChainHash::new(i)).expect("驱逐");
        }
        assert!(t.free_bytes() <= DEFAULT_FREE_CAP * 512, "小块池必须有上限");
        assert_eq!(t.used_bytes(), 0);
    }

    /// **键重挂**(前缀无感删除):内容零搬运;两条响亮拒各一例。
    #[test]
    fn rekey_moves_content_without_copy() {
        let mut t: HostTier = HostTier::new(256, 4096).expect("构造");
        let b = t.acquire().expect("块");
        t.put(ChainHash::new(1), b).expect("驻留");
        assert!(
            matches!(
                t.rekey(ChainHash::new(9), ChainHash::new(2)),
                Err(StoreError::Unknown(_))
            ),
            "不在账 ⇒ 拒(不许猜)"
        );
        let b2 = t.acquire().expect("块");
        t.put(ChainHash::new(2), b2).expect("驻留");
        assert!(
            matches!(
                t.rekey(ChainHash::new(1), ChainHash::new(2)),
                Err(StoreError::Invariant(_))
            ),
            "目标已在账 ⇒ 拒(不许覆盖)"
        );
        t.rekey(ChainHash::new(1), ChainHash::new(3)).expect("重挂");
        assert_eq!(
            t.resident_blocks(),
            vec![ChainHash::new(2), ChainHash::new(3)],
            "旧键消失、新键在场"
        );
        assert_eq!(t.used_bytes(), 512, "字节随账走(没有复制)");
    }

    #[test]
    fn double_put_and_unknown_take_are_loud() {
        let mut t: HostTier = HostTier::new(256, 4096).expect("构造");
        let b = t.acquire().expect("块");
        t.put(ChainHash::new(9), b).expect("驻留");
        let b2 = t.acquire().expect("块");
        assert!(matches!(t.put(ChainHash::new(9), b2), Err(StoreError::Invariant(_))));
        assert!(matches!(t.take(ChainHash::new(77)), Err(StoreError::Unknown(_))));
    }
}

//! 磁盘层(S6;设计卷 §3.5「磁盘 D = 缓存目录配额」/ §15.4 磁盘 connector)。
//!
//! # 位置:三层的最后一层(**store 零 I/O**)
//!
//! 层级次序(GPU → 宿主 → 磁盘)里,磁盘是**内容不丢的最后一站**:
//! 只有磁盘也放不下时才真丢(§12.3)。本文件**只做账与计划**:
//!
//! ```text
//! plan_spill(key)  -> SpillPlan { tmp, path, bytes }   调用方写 tmp + rename(原子)
//! commit_spill(&p)                                      落账(内容已写定)
//! plan_fill(key)   -> FillPlan  { path, bytes }         调用方读盘
//! touch(key)                                            命中/回填 = 触达
//! plan_evict(need, protect) -> Vec<(K, PathBuf)>        只出计划(最老先)
//! commit_evict(keys)                                    出账(文件删除由调用方)
//! adopt(entries)                                        启动扫描后**回填账**(不搬内容)
//! ```
//!
//! **为什么 I/O 不在 store**(§21 裁决):文件读写会阻塞命令线程(§15.4 实测警告),
//! 必须由 engine 在 `spawn_blocking`(或专用 I/O 线程)里做 ⇒ store 出计划、
//! engine 执行、回来 `commit_*`。这与 `moves.rs` 的 dtoh/upload 同一条律。
//!
//! # 磁盘 = **权威副本**,宿主 = 缓存
//!
//! `commit_fill` **不删磁盘条目**(只触达):回灌是"读回宿主",磁盘副本继续留着 ——
//! 于是宿主再次满时还能逐回磁盘(零丢内容),而不是"读一次就没了"。
//! 磁盘条目的消失只有两条路:`commit_evict`(真删)与 `adopt` 时的大小不符隔离。
//!
//! # 启动回加载(用户 2026-10-19 裁决)
//!
//! 进程重启后磁盘上的内容仍在 ⇒ 启动时**扫目录回填账**(`adopt`),
//! 内容**惰性**读回(命中时才 `plan_fill`)—— 27B 一条快照 159 MiB,
//! 全量预读会把启动拖垮;账先回来,内容按需。

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::error::{StoreError, StoreResult};
use crate::id::{ChainHash, SnapKey};

/// 磁盘层的键:内容身份 → **文件名**(可逆)。
///
/// 名字里带腿前缀(`kv-` / `snap-`)⇒ 同一目录可混放两条腿,重启扫描不会串腿。
pub trait DiskKey: Ord + Copy + std::fmt::Debug {
    /// 文件名主干(不含 `.bin`)。
    fn stem(self) -> String;
    /// 反解析(启动扫描用);不认识的形状 = `None`(**不猜**)。
    fn parse_stem(s: &str) -> Option<Self>;
}

impl DiskKey for ChainHash {
    fn stem(self) -> String {
        format!("kv-{:016x}", self.raw())
    }
    fn parse_stem(s: &str) -> Option<Self> {
        s.strip_prefix("kv-")
            .and_then(|h| u64::from_str_radix(h, 16).ok())
            .map(ChainHash::new)
    }
}

impl DiskKey for SnapKey {
    fn stem(self) -> String {
        format!("snap-{:016x}", self.boundary().raw())
    }
    fn parse_stem(s: &str) -> Option<Self> {
        s.strip_prefix("snap-")
            .and_then(|h| u64::from_str_radix(h, 16).ok())
            .map(|v| SnapKey::at_boundary(ChainHash::new(v)))
    }
}

/// 磁盘驻留的一条(账)。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiskEntry {
    /// 内容文件路径(最终名,非 `.tmp`)。
    pub path: PathBuf,
    /// 字节数(= 逻辑块字节)。
    pub bytes: usize,
    last_use: u64,
}

/// **溢写计划**(调用方执行:写 `tmp` → `rename` 到 `path`;
/// 先写中间文件再原子改名 ⇒ 崩溃不留半块)。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpillPlanOf<K> {
    /// 逻辑块身份。
    pub key: K,
    /// 中间文件(**先写它再 rename** ⇒ 崩溃不留半块)。
    pub tmp: PathBuf,
    /// 最终路径。
    pub path: PathBuf,
    /// 字节数。
    pub bytes: usize,
}

/// **回填计划**(调用方执行:读 `path`)。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FillPlanOf<K> {
    /// 逻辑块身份。
    pub key: K,
    /// 内容路径。
    pub path: PathBuf,
    /// 字节数。
    pub bytes: usize,
}

/// 驱动计划(engine 调用):**要腾磁盘**时按序删这些文件。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiskEvictPlanOf<K> {
    /// 待删条目(最老在前)。
    pub victims: Vec<(K, PathBuf)>,
    /// 合计字节(按账算)。
    pub freed_bytes: usize,
}

impl<K> Default for DiskEvictPlanOf<K> {
    fn default() -> Self {
        Self { victims: Vec::new(), freed_bytes: 0 }
    }
}

/// 启动扫描的**回填报告**(响亮:不认识的/装不下的逐条列出)。
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AdoptReport {
    /// 成功入账条数。
    pub adopted: usize,
    /// 成功入账字节。
    pub bytes: usize,
    /// 被隔离的条目(路径 + 原因)—— **不静默**。
    pub skipped: Vec<(PathBuf, String)>,
}

impl AdoptReport {
    /// 是否完全干净(无隔离)。
    pub fn is_clean(&self) -> bool {
        self.skipped.is_empty()
    }
}

/// 磁盘层(有界 + 可驱逐 + 启动可恢复)。
///
/// 形态律:
/// 1. **有界**:`budget_bytes = 0` = 关层(任何溢写都**响亮拒**);越界同样拒,
///    由调用方先 `plan_evict` 腾位(不许悄悄超发);
/// 2. **零 I/O**:本类型不碰文件系统 —— 路径只算不写(计划面);
/// 3. **可恢复**:`adopt` 把目录里的既有内容回填成账(惰性读回)。
pub struct DiskTier<K = ChainHash> {
    root: PathBuf,
    block_bytes: usize,
    budget_bytes: usize,
    resident: BTreeMap<K, DiskEntry>,
    used: usize,
    tick: u64,
    evicted_bytes: u64,
}

impl<K: DiskKey> DiskTier<K> {
    /// 构造(不碰盘;`root` 只作路径前缀 —— 目录创建/扫描归 engine)。
    pub fn new(root: impl Into<PathBuf>, block_bytes: usize, budget_bytes: usize) -> StoreResult<Self> {
        if block_bytes == 0 {
            return Err(StoreError::Invariant("磁盘层块字节为 0".into()));
        }
        if budget_bytes != 0 && budget_bytes < block_bytes {
            return Err(StoreError::Budget(format!(
                "磁盘层预算 {budget_bytes} 装不下单块 {block_bytes}(按块粒度对齐)"
            )));
        }
        Ok(Self {
            root: root.into(),
            block_bytes,
            budget_bytes,
            resident: BTreeMap::new(),
            used: 0,
            tick: 0,
            evicted_bytes: 0,
        })
    }

    /// 内容目录(engine 创建/扫描用)。
    pub fn root(&self) -> &Path {
        &self.root
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

    /// 驻留字节(账)。
    pub fn used_bytes(&self) -> usize {
        self.used
    }

    /// 驻留条数。
    pub fn resident_count(&self) -> usize {
        self.resident.len()
    }

    /// 还能装几个块。
    pub fn free_slots(&self) -> usize {
        if self.is_disabled() {
            return 0;
        }
        self.budget_bytes.saturating_sub(self.used) / self.block_bytes
    }

    /// 累计驱逐字节(报告面)。
    pub fn evicted_bytes(&self) -> u64 {
        self.evicted_bytes
    }

    /// 一条现在在哪(不在账 = `None`)。
    pub fn path_of(&self, key: K) -> Option<&Path> {
        self.resident.get(&key).map(|e| e.path.as_path())
    }

    /// 是否在账。
    pub fn contains(&self, key: K) -> bool {
        self.resident.contains_key(&key)
    }

    /// 驻留键(诊断/报告)。
    pub fn resident_keys(&self) -> Vec<K> {
        self.resident.keys().copied().collect()
    }

    /// 最终文件名(纯计算;engine 扫描时同源 —— 只有一处规则)。
    pub fn file_name(&self, key: K) -> String {
        format!("{}.bin", key.stem())
    }

    /// 中间文件名(先写它 + rename ⇒ 崩溃不留半块)。
    pub fn tmp_name(&self, key: K) -> String {
        format!("{}.bin.tmp", key.stem())
    }

    /// 最终路径。
    pub fn path_for(&self, key: K) -> PathBuf {
        self.root.join(self.file_name(key))
    }

    /// **文件名 → 键**(启动扫描;不认识 = `None`)。
    pub fn key_of_file(name: &str) -> Option<K> {
        name.strip_suffix(".bin").and_then(K::parse_stem)
    }

    /// **启动扫描回填**(engine 读完目录后调用):把既有内容登记成账。
    ///
    /// 校验(逐条;不符 = 进 `skipped`,**不静默**):
    /// - 大小 ≠ 单块字节(半块/异形—— 写盘是原子 rename,出现即为外来物/破损);
    /// - 超预算(按扫描序装,装不下 = 隔离:配额是硬约束);
    /// - 重复键(后到者隔离)。
    pub fn adopt(&mut self, entries: Vec<(K, PathBuf, usize)>) -> AdoptReport {
        let mut rep = AdoptReport::default();
        for (key, path, bytes) in entries {
            if bytes != self.block_bytes {
                rep.skipped.push((
                    path,
                    format!("大小不符:{} ≠ 单块 {}", bytes, self.block_bytes),
                ));
                continue;
            }
            if self.resident.contains_key(&key) {
                rep.skipped.push((path, "重复键(同内容身份出现两次)".into()));
                continue;
            }
            if !self.is_disabled() && self.used + bytes > self.budget_bytes {
                rep.skipped.push((
                    path,
                    format!("超预算:{}+{} > {}", self.used, bytes, self.budget_bytes),
                ));
                continue;
            }
            self.tick += 1;
            let last_use = self.tick;
            self.used += bytes;
            self.resident.insert(key, DiskEntry { path, bytes, last_use });
            rep.adopted += 1;
            rep.bytes += bytes;
        }
        rep
    }

    /// **溢写计划**(纯 host):要写哪、能不能写。
    ///
    /// 关层/越界 = **响亮拒**(调用方应先 `plan_evict`;不许静默降级成"丢内容")。
    /// 已在账 ⇒ 幂等返回(覆盖写;调用方仍走 tmp+rename)。
    pub fn plan_spill(&self, key: K) -> StoreResult<SpillPlanOf<K>> {
        if self.is_disabled() {
            return Err(StoreError::Exhausted(
                "磁盘层已关闭(disk_cache_mb = 0)".into(),
            ));
        }
        let bytes = self.block_bytes;
        if !self.resident.contains_key(&key) && self.used + bytes > self.budget_bytes {
            return Err(StoreError::Exhausted(format!(
                "磁盘层越界:{}+{} > {}(先 plan_evict 腾位,或调大 disk_cache_mb)",
                self.used, bytes, self.budget_bytes
            )));
        }
        Ok(SpillPlanOf {
            key,
            tmp: self.root.join(self.tmp_name(key)),
            path: self.path_for(key),
            bytes,
        })
    }

    /// **落账**(调用方已把内容写到 `path`)。幂等(同键覆盖只触达)。
    pub fn commit_spill(&mut self, plan: &SpillPlanOf<K>) -> StoreResult<()> {
        if plan.bytes != self.block_bytes {
            return Err(StoreError::Invariant(format!(
                "溢写计划字节 {} ≠ 单块 {}",
                plan.bytes, self.block_bytes
            )));
        }
        self.tick += 1;
        let last_use = self.tick;
        match self.resident.get_mut(&plan.key) {
            Some(e) => {
                e.last_use = last_use;
                e.path = plan.path.clone();
            }
            None => {
                if !self.is_disabled() && self.used + plan.bytes > self.budget_bytes {
                    return Err(StoreError::Exhausted(format!(
                        "磁盘层越界(commit 重验):{}+{} > {}",
                        self.used, plan.bytes, self.budget_bytes
                    )));
                }
                self.used += plan.bytes;
                self.resident.insert(
                    plan.key,
                    DiskEntry {
                        path: plan.path.clone(),
                        bytes: plan.bytes,
                        last_use,
                    },
                );
            }
        }
        Ok(())
    }

    /// 溢写失败回滚(调用方删 tmp;本层无账变化 —— 幂等)。
    pub fn abort_spill(&self, _plan: &SpillPlanOf<K>) {}

    /// **回填计划**(读盘由调用方做;不在账 = 响亮拒)。
    pub fn plan_fill(&self, key: K) -> StoreResult<FillPlanOf<K>> {
        let e = self
            .resident
            .get(&key)
            .ok_or_else(|| StoreError::Unknown(format!("{key:?} 不在磁盘层")))?;
        Ok(FillPlanOf {
            key,
            path: e.path.clone(),
            bytes: e.bytes,
        })
    }

    /// **读回完成**(触达;**不删条目** —— 磁盘是权威副本,见模块头注)。
    pub fn commit_fill(&mut self, key: K) -> StoreResult<()> {
        self.touch(key)
    }

    /// 触达(命中/回填时更新新鲜度)。
    pub fn touch(&mut self, key: K) -> StoreResult<()> {
        self.tick += 1;
        let last_use = self.tick;
        let e = self
            .resident
            .get_mut(&key)
            .ok_or_else(|| StoreError::Unknown(format!("{key:?} 不在磁盘层")))?;
        e.last_use = last_use;
        Ok(())
    }

    /// **驱逐计划**(纯 host;最老在前,`protect` 绝不入列)。
    pub fn plan_evict(&self, need_bytes: usize, protect: &[K]) -> DiskEvictPlanOf<K> {
        let mut cands: Vec<(K, u64)> = self
            .resident
            .iter()
            .filter(|(k, _)| !protect.contains(k))
            .map(|(k, e)| (*k, e.last_use))
            .collect();
        cands.sort_by_key(|(_, t)| *t);
        let mut plan = DiskEvictPlanOf::default();
        for (k, _) in cands {
            if plan.freed_bytes >= need_bytes {
                break;
            }
            if let Some(e) = self.resident.get(&k) {
                plan.victims.push((k, e.path.clone()));
                plan.freed_bytes += e.bytes;
            }
        }
        plan
    }

    /// **出账**(调用方已删文件)。返回释放字节。
    pub fn commit_evict(&mut self, keys: &[K]) -> usize {
        let mut freed = 0usize;
        for k in keys {
            if let Some(e) = self.resident.remove(k) {
                self.used = self.used.saturating_sub(e.bytes);
                freed += e.bytes;
                self.evicted_bytes += e.bytes as u64;
            }
        }
        freed
    }

    /// 报告(诊断面)。
    pub fn stats(&self) -> DiskStats {
        DiskStats {
            root: self.root.display().to_string(),
            budget_bytes: self.budget_bytes,
            used_bytes: self.used,
            entries: self.resident.len(),
            evicted_bytes: self.evicted_bytes,
        }
    }
}

/// 磁盘层报告。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiskStats {
    /// 内容目录。
    pub root: String,
    /// 配额(0 = 关层)。
    pub budget_bytes: usize,
    /// 账上驻留字节。
    pub used_bytes: usize,
    /// 账上条数。
    pub entries: usize,
    /// 累计驱逐字节。
    pub evicted_bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    const BLK: usize = 1 << 20;

    fn tier(budget: usize) -> DiskTier<ChainHash> {
        DiskTier::new("/tmp/owl-disk-test", BLK, budget).expect("构造")
    }

    #[test]
    fn file_names_round_trip_per_leg() {
        let c = ChainHash::new(0xDEAD_BEEF);
        assert_eq!(c.stem(), "kv-00000000deadbeef");
        assert_eq!(ChainHash::parse_stem(&c.stem()), Some(c));
        assert_eq!(DiskTier::<ChainHash>::key_of_file("kv-00000000deadbeef.bin"), Some(c));
        assert_eq!(DiskTier::<ChainHash>::key_of_file("kv-00000000deadbeef.bin.tmp"), None);
        let s = SnapKey::at_boundary(ChainHash::new(7));
        assert_eq!(s.stem(), "snap-0000000000000007");
        assert_eq!(DiskTier::<SnapKey>::key_of_file("snap-0000000000000007.bin"), Some(s));
        // 腿前缀不串(kv 文件不能被 snap 层解析)
        assert_eq!(DiskTier::<SnapKey>::key_of_file("kv-0000000000000007.bin"), None);
    }

    #[test]
    fn disabled_tier_rejects_spill_loudly() {
        let d = tier(0);
        assert!(d.is_disabled());
        assert!(matches!(d.plan_spill(ChainHash::new(1)), Err(StoreError::Exhausted(_))));
    }

    #[test]
    fn budget_is_enforced_and_evict_plan_is_lru() {
        let mut d = tier(2 * BLK);
        for i in 1..=2u64 {
            let p = d.plan_spill(ChainHash::new(i)).expect("计划");
            d.commit_spill(&p).expect("落账");
        }
        // 第三块:越界 ⇒ 响亮拒
        assert!(matches!(d.plan_spill(ChainHash::new(3)), Err(StoreError::Exhausted(_))));
        // 触达 1(它变新)⇒ 最老 = 2
        d.touch(ChainHash::new(1)).expect("触达");
        let plan = d.plan_evict(BLK, &[]);
        assert_eq!(plan.victims.len(), 1);
        assert_eq!(plan.victims[0].0, ChainHash::new(2), "最老先出");
        assert_eq!(plan.freed_bytes, BLK);
        let freed = d.commit_evict(&[plan.victims[0].0]);
        assert_eq!(freed, BLK);
        assert_eq!(d.used_bytes(), BLK);
        assert_eq!(d.evicted_bytes(), BLK as u64);
        // 腾位后第三块能装
        let p = d.plan_spill(ChainHash::new(3)).expect("腾位后可写");
        d.commit_spill(&p).expect("落账");
        assert_eq!(d.resident_count(), 2);
    }

    #[test]
    fn protect_wins_over_evict_plan() {
        let mut d = tier(3 * BLK);
        for i in 1..=3u64 {
            let p = d.plan_spill(ChainHash::new(i)).expect("计划");
            d.commit_spill(&p).expect("落账");
        }
        let plan = d.plan_evict(usize::MAX, &[ChainHash::new(1), ChainHash::new(2)]);
        assert_eq!(plan.victims.len(), 1, "只动未被 protect 的");
        assert_eq!(plan.victims[0].0, ChainHash::new(3));
    }

    /// **启动回加载**:扫描回填账;大小不符/超预算/重复 = **隔离并列出**(不静默)。
    #[test]
    fn adopt_restores_ledger_and_isolates_bad_entries() {
        let mut d = tier(2 * BLK);
        let rep = d.adopt(vec![
            (ChainHash::new(1), PathBuf::from("/tmp/owl-disk-test/kv-0000000000000001.bin"), BLK),
            (ChainHash::new(2), PathBuf::from("/tmp/owl-disk-test/kv-0000000000000002.bin"), BLK),
            // 半块(崩在 rename 前的残件不该出现;出现即隔离)
            (ChainHash::new(3), PathBuf::from("/tmp/owl-disk-test/kv-0000000000000003.bin"), BLK / 2),
            // 超预算(配额是硬约束)
            (ChainHash::new(4), PathBuf::from("/tmp/owl-disk-test/kv-0000000000000004.bin"), BLK),
            // 重复键
            (ChainHash::new(1), PathBuf::from("/tmp/owl-disk-test/dup.bin"), BLK),
        ]);
        assert_eq!(rep.adopted, 2);
        assert_eq!(rep.bytes, 2 * BLK);
        assert_eq!(rep.skipped.len(), 3, "{:?}", rep.skipped);
        assert!(!rep.is_clean());
        assert_eq!(d.used_bytes(), 2 * BLK);
        // 隔离的条目**不入账** ⇒ 后续 plan 也看不见
        assert!(d.plan_fill(ChainHash::new(3)).is_err());
    }

    /// **回填 = 触达,不删条目**(磁盘是权威副本);随后还能再逐回磁盘。
    #[test]
    fn fill_keeps_the_disk_copy() {
        let mut d = tier(4 * BLK);
        let p = d.plan_spill(ChainHash::new(9)).expect("计划");
        d.commit_spill(&p).expect("落账");
        d.commit_fill(ChainHash::new(9)).expect("读回");
        assert!(d.contains(ChainHash::new(9)), "读回后磁盘副本仍在");
        assert_eq!(d.used_bytes(), BLK);
        // 幂等:重复溢写同键不重复计账
        let p2 = d.plan_spill(ChainHash::new(9)).expect("再写");
        d.commit_spill(&p2).expect("落账");
        assert_eq!(d.used_bytes(), BLK, "同键覆盖不重复记账");
        assert_eq!(d.resident_count(), 1);
    }
}

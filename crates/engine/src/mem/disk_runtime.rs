//! **磁盘 I/O 运行层（S6）**：store 出计划/记账，这里做真 I/O。
//!
//! 本模块从 `RunningEngine` 抽出的独立存储服务：
//! - 只持有磁盘配额（KV / 快照两腿）与重试/腾位策略；
//! - store 侧状态一律经 [`owl_store::Rel`] 的语义方法访问；
//! - **不持有** `SessionBank` / 设备池 / 选择层 —— 调用方把 `&mut Rel` 传进来。
//!
//! 边界：`owl-store` 零 I/O（设计卷 §21）；真正写盘/读盘/删文件在
//! [`crate::mem::disk_io`]。因此本结构是“I/O 编排”，不是第二本账。

use std::path::PathBuf;

use owl_iface::contract::ModelError;
use owl_store::{ChainHash, HostBlock, Rel, SnapKey, SpillPlanOf, StoreError, Tier};

use crate::mem::disk_io;
use crate::Result;

/// checkpoint 写盘的一次计数（设备 D2H / 宿主 clone 分开记）。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SpillStats {
    /// 从设备只读 D2H 后写盘的块数。
    pub dev_blocks: usize,
    /// 从宿主副本直接写盘的块数。
    pub host_blocks: usize,
}

/// 磁盘 I/O 运行器（boot 一次装配配额；运行期只持“待删文件”队列）。
pub(crate) struct DiskRuntime {
    /// store 已出账、但文件删除必须由引擎做的路径。
    ///
    /// `owl-store` 零 I/O：`drop_snap_asset` / `register_snap_captured`
    /// 这类纯 host 面只能把盘路径交出来；引擎在下一个异步安全点统一
    /// `unlink`。队列不能直接丢掉，否则就是盘上死文件。
    pending_unlinks: Vec<PathBuf>,
}

impl DiskRuntime {
    /// 装配（配额/计划/记账都归 `owl-store`）。
    pub(crate) fn new() -> Self {
        Self {
            pending_unlinks: Vec::new(),
        }
    }

    /// 登记一条**已经出账**、但尚未从盘上删除的文件。
    pub(crate) fn enqueue_unlink(&mut self, path: PathBuf) {
        self.pending_unlinks.push(path);
    }

    /// 待删文件数（测试）。
    #[cfg(test)]
    pub(crate) fn pending_unlinks(&self) -> usize {
        self.pending_unlinks.len()
    }

    /// 删除所有已出账的盘文件。返回值 = 实际/视为删除的文件数。
    ///
    /// 必须在**任何可能写同一路径的盘操作之前**调用：同名快照重拍时，
    /// 旧文件还没 unlink 就被新 `write+rename` 覆盖，再删就会误删新文件。
    pub(crate) async fn flush_unlinks(&mut self) -> usize {
        if self.pending_unlinks.is_empty() {
            return 0;
        }
        let paths = std::mem::take(&mut self.pending_unlinks);
        let want = paths.len();
        let n = disk_io::remove_files(paths).await;
        if n < want {
            owl_shared::log_warn!(
                "store",
                "盘上死文件清理未完成:计划 {want} / 完成 {n}(文件仍在,下轮再试)"
            );
        }
        owl_shared::metrics::with_metrics_store(|m| {
            m.counter_add("store.disk.unlink", n as u64, file!(), line!());
        });
        n
    }

    /// **KV 溢写**（宿主 → 磁盘；内容移动）。返回 `true` = 真搬了。
    ///
    /// - 不在宿主（已在盘/在设备）= 幂等跳过；
    /// - 盘满 ⇒ 先按 store 的驱逐计划腾位（删文件 + 出账），再试一次；
    /// - 写盘失败 ⇒ **内容放回宿主**（不丢）+ 响亮打点 ⇒ `Ok(false)`。
    pub(crate) async fn spill_kv(
        &self,
        rel: &mut Rel,
        chain: ChainHash,
        active_sessions: &[u64],
    ) -> Result<bool> {
        if !rel.disk_enabled() || rel.kv_disk_contains(chain) {
            return Ok(false);
        }
        let buf = match rel.host_take(chain) {
            Ok(b) => b,
            Err(StoreError::Unknown(_)) => return Ok(false), // 不在宿主
            Err(e) => return Err(e.into()),
        };
        let plan = match self.plan_spill_kv(rel, chain, active_sessions).await? {
            Some(p) => p,
            None => {
                // 盘装不下（腾完也不够）⇒ 内容还回宿主：本单该条不出票。
                // **运行期盘满不伪造 Drop**：下一拍 `plan_capacity` 仍会按
                // 统一 Verdict 决定是否丢内容；这里只把"盘满挡路"记成指标。
                owl_shared::counter_inc!("store.disk.spill_full");
                rel.host_put(chain, buf)?;
                return Ok(false);
            }
        };
        match disk_io::write_block(plan.path.clone(), plan.tmp.clone(), buf.into_bytes()).await {
            Ok(_) => {
                rel.kv_disk_commit_spill(&plan)?;
                owl_shared::metrics::with_metrics_store(|m| {
                    m.counter_add("store.disk.spill", 1, file!(), line!());
                });
                Ok(true)
            }
            Err(f) => {
                // 内容所有权被带回 ⇒ 放回宿主（溢写失败 ≠ 丢内容）
                rel.host_put(chain, HostBlock::from_bytes(f.bytes))?;
                owl_shared::log_warn!("store", "KV 溢写失败(内容已回宿主):{}", f.err);
                owl_shared::metrics::with_metrics_store(|m| {
                    m.counter_add("store.disk.spill_fail", 1, file!(), line!());
                });
                Ok(false)
            }
        }
    }

    /// **按字节写盘**（checkpoint 的复制语义；写失败时副本直接丢，原驻留仍在）。
    pub(crate) async fn spill_bytes(
        &self,
        rel: &mut Rel,
        chain: ChainHash,
        bytes: Vec<u8>,
        active_sessions: &[u64],
    ) -> Result<bool> {
        if !rel.disk_enabled() || rel.kv_disk_contains(chain) {
            return Ok(false); // 幂等：已在盘
        }
        let plan = match self.plan_spill_kv(rel, chain, active_sessions).await? {
            Some(p) => p,
            None => {
                // 复制语义：原驻留仍在，盘满只停本单；Drop 由下一拍压力计划出。
                owl_shared::counter_inc!("store.disk.checkpoint.full");
                return Ok(false);
            }
        };
        match disk_io::write_block(plan.path.clone(), plan.tmp.clone(), bytes).await {
            Ok(_) => {
                rel.kv_disk_commit_spill(&plan)?;
                owl_shared::metrics::with_metrics_store(|m| {
                    m.counter_add("store.disk.checkpoint.block", 1, file!(), line!());
                });
                Ok(true)
            }
            Err(f) => {
                // 我们持有的是**副本**（设备/宿主原驻留还在）⇒ 内容不丢，如实告警
                owl_shared::log_warn!(
                    "store",
                    "KV checkpoint 写盘失败(副本可弃;链 {chain:?}):{}",
                    f.err
                );
                owl_shared::metrics::with_metrics_store(|m| {
                    m.counter_add("store.disk.checkpoint.fail", 1, file!(), line!());
                });
                Ok(false)
            }
        }
    }

    /// **checkpoint 宿主块批量写盘**：clone 字节（复制语义），失败/配额到顶即停。
    /// 返回计数写进 `out`；`false` 的块不计数，原宿主副本不动。
    pub(crate) async fn spill_host_blocks(
        &self,
        rel: &mut Rel,
        host: &[ChainHash],
        active_sessions: &[u64],
        out: &mut SpillStats,
    ) -> Result<()> {
        for h in host {
            let Some(bytes) = rel.host_block(*h).map(|b| b.as_slice().to_vec()) else {
                break;
            };
            if self.spill_bytes(rel, *h, bytes, active_sessions).await? {
                out.host_blocks += 1;
            } else {
                break;
            }
        }
        Ok(())
    }

    /// **checkpoint 设备块写盘**：调用方已完成 D2H 交换并交出 `table`；
    /// 这里逐槽读、写盘、计数。缓冲由 `recycle_read_plan` 统一归还。
    pub(crate) async fn spill_device_table(
        &self,
        rel: &mut Rel,
        chains: &[ChainHash],
        table: &owl_store::SlotTable,
        block_bytes: usize,
        active_sessions: &[u64],
        out: &mut SpillStats,
    ) -> Result<()> {
        use owl_iface::contract::Connector as _;
        for (i, h) in chains.iter().enumerate() {
            let mut buf = vec![0u8; block_bytes];
            if let Err(e) = table.get(owl_iface::contract::ConnSlot(i as u64), 0, &mut buf) {
                owl_shared::log_warn!("store", "KV checkpoint 读槽 {i} 失败:{e}");
                break;
            }
            if !self.spill_bytes(rel, *h, buf, active_sessions).await? {
                break;
            }
            out.dev_blocks += 1;
        }
        Ok(())
    }

    /// **快照溢写**（状态腿宿主 → 磁盘）。语义同 [`Self::spill_kv`]。
    pub(crate) async fn spill_snap(
        &mut self,
        rel: &mut Rel,
        key: SnapKey,
        active_sessions: &[u64],
    ) -> Result<bool> {
        // 先清掉旧身份的死文件：同一路径可能马上被本次写入复用。
        self.flush_unlinks().await;
        if !rel.snap_disk_enabled() || rel.snap_disk_contains(key) {
            return Ok(false);
        }
        let buf = match rel.snap_host_take(key) {
            Ok(b) => b,
            Err(StoreError::Unknown(_)) => return Ok(false),
            Err(e) => return Err(e.into()),
        };
        let mut plan = None;
        // 三轮：① 直接 plan ② 账内 LRU 腾位 ③ 清未入账后重试
        for attempt in 0..3 {
            match rel.snap_disk_plan_spill(key) {
                Ok(p) => {
                    plan = Some(p);
                    break;
                }
                Err(StoreError::Exhausted(_)) if attempt < 2 => {
                    let block =
                        rel.snap_disk_block_bytes()
                            .ok_or_else(|| ModelError::EngineInvariant {
                                what: "快照溢写: snap_disk_enabled 但块字节缺失".into(),
                            })?;
                    let over = rel
                        .snap_disk_used_bytes()
                        .saturating_sub(rel.snap_disk_budget_bytes());
                    let need = over.saturating_add(block);
                    let evp = rel.snap_disk_plan_evict_sessions(need, &[], active_sessions);
                    if evp.victims.is_empty() {
                        // 账内腾不动 ⇒ 清未入账（坏尺寸/重复/孤儿；删除时机 = 写盘时）
                        let keep = rel.snap_disk_live_paths();
                        let root = rel
                            .snap_disk_root()
                            .ok_or_else(|| ModelError::EngineInvariant {
                                what: "快照溢写: snap_disk_enabled 但根目录缺失".into(),
                            })?
                            .to_path_buf();
                        let sw = disk_io::sweep_unaccounted(&root, &keep);
                        if sw.removed > 0 {
                            owl_shared::log_warn!(
                                "store",
                                "快照盘腾位:清掉未入账文件 {} 个 / {} MiB",
                                sw.removed,
                                sw.bytes_freed >> 20
                            );
                            continue; // 腾出来了一点 ⇒ 回 for 重试 plan_spill
                        }
                        break;
                    }
                    let paths: Vec<PathBuf> = evp.victims.iter().map(|(_, p)| p.clone()).collect();
                    let keys: Vec<SnapKey> = evp.victims.iter().map(|(k, _)| *k).collect();
                    disk_io::remove_files(paths).await;
                    rel.snap_disk_commit_evict(&keys)?;
                }
                Err(_) => break,
            }
        }
        let Some(plan) = plan else {
            owl_shared::counter_inc!("store.disk.snap.spill_full");
            rel.snap_host_put(key, buf)?;
            return Ok(false);
        };
        match disk_io::write_block(plan.path.clone(), plan.tmp.clone(), buf.into_bytes()).await {
            Ok(_) => {
                rel.snap_disk_commit_spill(&plan)?;
                rel.set_snap_tier(key, Tier::Disk)?;
                owl_shared::metrics::with_metrics_store(|m| {
                    m.counter_add("store.disk.snap.spill", 1, file!(), line!());
                });
                Ok(true)
            }
            Err(f) => {
                rel.snap_host_put(key, HostBlock::from_bytes(f.bytes))?;
                owl_shared::log_warn!("store", "快照溢写失败(内容已回宿主):{}", f.err);
                Ok(false)
            }
        }
    }

    /// **KV 回填**（磁盘 → 宿主；S6）。返回 `true` = 真读了盘。
    pub(crate) async fn fill_kv(&self, rel: &mut Rel, chain: ChainHash) -> Result<bool> {
        if !rel.kv_disk_contains(chain) {
            return Ok(false);
        }
        if rel.host_contains(chain) {
            let _ = rel.kv_disk_touch(chain);
            return Ok(false); // 已在宿主（盘副本仍在）
        }
        let plan = rel.kv_disk_plan_fill(chain)?;
        let bytes = disk_io::read_block(plan.path.clone(), plan.bytes)
            .await
            .map_err(owl_iface::contract::ModelError::Msg)?;
        // 宿主满 ⇒ 先逐最冷再落（与 offload 同一政策）
        let r0 = rel.host_relief_blocks();
        rel.host_fill(chain, HostBlock::from_bytes(bytes))?;
        let relief = rel.host_relief_blocks().saturating_sub(r0);
        if relief > 0 {
            owl_shared::metrics::with_metrics_store(|m| {
                m.counter_add("store.host.relief_blocks", relief, file!(), line!());
            });
            owl_shared::log_warn!(
                "store",
                "宿主让位:磁盘回填前先逐最冷 {relief} 块(链 {chain:?})"
            );
        }
        rel.kv_disk_commit_fill(chain)?;
        owl_shared::metrics::with_metrics_store(|m| {
            m.counter_add("store.disk.fill", 1, file!(), line!());
        });
        Ok(true)
    }

    /// **一批 KV 回填**（磁盘 → 宿主；整窗一次线程跳）。
    pub(crate) async fn fill_kv_window(
        &self,
        rel: &mut Rel,
        chains: &[ChainHash],
    ) -> Result<usize> {
        if !rel.disk_enabled() || chains.is_empty() {
            return Ok(0);
        }
        // 判：盘上有 + 宿主没有 ⇒ 要读
        let mut todo: Vec<(ChainHash, PathBuf, usize)> = Vec::new();
        for c in chains {
            if !rel.kv_disk_contains(*c) || rel.host_contains(*c) {
                continue;
            }
            let plan = rel.kv_disk_plan_fill(*c)?;
            todo.push((*c, plan.path, plan.bytes));
        }
        if todo.is_empty() {
            return Ok(0);
        }
        let items: Vec<(PathBuf, usize)> = todo.iter().map(|(_, p, b)| (p.clone(), *b)).collect();
        let bufs = disk_io::read_blocks(items).await;
        let mut n = 0usize;
        for ((c, _, _), r) in todo.iter().zip(bufs) {
            match r {
                Ok(bytes) => {
                    // 宿主满 ⇒ 先逐最冷再落（与逐块版同政策）
                    let r0 = rel.host_relief_blocks();
                    rel.host_fill(*c, HostBlock::from_bytes(bytes))?;
                    let relief = rel.host_relief_blocks().saturating_sub(r0);
                    if relief > 0 {
                        owl_shared::metrics::with_metrics_store(|m| {
                            m.counter_add("store.host.relief_blocks", relief, file!(), line!());
                        });
                    }
                    rel.kv_disk_commit_fill(*c)?;
                    n += 1;
                }
                Err(e) => {
                    owl_shared::log_warn!("store", "回灌读盘失败(链 {c:?}):{e}");
                    return Err(owl_iface::contract::ModelError::Msg(e));
                }
            }
        }
        Ok(n)
    }

    /// **快照回填**（磁盘 → 宿主；S6）。
    pub(crate) async fn fill_snap(&self, rel: &mut Rel, key: SnapKey) -> Result<bool> {
        if !rel.snap_disk_contains(key) {
            return Ok(false);
        }
        if rel.snap_host_contains(key) {
            let _ = rel.snap_disk_touch(key);
            return Ok(false);
        }
        let plan = rel.snap_disk_plan_fill(key)?;
        let bytes = disk_io::read_block(plan.path.clone(), plan.bytes)
            .await
            .map_err(owl_iface::contract::ModelError::Msg)?;
        rel.snap_host_put(key, HostBlock::from_bytes(bytes))?;
        rel.snap_disk_commit_fill(key)?;
        rel.set_snap_tier(key, Tier::Host)?;
        owl_shared::metrics::with_metrics_store(|m| {
            m.counter_add("store.disk.snap.fill", 1, file!(), line!());
        });
        Ok(true)
    }

    /// **回灌前把盘上的块读回宿主**（两个回灌口共用）。
    pub(crate) async fn ensure_host_resident(
        &self,
        rel: &mut Rel,
        chains: &[ChainHash],
    ) -> Result<usize> {
        let mut n = 0usize;
        for c in chains {
            if self.fill_kv(rel, *c).await? {
                n += 1;
            }
        }
        Ok(n)
    }

    /// **写链索引**（与内容同源；索引缺失 = 该链跨重启不可复用）。
    pub(crate) async fn write_chain_index(
        &self,
        rel: &mut Rel,
        tokens: &[u32],
        tags: &[owl_store::BlockTag],
        page: usize,
    ) -> Result<()> {
        if tokens.is_empty() || !rel.disk_enabled() {
            return Ok(());
        }
        let page = page.max(1);
        let full = tokens.len() / page;
        if full == 0 {
            return Ok(());
        }
        // 身份必须与树**同源**（带图像指纹）：否则盘索引写的是“纯文本身份”，
        // 重启 adopted 回来的含图链与活链对不上（且可能被纯文本请求误命中）。
        let Some(tail) = rel.tail_hash_tagged(tokens, full, tags) else {
            return Ok(());
        };
        let Some(root) = rel.kv_disk_root().map(|p| p.to_path_buf()) else {
            return Ok(());
        };
        let stem = owl_store::DiskKey::stem(tail);
        let path = root.join(format!("{stem}.idx"));
        // 不因“文件已在”而早退：`OWLIDX01` 的旧文件没有逐块指纹，
        // 只能用新格式原子覆盖（否则重启后图像/位置链身份对不上）。
        let tmp = root.join(format!("{stem}.idx.tmp"));
        let idx = match owl_store::ChainIndex::with_tags(
            page,
            tokens.to_vec(),
            tags.iter().take(full).copied().collect(),
        ) {
            Ok(i) => i,
            Err(e) => {
                owl_shared::log_warn!("store", "链索引构造失败(tail={tail:?}):{e}");
                return Ok(());
            }
        };
        match disk_io::write_block(path, tmp, idx.encode()).await {
            Ok(_) => {
                owl_shared::metrics::with_metrics_store(|m| {
                    m.counter_add("store.disk.index.write", 1, file!(), line!());
                });
            }
            Err(f) => {
                owl_shared::log_warn!(
                    "store",
                    "链索引写盘失败:{}(索引缺失 = 该链跨重启不可复用)",
                    f.err
                );
            }
        }
        Ok(())
    }

    /// 取 KV 溢写计划（盘满则先腾位重试；`None` = 腾不出）。
    async fn plan_spill_kv(
        &self,
        rel: &mut Rel,
        chain: ChainHash,
        active_sessions: &[u64],
    ) -> Result<Option<SpillPlanOf<ChainHash>>> {
        // 三轮：① 直接 plan ② 账内 LRU 腾位后重试 ③ 清未入账后重试
        for attempt in 0..3 {
            match rel.kv_disk_plan_spill(chain) {
                Ok(p) => return Ok(Some(p)),
                Err(StoreError::Exhausted(_)) if attempt < 2 => {
                    // 腾够**能写进这一块**的量：超配额部分 + 本次一块。
                    let block =
                        rel.kv_disk_block_bytes()
                            .ok_or_else(|| ModelError::EngineInvariant {
                                what: "KV 溢写: disk_enabled 但块字节缺失".into(),
                            })?;
                    let over = rel
                        .disk_used_bytes()
                        .saturating_sub(rel.kv_disk_budget_bytes());
                    let need = over.saturating_add(block);
                    let evp = rel.kv_disk_plan_evict_sessions(need, &[], active_sessions);
                    if evp.victims.is_empty() {
                        // 账内 LRU 腾不动 ⇒ 清“未入账”的盘上文件
                        let keep = rel.disk_live_paths();
                        let root = rel
                            .kv_disk_root()
                            .ok_or_else(|| ModelError::EngineInvariant {
                                what: "KV 溢写: disk_enabled 但根目录缺失".into(),
                            })?
                            .to_path_buf();
                        let sw = disk_io::sweep_unaccounted(&root, &keep);
                        if sw.removed > 0 {
                            owl_shared::log_warn!(
                                "store",
                                "盘腾位:清掉未入账文件 {} 个 / {} MiB(簿记外的坏尺寸/重复/孤儿)",
                                sw.removed,
                                sw.bytes_freed >> 20
                            );
                            owl_shared::metrics::with_metrics_store(|m| {
                                m.counter_add(
                                    "store.disk.sweep_unaccounted",
                                    sw.removed as u64,
                                    file!(),
                                    line!(),
                                );
                            });
                        }
                        if sw.remaining_bytes > 0 || sw.removed == 0 {
                            return Ok(None); // 真没地方了 ⇒ 放弃本次溢写（内容留在宿主）
                        }
                        continue; // 腾出来了一点，回到 for 里重试 plan_spill
                    }
                    let paths: Vec<PathBuf> = evp.victims.iter().map(|(_, p)| p.clone()).collect();
                    let keys: Vec<ChainHash> = evp.victims.iter().map(|(k, _)| *k).collect();
                    let n = disk_io::remove_files(paths).await;
                    let freed = rel.kv_disk_commit_evict(&keys);
                    owl_shared::log_info!(
                        "store",
                        "磁盘腾位:删 {n} 文件 / 出账 {freed}B(盘按 LRU 逐最老)"
                    );
                    owl_shared::metrics::with_metrics_store(|m| {
                        m.counter_add("store.disk.evict", freed as u64, file!(), line!());
                    });
                }
                Err(StoreError::Exhausted(m)) => {
                    owl_shared::log_warn!("store", "磁盘装不下:{m}");
                    return Ok(None);
                }
                Err(e) => return Err(e.into()),
            }
        }
        Ok(None)
    }
}

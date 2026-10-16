//! 调度决策词汇 + 决策逻辑(S0,2026-09-30 调度层铺开立项施工;
//! 2026-10-01 拆分:`schedule()` 自 engine.rs 收编至此,决策词汇与
//! 决策逻辑同屋)。
//!
//! **决策/执行分离**(对标 vLLM v1 `Scheduler::schedule() → SchedulerOutput`):
//! - **决策**([`RunningEngine::schedule`],本模块):只动主机侧账
//!   (队列/会话账/块账房/活跃 turn),**零设备 IO**,同步可单测 ——
//!   输出 = 本 pump 步的完整计划;
//! - **执行**(exec.rs async 面):设备侧照办(reset/快照恢复/write_bt/
//!   prefill chunk/decode step/采样),无决策权。
//!
//! ≤8 并发收缩口径(需求 §二):无抢占、无优先级、无换出 ——
//! continuous batching 值得搬的只有档位拼批(S3)。

use owl_iface::contract::{DeviceClient, ModelError};

use crate::running::RunningEngine;

/// 待回灌块的**占位槽号**(只活在 schedule → execute_begin 之间;执行器在
/// `write_bt` 之前必须把全部占位替换成真槽 —— 否则写进设备的就是坏块表)。
pub(crate) const PENDING_LOAD_PLACEHOLDER: u32 = u32::MAX;

type Result<T> = std::result::Result<T, ModelError>;

/// 一次 pump 的调度决策(执行器照办的完整计划)
pub(crate) enum SchedulerOutput {
    /// 队列排空且无活跃(调用方可安全挂起等新提交)
    Idle,
    /// 本步计划(turn 启动可与本步动作同发 —— 保持历史语义:
    /// 开 turn 与首个 chunk/decode 在同一次 pump 完成)
    Step {
        begin: Option<BeginPlan>,
        action: StepAction,
    },
}

/// turn 启动执行指令(缓存复用判定已在决策面定案,事实记入
/// ActiveTurn;执行器只收设备侧动作所需的最小四元组)
pub(crate) struct BeginPlan {
    pub session_id: u64,
    pub gdn_slot: usize,
    /// 前缀命中时 = 快照恢复键(**边界块的内容身份**;物理块号会被回收复用,
    /// 不可作键 —— 见 `owl_store::ChainCursor` 头注);GuardHit/Fresh = None
    pub restore_key: Option<owl_store::SnapKey>,
    /// Fresh(零复用)→ 执行器重置会话 GDN 格;复用路径保持状态原样
    pub reset_gdn: bool,
    /// **待回灌清单**(S3-3b):命中面里内容在宿主层的块 = `(块表下标, 内容身份)`。
    /// 执行器在**段边界**逐块 `store::onload_block` 并把返回的新槽写回块表
    /// (在 `write_bt` 之前)⇒ 决策在 schedule(纯 host),搬运在 exec(段边界)。
    pub pending_loads: Vec<(usize, owl_store::ChainHash)>,
}

/// 本步动作(相位决策;执行器无脑照办)
pub(crate) enum StepAction {
    /// 灌一个 prefill chunk(块式;is_last = 末块,末行采样出首 token)
    Prefill {
        chunk_ids: Vec<u32>,
        /// KV 行基址(增量 turn 从 cached_len 起步)
        base: usize,
        is_last: bool,
    },
    /// decode 单步(host 派生量已备齐 —— 决策面可单测的关键)
    Decode {
        /// frontier token(上一步采样输出;本步是首步时 = 末块采样结果)
        token: u32,
        /// KV 行位(= 活跃 fed;物理槽换算决策面完成)
        pos: usize,
        /// 物理槽(块链换算;legacy = pos)
        kv_slot: u32,
        gdn_slot: usize,
        /// 块链跨页增长 → 执行器重写持久块表(write_bt)
        grew: bool,
    },
    /// 投机轮(E5-M2;spec_depth > 0 时替代 Decode):块 = [anchor,
    /// d1..d_depth],槽位 depth+1 个(pos..pos+depth)
    SpecRound {
        token: u32,
        pos: usize,
        kv_slots: [u32; 9],
        gdn_slot: usize,
        grew: bool,
    },
}

impl<D: DeviceClient> RunningEngine<D> {
    /// **前缀命中的复用面**(单源:scheduler 与需求驱动保洁 §12.4 同用)。
    ///
    /// 返回 `(m, dev)`:
    /// - `m` = **可复用块数**(受两条裁剪:**完全对齐保留末块重算** /
    ///   **边界快照不在场则降一** —— 与旧 scheduler 逐条一致);
    /// - `dev` = 其中**设备已驻留**的块数(换入时**不占新槽**)。
    ///
    /// 为什么 `dev` 单独给:host 命中的块虽"命中",回灌时**照样要设备槽**
    /// ⇒ 需求侧只能减 `dev`,不能减 `m`(否则少腾槽 → 回灌时 Exhausted)。
    pub(crate) fn reuse_face(
        rel: &owl_store::Rel,
        hit: &owl_store::ChainHit,
        recall: &owl_store::RecallHit,
        prompt_len: usize,
        page: usize,
    ) -> (usize, usize) {
        let mut m = hit.matched_blocks;
        let tail = m;
        if m > 0 && m * page == prompt_len {
            m -= 1; // 完全对齐保留末块重算(非空 prefill;xinfer 同语义)
        }
        // **门面给的两腿语义(§25.2)**:尾部边界的快照**已在场**(设备或宿主
        // ——`RecallHit.snap_tier` 就是它)且没被上面的"末块重算"裁掉 ⇒
        // 复用面就是 m,不必再逐边界回查 store 的账(省 O(边界) 次查询)。
        // 这就是 `RecallMode::Exact` 的判据(块链 + 快照都在)在调度面的落点。
        if m > 0 && m == tail && recall.snap_tier.is_some() {
            let dev = hit.slots[..m].iter().filter(|s| s.is_some()).count();
            return (m, dev);
        }
        while m > 0 {
            // 快照键 = **边界块的内容身份**(物理块号会被回收复用 ⇒ 旧槽号会命中
            // 别人的状态:静默错值。store 卷 §9.1)。
            //
            // **S2(§25.2)**:在场性查 **store 的层账**(`StateStore`)——
            // 设备层与宿主层都算"在场"(宿主层在段边界回灌;`RecallMode::Exact`
            // 的判据就是这一条)。旧实现只看引擎的设备池 ⇒ 下沉后的快照被当成
            // "不在场"、白裁掉复用面。
            if rel.state.locate(owl_store::SnapKey::at_boundary(hit.hashes[m - 1])).is_some() {
                break;
            }
            m -= 1; // 无快照的边界不可复用(GDN 态对不上)
        }
        let dev = hit.slots[..m].iter().filter(|s| s.is_some()).count();
        (m, dev)
    }

    /// **prefill 本步块长**（单源：调度决策与需求驱动保洁 §12.4 同用这一份）。
    ///
    /// `None` = 本步不是 prefill（已说完 prompt，或没有活跃 turn）。
    /// 块长 = min(配置块长, 剩余);`max(1)` 保证结尾剩 1 token 也推得动。
    pub(crate) fn prefill_step_chunk(
        fed: usize,
        prompt_len: usize,
        chunk_cfg: usize,
    ) -> Option<usize> {
        let remaining = prompt_len.checked_sub(fed)?;
        if remaining == 0 {
            return None;
        }
        Some(remaining.min(chunk_cfg).max(1))
    }

    /// 调度决策(纯主机侧)：turn 启动判定(会话守卫/前缀匹配/快照
    /// 边界)+ 相位决策(prefill chunk / decode 步,host 派生量备齐)。
    /// 只动账(队列/会话账/块账房/活跃 turn),**零 await**。
    pub(crate) fn schedule(&mut self) -> Result<SchedulerOutput> {
        let mut begin = None;
        if self.turns.active.is_none() {
            // **准入排队中**(S3b §23.4 `Queue` 姿态):留队首不弹 —— 起 turn
            // 的权利在准入判决手里;弹了就会绕过判决直接撞 `ensure_for_len`。
            if let Some((waiting, _)) = self.turns.admit_wait {
                if self.turns.queue.front().map(|t| t.id) == Some(waiting) {
                    return Ok(SchedulerOutput::Idle);
                }
            }
            let Some(turn) = self.turns.queue.pop_front() else {
                return Ok(SchedulerOutput::Idle);
            };
            // **准入登记**(S3b §23.2):turn 从此占用一块保留额(实占随
            // `sync_pool_account` 在段边界更新;天花板门 count 靠它)。
            self.bank.rel.admission.admit(turn.id, 0, 0)?;
            self.turns.admit_wait = None;
            // **S2b(§28):turn 起借 GDN 行**(账在 store:行 = 活跃 turn 的资源;
            // 物理行号就是账里的行号 —— 一本账,不两份)。行耗尽 = 响亮拒
            // (`Exhausted`;正常路径由准入面 `max_active_turns` 天花板先挡)。
            let gdn_row = self.bank.rel.state.borrow_row()?;
            // E2c 复用形态:**PrefixHit(内容寻址,块链复用 + 快照恢复)/ Fresh(全量)**
            //
            // **S2(§25.1①)删掉 GuardHit 快路**:idle 会话不再持设备表 ⇒
            // "同会话原样复用块链"这条快路不再成立(表已释放),"同会话增量"
            // 由**内容寻址**接管:tokens 前缀一致 ⇒ 命中;客户端改写历史 ⇒
            // 命中自然变短(自回退全量)—— 守卫(isolation)的活由内容身份干,
            // 少一条与内容寻址并行的"账本信任"路径。
            let (cached_len, _gdn_slot, restore_key, pending_loads) = {
                let s = self.bank.sessions
                    .get_mut(turn.session_id)
                    .expect("submit 已建会话账");
                if self.pool.paged {
                    // 前缀缓存匹配(先匹配后释放旧链;内容寻址 = 跨会话)。
                    // **S3b 收尾**:经 `Rel` 门面**一次问两腿**(块链 + 快照层)
                    // —— 少一条"引擎自己分别查两处"的旁路(§10.4 律⑤)。
                    let (hit, recall) = self.bank.rel.match_prefix(&turn.prompt_ids)?;
                    // S3-3b:宿主承载格(内容在 host)不再整条拒 —— 占位 + 记进
                    // `pending_loads`,由 `execute_begin` 在段边界回灌(真槽写回,
                    // 必须在 `write_bt` 之前换掉)。
                    let (matched_raw, hashes) = (hit.matched_blocks, hit.hashes.clone());
                    let mut chain: Vec<u32> = Vec::with_capacity(hit.slots.len());
                    let mut pending_loads: Vec<(usize, owl_store::ChainHash)> = Vec::new();
                    for (i, (slot, h)) in hit.slots.iter().zip(&hashes).enumerate() {
                        match slot {
                            Some(s) => chain.push(*s),
                            None => {
                                chain.push(PENDING_LOAD_PLACEHOLDER);
                                pending_loads.push((i, *h));
                            }
                        }
                    }
                    if self.cfg.knobs.probes.debug {
                        // 在场性 = **store 的层账**(设备/宿主都算);层号一并打出
                        let snaps: Vec<&str> = hashes
                            .iter()
                            .take(4)
                            .map(|h| {
                                match self
                                    .bank
                                    .rel
                                    .state
                                    .locate(owl_store::SnapKey::at_boundary(*h))
                                {
                                    Some(owl_store::Tier::Gpu) => "dev",
                                    Some(owl_store::Tier::Host) => "host",
                                    Some(owl_store::Tier::Disk) => "disk",
                                    None => "-",
                                }
                            })
                            .collect();
                        owl_shared::log_debug!(
                            "dbg prefix",
                            "match = {matched_raw} blocks / chain = {chain:?} · snap_used={}/{} 边界快照在场(前4)={snaps:?}",
                            self.pool.snap_used(),
                            self.pool.snap_capacity()
                        );
                    }
                    // 裁剪(末块对齐 + 快照在场)= **单源** `reuse_face`
                    // (需求驱动保洁 §12.4 用同一份算式,不许各算一份)
                    let (m, _dev) = Self::reuse_face(
                        &self.bank.rel,
                        &hit,
                        &recall,
                        turn.prompt_ids.len(),
                        self.pool.page,
                    );
                    if m > 0 {
                        // 只对**已驻留**块加会话引用(宿主格的引用由回灌侧
                        // `alloc_slot` + `rebind` 建立;占位值不参与账)
                        for &b in &chain[..m] {
                            if b != PENDING_LOAD_PLACEHOLDER {
                                self.bank.rel.leg.incref(b)?;
                            }
                        }
                        pending_loads
                            .truncate(pending_loads.iter().take_while(|(i, _)| *i < m).count());
                        let mut old = std::mem::take(&mut s.block_table);
                        self.bank.rel.leg.release_table(&mut old)?;
                        s.block_table = chain[..m].to_vec();
                        // MTP 链同步重置(E5-M2b):保留前缀的 mtp 条目虽
                        // 语义可复用(同 token 同 hidden 确定性),但 M2b
                        // 无 prefill extend,重建面 = 0 起步(草稿质量降,
                        // 恒等不受影响 —— 草稿只影响速度)
                        let mut mold = std::mem::take(&mut s.mtp_block_table);
                        self.bank.leg_mtp.release_table(&mut mold)?;
                        s.reset();
                        // 链哈希游标播种:身份从缓存哈希链接管(此后 prefill/
                        // decode 的边界快照键从游标出,不再碰物理块号)
                        s.chain.seed(m, hashes[m - 1]);
                        s.cached_len = m * self.pool.page;
                        (
                            m * self.pool.page,
                            gdn_row,
                            Some(owl_store::SnapKey::at_boundary(hashes[m - 1])),
                            pending_loads,
                        )
                    } else {
                        let mut table = std::mem::take(&mut s.block_table);
                        self.bank.rel.leg.release_table(&mut table)?;
                        let mut mold = std::mem::take(&mut s.mtp_block_table);
                        self.bank.leg_mtp.release_table(&mut mold)?;
                        s.reset();
                        (0, gdn_row, None, Vec::new())
                    }
                } else {
                    let mut table = std::mem::take(&mut s.block_table);
                    self.bank.rel.leg.release_table(&mut table)?;
                    let mut mold = std::mem::take(&mut s.mtp_block_table);
                    self.bank.leg_mtp.release_table(&mut mold)?;
                    s.reset();
                    (0, gdn_row, None, Vec::new())
                }
            };
            // E2b:块表确保覆盖 prompt(增量 turn 块链已存,只长新增段)
            {
                let s = self.bank.sessions
                    .get_mut(turn.session_id)
                    .expect("submit 已建会话账");
                self.bank.rel.leg.ensure_for_len(&mut s.block_table, turn.prompt_ids.len())?;
            }
            begin = Some(BeginPlan {
                session_id: turn.session_id,
                gdn_slot: gdn_row,
                restore_key,
                reset_gdn: cached_len == 0,
                pending_loads,
            });
            let ignore_eos = turn.spec.ignore_eos;
            let opts = turn.opts.clone();
            // 采样面合并(AGT-06):请求覆写按字段并入引擎 knobs;
            // 请求未表达采样意图 ⇒ 完全走 knobs(flavor 语义不变)
            let (eff_sampler, sampler_on) = match opts.sampler {
                Some(o) if !o.is_empty() => {
                    let mut s = self.cfg.knobs.sampler;
                    if let Some(t) = o.temp {
                        s.temp = t;
                    }
                    if let Some(k) = o.topk {
                        s.topk = k;
                    }
                    if let Some(p) = o.topp {
                        s.topp = p;
                    }
                    let on = s.temp > 0.0;
                    (s, on)
                }
                _ => (self.cfg.knobs.sampler, self.cfg.knobs.sampler_enabled),
            };
            self.turns.active = Some(crate::turn::ActiveTurn {
                id: turn.id,
                session_id: turn.session_id,
                gdn_slot: gdn_row,
                ephemeral: turn.ephemeral,
                prompt_ids: turn.prompt_ids,
                max_new: turn.spec.max_new,
                ignore_eos,
                sampler: eff_sampler,
                sampler_on,
                stop: opts.stop,
                emitted: 0,
                trim_text_to: None,
                seed: opts.seed,
                out: Vec::new(),
                fed: cached_len,
                cached_start: cached_len,
                decoded: String::new(),
                spec: crate::turn::TurnSpecState {
                    // B4 分母**跨 turn 保鲜**(见 RunningEngine::bare_cost_carry_ms);
                    // 其余 spec 态仍全零 = 生命周期收口纪律不动
                    bare_cost_ewma_ms: self.spec.carry_ms,
                    bare_cost_ctx: self.spec.carry_ctx,
                    ..Default::default()
                },
            });
        }

        // 相位决策:prompt 相位(块式 prefill;W1/PF1)优先
        let act = self.turns.active.as_ref().expect("begin 或已有活跃");
        if act.fed < act.prompt_ids.len() {
            let base = act.fed;
            let remaining = act.prompt_ids.len() - base;
            // 块长:配置块长 ∩ 剩余(E1 paged 核后无 256 窗钳制;
            // 全局注意力语义,长 ctx 安全)
            let chunk = Self::prefill_step_chunk(base, act.prompt_ids.len(), self.cfg.prefill_chunk)
                .unwrap_or_else(|| remaining.max(1));
            let action = StepAction::Prefill {
                chunk_ids: act.prompt_ids[base..base + chunk].to_vec(),
                base,
                is_last: base + chunk >= act.prompt_ids.len(),
            };
            return Ok(SchedulerOutput::Step { begin, action });
        }

        // 生成相位:decode 单步 / 投机轮(host 派生量备齐 —— 物理槽/跨页增长)
        let (sid, token, pos) = {
            let act = self.turns.active.as_ref().expect("活跃");
            (act.session_id, *act.out.last().expect("生成中"), act.fed)
        };
        // B4 自适应降级(§6.24):降级态发裸 Decode(草稿池逐 token 同步);
        // 每 probe_every 步发一轮真 SpecRound 探测(m≥1 → 解除降级,
        // 见 execute_spec_round 尾态机)。非降级 = 原调度,零变化。
        // B4 三态在 ActiveTurn.spec(turn 生命周期;2026-10-10 收口)
        let spec = if self.spec.depth == 0 {
            0
        } else if !self.turns.active.as_ref().expect("活跃").spec.spec_degraded {
            self.spec.depth
        } else if self.turns.active.as_ref().expect("活跃").spec.spec_steps_degraded
            >= self.turns.active.as_ref().expect("活跃").spec.spec_probe_every
        {
            self.turns.active.as_mut().expect("活跃").spec.spec_steps_degraded = 0;
            self.spec.depth
        } else {
            0
        };
        let gdn_row = self.turns.active.as_ref().map(|a| a.gdn_slot).ok_or_else(|| {
            owl_iface::contract::ModelError::Msg("decode 相位:无活跃 turn(行号真源缺失)".into())
        })?;
        let (kv_slot, kv_slots, gdn_slot, grew) = {
            let s = self.bank.sessions.get_mut(sid).expect("账在");
            let need = pos + spec + 1;
            let before = s.block_table.len();
            self.bank.rel.leg.ensure_for_len(&mut s.block_table, need)?;
            let grew = s.block_table.len() != before;
            let page = self.pool.page;
            let slot_at = |p: usize| {
                let b = s.block_table[p / page];
                b * page as u32 + (p % page) as u32
            };
            if spec > 0 {
                // 槽表 = verify 块位(pos..pos+spec);余槽补 0(消费面
                // 自算物理槽,此表仅 decode 回退臂消费 depth+1 个)
                let mut kv_slots = [0u32; 9];
                for (i, sl) in kv_slots.iter_mut().enumerate() {
                    if i <= spec {
                        *sl = slot_at(pos + i);
                    }
                }
                (slot_at(pos), Some(kv_slots), gdn_row, grew)
            } else {
                (slot_at(pos), None, gdn_row, grew)
            }
        };
        match kv_slots {
            Some(kv_slots) => Ok(SchedulerOutput::Step {
                begin,
                action: StepAction::SpecRound { token, pos, kv_slots, gdn_slot, grew },
            }),
            None => Ok(SchedulerOutput::Step {
                begin,
                action: StepAction::Decode { token, pos, kv_slot, gdn_slot, grew },
            }),
        }
    }
}

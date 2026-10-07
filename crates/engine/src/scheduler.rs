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
    /// 前缀命中时 = 快照恢复键(链尾物理块 id);GuardHit/Fresh = None
    pub restore_key: Option<u32>,
    /// Fresh(零复用)→ 执行器重置会话 GDN 格;复用路径保持状态原样
    pub reset_gdn: bool,
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
    /// 调度决策(纯主机侧):turn 启动判定(会话守卫/前缀匹配/快照
    /// 边界)+ 相位决策(prefill chunk / decode 步,host 派生量备齐)。
    /// 只动账(队列/会话账/块账房/活跃 turn),**零 await**。
    pub(crate) fn schedule(&mut self) -> Result<SchedulerOutput> {
        let mut begin = None;
        if self.active.is_none() {
            let Some(turn) = self.queue.pop_front() else {
                return Ok(SchedulerOutput::Idle);
            };
            // E2c 复用形态:GuardHit(同会话,块链/格态原样)/
            // PrefixHit(跨会话,块链复用 + 快照恢复)/ Fresh(全量)
            let (cached_len, gdn_slot, restore_key) = {
                let s = self
                    .sessions
                    .get_mut(turn.session_id)
                    .expect("submit 已建会话账");
                // ⚠️ 新会话(空账)guard 恒真 —— cached_len == 0 必须落到
                // 前缀匹配分支(跨会话内容复用正是新会话的场景)
                if s.guard(&turn.prompt_ids) && s.cached_len > 0 {
                    (s.cached_len, s.gdn_slot, None)
                } else if self.pool.paged {
                    // 前缀缓存匹配(先匹配后释放旧链;内容寻址 = 跨会话)
                    let (mut m, chain) = self.blocks_m.match_prefix(&turn.prompt_ids);
                    if self.blocks_m.debug {
                        eprintln!("[dbg prefix] match = {m} blocks / chain = {chain:?}");
                    }
                    if m > 0 && m * self.pool.page == turn.prompt_ids.len() {
                        m -= 1; // 完全对齐保留末块重算(非空 prefill;xinfer 同语义)
                    }
                    while m > 0 {
                        let key = chain[m - 1];
                        if self.pool.has_snap(key) {
                            break;
                        }
                        m -= 1; // 无快照的边界不可复用(GDN 态对不上)
                    }
                    if m > 0 {
                        for &b in &chain[..m] {
                            self.blocks_m.incref(b);
                        }
                        let mut old = std::mem::take(&mut s.block_table);
                        self.blocks_m.release_table(&mut old);
                        s.block_table = chain[..m].to_vec();
                        // MTP 链同步重置(E5-M2b):保留前缀的 mtp 条目虽
                        // 语义可复用(同 token 同 hidden 确定性),但 M2b
                        // 无 prefill extend,重建面 = 0 起步(草稿质量降,
                        // 恒等不受影响 —— 草稿只影响速度)
                        let mut mold = std::mem::take(&mut s.mtp_block_table);
                        self.blocks_mtp.release_table(&mut mold);
                        s.reset();
                        s.cached_len = m * self.pool.page;
                        (m * self.pool.page, s.gdn_slot, Some(chain[m - 1]))
                    } else {
                        let mut table = std::mem::take(&mut s.block_table);
                        self.blocks_m.release_table(&mut table);
                        let mut mold = std::mem::take(&mut s.mtp_block_table);
                        self.blocks_mtp.release_table(&mut mold);
                        s.reset();
                        (0, s.gdn_slot, None)
                    }
                } else {
                    let mut table = std::mem::take(&mut s.block_table);
                    self.blocks_m.release_table(&mut table);
                    let mut mold = std::mem::take(&mut s.mtp_block_table);
                    self.blocks_mtp.release_table(&mut mold);
                    s.reset();
                    (0, s.gdn_slot, None)
                }
            };
            // E2b:块表确保覆盖 prompt(增量 turn 块链已存,只长新增段)
            {
                let s = self
                    .sessions
                    .get_mut(turn.session_id)
                    .expect("submit 已建会话账");
                self.blocks_m.ensure_for_len(&mut s.block_table, turn.prompt_ids.len())?;
            }
            begin = Some(BeginPlan {
                session_id: turn.session_id,
                gdn_slot,
                restore_key,
                reset_gdn: cached_len == 0,
            });
            self.active = Some(crate::running::ActiveTurn {
                id: turn.id,
                session_id: turn.session_id,
                ephemeral: turn.ephemeral,
                prompt_ids: turn.prompt_ids,
                max_new: turn.spec.max_new,
                out: Vec::new(),
                fed: cached_len,
                decoded: String::new(),
                spec: crate::running::TurnSpecState::default(),
            });
        }

        // 相位决策:prompt 相位(块式 prefill;W1/PF1)优先
        let act = self.active.as_ref().expect("begin 或已有活跃");
        if act.fed < act.prompt_ids.len() {
            let base = act.fed;
            let remaining = act.prompt_ids.len() - base;
            // 块长:配置块长 ∩ 剩余(E1 paged 核后无 256 窗钳制;
            // 全局注意力语义,长 ctx 安全)
            let chunk = remaining.min(self.cfg.prefill_chunk).max(1);
            let action = StepAction::Prefill {
                chunk_ids: act.prompt_ids[base..base + chunk].to_vec(),
                base,
                is_last: base + chunk >= act.prompt_ids.len(),
            };
            return Ok(SchedulerOutput::Step { begin, action });
        }

        // 生成相位:decode 单步 / 投机轮(host 派生量备齐 —— 物理槽/跨页增长)
        let (sid, token, pos) = {
            let act = self.active.as_ref().expect("活跃");
            (act.session_id, *act.out.last().expect("生成中"), act.fed)
        };
        // B4 自适应降级(§6.24):降级态发裸 Decode(草稿池逐 token 同步);
        // 每 probe_every 步发一轮真 SpecRound 探测(m≥1 → 解除降级,
        // 见 execute_spec_round 尾态机)。非降级 = 原调度,零变化。
        // B4 三态在 ActiveTurn.spec(turn 生命周期;2026-10-10 收口)
        let spec = if self.spec_depth == 0 {
            0
        } else if !self.active.as_ref().expect("活跃").spec.spec_degraded {
            self.spec_depth
        } else if self.active.as_ref().expect("活跃").spec.spec_steps_degraded
            >= self.active.as_ref().expect("活跃").spec.spec_probe_every
        {
            self.active.as_mut().expect("活跃").spec.spec_steps_degraded = 0;
            self.spec_depth
        } else {
            0
        };
        let (kv_slot, kv_slots, gdn_slot, grew) = {
            let s = self.sessions.get_mut(sid).expect("账在");
            let need = if spec > 0 { pos + spec + 1 } else { pos + 1 };
            let before = s.block_table.len();
            self.blocks_m.ensure_for_len(&mut s.block_table, need)?;
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
                (slot_at(pos), Some(kv_slots), s.gdn_slot, grew)
            } else {
                (slot_at(pos), None, s.gdn_slot, grew)
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

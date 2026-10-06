//! 客户端 Session:agent 连续会话(应用层词汇;2026-09-26 用户裁定)。
//!
//! # 三层词汇(引擎侧权威定义)
//!
//! - **Session(本模块)**:一个 agent 的**连续会话**(多轮 prompt)。
//!   引擎侧的感知形态 = **KV cache 区域 + 会话 token 账**:客户端每轮
//!   发全量 prompt,引擎对已缓存前缀**不重算**(增量 prefill 只喂新增
//!   token)—— 这就是"引擎感知 session"的全部含义;
//! - **Turn**(turn.rs):用户发来一条新消息(全量 prompt)→ 一个推理
//!   生命周期(提交 → 推理 → 产出文本/工具调用);
//! - **Step**:turn 内一次**推理段** —— 模型推理出工具调用 → 客户端
//!   执行工具 → 结果回传 → 继续推理 = 下一个 step。step 的 prompt 段
//!   内部再走 chunked prefill(块循环)。
//!
//! 层级:Session ⊃ Turn ⊃ Step。图捕获/回放的底层概念正名
//! [`GraphPlan`](crate::graph_plan),与本模块无关。
//!
//! # M0.5 形态
//!
//! 单槽串行:同一时刻一个活跃 session 独占 KV 槽区(0 号区起);
//! `cached_len` 记录该会话已入 KV 的前缀长度 —— 下一个 turn 的全量
//! prompt 中 `[0, cached_len)` 直接复用,**只 prefill 新增后缀**。
//!
//! **S0 接线(2026-09-26)**:SessionTable 已入 RunningEngine —— 同会话
//! turn 开启时跳过 GDN 重置、prefill 从 `cached_len` 起步;`submit(None)`
//! = 临时会话(每 turn 独立,终了即焚,行为同 M0.5)。
//! **S1 token 账 + 前缀守卫(同日精简版落地)**:`tokens` 账存已入 KV 的
//! token 原文,`guard()` 比对新 prompt 前缀 —— 失配(客户端改写历史/
//! 模板渲染不稳定)**回退全量重算**(正确性优先,缓存命中其次);
//! 真前缀匹配/radix 树随 M2 continuous batching 立项。

use crate::Result;

/// 客户端会话账(引擎侧感知面;KV cache 连续性的记账形态)
#[derive(Clone, Debug)]
pub struct AgentSession {
    /// 会话 id(引擎分配;submit 按 session_id 归队)
    pub id: u64,
    /// 已入 KV 的前缀长度(token)。KV 行 `[0, cached_len)` 为该会话
    /// 已缓存内容;下一个 turn 只 prefill `prompt[cached_len..]`
    pub cached_len: usize,
    /// KV 槽基址(M0.5 遗留;E2b 真块表后由 `block_table` 取代)
    pub slot_base: usize,
    /// 会话 token 账(S1 前缀守卫的比对基准):已入 KV 的 token 原文,
    /// 顺序 = prompt 段 + 生成段;`guard()` 逐 id 比对
    pub tokens: Vec<u32>,
    /// 物理块链(E2b 真块表):逻辑块序 → 物理块 id;块池共享,
    /// 跨 turn 存续(会话存活期间块不归还)
    pub block_table: Vec<u32>,
    /// E5:LRU 逐出账(SessionTable.tick 触达时间;越小越老)
    /// MTP 草稿链块链(E5-M2b):独立页池的第二链;mtp 模式由
    /// propose 路径 ensure_for_len,与 target 链同步跨 turn 存续
    pub mtp_block_table: Vec<u32>,
    /// GDN 状态格号(E2b 多会话隔离;格 = 每会话一格,容量 GDN_SLOTS)
    pub gdn_slot: usize,
    /// E5:LRU 触达时间(SessionTable.tick;越小越老,逐出候选)
    pub last_use: u64,
}

impl AgentSession {
    /// 新会话(零缓存起步;块表空,首次 turn 由账房分配)
    pub fn new(id: u64, slot_base: usize, gdn_slot: usize) -> Self {
        Self {
            id,
            cached_len: 0,
            slot_base,
            tokens: Vec::new(),
            block_table: Vec::new(),
            mtp_block_table: Vec::new(),
            gdn_slot,
            last_use: 0,
        }
    }

    /// 前缀守卫(S1):客户端全量重发的 prompt,其 `[0, cached_len)` 段
    /// 是否与账本逐 id 一致。true = 增量 prefill 安全;false = 失配,
    /// 调用方须 [`Self::reset`] 后全量重算(GDN 状态一并重置)。
    pub fn guard(&self, prompt: &[u32]) -> bool {
        prompt.len() >= self.cached_len && self.tokens[..self.cached_len] == prompt[..self.cached_len]
    }

    /// 账本重置(前缀失配回退):清账 + 清块表(块归还由引擎调账房,
    /// 本函数只负责账面 —— 引擎在 reset 路径调 `release_table`)
    pub fn reset(&mut self) {
        self.cached_len = 0;
        self.tokens.clear();
    }

    /// 会话 KV 预算校验:全量 prompt + 本 turn 生成长度 ≤ 槽区容量
    pub fn fits(&self, prompt_len: usize, max_new: usize, capacity: usize) -> bool {
        prompt_len.max(self.cached_len) + max_new <= capacity
    }

    /// 提交一个 turn 的增量账:全量 prompt 长度 → 引擎实际需 prefill 的
    /// 新增段(`cached_len..prompt_len`);返回 None = 全量已缓存(纯
    /// 生成续段,零 prefill)
    pub fn delta_range(&self, prompt_len: usize) -> Option<(usize, usize)> {
        if prompt_len > self.cached_len {
            Some((self.cached_len, prompt_len))
        } else {
            None
        }
    }

    /// 回填账(turn 收口):KV 此刻已含「prompt 全量 + 生成段」,GDN
    /// 状态跨 turn 连续。账本 = prompt ++ generated,`cached_len` 同步
    /// 推进到总长(下一个 turn 的增量起点)。
    pub fn commit_turn(&mut self, prompt: &[u32], generated: &[u32]) {
        self.tokens.clear();
        self.tokens.extend_from_slice(prompt);
        self.tokens.extend_from_slice(generated);
        self.cached_len = self.tokens.len();
    }
}

/// 会话注册表(E2b 多会话:会话 × 块链 × GDN 格绑定;池容量策略在
/// 引擎侧账房)
#[derive(Default, Debug)]
pub struct SessionTable {
    sessions: std::collections::HashMap<u64, AgentSession>,
    next_id: u64,
    /// E5:LRU 触达时钟(get_or_create 递增;会话账 last_use)
    tick: u64,
}

impl SessionTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// 取或建(首次 submit 隐式建会话;session_id = 0 起步)。
    /// GDN 格分配 = 扫描首个未被存活会话占用的格号;格耗尽报错
    /// (并发上限 = GDN_SLOTS,需求 ≤8 会话;引擎层 LRU 逐出后重试)。
    /// 已存在的会话直接复用(不占新格;旧实现先扫格后查存在,
    /// 格满时重提交既有会话被误拒 —— 2026-10-10 实录修复)
    pub fn get_or_create(
        &mut self,
        id: Option<u64>,
        gdn_slots: usize,
    ) -> Result<(u64, &mut AgentSession)> {
        let id = id.unwrap_or_else(|| {
            let id = self.next_id;
            self.next_id += 1;
            id
        });
        self.tick += 1;
        let tick = self.tick;
        if self.sessions.contains_key(&id) {
            // 已存在会话直接复用(不占新格;last_use 触达 = LRU 新鲜度)
            if let Some(s) = self.sessions.get_mut(&id) {
                s.last_use = tick;
                return Ok((id, s));
            }
            unreachable!("contains_key 已判定");
        }
        let next = self.next_id;
        let used: std::collections::HashSet<usize> =
            self.sessions.values().map(|s| s.gdn_slot).collect();
        let gdn_slot = (0..gdn_slots)
            .find(|g| !used.contains(g))
            .ok_or_else(|| {
                owl_iface::contract::ModelError::Msg(format!(
                    "GDN 状态格耗尽({gdn_slots} 格;会话数超并发上限)"
                ))
            })?;
        let s = self
            .sessions
            .entry(id)
            .or_insert_with(|| AgentSession::new(id, 0, gdn_slot));
        s.last_use = tick;
        self.next_id = next.max(id + 1);
        Ok((id, s))
    }

    /// E5:LRU 逐出候选(最老触达;调用方保证无活跃 turn ——
    /// actor 纪律:submit 与 pump 不相交)。None = 表空
    pub fn lru_victim(&self) -> Option<u64> {
        self.sessions
            .values()
            .min_by_key(|s| s.last_use)
            .map(|s| s.id)
    }

    pub fn get(&self, id: u64) -> Option<&AgentSession> {
        self.sessions.get(&id)
    }

    pub fn get_mut(&mut self, id: u64) -> Option<&mut AgentSession> {
        self.sessions.get_mut(&id)
    }

    /// 会话结束(客户端显式关闭;KV 区域归还 —— M0.5 单槽 = 账本清理)
    pub fn close(&mut self, id: u64) -> Result<()> {
        self.sessions
            .remove(&id)
            .map(|_| ())
            .ok_or_else(|| owl_iface::contract::ModelError::Msg(format!("session {id} 不存在")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_delta_guard_and_commit() {
        let mut st = SessionTable::new();
        let (id, s) = st.get_or_create(None, 8).expect("建会话");
        assert_eq!(id, 0);
        assert_eq!(s.gdn_slot, 0, "首会话绑 0 号格");
        assert_eq!(s.cached_len, 0);

        // turn1:全量 10 token → 零缓存守卫恒真,全量 prefill;收口账本
        // = prompt(10) + 生成(2)
        let p1: Vec<u32> = (0..10).collect();
        assert!(s.guard(&p1));
        assert_eq!(s.delta_range(10), Some((0, 10)));
        s.commit_turn(&p1, &[90, 91]);
        assert_eq!(s.cached_len, 12);
        assert_eq!(s.tokens, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 90, 91]);

        // turn2:全量 14 token(历史 12 + 新消息 2)→ 前缀命中,只 prefill
        // [12,14);收口后再推进
        let mut p2 = p1.clone();
        p2.extend([90, 91, 7, 7]);
        assert!(s.guard(&p2), "历史逐 id 一致应命中");
        assert_eq!(s.delta_range(14), Some((12, 14)));
        s.commit_turn(&p2, &[42]);
        assert_eq!(s.cached_len, 15);

        // turn3:纯生成续段(无新增 prompt)→ None
        assert_eq!(s.delta_range(15), None);
        assert!(s.fits(15, 8, 32));
        assert!(!s.fits(15, 24, 32));

        // 前缀失配(客户端改写历史/模板漂移)→ 守卫拒 → 回退全量
        let mut bad = p2.clone();
        bad[3] = 999;
        assert!(!s.guard(&bad));
        s.reset();
        assert_eq!(s.cached_len, 0);
        assert!(s.tokens.is_empty());
        assert!(s.guard(&bad), "重置后零缓存守卫恒真(全量重算)");

        // 截短的历史(prompt 比账本还短)也拒
        s.commit_turn(&p2, &[]);
        assert!(!s.guard(&p2[..8]));
    }

    #[test]
    fn session_table_lifecycle() {
        let mut st = SessionTable::new();
        let (a, sa) = st.get_or_create(Some(7), 2).expect("显式 id");
        assert_eq!(a, 7, "显式 id 直取");
        assert_eq!(sa.gdn_slot, 0);
        let (b, sb) = st.get_or_create(None, 2).expect("自动分配");
        assert_eq!(b, 8, "自动分配递增");
        assert_eq!(sb.gdn_slot, 1, "第二会话绑下一格");
        // 格耗尽:两格全占后再建报错(close 前)
        let err = st.get_or_create(None, 2).unwrap_err();
        assert!(format!("{err}").contains("GDN 状态格耗尽"));
        assert!(st.get(7).is_some());
        st.close(7).expect("close");
        assert!(st.get(7).is_none());
        assert!(st.close(7).is_err(), "重复 close 报错");
        // 复活:close 后同 id 重建 = 零缓存新会话(重新绑格)
        let (_, s) = st.get_or_create(Some(7), 2).expect("复活");
        assert_eq!(s.cached_len, 0);
        assert_eq!(s.gdn_slot, 0, "7 号 close 已释放 0 号格");
    }
    /// E5:LRU 逐出候选 + 已存在会话复用不占格(2026-10-10 实录修复)
    #[test]
    fn session_lru_victim_and_reuse() {
        let mut st = SessionTable::new();
        for i in 0..4u64 {
            st.get_or_create(Some(100 + i), 4).expect("建");
        }
        // 触达 100 与 102(变新鲜);100 是最老 → 候选
        st.get_or_create(Some(100), 4).expect("复用");
        st.get_or_create(Some(102), 4).expect("复用");
        assert_eq!(st.lru_victim(), Some(101), "101 未再触达 = 最老");
        // 复用不占新格:格满时重提交既有会话必须成功(旧实现误拒)
        st.get_or_create(Some(103), 4).expect("第 4 格");
        let err = st.get_or_create(Some(999), 4).unwrap_err();
        assert!(format!("{err}").contains("GDN 状态格耗尽"), "新会话仍受格上限");
        st.get_or_create(Some(102), 4).expect("格满复用既有会话 = 成功");
        // 逐出后新会话可入
        let v = st.lru_victim().expect("有候选");
        st.close(v).expect("close");
        st.get_or_create(Some(999), 4).expect("逐出后可建");
    }
}

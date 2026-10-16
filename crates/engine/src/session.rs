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

use owl_store::{BlockClass, BlockLeg, ChainCursor, LegConfig, PoolLayout, Rel};

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
    // (S2b:GDN 行不再是会话资源 —— 行随 **turn** 借还,见 `ActiveTurn.gdn_slot`
    //  与 `StateStore::{borrow_row, return_row}`;会话只剩身份 + token 历史)
    /// E5:LRU 触达时间(SessionTable.tick;越小越老,逐出候选)
    pub last_use: u64,
    /// **链哈希游标**(快照键真源):已哈希整块数 + 链尾内容身份。
    /// 懒惰推进(只喂新增整块)⇒ 全 turn O(n);键必须是**内容身份**
    /// (物理块号会被回收复用 —— `owl_store::ChainCursor` 头注)。
    /// 生命周期跟随链:前缀失配 `reset()` 归零;前缀命中由调度层 `seed()`。
    pub(crate) chain: ChainCursor,
}

impl AgentSession {
    /// 新会话(零缓存起步;块表空,首次 turn 由账房分配)
    pub fn new(id: u64, slot_base: usize) -> Self {
        Self {
            id,
            cached_len: 0,
            slot_base,
            tokens: Vec::new(),
            block_table: Vec::new(),
            mtp_block_table: Vec::new(),
            last_use: 0,
            chain: ChainCursor::new(),
        }
    }

    /// 前缀守卫(S1):客户端全量重发的 prompt,其 `[0, cached_len)` 段
    /// 是否与账本逐 id 一致。true = 增量 prefill 安全;false = 失配,
    /// 调用方须 [`Self::reset`] 后全量重算(GDN 状态一并重置)。
    pub fn guard(&self, prompt: &[u32]) -> bool {
        prompt.len() >= self.cached_len && self.tokens[..self.cached_len] == prompt[..self.cached_len]
    }

    /// 账本重置(前缀失配回退):清账 + 清块表(块归还由引擎调账房,
    /// 本函数只负责账面 —— 引擎在 reset 路径调 `release_table`)。
    /// 链哈希游标一并归零(新链从零起 —— 身份跟着链走)。
    pub fn reset(&mut self) {
        self.cached_len = 0;
        self.tokens.clear();
        self.chain.reset();
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
    ///
    /// **S2b(§28)**:会话**不再占 GDN 行**(行 = 活跃 turn 的资源,随 turn 借还)
    /// ⇒ 建会话不再有"格耗尽"这条路,也不再需要"LRU 逐出会话"来腾格。
    /// 能挂多少会话只受**缓存面 + 宿主预算**约束(§2.3 的目标形态)。
    pub fn get_or_create(&mut self, id: Option<u64>) -> Result<(u64, &mut AgentSession)> {
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
        let s = self.sessions.entry(id).or_insert_with(|| AgentSession::new(id, 0));
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

// ===========================================================================
// SessionBank(2026-10-14 拆分自 running.rs:RunningEngine 万能结构体
// 按状态域拆解 —— 本节收编**会话账 + 双块账房**及其复合操作)
// ===========================================================================

/// 块腿几何(boot 一次;S3-3 后 engine 只认 `BlockLeg`,`KvStore` 收成 store 内部件)。
pub(crate) struct BankPlan {
    /// 池块容量(**管理面**页数)
    pub nb: usize,
    /// 单块 token 数(页)
    pub page: usize,
    /// target 池每页字节(`StatePool::block_bytes()`)
    pub block_bytes: usize,
    /// MTP 池每页字节(`StatePool::mtp_block_bytes()`;未开 = 0)
    pub mtp_block_bytes: usize,
    /// paged 池(legacy 无块表)
    pub paged: bool,
    /// MTP 草稿链页池启用
    pub mtp: bool,
    /// 前缀缓存开关(容量 = 池半)
    pub prefix_cache: bool,
    /// target 腿宿主层预算(字节;0 = 关层,REQ C4 默认关)
    pub host_bytes: usize,
}

/// 会话 + 块账房(组合叶;RunningEngine 持有)。会话账与两套物理块链
/// (target 链 / MTP 草稿链)的变更永远两者同屋 —— 逐出/关闭/重置
/// 都是"账 + 链"复合操作,拆散在两处曾是泄漏温床。
///
/// **S3-2(2026-10-16)**:两本账(物理块引用账 + 前缀内容树)已自 engine
/// 迁入 `owl-store`(`KvStore` = 槽池 + 内容身份前缀树),engine 只留
/// 会话块表(`Vec<u32>`)与设备叶子 —— 裁决①的落点。设备叶子(池缓冲)
/// 仍归 `StatePool`;store 只发槽号与身份。
/// **S3b(2026-10-17)**:target 链从裸 `BlockLeg` 升为 **`Rel` 门面**
/// (块腿 + 状态账 + 预算 + 准入账;设计卷 §23.1)—— engine 经门面走,
/// 准入判决也从这里出。MTP 链仍为裸腿(独立页池,裁决④推后 S5)。
pub(crate) struct SessionBank {
    /// 会话账(GDN 格分配/LRU 逐出/前缀守卫)
    pub sessions: SessionTable,
    /// target 链门面(块腿 = `rel.leg`;预算/准入账同屋)
    pub rel: Rel,
    /// MTP 链块腿(独立页池,无前缀缓存;宿主面恒关 —— 裁决④推后)
    pub leg_mtp: BlockLeg,
}

impl SessionBank {
    /// boot 构造(target 链门面 + MTP 链账房;几何随池)。前缀缓存
    /// 只挂 target 链(容量 = 池半);MTP 链无前缀缓存,非 mtp 模式
    /// 0 块 —— 误用 = 结构化报错(池耗尽语义)。
    ///
    /// `inputs`/`policy` = 准入预算与姿态(engine 在池/图落位后实测组装;
    /// 见 `Engine::run`)。装配失败 = **启动失败**(A5.1:没有数字不跑)。
    pub(crate) fn new(
        plan: BankPlan,
        inputs: owl_store::BudgetInputs,
        policy: owl_store::AdmissionPolicy,
        target_layout: PoolLayout,
        mtp_layout: Option<PoolLayout>,
    ) -> Result<Self> {
        // **缓存容量 = 设备槽 + 宿主槽**(S3-3b 定案):前缀索引**跨三层** ——
        // 卸载到宿主的块仍占一个条目名额,若容量只按设备池算(旧:`nb/2`),
        // 卸载后的条目会被容量驱逐立刻淘汰 ⇒ 宿主层攒不住东西(实测:换出 2 块
        // 后条目即被逐,回灌永远不触发)。宿主面关闭时 = 旧行为。
        let host_slots = if plan.host_bytes > 0 && plan.block_bytes > 0 {
            plan.host_bytes / plan.block_bytes
        } else {
            0
        };
        // 宿主面关闭(host_slots = 0)= **旧行为逐字不变**(nb/2);开宿主面才扩容
        let cache_cap = if plan.paged && plan.prefix_cache {
            (plan.nb / 2 + host_slots).max(1)
        } else {
            0
        };
        let mut rel = Rel::open(
            inputs,
            policy,
            LegConfig {
                slots: plan.nb,
                block_tokens: plan.page,
                block_bytes: plan.block_bytes,
                cache_cap,
                host_bytes: plan.host_bytes,
                class: BlockClass::KvPage,
                label: "target",
            },
        )?;
        rel.leg.attach_pool(target_layout)?;
        let mut leg_mtp = BlockLeg::open(LegConfig {
            slots: if plan.mtp { plan.nb } else { 0 },
            block_tokens: plan.page,
            block_bytes: if plan.mtp { plan.mtp_block_bytes } else { 1 },
            cache_cap: 0,
            host_bytes: 0,
            class: BlockClass::KvPage,
            label: "mtp",
        })?;
        if let Some(l) = mtp_layout {
            leg_mtp.attach_pool(l)?;
        }
        Ok(Self { sessions: SessionTable::new(), rel, leg_mtp })
    }

    // （块腿的复合口做成**自由函数**:借用只落在 `&mut KvStore` 那一个字段上
    //   —— 调用点多处仍持 `&mut AgentSession`(会话表另行借出),方法口会整借
    //   bank 而冲突;自由函数与"字段不相交"的借用形状对齐。）

    /// 取或建(S2b 后**无逐出**):会话不再占 GDN 行 ⇒ 没有"格耗尽"这条路。
    /// 历史资源的回收由**缓存面 LRU + 宿主预算**负责(§2.3 目标形态);
    /// 会话表本身只存元数据(µs 级)。
    pub(crate) fn get_or_create(&mut self, id: Option<u64>) -> Result<u64> {
        self.sessions.get_or_create(id).map(|(id, _)| id)
    }

    /// 显式关会话(客户端声明不再续;账本清票,双链归还账房 ——
    /// 与 evict/burn 三路径同构;2026-10-15 结案拆分时挂的 FIXME:
    /// 原只还 target 链漏 MTP 页,mtp 模式下显式 close = 页池泄漏)。
    pub(crate) fn close(&mut self, id: u64) -> Result<()> {
        self.release_session_chains(id)?;
        self.sessions.close(id)
    }

    /// 临时会话烧毁(complete/abandon 共用;块链归账房 + 表移除释放格,
    /// 双链同还)。关闭失败不许静默(律 #1)。
    pub(crate) fn burn(&mut self, session_id: u64) {
        if let Err(e) = self.release_session_chains(session_id) {
            owl_shared::log_warn!("session", "临时会话 #{session_id} 链归还失败: {e}");
        }
        if let Err(e) = self.sessions.close(session_id) {
            owl_shared::log_warn!("session", "临时会话 #{session_id} 关闭失败: {e}");
        }
    }

    /// 单会话双链归还(target 链 + MTP 链;逐出/关闭/烧毁共用的
    /// 复合操作 —— 链归还必须与账本操作同屋,防单边清理)。
    /// 归还失败(槽引用账不平)**不许静默**:记 warn 再抛(律 #1)。
    fn release_session_chains(&mut self, id: u64) -> Result<()> {
        // 先摘表(借用随块结束),再归还 —— 归还口是 self 的方法(账房自肃)
        let mut table = Vec::new();
        let mut mtp_table = Vec::new();
        if let Some(s) = self.sessions.get_mut(id) {
            table = std::mem::take(&mut s.block_table);
            mtp_table = std::mem::take(&mut s.mtp_block_table);
        }
        self.rel.release_table(&mut table)?;
        self.leg_mtp.release_table(&mut mtp_table)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_delta_guard_and_commit() {
        let mut st = SessionTable::new();
        let (id, s) = st.get_or_create(None).expect("建会话");
        assert_eq!(id, 0);
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
        let (a, sa) = st.get_or_create(Some(7)).expect("显式 id");
        assert_eq!(a, 7, "显式 id 直取");
        let (b, _sb) = st.get_or_create(None).expect("自动分配");
        assert_eq!(b, 8, "自动分配递增");
        // **S2b**:建会话不再受"行上限"约束(行是 turn 的资源)——
        // 会话只受缓存面/宿主预算约束(§2.3 目标形态)
        assert!(st.get_or_create(None).is_ok(), "会话数不再有硬上限");
        assert!(st.get(7).is_some());
        st.close(7).expect("close");
        assert!(st.get(7).is_none());
        assert!(st.close(7).is_err(), "重复 close 报错");
        // 复活:close 后同 id 重建 = 零缓存新会话(重新绑格)
        let (_, s) = st.get_or_create(Some(7)).expect("复活");
        assert_eq!(s.cached_len, 0);
    }
    /// E5:LRU 逐出候选 + 已存在会话复用不占格(2026-10-10 实录修复)
    #[test]
    fn session_lru_victim_and_reuse() {
        let mut st = SessionTable::new();
        for i in 0..4u64 {
            st.get_or_create(Some(100 + i)).expect("建");
        }
        // 触达 100 与 102(变新鲜);101 是最老 → 候选(会话级 LRU 仍可用于
        // 将来的"显式上限"策略;S2b 后它不再是行耗尽的重试路径)
        st.get_or_create(Some(100)).expect("复用");
        st.get_or_create(Some(102)).expect("复用");
        assert_eq!(st.lru_victim(), Some(101), "101 未再触达 = 最老");
        st.get_or_create(Some(999)).expect("S2b:新会话不再受行上限");
        // 会话级 LRU 仍可作为**将来显式上限**的候选源(现在没人调 —— 见 §28)
        st.get_or_create(Some(102)).expect("复用既有会话 = 成功");
        // 逐出后新会话可入
        let v = st.lru_victim().expect("有候选");
        st.close(v).expect("close");
        st.get_or_create(Some(999)).expect("逐出后可建");
    }
}

#[cfg(test)]
mod bank_tests {
    use super::*;

    /// T4 结案门(2026-10-15):显式 close 必须双链同还 —— MTP 链曾漏还
    /// (与 evict/burn 不对称,mtp 模式下显式关会话 = 页池泄漏)。
    fn plan(nb: usize, page: usize, mtp: bool, prefix_cache: bool) -> BankPlan {
        BankPlan {
            nb,
            page,
            block_bytes: 1 << 20,
            mtp_block_bytes: 1 << 20,
            paged: true,
            mtp,
            prefix_cache,
            host_bytes: 0,
        }
    }

    /// 测试用装配:布局 = 单片段(只跑账,不跑搬运)
    fn bank_of(p: BankPlan) -> SessionBank {
        let target = PoolLayout {
            class: BlockClass::KvPage,
            epoch: 0,
            pieces: vec![owl_store::PieceSpec {
                buf: owl_iface::contract::Bytes::new(0, 1, owl_iface::contract::Dtype::F32, 1),
                stride_bytes: p.block_bytes,
            }],
        };
        let mtp = p.mtp.then(|| PoolLayout {
            class: BlockClass::KvPage,
            epoch: 0,
            pieces: vec![owl_store::PieceSpec {
                buf: owl_iface::contract::Bytes::new(0, 1, owl_iface::contract::Dtype::F32, 1),
                stride_bytes: p.mtp_block_bytes,
            }],
        });
        // 测试用预算:固定池档(池面 = 块字节 × 槽;准入按池墙判)
        let inputs = owl_store::BudgetInputs {
            v_card_bytes: 8 << 30,
            weights_bytes: 1 << 30,
            runtime_floor_bytes: 64 << 20,
            draft_pool_bytes: 0,
            gdn_row_bytes: 1 << 20,
            kv_bytes_per_token: (p.block_bytes / p.page.max(1)).max(1),
            host_bytes: p.host_bytes,
            disk_bytes: 0,
            kv_face_bytes: p.nb * p.block_bytes,
        };
        let policy = owl_store::AdmissionPolicy::new(4, owl_store::QueuePolicy::Reject);
        SessionBank::new(p, inputs, policy, target, mtp).expect("装配块腿")
    }

    #[test]
    fn close_releases_both_chains() {
        let mut bank = bank_of(plan(64, 8, true, false));
        let mtp_before = bank.leg_mtp.report().slots_free;
        let tgt_before = bank.rel.leg.report().slots_free;

        let (id, s) = bank.sessions.get_or_create(Some(7)).expect("建会话");
        assert_eq!(id, 7);
        // target 链 2 块(24 tok / 块 8)+ MTP 链 1 块(9 tok)
        bank.rel.leg.ensure_for_len(&mut s.block_table, 24).expect("target 块");
        bank.leg_mtp.ensure_for_len(&mut s.mtp_block_table, 9).expect("mtp 块");
        assert!(bank.leg_mtp.report().slots_free < mtp_before, "mtp 已占用");

        bank.close(id).expect("close");
        assert_eq!(bank.leg_mtp.report().slots_free, mtp_before, "MTP 链必须归还");
        assert_eq!(bank.rel.leg.report().slots_free, tgt_before, "target 链归还");
        assert!(bank.sessions.get(id).is_none(), "账本移除");
    }

    /// 无 mtp 模式回归:0 块页池,release 空表 = no-op,close 正常。
    #[test]
    fn close_without_mtp_is_noop_on_empty_chain() {
        let mut bank = bank_of(plan(32, 8, false, true));
        let (id, s) = bank.sessions.get_or_create(Some(1)).expect("建会话");
        bank.rel.leg.ensure_for_len(&mut s.block_table, 8).expect("target 块");
        bank.close(id).expect("close");
        assert!(bank.sessions.get(id).is_none());
    }
}

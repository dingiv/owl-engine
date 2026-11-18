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
//! 层级:Session ⊃ Turn ⊃ Step。图捕获/回放的底层概念(GraphPlan)与
//! 本模块无关。
//!
//! **2026-11-15 自 `owl-engine::serve::session` 下沉到 `owl-store`**：
//! KV/快照/准入的容量决策需要直接看到会话账（`tokens` / 链 / 快照资产），
//! 会话账因此与 `Rel`/`BlockLeg`/`StateStore` 同屋；engine 侧保留
//! `crate::session` 路径兼容。
//!
//! **2026-11-16 语义收口**：历史长度只有一个真源 = `tokens.len()`；
//! 不再维护第二份 `cached_len`。设备表覆盖的前缀属于 turn 执行态
//! (`ActiveTurn.cached_start` / `BeginPlan`)，由调度面持有；会话账只记历史。
//!
//! # M0.5 形态
//!
//! 单槽串行:同一时刻一个活跃 session 独占 KV 槽区(0 号区起);
//! `tokens` 记录会话历史，**历史长度唯一真源 = `tokens.len()`**；
//! 实际可复用面由 store 内容树/快照回答，不再维护第二份 `cached_len`。
//!
//! **S0 接线(2026-09-26)**:SessionTable 已入 RunningEngine —— 同会话
//! turn 开启时跳过 GDN 重置;`submit(None)` = 临时会话(每 turn 独立,终了即焚)。
//! **S1 token 账 + 前缀守卫(同日精简版落地)**:`tokens` 账存已入 KV 的
//! token 原文,`guard()` 比对新 prompt 前缀 —— 失配(客户端改写历史/
//! 模板渲染不稳定)**回退全量重算**(正确性优先,缓存命中其次);
//! 真前缀匹配/radix 树随 M2 continuous batching 立项。

use owl_iface::contract::ModelError;

use crate::{BlockClass, BlockLeg, ChainCursor, LegConfig, PoolLayout, Rel};

/// 会话账错误（契约面 `ModelError`；不是存储 I/O 的 `StoreError`）。
type Result<T> = std::result::Result<T, ModelError>;

/// 一段图像占位(**含** `<|vision_start|>` / `<|vision_end|>` 标记)的坐标账。
///
/// 两套坐标(必须同时记):
/// - `[start, end)` = **padded** 序列里的删除候选(整页对齐;连 pad 一起删);
/// - `[text_start, text_end)` = **unpadded**(客户端可见)序列里的图像项 ——
///   下一轮的删除识别在文本坐标里做。
///
/// `grid` 与 engine 的 `GridThw` 是同一个 tuple 类型；store 不依赖
/// `owl-models`，故这里直接写 `(usize, usize, usize)`。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImgSpan {
    /// padded 段首(= pad_before 块起点;**页对齐**)
    pub start: usize,
    /// padded 段尾(不含;含 pad_after 块)
    pub end: usize,
    /// unpadded 段首(客户端文本坐标)
    pub text_start: usize,
    /// unpadded 段尾(不含)
    pub text_end: usize,
    /// 段首 pad 块长度(padded `start` → 图像项起点)
    pub pad_before: usize,
    /// 段尾 pad 块长度(padded `end` 前的尾巴)
    pub pad_after: usize,
    /// 图像 patch 网格(重建 m-RoPE 位置表要用 —— 位置推进 `max(h,w)/merge`)
    pub grid: (usize, usize, usize),
    /// 客户端标记:这段将来可能被删(未标记 = 永不吸收删除)
    pub evict: bool,
    /// **图像内容指纹**(64 位;0 = 未知/无内容)。
    ///
    /// 用途 = 块身份([`crate::BlockTag`]):token 流里图像只是字面相同的
    /// 占位符,**只有它**能把"同文本不同截图"区分开。由 `submit` 侧按
    /// `VlmImageKey`(像素位哈希)盖章;`pad_image_spans` 本身不知道像素,留 0。
    pub img_tag: u64,
}

impl ImgSpan {
    /// padded 序列里的两个 pad 块(`(lo, hi)` 半开区间)。
    pub fn pad_ranges(&self) -> [(usize, usize); 2] {
        [
            (self.start - self.pad_before, self.start),
            (self.end - self.pad_after, self.end),
        ]
    }
}

/// **位置三元组**(与 `MropeCfg` 的 `[T,H,W]` 同序;i32 足够覆盖 262k)。
pub type Pos3 = [i32; 3];

/// 客户端会话账(引擎侧感知面;KV cache 连续性的记账形态)
#[derive(Clone, Debug)]
pub struct AgentSession {
    /// 会话 id(引擎分配;submit 按 session_id 归队)
    pub id: u64,
    /// KV 槽基址(M0.5 遗留;E2b 真块表后由 `block_table` 取代)
    pub slot_base: usize,
    /// 会话 token 账(S1 前缀守卫的比对基准):已入 KV 的 token 原文,
    /// 顺序 = prompt 段 + 生成段;`guard()` 逐 id 比对。
    /// **这是历史长度的唯一真源**，不再另设 `cached_len`。
    pub tokens: Vec<u32>,
    /// 物理块链(E2b 真块表):逻辑块序 → 物理块 id;块池共享。
    /// 只在 turn 内由本表持有；turn 末 `release_turn_table` 归还，
    /// 下一轮按内容/位置重新取链或全量重建，不跨 turn 钉住块。
    pub block_table: Vec<u32>,
    /// **逐块内容身份**(与 `block_table` 对齐;2026-11-13 leave-gap 同会话复用)。
    ///
    /// 长度 = `chain.blocks()`(完整块数)。leave-gap 后的 gapped 链会把
    /// **非紧凑 position 签名**折进块身份（`block_tags_pos`），因此它也能
    /// 安全发布到共享树；这里仍保留 session-local 快路：同一 turn 内表还在手
    /// 时直接用 `block_hashes`，少一次树查询。
    pub block_hashes: Vec<crate::ChainHash>,
    /// **本会话是否持有一条 gapped 链**（leave-gap 后；2026-11-13）。
    ///
    /// 只表示**当前 turn 内**持有 leave-gap 链；turn 末表一律释放，下一轮按
    /// 内容+位置签名重新 reuse，不保留跨 turn 表引用。
    pub local_chain: bool,
    /// **图像段账**(前缀无感删除;2026-10-23):上轮 prompt 里各图像占位段
    /// (含页对齐 pad)的会话坐标 + 网格 + 客户端可删标记。
    /// 生命周期跟随 `tokens`:turn 收口由 `set_img_spans` 写回,`reopen_chain`/`rebase` 清空。
    pub img_spans: Vec<ImgSpan>,
    /// **pad 区间账**(padded 坐标,升序):`unpad(tokens, pads)` = 客户端可见
    /// 序列。与 `img_spans` 同一次写回(两处各写一份 = 下一轮 diff 拿错坐标)。
    pub pads: Vec<(usize, usize)>,
    /// **逐 token 的 m-RoPE 绝对位置**(leave-gap;2026-10-25)。
    /// 与 `tokens` 等长;文本 `[p,p,p]`,图像 `[t,h,w]`。
    /// 删除图标段后幸存 token 的位置**不移动**,新尾巴从 `max_pos+1` 追加。
    /// 非 leave-gap 时写紧凑位置，作为统一账。
    pub positions: Vec<[i32; 3]>,
    /// 当前最大绝对位置(三轴取 max;空账 = 0)。
    pub max_pos: i32,
    /// 位置账代数(consolidation/rebase 时递增;缓存/诊断面用)。
    pub position_epoch: u32,
    /// E5:LRU 逐出账(SessionTable.tick 触达时间;越小越老)
    /// MTP 草稿链块链(E5-M2b):独立页池的第二链;mtp 模式由
    /// propose 路径 ensure_for_len,与 target 链同步跨 turn 存续
    pub mtp_block_table: Vec<u32>,
    // (S2b:GDN 行不再是会话资源 —— 行随 **turn** 借还,见 `ActiveTurn.gdn_slot`
    //  与 `StateStore::{borrow_row, return_row}`;会话只剩身份 + token 历史)
    /// E5:LRU 触达时间(SessionTable.tick;越小越老,逐出候选)
    pub last_use: u64,
    /// **会话尾锚点快照**（L1 资产；v4）。
    ///
    /// 最近一次 turn 结束/中断的最后完整页边界。跨 turn 保留 —— 空闲会话的
    /// 尾锚点是 **Tail 角色**（最后淘汰，但空间不足时可逐出；不是 pin）。
    /// 前缀失配 `rebase()` 时清空。
    pub tail_snap: Option<crate::SnapKey>,
    /// **尾锚点的 token offset**（页对齐；与 `tail_snap` 同源）。
    ///
    /// v4 的 Tail/History 布局按 token 位置重算，必须知道尾快照落在哪。
    pub tail_at: Option<usize>,
    /// **当前持久快照布局点**（Tail + History；页对齐 token offset，升序）。
    ///
    /// turn 收口时从 `ActiveTurn.msg_snaps` 写回，跨 turn 保留；下一轮
    /// submit 用它携带同 offset 的键并重算 Tail/History。
    pub msg_snaps: Vec<usize>,
    /// **与 `msg_snaps` 同序的角色**（Tail / History）。
    pub msg_snap_roles: Vec<crate::SnapRole>,
    /// **与 `msg_snaps` 同序的快照键**（L1 资产）。
    ///
    /// 布局重算只改角色，不搬内容；`None` = 该边界没拍/已被逐出。
    pub msg_snap_keys: Vec<Option<crate::SnapKey>>,
    /// **上一轮 prompt 段的 padded 长度**（生成段接在其后）。
    ///
    /// 内容身份只要求覆盖到这里：生成段是**模型产物**，客户端回填时按文本
    /// 重新渲染再分词，末段本来就对不上（2026-11-16 真机定谳：要求覆盖整账
    /// 会让每一轮都认不出会话 ⇒ 会话表每轮 +1）。`0` = 还没收过口 ⇒ 不可
    /// 用于身份匹配。
    pub prompt_boundary: usize,
    /// **链哈希游标**(快照键真源):已哈希整块数 + 链尾内容身份。
    /// 懒惰推进(只喂新增整块)⇒ 全 turn O(n);键必须是**内容身份**
    /// (物理块号会被回收复用 —— `crate::ChainCursor` 头注)。
    /// 生命周期跟随链:前缀失配 `rebase()` 归零;前缀命中由调度层 `seed()`。
    pub chain: ChainCursor,
}

impl AgentSession {
    /// 新会话(零缓存起步;块表空,首次 turn 由账房分配)
    pub fn new(id: u64, slot_base: usize) -> Self {
        Self {
            id,
            slot_base,
            tokens: Vec::new(),
            block_table: Vec::new(),
            block_hashes: Vec::new(),
            local_chain: false,
            mtp_block_table: Vec::new(),
            last_use: 0,
            tail_snap: None,
            tail_at: None,
            msg_snaps: Vec::new(),
            msg_snap_roles: Vec::new(),
            msg_snap_keys: Vec::new(),
            chain: ChainCursor::new(),
            prompt_boundary: 0,
            img_spans: Vec::new(),
            pads: Vec::new(),
            positions: Vec::new(),
            max_pos: 0,
            position_epoch: 0,
        }
    }

    /// 前缀守卫(S1):客户端全量重发的 prompt,其历史前缀是否与
    /// `tokens` 逐 id 一致。true = 历史可作内容寻址的提示;false = 失配。
    ///
    /// **不再把“历史长度”当“设备驻留长度”**：命中面由 store 内容树回答；
    /// 本函数只回答历史是否连续，用于诊断/旧键降级。
    pub fn guard(&self, prompt: &[u32]) -> bool {
        prompt.len() >= self.tokens.len() && self.tokens == prompt[..self.tokens.len()]
    }

    /// **上一轮 prompt 段的可见长度**（unpadded；内容身份的参考长度）。
    ///
    /// `None` = 还没收过口（`prompt_boundary = 0`）或 pad 账坏（越界/乱序/
    /// 跨出 prompt 段）—— 调用方据此把该会话排除在身份匹配之外。
    pub fn prompt_visible_len(&self) -> Option<usize> {
        if self.prompt_boundary == 0 || self.prompt_boundary > self.tokens.len() {
            return None;
        }
        let mut cursor = 0usize;
        let mut skip = 0usize;
        for &(lo, hi) in &self.pads {
            if lo < cursor || hi <= lo || hi > self.prompt_boundary {
                return None;
            }
            skip += hi - lo;
            cursor = hi;
        }
        Some(self.prompt_boundary - skip)
    }

    /// **客户端可见前缀的公共长度**(内容身份;2026-11-16)。
    ///
    /// 与 [`Self::guard`] 的分工:`guard` 比的是 **padded** 序列(它守的是
    /// KV 账);这里比的是 **unpadded**(客户端可见)序列 —— 会话身份来自
    /// 客户端发来的原文,图像 pad 是引擎侧派生量,不参与身份。
    ///
    /// 实现上**跳过 pad 区间**而不是构造 unpad 副本(零分配;会话账可达
    /// 数十万 token,每轮重建副本会把 submit 变成 O(历史) 拷贝)。
    ///
    /// 返回 = 从头逐 id 相同的可见 token 数。`pads` 账非法(越界/乱序/
    /// 空区间)⇒ 按"到此为止"处理:宁可判成不是同一条链,也不按坏账拼接。
    pub fn common_prefix_text(&self, prompt: &[u32]) -> usize {
        let mut i = 0usize;
        let mut pad = 0usize;
        let mut n = 0usize;
        while i < self.tokens.len() && n < prompt.len() {
            while pad < self.pads.len() && self.pads[pad].0 <= i {
                let (lo, hi) = self.pads[pad];
                if lo != i || hi <= lo || hi > self.tokens.len() {
                    return n;
                }
                i = hi;
                pad += 1;
            }
            if i >= self.tokens.len() || self.tokens[i] != prompt[n] {
                break;
            }
            i += 1;
            n += 1;
        }
        n
    }

    /// **首个分叉位置**(诊断;`None` = 前缀全同)。
    ///
    /// 用途:`[dbg prefix]` 取证 —— 守卫失配时告诉人"第几个 token 开始不同、
    /// 两侧各是什么",而不是只报一个 `cached_tokens=0`(那要人肉二分)。
    /// 上界 = `min(tokens.len(), prompt.len())`。
    pub fn first_divergence(&self, prompt: &[u32]) -> Option<usize> {
        let n = self.tokens.len().min(prompt.len());
        (0..n).find(|&i| self.tokens[i] != prompt[i])
    }

    /// 账本重置(链面):只清 `tokens/positions/chain` 等账面；块表由
    /// `SessionBank::reset_session_for_rebuild` / `reopen_session_chain`
    /// 在归还旧链后替换（本函数不碰设备表，避免账/链两处各清一半）。
    ///
    /// **不清 L1 资产**(`tail_snap` / `msg_snaps` / `msg_snap_keys`):
    /// 前缀命中的续写路径走 `reopen_chain`(内部调本函数),那一次不是重建;
    /// L1 资产要到 turn 收口/fresh 重建时才由 `rebase` 处理。否则每轮命中都会
    /// 把旧 tail 的 Tail 归属丢掉(旧 key 永久留在 Tail 角色,见 2026-11-13
    /// 前置评审)。
    fn reset(&mut self) {
        // 内容身份的参考边界随账清零（历史长度唯一真源 = `tokens.len()`）
        self.prompt_boundary = 0;
        self.tokens.clear();
        self.block_hashes.clear();
        self.chain.reset();
        self.img_spans.clear();
        self.pads.clear();
        self.positions.clear();
        self.max_pos = 0;
    }

    /// **重开链**（前缀命中续写）：消费旧会话，清空 token/位置/链游标，
    /// **保留 L1 快照资产**（tail/msg 边界属于同一条对话，不能因本轮重开链
    /// 而丢归属）。与 [`Self::rebase`] 的分工：rebase 是换了一条链，L1 全清；
    /// 本函数只清执行态，调用方随后把新块表/哈希写回。
    ///
    /// 返回新值，调用方必须用 [`SessionTable::put`] 替换旧账 —— 不再提供
    /// `&mut self` 的 reset 公共入口。
    pub fn reopen_chain(mut self) -> Self {
        self.reset();
        self
    }

    /// **整段重建**(前缀失配 / 位置压实):消费旧会话,返回全新账。
    ///
    /// 不再提供 `&mut self` 的 rebase:调用方必须把旧值销毁、把新值放回
    /// 会话表,避免"手里还捏着旧账引用"的半态。L1 资产属于旧链,这里
    /// 一并清掉;前缀命中的续写路径**不得**走本函数。
    pub fn rebase(mut self) -> Self {
        self.reset();
        // 旧 tail/布局点属于旧链;内容仍可由 L2/L3 按 LRU/角色回收。
        self.tail_snap = None;
        self.tail_at = None;
        self.msg_snaps.clear();
        self.msg_snap_roles.clear();
        self.msg_snap_keys.clear();
        self.local_chain = false;
        self.position_epoch = self.position_epoch.wrapping_add(1);
        self
    }

    /// **写入当前前缀的位置账**(删除/前缀命中后、turn 收口前使用)。
    ///
    /// `positions` 必须与 `tokens` 同长;越界 = 结构错(显式拒,不静默截断)。
    pub fn set_positions(&mut self, positions: &[[i32; 3]]) -> Result<()> {
        // FIXME: 不推荐可变范式。
        if positions.len() != self.tokens.len() {
            return Err(owl_iface::contract::ModelError::EngineInvariant {
                what: format!(
                    "位置账长度 {} ≠ token 账 {}（会话 {}）",
                    positions.len(),
                    self.tokens.len(),
                    self.id
                ),
            });
        }
        self.positions.clear();
        self.positions.extend_from_slice(positions);
        self.max_pos = self
            .positions
            .iter()
            .flat_map(|p| p.iter())
            .copied()
            .max()
            .unwrap_or(0);
        Ok(())
    }

    /// **图像段账写回**(turn 收口;坐标 = `tokens` 的 prompt 段坐标)。
    ///
    /// 只由 `commit_turn` 的调用点写(单一产地):spans 与 `tokens` 必须
    /// 描述同一份 prompt —— 两处各写一份 = 下一轮删除识别拿错坐标。
    pub fn set_img_spans(&mut self, spans: Vec<ImgSpan>, pads: Vec<(usize, usize)>) {
        // FIXME:  不推荐可变范式。
        self.img_spans = spans;
        self.pads = pads;
    }

    /// 会话 KV 预算校验:全量 prompt + 本 turn 生成长度 ≤ 槽区容量
    pub fn fits(&self, prompt_len: usize, max_new: usize, capacity: usize) -> bool {
        prompt_len.max(self.tokens.len()) + max_new <= capacity
    }

    /// 提交一个 turn 的历史增量:全量 prompt 长度 → 历史前缀之后的新增段。
    /// 返回 None = prompt 没有超出历史（纯生成续段）。
    pub fn delta_range(&self, prompt_len: usize) -> Option<(usize, usize)> {
        if prompt_len > self.tokens.len() {
            Some((self.tokens.len(), prompt_len))
        } else {
            None
        }
    }

    /// 回填账(turn 收口):KV 此刻已含「prompt 全量 + 生成段」,GDN
    /// 状态跨 turn 连续。账本 = prompt ++ generated；历史长度唯一真源
    /// 就是 `tokens.len()`。
    pub fn commit_turn(
        &mut self,
        prompt: &[u32],
        generated: &[u32],
        positions: &[[i32; 3]],
    ) -> Result<()> {
        // FIXME: 不推荐可变范式。要改东西，就创建一个新的。
        let total = prompt.len() + generated.len();
        if positions.len() != total {
            return Err(owl_iface::contract::ModelError::EngineInvariant {
                what: format!(
                    "turn 收口位置账 {} ≠ prompt+generated {}（会话 {}）",
                    positions.len(),
                    total,
                    self.id
                ),
            });
        }
        self.tokens.clear();
        self.tokens.extend_from_slice(prompt);
        self.tokens.extend_from_slice(generated);
        // 内容身份只要求覆盖**上一轮 prompt 段**（生成段由客户端按文本重渲染，
        // 尾部必然不同 —— 2026-11-16 真机定谳）
        self.prompt_boundary = prompt.len();
        self.positions.clear();
        self.positions.extend_from_slice(positions);
        self.max_pos = self
            .positions
            .iter()
            .flat_map(|p| p.iter())
            .copied()
            .max()
            .unwrap_or(0);
        Ok(())
    }
}

/// **L1 快照键沿用**(纯;2026-11-12 二刀)。
///
/// 按 token offset 对齐上一轮的 `msg_snaps` / `msg_snap_keys`:纯 append
/// (前缀守卫生效)时位置一一对应;新出现的边界 = `None`(等本 turn 拍)。
/// old 位置不在 new 里(截短/回撤)= 丢弃;old 键已被逐出 = `None`。
pub fn carry_msg_snap_keys(
    old_points: &[usize],
    old_keys: &[Option<crate::SnapKey>],
    new_points: &[usize],
) -> Vec<Option<crate::SnapKey>> {
    let mut out = vec![None; new_points.len()];
    for (i, at) in new_points.iter().enumerate() {
        if let Ok(j) = old_points.binary_search(at) {
            out[i] = old_keys.get(j).copied().flatten();
        }
    }
    out
}

/// **L1 会话资产只读投影**(2026-11-12 三层架构)。
///
/// 只回答"这个会话拥有哪些资产":历史长度、尾锚点快照、消息边界;
/// 不暴露物理槽号 / tier —— 那些属于 L2/L3。
#[derive(Clone, Debug)]
pub struct SessionAssets {
    /// 会话 id。
    pub id: u64,
    /// 会话历史长度(token；`tokens.len()` 的唯一投影)。
    pub history_len: usize,
    /// 本会话的尾锚点快照(L1 资产;`None` = 无/已逐出)。
    pub tail_snap: Option<crate::SnapKey>,
    /// 尾锚点 offset（页对齐 token）。
    pub tail_at: Option<usize>,
    /// 持久快照布局点（Tail + History；页对齐 token offset；升序）。
    pub msg_snaps: Vec<usize>,
    /// 与 `msg_snaps` 同序的角色。
    pub msg_snap_roles: Vec<crate::SnapRole>,
    /// 与 `msg_snaps` 同序的快照键（`None` = 未拍/已逐出）
    pub msg_snap_keys: Vec<Option<crate::SnapKey>>,
    /// 是否为 leave-gap 后的本地链（仅 turn 内）
    pub local_chain: bool,
    /// LRU 触达时钟(越小越老；仅淘汰顺序用)。
    pub last_use: u64,
}

/// **内容身份匹配的尾部容差**（token）。
///
/// 客户端把新消息接在会话账后面时，BPE 会在**接缝**处重新合并（会话账末尾
/// 1~3 个 token 的编号可能变）⇒ "短方被完整覆盖"允许末尾这么多对不上。
///
/// **不能大**：容差就是"认错对话"的窗口 —— 容差 ≥ chat 模板前缀长度时，
/// 无关对话只靠模板前缀就能匹配上（2026-11-16 真机故障形态：每个请求都
/// 命中 MRU 会话，`cached_tokens` 报一个假 32）。
pub const MATCH_TAIL_TOLERANCE: usize = 4;

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
    /// **会话账查询**(显式 `Result`;2026-10-17 裁决:账缺失**不 panic**)。
    pub fn session(&self, id: u64) -> Result<&AgentSession> {
        self.sessions
            .get(&id)
            .ok_or_else(|| owl_iface::contract::ModelError::EngineInvariant {
                what: format!("会话账缺失(会话 {id} 不在账)"),
            })
    }

    /// 同上(可变)。
    pub fn session_mut(&mut self, id: u64) -> Result<&mut AgentSession> {
        self.sessions
            .get_mut(&id)
            .ok_or_else(|| owl_iface::contract::ModelError::EngineInvariant {
                what: format!("会话账缺失(会话 {id} 不在账)"),
            })
    }

    /// 把会话从账中取出（所有权转移；用于"销毁旧账 → 新建替换"路径）。
    /// 取出后调用方必须在所有成功/失败路径把新账 `put` 回来。
    pub fn take(&mut self, id: u64) -> Result<AgentSession> {
        self.sessions
            .remove(&id)
            .ok_or_else(|| owl_iface::contract::ModelError::EngineInvariant {
                what: format!("会话账缺失(会话 {id} 不在账)"),
            })
    }

    /// 把新账放回表中（`take` 的配对口；重复 id = 结构错，响亮拒绝）。
    pub fn put(&mut self, session: AgentSession) -> Result<()> {
        let id = session.id;
        if self.sessions.insert(id, session).is_some() {
            return Err(owl_iface::contract::ModelError::EngineInvariant {
                what: format!("会话账重复插入(会话 {id} 已在账)"),
            });
        }
        self.next_id = self.next_id.max(id.saturating_add(1));
        Ok(())
    }

    /// 新表(空账；首次 `submit` 隐式建会话)。
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
        let s = self
            .sessions
            .entry(id)
            .or_insert_with(|| AgentSession::new(id, 0));
        s.last_use = tick;
        self.next_id = next.max(id + 1);
        Ok((id, s))
    }

    /// **会话身份 = 内容身份**:在**空闲**会话里找与 `prompt` 最长公共前缀的
    /// 一条(原始算法 §32;2026-11-16)。
    ///
    /// - **完整覆盖**：公共前缀必须盖住 `min(会话可见长, prompt 长)` ——
    ///   同一条对话的续写（会话账 ⊑ prompt）或客户端截短重发（prompt ⊑ 会话账）
    ///   都满足；只在**模板前缀**上撞车的无关对话盖不住 ⇒ 不算。
    ///   ⚠️ 这条 2026-11-16 真机定谳：只按"≥ 一页"判会让所有请求都匹配到
    ///   MRU 会话（chat 模板的 `<|im_start|>system…` 本身就超过一页），
    ///   于是每轮都把别人的账 rebase 掉（`cached_tokens` 假命中 32）。
    /// - `min_tokens` = 命中下限(实参 = 一页):更短的对话没有区分度;
    /// - 平手取 `last_use` 最大(MRU):同前缀的两条会话里,刚用过的那条才是
    ///   客户端正在续写的那条;
    /// - `busy` = 活跃/排队会话:它们的账是"未来态"(尾部正在写),不参与匹配
    ///   —— 匹配到它们会让 P4 的同会话串行把两个不相干的请求串成一串。
    ///
    /// 返回 `(会话 id, 命中可见 token 数)`;`None` = 没有会话够格(调用方新建)。
    /// 匹配是**只读**的(不 touch `last_use`):"命中"只有引擎确认要续写时才成立。
    pub fn match_prefix(
        &self,
        prompt: &[u32],
        min_tokens: usize,
        busy: &std::collections::HashSet<u64>,
    ) -> Option<(u64, usize)> {
        if prompt.is_empty() {
            return None;
        }
        // (id, 命中长度, last_use)
        let mut best: Option<(u64, usize, u64)> = None;
        for s in self.sessions.values() {
            if busy.contains(&s.id) {
                continue;
            }
            // 上界剪枝:(tokens 的可见长 ≤ tokens.len(),故它只是**上界**)
            // 连上界都赢不过现最佳就不必扫。取 `<` 而非 `<=`:平手仍可能
            // 靠 MRU 胜出。
            if let Some((_, bn, _)) = best {
                if s.tokens.len().min(prompt.len()) < bn {
                    continue;
                }
            }
            // 参考长 = **上一轮 prompt 段**（不含生成段）。账不可信（未收口/
            // pad 坏）⇒ 该会话退出身份匹配，不猜一个数字。
            let Some(ref_len) = s.prompt_visible_len() else {
                continue;
            };
            let n = s.common_prefix_text(prompt);
            // 短的一方必须被完整覆盖（接缝处允许 `MATCH_TAIL_TOLERANCE`）
            if n < min_tokens || n + MATCH_TAIL_TOLERANCE < ref_len.min(prompt.len()) {
                continue;
            }
            let better = match best {
                None => true,
                Some((_, bn, bl)) => n > bn || (n == bn && s.last_use > bl),
            };
            if better {
                best = Some((s.id, n, s.last_use));
            }
        }
        best.map(|(id, n, _)| (id, n))
    }

    /// E5:LRU 逐出候选(最老触达;调用方保证无活跃 turn ——
    /// actor 纪律:submit 与 pump 不相交)。None = 表空
    pub fn lru_victim(&self) -> Option<u64> {
        self.sessions
            .values()
            .min_by_key(|s| s.last_use)
            .map(|s| s.id)
    }

    /// 会话数（会话表上限的判据；F-22）。
    /// 会话数（会话表上限的判据；F-22）。
    pub fn len(&self) -> usize {
        self.sessions.len()
    }

    /// 会话表是否为空。
    pub fn is_empty(&self) -> bool {
        self.sessions.is_empty()
    }

    /// **LRU 淘汰候选（排除忙碌会话）**：`busy` 至少包含活跃/排队 turn 的
    /// 会话 id；空闲会话才可被淘汰（设备表在 turn 收口已归还，淘汰只丢
    /// host token 历史 ⇒ 下轮全量重算，正确性不变）。
    pub fn lru_victim_excluding(&self, busy: &std::collections::HashSet<u64>) -> Option<u64> {
        self.sessions
            .values()
            .filter(|s| !busy.contains(&s.id))
            .min_by_key(|s| s.last_use)
            .map(|s| s.id)
    }

    /// 按 id 查会话账(不存在 = `None`；带错误的入口是
    /// [`Self::session`]/[`Self::session_mut`])。
    pub fn get(&self, id: u64) -> Option<&AgentSession> {
        self.sessions.get(&id)
    }

    /// 按 id 查会话账(可变;不存在 = `None`)。
    pub fn get_mut(&mut self, id: u64) -> Option<&mut AgentSession> {
        self.sessions.get_mut(&id)
    }

    /// 全部会话(只读;会话资产投影 / 诊断)。
    pub fn iter(&self) -> impl Iterator<Item = &AgentSession> {
        self.sessions.values()
    }

    /// **L1 资产只读投影**(`/debug/vram` 的 `sessions` 面;按会话 id 升序)。
    pub fn assets(&self) -> Vec<SessionAssets> {
        let mut out: Vec<SessionAssets> = self
            .sessions
            .values()
            .map(|s| SessionAssets {
                id: s.id,
                history_len: s.tokens.len(),
                tail_snap: s.tail_snap,
                tail_at: s.tail_at,
                msg_snaps: s.msg_snaps.clone(),
                msg_snap_roles: s.msg_snap_roles.clone(),
                msg_snap_keys: s.msg_snap_keys.clone(),
                local_chain: s.local_chain,
                last_use: s.last_use,
            })
            .collect();
        out.sort_by_key(|s| s.id);
        out
    }

    /// 会话结束(客户端显式关闭;KV 区域归还 —— M0.5 单槽 = 账本清理)
    pub fn close(&mut self, id: u64) -> Result<()> {
        self.sessions
            .remove(&id)
            .map(|_| ())
            .ok_or_else(|| owl_iface::contract::ModelError::Msg(format!("session {id} 不存在")))?;
        Ok(())
    }
}

// ===========================================================================
// SessionBank(2026-10-14 拆分自 running.rs:RunningEngine 万能结构体
// 按状态域拆解 —— 本节收编**会话账 + 双块账房**及其复合操作)
// ===========================================================================

/// 块腿几何(boot 一次;S3-3 后 engine 只认 `BlockLeg`,`KvStore` 收成 store 内部件)。
pub struct BankPlan {
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
    /// MTP 草稿 KV 物理环(占位不向 BlockLeg 领页;表由 ring 映射)
    pub mtp_ring: bool,
    /// 前缀缓存开关(容量策略见 [`crate::CachePolicy`])
    pub prefix_cache: bool,
    /// target 腿宿主层预算(字节;0 = 关层,REQ C4 默认关)
    pub host_bytes: usize,
    /// **TP world**(2026-10-20):宿主腿每条链占 `world` 个平面
    /// (单卡 1;双 rank 2 ⇒ 单链宿主字节 = world × 单页,**预算语义不变**)。
    pub world: usize,
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
pub struct SessionBank {
    /// 会话账(GDN 格分配/LRU 逐出/前缀守卫)
    sessions: SessionTable,
    /// target 链门面(块腿 = `rel.leg`;预算/准入账同屋)
    rel: Rel,
    /// MTP 链块腿(独立页池,无前缀缓存;宿主面恒关 —— 裁决④推后)
    leg_mtp: BlockLeg,
    /// MTP 草稿 KV 物理环(T5;true = 不向 leg_mtp 领页/不归还)
    mtp_ring: bool,
}

/// `SessionBank` 的一次性 split-borrow 视图。
///
/// **为什么不是逐字段访问器**：`sessions` 与 `rel` 的复合操作需要同时可变
/// 借用两个不相交字段；`&mut self` 方法会整借 bank，调用点会退化成把
/// 一个字段临时 `take` 出来或再引入一套旁路。这里一次性拆开四个字段，
/// 借用关系在类型上显式可见，调用点不能再绕过 bank 的复合语义口。
///
/// 单字段只读路径用 [`SessionBank::parts`]；需要写任一分量时用
/// [`SessionBank::parts_mut`]，并把绑定限制在真正需要同时访问的作用域内。
pub struct SessionBankParts<'a> {
    /// 会话账。
    pub sessions: &'a mut SessionTable,
    /// target 链门面。
    pub rel: &'a mut Rel,
    /// MTP 链块腿。
    pub leg_mtp: &'a mut BlockLeg,
    /// MTP 环模式（标量快照，拆借后仍可读）。
    pub mtp_ring: bool,
}

/// `SessionBank` 的只读 split-borrow 视图（供诊断/查询路径使用）。
pub struct SessionBankRef<'a> {
    /// 会话账。
    pub sessions: &'a SessionTable,
    /// target 链门面。
    pub rel: &'a Rel,
    /// MTP 链块腿。
    pub leg_mtp: &'a BlockLeg,
    /// MTP 环模式。
    pub mtp_ring: bool,
}

impl SessionBank {
    /// 只读拆借（四个分量一次借出；不暴露字段本身）。
    pub fn parts(&self) -> SessionBankRef<'_> {
        SessionBankRef {
            sessions: &self.sessions,
            rel: &self.rel,
            leg_mtp: &self.leg_mtp,
            mtp_ring: self.mtp_ring,
        }
    }

    /// 可变拆借（sessions/rel/leg_mtp 三路不相交）。
    pub fn parts_mut(&mut self) -> SessionBankParts<'_> {
        SessionBankParts {
            sessions: &mut self.sessions,
            rel: &mut self.rel,
            leg_mtp: &mut self.leg_mtp,
            mtp_ring: self.mtp_ring,
        }
    }

    /// MTP 是否走物理环（不领 `leg_mtp` 页）。
    pub fn mtp_ring(&self) -> bool {
        self.mtp_ring
    }

    /// boot 构造(target 链门面 + MTP 链账房;几何随池)。前缀缓存
    /// 只挂 target 链(容量 = 池半);MTP 链无前缀缓存,非 mtp 模式
    /// 0 块 —— 误用 = 结构化报错(池耗尽语义)。
    ///
    /// `inputs`/`policy` = 准入预算与姿态(engine 在池/图落位后实测组装;
    /// 见 `Engine::run`)。装配失败 = **启动失败**(A5.1:没有数字不跑)。
    pub fn new(
        plan: BankPlan,
        inputs: crate::BudgetInputs,
        policy: crate::AdmissionPolicy,
        target_layout: PoolLayout,
        mtp_layout: Option<PoolLayout>,
        ledger: crate::SharedLedger,
    ) -> Result<Self> {
        // **缓存容量 = 整池**(2026-10-22 用户令:删 `prefix_cache_share`)。
        // 前缀缓存可以持有全部设备页 + 宿主槽;新请求进来时按
        // `KvStore::ensure_for_len` 逐出设备叶 → 宿主,宿主不够再走磁盘。
        let cache = if plan.paged && plan.prefix_cache {
            crate::CachePolicy::Share { permille: 1000 }
        } else {
            crate::CachePolicy::Off
        };
        let mut rel = Rel::open_world(
            inputs,
            policy,
            LegConfig {
                slots: plan.nb,
                block_tokens: plan.page,
                block_bytes: plan.block_bytes,
                cache,
                host_bytes: plan.host_bytes,
                class: BlockClass::KvPage,
                label: "target",
            },
            ledger,
            plan.world.max(1),
        )?;
        rel.attach_pool(target_layout)?;
        let mtp_ring = plan.mtp_ring;
        let mut leg_mtp = BlockLeg::open(LegConfig {
            slots: if plan.mtp && !mtp_ring { plan.nb } else { 0 },
            block_tokens: plan.page,
            block_bytes: if plan.mtp && !mtp_ring {
                plan.mtp_block_bytes
            } else {
                1
            },
            cache: crate::CachePolicy::Off,
            host_bytes: 0,
            class: BlockClass::KvPage,
            label: "mtp",
        })?;
        if !mtp_ring {
            if let Some(l) = mtp_layout {
                leg_mtp.attach_pool(l)?;
            }
        }
        Ok(Self {
            sessions: SessionTable::new(),
            rel,
            leg_mtp,
            mtp_ring,
        })
    }

    // （块腿的复合口做成**自由函数**:借用只落在 `&mut KvStore` 那一个字段上
    //   —— 调用点多处仍持 `&mut AgentSession`(会话表另行借出),方法口会整借
    //   bank 而冲突;自由函数与"字段不相交"的借用形状对齐。）

    /// 取或建(S2b 后**无逐出**):会话不再占 GDN 行 ⇒ 没有"格耗尽"这条路。
    /// 历史资源的回收由**缓存面 LRU + 宿主预算**负责(§2.3 目标形态);
    /// 会话表本身只存元数据(µs 级)。
    pub fn get_or_create(&mut self, id: Option<u64>) -> Result<u64> {
        self.sessions.get_or_create(id).map(|(id, _)| id)
    }

    /// 显式关会话(客户端声明不再续;账本清票,双链归还账房 ——
    /// 与 evict/burn 三路径同构;2026-10-15 结案拆分时挂的 FIXME:
    /// 原只还 target 链漏 MTP 页,mtp 模式下显式 close = 页池泄漏)。
    pub fn close(&mut self, id: u64) -> Result<()> {
        self.release_session_chains(id)?;
        self.sessions.close(id)?;
        // **快照归属随会话一起收口**：否则 LRU 淘汰/临时会话烧毁后，
        // 快照仍记在已不存在的 session 名下（id 复用时会串到新会话）。
        self.detach_session_snaps(id);
        Ok(())
    }

    /// 会话关闭后的快照归属释放（幂等；无快照 = 0）。
    fn detach_session_snaps(&mut self, id: u64) {
        let detached = self.rel.detach_session_snaps(id);
        if detached > 0 {
            owl_shared::metrics::with_metrics_store(|m| {
                m.counter_add("snap.session.detached", detached as u64, file!(), line!());
            });
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
        // **两条链都尝试**：target 失败不许截断 MTP 归还；失败时把未动的表
        // 放回会话，保证下次 close/burn 还能重试（release_table 已做全量校验，
        // 失败时表内容不变）。
        let target = self.rel.release_table(&mut table).map_err(ModelError::from);
        if target.is_err() {
            if let Some(s) = self.sessions.get_mut(id) {
                s.block_table = table;
            }
        }
        let mtp = self.release_mtp_table(&mut mtp_table);
        if mtp.is_err() {
            if let Some(s) = self.sessions.get_mut(id) {
                s.mtp_block_table = mtp_table;
            }
        }
        match (target, mtp) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(e), Ok(())) | (Ok(()), Err(e)) => Err(e),
            (Err(a), Err(b)) => Err(ModelError::EngineInvariant {
                what: format!("双链归还均失败: target={a}; mtp={b}"),
            }),
        }
    }

    /// **段边界池账同步**（S3b §23.2；2026-11-15 自 engine 下沉）。
    ///
    /// 真源 = 块腿报告的设备驻留字节 + 活跃 turn 的块表（按槽号去重）。
    /// 恒等式：`reserved(唯一活跃实占) + session_held + cache = 池实占`。
    /// 共享前缀出现在多个 turn 的块表里时，保留面只计一次。
    ///
    /// `active` = `(turn, session)` 列表；store 直接读自己的会话账，不再由
    /// 引擎在外部拼 slots（session 感知落在同屋）。
    pub fn sync_pool_account(&mut self, active: &[(u64, u64)]) -> Result<()> {
        let block_bytes = self.rel.slot_bytes();
        let occupied = self.rel.occupied_bytes();
        let movable = self.rel.movable_bytes();

        // **先全量预检，再写账**：缺会话/缺 turn/槽宽漂移必须在任何
        // `set_*` 之前拒绝，否则前面的 turn 已改、后面的没改，准入账半态。
        if block_bytes == 0 {
            return Err(ModelError::EngineInvariant {
                what: "sync_pool_account: block_bytes = 0".into(),
            });
        }
        let known_width = self.rel.admission().slot_bytes();
        if known_width != 0 && known_width != block_bytes {
            return Err(ModelError::EngineInvariant {
                what: format!("sync_pool_account: 槽宽漂移 {known_width} ≠ {block_bytes}"),
            });
        }
        let mut per_turn: Vec<(u64, Vec<u32>)> = Vec::with_capacity(active.len());
        for (turn, session) in active {
            let s = self
                .sessions
                .get(*session)
                .ok_or_else(|| ModelError::EngineInvariant {
                    what: format!("sync_pool_account: 活跃 turn {turn} 的会话 {session} 不在账"),
                })?;
            if !self.rel.admission().has_active(*turn) {
                return Err(ModelError::EngineInvariant {
                    what: format!("sync_pool_account: 活跃 turn {turn} 不在准入账"),
                });
            }
            let slots: Vec<u32> = s
                .block_table
                .iter()
                .copied()
                .filter(|b| !crate::is_host_placeholder(*b))
                .collect();
            per_turn.push((*turn, slots));
        }

        for (turn, slots) in &per_turn {
            self.rel
                .admission_mut()
                .set_active_slots(*turn, slots.iter().copied(), block_bytes)?;
        }
        self.rel.admission_mut().set_cache_bytes(movable);
        let held = self.rel.admission().reserved_bytes();
        let accounted = movable.saturating_add(held);
        if accounted > occupied {
            return Err(ModelError::EngineInvariant {
                what: format!(
                    "sync_pool_account: 账不闭合 occupied={occupied} < movable={movable} + held={held}"
                ),
            });
        }
        self.rel
            .admission_mut()
            .set_session_held(occupied - accounted);
        Ok(())
    }

    /// **turn 末释放设备表**（2026-11-15 自 engine 下沉）。
    ///
    /// 表出账归还（引用减一）；**历史 `tokens` 保留**，turn 执行态的
    /// 驻留前缀由 `ActiveTurn.cached_start` 持有。不在 turn 之间保留任何
    /// `block_table`；所有块回到 store 缓存面，下一轮按内容命中。
    pub fn release_turn_table(&mut self, session_id: u64) -> Result<()> {
        let mut table = Vec::new();
        if let Some(s) = self.sessions.get_mut(session_id) {
            table = std::mem::take(&mut s.block_table);
            s.block_hashes.clear();
            s.chain.reset();
            s.local_chain = false;
        }
        self.rel.release_table(&mut table)?;
        Ok(())
    }

    // **会话表软上限已退休**（2026-11-16 用户裁决：会话数不设上限 ——
    // 常驻成本 = 元数据 + 可驱逐资产，上限由池 + host/disk 预算 + 淘汰机制给；
    // 硬约束只有"同时推进" = `store.gdn.gdn_slots`）。原 `enforce_session_cap` 删除。

    /// MTP 链归还:线性档走块腿;ring 档只清表(页池固定,不归槽)。
    pub fn release_mtp_table(&mut self, table: &mut Vec<u32>) -> Result<()> {
        if self.mtp_ring {
            table.clear();
            Ok(())
        } else {
            self.leg_mtp.release_table(table)?;
            Ok(())
        }
    }

    /// **会话重建前置**（`plan_fresh` / 位置压实共用）：双链归还 + 账本
    /// rebase。复合操作必须与 bank 同屋 —— engine 不再自己拆 `sessions`
    /// 与 `leg_mtp` 后逐项归还（拆开就是单边泄漏的入口）。
    ///
    /// 归还失败时 `release_session_chains` 会把未归还的表放回会话，
    /// 本函数不 rebase、原样上抛；下一次 close/重建仍可重试。
    pub fn reset_session_for_rebuild(&mut self, id: u64) -> Result<()> {
        self.release_session_chains(id)?;
        let old = self.sessions.take(id)?;
        self.sessions.put(old.rebase())
    }

    /// **前缀命中后的链重开**：旧账清执行态（保留 L1），写入本次复用的
    /// 块表/哈希/链游标。`table.len() == hashes.len()` 由调用方保证；
    /// 这里再做一次显式校验，错长 = 结构错，不静默截断。
    pub fn reopen_session_chain(
        &mut self,
        id: u64,
        table: Vec<u32>,
        hashes: Vec<crate::ChainHash>,
        tail: crate::ChainHash,
    ) -> Result<()> {
        if table.len() != hashes.len() {
            return Err(ModelError::EngineInvariant {
                what: format!(
                    "重开链表长不一致：session={id} table={} hashes={}",
                    table.len(),
                    hashes.len()
                ),
            });
        }
        let old = self.sessions.take(id)?;
        let mut next = old.reopen_chain();
        next.chain.seed(hashes.len(), tail);
        next.block_table = table;
        next.block_hashes = hashes;
        self.sessions.put(next)
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
        assert_eq!(s.tokens.len(), 0);

        // turn1:全量 10 token → 零缓存守卫恒真,全量 prefill;收口账本
        // = prompt(10) + 生成(2)
        let p1: Vec<u32> = (0..10).collect();
        assert!(s.guard(&p1));
        assert_eq!(s.delta_range(10), Some((0, 10)));
        let pos1: Vec<[i32; 3]> = (0..12).map(|i| [i, i, i]).collect();
        s.commit_turn(&p1, &[90, 91], &pos1).expect("收口 turn1");
        assert_eq!(s.tokens.len(), 12);
        assert_eq!(s.tokens, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 90, 91]);

        // turn2:全量 14 token(历史 12 + 新消息 2)→ 前缀命中,只 prefill
        // [12,14);收口后再推进
        let mut p2 = p1.clone();
        p2.extend([90, 91, 7, 7]);
        assert!(s.guard(&p2), "历史逐 id 一致应命中");
        assert_eq!(s.delta_range(14), Some((12, 14)));
        let pos2: Vec<[i32; 3]> = (0..15).map(|i| [i, i, i]).collect();
        s.commit_turn(&p2, &[42], &pos2).expect("收口 turn2");
        assert_eq!(s.tokens.len(), 15);

        // turn3:纯生成续段(无新增 prompt)→ None
        assert_eq!(s.delta_range(15), None);
        assert!(s.fits(15, 8, 32));
        assert!(!s.fits(15, 24, 32));

        // 前缀失配(客户端改写历史/模板漂移)→ 守卫拒 → 回退全量
        let mut bad = p2.clone();
        bad[3] = 999;
        assert!(!s.guard(&bad));
        s.reset();
        assert!(s.tokens.is_empty());
        assert!(s.guard(&bad), "重置后零缓存守卫恒真(全量重算)");

        // 截短的历史(prompt 比账本还短)也拒
        let pos2: Vec<[i32; 3]> = (0..p2.len())
            .map(|i| [i as i32, i as i32, i as i32])
            .collect();
        s.commit_turn(&p2, &[], &pos2).expect("收口 turn3");
        assert!(!s.guard(&p2[..8]));
    }

    #[test]
    fn session_table_lifecycle() {
        let mut st = SessionTable::new();
        let (a, _sa) = st.get_or_create(Some(7)).expect("显式 id");
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
        assert_eq!(s.tokens.len(), 0);
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
            mtp_ring: false,
            prefix_cache,
            host_bytes: 0,
            world: 1, // 测试：单卡平面（TP 宿主腿的 world 语义由 store 门覆盖）
        }
    }

    /// 测试用装配:布局 = 单片段(只跑账,不跑搬运)
    fn bank_of(p: BankPlan) -> SessionBank {
        let target = PoolLayout {
            class: BlockClass::KvPage,
            epoch: 0,
            pieces: vec![crate::PieceSpec {
                buf: owl_iface::contract::Bytes::new(0, 1, owl_iface::contract::Dtype::F32, 1),
                stride_bytes: p.block_bytes,
            }],
        };
        let mtp = p.mtp.then(|| PoolLayout {
            class: BlockClass::KvPage,
            epoch: 0,
            pieces: vec![crate::PieceSpec {
                buf: owl_iface::contract::Bytes::new(0, 1, owl_iface::contract::Dtype::F32, 1),
                stride_bytes: p.mtp_block_bytes,
            }],
        });
        // 测试用预算:固定池档(池面 = 块字节 × 槽;准入按池墙判)
        let inputs = crate::BudgetInputs {
            v_card_bytes: 8 << 30,
            weights_bytes: 1 << 30,
            runtime_floor_bytes: 64 << 20,
            draft_pool_bytes: 0,
            gdn_row_bytes: 1 << 20,
            kv_bytes_per_token: (p.block_bytes / p.page.max(1)).max(1),
            page_tokens: p.page.max(1),
            host_bytes: p.host_bytes,
            disk_bytes: 0,
            kv_face_bytes: p.nb * p.block_bytes,
        };
        let policy = crate::AdmissionPolicy::new(4, crate::QueuePolicy::Reject);
        let cell = crate::shared_ledger(crate::TierBudget {
            host_bytes: inputs.host_bytes,
            disk_bytes: inputs.disk_bytes,
        });
        SessionBank::new(p, inputs, policy, target, mtp, cell).expect("装配块腿")
    }

    /// **release_turn_table 不得清历史**：turn 末释放设备表后，guard 仍能
    /// 用 tokens 判断客户端重发的历史是否连续（实际复用面另由内容树决定）。
    #[test]
    fn release_turn_table_keeps_history_guard() {
        let mut bank = bank_of(plan(64, 8, true, false));
        let (id, s) = bank.sessions.get_or_create(Some(7)).expect("建会话");
        let toks = [1u32, 2, 3, 4];
        let pos: Vec<[i32; 3]> = toks.iter().map(|t| [*t as i32; 3]).collect();
        s.commit_turn(&toks, &[], &pos).expect("收口历史");
        bank.release_turn_table(id).expect("释放设备表");
        let s = bank.sessions.session(id).expect("会话仍在");
        assert!(s.guard(&toks), "历史连续 ⇒ guard 真");
        assert!(!s.guard(&[1, 9, 3, 4]), "历史分叉 ⇒ guard 假");
        assert_eq!(s.tokens.len(), 4, "tokens 历史不得被 release 清掉");
    }

    /// **段边界同步缺会话必须响亮拒**：不能把空槽集当成“该 turn 没有块”。
    #[test]
    fn sync_pool_account_missing_session_is_loud() {
        let mut bank = bank_of(plan(64, 8, true, false));
        let err = bank
            .sync_pool_account(&[(1, 0xDEAD)])
            .expect_err("缺失会话必须拒");
        assert!(
            matches!(err, ModelError::EngineInvariant { .. }),
            "得 {err:?}"
        );
    }

    /// **close 必须释放快照归属**（会话 id 可复用，旧归属不能串台）。
    #[test]
    fn close_detaches_snapshots() {
        let mut bank = bank_of(plan(64, 8, true, false));
        let (id, _) = bank.sessions.get_or_create(Some(7)).expect("建会话");
        let key = crate::SnapKey::at_boundary(crate::ChainHash::new(0x51));
        bank.rel.register_snap_owned(
            key,
            4096,
            crate::Tier::Gpu,
            crate::SnapDurability::Durable,
            id,
            1,
        );
        assert_eq!(bank.rel.snap_session_info(key), Some((id, 1)));
        bank.close(id).expect("close");
        assert_eq!(
            bank.rel.snap_session_info(key),
            Some((crate::policy::NO_SESSION, 1)),
            "关闭后快照归属必须转无主"
        );
    }

    /// **重建复合口**：双链归还 + 账本换新；旧 tokens/L1 不得残留，id 保留。
    #[test]
    fn reset_session_for_rebuild_replaces_account() {
        let mut bank = bank_of(plan(64, 8, true, false));
        let mtp_before = bank.leg_mtp.report().slots_free;
        let tgt_before = bank.rel.slots_free();
        let (id, s) = bank.sessions.get_or_create(Some(7)).expect("建会话");
        let toks = [1u32, 2, 3, 4];
        let pos: Vec<[i32; 3]> = toks.iter().map(|t| [*t as i32; 3]).collect();
        s.commit_turn(&toks, &[], &pos).expect("收口历史");
        s.tail_snap = Some(crate::SnapKey::at_boundary(crate::ChainHash::new(0x77)));
        s.position_epoch = 3;
        bank.rel
            .ensure_for_len_unreserved(&mut s.block_table, 24)
            .expect("target 块");
        bank.leg_mtp
            .ensure_for_len(&mut s.mtp_block_table, 9)
            .expect("mtp 块");

        bank.reset_session_for_rebuild(id).expect("重建");

        let s = bank.sessions.session(id).expect("会话 id 保留");
        assert_eq!(s.id, id);
        assert!(s.tokens.is_empty(), "旧历史必须销毁");
        assert!(s.positions.is_empty(), "旧位置账必须销毁");
        assert!(s.tail_snap.is_none(), "旧 L1 资产必须销毁");
        assert!(s.block_table.is_empty(), "新账从空设备表起步");
        assert!(s.mtp_block_table.is_empty(), "MTP 表同清");
        assert_eq!(s.position_epoch, 4, "重建代数递增");
        assert_eq!(bank.rel.slots_free(), tgt_before, "target 链归还");
        assert_eq!(bank.leg_mtp.report().slots_free, mtp_before, "MTP 链归还");
    }

    #[test]
    fn close_releases_both_chains() {
        let mut bank = bank_of(plan(64, 8, true, false));
        let mtp_before = bank.leg_mtp.report().slots_free;
        let tgt_before = bank.rel.slots_free();

        let (id, s) = bank.sessions.get_or_create(Some(7)).expect("建会话");
        assert_eq!(id, 7);
        // target 链 2 块(24 tok / 块 8)+ MTP 链 1 块(9 tok)
        bank.rel
            .ensure_for_len_unreserved(&mut s.block_table, 24)
            .expect("target 块");
        bank.leg_mtp
            .ensure_for_len(&mut s.mtp_block_table, 9)
            .expect("mtp 块");
        assert!(bank.leg_mtp.report().slots_free < mtp_before, "mtp 已占用");

        bank.close(id).expect("close");
        assert_eq!(
            bank.leg_mtp.report().slots_free,
            mtp_before,
            "MTP 链必须归还"
        );
        assert_eq!(bank.rel.slots_free(), tgt_before, "target 链归还");
        assert!(bank.sessions.get(id).is_none(), "账本移除");
    }

    /// 无 mtp 模式回归:0 块页池,release 空表 = no-op,close 正常。
    #[test]
    fn close_without_mtp_is_noop_on_empty_chain() {
        let mut bank = bank_of(plan(32, 8, false, true));
        let (id, s) = bank.sessions.get_or_create(Some(1)).expect("建会话");
        bank.rel
            .ensure_for_len_unreserved(&mut s.block_table, 8)
            .expect("target 块");
        bank.close(id).expect("close");
        assert!(bank.sessions.get(id).is_none());
    }

    /// L1 键沿用:append 位置对齐;截短/回撤丢弃;缺键 = None。
    #[test]
    fn carry_msg_snap_keys_aligns_by_offset() {
        let a = crate::SnapKey::at_boundary(crate::ChainHash::new(1));
        let b = crate::SnapKey::at_boundary(crate::ChainHash::new(2));
        let old_points = vec![32usize, 64, 96];
        let old_keys = vec![Some(a), None, Some(b)];
        assert_eq!(
            carry_msg_snap_keys(&old_points, &old_keys, &[32, 64, 96, 128]),
            vec![Some(a), None, Some(b), None],
            "append:旧边界沿用,新边界等本 turn 拍"
        );
        assert_eq!(
            carry_msg_snap_keys(&old_points, &old_keys, &[32, 64]),
            vec![Some(a), None],
            "截短:只保留仍在 new 里的位置"
        );
        assert_eq!(
            carry_msg_snap_keys(&old_points, &old_keys, &[32, 80]),
            vec![Some(a), None],
            "回撤后位置变了:旧 96 不沿用"
        );
        assert_eq!(
            carry_msg_snap_keys(&old_points, &[Some(a)], &[32, 64]),
            vec![Some(a), None],
            "old_keys 短于 old_points 时按缺键处理(不 panic)"
        );
    }

    /// **内容身份三态**（`match_prefix`；2026-11-16）。
    ///
    /// 判据 = **短的一方被完整覆盖**：同一条对话的续写（会话账 ⊑ prompt）
    /// 命中；只在模板前缀上相同的无关对话不命中（真机定谳：只按"≥ 一页"判
    /// 会让所有请求串到 MRU 会话）。
    #[test]
    fn match_prefix_requires_full_coverage_of_the_shorter_side() {
        let mut st = SessionTable::new();
        let a_toks: Vec<u32> = (0..40).collect();
        let b_toks: Vec<u32> = (0..40).chain(100..140).collect();
        {
            let (_, a) = st.get_or_create(Some(1)).expect("A");
            a.tokens = a_toks.clone();
            a.prompt_boundary = a_toks.len();
            let (_, b) = st.get_or_create(Some(2)).expect("B");
            b.tokens = b_toks.clone();
            b.prompt_boundary = b_toks.len();
        }
        let busy = std::collections::HashSet::new();

        // A 的续写（prompt = A 账 + 新 12 token）：A 被完整覆盖 ⇒ 命中 A；
        // B 只覆盖到第 40 个（prompt 52 > 40 + 容差）⇒ 不构成完整覆盖 ⇒ 不参与。
        let mut p = a_toks.clone();
        p.extend(7..19);
        assert_eq!(st.match_prefix(&p, 32, &busy), Some((1, 40)));

        // **接缝容差**：BPE 会在"会话账末尾 + 新消息"的接缝处重合并（末尾
        // 几个 token 变号）⇒ 允许末尾 `MATCH_TAIL_TOLERANCE` 个对不上。
        let mut seam = a_toks.clone();
        let n = seam.len();
        seam[n - 2] = 999;
        seam[n - 1] = 998;
        let (seam_id, seam_n) = st
            .match_prefix(&seam, 32, &busy)
            .expect("接缝容差：2 个 token 变号仍应认到会话");
        assert_eq!(seam_n, 38, "命中长度 = 分歧点（38）");
        assert!(
            seam_id == 1 || seam_id == 2,
            "A/B 都被覆盖 ⇒ 按 MRU 取（实际 {seam_id}）"
        );

        // B 的续写：B（80）完整覆盖，A（40）也覆盖但短 ⇒ 取更长 / 更具体。
        let mut pb = b_toks.clone();
        pb.extend([9]);
        assert_eq!(st.match_prefix(&pb, 32, &busy), Some((2, 80)));

        // 全新对话（只有 4 token 公共前缀）⇒ 不完整覆盖 ⇒ None。
        assert_eq!(st.match_prefix(&[0, 1, 2, 3, 99], 32, &busy), None);
        // 空 prompt ⇒ None（没有身份证据）。
        assert_eq!(st.match_prefix(&[], 32, &busy), None);
        // **模板前缀撞车**：C 与 prompt 只共享 A 那 40 个 token（= 系统提示），
        // 之后各自展开 40 个不同 token ⇒ 不算同一条对话（C 出局），
        // 而 A 账（40）被 prompt 完整覆盖 ⇒ 命中 A（截短续写）。
        let mut p2: Vec<u32> = a_toks.clone();
        p2.extend(700..740);
        let (_, c) = st.get_or_create(Some(3)).expect("C");
        c.tokens = a_toks.iter().copied().chain(500..540).collect();
        c.prompt_boundary = c.tokens.len();
        assert_eq!(st.match_prefix(&p2, 32, &busy), Some((1, 40)));
    }

    /// 忙会话不参与匹配（尾账正在写，是未来态）。
    #[test]
    fn match_prefix_skips_busy_sessions() {
        let mut st = SessionTable::new();
        let toks: Vec<u32> = (0..64).collect();
        {
            let (_, s) = st.get_or_create(Some(5)).expect("建会话");
            s.tokens = toks.clone();
            s.prompt_boundary = toks.len();
        }
        let mut busy = std::collections::HashSet::new();
        busy.insert(5u64);
        assert_eq!(st.match_prefix(&toks, 32, &busy), None, "忙会话不得被匹配");
        busy.clear();
        assert_eq!(st.match_prefix(&toks, 32, &busy), Some((5, 64)));
    }

    /// 图像 pad 不算身份：会话账是 padded 序列，客户端 prompt 是原文。
    #[test]
    fn match_prefix_compares_visible_tokens_only() {
        let mut st = SessionTable::new();
        {
            let (_, s) = st.get_or_create(Some(1)).expect("建会话");
            // 可见 = [1,2,3,4,5]；[900,901] 是图像 pad 块。
            s.tokens = vec![1, 2, 3, 900, 901, 4, 5];
            s.pads = vec![(3, 5)];
            s.prompt_boundary = s.tokens.len();
        }
        let busy = std::collections::HashSet::new();
        // 可见账 = [1..5]（5 tok）被 prompt 完整覆盖 ⇒ 命中。
        assert_eq!(st.match_prefix(&[1, 2, 3, 4, 5, 6], 3, &busy), Some((1, 5)));
        // 坏 pad 账（越界）不得拼接出假命中：公共前缀止于 pad 之前（3 tok）
        // ⇒ 盖不住"短的一方"（5）⇒ None（宁可不认，也不认错）。
        st.session_mut(1).expect("会话").pads = vec![(3, 99)];
        assert_eq!(st.match_prefix(&[1, 2, 3, 4, 5, 6], 3, &busy), None);
    }

    /// **生成段不参与身份**（真机定谳 2026-11-16）：会话账 = 上一轮 prompt +
    /// 模型生成的尾部；客户端回填的是按文本重渲染的 prompt ⇒ 尾部对不上是
    /// **常态**。身份判据只要求覆盖到上一轮的 **prompt 边界**。
    #[test]
    fn match_prefix_ignores_generated_tail() {
        let mut st = SessionTable::new();
        let prompt: Vec<u32> = (0..64).collect();
        let generated: Vec<u32> = (900..924).collect(); // 模型产物（客户端回填后不同）
        {
            let (_, s) = st.get_or_create(Some(1)).expect("建会话");
            s.tokens = prompt.iter().copied().chain(generated).collect();
            s.prompt_boundary = prompt.len();
        }
        let busy = std::collections::HashSet::new();
        // 续写：上一轮 prompt + 新的用户消息（客户端重渲染 ⇒ 生成段不出现）
        let mut next = prompt.clone();
        next.extend([700, 701, 702]);
        assert_eq!(
            st.match_prefix(&next, 32, &busy),
            Some((1, 64)),
            "覆盖到 prompt 边界即算同一条会话"
        );
        // 未收口的会话（prompt_boundary = 0）不得参与匹配
        {
            let (_, s) = st.get_or_create(Some(2)).expect("建会话");
            s.tokens = (0..64).collect();
            s.prompt_boundary = 0;
        }
        assert_eq!(
            st.match_prefix(&next, 32, &busy),
            Some((1, 64)),
            "未收口会话出局，仍命中已收口的那条"
        );
    }

    /// **前缀命中路径的 reset 必须保留 L1 资产**；只有 `rebase`（真换链）
    /// 才清 tail/msg keys。否则旧 tail 永远等不到降级，Tail 角色会积累。
    #[test]
    fn reset_preserves_l1_assets_but_rebase_clears_them() {
        let mut st = SessionTable::new();
        let (_, s) = st.get_or_create(Some(7)).expect("建会话");
        let k = crate::SnapKey::at_boundary(crate::ChainHash::new(9));
        s.tail_snap = Some(k);
        s.tail_at = Some(64);
        s.msg_snaps = vec![32, 64];
        s.msg_snap_roles = vec![crate::SnapRole::History, crate::SnapRole::Tail];
        s.msg_snap_keys = vec![Some(k), None];
        s.local_chain = true;

        s.reset();
        assert_eq!(s.tail_snap, Some(k), "命中续写不得丢 tail 归属");
        assert_eq!(s.tail_at, Some(64), "命中续写不得丢 tail offset");
        assert_eq!(s.msg_snaps, vec![32, 64], "命中续写不得丢布局点");
        assert_eq!(s.msg_snap_keys, vec![Some(k), None], "键沿用不得断");
        assert!(s.local_chain, "命中续写仍持 gapped 链");

        let s = s.clone().rebase();
        assert!(!s.local_chain, "真换链必须解除 gapped 表保留");
        assert!(s.tail_snap.is_none(), "真换链清 tail");
        assert!(s.tail_at.is_none(), "真换链清 tail offset");
        assert!(s.msg_snaps.is_empty(), "真换链清布局点");
        assert!(s.msg_snap_roles.is_empty(), "真换链清角色");
        assert!(s.msg_snap_keys.is_empty(), "真换链清键");
    }
}

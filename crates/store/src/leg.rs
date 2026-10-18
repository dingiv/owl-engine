//! 块腿(`BlockLeg`)—— **一条链**的"账 + 宿主物理 + 搬运口"(设计卷 §10)。
//!
//! # 分工(裁决⑪)
//!
//! ```text
//! engine  决定"什么时候换、换哪条链、谁不能动"     ← 调用方
//! BlockLeg 实现"换什么、搬到哪层、账怎么走"        ← 本文件
//! server  执行"地址计算 / DMA / 一次 ack"          ← 只认 Bytes 与字节长度
//! ```
//!
//! # 为什么是"腿"而不是裸 `KvStore`
//!
//! engine 侧只该认一个类型(接口唯一);`KvStore` 收成 store 的内部件 ⇒ 本类型对外
//! 提供:①控制面(命中/增长/登记/归还;同步、零 I/O)②**布局注册**(id 级,`epoch`)③
//! 数据面(换出/换回,经 [`crate::moves`] 的自由函数,显式收 `&mut D`)。
//!
//! # 注册制(设计卷 §10.3 推论 A)
//!
//! 换出要把"逻辑块"落到**每层缓冲的一个页**上,所以腿必须知道"块 → 片段"的映射。
//! 该映射**由 engine 在 boot 注册一次**(id 级:`Bytes` 句柄 + 每页字节),此后所有交换
//! 只谈块身份;`epoch` 挡陈旧布局(池重建/几何变化 ⇒ 重注册)。
//!
//! # 现状(2026-10-16)
//!
//! - 控制面 / 布局 / 换出换回**已接线**;`host_bytes = 0`(默认)= **关宿主层**,
//!   此时换出会**响亮拒**(不静默降级)—— 行为与接线前完全一致。
//! - 片段读走 [`moves::download_span`](=`dtoh_at`;CUDA 面待补,见设计卷 §10.5)。
//!   批面(`DtohBatch`/`HtodBatch`)是下一步;它**不改本文件的 API**(实现换成批命令即可)。

use owl_iface::contract::{Bytes, ConnSlot, Connector, SwapPiece};

use crate::error::{StoreError, StoreResult};
use crate::host::{HostBlock, HostTier, SlotTable};
use crate::id::ChainHash;
use crate::kv::{CacheUpdate, KvStore, PrefixMatch};
#[cfg(test)]
use crate::moves::{download_span, upload_span}; // 测试的种图样/读回路径
use crate::policy::EvictCandidate;

/// **块类**(策略维度;设计卷 §10.6)。
///
/// 每类将来各挂一套策略对象(KV = LRU;专家 = 热度;稠密核 = 永不驱逐)。
/// 现在只有 `KvPage` 在位,余下三类先把轴立起来(成本 = 一个枚举)。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockClass {
    /// attention KV 页(= 本引擎的逻辑块,32 token)。
    KvPage,
    /// GDN/线性注意力行(状态腿,S2)。
    GdnRow,
    /// MoE 路由专家(可分层;热度驱动)。
    Expert,
    /// 稠密核权重(常驻,永不驱逐)。
    DenseCore,
}

/// **载体** —— store 里"东西在哪"的唯一表达(设计卷 §10.1)。
///
/// 身份(内容哈希)与载体**正交**:换层 = 换载体 + 台账挪一行,身份不变。
/// 裸指针只有 server 有,这里只有 id。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Carrier {
    /// 显存:槽号(server 持"槽 → 每层片段地址")。
    Device(u32),
    /// 宿主:宿主 slab(内容身份即键,故只记存在性)。
    Host,
    /// 磁盘(S6 落地;位先留 —— 出现即"未接线"响亮拒)。
    Disk,
}

/// 一块在设备上的**片段**(注册制;id 级 + 每页字节)。
#[derive(Clone, Debug)]
pub struct PieceSpec {
    /// 设备块句柄(**元数据,不是所有权** —— 生命周期归 engine 的池)。
    pub buf: Bytes,
    /// 每页字节(= 逻辑块在该缓冲内的步长)。
    pub stride_bytes: usize,
}

/// **池布局**(engine 在 boot 注册一次)。
#[derive(Clone, Debug)]
pub struct PoolLayout {
    /// 块类(策略分发键)。
    pub class: BlockClass,
    /// 布局代数(池重建/几何变化 ⇒ +1;旧代号的交换调用**响亮拒**)。
    pub epoch: u64,
    /// 每层 k/v 各一项(顺序无关,只要步长与句柄正确)。
    pub pieces: Vec<PieceSpec>,
}

impl PoolLayout {
    /// 一个逻辑块的总字节(Σ 片段步长)= 宿主缓冲的单块尺寸。
    pub fn slot_bytes(&self) -> usize {
        self.pieces.iter().map(|p| p.stride_bytes).sum()
    }
}

/// **链级卸载候选**(§14.6 链级卸载律):一整条可从链尾向链头搬完的链。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChainCand {
    /// 链尾身份(叶;沉这一条链的句柄)。
    pub tail: ChainHash,
    /// 链头身份(= 前缀起点;诊断/回灌面用)。
    pub head: ChainHash,
    /// 链上**设备驻留**的块数(本链可腾出的槽数)。
    pub dev_blocks: usize,
    /// 链总块数(含已在宿主的块)。
    pub blocks: usize,
    /// 链尾新鲜度(排序键;§14.6 律 4:链级新鲜度按**链尾**取)。
    pub last_use: u64,
}

/// **前缀缓存容量策略**(store 自决 —— 它是显存管理者;engine 只给事实与份额)。
///
/// 为什么放在 store:容量 = "多少块可以留在显存里等复用",这是**驻留决策**,
/// 与 `plan(need, protect) -> Verdict` 同一层。engine 只该给两件事实
/// (设备池页数 / 宿主预算)和一条**份额**政策,不该自己算 `nb/2`。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CachePolicy {
    /// 关缓存:树不匹配、不登记。
    Off,
    /// **显式条数上限**(测试/特殊档:绕开份额算式,语义精确)。
    Cap(usize),
    /// **按份额**:设备池的 `permille‰` + **宿主槽全额**。
    ///
    /// 为什么加宿主槽:前缀索引**跨三层** —— 卸载到宿主的块仍占一个条目名额;
    /// 容量只按设备池算(旧 `nb/2`)会让"卸载后的条目立刻被容量逐出",
    /// 宿主层攒不住东西(实测:换出 2 块后条目即被逐 ⇒ 回灌永不触发)。
    Share {
        /// 设备池占比(千分比;1000 = 允许缓存与设备池等量)。
        permille: u16,
    },
}

/// 开腿配置(boot 一次)。
#[derive(Clone, Copy, Debug)]
pub struct LegConfig {
    /// 池块容量(管理面页数)。
    pub slots: usize,
    /// 单块 token 数(本引擎 = 32;与 XQA `tokensPerPage` 对齐)。
    pub block_tokens: usize,
    /// 单块设备字节(**真源**,坑册 §42)。
    pub block_bytes: usize,
    /// 前缀缓存容量策略(见 [`CachePolicy`])。
    pub cache: CachePolicy,
    /// 宿主层预算(字节;**0 = 关层**,REQ C4 默认关)。
    pub host_bytes: usize,
    /// 块类。
    pub class: BlockClass,
    /// 链名(诊断:容量不足的文案里带上,池耗尽排障第一现场)。
    pub label: &'static str,
}

/// 腿的运行计数(报告面)。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LegOps {
    /// 换出次数。
    pub offloads: u64,
    /// 换回次数。
    pub onloads: u64,
    /// 换出字节。
    pub offload_bytes: u64,
    /// 换回字节。
    pub onload_bytes: u64,
    /// **`ensure_for_len` 兜底驱逐的块数**(需求驱动保洁不足的指纹,§12.4):
    /// 需求驱动做实后,该值只在真容量不足(或保洁误算)时非零。
    pub ensure_evicted: u64,
    /// 换出**命令**次数(块数见 `offloads`;§15 批面:一条 swap 搬 N 块)。
    pub offload_batches: u64,
    /// 换回**命令**次数(块数见 `onloads`)。
    pub onload_batches: u64,
}

/// 腿报告(设计卷 §10 报告口径)。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LegStats {
    /// 块类。
    pub class: BlockClass,
    /// 布局代数(`None` = 未注册)。
    pub epoch: Option<u64>,
    /// 单块设备字节(对账口:字节 = 块数 × 本值)。
    pub block_bytes: usize,
    /// 单块 token 数(页)。
    pub block_tokens: usize,
    /// 槽总容量。
    pub slots_total: usize,
    /// 自由槽。
    pub slots_free: usize,
    /// 缓存条目数。
    pub cached_blocks: usize,
    /// host 承载条目数。
    pub host_blocks: usize,
    /// 设备侧缓存字节(树绑定条目面)。
    pub gpu_bytes: usize,
    /// **池实占字节**(槽账真源:已用槽 × 块字节)。
    pub occupied_bytes: usize,
    /// **可直接搬动的设备面字节**(引用 = 1;§23.2 准入可动面真源)。
    pub movable_bytes: usize,
    /// 宿主侧驻留字节。
    pub host_bytes: usize,
    /// 宿主层预算(0 = 关层)。
    pub host_budget_bytes: usize,
    /// 运行计数。
    pub ops: LegOps,
}

/// **块腿**:一条链的账 + 宿主物理 + 搬运口。
pub struct BlockLeg {
    kv: KvStore,
    host: HostTier,
    layout: Option<PoolLayout>,
    ops: LegOps,
    class: BlockClass,
    label: &'static str,
}

impl BlockLeg {
    /// 开腿(唯一入口;宿主层 `host_bytes = 0` ⇒ 关层)。
    ///
    /// `slots = 0` = **该腿未启用**(例:非投机档的 MTP 链)—— 合法;
    /// 误用(去要槽)= `Exhausted` 响亮拒,与接线前的 0 块页池同语义。
    pub fn open(cfg: LegConfig) -> StoreResult<Self> {
        if cfg.block_tokens == 0 || cfg.block_bytes == 0 {
            return Err(StoreError::Invariant(format!(
                "块腿几何非法:slots={} block_tokens={} block_bytes={}",
                cfg.slots, cfg.block_tokens, cfg.block_bytes
            )));
        }
        Ok(Self {
            // **缓存容量 = store 自决**(单源;见 `CachePolicy`):份额 × 设备池
            // + 宿主槽全额(跨层索引:沉到宿主的条目仍占名额)。
            kv: KvStore::new(
                cfg.slots,
                cfg.block_tokens,
                cfg.block_bytes,
                match cfg.cache {
                    CachePolicy::Off => 0,
                    CachePolicy::Cap(n) => n,
                    CachePolicy::Share { permille } => {
                        if permille == 0 {
                            0
                        } else {
                            let dev = (cfg.slots as u128 * permille as u128 / 1000) as usize;
                            let host_slots = if cfg.host_bytes > 0 && cfg.block_bytes > 0 {
                                cfg.host_bytes / cfg.block_bytes
                            } else {
                                0
                            };
                            dev.max(1) + host_slots
                        }
                    }
                },
            ),
            host: HostTier::new(cfg.block_bytes, cfg.host_bytes)?,
            layout: None,
            ops: LegOps::default(),
            class: cfg.class,
            label: cfg.label,
        })
    }

    /// **注册池布局**(boot 一次;id 级)。`epoch` 单调增即可(用调用方给的值)。
    ///
    /// 校验:布局的单块字节必须等于开腿时的 `block_bytes`(否则宿主缓冲与设备页
    /// 不同尺寸 ⇒ 换出必错;此处**响亮拒**,不留隐患)。
    pub fn attach_pool(&mut self, layout: PoolLayout) -> StoreResult<u64> {
        let want = self.kv.block_bytes();
        let got = layout.slot_bytes();
        if got != want {
            return Err(StoreError::Invariant(format!(
                "池布局单块字节不符:布局 {got} ≠ 腿 {want}(检查每页字节/层数/量化档)"
            )));
        }
        let epoch = layout.epoch;
        self.layout = Some(layout);
        Ok(epoch)
    }

    /// 已注册布局的代数(`None` = 未注册)。
    pub fn layout_epoch(&self) -> Option<u64> {
        self.layout.as_ref().map(|l| l.epoch)
    }

    /// 块类。
    pub fn class(&self) -> BlockClass {
        self.class
    }

    /// 单块设备字节。
    pub fn slot_bytes(&self) -> usize {
        self.kv.block_bytes()
    }

    /// 单块 token 数。
    pub fn block_tokens(&self) -> usize {
        self.kv.block_tokens()
    }

    // ───────────────────────── 控制面(同步、零设备 I/O) ─────────────────────────

    /// 前缀命中(**engine 侧的三元组**:块数 / 逐块设备槽 / 内容身份链)。
    /// `slot = None` = 内容在宿主层,由 engine 在段边界回灌(S3-3b)。
    ///
    /// 命中面含宿主承载格(`slot = None`)时**响亮拒** —— 回灌未接线(S3-3b)之前,
    /// 宁可整条不复用,也不静默少用几块(设计卷 §10.4 律⑤)。
    pub fn match_prefix(&mut self, tokens: &[u32]) -> StoreResult<ChainHit> {
        let pm: PrefixMatch = self.kv.match_prefix(tokens);
        if pm.matched_blocks == 0 {
            return Ok(ChainHit::default());
        }
        let slots: Vec<Option<u32>> = pm.blocks.iter().map(|b| b.slot).collect();
        let hashes = pm.blocks.iter().map(|b| b.chain).collect();
        Ok(ChainHit { matched_blocks: pm.matched_blocks, slots, hashes })
    }

    /// 块表增长(压力驱逐在腿内;逐出的宿主条目 = 未接线,响亮拒)。
    ///
    /// §12.4:需求驱动保洁到位时,这条路上的同步驱逐应**恒为空**
    /// (腾位发生在段边界);非空 = 保洁不足(计入 `ops.ensure_evicted`,
    /// 报告面可查 —— 不许静默)。
    pub fn ensure_for_len(&mut self, table: &mut Vec<u32>, target_len: usize) -> StoreResult<()> {
        let evicted = self.kv.ensure_for_len(table, target_len).map_err(|e| match e {
            StoreError::Exhausted(m) => StoreError::Exhausted(format!("{} 链:{m}", self.label)),
            other => other,
        })?;
        if !evicted.is_empty() {
            self.ops.ensure_evicted += evicted.len() as u64;
        }
        self.drop_host_evicted(&evicted)
    }

    /// 压力驱逐产出宿主条目 = 内容整体淘汰:把宿主副本**真的丢掉**(S3-3b 宿主面在位)。
    fn drop_host_evicted(&mut self, evicted: &[(ChainHash, Option<u32>)]) -> StoreResult<()> {
        for (chain, slot) in evicted {
            if slot.is_none() {
                let _ = self.host.evict(*chain)?;
            }
        }
        Ok(())
    }

    /// 宿主层是否开启(0 = 关层)。
    pub fn host_enabled(&self) -> bool {
        !self.host.is_disabled()
    }

    /// 宿主层驻留字节。
    pub fn host_used_bytes(&self) -> usize {
        self.host.used_bytes()
    }

    /// 宿主层**还能装几个块**(预算 − 驻留 − 空转缓冲;关层 = 0)。
    pub fn host_free_slots(&self) -> usize {
        let used = self.host.used_bytes() + self.host.free_bytes();
        self.host
            .budget_bytes()
            .saturating_sub(used)
            / self.kv.block_bytes().max(1)
    }

    /// 宿主层预算(字节;0 = 关层)。
    pub fn host_budget_bytes(&self) -> usize {
        self.host.budget_bytes()
    }

    /// 宿主层**空转缓冲**字节(真实 RAM;报告必须算上 —— 已 acquire 未 put 的池租约)。
    pub fn host_free_bytes(&self) -> usize {
        self.host.free_bytes()
    }

    /// 前缀缓存探针计数(命中/未命中入账;命中面查询后调用)。
    pub fn note_probe(&mut self, hit: bool) {
        self.kv.note_probe(hit);
    }

    /// 一条链的块数(诊断/报告;不在账 = 0)。
    pub fn chain_len(&self, chain: ChainHash) -> usize {
        self.kv.chain_len(chain)
    }

    /// 整块前缀的内容身份(身份原语的腿面出口;engine 播种链游标 / 测试)。
    pub fn tail_hash(&self, tokens: &[u32], full_blocks: usize) -> Option<ChainHash> {
        self.kv.tail_hash(tokens, full_blocks)
    }

    /// 块账统计(内容身份树 + 槽池面;门面报告用 —— 单一真源)。
    pub fn kv_stats(&self) -> crate::kv::KvStats {
        self.kv.stats()
    }

    /// **可直接搬动的设备面字节**(S3b §23.2):设备绑定且**引用数 = 1** 的块
    /// (纯缓存面;被任何会话链/活跃 turn 持着的块不算)。
    ///
    /// 为什么是真源而不是"池实占 − 活跃实占":
    /// - 活跃面与 idle 会话**都拿会话引用** ⇒ 字节相减会把 idle 会话持着的
    ///   历史算成可动,而 `chain_down_span`/`ensure_offloadable` 会 Busy
    ///   ⇒ 准入就会先把话说满(`EvictThen`)再搬不动(退回 `Exhausted`)。
    /// - 这里把"能不能搬"的真判据(引用数)直接当账——两个数不再两套口径。
    ///
    /// 注:idle 会话持着的历史在本实现里也算"不可动"(目标模型 §2.3 里它
    /// 应该可动 —— 那条律等"会话引用 ≠ 活跃引用"拆分刀)。
    pub fn movable_bytes(&self) -> usize {
        self.kv.movable_bytes()
    }

    /// **池实占字节**(槽账真源:已用槽 × 块字节;S3b §23.2)。
    ///
    /// 为什么不看 [`Self::kv_stats`] 的 `gpu_bytes`:那是**树里绑定条目**的面 ——
    /// 尾部不满一块的前缀(`tokens.len()/page` 截断)不进树,但槽已经占了;
    /// 准入按"绑定面"算就会**少算实占**(实测:8 会话占满 8 槽,绑定面只 6 块)。
    pub fn occupied_bytes(&self) -> usize {
        let used = self.kv.slots_total().saturating_sub(self.kv.slots_free());
        used * self.kv.block_bytes()
    }

    /// 链名(诊断/拒发文案)。
    pub fn label(&self) -> &'static str {
        self.label
    }

    /// **驱逐一块**(内容真丢;引用必须为 1 ⇒ 活跃面 = `Busy` 响亮拒)。
    /// 设备条目:移条目 + 还槽;宿主条目:丢副本 + 移条目。
    pub fn drop_cached(&mut self, chain: ChainHash) -> StoreResult<()> {
        match self.kv.locate(chain) {
            Some(crate::tier::Tier::Gpu) => {
                let slot = self.kv.bound_slot(chain)?;
                let refs = self.kv.slot_refs(slot);
                if refs != 1 {
                    return Err(StoreError::Busy(format!(
                        "块 {chain:?}(槽 {slot})引用 {refs} > 1,不可驱逐(活跃面)"
                    )));
                }
                let _ = self.kv.drop_entry(chain)?; // 内部 decref ⇒ 槽回池
            }
            Some(crate::tier::Tier::Host) => {
                let _ = self.host.evict(chain)?; // 宿主副本丢
                let _ = self.kv.drop_entry(chain)?;
            }
            Some(crate::tier::Tier::Disk) => {
                let _ = self.kv.drop_entry(chain)?;
            }
            None => return Err(StoreError::Unknown(format!("块 {chain:?} 不在账里"))),
        }
        Ok(())
    }

    /// 自由槽数(engine 保洁水位用)。
    pub fn slots_free(&self) -> usize {
        self.kv.slots_free()
    }

    /// 槽总容量。
    pub fn slots_total(&self) -> usize {
        self.kv.slots_total()
    }

    /// **按策略序取可下沉候选**(叶 + 未保护;S3-3b 保洁用)。
    ///
    /// ⚠️ 这是**叶级**候选 —— 用于退化路径(没有可整条下沉的链时)与
    /// **驱逐**(`Sink::Drop`,丢内容仍可按叶)。要「对复用有贡献的下沉」
    /// 请用 [`Self::chain_offload_candidates`](§14.6 链级卸载律)。
    pub fn offload_candidates(&self, protect: &[ChainHash], limit: usize) -> Vec<ChainHash> {
        let mut c = self.kv.evict_candidates(protect);
        // **只在设备上的**才是下沉候选(已在 host/盘的条目没有绑定可摘 ——
        // 送进去会撞 `bound_slot` 的不变量)
        c.retain(|x| x.tier == crate::tier::Tier::Gpu);
        c.sort_by_key(|x| x.last_use);
        c.into_iter().take(limit).filter_map(|x| x.chain).collect()
    }

    /// **链级卸载候选**(§14.6 链级卸载律)—— 按**链尾新鲜度**出**整条链**。
    ///
    /// 为什么不是叶:前缀匹配**从链头开始走**;只沉链尾 ⇒ 下次命中根本走不到
    /// 那块(它不在命中面里)⇒ 对复用零贡献、回灌永不触发(实测现场:第 4 轮
    /// `pending=0` —— 被沉的那块是**生成段碎块**,不在 89-token 前缀里)。
    ///
    /// 出票条件(整条一起,半截不出票 —— 见 `KvStore::chain_down_span`):
    /// 1. 叶本身在设备上;
    /// 2. 链上每块都可达身份证(`chain_down_span = Some`);
    /// 3. 未被 `protect`。
    ///
    /// `budget_slots` = 期望腾出的**设备槽数**(按链累计;一条链可能超出预算,
    /// 因为粒度就是链)。返回按链尾 LRU 升序(最冷在前)。
    pub fn chain_offload_candidates(
        &self,
        protect: &[ChainHash],
        budget_slots: usize,
    ) -> Vec<ChainCand> {
        let mut leaves = self.kv.evict_candidates(protect);
        leaves.retain(|c| c.tier == crate::tier::Tier::Gpu);
        leaves.sort_by_key(|x| x.last_use);
        let mut out = Vec::new();
        let mut got = 0usize;
        for c in leaves {
            let Some(tail) = c.chain else { continue };
            let Some(dev_blocks) = self.kv.chain_down_span(tail) else { continue };
            let chain = self.kv.chain_of(tail);
            let head = chain.first().map(|b| b.chain).unwrap_or(tail);
            out.push(ChainCand {
                tail,
                head,
                dev_blocks,
                blocks: chain.len(),
                last_use: c.last_use,
            });
            got += dev_blocks;
            if got >= budget_slots {
                break;
            }
        }
        out
    }

    /// 登记一条链(turn 收口;缓存取一份绑定引用)。
    ///
    /// ⚠️ 登记**内部会因容量驱逐**(`cache_cap`)—— 被逐的宿主条目必须把宿主副本
    /// 一起丢掉(`ensure_for_len` 那条路有,这里原先漏了:副本留尸)。
    pub fn register_chain(&mut self, tokens: &[u32], table: &[u32]) -> StoreResult<CacheUpdate> {
        let up = self.kv.register_chain(tokens, table)?;
        self.drop_host_evicted(&up.evicted)?;
        Ok(up)
    }

    /// 归还会话块表。
    /// **增量链登记**(B-3e-2 第 2b 项;见 [`KvStore::register_chain_inc`])。
    pub fn register_chain_inc(
        &mut self,
        parent0: ChainHash,
        blocks: &[(&[u32], u32)],
    ) -> StoreResult<crate::kv::CacheUpdate> {
        self.kv.register_chain_inc(parent0, blocks)
    }

    /// 释放整张表的引用(表回空;与逐块 `release` 同源)。
    pub fn release_table(&mut self, table: &mut Vec<u32>) -> StoreResult<()> {
        self.kv.release_table(table)
    }

    /// 引用加一(前缀复用)。
    pub fn incref(&mut self, slot: u32) -> StoreResult<()> {
        self.kv.incref(slot)
    }

    /// 槽引用数(诊断/前置判定:下沉前必须知道"是不是只有会话持着")。
    pub fn slot_refs(&self, slot: u32) -> u32 {
        self.kv.slot_refs(slot)
    }

    /// **引用减一**(B-3e-2 第 2 项:会话放手一块 —— 下沉前的必经步)。
    /// 归零 = 槽回自由队列(**不要**在这里放给未持有者:账由 `KvStore` 判)。
    pub fn decref(&mut self, slot: u32) -> StoreResult<()> {
        self.kv.decref(slot)
    }

    /// 驱逐候选(叶 + 未保护;策略由调用方出序)。
    pub fn evict_candidates(&self, protect: &[ChainHash]) -> Vec<EvictCandidate> {
        self.kv.evict_candidates(protect)
    }

    /// 一块现在在哪(不在账里 = 响亮拒)。
    pub fn carrier_of(&self, chain: ChainHash) -> StoreResult<Carrier> {
        match self.kv.locate(chain) {
            Some(crate::tier::Tier::Gpu) => Ok(Carrier::Device(self.kv.bound_slot(chain)?)),
            Some(crate::tier::Tier::Host) => Ok(Carrier::Host),
            Some(crate::tier::Tier::Disk) => Ok(Carrier::Disk),
            None => Err(StoreError::Unknown(format!("块 {chain:?} 不在账里"))),
        }
    }

    /// 报告。
    pub fn report(&self) -> LegStats {
        let kv = self.kv.stats();
        LegStats {
            class: self.class,
            epoch: self.layout_epoch(),
            block_bytes: self.kv.block_bytes(),
            block_tokens: self.kv.block_tokens(),
            slots_total: kv.slots_total,
            slots_free: kv.slots_free,
            cached_blocks: kv.cached_blocks,
            host_blocks: kv.host_blocks,
            gpu_bytes: kv.gpu_bytes,
            occupied_bytes: self.occupied_bytes(),
            movable_bytes: self.movable_bytes(),
            host_bytes: self.host.used_bytes(),
            host_budget_bytes: self.host.budget_bytes(),
            ops: self.ops,
        }
    }

    /// 片段列表(内部;换出换回用)。
    fn pieces(&self) -> StoreResult<&[PieceSpec]> {
        self.layout
            .as_ref()
            .map(|l| l.pieces.as_slice())
            .ok_or_else(|| {
                StoreError::Invariant(
                    "块腿未注册池布局(engine 应在 boot 调 attach_pool)".into(),
                )
            })
    }
}

/// 命中面(engine 侧三元组;三者同序同长)。
#[derive(Clone, Debug, Default)]
pub struct ChainHit {
    /// 命中整块数。
    pub matched_blocks: usize,
    /// 逐块设备槽(逻辑序)。`None` = 内容在**宿主层**(S3-3b:由 engine 在段边界
    /// 调 `onload_block` 回灌,不再"整条拒")。
    pub slots: Vec<Option<u32>>,
    /// 内容身份链(快照键的真源)。
    pub hashes: Vec<ChainHash>,
}

impl ChainHit {
    /// 全部命中块都已设备驻留?
    pub fn all_resident(&self) -> bool {
        self.slots.iter().all(|s| s.is_some())
    }

    /// 需回灌的块身份(逻辑序;S3-3b 的 engine 入口)。
    pub fn pending_host(&self) -> Vec<ChainHash> {
        self.slots
            .iter()
            .zip(&self.hashes)
            .filter(|(s, _)| s.is_none())
            .map(|(_, h)| *h)
            .collect()
    }
}

// ───────────────────────────── 数据面(自由函数) ─────────────────────────────

// ─────────────────── 搬运计划 / 记账(纯 host;§21 裁决)───────────────────
//
// **store 不做 I/O**(连 `face` 都不认):它只出**计划**(算片 + 借宿主台)与**记账**
// (落账 / 摘绑定 / 回滚)。搬运由 **engine 编排**:取计划 → `face.swap` → 回执 → 记账。
// 形态律:
// - `plan_*` 纯 host(任意时刻可调,只动账与内存;组片走注册的布局);
// - `commit_*` 纯 host(搬运成功后回账);
// - `abort_*` 纯 host(搬运失败后回滚 —— **内容已随命令消失 ⇒ 条目一并删**,不留假账)。

/// **换出计划**(纯 host;不碰 face)。
pub struct OutPlan {
    /// 逐块(身份, 原设备槽) —— 记账与回滚用。
    pub blocks: Vec<(ChainHash, u32)>,
    /// 设备 → 连接器的片(每块 × 每层片段)。
    pub pieces: Vec<SwapPiece>,
    /// 宿主侧槽台(engine 原箱交给 `face.swap`,回执后交还 [`commit_offload`])。
    pub table: SlotTable,
}

/// **逐块下沉计划**(B-3e-2;2026-10-18):给定"调用方已决定沉下去"的块清单,
/// **逐块**出票 —— 与链级 [`plan_offload`] 的两处粗化正相反:
///
/// | 维度 | 链级 `plan_offload(tail)` | 逐块 `plan_offload_blocks(chains)` |
/// |---|---|---|
/// | 单位 | **整条链**(root→tail,all-or-nothing) | **逐块**(给了哪些就办哪些) |
/// | 活跃引用 | **一块活跃 ⇒ 整链否决**(`Busy`) | 活跃块**跳过并记因**,不连坐别的块 |
///
/// 为什么要它:活跃会话的链**部分宿主驻留**是单卡长 ctx(G1 262k)的前置 ——
/// 工作集(W)留设备、被选择层踢出的块沉走;链级票做不到这件事。
///
/// **前置(调用方职责)**:要让某块可沉,必须先**释放会话对它的引用**
/// (`decref`;否则 `refs != 1`,该块会被记成 [`SkipWhy::LiveRefs`] 跳过)。
/// 这正是"占位 + 回灌"流程里会话表改写成占位符那一步在账上的对应动作。
///
/// 纯 host(算片 + 借台);无票块**不影响**有票块;全无票 ⇒ `blocks` 为空、
/// 不借宿主台(调用方零副作用即可早退)。
pub fn plan_offload_blocks(
    leg: &mut BlockLeg,
    chains: &[ChainHash],
) -> StoreResult<BlockOutPlan> {
    leg.pieces()?; // 未注册布局 = 先拒
    let mut blocks: Vec<(ChainHash, u32)> = Vec::new();
    let mut skipped: Vec<(ChainHash, SkipWhy)> = Vec::new();
    for c in chains {
        match leg.kv.locate(*c) {
            None => skipped.push((*c, SkipWhy::Unknown)),
            Some(crate::tier::Tier::Host) => skipped.push((*c, SkipWhy::AlreadyHost)),
            Some(crate::tier::Tier::Disk) => skipped.push((*c, SkipWhy::Unknown)), // S6 未接线
            Some(crate::tier::Tier::Gpu) => {
                // 逐块判活跃(链级票在这一步是"整链否决")
                match leg.kv.bound_slot(*c) {
                    Ok(slot) if leg.kv.slot_refs(slot) == 1 => blocks.push((*c, slot)),
                    Ok(_) => skipped.push((*c, SkipWhy::LiveRefs)),
                    Err(_) => skipped.push((*c, SkipWhy::Unknown)),
                }
            }
        }
    }
    if blocks.is_empty() {
        return Ok(BlockOutPlan { blocks, pieces: Vec::new(), table: SlotTable::new(Vec::new()), skipped });
    }
    let bufs = leg.host.lend_slots(blocks.len())?; // 预算在借台处查(超界零副作用)
    let pieces = out_pieces(leg, &blocks)?;
    Ok(BlockOutPlan { blocks, pieces, table: SlotTable::new(bufs), skipped })
}

/// **宿主占位符**(B-3e-2):会话块表里表示"该块内容在宿主层、无设备槽"的哨兵。
///
/// 为什么需要它:部分驻留档下链上会**出洞**(块被下沉),而会话块表是 `Vec<u32>`
/// 槽号数组 —— 洞必须有明确编码才会被响亮拒(而不是当页号 0 静默读错页)。
/// 引擎侧同名常量 = `engine::scheduler::PENDING_LOAD_PLACEHOLDER`(同值 `u32::MAX`;
/// 两侧同源,store 侧只提供判定函数,不依赖 engine)。
pub const HOST_PLACEHOLDER: u32 = u32::MAX;

/// 是不是宿主占位符。
#[inline]
pub fn is_host_placeholder(slot: u32) -> bool {
    slot == HOST_PLACEHOLDER
}

/// 逐块下沉计划(见 [`plan_offload_blocks`])。
pub struct BlockOutPlan {
    /// 有票的块(身份, 原设备槽) —— 交给 [`commit_offload`] / [`abort_offload`]。
    pub blocks: Vec<(ChainHash, u32)>,
    /// 设备 → 连接器的片(每块 × 每层片段)。
    pub pieces: Vec<SwapPiece>,
    /// 宿主侧槽台(engine 原箱交给 `face.swap`)。
    pub table: SlotTable,
    /// **无票块 + 原因**(报告面:**不**否决有票块 —— 这正是本函数存在的理由)。
    pub skipped: Vec<(ChainHash, SkipWhy)>,
}

impl BlockOutPlan {
    /// 有票块数。
    pub fn len(&self) -> usize {
        self.blocks.len()
    }
    /// 一块都没票(调用方可零副作用早退)。
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }
}

/// 逐块下沉里**跳过**一块的原因(报告面;由账直接判,不靠调用方猜)。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SkipWhy {
    /// 不在缓存树里(传错身份 / 已被驱逐)。
    Unknown,
    /// 内容已在宿主层(重复请求;幂等跳过)。
    AlreadyHost,
    /// 设备块引用 > 1(仍有会话表持有)⇒ 调用方忘了先 `decref`。
    LiveRefs,
}

/// **回灌计划**(纯 host):宿主内容取出 + 设备槽分配 + 组片。
pub struct InPlan {
    /// 待回灌的块身份(与 `slots` 同序)。
    pub chains: Vec<ChainHash>,
    /// 新分配的设备槽(同序)。
    pub slots: Vec<u32>,
    /// 连接器 → 设备的片。
    pub pieces: Vec<SwapPiece>,
    /// 宿主侧槽台(engine 交给 `face.swap(In, …)`)。
    pub table: SlotTable,
}

/// 回执取回槽台(单主语义:命令把 connector 原箱奉还;类型不符 = 响亮拒)。
pub fn reclaim_slot_table(conn: Box<dyn Connector + Send>) -> StoreResult<SlotTable> {
    conn.unbox().downcast::<SlotTable>().map(|b| *b).map_err(|_| {
        StoreError::Invariant("swap 回执的连接器不是 SlotTable(实现面自洽性违约)".into())
    })
}

/// **换出计划**:整条链(§14.6 粒度律)—— 先整条校验(`chain_down_span`)再算片;
/// 已在宿主的块**幂等跳过**(同链/他链已搬走它)。
/// 活跃面(设备块引用 > 1)或无可搬块 ⇒ `Busy`(**零副作用**,调用方跳过该条)。
pub fn plan_offload(leg: &mut BlockLeg, tail: ChainHash) -> StoreResult<OutPlan> {
    leg.pieces()?; // 未注册布局 = 先拒
    leg.kv.chain_down_span(tail).ok_or_else(|| {
        StoreError::Busy(format!(
            "链 {tail:?} 不可整条下沉(链尾不在账 / 有设备块引用 > 1 / 全链已在宿主)"
        ))
    })?;
    let chain = leg.kv.chain_of(tail);
    let mut blocks: Vec<(ChainHash, u32)> = Vec::new();
    for b in &chain {
        if b.slot.is_some() {
            blocks.push((b.chain, leg.kv.bound_slot(b.chain)?));
        }
    }
    if blocks.is_empty() {
        return Err(StoreError::Busy(format!("链 {tail:?} 没有可搬的设备块")));
    }
    let bufs = leg.host.lend_slots(blocks.len())?; // 预算在借台处查(超界零副作用)
    let pieces = out_pieces(leg, &blocks)?;
    Ok(OutPlan { blocks, pieces, table: SlotTable::new(bufs) })
}

/// **换出记账**(搬运成功后调用;纯 host):内容装回宿主层 + 摘绑定还槽 + 计数。
pub fn commit_offload(
    leg: &mut BlockLeg,
    blocks: &[(ChainHash, u32)],
    table: SlotTable,
) -> StoreResult<usize> {
    let n = blocks.len();
    let items: Vec<(ChainHash, HostBlock)> =
        blocks.iter().map(|(c, _)| *c).zip(table.into_slots()).collect();
    leg.host.land(items)?;
    for (chain, _) in blocks {
        leg.kv.unbind(*chain)?; // 摘绑定 + 还槽(缓存那份引用随绑定一起还)
    }
    leg.ops.offloads += n as u64;
    leg.ops.offload_bytes += (n * leg.slot_bytes()) as u64;
    leg.ops.offload_batches += 1;
    Ok(n)
}

/// **换出回滚**(搬运失败后调用;纯 host):内容已随命令消失 ⇒ **条目一并删**
/// (留着 `slot = None` 的条目就是假账,下次回灌必 `Unknown`)。
/// 槽台随失败的 connector 一起消失(内存释放),不在此回收。
pub fn abort_offload(leg: &mut BlockLeg, blocks: &[(ChainHash, u32)]) {
    for (chain, _) in blocks {
        let _ = leg.kv.drop_entry(*chain);
    }
}

/// **回灌计划**(纯 host):逐块取宿主内容 → 分配槽 → 组片。
/// 任一环节失败即回滚已做的部分(内容放回 / 槽归还),不留半态。
pub fn plan_onload(leg: &mut BlockLeg, chains: &[ChainHash]) -> StoreResult<InPlan> {
    if chains.is_empty() {
        return Ok(InPlan {
            chains: Vec::new(),
            slots: Vec::new(),
            pieces: Vec::new(),
            table: SlotTable::new(Vec::new()),
        });
    }
    leg.pieces()?; // 未注册布局 = 先拒(别先动宿主)
    let mut bufs: Vec<HostBlock> = Vec::with_capacity(chains.len());
    for c in chains {
        match leg.host.take(*c) {
            Ok(b) => bufs.push(b),
            Err(e) => {
                leg.host.recycle_many(bufs);
                return Err(e);
            }
        }
    }
    let mut slots: Vec<u32> = Vec::with_capacity(chains.len());
    for _ in chains {
        match leg.kv.alloc_slot() {
            Ok(s) => slots.push(s),
            Err(e) => {
                for s in &slots {
                    let _ = leg.kv.decref(*s);
                }
                leg.host.recycle_many(bufs);
                return Err(e);
            }
        }
    }
    let planes: Vec<(ChainHash, u32)> = chains.iter().copied().zip(slots.iter().copied()).collect();
    let pieces = in_pieces(leg, &planes, 0)?;
    Ok(InPlan { chains: chains.to_vec(), slots, pieces, table: SlotTable::new(bufs) })
}

/// **回灌记账**(搬运成功后调用;纯 host):缓冲回池 + 绑定回设备(引用 +1)+ 计数。
/// 返回与入参**同序**的槽(调用方写回块表)。
pub fn commit_onload(
    leg: &mut BlockLeg,
    chains: &[ChainHash],
    slots: &[u32],
    table: SlotTable,
) -> StoreResult<Vec<u32>> {
    leg.host.recycle_many(table.into_slots());
    for (chain, slot) in chains.iter().zip(slots) {
        leg.kv.rebind(*chain, *slot)?;
    }
    leg.ops.onloads += chains.len() as u64;
    leg.ops.onload_bytes += (chains.len() * leg.slot_bytes()) as u64;
    leg.ops.onload_batches += 1;
    Ok(slots.to_vec())
}

/// **回灌回滚**(搬运失败后调用;纯 host):槽归还 + 条目删(内容已丢)。
pub fn abort_onload(leg: &mut BlockLeg, chains: &[ChainHash], slots: &[u32]) {
    for s in slots {
        let _ = leg.kv.decref(*s);
    }
    for c in chains {
        let _ = leg.kv.drop_entry(*c);
    }
}

/// 组换出片:每块 × 每层片段一片
/// (`dev_off = 槽 × 每页字节`;`conn_off` = 该层片段在逻辑块内的起点)。
fn out_pieces(leg: &BlockLeg, planes: &[(ChainHash, u32)]) -> StoreResult<Vec<SwapPiece>> {
    let specs = leg.pieces()?;
    let mut out = Vec::with_capacity(planes.len() * specs.len());
    for (k, (_chain, slot)) in planes.iter().enumerate() {
        let mut conn_off = 0usize;
        for p in specs {
            out.push(SwapPiece {
                block: p.buf.id,
                dev_off: *slot as usize * p.stride_bytes,
                conn_off,
                len: p.stride_bytes,
                slot: ConnSlot(k as u64),
            });
            conn_off += p.stride_bytes;
        }
    }
    Ok(out)
}

/// 组换入片(与 [`out_pieces`] 对称;`base_slot` = 台面槽位起点)。
fn in_pieces(
    leg: &BlockLeg,
    planes: &[(ChainHash, u32)],
    base_slot: u64,
) -> StoreResult<Vec<SwapPiece>> {
    let specs = leg.pieces()?;
    let mut out = Vec::with_capacity(planes.len() * specs.len());
    for (k, (_chain, slot)) in planes.iter().enumerate() {
        let mut conn_off = 0usize;
        for p in specs {
            out.push(SwapPiece {
                block: p.buf.id,
                dev_off: *slot as usize * p.stride_bytes,
                conn_off,
                len: p.stride_bytes,
                slot: ConnSlot(base_slot + k as u64),
            });
            conn_off += p.stride_bytes;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::TOKENS_PER_BLOCK;
    use owl_cpu::CpuFace;
    use owl_iface::contract::Dtype;
    use owl_iface::contract::{DeviceClient, SwapDir};

    /// **测试侧编排**(模拟 engine):取计划 → `face.swap` → 回执 → 记账。
    /// 生产里这段编排在 engine(§21);store 侧只出计划与记账 —— 这里替身是为了
    /// 让 store 的搬运闭环仍能在**无 engine** 下测到端到端。
    async fn do_offload(leg: &mut BlockLeg, face: &mut CpuFace, tail: ChainHash) -> StoreResult<usize> {
        let (blocks, pieces, table) = {
            let p = plan_offload(leg, tail)?;
            (p.blocks, p.pieces, p.table)
        };
        let conn = face
            .swap(SwapDir::Out, &pieces, Box::new(table))
            .await
            .map_err(|e| StoreError::Invariant(format!("测试编排 swap(Out) 失败:{e}")))?;
        let table = reclaim_slot_table(conn)?;
        commit_offload(leg, &blocks, table)
    }

    /// 同上:回灌一条命令。
    async fn do_onload(leg: &mut BlockLeg, face: &mut CpuFace, chains: &[ChainHash]) -> StoreResult<Vec<u32>> {
        let (cs, slots, pieces, table) = {
            let p = plan_onload(leg, chains)?;
            (p.chains, p.slots, p.pieces, p.table)
        };
        let conn = face
            .swap(SwapDir::In, &pieces, Box::new(table))
            .await
            .map_err(|e| StoreError::Invariant(format!("测试编排 swap(In) 失败:{e}")))?;
        let table = reclaim_slot_table(conn)?;
        commit_onload(leg, &cs, &slots, table)
    }

    const PIECE_ELEMS: usize = 32; // 128B/片段
    const PAGE_BYTES: usize = PIECE_ELEMS * 4;

    /// 两片段 × `slots` 槽的设备池(CpuFace 真 face)。
    async fn pieces_n(face: &mut CpuFace, slots: usize) -> Vec<PieceSpec> {
        let mut out = Vec::new();
        for _ in 0..2 {
            let b = face.alloc(Dtype::F32, PIECE_ELEMS * slots, "t").await.expect("池块");
            out.push(PieceSpec { buf: b, stride_bytes: PAGE_BYTES });
        }
        out
    }

    fn leg_with(slots: usize) -> BlockLeg {
        BlockLeg::open(LegConfig {
            slots,
            block_tokens: TOKENS_PER_BLOCK,
            block_bytes: PIECE_ELEMS * 4 * 2, // 两片段合计 = 一个逻辑块
            cache: CachePolicy::Cap(4 + slots),
            host_bytes: 1 << 20,
            class: BlockClass::KvPage,
            label: "test",
        })
        .expect("开腿")
    }

    fn leg() -> BlockLeg {
        leg_with(2)
    }

    /// 往"槽 slot 的每个片段"写可辨识图样(经带偏移写;与换出读同一路径)。
    async fn seed_slot(face: &mut CpuFace, pieces: &[PieceSpec], slot: u32, seed: u8) {
        for p in pieces {
            let pat: Vec<u8> = (0..p.stride_bytes).map(|i| seed.wrapping_add(i as u8)).collect();
            upload_span(face, &pat, &p.buf, slot as usize * p.stride_bytes)
                .await
                .expect("seed 写");
        }
        face.sync().await.expect("sync");
    }

    /// 读"槽 slot 的每个片段"(换出读路径)。
    async fn read_slot(face: &mut CpuFace, pieces: &[PieceSpec], slot: u32) -> Vec<u8> {
        let mut out = Vec::new();
        for p in pieces {
            let mut dst = vec![0u8; p.stride_bytes];
            download_span(face, &p.buf, slot as usize * p.stride_bytes, &mut dst)
                .await
                .expect("读");
            out.extend_from_slice(&dst);
        }
        out
    }

    #[test]
    fn geometry_is_checked_loudly() {
        // slots = 0 = 未启用腿(合法);误用 = 池耗尽响亮拒
        let mut off = BlockLeg::open(LegConfig {
            slots: 0,
            block_tokens: 32,
            block_bytes: 1,
            cache: CachePolicy::Cap(0),
            host_bytes: 0,
            class: BlockClass::KvPage,
            label: "off",
        })
        .expect("未启用腿可开");
        let mut t = Vec::new();
        assert!(
            matches!(off.ensure_for_len(&mut t, 32), Err(StoreError::Exhausted(_))),
            "零槽腿误用必须响亮拒"
        );
        let mut l = leg();
        let bad = PoolLayout { class: BlockClass::KvPage, epoch: 1, pieces: vec![] };
        assert!(l.attach_pool(bad).is_err(), "空布局(0 字节)必须拒");
        assert!(l.layout_epoch().is_none());
    }

    #[tokio::test]
    async fn offload_then_onload_roundtrip_is_byte_exact() {
        let mut face = CpuFace::default();
        let pieces = pieces_n(&mut face, 2).await;
        let mut l = leg();
        assert_eq!(l.attach_pool(PoolLayout { class: BlockClass::KvPage, epoch: 7, pieces: pieces.clone() }).expect("注册"), 7);
        assert_eq!(l.layout_epoch(), Some(7));

        // 建一条链(两页 ⇒ 槽 0/1),登记缓存后释放会话引用 ⇒ 缓存独占(可换出)
        let tokens: Vec<u32> = (0..64).collect();
        let mut table = Vec::new();
        l.ensure_for_len(&mut table, 64).expect("块表");
        l.register_chain(&tokens, &table).expect("登记");
        let hit = l.match_prefix(&tokens).expect("命中");
        assert_eq!((hit.matched_blocks, hit.slots.len(), hit.hashes.len()), (2, 2, 2));
        let tail = hit.hashes[1];
        let tail_slot = hit.slots[1].expect("尾块在设备上");
        l.release_table(&mut table).expect("会话归还");
        assert_eq!(l.report().slots_free, 0, "两页都被缓存持着(缓存独占)");

        // 种图样(直接写设备),再换出
        seed_slot(&mut face, &pieces, tail_slot, 0x40).await;
        let before = read_slot(&mut face, &pieces, tail_slot).await;
        let free_before = l.report().slots_free;
        do_offload(&mut l, &mut face, tail).await.expect("换出");
        let rep = l.report();
        // 新 API 是**链级**下沉(§14.6):tail 一站 ⇒ 整条链(2 块 = 4 片段)全部进宿主。
        assert_eq!(real(rep.host_bytes), PAGE_BYTES * 4, "宿主应持整条链(2 块)");
        assert_eq!(rep.slots_free, free_before + 2, "摘绑定 ⇒ 两槽回池");
        assert_eq!(l.carrier_of(tail).expect("在账"), Carrier::Host, "载体 = 宿主");
        // S3-3b:宿主格命中**不再整条拒** —— 命中面逐块给出载体,engine 段边界回灌
        let h2 = l.match_prefix(&tokens).expect("宿主格也可命中");
        assert_eq!(h2.matched_blocks, 2, "内容仍在(只是载体在宿主)");
        assert_eq!(h2.slots[1], None, "尾块在宿主层");
        assert!(!h2.all_resident(), "整条未全驻留");
        assert_eq!(h2.pending_host(), vec![hit.hashes[0], tail], "整条链待回灌(链头段在内)");

        // 换回:整条链回到设备(host 只留 0)
        let back = do_onload(&mut l, &mut face, &h2.pending_host()).await.expect("换回");
        assert_eq!(back.len(), 2, "整条回灌");
        let slot2 = back[1];
        assert_eq!(l.carrier_of(tail).expect("在账"), Carrier::Device(slot2));
        assert_eq!(real(l.report().host_bytes), 0, "宿主缓冲已回收");
        let after = read_slot(&mut face, &pieces, slot2).await;
        assert_eq!(after, before, "换出→换回必须逐字节一致");
    }

    #[tokio::test]
    async fn offload_is_rejected_when_a_session_holds_or_host_is_off() {
        let mut face = CpuFace::default();
        let pieces = pieces_n(&mut face, 2).await;
        // ① 宿主层关闭(host_bytes = 0)= 默认档:第一枪响亮拒
        let mut closed = BlockLeg::open(LegConfig {
            slots: 2,
            block_tokens: TOKENS_PER_BLOCK,
            block_bytes: PIECE_ELEMS * 4 * 2,
            cache: CachePolicy::Cap(4),
            host_bytes: 0,
            class: BlockClass::KvPage,
            label: "closed",
        })
        .expect("开腿");
        closed
            .attach_pool(PoolLayout { class: BlockClass::KvPage, epoch: 1, pieces: pieces.clone() })
            .expect("注册");
        let tokens: Vec<u32> = (0..32).collect();
        let mut table = Vec::new();
        closed.ensure_for_len(&mut table, 32).expect("块表");
        closed.register_chain(&tokens, &table).expect("登记");
        let tail = closed.match_prefix(&tokens).expect("命中").hashes[0];
        closed.release_table(&mut table).expect("归还");
        let err = do_offload(&mut closed, &mut face, tail).await.expect_err("关层必拒");
        assert!(matches!(err, StoreError::Exhausted(_)), "得 {err:?}");

        // ② 会话仍持引用 ⇒ 活跃面不可下沉
        let mut l = leg();
        l.attach_pool(PoolLayout { class: BlockClass::KvPage, epoch: 1, pieces: pieces.clone() })
            .expect("注册");
        let mut t2 = Vec::new();
        l.ensure_for_len(&mut t2, 32).expect("块表");
        l.register_chain(&tokens, &t2).expect("登记");
        let c2 = l.match_prefix(&tokens).expect("命中").hashes[0];
        let err = do_offload(&mut l, &mut face, c2).await.expect_err("活跃面必拒");
        assert!(matches!(err, StoreError::Busy(_)), "活跃面拒 = Busy(调用方据此跳过候选),得 {err:?}");
    }

    /// **§14.6 链级卸载**:一条链(尾 → 头)一次搬完 —— 链头段也在宿主,
    /// 下次同前缀命中**从链头就是宿主格**(`pending > 0` ⇒ 回灌真会触发)。
    #[tokio::test]
    async fn offload_chain_moves_the_whole_chain_tail_to_head() {
        let mut face = CpuFace::default();
        let pieces = pieces_n(&mut face, 3).await;
        let mut l = leg_with(3);
        l.attach_pool(PoolLayout { class: BlockClass::KvPage, epoch: 1, pieces: pieces.clone() })
            .expect("注册");

        // 三块链(96 token);登记后会话归还 ⇒ 缓存独占
        let tokens: Vec<u32> = (0..96).collect();
        let mut table = Vec::new();
        l.ensure_for_len(&mut table, 96).expect("块表");
        l.register_chain(&tokens, &table).expect("登记");
        let hit = l.match_prefix(&tokens).expect("命中");
        assert_eq!(hit.matched_blocks, 3);
        let (head, tail) = (hit.hashes[0], hit.hashes[2]);
        l.release_table(&mut table).expect("会话归还");

        // 种可辨识图样,记下每块内容(逻辑序)
        let mut before = Vec::new();
        for (i, s) in hit.slots.iter().enumerate() {
            let slot = s.expect("设备上");
            seed_slot(&mut face, &pieces, slot, 0x10 * (i as u8 + 1)).await;
            before.push(read_slot(&mut face, &pieces, slot).await);
        }

        // 候选:整条链一张票(dev_blocks = 3)
        let cands = l.chain_offload_candidates(&[], 1);
        assert_eq!(cands.len(), 1, "只有一条链:{cands:?}");
        assert_eq!(cands[0].tail, tail);
        assert_eq!(cands[0].head, head);
        assert_eq!((cands[0].dev_blocks, cands[0].blocks), (3, 3));

        // 整条搬:链头段也必须到宿主(§14.6 律 1:半截链 = 碎片)
        let moved = do_offload(&mut l, &mut face, tail).await.expect("整条下沉");
        assert_eq!(moved, 3);
        for (i, h) in hit.hashes.iter().enumerate() {
            assert_eq!(l.carrier_of(*h).expect("在账"), Carrier::Host, "第 {i} 块应在宿主");
        }
        assert_eq!(l.report().slots_free, 3, "三块设备槽全回池");

        // 命中面:三格全 host ⇒ 三块待回灌(engine 在段边界逐块回灌)
        let h2 = l.match_prefix(&tokens).expect("宿主格仍命中");
        assert_eq!(h2.matched_blocks, 3);
        assert_eq!(h2.pending_host().len(), 3, "真源:待回灌清单非空");
        assert!(!h2.all_resident());

        // 整条再搬一次:幂等(全在宿主 ⇒ 无槽可腾 ⇒ 响亮拒,零副作用)
        let err = do_offload(&mut l, &mut face, tail).await.expect_err("空票必拒");
        assert!(matches!(err, StoreError::Busy(_)), "得 {err:?}");

        // 回灌:逐字节复原(逻辑序)
        for (i, h) in h2.hashes.iter().enumerate() {
            let slot = do_onload(&mut l, &mut face, &[*h]).await.expect("回灌")[0];
            assert_eq!(read_slot(&mut face, &pieces, slot).await, before[i], "块 {i} 逐字节一致");
        }
        assert_eq!(l.report().host_blocks, 0, "宿主副本已全部移出");
    }

    /// **批量回灌(§15 批面)**:一条 swap 搬 N 块 —— 同序新槽、逐字节复原、
    /// **命令计数 = 1**(块计数 = N);空表 = 空结果(不触命令)。
    #[tokio::test]
    async fn onload_blocks_moves_n_blocks_in_one_command() {
        let mut face = CpuFace::default();
        let pieces = pieces_n(&mut face, 3).await;
        let mut l = leg_with(3);
        l.attach_pool(PoolLayout { class: BlockClass::KvPage, epoch: 1, pieces: pieces.clone() })
            .expect("注册");

        // 两块的链;登记后归还 ⇒ 缓存独占
        let tokens: Vec<u32> = (0..64).collect();
        let mut table = Vec::new();
        l.ensure_for_len(&mut table, 64).expect("块表");
        l.register_chain(&tokens, &table).expect("登记");
        let hit = l.match_prefix(&tokens).expect("命中");
        let hashes = hit.hashes.clone();
        l.release_table(&mut table).expect("会话归还");

        // 种图样(逐块可辨识)后整条沉宿主
        let mut before = Vec::new();
        for (i, s) in hit.slots.iter().enumerate() {
            let slot = s.expect("设备上");
            seed_slot(&mut face, &pieces, slot, 0x20 * (i as u8 + 1)).await;
            before.push(read_slot(&mut face, &pieces, slot).await);
        }
        let tail = *hashes.last().expect("尾块");
        assert_eq!(do_offload(&mut l, &mut face, tail).await.expect("整条沉"), 2);
        let ops0 = l.report().ops;
        assert_eq!((ops0.offload_batches, ops0.offloads), (1, 2), "换出:1 命令 2 块");

        // 批量回灌:**一条命令**
        let slots = do_onload(&mut l, &mut face, &hashes).await.expect("批量回灌");
        assert_eq!(slots.len(), 2, "同序返回两个槽");
        let ops1 = l.report().ops;
        assert_eq!((ops1.onload_batches, ops1.onloads), (1, 2), "回灌:1 命令 2 块");
        for (i, slot) in slots.iter().enumerate() {
            assert_eq!(read_slot(&mut face, &pieces, *slot).await, before[i], "块 {i} 逐字节一致");
        }
        assert_eq!(l.report().host_blocks, 0, "宿主副本已全部取出");
        // 空表 = 空结果(不触命令、不报错 —— 调用方的"无待回灌"路径)
        assert!(do_onload(&mut l, &mut face, &[]).await.expect("空表").is_empty());
    }

    /// 活跃面(会话持有整条链)⇒ 链级候选不出票、整条搬**零副作用**拒。
    #[tokio::test]
    async fn chain_offload_refuses_while_a_session_holds() {
        let mut face = CpuFace::default();
        let pieces = pieces_n(&mut face, 2).await;
        let mut l = leg();
        l.attach_pool(PoolLayout { class: BlockClass::KvPage, epoch: 1, pieces }).expect("注册");
        let tokens: Vec<u32> = (0..64).collect();
        let mut table = Vec::new();
        l.ensure_for_len(&mut table, 64).expect("块表");
        l.register_chain(&tokens, &table).expect("登记");
        let tail = l.match_prefix(&tokens).expect("命中").hashes[1];
        // 会话仍持引用
        assert!(l.chain_offload_candidates(&[], 4).is_empty(), "活跃面不得出票");
        let err = do_offload(&mut l, &mut face, tail).await.expect_err("活跃面必拒");
        assert!(matches!(err, StoreError::Busy(_)), "得 {err:?}");
        assert_eq!(l.report().host_blocks, 0, "拒 = 零副作用(不许搬半截)");
        assert_eq!(l.report().ops.offloads, 0);
        l.release_table(&mut table).expect("收拾");
    }

    /// 关宿主层时:压力驱逐仍走纯设备路径(行为与接线前一致)。
    #[test]
    fn pressure_without_host_never_touches_host_tier() {
        let mut l = BlockLeg::open(LegConfig {
            slots: 2,
            block_tokens: TOKENS_PER_BLOCK,
            block_bytes: 1 << 20,
            cache: CachePolicy::Cap(2),
            host_bytes: 0,
            class: BlockClass::KvPage,
            label: "no-host",
        })
        .expect("开腿");
        let mut t = Vec::new();
        l.ensure_for_len(&mut t, 32).expect("块表");
        l.register_chain(&(0..32u32).collect::<Vec<_>>(), &t).expect("登记");
        l.release_table(&mut t).expect("归还");
        let mut t2 = Vec::new();
        l.ensure_for_len(&mut t2, 64).expect("压力驱逐放行");
        let rep = l.report();
        assert_eq!(rep.host_bytes, 0, "关宿主层:压力驱逐不得碰宿主");
        assert_eq!(rep.ops.offloads, 0);
    }

    /// 字节口径直读(测试可读性;`usize` 本身即字节)。
    fn real(v: usize) -> usize {
        v
    }
}

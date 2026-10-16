//! 设备状态块池(StatePool;2026-10-01 engine.rs 拆分立)。
//!
//! 引擎全部设备侧状态块的**治理对象**:KV paged 池(kvs)/ GDN 状态格
//! (gdns)/ GDN 快照池(snaps)/ 持久块表(bt)+ 池几何(page/nb/
//! paged/x/dims)。职责 = 分配([`StatePool::alloc`])+ 叶子句柄发放
//! (kv_caches/gdn_caches/bt_leaf)+ 状态治理(reset_gdn/capture_snap/
//! restore_snap/write_bt)。块账房(host 侧记账)不在此 —— 那是
//! `owl_store` 的块腿(槽池 + 内容身份前缀树);本池只管设备缓冲与池内快照 LRU。
//!
//! 布局策略表驱动(REQ-HW-01):KV 布局由 `kv_paged_policy(dtype)` 分发
//! —— Some = paged classic(kc [nb,Hkv,hd/x,page,x] / vc [nb,Hkv,hd,page];
//! 恒等块表:物理块 = 逻辑块,物理槽 = pos);None = legacy token-major
//! 直排(page = 容量,nb = 1)。
//!
//! dtype 定盘(F5 整模切换):KV cache 随 spec.dtype;GDN state 恒 f32
//! (三方先例 + 旧世界律);slots/kv_lens/bt f32(契约 5)。

use owl_iface::contract::{Bytes, DeviceClient, Dtype, ModelError};
use owl_store::SnapKey;
use owl_models::interpreters::eval_ops;
use owl_models::module::kv_paged_policy;
use owl_models::TensorOps;

type Result<T> = std::result::Result<T, ModelError>;

pub(crate) type BlockN = (Bytes, usize); // (块句柄, 元素数;重置分块写用)



/// GDN 状态格容量(**与会话数绑定,与 max_seq_tokens 解耦**)。
/// 格语义 = 每会话一格(SessionTable 分配/释放);按位分配是历史包袱:
/// s=4096 时 rec(1 MiB/格/层 × 18 层)将达 72 GB,而真实需求 = 格数。
/// 8 = 覆盖需求并发上限(§二 并发 ≤8);会话 close 即释放格号。
/// 槽数 = EngineKnobs.gdn_slots(显式传入;原 OWL_GDN_SLOTS,缺省 8)。
///
/// GDN 快照槽(E2c):key = [`SnapKey`](**边界块的内容身份**;逐块父链哈希,
/// `owl_store::ChainCursor`)。为什么不是物理块号:物理块号是会被回收复用的
/// 槽号 ⇒ 旧键会命中别人的状态(静默错值;store 卷 §9.1)。键类型来自 store
/// ⇒ **物理槽号在类型上就进不了快照键**。
/// bufs = 逐层逐块行宽副本(与 gdns 同序同构,行宽 = elems / GDN_SLOTS)。
/// 池静态预分配,驱逐 = 换 key 覆盖写(零 alloc/free 往返)。
pub(crate) struct GdnSnapSlot {
    key: Option<SnapKey>,
    last_use: u64,
    bufs: Vec<BlockN>,
}

pub(crate) struct KvBlocks {
    pub(crate) k_cache: BlockN,
    pub(crate) v_cache: BlockN,
}

struct GdnBlocks {
    conv_q: BlockN,
    conv_k: BlockN,
    conv_v: BlockN,
    rec: BlockN,
}

/// 元素字节宽(账口径唯一产地:坑册 §42 —— 块句柄自述 dtype 是标注,
/// 但**同进程内** GPU 面的 dtype 就是真值;跨 face 的字节真源另见调用方)。
pub(crate) fn elem_bytes(d: Dtype) -> usize {
    match d {
        Dtype::F16 | Dtype::BF16 => 2,
        Dtype::F32 | Dtype::U32 => 4,
        Dtype::U8 => 1,
    }
}

/// prefill 块构造所需维度账(run 期从 spec 提取;Copy 免 Clone 传播)
#[derive(Clone, Copy)]
pub(crate) struct ModelDims {
    pub(crate) hq: usize,
    pub(crate) hkv: usize,
    pub(crate) hd: usize,
    pub(crate) nk: usize,
    pub(crate) hk: usize,
    pub(crate) nv: usize,
    pub(crate) hv: usize,
    pub(crate) hidden: usize,
    pub(crate) dtype: Dtype,
}

/// GDN 状态叶子四件套(slots 由调用方按步注入;见 gdn_caches)
pub(crate) struct GdnLeaf {
    pub(crate) conv_q: TensorOps,
    pub(crate) conv_k: TensorOps,
    pub(crate) conv_v: TensorOps,
    pub(crate) rec: TensorOps,
}

/// 设备状态块池(KV paged 池 / GDN 状态格 / 快照池 / 持久块表)。
/// 治理对象(charter A1 味):S2 图档与执行器只经方法面触碰状态块,
/// 不直摸块句柄。
/// 池装配计划(engine run 从 knobs/spec_mode 派生;显式依赖律的
/// 参数对象 —— 原散列布尔/数值参数群收编:量化 = KvQuant 枚举,
/// 投机形态 = SpecMode 枚举,互斥语义由派生保证,调用点零自洽负担)
#[derive(Clone, Debug)]
pub(crate) struct PoolPlan {
    /// 单会话容量(= max_seq_tokens)
    pub seq_tokens: usize,
    /// 池容量(治理/钉值定谳后的 token 数)
    pub pool_tokens: usize,
    /// 主池 KV 量化
    pub kv: owl_models::env::KvQuant,
    /// FI 直读面启用的池量化(None = 无 FI 面)。
    /// **不是影子池量化** —— 影子池已随统一契约 P3 删除;本字段现在只表示
    /// "FI prefill 面启用",量化恒等于主池 `kv`(名随实)
    pub fi_quant: Option<owl_models::env::KvQuant>,
    /// 草稿池量化
    pub draft_kv: owl_models::env::KvQuant,
    /// 投机形态(快照/MTP 链/草稿池三件的唯一事实来源)
    pub spec_mode: crate::spec_runtime::SpecMode,
    /// 投机深度(verify 块 = depth+1;verify v2 scratch 定形用;0 = off)
    pub spec_depth: usize,
    /// XQA verify 臂(SPEC_DEC=1 `M_TILESIZE=32`)
    pub xqa_verify: bool,
    /// XQA 裸 decode 臂(SPEC_DEC=1 `M_TILESIZE=16`)
    pub xqa_decode: bool,
    /// **XQA prefill 臂**(chunked q=T;同一份 `M_TILESIZE=32` 实例化)
    pub xqa_prefill: bool,
    /// GDN 状态格容量
    pub gdn_slots: usize,
    /// spec 快照池深度
    pub snap_max: usize,
    /// VRAM 预算目标占比(0 = 治理禁用)
    pub vram_target: f64,
    /// VRAM 预留 MiB
    pub vram_reserve_mb: u64,
    /// GDN 画像 dump 全槽
    pub dump_all: bool,
    /// **prefill 桶 padding 的 KV dump 页**(M2b,2026-10-15;设计 §4.2 律②)。
    /// true ⇒ 主 KV 池多分配 **1 页**作"假行落的点"(不在管理面块表里 ⇒
    /// 注意力结构上读不到);false ⇒ 池几何与历史逐字不变。
    pub prefill_dump_page: bool,
    /// reset_gdn 后 GDN rec 格画像(诊断面;原 OWL_GDN_DBG)
    pub gdn_dbg: bool,
}

pub(crate) struct StatePool {
    /// GDN 状态格容量(knobs.gdn_slots,boot 定谳;原自由函数 gdn_slots())
    pub(crate) gdn_slots: usize,
    /// GDN 画像 dump 全槽(knobs.gdn_dump_all)
    #[allow(dead_code)] // 诊断面字段(写入后当前无读者)
    pub(crate) dump_all: bool,
    /// reset_gdn 后 rec 格画像([probes] gdn_dbg;诊断面)
    pub(crate) gdn_dbg: bool,
    pub(crate) kvs: Vec<KvBlocks>,
    /// FI 直读旗标(统一契约 P3:影子池删除,FI 直读主池 kNHD 叶子)
    pub(crate) fi_face: bool,
    /// B6.2:主 KV 池 fp8 e4m3 承载(true = 池块 U32 字节承载 1B/elem;
    /// 读核走 *_fp8 变体,写核 K0 转换写)
    pub(crate) kv_fp8: bool,
    pub(crate) dflash_fp8: bool,
    gdns: Vec<GdnBlocks>,
    snaps: Vec<GdnSnapSlot>,
    snap_tick: u64,
    bt: Bytes,
    /// v2 分页 decode scratch(E-decode 2026-10-04;paged 时恒分配,
    /// None = legacy/无页策略;引擎注入 ctx → 层走 v2 + LSE 归并)
    pub(crate) attn_v2: Option<owl_models::module::AttnV2Scratch>,
    /// verify v2 scratch(seqs=depth1;§三十一)
    pub(crate) verify_attn_v2: Option<owl_models::module::AttnV2Scratch>,
    /// XQA verify 面(2026-10-13 D3):常量页表/掩码 + semaphores + scratch
    /// (seq_lens 逐轮变 → 由图输入槽供给,不在本结构)
    /// XQA 资源面(verify 与裸 decode **共用**;见 `plan.verify_xqa` /
    /// `plan.decode_xqa` 门 —— 任一开就分配)
    pub(crate) xqa: Option<owl_models::module::XqaVerifyCtx>,
    /// spec 快照(E5-M2:投机轮回滚缓冲;OWL_SPEC_DEPTH>0 时分配,
    /// 内容 = 轮首 GDN 态 state@B-1;restore 后仍有效,全接受后失效重拍)
    pub(crate) spec_snap: Option<GdnSnapSlot>,
    /// MTP 草稿链 KV 池(E5-M2b;独立页池,1 层;mtp 模式分配)
    pub(crate) mtp_kvs: Option<KvBlocks>,
    /// DFlash2 草稿 KV 池(E5-DF2;独立页池,5 层草稿几何 8/128)
    pub(crate) dflash_kvs: Option<Vec<KvBlocks>>,
    /// MTP 链持久块表槽(E5-M5 propose 图烘焙;[1, nb] f32,每轮重写)
    pub(crate) bt_mtp: Option<Bytes>,
    /// **GDN 序列表持久槽**(P-6b,2026-10-15;设计 §4.3e):`gdn_chunked` 六核的
    /// 三张 FLA 表(`cu_seqlens|chunk_offsets|chunk_indices`)按**本 chunk 真行数**
    /// 每 chunk 原位写;runtime 只读。
    ///
    /// 为什么必须持久且必须由引擎写:①核的行区间与**递推落点**全由这三张表定
    /// —— 表按布局 t 生成会把 padding 假行算进序列(递推态污染,探针
    /// `gdn_bucket_pad_probe` 实测终态 Δ=3.6e-2 → 0)②图(M2b)烘的是**指针**
    /// ⇒ 地址必须恒定。None = 本档无 GDN 层。
    pub(crate) gdn_tables: Option<GdnTableBlocks>,
    /// **FlashInfer 表件持久槽**(P-3,2026-10-15;设计 §4.1c P-3):
    /// `q_cu/indices/indptr/last_len` 四件在 boot 一次分配、每 chunk **原位写**
    /// (`write_blocks_f32`,单次批量命令 + 回执即落地)。
    /// 为什么必须持久:①图(M2b)会把它们的**指针**烘进图 ⇒ 地址必须恒定;
    /// ②eager 路旧实现每 chunk `eval_ops(from_host)` × 4 = **4 次往返同步**,
    /// 现 1 次批量写 ⇒ 少 3 次往返。None = 非 FI 档(`pool.fi_face = false`)。
    pub(crate) fi_tables: Option<FiTableBlocks>,
    /// paged 池几何(页;legacy 模式 page = 容量,nb = 1)
    pub(crate) page: usize,
    /// **管理面**页数(store 块腿/会话块表/FI 页链槽只认这些)
    pub(crate) nb: usize,
    /// **物理**页数(含 padding dump 页;= `nb + dump`)
    pub(crate) nb_kv: usize,
    /// padding dump 页索引(假行 KV 槽落点;None = 本档无桶)
    ///
    /// 为什么假行不能落在真页:
    /// 假行的 `slot` 若指向后面 chunk 还没写的真页 → ①KV 被脏值占位
    /// (下个 chunk 真写会覆盖,但本层注意力若先读到就是错值)②末块尾部的位置
    /// 可能在**块链之外**(越界槽号)。dump 页两种都堵死:恒不在链里,也恒可写。
    pub(crate) dump_page: Option<usize>,
    pub(crate) paged: bool,
    x: usize,
    /// 池几何输入(spec 提取;prefill 叶子构造/执行器 dtype 分支用)
    pub(crate) dims: ModelDims,
}

/// GDN 序列表持久槽(P-6b;见 [`StatePool::gdn_tables`])
pub(crate) struct GdnTableBlocks {
    /// `cu|coff|idx` 三件同块(布局 = `owl_models::layers::gdn::gdn_table_*`,单源在 kernels::family)
    pub table: Bytes,
    /// 块数上限(`seq_tokens.div_ceil(BT)`;超界即结构性拒)
    pub nt_cap: usize,
}

/// FlashInfer 表件持久槽(P-3;见 [`StatePool::fi_tables`])。
/// `Clone` = 4 个块句柄的复制(零拷贝;调用方拿去持有时用)
#[derive(Clone)]
pub(crate) struct FiTableBlocks {
    /// `[0, T]`(u32 位型,f32 账)
    pub q_cu: Bytes,
    /// 本会话页链(容量 = 池页数;每 chunk 写前 `nb_used` 项)
    pub indices: Bytes,
    /// `[0, nb_used]`
    pub indptr: Bytes,
    /// 末页有效 token 数
    pub last_len: Bytes,
    /// 容量(池页数,越界即结构性拒 —— 写前由调用方比对)
    pub nb_cap: usize,
}

impl StatePool {
    /// **单逻辑块(页)的设备字节**(账/搬运长度真源,坑册 §42):Σ 全注意力层
    /// (k+v) 每页实占字节。fp8 池以 U32 承载(1B/elem)⇒ 元素数 ÷4、元素宽 ×4
    /// ⇒ 每页字节 = hkv×hd×page ×1;f16 ⇒ ×2。dump 页不算(**÷ nb_kv**)。
    pub(crate) fn block_bytes(&self) -> usize {
        let total: usize = self
            .kvs
            .iter()
            .map(|kv| {
                kv.k_cache.1 * elem_bytes(kv.k_cache.0.dtype)
                    + kv.v_cache.1 * elem_bytes(kv.v_cache.0.dtype)
            })
            .sum();
        total / self.nb_kv.max(1)
    }

    /// **主池布局**(engine 注册给 store 的 id 级描述;设计卷 §10.3 推论 A)。
    /// 每层 k/v 各一项;`stride_bytes` = 每页字节(槽步长)。
    pub(crate) fn pool_layout(&self, epoch: u64) -> owl_store::PoolLayout {
        let mut pieces = Vec::with_capacity(self.kvs.len() * 2);
        for kv in &self.kvs {
            pieces.push(owl_store::PieceSpec {
                buf: kv.k_cache.0.clone(),
                stride_bytes: self.per_page_bytes(kv.k_cache.0.dtype, kv.k_cache.1),
            });
            pieces.push(owl_store::PieceSpec {
                buf: kv.v_cache.0.clone(),
                stride_bytes: self.per_page_bytes(kv.v_cache.0.dtype, kv.v_cache.1),
            });
        }
        owl_store::PoolLayout { class: owl_store::BlockClass::KvPage, epoch, pieces }
    }

    /// **MTP 草稿链池布局**(独立页池;1 层 × k/v)。未开 = None。
    pub(crate) fn mtp_pool_layout(&self, epoch: u64) -> Option<owl_store::PoolLayout> {
        let kv = self.mtp_kvs.as_ref()?;
        Some(owl_store::PoolLayout {
            class: owl_store::BlockClass::KvPage,
            epoch,
            pieces: vec![
                owl_store::PieceSpec {
                    buf: kv.k_cache.0.clone(),
                    stride_bytes: self.per_page_bytes(kv.k_cache.0.dtype, kv.k_cache.1),
                },
                owl_store::PieceSpec {
                    buf: kv.v_cache.0.clone(),
                    stride_bytes: self.per_page_bytes(kv.v_cache.0.dtype, kv.v_cache.1),
                },
            ],
        })
    }

    /// MTP 池的每页字节(= 那条腿的 `block_bytes`;未开 = 0)。
    pub(crate) fn mtp_block_bytes(&self) -> usize {
        self.mtp_kvs
            .as_ref()
            .map(|kv| {
                self.per_page_bytes(kv.k_cache.0.dtype, kv.k_cache.1)
                    + self.per_page_bytes(kv.v_cache.0.dtype, kv.v_cache.1)
            })
            .unwrap_or(0)
    }

    /// 一个缓冲的每页字节(元素数 × 元素宽 ÷ 管理面页数)。
    fn per_page_bytes(&self, dtype: Dtype, elems: usize) -> usize {
        elems * elem_bytes(dtype) / self.nb_kv.max(1)
    }

    /// **GDN 行字节 / 每活跃 turn**(`owl_store::BudgetInputs::gdn_row_bytes` 真源)。
    ///
    /// 口径:每层四件(conv_q/conv_k/conv_v/rec)的分配宽 = `gdn_slots × 单槽宽`
    /// (`StatePool::alloc` 里显式乘过)⇒ **行字节 = 池总字节 ÷ gdn_slots**。
    /// 读真分配,不按公式重算(0.8B 档实测:18 层 × 1.07MiB ≈ 19.3MiB/行)。
    pub(crate) fn gdn_row_bytes(&self) -> usize {
        let pool: usize = self
            .gdns
            .iter()
            .map(|g| {
                (g.conv_q.1 + g.conv_k.1 + g.conv_v.1 + g.rec.1) * elem_bytes(g.rec.0.dtype)
            })
            .sum();
        pool / self.gdn_slots.max(1)
    }

    /// **快照池字节**(E2c 静态预分配;固定面的一项,含 spec 快照缓冲)。
    pub(crate) fn snapshot_pool_bytes(&self) -> usize {
        let pool: usize = self
            .snaps
            .iter()
            .map(|s| s.bufs.iter().map(|b| b.1 * elem_bytes(b.0.dtype)).sum::<usize>())
            .sum();
        let spec = self
            .spec_snap
            .as_ref()
            .map(|s| s.bufs.iter().map(|b| b.1 * elem_bytes(b.0.dtype)).sum::<usize>())
            .unwrap_or(0);
        pool + spec
    }

    /// **草稿池字节**(MTP 链 / DFlash2 五层;关 = 0)。按 `pool_tokens` 全额
    /// 预留 ⇒ 开 draft 的档等于第二个主池(`BudgetInputs::draft_pool_bytes`)。
    pub(crate) fn draft_pool_bytes(&self) -> usize {
        fn kv_bytes(kv: &KvBlocks) -> usize {
            kv.k_cache.1 * elem_bytes(kv.k_cache.0.dtype)
                + kv.v_cache.1 * elem_bytes(kv.v_cache.0.dtype)
        }
        let mtp = self.mtp_kvs.as_ref().map(kv_bytes).unwrap_or(0);
        let df: usize = self.dflash_kvs.as_ref().map(|ks| ks.iter().map(kv_bytes).sum()).unwrap_or(0);
        mtp + df
    }

    /// 分配全部状态块(零初始化;GDN 段每 turn 开始按格重置)。
    /// `pool_tokens` = 块池 token 口径容量(E2b:默认 2 × 单会话容量,
    /// OWL_POOL_TOKENS 可覆写;调用方读 env,本函数只管取整到页)。
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn alloc<D: DeviceClient>(
        face: &mut D,
        dims: ModelDims,
        layer_types: &[bool],
        plan: &PoolPlan,
    ) -> Result<StatePool> {
        let seq_tokens = plan.seq_tokens;
        let pool_tokens = plan.pool_tokens;
        // 布尔和稀泥废除(2026-10-10):量化 = KvQuant 枚举,投机形态 =
        // SpecMode 枚举(快照/MTP 链/草稿池三件由 mode 派生,互斥不再靠
        // 调用点自洽)
        let kv_fp8 = plan.kv == owl_models::env::KvQuant::Fp8E4M3;
        let dflash_fp8 = plan.draft_kv == owl_models::env::KvQuant::Fp8E4M3;
        let spec = plan.spec_mode != crate::spec_runtime::SpecMode::Off;
        let mtp = plan.spec_mode == crate::spec_runtime::SpecMode::Mtp;
        let dflash = plan.spec_mode == crate::spec_runtime::SpecMode::DFlash2;
        let gdn_slots = plan.gdn_slots;
        let snap_max = plan.snap_max;
        let dump_all = plan.dump_all;
        let gdn_dbg = plan.gdn_dbg;
        let t = std::time::Instant::now();
        let n_full = layer_types.iter().filter(|&&f| f).count();
        let n_gdn = layer_types.len() - n_full;
        owl_shared::log_info!("boot", "StatePool::alloc 进入(n_full={n_full} kv_fp8={kv_fp8})");

        // B6.5 显存硬预算治理(2026-10-10):池容量 =
        // min(手工上限, (target×total − live − reserve)/每 token 字节)。
        // target 默认 0.97;reserve 默认 1024MB(盖 CUDA ctx ~430MB + 逐 turn 瞬态 +
        // async 池惰性保留;实测 300MB 在连续 turn 下瞬态 OOM)
        // + 捕获瞬态 + 非追踪惰性保留。OWL_VRAM_TARGET/OWL_VRAM_RESERVE_MB
        // 可调;TOTAL 未落账(=0)或 OWL_VRAM_TARGET=0 → 治理禁用。
        // FIXME: 大量的硬编码
        let per_tok_bytes = (n_full * 2 * dims.hkv * dims.hd) * if kv_fp8 { 1 } else { 2 }
            + if dflash { 5 * 2 * 8 * 128 * 2 } else { 0 }; // 草稿池按 token 线性(页取整后略同)
        // 治理器之后的固定分配(GDN 格/快照 f32 同式;图 slab + ctx + 杂项 flat)
        let gdn_slot_elems = dims.nk * dims.hk * 3 + dims.nv * dims.hv * 3 + dims.nv * dims.hk * dims.hv;
        let fixed_after = n_gdn * gdn_slots * gdn_slot_elems * 4
            + snap_max * n_gdn * gdn_slot_elems * 4
            + (400 << 20); // 图 slab + CUDA ctx + warmup 瞬态 + 杂项
        let vram_reserve = plan.vram_reserve_mb << 20;
        let vram_total = owl_shared::vram::total_bytes();
        let pool_cap = if plan.vram_target > 0.0 && plan.vram_target < 1.0 && vram_total > 0 {
            let budget = (vram_total as f64 * plan.vram_target) as i64
                - vram_reserve as i64
                - fixed_after as i64
                - owl_shared::vram::live_bytes() as i64;
            let cap = (budget.max(0) as u64 / per_tok_bytes.max(1) as u64) as usize;
            owl_shared::log_info!("boot", "vram 治理:total={:.1}G free={:.2}G live={:.1}G target={:.0}% reserve={}MB fixed_after={:.2}G budget={:.2}G per_tok={}B → 池上限 {} tok(manual={})",
                vram_total as f64 / 1073741824.0,
                owl_shared::vram::free_bytes() as f64 / 1073741824.0,
                owl_shared::vram::live_bytes() as f64 / 1073741824.0,
                plan.vram_target * 100.0,
                vram_reserve >> 20,
                fixed_after as f64 / 1073741824.0,
                budget as f64 / 1073741824.0,
                per_tok_bytes,
                cap,
                pool_tokens
            );
            Some(cap)
        } else {
            None
        };
        let pool_tokens = match pool_cap {
            Some(cap) => pool_tokens.min(cap),
            None => pool_tokens,
        };

        let pol = kv_paged_policy(dims.dtype);
        let (page, x, nb, paged) = match &pol {
            Some(p) => (p.page, p.x, seq_tokens.div_ceil(p.page), true),
            None => (seq_tokens, 1usize, 1usize, false),
        };
        // E2b 块池容量:pool_tokens 向上取整到页;不低于单会话容量
        let nb = if paged { pool_tokens.div_ceil(page) } else { nb }.max(nb);
        // prefill 桶 padding dump 页(M2b;设计 §4.2 律②):物理多 1 页,
        // **管理面 nb 不变**(store 块腿/块表/页链槽容量都不受影响)。
        let dump_page = plan.prefill_dump_page.then_some(nb);
        let nb_kv = nb + dump_page.map(|_| 1).unwrap_or(0);

        let mut kvs: Vec<KvBlocks> = Vec::new();
        for l in 0..n_full {
            // B6.2:fp8 池 = U32 字节承载 1B/elem(元素数不变,块大小减半);
            // 读核 *_fp8 变体读入即转 half,写核 K0 转换写
            let (k_elems, k_dt) = if kv_fp8 {
                (nb_kv * dims.hkv * dims.hd * page / 4, Dtype::U32)
            } else {
                (nb_kv * dims.hkv * dims.hd * page, dims.dtype)
            };
            let (v_elems, v_dt) = if kv_fp8 {
                (nb_kv * dims.hkv * dims.hd * page / 4, Dtype::U32)
            } else {
                (nb_kv * dims.hkv * dims.hd * page, dims.dtype)
            };
            let k_cache = zero_block_dt(face, k_elems, k_dt, &format!("kv.k.{l}")).await?;
            let v_cache = zero_block_dt(face, v_elems, v_dt, &format!("kv.v.{l}")).await?;
            kvs.push(KvBlocks { k_cache, v_cache });
        }
        // MTP 草稿链 KV(E5-M2b):独立页池(1 层,几何 = full 层同款,
        // 容量同 pool_tokens);不接前缀缓存 —— mtp KV 依赖 hidden,
        // 不可 token 哈希。仅 paged(链吃块表)。
        let mtp_kvs = if mtp && paged {
            Some(KvBlocks {
                k_cache: zero_block_dt(face, nb * dims.hkv * dims.hd * page, dims.dtype, "mtp.kv.k").await?,
                v_cache: zero_block_dt(face, nb * dims.hkv * dims.hd * page, dims.dtype, "mtp.kv.v").await?,
            })
        } else {
            None
        };
        // DFlash2 草稿池(E5-DF2):5 层 × 草稿几何(8 KV 头 × hd128;
        // z-lab 检查点家族定形),容量同 pool_tokens。仅 paged。
        // dtype = BF16(E5-DF3 同日十四;草稿路径 BF16 全程,独立于目标池
        // dims.dtype —— 同 2B/elem,几何 page/x 不变)
        let dflash_kvs = if dflash && paged {
            let (dkv, dhd, dlayers) = (8usize, 128usize, 5usize);
            // B6 偷显存:草稿池 e4m3(OWL_DFLASH_KV_FP8;1B/elem,U32 字节承载)
            let (df_elems, df_dt) = if dflash_fp8 {
                (nb * dkv * dhd * page / 4, Dtype::U32)
            } else {
                (nb * dkv * dhd * page, Dtype::BF16)
            };
            let mut ks = Vec::with_capacity(dlayers);
            for l in 0..dlayers {
                ks.push(KvBlocks {
                    k_cache: zero_block_dt(face, df_elems, df_dt, &format!("dfk.k.{l}")).await?,
                    v_cache: zero_block_dt(face, df_elems, df_dt, &format!("dfk.v.{l}")).await?,
                });
            }
            Some(ks)
        } else {
            None
        };
        // 影子池已删(统一契约 P3:kNHD 单池,FI 直读主池叶子)
        // 块表持久块(E2b):内容 = 活跃会话块链,turn 切换/增长时重写
        // (write_bt;图烘焙 bt 指针,指针稳定图不重捕);legacy = 哑表
        let bt = if paged {
            eval_ops(TensorOps::zeros(Dtype::F32, vec![1, nb]).step(), face).await?
        } else {
            eval_ops(TensorOps::zeros(Dtype::F32, vec![1]).step(), face).await?
        };
        // GDN 序列表持久槽(P-6b):有 GDN 层就建(尺寸 ~KB 级)
        let gdn_tables = if n_gdn > 0 {
            let nt_cap = seq_tokens.div_ceil(owl_models::kernel::GDN_BT).max(1);
            let bytes = owl_models::kernel::gdn_table_bytes(nt_cap);
            Some(GdnTableBlocks { table: face.alloc(Dtype::F32, bytes.div_ceil(4), "gdn.table").await?, nt_cap })
        } else {
            None
        };
        // FI 表件持久槽(P-3):与 bt 同族(paged + FI 档一次分配,内容每 chunk 重写)
        let fi_face = plan.fi_quant.is_some();
        let fi_tables = if paged && fi_face {
            Some(FiTableBlocks {
                q_cu: face.alloc(Dtype::F32, 2, "fi.q_cu").await?,
                indices: face.alloc(Dtype::F32, nb, "fi.indices").await?,
                indptr: face.alloc(Dtype::F32, 2, "fi.indptr").await?,
                last_len: face.alloc(Dtype::F32, 1, "fi.last_len").await?,
                nb_cap: nb,
            })
        } else {
            None
        };
        // MTP 链持久块表槽(E5-M5;propose 图烘焙;mtp 模式)
        let bt_mtp = if mtp && paged {
            Some(eval_ops(TensorOps::zeros(Dtype::F32, vec![1, nb]).step(), face).await?)
        } else {
            None
        };
        // v2 decode scratch(E-decode):exp_sums/max_logits [1,hq,nparts]
        // f32 + tmp_out [hq·nparts·hd] f16;nparts = ceil(nb·page/512)
        // (PARTITION=512;boot 一次性持久块,不进竞技场收割面,图安全同 bt)
        let attn_v2 = if paged {
            let nparts = (nb * page).div_ceil(512);
            let es = zero_block_dt(face, dims.hq * nparts, Dtype::F32, "attn2.exp_sums").await?;
            let ml = zero_block_dt(face, dims.hq * nparts, Dtype::F32, "attn2.max_logits").await?;
            let to = zero_block_dt(face, dims.hq * nparts * dims.hd, dims.dtype, "attn2.tmp_out").await?;
            Some(owl_models::module::AttnV2Scratch {
                exp_sums: block_leaf(&es.0, vec![1, dims.hq * nparts]),
                max_logits: block_leaf(&ml.0, vec![1, dims.hq * nparts]),
                tmp_out: block_leaf_dt(&to.0, vec![dims.hq * nparts * dims.hd], dims.dtype),
                nparts,
                seqs: 1,
            })
        } else {
            None
        };
        // verify v2 scratch(§三十一:8 伪序列 partition 并行;seqs=depth1;
        // 形状 = decode 版 × depth1,展平 [seq][head][part] 与 vLLM 核同构)
        let verify_attn_v2 = if paged && plan.spec_depth > 0 {
            let nparts = (nb * page).div_ceil(512);
            let seqs = plan.spec_depth + 1;
            let es = zero_block_dt(face, seqs * dims.hq * nparts, Dtype::F32, "vattn2.exp_sums").await?;
            let ml = zero_block_dt(face, seqs * dims.hq * nparts, Dtype::F32, "vattn2.max_logits").await?;
            let to = zero_block_dt(face, seqs * dims.hq * nparts * dims.hd, dims.dtype, "vattn2.tmp_out").await?;
            Some(owl_models::module::AttnV2Scratch {
                exp_sums: block_leaf(&es.0, vec![seqs, dims.hq * nparts]),
                max_logits: block_leaf(&ml.0, vec![seqs, dims.hq * nparts]),
                tmp_out: block_leaf_dt(&to.0, vec![seqs * dims.hq * nparts * dims.hd], dims.dtype),
                nparts,
                seqs,
            })
        } else {
            None
        };
        // XQA 资源面(2026-10-13 D3 + ②;kNHD fp8 池 + grp6 + hd256 + page32)
        // **verify 与裸 decode 共用同一份**:恒定件(页表/掩码/semaphores/
        // scratch)两图都能用,只有 `seq_lens`/`semaphores` 需按图各挂一个
        // 输入槽(verify 图 vs decode 图各自的 `xqa_seq`/`xqa_sem`)。
        // 掩码:q=1 时只读 index 0,而 index 0 = `(1<<1)-1 = 1` —— 恰好是
        // decode 需要的"bit0 允许自身",**两形状共用同一张量**。
        let xqa = if paged
            && (plan.xqa_verify || plan.xqa_decode || plan.xqa_prefill)
            && matches!(plan.kv, owl_models::env::KvQuant::Fp8E4M3)
            && dims.hd == 256
            && page == 32
            && dims.hq / dims.hkv == 6
        {
            let nb_i = nb;
            // 常量页表:单会话恒等链(0..nb)以 **i32 位型走私**进 f32 账叶
            let pt: Vec<f32> = (0..nb_i).map(|i| f32::from_bits(i as u32)).collect();
            let ptb = face.alloc(Dtype::F32, nb_i, "xqa.pt").await?;
            face.write_block_f32(&ptb, 0, &pt).await?;
            // 常量掩码:行 i = 块内第 i 个 token,bit k = 允许看第 k 个 key,
            // 因果 = (1<<(i+1))-1(含自身);q=1 只读 index 0(= 1)
            let seqs = (plan.spec_depth + 1).max(1);
            let mask: Vec<f32> = (0..seqs)
                .map(|i| f32::from_bits((1u32 << (i + 1)) - 1))
                .collect();
            let mkb = face.alloc(Dtype::F32, seqs, "xqa.mask").await?;
            face.write_block_f32(&mkb, 0, &mask).await?;
            // semaphores:u32[round_up(hkv·batch,2)+2+hkv·batch+2](batch=1);
            // **每次发射前必须清零**(KV-split 归约计数)
            let sem_n = dims.hkv.div_ceil(2) * 2 + 2 + dims.hkv + 2;
            let sem = zero_block_dt(face, sem_n, Dtype::F32, "xqa.sem").await?;
            // scratch:持久工作区(参考实现 256MiB)
            // 工作区定档 **32MiB**(2026-10-13 定谳):本形状(batch=1、hkv=4、
            // hd256、q=8、ctx≤16k)的 KV-split 归约 partial 只有 ~0.7MB,32MiB
            // 有 40× 余量;参考实现给 256MiB 是因为它不知道形状。
            // ⚠️ 挂账(未结):≥128MiB 时 **e2e 门那套引擎实例**会报
            // `pump: DeadBlock`(server 路径 256MiB 正常)—— 疑与大额 alloc 撞
            // 引擎的显存治理/捕获 slab(门设 capture_slab_mb=160)。32MiB 下
            // 恒等门绿,故先按需定档;治理交互另案。
            const XQA_SCRATCH_MB: usize = 32;
            let scr_mb = XQA_SCRATCH_MB;
            let scr = zero_block_dt(face, (scr_mb << 20) / 2, Dtype::F16, "xqa.scratch").await?;
            Some(owl_models::module::XqaVerifyCtx {
                // ⚠️ 四件套 = **图输入槽**(每步写 f32 数值;核按 u32 读同一
                // 4 字节 = 位型走私)。故叶子注解与输入槽同 dtype = F32
                // (2026-10-15 事实面审计:声明也必须写 F32,见 ops/xqa.rs 头注)
                page_table: block_leaf(&ptb, vec![nb_i]),
                seq_lens: block_leaf(&ptb, vec![1]), // 占位:图装配时换成输入槽
                mask: block_leaf(&mkb, vec![seqs]),
                // 契约 `XQA_PARAMS` 的 semaphores 是 **r2**(与图路输入槽
                // `xqa_sem` = InputSlot::f32 → [1, n] 同口径);此处旧声明 r1 是
                // 遗留 —— eager 路会因此被准入门拒(2026-10-15 B6 用例暴露)。
                semaphores: block_leaf(&sem.0, vec![1, sem_n]),
                scratch: block_leaf_dt(&scr.0, vec![(scr_mb << 20) / 2], Dtype::F16),
                q_seq_len: seqs,
                max_seq_len: nb_i * page,
                sm_count: XQA_SM_COUNT,
            })
        } else {
            None
        };
        let mut gdns: Vec<GdnBlocks> = Vec::new();
        for _ in 0..n_gdn {
            let conv_q = zero_block(face, gdn_slots * dims.nk * dims.hk * 3, "gdn.conv_q").await?;
            let conv_k = zero_block(face, gdn_slots * dims.nv * dims.hv * 3, "gdn.conv_k").await?;
            let conv_v = zero_block(face, gdn_slots * dims.nv * dims.hv * 3, "gdn.conv_v").await?;
            let rec = zero_block(face, gdn_slots * dims.nv * dims.hk * dims.hv, "gdn.recurrence").await?;
            gdns.push(GdnBlocks { conv_q, conv_k, conv_v, rec });
        }
        // GDN 快照池(E2c):行宽副本 × SNAP_MAX 份;仅 paged 形态
        let mut snaps: Vec<GdnSnapSlot> = Vec::new();
        if paged {
            for _ in 0..snap_max {
                let mut bufs: Vec<BlockN> = Vec::new();
                for _ in 0..n_gdn {
                    bufs.push(zero_block(face, dims.nk * dims.hk * 3, "gdn.slot.conv_q").await?);
                    bufs.push(zero_block(face, dims.nv * dims.hv * 3, "gdn.slot.conv_k").await?);
                    bufs.push(zero_block(face, dims.nv * dims.hv * 3, "gdn.slot.conv_v").await?);
                    bufs.push(zero_block(face, dims.nv * dims.hk * dims.hv, "gdn.slot.rec").await?);
                }
                snaps.push(GdnSnapSlot { key: None, last_use: 0, bufs });
            }
        }
        // 池 dtype 标签随实报(原硬写 "kv f16" 在 fp8 档说谎 —— 2026-10-13)
        let kv_label = match plan.kv {
            owl_models::env::KvQuant::Fp8E4M3 => "kv fp8 e4m3",
            owl_models::env::KvQuant::None => "kv f16",
        };
        owl_shared::log_info!("boot", "状态块分配 {:.2}s({kv_label} ×{} / gdn ×{},会话容量 {},池 {nb} 页 × {page} tok,页表 nb = {nb},dflash = {df})",
            t.elapsed().as_secs_f32(),
            kvs.len(),
            gdns.len(),
            seq_tokens,
            nb = nb,
            page = page,
            df = dflash_kvs.is_some()
        );
        // spec 快照(E5-M2):专用缓冲(独立于 E2c 前缀快照池,免 LRU 逐出)
        let spec_snap = if spec && paged {
            let mut bufs = Vec::new();
            for _ in 0..n_gdn {
                bufs.push(zero_block(face, dims.nk * dims.hk * 3, "gdn.ring.conv_q").await?);
                bufs.push(zero_block(face, dims.nv * dims.hv * 3, "gdn.ring.conv_k").await?);
                bufs.push(zero_block(face, dims.nv * dims.hv * 3, "gdn.ring.conv_v").await?);
                bufs.push(zero_block(face, dims.nv * dims.hk * dims.hv, "gdn.ring.rec").await?);
            }
            Some(GdnSnapSlot { key: None, last_use: 0, bufs })
        } else {
            None
        };
        Ok(StatePool { gdn_slots, dump_all, gdn_dbg, kvs, fi_face, fi_tables, gdn_tables, verify_attn_v2, xqa,
            kv_fp8, dflash_fp8, gdns, snaps, snap_tick: 0, bt, bt_mtp, spec_snap, mtp_kvs, dflash_kvs, attn_v2, page, nb, nb_kv, dump_page, paged, x, dims })
    }

    /// MTP 链块表叶子(propose 图烘焙;None = 未启用)
    pub(crate) fn bt_mtp_leaf(&self) -> Option<TensorOps> {
        self.bt_mtp
            .as_ref()
            .map(|b| TensorOps::of_block(b.id, Dtype::F32, vec![1, self.nb]))
    }

    /// MTP 链块表槽重写(每轮 propose 前;内容 = 会话 mtp 链)
    pub(crate) async fn write_bt_mtp<D: DeviceClient>(
        &self,
        face: &mut D,
        table: &[u32],
    ) -> Result<()> {
        let b = self.bt_mtp.as_ref().expect("bt_mtp 未分配(mtp 模式)");
        let mut v = vec![0f32; self.nb];
        for (i, x) in table.iter().enumerate() {
            v[i] = *x as f32;
        }
        face.write_block_f32(b, 0, &v).await
    }

    /// spec 快照缓冲句柄组(E5-M5 fold 图烘焙:状态 args = 快照 buf
    /// 就地,slot 恒 0;fold 后 restore 回写会话格)。序 = 层 × 4
    /// (conv_q/k/v/rec),与 capture/restore 同构。
    pub(crate) fn spec_snap_bufs(&self) -> Vec<BlockN> {
        self.spec_snap
            .as_ref()
            .map(|s| s.bufs.clone())
            .unwrap_or_default()
    }

    /// GDN 层数(fold 记录槽计数用)
    pub(crate) fn gdn_count(&self) -> usize {
        self.gdns.len()
    }

    // DFlash2 草稿 KV 叶子组(dflash 模式;5 层 × (k, v);None = 未启用)
    // ── DFlash2 草稿池寻址权威(唯一出口;slot≥1 案配套,2026-10-10)──
    // 草稿池 = **线性逻辑寻址**:NC 核签名无块表参数,前缀读 [0, kv_len)
    // 从池基址直排;encode 写槽 = 逻辑位。写者(exec encode)与读者
    // (propose 图/NC)禁止各自推槽 —— 两套语言曾分叉(bt 物理写 vs 线性
    // 读),首 session 巧合重合,后续 session 全崩(slot≥1 案)。
    // 并发多 turn(B5 batch)共享线性空间会互踩 → B5 需 per-session 分区。

    /// encode 行 i(基址 base)的草稿 K/V 槽(K0 写入位)
    pub(crate) fn draft_encode_slot(base: usize, i: usize) -> f32 {
        (base + i) as f32
    }

    /// propose 噪声块自槽(块内行 i;K/V 核输入直读,不入池)
    pub(crate) fn draft_self_slot(pos: usize, i: usize) -> f32 {
        (pos + i) as f32
    }

    /// propose 前缀读窗:逻辑位 p(NC 线性直排)
    pub(crate) fn draft_prefix_slot(p: usize) -> f32 {
        p as f32
    }

    /// propose 读窗长(fp = 下轮 fed 位)
    #[allow(dead_code)] // 草稿面预测助手(现无读者,保留供 NC v2 路径)
    pub(crate) fn draft_kv_len(fp: usize) -> f32 {
        (fp + 8) as f32
    }

    pub(crate) fn dflash_kv_leaves(&self) -> Option<Vec<(TensorOps, TensorOps)>> {
        let ks = self.dflash_kvs.as_ref()?;
        let (dkv, dhd) = (8usize, 128usize);
        // B6:fp8 草稿池 = U32 扁平字节账(主池同款)
        if self.dflash_fp8 {
            let shape = vec![self.nb * dkv * dhd * self.page / 4];
            return Some(ks
                .iter()
                .map(|kb| {
                    (
                        block_leaf_dt(&kb.k_cache.0, shape.clone(), Dtype::U32),
                        block_leaf_dt(&kb.v_cache.0, shape.clone(), Dtype::U32),
                    )
                })
                .collect());
        }
        Some(ks
            .iter()
            .map(|kb| {
                (
                    // 草稿池是 **BF16 分配**(见 StatePool::alloc 的 dflash_kvs 段:
                    // "草稿路径 BF16 全程,独立于目标池 dims.dtype")⇒ 叶子注解也必须
                    // 是 BF16。旧实现传 `self.dims.dtype`(目标池 F16)= 注解撒谎,
                    // 被 L3 事实面当场抳住(2026-10-15)。
                    block_leaf_dt(
                        &kb.k_cache.0,
                        vec![self.nb, dkv, dhd / self.x, self.page, self.x],
                        Dtype::BF16,
                    ),
                    block_leaf_dt(
                        &kb.v_cache.0,
                        vec![self.nb, dkv, dhd, self.page],
                        Dtype::BF16,
                    ),
                )
            })
            .collect())
    }

    /// MTP 草稿链 KV 叶子对(mtp 模式;None = 未启用)
    pub(crate) fn mtp_kv_leaf(&self) -> Option<(TensorOps, TensorOps)> {
        let mk = self.mtp_kvs.as_ref()?;
        Some((
            block_leaf_dt(&mk.k_cache.0, self.k_shape(), self.dims.dtype),
            block_leaf_dt(&mk.v_cache.0, self.v_shape(), self.dims.dtype),
        ))
    }

    /// v2 scratch 叶子(None = legacy;引擎按步注入 ctx.attn_v2)
    pub(crate) fn attn_v2_scratch(&self) -> Option<owl_models::module::AttnV2Scratch> {
        self.attn_v2.clone()
    }

    /// verify v2 scratch(seqs=depth1;§三十一)
    pub(crate) fn verify_attn_v2_scratch(&self) -> Option<owl_models::module::AttnV2Scratch> {
        self.verify_attn_v2.clone()
    }

    /// XQA 资源面(常量件;`seq_lens`/`semaphores` 由图输入槽供给,调用方覆盖)
    pub(crate) fn xqa_ctx(&self) -> Option<owl_models::module::XqaVerifyCtx> {
        self.xqa.clone()
    }

    /// XQA semaphores 元素数(图输入槽写零用;0 = 本档无 XQA 面)
    pub(crate) fn xqa_sem_n(&self) -> usize {
        self.xqa
            .as_ref()
            .map(|x| x.semaphores.shape().iter().product::<usize>())
            .unwrap_or(0)
    }

    /// XQA semaphores 块 + 字节数(每轮发射前清零用)
    #[allow(dead_code)] // XQA sem 清零助手(半接入,保留)
    pub(crate) fn xqa_sem_zero(&self) -> Option<(owl_iface::contract::Bytes, usize)> {
        let x = self.xqa.as_ref()?;
        let n = x.semaphores.shape().iter().product::<usize>();
        // 事实面随叶子(位型走私:写 f32 数值、核按 u32 读)
        Some((
            owl_iface::contract::Bytes::new(
                x.semaphores.id(),
                n,
                x.semaphores.dtype(),
                x.semaphores.shape().len() as u8,
            ),
            n * 4,
        ))
    }

    /// KV 池形状(模式相关;与层分派同源策略)
    pub(crate) fn k_shape(&self) -> Vec<usize> {
        let d = &self.dims;
        if self.paged {
            vec![self.nb, d.hkv, d.hd / self.x, self.page, self.x]
        } else {
            vec![self.nb, d.hkv, d.hd] // legacy:nb=1,page=容量
        }
    }
    pub(crate) fn v_shape(&self) -> Vec<usize> {
        let d = &self.dims;
        if self.paged {
            vec![self.nb, d.hkv, d.hd, self.page]
        } else {
            vec![self.nb, d.hkv, d.hd]
        }
    }

    /// 主 KV 池 K 叶子形状。**含 padding dump 页**(块尺寸 = 形状元素数;
    /// 盘外页只作 bucket padding 假行的槽落点,永不进页链 ⇒ 绝不被读)。
    /// ⚠️ MTP/草稿池用 [`StatePool::k_shape`](本函数不含 dump 页:那两个池单独分配,没多页)。
    pub(crate) fn k_shape_kv(&self) -> Vec<usize> {
        let d = &self.dims;
        if self.paged {
            vec![self.nb_kv, d.hkv, d.hd / self.x, self.page, self.x]
        } else {
            vec![self.nb_kv, d.hkv, d.hd]
        }
    }

    pub(crate) fn v_shape_kv(&self) -> Vec<usize> {
        let d = &self.dims;
        if self.paged {
            vec![self.nb_kv, d.hkv, d.hd, self.page]
        } else {
            vec![self.nb_kv, d.hkv, d.hd]
        }
    }

    /// padding dump 页的 KV 平面槽基址(槽号 = 页号 × page;None = 无桶档)
    pub(crate) fn dump_slot_base(&self) -> Option<usize> {
        self.dump_page.map(|p| p * self.page)
    }

    /// 逐层 KV cache 叶子句柄((k, v);slots/kv_lens/block_tables 由
    /// 调用方按步注入 —— 图闭包注输入槽,prefill 注 from_host 标量行)
    pub(crate) fn kv_caches(&self) -> Vec<(TensorOps, TensorOps)> {
        // B6.2:fp8 池 = U32 扁平字节账叶(指针面;读核按元素序自寻址,
        // FI 影子池 fp8 档同款)
        if self.kv_fp8 {
            let shape = vec![self.nb_kv * self.dims.hkv * self.dims.hd * self.page / 4];
            return self
                .kvs
                .iter()
                .map(|b| {
                    (
                        block_leaf_dt(&b.k_cache.0, shape.clone(), Dtype::U32),
                        block_leaf_dt(&b.v_cache.0, shape.clone(), Dtype::U32),
                    )
                })
                .collect();
        }
        let k_shape = self.k_shape_kv();
        let v_shape = self.v_shape_kv();
        self.kvs
            .iter()
            .map(|b| {
                (
                    block_leaf_dt(&b.k_cache.0, k_shape.clone(), self.dims.dtype),
                    block_leaf_dt(&b.v_cache.0, v_shape.clone(), self.dims.dtype),
                )
            })
            .collect()
    }

    /// GDN 序列表持久槽(P-6b;None = 无 GDN 层)
    pub(crate) fn gdn_table_blocks(&self) -> Option<&GdnTableBlocks> {
        self.gdn_tables.as_ref()
    }

    /// FI 表件持久槽(P-3;None = 非 FI 档)
    pub(crate) fn fi_table_blocks(&self) -> Option<&FiTableBlocks> {
        self.fi_tables.as_ref()
    }

    /// FlashInfer K/V 叶子(统一契约 P3:影子池删除,直读主池 kNHD;
    /// 签名保持 = FI 通道上层零改动;fp8 档 = U32 扁平字节账叶,FI 只吃指针)
    pub(crate) fn kv_fi_leaves(&self) -> Vec<(TensorOps, TensorOps)> {
        self.kv_caches()

    }

    /// D1 取证:会话格 GDN 状态画像(整块回读 → 活跃格行 maxabs/sum/首值。
    /// OWL_GDN_DUMP_ALL=1 → 全层 cv 扫描(定位首个分歧层);否则首末层
    /// cv+rec)。调试仪表,正常路径零调用。27B rec 整块 ~25MB/层,dtoh
    /// 有感 —— 仅取证轮启用。
    pub(crate) async fn gdn_slot_profile<D: DeviceClient>(
        &self,
        face: &mut D,
        gdn_slot: usize,
        step: u64,
        all: bool,
    ) -> Result<Vec<String>> {
        let nl = self.gdns.len();
        let mut out = Vec::new();
        for (li, g) in self.gdns.iter().enumerate() {
            let last = li + 1 == nl;
            // 全层扫 = 仅 cv;默认 = 首末层 cv+rec
            if all && !last {
                let line = self.profile_block(face, li, "cv", &g.conv_v, gdn_slot, step, false).await?;
                out.push(line);
                continue;
            }
            if !all && li != 0 && !last {
                continue;
            }
            for (name, bn) in [("cv", &g.conv_v), ("rec", &g.rec)] {
                let line = self.profile_block(face, li, name, bn, gdn_slot, step, true).await?;
                out.push(line);
            }
        }
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    async fn profile_block<D: DeviceClient>(
        &self,
        face: &mut D,
        li: usize,
        name: &str,
        bn: &BlockN,
        gdn_slot: usize,
        step: u64,
        verbose: bool,
    ) -> Result<String> {
        let row = bn.1 / self.gdn_slots;
        let mut buf = vec![0u8; bn.1 * 4];
        face.dtoh(&bn.0, &mut buf).await?;
        let floats: Vec<f32> = buf[gdn_slot * row * 4..(gdn_slot + 1) * row * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let maxabs = floats.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let sum: f32 = floats.iter().take(4096).sum();
        let head: Vec<String> = floats.iter().take(3).map(|v| format!("{v:.4}")).collect();
        Ok(format!(
            "[gdn] s{step} L{li}/{name} n={} max|.|={maxabs:.4e} sum4k={sum:.4}{}",
            floats.len(),
            if verbose { format!(" head={}", head.join(",")) } else { String::new() },
        ))
    }

    /// 逐层 GDN 状态叶子四件套(conv_q/conv_k/conv_v/rec;slots 由
    /// 调用方按步注入:图闭包注输入槽,prefill 注 from_host)
    pub(crate) fn gdn_caches(&self) -> Vec<GdnLeaf> {
        let d = &self.dims;
        self.gdns
            .iter()
            .map(|g| GdnLeaf {
                conv_q: block_leaf(&g.conv_q.0, vec![self.gdn_slots, d.nk * d.hk, 3]),
                conv_k: block_leaf(&g.conv_k.0, vec![self.gdn_slots, d.nv * d.hv, 3]),
                conv_v: block_leaf(&g.conv_v.0, vec![self.gdn_slots, d.nv * d.hv, 3]),
                rec: block_leaf(&g.rec.0, vec![self.gdn_slots, d.nv, d.hk, d.hv]),
            })
            .collect()
    }

    /// 块表叶子(模式相关:paged [1,nb] / legacy [1];run 图闭包用)
    pub(crate) fn bt_leaf(&self) -> TensorOps {
        if self.paged {
            TensorOps::of_block(self.bt.id, Dtype::F32, vec![1, self.nb])
        } else {
            TensorOps::of_block(self.bt.id, Dtype::F32, vec![1])
        }
    }

    /// 块表叶子(恒 [1,nb];prefill_chunk 既有形状,legacy nb=1 → [1,1])
    pub(crate) fn bt_leaf_flat(&self) -> TensorOps {
        TensorOps::of_block(self.bt.id, Dtype::F32, vec![1, self.nb])
    }

    /// 快照存在性(调度面复用边界判定:无快照的边界不可复用,GDN 态
    /// 对不上)
    /// 快照池容量(`snap_max`;0 = 快照面整体关闭 —— 调用方须跳过 capture,
    /// 否则 `capture_snap` 会因池空而报错,见 agent 专项 A2)
    pub(crate) fn snap_capacity(&self) -> usize {
        self.snaps.len()
    }

    /// 快照池已用槽数(诊断/报告面:在场多少张边界快照)。
    pub(crate) fn snap_used(&self) -> usize {
        self.snaps.iter().filter(|s| s.key.is_some()).count()
    }

    pub(crate) fn has_snap(&self, key: SnapKey) -> bool {
        self.snaps.iter().any(|s| s.key == Some(key))
    }

    /// GDN 状态重置(turn-open;新序列零态语义)。按格重置(offset =
    /// 格号 × 行字节)—— 多会话各清各格,其余格的其他会话状态不受扰。
    /// 设备侧 memset(2026-09-26 定谳:host 往返清零 1.2GB = 14s/turn,
    /// 黑洞实测)。KV 不重置:kv_len 窗口 + 先写后打分语义下,本 turn
    /// 打分的槽位全部由本 turn 写过。
    pub(crate) async fn reset_gdn<D: DeviceClient>(
        &self,
        face: &mut D,
        gdn_slot: usize,
    ) -> Result<()> {
        for g in &self.gdns {
            let blocks = [
                (&g.conv_q.0, g.conv_q.1),
                (&g.conv_k.0, g.conv_k.1),
                (&g.conv_v.0, g.conv_v.1),
                (&g.rec.0, g.rec.1),
            ];
            for (bn, elems) in blocks {
                let row = elems / self.gdn_slots;
                face.memset_zero_at(bn, gdn_slot * row * 4, row * 4).await?;
            }
        }
        face.sync().await?;
        // 临时诊断(§三十二 会话污染定位;[probes] gdn_dbg 启用,收口后删)
        if self.gdn_dbg {
            let g = &self.gdns[0];
            let row_elems = g.rec.1 / self.gdn_slots;
            let mut buf = vec![0u8; g.rec.1 * 4];
            face.dtoh(&g.rec.0, &mut buf).await?;
            let seg = &buf[gdn_slot * row_elems * 4..(gdn_slot + 1) * row_elems * 4];
            let nz = seg.iter().filter(|&&b| b != 0).count();
            owl_shared::log_debug!("gdn-dbg", "reset_gdn slot={gdn_slot} rec非零字节={nz}/{}", seg.len());
        }
        Ok(())
    }

    /// GDN 快照拍摄(E2c;块边界跨越时调用):会话格状态 → 槽位缓冲
    /// (D2D;每层 4 块按行宽拷贝)。同 key 覆盖写;池满 LRU 换 key。
    /// (key 解析 = 边界覆盖块,由调用方从会话块链取;见 exec.rs)
    /// 泄漏案探针(2026-10-10):指定 slot 的 GDN 状态区逐块 FNV。
    /// turn(slot0) 前后对比 slot1 指纹 —— 变了 = 跨 slot 写实锤。
    #[allow(dead_code)] // 泄漏案探针,仅单测消费
    pub(crate) async fn gdn_slot_fingerprint<D: DeviceClient>(
        &self,
        face: &mut D,
        slot: usize,
    ) -> Result<Vec<u64>> {
        let mut out = Vec::new();
        for g in &self.gdns {
            for (bn, elems) in [
                (&g.conv_q.0, g.conv_q.1),
                (&g.conv_k.0, g.conv_k.1),
                (&g.conv_v.0, g.conv_v.1),
                (&g.rec.0, g.rec.1),
            ] {
                let row = elems / self.gdn_slots;
                let mut buf = vec![0u8; elems * 4]; // 整块回读(f32)
                face.dtoh(bn, &mut buf).await?;
                let mut h: u64 = 0xcbf2_9ce4_8422_2325;
                for &b in &buf[slot * row * 4..(slot + 1) * row * 4] {
                    h ^= b as u64;
                    h = h.wrapping_mul(0x100_0000_01b3);
                }
                out.push(h);
            }
        }
        Ok(out)
    }

    /// 泄漏案探针第二步:reset_gdn(slot) 后回读该 slot 是否全零。
    /// 返回 slot 区非零字节数(0 = reset 生效)。
    #[allow(dead_code)] // 泄漏案探针,仅单测消费
    pub(crate) async fn gdn_slot_zero_check<D: DeviceClient>(
        &mut self,
        face: &mut D,
        slot: usize,
    ) -> Result<usize> {
        self.reset_gdn(face, slot).await?;
        let mut nonzero = 0usize;
        for g in &self.gdns {
            for (bn, elems) in [
                (&g.conv_q.0, g.conv_q.1),
                (&g.conv_k.0, g.conv_k.1),
                (&g.conv_v.0, g.conv_v.1),
                (&g.rec.0, g.rec.1),
            ] {
                let row = elems / self.gdn_slots;
                let mut buf = vec![0u8; elems * 4]; // 整块回读(f32)
                face.dtoh(bn, &mut buf).await?;
                nonzero += buf[slot * row * 4..(slot + 1) * row * 4]
                    .iter()
                    .filter(|&&b| b != 0)
                    .count();
            }
        }
        Ok(nonzero)
    }

    pub(crate) async fn capture_snap<D: DeviceClient>(
        &mut self,
        face: &mut D,
        gdn_slot: usize,
        key: SnapKey,
    ) -> Result<()> {
        // 池空(snap_max = 0)不再是 panic(原 `expect("池非空")` 会炸掉整个
        // owl-engine 线程 → 客户端收 503):退化为**结构化错误** —— 调用方
        // (`capture_gdn_snap`)已按 snap_capacity() 先行跳过,故此处只作兜底
        // (agent 专项 A2,2026-10-14)。
        let idx = match self.snaps.iter().position(|s| s.key == Some(key)) {
            Some(i) => i,
            None => {
                let free = self.snaps.iter().position(|s| s.key.is_none());
                let lru = self
                    .snaps
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, s)| s.last_use)
                    .map(|(i, _)| i);
                match free.or(lru) {
                    Some(i) => i,
                    None => {
                        return Err(ModelError::Msg(
                            "GDN 快照池为空(snap_max = 0):调用方应跳过 capture \u{2014}\u{2014} \
                             见 roadmap.local/agent-服务面-剩余工作项 A2"
                                .into(),
                        ))
                    }
                }
            }
        };
        self.snap_tick += 1;
        for (li, g) in self.gdns.iter().enumerate() {
            let quads = [
                (&g.conv_q.0, g.conv_q.1),
                (&g.conv_k.0, g.conv_k.1),
                (&g.conv_v.0, g.conv_v.1),
                (&g.rec.0, g.rec.1),
            ];
            for (j, (src, elems)) in quads.iter().enumerate() {
                let row = elems / self.gdn_slots;
                let dst = &self.snaps[idx].bufs[li * 4 + j];
                face.copy_block_at(src, gdn_slot * row * 4, &dst.0, 0, row * 4).await?;
            }
        }
        let slot = &mut self.snaps[idx];
        slot.key = Some(key);
        slot.last_use = self.snap_tick;
        Ok(())
    }

    /// GDN 快照恢复(E2c;前缀命中时调用):槽位缓冲 → 会话格(D2D
    /// 反向)。同流保序,后续 prefill 天然在恢复态之后。
    /// `key` = 边界块内容身份(与拍摄侧同类型 ⇒ 槽号进不来)。
    pub(crate) async fn restore_snap<D: DeviceClient>(
        &mut self,
        face: &mut D,
        gdn_slot: usize,
        key: SnapKey,
    ) -> Result<()> {
        let idx = self
            .snaps
            .iter()
            .position(|s| s.key == Some(key))
            .ok_or_else(|| ModelError::Msg(format!("快照 {key:?} 已被逐出")))?;
        self.snap_tick += 1;
        for (li, g) in self.gdns.iter().enumerate() {
            let quads = [
                (&g.conv_q.0, g.conv_q.1),
                (&g.conv_k.0, g.conv_k.1),
                (&g.conv_v.0, g.conv_v.1),
                (&g.rec.0, g.rec.1),
            ];
            for (j, (dst, elems)) in quads.iter().enumerate() {
                let row = elems / self.gdn_slots;
                let src = &self.snaps[idx].bufs[li * 4 + j];
                face.copy_block_at(&src.0, 0, dst, gdn_slot * row * 4, row * 4).await?;
            }
        }
        Ok(())
    }

    /// spec 快照拍摄(E5-M2;轮首/全接受后):会话格 → 专用缓冲。
    /// 与 E2c 前缀快照分池(spec 缓冲不参与 LRU,恢复后仍有效)。
    pub(crate) async fn capture_spec_snap<D: DeviceClient>(
        &mut self,
        face: &mut D,
        gdn_slot: usize,
    ) -> Result<()> {
        let n = self.gdns.len();
        let snap = self
            .spec_snap
            .as_mut()
            .ok_or_else(|| ModelError::Msg("spec 快照缓冲未分配(OWL_SPEC_DEPTH 未开)".into()))?;
        // E5-M4:批量通道(192 次 actor 往返 → 1;单 ack 保序)
        let mut copies: Vec<(owl_iface::Bytes, usize, owl_iface::Bytes, usize, usize)> = Vec::with_capacity(n * 4);
        for (li, g) in self.gdns.iter().enumerate() {
            let quads = [
                (&g.conv_q.0, g.conv_q.1),
                (&g.conv_k.0, g.conv_k.1),
                (&g.conv_v.0, g.conv_v.1),
                (&g.rec.0, g.rec.1),
            ];
            for (j, (src, elems)) in quads.iter().enumerate() {
                let row = elems / self.gdn_slots;
                let dst = &snap.bufs[li * 4 + j];
                copies.push(((*src).clone(), gdn_slot * row * 4, dst.0.clone(), 0usize, row * 4));
            }
        }
        face.copy_batch(copies.as_slice()).await?;
        let _ = n;
        Ok(())
    }

    /// spec 快照恢复(E5-M2;部分接受后):专用缓冲 → 会话格(反向 D2D)。
    pub(crate) async fn restore_spec_snap<D: DeviceClient>(
        &mut self,
        face: &mut D,
        gdn_slot: usize,
    ) -> Result<()> {
        let n = self.gdns.len();
        let snap = self
            .spec_snap
            .as_ref()
            .ok_or_else(|| ModelError::Msg("spec 快照缓冲未分配".into()))?;
        // E5-M4:批量通道(同 capture)
        let mut copies: Vec<(owl_iface::Bytes, usize, owl_iface::Bytes, usize, usize)> = Vec::with_capacity(n * 4);
        for (li, g) in self.gdns.iter().enumerate() {
            let quads = [
                (&g.conv_q.0, g.conv_q.1),
                (&g.conv_k.0, g.conv_k.1),
                (&g.conv_v.0, g.conv_v.1),
                (&g.rec.0, g.rec.1),
            ];
            for (j, (dst, elems)) in quads.iter().enumerate() {
                let row = elems / self.gdn_slots;
                let src = &snap.bufs[li * 4 + j];
                copies.push((src.0.clone(), 0usize, (*dst).clone(), gdn_slot * row * 4, row * 4));
            }
        }
        face.copy_batch(copies.as_slice()).await?;
        let _ = n;
        Ok(())
    }

    /// 活跃会话块表 → 持久 bt 块(设备)。图烘焙的是 bt 指针,内容随
    /// 会话切换/块链增长重写(指针稳定 = 图不重捕);f32 过线(契约 5)
    pub(crate) async fn write_bt<D: DeviceClient>(
        &self,
        face: &mut D,
        table: &[u32],
    ) -> Result<()> {
        let mut v = vec![0f32; self.nb];
        for (i, b) in table.iter().enumerate() {
            v[i] = *b as f32;
        }
        face.write_block_f32(&self.bt, 0, &v).await
    }
}

/// XQA 的 `multiProcessorCount`(KV-split 启发式;**只影响切分粒度,不影响
/// 正确性**)。⚠️ FIXME:硬编码 sm86 双卡(SM=84);3080 为 68 —— 挂账:后端
/// 暴露 multi_processor_count 后改查询(DeviceClient 现无此面)。
const XQA_SM_COUNT: usize = 84;

/// 按 dtype 清零分配(KV/GDN 状态块)。**设备侧 alloc + memset**,不走
/// host 零向量 —— 曾用 `vec![0f32; n]` + to_le_bytes 逐元素造 64MB(rec
/// 单层 16M 元素),debug 循环烧掉 ~90s(F5-4 同款 debug 转换税);池
/// alloc 无零保证,必须显式 memset(F5-4 已去 fill(0),不能省)
async fn zero_block_dt<D: DeviceClient>(face: &mut D, n: usize, dt: Dtype, name: &str) -> Result<BlockN> {
    // 2B 激活域显式臂(BF16 曾落 else 按 4B 误解释 —— dtype 表驱动律)
    let esz = if matches!(dt, Dtype::F16 | Dtype::BF16) { 2 } else { 4 };
    let b = face.alloc(dt, n, name).await?;
    face.memset_zero(&b, n * esz).await?;
    Ok((b, n))
}

fn block_leaf_dt(b: &Bytes, shape: Vec<usize>, dt: Dtype) -> TensorOps {
    // 如实传递(2026-10-15 L3 事实面审计):原实现对非 2B 类型一律折叠成
    // `Dtype::F32`(`_ => of_block(F32)`)—— legacy "f32 基线"残留。后果:
    // fp8 KV 池块以 `Dtype::U32`(bytes/4 口径)分配,叶子却注解 F32,
    // 声明面(XQA/FI 的 `k_cache: U8`)与实际喂的视图**三方各说各话**。
    // 折叠去掉后叶子注解 = 分配口径;池的"e4m3 字节语义"由 `PoolAccess`
    // 声明承担(见 ops/xqa.rs 的 `pools::KV_MAIN` 项)。
    TensorOps::of_block(b.id, dt, shape)
}

async fn zero_block<D: DeviceClient>(face: &mut D, n: usize, name: &str) -> Result<BlockN> {
    zero_block_dt(face, n, Dtype::F32, name).await
}

fn block_leaf(b: &Bytes, shape: Vec<usize>) -> TensorOps {
    TensorOps::of_block(b.id, Dtype::F32, shape)
}

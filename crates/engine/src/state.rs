//! 设备状态块池(StatePool;2026-10-01 engine.rs 拆分立)。
//!
//! 引擎全部设备侧状态块的**治理对象**:KV paged 池(kvs)/ GDN 行
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

use owl_iface::contract::{Bytes, DeviceClient, Dtype, ModelError, VramSnapshot};
use owl_store::SnapKey;
use owl_models::interpreters::eval_ops;
use owl_models::module::kv_paged_policy;
use owl_models::TensorOps;

type Result<T> = std::result::Result<T, ModelError>;

pub(crate) type BlockN = (Bytes, usize); // (块句柄, 元素数;重置分块写用)



/// GDN 行容量(**与会话数绑定,与 max_seq_tokens 解耦**)。
/// 格语义 = 每会话一格(SessionTable 分配/释放);按位分配是历史包袱:
/// s=4096 时 rec(1 MiB/格/层 × 18 层)将达 72 GB,而真实需求 = 格数。
/// 8 = 覆盖需求并发上限(§二 并发 ≤8);会话 close 即释放格号。
/// 槽数 = EngineKnobs.gdn_slots(显式传入;原 OWL_MAX_ACTIVE_TURNS,缺省 8)。
///
/// GDN 快照槽(E2c):key = [`SnapKey`](**边界块的内容身份**;逐块父链哈希,
/// `owl_store::ChainCursor`)。为什么不是物理块号:物理块号是会被回收复用的
/// 槽号 ⇒ 旧键会命中别人的状态(静默错值;store 卷 §9.1)。键类型来自 store
/// ⇒ **物理槽号在类型上就进不了快照键**。
/// bufs = 逐层逐块行宽副本(与 gdns 同序同构,行宽 = elems / MAX_ACTIVE_TURNS)。
/// 池静态预分配,驱逐 = 换 key 覆盖写(零 alloc/free 往返)。
/// **快照池(S2 三层化;§25.2)**:矩阵化 —— 每 (层,quad) **一条**缓冲,
/// 长 = `snap_max × 行宽` ⇒ 一条快照 = 一个"槽"(片 = 逐 (层,quad) 的行宽,
/// 与 KV 腿的 `PoolLayout` 同构 ⇒ 直接套 store 的 swap 机制)。
///
/// **账的真源在 store 的 `StateStore`**(key → 层/字节/新鲜度);本池只做
/// 设备物理与**槽位索引**(槽 → key)。所以 `last_use` 仍在本池(设备槽的
/// 替换序是设备面的事),而"内容还在不在(设备/宿主)"归 store。
/// 槽位选择结果(纯判定;执行在 engine —— store 出账、face 搬运)。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SnapSlotPick {
    /// 同一张快照已在设备(原地覆盖/直接恢复)。
    Hit(u32),
    /// 空槽。
    Free(u32),
    /// 需替换:最冷的设备快照(槽, 它的 key)。
    Victim(u32, SnapKey),
}

/// spec 快照缓冲(E5-M2;独立池,**不参与 LRU** —— 恢复后仍有效)。
/// 序 = 层 × 4(conv_q/k/v/rec),与 capture/restore 同构。
pub(crate) struct SpecSnapBuf {
    pub(crate) bufs: Vec<BlockN>,
}

pub(crate) struct GdnSnapPool {
    /// 逐 (层,quad) 的池矩阵(与 `gdns` 同序同构)。
    bufs: Vec<BlockN>,
    /// 槽 → key(`None` = 空槽)。
    keys: Vec<Option<SnapKey>>,
    /// 槽新鲜度(设备槽替换序)。
    last_use: Vec<u64>,
    /// 每槽每片字节(= 行宽;`attach` 后由布局定谳)。
    row_bytes: Vec<usize>,
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
        // 8 位族(§S 2026-10-17 dtype 分离):fp8×2 / int8 各自点名,同宽 1
        Dtype::U8 | Dtype::F8E4M3 | Dtype::F8E5M2 | Dtype::I8 => 1,
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

/// **每 token KV 字节**(纯函数;S4 ④ 提取 —— 池容量公式的**测试锚点**)。
///
/// 口径(与 `StatePool::alloc` 同源):主池 = 每层每 token `2(K+V) × hkv × hd`
/// × 元素宽(f16=2 / fp8=1);草稿池按 token 线性。**TP 分片自动生效**:
/// `dims.hkv/hd` 来自 sharded spec ⇒ 每 rank 的 KV 体积按头数切分。
#[cfg(test)] // 只被单测消费(池容量公式的锚点);生产路径另有 alloc 内联算式
pub(crate) fn kv_per_tok_bytes(
    n_full: usize,
    dims: &ModelDims,
    kv_fp8: bool,
    dflash: bool,
) -> usize {
    (n_full * 2 * dims.hkv * dims.hd) * if kv_fp8 { 1 } else { 2 }
        + if dflash { 5 * 2 * 8 * 128 * 2 } else { 0 } // 草稿池按 token 线性(页取整后略同)
}

/// 设备状态块池(KV paged 池 / GDN 行 / 快照池 / 持久块表)。
/// 治理对象(charter A1 味):S2 图档与执行器只经方法面触碰状态块,
/// 不直摸块句柄。
/// 池装配计划(engine run 从 knobs/spec_mode 派生;显式依赖律的
/// 参数对象 —— 原散列布尔/数值参数群收编:量化 = KvQuant 枚举,
/// 投机形态 = SpecMode 枚举,互斥语义由派生保证,调用点零自洽负担)
#[derive(Clone, Debug)]
pub(crate) struct PoolPlan {    /// 单会话容量(= max_seq_tokens)
    pub seq_tokens: usize,
    /// 池容量(token;**`pool_auto = true` 时本值被忽略**,按 `seq_tokens` 自动)
    pub pool_tokens: usize,
    /// **池容量自动派生**(`[pool].pool_tokens` 未写;2026-10-21 用户令:
    /// "池子一般不需要手动指定 —— 算出来、把最大 ctx 显示出来")。
    ///
    /// 自动档语义:池 = 单会话 ctx(`ceil(seq_tokens/page)` 页)+ 页对齐;
    /// 显存治理器的 `cap` 退化为**上限校验 + 报告**(`ctx_max = cap − page`);
    /// `seq_tokens > ctx_max` ⇒ 拒启并给出可服务的最大 ctx。
    pub pool_auto: bool,
    /// **最终 query replay**(KVMem §5.2):需要专用 GDN 回滚缓冲。
    pub query_replay: bool,
    /// **允许池小于上下文**(锁①;`[kvmem].pool_overcommit`)。
    ///
    /// 关着时池有硬下限 `nb ≥ ceil(seq_tokens/page)` ⇒ **选择层只省访问面、
    /// 不省显存**。开着才允许 `pool_tokens` 真的压小,把 KV 挤到宿主。
    /// ⚠️ 只有引擎侧已验过"虚拟化真的开着"才会传 true(engine 侧判据已行;
    /// 本件只执行事实,不做策略)。
    pub pool_overcommit: bool,
    /// 主池 KV 量化
    pub kv: owl_models::env::KvQuant,
    /// FI 直读面启用的池量化(None = 无 FI 面)。
    /// **不是影子池量化** —— 影子池已随统一契约 P3 删除;本字段现在只表示
    /// "FI prefill 面启用",量化恒等于主池 `kv`(名随实)
    pub fi_quant: Option<owl_models::env::KvQuant>,
    /// 草稿池量化
    pub draft_kv: owl_models::env::KvQuant,
    /// **本引擎的 rank 数**(1 = 单卡;2 = TP2)—— 仅诊断/旧算式用;
    /// 2026-10-24 起池容量读 `vram` 快照(本卡),不再摊全局 LIVE。
    pub tp_ranks: usize,
    /// **池分配前的每卡 live 基线**(TP2:两 rank 共用同一快照)。
    /// `None` = 旧路/测试,回退本设备快照的 live。
    /// 为什么必须固定:池是逐 rank 分配的;rank1 计算时 rank0 池已进
    /// 本 rank 的账(per-device)不会串,但测试/旧路仍留 fallback。
    pub live_baseline_per_card: Option<u64>,
    /// **本 rank 的显存快照**(total/free/live;2026-10-24 per-device 账)。
    /// 池容量决策只读它,不再读 `owl_shared::vram` 全局三元组。
    pub vram: VramSnapshot,
    /// **草稿池几何**(层数, hkv, hd;**分片后**;P2.1-b)。`None` = 非 DFlash2 档
    /// ⇒ 池不分配;`Some` ⇒ 用它替掉写死的 `DRAFT_POOL`(双 rank 下 hkv 减半)
    pub draft_geom: Option<(usize, usize, usize)>,
    /// 草稿 dtype(与 `spec_runtime::draft_dt` 同源:恒 BF16 —— 单源
    /// `qwen35::draft_exec_dt`)
    pub draft_dt: Dtype,
    /// 投机形态(快照/MTP 链/草稿池三件的唯一事实来源)
    pub spec_mode: crate::spec_runtime::SpecMode,
    /// 投机深度(verify 块 = depth+1;verify v2 scratch 定形用;0 = off)
    pub spec_depth: usize,
    /// XQA verify 臂(SPEC_DEC=1 `M_TILESIZE=32`)
    pub xqa_verify: bool,
    /// XQA 裸 decode 臂(SPEC_DEC=1 `M_TILESIZE=16`)
    pub xqa_decode: bool,
    /// **XQA prefill 臂**(chunked q=T;同一份 `M_TILESIZE=32` 实例化)
    /// GDN 行容量
    pub gdn_slots: usize,
    /// **合批上限**(B5.1;1 = 串行)。>1 ⇒ 分配批页表块 + seqs=N 注意力 scratch
    pub max_batch: usize,
    /// spec 快照池深度
    pub snap_max: usize,
    /// VRAM 预算目标占比(0 = 治理禁用)
    pub vram_target: f64,
    /// VRAM 预留 MiB
    pub vram_reserve_mb: u64,
    /// **选择层视图块表槽**(B-2;`OWL_COVERAGE_PROBE=1` = 召回点探针档)。
    /// true ⇒ 主池多分配一个 `[1, nb]` 的视图页表块(内容 = **选中块**的物理页
    /// 按时间序;每召回点重写),与全量块表 `bt` **并存**(两者都要在场:
    /// 视图给选择层用,全量给 dense 对照与未选中块的回灌)。
    pub select_view: bool,
}

/// **状态池**(2026-10-21 S5 收户):四户组合,字段私有。
///
/// 为什么拆:原结构 31 个字段一袋 —— 几何(不变量)、主 KV 池、GDN 户、
/// 草稿/投机户、诊断旋钮混在一起,新增特性只能继续往袋里塞 `Option`。
/// 现在按**寿命与归属**分开:
///
/// | 户 | 内容 | 可变性 |
/// |---|---|---|
/// | [`PoolGeom`] | 尺寸/页数/dtype/档位旋钮 | **boot 后只读**(`Copy`) |
/// | [`KvBank`] | 主 KV 池 + 页表 + attention scratch + 表槽 | 运行期重写(内容) |
/// | [`GdnBank`] | GDN 格 + 前缀快照池 | 运行期重写(内容) |
/// | [`DraftBank`] | MTP/DFlash2 草稿池 + 草稿块表 + spec 快照 | 运行期重写(内容) |
///
/// 纪律:三户的字段**私有**(只有本模块能摸);对外只给语义方法
/// (`kv_caches()` / `write_bt()` / `snap_place()` …)。
pub(crate) struct StatePool {
    pub(crate) geom: PoolGeom,
    pub(crate) kv: KvBank,
    pub(crate) gdn: GdnBank,
    pub(crate) draft: DraftBank,
}

impl StatePool {
    /// 主 KV 池层视图(外部只读面;写只经本模块方法)。
    pub(crate) fn kvs(&self) -> &[KvBlocks] {
        self.kv.kvs()
    }

    // ── 表件写入的**接口函数**(S6e)──
    // 状态归 `KvBank` 所有 ⇒ 逻辑在 `impl KvBank`;这里只是池的对外接口
    // (引擎拿的是 `StatePool`,不该为写三张表去拆组合)。

    /// **FI plan 表规格**(唯一构造点):`t` 行 × `ctx_total` 列,几何取自池。
    ///
    /// 三个刷表点(桶图 / decode / verify)原本各拼一份 `FiPlanSpec` ——
    /// 任何几何口径变化都要三处同步。
    pub(crate) fn fi_plan_spec(&self, t: usize, ctx_total: usize) -> owl_iface::contract::FiPlanSpec {
        let d = self.geom.dims;
        owl_iface::contract::FiPlanSpec {
            t,
            ctx_total,
            hq: d.hq,
            hkv: d.hkv,
            hd: d.hd,
            page: self.geom.page,
        }
    }

    /// FI 表四件写入(见 [`KvBank::write_fi_tables`])。
    pub(crate) async fn write_fi_tables<D: DeviceClient>(
        &self,
        face: &mut D,
        fi: &crate::exec::prefill_chunk::FiTables,
    ) -> Result<FiTableLeaves> {
        self.kv.write_fi_tables(face, fi).await
    }

    /// GDN 序列表写入(见 [`KvBank::write_gdn_tables`])。
    pub(crate) async fn write_gdn_tables<D: DeviceClient>(
        &self,
        face: &mut D,
        t_real: usize,
        nt: usize,
    ) -> Result<Option<GdnTableSlotOwned>> {
        self.kv.write_gdn_tables(face, t_real, nt).await
    }

}

/// **池几何 / 档位**(boot 期定型,运行期只读)。
#[derive(Clone, Copy)]
pub(crate) struct PoolGeom {
    /// 池几何输入(spec 提取;prefill 叶子构造/执行器 dtype 分支用)
    pub(crate) dims: ModelDims,
    /// paged 池几何(页;legacy 模式 page = 容量,nb = 1)
    pub(crate) page: usize,
    /// **管理面**页数(store 块腿/会话块表/FI 页链槽只认这些)
    pub(crate) nb: usize,
    /// **物理**页数(历史字段;prefill 桶退役后恒 = `nb`)
    pub(crate) nb_kv: usize,
    pub(crate) paged: bool,
    /// GDN 行容量(knobs.gdn_slots,boot 定谳;原自由函数 gdn_slots())
    pub(crate) gdn_slots: usize,
    /// FI 直读旗标(统一契约 P3:影子池删除,FI 直读主池 kNHD 叶子)
    pub(crate) fi_face: bool,
    /// B6.2:主 KV 池 fp8 e4m3 承载(true = 池块 U32 字节承载 1B/elem;
    /// 读核走 *_fp8 变体,写核 K0 转换写)
    pub(crate) kv_fp8: bool,
    /// 每页 token 折叠因子(页内行布局;legacy = 1)
    x: usize,
}

/// **主 KV 池 + 页表 + attention scratch + 表槽**(运行期重写内容,地址恒定)。
pub(crate) struct KvBank {
    /// 每 full-attention 层一对 K/V 池块
    kvs: Vec<KvBlocks>,
    /// 会话真链块表(持久块;图烘焙指针 ⇒ 地址必须恒定)
    bt: Bytes,
    /// **选择层视图页表**(B-2;`None` = 本档无视图面)。与 `bt` 并存:
    /// `bt` = 会话真链(dense),本块 = 选中块的物理页按**时间序**。
    bt_view: Option<Bytes>,
    /// **批 decode 页表块**([max_batch, nb] f32;每行一条会话链)
    bt_batch: Option<Bytes>,
    /// v2 分页 decode scratch(E-decode;`None` = legacy/无页策略)
    attn_v2: Option<owl_models::module::AttnV2Scratch>,
    /// verify v2 scratch(seqs=depth1;§三十一)
    verify_attn_v2: Option<owl_models::module::AttnV2Scratch>,
    /// **批 decode v2 scratch**(B5.1;seqs=max_batch)
    batch_attn_v2: Option<owl_models::module::AttnV2Scratch>,
    /// XQA 资源面(verify 与裸 decode **共用**)
    xqa: Option<owl_models::module::XqaVerifyCtx>,
    /// **GDN 序列表持久槽**(P-6b):`gdn_chunked` 六核的三张 FLA 表,
    /// 每 chunk 原位写(图烘焙指针 ⇒ 地址必须恒定)
    gdn_tables: Option<GdnTableBlocks>,
    /// **FlashInfer 表件持久槽**(P-3):`q_cu/indices/indptr/last_len` 四件
    fi_tables: Option<FiTableBlocks>,
}

/// **v2 分页 attention 的 PARTITION 长**(token;与核契约同源)。
///
/// `nparts = ceil(池 token 数 / 本值)` —— 三档 scratch 与叶子形状的**单源**。
pub(crate) const V2_PARTITION_TOKENS: usize = 512;

impl KvBank {
    /// **v2 attention scratch 工厂**(decode / verify / 合批 三档同构)。
    ///
    /// 只差伪序列数 `seqs`(decode=1 / verify=depth+1 / 合批=max_batch)与标签;
    /// 形状/dtype 规则写在这里,三处不再各抄一遍。
    async fn alloc_attn_v2<D: DeviceClient>(
        face: &mut D,
        dims: ModelDims,
        nparts: usize,
        seqs: usize,
        tag: &str,
    ) -> Result<owl_models::module::AttnV2Scratch> {
        let es = zero_block_dt(face, seqs * dims.hq * nparts, Dtype::F32, &format!("{tag}.exp_sums")).await?;
        let ml = zero_block_dt(face, seqs * dims.hq * nparts, Dtype::F32, &format!("{tag}.max_logits")).await?;
        let to = zero_block_dt(
            face,
            seqs * dims.hq * nparts * dims.hd,
            dims.dtype,
            &format!("{tag}.tmp_out"),
        )
        .await?;
        Ok(owl_models::module::AttnV2Scratch {
            exp_sums: block_leaf(&es.0, vec![seqs, dims.hq * nparts]),
            max_logits: block_leaf(&ml.0, vec![seqs, dims.hq * nparts]),
            tmp_out: block_leaf_dt(&to.0, vec![seqs * dims.hq * nparts * dims.hd], dims.dtype),
            nparts,
            seqs,
        })
    }

    /// 主 KV 池层视图(层数 = full-attention 层数)。
    pub(crate) fn kvs(&self) -> &[KvBlocks] {
        &self.kvs
    }

    /// **FI 表四件写入**(P-3):持久槽 ⇒ 地址恒定(图可直接烘);
    /// 单条批量命令 + 回执即落地(旧实现每 chunk 4 次 `eval_ops(from_host)`
    /// = 4 次往返同步)。
    ///
    /// 返回四件叶子(形状随本 chunk 实际用量;`nb_used` 只影响 indices 声明宽)。
    pub(crate) async fn write_fi_tables<D: DeviceClient>(
        &self,
        face: &mut D,
        fi: &crate::exec::prefill_chunk::FiTables,
    ) -> Result<FiTableLeaves> {
        let fb = self.fi_tables.as_ref().ok_or_else(|| ModelError::EngineInvariant {
            what: "FI 表槽未分配(池几何与档位不一致)".into(),
        })?;
        if fi.nb_used > fb.nb_cap {
            return Err(ModelError::Msg(format!(
                "FI 页链超槽容量:nb_used={} > nb_cap={}(池预算/几何不一致)",
                fi.nb_used, fb.nb_cap
            )));
        }
        let (w_q_cu, w_idx) = (crate::exec::u32s_as_f32(&fi.q_cu), crate::exec::u32s_as_f32(&fi.indices));
        let (w_ind, w_ll) = (crate::exec::u32s_as_f32(&fi.indptr), crate::exec::u32s_as_f32(&fi.last_len));
        face.write_blocks_f32_on(
            owl_iface::contract::UpStream::Compute,
            &[
                (&fb.q_cu, 0, w_q_cu.as_slice()),
                (&fb.indices, 0, w_idx.as_slice()),
                (&fb.indptr, 0, w_ind.as_slice()),
                (&fb.last_len, 0, w_ll.as_slice()),
            ],
        )
        .await?;
        Ok(FiTableLeaves {
            q_cu: TensorOps::of_block(fb.q_cu.id, Dtype::U32, vec![2]),
            indices: TensorOps::of_block(fb.indices.id, Dtype::U32, vec![fi.nb_used]),
            indptr: TensorOps::of_block(fb.indptr.id, Dtype::U32, vec![2]),
            last_len: TensorOps::of_block(fb.last_len.id, Dtype::U32, vec![1]),
        })
    }

    /// **GDN 序列表写入**(P-6b):三张 FLA 表(`cu|coff|idx`)按**本 chunk
    /// 真行数 t** 铸,写进持久槽(地址恒定 ⇒ 图可烘)。
    ///
    /// 为什么必须按真行数:核的行区间与递推落点全由这三张表定 —— 按桶几何
    /// `t_pad` 生成会把 padding 假行算进序列(递推态污染,`gdn_bucket_pad_probe`
    /// 实测终态 Δ=3.6e-2 → 0)。
    ///
    /// 返回 `(表叶子, nt)`;无 GDN 层 = `None`。
    /// `t_real` = 本块**真行数**(表的 `cu`/内容口径);`nt` = 布局口径的 chunk 数
    /// (eager = `ceil(t_real/BT)`;桶路 = `ceil(t_pad/BT)` —— 图内烘的是它)。
    /// 写入流恒为默认流(桶路已退役 ⇒ 单一路径;原 `GdnTableStream` 参数删除)。
    pub(crate) async fn write_gdn_tables<D: DeviceClient>(
        &self,
        face: &mut D,
        t_real: usize,
        nt: usize,
    ) -> Result<Option<GdnTableSlotOwned>> {
        let Some(gt) = self.gdn_tables.as_ref() else {
            return Ok(None);
        };
        let nt = nt.max(1);
        if nt > gt.nt_cap {
            return Err(ModelError::Msg(format!(
                "GDN 序列表超槽容量:nt={nt} > nt_cap={}(seq_tokens/chunk 配置不一致)",
                gt.nt_cap
            )));
        }
        let bytes = owl_models::kernel::gdn_table_host_bytes(t_real, nt);
        let f32bits: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().expect("4 字节")))
            .collect();
        face.write_block_f32(&gt.table, 0, &f32bits).await?;
        Ok(Some(GdnTableSlotOwned {
            leaf: TensorOps::of_block(
                gt.table.id,
                Dtype::U32,
                vec![owl_models::kernel::gdn_table_bytes(nt) / 4],
            ),
            nt,
        }))
    }

}

/// FI 表四件叶子(形状随本 chunk 用量;`kcs/vcs` 由池的层视图另取)。
pub(crate) struct FiTableLeaves {
    pub q_cu: TensorOps,
    pub indices: TensorOps,
    pub indptr: TensorOps,
    pub last_len: TensorOps,
}

/// GDN 序列表槽:叶子 + 本 chunk 的 chunk 数(`nt = ceil(t / BT)`)。
pub(crate) struct GdnTableSlotOwned {
    pub leaf: TensorOps,
    pub nt: usize,
}

/// **GDN 户**:格状态 + 前缀快照池(仅非 full 层)。
pub(crate) struct GdnBank {
    gdns: Vec<GdnBlocks>,
    snaps: GdnSnapPool,
    snap_tick: u64,
}

/// **草稿 / 投机户**:MTP、DFlash2 与 spec 回滚缓冲(与主池互斥/伴随 spec 模式)。
pub(crate) struct DraftBank {
    /// spec 快照(E5-M2:投机轮回滚缓冲;`OWL_SPEC_DEPTH>0` 时分配)
    spec_snap: Option<SpecSnapBuf>,
    /// **最终 query replay** 的 GDN 回滚缓冲(`[kvmem].query_replay`)。
    replay_snap: Option<SpecSnapBuf>,
    /// MTP 草稿链 KV 池(E5-M2b;独立页池,1 层)
    mtp_kvs: Option<KvBlocks>,
    /// DFlash2 草稿 KV 池(E5-DF2;独立页池,5 层草稿几何 8/128)
    dflash_kvs: Option<Vec<KvBlocks>>,
    /// MTP 链持久块表槽(E5-M5 propose 图烘焙;[1, nb] f32)
    bt_mtp: Option<Bytes>,
    /// 草稿池是否 fp8 承载(B6:U32 扁平字节账;叶子注解随此)
    fp8: bool,
    /// 草稿池 elem dtype(非 fp8 时;随检查点事实:量化族 BF16 / 未量化族 F16)
    dt: Dtype,
    /// **DFlash2 ring 槽数**(0 = 线性全池;>0 = 逻辑位 `% ring_slots`)。
    /// 草稿 config 的 sliding_window=2048 + 自块 8 ⇒ 2080(页对齐)。
    ring_slots: usize,
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

// ============================================================================
// 显存预算的**具名项**(B1,2026-10-17)
// ============================================================================
//
// 病(原状,首轮审查 §B1):预算 = 一行乘法(`per_tok_bytes`)+ 一锅端
// `(400<<20)`——图 slab 随图族/臂换代变大时**静默失真** ⇒ 池上限偏大 ⇒
// 请求级 OOM(20G 档生产 OOM 正是这类形状)。故:①每项具名;②每项在 boot
// 日志**逐项打印**(可与实测对账);③能用实测的项一律带实测来源(禁裸常数)。

/// DFlash2 草稿池几何(**state.rs 内单源**;数值来源 = z-lab 检查点家族定形
/// = 5 层 × 8 KV 头 × hd128)。预算项与分配项都必须从这里派生 ——
/// 原两处各写一遍(`5*2*8*128*2` 逐 token 项 / `(8,128,5)` 分配项)。
/// (挂账:更远的单源是草稿模型实例自己的 hkv/hd/n_layers —— 需 models 侧
/// 开只读访问器 + PoolPlan 携几何;本轮只收同文件内两份。)
#[derive(Debug, Clone, Copy)]
struct DraftPoolGeom {
    layers: usize,
    hkv: usize,
    hd: usize,
}

impl DraftPoolGeom {
    /// **从声明事实构造**(P2.1-b;分片后的 (层数, hkv, hd))
    fn from_geom(layers: usize, hkv: usize, hd: usize) -> Self {
        DraftPoolGeom { layers, hkv, hd }
    }

    /// 逐 token 字节(K+V 两张量 × 层数 × 头 × hd × elem)
    fn per_tok_bytes(self, elem: usize) -> usize {
        self.layers * 2 * self.hkv * self.hd * elem
    }

    /// 单层池元素数(页口径:K/V 同形,dtype 由调用方按池量化定)
    fn layer_elems(self, nb: usize, page: usize) -> usize {
        nb * self.hkv * self.hd * page
    }
}

/// 草稿池几何值(z-lab 家族;与 `DFlash2Draft::new_with_plan_dt` 的
/// hkv/hd/n_layers 同形)
/// ⚠️ **单卡缺省**(z-lab 家族事实;双 rank 走 `PoolPlan::draft_geom` —— 见 P2.1-b)
const DRAFT_POOL: DraftPoolGeom = DraftPoolGeom { layers: 5, hkv: 8, hd: 128 };

/// **逐 token 线性项**:池上限 = 可用预算 / 本值。
/// 组成 = 主池 KV(全注意层 × K/V × hkv × hd × elem)+ 草稿池(同上,草稿几何)。
#[derive(Debug, Clone, Copy)]
struct PerTokBytes {
    main_kv: usize,
    /// 草稿池(DFlash2;按层 × 几何 × elem 线性)
    draft_pool: usize,
    /// **MTP 草稿链**(1 层 full 几何 × k+v;此前**根本没进预算** ⇒ `pool_tokens`
    /// 是手写值时主池会吃满、MTP 池在预算外另占 864 MiB@192k ⇒ turn 级 OOM)
    mtp_chain: usize,
}

impl PerTokBytes {
    /// `draft_elem` = `Some(池 elem 字节)` 当且仅当 DFlash2 档(草稿池按 token 线性;
    /// 页取整后略同)。**elem 来自 `KvQuant::size_bytes` 单一映射** —— 原先草稿池
    /// 项写死 2B,池改 fp8 时预算不会跟(= 声明没驱动算式)。
    #[allow(clippy::too_many_arguments)]
    fn of(
        dims: &ModelDims,
        n_full: usize,
        kv_elem: usize,
        draft_elem: Option<usize>,
        draft_geom: Option<(usize, usize, usize)>,
        mtp_kv_elem: Option<usize>,
    ) -> Self {
        // 几何单源:配置/声明给的 `draft_geom` 优先(分片后),否则单卡缺省
        let geom = draft_geom
            .map(|(l, h, d)| DraftPoolGeom::from_geom(l, h, d))
            .unwrap_or(DRAFT_POOL);
        PerTokBytes {
            main_kv: n_full * 2 * dims.hkv * dims.hd * kv_elem,
            draft_pool: draft_elem.map(|e| geom.per_tok_bytes(e)).unwrap_or(0),
            // MTP 链 = **1 层 × full 几何 × (k+v)**;elem 由 `[spec] draft_kv_fp8` 定
            mtp_chain: mtp_kv_elem.map(|e| 2 * dims.hkv * dims.hd * e).unwrap_or(0),
        }
    }

    fn total(self) -> usize {
        self.main_kv + self.draft_pool + self.mtp_chain
    }

    fn explain(self) -> String {
        format!(
            "主池 {}B + 草稿池 {}B + MTP 链 {}B",
            self.main_kv, self.draft_pool, self.mtp_chain
        )
    }
}

/// **图 slab 预算项**(实测校准,2026-10-17):spec-dflash 生产档
/// `slab 总账` = session 32 + verify 165(17 段/16 断点)+ fold 58(7)
/// + dflash 6 = **261MiB**(ctx16k / fp8 池 / FI 双图;实测精确值略高于 261MiB,
/// 故取整到 **262MiB** 上界 —— 日志按 MiB 取整打印,常数不得骑在边界上)。
/// ⚠️ 随图族变(depth / 臂换代 / 桶)⇒ 捕获后由 [`crate::graph_fleet`]
/// 把**实测 Σ** 与本品对账,超了打 ⚠️(池上限可能偏大 ⇒ 请求级 OOM 风险)。
/// ⚠️ **2026-10-21 prefill 桶退役:640 → 320 MiB**。
/// 历史:2026-10-19 为桶族(296.5)+ 尾档 t1024(502.3)把 262 抬到 640;
/// 桶已删除,非桶图族(spec-dflash 实测 261MiB)回到 320 = 262 + 58MiB 余量。
/// 总治理口径 778 → 458 MiB(WARMUP_MISC 138 不变),KV 池多 ~320MiB。
pub(crate) const GRAPH_SLAB_BUDGET: usize = 320 << 20;

/// warmup 瞬态 + 杂项(治理器之后、图之外的固定分配)。未单列实测;
/// = 原一锅端 `400MiB` 减图项后的余量(总分不变 ⇒ 历史治理数字可比):
/// 400 − 262 = 138MiB。
const WARMUP_MISC_BUDGET: usize = 138 << 20;

/// **治理器之后的固定分配项**(原为一行 `n_gdn*slots*elems*4 + snap*… + 400MiB`)
#[derive(Debug, Clone, Copy)]
struct FixedAfter {
    /// GDN 状态格(gdn_slots 格 × 每格元素 × f32)。
    /// **对账口径**:真分配应等于此式(`tests.rs` 用池访问器逐项对账;
    /// 访问器见 `StatePool::{gdn_row_bytes,snapshot_pool_bytes,draft_pool_bytes}`,
    /// `#[cfg(test)]` 门控 —— 本预算在池分配之前就要算,故生产面不能读池)。
    /// 0.8B 档实测 18 层 × 1.07MiB ≈ 19.3MiB/行。
    gdn_grid: usize,
    /// GDN 快照池(同格 × snap_max)
    gdn_snap: usize,
    /// **spec 快照缓冲**(独立于前缀快照池;`spec_mode != Off` 且 paged 时分配)
    spec_snap: usize,
    /// **query replay 回滚缓冲**(独立于 spec 快照;`query_replay` 且 paged 时分配)
    replay_snap: usize,
    /// 图 slab 预算(见 [`GRAPH_SLAB_BUDGET`])
    graph_slab: usize,
    /// warmup 瞬态 + 杂项(见 [`WARMUP_MISC_BUDGET`])
    warmup_misc: usize,
    /// **DFlash2 草稿池 ring**(固定项;不再按 token 线性)
    draft_ring: usize,
}

impl FixedAfter {
    fn of(
        n_gdn: usize,
        gdn_slots: usize,
        gdn_slot_elems: usize,
        snap_max: usize,
        draft_ring: usize,
        spec_snap: bool,
        replay_snap: bool,
    ) -> Self {
        let grid = n_gdn * gdn_slots * gdn_slot_elems * 4;
        let snap_row = n_gdn * gdn_slot_elems * 4;
        FixedAfter {
            gdn_grid: grid,
            gdn_snap: snap_max * snap_row,
            spec_snap: if spec_snap { snap_row } else { 0 },
            replay_snap: if replay_snap { snap_row } else { 0 },
            graph_slab: GRAPH_SLAB_BUDGET,
            warmup_misc: WARMUP_MISC_BUDGET,
            draft_ring,
        }
    }

    fn total(self) -> usize {
        self.gdn_grid
            + self.gdn_snap
            + self.spec_snap
            + self.replay_snap
            + self.graph_slab
            + self.warmup_misc
            + self.draft_ring
    }

    fn explain(self) -> String {
        let mbf = |v: usize| v as f64 / 1048576.0;
        format!(
            "gdn格 {:.0}MiB + gdn快照 {:.0}MiB + spec快照 {:.0}MiB + replay快照 {:.0}MiB + 图slab {:.0}MiB + warmup/杂项 {:.0}MiB + 草稿ring {:.0}MiB",
            mbf(self.gdn_grid),
            mbf(self.gdn_snap),
            mbf(self.spec_snap),
            mbf(self.replay_snap),
            mbf(self.graph_slab),
            mbf(self.warmup_misc),
            mbf(self.draft_ring)
        )
    }
}

impl StatePool {
    /// **单逻辑块(页)的设备字节**(账/搬运长度真源,坑册 §42):Σ 全注意力层
    /// (k+v) 每页实占字节。fp8 池以 U32 承载(1B/elem)⇒ 元素数 ÷4、元素宽 ×4
    /// ⇒ 每页字节 = hkv×hd×page ×1;f16 ⇒ ×2。dump 页不算(**÷ nb_kv**)。
    pub(crate) fn block_bytes(&self) -> usize {
        let total: usize = self.kv.kvs
            .iter()
            .map(|kv| {
                kv.k_cache.1 * elem_bytes(kv.k_cache.0.dtype)
                    + kv.v_cache.1 * elem_bytes(kv.v_cache.0.dtype)
            })
            .sum();
        total / self.geom.nb_kv.max(1)
    }

    /// **主池布局**(engine 注册给 store 的 id 级描述;设计卷 §10.3 推论 A)。
    /// 每层 k/v 各一项;`stride_bytes` = 每页字节(槽步长)。
    pub(crate) fn pool_layout(&self, epoch: u64) -> owl_store::PoolLayout {
        let mut pieces = Vec::with_capacity(self.kv.kvs.len() * 2);
        for kv in &self.kv.kvs {
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
        let kv = self.draft.mtp_kvs.as_ref()?;
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
        self.draft.mtp_kvs
            .as_ref()
            .map(|kv| {
                self.per_page_bytes(kv.k_cache.0.dtype, kv.k_cache.1)
                    + self.per_page_bytes(kv.v_cache.0.dtype, kv.v_cache.1)
            })
            .unwrap_or(0)
    }

    /// 一个缓冲的每页字节(元素数 × 元素宽 ÷ 管理面页数)。
    fn per_page_bytes(&self, dtype: Dtype, elems: usize) -> usize {
        elems * elem_bytes(dtype) / self.geom.nb_kv.max(1)
    }

    // ── 下面三件：**测试对账面**（`tests.rs` 用它们把"池真分配"与预算算式对账）。
    // 生产零调用者（预算算式在池分配**之前**就要用 ⇒ 结构上不能读池），
    // 故按 `tests` 模块同款门控 —— 既保住对账能力，也不让 release 面背死代码告警
    // （2026-10-17 审查：先删后补，补的理由见 commit 信息）。
    // ⚠️ **门必须是 `any(test, feature = "e2e")`**（2026-10-18 W5 修）：`tests.rs` 在
    // `feature = "e2e"` 下也会编进 lib（无 cfg(test)）⇒ 只写 `#[cfg(test)]` 会让
    // **e2e 目标整体编不过**（`engine.e2e.rs` 恒 build 失败 = 那条门从未跑起来）。
    /// **GDN 行字节 / 每活跃 turn**(`owl_store::BudgetInputs::gdn_row_bytes` 真源)。
    ///
    /// 口径:每层四件(conv_q/conv_k/conv_v/rec)的分配宽 = `gdn_slots × 单槽宽`
    /// (`StatePool::alloc` 里显式乘过)⇒ **行字节 = 池总字节 ÷ gdn_slots**。
    /// 读真分配,不按公式重算(0.8B 档实测:18 层 × 1.07MiB ≈ 19.3MiB/行)。
    // 调用约束:**池已装配之后**才可读(准入预算组装 `assemble_budget` / 预算探针);
    // 生产热路径不调。原 `#[cfg(any(test, feature = "e2e"))]` 门对前者过严(2026-10-19 合并)。
    pub(crate) fn gdn_row_bytes(&self) -> usize {
        let pool: usize = self.gdn.gdns
            .iter()
            .map(|g| {
                (g.conv_q.1 + g.conv_k.1 + g.conv_v.1 + g.rec.1) * elem_bytes(g.rec.0.dtype)
            })
            .sum();
        pool / self.geom.gdn_slots.max(1)
    }

    /// **快照池字节**(E2c 静态预分配;固定面的一项,含 spec / replay 快照缓冲)。
    // 调用约束:**池已装配之后**才可读(准入预算组装 `assemble_budget` / 预算探针);
    // 生产热路径不调。原 `#[cfg(any(test, feature = "e2e"))]` 门对前者过严(2026-10-19 合并)。
    pub(crate) fn snapshot_pool_bytes(&self) -> usize {
        let pool: usize = self.gdn.snaps
            .bufs
            .iter()
            .map(|b| b.1 * elem_bytes(b.0.dtype))
            .sum();
        let spec = self.draft.spec_snap
            .as_ref()
            .map(|s| s.bufs.iter().map(|b| b.1 * elem_bytes(b.0.dtype)).sum::<usize>())
            .unwrap_or(0);
        let replay = self.draft.replay_snap
            .as_ref()
            .map(|s| s.bufs.iter().map(|b| b.1 * elem_bytes(b.0.dtype)).sum::<usize>())
            .unwrap_or(0);
        pool + spec + replay
    }

    /// **草稿池字节**(MTP 链 / DFlash2 ring;关 = 0)。DFlash2 自 2026-10-22
    /// 起按 **window+block 的 ring** 分配(不再等于第二个主池);MTP 链仍按
    /// `pool_tokens` 线性(`BudgetInputs::draft_pool_bytes`)。
    // 调用约束:**池已装配之后**才可读(准入预算组装 `assemble_budget` / 预算探针);
    // 生产热路径不调。原 `#[cfg(any(test, feature = "e2e"))]` 门对前者过严(2026-10-19 合并)。
    pub(crate) fn draft_pool_bytes(&self) -> usize {
        fn kv_bytes(kv: &KvBlocks) -> usize {
            kv.k_cache.1 * elem_bytes(kv.k_cache.0.dtype)
                + kv.v_cache.1 * elem_bytes(kv.v_cache.0.dtype)
        }
        let mtp = self.draft.mtp_kvs.as_ref().map(kv_bytes).unwrap_or(0);
        let df: usize = self.draft.dflash_kvs.as_ref().map(|ks| ks.iter().map(kv_bytes).sum()).unwrap_or(0);
        mtp + df
    }

    /// 分配全部状态块(零初始化;GDN 段每 turn 开始按格重置)。
    /// `pool_tokens` = 块池 token 口径容量(E2b:默认 2 × 单会话容量,
    /// OWL_POOL_TOKENS 可覆写;调用方读 env,本函数只管取整到页)。
    #[allow(clippy::too_many_arguments)]
    /// **池装配**(S5b 分解):定形/预算 → 三户各自分配。
    ///
    /// 为什么拆:`alloc` 原本 448 行一竿子到底(决策 + 预算 + 8 类资源分配),
    /// 新增一个特性只能往函数尾追加。现在:
    /// - `AllocCtx::build` = **纯 host** 的决策/预算/几何(零设备调用);
    /// - `KvBank::alloc` / `GdnBank::alloc` / `DraftBank::alloc`
    ///   = 各自的"哪档分配哪几个字段"(判据与资源同屋)。
    pub(crate) async fn alloc<D: DeviceClient>(
        face: &mut D,
        dims: ModelDims,
        layer_types: &[bool],
        plan: &PoolPlan,
    ) -> Result<StatePool> {
        let ctx = AllocCtx::build(dims, layer_types, plan)?;
        let t = std::time::Instant::now();
        let kv = KvBank::alloc(face, &ctx).await?;
        let gdn = GdnBank::alloc(face, &ctx).await?;
        let draft = DraftBank::alloc(face, &ctx).await?;
        // 池 dtype 标签随实报(原硬写 "kv f16" 在 fp8 档说谎 —— 2026-10-13)
        let kv_label = match plan.kv {
            owl_models::env::KvQuant::Fp8E4M3 => "kv fp8 e4m3",
            owl_models::env::KvQuant::Fp8E5M2 => "kv fp8 e5m2",
            owl_models::env::KvQuant::None => "kv f16",
        };
        owl_shared::log_info!("boot", "状态块分配 {:.2}s({kv_label} ×{} / gdn ×{},会话容量 {},池 {nb} 页 × {page} tok,页表 nb = {nb},dflash = {df},XQA sm = {xqa_sm})",
            t.elapsed().as_secs_f32(),
            kv.kvs.len(),
            gdn.gdns.len(),
            ctx.seq_tokens,
            nb = ctx.geom.nb,
            page = ctx.geom.page,
            df = draft.dflash_kvs.is_some(),
            xqa_sm = kv.xqa.as_ref().map(|c| c.sm_count).unwrap_or(0),
        );
        Ok(StatePool { geom: ctx.geom, kv, gdn, draft })
    }

    /// MTP 链块表叶子(propose 图烘焙;None = 未启用)
    pub(crate) fn bt_mtp_leaf(&self) -> Option<TensorOps> {
        self.draft.bt_mtp
            .as_ref()
            .map(|b| TensorOps::of_block(b.id, Dtype::F32, vec![1, self.geom.nb]))
    }

    /// MTP 链块表槽重写(每轮 propose 前;内容 = 会话 mtp 链)
    pub(crate) async fn write_bt_mtp<D: DeviceClient>(
        &self,
        face: &mut D,
        table: &[u32],
    ) -> Result<()> {
        let b = self.draft.bt_mtp.as_ref().ok_or_else(|| ModelError::EngineInvariant { what: "bt_mtp 未分配(mtp 档位不一致)".into() })?;
        let mut v = vec![0f32; self.geom.nb];
        for (i, x) in table.iter().enumerate() {
            v[i] = *x as f32;
        }
        face.write_block_f32_on(owl_iface::contract::UpStream::Compute, b, 0, &v)
            .await
    }

    /// GDN 层数(fold 记录槽计数用)
    pub(crate) fn gdn_count(&self) -> usize {
        self.gdn.gdns.len()
    }

    // DFlash2 草稿池寻址权威(唯一出口;slot≥1 案配套,2026-10-10)
    // 草稿池 = **逻辑位 → ring 槽**(2026-10-22;0 = 线性):NC 核前缀读
    // 按 `p % ring_slots` 映射;encode 写槽与 propose 自槽同源。写者
    // (exec encode)/读者(propose 图/NC)禁止各自推槽(两套语言曾分叉)。
    // 并发多 turn(B5 batch)共享线性空间会互踩 → B5 需 per-session 分区。

    /// **DFlash2 ring 槽数**(0 = 线性;报告/闭包捕获用)
    pub(crate) fn draft_ring_slots(&self) -> usize {
        self.draft.ring_slots
    }

    /// DFlash2 encode 行 i(基址 base)的 ring 槽
    pub(crate) fn draft_dflash_encode_slot(&self, base: usize, i: usize) -> f32 {
        self.draft.dflash_slot(base + i)
    }

    /// DFlash2 propose 自块行 i 的 ring 槽
    pub(crate) fn draft_dflash_self_slot(&self, pos: usize, i: usize) -> f32 {
        self.draft.dflash_slot(pos + i)
    }

    /// propose 读窗长(fp = 下轮 fed 位)
    #[allow(dead_code)] // 草稿面预测助手(现无读者,保留供 NC v2 路径)
    pub(crate) fn draft_kv_len(fp: usize) -> f32 {
        (fp + 8) as f32
    }

    pub(crate) fn dflash_kv_leaves(&self) -> Option<Vec<(TensorOps, TensorOps)>> {
        let ks = self.draft.dflash_kvs.as_ref()?;
        let (dkv, dhd) = (8usize, 128usize);
        // B6:fp8 草稿池 = U32 扁平字节账(主池同款)
        if self.draft.fp8 {
            let shape = vec![self.geom.nb * dkv * dhd * self.geom.page / 4];
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
                    // 是 BF16。旧实现传 `self.geom.dims.dtype`(目标池 F16)= 注解撒谎,
                    // 被 L3 事实面当场抳住(2026-10-15)。
                    block_leaf_dt(
                        &kb.k_cache.0,
                        vec![self.geom.nb, dkv, dhd / self.geom.x, self.geom.page, self.geom.x],
                        self.draft.dt,
                    ),
                    block_leaf_dt(
                        &kb.v_cache.0,
                        vec![self.geom.nb, dkv, dhd, self.geom.page],
                        self.draft.dt,
                    ),
                )
            })
            .collect())
    }

    /// MTP 草稿链 KV 叶子对(mtp 模式;None = 未启用)
    ///
    /// fp8 档(`[spec] draft_kv_fp8`)= **U32 扁平字节账**(元素数 = 字节/4;
    /// 与主池/DFlash2 池同一承载律)—— 叶子注解必须跟着走,否则 L3 事实面
    /// 会把"U32 字节账"当成元素账(旧实现恒传 `dims.dtype` = 注解撒谎)。
    pub(crate) fn mtp_kv_leaf(&self) -> Option<(TensorOps, TensorOps)> {
        let mk = self.draft.mtp_kvs.as_ref()?;
        if self.draft.fp8 {
            let d = &self.geom.dims;
            let shape = vec![self.geom.nb * d.hkv * d.hd * self.geom.page / 4];
            return Some((
                block_leaf_dt(&mk.k_cache.0, shape.clone(), Dtype::U32),
                block_leaf_dt(&mk.v_cache.0, shape, Dtype::U32),
            ));
        }
        Some((
            block_leaf_dt(&mk.k_cache.0, self.k_shape(), self.geom.dims.dtype),
            block_leaf_dt(&mk.v_cache.0, self.v_shape(), self.geom.dims.dtype),
        ))
    }

    /// v2 scratch 叶子(None = legacy;引擎按步注入 ctx.attn_v2)
    pub(crate) fn attn_v2_scratch(&self) -> Option<owl_models::module::AttnV2Scratch> {
        self.kv.attn_v2.clone()
    }

    /// verify v2 scratch(seqs=depth1;§三十一)
    pub(crate) fn verify_attn_v2_scratch(&self) -> Option<owl_models::module::AttnV2Scratch> {
        self.kv.verify_attn_v2.clone()
    }

    /// 批 decode v2 scratch(seqs=max_batch;None = 串行档)
    pub(crate) fn batch_attn_v2_scratch(&self) -> Option<owl_models::module::AttnV2Scratch> {
        self.kv.batch_attn_v2.clone()
    }

    /// 批 decode 页表叶子([max_batch, nb] —— seqs 维必须与 scratch 同源)
    pub(crate) fn bt_batch_leaf(&self) -> Option<TensorOps> {
        let seqs = self.kv.batch_attn_v2.as_ref().map(|s| s.seqs)?;
        self.kv.bt_batch
            .as_ref()
            .map(|b| TensorOps::of_block(b.id, Dtype::F32, vec![seqs, self.geom.nb]))
    }

    /// **批页表重写**:每行一条会话链(不足 nb 补 0 = padding 行)。
    /// 与 [`Self::write_bt`] 同守占位符门(洞不许进设备表)。
    pub(crate) async fn write_bt_batch<D: DeviceClient>(
        &self,
        face: &mut D,
        rows: &[Vec<u32>],
    ) -> Result<()> {
        let b = self.kv.bt_batch.as_ref().ok_or_else(|| ModelError::EngineInvariant {
            what: "bt_batch 未分配(max_batch 档位不一致)".into(),
        })?;
        for (r, table) in rows.iter().enumerate() {
            if let Some(i) = table.iter().position(|x| owl_store::is_host_placeholder(*x)) {
                return Err(ModelError::Msg(format!(
                    "write_bt_batch:行 {r} 第 {i} 项是宿主占位符 —— 不得写设备块表"
                )));
            }
        }
        let w = self.geom.nb;
        let mut v = vec![0f32; rows.len() * w];
        for (r, table) in rows.iter().enumerate() {
            for (i, x) in table.iter().take(w).enumerate() {
                v[r * w + i] = *x as f32;
            }
        }
        face.write_block_f32_on(owl_iface::contract::UpStream::Compute, b, 0, &v)
            .await
    }

    /// XQA 资源面(常量件;`seq_lens`/`semaphores` 由图输入槽供给,调用方覆盖)
    pub(crate) fn xqa_ctx(&self) -> Option<owl_models::module::XqaVerifyCtx> {
        self.kv.xqa.clone()
    }

    /// XQA semaphores 元素数(图输入槽写零用;0 = 本档无 XQA 面)
    pub(crate) fn xqa_sem_n(&self) -> usize {
        self.kv.xqa
            .as_ref()
            .map(|x| x.semaphores.shape().iter().product::<usize>())
            .unwrap_or(0)
    }

    /// XQA semaphores 块 + 字节数(每轮发射前清零用)
    #[allow(dead_code)] // XQA sem 清零助手(半接入,保留)
    pub(crate) fn xqa_sem_zero(&self) -> Option<(owl_iface::contract::Bytes, usize)> {
        let x = self.kv.xqa.as_ref()?;
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
        let d = &self.geom.dims;
        if self.geom.paged {
            vec![self.geom.nb, d.hkv, d.hd / self.geom.x, self.geom.page, self.geom.x]
        } else {
            vec![self.geom.nb, d.hkv, d.hd] // legacy:nb=1,page=容量
        }
    }
    pub(crate) fn v_shape(&self) -> Vec<usize> {
        let d = &self.geom.dims;
        if self.geom.paged {
            vec![self.geom.nb, d.hkv, d.hd, self.geom.page]
        } else {
            vec![self.geom.nb, d.hkv, d.hd]
        }
    }

    /// 主 KV 池 K 叶子形状。**含 padding dump 页**(块尺寸 = 形状元素数;
    /// 盘外页只作 bucket padding 假行的槽落点,永不进页链 ⇒ 绝不被读)。
    /// ⚠️ MTP/草稿池用 [`StatePool::k_shape`](本函数不含 dump 页:那两个池单独分配,没多页)。
    pub(crate) fn k_shape_kv(&self) -> Vec<usize> {
        let d = &self.geom.dims;
        if self.geom.paged {
            vec![self.geom.nb_kv, d.hkv, d.hd / self.geom.x, self.geom.page, self.geom.x]
        } else {
            vec![self.geom.nb_kv, d.hkv, d.hd]
        }
    }

    pub(crate) fn v_shape_kv(&self) -> Vec<usize> {
        let d = &self.geom.dims;
        if self.geom.paged {
            vec![self.geom.nb_kv, d.hkv, d.hd, self.geom.page]
        } else {
            vec![self.geom.nb_kv, d.hkv, d.hd]
        }
    }

    /// padding dump 页的 KV 平面槽基址(槽号 = 页号 × page;None = 无桶档)
    /// 逐层 KV cache 叶子句柄((k, v);slots/kv_lens/block_tables 由
    /// 调用方按步注入 —— 图闭包注输入槽,prefill 注 from_host 标量行)
    pub(crate) fn kv_caches(&self) -> Vec<(TensorOps, TensorOps)> {
        // B6.2:fp8 池 = U32 扁平字节账叶(指针面;读核按元素序自寻址,
        // FI 影子池 fp8 档同款)
        if self.geom.kv_fp8 {
            let shape = vec![self.geom.nb_kv * self.geom.dims.hkv * self.geom.dims.hd * self.geom.page / 4];
            return self.kv.kvs
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
        self.kv.kvs
            .iter()
            .map(|b| {
                (
                    block_leaf_dt(&b.k_cache.0, k_shape.clone(), self.geom.dims.dtype),
                    block_leaf_dt(&b.v_cache.0, v_shape.clone(), self.geom.dims.dtype),
                )
            })
            .collect()
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
    ) -> Result<Vec<String>> {
        let nl = self.gdn.gdns.len();
        let mut out = Vec::new();
        for (li, g) in self.gdn.gdns.iter().enumerate() {
            let last = li + 1 == nl;
            // 首末层 cv+rec(全层扫档 gdn_dump_all 已退役,2026-10-27)
            if li != 0 && !last {
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
        let row = bn.1 / self.geom.gdn_slots;
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
        let d = &self.geom.dims;
        self.gdn.gdns
            .iter()
            .map(|g| GdnLeaf {
                conv_q: block_leaf(&g.conv_q.0, vec![self.geom.gdn_slots, d.nk * d.hk, 3]),
                conv_k: block_leaf(&g.conv_k.0, vec![self.geom.gdn_slots, d.nv * d.hv, 3]),
                conv_v: block_leaf(&g.conv_v.0, vec![self.geom.gdn_slots, d.nv * d.hv, 3]),
                rec: block_leaf(&g.rec.0, vec![self.geom.gdn_slots, d.nv, d.hk, d.hv]),
            })
            .collect()
    }

    /// 逐层 KV cache K 的**块句柄**(门/诊断:按槽偏移 D2H 读 K 字节)。
    /// 生产路径不读,只服务于机械门与取证工具。
    #[allow(dead_code)]
    pub(crate) fn kv_k_blocks(&self) -> Vec<owl_iface::contract::Bytes> {
        self.kvs().iter().map(|b| b.k_cache.0.clone()).collect()
    }

    /// 块表叶子(模式相关:paged [1,nb] / legacy [1];run 图闭包用)
    pub(crate) fn bt_leaf(&self) -> TensorOps {
        if self.geom.paged {
            TensorOps::of_block(self.kv.bt.id, Dtype::F32, vec![1, self.geom.nb])
        } else {
            TensorOps::of_block(self.kv.bt.id, Dtype::F32, vec![1])
        }
    }

    /// 块表叶子(恒 [1,nb];prefill_chunk 既有形状,legacy nb=1 → [1,1])
    pub(crate) fn bt_leaf_flat(&self) -> TensorOps {
        TensorOps::of_block(self.kv.bt.id, Dtype::F32, vec![1, self.geom.nb])
    }

    /// 快照存在性(调度面复用边界判定:无快照的边界不可复用,GDN 态
    /// 对不上)
    /// 快照池容量(`snap_max`;0 = 快照面整体关闭 —— 调用方须跳过 capture,
    /// 否则 `capture_snap` 会因池空而报错,见 agent 专项 A2)
    pub(crate) fn snap_capacity(&self) -> usize {
        self.gdn.snaps.keys.len()
    }

    /// 快照池已用槽数(诊断/报告面:设备上在场多少张边界快照)。
    pub(crate) fn snap_used(&self) -> usize {
        self.gdn.snaps.keys.iter().filter(|k| k.is_some()).count()
    }


    /// 快照在设备槽上的位置(`None` = 不在设备层)。
    pub(crate) fn snap_slot_of(&self, key: SnapKey) -> Option<u32> {
        self.gdn.snaps
            .keys
            .iter()
            .position(|k| *k == Some(key))
            .map(|i| i as u32)
    }

    /// **槽位选择**(纯;S2 §25.2):同 key 已在设备 ⇒ 原地;否则空槽;
    /// 否则按**store 给的出序**取第一个仍在槽上的快照(§30.6-②)。
    ///
    /// `order` = [`owl_store::Rel::snap_victim_order`] 的结果(该走的先后;
    /// 事实档 + `EvictPolicy`,与容量调度**同一份判据**)。本函数只做
    /// **key→槽的物理映射**——判据不再在池里(旧实现自带 `rank + LRU` = 第二套规范)。
    ///
    /// 兜底:`order` 为空/全是已不在槽的键(判与选之间被别处移走)⇒ 池内 LRU
    /// (只按 `last_use`;这是"物理映射的退化",不是另一套语义档)。
    pub(crate) fn snap_pick_slot(
        &self,
        key: SnapKey,
        order: &[SnapKey],
    ) -> Option<SnapSlotPick> {
        if self.gdn.snaps.keys.is_empty() {
            return None;
        }
        if let Some(i) = self.gdn.snaps.keys.iter().position(|k| *k == Some(key)) {
            return Some(SnapSlotPick::Hit(i as u32));
        }
        if let Some(i) = self.gdn.snaps.keys.iter().position(|k| k.is_none()) {
            return Some(SnapSlotPick::Free(i as u32));
        }
        for want in order {
            if let Some(i) = self.gdn.snaps.keys.iter().position(|k| *k == Some(*want)) {
                return Some(SnapSlotPick::Victim(i as u32, *want));
            }
        }
        let (i, _) = self.gdn.snaps
            .keys
            .iter()
            .enumerate()
            .filter_map(|(i, k)| k.map(|k| (i, k)))
            .min_by_key(|(i, _)| self.gdn.snaps.last_use[*i])?;
        Some(SnapSlotPick::Victim(i as u32, self.gdn.snaps.keys[i]?))
    }

    /// 片规格(逐 (层,quad);槽内偏移由 store 的 `plan_snap_*` 按
    /// `dev_off = 槽 × 行宽` 算 —— 布局本身与槽无关)。
    pub(crate) fn snap_pieces_at(&self, _slot: u32) -> Vec<owl_store::PieceSpec> {
        self.gdn.snaps
            .bufs
            .iter()
            .zip(&self.gdn.snaps.row_bytes)
            .map(|(b, rb)| owl_store::PieceSpec {
                buf: b.0.clone(),
                stride_bytes: *rb,
            })
            .collect()
    }

    /// **状态腿设备布局**(boot 注册给 store;§25.2):快照 = 一种"块"
    /// (槽 × 片),片 = 每 (层,quad) 一行宽。
    pub(crate) fn snap_pool_layout(&self, epoch: u64) -> owl_store::PoolLayout {
        owl_store::PoolLayout {
            class: owl_store::BlockClass::GdnRow,
            epoch,
            pieces: self.snap_pieces_at(0),
        }
    }

    /// 一条快照的字节(Σ 行宽;布局真源)。
    pub(crate) fn snap_bytes(&self) -> usize {
        self.gdn.snaps.row_bytes.iter().sum()
    }

    /// 标记槽上放的是哪张快照(拍摄/回灌落位后)。
    pub(crate) fn snap_place(&mut self, slot: u32, key: SnapKey) {
        self.gdn.snap_tick += 1;
        let tick = self.gdn.snap_tick;
        let i = slot as usize;
        if i < self.gdn.snaps.keys.len() {
            self.gdn.snaps.keys[i] = Some(key);
            self.gdn.snaps.last_use[i] = tick;
        }
    }

    /// **快照键重挂**(前缀无感删除;2026-10-23):槽上那条快照换身份。
    ///
    /// 只动账 —— 内容零搬运(删除是"搬运记忆",状态本身不必重算)。
    /// `false` = 槽上没有旧键(账/槽不平;调用方查账已确认,属结构性异常)。
    pub(crate) fn snap_rekey(&mut self, old: SnapKey, new: SnapKey) -> bool {
        match self.gdn.snaps.keys.iter().position(|k| *k == Some(old)) {
            Some(i) => {
                self.gdn.snaps.keys[i] = Some(new);
                true
            }
            None => false,
        }
    }

    /// 清槽(快照下沉到宿主后:设备缓冲空出来了)。
    pub(crate) fn snap_clear_slot(&mut self, slot: u32) {
        let i = slot as usize;
        if i < self.gdn.snaps.keys.len() {
            self.gdn.snaps.keys[i] = None;
            self.gdn.snaps.last_use[i] = 0;
        }
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
        for g in &self.gdn.gdns {
            let blocks = [
                (&g.conv_q.0, g.conv_q.1),
                (&g.conv_k.0, g.conv_k.1),
                (&g.conv_v.0, g.conv_v.1),
                (&g.rec.0, g.rec.1),
            ];
            for (bn, elems) in blocks {
                let row = elems / self.geom.gdn_slots;
                face.memset_zero_at(bn, gdn_slot * row * 4, row * 4).await?;
            }
        }
        face.sync().await?;
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
        for g in &self.gdn.gdns {
            for (bn, elems) in [
                (&g.conv_q.0, g.conv_q.1),
                (&g.conv_k.0, g.conv_k.1),
                (&g.conv_v.0, g.conv_v.1),
                (&g.rec.0, g.rec.1),
            ] {
                let row = elems / self.geom.gdn_slots;
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
        for g in &self.gdn.gdns {
            for (bn, elems) in [
                (&g.conv_q.0, g.conv_q.1),
                (&g.conv_k.0, g.conv_k.1),
                (&g.conv_v.0, g.conv_v.1),
                (&g.rec.0, g.rec.1),
            ] {
                let row = elems / self.geom.gdn_slots;
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

    /// `slot` = 已选好的设备槽(`None` = 池关 ⇒ 结构性拒);槽由 engine 选
    /// (`snap_pick_slot`:原地 / 空槽 / 最冷下沉)。
    pub(crate) async fn capture_snap<D: DeviceClient>(
        &mut self,
        face: &mut D,
        gdn_slot: usize,
        key: SnapKey,
        slot: Option<u32>,
    ) -> Result<()> {
        // 池空(snap_max = 0)不再是 panic(原 `expect("池非空")` 会炸掉整个
        // owl-engine 线程 → 客户端收 503):退化为**结构化错误** —— 调用方
        // (`capture_gdn_snap`)已按 snap_capacity() 先行跳过,故此处只作兜底
        // (agent 专项 A2,2026-10-14)。
        // 槽位由调用方选好(engine:空槽 / 最冷下沉 / 原地覆盖)—— 见
        // `snap_pick_slot`;此处只认已定的槽(池关 = 结构性拒)。
        let Some(slot) = slot else {
            return Err(ModelError::Msg(
                "GDN 快照池为空(snap_max = 0):调用方应跳过 capture \u{2014}\u{2014} \
                 见 roadmap.local/agent-服务面-剩余工作项 A2"
                    .into(),
            ));
        };
        let idx = slot as usize;
        // **P1(§20-A2)**:一条 `copy_batch` 搬完(逐片 await = 0.8B 72 次 /
        // 27B 192 次 actor 往返 ⇒ 一次命令;与 `capture_spec_snap` 同构)。
        let mut copies: Vec<(owl_iface::contract::Bytes, usize, owl_iface::contract::Bytes, usize, usize)> =
            Vec::with_capacity(self.gdn.gdns.len() * 4);
        for (li, g) in self.gdn.gdns.iter().enumerate() {
            let quads = [
                (&g.conv_q.0, g.conv_q.1),
                (&g.conv_k.0, g.conv_k.1),
                (&g.conv_v.0, g.conv_v.1),
                (&g.rec.0, g.rec.1),
            ];
            for (j, (src, elems)) in quads.iter().enumerate() {
                let row = elems / self.geom.gdn_slots;
                let dst = &self.gdn.snaps.bufs[li * 4 + j];
                copies.push(((*src).clone(), gdn_slot * row * 4, dst.0.clone(), idx * row * 4, row * 4));
            }
        }
        face.copy_batch(copies.as_slice()).await?;
        self.snap_place(slot, key);
        Ok(())
    }

    /// GDN 快照恢复(E2c;前缀命中时调用):槽位缓冲 → 会话格(D2D
    /// 反向)。同流保序,后续 prefill 天然在恢复态之后。
    /// `key` = 边界块内容身份(与拍摄侧同类型 ⇒ 槽号进不来)。
    /// `slot` = 快照**已在设备**的槽(engine 先按 store 的层账把宿主层快照
    /// 回灌到设备再调这里 —— §25.2)。
    pub(crate) async fn restore_snap<D: DeviceClient>(
        &mut self,
        face: &mut D,
        gdn_slot: usize,
        key: SnapKey,
        slot: u32,
    ) -> Result<()> {
        let idx = slot as usize;
        if self.gdn.snaps.keys.get(idx).copied().flatten() != Some(key) {
            return Err(ModelError::Msg(format!(
                "快照 {key:?} 不在设备槽 {slot}(先回灌)"
            )));
        }
        // **P1**:一条 `copy_batch` 搬回(与拍摄侧同构)
        let mut copies: Vec<(owl_iface::contract::Bytes, usize, owl_iface::contract::Bytes, usize, usize)> =
            Vec::with_capacity(self.gdn.gdns.len() * 4);
        for (li, g) in self.gdn.gdns.iter().enumerate() {
            let quads = [
                (&g.conv_q.0, g.conv_q.1),
                (&g.conv_k.0, g.conv_k.1),
                (&g.conv_v.0, g.conv_v.1),
                (&g.rec.0, g.rec.1),
            ];
            for (j, (dst, elems)) in quads.iter().enumerate() {
                let row = elems / self.geom.gdn_slots;
                let src = &self.gdn.snaps.bufs[li * 4 + j];
                copies.push((src.0.clone(), idx * row * 4, (*dst).clone(), gdn_slot * row * 4, row * 4));
            }
        }
        face.copy_batch(copies.as_slice()).await?;
        self.gdn.snap_tick += 1;
        let tick = self.gdn.snap_tick;
        if idx < self.gdn.snaps.last_use.len() {
            self.gdn.snaps.last_use[idx] = tick;
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
        let n = self.gdn.gdns.len();
        let snap = self.draft.spec_snap
            .as_mut()
            .ok_or_else(|| ModelError::Msg("spec 快照缓冲未分配(OWL_SPEC_DEPTH 未开)".into()))?;
        // E5-M4:批量通道(192 次 actor 往返 → 1;单 ack 保序)
        let mut copies: Vec<(owl_iface::Bytes, usize, owl_iface::Bytes, usize, usize)> = Vec::with_capacity(n * 4);
        for (li, g) in self.gdn.gdns.iter().enumerate() {
            let quads = [
                (&g.conv_q.0, g.conv_q.1),
                (&g.conv_k.0, g.conv_k.1),
                (&g.conv_v.0, g.conv_v.1),
                (&g.rec.0, g.rec.1),
            ];
            for (j, (src, elems)) in quads.iter().enumerate() {
                let row = elems / self.geom.gdn_slots;
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
        let n = self.gdn.gdns.len();
        let snap = self.draft.spec_snap
            .as_ref()
            .ok_or_else(|| ModelError::Msg("spec 快照缓冲未分配".into()))?;
        // E5-M4:批量通道(同 capture)
        let mut copies: Vec<(owl_iface::Bytes, usize, owl_iface::Bytes, usize, usize)> = Vec::with_capacity(n * 4);
        for (li, g) in self.gdn.gdns.iter().enumerate() {
            let quads = [
                (&g.conv_q.0, g.conv_q.1),
                (&g.conv_k.0, g.conv_k.1),
                (&g.conv_v.0, g.conv_v.1),
                (&g.rec.0, g.rec.1),
            ];
            for (j, (dst, elems)) in quads.iter().enumerate() {
                let row = elems / self.geom.gdn_slots;
                let src = &snap.bufs[li * 4 + j];
                copies.push((src.0.clone(), 0usize, (*dst).clone(), gdn_slot * row * 4, row * 4));
            }
        }
        face.copy_batch(copies.as_slice()).await?;
        let _ = n;
        Ok(())
    }

    /// **最终 query replay**:会话格 GDN → 专用 replay 缓冲。
    pub(crate) async fn capture_replay_snap<D: DeviceClient>(
        &mut self,
        face: &mut D,
        gdn_slot: usize,
    ) -> Result<()> {
        let n = self.gdn.gdns.len();
        let snap = self.draft.replay_snap.as_mut().ok_or_else(|| {
            ModelError::Msg("replay 快照缓冲未分配([kvmem].query_replay 未开?)".into())
        })?;
        let mut copies: Vec<(owl_iface::Bytes, usize, owl_iface::Bytes, usize, usize)> =
            Vec::with_capacity(n * 4);
        for (li, g) in self.gdn.gdns.iter().enumerate() {
            let quads = [
                (&g.conv_q.0, g.conv_q.1),
                (&g.conv_k.0, g.conv_k.1),
                (&g.conv_v.0, g.conv_v.1),
                (&g.rec.0, g.rec.1),
            ];
            for (j, (src, elems)) in quads.iter().enumerate() {
                let row = elems / self.geom.gdn_slots;
                let dst = &snap.bufs[li * 4 + j];
                copies.push(((*src).clone(), gdn_slot * row * 4, dst.0.clone(), 0usize, row * 4));
            }
        }
        face.copy_batch(copies.as_slice()).await?;
        Ok(())
    }

    /// **最终 query replay**:replay 缓冲 → 会话格 GDN(反向 D2D)。
    pub(crate) async fn restore_replay_snap<D: DeviceClient>(
        &mut self,
        face: &mut D,
        gdn_slot: usize,
    ) -> Result<()> {
        let n = self.gdn.gdns.len();
        let snap = self.draft.replay_snap.as_ref().ok_or_else(|| {
            ModelError::Msg("replay 快照缓冲未分配([kvmem].query_replay 未开?)".into())
        })?;
        let mut copies: Vec<(owl_iface::Bytes, usize, owl_iface::Bytes, usize, usize)> =
            Vec::with_capacity(n * 4);
        for (li, g) in self.gdn.gdns.iter().enumerate() {
            let quads = [
                (&g.conv_q.0, g.conv_q.1),
                (&g.conv_k.0, g.conv_k.1),
                (&g.conv_v.0, g.conv_v.1),
                (&g.rec.0, g.rec.1),
            ];
            for (j, (dst, elems)) in quads.iter().enumerate() {
                let row = elems / self.geom.gdn_slots;
                let src = &snap.bufs[li * 4 + j];
                copies.push((src.0.clone(), 0usize, (*dst).clone(), gdn_slot * row * 4, row * 4));
            }
        }
        face.copy_batch(copies.as_slice()).await?;
        Ok(())
    }

    /// 活跃会话块表 → 持久 bt 块(设备)。图烘焙的是 bt 指针,内容随
    /// 会话切换/块链增长重写(指针稳定 = 图不重捕);f32 过线(契约 5)
    pub(crate) async fn write_bt<D: DeviceClient>(
        &self,
        face: &mut D,
        table: &[u32],
    ) -> Result<()> {
        // **占位符守卫**(B-3e-2):表里含宿主占位符(u32::MAX)⇒ 写进设备 bt 块
        // 就是把 u32::MAX 当页号喂核 = ILLEGAL_ADDRESS(27B Stage2 实测同类)。
        // 部分驻留下来的洞**只能**由视图页表表达 ⇒ 这里响亮拒(调用方应先在
        // 段边界回灌,或改用视图路径)。
        if let Some(i) = table.iter().position(|b| owl_store::is_host_placeholder(*b)) {
            return Err(ModelError::Msg(format!(
                "write_bt:块表第 {i} 项是宿主占位符(部分驻留的链洞)—— 不得写设备块表;                 请先回灌(段边界)或走视图页表"
            )));
        }
        let mut v = vec![0f32; self.geom.nb];
        for (i, b) in table.iter().enumerate() {
            v[i] = *b as f32;
        }
        face.write_block_f32_on(owl_iface::contract::UpStream::Compute, &self.kv.bt, 0, &v)
            .await
    }

    // ── B-2 选择层视图面(§8.11)──────────────────────────────────────

    /// 视图页表容量(块数;= 池管理面页数 `nb`)。
    pub(crate) fn select_view_capacity(&self) -> usize {
        self.geom.nb
    }

    /// 写视图页表(**选中块按时间序**的物理页;零填充到 `nb`)。
    ///
    /// 超容量 = **响亮拒**(调用方必须先按预算裁到 `nb` 以内 —— 静默截断 =
    /// 静默漏块,选择层的账就假了)。
    pub(crate) async fn write_select_view<D: DeviceClient>(
        &self,
        face: &mut D,
        pages: &[u32],
    ) -> Result<()> {
        let b = self.kv.bt_view.as_ref().ok_or_else(|| {
            ModelError::Msg("选择层视图块表未分配(OWL_COVERAGE_PROBE 未开?)".into())
        })?;
        if pages.len() > self.geom.nb {
            return Err(ModelError::Msg(format!(
                "视图页表超容量:{} 页 > 槽容量 {}(预算须先裁)",
                pages.len(),
                self.geom.nb
            )));
        }
        let mut v = vec![0f32; self.geom.nb];
        for (i, p) in pages.iter().enumerate() {
            v[i] = *p as f32;
        }
        face.write_block_f32(b, 0, &v).await
    }

    /// 视图页表叶子(`[1, nb]` f32;喂 `KvBuffers::block_tables`)。
    pub(crate) fn select_view_leaf(&self) -> Option<TensorOps> {
        self.kv.bt_view
            .as_ref()
            .map(|b| TensorOps::of_block(b.id, Dtype::F32, vec![1, self.geom.nb]))
    }
}

/// **池装配上下文**(S5b):决策/预算/几何的产物 —— 纯 host,零设备。
///
/// 每个 bank 的 `alloc` 只从这里读"分多少 / 什么 dtype / 哪档",不再各自推导
/// (原实现里 `nb`/`page`/`paged` 在多处重复推导,新增一档就要同步多处)。
struct AllocCtx<'a> {
    plan: &'a PoolPlan,
    geom: PoolGeom,
    /// KV 池元素 dtype(量化枚举的单一映射)
    kv_dt: Dtype,
    kv_bytes_elem: usize,
    n_full: usize,
    n_gdn: usize,
    seq_tokens: usize,
    snap_max: usize,
}

impl<'a> AllocCtx<'a> {
    /// **决策 + 预算 + 几何**(原 `alloc` 的前半;无任何设备调用)。
    fn build(dims: ModelDims, layer_types: &[bool], plan: &'a PoolPlan) -> Result<Self> {
        let seq_tokens = plan.seq_tokens;
        // 布尔和稀泥废除(2026-10-10):量化 = KvQuant 枚举,投机形态 =
        // SpecMode 枚举(快照/MTP 链/草稿池三件由 mode 派生,互斥不再靠
        // 调用点自洽)
        // 格式从枚举**单一映射**出来(2026-10-16:禁"bool 传格式");
        // 池的**存储**恒为 U32 扁平字节账(元素数 = 字节/4,见下)。
        // 池**元素 dtype** 与清单门同源(E1 2026-10-17:`kv_pool_dtype`:
        // fp8 ⇒ 配置点名的格式;否则 ⇒ 模型 dtype —— 原先此处自己写
        // `if kv_bytes_elem == 1 {…} else { dims.dtype }`,与门各说各话)
        let kv_dt = owl_models::kernel::kv_pool_dtype(plan.kv, dims.dtype);
        let kv_bytes_elem = kv_dt.size_bytes();
        let kv_fp8 = kv_bytes_elem == 1;
        // 未接臂的格式**响亮拒**(e5m2 暂无核臂;不许静默按 e4m3 跑)
        for (tag, q) in [("kv", plan.kv), ("draft_kv", plan.draft_kv)] {
            if q == owl_models::env::KvQuant::Fp8E5M2 {
                return Err(owl_iface::contract::ModelError::Msg(format!(
                    "{tag} = fp8-e5m2:本仓尚无 e5m2 核臂(只有 e4m3);请改 kv_fp8 档或等臂接入"
                )));
            }
        }
        let dflash = plan.spec_mode == crate::spec_runtime::SpecMode::DFlash2;
        let gdn_slots = plan.gdn_slots;
        let snap_max = plan.snap_max;
        let n_full = layer_types.iter().filter(|&&f| f).count();
        let n_gdn = layer_types.len() - n_full;
        // 页宽策略只推导一次（FixedAfter 的 spec/replay 缓冲与后面的池几何共用）
        let pol = kv_paged_policy(dims.dtype);
        let paged = pol.is_some();
        owl_shared::log_info!("boot", "StatePool::alloc 进入(n_full={n_full} kv_fp8={kv_fp8})");

        // B6.5 显存硬预算治理(2026-10-10):池容量 =
        // min(手工上限, (target×total − live − reserve)/每 token 字节)。
        // target 默认 0.97;reserve 默认 1024MB(盖 CUDA ctx ~430MB + 逐 turn 瞬态 +
        // async 池惰性保留;实测 300MB 在连续 turn 下瞬态 OOM)
        // + 捕获瞬态 + 非追踪惰性保留。OWL_VRAM_TARGET/OWL_VRAM_RESERVE_MB
        // 可调;TOTAL 未落账(=0)或 OWL_VRAM_TARGET=0 → 治理禁用。
        // 口径与项来源见 [`PerTokBytes`] / [`FixedAfter`](B1,2026-10-17 具名化)。
        // 实测对账(spec-dflash 生产档,2026-10-17):per_tok = 32768(主池 fp8)
        // + 20480(草稿池 bf16) = **53248B**;fixed_after 里图项实测 **261MiB**、
        // 非追踪实测 = 进程占用 20096MiB − 引擎账 19552MiB = **544MiB**(> reserve 512MB,
        // 即 reserve 量级正确但零余量 ⇒ 见 graph_fleet 的对账行)。
        let mtp_mode = plan.spec_mode == crate::spec_runtime::SpecMode::Mtp;
        let per_tok = PerTokBytes::of(
            &dims,
            n_full,
            kv_bytes_elem,
            // **DFlash2 草稿池不再按 token 线性**(2026-10-22 ring):
            // 草稿 config 是 sliding_window=2048;池只留窗口+自块,固定项进
            // `FixedAfter.draft_ring`(见下)。MTP 链仍按 token 线性。
            None,
            plan.draft_geom,
            mtp_mode.then(|| plan.draft_kv.size_bytes()),
        );
        // 治理器之后的固定分配(GDN 格/快照 f32 同式;图 slab + warmup + 草稿 ring)
        let gdn_slot_elems = dims.nk * dims.hk * 3 + dims.nv * dims.hv * 3 + dims.nv * dims.hk * dims.hv;
        let page_for_draft = kv_paged_policy(dims.dtype).map(|p| p.page).unwrap_or(32);
        let draft_ring = if dflash {
            let geom = plan
                .draft_geom
                .map(|(l, h, d)| DraftPoolGeom::from_geom(l, h, d))
                .unwrap_or(DRAFT_POOL);
            owl_models::layers::dflash2::draft_ring_slots(page_for_draft)
                * geom.per_tok_bytes(plan.draft_kv.size_bytes())
        } else {
            0
        };
        let fixed_after = FixedAfter::of(
            n_gdn,
            gdn_slots,
            gdn_slot_elems,
            snap_max,
            draft_ring,
            paged && plan.spec_mode != crate::spec_runtime::SpecMode::Off,
            paged && plan.query_replay,
        );
        let pool_cap = pool_cap(plan, &per_tok, &fixed_after);
        // ── 池容量:**手写值** 或 **自动 = 本档 ctx**(2026-10-21)──
        //
        // 自动档把"池多大"从配置里拿掉:池只为 `max_seq` 分配(页对齐),
        // 显存治理器的 `cap` 只做两件事 —— ① 兜底:`max_seq` 超预算即拒启;
        // ② 报告:`ctx_max = cap − 1 页`(Strict 准入 = 整 ctx 常驻)。
        let page_guess = 32usize; // 与 `kv_paged_policy` 同值(下面会再取权威值)
        let ctx_max = pool_cap.map(|c| c.saturating_sub(page_guess));
        let pool_tokens = if plan.pool_auto {
            match ctx_max {
                Some(m) if plan.seq_tokens > m => {
                    // **超卖档(锁①,2026-10-27 修)**:`[kvmem].pool_overcommit` 开的语义就是
                    // "池 < 上下文" —— 多出的链块靠选择层下沉 host/disk。旧实现无条件拒启
                    // (2026-10-21 自动池重构漏了这条门),把超卖档挡在门外。
                    if !plan.pool_overcommit {
                        return Err(ModelError::Msg(format!(
                            "[pool] 自动池装不下本档:max_seq={}(+页对齐) > 预算可服务上限 ctx_max={}。\
                             本档 per_tok={}B ⇒ cap={:?} tok。修法:降 [runtime].max_seq 到 ≤ {}、\
                             或降 per_tok(池 fp8 / 减草稿池)、或开 [kvmem] 超卖(select_decode+select_partial+pool_overcommit)。",
                            plan.seq_tokens,
                            m,
                            per_tok.total(),
                            pool_cap,
                            m
                        )));
                    }
                    // 预填**瞬态**余量:超卖档池一旦吃满 cap,chunk 的激活/工作区
                    // (a8 实测 24-68 MiB,2048 块)+ 选择层视图 + 图 warmup 就得从
                    // 池外的空里挤 ⇒ 留 1 GiB(实测定值:512 MiB 在 27B/a8/视觉档
                    // 会撞 `vram-hard-budget` 拒发)。
                    let transient_tok = ((1024usize << 20) / per_tok.total().max(1)).max(1);
                    let pool = m.saturating_sub(transient_tok).max(1);
                    owl_shared::log_info!(
                        "pool",
                        "**超卖档**:max_seq={} > ctx_max={} ⇒ 设备池 = cap − 瞬态余量(512 MiB = {} tok) = {} tok(超卖 {:.2}×);\
                         其余链块靠选择层下沉 host/disk",
                        plan.seq_tokens,
                        m,
                        transient_tok,
                        pool,
                        plan.seq_tokens as f64 / pool as f64
                    );
                    pool
                }
                Some(m) => plan.seq_tokens.min(m),
                None => plan.seq_tokens,
            }
        } else {
            match pool_cap {
                Some(cap) => plan.pool_tokens.min(cap),
                None => plan.pool_tokens,
            }
        };

        // ── **[ctx-max] 报告**(用户令:最大支持多少要直接显示)──
        //
        // 口径:`cap` = 预算允许的**池 token 数**(治理禁用时 None);
        // `ctx_max` = 可服务的最长「prompt+生成」(Strict 准入 ⇒ 整 ctx 常驻,
        // 扣 1 页对齐余量);`pool` = 本档实际分配的池。
        match (pool_cap, ctx_max) {
            (Some(cap), Some(m)) => {
                let pool_mib = pool_tokens as u64 * per_tok.total() as u64 / (1 << 20);
                let pct = 100.0 * plan.seq_tokens as f64 / m.max(1) as f64;
                eprintln!(
                    "[ctx-max] per_tok={}B(主池 {} + 草稿 {} + MTP 链 {}) ⇒ 预算 cap={} tok \
                     ⇒ **本档最大 ctx ≈ {} tok**;当前 max_seq={}(占 {:.0}%)⇒ 池 {} tok / {} MiB{}",
                    per_tok.total(),
                    per_tok.main_kv,
                    per_tok.draft_pool,
                    per_tok.mtp_chain,
                    cap,
                    m,
                    plan.seq_tokens,
                    pct,
                    pool_tokens,
                    pool_mib,
                    if plan.pool_auto { "(自动)" } else { "(手写)" }
                );
                if plan.seq_tokens + page_guess > m {
                    eprintln!(
                        "[ctx-max] ⚠️ max_seq 已到算式上限(剩 {} tok;再大要降 per_tok 或加卡)",
                        m.saturating_sub(plan.seq_tokens)
                    );
                }
            }
            _ => eprintln!(
                "[ctx-max] 显存治理禁用(vram_target=0 或 total 未知)⇒ 池按 max_seq={} tok 分配(自动档无上限校验)",
                plan.seq_tokens
            ),
        }
        // 单会话容量(页数)= 旧实现的池下限;`pool_overcommit` 关时仍按它钉住
        let seq_blocks = seq_tokens.div_ceil(pol.as_ref().map_or(1, |p| p.page).max(1));
        let (page, x, nb) = match &pol {
            Some(p) => (p.page, p.x, seq_tokens.div_ceil(p.page)),
            None => (seq_tokens, 1usize, 1usize),
        };
        // E2b 块池容量:pool_tokens 向上取整到页。
        //
        // **锁①(2026-10-19)**:原实现是无条件 `.max(nb)` —— 池被钉在
        // **单会话全量上下文**上,所以选择层只省访问面、不省显存。现在按
        // `plan.pool_overcommit` 二选一:
        // - 关(缺省)= 旧行为(`max`),高显存档与现状逐位不变;
        // - 开 = **允许池小于上下文** → 多出的链块必须靠选择层的下沉
        //   (占位符)挤到宿主;容量不足时 store 会 `Exhausted` 响亮拒
        //   (不静默落空槽)。
        // 两侧都保证不低于"本档必须常住的那部分":开档时靠 boot 的
        // "工作集装得下" 硬校验(见 engine 侧 `kvmem_boot_checks`)。
        let nb = if paged { pool_tokens.div_ceil(page) } else { nb };
        let nb = if plan.pool_overcommit { nb } else { nb.max(seq_blocks) };
        // **非超卖档的硬门**(2026-10-20):池必须装得下整条上下文。cap 是预算
        // 上限;若 cap < seq_tokens,说明本档在治理器预算下不可行 —— 不允许
        // 再靠 `nb.max(seq_blocks)` 静默超预算(旧行为:日志说 cap 小,实际池
        // 仍按 seq_tokens 分配)。显式拒,给出降 context / 开 overcommit 的出路。
        if let Some(cap) = pool_cap {
            let effective = nb.saturating_mul(page);
            if !plan.pool_overcommit
                && effective > cap.saturating_add(page.saturating_sub(1))
            {
                return Err(ModelError::Msg(format!(
                    "非超卖档池预算不足:seq_tokens={seq_tokens}(effective={effective} tok) > \
                     治理器 cap={cap} tok;本档装不下整条上下文。\
                     修法:降 [runtime].max_seq / 减少 fixed/激活,或显式开 [kvmem] 超卖。"
                )));
            }
        }
        let nb_kv = nb;

        // FI 面启用 = 能力条件(paged + f16 池),不是开关;与 tables 块同源
        let fi_face = plan.fi_quant.is_some();
        Ok(Self {
            plan,
            geom: PoolGeom {
                dims, page, nb, nb_kv, paged, gdn_slots, fi_face, kv_fp8,
                x,
            },
            kv_dt,
            kv_bytes_elem,
            n_full,
            n_gdn,
            seq_tokens,
            snap_max,
        })
    }
}

    /// **显存治理**:池 token 上限(None = 治理禁用)。
    ///
    /// 口径(2026-10-22 起):`base = min(free, free − total×(1−target))`
    /// (驱动实测 free;target 只作上限),`cap = (base − reserve − fixed_after)/per_tok`。
    /// 旧式用引擎账 `live` 近似,漏掉 CUDA ctx / FI ws / allocator 惰性 reserved
    /// (实测差 ~2-3 GiB)⇒ cap 虚高;`live` 仍打印但仅作参考。
    fn pool_cap(plan: &PoolPlan, per_tok: &PerTokBytes, fixed_after: &FixedAfter) -> Option<usize> {
        let per_tok_bytes = per_tok.total();
        let vram_reserve = plan.vram_reserve_mb << 20;
        let vram_total = plan.vram.total;
        // `vram_target = 0` = 治理禁用;**`1.0` 是合法档**(吃满总显存,`reserve`
        // 仍是唯一缓冲)—— 旧条件写 `target < 1.0` 把 1.0 当"禁用" ⇒ 硬预算
        // 静默失效(2026-10-21 实测:`target=1.0` 的档 boot 里根本没有 cap 行,
        // 池怎么设都不校验,直到驱动 OOM)。
        if plan.vram_target > 0.0 && vram_total > 0 {
            // ★ **按卡账**(2026-10-24):本 rank 的 total/free/live 全部来自
            // `PoolPlan.vram`(设备 account 快照),不再有进程级 LIVE 摊分。
            let live_per_card = plan
                .live_baseline_per_card
                .unwrap_or(plan.vram.live);
            // **预算基 = 驱动 free**(2026-10-22):旧式
            // `total×target − live − fixed − reserve` 的 `live` 是**引擎账**,
            // 不含 CUDA ctx / FI workspace / allocator 惰性 reserved / 碎片
            // (实测差 ~2-3 GiB)⇒ cap 322k 但 262k 请求就撞 5/30MiB 瞬态。
            // `free` 是驱动实测(每卡;boot 已落账),它自动把这些未入账项算进去。
            // target 仍作**上限**(防别的进程刚释放时把池吃满):
            //   target_cap = free − total×(1−target)
            let free = plan.vram.free;
            let target_cap = free as i64
                - ((vram_total as f64 * (1.0 - plan.vram_target)) as i64);
            // 无 free 落账(测试/旧路)⇒ 回退旧口径
            let base = if free > 0 {
                (free as i64).min(target_cap.max(0))
            } else {
                (vram_total as f64 * plan.vram_target) as i64 - live_per_card as i64
            };
            let budget = base
                - vram_reserve as i64
                - fixed_after.total() as i64;
            let cap = (budget.max(0) as u64 / per_tok_bytes.max(1) as u64) as usize;
            // **A1 取证**(2026-10-19):治理器是"池能多大"的唯一产地;这条打出
            // 它的输入与产出(含主干 `FixedAfter`/`PerTokBytes` 的分解),免得再靠猜。
            // 用 `eprintln!`(取证行惯例):`log_*` 在无 `log` feature 的构建里被编译剔除,
            // 而本行必须在探针/测试里可见。
            eprintln!(
                "[vram-gov] total={:.2}G free={:.2}G target={:.0}% reserve={}M fixed_after={:.2}G[{}] live={:.2}G(引擎账;仅参考;ranks={}) per_tok={}B[{}] ⇒ budget={:.2}G cap={} tok(manual={})",
                vram_total as f64 / 1073741824.0,
                free as f64 / 1073741824.0,
                plan.vram_target * 100.0,
                vram_reserve >> 20,
                fixed_after.total() as f64 / 1073741824.0,
                fixed_after.explain(),
                live_per_card as f64 / 1073741824.0,
                plan.tp_ranks.max(1),
                per_tok_bytes,
                per_tok.explain(),
                budget as f64 / 1073741824.0,
                cap,
                plan.pool_tokens,
            );
            Some(cap)
        } else {
            None
        }
    }


impl KvBank {
    /// 主 KV 池 + 页表 + attention scratch + 表槽。
    ///
    /// 四段各自成函数(S5b):每段 = 一类资源 + 它的分配判据,不再一竿到底。
    async fn alloc<D: DeviceClient>(face: &mut D, ctx: &AllocCtx<'_>) -> Result<Self> {
        let kvs = Self::alloc_pages(face, ctx).await?;
        let (bt, bt_view, gdn_tables, fi_tables) = Self::alloc_tables(face, ctx).await?;
        let (attn_v2, verify_attn_v2, batch_attn_v2, bt_batch) =
            Self::alloc_scratch(face, ctx).await?;
        let xqa = Self::alloc_xqa(face, ctx).await?;
        Ok(KvBank {
            kvs,
            bt,
            bt_view,
            bt_batch,
            attn_v2,
            verify_attn_v2,
            batch_attn_v2,
            xqa,
            gdn_tables,
            fi_tables,
        })
    }

    /// 主 KV 池(每 full-attention 层一对 K/V;fp8 档 = U32 字节承载)。
    async fn alloc_pages<D: DeviceClient>(
        face: &mut D,
        ctx: &AllocCtx<'_>,
    ) -> Result<Vec<KvBlocks>> {
        let dims = ctx.geom.dims;
        let page = ctx.geom.page;
        let nb_kv = ctx.geom.nb_kv;
        let n_full = ctx.n_full;
        let kv_dt = ctx.kv_dt;
        let kv_bytes_elem = ctx.kv_bytes_elem;
        let mut kvs: Vec<KvBlocks> = Vec::new();
        for l in 0..n_full {
            // B6.2:fp8 池 = U32 字节承载 1B/elem(元素数不变,块大小减半);
            // 读核 *_fp8 变体读入即转 half,写核 K0 转换写
            // 存储账:fp8 ⇒ U32 扁平字节账(字节数 = 元素数 ×1B,字数 = /4);
            // f16/bf16 ⇒ 元素账(2B/dims.dtype)。**由 KvQuant::size_bytes 单一决定**。
            let kv_elems = nb_kv * dims.hkv * dims.hd * page;
            let (k_elems, k_dt) = if kv_bytes_elem == 1 {
                (kv_elems / 4, Dtype::U32)
            } else {
                (kv_elems, kv_dt)
            };
            let (v_elems, v_dt) = (k_elems, k_dt);
            let k_cache = zero_block_dt(face, k_elems, k_dt, &format!("kv.k.{l}")).await?;
            let v_cache = zero_block_dt(face, v_elems, v_dt, &format!("kv.v.{l}")).await?;
            kvs.push(KvBlocks { k_cache, v_cache });
        }
        Ok(kvs)
    }

    /// 块表 + 两张表槽(bt / bt_view / gdn_tables / fi_tables)。
    ///
    /// 共同约束:**持久块 ⇒ 地址恒定**(图会把指针烘进去)。
    async fn alloc_tables<D: DeviceClient>(
        face: &mut D,
        ctx: &AllocCtx<'_>,
    ) -> Result<(
        Bytes,
        Option<Bytes>,
        Option<GdnTableBlocks>,
        Option<FiTableBlocks>,
    )> {
        let plan = ctx.plan;
        let nb = ctx.geom.nb;
        let paged = ctx.geom.paged;
        let n_gdn = ctx.n_gdn;
        let seq_tokens = ctx.seq_tokens;
        // 影子池已删(统一契约 P3:kNHD 单池,FI 直读主池叶子)
        // 块表持久块(E2b):内容 = 活跃会话块链,turn 切换/增长时重写
        // (write_bt;图烘焙 bt 指针,指针稳定图不重捕);legacy = 哑表
        let bt = if paged {
            eval_ops(TensorOps::zeros(Dtype::F32, vec![1, nb]).step(), face).await?
        } else {
            eval_ops(TensorOps::zeros(Dtype::F32, vec![1]).step(), face).await?
        };
        // **B-2 选择层视图块表槽**:与 bt 同族(持久块 ⇒ 将来图输入槽/烘焙
        // 指针稳定);内容由召回点每步重写(§8.11)
        let bt_view = if paged && plan.select_view {
            Some(eval_ops(TensorOps::zeros(Dtype::F32, vec![1, nb]).step(), face).await?)
        } else {
            None
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
        Ok((bt, bt_view, gdn_tables, fi_tables))
    }

    /// attention scratch 四件(decode / verify / 合批 v2 + 合批块表)。
    async fn alloc_scratch<D: DeviceClient>(
        face: &mut D,
        ctx: &AllocCtx<'_>,
    ) -> Result<(
        Option<owl_models::module::AttnV2Scratch>,
        Option<owl_models::module::AttnV2Scratch>,
        Option<owl_models::module::AttnV2Scratch>,
        Option<Bytes>,
    )> {
        let plan = ctx.plan;
        let dims = ctx.geom.dims;
        let page = ctx.geom.page;
        let nb = ctx.geom.nb;
        let paged = ctx.geom.paged;
        // v2 分页 attention scratch 三档(decode/verify/合批):同构,只差伪序列数
        // —— 分配判据归 `KvBank::alloc_attn_v2`(单源,含 PARTITION 常量)。
        let nparts = (nb * page).div_ceil(V2_PARTITION_TOKENS);
        let max_batch = plan.max_batch.max(1);
        let attn_v2 = if paged {
            Some(Self::alloc_attn_v2(face, dims, nparts, 1, "attn2").await?)
        } else {
            None
        };
        let verify_attn_v2 = if paged && plan.spec_depth > 0 {
            Some(Self::alloc_attn_v2(face, dims, nparts, plan.spec_depth + 1, "vattn2").await?)
        } else {
            None
        };
        let batch_attn_v2 = if paged && max_batch > 1 {
            Some(Self::alloc_attn_v2(face, dims, nparts, max_batch, "battn2").await?)
        } else {
            None
        };
        // 批页表块([max_batch, nb] f32;内容 = 各会话链,每步重写)
        let bt_batch = if paged && max_batch > 1 {
            Some(eval_ops(TensorOps::zeros(Dtype::F32, vec![max_batch, nb]).step(), face).await?)
        } else {
            None
        };

        Ok((attn_v2, verify_attn_v2, batch_attn_v2, bt_batch))
    }

    /// XQA 资源面(常量页表/掩码/semaphores/scratch;verify 与裸 decode 共用)。
    async fn alloc_xqa<D: DeviceClient>(
        face: &mut D,
        ctx: &AllocCtx<'_>,
    ) -> Result<Option<owl_models::module::XqaVerifyCtx>> {
        let plan = ctx.plan;
        let dims = ctx.geom.dims;
        let page = ctx.geom.page;
        let nb = ctx.geom.nb;
        let paged = ctx.geom.paged;
        // XQA 资源面(2026-10-13 D3 + ②;kNHD fp8 池 + grp6 + hd256 + page32)
        // **verify 与裸 decode 共用同一份**:恒定件(页表/掩码/semaphores/
        // scratch)两图都能用,只有 `seq_lens`/`semaphores` 需按图各挂一个
        // 输入槽(verify 图 vs decode 图各自的 `xqa_seq`/`xqa_sem`)。
        // 掩码:q=1 时只读 index 0,而 index 0 = `(1<<1)-1 = 1` —— 恰好是
        // decode 需要的"bit0 允许自身",**两形状共用同一张量**。
        // XQA 的 SM 数:**取设备真值**(2026-10-17 审查:原硬编码 84 = sm86 双卡,
        // 3080 档 = 68 ⇒ 启发式偏大)。查询不可用 = 回退常量 + 响亮告警(不静默)。
        let xqa_sm_count = match face.sm_count().await {
            Ok(v) => v,
            Err(e) => {
                owl_shared::log_warn!(
                    "boot",
                    "设备 SM 数查询不可用({e})⇒ XQA 取回退常量 {XQA_SM_COUNT_FALLBACK}(切分粒度可能偏大)"
                );
                XQA_SM_COUNT_FALLBACK
            }
        };
        // **XQA 能力契约**(S6h):几何前提集中成表,不满足则**响亮告知**
        // (原实现静默 `None` —— 换模型后 XQA 消失且无一行日志)。
        let xqa_want = plan.xqa_verify || plan.xqa_decode;
        if xqa_want {
            if let Err(why) = xqa_caps(&dims, page, plan.kv) {
                owl_shared::log_warn!(
                    "boot",
                    "XQA 面被**几何契约**拒绝(本档请求了 XQA,但 {why})—— 该面关闭,其余档位不受影响"
                );
            }
        }
        let xqa = if paged && xqa_want && xqa_caps(&dims, page, plan.kv).is_ok() {
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
                sm_count: xqa_sm_count,
            })
        } else {
            None
        };
        Ok(xqa)
    }
}

impl GdnBank {
    /// GDN 格状态 + 前缀快照池(仅非 full 层;快照池仅 paged 形态)。
    async fn alloc<D: DeviceClient>(face: &mut D, ctx: &AllocCtx<'_>) -> Result<Self> {
        let dims = ctx.geom.dims;
        let gdn_slots = ctx.geom.gdn_slots;
        let paged = ctx.geom.paged;
        let n_gdn = ctx.n_gdn;
        let snap_max = ctx.snap_max;
        let mut gdns: Vec<GdnBlocks> = Vec::new();
        for _ in 0..n_gdn {
            let conv_q = zero_block(face, gdn_slots * dims.nk * dims.hk * 3, "gdn.conv_q").await?;
            let conv_k = zero_block(face, gdn_slots * dims.nv * dims.hv * 3, "gdn.conv_k").await?;
            let conv_v = zero_block(face, gdn_slots * dims.nv * dims.hv * 3, "gdn.conv_v").await?;
            let rec = zero_block(face, gdn_slots * dims.nv * dims.hk * dims.hv, "gdn.recurrence").await?;
            gdns.push(GdnBlocks { conv_q, conv_k, conv_v, rec });
        }
        // GDN 快照池(E2c;S2 矩阵化):每 (层,quad) 一条 `snap_max × 行宽`
        // 缓冲 ⇒ 一条快照 = 一个槽(片 = 行宽)。仅 paged 形态。
        let mut snaps = GdnSnapPool {
            bufs: Vec::new(),
            keys: vec![None; snap_max],
            last_use: vec![0; snap_max],
            row_bytes: Vec::new(),
        };
        if paged {
            for _ in 0..n_gdn {
                for (elems, name) in [
                    (dims.nk * dims.hk * 3, "gdn.slot.conv_q"),
                    (dims.nv * dims.hv * 3, "gdn.slot.conv_k"),
                    (dims.nv * dims.hv * 3, "gdn.slot.conv_v"),
                    (dims.nv * dims.hk * dims.hv, "gdn.slot.rec"),
                ] {
                    let b = zero_block(face, elems * snap_max, name).await?;
                    snaps.row_bytes.push(elems * elem_bytes(b.0.dtype));
                    snaps.bufs.push(b);
                }
            }
        }
        Ok(GdnBank { gdns, snaps, snap_tick: 0 })
    }
}

impl DraftBank {
    /// 草稿 / 投机户:各池**仅在对应 spec 模式**分配(互斥由 `SpecMode` 派生)。
    async fn alloc<D: DeviceClient>(face: &mut D, ctx: &AllocCtx<'_>) -> Result<Self> {
        use crate::spec_runtime::SpecMode;
        let plan = ctx.plan;
        let dims = ctx.geom.dims;
        let page = ctx.geom.page;
        let nb = ctx.geom.nb;
        let paged = ctx.geom.paged;
        let n_gdn = ctx.n_gdn;
        let spec = plan.spec_mode != SpecMode::Off;
        let mtp = plan.spec_mode == SpecMode::Mtp;
        let dflash = plan.spec_mode == SpecMode::DFlash2;
        // MTP 草稿链 KV(E5-M2b):独立页池(1 层,几何 = full 层同款,
        // 容量同 pool_tokens);不接前缀缓存 —— mtp KV 依赖 hidden,
        // 不可 token 哈希。仅 paged(链吃块表)。
        //
        // **fp8 承载**(2026-10-21):`[spec] draft_kv_fp8` 此前只对 DFlash2 生效,
        // MTP 池恒 F16(= 64 KiB/token,192k 时白吃 864 MiB)。现按 `plan.draft_kv`
        // 定,存储与主池同款:**U32 扁平字节账**(元素数 = 字节/4,1 B/elem)
        // ⇒ 每 token 4 KiB → 2 KiB。读核由 `draft_env.kv.quant` 同步成 e4m3。
        let mtp_elem_b = plan.draft_kv.size_bytes();
        let mtp_elems = nb * dims.hkv * dims.hd * page;
        let (mtp_e, mtp_dt) = if mtp_elem_b == 1 {
            (mtp_elems / 4, Dtype::U32)
        } else {
            (mtp_elems, dims.dtype)
        };
        let mtp_kvs = if mtp && paged {
            Some(KvBlocks {
                k_cache: zero_block_dt(face, mtp_e, mtp_dt, "mtp.kv.k").await?,
                v_cache: zero_block_dt(face, mtp_e, mtp_dt, "mtp.kv.v").await?,
            })
        } else {
            None
        };
        // DFlash2 草稿池(E5-DF2):5 层 × 草稿几何(8 KV 头 × hd128;
        // z-lab 检查点家族定形),容量同 pool_tokens。仅 paged。
        // dtype = BF16(E5-DF3 同日十四;草稿路径 BF16 全程,独立于目标池
        // dims.dtype —— 同 2B/elem,几何 page/x 不变)
        // **DFlash2 ring**(2026-10-22):草稿 NC 核只读最近 2048 行(自块恒全可见),
        // 池按 max_seq 全长分配 = 10240B/token(184k 白吃 1.8GiB)。ring 只留
        // 窗口+自块(页对齐;32 页宽 ⇒ 2080 slots = 65 页)。
        let ring_slots = if dflash && paged {
            owl_models::layers::dflash2::draft_ring_slots(page)
        } else {
            0
        };
        let ring_nb = if ring_slots > 0 { ring_slots.div_ceil(page) } else { nb };
        let dflash_kvs = if dflash && paged {
            // 几何单源 = [`DRAFT_POOL`](原此处另写 `(8,128,5)`,与预算项两处各一遍)
            let geom = plan
                .draft_geom
                .map(|(l, h, d)| DraftPoolGeom::from_geom(l, h, d))
                .unwrap_or(DRAFT_POOL);
            let dlayers = geom.layers;
            // B6 偷显存:草稿池 e4m3(OWL_DFLASH_KV_FP8;1B/elem,U32 字节承载)
            // 分配项与预算项共用 `KvQuant::size_bytes` + 几何单源 ⇒ 两处不会再漂
            let df_bytes_elem = plan.draft_kv.size_bytes();
            let df_elems = geom.layer_elems(ring_nb, page);
            let (df_elems, df_dt) = if df_bytes_elem == 1 {
                (df_elems / 4, Dtype::U32)
            } else {
                // 池 dtype **随草稿检查点事实**(2026-10-19):量化族 = BF16(旧行为逐字节不变);
                // 未量化族(z-lab)= F16 —— 普通路只有 F16 算子变体(`vllm_reshape_and_cache_f16`
                // 的 key_cache 就要求 F16),池按 BF16 分配会在装配门响亮拒。
                (df_elems, plan.draft_dt)
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
        // MTP 链持久块表槽(E5-M5;propose 图烘焙;mtp 模式)
        let bt_mtp = if mtp && paged {
            Some(eval_ops(TensorOps::zeros(Dtype::F32, vec![1, nb]).step(), face).await?)
        } else {
            None
        };
        // spec 快照(E5-M2):专用缓冲(独立于 E2c 前缀快照池,免 LRU 逐出)
        let spec_snap = if spec && paged {
            let mut bufs = Vec::new();
            for _ in 0..n_gdn {
                bufs.push(zero_block(face, dims.nk * dims.hk * 3, "gdn.ring.conv_q").await?);
                bufs.push(zero_block(face, dims.nv * dims.hv * 3, "gdn.ring.conv_k").await?);
                bufs.push(zero_block(face, dims.nv * dims.hv * 3, "gdn.ring.conv_v").await?);
                bufs.push(zero_block(face, dims.nv * dims.hk * dims.hv, "gdn.ring.rec").await?);
            }
            Some(SpecSnapBuf { bufs })
        } else {
            None
        };
        // 最终 query replay 的 GDN 回滚缓冲(当前仅 spec off 档启用)
        let replay_snap = if plan.query_replay && paged {
            let mut bufs = Vec::new();
            for _ in 0..n_gdn {
                bufs.push(zero_block(face, dims.nk * dims.hk * 3, "gdn.replay.conv_q").await?);
                bufs.push(zero_block(face, dims.nv * dims.hv * 3, "gdn.replay.conv_k").await?);
                bufs.push(zero_block(face, dims.nv * dims.hv * 3, "gdn.replay.conv_v").await?);
                bufs.push(zero_block(face, dims.nv * dims.hk * dims.hv, "gdn.replay.rec").await?);
            }
            Some(SpecSnapBuf { bufs })
        } else {
            None
        };
        Ok(DraftBank {
            spec_snap,
            replay_snap,
            mtp_kvs,
            dflash_kvs,
            bt_mtp,
            fp8: plan.draft_kv.size_bytes() == 1,
            dt: plan.draft_dt,
            ring_slots,
        })
    }

    /// DFlash2 逻辑位 → ring 槽(0 = 线性;stdio/编码/propose 三面同源)
    fn dflash_slot(&self, p: usize) -> f32 {
        if self.ring_slots == 0 {
            p as f32
        } else {
            (p % self.ring_slots) as f32
        }
    }
}

/// XQA 的 `multiProcessorCount`(**回退常量**;2026-10-17 审查收口)。
///
/// KV-split 启发式参数,**只影响切分粒度、不影响正确性**。现役取值路径 =
/// 设备真值(`DeviceClient::sm_count` → `cuDeviceGetAttribute(MULTIPROCESSOR_COUNT)`);
/// 本常量只在**查询不可用**时兜底(且回退时**响亮告警** —— 不静默:84 = sm86 双卡,
/// 在 3080(68)上会让启发式系统性偏大)。
const XQA_SM_COUNT_FALLBACK: usize = 84;

/// **XQA 几何契约**(S6h):核只对 (hd=256, page=32, grp=6, fp8 e4m3) 实例化。
///
/// 单源:装配期**一处**判定;不满足返回原因(调用方**响亮告知**,不静默关脸)。
fn xqa_caps(dims: &ModelDims, page: usize, kv: owl_models::env::KvQuant) -> std::result::Result<(), String> {
    use owl_models::env::KvQuant;
    if !matches!(kv, KvQuant::Fp8E4M3) {
        return Err(format!("池量化 = {kv:?},契约要求 fp8 e4m3"));
    }
    if dims.hd != 256 {
        return Err(format!("head_dim = {} ,契约要求 256", dims.hd));
    }
    if page != 32 {
        return Err(format!("页宽 = {page} tok,契约要求 32"));
    }
    let grp = dims.hq / dims.hkv.max(1);
    if grp != 6 {
        return Err(format!("GQA 组 = {grp},契约要求 6"));
    }
    Ok(())
}

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

#[cfg(test)]
mod tp_shard_tests {
    //! S4 ④:`ModelDims` 派生 + 池公式的**分片感知**(纯逻辑;零设备)。
    use super::*;

    /// 从(可选分片的)27B spec 造 dims —— 与 `Engine::run` 同式
    fn dims27b(shard: Option<(usize, usize)>) -> ModelDims {
        use owl_models::plan::tp::TpPlan;
        let full = owl_models::formats::qwen35::qwen3_8_27b();
        let spec = match shard {
            Some((id, n)) => full.sharded(TpPlan::new(id, n).unwrap()).unwrap(),
            None => full,
        };
        let (hq, hkv, hd) = spec.full_heads;
        let (nk, hk, nv, hv) = spec.gdn_heads;
        ModelDims { hq, hkv, hd, nk, hk, nv, hv, hidden: spec.hidden, dtype: spec.dtype }
    }

    /// 分片 spec ⇒ dims **自动分片**(头减半;hd/hidden 不变)—— 引擎零改动即得
    #[test]
    fn dims_follow_sharded_spec() {
        let d = dims27b(None);
        let s = dims27b(Some((1, 2)));
        assert_eq!((d.hq, d.hkv, d.hd), (24, 4, 256));
        assert_eq!((s.hq, s.hkv, s.hd), (12, 2, 256), "头切半,hd 不变");
        assert_eq!((d.nk, d.nv), (16, 48));
        assert_eq!((s.nk, s.nv), (8, 24), "GDN 头切半");
        assert_eq!(s.hidden, d.hidden, "hidden 不切");
        // GQA 比例保持(XQA 判据 `hq/hkv == 6` 因此不受分片影响)
        assert_eq!(d.hq / d.hkv, s.hq / s.hkv);
    }

    /// **每 token KV 字节**:TP=2 正好减半(20G 卡的关键账);fp8 再减半;
    /// 草稿池部分不随分片变(固定 8×128 编)
    #[test]
    fn kv_per_tok_halves_under_tp2() {
        let n_full = 40usize; // 27B 的 full 层数(3:1 中 16 层?—— 此处只作线性因子)
        let full = kv_per_tok_bytes(n_full, &dims27b(None), false, false);
        let shard = kv_per_tok_bytes(n_full, &dims27b(Some((0, 2))), false, false);
        assert_eq!(shard * 2, full, "TP=2 ⇒ 每 token KV 减半");
        let full_fp8 = kv_per_tok_bytes(n_full, &dims27b(None), true, false);
        assert_eq!(full_fp8 * 2, full, "fp8 KV ⇒ 减半");
        let dd_shard = kv_per_tok_bytes(n_full, &dims27b(Some((0, 2))), false, true)
            - kv_per_tok_bytes(n_full, &dims27b(Some((0, 2))), false, false);
        assert_eq!(dd_shard, 5 * 2 * 8 * 128 * 2, "草稿池不随分片变");
    }
}

#[cfg(test)]
mod vram_budget_tests {
    use super::*;

    /// 27B 生产档几何(qwen3_8_27b:64 层 = 16 full + 48 GDN;
    /// `full_heads=(24,4,256)` ⇒ hkv·hd = 4·256 = 1024/层档;GDN `(16,128,48,128)`)
    fn dims_27b() -> ModelDims {
        ModelDims {
            hq: 24,
            hkv: 4,
            hd: 256,
            nk: 16,
            hk: 128,
            nv: 48,
            hv: 128,
            hidden: 5120,
            dtype: Dtype::F16,
        }
    }

    fn gdn_slot_elems(d: &ModelDims) -> usize {
        d.nk * d.hk * 3 + d.nv * d.hv * 3 + d.nv * d.hk * d.hv
    }

    /// 逐 token 项:**ring 后生产 per_tok = 主池 only**;草稿池几何仍单列
    /// (固定项,不再进 per_tok),且随池量化声明走。
    #[test]
    fn per_tok_matches_production_and_follows_declaration() {
        let d = dims_27b();
        // 生产调用(StatePool::alloc)对 DFlash2 传 `None` ⇒ 主池 only
        let main_only = PerTokBytes::of(&d, 16, 1, None, None, None);
        assert_eq!(main_only.total(), 32768, "ring 后生产 per_tok(主池 fp8)");
        // 草稿池几何helper:bf16 20480B/token、fp8 10240B/token(ring 固定项用)
        let bf16_draft = PerTokBytes::of(&d, 16, 1, Some(2), None, None);
        assert_eq!((bf16_draft.main_kv, bf16_draft.draft_pool), (32768, 20480));
        let fp8_draft = PerTokBytes::of(&d, 16, 1, Some(1), None, None);
        assert_eq!(fp8_draft.draft_pool, 10240, "草稿池 fp8 ⇒ 逐 token 减半");
        // 非投机档 = 纯主池(池 f16 2B/elem)
        assert_eq!(PerTokBytes::of(&d, 16, 2, None, None, None).total(), 65536);
        // MTP 链项(2026-10-21):fp8 = 2 KiB/token、f16 = 4 KiB/token —— 单列可验
        assert_eq!(PerTokBytes::of(&d, 16, 1, None, None, Some(1)).mtp_chain, 2 * d.hkv * d.hd);
        assert_eq!(PerTokBytes::of(&d, 16, 1, None, None, Some(2)).mtp_chain, 4 * d.hkv * d.hd);
        // ring 固定项(TP2 分片后 hkv=4):2080 slots(65 页 × 32)× 10240B/token = 20.3MiB
        let ring = owl_models::layers::dflash2::draft_ring_slots(32)
            * DraftPoolGeom::from_geom(5, 4, 128).per_tok_bytes(2);
        assert_eq!(ring, 2080 * 10240);
    }

    /// 固定项:GDN 格/快照按几何算,图 slab + warmup 两常数之和**必须仍是
    /// 历史校准的 400MiB**(B1 拆分不改总分 ⇒ 历史治理数字可比),且 27B 档
    /// 合计复现 boot 日志 `fixed_after=0.68G`(= 697MiB)。
    #[test]
    fn fixed_after_matches_production_and_keeps_historical_total() {
        // ⚠️ 2026-10-21 桶退役后:320 图 + 138 杂项 = **458 MiB**
        // (非桶图族实测 261 + 59MiB 余量;KV 池多 ~320MiB)。
        assert_eq!(
            GRAPH_SLAB_BUDGET + WARMUP_MISC_BUDGET,
            458 << 20,
            "桶退役后预算校准(2026-10-21:640→320)"
        );
        let f = FixedAfter::of(48, 1, gdn_slot_elems(&dims_27b()), 1, 0, false, false);
        assert_eq!(f.gdn_grid, 155_713_536);
        assert_eq!(f.gdn_snap, 155_713_536);
        assert_eq!(f.graph_slab, 320 << 20, "图项 = 非桶图族 261 + 余量");
        assert_eq!(f.warmup_misc, 138 << 20);
        assert_eq!(f.total(), 730_857_472 + (58 << 20), "fixed_after = 历史 400MiB + 图项 58MiB");
        // DFlash2 ring:固定项计入 total;2080 slots(65 页)×10240B/token
        let ring = 2080 * 10240;
        let fr = FixedAfter::of(48, 1, gdn_slot_elems(&dims_27b()), 1, ring, false, false);
        assert_eq!(fr.draft_ring, ring);
        assert_eq!(fr.total(), f.total() + ring);
        // spec/replay 快照各是一整行 GDN 状态（此前漏账 ⇒ cap 虚高 4860 tok/份）
        let fs = FixedAfter::of(48, 1, gdn_slot_elems(&dims_27b()), 1, 0, true, true);
        assert_eq!(fs.spec_snap, fs.gdn_snap, "spec 快照 = 一行 GDN");
        assert_eq!(fs.replay_snap, fs.gdn_snap, "replay 快照 = 一行 GDN");
        assert_eq!(fs.total(), f.total() + 2 * fs.gdn_snap);
    }

    /// 草稿池几何单源:逐 token 项与分配项同一来源(原两处各写一遍)
    #[test]
    fn draft_pool_geometry_is_single_sourced() {
        assert_eq!(DRAFT_POOL.per_tok_bytes(1), 10240);
        assert_eq!(DRAFT_POOL.per_tok_bytes(2), 20480);
        // 分配项口径:一页同时装 K/V 的同形切片(16k 档 nb=528 / page=32 实测几何)
        assert_eq!(DRAFT_POOL.layer_elems(528, 32), 528 * 8 * 128 * 32);
        assert_eq!(DRAFT_POOL.layer_elems(528, 32), 17_301_504);
    }
}

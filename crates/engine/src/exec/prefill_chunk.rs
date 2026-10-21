//! 块式 prefill(树根装配与执行;taps/encode/BISECT 取证)
use super::*;
use crate::engine::EngineKnobs;
use crate::running::StepProbes;
use crate::session::SessionBank;
use owl_iface::contract::ModelError;

// ============================================================================
// chunk host 侧输入表(自 `feat/prefill-graph` 择取,2026-10-15;**不含图路**)
// ============================================================================
//
// 为什么收成结构:这些表的取值逻辑原先散在本文件里(FI 四件套 / slots / lens /
// ids / pos / seq_cu / gdn_slot),而 XQA prefill 臂需要**同一批值**(页链/整请求
// 长/位掩码)⇒ 抽成唯一取值点(纯计算:零 await/零设备/零副作用)。
// 顺带收敛:eager 的 slots/lens/gdn_slot 由**逐层各一份**收成**共享一份**
// (值逐字节相同 ⇒ 零行为变化,纯命令数下降)。
//
// ⚠️ 上游的图路(prefill-graph S0/S1)未取;本结构只服务 eager 路 + XQA 表件。
/// **本 chunk 的设备输入束**(S6i):KV/GDN 逐层缓冲 + 五张逐行叶子 + 块表。
///
/// 为什么成对象:这些叶子全是**池缓冲的视图**(地址由 `StatePool` 拥有)——
/// 执行函数只该拿"一手交齐"的束去建 `ForwardCtx`,不该逐行摊开 9 个 `let`
/// 与两段 `map`(旧形 40 行,一改池几何就要在 5 处同步)。
pub(crate) struct ChunkInputs {
    pub kvs_step: Vec<KvBuffers>,
    pub gdns_step: Vec<GdnBuffers>,
    pub ids_t: TensorOps,
    pub pos_t: TensorOps,
    pub slots_leaf: TensorOps,
    pub lens_leaf: TensorOps,
    pub gdn_slot: TensorOps,
    pub seq_cu: TensorOps,
    pub bt_leaf: TensorOps,
}

impl ChunkInputs {
    /// 从池的层视图 + 本 chunk 的 host 表构造(纯 host 叶子;无设备往返)。
    pub(crate) fn build(pool: &crate::state::StatePool, tb: &ChunkTables, t: usize, bt_leaf: TensorOps) -> Self {
        let slots_leaf = TensorOps::from_host(Dtype::F32, vec![t], &f32b(&tb.slots));
        let lens_leaf = TensorOps::from_host(Dtype::F32, vec![t], &f32b(&tb.lens));
        let gdn_slot = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[tb.gdn_slot]));
        let kvs_step = pool
            .kv_caches()
            .iter()
            .map(|(k, v)| KvBuffers {
                k_cache: k.clone(),
                v_cache: v.clone(),
                slots: slots_leaf.clone(),
                kv_lens: lens_leaf.clone(),
                block_tables: bt_leaf.clone(),
            })
            .collect();
        let gdns_step = pool
            .gdn_caches()
            .iter()
            .map(|g| GdnBuffers {
                conv_q: g.conv_q.clone(),
                conv_k: g.conv_k.clone(),
                conv_v: g.conv_v.clone(),
                rec: g.rec.clone(),
                slots: gdn_slot.clone(),
            })
            .collect();
        Self {
            kvs_step,
            gdns_step,
            ids_t: TensorOps::from_host(Dtype::F32, vec![t], &f32b(&tb.ids)),
            pos_t: TensorOps::from_host(Dtype::F32, vec![t], &f32b(&tb.pos)),
            slots_leaf,
            lens_leaf,
            gdn_slot,
            seq_cu: TensorOps::from_host(Dtype::F32, vec![2], &f32b(&tb.seq_cu)),
            bt_leaf,
        }
    }
}

pub(crate) struct ChunkTables {
    pub ids: Vec<f32>,
    pub pos: Vec<f32>,
    /// KV 槽(物理页槽;legacy = 恒等直排)
    pub slots: Vec<f32>,
    pub lens: Vec<f32>,
    pub gdn_slot: f32,
    /// `[0, t]`(attn/gdn 的 cu_seqlens)
    pub seq_cu: Vec<f32>,
    pub fi: Option<FiTables>,
    /// XQA prefill 表件(`dispatch.xqa_prefill` 开才有;见 [`XqaTables`])
    pub xqa: Option<XqaTables>,
}

/// FlashInfer 表四件套(U32 位型;`nb_used` = 本 chunk 实际覆盖页数)
pub(crate) struct FiTables {
    pub q_cu: Vec<u32>,
    pub indices: Vec<u32>,
    pub indptr: Vec<u32>,
    pub last_len: Vec<u32>,
    pub nb_used: usize,
}

/// XQA prefill 表件(U32 位型;`q_seq_len` 由调用方按本 chunk 真行数传)
///
/// - `page_table` = 本会话**物理页链**(= FI 的 `indices` 同值 —— 两者是
///   "页链"的两种编码;kNHD 池下同一份);
/// - `seq_len` = 整请求 KV 长(`ctx_total`)—— XQA 的 `seqLens` 语义就是它,
///   块内因果**只由 mask 表达**(见 `cu/xqa/README.md` 调用契约);
/// - `mask` = spec-dec 位掩码 `[t, ceil(t/32)]` u32,行 i 的 bit 0..i 置 1。
///   行跨 `ceil(t/32)` ⇒ 表件随行数定制,`qSeqLen` 槽就传本 chunk 真行数
///   (尾块不越写)。
pub(crate) struct XqaTables {
    pub page_table: Vec<u32>,
    pub seq_len: u32,
    pub mask: Vec<u32>,
}

impl ChunkTables {
    /// 构造本 chunk 的表(纯计算:无 await、无设备、无副作用)。
    ///
    /// `past = Some((pages, tokens))` ⇒ 本 chunk 的"过去"由**子集页链**表达
    /// (选中块;`tokens` = 子集 token 数),`context_lens[t'] = tokens + (p+1−recall)`。
    /// 语义 = 稀疏注意力:被踢出工作集的块**不参与**本 chunk 的过去。
    ///
    /// `slots` **不变**(新 token 仍写自己会话链的槽)⇒ 与下沉/回灌正交。
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build_view(
        ids: &[u32],
        base: usize,
        t: usize,
        paged: bool,
        page: usize,
        bt_chain: &[u32],
        gdn_slot_v: usize,
        fi_face: bool,
        xqa_face: bool,
        past: Option<(&[u32], usize)>,
    ) -> Result<Self> {
        let seq = |start: usize, n: usize| -> Vec<f32> {
            (0..n).map(|i| (start + i) as f32).collect()
        };
        let slots = if paged {
            (base..base + t)
                .map(|pos| {
                    let b = bt_chain[pos / page];
                    (b * page as u32 + (pos % page) as u32) as f32
                })
                .collect()
        } else {
            seq(base, t)
        };
        // **过去视图**:表 = 子集页链;`lens[t'] = 子集 token + (base + t' + 1 − past_base)`
        let (tbl, lens, base_for_len) = match past {
            Some((pages, tokens)) if paged => {
                // `tokens` = **本 chunk 起点**处的可见 token 数(视图等效偏移;
                // 契约见 `coverage.rs::past_view_of`)。它同时喂两处:
                //   ① `lens[i] = tokens + i + 1`(本 chunk 第 i 行的可见数 —— 因果正确)
                //   ② `fi_tables_view(base_for_len, ..)` —— **等效偏移**契约:
                //      核按 `ctx_total = base_for_len + t` 反推因果前缀长与末页有效数;
                //      误传真实 `base` ⇒ `last_len` ≈ 全 ctx ⇒ 注意力窗口 O(ctx)。
                (pages.to_vec(), Vec::new(), tokens)
            }
            _ => (bt_chain.to_vec(), seq(base + 1, t), base),
        };
        let lens = if lens.is_empty() {
            // 子集档:每行 = 子集 token + 本 chunk 已见行数(= 本 chunk 位置偏移)
            (0..t).map(|i| (base_for_len + i + 1) as f32).collect()
        } else {
            lens
        };
        // ⚠️ **传 `base_for_len`(视图等效偏移),不是 `base`** —— 子集档下两者差
        // `base − recall_pos`(随 ctx 增长);传错的后果 = 注意力窗口 = 全 ctx
        // (O(ctx)/chunk ⇒ 总 O(ctx²))+ 末页读越界(2026-10-19 定谳)。
        let fi = if fi_face { Some(fi_tables_view(base_for_len, t, &tbl, page)?) } else { None };
        Ok(Self {
            ids: ids.iter().map(|v| *v as f32).collect(),
            pos: seq(base, t),
            slots,
            lens,
            gdn_slot: gdn_slot_v as f32,
            seq_cu: vec![0.0, t as f32],
            fi,
            xqa: xqa_face.then(|| XqaTables {
                page_table: tbl,
                seq_len: (base + t) as u32,
                mask: causal_mask_words(t),
            }),
        })
    }

    /// **桶布局表(M2b)**:槽面按布局 `t_pad` 行铸,但**语义全真** —— 真行真值,
    /// 假行落 **dump 页**(不在会话页链 ⇒ 注意力结构上读不到),其余表件
    /// (`seq_cu`/FI 四件/GDN 三表)一律真口径。
    ///
    /// 设计 §4.2 三律的落地点:
    /// - 律① 假行不得推进状态 ← `seq_cu = [0, t_real]`(varlen 核按它划行区间)
    ///   + GDN 序列表按 `t_real` 铸(表内容见 `exec` 的 `gdn_table_host_bytes` 调用)
    /// - 律② 假行的池写必须无害 ← `slots[≥t_real] = dump 页槽`
    /// - 律③ 采样读真行 ← 出图尾按 `t_real` 窄切(不住本函数)
    ///
    /// `dump_slot_base` = dump 页首槽(`StatePool::dump_slot_base`);
    /// 假行在页内循环落槽 —— 同址并发写本就不是读集(内容永不被读),
    /// 但仍摊开以降低写冲突概率。
    pub(crate) fn build_bucket(
        ids: &[u32],
        base: usize,
        t_real: usize,
        t_pad: usize,
        page: usize,
        bt_chain: &[u32],
        gdn_slot_v: usize,
        fi_face: bool,
        dump_slot_base: usize,
    ) -> Result<Self> {
        debug_assert!(t_real >= 1 && t_real <= t_pad, "桶几何非法:{t_real}/{t_pad}");
        let mut out = Self {
            ids: vec![0.0; t_pad],
            pos: (0..t_pad).map(|i| (base + i) as f32).collect(),
            slots: vec![0.0; t_pad],
            lens: vec![0.0; t_pad],
            gdn_slot: gdn_slot_v as f32,
            seq_cu: vec![0.0, t_real as f32],
            fi: if fi_face { Some(fi_tables(base, t_real, bt_chain, page)?) } else { None },
            xqa: None,
        };
        for i in 0..t_pad {
            if i < t_real {
                out.ids[i] = ids[i] as f32;
                let pos = base + i;
                out.slots[i] =
                    (bt_chain[pos / page] * page as u32 + (pos % page) as u32) as f32;
                out.lens[i] = (base + i + 1) as f32;
            } else {
                // 假行:id 取 0(嵌入有限)、槽落 dump 页、kv_len 取本 chunk 末真长
                out.slots[i] = (dump_slot_base + (i - t_real) % page) as f32;
                out.lens[i] = (base + t_real) as f32;
            }
        }
        Ok(out)
    }
}

/// FlashInfer 表四件套(`nb_used` 按本 chunk **实际覆盖页数**截断;
/// 桶布局下传 `t = t_real` —— 设备表读的 `qo_len` 必须与 `kv_len` 同口径,
/// 否则因果窗口错位 `t_pad - t_real`(见设计 §4.1c 推论 A 的落地注记)。
fn fi_tables(base: usize, t: usize, bt_chain: &[u32], page: usize) -> Result<FiTables> {
    let ctx_total = base + t;
    let nb = bt_chain.len();
    // ⚠️ **块链可能长于「本 chunk 已覆盖页数」** —— 块表按**全 prompt**
    // 预分配(scheduler::ensure_for_len),而本 chunk 只盖到 ctx_total。
    // 三件套必须按**实际覆盖页数**截断:原式 `ctx_total - nb*page + page`
    // 在 nb 过大时下溢(debug panic)/回绕成巨值(release → FI plan 拿着
    // 巨 last_len 越界读 indices 数组外)。2026-10-13 由
    // gpu_longctx_fi_direct_gate 首跑暴露。
    // ⚠️ **响亮拒(2026-10-19 审查)**:`.min(nb)` 只防了"链**过长**"(2026-10-13
    // 修过),没防"链**过短**" —— 那时 `nb_used < ceil(ctx/page)` ⇒
    // `last_len = ctx_total − (nb_used−1)·page > page`(结构非法)⇒ 核拿巨 last_len
    // 越界读 `indices` 数组外(与旧 bug 同一失败模式,方向相反)。内部不变量失守
    // 一律结构化 Err 上抛(仓规:不许静默夹取)。
    let need = ctx_total.div_ceil(page.max(1)).max(1);
    if nb < need {
        return Err(ModelError::Msg(format!(
            "FI 表失守:块链 {nb} 页 < 本 chunk 需要 {need} 页(base={base} t={t} page={page} ctx={ctx_total});\
             旧行为会静默夹成 last_len > page ⇒ 核越界读页链"
        )));
    }
    let nb_used = need;
    Ok(FiTables {
        q_cu: vec![0, t as u32],
        indices: bt_chain[..nb_used].to_vec(),
        indptr: vec![0, nb_used as u32],
        last_len: vec![(ctx_total - (nb_used - 1) * page) as u32],
        nb_used,
    })
}

/// FI 表四件套(**子集档**;B-3e-2 第 3 项):页链 = 传入的 `tbl`(可能 = 子集),
/// `last_len` = 末页在本**视图**内的有效行数(= 视图 token 数 − (nb_used−1)·page)。
fn fi_tables_view(base: usize, t: usize, tbl: &[u32], page: usize) -> Result<FiTables> {
    // 视图内 token 数:本 chunk 之前覆盖的页 × page 中**属于视图的部分**
    // —— 由调用方把 base 换成"视图起点等效偏移"后即 ctx_total 口径自洽
    let ctx_total = base + t;
    let nb = tbl.len();
    // 同 `fi_tables`:链短于覆盖页数 = 账本失守 ⇒ 响亮拒(子集档的 `tbl` 是
    // **视图页链**,其长度必须与"视图 token 数 ÷ 页"自洽 —— 契约见
    // `coverage.rs::past_view_of` 与本文件 §等效偏移注)。
    let need = ctx_total.div_ceil(page.max(1)).max(1);
    if nb < need {
        return Err(ModelError::Msg(format!(
            "FI 视图表失守:视图页链 {nb} 页 < 视图需要 {need} 页(base={base} t={t} page={page} ctx={ctx_total})"
        )));
    }
    let nb_used = need;
    Ok(FiTables {
        q_cu: vec![0, t as u32],
        indices: tbl[..nb_used].to_vec(),
        indptr: vec![0, nb_used as u32],
        last_len: vec![(ctx_total - (nb_used - 1) * page) as u32],
        nb_used,
    })
}

/// XQA SPEC_DEC 因果位掩码:`[t, ceil(t/32)]` u32,行 i 的 bit 0..i 置 1。
/// 语义定谳与多方佐证见 `crates/kernels/cu/xqa/README.md`(掩码位序/极性/
/// 行序三组 A/B:唯"行 i 看块内 0..i、含自身"贴 host)。
pub(crate) fn causal_mask_words(t: usize) -> Vec<u32> {
    let wpr = t.div_ceil(32);
    let mut v = vec![0u32; t * wpr];
    for i in 0..t {
        for w in 0..wpr {
            let lo = w * 32;
            let hi = (i + 1).min(lo + 32);
            v[i * wpr + w] = if hi <= lo {
                0
            } else if hi - lo == 32 {
                u32::MAX
            } else {
                (1u32 << (hi - lo)) - 1
            };
        }
    }
    v
}

/// **B2:桶 padding 比护栏**(B3 实测定档 2026-10-16;设计 §4.7g)
///
/// 判据面:图路每 chunk 省下的是**固定节余 H**(短 ctx 三臂实测 ≈2.3–4.8ms),
/// 而白算成本 = `pad_rows × s`,斜率 s 随 ctx 增长(t128 桶 **0.28**ms/行 /
/// t256 **0.67** / t512 **0.57**;由 eager 臂同桶内长度差直接量得)
/// ⇒ 每桶盈亏平衡 ≈ H/s ⇒ **{t32:16, t64:16, t128:12, t256:4, t512:4}**。
///
/// 旧值(= 单档 32 行)按"固定段 20–30ms + 0.78ms/行"估,已被 B3 推翻:
/// **pad=32 实测净输 +12.9…+15.0ms**(224→t256 / 480→t512)、pad=29 ≈ 0。
/// 长提示 chunk 恒 pad=0 ⇒ 收紧不影响 std 五点收益(实测 +1.2…+2.9%)。
/// 作用:①`t512` 单桶档不再自伤 ②命中率/白算可预算;t32/t64 桶(无直接实测)
/// 取 16 行(行成本更低,盈亏平衡上移)。
pub(crate) const fn bucket_pad_max_rows(t_pad: usize) -> usize {
    match t_pad {
        32 | 64 => 16,
        128 => 12,
        256 | 512 => 4,
        _ => 4,
    }
}

/// 让位/护栏的**一次 warn** 闸(响亮告知,不静默;每类一条)
static BUCKET_BYPASS_WARNED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static BUCKET_PAD_WARNED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// **prefill 设备半的主机侧事实**(S4 ⑤-K3):两 rank 共用一份**只读**事实
/// (`boot_seq`/`probes`/`knobs`/`tok`/块链/gdn 槽)+ **仅主 rank** 的账房
/// (`bank`:草稿臂的 MTP 表用;peer 镜像路 = None ⇒ 草稿臂响亮拒)。
pub(crate) struct ChunkHost<'a> {
    pub(crate) boot_seq: u64,
    pub(crate) probes: StepProbes,
    pub(crate) knobs: &'a EngineKnobs,
    /// 本会话 id(草稿臂的 MTP 表落户用)
    pub(crate) sid: u64,
    /// 本会话块链(logical → 物理页)
    pub(crate) bt_chain: &'a [u32],
    pub(crate) gdn_slot: usize,
    /// host 账房(草稿臂;peer 路 None)
    pub(crate) bank: Option<&'a mut SessionBank>,
    /// **B 轴运行态**(host 单份;peer 路 None —— 覆盖率探针只在主 rank 采集)。
    /// 探针要的 `(prof, cov)` 通过 `analysis_pair()` 一次借出。
    pub(crate) sel: Option<&'a mut crate::select::SelectRuntime>,
    /// **部分驻留的子集页链**(host 半算好传入:`past_view_of` 依赖
    /// `knobs.select_partial` 与 `sel_view` —— 都是 host 面)
    pub(crate) past_view: Option<(Vec<u32>, usize)>,
    /// **VLM 备料**(Some = 本 turn 有图;prefill 绑 m-RoPE/image,桶图让位)
    pub(crate) vlm: Option<&'a crate::vlm::VlmPrepared>,
}

impl ChunkHost<'_> {
    /// 草稿臂取账房(peer 镜像路无 ⇒ 响亮拒 —— TP v1 本就拒 depth > 0)
    fn bank_mut(&mut self) -> Result<&mut SessionBank> {
        self.bank
            .as_deref_mut()
            .ok_or_else(|| ModelError::Msg("草稿臂需要 host 账房(peer 镜像路不支持草稿)".into()))
    }
}

/// prefill 末块的**收割票据**(入队半产出;每 rank 一份 —— 块 id 是本 rank 的)
pub(crate) struct PrefillTail {
    /// 设备采样 token 块(4B;不读也会归池)
    pub(crate) tok: Bytes,
    /// spec seed hidden(草稿臂;`None` = 非草稿档)
    pub(crate) seed_hidden: Option<Bytes>,
}

/// 收割半产物(主 rank 消费;peer 只取 `token` 丢弃)
pub(crate) struct PrefillHarvested {
    pub(crate) token: u32,
    pub(crate) seed_hidden: Option<Bytes>,
}

impl<D: DeviceClient> crate::engine::RankDev<D> {
    // ========================================================================
    // prefill 桶图路(M2b;设计 §4–§6)
    // ========================================================================

    /// **桶图命中 ⇒ 全走图路;否则返回 None(调用方走 eager 原路)。**
    ///
    /// 返回 `Some(Some(tail))` = 末块(票据交收割半);`Some(None)` = 中间块;
    /// `None` = 未命中(调用方走 eager 原路)。**本函数零同步**(读数在收割半)。
    ///
    /// 三律的落地点都在 [`ChunkTables::build_bucket`] 与 `bucket_tail`:
    /// ①假行不进序列(seq_cu/GDN 表真口径)②假行池写落 dump 页 ③采样读真行(出图尾)。
    /// **单一路径执行**:桶命中就不跑 eager(否则 GDN/KV 双推进 —— §4.6)。
    async fn bucket_prefill_chunk(
        &mut self,
        host: &mut ChunkHost<'_>,
        ids: &[u32],
        base: usize,
        is_last: bool,
        bt_chain: &[u32],
        gdn_slot_v: usize,
    ) -> Result<Option<Option<PrefillTail>>> {
        let t_all = std::time::Instant::now();
        let t = ids.len();
        // VLM:塔/m-RoPE/image 只在 eager 路;桶图不烘它们(让位)
        if host.vlm.is_some() {
            return Ok(None);
        }
        if !self.pool.geom.paged || self.graphs.prefill.is_empty() {
            return Ok(None);
        }
        // 命中判定(向上取整;槽不匹配/未捕获都在 `usable` 里结构化拒)
        let Some(bg) = self.graphs.prefill.get_mut(t, gdn_slot_v) else {
            return Ok(None);
        };
        let t_pad = bg.t_pad;
        // ── **B1 诊断档让位**(2026-10-15)──
        // bisect / stage-tap / dflash 探针的实现只存在于 eager 路 ⇒ 桶命中会
        // **静默绕过**它们(旧行为)。优先级:诊断 > 图;响亮告知一次。
        if host.knobs.pf_bisect
            || host.knobs.pf_stages.is_some()
            || host.probes.dflash_probe
            || host.knobs.coverage_probe
            || host.knobs.select_decode
        {
            mcnt("prefill.bucket.bypass.diag", 1);
            if !BUCKET_BYPASS_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                owl_shared::log_warn!(
                    "prefill-graph",
                    "诊断档/B 轴选择层在场(pf_bisect / pf_stages / dflash_probe / coverage_probe / select_decode)\
                     ⇒ prefill 桶**全部让位 eager**(Q 槽只在 eager 路接线;要图就关它们)\n                     挂账:桶图路学会 Q 槽后,生产档(select_decode)就不必再让位"
                );
            }
            return Ok(None);
        }
        // ── **B2 padding 护栏**(B3 实测定档;见 [`bucket_pad_max_rows`])──
        let pad_rows = t_pad.saturating_sub(t);
        let pad_max = bucket_pad_max_rows(t_pad);
        if pad_rows > pad_max {
            mcnt("prefill.bucket.bypass.pad", 1);
            mcnt("prefill.bucket.pad_rows_skipped", pad_rows as u64);
            if !BUCKET_PAD_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                owl_shared::log_warn!(
                    "prefill-graph",
                    "桶 padding 超比跳过(t_pad={t_pad} t_real={t} pad={pad_rows} 行 > {pad_max});\
                     该 chunk 走 eager(白算 > 固定节余)"
                );
            }
            return Ok(None);
        }
        // 命中计数(B4:命中率/桶分布/白算行数,给 flavor 定档用)
        mcnt(&format!("prefill.bucket.hit.t{t_pad}"), 1);
        mcnt("prefill.bucket.hit", 1);
        mcnt("prefill.bucket.pad_rows", pad_rows as u64);
        // dump 页(律②):档位开启时池多分配 1 页;缺了 = 配置错(响亮拒,
        // 不静默用错的槽点)
        let Some(dump_slot) = self.pool.dump_slot_base() else {
            return Err(owl_iface::contract::ModelError::Msg(format!(
                "prefill 桶 t_pad={t_pad} 可用但池无 dump 页(档位/池几何不一致)"
            )));
        };
        // 桶布局表(真行真值 + 假行 dump)
        let tb = ChunkTables::build_bucket(
            ids,
            base,
            t,
            t_pad,
            self.pool.geom.page,
            bt_chain,
            gdn_slot_v,
            self.pool.geom.fi_face,
            dump_slot,
        )?;
        // 注:FI 四件在**图路**走输入槽(`BucketGraph::replay` 里按布局长度装填)
        // —— 池里那套持久块(`StatePool.fi_tables`)是 **eager 路**的表,这里不写
        // (写了也没人读:图烘的是槽指针)。GDN 序列表不同:图烘的就是池块,
        // 下方必须写。
        // ── GDN 序列表写持久槽(**布局 nt = ceil(t_pad/BT)、cu = t_real**)──
        // 图内 `nt` 是烘死的标量(布局口径);三张表的**内容**由本写定死:
        // cu=[0,t_real] ⇒ 行区间/递推落点都在真行上(§4.3e P-6b;假行不进序列)。
        // 写槽动作归 `KvBank`(槽的拥有者);桶路只声明**口径**(nt = 布局、
        // cu = 真行)与**流**(COMPUTE:与紧随的图发射同流,见 graph_plan::fill_slots)
        {
            let nt = t_pad.div_ceil(owl_models::kernel::GDN_BT).max(1);
            let face = self.graphs.session.face_mut();
            self.pool
                .write_gdn_tables(face, t, nt, crate::state::GdnTableStream::Compute)
                .await?;
        }
        // ── **FI plan 表同步**(2026-10-19;桶路 × 单图档)──
        // `fi_plan_restore = false` ⇒ 图**不**自携 plan 表(捕获窗内不烘 memcpy)
        // ⇒ `int_ws` 是**共享区**,桶回放前必须由宿主把**本桶**的表刷回去;
        // 否则读到的是别的写者(解码步/上一条 eager 分块)留下的表。
        // 与 decode/verify 同律(`exec::phases` / `exec::spec`),参数取**本桶的
        // 烘焙口径**:`total_rows = t_pad`(图内 FI 节点的标量),`ctx_total = t_pad`
        // (桶捕获时同一值 —— 该 key 在 capture warmup 已算好并登记 graph_owned)。
        // `fi_plan_restore = true`(缺省)时整段跳过:桶图自携表,旧行为逐字节不变。
        if bg.needs_plan_sync {
            let spec = self.pool.fi_plan_spec(t_pad, base + t);
            bg.plan.face_mut().fi_plan_sync(spec).await?;
        }
        // ── 回放(1 条 graph_launch)──
        // B7:桶路也记 `prefill.chunk`(总时,host 口径;与 eager 同 tag ⇒
        // 运维/仪表盘不丢行);`prefill.graph` 另记"回放发射"那一段。
        let t_gr = std::time::Instant::now();
        let (root, taps) = {
            let Some(out) = bg.replay(&tb, t).await? else {
                return Ok(None); // 未捕获(竞态:在判定与回放之间被降级)
            };
            (out.root, out.taps)
        };
        // `prefill.graph` = 槽写 + `graph_launch` 的**发射**成本(异步 ⇒ 天然小)
        mrec("prefill.graph", t_gr.elapsed());
        let tail = self.bucket_tail(host, &root, &taps, t, ids, base, is_last).await?;
        mrec("prefill.chunk", t_all.elapsed());
        Ok(Some(tail))
    }

    /// 桶图路的**出图尾**(律③:图内只出全行 hidden/taps;末行窄切 + lm_head
    /// + argmax 一律图外 eager —— 偏移 `(t_real-1)·dd` 是烘不得的 host 标量)。
    ///
    /// 与 eager 尾逐字同构(同一个 `narrow` → `lm_head` → `argmax` 链、同一套
    /// `prefill_dflash_encode`/`prefill_extend`/seed 保留纪律;块生命周期不同:
    /// 图输出块归 `GraphPlan`,本函数**不 free 它**,只 free 自己创建的暂态块)。
    #[allow(clippy::too_many_arguments)]
    async fn bucket_tail(
        &mut self,
        host: &mut ChunkHost<'_>,
        root: &Bytes,
        taps: &[TensorOps],
        t: usize,
        ids: &[u32],
        base: usize,
        is_last: bool,
    ) -> Result<Option<PrefillTail>> {
        let d = self.pool.geom.dims;
        let d_model = d.hidden;
        let hidden = TensorOps::of_block(root.id, d.dtype, vec![t, d_model]);
        let hidden = &hidden;
        if !is_last {
            // 中间块:taps 编码 / mtp extend(免 lm_head [T,V] 计算与大 dtoh)
            if !taps.is_empty() {
                let ts = std::time::Instant::now();
                self.prefill_dflash_encode(host.probes, taps, base).await?;
                mrec("dflash.prefill.encode", ts.elapsed());
            } else if matches!(self.spec_rt.drafter.as_ref(), Some(crate::spec_runtime::Drafter::Mtp(_))) && t > 1 {
                let sid = host.sid;
                let bank = host.bank_mut()?;
                self.prefill_extend(bank, sid, root, t - 1, ids, base).await?;
            }
            if host.probes.prefill_cksum {
                self.pf_cksum(root, base, t, d.hidden).await?;
            }
            return Ok(None);
        }
        // 末块
        if host.knobs.pf_fin_check {
            self.pf_fin_check(host, root, t, d_model).await?;
        }
        if !taps.is_empty() {
            let ts = std::time::Instant::now();
            self.prefill_dflash_encode(host.probes, taps, base).await?;
            mrec("dflash.prefill.encode", ts.elapsed());
        }
        // 末行窄切(**物化**:SliceView 作根丢偏移,根必须真块)
        let last = TensorOps::call(owl_models::ops::SemanticKernel::Narrow)
            .arg(hidden)
            .arg_usize(1)
            .arg_usize(d_model)
            .arg_usize((t - 1) * d_model)
            .arg_usize(d_model)
            .with_shape(d.dtype, vec![1, d_model]);
        let (hb, arena2) = {
            let env = self.env();
            let face = self.graphs.session.face_mut();
            eval_ops_scoped_env(last.step(), face, env, self.boot.tp.as_ref()).await?
        };
        let hb_view = TensorOps::of_block(hb.id, d.dtype, vec![1, d_model]);
        // P1.1:stride = 本 rank logits 宽(词表切档 = 半表;半表档 token 只是局部
        // argmax —— 消费面在 `harvest_prefill` 用拼接后的全表重算)
        let vocab = self.boot.model.logits_vocab();
        let (b, arena3) = {
            let env = self.env();
            let face = self.graphs.session.face_mut();
            let logits = self.boot.model.embed.lm_head_matmul(&hb_view);
            let tok = owl_models::ops::argmax_f32idx(&logits, vocab, 0);
            eval_ops_scoped_env(tok.step(), face, env, self.boot.tp.as_ref()).await?
        };
        if matches!(self.spec_rt.drafter.as_ref(), Some(crate::spec_runtime::Drafter::Mtp(_))) && t > 1 {
            let sid = host.sid;
            let bank = host.bank_mut()?;
            self.prefill_extend(bank, sid, root, t - 1, ids, base).await?;
        }
        let mut seed_keep: Option<Bytes> = None;
        {
            let face = self.graphs.session.face_mut();
            face.free(&arena3).await?;
            face.free(&arena2).await?;
            if self.spec_rt.drafter.is_some() {
                seed_keep = Some(hb);
            } else {
                face.free(&[hb.id]).await?;
            }
        }
        // S4 ⑤-K3 分相:读数(token ro 回读)移入收割半(见 `prefill_harvest`)
        Ok(Some(PrefillTail { tok: b, seed_hidden: seed_keep }))
    }

    /// 取证面 rank 标识(TP2 双 rank 同进程 ⇒ 同一 `boot_seq`、同一 tag;
    /// `mfact` 是 `counter_set` 语义 ⇒ 后写覆盖、前值丢失。pf 系列打点必须带
    /// rank,否则"单卡 vs 双卡"对拍拿到的是混合值(2026-10-17 实测踩到)。
    /// 单卡档恒 0(与历史 tag 语义等价)。
    fn tp_rank(&self) -> usize {
        self.boot.tp.as_ref().map(|t| t.plan.rank()).unwrap_or(0)
    }

    /// fin 校验(dtoh 一块 hidden 打点):`pf_fin_check` 走 FNV 指纹(mfact),
    /// `prefill_cksum` 走 |x| 之和(两档历史打点各自保留 —— 字段语义不同,
    /// 不能合并成一套)
    async fn pf_fin_check(
        &mut self,
        host: &ChunkHost<'_>,
        root: &Bytes,
        t: usize,
        dd: usize,
    ) -> Result<()> {
        let mut buf = vec![0u8; t * dd * 2];
        {
            let face = self.graphs.session.face_mut();
            face.dtoh(root, &mut buf).await?;
        }
        mfact(host.boot_seq, "pf.fin", fnv_bytes(&buf) & 0x7fff_ffff_ffff_ffff);
        Ok(())
    }

    /// `prefill_cksum` 档的 |x| 之和(诊断)
    async fn pf_cksum(
        &mut self,
        root: &Bytes,
        base: usize,
        t: usize,
        dd: usize,
    ) -> Result<()> {
        let mut buf = vec![0u8; t * dd * 2];
        {
            let face = self.graphs.session.face_mut();
            face.dtoh(root, &mut buf).await?;
        }
        let cks: f32 = buf
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32().abs())
            .sum();
        eprintln!("[cksum] chunk base={base} t={t} sum|x|={cks:.4}");
        Ok(())
    }

    /// 块式 prefill(W1;批P5 契约):ids [T] 从 KV 行 base 起步,
    /// 单序列语义(gdn_slot 恒 0;KV 行随 token 走)。末块走 logits 根
    /// 返回末行;中间块走 last_hidden 根(免 lm_head [T,V] 计算与大
    /// dtoh)返回 None。
    /// **prefill 收割半**:读回末块 token(仅主 rank;`read = false` 的 peer
    /// 只归设备采样块)+ 归块。必须在**两 rank 都已入队**之后调用。
    pub(crate) async fn prefill_harvest(
        &mut self,
        tail: PrefillTail,
        read: bool,
    ) -> Result<PrefillHarvested> {
        let face = self.graphs.session.face_mut();
        let mut token = 0u32;
        if read {
            let mut buf = [0u8; 4];
            face.dtoh(&tail.tok, &mut buf).await?;
            token = f32::from_le_bytes(buf) as u32;
        }
        face.free(&[tail.tok.id]).await?;
        Ok(PrefillHarvested { token, seed_hidden: tail.seed_hidden })
    }

    /// **prefill 入队半**(S4 ⑤-K3 分相):除末块 token 回读外**零同步**
    /// (TP 镜像块契约)—— 末块返回 `PrefillTail`(收割票据,由主 rank 读回)。
    pub(crate) async fn prefill_chunk_enqueue(
        &mut self,
        host: &mut ChunkHost<'_>,
        ids: &[u32],
        base: usize,
        is_last: bool,
    ) -> Result<Option<PrefillTail>> {
        let t_chunk = std::time::Instant::now();
        let mut t_ext = std::time::Duration::ZERO;
        let d = self.pool.geom.dims;
        let t = ids.len();
        // E2b:活跃会话块链 + GDN 格(逻辑位 → 物理槽由块表换算)
        // 块链/gdn 槽 = 主机侧事实(唯一取值点:调用方从账房取好传入)
        let (bt_chain, gdn_slot_v) = (host.bt_chain.to_vec(), host.gdn_slot);
        // ── prefill 桶图命中(M2b;设计 §4–§6)──
        // 命中 ⇒ 槽写(真行真值 + 假行 dump 页)+ 1 条 graph_launch + 出图尾;
        // 未命中/不可用 ⇒ 下方 eager 原路(**单一路径执行**,绝不双推进 —— §4.6)。
        if !self.graphs.prefill.is_empty() {
            match self.bucket_prefill_chunk(host, ids, base, is_last, &bt_chain, gdn_slot_v).await? {
                Some(tail) => return Ok(tail),
                None => {
                    // B4:桶在场但本 chunk 走 eager(未命中/让位/护栏)⇒ eager 计数
                    // (与 `prefill.bucket.hit*` 相除 = 命中率;tokens 求和 = 平均块长)
                    mcnt("prefill.eager.chunks", 1);
                    mcnt("prefill.eager.tokens", t as u64);
                }
            }
        }
        // ── host 侧输入表**唯一取值点**(见 [`ChunkTables`])──
        // **B-3e-2 第 3 项**:部分驻留档下,本 chunk 的"过去" = **子集页链**
        // (选中块 ++ 新生成尾部)—— 因为被踢出工作集的块已沉到宿主,会话链上
        // 是**占位符**,直接读整条链就是静默错页。
        let past_view = host.past_view.clone();
        let tb = ChunkTables::build_view(
            ids,
            base,
            t,
            self.pool.geom.paged,
            self.pool.geom.page,
            &bt_chain,
            gdn_slot_v,
            self.pool.geom.fi_face,
            self.env().attn.xqa_prefill,
            past_view.as_ref().map(|(pages, tokens)| (pages.as_slice(), *tokens)),
        )?;
        // **B-3e-2 第 3 项**:子集档 ⇒ 把本 chunk 的"过去表"写进池里的视图块,
        // 并把它当作 `block_tables`(eager prefill 无图,块叶直接可用)。
        let bt_leaf = match past_view.as_ref() {
            Some((pages, _)) => {
                {
                    let face = self.graphs.session.face_mut();
                    self.pool.write_select_view(face, pages).await?;
                }
                self.pool.select_view_leaf().ok_or_else(|| {
                    owl_iface::contract::ModelError::Msg(
                        "部分驻留档缺视图表槽(PoolPlan::select_view 未开?)".into(),
                    )
                })?
            }
            None => self.pool.bt_leaf_flat(),
        };
        // **本 chunk 的设备输入束**(S6i):叶子全是池缓冲的视图(地址归池),
        // 由 `ChunkInputs::build` 一次备齐 —— 执行函数不再逐行摊 9 个 let。
        let ci = ChunkInputs::build(&self.pool, &tb, t, bt_leaf);
        // FlashInfer 表四件套(i32 设备;OWL_FLASHINFER=1 时非空,E1.5):
        // q_cu=[0,T] / indices=块链 / indptr=[0,nb] / last_len=末页有效数。
        // ⚠️ 预物化(eval 往返同步)而非 from_host 叶子:Htod 在 H2D 流、
        // FI 核在 COMPUTE 流,无跨流序 —— 异步竞态会让 FI 偶发读到半写入
        // 表(渐进腐败案,2026-10-03;09-28 upload_pinned 同病史)。
        let fi_leaves;
        let fi_kv;
        let fi = if let Some(ft) = tb.fi.as_ref() {
            // P-3:表件写**持久槽**(槽归 `KvBank` 所有 ⇒ 写入动作由它封装;
            // 单条批量命令 + 回执即落地 = 无跨流竞态)
            let face = self.graphs.session.face_mut();
            fi_leaves = self.pool.write_fi_tables(face, ft).await?;
            let (kcs, vcs): (Vec<TensorOps>, Vec<TensorOps>) =
                self.pool.kv_fi_leaves().iter().map(|(k, v)| (k.clone(), v.clone())).unzip();
            fi_kv = (kcs, vcs);
            Some(FiPrefillCtx {
                kcs: &fi_kv.0,
                vcs: &fi_kv.1,
                q_cu: &fi_leaves.q_cu,
                indices: &fi_leaves.indices,
                indptr: &fi_leaves.indptr,
                last_len: &fi_leaves.last_len,
            })
        } else {
            None
        };
        // P1′(2026-10-14):**cu_seqlens [0,T] 每 chunk 只上传一次**。
        // 原先 attn/gdn 层各自 `from_host([2])` 兜底(16 attn + 32 gdn = **48 次
        // 8 字节 H2D / chunk**)—— 每次 8B 上传的**往返延迟 ~3.5ms**(issue 侧仅
        // 16µs)⇒ ~170ms/chunk 白烧。注入面本就存在(verify 图先例 E5-M4:
        // `ctx.seq_cu`),eager prefill 只是没接。
        let seq_cu = ci.seq_cu.clone();
        // ── B 轴选择层:Q 槽接入 ──
        // 只在**块边界 chunk**采(`(base+t) % page == 0`):分析期索引账才完整
        // (§8.4)。
        // 🔴 **两扇门**(2026-10-19 拆分):**生产面**(`select_decode`)也要 Q 面 ——
        // 它才是打分的输入;`coverage_probe` 只是诊断面(额外算真实质量)。
        // 拆之前两条耦合在一起 ⇒ 低显存档想开功能就**必须**背诊断的账。
        // 缺省两门全关 ⇒ 槽恒 None、零开销。
        let cov_sink: Option<crate::exec::coverage::QSink> = ((host.knobs.coverage_probe
            || host.knobs.select_decode)
            && self.pool.geom.paged
            && (base + t).is_multiple_of(self.pool.geom.page))
        .then(|| crate::exec::coverage::QSink::new(Vec::new()));
        // ── XQA prefill 表件(`dispatch.xqa_prefill`;缺省关)──
        // 三张 U32 位型表(页链/整请求长/因果位掩码)与 FI 四件套同路:
        // **预物化**(eval_ops 往返同步)而非 from_host 叶子 —— Htod 在 H2D
        // 流、核在 COMPUTE 流,无跨流序 ⇒ 异步竞态会让核偶发读到半写入表
        // (2026-10-03 渐进腐败案同病史,见上方 FI 表注)。
        // semaphores/scratch 两件取池里的真块:单块几何(`XqaPrefillRuntime`
        // 强制 `nbSubSeq=1`)下核**结构上不解引用**它们,传真块只为"不隐瞒"。
        // XQA prefill 表三件:表件逐 chunk 物化、semaphores/scratch 取自池 ——
        // 两件都归 `KvBank` 所有 ⇒ 组装由它封装(`prefill_xqa`)。
        let xqa_pf = if self.env().attn.xqa_prefill {
            match tb.xqa.as_ref() {
                Some(xt) => {
                    let face = self.graphs.session.face_mut();
                    self.pool.prefill_xqa(face, xt, t).await?
                }
                None => None,
            }
        } else {
            None
        };
        // 诊断(2026-10-14 A13):`[probes] pf_fence = true` ⇒ from_host 叶子(pos/slots/
        // lens/id/gdn_slot)建完后插一道流栅栏 —— 判「H2D 叶子与 COMPUTE 核
        // 无跨流序」竞态(仓内 2026-10-03 FI 表同病史:预物化修法)
        if host.probes.pf_fence {
            self.graphs.session.face_mut().sync().await?;
        }
        let mut ctx = ForwardCtx::model_prefill(
            t, &ci.pos_t, &ci.kvs_step, &self.boot.rope, &ci.gdns_step, &ci.slots_leaf, &ci.lens_leaf, &ci.gdn_slot,
            // A13(2026-10-14):chunked/scalar 臂的 rec 态寻址吃这个 host 标量
            // —— 必须喂本会话真槽(曾恒 0 = 读写别人格 → 长 prompt 塌陷)
            Some(gdn_slot_v),
            base, fi,
        );
        ctx.seq_cu = Some(&seq_cu);
        // ── GDN 序列表(cu|coff|idx;P-6b 2026-10-15)──
        //
        // 三张 FLA 表按**本 chunk 真行数 t** 铸,写进持久槽(地址恒定 ⇒ 图可烘);
        // `t_layout`(桶几何下的 t_pad)只进张量形状 —— 核的行区间与递推落点全
        // 由这三张表定 ⇒ 假行**不在序列里**。旧路:runtime 内部按 `t` 生成 ⇒
        // padding 假行推进递推态(探针 `gdn_bucket_pad_probe`:终态 Δ=3.6e-2)。
        // 序列表槽归 `KvBank` 所有 ⇒ 铸表 + 写槽由它封装(`write_gdn_tables`)。
        let gdn_tbl = {
            let face = self.graphs.session.face_mut();
            let nt = t.div_ceil(owl_models::kernel::GDN_BT).max(1);
            self.pool
                .write_gdn_tables(face, t, nt, crate::state::GdnTableStream::Default)
                .await?
        };
        let gdn_tables = gdn_tbl.as_ref().map(|g| owl_models::module::GdnTableSlot {
            table: &g.leaf,
            nt: g.nt,
        });
        ctx.gdn_tables = gdn_tables;
        ctx.xqa = xqa_pf;
        // ── **VLM:本 chunk 的 m-RoPE / 图像注入**(2026-10-20)──
        // 行号表/余弦表按 chunk 行号切;`has_image=false` 的块只装 rope
        // (image_scatter 不必跑全 -1 表)。塔输出 `merged` 是 turn 域设备叶。
        let (vlm_cos, vlm_sin, vlm_idx);
        if let Some(v) = host.vlm {
            let (cos, sin, idx, has_image) = v.ins.chunk_tables(base, t);
            vlm_cos = cos;
            vlm_sin = sin;
            vlm_idx = idx;
            ctx = ctx.with_vlm_rope(&vlm_cos, &vlm_sin);
            if has_image {
                if let Some((_, _, merged)) = &v.merged {
                    ctx = ctx.with_vlm_image(merged, &vlm_idx);
                }
            }
        }
        // ── B 轴覆盖率探针(Q-2 第一段):Q 槽接入(ctx 建完即挂)──
        if let Some(s) = cov_sink.as_ref() {
            ctx.q_sink = Some(s);
        }
        ctx.env = self.env();
        ctx.trace_gate = host.knobs.probes.trace_gate;
        let _vocab = self.boot.model.vocab_size();
        // E5-DF4 逐层 bisect(OWL_PF_BISECT;非 dflash 路径):单遍 multi
        // eval 产出 48 层 taps + fin —— **fin 块直接作为主路输出**(bisect
        // 前向有 GDN 状态副作用,双执行会双推进;单执行 + 逐层 checksum
        // → metrics `pf.c{base}.l{i}`,双 boot/双跑 diff = 首个发散层)。
        // fin block-leaf 下游 eval = 零操作透传。
        let pf_bisect = host.knobs.pf_bisect && self.spec_rt.dflash_tap_count == 0;
        // GDN 状态 checksum(pre/post;崩坏排查:状态脏 = reset 病,状态净
        // 而输出异 = 核病)。rec [slots,nv,kd,vd] f32 全量 dtoh ≈ 3.1MB/层。
        let pf_state_check = pf_bisect;
        // 指定层五站 stage tap(OWL_PF_STAGES=层号;配合 BISECT 定位层内核)
        let pf_stage_layer: Option<usize> = host.knobs.pf_stages;
        let mut bisect_fin_block: Option<(TensorOps, owl_iface::contract::Bytes)> = None;
        let mut _bisect_fin_blk: Option<owl_iface::contract::Bytes> = None;
        if pf_bisect {
            let gdn_leaves = self.pool.gdn_caches();
            if pf_state_check {
                for (j, g) in gdn_leaves.iter().enumerate() {
                    for (sn, buf, nb) in [
                        ("convq", &g.conv_q, self.pool.geom.gdn_slots * d.nk * d.hk * 3),
                        ("convk", &g.conv_k, self.pool.geom.gdn_slots * d.nv * d.hv * 3),
                        ("convv", &g.conv_v, self.pool.geom.gdn_slots * d.nv * d.hv * 3),
                        ("rec", &g.rec, self.pool.geom.gdn_slots * d.nv * d.hk * d.hv),
                    ] {
                        let mut sb = vec![0u8; nb * 4];
                        {
                            let face = self.graphs.session.face_mut();
                            let blk = owl_models::interpreters::eval_ops(buf.step(), face).await?;
                            face.dtoh(&blk, &mut sb).await?;
                        }
                        mfact(
                            host.boot_seq,
                            &format!("pf.r{}.c{}.pre.g{j}.{sn}", self.tp_rank(), base),
                            fnv_bytes(&sb) & 0x7fff_ffff_ffff_ffff,
                        );
                    }
                }
            }
            let _ids_all: Vec<usize> = (0..self.boot.model.layers.len()).collect();
            // GDN 层内打点(M4 gdn_tap 现成):每 GDN 层 8 件
            // (q/k/v raw + q_n/k_n/v_c/g/beta)→ 定位 qkv-proj/l2norm/conv/gating
            // GDN 层内 8 件打点(gdn_tap)已启用:层六站展开 + 8 件随层带出
            // (旧案:gtap 节点直挂 multi 曾现节点/块尺寸错配,改由
            // forward_bisect_stages 随 mixer 调用自然求值,不直挂 multi)
            ctx.gdn_tap = Some(std::rc::Rc::new(std::cell::RefCell::new(Vec::new())));
            let (fin, nodes) =
                self.boot.model
                    .prefill_bisect(&ci.ids_t, &ctx, pf_stage_layer);
            let mut roots: Vec<TensorOps> = Vec::new();
            let mut names: Vec<String> = Vec::new();
            for (name, n) in &nodes {
                roots.push(n.clone());
                names.push(name.clone());
            }
            roots.push(fin);
            names.push("fin".into());
            let refs: Vec<&TensorOps> = roots.iter().collect();
            let outs = {
                let env = self.env();
                let face = self.graphs.session.face_mut();
                owl_models::interpreters::eval_ops_multi_env(&refs, face, env, self.boot.tp.as_ref()).await?
            };
            for (i, r) in roots.iter().enumerate() {
                let n: usize = r.shape().iter().product();
                // 形状/元素数上屏(2026-10-17 三版;env 门控 —— 判读量级前
                // **必须先看形状**(形状不同时量级不可比,踩过)
                if i < 24 && owl_shared::env_reader::flag("OWL_PF_TAPS") {
                    eprintln!(
                        "[pf-tap] rank{} {} shape={:?} dtype={:?} n={n}",
                        self.tp_rank(), names[i], r.shape(), r.dtype()
                    );
                }
                // ⚠️ **按 dtype 定宽**(2026-10-17 三版):原硬编码 `n*2` ⇒
                // **f32 拍只 dtoh 一半字节**,再按 f32 解 = 垃圾(量级探针
                // 的 2308/64928 就是这类伪影,不是模型值)。
                let mut b = vec![0u8; n * r.dtype().size_bytes()];
                {
                    let face = self.graphs.session.face_mut();
                    let res = face.dtoh(&outs[i], &mut b).await;
                    if let Err(e) = res {
                        owl_shared::log_debug!("pf-bisect", "dtoh 失败 i={i} name={} shape={:?} blk_bytes={} err={e}", names[i], r.shape(), b.len());
                        return Err(e);
                    }
                }
                mfact(
                    host.boot_seq,
                    &format!("pf.r{}.c{base}.{}", self.tp_rank(), names[i]),
                    fnv_bytes(&b) & 0x7fff_ffff_ffff_ffff,
                );
                // **量级探针**(2026-10-17 S6 诊断):逐层 |x| 和(f16 位型解回)。
                // 为什么另立一条:哈希面对"多根 eval 的块覆盖"敏感(实测单卡/双卡
                // 全异,零信息量),而**量级**即便被覆盖也只在同阶内跳 —— 逐层
                // 爆炸/塌陷仍一眼可见(定位 TP 前向在哪层溢出)。
                {
                    // 读法 = **均值 |x|**(形状无关:分片与全量的元素数不同,
                    // 和不可比 —— 2026-10-17 首版踩到)+ 非有限计数 + 元素数。
                    // 打包:高 20 位 = 非有限数,低 40 位 = mean|x| × 1e4,
                    // 再各留一格:元素数以 2^20 为单位塞进次高位。
                    // **dtype 感知**(2026-10-17 二版:首版硬按 f16 解,f32 拍
                    // 出来的均值是垃圾 ⇒ 单卡/双卡的"同一拍"可能 dtype 不同,
                    // 比值全乱)。
                    let dt = r.dtype();
                    let (w, mut sum, mut bad, mut cnt, mut mx) =
                        (dt.size_bytes(), 0f64, 0usize, 0usize, 0f32);
                    for c in b.chunks_exact(w) {
                        let v = match dt {
                            owl_iface::contract::Dtype::F16 => {
                                half::f16::from_le_bytes([c[0], c[1]]).to_f32()
                            }
                            owl_iface::contract::Dtype::F32 => {
                                f32::from_le_bytes([c[0], c[1], c[2], c[3]])
                            }
                            _ => f32::NAN,
                        };
                        cnt += 1;
                        if v.is_finite() {
                            sum += (v as f64).abs();
                            if v.abs() > mx {
                                mx = v.abs();
                            }
                        } else {
                            bad += 1;
                        }
                    }
                    let mean = if cnt > 0 { sum / cnt as f64 } else { 0.0 };
                    let mean_q = ((mean * 1e4) as u64).min(0xff_ffff_ffff);
                    mfact(
                        host.boot_seq,
                        &format!("pf.m.r{}.c{base}.{}", self.tp_rank(), names[i]),
                        ((bad.min(0xffff) as u64) << 40) | mean_q,
                    );
                    // **max|x|**(2026-10-17 二版):均值会掩盖离群 —— 判"输入是否
                    // 有界"(RMSNorm 输出应 ~γ 量级)必须看 max。
                    let maxq = ((mx * 1e3) as u64).min(0xff_ffff_ffff);
                    let dt_code = match dt {
                        owl_iface::contract::Dtype::F16 => 1u64,
                        owl_iface::contract::Dtype::F32 => 2,
                        owl_iface::contract::Dtype::BF16 => 3,
                        _ => 9,
                    };
                    mfact(
                        host.boot_seq,
                        &format!("pf.x.r{}.c{base}.{}", self.tp_rank(), names[i]),
                        (dt_code << 56) | maxq,
                    );
                }
                if i == roots.len() - 1 {
                    let d_model = d.hidden;
                    bisect_fin_block = Some((
                        TensorOps::of_block(outs[i].id, d.dtype, vec![t, d_model]),
                        outs[i].clone(),
                    ));
                }
            }
            if pf_state_check {
                for (j, g) in gdn_leaves.iter().enumerate() {
                    for (sn, buf, nb) in [
                        ("convq", &g.conv_q, self.pool.geom.gdn_slots * d.nk * d.hk * 3),
                        ("convk", &g.conv_k, self.pool.geom.gdn_slots * d.nv * d.hv * 3),
                        ("convv", &g.conv_v, self.pool.geom.gdn_slots * d.nv * d.hv * 3),
                        ("rec", &g.rec, self.pool.geom.gdn_slots * d.nv * d.hk * d.hv),
                    ] {
                        let mut sb = vec![0u8; nb * 4];
                        {
                            let face = self.graphs.session.face_mut();
                            let blk = owl_models::interpreters::eval_ops(buf.step(), face).await?;
                            face.dtoh(&blk, &mut sb).await?;
                        }
                        mfact(
                            host.boot_seq,
                            &format!("pf.r{}.c{}.post.g{j}.{sn}", self.tp_rank(), base),
                            fnv_bytes(&sb) & 0x7fff_ffff_ffff_ffff,
                        );
                    }
                }
            }
            owl_shared::log_debug!("pf-bisect", "chunk base={base} t={t} 层 checksum 已记");
        }
        let _face = self.graphs.session.face_mut();
        // P1.1:同桶图路(stride = 本 rank logits 宽;token 在收割半重算)
        let vocab = self.boot.model.logits_vocab();
        if is_last {
            // E3:末行设备 argmax(4B 回读,免 [T,V] 大 dtoh)。
            // E5-M5:树一 = hidden_full 根([T, hidden] 存活 → prefill
            // extend 消费行 0..T-1);树二 = 末行物化(OPS_NARROW,spec
            // 模式转正为 seed)+ lm_head → argmax。
            // DFlash2:同树多根(hidden + 5 taps;CSE 共享前向零额外计算)。
            // 树根 = concat(hidden, taps)(单根 scoped eval 竞技场回收照旧;
            // 块内切片消费 —— SliceView 作根丢偏移,故不走多根 plain eval)
        let (hidden_full, prefill_taps) = if pf_bisect {
                // bisect fin 块直通(block-leaf 下游 eval 零操作)
                let (decl, blk) = bisect_fin_block.take().ok_or_else(|| ModelError::EngineInvariant { what: "bisect fin 块缺席(诊断档口径)".into() })?;
                _bisect_fin_blk = Some(blk);
                (decl, None)
            } else if self.spec_rt.dflash_tap_count > 0 {
                let (h, tp) = self.boot.model.tapped_hidden(&ci.ids_t, &ctx, &[5, 19, 33, 47, 61]);
                let mut c = TensorOps::call(owl_models::ops::SemanticKernel::Concat).arg(&h);
                for t in &tp {
                    c = c.arg(t);
                }
                for _ in tp.len()..7 {
                    c = c.arg(&h);
                }
                let n = tp.len() + 1;
                let dd = h.shape()[1];
                // concat(n, r, d):r = 每份行数 t(非 t·dd!—— t·dd 会令
                // rd = t·dd² → in 恒 0 → taps 全变 hidden 越界复制)
                let root = c
                    .arg_usize(n)
                    .arg_usize(t)
                    .arg_usize(dd)
                    .with_shape(d.dtype, vec![n * t, dd]);
                (root, Some(tp))
            } else {
                (self.boot.model.last_hidden(&ci.ids_t, &ctx), None)
            };
            let d_model = hidden_full.shape()[1];
            let ld = live_diag_begin();
            let r_eval = self.eval_root_and_q(&hidden_full, cov_sink.as_ref()).await;
            live_diag_end(ld, base, t);
            let (hfb, arena1, cov_q) = r_eval?;
            // Q 回读(探针档;设备块当场 free;分析在块边界事件后做)
            self.dump_hidden_row(&hfb, base, t, d_model).await?;
            if let Some(qs) = cov_q {
                if let Some(sel) = host.sel.as_deref_mut() {
                    let (pl, pc) = sel.analysis_pair();
                    self.coverage_take_q(qs, base, t, pl, pc).await?;
                }
            }
            // E5 崩坏排查·轻探针(OWL_PF_FIN_CHECK;bisect 900 dtoh 强掩蔽
            // 0/10,本探针仅 1 dtoh —— 近零扰判决:fin 翻 = 前向内部腐坏;
            // fin 绿而 pf.last 翻 = lm_head/argmax 面腐坏)
            if host.knobs.pf_fin_check && !pf_bisect {
                let n_fin = t * d_model;
                let mut fb = vec![0u8; n_fin * 2];
                {
                    let face = self.graphs.session.face_mut();
                    face.dtoh(&hfb, &mut fb).await?;
                }
                mfact(
                    host.boot_seq,
                    "pf.fin",
                    fnv_bytes(&fb) & 0x7fff_ffff_ffff_ffff,
                );
            }
            // DFlash2:草稿 KV 物化(root 块行 [T, 6T) 切片出 taps → encode)
            if let Some(tp) = &prefill_taps {
                let ts = std::time::Instant::now();
                let n_tp = tp.len();
                let tap_views: Vec<TensorOps> = (0..n_tp)
                    .map(|i| {
                        TensorOps::of_block(hfb.id, d.dtype, vec![n_tp * t + t, d_model])
                            .slice_view((i + 1) * t * d_model, vec![t, d_model])
                    })
                    .collect();
                self.prefill_dflash_encode(host.probes, &tap_views, base).await?;
                // 探针:concat 父块整块落盘(python 对拍 taps;真块 dtoh)
                if host.probes.dflash_probe && !self.spec_rt.mem_dumped2 {
                    self.spec_rt.mem_dumped2 = true;
                    let face = self.graphs.session.face_mut();
                    let mut rbuf = vec![0u8; (n_tp + 1) * t * d_model * 2];
                    face.dtoh(&hfb, &mut rbuf).await?;
                    if let Err(e) = owl_shared::file_loader::write(&crate::probe_dump::probe_path("owl_dflash_root.bin"), &rbuf) {
                owl_shared::log_error!("dump", "落盘失败(诊断面不许静默): {e}");
            }
                    owl_shared::log_debug!("dflash-probe", "concat 父块落盘 /tmp/owl_dflash_root.bin t={t}");
                }
                owl_shared::log_debug!("dflash", "prefill encode {t} rows @{} {:?}", base, ts.elapsed());
            }
            let hview = TensorOps::of_block(hfb.id, d.dtype, vec![t, d_model]);
            // 末行窄切 **物化**(OPS_NARROW):SliceView 作根透传父块丢偏移
            // (刀 3b 同族),根必须真块
            let last = TensorOps::call(owl_models::ops::SemanticKernel::Narrow)
                .arg(&hview)
                .arg_usize(1)
                .arg_usize(d_model)
                .arg_usize((t - 1) * d_model)
                .arg_usize(d_model)
                .with_shape(d.dtype, vec![1, d_model]);
            let (hb, arena2) = {
                let env = self.env();
                let face = self.graphs.session.face_mut();
                eval_ops_scoped_env(last.step(), face, env, self.boot.tp.as_ref()).await?
            };
            let hb_view = TensorOps::of_block(hb.id, d.dtype, vec![1, d_model]);
            let blks = {
                let env = self.env();
                let face = self.graphs.session.face_mut();
                let logits = self.boot.model.embed.lm_head_matmul(&hb_view);
                // 崩坏排查:末行 logits top-2(id+值)——近局翻转判别
                if host.probes.dflash_probe {
                    let bs_seq = host.boot_seq;
                    let lb = owl_models::interpreters::eval_ops(logits.step(), face).await?;
                    let mut lbuf = vec![0u8; vocab * 2];
                    face.dtoh(&lb, &mut lbuf).await?;
                    let lb2 = owl_models::interpreters::eval_ops(logits.step(), face).await?;
                    let mut lbuf2 = vec![0u8; vocab * 2];
                    face.dtoh(&lb2, &mut lbuf2).await?;
                    let same_inproc = lbuf == lbuf2;
                    owl_shared::log_debug!("pf-probe", "boot{bs_seq} 同boot两次lm_head一致={same_inproc}");
                    let mut top: Vec<(f32, u32)> = lbuf
                        .chunks_exact(2)
                        .enumerate()
                        .map(|(i, c)| (half::f16::from_le_bytes([c[0], c[1]]).to_f32(), i as u32))
                        .collect();
                    top.select_nth_unstable_by(1, |a, b| b.0.total_cmp(&a.0));
                    top.truncate(2);
                    top.sort_by(|a, b| b.0.total_cmp(&a.0));
                    owl_shared::metrics::with_metrics_store(|s| {
                        s.counter_set("prefill.top1", u64::from(top[0].1), file!(), line!());
                        s.counter_set(
                            "prefill.top1.gap100",
                            ((top[0].0 - top[1].0) * 100.0) as u64,
                            file!(),
                            line!(),
                        );
                    });
                    owl_shared::log_debug!("pf-probe", "boot{} argmax(top2)={} gap={:.5} top=({} {:.4}) ({:.4})",
                        bs_seq,
                        top[0].1,
                        top[0].0 - top[1].0,
                        top[0].1, top[0].0, top[1].0
                    );
                }
                // 两根**一次**求值(顺序 = [logits, token]):logits 根必须拿到
                // 块句柄 —— prefill 末块不走图 ⇒ 会话图 "logits" 输出槽不会被
                // 写;只求 token 根的话 logits 只是 memo 中间量(句柄拿不到,
                // scoped 回收后更拿不到)。见下方发布块。
                let tok = owl_models::ops::argmax_f32idx(&logits, vocab, 0);
                let trees: [&TensorOps; 2] = [&logits, &tok];
                owl_models::interpreters::eval_ops_multi_scoped_env(
                    &trees,
                    face,
                    env,
                    self.boot.tp.as_ref(),
                )
                .await?
            };
            let b = blks[1].clone();
            // ── 取证面:末块 logits 发布进会话图输出槽 ──────────────────
            // 2026-10-17 定谳:`read_output_f32("logits")` 读的是**图 step** 的
            // 槽(`GraphPlan.last`);prefill 末块是 eager 路 ⇒ 首步读到的会是
            // 上一次图 step(装配/warmup)的残留 —— G8-b 的"首 token 互为
            // top-N"因此结构性不可判(实测:TP1/TP2 首条 logits 各是自己的
            // 残值,argmax 与 gen[0] 无关;i ≥ 1 步则严丝合缝 —— 每个 decode
            // 步的图自己写槽)。这里把末块 logits 拷进槽块(每 turn 一次,
            // D2D ≤ 500KB)。
            {
                let lb = &blks[0];
                if let Some(slot) = self.graphs.session.output_block("logits") {
                    let n = lb.len.min(slot.len) * lb.dtype.size_bytes();
                    let face = self.graphs.session.face_mut();
                    face.copy_block_at(lb, 0, &slot, 0, n).await?;
                }
            }
            if matches!(self.spec_rt.drafter.as_ref(), Some(crate::spec_runtime::Drafter::Mtp(_))) && t > 1 {
                // prefill extend(E5-M5):本 chunk 行 0..T-1 的 mtp KV
                // (行 T-1 的配对需下一 chunk 首 token,由其 extend 补;
                // 末位 P-1 由首轮 propose_first 的 extend 行补)
                let sid = host.sid;
                let bank = host.bank_mut()?;
                self.prefill_extend(bank, sid, &hfb, t - 1, ids, base).await?;
            }
            // S4 ⑤-K3 分相:**末块 token 回读移出本函数**(收割半 `prefill_harvest`)
            // —— 入队半必须零同步(TP 镜像块契约:对端 stage 未入队时读数 = AR 死锁)。
            let mut seed_keep: Option<owl_iface::contract::Bytes> = None;
            {
                let face = self.graphs.session.face_mut();
                // multi_scoped 只留根块(logits / token);中间块已内部抽干
                face.free(&[blks[0].id]).await?; // logits 根(已发布进槽)
                face.free(&arena2).await?;
                if self.spec_rt.drafter.is_some() {
                    seed_keep = Some(hb);
                } else {
                    face.free(&[hb.id]).await?;
                }
                face.free(&[hfb.id]).await?;
                face.free(&arena1).await?; // E2a:中间块归池(每 chunk 零净增)
            }

            // B7(2026-10-15):末块也记 `prefill.chunk`(中间块/桶路都有;此处缺行
            // 会让图表按"chunk 数"口径对不上)
            mrec("prefill.chunk", t_chunk.elapsed());
            Ok(Some(PrefillTail { tok: b, seed_hidden: seed_keep }))
        } else {
            // 中间块:last_hidden 根(状态推进完整;lm_head 免算)。
            // DFlash2:同构 concat 单根(taps 随块回收前切片 encode)
            let (tree, prefill_taps) = if pf_bisect {
                let (decl, blk) = bisect_fin_block.take().ok_or_else(|| ModelError::EngineInvariant { what: "bisect fin(非末)块缺席(诊断档口径)".into() })?;
                _bisect_fin_blk = Some(blk);
                (decl, None)
            } else if self.spec_rt.dflash_tap_count > 0 {
                let (h, tp) = self.boot.model.tapped_hidden(&ci.ids_t, &ctx, &[5, 19, 33, 47, 61]);
                let mut c = TensorOps::call(owl_models::ops::SemanticKernel::Concat).arg(&h);
                for tt in &tp {
                    c = c.arg(tt);
                }
                for _ in tp.len()..7 {
                    c = c.arg(&h);
                }
                let n = tp.len() + 1;
                let dd = h.shape()[1];
                // concat(n, r, d):r = 每份行数 t(非 t·dd!—— t·dd 会令
                // rd = t·dd² → in 恒 0 → taps 全变 hidden 越界复制)
                let root = c
                    .arg_usize(n)
                    .arg_usize(t)
                    .arg_usize(dd)
                    .with_shape(d.dtype, vec![n * t, dd]);
                (root, Some((tp, dd)))
            } else {
                (self.boot.model.last_hidden(&ci.ids_t, &ctx), None)
            };
            let ld = live_diag_begin();
            let r_eval = self.eval_root_and_q(&tree, cov_sink.as_ref()).await;
            live_diag_end(ld, base, t);
            let (b, arena, cov_q) = r_eval?;
            self.dump_hidden_row(&b, base, t, d.hidden).await?;
            if let Some(qs) = cov_q {
                if let Some(sel) = host.sel.as_deref_mut() {
                    let (pl, pc) = sel.analysis_pair();
                    self.coverage_take_q(qs, base, t, pl, pc).await?;
                }
            }
            {
                let face = self.graphs.session.face_mut();
                let esz = if d.dtype == Dtype::F16 { 2 } else { 4 };
                // DFlash2:b = concat 根 [n_tp·t + t, hidden](行 0 块 =
                // hidden,taps 同块后随)—— dtoh 整父块契约必须等大读
                // (曾用 t·hidden → 多 chunk DFlash2 首跑即炸;MTP 无 taps
                // 时 n_tp=0 同式兼容)
                let rows_b = if let Some(tp) = &prefill_taps { tp.0.len() + 1 } else { 1 } * t;
                // ⚠️ 2026-10-17(K3 分相):原为**无条件** dtoh(整块隐层回读),
                // 但 `buf` 只被 `prefill_cksum` 探针消费 ⇒ 纯浪费 + 每 chunk 一次
                // 同步(TP 入队半的硬违例)。收进探针门内(单卡路语义不变)。
                if host.probes.prefill_cksum {
                    let mut buf = vec![0u8; rows_b * d.hidden * esz];
                    face.dtoh(&b, &mut buf).await?;
                    // 临时取证:每 chunk 隐层校验和(FI 开/关对比找第一分歧)
                    let cks: f32 = buf.chunks_exact(2)
                        .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32().abs())
                        .sum();
                    owl_shared::log_debug!("cksum", "chunk base={base} t={t} sum|x|={cks:.4}");
                }
                // prefill extend(E5-M5):行 0..T-1 的 mtp KV(行 T-1 配对需
                // 下一 chunk 首 token,由其 extend 补)。DFlash2:草稿 KV
                // 物化(整 chunk 全行 taps → encode)
                if let Some((tp, dd)) = &prefill_taps {
                    let n_tp = tp.len();
                    let tap_views: Vec<TensorOps> = (0..n_tp)
                        .map(|i| {
                            TensorOps::of_block(b.id, d.dtype, vec![n_tp * t + t, *dd])
                                .slice_view((i + 1) * t * (*dd), vec![t, *dd])
                        })
                        .collect();
                    self.prefill_dflash_encode(host.probes, &tap_views, base).await?;
                } else if matches!(self.spec_rt.drafter.as_ref(), Some(crate::spec_runtime::Drafter::Mtp(_))) && t > 1 {
                    let ts = std::time::Instant::now();
                    let sid = host.sid;
                    let bank = host.bank_mut()?;
                    self.prefill_extend(bank, sid, &b, t - 1, ids, base).await?;
                    t_ext += ts.elapsed();
                }
                let face = self.graphs.session.face_mut();
                face.free(&[b.id]).await?;
                face.free(&arena).await?; // E2a:中间块归池(每 chunk 零净增)
            }
            mrec("prefill.chunk", t_chunk.elapsed());
            if t_ext > std::time::Duration::ZERO {
                mrec("prefill.extend", t_ext);
            }
            Ok(None)
        }
    }


}

// =====================================================================

// ============================================================================
// **chunk 活性归因诊断**(门:`OWL_LIVE_DIAG`;2026-10-19)
// ============================================================================
//
// 问题面:"27B prefill chunk 的激活面 16.5 MiB/token"(单 chunk 峰值 ~2-3 GiB,
// 直接把 KV 池挤到剩 1 GiB)。解释器侧**已有**中途活性回收(refcount 归零即
// free),所以"为什么还这么大"只能靠归因面回答:
//   - 影子 arena 双口径(容量峰值 vs 活跃峰值)的**构成 top-N**(按块名);
//   - 每个回收出口的**拒因直方图**(pending/users/alias/非竞技场/keep)+ 按名分账;
//   - 引用计数推送次数(证明求值器真在推,不是面断了)。
//
// 用法:`OWL_LIVE_DIAG=1`,跑一个 prefill 档,看 `[live-diag]` 四行。
fn live_diag_begin() -> bool {
    if std::env::var_os("OWL_LIVE_DIAG").is_none() {
        return false;
    }
    owl_shared::slab_hint::probe_on(true);
    true
}

fn live_diag_end(on: bool, base: usize, t: usize) {
    if !on {
        return;
    }
    let peak = owl_shared::slab_hint::probe_end();
    eprintln!(
        "[live-diag] chunk@{base} t={t}:影子 arena 高水位 = {} MiB",
        peak >> 20
    );
    let rep = owl_shared::slab_hint::take_probe_report();
    if !rep.is_empty() {
        eprintln!("[live-diag] 峰值构成:{rep}");
    }
    eprintln!(
        "[live-diag] 回收出口:{}",
        owl_shared::slab_hint::live_diag::take_report()
    );
    eprintln!(
        "[live-diag] 引用计数推送 {} 次",
        owl_shared::slab_hint::refs_pushed()
    );
}

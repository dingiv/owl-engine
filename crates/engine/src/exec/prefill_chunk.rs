//! 块式 prefill(树根装配与执行;taps/encode/BISECT 取证)
use super::*;
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
    /// 纯计算:无 await、无设备、无副作用
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build(
        ids: &[u32],
        base: usize,
        t: usize,
        paged: bool,
        page: usize,
        bt_chain: &[u32],
        gdn_slot_v: usize,
        fi_face: bool,
        xqa_face: bool,
    ) -> Self {
        Self::build_view(ids, base, t, paged, page, bt_chain, gdn_slot_v, fi_face, xqa_face, None)
    }

    /// 同 [`Self::build`],多一个**过去视图**(B-3e-2 第 3 项)。
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
    ) -> Self {
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
        let fi = fi_face.then(|| fi_tables_view(base_for_len, t, &tbl, page));
        Self {
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
        }
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
    ) -> Self {
        debug_assert!(t_real >= 1 && t_real <= t_pad, "桶几何非法:{t_real}/{t_pad}");
        let mut out = Self {
            ids: vec![0.0; t_pad],
            pos: (0..t_pad).map(|i| (base + i) as f32).collect(),
            slots: vec![0.0; t_pad],
            lens: vec![0.0; t_pad],
            gdn_slot: gdn_slot_v as f32,
            seq_cu: vec![0.0, t_real as f32],
            fi: fi_face.then(|| fi_tables(base, t_real, bt_chain, page)),
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
        out
    }
}

/// FlashInfer 表四件套(`nb_used` 按本 chunk **实际覆盖页数**截断;
/// 桶布局下传 `t = t_real` —— 设备表读的 `qo_len` 必须与 `kv_len` 同口径,
/// 否则因果窗口错位 `t_pad - t_real`(见设计 §4.1c 推论 A 的落地注记)。
fn fi_tables(base: usize, t: usize, bt_chain: &[u32], page: usize) -> FiTables {
    let ctx_total = base + t;
    let nb = bt_chain.len();
    // ⚠️ **块链可能长于「本 chunk 已覆盖页数」** —— 块表按**全 prompt**
    // 预分配(scheduler::ensure_for_len),而本 chunk 只盖到 ctx_total。
    // 三件套必须按**实际覆盖页数**截断:原式 `ctx_total - nb*page + page`
    // 在 nb 过大时下溢(debug panic)/回绕成巨值(release → FI plan 拿着
    // 巨 last_len 越界读 indices 数组外)。2026-10-13 由
    // gpu_longctx_fi_direct_gate 首跑暴露。
    let nb_used = ctx_total.div_ceil(page).min(nb).max(1);
    FiTables {
        q_cu: vec![0, t as u32],
        indices: bt_chain[..nb_used].to_vec(),
        indptr: vec![0, nb_used as u32],
        last_len: vec![(ctx_total - (nb_used - 1) * page) as u32],
        nb_used,
    }
}

/// FI 表四件套(**子集档**;B-3e-2 第 3 项):页链 = 传入的 `tbl`(可能 = 子集),
/// `last_len` = 末页在本**视图**内的有效行数(= 视图 token 数 − (nb_used−1)·page)。
fn fi_tables_view(base: usize, t: usize, tbl: &[u32], page: usize) -> FiTables {
    // 视图内 token 数:本 chunk 之前覆盖的页 × page 中**属于视图的部分**
    // —— 由调用方把 base 换成"视图起点等效偏移"后即 ctx_total 口径自洽
    let ctx_total = base + t;
    let nb = tbl.len();
    let nb_used = ctx_total.div_ceil(page).min(nb).max(1);
    FiTables {
        q_cu: vec![0, t as u32],
        indices: tbl[..nb_used].to_vec(),
        indptr: vec![0, nb_used as u32],
        last_len: vec![(ctx_total - (nb_used - 1) * page) as u32],
        nb_used,
    }
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

impl<D: DeviceClient> crate::running::RunningEngine<D> {
    // ========================================================================
    // prefill 桶图路(M2b;设计 §4–§6)
    // ========================================================================

    /// **桶图命中 ⇒ 全走图路;否则返回 None(调用方走 eager 原路)。**
    ///
    /// 返回 `Some(nt)` = 末块 token;`Some(None)` = 中间块;`None` = 未命中。
    ///
    /// 三律的落地点都在 [`ChunkTables::build_bucket`] 与 `bucket_tail`:
    /// ①假行不进序列(seq_cu/GDN 表真口径)②假行池写落 dump 页 ③采样读真行(出图尾)。
    /// **单一路径执行**:桶命中就不跑 eager(否则 GDN/KV 双推进 —— §4.6)。
    async fn bucket_prefill_chunk(
        &mut self,
        ids: &[u32],
        base: usize,
        is_last: bool,
        bt_chain: &[u32],
        gdn_slot_v: usize,
    ) -> Result<Option<Option<u32>>> {
        let t_all = std::time::Instant::now();
        let t = ids.len();
        if !self.pool.paged || self.graphs.prefill.is_empty() {
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
        if self.cfg.knobs.pf_bisect
            || self.cfg.knobs.pf_stages.is_some()
            || self.probes.dflash_probe
            || self.cfg.knobs.coverage_probe
            || self.cfg.knobs.select_decode
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
            self.pool.page,
            bt_chain,
            gdn_slot_v,
            self.pool.fi_face,
            dump_slot,
        );
        // 注:FI 四件在**图路**走输入槽(`BucketGraph::replay` 里按布局长度装填)
        // —— 池里那套持久块(`StatePool.fi_tables`)是 **eager 路**的表,这里不写
        // (写了也没人读:图烘的是槽指针)。GDN 序列表不同:图烘的就是池块,
        // 下方必须写。
        // ── GDN 序列表写持久槽(**布局 nt = ceil(t_pad/BT)、cu = t_real**)──
        // 图内 `nt` 是烘死的标量(布局口径);三张表的**内容**由本写定死:
        // cu=[0,t_real] ⇒ 行区间/递推落点都在真行上(§4.3e P-6b;假行不进序列)。
        if let Some(gt) = self.pool.gdn_table_blocks() {
            let nt = t_pad.div_ceil(owl_models::kernel::GDN_BT).max(1);
            if nt > gt.nt_cap {
                return Err(owl_iface::contract::ModelError::Msg(format!(
                    "GDN 序列表超槽容量:nt={nt} > nt_cap={}",
                    gt.nt_cap
                )));
            }
            let bytes = owl_models::kernel::gdn_table_host_bytes(t, nt);
            let f32bits: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().expect("4 字节")))
                .collect();
            let face = self.graphs.session.face_mut();
            // 参数面 → COMPUTE(与紧随的图发射同流;见 graph_plan::fill_slots 注)
            face.write_block_f32_on(owl_iface::contract::UpStream::Compute, &gt.table, 0, &f32bits)
                .await?;
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
        let nt = self.bucket_tail(&root, &taps, t, ids, base, is_last).await?;
        mrec("prefill.chunk", t_all.elapsed());
        Ok(Some(nt))
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
        root: &Bytes,
        taps: &[TensorOps],
        t: usize,
        ids: &[u32],
        base: usize,
        is_last: bool,
    ) -> Result<Option<u32>> {
        let d = self.pool.dims;
        let d_model = d.hidden;
        let hidden = TensorOps::of_block(root.id, d.dtype, vec![t, d_model]);
        let hidden = &hidden;
        if !is_last {
            // 中间块:taps 编码 / mtp extend(免 lm_head [T,V] 计算与大 dtoh)
            if !taps.is_empty() {
                let ts = std::time::Instant::now();
                self.prefill_dflash_encode(taps, base).await?;
                eprintln!("[dflash] prefill encode {t} rows @{} {:?}", base, ts.elapsed());
            } else if self.spec.drafter.is_some() && t > 1 {
                self.prefill_extend(root, t - 1, ids, base).await?;
            }
            if self.probes.prefill_cksum {
                self.pf_cksum(root, base, t, d.hidden).await?;
            }
            return Ok(None);
        }
        // 末块
        if self.cfg.knobs.pf_fin_check {
            self.pf_fin_check(root, t, d_model).await?;
        }
        if !taps.is_empty() {
            let ts = std::time::Instant::now();
            self.prefill_dflash_encode(taps, base).await?;
            eprintln!("[dflash] prefill encode {t} rows @{} {:?}", base, ts.elapsed());
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
            let face = self.graphs.session.face_mut();
            eval_ops_scoped_env(last.step(), face, self.env).await?
        };
        let hb_view = TensorOps::of_block(hb.id, d.dtype, vec![1, d_model]);
        let vocab = self.model.vocab_size();
        let (b, arena3) = {
            let face = self.graphs.session.face_mut();
            let logits = self.model.embed.lm_head_matmul(&hb_view);
            let tok = owl_models::ops::argmax_f32idx(&logits, vocab, 0);
            eval_ops_scoped_env(tok.step(), face, self.env).await?
        };
        if self.spec.drafter.is_some() && t > 1 {
            self.prefill_extend(root, t - 1, ids, base).await?;
        }
        let mut buf_t = [0u8; 4];
        let mut seed_keep: Option<Bytes> = None;
        {
            let face = self.graphs.session.face_mut();
            face.dtoh(&b, &mut buf_t).await?;
            face.free(&[b.id]).await?;
            face.free(&arena3).await?;
            face.free(&arena2).await?;
            if self.spec.drafter.is_some() {
                seed_keep = Some(hb);
            } else {
                face.free(&[hb.id]).await?;
            }
        }
        if let Some(hb) = seed_keep {
            self.tspec()?.spec_seed_hidden = Some(hb);
        }
        let nt = f32::from_le_bytes(buf_t) as u32;
        mfact(self.boot_seq, "pf.last", nt as u64);
        mfact(self.boot_seq, "pf.eos", self.tok.is_eos(nt) as u64);
        Ok(Some(nt))
    }

    /// fin 校验(dtoh 一块 hidden 打点):`pf_fin_check` 走 FNV 指纹(mfact),
    /// `prefill_cksum` 走 |x| 之和(两档历史打点各自保留 —— 字段语义不同,
    /// 不能合并成一套)
    async fn pf_fin_check(&mut self, root: &Bytes, t: usize, dd: usize) -> Result<()> {
        let mut buf = vec![0u8; t * dd * 2];
        {
            let face = self.graphs.session.face_mut();
            face.dtoh(root, &mut buf).await?;
        }
        mfact(self.boot_seq, "pf.fin", fnv_bytes(&buf) & 0x7fff_ffff_ffff_ffff);
        Ok(())
    }

    /// `prefill_cksum` 档的 |x| 之和(诊断)
    async fn pf_cksum(&mut self, root: &Bytes, base: usize, t: usize, dd: usize) -> Result<()> {
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
    pub(crate) async fn prefill_chunk(
        &mut self,
        ids: &[u32],
        base: usize,
        is_last: bool,
    ) -> Result<Option<u32>> {
        let t_chunk = std::time::Instant::now();
        let mut t_ext = std::time::Duration::ZERO;
        let d = self.pool.dims;
        let t = ids.len();
        // E2b:活跃会话块链 + GDN 格(逻辑位 → 物理槽由块表换算)
        let (bt_chain, gdn_slot_v) = {
            let act = self.turns.active()?;
            let s = self.bank.sessions.session(act.session_id)?;
            // GDN 行 = **turn 域资源**(§28 行借还:行号就是账里的行号)
            (s.block_table.clone(), act.gdn_slot)
        };
        // ── prefill 桶图命中(M2b;设计 §4–§6)──
        // 命中 ⇒ 槽写(真行真值 + 假行 dump 页)+ 1 条 graph_launch + 出图尾;
        // 未命中/不可用 ⇒ 下方 eager 原路(**单一路径执行**,绝不双推进 —— §4.6)。
        if !self.graphs.prefill.is_empty() {
            match self.bucket_prefill_chunk(ids, base, is_last, &bt_chain, gdn_slot_v).await? {
                Some(nt) => return Ok(nt),
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
        let past_view = self.past_view_of(base, t);
        let tb = ChunkTables::build_view(
            ids,
            base,
            t,
            self.pool.paged,
            self.pool.page,
            &bt_chain,
            gdn_slot_v,
            self.pool.fi_face,
            self.env.attn.xqa_prefill,
            past_view.as_ref().map(|(pages, tokens)| (pages.as_slice(), *tokens)),
        );
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
        let slots_leaf = TensorOps::from_host(Dtype::F32, vec![t], &f32b(&tb.slots));
        let lens_leaf = TensorOps::from_host(Dtype::F32, vec![t], &f32b(&tb.lens));
        let gdn_slot_leaf = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[tb.gdn_slot]));
        let kv_caches = self.pool.kv_caches();
        let kvs_step: Vec<KvBuffers> = kv_caches
            .iter()
            .map(|(k, v)| KvBuffers {
                k_cache: k.clone(),
                v_cache: v.clone(),
                slots: slots_leaf.clone(),
                kv_lens: lens_leaf.clone(),
                block_tables: bt_leaf.clone(),
            })
            .collect();
        let gdn_caches = self.pool.gdn_caches();
        let gdns_step: Vec<GdnBuffers> = gdn_caches
            .iter()
            .map(|g| GdnBuffers {
                conv_q: g.conv_q.clone(),
                conv_k: g.conv_k.clone(),
                conv_v: g.conv_v.clone(),
                rec: g.rec.clone(),
                slots: gdn_slot_leaf.clone(),
            })
            .collect();
        let ids_t = TensorOps::from_host(Dtype::F32, vec![t], &f32b(&tb.ids));
        let pos_t = TensorOps::from_host(Dtype::F32, vec![t], &f32b(&tb.pos));
        let gdn_slot = gdn_slot_leaf.clone();
        // FlashInfer 表四件套(i32 设备;OWL_FLASHINFER=1 时非空,E1.5):
        // q_cu=[0,T] / indices=块链 / indptr=[0,nb] / last_len=末页有效数。
        // ⚠️ 预物化(eval 往返同步)而非 from_host 叶子:Htod 在 H2D 流、
        // FI 核在 COMPUTE 流,无跨流序 —— 异步竞态会让 FI 偶发读到半写入
        // 表(渐进腐败案,2026-10-03;09-28 upload_pinned 同病史)。
        let u32le = |v: &[u32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let fi_tensors;
        let fi = if let Some(ft) = tb.fi.as_ref() {
            let k_fi_leaf = self.pool.kv_fi_leaves();
            // P-3(2026-10-15):表件写**持久槽**(地址恒定 ⇒ 图可直接烘;
            // 旧实现每 chunk 4 次 `eval_ops(from_host)` = 4 次往返同步)。
            // 写经 `write_blocks_f32` = 单条批量 HtodChunks 命令,回执 =
            // 设备侧搬运完成的 host callback ⇒ **回执即落地**(无跨流竞态:
            // 旧注释里"预物化"要的就是这个序;批量只是把 4 次压成 1 次)。
            let fb = self
                .pool
                .fi_table_blocks()
                .ok_or_else(|| ModelError::EngineInvariant { what: "FI 表槽未分配(池几何与档位不一致)".into() })?
                .clone();
            assert!(
                ft.nb_used <= fb.nb_cap,
                "FI 页链超槽容量:nb_used={} > nb_cap={}",
                ft.nb_used,
                fb.nb_cap
            );
            let bits = |v: &[u32]| -> Vec<f32> { v.iter().map(|x| f32::from_bits(*x)).collect() };
            let w_q_cu = bits(&ft.q_cu);
            let w_idx = bits(&ft.indices);
            let w_ind = bits(&ft.indptr);
            let w_ll = bits(&ft.last_len);
            let face = self.graphs.session.face_mut();
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
            let q_cu = TensorOps::of_block(fb.q_cu.id, Dtype::U32, vec![2]);
            let indices = TensorOps::of_block(fb.indices.id, Dtype::U32, vec![ft.nb_used]);
            let indptr = TensorOps::of_block(fb.indptr.id, Dtype::U32, vec![2]);
            let last_len = TensorOps::of_block(fb.last_len.id, Dtype::U32, vec![1]);
            let (kcs, vcs): (Vec<TensorOps>, Vec<TensorOps>) =
                k_fi_leaf.iter().map(|(k, v)| (k.clone(), v.clone())).unzip();
            fi_tensors = (kcs, vcs, q_cu, indices, indptr, last_len);
            Some(FiPrefillCtx {
                kcs: &fi_tensors.0,
                vcs: &fi_tensors.1,
                q_cu: &fi_tensors.2,
                indices: &fi_tensors.3,
                indptr: &fi_tensors.4,
                last_len: &fi_tensors.5,
            })
        } else {
            None
        };
        // P1′(2026-10-14):**cu_seqlens [0,T] 每 chunk 只上传一次**。
        // 原先 attn/gdn 层各自 `from_host([2])` 兜底(16 attn + 32 gdn = **48 次
        // 8 字节 H2D / chunk**)—— 每次 8B 上传的**往返延迟 ~3.5ms**(issue 侧仅
        // 16µs)⇒ ~170ms/chunk 白烧。注入面本就存在(verify 图先例 E5-M4:
        // `ctx.seq_cu`),eager prefill 只是没接。
        let seq_cu = TensorOps::from_host(Dtype::F32, vec![2], &f32b(&tb.seq_cu));
        // ── B 轴选择层:Q 槽接入 ──
        // 只在**块边界 chunk**采(`(base+t) % page == 0`):分析期索引账才完整
        // (§8.4)。
        // 🔴 **两扇门**(2026-10-19 拆分):**生产面**(`select_decode`)也要 Q 面 ——
        // 它才是打分的输入;`coverage_probe` 只是诊断面(额外算真实质量)。
        // 拆之前两条耦合在一起 ⇒ 低显存档想开功能就**必须**背诊断的账。
        // 缺省两门全关 ⇒ 槽恒 None、零开销。
        let cov_sink: Option<crate::exec::coverage::QSink> = ((self.cfg.knobs.coverage_probe
            || self.cfg.knobs.select_decode)
            && self.pool.paged
            && (base + t).is_multiple_of(self.pool.page))
        .then(|| crate::exec::coverage::QSink::new(Vec::new()));
        // ── XQA prefill 表件(`dispatch.xqa_prefill`;缺省关)──
        // 三张 U32 位型表(页链/整请求长/因果位掩码)与 FI 四件套同路:
        // **预物化**(eval_ops 往返同步)而非 from_host 叶子 —— Htod 在 H2D
        // 流、核在 COMPUTE 流,无跨流序 ⇒ 异步竞态会让核偶发读到半写入表
        // (2026-10-03 渐进腐败案同病史,见上方 FI 表注)。
        // semaphores/scratch 两件取池里的真块:单块几何(`XqaPrefillRuntime`
        // 强制 `nbSubSeq=1`)下核**结构上不解引用**它们,传真块只为"不隐瞒"。
        let xqa_tensors;
        let xqa_pf = if self.env.attn.xqa_prefill {
            match (self.pool.xqa_ctx(), tb.xqa.as_ref()) {
                (Some(px), Some(xt)) => {
                    let face = self.graphs.session.face_mut();
                    let pt_b = owl_models::interpreters::eval_ops(
                        TensorOps::from_host(
                            Dtype::U32,
                            vec![xt.page_table.len()],
                            &u32le(&xt.page_table),
                        )
                        .step(),
                        face,
                    )
                    .await?;
                    let sl_b = owl_models::interpreters::eval_ops(
                        TensorOps::from_host(Dtype::U32, vec![1], &u32le(&[xt.seq_len])).step(),
                        face,
                    )
                    .await?;
                    let mk_b = owl_models::interpreters::eval_ops(
                        TensorOps::from_host(Dtype::U32, vec![xt.mask.len()], &u32le(&xt.mask))
                            .step(),
                        face,
                    )
                    .await?;
                    xqa_tensors = (
                        TensorOps::of_block(pt_b.id, Dtype::U32, vec![xt.page_table.len()]),
                        TensorOps::of_block(sl_b.id, Dtype::U32, vec![1]),
                        TensorOps::of_block(mk_b.id, Dtype::U32, vec![xt.mask.len()]),
                    );
                    Some(owl_models::module::XqaVerifyCtx {
                        page_table: xqa_tensors.0.clone(),
                        seq_lens: xqa_tensors.1.clone(),
                        mask: xqa_tensors.2.clone(),
                        semaphores: px.semaphores.clone(),
                        scratch: px.scratch.clone(),
                        q_seq_len: t,
                        max_seq_len: px.max_seq_len,
                        sm_count: px.sm_count,
                    })
                }
                _ => None,
            }
        } else {
            None
        };
        // 诊断(2026-10-14 A13):`[probes] pf_fence = true` ⇒ from_host 叶子(pos/slots/
        // lens/id/gdn_slot)建完后插一道流栅栏 —— 判「H2D 叶子与 COMPUTE 核
        // 无跨流序」竞态(仓内 2026-10-03 FI 表同病史:预物化修法)
        if self.probes.pf_fence {
            self.graphs.session.face_mut().sync().await?;
        }
        let mut ctx = ForwardCtx::model_prefill(
            t, &pos_t, &kvs_step, &self.rope, &gdns_step, &slots_leaf, &lens_leaf, &gdn_slot,
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
        let gdn_nt = t.div_ceil(owl_models::kernel::GDN_BT).max(1);
        let gdn_tbl_leaf;
        let gdn_tables = match self.pool.gdn_table_blocks() {
            Some(gt) => {
                if gdn_nt > gt.nt_cap {
                    return Err(owl_iface::contract::ModelError::Msg(format!(
                        "GDN 序列表超槽容量:nt={gdn_nt} > nt_cap={}(seq_tokens/chunk 配置不一致)",
                        gt.nt_cap
                    )));
                }
                let bytes = owl_models::kernel::gdn_table_host_bytes(t, gdn_nt);
                let f32bits: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().expect("4 字节")))
                    .collect();
                let face = self.graphs.session.face_mut();
                face.write_block_f32(&gt.table, 0, &f32bits).await?;
                gdn_tbl_leaf = TensorOps::of_block(
                    gt.table.id,
                    Dtype::U32,
                    vec![owl_models::kernel::gdn_table_bytes(gdn_nt) / 4],
                );
                Some(owl_models::module::GdnTableSlot { table: &gdn_tbl_leaf, nt: gdn_nt })
            }
            None => None,
        };
        ctx.gdn_tables = gdn_tables;
        ctx.xqa = xqa_pf;
        // ── B 轴覆盖率探针(Q-2 第一段):Q 槽接入(ctx 建完即挂)──
        if let Some(s) = cov_sink.as_ref() {
            ctx.q_sink = Some(s);
        }
        ctx.env = self.env;
        ctx.trace_gate = self.cfg.knobs.probes.trace_gate;
        let _vocab = self.model.vocab_size();
        // E5-DF4 逐层 bisect(OWL_PF_BISECT;非 dflash 路径):单遍 multi
        // eval 产出 48 层 taps + fin —— **fin 块直接作为主路输出**(bisect
        // 前向有 GDN 状态副作用,双执行会双推进;单执行 + 逐层 checksum
        // → metrics `pf.c{base}.l{i}`,双 boot/双跑 diff = 首个发散层)。
        // fin block-leaf 下游 eval = 零操作透传。
        let pf_bisect = self.cfg.knobs.pf_bisect && self.spec.dflash_tap_count == 0;
        // GDN 状态 checksum(pre/post;崩坏排查:状态脏 = reset 病,状态净
        // 而输出异 = 核病)。rec [slots,nv,kd,vd] f32 全量 dtoh ≈ 3.1MB/层。
        let pf_state_check = pf_bisect;
        // 指定层五站 stage tap(OWL_PF_STAGES=层号;配合 BISECT 定位层内核)
        let pf_stage_layer: Option<usize> = self.cfg.knobs.pf_stages;
        let mut bisect_fin_block: Option<(TensorOps, owl_iface::contract::Bytes)> = None;
        let mut _bisect_fin_blk: Option<owl_iface::contract::Bytes> = None;
        if pf_bisect {
            let gdn_leaves = self.pool.gdn_caches();
            if pf_state_check {
                for (j, g) in gdn_leaves.iter().enumerate() {
                    for (sn, buf, nb) in [
                        ("convq", &g.conv_q, self.pool.gdn_slots * d.nk * d.hk * 3),
                        ("convk", &g.conv_k, self.pool.gdn_slots * d.nv * d.hv * 3),
                        ("convv", &g.conv_v, self.pool.gdn_slots * d.nv * d.hv * 3),
                        ("rec", &g.rec, self.pool.gdn_slots * d.nv * d.hk * d.hv),
                    ] {
                        let mut sb = vec![0u8; nb * 4];
                        {
                            let face = self.graphs.session.face_mut();
                            let blk = owl_models::interpreters::eval_ops(buf.step(), face).await?;
                            face.dtoh(&blk, &mut sb).await?;
                        }
                        mfact(
                            self.boot_seq,
                            &format!("pf.c{}.pre.g{j}.{sn}", base),
                            fnv_bytes(&sb) & 0x7fff_ffff_ffff_ffff,
                        );
                    }
                }
            }
            let _ids_all: Vec<usize> = (0..self.model.layers.len()).collect();
            // GDN 层内打点(M4 gdn_tap 现成):每 GDN 层 8 件
            // (q/k/v raw + q_n/k_n/v_c/g/beta)→ 定位 qkv-proj/l2norm/conv/gating
            // GDN 层内 8 件打点(gdn_tap)已启用:层六站展开 + 8 件随层带出
            // (旧案:gtap 节点直挂 multi 曾现节点/块尺寸错配,改由
            // forward_bisect_stages 随 mixer 调用自然求值,不直挂 multi)
            ctx.gdn_tap = Some(std::rc::Rc::new(std::cell::RefCell::new(Vec::new())));
            let (fin, nodes) =
                self.model
                    .prefill_bisect(&ids_t, &ctx, pf_stage_layer);
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
                let face = self.graphs.session.face_mut();
                owl_models::interpreters::eval_ops_multi_env(&refs, face, self.env).await?
            };
            for (i, r) in roots.iter().enumerate() {
                let n: usize = r.shape().iter().product();
                let mut b = vec![0u8; n * 2];
                {
                    let face = self.graphs.session.face_mut();
                    let res = face.dtoh(&outs[i], &mut b).await;
                    if let Err(e) = res {
                        owl_shared::log_debug!("pf-bisect", "dtoh 失败 i={i} name={} shape={:?} blk_bytes={} err={e}", names[i], r.shape(), b.len());
                        return Err(e);
                    }
                }
                mfact(
                    self.boot_seq,
                    &format!("pf.c{base}.{}", names[i]),
                    fnv_bytes(&b) & 0x7fff_ffff_ffff_ffff,
                );
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
                        ("convq", &g.conv_q, self.pool.gdn_slots * d.nk * d.hk * 3),
                        ("convk", &g.conv_k, self.pool.gdn_slots * d.nv * d.hv * 3),
                        ("convv", &g.conv_v, self.pool.gdn_slots * d.nv * d.hv * 3),
                        ("rec", &g.rec, self.pool.gdn_slots * d.nv * d.hk * d.hv),
                    ] {
                        let mut sb = vec![0u8; nb * 4];
                        {
                            let face = self.graphs.session.face_mut();
                            let blk = owl_models::interpreters::eval_ops(buf.step(), face).await?;
                            face.dtoh(&blk, &mut sb).await?;
                        }
                        mfact(
                            self.boot_seq,
                            &format!("pf.c{}.post.g{j}.{sn}", base),
                            fnv_bytes(&sb) & 0x7fff_ffff_ffff_ffff,
                        );
                    }
                }
            }
            owl_shared::log_debug!("pf-bisect", "chunk base={base} t={t} 层 checksum 已记");
        }
        let _face = self.graphs.session.face_mut();
        let vocab = self.model.vocab_size();
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
            } else if self.spec.dflash_tap_count > 0 {
                let (h, tp) = self.model.tapped_hidden(&ids_t, &ctx, &[5, 19, 33, 47, 61]);
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
                (self.model.last_hidden(&ids_t, &ctx), None)
            };
            let d_model = hidden_full.shape()[1];
            let ld = live_diag_begin();
            let r_eval = self.eval_root_and_q(&hidden_full, cov_sink.as_ref()).await;
            live_diag_end(ld, base, t);
            let (hfb, arena1, cov_q) = r_eval?;
            // Q 回读(探针档;设备块当场 free;分析在块边界事件后做)
            self.dump_hidden_row(&hfb, base, t, d_model).await?;
            if let Some(qs) = cov_q {
                self.coverage_take_q(qs, base, t).await?;
            }
            // E5 崩坏排查·轻探针(OWL_PF_FIN_CHECK;bisect 900 dtoh 强掩蔽
            // 0/10,本探针仅 1 dtoh —— 近零扰判决:fin 翻 = 前向内部腐坏;
            // fin 绿而 pf.last 翻 = lm_head/argmax 面腐坏)
            if self.cfg.knobs.pf_fin_check && !pf_bisect {
                let n_fin = t * d_model;
                let mut fb = vec![0u8; n_fin * 2];
                {
                    let face = self.graphs.session.face_mut();
                    face.dtoh(&hfb, &mut fb).await?;
                }
                mfact(
                    self.boot_seq,
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
                self.prefill_dflash_encode(&tap_views, base).await?;
                // 探针:concat 父块整块落盘(python 对拍 taps;真块 dtoh)
                if self.probes.dflash_probe && !self.spec.mem_dumped2 {
                    self.spec.mem_dumped2 = true;
                    let face = self.graphs.session.face_mut();
                    let mut rbuf = vec![0u8; (n_tp + 1) * t * d_model * 2];
                    face.dtoh(&hfb, &mut rbuf).await?;
                    if let Err(e) = owl_shared::file_loader::write("/tmp/owl_dflash_root.bin", &rbuf) {
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
                let face = self.graphs.session.face_mut();
                eval_ops_scoped_env(last.step(), face, self.env).await?
            };
            let hb_view = TensorOps::of_block(hb.id, d.dtype, vec![1, d_model]);
            let (b, arena3) = {
                let face = self.graphs.session.face_mut();
                let logits = self.model.embed.lm_head_matmul(&hb_view);
                // 崩坏排查:末行 logits top-2(id+值)——近局翻转判别
                if self.probes.dflash_probe {
                    let bs_seq = self.boot_seq;
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
                let tok = owl_models::ops::argmax_f32idx(&logits, vocab, 0);
                eval_ops_scoped_env(tok.step(), face, self.env).await?
            };
            if self.spec.drafter.is_some() && t > 1 {
                // prefill extend(E5-M5):本 chunk 行 0..T-1 的 mtp KV
                // (行 T-1 的配对需下一 chunk 首 token,由其 extend 补;
                // 末位 P-1 由首轮 propose_first 的 extend 行补)
                self.prefill_extend(&hfb, t - 1, ids, base).await?;
            }
            let buf: [u8; 4];
            let mut seed_keep: Option<owl_iface::contract::Bytes> = None;
            {
                let face = self.graphs.session.face_mut();
                let mut buf_t = [0u8; 4];
                face.dtoh(&b, &mut buf_t).await?;
                face.free(&[b.id]).await?;
                face.free(&arena3).await?;
                face.free(&arena2).await?;
                if self.spec.drafter.is_some() {
                    seed_keep = Some(hb);
                } else {
                    face.free(&[hb.id]).await?;
                }
                face.free(&[hfb.id]).await?;
                face.free(&arena1).await?; // E2a:中间块归池(每 chunk 零净增)
                buf = buf_t;
            }
            // seed 块归 turn 域 spec 状态(face 借位结束后落地)
            if let Some(hb) = seed_keep {
                self.tspec()?.spec_seed_hidden = Some(hb);
            }
            let nt = f32::from_le_bytes(buf) as u32;
            let bs = self.boot_seq;
            mfact(bs, "pf.last", nt as u64);
            mfact(bs, "pf.eos", self.tok.is_eos(nt) as u64);
            // B7(2026-10-15):末块也记 `prefill.chunk`(中间块/桶路都有;此处缺行
            // 会让图表按"chunk 数"口径对不上)
            mrec("prefill.chunk", t_chunk.elapsed());
            Ok(Some(nt))
        } else {
            // 中间块:last_hidden 根(状态推进完整;lm_head 免算)。
            // DFlash2:同构 concat 单根(taps 随块回收前切片 encode)
            let (tree, prefill_taps) = if pf_bisect {
                let (decl, blk) = bisect_fin_block.take().ok_or_else(|| ModelError::EngineInvariant { what: "bisect fin(非末)块缺席(诊断档口径)".into() })?;
                _bisect_fin_blk = Some(blk);
                (decl, None)
            } else if self.spec.dflash_tap_count > 0 {
                let (h, tp) = self.model.tapped_hidden(&ids_t, &ctx, &[5, 19, 33, 47, 61]);
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
                (self.model.last_hidden(&ids_t, &ctx), None)
            };
            let ld = live_diag_begin();
            let r_eval = self.eval_root_and_q(&tree, cov_sink.as_ref()).await;
            live_diag_end(ld, base, t);
            let (b, arena, cov_q) = r_eval?;
            self.dump_hidden_row(&b, base, t, d.hidden).await?;
            if let Some(qs) = cov_q {
                self.coverage_take_q(qs, base, t).await?;
            }
            {
                let face = self.graphs.session.face_mut();
                let esz = if d.dtype == Dtype::F16 { 2 } else { 4 };
                // DFlash2:b = concat 根 [n_tp·t + t, hidden](行 0 块 =
                // hidden,taps 同块后随)—— dtoh 整父块契约必须等大读
                // (曾用 t·hidden → 多 chunk DFlash2 首跑即炸;MTP 无 taps
                // 时 n_tp=0 同式兼容)
                let rows_b = if let Some(tp) = &prefill_taps { tp.0.len() + 1 } else { 1 } * t;
                let mut buf = vec![0u8; rows_b * d.hidden * esz];
                face.dtoh(&b, &mut buf).await?;
                if self.probes.prefill_cksum {
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
                    self.prefill_dflash_encode(&tap_views, base).await?;
                } else if self.spec.drafter.is_some() && t > 1 {
                    let ts = std::time::Instant::now();
                    self.prefill_extend(&b, t - 1, ids, base).await?;
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

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
}

impl ChunkInputs {
    /// 从池的层视图 + 本 chunk 的 host 表构造(纯 host 叶子;无设备往返)。
    pub(crate) fn build(
        pool: &crate::state::StatePool,
        tb: &ChunkTables,
        t: usize,
        bt_leaf: TensorOps,
    ) -> Self {
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
    /// **建 FI 表时用的 `ctx_total`**(= `base_for_len + t`;单源)。
    ///
    /// 为什么必须带出来:模型层喂核的 `ctx_total` 默认是 `ctx_base + tokens`
    /// (**绝对**口径)。子集档下两者差 `base − view_tokens` ⇒ 核会越过
    /// `indptr` 读越界页号、静默喂垃圾(2026-10-28 定谳的长上下文塌陷)。
    /// `prefill_chunk_enqueue` 用它做**硬断**,把这条口径错配焊死。
    pub fi_ctx: Option<usize>,
}

/// FlashInfer 表四件套(U32 位型;`nb_used` = 本 chunk 实际覆盖页数)
pub(crate) struct FiTables {
    pub q_cu: Vec<u32>,
    pub indices: Vec<u32>,
    pub indptr: Vec<u32>,
    pub last_len: Vec<u32>,
    pub nb_used: usize,
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
        past: Option<(&[u32], usize)>,
        positions: Option<&[[i32; 3]]>,
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
        // **视图账逐 chunk 取证**(2026-10-28;`--log-level debug` 可见):
        // 稀疏 prefill 的硬塌(collapse)要逐 chunk diff 才能定位 —— 一行里把
        // "链长 vs 记账长度 vs FI 四件套 vs 本 chunk 写入槽"全摆出来。
        owl_shared::log_debug!(
            "view-acc",
            "base={base} t={t} paged={paged} page={page} view={:?} view_tokens={} ctx_total={} \
             need={} last_len={:?} idx_first={:?} idx_last={:?} lens_first={:?} lens_last={:?} \
             slot_first={:?} slot_last={:?} bt_len={}",
            past.map(|(p, n)| (p.len(), n)),
            base_for_len,
            base_for_len + t,
            (base_for_len + t).div_ceil(page.max(1)).max(1),
            fi.as_ref().map(|f| f.last_len.clone()),
            fi.as_ref().map(|f| f.indices.first().copied()),
            fi.as_ref().map(|f| f.indices.last().copied()),
            lens.first().copied(),
            lens.last().copied(),
            slots.first().copied(),
            slots.last().copied(),
            bt_chain.len()
        );
        Ok(Self {
            fi_ctx: fi.as_ref().map(|_| base_for_len + t),
            ids: ids.iter().map(|v| *v as f32).collect(),
            pos: match positions {
                Some(p) => {
                    if p.len() < base + t {
                        return Err(ModelError::Msg(format!(
                            "ChunkTables: 位置表 {} 短于 [{}..{})",
                            p.len(), base, base + t
                        )));
                    }
                    // `ChunkTables.pos` 是**标量 rope 槽**(长度 t);m-RoPE
                    // 的三轴表由 VLM 注入面覆盖(`ctx.pos = vlm_pos`)。
                    p[base..base + t].iter().map(|v| v[0] as f32).collect()
                }
                None => seq(base, t),
            },
            slots,
            lens,
            gdn_slot: gdn_slot_v as f32,
            seq_cu: vec![0.0, t as f32],
            fi,
        })
    }
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
    /// 本 turn 的逐 token 绝对位置(prompt 坐标;leave-gap 含洞)
    pub(crate) positions: &'a [[i32; 3]],
    pub(crate) gdn_slot: usize,
    /// host 账房(草稿臂;peer 路 None)
    pub(crate) bank: Option<&'a mut SessionBank>,
    /// **B 轴运行态**(host 单份;peer 路 None —— 覆盖率探针只在主 rank 采集)。
    /// 探针要的 `(prof, cov)` 通过 `analysis_pair()` 一次借出。
    pub(crate) sel: Option<&'a mut crate::select::SelectRuntime>,
    /// **部分驻留的子集页链**(host 半算好传入:`past_view_of` 依赖
    /// `knobs.select_partial` 与 `sel_view` —— 都是 host 面)
    pub(crate) past_view: Option<(Vec<u32>, usize)>,
    /// 本边界是否需要 Q 面(完整召回/诊断);false = 廉价边界,省 Q D2H+逆RoPE。
    pub(crate) q_needed: bool,
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
            past_view.as_ref().map(|(pages, tokens)| (pages.as_slice(), *tokens)),
            Some(host.positions),
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
        let cov_sink: Option<crate::exec::coverage::QSink> = (host.q_needed
            && self.pool.geom.paged
            && (base + t).is_multiple_of(self.pool.geom.page))
        .then(|| crate::exec::coverage::QSink::new(Vec::new()));
        let mut ctx = ForwardCtx::model_prefill(
            t, &ci.pos_t, &ci.kvs_step, &self.boot.rope, &ci.gdns_step, &ci.slots_leaf, &ci.lens_leaf, &ci.gdn_slot,
            // A13(2026-10-14):chunked/scalar 臂的 rec 态寻址吃这个 host 标量
            // —— 必须喂本会话真槽(曾恒 0 = 读写别人格 → 长 prompt 塌陷)
            Some(gdn_slot_v),
            base, fi,
        );
        // 🔴 **FI 的 plan ctx 必须是“视图口径”,不是绝对位置**(2026-10-28 定谳)。
        //
        // 子集档(`prefill_view = always` / 超池后被迫稀疏)下,给 FI 的页链只有
        // `view_tokens 页 + 本 chunk 页`,而 `build_view` 建的 `indptr/last_len` 也按
        // `ctx_total = view_tokens + t` 算。但注意力层默认喂 `ctx_base + tokens`
        // = **绝对** base + t —— 两者差 `base − view_tokens`(= 被丢块数×页),
        // 核据此认为链更长 ⇒ 越过 `indptr` 读**越界页号** ⇒ 注意力喂垃圾。
        //
        // 实测征状(每一条都对得上):同样的 ws 下 prefill 走全链(`on-hole`)完全正常;
        // 一旦视图是真子集(哪怕只丢 6 块)⇒ 任何文档问题都退到 `:` 后 EOS;
        // 而宿主侧对照(`scheme_compare`)**拧不到这里**(它不过核)。
        if let Some((_, view_tokens)) = past_view.as_ref() {
            ctx.fi_plan_ctx = Some(view_tokens + t);
        }
        // **硬断(单源)**:喂 FI 的 plan ctx 必须等于建表用的 `ctx_total`。
        // 缺了它,`ctx_base` 的绝对口径会静静顶上去 —— 2026-10-28 的长上下文
        // 塌陷(64k 起 out=1、108k 修复后正常)就是这个错配,而且**宿主侧对照
        // 照不到**(`scheme_compare` 不过核)。这里改成响亮拒,不再靠人眼核表。
        if let Some(want) = tb.fi_ctx {
            let got = ctx.fi_plan_ctx.unwrap_or(ctx.ctx_base + t);
            if got != want {
                return Err(owl_iface::contract::ModelError::EngineInvariant {
                    what: format!(
                        "prefill FI 的 ctx_total 口径错配:喂核 {got} ≠ 建表 {want} \
                         (base={base} t={t} view={:?})—— 核会越 indptr 读越界页号",
                        past_view.as_ref().map(|(p, n)| (p.len(), *n))
                    ),
                });
            }
        }
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
                .write_gdn_tables(face, t, nt)
                .await?
        };
        let gdn_tables = gdn_tbl.as_ref().map(|g| owl_models::module::GdnTableSlot {
            table: &g.leaf,
            nt: g.nt,
        });
        ctx.gdn_tables = gdn_tables;
        // ── **VLM:本 chunk 的 m-RoPE / 图像注入**(2026-10-20)──
        // 行号表/余弦表按 chunk 行号切;`has_image=false` 的块只装 rope
        // (image_scatter 不必跑全 -1 表)。塔输出 `merged` 是 turn 域设备叶。
        let (vlm_cos, vlm_sin, vlm_idx, vlm_pos);
        if let Some(v) = host.vlm {
            let (cos, sin, idx, pos, has_image) = v.ins.chunk_tables(base, t);
            vlm_cos = cos;
            vlm_sin = sin;
            vlm_idx = idx;
            vlm_pos = pos;
            ctx = ctx.with_vlm_rope(&vlm_cos, &vlm_sin);
            // cos/sin 是本 chunk 的行表 ⇒ rope 的行号也必须是本 chunk 局部号
            ctx.pos = Some(&vlm_pos);
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
        let (hidden_full, prefill_taps) = if self.spec_rt.dflash_tap_count > 0 {
                let (h, tp) = self.boot.model.tapped_hidden(&ci.ids_t, &ctx, &self.spec_rt.dflash_tap_ids);
                let mut c = TensorOps::call(owl_models::ops::SemanticKernel::Concat).arg(&h);
                for t in &tp {
                    c = c.arg(t);
                }
                for _ in tp.len()..(owl_models::kernel::CONCAT_ROWS_SLOTS - 1) {
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
            let ld = live_diag_begin(host.probes.live_diag);
            let r_eval = self.eval_root_and_q(&hidden_full, cov_sink.as_ref()).await;
            live_diag_end(ld, base, t);
            let (hfb, arena1, cov_q) = r_eval?;
            // Q 回读(探针档;设备块当场 free;分析在块边界事件后做)
            if let Some(qs) = cov_q {
                if let Some(sel) = host.sel.as_deref_mut() {
                    let (pl, pc) = sel.analysis_pair();
                    let w_tok = host.knobs.select_query_tokens.max(1);
                    self.coverage_take_q(qs, base, t, w_tok, pl, pc).await?;
                }
            } else if !host.q_needed
                && self.pool.geom.paged
                && (base + t).is_multiple_of(self.pool.geom.page)
            {
                if let Some(sel) = host.sel.as_deref_mut() {
                    let (_pl, pc) = sel.analysis_pair();
                    self.coverage_mark_cheap(base, t, pc)?;
                }
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
                let logits = self.boot.model.embed.lm_head_matmul(&hb_view, &env.gguf);
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
            let (tree, prefill_taps) = if self.spec_rt.dflash_tap_count > 0 {
                let (h, tp) = self.boot.model.tapped_hidden(&ci.ids_t, &ctx, &self.spec_rt.dflash_tap_ids);
                let mut c = TensorOps::call(owl_models::ops::SemanticKernel::Concat).arg(&h);
                for tt in &tp {
                    c = c.arg(tt);
                }
                for _ in tp.len()..(owl_models::kernel::CONCAT_ROWS_SLOTS - 1) {
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
            let ld = live_diag_begin(host.probes.live_diag);
            let r_eval = self.eval_root_and_q(&tree, cov_sink.as_ref()).await;
            live_diag_end(ld, base, t);
            let (b, arena, cov_q) = r_eval?;
            if let Some(qs) = cov_q {
                if let Some(sel) = host.sel.as_deref_mut() {
                    let (pl, pc) = sel.analysis_pair();
                    let w_tok = host.knobs.select_query_tokens.max(1);
                    self.coverage_take_q(qs, base, t, w_tok, pl, pc).await?;
                }
            } else if !host.q_needed
                && self.pool.geom.paged
                && (base + t).is_multiple_of(self.pool.geom.page)
            {
                if let Some(sel) = host.sel.as_deref_mut() {
                    let (_pl, pc) = sel.analysis_pair();
                    self.coverage_mark_cheap(base, t, pc)?;
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
                // 勘误(2026-10-27):原 `prefill_cksum` 探针已退役 —— 那段
                // 无条件 dtoh 在 2026-10-17 已收进门内,现在整段去掉(零回读)。
                let _ = (&face, rows_b, esz);
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
fn live_diag_begin(on: bool) -> bool {
    if !on {
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
    owl_shared::log_debug!(
        "live-diag",
        "chunk@{base} t={t}:影子 arena 高水位 = {} MiB",
peak >> 20
    );
    let rep = owl_shared::slab_hint::take_probe_report();
    if !rep.is_empty() {
        owl_shared::log_debug!(
            "live-diag",
            "峰值构成:{rep}");
    }
    owl_shared::log_debug!(
        "live-diag",
        "回收出口:{}",
owl_shared::slab_hint::live_diag::take_report()
    );
    owl_shared::log_debug!(
        "live-diag",
        "引用计数推送 {} 次",
owl_shared::slab_hint::refs_pushed()
    );
}

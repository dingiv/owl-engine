//! 块式 prefill(树根装配与执行;taps/encode/BISECT 取证)
use super::*;

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
        let fi = fi_face.then(|| {
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
        });
        Self {
            ids: ids.iter().map(|v| *v as f32).collect(),
            pos: seq(base, t),
            slots,
            lens: seq(base + 1, t),
            gdn_slot: gdn_slot_v as f32,
            seq_cu: vec![0.0, t as f32],
            fi,
            xqa: xqa_face.then(|| XqaTables {
                page_table: bt_chain.to_vec(),
                seq_len: (base + t) as u32,
                mask: causal_mask_words(t),
            }),
        }
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

impl<D: DeviceClient> crate::running::RunningEngine<D> {
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
            let act = self.turns.active.as_ref().expect("active 已保证");
            let s = self.bank.sessions.get(act.session_id).expect("账在");
            (s.block_table.clone(), s.gdn_slot)
        };
        // ── host 侧输入表**唯一取值点**(见 [`ChunkTables`])──
        let tb = ChunkTables::build(
            ids,
            base,
            t,
            self.pool.paged,
            self.pool.page,
            &bt_chain,
            gdn_slot_v,
            self.pool.fi_face,
            self.env.attn.xqa_prefill,
        );
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
                block_tables: self.pool.bt_leaf_flat(),
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
                .expect("FI 槽未分配(pool.fi_face 档恒有)")
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
            face.write_blocks_f32(&[
                (&fb.q_cu, 0, w_q_cu.as_slice()),
                (&fb.indices, 0, w_idx.as_slice()),
                (&fb.indptr, 0, w_ind.as_slice()),
                (&fb.last_len, 0, w_ll.as_slice()),
            ])
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
        ctx.xqa = xqa_pf;
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
                let (decl, blk) = bisect_fin_block.take().expect("bisect fin");
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
            let (hfb, arena1) = {
                let face = self.graphs.session.face_mut();
                eval_ops_scoped_env(hidden_full.step(), face, self.env).await?
            };
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
                    top.select_nth_unstable_by(1, |a, b| b.0.partial_cmp(&a.0).unwrap());
                    top.truncate(2);
                    top.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
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
                self.tspec().spec_seed_hidden = Some(hb);
            }
            let nt = f32::from_le_bytes(buf) as u32;
            let bs = self.boot_seq;
            mfact(bs, "pf.last", nt as u64);
            mfact(bs, "pf.eos", self.tok.is_eos(nt) as u64);
            Ok(Some(nt))
        } else {
            // 中间块:last_hidden 根(状态推进完整;lm_head 免算)。
            // DFlash2:同构 concat 单根(taps 随块回收前切片 encode)
            let (tree, prefill_taps) = if pf_bisect {
                let (decl, blk) = bisect_fin_block.take().expect("bisect fin(非末)");
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
            let (b, arena) = {
                let face = self.graphs.session.face_mut();
                eval_ops_scoped_env(tree.step(), face, self.env).await?
            };
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

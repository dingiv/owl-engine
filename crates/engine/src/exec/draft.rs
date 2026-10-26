//! 草稿器面(DFlash2 单图 encode+propose / eager / MTP 链 / verify 前向)
use super::*;
use super::lookup::{fuse_lookup, suffix_lookup, LookupCfg, LookupProposal};
use super::mtp_window::{MtpWindowCfg, MtpWindowPlan};
use owl_models::ModelError;

impl<D: DeviceClient> crate::running::RunningEngine<D> {
    /// DFlash2 propose 单图一轮(E5-DF4):输入槽装填(tokens/anchor/
    /// enc_pos/enc_slots/prop_pos/prop_slots/kv_len)→ replay → drafts
    /// 回读(select 缓冲前 7)。bt 覆盖自保同 eager 路(ensure + write_bt
    /// 同步律);图不消费 bt(草稿池 slots 直写),但槽号换算需要块链。
    pub(crate) async fn dflash_graph_round(
        &mut self,
        m: usize,
        pos: usize,
        bonus: u32,
        lookup: Option<&LookupProposal>,
    ) -> Result<Vec<u32>> {
        let t_build = std::time::Instant::now();
        let fp = pos + m + 1;
        let depth = self.dev.spec_rt.depth;
        let block = depth + 1;
        self.tspec()?.spec_q_host = None;
        let sampler_on = self.turns.active()?.sampler_on;
        let (q_temp, q_seed) = if sampler_on {
            let act = self.turns.active()?;
            (
                act.sampler.temp,
                act.seed.unwrap_or_else(|| act.id.wrapping_mul(0x9E37_79B9_7F4A_7C15)),
            )
        } else {
            (0.0, 0)
        };
        let _page = self.dev.pool.geom.page;
        {
            let act = self.turns.active()?;
            let s = self.bank.sessions.session_mut(act.session_id)?;
            let before = s.block_table.len();
            // FIXME(store-hook): H2 槽兜底旁路(DFlash2 target 预增长);
            // 应改为 Acquire(Grow) 预留后纯分配。
            self.bank.rel.leg.ensure_for_len(&mut s.block_table, fp + block)?;
            if s.block_table.len() != before {
                let sid = act.session_id;
                self.write_bt(sid).await?;
            }
        }
        // 草稿池寻址一律经 StatePool 权威(slot≥1 案配套;分派点零推槽)
        // ring(2026-10-22):逻辑位 % ring_slots;captured Copy ⇒ 不借 self
        let ring_slots = self.dev.pool.draft_ring_slots();
        let slot_at = |p: usize| {
            if ring_slots == 0 { p as f32 } else { (p % ring_slots) as f32 }
        };
        let _f32b = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|f| f.to_le_bytes()).collect() };
        // mask token 走 spec 运行态(单源:GGUF 草稿未导出该 KV ⇒ 家族兜底值已入账)
        let mask = self.dev.spec_rt.dflash_mask_token as f32;
        let tokens_f: Vec<f32> = (0..block)
            .map(|i| if i == 0 { bonus as f32 } else { mask })
            .collect();
        let enc_pos: Vec<f32> = (0..block).map(|i| (pos + i) as f32).collect();
        let enc_slots: Vec<f32> = (0..block).map(|i| slot_at(pos + i)).collect();
        // 逐行可见长(含自身;eager dflash_encode 同式 —— AL 劣化案修复)
        let enc_kv_lens: Vec<f32> = (0..block).map(|i| (pos + i + 1) as f32).collect();
        let prop_pos: Vec<f32> = (0..block).map(|i| (fp + i) as f32).collect();
        let prop_slots: Vec<f32> = (0..block).map(|i| slot_at(fp + i)).collect();
        // propose 噪声块逐行可见长(eager dflash_propose 同式 fp+1+i)
        let prop_kv_lens: Vec<f32> = (0..block).map(|i| (fp + 1 + i) as f32).collect();
        let kv_len = vec![(fp + block) as f32];
        let anchor = vec![bonus as f32];
        // 档位门(实际图在下面按需再借;此处只判在场,故不持绑定)
        if self.dev.graphs.dflash.is_none() {
            return Err(ModelError::EngineInvariant { what: "dflash_graph 缺席(档位不一致)".into() });
        }
        let prof = self.boot.probes.step_profile;
        // **归因探针(2026-10-18 propose 立案;仅 prof 档)**:发射前先把 COMPUTE 流
        // 排干 ⇒ 把"本相位墙钟"拆成三块,互不污染:
        //   ① `drain`  = 排水耗时(上一相/回滚/fold 的残余 GPU 工作 —— 首读一直在吸收它,
        //                而它**不是 propose 的成本**);
        //   ② `first`  = 排水后**首次**发射+读 = 本图真实首执行 GPU 时间;
        //   ③ `pure`   = 再发一次+读 = 二次执行(判"首执行是否被冷态/工作集拖慢")。
        // 无排水 ⇒ 三块混在一起,读数会系统性高估 propose(实测:相位 21 ms 而二次执行 9.4 ms)。
        let t_drain0 = prof.then(std::time::Instant::now);
        if prof {
            let face = self.dev.graphs.session.face_mut();
            face.sync().await?;
        }
        let t0 = prof.then(std::time::Instant::now);
        // ★ **双 rank 镜像**(P2.1-c,2026-10-19):草稿 propose **捕获图里烘焙着 AR**
        // ⇒ 两 rank 必须各自回放(否则 AR 失步)。单卡档 = 原路(零分支)。
        // 读数(`drafts`)只由主 rank 消费;两半都校验。
        let ins: [(&str, &[f32]); 9] = [
            ("tokens", tokens_f.as_slice()),
            ("anchor", anchor.as_slice()),
            ("enc_pos", enc_pos.as_slice()),
            ("enc_slots", enc_slots.as_slice()),
            ("enc_kv_lens", enc_kv_lens.as_slice()),
            ("prop_pos", prop_pos.as_slice()),
            ("prop_slots", prop_slots.as_slice()),
            ("prop_kv_lens", prop_kv_lens.as_slice()),
            ("kv_len", kv_len.as_slice()),
        ];
        mrec("spec.propose.build", t_build.elapsed());
        let t_launch = std::time::Instant::now();
        {
            let (dev, peer) = (&mut self.dev, self.peer.as_mut());
            match peer {
                None => {
                    dev.graphs
                        .dflash
                        .as_mut()
                        .ok_or_else(|| ModelError::EngineInvariant {
                            what: "dflash_graph 缺席(档位不一致)".into(),
                        })?
                        .step(&ins)
                        .await?;
                }
                Some(p) => {
                    let g0 = dev.graphs.dflash.as_mut().ok_or_else(|| {
                        ModelError::EngineInvariant { what: "dflash_graph 缺席(档位不一致)".into() }
                    })?;
                    let g1 = p.graphs.dflash.as_mut().ok_or_else(|| {
                        ModelError::EngineInvariant { what: "peer dflash_graph 缺席".into() }
                    })?;
                    let (r0, r1) = crate::exec::join2(g0.step(&ins), g1.step(&ins)).await;
                    r1?;
                    r0?;
                }
            }
        }
        mrec("spec.propose.launch", t_launch.elapsed());
        let t_sel = std::time::Instant::now();
        let t1 = prof.then(std::time::Instant::now);
        // ★ **切档收割面**（P1.1 × P2.1,2026-10-19）:图本档只出半表候选
        // （`drafts` 槽不写）⇒ 走"读两 rank 候选 → host 合并 → rank0 select"。
        if self.head_split_active() {
            let mut d = self.dflash_select_merged(bonus).await?;
            mrec("spec.propose.sel", t_sel.elapsed());
            if self.boot.probes.dflash_probe {
                owl_shared::log_debug!("dflash-probe", "pos={pos} anchor={bonus} drafts={d:?}");
            }
            self.apply_dflash_lookup(&mut d, lookup);
            return Ok(d);
        }
        let pg = self
            .dev
            .graphs
            .dflash
            .as_mut()
            .ok_or_else(|| ModelError::EngineInvariant { what: "dflash_graph 缺席(档位不一致)".into() })?;
        let sel = pg.read_output_f32("sel").await?;
        let out = &sel[..depth];
        // proposal_q（采样档）：`sel` 后段是 scores；`topk_all` 是选择器
        // 候选（值区‖索引区）。两槽都由 dflash 图同轮产出。
        let q_slots = if sampler_on && pg.has_output("sel") && pg.has_output("topk_all") {
            Some((
                sel[depth..depth + depth * 16 * 16].to_vec(),
                pg.read_output_f32("topk_all").await?,
            ))
        } else {
            None
        };
        // 纯 GPU 判别(E5-DF4 性能;2026-10-18 立案改三段):排水后首次执行 vs 二次执行
        if let (Some(t_drain), Some(t0v), Some(t1v)) = (t_drain0, t0, t1) {
            let t2 = std::time::Instant::now();
            let first = t2 - t1v; // 首次发射+读(流已排干 ⇒ 纯本图)
            pg.replay().await?;
            // 失败已 `?` 上报;成功值此处不需要(仅为同步点)
            let _ = pg.read_output_f32("sel").await?;
            owl_shared::log_info!(
                "dflash-prof",
                "drain={:?} first_gpu={first:?} pure_gpu={:?}",
                t0v - t_drain,
                t2.elapsed()
            );
        }
        if let (Some(a), Some(b)) = (t0, t1) {
            // E2 语义勘误(原 fill/launch/read 误导):total = 轮内全程;
            // step = 装填+发射段;read_extra = 首读段(含队列排水);
            // pure_gpu(上行)= 同步后裸 replay = 图独占 GPU 时间
            owl_shared::log_info!("dflash-prof", "total={:?} step={:?} read_extra={:?}",
                a.elapsed(),
                b - a,
                b.elapsed()
            );
        }
        if let Some((scores, topk)) = q_slots {
            let cand_u: Vec<u32> = topk[depth * 16..2 * depth * 16]
                .iter()
                .map(|&v| v as u32)
                .collect();
            let mut q = super::proposal_q::DraftQ::from_scores(
                &scores,
                &cand_u,
                depth,
                q_temp,
                q_seed,
                fp,
            )?;
            let before = q.drafts().to_vec();
            self.apply_dflash_lookup(q.drafts_mut(), lookup);
            let after = q.drafts().to_vec();
            for (i, (&a, &b)) in before.iter().zip(after.iter()).enumerate() {
                if a != b {
                    q.force_one_hot(i, b);
                }
            }
            let d = q.drafts().to_vec();
            self.tspec()?.spec_q_host = Some(q);
            return Ok(d);
        }
        let mut drafts: Vec<u32> = out.iter().map(|&v| v as u32).collect();
        self.apply_dflash_lookup(&mut drafts, lookup);
        self.tspec()?.spec_q_host = None;
        Ok(drafts)
    }

    /// DFlash2 lookup 融合(草稿头;depth 截断)。仅 `lookup` 开启时由调用方传入。
    fn apply_dflash_lookup(&self, drafts: &mut [u32], lookup: Option<&LookupProposal>) {
        let Some(p) = lookup else { return };
        let k = self.dev.spec_rt.depth.min(drafts.len());
        if k == 0 {
            return;
        }
        if fuse_lookup(&mut drafts[..k], p, LookupCfg::from_runtime(&self.dev.spec_rt)) {
            mcnt("spec.lookup.hit", 1);
        }
    }

    /// 当前请求的 token 历史 = `prompt_ids + out`(out 末位 = 当前 token)
    /// + 本轮 verify 块 `tail`(首轮空;轮末 = `toks` = ids[0..=m],末位 bonus)。
    /// 只在 `[spec] lookup = true` 且 DFlash2 档调用。
    pub(crate) fn dflash_lookup_for_tail(
        &self,
        tail: &[u32],
    ) -> Result<Option<LookupProposal>> {
        if !self.dev.spec_rt.lookup {
            return Ok(None);
        }
        let t0 = std::time::Instant::now();
        let act = self.turns.active()?;
        let mut hist =
            Vec::with_capacity(act.prompt_ids.len() + act.out.len() + tail.len());
        hist.extend_from_slice(&act.prompt_ids);
        hist.extend_from_slice(&act.out);
        hist.extend_from_slice(tail);
        let k = self.dev.spec_rt.depth;
        let cfg = LookupCfg::from_runtime(&self.dev.spec_rt);
        mcnt("spec.lookup.rounds", 1);
        let p = suffix_lookup(&hist, k, cfg);
        if let Some(p) = &p {
            mcnt("spec.lookup.match", p.match_len as u64);
        }
        mrec("spec.lookup", t0.elapsed());
        Ok(p)
    }


    /// DFlash2 propose(E5-DF3):噪声块 [anchor, MASK×7] @ pos..pos+7,
    /// 草稿 KV 窗口 [0, pos+8)(前缀 pos 行已物化 + 自块 8 行直读)。
    /// 返回 (草稿 host 表, 观测面块)。首轮(pos = fed)与轮末同式。
    pub(crate) async fn dflash_propose(
        &mut self,
        anchor: u32,
        pos: usize,
        lookup: Option<&LookupProposal>,
    ) -> Result<Vec<u32>> {
        let d2 = match self.dev.spec_rt.drafter.as_ref() {
            Some(crate::spec_runtime::Drafter::DFlash2(d)) => std::sync::Arc::clone(d),
            _ => unreachable!("dflash_propose 非 dflash 模式"),
        };
        let rope = self.dev.spec_rt.draft_rope.as_ref().ok_or_else(|| ModelError::EngineInvariant { what: "draft rope 缺席(草稿档位不一致)".into() })?.clone();
        let f32b = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|f| f.to_le_bytes()).collect() };
        self.tspec()?.spec_q_host = None;
        let depth = d2.depth();
        let block = d2.block();
        // 噪声块 token/位置表(W = K+1 行)
        let mut toks = vec![anchor as f32];
        toks.resize(block, self.dev.spec_rt.dflash_mask_token as f32);
        let toks_t = TensorOps::from_host(Dtype::F32, vec![block], &f32b(&toks));
        let pos_t = TensorOps::from_host(
            Dtype::F32,
            vec![block],
            &f32b(&(0..block).map(|i| (pos + i) as f32).collect::<Vec<_>>()),
        );
        let anchor_t = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[anchor as f32]));
        // 草稿 KV 叶子;自块槽表 = **物理槽 pos..pos+8**(sglang
        // assign_extend_cache_locs [prefix_len, prefix_len+block_size) 同位;
        // 曾误填 0..8 —— 每轮把位置 0..7 的 memory 前缀 KV 磨成噪声块
        // 残写,永久污染窗口头部 → AL=0 案真凶,2026-10-08 定谳)
        let leaves = self.dev.pool.dflash_kv_leaves().ok_or_else(|| ModelError::EngineInvariant { what: "dflash KV 池句柄缺席(档位/池几何不一致)".into() })?;
        let bt = self.dev.pool.bt_leaf_flat();
        let (_bt_chain, _) = {
            let act = self.turns.active()?;
            let s = self.bank.sessions.session(act.session_id)?;
            // GDN 行 = turn 域资源(§28)
            (s.block_table.clone(), act.gdn_slot)
        };
        let page = self.dev.pool.geom.page;
        // 块表覆盖自保(⑦ propose 跑在 fp = pos+m+1 = 下轮 fed 位,先于
        // 下轮 scheduler 的 ensure_for_len —— 跨页时 bt 未长,曾于 pos+8
        // 越 page 界 panic)。**grew → write_bt 同步律**:块表增长必须
        // 伴设备侧重写,否则 scheduler 下轮 grew=false 跳过 write_bt,
        // 烘焙块表停代(gate9 文本分歧 / gate10 空轮两案真凶)
        {
            let act = self.turns.active()?;
            let s = self.bank.sessions.session_mut(act.session_id)?;
            let before = s.block_table.len();
            // FIXME(store-hook): H2 槽兜底旁路(DFlash2 target 跨页);
            // 应改为 Acquire(Grow) 预留后纯分配。
            self.bank.rel.leg.ensure_for_len(&mut s.block_table, pos + block)?;
            if s.block_table.len() != before {
                let sid = act.session_id;
                self.write_bt(sid).await?;
            }
        }
        let (bt_chain, _) = {
            let act = self.turns.active()?;
            let s = self.bank.sessions.session(act.session_id)?;
            // GDN 行 = turn 域资源(§28)
            (s.block_table.clone(), act.gdn_slot)
        };
        if self.boot.probes.dflash_probe {
            owl_shared::log_debug!("dflash-probe", "propose pos={pos} bt_len={} page={page}", bt_chain.len());
        }
        // 噪声块自槽(经寻址权威;线性逻辑位)
        let self_slots: Vec<f32> = (0..block)
            .map(|i| self.dev.pool.draft_dflash_self_slot(pos, i))
            .collect();
        let kvs: Vec<KvBuffers> = leaves
            .iter()
            .map(|(k, v)| KvBuffers {
                k_cache: k.clone(),
                v_cache: v.clone(),
                slots: TensorOps::from_host(Dtype::F32, vec![block], &f32b(&self_slots)),
                kv_lens: TensorOps::from_host(Dtype::F32, vec![block], &f32b(&(0..block).map(|i| (pos + 1 + i) as f32).collect::<Vec<_>>())),
                block_tables: bt.clone(),
            })
            .collect();
        let mut ctx = ForwardCtx::minimal(block);
        // 草稿算子环境:几何 = **草稿自己**(hd=128 / GQA 比),其余随引擎 env
        ctx.env = self.dev.spec_rt.draft_env;
        // 探针:草稿池 K 前缀行 NaN 检查(定谳 encode vs NC 核;解析 dtype
        // 随草稿池 —— 同日十四后 = BF16)
        if self.boot.probes.dflash_probe {
            let parse2: fn([u8; 2]) -> f32 = if d2.dtype() == Dtype::BF16 {
                |c| half::bf16::from_le_bytes(c).to_f32()
            } else {
                |c| half::f16::from_le_bytes(c).to_f32()
            };
            let (k0, _v0) = &leaves[0];
            let nrow = pos.min(24);
            let row_elems = 8 * 128;
            let kslice = k0.slice_view(0, vec![nrow * row_elems]);
            let kb = {
                let face = self.dev.graphs.session.face_mut();
                owl_models::interpreters::eval_ops(kslice.step(), face).await?
            };
            let mut kbuf = vec![0u8; 786432]; // 整池块(dtoh 强等;12页×8×128×32×2B)
            {
                let face = self.dev.graphs.session.face_mut();
                face.dtoh(&kb, &mut kbuf).await?;
            }
            if pos <= 18 {
                if let Err(e) = owl_shared::file_loader::write(&crate::probe_dump::probe_path("owl_dflash_kpool.bin"), &kbuf) {
                owl_shared::log_error!("dump", "落盘失败(诊断面不许静默): {e}");
            }
                owl_shared::log_debug!("dflash-probe", "草稿 K 池首块落盘 /tmp/owl_dflash_kpool.bin pos={pos}");
            }
            for r in 0..nrow {
                let s: f32 = kbuf[r * row_elems * 2..(r + 1) * row_elems * 2]
                    .chunks_exact(2)
                    .take(64)
                    .map(|c| {
                        let v = parse2([c[0], c[1]]);
                        if v.is_nan() { f32::NAN } else { v.abs() }
                    })
                    .sum();
                if r < 3 || s.is_nan() {
                    owl_shared::log_debug!("dflash-probe", "draftK row{r} |x|前64和 = {s}");
                }
            }
        }
        // 诊断开关:OWL_DFLASH_DUMB=1 → 哑草稿(anchor 重复,跳过 draft
        // 前向)—— 隔离「草稿值泄漏」vs「轮管线污染」(2026-10-07 恒等门
        // AL=0 分歧排查;正式开关挂 C7)
        // OWL_DFLASH_DUMB=1:零副作用哑草稿(不进 GPU;② 直接 host 表)
        if self.boot.probes.dflash_dumb {
            return Ok(vec![anchor; depth]);
        }
        let mut probe_roots: Vec<TensorOps> = Vec::new();
        let probe_on = self.boot.probes.dflash_probe;
        let kv_len_t = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[(pos + block) as f32]));
        let (hidden, drafts, _scores, logits, _cand, _topk) = d2.propose_block(
            &toks_t, &pos_t, &anchor_t, &kvs, &rope, &self.dev.boot.model.embed, &kv_len_t, &ctx,
            if probe_on { Some(&mut probe_roots) } else { None },
        );
        // ★ **peer 侧草稿树(按 peer 自己的块重建)**(P2.1-c,2026-10-19)
        //
        // 为什么不能复用 rank0 的树:ops 树里挂的是**设备块句柄**(embed 权重 /
        // 草稿池叶子 / 块表)—— rank0 的块在 peer 账本里**不存在** ⇒ peer 求值当场拒
        // (`owl_embed_f16: 槽 0 的块不在本账本`);而 `join2` 会等 rank0 到底 ⇒
        // 现场表现 = "挂住"(错被掩住 —— 纪律:可失败返回值不许丢)。
        // host 输入(tokens/pos/slots/kv_lens/bt_chain)两 rank 同值 ⇒ 只换块。
        let peer_tree = |p: &mut crate::engine::RankDev<D>| -> Result<owl_models::TensorOps> {
            let leaves_p = p.pool.dflash_kv_leaves().ok_or_else(|| {
                ModelError::EngineInvariant { what: "peer 草稿 KV 池句柄缺席".into() }
            })?;
            let bt_p = p.pool.bt_leaf_flat();
            let kvs_p: Vec<KvBuffers> = leaves_p
                .iter()
                .map(|(k, v)| KvBuffers {
                    k_cache: k.clone(),
                    v_cache: v.clone(),
                    slots: TensorOps::from_host(Dtype::F32, vec![block], &f32b(&self_slots)),
                    kv_lens: TensorOps::from_host(
                        Dtype::F32,
                        vec![block],
                        &f32b(&(0..block).map(|i| (pos + 1 + i) as f32).collect::<Vec<_>>()),
                    ),
                    block_tables: bt_p.clone(),
                })
                .collect();
            // ⚠️ 用 **peer 自己的 drafter / rope**(每 rank 各一份实例、各一套设备块;
            // 用 rank0 的 `d2`/`rope` 建树 ⇒ 权重块不在 peer 账本 ⇒ 当场拒)
            let d2_p = match p.spec_rt.drafter.as_ref() {
                Some(crate::spec_runtime::Drafter::DFlash2(d)) => std::sync::Arc::clone(d),
                _ => {
                    return Err(ModelError::EngineInvariant {
                        what: "peer 草稿器缺席/非 DFlash2(镜像装配不一致)".into(),
                    })
                }
            };
            let rope_p = p.spec_rt.draft_rope.as_ref().ok_or_else(|| {
                ModelError::EngineInvariant { what: "peer 草稿 rope 缺席".into() }
            })?;
            let (_, d_p, _, _, _, _) = d2_p.propose_block(
                &toks_t, &pos_t, &anchor_t, &kvs_p, rope_p, &p.boot.model.embed, &kv_len_t, &ctx, None,
            );
            Ok(d_p)
        };
        // 探针:首个 propose 的 hidden + 全部探针根落盘(python 逐阶段对拍;
        // **逐根独立 eval** —— multi 后段会把早根块回收复用,dtoh 读到陈旧
        // 块(root0 曾读出 1e6 级垃圾假象);单链重执行确定性,值可信)
        if self.boot.probes.dflash_probe && !self.dev.spec_rt.hid_dumped {
            self.dev.spec_rt.hid_dumped = true;
            for (i, r) in probe_roots.iter().enumerate() {
                let face = self.dev.graphs.session.face_mut();
                let b = owl_models::interpreters::eval_ops(r.step(), face).await?;
                let n: usize = r.shape().iter().product();
                let mut blk = vec![0u8; n * 2];
                face.dtoh(&b, &mut blk).await?;
                if let Err(e) = owl_shared::file_loader::write(&crate::probe_dump::probe_path(&format!("owl_p{i}.bin")), &blk) {
                owl_shared::log_error!("dump", "落盘失败(诊断面不许静默): {e}");
            }
            }
            let face = self.dev.graphs.session.face_mut();
            let b = owl_models::interpreters::eval_ops(hidden.step(), face).await?;
            let n: usize = hidden.shape().iter().product();
            let mut hbuf = vec![0u8; n * 2];
            face.dtoh(&b, &mut hbuf).await?;
            if let Err(e) = owl_shared::file_loader::write(&crate::probe_dump::probe_path("owl_dflash_hid.bin"), &hbuf) {
                owl_shared::log_error!("dump", "落盘失败(诊断面不许静默): {e}");
            }
            let mut m = Vec::new();
            m.extend_from_slice(&(pos as u32).to_le_bytes());
            m.extend_from_slice(&anchor.to_le_bytes());
            if let Err(e) = owl_shared::file_loader::write(&crate::probe_dump::probe_path("owl_dflash_meta.bin"), &m) {
                owl_shared::log_error!("dump", "落盘失败(诊断面不许静默): {e}");
            }
            owl_shared::log_debug!("dflash-probe", "{} probe roots + hidden 落盘(逐根独立 eval)pos={pos} anchor={anchor}", probe_roots.len());
        }
        // 探针:logits 前 4 值 f16(NaN 定位;OWL_DFLASH_PROBE)
        let logits_probe = if self.boot.probes.dflash_probe {
            Some(logits.clone())
        } else {
            None
        };
        let (b, arena) = {
            let env0 = self.dev.spec_rt.draft_env;
            let tp0 = self.dev.boot.tp.clone();
            // peer 树必须在借 face 之前建(`peer_tree` 要 `&mut p` 读池/模型)
            let peer_t = match self.peer.as_mut() {
                Some(p) => Some(peer_tree(p)?),
                None => None,
            };
            let (dev, peer) = (&mut self.dev, self.peer.as_mut());
            match (peer, peer_t) {
                (None, _) => {
                    let face = dev.graphs.session.face_mut();
                    owl_models::interpreters::eval_ops_scoped_env(drafts.step(), face, env0, tp0.as_ref()).await?
                }
                (Some(p), Some(pt)) => {
                    // ★ **双 rank 镜像**(P2.1-c):草稿体里有 RowParallel 的 allreduce
                    // ⇒ 两 rank 必须同跑(否则 AR 失步)。单卡档 = 原路。
                    // 读数只取主 rank(确定性 ⇒ 两 rank 同值);**两半都校验**。
                    let env1 = p.spec_rt.draft_env;
                    let tp1 = p.boot.tp.clone();
                    let (f0, f1) = (dev.graphs.session.face_mut(), p.graphs.session.face_mut());
                    let peer_fut = async move {
                        let r = owl_models::interpreters::eval_ops_scoped_env(
                            pt.step(), f1, env1, tp1.as_ref(),
                        )
                        .await;
                        if let Err(ref e) = r {
                            // **响亮**:peer 的错不许静默(join2 会等 rank0 到底 ⇒
                            // 不主动打就永远看不见;现场第一版就是"挂"的样子)
                            owl_shared::log_error!("spec-mirror", "peer 草稿 eval 失败: {e}");
                        }
                        r
                    };
                    let (r0, r1) = crate::exec::join2(
                        owl_models::interpreters::eval_ops_scoped_env(drafts.step(), f0, env0, tp0.as_ref()),
                        peer_fut,
                    )
                    .await;
                    r1?;
                    r0?
                }
                (Some(_), None) => {
                    return Err(ModelError::EngineInvariant {
                        what: "peer 束在场但草稿树缺席(镜像装配不一致)".into(),
                    });
                }
            }
        };
        if let Some(lp) = logits_probe {
            // 单遍 multi(共享 memo;独立 eval 重执行副作用层 = 原地
            // fused_add 累加 = 探针自污染 —— testkit 重放语义警告同源)
            let mut roots: Vec<TensorOps> = probe_roots.clone();
            roots.push(lp);
            let refs: Vec<&TensorOps> = roots.iter().collect();
            let outs = {
                let face = self.dev.graphs.session.face_mut();
                owl_models::interpreters::eval_ops_multi_env(&refs, face, self.dev.spec_rt.draft_env, self.dev.boot.tp.as_ref()).await?
            };
            for (i, pr) in probe_roots.iter().enumerate() {
                let n: usize = pr.shape().iter().product();
                let mut pbuf = vec![0u8; n * 2];
                {
                    let face = self.dev.graphs.session.face_mut();
                    face.dtoh(&outs[i], &mut pbuf).await?;
                }
                // 探针根 dtype 随草稿(bf16 后同 2B,解析面分派)
                let parse2: fn([u8; 2]) -> f32 = if pr.dtype() == Dtype::BF16 {
                    |c| half::bf16::from_le_bytes(c).to_f32()
                } else {
                    |c| half::f16::from_le_bytes(c).to_f32()
                };
                let s: f32 = pbuf
                    .chunks_exact(2)
                    .take(512)
                    .map(|c| {
                        let v = parse2([c[0], c[1]]);
                        if v.is_nan() { f32::NAN } else { v.abs() }
                    })
                    .sum();
                let head: Vec<f32> = pbuf[..16]
                    .chunks_exact(2)
                    .map(|c| parse2([c[0], c[1]]))
                    .collect();
                owl_shared::log_debug!("dflash-probe", "root{i} ({}) |x|前512和 = {s} head={head:?}", pr.shape().iter().map(|v| v.to_string()).collect::<Vec<_>>().join("x"));
            }
            let lb = outs.last().ok_or_else(|| ModelError::EngineInvariant { what: "draft:logits root 缺席".into() })?;
            let mut lbuf = vec![0u8; 7 * 248320 * 2];
            {
                let face = self.dev.graphs.session.face_mut();
                face.dtoh(lb, &mut lbuf).await?;
            }
            let lvals: Vec<f32> = lbuf[..16]
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect();
            owl_shared::log_debug!("dflash-probe", "logits[0..4]={lvals:?}");
            // 槽 0 候选 vs 发出草稿(选择器诊断:topk 面若合理而 toks 离谱
            // = 码本项支配 walk;topk 已离谱 = hidden 坏)
            {
                let row0: Vec<(f32, usize)> = lbuf[..248320]
                    .chunks_exact(2)
                    .enumerate()
                    .map(|(i, c)| {
                        (half::f16::from_le_bytes([c[0], c[1]]).to_f32(), i)
                    })
                    .collect();
                let mut top: Vec<(f32, usize)> = row0.into_iter().collect();
                // 诊断面**不许 panic**:全非有限/空 logits(实测 NaN 步)时
                // `select_nth_unstable_by(7)` 会越界崩引擎 —— 先按实际长度夹紧。
                if !top.is_empty() {
                    let k = top.len().saturating_sub(1).min(7);
                    top.select_nth_unstable_by(k, |a, b| b.0.total_cmp(&a.0));
                }
                top.truncate(8);
                top.sort_by(|a, b| b.0.total_cmp(&a.0));
                owl_shared::log_debug!("dflash-probe", "slot0 lm_head top8 = {:?}", top);
            }
        }
        let mut out = {
            let face = self.dev.graphs.session.face_mut();
            // drafts 根 = select 单缓冲的 SliceView —— dtoh 回读整父块
            // [S·(K·K+1) f32],取前 S 个(dflash2.rs select 同坑,测试侧同修)
            let mut buf = vec![0u8; depth * (16 * 16 + 1) * 4];
            face.dtoh(&b, &mut buf).await?;
            // 根块 = drafts 的 **select 单缓冲**(`b` 是它的 SliceView)——
            // harvest 完就是 turn 域死块:必须与 arena 一起还,否则每轮漏一条
            // (`/debug/vram` 的 `turn:N` 列逐轮爬坡;2026-10-19 发射账取证:
            //  alloc→select→dtoh 三条之后既无 free 也不在 arena 候选里)。
            let mut ids = arena;
            ids.push(b.id);
            face.free(&ids).await?;
            buf.chunks_exact(4)
                .take(depth) // 父块 = select 单缓冲(SliceView dtoh 回整块);草稿仅前 K
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u32)
                .collect::<Vec<_>>()
        };
        // eager 草稿路的 q：不在这里二次求值选择器树 —— 该树含 RowParallel
        // AR，单边重跑会与 peer 失步挂住（2026-10-23 实测）。首轮/无图档
        // 走旧的「目标抽样 == draft」等价式；图档在 `dflash_graph_round`
        // 用同图输出的 scores/topk 建 q。
        self.apply_dflash_lookup(&mut out, lookup);
        self.tspec()?.spec_q_host = None;
        if self.boot.probes.dflash_probe {
            owl_shared::log_debug!("dflash-probe", "pos={pos} anchor={anchor} drafts={out:?}");
        }
        Ok(out)
    }


    /// DFlash2 encode(E5-DF2;sglang _append_target_hidden 同语义):
    /// verify 块行 [0..=m+1](taps 前缀视图)→ memory → 5 层 kv-only 写
    /// 草稿池 @ pos..pos+m+2。**含 bonus 行**(行 m+1)—— 噪声块前缀须
    /// 覆盖至下一空位,bonus 行不编 = 前缀未初始化读 = NaN 塌缩
    /// (sglang 靠上轮噪声行 0 残写覆盖同槽;owl 显式编,语义同)。
    /// 下轮重编行 0 幂等(同 token 同位确定性同值)。
    pub(crate) async fn dflash_encode(
        &mut self,
        d2: &std::sync::Arc<owl_models::layers::dflash2::DFlash2Draft>,
        taps: &[owl_iface::contract::Bytes],
        m: usize,
        pos: usize,
    ) -> Result<()> {
        // 行数 = verify 块全行 W(运行时 K+1;taps 块仅 W 行,m=K 时
        // m+2=K+2 越界读 —— 多编行下轮被自块写/重编覆盖,幂等安全)
        let rows = (m + 2).min(d2.block());
        let rope = self.dev.spec_rt.draft_rope.as_ref().ok_or_else(|| ModelError::EngineInvariant { what: "draft rope 缺席(草稿档位不一致)".into() })?.clone();
        let dtype = self.dev.pool.geom.dims.dtype;
        let views: Vec<TensorOps> = taps
            .iter()
            .map(|b| TensorOps::of_block(b.id, dtype, vec![rows, d2.hidden()]))
            .collect();
        let f32b = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|f| f.to_le_bytes()).collect() };
        let pos_t = TensorOps::from_host(
            Dtype::F32,
            vec![rows],
            &f32b(&(0..rows).map(|i| (pos + i) as f32).collect::<Vec<_>>()),
        );
        // 编码槽表 = 线性逻辑位(pos..pos+rows;草稿池线性寻址,同 ②)
        let _page = self.dev.pool.geom.page;
        let leaves = self.dev.pool.dflash_kv_leaves().ok_or_else(|| ModelError::EngineInvariant { what: "dflash KV 池句柄缺席(档位/池几何不一致)".into() })?;
        let bt = self.dev.pool.bt_leaf_flat();
        let kvs: Vec<KvBuffers> = leaves
            .iter()
            .map(|(k, v)| KvBuffers {
                k_cache: k.clone(),
                v_cache: v.clone(),
                slots: TensorOps::from_host(
                    Dtype::F32,
                    vec![rows],
                    &f32b(&(0..rows)
                        .map(|i| self.dev.pool.draft_dflash_encode_slot(pos, i))
                        .collect::<Vec<_>>()),
                ),
                kv_lens: TensorOps::from_host(
                    Dtype::F32,
                    vec![rows],
                    &f32b(&(0..rows).map(|i| (pos + i + 1) as f32).collect::<Vec<_>>()),
                ),
                block_tables: bt.clone(),
            })
            .collect();
        let ctx = ForwardCtx::minimal(rows);
        let memory = d2.project_memory(&views, &ctx);
        // 三段 bisect(E5-DF3 同日十一):①memory 独立 eval ②multi 根=[memory]
        // ③encode multi → 各自读回行 0 |max|/池非零计,定位“in-multi 零、
        // standalone 有限”的分裂点(OWL_DFLASH_PROBE)
        let probe = self.boot.probes.dflash_probe;
        if probe {
            let b1 = {
                let face = self.dev.graphs.session.face_mut();
                owl_models::interpreters::eval_ops(memory.step(), face).await?
            };
            {
                let mut vb = vec![0u8; rows * 5120 * 2];
                let face = self.dev.graphs.session.face_mut();
                face.dtoh(&b1, &mut vb).await?;
                let mx = vb[..5120 * 2]
                    .chunks_exact(2)
                    .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32().abs())
                    .fold(0f32, f32::max);
                let nz = vb[..5120 * 2]
                    .chunks_exact(2)
                    .filter(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32() != 0.0)
                    .count();
                owl_shared::log_debug!("enc-bisect", "①eval_ops(memory) 独立: 行0 非零={nz}/5120 |max|={mx:.4}");
            }
            let refs1 = vec![&memory];
            let o1 = {
                let face = self.dev.graphs.session.face_mut();
                owl_models::interpreters::eval_ops_multi(&refs1, face, self.dev.boot.tp.as_ref()).await?
            };
            {
                let mut vb = vec![0u8; rows * 5120 * 2];
                let face = self.dev.graphs.session.face_mut();
                face.dtoh(&o1[0], &mut vb).await?;
                let mx = vb[..5120 * 2]
                    .chunks_exact(2)
                    .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32().abs())
                    .fold(0f32, f32::max);
                let nz = vb[..5120 * 2]
                    .chunks_exact(2)
                    .filter(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32() != 0.0)
                    .count();
                owl_shared::log_debug!("enc-bisect", "②multi 根=[memory]: 行0 非零={nz}/5120 |max|={mx:.4}");
            }
        }
        let roots = d2.encode_kv(&memory, &pos_t, &kvs, &rope, &ctx);
        {
            let face = self.dev.graphs.session.face_mut();
            let refs: Vec<&TensorOps> = roots.iter().collect();
            // 副作用根口径:encode 的末位出参 = 契约 4 占位(核就地写池),没人读 ⇒ 收尾一并归还
            owl_models::interpreters::eval_ops_side_effect_env(&refs, face, self.dev.spec_rt.draft_env, self.dev.boot.tp.as_ref()).await?;
        }
        if probe {
            let (k0, _) = &leaves[0];
            let kn: usize = k0.shape().iter().product();
            let mut kb = vec![0u8; kn * 2];
            {
                let face = self.dev.graphs.session.face_mut();
                let b = owl_models::interpreters::eval_ops(k0.step(), face).await?;
                face.dtoh(&b, &mut kb).await?;
            }
            let nz = kb
                .chunks_exact(2)
                .filter(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32() != 0.0)
                .count();
            owl_shared::log_debug!("enc-bisect", "③encode 后 K 池非零={nz}/{}", kn);
        }
        Ok(())
    }


    /// 首轮 propose(E5-M2b;m=0 特例):pair (h_{pos-1}, emb(anchor)) @
    /// 位置 pos-1 = M1 propose 本尊;seed hidden = prefill 末块 harvest。
    /// 产出 k 草稿块持久(spec_drafts)。
    pub(crate) async fn propose_first(
        &mut self,
        anchor: u32,
        pos: usize,
        depth: usize,
    ) -> Result<()> {
        let seed = self
            .tspec()?
            .spec_seed_hidden
            .take()
            .ok_or_else(|| owl_models::ModelError::Msg(
                "spec seed 缺失(prefill 末块 harvest 未落;mtp 模式要求)".into(),
            ))?;
        // mtp 链表:覆盖 pos-1(extend 行)+ pos..pos+depth-2(链步)
        let bt = self.mtp_chain_bt(pos + depth - 1)?;
        let (kc, vc) = self.dev.pool.mtp_kv_leaf().ok_or_else(|| ModelError::EngineInvariant { what: "MTP KV 池句柄缺席(draft 档位不一致)".into() })?;
        let page = self.dev.pool.geom.page;
        let f32b = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|f| f.to_le_bytes()).collect() };
        let requested = self.dev.spec_rt.mtp_window;
        let attn_v2 = self.dev.pool.attn_v2_scratch();
        let windowed = requested.active() && attn_v2.is_some();
        let cfg = if windowed { requested } else { MtpWindowCfg::default() };
        let plan = MtpWindowPlan::new(&bt, page, cfg)?;
        // 步 i:位置 pos-1+i,kv_len = 每步可见长度(含自身)
        let mut kvs = Vec::new();
        let mut pos_ts = Vec::new();
        for i in 0..depth {
            let p = pos - 1 + i;
            let view = plan.row(p)?;
            let bt_step = TensorOps::from_host(
                Dtype::F32,
                vec![1, view.blocks.len()],
                &f32b(&view.blocks.iter().map(|&b| b as f32).collect::<Vec<_>>()),
            );
            kvs.push(KvBuffers {
                k_cache: kc.clone(),
                v_cache: vc.clone(),
                slots: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[(bt[p / page] * page as u32 + (p % page) as u32) as f32])),
                kv_lens: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[view.visible as f32])),
                block_tables: bt_step,
            });
            pos_ts.push(TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[p as f32])));
        }
        let anchor_t = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[anchor as f32]));
        let seed_view =
            TensorOps::of_block(seed.id, self.dev.pool.geom.dims.dtype, vec![1, self.dev.pool.geom.dims.hidden]);
        let drafts = {
            let env = self.dev.spec_rt.draft_env;
            let vocab = self.dev.boot.model.vocab_size();
            let rope = self.dev.boot.rope.clone();
            // 链步 decode 读核跟池布局:kNH 池必须走 v2(否则落 v1 classic
            // ↠ 布局错位、草稿全垃圾;2026-10-13 布局守卫捕获)
            match self.dev.spec_rt.drafter.as_ref().ok_or_else(|| ModelError::EngineInvariant {
                what: "drafter 句柄缺席(草稿档位不一致)".into(),
            })? {
                crate::spec_runtime::Drafter::Mtp(mtp) => mtp.propose(
                    &anchor_t, &seed_view, depth, &self.dev.boot.model.embed, &rope, &kvs, &pos_ts,
                    vocab, env, attn_v2,
                ),
                crate::spec_runtime::Drafter::DFlash2(_) => unreachable!("DFlash2 不走 propose_first(seed 链)"),
            }
        };
        let (b, arena) = {
            let face = self.dev.graphs.session.face_mut();
            owl_models::interpreters::eval_ops_scoped_env(drafts.step(), face, self.dev.spec_rt.draft_env, self.dev.boot.tp.as_ref()).await?
        };
        {
            let face = self.dev.graphs.session.face_mut();
            face.free(&arena).await?;
            face.free(&[seed.id]).await?; // seed 块已消费,归还
            let mut buf = vec![0u8; 4 * depth];
            face.dtoh(&b, &mut buf).await?;
            face.free(&[b.id]).await?;
            self.tspec()?.spec_drafts_host = Some(
                buf.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u32)
                    .collect(),
            );
        }
        Ok(())
    }


    /// 轮末 propose(E5-M2b;§9.2):extend 行 = (h_{F+i}, emb(t_{F+i+1}))
    /// @ F..F+m(hiddens = verify hidden 块行 0..m 前缀视图,tokens =
    /// [d1..dm, bonus])+ 链步 @ F'..F'+k-2 → 下轮草稿块。
    pub(crate) async fn propose_round(
        &mut self,
        hidden: &owl_iface::contract::Bytes,
        m: usize,
        toks: &[u32],
        pos: usize,
        depth: usize,
    ) -> Result<owl_iface::contract::Bytes> {
        // mtp 链表覆盖:extend F..F+m + 链 F'..F'+depth-2(F' = F+m+1)
        let fp = pos + m + 1;
        // 链步最远位 = fp+depth-2 → 链表需覆盖 token 数 fp+depth-1
        let bt = self.mtp_chain_bt(fp + depth - 1)?;
        let (kc, vc) = self.dev.pool.mtp_kv_leaf().ok_or_else(|| ModelError::EngineInvariant { what: "MTP KV 池句柄缺席(draft 档位不一致)".into() })?;
        let page = self.dev.pool.geom.page;
        let f32b = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|f| f.to_le_bytes()).collect() };
        let requested = self.dev.spec_rt.mtp_window;
        let extend_v2 = self
            .dev
            .pool
            .verify_attn_v2_scratch()
            .and_then(|s| s.prefix(m + 1));
        let attn_v2 = self.dev.pool.attn_v2_scratch();
        // boot 已对 window>0 拒缺 v2；这里仍保留显式 full fallback（例如
        // scratch prefix 越界），绝不让 window 表配 full lens。
        let windowed = requested.active() && extend_v2.is_some() && attn_v2.is_some();
        let cfg = if windowed { requested } else { MtpWindowCfg::default() };
        let plan = MtpWindowPlan::new(&bt, page, cfg)?;
        let ext_views = plan.rows(pos, m + 1)?;
        if cfg.active() {
            let full_rows = ext_views.iter().filter(|v| v.blocks.len() >= bt.len()).count();
            if full_rows == ext_views.len() {
                mcnt("spec.mtp.window.full", 1);
            } else {
                mcnt("spec.mtp.window.rounds", 1);
            }
            mcnt(
                "spec.mtp.window.visible_tokens",
                ext_views.iter().map(|v| v.visible as u64).sum(),
            );
        }
        let bt_t = TensorOps::from_host(
            Dtype::F32,
            vec![1, bt.len()],
            &f32b(&bt.iter().map(|&b| b as f32).collect::<Vec<_>>()),
        );
        let slot_at = |p: usize| -> f32 { (bt[p / page] * page as u32 + (p % page) as u32) as f32 };
        // extend 批行建表([M+1])
        let slots_x: Vec<f32> = (0..=m).map(|i| slot_at(pos + i)).collect();
        let lens_x: Vec<f32> = ext_views.iter().map(|v| v.visible as f32).collect();
        let pos_x: Vec<f32> = (0..=m).map(|i| (pos + i) as f32).collect();
        let extend_kv = KvBuffers {
            k_cache: kc.clone(),
            v_cache: vc.clone(),
            slots: TensorOps::from_host(Dtype::F32, vec![m + 1], &f32b(&slots_x)),
            kv_lens: TensorOps::from_host(Dtype::F32, vec![m + 1], &f32b(&lens_x)),
            block_tables: bt_t.clone(),
        };
        let tokens_t = TensorOps::from_host(
            Dtype::F32,
            vec![m + 1],
            &f32b(&toks.iter().map(|&v| v as f32).collect::<Vec<_>>()),
        );
        let hiddens =
            TensorOps::of_block(hidden.id, self.dev.pool.geom.dims.dtype, vec![m + 1, self.dev.pool.geom.dims.hidden]);
        let extend_pos = TensorOps::from_host(Dtype::F32, vec![m + 1], &f32b(&pos_x));
        let extend_slots = TensorOps::from_host(Dtype::F32, vec![m + 1], &f32b(&slots_x));
        let extend_lens = TensorOps::from_host(Dtype::F32, vec![m + 1], &f32b(&lens_x));
        // 链步建表(k-1 步 @ F'..F'+depth-2)
        let mut chain = Vec::new();
        for j in 1..depth {
            let p = fp + j - 1;
            let view = plan.row(p)?;
            let bt_step = TensorOps::from_host(
                Dtype::F32,
                vec![1, view.blocks.len()],
                &f32b(&view.blocks.iter().map(|&b| b as f32).collect::<Vec<_>>()),
            );
            let kv = KvBuffers {
                k_cache: kc.clone(),
                v_cache: vc.clone(),
                slots: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[slot_at(p)])),
                kv_lens: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[view.visible as f32])),
                block_tables: bt_step,
            };
            let pt = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[p as f32]));
            chain.push((pt, kv));
        }
        let chain_refs: Vec<(&TensorOps, &KvBuffers)> =
            chain.iter().map(|(p, k)| (p, k)).collect();
        // MTP extend 页表:每行一条读视图(短 ctx/关闭时 = 完整链表);
        // 行宽固定为池页数,余页补零。
        let nb = self.dev.pool.geom.nb;
        let mut bt8_f = vec![0f32; (m + 1) * nb];
        for (r, view) in ext_views.iter().enumerate() {
            for (i, &b) in view.blocks.iter().take(nb).enumerate() {
                bt8_f[r * nb + i] = b as f32;
            }
        }
        let bt8_t = TensorOps::from_host(
            Dtype::F32,
            vec![m + 1, nb],
            &f32b(&bt8_f),
        );
        let ts_build = std::time::Instant::now();
        // cu(E5-M5:eager 路自建合法;图态走 propose_graph_step 的输入槽)
        let cu_t = TensorOps::from_host(
            Dtype::F32,
            vec![2],
            &[0.0f32, (pos + m + 1) as f32]
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect::<Vec<u8>>(),
        );
        let drafts = {
            let env = self.dev.spec_rt.draft_env;
            let vocab = self.dev.boot.model.vocab_size();
            let rope = self.dev.boot.rope.clone();
            match self.dev.spec_rt.drafter.as_ref().ok_or_else(|| ModelError::EngineInvariant {
                what: "drafter 句柄缺席(草稿档位不一致)".into(),
            })? {
                crate::spec_runtime::Drafter::Mtp(mtp) => mtp.propose_ext(
                    &tokens_t, &hiddens, &extend_pos, &extend_kv, &extend_slots, &extend_lens,
                    &cu_t, &chain_refs, &self.dev.boot.model.embed, &rope, vocab, env, attn_v2,
                    extend_v2, Some(&bt8_t),
                ),
                crate::spec_runtime::Drafter::DFlash2(_) => unreachable!("DFlash2 走 dflash_propose"),
            }
        };
        mrec("spec.propose.build", ts_build.elapsed());
        let ts_eval = std::time::Instant::now();
        let (b, arena) = {
            let face = self.dev.graphs.session.face_mut();
            owl_models::interpreters::eval_ops_scoped_env(drafts.step(), face, self.dev.spec_rt.draft_env, self.dev.boot.tp.as_ref()).await?
        };
        mrec("spec.propose.eval", ts_eval.elapsed());
        {
            let face = self.dev.graphs.session.face_mut();
            face.free(&arena).await?;
        }
        Ok(b)
    }


    /// prefill extend(E5-M5;sglang _append_target_hidden_to_draft_kv
    /// 同语义):chunk 的 (h_p, emb(t_{p+1})) 对 @ 位置 p 写 mtp KV,
    /// 草稿链看得见提示词。行数 = t_rows(= T-1;末行配对归下一 chunk /
    /// propose_first)。eager from_host 表(每 chunk 一次;非图)。


    /// propose 图步(E5-M5;桶 m):装填 extend/链表 + bt_mtp 持久槽
    /// 重写 → 单 graph_launch → drafts dtoh(host 账)。
    pub(crate) async fn propose_graph_step(
        &mut self,
        m: usize,
        toks: &[u32],
        pos: usize,
        depth: usize,
    ) -> Result<Vec<u32>> {
        let page = self.dev.pool.geom.page;
        let bt = {
            let sid = self.turns.active()?.session_id;
            let s = self.bank.sessions.get_mut(sid).expect("账在");
            // FIXME(store-hook): H2 槽兜底旁路(MTP propose 链);
            // 应改为 Acquire(Grow) 预留后纯分配。
            self.bank.leg_mtp.ensure_for_len(&mut s.mtp_block_table, pos + m + depth)?;
            s.mtp_block_table.clone()
        };
        {
            let ts_w = std::time::Instant::now();
            let face = self.dev.graphs.session.face_mut();
            self.dev.pool.write_bt_mtp(face, &bt).await?;
            mrec("spec.propose.btwt", ts_w.elapsed());
        }
        let fp = pos + m + 1;
        let slot_at =
            |p: usize| (bt[p / page] * page as u32 + (p % page) as u32) as f32;
        let cfg = self.dev.spec_rt.mtp_window;
        let plan = MtpWindowPlan::new(&bt, page, cfg)?;
        let ext_views = plan.rows(pos, m + 1)?;
        if cfg.active() {
            let full_rows = ext_views.iter().filter(|v| v.blocks.len() >= bt.len()).count();
            if full_rows == ext_views.len() {
                mcnt("spec.mtp.window.full", 1);
            } else {
                mcnt("spec.mtp.window.rounds", 1);
            }
            mcnt(
                "spec.mtp.window.visible_tokens",
                ext_views.iter().map(|v| v.visible as u64).sum(),
            );
        }
        let tok_f: Vec<f32> = toks.iter().map(|&v| v as f32).collect();
        let pos_f: Vec<f32> = (pos..pos + m + 1).map(|p| p as f32).collect();
        let slots_f: Vec<f32> = (pos..pos + m + 1).map(slot_at).collect();
        let lens_f: Vec<f32> = ext_views.iter().map(|v| v.visible as f32).collect();
        // 图槽容量 = (m+1) × 池页数(固定);会话链表只覆盖当前所需页,
        // 必须补零到全宽。window 视图只改读表,lens 用压缩可见长度。
        let nb = self.dev.pool.geom.nb;
        let mut bt8_f = vec![0f32; (m + 1) * nb];
        for (r, view) in ext_views.iter().enumerate() {
            for (i, &b) in view.blocks.iter().take(nb).enumerate() {
                bt8_f[r * nb + i] = b as f32;
            }
        }
        let mut ins: Vec<(&str, &[f32])> = vec![
            ("tok_ext", tok_f.as_slice()),
            ("pos_ext", pos_f.as_slice()),
            ("slots_ext", slots_f.as_slice()),
            ("lens_ext", lens_f.as_slice()),
            ("v2_bt8", bt8_f.as_slice()),
        ];
        let mut chain_bufs: Vec<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> = Vec::new();
        for j in 1..depth {
            let p = fp + j - 1;
            let view = plan.row(p)?;
            let mut bt_f = vec![0f32; nb];
            for (i, &b) in view.blocks.iter().take(nb).enumerate() {
                bt_f[i] = b as f32;
            }
            chain_bufs.push((
                vec![p as f32],
                vec![slot_at(p)],
                vec![view.visible as f32],
                bt_f,
            ));
        }
        let names: Vec<String> = (0..chain_bufs.len())
            .flat_map(|j| {
                [
                    format!("c{j}_pos"),
                    format!("c{j}_slots"),
                    format!("c{j}_lens"),
                    format!("c{j}_bt"),
                ]
            })
            .collect();
        for (j, (a, b, c, d)) in chain_bufs.iter().enumerate() {
            ins.push((names[j * 4].as_str(), a.as_slice()));
            ins.push((names[j * 4 + 1].as_str(), b.as_slice()));
            ins.push((names[j * 4 + 2].as_str(), c.as_slice()));
            ins.push((names[j * 4 + 3].as_str(), d.as_slice()));
        }
        let pg = self.dev.graphs.propose_mut(depth, m)?;
        let t1 = std::time::Instant::now();
        pg.step(&ins).await?;
        let t_step = t1.elapsed();
        let d = pg.read_output_f32("drafts").await?;
        mrec("spec.propose.step", t_step);
        mrec("spec.propose.read", t1.elapsed() - t_step);
        Ok(d.iter().map(|&v| v as u32).collect())
    }


    /// **切档 propose 的收割面**（P1.1 × P2.1,2026-10-19；刀3 2026-10-26 下沉）
    ///
    /// 图（每 rank，半表）已出两槽：`topk`（`[7,16]` 值区 ‖ 索引区）与
    /// `sel_proj`（`[7,rank]`）。收割 = ① 两半候选合并成全表 top-16；
    /// ② **rank0** 跑 selector（全表域；码本全量）→ drafts。
    ///
    /// 为什么正确:全局 top-16 ⊆ 两半各自 top-16 的并集（截断性质）。
    /// 为什么 rank0:读数面单点；peer 的图输出照旧丢弃。
    ///
    /// **刀3**:①②整体下沉到 `dflash_select` 图（两半合并核 + selector），
    /// 本 rank 的 `topk`/`sel_proj` 直接吃 propose 图的输出块（零回读）。
    /// 唯一必须主机参与的只剩 peer 候选:两 rank 的图**并发发射**，谁都不能
    /// 读对方本轮产物 ⇒ 一次 dtoh 既取数据也当同步点。
    /// 图缺席（未切档/捕获降级/单卡）⇒ 回落原 host 合并 + eager select。
    async fn dflash_select_merged(&mut self, bonus: u32) -> Result<Vec<u32>> {
        use owl_models::layers::dflash2::TOP_K;
        let rows = self.dev.spec_rt.depth;
        let k = TOP_K;
        let half = self.dev.boot.model.logits_vocab();
        let t_r1 = std::time::Instant::now();
        // ① peer 同槽（候选；半表域）。这一次 dtoh = **peer 图完成同步点**
        //    （本 rank 的图不必单独等:收割图与它同流,回放天然有序）
        let peer = {
            let p = self
                .peer
                .as_mut()
                .ok_or_else(|| ModelError::EngineInvariant { what: "切档 propose 缺 peer".into() })?;
            let pg1 = p
                .graphs
                .dflash
                .as_mut()
                .ok_or_else(|| ModelError::EngineInvariant { what: "peer dflash_graph 缺席".into() })?;
            let t = pg1.read_output_f32("topk").await?;
            if t.len() != 2 * rows * k {
                return Err(ModelError::EngineInvariant {
                    what: format!("peer 切档 propose topk 槽尺寸不符:{} ≠ {}", t.len(), 2 * rows * k),
                });
            }
            t
        };
        mrec("spec.propose.sel.read1", t_r1.elapsed());

        // ② **图内收割**（刀3）:合并核 + selector 全在设备侧
        if self.dev.graphs.dflash_select.is_some() {
            let anchor = [bonus as f32];
            let t_step = std::time::Instant::now();
            {
                let pg = self.dev.graphs.dflash_select.as_mut().ok_or_else(|| {
                    ModelError::EngineInvariant { what: "select 图并发态自毁(不应发生)".into() }
                })?;
                pg.step(&[("peercand", &peer), ("anchor", &anchor)]).await?;
            }
            mrec("spec.propose.sel.step", t_step.elapsed());
            let t_read = std::time::Instant::now();
            let pg = self.dev.graphs.dflash_select.as_mut().ok_or_else(|| {
                ModelError::EngineInvariant { what: "select 图并发态自毁(不应发生)".into() }
            })?;
            let sel = pg.read_output_f32("sel").await?;
            mrec("spec.propose.sel.read", t_read.elapsed());
            if sel.len() < rows {
                return Err(ModelError::EngineInvariant {
                    what: format!("select 图输出槽长 {} < rows {rows}", sel.len()),
                });
            }
            owl_shared::log_debug!(
                "dflash-probe",
                "切档 propose 收割(图):两半 {} + {} → 全表 {rows} 行 top-{k}",
                half,
                half
            );
            return Ok(sel[..rows].iter().map(|&v| v as u32).collect());
        }

        // ---- 兜底:host 合并 + eager select（图缺席时的原路,语义逐字不变）----
        let t_r0 = std::time::Instant::now();
        // rank0 两槽（`topk` dtoh、`sel_proj` 留设备块）
        let (v0, c0, proj_blk) = {
            let pg = self
                .dev
                .graphs
                .dflash
                .as_mut()
                .ok_or_else(|| ModelError::EngineInvariant { what: "dflash_graph 缺席(档位不一致)".into() })?;
            let t = pg.read_output_f32("topk").await?;
            let p = pg.output_block("sel_proj").ok_or_else(|| ModelError::EngineInvariant {
                what: "切档 propose sel_proj 输出槽缺席(先 step)".into(),
            })?;
            if t.len() != 2 * rows * k {
                return Err(ModelError::EngineInvariant {
                    what: format!("切档 propose topk 槽尺寸不符:{} ≠ {}", t.len(), 2 * rows * k),
                });
            }
            (t[..rows * k].to_vec(), t[rows * k..].to_vec(), p)
        };
        mrec("spec.propose.sel.read0", t_r0.elapsed());
        let t_merge = std::time::Instant::now();
        let (v1, c1) = (peer[..rows * k].to_vec(), peer[rows * k..].to_vec());
        let (mv, mc) = crate::running::merge_topk_two_halves(&v0, &c0, &v1, &c1, rows, k, half)?;
        mrec("spec.propose.sel.merge", t_merge.elapsed());
        let t_eval = std::time::Instant::now();
        // rank0 select（eager;全表域 —— 候选已合并、码本全量）
        let d2 = match self.dev.spec_rt.drafter.as_ref() {
            Some(crate::spec_runtime::Drafter::DFlash2(d)) => std::sync::Arc::clone(d),
            _ => {
                return Err(ModelError::EngineInvariant {
                    what: "切档 propose 收割:草稿器缺席/非 DFlash2".into(),
                })
            }
        };
        let rank = d2.selector_rank();
        if mv.len() != rows * k || mc.len() != rows * k || proj_blk.len != rows * rank {
            return Err(ModelError::EngineInvariant {
                what: format!(
                    "切档 propose 槽形状不符:vals {} / ids {} / proj {}(期望 {}×{}/{})",
                    mv.len(),
                    mc.len(),
                    proj_blk.len,
                    rows,
                    k,
                    rows * rank
                ),
            });
        }
        let f32b = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|f| f.to_le_bytes()).collect() };
        let cand_v: Vec<f32> = mc.iter().map(|&i| i as f32).collect();
        let cand_t = owl_models::TensorOps::from_host(Dtype::F32, vec![rows, k], &f32b(&cand_v));
        let vals_t = owl_models::TensorOps::from_host(Dtype::F32, vec![rows, k], &f32b(&mv));
        // proj = 图输出块的**只读**叶子（dtype/形状取自槽事实面，零往返）
        let proj_t = owl_models::TensorOps::of_block(proj_blk.id, proj_blk.dtype, vec![rows, rank]);
        let anchor_t = owl_models::TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[bonus as f32]));
        let (drafts, _scores) =
            d2.selector()
                .select_from_candidates(&cand_t, &vals_t, &proj_t, &anchor_t, rows);
        let t_prep = t_eval.elapsed();
        let face = self.dev.graphs.session.face_mut();
        let (b, arena) = owl_models::interpreters::eval_ops_scoped_env(
            drafts.step(),
            face,
            self.dev.spec_rt.draft_env,
            self.dev.boot.tp.as_ref(),
        )
        .await?;
        let t_launch = t_eval.elapsed();
        let mut buf = vec![0u8; 4 * rows];
        let face = self.dev.graphs.session.face_mut();
        face.dtoh(&b, &mut buf).await?;
        face.free(&[b.id]).await?;
        face.free(&arena).await?;
        mrec("spec.propose.sel.prep", t_prep);
        mrec("spec.propose.sel.launch", t_launch - t_prep);
        owl_shared::log_debug!(
            "dflash-probe",
            "切档 propose 收割(eager 兜底):候选合并 {}+{} → 全表 {rows} 行 top-{k}",
            half,
            half
        );
        mrec("spec.propose.sel.eval", t_eval.elapsed());
        Ok(buf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u32)
            .collect())
    }

    /// mtp 链表 ensure(E5-M2b;propose eager 路径用 from_host 表,非持久
    /// bt 块 —— 图不烘焙 mtp 链)。返回 host 块链副本。
    pub(crate) fn mtp_chain_bt(&mut self, cover: usize) -> Result<Vec<u32>> {
        let sid = self.turns.active()?.session_id;
        let s = self.bank.sessions.get_mut(sid).expect("账在");
        // FIXME(store-hook): H2 槽兜底旁路(MTP 链覆盖);
        // 应改为 Acquire(Grow) 预留后纯分配。
        self.bank.leg_mtp.ensure_for_len(&mut s.mtp_block_table, cover)?;
        Ok(s.mtp_block_table.clone())
    }


    /// verify 块前向(E5-M2b):prefill 通道 T = 块长(base = F,FI 表四件套
    /// 同款)。双树求值:树一 = last_hidden(根块存活返回 —— extend 同迭代
    /// 消费);树二 = lm_head → 逐行 argmax → 行栈(引用树一根)。返回
    /// (各行 argmax = 各位置贪心提名,hidden 块 [t, hidden];调用方在
    /// extend 后显式 free)。
    /// **eager verify(回滚/重放路)** —— 双 rank 镜像(P2.1-c):设备半在
    /// [`crate::engine::RankDev::verify_forward_dev`];单卡档 = 原路(零分支)。
    /// `pos_delta` = VLM m-RoPE 偏移(只加 rope pos;KV slots/lens 用 `base`)。
    pub(crate) async fn verify_forward(
        &mut self,
        block: &[u32],
        base: usize,
        gdn_slot: usize,
        want_logits: bool,
        pos_delta: i64,
    ) -> Result<(Vec<u32>, owl_iface::contract::Bytes, Option<Vec<f32>>)> {
        let bt_chain = {
            let act = self.turns.active()?;
            let s = self.bank.sessions.get(act.session_id).expect("账在");
            s.block_table.clone()
        };
        // P1.1 词表切:切档 ⇒ 两 rank 都读半表行(rank0 host 拼接);
        // 未切档 = 调用方原值(peer 那份 D2H 不付)。
        // 实效判据 = 本 rank logits 宽 < 全量(不读意图位;在借用之前取)。
        let head_split = self.head_split_active();
        let (dev, peer) = (&mut self.dev, self.peer.as_mut());
        let want_logits = want_logits || head_split;
        let half_vocab = dev.boot.model.logits_vocab();
        let out = match peer {
            None => {
                dev.verify_forward_dev(block, base, gdn_slot, want_logits, &bt_chain, pos_delta)
                    .await?
            }
            Some(p) => {
                // 两半都校验(纪律:`join2` 等另一半到底,吞错 = 挂住)
                let bt_p = bt_chain.clone();
                let peer_fut = async move {
                    let r = p
                        .verify_forward_dev(block, base, gdn_slot, want_logits, &bt_p, pos_delta)
                        .await;
                    if let Err(ref e) = r {
                        owl_shared::log_error!("spec-mirror", "peer verify_forward 失败: {e}");
                    }
                    r
                };
                let (r0, r1) = crate::exec::join2(
                    dev.verify_forward_dev(block, base, gdn_slot, want_logits, &bt_chain, pos_delta),
                    peer_fut,
                )
                .await;
                let (_, _, lg1) = r1?;
                let (ids0, hid0, lg0) = r0?;
                if head_split {
                    // 半表行 → 全表行(rank0 低半 ++ rank1 高半),再按全表重算
                    // 每行 argmax —— 设备 ids 在切档下是**局部** index,不可用。
                    let (a, b) = match (lg0, lg1) {
                        (Some(a), Some(b)) => (a, b),
                        _ => {
                            return Err(owl_iface::contract::ModelError::EngineInvariant {
                                what: "词表切 verify_forward:半表行缺席(镜像/读面不一致)"
                                    .into(),
                            })
                        }
                    };
                    let full = crate::running::concat_head_rows(&a, &b, half_vocab)?;
                    let v = dev.boot.model.vocab_size();
                    if full.len() < block.len() * v {
                        return Err(owl_iface::contract::ModelError::EngineInvariant {
                            what: format!(
                                "词表切 verify_forward:全表行尺寸不符 {} < {}×{}",
                                full.len(),
                                block.len(),
                                v
                            ),
                        });
                    }
                    let ids: Vec<u32> = (0..block.len())
                        .map(|r| {
                            full[r * v..(r + 1) * v]
                                .iter()
                                .enumerate()
                                .fold((0usize, f32::NEG_INFINITY), |a, (i, &x)| {
                                    if x > a.1 {
                                        (i, x)
                                    } else {
                                        a
                                    }
                                })
                                .0 as u32
                        })
                        .collect();
                    (ids, hid0, Some(full))
                } else {
                    (ids0, hid0, lg0)
                }
            }
        };
        Ok(out)
    }

    // ====================================================================
    // 取证探针(P1.5:仪表从 execute_decode 抽出;全部 probes 门控,
    // 正常路径零开销。TS 探针在声明层,见 model.rs last_hidden)
    // ====================================================================





}

use crate::session::SessionBank;

impl<D: DeviceClient> crate::engine::RankDev<D> {
    pub(crate) async fn prefill_dflash_encode(
        &mut self,
        probes: crate::running::StepProbes,
        tap_views: &[TensorOps],
        base: usize,
    ) -> Result<()> {
        let d2 = match self.spec_rt.drafter.as_ref() {
            Some(crate::spec_runtime::Drafter::DFlash2(d)) => std::sync::Arc::clone(d),
            _ => unreachable!("prefill_dflash_encode 非 dflash 模式"),
        };
        let rope = self.spec_rt.draft_rope.as_ref().expect("draft rope").clone();
        let t = tap_views[0].shape()[0];
        let f32b = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|f| f.to_le_bytes()).collect() };
        let pos_t = TensorOps::from_host(
            Dtype::F32,
            vec![t],
            &f32b(&(0..t).map(|i| (base + i) as f32).collect::<Vec<_>>()),
        );
        // 草稿池 ring 寻址(2026-10-22):slot = 逻辑位 % ring_slots;NC 核
        // 前缀读同样按 ring 映射(窗口 2048 + 自块)。bt 物理槽仅首 session
        // 重合 —— slot≥1 案修,2026-10-10。
        let _page = self.pool.geom.page;
        let leaves = self.pool.dflash_kv_leaves().expect("dflash 池");
        let bt = self.pool.bt_leaf_flat();
        let kvs: Vec<KvBuffers> = leaves
            .iter()
            .map(|(k, v)| KvBuffers {
                k_cache: k.clone(),
                v_cache: v.clone(),
                slots: TensorOps::from_host(
                    Dtype::F32,
                    vec![t],
                    &f32b(&(0..t)
                        .map(|i| self.pool.draft_dflash_encode_slot(base, i))
                        .collect::<Vec<_>>()),
                ),
                kv_lens: TensorOps::from_host(
                    Dtype::F32,
                    vec![t],
                    &f32b(&(0..t).map(|i| (base + i + 1) as f32).collect::<Vec<_>>()),
                ),
                block_tables: bt.clone(),
            })
            .collect();
        let ctx = ForwardCtx::minimal(t);
        let memory = d2.project_memory(tap_views, &ctx);
        let roots = d2.encode_kv(&memory, &pos_t, &kvs, &rope, &ctx);
        {
            let face = self.graphs.session.face_mut();
            let refs: Vec<&TensorOps> = roots.iter().collect();
            // 副作用根口径(同 encode 主路:末位出参是契约 4 占位,没人读)
            owl_models::interpreters::eval_ops_side_effect_env(&refs, face, self.spec_rt.draft_env, self.boot.tp.as_ref()).await?;
        }
        // 探针:首回合 memory 落盘(python 重算 fc 对拍;真块 dtoh 合法。
        // taps 由调用方从 concat 父块整块落盘 —— 切片视图 dtoh 整父块契约)
        if probes.dflash_probe && !self.spec_rt.mem_dumped {
            self.spec_rt.mem_dumped = true;
            let face = self.graphs.session.face_mut();
            let refs: Vec<&TensorOps> = std::iter::once(&memory).collect();
            let outs = owl_models::interpreters::eval_ops_multi(&refs, face, self.boot.tp.as_ref()).await?;
            let n: usize = memory.shape().iter().product();
            let mut buf = vec![0u8; n * 2];
            face.dtoh(&outs[0], &mut buf).await?;
            if let Err(e) = owl_shared::file_loader::write(&crate::probe_dump::probe_path("owl_dflash_mem.bin"), &buf) {
                owl_shared::log_error!("dump", "落盘失败(诊断面不许静默): {e}");
            }
            owl_shared::log_debug!("dflash-probe", "memory 落盘 /tmp/owl_dflash_mem.bin t={t}");
        }
        Ok(())
    }

    pub(crate) async fn prefill_extend(
        &mut self,
        bank: &mut SessionBank,
        sid: u64,
        hid: &Bytes,
        t_rows: usize,
        ids: &[u32],
        base: usize,
    ) -> Result<()> {
        let t_all = std::time::Instant::now();
        // DFlash2 早退(无链式 extend;草稿 KV 走 prefill_dflash_encode)
        // —— 门必须先于 blocks_mtp 预留(mtp 账房非 mtp 模式 = 0 块)
        if !matches!(self.spec_rt.drafter.as_ref(), Some(crate::spec_runtime::Drafter::Mtp(_))) {
            return Ok(());
        }
        let d = self.pool.geom.dims;
        let page = self.pool.geom.page;
        let bt = {
            // FIXME(store-hook): H2 槽兜底旁路(MTP 批量行);
            // 应改为 Acquire(Grow) 预留后纯分配。
            bank.leg_mtp
                .ensure_for_len(&mut bank.sessions.session_mut(sid)?.mtp_block_table, base + t_rows)?;
            bank.sessions.session(sid)?.mtp_block_table.clone()
        };
        let (kc, vc) = self.pool.mtp_kv_leaf().ok_or_else(|| ModelError::EngineInvariant { what: "MTP KV 池句柄缺席(draft 档位不一致)".into() })?;
        let f32b = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|f| f.to_le_bytes()).collect() };
        let tokens: Vec<u32> = ids[1..=t_rows].to_vec();
        let tok_t = TensorOps::from_host(
            Dtype::F32,
            vec![t_rows],
            &tokens.iter().flat_map(|v| (*v as f32).to_le_bytes()).collect::<Vec<u8>>(),
        );
        let hid_view = TensorOps::of_block(hid.id, d.dtype, vec![t_rows, d.hidden]);
        let pos_t = TensorOps::from_host(
            Dtype::F32,
            vec![t_rows],
            &f32b(&(base..base + t_rows).map(|p| p as f32).collect::<Vec<_>>()),
        );
        let slots_t = TensorOps::from_host(
            Dtype::F32,
            vec![t_rows],
            &f32b(
                &(base..base + t_rows)
                    .map(|p| (bt[p / page] * page as u32 + (p % page) as u32) as f32)
                    .collect::<Vec<_>>(),
            ),
        );
        let lens_t = TensorOps::from_host(
            Dtype::F32,
            vec![t_rows],
            &f32b(&((base + 1)..=base + t_rows).map(|p| p as f32).collect::<Vec<_>>()),
        );
        let cu_t = TensorOps::from_host(
            Dtype::F32,
            vec![2],
            &f32b(&[0.0, t_rows as f32]),
        );
        let kv = owl_models::module::KvBuffers {
            k_cache: kc.clone(),
            v_cache: vc.clone(),
            slots: slots_t.clone(),
            kv_lens: lens_t.clone(),
            block_tables: TensorOps::from_host(
                Dtype::F32,
                vec![1, bt.len()],
                &f32b(&bt.iter().map(|&b| b as f32).collect::<Vec<_>>()),
            ),
        };
        mrec("spec.extend.pre", t_all.elapsed());
        let ts_eval = std::time::Instant::now();
        let mtp = match self.spec_rt.drafter.as_ref() {
            Some(crate::spec_runtime::Drafter::Mtp(m)) => m,
            _ => return Ok(()), // DFlash2 无链式 extend(草稿 KV 走 dflash_encode)
        };
        // MTP 链池是 F16；主 env 生产是 fp8，必须显式用草稿 env
        // （否则 K0 写核选 fp8 变体，槽 2 的 F16 cache 当场 dtype 拒绝）。
        let env = self.spec_rt.draft_env;
        let root = mtp.extend_block(
            &tok_t, &hid_view, &pos_t, &kv, &slots_t, &lens_t, &cu_t,
            &self.boot.model.embed, &self.boot.rope, env,
        );
        // 原为 `_b`(弃值)—— 那正是漏点的形状:根块没人读也没人还
        let (b, arena) = {
            let face = self.graphs.session.face_mut();
            owl_models::interpreters::eval_ops_scoped_env(root.step(), face, env, self.boot.tp.as_ref()).await?
        };
        mrec("spec.extend.eval", ts_eval.elapsed());
        mrec("spec.extend", t_all.elapsed());
        {
            let face = self.graphs.session.face_mut();
            // **根块一并无条件归还**:本路只在图外扩展草稿池槽,根的值**无人读**
            // (只借它触发执行)—— 单根 scoped 的 root 不在 `arena` 候选里,
            // 不显式还就是每轮一条的漏(turn 列爬坡的第二条指纹)。
            let mut ids = arena;
            ids.push(b.id);
            face.free(&ids).await?;
        }
        Ok(())
    }







    // ====================================================================
    // 取证探针(P1.5:仪表从 execute_decode 抽出;全部 probes 门控,
    // 正常路径零开销。TS 探针在声明层,见 model.rs last_hidden)
    // ====================================================================


}





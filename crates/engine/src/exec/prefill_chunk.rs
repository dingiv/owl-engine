//! 块式 prefill(树根装配与执行;taps/encode/BISECT 取证)
use super::*;

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
            let act = self.active.as_ref().expect("active 已保证");
            let s = self.sessions.get(act.session_id).expect("账在");
            (s.block_table.clone(), s.gdn_slot)
        };
        let f32seq = |start: usize, n: usize| -> Vec<u8> {
            (0..n)
                .flat_map(|i| ((start + i) as f32).to_le_bytes())
                .collect()
        };
        let kv_caches = self.pool.kv_caches();
        let kvs_step: Vec<KvBuffers> = kv_caches
            .iter()
            .map(|(k, v)| KvBuffers {
                k_cache: k.clone(),
                v_cache: v.clone(),
                slots: TensorOps::from_host(Dtype::F32, vec![t], &f32seq(base, t)),
                kv_lens: TensorOps::from_host(Dtype::F32, vec![t], &f32seq(base + 1, t)),
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
                slots: TensorOps::from_host(Dtype::F32, vec![1], &f32seq(0, 1)),
            })
            .collect();
        let ids_t = TensorOps::from_host(
            Dtype::F32,
            vec![t],
            &ids.iter().flat_map(|v| (*v as f32).to_le_bytes()).collect::<Vec<u8>>(),
        );
        let pos_t = TensorOps::from_host(Dtype::F32, vec![t], &f32seq(base, t));
        // E2b:slots = 物理槽(块链换算);legacy 恒等直排不变
        let slots_t = if self.pool.paged {
            let page = self.pool.page;
            let phys: Vec<f32> = (base..base + t)
                .map(|pos| {
                    let b = bt_chain[pos / page];
                    (b * page as u32 + (pos % page) as u32) as f32
                })
                .collect();
            TensorOps::from_host(Dtype::F32, vec![t], &f32b(&phys))
        } else {
            TensorOps::from_host(Dtype::F32, vec![t], &f32seq(base, t))
        };
        let lens_t = TensorOps::from_host(Dtype::F32, vec![t], &f32seq(base + 1, t));
        let gdn_slot = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[gdn_slot_v as f32]));
        // FlashInfer 表四件套(i32 设备;OWL_FLASHINFER=1 时非空,E1.5):
        // q_cu=[0,T] / indices=块链 / indptr=[0,nb] / last_len=末页有效数。
        // ⚠️ 预物化(eval 往返同步)而非 from_host 叶子:Htod 在 H2D 流、
        // FI 核在 COMPUTE 流,无跨流序 —— 异步竞态会让 FI 偶发读到半写入
        // 表(渐进腐败案,2026-10-03;09-28 upload_pinned 同病史)。
        let i32le = |v: &[i32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let fi_tensors;
        let fi = if self.pool.fi_face {
            let page = self.pool.page;
            let ctx_total = base + t;
            let nb = bt_chain.len();
            let k_fi_leaf = self.pool.kv_fi_leaves();
            let face = self.session.face_mut();
            let q_cu_b = owl_models::interpreters::eval_ops(
                TensorOps::from_host(Dtype::U32, vec![2], &i32le(&[0, t as i32])).step(), face).await?;
            let idx_b = owl_models::interpreters::eval_ops(
                TensorOps::from_host(Dtype::U32, vec![nb],
                    &i32le(&bt_chain.iter().map(|&b| b as i32).collect::<Vec<_>>())).step(), face).await?;
            let ind_b = owl_models::interpreters::eval_ops(
                TensorOps::from_host(Dtype::U32, vec![2], &i32le(&[0, nb as i32])).step(), face).await?;
            let ll_b = owl_models::interpreters::eval_ops(
                TensorOps::from_host(Dtype::U32, vec![1],
                    &i32le(&[(ctx_total - nb * page + page) as i32])).step(), face).await?;
            let q_cu = TensorOps::of_block(q_cu_b.id, Dtype::U32, vec![2]);
            let indices = TensorOps::of_block(idx_b.id, Dtype::U32, vec![nb]);
            let indptr = TensorOps::of_block(ind_b.id, Dtype::U32, vec![2]);
            let last_len = TensorOps::of_block(ll_b.id, Dtype::U32, vec![1]);
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
        let mut ctx = ForwardCtx::model_prefill(
            t, &pos_t, &kvs_step, &self.rope, &gdns_step, &slots_t, &lens_t, &gdn_slot, base, fi,
        );
        ctx.env = self.env;
        ctx.trace_gate = self.cfg.knobs.probes.trace_gate;
        let vocab = self.model.vocab_size();
        // E5-DF4 逐层 bisect(OWL_PF_BISECT;非 dflash 路径):单遍 multi
        // eval 产出 48 层 taps + fin —— **fin 块直接作为主路输出**(bisect
        // 前向有 GDN 状态副作用,双执行会双推进;单执行 + 逐层 checksum
        // → metrics `pf.c{base}.l{i}`,双 boot/双跑 diff = 首个发散层)。
        // fin block-leaf 下游 eval = 零操作透传。
        let pf_bisect = self.cfg.knobs.pf_bisect && self.dflash_tap_count == 0;
        // GDN 状态 checksum(pre/post;崩坏排查:状态脏 = reset 病,状态净
        // 而输出异 = 核病)。rec [slots,nv,kd,vd] f32 全量 dtoh ≈ 3.1MB/层。
        let pf_state_check = pf_bisect;
        // 指定层五站 stage tap(OWL_PF_STAGES=层号;配合 BISECT 定位层内核)
        let pf_stage_layer: Option<usize> = self.cfg.knobs.pf_stages;
        let mut bisect_fin_block: Option<(TensorOps, owl_iface::contract::Bytes)> = None;
        let mut bisect_fin_blk: Option<owl_iface::contract::Bytes> = None;
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
                            let face = self.session.face_mut();
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
            let ids_all: Vec<usize> = (0..self.model.layers.len()).collect();
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
                let face = self.session.face_mut();
                owl_models::interpreters::eval_ops_multi_env(&refs, face, self.env).await?
            };
            for (i, r) in roots.iter().enumerate() {
                let n: usize = r.shape().iter().product();
                let mut b = vec![0u8; n * 2];
                {
                    let face = self.session.face_mut();
                    let res = face.dtoh(&outs[i], &mut b).await;
                    if let Err(e) = res {
                        eprintln!("[pf-bisect] dtoh 失败 i={i} name={} shape={:?} blk_bytes={} err={e}", names[i], r.shape(), b.len());
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
                            let face = self.session.face_mut();
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
            eprintln!("[pf-bisect] chunk base={base} t={t} 层 checksum 已记");
        }
        let face = self.session.face_mut();
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
                bisect_fin_blk = Some(blk);
                (decl, None)
            } else if self.dflash_tap_count > 0 {
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
                let face = self.session.face_mut();
                eval_ops_scoped_env(hidden_full.step(), face, self.env).await?
            };
            // E5 崩坏排查·轻探针(OWL_PF_FIN_CHECK;bisect 900 dtoh 强掩蔽
            // 0/10,本探针仅 1 dtoh —— 近零扰判决:fin 翻 = 前向内部腐坏;
            // fin 绿而 pf.last 翻 = lm_head/argmax 面腐坏)
            if self.cfg.knobs.pf_fin_check && !pf_bisect {
                let n_fin = t * d_model;
                let mut fb = vec![0u8; n_fin * 2];
                {
                    let face = self.session.face_mut();
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
                if self.probes.dflash_probe && !self.dflash_mem_dumped2 {
                    self.dflash_mem_dumped2 = true;
                    let face = self.session.face_mut();
                    let mut rbuf = vec![0u8; (n_tp + 1) * t * d_model * 2];
                    face.dtoh(&hfb, &mut rbuf).await?;
                    owl_shared::file_loader::write("/tmp/owl_dflash_root.bin", &rbuf).ok();
                    eprintln!("[dflash-probe] concat 父块落盘 /tmp/owl_dflash_root.bin t={t}");
                }
                eprintln!("[dflash] prefill encode {t} rows @{} {:?}", base, ts.elapsed());
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
                let face = self.session.face_mut();
                eval_ops_scoped_env(last.step(), face, self.env).await?
            };
            let hb_view = TensorOps::of_block(hb.id, d.dtype, vec![1, d_model]);
            let (b, arena3) = {
                let face = self.session.face_mut();
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
                    eprintln!("[pf-probe] boot{bs_seq} 同boot两次lm_head一致={same_inproc}");
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
                    eprintln!(
                        "[pf-probe] boot{} argmax(top2)={} gap={:.5} top=({} {:.4}) ({:.4})",
                        bs_seq,
                        top[0].1,
                        top[0].0 - top[1].0,
                        top[0].1, top[0].0, top[1].0
                    );
                }
                let tok = owl_models::ops::argmax_f32idx(&logits, vocab, 0);
                eval_ops_scoped_env(tok.step(), face, self.env).await?
            };
            if self.drafter.is_some() && t > 1 {
                // prefill extend(E5-M5):本 chunk 行 0..T-1 的 mtp KV
                // (行 T-1 的配对需下一 chunk 首 token,由其 extend 补;
                // 末位 P-1 由首轮 propose_first 的 extend 行补)
                self.prefill_extend(&hfb, t - 1, ids, base).await?;
            }
            let buf: [u8; 4];
            let mut seed_keep: Option<owl_iface::contract::Bytes> = None;
            {
                let face = self.session.face_mut();
                let mut buf_t = [0u8; 4];
                face.dtoh(&b, &mut buf_t).await?;
                face.free(&[b.id]).await?;
                face.free(&arena3).await?;
                face.free(&arena2).await?;
                if self.drafter.is_some() {
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
                bisect_fin_blk = Some(blk);
                (decl, None)
            } else if self.dflash_tap_count > 0 {
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
                let face = self.session.face_mut();
                eval_ops_scoped_env(tree.step(), face, self.env).await?
            };
            {
                let face = self.session.face_mut();
                let esz = if d.dtype == Dtype::F16 { 2 } else { 4 };
                // DFlash2:b = concat 根 [n_tp·t + t, hidden](行 0 块 =
                // hidden,taps 同块后随)—— dtoh 整父块契约必须等大读
                // (曾用 t·hidden → 多 chunk DFlash2 首跑即炸;MTP 无 taps
                // 时 n_tp=0 同式兼容)
                let rows_b = if prefill_taps.is_some() { prefill_taps.as_ref().unwrap().0.len() + 1 } else { 1 } * t;
                let mut buf = vec![0u8; rows_b * d.hidden * esz];
                face.dtoh(&b, &mut buf).await?;
                if self.probes.prefill_cksum {
                    // 临时取证:每 chunk 隐层校验和(FI 开/关对比找第一分歧)
                    let cks: f32 = buf.chunks_exact(2)
                        .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32().abs())
                        .sum();
                    eprintln!("[cksum] chunk base={base} t={t} sum|x|={cks:.4}");
                }
                drop(face);
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
                } else if self.drafter.is_some() && t > 1 {
                    let ts = std::time::Instant::now();
                    self.prefill_extend(&b, t - 1, ids, base).await?;
                    t_ext += ts.elapsed();
                }
                let face = self.session.face_mut();
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

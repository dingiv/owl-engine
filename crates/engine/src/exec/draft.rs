//! 草稿器面(DFlash2 单图 encode+propose / eager / MTP 链 / verify 前向)
use super::*;

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
    ) -> Result<Vec<u32>> {
        let fp = pos + m + 1;
        let page = self.pool.page;
        {
            let act = self.active.as_ref().expect("active 已保证");
            let s = self.sessions.get_mut(act.session_id).expect("账在");
            let before = s.block_table.len();
            self.blocks_m.ensure_for_len(&mut s.block_table, fp + 8)?;
            if s.block_table.len() != before {
                let sid = act.session_id;
                self.write_bt(sid).await?;
            }
        }
        // 草稿池寻址一律经 StatePool 权威(slot≥1 案配套;分派点零推槽)
        use crate::state::StatePool as SP;
        let slot_at = |p: usize| SP::draft_prefix_slot(p);
        let _ = page;
        let f32b = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|f| f.to_le_bytes()).collect() };
        let tokens_f: Vec<f32> = (0..8)
            .map(|i| if i == 0 { bonus as f32 } else { owl_models::layers::dflash2::MASK_TOKEN_ID as f32 })
            .collect();
        let enc_pos: Vec<f32> = (0..8).map(|i| (pos + i) as f32).collect();
        let enc_slots: Vec<f32> = (0..8).map(|i| slot_at(pos + i)).collect();
        // 逐行可见长(含自身;eager dflash_encode 同式 —— AL 劣化案修复)
        let enc_kv_lens: Vec<f32> = (0..8).map(|i| (pos + i + 1) as f32).collect();
        let prop_pos: Vec<f32> = (0..8).map(|i| (fp + i) as f32).collect();
        let prop_slots: Vec<f32> = (0..8).map(|i| slot_at(fp + i)).collect();
        // propose 噪声块逐行可见长(eager dflash_propose 同式 fp+1+i)
        let prop_kv_lens: Vec<f32> = (0..8).map(|i| (fp + 1 + i) as f32).collect();
        let kv_len = vec![(fp + 8) as f32];
        let anchor = vec![bonus as f32];
        let pg = self.dflash_graph.as_mut().expect("dflash_graph");
        let prof = self.probes.step_profile;
        let t0 = prof.then(std::time::Instant::now);
        pg.step(&[
            ("tokens", tokens_f.as_slice()),
            ("anchor", anchor.as_slice()),
            ("enc_pos", enc_pos.as_slice()),
            ("enc_slots", enc_slots.as_slice()),
            ("enc_kv_lens", enc_kv_lens.as_slice()),
            ("prop_pos", prop_pos.as_slice()),
            ("prop_slots", prop_slots.as_slice()),
            ("prop_kv_lens", prop_kv_lens.as_slice()),
            ("kv_len", kv_len.as_slice()),
        ])
        .await?;
        let t1 = prof.then(std::time::Instant::now);
        let out = pg.read_output_f32("drafts").await?;
        // 纯 GPU 判别(E5-DF4 性能):首读含队列排水(verify 尾部);同步后
        // 重发一次图再读 = 图的独占 GPU 时间
        if prof {
            let t2 = std::time::Instant::now();
            pg.replay().await?;
            let _ = pg.read_output_f32("drafts").await?;
            eprintln!("[dflash-prof] pure_gpu={:?}", t2.elapsed());
        }
        if let (Some(a), Some(b)) = (t0, t1) {
            // E2 语义勘误(原 fill/launch/read 误导):total = 轮内全程;
            // step = 装填+发射段;read_extra = 首读段(含队列排水);
            // pure_gpu(上行)= 同步后裸 replay = 图独占 GPU 时间
            eprintln!(
                "[dflash-prof] total={:?} step={:?} read_extra={:?}",
                a.elapsed(),
                b - a,
                b.elapsed()
            );
        }
        Ok(out[..7].iter().map(|&v| v as u32).collect())
    }


    /// DFlash2 prefill encode(prefill chunk 全行草稿 KV 物化):taps
    /// [T, hidden] 视图 → memory → 5 层 kv-only 写 @ base..base+T。
    pub(crate) async fn prefill_dflash_encode(
        &mut self,
        tap_views: &[TensorOps],
        base: usize,
    ) -> Result<()> {
        let d2 = match self.drafter.as_ref() {
            Some(crate::running::Drafter::DFlash2(d)) => std::sync::Arc::clone(d),
            _ => unreachable!("prefill_dflash_encode 非 dflash 模式"),
        };
        let rope = self.draft_rope.as_ref().expect("draft rope").clone();
        let t = tap_views[0].shape()[0];
        let f32b = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|f| f.to_le_bytes()).collect() };
        let pos_t = TensorOps::from_host(
            Dtype::F32,
            vec![t],
            &f32b(&(0..t).map(|i| (base + i) as f32).collect::<Vec<_>>()),
        );
        // 草稿池线性逻辑寻址(slot = 逻辑位;NC 前缀读 [0, prefix) 直排;
        // bt 物理槽仅首 session 重合 —— slot≥1 案修,2026-10-10)
        let (bt_chain, _) = {
            let act = self.active.as_ref().expect("active 已保证");
            let s = self.sessions.get(act.session_id).expect("账在");
            (s.block_table.clone(), s.gdn_slot)
        };
        let _ = bt_chain;
        let page = self.pool.page;
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
                        .map(|i| crate::state::StatePool::draft_encode_slot(base, i))
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
            let face = self.session.face_mut();
            let refs: Vec<&TensorOps> = roots.iter().collect();
            owl_models::interpreters::eval_ops_multi_scoped_env(&refs, face, self.env).await?;
        }
        // 探针:首回合 memory 落盘(python 重算 fc 对拍;真块 dtoh 合法。
        // taps 由调用方从 concat 父块整块落盘 —— 切片视图 dtoh 整父块契约)
        if self.probes.dflash_probe && !self.dflash_mem_dumped {
            self.dflash_mem_dumped = true;
            let face = self.session.face_mut();
            let refs: Vec<&TensorOps> = std::iter::once(&memory).collect();
            let outs = owl_models::interpreters::eval_ops_multi(&refs, face).await?;
            let n: usize = memory.shape().iter().product();
            let mut buf = vec![0u8; n * 2];
            face.dtoh(&outs[0], &mut buf).await?;
            owl_shared::file_loader::write("/tmp/owl_dflash_mem.bin", &buf).ok();
            eprintln!("[dflash-probe] memory 落盘 /tmp/owl_dflash_mem.bin t={t}");
        }
        Ok(())
    }


    /// DFlash2 propose(E5-DF3):噪声块 [anchor, MASK×7] @ pos..pos+7,
    /// 草稿 KV 窗口 [0, pos+8)(前缀 pos 行已物化 + 自块 8 行直读)。
    /// 返回 (草稿 host 表, 观测面块)。首轮(pos = fed)与轮末同式。
    pub(crate) async fn dflash_propose(&mut self, anchor: u32, pos: usize) -> Result<Vec<u32>> {
        let d2 = match self.drafter.as_ref() {
            Some(crate::running::Drafter::DFlash2(d)) => std::sync::Arc::clone(d),
            _ => unreachable!("dflash_propose 非 dflash 模式"),
        };
        let rope = self.draft_rope.as_ref().expect("draft rope").clone();
        let f32b = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|f| f.to_le_bytes()).collect() };
        // 噪声块 token/位置表(8 行定形)
        let mut toks = vec![anchor as f32];
        toks.resize(8, owl_models::layers::dflash2::MASK_TOKEN_ID as f32);
        let toks_t = TensorOps::from_host(Dtype::F32, vec![8], &f32b(&toks));
        let pos_t = TensorOps::from_host(
            Dtype::F32,
            vec![8],
            &f32b(&(0..8).map(|i| (pos + i) as f32).collect::<Vec<_>>()),
        );
        let anchor_t = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[anchor as f32]));
        // 草稿 KV 叶子;自块槽表 = **物理槽 pos..pos+8**(sglang
        // assign_extend_cache_locs [prefix_len, prefix_len+block_size) 同位;
        // 曾误填 0..8 —— 每轮把位置 0..7 的 memory 前缀 KV 磨成噪声块
        // 残写,永久污染窗口头部 → AL=0 案真凶,2026-10-08 定谳)
        let leaves = self.pool.dflash_kv_leaves().expect("dflash 池");
        let bt = self.pool.bt_leaf_flat();
        let (bt_chain, _) = {
            let act = self.active.as_ref().expect("active 已保证");
            let s = self.sessions.get(act.session_id).expect("账在");
            (s.block_table.clone(), s.gdn_slot)
        };
        let page = self.pool.page;
        // 块表覆盖自保(⑦ propose 跑在 fp = pos+m+1 = 下轮 fed 位,先于
        // 下轮 scheduler 的 ensure_for_len —— 跨页时 bt 未长,曾于 pos+8
        // 越 page 界 panic)。**grew → write_bt 同步律**:块表增长必须
        // 伴设备侧重写,否则 scheduler 下轮 grew=false 跳过 write_bt,
        // 烘焙块表停代(gate9 文本分歧 / gate10 空轮两案真凶)
        {
            let act = self.active.as_ref().expect("active 已保证");
            let s = self.sessions.get_mut(act.session_id).expect("账在");
            let before = s.block_table.len();
            self.blocks_m.ensure_for_len(&mut s.block_table, pos + 8)?;
            if s.block_table.len() != before {
                let sid = act.session_id;
                self.write_bt(sid).await?;
            }
        }
        let (bt_chain, _) = {
            let act = self.active.as_ref().expect("active 已保证");
            let s = self.sessions.get(act.session_id).expect("账在");
            (s.block_table.clone(), s.gdn_slot)
        };
        if self.probes.dflash_probe {
            eprintln!("[dflash-probe] propose pos={pos} bt_len={} page={page}", bt_chain.len());
        }
        // 噪声块自槽(经寻址权威;线性逻辑位)
        let self_slots: Vec<f32> = (0..8usize)
            .map(|i| crate::state::StatePool::draft_self_slot(pos, i))
            .collect();
        let kvs: Vec<KvBuffers> = leaves
            .iter()
            .map(|(k, v)| KvBuffers {
                k_cache: k.clone(),
                v_cache: v.clone(),
                slots: TensorOps::from_host(Dtype::F32, vec![8], &f32b(&self_slots)),
                kv_lens: TensorOps::from_host(Dtype::F32, vec![8], &f32b(&(0..8).map(|i| (pos + 1 + i) as f32).collect::<Vec<_>>())),
                block_tables: bt.clone(),
            })
            .collect();
        let mut ctx = ForwardCtx::minimal(8);
        ctx.env = self.env;
        // 探针:草稿池 K 前缀行 NaN 检查(定谳 encode vs NC 核;解析 dtype
        // 随草稿池 —— 同日十四后 = BF16)
        if self.probes.dflash_probe {
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
                let face = self.session.face_mut();
                owl_models::interpreters::eval_ops(kslice.step(), face).await?
            };
            let mut kbuf = vec![0u8; 786432]; // 整池块(dtoh 强等;12页×8×128×32×2B)
            {
                let face = self.session.face_mut();
                face.dtoh(&kb, &mut kbuf).await?;
            }
            if pos <= 18 {
                owl_shared::file_loader::write("/tmp/owl_dflash_kpool.bin", &kbuf).ok();
                eprintln!("[dflash-probe] 草稿 K 池首块落盘 /tmp/owl_dflash_kpool.bin pos={pos}");
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
                    eprintln!("[dflash-probe] draftK row{r} |x|前64和 = {s}");
                }
            }
        }
        // 诊断开关:OWL_DFLASH_DUMB=1 → 哑草稿(anchor 重复,跳过 draft
        // 前向)—— 隔离「草稿值泄漏」vs「轮管线污染」(2026-10-07 恒等门
        // AL=0 分歧排查;正式开关挂 C7)
        // OWL_DFLASH_DUMB=1:零副作用哑草稿(不进 GPU;② 直接 host 表)
        if self.probes.dflash_dumb {
            return Ok(vec![anchor; 7]);
        }
        let mut probe_roots: Vec<TensorOps> = Vec::new();
        let probe_on = self.probes.dflash_probe;
        let kv_len_t = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[(pos + 8) as f32]));
        let (hidden, drafts, _scores, logits) = d2.propose_block(
            &toks_t, &pos_t, &anchor_t, &kvs, &rope, &self.model.embed, &kv_len_t, &ctx,
            if probe_on { Some(&mut probe_roots) } else { None },
        );
        // 探针:首个 propose 的 hidden + 全部探针根落盘(python 逐阶段对拍;
        // **逐根独立 eval** —— multi 后段会把早根块回收复用,dtoh 读到陈旧
        // 块(root0 曾读出 1e6 级垃圾假象);单链重执行确定性,值可信)
        if self.probes.dflash_probe && !self.dflash_hid_dumped {
            self.dflash_hid_dumped = true;
            for (i, r) in probe_roots.iter().enumerate() {
                let face = self.session.face_mut();
                let b = owl_models::interpreters::eval_ops(r.step(), face).await?;
                let n: usize = r.shape().iter().product();
                let mut blk = vec![0u8; n * 2];
                face.dtoh(&b, &mut blk).await?;
                owl_shared::file_loader::write(format!("/tmp/owl_p{i}.bin"), &blk).ok();
            }
            let face = self.session.face_mut();
            let b = owl_models::interpreters::eval_ops(hidden.step(), face).await?;
            let n: usize = hidden.shape().iter().product();
            let mut hbuf = vec![0u8; n * 2];
            face.dtoh(&b, &mut hbuf).await?;
            owl_shared::file_loader::write("/tmp/owl_dflash_hid.bin", &hbuf).ok();
            let mut m = Vec::new();
            m.extend_from_slice(&(pos as u32).to_le_bytes());
            m.extend_from_slice(&(anchor as u32).to_le_bytes());
            owl_shared::file_loader::write("/tmp/owl_dflash_meta.bin", &m).ok();
            eprintln!("[dflash-probe] {} probe roots + hidden 落盘(逐根独立 eval)pos={pos} anchor={anchor}", probe_roots.len());
        }
        let _ = hidden;
        // 探针:logits 前 4 值 f16(NaN 定位;OWL_DFLASH_PROBE)
        let logits_probe = if self.probes.dflash_probe {
            Some(logits.clone())
        } else {
            None
        };
        let (b, arena) = {
            let face = self.session.face_mut();
            owl_models::interpreters::eval_ops_scoped_env(drafts.step(), face, self.env).await?
        };
        if let Some(lp) = logits_probe {
            // 单遍 multi(共享 memo;独立 eval 重执行副作用层 = 原地
            // fused_add 累加 = 探针自污染 —— testkit 重放语义警告同源)
            let mut roots: Vec<TensorOps> = probe_roots.clone();
            roots.push(lp);
            let refs: Vec<&TensorOps> = roots.iter().collect();
            let outs = {
                let face = self.session.face_mut();
                owl_models::interpreters::eval_ops_multi_env(&refs, face, self.env).await?
            };
            for (i, pr) in probe_roots.iter().enumerate() {
                let n: usize = pr.shape().iter().product();
                let mut pbuf = vec![0u8; n * 2];
                {
                    let face = self.session.face_mut();
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
                eprintln!("[dflash-probe] root{i} ({}) |x|前512和 = {s} head={head:?}", pr.shape().iter().map(|v| v.to_string()).collect::<Vec<_>>().join("x"));
            }
            let lb = outs.last().expect("logits root");
            let mut lbuf = vec![0u8; 7 * 248320 * 2];
            {
                let face = self.session.face_mut();
                face.dtoh(lb, &mut lbuf).await?;
            }
            let lvals: Vec<f32> = lbuf[..16]
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect();
            eprintln!("[dflash-probe] logits[0..4]={lvals:?}");
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
                top.select_nth_unstable_by(7, |a, b| b.0.partial_cmp(&a.0).unwrap());
                top.truncate(8);
                top.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                eprintln!("[dflash-probe] slot0 lm_head top8 = {:?}", top);
            }
        }
        let out = {
            let face = self.session.face_mut();
            // drafts 根 = select 单缓冲的 SliceView —— dtoh 回读整父块
            // [S·(K·K+1) f32],取前 S 个(dflash2.rs select 同坑,测试侧同修)
            let mut buf = vec![0u8; 7196];
            face.dtoh(&b, &mut buf).await?;
            face.free(&arena).await?;
            buf.chunks_exact(4)
                .take(7) // 父块 = select 单缓冲(SliceView dtoh 回整块);草稿仅前 7
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u32)
                .collect::<Vec<_>>()
        };
        let _ = b;
        if self.probes.dflash_probe {
            eprintln!("[dflash-probe] pos={pos} anchor={anchor} drafts={out:?}");
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
        // 行数钉 8(verify 块全行;taps 块仅 8 行,m=7 时 m+2=9 越界读 ——
        // 多编行下轮被自块写/重编覆盖,幂等安全;E5-DF4 与图路径同式)
        let rows = (m + 2).min(8);
        let rope = self.draft_rope.as_ref().expect("draft rope").clone();
        let dtype = self.pool.dims.dtype;
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
        let page = self.pool.page;
        let _ = page;
        let leaves = self.pool.dflash_kv_leaves().expect("dflash 池");
        let bt = self.pool.bt_leaf_flat();
        let kvs: Vec<KvBuffers> = leaves
            .iter()
            .map(|(k, v)| KvBuffers {
                k_cache: k.clone(),
                v_cache: v.clone(),
                slots: TensorOps::from_host(
                    Dtype::F32,
                    vec![rows],
                    &f32b(&(0..rows)
                        .map(|i| crate::state::StatePool::draft_encode_slot(pos, i))
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
        let probe = self.probes.dflash_probe;
        if probe {
            let b1 = {
                let face = self.session.face_mut();
                owl_models::interpreters::eval_ops(memory.step(), face).await?
            };
            {
                let mut vb = vec![0u8; rows * 5120 * 2];
                let face = self.session.face_mut();
                face.dtoh(&b1, &mut vb).await?;
                let mx = vb[..5120 * 2]
                    .chunks_exact(2)
                    .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32().abs())
                    .fold(0f32, f32::max);
                let nz = vb[..5120 * 2]
                    .chunks_exact(2)
                    .filter(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32() != 0.0)
                    .count();
                eprintln!("[enc-bisect] ①eval_ops(memory) 独立: 行0 非零={nz}/5120 |max|={mx:.4}");
            }
            let refs1 = vec![&memory];
            let o1 = {
                let face = self.session.face_mut();
                owl_models::interpreters::eval_ops_multi(&refs1, face).await?
            };
            {
                let mut vb = vec![0u8; rows * 5120 * 2];
                let face = self.session.face_mut();
                face.dtoh(&o1[0], &mut vb).await?;
                let mx = vb[..5120 * 2]
                    .chunks_exact(2)
                    .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32().abs())
                    .fold(0f32, f32::max);
                let nz = vb[..5120 * 2]
                    .chunks_exact(2)
                    .filter(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32() != 0.0)
                    .count();
                eprintln!("[enc-bisect] ②multi 根=[memory]: 行0 非零={nz}/5120 |max|={mx:.4}");
            }
        }
        let roots = d2.encode_kv(&memory, &pos_t, &kvs, &rope, &ctx);
        {
            let face = self.session.face_mut();
            let refs: Vec<&TensorOps> = roots.iter().collect();
            owl_models::interpreters::eval_ops_multi_scoped_env(&refs, face, self.env).await?;
        }
        if probe {
            let (k0, _) = &leaves[0];
            let kn: usize = k0.shape().iter().product();
            let mut kb = vec![0u8; kn * 2];
            {
                let face = self.session.face_mut();
                let b = owl_models::interpreters::eval_ops(k0.step(), face).await?;
                face.dtoh(&b, &mut kb).await?;
            }
            let nz = kb
                .chunks_exact(2)
                .filter(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32() != 0.0)
                .count();
            eprintln!("[enc-bisect] ③encode 后 K 池非零={nz}/{}", kn);
        }
        Ok(())
    }


    /// 首轮 propose(E5-M2b;m=0 特例):pair (h_{pos-1}, emb(anchor)) @
    /// 位置 pos-1 = M1 propose 本尊;seed hidden = prefill 末块 harvest。
    /// 产出 k 草稿块持久(spec_drafts)。
    pub(crate) async fn propose_first(&mut self, anchor: u32, pos: usize) -> Result<()> {
        let depth = self.spec_depth;
        let seed = self
            .tspec()
            .spec_seed_hidden
            .take()
            .ok_or_else(|| owl_models::ModelError::Msg(
                "spec seed 缺失(prefill 末块 harvest 未落;mtp 模式要求)".into(),
            ))?;
        // mtp 链表:覆盖 pos-1(extend 行)+ pos..pos+depth-2(链步)
        let bt = self.mtp_chain_bt(pos + depth - 1)?;
        let (kc, vc) = self.pool.mtp_kv_leaf().expect("mtp 池");
        let page = self.pool.page;
        let f32b = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|f| f.to_le_bytes()).collect() };
        let bt_t = TensorOps::from_host(
            Dtype::F32,
            vec![1, bt.len()],
            &f32b(&bt.iter().map(|&b| b as f32).collect::<Vec<_>>()),
        );
        // 步 i:位置 pos-1+i,kv_len = pos+i(含自身)
        let mut kvs = Vec::new();
        let mut pos_ts = Vec::new();
        for i in 0..depth {
            let p = pos - 1 + i;
            kvs.push(KvBuffers {
                k_cache: kc.clone(),
                v_cache: vc.clone(),
                slots: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[(bt[p / page] * page as u32 + (p % page) as u32) as f32])),
                kv_lens: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[(p + 1) as f32])),
                block_tables: bt_t.clone(),
            });
            pos_ts.push(TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[p as f32])));
        }
        let anchor_t = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[anchor as f32]));
        let seed_view =
            TensorOps::of_block(seed.id, self.pool.dims.dtype, vec![1, self.pool.dims.hidden]);
        let drafts = {
            let env = self.env;
            let vocab = self.model.vocab_size();
            let rope = self.rope.clone();
            match self.drafter.as_ref().expect("drafter") {
                crate::running::Drafter::Mtp(mtp) => mtp.propose(
                    &anchor_t, &seed_view, depth, &self.model.embed, &rope, &kvs, &pos_ts,
                    vocab, env,
                ),
                crate::running::Drafter::DFlash2(_) => unreachable!("DFlash2 不走 propose_first(seed 链)"),
            }
        };
        let (b, arena) = {
            let face = self.session.face_mut();
            owl_models::interpreters::eval_ops_scoped_env(drafts.step(), face, self.env).await?
        };
        {
            let face = self.session.face_mut();
            face.free(&arena).await?;
            face.free(&[seed.id]).await?; // seed 块已消费,归还
            let mut buf = vec![0u8; 4 * depth];
            face.dtoh(&b, &mut buf).await?;
            face.free(&[b.id]).await?;
            self.tspec().spec_drafts_host = Some(
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
    ) -> Result<owl_iface::contract::Bytes> {
        let depth = self.spec_depth;
        // mtp 链表覆盖:extend F..F+m + 链 F'..F'+depth-2(F' = F+m+1)
        let fp = pos + m + 1;
        // 链步最远位 = fp+depth-2 → 链表需覆盖 token 数 fp+depth-1
        let bt = self.mtp_chain_bt(fp + depth - 1)?;
        let (kc, vc) = self.pool.mtp_kv_leaf().expect("mtp 池");
        let page = self.pool.page;
        let f32b = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|f| f.to_le_bytes()).collect() };
        let bt_t = TensorOps::from_host(
            Dtype::F32,
            vec![1, bt.len()],
            &f32b(&bt.iter().map(|&b| b as f32).collect::<Vec<_>>()),
        );
        let slot_at = |p: usize| -> f32 { (bt[p / page] * page as u32 + (p % page) as u32) as f32 };
        // extend 批行建表([M+1])
        let slots_x: Vec<f32> = (0..=m).map(|i| slot_at(pos + i)).collect();
        let lens_x: Vec<f32> = (0..=m).map(|i| (pos + i + 1) as f32).collect();
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
            TensorOps::of_block(hidden.id, self.pool.dims.dtype, vec![m + 1, self.pool.dims.hidden]);
        let extend_pos = TensorOps::from_host(Dtype::F32, vec![m + 1], &f32b(&pos_x));
        let extend_slots = TensorOps::from_host(Dtype::F32, vec![m + 1], &f32b(&slots_x));
        let extend_lens = TensorOps::from_host(Dtype::F32, vec![m + 1], &f32b(&lens_x));
        // 链步建表(k-1 步 @ F'..F'+depth-2)
        let mut chain = Vec::new();
        for j in 1..depth {
            let p = fp + j - 1;
            let kv = KvBuffers {
                k_cache: kc.clone(),
                v_cache: vc.clone(),
                slots: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[slot_at(p)])),
                kv_lens: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[(p + 1) as f32])),
                block_tables: bt_t.clone(),
            };
            let pt = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[p as f32]));
            chain.push((pt, kv));
        }
        let chain_refs: Vec<(&TensorOps, &KvBuffers)> =
            chain.iter().map(|(p, k)| (p, k)).collect();
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
            let env = self.env;
            let vocab = self.model.vocab_size();
            let rope = self.rope.clone();
            match self.drafter.as_ref().expect("drafter") {
                crate::running::Drafter::Mtp(mtp) => mtp.propose_ext(
                    &tokens_t, &hiddens, &extend_pos, &extend_kv, &extend_slots, &extend_lens,
                    &cu_t, &chain_refs, &self.model.embed, &rope, vocab, env,
                ),
                crate::running::Drafter::DFlash2(_) => unreachable!("DFlash2 走 dflash_propose"),
            }
        };
        mrec("spec.propose.build", ts_build.elapsed());
        let ts_eval = std::time::Instant::now();
        let (b, arena) = {
            let face = self.session.face_mut();
            owl_models::interpreters::eval_ops_scoped_env(drafts.step(), face, self.env).await?
        };
        mrec("spec.propose.eval", ts_eval.elapsed());
        {
            let face = self.session.face_mut();
            face.free(&arena).await?;
        }
        Ok(b)
    }


    /// prefill extend(E5-M5;sglang _append_target_hidden_to_draft_kv
    /// 同语义):chunk 的 (h_p, emb(t_{p+1})) 对 @ 位置 p 写 mtp KV,
    /// 草稿链看得见提示词。行数 = t_rows(= T-1;末行配对归下一 chunk /
    /// propose_first)。eager from_host 表(每 chunk 一次;非图)。
    pub(crate) async fn prefill_extend(
        &mut self,
        hid: &Bytes,
        t_rows: usize,
        ids: &[u32],
        base: usize,
    ) -> Result<()> {
        let t_all = std::time::Instant::now();
        // DFlash2 早退(无链式 extend;草稿 KV 走 prefill_dflash_encode)
        // —— 门必须先于 blocks_mtp 预留(mtp 账房非 mtp 模式 = 0 块)
        if !matches!(self.drafter.as_ref(), Some(crate::running::Drafter::Mtp(_))) {
            return Ok(());
        }
        let d = self.pool.dims;
        let page = self.pool.page;
        let bt = {
            let sid = self.active.as_ref().expect("active 已保证").session_id;
            let s = self.sessions.get_mut(sid).expect("账在");
            self.blocks_mtp.ensure_for_len(&mut s.mtp_block_table, base + t_rows)?;
            s.mtp_block_table.clone()
        };
        let (kc, vc) = self.pool.mtp_kv_leaf().expect("mtp 池");
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
        let mtp = match self.drafter.as_ref() {
            Some(crate::running::Drafter::Mtp(m)) => m,
            _ => return Ok(()), // DFlash2 无链式 extend(草稿 KV 走 dflash_encode)
        };
        let root = mtp.extend_block(
            &tok_t, &hid_view, &pos_t, &kv, &slots_t, &lens_t, &cu_t,
            &self.model.embed, &self.rope, self.env,
        );
        let (_b, arena) = {
            let face = self.session.face_mut();
            owl_models::interpreters::eval_ops_scoped_env(root.step(), face, self.env).await?
        };
        mrec("spec.extend.eval", ts_eval.elapsed());
        mrec("spec.extend", t_all.elapsed());
        {
            let face = self.session.face_mut();
            face.free(&arena).await?;
        }
        Ok(())
    }


    /// propose 图步(E5-M5;桶 m):装填 extend/链表 + bt_mtp 持久槽
    /// 重写 → 单 graph_launch → drafts dtoh(host 账)。
    pub(crate) async fn propose_graph_step(
        &mut self,
        m: usize,
        toks: &[u32],
        pos: usize,
    ) -> Result<Vec<u32>> {
        let depth = self.spec_depth;
        let page = self.pool.page;
        let bt = {
            let sid = self.active.as_ref().expect("active 已保证").session_id;
            let s = self.sessions.get_mut(sid).expect("账在");
            self.blocks_mtp.ensure_for_len(&mut s.mtp_block_table, pos + m + depth)?;
            s.mtp_block_table.clone()
        };
        {
            let ts_w = std::time::Instant::now();
            let face = self.session.face_mut();
            self.pool.write_bt_mtp(face, &bt).await?;
            mrec("spec.propose.btwt", ts_w.elapsed());
        }
        let fp = pos + m + 1;
        let slot_at =
            |p: usize| (bt[p / page] * page as u32 + (p % page) as u32) as f32;
        let tok_f: Vec<f32> = toks.iter().map(|&v| v as f32).collect();
        let pos_f: Vec<f32> = (pos..pos + m + 1).map(|p| p as f32).collect();
        let slots_f: Vec<f32> = (pos..pos + m + 1).map(slot_at).collect();
        let lens_f: Vec<f32> = ((pos + 1)..=(pos + m + 1)).map(|p| p as f32).collect();
        let mut ins: Vec<(&str, &[f32])> = vec![
            ("tok_ext", tok_f.as_slice()),
            ("pos_ext", pos_f.as_slice()),
            ("slots_ext", slots_f.as_slice()),
            ("lens_ext", lens_f.as_slice()),
        ];
        let mut chain_bufs: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = Vec::new();
        for j in 1..depth {
            let p = fp + j - 1;
            chain_bufs.push((
                vec![p as f32],
                vec![slot_at(p)],
                vec![(p + 1) as f32],
            ));
        }
        let names: Vec<String> =
            (0..chain_bufs.len()).flat_map(|j| [format!("c{j}_pos"), format!("c{j}_slots"), format!("c{j}_lens")]).collect();
        for (j, (a, b, c)) in chain_bufs.iter().enumerate() {
            ins.push((names[j * 3].as_str(), a.as_slice()));
            ins.push((names[j * 3 + 1].as_str(), b.as_slice()));
            ins.push((names[j * 3 + 2].as_str(), c.as_slice()));
        }
        let pg = &mut self.propose_graphs[m];
        let t1 = std::time::Instant::now();
        pg.step(&ins).await?;
        let t_step = t1.elapsed();
        let d = pg.read_output_f32("drafts").await?;
        mrec("spec.propose.step", t_step);
        mrec("spec.propose.read", t1.elapsed() - t_step);
        Ok(d.iter().map(|&v| v as u32).collect())
    }


    /// mtp 链表 ensure(E5-M2b;propose eager 路径用 from_host 表,非持久
    /// bt 块 —— 图不烘焙 mtp 链)。返回 host 块链副本。
    pub(crate) fn mtp_chain_bt(&mut self, cover: usize) -> Result<Vec<u32>> {
        let sid = self.active.as_ref().expect("active 已保证").session_id;
        let s = self.sessions.get_mut(sid).expect("账在");
        self.blocks_mtp.ensure_for_len(&mut s.mtp_block_table, cover)?;
        Ok(s.mtp_block_table.clone())
    }


    /// verify 块前向(E5-M2b):prefill 通道 T = 块长(base = F,FI 表四件套
    /// 同款)。双树求值:树一 = last_hidden(根块存活返回 —— extend 同迭代
    /// 消费);树二 = lm_head → 逐行 argmax → 行栈(引用树一根)。返回
    /// (各行 argmax = 各位置贪心提名,hidden 块 [t, hidden];调用方在
    /// extend 后显式 free)。
    pub(crate) async fn verify_forward(
        &mut self,
        block: &[u32],
        base: usize,
        gdn_slot: usize,
    ) -> Result<(Vec<u32>, owl_iface::contract::Bytes)> {
        let t = block.len();
        let (bt_chain, _) = {
            let act = self.active.as_ref().expect("active 已保证");
            let s = self.sessions.get(act.session_id).expect("账在");
            (s.block_table.clone(), s.gdn_slot)
        };
        let kv_caches = self.pool.kv_caches();
        let kvs_step: Vec<KvBuffers> = {
            let page = self.pool.page;
            let phys: Vec<f32> = (base..base + t)
                .map(|p| {
                    let b = bt_chain[p / page];
                    (b * page as u32 + (p % page) as u32) as f32
                })
                .collect();
            let f32b = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|f| f.to_le_bytes()).collect() };
            kv_caches
                .iter()
                .map(|(k, v)| KvBuffers {
                    k_cache: k.clone(),
                    v_cache: v.clone(),
                    slots: TensorOps::from_host(Dtype::F32, vec![t], &f32b(&phys)),
                    kv_lens: TensorOps::from_host(
                        Dtype::F32,
                        vec![t],
                        &f32b(&((base + 1..=base + t).map(|v| v as f32).collect::<Vec<_>>())),
                    ),
                    block_tables: self.pool.bt_leaf_flat(),
                })
                .collect()
        };
        let gdns_step: Vec<GdnBuffers> = {
            let gdn_caches = self.pool.gdn_caches();
            gdn_caches
                .iter()
                .map(|g| GdnBuffers {
                    conv_q: g.conv_q.clone(),
                    conv_k: g.conv_k.clone(),
                    conv_v: g.conv_v.clone(),
                    rec: g.rec.clone(),
                    slots: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[gdn_slot as f32])),
                })
                .collect()
        };
        let ids_t = TensorOps::from_host(
            Dtype::F32,
            vec![t],
            &block.iter().flat_map(|v| (*v as f32).to_le_bytes()).collect::<Vec<u8>>(),
        );
        let pos_t = TensorOps::from_host(
            Dtype::F32,
            vec![t],
            &f32b(&(base..base + t).map(|v| v as f32).collect::<Vec<_>>()),
        );
        let f32b = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|f| f.to_le_bytes()).collect() };
        let slots_t = TensorOps::from_host(
            Dtype::F32,
            vec![t],
            &f32b(&{
                let page = self.pool.page;
                (base..base + t)
                    .map(|p| {
                        let b = bt_chain[p / page];
                        (b * page as u32 + (p % page) as u32) as f32
                    })
                    .collect::<Vec<_>>()
            }),
        );
        let lens_t = TensorOps::from_host(
            Dtype::F32,
            vec![t],
            &f32b(&((base + 1..=base + t).map(|v| v as f32).collect::<Vec<_>>())),
        );
        let gdn_slot_t = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[gdn_slot as f32]));

        // FI 表四件套(与 prefill_chunk 同款;ctx_total = base + t)
        let i32le = |v: &[i32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let fi_tensors;
        let fi = if self.pool.fi_face {
            let page = self.pool.page;
            let ctx_total = base + t;
            let nb = bt_chain.len();
            let k_fi_leaf = self.pool.kv_fi_leaves();
            let face = self.session.face_mut();
            let q_cu_b = owl_models::interpreters::eval_ops(
                TensorOps::from_host(Dtype::U32, vec![2], &i32le(&[0, t as i32])).step(), face)
                .await?;
            let idx_b = owl_models::interpreters::eval_ops(
                TensorOps::from_host(Dtype::U32, vec![nb],
                    &i32le(&bt_chain.iter().map(|&b| b as i32).collect::<Vec<_>>())).step(), face)
                .await?;
            let ind_b = owl_models::interpreters::eval_ops(
                TensorOps::from_host(Dtype::U32, vec![2], &i32le(&[0, nb as i32])).step(), face)
                .await?;
            let ll_b = owl_models::interpreters::eval_ops(
                TensorOps::from_host(Dtype::U32, vec![1],
                    &i32le(&[(ctx_total - nb * page + page) as i32])).step(), face)
                .await?;
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

        let mut ctx = owl_models::module::ForwardCtx::model_prefill(
            t, &pos_t, &kvs_step, &self.rope, &gdns_step, &slots_t, &lens_t, &gdn_slot_t, base, fi,
        );
        ctx.env = self.env;
        let face = self.session.face_mut();
        let vocab = self.model.vocab_size();
        // 树一:hidden 根([t, hidden];E5-M2b:extend 同迭代消费 → 根块
        // 存活返回,中间块即焚)。原单树版的 argmax 根与 _arena 均 drop 未
        // free(每轮泄漏 verify 全部中间块 —— M2a 存量雷,本轮修复)。
        let hidden = self.model.last_hidden(&ids_t, &ctx);
        let hd = hidden.shape()[1];
        let (hb, arena1) =
            owl_models::interpreters::eval_ops_scoped_env(hidden.step(), face, self.env).await?;
        // 树二:lm_head + 逐行 argmax(引用树一根块,不重建整模前向)
        let hview = TensorOps::of_block(hb.id, self.pool.dims.dtype, vec![t, hd]);
        let logits = self.model.embed.lm_head_matmul(&hview);
        let rows: Vec<TensorOps> =
            (0..t).map(|r| owl_models::ops::argmax_f32idx(&logits, vocab, r * vocab)).collect();
        let refs: Vec<&TensorOps> = rows.iter().collect();
        let mut root = TensorOps::call(owl_models::ops::SemanticKernel::Concat);
        for i in 0..8 {
            root = root.arg(refs.get(i).copied().unwrap_or(refs[0]));
        }
        let root = root.arg_usize(t).arg_usize(1).arg_usize(1).with_shape(Dtype::F32, vec![t, 1]);
        let (b, arena2) =
            owl_models::interpreters::eval_ops_scoped_env(root.step(), face, self.env).await?;
        let mut buf = vec![0u8; 4 * t];
        face.dtoh(&b, &mut buf).await?;
        // 显式归还:argmax 根 + 两树中间块(hidden 根 hb 存活,随返回值)
        face.free(&[b.id]).await?;
        face.free(&arena2).await?;
        face.free(&arena1).await?;
        let ids: Vec<u32> = buf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u32)
            .collect();
        Ok((ids, hb))
    }

    // ====================================================================
    // 取证探针(P1.5:仪表从 execute_decode 抽出;全部 probes 门控,
    // 正常路径零开销。TS 探针在声明层,见 model.rs last_hidden)
    // ====================================================================


}

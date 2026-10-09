//! spec 轮机器(execute_spec_round)+ 块表写 + GDN 快照
use super::*;

impl<D: DeviceClient> crate::running::RunningEngine<D> {
    /// 投机轮执行(E5-M2b;施工方案 §9.2):快照 → 草稿到位 → verify
    /// (depth+1 行 prefill 通道,返回 ids + hidden 块)→ greedy 接受 →
    /// (restore+重放)→ **extend+propose(同迭代,零跨轮 hidden 持久)**
    /// → 提交发射。发射 m+1 token(接受 drafts + bonus),事件入
    /// pending_events 队列(pump 逐个出)。
    pub(crate) async fn execute_spec_round(
        &mut self,
        token: u32,
        pos: usize,
        kv_slots: [u32; 9],
        gdn_slot: usize,
        grew: bool,
    ) -> Result<TurnEvent> {
        let sid = self.active.as_ref().expect("active 已保证").session_id;
        if grew {
            let ts_bt = std::time::Instant::now();
            self.write_bt(sid).await?;
            mrec("spec.bt", ts_bt.elapsed());
        }
        let prof = self.probes.step_profile;
        let t0 = prof.then(std::time::Instant::now);
        let t_round = std::time::Instant::now();
        let mut t_snap = std::time::Duration::ZERO;
        let mut t_verify = std::time::Duration::ZERO;
        let mut t_rollback = std::time::Duration::ZERO;
        let mut t_propose = std::time::Duration::ZERO;
        let depth = self.spec_depth;

        // ① 快照(轮首;恢复后仍有效 → 只在失效时拍)
        if !self.tspec().spec_snap_valid {
            let ts = std::time::Instant::now();
            let face = self.session.face_mut();
            self.pool.capture_spec_snap(face, gdn_slot).await?;
            self.tspec().spec_snap_valid = true;
            t_snap = ts.elapsed();
        }

        // ② 草稿到位(host 账;轮末 propose 图/eager dtoh 产出。首轮 =
        // prefill seed propose(eager);Dumb = anchor 重复)
        let drafts: Vec<u32> = if let Some(mut d) = self.tspec().spec_drafts_host.take() {
            // depth 联动截断(2026-10-09 depth=3 案):⑦ propose 恒产 7
            // 草稿,depth<7 时下轮 block=1+7 行装不进 T=depth+1 捕获图
            // (ids 8≠4 装填拒,pump 退役)。与轮首直调分支的
            // d.truncate(depth) 同款,收口在消费点单处。
            d.truncate(depth);
            d
        } else if let Some(crate::running::Drafter::DFlash2(_)) = self.drafter.as_ref() {
            let ts = std::time::Instant::now();
            let mut d = self.dflash_propose(token, pos).await?;
            d.truncate(depth); // depth ≤ 7(草稿恒 7 产;T 图定形 depth+1)
            t_propose += ts.elapsed();
            d
        } else if self.drafter.is_some() {
            let ts = std::time::Instant::now();
            self.propose_first(token, pos).await?;
            t_propose += ts.elapsed();
            self.tspec().spec_drafts_host.take().expect("propose_first 产出")
        } else {
            vec![token; depth] // Dumb:哑草稿 = anchor 重复
        };

        // ③ verify 块 [anchor, d1..dk] @ pos..pos+depth:图态(一次 replay
        // + fold 记录)或 eager fallback(旧 verify_forward)。fold 记录
        // = 每层 8 件(conv 三 raw + 递推五件)。
        let block: Vec<u32> = std::iter::once(token).chain(drafts.iter().copied()).collect();
        let (ids, hidden, records, hidden_owned) = if self.verify_graph.is_some() {
            let ts = std::time::Instant::now();
            let vg = self.verify_graph.as_mut().expect("verify_graph");
            let page = self.pool.page;
            let bt_chain = {
                let s = self.sessions.get(sid).expect("账在");
                s.block_table.clone()
            };
            let ids_f: Vec<f32> = block.iter().map(|&v| v as f32).collect();
            let pos_f: Vec<f32> =
                (pos..pos + block.len()).map(|p| p as f32).collect();
            if bt_chain.len() <= (pos + block.len() - 1) / page {
                eprintln!("[spec-dbg] pos={pos} block={} bt_len={} page={page}", block.len(), bt_len = bt_chain.len());
            }
            let slots_f: Vec<f32> = (pos..pos + block.len())
                .map(|p| (bt_chain[p / page] * page as u32 + (p % page) as u32) as f32)
                .collect();
            let lens_f: Vec<f32> = ((pos + 1)..=(pos + block.len())).map(|p| p as f32).collect();
            let t_step_v = std::time::Instant::now();
            vg.step(&[
                ("ids", ids_f.as_slice()),
                ("pos", pos_f.as_slice()),
                ("kv_slots", slots_f.as_slice()),
                ("kv_lens", lens_f.as_slice()),
                ("gdn_slot", &[gdn_slot as f32]),
            ])
            .await?;
            let t_read_v = std::time::Instant::now();
            let tok_f = vg.read_output_f32("tok").await?;
            // E5 性能:verify 分相(step=上传+launch;read=同步+回读;
            // pure=同步后裸 replay+read = 图独占 GPU 时间)
            if self.probes.step_profile {
                let t_v1 = std::time::Instant::now();
                vg.replay().await?;
                let _ = vg.read_output_f32("tok").await?;
                eprintln!(
                    "[verify-prof] step={:?} read={:?} pure={:?}",
                    t_read_v - t_step_v,
                    t_v1 - t_read_v,
                    t_v1.elapsed()
                );
            }
            mrec("spec.verify.step", t_read_v - t_step_v);
            mrec("spec.verify.read", t_read_v.elapsed());
            let ids: Vec<u32> = tok_f.iter().map(|&v| v as u32).collect();
            let hidden = vg.output_block("hid").expect("hid 输出槽");
            let n_rec = self.pool.gdn_count() * 8;
            let records: Vec<Bytes> = (0..n_rec)
                .map(|i| vg.output_block(&format!("r{i}")).expect("fold 记录槽"))
                .collect();
            t_verify = ts.elapsed();
            (ids, hidden, records, false) // 图输出槽:持久,不 free
        } else {
            let ts = std::time::Instant::now();
            let (ids, hidden) = self.verify_forward(&block, pos, gdn_slot).await?;
            t_verify = ts.elapsed();
            (ids, hidden, Vec::new(), true)
        };
        if prof {
            eprintln!("[spec-prof] pos={pos} verify={:?}", t0.unwrap().elapsed());
        }

        // ④ greedy 接受:行 i argmax vs draft i;i < depth
        let m = (0..depth).take_while(|&i| drafts[i] == ids[i]).count();
        let bonus = ids[m];
        // B4 自适应降级态机(§6.24;DFlash2 专属 —— 同步机制走 dflash encode)
        // 2026-10-12 深夜重放 + 滚动 AL 窗口:连击 streak 会被"偶发 m=1"
        // 破解(随机域 AL≈1.1 时 9.3 t/s 案),窗口均值 < 盈亏线即降级
        let mut degraded_now;
        if self.dflash_tap_count > 0 && self.probes.degrade_after > 0 {
            let (degrade_after, probe_every0) = (self.probes.degrade_after, self.probes.probe_every);
            let spec = self.tspec();
            spec.recent_rounds += 1;
            spec.recent_tokens += (m + 1) as u32;
            let (streak, degraded0, probe_every) = spec_degrade_transition(
                spec.spec_zero_streak,
                spec.spec_degraded,
                spec.spec_probe_every,
                m,
                degrade_after,
                probe_every0,
            );
            let window_starved = if spec.recent_rounds >= 16 {
                let avg = spec.recent_tokens as f32 / spec.recent_rounds as f32;
                spec.recent_rounds = 0;
                spec.recent_tokens = 0;
                avg < 1.2
            } else {
                false
            };
            let degraded = degraded0 || window_starved;
            if degraded && !spec.spec_degraded {
                eprintln!(
                    "[spec-degrade] m=0×{streak} win_starved={window_starved} → 降级裸 decode(probe 每 {probe_every} tok)"
                );
                mcnt("spec.degrade", 1);
                spec.spec_drafts_host = None;
            } else if !degraded && spec.spec_degraded {
                eprintln!("[spec-degrade] 探测命中 m={m} → 恢复 spec");
                mcnt("spec.undegrad", 1);
            }
            spec.spec_zero_streak = streak;
            spec.spec_degraded = degraded;
            spec.spec_probe_every = probe_every;
            // 滚动 AL gauge(窗口均值 ×100;/debug/metrics 一屏可见盈亏)
            mcnt("spec.al.win", u64::from(spec.recent_tokens * 100 / spec.recent_rounds.max(1)));
            degraded_now = degraded;
        } else {
            degraded_now = self.tspec_ref().spec_degraded;
        }
        // P0 修(2026-10-12):降级态在 emit 循环前定格 —— 预算/EOS 会在
        // 循环内 complete()(active 被取走),循环后触 tspec() = panic
        // mcnt("spec.pos.last", ...) 在 metrics 段(pos+m+1 gauge)
        // 对齐探针(E5-DF3 AL=0 排查):drafts vs 目标验证行逐位对照 ——
        // 附近命中(drafts[i]==ids[i±1]) = 位移对齐 bug;全散 = 分布质量
        if self.probes.dflash_probe {
            let hits: Vec<String> = (0..depth)
                .map(|i| {
                    let near = if i > 0 && drafts[i] == ids[i - 1] {
                        "←(i-1)"
                    } else if drafts[i] == ids[i] {
                        "HIT"
                    } else if i + 1 < ids.len() && drafts[i] == ids[i + 1] {
                        "→(i+1)"
                    } else {
                        ""
                    };
                    format!("d{}={}{}", i, drafts[i], near)
                })
                .collect();
            eprintln!("[dflash-probe] pos={pos} ids={:?} {}", &ids[..depth.min(ids.len())], hits.join(" "));
        }

        // ⑤ 部分接受 → GDN 回滚 + 已接受前缀重放(v1 窗口重处理,施工
        // 方案 §四.4);快照无条件失效(下轮轮首重拍)。
        // 轮不变量:restore 只回到 state@F-1,而下一轮需要 state@fed'-1
        // = state@F+m —— [anchor, d1..dm] 的贡献在 verify 中已推进、被
        // restore 回滚,必须重放补回(已接受行 KV 覆写幂等 —— 同快照同
        // 输入确定性同值)。m=0 也要重放 anchor 一行。
        // ⑤ extend + propose(同迭代消费 verify hidden;产出下轮草稿
        // host 账)。图态 = 桶 m 一发;eager fallback = propose_round。
        // (先于 fold:fold 图发射后的设备排队会与 eager alloc/launch
        // 交织放大 —— M5 实测 eager propose 52.8ms ← 8.5ms,换序复原)
        let toks: Vec<u32> = {
            let mut t = drafts[..m].to_vec();
            t.push(bonus);
            t
        };
        // DFlash2:propose 统一在 ⑦(fed 提交后)——此处跳过免双跑
        let d2_arc = match self.drafter.as_ref() {
            Some(crate::running::Drafter::DFlash2(_)) => None,
            _ => Some(()),
        };
        if d2_arc.is_none() {
            // skip(⑦ 统一 encode + propose)
        } else if !self.propose_graphs.is_empty() && !self.probes.propose_eager {
            let ts = std::time::Instant::now();
            let d = self.propose_graph_step(m, &toks, pos).await?;
            t_propose += ts.elapsed();
            self.tspec().spec_drafts_host = Some(d);
        } else if self.drafter.is_some() {
            let ts = std::time::Instant::now();
            let nd = self.propose_round(&hidden, m, &toks, pos).await?;
            t_propose += ts.elapsed();
            // 块 → host(轮首消费;块即焚)
            let mut buf = vec![0u8; 4 * depth];
            {
                let face = self.session.face_mut();
                face.dtoh(&nd, &mut buf).await?;
                face.free(&[nd.id]).await?;
            }
            self.tspec().spec_drafts_host = Some(
                buf.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u32)
                    .collect(),
            );
        }
        if hidden_owned {
            // eager 路的 hidden = 竞技场根块,收割后归池;图态 = 图输出槽
            // (持久复用,free = 炸图,M4 立案注记)
            let face = self.session.face_mut();
            face.free(&[hidden.id]).await?;
        }
        // ⑥ fold(尾部:图发射异步,restore 单命令;下一轮 verify 流序
        // 排其后,GPU ~1ms 藏于前端)
        if m < depth {
            let ts = std::time::Instant::now();
            if !self.fold_graphs.is_empty() {
                // 图态 fold(E5-M5):快照 buf 就地重放(state@F-1 →
                // state@F+m;slot 恒 0)→ restore 回写会话格。单 graph_launch。
                let fg = &mut self.fold_graphs[m];
                fg.step(&[("slots", &[0.0f32])]).await?;
                {
                    let face = self.session.face_mut();
                    self.pool.restore_spec_snap(face, gdn_slot).await?;
                }
                t_rollback = ts.elapsed();
            } else {
            {
                let face = self.session.face_mut();
                self.pool.restore_spec_snap(face, gdn_slot).await?;
            }
            if records.len() == self.pool.gdn_count() * 8 {
                // fold(E5-M4):快照态 + verify 记录前缀 → state@F+m,
                // **零整模前向**(96 副作用根一次 multi-root eval)
                let m1 = m + 1;
                let d = self.pool.dims;
                let key_dim = d.nk * d.hk;
                let value_dim = d.nv * d.hv;
                let views: Vec<TensorOps> = records
                    .iter()
                    .enumerate()
                    .map(|(i, b)| {
                        let shape = match i % 8 {
                            0 | 1 => vec![m1, key_dim],
                            2 => vec![m1, value_dim],
                            3 | 4 => vec![m1, d.nk, d.hk],
                            5 => vec![m1, d.nv, d.hv],
                            _ => vec![m1, d.nv],
                        };
                        TensorOps::of_block(b.id, d.dtype, shape)
                    })
                    .collect();
                let slots_t = TensorOps::from_host(
                    Dtype::F32,
                    vec![1],
                    &[gdn_slot as f32]
                        .iter()
                        .flat_map(|f| f.to_le_bytes())
                        .collect::<Vec<u8>>(),
                );
                let gdns: Vec<owl_models::layers::gdn::GdnBuffers> = self
                    .pool
                    .gdn_caches()
                    .into_iter()
                    .map(|l| owl_models::layers::gdn::GdnBuffers {
                        conv_q: l.conv_q,
                        conv_k: l.conv_k,
                        conv_v: l.conv_v,
                        rec: l.rec,
                        slots: slots_t.clone(),
                    })
                    .collect();
                let cu_t = TensorOps::from_host(
                    Dtype::F32,
                    vec![2],
                    &[0.0f32, m1 as f32]
                        .iter()
                        .flat_map(|f| f.to_le_bytes())
                        .collect::<Vec<u8>>(),
                );
                let roots = owl_models::spec::fold_tree(
                    &self.model,
                    &views,
                    &gdns,
                    &slots_t,
                    &cu_t,
                    gdn_slot,
                    m,
                    self.env.gdn.scalar,
                )?;
                {
                    let face = self.session.face_mut();
                    let refs: Vec<&TensorOps> = roots.iter().collect();
                    owl_models::interpreters::eval_ops_multi_scoped_env(&refs, face, self.env).await?;
                }
                t_rollback = ts.elapsed();
            } else {
                // eager fallback:旧路 —— 前缀重放(整模前向,已接受行
                // KV 覆写幂等;m=0 也重放 anchor 一行)
                let replay: Vec<u32> = std::iter::once(token)
                    .chain(drafts[..m].iter().copied())
                    .collect();
                let (_, replay_hidden) = self.verify_forward(&replay, pos, gdn_slot).await?;
                let face = self.session.face_mut();
                face.free(&[replay_hidden.id]).await?;
                t_rollback = ts.elapsed();
            }
            }
        }
        self.tspec().spec_snap_valid = false;

        // ⑥ 提交:fed = bonus 位(F');发射 = 接受 drafts + bonus(m+1)
        let toks: Vec<u32> = drafts[..m].to_vec();
        let mut toks = toks;
        toks.push(bonus);
        {
            let act = self.active.as_mut().expect("活跃");
            act.fed = pos + m + 1;
        }

        // ⑦ extend + propose(同迭代消费 verify hidden;产出下轮草稿
        // host 账)。DFlash2 = encode + 噪声块 propose;图态 = 桶 m 一发
        // (MTP);eager fallback = propose_round(MTP)。
        let d2_arc7 = match self.drafter.as_ref() {
            Some(crate::running::Drafter::DFlash2(d)) => Some(std::sync::Arc::clone(d)),
            _ => None,
        };
        if let Some(d2) = d2_arc7 {
            let ts = std::time::Instant::now();
            // E5-DF4:图态优先(单图一轮 encode+propose;eager 回退 =
            // OWL_DFLASH_EAGER=1 或图降级)
            if self.dflash_graph.is_some() && !self.probes.dflash_eager {
                let d = self.dflash_graph_round(m, pos, toks[m]).await?;
                t_propose += ts.elapsed();
                self.tspec().spec_drafts_host = Some(d);
            } else {
            let noencode = self.probes.dflash_noencode;
            let taps: Vec<owl_iface::contract::Bytes> = (0..self.dflash_tap_count)
                .map(|i| {
                    self.verify_graph
                        .as_ref()
                        .expect("dflash 模式要求 verify 图(taps 输出)")
                        .output_block(&format!("tap{i}"))
                        .expect("tap 输出槽")
                })
                .collect();
            if !noencode {
                self.dflash_encode(&d2, &taps, m, pos).await?;
            }
            let fp = pos + m + 1;
            let bonus = toks[m];
            let d = self.dflash_propose(bonus, fp).await?;
            if self.probes.dflash_probe && self.spec_stats.rounds < 3 {
                let toks_txt = self.tok.decode(&d);
                eprintln!("[dflash-probe] pos={fp} bonus={bonus} drafts={d:?} txt={toks_txt:?}");
            }
            t_propose += ts.elapsed();
            self.tspec().spec_drafts_host = Some(d);
            }
        } else if !self.propose_graphs.is_empty() && !self.probes.propose_eager {
            let ts = std::time::Instant::now();
            let d = self.propose_graph_step(m, &toks, pos).await?;
            t_propose += ts.elapsed();
            self.tspec().spec_drafts_host = Some(d);
        } else if self.drafter.is_some() {
            let ts = std::time::Instant::now();
            let nd = self.propose_round(&hidden, m, &toks, pos).await?;
            t_propose += ts.elapsed();
            // 块 → host(轮首消费;块即焚)
            let mut buf = vec![0u8; 4 * depth];
            {
                let face = self.session.face_mut();
                face.dtoh(&nd, &mut buf).await?;
                face.free(&[nd.id]).await?;
            }
            self.tspec().spec_drafts_host = Some(
                buf.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u32)
                    .collect(),
            );
        }
        if hidden_owned {
            // eager 路的 hidden = 竞技场根块,收割后归池;图态 = 图输出槽
            // (持久复用,free = 炸图,M4 立案注记)
            let face = self.session.face_mut();
            face.free(&[hidden.id]).await?;
        }

        // ⑧ 逐 token 发射(eos/预算;事件入队,pump 逐个出)
        {
            // 崩坏排查打点(E5-DF4):per-boot per-round 事实值
            // (OWL_FACT_PROBE 门控;案结后默认关 —— 动态 tag 每轮 ×9
            // 无限增殖,metrics tag 爆炸 + 512 上限挤掉真账,2026-10-10)
            if self.probes.fact_probe {
                let bs = self.boot_seq;
                let rn = self.spec_stats.rounds;
                mfact(bs, &format!("r{rn}.pos"), pos as u64);
                mfact(bs, &format!("r{rn}.m"), m as u64);
                mfact(bs, &format!("r{rn}.bonus"), bonus as u64);
                for (i, (&dv, &iv)) in drafts.iter().zip(ids.iter()).take(depth).enumerate() {
                    mfact(bs, &format!("r{rn}.d{i}"), dv as u64);
                    mfact(bs, &format!("r{rn}.i{i}"), iv as u64);
                }
            }
        }
        self.spec_stats.rounds += 1;
        self.spec_stats.sum_m += m;
        if m == depth {
            self.spec_stats.full += 1;
        } else if m == 0 {
            self.spec_stats.zero += 1;
        } else {
            self.spec_stats.partial += 1;
        }
        // E4:m 直方图进 metrics(tag 有界 0..=7;/debug/metrics 直读,
        // 免日志 grep;AL 均值 = spec.accepted/spec.rounds 同源可算)
        mcnt(&format!("spec.m{m}"), 1);
        let ts = std::time::Instant::now();
        let mut evq = std::collections::VecDeque::new();
        for &t in &toks {
            let ev = self.sample_and_emit(t).await?;
            let done = matches!(ev, TurnEvent::Completed { .. });
            evq.push_back(ev);
            if done {
                break; // eos/预算:余段作废
            }
        }
        mrec("spec.emit", ts.elapsed());
        self.pending_events = evq;
        // E5-M5:spec 分相落账(/debug/metrics;与 spec_stats 同源双面)
        let t_all = t_round.elapsed();
        mrec("spec.round", t_all);
        mrec("spec.verify", t_verify);
        if t_snap > std::time::Duration::ZERO {
            mrec("spec.snap", t_snap);
        }
        if t_rollback > std::time::Duration::ZERO {
            let tag = if records.len() == self.pool.gdn_count() * 8 {
                "spec.fold"
            } else {
                "spec.replay"
            };
            mrec(tag, t_rollback);
        }
        if t_propose > std::time::Duration::ZERO {
            mrec("spec.propose", t_propose);
        }
        mcnt("spec.rounds", 1);
        mcnt("spec.tokens", (m + 1) as u64);
        mcnt("spec.accepted", m as u64);
        mcnt("spec.pos.last", (pos + m + 1) as u64);
        if degraded_now {
            mcnt("spec.deg.rounds", 1);
            mrec("spec.round.deg", t_all);
        } else {
            mrec("spec.round.spec", t_all);
        }
        if prof {
            eprintln!(
                "[spec-prof] pos={pos} total={t_all:?} verify={t_verify:?} rollback={t_rollback:?} propose={t_propose:?} snap={t_snap:?} m={m} deg={degraded_now}"
            );
        }
        self.pending_events
            .pop_front()
            .ok_or_else(|| owl_models::ModelError::Msg("spec 轮零事件(空块?)".into()))
    }


    /// 活跃会话块表 → 持久 bt 块(设备;StatePool::write_bt 落地,
    /// 本壳只解析会话块链)
    pub(crate) async fn write_bt(&mut self, sid: u64) -> Result<()> {
        let table = self.sessions.get(sid).expect("账在").block_table.clone();
        let face = self.session.face_mut();
        self.pool.write_bt(face, &table).await
    }


    /// GDN 快照拍摄(E2c;块边界跨越时调用):会话格状态 → 槽位缓冲。
    /// key = **边界覆盖块**(block_table[fed/page - 1],即快照边界最后
    /// 一块的物理 id)—— 块表在 turn-open 预长满,表尾 ≠ 边界块。
    /// 槽位选择/LRU 覆盖写在 StatePool::capture_snap(state.rs)。
    pub(crate) async fn capture_gdn_snap(&mut self, sid: u64, fed: usize) -> Result<()> {
        let (gdn_slot, key) = {
            let s = self.sessions.get(sid).expect("账在");
            let bi = fed / self.pool.page - 1;
            (
                s.gdn_slot,
                *s.block_table
                    .get(bi)
                    .ok_or_else(|| ModelError::Msg("空块链不可拍快照".into()))?,
            )
        };
        let face = self.session.face_mut();
        self.pool.capture_snap(face, gdn_slot, key).await
    }


}

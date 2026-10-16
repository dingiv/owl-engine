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
        _kv_slots: [u32; 9],
        gdn_slot: usize,
        grew: bool,
    ) -> Result<TurnEvent> {
        let sid = self.turns.active.as_ref().expect("active 已保证").session_id;
        if grew {
            let ts_bt = std::time::Instant::now();
            self.write_bt(sid).await?;
            mrec("spec.bt", ts_bt.elapsed());
        }
        let prof = self.probes.step_profile;
        let t0 = prof.then(std::time::Instant::now);
        let t_round = std::time::Instant::now();
        let mut t_snap = std::time::Duration::ZERO;
        let t_verify;
        let mut t_rollback = std::time::Duration::ZERO;
        let mut t_propose = std::time::Duration::ZERO;
        let depth = self.spec.depth;

        // ① 快照(轮首;恢复后仍有效 → 只在失效时拍)
        if !self.tspec().spec_snap_valid {
            let ts = std::time::Instant::now();
            let face = self.graphs.session.face_mut();
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
        } else if let Some(crate::spec_runtime::Drafter::DFlash2(_)) = self.spec.drafter.as_ref() {
            let ts = std::time::Instant::now();
            let mut d = self.dflash_propose(token, pos).await?;
            d.truncate(depth); // depth ≤ 7(草稿恒 7 产;T 图定形 depth+1)
            t_propose += ts.elapsed();
            d
        } else if self.spec.drafter.is_some() {
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
        // verify 行数 = anchor + 草稿(depth+1;与图的 depth1 桶同宽)
        let rows_n = block.len();
        // A11:采样接受开关 + 采样参数 + 惩罚历史(必须在借 verify_graph 前取好)
        let (sampler_on, sampler_cfg) = {
            let act = self.turns.active.as_ref().expect("活跃");
            (act.sampler_on, act.sampler)
        };
        let mut hist: Vec<u32> = {
            let act = self.turns.active.as_ref().expect("活跃");
            let tail = act.prompt_ids.len().saturating_sub(64);
            act.prompt_ids[tail..].iter().chain(act.out.iter()).copied().collect()
        };
        let seed_base = {
            let act = self.turns.active.as_ref().expect("活跃");
            act.seed.unwrap_or_else(|| act.id.wrapping_mul(0x9E3779B97F4A7C15))
        };
        // XQA 页表:在借 self.graphs.verify 之前算好(自 sessions 取真块链)
        let xqa_pt_v = self.xqa_pt_of(sid);
        // FI 断点臂的宿主标量槽(self.graphs 的可变借用前取好,同 xqa_pt_v 纪律)
        let fi_ctx_cell = self.graphs.verify_fi_ctx();
        let (ids, hidden, records, hidden_owned, row_logits) =
            if let Some(vg) = self.graphs.verify.as_mut() {
            let ts = std::time::Instant::now();
            let page = self.pool.page;
            let bt_chain = {
                let s = self.bank.sessions.get(sid).expect("账在");
                s.block_table.clone()
            };
            let ids_f: Vec<f32> = block.iter().map(|&v| v as f32).collect();
            let pos_f: Vec<f32> =
                (pos..pos + block.len()).map(|p| p as f32).collect();
            if bt_chain.len() <= (pos + block.len() - 1) / page {
                owl_shared::log_debug!("spec-dbg", "pos={pos} block={} bt_len={bt_len} page={page}", block.len(), bt_len = bt_chain.len());
            }
            let slots_f: Vec<f32> = (pos..pos + block.len())
                .map(|p| (bt_chain[p / page] * page as u32 + (p % page) as u32) as f32)
                .collect();
            let lens_f: Vec<f32> = ((pos + 1)..=(pos + block.len())).map(|p| p as f32).collect();
            // §三十一:bt8 = 页链 × depth1 平铺(8 伪序列共享页链;v2 臂用);
            // 补零到池 nb(会话链短于池时尾部补无效页 0,kv_lens 限界不读)
            let nb_total = self.pool.nb.max(bt_chain.len());
            let bt8_f: Vec<f32> = (0..block.len())
                .flat_map(|_| {
                    bt_chain.iter().map(|&b| b as f32)
                        .chain(std::iter::repeat_n(0f32, nb_total - bt_chain.len()))
                })
                .collect();
            // XQA 臂:semaphores 每轮清零(KV-split 归约计数;**同流序先于图回放**;
            // 参考实现 = torch sem.zero_();图内不做,故在轮侧发 memset)
            // ⚠️ 必须用 **session face**(池块的账房;verify 图 face 是另一份
            // 账本 → 传别家的块 id = 打到别的块上 → `pump: DeadBlock` 实测踩过)
            // XQA seq_lens = **整请求总 KV 长**(含块内 depth+1 个位置;u32 位型走私)
            let xqa_seq_v = [f32::from_bits((pos + block.len()) as u32)];
            // semaphores 每轮写零(图输入槽;KV-split 归约计数)。
            // `.max(1)` 与声明端同式 —— "无 XQA 面"时槽仍存在(= 1 元素),
            // 避免零长写(挂账 #7)。
            let xqa_sem_v: Vec<f32> = vec![0.0; self.pool.xqa_sem_n().max(1)];
            // **FI 断点臂**的逐轮值(该臂未开 ⇒ 空,不写槽):
            // ① 宿主标量槽 `ctx_total`(= 本轮 kv 总长;plan 的 key)
            // ② 表四件套(q_cu/indices/indptr/last_len;u32 位型过线)
            let page_fi = self.pool.page;
            let ctx_total_fi = pos + block.len();
            let (fi_q_cu_v, fi_idx_v, fi_indptr_v, fi_ll_v): (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) =
                if let Some(cell) = fi_ctx_cell.as_ref() {
                    cell.store(ctx_total_fi, std::sync::atomic::Ordering::Relaxed);
                    let nb_used = ctx_total_fi
                        .div_ceil(page_fi)
                        .min(bt_chain.len())
                        .max(1);
                    let mut idx = vec![0f32; self.pool.nb];
                    for (i, b) in bt_chain.iter().take(nb_used).enumerate() {
                        idx[i] = f32::from_bits(*b);
                    }
                    (
                        vec![0.0, f32::from_bits(block.len() as u32)],
                        idx,
                        vec![0.0, f32::from_bits(nb_used as u32)],
                        vec![f32::from_bits((ctx_total_fi - (nb_used - 1) * page_fi) as u32)],
                    )
                } else {
                    (Vec::new(), Vec::new(), Vec::new(), Vec::new())
                };
            let gdn_slot_v1 = [gdn_slot as f32];
            let mut ins: Vec<(&str, &[f32])> = vec![
                ("ids", ids_f.as_slice()),
                ("pos", pos_f.as_slice()),
                ("kv_slots", slots_f.as_slice()),
                ("kv_lens", lens_f.as_slice()),
                ("gdn_slot", gdn_slot_v1.as_slice()),
                ("bt8", bt8_f.as_slice()),
                ("xqa_seq", xqa_seq_v.as_slice()),
                ("xqa_sem", xqa_sem_v.as_slice()),
                // XQA 页表:会话真链(槽在 verify 图无条件声明;
                // 不写 = 留着恒等链 = 读到别家 KV —— §7.6)
                ("xqa_pt", xqa_pt_v.as_slice()),
            ];
            if !fi_q_cu_v.is_empty() {
                ins.push(("fi_q_cu", fi_q_cu_v.as_slice()));
                ins.push(("fi_indices", fi_idx_v.as_slice()));
                ins.push(("fi_indptr", fi_indptr_v.as_slice()));
                ins.push(("fi_last_len", fi_ll_v.as_slice()));
            }
            let t_step_v = std::time::Instant::now();
            vg.step(&ins).await?;
            let t_read_v = std::time::Instant::now();
            let tok_f = vg.read_output_f32("tok").await?;
            // 诊断(perf 会话排查用;`OWL_PW_DUMP_TOK=1`):verify 行 0..7 的
            // 候选 token —— piecewise 断裂时最直接的判据(常量/零 = 读未写值)
            if owl_shared::env_reader::flag("OWL_PW_DUMP_TOK") {
                eprintln!("[pw-tok] pos={pos} tok={:?}", tok_f);
                let h = vg.read_output_f32("hid").await?;
                let (mut nan, mut mx, mut sum) = (0usize, 0f32, 0f64);
                for v in h.iter() {
                    if v.is_nan() {
                        nan += 1;
                    }
                    mx = mx.max(v.abs());
                    sum += *v as f64;
                }
                eprintln!(
                    "[pw-hid] pos={pos} n={} nan={nan} absmax={mx:.3e} mean={:.3e} first4={:?}",
                    h.len(),
                    sum / h.len().max(1) as f64,
                    &h[..4.min(h.len())]
                );
                let ops = vg.pw_break_operand_stats(0).await?;
                eprintln!(
                    "[pw-op] pos={pos} :: {}",
                    ops.iter()
                        .map(|(i, n, dt, nan, mx)| format!("#{i}[{dt:?}] n={n} nan={nan} max={mx:.2e}"))
                        .collect::<Vec<_>>()
                        .join(" | ")
                );
                let pins = vg.pw_pin_stats().await?;
                eprintln!(
                    "[pw-pin] pos={pos} n={} :: {}",
                    pins.len(),
                    pins.iter()
                        .take(4)
                        .map(|(n, nan, mx)| format!("{n}元素/nan={nan}/max={mx:.2e}"))
                        .collect::<Vec<_>>()
                        .join(" | ")
                );
            }
            let t_tok = std::time::Instant::now();
            // A11:采样档额外回读 per-row logits([depth1, vocab];仅采样档付)
            let row_logits = if sampler_on {
                Some(vg.read_output_f32("logits").await?)
            } else {
                None
            };
            let t_lg = std::time::Instant::now();
            // E5 性能:verify 分相(step=上传+launch;read=同步+回读;
            // pure=同步后裸 replay+read = 图独占 GPU 时间)
            if self.probes.step_profile {
                let t_v1 = std::time::Instant::now();
                vg.replay().await?;
                let _ = vg.read_output_f32("tok").await?;
                owl_shared::log_info!("verify-prof", "step={:?} read={:?} pure={:?}",
                    t_read_v - t_step_v,
                    t_v1 - t_read_v,
                    t_v1.elapsed()
                );
            }
            mrec("spec.verify.step", t_read_v - t_step_v);
            // 2026-10-15:把 `verify.read` 拆开 —— 首轮归因发现它比「设备时间 + 采样」
            // 多出 ~19 ms/轮(且与采样档无关:greedy A/B 只差 0.75 ms),必须定位。
            mrec("spec.verify.tokread", t_tok.elapsed());
            mrec("spec.verify.lgread", t_lg.elapsed());
            mrec("spec.verify.read", t_read_v.elapsed());
            let ids: Vec<u32> = tok_f.iter().map(|&v| v as u32).collect();
            let hidden = vg.output_block("hid").expect("hid 输出槽");
            let n_rec = self.pool.gdn_count() * 8;
            let records: Vec<Bytes> = (0..n_rec)
                .map(|i| vg.output_block(&format!("r{i}")).expect("fold 记录槽"))
                .collect();
            t_verify = ts.elapsed();
            (ids, hidden, records, false, row_logits) // 图输出槽:持久,不 free
        } else {
            let ts = std::time::Instant::now();
            let (ids, hidden, row_logits) =
                self.verify_forward(&block, pos, gdn_slot, sampler_on).await?;
            t_verify = ts.elapsed();
            (ids, hidden, Vec::new(), true, row_logits)
        };
        if prof {
            owl_shared::log_info!("spec-prof", "pos={pos} verify={:?}", t0.unwrap().elapsed());
        }

        // ④ 接受(A11):采样档 = **抽-比式**;贪心档 = 原式(argmax 比对)
        // 原实现只有贪心接受 ⇒ spec 臂完全不采样 ⇒ 请求温度/top_p/seed
        // 在 spec 开时失效 = 事实贪心(27B instruct 复读吸引子;A7/A8 根因)。
        let (m, emitted, bonus) = if sampler_on {
            let Some(lg) = row_logits.as_ref() else {
                return Err(ModelError::Msg(
                    "spec 采样接受需要 per-row logits(verify 路未出 logits 槽)".into(),
                ));
            };
            let vocab = self.model.vocab_size();
            if lg.len() < rows_n * vocab {
                return Err(ModelError::Msg(format!(
                    "spec logits 槽尺寸不符:得 {} < {}×{}",
                    lg.len(),
                    rows_n,
                    vocab
                )));
            }
            let mut s: Vec<u32> = Vec::with_capacity(rows_n);
            let mut scratch: Vec<f32> = Vec::with_capacity(vocab);
            for i in 0..rows_n {
                scratch.clear();
                scratch.extend_from_slice(&lg[i * vocab..(i + 1) * vocab]);
                // RNG:与裸 decode 同构(位置定序 ⇒ 可复现;不用轮内计数,
                // 免得降级态换路后抽点错位)
                let mut rng =
                    seed_base ^ ((pos + i) as u64).wrapping_mul(0xBF58476D1CE4E5B9);
                let tok = crate::sampler::sample(&mut scratch, &sampler_cfg, &mut rng, &hist);
                s.push(tok);
                hist.push(tok); // 已发出的 token 进惩罚集(与逐 token 解码同语义)
            }
            let m = spec_sampled_accept_prefix(&drafts, &s[..depth]);
            mcnt("spec.sample.rounds", 1);
            mcnt("spec.sample.accepted", m as u64);
            (m, s[..=m].to_vec(), s[m])
        } else {
            let m = (0..depth).take_while(|&i| drafts[i] == ids[i]).count();
            (m, ids[..=m].to_vec(), ids[m])
        };
        // B4 自适应降级态机(§6.24;DFlash2 专属 —— 同步机制走 dflash encode)
        // 2026-10-12 深夜重放 + 滚动 AL 窗口:连击 streak 会被"偶发 m=1"
        // 破解(随机域 AL≈1.1 时 9.3 t/s 案),窗口均值 < 盈亏线即降级
        let degraded_now;
        if self.spec.dflash_tap_count > 0 && self.probes.degrade_after > 0 {
            let (degrade_after, probe_every0) = (self.probes.degrade_after, self.probes.probe_every);
            let spec = self.tspec();
            spec.recent_rounds += 1;
            spec.recent_tokens += (m + 1) as u32;
            // 轮时进窗口(时间门 + 绝对吞吐;t_round 在下方 EWMA 处已用)
            let round_ms_now = t_round.elapsed().as_secs_f32() * 1000.0;
            spec.recent_ms += round_ms_now;
            let (streak, degraded0, probe_every) = spec_degrade_transition(
                spec.spec_zero_streak,
                spec.spec_degraded,
                spec.spec_probe_every,
                m,
                degrade_after,
                probe_every0,
            );
            // 动态盈亏线(2026-10-09;2026-10-14 修口径):
            // threshold = 滚动轮成本/裸步成本(夹 [2.0, 6.0])。
            // **三处修正**(旧判据偏软的根因):
            // ① **分母保鲜**:裸步 EWMA 只在降级期采样(非降级期没有裸步
            //    ⇒ 分母是上一次降级期的陈值,跨 ctx 换档后失真)⇒ ctx 涨过
            //    25% 即作废分母,并在重新采样前取**保守下限 2.5**
            //    (旧兜底 1.2 远低于真线,是"AL 1.2~4 亏本区"的来源);
            // ② **时间门**:窗口满 16 轮 **或** 满 600ms 即评估
            //    (超长轮 150-300ms 下等 16 轮 = 2.4-4.8s,反应太慢);
            // ③ 评估时给出**绝对吞吐**读数(tok/s)便于日志对账。
            let _ctx_sampled = spec.bare_cost_ctx;
            // ctx 换档**不作废**分母(2026-10-14 实测修正):裸步成本几乎不随
            // ctx 变(25.3→26.4 ms @610→9010,±5%),而轮时才是 ctx 相关项 ⇒
            // 保鲜的分母在换档后依然有效;作废只会退到保守兜底并**过度降级**
            // (实测 ctx 2116→7870 时 AL 2.0 本可盈利,却被兜底 2.5 误降)。
            // 采样点 `spec.bare_cost_ctx` 作日志/诊断用(_ctx_sampled)。
            let (rc, bc) = (spec.round_cost_ewma_ms, spec.bare_cost_ewma_ms);
            let break_even = if rc > 0.0 && bc > 0.0 { (rc / bc).clamp(2.0, 6.0) } else { 2.5 };
            const WINDOW_ROUNDS: u32 = 16;
            const WINDOW_MS: f32 = 600.0;
            let window_starved =
                if spec.recent_rounds >= WINDOW_ROUNDS || spec.recent_ms >= WINDOW_MS {
                    let avg = spec.recent_tokens as f32 / spec.recent_rounds.max(1) as f32;
                    let win_tps = spec.recent_tokens as f32 * 1000.0 / spec.recent_ms.max(1.0);
                    spec.recent_rounds = 0;
                    spec.recent_tokens = 0;
                    spec.recent_ms = 0.0;
                    let starved = avg < break_even;
                    if starved {
                        owl_shared::log_warn!("spec-degrade", "窗口 AL={avg:.2} < 盈亏线 {break_even:.2}(轮 {rc:.0}ms/裸 {bc:.0}ms;窗口吞吐 {win_tps:.1} tok/s)"
                        );
                    }
                    starved
                } else {
                    false
                };
            let degraded = degraded0 || window_starved;
            if degraded && !spec.spec_degraded {
                owl_shared::log_warn!("spec-degrade", "m=0×{streak} win_starved={window_starved} 盈亏线={break_even:.2} → 降级裸 decode(probe 每 {probe_every} tok)"
                );
                mcnt("spec.degrade", 1);
                spec.spec_drafts_host = None;
            } else if !degraded && spec.spec_degraded {
                owl_shared::log_warn!("spec-degrade", "探测命中 m={m} → 恢复 spec");
                mcnt("spec.undegrad", 1);
            }
            spec.spec_zero_streak = streak;
            spec.spec_degraded = degraded;
            spec.spec_probe_every = probe_every;
            // 滚动轮成本 ewma(动态盈亏线分子;α=0.15;emit 前 —— 轮尾
            // tspec 可能已随 complete 消失,P0 同款)
            {
                let rc = t_round.elapsed().as_secs_f32() * 1000.0;
                spec.round_cost_ewma_ms = if spec.round_cost_ewma_ms > 0.0 {
                    spec.round_cost_ewma_ms * 0.85 + rc * 0.15
                } else {
                    rc
                };
            }
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
            owl_shared::log_debug!("dflash-probe", "pos={pos} ids={:?} {}", &ids[..depth.min(ids.len())], hits.join(" "));
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
        let d2_arc = match self.spec.drafter.as_ref() {
            Some(crate::spec_runtime::Drafter::DFlash2(_)) => None,
            _ => Some(()),
        };
        if d2_arc.is_none() {
            // skip(⑦ 统一 encode + propose)
        } else if !self.graphs.propose.is_empty() && !self.probes.propose_eager {
            let ts = std::time::Instant::now();
            let d = self.propose_graph_step(m, &toks, pos).await?;
            t_propose += ts.elapsed();
            self.tspec().spec_drafts_host = Some(d);
        } else if self.spec.drafter.is_some() {
            let ts = std::time::Instant::now();
            let nd = self.propose_round(&hidden, m, &toks, pos).await?;
            t_propose += ts.elapsed();
            // 块 → host(轮首消费;块即焚)
            let mut buf = vec![0u8; 4 * depth];
            {
                let face = self.graphs.session.face_mut();
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
            let face = self.graphs.session.face_mut();
            face.free(&[hidden.id]).await?;
        }
        // ⑥ fold(尾部:图发射异步,restore 单命令;下一轮 verify 流序
        // 排其后,GPU ~1ms 藏于前端)
        if m < depth {
            let ts = std::time::Instant::now();
            if !self.graphs.fold.is_empty() {
                // 图态 fold(E5-M5):快照 buf 就地重放(state@F-1 →
                // state@F+m;slot 恒 0)→ restore 回写会话格。单 graph_launch。
                let fg = &mut self.graphs.fold[m];
                fg.step(&[("slots", &[0.0f32])]).await?;
                {
                    let face = self.graphs.session.face_mut();
                    self.pool.restore_spec_snap(face, gdn_slot).await?;
                }
                t_rollback = ts.elapsed();
            } else {
            {
                let face = self.graphs.session.face_mut();
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
                    let face = self.graphs.session.face_mut();
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
                let (_, replay_hidden, _) =
                    self.verify_forward(&replay, pos, gdn_slot, false).await?;
                let face = self.graphs.session.face_mut();
                face.free(&[replay_hidden.id]).await?;
                t_rollback = ts.elapsed();
            }
            }
        }
        self.tspec().spec_snap_valid = false;

        // ⑥ 提交:fed = bonus 位(F');发射 = 接受 drafts + bonus(m+1)
        // (A11:采样档 emitted = 采样表前 m+1 位;贪心档 = ids[..=m])
        let toks: Vec<u32> = emitted;
        {
            let act = self.turns.active.as_mut().expect("活跃");
            act.fed = pos + m + 1;
        }

        // ⑦ extend + propose(同迭代消费 verify hidden;产出下轮草稿
        // host 账)。DFlash2 = encode + 噪声块 propose;图态 = 桶 m 一发
        // (MTP);eager fallback = propose_round(MTP)。
        let d2_arc7 = match self.spec.drafter.as_ref() {
            Some(crate::spec_runtime::Drafter::DFlash2(d)) => Some(std::sync::Arc::clone(d)),
            _ => None,
        };
        if let Some(d2) = d2_arc7 {
            let ts = std::time::Instant::now();
            // E5-DF4:图态优先(单图一轮 encode+propose;eager 回退 =
            // OWL_DFLASH_EAGER=1 或图降级)
            if self.graphs.dflash.is_some() && !self.probes.dflash_eager {
                let d = self.dflash_graph_round(m, pos, toks[m]).await?;
                t_propose += ts.elapsed();
                self.tspec().spec_drafts_host = Some(d);
            } else {
            let noencode = self.probes.dflash_noencode;
            let taps: Vec<owl_iface::contract::Bytes> = (0..self.spec.dflash_tap_count)
                .map(|i| {
                    self.graphs.verify
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
            if self.probes.dflash_probe && self.spec.stats.rounds < 3 {
                let toks_txt = self.tok.decode(&d);
                owl_shared::log_debug!("dflash-probe", "pos={fp} bonus={bonus} drafts={d:?} txt={toks_txt:?}");
            }
            t_propose += ts.elapsed();
            self.tspec().spec_drafts_host = Some(d);
            }
        } else if !self.graphs.propose.is_empty() && !self.probes.propose_eager {
            let ts = std::time::Instant::now();
            let d = self.propose_graph_step(m, &toks, pos).await?;
            t_propose += ts.elapsed();
            self.tspec().spec_drafts_host = Some(d);
        } else if self.spec.drafter.is_some() {
            let ts = std::time::Instant::now();
            let nd = self.propose_round(&hidden, m, &toks, pos).await?;
            t_propose += ts.elapsed();
            // 块 → host(轮首消费;块即焚)
            let mut buf = vec![0u8; 4 * depth];
            {
                let face = self.graphs.session.face_mut();
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
            let face = self.graphs.session.face_mut();
            face.free(&[hidden.id]).await?;
        }

        // ⑧ 逐 token 发射(eos/预算;事件入队,pump 逐个出)
        {
            // 崩坏排查打点(E5-DF4):per-boot per-round 事实值
            // (OWL_FACT_PROBE 门控;案结后默认关 —— 动态 tag 每轮 ×9
            // 无限增殖,metrics tag 爆炸 + 512 上限挤掉真账,2026-10-10)
            if self.probes.fact_probe {
                let bs = self.boot_seq;
                let rn = self.spec.stats.rounds;
                mfact(bs, &format!("r{rn}.pos"), pos as u64);
                mfact(bs, &format!("r{rn}.m"), m as u64);
                mfact(bs, &format!("r{rn}.bonus"), bonus as u64);
                for (i, (&dv, &iv)) in drafts.iter().zip(ids.iter()).take(depth).enumerate() {
                    mfact(bs, &format!("r{rn}.d{i}"), dv as u64);
                    mfact(bs, &format!("r{rn}.i{i}"), iv as u64);
                }
            }
        }
        self.spec.stats.rounds += 1;
        self.spec.stats.sum_m += m;
        if m == depth {
            self.spec.stats.full += 1;
        } else if m == 0 {
            self.spec.stats.zero += 1;
        } else {
            self.spec.stats.partial += 1;
        }
        // E4:m 直方图进 metrics(tag 有界 0..=7;/debug/metrics 直读,
        // 免日志 grep;AL 均值 = spec.accepted/spec.rounds 同源可算)
        mcnt(&format!("spec.m{m}"), 1);
        let ts = std::time::Instant::now();
        let mut evq = std::collections::VecDeque::new();
        for &t in &toks {
            let ev = self.sample_and_emit(t).await?;
            // 预算尽的末 token 走 Token + pending Completed(见 phases)——
            // 故 active 已尽也算 done,余段作废
            let done = matches!(ev, TurnEvent::Completed { .. }) || self.turns.active.is_none();
            evq.push_back(ev);
            if done {
                break; // eos/预算:余段作废
            }
        }
        mrec("spec.emit", ts.elapsed());
        // ⚠️ **不许覆盖**:预算尽时 `sample_and_emit` 已把终态 Completed 压入
        // 本队列(它同时返回 Token),直接赋值 = 丢掉 Completed → actor 永不
        // 派发终态 → handler 卡在 recv → 客户端一帧不收且不 EOF(2026-10-13
        // 定谳:spec 档 + 跑满预算的请求挂死,**与流式无关**)。
        // 序:本轮 Token… 之后接终态事件。
        let mut merged = evq;
        merged.append(&mut self.turns.pending_events);
        self.turns.pending_events = merged;
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
            owl_shared::log_info!("spec-prof", "pos={pos} total={t_all:?} verify={t_verify:?} rollback={t_rollback:?} propose={t_propose:?} snap={t_snap:?} m={m} deg={degraded_now}"
            );
        }
        self.turns.pending_events
            .pop_front()
            .ok_or_else(|| owl_models::ModelError::Msg("spec 轮零事件(空块?)".into()))
    }


    /// 活跃会话块表 → 持久 bt 块(设备;StatePool::write_bt 落地,
    /// 本壳只解析会话块链)
    /// XQA 页表实时值 = **会话真块链**(`f32::from_bits(block_id)`;
    /// XQA 核按 `const uint32_t*` 读,与 state.rs 的 init 编码同源),
    /// 尾部补零到池 nb(与 bt8 同律:kv_lens 限界不读)。
    ///
    /// 为什么必须有它:图里若只用常量恒等链,则**只有单会话独占池**才正确 ——
    /// agent 面每请求新会话(pi 不发 session 键),块链从空闲链续,XQA 却仍读
    /// 0..nb ⇒ 读到别家 KV(§7.6 定谳:多请求"答上一题")。
    pub(crate) fn xqa_pt_of(&self, sid: u64) -> Vec<f32> {
        let empty: &[u32] = &[];
        let table = self.bank.sessions.get(sid).map(|s| s.block_table.as_slice()).unwrap_or(empty);
        let nb = self.pool.nb.max(table.len());
        let mut v: Vec<f32> = table.iter().map(|&b| f32::from_bits(b)).collect();
        v.resize(nb, 0.0);
        v
    }

    pub(crate) async fn write_bt(&mut self, sid: u64) -> Result<()> {
        let table = self.bank.sessions.get(sid).expect("账在").block_table.clone();
        let face = self.graphs.session.face_mut();
        self.pool.write_bt(face, &table).await
    }


    /// GDN 快照拍摄(E2c;块边界跨越时调用):会话格状态 → 槽位缓冲。
    /// key = **边界覆盖块**(block_table[fed/page - 1],即快照边界最后
    /// 一块的物理 id)—— 块表在 turn-open 预长满,表尾 ≠ 边界块。
    /// 槽位选择/LRU 覆盖写在 StatePool::capture_snap(state.rs)。
    pub(crate) async fn capture_gdn_snap(&mut self, sid: u64, fed: usize) -> Result<()> {
        // snap_max = 0 ⇒ 快照面整体关闭:跳过(该档无语义需求;不得靠内部错误
        // 兜底 —— 那是 panic 换 503,见 agent 专项 A2)
        if self.pool.snap_capacity() == 0 {
            return Ok(());
        }
        let (gdn_slot, key) = {
            let s = self.bank.sessions.get(sid).expect("账在");
            let bi = fed / self.pool.page - 1;
            (
                s.gdn_slot,
                *s.block_table
                    .get(bi)
                    .ok_or_else(|| ModelError::Msg("空块链不可拍快照".into()))?,
            )
        };
        let face = self.graphs.session.face_mut();
        self.pool.capture_snap(face, gdn_slot, key).await
    }


}

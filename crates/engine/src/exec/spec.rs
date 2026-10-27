//! spec 轮机器(execute_spec_round)+ 块表写 + GDN 快照
use super::*;
use owl_models::ModelError;

/// **采样惩罚的历史窗**(tok):只回看最近这么多已发 token。
///
/// 口径 = 与裸 decode 的 `hist` 同式(逐 token 追加、窗口截尾),不是配置项:
/// 它约束的是"惩罚集有多大",过大会让长思考被无谓压制。
const SPEC_HIST_KEEP: usize = 64;

/// 自适应 depth 的观察窗（轮）：DOCS 29.7k 的 d7 尾部在第 8 轮后已足够
/// 稳定；16 轮会把 d7 的税拖得过长。
/// **verify 的逐行视图表**(KVMem 部分驻留档;v2 臂)。
///
/// 只在会话链真的含 `PENDING_LOAD_PLACEHOLDER` 时构造;无洞时保持旧 dense
/// 路径(逐字不变)。`bt8` 是 `[rows, nb]`,每行一条视图页链;`lens` 是视图内
/// token 数(不是逻辑 pos);`slots` 仍是会话链物理页的扁平槽号(K/V 写入位)。
struct VerifyViewTables {
    bt8: Vec<f32>,
    lens: Vec<f32>,
    slots: Vec<f32>,
}

const SPEC_ADAPTIVE_WARMUP_ROUNDS: u32 = 8;
/// 本轮**提交面** live 草稿数 P：预算 rem 个 token 时最多 P+1 个提交行。
/// rem=1 ⇒ P=0（裸 decode，调用方不应发 spec）；rem>=2 ⇒ P=min(K, rem-1)。
///
/// ⚠️ **只裁提交/接受，不裁 verify 图**：verify 图与 FI plan 恒为 `K+1` 行
/// （2026-10-25 P0 判例；图形状改小会复用满宽 CUDA graph 并自旋）。
fn live_depth_for_budget(depth: usize, remaining: usize) -> usize {
    if remaining <= 1 {
        0
    } else {
        depth.min(remaining - 1)
    }
}
/// 纯函数（单测锚）：滚动 tokens/round 是否低于降档门。
fn adaptive_should_downshift(rounds: u32, tokens: u32, min_tpr: f32) -> bool {
    rounds >= SPEC_ADAPTIVE_WARMUP_ROUNDS && (tokens as f32 / rounds as f32) < min_tpr
}

/// **两半表 (max 值, 索引) 合并**(2026-10-26 分布式 argmax)。
///
/// `a` / `b` = 每行 2 个 f32(`[val, idx]`),`a` = rank0(低半词表),
/// **平局归 rank0** —— 与旧 host fold 的「取最小索引」逐位一致。
/// 返回每行**全局索引**(f32 过线,与 `tok` 槽同口径)。
fn dist_argmax_merge(a: &[f32], b: &[f32], half_vocab: usize) -> Result<Vec<f32>> {
    if a.len() != b.len() || a.len() % 2 != 0 {
        return Err(ModelError::EngineInvariant {
            what: format!(
                "分布式 argmax 行尺寸不符:a={} b={}(应为 2×行数)",
                a.len(),
                b.len()
            ),
        });
    }
    let rows = a.len() / 2;
    let mut out = Vec::with_capacity(rows);
    for r in 0..rows {
        let (va, ia) = (a[2 * r], a[2 * r + 1]);
        let (vb, ib) = (b[2 * r], b[2 * r + 1]);
        let idx = if vb > va {
            ib as usize + half_vocab
        } else {
            ia as usize
        };
        out.push(idx as f32);
    }
    Ok(out)
}

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
        _kv_slots: [u32; owl_shared::config::SPEC_VERIFY_ROWS_MAX + 1],
        gdn_slot: usize,
        grew: bool,
        depth: usize,
    ) -> Result<TurnEvent> {
        if depth == 0 || depth > self.dev.spec_rt.depth {
            return Err(ModelError::Msg(format!(
                "spec round depth={depth} 越界（max={}）",
                self.dev.spec_rt.depth
            )));
        }
        let sid = self.turns.active()?.session_id;
        if grew {
            let ts_bt = std::time::Instant::now();
            self.write_bt(sid).await?;
            mrec("spec.bt", ts_bt.elapsed());
        }
        let prof = self.boot.probes.step_profile;
        let t0 = prof.then(std::time::Instant::now);
        let t_round = std::time::Instant::now();
        let t_snap;
        let t_verify;
        let mut t_rollback = std::time::Duration::ZERO;
        let mut t_propose = std::time::Duration::ZERO;
        // **VLM m-RoPE delta**(V5):图像提示经塔后,文本主干的 rope 位置
        // 相对真实 token 位置有常量偏移;verify 行必须与 decode 同式
        // (`pos_rope = pos + delta`),但 KV slots/lens 仍用真实位置。
        let (vlm_delta, rope_delta) = {
            let act = self.turns.active()?;
            let vlm_delta = act.vlm.as_ref().map(|v| v.ins.delta).unwrap_or(0);
            // **leave-gap**:verify/replay 的 rope 位置用绝对位置账;
            // block[0] = 上一枚已发射 token ⇒ 其位置 = next_rope_pos − 1。
            // draft 仍沿用草稿池的既有 delta（其正确性只影响 AL，不影响
            // target greedy 恒等）。
            let rope_delta =
                i64::from(act.next_rope_pos.saturating_sub(1)) - pos as i64;
            (vlm_delta, rope_delta)
        };
        owl_shared::log_debug!(
            "spec-round",
            "轮开始 token={token} pos={pos} depth={depth} vlm_delta={vlm_delta} rope_delta={rope_delta}"
        );

        // ① 快照(轮首;恢复后仍有效 → 只在失效时拍)
        t_snap = self.ensure_spec_snap(gdn_slot).await?;

        // ② 草稿到位(host 账;轮末 propose 图/eager dtoh 产出。首轮 =
        // prefill seed propose(eager);Dumb = anchor 重复)
        let (drafts, t_drafts) = self.acquire_drafts(token, pos, depth).await?;
        t_propose += t_drafts;
        // proposal q 与草稿同轮产出；lookup 改写后 reconcile 成 one-hot。
        let mut proposal_q = self.tspec()?.spec_q_host.take();
        if let Some(q) = proposal_q.as_mut() {
            let _ = q.reconcile(&drafts);
            mcnt("spec.q.round", 1);
        } else {
            mcnt("spec.q.none", 1);
        }
        // **预算只裁提交面，不裁 verify 图的 target 形状**（2026-10-25 P0 修）：
        // proposal 与 verify 图恒为 W=K+1 行；预算不足时只接受/提交前 P+1 行。
        // 旧实现把 `live+1` 传给 `verify_round`，同时改掉 FI plan/GDN 行数，
        // 但 CUDA graph 是按 K+1 行捕获的 ⇒ T<K+1 的尾批复用满宽图，
        // FI 尾部 CTA 等待永不到来的信号（GPU 100% 自旋）。
        // 这里只算 `live` 作为提交上限；verify 仍跑满 `block.len()`，
        // 未接受的行由既有 fold/回滚路径丢弃。
        let live = {
            let act = self.turns.active()?;
            let rem = act.max_new.saturating_sub(act.out.len());
            live_depth_for_budget(depth, rem)
        };
        if live < depth {
            mcnt("spec.tail.cap", 1);
        }

        // ③ verify 块 [anchor, d1..dk] @ pos..pos+depth:图态(一次 replay
        // + fold 记录)或 eager fallback(旧 verify_forward)。fold 记录
        // = 每层 8 件(conv 三 raw + 递推五件)。
        let block: Vec<u32> = std::iter::once(token).chain(drafts.iter().copied()).collect();
        // verify 行数 = anchor + 草稿(depth+1;与图的 depth1 桶同宽)
        let rows_n = block.len();
        // A11:采样接受开关 + 采样参数 + 惩罚历史(必须在借 verify_graph 前取好)
        let (sampler_on, sampler_cfg) = {
            let act = self.turns.active()?;
            (act.sampler_on, act.sampler)
        };
        // **分布式采样门**(2026-10-26):采样档 + 词表切 + 图带 cand 槽
        // + topk 有界。重复惩罚由设备端 top-32 核先行施加(2026-10-26),
        // 不再回退整行 logits。
        let use_cand = sampler_on
            && self.head_split_active()
            && sampler_cfg.topk > 0
            && sampler_cfg.topk <= 32
            && proposal_q.is_none();
        let mut hist: Vec<u32> = {
            let act = self.turns.active()?;
            let tail = act.prompt_ids.len().saturating_sub(SPEC_HIST_KEEP);
            act.prompt_ids[tail..].iter().chain(act.out.iter()).copied().collect()
        };
        let seed_base = {
            let act = self.turns.active()?;
            act.seed.unwrap_or_else(|| act.id.wrapping_mul(0x9E3779B97F4A7C15))
        };
        // XQA 页表:在借 self.dev.graphs.verify 之前算好(自 sessions 取真块链)
        let xqa_pt_v = self.xqa_pt_of(sid);

        let (ids, hidden, records, hidden_owned, row_logits, cand_rows, t_v) = self
            .verify_round(
                &block,
                pos,
                gdn_slot,
                sampler_on,
                rope_delta,
                &xqa_pt_v,
                sid,
                depth,
                use_cand,
            )
            .await?;
        t_verify = t_v;
        if prof {
            owl_shared::log_info!("spec-prof", "pos={pos} verify={:?}", t0.unwrap().elapsed());
        }

        // ④ 接受(A11):采样档 = **抽-比式**;贪心档 = 原式(argmax 比对)
        // 原实现只有贪心接受 ⇒ spec 臂完全不采样 ⇒ 请求温度/top_p/seed
        // 在 spec 开时失效 = 事实贪心(27B instruct 复读吸引子;A7/A8 根因)。
        let (m, emitted, bonus) = if sampler_on {
            if let Some(cs) = cand_rows.as_ref() {
                // **候选式采样**(2026-10-26):只在 ≤64 个候选上做 temperature →
                // top-k → softmax → top-p → 抽样;RNG/历史语义与 logits 路逐字同。
                if cs.len() < live + 1 {
                    return Err(ModelError::EngineInvariant {
                        what: format!("分布式采样行数不足:{} < {}", cs.len(), live + 1),
                    });
                }
                let mut s: Vec<u32> = Vec::with_capacity(live + 1);
                for i in 0..=live {
                    let mut rng =
                        seed_base ^ ((pos + i) as u64).wrapping_mul(0xBF58476D1CE4E5B9);
                    let tok =
                        crate::sampler::sample_from_candidates(&cs[i], &sampler_cfg, &mut rng);
                    s.push(tok);
                    hist.push(tok);
                }
                let m = spec_sampled_accept_prefix(&drafts[..live], &s[..live]);
                (m, s[..=m].to_vec(), s[m])
            } else {
            let Some(lg) = row_logits.as_ref() else {
                return Err(ModelError::Msg(
                    "spec 采样接受需要 per-row logits(verify 路未出 logits 槽)".into(),
                ));
            };
            let vocab = self.dev.boot.model.vocab_size();
            if lg.len() < rows_n * vocab {
                return Err(ModelError::Msg(format!(
                    "spec logits 槽尺寸不符:得 {} < {}×{}",
                    lg.len(),
                    rows_n,
                    vocab
                )));
            }
            if let Some(q) = proposal_q.as_ref() {
                // NInfer 路线：min(1,p/q) 接受；拒绝时从 max(p-q,0) 抽 correction。
                let (m, emitted, bonus) = super::proposal_q::accept_rejection(
                    lg,
                    vocab,
                    live + 1,
                    &drafts[..live],
                    live,
                    q,
                    &sampler_cfg,
                    &mut hist,
                    seed_base,
                    pos,
                )?;
                if m < live {
                    mcnt("spec.reject", 1);
                }
                (m, emitted, bonus)
            } else {
                // 非 DFlash2（MTP/Dumb）：旧「目标抽样 == draft」等价式。
                let mut s: Vec<u32> = Vec::with_capacity(live + 1);
                let mut scratch: Vec<f32> = Vec::with_capacity(vocab);
                for i in 0..=live {
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
                let m = spec_sampled_accept_prefix(&drafts[..live], &s[..live]);
                (m, s[..=m].to_vec(), s[m])
            }
            } // 候选式采样 / logits 老路
        } else {
            // **argmax/greedy 路**:只比 target live 前 P 个 draft
            let m = (0..live).take_while(|&i| drafts[i] == ids[i]).count();
            (m, ids[..=m].to_vec(), ids[m])
        };
        // **AL 计数提到分支外**(2026-10-19 修):两条路(采样/argmax)都要记 ——
        // 旧形只挂在采样路 ⇒ `mode = "greedy"` 档的 AL **在仪表上恒为 1.00**
        // (实测:greedy 跑 22 轮出 64 tok = 真 AL≈2.9,而计数器读 0 ⇒ 判读被带偏;
        //  引擎行为无错,错的是仪表)。语义 = 每轮接受数;降级态机读的是
        // `spec.recent_*`(与本计数无关,不受影响)。
        mcnt("spec.sample.rounds", 1);
        mcnt("spec.sample.accepted", m as u64);
        // 自适应 depth 的 active 事实（tag 有界：d3/d7；/debug/metrics 直读）
        mcnt(&format!("spec.depth.d{depth}"), 1);
        // **AL/成本降档门（T1b）**：滚动 tokens/round 低于阈值 ⇒ 只降不升。
        // 静态 ctx 阈值在 DOCS 29.7k 会误选 d7；这个门用真实接受账补上。
        if self.dev.spec_rt.adaptive_depth {
            let min_depth = self.dev.spec_rt.depth_min;
            let min_tpr = self.dev.spec_rt.adaptive_min_tpr;
            let t = self.tspec()?;
            t.adaptive_rounds = t.adaptive_rounds.saturating_add(1);
            t.adaptive_tokens = t.adaptive_tokens.saturating_add((m + 1) as u32);
            let avg = t.adaptive_tokens as f32 / t.adaptive_rounds as f32;
            let ready = t.adaptive_rounds >= SPEC_ADAPTIVE_WARMUP_ROUNDS;
            let low = adaptive_should_downshift(t.adaptive_rounds, t.adaptive_tokens, min_tpr);
            let down = !t.adaptive_downshift && low && t.adaptive_low_windows >= 1;
            // **per-round m 账**（T1b 校准用）:窗口内每轮一行，直接看 AL 形状。
            owl_shared::log_info!(
                "spec-adaptive",
                "pos={pos} depth={depth} m={m} window_rounds={} window_tokens={} tpr={avg:.2} min_tpr={min_tpr:.2} ready={ready} low={low} low_windows={} down={down}",
                t.adaptive_rounds,
                t.adaptive_tokens,
                t.adaptive_low_windows
            );
            if !t.adaptive_downshift {
                if low {
                    t.adaptive_low_windows = t.adaptive_low_windows.saturating_add(1);
                } else if ready {
                    t.adaptive_low_windows = 0;
                }
            }
            if down {
                t.adaptive_downshift = true;
                mcnt("spec.adaptive.down", 1);
                owl_shared::log_info!(
                    "spec-adaptive",
                    "pos={pos} 连续 2 窗 tokens/round={avg:.2} < {min_tpr:.2}（{} 轮）⇒ depth {} → {}（只降不升）",
                    t.adaptive_rounds,
                    depth,
                    min_depth
                );
            } else if ready && !t.adaptive_downshift {
                // 本窗不降：开下一个观察窗（累计平均会拖到 70+ 轮才看见尾部衰减）
                t.adaptive_rounds = 0;
                t.adaptive_tokens = 0;
            }
        }
        // B4 自适应降级态机(§6.24;DFlash2 专属 —— 同步机制走 dflash encode)
        // 2026-10-12 深夜重放 + 滚动 AL 窗口:连击 streak 会被"偶发 m=1"
        // 破解(随机域 AL≈1.1 时 9.3 t/s 案),窗口均值 < 盈亏线即降级
        let degraded_now;
        if self.dev.spec_rt.dflash_tap_count > 0 && self.boot.probes.degrade_after > 0 {
            let (degrade_after, probe_every0) = (self.boot.probes.degrade_after, self.boot.probes.probe_every);
            let spec = self.tspec()?;
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
            // ⚠️ **2026-10-14 那条结论已被 2026-10-17 实测推翻**:当时看“裸步成本
            // 几乎不随 ctx 变(25.3→26.4 ms @610→9010,±5%)”⇒ 取消 ctx 作废;
            // 但在 **32k** 上裸步 = 38.4 ms(pos 564 时为 24.5 ms,**+57%**)——
            // 那条实测只覆盖到 9k ⇒ 作废规则已**恢复**(`scheduler.rs` “分母与
            // ctx 配对”),并用**播种步**重采(否则作废后会回到无分母态)。
            let (rc, bc) = (spec.round_cost_ewma_ms, spec.bare_cost_ewma_ms);
            // 盈亏线 = 轮/裸 实测比(夹 [2.0,6.0]);`None` = 分母未测
            // ⇒ **窗口判据不生效**(2026-10-17:旧兜底常量 2.5 在 32k 会把
            // 盈利档 AL 2.25 误降级;分母已由首 turn 播种步补上,见 break_even_line)
            let break_even = crate::exec::break_even_line(rc, bc);
            const WINDOW_ROUNDS: u32 = 16;
            const WINDOW_MS: f32 = 600.0;
            let window_starved =
                if spec.recent_rounds >= WINDOW_ROUNDS || spec.recent_ms >= WINDOW_MS {
                    let avg = spec.recent_tokens as f32 / spec.recent_rounds.max(1) as f32;
                    let win_tps = spec.recent_tokens as f32 * 1000.0 / spec.recent_ms.max(1.0);
                    spec.recent_rounds = 0;
                    spec.recent_tokens = 0;
                    spec.recent_ms = 0.0;
                    let starved = break_even.is_some_and(|line| avg < line);
                    if starved || break_even.is_none() {
                        let line = crate::exec::break_even_text(break_even);
                        let why = if break_even.is_none() {
                            "(分母未测 ⇒ 本轮不判;播种步在首 turn 发出)"
                        } else {
                            ""
                        };
                        owl_shared::log_warn!("spec-degrade", "窗口 AL={avg:.2} vs 盈亏线 {line}{why}(轮 {rc:.0}ms/裸 {bc:.0}ms;窗口吞吐 {win_tps:.1} tok/s)"
                        );
                    }
                    starved
                } else {
                    false
                };
            // ── **连击判据的闸门**(2026-10-18 实测立案)──
            //
            // 连击(m=0×6)只说明"最近 6 轮草稿全灭",**不代表投机本身亏本**:
            // 实测(重复上下文,ctx 8.5k)一次连击触发降级时,窗口 AL ≈ **2.9-3.9
            // tok/轮**,而盈亏线 = 轮/裸 ≈ 2.0 ⇒ 留在 spec 每 token ≈ 9.4 ms,
            // 降级裸步 **27.5 ms/token** ⇒ **一次误降级罚 ~1.4 s**(占该请求
            // 解码相位 35%)。而连击真正该抓的是"草稿系统性坏掉"(此时窗口 AL
            // 也会塌到线下)。
            //
            // ⇒ 连击只在**窗口 AL 也低于盈亏线**时才降级(用**当前部分窗口**,
            // 不等 16 轮门);已降级态照旧保持(出口 = 探测命中)。
            // 分母未测(break_even=None)⇒ 不因连击降级(保守留在 spec;窗口判据
            // 此时同样不生效,播种步会在首 turn 补上分母)。
            let al_partial = if spec.recent_rounds > 0 {
                spec.recent_tokens as f32 / spec.recent_rounds as f32
            } else {
                f32::INFINITY
            };
            let streak_gate = break_even.is_some_and(|line| al_partial < line);
            let degraded = window_starved || (degraded0 && (spec.spec_degraded || streak_gate));
            if degraded && !spec.spec_degraded {
                let line = crate::exec::break_even_text(break_even);
                owl_shared::log_warn!("spec-degrade", "m=0×{streak} win_starved={window_starved} 盈亏线={line} → 降级裸 decode(probe 每 {probe_every} tok)"
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
            degraded_now = self.tspec_ref()?.spec_degraded;
        }
        // P0 修(2026-10-12):降级态在 emit 循环前定格 —— 预算/EOS 会在
        // 循环内 complete()(active 被取走),循环后触 tspec() = panic
        // mcnt("spec.pos.last", ...) 在 metrics 段(pos+m+1 gauge)
        // 对齐探针(E5-DF3 AL=0 排查):drafts vs 目标验证行逐位对照 ——
        // 附近命中(drafts[i]==ids[i±1]) = 位移对齐 bug;全散 = 分布质量
        if self.boot.probes.dflash_probe {
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

        // pos = 本轮入口位置(接受前)。带上它 ⇒ 不用 `step_profile`(会加
        // ~30ms/轮同步读、实测把 AL 从 2.90 抬到 4.54)也能画 AL-vs-position。
        owl_shared::log_debug!("spec-round", "pos={pos} 接受完成 m={m} bonus={bonus} emitted={}", emitted.len());
        // ⑤ 部分接受 → GDN 回滚 + 已接受前缀重放(v1 窗口重处理,施工
        // 方案 §四.4);快照无条件失效(下轮轮首重拍)。
        // 轮不变量:restore 只回到 state@F-1,而下一轮需要 state@fed'-1
        // = state@F+m —— [anchor, d1..dm] 的贡献在 verify 中已推进、被
        // restore 回滚,必须重放补回(已接受行 KV 覆写幂等 —— 同快照同
        // 输入确定性同值)。m=0 也要重放 anchor 一行。
        // ⑤ (已合并) 原此处先对 MTP 跑一次 propose，⑦ fed 提交后又跑一次
        // —— 两个分支输入/输出完全相同，MTP 每轮白付一倍 propose 税
        // （2026-10-21 定谳：8k ctx 时 propose 40.6ms/轮 = 2×19.4ms read）。
        // 统一只在 ⑦ 跑一次；toks 仍在这里准备。
        let toks: Vec<u32> = {
            let mut t = drafts[..m].to_vec();
            t.push(bonus);
            t
        };
        if hidden_owned {
            // eager 路的 hidden = 竞技场根块,收割后归池;图态 = 图输出槽
            // (持久复用,free = 炸图,M4 立案注记)
            let face = self.dev.graphs.session.face_mut();
            face.free(&[hidden.id]).await?;
        }
        owl_shared::log_debug!("spec-round", "⑤ 草稿入参准备完成(propose 统一在⑦)");
        // ⑥ fold(尾部:图发射异步,restore 单命令;下一轮 verify 流序
        // 排其后,GPU ~1ms 藏于前端)
        //
        // **轮内页边界快照**(§9.1(6)a;`[pool] snap_decode_stride`):本轮提交
        // `(pos, pos+m]` 跨过页边界 b ⇒ 先把池态卷到 state@b 拍一张,再卷到
        // state@fed —— 这样生成相位的每个页边界都有快照(旧状:只有 prefill 段有
        // ⇒ 短 prompt 的尾部复用面受限于 prompt 里最后一个整页)。
        // 可行前提 = `spec_snap` 缓冲在本轮**保持 pristine**(= state@pos):
        // fold 族自本日起与 eager 路径同形(**写池**,不再就地写缓冲)。
        if m < depth {
            let ts = std::time::Instant::now();
            // 记录齐 = 可折(图态桶 / eager fold_tree 两路都吃记录);
            // 不齐 = 旧"前缀重放"(整模前向)唯一可行路, 该档不做轮内边界快照
            let foldable =
                records.len() == self.dev.pool.gdn_count() * owl_shared::config::SPEC_FOLD_RECORDS_PER_LAYER;
            if foldable {
                // **消费方门**(2026-10-19,同 prefill 的 `snap_on` 判据):轮内页边界
                // 快照的唯一消费者 = **前缀复用**(`restore_snap` 只在 prefix 命中时调);
                // `prefix_cache = false` ⇒ 缓存区份额 0 ⇒ 复用面结构上恒 0
                // ⇒ 下面的「卷到边界 + 拍 + 卷回」三段是**纯浪费**(实测
                // `spec.bnd.fold1` 1.17ms + `spec.snap` 0.90ms ≈ 2.1ms/边界)。
                // 注:`snap_decode_stride = 0` 同样关(两处门同向)。
                let snap_consumer = self.boot.cfg.knobs.prefix_cache && self.dev.pool.snap_capacity() > 0;
                if let Some(b) = snap_consumer
                    .then(|| {
                        Self::decode_boundary_in(
                            pos,
                            pos + m + 1,
                            self.dev.pool.geom.page,
                            self.boot.cfg.knobs.snap_decode_stride,
                        )
                    })
                    .flatten()
                {
                    let t_b = std::time::Instant::now();
                    self.spec_fold_to(gdn_slot, &records, b - pos, depth).await?;
                    let t_fold1 = t_b.elapsed();
                    // 尾 token = 本轮已提交、尚未进 `act.out` 的那几枚
                    let tail = &toks[..b - 1 - pos];
                    owl_shared::log_debug!("spec-round", "边界:snap_tail 起 b={b}");
                    self.capture_gdn_snap_tail(
                        sid,
                        b,
                        tail,
                        owl_store::SnapDurability::Transient,
                        true,
                        true,
                    )
                    .await?;
                    owl_shared::log_debug!("spec-round", "边界:snap_tail 完");
                    let t_snap = t_b.elapsed() - t_fold1;
                    // 分相计时(2026-10-19;判"轮内边界值不值"的账:
                    // 卷/拍/回卷三段与 prefill 边界同粒度)
                    mrec("spec.bnd.fold1", t_fold1);
                    mrec("spec.bnd.snap", t_snap);
                    if self.boot.cfg.knobs.probes.debug {
                        owl_shared::log_debug!(
                            "dbg snap",
                            "轮内边界 fed={b}(pos={pos} m={m})⇒ 快照已拍(卷 {:.2}ms/拍 {:.2}ms),卷回 state@{}",
                            t_fold1.as_secs_f64() * 1000.0,
                            t_snap.as_secs_f64() * 1000.0,
                            pos + m + 1
                        );
                    }
                }
                owl_shared::log_debug!("spec-round", "非边界:fold(m+1={}) 起", m + 1);
                self.spec_fold_to(gdn_slot, &records, m + 1, depth).await?;
                owl_shared::log_debug!("spec-round", "非边界:fold 完");
            } else {
                // eager fallback:旧路 —— 前缀重放(整模前向,已接受行
                // KV 覆写幂等;m=0 也重放 anchor 一行)
                {
                    // ★ **双 rank 镜像**(P2.1-c,2026-10-19):GDN 快照是**每 rank 各自的池**
                    // ⇒ 两边都要恢复。只恢复 rank0 ⇒ 两 rank 状态不同源 ⇒ 重放出的
                    // logits 错 ⇒ **m=0 回滚路的输出与裸 decode 不一致**(现场定谳:
                    // `dumb` 档必走此路,同样退化)。
                    let (dev, peer) = (&mut self.dev, self.peer.as_mut());
                    match peer {
                        None => {
                            let face = dev.graphs.session.face_mut();
                            dev.pool.restore_spec_snap(face, gdn_slot).await?
                        }
                        Some(p) => {
                            let (r0, r1) = crate::exec::join2(
                                dev.pool
                                    .restore_spec_snap(dev.graphs.session.face_mut(), gdn_slot),
                                p.pool.restore_spec_snap(p.graphs.session.face_mut(), gdn_slot),
                            )
                            .await;
                            r1?;
                            r0?
                        }
                    }
                }
                let replay: Vec<u32> = std::iter::once(token)
                    .chain(drafts[..m].iter().copied())
                    .collect();
                owl_shared::log_debug!("spec-round", "回滚重放:eager verify_forward 起(peer 镜像)");
                let (_, replay_hidden, _) =
                    self.verify_forward(&replay, pos, gdn_slot, false, rope_delta).await?;
                owl_shared::log_debug!("spec-round", "回滚重放:eager verify_forward 完");
                let face = self.dev.graphs.session.face_mut();
                face.free(&[replay_hidden.id]).await?;
            }
            t_rollback = ts.elapsed();
        }
        self.tspec()?.spec_snap_valid = false;

        // ⑥ 提交:fed = bonus 位(F');发射 = 接受 drafts + bonus(m+1)
        // (A11:采样档 emitted = 采样表前 m+1 位;贪心档 = ids[..=m])
        let toks: Vec<u32> = emitted;
        {
            let act = self.turns.active_mut()?;
            act.fed = pos + m + 1;
        }
        // ⚠️ **不要在 spec 轮里拍页边界快照**(2026-10-19 试过并撤回):
        // ①`act.out` 的提交在本函数更后面 ⇒ 此处 `fed` 领先 `out`,
        // `capture_gdn_snap` 的 `tokens_upto` 立刻报"token N 未喂"(实测);
        // ②即便挪到提交之后,spec 轮**大多跨过**边界、状态已过界 ⇒ 按边界键存
        // = 静默错值。中段回卷法(用 `fold` 族把状态卷到边界再拍)见
        // `docs/arch/agent-服务面-设计与施工.md` §9.1(5)a —— 未做。
        let _ = sid;

        // ⑦ extend + propose(同迭代消费 verify hidden;产出下轮草稿
        // host 账)。DFlash2 = encode + 噪声块 propose;图态 = 桶 m 一发
        // (MTP);eager fallback = propose_round(MTP)。
        // **预算尽则不再 propose**:tail 轮的 target 只算到 live 行，
        // 未算的 tap 行不能给下一轮满 W 草稿；此轮已是末轮，直接省掉。
        let round_done = {
            let act = self.turns.active()?;
            act.out.len().saturating_add(toks.len()) >= act.max_new
        };
        if !round_done {
        // DFlash2 lookup 草稿:历史 = prompt_ids + out(含当前 token)
        // + 本轮 verify 块 `toks`(ids[0..=m];末位 bonus 是下一轮 anchor)。
        let lookup = self.dflash_lookup_for_tail(&toks)?;
        let d2_arc7 = match self.dev.spec_rt.drafter.as_ref() {
            Some(crate::spec_runtime::Drafter::DFlash2(d)) => Some(std::sync::Arc::clone(d)),
            _ => None,
        };
        if let Some(d2) = d2_arc7 {
            let ts = std::time::Instant::now();
            // E5-DF4:图态优先(单图一轮 encode+propose;eager 回退 =
            // OWL_DFLASH_EAGER=1 或图降级)
            if self.dev.graphs.dflash.is_some() && !self.boot.probes.dflash_eager {
                let d = self
                    .dflash_graph_round(m, pos, toks[m], lookup.as_ref())
                    .await?;
                t_propose += ts.elapsed();
                self.tspec()?.spec_drafts_host = Some(d);
            } else {
            let noencode = self.boot.probes.dflash_noencode;
            let taps: Vec<owl_iface::contract::Bytes> = (0..self.dev.spec_rt.dflash_tap_count)
                .map(|i| {
                    self.dev.graphs
                        .verify_of(depth)
                        .ok_or_else(|| ModelError::EngineInvariant {
                            what: "dflash 模式要求 verify 图(taps 输出)".into(),
                        })?
                        .output_block(&format!("tap{i}"))
                        .ok_or_else(|| ModelError::EngineInvariant {
                            what: format!("tap{i} 输出槽缺席(草稿档位不一致)"),
                        })
                })
                .collect::<Result<Vec<_>>>()?;
            if !noencode {
                self.dflash_encode(&d2, &taps, m, pos).await?;
            }
            let fp = pos + m + 1;
            let bonus = toks[m];
            let d = self.dflash_propose(bonus, fp, lookup.as_ref()).await?;
            if self.boot.probes.dflash_probe && self.dev.spec_rt.stats.rounds < 3 {
                let toks_txt = self.boot.tok.decode(&d);
                owl_shared::log_debug!("dflash-probe", "pos={fp} bonus={bonus} drafts={d:?} txt={toks_txt:?}");
            }
            t_propose += ts.elapsed();
            self.tspec()?.spec_drafts_host = Some(d);
            }
        } else if self.dev.graphs.has_propose(depth) && !self.boot.probes.propose_eager {
            let ts = std::time::Instant::now();
            let d = self.propose_graph_step(m, &toks, pos, depth).await?;
            t_propose += ts.elapsed();
            self.tspec()?.spec_drafts_host = Some(d);
        } else if self.dev.spec_rt.drafter.is_some() {
            let ts = std::time::Instant::now();
            let nd = self.propose_round(&hidden, m, &toks, pos, depth).await?;
            t_propose += ts.elapsed();
            // 块 → host(轮首消费;块即焚)
            let mut buf = vec![0u8; 4 * depth];
            {
                let face = self.dev.graphs.session.face_mut();
                face.dtoh(&nd, &mut buf).await?;
                face.free(&[nd.id]).await?;
            }
            self.tspec()?.spec_drafts_host = Some(
                buf.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u32)
                    .collect(),
            );
        }
        } else {
            self.tspec()?.spec_drafts_host = None;
            mcnt("spec.propose.skip", 1);
        }
        if hidden_owned {
            // eager 路的 hidden = 竞技场根块,收割后归池;图态 = 图输出槽
            // (持久复用,free = 炸图,M4 立案注记)
            let face = self.dev.graphs.session.face_mut();
            face.free(&[hidden.id]).await?;
        }

        // ⑧ 逐 token 发射(eos/预算;事件入队,pump 逐个出)
        {
            // 崩坏排查打点(E5-DF4):per-boot per-round 事实值
            // (OWL_FACT_PROBE 门控;案结后默认关 —— 动态 tag 每轮 ×9
            // 无限增殖,metrics tag 爆炸 + 512 上限挤掉真账,2026-10-10)
            if self.boot.probes.fact_probe {
                let bs = self.boot.seq;
                let rn = self.dev.spec_rt.stats.rounds;
                mfact(bs, &format!("r{rn}.pos"), pos as u64);
                mfact(bs, &format!("r{rn}.m"), m as u64);
                mfact(bs, &format!("r{rn}.bonus"), bonus as u64);
                for (i, (&dv, &iv)) in drafts.iter().zip(ids.iter()).take(depth).enumerate() {
                    mfact(bs, &format!("r{rn}.d{i}"), dv as u64);
                    mfact(bs, &format!("r{rn}.i{i}"), iv as u64);
                }
            }
        }
        self.dev.spec_rt.stats.rounds += 1;
        self.dev.spec_rt.stats.sum_m += m;
        if m == depth {
            self.dev.spec_rt.stats.full += 1;
        } else if m == 0 {
            self.dev.spec_rt.stats.zero += 1;
        } else {
            self.dev.spec_rt.stats.partial += 1;
        }
        // E4:m 直方图进 metrics(tag 有界 0..=7;/debug/metrics 直读,
        // 免日志 grep;AL 均值 = spec.accepted/spec.rounds 同源可算)
        mcnt(&format!("spec.m{m}"), 1);
        // **先落本 turn 诊断账,再发事件**:`sample_and_emit` 可能在预算尽/
        // EOS 时把活跃 turn 收掉;若等发完再 `tspec()` 会命中"活跃 0 个"。
        {
            let ts = self.tspec()?;
            ts.spec_rounds += 1;
            ts.spec_accepted += m as u64;
            ts.spec_emitted += (m + 1) as u64;
        }
        let ts = std::time::Instant::now();
        let mut evq = std::collections::VecDeque::new();
        for &t in &toks {
            let ev = self.sample_and_emit(t).await?;
            // 预算尽的末 token 走 Token + pending Completed(见 phases)——
            // 故 active 已尽也算 done,余段作废
            let done = matches!(ev, TurnEvent::Completed { .. }) || self.turns.active.is_empty();
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
            let tag = if records.len()
                == self.dev.pool.gdn_count() * owl_shared::config::SPEC_FOLD_RECORDS_PER_LAYER
            {
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


    /// **回卷到 state@(pos + rows)**（spec 轮内唯一折叠口；§9.1(6)a）。
    ///
    /// 语义：`spec_snap` 缓冲（轮内**保持 pristine** = state@pos）→ 池，
    /// 再把 verify 记录的前 `rows` 行重放进池（行 0 = anchor）。
    ///
    /// - **图态** = 桶 `rows-1` 一发（fold 族叶子 = 池 GDN 态，`slots` = 会话行；
    ///   与 eager 路径**同形**）⇒ 缓冲不被写 ⇒ 同一轮可多次调用
    ///   （跨页边界： state@b 拍快照 → 再 state@fed）；
    /// - **eager 档**（无图）= 记录前缀一次 multi-root eval（零整模前向）。
    ///
    /// 调用方保证 `records` 齐（= 8 × GDN 层数）且 `1 ≤ rows ≤ depth`。
    ///
    /// ⚠️ **TP 面现状**（对齐 `capture_gdn_snap` 的双 rank 形态看缺口）：rollback
    /// 目前只走本 rank（`self.dev`）；capture 已是 dual-rank（`snap_dev` + `join2`）。
    /// 两半不对称 = TP 真开时 fold 需同样补 peer 臂（W4A8-TP 的 lane，见其卷）。
    async fn spec_fold_to(
        &mut self,
        gdn_slot: usize,
        records: &[Bytes],
        rows: usize,
        depth: usize,
    ) -> Result<()> {
        if rows == 0 || rows > depth {
            return Err(ModelError::Msg(format!(
                "spec fold rows={rows} 越界（active depth={depth}）"
            )));
        }
        // ★ **双 rank 镜像**(P2.1-c,2026-10-19):fold 写的是**每 rank 自己的 GDN 池**
        // ⇒ 两 rank 必须同跑(否则下一轮 verify 读到本 rank 未 fold 的状态)。
        // 与 `capture_gdn_snap` 的双 rank 形态对齐(旧注释自认的缺口);单卡 = 原路。
        if self.dev.graphs.has_fold(depth) {
            let (dev, peer) = (&mut self.dev, self.peer.as_mut());
            crate::exec::mirror2!(dev, peer,
                d => d.fold_step_dev(gdn_slot, rows, depth),
                p => p.fold_step_dev(gdn_slot, rows, depth))?;
            return Ok(());
        }
        {
            let face = self.dev.graphs.session.face_mut();
            self.dev.pool.restore_spec_snap(face, gdn_slot).await?;
        }
        // eager fold（E5-M4）：快照态 + 记录前缀 → state@pos+rows，
        // **零整模前向**（96 副作用根一次 multi-root eval）
        let m1 = rows;
        let m = rows - 1;
        let d = self.dev.pool.geom.dims;
        let key_dim = d.nk * d.hk;
        let value_dim = d.nv * d.hv;
        let views: Vec<TensorOps> = records
            .iter()
            .enumerate()
            .map(|(i, b)| {
                let shape = match i % owl_shared::config::SPEC_FOLD_RECORDS_PER_LAYER {
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
            .dev
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
            &self.dev.boot.model,
            &views,
            &gdns,
            &slots_t,
            &cu_t,
            gdn_slot,
            m,
            self.dev.env().gdn.scalar,
        )?;
        let refs: Vec<&TensorOps> = roots.iter().collect();
        // env 值语义:先取(方法借 `&self`)再借 face
        let env = self.dev.env();
        let face = self.dev.graphs.session.face_mut();
        owl_models::interpreters::eval_ops_multi_scoped_env(
            &refs,
            face,
            env,
            self.dev.boot.tp.as_ref(),
        )
        .await?;
        Ok(())
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
        // 槽容量 = 池页数。overcommit 档的会话链会大于池(select_partial),
        // XQA 此时已禁用;仍按槽容量截断,避免 verify 输入装填越界。
        let nb = self.dev.pool.geom.nb.max(1);
        let mut v: Vec<f32> = table.iter().take(nb).map(|&b| f32::from_bits(b)).collect();
        v.resize(nb, 0.0);
        v
    }

    /// **verify 的逐行视图表**(KVMem 部分驻留档;v2 臂)。
    ///
    /// 判据:
    /// - 会话链 `[0, ceil((pos+rows)/page))` **无占位符** ⇒ `None`(旧 dense 路径);
    /// - 有洞 + 有视图 ⇒ 逐行 `bt8/lens/slots`;
    /// - 有洞 + 无视图 / `pos < recall_pos` / 页越界 / 视图超容量 ⇒ 结构化 Err。
    ///
    /// `len_r = selected_tokens + (p + 1 - recall_pos)` 是**视图内 token 数**;
    /// `bt8[r]` 是选中页 ++ 尾部页(到 `p/page`),零填充到 `nb`。
    /// `slots[r]` 仍是会话链物理页的扁平槽号 —— 新 token 的 K/V 写入位不变。
    fn verify_view_tables(
        &self,
        sid: u64,
        pos: usize,
        rows: usize,
        page: usize,
        nb: usize,
    ) -> Result<Option<VerifyViewTables>> {
        let page = page.max(1);
        let chain = self.bank.sessions.session(sid)?.block_table.clone();
        let need_pages = pos.saturating_add(rows).div_ceil(page);
        if chain.len() < need_pages {
            return Err(ModelError::Msg(format!(
                "spec verify 视图:会话链 {} 页 < 本轮回需要的 {need_pages} 页(pos={pos} rows={rows} page={page})",
                chain.len()
            )));
        }
        let has_hole = chain[..need_pages]
            .iter()
            .any(|p| *p == crate::scheduler::PENDING_LOAD_PLACEHOLDER);
        if !has_hole {
            return Ok(None);
        }
        let view = self.sel.view().ok_or_else(|| {
            ModelError::Msg(format!(
                "spec verify:会话链前 {need_pages} 页含宿主占位符,但选择层视图缺席\
                 (pos={pos} recall=?)。部分驻留档必须每步有视图;拒发本轮。"
            ))
        })?;
        if pos < view.recall_pos {
            return Err(ModelError::Msg(format!(
                "spec verify:pos={pos} < 视图召回点 {} —— 视图不覆盖本轮,不能用它兜洞",
                view.recall_pos
            )));
        }
        // 选中页必须是设备页且不越界(写视图时已校验,这里防账漂)。
        for &pg in &view.pages {
            if pg == crate::scheduler::PENDING_LOAD_PLACEHOLDER || pg as usize >= nb {
                return Err(ModelError::Msg(format!(
                    "spec verify 视图:选中页 {pg} 非法(占位符或 ≥ 池页数 {nb})"
                )));
            }
        }
        // **所有行共用同一张页表**(= 最后一行,尾部最长),只让 `kv_lens` 逐行
        // 变短。dense verify 本来就是 8 行同表 + 逐行 lens;逐行不同页表虽然
        // v2 签名支持,但会把“尾部页边界”变成 8 份输入,且实测在 DFlash2
        // 接受行上暴露错值。同表 + 逐行 lens 与 dense 路径同构。
        let last = pos.saturating_add(rows.saturating_sub(1));
        let (pages, _) = crate::exec::coverage::view_pages_at(
            &view.pages,
            view.tokens,
            &chain,
            view.recall_pos,
            last,
            page,
        )
        .ok_or_else(|| {
            ModelError::Msg(format!(
                "spec verify 视图:pos={last} 在召回点 {} 之前,视图不覆盖",
                view.recall_pos
            ))
        })?;
        let mut pages = pages;
        // **verify 带宽上限**:只裁 verify 读的页链尾部;decode 仍用完整工作集。
        let verify_cap = self.boot.cfg.knobs.verify_view_tokens;
        let _ = crate::exec::coverage::cap_view_pages(&mut pages, page, verify_cap);
        let view_cap_tokens = pages.len().saturating_mul(page);
        if pages.len() > nb {
            return Err(ModelError::Msg(format!(
                "spec verify 视图超容量:{} 页 > 池 {nb}(pos={last};选中 {}),\
                 需先重选/回灌",
                pages.len(),
                view.pages.len()
            )));
        }
        for (i, &pg) in pages.iter().enumerate() {
            if pg == crate::scheduler::PENDING_LOAD_PLACEHOLDER || pg as usize >= nb {
                return Err(ModelError::Msg(format!(
                    "spec verify 视图:第 {i} 页={pg} 非法(占位符或 ≥ 池页数 {nb};pos={last})"
                )));
            }
        }
        let mut bt8 = vec![0.0f32; rows.saturating_mul(nb)];
        let mut lens = vec![0.0f32; rows];
        let mut slots = vec![0.0f32; rows];
        for r in 0..rows {
            let p = pos + r;
            let raw_len = view.tokens + (p + 1).saturating_sub(view.recall_pos);
            let len = raw_len.min(view_cap_tokens);
            if verify_cap == 0 && raw_len > view_cap_tokens {
                return Err(ModelError::Msg(format!(
                    "spec verify 视图:len={raw_len} > 页容量 {view_cap_tokens}(pos={p}),视图账不平"
                )));
            }
            let row = &mut bt8[r * nb..(r + 1) * nb];
            for (i, &pg) in pages.iter().enumerate() {
                row[i] = pg as f32;
            }
            lens[r] = len as f32;
            let pg = *chain.get(p / page).ok_or_else(|| {
                ModelError::Msg(format!(
                    "spec verify 视图:写入页 {} 越界(链 {} 页,pos={p})",
                    p / page,
                    chain.len()
                ))
            })?;
            if pg == crate::scheduler::PENDING_LOAD_PLACEHOLDER || pg as usize >= nb {
                return Err(ModelError::Msg(format!(
                    "spec verify 视图:写入页 {}={pg} 非法(pos={p})",
                    p / page
                )));
            }
            slots[r] = (pg * page as u32 + (p % page) as u32) as f32;
        }
        if self.boot.probes.dflash_probe {
            owl_shared::log_info!(
                "view-diag",
                "verify pos={} rows={} recall={} selected={} view_tokens={} cap={} pages={} len0={} slot0={} first={:?} last={:?}",
                pos,
                rows,
                view.recall_pos,
                view.pages.len(),
                view.tokens,
                verify_cap,
                pages.len(),
                lens.first().copied().unwrap_or(-1.0),
                slots.first().copied().unwrap_or(-1.0),
                &pages[..pages.len().min(6)],
                &pages[pages.len().saturating_sub(6)..]
            );
        }
        Ok(Some(VerifyViewTables { bt8, lens, slots }))
    }

    /// **③ verify 轮**(S6d):`[anchor, d1..dk]` 一次前向。
    ///
    /// 图态 = 一次 replay + fold 记录(每层 8 件);eager fallback = 旧
    /// `verify_forward`。TP 档两 rank 必须同跑(verify 图含 AR,T=depth+1 行
    /// RowParallel);词表切档 = 两 rank 各出半表、rank0 host 拼接。
    ///
/// 返回 `(ids, hidden, records, hidden_owned, row_logits, 耗时)`。
    /// `hidden_owned = true` ⇒ 调用方负责 free(eager 路的中间块)。
    #[allow(clippy::type_complexity)]
    async fn verify_round(
        &mut self,
        block: &[u32],
        pos: usize,
        gdn_slot: usize,
        sampler_on: bool,
        vlm_delta: i64,
        xqa_pt_v: &[f32],
        sid: u64,
        depth: usize,
        want_cand: bool,
    ) -> Result<(
        Vec<u32>,
        owl_iface::contract::Bytes,
        Vec<owl_iface::contract::Bytes>,
        bool,
        Option<Vec<f32>>,
        Option<Vec<Vec<(f32, u32)>>>,
        std::time::Duration,
    )> {
        let verify_elapsed;
        // **verify 图/FI plan 恒为满宽 depth+1**(2026-10-25 P0 修)。
        // 预算裁剪只作用于提交面(`execute_spec_round` 的 `live`)；
        // 这里不再收 target_rows 参数 —— 否则调用点可再次把图形状改小，
        // 复用满宽 CUDA graph 会触发 FI 尾部 CTA 自旋。
        let target_rows = block.len();
        // 图形状是 `depth+1` 行；块行数不符时**响亮拒**，不许带错形状复用图
        // （否则又回到 2026-10-25 P0 的 FI 尾 CTA 自旋）。
        if self.dev.graphs.verify_of(depth).is_some() && block.len() != depth + 1 {
            return Err(ModelError::EngineInvariant {
                what: format!(
                    "verify 块行数 {} != 图 depth+1 {}（预算裁剪只许改提交面）",
                    block.len(),
                    depth + 1
                ),
            });
        }
        // **视图化 verify(2026-10-23)**:先在 `vg` 借用之前取视图表 ——
        // `verify_view_tables` 只读 bank/sel;有洞 + 有视图 = 逐行表,
        // 无洞 = None(旧 dense 路径逐字不变)。
        let page = self.dev.pool.geom.page.max(1);
        let view_tables =
            self.verify_view_tables(sid, pos, block.len(), page, self.dev.pool.geom.nb)?;
        if view_tables.is_some() {
            // 证据面:本轮回真的走了逐行视图(而不是 dense 表)。
            owl_shared::metrics::with_metrics_store(|m| {
                m.counter_add("spec.verify.view", 1, file!(), line!());
            });
        }
        let (ids, hidden, records, hidden_owned, row_logits, cand_rows) =
        if self.dev.graphs.verify_of(depth).is_some() {
            let vg = self.dev.graphs.verify_mut(depth)?;
        let ts = std::time::Instant::now();
        let bt_chain = {
            let s = self.bank.sessions.session(sid)?;
            s.block_table.clone()
        };
        let ids_f: Vec<f32> = block.iter().map(|&v| v as f32).collect();
        let rope_base = usize::try_from(pos as i64 + vlm_delta).map_err(|_| {
            ModelError::Msg(format!(
                "spec verify: VLM 位置为负(pos={pos}, delta={vlm_delta})"
            ))
        })?;
        let pos_f: Vec<f32> =
            (rope_base..rope_base + block.len()).map(|p| p as f32).collect();
        if bt_chain.len() <= (pos + block.len() - 1) / page {
            owl_shared::log_debug!("spec-dbg", "pos={pos} block={} bt_len={bt_len} page={page}", block.len(), bt_len = bt_chain.len());
        }
        // **视图化 verify**:有洞时用逐行视图表;无洞时用旧 dense 表
        // (8 伪序列共享会话链,逐字不变)。
        let (slots_f, lens_f, bt8_f) = if let Some(v) = &view_tables {
            (v.slots.clone(), v.lens.clone(), v.bt8.clone())
        } else {
            let slots_f: Vec<f32> = (pos..pos + block.len())
                .map(|p| (bt_chain[p / page] * page as u32 + (p % page) as u32) as f32)
                .collect();
            let lens_f: Vec<f32> =
                ((pos + 1)..=(pos + block.len())).map(|p| p as f32).collect();
            // §三十一:bt8 = 页链 × depth1 平铺(8 伪序列共享页链;v2 臂用);
            // 补零到池 nb(会话链短于池时尾部补无效页 0,kv_lens 限界不读)
            let nb_total = self.dev.pool.geom.nb.max(bt_chain.len());
            let bt8_f: Vec<f32> = (0..block.len())
                .flat_map(|_| {
                    bt_chain.iter().map(|&b| b as f32)
                        .chain(std::iter::repeat_n(0f32, nb_total - bt_chain.len()))
                })
                .collect();
            (slots_f, lens_f, bt8_f)
        };
        // XQA 臂:semaphores 每轮清零(KV-split 归约计数;**同流序先于图回放**;
        // 参考实现 = torch sem.zero_();图内不做,故在轮侧发 memset)
        // ⚠️ 必须用 **session face**(池块的账房;verify 图 face 是另一份
        // 账本 → 传别家的块 id = 打到别的块上 → `pump: DeadBlock` 实测踩过)
        // XQA seq_lens = **整请求总 KV 长**(含块内 depth+1 个位置;u32 位型走私)
        let xqa_seq_v = [f32::from_bits((pos + target_rows) as u32)];
        // semaphores 每轮写零(图输入槽;KV-split 归约计数)。
        // `.max(1)` 与声明端同式 —— "无 XQA 面"时槽仍存在(= 1 元素),
        // 避免零长写(挂账 #7)。
        let xqa_sem_v: Vec<f32> = vec![0.0; self.dev.pool.xqa_sem_n().max(1)];
        // **FI 断点臂**的逐轮值(该臂未开 ⇒ 空,不写槽):
        // ① 宿主标量槽 `ctx_total`(= 本轮 kv 总长;plan 的 key)
        // ② 表四件套(q_cu/indices/indptr/last_len;u32 位型过线)
        let page_fi = self.dev.pool.geom.page;
        let ctx_total_fi = pos + target_rows;
        // 门 = **槽在场**(槽属于**图**;判"某核在场"必须看节点/槽,不看旋钮)
        // 槽恒在 verify 图里声明(拓扑恒常);真正的“FI 臂是否消费”看单值臂。
        // **必须把槽在场与臂在场分开**:v2 档槽也在,但表不能按会话链铺
        // (overcommit 时链长 > 池页数,旧代码会写越 idx ⇒ panic)。
        let fi_slots_on = vg.has_input("fi_q_cu")
            && self.boot.cfg.knobs.env.attn.verify_arm.uses_fi();
        if fi_slots_on && view_tables.is_some() {
            return Err(ModelError::Msg(
                "spec verify:FI 臂不支持逐行视图页表(select_partial);\
                 请设 [kernel] verify_arm = \"v2\""
                    .into(),
            ));
        }
        let (fi_q_cu_v, fi_idx_v, fi_indptr_v, fi_ll_v): (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) =
            if fi_slots_on {
                // ⚠️ **必须写本轮真值**(verify 不吃桶化):T=8 的 plan split
                // chunk 数必须与核运行期 `ceil_div(kv_len, chunk)` 一致,否则
                // partial 行 stride 与 `merge_indptr` 打架(同参 A/B:桶 2048
                // ⇒ 空文本/`!!!`、42.9 t/s;真值 ⇒ 连贯 71 t/s)。
                let nb_used = ctx_total_fi
                    .div_ceil(page_fi)
                    .min(bt_chain.len())
                    .max(1);
                let mut idx = vec![0f32; self.dev.pool.geom.nb];
                for (i, b) in bt_chain.iter().take(nb_used).enumerate() {
                    idx[i] = f32::from_bits(*b);
                }
                (
                    vec![0.0, f32::from_bits(target_rows as u32)],
                    idx,
                    vec![0.0, f32::from_bits(nb_used as u32)],
                    vec![f32::from_bits((ctx_total_fi - (nb_used - 1) * page_fi) as u32)],
                )
            } else {
                (Vec::new(), Vec::new(), Vec::new(), Vec::new())
            };
        // **设备端重复惩罚历史**(2026-10-26):v1 = committed history
        // (prompt 尾部 + 已生成),不含本轮草稿。采样关/rep<=1 时写全哨兵
        // +1.0,核内跳过。两 rank 写同一份全局 id 历史(各自按 vocab_offset
        // 查本半表),图槽内容不触发重捕获。
        let (rep_hist_v, rep_penalty_v) = if vg.has_input("rep_hist") {
            let act = self.turns.active()?;
            let rep = if sampler_on { act.sampler.rep_penalty } else { 1.0 };
            let hist = if rep > 1.0 {
                crate::sampler::penalty_history(&act.prompt_ids, &act.out)
            } else {
                vec![crate::sampler::REP_HISTORY_SENTINEL; crate::sampler::REP_HISTORY_WINDOW]
            };
            (
                hist.into_iter().map(f32::from_bits).collect::<Vec<_>>(),
                vec![rep],
            )
        } else {
            (Vec::new(), Vec::new())
        };
        let gdn_slot_v1 = [gdn_slot as f32];
        let gdn_cu_v = [0.0f32, target_rows as f32];
        let mut ins: Vec<(&str, &[f32])> = vec![
            ("ids", ids_f.as_slice()),
            ("pos", pos_f.as_slice()),
            ("kv_slots", slots_f.as_slice()),
            ("kv_lens", lens_f.as_slice()),
            ("gdn_slot", gdn_slot_v1.as_slice()),
            ("gdn_cu", gdn_cu_v.as_slice()),
            ("bt8", bt8_f.as_slice()),
            ("xqa_seq", xqa_seq_v.as_slice()),
            ("xqa_sem", xqa_sem_v.as_slice()),
            // XQA 页表:会话真链(槽在 verify 图无条件声明;
            // 不写 = 留着恒等链 = 读到别家 KV —— §7.6)
            ("xqa_pt", xqa_pt_v),
        ];
        // 设备端重复惩罚槽只随词表切存在;未切档不写(无消费者)。
        if vg.has_input("rep_hist") {
            ins.push(("rep_hist", rep_hist_v.as_slice()));
            ins.push(("rep_penalty", rep_penalty_v.as_slice()));
        }
        if !fi_q_cu_v.is_empty() {
            ins.push(("fi_q_cu", fi_q_cu_v.as_slice()));
            ins.push(("fi_indices", fi_idx_v.as_slice()));
            ins.push(("fi_indptr", fi_indptr_v.as_slice()));
            ins.push(("fi_last_len", fi_ll_v.as_slice()));
        }
        // **FI plan 表同步(W4)**:单图档(`fi_plan_bake`)的 verify 图里 FI 核
        // 不带自携表 ⇒ 每轮必须由图外刷**当轮精确表**(t=depth+1,ctx=本轮总长)。
        // 与 decode 同律:表必须与运行期 chunk 数一一对应(T=8 过供给会错)。
        // 门 = **档位在场** + 槽在场（W5：verify 恒随 `fi_plan_bake`；槽只证
        // "本图带 FI 面"，档位证"表刷新面有人消费"）。
        // ★ **双 rank 镜像**(P2.1-c,2026-10-19):verify 图含 AR(T=depth+1 行的
        // RowParallel)⇒ 两 rank 必须同跑(否则 AR 失步;`dumb` 档实测同样挂 ⇒
        // 与草稿无关,卡点就在这一句)。单卡档 = 原路(零分支)。
        // FI 表两 rank 各自的图都要刷 ⇒ 规格传进 `verify_dev`。
        let fi_spec = if self.boot.cfg.knobs.fi_plan_bake.is_some() && fi_slots_on {
            Some(self.dev.pool.fi_plan_spec(target_rows, ctx_total_fi))
        } else {
            None
        };
        let t_step_v = std::time::Instant::now();
        owl_shared::log_debug!("spec-round", "verify 镜像:ins={} peer={} fi={}", ins.len(), self.peer.is_some(), fi_spec.is_some());
        // P1.1 词表切:切档 ⇒ 两 rank 各出**半表行**、rank0 host 拼接成全表
        // （实效判据：本 rank logits 宽 < 全量；tied 档装载期已响亮拒）
        let head_split = self.head_split_active();
        let half_vocab = self.dev.boot.model.logits_vocab();
        // **分布式 argmax 档**(2026-10-26):词表切 + 贪心 ⇒ 两 rank 各出
        // (max 值, 索引),host 只比较 2 个数/行;不再回读 4MB 半表 logits。
        // 采样档保留 logits 路径(host 采样需要完整分布)。
        let dist_argmax = head_split && !sampler_on;
        // 分布式采样(候选式):graph 有 cand 槽时不再读 logits。
        let dist_sample = want_cand;
        let (tok_f, row_logits, cand_rows) = {
            let (dev, peer) = (&mut self.dev, self.peer.as_mut());
            match peer {
                None => {
                    let (tok, _tokv, _cand, lg) =
                        dev.verify_dev(&ins, fi_spec, sampler_on, depth, 0, false, false).await?;
                    (tok, lg, None)
                }
                Some(p) => {
                    // 两半都校验(纪律:`join2` 等另一半到底,吞错 = 挂住)
                    // 切档 ⇒ 两 rank 都要 logits 半表(rank0 拼接用);未切档
                    // = 原路(peer 不读 ⇒ 省那份 D2H 与 host 内存)。
                    let (r0, r1) = crate::exec::join2(
                        dev.verify_dev(
                            &ins,
                            fi_spec,
                            (sampler_on || head_split) && !dist_argmax && !dist_sample,
                            depth,
                            0,
                            dist_argmax,
                            dist_sample,
                        ),
                        p.verify_dev(
                            &ins,
                            fi_spec,
                            head_split && !dist_argmax && !dist_sample,
                            depth,
                            1,
                            dist_argmax,
                            dist_sample,
                        ),
                    )
                    .await;
                    let (_, tv1, cd1, lg1) = r1?;
                    let (tok0, tv0, cd0, lg0) = r0?;
                    if dist_argmax {
                        let (a, b) = match (tv0, tv1) {
                            (Some(a), Some(b)) => (a, b),
                            _ => {
                                return Err(ModelError::EngineInvariant {
                                    what: "词表切 verify:分布式 argmax 的 (值,索引) 行缺席"
                                        .into(),
                                })
                            }
                        };
                        (dist_argmax_merge(&a, &b, half_vocab)?, None, None)
                    } else if dist_sample {
                        // **候选式采样**(2026-10-26):两 rank 各出 top-32,
                        // host 合并成全局候选(≤64),不再读 2MB 半表 logits。
                        let (cd0, cd1) = match (cd0, cd1) {
                            (Some(a), Some(b)) => (a, b),
                            _ => {
                                return Err(ModelError::EngineInvariant {
                                    what: "词表切 verify:分布式采样候选行缺席".into(),
                                })
                            }
                        };
                        (
                            tok0,
                            None,
                            Some(merge_sample_cands(&cd0, &cd1, block.len(), half_vocab)?),
                        )
                    } else if head_split {
                        let (a, b) = match (lg0, lg1) {
                            (Some(a), Some(b)) => (a, b),
                            _ => {
                                return Err(ModelError::EngineInvariant {
                                    what: "词表切 verify:半表行缺席(装配/读面不一致)"
                                        .into(),
                                })
                            }
                        };
                        (
                            tok0,
                            Some(crate::running::concat_head_rows(&a, &b, half_vocab)?),
                            None,
                        )
                    } else {
                        (tok0, lg0, None)
                    }
                }
            }
        };
        owl_shared::log_debug!("spec-round", "verify 完成:tok={}", tok_f.len());
        let t_read_v = std::time::Instant::now();
        let t_tok = std::time::Instant::now();
        // 后续（块句柄 / 探针）继续用主 rank 的图
        let vg = self.dev.graphs.verify_mut(depth)?;
        let t_lg = std::time::Instant::now();
        // E5 性能:verify 分相(step=上传+launch;read=同步+回读;
        // pure=同步后裸 replay+read = 图独占 GPU 时间)
        if self.boot.probes.step_profile {
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
        // P1.1:切档的设备 token 只是**半表局部** argmax ⇒ 从拼接后的全表行
        // 重算全局 argmax(贪心接受判据的唯一依据);未切档 = 原值。
        let ids: Vec<u32> = if head_split && row_logits.is_some() {
            let lg = row_logits.as_ref().ok_or_else(|| ModelError::EngineInvariant {
                what: "词表切 verify:全表行 logits 缺席(greedy 判据无据)".into(),
            })?;
            let v = self.dev.boot.model.vocab_size();
            if lg.len() < block.len() * v {
                return Err(ModelError::EngineInvariant {
                    what: format!(
                        "词表切 verify:行 logits 尺寸不符 {} < {}×{}",
                        lg.len(),
                        block.len(),
                        v
                    ),
                });
            }
            (0..block.len())
                .map(|r| {
                    lg[r * v..(r + 1) * v]
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
                .collect()
        } else {
            tok_f.iter().map(|&v| v as u32).collect()
        };
        let hidden = vg.output_block("hid").ok_or_else(|| ModelError::EngineInvariant { what: "verify 图输出槽 hid 缺席(图装配与声明不一致)".into() })?;
        let n_rec = self.dev.pool.gdn_count() * owl_shared::config::SPEC_FOLD_RECORDS_PER_LAYER;
        let records: Vec<Bytes> = (0..n_rec)
            .map(|i| {
                vg.output_block(&format!("r{i}")).ok_or_else(|| ModelError::EngineInvariant {
                    what: format!("fold 记录槽 r{i} 缺席(图装配与声明不一致)"),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        verify_elapsed = ts.elapsed();
        (ids, hidden, records, false, row_logits, cand_rows) // 图输出槽:持久,不 free
    } else {
        if view_tables.is_some() {
            return Err(ModelError::Msg(
                "spec verify:eager 路径不支持逐行视图页表(select_partial + 有洞);\
                 请开 verify 图(不要 [graph] no_graph)"
                    .into(),
            ));
        }
        let ts = std::time::Instant::now();
        let (ids, hidden, row_logits) =
            self.verify_forward(&block, pos, gdn_slot, sampler_on, vlm_delta).await?;
        verify_elapsed = ts.elapsed();
        (ids, hidden, Vec::new(), true, row_logits, None)
        };
        Ok((ids, hidden, records, hidden_owned, row_logits, cand_rows, verify_elapsed))
    }

    /// **① 轮首快照**(S6d):GDN 态回滚缓冲;仅失效时拍。
    ///
    /// ★ **双 rank 镜像**(P2.1-c,2026-10-19):快照缓冲是**每 rank 各自的池**
    /// ⇒ 两边都要拍(只拍 rank0 ⇒ peer 恢复的是陈旧/未初始化态 ⇒ 重放
    /// logits 错 ⇒ m=0 回滚路输出与裸 decode 不一致)。单卡 = 原路。
    ///
    /// 返回本次拍照耗时(已有效 = ZERO)。
    async fn ensure_spec_snap(&mut self, gdn_slot: usize) -> Result<std::time::Duration> {
        if self.tspec()?.spec_snap_valid {
            return Ok(std::time::Duration::ZERO);
        }
        let ts = std::time::Instant::now();
        {
            let (dev, peer) = (&mut self.dev, self.peer.as_mut());
            crate::exec::mirror2!(dev, peer,
                d => d.pool.capture_spec_snap(d.graphs.session.face_mut(), gdn_slot),
                p => p.pool.capture_spec_snap(p.graphs.session.face_mut(), gdn_slot))?;
        }
        self.tspec()?.spec_snap_valid = true;
        Ok(ts.elapsed())
    }

    /// **② 取本轮草稿**(S6d):轮末 propose 的产物优先;首轮走 eager propose;
    /// Dumb 档 = anchor 重复。返回 (草稿, propose 耗时)。
    async fn acquire_drafts(
        &mut self,
        token: u32,
        pos: usize,
        depth: usize,
    ) -> Result<(Vec<u32>, std::time::Duration)> {
        let mut spent = std::time::Duration::ZERO;
        let drafts: Vec<u32> = if let Some(mut d) = self.tspec()?.spec_drafts_host.take() {
            // depth 联动截断(2026-10-09 depth=3 案):⑦ propose 恒产 7
            // 草稿,depth<7 时下轮 block=1+7 行装不进 T=depth+1 捕获图
            // (ids 8≠4 装填拒,pump 退役)。与轮首直调分支的
            // d.truncate(depth) 同款,收口在消费点单处。
            d.truncate(depth);
            d
        } else if let Some(crate::spec_runtime::Drafter::DFlash2(_)) =
            self.dev.spec_rt.drafter.as_ref()
        {
            let ts = std::time::Instant::now();
            let lookup = self.dflash_lookup_for_tail(&[])?;
            let mut d = self
                .dflash_propose(token, pos, lookup.as_ref())
                .await?;
            d.truncate(depth); // depth ≤ 7(草稿恒 7 产;T 图定形 depth+1)
            spent = ts.elapsed();
            d
        } else if self.dev.spec_rt.drafter.is_some() {
            let ts = std::time::Instant::now();
            self.propose_first(token, pos, depth).await?;
            spent = ts.elapsed();
            self.tspec()?.spec_drafts_host.take().ok_or_else(|| {
                ModelError::EngineInvariant {
                    what: "propose_first 未产出草稿(上游相位漂移)".into(),
                }
            })?
        } else {
            vec![token; depth] // Dumb:哑草稿 = anchor 重复
        };
        Ok((drafts, spent))
    }

    pub(crate) async fn write_bt(&mut self, sid: u64) -> Result<()> {
        // **视图态:不写全量 bt**(2026-10-23)。部分驻留档的链上有宿主占位符;
        // decode 走 `sel_pt`、verify 走逐行 `bt8`,全量 `bt` 不是权威。
        // 之前 spec 轮 `grew`/DFlash2 propose 仍无条件写 ⇒ 撞占位符守卫。
        if self.sel.has_view() && self.dev.graphs.session.has_input("sel_pt") {
            return Ok(());
        }
        let table = self.bank.sessions.session(sid)?.block_table.clone();
        // ★ **双 rank 镜像**(P2.1-c,2026-10-19):页表是**每 rank 各自**的设备块 ⇒
        // TP 档必须两边都写(否则 peer 的寻址停代)。单卡档 = 原路(零分支)。
        match self.peer.as_mut() {
            None => {
                let face = self.dev.graphs.session.face_mut();
                self.dev.pool.write_bt(face, &table).await
            }
            Some(p) => {
                let (r0, r1) = crate::exec::join2(
                    self.dev.pool.write_bt(self.dev.graphs.session.face_mut(), &table),
                    p.pool.write_bt(p.graphs.session.face_mut(), &table),
                )
                .await;
                r1?;
                r0
            }
        }
    }


    /// GDN 快照拍摄(E2c;块边界跨越时调用):会话格状态 → 槽位缓冲。
    ///
    /// **key = 边界块的内容身份**(逐块父链哈希;`owl_store::ChainCursor`)
    /// —— **不是**物理块号:物理块号是会被回收复用的槽号(`decref` 归零 →
    /// 自由队列 → 下次分配即拿回),拿它当键 ⇒ 槽复用后 `has_snap(旧槽号)`
    /// 命中**别人的状态** = 静默错值(定谳见 store 设计卷 §9.1)。
    /// 边界 token 段来源 = 本 turn 的完整 token 链(`prompt_ids ++ out`;
    /// prefill 期落在 prompt 内,decode 期跨到生成段)—— `fed` 单调不减。
    /// 槽位选择/LRU 覆盖写在 StatePool::capture_snap(state.rs)。
    /// **块边界事件**(唯一入口;一次算身份、三件事共用)。
    ///
    /// 三件(2026-10-19 解耦;原先全被 `snap_max == 0` 一起门掉 —— **语义错**):
    /// ① **链推进 + 逐块身份哈希**(总是;身份是索引与缓存树的共同键);
    /// ② **增量登记进缓存树**(`select_partial`;不登记 ⇒ 块永远 refs==1 ⇒
    ///    结构性不可沉,超卖档直接失效);
    /// ③ **索引采集**(`index_probe`;打分的前置件);
    /// ④ **GDN 快照拍摄**(仅 `pool.snap_capacity() > 0`;前缀复用面)。
    ///
    /// 为什么必须解耦:①③ 是**超卖机制的必需品**,② 是下沉的必需品 ——
    /// 它们与"要不要拍快照"无关。旧实现把四件塞在一个 `snap_max == 0 ⇒ return`
    /// 之后:关快照面 ⇒ **索引不采 + 块不入树 ⇒ 静默不沉**(实测 `snap_max = 0`
    /// 档:下沉 0 块 / 不可沉 1680 块;prefill 假快 942 tok/s)。
    pub(crate) async fn capture_gdn_snap(
        &mut self,
        sid: u64,
        fed: usize,
        snap_keep: bool,
        index_now: bool,
    ) -> Result<()> {
        // **前缀段**(prefill chunk 边界)专用:prompt 页边界有跨会话共享价值
        // ⇒ Durable(可下沉宿主)。生成段边界另走
        // [`Self::capture_gdn_snap_tail`] 的 Transient 档。
        //
        // `snap_keep` = 本边界**是否值得留快照**(2026-10-20 节奏化):
        // 链推进 / 增量登记 / 索引采集**每边界必做**(它们不是为了复用),
        // 只有"拍快照 + 槽满 DtoH 下沉(159 MB ≈ 31 ms)"这一半按节奏做 ——
        // 冷长 prefill 每 512/1024 tok 都拍 = 纯开销;复用只需要**末边界** +
        // 稀疏锚点(见 `phases.rs` 的 `SNAP_EVERY`)。
        self.capture_gdn_snap_tail(
            sid,
            fed,
            &[],
            owl_store::SnapDurability::Durable,
            snap_keep,
            index_now,
        )
        .await
    }

    /// 同上,但 `tail` = 最后几个**已提交尚未入 `act.out`** 的 token
    /// (spec 轮内边界快照:提交在 emit 之前,`out` 还差本轮那几枚;
    /// 位置 = `out` 尾的下一枚起 —— `out` 不变式:`out.len() = fed - prompt_len + 1`)、
    /// `durability` = 重要性档(生成段逐页边界 = [`Transient`](owl_store::SnapDurability::Transient):
    /// 链一往前就被取代 ⇒ 逐出即丢,不下沉宿主)。
    pub(crate) async fn capture_gdn_snap_tail(
        &mut self,
        sid: u64,
        fed: usize,
        tail: &[u32],
        durability: owl_store::SnapDurability,
        snap_keep: bool,
        index_now: bool,
    ) -> Result<()> {
        // 快照面开关(只门住第 ④ 步;前三步照走)。
        //
        // ⚠️ 2026-10-19 **两道门合并**(两条线各自加的,合入时叠加,都留):
        // ① **消费方门**(perf 线):快照的唯一消费者是**前缀复用**
        //    (`reuse_face` 要求边界快照在场;`restore_snap` 只在 prefix 命中时调用)。
        //    `prefix_cache = false` ⇒ 缓存区份额为 0 ⇒ 复用面结构上恒为 0
        //    ⇒ 每块 152MiB 的 GDN 快照(= 192 片 copy_batch,槽满还要 DtoH 下沉)
        //    是**纯浪费**,实测 67ms/块(8k 请求 ≈ 1.0s ≈ prefill 的 17%:
        //    `prefill.bnd` 3039ms → 0)。
        // ② **容量门**(本线):`snap_max = 0` ⇒ 池结构上无槽(调用方已先行跳过)。
        let snap_on = snap_keep && self.dev.pool.snap_capacity() > 0 && self.boot.cfg.knobs.prefix_cache;
        let page = self.dev.pool.geom.page;
        let full_blocks = fed / page;
        if page == 0 || full_blocks == 0 {
            return Ok(()); // 边界 0 = 无已完整块,无状态可拍
        }
        // 1) 纯读游标位置(借用即散)
        let start = self.bank.sessions.get(sid).map(|s| s.chain.blocks()).unwrap_or(0);
        if self.boot.cfg.knobs.probes.debug {
            owl_shared::log_debug!(
                "dbg snap",
                "fed={fed} full_blocks={full_blocks} start={start} snap_on={snap_on} cap={} used={}",
                self.dev.pool.snap_capacity(),
                self.dev.pool.snap_used()
            );
        }
        if start >= full_blocks {
            return Ok(()); // 幂等:同一 fed 不重复拍(状态由同一 token 链唯一确定)
        }
        let t_diag = std::time::Instant::now();
        // 2) 取新增整块的 token 段(活跃 turn 的链面;轮内调用带 `tail`)
        let seg = self.tokens_upto_with(start * page, full_blocks * page, tail)?;
        // 3) 推进身份链(段长不符 = 结构性拒,不静默;逐块哈希在同一处重放
        //    —— 身份与游标同源 `owl_store::hash_block`)
        let (gdn_slot, key, per_block, parent0) = {
            let s = self
                .bank
                .sessions
                .get_mut(sid)
                .ok_or_else(|| ModelError::Msg(format!("会话 {sid} 不在账")))?;
            let parent0 = s.chain.tail();
            let tail = s.chain.advance(page, &seg, full_blocks).ok_or_else(|| {
                ModelError::Msg(format!(
                    "链哈希游标推进失败:sid={sid} fed={fed} start={start} 新增 {} 块",
                    full_blocks.saturating_sub(start)
                ))
            })?;
            // **B-0 索引面:本事件新增的**每一个**整块身份**(2026-10-18 修:
            // 原先只登记 `full_blocks-1` 一块 —— chunk 512 时每次事件跨 16 块,
            // 洋 15 块永不进索引(实测 112 块只有 7 块在场,覆盖率曲线被打崩)。
            // 游标只留链尾 ⇒ 逐块哈希在此重放一小段;重放尾必与 advance 同值
            // (dev 档硬断;身份只有一份实现)。
            let mut per = Vec::with_capacity(full_blocks - start);
            let mut parent = parent0;
            for b in start..full_blocks {
                let lo = (b - start) * page;
                parent = owl_store::hash_block(parent, &seg[lo..lo + page]);
                per.push((parent, b));
            }
            debug_assert_eq!(parent, tail, "逐块重放尾哈希须与 advance 一致(身份单源)");
            // S2b:快照拍摄用的是**本 turn 的行**(turn 域资源)
            (
                self.turns.active().map(|a| a.gdn_slot).unwrap_or(0),
                owl_store::SnapKey::at_boundary(tail),
                per,
                parent0, // = 本次推进前的链尾(增量登记起点)
            )
        };
        // **S2(§25.2)**:槽位选择 + 冷快照下沉 —— 快照账的真源是 store 的
        // `StateStore`;设备池只做槽位索引。槽满时先**下沉最冷的一条**
        // (内容留宿主),沉不下(宿主满/关)才丢(响亮打点)。
        if snap_on {
            let slot = self.snap_ensure_slot(key).await?;
            // 两 rank 各拍各自的池(镜像分配 ⇒ slot 号同构);
            // store 的 state 账是 **host 单份** ⇒ 身份只登记一次(内容相同)。
            {
                let (dev, peer) = (&mut self.dev, self.peer.as_mut());
                crate::exec::mirror2!(dev, peer,
                    d => d.snap_dev(gdn_slot, key, slot),
                    p => p.snap_dev(gdn_slot, key, slot))?;
            }
            if self.boot.cfg.knobs.probes.debug {
                owl_shared::log_debug!(
                    "dbg snap",
                    "捕获 fed={fed} key={key:?} slot={slot} ⇒ used={}",
                    self.dev.pool.snap_used()
                );
            }
            // 登记/触达(层 = 设备;档 = 调用方给的事实)
            // **会话感知**(2026-10-19):记归属与边界深度 —— 淘汰时同会话
            // **末端(深边界)先走**,保留链头前缀(否则头部被逐 ⇒ 后面全垃圾)。
            self.bank.rel.state.register_snap_owned(
                key,
                self.dev.pool.snap_bytes(),
                owl_store::Tier::Gpu,
                durability,
                sid,
                full_blocks,
            );
        }
        let snap_ms = t_diag.elapsed().as_secs_f64() * 1000.0;
        // **B 轴 B-0:同一处顺手采索引**(同一"块边界 + 链尾身份"事件 ⇒
        // 一次算身份、一次推进游标;索引与 GDN 快照共享边界身份)。
        // 逐块登记:物理页由**会话块表**定(池可被多会话共享;旧实现拿
        // 逻辑块号当物理页号 = 单会话且顺序分配才对)。
        // 会话块表快照(两个块共用:增量登记要真槽、索引采集要物理页)
        let bt_now: Vec<u32> = self
            .bank
            .sessions
            .get(sid)
            .map(|s| s.block_table.clone())
            .unwrap_or_default();
        // ── **B-3e-2 第 2b 项:轮内增量登记**(`select_partial` 档)──
        // 宿主层按**缓存树条目**键控 ⇒ 不登记就沉不了(refs == 1;硬沉会造自由槽)。
        // `parent0` = 本次推进前的链尾(= 上次登记的尾,逐块父链连续)⇒
        // 每块一次哈希,全 turn O(n)。只登记**真槽**块(占位块本来就在树里)。
        if self.boot.cfg.knobs.select_partial {
            let bt = bt_now.clone();
            let mut blk_slices: Vec<(&[u32], u32)> = Vec::new();
            for (_, blk) in per_block.iter() {
                let slot = bt.get(*blk).copied().unwrap_or(u32::MAX);
                if owl_store::is_host_placeholder(slot) {
                    continue;
                }
                let lo = (*blk - start) * page;
                blk_slices.push((&seg[lo..lo + page], slot));
            }
            if !blk_slices.is_empty() {
                let up = self.bank.rel.leg.register_chain_inc(parent0, &blk_slices)?;
                self.bank.rel.leg.set_owner(sid, &up.inserted);
                mcnt("store.chain.inc.blocks", up.inserted.len() as u64);
                mcnt("store.chain.inc.evicted", up.evicted.len() as u64);
                // **A1 取证**(`OWL_RECON_DBG=1`):注册口三数 —— 送入 / 入树 / 被逐。
                // 入树为 0 就是"块永远 refs==1 ⇒ 不可沉"的直接原因。
                if self.boot.probes.recon_dbg {
                    owl_shared::log_debug!(
                        "chain-inc",
                        "送 {} 片 · **入树 {}** · 被逐 {} · parent0={:?}",
                        blk_slices.len(),
                        up.inserted.len(),
                        up.evicted.len(),
                        parent0
                    );
                }
            } else if self.boot.probes.recon_dbg {
                owl_shared::log_debug!("chain-inc", "空:per_block={} 全被跳过", per_block.len());
            }
        }
        let reg_ms = t_diag.elapsed().as_secs_f64() * 1000.0;
        // **B 轴 B-0:索引采集**(同一"块边界"事件;物理页由会话块表定)
        if self.boot.cfg.knobs.index_probe {
            let mut todo: Vec<(owl_store::ChainHash, usize, usize)> =
                Vec::with_capacity(per_block.len());
            for (hash, blk) in &per_block {
                let Some(phys) = bt_now.get(*blk).copied() else {
                    return Err(ModelError::Msg(format!(
                        "索引采集:逻辑块 {blk} 无物理页(块表 {} 项)",
                        bt_now.len()
                    )));
                };
                // **占位块跳过**(部分驻留档:内容在宿主 ⇒ 没有设备页可回读)。
                // 索引是**内容身份**键控的 ⇒ 该块在沉走之前就已采过(幂等早退也会挡),
                // 拿占位符当页号算 dtoh 偏移 = 越界读(27B Stage2 实测踩出)。
                if owl_store::is_host_placeholder(phys) {
                    mcnt("store.index.skip.host", 1);
                    continue;
                }
                todo.push((*hash, *blk, phys as usize));
            }
            if index_now {
                // 完整召回前一次补齐：把廉价边界攒下的项一起送进一次 D2H。
                let mut all = self.sel.take_index_pending();
                all.append(&mut todo);
                self.collect_index_for_blocks(&all).await?;
            } else {
                // 廉价边界：只记身份/页号，不触发 D2H（块在调和时受保护）。
                self.sel.push_index_pending(todo);
            }
        }
        if self.boot.probes.vram_dbg {
            let idx_ms = t_diag.elapsed().as_secs_f64() * 1000.0;
            owl_shared::log_debug!(
                "snap",
                "fed={fed} 快照段 {snap_ms:.0}ms(含链/槽/拷贝/登记)· 增量登记 {:.0}ms · 索引 {:.0}ms",
                reg_ms - snap_ms,
                idx_ms - reg_ms
            );
        }
        Ok(())
    }

    /// **B-0 索引采集（GPU 版；2026-10-23）**。
    ///
    /// 池内 K **不回读**：逐块 × 逐层发射 `owl_mean_k_block_{f16,fp8}`，
    /// 核内完成逆 RoPE + page 个 token 的 Mean-K 归约；D2H 只有
    /// `[blocks][layers][hkv][hd]` f16（27B 几何 = 32 KiB/块）。
    ///
    /// 辅助面（物理页号 + cos/sin）打成一个 U8 块一次上传，再用
    /// `BlockSlice` 切给核；输出块最后登记为 `out`（槽序契约 4）。
    async fn collect_index_gpu(
        &mut self,
        todo: &[&(owl_store::ChainHash, usize, usize)],
        dim: usize,
        layers: usize,
        kv_heads: usize,
    ) -> Result<()> {
        let page = self.dev.pool.geom.page;
        let (hkv, hd) = (self.dev.pool.geom.dims.hkv, self.dev.pool.geom.dims.hd);
        let half = self.dev.boot.rope.rotary_dim() / 2;
        if half == 0 || 2 * half > hd {
            return Err(ModelError::Msg(format!(
                "Mean-K 索引：rotary half={half} 与 head_dim={hd} 不自洽"
            )));
        }
        let n = todo.len();
        let fp8 = self.dev.pool.geom.kv_fp8;
        let entry = if fp8 { "owl_mean_k_block_fp8" } else { "owl_mean_k_block_f16" };
        // 池格式与索引几何必须同口径（f16 走 `__half`，bf16 尚未建臂）
        let per_block_elems = layers * kv_heads * dim;
        let out_elems = n * per_block_elems;
        let k_dtype = if fp8 { Dtype::U32 } else { Dtype::F16 };
        let k_layers: Vec<(u64, Dtype)> = self
            .dev
            .pool
            .kvs()
            .iter()
            .take(layers)
            .map(|kv| {
                let dt = kv.k_cache.0.dtype;
                (kv.k_cache.0.id, dt)
            })
            .collect();
        if k_layers.len() != layers {
            return Err(ModelError::Msg(format!(
                "Mean-K 索引：池层数 {} ≠ 索引层数 {layers}",
                k_layers.len()
            )));
        }
        for (id, dt) in &k_layers {
            if *dt != k_dtype {
                return Err(ModelError::Msg(format!(
                    "Mean-K 索引：层块 id={id} dtype={dt:?}，期望 {k_dtype:?}"
                )));
            }
        }
        let t_index = std::time::Instant::now();
        // 辅助块 = phys[n] u32 + cos[n*page*half] f32 + sin[n*page*half] f32
        let phys_bytes = n * 4;
        let rope_elems = n * page * half;
        let cos_off = phys_bytes;
        let sin_off = cos_off + rope_elems * 4;
        let aux_len = sin_off + rope_elems * 4;
        let mut aux = Vec::with_capacity(aux_len);
        for (_, _, phys) in todo {
            aux.extend_from_slice(&(*phys as u32).to_le_bytes());
        }
        let mut sin = Vec::with_capacity(rope_elems * 4);
        for (_, blk, _) in todo {
            for t in 0..page {
                let pos = blk * page + t;
                let Some((c, s)) = self.dev.boot.rope.cos_sin_at(pos) else {
                    return Err(ModelError::Msg(format!(
                        "Mean-K 索引：rope 表不含位置 {pos}（表长 {}）",
                        self.dev.boot.rope.max_pos()
                    )));
                };
                for v in c {
                    aux.extend_from_slice(&v.to_le_bytes());
                }
                for v in s {
                    sin.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
        aux.extend_from_slice(&sin);
        let t_upload = std::time::Instant::now();
        let face = self.dev.graphs.session.face_mut();
        let aux_b = face
            .htod(Dtype::U8, &vec![aux_len], &aux, "kvmem.index.aux")
            .await?;
        let out_b = match face.alloc_uninit(Dtype::F16, out_elems, "kvmem.index.mean_k").await {
            Ok(b) => b,
            Err(e) => {
                if let Err(fe) = face.free(&[aux_b.id]).await {
                    return Err(ModelError::Msg(format!(
                        "Mean-K 索引：输出分配失败 {e}；辅助块释放失败 {fe}"
                    )));
                }
                return Err(e);
            }
        };
        let t_launch = std::time::Instant::now();
        let src = owl_models::kernel::source(entry).to_string();
        let launch_res: Result<()> = async {
            for (layer, (id, dt)) in k_layers.iter().enumerate() {
                let msg = owl_iface::contract::LaunchMsg {
                    kernel: owl_iface::contract::KernelSource {
                        name: entry.to_string(),
                        source: src.clone(),
                    },
                    args: vec![
                        owl_iface::contract::Arg::block(*id, *dt, 1),
                        owl_iface::contract::Arg::block_slice(
                            aux_b.id, cos_off as u64, rope_elems as u64, Dtype::F32, 2,
                        ),
                        owl_iface::contract::Arg::block_slice(
                            aux_b.id, sin_off as u64, rope_elems as u64, Dtype::F32, 2,
                        ),
                        owl_iface::contract::Arg::block_slice(
                            aux_b.id, 0, n as u64, Dtype::U32, 1,
                        ),
                        owl_iface::contract::Arg::I32(page as i32),
                        owl_iface::contract::Arg::I32(hkv as i32),
                        owl_iface::contract::Arg::I32(hd as i32),
                        owl_iface::contract::Arg::I32(half as i32),
                        owl_iface::contract::Arg::I32(layer as i32),
                        owl_iface::contract::Arg::I32(layers as i32),
                        owl_iface::contract::Arg::block(out_b.id, Dtype::F16, 2),
                    ],
                    grid: (n as u32, hkv as u32, 1),
                    block: (hd as u32, 1, 1),
                    shared_mem: 0,
                    out_elems,
                };
                let got = face.launch(msg).await?;
                if got.id != out_b.id {
                    return Err(ModelError::Msg(format!(
                        "Mean-K 索引：发射回执输出块 {} ≠ 预期 {}",
                        got.id, out_b.id
                    )));
                }
            }
            Ok(())
        }
        .await;
        let t_dtoh = std::time::Instant::now();
        let dtoh_res = match launch_res {
            Ok(()) => {
                let mut bytes = vec![0u8; out_elems * 2];
                face.dtoh(&out_b, &mut bytes).await.map(|()| bytes)
            }
            Err(e) => Err(e),
        };
        let free_res = face.free(&[aux_b.id, out_b.id]).await;
        let bytes = match (dtoh_res, free_res) {
            (Ok(b), Ok(())) => b,
            (Ok(_), Err(fe)) => return Err(fe),
            (Err(e), Ok(())) => return Err(e),
            (Err(e), Err(fe)) => {
                return Err(ModelError::Msg(format!(
                    "Mean-K 索引：采集失败 {e}；且 scratch 释放失败 {fe}"
                )))
            }
        };
        if bytes.len() != out_elems * 2 {
            return Err(ModelError::Msg(format!(
                "Mean-K 索引：回读 {}B ≠ {out_elems}×2B",
                bytes.len()
            )));
        }
        let t_reg = std::time::Instant::now();
        let mut off = 0usize;
        for (hash, _, _) in todo {
            let end = off + per_block_elems * 2;
            let flat: Vec<u16> = bytes[off..end]
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            self.bank.rel.index_register_flat(*hash, flat)?;
            off = end;
        }
        let el = t_index.elapsed().as_secs_f64() * 1000.0;
        self.sel.note_index(el);
        if self.boot.probes.idx_dbg {
            let upload_ms = t_launch.duration_since(t_upload).as_secs_f64() * 1000.0;
            let launch_ms = t_dtoh.duration_since(t_launch).as_secs_f64() * 1000.0;
            let dtoh_ms = t_reg.duration_since(t_dtoh).as_secs_f64() * 1000.0;
            let reg_ms = t_reg.elapsed().as_secs_f64() * 1000.0;
            owl_shared::log_debug!(
                "idx",
                "GPU 批 {} 块 / {layers} 层：总 {el:.1}ms = 上传 {upload_ms:.1} + 发射 {launch_ms:.1} + D2H {dtoh_ms:.1} + 登记 {reg_ms:.1}",
                n
            );
        }
        Ok(())
    }

    /// **B-0 索引采集(批版)**(回读形态;§8.1 路 A;2026-10-20 批化+并行):
    /// 块边界新增的 N 块**一次 D2H** + **多线程 host 解码**,再逐块登记。
    ///
    /// 旧实现逐块 `dtoh_batch` + 单线程逐元素解码/逆 RoPE:27B/GGUF chunk1024
    /// 实测 **60 ms/边界**(32 块)—— 占 prefill 的 6%,且随 chunk 线性。
    /// 本版两条都不改语义:每块的 (层,头) 累加顺序与旧版**逐位相同**(线程按块
    /// 切分,块内循环不变);D2H 片序 = (块, 层)。
    ///
    /// - `items` = `(block_hash, logical_block, phys_page)`;
    /// - 幂等:已在索引账的块跳过(重复边界只会是重拍);
    /// - 关(缺省)= 调用方不调(零开销)。
    pub(crate) async fn collect_index_for_blocks(
        &mut self,
        items: &[(owl_store::ChainHash, usize, usize)],
    ) -> Result<()> {
        if items.is_empty() {
            return Ok(());
        }
        let Some(ix_geom) = self.bank.rel.index.as_ref().map(|i| i.geometry()) else {
            return Ok(()); // 索引未注册(未开采集)
        };
        let (dim, layers, kv_heads) = ix_geom;
        let page = self.dev.pool.geom.page;
        let (hkv, hd) = (self.dev.pool.geom.dims.hkv, self.dev.pool.geom.dims.hd);
        if hd != dim || hkv != kv_heads {
            return Err(owl_iface::contract::ModelError::Msg(format!(
                "索引几何与池不符:index (dim={dim}, hkv={kv_heads}) ≠ pool (hd={hd}, hkv={hkv})"
            )));
        }
        // **池页格式从池实况取**(单源:`KvPageFormat::of_pool`)——
        // f16 / fp8 e4m3 两侧同一寻址式,只差元素宽与转换律。
        let fmt = crate::exec::kvpage::KvPageFormat::of_pool(self.dev.pool.geom.kv_fp8);
        let todo: Vec<&(owl_store::ChainHash, usize, usize)> = items
            .iter()
            .filter(|(h, _, _)| {
                !self.bank.rel.index.as_ref().is_some_and(|i| i.contains(*h))
            })
            .collect();
        if todo.is_empty() {
            return Ok(());
        }
        if self.boot.cfg.knobs.index_gpu {
            return self.collect_index_gpu(&todo, dim, layers, kv_heads).await;
        }
        let t_index = std::time::Instant::now();
        let page_bytes = fmt.page_bytes(page, hkv, hd);
        let per_block_bytes = layers * page_bytes;
        // 一条批命令读完 N 块 × layers 层的 K 页(片序 = 块, 层)
        let mut pieces: Vec<(&owl_iface::contract::Bytes, usize, usize)> =
            Vec::with_capacity(todo.len() * layers);
        for (_, _, phys) in &todo {
            for kv in self.dev.pool.kvs().iter().take(layers) {
                pieces.push((&kv.k_cache.0, *phys * page_bytes, page_bytes));
            }
        }
        let all = {
            let face = self.dev.graphs.session.face_mut();
            face.dtoh_batch(&pieces).await?
        };
        let t_host = std::time::Instant::now();
        let dtoh_ms = t_host.duration_since(t_index).as_secs_f64() * 1000.0;
        // **cos/sin 表在主机线程取好**(位置 = 逻辑块;`Rope` 句柄不跨线程共享)
        let ropes: Vec<Vec<(Vec<f32>, Vec<f32>)>> = todo
            .iter()
            .map(|(_, blk, _)| {
                (0..page)
                    .map(|t| {
                        self.dev
                            .boot
                            .rope
                            .cos_sin_at(blk * page + t)
                            .map(|(c, s)| (c.to_vec(), s.to_vec()))
                            .unwrap_or_default()
                    })
                    .collect()
            })
            .collect();
        let n_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .clamp(1, 16)
            .min(todo.len());
        // 原子游标放在 scope **外**:scoped 线程借用它到 scope 结束
        let next = std::sync::atomic::AtomicUsize::new(0);
        let results: Vec<(usize, Vec<owl_store::MeanKEntry>)> = std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(n_threads);
            for _ in 0..n_threads {
                let (next, todo, all, ropes) = (&next, &todo, &all, &ropes);
                handles.push(s.spawn(
                    move || -> Result<Vec<(usize, Vec<owl_store::MeanKEntry>)>> {
                        let mut local = Vec::new();
                        loop {
                            let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if i >= todo.len() {
                                break;
                            }
                            let kbuf = &all[i * per_block_bytes..(i + 1) * per_block_bytes];
                            let mut collector = owl_store::MeanKCollector::new(dim, layers, kv_heads)?;
                            let mut row = vec![0.0f32; hd];
                            for li in 0..layers {
                                let lb = &kbuf[li * page_bytes..(li + 1) * page_bytes];
                                for t in 0..page {
                                    let Some((cos, sin)) = ropes[i].get(t) else { continue };
                                    if cos.is_empty() {
                                        continue; // 位置超 rope 表(与旧版 `continue` 同)
                                    }
                                    for h in 0..hkv {
                                        fmt.decode_row(lb, t, h, hkv, hd, &mut row)?;
                                        collector.push_post_rope_halves(li, h, &row, cos, sin)?;
                                    }
                                }
                            }
                            local.push((i, collector.finish()?));
                        }
                        Ok(local)
                    },
                ));
            }
            let mut acc = Vec::new();
            let mut err: Option<owl_iface::contract::ModelError> = None;
            for h in handles {
                match h.join() {
                    Ok(Ok(v)) => acc.extend(v),
                    Ok(Err(e)) => {
                        err.get_or_insert(e);
                    }
                    Err(_) => {
                        err.get_or_insert(owl_iface::contract::ModelError::Msg(
                            "索引采集线程 panic(设计外)".into(),
                        ));
                    }
                }
            }
            match err {
                Some(e) => Err(e),
                None => Ok(acc),
            }
        })?;
        // 顺序登记(身份 → 条目),与旧版同序
        let mut by_idx: Vec<Option<Vec<owl_store::MeanKEntry>>> =
            (0..todo.len()).map(|_| None).collect();
        for (i, e) in results {
            by_idx[i] = Some(e);
        }
        for (i, entries) in by_idx.into_iter().enumerate() {
            let (hash, _, _) = todo[i];
            self.bank.rel.index_register(*hash, entries.ok_or_else(|| {
                owl_iface::contract::ModelError::Msg(format!("索引采集漏块:第 {i} 块无条目"))
            })?)?;
        }
        let el = t_index.elapsed().as_secs_f64() * 1000.0;
        self.sel.note_index(el);
        if self.boot.probes.idx_dbg {
            let host_ms = t_host.elapsed().as_secs_f64() * 1000.0;
            owl_shared::log_debug!(
                "idx",
                "批 {} 块 / {layers} 层:总 {el:.1}ms = D2H {dtoh_ms:.1} + host {host_ms:.1} ({n_threads} 线程)",
                todo.len()
            );
        }
        Ok(())
    }

    /// **快照下沉**(设备槽 → 宿主;一条 swap 搬 N 片)。
    ///
    /// ⚠️ 宿主梯满时由 **store 政策**先逐最冷条目(见 `plan_snap_offload`)——
    /// 逐出量在此上报(store 零日志依赖:`evicted_bytes` 差分)。
    pub(crate) async fn snap_offload(&mut self, key: owl_store::SnapKey, slot: u32) -> Result<()> {
        let ev0 = self.bank.rel.snap_host_evicted_bytes();
        let (pieces, table) = self.bank.rel.plan_snap_offload(key, slot)?;
        let table = match self
            .swap_all_ranks(owl_iface::contract::SwapDir::Out, &pieces, table)
            .await
        {
            Ok(t) => t,
            Err(e) => {
                self.bank.rel.abort_snap_offload(key);
                return Err(e);
            }
        };
        self.bank.rel.commit_snap_offload(key, table)?;
        let ev = self.bank.rel.snap_host_evicted_bytes().saturating_sub(ev0);
        if ev > 0 {
            owl_shared::metrics::with_metrics_store(|m| {
                m.counter_add("store.snap.host_evict_bytes", ev, file!(), line!());
            });
            owl_shared::log_debug!(
                "store",
                "宿主梯满 ⇒ 逐最冷快照 {ev}B(政策 = 与 KV 腿同一份 EvictPolicy;被逐边界从此不可复用)"
            );
        }
        owl_shared::metrics::with_metrics_store(|m| {
            m.counter_add("store.snap.out", 1, file!(), line!());
            m.counter_add(
                "store.snap.out.bytes",
                self.bank.rel.snap_bytes() as u64,
                file!(),
                line!(),
            );
        });
        // **大页提示可观测面**(2026-10-19):宿主块若未拿到大页,冷写回到 4KiB
        // 缺页速度(159 MiB 约 80ms vs 12ms)⇒ 不许静默(指标 + 首次点名)。
        let hint_fail = self.bank.rel.snap_host_hint_failures();
        if hint_fail > 0 {
            owl_shared::metrics::with_metrics_store(|m| {
                m.counter_add("store.host.hint_fail", hint_fail as u64, file!(), line!());
            });
            owl_shared::log_debug!(
                "store",
                "宿主块大页提示未生效 {hint_fail} 块:冷写按 4KiB 缺页(下沉会更慢;查 THP 姿态 /sys/kernel/mm/transparent_hugepage/enabled)"
            );
        }
        Ok(())
    }

    /// **快照回灌**(宿主 → 设备槽;一条 swap 搬 N 片)。
    pub(crate) async fn snap_onload(&mut self, key: owl_store::SnapKey, slot: u32) -> Result<()> {
        let (pieces, table) = self.bank.rel.plan_snap_onload(key, slot)?;
        let table = match self
            .swap_all_ranks(owl_iface::contract::SwapDir::In, &pieces, table)
            .await
        {
            Ok(t) => t,
            Err(e) => {
                // 台子随失败的 connector 消失 ⇒ 内容已丢 ⇒ 条目一并删
                self.bank.rel.abort_snap_onload(key);
                return Err(e);
            }
        };
        self.bank.rel.commit_snap_onload(key, table)?;
        self.dev.pool.snap_place(slot, key);
        // **TP:peer 的池槽索引同刻对齐**(物理槽由 swap 两 rank 同步写入;
        // 索引不镜像 ⇒ 后续逐出/复用判据在两侧漂移)
        if let Some(pr) = self.peer.as_mut() {
            pr.pool.snap_place(slot, key);
        }
        // **触达入 store 账**(§30.6-②):store 的 `last_use` 是出序输入;
        // 池的 `last_use` 只是物理面。两处同刻更新 ⇒ 判据与池不再漂。
        self.bank.rel.state.touch(key);
        owl_shared::metrics::with_metrics_store(|m| {
            m.counter_add("store.snap.in", 1, file!(), line!());
            m.counter_add(
                "store.snap.in.bytes",
                self.bank.rel.snap_bytes() as u64,
                file!(),
                line!(),
            );
        });
        Ok(())
    }

    /// **为一次快照拍摄选设备槽**（Phase 3）：池满时的受害者下沉/丢弃
    /// 全部在 `StoreHook::Acquire(State::Slot)` 内完成。
    pub(crate) async fn snap_ensure_slot(
        &mut self,
        key: owl_store::SnapKey,
    ) -> Result<u32> {
        let out = self
            .store_hook(crate::store_hook::StoreHook::state_slot(key))
            .await?;
        out.state_slot.ok_or_else(|| {
            ModelError::EngineInvariant {
                what: format!("Acquire(State::Slot) 未返回快照槽: {key:?}"),
            }
        })
    }

    /// **快照就位到设备**（restore 前置；§25.2）。
    ///
    /// Phase 3 起只经 `StoreHook::Acquire(State)`：Gpu 命中 / Host 回灌 /
    /// Disk→Host→Device 两跳、受害者下沉/丢弃全部在 store hook 内完成。
    pub(crate) async fn snap_ensure_resident(
        &mut self,
        key: owl_store::SnapKey,
    ) -> Result<u32> {
        let out = self
            .store_hook(crate::store_hook::StoreHook::state(key))
            .await?;
        out.state_slot.ok_or_else(|| {
            ModelError::EngineInvariant {
                what: format!("Acquire(State) 未返回快照槽: {key:?}"),
            }
        })
    }

    /// 取本 turn 已喂入 KV 的 token 段 `[lo, hi)`(= `prompt_ids ++ out`)。
    /// 越过已喂面 = **结构性拒**(快照边界不得越过 KV 真实内容 —— 静默截断
    /// 会让快照身份与实际内容脱节)。
    /// 本 turn 的 token 链区间 `[lo, hi)`(快照边界段来源;`coverage.rs` 回算
    /// 块身份同用 —— 链身份必须与索引登记同源)。
    pub(crate) fn tokens_upto(&self, lo: usize, hi: usize) -> Result<Vec<u32>> {
        self.tokens_upto_with(lo, hi, &[])
    }

    /// 同上,但 `hi` 越过 `act.out` 尾时由 `tail` 补(spec 轮内边界快照专用:
    /// 本轮已提交的 token 还在 `toks` 里,尚未进 `out`)。越界 = 响亮拒。
    pub(crate) fn tokens_upto_with(
        &self,
        lo: usize,
        hi: usize,
        tail: &[u32],
    ) -> Result<Vec<u32>> {
        let act = self.turns.active().map_err(|_| {
            ModelError::Msg("无活跃 turn:快照边界必须落在本 turn 的 token 链上".into())
        })?;
        let plen = act.prompt_ids.len();
        let mut seg = Vec::with_capacity(hi.saturating_sub(lo));
        for i in lo..hi {
            if i < plen {
                seg.push(act.prompt_ids[i]);
            } else {
                let j = i - plen;
                // `out` 在先,`tail` 接在 `out` 尾之后(单源位置:不乱猜)
                let t = match act.out.get(j) {
                    Some(t) => Some(*t),
                    None => j.checked_sub(act.out.len()).and_then(|k| tail.get(k)).copied(),
                };
                match t {
                    Some(t) => seg.push(t),
                    None => {
                        return Err(ModelError::Msg(format!(
                            "快照边界越界:token {i} 未喂(plen={plen} out={} tail={})",
                            act.out.len(),
                            tail.len()
                        )))
                    }
                }
            }
        }
        Ok(seg)
    }
}

impl<D: DeviceClient> crate::engine::RankDev<D> {
    /// **GDN 快照拍摄设备半**(两 rank 各拍各自的池;键 = 边界块**内容身份**)。
    ///
    /// `slot` 由 host 侧 `snap_ensure_slot` 选好(store 的 state 账是单份,
    /// 镜像分配 ⇒ 每 rank 的池同槽号);本方法只做 D2D 拍摄。
    pub(crate) async fn snap_dev(
        &mut self,
        gdn_slot: usize,
        key: owl_store::SnapKey,
        slot: u32,
    ) -> Result<()> {
        let face = self.graphs.session.face_mut();
        self.pool.capture_snap(face, gdn_slot, key, Some(slot)).await
    }
}


#[cfg(test)]
mod adaptive_tests {
    use super::*;

    #[test]
    fn downshift_only_after_warmup_and_below_threshold() {
        assert!(!adaptive_should_downshift(0, 0, 3.0));
        assert!(!adaptive_should_downshift(SPEC_ADAPTIVE_WARMUP_ROUNDS - 1, 100, 3.0));
        // DOCS 29.7k d7: 2.8 tokens/round -> 降
        assert!(adaptive_should_downshift(SPEC_ADAPTIVE_WARMUP_ROUNDS, 23, 3.0));
        // MATH 32k d7: 4.98 -> 不降
        assert!(!adaptive_should_downshift(SPEC_ADAPTIVE_WARMUP_ROUNDS, 40, 3.0));
        // 4.0 早降门
        assert!(adaptive_should_downshift(SPEC_ADAPTIVE_WARMUP_ROUNDS, 28, 4.0));
        assert!(!adaptive_should_downshift(SPEC_ADAPTIVE_WARMUP_ROUNDS, 36, 4.0));
    }
}


#[cfg(test)]
mod live_depth_tests {
    use super::live_depth_for_budget;

    #[test]
    fn cap_reserves_bonus_slot() {
        assert_eq!(live_depth_for_budget(15, 0), 0);
        assert_eq!(live_depth_for_budget(15, 1), 0);
        assert_eq!(live_depth_for_budget(15, 2), 1);
        assert_eq!(live_depth_for_budget(15, 8), 7);
        assert_eq!(live_depth_for_budget(15, 16), 15);
        assert_eq!(live_depth_for_budget(7, 100), 7);
    }
}

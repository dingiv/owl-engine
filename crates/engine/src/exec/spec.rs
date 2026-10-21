//! spec 轮机器(execute_spec_round)+ 块表写 + GDN 快照
use super::*;
use owl_models::ModelError;

/// **采样惩罚的历史窗**(tok):只回看最近这么多已发 token。
///
/// 口径 = 与裸 decode 的 `hist` 同式(逐 token 追加、窗口截尾),不是配置项:
/// 它约束的是"惩罚集有多大",过大会让长思考被无谓压制。
const SPEC_HIST_KEEP: usize = 64;

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
        let depth = self.dev.spec_rt.depth;
        // **VLM m-RoPE delta**(V5):图像提示经塔后,文本主干的 rope 位置
        // 相对真实 token 位置有常量偏移;verify 行必须与 decode 同式
        // (`pos_rope = pos + delta`),但 KV slots/lens 仍用真实位置。
        let vlm_delta = self
            .turns
            .active()
            .ok()
            .and_then(|a| a.vlm.as_ref())
            .map(|v| v.ins.delta)
            .unwrap_or(0);
        owl_shared::log_debug!("spec-round", "轮开始 token={token} pos={pos} depth={depth} vlm_delta={vlm_delta}");

        // ① 快照(轮首;恢复后仍有效 → 只在失效时拍)
        t_snap = self.ensure_spec_snap(gdn_slot).await?;

        // ② 草稿到位(host 账;轮末 propose 图/eager dtoh 产出。首轮 =
        // prefill seed propose(eager);Dumb = anchor 重复)
        let (drafts, t_drafts) = self.acquire_drafts(token, pos, depth).await?;
        t_propose += t_drafts;

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

        let (ids, hidden, records, hidden_owned, row_logits, t_v) = self
            .verify_round(&block, pos, gdn_slot, sampler_on, vlm_delta, &xqa_pt_v, sid)
            .await?;
        t_verify = t_v;
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
            let vocab = self.dev.boot.model.vocab_size();
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
            (m, s[..=m].to_vec(), s[m])
        } else {
            // **argmax/greedy 路**:直接拿 verify 行的 ids 比(drafts[i] == ids[i])
            let m = (0..depth).take_while(|&i| drafts[i] == ids[i]).count();
            (m, ids[..=m].to_vec(), ids[m])
        };
        // **AL 计数提到分支外**(2026-10-19 修):两条路(采样/argmax)都要记 ——
        // 旧形只挂在采样路 ⇒ `mode = "greedy"` 档的 AL **在仪表上恒为 1.00**
        // (实测:greedy 跑 22 轮出 64 tok = 真 AL≈2.9,而计数器读 0 ⇒ 判读被带偏;
        //  引擎行为无错,错的是仪表)。语义 = 每轮接受数;降级态机读的是
        // `spec.recent_*`(与本计数无关,不受影响)。
        mcnt("spec.sample.rounds", 1);
        mcnt("spec.sample.accepted", m as u64);
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

        owl_shared::log_debug!("spec-round", "接受完成 m={m} bonus={bonus} emitted={}", emitted.len());
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
                records.len() == self.dev.pool.gdn_count() * owl_shared::config::SPEC_VERIFY_ROWS_MAX;
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
                    self.spec_fold_to(gdn_slot, &records, b - pos).await?;
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
                self.spec_fold_to(gdn_slot, &records, m + 1).await?;
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
                    self.verify_forward(&replay, pos, gdn_slot, false, vlm_delta).await?;
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
        let d2_arc7 = match self.dev.spec_rt.drafter.as_ref() {
            Some(crate::spec_runtime::Drafter::DFlash2(d)) => Some(std::sync::Arc::clone(d)),
            _ => None,
        };
        if let Some(d2) = d2_arc7 {
            let ts = std::time::Instant::now();
            // E5-DF4:图态优先(单图一轮 encode+propose;eager 回退 =
            // OWL_DFLASH_EAGER=1 或图降级)
            if self.dev.graphs.dflash.is_some() && !self.boot.probes.dflash_eager {
                let d = self.dflash_graph_round(m, pos, toks[m]).await?;
                t_propose += ts.elapsed();
                self.tspec()?.spec_drafts_host = Some(d);
            } else {
            let noencode = self.boot.probes.dflash_noencode;
            let taps: Vec<owl_iface::contract::Bytes> = (0..self.dev.spec_rt.dflash_tap_count)
                .map(|i| {
                    self.dev.graphs
                        .verify
                        .as_ref()
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
            let d = self.dflash_propose(bonus, fp).await?;
            if self.boot.probes.dflash_probe && self.dev.spec_rt.stats.rounds < 3 {
                let toks_txt = self.boot.tok.decode(&d);
                owl_shared::log_debug!("dflash-probe", "pos={fp} bonus={bonus} drafts={d:?} txt={toks_txt:?}");
            }
            t_propose += ts.elapsed();
            self.tspec()?.spec_drafts_host = Some(d);
            }
        } else if !self.dev.graphs.propose.is_empty() && !self.boot.probes.propose_eager {
            let ts = std::time::Instant::now();
            let d = self.propose_graph_step(m, &toks, pos).await?;
            t_propose += ts.elapsed();
            self.tspec()?.spec_drafts_host = Some(d);
        } else if self.dev.spec_rt.drafter.is_some() {
            let ts = std::time::Instant::now();
            let nd = self.propose_round(&hidden, m, &toks, pos).await?;
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
                == self.dev.pool.gdn_count() * owl_shared::config::SPEC_VERIFY_ROWS_MAX
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
    ) -> Result<()> {
        debug_assert!(rows >= 1 && rows <= self.dev.spec_rt.depth, "fold 行数越界: {rows}");
        // ★ **双 rank 镜像**(P2.1-c,2026-10-19):fold 写的是**每 rank 自己的 GDN 池**
        // ⇒ 两 rank 必须同跑(否则下一轮 verify 读到本 rank 未 fold 的状态)。
        // 与 `capture_gdn_snap` 的双 rank 形态对齐(旧注释自认的缺口);单卡 = 原路。
        if !self.dev.graphs.fold.is_empty() {
            let (dev, peer) = (&mut self.dev, self.peer.as_mut());
            crate::exec::mirror2!(dev, peer,
                d => d.fold_step_dev(gdn_slot, rows),
                p => p.fold_step_dev(gdn_slot, rows))?;
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
        let nb = self.dev.pool.geom.nb.max(table.len());
        let mut v: Vec<f32> = table.iter().map(|&b| f32::from_bits(b)).collect();
        v.resize(nb, 0.0);
        v
    }

    /// **① 轮首快照**(S6d):GDN 态回滚缓冲;仅失效时拍。
    ///
    /// ★ **双 rank 镜像**(P2.1-c,2026-10-19):快照缓冲是**每 rank 各自的池**
    /// ⇒ 两边都要拍(只拍 rank0 ⇒ peer 恢复的是陈旧/未初始化态 ⇒ 重放
    /// logits 错 ⇒ m=0 回滚路输出与裸 decode 不一致)。单卡 = 原路。
    ///
    /// 返回本次拍照耗时(已有效 = ZERO)。
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
    ) -> Result<(
        Vec<u32>,
        owl_iface::contract::Bytes,
        Vec<owl_iface::contract::Bytes>,
        bool,
        Option<Vec<f32>>,
        std::time::Duration,
    )> {
        let verify_elapsed;
        let (ids, hidden, records, hidden_owned, row_logits) =
        if let Some(vg) = self.dev.graphs.verify.as_mut() {
        let ts = std::time::Instant::now();
        let page = self.dev.pool.geom.page;
        let bt_chain = {
            let s = self.bank.sessions.session(sid)?;
            s.block_table.clone()
        };
        // **部分驻留守卫**(2026-10-21,合并自 feat/gguf 08790fa):`select_partial`
        // 下沉过的块在会话链上是**宿主占位符**(u32::MAX)。verify 的 `bt8`/`slots`
        // 直接取会话链 ⇒ 占位符被 v2/attention 核当页号(乘页字节 = 天文地址)
        // ⇒ CUDA 700,且**污染整个上下文**(之后每个 turn 都炸)。响亮拒发本轮,
        // 错误上抛到 pump(弃置该 turn,不发射任何核)。
        // 为什么不做"视图化 verify":方案 A 的视图把 KV 压成选中子集,而 verify
        // 要按**原位置**逐行因果写 8 行 KV + 逐行长 lens —— 页表与 lens 的语义
        // 还没定谱(设计卷 B-3e 挂账),先拒不发假绿。
        {
            let page_g = self.dev.pool.geom.page.max(1);
            let need_pages = (pos + block.len()).div_ceil(page_g);
            if let Some((i, _)) = bt_chain
                .iter()
                .take(need_pages)
                .enumerate()
                .find(|(_, p)| **p == crate::scheduler::PENDING_LOAD_PLACEHOLDER)
            {
                return Err(ModelError::Msg(format!(
                    "spec verify 需要第 {i} 块但它在宿主(部分驻留下沉过;pos={pos} 需 {need_pages} 页)                     ⇒ 拒发本轮。verify 的视图化页表未实现:请关 `[kvmem] select_partial`                     (只留 select_decode)或关投机"
                )));
            }
        }
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
        let slots_f: Vec<f32> = (pos..pos + block.len())
            .map(|p| (bt_chain[p / page] * page as u32 + (p % page) as u32) as f32)
            .collect();
        let lens_f: Vec<f32> = ((pos + 1)..=(pos + block.len())).map(|p| p as f32).collect();
        // §三十一:bt8 = 页链 × depth1 平铺(8 伪序列共享页链;v2 臂用);
        // 补零到池 nb(会话链短于池时尾部补无效页 0,kv_lens 限界不读)
        let nb_total = self.dev.pool.geom.nb.max(bt_chain.len());
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
        let xqa_sem_v: Vec<f32> = vec![0.0; self.dev.pool.xqa_sem_n().max(1)];
        // **FI 断点臂**的逐轮值(该臂未开 ⇒ 空,不写槽):
        // ① 宿主标量槽 `ctx_total`(= 本轮 kv 总长;plan 的 key)
        // ② 表四件套(q_cu/indices/indptr/last_len;u32 位型过线)
        let page_fi = self.dev.pool.geom.page;
        let ctx_total_fi = pos + block.len();
        // 门 = **槽在场**(槽属于**图**;判"某核在场"必须看节点/槽,不看旋钮)
        let fi_slots_on = vg.has_input("fi_q_cu");
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
            ("xqa_pt", xqa_pt_v),
        ];
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
            Some(self.dev.pool.fi_plan_spec(block.len(), ctx_total_fi))
        } else {
            None
        };
        let t_step_v = std::time::Instant::now();
        owl_shared::log_debug!("spec-round", "verify 镜像:ins={} peer={} fi={}", ins.len(), self.peer.is_some(), fi_spec.is_some());
        // P1.1 词表切:切档 ⇒ 两 rank 各出**半表行**、rank0 host 拼接成全表
        // （实效判据：本 rank logits 宽 < 全量；tied 档装载期已响亮拒）
        let head_split = self.head_split_active();
        let half_vocab = self.dev.boot.model.logits_vocab();
        let (tok_f, row_logits) = {
            let (dev, peer) = (&mut self.dev, self.peer.as_mut());
            match peer {
                None => dev.verify_dev(&ins, fi_spec, sampler_on).await?,
                Some(p) => {
                    // 两半都校验(纪律:`join2` 等另一半到底,吞错 = 挂住)
                    // 切档 ⇒ 两 rank 都要 logits 半表(rank0 拼接用);未切档
                    // = 原路(peer 不读 ⇒ 省那份 D2H 与 host 内存)。
                    let (r0, r1) = crate::exec::join2(
                        dev.verify_dev(&ins, fi_spec, sampler_on || head_split),
                        p.verify_dev(&ins, fi_spec, head_split),
                    )
                    .await;
                    let (_, lg1) = r1?;
                    let (tok0, lg0) = r0?;
                    if head_split {
                        let (a, b) = match (lg0, lg1) {
                            (Some(a), Some(b)) => (a, b),
                            _ => {
                                return Err(ModelError::EngineInvariant {
                                    what: "词表切 verify:半表行缺席(装配/读面不一致)"
                                        .into(),
                                })
                            }
                        };
                        (tok0, Some(crate::running::concat_head_rows(&a, &b, half_vocab)?))
                    } else {
                        (tok0, lg0)
                    }
                }
            }
        };
        owl_shared::log_debug!("spec-round", "verify 完成:tok={}", tok_f.len());
        let t_read_v = std::time::Instant::now();
        let t_tok = std::time::Instant::now();
        // 后续（块句柄 / 探针）继续用主 rank 的图
        let vg = self
            .dev
            .graphs
            .verify
            .as_mut()
            .ok_or_else(|| ModelError::EngineInvariant {
                what: "verify 图缺席(档位与装配不一致)".into(),
            })?;
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
        let ids: Vec<u32> = if head_split {
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
        let n_rec = self.dev.pool.gdn_count() * owl_shared::config::SPEC_VERIFY_ROWS_MAX;
        let records: Vec<Bytes> = (0..n_rec)
            .map(|i| {
                vg.output_block(&format!("r{i}")).ok_or_else(|| ModelError::EngineInvariant {
                    what: format!("fold 记录槽 r{i} 缺席(图装配与声明不一致)"),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        verify_elapsed = ts.elapsed();
        (ids, hidden, records, false, row_logits) // 图输出槽:持久,不 free
    } else {
        let ts = std::time::Instant::now();
        let (ids, hidden, row_logits) =
            self.verify_forward(&block, pos, gdn_slot, sampler_on, vlm_delta).await?;
        verify_elapsed = ts.elapsed();
        (ids, hidden, Vec::new(), true, row_logits)
        };
        Ok((ids, hidden, records, hidden_owned, row_logits, verify_elapsed))
    }

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
            let mut d = self.dflash_propose(token, pos).await?;
            d.truncate(depth); // depth ≤ 7(草稿恒 7 产;T 图定形 depth+1)
            spent = ts.elapsed();
            d
        } else if self.dev.spec_rt.drafter.is_some() {
            let ts = std::time::Instant::now();
            self.propose_first(token, pos).await?;
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
                    eprintln!(
                        "[chain-inc] 送 {} 片 · **入树 {}** · 被逐 {} · parent0={:?}",
                        blk_slices.len(),
                        up.inserted.len(),
                        up.evicted.len(),
                        parent0
                    );
                }
            } else if self.boot.probes.recon_dbg {
                eprintln!("[chain-inc] 空:per_block={} 全被跳过", per_block.len());
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
            self.collect_index_for_blocks(&todo).await?;
        }
        if self.boot.probes.vram_dbg {
            let idx_ms = t_diag.elapsed().as_secs_f64() * 1000.0;
            eprintln!(
                "[snap] fed={fed} 快照段 {snap_ms:.0}ms(含链/槽/拷贝/登记)· 增量登记 {:.0}ms · 索引 {:.0}ms",
                reg_ms - snap_ms,
                idx_ms - reg_ms
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
            eprintln!(
                "[idx] 批 {} 块 / {layers} 层:总 {el:.1}ms = D2H {dtoh_ms:.1} + host {host_ms:.1} ({n_threads} 线程)",
                todo.len()
            );
        }
        Ok(())
    }

    /// **快照落设备槽**(S2 §25.2;编排:store 出计划 → `face.swap` → 记账)。
    ///
    /// - 同 key 已在设备 ⇒ 原地槽(幂等重拍);
    /// - 空槽 ⇒ 直接用;
    /// - 池满 ⇒ 最冷的设备快照**先下沉**(内容留);宿主满/关 ⇒ 丢(丢 = 状态没了,
    ///   下次该边界不可复用 —— 与旧行为同,但现在有账可查);
    /// - 池关(`snap_max = 0`)⇒ 结构性拒(调用方已先行跳过)。
    pub(crate) async fn snap_ensure_slot(&mut self, key: owl_store::SnapKey) -> Result<u32> {
        use crate::state::SnapSlotPick;
        // **判据 = store(唯一产地;§30.6-②)**:出序(事实档 + EvictPolicy)从
        // `Rel::snap_victim_order` 拿;池只做 key→槽映射。旧实现在池内自带
        // `rank + LRU` = 第二套规范(与容量调度可能各说各话)。
        let order = self.bank.rel.snap_victim_order(&[]);
        let pick = self.dev.pool.snap_pick_slot(key, &order).ok_or_else(|| {
            owl_iface::contract::ModelError::Msg(
                "GDN 快照池为空(snap_max = 0):调用方应跳过 capture".into(),
            )
        })?;
        match pick {
            SnapSlotPick::Hit(slot) | SnapSlotPick::Free(slot) => Ok(slot),
            SnapSlotPick::Victim(slot, victim) => {
                // **生成段(Transient)= 逐出即丢**:store 的不变量本就拒它下沉
                // ⇒ 试了只会换来一条“失败”WARN(把设计行为报成故障)。
                // 2026-10-19 pi 真机实测:一次会话 9 条。
                let transient = matches!(
                    self.bank.rel.state.snap_durability(victim),
                    Some(owl_store::SnapDurability::Transient)
                );
                if transient {
                    owl_shared::log_debug!(
                        "store",
                        "逐出生成段快照 {victim:?}(逐出即丢,不下沉)"
                    );
                } else {
                    match self.snap_offload(victim, slot).await {
                        Ok(()) => {
                            self.dev.pool.snap_clear_slot(slot);
                            if let Some(pr) = self.peer.as_mut() {
                                pr.pool.snap_clear_slot(slot);
                            }
                            return Ok(slot);
                        }
                        Err(e) => {
                            // 收编为结构化 WARN(2026-10-18 T1;原为 `DBG-SNAP` 直接打印)
                            owl_shared::log_warn!("store", "快照下沉失败: {e}");
                        }
                    }
                }
                // 沉不下 ⇒ 丢(内容整体消失;宿主副本同步丢)
                owl_shared::log_debug!(
                    "store",
                    "快照槽满且宿主不可用 ⇒ 丢弃最冷快照 {victim:?}(该边界将不可复用)"
                );
                self.bank.rel.drop_snap_host(victim)?;
                if let Err(e) = self.bank.rel.state.drop_snap(victim) {
                    owl_shared::log_debug!("store", "快照 {victim:?} 出账: {e}");
                }
                owl_shared::metrics::with_metrics_store(|m| {
                    m.counter_add("store.snap.drop", 1, file!(), line!());
                });
                self.dev.pool.snap_clear_slot(slot);
                if let Some(pr) = self.peer.as_mut() {
                    pr.pool.snap_clear_slot(slot);
                }
                Ok(slot)
            }
        }
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

    /// **快照就位到设备**(restore 前置;§25.2):设备层 ⇒ 原槽;宿主层 ⇒
    /// 先找槽再回灌;不在账 ⇒ 结构性拒(该边界不可复用,调用方应已裁掉)。
    pub(crate) async fn snap_ensure_resident(
        &mut self,
        key: owl_store::SnapKey,
    ) -> Result<u32> {
        match self.bank.rel.state.locate(key) {
            Some(owl_store::Tier::Gpu) => {
                let slot = self.dev.pool.snap_slot_of(key).ok_or_else(|| {
                    owl_iface::contract::ModelError::Msg(format!(
                        "快照 {key:?} 账在设备层但池里找不到槽(账目不平)"
                    ))
                })?;
                // 使用即触达(store 出序的真源)
                self.bank.rel.state.touch(key);
                Ok(slot)
            }
            Some(owl_store::Tier::Host) => {
                let slot = self.snap_ensure_slot(key).await?;
                self.snap_onload(key, slot).await?;
                Ok(slot)
            }
            Some(owl_store::Tier::Disk) => {
                // **S6**:盘 → 宿主 → 设备(两跳);读回后层账回 Host,再走 Host 分支
                self.fill_snap_from_disk(key).await?;
                let slot = self.snap_ensure_slot(key).await?;
                self.snap_onload(key, slot).await?;
                Ok(slot)
            }
            None => Err(owl_iface::contract::ModelError::Msg(format!(
                "快照 {key:?} 不在账(已被逐出)"
            ))),
        }
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

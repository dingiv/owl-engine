//! 执行器(设备侧照办;2026-10-01 拆分自 engine.rs §3)。
//!
//! S0 决策/执行分离的**执行半**:调度面(scheduler.rs)产出
//! SchedulerOutput,本模块无决策权 —— execute_begin(GDN 重置/快照
//! 恢复/块表重写)、execute_prefill(块式预填 + 边界快照)、
//! execute_decode(图步进 + 设备采样回读)、采样收口
//! (sample_and_emit/complete)+ prefill_chunk(树根装配与执行)。
//! 状态块触碰一律经 StatePool(state.rs),不直摸块句柄。

use owl_iface::contract::{Bytes, DeviceClient, Dtype, ModelError};
use owl_models::interpreters::eval_ops_scoped_env;
use owl_models::layers::gdn::GdnBuffers;
use owl_models::module::{FiPrefillCtx, ForwardCtx, KvBuffers, Module};
use owl_models::tokenizer::Tokenizer;
use owl_models::TensorOps;

use crate::graph_plan::f32b;
use crate::scheduler::BeginPlan;
use crate::turn::TurnEvent;


/// E5-M5 spec 分相探针(owl-metrics 直用 API:release 恒开,热路径每轮
/// ~8 条环形推入 ≈ µs 级;/debug/metrics 查询。宏是 debug-only,装载域
/// 「直用 API」双轨纪律的 server 侧复刻)
fn mrec(tag: &str, dur: std::time::Duration) {
    owl_shared::metrics::with_metrics_store(|s| {
        s.timer_record_tag(tag, dur, file!(), line!());
    });
}
fn mcnt(tag: &str, n: u64) {
    owl_shared::metrics::with_metrics_store(|s| s.counter_add(tag, n, file!(), line!()));
}

type Result<T> = std::result::Result<T, ModelError>;

impl<D: DeviceClient> crate::running::RunningEngine<D> {
    /// turn 启动执行(设备侧照办):GDN 格重置 / 边界快照恢复 + 持久
    /// 块表重写(write_bt;图烘焙 bt 指针,内容重写指针稳定图不重捕)
    pub(crate) async fn execute_begin(&mut self, b: BeginPlan) -> Result<()> {
        if b.reset_gdn {
            let face = self.session.face_mut();
            self.pool.reset_gdn(face, b.gdn_slot).await?;
        } else if let Some(key) = b.restore_key {
            // 前缀命中:恢复边界快照到会话格(免重灌;GDN 态对齐)
            let face = self.session.face_mut();
            self.pool.restore_snap(face, b.gdn_slot, key).await?;
        }
        self.write_bt(b.session_id).await
    }

    /// prefill chunk 执行(设备侧;块已在决策面切好)
    pub(crate) async fn execute_prefill(
        &mut self,
        chunk_ids: Vec<u32>,
        base: usize,
        is_last: bool,
    ) -> Result<TurnEvent> {
        let (turn_id, sid) = {
            let act = self.active.as_ref().expect("活跃");
            (act.id, act.session_id)
        };
        let last_row = self.prefill_chunk(&chunk_ids, base, is_last).await?;
        // E2c:块边界跨越 → GDN 快照拍摄(可复用身份 = 链尾块)
        let new_fed = base + chunk_ids.len();
        if self.pool.paged && new_fed % self.pool.page == 0 {
            self.capture_gdn_snap(sid, new_fed).await?;
        }
        self.active.as_mut().expect("活跃").fed += chunk_ids.len();
        if !is_last {
            // 块落定,无文本产出 —— 不谎报 Idle
            let act = self.active.as_ref().expect("活跃");
            return Ok(TurnEvent::Prefill {
                turn: turn_id,
                fed: act.fed,
                total: act.prompt_ids.len(),
            });
        }
        // 末块:末行采样(eos / 预算 / Token)
        self.sample_and_emit(last_row.expect("末块必有 logits")).await
    }

    /// decode 单步执行(设备侧;host 派生量由决策面备齐)
    pub(crate) async fn execute_decode(
        &mut self,
        token: u32,
        pos: usize,
        kv_slot: u32,
        gdn_slot: usize,
        grew: bool,
    ) -> Result<TurnEvent> {
        let sid = self.active.as_ref().expect("活跃").session_id;
        let prof = self.probes.step_profile;
        let t0 = prof.then(std::time::Instant::now);
        let t_step = std::time::Instant::now();
        // E2b:块链跨页增长 → 重写持久块表(图烘焙 bt 指针)
        if grew {
            self.write_bt(sid).await?;
        }
        self.session
            .step(&[
                ("frontier", &[token as f32]),
                ("pos", &[pos as f32]),
                ("kv_len", &[(pos + 1) as f32]),
                ("kv_slot", &[kv_slot as f32]),
                ("gdn_slot", &[gdn_slot as f32]),
            ])
            .await?;
        if prof {
            eprintln!("[step-prof] pos={pos} fill+launch={:?} wall={}", t0.unwrap().elapsed(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() % 100000);
        }
        if prof {
            let total = t0.unwrap().elapsed();
            eprintln!("[step-prof] pos={pos} step-total={total:?} wall2={}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() % 100000);
        }
        self.active.as_mut().expect("活跃").fed += 1;
        // E2c:decode 跨页边界 → 快照拍摄(同流保序,先拍再采样)
        {
            let act = self.active.as_ref().expect("活跃");
            if self.pool.paged && act.fed % self.pool.page == 0 {
                self.capture_gdn_snap(act.session_id, act.fed).await?;
            }
        }
        // D1 取证:GDN 状态格画像(OWL_GDN_DUMP;前 6 步,首末层 conv_v/rec)
        if self.probes.gdn_dump {
            let step = self.active.as_ref().expect("活跃").out.len() as u64;
            self.probe_gdn_dump(gdn_slot, step).await?;
        }
        // E3 设备采样:4B token 回读(旧 = 600KB logits dtoh + host argmax)。
        // OWL_HOST_ARGMAX=1 = 对照开关(host argmax 校准设备采样数值)。
        // S3-b 采样(v1 host 侧,sampler.rs):默认开 —— greedy 在
        // instruct 模型上嚌进复读吸引子(2026-10-01 27B 实测);cost =
        // 每步 logits dtoh(~1MB);device 采样核留 E4 靶面。
        let t_dtoh = prof.then(std::time::Instant::now);
        let nt = if crate::sampler::enabled() {
            let mut logits = self.session.read_output_f32("logits").await?;
            // 诊断:分布形状(绝对尺度 / top1-top2 gap / 候选词面)——
            // 复读病理定位的仪表盘(OWL_DEBUG 门控,前 16 步)
            let (turn_id, step) = {
                let act = self.active.as_ref().expect("活跃");
                (act.id, act.out.len() as u64)
            };
            // 诊断:KV 池回读(取证图内 K0 写是否落盘)—— 槽 2 = prompt
            // (eager prefill 写);槽 pos = 本步 K0(图内写)。layout
            // [nb,hkv,hd/x,page,x]:b=0 头 0 组 0 → 元素 s*8。
            if self.probes.debug && step < 16 {
                let kv0 = &self.pool.kvs[0].k_cache;
                let mut whole = vec![0u8; kv0.1 * 2];
                let face = self.session.face_mut();
                face.dtoh(&kv0.0, &mut whole).await?;
                let probe = |slot: usize| -> usize {
                    whole[slot * 8 * 2..slot * 8 * 2 + 16]
                        .chunks_exact(2)
                        .filter(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32() != 0.0)
                        .count()
                };
                let pos_now = pos;
                eprintln!(
                    "[kv-dump] pos={pos_now} 槽2非零={}/8 槽{pos_now}非零={}/8 槽{}非零={}/8",
                    probe(2),
                    probe(pos_now),
                    pos_now.saturating_sub(1),
                    probe(pos_now.saturating_sub(1))
                );
                // 取证(C1-W2 融合核 27B 案):当前槽 k 值 hex(头 0,维 0..8)
                let hex: String = whole[pos_now * 16..pos_now * 16 + 16]
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect();
                eprintln!("[kv-hex] 槽{pos_now} k[0..8] = {hex}");
            }
            if self.probes.debug && step < 16 {
                let mut top: Vec<(f32, u32)> = logits
                    .iter()
                    .enumerate()
                    .filter(|(_, &v)| v.is_finite())
                    .map(|(i, &v)| (v, i as u32))
                    .collect();
                top.select_nth_unstable_by(7, |a, b| b.0.partial_cmp(&a.0).unwrap());
                top.truncate(8);
                top.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                let pretty: Vec<String> = top
                    .iter()
                    .map(|(v, i)| {
                        format!("{:.2}:'{}'", v, self.tok.decode(&[*i]).replace('\n', "\\n"))
                    })
                    .collect();
                eprintln!("[logits-dump] t{turn_id} s{step}: {}", pretty.join(" | "));
            }
            // RNG seed = turn id ⊕ 步数(同 turn 重放确定性,跨 turn 独立)
            let mut rng = turn_id.wrapping_mul(0x9E3779B97F4A7C15)
                ^ step.wrapping_mul(0xBF58476D1CE4E5B9);
            // 惩罚集 = prompt 尾(64)+ 已生成(反循环;llama.cpp 同语义)
            let history: Vec<u32> = {
                let act = self.active.as_ref().expect("活跃");
                let tail = act.prompt_ids.len().saturating_sub(64);
                act.prompt_ids[tail..]
                    .iter()
                    .chain(act.out.iter())
                    .copied()
                    .collect()
            };
            let t_sample = prof.then(std::time::Instant::now);
            let nt = crate::sampler::sample(
                &mut logits,
                &crate::sampler::SamplerCfg::from_env(),
                &mut rng,
                &history,
            );
            if prof {
                eprintln!("[step-prof] pos={pos} host-sample={:?} wall3={}", t_sample.unwrap().elapsed(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() % 100000);
            }
            if self.probes.debug {
                eprintln!("[sample-dump] t{turn_id} s{step}: sampled={nt} hist={} pen={}", history.len(), std::env::var("OWL_REP_PENALTY").unwrap_or_else(|_| "d".into()));
            }
            nt
        } else if self.env.diag.host_argmax {
            let mut logits = self.session.read_output_f32("logits").await?;
            logits.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |a, (i, &v)| {
                if v > a.1 { (i, v) } else { a }
            })
            .0 as u32
        } else {
            // D1 取证:greedy 路径 logits 形状仪表(OWL_DEBUG;前 16 步,与
            // 采样分支同格式 —— 基线/融合 A/B 对比用)
            if self.probes.debug {
                let (turn_id, step) = {
                    let act = self.active.as_ref().expect("活跃");
                    (act.id, act.out.len() as u64)
                };
                if step < 16 {
                    let mut logits = self.session.read_output_f32("logits").await?;
                    let mut top: Vec<(f32, u32)> = logits
                        .iter()
                        .enumerate()
                        .filter(|(_, &v)| v.is_finite())
                        .map(|(i, &v)| (v, i as u32))
                        .collect();
                    top.select_nth_unstable_by(7, |a, b| b.0.partial_cmp(&a.0).unwrap());
                    top.truncate(8);
                    top.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                    let pretty: Vec<String> = top
                        .iter()
                        .map(|(v, i)| {
                            format!("{:.2}:'{}'", v, self.tok.decode(&[*i]).replace('\n', "\\n"))
                        })
                        .collect();
                    eprintln!("[logits-dump] t{turn_id} s{step}: {}", pretty.join(" | "));
                }
            }
            let tok_f = self.session.read_output_f32("token").await?;
            tok_f[0] as u32
        };
        if prof {
            eprintln!("[step-prof] pos={pos} token-dtoh={:?}", t_dtoh.unwrap().elapsed());
        }
        mrec("decode.step", t_step.elapsed());
        self.sample_and_emit(nt).await
    }

    // ====================================================================
    // E5-M2:投机轮(哑草稿版;草稿 = 3 × anchor 重复)
    // 轮不变量:轮首 GDN = state@F-1(F = fed = anchor 位;anchor KV 未写
    // —— 由本轮行 0 补写),快照 = state@F-1。部分接受 → restore(不变量
    // 保持,pending 延续);全接受 → 状态直落 state@F+k = 下轮不变量。
    // ====================================================================

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
            self.write_bt(sid).await?;
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
        if !self.spec_snap_valid {
            let ts = std::time::Instant::now();
            let face = self.session.face_mut();
            self.pool.capture_spec_snap(face, gdn_slot).await?;
            self.spec_snap_valid = true;
            t_snap = ts.elapsed();
        }

        // ② 草稿到位(host 账;轮末 propose 图/eager dtoh 产出。首轮 =
        // prefill seed propose(eager);Dumb = anchor 重复)
        let drafts: Vec<u32> = if let Some(d) = self.spec_drafts_host.take() {
            d
        } else if let Some(crate::running::Drafter::DFlash2(_)) = self.drafter.as_ref() {
            let ts = std::time::Instant::now();
            let (d, _) = self.dflash_propose(token, pos).await?;
            t_propose += ts.elapsed();
            d
        } else if self.drafter.is_some() {
            let ts = std::time::Instant::now();
            self.propose_first(token, pos).await?;
            t_propose += ts.elapsed();
            self.spec_drafts_host.take().expect("propose_first 产出")
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
            let slots_f: Vec<f32> = (pos..pos + block.len())
                .map(|p| (bt_chain[p / page] * page as u32 + (p % page) as u32) as f32)
                .collect();
            let lens_f: Vec<f32> = ((pos + 1)..=(pos + block.len())).map(|p| p as f32).collect();
            vg.step(&[
                ("ids", ids_f.as_slice()),
                ("pos", pos_f.as_slice()),
                ("kv_slots", slots_f.as_slice()),
                ("kv_lens", lens_f.as_slice()),
                ("gdn_slot", &[gdn_slot as f32]),
            ])
            .await?;
            let tok_f = vg.read_output_f32("tok").await?;
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
        let d2_arc = match self.drafter.as_ref() {
            Some(crate::running::Drafter::DFlash2(d)) => Some(std::sync::Arc::clone(d)),
            _ => None,
        };
        if let Some(d2) = d2_arc {
            // DFlash2:encode(verify 块行 [0..=m] → 草稿 KV)+ propose
            // (噪声块 [bonus, MASK×7] @ fp..fp+7;fp = pos+m+1)
            let ts = std::time::Instant::now();
            let taps: Vec<owl_iface::contract::Bytes> = (0..self.dflash_tap_count)
                .map(|i| {
                    self.verify_graph.as_ref()
                        .expect("dflash 模式要求 verify 图(taps 输出)")
                        .output_block(&format!("tap{i}"))
                        .expect("tap 输出槽")
                })
                .collect();
            self.dflash_encode(&d2, &taps, m, pos).await?;
            let fp = pos + m + 1;
            let bonus = toks[m];
            let (d, _) = self.dflash_propose(bonus, fp).await?;
            t_propose += ts.elapsed();
            self.spec_drafts_host = Some(d);
        } else if !self.propose_graphs.is_empty() && std::env::var_os("OWL_PROPOSE_EAGER").is_none() {
            let ts = std::time::Instant::now();
            let d = self.propose_graph_step(m, &toks, pos).await?;
            t_propose += ts.elapsed();
            self.spec_drafts_host = Some(d);
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
            self.spec_drafts_host = Some(
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
                    owl_models::interpreters::eval_ops_multi(&refs, face).await?;
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
        self.spec_snap_valid = false;

        // ⑥ 提交:fed = bonus 位(F');发射 = 接受 drafts + bonus(m+1)
        let toks: Vec<u32> = drafts[..m].to_vec();
        let mut toks = toks;
        toks.push(bonus);
        {
            let act = self.active.as_mut().expect("活跃");
            act.fed = pos + m + 1;
        }

        // ⑦ extend + propose(同迭代消费 verify hidden;产出下轮草稿
        // host 账)。图态 = 桶 m 一发;eager fallback = propose_round。
        if !self.propose_graphs.is_empty() && std::env::var_os("OWL_PROPOSE_EAGER").is_none() {
            let ts = std::time::Instant::now();
            let d = self.propose_graph_step(m, &toks, pos).await?;
            t_propose += ts.elapsed();
            self.spec_drafts_host = Some(d);
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
            self.spec_drafts_host = Some(
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
        self.spec_stats.rounds += 1;
        self.spec_stats.sum_m += m;
        if m == depth {
            self.spec_stats.full += 1;
        } else if m == 0 {
            self.spec_stats.zero += 1;
        } else {
            self.spec_stats.partial += 1;
        }
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
        if prof {
            eprintln!(
                "[spec-prof] pos={pos} total={t_all:?} verify={t_verify:?} rollback={t_rollback:?} propose={t_propose:?} snap={t_snap:?} m={m}"
            );
        }
        self.pending_events
            .pop_front()
            .ok_or_else(|| owl_models::ModelError::Msg("spec 轮零事件(空块?)".into()))
    }

    /// DFlash2 prefill encode(prefill chunk 全行草稿 KV 物化):taps
    /// [T, hidden] 视图 → memory → 5 层 kv-only 写 @ base..base+T。
    async fn prefill_dflash_encode(
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
        let (bt_chain, _) = {
            let act = self.active.as_ref().expect("active 已保证");
            let s = self.sessions.get(act.session_id).expect("账在");
            (s.block_table.clone(), s.gdn_slot)
        };
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
                        .map(|i| {
                            let p = base + i;
                            (bt_chain[p / page] * page as u32 + (p % page) as u32) as f32
                        })
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
            owl_models::interpreters::eval_ops_multi(&refs, face).await?;
        }
        Ok(())
    }

    /// DFlash2 propose(E5-DF3):噪声块 [anchor, MASK×7] @ pos..pos+7,
    /// 草稿 KV 窗口 [0, pos+8)(前缀 pos 行已物化 + 自块 8 行直读)。
    /// 返回 (草稿 host 表, 观测面块)。首轮(pos = fed)与轮末同式。
    async fn dflash_propose(
        &mut self,
        anchor: u32,
        pos: usize,
    ) -> Result<(Vec<u32>, owl_iface::contract::Bytes)> {
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
        // 草稿 KV 叶子(NC 臂不消费 slots/lens;恒等表占位)
        let leaves = self.pool.dflash_kv_leaves().expect("dflash 池");
        let bt = self.pool.bt_leaf_flat();
        let kvs: Vec<KvBuffers> = leaves
            .iter()
            .map(|(k, v)| KvBuffers {
                k_cache: k.clone(),
                v_cache: v.clone(),
                slots: TensorOps::from_host(Dtype::F32, vec![8], &f32b(&(0..8).map(|i| i as f32).collect::<Vec<_>>())),
                kv_lens: TensorOps::from_host(Dtype::F32, vec![8], &f32b(&(0..8).map(|i| (pos + 1 + i) as f32).collect::<Vec<_>>())),
                block_tables: bt.clone(),
            })
            .collect();
        let mut ctx = ForwardCtx::minimal(8);
        ctx.env = self.env;
        let (hidden, drafts, scores) =
            d2.propose_block(&toks_t, &pos_t, &anchor_t, &kvs, &rope, &self.model.embed, pos + 8, &ctx);
        let _ = hidden;
        let (b, arena) = {
            let face = self.session.face_mut();
            owl_models::interpreters::eval_ops_scoped_env(drafts.step(), face, self.env).await?
        };
        let out = {
            let face = self.session.face_mut();
            let mut buf = vec![0u8; 28];
            face.dtoh(&b, &mut buf).await?;
            face.free(&arena).await?;
            buf.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u32)
                .collect::<Vec<_>>()
        };
        Ok((out, b))
    }

    /// DFlash2 encode(E5-DF2;sglang _append_target_hidden 同语义):
    /// verify 块行 [0..=m](taps 前缀视图)→ memory → 5 层 kv-only 写
    /// 草稿池 @ pos..pos+m+1。bonus 行(行 m+1)不写(下轮作行 0)。
    async fn dflash_encode(
        &mut self,
        d2: &std::sync::Arc<owl_models::layers::dflash2::DFlash2Draft>,
        taps: &[owl_iface::contract::Bytes],
        m: usize,
        pos: usize,
    ) -> Result<()> {
        let rows = m + 1;
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
        // 编码槽表 = 目标块链物理槽(pos..pos+rows)
        let (bt_chain, _) = {
            let act = self.active.as_ref().expect("active 已保证");
            let s = self.sessions.get(act.session_id).expect("账在");
            (s.block_table.clone(), s.gdn_slot)
        };
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
                    vec![rows],
                    &f32b(&(0..rows)
                        .map(|i| {
                            let p = pos + i;
                            (bt_chain[p / page] * page as u32 + (p % page) as u32) as f32
                        })
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
        let roots = d2.encode_kv(&memory, &pos_t, &kvs, &rope, &ctx);
        {
            let face = self.session.face_mut();
            let refs: Vec<&TensorOps> = roots.iter().collect();
            owl_models::interpreters::eval_ops_multi(&refs, face).await?;
        }
        Ok(())
    }

    /// 首轮 propose(E5-M2b;m=0 特例):pair (h_{pos-1}, emb(anchor)) @
    /// 位置 pos-1 = M1 propose 本尊;seed hidden = prefill 末块 harvest。
    /// 产出 k 草稿块持久(spec_drafts)。
    async fn propose_first(&mut self, anchor: u32, pos: usize) -> Result<()> {
        let depth = self.spec_depth;
        let seed = self
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
            self.spec_drafts_host = Some(
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
    async fn propose_round(
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
    async fn prefill_extend(
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
    async fn propose_graph_step(
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
    fn mtp_chain_bt(&mut self, cover: usize) -> Result<Vec<u32>> {
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
    async fn verify_forward(
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
        let fi = if !self.pool.k_fis.is_empty() {
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
        let mut root = TensorOps::call(owl_models::ops::ids::OPS_CONCAT);
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

    /// GDN 状态格画像(首末层 conv_v/rec;owl_gdn_slot_profile 委托)
    async fn probe_gdn_dump(
        &mut self,
        gdn_slot: usize,
        step: u64,
    ) -> Result<()> {
        if step >= 6 {
            return Ok(());
        }
        let lines = {
            let face = self.session.face_mut();
            self.pool.gdn_slot_profile(face, gdn_slot, step).await?
        };
        for l in lines {
            eprintln!("{l}");
        }
        Ok(())
    }

    /// 采样 + 事件产出(Greedy;E3 设备 argmax,token id 直入)。
    /// eos 命中或预算尽 → Completed;否则 Token(delta = 解码文本增量)。
    async fn sample_and_emit(&mut self, nt: u32) -> Result<TurnEvent> {
        let turn_id = self.active.as_ref().expect("active 已保证").id;
        if self.tok.is_eos(nt) {
            return self.complete().await;
        }
        let (delta, budget_done) = {
            let act = self.active.as_mut().expect("active 已保证");
            act.out.push(nt);
            (decode_delta(&self.tok, &act.out, &mut act.decoded), act.out.len() >= act.max_new)
        };
        if budget_done {
            return self.complete().await;
        }
        Ok(TurnEvent::Token { turn: turn_id, delta })
    }

    /// 活跃会话块表 → 持久 bt 块(设备;StatePool::write_bt 落地,
    /// 本壳只解析会话块链)
    async fn write_bt(&mut self, sid: u64) -> Result<()> {
        let table = self.sessions.get(sid).expect("账在").block_table.clone();
        let face = self.session.face_mut();
        self.pool.write_bt(face, &table).await
    }

    /// GDN 快照拍摄(E2c;块边界跨越时调用):会话格状态 → 槽位缓冲。
    /// key = **边界覆盖块**(block_table[fed/page - 1],即快照边界最后
    /// 一块的物理 id)—— 块表在 turn-open 预长满,表尾 ≠ 边界块。
    /// 槽位选择/LRU 覆盖写在 StatePool::capture_snap(state.rs)。
    async fn capture_gdn_snap(&mut self, sid: u64, fed: usize) -> Result<()> {
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

    /// 块式 prefill(W1;批P5 契约):ids [T] 从 KV 行 base 起步,
    /// 单序列语义(gdn_slot 恒 0;KV 行随 token 走)。末块走 logits 根
    /// 返回末行;中间块走 last_hidden 根(免 lm_head [T,V] 计算与大
    /// dtoh)返回 None。
    async fn prefill_chunk(
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
        let fi = if !self.pool.k_fis.is_empty() {
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
            let (hidden_full, prefill_taps) = if self.dflash_tap_count > 0 {
                let (h, tp) = self.model.tapped_hidden(&ids_t, &ctx, &[5, 19, 33, 47, 61]);
                let mut c = TensorOps::call(owl_models::ops::ids::OPS_CONCAT).arg(&h);
                for t in &tp {
                    c = c.arg(t);
                }
                for _ in tp.len()..7 {
                    c = c.arg(&h);
                }
                let n = tp.len() + 1;
                let dd = h.shape()[1];
                let root = c
                    .arg_usize(n)
                    .arg_usize(t * dd)
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
            // DFlash2:草稿 KV 物化(root 块行 [T, 6T) 切片出 taps → encode)
            if let Some(tp) = &prefill_taps {
                let ts = std::time::Instant::now();
                let n_tp = tp.len();
                let tap_blocks: Vec<owl_iface::contract::Bytes> = (0..n_tp)
                    .map(|_| owl_iface::contract::Bytes::new(hfb.id, t * d_model))
                    .collect();
                let _ = tap_blocks; // taps 与 hidden 同块 —— 直接切片视图喂 encode
                let tap_views: Vec<TensorOps> = (0..n_tp)
                    .map(|i| {
                        TensorOps::of_block(hfb.id, d.dtype, vec![n_tp * t + t, d_model])
                            .slice_view((i + 1) * t * d_model, vec![t, d_model])
                    })
                    .collect();
                self.prefill_dflash_encode(&tap_views, base).await?;
                eprintln!("[dflash] prefill encode {t} rows @{} {:?}", base, ts.elapsed());
            }
            let hview = TensorOps::of_block(hfb.id, d.dtype, vec![t, d_model]);
            // 末行窄切 **物化**(OPS_NARROW):SliceView 作根透传父块丢偏移
            // (刀 3b 同族),根必须真块
            let last = TensorOps::call(owl_models::ops::ids::OPS_NARROW)
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
            {
                let face = self.session.face_mut();
                let mut buf_t = [0u8; 4];
                face.dtoh(&b, &mut buf_t).await?;
                face.free(&[b.id]).await?;
                face.free(&arena3).await?;
                face.free(&arena2).await?;
                if self.drafter.is_some() {
                    self.spec_seed_hidden = Some(hb);
                } else {
                    face.free(&[hb.id]).await?;
                }
                face.free(&[hfb.id]).await?;
                face.free(&arena1).await?; // E2a:中间块归池(每 chunk 零净增)
                buf = buf_t;
            }
            let nt = f32::from_le_bytes(buf) as u32;
            Ok(Some(nt))
        } else {
            // 中间块:last_hidden 根(状态推进完整;lm_head 免算)。
            // DFlash2:同构 concat 单根(taps 随块回收前切片 encode)
            let (tree, prefill_taps) = if self.dflash_tap_count > 0 {
                let (h, tp) = self.model.tapped_hidden(&ids_t, &ctx, &[5, 19, 33, 47, 61]);
                let mut c = TensorOps::call(owl_models::ops::ids::OPS_CONCAT).arg(&h);
                for tt in &tp {
                    c = c.arg(tt);
                }
                for _ in tp.len()..7 {
                    c = c.arg(&h);
                }
                let n = tp.len() + 1;
                let dd = h.shape()[1];
                let root = c
                    .arg_usize(n)
                    .arg_usize(t * dd)
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
                let mut buf = vec![0u8; t * d.hidden * esz];
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

    async fn complete(&mut self) -> Result<TurnEvent> {
        let act = self.active.take().expect("active");
        // spec 态清理(E5-M2b):陈旧草稿块作废( Continuation 已变,
        // 下一 turn 由 prefill seed 重新 propose;草稿只影响速度,不清也
        // 恒等 —— 但清了省 dtoh + free 账目干净)
        self.spec_drafts_host = None;
        // 会话账落地(S1 收口):KV 此刻 = prompt 全量 + 生成段;GDN 状态
        // 跨 turn 连续。临时会话终了即焚(表不随 turn 数无界增长)
        if let Some(s) = self.sessions.get_mut(act.session_id) {
            s.commit_turn(&act.prompt_ids, &act.out);
            // E2c:块链登记前缀缓存(缓存持引用;会话释放后块仍驻留)
            if self.pool.paged {
                self.blocks_m.cache_seq(&s.tokens, &s.block_table);
            }
        }
        if act.ephemeral {
            // 终了即焚:块链归还账房 + 会话表移除(GDN 格随之释放)
            if let Some(s) = self.sessions.get_mut(act.session_id) {
                let mut table = std::mem::take(&mut s.block_table);
                self.blocks_m.release_table(&mut table);
                // MTP 链第二链同步归还(E5-M2b)
                let mut mtp_table = std::mem::take(&mut s.mtp_block_table);
                self.blocks_mtp.release_table(&mut mtp_table);
            }
            self.sessions.close(act.session_id).ok();
        }
        let text = self.tok.decode(&act.out);
        Ok(TurnEvent::Completed { turn: act.id, text })
    }
}

/// 增量解码:全量重解取后缀差分。多字节字符跨 token 时,半截字节经
/// tokenizers 解出 U+FFFD 占位(字节数与真前缀不等,**不能按字节
/// index 切**)—— 差分按字节公共前缀比对,双侧回退到字符边界;尾部
/// 占位符扣住不发,等拼全后随下一笔 delta 出(终文由 complete()
/// 全文重解兜底,流式末字符可能延后一笔)
pub(crate) fn decode_delta(tok: &Tokenizer, out: &[u32], decoded: &mut String) -> String {
    const REPL: &str = "\u{FFFD}";
    let full = tok.decode(out);
    let common = decoded
        .as_bytes()
        .iter()
        .zip(full.as_bytes())
        .take_while(|(a, b)| a == b)
        .count();
    let mut cut = common;
    while cut > 0 && (!full.is_char_boundary(cut) || !decoded.is_char_boundary(cut)) {
        cut -= 1;
    }
    let mut emit = full[cut..].to_string();
    if emit.ends_with(REPL) {
        emit.truncate(emit.len() - REPL.len());
    }
    *decoded = full[..cut + emit.len()].to_string();
    emit
}

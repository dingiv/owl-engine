//! 执行器(设备侧照办;2026-10-01 拆分自 engine.rs §3)。
//!
//! S0 决策/执行分离的**执行半**:调度面(scheduler.rs)产出
//! SchedulerOutput,本模块无决策权 —— execute_begin(GDN 重置/快照
//! 恢复/块表重写)、execute_prefill(块式预填 + 边界快照)、
//! execute_decode(图步进 + 设备采样回读)、采样收口
//! (sample_and_emit/complete)+ prefill_chunk(树根装配与执行)。
//! 状态块触碰一律经 StatePool(state.rs),不直摸块句柄。

use owl_iface::contract::{DeviceClient, Dtype, ModelError};
use owl_models::interpreters::{eval_ops_scoped, eval_ops_scoped_env};
use owl_models::layers::gdn::GdnBuffers;
use owl_models::module::{FiPrefillCtx, ForwardCtx, KvBuffers, Module};
use owl_models::tokenizer::Tokenizer;
use owl_models::TensorOps;

use crate::graph_plan::f32b;
use crate::scheduler::BeginPlan;
use crate::turn::TurnEvent;

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
        let prof = std::env::var_os("OWL_STEP_PROFILE").is_some();
        let t0 = prof.then(std::time::Instant::now);
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
        if std::env::var_os("OWL_GDN_DUMP").is_some() {
            let step = self.active.as_ref().expect("活跃").out.len() as u64;
            if step < 6 {
                let lines = {
                    let face = self.session.face_mut();
                    self.pool.gdn_slot_profile(face, gdn_slot, step).await?
                };
                for l in lines {
                    eprintln!("{l}");
                }
            }
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
            if std::env::var_os("OWL_DEBUG").is_some() && step < 16 {
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
            if std::env::var_os("OWL_DEBUG").is_some() && step < 16 {
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
            if std::env::var_os("OWL_DEBUG").is_some() {
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
            if std::env::var_os("OWL_DEBUG").is_some() {
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
        self.sample_and_emit(nt).await
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
            // E3:末行设备 argmax(4B 回读,免 [T,V] 大 dtoh);
            // 2026-10-02 forward_last:hidden 先窄末行再 lm_head ——
            // logits [1,V] 而非 [T,V](大 chunk 内存/算力双省)
            let tree = self.model.forward_last(&ids_t, &ctx);
            let tok = owl_models::ops::argmax_f32idx(&tree, vocab, 0);
            let (b, arena) = eval_ops_scoped_env(tok.step(), face, self.env).await?;
            let mut buf = [0u8; 4];
            face.dtoh(&b, &mut buf).await?;
            face.free(&arena).await?; // E2a:根已收割,中间块归池
            let nt = f32::from_le_bytes(buf) as u32;
            Ok(Some(nt))
        } else {
            // 中间块:last_hidden 根(状态推进完整;lm_head 免算)
            let tree = self.model.last_hidden(&ids_t, &ctx);
            let (b, arena) = eval_ops_scoped_env(tree.step(), face, self.env).await?;
            let esz = if d.dtype == Dtype::F16 { 2 } else { 4 };
            let mut buf = vec![0u8; t * d.hidden * esz];
            face.dtoh(&b, &mut buf).await?;
            if std::env::var_os("OWL_PREFILL_CKSUM").is_some() {
                // 临时取证:每 chunk 隐层校验和(FI 开/关对比找第一分歧)
                let cks: f32 = buf.chunks_exact(2)
                    .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32().abs())
                    .sum();
                eprintln!("[cksum] chunk base={base} t={t} sum|x|={cks:.4}");
            }
            face.free(&arena).await?; // E2a:中间块归池(每 chunk 零净增)
            Ok(None)
        }
    }

    async fn complete(&mut self) -> Result<TurnEvent> {
        let act = self.active.take().expect("active");
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

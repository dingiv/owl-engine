//! 调度三相位执行(begin/prefill/decode)+ 采样收口 + GDN 画像
use super::*;
use owl_models::ops::PlanNode; // ⚙ diag:池叶 → 块 id(写者足迹取证)

impl<D: DeviceClient> crate::running::RunningEngine<D> {
    /// turn 启动执行(设备侧照办):GDN 格重置 / 边界快照恢复 + 持久
    /// 块表重写(write_bt;图烘焙 bt 指针,内容重写指针稳定图不重捕)
    pub(crate) async fn execute_begin(&mut self, b: BeginPlan) -> Result<()> {
        // 性能面:本轮 epoch 起点(/debug/perf?epoch=1 = 只看本轮;不 reset,
        // 历史样本仍可全量查询 —— perf-framework.md §三)
        owl_shared::metrics::mark_epoch();
        if b.reset_gdn {
            let face = self.graphs.session.face_mut();
            self.pool.reset_gdn(face, b.gdn_slot).await?;
        } else if let Some(key) = b.restore_key {
            // 前缀命中:恢复边界快照到会话格(免重灌;GDN 态对齐)
            let face = self.graphs.session.face_mut();
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
            let act = self.turns.active.as_ref().expect("活跃");
            (act.id, act.session_id)
        };
        // 调试门:prefill 块粒度(见 owl_shared::stepgate;`--step-gate` 控制)
        owl_shared::step_gate!("engine.prefill").await;
        let last_row = self.prefill_chunk(&chunk_ids, base, is_last).await?;
        // E2c:块边界跨越 → GDN 快照拍摄(可复用身份 = 链尾块)
        let new_fed = base + chunk_ids.len();
        if self.pool.paged && new_fed.is_multiple_of(self.pool.page) {
            self.capture_gdn_snap(sid, new_fed).await?;
        }
        self.turns.active.as_mut().expect("活跃").fed += chunk_ids.len();
        if !is_last {
            // 块落定,无文本产出 —— 不谎报 Idle
            let act = self.turns.active.as_ref().expect("活跃");
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
        let sid = self.turns.active.as_ref().expect("活跃").session_id;
        // 调试门:decode 步粒度
        owl_shared::step_gate!("engine.decode").await;
        let prof = self.probes.step_profile;
        let t0 = prof.then(std::time::Instant::now);
        let t_step = std::time::Instant::now();
        // E2b:块链跨页增长 → 重写持久块表(图烘焙 bt 指针)
        if grew {
            self.write_bt(sid).await?;
        }
        // XQA 裸 decode 臂(2026-10-13 ②):`xqa_seq` = kv_len 的 **u32 位型**
        // (核按 `const uint32_t*` 读;`(pos+1) as f32` 会拿到浮点位型);
        // `xqa_sem` 每轮清零(KV-split 归约计数 —— 参考实现 `sem.zero_()`)。
        // 两槽只在 `decode_xqa` 开时存在于图中(否则 `step` 会报未声明槽)。
        let mut step_slots: Vec<(&str, Vec<f32>)> = vec![
            ("frontier", vec![token as f32]),
            ("pos", vec![pos as f32]),
            ("kv_len", vec![(pos + 1) as f32]),
            ("kv_slot", vec![kv_slot as f32]),
            ("gdn_slot", vec![gdn_slot as f32]),
        ];
        if self.graphs.session.has_input("xqa_seq") {
            step_slots.push(("xqa_seq", vec![f32::from_bits((pos + 1) as u32)]));
            let n = self.graphs.session.input_len("xqa_sem").unwrap_or(0);
            step_slots.push(("xqa_sem", vec![0.0; n]));
            // XQA 页表 = 本会话真页链(不喂 = 图里恒等链 = 读到别家 KV;§7.6)
            if self.graphs.session.has_input("xqa_pt") {
                step_slots.push(("xqa_pt", self.xqa_pt_of(sid)));
            }
        }
        // **FI 断点臂(V1c;T=1)**:宿主标量槽 ctx_total + 表四件套(位型过线)。
        // 判据 = 槽在场(与 xqa_seq 同款纪律:`decode_fi` 关时图里没有这些槽)
        if self.graphs.session.has_input("fi_q_cu") {
            if let Some(cell) = self.graphs.decode_fi_ctx() {
                let ctx_total = pos + 1;
                // ⚙ diag(`OWL_FI_BUCKET=<n>`):把**宿主标量槽**按桶上取整 ⇒ plan
                // 只在跨桶时重算(表仍按真值写 ⇒ 每行 KV 上界由设备表定)。
                // 判据:文本仍 ≡ 参考臂(plan 过供给只是多些被掩的 CTA)。
                let bucket = owl_shared::env_reader::parse::<usize>("OWL_FI_BUCKET").unwrap_or(0);
                let plan_ctx = if bucket > 0 {
                    ctx_total.div_ceil(bucket) * bucket
                } else {
                    ctx_total
                };
                cell.store(plan_ctx, std::sync::atomic::Ordering::Relaxed);
                let page = self.pool.page;
                let bt_chain = {
                    let s = self.bank.sessions.get(sid).expect("账在");
                    s.block_table.clone()
                };
                let nb_used = ctx_total.div_ceil(page).min(bt_chain.len()).max(1);
                let mut idx = vec![0f32; self.pool.nb];
                for (i, b) in bt_chain.iter().take(nb_used).enumerate() {
                    idx[i] = f32::from_bits(*b);
                }
                step_slots.push(("fi_q_cu", vec![0.0, f32::from_bits(1)]));
                step_slots.push(("fi_indices", idx));
                step_slots.push(("fi_indptr", vec![0.0, f32::from_bits(nb_used as u32)]));
                step_slots.push((
                    "fi_last_len",
                    vec![f32::from_bits((ctx_total - (nb_used - 1) * page) as u32)],
                ));
            }
        }
        let step_refs: Vec<(&str, &[f32])> =
            step_slots.iter().map(|(n, v)| (*n, v.as_slice())).collect();
        // ⚙ KV 页字节差分(OWL_KV_DUMP=<pos>,一次):写前/写后各 d2h 一页 ⇒
        // **写者的足迹**直接给出块内元素序(页面连续 vs 8B 小块 strided),
        // 不需要任何参考值(见 docs §11.11「写者 A/B」)。
        let kv_dump_at = owl_shared::env_reader::parse::<usize>("OWL_KV_DUMP");
        if kv_dump_at == Some(pos) {
            self.dump_kv_page(sid, kv_slot, "before").await?;
        }
        self.graphs.session.step(&step_refs).await?;
        if kv_dump_at == Some(pos) {
            self.dump_kv_page(sid, kv_slot, "after").await?;
        }
        if prof {
            eprintln!("[step-prof] pos={pos} fill+launch={:?} wall={}", t0.unwrap().elapsed(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() % 100000);
        }
        if prof {
            let total = t0.unwrap().elapsed();
            eprintln!("[step-prof] pos={pos} step-total={total:?} wall2={}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() % 100000);
        }
        self.turns.active.as_mut().expect("活跃").fed += 1;
        // E2c:decode 跨页边界 → 快照拍摄(同流保序,先拍再采样)
        {
            let act = self.turns.active.as_ref().expect("活跃");
            if self.pool.paged && act.fed.is_multiple_of(self.pool.page) {
                self.capture_gdn_snap(act.session_id, act.fed).await?;
            }
        }
        // D1 取证:GDN 状态格画像(OWL_GDN_DUMP;前 6 步,首末层 conv_v/rec)
        if self.probes.gdn_dump {
            let step = self.turns.active.as_ref().expect("活跃").out.len() as u64;
            self.probe_gdn_dump(gdn_slot, step).await?;
        }
        // E3 设备采样:4B token 回读(旧 = 600KB logits dtoh + host argmax)。
        // OWL_HOST_ARGMAX=1 = 对照开关(host argmax 校准设备采样数值)。
        // S3-b 采样(v1 host 侧,sampler.rs):默认开 —— greedy 在
        // instruct 模型上嚌进复读吸引子(2026-10-01 27B 实测);cost =
        // 每步 logits dtoh(~1MB);device 采样核留 E4 靶面。
        let t_dtoh = prof.then(std::time::Instant::now);
        // 采样开/关(AGT-06):**请求带采样参数则以请求为准**
        // (`temperature=0` = 本 turn 贪心);未带 = 引擎 knobs(flavor 语义)。
        // 旧行为不变(GDN 恒等门类测试不带请求参数)。
        let eff_sampler = self.turns.active.as_ref().expect("活跃").sampler;
        let sampler_on = self.turns.active.as_ref().expect("活跃").sampler_on;
        let nt = if sampler_on {
            let mut logits = self.graphs.session.read_output_f32("logits").await?;
            // 诊断:分布形状(绝对尺度 / top1-top2 gap / 候选词面)——
            // 复读病理定位的仪表盘(OWL_DEBUG 门控,前 16 步)
            let (turn_id, step) = {
                let act = self.turns.active.as_ref().expect("活跃");
                (act.id, act.out.len() as u64)
            };
            // 诊断:KV 池回读(取证图内 K0 写是否落盘)—— 槽 2 = prompt
            // (eager prefill 写);槽 pos = 本步 K0(图内写)。
            // 回读偏移走**布局单源**(`env::knhd_offset`;P6-A 后单布局,
            // 原 classic x 交错分支已删)。头 0 / 维 0..8。
            // B6.3:fp8 池字节域同序(1B/elem,e4m3);A/B 池对拍仪表
            // (f16 boot 半值 vs fp8 boot 字节,host RNE 桥对)
            if self.probes.debug && step < 16 {
                let fp8 = self.pool.kv_fp8;
                let esz = if fp8 { 1usize } else { 2usize };
                let kv0 = &self.pool.kvs[0].k_cache;
                let mut whole = vec![0u8; kv0.1 * esz];
                let face = self.graphs.session.face_mut();
                face.dtoh(&kv0.0, &mut whole).await?;
                let (hkv, hd) = (self.pool.dims.hkv, self.pool.dims.hd);
                let _page = self.pool.page;
                let k_off =
                    |slot: usize, d: usize| -> usize { owl_models::env::knhd_offset(slot, 0, d, hkv, hd) };
                let probe = |slot: usize| -> usize {
                    let base = k_off(slot, 0) * esz;
                    if fp8 {
                        whole[base..base + 8].iter().filter(|&&b| b != 0).count()
                    } else {
                        whole[base..base + 16]
                            .chunks_exact(2)
                            .filter(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32() != 0.0)
                            .count()
                    }
                };
                let pos_now = pos;
                eprintln!(
                    "[kv-dump] fp8={fp8} pos={pos_now} 槽2非零={}/8 槽{pos_now}非零={}/8 槽{}非零={}/8",
                    probe(2),
                    probe(pos_now),
                    pos_now.saturating_sub(1),
                    probe(pos_now.saturating_sub(1))
                );
                // 取证:当前槽 k 值(f16 半值 / fp8 原始字节,头 0,维 0..8)
                let base_now = k_off(pos_now, 0) * esz;
                if fp8 {
                    let hex: String = whole[base_now..base_now + 8]
                        .iter()
                        .map(|b| format!("{b:02x}"))
                        .collect();
                    eprintln!("[kv-hex] fp8 槽{pos_now} k[0..8] = {hex}");
                } else {
                    let hex: String = whole[base_now..base_now + 16]
                        .iter()
                        .map(|b| format!("{b:02x}"))
                        .collect();
                    eprintln!("[kv-hex] 槽{pos_now} k[0..8] = {hex}");
                }
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
            // RNG seed = turn id ⊕ 步数(同 turn 重放确定性,跨 turn 独立);
            // 请求给了 `seed` ⇒ 以它为基(AGT-06)
            let seed_base = self.turns.active
                .as_ref()
                .expect("活跃")
                .seed
                .unwrap_or_else(|| turn_id.wrapping_mul(0x9E3779B97F4A7C15));
            let mut rng = seed_base ^ step.wrapping_mul(0xBF58476D1CE4E5B9);
            // 惩罚集 = prompt 尾(64)+ 已生成(反循环;llama.cpp 同语义)
            let history: Vec<u32> = {
                let act = self.turns.active.as_ref().expect("活跃");
                let tail = act.prompt_ids.len().saturating_sub(64);
                act.prompt_ids[tail..]
                    .iter()
                    .chain(act.out.iter())
                    .copied()
                    .collect()
            };
            let t_sample = prof.then(std::time::Instant::now);
            let nt = crate::sampler::sample(&mut logits, &eff_sampler, &mut rng, &history);
            if prof {
                eprintln!("[step-prof] pos={pos} host-sample={:?} wall3={}", t_sample.unwrap().elapsed(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() % 100000);
            }
            if self.probes.debug {
                eprintln!("[sample-dump] t{turn_id} s{step}: sampled={nt} hist={} pen={}", history.len(), eff_sampler.rep_penalty);
            }
            nt
        } else if self.env.diag.host_argmax {
            let logits = self.graphs.session.read_output_f32("logits").await?;
            logits.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |a, (i, &v)| {
                if v > a.1 { (i, v) } else { a }
            })
            .0 as u32
        } else {
            // D1 取证:greedy 路径 logits 形状仪表(OWL_DEBUG;前 16 步,与
            // 采样分支同格式 —— 基线/融合 A/B 对比用)
            if self.probes.debug {
                let (turn_id, step) = {
                    let act = self.turns.active.as_ref().expect("活跃");
                    (act.id, act.out.len() as u64)
                };
                if step < 16 {
                    let logits = self.graphs.session.read_output_f32("logits").await?;
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
            let tok_f = self.graphs.session.read_output_f32("token").await?;
            tok_f[0] as u32
        };
        if prof {
            eprintln!("[step-prof] pos={pos} token-dtoh={:?}", t_dtoh.unwrap().elapsed());
        }
        mrec("decode.step", t_step.elapsed());
        // B4 降级期草稿同步(§6.24 方案 A):decode 图 taps → 1 行 encode。
        // 毒化带 ≤8 token 即被覆写,探测重入零损;成本 ≈ 0.5ms/tok(2%)
        if self.tspec_ref().spec_degraded && self.spec.dflash_tap_count > 0 {
            // 降级步分相(2026-10-12 探针完善):sync encode 单独计时 ——
            // 9.3 t/s 案的 80ms/步嫌疑犯,一屏定谳
            let t_syn = std::time::Instant::now();
            let dtype = self.pool.dims.dtype;
            let hidden = self.pool.dims.hidden;
            let taps: Vec<TensorOps> = (0..self.spec.dflash_tap_count)
                .filter_map(|i| self.graphs.session.output_block(&format!("tap{i}")))
                .map(|b| TensorOps::of_block(b.id, dtype, vec![1, hidden]))
                .collect();
            if taps.len() == self.spec.dflash_tap_count {
                self.prefill_dflash_encode(&taps, pos).await?;
            }
            mrec("decode.sync", t_syn.elapsed());
            self.tspec().spec_steps_degraded += 1;
            // 滚动裸步成本 ewma(动态盈亏线分母;α=0.15)
            {
                let bc = t_step.elapsed().as_secs_f32() * 1000.0;
                let spec = self.tspec();
                spec.bare_cost_ewma_ms = if spec.bare_cost_ewma_ms > 0.0 {
                    spec.bare_cost_ewma_ms * 0.85 + bc * 0.15
                } else {
                    bc
                };
                // 采样时的 ctx(换档失效判据;见 SpecDegrade.bare_cost_ctx)
                spec.bare_cost_ctx = pos;
            }
            mrec("decode.step.deg", t_step.elapsed());
        }
        self.sample_and_emit(nt).await
    }

    // ====================================================================
    // E5-M2:投机轮(哑草稿版;草稿 = 3 × anchor 重复)
    // 轮不变量:轮首 GDN = state@F-1(F = fed = anchor 位;anchor KV 未写
    // —— 由本轮行 0 补写),快照 = state@F-1。部分接受 → restore(不变量
    // 保持,pending 延续);全接受 → 状态直落 state@F+k = 下轮不变量。
    // ====================================================================


    /// GDN 状态格画像(首末层 conv_v/rec;owl_gdn_slot_profile 委托)
    pub(crate) async fn probe_gdn_dump(
        &mut self,
        gdn_slot: usize,
        step: u64,
    ) -> Result<()> {
        if step >= 6 {
            return Ok(());
        }
        let lines = {
            let face = self.graphs.session.face_mut();
            self.pool.gdn_slot_profile(face, gdn_slot, step, self.cfg.knobs.gdn_dump_all).await?
        };
        for l in lines {
            eprintln!("{l}");
        }
        Ok(())
    }


    /// 采样 + 事件产出(Greedy;E3 设备 argmax,token id 直入)。
    /// eos 命中或预算尽 → Completed;否则 Token(delta = 解码文本增量)。
    pub(crate) async fn sample_and_emit(&mut self, nt: u32) -> Result<TurnEvent> {
        let turn_id = self.turns.active.as_ref().expect("active 已保证").id;
        {
            // 崩坏排查打点:首 token / EOS 事实(decode 与 spec 两路汇合)
            let bs = self.boot_seq;
            let first = self.turns.active
                .as_ref()
                .map(|a| a.out.is_empty())
                .unwrap_or(false);
            if first {
                mfact(bs, "emit.first", nt as u64);
            }
            if self.tok.is_eos(nt) {
                mfact(bs, "emit.eos", nt as u64);
            }
        }
        // ignore_eos(OpenAI 兼容;测速口径):EOS 只记账不终止
        let ignore_eos = self.turns.active.as_ref().map(|a| a.ignore_eos).unwrap_or(false);
        if self.tok.is_eos(nt) && !ignore_eos {
            return self.complete(crate::turn::FinishReason::Stop).await;
        }
        let budget_done = {
            let act = self.turns.active.as_mut().expect("active 已保证");
            act.out.push(nt);
            // `decoded` 由全量重解差分维护(delta 本步统一从 `emitted` 切片取,
            // 以便 stop 尾缓冲能回吐);返回值丢弃是有意的
            let _appended = decode_delta(&self.tok, &act.out, &mut act.decoded);
            act.out.len() >= act.max_new
        };
        // ── stop 串(AGT-06;OpenAI `stop`)──
        // ① 命中:输出截断于停止串起点(prompt 里也不能多一个字节),
        //    本 turn 以 `Stop` 终止;
        // ② 未命中:**尾缓冲上界** —— 发出的文本不得含任一停止串的真前缀
        //    (否则半截停止串已流给客户端,命中时收不回来)。
        // 空 stop 列表 = 零额外开销(`safe_end` 直接返回全长)。
        let (delta, stop_hit) = {
            let act = self.turns.active.as_mut().expect("active 已保证");
            let hit = if act.stop.is_empty() {
                None
            } else {
                act.stop
                    .iter()
                    .filter_map(|s| act.decoded.find(s.as_str()).map(|p| (p, s.clone())))
                    .min_by_key(|(p, _)| *p)
            };
            let start = floor_boundary(&act.decoded, act.emitted);
            let end = match &hit {
                Some((p, _)) => floor_boundary(&act.decoded, *p).max(start),
                None => safe_end(&act.decoded, &act.stop).max(start),
            };
            let out = act.decoded[start..end].to_string();
            act.emitted = end;
            if let Some((p, _)) = &hit {
                // 终了文本也要截断(Completed.text 不含停止串)
                act.trim_text_to = Some(*p);
            }
            (out, hit)
        };
        if let Some((_, s)) = stop_hit {
            // 事件序 = Token(若有回吐) → Completed(Stop);与预算尽同构
            owl_shared::log_debug!("engine", "turn {turn_id} 命中 stop 串 {:?},输出截断", s);
            let done_ev = self.complete(crate::turn::FinishReason::Stop).await?;
            if delta.is_empty() {
                return Ok(done_ev);
            }
            self.turns.pending_events.push_back(done_ev);
            return Ok(TurnEvent::Token { turn: turn_id, delta });
        }
        if budget_done {
            // 末 token 也要作为 Token 事件发出 —— 否则 (a) usage.completion
            // 少 1(服务端按 Token 事件计数),(b) 流式丢末段文本(2026-10-13
            // 修:实测 max_tokens=64 时 usage 报 63)。事件序 = Token →
            // Completed(pending 队列先出,见 running::pump)
            let done_ev = self.complete(crate::turn::FinishReason::Length).await?;
            self.turns.pending_events.push_back(done_ev);
            return Ok(TurnEvent::Token { turn: turn_id, delta });
        }
        Ok(TurnEvent::Token { turn: turn_id, delta })
    }



    /// ⚙ diag:把"当前槽所在页"的 KV 字节 d2h 落盘(层 0 的 k 池;写者足迹取证)。
    /// 落 `/tmp/kv-{tag}.bin` + `/tmp/kv-meta.txt`(blk/off/hkv/hd/page/字节数)。
    async fn dump_kv_page(&mut self, sid: u64, kv_slot: u32, tag: &str) -> Result<()> {
        let d = &self.pool.dims;
        let page = self.pool.page;
        let (hkv, hd) = (d.hkv, d.hd);
        // fp8 池 = 1B/元素(扁平 u8 账);f16 池 = 2B ✓
        let esize = if self.pool.kv_fp8 { 1 } else { d.dtype.size_bytes() };
        let row_bytes = hkv * hd * esize;
        let blk = {
            let s = self.bank.sessions.get(sid).expect("账在");
            let p = (kv_slot as usize) / page;
            s.block_table.get(p).copied().unwrap_or(0)
        };
        let pool_leaf = self.pool.kv_caches()[0].0.clone();
        let pool_id = match pool_leaf.op() {
            owl_models::ops::Op::Block { id } => *id,
            _ => return Ok(()),
        };
        let n = row_bytes * page;
        let face = self.graphs.session.face_mut();
        let scratch = face.alloc(owl_iface::contract::Dtype::U32, n / 4).await?;
        let src = owl_iface::contract::Bytes::new(
            pool_id,
            n / 4,
            owl_iface::contract::Dtype::U32,
            0,
        );
        face.copy_block_at(
            &src,
            blk as usize * n,
            &scratch,
            0,
            n,
        )
        .await?;
        let mut buf = vec![0u8; n];
        face.dtoh(&scratch, &mut buf).await?;
        let _ = face.free(&[scratch.id]).await;
        std::fs::write(format!("/tmp/kv-{tag}.bin"), &buf)
            .map_err(|e| owl_iface::contract::ModelError::Msg(format!("kv dump 落盘: {e}")))?;
        let meta = format!(
            "tag={tag} slot={kv_slot} blk={blk} off={} hkv={hkv} hd={hd} page={page} row_bytes={row_bytes} page_bytes={n}\n",
            kv_slot as usize % page
        );
        {
            use std::io::Write as _;
            let mut f = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open("/tmp/kv-meta.txt")
                .map_err(|e| owl_iface::contract::ModelError::Msg(format!("kv meta: {e}")))?;
            f.write_all(meta.as_bytes())
                .map_err(|e| owl_iface::contract::ModelError::Msg(format!("kv meta 写: {e}")))?;
        }
        Ok(())
    }

}

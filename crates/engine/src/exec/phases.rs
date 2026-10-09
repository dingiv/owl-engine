//! 调度三相位执行(begin/prefill/decode)+ 采样收口 + GDN 画像
use super::*;

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
        let nt = if self.cfg.knobs.sampler_enabled {
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
            // B6.3:fp8 池字节域同序(1B/elem,e4m3);A/B 池对拍仪表
            // (f16 boot 半值 vs fp8 boot 字节,host RNE 桥对)
            if self.probes.debug && step < 16 {
                let fp8 = self.pool.kv_fp8;
                let esz = if fp8 { 1usize } else { 2usize };
                let kv0 = &self.pool.kvs[0].k_cache;
                let mut whole = vec![0u8; kv0.1 * esz];
                let face = self.session.face_mut();
                face.dtoh(&kv0.0, &mut whole).await?;
                let probe = |slot: usize| -> usize {
                    let base = slot * 8 * esz;
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
                if fp8 {
                    let hex: String = whole[pos_now * 8..pos_now * 8 + 8]
                        .iter()
                        .map(|b| format!("{b:02x}"))
                        .collect();
                    eprintln!("[kv-hex] fp8 槽{pos_now} k[0..8] = {hex}");
                } else {
                    let hex: String = whole[pos_now * 16..pos_now * 16 + 16]
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
            let nt = crate::sampler::sample(&mut logits, &self.cfg.knobs.sampler, &mut rng, &history);
            if prof {
                eprintln!("[step-prof] pos={pos} host-sample={:?} wall3={}", t_sample.unwrap().elapsed(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() % 100000);
            }
            if self.probes.debug {
                eprintln!("[sample-dump] t{turn_id} s{step}: sampled={nt} hist={} pen={}", history.len(), self.cfg.knobs.sampler.rep_penalty);
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
        // B4 降级期草稿同步(§6.24 方案 A):decode 图 taps → 1 行 encode。
        // 毒化带 ≤8 token 即被覆写,探测重入零损;成本 ≈ 0.5ms/tok(2%)
        if self.tspec_ref().spec_degraded && self.dflash_tap_count > 0 {
            // 降级步分相(2026-10-12 探针完善):sync encode 单独计时 ——
            // 9.3 t/s 案的 80ms/步嫌疑犯,一屏定谳
            let t_syn = std::time::Instant::now();
            let dtype = self.pool.dims.dtype;
            let hidden = self.pool.dims.hidden;
            let taps: Vec<TensorOps> = (0..self.dflash_tap_count)
                .filter_map(|i| self.session.output_block(&format!("tap{i}")))
                .map(|b| TensorOps::of_block(b.id, dtype, vec![1, hidden]))
                .collect();
            if taps.len() == self.dflash_tap_count {
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
            let face = self.session.face_mut();
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
        let turn_id = self.active.as_ref().expect("active 已保证").id;
        {
            // 崩坏排查打点:首 token / EOS 事实(decode 与 spec 两路汇合)
            let bs = self.boot_seq;
            let first = self
                .active
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


}

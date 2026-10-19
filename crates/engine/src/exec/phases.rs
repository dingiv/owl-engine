//! 调度三相位执行(begin/prefill/decode)+ 采样收口 + GDN 画像
use super::prefill_chunk::ChunkHost;
use super::*;
use owl_models::ops::PlanNode; // ⚙ diag:池叶 → 块 id(写者足迹取证)
use owl_iface::contract::ModelError;

impl<D: DeviceClient> crate::running::RunningEngine<D> {
    /// turn 启动执行(设备侧照办):GDN 格重置 / 边界快照恢复 + 持久
    /// 块表重写(write_bt;图烘焙 bt 指针,内容重写指针稳定图不重捕)
    pub(crate) async fn execute_begin(&mut self, b: BeginPlan) -> Result<()> {
        // 性能面:本轮 epoch 起点(/debug/perf?epoch=1 = 只看本轮;不 reset,
        // 历史样本仍可全量查询 —— perf-framework.md §三)
        owl_shared::metrics::mark_epoch();
        // 快照就位(host 侧:store 层账 + 必要时一条 swap 回灌)——
        // **必须在设备半之前**:两 rank 的快照恢复都要用就位后的槽。
        let restore_slot = if !b.reset_gdn {
            match b.restore_key {
                Some(key) => Some(self.snap_ensure_resident(key).await?),
                None => None,
            }
        } else {
            None
        };
        // **S3-3b:待回灌块**(内容在宿主层)在段边界搬到设备,并把新槽写回块表 ——
        // 必须在 `write_bt` 之前(否则写进设备的是占位坏槽)。搬运实现全在 store。
        if !b.pending_loads.is_empty() {
            let t_stage = std::time::Instant::now();
            // §21:搬运由 engine 编排(取计划 → face.swap → 回执 → 记账);
            // **一条命令搬 N 块**(逐块 = 每块一次往返,回灌收益被吃光)。
            let chains: Vec<owl_store::ChainHash> =
                b.pending_loads.iter().map(|(_, c)| *c).collect();
            // **S6**:盘上的块先读回宿主(盘 → 设备要两跳;判据与账都在 store)
            let _ = self.ensure_host_resident(&chains).await?;
            let leg = &mut self.bank.rel.leg;
            let face = self.dev.graphs.session.face_mut();
            let (cs, new_slots, pieces, table) = {
                let p = owl_store::plan_onload(leg, &chains)?;
                (p.chains, p.slots, p.pieces, p.table)
            };
            let conn = match face
                .swap(owl_iface::contract::SwapDir::In, &pieces, Box::new(table))
                .await
            {
                Ok(c) => c,
                Err(e) => {
                    owl_store::abort_onload(leg, &cs, &new_slots);
                    return Err(owl_iface::contract::ModelError::Msg(format!(
                        "回灌 swap 失败(engine 编排):{e}"
                    )));
                }
            };
            let table = owl_store::reclaim_slot_table(conn)?;
            let slots = owl_store::commit_onload(leg, &cs, &new_slots, table)?;
            // **B-3e 实测账**:本条命令的块数/字节/墙钟(含发射 + 回执)
            self.stage.blocks += cs.len();
            self.stage.bytes += cs.len() * self.dev.pool.block_bytes();
            self.stage.nanos += t_stage.elapsed().as_nanos() as u64;
            self.stage.from_prefill += cs.len();
            let s = self.bank.sessions.get_mut(b.session_id).ok_or_else(|| {
                owl_iface::contract::ModelError::Msg(format!("回灌:会话 {} 不在账", b.session_id))
            })?;
            for ((idx, _), slot) in b.pending_loads.iter().zip(&slots) {
                if *idx >= s.block_table.len() {
                    return Err(owl_iface::contract::ModelError::Msg(format!(
                        "回灌下标 {idx} 越界(表长 {})",
                        s.block_table.len()
                    )));
                }
                s.block_table[*idx] = *slot;
            }
            owl_shared::log_debug!(
                "store",
                "段边界回灌 {} 块(session={};宿主→设备;一条命令)",
                b.pending_loads.len(),
                b.session_id
            );
        }
        let table = self
            .bank
            .sessions
            .session(b.session_id)?
            .block_table
            .clone();
        // 设备半(两 rank **并发**推进):GDN 重置/快照恢复 + 持久块表重写
        // (并发理由同 prefill:单边先行 ⇒ 对端 AR 撞 5s 自旋)
        let (reset_gdn, restore_key, gdn_slot) = (b.reset_gdn, b.restore_key, b.gdn_slot);
        {
            let (dev, peer) = (&mut self.dev, self.peer.as_mut());
            match peer {
                None => dev.begin_dev(reset_gdn, restore_key, restore_slot, gdn_slot, &table).await?,
                Some(p) => {
                    let (r0, r1) = join2(
                        dev.begin_dev(reset_gdn, restore_key, restore_slot, gdn_slot, &table),
                        p.begin_dev(reset_gdn, restore_key, restore_slot, gdn_slot, &table),
                    )
                    .await;
                    r0?;
                    r1?;
                }
            }
        }
        Ok(())
    }

    /// **prefill 收割半(主 rank)**:读 token + 记账 + seed 落地(host 写只在主 rank)
    async fn harvest_prefill(
        &mut self,
        tail: Option<crate::exec::prefill_chunk::PrefillTail>,
        boot_seq: u64,
    ) -> Result<Option<u32>> {
        let Some(t) = tail else { return Ok(None) };
        let h = self.dev.prefill_harvest(t, true).await?;
        mfact(boot_seq, "pf.last", h.token as u64);
        mfact(boot_seq, "pf.eos", self.tok.is_eos(h.token) as u64);
        if let Some(hb) = h.seed_hidden {
            self.tspec()?.spec_seed_hidden = Some(hb);
        }
        Ok(Some(h.token))
    }

    /// prefill chunk 执行(设备侧;块已在决策面切好)
    pub(crate) async fn execute_prefill(
        &mut self,
        chunk_ids: Vec<u32>,
        base: usize,
        is_last: bool,
    ) -> Result<TurnEvent> {
        let (turn_id, sid) = {
            let act = self.turns.active()?;
            (act.id, act.session_id)
        };
        // 调试门:prefill 块粒度(见 owl_shared::stepgate;`--step-gate` 控制)
        owl_shared::step_gate!("engine.prefill").await;
        // **锁①后半:块表按 chunk 增量长**(只在虚拟化档;2026-10-19)。
        //
        // 原行为:turn 起点一次性把块表按**整个 prompt** 长到位
        // (`scheduler.rs` 的 `ensure_for_len(prompt_len)`)—— 池 ≥ ctx 时从不暴露,
        // 但池 < ctx 时那一步就要一次性拿到几千页的槽 ⇒ **必 `Exhausted`**。
        // 虚拟化档改成"**用到哪长到哪**":本 chunk 要写的 `[base, base+t)` 落在
        // 哪几页就把表长到那几页,每步只增量 `chunk/page` 页;
        // 腾位由召回点的驻留调和(A/B 两轴)负责 —— 也正是超卖成立的方式。
        //
        // 幂等(`ensure_for_len` 只增不减)⇒ 非虚拟化档也调不坏(但为不改旧行为,
        // 只在本档执行)。
        if self.cfg.knobs.kvmem_virtualization {
            let need = base + chunk_ids.len();
            let s = self
                .bank
                .sessions
                .get_mut(sid)
                .ok_or_else(|| ModelError::Msg("prefill 块表增长:会话账不在场".into()))?;
            self.bank.rel.leg.ensure_for_len(&mut s.block_table, need)?;
        }
        // ── 主机侧事实(唯一取值点;两 rank 共用)────────────────────────
        let (bt_chain, gdn_slot) = {
            let act = self.turns.active()?;
            let s = self.bank.sessions.session(act.session_id)?;
            (s.block_table.clone(), act.gdn_slot)
        };
        // 部分驻留档的"过去"子集页链(host 面算好;两 rank 同值)
        let past_view = self.past_view_of(base, chunk_ids.len());
        let mut host0 = ChunkHost {
            boot_seq: self.boot_seq,
            probes: self.probes,
            knobs: &self.cfg.knobs,
            sid,
            bt_chain: &bt_chain,
            gdn_slot,
            bank: Some(&mut self.bank),
            prof: Some(&mut self.prof),
            cov: Some(&mut self.cov_pending),
            past_view: past_view.clone(),
        };
        let mut host1 = ChunkHost {
            boot_seq: self.boot_seq,
            probes: self.probes,
            knobs: &self.cfg.knobs,
            sid,
            bt_chain: &bt_chain,
            gdn_slot,
            bank: None, // peer:镜像路不给账房(草稿臂在本路响亮拒)
            prof: None, // peer:诊断账只在主 rank
            cov: None,
            past_view,
        };
        // ── ① 两 rank **并发**入队半(同一份设备侧代码,各自 rank 束)──────
        // ⚠️ 必须并发:单边先行 ⇒ 对端 AR 自旋 5s 超时 + 游标永久失步
        // (2026-10-17 G8 定谳;host 在入队半里存在设备等待点,串行必死)
        let boot_seq = host0.boot_seq;
        let tail0 = {
            let (dev, peer) = (&mut self.dev, self.peer.as_mut());
            match peer {
                None => dev.prefill_chunk_enqueue(&mut host0, &chunk_ids, base, is_last).await?,
                Some(p) => {
                    let (r0, r1) = join2(
                        dev.prefill_chunk_enqueue(&mut host0, &chunk_ids, base, is_last),
                        p.prefill_chunk_enqueue(&mut host1, &chunk_ids, base, is_last),
                    )
                    .await;
                    let tail0 = r0?;
                    let tail1 = r1?;
                    // peer 归块(不读 —— 读数只在主 rank)
                    if let Some(t1) = tail1 {
                        let _ = p.prefill_harvest(t1, false).await?;
                    }
                    tail0
                }
            }
        };
        // 主 rank 收割(**两 rank 都已入队之后**才读;borrow 已释放)
        let last_row = self.harvest_prefill(tail0, boot_seq).await?;
        let t_bnd = std::time::Instant::now();
        // E2c:块边界跨越 → GDN 快照拍摄(快照身份 = 边界块**内容身份**;
        // 物理块号会被回收复用,不可作键)
        let new_fed = base + chunk_ids.len();
        if self.dev.pool.paged && new_fed.is_multiple_of(self.dev.pool.page) {
            // GDN 快照拍摄(**唯一入口** `capture_gdn_snap`:链推进 + 增量登记 +
            // 索引采集 + 快照;内部已 TP 感知 —— peer 在场时两 rank 各拍各自的池,
            // store 的 state 账仍单份)。
            self.capture_gdn_snap(sid, new_fed).await?;
            // **边界相位拆分**(`OWL_VRAM_DBG=1`;2026-10-19 A3):
            // 边界 = 快照拍摄(链哈希 + GDN 拷贝 + 登记)‖ 索引采集 ‖ B 轴(打分/选择/视图/调和)。
            let dbg = std::env::var_os("OWL_VRAM_DBG").is_some();
            // ── B 轴选择层(生产面 + 可选诊断面)──
            // 必须**晚于**边界事件:索引/快照刚登记完,候选块才齐(§8.4)。
            // 两扇门:`select_decode`(生产)= 打分/选择/视图/调和;
            // `coverage_probe`(诊断)= 额外算真实质量与覆盖率读数。
            if self.cfg.knobs.coverage_probe || self.cfg.knobs.select_decode {
                // **末个边界**(≠ 末个 chunk):prompt 末尾未必页对齐,而边界只在
                // **chunk 末且页对齐**处发生 ⇒ 预测「下一条边界会不会来」用的是
                // 调度器**同一份**块长函数(`Scheduler::prefill_step_chunk`),
                // 而非在这里另写一套分块算术(接口单源)。
                let prompt_len = self
                    .turns
                    .active
                    .as_ref()
                    .map(|a| a.prompt_ids.len())
                    .unwrap_or(new_fed);
                let next_end = Self::prefill_step_chunk(
                    new_fed,
                    prompt_len,
                    self.cfg.prefill_chunk,
                    self.cfg.knobs.prefill_tail_merge,
                    Self::prefill_align_page(
                        self.cfg.knobs.prefill_align_tail,
                        self.cfg.knobs.prefix_cache,
                        self.dev.pool.page,
                    ),
                )
                .map(|n| new_fed + n)
                .unwrap_or(new_fed);
                let last_boundary =
                    next_end == new_fed || !next_end.is_multiple_of(self.dev.pool.page);
                let t_cov = std::time::Instant::now();
                self.coverage_analyze(sid, new_fed, last_boundary).await?;
                if dbg {
                    let cov_ms = t_cov.elapsed().as_secs_f64() * 1000.0;
                    eprintln!("[vram-b] 边界@{new_fed}: B 轴 {cov_ms:.0}ms");
                }
            } else if dbg {
                eprintln!("[vram-b] 边界@{new_fed}: 选择层关,无索引/B 轴");
            }
        }
        // **边界事件计时 → metrics**(2026-10-18 T1):前向已有 `prefill.chunk`
        // (prefill_chunk.rs 内部记),这里补“边界那一半”(快照 + 召回/打分/调和)
        // ⇒ “每 chunk 随 ctx 变慢”在 `/debug/perf` 里直接分两半,不再靠打印。
        mrec("prefill.bnd", t_bnd.elapsed());
        // **逐 chunk 显存轨迹**(`OWL_VRAM_DBG=1`;262k 冲刺的取证面):
        // live/free 测在**边界之后** —— 判“中间件是否归还”。计时已改走 metrics。
        if std::env::var_os("OWL_VRAM_DBG").is_some() {
            eprintln!(
                "[vram-c] chunk@{} t={} ⇒ live={} MiB · 驱动 free={} MiB",
                base,
                chunk_ids.len(),
                owl_shared::vram::live_bytes() >> 20,
                owl_shared::vram::free_bytes() >> 20,
            );
        }
        self.turns.active_mut()?.fed += chunk_ids.len();
        if !is_last {
            // 块落定,无文本产出 —— 不谎报 Idle
            let act = self.turns.active()?;
            return Ok(TurnEvent::Prefill {
                turn: turn_id,
                fed: act.fed,
                total: act.prompt_ids.len(),
            });
        }
        // 末块:末行采样(eos / 预算 / Token)
        self.sample_and_emit(last_row.ok_or_else(|| ModelError::EngineInvariant { what: "末块缺 logits(采样面与声明不一致)".into() })?).await
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
        let sid = self.turns.active()?.session_id;
        // 调试门:decode 步粒度
        owl_shared::step_gate!("engine.decode").await;
        let prof = self.probes.step_profile;
        let t0 = prof.then(std::time::Instant::now);
        let t_step = std::time::Instant::now();
        // E2b:块链跨页增长 → 重写持久块表(图烘焙 bt 指针)。
        // **B-3e-2**:视图在场时**不写**(本步页表走 `sel_pt` 输入槽;而链上可能有
        // 洞 = 占位符,写进设备块表就是把 u32::MAX 当页号 —— 守卫会响亮拒且白跑)。
        // 设备半(每 rank 各自写)在 `decode_dev` 里;这里只算出表内容。
        let grow_bt = (grew && self.sel_view.is_none()).then(|| {
            self.bank
                .sessions
                .get(sid)
                .expect("账在")
                .block_table
                .clone()
        });
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
        if self.dev.graphs.session.has_input("xqa_seq") {
            step_slots.push(("xqa_seq", vec![f32::from_bits((pos + 1) as u32)]));
            let n = self.dev.graphs.session.input_len("xqa_sem").unwrap_or(0);
            step_slots.push(("xqa_sem", vec![0.0; n]));
            // XQA 页表 = 本会话真页链(不喂 = 图里恒等链 = 读到别家 KV;§7.6)
            if self.dev.graphs.session.has_input("xqa_pt") {
                step_slots.push(("xqa_pt", self.xqa_pt_of(sid)));
            }
        }
        // **FI 图内臂每步 ingress**:**表四件套**(位型过线)。图里 `ctx_total`
        // 是**烘死的上界**(调度表形状),当步真值靠**表**(下面 `fi_plan_sync` 刷)。
        // ⚠️ 判据 = **槽在场**(与 xqa_seq 同款纪律):槽属于**图**;曾按"宿主标量
        // cell 在不在"判 ⇒ 单图档四件套一条不写 ⇒ 核读陈旧页表 ⇒ 输出 64 个空
        // token(2026-10-18 实测踩过)。
        if self.dev.graphs.session.has_input("fi_q_cu") {
            let ctx_total = pos + 1;
            // 四件套:真值(与承载无关 —— 图内/窗外都读同一批槽)
            let page = self.dev.pool.page;
            let bt_chain = {
                let s = self.bank.sessions.session(sid)?;
                s.block_table.clone()
            };
            // ⚠️ **响亮拒(2026-10-19 审查)**:`.min(bt_chain.len())` 只防"链过长",
            // 链过短时 `nb_used < ceil(ctx/page)` ⇒ `fi_last_len > page`(结构非法)
            // ⇒ 核越界读页链。与 prefill 侧 `fi_tables` 同型,一并改成结构化 Err。
            let need = ctx_total.div_ceil(page.max(1)).max(1);
            if bt_chain.len() < need {
                return Err(ModelError::Msg(format!(
                    "decode FI 表失守:块链 {} 页 < 本步需要 {need} 页(pos={pos} page={page} ctx={ctx_total})",
                    bt_chain.len()
                )));
            }
            let nb_used = need;
            let mut idx = vec![0f32; self.dev.pool.nb];
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
        // **B-3d 选择层视图**:`select_decode` 档 ⇒ 页表与上下文长走视图输入槽
        // (视图 = 选中页 ++ 新生成尾部;无视图态 ⇒ 写全量链,与不开时等价)
        if self.dev.graphs.session.has_input("sel_pt") {
            let (pt, len) = self.select_view_step(sid, pos);
            step_slots.push(("sel_pt", pt));
            step_slots.push(("sel_len", vec![len as f32]));
        }
        let step_refs: Vec<(&str, &[f32])> =
            step_slots.iter().map(|(n, v)| (*n, v.as_slice())).collect();
        // ⚙ KV 页字节差分(OWL_KV_DUMP=<pos>,一次):写前/写后各 d2h 一页 ⇒
        // **写者的足迹**直接给出块内元素序(页面连续 vs 8B 小块 strided),
        // 不需要任何参考值(见 docs §11.11「写者 A/B」)。
        let kv_dump_at = self.probes.kv_dump_at;
        if kv_dump_at == Some(pos) {
            self.dump_kv_page(sid, kv_slot, "before").await?;
        }
        // 设备半(两 rank **并发**):块表重写(跨页)+ FI plan 表同步 + 会话图入队
        // —— **零同步**(并发理由同 prefill:单边先行 ⇒ 对端 AR 5s 自旋超时)
        //
        // **FI plan 表同步**(W3;`[dispatch] fi_plan_bake` 档):图内 FI 核的回放
        // **不经过宿主处理器** ⇒ 表(`int_ws`)必须由图外每步显式刷新;否则它读到的
        // 是别的写者(如逐算子 eager 的 prefill 分块)留下的表,而 split 档的表
        // 必须与**当步真 ctx** 对应(实测症状:输出 64 个空 token)。
        // 条件 = 档位在场 + 本图确实带 FI 输入槽(臂身份自证,不猜)。
        let fi_bake = self.cfg.knobs.fi_plan_bake;
        let (dev, peer) = (&mut self.dev, self.peer.as_mut());
        match peer {
            None => dev.decode_dev(&step_refs, grow_bt.as_deref(), fi_bake, pos).await?,
            Some(p) => {
                let (r0, r1) = join2(
                    dev.decode_dev(&step_refs, grow_bt.as_deref(), fi_bake, pos),
                    p.decode_dev(&step_refs, grow_bt.as_deref(), fi_bake, pos),
                )
                .await;
                r0?;
                r1?;
            }
        }
        if kv_dump_at == Some(pos) {
            self.dump_kv_page(sid, kv_slot, "after").await?;
        }
        if prof {
            owl_shared::log_info!("step-prof", "pos={pos} fill+launch={:?} wall={}", t0.unwrap().elapsed(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() % 100000);
        }
        if prof {
            let total = t0.unwrap().elapsed();
            owl_shared::log_info!("step-prof", "pos={pos} step-total={total:?} wall2={}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() % 100000);
        }
        self.turns.active_mut()?.fed += 1;
        // E2c:decode 跨页边界 → 快照拍摄(同流保序,先拍再采样)
        //
        // **档 = Transient**(2026-10-19):生成段边界链一往前就被取代 ⇒ 逐出即丢,
        // 不下沉宿主(下沉一条 159 MiB ≈ 50 ms;生成段每 32 tok 就出一条 ——
        // 旧档(可沉)曾把 prefill 检查点顶出去并制造 PCIe 尖刺,见 §9.1(3))。
        // 链头那条永远最新 ⇒ 永远在设备(LRU)⇒ 复用面无损。
        {
            let act = self.turns.active()?;
            if self.dev.pool.paged && act.fed.is_multiple_of(self.dev.pool.page) {
                // 同 prefill:唯一入口 `capture_gdn_snap`(内部 TP 感知);
                // **档 = Transient** —— 生成段边界链一往前就被取代
                // (见 §9.1(7)/(8):下沉一条 159 MiB 实测 ≈ 100ms,不值得)
                self.capture_gdn_snap_tail(
                    act.session_id,
                    act.fed,
                    &[],
                    owl_store::SnapDurability::Transient,
                )
                .await?;
            }
        }
        // D1 取证:GDN 状态格画像(OWL_GDN_DUMP;前 6 步,首末层 conv_v/rec)
        if self.probes.gdn_dump {
            let step = self.turns.active()?.out.len() as u64;
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
        let eff_sampler = self.turns.active()?.sampler;
        let sampler_on = self.turns.active()?.sampler_on;
        let nt = if sampler_on {
            let mut logits = self.dev.graphs.session.read_output_f32("logits").await?;
            // 诊断:分布形状(绝对尺度 / top1-top2 gap / 候选词面)——
            // 复读病理定位的仪表盘(OWL_DEBUG 门控,前 16 步)
            let (turn_id, step) = {
                let act = self.turns.active()?;
                (act.id, act.out.len() as u64)
            };
            // 诊断:KV 池回读(取证图内 K0 写是否落盘)—— 槽 2 = prompt
            // (eager prefill 写);槽 pos = 本步 K0(图内写)。
            // 回读偏移走**布局单源**(`env::knhd_offset`;P6-A 后单布局,
            // 原 classic x 交错分支已删)。头 0 / 维 0..8。
            // B6.3:fp8 池字节域同序(1B/elem,e4m3);A/B 池对拍仪表
            // (f16 boot 半值 vs fp8 boot 字节,host RNE 桥对)
            if self.probes.debug && step < 16 {
                let fp8 = self.dev.pool.kv_fp8;
                let esz = if fp8 { 1usize } else { 2usize };
                let kv0 = &self.dev.pool.kvs[0].k_cache;
                let mut whole = vec![0u8; kv0.1 * esz];
                let face = self.dev.graphs.session.face_mut();
                face.dtoh(&kv0.0, &mut whole).await?;
                let (hkv, hd) = (self.dev.pool.dims.hkv, self.dev.pool.dims.hd);
                let _page = self.dev.pool.page;
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
                owl_shared::log_debug!(
                    "kv-dump", "fp8={fp8} pos={pos_now} 槽2非零={}/8 槽{pos_now}非零={}/8 槽{}非零={}/8",
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
                    owl_shared::log_debug!("kv-hex", "fp8 槽{pos_now} k[0..8] = {hex}");
                } else {
                    let hex: String = whole[base_now..base_now + 16]
                        .iter()
                        .map(|b| format!("{b:02x}"))
                        .collect();
                    owl_shared::log_debug!("kv-hex", "槽{pos_now} k[0..8] = {hex}");
                }
            }
            if self.probes.debug && step < 16 {
                let mut top: Vec<(f32, u32)> = logits
                    .iter()
                    .enumerate()
                    .filter(|(_, &v)| v.is_finite())
                    .map(|(i, &v)| (v, i as u32))
                    .collect();
                top.select_nth_unstable_by(7, |a, b| b.0.total_cmp(&a.0));
                top.truncate(8);
                top.sort_by(|a, b| b.0.total_cmp(&a.0));
                let pretty: Vec<String> = top
                    .iter()
                    .map(|(v, i)| {
                        format!("{:.2}:'{}'", v, self.tok.decode(&[*i]).replace('\n', "\\n"))
                    })
                    .collect();
                owl_shared::log_debug!("logits-dump", "t{turn_id} s{step}: {}", pretty.join(" | "));
            }
            // RNG seed = turn id ⊕ 步数(同 turn 重放确定性,跨 turn 独立);
            // 请求给了 `seed` ⇒ 以它为基(AGT-06)
            let seed_base = self.turns.active()?
                .seed
                .unwrap_or_else(|| turn_id.wrapping_mul(0x9E3779B97F4A7C15));
            let mut rng = seed_base ^ step.wrapping_mul(0xBF58476D1CE4E5B9);
            // 惩罚集 = prompt 尾(64)+ 已生成(反循环;llama.cpp 同语义)
            let history: Vec<u32> = {
                let act = self.turns.active()?;
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
                owl_shared::log_info!("step-prof", "pos={pos} host-sample={:?} wall3={}", t_sample.unwrap().elapsed(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() % 100000);
            }
            if self.probes.debug {
                owl_shared::log_debug!("sample-dump", "t{turn_id} s{step}: sampled={nt} hist={} pen={}", history.len(), eff_sampler.rep_penalty);
            }
            nt
        } else if self.dev.env.diag.host_argmax {
            let logits = self.dev.graphs.session.read_output_f32("logits").await?;
            logits.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |a, (i, &v)| {
                if v > a.1 { (i, v) } else { a }
            })
            .0 as u32
        } else {
            // D1 取证:greedy 路径 logits 形状仪表(OWL_DEBUG;前 16 步,与
            // 采样分支同格式 —— 基线/融合 A/B 对比用)
            if self.probes.debug {
                let (turn_id, step) = {
                    let act = self.turns.active()?;
                    (act.id, act.out.len() as u64)
                };
                if step < 16 {
                    let logits = self.dev.graphs.session.read_output_f32("logits").await?;
                    let mut top: Vec<(f32, u32)> = logits
                        .iter()
                        .enumerate()
                        .filter(|(_, &v)| v.is_finite())
                        .map(|(i, &v)| (v, i as u32))
                        .collect();
                    top.select_nth_unstable_by(7, |a, b| b.0.total_cmp(&a.0));
                    top.truncate(8);
                    top.sort_by(|a, b| b.0.total_cmp(&a.0));
                    let pretty: Vec<String> = top
                        .iter()
                        .map(|(v, i)| {
                            format!("{:.2}:'{}'", v, self.tok.decode(&[*i]).replace('\n', "\\n"))
                        })
                        .collect();
                    owl_shared::log_debug!("logits-dump", "t{turn_id} s{step}: {}", pretty.join(" | "));
                }
            }
            let tok_f = self.dev.graphs.session.read_output_f32("token").await?;
            tok_f[0] as u32
        };
        if prof {
            owl_shared::log_info!("step-prof", "pos={pos} token-dtoh={:?}", t_dtoh.unwrap().elapsed());
        }
        mrec("decode.step", t_step.elapsed());
        // B4 降级期草稿同步(§6.24 方案 A):decode 图 taps → 1 行 encode。
        // 毒化带 ≤8 token 即被覆写,探测重入零损;成本 ≈ 0.5ms/tok(2%)
        // 2026-10-17:**播种裸步**走同一臂(两者同构:都要同步草稿池 + 计时进分母)
        let degraded = self.tspec_ref()?.spec_degraded;
        let seeding = self.tspec_ref()?.bare_seed_step;
        if (degraded || seeding) && self.dev.spec_rt.dflash_tap_count > 0 {
            // 降级步分相(2026-10-12 探针完善):sync encode 单独计时 ——
            // 9.3 t/s 案的 80ms/步嫌疑犯,一屏定谳
            let t_syn = std::time::Instant::now();
            let dtype = self.dev.pool.dims.dtype;
            let hidden = self.dev.pool.dims.hidden;
            let taps: Vec<TensorOps> = (0..self.dev.spec_rt.dflash_tap_count)
                .filter_map(|i| self.dev.graphs.session.output_block(&format!("tap{i}")))
                .map(|b| TensorOps::of_block(b.id, dtype, vec![1, hidden]))
                .collect();
            if taps.len() == self.dev.spec_rt.dflash_tap_count {
                self.dev.prefill_dflash_encode(self.probes, &taps, pos).await?;
            }
            mrec("decode.sync", t_syn.elapsed());
            if degraded {
                self.tspec()?.spec_steps_degraded += 1;
            }
            // 滚动裸步成本 ewma(动态盈亏线分母;α=0.15)
            {
                let bc = t_step.elapsed().as_secs_f32() * 1000.0;
                let spec = self.tspec()?;
                let was_unmeasured = spec.bare_cost_ewma_ms <= 0.0;
                spec.bare_cost_ewma_ms = if was_unmeasured {
                    bc
                } else {
                    spec.bare_cost_ewma_ms * 0.85 + bc * 0.15
                };
                // 采样时的 ctx(换档失效判据;见 SpecDegrade.bare_cost_ctx)
                spec.bare_cost_ctx = pos;
                if was_unmeasured {
                    owl_shared::log_info!("spec-seed", "分母已播种:裸步成本 = {bc:.1}ms(pos={pos})⇒ 盈亏线改用实测比(原兜底常量 2.5)"
                    );
                }
            }
            mrec(if degraded { "decode.step.deg" } else { "decode.step.seed" }, t_step.elapsed());
            self.tspec()?.bare_seed_step = false;
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
            let face = self.dev.graphs.session.face_mut();
            self.dev.pool.gdn_slot_profile(face, gdn_slot, step, self.cfg.knobs.gdn_dump_all).await?
        };
        for l in lines {
            // GDN 画像行(probes.gdn_dump 门内;行体自含格式,只挂 tag)
            owl_shared::log_debug!("gdn-dump", "{l}");
        }
        Ok(())
    }


    /// 采样 + 事件产出(Greedy;E3 设备 argmax,token id 直入)。
    /// eos 命中或预算尽 → Completed;否则 Token(delta = 解码文本增量)。
    pub(crate) async fn sample_and_emit(&mut self, nt: u32) -> Result<TurnEvent> {
        let turn_id = self.turns.active()?.id;
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
            let act = self.turns.active_mut()?;
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
            let act = self.turns.active_mut()?;
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
        let d = &self.dev.pool.dims;
        let page = self.dev.pool.page;
        let (hkv, hd) = (d.hkv, d.hd);
        // fp8 池 = 1B/元素(扁平 u8 账);f16 池 = 2B ✓
        let esize = if self.dev.pool.kv_fp8 { 1 } else { d.dtype.size_bytes() };
        let row_bytes = hkv * hd * esize;
        let blk = {
            let s = self.bank.sessions.session(sid)?;
            let p = (kv_slot as usize) / page;
            s.block_table.get(p).copied().unwrap_or(0)
        };
        let pool_leaf = self.dev.pool.kv_caches()[0].0.clone();
        let pool_id = match pool_leaf.op() {
            owl_models::ops::Op::Block { id } => *id,
            _ => return Ok(()),
        };
        let n = row_bytes * page;
        let face = self.dev.graphs.session.face_mut();
        // 记名(全量):KV 页 dump 探针的临时块(按 tag 区分调用点)
        let scratch = face
            .alloc(owl_iface::contract::Dtype::U32, n / 4, &format!("probe.kv_page.{tag}"))
            .await?;
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

impl<D: DeviceClient> crate::engine::RankDev<D> {
    /// **begin 设备半**(S4 ⑤-K3):GDN 重置 / 快照恢复 / 块表重写 ——
    /// 现役字段 = 当前 rank 束(peer 镜像时已换装)。零读数(TP 镜像块契约)。
    ///
    /// `restore_slot` = 快照**已就位**的槽号(host 侧 `snap_ensure_resident` 先做过
    /// store 层账/必要时回灌 —— 设备半只做 D2D 恢复,不碰 host 账)。
    pub(crate) async fn begin_dev(
        &mut self,
        reset_gdn: bool,
        restore_key: Option<owl_store::SnapKey>,
        restore_slot: Option<u32>,
        gdn_slot: usize,
        table: &[u32],
    ) -> Result<()> {
        if reset_gdn {
            let face = self.graphs.session.face_mut();
            self.pool.reset_gdn(face, gdn_slot).await?;
        } else if let (Some(key), Some(slot)) = (restore_key, restore_slot) {
            // 前缀命中:恢复边界快照到会话格(免重灌;GDN 态对齐)
            let face = self.graphs.session.face_mut();
            self.pool.restore_snap(face, gdn_slot, key, slot).await?;
        }
        let face = self.graphs.session.face_mut();
        self.pool.write_bt(face, table).await
    }

    /// **decode 设备半**(S4 ⑤-K3):块表重写(跨页)+ FI plan 表同步 + 会话图入队
    /// —— **零同步**(TP 镜像块契约;读数全在采样面)。
    ///
    /// `fi_bake` = `[dispatch] fi_plan_bake` 档(每 rank 各自的图都要同步表);
    /// `ctx_pos` = 本步位置(表必须与**当步真 ctx** 对应)。
    pub(crate) async fn decode_dev(
        &mut self,
        refs: &[(&str, &[f32])],
        bt: Option<&[u32]>,
        fi_bake: Option<usize>,
        ctx_pos: usize,
    ) -> Result<()> {
        if let Some(table) = bt {
            let face = self.graphs.session.face_mut();
            self.pool.write_bt(face, table).await?;
        }
        // **FI plan 表同步**(W3):图内 FI 核的回放**不经过宿主处理器** ⇒ 表必须由
        // 图外每步显式刷新(否则读到别的写者留下的表 ⇒ 输出 64 个空 token,实测踩过)。
        if fi_bake.is_some() && self.graphs.session.has_input("fi_q_cu") {
            let d = self.pool.dims;
            let spec = owl_iface::contract::FiPlanSpec {
                t: 1,
                ctx_total: ctx_pos + 1,
                hq: d.hq,
                hkv: d.hkv,
                hd: d.hd,
                page: self.pool.page,
            };
            self.graphs.session.face_mut().fi_plan_sync(spec).await?;
        }
        self.graphs.session.step(refs).await
    }
}

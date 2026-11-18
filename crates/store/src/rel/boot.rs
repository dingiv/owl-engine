//! `Rel` 的装配 / boot 显存计划 / 轻量几何访问器。

use super::*;

impl Rel {
    /// 装配(唯一入口:预算分解失败 = **拒绝启动**,A5.1)。
    ///
    /// `kv_cfg` = KV 腿几何(含宿主层预算) —— 每 token KV 字节由它推
    /// (`block_bytes / block_tokens`),与 `inputs.kv_bytes_per_token`
    /// **交叉校验**(同一件事只有一个入口;矛盾 = 装配错,响亮拒)。
    pub fn open(
        inputs: BudgetInputs,
        policy: AdmissionPolicy,
        kv_cfg: LegConfig,
        ledger: crate::tier::SharedLedger,
    ) -> StoreResult<Self> {
        Self::open_world(inputs, policy, kv_cfg, ledger, 1)
    }

    /// **TP 档装配**(`world = 2`):KV 腿与状态腿的宿主槽都按 `world` 个平面开
    /// (`ram_pool_size` 语义不变 = **两 rank 合计**;单链字节 = `world ×` 单页)。
    ///
    /// 搬运面契约:engine 对每个 rank 各发一条 `swap`(片偏移用
    /// [`crate::pieces_on_plane`]),槽台在两条命令之间**原箱传递** ⇒ 一次
    /// 借台/一次落账(预算与账目仍是一条链一份)。
    pub fn open_world(
        inputs: BudgetInputs,
        policy: AdmissionPolicy,
        kv_cfg: LegConfig,
        ledger: crate::tier::SharedLedger,
        world: usize,
    ) -> StoreResult<Self> {
        if world == 0 {
            return Err(StoreError::Invariant("Rel::open_world: world = 0".into()));
        }
        if kv_cfg.block_bytes == 0 || kv_cfg.block_tokens == 0 || kv_cfg.slots == 0 {
            return Err(StoreError::Budget(format!(
                "KV 腿几何非法:slots={} block_tokens={} block_bytes={}",
                kv_cfg.slots, kv_cfg.block_tokens, kv_cfg.block_bytes
            )));
        }
        let per_token = kv_cfg.block_bytes / kv_cfg.block_tokens;
        if per_token != inputs.kv_bytes_per_token {
            return Err(StoreError::Budget(format!(
                "每 token KV 字节不一致:腿几何算得 {per_token} ≠ 预算入参 {}(装配参数矛盾)",
                inputs.kv_bytes_per_token
            )));
        }
        if kv_cfg.block_tokens != inputs.page_tokens {
            return Err(StoreError::Budget(format!(
                "页宽不一致:腿几何 block_tokens={} ≠ 预算入参 page_tokens={}\
                 (地板按页口径算、链增长也按页 —— 两把尺必须同一把)",
                kv_cfg.block_tokens, inputs.page_tokens
            )));
        }
        // host 预算只有一个总量(`inputs.host_bytes` = 整个 store 的中层预算),腿拿其中
        // 一段 ⇒ 腿那段不许超过总量(超了就是两套规范,不许静默对不上)。
        if kv_cfg.host_bytes > inputs.host_bytes {
            return Err(StoreError::Budget(format!(
                "KV 腿 host 预算 {} > store host 总量 {}(中层分片不许超总账)",
                kv_cfg.host_bytes, inputs.host_bytes
            )));
        }
        let budget = VramBudget::decompose(inputs, policy.gdn_slots)?;
        // **账本预算交叉校验**(2026-10-19):账本由 boot 先造(DeviceHandle 要拿着它
        // 包住装载/捕获期的分配),这里**必须**与预算分解算出的三层预算**逐字相符** ——
        // 不符 = 两处算式漂了(响亮拒,不是静默采信任何一方)。
        {
            let want = budget.tier_budget();
            let got = crate::tier::book_lock(&ledger).tiers().budget();
            if got != want {
                return Err(StoreError::Budget(format!(
                    "账本预算与分解不符:账本 {got:?} ≠ 分解 {want:?}(boot 的算式与 `BudgetInputs` 漂了)"
                )));
            }
        }
        let state = StateStore::new(policy.gdn_slots, inputs.gdn_row_bytes);
        // **宿主 = 一层一个总池**（2026-11-18 用户裁决）：KV 与快照共享
        // `inputs.host_bytes`，不再从 `kv_cfg.host_bytes` 里预切状态腿分片。
        // 两条腿各自保留驻留表/键空间，但物理预算记在同一本 `SharedTierBudget`。
        let host_budget = crate::shared_budget::SharedTierBudget::new(inputs.host_bytes);
        Ok(Self {
            leg: BlockLeg::open_planes_shared(kv_cfg, world, host_budget.clone())?,
            index: None,
            state,
            snap_host: HostTier::new_shared(1, host_budget)?,
            disk: None,
            snap_disk: None,
            ledger,
            policy,
            admission: Admission::new(),
            reservations: ReservationLedger::new(),
            budget,
            evict: Box::new(LruPolicy),
            bulk_drop_when_tier_full: false,
            snap_layout: None,
            snap_layout_peer: None,
            snap_bytes: 0,
            world,
            plan: None,
        })
    }

    /// **注入 boot 显存计划**(V0/V1):投影必须与 `Rel::open` 的预算入参逐字一致,
    /// 否则就是“plan 一套、准入另一套” —— 响亮拒,不静默采信任何一方。
    pub fn attach_memory_plan(&mut self, plan: MemoryPlan) -> StoreResult<()> {
        let projected = plan.to_budget_inputs()?;
        let opened = self.budget.inputs();
        if projected != opened {
            return Err(StoreError::Invariant(format!(
                "MemoryPlan 投影与 Rel 预算入参不符:plan={projected:?} rel={opened:?}"
            )));
        }
        self.plan = Some(plan);
        Ok(())
    }

    /// 本 store 的 boot 显存计划(未注入 = None)。
    pub fn memory_plan(&self) -> Option<&MemoryPlan> {
        self.plan.as_ref()
    }

    /// 宿主腿平面数(= TP world;1 = 单卡)。
    pub fn world(&self) -> usize {
        self.world
    }
}

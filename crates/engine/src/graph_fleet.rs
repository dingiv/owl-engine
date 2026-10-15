//! 图族域(2026-10-14 拆分自 running.rs:RunningEngine 万能结构体按
//! 状态域拆解 —— 本模块收编**全部 GraphPlan 实例**:主会话图 + spec
//! 三族图桶形阵 + 捕获三态结果)。
//!
//! 2026-10-15 第二刀:`Engine::run()` 的四套图装配(主会话图 forward
//! 闭包 / verify 桶形图 / fold+propose 族 / DFlash2 propose 单图)整体
//! 下沉到本模块 —— run() 只留编排,图拓扑知识全部封装在图族域。
//!
//! 图生命周期纪律(charter A1):实例化后禁 trim/池属性变更;相位
//! 定相(perf tag 维度)须同步全部图 —— [`GraphFleet::set_phase`] 是
//! 唯一扇出点,禁止散点逐图设置(漏一张 = perf 归因错相)。

use std::sync::Arc;

use owl_iface::contract::{Bytes, DeviceClient, Dtype};
use owl_models::env::EnvProvider;
use owl_models::layers::gdn::GdnBuffers;
use owl_models::module::{ForwardCtx, KvBuffers, Module};
use owl_models::model::Model;
use owl_models::tensor::TensorOps;

use crate::engine::EngineKnobs;
use crate::graph_plan::{GraphPlan, GraphPlanDesc, InputSlot, OutputSlot, PlanCtx, PlanOutcome};
use crate::spec_runtime::{Drafter, SpecRuntime};
use crate::state::{ModelDims, StatePool};
use crate::Result;

type T = std::time::Instant;

/// 图族(组合叶;RunningEngine 持有)。字段 pub(crate) 供 exec 直读
/// (`graphs.session.face_mut()` 等),相位扇出走方法。
pub(crate) struct GraphFleet<D: DeviceClient> {
    /// 主会话图(prefill/decode 主路径;槽 = frontier/pos/kv_len/…)
    pub session: GraphPlan<D>,
    /// 捕获三态结果(Captured = 回放态;EagerFallback = 同闭包直发)
    pub capture_outcome: PlanOutcome,
    /// verify 桶形图(E5-M4;T = depth+1 定形,输出 tok/hid + fold 记录;
    /// None = eager fallback 走旧 verify+重放路)
    pub verify: Option<GraphPlan<D>>,
    /// fold 桶形图族(E5-M5;index = m,状态 = spec 快照 buf 就地,
    /// slot 恒 0;重放后 restore 回写会话格)
    pub fold: Vec<GraphPlan<D>>,
    /// propose 桶形图族(E5-M5;index = m,extend m+1 行 + k-1 链步,
    /// 输出 drafts;首轮 propose_first 保持 eager)
    pub propose: Vec<GraphPlan<D>>,
    /// DFlash2 propose 单图(E5-DF4;encode 全 8 行 + 噪声块 + selector,
    /// 固定几何;None = eager 回退)
    pub dflash: Option<GraphPlan<D>>,
    /// **prefill 桶族(M2b)**:一桶一张整段单图(含 KV 写 + FI 注意力);
    /// 空 = 档位 off / 未建(该档全走 eager 原路)
    pub prefill: crate::prefill_plan::PrefillBuckets<D>,
}

impl<D: DeviceClient> GraphFleet<D> {
    /// 相位定相扇出(唯一入口;引擎 env 快照与各图 env 同步 ——
    /// GraphPlan 构造期只拿到 env 快照 ⇒ 逐相位显式下发,
    /// perf-framework.md §十.3)。
    pub(crate) fn set_phase(&mut self, p: owl_models::env::EvalPhase) {
        self.session.set_phase(p);
        self.prefill.set_phase(p);
        if let Some(g) = self.verify.as_mut() {
            g.set_phase(p);
        }
        for g in self.fold.iter_mut() {
            g.set_phase(p);
        }
        for g in self.propose.iter_mut() {
            g.set_phase(p);
        }
        if let Some(g) = self.dflash.as_mut() {
            g.set_phase(p);
        }
    }
}

// ===========================================================================
// 装配面(run() 委托;2026-10-15 自 engine.rs 下沉)
// ===========================================================================

/// 装配上下文(boot 侧只读事实的显式快照 —— 图装配零隐式依赖;
/// `pool` 借用:装配完成后 run() 仍要把 pool 移交 RunningEngine)
pub(crate) struct FleetBoot<'a> {
    /// 解释器执行环境(捕获期烘焙进各图闭包;EnvProvider: Copy)
    pub(crate) env: EnvProvider,
    /// 模型几何(hq/hkv/…/hidden/dtype;Copy)
    pub(crate) dims: ModelDims,
    pub(crate) pool: &'a StatePool,
    pub(crate) model: Arc<Model>,
    pub(crate) rope: owl_models::layers::rope::Rope,
    /// 引擎级 spec 运行态(模式/深度/草稿器/草稿 rope/tap 开关)
    pub(crate) spec: &'a SpecRuntime,
    /// boot 旋钮(no_graph/ts_probe/capture_slab/debug/draft.notaps/tapdecl)
    pub(crate) knobs: &'a EngineKnobs,
}

impl<D: DeviceClient> GraphFleet<D> {
    /// **图族总装配**(顺序有语义,勿重排 —— `extra_faces` 是 pop 消费的
    /// 桶,fold 先 pop、dflash 次之、propose 末尾):
    /// ① 主会话图(warmup+捕获)→ ② verify 桶形图(spec 且 ① Captured)
    /// → ③ fold 族 / DFlash2 propose 单图 / MTP propose 族(全部依赖
    /// verify 图的输出块)。
    ///
    /// 降级语义(A1.4):任何一环捕获失败只降级该环(对应回 eager 路),
    /// 不连锁;`capture_outcome` 只记主会话图(它决定回放态/直发态)。
    pub(crate) async fn assemble(
        face: D,
        vface: Option<D>,
        mut extra_faces: Vec<D>,
        boot: FleetBoot<'_>,
        t0: T,
    ) -> Result<Self> {
        // ① 主会话图(先留一个句柄给 ④ 的 prefill 桶族 —— 桶族在最后建,
        // 那时 `face`/`vface` 都已被消费)
        let pface_src = face.face_clone();
        let (session, capture_outcome) = Self::plan_session(face, &boot, t0).await?;
        // ② verify 桶形图(spec 模式且主图捕获成功才有意义)
        let verify = if boot.spec.depth > 0
            && vface.is_some()
            && matches!(capture_outcome, PlanOutcome::Captured)
        {
            let vface = vface.expect("vface.is_some() 已守门");
            Self::plan_verify(vface, &boot, t0).await?
        } else {
            None
        };
        // ③ fold/DFlash2/MTP 族(全部以 verify 图输出块为烘焙叶)
        let (fold, dflash, propose) = if let Some(vg) = verify.as_ref() {
            let fold = Self::plan_fold_family(vg, &mut extra_faces, &boot, t0).await?;
            let dflash = Self::plan_dflash(vg, &mut extra_faces, &boot, t0).await?;
            let propose = Self::plan_propose_family(vg, &mut extra_faces, &boot, t0).await?;
            (fold, dflash, propose)
        } else {
            (Vec::new(), None, Vec::new())
        };
        // ④ **prefill 桶族(M2b)**:最后建 —— 自己的槽面/自己的输出块,
        // 与前面几族无共享羁绊;必须在家族 scratch 只增不减的律下**降序**捕
        // (桶内部自己排;见 `prefill_plan` 模块头注硬纪律 #2)。
        let prefill = if boot.knobs.prefill_bucket == owl_shared::config::PrefillBucket::Off {
            crate::prefill_plan::PrefillBuckets::empty()
        } else if boot.knobs.no_graph {
            owl_shared::log_warn!(
                "prefill-graph",
                "[graph] no_graph = true ⇒ prefill 桶不建(桶属图族;要诊断桶请去掉该总闸)"
            );
            crate::prefill_plan::PrefillBuckets::empty()
        } else if boot.knobs.env.attn.xqa_prefill {
            // XQA prefill 臂在场 ⇒ 不建桶:桶图的槽面是 FI 四件套,XQA 要的是
            // 页链/整请求长/位掩码三件 ⇒ 图内走 FI、eager 走 XQA(图 ≢ eager)。
            // 响亮告知(不是静默降级)。
            owl_shared::log_warn!(
                "prefill-graph",
                "dispatch.xqa_prefill 在场 ⇒ prefill 桶不建(桶图槽面 = FI 四件套;回 eager 原路)"
            );
            crate::prefill_plan::PrefillBuckets::empty()
        } else if boot.pool.fi_face {
            match pface_src {
                None => {
                    owl_shared::log_warn!(
                        "prefill-graph",
                        "face_clone 不可用 ⇒ prefill 桶不建(回 eager 原路)"
                    );
                    crate::prefill_plan::PrefillBuckets::empty()
                }
                Some(pf) => {
                    let boot_pg = crate::prefill_plan::BucketBoot {
                        env: boot.env,
                        model: Arc::clone(&boot.model),
                        rope: boot.rope.clone(),
                        pool: boot.pool,
                        dims: boot.dims,
                        taps: boot.spec.dflash_tap_count,
                        ladder: crate::prefill_plan::ladder(boot.knobs.prefill_bucket),
                    };
                    let buckets = crate::prefill_plan::PrefillBuckets::capture(&pf, &boot_pg).await;
                    owl_shared::log_info!(
                        "prefill-graph",
                        "桶族摘要:t_pad 命中阶梯 = {}",
                        buckets.summary()
                    );
                    buckets
                }
            }
        } else {
            owl_shared::log_warn!(
                "prefill-graph",
                "非 FI 档(pool.fi_face = false)⇒ prefill 桶不建(回 eager 原路)"
            );
            crate::prefill_plan::PrefillBuckets::empty()
        };
        Ok(Self { session, capture_outcome, verify, fold, propose, dflash, prefill })
    }

    /// ① 主会话图(槽 → 整模单树;状态句柄捕获,模型 Arc 共享)。
    /// 捕获期禁 Htod:常量标量走持久槽;KV/GDN 双槽分立 ——
    /// KV 槽 = 本 token 自己的格子(窗 = [slot-kv_len+1, slot],随步进;
    /// 2026-09-26 窗口语义探针定谳:slots 恒 0 读块外垃圾行);
    /// GDN 槽 = 序列状态格,turn 内恒 0(递推状态按序列累积)。
    /// **face 句柄克隆**(诊断/投影面出口:`GET /debug/blocks` 的 GPU 命令通道;
    /// 取主会话图的 face —— 克隆 = 同一个 actor 的另一个发送端)
    pub fn face_clone(&self) -> Option<D> {
        self.session.face_clone()
    }

    async fn plan_session(face: D, boot: &FleetBoot<'_>, t0: T) -> Result<(GraphPlan<D>, PlanOutcome)> {
        let env = boot.env;
        let pool = boot.pool;
        let dims = boot.dims;
        let model = Arc::clone(&boot.model);
        let rp = boot.rope.clone();
        let vocab = boot.model.vocab_size();
        let knobs = boot.knobs;
        let kv_caches = pool.kv_caches();
        let gdn_caches = pool.gdn_caches();
        let bt_leaf = pool.bt_leaf();
        let attn_v2 = pool.attn_v2_scratch();
        // XQA 裸 decode 面(2026-10-13 ②):池常量件 + 两个图输入槽
        // (`xqa_seq` = kv_len 的 **u32 位型** —— 核按 `const uint32_t*` 读,
        //  直接写 f32 数值会拿到浮点位型;`xqa_sem` = 每轮写零)。
        // 槽只在 `decode_xqa` 开时才加 ⇒ 关时图拓扑与历史一致(零影响)。
        let xqa_decode_on = knobs.env.attn.decode_xqa;
        let xqa_decode_ctx = if xqa_decode_on { pool.xqa_ctx() } else { None };
        let xqa_dec_sem_n = if xqa_decode_on { pool.xqa_sem_n() } else { 0 };
        // B4:decode 图 taps(DFlash2 降级期逐 token 草稿同步的取数面;
        // notaps 同款逃生开关)。tapped_hidden 零额外计算,
        // 仅多 5 个 [1,hidden] 输出槽写;非 dflash 走原 forward(拓扑不变)
        let decode_taps = boot.spec.dflash_tap_count > 0;
        let forward = move |sc: &PlanCtx| -> Result<()> {
            let ids = sc.input("frontier")?;
            let pos = sc.input("pos")?;
            let kv_len = sc.input("kv_len")?;
            let kv_slot = sc.input("kv_slot")?;
            let gdn_slot = sc.input("gdn_slot")?;
            let ts_buf = sc.input_opt("ts_buf")?;
            let kvs_step: Vec<KvBuffers> = kv_caches
                .iter()
                .map(|(k, v)| KvBuffers {
                    k_cache: k.clone(),
                    v_cache: v.clone(),
                    slots: kv_slot.clone(),
                    kv_lens: kv_len.clone(),
                    block_tables: bt_leaf.clone(),
                })
                .collect();
            let gdns_step: Vec<GdnBuffers> = gdn_caches
                .iter()
                .map(|g| GdnBuffers {
                    conv_q: g.conv_q.clone(),
                    conv_k: g.conv_k.clone(),
                    conv_v: g.conv_v.clone(),
                    rec: g.rec.clone(),
                    slots: gdn_slot.clone(),
                })
                .collect();
            let mut ctx = ForwardCtx::model_decode(1, &pos, &kvs_step, &rp, &gdns_step);
            ctx.env = env; // 捕获期烘焙(图闭包 env 定格;回放沿用)
            ctx.ts_buf = ts_buf; // 刀D 时间戳探针(None = 关)
            ctx.attn_v2 = attn_v2.clone(); // v2 分页 decode scratch(None = v1)
            // XQA 裸 decode 臂(2026-10-13 ②):面 = pool 常量件 + 图输入 seq/sem
            // (sem 走输入槽的正解见工单 §2.7 —— 禁跨账本 memset 命令)
            if let Some(x) = xqa_decode_ctx.as_ref() {
                let seq_in = sc.input("xqa_seq")?;
                let pt_in = sc.input("xqa_pt")?;
                let sem_in = sc.input("xqa_sem")?;
                ctx.xqa = Some(owl_models::module::XqaVerifyCtx {
                    // 页表 = 输入槽(由核侧每轮写会话真链);verify 图同款
                    page_table: pt_in,
                    seq_lens: seq_in,
                    mask: x.mask.clone(),
                    semaphores: sem_in,
                    scratch: x.scratch.clone(),
                    q_seq_len: 1,
                    max_seq_len: x.max_seq_len,
                    sm_count: x.sm_count,
                });
            }
            // B4:降级期取数面 —— tapped_hidden 拿末层 hidden + 5 层 taps
            // (零额外前向;l m_head 与 forward 同式)。非 dflash 原路(位等价)
            let tree = if decode_taps {
                let (hidden, taps) = model.tapped_hidden(&ids, &ctx, &[5, 19, 33, 47, 61]);
                for (i, t) in taps.iter().enumerate() {
                    sc.output(format!("tap{i}"), t)?;
                }
                model.embed.lm_head_matmul(&hidden)
            } else {
                model.forward(&ids, &ctx)
            };
            // E3 设备采样:token = argmax(logits)(f32 数值过线,契约 5)
            let tok = owl_models::ops::argmax_f32idx(&tree, vocab, 0);
            sc.output("token", &tok)?;
            sc.output("logits", &tree)
        };

        let (session, capture_outcome) = GraphPlan::plan(
            face,
            env,
            GraphPlanDesc {
                inputs: {
                    let mut v = vec![
                        InputSlot::f32("frontier", 1),
                        InputSlot::f32("pos", 1),
                        InputSlot::f32("kv_len", 1),
                        InputSlot::f32("kv_slot", 1),
                        InputSlot::f32("gdn_slot", 1).init(vec![0.0]),
                    ];
                    if xqa_dec_sem_n > 0 {
                        v.push(InputSlot::f32("xqa_seq", 1).init(vec![0.0]));
                        v.push(InputSlot::f32("xqa_sem", xqa_dec_sem_n).init(vec![0.0; xqa_dec_sem_n]));
                        // XQA **页表必须走输入槽**(每轮写会话真页链):
                        // 常量恒等链 0..nb 只在"单会话独占池(块从 0 起)"时成立;
                        // 多会话(agent:每请求新会话,块链从空闲链续)会让 XQA
                        // 整段读到**别家 KV** —— 实测"答上一题"(§7.6 定谳,
                        // 2026-10-14)。init 保留恒等链(单会话行为不变);
                        // 槽尺寸 = 池 nb(与会话链上界一致,尾部补零)
                        v.push(InputSlot::f32("xqa_pt", pool.nb).init(
                            (0..pool.nb).map(|i| f32::from_bits(i as u32)).collect(),
                        ));
                    }
                    if knobs.ts_probe {
                        v.push(InputSlot::f32("ts_buf", 4096));
                    }
                    v
                },
                outputs: {
                    let mut o = vec![
                        OutputSlot { name: "logits".into(), shape: vec![1, vocab], dtype: dims.dtype },
                        OutputSlot { name: "token".into(), shape: vec![1], dtype: Dtype::F32 },
                    ];
                    if decode_taps {
                        for i in 0..5 {
                            o.push(OutputSlot {
                                name: format!("tap{i}"),
                                shape: vec![1, dims.hidden],
                                dtype: dims.dtype,
                            });
                        }
                    }
                    o
                },
                // 逃生开关(C1 禁 graph 裁决配套):no_graph → eager
                // 直发(decode 逐步 eval,无捕获回放)—— 图内/图外行为
                // A/B 的对照臂(2026-10-01 paged decode 质量案)
                capture: !knobs.no_graph,
                slab_fixed: knobs.cuda.capture_slab_mb.is_some(),
                debug_probe: knobs.probes.debug,
                // 本族不加锚定回收(plain 口径:窗内块常驻 —— 历史行为零变化)
                // 锚定活性回收(2026-10-15 归因之修②;理由同族桶形图):
                // plain = 分配总和(主会话 13.6MiB / 该图实测总量级更高)⇒
                // 开 scoped,容量降到峰值活跃集;恒等门同族。
                reclaim_capture: true,
                warmup: true,
            },
            forward,
        )
        .await?;
        if let PlanOutcome::EagerFallback { reason } = &capture_outcome {
            owl_shared::log_warn!("engine", "捕获降级为 eager:{reason}");
        }
        owl_shared::log_info!("boot", "GraphPlan plan(warmup+捕获) {:.2}s", t0.elapsed().as_secs_f32());
        Ok((session, capture_outcome))
    }

    /// ② verify 桶形图(E5-M4;T = depth+1 定形;spec 模式专用)—— 消
    /// 投机轮的整模 eager 发射税(~830 发 × ~25µs)。FI 不入图(T=4
    /// attention 成本可忽略;回避 plan-in-graph);输出 = tok/hid +
    /// **GDN fold 记录**(每层 8 件,tap 收集 → 多根共享 memo 单次归约,
    /// 刀 1.6 语义)。捕获失败 → None → 执行器落旧 eager verify+重放路。
    async fn plan_verify(vface: D, boot: &FleetBoot<'_>, t0: T) -> Result<Option<GraphPlan<D>>> {
        let pool = boot.pool;
        let dims = boot.dims;
        let env = boot.env;
        let knobs = boot.knobs;
        let spec_mode = boot.spec.mode;
        let depth1 = boot.spec.depth + 1;
        let model_v = Arc::clone(&boot.model);
        let rp_v = boot.rope.clone();
        let kv_caches = pool.kv_caches();
        let gdn_caches = pool.gdn_caches();
        let bt_leaf = pool.bt_leaf_flat();
        let bt_leaf_nb = *bt_leaf.shape().last().unwrap_or(&1);
        let verify_attn_v2 = pool.verify_attn_v2_scratch();
        // XQA 面(2026-10-13 D3):池常量件 + 图槽 init(i32/u32 位型走私进
        // f32 账叶 —— 与 sm_scale/fi_scale_bits 同款约定)
        let xqa_pool_ctx = pool.xqa_ctx();
        let xqa_pt_nb = bt_leaf_nb;
        let xqa_pt_init: Vec<f32> = (0..xqa_pt_nb)
            .map(|i| f32::from_bits(i as u32))
            .collect();
        let xqa_sem_n = xqa_pool_ctx.as_ref().map(|x| x.semaphores.shape().iter().product::<usize>()).unwrap_or(0);
        let xqa_mask_init: Vec<f32> = (0..depth1)
            .map(|i| f32::from_bits((1u32 << (i + 1)) - 1))
            .collect();
        let key_dim = dims.nk * dims.hk;
        let value_dim = dims.nv * dims.hv;
        let mut vouts = vec![
            OutputSlot { name: "tok".into(), shape: vec![depth1, 1], dtype: Dtype::F32 },
            OutputSlot { name: "hid".into(), shape: vec![depth1, dims.hidden], dtype: dims.dtype },
            // A11(2026-10-14):per-row logits 出槽 —— 投机轮的**采样接受**
            // 需要各行目标分布(原先只有 argmax ⇒ spec 臂不采样 =
            // 请求温度失效;A7/A8 根因,见 agent 专项 §7.11)。
            // 贪心档不回读(零额外 D2H);仅采样档付 ~2.4MB/轮。
            OutputSlot { name: "logits".into(), shape: vec![depth1, model_v.vocab_size()], dtype: Dtype::F16 },
        ];
        // DFlash2 target taps(E5-DF2;层输出残差流,sglang capture 同语义)
        // notaps → 退回 last_hidden(排查开关:隔离 taps 图)
        let dflash_taps = boot.spec.dflash_tap_count > 0;
        // TAPDECL:声明期 tapped(收集)但不挂输出槽 —— 二分「tap 节点
        // 本身」vs「输出槽机制」(2026-10-07 恒等门排查)
        let dflash_tap_decl =
            dflash_taps || (spec_mode == crate::spec_runtime::SpecMode::DFlash2 && knobs.draft.tapdecl);
        if dflash_taps {
            for i in 0..5 {
                vouts.push(OutputSlot { name: format!("tap{i}"), shape: vec![depth1, dims.hidden], dtype: dims.dtype });
            }
        }
        for i in 0..gdn_caches.len() * 8 {
            let (_gi, f) = (i / 8, i % 8);
            let shape = match f {
                0 | 1 => vec![depth1, key_dim],
                2 => vec![depth1, value_dim],
                3 | 4 => vec![depth1, dims.nk, dims.hk],
                5 => vec![depth1, dims.nv, dims.hv],
                _ => vec![depth1, dims.nv],
            };
            vouts.push(OutputSlot { name: format!("r{i}") , shape, dtype: dims.dtype });
        }
        let (vg, voutcome) = GraphPlan::plan(
            vface,
            env,
            GraphPlanDesc {
                inputs: vec![
                    InputSlot::f32("ids", depth1),
                    InputSlot::f32("pos", depth1),
                    InputSlot::f32("kv_slots", depth1),
                    InputSlot::f32("kv_lens", depth1),
                    InputSlot::f32("gdn_slot", 1),
                    InputSlot::f32("gdn_cu", 2).init(vec![0.0, depth1 as f32]),
                    InputSlot::f32("bt8", depth1 * bt_leaf_nb),
                    // XQA 面(2026-10-13 D3):seq_lens 逐轮变(u32 位型走私);
                    // 页表/掩码是常量,在槽 init 里填一次(捕获前 Htod 合法)
                    // ⚠️ **零长槽地雷**:`xqa_sem_n` 在"无 XQA 面"时为 0,
                    // 无条件声明就是零长槽 ⇒ 每轮 step 写 0 字节 ⇒ 后端
                    // `malloc_host(0)` 空指针崩(2026-10-13 挂账 #7 定谳:
                    // 0.8B spec 档踩的就是它 —— 之前只跑 fp8 门所以未暴露)。
                    // 取 `.max(1)`:拓扑恒常(不做条件声明),多写 4 字节无害。
                    InputSlot::f32("xqa_seq", 1).init(vec![0.0]),
                    InputSlot::f32("xqa_pt", xqa_pt_nb).init(xqa_pt_init.clone()),
                    InputSlot::f32("xqa_mask", depth1).init(xqa_mask_init.clone()),
                    // semaphores 走**图输入槽**(每轮 step 写零):
                    // ⚠️ 曾用 `memset_zero` 命令 → 跨账本解析踩雷(见工单
                    // §2.7 定谳);输入槽与 xqa_seq 同一条已验证通路
                    InputSlot::f32("xqa_sem", xqa_sem_n.max(1))
                        .init(vec![0.0; xqa_sem_n.max(1)]),
                ],
                outputs: vouts,
                capture: !knobs.no_graph,
                slab_fixed: knobs.cuda.capture_slab_mb.is_some(),
                debug_probe: knobs.probes.debug,
                // 本族不加锚定回收(plain 口径:窗内块常驻 —— 历史行为零变化)
                // 锚定活性回收(2026-10-15 归因之修②;理由同族桶形图):
                // plain = 分配总和(主会话 13.6MiB / 该图实测总量级更高)⇒
                // 开 scoped,容量降到峰值活跃集;恒等门同族。
                reclaim_capture: true,
                warmup: true,
            },
            move |sc: &PlanCtx| -> Result<()> {
                let ids = sc.input("ids")?;
                let pos = sc.input("pos")?;
                let slots_in = sc.input("kv_slots")?;
                let lens_in = sc.input("kv_lens")?;
                let gdn_slot = sc.input("gdn_slot")?;
                let gdn_cu = sc.input("gdn_cu")?;
                let bt8_in = sc.input("bt8")?;
                // §三十一:v2 臂在(seqs=depth1)→ 页表 = bt8([8×nb]);
                // 回退(无 scratch)→ 单表 bt_leaf(split/chunked 语义不变)
                let kv_bt = if verify_attn_v2.is_some() { bt8_in.clone() } else { bt_leaf.clone() };
                let kvs_step: Vec<KvBuffers> = kv_caches
                    .iter()
                    .map(|(k, v)| KvBuffers {
                        k_cache: k.clone(),
                        v_cache: v.clone(),
                        slots: slots_in.clone(),
                        kv_lens: lens_in.clone(),
                        block_tables: kv_bt.clone(),
                    })
                    .collect();
                let gdns_step: Vec<GdnBuffers> = gdn_caches
                    .iter()
                    .map(|g| GdnBuffers {
                        conv_q: g.conv_q.clone(),
                        conv_k: g.conv_k.clone(),
                        conv_v: g.conv_v.clone(),
                        rec: g.rec.clone(),
                        slots: gdn_slot.clone(),
                    })
                    .collect();
                let mut ctx = ForwardCtx::model_prefill(
                    depth1, &pos, &kvs_step, &rp_v, &gdns_step, &slots_in, &lens_in,
                    &gdn_slot,
                    // 捕获图:槽号是回放期输入(捕获期无真值)—— T=depth1
                    // ≤ 16 < 64,chunked/scalar 臂结构上不可能启用,故
                    // None 安全;若将来在此捕 ≥64 图,层侧会拒发(poisoned)
                    // 而不是静默写槽 0(显式依赖律)
                    None,
                    0, None,
                );
                ctx.seq_cu = Some(&gdn_cu);
                ctx.env = env;
                // §三十一:verify v2 臂(8 伪序列 partition 并行;scratch
                // = depth1 倍;bt8 = 8× 页链平铺,engine 每轮 host 写入)
                ctx.attn_v2 = verify_attn_v2.clone();
                ctx.v2_bt8 = Some(&bt8_in);
                // XQA verify 臂(2026-10-13 D3):面 = pool 常量件 + 图输入 seq
                if let Some(x) = xqa_pool_ctx.as_ref() {
                    let seq_in = sc.input("xqa_seq")?;
                    let pt_in = sc.input("xqa_pt")?;
                    let mk_in = sc.input("xqa_mask")?;
                    let sem_in = sc.input("xqa_sem")?;
                    ctx.xqa = Some(owl_models::module::XqaVerifyCtx {
                        page_table: pt_in,
                        seq_lens: seq_in,
                        mask: mk_in,
                        semaphores: sem_in,
                        scratch: x.scratch.clone(),
                        q_seq_len: depth1,
                        max_seq_len: x.max_seq_len,
                        sm_count: x.sm_count,
                    });
                }
                let tap: std::rc::Rc<std::cell::RefCell<Vec<TensorOps>>> =
                    std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
                ctx.gdn_tap = Some(tap.clone());
                let vocab = model_v.vocab_size();
                let (hidden, taps) = if dflash_tap_decl {
                    let (h, t) = model_v.tapped_hidden(&ids, &ctx, &[5, 19, 33, 47, 61]);
                    if dflash_taps {
                        for (i, t) in t.iter().enumerate() {
                            sc.output(format!("tap{i}"), t)?;
                        }
                    }
                    (h, t)
                } else {
                    let h = model_v.last_hidden(&ids, &ctx);
                    (h.clone(), vec![h])
                };
                let _ = &taps;
                let logits = model_v.embed.lm_head_matmul(&hidden);
                // A11(2026-10-14):per-row logits 出槽 —— 投机轮的
                // **采样接受**需要各行目标分布(原先只出 argmax ⇒ spec 臂
                // 完全不采样,请求温度/top_p/seed 失效 = 事实贪心;
                // A7 工具遵循度塌陷 + A8 不可复现的根因,见
                // docs/arch/agent-服务面-设计与施工.md §7.11)。
                // 仅采样档回读(贪心档零额外 D2H)。
                sc.output("logits", &logits)?;
                let rows: Vec<TensorOps> = (0..depth1)
                    .map(|r| owl_models::ops::argmax_f32idx(&logits, vocab, r * vocab))
                    .collect();
                let mut root = TensorOps::call(owl_models::ops::SemanticKernel::Concat);
                for i in 0..8 {
                    root = root.arg(rows.get(i).unwrap_or(&rows[0]));
                }
                let root = root
                    .arg_usize(depth1)
                    .arg_usize(1)
                    .arg_usize(1)
                    .with_shape(Dtype::F32, vec![depth1, 1]);
                sc.output("tok", &root)?;
                sc.output("hid", &hidden)?;
                for (i, t) in tap.borrow().iter().enumerate() {
                    sc.output(format!("r{i}"), t)?;
                }
                Ok(())
            },
        )
        .await?;
        match voutcome {
            PlanOutcome::Captured => {
                owl_shared::log_info!("boot", "verify 桶形图捕获(T={depth1}) +{:.2}s", t0.elapsed().as_secs_f32());
                Ok(Some(vg))
            }
            other => {
                owl_shared::log_info!("boot", "verify 图降级({other:?})→ eager verify+重放路");
                Ok(None)
            }
        }
    }

    /// ③a fold 桶形图族(E5-M5;index = m;状态 = spec 快照 buf 就地,
    /// slot 恒 0 —— 快照即工作缓冲,fold 后 restore 回写会话格;
    /// 记录 = verify 图输出块前缀视图,指针捕获期烘焙)。96 发射的
    /// eager fold → 单 graph_launch。
    async fn plan_fold_family(
        vg: &GraphPlan<D>,
        extra_faces: &mut Vec<D>,
        boot: &FleetBoot<'_>,
        t0: T,
    ) -> Result<Vec<GraphPlan<D>>> {
        let pool = boot.pool;
        let dims = boot.dims;
        let env = boot.env;
        let knobs = boot.knobs;
        let mut fold_graphs = Vec::new();
        let n_rec = pool.gdn_count() * 8;
        let rec_blks: Vec<Bytes> = (0..n_rec)
            .map(|i| vg.output_block(&format!("r{i}")).expect("fold 记录槽"))
            .collect();
        let snap_bufs = pool.spec_snap_bufs();
        let n_roots = pool.gdn_count() * 4;
        for m in 0..boot.spec.depth {
            let m1 = m + 1;
            let model_f = Arc::clone(&boot.model);
            let gdn_caches = pool.gdn_caches();
            let snap = snap_bufs.clone();
            let rec_blks = rec_blks.clone();
            let scalar = env.gdn.scalar;
            let d = dims;
            let mut fouts = Vec::with_capacity(n_roots);
            for i in 0..n_roots {
                let shape = match i % 4 {
                    0 => vec![m1, d.nk * d.hk],
                    1 => vec![m1, d.nk * d.hk],
                    2 => vec![m1, d.nv * d.hv],
                    _ => vec![m1, d.nv, d.hv],
                };
                fouts.push(OutputSlot { name: format!("f{i}"), shape, dtype: Dtype::F16 });
            }
            let Some(fface) = extra_faces.pop() else {
                owl_shared::log_info!("boot", "fold 图 face 耗尽 → eager fold");
                break;
            };
            let (fg, fout) = GraphPlan::plan(
                fface,
                env,
                GraphPlanDesc {
                    inputs: vec![
                        InputSlot::f32("slots", 1).init(vec![0.0]),
                        InputSlot::f32("cu", 2).init(vec![0.0, m1 as f32]),
                    ],
                    outputs: fouts,
                    capture: !knobs.no_graph,
                    slab_fixed: knobs.cuda.capture_slab_mb.is_some(),
                    debug_probe: knobs.probes.debug,
                    // 本族不加锚定回收(plain 口径:窗内块常驻 —— 历史行为零变化)
                    // 锚定活性回收(2026-10-15 归因之修②):本族 plain 口径是**分配总和**
                    // (verify 图实测 133MiB 而真活跃 ~19MB)⇒ 开 scoped 归约,
                    // slab 容量 = 峰值活跃集。机制门 = graph_plan::gpu_capture_reclaim_matches_eager
                    // + 族恒等门(fold/propose/verify 各有对拍)。
                    reclaim_capture: true,
                    warmup: true,
                },
                move |sc: &PlanCtx| -> Result<()> {
                    let slots = sc.input("slots")?;
                    let cu = sc.input("cu")?;
                    let rec_views: Vec<TensorOps> = rec_blks
                        .iter()
                        .enumerate()
                        .map(|(i, b)| {
                            let shape = match i % 8 {
                                0 | 1 => vec![m1, d.nk * d.hk],
                                2 => vec![m1, d.nv * d.hv],
                                3 | 4 => vec![m1, d.nk, d.hk],
                                5 => vec![m1, d.nv, d.hv],
                                _ => vec![m1, d.nv],
                            };
                            TensorOps::of_block(b.id, d.dtype, shape)
                        })
                        .collect();
                    let gdns: Vec<GdnBuffers> = (0..gdn_caches.len())
                        .map(|li| GdnBuffers {
                            conv_q: leaf_flat(&snap[li * 4]),
                            conv_k: leaf_flat(&snap[li * 4 + 1]),
                            conv_v: leaf_flat(&snap[li * 4 + 2]),
                            rec: leaf_flat(&snap[li * 4 + 3]),
                            slots: slots.clone(),
                        })
                        .collect();
                    let roots = owl_models::spec::fold_tree(
                        &model_f, &rec_views, &gdns, &slots, &cu, 0, m, scalar,
                    )?;
                    for (i, r) in roots.iter().enumerate() {
                        sc.output(format!("f{i}"), r)?;
                    }
                    Ok(())
                },
            )
            .await?;
            match fout {
                PlanOutcome::Captured => {
                    owl_shared::log_info!("boot", "fold 桶形图捕获(m={m}) +{:.2}s", t0.elapsed().as_secs_f32());
                    fold_graphs.push(fg);
                }
                other => owl_shared::log_info!("boot", "fold 图降级(m={m},{other:?})→ eager fold"),
            }
        }
        Ok(fold_graphs)
    }

    /// ③b DFlash2 propose **单图**(E5-DF4;encode 全 8 行 + 噪声块 +
    /// selector 固定几何;多编行下轮被自块写/重编覆盖,幂等安全)。
    /// 输入槽 = tokens/anchor/enc_pos/enc_slots/prop_pos/prop_slots/
    /// kv_len;烘焙叶 = verify taps 槽 + 草稿池叶。prefix_len 曾为
    /// 宿主烘焙标量,已改核内读 kv_len 张量(契约 5)。
    async fn plan_dflash(
        vg: &GraphPlan<D>,
        extra_faces: &mut Vec<D>,
        boot: &FleetBoot<'_>,
        t0: T,
    ) -> Result<Option<GraphPlan<D>>> {
        let Some(Drafter::DFlash2(d2)) = boot.spec.drafter.as_ref() else {
            return Ok(None);
        };
        let pool = boot.pool;
        let env = boot.env;
        let knobs = boot.knobs;
        let tap_blks: Vec<Bytes> = (0..5)
            .map(|i| vg.output_block(&format!("tap{i}")).expect("tap 输出槽"))
            .collect();
        let kv_leaves = pool.dflash_kv_leaves().expect("dflash 池");
        let bt_leaf = pool.bt_leaf_flat();
        let model_df = Arc::clone(&boot.model);
        let rp_df = boot.spec.draft_rope.as_ref().expect("draft rope").clone();
        let d2g = Arc::clone(d2);
        let hidd = d2.hidden();
        let Some(gface) = extra_faces.pop() else {
            owl_shared::log_info!("boot", "dflash propose 图 face 耗尽 → eager propose");
            return Ok(None);
        };
        let (dg, dout) = GraphPlan::plan(
            gface,
            env,
            GraphPlanDesc {
                inputs: vec![
                    InputSlot::f32("tokens", 8),
                    InputSlot::f32("anchor", 1),
                    InputSlot::f32("enc_pos", 8),
                    InputSlot::f32("enc_slots", 8),
                    InputSlot::f32("enc_kv_lens", 8),
                    InputSlot::f32("prop_pos", 8),
                    InputSlot::f32("prop_slots", 8),
                    InputSlot::f32("prop_kv_lens", 8),
                    InputSlot::f32("kv_len", 1),
                ],
                outputs: vec![
                    OutputSlot::f32("drafts", &[7 * (16 * 16 + 1)]),
                    OutputSlot { name: "e0".into(), shape: vec![1], dtype: Dtype::BF16 },
                    OutputSlot { name: "e1".into(), shape: vec![1], dtype: Dtype::BF16 },
                    OutputSlot { name: "e2".into(), shape: vec![1], dtype: Dtype::BF16 },
                    OutputSlot { name: "e3".into(), shape: vec![1], dtype: Dtype::BF16 },
                    OutputSlot { name: "e4".into(), shape: vec![1], dtype: Dtype::BF16 },
                ],
                capture: !knobs.no_graph,
                slab_fixed: knobs.cuda.capture_slab_mb.is_some(),
                debug_probe: knobs.probes.debug,
                // 本族不加锚定回收(plain 口径:窗内块常驻 —— 历史行为零变化)
                // 锚定活性回收(2026-10-15 归因之修②;理由同族桶形图):
                // plain = 分配总和(主会话 13.6MiB / 该图实测总量级更高)⇒
                // 开 scoped,容量降到峰值活跃集;恒等门同族。
                reclaim_capture: true,
                warmup: true,
            },
            move |sc: &PlanCtx| -> Result<()> {
                let taps: Vec<TensorOps> = tap_blks
                    .iter()
                    .map(|b| TensorOps::of_block(b.id, Dtype::F16, vec![8, hidd]))
                    .collect();
                let tokens = sc.input("tokens")?;
                let anchor = sc.input("anchor")?;
                let enc_pos = sc.input("enc_pos")?;
                let enc_slots = sc.input("enc_slots")?;
                // 逐行可见长(含自身;eager dflash_encode 同式
                // pos+i+1 —— 原绑 enc_pos 缺自身 = AL 劣化根因)
                let enc_kv_lens = sc.input("enc_kv_lens")?;
                let prop_pos = sc.input("prop_pos")?;
                let prop_slots = sc.input("prop_slots")?;
                // propose 噪声块逐行可见长(fp+1+i;原绑标量
                // kv_len,1 值槽供 8 行读 = 1~7 行越界垃圾)
                let prop_kv_lens = sc.input("prop_kv_lens")?;
                let kv_len = sc.input("kv_len")?;
                let bt = bt_leaf.clone();
                let enc_kvs: Vec<KvBuffers> = kv_leaves
                    .iter()
                    .map(|(k, v)| KvBuffers {
                        k_cache: k.clone(),
                        v_cache: v.clone(),
                        slots: enc_slots.clone(),
                        kv_lens: enc_kv_lens.clone(),
                        block_tables: bt.clone(),
                    })
                    .collect();
                let prop_kvs: Vec<KvBuffers> = kv_leaves
                    .iter()
                    .map(|(k, v)| KvBuffers {
                        k_cache: k.clone(),
                        v_cache: v.clone(),
                        slots: prop_slots.clone(),
                        kv_lens: prop_kv_lens.clone(),
                        block_tables: bt.clone(),
                    })
                    .collect();
                let ctx = ForwardCtx::minimal(8);
                let mem = d2g.project_memory(&taps, &ctx);
                let enc_roots = d2g.encode_kv(&mem, &enc_pos, &enc_kvs, &rp_df, &ctx);
                for (i, r) in enc_roots.iter().enumerate() {
                    sc.output(format!("e{i}"), r)?;
                }
                let (_h, drafts, _s, _l) = d2g.propose_block(
                    &tokens, &prop_pos, &anchor, &prop_kvs, &rp_df,
                    &model_df.embed, &kv_len, &ctx, None,
                );
                // drafts = select 单缓冲前 7 元(offset 0;输出槽
                // 形状 = 整缓冲 —— dtoh 整父块契约)
                sc.output("drafts", &drafts)
            },
        )
        .await?;
        match dout {
            PlanOutcome::Captured => {
                owl_shared::log_info!("boot", "dflash propose 图捕获(encode+propose 单图,固定 8 行) +{:.2}s", t0.elapsed().as_secs_f32());
                Ok(Some(dg))
            }
            other => {
                owl_shared::log_info!("boot", "dflash propose 图降级({other:?})→ eager propose");
                Ok(None)
            }
        }
    }

    /// ③c MTP propose 桶形图族(E5-M5;index = m ∈ 0..=depth —— 全接受
    /// m=depth 也要出下轮草稿;extend m+1 行 + k-1 链步;
    /// hidden = verify hid 槽前缀视图;lm_head/链全部入图)。
    async fn plan_propose_family(
        vg: &GraphPlan<D>,
        extra_faces: &mut Vec<D>,
        boot: &FleetBoot<'_>,
        _t0: T,
    ) -> Result<Vec<GraphPlan<D>>> {
        let Some(Drafter::Mtp(mtp)) = boot.spec.drafter.as_ref() else {
            return Ok(Vec::new());
        };
        let pool = boot.pool;
        let dims = boot.dims;
        let env = boot.env;
        let knobs = boot.knobs;
        let spec_depth = boot.spec.depth;
        let mut propose_graphs = Vec::new();
        let hid_blk = vg.output_block("hid").expect("hid 输出槽");
        // 链步 decode 读核 scratch(seqs=1;与 draft.rs eager 路同源):
        // kNHD 池落 v1 = 布局错位 → 草稿垃圾(2026-10-13 守卫捕获)
        let attn_v2_pool = pool.attn_v2_scratch();
        for m in 0..=spec_depth {
            // move 闭包逐个消耗 → 每轮取一份克隆
            let attn_v2_p = attn_v2_pool.clone();
            let m1 = m + 1;
            let model_p = Arc::clone(&boot.model);
            let mtp = Arc::clone(mtp);
            let rp_p = boot.rope.clone();
            let kv_leaf = pool.mtp_kv_leaf().expect("mtp 池");
            let bt_leaf = pool.bt_mtp_leaf().expect("bt_mtp 槽");
            let hid = hid_blk.clone();
            let env_p = env;
            let vocab = boot.model.vocab_size();
            let mut pins = vec![InputSlot::f32("tok_ext", m1), InputSlot::f32("pos_ext", m1), InputSlot::f32("slots_ext", m1), InputSlot::f32("lens_ext", m1)];
            for j in 0..spec_depth - 1 {
                pins.push(InputSlot::f32(format!("c{j}_pos"), 1));
                pins.push(InputSlot::f32(format!("c{j}_slots"), 1));
                pins.push(InputSlot::f32(format!("c{j}_lens"), 1));
            }
            pins.push(InputSlot::f32("seq_cu", 2).init(vec![0.0, m1 as f32]));
            let Some(pface) = extra_faces.pop() else {
                return Err(owl_iface::contract::ModelError::Msg(format!(
                    "propose 图 face 耗尽(m={m}/depth={spec_depth}):extra_faces 预算 2*depth.max(1)+2 不足以覆盖 fold+dflash+propose 族"
                )));
            };
            let (pg, pout) = GraphPlan::plan(
                pface,
                env,
                GraphPlanDesc {
                    inputs: pins,
                    outputs: vec![OutputSlot { name: "drafts".into(), shape: vec![spec_depth, 1], dtype: Dtype::F32 }],
                    capture: !knobs.no_graph,
                    slab_fixed: knobs.cuda.capture_slab_mb.is_some(),
                    debug_probe: knobs.probes.debug,
                    // 本族不加锚定回收(plain 口径:窗内块常驻 —— 历史行为零变化)
                    // 锚定活性回收(2026-10-15 归因之修②):本族 plain 口径是**分配总和**
                    // (verify 图实测 133MiB 而真活跃 ~19MB)⇒ 开 scoped 归约,
                    // slab 容量 = 峰值活跃集。机制门 = graph_plan::gpu_capture_reclaim_matches_eager
                    // + 族恒等门(fold/propose/verify 各有对拍)。
                    reclaim_capture: true,
                    warmup: true,
                },
                move |sc: &PlanCtx| -> Result<()> {
                    // PlanCtx 输入声明形 [1, len] → flatten(行主前缀)
                    let tok_ext = sc.input("tok_ext")?.reshape(vec![m1]);
                    let pos_ext = sc.input("pos_ext")?.reshape(vec![m1]);
                    let slots_ext = sc.input("slots_ext")?.reshape(vec![m1]);
                    let lens_ext = sc.input("lens_ext")?.reshape(vec![m1]);
                    let seq_cu = sc.input("seq_cu")?;
                    let bt = bt_leaf.clone();
                    let kv_ext = KvBuffers {
                        k_cache: kv_leaf.0.clone(),
                        v_cache: kv_leaf.1.clone(),
                        slots: slots_ext.clone(),
                        kv_lens: lens_ext.clone(),
                        block_tables: bt.clone(),
                    };
                    let hid_view = TensorOps::of_block(hid.id, dims.dtype, vec![m1, dims.hidden]);
                    let mut chain: Vec<(TensorOps, KvBuffers)> = Vec::new();
                    for j in 0..spec_depth - 1 {
                        let pos_c = sc.input(&format!("c{j}_pos"))?;
                        let slots_c = sc.input(&format!("c{j}_slots"))?;
                        let lens_c = sc.input(&format!("c{j}_lens"))?;
                        chain.push((
                            pos_c,
                            KvBuffers {
                                k_cache: kv_leaf.0.clone(),
                                v_cache: kv_leaf.1.clone(),
                                slots: slots_c.clone(),
                                kv_lens: lens_c.clone(),
                                block_tables: bt.clone(),
                            },
                        ));
                    }
                    let chain_refs: Vec<(&TensorOps, &KvBuffers)> =
                        chain.iter().map(|(p, k)| (p, k)).collect();
                    let droot = mtp.propose_ext(
                        &tok_ext,
                        &hid_view,
                        &pos_ext,
                        &kv_ext,
                        &slots_ext,
                        &lens_ext,
                        &seq_cu,
                        &chain_refs,
                        &model_p.embed,
                        &rp_p,
                        vocab,
                        env_p,
                        // 图步链读核也走 v2(同 draft.rs:kNHD 池落 v1 = 错位)
                        attn_v2_p.clone(),
                    );
                    sc.output("drafts", &droot)
                },
            )
            .await?;
            match pout {
                PlanOutcome::Captured => {
                    owl_shared::log_info!("boot", "propose 桶形图捕获(m={m})");
                    propose_graphs.push(pg);
                }
                other => owl_shared::log_info!("boot", "propose 图降级(m={m},{other:?})→ eager propose"),
            }
        }
        Ok(propose_graphs)
    }
}

/// E5-M5:fold 图的状态叶子(flat 单槽行;slot 恒 0 = 基指针;
/// 自由函数 —— 图闭包 move 语义不可借 self)
fn leaf_flat(b: &crate::state::BlockN) -> TensorOps {
    let n = b.1;
    TensorOps::of_block(b.0.id, Dtype::F32, vec![n])
}

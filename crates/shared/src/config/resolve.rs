//! **唯一派生器**:Intent + Site + 显式低层键 + CLI `--set` → 同一份
//! [`super::OwlConfig`](派生在反序列化之前,作用在 TOML 值上)。
//!
//! 顺序(有依赖,不可乱;设计卷 §6.2):
//! 1. `[site]` 映射(路径/卡/端口);
//! 2. `[intent]` → 派生低层键(模型/目标/上下文/并发/投机/KV/KVMem/图/复用);
//! 3. 显式低层键覆盖派生值(高级覆盖;provenance 记文件);
//! 4. `context = "max"` 用内存模型求解(显式 `max_seq` 优先;池由引擎按剩余显存自动定);
//! 5. 反序列化 + `OwlConfig::validate`(同一套语义门)。
//!
//! **来源追踪**:每个叶子键记一条 [`Origin`](缺省/文件/站点/CLI/派生规则);
//! `--print-config` / `--explain` / `/debug/config` 都读它。派生**不是黑魔法**
//! —— rule 文本必须能解释"为什么是这个值"。

use std::collections::BTreeMap;

use serde::Serialize;
use toml::Value;

use super::intent::{
    self, ContextSpec, Goal, GraphIntent, Intent, KvIntent, KvmemIntent, ReuseIntent, SpecIntent,
};
use super::memory::{self, MemoryEstimate, MemoryInputs, STABILITY_MARGIN_BYTES};

/// 逐键来源表(键 = 点路径,如 `pool.prefill_chunk`)
pub type Origins = BTreeMap<String, Origin>;

/// 一个低层键的来源
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", content = "detail", rename_all = "snake_case")]
pub enum Origin {
    /// 代码缺省(未写、也未派生)
    Default,
    /// 配置文件显式写(路径 = 实际文件)
    File(String),
    /// `[site]` 表映射
    Site(String),
    /// CLI `--set` / 旗标
    Cli,
    /// 意图派生(rule = 依据)
    Derived(String),
}

impl Origin {
    pub fn describe(&self) -> String {
        match self {
            Self::Default => "default(代码缺省)".into(),
            Self::File(p) => format!("file:{p}"),
            Self::Site(p) => format!("site:{p}"),
            Self::Cli => "cli".into(),
            Self::Derived(r) => format!("derived:{r}"),
        }
    }
}

/// 解析元数据(`--check` / `--print-config` 的 explain 面)
#[derive(Clone, Debug, Serialize)]
pub struct IntentMeta {
    pub intent: Intent,
    /// 档路径(`-p`;None = 只有位置参数)
    pub profile: Option<String>,
    /// 派生规则(点路径 → 依据)
    pub rules: BTreeMap<String, String>,
    /// 非致命告警(自动降级/余量不足/显式覆盖被夹)
    pub warnings: Vec<String>,
}

/// 一次解析的产物
#[derive(Clone, Debug)]
pub struct ResolveMeta {
    pub intent: IntentMeta,
    pub estimate: Option<MemoryEstimate>,
}

/// 解析入口:**原地**改写 TOML 值(把 intent/site 展开成低层键)。
///
/// `origins` 由 loader 预先填好显式键(文件/CLI);本函数只补充派生键。
/// 无 `[intent]` 表 ⇒ 原样返回 `Ok(None)`(旧 flavor / testdata 零变化)。
pub fn apply(value: &mut Value, origins: &mut Origins) -> Result<Option<ResolveMeta>, String> {
    // ── ① [site]:路径/卡/端口(site 层永远覆盖前层;不参与 intent 派生)──
    apply_site(value, origins)?;

    // ── ② [intent]:没有就结束(旧档逐字节不变) ──
    let Some(intent_v) = value.as_table_mut().and_then(|t| t.remove("intent")) else {
        return Ok(None);
    };
    let intent: Intent = intent_v.try_into().map_err(|e: toml::de::Error| {
        format!(
            "[intent] 非法: {e}\n可用键:{:?}(设计卷 docs/arch/配置面-体验优化-设计与施工.md §5)",
            Intent::KEYS
        )
    })?;
    if intent.concurrency == 0 {
        return Err("[intent].concurrency = 0 非法:至少 1(1 = 单会话语义)".into());
    }

    // ── ③ 注册表与模型冲突门 ──
    let model = intent::model_profile(&intent.model)?;
    check_explicit_model(value, model)?;
    let tp = value
        .get("runtime")
        .and_then(|r| r.get("tp_ordinals"))
        .is_some();
    let hw = match intent.hardware.as_deref() {
        Some(k) => intent::hardware_profile(k)?,
        None => {
            let dev = value
                .get("runtime")
                .and_then(|r| r.get("device"))
                .cloned()
                .map(|v| v.try_into::<crate::device::DeviceSpec>())
                .transpose()
                .map_err(|e| format!("[runtime].device 非法: {e}"))?
                .unwrap_or_default();
            intent::infer_hardware(&dev).ok_or_else(|| {
                format!(
                    "[intent].hardware 未写且无法从 [runtime].device 推断 —— \
                     请写 hardware = \"{}\"(可用:{:?})",
                    intent::RTX_3090TI_24G.key,
                    intent::hardware_keys()
                )
            })?
        }
    };

    // ── ④ 派生低层键(依赖顺序:model → 目标 → 并发 → 投机/KV/KVMem → 图 → 复用)──
    let mut warnings: Vec<String> = Vec::new();
    let mut b = Builder::default();
    b.set(
        "model.format",
        Value::String(format_name(model.format).into()),
        "intent.model → 模型注册表",
    );
    b.set(
        "model.quant",
        Value::String(quant_name(model.quant).into()),
        "intent.model → 模型注册表",
    );
    b.set(
        "model.head_quant",
        Value::String(head_name(model.head_quant).into()),
        "intent.model → 模型注册表",
    );

    let reuse = intent.reuse.unwrap_or(match intent.goal {
        Goal::MaxContext => ReuseIntent::Off,
        Goal::MaxSpeed => ReuseIntent::Global,
        Goal::Balanced => ReuseIntent::Session,
        Goal::LowMemory => ReuseIntent::Global,
    });
    match intent.goal {
        Goal::MaxContext => {
            b.set("runtime.prefill_chunk", 512i64.into(), "goal=max-context:桶上界 1024");
            b.set("runtime.prefill_tail_merge", 512i64.into(), "goal=max-context:尾块并入桶");
            b.set("pool.vram_target", 0.99f64.into(), "goal=max-context:贴顶");
            b.set("pool.vram_reserve_mb", 256i64.into(), "goal=max-context:最小守卫");
        }
        Goal::MaxSpeed => {
            b.set("runtime.prefill_chunk", 1024i64.into(), "goal=max-speed:大块摊薄固定税");
            b.set("runtime.prefill_tail_merge", 0i64.into(), "goal=max-speed:不做尾合并");
            b.set("pool.vram_target", 0.99f64.into(), "goal=max-speed:贴顶");
            b.set("pool.vram_reserve_mb", 512i64.into(), "goal=max-speed:留瞬态余量");
        }
        Goal::Balanced => {
            b.set("runtime.prefill_chunk", 512i64.into(), "goal=balanced");
            b.set("runtime.prefill_tail_merge", 512i64.into(), "goal=balanced");
            b.set("pool.vram_target", 0.97f64.into(), "goal=balanced");
            b.set("pool.vram_reserve_mb", 512i64.into(), "goal=balanced");
        }
        Goal::LowMemory => {
            b.set("runtime.prefill_chunk", 512i64.into(), "goal=low-memory:小激活面");
            b.set("runtime.prefill_tail_merge", 0i64.into(), "goal=low-memory");
            b.set("pool.vram_target", 0.99f64.into(), "goal=low-memory");
            b.set("pool.vram_reserve_mb", 512i64.into(), "goal=low-memory:留瞬态余量");
        }
    }

    // 并发单键三处同源
    let c = intent.concurrency;
    let mut max_batch = c.min(8);
    if intent.kvmem == KvmemIntent::Overcommit || tp {
        // M1 合批与选择层视图/TP 不共戴
        max_batch = 1;
    }
    if c > 8 {
        warnings.push(format!(
            "concurrency={c} > 8:max_batch 限制为 8(Concat 8 槽),其余请求按 max_concurrency 排队"
        ));
    }
    if c > 1 {
        warnings.push(format!(
            "concurrency={c}:≥2 活跃 turn 的批路走 v2 注意力(FI 臂只服务 T=1 单行);\
             非超卖档池按上下文×{c} 计"
        ));
    }
    b.set("pool.gdn_slots", (c as i64).into(), "intent.concurrency → GDN 状态格");
    b.set("runtime.max_concurrency", (c as i64).into(), "intent.concurrency → 并发闸门");
    b.set("runtime.max_batch", (max_batch as i64).into(), "intent.concurrency → 合批上限(≤8)");

    match intent.spec {
        SpecIntent::Off => {
            b.set("spec.depth", 0i64.into(), "intent.spec = off");
        }
        SpecIntent::Dflash => {
            b.set("spec.depth", 7i64.into(), "intent.spec = dflash → depth 7(生产形状)");
            if value
                .get("model")
                .and_then(|m| m.get("dflash2_dir"))
                .is_none()
            {
                return Err(
                    "[intent].spec = \"dflash\" 需要草稿目录 —— 在站点档写 \
                     `[site].draft_dir = \"…/Qwen3.8-27B-DFlash2-W4A16\"`(或 [model].dflash2_dir)"
                        .into(),
                );
            }
        }
    }

    match intent.kv {
        KvIntent::Fp8 => {
            b.set("pool.kv_fp8", true.into(), "intent.kv = fp8(容量 ×2;FI 读核前提)");
        }
        KvIntent::F16 => {
            b.set("pool.kv_fp8", false.into(), "intent.kv = f16");
        }
    }

    let mut overcommit = intent.kvmem == KvmemIntent::Overcommit;
    match intent.kvmem {
        KvmemIntent::Off => {
            for k in ["select_decode", "select_partial", "index", "coverage_probe"] {
                b.set(
                    &format!("kvmem.{k}"),
                    false.into(),
                    "intent.kvmem = off(一个键替四键)",
                );
            }
            b.set("kvmem.pool_overcommit", false.into(), "intent.kvmem = off");
            b.set("pool.host_cache_mb", 0i64.into(), "kvmem=off:无宿主 KV 腿");
            b.set("pool.disk_cache_mb", 0i64.into(), "kvmem=off:无磁盘 KV 腿");
            b.set("pool.snap_disk_mb", 0i64.into(), "kvmem=off:无磁盘状态腿");
        }
        KvmemIntent::Overcommit => {
            for k in ["select_decode", "select_partial", "index"] {
                b.set(
                    &format!("kvmem.{k}"),
                    true.into(),
                    "intent.kvmem = overcommit(虚拟化两扇门 + 索引)",
                );
            }
            b.set("kvmem.pool_overcommit", true.into(), "intent.kvmem = overcommit");
            b.set("kvmem.workspace_per_req", 1024i64.into(), "kvmem=overcommit:每 turn 工作集");
            b.set("pool.host_cache_mb", 4096i64.into(), "kvmem=overcommit:宿主 KV 腿");
            b.set("pool.snap_host_mb", 2048i64.into(), "kvmem=overcommit:宿主状态腿");
        }
    }

    // 图臂(自动降级必须打印原因)
    match intent.graphs {
        GraphIntent::Off => {
            b.set("cuda.graph", false.into(), "intent.graphs = off(eager 直发)");
            b.set("cuda.graph", false.into(), "intent.graphs = off(eager 直发)");
            b.set("kernel.fi_cuda_graph", false.into(), "intent.graphs = off");
            b.set("kernel.fi_plan_restore", true.into(), "intent.graphs = off");
            b.set("kernel.decode_arm", "v2".into(), "intent.graphs = off");
            b.set("kernel.verify_arm", "v2".into(), "intent.graphs = off");
            false
        }
        GraphIntent::Auto => {
            let fallback = if overcommit {
                Some("kvmem=overcommit 的选择层视图与 FI 图/M1 页表不共戴")
            } else if intent.kv == KvIntent::F16 {
                Some("FI 图的 plan 形状恒定需要 fp8 池(in-kernel FP8 读臂)")
            } else if tp {
                Some("TP 档的 FI 图未审")
            } else {
                None
            };
            match fallback {
                Some(reason) => {
                    warnings.push(format!(
                        "graphs=auto 已降级为 eager/piecewise:{reason}(仍保留 decode 普通图)"
                    ));
                    b.set("cuda.graph", true.into(), "graphs=auto(降级:保留普通 decode 图)");
                    b.set("cuda.graph", true.into(), "graphs=auto(降级:保留普通 decode 图)");
                    b.set("kernel.fi_cuda_graph", false.into(), "graphs=auto 降级");
                    b.set("kernel.fi_plan_restore", true.into(), "graphs=auto 降级");
                    b.set("kernel.decode_arm", "v2".into(), "graphs=auto 降级");
                    b.set("kernel.verify_arm", "v2".into(), "graphs=auto 降级:verify 回 v2");
                    false
                }
                None => {
                    b.set("cuda.graph", true.into(), "graphs=auto");
                    b.set("cuda.graph", true.into(), "graphs=auto");
                    // **单键臂**(2026-10-21):"v2" = 分页 v2 主核;prefill 桶已退役
                    b.set("kernel.decode_arm", "fi".into(), "graphs=auto:decode FI 单图");
                    if intent.spec == SpecIntent::Dflash {
                        b.set("kernel.verify_arm", "fi".into(), "graphs=auto:verify 走 FI(有投机时)");
                    } else {
                        b.set("kernel.verify_arm", "v2".into(), "graphs=auto:无投机 ⇒ verify 臂不影响,留 v2");
                    }
                    b.set("kernel.fi_cuda_graph", true.into(), "graphs=auto:图档瓦(plan 形状恒定)");
                    b.set("kernel.fi_plan_bake", true.into(), "graphs=auto:烘 plan 进图(上界 = max_seq)");
                    b.set("kernel.fi_plan_restore", false.into(), "graphs=auto:表由窗外 fi_plan_sync 刷");
                    true
                }
            }
        }
    };

    // 复用(快照/前缀缓存);虚拟化档强制前缀缓存
    let reuse = if overcommit && reuse == ReuseIntent::Off {
        warnings.push("kvmem=overcommit 需要 prefix_cache=true —— reuse=off 已被提升为 session".into());
        ReuseIntent::Session
    } else {
        reuse
    };
    match reuse {
        ReuseIntent::Off => {
            b.set("pool.prefix_cache", false.into(), "reuse=off:单会话最大上下文档");
            b.set("pool.snap_max", 0i64.into(), "reuse=off:无消费方");
            b.set("pool.snap_host_mb", 0i64.into(), "reuse=off");
            b.set("pool.host_cache_mb", 0i64.into(), "reuse=off");
        }
        ReuseIntent::Session => {
            b.set("pool.prefix_cache", true.into(), "reuse=session");
            b.set("pool.snap_max", 1i64.into(), "reuse=session:1 条边界快照");
                        b.set("pool.snap_host_mb", 1024i64.into(), "reuse=session:状态腿下沉");
        }
        ReuseIntent::Global => {
            b.set("pool.prefix_cache", true.into(), "reuse=global");
            b.set("pool.snap_max", 4i64.into(), "reuse=global:4 条快照(跨会话)");
                        b.set("pool.snap_host_mb", 3072i64.into(), "reuse=global:状态腿下沉");
            b.set("pool.host_cache_mb", 4096i64.into(), "reuse=global:KV 宿主腿");
        }
    }
    // 超卖档强制前缀缓存(否则 select_* 永远不出洞 ⇒ 池装不下就 Exhausted)
    if overcommit {
        if !get_bool(value, "pool.prefix_cache").unwrap_or(false) {
            b.set("pool.prefix_cache", true.into(), "kvmem=overcommit:虚拟化前提");
        }
        if get_usize(value, "pool.snap_max").unwrap_or(0) == 0 {
            b.set("pool.snap_max", 1i64.into(), "kvmem=overcommit:前缀复用需要快照");
        }
    }

    // ── ⑤ 显式低层键覆盖派生值(高级覆盖;provenance 仍是文件) ──
    let explicit = std::mem::replace(value, Value::Table(Default::default()));
    let mut out = b.v;
    super::merge(&mut out, explicit);
    for (path, rule) in b.rules {
        origins.entry(path).or_insert(Origin::Derived(rule));
    }
    // `overcommit` 以最终值为准(显式 [kvmem] 覆盖可改变它)
    overcommit = get_bool(&out, "kvmem.pool_overcommit").unwrap_or(overcommit);

    // ── ⑥ context 求解(显式 max_seq 优先;池不含在配置里) ──
    let estimate;
    let (act, act_note) = memory::activation_for(
        get_usize(&out, "runtime.prefill_chunk").unwrap_or(512),
        get_usize(&out, "runtime.prefill_tail_merge").unwrap_or(0),
    );
    let inp = MemoryInputs {
        total_bytes: hw.total_bytes,
        vram_target: get_f64(&out, "pool.vram_target").unwrap_or(0.97),
        reserve_bytes: (get_usize(&out, "pool.vram_reserve_mb").unwrap_or(1024) as u64) << 20,
        model,
        spec_dflash: intent.spec == SpecIntent::Dflash,
        kv_fp8: get_bool(&out, "pool.kv_fp8").unwrap_or(false),
        gdn_slots: get_usize(&out, "pool.gdn_slots").unwrap_or(8),
        snap_max: get_usize(&out, "pool.snap_max").unwrap_or(4),
        sessions: c,
        tp_world: if tp { 2 } else { 1 },
        vision: get_bool(&out, "model.vision").unwrap_or(false),
        activation_bytes: act,
        activation_note: act_note,
        stability_margin_bytes: STABILITY_MARGIN_BYTES,
    };
    let explicit_ctx = origins.contains_key("runtime.max_seq");
    let want_max = matches!(intent.context, ContextSpec::Max) && !explicit_ctx;
    // TP 的 weights/KV/GDN 按卡摊(27B TP2 权重是实测;视觉塔每 rank 全量)
    let (ctx, _solve_pool) = if want_max {
        let (ctx, pool, _cap) = memory::solve_max(&inp, model.geom.max_pos, overcommit)?;
        (ctx, pool)
    } else {
        // 显式 max_seq(文件/--set)优先于 intent.context
        let n = if explicit_ctx {
            get_usize(&out, "runtime.max_seq").unwrap_or(0)
        } else {
            match intent.context {
                ContextSpec::Tokens(n) => n,
                ContextSpec::Max => 0,
            }
        };
        if n == 0 {
            return Err("[intent].context 解析为 0 —— 写数字或 \"max\"".into());
        }
        let pool = if overcommit {
            let (_, p, _) = memory::solve_max(&inp, model.geom.max_pos, true)?;
            p
        } else {
            // 池是全局容量:每个会话都可能吃满上下文
            n.saturating_mul(c.max(1))
        };
        (n, pool)
    };
    // 逻辑池 = 上下文 × 并发(非超卖下限);**物理池**由引擎 `[vram-gov]`
    // 按"剩余显存 − reserve − fixed"自动定,不再进配置。
    let pool = ctx.saturating_mul(if overcommit { 1 } else { c.max(1) });
    if want_max && ctx < 8_192 {
        warnings.push(format!(
            "context=max 仅解出 {ctx} tok(<8192)—— 该硬件装 27B 的余量太小;\
             请换卡/换更小模型,或用 `[intent].kvmem=\"overcommit\"` 换长上下文"
        ));
    }
    let ctx_rule = format!(
        "context={}:total {:.2}G×{:.0}% − reserve {:.2}G − fixed {:.2}G − weights {:.2}G \
         = budget {:.2}G;− 激活 {} − 稳定裕度 {:.2}G ⇒ {} tok({} 取整)",
        if want_max {
            "\"max\""
        } else if explicit_ctx {
            "显式覆盖"
        } else {
            "intent.context"
        },
        hw.total_bytes as f64 / GB,
        inp.vram_target * 100.0,
        inp.reserve_bytes as f64 / GB,
        (memory::budget_parts(&inp).0).total() as f64 / GB,
        memory::weights_bytes(&inp) as f64 / GB,
        memory::budget_parts(&inp).2 as f64 / GB,
        act_note,
        inp.stability_margin_bytes as f64 / GB,
        ctx,
        memory::CONTEXT_ROUND_TOKENS,
    );
    if !origins.contains_key("runtime.max_seq") {
        set_path(&mut out, "runtime.max_seq", Value::Integer(ctx as i64));
        origins.insert("runtime.max_seq".into(), Origin::Derived(ctx_rule));
    }
    if tp {
        warnings.push(
            "TP 档:内存按**单卡口径**估算(TP2 实测 weights/KV;视觉塔每 rank 全量)。\
             真机以 [vram-gov] 为准"
                .into(),
        );
    }
    let est = memory::estimate(&inp, ctx, pool, !overcommit)?;
    warnings.extend(est.notes.iter().cloned());
    estimate = Some(est);

    *value = out;
    Ok(Some(ResolveMeta {
        intent: IntentMeta {
            intent,
            profile: None,
            rules: origins
                .iter()
                .filter_map(|(k, o)| match o {
                    Origin::Derived(r) => Some((k.clone(), r.clone())),
                    _ => None,
                })
                .collect(),
            warnings,
        },
        estimate,
    }))
}

const GB: f64 = 1073741824.0;

// ============================================================================
// 派生器内部
// ============================================================================

struct Builder {
    v: Value,
    rules: BTreeMap<String, String>,
}

impl Default for Builder {
    fn default() -> Self {
        Self { v: Value::Table(Default::default()), rules: BTreeMap::new() }
    }
}

impl Builder {
    fn set(&mut self, path: &str, val: Value, rule: impl Into<String>) {
        set_path(&mut self.v, path, val);
        self.rules.insert(path.to_string(), rule.into());
    }
}

/// `[site]` → 低层键(site 层覆盖前层;未知键拒)
fn apply_site(value: &mut Value, origins: &mut Origins) -> Result<(), String> {
    let Some(site_v) = value.as_table_mut().and_then(|t| t.remove("site")) else {
        return Ok(());
    };
    let site = site_v
        .as_table()
        .ok_or_else(|| format!("[site] 必须是表,得到 {}", super::toml_type_name(&site_v)))?;
    for (k, v) in site {
        let path = match k.as_str() {
            "model_dir" => "model.path",
            "draft_dir" => "spec.draft_path",
            "bind" => "server.bind",
            // 卡位单串 → `[parallel] device = [..]`(site 层只给一项)
            "device" => "parallel.device",
            other => {
                return Err(format!(
                    "[site].{other} 未登记(可用:model_dir | draft_dir | bind | device)"
                ))
            }
        };
        // `[site] device` 接受单串或数组:配置面统一是 `[parallel] device = [..]`
        // (单串 = 一项,免去站点档记两种类型)。
        let v = if path == "parallel.device" {
            match v {
                toml::Value::String(s) => toml::Value::Array(vec![toml::Value::String(s.clone())]),
                other => other.clone(),
            }
        } else {
            v.clone()
        };
        set_path(value, path, v);
        origins.insert(path.into(), Origin::Site("[site]".into()));
    }
    Ok(())
}

/// 显式 `[model]` 三键必须与 intent.model **同值**(同值 = 无害的显式复述;
/// 不同值 = 两个真源,拒)
fn check_explicit_model(
    v: &Value,
    p: &intent::ModelProfile,
) -> Result<(), String> {
    let Some(m) = v.get("model").and_then(|x| x.as_table()) else {
        return Ok(());
    };
    for (key, want) in [
        ("format", format_name(p.format)),
        ("quant", quant_name(p.quant)),
        ("head_quant", head_name(p.head_quant)),
    ] {
        let Some(val) = m.get(key) else { continue };
        let got = val.as_str().ok_or_else(|| {
            format!("[model].{key} 必须是字符串(得到 {})", super::toml_type_name(val))
        })?;
        if got != want {
            return Err(format!(
                "[intent].model = {:?} 派生 [model].{key} = {want:?},但显式写的是 {got:?} \
                 —— 模型档只能有一个真源(要换模型请改 intent.model)",
                p.key
            ));
        }
    }
    Ok(())
}

fn format_name(f: super::ModelFormat) -> &'static str {
    match f {
        super::ModelFormat::Safetensors => "safetensors",
        super::ModelFormat::Awq => "awq",
        super::ModelFormat::Gguf => "gguf",
    }
}

fn quant_name(q: super::QuantScheme) -> &'static str {
    match q {
        super::QuantScheme::W4a16 => "w4a16",
        super::QuantScheme::W4a8 => "w4a8",
        super::QuantScheme::W8a8i => "w8a8i",
        super::QuantScheme::W8a8f => "w8a8f",
        super::QuantScheme::W8a8 => "w8a8",
        super::QuantScheme::Q4k => "q4k",
        super::QuantScheme::Iq4 => "iq4",
        super::QuantScheme::Iq3 => "iq3",
    }
}

fn head_name(h: super::HeadQuant) -> &'static str {
    match h {
        super::HeadQuant::Off => "off",
        super::HeadQuant::RtnInt4G32 => "rtn-int4-g32",
        super::HeadQuant::RtnInt8Row => "rtn-int8-row",
    }
}

// ── TOML 路径读写(点路径;只在派生面用) ──

pub(crate) fn get<'a>(v: &'a Value, path: &str) -> Option<&'a Value> {
    let mut cur = v;
    for seg in path.split('.') {
        cur = cur.get(seg)?;
    }
    Some(cur)
}

pub(crate) fn get_usize(v: &Value, path: &str) -> Option<usize> {
    get(v, path).and_then(|x| x.as_integer()).and_then(|n| usize::try_from(n).ok())
}

pub(crate) fn get_bool(v: &Value, path: &str) -> Option<bool> {
    get(v, path).and_then(|x| x.as_bool())
}

pub(crate) fn get_f64(v: &Value, path: &str) -> Option<f64> {
    get(v, path).and_then(|x| match x {
        Value::Float(f) => Some(*f),
        Value::Integer(i) => Some(*i as f64),
        _ => None,
    })
}

pub(crate) fn set_path(v: &mut Value, path: &str, val: Value) {
    let mut cur = v;
    let mut segs = path.split('.').peekable();
    while let Some(seg) = segs.next() {
        if segs.peek().is_none() {
            cur.as_table_mut()
                .expect("配置根/中间节点必须是表(内部不变量)")
                .insert(seg.into(), val);
            return;
        }
        let table = cur
            .as_table_mut()
            .expect("配置根/中间节点必须是表(内部不变量)");
        cur = table
            .entry(seg)
            .or_insert_with(|| Value::Table(Default::default()));
    }
}

// ============================================================================
// 测试(纯函数;无 GPU)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn resolve_str(s: &str) -> (Value, Origins) {
        let mut v: Value = toml::from_str(s).expect("toml");
        let mut origins = Origins::new();
        super::apply(&mut v, &mut origins).expect("resolve");
        (v, origins)
    }

    /// 27B/3090 意图档(旧 `profiles/qwen38_27b-3090-w4a16.toml` 的 `[intent]` 块;
    /// 2026-10-21 档已转全量显式,意图面留在本测试里做**派生器对照锚**)。
    const INTENT_AWQ_3090_MAX: &str = r#"
[intent]
model       = "qwen3.8-27b-awq"
hardware    = "3090ti-24g"
goal        = "max-context"
context     = "max"
concurrency = 1
spec        = "off"
kv          = "fp8"
kvmem       = "off"
graphs      = "auto"
reuse       = "off"

# 测速/恒等门口径(与仓库档一致;高级覆盖:意图档也允许写任意低层键)
[sampling]
mode = "greedy"
"#;

    /// **P0 验收**:意图派生结果 = 仓库全量档 `profiles/qwen38_27b-3090-w4a16.toml`
    /// 的逐键值(准入由 tag filter 派生)。
    #[test]
    fn profile_matches_handwritten_flavor() {
        let (p, _po) = resolve_str(INTENT_AWQ_3090_MAX);
        let c: crate::config::OwlConfig = p.try_into().expect("intent 反序列化");
        let f = crate::config::OwlConfig::load(
            Some(&crate::profiles::profile_path("qwen38_27b-3090-w4a16").expect("profile")),
            None,
        )
        .expect("全量档装载");
        // 模型面
        assert_eq!(c.model.format, f.model.format);
        assert_eq!(c.model.quant, f.model.quant);
        assert_eq!(c.model.head_quant, f.model.head_quant);
        // 上下文/池
        assert_eq!(c.runtime.max_seq, f.runtime.max_seq, "128k");
        assert_eq!(c.runtime.prefill_chunk, f.runtime.prefill_chunk);
        assert_eq!(c.runtime.prefill_tail_merge, f.runtime.prefill_tail_merge);
        assert_eq!(c.runtime.max_batch, f.runtime.max_batch);
        assert_eq!(c.pool.gdn_slots, f.pool.gdn_slots);
        assert_eq!(c.runtime.max_concurrency, f.runtime.max_concurrency);
        assert_eq!(c.pool.vram_target, f.pool.vram_target);
        assert_eq!(c.pool.vram_reserve_mb, f.pool.vram_reserve_mb);
        // prefix_cache 由 intent.reuse 派生(本档 reuse=off);手写档继承
        // full.toml 的 true —— 两者**有意不同**,不在此门对拍。
        assert_eq!(c.pool.snap_max, f.pool.snap_max);
        assert_eq!(c.pool.host_cache_mb, f.pool.host_cache_mb);
        assert_eq!(c.pool.snap_host_mb, f.pool.snap_host_mb);
        // 臂/KV/图/投机
        assert_eq!(c.pool.kv_fp8, f.pool.kv_fp8);
        assert_eq!(c.kernel.gdn_chunked, f.kernel.gdn_chunked);
        assert_eq!(c.kernel.decode_arm, f.kernel.decode_arm);
        assert_eq!(c.kernel.verify_arm, f.kernel.verify_arm);
        assert_eq!(c.kernel.fi_cuda_graph, f.kernel.fi_cuda_graph);
        assert_eq!(c.kernel.fi_plan_bake, f.kernel.fi_plan_bake);
        assert_eq!(c.kernel.fi_plan_restore, f.kernel.fi_plan_restore);
        assert_eq!(c.spec.depth, f.spec.depth);
        assert_eq!(c.kvmem.select_decode, f.kvmem.select_decode);
        assert_eq!(c.kvmem.select_partial, f.kvmem.select_partial);
        assert_eq!(c.kvmem.index, f.kvmem.index);
        assert_eq!(c.kvmem.pool_overcommit, f.kvmem.pool_overcommit);
        assert_eq!(c.sampling.mode, f.sampling.mode);
        assert_eq!(c.runtime.prefill_chunk, 512);
    }

    /// `--check` 的估算必须与手写档 boot 现场对齐(128k / 池 4G / fixed 0.90G)
    #[test]
    fn profile_estimate_matches_boot_ledger() {
        let mut v: Value = toml::from_str(INTENT_AWQ_3090_MAX).expect("toml");
        let mut origins = Origins::new();
        let meta = super::apply(&mut v, &mut origins).expect("resolve").expect("intent");
        let e = meta.estimate.expect("estimate");
        assert_eq!(e.context_tokens, 131_072);
        assert_eq!(e.pool_tokens, e.context_tokens, "逻辑池 = 上下文(物理池自动)");
        assert_eq!(e.pool_bytes, 4 << 30);
        assert_eq!(e.fixed_bytes, (926 << 20) + (1 << 19), "boot 实测 0.90G = 926.5 MiB");
        assert!((e.budget_bytes as f64 / GB - 5.95).abs() < 0.02, "budget={:?}", e.budget_bytes);
        assert_eq!(e.stability_margin_bytes, STABILITY_MARGIN_BYTES);
        assert!(e.slack_bytes >= 0, "导出的 128k 应有正余量:{e:?}");
    }

    /// `[site]` 映射:路径/端口/卡位;未知键拒
    #[test]
    fn site_table_maps_and_can_override() {
        let (v, origins) = resolve_str(
            r#"
[intent]
model = "qwen3.8-27b-awq"
hardware = "3090ti-24g"
goal = "max-context"
context = 8192
[site]
model_dir = "/models/27b"
bind = "127.0.0.1:8140"
device = { bdf = "42:00.0" }
"#,
        );
        assert_eq!(get(&v, "model.path").and_then(|x| x.as_str()), Some("/models/27b"));
        assert_eq!(get(&v, "server.bind").and_then(|x| x.as_str()), Some("127.0.0.1:8140"));
        assert!(matches!(origins.get("server.bind"), Some(Origin::Site(_))));
        let mut bad: Value = toml::from_str("[site]\nmodel_dirz = \"x\"\n").expect("toml");
        let e = super::apply(&mut bad, &mut Origins::new()).expect_err("未知 site 键拒");
        assert!(e.contains("model_dirz"), "{e}");
    }

    /// 显式 context + concurrency:非超卖池 = 上下文 × 并发
    #[test]
    fn explicit_context_scales_pool_by_concurrency() {
        let (v, _o) = resolve_str(
            r#"
[intent]
model = "qwen3.8-27b-awq"
hardware = "3090ti-24g"
goal = "max-context"
context = 32768
concurrency = 2
"#,
        );
        assert_eq!(get_usize(&v, "runtime.max_seq"), Some(32_768));
        assert_eq!(get_usize(&v, "pool.gdn_slots"), Some(2));
        assert_eq!(get_usize(&v, "runtime.max_batch"), Some(2));
    }

    /// CLI `--set` 层(loader 已合并,origin=Cli)覆盖派生值
    #[test]
    fn cli_override_beats_derived() {
        let mut v: Value = toml::from_str(
            r#"
[intent]
model = "qwen3.8-27b-awq"
hardware = "3090ti-24g"
goal = "max-context"
context = "max"
"#,
        )
        .expect("toml");
        // 模拟 loader 的 --set 层:先合并,并记 Cli 来源
        let cli: Value = toml::from_str("[runtime]\nprefill_chunk = 1024\n").expect("cli");
        super::super::merge(&mut v, cli);
        let mut origins = Origins::new();
        origins.insert("runtime.prefill_chunk".into(), Origin::Cli);
        super::apply(&mut v, &mut origins).expect("resolve");
        assert_eq!(get_usize(&v, "runtime.prefill_chunk"), Some(1024));
        assert_eq!(origins.get("runtime.prefill_chunk"), Some(&Origin::Cli));
        // 激活面随之变化(1024 档 1.05G),context=max 解更小(96k)
        assert_eq!(get_usize(&v, "runtime.max_seq"), Some(98_304));
    }

    /// 意图与显式 [model] 冲突 = 拒(模型只能有一个真源)
    #[test]
    fn intent_model_conflicts_with_explicit_model() {
        let mut v: Value = toml::from_str(
            r#"
[intent]
model = "qwen3.8-27b-awq"
hardware = "3090ti-24g"
context = 4096
[model]
format = "safetensors"
"#,
        )
        .expect("toml");
        let e = super::apply(&mut v, &mut Origins::new()).expect_err("冲突必拒");
        assert!(e.contains("真源") && e.contains("intent.model"), "{e}");
    }

    /// graphs=auto 在 f16 池上必须降级且打印原因(不许静默错档)
    #[test]
    fn graphs_auto_falls_back_on_f16() {
        let (v, _o) = resolve_str(
            r#"
[intent]
model = "qwen3.8-27b-awq"
hardware = "3090ti-24g"
goal = "max-speed"
context = 8192
kv = "f16"
graphs = "auto"
"#,
        );
        assert_eq!(get_bool(&v, "kernel.fi_cuda_graph"), Some(false));
        assert_eq!(get(&v, "kernel.decode_arm").and_then(|x| x.as_str()), Some("v2"));
        assert!(get(&v, "graph.prefill_bucket").is_none(), "prefill 桶已退役");
    }

    /// 旧档(无 intent)零变化:apply 返回 None 且不增删键
    #[test]
    fn legacy_config_is_untouched() {
        let src = "[runtime]\nmax_seq = 4096\n[pool]\ngdn_slots = 2\n";
        let mut v: Value = toml::from_str(src).expect("toml");
        let before = v.clone();
        let mut origins = Origins::new();
        assert!(super::apply(&mut v, &mut origins).expect("ok").is_none());
        assert_eq!(v, before);
        assert!(origins.is_empty());
    }

    /// 超卖档:虚拟化开关与前缀缓存被强制打开(物理池由引擎按剩余显存自动定)
    #[test]
    fn overcommit_flags_and_prefix_cache() {
        let (v, _o) = resolve_str(
            r#"
[intent]
model = "qwen3.8-27b-awq"
hardware = "3090ti-24g"
goal = "low-memory"
context = "max"
kvmem = "overcommit"
"#,
        );
        let ctx = get_usize(&v, "runtime.max_seq").expect("ctx");
        assert_eq!(ctx, 262_144, "超卖档逻辑上下文可到模型上限");
        assert_eq!(get_bool(&v, "pool.prefix_cache"), Some(true));
        assert_eq!(get_bool(&v, "kvmem.select_decode"), Some(true));
        assert_eq!(get_bool(&v, "kvmem.pool_overcommit"), Some(true));
        // 选择层档不得建 FI 桶图
        assert!(get(&v, "graph.prefill_bucket").is_none(), "prefill 桶已退役");
    }
}

//! **GGUF 逐张量量化计划**(`PlanTable` 实现;2026-10-19 刀 7b)。
//!
//! GGUF 的量化型是**逐张量**事实(`Entry.ty`),而 owl 的层只声明"这个键
//! 是什么角色 + 形状"⇒ 计划表 = **键 → 计划** 的翻译层。与
//! [`GgufConvention`](super::convention) 同一分工:
//!
//! | 面 | 谁答 | 出处 |
//! |---|---|---|
//! | 键**名** | `GgufConvention`(`KeyConvention`) | `blk.N.attn_q.weight` |
//! | 键**计划** | 本模块(`PlanTable`) | `QuantPlan::Gguf(Q6K)` / `F16` |
//! | 键 → 字节 | [`GgufSource`](super::source)(`WeightSource`) | 压缩块原样 |
//!
//! # 三条**响亮拒**(全部走 `unresolved`,由 `Model::new` 在构造期一次性拒)
//!
//! 1. **缺键**:声明了索引里没有的键(约定漏臂 / 拓扑差异);
//! 2. **形状不符**:索引 `[rows × inner]` ≠ 声明 `[out × in]`(翻转律失守、
//!    或 owl 的融合声明与 GGUF 的分键形态不一致);
//! 3. **型未移植**:量化型在场但内核族没有入口(`entry_for = None`)
//!    —— 「解析得了 ≠ 跑得了」,两者分离(设计 §L3)。
//!
//! 为什么在**构造期**判而不在装载期:计划定形 ⇒ 槽形状定形 ⇒ 显存账定形;
//! 若此刻静默落 F16,账会在装载后才露出(设计 I-2)。
//!
//! ⚠️ **F32/F16/BF16 走 `QuantPlan::F16`**(原生臂)—— 不是"缺省降级":
//! 索引里的型本身就是原生型,精度损失 = f16 舍入(与既有 HF 路径同律)。

use std::collections::BTreeMap;
use std::sync::Mutex;

use crate::contract::ModelError;
use crate::module::{PlanTable, QuantPlan};

use super::reader::Gguf;
use super::shard::{role_of, Role, ShardSel};
use super::transform::GgufDims;
use owl_kernels::family::gguf::{entry_for, GgmlTy};

/// GGUF 计划表(借用索引;**无状态**读写面 `&self` ⇒ `unresolved` 内部可变)
pub struct GgufPlan<'g> {
    g: &'g Gguf,
    /// **TP 档的形状换算事实**(`None` = 单卡:声明维 == 文件维)。
    ///
    /// 分片档的层声明的是**分片维**(`spec.sharded` 先行),而文件是**全量维**
    /// ⇒ 形状门必须按角色把声明维**放大 world 倍**再比(切哪一维由角色表答;
    /// 见 `super::shard`)。
    tp: Option<(GgufDims, ShardSel)>,
    /// 模型问过而表答不了/答不全的键(键 → 原因)。`BTreeMap` 为确定性排序
    /// (诊断面稳定;`Model::new` 也再排一次)。
    unresolved: Mutex<BTreeMap<String, String>>,
}

impl<'g> GgufPlan<'g> {
    pub fn new(g: &'g Gguf) -> Self {
        Self { g, tp: None, unresolved: Mutex::new(BTreeMap::new()) }
    }

    /// **TP 档计划表**(形状门按角色放大;`sel.world == 1` 等价 [`Self::new`])
    pub fn for_shard(g: &'g Gguf, dims: GgufDims, sel: ShardSel) -> Self {
        let tp = (sel.world > 1).then_some((dims, sel));
        Self { g, tp, unresolved: Mutex::new(BTreeMap::new()) }
    }

    /// 声明维(分片)→ 文件维(全量):按角色把**被切的那一维**放大 `world` 倍
    fn inflate(&self, key: &str, out_dim: usize, in_dim: usize)
        -> Result<(usize, usize), String>
    {
        let Some((dims, sel)) = self.tp else { return Ok((out_dim, in_dim)) };
        let role = role_of(key, &dims, sel.head_split).map_err(|e| e.to_string())?;
        let w = sel.world;
        Ok(match role {
            Role::Replicate => (out_dim, in_dim),
            Role::Out | Role::OutHead | Role::Segs { .. } => (out_dim * w, in_dim),
            Role::In | Role::InHead => (out_dim, in_dim * w),
        })
    }

    /// 记一条未解析(锁内只做插入,无 I/O ⇒ 中毒恢复取回数据即可;
    /// 见 `GgufSource::row_map` 同款纪律)
    fn note(&self, key: &str, why: &str) {
        let mut m = self.unresolved.lock().unwrap_or_else(|e| e.into_inner());
        m.entry(key.to_string()).or_insert_with(|| why.to_string());
    }

    /// 键 → ggml 型(诊断/门用;`None` = 索引无此键)
    pub fn ty_of(&self, key: &str) -> Option<GgmlTy> {
        self.g.get_tensor(key).map(|t| t.ty)
    }
}

impl PlanTable for GgufPlan<'_> {
    fn plan_at(&self, full_key: &str, out_dim: usize, in_dim: usize) -> QuantPlan {
        let Some(t) = self.g.get_tensor(full_key) else {
            self.note(full_key, "GGUF 索引里没有该键(约定漏臂 / 拓扑差异?)");
            return QuantPlan::F16;
        };
        // 形状门:索引是 GGML 序(`ne[0]` = 内维)⇒ 与声明的 [out, in] 直比
        // (TP 档先按角色把声明维放大到全量维)
        let (want_out, want_in) = match self.inflate(full_key, out_dim, in_dim) {
            Ok(v) => v,
            Err(e) => {
                self.note(full_key, &format!("分片角色推导失败:{e}"));
                return QuantPlan::F16;
            }
        };
        if t.rows() as usize != want_out || t.inner() as usize != want_in {
            self.note(
                full_key,
                &format!(
                    "形状不符:索引 [{}×{}](GGML ne{:?})vs 声明 [{want_out}×{want_in}]{}",
                    t.rows(),
                    t.inner(),
                    t.ne,
                    if self.tp.is_some() { "(声明维已按 TP 角色放大)" } else { "" }
                ),
            );
            return QuantPlan::F16;
        }
        if t.ty.native_dtype().is_some() {
            return QuantPlan::F16; // 原生型(F32/F16/BF16)⇒ f16 直读臂
        }
        if entry_for(t.ty).is_none() {
            // 解析得了 ≠ 跑得了;落 F16 只为"槽形态合法",**必然被拒**(unresolved)
            self.note(
                full_key,
                &format!("型 {} 未移植(内核族无入口;PORTED 见 family::gguf)", t.ty.name()),
            );
            return QuantPlan::F16;
        }
        QuantPlan::Gguf(t.ty)
    }

    fn take_unresolved(&self) -> Vec<String> {
        let mut m = self.unresolved.lock().unwrap_or_else(|e| e.into_inner());
        let out: Vec<String> = m
            .iter()
            .map(|(k, why)| format!("{k} ({why})"))
            .collect();
        m.clear();
        out
    }
}

/// 计划表自检(M1 门用;不依赖模型:**索引直接问**)
///
/// 返回 `(型直方图, 未移植型清单)` —— 判据是"**索引里出现的量化型**都必须
/// 有内核入口",而不是"索引里不许有其它型"(F32 是常态)。
pub fn ty_histogram(g: &Gguf) -> (Vec<(GgmlTy, usize)>, Vec<(GgmlTy, usize)>) {
    let mut all: BTreeMap<u32, usize> = BTreeMap::new();
    for t in &g.tensors {
        *all.entry(t.ty as u32).or_default() += 1;
    }
    let mut hist = Vec::new();
    let mut missing = Vec::new();
    for (&code, &n) in &all {
        let Some(ty) = GgmlTy::from_code(code) else {
            continue; // 解析层已拒未知型(reader 的响亮度由它负责)
        };
        hist.push((ty, n));
        if ty.native_dtype().is_none() && entry_for(ty).is_none() {
            missing.push((ty, n));
        }
    }
    (hist, missing)
}

/// 计划表构造的 `Result` 壳(把"未移植型"从诊断面提到**响亮拒**面:
/// 引擎接线期一次判死,而不是等 `Model::new` 的 unresolved 汇总)
pub fn require_portable(g: &Gguf) -> Result<Vec<(GgmlTy, usize)>, ModelError> {
    let (hist, missing) = ty_histogram(g);
    if !missing.is_empty() {
        let list: Vec<String> = missing
            .iter()
            .map(|(ty, n)| format!("{}×{n}", ty.name()))
            .collect();
        return Err(ModelError::Msg(format!(
            "GGUF 计划表:文件含未移植的量化型({};本仓 PORTED = Q8_0/Q4_K/Q5_K/Q6_K)\
             —— 解析得了 ≠ 跑得了,响亮拒(设计 §L3)",
            list.join(", ")
        )));
    }
    Ok(hist)
}

#[cfg(test)]
mod tests {
    use super::*;
    use owl_shared::testkit::{resource, resource_ok};
    use std::path::PathBuf;

    fn open(key: &str) -> Option<Gguf> {
        resource_ok(key).then(|| Gguf::open(&PathBuf::from(resource(key).unwrap())).expect("open"))
    }

    /// **型直方图 + 未移植门**(零设备;刀 7b 门之一)
    #[test]
    fn histogram_and_portability_gate() {
        for key in ["gguf:4b", "gguf:27b"] {
            let Some(g) = open(key) else {
                owl_shared::testkit::report::skip("资源清单缺 \"{key}\"");
                continue;
            };
            let hist = require_portable(&g).unwrap_or_else(|e| panic!("[{key}] {e}"));
            let total: usize = hist.iter().map(|(_, n)| n).sum();
            assert_eq!(total, g.tensors.len(), "[{key}] 直方图覆盖全部张量");
            eprintln!(
                "[{key}] 型直方图:{:?}",
                hist.iter().map(|(t, n)| format!("{}×{n}", t.name())).collect::<Vec<_>>()
            );
        }
    }

    /// **计划表逐键覆盖**:模型每个声明键都能拿到合法计划(4B 全模型)
    #[test]
    fn plan_covers_declared_keys() {
        use crate::contract::Dtype;
        use crate::formats::gguf::{spec_from_gguf, GgufConvention};
        use crate::model::Model;
        use crate::module::{Loadable, LoaderCtx, PlanSource, SourceFacts};
        let Some(g) = open("gguf:4b") else {
            owl_shared::testkit::report::skip("资源清单缺 \"gguf:4b\"");
            return;
        };
        let (spec, _rep) = spec_from_gguf(&g).expect("spec");
        let plan = GgufPlan::new(&g);
        let keys = GgufConvention::new();
        let src = PlanSource::keyed(&plan, &keys, SourceFacts::GGUF);
        let model = Model::new(&spec, keys, src, src).expect("Model::new(计划表全覆盖)");
        let ctx = LoaderCtx { dtype: Dtype::F16, ..Default::default() };
        let ops = model.layout(&ctx);
        // 声明数(4B = 402 件)与计划表查询数同量级
        assert!(ops.wants().len() > 300, "声明太少:{}", ops.wants().len());
        assert!(plan.take_unresolved().is_empty(), "计划表仍有未解析键");
        eprintln!("[gguf:4b] 计划表覆盖绿(声明 {} 件)", ops.wants().len());
    }
}

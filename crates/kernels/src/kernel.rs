//! **算子值 + 构造面**(G 轮 2026-10-13 自 `list.rs` 拆出)。
//!
//! 三件事:
//! - [`LaunchShape`]:发射配置(grid 哨兵 (0,0,0) = 解释层按输出元素数自动 1D);
//! - [`Kernel`]:一次发射的全部输入(名字 + 资产 id + 发射配置 + 准入声明);
//! - 构造查询面:[`lookup`] / [`source`] / [`kernel`] / [`kernel_with`] /
//!   [`with_pick`] / [`launch_shape`](上层经 `models::kernel` 垫子续用旧路径)。
//!
//! **F 轮后名字与源码都只来自声明面**(`ops::meta`)—— 本模块不持有任何表。
//! `.cu` 取证面(源文本读盘 / 资产符号清单 / 签名解析)另住 [`crate::cu_sig`]。
//!
//! 对位契约(`roadmap.local/api-stabilize-plan.md`):标量形参宽度必须与
//! `Arg` 严格对位(`arg_usize` ↔ `size_t` 8B;`arg_f32` ↔ `float`;
//! `arg_i32` ↔ `int`),**输出块固定末参**。

/// 发射配置(grid 哨兵 (0,0,0) = 自动 1D)
#[derive(Clone, Debug)]
pub struct LaunchShape {
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub shared_mem: u32,
}

impl Default for LaunchShape {
    fn default() -> Self {
        Self { grid: (0, 0, 0), block: (256, 1, 1), shared_mem: 0 }
    }
}

#[derive(Clone, Debug)]
pub struct Kernel {
    pub name: &'static str,
    pub src: &'static str,
    pub launch: LaunchShape,
    /// 逃生舱签名(非注册 kernel 必带;格式同 Entry.args,含末位输出 T)。
    /// 注册 kernel 此字段留空 —— 槽序权威取注册表/算子元数据。
    pub sig: &'static str,
    /// **算子准入声明**(2026-10-13 P3):已建表算子由
    /// [`with_pick`](crate::kernel::with_pick) 从 `KernelPick` 带入;
    /// 未建表(仅引入)与逃生舱核为 `None`。
    pub meta: Option<&'static crate::spec::KernelMeta>,
}

impl Kernel {
    pub fn new(name: &'static str, src: &'static str) -> Self {
        // F 轮:准入声明是槽序/几何的唯一真源 ⇒ 构造期**自动挂载**
        // (名字已在声明面就带上;逃生舱核自然为 None,走 `sig`)。
        Self { name, src, launch: LaunchShape::default(), sig: "", meta: crate::ops::meta(name) }
    }

    /// 显式发射配置(覆盖自动 1D)
    pub fn with_launch(mut self, grid: (u32, u32, u32), block: (u32, u32, u32), shared_mem: u32) -> Self {
        self.launch = LaunchShape { grid, block, shared_mem };
        self
    }

    /// 逃生舱签名(非注册 kernel 的槽序权威;注册 kernel 勿用)
    pub fn with_sig(mut self, sig: &'static str) -> Self {
        self.sig = sig;
        self
    }

    /// 挂算子准入声明(槽序/检查函数随之而来)
    pub fn with_meta(mut self, meta: &'static crate::spec::KernelMeta) -> Self {
        self.meta = Some(meta);
        self
    }

    /// **槽序权威**(F 轮后优先序):算子元数据 → 逃生舱签名。
    /// 两条都是 `'static`,零分配;有声明时它就是唯一真源
    /// (登记表已退役 —— 槽序不再有第三条落点)。
    pub fn slot_tokens(&self) -> Vec<&'static str> {
        if let Some(m) = self.meta {
            return m.slot_tokens();
        }
        // 逃生舱核(无声明,如测试/诊断核):槽序必须显式给签名
        assert!(
            !self.sig.is_empty(),
            "kernel::slot_tokens({}): 既无准入声明也无逃生舱槽序 —— 拒发",
            self.name
        );
        self.sig.split(',').collect()
    }

    /// 槽的事实来源(父张量 / 输出 / 标量)—— 与 `lower_kernel` 的装配序
    /// **单源**(同一条 token 规则,声明面与逃生舱签名共用;2026-10-15)。
    pub fn slot_sources(&self) -> Vec<crate::spec::SlotSrc> {
        crate::spec::slot_sources_of_tokens(&self.slot_tokens())
    }
}

/// **算子准入查询**(原"登记表查名字"入口;F 轮 2026-10-13 起**唯一真源
/// = 算子自托管声明**,登记表已退役)。
pub fn lookup(name: &str) -> Option<&'static crate::spec::KernelMeta> {
    crate::ops::meta(name)
}
/// 算子名 → **资产 id**(`KernelSpec::SRC` 的唯一读取口;F 轮后不再有
/// 登记表兜底 —— 未建表 = 未准入 = 编程错误)。
pub fn source(name: &str) -> &'static str {
    crate::ops::source(name)
}

/// 构造 Kernel 值(默认发射配置:grid 哨兵 = 自动 1D ceil/256)
pub fn kernel(name: &'static str) -> Kernel {
    Kernel::new(name, source(name))
}

/// 构造 Kernel 值(显式发射配置;行核 embed/rope/attn 等非 ceil/256 网格用)
pub fn kernel_with(name: &'static str, grid: (u32, u32, u32), block: (u32, u32, u32), shared_mem: u32) -> Kernel {
    Kernel::new(name, source(name)).with_launch(grid, block, shared_mem)
}

/// 构造 Kernel 值(driver 拾取形态;**新调用点一律走此入口**,名字/参数
/// 由 owl-kernels::driver 单源推导 —— 字符串核名与手抄 grid/block/smem
/// 在 models 层属遗留,迁移完归零,grep 审计入验收)。
///
/// **已建表算子**(`pick.meta()` 非空):名字/源码/槽序全部取算子自托管的
/// 元数据(foreign 家族 `SOURCE` 为空 ⇒ 源码为空串,server 走家族 FFI);
/// **未建表**(仅引入):回落登记表取源。
pub fn with_pick(p: crate::driver::KernelPick) -> Kernel {
    let launch = LaunchShape { grid: p.shape.grid, block: p.shape.block, shared_mem: p.shape.smem };
    match p.meta() {
        Some(m) => Kernel { name: m.entry, src: m.src, launch, sig: "", meta: Some(m) },
        None => Kernel { name: p.name, src: source(p.name), launch, sig: "", meta: None },
    }
}

/// 发射配置便捷(与 kernel_with 同参,LaunchShape 形态)
pub fn launch_shape(grid: (u32, u32, u32), block: (u32, u32, u32), shared_mem: u32) -> LaunchShape {
    LaunchShape { grid, block, shared_mem }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops;

    /// **每张原生表的资产与符号清单在场**(原"登记表有源"门的等价升级)。
    #[test]
    fn every_native_table_declares_out_dtype() {
        for m in ops::all() {
            if m.src.is_empty() { continue; }
            assert!(m.out_dtype().is_some(), "{}: 无输出槽 ⇒ 声明口径 dtype 不可派生", m.entry);
        }
    }

    /// **每张原生表的资产与符号清单在场**(原"登记表有源"门的等价升级)。
    #[test]
    fn knhd_variants_mirror_classic_signature() {
        // kNHD 侧**独有**条目(classic 无对偶;纯新增)。每加一条必须登
        // 在此并写理由 —— 否则下面循环会把它当“换签名”报红,谁也别想混过去。
        const KNHD_ONLY: &[(&str, &str)] = &[
            (
                "vllm_paged_attention_v2_f16_knhd_hd256",
                "KNHD 侧新增:f16 无 bs32 以外的 classic v2 对偶(fp8 档才有)",
            ),
            (
                "vllm_paged_attention_v2_f16_knhd_hd128",
                "同上",
            ),
        ];
        let mut paired = 0usize;
        for e in ops::all() {
            if !e.entry.contains("_knhd") {
                continue;
            }
            if KNHD_ONLY.iter().any(|(n, _)| *n == e.entry) {
                continue;
            }
            let classic = e.entry.replace("_knhd", "");
            let Some(c) = ops::all().find(|x| x.entry == classic.as_str()) else {
                panic!(
                    "{}: 无 classic 伙伴 `{classic}` —— 要么补对偶,要么登进 KNHD_ONLY 并写理由",
                    e.entry
                );
            };
            assert_eq!(
                c.args_str(), e.args_str(),
                "{}: 槽序与 classic 伙伴不符 —— 翻转须只换名(实参表同一)",
                e.entry
            );
            assert_eq!(c.src, e.src, "{}: 与 classic 伙伴不同源", e.entry);
            paired += 1;
        }
        // 覆盖锁:22 = 2026-10-13 F 轮实况对数(新增 kNHD 核时同步上调)
        // (C 轮拆文件后 attn 域补入 4 个 knhd NC 变体:18 → 22)
        assert_eq!(paired, 22, "配对 kNHD 变体数异常(实 {paired})—— 漏登记或误删?");
        for (n, why) in KNHD_ONLY {
            assert!(
                ops::all().any(|e| e.entry == *n),
                "KNHD_ONLY 条目 `{n}` 已不在登记表({why}),请清理白名单"
            );
        }
        eprintln!("[knhd] {paired} 个 kNHD 变体与 classic 伙伴签名逐位相同 ✓(另有 {} 条 kNHD 独有,已登记理由)", KNHD_ONLY.len());
    }

    /// **每张原生表的资产与符号清单在场**(原"登记表有源"门的等价升级)。
    #[test]
    fn lookup_known_and_unknown() {
        assert!(lookup("owl_add_f32").is_some());
        assert!(lookup("owl_not_registered").is_none());
    }

    /// **每张原生表的资产与符号清单在场**(原"登记表有源"门的等价升级)。
    #[test]
    fn op_names_are_unique() {
        let mut seen = std::collections::HashSet::new();
        for e in ops::all() {
            assert!(seen.insert(e.entry), "{} 重复声明", e.entry);
        }
    }

    /// **每张原生表的资产与符号清单在场**(原"登记表有源"门的等价升级)。
    #[test]
    fn driver_picks_are_registered() {
        use crate::driver::{self, DType};
        let names = [
            driver::gdn::gating_g(DType::F16).name,
            driver::gdn::gating_g(DType::F32).name,
            driver::gdn::l2norm(DType::F16, 1).name,
            driver::gdn::l2norm(DType::F32, 1).name,
            driver::gdn::conv_upd(DType::F16).name,
            driver::gdn::conv_upd(DType::F32).name,
            driver::gdn::delta_dec(DType::F16, 128, 1, 16, 128).name,
            driver::gdn::delta_dec(DType::F32, 128, 1, 16, 128).name,
            driver::gdn::conv_fwd(DType::F16, 6144).name,
            driver::gdn::recurrence_varlen_gqa(DType::F16, 128, 16, 128).name,
            driver::gdn::norm_act(DType::F16, 1, 2048, 128).name,
            driver::gdn::norm_act(DType::F32, 1, 2048, 128).name,
            driver::ops::sigmoid(DType::F16).name,
            driver::ops::sigmoid(DType::F32).name,
        ];
        for n in names {
            assert!(lookup(n).is_some(), "driver 拾取名未登记:{n}");
        }
    }

}

//! 构建期 cubin 资产 —— native 核的**零运行期编译**载体(2026-10-13 用户裁决)。
//!
//! ```text
//! build.rs  nvcc -fatbin  ──→  $OUT_DIR/cubins/<asset>.fatbin
//!           │
//!           ├─ 源文本 include_str! 闭包 = 编哪些 .cu(零第二份清单)
//!           └─ 生成 $OUT_DIR/cubin_assets.rs(本模块 include! 进来)
//!                        │
//!                        ▼
//!             asset_of_source(源码文本) → fatbin 字节 → cuModuleLoadData
//! ```
//!
//! # 为什么查表键是**源内容**而不是名字
//!
//! 名字是"第三份表"(算子表 / 登记表 / 资产表各一份),而源内容天然只有
//! 一份:`KernelSpec::SOURCE` 就地 `include_str!`(R5)。于是
//! "这个算子有没有 AOT 资产"退化成一次哈希比对,**不可能出现**
//! "表里加了新核但资产表忘了加"这类两处对不上(命中失败是可见的、fail-closed 的)。
//!
//! # 代际维度
//!
//! `CUBIN_ARCHS` = 构建期 `OWL_CUBIN_ARCHS` 解析结果;每个资产是**一个 fatbin**
//! (内含每代际一份 cubin,运行期由驱动按设备挑最匹配)。设备 sm ∉ 本集 ⇒
//! 装配期拒启(把 "no kernel image is available for execution on the device"
//! 从发射时提前到装配时)。

include!(concat!(env!("OUT_DIR"), "/cubin_assets.rs"));

/// **资产指纹**(构建期算出,见 `build.rs`;覆盖 cubin / xqa / gdn 全部资产)。
///
/// 为什么需要它:`include_bytes!` 指向 `$OUT_DIR` 的产物时,cargo **不保证**
/// 产物一变就重编 crate(实测踩过:fatbin 已更新、rlib 仍嵌旧字节 ⇒ 装配期
/// 布局门报"内核 0 B vs Rust 镜像 40 B")。把指纹经 `rustc-env` 发出 = 指纹变
/// ⇒ **cargo 强制重编该 crate**(rustc-env 是 cargo 跟踪的构建脚本输出)。
/// 它同时是 boot 报告的"资产代际"人眼锚点。
pub const ASSET_FP: &str = env!("OWL_ASSET_FP");

/// FNV-1a 64 —— **必须**与 `build.rs::hash_source` 逐位一致。
///
/// 两处一份算法是"两处对不上"的经典温床,故不自证:由
/// `tests/cubin_assets.rs` 对**全部**资产回算源哈希做机器锁
/// (任一资产对不上即红,构建期失配立刻可见)。
pub const fn hash_source(s: &str) -> u64 {
    let bytes = s.as_bytes();
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut i = 0;
    while i < bytes.len() {
        h ^= bytes[i] as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
        i += 1;
    }
    h
}

/// 源文本 → 资产(命中 = 该源有构建期 fatbin,可零 NVRTC 发射)。
///
/// 双判据(哈希 + 长度):64 位哈希对 20 个资产本已足够,但装配错了核
/// 是灾难级(错数值而非崩),故要求两个独立指纹同时命中。
/// 空源(`foreign` 虚核的 `KernelSource.src`)直接不命中 —— 它们是
/// 家族 runtime 的路径,不走 native 编译面。
pub fn asset(src_id: &str) -> Option<&'static CubinAsset> {
    if src_id.is_empty() {
        return None;
    }
    CUBIN_ASSETS.iter().find(|a| a.id == src_id)
}

/// 是否具备构建期资产(装配期门/覆盖率报告的谓词)
pub fn has_aot(src_id: &str) -> bool {
    asset(src_id).is_some()
}

/// **AOT 覆盖率**(登记表全量):`(已覆盖, 总数, 缺陷名)`。
///
/// 符号集取 [`crate::list::REGISTRY`] 而非 `ops::TABLES` —— 前者是**发射面**
/// 的全集(含未建表的"仅引入"入口),后者只是已建表子集。报表要的是
/// "还有哪个入口会走 NVRTC";空源(foreign 虚核)不计入。
pub fn coverage() -> (usize, usize, Vec<&'static str>) {
    let mut covered = 0usize;
    let mut total = 0usize;
    let mut missing: Vec<&'static str> = Vec::new();
    for e in crate::list::REGISTRY {
        if e.src.is_empty() {
            continue;
        }
        total += 1;
        if has_aot(e.src) {
            covered += 1;
        } else {
            missing.push(e.name);
        }
    }
    (covered, total, missing)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **自洽锁**:Rust 侧 `hash_source` 与 build.rs 侧逐位一致。
    ///
    /// E 轮后二进制不再内联 `.cu` ⇒ 回算样本从"内嵌文本"改为**磁盘上的源**
    /// (测试面允许 IO)。判据不变:算法漂移一处即红。
    #[test]
    fn generated_hashes_match_rust_hash_fn() {
        assert!(!CUBIN_ASSETS.is_empty(), "资产表为空 —— build.rs 未产 cubin?");
        for a in CUBIN_ASSETS {
            let text = crate::list::cu_text(a.id);
            assert_eq!(
                hash_source(&text),
                a.src_hash,
                "资产 {} 的源哈希与 build.rs 不一致(算法漂移)",
                a.id
            );
            assert!(asset(a.id).is_some_and(|x| x.module == a.module), "资产 {} 反查失败", a.id);
        }
    }

    /// 每个**已建表**算子的源都必须命中资产 —— **零运行期编译**的静态面。
    ///
    /// 这是"加了算子表却没人编"的机器拦截;不依赖任何运行期/GPU 状态。
    /// foreign 家族(marlin/cublas/xqa/flashinfer/gdn_*)的 `SOURCE` 是空串
    /// —— 它们走 FFI/家族 runtime,不在 native 编译面,天然豁免。
    #[test]
    fn every_declared_op_source_has_asset() {
        let mut missing: Vec<&str> = Vec::new();
        let mut native = 0usize;
        for m in crate::ops::all() {
            if m.src.is_empty() {
                continue;
            }
            native += 1;
            if !has_aot(m.src) {
                missing.push(m.entry);
            }
        }
        assert!(native > 0, "没有任何 native 算子建表?");
        assert!(
            missing.is_empty(),
            "{} 个已建表算子的源没有构建期 cubin 资产(会退化到 NVRTC): {missing:?}",
            missing.len()
        );
    }

    /// **发射面全覆盖**:登记表每一条(含未建表的"仅引入"入口)都有资产。
    /// 这是 boot 期「零运行期编译硬门」能过的前提(任一缺席即拒启)。
    #[test]
    fn registry_coverage_is_total() {
        let (covered, total, missing) = coverage();
        assert!(total >= 100, "登记表只有 {total} 条?");
        assert_eq!(
            covered,
            total,
            "{} 个登记表核无构建期 cubin 资产: {missing:?}",
            missing.len()
        );
    }
}

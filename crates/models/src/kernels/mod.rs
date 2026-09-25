//! kernel 注册表(垫子):源码之家(owl-kernels)→ 登记表 → layers/TensorOps。
//!
//! 职责(2026-09-25 用户裁决):上层(layers / TensorOps 组合面)**不再
//! 直接感知 kernel 源码** —— .cu 源只住在 owl-kernels(`cu/` 目录),
//! 本模块把源码按名登记成封闭注册表,导出给组合面使用:
//!
//! ```text
//! owl-kernels(cu/*.cu,源码之家,include_str)
//!        ↓ 导入
//! models::kernels(本模块:名字 → 源 的封闭登记表)
//!        ↓ lookup / kernel / kernel_with
//! layers + TensorOps::of(Kernel)(组合时只出现名字与发射配置)
//! ```
//!
//! 登记表是**编译期封闭常量**:新增 kernel = owl-kernels 加源 + 本表加
//! 一条;查不到名字 = 编程错误(拼错名),`source` 直接 panic(带登记
//! 表提示)—— 这发生在组合构造期,不是运行期可恢复错误。

use crate::kernel::{Kernel, LaunchShape};

/// 登记表条目:发射名 → 源码(多名字可共源;名字 = server 编译缓存键)
pub struct Entry {
    pub name: &'static str,
    pub source: &'static str,
}

use owl_kernels::sources;
use owl_kernels::sources::text;

/// 全量登记(封闭;按域分组)
pub static REGISTRY: &[Entry] = &[
    // ---- 语义算子动作表(ops.cu 母本;lower_* 一一对应)----
    Entry { name: "owl_add_f32", source: sources::OPS_F32 },
    Entry { name: "owl_mul_f32", source: sources::OPS_F32 },
    Entry { name: "owl_silu_f32", source: sources::OPS_F32 },
    Entry { name: "owl_matmul_f32", source: sources::OPS_F32 },
    Entry { name: "owl_rmsnorm_f32", source: sources::OPS_F32 },
    // ---- 文本主干(Qwen3.5 mini-demo)----
    Entry { name: "owl_embed_f32", source: text::EMBED_F32 },
    Entry {
        name: "owl_rope_interleaved_partial_f32",
        source: text::ROPE_INTERLEAVED_F32,
    },
];

/// 查登记(组合面一般用 [`source`]/[`kernel`];本函数供测试/文档)
pub fn lookup(name: &str) -> Option<&'static Entry> {
    REGISTRY.iter().find(|e| e.name == name)
}

/// 按名取源(未登记 = 编程错误,panic 提示登记表位置)
pub fn source(name: &str) -> &'static str {
    match lookup(name) {
        Some(e) => e.source,
        None => panic!(
            "kernels::source(\"{name}\"): 注册表未登记 —— 新 kernel 须先在 \
             owl-kernels cu/ 加源,再在 models::kernels::REGISTRY 登记一条"
        ),
    }
}

/// 构造 Kernel 值(默认发射配置:grid 哨兵 = 自动 1D ceil/256)
pub fn kernel(name: &'static str) -> Kernel {
    Kernel::new(name, source(name))
}

/// 构造 Kernel 值(显式发射配置;行核 embed/rope/attn 等非 ceil/256 网格用)
pub fn kernel_with(name: &'static str, grid: (u32, u32, u32), block: (u32, u32, u32), shared_mem: u32) -> Kernel {
    Kernel::new(name, source(name)).with_launch(grid, block, shared_mem)
}

/// 发射配置便捷(与 kernel_with 同参,LaunchShape 形态)
pub fn launch_shape(grid: (u32, u32, u32), block: (u32, u32, u32), shared_mem: u32) -> LaunchShape {
    LaunchShape { grid, block, shared_mem }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_entries_have_sources() {
        for e in REGISTRY {
            assert!(!e.source.is_empty(), "{} 空源", e.name);
            assert!(e.source.contains("extern \"C\" __global__"), "{} 源缺核", e.name);
            assert!(
                e.source.contains(e.name),
                "{} 源里找不到同名核(名字与源不符)", e.name
            );
        }
    }

    #[test]
    fn lookup_known_and_unknown() {
        assert!(lookup("owl_add_f32").is_some());
        assert!(lookup("owl_not_registered").is_none());
    }
}

//! Kernel:核函数值 + 注册表(垫子)。
//!
//! 两半一个主题(2026-09-26 重组:原 kernel.rs + kernels/mod.rs 合并;
//! 空目录 kernels/cuda/、interpreters/ 随删):
//!
//! ```text
//! owl-kernels(cu/*.cu,源码之家,include_str)
//!        ↓ 导入
//! kernel.rs(本模块:Kernel/LaunchShape 值 + 名字 → 源 的封闭登记表)
//!        ↓ kernel / kernel_with / source
//! layers + TensorOps::of(Kernel)(组合时只出现名字与发射配置)
//! ```
//!
//! - `name`:kernel 入口名(server 编译缓存键);
//! - `source`:.cu 源码(只经注册表导出;上层零源码感知 —— 2026-09-25
//!   用户裁决);
//! - `launch`:发射配置(grid = (0,0,0) 哨兵 = 解释层按输出元素数自动
//!   1D ceil/256;行核用 `with_launch` 显式);
//! - 后端懒编译:lower 为 LaunchMsg,server 按 (name, source) 懒编译
//!   (nvrtc → PTX → module),缓存后逐次直发。
//!
//! 登记表是**编译期封闭常量**:新增 kernel = owl-kernels 加源 + 本表加
//! 一条;查不到名字 = 编程错误(拼错名),`source` 直接 panic(带登记
//! 表提示)—— 这发生在组合构造期,不是运行期可恢复错误。
//!
//! 对位契约(C2 schema 化挂账,见 roadmap.local/api-stabilize-plan.md):
//! 标量形参宽度必须与 Arg 严格对位(arg_usize ↔ `size_t` 8B;arg_f32 ↔
//! `float`;arg_i32 ↔ `int`),**输出块固定末参**。

use owl_kernels::sources;
use owl_kernels::sources::text;

// ============================================================================
// §1 Kernel 值(核函数的身份证 + 源码 + 发射配置)
// ============================================================================

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
    pub source: &'static str,
    pub launch: LaunchShape,
}

impl Kernel {
    pub fn new(name: &'static str, source: &'static str) -> Self {
        Self { name, source, launch: LaunchShape::default() }
    }

    /// 显式发射配置(覆盖自动 1D)
    pub fn with_launch(mut self, grid: (u32, u32, u32), block: (u32, u32, u32), shared_mem: u32) -> Self {
        self.launch = LaunchShape { grid, block, shared_mem };
        self
    }
}

// ============================================================================
// §2 注册表:名字 → 源(封闭常量)
// ============================================================================

/// 登记表条目:发射名 → 源码(多名字可共源;名字 = server 编译缓存键)
pub struct Entry {
    pub name: &'static str,
    pub source: &'static str,
}

/// 全量登记(封闭;按域分组)
pub static REGISTRY: &[Entry] = &[
    // ---- 语义算子动作表(ops.cu 母本;lower_* 一一对应)----
    Entry { name: "owl_add_f32", source: sources::OPS_F32 },
    Entry { name: "owl_mul_f32", source: sources::OPS_F32 },
    Entry { name: "owl_sigmoid_f32", source: sources::OPS_F32 },
    Entry { name: "owl_silu_f32", source: sources::OPS_F32 },
    Entry { name: "owl_matmul_f32", source: sources::OPS_F32 },
    Entry { name: "owl_rmsnorm_f32", source: sources::OPS_F32 },
    // ---- 文本主干(Qwen3.5 mini-demo)----
    Entry { name: "owl_embed_f32", source: text::EMBED_F32 },
    Entry {
        name: "owl_rope_interleaved_partial_f32",
        source: text::ROPE_INTERLEAVED_F32,
    },
    Entry {
        name: "owl_narrow_strided_f32",
        source: text::ATTENTION_F32,
    },
    Entry {
        name: "owl_naive_decode_attn_f32",
        source: text::ATTENTION_F32,
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
            "kernel::source(\"{name}\"): 注册表未登记 —— 新 kernel 须先在 \
             owl-kernels cu/ 加源,再在 kernel::REGISTRY 登记一条"
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

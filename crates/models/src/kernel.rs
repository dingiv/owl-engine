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
    /// 逃生舱签名(非注册 kernel 必带;格式同 Entry.args,含末位输出 T)。
    /// 注册 kernel 此字段留空 —— 槽序权威取注册表。
    pub sig: &'static str,
}

impl Kernel {
    pub fn new(name: &'static str, source: &'static str) -> Self {
        Self { name, source, launch: LaunchShape::default(), sig: "" }
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
}

// ============================================================================
// §2 注册表:名字 → 源(封闭常量)
// ============================================================================

/// 登记表条目:发射名 → 源码 + 形参签名(多名字可共源;名字 = server 编译
/// 缓存键)。
///
/// `args` = 槽序契约的机器可读形(C2,2026-09-26):逗号分隔的形参类型序
/// 列,`T`=块参 / `sz`=8 字节标量(Arg::U64 ↔ kernel `size_t`)/ `i32` /
/// `f32`;**顺序即 LaunchMsg args 序**,对 Kernel 节点路径输出块固定末参。
/// `lower_kernel` 对表校验声明槽序;下方测试解析 .cu 实签名互证 ——
/// u32/sz 错位、输出不在末参两颗雷在这里机器拦截。
///
/// `dtype` = 块参 dtype 标注(f16 基线守门,2026-09-26):eval 发射前
/// 对账声明 dtype,不符 = 结构化报错 —— 堵死「f32 核读 f16 字节 =
/// 静默垃圾」路径。现状全族 F32;F2 起逐核加 f16 变体条目。
/// dtype = **输出/声明口径**(守门对账节点声明);混合核(F4:conv_upd /
/// delta_dec 激活 f16 + state f32)输入位宽由 .cu 源自证,守门不查入参。
pub struct Entry {
    pub name: &'static str,
    pub source: &'static str,
    pub args: &'static str,
    pub dtype: crate::contract::Dtype,
}

/// 全量登记(封闭;按域分组)
pub static REGISTRY: &[Entry] = &[
    // ---- 语义算子动作表(ops.cu 母本;lower_* 一一对应;
    //      此族经 lower_* 硬编码装配,out 位置随 .cu 签名)----
    Entry { name: "owl_add_f32", source: sources::OPS_F32, args: "T,T,T,sz", dtype: crate::contract::Dtype::F32 },
    Entry { name: "owl_mul_f32", source: sources::OPS_F32, args: "T,T,T,sz", dtype: crate::contract::Dtype::F32 },
    Entry { name: "owl_sigmoid_f32", source: sources::OPS_F32, args: "T,T,sz", dtype: crate::contract::Dtype::F32 },
    Entry { name: "owl_silu_f32", source: sources::OPS_F32, args: "T,T,sz", dtype: crate::contract::Dtype::F32 },
    Entry { name: "owl_matmul_f32", source: sources::OPS_F32, args: "T,T,T,i32,i32,i32", dtype: crate::contract::Dtype::F32 },
    Entry { name: "owl_matmul_nt_f32", source: sources::OPS_F32, args: "T,T,T,i32,i32,i32", dtype: crate::contract::Dtype::F32 },
    Entry { name: "owl_rmsnorm_f32", source: sources::OPS_F32, args: "T,T,T,i32,f32,i32", dtype: crate::contract::Dtype::F32 },
    // ---- f16 基线变体(F2;桥宏:读 half 算 float 写 half;matmul 无 f16 = cuBLAS)----
    Entry { name: "owl_add_f16", source: sources::OPS_F16, args: "T,T,T,sz", dtype: crate::contract::Dtype::F16 },
    Entry { name: "owl_mul_f16", source: sources::OPS_F16, args: "T,T,T,sz", dtype: crate::contract::Dtype::F16 },
    Entry { name: "owl_sigmoid_f16", source: sources::OPS_F16, args: "T,T,sz", dtype: crate::contract::Dtype::F16 },
    Entry { name: "owl_silu_f16", source: sources::OPS_F16, args: "T,T,sz", dtype: crate::contract::Dtype::F16 },
    Entry { name: "owl_rmsnorm_f16", source: sources::OPS_F16, args: "T,T,T,i32,f32,i32", dtype: crate::contract::Dtype::F16 },
    // ---- 模型琐核 f16 变体(F3;embed/rope 同源文件追加,narrow 在 attention.cu)----
    Entry { name: "owl_embed_f16", source: text::EMBED_F32, args: "T,T,sz,T", dtype: crate::contract::Dtype::F16 },
    Entry { name: "owl_rope_half_partial_f16", source: text::ROPE_HALF_PARTIAL_F32, args: "T,T,T,T,sz,sz,sz,T", dtype: crate::contract::Dtype::F16 },
    Entry { name: "owl_narrow_strided_f16", source: text::ATTENTION_F32, args: "T,sz,sz,sz,sz,T", dtype: crate::contract::Dtype::F16 },
    // ---- GDN + attention f16 变体(F4;state 恒 f32 混合核,输出口径 F16)----
    Entry { name: "owl_gdn_gating_g_f16", source: text::GDN_F32, args: "T,T,T,sz,sz,T", dtype: crate::contract::Dtype::F16 },
    Entry { name: "owl_gdn_l2norm_f16", source: text::GDN_F32, args: "T,sz,sz,f32,T", dtype: crate::contract::Dtype::F16 },
    Entry { name: "owl_gdn_conv_upd_f16", source: text::GDN_F32, args: "T,T,T,T,sz,sz,sz,i32,T", dtype: crate::contract::Dtype::F16 },
    Entry { name: "owl_gdn_delta_dec_f16", source: text::GDN_F32, args: "T,T,T,T,T,T,T,sz,sz,sz,sz,sz,f32,T", dtype: crate::contract::Dtype::F16 },
    Entry { name: "owl_gdn_norm_act_f16", source: text::GDN_F32, args: "T,T,T,sz,sz,sz,f32,i32,T", dtype: crate::contract::Dtype::F16 },
    Entry { name: "owl_naive_decode_attn_f16", source: text::ATTENTION_F32, args: "T,T,T,T,T,T,T,sz,sz,sz,sz,T", dtype: crate::contract::Dtype::F16 },
    // ---- PF1b 批核(chunked prefill;T 循环核内,工单 G 换源)----
    Entry { name: "owl_gdn_conv_fwd_f16", source: text::GDN_F32, args: "T,T,T,T,T,i32,i32,i32,T", dtype: crate::contract::Dtype::F16 },
    Entry { name: "owl_gdn_recurrence_varlen_gqa_f16", source: text::GDN_F32, args: "T,T,T,T,T,T,T,T,sz,sz,sz,sz,sz,f32,T", dtype: crate::contract::Dtype::F16 },
    // ---- owl 移植位(工单 N;NInfer sigmoid_gate_mul,attention 门融合)----
    Entry { name: "owl_sigmoid_gate_mul_f16", source: sources::owl::SIGMOID_GATE_MUL_F16, args: "T,T,sz,T", dtype: crate::contract::Dtype::F16 },
    // ---- 文本主干(Qwen3.5 mini-demo;Kernel 节点路径,输出块末参)----
    Entry { name: "owl_embed_f32", source: text::EMBED_F32, args: "T,T,sz,T", dtype: crate::contract::Dtype::F32 },
    Entry {
        name: "owl_rope_half_partial_f32",
        source: text::ROPE_HALF_PARTIAL_F32,
        args: "T,T,T,T,sz,sz,sz,T",
        dtype: crate::contract::Dtype::F32,
    },
    Entry {
        name: "owl_narrow_strided_f32",
        source: text::ATTENTION_F32,
        args: "T,sz,sz,sz,sz,T",
        dtype: crate::contract::Dtype::F32,
    },
    Entry {
        name: "owl_naive_decode_attn_f32",
        source: text::ATTENTION_F32,
        args: "T,T,T,T,T,T,T,sz,sz,sz,sz,T",
        dtype: crate::contract::Dtype::F32,
    },
    // ---- GDN 线性注意力(Qwen3.5 mini-demo;Kernel 节点路径,输出块末参)----
    // (beta 臂 = sigmoid(b) 复用 owl_sigmoid_f32,不登记)
    Entry {
        name: "owl_gdn_gating_g_f32",
        source: text::GDN_F32,
        args: "T,T,T,sz,sz,T",
        dtype: crate::contract::Dtype::F32,
    },
    Entry {
        name: "owl_gdn_l2norm_f32",
        source: text::GDN_F32,
        args: "T,sz,sz,f32,T",
        dtype: crate::contract::Dtype::F32,
    },
    Entry {
        name: "owl_gdn_conv_upd_f32",
        source: text::GDN_F32,
        args: "T,T,T,T,sz,sz,sz,i32,T",
        dtype: crate::contract::Dtype::F32,
    },
    Entry {
        name: "owl_gdn_delta_dec_f32",
        source: text::GDN_F32,
        args: "T,T,T,T,T,T,T,sz,sz,sz,sz,sz,f32,T",
        dtype: crate::contract::Dtype::F32,
    },
    Entry {
        name: "owl_gdn_norm_act_f32",
        source: text::GDN_F32,
        args: "T,T,T,sz,sz,sz,f32,i32,T",
        dtype: crate::contract::Dtype::F32,
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
    use crate::contract::Bytes;

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

    // ======================================================================
    // C2:登记 args ↔ .cu 实签名互证(u32/sz 错位、out 位置两雷的机器拦截)
    // ======================================================================

    /// 解析 .cu 源里 `extern "C" __global__ void <name>(...)` 的形参序列,
    /// 映射为 Entry.args 同构字符串(无 regex,朴素扫描)。
    fn parse_cu_sig(source: &str, name: &str) -> Option<String> {
        // 先全文剥行注释(形参注记含逗号/括号,会干扰边界与切分)
        let clean: String = source
            .lines()
            .map(|l| l.split("//").next().unwrap_or(""))
            .collect::<Vec<_>>()
            .join("\n");
        // 宏展开核(F5 ops_pair):实例行 `OWL_<FAMILY>_KERNEL(<name>, <T>, PARAMS)`
        // —— PARAMS 自带类型化形参,机器对账以实例为准(C2 不弱化)
        for line in clean.lines() {
            let t = line.trim();
            if t.starts_with("OWL_") && t.contains("KERNEL(") {
                if let Some(rest) = t.splitn(2, '(').nth(1) {
                    let mut parts = rest.splitn(3, ',');
                    if let Some(n) = parts.next().map(|x| x.trim()) {
                        if n == name {
                            let params = parts
                                .nth(1)
                                .and_then(|p| p.trim().strip_prefix('('))
                                .and_then(|p| p.strip_suffix(')'))
                                .unwrap_or("");
                            return Some(
                                params
                                    .split(',')
                                    .map(|p| {
                                        let p = p.replace("const ", "").replace("__restrict__ ", "").trim().to_string();
                                        if p.contains('*') {
                                            "T".to_string()
                                        } else if p.contains("size_t") || p.contains("unsigned long long") {
                                            "sz".to_string()
                                        } else if p.contains("float") {
                                            "f32".to_string()
                                        } else if p.contains("int") {
                                            assert!(!p.contains("unsigned"), "{name}: 禁止 unsigned int 形参");
                                            "i32".to_string()
                                        } else {
                                            panic!("{name}: 无法识别的宏形参 `{p}`(args 词表:T/sz/i32/f32)");
                                        }
                                    })
                                    .collect::<Vec<_>>()
                                    .join(","),
                            );
                        }
                    }
                }
            }
        }
        let anchor = format!("__global__ void {name}(");
        let start = clean.find(&anchor)? + anchor.len();
        let end = clean[start..].find(')')? + start;
        let params = clean[start..end]
            .split(',')
            .map(|p| {
                let p = p.replace("const ", "").trim().to_string();
                if p.contains('*') {
                    "T".to_string()
                } else if p.contains("size_t") || p.contains("unsigned long long") {
                    "sz".to_string()
                } else if p.contains("float") {
                    "f32".to_string()
                } else if p.contains("int") {
                    // 含 unsigned int:4 字节无符号 —— Arg 无此宽度,登记表禁止
                    assert!(!p.contains("unsigned"), "{name}: 禁止 unsigned int 形参(用 size_t;Arg::U64 8 字节对位)");
                    "i32".to_string()
                } else {
                    panic!("{name}: 无法识别的形参 `{p}`(args 词表:T/sz/i32/f32)");
                }
            })
            .collect::<Vec<_>>()
            .join(",");
        Some(params)
    }

    #[test]
    fn registry_sig_matches_cu_signatures() {
        for e in REGISTRY {
            let actual = parse_cu_sig(e.source, e.name)
                .unwrap_or_else(|| panic!("{}: .cu 源里找不到同名核", e.name));
            assert_eq!(
                actual, e.args,
                "{}: 登记签名与 .cu 实签名不符(改签名须同步登记表)",
                e.name
            );
            // Kernel 节点路径(text/ 域四核)输出块必须末参;语义算子族
            // (lower_* 硬编码装配)out 位置随 .cu 签名,不受此限
            if matches!(
                e.name,
                "owl_embed_f32" | "owl_embed_f16"
                    | "owl_rope_half_partial_f32" | "owl_rope_half_partial_f16"
                    | "owl_narrow_strided_f32" | "owl_narrow_strided_f16"
                    | "owl_naive_decode_attn_f32" | "owl_naive_decode_attn_f16"
                    | "owl_sigmoid_gate_mul_f16"
            ) {
                assert!(e.args.ends_with("T"), "{}: Kernel 节点路径输出块必须末参", e.name);
            }
        }
    }

    #[test]
    fn lower_kernel_rejects_scalar_mismatch() {
        // 标量槽宽度与登记 args 不符 → 组合期 panic(机器拦截 u32/sz 错位类雷)
        let k = kernel_with("owl_narrow_strided_f32", (0, 0, 0), (256, 1, 1), 0);
        let scalars = vec![
            crate::ops::KernelArg::I32(2), // 首标量应为 sz(Bit)—— 故意错
            crate::ops::KernelArg::Bits(3),
            crate::ops::KernelArg::Bits(4),
            crate::ops::KernelArg::Bits(5),
        ];
        let ins = vec![Bytes::new(1, 0)];
        let out = Bytes::new(9, 0);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = crate::ops::lower_kernel(&k, &scalars, &ins, &out, 0);
        }));
        assert!(result.is_err(), "标量宽度错位应 panic");
    }

    #[test]
    fn lower_kernel_rejects_parent_count_mismatch() {
        // 父依赖数 != 签名 T 槽数 → 组合期 panic
        let k = kernel_with("owl_narrow_strided_f32", (0, 0, 0), (256, 1, 1), 0);
        let scalars = vec![
            crate::ops::KernelArg::Bits(1),
            crate::ops::KernelArg::Bits(2),
            crate::ops::KernelArg::Bits(3),
            crate::ops::KernelArg::Bits(4),
        ];
        let ins = vec![Bytes::new(1, 0), Bytes::new(2, 0)]; // 应为 1 个父
        let out = Bytes::new(9, 0);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = crate::ops::lower_kernel(&k, &scalars, &ins, &out, 0);
        }));
        assert!(result.is_err(), "父数不符应 panic");
    }

    #[test]
    fn lower_kernel_happy_path_orders_slots() {
        // 正确声明 → 槽序 = 签名序(T 块 + 标量交错,输出末位)
        let k = kernel_with("owl_narrow_strided_f32", (0, 0, 0), (256, 1, 1), 0);
        let scalars = vec![
            crate::ops::KernelArg::Bits(10),
            crate::ops::KernelArg::Bits(20),
            crate::ops::KernelArg::Bits(30),
            crate::ops::KernelArg::Bits(40),
        ];
        let ins = vec![Bytes::new(7, 0)];
        let out = Bytes::new(9, 8);
        let msg = crate::ops::lower_kernel(&k, &scalars, &ins, &out, 8);
        assert_eq!(msg.args.len(), 6); // T + sz×4 + out
        assert!(matches!(&msg.args[0], owl_iface::contract::Arg::Block { id: 7 }));
        assert!(matches!(&msg.args[1], owl_iface::contract::Arg::U64(10)));
        assert!(matches!(&msg.args[5], owl_iface::contract::Arg::Block { id: 9 }));
        assert_eq!(msg.out_elems, 8);
    }
}

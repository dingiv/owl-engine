//! Dtype 与标量桥(T1 核心,主 agent 亲做)。
//!
//! 语义表(SEMANTICS,写死即法律,搬运期不许临场发挥):
//!
//! **S1 dtype 提升禁止**:二元算子要求两侧 dtype **完全一致**,不做
//! 隐式提升(candle 同律)。混合精度只经显式 `to_dtype`(转换点 =
//! 权重装载与 logits 前,调用点搬运时人工标注)。
//!
//! **S2 broadcast 右对齐**:candle 语义——shape 从右往左逐维配对,
//! 相等或其一为 1 可播,维数不足左补 1。示例:
//! `[M,K] op [K]` → `[M,K]`;`[B,M,K] op [M,1]` → `[B,M,K]`。
//! 禁 numpy 式左对齐。
//!
//! **S3 contiguous 显式**:owl Tensor 无惰性布局,stride 永远紧凑;
//! candle 的 lazy transpose/contiguous 语义翻译为:视图方法
//! (`transpose/permute`)返回**布局描述**, materialize 只发生在
//! 需要物理布局的 kernel 入口(由算子层 assert 触发),不静默拷贝。
//!
//! **S4 整数语义**:U32 = token id / 索引 / slot;I64 仅 GGUF 边界
//! (装载即转 U32);索引类算子(index_select/gather/scatter)的
//! 索引参数 **恒 U32**,其他 dtype 拒绝。
//!
//! **S5 量化占位**:Q8/F8E8M0 仅 Dtype 枚举占位(quant feature),
//! 无 Scalar impl;量化权重经 marlin-ffi 路径另行表达,不进通用
//! Tensor 算子面。

use owl_iface::MemValue;

/// 张量元素类型。
///
/// 有 Scalar impl 的变体 = 可入池可运算;`quant` feature 下的占位变体
/// (Q8/F8E8M0)只参与类型层搬运(编译口径),无设备路径。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dtype {
    F32,
    F16,
    BF16,
    U8,
    U32,
    I64,
    /// 量化占位(feature = "quant");无 Scalar impl
    Q8,
    /// FP8 占位(feature = "quant");无 Scalar impl
    F8E8M0,
}

impl Dtype {
    /// 字节宽度(布局/装载用;占位变体按其物理定义给宽)
    pub const fn size_bytes(self) -> usize {
        match self {
            Dtype::F32 => 4,
            Dtype::F16 | Dtype::BF16 | Dtype::F8E8M0 => 2,
            Dtype::U8 | Dtype::Q8 => 1,
            Dtype::U32 | Dtype::I64 => 4,
        }
    }

    pub const fn is_floating(self) -> bool {
        matches!(self, Dtype::F32 | Dtype::F16 | Dtype::BF16 | Dtype::F8E8M0)
    }

    pub const fn is_integer(self) -> bool {
        matches!(self, Dtype::U8 | Dtype::U32 | Dtype::I64 | Dtype::Q8)
    }
}

impl core::fmt::Display for Dtype {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let s = match self {
            Dtype::F32 => "f32",
            Dtype::F16 => "f16",
            Dtype::BF16 => "bf16",
            Dtype::U8 => "u8",
            Dtype::U32 => "u32",
            Dtype::I64 => "i64",
            Dtype::Q8 => "q8",
            Dtype::F8E8M0 => "f8e8m0",
        };
        f.write_str(s)
    }
}

/// 可入池标量:Rust 类型 ↔ Dtype 元数据桥(S1:二元算子用
/// `T::DTYPE == U::DTYPE` 做编译期同型断言,无运行时提升)。
pub trait Scalar: MemValue {
    const DTYPE: Dtype;
}

/// BF16 位型(bf16 = f32 的高 16 位截断;设备路径回填时用位运算,
/// host 侧仅作类型搬运与位比较,不做算术——编译口径下足够)。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Bf16(pub u16);

/// F16 位型(IEEE 754 half;同上,host 侧不解释数值)。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct F16(pub u16);

impl Scalar for f32 {
    const DTYPE: Dtype = Dtype::F32;
}
impl Scalar for F16 {
    const DTYPE: Dtype = Dtype::F16;
}
impl Scalar for Bf16 {
    const DTYPE: Dtype = Dtype::BF16;
}
impl Scalar for u8 {
    const DTYPE: Dtype = Dtype::U8;
}
impl Scalar for u32 {
    const DTYPE: Dtype = Dtype::U32;
}
impl Scalar for i64 {
    const DTYPE: Dtype = Dtype::I64;
}

/// 索引合法性(S4):索引类算子的索引张量恒 U32。
pub trait IndexScalar: Scalar {}
impl IndexScalar for u32 {}

// ---------------------------------------------------------------------------
// S6 模型层算子风格律(2026-09-22 裁决:单一风格,拒绝双轨)
//
// - 模型/层代码统一 functional-via-ctx:`fn(ctx: &KernelCtx, ...) -> Result<Tensor>`;
// - 中间张量分配唯一入口 = `ctx.scratch_tensor`(Live/Capturing 相合法;
//   捕获期创建 → token 自动 emit → 图租约;生命周期 = 本次 forward);
// - **运算符重载禁用**(Add/Mul 等):无法返回 Result,会把 CUDA 错误
//   变回 panic 面——`a + b` 一律改 `ctx.add(&a, &b)`;
// - out-param 不属于层代码风格,只存在于图边界接缝(weights = P 阶段;
//   logits_out = 租约绑定缓冲,地址稳定是物理约束);
// - Persistent 分配(权重/KV)仍只在 P 阶段经 Pool 工厂,ctx 不开此口。
// ---------------------------------------------------------------------------


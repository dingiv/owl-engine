//! Qwen3.5 文本主干 layers(容器 + load 钩子 + forward 纯声明)。
//!
//! 生命周期三段(2026-09-25 用户裁决;与 TensorOps 完全对称):
//! - **`new` = 准备容器**(零数据、零副作用:登记槽位键名/形状/布局);
//! - **`load_ops(&self)` = 装载的纯描述**(返回 LoaderOps,零副作用;
//!   执行 = `interpreter::eval_load(ops, face, src)` → LoadedBlocks;
//!   毒值/缺键/长度违约在此收割);`mount(&mut self, blocks)` = 纯内存回填;
//! - **`forward` = 纯声明**(槽 decl():未装载 → 毒值声明,eval 边界收割
//!   —— 全程 total,零 panic 零 expect)。
//!
//! 复杂算子 = Kernel 节点(源经 `crate::kernels` 注册表按名取用,层内
//! 零源码);索引/位置量(ids/pos/slots)以 f32 数值形态过线
//! (<2^24 精度无损;S4 u32 索引律的 dtype 扩展留后)。
//!
//! 模块地图(试水批):
//! - [`linear`] / [`rmsnorm`] / [`mlp`]:纯语义算子层(双 face 可对拍)
//! - [`embedding`] / [`rope`]:Kernel 节点试水件
//! - [`attention`]:M-b 批(qk-norm 复用 rmsnorm w_off + gate 切分/naive attn kernel)
//! - [`gdn`]:M-c 批(GatedDeltaNet 18/24 层;批 1 gating 起逐核小步移植)
//! - [`decoder`]:M-d 批(DecoderLayer Full/Gdn 枚举 + 双残差)
//!
//! 主干(批 8)在 [`crate::model`](共有机制);Qwen3.5 预设在
//! [`crate::formats::qwen35`]。
//!
//! 未搬(试水通过后逐个立项,勿一把梭):
//! vision ViT(进行中:`feat/vision-tower`;已落 [`layernorm`] 第一件)、MTP、chunked prefill、量化。

pub mod attention;
pub mod dflash2;
pub mod decoder;
pub mod embedding;
pub mod gdn;
/// 前缀无感删除的 KV 基元(Δ 表 + 池内 K 原地重转)
pub mod layernorm;
pub mod linear;
pub mod mlp;
pub mod mrope;
pub mod mtp;
pub mod rmsnorm;
pub mod rope;
pub mod vision;

// ============================================================================
// 层间共享 Kernel 帮手(声明构造;零状态)
// ============================================================================

use crate::TensorOps;

/// **逐元素乘常数**(f16/bf16;`2` 的幂 = 精确移位)。唯一消费者 =
/// GGUF 块型 GEMM 的**成对指数移位**(见 `Linear::forward` 的 out_dt 臂)。
pub(crate) fn scale16(x: &TensorOps, s: f32) -> TensorOps {
    let shape = x.shape().to_vec();
    let n: usize = shape.iter().product();
    TensorOps::call(crate::ops::SemanticKernel::Scale)
        .arg(x)
        .arg_i32(n as i32)
        .arg_f32(s)
        .with_shape(x.dtype, shape)
}

/// **f16 ↔ bf16 铸边界**(两向;同 dtype 直通,零节点)。
///
/// 唯一实现,两个消费者:
/// - 草稿路径(`dflash2::ensure_dt`,BF16 执行域的 cast 边界);
/// - `Linear` 的**输出 dtype 钉定**(GGUF 块型臂的 GEMM 核写 `half*` ⇒
///   BF16 草稿里必须铸回 BF16,否则下游按 BF16 读 F16 位型 = 静默错值)。
///
/// 其它 dtype 组合 = **调用方违约**(草稿执行域只有这两档),返回毒化节点
/// 而不是 panic(图内错误走毒化链)。
pub(crate) fn cast_16(x: &TensorOps, dt: crate::contract::Dtype) -> TensorOps {
    use crate::contract::Dtype;
    if x.dtype == dt {
        return x.clone();
    }
    let variant = match (x.dtype, dt) {
        (Dtype::F16, Dtype::BF16) => crate::ops::SemanticKernel::CastF16Bf16,
        (Dtype::BF16, Dtype::F16) => crate::ops::SemanticKernel::CastBf16F16,
        other => {
            return TensorOps::poisoned(
                dt,
                x.shape().to_vec(),
                format!("cast_16: 无 {other:?} 臂(草稿/GGUF 铸边界只接 f16↔bf16)"),
            )
        }
    };
    let shape = x.shape().to_vec();
    let n: usize = shape.iter().product();
    TensorOps::call(variant)
        .arg(x)
        .arg_i32(n as i32)
        .with_shape(dt, shape)
}

/// 非连续窄切物化(owl_narrow_strided_{f32,f16}):行展平
/// r ∈ [0, outer),dst[r·out_dim + d] = src[r·src_dim + start + d]。
/// attention q gate 切分 / GDN qkv 投影列切分与 conv 权重行切分共用。
/// dtype 跟随 src 声明(F5 整模切换;纯 gather 拷贝,f16 位型直搬)。
pub(crate) fn narrow_strided(
    src: &TensorOps,
    outer: usize,
    src_dim: usize,
    start: usize,
    out_dim: usize,
    shape: crate::contract::Shape,
) -> TensorOps {
    // 刀1 快路:连续切片 → SliceView(零内核零节点派发)。平铺位置
    // r·src_dim + start + d 连续 ⟺ outer==1(单行)∨ src_dim==out_dim
    // (无跨行间隙)——两者平铺域同为 [start, start + outer·out_dim)。
    // decode T=1 的全部 narrow 命中此臂;非连续(prefill 批量)仍物化。
    if outer == 1 || src_dim == out_dim {
        return src.slice_view(start, shape);
    }
    let dt = src.dtype;
    if dt == crate::tensor::Dtype::F16 {
        return TensorOps::call(crate::ops::SemanticKernel::Narrow) // 哨兵:逐元素核,自动 1D ceil/256
            .arg(src)
            .arg_usize(outer)
            .arg_usize(src_dim)
            .arg_usize(start)
            .arg_usize(out_dim)
            .with_shape(dt, shape);
    }
    // f32 语义锚链同通道(名字 = driver 按 dt 单源)
    TensorOps::call(crate::ops::SemanticKernel::Narrow)
    .arg(src)
    .arg_usize(outer)
    .arg_usize(src_dim)
    .arg_usize(start)
    .arg_usize(out_dim)
    .with_shape(dt, shape)
}

/// 行栈 concat(PF1a;owl_concat_rows_f16,arity 16):n 份同形 [r, d]
/// → [n·r, d]。n ≤ 16(展开路径 = 测试锚封顶;生产 prefill 走 PF1b 批核
/// varlen 核,大 T 不经此核)。n < 16 多余指针位重复首块(核内防读)。
/// dtype 跟随首输入(f16)。
pub fn concat_rows(inputs: &[&TensorOps], r: usize, d: usize) -> TensorOps {
    let n = inputs.len();
    assert!((1..=crate::kernel::CONCAT_ROWS_SLOTS).contains(&n), "concat_rows: arity 封顶 {},得 {n}", crate::kernel::CONCAT_ROWS_SLOTS);
    let dt = inputs[0].dtype;
    let mut k = if dt == crate::tensor::Dtype::F16 {
        TensorOps::call(crate::ops::SemanticKernel::Concat) // 哨兵:逐元素核,自动 1D ceil/256
    } else {
        // f32 语义锚链同通道
        TensorOps::call(crate::ops::SemanticKernel::Concat)
    };
    for i in 0..crate::kernel::CONCAT_ROWS_SLOTS {
        k = k.arg(inputs.get(i).copied().unwrap_or(inputs[0]));
    }
    k.arg_usize(n)
        .arg_usize(r)
        .arg_usize(d)
        .with_shape(dt, vec![n * r, d])
}

/// 层级行栈(T > 8:组内 8 栈一层,组间再栈一层;T ≤ 64 = 单层嵌套,
/// 组均匀约束由调用方的块规则保证 —— 非末块对齐 8)。
pub(crate) fn concat_rows_hier(inputs: &[&TensorOps], d: usize) -> TensorOps {
    let n = inputs.len();
    assert!((1..=64).contains(&n), "concat_rows 层级栈封顶 64,得 {n}");
    if n <= 8 {
        return concat_rows(inputs, 1, d);
    }
    let groups: Vec<TensorOps> = inputs.chunks(8).map(|g| concat_rows(g, 1, d)).collect();
    let refs: Vec<&TensorOps> = groups.iter().collect();
    concat_rows(&refs, 8, d)
}

//! 算子分发 —— REQ-HW-01(arch 分发表)与公理 A1.5(可捕获 kernel 契约)
//! 的落点。
//!
//! 借力原则(REQ-DESIGN):GEMM = marlin-ffi(xinfer P1-P3 产物,活跃所有权)
//! 直接接线;attention = attention-rs 快照;这里只定义**契约与分发表**,
//! 不实现 kernel(REQ-DESIGN-03:自研是最后手段)。

/// Arch 分发表键(REQ-HW-01):业务代码禁止写死 arch,一律经此查询。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Arch {
    /// RTX 30 系一等公民
    Sm86,
    /// RTX 40 系顺带兼容
    Sm89,
    // 新架构 = 新枚举 + 注册新实现,不改调用方
}

/// 可捕获 kernel 契约(A1.5)—— 实现者签字画押处。
///
/// 任何将进入捕获段的 kernel 实现必须满足:
/// 1. 形状参数只来自 `GraphProfile` 档位,运行时不可变;
/// 2. 动态量(seqlen/pos/frontier/KV 表)一律从 device 内存读;
/// 3. 循环按最坏情况满员发射 + 谓词化退出(predication);
/// 4. 无同步、无 D2H、无 host 分支、无裸 cudaMalloc(池化由 graph 层管)。
///
/// 违反 = 架构错误:code review 门禁 + M1 起的捕获冒烟测试兜底。
pub trait CaptureSafeKernel {
    /// 实现者声明该 kernel 满足上述 4 条(编译期 marker,防"顺手接入")
    const CAPTURE_SAFE: bool;
}

/// GEMM 后端契约:marlin-ffi W4A16/W4A8 在 M1 接线为首个实现。
/// 签名占位 —— 参数形态待与 marlin-ffi 的 host emitter 对齐后冻结。
pub trait QuantGemm {
    fn w4a16(&self, arch: Arch) -> bool;
    fn w4a8(&self, arch: Arch) -> bool;
    /// marlin_moe_wna16(fused MoE,REQ-DEC-03)
    fn moe_wna16(&self, arch: Arch) -> bool;
}

/// arch 分发表:目前 sm86 全绿(借力件全部以 sm86 为目标构建)。
pub fn dispatch(arch: Arch) -> &'static str {
    match arch {
        Arch::Sm86 | Arch::Sm89 => "marlin-ffi+attention-rs(快照)",
    }
}

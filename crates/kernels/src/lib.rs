//! 算子分发 —— REQ-HW-01(arch 分发表)与公理 A1.5(可捕获 kernel 契约)
//! 的落点。
//!
//! 借力原则(REQ-DESIGN):GEMM = marlin-ffi(xinfer P1-P3 产物,活跃所有权)
//! 直接接线;attention = attention-rs 快照;这里只定义**契约与分发表**,
//! 不实现 kernel(REQ-DESIGN-03:自研是最后手段)。
//!
//! `.cu` 源常量的住址:已建表算子住 `ops/<域>.rs`(`KernelSpec::SOURCE`),
//! 未建表「仅引入」算子住 `list.rs` §0 —— 集中模块 `sources.rs` 已退役。

/// 契约面(类型权威:Dtype/线格式/Bytes/LaunchMsg/OpError/OpId;
/// 2026-10-12 M2 自 owl-iface::contract 迁入 —— iface 依赖 kernels,
/// kernels 不依赖 iface,契约老家即此;iface 经 re-export 续用旧路径)
pub mod contract;

/// 诊断输出口(唯一):Rust 侧 `klog_*!` 宏 + C/C++ 侧 `cu/owl_log.h`
/// 共用同一条 `LogSink`(boot 期由后端经 [`server::DiagTune::sink`] 注入)。
/// 编译门 = cargo feature `log`(缺省开;不开 = 两侧逐字节零成本)。
pub mod sink;

/// 登记表(list:native kernel 名字 → 源 + 槽序契约 + dtype 标注;
/// 2026-10-12 自 models::kernel 迁入,同日由 native.rs 更名 list.rs ——
/// "登记表"才是它的本体,native/foreign 之别交给 driver/registry)
pub mod kernel;
pub mod bootcheck;
pub mod cu_sig;

/// 生产算子元数据表(P3;`impl KernelSpec` 声明,按域分模块)
pub mod ops;

/// tag → 算子倒排索引(2026-10-20;运行时显式构造,无全局)
pub mod tag_index;

/// 配置派生的选核 tag 过滤器(互斥 facet;无显式 allow_tags)
pub mod tag_filter;

/// 算子准入契约(单算子声明式元数据:参数表/形状族/池访问/数值律 +
/// 编译期擦除 KernelMeta;2026-10-13 P3,**零 feature 门** —— models 经
/// driver 的 KernelPick 消费同一份元数据,不需要 device 面)
pub mod spec;

/// 家族底座(资产/FFI/ABI:两平面共用的单源 —— cubins/OwlCublas/marlin
/// FFI/flashinfer FFI;cublas/marlin/flashinfer 随各自 FFI feature 门)
pub mod family;

/// ── 算子两平面(按家族拆分;P0 名字无关的两个住址)──
/// 客户端平面(类型化 Call builder + 线格式单源;models 视角,
/// 零 cudarc 依赖 —— 随 lib 全 feature 可用)
pub mod client;
/// 服务端平面(家族 runtime:KernelFamily 实现;feature=device ——
/// KernelFamily 签名引用 device::Exec)
#[cfg(feature = "device")]
pub mod server;

/// 服务端面原语(DeviceRes 资源面 trait + Exec 执行引擎;feature=device,
/// 纯 cudarc driver/nvrtc;旧 "cuda" feature(cuda_ops PTX 预编链)已随
/// 三形态归一退役删除,勿复活)
#[cfg(feature = "device")]
pub mod device;

/// 算子注册表(OpId 唯一住址 + KernelFamily 面 + OpRegistry;
/// 同属服务端面 —— KernelFamily 签名引用 device::Exec,随 device 门)
#[cfg(feature = "device")]
pub mod registry;

/// **代际(`Arch`)表** —— 构建期(`build.rs` 同源引入)与运行期共用。
pub mod arch;
/// **算子族 × 代际的准入表**(「这族不该在这张卡上跑」的唯一出口)。
pub mod admit;

pub use arch::{Arch, Caps, Family};
pub use admit::{Admit, AdmitCtx, OpFamily};

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

/// arch 分发表(消费级四代 + 表外兜底):
///
/// - `sm_86 / sm_89 / sm_120`:Marlin-ffi(W4A16/W4A8/MoE)+ attention-rs 快照;
/// - `sm_75`(RTX 20 系):没有 `cp.async`/marlin 所需的 Ampere 原语 ⇒ 只承诺最基础的臂;
/// - `Other`(表外代际,如数据中心卡):按 `caps()` 的硬件事实回答。
///
/// **无通配臂**:新增 `Arch` 变体时编译器会在**这里**逼作者表态
/// (信条:响亮拒绝,不静默继承旧答案)。
pub fn dispatch(arch: Arch) -> &'static str {
    match arch {
        Arch::Sm75 => "attention-rs(快照;无 marlin)",
        Arch::Sm86 | Arch::Sm89 | Arch::Sm120 => "marlin-ffi+attention-rs(快照)",
        Arch::Other { .. } => {
            if arch.caps().cp_async {
                "marlin-ffi+attention-rs(快照)"
            } else {
                "attention-rs(快照;无 marlin)"
            }
        }
    }
}

pub mod driver;


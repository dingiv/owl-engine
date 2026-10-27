//! 代际(`Arch`)—— **构建期与运行期共用同一张表**。
//!
//! 为什么要把"代际"单独抽出来(2026-10-27,RTX 50 / sm_120 移植驱动):
//!
//! 1. **一个事实,三个消费者**:`build.rs` 要按代际出设备码(`nvcc -gencode` /
//!    Marlin `-arch` / FlashInfer `-arch` / Triton AOT target),`owl-cuda` 要在
//!    boot 期把驱动查到的 CC 变成代际并做架构门,`owl-kernels` 的选核面
//!    (`dispatch` / `QuantGemm`)要按代际回答"这条臂在不在"。三处各写一份
//!    字符串/枚举 = 必然会漂(历史上就是:FI 写死 `sm_86`、Marlin 缺省 `86`、
//!    Triton target 写死 `GPUTarget("cuda", 86, 32)`)。
//! 2. **本文件被 `build.rs` 以 `#[path = "src/arch.rs"] mod arch;` 直接引入**
//!    ⇒ 它就是那张唯一表。代价:**本文件不许引用 crate 内其它模块**
//!    (build.rs 是独立 crate,看不到 `crate::...`)——只允许 `core`/`std`。
//! 3. 收录口径 = **与选核/发射有关的事实**(ISA 能力 + 家族),**不收录性能数字**
//!    (SM 数 / 显存 / 频率一律运行期问驱动,不在这里猜)。
//!
//! 新增代际的操作面只有两个:
//! - 在 [`Arch`] 加变体(编译器会在 `marlin` 能力面与 `dispatch` 的**无通配 match**
//!   上逼着作者明确表态 —— 这条纪律见 `family/marlin/mod.rs`);
//! - 在 [`Arch::caps`] 写它的 ISA 事实(或让 `Other` 的 CC 推导兜底)。

use core::fmt;

/// 计算能力代际(`sm_XX`)。**本仓的选核键**。
///
/// **只收消费级四代**(用户裁决 2026-10-27):RTX 20 / 30 / 40 / 50 系。
/// 数据中心卡(A100 / H100 / B200…)不进具名表 —— 驱动给出表外 CC 时落
/// [`Arch::Other`],能力仍按 CC 推导([`Arch::caps`]),**如实记录不冒充邻近代际**
/// (冒充的后果是选错 kernel 臂)。要正式收录某代 = 加变体 + 加 `arch-smXX` feature。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Arch {
    /// Turing(2018):**RTX 20 系**。有 INT8 张量核,无 `cp.async`
    Sm75,
    /// Ampere(2020):**RTX 30 系**(本仓最早的生产档)
    Sm86,
    /// Ada(2022):**RTX 40 系**。首个有 FP8 张量核的消费代际
    Sm89,
    /// Blackwell(2025):**RTX 50 系**(sm_120)
    Sm120,
    /// 驱动给到、本仓未收录的代际(数据中心卡 / 未上市代际)⇒ 如实记录
    Other {
        major: u8,
        minor: u8,
    },
}

/// 世代标签:只用于日志/文档/告警归类,**不参与能力判定**
/// (能力一律走 [`Arch::caps`],那是 ISA 事实)。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    Turing,
    Ampere,
    Ada,
    Hopper,
    Blackwell,
    Unknown,
}

impl fmt::Display for Family {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Turing => "Turing",
            Self::Ampere => "Ampere",
            Self::Ada => "Ada",
            Self::Hopper => "Hopper",
            Self::Blackwell => "Blackwell",
            Self::Unknown => "Unknown",
        };
        f.write_str(s)
    }
}

/// 与**选核/发射**有关的 ISA 事实。
///
/// ⚠️ 这是"硬件能做什么",**不是**"本仓已验证过什么"。验证状态属于各 family
/// 自己的能力面(如 `family::marlin` 的 `QuantGemm` 实现)与装配期架构门。
///
/// 阈值来源 = compute capability 与 PTX ISA 的引入版本(文档级事实):
/// - `dp4a` sm_61+;`int8_mma`(IMMA `s8s8s32`)sm_75+
/// - `tf32` sm_80+;`cp_async` sm_80+
/// - `fp8_mma`(e4m3/e5m2)sm_89+
/// - `fp4_mma`(`kind::f8f6f4`,e2m1 操作数)**仅 `sm_120a`**
///   (实测:`-arch=sm_120` 直接拒 `Feature '.kind::f8f6f4' not supported`)
/// - `fp4_blockscale`(`kind::mxf4nvf4.block_scale`,e4m3 每 16)同样**仅 `a` 档**;
///   实测 m16n8k64 是 m16n8k32 无缩放版的 **3.7×**(713 vs 192 TFLOPS,5070 Ti)
/// - `tma`(`cp.async.bulk.tensor`)/ `clusters` sm_90+
/// - `wgmma` 仅 `sm_90a`;`tcgen05` 仅 `sm_100a`/`sm_103a`(消费 Blackwell 没有)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Caps {
    /// 点积指令 `dp4a`(int8 快速路)
    pub dp4a: bool,
    /// INT8 张量核(`mma.sync.aligned.*.s8.s8.s32`)
    pub int8_mma: bool,
    /// TF32 张量核(`mma ... .tf32`)
    pub tf32: bool,
    /// `cp.async`(异步拷入 smem,Ampere 起的流水线基座)
    pub cp_async: bool,
    /// FP8 张量核(e4m3 / e5m2 输入,FP16/FP32 累加)
    pub fp8_mma: bool,
    /// FP4 张量核(`kind::f8f6f4`,e2m1 操作数;**需 `a` 编译目标**)
    pub fp4_mma: bool,
    /// 块缩放 FP4(`mxf4nvf4.block_scale`,e4m3 每 16 = NVFP4;**需 `a` 目标**)
    pub fp4_blockscale: bool,
    /// TMA(`cp.async.bulk*`;张量维度搬运)
    pub tma: bool,
    /// 线程块 cluster(`cluster.sync` / DSMEM)
    pub clusters: bool,
    /// `wgmma`(异步 warpgroup MMA;**只在 `sm_90a` 目标下可发**)
    pub wgmma: bool,
    /// `tcgen05`(第五代张量核指令;**只在 `sm_100a` 系目标下可发**)
    pub tcgen05: bool,
}

impl Caps {
    /// 一行摘要(日志/告警用;不参与判定)
    pub fn summary(self) -> &'static str {
        // 组合只有少数几种,给常见代际直接命中,避免日志里一串 bool
        if self.tma && self.clusters {
            "cp.async+TMA+cluster"
        } else if self.cp_async && self.fp8_mma {
            "cp.async+fp8"
        } else if self.cp_async {
            "cp.async"
        } else if self.int8_mma {
            "int8-tc"
        } else {
            "baseline"
        }
    }
}

impl Arch {
    /// 本仓缺省代际(缺省构建目标;**不是**"设备真值" —— 真值一律探测)
    pub const DEFAULT: Arch = Arch::Sm86;

    /// 全部具名代际(错误信息 / 构建脚本的"合法取值"清单)
    pub const ALL: &'static [Arch] = &[Arch::Sm75, Arch::Sm86, Arch::Sm89, Arch::Sm120];

    /// `(major, minor)` —— 驱动 `COMPUTE_CAPABILITY_*` 的同一口径
    pub const fn cc(self) -> (u8, u8) {
        match self {
            Arch::Sm75 => (7, 5),
            Arch::Sm86 => (8, 6),
            Arch::Sm89 => (8, 9),
            Arch::Sm120 => (12, 0),
            Arch::Other { major, minor } => (major, minor),
        }
    }

    /// 驱动 CC → 代际。表外 CC **如实**落 `Other`(不四舍五入到邻近代际)
    pub const fn from_cc(major: u8, minor: u8) -> Arch {
        match (major, minor) {
            (7, 5) => Arch::Sm75,
            (8, 6) => Arch::Sm86,
            (8, 9) => Arch::Sm89,
            (12, 0) => Arch::Sm120,
            _ => Arch::Other { major, minor },
        }
    }

    /// `sm_120` 形(`nvcc -arch=` / 装配期架构门 / 日志用)
    pub fn as_sm(self) -> String {
        let (major, minor) = self.cc();
        format!("sm_{major}{minor}")
    }

    /// `120` 形(nvcc `-gencode arch=compute_<n>`、`-code=sm_<n>`、Triton target 用)
    pub fn nvcc_num(self) -> String {
        let (major, minor) = self.cc();
        format!("{major}{minor}")
    }

    /// 世代标签(归类用,非能力判定)
    pub fn family(self) -> Family {
        match self {
            Arch::Sm75 => Family::Turing,
            Arch::Sm86 => Family::Ampere,
            Arch::Sm89 => Family::Ada,
            Arch::Sm120 => Family::Blackwell,
            // 表外(数据中心卡等):按 major 归类能力,**但不当成消费品类**
            Arch::Other { major, .. } => match major {
                7 => Family::Turing,
                8 => Family::Ampere,
                9 => Family::Hopper,
                10 | 12 => Family::Blackwell,
                _ => Family::Unknown,
            },
        }
    }

    /// ISA 能力(阈值推导;详见 [`Caps`] 头注)
    pub fn caps(self) -> Caps {
        // 统一按 CC 阈值推导:`Other` 因此自动获得合理答案(表外代际不等于
        // 没有能力),具名代际的 cc() 是常量 ⇒ 结果可被编译器折叠。
        let (major, minor) = self.cc();
        let at_least = |m: u8, n: u8| (major, minor) >= (m, n);
        // FP4 只到 `a` 档:消费级里只有 sm_120(Blackwell)。判断写死成 CC 相等
        // 而不是 `>=` —— 下一代消费卡的 FP4 能力得重新拿实测确认,不能继承。
        let fp4 = (major, minor) == (12, 0) || (major, minor) == (12, 1);
        Caps {
            dp4a: at_least(6, 1),
            int8_mma: at_least(7, 5),
            tf32: at_least(8, 0),
            cp_async: at_least(8, 0),
            fp8_mma: at_least(8, 9),
            fp4_mma: fp4,
            fp4_blockscale: fp4,
            tma: at_least(9, 0),
            clusters: at_least(9, 0),
            // 这两个是 "a" 档专有:`Arch` 只表达计算能力(不带 a 后缀),
            // 因此恒 false —— 需要它们的臂必须另立编译目标,不许在这里放假话。
            wgmma: false,
            tcgen05: false,
        }
    }

    /// **arch-specific('a')编译目标** —— 需要 `a` 才算能发的指令集
    /// (FP4 / wgmma / tcgen05)用它的 cubin。
    ///
    /// 返回 `Some("sm_120a")` 的判据 = 本代有 `a` 变体;`None` = 只能编非-a。
    /// 用法:`build.rs` 对**声明了 arch-specific 的源**走
    /// `-gencode arch=compute_<n>a,code=sm_<n>a`(只写 `-arch=sm_120a` 不够:
    /// nvcc 会把 PTX 目标退成 `compute_120`,ptxas 照样拒)。
    pub fn as_sm_a(self) -> Option<String> {
        match self.cc() {
            (9, 0) | (10, 0) | (12, 0) | (12, 1) => Some(format!("sm_{}a", self.nvcc_num())),
            _ => None,
        }
    }

    /// 解析人给的代际字面量(build 旋钮 / 配置 / CLI 共用)。
    ///
    /// 接受:`sm_120` `sm120` `compute_120` `120` `12.0`(大小写不敏感)。
    /// 拒绝空串/非数字;表外数值**接受**为 `Other`(构建期可以为一个还没收录
    /// 的代际出设备码 —— 这是"接口做宽"的关键,不是疏漏)。
    pub fn parse(s: &str) -> Result<Arch, ArchParseError> {
        let raw = s.trim();
        if raw.is_empty() {
            return Err(ArchParseError { input: s.to_string() });
        }
        let low = raw.to_ascii_lowercase();
        let body = low
            .strip_prefix("compute_")
            .or_else(|| low.strip_prefix("compute"))
            .or_else(|| low.strip_prefix("sm_"))
            .or_else(|| low.strip_prefix("sm"))
            .unwrap_or(&low);
        // `sm_120a` / `compute_120a`:`a` 是 "arch-specific 指令集" 后缀,
        // **不改变计算能力** —— 本表只表达 CC,所以丢掉后缀照收(把 a 档当成
        // 独立代际会让 `Arch` 变成编译目标,那是另一件事)。
        let body = body.strip_suffix('a').unwrap_or(body);
        // `12.0` 形(smi 口径)
        let (maj, min) = if let Some((a, b)) = body.split_once('.') {
            (a.parse::<u8>(), b.parse::<u8>())
        } else {
            let digits: String = body.chars().filter(|c| c.is_ascii_digit()).collect();
            if digits.len() != body.len() {
                return Err(ArchParseError { input: s.to_string() });
            }
            match digits.len() {
                1 => (digits.parse::<u8>(), Ok(0)),
                2 => (
                    digits[0..1].parse::<u8>(),
                    digits[1..2].parse::<u8>(),
                ),
                3 => (
                    digits[0..2].parse::<u8>(),
                    digits[2..3].parse::<u8>(),
                ),
                _ => (
                    digits[0..2].parse::<u8>(),
                    digits[2..4].parse::<u8>(),
                ),
            }
        };
        match (maj, min) {
            (Ok(major), Ok(minor)) => Ok(Arch::from_cc(major, minor)),
            _ => Err(ArchParseError { input: s.to_string() }),
        }
    }
}

impl fmt::Display for Arch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (major, minor) = self.cc();
        write!(f, "sm_{major}{minor}")
    }
}

/// 代际字面量非法(错误信息里给出**合法取值清单** —— 拒启必须响亮且可操作)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchParseError {
    pub input: String,
}

impl fmt::Display for ArchParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let known: Vec<String> = Arch::ALL.iter().map(|a| a.as_sm()).collect();
        write!(
            f,
            "无法解析代际 {:?};接受 `sm_120` / `compute_120` / `120` / `12.0` 这几种写法。\
             本仓具名代际:[{}](表外数值也接受,按 `Other` 处理)",
            self.input,
            known.join(", ")
        )
    }
}

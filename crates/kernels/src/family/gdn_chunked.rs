//! GDN chunked delta rule(foreign-kernel 通道;2026-10-03 立案执行)。
//!
//! 溯源:kernels = FLA 0.6.0(fla-org/flash-linear-attention @ 9f38d24)的
//! Triton kernel **AOT cubin**(triton 3.7.1,sm86;goldens 同源:
//! testdata/gdn_fla/cases/ 五级金标)。owl 侧只做编排与 ABI 适配 ——
//! 「先 copy 后超越」完全体:kernel 数值 = FLA 本尊,零 port 数值风险。
//!
//! **五核流水**(单 Call;server 编排,grid/block 内嵌常量单源):
//! 1. cumsum:g [T,HV] f32 → g_cum(log2 域局部 cumsum;RCP_LN2 在核内 scale)
//! 2. kkt:k/g/beta → A [T,HV,64](下三角 WY 系数,fused solve)
//! 3. wu:k/v/beta/A/g → w,u [T,HV,KD]
//! 4. h:k/v/w/u/g + h0 → h [NT,HV,KD,VD], v_new, ht(终态)
//! 5. o:q/k/v_new/h/g → o [T,HV,VD] f32(scale = kd^-0.5)
//!
//! **六核流水**(fork 六核;kkt/solve 分立):
//! 1. cumsum:g [T,HV] f32 → g_cum(**nats 域**局部累计;RCP_LN2 烘核内不施放)
//! 2. kkt:k/beta/**g_cum** → A [T,HV,64] f32(下三角 WY 系数)
//!    ⚠️ **g 语义律**(2026-10-11 终案):kkt/h/o 的 "g" 参数全部吃 **cumsum
//!    产物**;喂 raw g = exp(±24) 爆炸(案录:perf-roadmap 补录Ⅷ/Ⅸ,坑册 §17)
//! 3. merge:solve_tril(BT=64)A → Ai(**bf16 字节**;生产链 output=k.dtype)
//! 4. wu:k/v/beta/Ai/**g_cum** → w,u [T,HV,KD] bf16
//! 5. h:k/u/w/**g_cum** + h0 → h [NT,HV,VD,KD](**V 主**), v_new, ht(f32)
//!    实参序 = SASS 审计(gk 被 triton 剪参);grid 二维 (cdiv(VD,BV), HV)
//! 6. o:q/k/v_new/h/**g_cum** → o [T,HV,VD] bf16(scale = kd^-0.5)
//!
//! **ABI**(triton 3.7 cubin;参数序 = tt.func 声明序;global_scratch=0 无尾参;
//! NV=48/KD=VD=128/BT=64 已烘 constexpr;T 运行时标量;dtype 配方 =
//! q/k/v/beta bf16 + g/g_cum/A f32 + Ai/w/u/h/v_new/o bf16 + state f32):
//! ```text
//! cumsum (s,o:*f32, cu:i64*, idx:i64*, T:i32)                grid (NT, HV)
//! kkt    (k,beta,g_cum,A:*f32, cu, idx, T)                    grid (NT, HV)
//! merge  (A,Ai:*f32, cu, idx, T)                              grid (NT, HV)
//! wu     (k,v,beta,w,u,Ai,g_cum:*f32, cu, idx, T)             grid (NT, HV)
//! h      (k,v,w,v_new,g_cum,h,ht?:*f32, h0?:*f32, cu, coff:i64*, T) grid (cdiv(VD,BV), HV)
//! o      (q,k,v_new,h,g_cum,o:*f32, cu, idx, scale:f32, T:i32)    grid (BV组, NT, HV)
//! ```
//! h 臂取 **h0 变体**(shared 49412,warps 4)—— 引擎恒带初态(首 turn 传
//! 零块);ht(终态)= 写回 state 槽。h0/ht/h_buf 布局 = [V,K](与池
//! [K,V] 互转由 handler 转置核承担);金标对拍见 tests/gdn_chunked_golden.rs
//! (c1-c4 pip-fla [K,V] 需转置,c5 fork [V,K] 直比——布局代际律)。
//!
//! **发射通道**:server foreign-kernel 单臂([`GDN_CHUNKED_FWD`]);handler
//! 持有 cubin 模块句柄 + 持久 scratch(按 max_chunk 预分配),逐 chunk 复用。
//! 中间张量(g_cum/A/w/u/h/v_new/chunk 表)全为 handler 私有 scratch,
//! 账外不回流 SSA(与 FI 工作空间同纪律)。

/// foreign-kernel 虚拟核名(server 分派臂;字面量住址 = contract::names)
pub const GDN_CHUNKED_FWD: &str = crate::contract::names::GDN_CHUNKED;

/// 槽序契约(LaunchMsg.args):
/// `[T q, T k, T v, T g, T beta, T state(in/out f32 槽块), O out, T o_f32(scratch 出),
///   sz T, sz slot, sz hv, sz nk, sz kd, sz scale_bits, sz nt]`
/// = 7 Block + 1 O + 1 Block(o_f32:f32 中间,cast 由 handler 内置核转 out)+ 7 sz。
/// state = 槽寻址块 [slots, HV, KD, VD](slot 标量寻址;in/out 同块)。
pub const GDN_CHUNKED_SLOTS: &str =
    "[T q, T k, T v, T g, T beta, T state, O out, T o_f32, \
     sz T, sz slot, sz hv, sz nk, sz kd, sz scale_bits, sz nt]";

/// foreign 分派谓词
pub fn is_foreign(name: &str) -> bool {
    name == GDN_CHUNKED_FWD
}

/// 设备码资产(**构建期 AOT 自编**,2026-10-13 C4b-β)。
///
/// 源 = 树内 vendor 的 FLA(`triton/fla/`)+ 配置表 `triton/gdn_chunked_aot.json`;
/// 编译 = `triton/gdn_aot_compile.py`(build.rs 默认跑,`GDN_TRITON_BUILD=0` 可关)。
/// **不留入库采集件**:产物是构建产物,与 native cubin / marlin / FI 同形态。
///
/// 取用面返回 `Option`:缺 = 装配期**结构化拒启**(不 panic;见
/// `server/gdn_chunked.rs::assemble`)。构建期缺失只可能来自
/// `GDN_TRITON_BUILD=0`(显式关)或脚本失败。
pub mod cubins {
    #[cfg(feature = "device")]
    fn asset(name: &'static str) -> Option<&'static [u8]> {
        crate::family::cubins::gdn_asset(name)
    }
    #[cfg(not(feature = "device"))]
    fn asset(_name: &'static str) -> Option<&'static [u8]> {
        None
    }

    pub fn cumsum() -> Option<&'static [u8]> {
        asset("cumsum")
    }
    pub fn kkt() -> Option<&'static [u8]> {
        asset("kkt")
    }
    pub fn wu() -> Option<&'static [u8]> {
        asset("wu")
    }
    pub fn h() -> Option<&'static [u8]> {
        asset("h")
    }
    pub fn o() -> Option<&'static [u8]> {
        asset("o")
    }
    /// solve_tril(BT=64)单核形态:merge_16x16_to_64x64_inverse
    pub fn merge() -> Option<&'static [u8]> {
        asset("merge")
    }
    /// 六核清单(装配期报告/缺件报错用)
    pub const NAMES: &[&str] = &["cumsum", "kkt", "merge", "wu", "h", "o"];

    /// 发射常量(2026-10-11 fork-bf16 重采集:torch profiler 实测指纹,
    /// 采集于 T=1024/NK16/NV48/KD=VD=128/g-f32 配方;变体 = autotune 选中)
    pub mod launch {
        pub const CUMSUM_SHARED: u32 = 8;
        pub const CUMSUM_WARPS: u32 = 4;
        pub const KKT_SHARED: u32 = 24576;
        pub const KKT_WARPS: u32 = 8; // block 256!
        pub const MERGE_SHARED: u32 = 10240;
        pub const MERGE_WARPS: u32 = 2;
        pub const WU_SHARED: u32 = 32768;
        pub const WU_WARPS: u32 = 4;
        pub const H_SHARED: u32 = 49412;
        pub const H_WARPS: u32 = 4;
        /// h/o 核 grid 轴 0 的 BV(选中变体:grid (cdiv(V,64), ...) = 2)
        pub const H_BV: u32 = 64;
        pub const O_BV: u32 = 64;
        pub const O_SHARED: u32 = 24576;
        pub const O_WARPS: u32 = 4;
        // (pip-fla f32 时代的对照臂常量与资产已于 2026-10-13 退役删除)
    }
}

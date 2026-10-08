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
//! **ABI**(triton 3.7 cubin;参数序 = tt.func 声明序;global_scratch=0 无尾参;
//! NV=48/KD=VD=128/BT=64 已烘 constexpr;T 运行时标量):
//! ```text
//! cumsum (s,o:*f32, scale:f32, cu:i64*, idx:i64*, T:i32)      grid (NT, HV)
//! kkt    (k,g,beta,A:*f32, cu, idx, T)                         grid (NT, HV)
//! wu     (k,v,beta,w,u,A,g:*f32, cu, idx, T)                   grid (NT, HV)
//! h      (k,v,w,v_new,g,h,ht?:*f32, h0?:*f32, cu, coff:i64*, T) grid (cdiv(VD,BV)*HV,)
//! o      (q,k,v,h,g,o:*f32, cu, idx, scale:f32, T:i32)         grid (BV组, NT, HV)
//! ```
//! h 臂取 **h0 变体**(shared 24576,warps 4)—— 引擎恒带初态(首 turn 传
//! 零块);ht(终态)= 写回 state 槽。
//!
//! **发射通道**:server foreign-kernel 单臂([`GDN_CHUNKED_FWD`]);handler
//! 持有 cubin 模块句柄 + 持久 scratch(按 max_chunk 预分配),逐 chunk 复用。
//! 中间张量(g_cum/A/w/u/h/v_new/chunk 表)全为 handler 私有 scratch,
//! 账外不回流 SSA(与 FI 工作空间同纪律)。

/// foreign-kernel 虚拟核名(server `is_foreign` 分派臂)
pub const GDN_CHUNKED_FWD: &str = "gdn_chunked_delta_rule_fwd";

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

/// AOT 资产(cubin 内嵌;sm86 / triton 3.7.1 产出)
pub mod cubins {
    pub const CUMSUM: &[u8] =
        include_bytes!("../assets/gdn_chunked/cumsum.cubin");
    pub const KKT: &[u8] =
        include_bytes!("../assets/gdn_chunked/kkt.cubin");
    pub const WU: &[u8] =
        include_bytes!("../assets/gdn_chunked/wu.cubin");
    pub const H: &[u8] =
        include_bytes!("../assets/gdn_chunked/h.cubin");
    /// 无初态变体(参数表无 h0;fresh 序列;shared 98564 / w4 / BV=64)
    pub const H_NOH0: &[u8] =
        include_bytes!("../assets/gdn_chunked/h_noh0.cubin");
    pub const O: &[u8] =
        include_bytes!("../assets/gdn_chunked/o.cubin");
    /// solve_tril(BT=64)单核形态:merge_16x16_to_64x64_inverse
    pub const MERGE: &[u8] =
        include_bytes!("../assets/gdn_chunked/merge.cubin");

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
        // 旧 pip-fla f32 时代常量已随资产备份(assets/gdn_chunked_pipfla_f32)
    }
}

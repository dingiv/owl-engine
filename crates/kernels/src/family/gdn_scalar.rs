//! GDN 标量门 chunked delta-rule 前向(lmdeploy pre_sm90 port;2026-10-03)。
//!
//! 溯源与动机:FLA AOT 五核路线(见 [`crate::gdn_chunked`])在 fwd_h 核上
//! 踩到装载层非法访问;调研定案全社区 sm86 唯一完整 CUDA 源码实现 =
//! lmdeploy turbomind `pre_sm90/chunked.cu`(Qwen3.5/3.6 turbomind 在
//! Ampere 上的现役路径)。本模块 = 该内核的 owl 移植(单核替代五核流水,
//! 数值 fp32 全程;金标 `testdata/gdn_fla` 直接对拍)。
//!
//! **单核流水**(g = log-e 负域标量门,expf 直接吃;状态 in/out 原地):
//! `o_t = scale·q_t·S_t`,S 逐 token 步进 `S ← exp(g)S + k⊗(β(v − k·S))`,
//! 终态 ht 原地写回 state 槽。grid (ns, HV) × block 256,每 block 独占
//! 一条序列一个 v-head 的 [KD×VD] 状态片。
//!
//! **发射通道**:server foreign-kernel 单臂([`GDN_SCALAR_FWD`];与 FLA
//! 五核臂 [`crate::gdn_chunked`] 并存为对照臂,env 选路)。
//! 中间张量零个 —— 状态即全部,账外 scratch 仅 o_f32(引擎侧 cast 用)。

/// foreign-kernel 虚拟核名(server 分派臂;字面量住址 = contract::names)
pub const GDN_SCALAR_FWD: &str = crate::contract::names::GDN_SCALAR;

/// foreign 分派谓词
pub fn is_foreign(name: &str) -> bool {
    name == GDN_SCALAR_FWD
}

/// 槽序契约(LaunchMsg.args;与 gdn_chunked 同构):
/// `[T q, T k, T v, T g, T beta, T state, O out,
///   sz T, sz slot, sz ns, sz hv, sz nk, sz kd, sz scale_bits]`
/// = 7 Block + 7 sz(q/k/v/g/beta 为层侧 CAST_F16_F32 产物,f32 块直入;
/// handler 直接在块指针上发射,零中间拷贝)。
/// - state = 槽寻址块 [slots, HV, KD, VD](slot 标量寻址;in/out 同块);
/// - ns = 本调用序列数(varlen 就位;单序列 prefills = 1,seq_off 由
///   handler 算入 scratch [0,T]);
/// - o_f32 为 handler 私有 scratch(不入契约,同 gdn_chunked 纪律)。
pub const GDN_SCALAR_SLOTS: &str = "[T q, T k, T v, T g, T beta, T state, O out, \
     sz T, sz slot, sz ns, sz hv, sz nk, sz kd, sz scale_bits]";

/// AOT 资产(cubin 入库;gdn_chunked 同款先例 —— 预编产物进 assets,日常
/// 构建零 nvcc;源改动重编一条命令,见 assets/gdn_scalar/build.sh)
pub mod cubin {
    /// owl_gdn_chunk_scalar_f32(f32 金标通道;f16/bf16 引擎通道接线时加)
    pub const CHUNK_SCALAR_F32: &[u8] = include_bytes!("../../assets/gdn_scalar/chunk_scalar_f32.cubin");

    /// 内核符号名(extern "C",名稳)
    pub const KERNEL_F32: &str = "owl_gdn_chunk_scalar_f32";

    /// 发射常量(与 .cu 内 kBlockDim/kChunkSize/d=128 一致;单源律:改两处)
    pub const BLOCK: u32 = 256;
    pub const CHUNK: usize = 16;
    /// smem = 3·16·(128+4)·4 + 2·16·4 = 25472 B(< 48KB,免 setattr)
    pub const SMEM_D128: u32 = 25472;
}

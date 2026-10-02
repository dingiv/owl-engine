//! FlashInfer prefill(foreign-kernel 通道;E1.5 工作包,2026-10-03 落地)。
//!
//! 溯源:adapter 预编 .a(预编源 `cu/flashinfer/owl_fi_attention.cu`,
//! ported from repos/attention.rs flashinfer_adapter_prefill.cu 非 SM90/FA2
//! 分支;guoqingbao/flashinfer fork @ 0f06c230,Apache-2.0)。.a 入库
//! (`cu/flashinfer/prebuilt/libowl_flashinfer.a`,marlin .a 先例;
//! FLASHINFER_FORCE_BUILD=1 重编,nvcc + fork include 面)。
//!
//! **为什么是 FlashInfer**(REQ-PRE-01 / REQ-DESIGN:禁自研核进热路径):
//! 旧 chunked_prefill_paged_attn_opt(vendor port)~6 TFLOPS(标量,1
//! thread/query 串行扫);自研 split(flash-decoding)同分块 O(N²) 形态,
//! 8k+ 才 +5.3%。FA2 级 tensor-core kernel 是社区标准答案,E2 分页底座
//! 落地后前置解除(2026-09-27 定盘)。
//!
//! **布局契约**(paged_kv_t QKVLayout::kNHD,页粒度;page.cuh
//! get_elem_offset = page·stride + h·hd + entry·(Hkv·hd) + d):
//! - k_fi / v_fi(K/V 影子池):`[num_blocks, page_size, Hkv, head_dim]`
//!   f16 —— 由 `owl_reshape_and_cache_dual_f16`(K0-dual)与 classic K/V
//!   池同发射写出(classic K = x-interleave、V = dim-major,均不匹配)
//! - 页表:`indices`(物理页 id 拼接)/ `indptr`([0, nb])/ `last_len`
//!   (末页有效数 = ctx − (nb−1)·page;整页 = page,非 %)—— i32 设备表,
//!   engine 每 chunk 构造(ForwardCtx.fi)
//!
//! **plan/run 分离**:plan = host 侧调度规划(PrefillPlan;split-kv 决策 +
//! tile 选择),每 chunk 一次,server 侧 1 项缓存(全层同参);run = 每
//! attention 层一次(BatchPrefillWithPagedKVCacheDispatched,causal,
//! chunked 语义 = q 对齐 kv 尾部,与 K0 先行的池读序配套)。
//!
//! **发射通道**:server foreign-kernel(虚拟核名 [`PREFILL_FI`];marlin/
//! cublas 同款,Launch 即唯一执行命令)。批 = 1(owl 单会话)。

use std::ffi::c_void;

/// foreign-kernel 虚拟核名(server `is_foreign` 分派臂;f16 KV)
pub const PREFILL_FI: &str = "flashinfer_prefill_paged_f16";
/// fp8 KV 变体(e4m3 KV + half Q/Out;槽序同 PREFILL_FI)
pub const PREFILL_FI_FP8KV: &str = "flashinfer_prefill_paged_fp8kv";

/// 槽序契约(LaunchMsg.args):
/// `[T q, T k_fi, T v_fi, T q_cu_seqlens, T indices, T indptr, T last_len,
///   T wr(树序依赖边,K0-dual 先于读池;FI 不解引用),
///   O out, sz total_rows, sz ctx_total, sz T, sz hq, sz hkv, sz hd,
///   sz page, sz sm_scale_bits, sz nb]`
/// = 8 Block + 1 O(输出 = 槽序契约 4 变体,O 槽显式位)+ 8 sz;
/// sm_scale 以 `f32::to_bits` 过线,server 侧 `f32::from_bits` 还原。
pub const PREFILL_FI_SLOTS: &str =
    "[T q, T kc_fi, T vc, T q_cu, T indices, T indptr, T last_len, T wr, O out, \
     sz total_rows, sz ctx_total, sz T, sz hq, sz hkv, sz hd, sz page, \
     sz sm_scale_bits, sz nb]";

/// FI plan workspace 尺寸(设备 int;server 句柄懒分配)
pub const FI_INT_WS_BYTES: usize = 32 * 1024 * 1024;
/// FI plan workspace 尺寸(设备 float;split-kv tmp_v/tmp_s)
pub const FI_FLOAT_WS_BYTES: usize = 64 * 1024 * 1024;
/// FI page-locked/host 暂冲(H2D staging;PrefillPlan 用)
pub const FI_HOST_STAGING_BYTES: usize = 32 * 1024 * 1024;

/// foreign 分派谓词(server handle_launch 前置检查)
pub fn is_foreign(name: &str) -> bool {
    name == PREFILL_FI || name == PREFILL_FI_FP8KV
}

extern "C" {
    /// host 侧调度规划(见 cu/flashinfer/owl_fi_attention.cu 契约)
    pub fn owl_fi_prefill_plan(
        float_ws: *mut c_void, float_ws_size: usize,
        int_ws: *mut c_void, int_ws_size: usize,
        page_locked: *mut c_void, page_locked_size: usize,
        out_plan15: *mut i64,
        out_cta_tile_q: *mut i32,
        out_split_kv: *mut i32,
        qo_indptr_h: *const i32,
        kv_indptr_h: *const i32,
        total_num_rows: i32,
        batch_size: i32,
        num_qo_heads: i32, num_kv_heads: i32,
        head_dim: i32, page_size: i32,
        stream: *mut c_void,
    ) -> i32;

    /// fp8 KV paged prefill(e4m3 KV + half Q/Out;scale 隐式 1.0)
    pub fn owl_fi_prefill_run_fp8kv(
        q_ptr: *const c_void,
        k_data: *const c_void,
        v_data: *const c_void,
        out_ptr: *mut c_void,
        q_cu_seqlens: *mut i32,
        indices: *mut i32,
        indptr: *mut i32,
        last_len: *mut i32,
        plan15: *const i64,
        int_ws: *mut c_void, int_ws_size: usize,
        float_ws: *mut c_void, float_ws_size: usize,
        batch_size: i32,
        num_qo_heads: i32, num_kv_heads: i32,
        head_dim: i32, page_size: i32,
        total_num_rows: i32,
        sm_scale: f32,
        stream: *mut c_void,
    ) -> i32;

    /// paged FA2 prefill 发射(见 cu/flashinfer/owl_fi_attention.cu 契约)
    pub fn owl_fi_prefill_run(
        q_ptr: *const c_void,
        k_data: *const c_void,
        v_data: *const c_void,
        out_ptr: *mut c_void,
        q_cu_seqlens: *mut i32,
        indices: *mut i32,
        indptr: *mut i32,
        last_len: *mut i32,
        plan15: *const i64,
        int_ws: *mut c_void, int_ws_size: usize,
        float_ws: *mut c_void, float_ws_size: usize,
        batch_size: i32,
        num_qo_heads: i32, num_kv_heads: i32,
        head_dim: i32, page_size: i32,
        total_num_rows: i32,
        sm_scale: f32,
        stream: *mut c_void,
    ) -> i32;
}

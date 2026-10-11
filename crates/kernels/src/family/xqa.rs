//! XQA(kNHD fp8 KV 的 decode/verify 注意力;foreign-kernel 通道)。
//!
//! 溯源:vendor `crates/kernels/cu/xqa/`(移植自
//! repos/vllm-ampere-optimized/flashampere/xqa/csrc,Ampere 闭包;上游 =
//! FlashInfer csrc/xqa)。预编 `.a` 入库(`cu/xqa/prebuilt/libowl_xqa.a`,
//! marlin/FI 先例;`XQA_FORCE_BUILD=1` 重编),feature `xqa`。
//!
//! **为什么用它**(长 ctx verify 立案 D3):现役 v2 伪序列臂 = 8 条伪序列
//! 各扫一遍全 ctx(元素数 ×8);XQA 是 **decode 形状**(KV-split +
//! warp-specialized + GQA 内共享 K/V 读),`SPEC_DEC` 模式 q=1+K 的 q 维扩展
//! 近乎免费。kernels crate 单算子实测(ctx 8192,q=8):**0.495ms vs v2
//! 2.673ms = 5.40×**,数值 max|XQA−host| 4.5e-4(v2 4.6e-4)同级。
//!
//! **布局契约**(与 owl kNHD 主池逐字对齐;XQA 文档名 NHD):
//! - KV:`[num_pages, tokens_per_page, Hkv, head_dim]` **fp8 e4m3**;
//!   strides 以**元素**过线(host 契约)
//! - q/out:`[batch, 1, q_seq_len, Hq, head_dim]` f16(owl verify 的
//!   `[T, hq*hd]` 是它的 free view)
//! - page_table **int32** `[batch, max_pages]`(owl 现有 bt 是 f32 → 需 i32 孪生)
//! - seq_lens **uint32** `[batch]` = **整请求总 KV 长**(不是逐行前缀长!
//!   块内 q 行的因果**只由 mask 表达**;踩坑留档见 cu/xqa/README.md)
//! - mask **uint32** `[batch, q_seq_len, ceil(q/32)]`:行 i = 块内第 i 个
//!   token,bit k = 允许看块内第 k 个 key,因果 = `bits 0..i`(含自身)
//!   ⇒ `mask[i] = (1<<(i+1))-1`
//!
//! **workspace**:semaphores 与 scratch 由 runtime 持久持有(图内安全:
//! 热路径零分配;参考实现 scratch 给 256MiB)。

use std::ffi::c_void;

/// foreign-kernel 虚拟核名(server 分派臂;字面量住址 = contract::names)
pub const VERIFY_XQA: &str = crate::contract::names::VERIFY_XQA;

/// 槽序契约(LaunchMsg.args;6 Block + 10 sz):
/// `[T q, T k_cache, T v_cache, T page_table(i32), T seq_lens(u32),
///   T mask(u32), T semaphores(u32), T scratch, T wr, O out, sz q_seq_len, sz hkv,
///   sz hd, sz page, sz max_seq_len,
///   sz q_scale_bits, sz sm_count, sz stride_page, sz stride_token,
///   sz stride_head]`
/// q_scale 以 `f32::to_bits` 过线(= attn_scale·sqrt(hd);核内再除 sqrt(hd))。
pub const VERIFY_XQA_SLOTS: &str =
    "[T q, T k_cache, T v_cache, T page_table, T seq_lens, T mask, \
     T semaphores, T scratch, T wr, O out, sz q_seq_len, sz hkv, sz hd, sz page, \
     sz max_seq_len, sz q_scale_bits, sz sm_count, sz stride_page, \
     sz stride_token, sz stride_head]";

/// **线槽序权威**(plain type-list;`lower_kernel`/`foreign_sig` 消费,
/// 必须与层侧 `.arg` 链同序:9 入参 T + O 输出 + 10 sz)
pub const VERIFY_XQA_SIG: &str =
    "T,T,T,T,T,T,T,T,T,O,sz,sz,sz,sz,sz,sz,sz,sz,sz,sz";

/// XQA 工作区(参考实现给 256MiB;持久复用)
pub const XQA_SCRATCH_BYTES: usize = 256 * 1024 * 1024;

extern "C" {
    /// 发射(见 cu/xqa/owl_xqa_launch.cu 契约;返回 0 = 成功,非零 = 参数拒绝)
    #[allow(clippy::too_many_arguments)]
    pub fn owl_xqa_launch(
        sm_count: u32,
        nb_k_heads: u32,
        sliding_win_size: u32,
        q_scale: f32,
        output: *mut c_void,
        q: *const c_void,
        k_cache: *mut c_void,
        v_cache: *mut c_void,
        page_table: *const i32,
        max_seq_len: u32,
        seq_lens: *const u32,
        batch_size: u32,
        kv_scale: f32,
        q_seq_len: u32,
        mask: *const u32,
        semaphores: *mut u32,
        scratch: *mut c_void,
        kv_stride_page: u64,
        kv_stride_token: u64,
        kv_stride_head: u64,
        stream: *mut c_void,
    ) -> i32;
}

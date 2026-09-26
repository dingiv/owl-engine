//! Marlin W4A16(f16 激活 × u4 权重 × f16 scales)—— foreign-kernel 通道。
//!
//! 溯源:kernel/发射器 vendor 自 xinfer `crates/kernels/cuda/csrc/marlin`
//! (vLLM 增强版 Marlin 零 torch 解包;上游 IST-DASLab/marlin + Neural Magic,
//! Apache-2.0;SM86 调优,99KB 动态 smem)。移植裁决见
//! roadmap.local/f16-cublas-campaign.md §七 与 porting-workorders.md 工单 M。
//!
//! **布局契约**(upstream `marlin/__init__.py` Layer.pack 语义):
//! - A:row-major f16,`(m, k)`
//! - B:marlin-packed i32,`(k/16, n*16/8)`(tile-perm 打包,`repack::pack_marlin_b`)
//! - C:row-major f16,`(m, n)`
//! - scales:f16,`(k/groupsize, n)`(scale_perm 重排,`repack::pack_marlin_s`)
//! - workspace:零初始化 i32,长度 ≥ [`v2_workspace_len(n)`]
//! - c_tmp:f32 reduce 缓冲(本固化路径 use_fp32_reduce=false,不触碰;
//!   传空指针即可,长度公式 [`c_tmp_float_len_v2`] 留作升级用)
//! - groupsize:`-1`(per-channel)或 `128`
//!
//! **发射通道**:server foreign-kernel(虚拟核名 [`GEMM_W4A16`],槽序契约
//! 见下方常量文档)——与 cublas_gemm_f16 同款,Launch 即唯一执行命令。
//!
//! 固化范围:no act-order / no zp / no bias / use_atomic_add=false /
//! use_fp32_reduce=false。AWQ(kU4 非对称)与 W4A8 变体在 .a 内保留,
//! 封装未引(需要时照 gemm_v2_raw 模式加)。

use std::ffi::c_void;

/// foreign-kernel 虚拟核名(server `is_foreign` 分派谓词的第二个臂)
pub const GEMM_W4A16: &str = "marlin_gemm_w4a16";

/// 槽序契约(LaunchMsg.args;与 cublas.rs 文档同构):
/// `[T a, T b, T out, T scales, T workspace, T c_tmp, sz m, sz k, sz n, sz groupsize]`
/// workspace 须零初始化(≥ [`v2_workspace_len(n)] i32);c_tmp 可传零容量块
/// (本固化路径不触碰)。
pub const GEMM_W4A16_SLOTS: &str =
    "[T a, T b, T out, T scales, T ws, T c_tmp, sz m, sz k, sz n, sz groupsize]";

/// foreign 分派谓词(server handle_launch 前置检查;cublas 同款)
pub fn is_foreign(name: &str) -> bool {
    name == GEMM_W4A16
}

pub const V2_ERR_NO_CONFIG: i32 = 3;
pub const V2_ERR_BAD_SHAPE: i32 = 4;
pub const V2_ERR_UNSUPPORTED_DEVICE: i32 = 5;

/// v2 workspace 长度(i32 个数):覆盖 host max_par=128 上限(n/64 列锁)。
pub fn v2_workspace_len(n: usize) -> usize {
    (n / 64) * 128
}

/// c_tmp 缓冲长度(f32 个数;use_fp32_reduce=true 路径才需要)。
pub fn c_tmp_float_len_v2(m: usize, sms: usize) -> usize {
    let m_block = (m.div_ceil(16) * 16).min(64);
    sms * m_block * 256
}

pub fn v2_err_str(err: i32) -> &'static str {
    match err {
        V2_ERR_NO_CONFIG => "no kernel config for this shape/dtype",
        V2_ERR_BAD_SHAPE => "invalid m/n/k/groupsize",
        V2_ERR_UNSUPPORTED_DEVICE => "device capability < 7.5 or query failed",
        _ => "see cudaGetErrorString(经 server 映射)",
    }
}

/// vLLM 增强 kernel 的 GEMM(裸指针;`stream` = cudaStream_t as usize,
/// `dev` = SM 数查询用 ordinal)。与 upstream 差异:vLLM 修正过的大 m
/// 发射循环、thread 配置自动搜索、sm_86 99KB 动态 smem。
///
/// # Safety
/// 指针须指向合法且长度匹配布局契约的设备内存;workspace 须零初始化
/// 且长度 ≥ [`v2_workspace_len(n)`]。
pub unsafe fn gemm_v2_raw(
    a: *const u16,
    b: *const i32,
    c: *mut u16,
    scales: *const u16,
    c_tmp: *const c_void,
    m: i32,
    n: i32,
    k: i32,
    workspace: *mut i32,
    groupsize: i32,
    dev: i32,
    stream: usize,
) -> Result<(), i32> {
    let err = marlin_gemm_v2_ffi(
        a as *const c_void,
        b as *const c_void,
        c as *mut c_void,
        c_tmp as *mut c_void,
        scales as *const c_void,
        m,
        n,
        k,
        workspace as *mut c_void,
        groupsize,
        dev,
        stream,
    );
    if err == 0 {
        Ok(())
    } else {
        Err(err)
    }
}

extern "C" {
    // csrc/marlin/marlin_host.cu(裸指针发射器;C ABI,零 torch)
    fn marlin_gemm_v2_ffi(
        a: *const c_void,
        b: *const c_void,
        c: *mut c_void,
        c_tmp: *mut c_void,
        b_scales: *const c_void,
        prob_m: i32,
        prob_n: i32,
        prob_k: i32,
        workspace: *mut c_void,
        group_size: i32,
        dev: i32,
        stream: usize,
    ) -> i32;
}

pub mod repack;

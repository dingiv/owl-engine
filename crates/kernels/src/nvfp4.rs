//! **NVFP4 块缩放 GEMM(sm_120a)** —— CUTLASS 适配器的 Rust 面。
//!
//! 库本体 = `cu/nvfp4/nvfp4_gemm.cu`(nvcc → `libnvfp4.a`,目标是
//! `compute_120a/sm_120a`)。本模块只是**薄 FFI 包装**,不含任何调度决策:
//! 选不选这条臂由 [`crate::driver::KernelPick`] 的准入表说话(`Caps::fp4_blockscale`
//! × `m >= 128`),不在这里另起一套 if。
//!
//! ## 生命周期(调用方职责)
//! 1. [`Nvfp4Plan::new`] —— 形状 `(M,N,K)` 定死;`M ≡ 0 (mod 128)`、`K ≡ 0 (mod 16)`。
//! 2. **权重**(B):`pack_b` 一次,产出 `q`(`N*K/2` 字节)+ `sf`(`sfb_bytes`),跨 M 复用
//!    (SFB 布局只依赖 `N,K`)。
//! 3. **激活**(A):每个 chunk `pack_a`(`M*K/2` + `sfa_bytes`);设备侧量化器见后续刀。
//! 4. `run` —— 发射到给定 stream(可被 CUDA graph 捕获)。
//!
//! ## workspace
//! [`Nvfp4Plan::ws_bytes`] 由 CUTLASS `get_workspace_size` 真算(实测 0,但**别依赖
//! 这个巧合**)。调用方负责按字节预分配、登记进显存账本,再把指针传给 `run`。

#![allow(unsafe_code)]

use std::ffi::{c_char, c_void, CStr};

extern "C" {
    fn owl_nvfp4_pack_a(m: i32, k: i32, a: *const c_void, q: *mut u8, sf: *mut u8) -> f32;
    fn owl_nvfp4_pack_b(n: i32, k: i32, b: *const c_void, q: *mut u8, sf: *mut u8) -> f32;
    /// 并行版(行切;与串行版**逐字节等价**)
    fn owl_nvfp4_pack_b_parallel(
        n: i32,
        k: i32,
        b: *const c_void,
        q: *mut u8,
        sf: *mut u8,
        nthreads: i32,
    ) -> f32;
    fn owl_nvfp4_plan_new(m: i32, n: i32, k: i32) -> *mut c_void;
    fn owl_nvfp4_plan_free(plan: *mut c_void);
    fn owl_nvfp4_plan_sfa_bytes(plan: *mut c_void) -> usize;
    fn owl_nvfp4_plan_sfb_bytes(plan: *mut c_void) -> usize;
    fn owl_nvfp4_plan_ws_bytes(plan: *mut c_void) -> usize;
    fn owl_nvfp4_run(
        plan: *mut c_void,
        stream: *mut c_void,
        aq: *const u8,
        asf: *const u8,
        ga: f32,
        bq: *const u8,
        bsf: *const u8,
        gb: f32,
        d: *mut c_void,
        ws: *mut c_void,
    ) -> i32;
    fn owl_nvfp4_banner() -> *const c_char;
    /// **`alpha` 来自设备指针**的发射版(进图路径;`alpha_dev` = f32[1])
    fn owl_nvfp4_run_dev(
        plan: *mut c_void,
        stream: *mut c_void,
        aq: *const u8,
        asf: *const u8,
        alpha_dev: *const f32,
        bq: *const u8,
        bsf: *const u8,
        d: *mut c_void,
        ws: *mut c_void,
    ) -> i32;
    fn owl_nvfp4_quant_act(
        plan: *mut c_void,
        x: *const c_void,
        q: *mut u8,
        sf: *mut u8,
        d_g: *mut f32,
        gb: f32,
        stream: *mut c_void,
    ) -> i32;
}

/// fp4 数据字节数(`k` 个半字节 = `k/2` 字节)
#[must_use]
pub const fn q_bytes(rows: usize, k: usize) -> usize {
    rows * k / 2
}

/// 缩放字节数(`N*K/16`,cosize 口径)—— **只依赖 `(N,K)`,与 `M` 无关**。
///
/// 依据:SFB 布局由 `tile_atom_to_shape_SFB(make_shape(1,N,K,1))` 定,`cosize`
/// 实测恒等于 `N*K/16`(M=1/7/128/512/2048 同 N,K 逐个核过)。装载期按它
/// 分配权重 `sf` 槽,运行期按同式验账长 ⇒ 两侧不许各算一套。
#[must_use]
pub const fn sf_bytes(n: usize, k: usize) -> usize {
    n * k / 16
}

/// **host 侧权重打包**(装载期唯一入口;不需要 plan)。
///
/// 为什么不需要 plan:C 侧 `owl_nvfp4_pack_b` 内部按 `make_shape(1,N,K,1)`
/// 自建 SFB 布局(只取 B 操作数的范围)⇒ 权重字节只依赖 `(N,K)`,
/// 与运行期的 `M` 无关(见 [`sf_bytes`])。
///
/// 返回 per-tensor 全局缩放 `gb` —— 它与写出的字节**必须同源**(`run(gb=…)`
/// 用的就是这里返回的值;换成别的 = 静默错值)。
///
/// # Errors
/// 形状不合规(`K%16`、`N%8`)或缓冲不足 ⇒ `Err`(**不许**截断装下)。
pub fn pack_weight_b(
    n: usize,
    k: usize,
    b_f16: &[u8],
    q: &mut [u8],
    sf: &mut [u8],
) -> Result<f32, String> {
    if k % 16 != 0 || n % 8 != 0 {
        return Err(format!(
            "NVFP4 权重形状 {n}x{k} 不合规(K 需 16 对齐、N 需 8 对齐)"
        ));
    }
    let (need_b, need_q, need_sf) = (n * k * 2, q_bytes(n, k), sf_bytes(n, k));
    if b_f16.len() < need_b || q.len() < need_q || sf.len() < need_sf {
        return Err(format!(
            "NVFP4 打包缓冲不足(b={}/{need_b} q={}/{need_q} sf={}/{need_sf})",
            b_f16.len(),
            q.len(),
            sf.len()
        ));
    }
    let gb = i32::try_from(n)
        .ok()
        .zip(i32::try_from(k).ok())
        .ok_or_else(|| format!("NVFP4 权重形状 {n}x{k} 超 i32"))?;
    // 行数够大才值得开线程(小张量:线程启动 > 收益)
    let nthreads = std::thread::available_parallelism()
        .map(|v| v.get())
        .unwrap_or(1)
        .min(8);
    // SAFETY:上面已按 `n*k*2` / `n*k/2` / `n*k/16` 校验缓冲长度;三个指针
    // 都指向本函数借用的活切片,C 侧只读 `b_f16`、只写 `q`/`sf` 的前若干字节。
    Ok(unsafe {
        owl_nvfp4_pack_b_parallel(
            gb.0,
            gb.1,
            b_f16.as_ptr().cast(),
            q.as_mut_ptr(),
            sf.as_mut_ptr(),
            nthreads as i32,
        )
    })
}

/// 自述串(诊断面/boot 打点)
#[must_use]
pub fn banner() -> &'static str {
    // SAFETY: 返回的是静态字符串字面量,进程生命周期内有效。
    unsafe { CStr::from_ptr(owl_nvfp4_banner()) }
        .to_str()
        .unwrap_or("owl_nvfp4")
}

/// 一个 `(M,N,K)` 绑定的执行计划(持 CUTLASS `GemmUniversalAdapter` 的参数面)。
pub struct Nvfp4Plan {
    ptr: *mut c_void,
    m: usize,
    n: usize,
    k: usize,
    sfa_bytes: usize,
    sfb_bytes: usize,
    ws_bytes: usize,
}

// SAFETY: 计划只持主机侧 POD(布局/尺寸),不碰设备资源;并发发射由调用方
// 保证 stream 语义(与其它 foreign 家族一致)。
unsafe impl Send for Nvfp4Plan {}

impl Nvfp4Plan {
    /// 建计划。`M % 128 == 0`、`N % 128 == 0`、`K % 16 == 0` 为前提(库侧拒绝时返回 `None`)。
    #[must_use]
    pub fn new(m: usize, n: usize, k: usize) -> Option<Self> {
        let (mi, ni, ki) = (i32::try_from(m).ok()?, i32::try_from(n).ok()?, i32::try_from(k).ok()?);
        // SAFETY: 纯主机构造,不逃逸指针。
        let ptr = unsafe { owl_nvfp4_plan_new(mi, ni, ki) };
        if ptr.is_null() {
            return None;
        }
        Some(Self {
            ptr,
            m,
            n,
            k,
            sfa_bytes: unsafe { owl_nvfp4_plan_sfa_bytes(ptr) },
            sfb_bytes: unsafe { owl_nvfp4_plan_sfb_bytes(ptr) },
            ws_bytes: unsafe { owl_nvfp4_plan_ws_bytes(ptr) },
        })
    }

    #[must_use]
    pub const fn shape(&self) -> (usize, usize, usize) {
        (self.m, self.n, self.k)
    }
    /// 激活 SF 字节数(`[M, ceil(K/16)]` 的 CUTLASS 交错布局)
    #[must_use]
    pub const fn sfa_bytes(&self) -> usize {
        self.sfa_bytes
    }
    /// 权重 SF 字节数(只依赖 `N,K`)
    #[must_use]
    pub const fn sfb_bytes(&self) -> usize {
        self.sfb_bytes
    }
    /// CUTLASS 报告的 workspace 字节数(实测 0;预分配请照此真值)
    #[must_use]
    pub const fn ws_bytes(&self) -> usize {
        self.ws_bytes
    }
    #[must_use]
    pub const fn a_q_bytes(&self) -> usize {
        q_bytes(self.m, self.k)
    }
    #[must_use]
    pub const fn b_q_bytes(&self) -> usize {
        q_bytes(self.n, self.k)
    }

    /// host 打包激活(f16 → e2m1 + e4m3/16)。返回 per-tensor 全局缩放。
    ///
    /// # Safety
    /// `a_f16` 至少 `m*k` 个 `half`;`q` ≥ [`Self::a_q_bytes`];`sf` ≥ [`Self::sfa_bytes`]。
    pub unsafe fn pack_a(&self, a_f16: &[u8], q: &mut [u8], sf: &mut [u8]) -> f32 {
        debug_assert!(a_f16.len() >= self.m * self.k * 2);
        debug_assert!(q.len() >= self.a_q_bytes() && sf.len() >= self.sfa_bytes);
        owl_nvfp4_pack_a(
            self.m as i32,
            self.k as i32,
            a_f16.as_ptr().cast(),
            q.as_mut_ptr(),
            sf.as_mut_ptr(),
        )
    }

    /// host 打包权重。返回 per-tensor 全局缩放。SFB 只依赖 `N,K` ⇒ 跨 `M` 复用。
    ///
    /// # Safety
    /// `b_f16` 至少 `n*k` 个 `half`;`q` ≥ [`Self::b_q_bytes`];`sf` ≥ [`Self::sfb_bytes`]。
    pub unsafe fn pack_b(&self, b_f16: &[u8], q: &mut [u8], sf: &mut [u8]) -> f32 {
        debug_assert!(b_f16.len() >= self.n * self.k * 2);
        debug_assert!(q.len() >= self.b_q_bytes() && sf.len() >= self.sfb_bytes);
        owl_nvfp4_pack_b(
            self.n as i32,
            self.k as i32,
            b_f16.as_ptr().cast(),
            q.as_mut_ptr(),
            sf.as_mut_ptr(),
        )
    }

    /// 设备侧量化器(kernel 内;同一 `.a`)—— f16 `[M,K]` → fp4 + SF。
    ///
    /// 三发:**清 `d_g` → amax → (`g = amax/(6·448)` 与 `alpha = g·gb`)**。
    /// **一槽两值**(`d_g` = f32[2]):`d_g[0]` = `g`(量化核读),
    /// `d_g[1]` = `alpha`(GEMM 的 `alpha_ptr` = `d_g + 1`)。
    /// 全局缩放**留在设备侧** ⇒ 零 D2H、零同步 ⇒ 可进 CUDA graph。
    ///
    /// `gb` 是**权重侧 per-tensor 全局缩放**(装载期 `pack_weight_b` 算出、层上常驻),
    /// 所以乘积能在核内直接做。
    ///
    /// 返回 0 = 已发射(未同步);非 0 = 主机侧失败。
    ///
    /// # Safety
    /// `x` = 设备端 f16 `[m,k]`;`q` ≥ [`Self::a_q_bytes`];`sf` ≥ [`Self::sfa_bytes`];
    /// `d_g` = 设备端 **8 字节**(f32[2];**由本函数自己清零**,调用方不必预清);
    /// `stream` = 有效 `cudaStream_t`。
    pub unsafe fn quant_act(
        &self,
        x: *const c_void,
        q: *mut u8,
        sf: *mut u8,
        d_g: *mut f32,
        gb: f32,
        stream: *mut c_void,
    ) -> i32 {
        owl_nvfp4_quant_act(self.ptr, x, q, sf, d_g, gb, stream)
    }

    /// 发射 `D = (ga*gb)·A×Bᵀ`(`D` 是 f16 `[M,N]` 行主序)。
    ///
    /// `alpha` 由 **host** 给 —— 只留给**非捕获面**(装载期 oracle / 探针 /
    /// 对照实验)。生产路径走 [`Self::run_dev`]。
    ///
    /// 返回 0 = 已发射(未同步);负值 = 主机侧失败(`-2` 形状不支持 / `-3` 初始化失败 / `-4` 发射失败)。
    ///
    /// # Safety
    /// 四个设备指针与 `ws` 必须是本 plan 形状下的合法分配;`stream` = 有效的 `cudaStream_t`。
    pub unsafe fn run(
        &self,
        stream: *mut c_void,
        aq: *const u8,
        asf: *const u8,
        ga: f32,
        bq: *const u8,
        bsf: *const u8,
        gb: f32,
        d: *mut c_void,
        ws: *mut c_void,
    ) -> i32 {
        owl_nvfp4_run(self.ptr, stream, aq, asf, ga, bq, bsf, gb, d, ws)
    }

    /// 发射 `D = alpha·A×Bᵀ`,**`alpha` 从设备读**(`alpha_dev` = f32[1])。
    ///
    /// 这是**进图路径**(2026-10-28):`alpha = g·gb` 已由[`Self::quant_act`] 在
    /// 设备上写好 ⇒ 整条链零 host 工作 ⇒ 可被 CUDA graph 捕获。
    ///
    /// 返回 0 = 已发射(未同步);负值同上。
    ///
    /// # Safety
    /// `alpha_dev` 指向设备端 4 字节且在本 kernel 读它之前已写定(同流序由
    /// `quant_act` 保证);其余指针同 [`Self::run`]。
    pub unsafe fn run_dev(
        &self,
        stream: *mut c_void,
        aq: *const u8,
        asf: *const u8,
        alpha_dev: *const f32,
        bq: *const u8,
        bsf: *const u8,
        d: *mut c_void,
        ws: *mut c_void,
    ) -> i32 {
        owl_nvfp4_run_dev(self.ptr, stream, aq, asf, alpha_dev, bq, bsf, d, ws)
    }
}

impl Drop for Nvfp4Plan {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: 指针由 `plan_new` 产出且只在此释放一次。
            unsafe { owl_nvfp4_plan_free(self.ptr) };
        }
    }
}

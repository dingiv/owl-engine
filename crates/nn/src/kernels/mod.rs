//! port 自 candle-kernels 的 .cu + cudarc nvrtc 编译封装(T0)。
//!
//! 出处:见 `kernels/cu/owl_nn_kernels.cu` 头部注释。
//! 设计约束(charter 裁决 3①):launch 参数只用标量与裸指针,
//! 无运行时 htod 形状数组;contiguous-only。

pub mod attention;
pub mod gdn_kernels;

use owl_cuda::ffi::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use owl_cuda::ffi::nvrtc::{compile_ptx_with_opts, CompileOptions};
use std::collections::HashSet;
use std::sync::Arc;

/// kernel 源码(include_str 保编译期一致性;出处见 .cu 头部)
const KERNEL_SRC: &str = include_str!("../../kernels/cu/owl_nn_kernels.cu");

pub struct Kernels {
    ctx: Arc<CudaContext>,
    module: Arc<owl_cuda::ffi::CudaModule>,
    loaded: HashSet<&'static str>,
}

impl Kernels {
    /// nvrtc 编译 + 加载(进程内一次;编译结果随 module 复用)
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, String> {
        // nvrtc 无内建系统头;显式给 CUDA include 路径(cuda_fp16.h 所在)
        let include = std::env::var("CUDA_INCLUDE")
            .unwrap_or_else(|_| "/usr/local/cuda/include".to_string());
        let opts = CompileOptions {
            include_paths: vec![include],
            use_fast_math: Some(true),
            ..Default::default()
        };
        let ptx =
            compile_ptx_with_opts(KERNEL_SRC, opts).map_err(|e| format!("nvrtc: {e}"))?;
        let module = ctx.load_module(ptx).map_err(|e| format!("load_module: {e}"))?;
        Ok(Self {
            ctx: Arc::clone(ctx),
            module,
            loaded: HashSet::new(),
        })
    }

    fn func(&mut self, name: &'static str) -> Result<CudaFunction, String> {
        if self.loaded.insert(name) {
            // load_function 幂等,但缓存调用减少字符串查找
            self.module
                .load_function(name)
                .map_err(|e| format!("load_function({name}): {e}"))?;
        }
        self.module
            .load_function(name)
            .map_err(|e| format!("load_function({name}): {e}"))
    }

    fn launch_f32(
        &mut self,
        stream: &CudaStream,
        name: &'static str,
        n: usize,
        args: &[u64],
    ) -> Result<(), String> {
        let func = self.func(name)?;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        unsafe {
            let mut builder = stream.launch_builder(&func);
            for a in args {
                builder.arg(a);
            }
            builder
                .launch(cfg)
                .map_err(|e| format!("launch({name}): {e}"))?;
        }
        let _ = &self.ctx;
        Ok(())
    }

    // ---- 类型安全薄封装(标量 + 裸指针,裁决 3①)----

    pub fn add_f32(
        &mut self,
        stream: &CudaStream,
        n: usize,
        a: *const f32,
        b: *const f32,
        out: *mut f32,
    ) -> Result<(), String> {
        self.launch_f32(
            stream,
            "owl_add_f32",
            n,
            &[n as u64, a as u64, b as u64, out as u64],
        )
    }

    pub fn mul_f32(
        &mut self,
        stream: &CudaStream,
        n: usize,
        a: *const f32,
        b: *const f32,
        out: *mut f32,
    ) -> Result<(), String> {
        self.launch_f32(
            stream,
            "owl_mul_f32",
            n,
            &[n as u64, a as u64, b as u64, out as u64],
        )
    }

    pub fn silu_f32(
        &mut self,
        stream: &CudaStream,
        n: usize,
        inp: *const f32,
        out: *mut f32,
    ) -> Result<(), String> {
        self.launch_f32(stream, "owl_silu_f32", n, &[n as u64, inp as u64, out as u64])
    }

    pub fn exp_f32(
        &mut self,
        stream: &CudaStream,
        n: usize,
        inp: *const f32,
        out: *mut f32,
    ) -> Result<(), String> {
        self.launch_f32(stream, "owl_exp_f32", n, &[n as u64, inp as u64, out as u64])
    }

    pub fn gelu_f32(
        &mut self,
        stream: &CudaStream,
        n: usize,
        inp: *const f32,
        out: *mut f32,
    ) -> Result<(), String> {
        self.launch_f32(stream, "owl_gelu_f32", n, &[n as u64, inp as u64, out as u64])
    }

    /// softmax last-dim:src/dst 均为 [rows, n_cols] contiguous
    pub fn softmax_f32(
        &mut self,
        stream: &CudaStream,
        rows: usize,
        n_cols: usize,
        src: *const f32,
        dst: *mut f32,
    ) -> Result<(), String> {
        if n_cols == 0 {
            return Ok(());
        }
        let func = self.func("owl_softmax_f32")?;
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 256 * 4,
        };
        let src_u = src as u64;
        let dst_u = dst as u64;
        let cols = n_cols as i32;
        unsafe {
            let mut b = stream.launch_builder(&func);
            b.arg(&src_u);
            b.arg(&dst_u);
            b.arg(&cols);
            b.launch(cfg).map_err(|e| format!("launch(softmax): {e}"))?;
        }
        Ok(())
    }

    /// rmsnorm:[rows, n_cols],alpha 长度 n_cols
    pub fn rmsnorm_f32(
        &mut self,
        stream: &CudaStream,
        rows: usize,
        n_cols: usize,
        src: *const f32,
        alpha: *const f32,
        dst: *mut f32,
        eps: f32,
        w_off: i32,
    ) -> Result<(), String> {
        if n_cols == 0 {
            return Ok(());
        }
        let func = self.func("owl_rmsnorm_f32")?;
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 256 * 4,
        };
        let src_u = src as u64;
        let dst_u = dst as u64;
        let alpha_u = alpha as u64;
        let cols = n_cols as i32;
        unsafe {
            let mut b = stream.launch_builder(&func);
            b.arg(&src_u);
            b.arg(&dst_u);
            b.arg(&alpha_u);
            b.arg(&cols);
            b.arg(&eps);
            b.arg(&w_off);
            b.launch(cfg).map_err(|e| format!("launch(rmsnorm): {e}"))?;
        }
        Ok(())
    }

    /// rope rotate-half:[rows, n_cols],pos: [rows] i64(token 位置),
    /// theta_base 标量(裁决 3①:动态量 device 化,launch 无 htod)。
    pub fn rope_f32(
        &mut self,
        stream: &CudaStream,
        rows: usize,
        n_cols: usize,
        x: *const f32,
        pos: *const i64,
        out: *mut f32,
        theta_base: f32,
    ) -> Result<(), String> {
        if n_cols == 0 || n_cols % 2 != 0 {
            return Err(format!("rope: n_cols 必须为正偶数,得到 {n_cols}"));
        }
        let func = self.func("owl_rope_f32")?;
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let x_u = x as u64;
        let pos_u = pos as u64;
        let out_u = out as u64;
        let cols = n_cols as i32;
        unsafe {
            let mut b = stream.launch_builder(&func);
            b.arg(&x_u);
            b.arg(&pos_u);
            b.arg(&out_u);
            b.arg(&cols);
            b.arg(&theta_base);
            b.launch(cfg).map_err(|e| format!("launch(rope): {e}"))?;
        }
        Ok(())
    }

    /// embedding lookup:table [vocab, n_cols],ids [rows] i32,out [rows, n_cols]。
    pub fn embedding_f32(
        &mut self,
        stream: &CudaStream,
        rows: usize,
        n_cols: usize,
        table: *const f32,
        ids: *const i32,
        out: *mut f32,
    ) -> Result<(), String> {
        if n_cols == 0 {
            return Ok(());
        }
        let func = self.func("owl_embedding_f32")?;
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let table_u = table as u64;
        let ids_u = ids as u64;
        let out_u = out as u64;
        let cols = n_cols as i32;
        unsafe {
            let mut b = stream.launch_builder(&func);
            b.arg(&table_u);
            b.arg(&ids_u);
            b.arg(&out_u);
            b.arg(&cols);
            b.launch(cfg).map_err(|e| format!("launch(embedding): {e}"))?;
        }
        Ok(())
    }

    // ---- P1 索引/归约/broadcast 族(owl 原创核,2026-09-22)----

    /// gather:dst[i] = src[idx[i]]
    pub fn gather_f32(
        &mut self,
        stream: &CudaStream,
        n: usize,
        src: *const f32,
        idx: *const u32,
        dst: *mut f32,
    ) -> Result<(), String> {
        self.launch_f32(stream, "owl_gather_f32", n, &[n as u64, src as u64, idx as u64, dst as u64])
    }

    /// tile 单维(任意 dim;dim0 → outer=1,last → inner=1)
    pub fn tile_dim_f32(
        &mut self,
        stream: &CudaStream,
        outer: u64,
        d_size: u64,
        inner: u64,
        tiles: u64,
        src: *const f32,
        dst: *mut f32,
    ) -> Result<(), String> {
        let n = (outer * d_size * tiles * inner) as usize;
        self.launch_f32(
            stream,
            "owl_tile_dim_f32",
            n,
            &[outer, d_size, inner, tiles, src as u64, dst as u64],
        )
    }

    /// repeat_interleave 单维
    pub fn rep_interleave_dim_f32(
        &mut self,
        stream: &CudaStream,
        outer: u64,
        d_size: u64,
        inner: u64,
        repeats: u64,
        src: *const f32,
        dst: *mut f32,
    ) -> Result<(), String> {
        let n = (outer * d_size * repeats * inner) as usize;
        self.launch_f32(
            stream,
            "owl_rep_interleave_dim_f32",
            n,
            &[outer, d_size, inner, repeats, src as u64, dst as u64],
        )
    }

    /// rows-gather(dim0 index_select;idx 设备侧 U32)
    pub fn rows_gather_f32(
        &mut self,
        stream: &CudaStream,
        n_idx: u64,
        inner: u64,
        src: *const f32,
        idx: *const u32,
        dst: *mut f32,
    ) -> Result<(), String> {
        self.launch_f32(
            stream,
            "owl_rows_gather_f32",
            (n_idx * inner) as usize,
            &[n_idx, inner, src as u64, idx as u64, dst as u64],
        )
    }

    /// scatter_add:atomicAdd(&dst[idx[i]], src[i])
    pub fn scatter_add_f32(
        &mut self,
        stream: &CudaStream,
        n: usize,
        src: *const f32,
        idx: *const u32,
        dst: *mut f32,
    ) -> Result<(), String> {
        self.launch_f32(stream, "owl_scatter_add_f32", n, &[n as u64, src as u64, idx as u64, dst as u64])
    }

    /// 轴归约 sum:src [outer, axis, inner] → dst [outer, inner]
    pub fn sum_axis_f32(
        &mut self,
        stream: &CudaStream,
        outer: usize,
        axis: usize,
        inner: usize,
        src: *const f32,
        dst: *mut f32,
    ) -> Result<(), String> {
        self.launch_f32(
            stream,
            "owl_sum_axis_f32",
            outer * inner,
            &[outer as u64, axis as u64, inner as u64, src as u64, dst as u64],
        )
    }

    /// 轴归约 max:同 sum_axis 形状约定
    pub fn max_axis_f32(
        &mut self,
        stream: &CudaStream,
        outer: usize,
        axis: usize,
        inner: usize,
        src: *const f32,
        dst: *mut f32,
    ) -> Result<(), String> {
        self.launch_f32(
            stream,
            "owl_max_axis_f32",
            outer * inner,
            &[outer as u64, axis as u64, inner as u64, src as u64, dst as u64],
        )
    }

    /// 右对齐 broadcast 加(S2:b 任意轴 = 1)
    #[allow(clippy::too_many_arguments)]
    pub fn add_bcast_f32(
        &mut self,
        stream: &CudaStream,
        outer: usize,
        mid: usize,
        inner: usize,
        b_mid: usize,
        a: *const f32,
        b: *const f32,
        out: *mut f32,
    ) -> Result<(), String> {
        self.launch_f32(
            stream,
            "owl_add_bcast_f32",
            outer * mid * inner,
            &[outer as u64, mid as u64, inner as u64, b_mid as u64, a as u64, b as u64, out as u64],
        )
    }

    /// 右对齐 broadcast 乘(同 add_bcast 形状约定)
    #[allow(clippy::too_many_arguments)]
    pub fn mul_bcast_f32(
        &mut self,
        stream: &CudaStream,
        outer: usize,
        mid: usize,
        inner: usize,
        b_mid: usize,
        a: *const f32,
        b: *const f32,
        out: *mut f32,
    ) -> Result<(), String> {
        self.launch_f32(
            stream,
            "owl_mul_bcast_f32",
            outer * mid * inner,
            &[outer as u64, mid as u64, inner as u64, b_mid as u64, a as u64, b as u64, out as u64],
        )
    }
}

// f16 封装:任务要求 add/mul f16(容差放宽);ptr 仍为裸指针
impl Kernels {
    pub fn add_f16(
        &mut self,
        stream: &CudaStream,
        n: usize,
        a: *const u16, // __half 以 u16 位型传递
        b: *const u16,
        out: *mut u16,
    ) -> Result<(), String> {
        self.launch_f32(
            stream,
            "owl_add_f16",
            n,
            &[n as u64, a as u64, b as u64, out as u64],
        )
    }

    pub fn mul_f16(
        &mut self,
        stream: &CudaStream,
        n: usize,
        a: *const u16,
        b: *const u16,
        out: *mut u16,
    ) -> Result<(), String> {
        self.launch_f32(
            stream,
            "owl_mul_f16",
            n,
            &[n as u64, a as u64, b as u64, out as u64],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use owl_cuda::ffi::{CudaSlice, DevicePtr};

    pub(crate) fn setup() -> (Arc<CudaContext>, Arc<CudaStream>, Kernels) {
        let ctx = CudaContext::new(0).expect("需要 CUDA 设备");
        // M1②:非阻塞流(与设备主流纪律一致;legacy 流不可捕获)
        let stream = ctx.new_stream().expect("non-blocking 流");
        let k = Kernels::new(&ctx).expect("nvrtc 编译");
        (ctx, stream, k)
    }

    pub(crate) fn htod_f32(stream: &Arc<CudaStream>, v: &[f32]) -> CudaSlice<f32> {
        let mut s = stream.alloc_zeros::<f32>(v.len()).unwrap();
        stream.memcpy_htod(v, &mut s).unwrap();
        s
    }

    pub(crate) fn dtoh_f32(stream: &Arc<CudaStream>, s: &CudaSlice<f32>) -> Vec<f32> {
        let mut out = vec![0f32; s.len()];
        stream.memcpy_dtoh(s, &mut out).unwrap();
        out
    }

    /// 固定种子 LCG(可复现输入;裁决 4)
    pub(crate) const SEED: u64 = 20260922;
    pub(crate) struct Lcg(pub(crate) u64);
    impl Lcg {
        pub(crate) fn next_f32(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        }
    }

    /// DevicePtr 版本返回 (CUdeviceptr, SyncOnDrop);测试用简单映射:
    /// 分配后的 CudaSlice 指针在 stream 存活期内稳定,直接转 u64。
    fn slice_ptr<T>(s: &CudaSlice<T>, stream: &Arc<CudaStream>) -> u64 {
        let (p, _sync) = s.device_ptr(stream);
        p
    }

    pub(crate) fn assert_close(a: &[f32], b: &[f32], abs: f32, what: &str) {
        assert!(
            a.len() == b.len()
                && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= abs + 1e-5 * y.abs().max(1.0)),
            "{what} 不匹配:\n  got  {a:?}\n  want {b:?}"
        );
    }

    #[test]
    fn binary_add_mul_f32() {
        let (_ctx, stream, mut k) = setup();
        let mut rng = Lcg(SEED);
        let n = 4096;
        let a: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
        let b: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
        let da = htod_f32(&stream, &a);
        let db = htod_f32(&stream, &b);
        let dc = stream.alloc_zeros::<f32>(n).unwrap();

        let (pa, _sa) = da.device_ptr(&stream);
        let (pb, _sb) = db.device_ptr(&stream);
        let pc = slice_ptr(&dc, &stream);

        k.add_f32(&stream, n, pa as *const f32, pb as *const f32, pc as *mut f32)
            .unwrap();
        let got = dtoh_f32(&stream, &dc);
        let want: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
        assert_close(&got, &want, 1e-5, "add_f32");

        let (pa, _sa) = da.device_ptr(&stream);
        let (pb, _sb) = db.device_ptr(&stream);
        let pc = slice_ptr(&dc, &stream);
        k.mul_f32(&stream, n, pa as *const f32, pb as *const f32, pc as *mut f32)
            .unwrap();
        let got = dtoh_f32(&stream, &dc);
        let want: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x * y).collect();
        assert_close(&got, &want, 1e-5, "mul_f32");
    }

    #[test]
    fn unary_silu_exp_gelu_f32() {
        let (_ctx, stream, mut k) = setup();
        let mut rng = Lcg(SEED ^ 0xBEEF);
        let n = 2048;
        let a: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
        let da = htod_f32(&stream, &a);
        let dc = stream.alloc_zeros::<f32>(n).unwrap();
        let (pa, _sa) = da.device_ptr(&stream);
        let pc = slice_ptr(&dc, &stream);

        k.silu_f32(&stream, n, pa as *const f32, pc as *mut f32).unwrap();
        let got = dtoh_f32(&stream, &dc);
        let want: Vec<f32> = a.iter().map(|x| x / (1.0 + (-x).exp())).collect();
        assert_close(&got, &want, 1e-5, "silu_f32");

        let (pa, _sa) = da.device_ptr(&stream);
        let pc = slice_ptr(&dc, &stream);
        k.exp_f32(&stream, n, pa as *const f32, pc as *mut f32).unwrap();
        let got = dtoh_f32(&stream, &dc);
        let want: Vec<f32> = a.iter().map(|x| x.exp()).collect();
        assert_close(&got, &want, 1e-5, "exp_f32");

        let (pa, _sa) = da.device_ptr(&stream);
        let pc = slice_ptr(&dc, &stream);
        k.gelu_f32(&stream, n, pa as *const f32, pc as *mut f32).unwrap();
        let got = dtoh_f32(&stream, &dc);
        let want: Vec<f32> = a
            .iter()
            .map(|x| 0.5 * x * (1.0 + (0.7978845608028654 * (x + 0.044715 * x * x * x)).tanh()))
            .collect();
        assert_close(&got, &want, 1e-5, "gelu_f32");
    }

    #[test]
    fn softmax_f32_last_dim() {
        let (_ctx, stream, mut k) = setup();
        let mut rng = Lcg(SEED ^ 0xCAFE);
        let rows = 8;
        let cols = 512;
        let src: Vec<f32> = (0..rows * cols).map(|_| rng.next_f32() * 4.0).collect();
        let dsrc = htod_f32(&stream, &src);
        let ddst = stream.alloc_zeros::<f32>(rows * cols).unwrap();
        let (ps, _s1) = dsrc.device_ptr(&stream);
        let pd = slice_ptr(&ddst, &stream);

        k.softmax_f32(&stream, rows, cols, ps as *const f32, pd as *mut f32)
            .unwrap();
        let got = dtoh_f32(&stream, &ddst);
        for r in 0..rows {
            let row = &src[r * cols..(r + 1) * cols];
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = row.iter().map(|x| (x - max).exp()).sum();
            let want: Vec<f32> = row.iter().map(|x| (x - max).exp() / sum).collect();
            assert_close(&got[r * cols..(r + 1) * cols], &want, 1e-5, &format!("softmax row {r}"));
            let s: f32 = got[r * cols..(r + 1) * cols].iter().sum();
            assert!((s - 1.0).abs() < 1e-4, "softmax 行和 != 1: {s}");
        }
    }

    #[test]
    fn rmsnorm_f32() {
        let (_ctx, stream, mut k) = setup();
        let mut rng = Lcg(SEED ^ 0xD00D);
        let rows = 4;
        let cols = 256;
        let eps = 1e-6;
        let src: Vec<f32> = (0..rows * cols).map(|_| rng.next_f32()).collect();
        let alpha: Vec<f32> = (0..cols).map(|i| 0.5 + 0.1 * i as f32).collect();
        let dsrc = htod_f32(&stream, &src);
        let dalpha = htod_f32(&stream, &alpha);
        let ddst = stream.alloc_zeros::<f32>(rows * cols).unwrap();
        let (ps, _s1) = dsrc.device_ptr(&stream);
        let pa = slice_ptr(&dalpha, &stream);
        let pd = slice_ptr(&ddst, &stream);

        k.rmsnorm_f32(&stream, rows, cols, ps as *const f32, pa as *const f32, pd as *mut f32, eps, 0)
            .unwrap();
        let got = dtoh_f32(&stream, &ddst);
        for r in 0..rows {
            let row = &src[r * cols..(r + 1) * cols];
            let ms: f32 = row.iter().map(|x| x * x).sum::<f32>() / cols as f32;
            let inv = 1.0 / (ms + eps).sqrt();
            let want: Vec<f32> = row
                .iter()
                .zip(&alpha)
                .map(|(x, a)| x * inv * a)
                .collect();
            assert_close(&got[r * cols..(r + 1) * cols], &want, 1e-4, &format!("rmsnorm row {r}"));
        }
    }

    /// f16 add:位型 u16 直接上卡,容差放宽(f16 精度)
    #[test]
    fn add_f16_bitpattern() {
        let (_ctx, stream, mut k) = setup();
        let mut rng = Lcg(SEED ^ 0xF16);
        let n = 1024;
        let a: Vec<u16> = (0..n)
            .map(|_| f32_to_f16_bits((rng.next_f32() * 2.0 - 1.0) * 4.0))
            .collect();
        let b: Vec<u16> = (0..n)
            .map(|_| f32_to_f16_bits((rng.next_f32() * 2.0 - 1.0) * 4.0))
            .collect();
        let mut da = stream.alloc_zeros::<u16>(n).unwrap();
        stream.memcpy_htod(&a, &mut da).unwrap();
        let mut db = stream.alloc_zeros::<u16>(n).unwrap();
        stream.memcpy_htod(&b, &mut db).unwrap();
        let dc = stream.alloc_zeros::<u16>(n).unwrap();
        let pa = slice_ptr(&da, &stream);
        let pb = slice_ptr(&db, &stream);
        let pc = slice_ptr(&dc, &stream);

        k.add_f16(&stream, n, pa as *const u16, pb as *const u16, pc as *mut u16)
            .unwrap();
        let mut gotv = vec![0u16; n];
        stream.memcpy_dtoh(&dc, &mut gotv).unwrap();
        let got = gotv;
        for i in 0..n {
            let x = f16_bits_to_f32(a[i]);
            let y = f16_bits_to_f32(b[i]);
            let want = x + y;
            let gotv = f16_bits_to_f32(got[i]);
            assert!(
                (gotv - want).abs() < 1e-2 * want.abs().max(1.0),
                "add_f16[{i}]: {gotv} vs {want}"
            );
        }
    }

    // ---- 最小 f16 位型工具(不引依赖;仅测试域使用)----
    fn f32_to_f16_bits(f: f32) -> u16 {
        let bits = f.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let abs = bits & 0x7FFF_FFFF;
        if abs >= 0x477F_F000 {
            return sign | 0x7BFF;
        }
        if abs < 0x3300_0000 {
            return sign;
        }
        let exp = ((abs >> 23) as i32) - 127 + 15;
        let man = (abs >> 13) & 0x3FF;
        sign | (((exp as u16) & 0x1F) << 10) | (man as u16)
    }

    fn f16_bits_to_f32(h: u16) -> f32 {
        let sign = ((h & 0x8000) as u32) << 16;
        let exp = ((h & 0x7C00) >> 10) as u32;
        let man = (h & 0x3FF) as u32;
        let bits = if exp == 0 {
            sign
        } else {
            sign | ((exp + 127 - 15) << 23) | (man << 13)
        };
        f32::from_bits(bits)
    }
}

#[cfg(test)]
mod rope_embed_tests {
    use super::tests::{dtoh_f32, htod_f32, setup, Lcg, SEED};
    
    use owl_cuda::ffi::DevicePtr;

    /// rope rotate-half:CPU f64 参考(与 kernel 同公式)
    fn rope_ref(x: &[f32], pos: &[i64], n_cols: usize, theta: f32) -> Vec<f64> {
        let half = n_cols / 2;
        let rows = pos.len();
        let mut out = vec![0f64; rows * n_cols];
        for (r, &p) in pos.iter().enumerate() {
            for i in 0..half {
                let inv_freq = (theta as f64).powf(-(2.0 * i as f64) / n_cols as f64);
                let ang = p as f64 * inv_freq;
                let (c, s) = (ang.cos(), ang.sin());
                let x1 = x[r * n_cols + i] as f64;
                let x2 = x[r * n_cols + i + half] as f64;
                out[r * n_cols + i] = x1 * c - x2 * s;
                out[r * n_cols + i + half] = x1 * s + x2 * c;
            }
        }
        out
    }

    #[test]
    fn rope_f32_rotate_half() {
        let (ctx, stream, mut k) = setup();
        let rows = 8;
        let n_cols = 64;
        let mut rng = Lcg(SEED);
        let x: Vec<f32> = (0..rows * n_cols).map(|_| rng.next_f32()).collect();
        let pos: Vec<i64> = (0..rows).map(|i| (i * 97) as i64).collect();

        let dx = htod_f32(&stream, &x);
        let mut dpos = stream.alloc_zeros::<i64>(rows).unwrap();
        stream.memcpy_htod(&pos, &mut dpos).unwrap();
        let dout = stream.alloc_zeros::<f32>(rows * n_cols).unwrap();

        k.rope_f32(&stream, rows, n_cols, dx.device_ptr(&stream).0 as *const f32, dpos.device_ptr(&stream).0 as *const i64, dout.device_ptr(&stream).0 as *mut f32, 10000.0)
            .unwrap();
        ctx.synchronize().unwrap();

        let got = dtoh_f32(&stream, &dout);
        let want = rope_ref(&x, &pos, n_cols, 10000.0);
        let max_diff = got
            .iter()
            .zip(&want)
            .map(|(g, w)| ((*g as f64) - *w).abs())
            .fold(0.0f64, f64::max);
        assert!(max_diff < 1e-4, "rope 最大偏差 {max_diff}");
    }

    #[test]
    fn embedding_f32_lookup() {
        let (ctx, stream, mut k) = setup();
        let vocab = 16;
        let n_cols = 8;
        let rows = 4;
        let mut rng = Lcg(SEED + 1);
        let table: Vec<f32> = (0..vocab * n_cols).map(|_| rng.next_f32()).collect();
        let ids: Vec<i32> = vec![3, 0, 15, 7];

        let dtable = htod_f32(&stream, &table);
        let mut dids = stream.alloc_zeros::<i32>(rows).unwrap();
        stream.memcpy_htod(&ids, &mut dids).unwrap();
        let dout = stream.alloc_zeros::<f32>(rows * n_cols).unwrap();

        k.embedding_f32(
            &stream,
            rows,
            n_cols,
            dtable.device_ptr(&stream).0 as *const f32,
            dids.device_ptr(&stream).0 as *const i32,
            dout.device_ptr(&stream).0 as *mut f32,
        )
        .unwrap();
        ctx.synchronize().unwrap();

        let got = dtoh_f32(&stream, &dout);
        for (r, &id) in ids.iter().enumerate() {
            let want = &table[id as usize * n_cols..(id as usize + 1) * n_cols];
            let got_row = &got[r * n_cols..(r + 1) * n_cols];
            for (c, (g, w)) in got_row.iter().zip(want).enumerate() {
                assert!(
                    (g - w).abs() <= 1e-6 + 1e-5 * w.abs(),
                    "embedding row {r} col {c} (id={id}): {g} != {w}"
                );
            }
        }
    }
}

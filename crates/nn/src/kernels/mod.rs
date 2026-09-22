//! port 自 candle-kernels 的 .cu + cudarc nvrtc 编译封装(T0)。
//!
//! 出处:见 `kernels/cu/owl_nn_kernels.cu` 头部注释。
//! 设计约束(charter 裁决 3①):launch 参数只用标量与裸指针,
//! 无运行时 htod 形状数组;contiguous-only。

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
            b.launch(cfg).map_err(|e| format!("launch(rmsnorm): {e}"))?;
        }
        Ok(())
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

    fn setup() -> (Arc<CudaContext>, Arc<CudaStream>, Kernels) {
        let ctx = CudaContext::new(0).expect("需要 CUDA 设备");
        let stream = ctx.default_stream();
        let k = Kernels::new(&ctx).expect("nvrtc 编译");
        (ctx, stream, k)
    }

    fn htod_f32(stream: &Arc<CudaStream>, v: &[f32]) -> CudaSlice<f32> {
        let mut s = stream.alloc_zeros::<f32>(v.len()).unwrap();
        stream.memcpy_htod(v, &mut s).unwrap();
        s
    }

    fn dtoh_f32(stream: &Arc<CudaStream>, s: &CudaSlice<f32>) -> Vec<f32> {
        let mut out = vec![0f32; s.len()];
        stream.memcpy_dtoh(s, &mut out).unwrap();
        out
    }

    /// 固定种子 LCG(可复现输入;裁决 4)
    const SEED: u64 = 20260922;
    struct Lcg(u64);
    impl Lcg {
        fn next_f32(&mut self) -> f32 {
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

    fn assert_close(a: &[f32], b: &[f32], abs: f32, what: &str) {
        assert!(
            a.len() == b.len()
                && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= abs + 1e-5 * y.abs().max(1.0)),
            "{what} 不匹配:\n  got  {a:?}\n  want {b:?}"
        );
    }

    #[test]
    fn binary_add_mul_f32() {
        let (ctx, stream, mut k) = setup();
        let mut rng = Lcg(SEED);
        let n = 4096;
        let a: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
        let b: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
        let da = htod_f32(&stream, &a);
        let db = htod_f32(&stream, &b);
        let mut dc = stream.alloc_zeros::<f32>(n).unwrap();

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
        let (ctx, stream, mut k) = setup();
        let mut rng = Lcg(SEED ^ 0xBEEF);
        let n = 2048;
        let a: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
        let da = htod_f32(&stream, &a);
        let mut dc = stream.alloc_zeros::<f32>(n).unwrap();
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
        let (ctx, stream, mut k) = setup();
        let mut rng = Lcg(SEED ^ 0xCAFE);
        let rows = 8;
        let cols = 512;
        let src: Vec<f32> = (0..rows * cols).map(|_| rng.next_f32() * 4.0).collect();
        let dsrc = htod_f32(&stream, &src);
        let mut ddst = stream.alloc_zeros::<f32>(rows * cols).unwrap();
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
        let (ctx, stream, mut k) = setup();
        let mut rng = Lcg(SEED ^ 0xD00D);
        let rows = 4;
        let cols = 256;
        let eps = 1e-6;
        let src: Vec<f32> = (0..rows * cols).map(|_| rng.next_f32()).collect();
        let alpha: Vec<f32> = (0..cols).map(|i| 0.5 + 0.1 * i as f32).collect();
        let dsrc = htod_f32(&stream, &src);
        let dalpha = htod_f32(&stream, &alpha);
        let mut ddst = stream.alloc_zeros::<f32>(rows * cols).unwrap();
        let (ps, _s1) = dsrc.device_ptr(&stream);
        let pa = slice_ptr(&dalpha, &stream);
        let pd = slice_ptr(&ddst, &stream);

        k.rmsnorm_f32(&stream, rows, cols, ps as *const f32, pa as *const f32, pd as *mut f32, eps)
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
        let (ctx, stream, mut k) = setup();
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
        let mut dc = stream.alloc_zeros::<u16>(n).unwrap();
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

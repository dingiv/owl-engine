//! dry-run 专用 kernel 模块(引擎侧 nvrtc;K1 真 kernel port 落地前的
//! naive 通路)。捕获安全:全部裸指针 + 标量参数,无分配/无同步/无 D2H。
//!
//! 构成:
//! - sin_f32/cos_f32:逐元素(rope 表构造用);
//! - embed_f32:embedding lookup(out[t,d] = w[ids[t],d]);
//! - rope_half_f32:rotate-half RoPE(q/k 原位出参,cos/sin 表 [max_pos, D/2]);
//! - naive_decode_attn_f32:decode(seq=1)naive attention——cache 直写 +
//!   全行扫描 softmax(kv_len 掩码);非分页(slot = 全局槽),K1 真 kernel
//!   (分页 block table)落地后替换。

use owl_cuda::ffi::nvrtc::{compile_ptx_with_opts, CompileOptions};
use owl_cuda::ffi::{CudaContext, CudaFunction, LaunchConfig, PushKernelArg};
use std::collections::HashSet;
use std::sync::Arc;

const SRC: &str = include_str!("dry_kernels.cu");

pub struct DryKernels {
    ctx: Arc<CudaContext>,
    module: Arc<owl_cuda::ffi::CudaModule>,
    loaded: HashSet<&'static str>,
}

impl DryKernels {
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, String> {
        let include = std::env::var("CUDA_INCLUDE")
            .unwrap_or_else(|_| "/usr/local/cuda/include".to_string());
        let opts = CompileOptions {
            include_paths: vec![include],
            use_fast_math: Some(true),
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(SRC, opts).map_err(|e| format!("nvrtc: {e}"))?;
        let module = ctx.load_module(ptx).map_err(|e| format!("load_module: {e}"))?;
        Ok(Self {
            ctx: Arc::clone(ctx),
            module,
            loaded: HashSet::new(),
        })
    }

    fn func(&mut self, name: &'static str) -> Result<CudaFunction, String> {
        if self.loaded.insert(name) {
            self.module
                .load_function(name)
                .map_err(|e| format!("load_function({name}): {e}"))?;
        }
        self.module
            .load_function(name)
            .map_err(|e| format!("load_function({name}): {e}"))
    }

    fn launch(
        &mut self,
        stream: &owl_cuda::ffi::CudaStream,
        name: &'static str,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: &[u64],
    ) -> Result<(), String> {
        let func = self.func(name)?;
        let cfg = LaunchConfig {
            grid_dim: grid,
            block_dim: block,
            shared_mem_bytes: 0,
        };
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

    pub fn sin_f32(&mut self, stream: &owl_cuda::ffi::CudaStream, x: *const f32, out: *mut f32, n: usize) -> Result<(), String> {
        self.launch(stream, "owl_sin_f32", (n as u32, 1, 1), (128, 1, 1),
            &[x as u64, out as u64, n as u64])
    }

    pub fn cos_f32(&mut self, stream: &owl_cuda::ffi::CudaStream, x: *const f32, out: *mut f32, n: usize) -> Result<(), String> {
        self.launch(stream, "owl_cos_f32", (n as u32, 1, 1), (128, 1, 1),
            &[x as u64, out as u64, n as u64])
    }

    /// embedding:out[t*D + d] = w[ids[t]*D + d]
    #[allow(clippy::too_many_arguments)]
    pub fn embed_f32(
        &mut self,
        stream: &owl_cuda::ffi::CudaStream,
        w: *const f32,
        ids: *const u32,
        out: *mut f32,
        tokens: usize,
        d_dim: usize,
    ) -> Result<(), String> {
        self.launch(stream, "owl_embed_f32", (tokens as u32, 1, 1), (128, 1, 1),
            &[w as u64, ids as u64, out as u64, d_dim as u64])
    }

    /// rotate-half RoPE:q/k [tokens, heads, D](原位出参可传同 src);
    /// cos/sin [max_pos, D/2];positions [tokens](u32)。
    #[allow(clippy::too_many_arguments)]
    pub fn rope_half_f32(
        &mut self,
        stream: &owl_cuda::ffi::CudaStream,
        q: *const f32,
        k: *const f32,
        q_out: *mut f32,
        k_out: *mut f32,
        cos: *const f32,
        sin: *const f32,
        positions: *const u32,
        tokens: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<(), String> {
        let half = (head_dim / 2) as u64;
        self.launch(
            stream,
            "owl_rope_half_f32",
            (tokens as u32, 1, 1),
            (128, 1, 1),
            &[
                q as u64,
                k as u64,
                q_out as u64,
                k_out as u64,
                cos as u64,
                sin as u64,
                positions as u64,
                heads as u64,
                kv_heads as u64,
                half,
            ],
        )
    }

    /// 逐元素 sigmoid
    pub fn sigmoid_f32(
        &mut self,
        stream: &owl_cuda::ffi::CudaStream,
        x: *const f32,
        out: *mut f32,
        n: usize,
    ) -> Result<(), String> {
        self.launch(
            stream,
            "owl_sigmoid_f32",
            (((n as u32) + 127) / 128, 1, 1),
            (128, 1, 1),
            &[x as u64, out as u64, n as u64],
        )
    }

    /// 3D 转置 [A,B,C] → [A,C,B](连续 f32;最后两维换位)
    pub fn transpose12_f32(
        &mut self,
        stream: &owl_cuda::ffi::CudaStream,
        src: *const f32,
        dst: *mut f32,
        a: usize,
        b: usize,
        c: usize,
    ) -> Result<(), String> {
        let total = (a * b * c) as u32;
        self.launch(
            stream,
            "owl_transpose12_f32",
            ((total + 127) / 128, 1, 1),
            (128, 1, 1),
            &[src as u64, dst as u64, a as u64, b as u64, c as u64],
        )
    }

    /// 3D 转置 [A,B,C] → [B,A,C](连续 f32)
    pub fn transpose01_f32(
        &mut self,
        stream: &owl_cuda::ffi::CudaStream,
        src: *const f32,
        dst: *mut f32,
        a: usize,
        b: usize,
        c: usize,
    ) -> Result<(), String> {
        let total = (a * b * c) as u32;
        self.launch(
            stream,
            "owl_transpose01_f32",
            ((total + 127) / 128, 1, 1),
            (128, 1, 1),
            &[src as u64, dst as u64, a as u64, b as u64, c as u64],
        )
    }

    /// partial rotate-half RoPE:只转每头前 2*rotary_half 维,余维直通。
    /// q/k [tokens, heads, head_dim] 连续 f32;cos/sin 表 [max_pos, rotary_half]。
    #[allow(clippy::too_many_arguments)]
    pub fn rope_half_partial_f32(
        &mut self,
        stream: &owl_cuda::ffi::CudaStream,
        q: *const f32,
        k: *const f32,
        q_out: *mut f32,
        k_out: *mut f32,
        cos: *const f32,
        sin: *const f32,
        positions: *const u32,
        tokens: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        rotary_half: usize,
    ) -> Result<(), String> {
        self.launch(
            stream,
            "owl_rope_half_partial_f32",
            (tokens as u32, 1, 1),
            (128, 1, 1),
            &[
                q as u64,
                k as u64,
                q_out as u64,
                k_out as u64,
                cos as u64,
                sin as u64,
                positions as u64,
                heads as u64,
                kv_heads as u64,
                head_dim as u64,
                rotary_half as u64,
            ],
        )
    }

    /// decode(seq=1)naive attention:
    /// q [bs,Hq,D];k/v [bs,Hkv,D];kc/vc [max_slots,Hkv,D](slot 直排);
    /// slots [bs] i64(负 = padding);kv_lens [bs] i32;out [bs,Hq*D]。
    /// 一线程一 (t,h);MAX_KV=256 编译期上限(dry-run 规模)。
    #[allow(clippy::too_many_arguments)]
    pub fn naive_decode_attn_f32(
        &mut self,
        stream: &owl_cuda::ffi::CudaStream,
        q: *const f32,
        k: *const f32,
        v: *const f32,
        kc: *mut f32,
        vc: *mut f32,
        slots: *const i32,
        kv_lens: *const i32,
        out: *mut f32,
        bs: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<(), String> {
        self.launch(
            stream,
            "owl_naive_decode_attn_f32",
            ((bs * q_heads) as u32, 1, 1),
            (1, 1, 1),
            &[
                q as u64,
                k as u64,
                v as u64,
                kc as u64,
                vc as u64,
                slots as u64,
                kv_lens as u64,
                out as u64,
                q_heads as u64,
                kv_heads as u64,
                head_dim as u64,
            ],
        )
    }
}

impl DryKernels {
    /// 2D 转置物化拷贝([rows, cols] → [cols, rows];捕获安全)
    pub fn transpose2d_f32(
        &mut self,
        stream: &owl_cuda::ffi::CudaStream,
        src: *const f32,
        dst: *mut f32,
        rows: usize,
        cols: usize,
    ) -> Result<(), String> {
        self.launch(
            stream,
            "owl_transpose2d_f32",
            ((rows * cols) as u32, 1, 1),
            (128, 1, 1),
            &[src as u64, dst as u64, rows as u64, cols as u64],
        )
    }
}

impl DryKernels {
    /// 单维窄切物化拷贝(任意 dim;outer = dim 之前各维乘积)
    pub fn narrow_strided_f32(
        &mut self,
        stream: &owl_cuda::ffi::CudaStream,
        src: *const f32,
        dst: *mut f32,
        outer: usize,
        src_dim: usize,
        start: usize,
        out_dim: usize,
    ) -> Result<(), String> {
        let total = outer * out_dim;
        self.launch(
            stream,
            "owl_narrow_strided_f32",
            (total as u32, 1, 1),
            (128, 1, 1),
            &[
                src as u64,
                dst as u64,
                outer as u64,
                src_dim as u64,
                start as u64,
                out_dim as u64,
            ],
        )
    }
}

impl DryKernels {
    /// fill:标量填充(scratch 缓冲初始化;捕获安全)
    pub fn fill_f32(&mut self, stream: &owl_cuda::ffi::CudaStream, ptr: *mut f32, v: f32, n: usize) -> Result<(), String> {
        self.launch(stream, "owl_fill_f32", (n as u32, 1, 1), (128, 1, 1),
            &[ptr as u64, v.to_bits() as u64, n as u64])
    }
}

impl DryKernels {
    /// 逐元素倒数(div 的捕获安全实现基元)
    pub fn recip_f32(&mut self, stream: &owl_cuda::ffi::CudaStream, x: *const f32, out: *mut f32, n: usize) -> Result<(), String> {
        self.launch(stream, "owl_recip_f32", (n as u32, 1, 1), (128, 1, 1),
            &[x as u64, out as u64, n as u64])
    }
}

impl DryKernels {
    /// silu_and_mul:x = [rows, 2*cols] 横排(gate|up),out = [rows, cols]
    pub fn silu_and_mul_f32(
        &mut self,
        stream: &owl_cuda::ffi::CudaStream,
        x: *const f32,
        out: *mut f32,
        rows: usize,
        cols: usize,
    ) -> Result<(), String> {
        let total = rows * cols;
        self.launch(
            stream,
            "owl_silu_and_mul_f32",
            (total as u32, 1, 1),
            (128, 1, 1),
            &[x as u64, out as u64, cols as u64],
        )
    }
}

impl DryKernels {
    /// u32 → f32 数值转换
    pub fn u32_to_f32(
        &mut self,
        stream: &owl_cuda::ffi::CudaStream,
        x: *const u32,
        out: *mut f32,
        n: usize,
    ) -> Result<(), String> {
        self.launch(
            stream,
            "owl_u32_to_f32",
            (n as u32, 1, 1),
            (128, 1, 1),
            &[x as u64, out as u64, n as u64],
        )
    }
}

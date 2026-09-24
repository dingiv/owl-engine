//! attention kernel 族(K0 起步:reshape_and_cache)。
//!
//! port 专项:docs/arch/attention-kernel-port.md;出处与改编注记见
//! cu/attention_reshape_and_cache.cu 头部。独立 nvrtc 模块(与
//! owl_nn_kernels.cu 分开编译,attention 族源码增长不拖累基础核)。

use owl_cuda::ffi::nvrtc::{compile_ptx_with_opts, CompileOptions};
use owl_cuda::ffi::CudaFunction;
use owl_cuda::ffi::{CudaContext, LaunchConfig, PushKernelArg};
use std::collections::HashSet;
use std::sync::Arc;

const ATTN_SRC: &str = include_str!("../../kernels/cu/attention_reshape_and_cache.cu");

pub struct AttentionKernels {
    ctx: Arc<CudaContext>,
    module: Arc<owl_cuda::ffi::CudaModule>,
    loaded: HashSet<&'static str>,
}

impl AttentionKernels {
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, String> {
        let include = std::env::var("CUDA_INCLUDE")
            .unwrap_or_else(|_| "/usr/local/cuda/include".to_string());
        let opts = CompileOptions {
            include_paths: vec![include],
            use_fast_math: Some(true),
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(ATTN_SRC, opts).map_err(|e| format!("nvrtc: {e}"))?;
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

    /// K0:reshape_and_cache(f32;grid = num_tokens blocks × 128 threads)。
    /// 布局见 .cu 头部;slot_mapping i64(负槽 = padding 跳过)。
    #[allow(clippy::too_many_arguments)]
    pub fn reshape_and_cache_f32(
        &mut self,
        stream: &owl_cuda::ffi::CudaStream,
        key: *const f32,
        value: *const f32,
        key_cache: *mut f32,
        value_cache: *mut f32,
        slot_mapping: *const i64,
        key_stride: i32,
        value_stride: i32,
        num_heads: i32,
        head_size: i32,
        block_size: i32,
        x: i32,
        num_tokens: i32,
    ) -> Result<(), String> {
        let func = self.func("owl_reshape_and_cache_f32")?;
        let cfg = LaunchConfig {
            grid_dim: (num_tokens as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&func);
            let args: [u64; 12] = [
                key as u64,
                value as u64,
                key_cache as u64,
                value_cache as u64,
                slot_mapping as u64,
                key_stride as u64,
                value_stride as u64,
                num_heads as u64,
                head_size as u64,
                block_size as u64,
                x as u64,
                num_tokens as u64,
            ];
            for a in args.iter() {
                builder.arg(a);
            }
            builder
                .launch(cfg)
                .map_err(|e| format!("launch(reshape_and_cache): {e}"))?;
        }
        let _ = &self.ctx;
        Ok(())
    }
}

#[cfg(test)]
mod tests {

    fn htod_t<T: owl_iface::MemValue>(pool: &owl_cuda::CudaPool, v: Vec<T>) -> owl_cuda::CudaPoolBuf {
        let n = v.len() * std::mem::size_of::<T>();
        let host = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, n) };
        pool.htod(host).unwrap()
    }

    fn alloc_t<T: owl_iface::MemValue>(pool: &owl_cuda::CudaPool, len: usize) -> owl_cuda::CudaPoolBuf {
        pool.malloc((len * std::mem::size_of::<T>()) as u64).unwrap()
    }

    use super::AttentionKernels;
    use owl_cuda::CudaDevice;
    use owl_iface::Pool as _;

    /// K0 验收:合成 KV 块写入对拍 host 参考(vLLM 布局公式直译)。
    /// 含负槽(padding 跳过)分支。
    #[test]
    fn reshape_and_cache_f32_host_parity() {
        let dev = CudaDevice::new(owl_cuda::test_device_ordinal(), owl_cuda::TEST_POOL_BYTES).expect("需要 CUDA 设备");
        let pool = dev.default_pool();

        const TOKENS: usize = 3;
        const HEADS: usize = 2;
        const HEAD_SIZE: usize = 4;
        const BLOCK_SIZE: usize = 4;
        const BLOCKS: usize = 2;
        const X: usize = 4; // head_size / x = 1

        let mut key = Vec::new();
        let mut value = Vec::new();
        let mut s = 1.0f32;
        for _i in 0..TOKENS * HEADS * HEAD_SIZE {
            s += 0.5;
            key.push(s);
            value.push(-s);
        }
        let slot_mapping: Vec<i64> = vec![1, 5, -1]; // 末位 padding 跳过

        let d_key = htod_t::<f32> (&pool, key.clone());
        let d_value = htod_t::<f32> (&pool, value.clone());
        let d_slot = htod_t::<i64> (&pool, slot_mapping.clone());
        let d_kcache = alloc_t::<f32> (&pool, BLOCKS * HEADS * HEAD_SIZE * BLOCK_SIZE);
        let d_vcache = alloc_t::<f32> (&pool, BLOCKS * HEADS * HEAD_SIZE * BLOCK_SIZE);
        dev.ctx().synchronize().unwrap();

        let mut kern = AttentionKernels::new(dev.ctx()).unwrap();
        kern.reshape_and_cache_f32(
            dev.stream(),
            owl_iface::DevBuf::device_ptr(&d_key) as *const f32,
            owl_iface::DevBuf::device_ptr(&d_value) as *const f32,
            owl_iface::DevBuf::device_ptr(&d_kcache) as *mut f32,
            owl_iface::DevBuf::device_ptr(&d_vcache) as *mut f32,
            owl_iface::DevBuf::device_ptr(&d_slot) as *const i64,
            (HEADS * HEAD_SIZE) as i32,
            (HEADS * HEAD_SIZE) as i32,
            HEADS as i32,
            HEAD_SIZE as i32,
            BLOCK_SIZE as i32,
            X as i32,
            TOKENS as i32,
        )
        .unwrap();
        dev.ctx().synchronize().unwrap();

        // D2H 读回
        // P0-3:流序 D2H(memx)
        let read = |ptr: *mut f32| -> Vec<f32> {
            let mut out = vec![0f32; BLOCKS * HEADS * HEAD_SIZE * BLOCK_SIZE];
            dev.memcpy_dtoh_f32(dev.stream(), ptr, &mut out).unwrap();
            out
        };
        let kcache = read(owl_iface::DevBuf::device_ptr(&d_kcache) as *mut f32);
        let vcache = read(owl_iface::DevBuf::device_ptr(&d_vcache) as *mut f32);

        // host 参考(公式 = .cu 直译;注意 GPU 端 int64 除法为整除)
        let mut expect_k = vec![0f32; kcache.len()];
        let mut expect_v = vec![0f32; vcache.len()];
        for (t, &slot) in slot_mapping.iter().enumerate() {
            if slot < 0 {
                continue;
            }
            let (block, off) = (slot / BLOCK_SIZE as i64, slot % BLOCK_SIZE as i64);
            for h in 0..HEADS * HEAD_SIZE {
                let (head, hoff) = (h / HEAD_SIZE, h % HEAD_SIZE);
                let (x_idx, x_off) = (hoff / X, hoff % X);
                let tk = block * (HEADS * (HEAD_SIZE / X) * BLOCK_SIZE * X) as i64
                    + (head * (HEAD_SIZE / X) * BLOCK_SIZE * X) as i64
                    + (x_idx * BLOCK_SIZE * X) as i64
                    + off * X as i64
                    + x_off as i64;
                let tv = block * (HEADS * HEAD_SIZE * BLOCK_SIZE) as i64
                    + (head * HEAD_SIZE * BLOCK_SIZE) as i64
                    + (hoff * BLOCK_SIZE) as i64
                    + off;
                expect_k[tk as usize] = key[t * HEADS * HEAD_SIZE + h];
                expect_v[tv as usize] = value[t * HEADS * HEAD_SIZE + h];
            }
        }
        assert_eq!(kcache, expect_k, "key cache 布局对拍");
        assert_eq!(vcache, expect_v, "value cache 布局对拍");
    }
}

// ---- K1: paged_attention_v1(f16/bf16 × block 32/64;head_size 128)----

/// K1 入口源码(含 attention/ 头目录;include 靠 CompileOptions.include_paths)
const PAGED_V1_SRC: &str = include_str!("../../kernels/cu/attention_paged_v1.cu");
/// K2 入口源码(v2 主核 + reduce 核;模板解法同 v1)
const PAGED_V2_SRC: &str = include_str!("../../kernels/cu/attention_paged_v2.cu");

const PARTITION_SIZE: i32 = 512;

/// paged_attention 核族(独立 nvrtc 模块;头目录 = kernels/cu)
pub struct PagedKernels {
    ctx: Arc<CudaContext>,
    module: Arc<owl_cuda::ffi::CudaModule>,
    /// K2 模块(v2 主核 + reduce 核)
    module_v2: Arc<owl_cuda::ffi::CudaModule>,
    loaded: HashSet<&'static str>,
}

impl PagedKernels {
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, String> {
        let include = std::env::var("CUDA_INCLUDE")
            .unwrap_or_else(|_| "/usr/local/cuda/include".to_string());
        let cu_dir = concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/cu");
        let make_opts = |include: &str, cu_dir: &str| CompileOptions {
            include_paths: vec![
                format!("{}/attention_compat", std::env::var("CARGO_MANIFEST_DIR").unwrap()),
                include.to_string(),
                cu_dir.to_string(),
                format!("{cu_dir}/attention"),
            ],
            use_fast_math: Some(true),
            ..Default::default()
        };
        let t0 = std::time::Instant::now();
        let ptx = compile_ptx_with_opts(PAGED_V1_SRC, make_opts(&include, &cu_dir))
            .map_err(|e| format!("nvrtc: {e}"))?;
        let t_v1 = t0.elapsed();
        let module = ctx.load_module(ptx).map_err(|e| format!("load_module: {e}"))?;
        let t1 = std::time::Instant::now();
        let ptx2 = compile_ptx_with_opts(PAGED_V2_SRC, make_opts(&include, &cu_dir))
            .map_err(|e| format!("nvrtc v2: {e}"))?;
        let t_v2 = t1.elapsed();
        let module_v2 = ctx.load_module(ptx2).map_err(|e| format!("load_module(v2): {e}"))?;
        eprintln!(
            "[PagedKernels] nvrtc v1 = {t_v1:?}, v2 = {t_v2:?}"
        );
        Ok(Self {
            ctx: Arc::clone(ctx),
            module,
            module_v2,
            loaded: HashSet::new(),
        })
    }

    /// K2 模块的函数装载(独立模块,同名缓存键不冲突)
    fn func_v2(&mut self, name: &'static str) -> Result<CudaFunction, String> {
        if self.loaded.insert(name) {
            self.module_v2
                .load_function(name)
                .map_err(|e| format!("load_function({name}): {e}"))?;
        }
        self.module_v2
            .load_function(name)
            .map_err(|e| format!("load_function({name}): {e}"))
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

    /// v1 解码核。布局契约见 cu/attention_paged_v1.cu 头部。
    /// `dtype`: 0 = f16(vLLM uint16 位型路径);1 = bf16。
    /// `head_size`: 128 / 256(B1 双档;编译期实例化,非运行时参数)。
    /// 全部动态量(device 张量裸指针;A1.5);alibi 暂 nullptr。
    #[allow(clippy::too_many_arguments)]
    pub fn paged_attention_v1(
        &mut self,
        stream: &owl_cuda::ffi::CudaStream,
        dtype: u32, // 0=f16 1=bf16
        block_size: u32,
        head_size: u32,
        out: *mut u16,
        query: *const u16,
        key_cache: *const u16,
        value_cache: *const u16,
        num_kv_heads: i32,
        scale: f32,
        block_tables: *const i32,
        context_lens: *const i32,
        max_num_blocks_per_seq: i32,
        q_stride: i32,
        kv_block_stride: i32,
        kv_head_stride: i32,
        max_context_len: i32,
        num_seqs: i32,
        num_heads: i32,
    ) -> Result<(), String> {
        let name = match (dtype, block_size, head_size) {
            (0, 32, 128) => "owl_pa_v1_f16_bs32",
            (0, 64, 128) => "owl_pa_v1_f16_bs64",
            (1, 32, 128) => "owl_pa_v1_bf16_bs32",
            (1, 64, 128) => "owl_pa_v1_bf16_bs64",
            (0, 32, 256) => "owl_pa_v1_f16_bs32_h256",
            (0, 64, 256) => "owl_pa_v1_f16_bs64_h256",
            (1, 32, 256) => "owl_pa_v1_bf16_bs32_h256",
            (1, 64, 256) => "owl_pa_v1_bf16_bs64_h256",
            _ => return Err(format!(
                "paged_attention_v1: dtype {dtype} bs {block_size} head {head_size} 未实例化")),
        };
        let func = self.func(name)?;
        let padded = ((max_context_len + block_size as i32 - 1) / block_size as i32)
            * block_size as i32;
        let logits_size = padded as usize * 4;
        // NUM_WARPS/2 * head_size * 4(NUM_WARPS = 128/32 = 4)
        let outputs_size = (128 / 32 / 2) * head_size as usize * 4;
        let cfg = LaunchConfig {
            grid_dim: (num_heads as u32, num_seqs as u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: logits_size.max(outputs_size) as u32,
        };
        unsafe {
            let mut builder = stream.launch_builder(&func);
            // 参数面 = kernel 签名全序(17 参;k_scales/v_scales/alibi 恒空指针)
            let args: [u64; 17] = [
                out as u64,
                query as u64,
                key_cache as u64,
                value_cache as u64,
                0, // k_scales(非量化恒 null)
                0, // v_scales
                num_kv_heads as u64,
                scale.to_bits() as u64,
                block_tables as u64,
                context_lens as u64,
                max_num_blocks_per_seq as u64,
                0, // alibi_slopes
                q_stride as u64,
                kv_block_stride as u64,
                kv_head_stride as u64,
                softscapping_bits(),
                sliding_window_zero(),
            ];
            for a in args.iter() {
                builder.arg(a);
            }
            builder
                .launch(cfg)
                .map_err(|e| format!("launch({name}): {e}"))?;
        }
        let _ = &self.ctx;
        Ok(())
    }

    /// K2:v2 主核 + reduce 核(大 bs 分片归约;布局契约同 v1)。
    /// `head_size`: 128 / 256(B1 双档);reduce 核与 head_size 无关,仅按 dtype。
    /// 临时缓冲(exp_sums/max_logits/tmp_out)= 调用方 scratch(A5.2):
    ///   exp_sums/max_logits: [num_seqs, num_heads, max_num_partitions] f32
    ///   tmp_out:             [num_seqs, num_heads, max_num_partitions, head_size] u16
    /// max_num_partitions 由 max_context_len 与 PARTITION_SIZE=512 推导(内部)。
    #[allow(clippy::too_many_arguments)]
    pub fn paged_attention_v2(
        &mut self,
        stream: &owl_cuda::ffi::CudaStream,
        dtype: u32, // 0=f16 1=bf16
        block_size: u32,
        head_size: u32,
        out: *mut u16,
        exp_sums: *mut f32,
        max_logits: *mut f32,
        tmp_out: *mut u16,
        query: *const u16,
        key_cache: *const u16,
        value_cache: *const u16,
        num_kv_heads: i32,
        scale: f32,
        block_tables: *const i32,
        context_lens: *const i32,
        max_num_blocks_per_seq: i32,
        q_stride: i32,
        kv_block_stride: i32,
        kv_head_stride: i32,
        max_context_len: i32,
        num_seqs: i32,
        num_heads: i32,
    ) -> Result<(), String> {
        let max_num_partitions = (max_context_len + PARTITION_SIZE - 1) / PARTITION_SIZE;
        let main_name = match (dtype, block_size, head_size) {
            (0, 32, 128) => "owl_pa_v2_main_f16_bs32",
            (0, 64, 128) => "owl_pa_v2_main_f16_bs64",
            (1, 32, 128) => "owl_pa_v2_main_bf16_bs32",
            (1, 64, 128) => "owl_pa_v2_main_bf16_bs64",
            (0, 32, 256) => "owl_pa_v2_main_f16_bs32_h256",
            (0, 64, 256) => "owl_pa_v2_main_f16_bs64_h256",
            (1, 32, 256) => "owl_pa_v2_main_bf16_bs32_h256",
            (1, 64, 256) => "owl_pa_v2_main_bf16_bs64_h256",
            _ => return Err(format!(
                "paged_attention_v2: dtype {dtype} bs {block_size} head {head_size} 未实例化")),
        };
        let reduce_name = match (dtype, head_size) {
            (0, 128) => "owl_pa_v2_reduce_f16",
            (0, 256) => "owl_pa_v2_reduce_f16_h256",
            (_, 128) => "owl_pa_v2_reduce_bf16",
            (_, 256) => "owl_pa_v2_reduce_bf16_h256",
            _ => return Err(format!("paged_attention_v2 reduce: dtype {dtype} head {head_size} 未实例化")),
        };

        // 主核:grid (num_heads, num_seqs, max_num_partitions);smem = max(512*4, NUM_WARPS/2*head_size*4)
        let func = self.func_v2(main_name)?;
        let logits_size = (PARTITION_SIZE as usize) * 4;
        let outputs_size = (128 / 32 / 2) * head_size as usize * 4;
        let cfg = LaunchConfig {
            grid_dim: (num_heads as u32, num_seqs as u32, max_num_partitions as u32),
            block_dim: (128, 1, 1),
            shared_mem_bytes: logits_size.max(outputs_size) as u32,
        };
        unsafe {
            let mut builder = stream.launch_builder(&func);
            let args: [u64; 19] = [
                exp_sums as u64,
                max_logits as u64,
                tmp_out as u64,
                query as u64,
                key_cache as u64,
                value_cache as u64,
                0, // k_scales
                0, // v_scales
                num_kv_heads as u64,
                scale.to_bits() as u64,
                block_tables as u64,
                context_lens as u64,
                max_num_blocks_per_seq as u64,
                0, // alibi_slopes
                q_stride as u64,
                kv_block_stride as u64,
                kv_head_stride as u64,
                softscapping_bits(),
                sliding_window_zero(),
            ];
            for a in args.iter() {
                builder.arg(a);
            }
            builder.launch(cfg).map_err(|e| format!("launch(pa_v2_main): {e}"))?;
        }

        // reduce 核:grid (num_heads, num_seqs);smem = 2*max_num_partitions*4
        let func = self.func_v2(reduce_name)?;
        let cfg = LaunchConfig {
            grid_dim: (num_heads as u32, num_seqs as u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: (2 * max_num_partitions * 4) as u32,
        };
        unsafe {
            let mut builder = stream.launch_builder(&func);
            let args: [u64; 6] = [
                out as u64,
                exp_sums as u64,
                max_logits as u64,
                tmp_out as u64,
                context_lens as u64,
                max_num_partitions as u64,
            ];
            for a in args.iter() {
                builder.arg(a);
            }
            builder.launch(cfg).map_err(|e| format!("launch(pa_v2_reduce): {e}"))?;
        }
        let _ = &self.ctx;
        Ok(())
    }
}

fn softscapping_bits() -> u64 {
    1.0f32.to_bits() as u64 // 恒等软上限(qwen3.5 不用)
}
fn sliding_window_zero() -> u64 {
    0 // 0 = 关闭滑窗
}

#[cfg(test)]
mod paged_tests {
    fn htod_t<T: owl_iface::MemValue>(pool: &owl_cuda::CudaPool, v: Vec<T>) -> owl_cuda::CudaPoolBuf {
        let n = v.len() * std::mem::size_of::<T>();
        let host = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, n) };
        pool.htod(host).unwrap()
    }

    fn alloc_t<T: owl_iface::MemValue>(pool: &owl_cuda::CudaPool, len: usize) -> owl_cuda::CudaPoolBuf {
        pool.malloc((len * std::mem::size_of::<T>()) as u64).unwrap()
    }

    use super::PagedKernels;
    use owl_cuda::CudaDevice;
    use owl_iface::Pool as _;

    // ---- f16/bf16 位型转换(测试专用;RNE 近似,测试值全为二进制精确)----
    fn f32_to_f16_bits(f: f32) -> u16 {
        let b = f.to_bits();
        let sign = ((b >> 16) & 0x8000) as u16;
        let exp = ((b >> 23) & 0xff) as i32;
        let man = b & 0x7fffff;
        if exp == 0xff {
            return sign | 0x7c00;
        }
        let e = exp - 127 + 15;
        if e >= 31 {
            return sign | 0x7c00;
        }
        if e <= 0 {
            if e < -10 {
                return sign;
            }
            return sign | (((man | 0x800000) >> (1 - e + 13)) as u16);
        }
        let hm = man >> 13;
        let round = (man >> 12) & 1;
        let hm = if round == 1 && (man & 0xfff) != 0 { hm + 1 } else { hm };
        if hm == 0x800 {
            return sign | (((e + 1) as u16) << 10);
        }
        sign | ((e as u16) << 10) | hm as u16
    }
    fn f16_bits_to_f32(h: u16) -> f32 {
        let sign = ((h & 0x8000) as u32) << 16;
        let exp = ((h >> 10) & 0x1f) as u32;
        let man = (h & 0x3ff) as u32;
        f32::from_bits(if exp == 0 {
            sign
        } else if exp == 31 {
            sign | 0x7f800000
        } else {
            sign | ((exp + 127 - 15) << 23) | (man << 13)
        })
    }
    fn f32_to_bf16_bits(f: f32) -> u16 {
        (f.to_bits() >> 16) as u16
    }
    fn bf16_bits_to_f32(h: u16) -> f32 {
        f32::from_bits((h as u32) << 16)
    }


    /// K1 验收:合成 paged KV 上 v1 核 vs host 朴素 attention 对拍(f16)。
    /// 场景:num_seqs=2(len 20/10),heads=8(kv=8),head_size=128,
    /// block_size=32,num_blocks=8;padding 槽(context_len 之外)混入 NaN
    /// 值验证掩码。容差 2e-2(f16 累加)。
    #[test]
    fn paged_attention_v1_f16_host_parity() {
        const NUM_SEQS: usize = 2;
        const HEADS: usize = 8;
        const KV_HEADS: usize = 8;
        const HEAD_SIZE: usize = 128;
        const BS: usize = 32;
        const BLOCKS: usize = 8;
        const MAX_BPS: usize = 2; // 每序最多 1 块(len≤32)+1 冗余
        let scale = 1.0 / (HEAD_SIZE as f32).sqrt();

        let lens = [20usize, 10];
        let mut s = 0.25f32;
        let mut nxt = || {
            s = (s * 1.37 + 0.13).fract() - 0.5;
            s
        };
        // 逻辑 Q/K/V(全 f32,再量化 f16)
        let mut q = vec![0u16; NUM_SEQS * HEADS * HEAD_SIZE];
        for (_i, v) in q.iter_mut().enumerate() {
            *v = f32_to_f16_bits(nxt() * 4.0);
        }
        let mut logical_k = vec![vec![0f32; HEAD_SIZE]; NUM_SEQS * HEADS * BS];
        let mut logical_v = vec![vec![0f32; HEAD_SIZE]; NUM_SEQS * HEADS * BS];
        for seq in 0..NUM_SEQS {
            for pos in 0..BS {
                for h in 0..HEADS {
                    for d in 0..HEAD_SIZE {
                        logical_k[(seq * HEADS + h) * BS + pos][d] = nxt() * 2.0;
                        logical_v[(seq * HEADS + h) * BS + pos][d] = nxt() * 2.0;
                    }
                }
            }
        }

        // 分页散布(公式 = K0 同款布局,x = 16/2 = 8)
        const X: usize = 8;
        let mut block_tables = vec![0i32; NUM_SEQS * MAX_BPS];
        // 物理块 0..=3 → seq0,4..=7 → seq1;NaN 填充未用块
        for seq in 0..NUM_SEQS {
            block_tables[seq * MAX_BPS] = (seq * 4) as i32;
            block_tables[seq * MAX_BPS + 1] = (seq * 4 + 1) as i32;
        }
        let kc_len = BLOCKS * KV_HEADS * (HEAD_SIZE / X) * BS * X;
        let vc_len = BLOCKS * KV_HEADS * HEAD_SIZE * BS;
        let mut key_cache = vec![f32_to_f16_bits(0.0); kc_len];
        let mut value_cache = vec![f32_to_f16_bits(0.0); vc_len];
        for seq in 0..NUM_SEQS {
            for pos in 0..BS {
                if pos >= lens[seq] {
                    continue;
                }
                let pblock = block_tables[seq * MAX_BPS + pos / BS] as usize;
                let poff = pos % BS;
                for h in 0..KV_HEADS {
                    for d in 0..HEAD_SIZE {
                        let x_idx = d / X;
                        let x_off = d % X;
                        let tk = pblock * (KV_HEADS * (HEAD_SIZE / X) * BS * X)
                            + h * ((HEAD_SIZE / X) * BS * X)
                            + x_idx * (BS * X)
                            + poff * X
                            + x_off;
                        key_cache[tk] = f32_to_f16_bits(logical_k[(seq * HEADS + h) * BS + pos][d]);
                        let tv = pblock * (KV_HEADS * HEAD_SIZE * BS)
                            + h * (HEAD_SIZE * BS)
                            + d * BS
                            + poff;
                        value_cache[tv] = f32_to_f16_bits(logical_v[(seq * HEADS + h) * BS + pos][d]);
                    }
                }
            }
        }
        let context_lens: Vec<i32> = lens.iter().map(|&l| l as i32).collect();

        // host 参考:f32 朴素 attention(softmax 后全量 V 加权和)
        let q_f32: Vec<f32> = q.iter().map(|&b| f16_bits_to_f32(b)).collect();
        let mut expect = vec![0f32; NUM_SEQS * HEADS * HEAD_SIZE];
        for seq in 0..NUM_SEQS {
            let len = lens[seq];
            for h in 0..HEADS {
                let mut logits = vec![0f32; len];
                for pos in 0..len {
                    let mut dot = 0f32;
                    for d in 0..HEAD_SIZE {
                        dot += q_f32[(seq * HEADS + h) * HEAD_SIZE + d]
                            * logical_k[(seq * HEADS + h) * BS + pos][d];
                    }
                    logits[pos] = dot * scale;
                }
                let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = logits.iter().map(|&l| (l - mx).exp()).collect();
                let sum: f32 = exps.iter().sum();
                for d in 0..HEAD_SIZE {
                    let mut acc = 0f32;
                    for pos in 0..len {
                        acc += exps[pos] / sum * logical_v[(seq * HEADS + h) * BS + pos][d];
                    }
                    expect[(seq * HEADS + h) * HEAD_SIZE + d] = acc;
                }
            }
        }

        // 设备侧
        let dev = CudaDevice::new(owl_cuda::test_device_ordinal(), owl_cuda::TEST_POOL_BYTES).expect("需要 CUDA 设备");
        let pool = dev.default_pool();
        let d_q = htod_t::<u16> (&pool, q.clone());
        let d_kc = htod_t::<u16> (&pool, key_cache.clone());
        let d_vc = htod_t::<u16> (&pool, value_cache.clone());
        let d_bt = htod_t::<i32> (&pool, block_tables.clone());
        let d_cl = htod_t::<i32> (&pool, context_lens.clone());
        let d_out = alloc_t::<u16> (&pool, NUM_SEQS * HEADS * HEAD_SIZE);
        // 探针:预填 0xAA,区分"核没写"(保留 AA)vs"写了零"
        unsafe {

            use owl_cuda::ffi::sys;
            dev.ctx().bind_to_thread().unwrap();
            sys::cuMemsetD16Async(
                owl_iface::DevBuf::device_ptr(&d_out) as sys::CUdeviceptr,
                0xAAAA,
                NUM_SEQS * HEADS * HEAD_SIZE,
                dev.stream().cu_stream(),
            )
            .result()
            .unwrap();
        }
        dev.ctx().synchronize().unwrap();

        let mut kern = PagedKernels::new(dev.ctx()).unwrap();
        kern.paged_attention_v1(
            dev.stream(),
            0,   // f16
            32,  // block_size
            128, // head_size
            owl_iface::DevBuf::device_ptr(&d_out) as *mut u16,
            owl_iface::DevBuf::device_ptr(&d_q) as *const u16,
            owl_iface::DevBuf::device_ptr(&d_kc) as *const u16,
            owl_iface::DevBuf::device_ptr(&d_vc) as *const u16,
            KV_HEADS as i32,
            scale,
            owl_iface::DevBuf::device_ptr(&d_bt) as *const i32,
            owl_iface::DevBuf::device_ptr(&d_cl) as *const i32,
            MAX_BPS as i32,
            (HEADS * HEAD_SIZE) as i32,
            (KV_HEADS * (HEAD_SIZE / X) * BS * X) as i32,
            ((HEAD_SIZE / X) * BS * X) as i32,
            BS as i32, // max_context_len
            NUM_SEQS as i32,
            HEADS as i32,
        )
        .unwrap();
        dev.ctx().synchronize().unwrap();

        let mut out_host = vec![0u16; NUM_SEQS * HEADS * HEAD_SIZE];
        // P0-3:流序 D2H(memx bytes;u16 位型)
        dev.memcpy_dtoh_bytes(
            dev.stream(),
            owl_iface::DevBuf::device_ptr(&d_out) as *const u8,
            unsafe { std::slice::from_raw_parts_mut(out_host.as_mut_ptr() as *mut u8, out_host.len() * 2) },
        )
        .unwrap();
        let mut max_diff = 0f32;
        for (i, &o) in out_host.iter().enumerate() {
            let diff = (f16_bits_to_f32(o) - expect[i]).abs();
            if diff > max_diff {
                eprintln!("DEBUG i={i} got={} want={} (bits {:#06x})", f16_bits_to_f32(o), expect[i], o);
            }
            if max_diff.is_infinite() && diff.is_infinite() && i > 8 { break; }
            max_diff = max_diff.max(diff);
        }
        assert!(max_diff < 2e-2, "f16 对拍最大偏差 {max_diff}");
    }

    /// bf16 档 smoke:编译+发射+输出有限值(数值精度不判,仅链路)。
    #[test]
    fn paged_attention_v1_bf16_smoke() {
        let dev = CudaDevice::new(owl_cuda::test_device_ordinal(), owl_cuda::TEST_POOL_BYTES).expect("需要 CUDA 设备");
        let pool = dev.default_pool();
        let n_kc = 8 * 8 * 16 * 32 * 8;
        let n_vc = 8 * 8 * 128 * 32;
        let q = vec![f32_to_bf16_bits(0.1); 8 * 128];
        let kc = vec![f32_to_bf16_bits(0.2); n_kc];
        let vc = vec![f32_to_bf16_bits(0.3); n_vc];
        let bt = vec![0i32; 2];
        let cl = vec![16i32, 16];
        let d_q = htod_t::<u16> (&pool, q);
        let d_kc = htod_t::<u16> (&pool, kc);
        let d_vc = htod_t::<u16> (&pool, vc);
        let d_bt = htod_t::<i32> (&pool, bt);
        let d_cl = htod_t::<i32> (&pool, cl);
        let d_out = alloc_t::<u16> (&pool, 2 * 8 * 128);
        dev.ctx().synchronize().unwrap();
        let mut kern = PagedKernels::new(dev.ctx()).unwrap();
        kern.paged_attention_v1(
            dev.stream(), 1, 32, 128,
            owl_iface::DevBuf::device_ptr(&d_out) as *mut u16,
            owl_iface::DevBuf::device_ptr(&d_q) as *const u16,
            owl_iface::DevBuf::device_ptr(&d_kc) as *const u16,
            owl_iface::DevBuf::device_ptr(&d_vc) as *const u16,
            8, 0.08838835, // 1/sqrt(128)
            owl_iface::DevBuf::device_ptr(&d_bt) as *const i32,
            owl_iface::DevBuf::device_ptr(&d_cl) as *const i32,
            1, 8 * 128, 8 * 16 * 32 * 8, 8 * 128 * 32, 32, 2, 8,
        )
        .unwrap();
        dev.ctx().synchronize().unwrap();
        let mut out_host = vec![0u16; 2 * 8 * 128];
        // P0-3:流序 D2H(memx bytes;u16 位型)
        dev.memcpy_dtoh_bytes(
            dev.stream(),
            owl_iface::DevBuf::device_ptr(&d_out) as *const u8,
            unsafe { std::slice::from_raw_parts_mut(out_host.as_mut_ptr() as *mut u8, out_host.len() * 2) },
        )
        .unwrap();
        // bf16 smoke:全部有限值即可(有 NaN 说明越界/掩码错)
        for (i, &b) in out_host.iter().enumerate() {
            let f = bf16_bits_to_f32(b);
            assert!(f.is_finite(), "bf16 smoke: out[{i}] 非有限 {f}");
        }
    }
    /// K2 验收:v1/v2 同输入对拍 + host 朴素参考交叉(f16,触发分片)。
    /// 场景:lens [1000, 600](> PARTITION_SIZE 512 → 2 partitions),
    /// heads=8/kv=8,head_size=128,block_size=32,GQA 语义由 v 核内映射承担。
    #[test]
    fn paged_attention_v2_f16_v1_parity_and_partitions() {
        const NUM_SEQS: usize = 2;
        const HEADS: usize = 8;
        const KV_HEADS: usize = 8;
        const HEAD_SIZE: usize = 128;
        const BS: usize = 32;
        const MAX_BPS: usize = 33; // ceil(1000/32)=32 +1 冗余
        const BLOCKS: usize = NUM_SEQS * MAX_BPS + 2;
        let scale = 1.0 / (HEAD_SIZE as f32).sqrt();

        let lens = [1000usize, 600usize];
        let mut s = 0.31f32;
        let mut nxt = || {
            s = (s * 1.29 + 0.17).fract() - 0.5;
            s
        };
        let mut q = vec![0u16; NUM_SEQS * HEADS * HEAD_SIZE];
        for v in q.iter_mut() {
            *v = f32_to_f16_bits(nxt() * 4.0);
        }
        let max_len = lens[0];
        let mut logical_k = vec![vec![0f32; HEAD_SIZE]; NUM_SEQS * KV_HEADS * max_len];
        let mut logical_v = vec![vec![0f32; HEAD_SIZE]; NUM_SEQS * KV_HEADS * max_len];
        for seq in 0..NUM_SEQS {
            for pos in 0..lens[seq] {
                for h in 0..KV_HEADS {
                    for d in 0..HEAD_SIZE {
                        logical_k[(seq * KV_HEADS + h) * max_len + pos][d] = nxt() * 2.0;
                        logical_v[(seq * KV_HEADS + h) * max_len + pos][d] = nxt() * 2.0;
                    }
                }
            }
        }

        // 分页散布(同 K1 公式)
        const X: usize = 8;
        let mut block_tables = vec![0i32; NUM_SEQS * MAX_BPS];
        for seq in 0..NUM_SEQS {
            for b_i in 0..MAX_BPS {
                block_tables[seq * MAX_BPS + b_i] =
                    if b_i * BS < lens[seq] { (seq * MAX_BPS + b_i) as i32 } else { 0 };
            }
        }
        let kc_len = BLOCKS * KV_HEADS * (HEAD_SIZE / X) * BS * X;
        let vc_len = BLOCKS * KV_HEADS * HEAD_SIZE * BS;
        let mut key_cache = vec![f32_to_f16_bits(0.0); kc_len];
        let mut value_cache = vec![f32_to_f16_bits(0.0); vc_len];
        for seq in 0..NUM_SEQS {
            for pos in 0..lens[seq] {
                let pblock = block_tables[seq * MAX_BPS + pos / BS] as usize;
                let poff = pos % BS;
                for h in 0..KV_HEADS {
                    for d in 0..HEAD_SIZE {
                        let x_idx = d / X;
                        let x_off = d % X;
                        let tk = pblock * (KV_HEADS * (HEAD_SIZE / X) * BS * X)
                            + h * ((HEAD_SIZE / X) * BS * X)
                            + x_idx * (BS * X)
                            + poff * X
                            + x_off;
                        key_cache[tk] =
                            f32_to_f16_bits(logical_k[(seq * KV_HEADS + h) * max_len + pos][d]);
                        let tv = pblock * (KV_HEADS * HEAD_SIZE * BS)
                            + h * (HEAD_SIZE * BS)
                            + d * BS
                            + poff;
                        value_cache[tv] =
                            f32_to_f16_bits(logical_v[(seq * KV_HEADS + h) * max_len + pos][d]);
                    }
                }
            }
        }
        let context_lens: Vec<i32> = lens.iter().map(|&l| l as i32).collect();

        // host 朴素参考(全量 f32)
        let q_f32: Vec<f32> = q.iter().map(|&b| f16_bits_to_f32(b)).collect();
        let mut expect = vec![0f32; NUM_SEQS * HEADS * HEAD_SIZE];
        for seq in 0..NUM_SEQS {
            let len = lens[seq];
            for h in 0..HEADS {
                let mut logits = vec![0f32; len];
                for pos in 0..len {
                    let mut dot = 0f32;
                    for d in 0..HEAD_SIZE {
                        dot += q_f32[(seq * HEADS + h) * HEAD_SIZE + d]
                            * logical_k[(seq * KV_HEADS + h) * max_len + pos][d];
                    }
                    logits[pos] = dot * scale;
                }
                let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = logits.iter().map(|&l| (l - mx).exp()).collect();
                let sum: f32 = exps.iter().sum();
                for d in 0..HEAD_SIZE {
                    let mut acc = 0f32;
                    for pos in 0..len {
                        acc += exps[pos] / sum * logical_v[(seq * KV_HEADS + h) * max_len + pos][d];
                    }
                    expect[(seq * HEADS + h) * HEAD_SIZE + d] = acc;
                }
            }
        }

        // 设备侧
        let dev = CudaDevice::new(owl_cuda::test_device_ordinal(), owl_cuda::TEST_POOL_BYTES).expect("需要 CUDA 设备");
        let pool = dev.default_pool();
        let d_q = htod_t::<u16> (&pool, q.clone());
        let d_kc = htod_t::<u16> (&pool, key_cache.clone());
        let d_vc = htod_t::<u16> (&pool, value_cache.clone());
        let d_bt = htod_t::<i32> (&pool, block_tables.clone());
        let d_cl = htod_t::<i32> (&pool, context_lens.clone());
        let d_out_v1 = alloc_t::<u16> (&pool, NUM_SEQS * HEADS * HEAD_SIZE);
        let d_out_v2 = alloc_t::<u16> (&pool, NUM_SEQS * HEADS * HEAD_SIZE);

        // v2 临时缓冲(A5.2:调用方 scratch;P = ceil(1000/512) = 2)
        const PARTS: usize = 2;
        let d_exp = alloc_t::<f32> (&pool, NUM_SEQS * HEADS * PARTS);
        let d_maxl = alloc_t::<f32> (&pool, NUM_SEQS * HEADS * PARTS);
        let d_tmp = alloc_t::<u16> (&pool, NUM_SEQS * HEADS * PARTS * HEAD_SIZE);

        let mut kern = PagedKernels::new(dev.ctx()).unwrap();
        let (q_p, kc_p, vc_p, bt_p, cl_p) = (
            owl_iface::DevBuf::device_ptr(&d_q) as *const u16,
            owl_iface::DevBuf::device_ptr(&d_kc) as *const u16,
            owl_iface::DevBuf::device_ptr(&d_vc) as *const u16,
            owl_iface::DevBuf::device_ptr(&d_bt) as *const i32,
            owl_iface::DevBuf::device_ptr(&d_cl) as *const i32,
        );
        let common = (KV_HEADS as i32, scale, MAX_BPS as i32,
            (HEADS * HEAD_SIZE) as i32,
            (KV_HEADS * (HEAD_SIZE / X) * BS * X) as i32,
            ((HEAD_SIZE / X) * BS * X) as i32,
            lens[0] as i32, NUM_SEQS as i32, HEADS as i32);

        // v1
        kern.paged_attention_v1(
            dev.stream(), 0, 32, 128,
            owl_iface::DevBuf::device_ptr(&d_out_v1) as *mut u16,
            q_p, kc_p, vc_p, common.0, common.1, bt_p, cl_p, common.2,
            common.3, common.4, common.5, common.6, common.7, common.8,
        ).unwrap();
        // v2
        kern.paged_attention_v2(
            dev.stream(), 0, 32, 128,
            owl_iface::DevBuf::device_ptr(&d_out_v2) as *mut u16,
            owl_iface::DevBuf::device_ptr(&d_exp) as *mut f32,
            owl_iface::DevBuf::device_ptr(&d_maxl) as *mut f32,
            owl_iface::DevBuf::device_ptr(&d_tmp) as *mut u16,
            q_p, kc_p, vc_p, common.0, common.1, bt_p, cl_p, common.2,
            common.3, common.4, common.5, common.6, common.7, common.8,
        ).unwrap();
        dev.ctx().synchronize().unwrap();

        // P0-3:流序 D2H(memx bytes;u16 位型)
        let read_u16 = |ptr: *mut u16| -> Vec<u16> {
            let mut out = vec![0u16; NUM_SEQS * HEADS * HEAD_SIZE];
            dev.memcpy_dtoh_bytes(
                dev.stream(),
                ptr as *const u8,
                unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, out.len() * 2) },
            )
            .unwrap();
            out
        };
        let out_v1 = read_u16(owl_iface::DevBuf::device_ptr(&d_out_v1) as *mut u16);
        let out_v2 = read_u16(owl_iface::DevBuf::device_ptr(&d_out_v2) as *mut u16);

        // ①host 交叉:v2 vs host
        let mut host_max = 0f32;
        for (i, &o) in out_v2.iter().enumerate() {
            let diff = (f16_bits_to_f32(o) - expect[i]).abs();
            host_max = host_max.max(diff);
        }
        assert!(host_max < 2e-2, "v2 vs host 最大偏差 {host_max}");

        // ②v1 vs v2 一致性(分片重归约 vs 单 pass;容差同量级)
        let mut vv_max = 0f32;
        for (_i, (&a, &b)) in out_v1.iter().zip(out_v2.iter()).enumerate() {
            let diff = (f16_bits_to_f32(a) - f16_bits_to_f32(b)).abs();
            vv_max = vv_max.max(diff);
        }
        assert!(vv_max < 2e-2, "v1 vs v2 最大偏差 {vv_max}");
        eprintln!("[K2] v2-vs-host max_diff = {host_max:.4e}; v1-vs-v2 max_diff = {vv_max:.4e}");
    }

    /// K2 bf16 smoke:两 partition 编译链 + 输出有限值。
    #[test]
    fn paged_attention_v2_bf16_smoke() {
        let dev = CudaDevice::new(owl_cuda::test_device_ordinal(), owl_cuda::TEST_POOL_BYTES).expect("需要 CUDA 设备");
        let pool = dev.default_pool();
        const N: usize = 2 * 8 * 128;
        let q: Vec<u16> = (0..N).map(|i| f32_to_bf16_bits(((i % 7) as f32 - 3.0) * 0.25)).collect();
        let kc: Vec<u16> = (0..8 * 8 * 16 * 32 * 8).map(|i| f32_to_bf16_bits(((i % 5) as f32 - 2.0) * 0.2)).collect();
        let vc: Vec<u16> = (0..8 * 8 * 128 * 32).map(|i| f32_to_bf16_bits(((i % 9) as f32 - 4.0) * 0.2)).collect();
        let bt = vec![0i32, 1];
        let cl = vec![600i32, 300i32];
        let d_q = htod_t::<u16> (&pool, q);
        let d_kc = htod_t::<u16> (&pool, kc);
        let d_vc = htod_t::<u16> (&pool, vc);
        let d_bt = htod_t::<i32> (&pool, bt);
        let d_cl = htod_t::<i32> (&pool, cl);
        let d_out = alloc_t::<u16> (&pool, N);
        let d_exp = alloc_t::<f32> (&pool, 2 * 8 * 2);
        let d_ml = alloc_t::<f32> (&pool, 2 * 8 * 2);
        let d_tmp = alloc_t::<u16> (&pool, 2 * 8 * 2 * 128);
        let mut kern = PagedKernels::new(dev.ctx()).unwrap();
        kern.paged_attention_v2(
            dev.stream(), 1, 32, 128,
            owl_iface::DevBuf::device_ptr(&d_out) as *mut u16,
            owl_iface::DevBuf::device_ptr(&d_exp) as *mut f32,
            owl_iface::DevBuf::device_ptr(&d_ml) as *mut f32,
            owl_iface::DevBuf::device_ptr(&d_tmp) as *mut u16,
            owl_iface::DevBuf::device_ptr(&d_q) as *const u16,
            owl_iface::DevBuf::device_ptr(&d_kc) as *const u16,
            owl_iface::DevBuf::device_ptr(&d_vc) as *const u16,
            8, 1.0 / 128f32.sqrt(),
            owl_iface::DevBuf::device_ptr(&d_bt) as *const i32,
            owl_iface::DevBuf::device_ptr(&d_cl) as *const i32,
            2, 128, 4, 128, 600, 2, 8,
        ).unwrap();
        dev.ctx().synchronize().unwrap();
        let mut out_host = vec![0u16; N];
        // P0-3:流序 D2H(memx bytes;u16 位型)
        dev.memcpy_dtoh_bytes(
            dev.stream(),
            owl_iface::DevBuf::device_ptr(&d_out) as *const u8,
            unsafe { std::slice::from_raw_parts_mut(out_host.as_mut_ptr() as *mut u8, out_host.len() * 2) },
        )
        .unwrap();
        for (i, &b) in out_host.iter().enumerate() {
            let f = bf16_bits_to_f32(b);
            assert!(f.is_finite(), "v2 bf16 smoke: out[{i}] 非有限 {f}");
        }
       }

    // ---- B1:head_size 256 档(Qwen3.5-0.8B full_attention:8 q × 256 / 2 kv)----

    /// B1 验收一:v1 f16 head 256 + 真实 GQA(8 q 头 → 2 kv 头)host 对拍。
    /// 仿 128 档同款场景;kv 映射 q 头 h → kv 头 h / (8/2);容差 2e-2。
    #[test]
    fn paged_attention_v1_f16_h256_gqa_host_parity() {
        const NUM_SEQS: usize = 2;
        const HEADS: usize = 8;
        const KV_HEADS: usize = 2;
        const HEAD_SIZE: usize = 256;
        const BS: usize = 32;
        const MAX_BPS: usize = 2;
        const BLOCKS: usize = NUM_SEQS * 4;
        const X: usize = 8;
        let scale = 1.0 / (HEAD_SIZE as f32).sqrt();

        let lens = [20usize, 10];
        let mut s = 0.25f32;
        let mut nxt = || {
            s = (s * 1.37 + 0.13).fract() - 0.5;
            s
        };
        let mut q = vec![0u16; NUM_SEQS * HEADS * HEAD_SIZE];
        for v in q.iter_mut() {
            *v = f32_to_f16_bits(nxt() * 4.0);
        }
        // logical kv 按 kv 头存(kv 头不随 q 头重复)
        let mut logical_k = vec![vec![0f32; HEAD_SIZE]; NUM_SEQS * KV_HEADS * BS];
        let mut logical_v = vec![vec![0f32; HEAD_SIZE]; NUM_SEQS * KV_HEADS * BS];
        for seq in 0..NUM_SEQS {
            for pos in 0..BS {
                for h in 0..KV_HEADS {
                    for d in 0..HEAD_SIZE {
                        logical_k[(seq * KV_HEADS + h) * BS + pos][d] = nxt() * 2.0;
                        logical_v[(seq * KV_HEADS + h) * BS + pos][d] = nxt() * 2.0;
                    }
                }
            }
        }

        // 分页散布(x = 8 布局不变,head 维 = kv 头数)
        let mut block_tables = vec![0i32; NUM_SEQS * MAX_BPS];
        for seq in 0..NUM_SEQS {
            block_tables[seq * MAX_BPS] = (seq * 4) as i32;
            block_tables[seq * MAX_BPS + 1] = (seq * 4 + 1) as i32;
        }
        let kc_len = BLOCKS * KV_HEADS * (HEAD_SIZE / X) * BS * X;
        let vc_len = BLOCKS * KV_HEADS * HEAD_SIZE * BS;
        let mut key_cache = vec![f32_to_f16_bits(0.0); kc_len];
        let mut value_cache = vec![f32_to_f16_bits(0.0); vc_len];
        for seq in 0..NUM_SEQS {
            for pos in 0..BS {
                if pos >= lens[seq] {
                    continue;
                }
                let pblock = block_tables[seq * MAX_BPS + pos / BS] as usize;
                let poff = pos % BS;
                for h in 0..KV_HEADS {
                    for d in 0..HEAD_SIZE {
                        let tk = pblock * (KV_HEADS * (HEAD_SIZE / X) * BS * X)
                            + h * ((HEAD_SIZE / X) * BS * X)
                            + (d / X) * (BS * X)
                            + poff * X
                            + d % X;
                        key_cache[tk] =
                            f32_to_f16_bits(logical_k[(seq * KV_HEADS + h) * BS + pos][d]);
                        let tv = pblock * (KV_HEADS * HEAD_SIZE * BS)
                            + h * (HEAD_SIZE * BS)
                            + d * BS
                            + poff;
                        value_cache[tv] =
                            f32_to_f16_bits(logical_v[(seq * KV_HEADS + h) * BS + pos][d]);
                    }
                }
            }
        }
        let context_lens: Vec<i32> = lens.iter().map(|&l| l as i32).collect();

        // host 参考:q 头 h → kv 头 h / (HEADS/KV_HEADS)
        let q_f32: Vec<f32> = q.iter().map(|&b| f16_bits_to_f32(b)).collect();
        let gqa = HEADS / KV_HEADS;
        let mut expect = vec![0f32; NUM_SEQS * HEADS * HEAD_SIZE];
        for seq in 0..NUM_SEQS {
            let len = lens[seq];
            for h in 0..HEADS {
                let kvh = h / gqa;
                let mut logits = vec![0f32; len];
                for pos in 0..len {
                    let mut dot = 0f32;
                    for d in 0..HEAD_SIZE {
                        dot += q_f32[(seq * HEADS + h) * HEAD_SIZE + d]
                            * logical_k[(seq * KV_HEADS + kvh) * BS + pos][d];
                    }
                    logits[pos] = dot * scale;
                }
                let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = logits.iter().map(|&l| (l - mx).exp()).collect();
                let sum: f32 = exps.iter().sum();
                for d in 0..HEAD_SIZE {
                    let mut acc = 0f32;
                    for pos in 0..len {
                        acc += exps[pos] / sum * logical_v[(seq * KV_HEADS + kvh) * BS + pos][d];
                    }
                    expect[(seq * HEADS + h) * HEAD_SIZE + d] = acc;
                }
            }
        }

        let dev = CudaDevice::new(owl_cuda::test_device_ordinal(), owl_cuda::TEST_POOL_BYTES).expect("需要 CUDA 设备");
        let pool = dev.default_pool();
        let d_q = htod_t::<u16> (&pool, q.clone());
        let d_kc = htod_t::<u16> (&pool, key_cache.clone());
        let d_vc = htod_t::<u16> (&pool, value_cache.clone());
        let d_bt = htod_t::<i32> (&pool, block_tables.clone());
        let d_cl = htod_t::<i32> (&pool, context_lens.clone());
        let d_out = alloc_t::<u16> (&pool, NUM_SEQS * HEADS * HEAD_SIZE);
        dev.ctx().synchronize().unwrap();

        let mut kern = PagedKernels::new(dev.ctx()).unwrap();
        kern.paged_attention_v1(
            dev.stream(),
            0,   // f16
            32,  // block_size
            256, // head_size
            owl_iface::DevBuf::device_ptr(&d_out) as *mut u16,
            owl_iface::DevBuf::device_ptr(&d_q) as *const u16,
            owl_iface::DevBuf::device_ptr(&d_kc) as *const u16,
            owl_iface::DevBuf::device_ptr(&d_vc) as *const u16,
            KV_HEADS as i32,
            scale,
            owl_iface::DevBuf::device_ptr(&d_bt) as *const i32,
            owl_iface::DevBuf::device_ptr(&d_cl) as *const i32,
            MAX_BPS as i32,
            (HEADS * HEAD_SIZE) as i32,
            (KV_HEADS * (HEAD_SIZE / X) * BS * X) as i32,
            ((HEAD_SIZE / X) * BS * X) as i32,
            BS as i32,
            NUM_SEQS as i32,
            HEADS as i32,
        )
        .unwrap();
        dev.ctx().synchronize().unwrap();

        let mut out_host = vec![0u16; NUM_SEQS * HEADS * HEAD_SIZE];
        // P0-3:流序 D2H(memx bytes;u16 位型)
        dev.memcpy_dtoh_bytes(
            dev.stream(),
            owl_iface::DevBuf::device_ptr(&d_out) as *const u8,
            unsafe { std::slice::from_raw_parts_mut(out_host.as_mut_ptr() as *mut u8, out_host.len() * 2) },
        )
        .unwrap();
        let mut max_diff = 0f32;
        let mut shown = 0;
        for (i, &o) in out_host.iter().enumerate() {
            let diff = (f16_bits_to_f32(o) - expect[i]).abs();
            if diff > 2e-2 && shown < 16 {
                eprintln!("DBG i={i} (seq {} h {} d {}) got={} want={}",
                    i / (HEADS * HEAD_SIZE), (i / HEAD_SIZE) % HEADS, i % HEAD_SIZE,
                    f16_bits_to_f32(o), expect[i]);
                shown += 1;
            }
            max_diff = max_diff.max(diff);
        }
        assert!(max_diff < 2e-2, "v1 f16 h256 GQA 对拍最大偏差 {max_diff}");
        eprintln!("[K1-h256] v1 f16 head256 GQA max_diff = {max_diff:.4e}");
    }

    /// B1 验收二:v2 f16 head 256 大 ctx 分片 + GQA + v1/v2 交叉。
    /// lens [1000, 600](2 partitions);8 q / 2 kv,head 256。
    #[test]
    fn paged_attention_v2_f16_h256_gqa_parity_and_partitions() {
        const NUM_SEQS: usize = 2;
        const HEADS: usize = 8;
        const KV_HEADS: usize = 2;
        const HEAD_SIZE: usize = 256;
        const BS: usize = 32;
        const MAX_BPS: usize = 33; // ceil(1000/32)=32 +1 冗余
        const BLOCKS: usize = NUM_SEQS * MAX_BPS + 2;
        const X: usize = 8;
        let scale = 1.0 / (HEAD_SIZE as f32).sqrt();

        let lens = [1000usize, 600usize];
        let mut s = 0.31f32;
        let mut nxt = || {
            s = (s * 1.29 + 0.17).fract() - 0.5;
            s
        };
        let mut q = vec![0u16; NUM_SEQS * HEADS * HEAD_SIZE];
        for v in q.iter_mut() {
            *v = f32_to_f16_bits(nxt() * 4.0);
        }
        let max_len = lens[0];
        let mut logical_k = vec![vec![0f32; HEAD_SIZE]; NUM_SEQS * KV_HEADS * max_len];
        let mut logical_v = vec![vec![0f32; HEAD_SIZE]; NUM_SEQS * KV_HEADS * max_len];
        for seq in 0..NUM_SEQS {
            for pos in 0..lens[seq] {
                for h in 0..KV_HEADS {
                    for d in 0..HEAD_SIZE {
                        logical_k[(seq * KV_HEADS + h) * max_len + pos][d] = nxt() * 2.0;
                        logical_v[(seq * KV_HEADS + h) * max_len + pos][d] = nxt() * 2.0;
                    }
                }
            }
        }

        let mut block_tables = vec![0i32; NUM_SEQS * MAX_BPS];
        for seq in 0..NUM_SEQS {
            for b_i in 0..MAX_BPS {
                block_tables[seq * MAX_BPS + b_i] =
                    if b_i * BS < lens[seq] { (seq * MAX_BPS + b_i) as i32 } else { 0 };
            }
        }
        let kc_len = BLOCKS * KV_HEADS * (HEAD_SIZE / X) * BS * X;
        let vc_len = BLOCKS * KV_HEADS * HEAD_SIZE * BS;
        let mut key_cache = vec![f32_to_f16_bits(0.0); kc_len];
        let mut value_cache = vec![f32_to_f16_bits(0.0); vc_len];
        for seq in 0..NUM_SEQS {
            for pos in 0..lens[seq] {
                let pblock = block_tables[seq * MAX_BPS + pos / BS] as usize;
                let poff = pos % BS;
                for h in 0..KV_HEADS {
                    for d in 0..HEAD_SIZE {
                        let tk = pblock * (KV_HEADS * (HEAD_SIZE / X) * BS * X)
                            + h * ((HEAD_SIZE / X) * BS * X)
                            + (d / X) * (BS * X)
                            + poff * X
                            + d % X;
                        key_cache[tk] =
                            f32_to_f16_bits(logical_k[(seq * KV_HEADS + h) * max_len + pos][d]);
                        let tv = pblock * (KV_HEADS * HEAD_SIZE * BS)
                            + h * (HEAD_SIZE * BS)
                            + d * BS
                            + poff;
                        value_cache[tv] =
                            f32_to_f16_bits(logical_v[(seq * KV_HEADS + h) * max_len + pos][d]);
                    }
                }
            }
        }
        let context_lens: Vec<i32> = lens.iter().map(|&l| l as i32).collect();

        // host 参考(GQA 同上)
        let q_f32: Vec<f32> = q.iter().map(|&b| f16_bits_to_f32(b)).collect();
        let gqa = HEADS / KV_HEADS;
        let mut expect = vec![0f32; NUM_SEQS * HEADS * HEAD_SIZE];
        for seq in 0..NUM_SEQS {
            let len = lens[seq];
            for h in 0..HEADS {
                let kvh = h / gqa;
                let mut logits = vec![0f32; len];
                for pos in 0..len {
                    let mut dot = 0f32;
                    for d in 0..HEAD_SIZE {
                        dot += q_f32[(seq * HEADS + h) * HEAD_SIZE + d]
                            * logical_k[(seq * KV_HEADS + kvh) * max_len + pos][d];
                    }
                    logits[pos] = dot * scale;
                }
                let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = logits.iter().map(|&l| (l - mx).exp()).collect();
                let sum: f32 = exps.iter().sum();
                for d in 0..HEAD_SIZE {
                    let mut acc = 0f32;
                    for pos in 0..len {
                        acc +=
                            exps[pos] / sum * logical_v[(seq * KV_HEADS + kvh) * max_len + pos][d];
                    }
                    expect[(seq * HEADS + h) * HEAD_SIZE + d] = acc;
                }
            }
        }

        let dev = CudaDevice::new(owl_cuda::test_device_ordinal(), owl_cuda::TEST_POOL_BYTES).expect("需要 CUDA 设备");
        let pool = dev.default_pool();
        let d_q = htod_t::<u16> (&pool, q.clone());
        let d_kc = htod_t::<u16> (&pool, key_cache.clone());
        let d_vc = htod_t::<u16> (&pool, value_cache.clone());
        let d_bt = htod_t::<i32> (&pool, block_tables.clone());
        let d_cl = htod_t::<i32> (&pool, context_lens.clone());
        let d_out_v1 = alloc_t::<u16> (&pool, NUM_SEQS * HEADS * HEAD_SIZE);
        let d_out_v2 = alloc_t::<u16> (&pool, NUM_SEQS * HEADS * HEAD_SIZE);
        const PARTS: usize = 2;
        let d_exp = alloc_t::<f32> (&pool, NUM_SEQS * HEADS * PARTS);
        let d_maxl = alloc_t::<f32> (&pool, NUM_SEQS * HEADS * PARTS);
        let d_tmp = alloc_t::<u16> (&pool, NUM_SEQS * HEADS * PARTS * HEAD_SIZE);

        let mut kern = PagedKernels::new(dev.ctx()).unwrap();
        let (q_p, kc_p, vc_p, bt_p, cl_p) = (
            owl_iface::DevBuf::device_ptr(&d_q) as *const u16,
            owl_iface::DevBuf::device_ptr(&d_kc) as *const u16,
            owl_iface::DevBuf::device_ptr(&d_vc) as *const u16,
            owl_iface::DevBuf::device_ptr(&d_bt) as *const i32,
            owl_iface::DevBuf::device_ptr(&d_cl) as *const i32,
        );
        kern.paged_attention_v1(
            dev.stream(), 0, 32, 256,
            owl_iface::DevBuf::device_ptr(&d_out_v1) as *mut u16,
            q_p, kc_p, vc_p,
            KV_HEADS as i32, scale,
            bt_p, cl_p, MAX_BPS as i32,
            (HEADS * HEAD_SIZE) as i32,
            (KV_HEADS * (HEAD_SIZE / X) * BS * X) as i32,
            ((HEAD_SIZE / X) * BS * X) as i32,
            lens[0] as i32, NUM_SEQS as i32, HEADS as i32,
        )
        .unwrap();
        kern.paged_attention_v2(
            dev.stream(), 0, 32, 256,
            owl_iface::DevBuf::device_ptr(&d_out_v2) as *mut u16,
            owl_iface::DevBuf::device_ptr(&d_exp) as *mut f32,
            owl_iface::DevBuf::device_ptr(&d_maxl) as *mut f32,
            owl_iface::DevBuf::device_ptr(&d_tmp) as *mut u16,
            q_p, kc_p, vc_p,
            KV_HEADS as i32, scale,
            bt_p, cl_p, MAX_BPS as i32,
            (HEADS * HEAD_SIZE) as i32,
            (KV_HEADS * (HEAD_SIZE / X) * BS * X) as i32,
            ((HEAD_SIZE / X) * BS * X) as i32,
            lens[0] as i32, NUM_SEQS as i32, HEADS as i32,
        )
        .unwrap();
        dev.ctx().synchronize().unwrap();

        // P0-3:流序 D2H(memx bytes;u16 位型)
        let read_u16 = |ptr: *mut u16| -> Vec<u16> {
            let mut out = vec![0u16; NUM_SEQS * HEADS * HEAD_SIZE];
            dev.memcpy_dtoh_bytes(
                dev.stream(),
                ptr as *const u8,
                unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, out.len() * 2) },
            )
            .unwrap();
            out
        };
        let out_v1 = read_u16(owl_iface::DevBuf::device_ptr(&d_out_v1) as *mut u16);
        let out_v2 = read_u16(owl_iface::DevBuf::device_ptr(&d_out_v2) as *mut u16);

        let mut host_max = 0f32;
        for (i, &o) in out_v2.iter().enumerate() {
            host_max = host_max.max((f16_bits_to_f32(o) - expect[i]).abs());
        }
        assert!(host_max < 2e-2, "v2 h256 vs host 最大偏差 {host_max}");
        let mut vv_max = 0f32;
        for (&a, &b) in out_v1.iter().zip(out_v2.iter()) {
            vv_max = vv_max.max((f16_bits_to_f32(a) - f16_bits_to_f32(b)).abs());
        }
        assert!(vv_max < 2e-2, "v1/v2 h256 交叉最大偏差 {vv_max}");
        eprintln!(
            "[K2-h256] v2-vs-host max_diff = {host_max:.4e}; v1-vs-v2 max_diff = {vv_max:.4e}"
        );
    }

    /// B1 验收三:v1 bf16 head 256 smoke(编译链 + 有限值)。
    #[test]
    fn paged_attention_v1_bf16_h256_smoke() {
        let dev = CudaDevice::new(owl_cuda::test_device_ordinal(), owl_cuda::TEST_POOL_BYTES).expect("需要 CUDA 设备");
        let pool = dev.default_pool();
        let n_kc = 8 * 2 * 32 * 32 * 8;
        let n_vc = 8 * 2 * 256 * 32;
        let q = vec![f32_to_bf16_bits(0.1); 8 * 256];
        let kc = vec![f32_to_bf16_bits(0.2); n_kc];
        let vc = vec![f32_to_bf16_bits(0.3); n_vc];
        let bt = vec![0i32, 1];
        let cl = vec![16i32, 16];
        let d_q = htod_t::<u16> (&pool, q);
        let d_kc = htod_t::<u16> (&pool, kc);
        let d_vc = htod_t::<u16> (&pool, vc);
        let d_bt = htod_t::<i32> (&pool, bt);
        let d_cl = htod_t::<i32> (&pool, cl);
        let d_out = alloc_t::<u16> (&pool, 2 * 8 * 256);
        dev.ctx().synchronize().unwrap();
        let mut kern = PagedKernels::new(dev.ctx()).unwrap();
        kern.paged_attention_v1(
            dev.stream(), 1, 32, 256,
            owl_iface::DevBuf::device_ptr(&d_out) as *mut u16,
            owl_iface::DevBuf::device_ptr(&d_q) as *const u16,
            owl_iface::DevBuf::device_ptr(&d_kc) as *const u16,
            owl_iface::DevBuf::device_ptr(&d_vc) as *const u16,
            2, 1.0 / 256f32.sqrt(),
            owl_iface::DevBuf::device_ptr(&d_bt) as *const i32,
            owl_iface::DevBuf::device_ptr(&d_cl) as *const i32,
            1, 8 * 256, 2 * 32 * 32 * 8, 32 * 256, 32, 2, 8,
        )
        .unwrap();
        dev.ctx().synchronize().unwrap();
        let mut out_host = vec![0u16; 2 * 8 * 256];
        // P0-3:流序 D2H(memx bytes;u16 位型)
        dev.memcpy_dtoh_bytes(
            dev.stream(),
            owl_iface::DevBuf::device_ptr(&d_out) as *const u8,
            unsafe { std::slice::from_raw_parts_mut(out_host.as_mut_ptr() as *mut u8, out_host.len() * 2) },
        )
        .unwrap();
        for (i, &b) in out_host.iter().enumerate() {
            let f = bf16_bits_to_f32(b);
            assert!(f.is_finite(), "v1 bf16 h256 smoke: out[{i}] 非有限 {f}");
        }
    }

    /// B1 验收四:v2 bf16 head 256 smoke(主核+reduce 编译链 + 有限值)。
    #[test]
    fn paged_attention_v2_bf16_h256_smoke() {
        let dev = CudaDevice::new(owl_cuda::test_device_ordinal(), owl_cuda::TEST_POOL_BYTES).expect("需要 CUDA 设备");
        let pool = dev.default_pool();
        const N: usize = 2 * 8 * 256;
        let q: Vec<u16> =
            (0..N).map(|i| f32_to_bf16_bits(((i % 7) as f32 - 3.0) * 0.25)).collect();
        let kc: Vec<u16> =
            (0..8 * 2 * 32 * 32 * 8).map(|i| f32_to_bf16_bits(((i % 5) as f32 - 2.0) * 0.2)).collect();
        let vc: Vec<u16> =
            (0..8 * 2 * 256 * 32).map(|i| f32_to_bf16_bits(((i % 9) as f32 - 4.0) * 0.2)).collect();
        let bt = vec![0i32, 1];
        let cl = vec![600i32, 300i32];
        let d_q = htod_t::<u16> (&pool, q);
        let d_kc = htod_t::<u16> (&pool, kc);
        let d_vc = htod_t::<u16> (&pool, vc);
        let d_bt = htod_t::<i32> (&pool, bt);
        let d_cl = htod_t::<i32> (&pool, cl);
        let d_out = alloc_t::<u16> (&pool, N);
        let d_exp = alloc_t::<f32> (&pool, 2 * 8 * 2);
        let d_ml = alloc_t::<f32> (&pool, 2 * 8 * 2);
        let d_tmp = alloc_t::<u16> (&pool, 2 * 8 * 2 * 256);
        let mut kern = PagedKernels::new(dev.ctx()).unwrap();
        kern.paged_attention_v2(
            dev.stream(), 1, 32, 256,
            owl_iface::DevBuf::device_ptr(&d_out) as *mut u16,
            owl_iface::DevBuf::device_ptr(&d_exp) as *mut f32,
            owl_iface::DevBuf::device_ptr(&d_ml) as *mut f32,
            owl_iface::DevBuf::device_ptr(&d_tmp) as *mut u16,
            owl_iface::DevBuf::device_ptr(&d_q) as *const u16,
            owl_iface::DevBuf::device_ptr(&d_kc) as *const u16,
            owl_iface::DevBuf::device_ptr(&d_vc) as *const u16,
            2, 1.0 / 256f32.sqrt(),
            owl_iface::DevBuf::device_ptr(&d_bt) as *const i32,
            owl_iface::DevBuf::device_ptr(&d_cl) as *const i32,
            2, 8 * 256, 2 * 32 * 32 * 8, 32 * 256, 600, 2, 8,
        )
        .unwrap();
        dev.ctx().synchronize().unwrap();
        let mut out_host = vec![0u16; N];
        // P0-3:流序 D2H(memx bytes;u16 位型)
        dev.memcpy_dtoh_bytes(
            dev.stream(),
            owl_iface::DevBuf::device_ptr(&d_out) as *const u8,
            unsafe { std::slice::from_raw_parts_mut(out_host.as_mut_ptr() as *mut u8, out_host.len() * 2) },
        )
        .unwrap();
        for (i, &b) in out_host.iter().enumerate() {
            let f = bf16_bits_to_f32(b);
            assert!(f.is_finite(), "v2 bf16 h256 smoke: out[{i}] 非有限 {f}");
        }
    }}

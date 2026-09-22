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
    use super::AttentionKernels;
    use owl_cuda::CudaDevice;
    use owl_iface::{Device as _, PoolConfig, PoolKind};

    /// K0 验收:合成 KV 块写入对拍 host 参考(vLLM 布局公式直译)。
    /// 含负槽(padding 跳过)分支。
    #[test]
    fn reshape_and_cache_f32_host_parity() {
        let dev = CudaDevice::new(owl_cuda::test_device_ordinal()).expect("需要 CUDA 设备");
        let pool = dev
            .create_pool(PoolConfig {
                name: format!("k0-{}", std::process::id()),
                kind: PoolKind::Weights,
                bytes: 4 << 20,
            })
            .unwrap();

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

        let d_key = dev.htod_persistent_in::<f32>(&pool, key.clone()).unwrap();
        let d_value = dev.htod_persistent_in::<f32>(&pool, value.clone()).unwrap();
        let d_slot = dev
            .htod_persistent_in::<i64>(&pool, slot_mapping.clone())
            .unwrap();
        let d_kcache = dev
            .alloc_persistent_in::<f32>(&pool, BLOCKS * HEADS * HEAD_SIZE * BLOCK_SIZE)
            .unwrap();
        let d_vcache = dev
            .alloc_persistent_in::<f32>(&pool, BLOCKS * HEADS * HEAD_SIZE * BLOCK_SIZE)
            .unwrap();
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
        use owl_cuda::ffi::sys;
        dev.ctx().bind_to_thread().unwrap();
        let read = |ptr: *mut f32| -> Vec<f32> {
            let mut out = vec![0f32; BLOCKS * HEADS * HEAD_SIZE * BLOCK_SIZE];
            unsafe {
                sys::cuMemcpyDtoH_v2(
                    out.as_mut_ptr() as *mut std::ffi::c_void,
                    ptr as sys::CUdeviceptr,
                    out.len() * 4,
                )
                .result()
                .unwrap();
            }
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

/// paged_attention 核族(独立 nvrtc 模块;头目录 = kernels/cu)
pub struct PagedKernels {
    ctx: Arc<CudaContext>,
    module: Arc<owl_cuda::ffi::CudaModule>,
    loaded: HashSet<&'static str>,
}

impl PagedKernels {
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, String> {
        let include = std::env::var("CUDA_INCLUDE")
            .unwrap_or_else(|_| "/usr/local/cuda/include".to_string());
        let cu_dir = concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/cu");
        let opts = CompileOptions {
            include_paths: vec![
                format!("{}/attention_compat", std::env::var("CARGO_MANIFEST_DIR").unwrap()),
                include,
                cu_dir.to_string(),
                format!("{cu_dir}/attention"),
            ],
            use_fast_math: Some(true),
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(PAGED_V1_SRC, opts).map_err(|e| format!("nvrtc: {e}"))?;
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

    /// v1 解码核。布局契约见 cu/attention_paged_v1.cu 头部。
    /// `dtype`: 0 = f16(vLLM uint16 位型路径);1 = bf16。
    /// 全部动态量(device 张量裸指针;A1.5);alibi 暂 nullptr。
    #[allow(clippy::too_many_arguments)]
    pub fn paged_attention_v1(
        &mut self,
        stream: &owl_cuda::ffi::CudaStream,
        dtype: u32, // 0=f16 1=bf16
        block_size: u32,
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
        let name = match (dtype, block_size) {
            (0, 32) => "owl_pa_v1_f16_bs32",
            (0, 64) => "owl_pa_v1_f16_bs64",
            (1, 32) => "owl_pa_v1_bf16_bs32",
            (1, 64) => "owl_pa_v1_bf16_bs64",
            _ => return Err(format!("paged_attention_v1: dtype {dtype} bs {block_size} 未实例化")),
        };
        let func = self.func(name)?;
        let padded = ((max_context_len + block_size as i32 - 1) / block_size as i32)
            * block_size as i32;
        let logits_size = padded as usize * 4;
        let outputs_size = (128 / 32 / 2) * 128 * 4; // NUM_WARPS/2 * head_size * 4
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
}

fn softscapping_bits() -> u64 {
    1.0f32.to_bits() as u64 // 恒等软上限(qwen3.5 不用)
}
fn sliding_window_zero() -> u64 {
    0 // 0 = 关闭滑窗
}

#[cfg(test)]
mod paged_tests {
    use super::PagedKernels;
    use owl_cuda::CudaDevice;
    use owl_iface::{Device as _, PoolConfig, PoolKind};

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
        let dev = CudaDevice::new(owl_cuda::test_device_ordinal()).expect("需要 CUDA 设备");
        let pool = dev
            .create_pool(PoolConfig {
                name: format!("k1-{}", std::process::id()),
                kind: PoolKind::Weights,
                bytes: 8 << 20,
            })
            .unwrap();
        let d_q = dev.htod_persistent_in::<u16>(&pool, q.clone()).unwrap();
        let d_kc = dev.htod_persistent_in::<u16>(&pool, key_cache.clone()).unwrap();
        let d_vc = dev.htod_persistent_in::<u16>(&pool, value_cache.clone()).unwrap();
        let d_bt = dev.htod_persistent_in::<i32>(&pool, block_tables.clone()).unwrap();
        let d_cl = dev.htod_persistent_in::<i32>(&pool, context_lens.clone()).unwrap();
        let d_out = dev
            .alloc_persistent_in::<u16>(&pool, NUM_SEQS * HEADS * HEAD_SIZE)
            .unwrap();
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
        use owl_cuda::ffi::sys;
        dev.ctx().bind_to_thread().unwrap();
        unsafe {
            sys::cuMemcpyDtoH_v2(
                out_host.as_mut_ptr() as *mut std::ffi::c_void,
                owl_iface::DevBuf::device_ptr(&d_out) as sys::CUdeviceptr,
                out_host.len() * 2,
            )
            .result()
            .unwrap();
        }
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
        let dev = CudaDevice::new(owl_cuda::test_device_ordinal()).expect("需要 CUDA 设备");
        let pool = dev
            .create_pool(PoolConfig {
                name: format!("k1b-{}", std::process::id()),
                kind: PoolKind::Weights,
                bytes: 8 << 20,
            })
            .unwrap();
        let n_kc = 8 * 8 * 16 * 32 * 8;
        let n_vc = 8 * 8 * 128 * 32;
        let q = vec![f32_to_bf16_bits(0.1); 8 * 128];
        let kc = vec![f32_to_bf16_bits(0.2); n_kc];
        let vc = vec![f32_to_bf16_bits(0.3); n_vc];
        let bt = vec![0i32; 2];
        let cl = vec![16i32, 16];
        let d_q = dev.htod_persistent_in::<u16>(&pool, q).unwrap();
        let d_kc = dev.htod_persistent_in::<u16>(&pool, kc).unwrap();
        let d_vc = dev.htod_persistent_in::<u16>(&pool, vc).unwrap();
        let d_bt = dev.htod_persistent_in::<i32>(&pool, bt).unwrap();
        let d_cl = dev.htod_persistent_in::<i32>(&pool, cl).unwrap();
        let d_out = dev.alloc_persistent_in::<u16>(&pool, 2 * 8 * 128).unwrap();
        dev.ctx().synchronize().unwrap();
        let mut kern = PagedKernels::new(dev.ctx()).unwrap();
        kern.paged_attention_v1(
            dev.stream(), 1, 32,
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
        use owl_cuda::ffi::sys;
        dev.ctx().bind_to_thread().unwrap();
        unsafe {
            sys::cuMemcpyDtoH_v2(
                out_host.as_mut_ptr() as *mut std::ffi::c_void,
                owl_iface::DevBuf::device_ptr(&d_out) as sys::CUdeviceptr,
                out_host.len() * 2,
            )
            .result()
            .unwrap();
        }
        // bf16 smoke:全部有限值即可(有 NaN 说明越界/掩码错)
        for (i, &b) in out_host.iter().enumerate() {
            let f = bf16_bits_to_f32(b);
            assert!(f.is_finite(), "bf16 smoke: out[{i}] 非有限 {f}");
        }
    }
}

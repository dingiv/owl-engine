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

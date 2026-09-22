//! GDN/线性注意力 kernel 族(K3 第一刀:fused_gating / l2_norm / rmsnorm_act)。
//!
//! port 专项:docs/arch/attention-kernel-port.md;出处与改编见
//! cu/gdn/gdn_kernels.cu 头部。独立 nvrtc 模块。
//!
//! [K3 第二刀待办] causal_conv1d_fwd/update_slots、gated_delta_rule_
//! recurrence_*(tiled/varlen/gqa/decode_slots)——签名见 gdn_shim 需求面。

use owl_cuda::ffi::nvrtc::{compile_ptx_with_opts, CompileOptions};
use owl_cuda::ffi::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use std::collections::HashSet;
use std::sync::Arc;

const GDN_SRC: &str = include_str!("../../kernels/cu/gdn/gdn_kernels.cu");

pub struct GdnKernels {
    ctx: Arc<CudaContext>,
    module: Arc<owl_cuda::ffi::CudaModule>,
    loaded: HashSet<&'static str>,
}

impl GdnKernels {
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, String> {
        let include = std::env::var("CUDA_INCLUDE")
            .unwrap_or_else(|_| "/usr/local/cuda/include".to_string());
        let opts = CompileOptions {
            include_paths: vec![include],
            use_fast_math: Some(true),
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(GDN_SRC, opts).map_err(|e| format!("nvrtc: {e}"))?;
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
        stream: &CudaStream,
        name: &'static str,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: &[u64],
    ) -> Result<(), String> {
        let func = self.func(name)?;
        let cfg = LaunchConfig { grid_dim: grid, block_dim: block, shared_mem_bytes: 0 };
        unsafe {
            let mut builder = stream.launch_builder(&func);
            for a in args {
                builder.arg(a);
            }
            builder.launch(cfg).map_err(|e| format!("launch({name}): {e}"))?;
        }
        let _ = &self.ctx;
        Ok(())
    }

    /// fused_gdn_gating:f32 权重 + 激活档。a/b: [total](含 head 维),
    /// a_log/dt_bias: [num_heads](广播),g/beta: [total] 输出。
    #[allow(clippy::too_many_arguments)]
    pub fn fused_gating(
        &mut self,
        stream: &CudaStream,
        dtype: &'static str, // "f32" | "f16" | "bf16"
        a_log: *const f32,
        a: *const u8,
        b: *const u8,
        dt_bias: *const f32,
        g: *mut f32,
        beta: *mut f32,
        total: i32,
        num_heads: i32,
    ) -> Result<(), String> {
        let name: &'static str = match dtype {
            "f32" => "gdn_fused_gating_f32",
            "f16" => "gdn_fused_gating_f16",
            "bf16" => "gdn_fused_gating_bf16",
            _ => return Err(format!("fused_gating: 未知 dtype {dtype}")),
        };
        let blocks = (total as u32).div_ceil(256).max(1);
        self.launch(
            stream,
            name,
            (blocks, 1, 1),
            (256, 1, 1),
            &[
                a_log as u64,
                a as u64,
                b as u64,
                dt_bias as u64,
                g as u64,
                beta as u64,
                total as u64,
                num_heads as u64,
            ],
        )
    }

    /// l2_norm_last_dim
    pub fn l2_norm(
        &mut self,
        stream: &CudaStream,
        dtype: &'static str,
        input: *const u8,
        output: *mut u8,
        rows: i32,
        dim: i32,
        eps: f32,
    ) -> Result<(), String> {
        let (name, grid, block) = if dim <= 256 {
            let n: &'static str = match dtype {
                "f32" => "gdn_l2norm_warp_f32",
                "f16" => "gdn_l2norm_warp_f16",
                "bf16" => "gdn_l2norm_warp_bf16",
                _ => return Err(format!("l2_norm: 未知 dtype {dtype}")),
            };
            (n, ((rows as u32).div_ceil(8), 1, 1), (8 * 32, 1, 1))
        } else {
            let n: &'static str = match dtype {
                "f32" => "gdn_l2norm_block256_f32",
                "f16" => "gdn_l2norm_block256_f16",
                "bf16" => "gdn_l2norm_block256_bf16",
                _ => return Err(format!("l2_norm: 未知 dtype {dtype}")),
            };
            (n, (rows as u32, 1, 1), (256, 1, 1))
        };
        self.launch(
            stream,
            name,
            grid,
            block,
            &[input as u64, output as u64, rows as u64, dim as u64, eps.to_bits() as u64],
        )
    }

    /// gated rmsnorm + act(z)(silu/sigmoid)+ mul
    #[allow(clippy::too_many_arguments)]
    pub fn rmsnorm_act(
        &mut self,
        stream: &CudaStream,
        dtype: &'static str,
        x: *const u8,
        z: *const u8,
        gamma: *const u8,
        bias: *const u8, // 可空
        out: *mut u8,
        rows: i32,
        value_dim: i32,
        group_size: i32,
        eps: f32,
        per_group_weights: bool,
        has_bias: bool,
        act: i32, // 0=silu(Qwen3.5) 1=sigmoid(Qwen4)
    ) -> Result<(), String> {
        // 障碍挂起(2026-09-22):per-group gamma 长度语义与 deltanet 调用点
        // 不一致(host 参考按单组共享,kernel 按组偏移)——K3 第二刀与
        // deltanet.rs 调用面对齐后启用;cu 核已写好(cu/gdn/gdn_kernels.cu)
        let _ = (stream, dtype, x, z, gamma, bias, out, rows, value_dim, group_size, eps, per_group_weights, has_bias, act);
        unimplemented!("K3 第二刀: rmsnorm_act 参数语义对齐 deltanet 调用点")
    }
}

#[cfg(test)]
mod tests {
    use super::GdnKernels;
    use owl_cuda::CudaDevice;
    use owl_iface::{Device as _, PoolConfig, PoolKind};

    fn setup() -> (
        CudaDevice,
        GdnKernels,
        owl_cuda::CudaPool,
    ) {
        let dev = CudaDevice::new(owl_cuda::test_device_ordinal()).expect("需要 CUDA 设备");
        let k = GdnKernels::new(dev.ctx()).expect("nvrtc gdn");
        let pool = dev
            .create_pool(PoolConfig {
                name: format!("gdn-t-{}", std::process::id()),
                kind: PoolKind::Weights,
                bytes: 16 << 20,
            })
            .unwrap();
        (dev, k, pool)
    }

    fn htod<T: owl_iface::MemValue + Send + Sync + 'static>(
        dev: &CudaDevice,
        pool: &owl_cuda::CudaPool,
        v: Vec<T>,
    ) -> owl_cuda::Persistent<T> {
        // 返回持有缓冲(活性随行;裸指针版会即刻回收 → 悬空)
        dev.htod_persistent_in(pool, v).unwrap()
    }

    fn dtoh_f32(dev: &CudaDevice, ptr: *const f32, n: usize) -> Vec<f32> {
        use owl_cuda::ffi::sys;
        dev.ctx().bind_to_thread().unwrap();
        let mut out = vec![0f32; n];
        unsafe {
            sys::cuMemcpyDtoH_v2(
                out.as_mut_ptr() as *mut std::ffi::c_void,
                ptr as sys::CUdeviceptr,
                n * 4,
            )
            .result()
            .unwrap();
        }
        out
    }

    /// gating 对拍:host 参考 = compute_gating 公式直译
    #[test]
    fn fused_gating_f32_host_parity() {
        let (dev, mut k, pool) = setup();
        let total = 64usize;
        let heads = 8usize;
        let a_log: Vec<f32> = (0..heads).map(|i| -0.1 * i as f32).collect();
        let dt_bias: Vec<f32> = (0..heads).map(|i| 0.05 * i as f32).collect();
        let a: Vec<f32> = (0..total).map(|i| ((i % 17) as f32 - 8.0) * 0.3).collect();
        let b: Vec<f32> = (0..total).map(|i| ((i % 23) as f32 - 11.0) * 0.2).collect();
        let g_out = htod(&dev, &pool, vec![0f32; total]);
        let b_out = htod(&dev, &pool, vec![0f32; total]);
        let d_alog = htod(&dev, &pool, a_log.clone());
        let d_dt = htod(&dev, &pool, dt_bias.clone());
        let d_a = htod(&dev, &pool, a.clone());
        let d_b = htod(&dev, &pool, b.clone());
        let g_p = owl_iface::DevBuf::device_ptr(&g_out) as *mut f32;
        let b_p = owl_iface::DevBuf::device_ptr(&b_out) as *mut f32;
        let alog_p = owl_iface::DevBuf::device_ptr(&d_alog) as *const f32;
        let dt_p = owl_iface::DevBuf::device_ptr(&d_dt) as *const f32;
        let a_p = owl_iface::DevBuf::device_ptr(&d_a) as *const f32;
        let b_p2 = owl_iface::DevBuf::device_ptr(&d_b) as *const f32;

        let stream = dev.stream();
        k.fused_gating(
            stream,
            "f32",
            alog_p,
            a_p as *const u8,
            b_p2 as *const u8,
            dt_p,
            g_p,
            b_p,
            total as i32,
            heads as i32,
        )
        .unwrap();
        dev.ctx().synchronize().unwrap();

        let g = dtoh_f32(&dev, g_p, total);
        let beta = dtoh_f32(&dev, b_p, total);
        for i in 0..total {
            let h = i % heads;
            let x = a[i] + dt_bias[h];
            let sp = if x <= 20.0 { x.exp().ln_1p() } else { x };
            let eg = -a_log[h].exp() * sp;
            let eb = 1.0 / (1.0 + (-b[i]).exp());
            assert!(
                (g[i] - eg).abs() < 1e-5 && (beta[i] - eb).abs() < 1e-5,
                "i={i} g={} want={eg} beta={} want={eb}",
                g[i],
                beta[i]
            );
        }
    }

    /// l2_norm 对拍(含 >256 大 dim 走 block 变体)
    #[test]
    fn l2_norm_f32_host_parity() {
        let (dev, mut k, pool) = setup();
        for dim in [64usize, 512usize] {
            let rows = 3usize;
            let input: Vec<f32> = (0..rows * dim).map(|i| ((i % 37) as f32 - 18.0) * 0.1).collect();
            let out_buf = htod(&dev, &pool, vec![0f32; rows * dim]);
            let d_in = htod(&dev, &pool, input.clone());
            let in_p = owl_iface::DevBuf::device_ptr(&d_in) as *const u8;
            let out_p = owl_iface::DevBuf::device_ptr(&out_buf) as *mut u8;
            let stream = dev.stream();
            k.l2_norm(
                stream,
                "f32",
                in_p,
                out_p,
                rows as i32,
                dim as i32,
                1e-6,
            )
            .unwrap();
            dev.ctx().synchronize().unwrap();
            let got = dtoh_f32(&dev, out_p as *const f32, rows * dim);
            for r in 0..rows {
                let sumsq: f32 = input[r * dim..(r + 1) * dim].iter().map(|v| v * v).sum();
                let inv = 1.0 / (sumsq.max(0.0) + 1e-6).sqrt();
                for c in 0..dim {
                    let want = input[r * dim + c] * inv;
                    assert!(
                        (got[r * dim + c] - want).abs() < 1e-4,
                        "dim={dim} r={r} c={c} got={} want={want}",
                        got[r * dim + c]
                    );
                }
            }
        }
    }

    /// gated rmsnorm + silu/sigmoid(z) mul 对拍(含 per-group 权重 + bias)
    #[test]
    #[ignore = "K3 第二刀: rmsnorm_act 参数语义对齐 deltanet 调用点后启用"]
    fn rmsnorm_act_f32_host_parity() {
        let (dev, mut k, pool) = setup();
        let rows = 2usize;
        let heads = 4usize;
        let group = 32usize;
        let vdim = heads * group;
        let x: Vec<f32> = (0..rows * vdim).map(|i| ((i % 29) as f32 - 14.0) * 0.2).collect();
        let z: Vec<f32> = (0..rows * vdim).map(|i| ((i % 13) as f32 - 6.0) * 0.4).collect();
        let gamma: Vec<f32> = (0..group).map(|i| 0.9 + 0.01 * i as f32).collect(); // per-group
        let bias: Vec<f32> = (0..group).map(|i| -0.02 * i as f32).collect();

        for act in [0i32, 1i32] {
            let out_buf = htod(&dev, &pool, vec![0f32; rows * vdim]);
            let d_x = htod(&dev, &pool, x.clone());
            let d_z = htod(&dev, &pool, z.clone());
            let d_g = htod(&dev, &pool, gamma.clone());
            let d_b = htod(&dev, &pool, bias.clone());
            let x_p = owl_iface::DevBuf::device_ptr(&d_x) as *const u8;
            let z_p = owl_iface::DevBuf::device_ptr(&d_z) as *const u8;
            let g_p = owl_iface::DevBuf::device_ptr(&d_g) as *const u8;
            let bias_p = owl_iface::DevBuf::device_ptr(&d_b) as *const u8;
            let out_p = owl_iface::DevBuf::device_ptr(&out_buf) as *mut u8;
            let stream = dev.stream();
            k.rmsnorm_act(
                stream,
                "f32",
                x_p,
                z_p,
                g_p,
                bias_p,
                out_p,
                rows as i32,
                vdim as i32,
                group as i32,
                1e-6,
                true,
                true,
                act,
            )
            .unwrap();
            dev.ctx().synchronize().unwrap();
            let got = dtoh_f32(&dev, out_p as *const f32, rows * vdim);
            for r in 0..rows {
                for grp in 0..heads {
                    let off = r * vdim + grp * group;
                    let sumsq: f32 = x[off..off + group].iter().map(|v| v * v).sum();
                    let inv = (sumsq / group as f32 + 1e-6).sqrt().powi(-1);
                    for c in 0..group {
                        let y = x[off + c] * inv * gamma[c] + bias[c];
                        let zv = z[off + c];
                        let gate = if act == 1 {
                            1.0 / (1.0 + (-zv).exp())
                        } else {
                            zv / (1.0 + (-zv).exp())
                        };
                        let want = y * gate;
                        assert!(
                            (got[off + c] - want).abs() < 1e-4,
                            "act={act} r={r} grp={grp} c={c} got={} want={want} (got[off..off+group]={:?})",
                            got[off + c],
                            &got[off..off + group]
                        );
                    }
                }
            }
        }
    }
}

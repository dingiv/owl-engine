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
        self.launch_smem(stream, name, grid, block, 0, args)
    }

    fn launch_smem(
        &mut self,
        stream: &CudaStream,
        name: &'static str,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem_bytes: u32,
        args: &[u64],
    ) -> Result<(), String> {
        let func = self.func(name)?;
        let cfg = LaunchConfig {
            grid_dim: grid,
            block_dim: block,
            shared_mem_bytes,
        };
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

    /// gated rmsnorm + act(z)(silu/sigmoid)+ mul。
    /// 语义(二刀定谳,对齐 deltanet 调用点):gamma/bias 恒 f32;
    /// per_group_weights=true → [group_size] 组内共享(生产加载形态),
    /// false → 全长 [value_dim];bias 可空(has_bias=false 时传 null)。
    #[allow(clippy::too_many_arguments)]
    pub fn rmsnorm_act(
        &mut self,
        stream: &CudaStream,
        dtype: &'static str, // "f32" | "f16" | "bf16"
        x: *const u8,
        z: *const u8,
        gamma: *const f32,
        bias: *const f32, // 可空
        out: *mut u8,
        rows: i32,
        value_dim: i32,
        group_size: i32,
        eps: f32,
        per_group_weights: bool,
        has_bias: bool,
        act: i32, // 0=silu(Qwen3.5) 1=sigmoid(Qwen4)
    ) -> Result<(), String> {
        let name: &'static str = match dtype {
            "f32" => "gdn_rmsnorm_act_f32",
            "f16" => "gdn_rmsnorm_act_f16",
            "bf16" => "gdn_rmsnorm_act_bf16",
            _ => return Err(format!("rmsnorm_act: 未知 dtype {dtype}")),
        };
        let num_groups = value_dim / group_size;
        let blocks = (rows * num_groups).max(1) as u32;
        self.launch(
            stream,
            name,
            (blocks, 1, 1),
            (256, 1, 1),
            &[
                x as u64,
                z as u64,
                gamma as u64,
                bias as u64,
                out as u64,
                rows as u64,
                value_dim as u64,
                group_size as u64,
                eps.to_bits() as u64,
                per_group_weights as u64,
                has_bias as u64,
                act as u64,
            ],
        )
    }

    /// causal_conv1d prefill(变长,kernel=4 收窄):因果卷积 + silu 可选,
    /// conv_state [batch, d_conv, 3] 恒F32 就地更新为各序列末 3 个输入;
    /// cu_seqlens [batch+1] u32 前缀和。x/out [total_tokens, d_conv]。
    /// MTP 快照(state_snapshots)未搬——T3 接线时按需求补。
    #[allow(clippy::too_many_arguments)]
    pub fn conv1d_fwd_k4(
        &mut self,
        stream: &CudaStream,
        dtype: &'static str,
        x: *const u8,
        weight: *const u8,
        bias: *const u8, // 可空
        conv_state: *mut f32,
        out: *mut u8,
        cu_seqlens: *const u32,
        batch: i32,
        d_conv: i32,
        silu: bool,
    ) -> Result<(), String> {
        let name: &'static str = match dtype {
            "f32" => "gdn_conv1d_fwd_k4_f32",
            "f16" => "gdn_conv1d_fwd_k4_f16",
            "bf16" => "gdn_conv1d_fwd_k4_bf16",
            _ => return Err(format!("conv1d_fwd_k4: 未知 dtype {dtype}")),
        };
        let grid_y = (d_conv as u32).div_ceil(256).max(1);
        self.launch(
            stream,
            name,
            (batch.max(1) as u32, grid_y, 1),
            (256, 1, 1),
            &[
                x as u64,
                weight as u64,
                bias as u64,
                conv_state as u64,
                out as u64,
                cu_seqlens as u64,
                batch as u64,
                d_conv as u64,
                silu as u64,
            ],
        )
    }

    /// causal_conv1d decode 单步(kernel=4 收窄):slot 寻址滑窗更新。
    /// conv_state [max_batch, d_conv, 3] 恒F32;slots [batch] i64(<0 跳过);
    /// x/out [batch, d_conv]。total = batch*d_conv。
    #[allow(clippy::too_many_arguments)]
    pub fn conv1d_update_slots_k4(
        &mut self,
        stream: &CudaStream,
        dtype: &'static str,
        x: *const u8,
        weight: *const u8,
        bias: *const u8, // 可空
        conv_state: *mut f32,
        slots: *const i64,
        out: *mut u8,
        batch: i32,
        d_conv: i32,
        silu: bool,
    ) -> Result<(), String> {
        let name: &'static str = match dtype {
            "f32" => "gdn_conv1d_upd_k4_f32",
            "f16" => "gdn_conv1d_upd_k4_f16",
            "bf16" => "gdn_conv1d_upd_k4_bf16",
            _ => return Err(format!("conv1d_update_slots_k4: 未知 dtype {dtype}")),
        };
        let total = batch * d_conv;
        let blocks = (total as u32).div_ceil(256).max(1);
        self.launch(
            stream,
            name,
            (blocks, 1, 1),
            (256, 1, 1),
            &[
                x as u64,
                weight as u64,
                bias as u64,
                conv_state as u64,
                slots as u64,
                out as u64,
                total as u64,
                d_conv as u64,
                silu as u64,
            ],
        )
    }

    /// gated_delta_rule prefill 递推(fallback 变体,任意 k_dim ≤ 256)。
    /// g 为**实空间 decay(已 exp)**;beta in (0,1)。
    /// q/k [BH,S,K],v/out [BH,S,V],state [BH,K,V] 恒F32 in/out。
    /// 档位 BV=64;smem = (2k+2)×4B。tiled/varlen 未搬(见汇报)。
    #[allow(clippy::too_many_arguments)]
    pub fn delta_recurrence_fallback(
        &mut self,
        stream: &CudaStream,
        dtype: &'static str,
        q: *const u8,
        k: *const u8,
        v: *const u8,
        g: *const f32,
        beta: *const f32,
        state: *mut f32,
        out: *mut f32,
        bh: i32,
        seq_len: i32,
        k_dim: i32,
        v_dim: i32,
    ) -> Result<(), String> {
        if k_dim > 256 {
            return Err(format!("delta_recurrence_fallback: k_dim={k_dim} > 256"));
        }
        let name: &'static str = match dtype {
            "f32" => "gdn_delta_rec_fb_f32",
            "f16" => "gdn_delta_rec_fb_f16",
            "bf16" => "gdn_delta_rec_fb_bf16",
            _ => return Err(format!("delta_recurrence_fallback: 未知 dtype {dtype}")),
        };
        let grid_x = (v_dim as u32).div_ceil(64).max(1);
        let smem = ((2 * k_dim + 2) * 4) as u32;
        self.launch_smem(
            stream,
            name,
            (grid_x, bh.max(1) as u32, 1),
            (64, 1, 1),
            smem,
            &[
                q as u64,
                k as u64,
                v as u64,
                g as u64,
                beta as u64,
                state as u64,
                out as u64,
                seq_len as u64,
                k_dim as u64,
                v_dim as u64,
            ],
        )
    }

    /// gated_delta_rule decode 单步(slot 寻址,GQA 映射)。k_dim ≤ 128(BK=128)。
    /// g 为 **log 空间(核内 exp)**;q 核内乘 q_scale。
    /// q/k [B,num_k_heads,K],v/out [B,num_v_heads,V],
    /// state [max_batch,num_v_heads,K,V] 恒F32 in/out,slots [batch] i64(<0 跳过)。
    /// 档位 BV=64;smem = (2×128+2)×4B = 1032B。
    #[allow(clippy::too_many_arguments)]
    pub fn delta_decode_slots_gqa(
        &mut self,
        stream: &CudaStream,
        dtype: &'static str,
        q: *const u8,
        k: *const u8,
        v: *const u8,
        g: *const f32,
        beta: *const f32,
        state: *mut f32,
        slots: *const i64,
        out: *mut u8,
        batch: i32,
        num_v_heads: i32,
        num_k_heads: i32,
        k_dim: i32,
        v_dim: i32,
        q_scale: f32,
    ) -> Result<(), String> {
        if k_dim > 128 {
            return Err(format!("delta_decode_slots_gqa: k_dim={k_dim} > 128"));
        }
        let name: &'static str = match dtype {
            "f32" => "gdn_delta_dec_gqa_f32",
            "f16" => "gdn_delta_dec_gqa_f16",
            "bf16" => "gdn_delta_dec_gqa_bf16",
            _ => return Err(format!("delta_decode_slots_gqa: 未知 dtype {dtype}")),
        };
        let grid_x = (v_dim as u32).div_ceil(64).max(1);
        let grid_y = (batch * num_v_heads).max(1) as u32;
        self.launch_smem(
            stream,
            name,
            (grid_x, grid_y, 1),
            (64, 1, 1),
            (2 * 128 + 2) * 4,
            &[
                q as u64,
                k as u64,
                v as u64,
                g as u64,
                beta as u64,
                state as u64,
                slots as u64,
                out as u64,
                batch as u64,
                num_v_heads as u64,
                num_k_heads as u64,
                k_dim as u64,
                v_dim as u64,
                q_scale.to_bits() as u64,
            ],
        )
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
    /// 语义 = deltanet 调用点:gamma/bias [group_size] 组内共享,恒 f32
    #[test]
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
            let g_p = owl_iface::DevBuf::device_ptr(&d_g) as *const f32;
            let bias_p = owl_iface::DevBuf::device_ptr(&d_b) as *const f32;
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

    /// causal_conv1d prefill 对拍(k=4,变长,bias+silu,末态写回)
    #[test]
    fn conv1d_fwd_k4_f32_host_parity() {
        let (dev, mut k, pool) = setup();
        let lens = [3usize, 1usize];
        let batch = lens.len();
        let total: usize = lens.iter().sum();
        let d_conv = 6usize;
        let mut cu = vec![0u32];
        for l in lens {
            cu.push(cu.last().unwrap() + l as u32);
        }
        let x: Vec<f32> = (0..total * d_conv).map(|i| ((i % 19) as f32 - 9.0) * 0.25).collect();
        let weight: Vec<f32> = (0..d_conv * 4).map(|i| ((i % 11) as f32 - 5.0) * 0.2).collect();
        let bias: Vec<f32> = (0..d_conv).map(|i| 0.03 * i as f32).collect();
        let state0: Vec<f32> = (0..batch * d_conv * 3).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect();

        let d_x = htod(&dev, &pool, x.clone());
        let d_w = htod(&dev, &pool, weight.clone());
        let d_b = htod(&dev, &pool, bias.clone());
        let d_cu = htod(&dev, &pool, cu.clone());
        let d_state = htod(&dev, &pool, state0.clone());
        let out_buf = htod(&dev, &pool, vec![0f32; total * d_conv]);
        let x_p = owl_iface::DevBuf::device_ptr(&d_x) as *const u8;
        let w_p = owl_iface::DevBuf::device_ptr(&d_w) as *const u8;
        let b_p = owl_iface::DevBuf::device_ptr(&d_b) as *const u8;
        let cu_p = owl_iface::DevBuf::device_ptr(&d_cu) as *const u32;
        let st_p = owl_iface::DevBuf::device_ptr(&d_state) as *mut f32;
        let out_p = owl_iface::DevBuf::device_ptr(&out_buf) as *mut u8;

        k.conv1d_fwd_k4(dev.stream(), "f32", x_p, w_p, b_p, st_p, out_p, cu_p,
                        batch as i32, d_conv as i32, true)
            .unwrap();
        dev.ctx().synchronize().unwrap();

        let got = dtoh_f32(&dev, out_p as *const f32, total * d_conv);
        let got_state = dtoh_f32(&dev, st_p, batch * d_conv * 3);
        let silu = |v: f32| v / (1.0 + (-v).exp());
        let mut want_state = state0.clone();
        for b in 0..batch {
            let (s, e) = (cu[b] as usize, cu[b + 1] as usize);
            for ch in 0..d_conv {
                let mut hist: Vec<f32> =
                    state0[(b * d_conv + ch) * 3..(b * d_conv + ch) * 3 + 3].to_vec();
                for t in s..e {
                    let x_t = x[t * d_conv + ch];
                    let mut sum = x_t * weight[ch * 4 + 3];
                    for j in 0..3 {
                        sum += hist[j] * weight[ch * 4 + j];
                    }
                    sum += bias[ch];
                    let want = silu(sum);
                    let go = got[t * d_conv + ch];
                    assert!((go - want).abs() < 1e-4,
                        "b={b} t={t} ch={ch} got={go} want={want}");
                    hist = vec![hist[1], hist[2], x_t];
                }
                let off = (b * d_conv + ch) * 3;
                want_state[off..off + 3].copy_from_slice(&hist);
            }
        }
        for (i, (g, w)) in got_state.iter().zip(&want_state).enumerate() {
            assert!((g - w).abs() < 1e-6, "state[{i}] got={g} want={w}");
        }
    }

    /// causal_conv1d decode 单步对拍(k=4,slot 寻址,含 slot<0 跳过)
    #[test]
    fn conv1d_update_slots_k4_f32_host_parity() {
        let (dev, mut k, pool) = setup();
        let batch = 3usize;
        let max_batch = 5usize;
        let d_conv = 4usize;
        let slots: Vec<i64> = vec![2, -1, 0]; // slot 3 闲置,slot -1 跳过
        let x: Vec<f32> = (0..batch * d_conv).map(|i| ((i % 13) as f32 - 6.0) * 0.4).collect();
        let weight: Vec<f32> = (0..d_conv * 4).map(|i| ((i % 9) as f32 - 4.0) * 0.3).collect();
        let state0: Vec<f32> = (0..max_batch * d_conv * 3).map(|i| ((i % 17) as f32 - 8.0) * 0.3).collect();

        let d_x = htod(&dev, &pool, x.clone());
        let d_w = htod(&dev, &pool, weight.clone());
        let d_cu_slots = htod(&dev, &pool, slots.clone());
        let d_state = htod(&dev, &pool, state0.clone());
        let out_buf = htod(&dev, &pool, vec![0f32; batch * d_conv]);
        let x_p = owl_iface::DevBuf::device_ptr(&d_x) as *const u8;
        let w_p = owl_iface::DevBuf::device_ptr(&d_w) as *const u8;
        let sl_p = owl_iface::DevBuf::device_ptr(&d_cu_slots) as *const i64;
        let st_p = owl_iface::DevBuf::device_ptr(&d_state) as *mut f32;
        let out_p = owl_iface::DevBuf::device_ptr(&out_buf) as *mut u8;

        k.conv1d_update_slots_k4(dev.stream(), "f32", x_p, w_p, std::ptr::null(), st_p,
                                 sl_p, out_p, batch as i32, d_conv as i32, true)
            .unwrap();
        dev.ctx().synchronize().unwrap();

        let got = dtoh_f32(&dev, out_p as *const f32, batch * d_conv);
        let got_state = dtoh_f32(&dev, st_p, max_batch * d_conv * 3);
        let silu = |v: f32| v / (1.0 + (-v).exp());
        let mut want_state = state0.clone();
        for b in 0..batch {
            let slot = slots[b];
            if slot < 0 {
                continue;
            }
            for ch in 0..d_conv {
                let off = ((slot as usize) * d_conv + ch) * 3;
                let hist: Vec<f32> = state0[off..off + 3].to_vec();
                let x_t = x[b * d_conv + ch];
                let mut sum = x_t * weight[ch * 4 + 3];
                for j in 0..3 {
                    sum += hist[j] * weight[ch * 4 + j];
                }
                sum = silu(sum);
                let go = got[b * d_conv + ch];
                assert!((go - sum).abs() < 1e-4, "b={b} ch={ch} got={go} want={sum}");
                want_state[off..off + 3].copy_from_slice(&[hist[1], hist[2], x_t]);
            }
        }
        for (i, (g, w)) in got_state.iter().zip(&want_state).enumerate() {
            assert!((g - w).abs() < 1e-6, "state[{i}] got={g} want={w}");
        }
    }

    /// gated_delta_rule prefill 递推对拍(fallback,g 实空间 decay)
    #[test]
    fn delta_recurrence_fallback_f32_host_parity() {
        let (dev, mut k, pool) = setup();
        let bh = 2usize;
        let seq = 3usize;
        let k_dim = 4usize; // < 128,验证非 0.8B 档也能走 fallback
        let v_dim = 8usize;
        let q: Vec<f32> = (0..bh * seq * k_dim).map(|i| ((i % 5) as f32 - 2.0) * 0.3).collect();
        let kv: Vec<f32> = (0..bh * seq * k_dim).map(|i| ((i % 7) as f32 - 3.0) * 0.2).collect();
        let v: Vec<f32> = (0..bh * seq * v_dim).map(|i| ((i % 11) as f32 - 5.0) * 0.25).collect();
        let g: Vec<f32> = (0..bh * seq).map(|i| 0.9 + 0.01 * (i % 3) as f32).collect(); // 实空间 decay
        let beta: Vec<f32> = (0..bh * seq).map(|i| 0.5 + 0.1 * (i % 2) as f32).collect();
        let state0: Vec<f32> = (0..bh * k_dim * v_dim).map(|i| ((i % 23) as f32 - 11.0) * 0.1).collect();

        let d_q = htod(&dev, &pool, q.clone());
        let d_k = htod(&dev, &pool, kv.clone());
        let d_v = htod(&dev, &pool, v.clone());
        let d_g = htod(&dev, &pool, g.clone());
        let d_beta = htod(&dev, &pool, beta.clone());
        let d_state = htod(&dev, &pool, state0.clone());
        let out_buf = htod(&dev, &pool, vec![0f32; bh * seq * v_dim]);
        let q_p = owl_iface::DevBuf::device_ptr(&d_q) as *const u8;
        let k_p = owl_iface::DevBuf::device_ptr(&d_k) as *const u8;
        let v_p = owl_iface::DevBuf::device_ptr(&d_v) as *const u8;
        let g_p = owl_iface::DevBuf::device_ptr(&d_g) as *const f32;
        let beta_p = owl_iface::DevBuf::device_ptr(&d_beta) as *const f32;
        let st_p = owl_iface::DevBuf::device_ptr(&d_state) as *mut f32;
        let out_p = owl_iface::DevBuf::device_ptr(&out_buf) as *mut f32;

        k.delta_recurrence_fallback(dev.stream(), "f32", q_p, k_p, v_p, g_p, beta_p,
                                    st_p, out_p, bh as i32, seq as i32,
                                    k_dim as i32, v_dim as i32)
            .unwrap();
        dev.ctx().synchronize().unwrap();

        let got = dtoh_f32(&dev, out_p, bh * seq * v_dim);
        let got_state = dtoh_f32(&dev, st_p, bh * k_dim * v_dim);
        let mut want_state = state0.clone();
        for h in 0..bh {
            // host 参考:fallback 公式直译
            let mut s = vec![0f32; k_dim * v_dim];
            s.copy_from_slice(&state0[h * k_dim * v_dim..(h + 1) * k_dim * v_dim]);
            for t in 0..seq {
                for j in 0..k_dim {
                    let decay = g[h * seq + t];
                    for vi in 0..v_dim {
                        s[j * v_dim + vi] *= decay;
                    }
                }
                for vi in 0..v_dim {
                    let mut kv_mem = 0.0f32;
                    for j in 0..k_dim {
                        kv_mem += s[j * v_dim + vi] * kv[h * seq * k_dim + t * k_dim + j];
                    }
                    let delta =
                        (v[h * seq * v_dim + t * v_dim + vi] - kv_mem) * beta[h * seq + t];
                    for j in 0..k_dim {
                        s[j * v_dim + vi] +=
                            kv[h * seq * k_dim + t * k_dim + j] * delta;
                    }
                }
                for vi in 0..v_dim {
                    let mut y = 0.0f32;
                    for j in 0..k_dim {
                        y += s[j * v_dim + vi] * q[h * seq * k_dim + t * k_dim + j];
                    }
                    let go = got[h * seq * v_dim + t * v_dim + vi];
                    assert!((go - y).abs() < 1e-4,
                        "h={h} t={t} vi={vi} got={go} want={y}");
                }
            }
            want_state[h * k_dim * v_dim..(h + 1) * k_dim * v_dim].copy_from_slice(&s);
        }
        for (i, (a, b)) in got_state.iter().zip(&want_state).enumerate() {
            assert!((a - b).abs() < 1e-5, "state[{i}] got={a} want={b}");
        }
    }

    /// gated_delta_rule decode 对拍(slot GQA,g log 空间,q_scale)
    #[test]
    fn delta_decode_slots_gqa_f32_host_parity() {
        let (dev, mut k, pool) = setup();
        let batch = 2usize;
        let nv = 4usize;
        let nk = 2usize; // kv_group = 2,验证 GQA 头映射
        let k_dim = 8usize; // < 128,BK=128 档内 j<k_dim 守护
        let v_dim = 8usize;
        let slots: Vec<i64> = vec![1, 0];
        let max_batch = 3usize;
        let q: Vec<f32> = (0..batch * nk * k_dim).map(|i| ((i % 7) as f32 - 3.0) * 0.2).collect();
        let kv: Vec<f32> = (0..batch * nk * k_dim).map(|i| ((i % 5) as f32 - 2.0) * 0.3).collect();
        let v: Vec<f32> = (0..batch * nv * v_dim).map(|i| ((i % 13) as f32 - 6.0) * 0.15).collect();
        let g: Vec<f32> = (0..batch * nv).map(|i| -0.02 * (i % 4) as f32).collect(); // log 空间
        let beta: Vec<f32> = (0..batch * nv).map(|i| 0.6 + 0.05 * (i % 3) as f32).collect();
        let state0: Vec<f32> =
            (0..max_batch * nv * k_dim * v_dim).map(|i| ((i % 19) as f32 - 9.0) * 0.2).collect();
        let q_scale = 0.5f32;

        let d_q = htod(&dev, &pool, q.clone());
        let d_k = htod(&dev, &pool, kv.clone());
        let d_v = htod(&dev, &pool, v.clone());
        let d_g = htod(&dev, &pool, g.clone());
        let d_beta = htod(&dev, &pool, beta.clone());
        let d_state = htod(&dev, &pool, state0.clone());
        let d_slots = htod(&dev, &pool, slots.clone());
        let out_buf = htod(&dev, &pool, vec![0f32; batch * nv * v_dim]);
        let q_p = owl_iface::DevBuf::device_ptr(&d_q) as *const u8;
        let k_p = owl_iface::DevBuf::device_ptr(&d_k) as *const u8;
        let v_p = owl_iface::DevBuf::device_ptr(&d_v) as *const u8;
        let g_p = owl_iface::DevBuf::device_ptr(&d_g) as *const f32;
        let beta_p = owl_iface::DevBuf::device_ptr(&d_beta) as *const f32;
        let st_p = owl_iface::DevBuf::device_ptr(&d_state) as *mut f32;
        let sl_p = owl_iface::DevBuf::device_ptr(&d_slots) as *const i64;
        let out_p = owl_iface::DevBuf::device_ptr(&out_buf) as *mut u8;

        k.delta_decode_slots_gqa(dev.stream(), "f32", q_p, k_p, v_p, g_p, beta_p,
                                 st_p, sl_p, out_p, batch as i32, nv as i32,
                                 nk as i32, k_dim as i32, v_dim as i32, q_scale)
            .unwrap();
        dev.ctx().synchronize().unwrap();

        let got = dtoh_f32(&dev, out_p as *const f32, batch * nv * v_dim);
        let got_state = dtoh_f32(&dev, st_p, max_batch * nv * k_dim * v_dim);
        let mut want_state = state0.clone();
        for b in 0..batch {
            let slot = slots[b] as usize;
            for vh in 0..nv {
                let kh = vh / (nv / nk);
                let decay = g[b * nv + vh].exp();
                let beta_t = beta[b * nv + vh];
                let mut s = vec![0f32; k_dim * v_dim];
                let soff = (slot * nv + vh) * k_dim * v_dim;
                s.copy_from_slice(&state0[soff..soff + k_dim * v_dim]);
                for vi in 0..v_dim {
                    let mut kv_mem = 0.0f32;
                    for j in 0..k_dim {
                        s[j * v_dim + vi] *= decay;
                        kv_mem += s[j * v_dim + vi] * kv[(b * nk + kh) * k_dim + j];
                    }
                    let delta =
                        (v[(b * nv + vh) * v_dim + vi] - kv_mem) * beta_t;
                    let mut y = 0.0f32;
                    for j in 0..k_dim {
                        s[j * v_dim + vi] += kv[(b * nk + kh) * k_dim + j] * delta;
                        y += s[j * v_dim + vi]
                            * q[(b * nk + kh) * k_dim + j]
                            * q_scale;
                    }
                    let go = got[(b * nv + vh) * v_dim + vi];
                    assert!((go - y).abs() < 1e-4,
                        "b={b} vh={vh} vi={vi} got={go} want={y}");
                }
                want_state[soff..soff + k_dim * v_dim].copy_from_slice(&s);
            }
        }
        for (i, (a, b2)) in got_state.iter().zip(&want_state).enumerate() {
            assert!((a - b2).abs() < 1e-5, "state[{i}] got={a} want={b2}");
        }
    }
}

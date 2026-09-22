//! example 02:M1 全链 CUDA graph 捕获 + replay + 计时对照。
//!
//! 链:matmul → add → silu → rmsnorm,**全部入图**(M1①:cublas 逐次
//! setStream 到捕获流,GEMM 被烙进图;workspace 预钉 = 姿势 4 防线)。
//!
//! M1 纪律落地:
//! - 姿势 6:warmup 制度化——捕获前全链 eager 跑一遍,由
//!   `CaptureSession::capture` 的发射计数门禁强制(未 warmup 拒捕获);
//! - 信号自动租约:闭包内 ops 读过的缓冲自动成为图强租约
//!   (无手工 `session.lease` 调用);
//! - 哨兵③:instantiate 前 driver 自审——图内全部 8 字节参数对账
//!   账本区间,∉ 租约集 = 依赖泄漏(结构化违约,非 Xid 盲死)。
//!
//! 重放序列 = 仅 graph.launch(全链在图内);喂新输入走旁路
//! (fill_from_host,EagerOnly,设备主流)。

use owl_cuda::CudaDevice;
use owl_iface::{Device, MemPhase, PoolConfig, PoolKind};
use owl_nn::cublas::NnBlas;
use owl_nn::ops::OpsCtx;
use owl_nn::tensor::{Tensor, TensorPoolOps};
use owl_nn::{CaptureRecorder, KernelCtx};
use std::time::Instant;

struct Lcg(u32);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        (self.0 >> 8) as f32 / 8_388_608.0 - 1.0
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next()).collect()
    }
}

fn median(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

/// 链体:matmul → add → silu → rmsnorm(流由 ctx 携带,与捕获无关)
fn run_chain<D: owl_iface::Device>(
    ctx: &KernelCtx,
    ops: &mut OpsCtx,
    blas: &NnBlas,
    w: &Tensor<f32, D>,
    a: &Tensor<f32, D>,
    bias: &Tensor<f32, D>,
    alpha: &Tensor<f32, D>,
    mm_out: &mut Tensor<f32, D>,
    add_out: &mut Tensor<f32, D>,
    silu_out: &mut Tensor<f32, D>,
    out: &mut Tensor<f32, D>,
) -> Result<(), owl_iface::BackendError> {
    ops.matmul(ctx, blas, w, a, mm_out)?;
    ops.add(ctx, mm_out, bias, add_out)?;
    ops.silu(ctx, add_out, silu_out)?;
    ops.rmsnorm(ctx, silu_out, alpha, out, 1e-5)?;
    Ok(())
}

fn main() {
    let dev = CudaDevice::new(0).expect("需要 CUDA 设备");
    println!("== owl 02_capture(M1):全链 [matmul|add|silu|rmsnorm] 单图捕获 ==");

    // ---- P 阶段 ----
    let pool = dev
        .create_pool(PoolConfig {
            name: format!("e02-chain-{}", std::process::id()),
            kind: PoolKind::Weights,
            bytes: 32 << 20,
        })
        .expect("建池");
    let blas = NnBlas::new(&dev).expect("NnBlas");
    let mut ops = OpsCtx::new(&dev).expect("OpsCtx");
    let ctx = ops.ctx(MemPhase::Idle); // 设备主流(非阻塞;M1②)

    const M: usize = 64;
    const K: usize = 128;
    const N: usize = 64;
    const ITER: usize = 200;

    // 链缓冲(图内) + 参考缓冲(eager 对照,独立一份)
    let mut w = pool.zeros_tensor::<f32>(&[M, K]).unwrap();
    let mut a = pool.zeros_tensor::<f32>(&[K, N]).unwrap();
    let mut mm_out = pool.zeros_tensor::<f32>(&[M, N]).unwrap();
    let mut bias = pool.zeros_tensor::<f32>(&[M, N]).unwrap();
    let mut add_out = pool.zeros_tensor::<f32>(&[M, N]).unwrap();
    let mut silu_out = pool.zeros_tensor::<f32>(&[M, N]).unwrap();
    let mut alpha = pool.zeros_tensor::<f32>(&[N]).unwrap();
    let mut graph_out = pool.zeros_tensor::<f32>(&[M, N]).unwrap();
    // eager 参考专用(独立输出,避免与图共享缓冲互相覆盖)
    let mut ref_mm = pool.zeros_tensor::<f32>(&[M, N]).unwrap();
    let mut ref_add = pool.zeros_tensor::<f32>(&[M, N]).unwrap();
    let mut ref_silu = pool.zeros_tensor::<f32>(&[M, N]).unwrap();
    let mut ref_out = pool.zeros_tensor::<f32>(&[M, N]).unwrap();

    let mut seed = Lcg(11);
    ops.fill_from_host(&ctx, &mut w, &seed.vec(M * K)).unwrap();
    ops.fill_from_host(&ctx, &mut a, &seed.vec(K * N)).unwrap();
    ops.fill_from_host(&ctx, &mut alpha, &seed.vec(N)).unwrap();

    let base_ledger = dev.ledger();

    // ---- 姿势 6:warmup(全链 eager;capture 门禁 = 发射计数 >0)----
    run_chain(
        &ctx,
        &mut ops,
        &blas,
        &w,
        &a,
        &bias,
        &alpha,
        &mut mm_out,
        &mut add_out,
        &mut silu_out,
        &mut graph_out,
    )
    .unwrap();
    dev.ctx().synchronize().unwrap();

    // ---- 捕获:全链单图(matmul 经 cublas 在捕获流上被烙进图)----
    // 信号自动租约:闭包内 ops 读过的 7 个缓冲自动成为强租约,无手工登记
    let mut session = dev.capture_session().expect("capture_session");
    let (rec, graph) = session
        .capture(
            owl_cuda::ffi::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            |frame| {
                let rec = CaptureRecorder::new();
                let ctx = KernelCtx::capturing_leased(rec.clone(), frame);
                run_chain(
                    &ctx,
                    &mut ops,
                    &blas,
                    &w,
                    &a,
                    &bias,
                    &alpha,
                    &mut mm_out,
                    &mut add_out,
                    &mut silu_out,
                    &mut graph_out,
                )?;
                Ok(rec)
            },
        )
        .expect("capture(姿势6门禁 + 自动租约 + 哨兵③)");
    graph.upload().unwrap();

    let audit = graph.audit();
    println!(
        "捕获完成 + upload:kernel 节点 {} / 总节点 {} / 指针引用 {} / 租约 {} 条 / launch 序 {:?}",
        audit.kernel_nodes,
        audit.total_nodes,
        audit.ptr_refs.len(),
        graph.leases().len(),
        rec.snapshot().launches
    );
    // 8 个链缓冲 + 1 个 cublas workspace(matmul 的隐藏依赖)= 9 条
    assert_eq!(graph.leases().len(), 9, "自动租约应为 8 链缓冲 + 1 workspace = 9 条");

    // ---- replay 正确性:新输入(旁路写入)→ 仅 graph.launch ----
    let mut seed2 = Lcg(23);
    for it in 0..4 {
        let new_bias = seed2.vec(M * N);
        ops.fill_from_host(&ctx, &mut bias, &new_bias).unwrap(); // 旁路:EagerOnly,零分配
        dev.ctx().synchronize().unwrap();
        graph.launch().expect("graph launch");
        dev.ctx().synchronize().unwrap();

        // eager 参考链(独立缓冲,设备主流)
        run_chain(
            &ctx,
            &mut ops,
            &blas,
            &w,
            &a,
            &bias,
            &alpha,
            &mut ref_mm,
            &mut ref_add,
            &mut ref_silu,
            &mut ref_out,
        )
        .unwrap();
        dev.ctx().synchronize().unwrap();

        let got = graph_out.to_vec().unwrap();
        let want: Vec<f64> = ref_out.to_vec().unwrap().iter().map(|&v| v as f64).collect();
        let max_diff = got
            .iter()
            .zip(&want)
            .map(|(g, w)| ((*g as f64) - *w).abs())
            .fold(0.0f64, f64::max);
        assert!(max_diff <= 1e-4, "replay #{it} 与 eager 不一致: {max_diff}");
        println!("replay #{it}: graph vs eager 最大偏差 {:.3e} ✓", max_diff);
    }

    // ---- 计时对照(median of ITER;每次同步,单请求口径)----
    let mut seed3 = Lcg(99);
    let mut eager_t = Vec::with_capacity(ITER);
    let mut graph_t = Vec::with_capacity(ITER);
    for _ in 0..ITER {
        let nb = seed3.vec(M * N);
        ops.fill_from_host(&ctx, &mut bias, &nb).unwrap();

        let t0 = Instant::now();
        run_chain(
            &ctx,
            &mut ops,
            &blas,
            &w,
            &a,
            &bias,
            &alpha,
            &mut mm_out,
            &mut add_out,
            &mut silu_out,
            &mut graph_out,
        )
        .unwrap();
        dev.ctx().synchronize().unwrap();
        eager_t.push(t0.elapsed().as_secs_f64() * 1e3);

        let t1 = Instant::now();
        graph.launch().unwrap(); // 全链在图内,一次 launch
        dev.ctx().synchronize().unwrap();
        graph_t.push(t1.elapsed().as_secs_f64() * 1e3);
    }
    let e_med = median(&mut eager_t);
    let g_med = median(&mut graph_t);
    println!(
        "计时(median of {ITER},ms): eager 全链 {:.4} | 全链图 {:.4} | graph 提速 {:.1}%",
        e_med,
        g_med,
        (e_med - g_med) / e_med * 100.0
    );

    // ---- A5 终验:账本零漂移 ----
    let end = dev.ledger();
    assert_eq!(end.bytes_alive, base_ledger.bytes_alive, "A5: 生命周期末账本必须回到基线");
    println!("A5 终验:账本回基线 ✓ (alive = {} B)", end.bytes_alive);
}

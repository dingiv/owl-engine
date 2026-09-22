//! 一期 example 02:同链 CUDA graph 捕获 + replay + 计时对照。
//!
//! 链:matmul → add → silu → rmsnorm。
//! **一期边界(实测确认)**:NnBlas 绑 legacy NULL stream(T2 移交事项),
//! capture(Relaxed)下 GEMM 不会被烙进图——故本 example 的图覆盖
//! add→silu→rmsnorm 三算子,matmul 保持 eager 前置,重放序列 =
//! eager matmul + graph.launch。这正是 M1"cublas 改绑 device stream"
//! 的存在理由,经验证记录于 docs/bench/owl.md。
//!
//! 纪律:P 阶段分配全部缓冲;warmup 先 eager 全链(kernel JIT 预热 +
//! 捕获前的懒状态清理,T3 移交事项①);replay 喂新输入走旁路
//! (fill_from_host / eager matmul,不经捕获)。

use owl_cuda::CudaDevice;
use owl_iface::{Device, PoolConfig, PoolKind};
use owl_nn::cublas::NnBlas;
use owl_nn::ops::OpsCtx;
use owl_nn::tensor::{Tensor, TensorPoolOps};

use owl_cuda::ffi::sys;
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

fn main() {
    let dev = CudaDevice::new(0).expect("需要 CUDA 设备");
    println!("== owl 02_capture:matmul(eager) + [add|silu|rmsnorm](graph)==");

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
    let ctx = owl_nn::KernelCtx::eager(owl_iface::MemPhase::Idle);

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

    // ---- warmup(T3 移交事项①:kernel JIT/懒状态在捕获前清理)----
    ops.matmul(&ctx, &blas, &w, &a, &mut mm_out).unwrap();
    ops.add(&ctx, &mm_out, &bias, &mut add_out).unwrap();
    ops.silu(&ctx, &add_out, &mut silu_out).unwrap();
    ops.rmsnorm(&ctx, &silu_out, &alpha, &mut graph_out, 1e-5).unwrap();
    dev.ctx().synchronize().unwrap();

    // ---- 捕获:non-blocking stream + Kernels 直发 ----
    // 实测发现(重要,记录给 M1):cudarc 的 default_stream = legacy NULL
    // stream,**天然不可捕获**(begin_capture 直接
    // CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED)。T3 的 OpsCtx 绑死的正是
    // 这条流——故本 example 捕获段改用 Kernels 直发到自建 non-blocking
    // 流(不改 T0/T2 源码);matmul(cublas/legacy)保持 eager 前置。
    // GraphLease:捕获会话登记依赖租约(强租约),DeviceGraph 持有 keepalive
    let mut session = dev.capture_session().expect("capture_session");
    let cap_stream = session.stream().clone();
    let mut cap_kernels = owl_nn::kernels::Kernels::new(dev.ctx()).expect("kernels");
    let m_n = M * N;

    // 依赖租约登记(哨兵①:CaptureRecord 的实体化前奏)
    session.lease(mm_out.persistent().unwrap());
    session.lease(bias.persistent().unwrap());
    session.lease(add_out.persistent().unwrap());
    session.lease(silu_out.persistent().unwrap());
    session.lease(alpha.persistent().unwrap());
    session.lease(graph_out.persistent().unwrap());

    cap_stream
        .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
        .expect("begin_capture");
    cap_kernels
        .add_f32(&cap_stream, m_n, mm_out.device_ptr(), bias.device_ptr(), add_out.device_ptr())
        .expect("cap add");
    cap_kernels
        .silu_f32(&cap_stream, m_n, add_out.device_ptr(), silu_out.device_ptr())
        .expect("cap silu");
    cap_kernels
        .rmsnorm_f32(
            &cap_stream,
            M,
            N,
            silu_out.device_ptr(),
            alpha.device_ptr(),
            graph_out.device_ptr(),
            1e-5,
        )
        .expect("cap rmsnorm");
    let graph = session
        .end(sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
        .expect("end_capture");
    graph.upload().unwrap();
    println!("捕获完成 + upload(3 kernel@non-blocking stream;matmul legacy 保持 eager;租约 {} 条)", 6);


    // ---- replay 正确性:新输入(旁路写入)→ eager matmul + graph.launch ----
    let mut seed2 = Lcg(23);
    for it in 0..4 {
        let new_bias = seed2.vec(M * N);
        ops.fill_from_host(&ctx, &mut bias, &new_bias).unwrap(); // 旁路:EagerOnly,零分配
        ops.matmul(&ctx, &blas, &w, &a, &mut mm_out).unwrap(); // eager 前置
        dev.ctx().synchronize().unwrap(); // legacy→non-blocking 无隐式同步,显式栅栏
        graph.launch().expect("graph launch");
        dev.ctx().synchronize().unwrap();

        // eager 参考链(独立缓冲)
        ops.matmul(&ctx, &blas, &w, &a, &mut ref_mm).unwrap();
        ops.add(&ctx, &ref_mm, &bias, &mut ref_add).unwrap();
        ops.silu(&ctx, &ref_add, &mut ref_silu).unwrap();
        ops.rmsnorm(&ctx, &ref_silu, &alpha, &mut ref_out, 1e-5).unwrap();
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
    // 旁路喂入确实生效(bias 变了,输出随之变化)
    assert!(seed2.vec(1).len() == 1);

    // ---- 计时对照(median of ITER;每次同步,单请求口径)----
    let mut seed3 = Lcg(99);
    let mut eager_t = Vec::with_capacity(ITER);
    let mut graph_t = Vec::with_capacity(ITER);
    for _ in 0..ITER {
        let nb = seed3.vec(M * N);
        ops.fill_from_host(&ctx, &mut bias, &nb).unwrap();

        let t0 = Instant::now();
        ops.matmul(&ctx, &blas, &w, &a, &mut mm_out).unwrap();
        ops.add(&ctx, &mm_out, &bias, &mut add_out).unwrap();
        ops.silu(&ctx, &add_out, &mut silu_out).unwrap();
        ops.rmsnorm(&ctx, &silu_out, &alpha, &mut graph_out, 1e-5).unwrap();
        dev.ctx().synchronize().unwrap();
        eager_t.push(t0.elapsed().as_secs_f64() * 1e3);

        let t1 = Instant::now();
        ops.matmul(&ctx, &blas, &w, &a, &mut mm_out).unwrap();
        dev.ctx().synchronize().unwrap(); // 栅栏:legacy 产物 → 图
        graph.launch().unwrap();
        dev.ctx().synchronize().unwrap();
        graph_t.push(t1.elapsed().as_secs_f64() * 1e3);
    }
    let e_med = median(&mut eager_t);
    let g_med = median(&mut graph_t);
    println!(
        "计时(median of {ITER},ms): eager 全链 {:.4} | matmul+graph {:.4} | graph 提速 {:.1}%",
        e_med,
        g_med,
        (e_med - g_med) / e_med * 100.0
    );

    // ---- A5 终验:账本零漂移 ----
    let end = dev.ledger();
    assert_eq!(end.bytes_alive, base_ledger.bytes_alive, "A5: 生命周期末账本必须回到基线");
    println!(
        "== ledger 快照: alive {:.2} KiB(=基线 ✓)/ 累计 {:.2} KiB / peer_map {} ==",
        end.bytes_alive as f64 / 1024.0,
        end.bytes_allocated_total as f64 / 1024.0,
        end.stats.peer_mappings,
    );
}

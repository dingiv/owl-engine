//! 一期 example 01:eager 链路(matmul → add → silu → rmsnorm)。
//!
//! 全部显存经 owl 统一显存管理 API 分配(池 + 账本);两阶段生命周期:
//! P 阶段(建池 + 一次到位分配)与 E 阶段(纯使用,零分配)严格分离
//! (roadmap.local.md 裁决 5)。结尾打印 ledger 快照并断言零漂移(A5)。

use owl_cuda::CudaDevice;
use owl_iface::{Device, PoolConfig, PoolKind};
use owl_nn::cublas::NnBlas;
use owl_nn::ops::OpsCtx;
use owl_nn::tensor::TensorPoolOps;

fn cpu_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; m * n];
    for i in 0..m {
        for p in 0..k {
            let aip = a[i * k + p] as f64;
            for j in 0..n {
                c[i * n + j] += aip * b[p * n + j] as f64;
            }
        }
    }
    c
}

fn cpu_silu(x: &[f32]) -> Vec<f64> {
    x.iter()
        .map(|&v| (v as f64) * (1.0 / (1.0 + (-(v as f64)).exp())))
        .collect()
}

fn cpu_rmsnorm(x: &[f64], alpha: &[f32], n: usize) -> Vec<f64> {
    x.chunks(n)
        .map(|row| {
            let rms = (row.iter().map(|v| v * v).sum::<f64>() / n as f64 + 1e-5).sqrt();
            row.iter()
                .zip(alpha)
                .map(|(v, a)| v / rms * *a as f64)
                .collect::<Vec<f64>>()
        })
        .flatten()
        .collect()
}

fn assert_close(got: &[f32], want: &[f64], atol: f64, rtol: f64) {
    let mut worst: (usize, f64) = (0, 0.0);
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let diff = (*g as f64 - *w).abs();
        let tol = atol + rtol * w.abs();
        assert!(diff <= tol, "对拍失败 @{i}: got {g}, want {w}, diff {diff}");
        if diff > worst.1 {
            worst = (i, diff);
        }
    }
    println!("  对拍通过(最大偏差 {:.3e} @ {})", worst.1, worst.0);
}

/// 确定性伪随机([-1,1))(与 crate 内测试同款 LCG,无 rand 依赖)
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

fn main() {
    let dev = CudaDevice::new(owl_cuda::test_device_ordinal(), owl_cuda::TEST_POOL_BYTES).expect("需要 CUDA 设备");
    println!("== owl 01_eager:统一显存管理下的 NN 链路 ==");
    println!("设备: {} ({:.1} GiB)", dev.desc().uuid, dev.desc().total_bytes as f64 / 2f64.powi(30));

    // ---- P 阶段:建池 + 一次到位分配(含链中间量)----
    let pool = dev
        .create_pool(PoolConfig {
            name: format!("e01-chain-{}", std::process::id()),
            kind: PoolKind::Weights,
            bytes: 16 << 20,
        })
        .expect("建池");
    let blas = NnBlas::new(&dev).expect("NnBlas(workspace 预钉)");
    let mut ops = OpsCtx::new(&dev).expect("OpsCtx");
    let ctx = ops.ctx(owl_iface::MemPhase::Idle);

    const M: usize = 64;
    const K: usize = 128;
    const N: usize = 64;
    let mut w = pool.zeros_tensor::<f32>(&[M, K]).unwrap();
    let mut a = pool.zeros_tensor::<f32>(&[K, N]).unwrap();
    let mut mm_out = pool.zeros_tensor::<f32>(&[M, N]).unwrap();
    let mut bias = pool.zeros_tensor::<f32>(&[M, N]).unwrap();
    let mut add_out = pool.zeros_tensor::<f32>(&[M, N]).unwrap();
    let mut silu_out = pool.zeros_tensor::<f32>(&[M, N]).unwrap();
    let mut alpha = pool.zeros_tensor::<f32>(&[N]).unwrap();
    let mut norm_out = pool.zeros_tensor::<f32>(&[M, N]).unwrap();

    let mut seed = Lcg(7);
    ops.fill_from_host(&ctx, &mut w, &seed.vec(M * K)).unwrap();
    ops.fill_from_host(&ctx, &mut a, &seed.vec(K * N)).unwrap();
    ops.fill_from_host(&ctx, &mut bias, &seed.vec(M * N)).unwrap();
    ops.fill_from_host(&ctx, &mut alpha, &seed.vec(N)).unwrap();

    let base_ledger = dev.ledger();
    println!(
        "P 阶段完成:账本存活 {:.2} KiB(累计分配 {:.2} KiB)",
        base_ledger.bytes_alive as f64 / 1024.0,
        base_ledger.bytes_allocated_total as f64 / 1024.0
    );

    // ---- E 阶段:纯使用,零分配 ----
    ops.matmul(&ctx, &blas, &w, &a, &mut mm_out).unwrap();
    ops.add(&ctx, &mm_out, &bias, &mut add_out).unwrap();
    ops.silu(&ctx, &add_out, &mut silu_out).unwrap();
    ops.rmsnorm(&ctx, &silu_out, &alpha, &mut norm_out, 1e-5).unwrap();

    let e_ledger = dev.ledger();
    assert_eq!(
        e_ledger.bytes_alive, base_ledger.bytes_alive,
        "A5: E 阶段账本必须零漂移"
    );
    println!("E 阶段完成:账本零漂移 ✓");

    // ---- CPU f64 对拍 ----
    let w_h = w.to_vec().unwrap();
    let a_h = a.to_vec().unwrap();
    let mm = cpu_matmul(&w_h, &a_h, M, K, N);
    let bias_h = bias.to_vec().unwrap();
    let added: Vec<f64> = mm.iter().zip(&bias_h).map(|(c, b)| c + *b as f64).collect();
    let silu = cpu_silu(&added.iter().map(|&v| v as f32).collect::<Vec<_>>());
    let alpha_h = alpha.to_vec().unwrap();
    let want = cpu_rmsnorm(&silu, &alpha_h, N);

    println!("对拍 matmul:");
    assert_close(&mm_out.to_vec().unwrap(), &mm, 2e-3, 2e-3);
    println!("对拍 silu(add):");
    assert_close(&silu_out.to_vec().unwrap(), &silu, 2e-3, 2e-3);
    println!("对拍 rmsnorm(终输出):");
    assert_close(&norm_out.to_vec().unwrap(), &want, 2e-2, 2e-2);

    println!("终输出前 4: {:?}", &norm_out.to_vec().unwrap()[..4]);
    println!(
        "== ledger 快照: alive {:.2} KiB / 累计 {:.2} KiB / 分配笔数 p={} s={} / 延迟 {} 归还 {} ==",
        e_ledger.bytes_alive as f64 / 1024.0,
        e_ledger.bytes_allocated_total as f64 / 1024.0,
        e_ledger.stats.persistent_allocs,
        e_ledger.stats.scratch_allocs,
        e_ledger.stats.deferred_frees,
        e_ledger.stats.drained_frees,
    );
}

//! RotaryEmbedding 真值对拍(独立集成测试二进制)。
//!
//! 放 tests/ 而非 lib 单元测试的原因:ctx_scope rig 是 OnceLock 进程一次安装,
//! lib 测试进程内其他 rig 消费者(如 deltanet e2e)会与本测试抢池
//! (谁先装谁定池大小,后者 scratch 不够即挂)。独立进程互不相扰。

use owl_engine::config::Config;
use owl_engine::models::dry_kernels::DryKernels;
use owl_engine::models::layers::ctx_scope;
use owl_engine::models::layers::rotary_emb::ScalingRotaryEmbedding;
use owl_engine::models::layers::Tensor;
use owl_iface::{Device as _, PoolConfig, PoolKind};
use owl_nn::cublas::NnBlas;
use owl_nn::Dtype as DType;

const REAL_CONFIG: &str = "/home/div/Documents/codes/models/Qwen/Qwen3.5-0.8B/config.json";

/// 真值对拍:0.8B 真参数(theta 1e7 / partial 0.25 / head_dim 256)
/// → rotary_dim 64、表 [262144, 32];数值对 f32 host 参考。
/// 显存核算:262144×32×4B = 33.55 MB/表,×2(cos+sin) = 67.1 MB(f32);
/// 头间共享不复制,f16/bf16 减半 —— 远低于阈值,无需按 max_model_len 收窄。
#[test]
fn rotary_truth_matches_reference_qwen35_08b() {
    if !std::path::Path::new(REAL_CONFIG).exists() {
        eprintln!("skip: 真模型 config 不在本机({REAL_CONFIG})");
        return;
    }
    let dev =
        owl_cuda::CudaDevice::new(owl_cuda::test_device_ordinal()).expect("需要 CUDA 设备");
    let scratch = std::sync::Arc::new(
        dev.create_pool(PoolConfig {
            name: format!("rope-scratch-{}", std::process::id()),
            kind: PoolKind::Scratch,
            bytes: 256 << 20,
        })
        .unwrap(),
    );
    let wpool = std::sync::Arc::new(
        dev.create_pool(PoolConfig {
            name: format!("rope-weights-{}", std::process::id()),
            kind: PoolKind::Weights,
            bytes: 512 << 20,
        })
        .unwrap(),
    );
    let ops = owl_nn::OpsCtx::new(&dev).unwrap();
    let blas = NnBlas::new(&dev).unwrap();
    let dry = DryKernels::new(dev.ctx()).unwrap();
    ctx_scope::install(ops, blas, dry, scratch.clone(), wpool.clone(), &dev);

    let json = std::fs::read_to_string(REAL_CONFIG).unwrap();
    let cfg = Config::from_json_str(&json).unwrap();
    assert_eq!(cfg.rope_theta, Some(10_000_000.0));
    assert_eq!(cfg.partial_rotary_factor, Some(0.25));
    assert_eq!(cfg.head_dim, Some(256));

    // 模拟 qwen3_5 构造点(F32 档);sin/cos/matmul 走 ctx 桥
    let ctx = ctx_scope::eager_ctx(owl_iface::MemPhase::Live).unwrap();
    let _guard = ctx_scope::push(&ctx);
    let rope = ScalingRotaryEmbedding::new(
        DType::F32,
        &cfg,
        &ctx_scope::with_device(),
        false,
        cfg.rope_theta,
    )
    .unwrap();

    // 形状/档位:rotary_dim = 0.25×256 = 64 → inv_freq 32 列
    assert_eq!(rope.0.rotary_dim, Some(64));
    assert_eq!(rope.0.cos.shape(), &[262144, 32]);
    assert_eq!(rope.0.sin.shape(), &[262144, 32]);
    assert_eq!(rope.0.cos.dtype(), DType::F32);

    let n = 262144 * 32;
    let d2h = |t: &Tensor| -> Vec<f32> {
        use owl_cuda::ffi::sys;
        dev.ctx().bind_to_thread().unwrap();
        dev.ctx().synchronize().unwrap(); // 非默认流 kernel 完成后再拷(否则偶发整块零读)
        let mut host = vec![0f32; n];
        unsafe {
            sys::cuMemcpyDtoH_v2(
                host.as_mut_ptr() as *mut std::ffi::c_void,
                t.device_ptr() as sys::CUdeviceptr,
                n * 4,
            )
            .result()
            .unwrap();
        }
        host
    };
    let cos_h = d2h(&rope.0.cos);
    let sin_h = d2h(&rope.0.sin);

    // host 参考:inv_freq[i] = theta^(-i/64)(i 步进 2 → i∈{0,2,..,62})
    let theta = 10_000_000.0f64;
    let inv_freq: Vec<f32> = (0..64)
        .step_by(2)
        .map(|i| 1f32 / (theta.powf(i as f64 / 64.0)) as f32)
        .collect();
    assert_eq!(inv_freq.len(), 32);

    // ① 近端全量:pos 0..8 × 全部 32 频率
    for pos in 0..8usize {
        for (j, inv) in inv_freq.iter().enumerate() {
            let ang = pos as f32 * inv;
            assert!(
                (cos_h[pos * 32 + j] - ang.cos()).abs() < 1e-5,
                "cos[{pos},{j}] {} vs {}",
                cos_h[pos * 32 + j],
                ang.cos()
            );
            assert!(
                (sin_h[pos * 32 + j] - ang.sin()).abs() < 1e-5,
                "sin[{pos},{j}] {} vs {}",
                sin_h[pos * 32 + j],
                ang.sin()
            );
        }
    }

    // ② 远端抽查:pos = 262143,取最大索引频率(θ 最小,误差可控)
    let pos = 262143usize;
    for j in [30usize, 31] {
        let ang = pos as f32 * inv_freq[j];
        assert!(
            (cos_h[pos * 32 + j] - ang.cos()).abs() < 1e-4,
            "cos[{pos},{j}] {} vs {}",
            cos_h[pos * 32 + j],
            ang.cos()
        );
        assert!(
            (sin_h[pos * 32 + j] - ang.sin()).abs() < 1e-4,
            "sin[{pos},{j}] {} vs {}",
            sin_h[pos * 32 + j],
            ang.sin()
        );
    }
}

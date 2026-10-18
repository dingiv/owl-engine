//! **显存硬预算门**(2026-10-16):server 侧**主动拒发**超出限度的分配请求 ——
//! 不让 `cudaMallocAsync` 去抛错(那是"已经太晚"的现场:错误语义模糊 + 池被撕碎)。
//!
//! 两条判据(见 `cuda::state::VramGuard::check`):
//! ① 驱动账:`free − reserve ≥ 请求`;② 逻辑上限(可选):`live + n ≤ budget`。
//! 拒发 = **结构化** `PoolExhausted`(要多少/还能给多少)+ `is_fatal() == true`
//! ⇒ 服务面据此走优雅关停(见 `apps/server/src/shutdown.rs`)。
//!
//! 跑法:`CUDA_VISIBLE_DEVICES=GPU-<3080> OWL_TEST_DEVICE=0 \
//!        cargo test -p owl-cuda --test vram_budget_probe -- --nocapture`

use owl_cuda::{
    test_device_ordinal, DeviceClient as _, DeviceSelector, DiagOpts, Dtype, GpuClient, GpuServer,
};
use owl_iface::contract::ModelError;

async fn client_with(diag: DiagOpts) -> GpuClient {
    let (tx, rx) = std::sync::mpsc::channel::<owl_cuda::Command>();
    let server =
        GpuServer::with_diag(rx, DeviceSelector::Ordinal(test_device_ordinal()), None, diag, false);
    let mut client = GpuClient::new(tx);
    std::thread::Builder::new()
        .spawn(move || server.run().expect("server run"))
        .expect("spawn server");
    client.sync().await.expect("sync");
    client
}

/// ② 逻辑上限:预算 1MiB ⇒ 2MiB 请求必被拒(且**不依赖卡上余量**,可重复)
#[tokio::test]
async fn logical_cap_refuses_before_cuda() {
    if owl_cuda::test_device_bdf().is_none() {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    }
    let mut c = client_with(DiagOpts { vram_budget_mb: Some(1), ..Default::default() }).await;
    // 先占 0.5MiB(在上限内):应放行
    let ok = c.alloc(Dtype::F32, (512 * 1024) / 4, "probe:budget-ok").await.expect("预算内应放行");
    drop(ok);
    // 再要 2MiB:live(0.5MiB 已归还但账已记)… 直接要 2MiB > 1MiB 上限 ⇒ 拒
    let err = c
        .alloc(Dtype::F32, (2 * 1024 * 1024) / 4, "probe:budget-over")
        .await
        .expect_err("超上限必须拒发");
    eprintln!("[硬预算门] 逻辑上限拒发: {err}");
    match &err {
        ModelError::PoolExhausted { pool, needed, .. } => {
            assert!(pool.contains("vram-hard-budget"), "拒发须带预算标识:{pool}");
            assert_eq!(*needed, 2 * 1024 * 1024);
        }
        other => panic!("期望 PoolExhausted,得 {other:?}"),
    }
    assert!(err.is_fatal(), "拒发属不可恢复类(服务面据此优雅关停)");
}

/// ① 驱动账:reserve 拉满(1TiB)⇒ 任何请求都被拒,且**一次 malloc 都没发**
#[tokio::test]
async fn driver_free_minus_reserve_refuses() {
    if owl_cuda::test_device_bdf().is_none() {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    }
    let mut c = client_with(DiagOpts { vram_reserve_mb: 1 << 20, ..Default::default() }).await;
    let err = c
        .alloc(Dtype::F32, 1024, "probe")
        .await
        .expect_err("reserve 拉满必须拒发");
    eprintln!("[硬预算门] 驱动账拒发: {err}");
    assert!(matches!(err, ModelError::PoolExhausted { .. }), "得 {err:?}");
    assert!(err.is_fatal());
}

/// 对照:默认预算 ⇒ 正常小分配照常放行(硬预算不是"一律拒绝")
#[tokio::test]
async fn default_budget_still_allocates() {
    if owl_cuda::test_device_bdf().is_none() {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    }
    let mut c = client_with(DiagOpts::default()).await;
    let b = c.alloc(Dtype::F32, 1024, "probe").await.expect("默认档应放行");
    let mut out = vec![0u8; 16];
    c.dtoh_at(&b, 0, &mut out).await.expect("回读");
    assert_eq!(out, vec![0u8; 16], "清零分配");
}

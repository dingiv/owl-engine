//! **显存硬预算门**(2026-10-16):server 侧**主动拒发**超出限度的分配请求 ——
//! 不让 `cudaMallocAsync` 去抛错(那是"已经太晚"的现场:错误语义模糊 + 池被撕碎)。
//!
//! 判据(见 `cuda::state::VramGuard::check`):
//! 驱动账 `free − reserve ≥ 请求`。
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

/// ① 驱动账:reserve 拉满(1TiB)⇒ 任何请求都被拒,且**一次 malloc 都没发**
#[tokio::test]
async fn driver_free_minus_reserve_refuses() {
    if owl_cuda::test_device_bdf().is_none() {
        owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
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
        owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
        return;
    }
    let mut c = client_with(DiagOpts::default()).await;
    let b = c.alloc(Dtype::F32, 1024, "probe").await.expect("默认档应放行");
    let mut out = vec![0u8; 16];
    c.dtoh_at(&b, 0, &mut out).await.expect("回读");
    assert_eq!(out, vec![0u8; 16], "清零分配");
}

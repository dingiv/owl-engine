//! DMA 微基准探针:pinned 分配 / memcpy 带宽 / alloc-free 循环,各 20 次取样。

use owl_cuda::{test_device_ordinal, DeviceClient as _, DeviceSelector, Dtype, GpuClient, GpuServer, Shape};
use std::time::Instant;

#[tokio::test]
async fn dma_microbench() {
    const MB: usize = 8 * 1024 * 1024; // 8MB 块(CHUNK_BYTES 量级)
    let (tx, rx) = std::sync::mpsc::channel::<owl_cuda::Command>();
    let server = GpuServer::new(rx, DeviceSelector::Ordinal(test_device_ordinal()), None);
    let mut client = GpuClient::new(tx);
    std::thread::Builder::new().spawn(move || server.run().expect("server run")).unwrap();
    client.sync().await.unwrap();

    let src: Vec<u8> = vec![0xABu8; MB];
    let dst = client.alloc(Dtype::U32, MB / 4).await.expect("alloc");

    // 1) htod 单块重复(含 to_vec 拷贝 + staging malloc/free 每次裸付)
    let t0 = Instant::now();
    for _ in 0..20 {
        let b = client.htod(Dtype::F16, &Shape::from(vec![MB / 2]), &src).await.expect("htod");
        let _ = b;
    }
    client.sync().await.unwrap();
    let htod_avg = (Instant::now() - t0).as_secs_f64() * 1e3 / 20.0;
    eprintln!("[dma] htod 8MB ×20: 平均 {htod_avg:.2} ms/块 = {:.0} MB/s", MB as f64 / 1024.0 / (htod_avg / 1e3));

    // 2) alloc_pinned/upload_pinned 流水(池命中后)
    let t0 = Instant::now();
    for _ in 0..20 {
        let mut lease = client.alloc_pinned(MB).await.expect("pinned");
        lease.slice_bytes_mut().copy_from_slice(&src);
        client.upload_pinned(lease, &dst, 0, MB).await.expect("upload");
    }
    client.sync().await.unwrap();
    let up_avg = (Instant::now() - t0).as_secs_f64() * 1e3 / 20.0;
    eprintln!("[dma] upload 8MB ×20: 平均 {up_avg:.2} ms/块 = {:.0} MB/s", MB as f64 / 1024.0 / (up_avg / 1e3));

    // 3) 对照:首块单独计时(cold,含池 miss + malloc_host)
    let t0 = Instant::now();
    let lease = client.alloc_pinned(MB * 4).await.expect("pinned cold 32MB");
    let cold = (Instant::now() - t0).as_secs_f64() * 1e3;
    eprintln!("[dma] alloc_pinned 32MB cold: {cold:.2} ms");
    drop(lease);
    client.sync().await.unwrap();
}

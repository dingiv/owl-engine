//! **`dtoh_at` 门**(状态存储搬出面,2026-10-16)。
//!
//! 判据:`dtoh_at` 给出的必须是**块内 [off, off+len) 那一段**的逐字节真值 ——
//! 老 `dtoh` 只读前缀,而"逻辑块 = 层缓冲内的一个页"必须按槽偏移读
//! (`docs/arch/store-状态存储-设计与迁移.md` §10.3 推论 B / §11.3 阻塞①)。
//!
//! 跑法:`CUDA_VISIBLE_DEVICES=GPU-<3080> OWL_TEST_DEVICE=0 \
//!        cargo test -p owl-cuda --test dtoh_at_probe -- --nocapture`

use owl_cuda::{test_device_ordinal, DeviceClient as _, DeviceSelector, Dtype, GpuClient, GpuServer};

const N: usize = 4096; // 设备块字节

fn pattern(n: usize) -> Vec<u8> {
    (0..n).map(|i| (i % 251) as u8).collect()
}

#[tokio::test]
async fn dtoh_at_reads_the_requested_span() {
    if owl_cuda::test_device_bdf().is_none() {
        owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
        return;
    }
    let (tx, rx) = std::sync::mpsc::channel::<owl_cuda::Command>();
    let server = GpuServer::new(rx, DeviceSelector::Ordinal(test_device_ordinal()), None);
    let mut client = GpuClient::new(tx);
    std::thread::Builder::new()
        .spawn(move || server.run().expect("server run"))
        .expect("spawn server");
    client.sync().await.expect("sync");

    let dev = client.alloc(Dtype::U32, N / 4, "probe").await.expect("alloc");
    // 内容:分三段(带偏移)写入 —— 顺带覆盖 upload_pinned 的多段路径
    let all = pattern(N);
    for (off, len) in [(0usize, 2048usize), (2048, 1024), (3072, 1024)] {
        let mut lease = client.alloc_pinned(len, "probe:dtoh_at").await.expect("pinned");
        lease.slice_bytes_mut()[..len].copy_from_slice(&all[off..off + len]);
        client.upload_pinned(lease, &dev, off, len).await.expect("upload");
    }
    client.sync().await.expect("sync");

    // ① 整块前缀读(与老 dtoh 同口径,回归)
    let mut prefix = vec![0u8; N];
    client.dtoh(&dev, &mut prefix).await.expect("dtoh");
    assert_eq!(prefix, all, "dtoh 前缀读回归");

    // ② 中段(既非前缀也非后缀):跨两个写入段
    let mut mid = vec![0u8; 1024];
    client.dtoh_at(&dev, 1024, &mut mid).await.expect("dtoh_at 中段");
    assert_eq!(mid, all[1024..2048], "dtoh_at 中段必须逐字节一致");

    // ③ 尾段 + 单字节边界
    let mut tail = vec![0u8; 1024];
    client.dtoh_at(&dev, 3072, &mut tail).await.expect("dtoh_at 尾段");
    assert_eq!(tail, all[3072..], "dtoh_at 尾段必须逐字节一致");
    let mut one = vec![0u8; 1];
    client.dtoh_at(&dev, 4095, &mut one).await.expect("末字节");
    assert_eq!(one[0], all[4095], "末字节");

    // ④ 越界 = **响亮拒**(off + len > 块字节)
    let mut over = vec![0u8; 2];
    let err = client
        .dtoh_at(&dev, 4095, &mut over)
        .await
        .expect_err("越界必须拒");
    eprintln!("[dtoh_at 门] 越界拒原文: {err}");
    assert!(format!("{err}").contains("dtoh"), "错误须自带定位:{err}");
}

/// **批命令门**:`DtohBatch` 回执 = 各段按入参顺序拼接(逐段等价于 `DtohAt`);
/// `HtodBatch` 写回后逐段读出必须一致。
#[tokio::test]
async fn dtoh_batch_and_htod_batch_round_trip() {
    if owl_cuda::test_device_bdf().is_none() {
        owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
        return;
    }
    let (tx, rx) = std::sync::mpsc::channel::<owl_cuda::Command>();
    let server = GpuServer::new(rx, DeviceSelector::Ordinal(test_device_ordinal()), None);
    let mut client = GpuClient::new(tx);
    std::thread::Builder::new()
        .spawn(move || server.run().expect("server run"))
        .expect("spawn server");
    client.sync().await.expect("sync");

    let _dev = client.alloc(Dtype::U32, N / 4, "probe").await.expect("alloc");
    let all = pattern(N);
    let a = client.alloc(Dtype::U32, 2048 / 4, "probe").await.expect("a"); // 两段:0..1024 与 1024..2048
    let b = client.alloc(Dtype::U32, 4096 / 4, "probe").await.expect("b"); // 留出偏移余量

    // 批量写:三段分落两块(带偏移)
    client
        .htod_batch(&[(&a, 0, &all[0..1024]), (&b, 512, &all[1024..3072]), (&a, 1024, &all[3072..4096])])
        .await
        .expect("htod_batch");
    client.sync().await.expect("sync");

    // 批量读:一次命令回三段(顺序拼接)
    let got = client
        .dtoh_batch(&[(&a, 0, 1024), (&b, 512, 2048), (&a, 1024, 1024)])
        .await
        .expect("dtoh_batch");
    assert_eq!(got, all, "批读回执 = 各段按序拼接,逐字节一致");

    // 逐段对照:与单段 `dtoh_at` 同值
    let mut one = vec![0u8; 2048];
    client.dtoh_at(&b, 512, &mut one).await.expect("dtoh_at 对照");
    assert_eq!(one, got[1024..3072], "批读中段 = 单段读");

    // 越界段 = 整条命令响亮拒(a 只有 2048 字节:2048 + 100 越界)
    let err = client
        .dtoh_batch(&[(&a, 0, 1024), (&a, 2048, 100)])
        .await
        .expect_err("越界段必须拒");
    eprintln!("[批命令门] 越界拒原文: {err}");
}

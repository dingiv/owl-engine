//! **E2b 复合命令探针**(设计 `docs/arch/三流事件序-设计与迁移.md` §3.2/§8)。
//!
//! 验两件事(都是"设备端事件序"的直接判据,不是端到端大跑):
//! 1. `UploadAndLaunch` = 搬运 + 发射,**序**必须成立 —— 核读的块正是刚上传的块
//!    (若序丢了,核会读到旧数据/半截数据 ⇒ 数值错);
//! 2. `LaunchAndHarvest` = 发射 + 回读,回读必须等到核写完 ⇒ 数值对;
//!    且**闭集校验**:harvest id ≠ 本 launch 输出块 ⇒ 必须**响亮拒**。
//!
//! 运行:`OWL_TEST_DEVICE=<空闲卡> cargo test -p owl-cuda --test composite_launch_probe`

use owl_cuda::{
    test_device_ordinal, Arg, Bytes, Command, DeviceClient as _, DeviceSelector, Dtype, GpuClient,
    GpuServer, KernelSource, LaunchMsg, Shape,
};

const ADD_KERNEL: &str = "owl_add_f32";

fn assemble() -> GpuClient {
    let (tx, rx) = std::sync::mpsc::channel::<Command>();
    let server = GpuServer::new(rx, DeviceSelector::Ordinal(test_device_ordinal()), None);
    let client = GpuClient::new(tx);
    std::thread::Builder::new()
        .name("owl-composite-probe".into())
        .spawn(move || server.run().expect("server run 异常退出"))
        .expect("起 server 线程失败");
    client
}

fn le_f32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn add_msg(a: &Bytes, b: &Bytes, out: &Bytes, n: usize) -> LaunchMsg {
    LaunchMsg {
        kernel: KernelSource {
            name: ADD_KERNEL.into(),
            source: owl_kernels::ops::source(ADD_KERNEL).to_string(),
        },
        args: vec![
            Arg::block(a.id, a.dtype, a.rank),
            Arg::block(b.id, b.dtype, b.rank),
            Arg::block(out.id, out.dtype, out.rank),
            Arg::U64(n as u64),
        ],
        grid: ((n as u32).div_ceil(256), 1, 1),
        block: (256, 1, 1),
        shared_mem: 0,
        out_elems: n,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn composites_hold_stream_order_and_closed_set_validation() {
    if !owl_shared::env_reader::flag("OWL_TEST_DEVICE") {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    }
    let mut c = assemble();
    let n = 4096usize;

    // 输入 A(普通 H2D)与 B 的**宿主值**(经复合命令上传)
    let av: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5).collect();
    let bv: Vec<f32> = (0..n).map(|i| (i as f32) * -0.25 + 1.0).collect();
    let want: Vec<f32> = av.iter().zip(&bv).map(|(x, y)| x + y).collect();

    let a = owl_iface::htod!(c, Dtype::F32, &Shape::from(vec![n]), &le_f32(&av))
        .await
        .expect("htod A");
    let b = owl_iface::client_alloc!(c, Dtype::F32, n)
        .await
        .expect("alloc B")
        .view(Dtype::F32, 1);
    let out1 = owl_iface::client_alloc!(c, Dtype::F32, n)
        .await
        .expect("alloc out1")
        .view(Dtype::F32, 1);
    let out2 = owl_iface::client_alloc!(c, Dtype::F32, n)
        .await
        .expect("alloc out2")
        .view(Dtype::F32, 1);

    // ① UploadAndLaunch:B 由复合命令搬运,**紧接着**的核必须读到它
    let mut lease = owl_iface::alloc_pinned!(c, n * 4).await.expect("pinned");
    let bb = le_f32(&bv);
    lease.slice_bytes_mut()[..bb.len()].copy_from_slice(&bb);
    let out_b = c
        .upload_and_launch(
            &b,
            vec![owl_cuda::UpChunk { offset_bytes: 0, len_bytes: n * 4, lease }],
            add_msg(&a, &b, &out1, n),
        )
        .await
        .expect("upload_and_launch");
    assert_eq!(out_b.id, out1.id, "复合 #1 回传的应是本 launch 输出块");

    // ② LaunchAndHarvest:发射 + 回读(回读必须等核写完)
    let got2 = c
        .launch_and_harvest(add_msg(&a, &b, &out2, n), out2.id, n * 4)
        .await
        .expect("launch_and_harvest");
    let got2: Vec<f32> = got2
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(got2.len(), n);
    let bad2 = got2
        .iter()
        .zip(&want)
        .filter(|(g, w)| (*g - *w).abs() > 1e-3)
        .count();
    assert_eq!(bad2, 0, "复合 #2 回读与期望不符:坏 {bad2}/{n}");

    // ③ 复合 #1 的输出也验一遍(证明序成立:若核读了半截/旧 B,这里会错)
    let mut buf = vec![0u8; n * 4];
    c.dtoh(&out1, &mut buf).await.expect("dtoh out1");
    let got1: Vec<f32> = buf
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let bad1 = got1
        .iter()
        .zip(&want)
        .filter(|(g, w)| (*g - *w).abs() > 1e-3)
        .count();
    assert_eq!(
        bad1, 0,
        "复合 #1 输出与期望不符(搬运→发射的序丢了?):坏 {bad1}/{n}"
    );

    // ④ 闭集校验负门:harvest id 不是本 launch 输出块 ⇒ 必须拒
    let err = c
        .launch_and_harvest(add_msg(&a, &b, &out2, n), out1.id, n * 4)
        .await
        .expect_err("harvest id ≠ 输出块时应拒");
    let msg = format!("{err}");
    assert!(
        msg.contains("闭集校验"),
        "拒错文案应点名闭集校验,实得: {msg}"
    );
    eprintln!("[composite] ①②③④ 全绿(序守恒 + 闭集校验)");

    c.sync().await.expect("sync");
}

/// **负门**:捕获期的复合命令必须**结构化拒**(内含搬运/回读 ⇒ 不可捕获;
/// `cuStreamWaitEvent` 也不可捕获)。判据 = 报错文案点名捕获。
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn composites_are_rejected_during_capture() {
    if !owl_shared::env_reader::flag("OWL_TEST_DEVICE") {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    }
    let mut c = assemble();
    let n = 256usize;
    let a = owl_iface::htod!(c, Dtype::F32, &Shape::from(vec![n]), &le_f32(&vec![1.0f32; n]))
        .await
        .expect("htod A");
    let b = owl_iface::client_alloc!(c, Dtype::F32, n)
        .await
        .expect("alloc B")
        .view(Dtype::F32, 1);
    let out = owl_iface::client_alloc!(c, Dtype::F32, n)
        .await
        .expect("alloc out")
        .view(Dtype::F32, 1);

    // 租约在**窗外**取(捕获白名单连 AllocPinned 也拒 —— 白名单之严顺带被本测佐证)
    let lease = owl_iface::alloc_pinned!(c, n * 4).await.expect("pinned");
    c.graph_begin().await.expect("捕获开始");
    // 捕获期发复合 #1(内含 H2D)⇒ 必须被白名单拒
    let e1 = c
        .upload_and_launch(
            &b,
            vec![owl_cuda::UpChunk { offset_bytes: 0, len_bytes: n * 4, lease }],
            add_msg(&a, &b, &out, n),
        )
        .await
        .expect_err("捕获期 UploadAndLaunch 应拒");
    // 捕获期发复合 #2(内含 Dtoh)⇒ 同样拒
    let e2 = c
        .launch_and_harvest(add_msg(&a, &b, &out, n), out.id, n * 4)
        .await
        .expect_err("捕获期 LaunchAndHarvest 应拒");
    eprintln!("[composite] 负门:捕获期两条复合均被拒:\n  {e1}\n  {e2}");
    assert!(
        format!("{e1}").contains("捕获") && format!("{e2}").contains("捕获"),
        "拒错文案应点名捕获: {e1} / {e2}"
    );
    // 收尾:被拒命令已把捕获窗置为 invalidated(CUDA 语义)⇒ abort 尽力而为
    // (判据在上面两条拒错文案;此处不把收尾错误当门)
    if let Err(e) = c.graph_abort().await {
        eprintln!("[composite] 负门收尾 abort(捕获已被拒命令作废): {e}");
    }
    c.sync().await.expect("sync");
}

/// **块账本不变量门**(长期价值):活块 id 唯一 + 地址区间不相交。
///
/// 为什么值得长期守:这两件事是"跨流 UAF / 写丢失"类缺陷的**唯一钥匙**
/// (仓内 2026-10-08 E5 跨流回收律、2026-10-03 FI 表渐进腐败案都栽在这里);
/// `/debug/blocks` 与同名的 HTTP 口(`GET /debug/blocks`)共用同一份判决。
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn block_ledger_is_clean() {
    if !owl_shared::env_reader::flag("OWL_TEST_DEVICE") {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    }
    let mut c = assemble();
    let n = 4096usize;
    // 造一批块(不同尺寸 ⇒ 尽量压到不同尺寸类/复用面)
    let a = owl_iface::htod!(c, Dtype::F32, &Shape::from(vec![n]), &le_f32(&vec![1.0f32; n]))
        .await
        .expect("htod");
    let b = owl_iface::client_alloc!(c, Dtype::F32, n).await.expect("alloc").view(Dtype::F32, 1);
    let big = owl_iface::client_alloc!(c, Dtype::F32, n * 64)
        .await
        .expect("alloc big")
        .view(Dtype::F32, 1);
    let _ = (a.id, b.id, big.id);

    let rows = c.block_ledger_dump().await.expect("block_ledger_dump");
    assert!(!rows.is_empty(), "账本不应为空");
    let mut ids: Vec<u64> = rows.iter().map(|r| r.id).collect();
    ids.sort_unstable();
    let dups: Vec<u64> = ids.windows(2).filter(|w| w[0] == w[1]).map(|w| w[0]).collect();
    assert!(dups.is_empty(), "同 id 两条目(账本被顶掉/复用): {dups:?}");

    let mut by_addr: Vec<(u64, usize, u64)> =
        rows.iter().map(|r| (r.addr, r.bytes, r.id)).collect();
    by_addr.sort_by_key(|t| t.0);
    for w in by_addr.windows(2) {
        let (a0, an, aid) = w[0];
        let (b0, _, bid) = w[1];
        assert!(
            a0 + an as u64 <= b0,
            "活块区间相交: id {aid} [{a0:#x}+{an:#x}) 与 id {bid} [{b0:#x})"
        );
    }
    eprintln!(
        "[ledger] {} 活块 / {} 字节;id 唯一 ✓ 区间不相交 ✓",
        rows.len(),
        rows.iter().map(|r| r.bytes).sum::<usize>()
    );
    c.sync().await.expect("sync");
}

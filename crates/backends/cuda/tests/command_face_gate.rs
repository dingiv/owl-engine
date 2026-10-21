//! **SDK ↔ server 命令面门**(2026-10-21):`exec` / `exec_batch` 的真卡往返。
//!
//! 覆盖四条(都是用户点名的面):
//! 1. 单条命令**端到端**(alloc/htod/dtoh),`Outcome` 变体与事实面正确;
//! 2. **原生批** = 一条信箱消息(`segments == 1`),逐条结果与投递序对位;
//! 3. 批内失败:`first_err` 定位 + **已投递的不回滚**(后继命令照常执行);
//! 4. 上层糖(具名 `alloc`)真的落到同一条 `exec` 咽喉(DeviceHandle 内层 = GpuClient)。
//!
//! 运行:`OWL_TEST_DEVICE=01:00.0 cargo test -p owl-cuda --test command_face_gate`
//! (未设 = skip,不隐式选卡。)

use owl_cuda::{
    test_device_ordinal_opt, Command as Envelope, DeviceClient, DeviceSelector, Dtype, GpuClient,
    GpuServer,
};
use owl_iface::command::{Command, Outcome};
use owl_iface::contract::UpStream;

/// 手工组装(与 gpu_server 同款:管道外部创建,server 跑专职线程)。
fn assemble() -> Option<GpuClient> {
    let ordinal = test_device_ordinal_opt()?;
    let (tx, rx) = std::sync::mpsc::channel::<Envelope>();
    let server = GpuServer::new(rx, DeviceSelector::Ordinal(ordinal), None);
    let client = GpuClient::new(tx);
    std::thread::Builder::new()
        .name("owl-cmd-face".into())
        .spawn(move || {
            if let Err(e) = server.run() {
                eprintln!("[command_face_gate] server 退出(Err): {e}");
            }
        })
        .expect("起 server 线程失败");
    Some(client)
}

/// ①-④:单条往返 + 原生批 + 批内失败语义。
#[tokio::test]
async fn exec_and_native_batch_round_trip() {
    let Some(mut c) = assemble() else {
        eprintln!("OWL_TEST_DEVICE 未设 = skip(不隐式选卡)");
        return;
    };

    // ① alloc(事实面:dtype/元素数)
    let a = c
        .exec(Command::Alloc { dtype: Dtype::F32, elems: 64, zero: true, name: "cmd.alloc".into() })
        .await
        .expect("exec alloc")
        .into_buf("exec alloc")
        .expect("alloc 应回 Buf");
    assert_eq!((a.len, a.dtype), (64, Dtype::F32));

    // ② htod(事实面:len/dtype/rank 由命令携带,不靠服务端猜)
    let src: Vec<u8> = (0..8).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let h = c
        .exec(Command::Htod {
            dtype: Dtype::F32,
            shape: vec![8],
            src: src.clone(),
            stream: UpStream::H2d,
            name: "cmd.htod".into(),
        })
        .await
        .expect("exec htod")
        .into_buf("exec htod")
        .expect("htod 应回 Buf");
    assert_eq!((h.len, h.dtype, h.rank), (8, Dtype::F32, 1));

    // ③ dtoh(字节逐位相等)
    let got = c
        .exec(Command::Dtoh { id: h.id, off_bytes: 0, want_bytes: 32 })
        .await
        .expect("exec dtoh")
        .into_bytes("exec dtoh")
        .expect("dtoh 应回 Bytes");
    assert_eq!(got, src, "htod → dtoh 必须逐字节相等");

    // ④ 原生批:一条信箱消息,结果按投递序对位
    let r = c
        .exec_batch(vec![
            Command::Alloc { dtype: Dtype::F32, elems: 4, zero: true, name: "batch.0".into() },
            Command::Alloc { dtype: Dtype::F32, elems: 8, zero: true, name: "batch.1".into() },
            Command::Sync,
        ])
        .await
        .expect("exec_batch 本体不失败(逐条结果在内)");
    assert_eq!(r.segments, 1, "原生批 = 一条信箱消息(不是逐条转发)");
    assert_eq!(r.outcomes.len(), 3);
    assert!(r.all_ok(), "first_err={:?} err={:?}", r.first_err,
        r.outcomes.iter().find(|o| o.is_err()));
    match &r.outcomes[0] {
        Ok(Outcome::Buf(b)) => assert_eq!(b.len, 4),
        other => panic!("批内第 0 条应为 Buf,得 {other:?}"),
    }
    match &r.outcomes[2] {
        Ok(Outcome::Done) => {}
        other => panic!("批内第 2 条(Sync)应为 Done,得 {other:?}"),
    }

    // ⑤ 批内失败:first_err 定位 + 已投递条不回滚
    let bad = c
        .exec_batch(vec![
            // 未知块 id ⇒ 服务端**硬拒**(§79 的显式拒绝律)
            Command::Free { ids: vec![u64::MAX] },
            Command::Alloc { dtype: Dtype::F32, elems: 2, zero: true, name: "after.err".into() },
        ])
        .await
        .expect("批本体仍 Ok");
    assert_eq!(bad.first_err, Some(0), "首个失败下标");
    assert!(bad.outcomes[0].is_err());
    assert!(bad.outcomes[1].is_ok(), "已投递的后继命令照常执行(不回滚)");
    assert_eq!(bad.segments, 1);

    // 收尾
    let freed = match (&r.outcomes[0], &r.outcomes[1]) {
        (Ok(Outcome::Buf(b0)), Ok(Outcome::Buf(b1))) => vec![a.id, h.id, b0.id, b1.id],
        _ => vec![a.id, h.id],
    };
    c.exec(Command::Free { ids: freed }).await.expect("free");
    c.close().await.ok();
}

/// ⑥ 上层糖与命令面**同一条咽喉**:`DeviceHandle::alloc` → GpuClient 缺省糖 → exec。
#[tokio::test]
async fn handle_sugar_lands_on_same_throat() {
    let Some(c) = assemble() else { return };
    let mut dev = owl_iface::handle::DeviceHandle::new(c, None);

    let b = dev.alloc(Dtype::F16, 16, "handle.alloc").await.expect("alloc 经命令面");
    assert_eq!((b.len, b.dtype), (16, Dtype::F16));
    dev.sync().await.expect("sync 经命令面");
    dev.free(&[b.id]).await.expect("free 经命令面");
    // 未知图 id=服务端**响亮拒**(不是静默 Ok —— 旧 GpuClient 覆写也是这条路)
    let e = dev.graph_destroy(999).await.expect_err("未知图必须 Err");
    assert!(format!("{e}").contains("未知图"), "得 {e}");
    if let Some(c) = dev.client_mut_gpu() {
        c.close().await.ok();
    }
}

/// 小工具:`DeviceHandle<GpuClient>` 取回内层(仅测试用,关服收尾)。
trait ClientMut {
    fn client_mut_gpu(&mut self) -> Option<&mut GpuClient>;
}

impl ClientMut for owl_iface::handle::DeviceHandle<GpuClient> {
    fn client_mut_gpu(&mut self) -> Option<&mut GpuClient> {
        Some(self.client_mut())
    }
}

// ============================================================================
// queue / flush(累积批):一条信箱消息,逐条发射按队列序执行
// ============================================================================

const ADD_KERNEL: &str = "owl_add_f32";

/// `owl_add_f32`(a+b→out)发射消息(与 gpu_server 门同款真资产核)
fn add_msg(a: &owl_iface::contract::Bytes, b: &owl_iface::contract::Bytes, out: &owl_iface::contract::Bytes, n: usize) -> owl_iface::contract::LaunchMsg {
    use owl_iface::contract::{Arg, KernelSource, LaunchMsg};
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

#[tokio::test]
async fn queued_launches_flush_as_one_batch() {
    let Some(mut c) = assemble() else { return };

    const N: usize = 256;
    // 输入走 **htod**(返回的是新块:htod = 分配 + 上传,不是写进既有块)
    let av: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let bv: Vec<f32> = (0..N).map(|i| 1000.0 + i as f32).collect();
    let ab: Vec<u8> = av.iter().flat_map(|v| v.to_le_bytes()).collect();
    let bb: Vec<u8> = bv.iter().flat_map(|v| v.to_le_bytes()).collect();
    let a = c
        .exec(Command::Htod {
            dtype: Dtype::F32,
            shape: vec![N],
            src: ab,
            stream: UpStream::H2d,
            name: "q.a".into(),
        })
        .await
        .expect("htod a")
        .into_buf("htod a")
        .expect("buf");
    let b = c
        .exec(Command::Htod {
            dtype: Dtype::F32,
            shape: vec![N],
            src: bb,
            stream: UpStream::H2d,
            name: "q.b".into(),
        })
        .await
        .expect("htod b")
        .into_buf("htod b")
        .expect("buf");
    // 输出块 = alloc(dtype/rank 未定形 ⇒ view 定形,与 gpu_server 门同律)
    let out1 = c
        .exec(Command::Alloc { dtype: Dtype::F32, elems: N, zero: true, name: "q.o1".into() })
        .await
        .expect("alloc o1")
        .into_buf("o1")
        .expect("buf")
        .view(Dtype::F32, 1);
    let out2 = c
        .exec(Command::Alloc { dtype: Dtype::F32, elems: N, zero: true, name: "q.o2".into() })
        .await
        .expect("alloc o2")
        .into_buf("o2")
        .expect("buf")
        .view(Dtype::F32, 1);

    // 两条发射**分两次 queue**(验证累积),按队列序执行:o1=a+b, o2=o1+a
    assert_eq!(c.queue(Command::Launch { msg: add_msg(&a, &b, &out1, N) }).expect("queue 0"), 0);
    assert_eq!(c.queue(Command::Launch { msg: add_msg(&out1, &a, &out2, N) }).expect("queue 1"), 1);

    // 队列非空 ⇒ 单条命令响亮拒(不隐式多跑一次往返)
    let e = c.exec(Command::Sync).await.expect_err("队列非空必须拒");
    assert!(format!("{e}").contains("未提交"), "得 {e}");

    let r = c.flush().await.expect("flush");
    assert_eq!(r.outcomes.len(), 2);
    assert!(r.all_ok(), "first_err={:?}", r.first_err);
    assert_eq!(r.segments, 1, "累积批 = 一条信箱消息");

    // 收割:顺序正确 ⇒ o2 = 2a + b
    let got = c
        .exec(Command::Dtoh { id: out2.id, off_bytes: 0, want_bytes: N * 4 })
        .await
        .expect("dtoh")
        .into_bytes("dtoh")
        .expect("bytes");
    let vals: Vec<f32> = got
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    for i in 0..N {
        let want = 2.0 * av[i] + bv[i];
        assert!((vals[i] - want).abs() < 1e-3, "i={i} 实得 {} 期望 {want}", vals[i]);
    }

    c.exec(Command::Free { ids: vec![a.id, b.id, out1.id, out2.id] })
        .await
        .expect("free");
    c.close().await.ok();
}

/// **往返成本账**(诊断,不设硬门):逐条 `exec` vs `queue+flush`。
///
/// 这是"为什么要批"的唯一直接证据:同一条 `owl_add_f32`(小网格),
/// 逐条 await = 每条一次信箱投递 + 一次唤醒;批量 = 一次投递 + 一次唤醒。
#[tokio::test]
async fn batch_vs_single_round_trip_cost() {
    let Some(mut c) = assemble() else { return };
    const N: usize = 64;
    const ROUNDS: usize = 256;

    let a = c
        .exec(Command::Htod {
            dtype: Dtype::F32,
            shape: vec![N],
            src: vec![0u8; N * 4],
            stream: UpStream::H2d,
            name: "bench.a".into(),
        })
        .await
        .expect("htod a")
        .into_buf("a")
        .expect("buf");
    let b = c
        .exec(Command::Htod {
            dtype: Dtype::F32,
            shape: vec![N],
            src: vec![0u8; N * 4],
            stream: UpStream::H2d,
            name: "bench.b".into(),
        })
        .await
        .expect("htod b")
        .into_buf("b")
        .expect("buf");
    let out = c
        .exec(Command::Alloc { dtype: Dtype::F32, elems: N, zero: true, name: "bench.o".into() })
        .await
        .expect("alloc o")
        .into_buf("o")
        .expect("buf")
        .view(Dtype::F32, 1);

    // ① 逐条:每条一次 await(= 一次信箱往返)
    let t0 = std::time::Instant::now();
    for _ in 0..ROUNDS {
        c.exec(Command::Launch { msg: add_msg(&a, &b, &out, N) }).await.expect("launch");
    }
    c.sync().await.expect("sync");
    let single = t0.elapsed();

    // ② 累积:一次 flush
    let t0 = std::time::Instant::now();
    for _ in 0..ROUNDS {
        c.queue(Command::Launch { msg: add_msg(&a, &b, &out, N) }).expect("queue");
    }
    let r = c.flush().await.expect("flush");
    assert!(r.all_ok());
    assert_eq!(r.segments, 1);
    c.sync().await.expect("sync");
    let batch = t0.elapsed();

    let per = |d: std::time::Duration| d.as_secs_f64() * 1e6 / ROUNDS as f64;
    println!(
        "[往返账] {ROUNDS} 条 launch:逐条 await = {:.2}µs/条;queue+flush = {:.2}µs/条(整批 {:.0}µs);提速 {:.1}×",
        per(single),
        per(batch),
        batch.as_secs_f64() * 1e6,
        single.as_secs_f64() / batch.as_secs_f64().max(1e-9)
    );
    // 只做"批不得更慢"的软门(计时硬门留给同机 A/B)
    assert!(batch.as_secs_f64() <= single.as_secs_f64() * 1.2, "批竟然更慢");

    c.exec(Command::Free { ids: vec![a.id, b.id, out.id] }).await.expect("free");
    c.close().await.ok();
}

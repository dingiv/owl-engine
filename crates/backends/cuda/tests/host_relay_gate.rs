//! **host-mapped allreduce 门(2026-11-04)** —— 无 P2P 驱动环境的兜底通道。
//!
//! 判据与 `ar_transport_gate` 同源:
//! - **G1**:三 dtype × 边界尺寸,`payload ← A + B` 逐位对 host 参考;
//! - **G2**:同输入两跑逐位一致(相对游标推进不改变结果);
//! - **G3**:捕获 → replay ≡ eager(两次回放),图存活时 `host_release` 被租约拒收。
//!
//! 机制:**不碰 VMM/BAR1/P2P**。rank0 `host_alloc` 两段
//! `cudaHostAlloc(MAPPED|PORTABLE)`,rank1 `host_attach` 同一 host 段;
//! 复用现有 `ar_stage`/`ar_reduce` 核与 header 协议。host 段只做 staging,
//! payload 仍在设备显存。
//!
//! 运行(无 P2P 也能跑;这里用两张 3080):
//! ```bash
//! CUDA_VISIBLE_DEVICES=0,4 OWL_PEER_TEST_DEV_A=0 OWL_PEER_TEST_DEV_B=1 \
//!   cargo test -p owl-cuda --test host_relay_gate -- --nocapture --test-threads=1
//! ```

use half::{bf16, f16};
use owl_cuda::comm::ar::{plan_ar, AllReduce, ArPhase, StagingLayout};
use owl_cuda::{
    Bytes, Command, DeviceClient as _, DeviceSelector, Dtype, GpuClient, GpuServer, ModelError,
    Shape,
};
use owl_kernels::client::comm::{ArReduceCall, ArStageCall};

const HDR: u64 = 64;
const SLOT: u64 = 8 << 20;
const N_SLOTS: u32 = 2;
const MAGIC: u32 = 0x4F57_4C41;

fn assemble(ordinal: usize, tag: &str) -> GpuClient {
    let (tx, rx) = std::sync::mpsc::channel::<Command>();
    // 低层门只测通信面;当前默认构建未带 nvfp4 家族,而 wildcard 激活会触发
    // “孤儿声明”注册门。用显式 tag 集排除 nvfp4(不改生产语义)。
    let ids: Vec<&str> = owl_kernels::spec::KERNEL_TAGS
        .iter()
        .map(|t| t.id)
        .filter(|id| *id != "quant.nvfp4")
        .collect();
    let tag_filter = owl_kernels::tag_filter::TagFilter::from_ids(&ids)
        .expect("tag filter")
        .with_prod_kv_layouts();
    let diag = owl_cuda::DiagOpts {
        tag_filter,
        ..Default::default()
    };
    let server = GpuServer::with_diag(rx, DeviceSelector::Ordinal(ordinal), None, diag, false);
    let client = GpuClient::new(tx);
    std::thread::Builder::new()
        .name(format!("owl-hostar-{tag}"))
        .spawn(move || server.run().expect("server run 异常退出"))
        .expect("起 server 线程失败");
    client
}

fn dev(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.parse().ok()
}

fn hdr_bytes(layout: StagingLayout) -> [u8; HDR as usize] {
    let mut b = [0u8; HDR as usize];
    b[16..20].copy_from_slice(&(layout.slot_bytes as u32).to_le_bytes());
    b[20..24].copy_from_slice(&layout.slots.to_le_bytes());
    b[24..28].copy_from_slice(&MAGIC.to_le_bytes());
    b
}

fn write_hdr_host(host: u64, layout: StagingLayout) {
    let b = hdr_bytes(layout);
    unsafe { std::ptr::copy_nonoverlapping(b.as_ptr(), host as *mut u8, b.len()) };
}

fn read_hdr_host(host: u64) -> [u8; HDR as usize] {
    let mut out = [0u8; HDR as usize];
    unsafe { std::ptr::copy_nonoverlapping(host as *const u8, out.as_mut_ptr(), out.len()) };
    out
}

fn hdr_u32(h: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(h[off..off + 4].try_into().expect("4B"))
}

fn hdr_u64(h: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(h[off..off + 8].try_into().expect("8B"))
}

fn lcg_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn width(dt: Dtype) -> usize {
    match dt {
        Dtype::F32 => 4,
        Dtype::F16 | Dtype::BF16 => 2,
        other => panic!("host relay 门只测三 dtype,拿到 {other:?}"),
    }
}

fn from_f32_vec(vals: &[f32], dt: Dtype) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * width(dt));
    for &v in vals {
        match dt {
            Dtype::F32 => out.extend_from_slice(&v.to_le_bytes()),
            Dtype::F16 => out.extend_from_slice(&f16::from_f32(v).to_bits().to_le_bytes()),
            Dtype::BF16 => out.extend_from_slice(&bf16::from_f32(v).to_bits().to_le_bytes()),
            other => panic!("{other:?}"),
        }
    }
    out
}

fn to_f32_vec(bytes: &[u8], dt: Dtype) -> Vec<f32> {
    match dt {
        Dtype::F32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
        Dtype::F16 => bytes
            .chunks_exact(2)
            .map(|c| f16::from_bits(u16::from_le_bytes(c.try_into().unwrap())).to_f32())
            .collect(),
        Dtype::BF16 => bytes
            .chunks_exact(2)
            .map(|c| bf16::from_bits(u16::from_le_bytes(c.try_into().unwrap())).to_f32())
            .collect(),
        other => panic!("{other:?}"),
    }
}

fn host_ref(a: &[u8], b: &[u8], dt: Dtype) -> Vec<u8> {
    let (va, vb) = (to_f32_vec(a, dt), to_f32_vec(b, dt));
    let sum: Vec<f32> = va.iter().zip(vb.iter()).map(|(x, y)| x + y).collect();
    from_f32_vec(&sum, dt)
}

async fn reset_payload(client: &mut GpuClient, b: &Bytes, data: &[u8]) -> Result<(), ModelError> {
    let n = data.len();
    let mut pin = client.alloc_pinned(n, "hostar:reset").await?;
    pin.slice_bytes_mut()[..n].copy_from_slice(data);
    client.upload_pinned(pin, b, 0, n).await?;
    client.sync().await
}

async fn emit_rank(
    client: &mut GpuClient,
    payload: &Bytes,
    own: &Bytes,
    peer: &Bytes,
    bytes: u64,
    layout: &StagingLayout,
) -> Result<(), ModelError> {
    for (phase, c) in AllReduce::new(plan_ar(bytes, layout)).steps() {
        let msg = match phase {
            ArPhase::Stage => ArStageCall::new(
                payload.clone(),
                own.clone(),
                peer.clone(),
                own.clone(),
                c.bytes,
                layout.slot_bytes,
                c.offset,
            )
            .to_launch(),
            ArPhase::Reduce => ArReduceCall::new(
                payload.clone(),
                own.clone(),
                peer.clone(),
                peer.clone(),
                c.bytes,
                layout.slot_bytes,
                c.offset,
            )
            .to_launch(),
        };
        client.launch(msg).await?;
    }
    Ok(())
}

async fn read_payload(
    client: &mut GpuClient,
    b: &Bytes,
    n_bytes: usize,
) -> Result<Vec<u8>, ModelError> {
    let mut buf = vec![0u8; n_bytes];
    client.dtoh(b, &mut buf).await?;
    Ok(buf)
}

#[tokio::test]
async fn host_relay_parity() {
    let (Some(oa), Some(ob)) = (dev("OWL_PEER_TEST_DEV_A"), dev("OWL_PEER_TEST_DEV_B")) else {
        owl_shared::testkit::report::skip(
            "需 OWL_PEER_TEST_DEV_A / OWL_PEER_TEST_DEV_B(两张空闲卡)",
        );
        return;
    };
    assert_ne!(oa, ob, "A/B 必须是两张不同卡");
    let mut a = assemble(oa, "a");
    let mut b = assemble(ob, "b");
    let layout = StagingLayout::new(SLOT, N_SLOTS);
    let bytes = HDR + SLOT * N_SLOTS as u64;

    // rank0 分配两段 host;rank1 attach 同一段(own=H1 / peer=H0)。
    let (own0, h0) = a
        .host_alloc(bytes as usize, "hostar:h0")
        .await
        .expect("A host_alloc H0");
    let (own1, h1) = a
        .host_alloc(bytes as usize, "hostar:h1")
        .await
        .expect("A host_alloc H1");
    let p0 = b
        .host_attach(h1, bytes as usize, "hostar:own.r1")
        .await
        .expect("B attach H1");
    let p1 = b
        .host_attach(h0, bytes as usize, "hostar:peer.r1")
        .await
        .expect("B attach H0");
    write_hdr_host(h0, layout);
    write_hdr_host(h1, layout);

    // ── G1:三 dtype × 边界尺寸 ─────────────────────────────────────
    let cases: [(usize, Dtype); 5] = [
        (4, Dtype::F32),
        (10 << 10, Dtype::F16),
        (1 << 20, Dtype::F32),
        (128 * 5120 * 2, Dtype::F16), // 27B chunk128 真实载荷
        (SLOT as usize, Dtype::BF16), // 单 slot 满格
    ];
    for (i, (n_bytes, dt)) in cases.iter().enumerate() {
        eprintln!("[host-G1] case {i}: {n_bytes}B {dt:?}");
        let w = width(*dt);
        let n = n_bytes / w;
        let n_alloc = n_bytes.div_ceil(w);
        let a_bytes = from_f32_vec(&lcg_vec(n_alloc, 0x1234_5678 + i as u64), *dt);
        let b_bytes = from_f32_vec(&lcg_vec(n_alloc, 0x9ABC_DEF0 + i as u64), *dt);
        let want = host_ref(&a_bytes[..n * w], &b_bytes[..n * w], *dt);
        let pa = a
            .htod(
                *dt,
                &Shape::from(vec![n_alloc]),
                &a_bytes,
                "hostar:payload-a",
            )
            .await
            .expect("A htod");
        let pb = b
            .htod(
                *dt,
                &Shape::from(vec![n_alloc]),
                &b_bytes,
                "hostar:payload-b",
            )
            .await
            .expect("B htod");
        let (ra, rb) = tokio::join!(
            emit_rank(&mut a, &pa, &own0, &own1, *n_bytes as u64, &layout),
            emit_rank(&mut b, &pb, &p0, &p1, *n_bytes as u64, &layout),
        );
        ra.expect("A emit");
        rb.expect("B emit");
        let (sa, sb) = tokio::join!(a.sync(), b.sync());
        sa.expect("A sync");
        sb.expect("B sync");
        assert_eq!(
            hdr_u32(&read_hdr_host(h0), 28),
            0,
            "A header err != 0(case {i})"
        );
        assert_eq!(
            hdr_u32(&read_hdr_host(h1), 28),
            0,
            "B header err != 0(case {i})"
        );
        for (cl, p, who) in [(&mut a, &pa, "A"), (&mut b, &pb, "B")] {
            let got = read_payload(cl, p, n_alloc * w)
                .await
                .expect("read payload");
            assert_eq!(
                &got[..n * w],
                &want[..],
                "{who} G1 mismatch(case {i}: {n_bytes}B {dt:?})"
            );
        }
        a.free(&[pa.id]).await.expect("free pa");
        b.free(&[pb.id]).await.expect("free pb");
    }

    // ── G4:copy-engine D2H push → 本端 host reduce ─────────────────
    // 目标:验证“生产者 CE 写**对端** host 段,消费者读**本端** host 段”
    // 是否比“CE 写本端、对端 GPU 跨卡读 host”更可靠;这是 H2D pull 前的
    // 最小协议探针。stage 核以 bytes=0 只发布 produced(数据已由 CE 搬走)。
    eprintln!("[host-G4] CE push + local-host reduce");
    {
        let n_bytes = 1 << 20;
        let n = n_bytes / 4;
        let a_bytes = from_f32_vec(&lcg_vec(n, 0xA4A4), Dtype::F32);
        let b_bytes = from_f32_vec(&lcg_vec(n, 0xB4B4), Dtype::F32);
        let want = host_ref(&a_bytes, &b_bytes, Dtype::F32);
        let pa = a
            .htod(Dtype::F32, &Shape::from(vec![n]), &a_bytes, "hostar:g4-a")
            .await
            .expect("htod a");
        let pb = b
            .htod(Dtype::F32, &Shape::from(vec![n]), &b_bytes, "hostar:g4-b")
            .await
            .expect("htod b");
        let prod0 = hdr_u64(&read_hdr_host(h0), 0);
        let prod1 = hdr_u64(&read_hdr_host(h1), 0);
        let slot0 = HDR + (prod0 % N_SLOTS as u64) * SLOT;
        let slot1 = HDR + (prod1 % N_SLOTS as u64) * SLOT;
        // A 的 peer = H1, B 的 peer = H0;CE D2H 写对端 host 槽。
        let (ca, cb) = tokio::join!(
            a.copy_block_at(&pa, 0, &own1, slot0 as usize, n_bytes),
            b.copy_block_at(&pb, 0, &p1, slot1 as usize, n_bytes),
        );
        ca.expect("A CE push");
        cb.expect("B CE push");
        // stage 核至少需要 1 个输出元素(算子准入);写 16B **同一份 payload**
        // 到同一 peer 槽是幂等的 —— 协议发布与数据正确性都不受影响。
        let sig_bytes = 16u64;
        let sig_a = ArStageCall::new(
            pa.clone(),
            own0.clone(),
            own1.clone(),
            own1.clone(),
            sig_bytes,
            layout.slot_bytes,
            0,
        )
        .to_launch();
        let sig_b = ArStageCall::new(
            pb.clone(),
            p0.clone(),
            p1.clone(),
            p1.clone(),
            sig_bytes,
            layout.slot_bytes,
            0,
        )
        .to_launch();
        let (sa, sb) = tokio::join!(a.launch(sig_a), b.launch(sig_b));
        sa.expect("A signal");
        sb.expect("B signal");
        let red_a = ArReduceCall::new(
            pa.clone(),
            own0.clone(),
            own1.clone(),
            own0.clone(),
            n_bytes as u64,
            layout.slot_bytes,
            0,
        )
        .to_launch();
        let red_b = ArReduceCall::new(
            pb.clone(),
            p0.clone(),
            p1.clone(),
            p0.clone(),
            n_bytes as u64,
            layout.slot_bytes,
            0,
        )
        .to_launch();
        let (ra, rb) = tokio::join!(a.launch(red_a), b.launch(red_b));
        ra.expect("A reduce");
        rb.expect("B reduce");
        let (sa, sb) = tokio::join!(a.sync(), b.sync());
        sa.expect("A sync");
        sb.expect("B sync");
        assert_eq!(hdr_u32(&read_hdr_host(h0), 28), 0, "A G4 header err");
        assert_eq!(hdr_u32(&read_hdr_host(h1), 28), 0, "B G4 header err");
        for (cl, p, who) in [(&mut a, &pa, "A"), (&mut b, &pb, "B")] {
            let got = read_payload(cl, p, n_bytes).await.expect("read payload");
            assert_eq!(got, want, "{who} G4 mismatch(CE push)");
        }
        a.free(&[pa.id]).await.expect("free pa");
        b.free(&[pb.id]).await.expect("free pb");
    }

    // ── G5(仅计时,非判据):CE push vs SM push 的真实搬运账 ───────────
    eprintln!("[host-G5] CE push / SM push 计时(8MiB, 5 轮)");
    {
        let n_bytes = SLOT as usize;
        let n = n_bytes / 4;
        let a_bytes = from_f32_vec(&lcg_vec(n, 0xC5C5), Dtype::F32);
        let b_bytes = from_f32_vec(&lcg_vec(n, 0xD5D5), Dtype::F32);
        let pa = a
            .htod(Dtype::F32, &Shape::from(vec![n]), &a_bytes, "hostar:g5-a")
            .await
            .expect("htod a");
        let pb = b
            .htod(Dtype::F32, &Shape::from(vec![n]), &b_bytes, "hostar:g5-b")
            .await
            .expect("htod b");
        let mut ce_total = std::time::Duration::ZERO;
        let mut sm_total = std::time::Duration::ZERO;
        let mut copy_total = std::time::Duration::ZERO;
        for round in 0..5 {
            reset_payload(&mut a, &pa, &a_bytes).await.expect("reset a");
            reset_payload(&mut b, &pb, &b_bytes).await.expect("reset b");
            let prod0 = hdr_u64(&read_hdr_host(h0), 0);
            let prod1 = hdr_u64(&read_hdr_host(h1), 0);
            let slot0 = HDR + (prod0 % N_SLOTS as u64) * SLOT;
            let slot1 = HDR + (prod1 % N_SLOTS as u64) * SLOT;
            // CE 单搬
            let t = std::time::Instant::now();
            let (ca, cb) = tokio::join!(
                a.copy_block_at(&pa, 0, &own1, slot0 as usize, n_bytes),
                b.copy_block_at(&pb, 0, &p1, slot1 as usize, n_bytes),
            );
            ca.expect("A CE");
            cb.expect("B CE");
            let (sa, sb) = tokio::join!(a.sync(), b.sync());
            sa.expect("A sync");
            sb.expect("B sync");
            let copy_dt = t.elapsed();
            copy_total += copy_dt;
            // CE push + local-host reduce
            let t = std::time::Instant::now();
            let (ca, cb) = tokio::join!(
                a.copy_block_at(&pa, 0, &own1, slot0 as usize, n_bytes),
                b.copy_block_at(&pb, 0, &p1, slot1 as usize, n_bytes),
            );
            ca.expect("A CE");
            cb.expect("B CE");
            let sig_a = ArStageCall::new(
                pa.clone(), own0.clone(), own1.clone(), own1.clone(), 16,
                layout.slot_bytes, 0,
            ).to_launch();
            let sig_b = ArStageCall::new(
                pb.clone(), p0.clone(), p1.clone(), p1.clone(), 16,
                layout.slot_bytes, 0,
            ).to_launch();
            let (sa, sb) = tokio::join!(a.launch(sig_a), b.launch(sig_b));
            sa.expect("A signal");
            sb.expect("B signal");
            let red_a = ArReduceCall::new(
                pa.clone(), own0.clone(), own1.clone(), own0.clone(),
                n_bytes as u64, layout.slot_bytes, 0,
            ).to_launch();
            let red_b = ArReduceCall::new(
                pb.clone(), p0.clone(), p1.clone(), p0.clone(),
                n_bytes as u64, layout.slot_bytes, 0,
            ).to_launch();
            let (ra, rb) = tokio::join!(a.launch(red_a), b.launch(red_b));
            ra.expect("A reduce");
            rb.expect("B reduce");
            let (sa, sb) = tokio::join!(a.sync(), b.sync());
            sa.expect("A sync");
            sb.expect("B sync");
            ce_total += t.elapsed();
            // SM push(stage 核全量写对端 host + reduce 读本端 host)
            reset_payload(&mut a, &pa, &a_bytes).await.expect("reset a");
            reset_payload(&mut b, &pb, &b_bytes).await.expect("reset b");
            let t = std::time::Instant::now();
            let st_a = ArStageCall::new(
                pa.clone(), own0.clone(), own1.clone(), own1.clone(),
                n_bytes as u64, layout.slot_bytes, 0,
            ).to_launch();
            let st_b = ArStageCall::new(
                pb.clone(), p0.clone(), p1.clone(), p1.clone(),
                n_bytes as u64, layout.slot_bytes, 0,
            ).to_launch();
            let (sa, sb) = tokio::join!(a.launch(st_a), b.launch(st_b));
            sa.expect("A stage");
            sb.expect("B stage");
            let red_a = ArReduceCall::new(
                pa.clone(), own0.clone(), own1.clone(), own0.clone(),
                n_bytes as u64, layout.slot_bytes, 0,
            ).to_launch();
            let red_b = ArReduceCall::new(
                pb.clone(), p0.clone(), p1.clone(), p0.clone(),
                n_bytes as u64, layout.slot_bytes, 0,
            ).to_launch();
            let (ra, rb) = tokio::join!(a.launch(red_a), b.launch(red_b));
            ra.expect("A reduce");
            rb.expect("B reduce");
            let (sa, sb) = tokio::join!(a.sync(), b.sync());
            sa.expect("A sync");
            sb.expect("B sync");
            sm_total += t.elapsed();
            eprintln!(
                "[host-G5] round {round}: copy={:?} ce_push={:?} sm_push={:?}",
                copy_dt, ce_total / (round + 1), sm_total / (round + 1)
            );
        }
        eprintln!(
            "[host-G5] avg 8MiB: CE copy {:.2} ms / CE push {:.2} ms / SM push {:.2} ms",
            copy_total.as_secs_f64() * 1000.0 / 5.0,
            ce_total.as_secs_f64() * 1000.0 / 5.0,
            sm_total.as_secs_f64() * 1000.0 / 5.0,
        );
        a.free(&[pa.id]).await.expect("free pa");
        b.free(&[pb.id]).await.expect("free pb");
    }

    // ── G2:同输入两跑逐位一致 ─────────────────────────────────────
    eprintln!("[host-G2] 两跑");
    {
        let n_bytes = 1 << 20;
        let n = n_bytes / 4;
        let a_bytes = from_f32_vec(&lcg_vec(n, 0x5151), Dtype::F32);
        let b_bytes = from_f32_vec(&lcg_vec(n, 0x9292), Dtype::F32);
        let want = host_ref(&a_bytes, &b_bytes, Dtype::F32);
        let pa = a
            .htod(Dtype::F32, &Shape::from(vec![n]), &a_bytes, "hostar:g2-a")
            .await
            .expect("htod a");
        let pb = b
            .htod(Dtype::F32, &Shape::from(vec![n]), &b_bytes, "hostar:g2-b")
            .await
            .expect("htod b");
        for round in 0..2 {
            if round > 0 {
                reset_payload(&mut a, &pa, &a_bytes).await.expect("reset a");
                reset_payload(&mut b, &pb, &b_bytes).await.expect("reset b");
            }
            let (ra, rb) = tokio::join!(
                emit_rank(&mut a, &pa, &own0, &own1, n_bytes as u64, &layout),
                emit_rank(&mut b, &pb, &p0, &p1, n_bytes as u64, &layout),
            );
            ra.expect("A emit");
            rb.expect("B emit");
            let (sa, sb) = tokio::join!(a.sync(), b.sync());
            sa.expect("A sync");
            sb.expect("B sync");
            let got = read_payload(&mut a, &pa, n_bytes).await.expect("read");
            assert_eq!(got, want, "G2 round {round}");
        }
        a.free(&[pa.id]).await.expect("free pa");
        b.free(&[pb.id]).await.expect("free pb");
    }

    // ── G3:图捕获 → 两次 replay ────────────────────────────────────
    eprintln!("[host-G3] capture/replay");
    let (ga, gb, pa, pb, a_bytes, b_bytes, want, n_bytes) = {
        let n_bytes = 1 << 20;
        let n = n_bytes / 4;
        let a_bytes = from_f32_vec(&lcg_vec(n, 0x7171), Dtype::F32);
        let b_bytes = from_f32_vec(&lcg_vec(n, 0x8383), Dtype::F32);
        let want = host_ref(&a_bytes, &b_bytes, Dtype::F32);
        let pa = a
            .htod(Dtype::F32, &Shape::from(vec![n]), &a_bytes, "hostar:g3-a")
            .await
            .expect("htod");
        let pb = b
            .htod(Dtype::F32, &Shape::from(vec![n]), &b_bytes, "hostar:g3-b")
            .await
            .expect("htod");
        a.graph_begin().await.expect("A graph_begin");
        b.graph_begin().await.expect("B graph_begin");
        emit_rank(&mut a, &pa, &own0, &own1, n_bytes as u64, &layout)
            .await
            .expect("A cap");
        emit_rank(&mut b, &pb, &p0, &p1, n_bytes as u64, &layout)
            .await
            .expect("B cap");
        let ga = a.graph_end().await.expect("A graph_end");
        let gb = b.graph_end().await.expect("B graph_end");
        (ga, gb, pa, pb, a_bytes, b_bytes, want, n_bytes)
    };
    for round in 0..2 {
        reset_payload(&mut a, &pa, &a_bytes).await.expect("reset a");
        reset_payload(&mut b, &pb, &b_bytes).await.expect("reset b");
        let (la, lb) = tokio::join!(a.graph_launch(ga), b.graph_launch(gb));
        la.expect("A launch");
        lb.expect("B launch");
        let (sa, sb) = tokio::join!(a.sync(), b.sync());
        sa.expect("A sync");
        sb.expect("B sync");
        let got = read_payload(&mut a, &pa, n_bytes).await.expect("read");
        assert_eq!(got, want, "G3 round {round}");
    }
    assert!(
        a.host_release(vec![own0.id]).await.is_err(),
        "图存活时 host_release 应被租约拒收"
    );
    a.graph_destroy(ga).await.expect("A graph_destroy");
    b.graph_destroy(gb).await.expect("B graph_destroy");
    a.free(&[pa.id]).await.expect("free pa");
    b.free(&[pb.id]).await.expect("free pb");

    // ── 收尾:先收 rank1 attach,再收 rank0 owner ────────────────────
    b.host_release(vec![p0.id, p1.id])
        .await
        .expect("B host_release");
    a.host_release(vec![own0.id, own1.id])
        .await
        .expect("A host_release");
    a.close().await.expect("close a");
    b.close().await.expect("close b");
    eprintln!("host_relay_parity 全绿:G1 5 case / G2 2 run / G3 2 replay + 图租约拒收");
}

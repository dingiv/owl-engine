//! **TP2 C3 门 —— allreduce 双 rank 真机对拍(G1/G2/G3)**。
//!
//! 设计出处:`docs/arch/TP2-施工设计.md` §5 门表 / §3 协议。
//!
//! | 门 | 判据 |
//! |---|---|
//! | **G1** host 参考对拍 | 三 dtype × 边界尺寸:`payload_A ← A + B` 逐**位**一致
//!   (核 = `from_float(to_float(dst)+to_float(src))`,与 host 参考同构 IEEE 序) |
//! | **G2** 确定性 | 同输入两跑,输出**逐位**相同 |
//! | **G3** 图捕获 | 捕获 → replay ≡ eager(逐位) |
//!
//! 结构:两 actor(env `OWL_PEER_TEST_DEV_A/B`,未设 = skip)+ VMM 对称握手
//! (S1 的 `peer_alloc`/`peer_import`)+ `owl_comm` 的 chunk 计划展开
//! (`AllReduce::steps`,两侧同 plan ⇒ 序列同构 I5)。
//!
//! ⚠️ **两侧必须并发推进**:reduce 核在流上等对端的 stage(超时 5 s),若
//! 单边串行发完才轮到另一边,会撞超时。本测试用 `tokio::join!` 交错推进
//! (actor 线程各自独立,设备侧真并行)。
//!
//! 运行:
//! ```bash
//! OWL_PEER_TEST_DEV_A=2 OWL_PEER_TEST_DEV_B=3 \
//!   cargo test -p owl-cuda --test comm_ar_parity -- --nocapture
//! ```

use half::{bf16, f16};
use owl_comm::ar::{plan_ar, AllReduce, ArPhase, StagingLayout};
use owl_cuda::{
    Bytes, Command, DeviceClient as _, DeviceSelector, Dtype, GpuClient, GpuServer, ModelError,
    Shape,
};
use owl_kernels::client::comm::{ArReduceCall, ArStageCall};

const HDR: usize = 64;
const SLOT: u64 = 32 << 20; // 设计值(施工设计 §6)
const N_SLOTS: u32 = 2;
const MAGIC: u32 = 0x4F57_4C41; // "OWLA"

// ============================================================================
// 组装 / 工具
// ============================================================================

fn assemble(ordinal: usize, tag: &str) -> GpuClient {
    let (tx, rx) = std::sync::mpsc::channel::<Command>();
    let server = GpuServer::new(rx, DeviceSelector::Ordinal(ordinal), None);
    let client = GpuClient::new(tx);
    std::thread::Builder::new()
        .name(format!("owl-ar-{tag}"))
        .spawn(move || server.run().expect("server run 异常退出"))
        .expect("起 server 线程失败");
    client
}

fn dev(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.parse().ok()
}

fn uuid_of(ordinal: usize) -> String {
    owl_cuda::ffi::device_rows()
        .expect("枚举设备")
        .into_iter()
        .find(|r| r.ordinal == ordinal)
        .map(|r| r.uuid)
        .unwrap_or_else(|| format!("GPU-ord{ordinal}"))
}

/// staging header(布局权威注记在 `cu/comm/ar_stage.cu`)
fn hdr_bytes() -> [u8; HDR] {
    let mut b = [0u8; HDR];
    b[16..20].copy_from_slice(&(SLOT as u32).to_le_bytes());
    b[20..24].copy_from_slice(&N_SLOTS.to_le_bytes());
    b[24..28].copy_from_slice(&MAGIC.to_le_bytes());
    b
}

fn hdr_u32(h: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(h[off..off + 4].try_into().expect("4B"))
}

/// 可复现随机数据(LCG;区间 [-1,1) —— f16 加法后无溢出风险)
fn lcg_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// dtype 宽度(元素字节)
fn width(dt: Dtype) -> usize {
    match dt {
        Dtype::F32 => 4,
        Dtype::F16 | Dtype::BF16 => 2,
        other => panic!("AR 门只测三 dtype,拿到 {other:?}"),
    }
}

/// 字节位型 → f32 向量(与核的 `to_float` 同构)
fn to_f32_vec(bytes: &[u8], dt: Dtype) -> Vec<f32> {
    match dt {
        Dtype::F32 => bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
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

/// f32 向量 → 字节位型(与核的 `from_float` 同构)
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

/// host 参考:`dst = from_float(to_float(a) + to_float(b))`(逐元素;与核同序)
fn host_ref(a: &[u8], b: &[u8], dt: Dtype) -> Vec<u8> {
    let (va, vb) = (to_f32_vec(a, dt), to_f32_vec(b, dt));
    let sum: Vec<f32> = va.iter().zip(vb.iter()).map(|(x, y)| x + y).collect();
    from_f32_vec(&sum, dt)
}

/// staging header 落地(upload_pinned 走 H2D 流;sync 后核才读)
async fn write_hdr(client: &mut GpuClient, staging: &Bytes, tag: &str) -> Result<(), ModelError> {
    let mut pin = client.alloc_pinned(HDR, tag).await?;
    // ⚠️ 租约是**池化区域**(容量 ≥ 请求)—— 只写前 HDR 字节
    pin.slice_bytes_mut()[..HDR].copy_from_slice(&hdr_bytes());
    client.upload_pinned(pin, staging, 0, HDR).await?;
    client.sync().await
}

/// payload 重写(upload_pinned + sync;G2/G3 的"重置"用)
async fn reset_payload(client: &mut GpuClient, b: &Bytes, data: &[u8]) -> Result<(), ModelError> {
    let n = data.len();
    let mut pin = client.alloc_pinned(n, "ar:reset").await?;
    pin.slice_bytes_mut()[..n].copy_from_slice(data);
    client.upload_pinned(pin, b, 0, n).await?;
    client.sync().await
}

/// 单 rank 的发射序列(stage_i → reduce_i;两侧必须同 plan —— 调用点保证)
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
                c.bytes,
                layout.slot_bytes,
                c.offset,
            )
            .to_launch(),
            ArPhase::Reduce => ArReduceCall::new(
                payload.clone(),
                own.clone(),
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

/// 读回 payload(整块)
async fn read_payload(client: &mut GpuClient, b: &Bytes, n_bytes: usize) -> Result<Vec<u8>, ModelError> {
    let mut buf = vec![0u8; n_bytes];
    client.dtoh(b, &mut buf).await?;
    Ok(buf)
}

/// 读回 staging 的 header(前 64 B)
async fn read_hdr(client: &mut GpuClient, staging: &Bytes) -> Result<[u8; HDR], ModelError> {
    let mut buf = vec![0u8; HDR];
    client.dtoh(staging, &mut buf).await?;
    let mut out = [0u8; HDR];
    out.copy_from_slice(&buf);
    Ok(out)
}

// ============================================================================
// 门
// ============================================================================

#[tokio::test]
async fn comm_ar_parity() {
    let (Some(oa), Some(ob)) = (dev("OWL_PEER_TEST_DEV_A"), dev("OWL_PEER_TEST_DEV_B")) else {
        eprintln!("skip: 需 OWL_PEER_TEST_DEV_A / OWL_PEER_TEST_DEV_B(两张可 P2P 的空闲卡)");
        return;
    };
    assert_ne!(oa, ob, "A/B 必须是两张不同卡");
    let (ua, ub) = (uuid_of(oa), uuid_of(ob));
    let mut a = assemble(oa, "a");
    let mut b = assemble(ob, "b");

    // ── VMM 对称握手(S1):staging + 跨卡别名 ─────────────────────────
    let staging_bytes = HDR as u64 + SLOT * N_SLOTS as u64;
    let (staging_a, cred_a) = a.peer_alloc(staging_bytes as usize, "ar:staging-a").await.expect("A PeerAlloc");
    let (staging_b, cred_b) = b.peer_alloc(staging_bytes as usize, "ar:staging-b").await.expect("B PeerAlloc");
    let alias_b_on_a = a
        .peer_import(cred_b, staging_bytes as usize, &ub, "ar:alias-b@a")
        .await
        .expect("A 导入 B");
    let alias_a_on_b = b
        .peer_import(cred_a, staging_bytes as usize, &ua, "ar:alias-a@b")
        .await
        .expect("B 导入 A");
    write_hdr(&mut a, &staging_a, "ar:hdr-a").await.expect("A header");
    write_hdr(&mut b, &staging_b, "ar:hdr-b").await.expect("B header");

    let layout = StagingLayout::new(SLOT, N_SLOTS);

    // ── G1:host 参考对拍(三 dtype × 边界尺寸)──────────────────────
    // 尺寸集:`4B`(单 f32 元素)/`10KB`/`1MB` 三 dtype;`32MiB`/`32MiB+1`(跨 chunk
    // 边界)calls = 1/2 —— 大尺寸只 f32(控时长)。
    let small: [(usize, Dtype); 5] = [
        (4, Dtype::F32),
        (4, Dtype::F16),
        (4, Dtype::BF16),
        (10 << 10, Dtype::F32),
        (10 << 10, Dtype::F16),
    ];
    let big: [(usize, Dtype); 3] = [
        (1 << 20, Dtype::F32),
        ((SLOT as usize), Dtype::F32),
        ((SLOT as usize) + 1, Dtype::F32),
    ];
    let mut g1_cases = 0usize;
    for (i, (bytes, dt)) in small.iter().chain(big.iter()).enumerate() {
        eprintln!("[G1] case {i}: {bytes}B {dt:?} …");
        let w = width(*dt);
        // 元素取整:参与 AR 的完整元素 = bytes/w;分配 = 向上取整(尾部不满元素
        // 的字节**不得被动** —— 这正是 case 7(32MiB+1)要证的边界)
        let n_elems = bytes / w;
        let n_alloc = bytes.div_ceil(w);
        let a_bytes = from_f32_vec(&lcg_vec(n_alloc, 0x1234_5678 + i as u64), *dt);
        let b_bytes = from_f32_vec(&lcg_vec(n_alloc, 0x9ABC_DEF0 + i as u64), *dt);
        let payload_a = a
            .htod(*dt, &Shape::from(vec![n_alloc]), &a_bytes, "ar:payload-a")
            .await
            .expect("A payload htod");
        let payload_b = b
            .htod(*dt, &Shape::from(vec![n_alloc]), &b_bytes, "ar:payload-b")
            .await
            .expect("B payload htod");
        let head = n_elems * w;
        let want = host_ref(&a_bytes[..head], &b_bytes[..head], *dt);

        let (ra, rb) = tokio::join!(
            emit_rank(&mut a, &payload_a, &staging_a, &alias_b_on_a, *bytes as u64, &layout),
            emit_rank(&mut b, &payload_b, &staging_b, &alias_a_on_b, *bytes as u64, &layout),
        );
        ra.expect("A 发射序列");
        rb.expect("B 发射序列");
        let (sa, sb) = tokio::join!(a.sync(), b.sync());
        sa.expect("A sync");
        sb.expect("B sync");

        // err 面:两侧 header 都必须无错
        for (h, who) in [(read_hdr(&mut a, &staging_a), "A"), (read_hdr(&mut b, &staging_b), "B")] {
            let h = h.await.expect("读 header");
            assert_eq!(hdr_u32(&h, 28), 0, "{who} 侧 err 非 0(case {i}: {bytes}B {dt:?})");
        }
        // 两侧结果都必须 == A + B(对称 allreduce;头部逐位 + 尾部不动)
        // 尾部期望 = **各侧自己的初始数据**(未被核处理的字节)
        for (cl, p, who, init) in [
            (&mut a, &payload_a, "A", &a_bytes),
            (&mut b, &payload_b, "B", &b_bytes),
        ] {
            let got = read_payload(cl, p, n_alloc * w).await.expect("读 payload");
            if got[..head] != want[..] {
                let pos = got[..head].iter().zip(want.iter()).position(|(x, y)| x != y);
                panic!(
                    "{who} 侧 G1 对拍失败(case {i}: {bytes}B {dt:?});首个不同字节 @ {pos:?}\n  got  = {:?}\n  want = {:?}",
                    got.get(pos.unwrap_or(0)..pos.unwrap_or(0) + 8.min(got.len())),
                    want.get(pos.unwrap_or(0)..pos.unwrap_or(0) + 8.min(want.len())),
                );
            }
            assert_eq!(
                &got[head..],
                &init[head..],
                "{who} 侧 case {i}: 非整元素尾部字节被核动了(应为原值)"
            );
        }
        a.free(&[payload_a.id]).await.expect("free a");
        b.free(&[payload_b.id]).await.expect("free b");
        g1_cases += 1;
        eprintln!("[G1] case {i} ok");
    }

    // ── G2:确定性(同输入两跑逐位相同)────────────────────────────
    eprintln!("[G2] 开始");
    {
        let bytes = 1 << 20;
        let n = bytes / 4;
        let a_bytes = from_f32_vec(&lcg_vec(n, 0xD00D), Dtype::F32);
        let b_bytes = from_f32_vec(&lcg_vec(n, 0xBEEF), Dtype::F32);
        let want = host_ref(&a_bytes, &b_bytes, Dtype::F32);
        let mut r1: Vec<u8> = Vec::new();
        for round in 0..2 {
            let payload_a = a
                .htod(Dtype::F32, &Shape::from(vec![n]), &a_bytes, "ar:g2-a")
                .await
                .expect("htod");
            let payload_b = b
                .htod(Dtype::F32, &Shape::from(vec![n]), &b_bytes, "ar:g2-b")
                .await
                .expect("htod");
            let (ra, rb) = tokio::join!(
                emit_rank(&mut a, &payload_a, &staging_a, &alias_b_on_a, bytes as u64, &layout),
                emit_rank(&mut b, &payload_b, &staging_b, &alias_a_on_b, bytes as u64, &layout),
            );
            ra.expect("G2 A");
            rb.expect("G2 B");
            let (sa, sb) = tokio::join!(a.sync(), b.sync());
            sa.expect("G2 sync a");
            sb.expect("G2 sync b");
            let got = read_payload(&mut a, &payload_a, bytes).await.expect("G2 读");
            assert_eq!(got, want, "G2 round {round} 与 host 参考不符");
            if round == 0 {
                r1 = got;
            } else {
                assert_eq!(got, r1, "G2:两跑逐位不同(非确定性)");
            }
            a.free(&[payload_a.id]).await.expect("free");
            b.free(&[payload_b.id]).await.expect("free");
        }
    }

    // ── G3:图捕获(捕获 → replay ≡ eager)──────────────────────────
    eprintln!("[G3] 开始");
    {
        let bytes = 1 << 20;
        let n = bytes / 4;
        let a_bytes = from_f32_vec(&lcg_vec(n, 0x5151), Dtype::F32);
        let b_bytes = from_f32_vec(&lcg_vec(n, 0x9292), Dtype::F32);
        let want = host_ref(&a_bytes, &b_bytes, Dtype::F32);
        let payload_a = a
            .htod(Dtype::F32, &Shape::from(vec![n]), &a_bytes, "ar:g3-a")
            .await
            .expect("htod");
        let payload_b = b
            .htod(Dtype::F32, &Shape::from(vec![n]), &b_bytes, "ar:g3-b")
            .await
            .expect("htod");

        // 捕获(窗内只发射;两侧都捕)
        a.graph_begin().await.expect("A graph_begin");
        b.graph_begin().await.expect("B graph_begin");
        emit_rank(&mut a, &payload_a, &staging_a, &alias_b_on_a, bytes as u64, &layout)
            .await
            .expect("A 窗内发射");
        emit_rank(&mut b, &payload_b, &staging_b, &alias_a_on_b, bytes as u64, &layout)
            .await
            .expect("B 窗内发射");
        let ga = a.graph_end().await.expect("A graph_end");
        let gb = b.graph_end().await.expect("B graph_end");

        // 重置 payload(eager 基线语义:payload = 原始数据)
        reset_payload(&mut a, &payload_a, &a_bytes).await.expect("reset a");
        reset_payload(&mut b, &payload_b, &b_bytes).await.expect("reset b");

        // replay(两侧并发提交 → 两侧 sync)
        let (la, lb) = tokio::join!(a.graph_launch(ga), b.graph_launch(gb));
        la.expect("A graph_launch");
        lb.expect("B graph_launch");
        let (sa, sb) = tokio::join!(a.sync(), b.sync());
        sa.expect("G3 sync a");
        sb.expect("G3 sync b");

        for (cl, p, who) in [(&mut a, &payload_a, "A"), (&mut b, &payload_b, "B")] {
            let got = read_payload(cl, p, bytes).await.expect("G3 读");
            assert_eq!(got, want, "G3({who}):replay 结果 ≠ host 参考");
        }
        // 图的进一步回放:再 replay 一次(游标继续推进;结果仍正确 —— 相对量 I3)
        reset_payload(&mut a, &payload_a, &a_bytes).await.expect("reset a2");
        reset_payload(&mut b, &payload_b, &b_bytes).await.expect("reset b2");
        let (la, lb) = tokio::join!(a.graph_launch(ga), b.graph_launch(gb));
        la.expect("A graph_launch 2");
        lb.expect("B graph_launch 2");
        let (sa, sb) = tokio::join!(a.sync(), b.sync());
        sa.expect("sync");
        sb.expect("sync");
        let got = read_payload(&mut a, &payload_a, bytes).await.expect("G3 二回读");
        assert_eq!(got, want, "G3:第二次 replay 结果 ≠ host 参考(游标推进应无影响)");
        // 图钉住 staging/alias/payload(捕获窗 block_ptr 登记租约)⇒ 先销图再回收
        // (顺带验证:租约未解除时 peer_release/free 会被**响亮拒收** —— 见下断言)
        assert!(
            a.peer_release(vec![staging_a.id]).await.is_err(),
            "图存活时 peer_release 应被图引用租约拒收"
        );
        a.graph_destroy(ga).await.expect("A graph_destroy");
        b.graph_destroy(gb).await.expect("B graph_destroy");
        a.free(&[payload_a.id]).await.expect("free");
        b.free(&[payload_b.id]).await.expect("free");
    }

    eprintln!("[收尾] peer_release + close");
    // ── 收尾(peer 块显式释放;闭 G5 语义)────────────────────────
    a.peer_release(vec![staging_a.id, alias_b_on_a.id]).await.expect("A 释放");
    b.peer_release(vec![staging_b.id, alias_a_on_b.id]).await.expect("B 释放");
    a.close().await.expect("close a");
    b.close().await.expect("close b");
    eprintln!(
        "comm_ar_parity 全绿:G1 {g1_cases} case(三 dtype + 32MiB 跨 chunk)/ G2 两跑逐位同 / G3 捕获 replay ≡ eager(双 replay)"
    );
}

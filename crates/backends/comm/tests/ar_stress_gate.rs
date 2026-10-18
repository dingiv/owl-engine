//! **AR 协议压力门**(2026-10-17;**无模型、纯通信**)—— 抓间歇性错。
//!
//! # 为什么另立一门
//!
//! `ar_transport_gate`(G1/G2/G3)每个 case **只跑一次**:它能证明"某一次对",
//! **证明不了"每次都对"**。而 2026-10-17 那个间歇性分歧(`tp_transport_bits_gate`
//! 在 3090 Ti 对上偶发"第 3 个 token 起分歧")正是**时序竞态** —— 这类错只能靠
//! **多轮 + 每轮判据**压出来。
//!
//! 本门**不加载模型**(2 rank 直接起、直接发射 AR)⇒ 迭代成本 = 秒级 ⇒ 可以
//! 几百轮地压,并且每轮都把现场打出来(轮次 / 首个不同字节 / 两侧 header 的 err)。
//!
//! # 判据(每轮,不抽样)
//!
//! 1. 两侧 `err`(=header +28)必须为 **0**:1=credit 超时 · 2=数据超时 · 3=校验失败;
//! 2. 两侧 payload 与 host 参考(`from_float(to_float(a)+to_float(b))`,逐位)一致;
//! 3. 失败即 `panic!`,报 **轮次 + 首个不同字节下标 + err 码**(留现场,不静默)。
//!
//! # 运行
//!
//! ```bash
//! OWL_PEER_TEST_DEV_A=0 OWL_PEER_TEST_DEV_B=1 \
//!   cargo test -p owl-comm --test ar_stress_gate -- --nocapture
//! ```
//! `OWL_STRESS_ITERS`(默认 300)· `OWL_STRESS_BYTES`(默认 10240 = decode 一行)
//! · `OWL_STRESS_DTYPE`(f16/f32/bf16)。未设 DEV_A/B ⇒ skip。

use half::{bf16, f16};
use owl_comm::ar::{plan_ar, AllReduce, ArPhase, StagingLayout};
use owl_cuda::{
    Bytes, Command, DeviceClient as _, DeviceSelector, Dtype, GpuClient, GpuServer, ModelError,
    Shape,
};
use owl_kernels::client::comm::{ArReduceCall, ArStageCall};

const HDR: usize = 64;
const SLOT: u64 = 32 << 20;
const N_SLOTS: u32 = 2;
const MAGIC: u32 = 0x4F57_4C41;

// ---------------------------------------------------------------- 组装/工具

fn assemble(ordinal: usize, tag: &str) -> GpuClient {
    let (tx, rx) = std::sync::mpsc::channel::<Command>();
    let server = GpuServer::new(rx, DeviceSelector::Ordinal(ordinal), None);
    let client = GpuClient::new(tx);
    std::thread::Builder::new()
        .name(format!("owl-stress-{tag}"))
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

fn hdr_bytes() -> [u8; HDR] {
    let mut b = [0u8; HDR];
    b[16..20].copy_from_slice(&(SLOT as u32).to_le_bytes());
    b[20..24].copy_from_slice(&N_SLOTS.to_le_bytes());
    b[24..28].copy_from_slice(&MAGIC.to_le_bytes());
    b
}

fn width(dt: Dtype) -> usize {
    match dt {
        Dtype::F32 => 4,
        Dtype::F16 | Dtype::BF16 => 2,
        other => panic!("压力门只测三 dtype,拿到 {other:?}"),
    }
}

fn to_f32(bytes: &[u8], dt: Dtype) -> Vec<f32> {
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

fn from_f32(vals: &[f32], dt: Dtype) -> Vec<u8> {
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

/// 可复现随机数据(|v| ≤ 1;区间窄 ⇒ f16 加法无溢出,且逐位可判)
fn lcg_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn host_ref(a: &[u8], b: &[u8], dt: Dtype) -> Vec<u8> {
    let (va, vb) = (to_f32(a, dt), to_f32(b, dt));
    let sum: Vec<f32> = va.iter().zip(vb.iter()).map(|(x, y)| x + y).collect();
    from_f32(&sum, dt)
}

async fn write_hdr(client: &mut GpuClient, staging: &Bytes, tag: &str) -> Result<(), ModelError> {
    let mut pin = client.alloc_pinned(HDR, tag).await?;
    pin.slice_bytes_mut()[..HDR].copy_from_slice(&hdr_bytes());
    client.upload_pinned(pin, staging, 0, HDR).await?;
    client.sync().await
}

async fn reset_payload(client: &mut GpuClient, b: &Bytes, data: &[u8], tag: &str) -> Result<(), ModelError> {
    let n = data.len();
    let mut pin = client.alloc_pinned(n, tag).await?;
    pin.slice_bytes_mut()[..n].copy_from_slice(data);
    client.upload_pinned(pin, b, 0, n).await?;
    client.sync().await
}

/// 单 rank 一次完整 AR(生产同序:`stage_i → reduce_i`;两侧同 plan ⇒ I5)
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

/// 读 header 前 64B(取 err 用)
async fn read_hdr(client: &mut GpuClient, staging: &Bytes) -> Result<[u8; HDR], ModelError> {
    let mut buf = vec![0u8; HDR];
    client.dtoh(staging, &mut buf).await?;
    let mut out = [0u8; HDR];
    out.copy_from_slice(&buf);
    Ok(out)
}

fn u32of(h: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(h[off..off + 4].try_into().unwrap())
}
fn u64of(h: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(h[off..off + 8].try_into().unwrap())
}

// ---------------------------------------------------------------- 压力门

#[tokio::test]
async fn ar_stress() {
    let (Some(oa), Some(ob)) = (dev("OWL_PEER_TEST_DEV_A"), dev("OWL_PEER_TEST_DEV_B")) else {
        eprintln!("skip: 需 OWL_PEER_TEST_DEV_A / OWL_PEER_TEST_DEV_B(两张可 P2P 的空闲卡)");
        return;
    };
    assert_ne!(oa, ob);
    let iters: u32 = std::env::var("OWL_STRESS_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(300);
    let bytes: usize = std::env::var("OWL_STRESS_BYTES").ok().and_then(|v| v.parse().ok()).unwrap_or(10240);
    let dt = match std::env::var("OWL_STRESS_DTYPE").as_deref() {
        Ok("f32") => Dtype::F32,
        Ok("bf16") => Dtype::BF16,
        _ => Dtype::F16,
    };
    let w = width(dt);
    let n_elems = bytes / w;
    let layout = StagingLayout::new(SLOT, N_SLOTS);

    let (ua, ub) = (uuid_of(oa), uuid_of(ob));
    let mut a = assemble(oa, "a");
    let mut b = assemble(ob, "b");
    let staging_bytes = HDR as u64 + SLOT * N_SLOTS as u64;
    let (staging_a, cred_a) = a.peer_alloc(staging_bytes as usize, "st:staging-a").await.expect("A PeerAlloc");
    let (staging_b, cred_b) = b.peer_alloc(staging_bytes as usize, "st:staging-b").await.expect("B PeerAlloc");
    let alias_b_on_a = a.peer_import(cred_b, staging_bytes as usize, &ub, "st:alias-b@a").await.expect("A 导入 B");
    let alias_a_on_b = b.peer_import(cred_a, staging_bytes as usize, &ua, "st:alias-a@b").await.expect("B 导入 A");
    write_hdr(&mut a, &staging_a, "st:hdr-a").await.expect("A header");
    write_hdr(&mut b, &staging_b, "st:hdr-b").await.expect("B header");

    let a_bytes = from_f32(&lcg_vec(n_elems, 0xA5A5_1234), dt);
    let b_bytes = from_f32(&lcg_vec(n_elems, 0x5A5A_9876), dt);
    let want = host_ref(&a_bytes, &b_bytes, dt);
    let payload_a = a.htod(dt, &Shape::from(vec![n_elems]), &a_bytes, "st:payload-a").await.expect("A htod");
    let payload_b = b.htod(dt, &Shape::from(vec![n_elems]), &b_bytes, "st:payload-b").await.expect("B htod");

    eprintln!(
        "[stress] devices A=ord{oa} B=ord{ob} | iters={iters} bytes={bytes} dtype={dt:?} \
         | 判据:err==0 且 payload 逐位 == a+b"
    );

    for it in 0..iters {
        // 每轮复位(⇒ 期望值恒定 = a+b;htod 的 host 往返不参与判据)
        reset_payload(&mut a, &payload_a, &a_bytes, "st:rst-a").await.expect("A reset");
        reset_payload(&mut b, &payload_b, &b_bytes, "st:rst-b").await.expect("B reset");

        let (ra, rb) = tokio::join!(
            emit_rank(&mut a, &payload_a, &staging_a, &alias_b_on_a, bytes as u64, &layout),
            emit_rank(&mut b, &payload_b, &staging_b, &alias_a_on_b, bytes as u64, &layout),
        );
        ra.expect("A 发射序列");
        rb.expect("B 发射序列");
        let (sa, sb) = tokio::join!(a.sync(), b.sync());
        sa.expect("A sync");
        sb.expect("B sync");

        // ① err 码(两侧) —— 现场打出来,这是"5s 超时"那类错的直接判据
        for (cl, st, who) in [(&mut a, &staging_a, "A"), (&mut b, &staging_b, "B")] {
            let h = read_hdr(cl, st).await.expect("读 header");
            let err = u32of(&h, 28);
            if err != 0 {
                panic!(
                    "[stress] ✗ 轮 {it} {who} 侧 err={err}(1=credit/rendezvous 超时 · \
                     2=数据超时 · 3=校验失败)| header: produced={} consumed={} wait_done={} \
                     blocks_done={}",
                    u64of(&h, 0),
                    u64of(&h, 8),
                    u64of(&h, 32),
                    u32of(&h, 40)
                );
            }
        }

        // ② 逐位对拍(两侧都要 == a+b)
        for (cl, p, who) in [(&mut a, &payload_a, "A"), (&mut b, &payload_b, "B")] {
            let mut got = vec![0u8; n_elems * w];
            cl.dtoh(p, &mut got).await.expect("读 payload");
            if got != want {
                let pos = got.iter().zip(want.iter()).position(|(x, y)| x != y).unwrap_or(0);
                let h = read_hdr(cl, if who == "A" { &staging_a } else { &staging_b }).await.expect("hdr");
                panic!(
                    "[stress] ✗ 轮 {it} {who} 侧 payload 逐位不符:首个不同字节 @ {pos} \
                     (got={:?} want={:?})| header: produced={} consumed={} wait_done={} \
                     blocks_done={} err={}",
                    &got[pos..(pos + 4).min(got.len())],
                    &want[pos..(pos + 4).min(want.len())],
                    u64of(&h, 0),
                    u64of(&h, 8),
                    u64of(&h, 32),
                    u32of(&h, 40),
                    u32of(&h, 28)
                );
            }
        }
        if it % 50 == 0 || it + 1 == iters {
            eprintln!("[stress] 轮 {}/{} ok", it + 1, iters);
        }
    }
    eprintln!("[stress] 全绿:{iters} 轮 × (err==0 且逐位 == a+b)");
}

// ---------------------------------------------------------------- 图回放压力

/// **捕获一次 → 回放 K 轮,每轮验 err 与逐位**(B 段)。
///
/// 为什么单列:AR 在生产里是**在图里回放**的(整步一次 `graph_launch`),而
/// `ar_transport_gate` 的 G3 只回放**两次**。协议里所有状态都是**相对游标**
/// (produced/consumed 设备内自增;`wait_done` 单调;`blocks_done` 末块回零)
/// —— "图回放安全"是设计承诺,**这里就是它的直接判据**:
/// 若哪一格烘了绝对值/没回零,回放到第 N 轮必然出错(err 或数值)。
///
/// 期望值可算:每轮 `payload ← payload + peer_slot`,而 peer_slot 每轮等于对端
/// 当时的 payload ⇒ 第 k 轮两侧都 = `2^(k-1)·(a+b)`(f32 下 k ≤ 20 精确可表示)。
#[tokio::test]
async fn ar_stress_graph() {
    let (Some(oa), Some(ob)) = (dev("OWL_PEER_TEST_DEV_A"), dev("OWL_PEER_TEST_DEV_B")) else {
        eprintln!("skip: 需 OWL_PEER_TEST_DEV_A / OWL_PEER_TEST_DEV_B");
        return;
    };
    let replays: u32 = std::env::var("OWL_STRESS_REPLAYS").ok().and_then(|v| v.parse().ok()).unwrap_or(20);
    // f32 + 窄区间:2^k 精确(不撞精度/溢出)
    let bytes = 1 << 20; // 1MiB(f32 = 262144 元素 > 64KB 门槛 ⇒ 走多 block 路径)
    let dt = Dtype::F32;
    let n = bytes / 4;
    let layout = StagingLayout::new(SLOT, N_SLOTS);
    let (ua, ub) = (uuid_of(oa), uuid_of(ob));
    let mut a = assemble(oa, "ga");
    let mut b = assemble(ob, "gb");
    let staging_bytes = HDR as u64 + SLOT * N_SLOTS as u64;
    let (staging_a, cred_a) = a.peer_alloc(staging_bytes as usize, "sg:staging-a").await.expect("A PeerAlloc");
    let (staging_b, cred_b) = b.peer_alloc(staging_bytes as usize, "sg:staging-b").await.expect("B PeerAlloc");
    let alias_b_on_a = a.peer_import(cred_b, staging_bytes as usize, &ub, "sg:alias-b@a").await.expect("A 导入 B");
    let alias_a_on_b = b.peer_import(cred_a, staging_bytes as usize, &ua, "sg:alias-a@b").await.expect("B 导入 A");
    write_hdr(&mut a, &staging_a, "sg:hdr-a").await.expect("A header");
    write_hdr(&mut b, &staging_b, "sg:hdr-b").await.expect("B header");

    let a_bytes = from_f32(&lcg_vec(n, 0x1357_9BDF), dt);
    let b_bytes = from_f32(&lcg_vec(n, 0x2468_ACE0), dt);
    let payload_a = a.htod(dt, &Shape::from(vec![n]), &a_bytes, "sg:pl-a").await.expect("A htod");
    let payload_b = b.htod(dt, &Shape::from(vec![n]), &b_bytes, "sg:pl-b").await.expect("B htod");

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

    // 期望值迭代:`exp_k = exp_{k-1} + exp_{k-1}`(= 2^(k-1)(a+b),f32 精确)
    let mut exp = host_ref(&a_bytes, &b_bytes, dt);
    eprintln!("[graph-stress] 捕获完成;回放 {replays} 轮(bytes={bytes}, f32, 判据:每轮 err==0 且逐位 == 2^(k-1)(a+b))");

    for k in 1..=replays {
        // ⚠️ 递推要从 k=2 起:第 1 轮两侧都只是 `a+b`(peer_slot 就是对方的初始值);
        //    写成"先加倍"会得 `2(a+b)` —— 差正好一倍(初版就这么错的,尾数同、阶码差 1)。
        if k > 1 {
            exp = host_ref(&exp, &exp, dt);
        }
        let (la, lb) = tokio::join!(a.graph_launch(ga), b.graph_launch(gb));
        la.expect("A graph_launch");
        lb.expect("B graph_launch");
        let (sa, sb) = tokio::join!(a.sync(), b.sync());
        sa.expect("A sync");
        sb.expect("B sync");

        for (cl, st, who) in [(&mut a, &staging_a, "A"), (&mut b, &staging_b, "B")] {
            let h = read_hdr(cl, st).await.expect("读 header");
            let err = u32of(&h, 28);
            if err != 0 {
                panic!(
                    "[graph-stress] ✗ 回放 {k} {who} 侧 err={err}(1=credit/rendezvous 超时 · \
                     2=数据超时 · 3=校验失败)| produced={} consumed={} wait_done={} blocks_done={}",
                    u64of(&h, 0),
                    u64of(&h, 8),
                    u64of(&h, 32),
                    u32of(&h, 40)
                );
            }
        }
        for (cl, p, who) in [(&mut a, &payload_a, "A"), (&mut b, &payload_b, "B")] {
            let mut got = vec![0u8; n * 4];
            cl.dtoh(p, &mut got).await.expect("读 payload");
            if got != exp {
                let pos = got.iter().zip(exp.iter()).position(|(x, y)| x != y).unwrap_or(0);
                panic!(
                    "[graph-stress] ✗ 回放 {k} {who} 侧逐位不符:首个不同字节 @ {pos} \
                     (got={:?} want={:?})—— 相对游标/回零哪一格烘了绝对值?",
                    &got[pos..(pos + 4).min(got.len())],
                    &exp[pos..(pos + 4).min(exp.len())]
                );
            }
        }
        if k % 5 == 0 || k == replays {
            eprintln!("[graph-stress] 回放 {k}/{replays} ok");
        }
    }
    eprintln!("[graph-stress] 全绿:{replays} 轮回放 × (err==0 且逐位)—— 相对游标 + 回放安全 ✓");
}

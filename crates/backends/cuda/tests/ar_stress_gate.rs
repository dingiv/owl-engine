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
//!   cargo test -p owl-cuda --test ar_stress_gate -- --nocapture
//! ```
//! `OWL_STRESS_ITERS`(默认 300)· `OWL_STRESS_BYTES`(默认 10240 = decode 一行)
//! · `OWL_STRESS_DTYPE`(f16/f32/bf16)。未设 DEV_A/B ⇒ skip。

use half::{bf16, f16};
use std::time::Instant;
use owl_cuda::comm::ar::{plan_ar, AllReduce, ArPhase, StagingLayout};
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
    hdr_bytes_at(0, 0)
}

/// 带初值的 header(`produced` / `consumed`;§38 的图内重叠探针用 HUGE 旁路)
fn hdr_bytes_at(produced: u64, consumed: u64) -> [u8; HDR] {
    let mut b = [0u8; HDR];
    b[0..8].copy_from_slice(&produced.to_le_bytes());
    b[8..16].copy_from_slice(&consumed.to_le_bytes());
    b[16..20].copy_from_slice(&(SLOT as u32).to_le_bytes());
    b[20..24].copy_from_slice(&N_SLOTS.to_le_bytes());
    b[24..28].copy_from_slice(&MAGIC.to_le_bytes());
    b
}

/// 与 [`write_hdr_at`] 同,但**自定义 `slot_bytes`**(核会校验 header 声明与实参一致)
fn hdr_bytes_full(produced: u64, consumed: u64, slot_bytes: u32) -> [u8; HDR] {
    let mut b = hdr_bytes_at(produced, consumed);
    b[16..20].copy_from_slice(&slot_bytes.to_le_bytes());
    b
}

async fn write_hdr_full(
    client: &mut GpuClient,
    staging: &Bytes,
    tag: &str,
    produced: u64,
    consumed: u64,
    slot_bytes: u32,
) -> Result<(), ModelError> {
    let mut pin = client.alloc_pinned(HDR, tag).await?;
    pin.slice_bytes_mut()[..HDR].copy_from_slice(&hdr_bytes_full(produced, consumed, slot_bytes));
    client.upload_pinned(pin, staging, 0, HDR).await?;
    client.sync().await
}

async fn write_hdr_at(
    client: &mut GpuClient,
    staging: &Bytes,
    tag: &str,
    produced: u64,
    consumed: u64,
) -> Result<(), ModelError> {
    let mut pin = client.alloc_pinned(HDR, tag).await?;
    pin.slice_bytes_mut()[..HDR].copy_from_slice(&hdr_bytes_at(produced, consumed));
    client.upload_pinned(pin, staging, 0, HDR).await?;
    client.sync().await
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
                own.clone(), // data:v1 = 写本端槽
                c.bytes,
                layout.slot_bytes,
                c.offset,
            )
            .to_launch(),
            ArPhase::Reduce => ArReduceCall::new(
                payload.clone(),
                own.clone(),
                peer.clone(),
                peer.clone(), // data:v1 = 读对端槽
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

// ============================================================================
// ★ **CE × SM 并行探针**(2026-10-18;sglang TBO 的最小验证)
// ============================================================================
//
// **问的问题**:把"不占 SM 的搬运(CE)"与"占 SM 的核"放进**同一张图的两条链**,
// 它们能不能真并行?这是"微批交错能不能拿到重叠"的**唯一前提** —— 也是 §33 那个
// 探针没能回答的问题(那次用 add 链当伴侣,而 add 是访存受限 ⇒ **判据与被测共享
// 瓶颈,无效**)。
//
// * **SM 链** = `emit_rank`(v1 两段 AR:stage 核 + reduce 核 —— 后者在读对端期间
//   **占着 SM 自旋**,正是要对照的那种"占 SM 的活");
// * **CE 链** = N 次 `copy_block_at`(DMA 引擎干活,**不占 SM**)。
//
// ```
// 档 A: OWL_OVL_CE=0    只有 SM 链                     ⇒ T_A
// 档 B: OWL_OVL_CE=N    再加 N 次 CE 拷(默认 AUX 流) ⇒ T_B
// 对照: OWL_OVL_CE=N OWL_OVL_CE_STREAM=same  (CE 也在 COMPUTE ⇒ 必然串行) ⇒ T_C
// ```
// **判据**:`T_B ≈ T_A` ⇒ 完全重叠(CE 白送);`T_B ≈ T_C` ⇒ 零重叠。
//
// 跑法:
// ```bash
// OWL_PEER_TEST_DEV_A=2 OWL_PEER_TEST_DEV_B=3 OWL_OVL_CE=120 OWL_OVL_AR=8 \
//   cargo test -p owl-cuda --test ar_stress_gate ce_sm_overlap_probe -- --nocapture
// ```
#[tokio::test]
async fn ce_sm_overlap_probe() {
    let (Some(oa), Some(ob)) = (dev("OWL_PEER_TEST_DEV_A"), dev("OWL_PEER_TEST_DEV_B")) else {
        eprintln!("skip: 需 OWL_PEER_TEST_DEV_A / OWL_PEER_TEST_DEV_B");
        return;
    };
    let ce_n: u32 = std::env::var("OWL_OVL_CE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let ar_n: u32 = std::env::var("OWL_OVL_AR").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    let k: u32 = std::env::var("OWL_OVL_K").ok().and_then(|v| v.parse().ok()).unwrap_or(20);
    let bytes = 1 << 20;
    let dt = Dtype::F32;
    let n = bytes / 4;
    let n_slots = N_SLOTS;
    let layout = StagingLayout::new(SLOT, n_slots);
    let (ua, ub) = (uuid_of(oa), uuid_of(ob));
    let mut a = assemble(oa, "ova");
    let mut b = assemble(ob, "ovb");
    let staging_bytes = HDR as u64 + SLOT * n_slots as u64;
    let (staging_a, cred_a) = a.peer_alloc(staging_bytes as usize, "ov:staging-a").await.expect("A PeerAlloc");
    let (staging_b, cred_b) = b.peer_alloc(staging_bytes as usize, "ov:staging-b").await.expect("B PeerAlloc");
    let alias_b_on_a = a.peer_import(cred_b, staging_bytes as usize, &ub, "ov:alias-b@a").await.expect("A 导入 B");
    let alias_a_on_b = b.peer_import(cred_a, staging_bytes as usize, &ua, "ov:alias-a@b").await.expect("B 导入 A");
    write_hdr(&mut a, &staging_a, "ov:hdr-a").await.expect("A header");
    write_hdr(&mut b, &staging_b, "ov:hdr-b").await.expect("B header");
    let a_bytes = from_f32(&lcg_vec(n, 0x0BAD_F00D), dt);
    let b_bytes = from_f32(&lcg_vec(n, 0xFEED_FACE), dt);
    let payload_a = a.htod(dt, &Shape::from(vec![n]), &a_bytes, "ov:pl-a").await.expect("A htod");
    let payload_b = b.htod(dt, &Shape::from(vec![n]), &b_bytes, "ov:pl-b").await.expect("B htod");

    // ── 捕获:SM 链(COMPUTE)+ 可选 CE 链(AUX)──────────────────────
    a.graph_begin().await.expect("A graph_begin");
    b.graph_begin().await.expect("B graph_begin");
    for _ in 0..ar_n {
        emit_rank(&mut a, &payload_a, &staging_a, &alias_b_on_a, bytes as u64, &layout).await.expect("A AR");
        emit_rank(&mut b, &payload_b, &staging_b, &alias_a_on_b, bytes as u64, &layout).await.expect("B AR");
    }
    if ce_n > 0 {
        let aux = std::env::var("OWL_OVL_CE_STREAM").map(|v| v != "same").unwrap_or(true);
        if aux {
            a.capture_route_to(owl_cuda::STREAM_COMPUTE_AUX as u8).await.expect("A fork→AUX");
        }
        for _ in 0..ce_n {
            // 对端 staging → 本端 payload(跨卡 DMA;目的地只为"有个在册终点")
            a.copy_block_at(&alias_b_on_a, 0, &payload_a, 0, bytes).await.expect("A CE 拷");
        }
        if aux {
            a.capture_route_to(1).await.expect("A 切回 COMPUTE");
        }
    }
    let ga = a.graph_end().await.expect("A graph_end");
    let gb = b.graph_end().await.expect("B graph_end");

    for _ in 0..3 {
        let (la, lb) = tokio::join!(a.graph_launch(ga), b.graph_launch(gb));
        la.expect("A warm"); lb.expect("B warm");
        let (sa, sb) = tokio::join!(a.sync(), b.sync());
        sa.expect("A sync"); sb.expect("B sync");
    }
    let t0 = Instant::now();
    for _ in 0..k {
        let (la, lb) = tokio::join!(a.graph_launch(ga), b.graph_launch(gb));
        la.expect("A replay"); lb.expect("B replay");
    }
    let (sa, sb) = tokio::join!(a.sync(), b.sync());
    sa.expect("A sync"); sb.expect("B sync");
    let ms = t0.elapsed().as_secs_f64() * 1e3 / f64::from(k);
    // 自证:AR 链真在跑(err != 0 就是超时/校验失败)
    for (cl, st, who) in [(&mut a, &staging_a, "A"), (&mut b, &staging_b, "B")] {
        let h = read_hdr(cl, st).await.expect("读 header");
        assert_eq!(u32of(&h, 28), 0, "[ce-sm] {who} 侧 AR err 非 0");
    }
    let stream = std::env::var("OWL_OVL_CE_STREAM").unwrap_or_else(|_| "aux".to_string());
    eprintln!(
        "[ce-sm] AR×{ar_n}(1MiB) + CE×{ce_n}({stream}) ⇒ **{ms:.3} ms/回放**(K={k};判据 T_B≈T_A ⇒ 完全重叠)"
    );
    a.graph_destroy(ga).await.expect("destroy ga");
    b.graph_destroy(gb).await.expect("destroy gb");
}

// ============================================================================
// ★★ **图内跨流重叠探针**(2026-10-18;TP2-施工设计 §38)—— 真形状
// ============================================================================
//
// ## 为什么需要它
//
// push AR 的账(§38 分解):`494 = 写腿 444 + 本地 RMW 44.6 + 串联 5.3`。
// **裸双流探针(`tools/p2p-rmw-probe.cu`)证了"写对端 ‖ 本地 RMW"重叠 97%** ——
// 但生产的 AR 是**图内回放**,而 §33.2 量到的是"**图内**跨流零重叠"。
// §33.3 自己指出那条判据**无效**(两条 add 链都 L2 带宽饱和 ⇒ 共享瓶颈)。
//
// 本档用**真形状**重测:**PCIe 写核**(push 绑定的 stage)‖ **本地 RMW 核**(reduce,
// I2 用 HUGE 旁路)—— 两条链瓶颈不同(PCIe 写 vs 本地访存)。
//
// ## 一笔就够(不需要第二对块)
//
//   · **AUX 链** = push 绑定的 stage,**单边**、`own = peer = data = 对端块`:
//     写对端槽 + 用对端 header 自洽(代发 credit 让它自己的 credit 恒真;
//     rendezvous 等的是它自己刚发布的 produced)⇒ **单边可跑**;
//   · **COMPUTE 链** = reduce,`own = peer = data = 本端块`,本端 header
//     `produced = HUGE` ⇒ I2 恒真、`consumed` 从 0 递增(本链自用,无人读)。
//
//   两条链碰的 header **完全不重叠**(一条只碰对端块,一条只碰本端块)⇒ 无干扰。
//
// ## 判据
//
//   `S`(AUX 写链单独)· `R`(COMPUTE RMW 链单独)· `D`(两条同图)
//   ⇒ **`D ≈ max(S,R)` = 图内真并行**;**`D ≈ S+R` = 图内串行化**(双流无地基)。
//
// 运行:`OWL_PEER_TEST_DEV_A=2 OWL_PEER_TEST_DEV_B=3 \
//   cargo test -p owl-cuda --test ar_stress_gate push_graph_overlap_probe -- --nocapture`
// 旋钮:`OWL_OVLP_CHAIN`(缺省 128)· `OWL_OVLP_BYTES`(缺省 5.24MB)· `OWL_OVLP_K`(缺省 20)。
#[tokio::test]
#[ignore = "probe:图内跨流重叠(真形状;需 P2P 对)"]
async fn push_graph_overlap_probe() {
    let (Some(oa), Some(ob)) = (dev("OWL_PEER_TEST_DEV_A"), dev("OWL_PEER_TEST_DEV_B")) else {
        eprintln!("skip: 需 OWL_PEER_TEST_DEV_A / OWL_PEER_TEST_DEV_B");
        return;
    };
    let chain: u32 =
        std::env::var("OWL_OVLP_CHAIN").ok().and_then(|v| v.parse().ok()).unwrap_or(128);
    let bytes: usize =
        std::env::var("OWL_OVLP_BYTES").ok().and_then(|v| v.parse().ok()).unwrap_or(5_242_880);
    let k: u32 = std::env::var("OWL_OVLP_K").ok().and_then(|v| v.parse().ok()).unwrap_or(20);
    const HUGE: u64 = 1u64 << 62;

    // ⚠️ metrics store 是**进程级 OnceLock** ⇒ **必须在捕获之前 init**
    // (捕获期 `graph_end` 往里记图结构统计;事后 init 只能看到 0 —— 踩过)
    let _ = owl_shared::metrics::init_metrics(owl_shared::metrics::MetricsStore::new());
    let (ua, ub) = (uuid_of(oa), uuid_of(ob));
    let mut a = assemble(oa, "ovlp-a");
    let mut b = assemble(ob, "ovlp-b");
    let staging_bytes = HDR as u64 + SLOT * N_SLOTS as u64;
    let (staging_a, cred_a) =
        a.peer_alloc(staging_bytes as usize, "ovlp:st-a").await.expect("A PeerAlloc");
    let (staging_b, cred_b) =
        b.peer_alloc(staging_bytes as usize, "ovlp:st-b").await.expect("B PeerAlloc");
    // push 档要写对端 ⇒ 别名必须 RW
    let alias_b_on_a = a
        .peer_import_rw(cred_b, staging_bytes as usize, &ub, "ovlp:rw-b@a")
        .await
        .expect("A 导入 B(RW)");
    let _alias_a_on_b = b
        .peer_import_rw(cred_a, staging_bytes as usize, &ua, "ovlp:rw-a@b")
        .await
        .expect("B 导入 A(RW)");

    let dt = Dtype::F16;
    let n = bytes / 2;
    let payload_bytes = vec![0u8; bytes];
    let payload_a = a
        .htod(dt, &Shape::from(vec![n]), &payload_bytes, "ovlp:pl-a")
        .await
        .expect("A htod");
    let payload_b = b
        .htod(dt, &Shape::from(vec![n]), &payload_bytes, "ovlp:pl-b")
        .await
        .expect("B htod");

    // 一臂:捕一张图 + 回放 k 次 ⇒ ms/回放
    async fn arm(
        a: &mut GpuClient,
        payload_a: &Bytes,
        staging_a: &Bytes,
        alias_b: &Bytes,
        bytes: u64,
        chain: u32,
        k: u32,
        aux_write: bool,
        comp_rmw: bool,
    ) -> f64 {
        arm2(a, payload_a, staging_a, alias_b, bytes, chain, k, aux_write, comp_rmw, false).await
    }

    /// `aux_rmw`(2026-10-18 §41):AUX 链也放**本地 RMW**(用对端块做 header ⇒ 与
    /// COMPUTE 链的 header 不重叠)⇒ **两条同质、纯本地、不饱和的链**。
    /// 与 `[PCIe 写 ‖ 本地 RMW]` 对照 ⇒ 把"真串行"与"PCIe 争用"分开。
    #[allow(clippy::too_many_arguments)]
    async fn arm2(
        a: &mut GpuClient,
        payload_a: &Bytes,
        staging_a: &Bytes,
        alias_b: &Bytes,
        bytes: u64,
        chain: u32,
        k: u32,
        aux_write: bool,
        comp_rmw: bool,
        aux_rmw: bool,
    ) -> f64 {
        arm3(a, payload_a, staging_a, alias_b, bytes, chain, k, aux_write, comp_rmw, aux_rmw, false)
            .await
    }

    /// `aux_ce`(2026-10-18 §41):AUX 链改用 **CE 写对端**(`copy_block_at` ⇒
    /// `cudaMemcpyAsync` ⇒ 拷贝引擎,**不吃 SM**)⇒ 把"SM 写核"与"PCIe 写通道"分开。
    #[allow(clippy::too_many_arguments)]
    async fn arm3(
        a: &mut GpuClient,
        payload_a: &Bytes,
        staging_a: &Bytes,
        alias_b: &Bytes,
        bytes: u64,
        chain: u32,
        k: u32,
        aux_write: bool,
        comp_rmw: bool,
        aux_rmw: bool,
        aux_ce: bool,
    ) -> f64 {
        arm4(a, payload_a, staging_a, alias_b, bytes, chain, k, aux_write, comp_rmw, aux_rmw, aux_ce, 1)
            .await
    }

    /// ★ **生产 push 绑定的写链**(2026-10-18 §41 收尾):`own = 本端 staging`(header 留本端)、
    /// `peer = 对端别名`、`data = 对端别名`(写对端)——
    /// 与前面"单边自洽"(把 header 也搬到对端)**不同**:后者核入口的三次校验读 + credit 读
    /// + `atomicAdd(blocks_done)` **全变远端**,实测把写链从 13.1GB/s 拖到 **6.4GB/s**
    /// (§37 教训的现场重演)。对端 header 用 HUGE 顶满 ⇒ 单边可跑。
    #[allow(clippy::too_many_arguments)]
    async fn arm5(
        a: &mut GpuClient,
        payload_a: &Bytes,
        staging_a: &Bytes,
        alias_b: &Bytes,
        bytes: u64,
        chain: u32,
        k: u32,
        comp_rmw: bool,
        rmw_mult: u32,
    ) -> f64 {
        const HUGE: u64 = 1u64 << 62;
        write_hdr_at(a, staging_a, "ovlp5:hdr-a", 0, 0).await.expect("A hdr");
        // 对端块:`consumed = HUGE` ⇒ 写链的 credit **恒过**(单边可跑);
        // `produced = HUGE` ⇒ 写链的 rendezvous 恒过
        write_hdr_at(a, alias_b, "ovlp5:hdr-b", HUGE, HUGE).await.expect("B hdr(HUGE,HUGE)");
        // RMW 链用**独立 header**(64B 块)⇒ 与写链的 header 完全解耦
        // (⚠️ 之前把 RMW 链也绑到对端块 ⇒ `cons=HUGE` ⇒ `HUGE > HUGE` 假 ⇒ 自旋死锁)
        // 专用 header 块:`slot_bytes = bytes`(**核会校验 header 声明与实参一致**)⇒
        // 账长必须 ≥ HDR + slot_bytes ⇒ 分配 `HDR + bytes`
        let (hdr_rmw, _c) = a
            .peer_alloc(HDR + bytes as usize, "ovlp5:hdr-rmw")
            .await
            .expect("hdr_rmw");
        write_hdr_full(a, &hdr_rmw, "ovlp5:hdr-rmw-w", HUGE, 0, bytes as u32)
            .await
            .expect("hdr_rmw 值");
        a.graph_begin().await.expect("graph_begin");
        a.capture_route_to(owl_cuda::STREAM_COMPUTE_AUX as u8).await.expect("fork→AUX");
        for _ in 0..chain {
            a.launch(
                ArStageCall::new(
                    payload_a.clone(),
                    staging_a.clone(),
                    alias_b.clone(),
                    alias_b.clone(),
                    bytes,
                    SLOT,
                    0,
                )
                .to_launch(),
            )
            .await
            .expect("AUX 生产绑定写链");
        }
        a.capture_route_to(1u8).await.expect("切回 COMPUTE");
        if comp_rmw {
            for _ in 0..(chain * rmw_mult) {
                // header 用**对端块**(与写链的本端 header 分开)· slots 用本端 staging
                a.launch(
                    ArReduceCall::new(
                        payload_a.clone(),
                        hdr_rmw.clone(),
                        hdr_rmw.clone(),
                        staging_a.clone(),
                        bytes,
                        bytes,
                        0,
                    )
                    .to_launch(),
                )
                .await
                .expect("COMPUTE RMW 链");
            }
        }
        a.capture_join().await.expect("join");
        let gid = a.graph_end().await.expect("graph_end");
        for _ in 0..3 {
            a.graph_launch(gid).await.expect("warm");
        }
        a.sync().await.expect("warm sync");
        let t0 = Instant::now();
        for _ in 0..k {
            a.graph_launch(gid).await.expect("replay");
        }
        a.sync().await.expect("sync");
        let ms = t0.elapsed().as_secs_f64() * 1e3 / f64::from(k);
        a.graph_destroy(gid).await.expect("graph_destroy");
        ms
    }

    /// `rmw_mult`(2026-10-18 §41):COMPUTE 链的核数 = `chain × rmw_mult`
    /// ⇒ **直接量"串行度"**:`D − S` 随 `R` 线性增长 ⇒ 串行;不变 ⇒ RMW 被吸收。
    #[allow(clippy::too_many_arguments)]
    async fn arm4(
        a: &mut GpuClient,
        payload_a: &Bytes,
        staging_a: &Bytes,
        alias_b: &Bytes,
        bytes: u64,
        chain: u32,
        k: u32,
        aux_write: bool,
        comp_rmw: bool,
        aux_rmw: bool,
        aux_ce: bool,
        rmw_mult: u32,
    ) -> f64 {
        const HUGE: u64 = 1u64 << 62;
        // A 的 header:`produced = HUGE`(RMW 链的 I2 恒真)、`consumed` 归 0
        write_hdr_at(a, staging_a, "ovlp:hdr-a", HUGE, 0).await.expect("A hdr");
        // B 的 header:全 0(写链自洽:代发 credit + 自等自)
        // (B 侧由调用方在臂外设好 —— 这里只碰 A)
        a.graph_begin().await.expect("graph_begin");
        if aux_write || aux_rmw || aux_ce {
            a.capture_route_to(owl_cuda::STREAM_COMPUTE_AUX as u8).await.expect("fork→AUX");
            for _ in 0..chain {
                if aux_ce {
                    // CE 写对端:payload_a → 对端块的 slot 区(绕过 SM)
                    a.copy_block_at(payload_a, 0, alias_b, HDR, bytes as usize)
                        .await
                        .expect("AUX CE 写");
                }
                if aux_write {
                    // push 绑定 + 单边:own = peer = data = 对端块
                    a.launch(
                        ArStageCall::new(
                            payload_a.clone(),
                            alias_b.clone(),
                            alias_b.clone(),
                            alias_b.clone(),
                            bytes,
                            SLOT,
                            0,
                        )
                        .to_launch(),
                    )
                    .await
                    .expect("AUX 写链");
                }
                if aux_rmw {
                    // 本地 RMW,header 用**对端块**(与 COMPUTE 链的本端块不重叠)
                    a.launch(
                        ArReduceCall::new(
                            payload_a.clone(),
                            alias_b.clone(),
                            alias_b.clone(),
                            alias_b.clone(),
                            bytes,
                            SLOT,
                            0,
                        )
                        .to_launch(),
                    )
                    .await
                    .expect("AUX RMW 链");
                }
            }
            a.capture_route_to(1u8).await.expect("切回 COMPUTE");
        }
        if comp_rmw {
            for _ in 0..(chain * rmw_mult) {
                // 本地 RMW:own = peer = data = 本端块(produced = HUGE ⇒ 不等待)
                a.launch(
                    ArReduceCall::new(
                        payload_a.clone(),
                        staging_a.clone(),
                        staging_a.clone(),
                        staging_a.clone(),
                        bytes,
                        SLOT,
                        0,
                    )
                    .to_launch(),
                )
                .await
                .expect("COMPUTE RMW 链");
            }
        }
        // ★ **末尾 join**(在 origin 上调用 = 给 origin 的**尾部**建边):
        // `StreamWaitEvent` 只影响**之后**发射的工作 ⇒ 上面已发射的 COMPUTE 链
        // **不会**被挡在 AUX 链之后 ⇒ 两条链中段独立、末尾合法 join。
        // ⚠️ 不 join 会被驱动以 UNJOINED 整窗拒掉(§39 实测:支流只要有工作就必须 join)。
        // ⚠️ 只在**本臂真的 fork 过**时 join(没有支流的窗调 join = 响亮拒;
        // 这正是纪律㉔要的"静默 no-op 不许存在")
        if aux_write || aux_rmw || aux_ce {
            a.capture_join().await.expect("join(origin 尾部等 AUX 链尾)");
        }
        let gid = a.graph_end().await.expect("graph_end");
        for _ in 0..3 {
            a.graph_launch(gid).await.expect("warm");
        }
        a.sync().await.expect("warm sync");
        let t0 = Instant::now();
        for _ in 0..k {
            a.graph_launch(gid).await.expect("replay");
        }
        a.sync().await.expect("sync");
        let ms = t0.elapsed().as_secs_f64() * 1e3 / f64::from(k);
        a.graph_destroy(gid).await.expect("graph_destroy");
        ms
    }

    // B 侧 header 归零(写链要读它;一次即可)
    write_hdr(&mut b, &staging_b, "ovlp:hdr-b").await.expect("B hdr");
    let _ = &payload_b;

    // ★ **采集纪律(用户口径 2026-10-18)**:**攒内存,最后集中 dump** ——
    // 三臂的读数先落结构体(零打印),图结构统计由 `graph_end` 记进 metrics store,
    // 全部跑完后再一次性出表(表 + CSV)。**热路径零打印。**
    let t_arm = |tag: &str, v: f64| (tag.to_string(), v);
    let mut rows: Vec<(String, f64)> = Vec::new();
    rows.push(t_arm("S 写链(AUX)", arm(&mut a, &payload_a, &staging_a, &alias_b_on_a, bytes as u64, chain, k, true, false).await));
    rows.push(t_arm("R RMW 链(COMPUTE)", arm(&mut a, &payload_a, &staging_a, &alias_b_on_a, bytes as u64, chain, k, false, true).await));
    rows.push(t_arm("D 双流(写‖RMW)", arm(&mut a, &payload_a, &staging_a, &alias_b_on_a, bytes as u64, chain, k, true, true).await));
    // ★ **对照臂**(§41):两条**同质本地 RMW** 链(AUX 用对端块做 header)
    //   ⇒ 与 `D` 对照即可把"真串行"与"PCIe 争用"分开。
    // E 臂的 AUX 链用**对端块**做 header ⇒ 它的 I2 等的是 `B.produced` ⇒ 必须先顶满
    // (否则 `0 <= 0` 永远成立 ⇒ 自旋死锁 —— 首跑就是这么挂的)
    write_hdr_at(&mut a, &alias_b_on_a, "ovlp:hdr-b-huge", HUGE, 0)
        .await
        .expect("B header(HUGE)");
    rows.push(t_arm("E 双流(RMW‖RMW)", arm2(&mut a, &payload_a, &staging_a, &alias_b_on_a, bytes as u64, chain, k, false, true, true).await));
    rows.push(t_arm("F 双流(CE写‖RMW)", arm3(&mut a, &payload_a, &staging_a, &alias_b_on_a, bytes as u64, chain, k, false, true, false, true).await));
    rows.push(t_arm("G CE 写链(AUX 单链)", arm3(&mut a, &payload_a, &staging_a, &alias_b_on_a, bytes as u64, chain, k, false, false, false, true).await));
    // ★ **串行度扫描**(§41):把 COMPUTE 的 RMW 链拉长,看 `D − S` 是否随 `R` 线性长。
    let rmw_mult: u32 = std::env::var("OWL_OVLP_RMW_MULT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    let r8 = arm4(&mut a, &payload_a, &staging_a, &alias_b_on_a, bytes as u64, chain, k, false, true, false, false, rmw_mult).await;
    let d8 = arm4(&mut a, &payload_a, &staging_a, &alias_b_on_a, bytes as u64, chain, k, true, true, false, false, rmw_mult).await;
    rows.push(t_arm("R8 RMW×mult", r8));
    rows.push(t_arm("D8 写‖RMW×mult", d8));
    // ★ **生产绑定**(header 留本端)的写链:单跑 / 与 RMW 同图
    let h = arm5(&mut a, &payload_a, &staging_a, &alias_b_on_a, bytes as u64, chain, k, false, 1).await;
    let i_arm = arm5(&mut a, &payload_a, &staging_a, &alias_b_on_a, bytes as u64, chain, k, true, rmw_mult).await;
    rows.push(t_arm("H 生产绑定写链(单)", h));
    rows.push(t_arm("I 生产绑定写‖RMW", i_arm));
    let get = |t: &str| rows.iter().find(|(k, _)| k.starts_with(t)).map(|(_, v)| *v).unwrap_or(0.0);
    let (s, r, d) = (get("S "), get("R "), get("D "));

    // ── 集中出表(跑完之后一次;含图结构统计)──────────────────────
    let cnt = |t: &str| -> u64 {
        owl_shared::metrics::query_metrics(&owl_shared::metrics::MetricsFilter::new().tag(t))
            .iter()
            .map(|x| x.counter)
            .sum()
    };
    eprintln!("\n[ovlp] ══ 图内跨流重叠(真形状;chain={chain} bytes={bytes})══");
    for (k, v) in &rows {
        eprintln!("[ovlp]   {k:<22} {v:8.3} ms");
    }
    eprintln!("[ovlp]   串行预期 {:.3} ms ⇒ **双流/串行 = {:.3}**", s + r, d / (s + r));
    let e = get("E ");
    eprintln!(
        "[ovlp] 判据:≈0.5 = 图内真并行;≈1.0 = 图内串行化(双流/微批都没地基)"
    );
    let f = get("F ");
    let g = get("G ");
    let (r8, d8) = (get("R8"), get("D8"));
    let (h, i_arm) = (get("H "), get("I "));
    eprintln!("[ovlp]   E 双流(RMW‖RMW)          {e:8.3} ms   (≈R ⇒ 并行;≈2R ⇒ 串行)");
    eprintln!("[ovlp]   F 双流(CE写‖RMW)         {f:8.3} ms   (≈G ⇒ 并行;≈G+R ⇒ 串行)");
    eprintln!("[ovlp]   G CE 写链(AUX 单链)       {g:8.3} ms");
    eprintln!(
        "[ovlp] ★ **生产绑定写链**(header 留本端):单跑 {h:.3} ms({:.2} GB/s) · 写‖RMW' {i_arm:.3} ms \
         ⇒ **吸收率 {:.1}%**",
        (bytes as f64 * chain as f64) / (h * 1e-3) / 1e9,
        100.0 * (1.0 - (i_arm - h) / r8.max(1e-9))
    );
    eprintln!(
        "[ovlp] ★ **串行度扫描**(mult={rmw_mult}):R'={r8:.3} D'={d8:.3} ⇒ **D'−S = {:.3} ms**, \
         R' = {r8:.3} ms ⇒ 吸收率 {:.1}%(0% = 完全串行, 100% = 完全吸收)",
        d8 - s,
        100.0 * (1.0 - (d8 - s) / r8.max(1e-9))
    );
    eprintln!(
        "[ovlp] ★ **分开\"SM 写核\"与\"PCIe 写通道\"**:D(SM 写‖RMW)/串行 = {:.3} · \
         F(CE 写‖RMW)/串行 = {:.3} —— F ≈1 而 D ≈1 ⇒ 与 SM 无关(是写通道);两者都 ≈1 ⇒ 写就是串行",
        d / (s + r),
        f / (g + r)
    );
    eprintln!(
        "[ovlp] 图结构(捕获期攒的簿记):nodes={} aux_nodes={} forks={} joins={} copies={}",
        cnt("graph.nodes"),
        cnt("graph.aux_nodes"),
        cnt("graph.forks"),
        cnt("graph.joins"),
        cnt("graph.copies")
    );
    eprintln!(
        "[ovlp] ★ 关键判据:`aux_nodes > 0` ⇒ 图里**真有** fork 出去的分支(否则 1.000 是假象)"
    );
}

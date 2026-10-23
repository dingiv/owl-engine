//! **AR 通道微基准(TP 通道工作面;隔离测量)**。
//!
//! 设计出处:`docs/arch/TP2-施工设计.md` §22.6(TP2 prefill 0.24× 立案)与
//! `docs/arch/TP2-通信-预研.md` §3(通信量账)。
//!
//! # 为什么单独一个基准
//!
//! §22.6 的账:`prefill ≈ 12.9ms/AR`(载荷 5.24MB ⇒ 有效 **~400MB/s**,对
//! P2P 13.2GB/s 差 33×),而 decode ≈ 37µs/AR —— 两侧差 350×。这条账是从
//! **端到端 tok/s 反推**出来的,混杂了 GEMM/胶水核;要提速必须先有**把通道
//! 单独拎出来**的读数。本基准就是那把尺子(不是门 —— 判据门另有
//! `comm_ar_parity` / `comm_ar_smoke` / `tp_transport_bits_gate`)。
//!
//! # 三段分解(全部走**生产同一批核**,零新增 kernel)
//!
//! **2026-10-17 归口**:本文件从 `crates/backends/cuda/tests/comm_ar_bench.rs`
//! 迁入**通信包**(用户口径:通信的测试住在通信包;且**不需加载模型**即可迭代)。
//!
//! | 探针 | 做法 | 量的是什么 |
//! |---|---|---|
//! | **S**(stage-only) | 把**对端** header 的 `consumed` 置 `2^62` ⇒ credit 永不阻塞 ⇒ 单边循环 stage | 本端 payload → 本端 staging 的**本地**拷贝(核的循环上界) |
//! | **R**(peer-read) | 把**对端** header 的 `produced` 置 `2^62` ⇒ I2 永不阻塞 ⇒ 单边循环 reduce | **对端读 + 本地 RMW** —— 大载荷慢路径的主角 |
//! | **AR**(full) | 两 rank 各入队 K 次 `stage_i → reduce_i`,tokio::join 后同 sync | **生产路径**(含 credit 耦合 + 两次 fence + 发射) |
//!
//! 三个数一摆,`AR − (S + R)` = 协议/耦合开销,`R` 对"同字节数换 dtype"
//! 的形状敏感度 = **是不是迭代延迟暴露**(元素数减半 ⇒ f32 若明显快于 f16,
//! 则是"每次访存一趟往返"而不是带宽受限)。
//!
//! 另有三个 **2026-10-17 追加**的探针(用户提醒"三条流你都可以用,两卡四条流对班"):
//!
//! | 探针 | 做法 | 量的是什么 |
//! |---|---|---|
//! | **RR** | 两 rank 同时 `reduce`(仍是 SM 读) | SM 路径的双向并发(⚠️ **不是**天花板) |
//! | **C1** | `memcpy_dtod_async(对端 staging → 本端 payload)` | **拷贝引擎(CE)** 单向对端读 |
//! | **C2** | 两 rank 同时互拷(各走自己的流) | **CE 双向并发** = "两卡各一条搬运流" |
//!
//! **结论(实测,§26.2b)**:`C2` = **10.3–12.9 GB/s**(与 NCCL 11.1 齐平),
//! 比 SM 的 `RR`(7.2)快 **1.4–1.5×**,而 CE **不吃 SM、可挂另一条流**。
//! 5.24MB 的完整 AR = 868µs,而纯搬运(C2)=461µs ⇒ **多出的 407µs 全是协议结构**
//! (staging 拷贝 + 两核串联 + 两次握手)。⇒ **`ar_v2` 的方向 = 换通道**
//! (搬运交 CE + 另一条流),不是"把 SM 核写得更快"。
//!
//! # 口径纪律
//!
//! - slot = 32MiB × 2(**与生产同参**)⇒ 载荷 ≤32MiB 时相邻两次调用落在
//!   **相距 32MiB 的不同 slot** ⇒ 不吃 L2 余温(3080 L2 = 6MB,不可缓存满载);
//! - 载荷尺寸取 27B 真形状(`hidden 5120` × f16):10KB(1 行)/
//!   1.31MB(128 行)/ 5.24MB(512 行)/ 10.49MB(1024 行)+ 边界 32MiB/+1
//!   + 80MiB(3 块,管道面 INFO);
//! - **host/设备分离**:`enq` = 入队 K 次(每发射 = 一次**同步往返**
//!   client↔server 线程,实测 **~35-37µs/发射**;`ack` 在**入队后即回**,
//!   不等核完成 —— 见 `server::handle_launch_inner`);`dev` = `total − enq`
//!   = 设备队列**尾排**。**判据只取 `dev`**:`dev ≈ 0` 表示设备比 host 快
//!   (该档只给出"≤ 发射往返"的上界),`dev > 0` 才是设备真时。
//!
//! ## ★ 两个已定谳的坑(本基准自己踩的;后人别重踩)
//!
//! 1. **小载荷奇相不是噪声**:2026-10-17 首轮扫描出"20KB 慢于 128KB"的
//!    怪相。先疑外部抢卡(确有一次:另一 agent 的 `engine_e2e` 占着 CUDA
//!    ordinal 3,`clocks.sm=1710MHz/P2`)⇒ 改到**完全空闲**的 3090 Ti 对
//!    (0/1)后**怪相照旧** ⇒ 不是干扰。真因 = 当时核里"展开主循环 + 逐条
//!    余数循环"的两段式:`n_vec < 8×blockDim` 时余数段占主体而落回逐条
//!    速度。**K 扫(50/200/800 同值)证明是 per-launch 而非一次性**。
//!    ⇒ 现在核 = **单循环 + 逐通道谓词**(所有尺寸同一条路)。
//! 2. **`dev` 是区间量,不是核时**:`dev<0` 会被 clamp 到 0 ⇒ 该档无分辨力。
//!    要真核时用 `OWL_GPU_PROF=1`(逐发射后同步 COMPUTE 流;见 server 注)
//!    或先看 `dev` 是否显著大于 0 再下结论。
//!
//! ## 实测存档(2026-10-17;3080 对 2/3;f16;K=200)
//!
//! | 载荷 | AR µs(修前) | AR µs(修后) | MB/s(修后) |
//! |---|---:|---:|---:|
//! | 10KB(decode 1 行)| 71 | 71(host 地板)| 137 |
//! | 5.24MB(chunk 512)| **14786** | **822** | 6084 |
//! | 10.49MB(chunk 1024)| 30106 | **1584** | 6315 |
//! | 32MiB(1 块上界)| 106297 | 7439 | 4300 |
//!
//! 修前 `R`(纯对端读)在**同字节不同 dtype** 上 f32 恰好比 f16 快 2×
//! ⇒ 病根 = **在飞访存数 = 1/线程**(每元素一趟 PCIe 往返 ~1.41µs),不是带宽;
//! 修后 f32/f16 同值 ⇒ 已是带宽受限(详账与四道证据链见
//! `docs/arch/TP2-施工设计.md` §25)。
//!
//! # 运行
//!
//! ```bash
//! OWL_PEER_TEST_DEV_A=2 OWL_PEER_TEST_DEV_B=3 \
//!   cargo test -p owl-cuda --test comm_ar_bench -- --ignored --nocapture
//! ```
//! (`2`/`3` = 3080 对;3090 Ti 对 = `0`/`1`。未设 env ⇒ skip。)
//!
//! **阶梯档不用重编译**:`OWL_ARB_SIZES=4096,10240,...`(逗号分隔字节)+ 
//! `OWL_ARB_DTYPE=f16|f32|bf16` + `OWL_ARB_K=200`。
//! **AR-2 臂**:`OWL_ARB_AR2=1`(≤256KiB;与 `OWL_ARB_GRAPH=1` 同跑,打印
//! `[arb-ar2]` 的图内 µs/次)。
//! `OWL_ARB_PUSH=1` 打开**推送臂探针**(远程写;当前**预期失败** ——
//! `peer_import` 的别名是只读授予,见 §25.7)。

use half::{bf16, f16};
use owl_cuda::comm::ar::{plan_ar, AllReduce, ArPhase, StagingLayout};
use owl_cuda::{
    Bytes, Command, DeviceClient as _, DeviceSelector, Dtype, GpuClient, GpuServer, ModelError,
    Shape,
};
use owl_kernels::client::comm::{Ar1StageCall, ArFusedCall, ArReduceCall, ArStageCall};
use std::time::Instant;

const HDR: usize = 64;
const SLOT: u64 = 32 << 20; // 设计值(施工设计 §6;与生产同参)
const N_SLOTS: u32 = 2;
const MAGIC: u32 = 0x4F57_4C41; // "OWLA"
const HUGE: u64 = 1u64 << 62; // "永真 credit/I2"哨兵(单边探针用)
const SLOT_1S: usize = 256 * 1024; // 1-stage 臂的对称 slot(≤256KiB 档)

// ============================================================================
// 组装 / 工具(与 comm_ar_parity.rs 同构)
// ============================================================================

fn assemble(ordinal: usize, tag: &str) -> GpuClient {
    let (tx, rx) = std::sync::mpsc::channel::<Command>();
    let server = GpuServer::new(rx, DeviceSelector::Ordinal(ordinal), None);
    let client = GpuClient::new(tx);
    std::thread::Builder::new()
        .name(format!("owl-arb-{tag}"))
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

fn hdr_bytes_slots(produced: u64, consumed: u64, n_slots: u32) -> [u8; HDR] {
    let mut b = [0u8; HDR];
    b[0..8].copy_from_slice(&produced.to_le_bytes());
    b[8..16].copy_from_slice(&consumed.to_le_bytes());
    b[16..20].copy_from_slice(&(SLOT as u32).to_le_bytes());
    b[20..24].copy_from_slice(&n_slots.to_le_bytes());
    b[24..28].copy_from_slice(&MAGIC.to_le_bytes());
    b
}

fn hdr_u32(h: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(h[off..off + 4].try_into().expect("4B"))
}

fn width(dt: Dtype) -> usize {
    match dt {
        Dtype::F32 => 4,
        Dtype::F16 | Dtype::BF16 => 2,
        other => panic!("AR 基准只测三 dtype,拿到 {other:?}"),
    }
}

/// 小值填充(`|v| ≤ 0.01`):reduce 会**原地累加** K 次 ⇒ 必须留足余量,
/// 不许撞 f16 上限(撞了就是 inf/NaN 的算术,虽然不改速度但坏了口径)。
fn memset_pattern(dt: Dtype, n_elems: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n_elems * width(dt));
    let mut s = 0x243F_6A88u64;
    for i in 0..n_elems {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let v = (((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0) * 0.01;
        let _ = i;
        match dt {
            Dtype::F32 => out.extend_from_slice(&v.to_le_bytes()),
            Dtype::F16 => out.extend_from_slice(&f16::from_f32(v).to_bits().to_le_bytes()),
            Dtype::BF16 => out.extend_from_slice(&bf16::from_f32(v).to_bits().to_le_bytes()),
            other => panic!("{other:?}"),
        }
    }
    out
}

/// 写 header(整 64B;走 pinned 上传 + sync ⇒ 核读到的一定是新值)
async fn set_hdr(
    client: &mut GpuClient,
    staging: &Bytes,
    produced: u64,
    consumed: u64,
) -> Result<(), ModelError> {
    set_hdr_slots(client, staging, produced, consumed, N_SLOTS).await
}

/// 同上,但可改 `n_slots` 声明(A3 探针要 `n_slots = 1`;块尺寸不变,
/// 只是**声明**变成单格 ⇒ stage 恒写 slot 0)。
async fn set_hdr_slots(
    client: &mut GpuClient,
    staging: &Bytes,
    produced: u64,
    consumed: u64,
    n_slots: u32,
) -> Result<(), ModelError> {
    let mut pin = client.alloc_pinned(HDR, "arb:hdr").await?;
    pin.slice_bytes_mut()[..HDR].copy_from_slice(&hdr_bytes_slots(produced, consumed, n_slots));
    client.upload_pinned(pin, staging, 0, HDR).await?;
    client.sync().await
}

/// 读 err 字段(每段探针收尾自证:D4"全有或全无"不许静默)
async fn read_err(client: &mut GpuClient, staging: &Bytes) -> u32 {
    let mut buf = vec![0u8; HDR];
    client.dtoh(staging, &mut buf).await.expect("读 header");
    hdr_u32(&buf, 28)
}

/// 一次 allreduce(生产同序:`stage_i → reduce_i`;两侧同 plan ⇒ I5)
async fn emit_ar(
    client: &mut GpuClient,
    payload: &Bytes,
    own: &Bytes,
    peer: &Bytes,
    bytes: u64,
    layout: &StagingLayout,
) -> Result<(), ModelError> {
    emit_ar_bind(client, payload, own, peer, bytes, layout, false).await
}

/// **push 档发射**(2026-10-18;§36.7):两核 `own`/`peer` **实参对调** ——
/// stage 写**对端**槽、reduce 读**本端**槽 ⇒ 跨 PCIe 只剩写。
/// ⚠️ 要求别名块是 **RW 导入**的(`OWL_ARB_AR_PUSH=1` 时走 `peer_import_rw`)。
async fn emit_ar_push(
    client: &mut GpuClient,
    payload: &Bytes,
    own: &Bytes,
    peer: &Bytes,
    bytes: u64,
    layout: &StagingLayout,
) -> Result<(), ModelError> {
    emit_ar_bind(client, payload, own, peer, bytes, layout, true).await
}

async fn emit_ar_bind(
    client: &mut GpuClient,
    payload: &Bytes,
    own: &Bytes,
    peer: &Bytes,
    bytes: u64,
    layout: &StagingLayout,
    push: bool,
) -> Result<(), ModelError> {
    // header 两档都不变;只换**数据块**(与生产 `eval.rs` 同一律)
    let (stage_data, reduce_data) =
        if push { (peer.clone(), own.clone()) } else { (own.clone(), peer.clone()) };
    for (phase, c) in AllReduce::new(plan_ar(bytes, layout)).steps() {
        let msg = match phase {
            ArPhase::Stage => ArStageCall::new(
                payload.clone(),
                own.clone(),
                peer.clone(),
                stage_data.clone(),
                c.bytes,
                layout.slot_bytes,
                c.offset,
            )
            .to_launch(),
            ArPhase::Reduce => ArReduceCall::new(
                payload.clone(),
                own.clone(),
                peer.clone(),
                reduce_data.clone(),
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

/// **AR-2 单发**(2026-10-23):barrier-only 小包;固定 v1 数据腿
/// (data_w=own, data_r=peer),复用 fused 同 8 槽线格式,只换发射名。
async fn emit_ar2(
    client: &mut GpuClient,
    payload: &Bytes,
    own: &Bytes,
    peer: &Bytes,
    bytes: u64,
    layout: &StagingLayout,
) -> Result<(), ModelError> {
    let msg = ArFusedCall::new(
        payload.clone(),
        own.clone(),
        peer.clone(),
        own.clone(),
        peer.clone(),
        bytes,
        layout.slot_bytes,
        0,
    )
    .with_barrier(true)
    .to_launch();
    client.launch(msg).await?;
    Ok(())
}

/// **1-stage 单发**(对称 slot):`out = src + peer_data`,header 走 produced/consumed。
async fn emit_1stage(
    client: &mut GpuClient,
    payload: &Bytes,
    src: &Bytes,
    own: &Bytes,
    peer: &Bytes,
    peer_data: &Bytes,
    bytes: u64,
) -> Result<(), ModelError> {
    // slot 由 peer_alloc 给出时是 U8;1-stage 的 src/peer_data 声明为 F16
    // ⇒ 只重声明 dtype/rank,id/指针不变。
    let msg = Ar1StageCall::new(
        payload.clone(),
        src.clone().view(Dtype::F16, 1),
        own.clone(),
        peer.clone(),
        peer_data.clone(),
        bytes,
    )
    .to_launch();
    client.launch(msg).await?;
    Ok(())
}

/// 单边探针的发射面:只发一个相位(另一相位由哨兵 header 旁路)
async fn emit_one(
    client: &mut GpuClient,
    payload: &Bytes,
    own: &Bytes,
    peer: &Bytes,
    bytes: u64,
    layout: &StagingLayout,
    reduce: bool,
) -> Result<(), ModelError> {
    emit_one_push(client, payload, own, peer, bytes, layout, reduce, false).await
}

/// 单边探针(push 绑定):`data` 按 push 选 —— stage 写对端槽 / reduce 读本端槽。
#[allow(clippy::too_many_arguments)]
async fn emit_one_push(
    client: &mut GpuClient,
    payload: &Bytes,
    own: &Bytes,
    peer: &Bytes,
    bytes: u64,
    layout: &StagingLayout,
    reduce: bool,
    push: bool,
) -> Result<(), ModelError> {
    let (s_data, r_data) =
        if push { (peer.clone(), own.clone()) } else { (own.clone(), peer.clone()) };
    let msg = if reduce {
        ArReduceCall::new(
            payload.clone(),
            own.clone(),
            peer.clone(),
            r_data,
            bytes,
            layout.slot_bytes,
            0,
        )
        .to_launch()
    } else {
        ArStageCall::new(
            payload.clone(),
            own.clone(),
            peer.clone(),
            s_data,
            bytes,
            layout.slot_bytes,
            0,
        )
        .to_launch()
    };
    client.launch(msg).await?;
    Ok(())
}

/// µs/次(墙钟包住"入队 K 次 + sync" —— 设备侧无 event 面)
fn per_call_us(total_s: f64, k: u32) -> f64 {
    total_s * 1e6 / f64::from(k)
}

// ============================================================================
// ★ A3:CE 三段 = **生产同形**(2026-10-17 接线的形状)
// ============================================================================
///
/// `stage(own=本端 staging) → CE 拷(对端 staging slot 0 → 本端 inbox slot 0)
/// → add(reduce 核绑 `own = peer = inbox`)`。
///
/// 协议侧与 v1 **完全同构**(credit / rendezvous 都在 stage 核里;inbox 的
/// `produced` 顶满 ⇒ add 核的 I2 恒真 —— 数据就绪由**流序**保证)。
///
/// `A2`(CE 拷 + 本地加)**旁路了协议**;`A3` 才是"CE 刀值不值"的直接账:
/// `n_slots=1` 的严格一深环(credit 无余量、两 rank 更贴身)的代价只有它量得到。
async fn emit_ce(
    client: &mut GpuClient,
    payload: &Bytes,
    own: &Bytes,
    peer: &Bytes,
    inbox: &Bytes,
    bytes: u64,
    layout: &StagingLayout,
) -> Result<(), ModelError> {
    client
        .launch(
            ArStageCall::new(
                payload.clone(),
                own.clone(),
                peer.clone(),
                own.clone(), // data:写本端槽
                bytes,
                layout.slot_bytes,
                0,
            )
            .to_launch(),
        )
        .await?;
    client.copy_block_at(peer, HDR, inbox, HDR, bytes as usize).await?;
    client
        .launch(
            ArReduceCall::new(
                payload.clone(),
                inbox.clone(),
                inbox.clone(),
                inbox.clone(), // data:读本端 inbox
                bytes,
                layout.slot_bytes,
                0,
            )
            .to_launch(),
        )
        .await?;
    Ok(())
}

/// 有效单向带宽(MB/s;按**跨 PCIe 搬走的载荷字节**算 —— 只有 reduce 读对端)
fn mbps(bytes: u64, us: f64) -> f64 {
    if us <= 0.0 {
        return f64::INFINITY;
    }
    bytes as f64 / 1024.0 / 1024.0 / (us / 1e6)
}

// ============================================================================
// 基准
// ============================================================================

/// 1-stage 臂的对称 slot/别名(只分配一次;每个 case 复用)
struct OneStageSyms {
    slot_a: Bytes,
    slot_b: Bytes,
    alias_b_on_a: Bytes,
    alias_a_on_b: Bytes,
}

/// 单档探针(主表与 `OWL_ARB_SIZES` 阶梯共用**同一条量法** —— 不留第二套尺子)
#[allow(clippy::too_many_arguments)]
async fn run_case(
    a: &mut GpuClient,
    b: &mut GpuClient,
    staging_a: &Bytes,
    staging_b: &Bytes,
    alias_b_on_a: &Bytes,
    alias_a_on_b: &Bytes,
    inbox_a: &Bytes,
    inbox_b: &Bytes,
    one_stage: &OneStageSyms,
    layout: &StagingLayout,
    bytes: usize,
    dt: Dtype,
    k: u32,
) {
    let w = width(dt);
    let n_elems = bytes / w;
    let payload_bytes = memset_pattern(dt, n_elems);

    let payload_a = a
        
        .htod(dt, &Shape::from(vec![n_elems]), &payload_bytes, "arb:payload-a")
        .await
        .expect("A payload htod");
    let payload_b = b
        
        .htod(dt, &Shape::from(vec![n_elems]), &payload_bytes, "arb:payload-b")
        .await
        .expect("B payload htod");

    // ── 常态 header(两侧归零;相对游标 ⇒ 任何同值起点都合法)──
    set_hdr(a, &staging_a, 0, 0).await.expect("A hdr");
    set_hdr(b, &staging_b, 0, 0).await.expect("B hdr");
    // ★ **push AR 档**(2026-10-18;§36.7):两核 own/peer 对调(需 RW 别名)
    let push_ar = std::env::var("OWL_ARB_AR_PUSH").is_ok();
    // 先跑 2 次完整 AR 把两个 slot 都填成真 f16(NaN 位型会坏口径)
    for _ in 0..2 {
        let (ra, rb) = if push_ar {
            tokio::join!(
                emit_ar_push(a, &payload_a, &staging_a, &alias_b_on_a, bytes as u64, &layout),
                emit_ar_push(b, &payload_b, &staging_b, &alias_a_on_b, bytes as u64, &layout),
            )
        } else {
            tokio::join!(
                emit_ar(a, &payload_a, &staging_a, &alias_b_on_a, bytes as u64, &layout),
                emit_ar(b, &payload_b, &staging_b, &alias_a_on_b, bytes as u64, &layout),
            )
        };
        ra.expect("A 预热");
        rb.expect("B 预热");
        let (sa, sb) = tokio::join!(a.sync(), b.sync());
        sa.expect("A sync");
        sb.expect("B sync");
    }

    // ── AR(生产路径)────────────────────────────────────────
    let t0 = Instant::now();
    let ar_enq: f64;
    {
        let fa = async {
            for _ in 0..k {
                if push_ar { emit_ar_push(a, &payload_a, &staging_a, &alias_b_on_a, bytes as u64, &layout).await } else { emit_ar(a, &payload_a, &staging_a, &alias_b_on_a, bytes as u64, &layout).await }.expect("A 发射");
            }
        };
        let fb = async {
            for _ in 0..k {
                if push_ar { emit_ar_push(b, &payload_b, &staging_b, &alias_a_on_b, bytes as u64, &layout).await } else { emit_ar(b, &payload_b, &staging_b, &alias_a_on_b, bytes as u64, &layout).await }.expect("B 发射");
            }
        };
        tokio::join!(fa, fb);
        ar_enq = t0.elapsed().as_secs_f64();
        let (sa, sb) = tokio::join!(a.sync(), b.sync());
        sa.expect("A sync");
        sb.expect("B sync");
    }
    let ar_tot = t0.elapsed().as_secs_f64();
    let ar_us = per_call_us(ar_tot, k);
    let ar_enq_us = per_call_us(ar_enq, k);
    let ar_dev_us = per_call_us((ar_tot - ar_enq).max(0.0), k);
    assert_eq!(read_err(a, &staging_a).await, 0, "A 侧 err 非 0({bytes}B {dt:?})");
    assert_eq!(read_err(b, &staging_b).await, 0, "B 侧 err 非 0({bytes}B {dt:?})");

    // ── ★ **push 分解**(2026-10-18;§38):把 push AR 的账拆成两笔 ──────────
    //   ① `pS` = **纯写腿**(双边 stage,push 绑定:data = 对端块)—— 含 stage 协议;
    //   ② `pR` = **纯本地 RMW**(单边 reduce,push 绑定:data = 本端块;对端 produced
    //      置 HUGE ⇒ I2 旁路)—— 含 reduce 协议。
    //   判据:`pS + pR` 对 `AR(push)` 的差 = 两核串联 + 两次握手 + 一次发射的税。
    if push_ar && (bytes as u64) <= layout.slot_bytes {
        let (k_s, k_r) = ((k / 2).max(1), (k / 2).max(1));
        set_hdr(a, &staging_a, 0, 0).await.expect("A hdr(pS)");
        set_hdr(b, &staging_b, 0, 0).await.expect("B hdr(pS)");
        let t0 = Instant::now();
        {
            let fa = async {
                for _ in 0..k_s {
                    emit_one_push(a, &payload_a, &staging_a, &alias_b_on_a, bytes as u64,
                                  &layout, false, true).await.expect("pS A");
                }
            };
            let fb = async {
                for _ in 0..k_s {
                    emit_one_push(b, &payload_b, &staging_b, &alias_a_on_b, bytes as u64,
                                  &layout, false, true).await.expect("pS B");
                }
            };
            tokio::join!(fa, fb);
        }
        let enq = t0.elapsed().as_secs_f64();
        let (sa, sb) = tokio::join!(a.sync(), b.sync());
        sa.expect("A sync"); sb.expect("B sync");
        let ps_dev = per_call_us((t0.elapsed().as_secs_f64() - enq).max(0.0), k_s);
        assert_eq!(read_err(a, &staging_a).await, 0, "pS A err");

        // ② 纯本地 RMW(data = 本端块;对端 produced HUGE ⇒ 不等待)
        set_hdr(a, &staging_a, 0, 0).await.expect("A hdr(pR)");
        set_hdr(b, &staging_b, HUGE, 0).await.expect("B hdr(pR)");
        let t0 = Instant::now();
        for _ in 0..k_r {
            emit_one_push(a, &payload_a, &staging_a, &alias_b_on_a, bytes as u64, &layout,
                          true, true).await.expect("pR A");
        }
        let enq = t0.elapsed().as_secs_f64();
        a.sync().await.expect("A sync");
        let pr_dev = per_call_us((t0.elapsed().as_secs_f64() - enq).max(0.0), k_r);
        assert_eq!(read_err(a, &staging_a).await, 0, "pR A err");
        eprintln!(
            "[arb-push] {bytes:>9} B  **分解**:pS(纯写腿)={ps_dev:8.2} · pR(纯本地 RMW)={pr_dev:7.2}              · 和={:8.2} · AR(push)={ar_us:8.2} ⇒ 串联/握手税={:6.2}",
            ps_dev + pr_dev,
            ar_us - ps_dev - pr_dev
        );
    }

    // ── 单边探针只对 `bytes ≤ slot` 的档做 ──
    // 核读的是 `peer + HDR + slot*slot_bytes` 起 `bytes` 字节;`bytes > slot_bytes`
    // 时那读会**越出对端 staging 账**(生产侧由解释器 `n_bytes > slot_bytes`
    // 响亮拒挡住;这里必须同门,否则探针自己越界读 PCIe)。
    if bytes as u64 > layout.slot_bytes {
        eprintln!(
            "[arb] {:>10} {:>5} {:>6} {:>7} {:>8.2} {:>8.2} {:>8.2} {:>8.0} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
            bytes, format!("{dt:?}"), n_elems, k, ar_enq_us, ar_dev_us, ar_us,
            mbps(bytes as u64, ar_us), "—", "—", "—", "—", "—", "—", "—", "—", "—", "—"
        );
        a.free(&[payload_a.id]).await.expect("free a");
        b.free(&[payload_b.id]).await.expect("free b");
        return;
    }

    // ── R(纯对端读:produced 置 HUGE ⇒ I2 旁路)──────────────
    // 只由 A 单边循环:对端(CUDA B)不参与 ⇒ 量的是**读对端 + 本地 RMW**。
    set_hdr(a, &staging_a, 0, 0).await.expect("A hdr");
    set_hdr(b, &staging_b, HUGE, HUGE).await.expect("B hdr(HUGE)");
    let k_r = (k / 2).max(1);
    let t0 = Instant::now();
    for _ in 0..k_r {
        emit_one(a, &payload_a, &staging_a, &alias_b_on_a, bytes as u64, &layout, true)
            .await
            .expect("A reduce 发射");
    }
    let r_enq = t0.elapsed().as_secs_f64();
    a.sync().await.expect("A sync");
    let r_tot = t0.elapsed().as_secs_f64();
    let r_dev_us = per_call_us((r_tot - r_enq).max(0.0), k_r);
    assert_eq!(read_err(a, &staging_a).await, 0, "A 侧 R 探针 err 非 0");

    // ── S(纯本地拷贝:对端 consumed 置 HUGE ⇒ credit 旁路)──
    set_hdr(a, &staging_a, 0, 0).await.expect("A hdr");
    set_hdr(b, &staging_b, 0, HUGE).await.expect("B hdr(credit HUGE)");
    // ⚠️ **stage 现在必须双边测**(2026-10-17 加了尾部 rendezvous:它要等对端
    // 本轮已发布 ⇒ 单边跑必然超时 err=1)。双边档 = 生产同形(含 rendezvous),
    // 反而更有代表性。
    set_hdr(a, staging_a, 0, 0).await.expect("A hdr(双边)");
    set_hdr(b, staging_b, 0, 0).await.expect("B hdr(双边)");
    let k_s = (k / 2).max(1);
    let t0 = Instant::now();
    {
        let fa = async {
            for _ in 0..k_s {
                emit_one(a, &payload_a, &staging_a, &alias_b_on_a, bytes as u64, &layout, false)
                    .await
                    .expect("A stage 发射");
            }
        };
        let fb = async {
            for _ in 0..k_s {
                emit_one(b, &payload_b, &staging_b, &alias_a_on_b, bytes as u64, &layout, false)
                    .await
                    .expect("B stage 发射");
            }
        };
        tokio::join!(fa, fb);
    }
    let s_enq = t0.elapsed().as_secs_f64();
    let (sa, sb) = tokio::join!(a.sync(), b.sync());
    sa.expect("A sync");
    sb.expect("B sync");
    let s_tot = t0.elapsed().as_secs_f64();
    let s_dev_us = per_call_us((s_tot - s_enq).max(0.0), k_s);
    assert_eq!(read_err(a, &staging_a).await, 0, "A 侧 S 探针 err 非 0");

    // ── RR(两 rank **同时**互读;SM 路径)────────────────────────
    // ⚠️ **它不是天花板**(2026-10-17 自我纠错,§26.1):`RR` 用的是**我们自己的核**
    // (单 block × 256 + 8 路展开)⇒ 只证明"我们这套 **SM 读**协议能跑多少"。
    // 外部参照(NCCL 2-rank all-reduce,`tools/nccl-ar-probe.cu`)在同一对卡上
    // 是 **11.1GB/s**,而 `RR` 只有 7.2 ⇒ 链路另有空间。**真正的分母见 C1/C2。**
    // 留它是因为它与 `R` 成对,能读"双向并发 vs 单向"的差(7.2 vs 8.6)。
    set_hdr(a, staging_a, HUGE, 0).await.expect("A hdr(produced HUGE)");
    set_hdr(b, staging_b, HUGE, 0).await.expect("B hdr(produced HUGE)");
    let k_rr = (k / 2).max(1);
    let t0 = Instant::now();
    {
        let fa = async {
            for _ in 0..k_rr {
                emit_one(a, &payload_a, &staging_a, &alias_b_on_a, bytes as u64, &layout, true)
                    .await
                    .expect("A reduce 发射");
            }
        };
        let fb = async {
            for _ in 0..k_rr {
                emit_one(b, &payload_b, &staging_b, &alias_a_on_b, bytes as u64, &layout, true)
                    .await
                    .expect("B reduce 发射");
            }
        };
        tokio::join!(fa, fb);
    }
    let rr_enq = t0.elapsed().as_secs_f64();
    let (sa, sb) = tokio::join!(a.sync(), b.sync());
    sa.expect("A sync");
    sb.expect("B sync");
    let rr_us = per_call_us(t0.elapsed().as_secs_f64(), k_rr);
    let rr_dev_us = per_call_us((t0.elapsed().as_secs_f64() - rr_enq).max(0.0), k_rr);

    // ── C1/C2(**拷贝引擎**对端读;用户口径 2026-10-17:用第二条流/四条流对班)──
    // 为什么另量:`R`/`RR` 是 **SM 用 `ld.global` 跨 PCIe 读**(占 SM 且延迟受限),
    // 而 `memcpy_dtod_async` 走**驱动 D2D = 拷贝引擎(CE)**,不吃 SM、可与计算流并行。
    // C1 = 单边(A 从 B 的 staging 拷进自己的 payload);C2 = **两 rank 同时互拷**
    // (= "两张卡、各一条拷贝流"的四流并行的数据面读数)。协议面旁路(只量搬运)。
    let slot_off = HDR as u64 + (layout.slot_of(0) as u64) * layout.slot_bytes;
    let k_c = (k / 2).max(1);
    let c1_dev_us: f64;
    let c2_dev_us: f64;
    {
        // C1:单边
        let t0 = Instant::now();
        for _ in 0..k_c {
            a.copy_block_at(&alias_b_on_a, slot_off as usize, &payload_a, 0, bytes)
                .await
                .expect("A CE 拷贝");
        }
        let enq = t0.elapsed().as_secs_f64();
        a.sync().await.expect("A sync");
        c1_dev_us = per_call_us((t0.elapsed().as_secs_f64() - enq).max(0.0), k_c);

        // C2:双边并发(两 rank 各一条流,方向相反)
        let t0 = Instant::now();
        {
            let fa = async {
                for _ in 0..k_c {
                    a.copy_block_at(&alias_b_on_a, slot_off as usize, &payload_a, 0, bytes)
                        .await
                        .expect("A CE 拷贝");
                }
            };
            let fb = async {
                for _ in 0..k_c {
                    b.copy_block_at(&alias_a_on_b, slot_off as usize, &payload_b, 0, bytes)
                        .await
                        .expect("B CE 拷贝");
                }
            };
            tokio::join!(fa, fb);
        }
        let enq = t0.elapsed().as_secs_f64();
        let (sa, sb) = tokio::join!(a.sync(), b.sync());
        sa.expect("A sync");
        sb.expect("B sync");
        c2_dev_us = per_call_us((t0.elapsed().as_secs_f64() - enq).max(0.0), k_c);
    }

    // ── A2(★ 阶段 A 候选:CE 搬运 + 本地加)────────────────────────
    // 交换半边换通道:① CE 拷(对端 out slot → 本端 inbox,`copy_block_at` =
    // `memcpy_dtod_async`)② **本地加** —— **零新核**:把 inbox 当 `peer` 传给现有
    // reduce 核即可(`peer` 只是"要读的那个块",不关心它在哪张卡);
    // inbox.produced 置 HUGE ⇒ I2 永不阻塞(这里量数据面,不量协议)。
    // **对照基准 = `R dev`**(v1 的 reduce:SM 跨 PCIe 读 + 本地 RMW)。
    // 判据:**A2 < R dev** 才值得换通道 —— 否则 CE 只是把"读与加并行"变成串行。
    #[allow(unused_assignments)]
    let mut a2_dev_us = f64::NAN;
    {
        set_hdr(a, inbox_a, HUGE, 0).await.expect("A inbox hdr(produced HUGE)");
        set_hdr(b, staging_b, HUGE, 0).await.expect("B hdr(produced HUGE)");
        // ⚠️ **伪影排除**:reduce 的 slot = `own.consumed % own.n_slots`(读的是 **own** 的
        // n_slots)⇒ 若 own = staging_a(n_slots=2)而 inbox 只有 1 slot,第 2 次就会
        // **越界读** inbox+64+32MiB。故这里把 staging_a 的游标归零、并**按 parity 把
        // CE 拷到 reduce 将要读的那个 slot** —— 无越界、且每次读到的都是刚拷进来的数。
        set_hdr(a, staging_a, 0, 0).await.expect("A hdr(归零)");
        let k_a2 = (k / 2).max(1);
        let t0 = Instant::now();
        for i in 0..k_a2 {
            let off = HDR as u64 + (i as u64 % 2) * layout.slot_bytes;
            a.copy_block_at(alias_b_on_a, slot_off as usize, inbox_a, off as usize, bytes)
                .await
                .expect("A CE 拷进 inbox");
            emit_one(a, &payload_a, &staging_a, inbox_a, bytes as u64, &layout, true)
                .await
                .expect("A 本地加");
        }
        let enq = t0.elapsed().as_secs_f64();
        a.sync().await.expect("A sync");
        a2_dev_us = per_call_us((t0.elapsed().as_secs_f64() - enq).max(0.0), k_a2);
    }

    // ── P(推送臂 = 远程**写**;pull 协议的对偶)────────────────────
    // 绑定 = `stage(local=本端 payload, own=对端 staging, peer=对端 staging)`:
    // 写对端 staging(**远程写**,posted)、credit = 对端 consumed(远程读 8B)、
    // 发布对端 produced。这正是"push 协议"的数据面 —— 此处单边量**远程写带宽**。
    // ⚠️ 默认**跳过**:`peer_import` 的别名是**只读**授予(见 `state.rs::peer_import`
    //    "只授本端设备 R")⇒ 经它**写**对端 staging 会当场 ILLEGAL_ADDRESS
    //    (实测 0.9s 内响亮失败)。要评估 push 协议必须先加"可写对端映射"(RW import);
    //    用 `OWL_ARB_PUSH=1` 显式打开(打开即预期失败,除非 RW 授予已落地)。
    let push_on = std::env::var("OWL_ARB_PUSH").is_ok();
    #[allow(unused_assignments)]
    let mut l_dev_us = f64::NAN;
    // ── L(**纯本地 RMW**:reduce 核拿本端 staging 当 peer)────────────
    // 它量的是"AR 的本地加法半边"能跑多快 —— 与 `S`(stage 核的本地拷贝)
    // 成对,可拆出"我们单 block 核的本地访存效率"。**不需要 RW 授予**
    // (与 push 无关),故独立于 `OWL_ARB_PUSH` 门。
// ── L(push 协议的消费半:本端 accum)─────────────────────────
// 绑定 = `reduce(out=本端 payload, own=本端 staging, peer=本端 staging)`:
// 等**本端** produced(本地自旋)、读**本端** staging(本地读)+ RMW。
set_hdr(a, staging_a, HUGE, 0).await.expect("A hdr(produced HUGE)");
let k_l = (k / 2).max(1);
let t0 = Instant::now();
for _ in 0..k_l {
    emit_one(a, &payload_a, &staging_a, &staging_a, bytes as u64, &layout, true)
        .await
        .expect("A accum 发射");
}
let l_enq = t0.elapsed().as_secs_f64();
a.sync().await.expect("A sync");
l_dev_us = per_call_us((t0.elapsed().as_secs_f64() - l_enq).max(0.0), k_l);

    #[allow(unused_assignments)]
    #[allow(unused_assignments)]
    if push_on {
    set_hdr(a, staging_a, 0, 0).await.expect("A hdr");
    set_hdr(b, staging_b, 0, HUGE).await.expect("B hdr(credit HUGE)");
    let k_p = (k / 2).max(1);
    let t0 = Instant::now();
    for _ in 0..k_p {
        a.launch(
            ArStageCall::new(
                payload_a.clone(),
                alias_b_on_a.clone(), // header:v1 位(此臂 = 直接写对端块)
                alias_b_on_a.clone(),
                alias_b_on_a.clone(), // data:写对端块(= 旧 push 臂语义)
                bytes as u64,
                layout.slot_bytes,
                0,
            )
            .to_launch(),
        )
        .await
        .expect("A 推送发射");
    }
    let p_enq = t0.elapsed().as_secs_f64();
    a.sync().await.expect("A sync");
    let _p_dev_us = per_call_us((t0.elapsed().as_secs_f64() - p_enq).max(0.0), k_p);

    } /* push_on */

    // ── ★ A3(CE 三段 = **生产同形**;2026-10-17 接线)──────────────
    // 与 `AR` 段**同构驱动**(两 rank 并发、含 credit/rendezvous 协议),唯一差别 =
    // 中段走拷贝引擎 + `n_slots=1`(CE 要求两端固定地址;见 `ar.rs::N_SLOTS_CE`)。
    // 判据(§29):AR(5.24MB) 590 → ≤450µs。
    #[allow(unused_assignments)]
    let mut a3_us = f64::NAN;
    #[allow(unused_assignments)]
    let mut a3_dev_us = f64::NAN;
    {
        // 声明改**单格**(块尺寸不变 —— 只是不再乒乓)
        set_hdr_slots(a, staging_a, 0, 0, 1).await.expect("A hdr(slots=1)");
        set_hdr_slots(b, staging_b, 0, 0, 1).await.expect("B hdr(slots=1)");
        set_hdr_slots(a, inbox_a, HUGE, 0, 1).await.expect("A inbox hdr(slots=1)");
        set_hdr_slots(b, inbox_b, HUGE, 0, 1).await.expect("B inbox hdr(slots=1)");
        for _ in 0..2 {
            let (ra, rb) = tokio::join!(
                emit_ce(a, &payload_a, &staging_a, &alias_b_on_a, &inbox_a, bytes as u64, &layout),
                emit_ce(b, &payload_b, &staging_b, &alias_a_on_b, &inbox_b, bytes as u64, &layout),
            );
            ra.expect("A CE 预热");
            rb.expect("B CE 预热");
            let (sa, sb) = tokio::join!(a.sync(), b.sync());
            sa.expect("A sync");
            sb.expect("B sync");
        }
        let t0 = Instant::now();
        {
            let fa = async {
                for _ in 0..k {
                    emit_ce(a, &payload_a, &staging_a, &alias_b_on_a, &inbox_a, bytes as u64, &layout)
                        .await
                        .expect("A CE 发射");
                }
            };
            let fb = async {
                for _ in 0..k {
                    emit_ce(b, &payload_b, &staging_b, &alias_a_on_b, &inbox_b, bytes as u64, &layout)
                        .await
                        .expect("B CE 发射");
                }
            };
            tokio::join!(fa, fb);
        }
        let a3_enq = t0.elapsed().as_secs_f64();
        let (sa, sb) = tokio::join!(a.sync(), b.sync());
        sa.expect("A sync");
        sb.expect("B sync");
        let a3_tot = t0.elapsed().as_secs_f64();
        a3_us = per_call_us(a3_tot, k);
        a3_dev_us = per_call_us((a3_tot - a3_enq).max(0.0), k);
        assert_eq!(read_err(a, &staging_a).await, 0, "A 侧 A3 探针 err 非 0");
        assert_eq!(read_err(b, &staging_b).await, 0, "B 侧 A3 探针 err 非 0");
        // 恢复两格声明(本块接着给下一个 case 用)
        set_hdr(a, staging_a, 0, 0).await.expect("A hdr(恢复)");
        set_hdr(b, staging_b, 0, 0).await.expect("B hdr(恢复)");
        set_hdr(a, inbox_a, HUGE, 0).await.expect("A inbox hdr(恢复)");
        set_hdr(b, inbox_b, HUGE, 0).await.expect("B inbox hdr(恢复)");
    }

    eprintln!(
        "[arb] {:>10} {:>5} {:>6} {:>7} {:>8.2} {:>8.2} {:>8.2} {:>8.0} {:>8.2} {:>8.2} {:>8.2} {:>8.0} {:>8.2} {:>8.2} {:>8.0} {:>8.2} {:>8.2} {:>8.0} {:>8.2} {:>8.2} {:>8.0}",
        bytes, format!("{dt:?}"), n_elems, k, ar_enq_us, ar_dev_us, ar_us,
        mbps(bytes as u64, ar_us), r_dev_us, s_dev_us, rr_dev_us,
        mbps(bytes as u64, rr_us), c1_dev_us, c2_dev_us,
        mbps(bytes as u64, c2_dev_us), l_dev_us, a2_dev_us,
        mbps(bytes as u64, a2_dev_us), a3_us, a3_dev_us,
        mbps(bytes as u64, a3_us)
    );

    // ── ★ **AR 旁路协议**(2026-10-19;§50.9 的判据)────────────────────────────
    // 两侧 header 全置 HUGE ⇒ 两个核**跳过所有跨卡等待**(credit 前置 / 数据 I2 / 尾部 rendezvous)。
    // **差值 = 协议钱**(§50.9 估的 50–70µs 是否成立,就看这一行)。
    // ⚠️ 旁路下数据是脏的(不等对端),只用来量**时间**;放在 run_case 末尾以免污染别的臂。
    {
        set_hdr(a, staging_a, HUGE, HUGE).await.expect("A hdr(HUGE,HUGE)");
        set_hdr(b, staging_b, HUGE, HUGE).await.expect("B hdr(HUGE,HUGE)");
        let t0 = Instant::now();
        {
            let fa = async {
                for _ in 0..k {
                    if push_ar { emit_ar_push(a, &payload_a, &staging_a, &alias_b_on_a, bytes as u64, &layout).await }
                    else { emit_ar(a, &payload_a, &staging_a, &alias_b_on_a, bytes as u64, &layout).await }.expect("A 旁路发射");
                }
            };
            let fb = async {
                for _ in 0..k {
                    if push_ar { emit_ar_push(b, &payload_b, &staging_b, &alias_a_on_b, bytes as u64, &layout).await }
                    else { emit_ar(b, &payload_b, &staging_b, &alias_a_on_b, bytes as u64, &layout).await }.expect("B 旁路发射");
                }
            };
            tokio::join!(fa, fb);
            let (sa, sb) = tokio::join!(a.sync(), b.sync());
            sa.expect("A sync"); sb.expect("B sync");
        }
        let byp = per_call_us(t0.elapsed().as_secs_f64(), k);
        eprintln!(
            "[arb-bypass] {bytes:>9} B  AR(全协议)={ar_us:8.2} · AR(旁路协议)={byp:8.2} \
             ⇒ **协议钱 = {:6.2} µs/次**({:.1}%)",
            ar_us - byp, 100.0 * (ar_us - byp) / ar_us.max(1e-9)
        );
    }

    // ── ★ **图内 AR 回放**(2026-10-19;§50.6 的缺口)────────────────────────────
    // 立案:`ar_channel_bench` 的 10 KB 档 `enq ≈ 37µs` **与载荷无关**(5.24MB 档也是 33.5µs)
    // ⇒ bench 的 host 命令面把设备盖住了,`dev ≈ 0` 因此**没有分辨力**
    //   (**"decode 侧 AR 有多贵"这个问题,bench 从架构上回答不了**)。
    // 做法:把 **N 对 AR(stage+reduce)捕进一张图**(两卡各一张,同时回放)⇒ host 只出一次 launch;
    //   **墙钟 / N = 每次 AR 的真设备时延**(含 credit 前置 / rendezvous 尾段那两次跨卡往返)。
    if std::env::var("OWL_ARB_GRAPH").is_ok() {
        let n: usize = std::env::var("OWL_ARB_GRAPH_N")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(32);
        set_hdr(a, staging_a, 0, 0).await.expect("A hdr(图内)");
        set_hdr(b, staging_b, 0, 0).await.expect("B hdr(图内)");
        a.graph_begin().await.expect("graph_begin A");
        for _ in 0..n {
            if push_ar {
                emit_ar_push(a, &payload_a, &staging_a, &alias_b_on_a, bytes as u64, &layout)
                    .await.expect("A push 发射");
            } else {
                emit_ar(a, &payload_a, &staging_a, &alias_b_on_a, bytes as u64, &layout)
                    .await.expect("A 发射");
            }
        }
        let ga = a.graph_end().await.expect("graph_end A");
        b.graph_begin().await.expect("graph_begin B");
        for _ in 0..n {
            if push_ar {
                emit_ar_push(b, &payload_b, &staging_b, &alias_a_on_b, bytes as u64, &layout)
                    .await.expect("B push 发射");
            } else {
                emit_ar(b, &payload_b, &staging_b, &alias_a_on_b, bytes as u64, &layout)
                    .await.expect("B 发射");
            }
        }
        let gb = b.graph_end().await.expect("graph_end B");
        {
            let (ra, rb) = tokio::join!(a.graph_launch(ga), b.graph_launch(gb));
            ra.expect("warm A"); rb.expect("warm B");
            let (sa, sb) = tokio::join!(a.sync(), b.sync());
            sa.expect("warm sync A"); sb.expect("warm sync B");
        }
        let t0 = Instant::now();
        {
            let (ra, rb) = tokio::join!(a.graph_launch(ga), b.graph_launch(gb));
            ra.expect("replay A"); rb.expect("replay B");
            let (sa, sb) = tokio::join!(a.sync(), b.sync());
            sa.expect("sync A"); sb.expect("sync B");
        }
        let us = t0.elapsed().as_secs_f64() * 1e6 / (n as f64);
        // ★ **同一张图,换个 header 再放一遍** ⇒ 协议钱(header 是运行期读的)
        set_hdr(a, staging_a, HUGE, HUGE).await.expect("A hdr(HUGE,HUGE)");
        set_hdr(b, staging_b, HUGE, HUGE).await.expect("B hdr(HUGE,HUGE)");
        {
            let (ra, rb) = tokio::join!(a.graph_launch(ga), b.graph_launch(gb));
            ra.expect("warm2 A"); rb.expect("warm2 B");
            let (sa, sb) = tokio::join!(a.sync(), b.sync());
            sa.expect("warm2 sync A"); sb.expect("warm2 sync B");
        }
        let t1 = Instant::now();
        {
            let (ra, rb) = tokio::join!(a.graph_launch(ga), b.graph_launch(gb));
            ra.expect("replay2 A"); rb.expect("replay2 B");
            let (sa, sb) = tokio::join!(a.sync(), b.sync());
            sa.expect("sync2 A"); sb.expect("sync2 B");
        }
        let byp = t1.elapsed().as_secs_f64() * 1e6 / (n as f64);
        eprintln!(
            "[arb-graph] {bytes:>9} B n={n:<4} **图内 AR = {us:8.2} µs/次**(真设备时延) · \
             旁路协议 = {byp:8.2} ⇒ **协议钱 = {:6.2} µs/次**({:.1}%)",
            us - byp, 100.0 * (us - byp) / us.max(1e-9)
        );
        assert_eq!(read_err(a, &staging_a).await, 0, "图内 A err 非 0");
        assert_eq!(read_err(b, &staging_b).await, 0, "图内 B err 非 0");
        // ★ **核数账**:同样捕 N 个**单核**(reduce 一个),看"每次一个核"的图内成本
        //   ⇒ 与上面的"两核 AR"相减 = **融合两核能省多少**(decode 档唯一还有量的方向)
        {
            let n1 = n;
            // ⚠️ **每条臂必须自带 header 状态**(§50.10 卡死教训:继承旁路后的污染值
            //    会让 reduce 的尾段 credit 等待进 5s 超时循环)⇒ 照 bench 的 `R` 臂约定置 HUGE。
            // ⚠️ **不能用 HUGE 当 "consumed" 的最终态**:计数一漂,reduce 的
            //    `while (*peer_prod <= cons)` 就永远不成立(实测卡满 5s 超时 ×N)。
            //    ⇒ 改用 **stage**(只等 credit/rendezvous):A=(0,0)、B=(0,HUGE) ⇒ 全部直过。
            set_hdr(a, staging_a, 0, 0).await.expect("A hdr(单核臂)");
            set_hdr(b, staging_b, 0, HUGE).await.expect("B hdr(单核臂)");
            a.graph_begin().await.expect("gb1");
            for _ in 0..n1 {
                emit_one(a, &payload_a, &staging_a, &alias_b_on_a, bytes as u64, &layout, false)
                    .await.expect("A stage 单核发射");
            }
            let ga1 = a.graph_end().await.expect("ge1");
            b.graph_begin().await.expect("gb2");
            for _ in 0..n1 {
                emit_one(b, &payload_b, &staging_b, &alias_a_on_b, bytes as u64, &layout, false)
                    .await.expect("B stage 单核发射");
            }
            let gb1 = b.graph_end().await.expect("ge2");
            {
                let (ra, rb) = tokio::join!(a.graph_launch(ga1), b.graph_launch(gb1));
                ra.expect("w1 A"); rb.expect("w1 B");
                let (sa, sb) = tokio::join!(a.sync(), b.sync());
                sa.expect("s1 A"); sb.expect("s1 B");
            }
            let t2 = Instant::now();
            {
                let (ra, rb) = tokio::join!(a.graph_launch(ga1), b.graph_launch(gb1));
                ra.expect("r1 A"); rb.expect("r1 B");
                let (sa, sb) = tokio::join!(a.sync(), b.sync());
                sa.expect("s1b A"); sb.expect("s1b B");
            }
            let one = t2.elapsed().as_secs_f64() * 1e6 / (n1 as f64);
            eprintln!(
                "[arb-graph] {bytes:>9} B  **单核(1 node)/次 = {one:8.2} µs** vs **两核 AR = {us:8.2} µs** \
                 ⇒ **第二个核的边际成本 = {:6.2} µs/次**",
                us - one
            );
            a.graph_destroy(ga1).await.expect("d1 A");
            b.graph_destroy(gb1).await.expect("d1 B");
        }

        a.graph_destroy(ga).await.expect("destroy A");
        b.graph_destroy(gb).await.expect("destroy B");
    }

    // ── ★ **AR-2 图内回放**(2026-10-23):与 v1 图内同尺,只换核 ──────
    // 独立开关 OWL_ARB_AR2=1;只测 ≤256KiB(生产 AR-2 门;大包仍 v1/fused)。
    if std::env::var("OWL_ARB_AR2").is_ok() && (bytes as u64) <= 256 * 1024 {
        let n: usize = std::env::var("OWL_ARB_GRAPH_N")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(32);
        // AR-2 游标在 header 48/56;set_hdr 会清零整 64B(含 48..64)。
        set_hdr(a, staging_a, 0, 0).await.expect("A hdr(AR2)");
        set_hdr(b, staging_b, 0, 0).await.expect("B hdr(AR2)");
        a.graph_begin().await.expect("graph_begin A2");
        for _ in 0..n {
            emit_ar2(a, &payload_a, &staging_a, &alias_b_on_a, bytes as u64, &layout)
                .await.expect("A AR2 发射");
        }
        let ga2 = a.graph_end().await.expect("graph_end A2");
        b.graph_begin().await.expect("graph_begin B2");
        for _ in 0..n {
            emit_ar2(b, &payload_b, &staging_b, &alias_a_on_b, bytes as u64, &layout)
                .await.expect("B AR2 发射");
        }
        let gb2 = b.graph_end().await.expect("graph_end B2");
        {
            let (ra, rb) = tokio::join!(a.graph_launch(ga2), b.graph_launch(gb2));
            ra.expect("AR2 warm A"); rb.expect("AR2 warm B");
            let (sa, sb) = tokio::join!(a.sync(), b.sync());
            sa.expect("AR2 warm sync A"); sb.expect("AR2 warm sync B");
        }
        let t0 = Instant::now();
        {
            let (ra, rb) = tokio::join!(a.graph_launch(ga2), b.graph_launch(gb2));
            ra.expect("AR2 replay A"); rb.expect("AR2 replay B");
            let (sa, sb) = tokio::join!(a.sync(), b.sync());
            sa.expect("AR2 sync A"); sb.expect("AR2 sync B");
        }
        let us = t0.elapsed().as_secs_f64() * 1e6 / (n as f64);
        eprintln!(
            "[arb-ar2] {bytes:>9} B n={n:<4} **AR-2 图内 = {us:8.2} µs/次**(真设备时延)"
        );
        assert_eq!(read_err(a, &staging_a).await, 0, "AR2 A err 非 0");
        assert_eq!(read_err(b, &staging_b).await, 0, "AR2 B err 非 0");
        a.graph_destroy(ga2).await.expect("AR2 destroy A");
        b.graph_destroy(gb2).await.expect("AR2 destroy B");
    }

    // ── ★ **1-stage 图内回放**(对称 slot;2026-10-23)──────────────
    // 语义:out = src + peer_data;src/peer_data 是对称 slot,不再经 staging 拷贝。
    // 只在 OWL_ARB_1STAGE=1 且 bytes≤256KiB 时跑。
    if std::env::var("OWL_ARB_1STAGE").is_ok() && (bytes as u64) <= SLOT_1S as u64 {
        let n: usize = std::env::var("OWL_ARB_GRAPH_N")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(32);
        set_hdr(a, staging_a, 0, 0).await.expect("A hdr(1stage)");
        set_hdr(b, staging_b, 0, 0).await.expect("B hdr(1stage)");
        a.graph_begin().await.expect("graph_begin A1s");
        for _ in 0..n {
            emit_1stage(
                a,
                &payload_a,
                &one_stage.slot_a,
                &staging_a,
                &alias_b_on_a,
                &one_stage.alias_b_on_a,
                bytes as u64,
            )
            .await
            .expect("A 1stage 发射");
        }
        let ga1s = a.graph_end().await.expect("graph_end A1s");
        b.graph_begin().await.expect("graph_begin B1s");
        for _ in 0..n {
            emit_1stage(
                b,
                &payload_b,
                &one_stage.slot_b,
                &staging_b,
                &alias_a_on_b,
                &one_stage.alias_a_on_b,
                bytes as u64,
            )
            .await
            .expect("B 1stage 发射");
        }
        let gb1s = b.graph_end().await.expect("graph_end B1s");
        {
            let (ra, rb) = tokio::join!(a.graph_launch(ga1s), b.graph_launch(gb1s));
            ra.expect("1s warm A"); rb.expect("1s warm B");
            let (sa, sb) = tokio::join!(a.sync(), b.sync());
            sa.expect("1s warm sync A"); sb.expect("1s warm sync B");
        }
        let t0 = Instant::now();
        {
            let (ra, rb) = tokio::join!(a.graph_launch(ga1s), b.graph_launch(gb1s));
            ra.expect("1s replay A"); rb.expect("1s replay B");
            let (sa, sb) = tokio::join!(a.sync(), b.sync());
            sa.expect("1s sync A"); sb.expect("1s sync B");
        }
        let us = t0.elapsed().as_secs_f64() * 1e6 / (n as f64);
        eprintln!(
            "[arb-1stage] {bytes:>9} B n={n:<4} **1-stage 图内 = {us:8.2} µs/次**(真设备时延)"
        );
        assert_eq!(read_err(a, &staging_a).await, 0, "1stage A err 非 0");
        assert_eq!(read_err(b, &staging_b).await, 0, "1stage B err 非 0");
        // 路径存在性:slot 两侧都清零 ⇒ 1-stage 的 out 应为全 0
        // (不是数值门;真数值门在 tp_g8_gate / ar_transport_gate)。
        let mut za = vec![0u8; bytes];
        a.dtoh(&payload_a, &mut za).await.expect("1s dtoh A");
        assert!(za.iter().all(|&x| x == 0), "1-stage A 输出非 0(路径没写?)");
        let mut zb = vec![0u8; bytes];
        b.dtoh(&payload_b, &mut zb).await.expect("1s dtoh B");
        assert!(zb.iter().all(|&x| x == 0), "1-stage B 输出非 0(路径没写?)");
        a.graph_destroy(ga1s).await.expect("1s destroy A");
        b.graph_destroy(gb1s).await.expect("1s destroy B");
    }

    a.free(&[payload_a.id]).await.expect("free a");
    b.free(&[payload_b.id]).await.expect("free b");
}

#[tokio::test]
#[ignore = "bench:AR 通道微基准(3080 对/3090 Ti 对;隔离测量)"]
async fn comm_ar_channel_bench() {
    let (Some(oa), Some(ob)) = (dev("OWL_PEER_TEST_DEV_A"), dev("OWL_PEER_TEST_DEV_B")) else {
        owl_shared::testkit::report::skip("需 OWL_PEER_TEST_DEV_A / OWL_PEER_TEST_DEV_B(两张可 P2P 的空闲卡)");
        return;
    };
    assert_ne!(oa, ob, "A/B 必须是两张不同卡");
    let (ua, ub) = (uuid_of(oa), uuid_of(ob));
    let mut a = assemble(oa, "a");
    let mut b = assemble(ob, "b");

    // ── VMM 对称握手(与生产同参:slot 32MiB × 2)──────────────────
    let staging_bytes = HDR as u64 + SLOT * N_SLOTS as u64;
    let (staging_a, cred_a) = a.peer_alloc(staging_bytes as usize, "arb:staging-a").await.expect("A PeerAlloc");
    let (staging_b, cred_b) = b.peer_alloc(staging_bytes as usize, "arb:staging-b").await.expect("B PeerAlloc");
    // push AR 档 ⇒ 别名必须 RW(否则 stage 写对端槽当场 ILLEGAL_ADDRESS)
    let push_ar_on = std::env::var("OWL_ARB_AR_PUSH").is_ok();
    let alias_b_on_a = if push_ar_on {
        a.peer_import_rw(cred_b, staging_bytes as usize, &ub, "arb:alias-rw-b@a").await
    } else {
        a.peer_import(cred_b, staging_bytes as usize, &ub, "arb:alias-b@a").await
    }
    .expect("A 导入 B");
    let alias_a_on_b = if push_ar_on {
        b.peer_import_rw(cred_a, staging_bytes as usize, &ua, "arb:alias-rw-a@b").await
    } else {
        b.peer_import(cred_a, staging_bytes as usize, &ua, "arb:alias-a@b").await
    }
    .expect("B 导入 A");
    let layout = StagingLayout::new(SLOT, N_SLOTS);
    // **A2 探针的收件箱**(阶段 A 候选:CE 把对端 slot 拷进这里,再用现有 reduce 核
    // 在本地做 `payload += inbox`)。单缓冲即可 —— 同一流保序 ⇒ 不存在"上一条 AR 还
    // 没读完就被下一条覆盖";大小 = 2 slot + header(reduce 的 slot = own.consumed%2 ⇒ 必须两格都在)。
    let (inbox_a, _inbox_cred) = a
        .peer_alloc((2 * SLOT + HDR as u64) as usize, "arb:inbox-a")
        .await
        .expect("A inbox peer_alloc");

    // A3 段需要两侧各一个收件箱(双向并发)
    let (inbox_b, _inbox_cred_b) = b
        .peer_alloc((2 * SLOT + HDR as u64) as usize, "arb:inbox-b")
        .await
        .expect("B inbox peer_alloc");

    // 1-stage 对称 slot(256KiB;只给 ≤256KiB 档;无 header)
    let (slot_a, cred_slot_a) = a
        .peer_alloc(SLOT_1S, "arb:1s-slot-a")
        .await
        .expect("A 1s slot");
    let (slot_b, cred_slot_b) = b
        .peer_alloc(SLOT_1S, "arb:1s-slot-b")
        .await
        .expect("B 1s slot");
    let alias_slot_b_on_a = a
        .peer_import(cred_slot_b, SLOT_1S, &ub, "arb:1s-alias-b@a")
        .await
        .expect("A 1s alias");
    let alias_slot_a_on_b = b
        .peer_import(cred_slot_a, SLOT_1S, &ua, "arb:1s-alias-a@b")
        .await
        .expect("B 1s alias");
    a.memset_zero(&slot_a, SLOT_1S).await.expect("A 1s zero");
    b.memset_zero(&slot_b, SLOT_1S).await.expect("B 1s zero");
    let one_stage = OneStageSyms {
        slot_a: slot_a.clone(),
        slot_b: slot_b.clone(),
        alias_b_on_a: alias_slot_b_on_a.clone(),
        alias_a_on_b: alias_slot_a_on_b.clone(),
    };

    eprintln!(
        "[arb] devices: A=ord{oa}({ua}) B=ord{ob}({ub}) slot={}MiB×{N_SLOTS} AR 档={}",
        SLOT >> 20,
        if push_ar_on { "push(写对端)" } else { "v1(读对端)" }
    );
    eprintln!(
        "[arb] {:>10} {:>5} {:>6} {:>7} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "bytes", "dtype", "elems", "K", "enq", "dev", "AR µs", "MB/s",
        "R dev", "S dev", "RR dev", "RR MB/s", "C1 dev", "C2 dev", "C2 MB/s", "L dev", "A2 dev", "A2 MB/s",
        "A3 µs", "A3 dev", "A3 MB/s"
    );

    // 档表:env `OWL_ARB_SIZES=4096,10240,...`(逗号分隔字节)+ `OWL_ARB_DTYPE=f16|f32`
    // + `OWL_ARB_K=200` 可覆盖 ⇒ **尺寸阶梯不用重编译**(小载荷异常待定谳)。
    let env_k: u32 = std::env::var("OWL_ARB_K").ok().and_then(|v| v.parse().ok()).unwrap_or(200);
    let env_dt = match std::env::var("OWL_ARB_DTYPE").as_deref() {
        Ok("f32") => Dtype::F32,
        Ok("bf16") => Dtype::BF16,
        _ => Dtype::F16,
    };
    let env_ladder: Option<Vec<(usize, Dtype, u32)>> = std::env::var("OWL_ARB_SIZES").ok().map(|v| {
        v.split(',')
            .filter_map(|t| t.trim().parse::<usize>().ok())
            .map(|b| (b, env_dt, env_k))
            .collect()
    });
    if let Some(ladder) = env_ladder {
        for (bytes, dt, k) in ladder {
            run_case(&mut a, &mut b, &staging_a, &staging_b, &alias_b_on_a, &alias_a_on_b, &inbox_a, &inbox_b, &one_stage, &layout, bytes, dt, k)
                .await;
        }
        eprintln!("[arb] done");
        return;
    }

    // 27B 真形状(hidden 5120)+ 边界档
    let cases: [(usize, Dtype, u32); 10] = [
        (4, Dtype::F16, 400),                              // 发射吞吐档(核 ≈ 0)
        (5120 * 2, Dtype::F16, 400),                       // decode 1 行 = 10KB
        (128 * 5120 * 2, Dtype::F16, 200),                 // 1.31MB(128 行)
        (512 * 5120 * 2, Dtype::F16, 100),                 // ★ 5.24MB(生产 chunk 512)
        (1024 * 5120 * 2, Dtype::F16, 60),                 // 10.49MB(生产 chunk 1024)
        ((SLOT as usize), Dtype::F16, 40),                 // 32MiB(1 块上界)
        ((SLOT as usize) + 2, Dtype::F16, 30),             // 32MiB+2(跨块)
        (80 << 20, Dtype::F16, 20),                        // 80MiB(3 块;INFO)
        // ★ **重复档**(2026-10-17):小载荷首跑异常 ⇒ 用"同档再跑一次"判别
        //   "一次性代价(冷启动/TLB/时钟)"与"尺寸依赖代价"。
        (5120 * 2, Dtype::F16, 400),                       // 10KB 复跑
        (128 * 5120 * 2, Dtype::F16, 200),                 // 1.31MB 复跑
        ];
    // 同字节 f32 对照(元素数减半 ⇒ 区分"迭代延迟暴露"与"带宽受限")
    let f32_cases: [(usize, Dtype, u32); 2] = [
        (512 * 5120 * 2, Dtype::F32, 60),
        (1024 * 5120 * 2, Dtype::F32, 30),
    ];

    for (bytes, dt, k) in cases.iter().chain(f32_cases.iter()) {
        run_case(&mut a, &mut b, &staging_a, &staging_b, &alias_b_on_a, &alias_a_on_b, &inbox_a, &inbox_b, &one_stage, &layout, *bytes, *dt, *k).await;
    }

    // ── 收尾:两侧 header 复原(不留 HUGE 哨兵给下一轮)──────────
    set_hdr(&mut a, &staging_a, 0, 0).await.expect("A hdr 收尾");
    set_hdr(&mut b, &staging_b, 0, 0).await.expect("B hdr 收尾");
    eprintln!("[arb] done");
}

//! **graph_memcpy_gate —— 图内设备搬运(`CopyBlock`)的"进图 / 回放生效 / 保序"门**。
//!
//! 立案出处:`docs/arch/TP2-施工设计.md` §31(CE 接线"能进图但 AR 未生效")。
//!
//! # 为什么先做这道门
//!
//! §30 把 CE 刀的死因收窄到两件事:**(a) 捕获白名单**(已放开)与
//! **(b) 放行 ≠ 生效**。后者**不涉模型、不涉 AR 协议** —— 用最小用例就能判定,
//! 是这一刀最便宜、最决定性的一步。
//!
//! | 判据 | 做法 | 能区分什么 |
//! |---|---|---|
//! | **T1a 进图** | 窗内 `CopyBlock(src→dst)` → 回放 → `dst == src` | 搬运是否**真被记进图**(而非静默丢弃) |
//! | **T1b 回放生效** | 图外**清零 dst** → 再回放 → `dst == src` | 是"**回放真执行**"还是"捕获期 eager 拷了一次" |
//! | **T2 保序** | 窗内 `CopyBlock` 后紧跟消费核(`out = dst + dst`)→ 回放 | 搬运节点与核节点的**流序保序**(§27.5 A4 的图内版) |
//!
//! ★ **T1a 与 T1b 必须一起绿**:单看 T1a,"捕获期只 eager 拷了一次"也会绿(假绿)。
//! T1b 的清零发生在**图外**、回放之前 ⇒ 只有"回放真的重新执行了搬运节点"才能复原。
//!
//! ★ T2 的判据是**双值可辨**的:窗内先把 dst 种成 `9.0` ⇒
//! `out[0] = 2.0`(读到 memcpy 后的 1.0)/ `18.0`(核读到旧值 9.0)/ `0`(都没跑)。
//! 缺"预种"这条前置,T2 的失败会被 T1a 的失败掩埋。
//!
//! # 运行
//!
//! ```bash
//! OWL_GRAPH_MEMCPY_DEV=2 cargo test -p owl-cuda --test graph_memcpy_gate -- --nocapture
//! ```
//! 未设 `OWL_GRAPH_MEMCPY_DEV` ⇒ 回落 `OWL_TEST_DEVICE`(默认 0)。
//! (CUDA ordinal 表:本 boot `0/1 = 3090 Ti`,`2/3 = 3080`;`nvidia-smi` 的 index 与之不同序。)

use owl_cuda::{DeviceClient as _, DeviceSelector, Dtype, GpuClient, GpuServer, Shape};
use owl_iface::contract::{Arg, Bytes, KernelSource, LaunchMsg};
// ============================================================================
// 组装(同 gpu_server.rs / peer_shared_gate.rs 家法)
// ============================================================================

fn assemble(ordinal: usize, tag: &str) -> GpuClient {
    let (tx, rx) = std::sync::mpsc::channel::<owl_cuda::Command>();
    let server = GpuServer::new(rx, DeviceSelector::Ordinal(ordinal), None);
    let client = GpuClient::new(tx);
    std::thread::Builder::new()
        .name(format!("owl-gpu-gmemcpy-{tag}"))
        .spawn(move || server.run().expect("server run 异常退出"))
        .expect("起 server 线程失败");
    client
}

fn dev() -> usize {
    std::env::var("OWL_GRAPH_MEMCPY_DEV")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(owl_cuda::test_device_ordinal)
}

fn le_f32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn de_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

// ============================================================================
// 消费核:`owl_add_f32`(真资产核 —— 零运行期编译硬门,不能用内联 `.cu`)
// ============================================================================

const ADD_KERNEL: &str = "owl_add_f32";

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

async fn read_f32(c: &mut GpuClient, blk: &Bytes, n: usize) -> Vec<f32> {
    c.sync().await.expect("sync(取数前)");
    let mut buf = vec![0u8; n * 4];
    c.dtoh(blk, &mut buf).await.expect("dtoh");
    de_f32(&buf)
}

// ============================================================================
// T1:进图 + 回放生效
// ============================================================================
#[tokio::test]
async fn t1_captured_memcpy_enters_graph_and_replays() {
    let d = dev();
    let mut c = assemble(d, "t1");
    const N: usize = 4096;
    let ones = le_f32(&vec![1.0f32; N]);
    let nines = le_f32(&vec![9.0f32; N]);

    let src = owl_iface::htod!(c, Dtype::F32, &Shape::from(vec![N]), &ones)
        .await
        .expect("htod src")
        .view(Dtype::F32, 1);
    let seed = owl_iface::htod!(c, Dtype::F32, &Shape::from(vec![N]), &nines)
        .await
        .expect("htod seed")
        .view(Dtype::F32, 1);
    let dst = owl_iface::alloc_uninit!(c, Dtype::F32, N)
        .await
        .expect("alloc dst")
        .view(Dtype::F32, 1);
    let junk = owl_iface::alloc_uninit!(c, Dtype::F32, N)
        .await
        .expect("alloc junk")
        .view(Dtype::F32, 1);
    c.copy_block_at(&seed, 0, &dst, 0, N * 4).await.expect("种 dst = 9.0");
    c.sync().await.expect("sync seed");
    assert_eq!(read_f32(&mut c, &dst, 1).await[0], 9.0, "前置:种 dst 未生效");

    // ── 捕获窗:[无关核(满足空窗哨兵), D2D 搬运] ────────────────────
    // ⚠️ 陪跑核必须与搬运**不同格**:否则"搬运是否生效"会被陪跑核的输出掩埋。
    c.graph_begin().await.expect("graph_begin");
    c.launch(add_msg(&src, &src, &junk, N)).await.expect("窗内陪跑核");
    c.copy_block_at(&src, 0, &dst, 0, N * 4).await.expect("窗内 CopyBlock 应被放行");
    let gid = c.graph_end().await.expect("graph_end");

    // ── T1a:回放一次 → dst 应为 src(1.0)───────────────────────────
    c.graph_launch(gid).await.expect("replay#1");
    let g1 = read_f32(&mut c, &dst, N).await;
    assert_eq!(
        g1[0], 1.0,
        "T1a 红:窗内 CopyBlock 在**首次回放后未生效**(dst[0]={})。\
         9.0 = 搬运没执行(节点没进图);0 = 从未写过。dev={d}",
        g1[0]
    );
    assert!(g1.iter().all(|&v| v == 1.0), "T1a 红:除首元素外也不符(g1[1]={})", g1[1]);

    // ── T1b:图外清零 dst → 再回放 → 只有"回放真执行"才能复原 ────────
    c.memset_zero(&dst, N * 4).await.expect("图外清零 dst");
    assert_eq!(
        read_f32(&mut c, &dst, 1).await[0], 0.0,
        "前置自证:图外 memset_zero 未生效(否则 T1b 无分辨力)"
    );
    c.graph_launch(gid).await.expect("replay#2");
    let g2 = read_f32(&mut c, &dst, N).await;
    assert_eq!(
        g2[0], 1.0,
        "T1b 红:清零后**回放不能复原** dst ⇒ 搬运只在捕获期(eager)执行过一次,\
         并未作为节点进入图(dst[0]={})。dev={d}",
        g2[0]
    );

    c.free(&[src.id, seed.id, dst.id, junk.id]).await.expect("free");
    c.graph_destroy(gid).await.expect("graph_destroy");
    eprintln!("[graph_memcpy_gate] T1a/T1b 绿(dev={d}, N={N}, 回放 2 次)");
}

// ============================================================================
// T2:搬运 → 消费核 的流序保序(§27.5 A4 的图内版)
// ============================================================================

#[tokio::test]
async fn t2_captured_memcpy_orders_before_consumer_kernel() {
    let d = dev();
    let mut c = assemble(d, "t2");
    const N: usize = 4096;
    let ones = le_f32(&vec![1.0f32; N]);
    let nines = le_f32(&vec![9.0f32; N]);

    let src = owl_iface::htod!(c, Dtype::F32, &Shape::from(vec![N]), &ones)
        .await
        .expect("htod src")
        .view(Dtype::F32, 1);
    let seed = owl_iface::htod!(c, Dtype::F32, &Shape::from(vec![N]), &nines)
        .await
        .expect("htod seed")
        .view(Dtype::F32, 1);
    let dst = owl_iface::alloc_uninit!(c, Dtype::F32, N)
        .await
        .expect("alloc dst")
        .view(Dtype::F32, 1);
    let out = owl_iface::alloc_uninit!(c, Dtype::F32, N)
        .await
        .expect("alloc out")
        .view(Dtype::F32, 1);
    c.copy_block_at(&seed, 0, &dst, 0, N * 4).await.expect("种 dst = 9.0");
    c.sync().await.expect("sync seed");

    // ── 捕获窗:[D2D 搬运, 消费核 `out = dst + dst`] ─────────────────
    c.graph_begin().await.expect("graph_begin");
    c.copy_block_at(&src, 0, &dst, 0, N * 4).await.expect("窗内 CopyBlock");
    c.launch(add_msg(&dst, &dst, &out, N)).await.expect("窗内消费核(读 dst)");
    let gid = c.graph_end().await.expect("graph_end");

    c.graph_launch(gid).await.expect("replay");
    let g = read_f32(&mut c, &out, N).await;
    assert_eq!(
        g[0], 2.0,
        "T2 红:消费核读到**旧值**(out[0]={}) ⇒ 搬运与核**未保序**或搬运没进图。\
         18.0 = 核先跑(读到种的 9.0);0 = 两者都没跑。dev={d}",
        g[0]
    );

    c.free(&[src.id, seed.id, dst.id, out.id]).await.expect("free");
    c.graph_destroy(gid).await.expect("graph_destroy");
    eprintln!("[graph_memcpy_gate] T2 绿(dev={d}, N={N})");
}

// ============================================================================
// T3:★ 真实 CE 形态 —— **跨卡 peer 块**作搬运源(源物理在**对端显存**)
// ============================================================================
//
// T1/T2 的源与目的都在**同一张卡**(普通块)⇒ 只证明了"图内设备内 D2D"这一族。
// 而 CE 刀的实际形状是 `copy_block_at(&peer_alias, …, &inbox, …)`:**源是 VMM
// 映射到对端显存的别名块** ⇒ 这趟搬运是**跨设备 DMA**(走 P2P)。图节点在实例化
// 时要把它烘死,回放时还必须重新解析 —— 本测试就是它的判据。
//
// 两条判据缺一不可:①第一次回放读到初始值;②**A 侧改数据后**再回放读到新值
// (只有"回放真执行了跨卡读"才能满足② —— 与 T1b 同型,这里是跨卡版)。

fn uuid_of(ordinal: usize) -> String {
    owl_cuda::ffi::device_rows()
        .expect("枚举设备")
        .into_iter()
        .find(|r| r.ordinal == ordinal)
        .map(|r| r.uuid)
        .unwrap_or_else(|| format!("GPU-ord{ordinal}"))
}

#[tokio::test]
async fn t3_captured_peer_memcpy_cross_device() {
    let (Some(oa), Some(ob)) = (
        std::env::var("OWL_PEER_TEST_DEV_A").ok().and_then(|v| v.parse().ok()),
        std::env::var("OWL_PEER_TEST_DEV_B").ok().and_then(|v| v.parse().ok()),
    ) else {
        eprintln!("skip T3:需 OWL_PEER_TEST_DEV_A / OWL_PEER_TEST_DEV_B(两张可 P2P 的空闲卡)");
        return;
    };
    assert_ne!(oa, ob, "A/B 必须是两张不同卡");
    const N: usize = 16384; // 64 KiB f32(≥ 16B 对齐 + 多 block 路径)
    const SZ: usize = 2 << 20;
    let x = le_f32(&vec![1.5f32; N]);
    let y = le_f32(&vec![7.25f32; N]);

    let mut a = assemble(oa, "t3a");
    let mut b = assemble(ob, "t3b");
    let (ua, ub) = (uuid_of(oa), uuid_of(ob));
    let (own_a, cred_a) = a.peer_alloc(SZ, "t3:own-a").await.expect("A PeerAlloc");
    let alias_a_on_b = b.peer_import(cred_a, SZ, &ua, "t3:alias-a@b").await.expect("B 导入 A");

    // A 侧写 X(htod 窗外;peer 块自身不能 htod ⇒ 用一块普通源 + 设备内拷过去)
    let src_x = owl_iface::htod!(a, Dtype::F32, &Shape::from(vec![N]), &x)
        .await
        .expect("A htod x")
        .view(Dtype::F32, 1);
    a.copy_block_at(&src_x, 0, &own_a, 0, N * 4).await.expect("A 写自己的 peer 块");
    a.sync().await.expect("A sync");

    let inbox = owl_iface::alloc_uninit!(b, Dtype::F32, N).await.expect("B alloc inbox").view(Dtype::F32, 1);
    let out = owl_iface::alloc_uninit!(b, Dtype::F32, N).await.expect("B alloc out").view(Dtype::F32, 1);

    // ── B 侧捕获窗:[跨卡 CE 拷(对端块 → 本端 inbox), 消费核] ────────
    b.graph_begin().await.expect("graph_begin");
    b.copy_block_at(&alias_a_on_b, 0, &inbox, 0, N * 4).await.expect("窗内跨卡 CopyBlock");
    b.launch(add_msg(&inbox, &inbox, &out, N)).await.expect("窗内消费核");
    let gid = b.graph_end().await.expect("graph_end");
    let _ = ub;

    // ── 判据①:回放读到 X(2×1.5 = 3.0)
    b.graph_launch(gid).await.expect("replay#1");
    let g1 = read_f32(&mut b, &out, 1).await;
    assert_eq!(
        g1[0], 3.0,
        "T3 红:跨卡 peer 搬运未生效(out[0]={};2×1.5=3.0 才对;0 = 拷没执行)。\
         dev A={oa} B={ob}",
        g1[0]
    );

    // ── 判据②(决定性):A 侧改数据 → 再回放 → 必须读到 Y(2×7.25 = 14.5)
    let src_y = owl_iface::htod!(a, Dtype::F32, &Shape::from(vec![N]), &y)
        .await
        .expect("A htod y")
        .view(Dtype::F32, 1);
    a.copy_block_at(&src_y, 0, &own_a, 0, N * 4).await.expect("A 改写 peer 块");
    a.sync().await.expect("A sync y");
    b.graph_launch(gid).await.expect("replay#2");
    let g2 = read_f32(&mut b, &out, 1).await;
    assert_eq!(
        g2[0], 14.5,
        "T3 红:第二次回放读到的是**旧值**(out[0]={};2×7.25=14.5 才对)⇒ 跨卡图节点\
         要么没进图、要么回放没真执行。dev A={oa} B={ob}",
        g2[0]
    );

    // ⚠️ 顺序:图引用租约会把**烘进图的块**(含跨卡 alias)钉住 ⇒
    // 必先 `graph_destroy` 再释放(这正是租约机制在起作用的现场证据)。
    b.graph_destroy(gid).await.expect("graph_destroy");
    b.free(&[inbox.id, out.id]).await.expect("B free");
    a.free(&[src_x.id, src_y.id]).await.expect("A free");
    b.peer_release(vec![alias_a_on_b.id]).await.expect("B peer_release");
    a.peer_release(vec![own_a.id]).await.expect("A peer_release");
    eprintln!("[graph_memcpy_gate] T3 绿(跨卡 peer 搬运,回放 2 次读到最新值;dev A={oa} B={ob})");
}

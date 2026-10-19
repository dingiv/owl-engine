//! **graph_multistream_gate —— 图内多流(fork/join)的"捕获 / 回放 / 并行"门**。
//!
//! 立案出处:`docs/arch/TP2-施工设计.md` §27.2 C3(b)/§27.5 B1(微批重叠的硬前置)。
//!
//! # 为什么需要它
//!
//! 本引擎的图**一直**是单流的(`graph_begin` 恒绑 COMPUTE,`wait_stream_on_tail`
//! 在捕获期响亮拒)。而"把 AR 藏进计算"的唯一结构 = **两条独立链拴进同一张图**
//! (两个 token 微批各一条)⇒ 必须先证明:
//!
//! | 判据 | 做法 | 能区分什么 |
//! |---|---|---|
//! | **M1 多流能捕获** | 窗内 `Launch(COMPUTE)` → `CaptureRoute(AUX)` → `Launch(AUX)` → `CaptureRoute(COMPUTE)` → `Launch(COMPUTE)` | `DISABLE_TIMING` 事件 + fork/join 边是否真的把第二条流**卷入同一张图** |
//! | **M2 join 边保序** | 第三条核**读** AUX 上算出的结果 | 没有 join 边 ⇒ 读到旧值/垃圾 |
//! | **M3 回放真执行两条链** | 改输入后再回放,数值跟着变 | "只在捕获期跑了一次"的假绿(与 `graph_memcpy_gate` T1b 同型) |
//! | **M4 非捕获期调用响亮拒** | 窗外的 `capture_route_to` | 该口**只在捕获期**合法(负例哨兵) |
//!
//! # 运行
//!
//! ```bash
//! OWL_GRAPH_MEMCPY_DEV=2 cargo test -p owl-cuda --test graph_multistream_gate -- --nocapture
//! ```
//! 未设 `OWL_GRAPH_MEMCPY_DEV` ⇒ 回落 `OWL_TEST_DEVICE`(默认 0)。

use owl_cuda::{
    DeviceClient as _, DeviceSelector, Dtype, GpuClient, GpuServer, Shape, STREAM_COMPUTE_AUX,
};
use owl_iface::contract::{Arg, Bytes, KernelSource, LaunchMsg};

// ============================================================================
// 组装 / 工具(与 graph_memcpy_gate 同家法)
// ============================================================================

fn assemble(ordinal: usize, tag: &str) -> GpuClient {
    let (tx, rx) = std::sync::mpsc::channel::<owl_cuda::Command>();
    let server = GpuServer::new(rx, DeviceSelector::Ordinal(ordinal), None);
    let client = GpuClient::new(tx);
    std::thread::Builder::new()
        .name(format!("owl-gpu-mstream-{tag}"))
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

const ADD_KERNEL: &str = "owl_add_f32";
fn msg_of(name: &str, args: Vec<Arg>, n: usize) -> LaunchMsg {
    LaunchMsg {
        kernel: KernelSource {
            name: name.into(),
            source: owl_kernels::ops::source(name).to_string(),
        },
        args,
        grid: ((n as u32).div_ceil(256), 1, 1),
        block: (256, 1, 1),
        shared_mem: 0,
        out_elems: n,
    }
}

/// `out = a + b`(真资产核;`owl_add_f32` 声明槽序 = [a, b, out, n])
fn add_msg(a: &Bytes, b: &Bytes, out: &Bytes, n: usize) -> LaunchMsg {
    msg_of(
        ADD_KERNEL,
        vec![
            Arg::block(a.id, a.dtype, a.rank),
            Arg::block(b.id, b.dtype, b.rank),
            Arg::block(out.id, out.dtype, out.rank),
            Arg::U64(n as u64),
        ],
        n,
    )
}

async fn read_f32(c: &mut GpuClient, blk: &Bytes, n: usize) -> Vec<f32> {
    c.sync().await.expect("sync(取数前)");
    let mut buf = vec![0u8; n * 4];
    c.dtoh(blk, &mut buf).await.expect("dtoh");
    de_f32(&buf)
}

// ============================================================================
// M1–M3:多流捕获 + join 保序 + 回放真执行
// ============================================================================

#[tokio::test]
async fn m1_multistream_capture_orders_and_replays() {
    let d = dev();
    let mut c = assemble(d, "m1");
    const N: usize = 8192;

    // 输入:两条链各自的源(A 链与 B 链是**独立**的 —— 微批并行的形状)
    let a_src = owl_iface::htod!(c, Dtype::F32, &Shape::from(vec![N]), &le_f32(&vec![1.0f32; N]))
        .await
        .expect("htod a_src")
        .view(Dtype::F32, 1);
    let b_src = owl_iface::htod!(c, Dtype::F32, &Shape::from(vec![N]), &le_f32(&vec![2.0f32; N]))
        .await
        .expect("htod b_src")
        .view(Dtype::F32, 1);
    let a_mid = owl_iface::alloc_uninit!(c, Dtype::F32, N).await.expect("a_mid").view(Dtype::F32, 1);
    let b_mid = owl_iface::alloc_uninit!(c, Dtype::F32, N).await.expect("b_mid").view(Dtype::F32, 1);
    let out = owl_iface::alloc_uninit!(c, Dtype::F32, N).await.expect("out").view(Dtype::F32, 1);
    c.sync().await.expect("sync setup");

    // ── 捕获窗:COMPUTE 链 + fork 到 AUX 的第二条链 + join 回来 ──────
    c.graph_begin().await.expect("graph_begin");
    c.launch(add_msg(&a_src, &a_src, &a_mid, N)).await.expect("A 链 k1(COMPUTE)");
    c.capture_route_to(STREAM_COMPUTE_AUX as u8)
        .await
        .expect("fork 到 AUX(该口只在捕获期合法)");
    c.launch(add_msg(&b_src, &b_src, &b_mid, N)).await.expect("B 链 k2(AUX):b_mid = 2·b_src");
    c.capture_join().await.expect("join 回 COMPUTE(主链等支链)");
    // k3 读**两条链**的结果 —— 没有 join 边这里就会读到未写的 b_mid
    c.launch(add_msg(&a_mid, &b_mid, &out, N)).await.expect("k3(COMPUTE,读两链)");
    let gid = c.graph_end().await.expect("graph_end");

    // ── M1+M2:回放 → a_mid=2.0, b_mid=102.0, out=104.0 ────────────
    c.graph_launch(gid).await.expect("replay#1");
    let g = read_f32(&mut c, &out, 1).await;
    assert_eq!(
        g[0], 6.0,
        "M1/M2 红:out[0]={}(期望 6.0 = 2·1.0 + 2·2.0)。0 = 两链都没跑;2.0 = 只有 COMPUTE 链跑了\
         (fork 没被卷入);4.0 = join 边缺 ⇒ k3 读到未写的 b_mid。dev={d}",
        g[0]
    );

    // ── M3:改 AUX 链的输入 → 再回放 → 数值必须跟着变(回放真执行两条链)──
    let b_src2 = owl_iface::htod!(c, Dtype::F32, &Shape::from(vec![N]), &le_f32(&vec![5.0f32; N]))
        .await
        .expect("htod b_src2")
        .view(Dtype::F32, 1);
    c.copy_block_at(&b_src2, 0, &b_src, 0, N * 4).await.expect("改写 b_src");
    c.sync().await.expect("sync 改写");
    c.graph_launch(gid).await.expect("replay#2");
    let g2 = read_f32(&mut c, &out, 1).await;
    assert_eq!(
        g2[0], 12.0,
        "M3 红:第二次回放未反映新输入(out[0]={},期望 12.0 = 2·1.0 + 2·5.0)⇒ 某条链只在捕获期\
         跑过一次、没进图。dev={d}",
        g2[0]
    );

    c.graph_destroy(gid).await.expect("graph_destroy");
    c.free(&[a_src.id, b_src.id, b_src2.id, a_mid.id, b_mid.id, out.id])
        .await
        .expect("free");
    eprintln!("[multistream] M1/M2/M3 绿(dev={d}, N={N}):fork 卷入 ✓ join 保序 ✓ 回放真执行两条链 ✓");
}

// ============================================================================
// M4:非捕获期调用响亮拒(负例哨兵)
// ============================================================================

#[tokio::test]
async fn m4_route_outside_capture_is_loud() {
    let d = dev();
    let mut c = assemble(d, "m4");
    let e = c
        .capture_route_to(STREAM_COMPUTE_AUX as u8)
        .await
        .expect_err("非捕获期的 capture_route_to 必须响亮拒(否则静默改路由 = 灾难)");
    let msg = format!("{e}");
    assert!(
        msg.contains("不在捕获期"),
        "拒收消息须点名\"不在捕获期\";实得:{msg}"
    );
    eprintln!("[multistream] M4 绿(dev={d}):窗外调用 → {msg}");
}

// ============================================================================
// M5:★ 双流并行的**吞吐**对照(图内并行到底成不成立)
// ============================================================================
//
// M1 只证明了**正确性**(fork/join 边成立)。但"两链都在图里"≠"两链并行跑"。
// 本档把这件事量清楚 —— 它决定"微批重叠"这条路是否成立:
//
//   · `OWL_MS_CHAIN=512`(单流) ⇒ COMPUTE 上 512 次 add
//   · 再加 `OWL_MS_DUAL=1`      ⇒ COMPUTE 512 次 **+** AUX 512 次
//   · 判据:`dual ≈ single` ⇒ **真并行**(图上标了并行、硬件也真的并行);
//           `dual ≈ 2×single` ⇒ 图上标了并行、执行却**串行**。
#[tokio::test]
async fn m5_dual_stream_throughput() {
    let d = dev();
    let chain: u32 = std::env::var("OWL_MS_CHAIN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(512);
    let dual = std::env::var("OWL_MS_DUAL").map(|v| v != "0" && !v.is_empty()).unwrap_or(false);
    let k: u32 = std::env::var("OWL_MS_K").ok().and_then(|v| v.parse().ok()).unwrap_or(20);
    let mut c = assemble(d, "m5");
    const N: usize = 1 << 18; // 1 MiB f32/链
    let one = owl_iface::htod!(c, Dtype::F32, &Shape::from(vec![N]), &le_f32(&vec![1.0f32; N]))
        .await
        .expect("htod base")
        .view(Dtype::F32, 1);
    let sink = owl_iface::alloc_uninit!(c, Dtype::F32, N).await.expect("sink").view(Dtype::F32, 1);
    c.sync().await.expect("sync setup");

    c.graph_begin().await.expect("graph_begin");
    // ★ **fork 必须在链的起点**(2026-10-18 M5 教训):边记在链尾 = 把两条链拴成串行。
    if dual {
        c.capture_route_to(STREAM_COMPUTE_AUX as u8).await.expect("fork→AUX(链起点)");
        for _ in 0..chain {
            c.launch(add_msg(&one, &one, &sink, N)).await.expect("AUX 链");
        }
        // 切回 origin = **纯切换(不建边)** —— 建边就会让 COMPUTE 链等 AUX 链
        c.capture_route_to(1u8).await.expect("切回 COMPUTE");
    }
    for _ in 0..chain {
        c.launch(add_msg(&one, &one, &sink, N)).await.expect("COMPUTE 链");
    }
    // ★ **末尾 join**(2026-10-18 §39):fork 出去的支流**必须有工作就要 join**,
    // 否则驱动以 UNJOINED 整窗拒。必须在**两条链都发完之后**记 ——
    // `StreamWaitEvent` 只影响之后发射的工作 ⇒ COMPUTE 链不会被挡在 AUX 链之后。
    if dual {
        c.capture_join().await.expect("join(origin 尾部等 AUX)");
    }
    let gid = c.graph_end().await.expect("graph_end");

    for _ in 0..3 {
        c.graph_launch(gid).await.expect("warm");
    }
    c.sync().await.expect("warm sync");
    let t0 = std::time::Instant::now();
    for _ in 0..k {
        c.graph_launch(gid).await.expect("replay");
    }
    c.sync().await.expect("sync");
    let ms = t0.elapsed().as_secs_f64() * 1e3 / f64::from(k);
    let mut got = vec![0u8; 16];
    c.dtoh(&sink, &mut got).await.expect("dtoh");
    assert_eq!(
        de_f32(&got)[0], 2.0,
        "数值自证:链没跑(1.0 + 1.0 应为 2.0)"
    );
    eprintln!(
        "[multistream] M5: chain={chain} dual={dual} ⇒ **{ms:.3} ms/回放**\
         (single vs dual 的比 = 并行度;≈1.0 真并行 / ≈2.0 串行)"
    );
    c.graph_destroy(gid).await.expect("graph_destroy");
    c.free(&[one.id, sink.id]).await.expect("free");
}

// ============================================================================
// M6 已迁出(2026-10-18 §38)
// ============================================================================
//
// 本文件原加过一格 `m6_engine_disjoint_parallelism`([CE 链 ‖ SM 链]),但
// **`cudaMemcpyAsync` 在非 origin 的捕获流上会让 `cuStreamEndCapture` 报
// `CUDA_ERROR_STREAM_CAPTURE_UNJOINED`**(实测;§32 的 `graph_memcpy_gate` T1-T3
// 验的是 origin 流上的图内搬运 ⇒ 不覆盖这一格)。
//
// ⇒ 真形状的重叠判据搬到**有 P2P 的门**里做:`ar_stress_gate::push_graph_overlap_probe`
// ([PCIe 写核 ‖ 本地 RMW 核];两条链的 header 互不重叠 ⇒ 一笔就够)。
// **定谳见 `TP2-施工设计.md` §38:图内跨流 = 完全串行(0.99-1.00),裸 CUDA 图同样
// 如此(同质对照 1.999)⇒ 不是 owl 的 bug。**

// ============================================================================
// M6:★ **图内搬运的 join 语义**(2026-10-18;§38 的 ⚠️ 归因复核)
// ============================================================================
//
// §38 的注解说"`cudaMemcpyAsync` 在非 origin 捕获流上会 UNJOINED"。**这条归因
// 需要复核** —— 因为同一批代码里还有一个更可疑的东西:`capture_join()` 的**调用时机**。
//
// `capture_join()` 的实现:`cur = c.stream; if ptr_eq(cur, origin) → **直接返回**`。
// 所以"先 `capture_route_to(origin)` 再 `capture_join()`" = **边根本没建**(no-op)。
//
// 本档用四种形状把两个嫌疑分开:
//
// | 形状 | AUX 链 | 收尾 | 预期 |
// |---|---|---|---|
// | A | K 个 **核** | 无 join | ？(ar_stress 的重叠探针就是这个形状,**跑通了**)|
// | B | K 个 **CE 拷贝** | 无 join | ？(graph_multistream_gate 的旧 M6 = 这个,UNJOINED)|
// | C | K 个 **CE 拷贝** | **在 AUX 上 join**(切回 origin 之前) | 应 OK |
// | D | K 个 **CE 拷贝** + COMPUTE 上 K 个核 | 无 join | ？ |
//
// 运行:`OWL_PEER_TEST_DEV_A=2 cargo test -p owl-cuda --test graph_multistream_gate \
//        m6_capture_join_semantics -- --nocapture`
#[tokio::test]
async fn m6_capture_join_semantics() {
    let d = dev();
    let k: u32 = std::env::var("OWL_MS6_K").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    let mut c = assemble(d, "m6");
    const N: usize = 1 << 16; // 256 KiB
    let src = owl_iface::htod!(c, Dtype::F32, &Shape::from(vec![N]), &le_f32(&vec![1.0f32; N]))
        .await
        .expect("htod")
        .view(Dtype::F32, 1);
    let dst = owl_iface::alloc_uninit!(c, Dtype::F32, N).await.expect("dst").view(Dtype::F32, 1);
    c.sync().await.expect("sync");

    // 一形状 = 捕一张图;返回 Ok/Err(不 panic —— 本档就是来分类错误的)
    async fn shape(
        c: &mut GpuClient,
        src: &Bytes,
        dst: &Bytes,
        n: usize,
        k: u32,
        aux_copy: bool,
        aux_kernel: bool,
        comp_kernel: bool,
        join_on_aux: bool,
    ) -> Result<(), String> {
        c.graph_begin().await.map_err(|e| format!("graph_begin: {e}"))?;
        c.capture_route_to(STREAM_COMPUTE_AUX as u8).await.map_err(|e| format!("fork: {e}"))?;
        for _ in 0..k {
            if aux_copy {
                c.copy_block_at(src, 0, dst, 0, n * 4).await.map_err(|e| format!("AUX copy: {e}"))?;
            }
            if aux_kernel {
                c.launch(add_msg(src, src, dst, n)).await.map_err(|e| format!("AUX kern: {e}"))?;
            }
        }
        if join_on_aux {
            // ★ **在 AUX 上 join**(切回 origin 之前)⇒ 边真的建起来
            c.capture_join().await.map_err(|e| format!("join(AUX): {e}"))?;
        } else {
            c.capture_route_to(1u8).await.map_err(|e| format!("switch: {e}"))?;
        }
        for _ in 0..k {
            if comp_kernel {
                c.launch(add_msg(src, src, dst, n)).await.map_err(|e| format!("COMP: {e}"))?;
            }
        }
        // ⚠️ **不补 join**:切回 origin 之后 `capture_join()` 已按纪律改成**响亮拒**
        // (那条边根本不会建)。本档就是来量"没 join 会怎样"的 ⇒ 让驱动自己判。
        match c.graph_end().await {
            Ok(gid) => {
                c.graph_destroy(gid).await.ok();
                Ok(())
            }
            Err(e) => Err(format!("graph_end: {e}")),
        }
    }

    let cases: [(&str, bool, bool, bool, bool); 4] = [
        ("A  AUX=核   · 无 join     ", false, true, false, false),
        ("B  AUX=CE   · 无 join     ", true, false, false, false),
        ("C  AUX=CE   · **在 AUX join**", true, false, false, true),
        ("D  AUX=CE + COMP=核 · 无 join", true, false, true, false),
    ];
    for (tag, ac, ak, ck, ja) in cases {
        let r = shape(&mut c, &src, &dst, N, k, ac, ak, ck, ja).await;
        match r {
            Ok(()) => eprintln!("[multistream] M6 {tag} ⇒ **✓ 捕获成功**"),
            Err(e) => eprintln!("[multistream] M6 {tag} ⇒ **✗ {e}**"),
        }
    }
    // ── 形状 E:在 **origin** 上调 join ⇒ 必须**响亮拒**(不是静默 no-op)──
    // 动机(2026-10-18 现场):"先切回 origin 再 join" = 边根本没建,而旧实现的
    // no-op 让错误一路飘到 `cuStreamEndCapture` 才以 UNJOINED 暴露(报错点离现场太远)。
    {
        c.graph_begin().await.expect("graph_begin(E)");
        c.launch(add_msg(&src, &src, &dst, N)).await.expect("E 链");
        match c.capture_join().await {
            Ok(()) => eprintln!("[multistream] M6 E  在 origin 上 join        ⇒ **✗ 静默 no-op(应当拒)**"),
            Err(e) => eprintln!("[multistream] M6 E  在 origin 上 join        ⇒ **✓ 响亮拒**:{e}"),
        }
        c.graph_end().await.expect("graph_end(E)");
    }
    c.free(&[src.id, dst.id]).await.expect("free");
}

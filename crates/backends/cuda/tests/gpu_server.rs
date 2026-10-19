//! owl-cuda gpu_server 看护测试(回归防线)。
//!
//! **只依赖 owl-cuda 公开面**(线格式 Command/LaunchMsg/Arg/Bytes 经根
//! re-export;不 import 上层 owl-models —— 上层是客户,server 不依赖客户)。
//!
//! 结构:一个大测试,九个步骤,只建一个 server,顺序执行。
//!
//! 步骤清单:
//! 1.  alloc 零化(N4 memset 回归哨)
//! 2.  htod/dtoh 往返
//! 3.  手工 LaunchMsg 发射(kernel + 槽序契约)
//! 4.  launch fire-and-forget 后 sync 栅栏收割(大数组)
//! 5.  图捕获/重放(捕获期 Alloc = carve + 图内 memset 节点)
//! 6.  图护栏四连(捕获期 Dtoh 拒 / 嵌套 begin 拒 / 空窗哨兵③ / 未知图 id)
//! 7.  close 语义(ServerClosed)
//! 8.  selector 错 UUID → boot 失败
//! 9.  图引用租约(捕进图的 Owned 块 free 拒收;自由块照常回收)
//! 10. 捕获窗内 Free + slab 复用(M1 峰值口径:6×16MiB 总和 > 64MiB slab)
//! 11. abort / destroy(窗内块整批作废;destroy 后解锁租约 + 图不可回放)
//! 12. 家族面捕获入境守卫(GDN chunked 真身;观察档放行 / OWL_CAP_STRICT 拒)
//!
//! 运行:OWL_TEST_DEVICE=<空闲卡> cargo test -p owl-cuda --test gpu_server
//! 需要一张空闲 GPU。

use owl_cuda::{
    test_device_ordinal, Arg, Bytes, Command, DeviceClient as _, DeviceSelector, Dtype, GpuClient,
    GpuServer, KernelSource, LaunchMsg, ModelError, Shape,
};

// ⚠️ 本套件原用**内联 `.cu` 源**(`owl_scale_f32`)—— NVRTC 退役后"零运行期编译"
// 硬门当场拒(自那次退役起本测试就失效)。2026-10-15 改为**真资产核**
// `owl_add_f32`(声明面已建表 + 构建期 cubin 在场):契约面覆盖不变,
// 且顺带把 L3 事实面(dtype/rank)一起过门。
const ADD_KERNEL: &str = "owl_add_f32";

// ============================================================================
// 组装与断言辅助(纯 owl_cuda 面)
// ============================================================================

/// 手工组装:管道外部创建,server/client 各拿一端;server 跑在专职线程
fn assemble() -> GpuClient {
    let ordinal = test_device_ordinal();
    let (tx, rx) = std::sync::mpsc::channel::<Command>();
    let server = GpuServer::new(rx, DeviceSelector::Ordinal(ordinal), None);
    let client = GpuClient::new(tx);
    std::thread::Builder::new()
        .name("owl-gpu-test".into())
        .spawn(move || server.run().expect("server run 异常退出"))
        .expect("起 server 线程失败");
    client
}

fn le_f32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn le_to_f32(buf: &[u8]) -> Vec<f32> {
    buf.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

async fn harvest(client: &mut GpuClient, out: &Bytes, n: usize) -> Vec<f32> {
    client.sync().await.expect("sync 栅栏");
    let mut buf = vec![0u8; n * 4];
    client.dtoh(out, &mut buf).await.expect("dtoh");
    le_to_f32(&buf)
}

fn assert_all_close(got: &[f32], want: f32, ctx: &str) {
    for (i, v) in got.iter().enumerate() {
        assert!(
            (v - want).abs() < 1e-6,
            "{ctx}[{i}] = {v}, 期望 {want}(全部 = {got:?})"
        );
    }
}

/// 手工装配 scale 发射(槽序契约:标量按声明序,输出块固定最后)
/// `owl_add_f32`(a+b→out)发射消息(真资产核;声明槽序 = [a,b,out,n])
fn add_msg(a: &Bytes, b: &Bytes, out: &Bytes, n: usize) -> LaunchMsg {
    LaunchMsg {
        // native 资产核:source 字段 = **资产 id**(server 按 (资产, 名) 查构建期 cubin)
        kernel: KernelSource { name: ADD_KERNEL.into(), source: owl_kernels::ops::source(ADD_KERNEL).to_string() },
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

// ============================================================================
// 步骤实现
// ============================================================================

/// 步骤 1:alloc 零化(N4 图外 memset 回归哨)
async fn step_alloc_zeroed(client: &mut GpuClient) {
    let out = owl_iface::client_alloc!(client, Dtype::F32, 8).await.expect("alloc").view(Dtype::F32, 1);
    let got = harvest(client, &out, 8).await;
    assert!(
        got.iter().all(|&v| v == 0.0),
        "alloc 应零化(N4 回归),实得 {got:?}"
    );
}

/// 步骤 2:htod/dtoh 往返
async fn step_htod_dtoh_roundtrip(client: &mut GpuClient) {
    let src = [1.5f32, -2.25, 3.125, 1e-6, 42.0];
    let block = owl_iface::htod!(client, Dtype::F32, &Shape::from(vec![src.len()]), &le_f32(&src))
        .await
        .expect("htod");
    let got = harvest(client, &block, src.len()).await;
    assert_eq!(got, src, "htod/dtoh 往返失真: {got:?} vs {src:?}");
}

/// 步骤 3:手工 LaunchMsg 发射(kernel + 槽序契约)
async fn step_launch_scale(client: &mut GpuClient) {
    let xs = [2.0f32; 4];
    let x = owl_iface::htod!(client, Dtype::F32, &Shape::from(vec![4]), &le_f32(&xs))
        .await
        .expect("htod");
    let out = owl_iface::client_alloc!(client, Dtype::F32, 4).await.expect("alloc").view(Dtype::F32, 1);
    let ys = [4.0f32; 4];
    let y = owl_iface::htod!(client, Dtype::F32, &Shape::from(vec![4]), &le_f32(&ys))
        .await
        .expect("htod y");
    client.launch(add_msg(&x, &y, &out, 4)).await.expect("launch");
    let got = harvest(client, &out, 4).await;
    assert_all_close(&got, 6.0, "add 槽序(2+4)");
}

/// 步骤 4:launch 后 sync 栅栏收割(大数组)
async fn step_sync_is_barrier(client: &mut GpuClient) {
    let src: Vec<f32> = (0..4096).map(|i| i as f32 * 0.5).collect();
    let x = owl_iface::htod!(client, Dtype::F32, &Shape::from(vec![src.len()]), &le_f32(&src))
        .await
        .expect("htod");
    let out = owl_iface::client_alloc!(client, Dtype::F32, src.len()).await.expect("alloc").view(Dtype::F32, 1);
    let zeros = vec![0.0f32; src.len()];
    let y = owl_iface::htod!(client, Dtype::F32, &Shape::from(vec![src.len()]), &le_f32(&zeros))
        .await
        .expect("htod y");
    client.launch(add_msg(&x, &y, &out, src.len())).await.expect("launch");
    let got = harvest(client, &out, src.len()).await;
    for (i, (g, w)) in got.iter().zip(src.iter()).enumerate() {
        assert_eq!(g, w, "sync 栅栏后 [{i}] = {g}, 期望 {w}");
    }
}

/// 步骤 5:图捕获/重放
/// 捕获期 Alloc = carve + 图内 memset 节点(重放重清零);scale 保发射数 > 0
async fn step_graph_capture_and_replay(client: &mut GpuClient, x: &Bytes) {
    // 捕获窗只放行 Launch/Alloc/MemsetZero/GraphEnd/Close ⇒ 输入须在 begin 之前上卡
    let ones = [1.0f32; 4];
    let y = owl_iface::htod!(client, Dtype::F32, &Shape::from(vec![4]), &le_f32(&ones))
        .await
        .expect("htod y");
    client.graph_begin().await.expect("graph_begin");

    // 捕获期 Alloc(切片块 + memset 进图)+ scale 发射
    let zeros = owl_iface::client_alloc!(client, Dtype::F32, 4).await.expect("captured alloc").view(Dtype::F32, 1);
    let out = owl_iface::client_alloc!(client, Dtype::F32, 4).await.expect("captured alloc").view(Dtype::F32, 1);
    client.launch(add_msg(x, &y, &out, 4)).await.expect("launch");

    let gid = client.graph_end().await.expect("graph_end");

    // 重放 ×2:memset 节点应重清零 zeros 块;add 输出落同一批块
    for i in 0..2 {
        client.graph_launch(gid).await.expect("graph_launch");
        let got_zeros = harvest(client, &zeros, 4).await;
        assert!(
            got_zeros.iter().all(|&v| v == 0.0),
            "图内 memset 重放应清零,实得 {got_zeros:?}"
        );
        let got = harvest(client, &out, 4).await;
        // x=[1.0;4] + y=[1.0;4] = 2.0(旧内联 scale 核口径是 x*3=3.0)
        assert_all_close(&got, 2.0, &format!("replay #{i}"));
    }
}

/// **步骤 5b:piecewise 骨架 —— 两段捕获 + 段间 eager 写 + 两次回放**
///
/// 这是 piecewise 设计(`docs/piecewise-断点图-设计与实现.md`)在 CUDA 面
/// 依赖的最小机制,先在此钉死(引擎侧 `PiecewisePlan` 复用同一批语义):
///   seg0(捕获) → **窗外 eager 算子**(断点) → seg1(捕获) → 回放序列 ×2
/// 判据:① 段间块**地址接力**正确(seg1 读的正是 eager 写的块);
///       ② 回放**可重复**(同序列两次结果一致 ⇒ 无跨段地址复用/竞态)。
async fn step_piecewise_two_segments(client: &mut GpuClient, x: &Bytes) {
    // 输入与跨段"钉块"都在窗外先物化(begin 前)——
    // 对应设计里"pin 块由 plan 捕获前分配、两期同址"那条不变量
    let one = owl_iface::htod!(client, Dtype::F32, &Shape::from(vec![4]), &le_f32(&[1.0f32; 4]))
        .await
        .expect("htod one");
    let mid = owl_iface::client_alloc!(client, Dtype::F32, 4).await.expect("pin 块(mid)").view(Dtype::F32, 1);
    let tail = owl_iface::client_alloc!(client, Dtype::F32, 4).await.expect("pin 块(tail)").view(Dtype::F32, 1);

    // ── 段0:out0 = x + one ──
    let out0 = owl_iface::client_alloc!(client, Dtype::F32, 4).await.expect("carve out0").view(Dtype::F32, 1);
    client.graph_begin().await.expect("seg0 begin");
    client.launch(add_msg(x, &one, &out0, 4)).await.expect("seg0 launch");
    let g0 = client.graph_end().await.expect("seg0 end");

    // ── 断点(窗外 eager):mid = out0 + out0(调用方自行求值,写进钉块)──
    let eager_mid = || add_msg(&out0, &out0, &mid, 4);
    client.launch(eager_mid()).await.expect("breakpoint eager");

    // ── 段1:tail = mid + one ──
    client.graph_begin().await.expect("seg1 begin");
    client.launch(add_msg(&mid, &one, &tail, 4)).await.expect("seg1 launch");
    let g1 = client.graph_end().await.expect("seg1 end");

    // ── 回放序列 ×2:seg0 → 断点重放 → seg1 ──
    for i in 0..2 {
        client.graph_launch(g0).await.expect("replay g0");
        client.launch(eager_mid()).await.expect("replay breakpoint");
        client.graph_launch(g1).await.expect("replay g1");
        let got = harvest(client, &tail, 4).await;
        // x=[1.0;4] → out0=2 → mid=4 → tail=5
        assert_all_close(&got, 5.0, &format!("piecewise 回放 #{i}"));
        let got_mid = harvest(client, &mid, 4).await;
        assert_all_close(&got_mid, 4.0, &format!("跨段钉块 mid #{i}"));
    }
}

/// 步骤 6:图护栏四连
async fn step_graph_guardrails(client: &mut GpuClient, x: &Bytes) {
    // a) graph_begin 后做 dtoh → Err
    client.graph_begin().await.expect("graph_begin");
    let r = client.dtoh(x, &mut [0u8; 16]).await;
    assert!(r.is_err(), "捕获期 dtoh 应被拒,实得 {r:?}");

    // b) 嵌套 graph_begin → Err
    let r = client.graph_begin().await;
    assert!(r.is_err(), "嵌套 graph_begin 应被拒,实得 {r:?}");

    // c) 零发射窗 graph_end → Err(哨兵③)
    let r = client.graph_end().await;
    assert!(r.is_err(), "空捕获窗 graph_end 应被拒(哨兵③),实得 {r:?}");

    // d) 未知图 id 重放 → Err
    let r = client.graph_launch(9999).await;
    assert!(r.is_err(), "未知图 id graph_launch 应被拒,实得 {r:?}");
}

/// 步骤 9:**图引用租约**(2026-10-15)—— 被图烘过的 `Owned` 块
/// `free` 拒收(仍在册/仍可发射/内容不变),自由块照常回收(死块结构化拒)。
///
/// 为什么值得一测:图烘的是**裸设备指针**;若被 free 的 `Owned` 块(池件/
/// 输入槽)被设备池原址重发给下一个 alloc,图即变"陈旧指针"—— **静默错值**。
/// 本步骤把"拒收 + 仍可用"与"自由块照常回收"两面都钉成断言。
async fn step_graph_ref_lease(client: &mut GpuClient) {
    let ones = [1.0f32; 4];
    // 图外分配/上卡 ⇒ Owned(可回收类;Carved 天然免疫,不在本例)
    let a = owl_iface::htod!(client, Dtype::F32, &Shape::from(vec![4]), &le_f32(&ones)).await.expect("htod a");
    let b = owl_iface::htod!(client, Dtype::F32, &Shape::from(vec![4]), &le_f32(&ones)).await.expect("htod b");
    let out = owl_iface::client_alloc!(client, Dtype::F32, 4).await.expect("alloc out").view(Dtype::F32, 1);

    // 捕获窗内解析三个块 ⇒ 全部入租约
    client.graph_begin().await.expect("graph_begin");
    client.launch(add_msg(&a, &b, &out, 4)).await.expect("launch");
    let gid = client.graph_end().await.expect("graph_end");
    client.graph_launch(gid).await.expect("graph_launch");
    let got = harvest(client, &out, 4).await;
    assert_all_close(&got, 2.0, "租约登记后回放");

    // (a) 释放租约块:静默拒收(不报错),块仍在册 —— 内容不变 + 仍可发射
    client
        .free(&[a.id, b.id, out.id])
        .await
        .expect("free(租约命中 = 拒收,应回报 Ok)");
    let got_a = harvest(client, &a, 4).await;
    assert_eq!(got_a, ones, "租约块应仍在册(内容不变),实得 {got_a:?}");
    let got_out = harvest(client, &out, 4).await;
    assert_all_close(&got_out, 2.0, "租约块仍可读(未被回收)");
    // 仍在册 ⇒ 仍可参与发射(若被回收,这里会因地址重发而读到别人内存)
    client.launch(add_msg(&a, &b, &out, 4)).await.expect("租约块仍可发射");
    let got2 = harvest(client, &out, 4).await;
    assert_all_close(&got2, 2.0, "租约后 eager 发射");

    // (b) 正例对照:未被任何图引用的块照常回收 ⇒ 之后再解析 = 结构化 DeadBlock
    let loose = owl_iface::htod!(client, Dtype::F32, &Shape::from(vec![4]), &le_f32(&ones)).await.expect("htod loose");
    client.free(&[loose.id]).await.expect("free(无租约应照常回收)");
    let r = client.dtoh(&loose, &mut [0u8; 16]).await;
    assert!(
        matches!(r, Err(ModelError::DeadBlock { .. })),
        "回收后块应结构化拒(DeadBlock),实得 {r:?}"
    );
}

/// 步骤 10:**捕获窗内 Free + slab 复用**(M1 峰值口径的真机判据)。
///
/// 手法:默认 slab 64MiB(未设 hint);窗内**交替 alloc/free** 共 6×16MiB
/// = 96MiB **总和** > 64MiB,而峰值活跃集只有 2×16MiB ⇒
/// - 有 arena 回收(本刀):`graph_end` 成功;
/// - 无回收(旧纯 bump):第 5 块即超 64MiB ⇒ `graph_end` 前就得 Err。
/// ⇒ 本步就是"容量 = 峰值活跃集而非总和"的集成判据。
async fn step_capture_window_reuse(client: &mut GpuClient) {
    const BLK: usize = 4 << 20; // 4 Mi 元素 × f32 = 16 MiB
    const N: usize = 6;
    let ones = [1.0f32; 4];
    let x = owl_iface::htod!(client, Dtype::F32, &Shape::from(vec![4]), &le_f32(&ones)).await.expect("htod x");
    let y = owl_iface::htod!(client, Dtype::F32, &Shape::from(vec![4]), &le_f32(&ones)).await.expect("htod y");
    client.graph_begin().await.expect("graph_begin");
    let mut live: Option<Bytes> = None;
    let mut total = 0usize;
    for _ in 0..N {
        let b = owl_iface::client_alloc!(client, Dtype::F32, BLK).await.expect("窗内 alloc(切 slab)");
        total += BLK * 4;
        if let Some(prev) = live.take() {
            // 交替释放上一块 ⇒ 活块数恒 ≤ 2(回 slab 内 arena,零 CUDA 调用)
            client.free(&[prev.id]).await.expect("窗内 free(回 arena)");
        }
        live = Some(b);
    }
    // 空窗哨兵③:发射数须 > 0
    let out = owl_iface::client_alloc!(client, Dtype::F32, 4).await.expect("alloc out").view(Dtype::F32, 1);
    client.launch(add_msg(&x, &y, &out, 4)).await.expect("launch");
    // **捕获窗内也能读块账**(全量记名 26-10-16):carve 出来的窗内块必须在册
    // 且带名 —— OOM 最常发生在窗内(slab 定尺失准),那刻的"谁占着"就靠这条。
    // ⚠️ `top` 要够大:窗内块多数是 KiB 级小块,渲染按字节降序取 top-N ⇒
    // top 小了会落进"…另 N 类"(本测试首版就这么假红了一次 —— 不是没记名,
    // 是没挤进前列);而"未命名块"计数与 top 无关,是更强的不变量。
    let rep = client.ledger_report(64).await.expect("窗内 ledger_report 应放行(纯读)");
    // 窗内那块的调用点已改用 `alloc!` ⇒ 名字是**编译器填的位置**
    // (`alloc@<文件>:<行>:<列>`)。断言只锁"形态 + 本文件",**不写死行号**
    // (行号会随编辑漂;写死它就是把脆弱判据请回来 —— 见 §8.2 教训)。
    assert!(
        rep.contains("alloc@") && rep.contains("gpu_server.rs"),
        "窗内 carve 块必须带名在册(top=64):{rep}"
    );
    assert!(
        !rep.contains("未命名块"),
        "窗内不得出现未命名块(与 top 无关的硬不变量):{rep}"
    );
    let gid = client.graph_end().await;
    assert!(
        gid.is_ok(),
        "总和 {total}B > 默认 slab 64MiB,峰值仅 2×16MiB ⇒ 有 arena 回收才应 Captured;实得 {gid:?}"
    );
    let gid = gid.unwrap();
    client.graph_launch(gid).await.expect("回放");
    let got = harvest(client, &out, 4).await;
    assert_all_close(&got, 2.0, "复用 slab 后回放数值");
}

/// 步骤 11:**abort / destroy 两把生命周期原语**。
/// - `graph_abort`:窗内出错 ⇒ 半成品图销毁 + 窗内自产块**整批作废**(再解析
///   = 结构化 `DeadBlock`);窗外调用 = 结构化拒。
/// - `graph_destroy`:图退休 ⇒ 图不可再回放 + 该图 slab 块作废 + **租约解除**
///   (被它钉住的 `Owned` 块随后可 free)。
async fn step_graph_abort_and_destroy(client: &mut GpuClient) {
    // ── abort ──
    let r = client.graph_abort().await;
    assert!(r.is_err(), "无捕获时 abort 应结构化拒,实得 {r:?}");
    let ones = [1.0f32; 4];
    let a = owl_iface::htod!(client, Dtype::F32, &Shape::from(vec![4]), &le_f32(&ones)).await.expect("htod a");
    let b = owl_iface::htod!(client, Dtype::F32, &Shape::from(vec![4]), &le_f32(&ones)).await.expect("htod b");
    client.graph_begin().await.expect("graph_begin");
    let carved = owl_iface::client_alloc!(client, Dtype::F32, 4).await.expect("窗内 alloc").view(Dtype::F32, 1);
    client.launch(add_msg(&a, &b, &carved, 4)).await.expect("launch");
    client.graph_abort().await.expect("abort 应成功");
    // 窗内自产块整批作废(不登记图 ⇒ 永不会被回放)
    let r = client.dtoh(&carved, &mut [0u8; 16]).await;
    assert!(matches!(r, Err(ModelError::DeadBlock { .. })), "abort 后 slab 块应作废,实得 {r:?}");
    // abort 后引擎仍可正常捕获(窗态已复位 —— 否则第二张图必炸)
    client.graph_begin().await.expect("abort 后再捕获");
    let out2 = owl_iface::client_alloc!(client, Dtype::F32, 4).await.expect("alloc").view(Dtype::F32, 1);
    client.launch(add_msg(&a, &b, &out2, 4)).await.expect("launch");
    let gid = client.graph_end().await.expect("abort 后 graph_end 应成功");

    // ── destroy ──
    let r = client.graph_destroy(9999).await;
    assert!(r.is_err(), "未知图 destroy 应拒,实得 {r:?}");
    client.graph_destroy(gid).await.expect("destroy 应成功");
    let r = client.graph_launch(gid).await;
    assert!(r.is_err(), "已销毁图不可回放,实得 {r:?}");
    let r = client.dtoh(&out2, &mut [0u8; 16]).await;
    assert!(matches!(r, Err(ModelError::DeadBlock { .. })), "destroy 后该图 slab 块应作废,实得 {r:?}");
    // 租约解除:a/b 曾被该图烘入 ⇒ destroy 后应可回收,再解析 = DeadBlock
    client.free(&[a.id, b.id]).await.expect("destroy 后应可 free(租约已解除)");
    let r = client.dtoh(&a, &mut [0u8; 16]).await;
    assert!(matches!(r, Err(ModelError::DeadBlock { .. })), "解钉后回收应生效,实得 {r:?}");

    // 窗内 destroy 结构化拒(先收窗)
    client.graph_begin().await.expect("begin");
    let r = client.graph_destroy(1).await;
    assert!(r.is_err(), "捕获窗内 destroy 应拒,实得 {r:?}");
    client.graph_abort().await.expect("收窗");
}

/// 步骤 12:**家族面捕获入境守卫**(设计 §4.3e)—— 本仓唯一的"家族真身"证据。
///
/// 真因:GDN chunked 家族的 `run()` 里会
/// ① `res.alloc`(scratch 槽 + `(t,nt)` 键的 meta 表)② `res.upload`(meta 表,
/// 源是**run 内现建的 `Vec<u8>`**,不是不变盒)。把它放进捕获窗:
/// - **观察档(缺省)**:放行 + 计数(`srv.count.fam_alloc_in_window` /
///   `fam_upload_in_window`)+ WARN —— 行为零变化,但图里已烙下 MEM_ALLOC
///   节点与"从栈上源拷"的 H2D 节点(后者**回放必错** ⇒ 本步骤故意不回放);
/// - **`OWL_CAP_STRICT=1`**:`alloc` 结构化拒 ⇒ 窗已脏 ⇒ 必须 `graph_abort`
///   收窗(顺带把 abort 在真失败路径上过一遍)。
///
/// 本步骤的断言按 `OWL_CAP_STRICT` 分支 ⇒ **两种档位各跑一次**才算完整。
async fn step_family_window_guard(client: &mut GpuClient) {
    use owl_kernels::client::gdn_chunked::{GdnChunkedCall, GdnShape};

    // 形状(t=8;kd=128 是家族律内核尺寸;nk=nv=1 让账长最小)
    let shape = GdnShape { t: 8, nv: 1, nk: 1, kd: 128 };
    let elems_qkv = shape.t as usize * shape.kd as usize; // nk=nv=1
    let ones_f16: Vec<f32> = vec![0.01; elems_qkv];
    let q = owl_iface::htod!(client, Dtype::F16, &Shape::from(vec![8, 1, 128]), &le_f16(&ones_f16)).await.expect("htod q");
    let k = owl_iface::htod!(client, Dtype::F16, &Shape::from(vec![8, 1, 128]), &le_f16(&ones_f16)).await.expect("htod k");
    let v = owl_iface::htod!(client, Dtype::F16, &Shape::from(vec![8, 1, 128]), &le_f16(&ones_f16)).await.expect("htod v");
    let beta = owl_iface::htod!(client, Dtype::F16, &Shape::from(vec![8, 1]), &le_f16(&[0.5; 8])).await.expect("htod beta");
    let gate = owl_iface::htod!(client, Dtype::F16, &Shape::from(vec![8, 1]), &le_f16(&[-0.01; 8])).await.expect("htod gate");
    let state = owl_iface::client_alloc!(client, Dtype::F32, 128 * 128).await.expect("alloc state").view(Dtype::F32, 4);
    let out = owl_iface::client_alloc!(client, Dtype::F16, elems_qkv).await.expect("alloc out").view(Dtype::F16, 3);
    // P-6b:序列表(cu|coff|idx)现造一块(本测只关心窗内 alloc/upload 守卫,
    // 表内容按 t=8 ⇒ nt=1;表在**窗外**造 —— 表本身归引擎写,不归 runtime)
    let nt = 8usize.div_ceil(owl_kernels::family::gdn_chunked::BT).max(1);
    let tbl_bytes = owl_kernels::family::gdn_chunked::table_host_bytes(8, nt);
    let tables = owl_iface::htod!(client, Dtype::U8, &Shape::from(vec![tbl_bytes.len()]), &tbl_bytes)
        .await
        .expect("htod tables")
        .view(Dtype::U32, 1);

    let call = GdnChunkedCall::builder(shape)
        .q(&q).expect("q 契约")
        .k(&k).expect("k 契约")
        .v(&v).expect("v 契约")
        .beta(&beta).expect("beta 契约")
        .gate_raw(&gate).expect("gate 契约")
        .state(&state, 0).expect("state 契约")
        .tables(&tables, nt).expect("tables 契约")
        .build().expect("家族调用装配");
    let msg = call.to_launch(out.clone());

    let strict = std::env::var("OWL_CAP_STRICT").map(|v| v != "0" && !v.is_empty()).unwrap_or(false);
    client.graph_begin().await.expect("graph_begin");
    let r = client.launch(msg).await;
    if strict {
        let e = r.expect_err("严格档:窗内家族 scratch 分配必须被拒");
        let m = format!("{e}");
        assert!(m.contains("捕获窗内不得现建 scratch"), "拒语应可行动,实得 {m}");
        // 窗已脏 ⇒ abort 收窗(真实失败路径上的 abort 演练)
        client.graph_abort().await.expect("失败窗须能 abort");
        // 收窗后同一家族调用在窗外仍合法(eager 路零回归)
        let call2 = GdnChunkedCall::builder(shape)
            .q(&q).expect("q2").k(&k).expect("k2").v(&v).expect("v2")
            .beta(&beta).expect("beta2").gate_raw(&gate).expect("gate2")
            .state(&state, 0).expect("state2")
            .tables(&tables, nt).expect("tables2")
            .build().expect("build2");
        client.launch(call2.to_launch(out.clone())).await.expect("窗外家族调用应照常");
    } else {
        r.expect("观察档:窗内家族调用放行(只记账,行为不变)");
        // ⚠️ **故意不回放**:图里含"从栈上源拷"的 H2D 节点 ⇒ 回放必错(这正是
        // 本条守卫要拦的东西)。收图 + 销毁(inspect 一道:节点数 > 0)。
        let gid = client.graph_end().await.expect("窗有发射 ⇒ 收图");
        client.graph_destroy(gid).await.expect("destroy");
    }
}

/// **步骤 13:全量记名 —— 名字真的到账了**(2026-10-16)。
///
/// 为什么必须端到端验:`cargo check` 只证明**签名带 name**;OOM dump 要的是
/// 名字**落在块账房**。本步拿客户端查询面读账房真值,锁三件事:
/// ① 名字原样可见(不是占位/被弄丢);② 字节口径对(512 f32 = 2.0KiB);
/// ③ **零未命名** —— 本步全用带名 API ⇒ 账房不该出现无名块(门不误报)。
/// 同时它就是这条链的"整段通路"测试:`alloc`/`htod` → `Command::{Alloc,Htod}`
/// → `new_block(name)` → `BlockEntry.name` → 按名聚合渲染。
/// ⚠️ 必须在 `step_close_semantics` **之前**跑(close 后 client 不可用)。
async fn step_ledger_is_named(client: &mut GpuClient) {
    // 本步故意用**显式语义名**(而不用 `owl_iface::alloc!` 这类宏):锁"调用点直报名"这条路
    // 仍然有效(生产代码走的就是它)。宏那条路由本文件另一处(窗内断言)覆盖
    // ⇒ 两种命名路径各有门。
    const A_NAME: &str = "ledgerprobe.alloc";
    const H_NAME: &str = "ledgerprobe.htod";
    let elems = 512usize;
    let _a = client.alloc(Dtype::F32, elems, A_NAME).await.expect("alloc 带名");
    let src = vec![1.0f32; elems];
    let _h = client
        .htod(Dtype::F32, &Shape::from(vec![elems]), &le_f32(&src), H_NAME)
        .await
        .expect("htod 带名");
    let rep = client.ledger_report(16).await.expect("ledger_report 查询");
    assert!(rep.contains(A_NAME), "alloc 名字必须进块账房:{rep}");
    assert!(rep.contains(H_NAME), "htod 名字必须进块账房:{rep}");
    assert!(
        rep.contains(&format!("{A_NAME}=2.0KiB×1")),
        "名字必须聚到位(512×f32=2KiB ⇒ KiB 口径×1):{rep}"
    );
    assert!(!rep.contains("未命名块"), "本步全带名 ⇒ 不得有未命名块:{rep}");

    // **结构化那个面**(OOM 落盘用的是同一序列化器):形状 + 字节数 + 名字。
    // 这里做"能被解析"的**轻量**校验(不引第三方 JSON 库):括号配平 + 关键片段。
    // 真解析在验证脚本里做(见 docs/arch/全量记名-与-OOM-dump.md §7 复现命令)。
    let js = client.ledger_json().await.expect("ledger_json 查询");
    // ① **真语法校验**(严格栈式;不是数括号那种近似):
    assert_valid_json(&js);
    // ② 两段都在(OOM 现场 = 块账 + 家族 scratch;少一半就少一半归因)。
    // ⚠️ **判据只用"真解析 + 片段在",不用字符串尾部猜形状**:首版写了
    // `ends_with("}}")` 并据此判红一次 —— 而服务端实测同一串 open/close 各 56
    // 完全配平(拼装逻辑是对的),错的是"数尾巴"这种脆弱判据本身。
    // 结构化数据一律交给 `assert_valid_json`(它才是权威)。
    // meta 段(时间戳/相位/引擎账;跨时间比对与"这是什么时候的现场"的前提)
    assert!(js.contains("\"meta\":{\"ts_ms\":"), "meta 段必须在:{js}");
    assert!(
        js.contains("\"phase\":\"") && (js.contains("\"phase\":\"eager\"") || js.contains("\"phase\":\"graph\"") || js.contains("\"phase\":\"capture\"")),
        "相位词表有界(eager/graph/capture):{js}"
    );
    // 外壳 = 两段各是自洽对象(**嵌套表达归属**,不靠顺序):
    //   {"blocks":{"blocks":[…],"total_*":…},"scratch":{"scratch":[…],"total_*":…}}
    assert!(js.starts_with("{\"meta\":{"), "外壳头(meta 在前)不对:{js}");
    assert!(js.contains("\"blocks\":{\"blocks\":["), "blocks 段(自洽对象)必须在:{js}");
    assert!(js.contains(",\"scratch\":{\"scratch\":["), "scratch 段(自洽对象)必须在:{js}");
    assert!(js.contains("\"total_bytes\":") && js.contains("\"total_blocks\":"), "两段都要 totals:{js}");
    // ③ 内容口径:名字 + 原始字节 + 分组块数
    assert!(
        js.contains(&format!(
            "{{\"name\":\"{A_NAME}\",\"bytes\":{},\"blocks\":1}}",
            elems * 4
        )),
        "结构化的名字/字节/块数不对:{js}"
    );
}


/// **JSON 语法校验器**(递归下降;判据 = "落盘现场能被标准解析器读")。
///
/// 为什么不引 `serde_json`:只为一条测试加依赖会把 `Cargo.lock`/构建图
/// 都拖进来;而这里要判的只有**结构合法性**。本实现覆盖 JSON 的完整语法
/// (对象/数组/字符串转义/数字/字面量/空白),失败信息指向第一个坏字符。
fn assert_valid_json(text: &str) {
    struct P<'a> {
        b: &'a [u8],
        i: usize,
    }
    impl<'a> P<'a> {
        fn ws(&mut self) {
            while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
                self.i += 1;
            }
        }
        fn take(&mut self, c: u8) -> bool {
            self.ws();
            if self.i < self.b.len() && self.b[self.i] == c {
                self.i += 1;
                true
            } else {
                false
            }
        }
        fn string(&mut self) -> bool {
            if !self.take(b'"') {
                return false;
            }
            while self.i < self.b.len() {
                match self.b[self.i] {
                    b'\\' => self.i += 2,
                    b'"' => {
                        self.i += 1;
                        return true;
                    }
                    _ => self.i += 1,
                }
            }
            false
        }
        fn value(&mut self) -> bool {
            self.ws();
            if self.i >= self.b.len() {
                return false;
            }
            match self.b[self.i] {
                b'{' => {
                    self.i += 1;
                    self.ws();
                    if self.take(b'}') {
                        return true;
                    }
                    loop {
                        if !self.string() {
                            return false;
                        }
                        if !self.take(b':') {
                            return false;
                        }
                        if !self.value() {
                            return false;
                        }
                        if self.take(b',') {
                            continue;
                        }
                        return self.take(b'}');
                    }
                }
                b'[' => {
                    self.i += 1;
                    self.ws();
                    if self.take(b']') {
                        return true;
                    }
                    loop {
                        if !self.value() {
                            return false;
                        }
                        if self.take(b',') {
                            continue;
                        }
                        return self.take(b']');
                    }
                }
                b'"' => self.string(),
                b't' => self.lit("true"),
                b'f' => self.lit("false"),
                b'n' => self.lit("null"),
                b'-' | b'0'..=b'9' => {
                    let start = self.i;
                    self.i += 1;
                    while self.i < self.b.len()
                        && matches!(self.b[self.i], b'0'..=b'9' | b'.' | b'e' | b'E' | b'+' | b'-')
                    {
                        self.i += 1;
                    }
                    self.i > start
                }
                _ => false,
            }
        }
        fn lit(&mut self, w: &str) -> bool {
            if self.b[self.i..].starts_with(w.as_bytes()) {
                self.i += w.len();
                true
            } else {
                false
            }
        }
    }
    let mut p = P { b: text.as_bytes(), i: 0 };
    assert!(p.value(), "非法 JSON(offset {}):{:.40}", p.i, &text[p.i.min(text.len())..]);
    p.ws();
    assert_eq!(p.i, p.b.len(), "JSON 尾部有残留:{}", &text[p.i..]);
}

/// **步骤 14:峰值快照门(真机)**——`OWL_VRAM_PEAK_MB` 门控开时,
/// 越过水位线的一次真分配必须落下一份现场,且那份是**合法 JSON**。
///
/// 为什么独立成步:①它验的是"触发面"(分配路径里的 `maybe_dump_peak`),
/// 与步骤 13 的查询面不同;②门是 env ⇒ 只在设了 env 的运行里生效,
/// 未设时本步**必须自行 skip**(不许静默假绿)。
async fn step_peak_snapshot(client: &mut GpuClient) {
    if std::env::var_os("OWL_VRAM_PEAK_MB").is_none() {
        eprintln!("[skip] 步骤 14:未设 OWL_VRAM_PEAK_MB(门控默认关)");
        return;
    }
    // ⚠️ 门值是**调用方**给的(测试用 1MiB),而**步进 5%** 是水位判定 ——
    // 前面的步骤已把 live 抬到 768MiB 级,所以这里必须分配得**足够大**才破
    // 步进(768×1.05≈806MiB ⇒ 30MiB 已够;给 64MiB 留余量)。首版只分 8MiB
    // ⇒ 没破步进 ⇒ 读到的是**上一次**的快照(不带本块名),看起来像"触发面断了",
    // 其实是测试没触发。教训:读"最近文件"这种判据必须与**触发条件**对齐。
    let before = peak_snapshot_files().len();
    let _big = client
        .alloc(Dtype::U32, 16 << 20, "peaksnap.big") // 64MiB
        .await
        .expect("大块分配");
    client.sync().await.expect("sync");
    // 落盘在 server 线程异步发生 ⇒ 轮询等"**新文件**出现"(比 before 多;带超时)
    let mut fresh: Option<(std::path::PathBuf, String)> = None;
    for _ in 0..100 {
        let mut hits = peak_snapshot_files();
        hits.sort();
        if hits.len() > before {
            let p = hits.last().expect("非空").clone();
            let body = std::fs::read_to_string(&p).expect("读快照");
            fresh = Some((p, body));
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    let (path, body) = fresh.expect("破了门却没落下新快照 ⇒ 触发面断了");
    assert_valid_json(&body);
    assert!(body.contains("\"meta\":{\"ts_ms\":"), "快照必须带 meta:{path:?}");
    assert!(body.contains("peaksnap.big"), "新快照里必须能看见刚分配的那块:{path:?}");
}

/// /tmp 下的峰值快照文件表(测试用;门控开时才有)
fn peak_snapshot_files() -> Vec<std::path::PathBuf> {
    std::fs::read_dir("/tmp")
        .map(|rd| {
            rd.flatten()
                .map(|e| e.path())
                .filter(|p| {
                    p.file_name()
                        .and_then(|n| n.to_str())
                        .is_some_and(|n| n.starts_with("owl-vram-peak-") && n.ends_with(".json"))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// 步骤 7:close 语义
async fn step_close_semantics(client: &mut GpuClient) {
    client.close().await.expect("close 应 Ok");
    let r = client.sync().await;
    assert!(
        matches!(r, Err(ModelError::ServerClosed)),
        "close 后 sync 应返回 ServerClosed,实得 {r:?}"
    );
    let r = owl_iface::client_alloc!(client, Dtype::F32, 4).await;
    assert!(
        matches!(r, Err(ModelError::ServerClosed)),
        "close 后 alloc 应返回 ServerClosed,实得 {r:?}"
    );
}

/// 步骤 8:selector 错 UUID → boot 失败(独立 mini 组装;不碰主 client)
fn step_selector_bad_uuid() {
    let (tx, rx) = std::sync::mpsc::channel::<Command>();
    let (boot_tx, boot_rx) = std::sync::mpsc::channel::<Result<(), String>>();
    let server = GpuServer::new(rx, DeviceSelector::Uuid([0xAB; 16]), Some(boot_tx));
    let _client = GpuClient::new(tx);
    std::thread::Builder::new()
        .spawn(move || {
            let _ = server.run();
        })
        .expect("起线程失败");
    let boot = boot_rx
        .recv_timeout(std::time::Duration::from_secs(30))
        .expect("boot 通道应回执而非悬挂");
    assert!(boot.is_err(), "错 UUID 应 boot 失败,实得 {boot:?}");
}

// ============================================================================
// 大测试:一条管道顺序跑完全部步骤
// ============================================================================

#[tokio::test]
async fn gpu_server_contract() {
    if owl_cuda::test_device_bdf().is_none() {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    }
    // selector 负路径必须最先跑(cuInit 枚举失败路径;不能放在主 context
    // 遥拆之后 —— 同进程驱动层死锁,实测挂死)
    step_selector_bad_uuid();

    let mut client = assemble();

    // ---- 正常路径 ----
    step_alloc_zeroed(&mut client).await; // 1
    step_htod_dtoh_roundtrip(&mut client).await; // 2
    step_launch_scale(&mut client).await; // 3
    step_sync_is_barrier(&mut client).await; // 4

    // ---- 图:捕获/重放 + 护栏(需要 Block 输入)----
    let xs = [1.0f32; 4];
    let x = owl_iface::htod!(client, Dtype::F32, &Shape::from(vec![4]), &le_f32(&xs))
        .await
        .expect("htod");
    step_graph_capture_and_replay(&mut client, &x).await;
    step_piecewise_two_segments(&mut client, &x).await; // 5
    step_graph_guardrails(&mut client, &x).await; // 6

    // ---- S1 探针:cuGraphLaunch host 成本(微型 vs 大图;零旗标已定谳无关)----
    step_graph_launch_probe(&mut client, &x).await;

    // ---- 图引用租约(捕获过的块不得回收;自由块照常回收)----
    step_graph_ref_lease(&mut client).await; // 9

    // ---- M1:窗内 Free 复用 / abort / destroy ----
    step_capture_window_reuse(&mut client).await; // 10
    step_graph_abort_and_destroy(&mut client).await; // 11

    // ---- 家族面捕获入境守卫(GDN chunked 真身;按 OWL_CAP_STRICT 分支)----
    step_family_window_guard(&mut client).await; // 12

    // ---- 关机(最后;此后主 client 不可再用)----
    // 前置:全量记名的端到端门(必须是 close 前最后一步 —— 它要读块账真值)
    step_ledger_is_named(&mut client).await; // 13
    step_peak_snapshot(&mut client).await; // 14(未设门 = 自行 skip)
    step_close_semantics(&mut client).await; // 7

    eprintln!("[t] 全部步骤完成");
}

/// S1 探针:小图(3 节点)与大图(~500 节点)各连发 100 次,分相计时。
/// 判决:小图也 ms 级 = 系统性(驱动/环境);大图独有 = 节点规模税。
async fn step_graph_launch_probe(client: &mut GpuClient, x: &Bytes) {
    let zeros = [0.0f32; 4];
    let y = owl_iface::htod!(client, Dtype::F32, &Shape::from(vec![4]), &le_f32(&zeros))
        .await
        .expect("htod y");
    for (tag, launches) in [("小图", 1usize), ("大图", 500usize)] {
        client.graph_begin().await.expect("begin");
        let out = owl_iface::client_alloc!(client, Dtype::F32, 4).await.expect("alloc").view(Dtype::F32, 1);
        for _ in 0..launches {
            client.launch(add_msg(x, &y, &out, 4)).await.expect("launch");
        }
        let gid = client.graph_end().await.expect("end");
        // 预热 1 次(首启分配路径)
        client.graph_launch(gid).await.expect("launch");
        client.sync().await.expect("sync");
        let t0 = std::time::Instant::now();
        let n = 100usize;
        for _ in 0..n {
            client.graph_launch(gid).await.expect("launch");
        }
        client.sync().await.expect("sync");
        eprintln!(
            "[launch-probe] {tag}(~{} 节点)×{} 次:总 {:?} / 均 {:?}",
            launches * 3 + 2,
            n,
            t0.elapsed(),
            t0.elapsed() / n as u32
        );
    }
}

// ============================================================================
// F1:cuBLAS GEMM(f16 基线)对拍 —— GPU f16 gemm vs host f32 参考
// ============================================================================

fn le_f16(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|f| half::f16::from_f32(*f).to_le_bytes())
        .collect()
}

fn f16_to_f32(buf: &[u8]) -> Vec<f32> {
    buf.chunks_exact(2)
        .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect()
}

#[tokio::test]
async fn gpu_gemm_f16_matches_host() {
    if owl_cuda::test_device_bdf().is_none() {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    }
    let mut client = assemble();
    // C[T=8, n=32] = A[8, k=64] × W[32, 64]^T(nt;owl Linear 惯例)
    let (t, k, n) = (8usize, 64usize, 32usize);
    let a: Vec<f32> = (0..t * k).map(|i| ((i as f32 * 0.37) - 4.0).sin() * 0.7).collect();
    let w: Vec<f32> = (0..n * k).map(|i| ((i as f32 * 0.11) - 2.0).cos() * 0.5).collect();

    // host f32 参考
    let mut want = vec![0f32; t * n];
    for (ti, a_row) in a.chunks_exact(k).enumerate() {
        for (wi, w_row) in w.chunks_exact(k).enumerate() {
            let acc: f32 = a_row.iter().zip(w_row).map(|(x, y)| x * y).sum();
            want[ti * n + wi] = acc;
        }
    }

    let da = owl_iface::htod!(client, Dtype::F16, &Shape::from(vec![t, k]), &le_f16(&a)).await.expect("htod a");
    let dw = owl_iface::htod!(client, Dtype::F16, &Shape::from(vec![n, k]), &le_f16(&w)).await.expect("htod w");
    let dout = owl_iface::client_alloc!(client, Dtype::F16, t * n).await.expect("alloc out").view(Dtype::F16, 2); // [m,n]
    let msg = LaunchMsg {
        kernel: KernelSource { name: "cublas_gemm_f16".into(), source: String::new() },
        args: vec![
            Arg::block(da.id, da.dtype, da.rank),
            Arg::block(dw.id, dw.dtype, dw.rank),
            Arg::block(dout.id, dout.dtype, dout.rank),
            Arg::U64(n as u64),
            Arg::U64(k as u64),
            Arg::U64(t as u64),
            Arg::U64(1),
        ],
        grid: (0, 0, 0),
        block: (0, 0, 0),
        shared_mem: 0,
        out_elems: t * n,
    };
    client.launch(msg).await.expect("gemm launch");

    let mut buf = vec![0u8; t * n * 2];
    client.dtoh(&dout, &mut buf).await.expect("dtoh");
    let got = f16_to_f32(&buf);

    let mut max_diff = 0f32;
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        let diff = (g - w).abs();
        max_diff = max_diff.max(diff);
        assert!(
            diff < 5e-2 * (1.0 + w.abs()),
            "[{i}] gemm f16 {g} vs host {w}"
        );
    }
    eprintln!("[gemm] max_diff = {max_diff:.5}(f16 in/acc32,容差 5e-2 相对)");
    client.sync().await.expect("sync");
}

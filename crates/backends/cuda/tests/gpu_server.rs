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
    let out = client.alloc(Dtype::F32, 8).await.expect("alloc").view(Dtype::F32, 1);
    let got = harvest(client, &out, 8).await;
    assert!(
        got.iter().all(|&v| v == 0.0),
        "alloc 应零化(N4 回归),实得 {got:?}"
    );
}

/// 步骤 2:htod/dtoh 往返
async fn step_htod_dtoh_roundtrip(client: &mut GpuClient) {
    let src = [1.5f32, -2.25, 3.125, 1e-6, 42.0];
    let block = client
        .htod(Dtype::F32, &Shape::from(vec![src.len()]), &le_f32(&src))
        .await
        .expect("htod");
    let got = harvest(client, &block, src.len()).await;
    assert_eq!(got, src, "htod/dtoh 往返失真: {got:?} vs {src:?}");
}

/// 步骤 3:手工 LaunchMsg 发射(kernel + 槽序契约)
async fn step_launch_scale(client: &mut GpuClient) {
    let xs = [2.0f32; 4];
    let x = client
        .htod(Dtype::F32, &Shape::from(vec![4]), &le_f32(&xs))
        .await
        .expect("htod");
    let out = client.alloc(Dtype::F32, 4).await.expect("alloc").view(Dtype::F32, 1);
    let ys = [4.0f32; 4];
    let y = client
        .htod(Dtype::F32, &Shape::from(vec![4]), &le_f32(&ys))
        .await
        .expect("htod y");
    client.launch(add_msg(&x, &y, &out, 4)).await.expect("launch");
    let got = harvest(client, &out, 4).await;
    assert_all_close(&got, 6.0, "add 槽序(2+4)");
}

/// 步骤 4:launch 后 sync 栅栏收割(大数组)
async fn step_sync_is_barrier(client: &mut GpuClient) {
    let src: Vec<f32> = (0..4096).map(|i| i as f32 * 0.5).collect();
    let x = client
        .htod(Dtype::F32, &Shape::from(vec![src.len()]), &le_f32(&src))
        .await
        .expect("htod");
    let out = client.alloc(Dtype::F32, src.len()).await.expect("alloc").view(Dtype::F32, 1);
    let zeros = vec![0.0f32; src.len()];
    let y = client
        .htod(Dtype::F32, &Shape::from(vec![src.len()]), &le_f32(&zeros))
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
    let y = client
        .htod(Dtype::F32, &Shape::from(vec![4]), &le_f32(&ones))
        .await
        .expect("htod y");
    client.graph_begin().await.expect("graph_begin");

    // 捕获期 Alloc(切片块 + memset 进图)+ scale 发射
    let zeros = client.alloc(Dtype::F32, 4).await.expect("captured alloc").view(Dtype::F32, 1);
    let out = client.alloc(Dtype::F32, 4).await.expect("captured alloc").view(Dtype::F32, 1);
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
    let a = client.htod(Dtype::F32, &Shape::from(vec![4]), &le_f32(&ones)).await.expect("htod a");
    let b = client.htod(Dtype::F32, &Shape::from(vec![4]), &le_f32(&ones)).await.expect("htod b");
    let out = client.alloc(Dtype::F32, 4).await.expect("alloc out").view(Dtype::F32, 1);

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
    let loose = client.htod(Dtype::F32, &Shape::from(vec![4]), &le_f32(&ones)).await.expect("htod loose");
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
    let x = client.htod(Dtype::F32, &Shape::from(vec![4]), &le_f32(&ones)).await.expect("htod x");
    let y = client.htod(Dtype::F32, &Shape::from(vec![4]), &le_f32(&ones)).await.expect("htod y");
    client.graph_begin().await.expect("graph_begin");
    let mut live: Option<Bytes> = None;
    let mut total = 0usize;
    for _ in 0..N {
        let b = client.alloc(Dtype::F32, BLK).await.expect("窗内 alloc(切 slab)");
        total += BLK * 4;
        if let Some(prev) = live.take() {
            // 交替释放上一块 ⇒ 活块数恒 ≤ 2(回 slab 内 arena,零 CUDA 调用)
            client.free(&[prev.id]).await.expect("窗内 free(回 arena)");
        }
        live = Some(b);
    }
    // 空窗哨兵③:发射数须 > 0
    let out = client.alloc(Dtype::F32, 4).await.expect("alloc out").view(Dtype::F32, 1);
    client.launch(add_msg(&x, &y, &out, 4)).await.expect("launch");
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
    let a = client.htod(Dtype::F32, &Shape::from(vec![4]), &le_f32(&ones)).await.expect("htod a");
    let b = client.htod(Dtype::F32, &Shape::from(vec![4]), &le_f32(&ones)).await.expect("htod b");
    client.graph_begin().await.expect("graph_begin");
    let carved = client.alloc(Dtype::F32, 4).await.expect("窗内 alloc").view(Dtype::F32, 1);
    client.launch(add_msg(&a, &b, &carved, 4)).await.expect("launch");
    client.graph_abort().await.expect("abort 应成功");
    // 窗内自产块整批作废(不登记图 ⇒ 永不会被回放)
    let r = client.dtoh(&carved, &mut [0u8; 16]).await;
    assert!(matches!(r, Err(ModelError::DeadBlock { .. })), "abort 后 slab 块应作废,实得 {r:?}");
    // abort 后引擎仍可正常捕获(窗态已复位 —— 否则第二张图必炸)
    client.graph_begin().await.expect("abort 后再捕获");
    let out2 = client.alloc(Dtype::F32, 4).await.expect("alloc").view(Dtype::F32, 1);
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
    let q = client.htod(Dtype::F16, &Shape::from(vec![8, 1, 128]), &le_f16(&ones_f16)).await.expect("htod q");
    let k = client.htod(Dtype::F16, &Shape::from(vec![8, 1, 128]), &le_f16(&ones_f16)).await.expect("htod k");
    let v = client.htod(Dtype::F16, &Shape::from(vec![8, 1, 128]), &le_f16(&ones_f16)).await.expect("htod v");
    let beta = client.htod(Dtype::F16, &Shape::from(vec![8, 1]), &le_f16(&[0.5; 8])).await.expect("htod beta");
    let gate = client.htod(Dtype::F16, &Shape::from(vec![8, 1]), &le_f16(&[-0.01; 8])).await.expect("htod gate");
    let state = client.alloc(Dtype::F32, 128 * 128).await.expect("alloc state").view(Dtype::F32, 4);
    let out = client.alloc(Dtype::F16, elems_qkv).await.expect("alloc out").view(Dtype::F16, 3);
    // P-6b:序列表(cu|coff|idx)现造一块(本测只关心窗内 alloc/upload 守卫,
    // 表内容按 t=8 ⇒ nt=1;表在**窗外**造 —— 表本身归引擎写,不归 runtime)
    let nt = 8usize.div_ceil(owl_kernels::family::gdn_chunked::BT).max(1);
    let tbl_bytes = owl_kernels::family::gdn_chunked::table_host_bytes(8, nt);
    let tables = client
        .htod(Dtype::U8, &Shape::from(vec![tbl_bytes.len()]), &tbl_bytes)
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

/// 步骤 7:close 语义
async fn step_close_semantics(client: &mut GpuClient) {
    client.close().await.expect("close 应 Ok");
    let r = client.sync().await;
    assert!(
        matches!(r, Err(ModelError::ServerClosed)),
        "close 后 sync 应返回 ServerClosed,实得 {r:?}"
    );
    let r = client.alloc(Dtype::F32, 4).await;
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
    let x = client
        .htod(Dtype::F32, &Shape::from(vec![4]), &le_f32(&xs))
        .await
        .expect("htod");
    step_graph_capture_and_replay(&mut client, &x).await; // 5
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
    step_close_semantics(&mut client).await; // 7

    eprintln!("[t] 全部步骤完成");
}

/// S1 探针:小图(3 节点)与大图(~500 节点)各连发 100 次,分相计时。
/// 判决:小图也 ms 级 = 系统性(驱动/环境);大图独有 = 节点规模税。
async fn step_graph_launch_probe(client: &mut GpuClient, x: &Bytes) {
    let zeros = [0.0f32; 4];
    let y = client
        .htod(Dtype::F32, &Shape::from(vec![4]), &le_f32(&zeros))
        .await
        .expect("htod y");
    for (tag, launches) in [("小图", 1usize), ("大图", 500usize)] {
        client.graph_begin().await.expect("begin");
        let out = client.alloc(Dtype::F32, 4).await.expect("alloc").view(Dtype::F32, 1);
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

    let da = client.htod(Dtype::F16, &Shape::from(vec![t, k]), &le_f16(&a)).await.expect("htod a");
    let dw = client.htod(Dtype::F16, &Shape::from(vec![n, k]), &le_f16(&w)).await.expect("htod w");
    let dout = client.alloc(Dtype::F16, t * n).await.expect("alloc out").view(Dtype::F16, 2); // [m,n]
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

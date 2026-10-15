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
    let one = client
        .htod(Dtype::F32, &Shape::from(vec![4]), &le_f32(&[1.0f32; 4]))
        .await
        .expect("htod one");
    let mid = client.alloc(Dtype::F32, 4).await.expect("pin 块(mid)").view(Dtype::F32, 1);
    let tail = client.alloc(Dtype::F32, 4).await.expect("pin 块(tail)").view(Dtype::F32, 1);

    // ── 段0:out0 = x + one ──
    let out0 = client.alloc(Dtype::F32, 4).await.expect("carve out0").view(Dtype::F32, 1);
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
    step_graph_capture_and_replay(&mut client, &x).await;
    step_piecewise_two_segments(&mut client, &x).await; // 5
    step_graph_guardrails(&mut client, &x).await; // 6

    // ---- S1 探针:cuGraphLaunch host 成本(微型 vs 大图;零旗标已定谳无关)----
    step_graph_launch_probe(&mut client, &x).await;

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

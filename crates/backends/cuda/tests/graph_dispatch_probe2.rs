//! 刀D 探针②:外来大核(marlin / cublas gemv)在图内的派发税。
//! 立案:引擎图 10.8µs/节点 vs 合成 add 链 2.7µs/节点(4× 内容罚金);
//! 嫌疑 = marlin/cublas 类节点每节点携带 ~50µs 派发(317×50 ≈ 16ms 吻合)。
//! 方法:同形 K 次链式捕获(输出→下输入)→ 暖机 → 重放计时,
//! 对照内核裸时长(marlin [16384,16384] ≈ 134MB/BW ≈ 65µs)。

use std::time::Instant;
use owl_cuda::{
    test_device_ordinal, Arg, DeviceClient as _, DeviceSelector, Dtype, GpuClient, GpuServer,
    LaunchMsg, Shape,
};

fn assemble() -> GpuClient {
    let ordinal = test_device_ordinal();
    let (tx, rx) = std::sync::mpsc::channel::<owl_cuda::Command>();
    let server = GpuServer::new(rx, DeviceSelector::Ordinal(ordinal), None);
    let client = GpuClient::new(tx);
    std::thread::Builder::new()
        .name("owl-probe2".into())
        .spawn(move || server.run().expect("server run"))
        .expect("起 server 线程失败");
    client
}

fn le_u16(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn le_i32(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[tokio::test]
async fn marlin_graph_dispatch_probe() {
    let mut client = assemble();
    client.sync().await.unwrap();

    let (m, n, k, g) = (1usize, 16384usize, 16384usize, 32usize);
    let q: Vec<u8> = (0..n * k).map(|i| ((i * 7 + 3) % 16) as u8).collect();
    let s: Vec<f32> = (0..n * (k / g)).map(|i| 0.5 + ((i * 13) % 13) as f32 * 0.08).collect();
    let t0 = Instant::now();
    let b_packed = owl_kernels::marlin::repack::pack_marlin_b(&q, k, n);
    let s_packed = owl_kernels::marlin::repack::pack_marlin_s(&s, n, k / g);
    eprintln!("[m-probe] repack {:.1}s", t0.elapsed().as_secs_f32());

    let da = client
        .htod(Dtype::F16, &Shape::from(vec![m, k]), &le_u16(
            &(0..m * k).map(|i| half::f16::from_f32(0.1).to_bits()).collect::<Vec<_>>(),
        ))
        .await
        .expect("htod a");
    let db = client.htod(Dtype::U32, &Shape::from(vec![b_packed.len()]), &le_i32(&b_packed)).await.expect("htod b");
    let ds = client.htod(Dtype::F16, &Shape::from(vec![s_packed.len()]), &le_u16(&s_packed)).await.expect("htod s");
    let ws_len = owl_kernels::marlin::v2_workspace_len(n).max(n / 128 * 16);
    let dws = client.alloc(Dtype::U32, ws_len).await.expect("ws");
    let dctmp = client.alloc(Dtype::U32, 1).await.expect("ctmp");
    let dc = client.alloc(Dtype::F16, m * n).await.expect("c");

    let msg = |a: &owl_cuda::Bytes| LaunchMsg {
        kernel: owl_cuda::KernelSpec { name: "marlin_gemm_w4a16".into(), source: String::new() },
        args: vec![
            Arg::Block { id: a.id },
            Arg::Block { id: db.id },
            Arg::Block { id: dc.id },
            Arg::Block { id: ds.id },
            Arg::Block { id: dws.id },
            Arg::Block { id: dctmp.id },
            Arg::U64(m as u64),
            Arg::U64(k as u64),
            Arg::U64(n as u64),
            Arg::U64(g as u64),
        ],
        grid: (0, 0, 0),
        block: (0, 0, 0),
        shared_mem: 0,
        out_elems: m * n,
    };

    // 单发 eager 裸时长(内核基线)
    client.launch(msg(&da)).await.expect("eager marlin");
    client.sync().await.unwrap();
    const R: usize = 30;
    let t0 = Instant::now();
    for _ in 0..R {
        client.launch(msg(&da)).await.expect("eager marlin");
    }
    client.sync().await.unwrap();
    let eager = (Instant::now() - t0).as_secs_f64() * 1e3 / R as f64;
    eprintln!("[m-probe] eager 单发: {eager:.1}µs(含 ack 往返)");

    // 图内链式 K 发(输出 → 下输入,同权重)
    const K: usize = 300;
    let mut cur = dc.clone();
    client.graph_begin().await.expect("graph_begin");
    for _ in 0..K {
        let nxt = client.alloc(Dtype::F16, m * n).await.expect("captured c");
        client.launch(msg(&cur)).await.expect("captured marlin");
        cur = nxt;
    }
    let sentinel_out = client.alloc(Dtype::F32, 4).await.expect("sentinel out");
    let sentinel_in = client.alloc(Dtype::F32, 4).await.expect("sentinel in");
    client.launch(LaunchMsg {
        kernel: owl_cuda::KernelSpec { name: "probe_add_f32".into(), source: r#"
extern "C" __global__ void probe_add_f32(
    const float* a, const float* b, float* out, const size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (int)n) { out[i] = a[i] + b[i]; }
}
"#.into() },
        args: vec![
            Arg::Block { id: sentinel_in.id },
            Arg::Block { id: sentinel_in.id },
            Arg::Block { id: sentinel_out.id },
            Arg::U64(4),
        ],
        grid: (1, 1, 1),
        block: (256, 1, 1),
        shared_mem: 0,
        out_elems: 4,
    }).await.expect("sentinel native launch");
    let gid = client.graph_end().await.expect("graph_end");
    for _ in 0..3 {
        client.graph_launch(gid).await.expect("warm");
    }
    client.sync().await.unwrap();
    let t0 = Instant::now();
    for _ in 0..10 {
        client.graph_launch(gid).await.expect("replay");
    }
    client.sync().await.unwrap();
    let wall = (Instant::now() - t0).as_secs_f64() * 1e3 / 10.0;
    let per = wall * 1e3 / K as f64;
    eprintln!(
        "[m-probe] 图链 K={K}: replay {wall:.2}ms → {per:.1}µs/节点(内核裸 ~{}µs → 派发税 {:.1}µs)",
        eager.max(1.0) as i64,
        per - eager
    );
}

#[tokio::test]
async fn cublas_graph_dispatch_probe() {
    let mut client = assemble();
    client.sync().await.unwrap();
    // b/a 同款 gemv 形状:nt [48, 5120] × [1, 5120] → [1, 48]
    let (mm, kk, nn) = (48usize, 5120usize, 1usize);
    let w = client.alloc(Dtype::F16, mm * kk).await.expect("w");
    let x = client.alloc(Dtype::F16, nn * kk).await.expect("x");
    let dc = client.alloc(Dtype::F16, nn * mm).await.expect("c");

    let msg = |a: &owl_cuda::Bytes| LaunchMsg {
        kernel: owl_cuda::KernelSpec { name: "cublas_gemm_f16".into(), source: String::new() },
        args: vec![
            Arg::Block { id: x.id },
            Arg::Block { id: w.id },
            Arg::Block { id: dc.id },
            Arg::U64(mm as u64),
            Arg::U64(kk as u64),
            Arg::U64(nn as u64),
            Arg::U64(1),
        ],
        grid: (0, 0, 0),
        block: (0, 0, 0),
        shared_mem: 0,
        out_elems: nn * mm,
    };

    client.launch(msg(&x)).await.expect("eager gemv");
    client.sync().await.unwrap();
    const R: usize = 50;
    let t0 = Instant::now();
    for _ in 0..R {
        client.launch(msg(&x)).await.expect("eager gemv");
    }
    client.sync().await.unwrap();
    let eager = (Instant::now() - t0).as_secs_f64() * 1e3 / R as f64;
    eprintln!("[c-probe] eager 单发: {eager:.1}µs");

    const K: usize = 300;
    const SCALE_CU: &str = r#"
extern "C" __global__ void probe_scale_f32(
    const float* x, float k, const size_t n, float* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (int)n) { out[i] = x[i] * k; }
}
"#;
    client.graph_begin().await.expect("graph_begin");
    let sentinel_out = client.alloc(Dtype::F32, 4).await.expect("sentinel out");
    client.launch(LaunchMsg {
        kernel: owl_cuda::KernelSpec { name: "probe_scale_f32".into(), source: SCALE_CU.into() },
        args: vec![
            Arg::Block { id: x.id },
            Arg::F32(2.0),
            Arg::U64(4),
            Arg::Block { id: sentinel_out.id },
        ],
        grid: (1, 1, 1),
        block: (256, 1, 1),
        shared_mem: 0,
        out_elems: 4,
    }).await.expect("sentinel native launch");
    for _ in 0..K {
        client.launch(msg(&x)).await.expect("captured gemv");
    }
    let gid = client.graph_end().await.expect("graph_end");
    for _ in 0..3 {
        client.graph_launch(gid).await.expect("warm");
    }
    client.sync().await.unwrap();
    let t0 = Instant::now();
    for _ in 0..10 {
        client.graph_launch(gid).await.expect("replay");
    }
    client.sync().await.unwrap();
    let wall = (Instant::now() - t0).as_secs_f64() * 1e3 / 10.0;
    let per = wall * 1e3 / K as f64;
    eprintln!(
        "[c-probe] 图链 K={K}: replay {wall:.2}ms → {per:.1}µs/节点(派发税 {:.1}µs)",
        per - eager
    );
}

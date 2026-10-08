//! 刀D 探针:图节点派发税裸速率(2026-10-03 立案:引擎图 10.8µs/节点 vs
//! vLLM ~1.4µs/节点,同机同驱动 7.7× 之谜)。
//!
//! 合成对照矩阵(全部 N 节点线性链):
//! - chain:单一 add 内核 × N(基线;驱动地板)
//! - distinct:K 个 distinct 内核轮转(模块/函数切换成本)
//! - wide:12 参大签名(参数量成本)
//! - big:大块(执行时间 > 派发:间隙是否被隐藏)
//! 另测同链 eager(host 直发)对照。

use owl_cuda::{test_device_ordinal, DeviceClient as _, DeviceSelector, Dtype, GpuClient, GpuServer};
use owl_iface::contract::{Arg, Bytes, KernelSource, LaunchMsg, Shape};
use std::time::Instant;

const ADD_CU: &str = r#"
extern "C" __global__ void probe_add_f32(
    const float* a, const float* b, float* out, const size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (int)n) { out[i] = a[i] + b[i]; }
}
"#;

fn add_msg(a: &Bytes, b: &Bytes, out: &Bytes, n: usize) -> LaunchMsg {
    LaunchMsg {
        kernel: KernelSource { name: "probe_add_f32".into(), source: ADD_CU.into() },
        args: vec![
            Arg::Block { id: a.id },
            Arg::Block { id: b.id },
            Arg::Block { id: out.id },
            Arg::U64(n as u64),
        ],
        grid: (((n as u32) + 255) / 256, 1, 1),
        block: (256, 1, 1),
        shared_mem: 0,
        out_elems: n,
    }
}

fn add_msg_owned(a: &Bytes, b: &Bytes, out_id: u64, n: usize) -> LaunchMsg {
    LaunchMsg {
        kernel: KernelSource { name: "probe_add_f32".into(), source: ADD_CU.into() },
        args: vec![
            Arg::Block { id: a.id },
            Arg::Block { id: b.id },
            Arg::Block { id: out_id },
            Arg::U64(n as u64),
        ],
        grid: (((n as u32) + 255) / 256, 1, 1),
        block: (256, 1, 1),
        shared_mem: 0,
        out_elems: n,
    }
}

async fn alloc(client: &mut GpuClient, elems: usize) -> Bytes {
    client.alloc(Dtype::F32, elems).await.expect("alloc")
}

/// 计时壳:捕获链 → 暖机 3 → 重放 R 次 sync 墙
async fn time_replays(client: &mut GpuClient, n_nodes: usize, elems: usize, tag: &str) {
    client.graph_begin().await.expect("graph_begin");
    let b = alloc(client, elems).await;
    let mut cur = alloc(client, elems).await;
    for _ in 0..n_nodes {
        let nxt = alloc(client, elems).await;
        client.launch(add_msg_owned(&cur, &b, nxt.id, elems)).await.expect("captured launch");
        cur = nxt;
    }
    let gid = client.graph_end().await.expect("graph_end");
    for _ in 0..3 {
        client.graph_launch(gid).await.expect("warm replay");
    }
    client.sync().await.unwrap();
    const R: usize = 20;
    let t0 = Instant::now();
    for _ in 0..R {
        client.graph_launch(gid).await.expect("replay");
    }
    client.sync().await.unwrap();
    let wall = (Instant::now() - t0).as_secs_f64() * 1e3 / R as f64;
    eprintln!(
        "[probe] {tag} N={n_nodes} elems={elems}: replay {wall:.2}ms → {:.2} µs/节点",
        wall * 1e3 / n_nodes as f64
    );
}

#[tokio::test]
async fn graph_dispatch_rate_probe() {
    let (tx, rx) = std::sync::mpsc::channel::<owl_cuda::Command>();
    let server = GpuServer::new(rx, DeviceSelector::Ordinal(test_device_ordinal()), None);
    let mut client = GpuClient::new(tx);
    std::thread::Builder::new().spawn(move || server.run().expect("server run")).unwrap();
    client.sync().await.unwrap();

    // A. 基线:单一小内核链(驱动地板;256 元素 = 内核 ~1-2µs)
    time_replays(&mut client, 200, 256, "chain").await;
    time_replays(&mut client, 1000, 256, "chain").await;
    time_replays(&mut client, 2000, 256, "chain").await;
    // B2. 异构形状交替(奇偶节点 elems 不同 → grid 互异;缓冲循环复用)
    {
        let elems_a = 256;
        let elems_b = 16384; // grid 1 块 vs 64 块交替;内核 ~1µs vs ~4µs
        let b = alloc(&mut client, elems_b).await;
        let mut cur = alloc(&mut client, elems_a).await;
        client.graph_begin().await.expect("graph_begin");
        for i in 0..1000 {
            let e = if i % 2 == 0 { elems_a } else { elems_b };
            let nxt = alloc(&mut client, e).await;
            let m = add_msg_owned(&cur, &b, nxt.id, e);
            client.launch(m).await.expect("captured launch");
            cur = nxt;
        }
        let gid = client.graph_end().await.expect("graph_end");
        for _ in 0..3 { client.graph_launch(gid).await.expect("warm"); }
        client.sync().await.unwrap();
        let t0 = Instant::now();
        for _ in 0..20 { client.graph_launch(gid).await.expect("replay"); }
        client.sync().await.unwrap();
        let wall = (Instant::now() - t0).as_secs_f64() * 1e3 / 20.0;
        eprintln!(
            "[probe] hetero N=1000: replay {wall:.2}ms → {:.2} µs/节点",
            wall * 1e3 / 1000.0
        );
    }

    // C.    // C. distinct 内核轮转(K=50:模块/函数切换成本)
    let k = 50;
    let sources: Vec<String> = (0..k)
        .map(|i| {
            format!(
                r#"extern "C" __global__ void probe_add_d{i}(
                    const float* a, const float* b, float* out, const size_t n) {{
                    int i2 = blockIdx.x * blockDim.x + threadIdx.x;
                    if (i2 < (int)n) {{ out[i2] = a[i2] + b[i2] + {i}.0f; }}
                }}"#
            )
        })
        .collect();
    let elems = 256;
    let a = alloc(&mut client, elems).await;
    let b = alloc(&mut client, elems).await;
    let mut cur = alloc(&mut client, elems).await;
    client.graph_begin().await.expect("graph_begin");
    for i in 0..1000 {
        let nxt = alloc(&mut client, elems).await;
        let ki = i % k;
        let msg = LaunchMsg {
            kernel: KernelSource { name: format!("probe_add_d{ki}"), source: sources[ki].clone() },
            args: vec![
                Arg::Block { id: cur.id },
                Arg::Block { id: b.id },
                Arg::Block { id: nxt.id },
                Arg::U64(elems as u64),
            ],
            grid: (((elems as u32) + 255) / 256, 1, 1),
            block: (256, 1, 1),
            shared_mem: 0,
            out_elems: elems,
        };
        client.launch(msg).await.expect("captured launch");
        cur = nxt;
    }
    let gid = client.graph_end().await.expect("graph_end");
    for _ in 0..3 {
        client.graph_launch(gid).await.expect("warm");
    }
    client.sync().await.unwrap();
    let t0 = Instant::now();
    for _ in 0..20 {
        client.graph_launch(gid).await.expect("replay");
    }
    client.sync().await.unwrap();
    let wall = (Instant::now() - t0).as_secs_f64() * 1e3 / 20.0;
    eprintln!(
        "[probe] distinct K={k} N=1000: replay {wall:.2}ms → {:.2} µs/节点",
        wall * 1e3 / 1000.0
    );

    // D. eager 对照(host 直发,每 launch 一次 ack 往返)
    let n_nodes = 1000;
    let t0 = Instant::now();
    for _ in 0..3 {
        let b = alloc(&mut client, elems).await;
        let mut cur = alloc(&mut client, elems).await;
        for _ in 0..n_nodes {
            let nxt = alloc(&mut client, elems).await;
            client.launch(add_msg_owned(&cur, &b, nxt.id, elems)).await.expect("eager launch");
            cur = nxt;
        }
        client.sync().await.unwrap();
    }
    let eager = (Instant::now() - t0).as_secs_f64() * 1e3 / 3.0;
    eprintln!("[probe] eager N={n_nodes}: {eager:.2}ms/链 → {:.2} µs/节点", eager * 1e3 / n_nodes as f64);
}

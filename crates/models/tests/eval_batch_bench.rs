//! **解释器批处理 A/B 夹具**(2026-10-21;诊断用,不做硬门)。
//!
//! 用途:同一张合成图,在 `queue/flush` 转换前后各跑一次,打印宿主侧耗时。
//! 关键变量 = `OWL_BLOCK_POOL`(块池预算;池开 ⇒ 逐节点 alloc 退化为零命令,
//! 发射才能真的攒成批;池关 ⇒ 每个节点前面必有一次 `flush`,批长退回 1)。
//!
//! 运行:
//! ```text
//! OWL_TEST_DEVICE=01:00.0 OWL_BLOCK_POOL=$((64<<20)) \
//!   cargo test -p owl-models --test eval_batch_bench -- --nocapture
//! ```

use owl_cuda::{test_device_ordinal_opt, Command as Envelope, DeviceSelector, GpuClient, GpuServer};
use owl_iface::contract::{DeviceClient, Dtype};
use owl_iface::handle::DeviceHandle;
use owl_models::env::EnvProvider;
use owl_models::interpreters::eval_ops_scoped_env;
use owl_models::tensor::TensorOps;

fn assemble() -> Option<DeviceHandle<GpuClient>> {
    let ordinal = test_device_ordinal_opt()?;
    let (tx, rx) = std::sync::mpsc::channel::<Envelope>();
    let server = GpuServer::new(rx, DeviceSelector::Ordinal(ordinal), None);
    let client = GpuClient::new(tx);
    std::thread::Builder::new()
        .name("owl-eval-bench".into())
        .spawn(move || {
            if let Err(e) = server.run() {
                eprintln!("[eval_batch_bench] server 退出(Err): {e}");
            }
        })
        .expect("起 server 线程失败");
    Some(DeviceHandle::new(client, None))
}

fn le_f32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

#[tokio::test]
async fn eval_host_cost_probe() {
    let Some(mut face) = assemble() else {
        eprintln!("OWL_TEST_DEVICE 未设 = skip");
        return;
    };
    const N: usize = 256;
    const DEPTH: usize = 200;

    let bv: Vec<f32> = (0..N).map(|i| 1.0 + i as f32 * 1e-3).collect();
    let b = TensorOps::from_host(Dtype::F32, vec![N], &le_f32(&bv));
    let mut x = TensorOps::from_host(Dtype::F32, vec![N], &le_f32(&bv));
    for _ in 0..DEPTH {
        x = x.add(&b);
    }

    // 预热(核缓存/池暖机)
    let (root, cands) = eval_ops_scoped_env(&x, &mut face, EnvProvider::default(), None)
        .await
        .expect("warmup eval");
    face.free(&cands).await.expect("free cands");

    let mut best = f64::MAX;
    for _ in 0..3 {
        let t0 = std::time::Instant::now();
        let (root, cands) = eval_ops_scoped_env(&x, &mut face, EnvProvider::default(), None)
            .await
            .expect("eval");
        let dt = t0.elapsed().as_secs_f64();
        face.free(&cands).await.expect("free cands");
        best = best.min(dt);
        let _ = &root;
    }
    let per_node_us = best * 1e6 / (DEPTH + 1) as f64;
    println!(
        "[eval 账] 池预算={} MiB · {DEPTH} 节点 add:总 {:.2}ms / {:.2}µs 每节点",
        std::env::var("OWL_BLOCK_POOL").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(0) >> 20,
        best * 1e3,
        per_node_us
    );
    let _ = root;
}

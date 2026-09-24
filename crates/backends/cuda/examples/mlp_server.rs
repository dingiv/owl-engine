//! mlp 样例(GPU server 执行器):与 models/examples/mlp.rs 共享同一份管线
//! (owl_models::demo::mlp_pipeline)——注入 GpuClient(server),真发射。

use owl_cuda::gpu_server::GpuClient;
use owl_models::client::DeviceClient as _;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ordinal = std::env::var("OWL_TEST_DEVICE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    // 注入 GPU server 执行器:同一份管线,CPU 版见 models/examples/mlp.rs
    let mut client = GpuClient::spawn(ordinal)?;
    client.sync().await.expect("client sync");
    println!("== mlp on GPU client (device {ordinal}) ==");

    let out = owl_models::demo::mlp_pipeline(&mut client).await?;
    println!("forward 输出 = {out:?}");
    println!("demo: 声明式 MLP 端到端 ✓(GpuClient;GPU == CPU 同一管线)");
    Ok::<(), Box<dyn std::error::Error>>(())
}

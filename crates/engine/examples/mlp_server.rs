//! mlp 端到端样例(engine 侧):同一份声明管线(owl_models::demo::mlp_pipeline)
//! 注入 owl-cuda 的 GpuClient 真发射——模型层/解释层/硬件层三层贯通的参考样例。
//!
//! ⚠️ 依赖 engine lib 编译通过(A4 迁移完成后方可 cargo run);
//! 同源路径的独立验证见 `cargo run -p owl-cuda --example kernel_node`。

use owl_cuda::{DeviceSelector, GpuClient};
use owl_models::client::DeviceClient as _;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ordinal = std::env::var("OWL_TEST_DEVICE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    // 注入 GPU server 执行器:同一份管线,CPU 版见 models/examples/mlp.rs
    let mut client = GpuClient::spawn(DeviceSelector::Ordinal(ordinal))?;
    client.sync().await.expect("client sync");
    println!("== mlp on GPU client (device {ordinal}) ==");

    let out = owl_models::demo::mlp_pipeline(&mut client).await?;
    println!("forward 输出 = {out:?}");
    client.close().await?;
    println!("demo: 声明式 MLP 端到端 ✓(GpuClient;GPU == CPU 同一管线)");
    Ok::<(), Box<dyn std::error::Error>>(())
}

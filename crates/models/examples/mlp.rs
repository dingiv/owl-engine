//! mlp 样例(CPU 执行器):与 owl-cuda 的 mlp_server 共享同一份管线
//! (owl_models::demo::mlp_pipeline)——注入 CpuFace,CPU 上归约。

use owl_models::demo::mlp_pipeline;
use owl_models::interpreter::CpuFace;

#[tokio::main]
async fn main() {
    // 注入 CPU 执行器:同一份管线,GPU 版见 owl-cuda/examples/mlp_server.rs
    let mut face = CpuFace::new();
    let out = mlp_pipeline(&mut face).await.expect("mlp_pipeline");
    println!("forward 输出 = {out:?}");
    println!("demo: 声明式 MLP 端到端 ✓(CpuFace;async 边界)");
}

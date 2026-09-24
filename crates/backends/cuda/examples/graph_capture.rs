//! 三固定流 + CUDA Graph 捕获/重放验收(server 护栏演示):
//!
//! 1. 流是固定三条(H2D/COMPUTE/D2H),客户端不可自创;
//! 2. warmup 契约:先 eager 跑同一声明树,验证逻辑正确;
//! 3. graph_begin → 捕获期重发同一声明(Alloc 自动切 slab,零 cudaMalloc)
//!    → graph_end(空捕获窗会被哨兵③拒绝);
//! 4. graph_launch ×N 重放,数据经 dtoh 验证;
//! 5. 捕获期违规命令(其他流任务/同步/搬运)被 server 结构化拒绝。

use owl_cuda::gpu_server::{Command, GpuClient, GpuServer};
use owl_models::client::{DeviceClient as _, GraphId};
use owl_models::shape::Shape;
use owl_models::{Dtype, Kernel, TensorOps};

const SCALE_CU: &str = r#"
extern "C" __global__ void owl_scale_f32(
    const float* x, float k, const size_t n, float* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (int)n) { out[i] = x[i] * k; }
}
"#;

/// x(Block 叶子)* k 的声明(捕获与重放共用同一棵树、同一批块)
fn scale_decl(x_block: u64, k: f32) -> TensorOps {
    TensorOps::of(Kernel::new("owl_scale_f32", SCALE_CU))
        .with_shape(Dtype::F32, Shape::from(vec![4]))
        .arg(&TensorOps::of_block(x_block, Dtype::F32, Shape::from(vec![4])))
        .arg_f32(k)
        .arg_usize(4)
}

async fn eval_and_harvest(client: &mut GpuClient, decl: &TensorOps) -> Vec<f32> {
    let out = owl_models::client::eval(decl, client).await.expect("eval");
    let n: usize = decl.shape().iter().product();
    let mut buf = vec![0u8; n * 4];
    client.dtoh(&out, &mut buf).await.expect("dtoh");
    buf.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ordinal = std::env::var("OWL_TEST_DEVICE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    // 手工组装:管道外部创建,server/client 各拿一端
    let (tx, rx) = std::sync::mpsc::channel::<Command>();
    let server = GpuServer::new(rx, ordinal, None);
    let mut client = GpuClient::new(tx);
    std::thread::Builder::new()
        .name("owl-gpu-manual".into())
        .spawn(move || server.run().expect("server run"))?;

    // ---- 数据物化(H2D 流;图外,先于捕获)----
    let xs: Vec<u8> = [1.0f32; 4].iter().flat_map(|f| f.to_le_bytes()).collect();
    let xb = client.htod(Dtype::F32, &Shape::from(vec![4]), &xs).await?;

    // ---- ① warmup 契约:eager 先跑同一声明树 ----
    let warm = eval_and_harvest(&mut client, &scale_decl(xb.id, 3.0)).await;
    println!("warmup(eager) = {warm:?}");
    assert!(warm.iter().all(|&v| (v - 3.0).abs() < 1e-6));

    // ---- ② 捕获:重发同一声明(Alloc 切 slab;Launch 进图)----
    client.graph_begin().await?;
    let cap_out = owl_models::client::eval(&scale_decl(xb.id, 3.0), &mut client)
        .await
        .expect("captured eval");
    let gid: GraphId = client.graph_end().await?;
    println!("图捕获完成 gid={gid}");

    // ---- ③ 重放 ×3,收割验证(指针稳定:slab 块随账房保活)----
    for i in 0..3 {
        client.graph_launch(gid).await?;
        let mut buf = vec![0u8; 16];
        client.dtoh(&cap_out, &mut buf).await?;
        let got: Vec<f32> = buf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        println!("replay #{i} = {got:?}");
        assert!(got.iter().all(|&v| (v - 3.0).abs() < 1e-6), "replay 不符: {got:?}");
    }

    // ---- ④ 护栏:捕获期违规被结构化拒绝 ----
    client.graph_begin().await?;
    // 捕获期在 D2H 流上搬数据 → 拒绝
    let rejected = client.dtoh(&cap_out, &mut vec![0u8; 16]).await;
    assert!(rejected.is_err(), "捕获期跨流 dtoh 应被拒");
    println!("捕获期违规拒绝 ✓({})", rejected.err().unwrap());
    // 该窗零发射 → graph_end 走哨兵③拒绝
    let empty = client.graph_end().await;
    assert!(empty.is_err(), "空捕获窗应被哨兵③拒绝");
    println!("空捕获窗哨兵③ ✓({})", empty.err().unwrap());
    let guard = client.graph_launch(gid).await;
    assert!(guard.is_ok());

    client.sync().await?;
    println!("demo: 三固定流(server 内部路由)+ 图捕获/重放 + server 护栏 ✓");
    Ok::<(), Box<dyn std::error::Error>>(())
}

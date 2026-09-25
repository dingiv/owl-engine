//! Kernel 节点端到端(动作表二期验收):
//! 用户自定义 kernel(`TensorOps::of(kernel).arg(..)`)→ eval → GpuClient 真发射。
//! 同时验证 grid 哨兵(自动 1D)与 with_launch 显式发射配置两条路径。

use owl_cuda::{Command, DeviceSelector, GpuClient, GpuServer};
use owl_models::contract::DeviceClient as _;
use owl_models::Dtype;
use owl_models::contract::Shape;
use owl_models::{Kernel, TensorOps};

// 槽序契约:标量/输入参数按声明序,输出块固定在最后一个形参
const SCALE_CU: &str = r#"
extern "C" __global__ void owl_scale_f32(
    const float* x, float k, const size_t n, float* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (int)n) { out[i] = x[i] * k; }
}
"#;

// 逃生舱签名:非注册 kernel 的槽序权威(T,f32,sz + 末位输出 T)

fn host(shape: &[usize], v: &[f32]) -> TensorOps {
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for f in v {
        bytes.extend_from_slice(&f.to_le_bytes());
    }
    TensorOps::from_host(Dtype::F32, Shape::from(shape.to_vec()), &bytes)
}

async fn run(client: &mut GpuClient, decl: &TensorOps) -> Vec<f32> {
    let out = owl_models::interpreter::eval_ops(decl, client).await.expect("eval");
    client.sync().await.expect("sync");
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
    let server = GpuServer::new(rx, DeviceSelector::Ordinal(ordinal), None);
    let mut client = GpuClient::new(tx);
    std::thread::Builder::new()
        .name("owl-gpu-manual".into())
        .spawn(move || server.run().expect("server run"))?;

    // ① grid 哨兵(自动 1D ceil/256)
    let x = host(&[4], &[1.0, 1.0, 1.0, 1.0]);
    let decl = TensorOps::of(Kernel::new("owl_scale_f32", SCALE_CU).with_sig("T,f32,sz,T"))
        .with_shape(Dtype::F32, Shape::from(vec![4]))
        .arg(&x)
        .arg_f32(3.0)
        .arg_usize(4);
    let got = run(&mut client, &decl).await;
    println!("scale(自动 grid) = {got:?}");
    assert!(got.iter().all(|&v| (v - 3.0).abs() < 1e-6), "结果不符: {got:?}");

    // ② 显式发射配置(with_launch)
    let x2 = host(&[4], &[2.0, 2.0, 2.0, 2.0]);
    let decl2 = TensorOps::of(Kernel::new("owl_scale_f32", SCALE_CU).with_sig("T,f32,sz,T").with_launch((1, 1, 1), (256, 1, 1), 0))
        .with_shape(Dtype::F32, Shape::from(vec![4]))
        .arg(&x2)
        .arg_f32(0.5)
        .arg_usize(4);
    let got2 = run(&mut client, &decl2).await;
    println!("scale(显式 launch) = {got2:?}");
    assert!(got2.iter().all(|&v| (v - 1.0).abs() < 1e-6), "结果不符: {got2:?}");

    client.sync().await?;
    println!("demo: Kernel 节点(动作表二期)端到端 ✓");
    Ok::<(), Box<dyn std::error::Error>>(())
}

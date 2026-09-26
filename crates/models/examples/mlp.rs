//! mlp 端到端样例:一份声明管线,双执行器同一份代码(原 src/demo.rs 整体迁入)。
//!
//! 执行器注入(相位/后端无感的验收形态):
//! - 默认:CpuFace(CPU 参考执行器;零 GPU 依赖)
//! - `gpu` 参数:GpuClient(owl-cuda server;真发射)——同一份管线先跑
//!   CPU 基准再跑 GPU,逐元素 allclose 对拍。"server 哑执行器 + 声明链"
//!   契约的端到端验收:换 face 即换后端,声明代码一行不改。
//!
//! 用法:
//! ```text
//! cargo run -p owl-models --example mlp          # CPU face
//! OWL_TEST_DEVICE=3 cargo run -p owl-models --example mlp -- gpu
//! ```

use owl_models::contract::DeviceClient;
use owl_models::contract::ModelError;
use owl_cpu::CpuFace;
use owl_models::interpreters::eval;
use owl_models::module::ForwardCtx;
use owl_models::tensor::Dtype;
use owl_models::TensorOps;

// ============================================================================
// 共享管线(自 src/demo.rs 迁入;example 私有)
// ============================================================================

fn f32b(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// 共享管线:容器 → LoaderOps 描述 → eval_load 装填 → mount → forward → eval。
/// face 注入执行器(CpuFace = CPU;GpuClient = GPU server)。
/// 同一份描述,换 face 即换后端——这就是"相位/后端无感"的验收形态。
/// 共享管线:声明 MLP → forward → eval 归约 → dtoh 收割 f32 输出。
/// face 注入执行器(CpuFace = CPU;GpuClient = GPU server)。
/// 同一份描述,换 face 即换后端——这就是"相位/后端无感"的验收形态。
async fn mlp_pipeline<D: DeviceClient>(face: &mut D) -> Result<Vec<f32>, ModelError> {
    let (hidden, intermediate) = (4usize, 8usize);

    // new = 准备容器(空包指针);layout = 唯一生命周期钩子(布局声明);
    // eval_load = 执行器(取数 → 物化 → 自动填空包)
    let mlp = owl_models::layers::mlp::Mlp::new(hidden, intermediate);
    let src = std::collections::HashMap::from([
        ("gate_proj".to_string(), vec![0.1; hidden * intermediate]),
        ("up_proj".to_string(), vec![0.2; hidden * intermediate]),
        ("down_proj".to_string(), vec![0.3; intermediate * hidden]),
    ]);
    // 装载执行(层入口:解释器驱动 layout + 物化 + 回填)
    owl_models::interpreters::eval_load(&mlp, face, &src, &Default::default()).await?;

    // 输入(host → 设备,声明进树)
    let xvec = vec![0.5f32; hidden];
    let x = TensorOps::from_host(Dtype::F32, vec![1, hidden], &f32b(&xvec));

    // 计算执行(层入口:解释器驱动 forward + 归约)
    let bytes = eval(&mlp, &x, face, &ForwardCtx::minimal(1)).await?;
    let n: usize = hidden;
    let mut out = vec![0u8; n * 4];
    face.dtoh(&bytes, &mut out).await?;
    Ok(out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

// ============================================================================
// 入口:backend 选择 + CPU/GPU 对拍
// ============================================================================

fn allclose(a: &[f32], b: &[f32], atol: f32) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= atol)
}

#[tokio::main]
async fn main() {
    let gpu = std::env::args().any(|a| a == "gpu" || a == "--gpu");

    // 同一份管线的 CPU 基准(两模式都跑;gpu 模式兼作对拍锚)
    let cpu_out = mlp_pipeline(&mut CpuFace::new()).await.expect("cpu pipeline");
    if !gpu {
        println!("forward 输出 = {cpu_out:?}");
        println!("demo: 声明式 MLP 端到端 ✓(CpuFace;async 边界)");
        return;
    }

    // GPU server face:owl-cuda actor(spawn = 建管道 + 起线程 + boot 握手)
    use owl_cuda::{test_device_ordinal, DeviceSelector, GpuClient};
    let mut client = GpuClient::spawn(DeviceSelector::Ordinal(test_device_ordinal()))
        .expect("gpu server boot");
    let gpu_out = mlp_pipeline(&mut client).await.expect("gpu pipeline");
    client.close().await.expect("server 关机");

    // 对拍:同一份声明,两个执行器必须逐元素一致
    // (CPU 朴素累加 vs GPU f32 kernel,1e-5 绝对容差;参考 testkit allclose 口径)
    assert!(
        allclose(&cpu_out, &gpu_out, 1e-5),
        "CPU/GPU 对拍失配:\n  cpu = {cpu_out:?}\n  gpu = {gpu_out:?}"
    );
    println!("forward 输出 = {gpu_out:?}");
    println!("demo: 声明式 MLP 端到端 ✓(GpuClient;GPU == CPU 同一管线,allclose 1e-5)");
}

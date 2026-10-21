//! **解释器累积批门(真卡)**(2026-10-21):`eval_ops` 的发射改走 `queue`,
//! 收尾一次 `flush` —— 本门验"数值不变 + 真在 GPU 上跑通"。
//!
//! 为什么放在 owl-models:`eval` 是这一层的资产,而 owl-cuda 已在 dev-deps
//! (样例驱动 server)。本门只依赖公开面:解释器 + DeviceClient + GpuClient。
//!
//! 运行:`OWL_TEST_DEVICE=01:00.0 cargo test -p owl-models --test eval_batch_gate`
//! (未设 = skip;不隐式选卡。)

use owl_cuda::{test_device_ordinal_opt, Command as Envelope, DeviceSelector, GpuClient, GpuServer};
use owl_iface::contract::{Bytes, DeviceClient, Dtype, ModelError};
use owl_iface::handle::DeviceHandle;
use owl_models::env::EnvProvider;
use owl_models::interpreters::eval_ops_env;
use owl_models::tensor::TensorOps;

fn assemble() -> Option<DeviceHandle<GpuClient>> {
    let ordinal = test_device_ordinal_opt()?;
    let (tx, rx) = std::sync::mpsc::channel::<Envelope>();
    let server = GpuServer::new(rx, DeviceSelector::Ordinal(ordinal), None);
    let client = GpuClient::new(tx);
    std::thread::Builder::new()
        .name("owl-eval-batch".into())
        .spawn(move || {
            if let Err(e) = server.run() {
                eprintln!("[eval_batch_gate] server 退出(Err): {e}");
            }
        })
        .expect("起 server 线程失败");
    Some(DeviceHandle::new(client, None))
}

fn le_f32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn le_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// 三节点图(`(a*b) + b`):两个 Htod 叶子 + 两个 Call 节点 ——
/// 解释器会把两条发射 `queue` 起来,收尾一次 `flush`。
#[tokio::test]
async fn interpreter_queue_flush_keeps_values() {
    let Some(mut face) = assemble() else {
        eprintln!("OWL_TEST_DEVICE 未设 = skip(不隐式选卡)");
        return;
    };

    const N: usize = 129; // 非 256 对齐:顺带验尾块
    let av: Vec<f32> = (0..N).map(|i| (i as f32) * 0.5 - 3.0).collect();
    let bv: Vec<f32> = (0..N).map(|i| 1.25 * (i as f32) + 7.0).collect();

    let a = TensorOps::from_host(Dtype::F32, vec![N], &le_f32(&av));
    let b = TensorOps::from_host(Dtype::F32, vec![N], &le_f32(&bv));
    let prod = a.mul(&b);
    let root = prod.add(&b);

    let out: Bytes = eval_ops_env(&root, &mut face, EnvProvider::default(), None)
        .await
        .expect("解释器求值(GPU)");

    let mut buf = vec![0u8; N * 4];
    face.dtoh(&out, &mut buf).await.expect("dtoh 根");
    let got = le_to_f32(&buf);
    for i in 0..N {
        let want = av[i] * bv[i] + bv[i];
        assert!(
            (got[i] - want).abs() <= 1e-3 * want.abs().max(1.0),
            "i={i} 实得 {} 期望 {want}",
            got[i]
        );
    }

    // 收尾:句柄队列必须已排空(否则后续命令会被"未提交"拒)
    let empty = face.flush().await.expect("收尾 flush 应 Ok");
    assert!(empty.outcomes.is_empty(), "解释器收尾应已把批提交干净");
    face.free(&[out.id]).await.expect("free");
}

/// **回归哨**:解释器内若有人再写 `face.launch(..).await`(绕过 queue),
/// 会在队列非空时被命令面**响亮拒** —— 本门跑一个两层图,只要有一条漏网就红。
#[tokio::test]
async fn interpreter_never_bypasses_queue() {
    let Some(mut face) = assemble() else { return };
    const N: usize = 64;
    let av: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let bv: Vec<f32> = (0..N).map(|i| 2.0 + i as f32).collect();
    let a = TensorOps::from_host(Dtype::F32, vec![N], &le_f32(&av));
    let b = TensorOps::from_host(Dtype::F32, vec![N], &le_f32(&bv));
    let mut x = a.add(&b);
    for _ in 0..3 {
        x = x.mul(&b);
    }
    let out = eval_ops_env(&x, &mut face, EnvProvider::default(), None).await;
    match out {
        Ok(b) => {
            let mut buf = vec![0u8; N * 4];
            face.dtoh(&b, &mut buf).await.expect("dtoh");
            let got = le_to_f32(&buf);
            let want = (av[0] + bv[0]) * bv[0].powi(3);
            assert!(
                (got[0] - want).abs() <= 1e-2 * want.abs().max(1.0),
                "实得 {} 期望 {want}",
                got[0]
            );
            face.free(&[b.id]).await.expect("free");
        }
        Err(ModelError::Msg(m)) if m.contains("未提交") => {
            panic!("解释器里有发射绕过了 queue(命令面拒绝):{m}")
        }
        Err(e) => panic!("求值失败:{e}"),
    }
}

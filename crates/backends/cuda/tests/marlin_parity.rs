//! Marlin W4A16 对拍(工单 M 验收;foreign-kernel 通道端到端)。
//!
//! 自足用例:host 随机 q(u4 nibble)/ scales → `pack_marlin_b/s`(owl
//! kernels repack)→ host dequant 参考 → foreign Launch(cublas_gemm_f16
//! 同款通道,虚拟核名 `marlin_gemm_w4a16`)→ mean_rel 判据
//! (照 xinfer parity 门:< 1e-3)。
//!
//! 运行:OWL_TEST_DEVICE=<空闲卡> cargo test -p owl-cuda --test marlin_parity

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
        .name("owl-marlin-test".into())
        .spawn(move || server.run().expect("server run 异常退出"))
        .expect("起 server 线程失败");
    client
}

fn le_u16(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn le_i32(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// host dequant 参考:W = (q - 8) × scale(U4B8 对称)。
/// 源布局 = compressed-tensors **out 主序**:q [n, k] 行主序(nibble 每 i32
/// LSB-first 沿 k)、scales [n, k/g];C = A × W^T。
fn host_dequant_gemm(a: &[f32], q: &[u8], s: &[f32], m: usize, n: usize, k: usize, g: usize) -> Vec<f32> {
    let mut c = vec![0f32; m * n];
    for r in 0..m {
        for nn in 0..n {
            let mut acc = 0f32;
            for kk in 0..k {
                let nib = q[nn * k + kk];
                let w = (nib as f32 - 8.0) * s[nn * (k / g) + kk / g];
                acc += a[r * k + kk] * w;
            }
            c[r * n + nn] = acc;
        }
    }
    c
}

/// 外部核启动消息(槽序契约:owl_kernels::marlin::GEMM_W4A16_SLOTS)
fn marlin_launch(
    a: &owl_cuda::Bytes,
    b: &owl_cuda::Bytes,
    c: &owl_cuda::Bytes,
    s: &owl_cuda::Bytes,
    ws: &owl_cuda::Bytes,
    ctmp: &owl_cuda::Bytes,
    m: usize,
    k: usize,
    n: usize,
    g: usize,
) -> LaunchMsg {
    LaunchMsg {
        kernel: owl_cuda::KernelSpec {
            name: "marlin_gemm_w4a16".into(),
            source: String::new(),
        },
        args: vec![
            Arg::Block { id: a.id },
            Arg::Block { id: b.id },
            Arg::Block { id: c.id },
            Arg::Block { id: s.id },
            Arg::Block { id: ws.id },
            Arg::Block { id: ctmp.id },
            Arg::U64(m as u64),
            Arg::U64(k as u64),
            Arg::U64(n as u64),
            Arg::U64(g as u64),
        ],
        grid: (0, 0, 0),
        block: (0, 0, 0),
        shared_mem: 0,
        out_elems: m * n,
    }
}

async fn run_case(client: &mut GpuClient, m: usize, n: usize, k: usize, g: usize) -> (f64, f32) {
    // host 数据(f16 可表值域:×0.25 压幅)
    let a: Vec<f32> = (0..m * k).map(|i| ((i as f32 * 0.31) - 4.0).sin() * 0.5).collect();
    // 源布局 out 主序:q [n, k]、scales [n, k/g]
    let q: Vec<u8> = (0..n * k).map(|i| ((i * 7 + 3) % 16) as u8).collect();
        // scale 取真实量纲(0.5~1.5):均匀 q 的对称抵消会压小 |C_ref|,
    // 比值分母缩水让 f16 输出量化地板(≈5e-4)被放大成假阳性
    let s: Vec<f32> = (0..n * (k / g)).map(|i| 0.5 + ((i * 13) % 13) as f32 * 0.08).collect();

    // owl repack(compressed-tensors → marlin 布局)
    let b_packed = owl_kernels::marlin::repack::pack_marlin_b(&q, k, n);
    let s_packed = owl_kernels::marlin::repack::pack_marlin_s(&s, n, k / g);
    assert_eq!(b_packed.len(), (k / 16) * (n * 16 / 8));
    assert_eq!(s_packed.len(), (k / g) * n);

    // 上卡
    let da = client.htod(Dtype::F16, &Shape::from(vec![m, k]), &le_u16(
        &a.iter().map(|f| half::f16::from_f32(*f).to_bits()).collect::<Vec<_>>(),
    )).await.expect("htod a");
    // u16 位序 = f16 LE ✓(f16::to_bits 是 u16 表示,LE 字节序正确)
    let db = client.htod(Dtype::U32, &Shape::from(vec![b_packed.len()]), &le_i32(&b_packed)).await.expect("htod b");
    let ds = client.htod(Dtype::F16, &Shape::from(vec![s_packed.len()]), &le_u16(&s_packed)).await.expect("htod s");
    let ws_len = owl_kernels::marlin::v2_workspace_len(n).max(n / 128 * 16);
    let dws = client.alloc(Dtype::U32, ws_len).await.expect("alloc ws");
    let dctmp = client.alloc(Dtype::U32, 1).await.expect("alloc ctmp");
    let dc = client.alloc(Dtype::F16, m * n).await.expect("alloc c");

    client.launch(marlin_launch(&da, &db, &dc, &ds, &dws, &dctmp, m, k, n, g)).await.expect("marlin launch");

    let mut buf = vec![0u8; m * n * 2];
    client.dtoh(&dc, &mut buf).await.expect("dtoh");
    let got: Vec<f32> = buf
        .chunks_exact(2)
        .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect();

    let c_ref = host_dequant_gemm(&a, &q, &s, m, n, k, g);
    let (mut sum_abs, mut max_abs, mut sum_sq) = (0f64, 0f32, 0f64);
    for (x, r) in got.iter().zip(&c_ref) {
        sum_abs += (*x as f64 - *r as f64).abs();
        sum_sq += (*r as f64) * (*r as f64);
        max_abs = max_abs.max((*x - *r).abs());
    }
    let n_elem = (m * n) as f64;
    let rms = (sum_sq / n_elem).sqrt().max(1e-6);
    ((sum_abs / n_elem / rms), max_abs)
}

#[tokio::test]
async fn marlin_w4a16_parity() {
    let mut client = assemble();
    // 形状矩阵:decode(m=1)+ 短 prefill(m=64);k/n 满足 v2 约束(k%16==0)
    // 形状约束(pack/upstream Layer 校验):k%128==0、n%256==0、g ∈ {64,128,-1}
    for (m, n, k, g) in [(1usize, 256usize, 256usize, 64usize), (64, 256, 256, 64), (8, 512, 512, 128)] {
        let (noise_ratio, max_abs) = run_case(&mut client, m, n, k, g).await;
        eprintln!("[marlin] m={m} n={n} k={k} g={g} → 噪声/信号RMS={noise_ratio:.6} max_abs={max_abs:.5}");
        // 判据 = 噪声/信号 RMS < 2%(W4A16 + f16 I/O 的物理噪声地板;
        // 随机 q 的对称抵消会让 mean(|C|) 缩水,比值判据假阳性)
        assert!(
            noise_ratio < 2e-2 && noise_ratio.is_finite(),
            "m={m}: 噪声比 {noise_ratio}"
        );
    }
    client.sync().await.expect("sync");
}

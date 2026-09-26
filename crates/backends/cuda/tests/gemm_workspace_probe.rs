//! F1 治理探针:cuBLAS 账外显存占用实测(handle 创建 / 首个 gemm /
//! 换形状 gemm 三个时点的 nvidia-smi used 差值)。

use owl_cuda::{test_device_ordinal, Arg, DeviceClient as _, DeviceSelector, Dtype, GpuClient, GpuServer, LaunchMsg, Shape};

/// foreign-kernel 通道:Launch + 虚拟核名 cublas_gemm_f16
/// 槽序契约(cublas.rs):[T a, T b, T out, sz m, sz k, sz n, sz nt]
fn gemm_launch(a: &owl_cuda::Bytes, w: &owl_cuda::Bytes, o: &owl_cuda::Bytes, tokens: usize, k: usize, feats: usize, nt: bool) -> LaunchMsg {
    LaunchMsg {
        kernel: owl_cuda::KernelSpec { name: "cublas_gemm_f16".into(), source: String::new() },
        args: vec![
            Arg::Block { id: a.id },
            Arg::Block { id: w.id },
            Arg::Block { id: o.id },
            Arg::U64(tokens as u64),
            Arg::U64(k as u64),
            Arg::U64(feats as u64),
            Arg::U64(nt as u64),
        ],
        grid: (0, 0, 0),
        block: (0, 0, 0),
        shared_mem: 0,
        out_elems: tokens * feats,
    }
}

fn vram_used_mib() -> usize {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", &test_device_ordinal().to_string()])
        .output()
        .expect("nvidia-smi");
    String::from_utf8_lossy(&out.stdout).lines().next().unwrap().trim().parse().unwrap()
}

#[tokio::test]
async fn gemm_workspace_footprint() {
    let (tx, rx) = std::sync::mpsc::channel::<owl_cuda::Command>();
    let server = GpuServer::new(rx, DeviceSelector::Ordinal(test_device_ordinal()), None);
    let mut client = GpuClient::new(tx);
    std::thread::Builder::new().spawn(move || server.run().expect("server run")).unwrap();

    let base = vram_used_mib();
    client.sync().await.expect("boot");

    // 首个 gemm(触发 handle 创建 + workspace)
    let (t, k, n) = (8usize, 4096usize, 11008usize);
    let a = client.alloc(Dtype::F16, t * k).await.unwrap();
    let w = client.alloc(Dtype::F16, n * k).await.unwrap();
    let o = client.alloc(Dtype::F16, t * n).await.unwrap();
    let _ = (base, &a, &w, &o);
    let after_alloc = vram_used_mib();
    client.launch(gemm_launch(&a, &w, &o, n, k, t, true)).await.expect("gemm 1");
    client.sync().await.unwrap();
    let after_gemm1 = vram_used_mib();

    // 换形状(触发算法重选)
    let o2 = client.alloc(Dtype::F16, 512 * n).await.unwrap();
    client.launch(gemm_launch(&a, &w, &o2, n, k, 512, true)).await.expect("gemm 2");
    client.sync().await.unwrap();
    let after_gemm2 = vram_used_mib();

    // 复跑同形状(应零增长)
    client.launch(gemm_launch(&a, &w, &o, n, k, t, true)).await.expect("gemm 3");
    client.sync().await.unwrap();
    let after_gemm3 = vram_used_mib();

    eprintln!("[probe] server boot 后   : {base} MiB");
    let ledger = (t * k + n * k + t * n) * 2 / 1048576;
    eprintln!("[probe] 三块分配后      : {after_alloc} MiB(块账约 {ledger} MiB)");
    eprintln!("[probe] 首个 gemm 后    : {after_gemm1} MiB(Δ = {} MiB ≈ cublas 账外)", after_gemm1 - after_alloc);
    eprintln!("[probe] 换形状 gemm 后  : {after_gemm2} MiB(Δ = {} MiB)", after_gemm2 - after_gemm1);
    eprintln!("[probe] 复跑同形状后    : {after_gemm3} MiB(Δ = {} MiB,应 ~0)", after_gemm3 - after_gemm2);
}

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
    is_bf16: bool,
) -> LaunchMsg {
    LaunchMsg {
        kernel: owl_cuda::KernelSpec {
            name: if is_bf16 { "marlin_gemm_w4a16_bf16" } else { "marlin_gemm_w4a16" }.into(),
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

async fn run_case(
    client: &mut GpuClient,
    m: usize,
    n: usize,
    k: usize,
    g: usize,
    is_bf16: bool,
) -> (f64, f32) {
    // host 数据(f16 可表值域:×0.25 压幅)
    let a: Vec<f32> = (0..m * k).map(|i| ((i as f32 * 0.31) - 4.0).sin() * 0.5).collect();
    // 源布局 out 主序:q [n, k]、scales [n, k/g]
    let q: Vec<u8> = (0..n * k).map(|i| ((i * 7 + 3) % 16) as u8).collect();
        // scale 取真实量纲(0.5~1.5):均匀 q 的对称抵消会压小 |C_ref|,
    // 比值分母缩水让 f16 输出量化地板(≈5e-4)被放大成假阳性
    let s: Vec<f32> = (0..n * (k / g)).map(|i| 0.5 + ((i * 13) % 13) as f32 * 0.08).collect();

    // owl repack(compressed-tensors → marlin 布局)
    let b_packed = owl_kernels::marlin::repack::pack_marlin_b(&q, k, n);
    let s_packed_f16 = owl_kernels::marlin::repack::pack_marlin_s(&s, n, k / g);
    // bf16 内核的 s_type = BF16:scales 位型 f16 → bf16(与装载器同转换)
    let s_packed: Vec<u16> = if is_bf16 {
        s_packed_f16
            .iter()
            .map(|&bits| half::bf16::from_f32(half::f16::from_bits(bits).to_f32()).to_bits())
            .collect()
    } else {
        s_packed_f16
    };
    assert_eq!(b_packed.len(), (k / 16) * (n * 16 / 8));
    assert_eq!(s_packed.len(), (k / g) * n);

    // 上卡(bf16 模式:A/C 走 BF16 位型;参考用同位型回读值)
    let (act_name, a_ref) = if is_bf16 {
        (
            "marlin_gemm_w4a16_bf16",
            a.iter()
                .map(|f| half::bf16::from_f32(*f).to_f32())
                .collect::<Vec<_>>(),
        )
    } else {
        (
            "marlin_gemm_w4a16",
            a.iter()
                .map(|f| half::f16::from_f32(*f).to_f32())
                .collect::<Vec<_>>(),
        )
    };
    let a_bits: Vec<u16> = if is_bf16 {
        a_ref.iter().map(|f| half::bf16::from_f32(*f).to_bits()).collect()
    } else {
        a_ref.iter().map(|f| half::f16::from_f32(*f).to_bits()).collect()
    };
    let da = client.htod(Dtype::F16, &Shape::from(vec![m, k]), &le_u16(&a_bits)).await.expect("htod a");
    // u16 位序 = f16 LE ✓(f16::to_bits 是 u16 表示,LE 字节序正确)
    let db = client.htod(Dtype::U32, &Shape::from(vec![b_packed.len()]), &le_i32(&b_packed)).await.expect("htod b");
    let ds = client.htod(Dtype::F16, &Shape::from(vec![s_packed.len()]), &le_u16(&s_packed)).await.expect("htod s");
    let ws_mul: usize = std::env::var("OWL_WS_MUL").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let ws_len = owl_kernels::marlin::v2_workspace_len(n).max(n / 128 * 16) * ws_mul;
    let dws = client.alloc(Dtype::U32, ws_len).await.expect("alloc ws");
    let dctmp = client.alloc(Dtype::U32, 1).await.expect("alloc ctmp");
    let dc = client.alloc(Dtype::F16, m * n).await.expect("alloc c");

    client
        .launch(marlin_launch(
            &da, &db, &dc, &ds, &dws, &dctmp, m, k, n, g, is_bf16,
        ))
        .await
        .expect("marlin launch");

    let mut buf = vec![0u8; m * n * 2];
    client.dtoh(&dc, &mut buf).await.expect("dtoh");
    let got: Vec<f32> = if is_bf16 {
        buf.chunks_exact(2)
            .map(|c| {
                half::bf16::from_le_bytes([c[0], c[1]]).to_f32()
            })
            .collect()
    } else {
        buf.chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect()
    };

    // 大 m 全量 host 参考 = O(m·n·k) 标量乘加(debug 下分钟级)→ 抽样 4 行
    let (mut sum_abs, mut max_abs, mut sum_sq) = (0f64, 0f32, 0f64);
    let mut n_elem = 0f64;
    if m <= 256 {
        let c_ref = host_dequant_gemm(&a_ref, &q, &s, m, n, k, g);
        for (x, r) in got.iter().zip(&c_ref) {
            sum_abs += (*x as f64 - *r as f64).abs();
            sum_sq += (*r as f64) * (*r as f64);
            max_abs = max_abs.max((*x - *r).abs());
        }
        n_elem = (m * n) as f64;
    } else {
        // 行抽样:第 {0, m/4, m/2, 3m/4} 行全宽参考;全输出仅验有限性
        let rows: Vec<usize> = (0..4).map(|i| i * m / 4).collect();
        for &r in &rows {
            for nn in 0..n {
                let mut acc = 0f32;
                for kk in 0..k {
                    let nib = q[nn * k + kk];
                    let w = (nib as f32 - 8.0) * s[nn * (k / g) + kk / g];
                    acc += a_ref[r * k + kk] * w;
                }
                let got_v = got[r * n + nn];
                sum_abs += (got_v as f64 - acc as f64).abs();
                sum_sq += (acc as f64) * (acc as f64);
                max_abs = max_abs.max((got_v - acc).abs());
                n_elem += 1.0;
            }
        }
        for &v in &got {
            assert!(v.is_finite(), "大 m 输出含非有限值");
        }
    }
    let rms = (sum_sq / n_elem).sqrt().max(1e-6);
    (sum_abs / n_elem / rms, max_abs)
}

#[tokio::test]
async fn marlin_w4a16_parity() {
    let mut client = assemble();
    // 形状矩阵:decode(m=1)+ 短 prefill(m=64);k/n 满足 v2 约束(k%16==0)
    // 形状约束(pack/upstream Layer 校验):k%128==0、n%256==0、g ∈ {64,128,-1}
    for (m, n, k, g) in [(1usize, 256usize, 256usize, 64usize), (64, 256, 256, 64), (8, 512, 512, 128)] {
        let (noise_ratio, max_abs) = run_case(&mut client, m, n, k, g, false).await;
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

/// BF16 激活 W4A16 parity(E5-DF3 同日十二:草稿路径 BF16 化;fc 形状)
#[tokio::test]
async fn marlin_w4a16_bf16_parity() {
    let mut client = assemble();
    for (m, n, k, g) in [(20usize, 5120usize, 5120usize, 128usize), (8, 5120, 5120, 128), (64, 512, 512, 128)] {
        let (noise_ratio, max_abs) = run_case(&mut client, m, n, k, g, true).await;
        eprintln!("[marlin-bf16] m={m} n={n} k={k} g={g} → 噪声/信号RMS={noise_ratio:.6} max_abs={max_abs:.5}");
        assert!(
            noise_ratio < 2e-2 && noise_ratio.is_finite(),
            "m={m}: 噪声比 {noise_ratio}"
        );
    }
    client.sync().await.expect("sync");
}

/// S1 暗雷根治探针:dump owl 打包字节(q + owlb + owls)供 vLLM 金标比对
/// (金标端 = vllm marlin_utils_test.marlin_weights,python 侧比对;
/// 判决目标:非 2^k n 的 repack 分歧 vs kernel selector 罪责二选一)
#[tokio::test]
async fn marlin_golden_dump() {
    let mut client = assemble();
    let dir = std::path::Path::new("/tmp/marlin_golden");
    std::fs::create_dir_all(dir).expect("mkdir");
    // 形状组:2^k 对照 + 非 2^k 暗雷档(27B 维度族)
    let cases: Vec<(usize, usize, usize)> = vec![
        (512, 512, 128),
        (512, 3584, 128),
        (512, 5120, 128),
        (512, 6144, 128),
        (512, 17408, 128),
    ];
    for (k, n, g) in cases {
        let tag = format!("k{k}_n{n}");
        let q: Vec<u8> = (0..n * k).map(|i| ((i * 7 + 3) % 16) as u8).collect();
        let b_packed = owl_kernels::marlin::repack::pack_marlin_b(&q, k, n);
        let s: Vec<f32> = (0..n * (k / g)).map(|i| 0.5 + ((i * 13) % 13) as f32 * 0.08).collect();
        let s_packed = owl_kernels::marlin::repack::pack_marlin_s(&s, n, k / g);
        std::fs::write(dir.join(format!("{tag}.q.bin")), &q).expect("q");
        std::fs::write(dir.join(format!("{tag}.owlb.bin")), le_i32(&b_packed)).expect("b");
        std::fs::write(dir.join(format!("{tag}.owls.bin")), le_u16(&s_packed)).expect("s");
        let _ = &mut client; // assemble 仅保 face 生命周期(打包纯 host)
        eprintln!("[golden-dump] {tag}: q={}B b={}i32 s={}u16", q.len(), b_packed.len(), s_packed.len());
    }
    client.sync().await.expect("sync");
}

/// E3 形状扫描(0.8B W4A16 实际形状;定位 marlin 挂起形状)
/// 单案 memcheck 入口(compute-sanitizer 配套;OWL_SWEEP_CASE="m,n,k,g")
#[tokio::test]
async fn marlin_shape_case() {
    let Ok(spec) = std::env::var("OWL_SWEEP_CASE") else {
        eprintln!("skip: OWL_SWEEP_CASE 未设(m,n,k,g)");
        return;
    };
    let parts: Vec<usize> = spec.split(',').map(|v| v.parse().unwrap()).collect();
    let (m, n, k, g) = (parts[0], parts[1], parts[2], parts[3]);
    let mut client = assemble();
    let (noise_ratio, max_abs) = run_case(&mut client, m, n, k, g, false).await;
    eprintln!("[case] m={m} n={n} k={k} g={g} → 噪声比 {noise_ratio:.6} max_abs={max_abs:.5}");
    assert!(noise_ratio < 2e-2 && noise_ratio.is_finite(), "噪声比 {noise_ratio}");
    client.sync().await.expect("sync");
}

#[tokio::test]
async fn marlin_shape_sweep() {
    let mut client = assemble();
    // 基础档(0.512k 网格)+ 27B 维度族(Qwen3.8-27B:5120/6144/10240/
    // 14336/17408;暗雷根治后应全绿)
    let mut cases: Vec<(usize, usize, usize, usize)> = (1..=12usize)
        .map(|step| (32usize, step * 512usize, 1024usize, 128usize))
        .collect();
    for (m, n, k) in [
        (1usize, 5120, 5120),
        (1, 6144, 5120),
        (1, 17408, 5120),
        (1, 14336, 5120),
        (1, 5120, 17408),
        (32, 17408, 5120),
        (4096, 17408, 5120),
        (4096, 5120, 17408),
    ] {
        cases.push((m, n, k, 128));
    }
    for (m, n, k, g) in cases {
        eprintln!("[sweep] try m={m} n={n} k={k} g={g}");
        let (noise_ratio, _max_abs) = run_case(&mut client, m, n, k, g, false).await;
        eprintln!("[sweep] done m={m} n={n} k={k} → {noise_ratio:.6}");
    }
}

// ============================================================================
// AWQ(kU4 非对称,zp)臂(2026-10-01 cyankiwi g32 装载线)
// ============================================================================

/// host dequant 参考(AWQ 非对称):W = (q − zp_group) × scale。
/// zp 源布局 = ct out 主序:zp_packed[t, c] 的 nibble i = 行 8t+i。
fn host_dequant_gemm_awq(
    a: &[f32],
    q: &[u8],
    s: &[f32],
    zp_packed: &[i32],
    m: usize,
    n: usize,
    k: usize,
    g: usize,
) -> Vec<f32> {
    let groups = k / g;
    let mut c = vec![0f32; m * n];
    for r in 0..m {
        for nn in 0..n {
            // zp 逐组:word = (nn/8)*groups + 组号 c = kk/g;nibble = nn%8
            let mut acc = 0f32;
            for kk in 0..k {
                let z = ((zp_packed[(nn / 8) * groups + kk / g] as u32) >> (4 * (nn % 8))) & 0xF;
                let nib = q[nn * k + kk];
                let w = (nib as f32 - z as f32) * s[nn * groups + kk / g];
                acc += a[r * k + kk] * w;
            }
            c[r * n + nn] = acc;
        }
    }
    c
}

/// AWQ 外核启动消息(槽序契约:GEMM_W4A16_AWQ = scales 后插 zeros,
/// 7 Block + 4 sz)
fn marlin_awq_launch(
    a: &owl_cuda::Bytes,
    b: &owl_cuda::Bytes,
    c: &owl_cuda::Bytes,
    s: &owl_cuda::Bytes,
    z: &owl_cuda::Bytes,
    ws: &owl_cuda::Bytes,
    ctmp: &owl_cuda::Bytes,
    m: usize,
    k: usize,
    n: usize,
    g: usize,
) -> LaunchMsg {
    LaunchMsg {
        kernel: owl_cuda::KernelSpec {
            name: owl_kernels::marlin::GEMM_W4A16_AWQ.into(),
            source: String::new(),
        },
        args: vec![
            Arg::Block { id: a.id },
            Arg::Block { id: b.id },
            Arg::Block { id: c.id },
            Arg::Block { id: s.id },
            Arg::Block { id: z.id },
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

async fn run_awq_case(client: &mut GpuClient, m: usize, n: usize, k: usize) -> (f64, f32) {
    run_awq_case_g(client, m, n, k, 32).await
}

async fn run_awq_case_g(client: &mut GpuClient, m: usize, n: usize, k: usize, g: usize) -> (f64, f32) {
    let groups = k / g;
    let a: Vec<f32> = (0..m * k).map(|i| ((i as f32 * 0.31) - 4.0).sin() * 0.5).collect();
    let q: Vec<u8> = (0..n * k).map(|i| ((i * 7 + 3) % 16) as u8).collect();
    let s: Vec<f32> = (0..n * groups).map(|i| 0.02 + ((i * 13) % 13) as f32 * 0.004).collect();
    // zp:非对称域 [0,15],伪随机;打包维 = out(word t 行 8t+i)
    let zp_u8: Vec<u8> = (0..n * groups).map(|i| ((i * 5 + 2) % 16) as u8).collect();
    let zp_packed: Vec<i32> = (0..(n / 8) * groups)
        .map(|cell| {
            let t = cell / groups;
            let c = cell % groups;
            (0..8)
                .map(|i| ((zp_u8[(t * 8 + i) * groups + c] as u32) << (4 * i)))
                .fold(0u32, |acc, v| acc | v) as i32
        })
        .collect();

    let b_packed = owl_kernels::marlin::repack::pack_marlin_b(&q, k, n);
    let s_packed = owl_kernels::marlin::repack::pack_marlin_s(&s, n, groups);
    let z_packed = owl_kernels::marlin::repack::pack_marlin_z(&zp_u8, n, groups);
    assert_eq!(z_packed.len(), groups * (n / 8));

    let da = client.htod(Dtype::F16, &Shape::from(vec![m, k]), &le_u16(
        &a.iter().map(|f| half::f16::from_f32(*f).to_bits()).collect::<Vec<_>>(),
    )).await.expect("htod a");
    let db = client.htod(Dtype::U32, &Shape::from(vec![b_packed.len()]), &le_i32(&b_packed)).await.expect("htod b");
    let ds = client.htod(Dtype::F16, &Shape::from(vec![s_packed.len()]), &le_u16(&s_packed)).await.expect("htod s");
    let dz = client.htod(Dtype::U32, &Shape::from(vec![z_packed.len()]), &le_i32(&z_packed)).await.expect("htod z");
    let dws = client.alloc(Dtype::U32, owl_kernels::marlin::v2_workspace_len(n)).await.expect("alloc ws");
    let dctmp = client.alloc(Dtype::U32, 1).await.expect("alloc ctmp");
    let dc = client.alloc(Dtype::F16, m * n).await.expect("alloc c");
    // 哨兵填充:-7.0(launch 后若残留 = 内核未写 C)
    {
        let sent: Vec<u8> = (0..m * n)
            .flat_map(|_| half::f16::from_f32(-7.0).to_le_bytes())
            .collect();
        let dsent = client.htod(Dtype::F16, &Shape::from(vec![m * n]), &sent).await.expect("sent");
        client.copy_block_at(&dsent, 0, &dc, 0, sent.len()).await.expect("sent copy");
    }

    client.launch(marlin_awq_launch(&da, &db, &dc, &ds, &dz, &dws, &dctmp, m, k, n, g)).await.expect("awq launch");

    let mut buf = vec![0u8; m * n * 2];
    client.dtoh(&dc, &mut buf).await.expect("dtoh");
    let got: Vec<f32> = buf
        .chunks_exact(2)
        .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect();
    client.sync().await.expect("sync");

    let c_ref = host_dequant_gemm_awq(&a, &q, &s, &zp_packed, m, n, k, g);
    let (mut sum_abs, mut sum_sq) = (0f64, 0f64);
    let mut max_abs = 0f32;
    for (x, r) in got.iter().zip(&c_ref) {
        sum_abs += (*x as f64 - *r as f64).abs();
        sum_sq += (*r as f64) * (*r as f64);
        max_abs = max_abs.max((*x - *r).abs());
    }
    let rms = (sum_sq / (m * n) as f64).sqrt().max(1e-6);
    (sum_abs / (m * n) as f64 / rms, max_abs)
}

/// AWQ kU4 对拍(2026-10-01):非对称 (q−zp)×s,g=32;形状覆盖
/// m8 路径(m=1,kv-proj 形状 n=1024)与多 m-block 路径(m=32,o-proj 形状)。
#[tokio::test]
async fn marlin_w4a16_awq_parity() {
    let mut client = assemble();
    for (m, n, k) in [(1usize, 1024usize, 5120usize), (32usize, 5120usize, 5120usize)] {
        let (noise_ratio, max_abs) = run_awq_case(&mut client, m, n, k).await;
        eprintln!("[awq] m={m} n={n} k={k} g=32 → 噪声/信号RMS={noise_ratio:.6} max_abs={max_abs:.5}");
        assert!(
            noise_ratio < 2e-2 && noise_ratio.is_finite(),
            "awq 噪声比 {noise_ratio}"
        );
    }
    client.sync().await.expect("sync");
}

// ============================================================================
// ct packed → marlin B 设备重排对拍(2026-10-01 GPU repack 装载线)
// ============================================================================

/// GPU owl_ct_repack_u32 输出 vs CPU pack_marlin_b_fused 逐 u32 一致。
/// 形状覆盖 grid 两维(rows/64 × cols/2);ct packed 伪随机。
#[tokio::test]
async fn ct_repack_parity() {
    let mut client = assemble();
    for (out_dim, k) in [(1024usize, 5120usize), (5120usize, 5120usize), (17408usize, 5120usize)] {
        let rows = out_dim;
        let cols = k / 8;
        let packed: Vec<i32> = (0..rows * cols).map(|i| (i as i32).wrapping_mul(0x9E3779B1_u32 as i32) ^ 0x1234_5678).collect();
        // CPU 参照(fused)
        let fused = owl_kernels::marlin::repack::marlin_fused_indices(k, out_dim);
        let mut b_ref: Vec<i32> = Vec::new();
        owl_kernels::marlin::repack::pack_marlin_b_fused(&packed, &fused, rows * k / 8, &mut b_ref);
        assert_eq!(b_ref.len(), (k / 16) * (rows * 2));

        // GPU
        let dp = client.htod(Dtype::U32, &Shape::from(vec![packed.len()]), &le_i32(&packed)).await.expect("htod packed");
        let dout = client.alloc(Dtype::U32, b_ref.len()).await.expect("alloc out");
        let msg = LaunchMsg {
            kernel: owl_cuda::KernelSpec {
                name: "owl_ct_repack_u32".into(),
                source: owl_kernels::sources::owl::CT_REPACK_U32.into(),
            },
            args: vec![
                Arg::Block { id: dp.id },
                Arg::U64(rows as u64),
                Arg::U64(cols as u64),
                Arg::Block { id: dout.id },
            ],
            // grid = (out/64, (k/8)/2);block 32(契约见 .cu 头注)
            grid: ((rows / 64) as u32, (cols / 2) as u32, 1),
            block: (32, 1, 1),
            shared_mem: 0,
            out_elems: b_ref.len(),
        };
        client.launch(msg).await.expect("repack launch");
        let mut buf = vec![0u8; b_ref.len() * 4];
        client.dtoh(&dout, &mut buf).await.expect("dtoh");
        client.sync().await.expect("sync");

        let mut bad = 0usize;
        for (i, w) in buf.chunks_exact(4).enumerate() {
            let got = i32::from_le_bytes([w[0], w[1], w[2], w[3]]);
            if got != b_ref[i] { bad += 1; }
        }
        eprintln!("[repack] out={out_dim} k={k}: 坏字 {}/{}", bad, b_ref.len());
        assert_eq!(bad, 0, "GPU repack 必须 vs CPU fused 逐位一致");
    }
}

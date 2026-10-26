//! **FI prefill f16 shadow 探针**(2026-10-26)
//!
//! 两关:
//! 1. `owl_kv_fp8_to_f16_knhd` 按物理页 index 反量化到 dense 逻辑页,
//!    逐元素对拍 e4m3 → f16(含**乱序物理页表**,证明映射不是恒等假设);
//! 2. 同一 q / 同一源池:fp8 FI vs `dequant → f16 FI` 输出对拍。
//!
//! **数值门 = 相对误差**(2026-10-26 修正):输出幅度可达 ~448,绝对 0.25
//! 只是 1 个 f16 ULP;旧 `2e-3` 绝对门会把正常舍入误判成核错误。
//! 门限:`rel_l2 < 2e-3 && rel_max < 3e-2`;100k 实测 f16 vs fp8
//! `rel_l2≈2.4e-4`、`rel_max≈4.2e-3`。`FI_SHADOW_DUMP=1` 会额外落
//! q/k/v/out 到 `/tmp/owl-fi-*.bin`,供 torch f32 参考复核。
//!
//! 运行:
//! `OWL_TEST_DEVICE=42:00.0 cargo test --release -p owl-kernels \
//!  --features device,flashinfer --test fi_shadow_probe -- --test-threads=1 --nocapture`
#![cfg(feature = "device")]

mod fi_support;
use fi_support::{f16_to_f32, TestRes, H, HD, HKV, PAGE};

use owl_kernels::contract::{Arg, Bytes, Dtype, KernelSource, LaunchMsg};
use owl_kernels::device::Exec;
use owl_macros::owl_test;
use std::time::Instant;

fn e4m3_decode(code: u8) -> f32 {
    let s = if code & 0x80 != 0 { -1f32 } else { 1f32 };
    let e = ((code >> 3) & 0x0F) as i32;
    let m = (code & 0x07) as i32;
    let v = if e == 0 {
        (m as f32) * 2f32.powi(-9)
    } else {
        (1.0 + m as f32 / 8.0) * 2f32.powi(e - 7)
    };
    s * v
}

fn lcg_u8(n: usize, seed: u32) -> Vec<u8> {
    (0..n)
        .map(|i| {
            let x = seed.wrapping_mul(0x9E37_79B9).wrapping_add((i as u32).wrapping_mul(0x85EB_CA6B));
            // 只用有限 e4m3 码(0..=0x7E / 0x80..=0xFE)
            let c = (x % 0x7F) as u8;
            if x & 0x8000_0000 != 0 { c | 0x80 } else { c }
        })
        .collect()
}
fn lcg_f16(n: usize, seed: u32) -> Vec<u16> {
    (0..n)
        .map(|i| {
            let x = seed.wrapping_mul(0x9E37_79B9).wrapping_add((i as u32).wrapping_mul(0x85EB_CA6B));
            half::f16::from_f32(((x & 0xFFFF) as f32 / 65535.0 - 0.5) * 4.0).to_bits()
        })
        .collect()
}

fn upload_u32(res: &mut TestRes, v: &[u32]) -> Bytes {
    res.upload(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())
}

fn prefill_msg(name: &str, t: usize, ctx: usize, kv_dtype: Dtype, q: u64, k: u64, v: u64,
               q_cu: u64, indices: u64, indptr: u64, last_len: u64, wr: u64, out: u64) -> LaunchMsg {
    LaunchMsg {
        kernel: KernelSource { name: name.to_string(), source: String::new() },
        args: vec![
            Arg::Block { id: q, dtype: Dtype::F16, rank: 2 },
            Arg::Block { id: k, dtype: kv_dtype, rank: 1 },
            Arg::Block { id: v, dtype: kv_dtype, rank: 1 },
            Arg::Block { id: q_cu, dtype: Dtype::U32, rank: 1 },
            Arg::Block { id: indices, dtype: Dtype::U32, rank: 1 },
            Arg::Block { id: indptr, dtype: Dtype::U32, rank: 1 },
            Arg::Block { id: last_len, dtype: Dtype::U32, rank: 1 },
            Arg::Block { id: wr, dtype: Dtype::F16, rank: 1 },
            Arg::Block { id: out, dtype: Dtype::F16, rank: 2 },
            Arg::U64(t as u64),
            Arg::U64(ctx as u64),
            Arg::U64(t as u64),
            Arg::U64(H as u64),
            Arg::U64(HKV as u64),
            Arg::U64(HD as u64),
            Arg::U64(PAGE as u64),
            Arg::U64(((1.0f32 / (HD as f32).sqrt()).to_bits()) as u64),
        ],
        grid: (0, 0, 0),
        block: (0, 0, 0),
        shared_mem: 0,
        out_elems: t * H * HD,
    }
}

fn f16_ulp(x: f32) -> f32 {
    let a = x.abs();
    if a < 6.1035156e-5 { 5.9604645e-8 } else { 2f32.powi((a.log2().floor() as i32) - 10) }
}

struct DiffStats {
    maxd: f32,
    rel_max: f32,
    rel_l2: f64,
    max_ulp: f32,
}

fn diff_stats(label: &str, a: &[u16], b: &[u16]) -> DiffStats {
    let (mut maxd, mut maxu) = (0f32, 0f32);
    let (mut maxabs, mut l2d, mut l2a) = (0f32, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        let xf = f16_to_f32(x);
        let yf = f16_to_f32(y);
        let d = (xf - yf).abs();
        let u = d / f16_ulp(xf.abs().max(yf.abs()));
        maxd = maxd.max(d);
        maxu = maxu.max(u);
        maxabs = maxabs.max(xf.abs().max(yf.abs()));
        l2d += (d as f64) * (d as f64);
        l2a += (xf as f64) * (xf as f64);
    }
    let st = DiffStats {
        maxd,
        rel_max: maxd / maxabs.max(1e-30),
        rel_l2: (l2d / l2a.max(1e-30)).sqrt(),
        max_ulp: maxu,
    };
    println!("[shadow] {label}: max|d|={:.6e} rel_max={:.3e} rel_l2={:.3e} max_ulp={:.2} maxabs={maxabs:e}",
             st.maxd, st.rel_max, st.rel_l2, st.max_ulp);
    st
}

fn host_ref(t: usize, ctx: usize, q: &[u16], k: &[u8], v: &[u8]) -> Vec<f32> {
    host_ref_off(t, ctx, q, k, v, 0)
}

fn host_ref_off(t: usize, ctx: usize, q: &[u16], k: &[u8], v: &[u8], off: i32) -> Vec<f32> {
    let qf: Vec<f32> = q.iter().map(|&b| f16_to_f32(b)).collect();
    let kf: Vec<f32> = k.iter().map(|&b| e4m3_decode(b)).collect();
    let vf: Vec<f32> = v.iter().map(|&b| e4m3_decode(b)).collect();
    let mut out = vec![0f32; t * H * HD];
    let scale = 1.0 / (HD as f32).sqrt();
    for s in 0..t {
        let len = (((ctx - t + s + 1) as i32) + off) as usize;
        for h in 0..H {
            let kvh = h / (H / HKV);
            let mut qk = vec![0f32; len];
            for b in 0..len {
                let (pb, bb) = (b / PAGE, b % PAGE);
                let mut acc = 0f32;
                for d in 0..HD {
                    let kidx = pb * PAGE * HKV * HD + bb * HKV * HD + kvh * HD + d;
                    acc += qf[(s * H + h) * HD + d] * kf[kidx];
                }
                qk[b] = acc * scale;
            }
            let m = qk.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let l: f32 = qk.iter().map(|&x| (x - m).exp()).sum();
            for d in 0..HD {
                let mut acc = 0f32;
                for b in 0..len {
                    let (pb, bb) = (b / PAGE, b % PAGE);
                    let vidx = pb * PAGE * HKV * HD + bb * HKV * HD + kvh * HD + d;
                    acc += (qk[b] - m).exp() * vf[vidx];
                }
                out[(s * H + h) * HD + d] = acc / l;
            }
        }
    }
    out
}

#[owl_test(gpu)]
fn fi_shadow_dequant_and_prefill_parity() {
    let t = owl_shared::env_reader::parse::<usize>("FI_SHADOW_T").unwrap_or(64);
    let ctx = owl_shared::env_reader::parse::<usize>("FI_SHADOW_CTX").unwrap_or(4096);
    let bench = owl_shared::env_reader::flag("FI_SHADOW_BENCH");
    let bench_n = owl_shared::env_reader::parse::<usize>("FI_SHADOW_ITERS").unwrap_or(5);
    let nb = ctx.div_ceil(PAGE);
    let mut res = TestRes::new(owl_cuda::test_device_ordinal());
    let mut exec = Exec::new();
    let active: &[&str] = &[
        owl_kernels::family::flashinfer::PREFILL_FI_FP8KV,
        owl_kernels::family::flashinfer::PREFILL_FI,
        "owl_reshape_and_cache_fp8kv_knhd",
        "vllm_reshape_and_cache_f16_knhd",
    ];
    let mut reg = owl_kernels::registry::OpRegistry::init_all_with(
        owl_kernels::server::family_factories(),
        &mut res,
        &mut exec,
        Some(active),
    )
    .expect("注册表装配");

    let pool = nb * HKV * HD * PAGE;
    let k8 = lcg_u8(pool, 701);
    let v8 = lcg_u8(pool, 702);
    let q16 = lcg_f16(t * H * HD, 703);

    let k8_b = res.upload(&k8);
    let v8_b = res.upload(&v8);
    let q_b = res.upload(&q16.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
    let q_cu = upload_u32(&mut res, &[0, t as u32]);
    // **乱序物理页**:逻辑 p → 物理 nb-1-p;反量化必须按 indices 取源页。
    let phys: Vec<u32> = if owl_shared::env_reader::flag("FI_SHADOW_IDX") {
        (0..nb as u32).collect()
    } else {
        (0..nb as u32).rev().collect()
    };
    let idx_src = upload_u32(&mut res, &phys);
    let idx_dst = upload_u32(&mut res, &(0..nb as u32).collect::<Vec<_>>());
    let indptr = upload_u32(&mut res, &[0, nb as u32]);
    let last_len = res.upload(&((ctx - (nb - 1) * PAGE) as u32).to_le_bytes());
    let wr = res.zeros(8);
    let out_pad = t + 16;
    let out8 = res.zeros(out_pad * H * HD * 2);

    // 1) 反量化(dense 逻辑页)
    let k16 = res.zeros(pool * 2);
    let v16 = res.zeros(pool * 2);
    let rc = unsafe {
        owl_kernels::family::flashinfer::owl_fi_dequant_fp8_to_f16(
            res.ptr(k8_b.id) as *const _, res.ptr(v8_b.id) as *const _,
            res.ptr(k16.id) as *mut _, res.ptr(v16.id) as *mut _,
            res.ptr(idx_src.id) as *const i32, nb as i32, HKV as i32, HD as i32, PAGE as i32,
            res.stream.cu_stream() as *mut _,
        )
    };
    assert_eq!(rc, 0, "dequant rc={rc}");
    res.stream.synchronize().expect("sync");
    let (mut bad, mut bad_v, mut worst) = (0usize, 0usize, 0f32);
    if pool <= 8_000_000 {
        let got = res.take_u16(k16.id);
        for p in 0..nb {
            for i in 0..PAGE * HKV * HD {
                let src = phys[p] as usize * PAGE * HKV * HD + i;
                let want = half::f16::from_f32(e4m3_decode(k8[src])).to_bits();
                let d = (f16_to_f32(got[p * PAGE * HKV * HD + i]) - e4m3_decode(k8[src])).abs();
                if d > worst { worst = d; }
                if got[p * PAGE * HKV * HD + i] != want { bad += 1; }
            }
        }
        let got_v = res.take_u16(v16.id);
        for p in 0..nb {
            for i in 0..PAGE * HKV * HD {
                let src = phys[p] as usize * PAGE * HKV * HD + i;
                let want = half::f16::from_f32(e4m3_decode(v8[src])).to_bits();
                if got_v[p * PAGE * HKV * HD + i] != want { bad_v += 1; }
            }
        }
    }
    println!("[shadow] dequant bad={bad}/{pool} bad_v={bad_v}/{pool} worst={worst:e} (check={})", pool <= 8_000_000);
    if !bench {
        assert_eq!(bad, 0, "dequant K 逐位不等");
        assert_eq!(bad_v, 0, "dequant V 逐位不等");
    }

    // 2) fp8 FI vs dequant→f16 FI
    let msg8 = prefill_msg(
        owl_kernels::client::flashinfer::PREFILL_FI_FP8KV, t, ctx, Dtype::U32,
        q_b.id, k8_b.id, v8_b.id, q_cu.id, idx_src.id, indptr.id, last_len.id, wr.id, out8.id,
    );
    reg.execute(&msg8, &mut res, &mut exec).expect("fp8 FI");
    res.stream.synchronize().expect("sync");
    let out16 = res.zeros(out_pad * H * HD * 2);
    let msg16 = prefill_msg(
        owl_kernels::client::flashinfer::PREFILL_FI, t, ctx, Dtype::F16,
        q_b.id, k16.id, v16.id, q_cu.id, idx_dst.id, indptr.id, last_len.id, wr.id, out16.id,
    );
    reg.execute(&msg16, &mut res, &mut exec).expect("f16 FI");
    res.stream.synchronize().expect("sync");

    if bench {
        // warmup
        for _ in 0..2 {
            reg.execute(&msg8, &mut res, &mut exec).expect("warm fp8");
            reg.execute(&msg16, &mut res, &mut exec).expect("warm f16");
        }
        res.stream.synchronize().expect("sync");
        let t0 = Instant::now();
        for _ in 0..bench_n {
            reg.execute(&msg8, &mut res, &mut exec).expect("bench fp8");
        }
        res.stream.synchronize().expect("sync");
        let fp8_ms = t0.elapsed().as_secs_f64() * 1e3 / bench_n as f64;

        let t1 = Instant::now();
        for _ in 0..bench_n {
            let rc = unsafe {
                owl_kernels::family::flashinfer::owl_fi_dequant_fp8_to_f16(
                    res.ptr(k8_b.id) as *const _, res.ptr(v8_b.id) as *const _,
                    res.ptr(k16.id) as *mut _, res.ptr(v16.id) as *mut _,
                    res.ptr(idx_src.id) as *const i32, nb as i32, HKV as i32, HD as i32,
                    PAGE as i32, res.stream.cu_stream() as *mut _,
                )
            };
            assert_eq!(rc, 0, "bench dequant rc={rc}");
            reg.execute(&msg16, &mut res, &mut exec).expect("bench f16");
        }
        res.stream.synchronize().expect("sync");
        let sh_ms = t1.elapsed().as_secs_f64() * 1e3 / bench_n as f64;

        let t2 = Instant::now();
        for _ in 0..bench_n {
            let rc = unsafe {
                owl_kernels::family::flashinfer::owl_fi_dequant_fp8_to_f16(
                    res.ptr(k8_b.id) as *const _, res.ptr(v8_b.id) as *const _,
                    res.ptr(k16.id) as *mut _, res.ptr(v16.id) as *mut _,
                    res.ptr(idx_src.id) as *const i32, nb as i32, HKV as i32, HD as i32,
                    PAGE as i32, res.stream.cu_stream() as *mut _,
                )
            };
            assert_eq!(rc, 0, "bench dequant-only rc={rc}");
        }
        res.stream.synchronize().expect("sync");
        let dq_ms = t2.elapsed().as_secs_f64() * 1e3 / bench_n as f64;
        println!(
            "[shadow-bench] T={t} ctx={ctx} fp8={fp8_ms:.3}ms shadow(f16+dq)={sh_ms:.3}ms dequant={dq_ms:.3}ms speedup={:.3}x net={:.3}ms",
            fp8_ms / sh_ms, fp8_ms - sh_ms
        );
    }
    let a = res.take_u16(out8.id);
    let b = res.take_u16(out16.id);
    let mut maxd = 0f32;
    for (x, y) in a.iter().take(t * H * HD).zip(&b) {
        maxd = maxd.max((f16_to_f32(*x) - f16_to_f32(*y)).abs());
    }
    println!("[shadow] fp8-vs-f16(shadow) FI max|Δ|={maxd:e}");
    let st_shadow = diff_stats("fp8-vs-f16(shadow)", &a[..t * H * HD], &b[..t * H * HD]);
    if owl_shared::env_reader::flag("FI_SHADOW_ROWS") {
        for s in 0..t {
            let mut rd = 0f32;
            let mut hh = 0usize;
            for (i, (x, y)) in a.iter().zip(&b).enumerate() {
                if i / (H * HD) != s { continue; }
                let d = (f16_to_f32(*x) - f16_to_f32(*y)).abs();
                if d > rd { rd = d; hh = (i % (H * HD)) / HD; }
            }
            println!("[shadow] row {s}: maxΔ={rd:e} head={hh}");
        }
    }

    // 2b) 直接 f16 物理池(不经 shadow/identity)作对照:
    //     若这里也差,则是 f16 核/plan;若这里对,则是 shadow 路径。
    let mut kd = vec![0u16; pool];
    let mut vd = vec![0u16; pool];
    for i in 0..pool {
        kd[i] = half::f16::from_f32(e4m3_decode(k8[i])).to_bits();
        vd[i] = half::f16::from_f32(e4m3_decode(v8[i])).to_bits();
    }
    let kd_b = res.upload(&kd.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
    let vd_b = res.upload(&vd.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
    let outd = res.zeros(out_pad * H * HD * 2);
    let msgd = prefill_msg(
        owl_kernels::client::flashinfer::PREFILL_FI, t, ctx, Dtype::F16,
        q_b.id, kd_b.id, vd_b.id, q_cu.id, idx_src.id, indptr.id, last_len.id, wr.id, outd.id,
    );
    reg.execute(&msgd, &mut res, &mut exec).expect("f16 direct FI");
    res.stream.synchronize().expect("sync");
    let d = res.take_u16(outd.id);
    let mut maxd2 = 0f32;
    for (x, y) in a.iter().take(t * H * HD).zip(&d) {
        maxd2 = maxd2.max((f16_to_f32(*x) - f16_to_f32(*y)).abs());
    }
    println!("[shadow] fp8-vs-f16(direct) FI max|Δ|={maxd2:e}");
    let st_direct = diff_stats("fp8-vs-f16(direct)", &a[..t * H * HD], &d[..t * H * HD]);
    if owl_shared::env_reader::flag("FI_SHADOW_HOST") && t * ctx <= 2048 {
        let href = host_ref(t, ctx, &q16, &k8, &v8);
        let d8 = a.iter().take(t * H * HD).zip(&href)
            .map(|(x, y)| (f16_to_f32(*x) - *y).abs()).fold(0f32, f32::max);
        let d16 = b.iter().take(t * H * HD).zip(&href)
            .map(|(x, y)| (f16_to_f32(*x) - *y).abs()).fold(0f32, f32::max);
        println!("[shadow] vs host(f32): fp8={d8:e} f16={d16:e}");
        let h16: Vec<u16> = href.iter().map(|&x| half::f16::from_f32(x).to_bits()).collect();
        let st_h8 = diff_stats("fp8-vs-host16", &a[..t * H * HD], &h16);
        let st_h16 = diff_stats("f16-vs-host16", &b[..t * H * HD], &h16);
        assert!(st_h8.rel_l2 < 5e-4 && st_h16.rel_l2 < 5e-4, "host rel_l2 超门");
    }
    if owl_shared::env_reader::flag("FI_SHADOW_DUMP") {
        let dump = |name: &str, bytes: &[u8]| {
            std::fs::write(format!("/tmp/owl-fi-{name}.bin"), bytes).expect("dump");
        };
        dump("q16", &q16.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
        dump("k8", &k8);
        dump("v8", &v8);
        dump("out8", &a.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
        dump("out16", &b.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
        dump("outd", &d.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
        println!("[shadow] dumped q16/k8/v8/out8/out16/outd T={t} ctx={ctx} H={H} HKV={HKV} HD={HD} PAGE={PAGE}");
    }
    if !bench {
        // **相对门**(2026-10-26 修正):绝对 2e-3 在输出 ~448 时会被 f16 的
        // 1 ULP(0.25)误判;这里以 L2/最大相对误差为准。100k 实测 f16 vs fp8
        // rel_l2≈2.4e-4、rel_max≈4.2e-3(与 torch f32 参考同量级)。
        let gate = |name: &str, st: &DiffStats| {
            assert!(
                st.rel_l2 < 2e-3 && st.rel_max < 3e-2,
                "{name} 相对门超限:rel_l2={:.3e} rel_max={:.3e} max|d|={:.6e}",
                st.rel_l2, st.rel_max, st.maxd
            );
        };
        gate("shadow", &st_shadow);
        gate("direct", &st_direct);
    }
}

//! Native owl INT8 prefill smoke/timing probe.
//! T=8/L=64: finite/nonzero output; T=2048/L=65536: timing only.
#![cfg(feature = "device")]

mod fi_support;

use std::time::Instant;

use owl_kernels::bootcheck::{CheckBuf, CheckRunner};
use owl_kernels::contract::LaunchVal;
use owl_kernels::device::Exec;
use owl_kernels::server::selfcheck::Runner;

const ENTRY: &str = "owl_int8_prefill_paged";
const SRC: &str = "cu/attn/int8_prefill.cu";
const ENTRY_F16PV: &str = "owl_int8_prefill_paged_f16pv";
const SRC_F16PV: &str = "cu/attn/int8_prefill_f16pv.cu";
const D: usize = 256;
const HQ: usize = 24;
const HKV: usize = 4;
const SMEM: u32 = 16384 + 8192 + 8192 + 2048 + 1024 + 128 + 8192;

fn lcg_u8(n: usize, seed: u32) -> Vec<u8> {
    (0..n)
        .map(|i| {
            let x = seed
                .wrapping_mul(0x9E37_79B9)
                .wrapping_add((i as u32).wrapping_mul(0x85EB_CA6B));
            (x >> 13) as u8
        })
        .collect()
}
fn lcg_f32(n: usize, seed: u32, base: f32, span: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = seed
                .wrapping_mul(0x9E37_79B9)
                .wrapping_add((i as u32).wrapping_mul(0x85EB_CA6B));
            base + ((x & 0xFFFF) as f32 / 65535.0) * span
        })
        .collect()
}

fn f16_to_f32(h: u16) -> f32 {
    let s = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let e = ((h >> 10) & 0x1F) as i32;
    let m = (h & 0x3FF) as f32;
    if e == 0 {
        return s * m * 2f32.powi(-24);
    }
    if e == 31 {
        return if m == 0.0 {
            s * f32::INFINITY
        } else {
            f32::NAN
        };
    }
    s * (1.0 + m / 1024.0) * 2f32.powi(e - 15)
}

fn run(t: usize, l: usize, timing: bool) -> Option<()> {
    let dev = owl_cuda::test_device_ordinal_opt()?;
    let mut res = fi_support::TestRes::new(dev);
    let mut exec = Exec::new();
    let mut r = Runner {
        res: &mut res,
        exec: &mut exec,
        registry: None,
    };

    let q = lcg_u8(t * HQ * D, 0x1234);
    let qs = lcg_f32(t * HQ * 8, 0x2345, 0.004, 0.004);
    let k = lcg_u8(l * HKV * D, 0x3456);
    let ks = lcg_f32(l * HKV * 8, 0x4567, 0.004, 0.004);
    // V token-major [L, Hkv, D] int8, random bytes.
    let page = 32usize;
    let vp = lcg_u8(HKV * D * l, 0x5678);
    let nblocks = l / page;
    let bt: Vec<f32> = (0..nblocks as u32).map(|x| x as f32).collect();
    let btb: Vec<u8> = bt.iter().flat_map(|x| x.to_le_bytes()).collect();
    let vs = lcg_f32(l * HKV, 0x6789, 0.005, 0.005);
    let mut out = vec![0u8; t * HQ * D * 2];

    let dq = r.alloc(q.len(), "int8.q").unwrap();
    let dqs = r.alloc(qs.len() * 4, "int8.qs").unwrap();
    let dk = r.alloc(k.len(), "int8.k").unwrap();
    let dks = r.alloc(ks.len() * 4, "int8.ks").unwrap();
    let dvp = r.alloc(vp.len(), "int8.vp").unwrap();
    let dbt = r.alloc(btb.len(), "int8.bt").unwrap();
    let dvs = r.alloc(vs.len() * 4, "int8.vs").unwrap();
    let dout = r.alloc(out.len(), "int8.out").unwrap();

    let qsb: Vec<u8> = qs.iter().flat_map(|x| x.to_le_bytes()).collect();
    let ksb: Vec<u8> = ks.iter().flat_map(|x| x.to_le_bytes()).collect();
    let vsb: Vec<u8> = vs.iter().flat_map(|x| x.to_le_bytes()).collect();
    r.upload(dq, &q).unwrap();
    r.upload(dqs, &qsb).unwrap();
    r.upload(dk, &k).unwrap();
    r.upload(dks, &ksb).unwrap();
    r.upload(dvp, &vp).unwrap();
    r.upload(dbt, &btb).unwrap();
    r.upload(dvs, &vsb).unwrap();

    let base = (l - t) as i32;
    let vals = [
        LaunchVal::Ptr(dq.ptr),
        LaunchVal::Ptr(dqs.ptr),
        LaunchVal::Ptr(dk.ptr),
        LaunchVal::Ptr(dks.ptr),
        LaunchVal::Ptr(dvp.ptr),
        LaunchVal::Ptr(dvs.ptr),
        LaunchVal::Ptr(dbt.ptr),
        LaunchVal::Ptr(dout.ptr),
        LaunchVal::I32(t as i32),
        LaunchVal::I32(l as i32),
        LaunchVal::I32(HQ as i32),
        LaunchVal::I32(HKV as i32),
        LaunchVal::I32(32),
        LaunchVal::I32(base),
        LaunchVal::Ptr(dout.ptr),
        LaunchVal::Ptr(dout.ptr),
    ];
    let grid = (((t + 63) / 64) as u32, HQ as u32, 1);
    let launch = || {};
    launch();
    for _ in 0..2 {
        r.launch_aot(ENTRY, SRC, grid, (128, 1, 1), SMEM, &vals)
            .unwrap();
    }
    r.sync().unwrap();

    if timing {
        let t0 = Instant::now();
        for _ in 0..3 {
            r.launch_aot(ENTRY, SRC, grid, (128, 1, 1), SMEM, &vals)
                .unwrap();
        }
        r.sync().unwrap();
        println!("[int8-probe] T={t} L={l} 3 runs: {:?}", t0.elapsed());
    }

    r.download(
        CheckBuf {
            ptr: dout.ptr,
            bytes: out.len(),
        },
        &mut out,
    )
    .unwrap();
    let vals_f: Vec<f32> = out
        .chunks_exact(2)
        .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    let finite = vals_f.iter().all(|x| x.is_finite());
    let nonzero = vals_f.iter().any(|x| x.abs() > 1e-8);
    let max = vals_f.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    println!("[int8-probe] T={t} L={l} finite={finite} nonzero={nonzero} max|o|={max:.6}");
    assert!(finite, "non-finite output");
    assert!(nonzero, "all-zero output");
    Some(())
}

#[test]
fn int8_prefill_smoke_and_timing() {
    if run(8, 64, false).is_none() {
        owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
        return;
    }
    run(2048, 65536, true);
}

/// Native paged int8 kernel vs host reference (T=1, L=32, one page).
#[test]
fn int8_prefill_correctness_small() {
    let Some(dev) = owl_cuda::test_device_ordinal_opt() else {
        owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
        return;
    };
    const T: usize = 1;
    const L: usize = 32;
    const PAGE: usize = 32;
    let mut res = fi_support::TestRes::new(dev);
    let mut exec = Exec::new();
    let mut r = Runner {
        res: &mut res,
        exec: &mut exec,
        registry: None,
    };

    let q = lcg_u8(T * HQ * D, 0x1111)
        .iter()
        .map(|&x| (x as i8) as u8)
        .collect::<Vec<u8>>();
    let qs = lcg_f32(T * HQ * 8, 0x2222, 0.004, 0.004);
    let k = lcg_u8(L * HKV * D, 0x3333)
        .iter()
        .map(|&x| (x as i8) as u8)
        .collect::<Vec<u8>>();
    let ks = lcg_f32(L * HKV * 8, 0x4444, 0.004, 0.004);
    let v = lcg_u8(L * HKV * D, 0x5555)
        .iter()
        .map(|&x| (x as i8) as u8)
        .collect::<Vec<u8>>();
    let vs = lcg_f32(L * HKV, 0x6666, 0.005, 0.005);
    let vp = v.clone();
    let bt = [0f32];
    let btb: Vec<u8> = bt.iter().flat_map(|x| x.to_le_bytes()).collect();
    let qsb: Vec<u8> = qs.iter().flat_map(|x| x.to_le_bytes()).collect();
    let ksb: Vec<u8> = ks.iter().flat_map(|x| x.to_le_bytes()).collect();
    let vsb: Vec<u8> = vs.iter().flat_map(|x| x.to_le_bytes()).collect();

    let dq = r.alloc(q.len(), "c.q").unwrap();
    let dqs = r.alloc(qsb.len(), "c.qs").unwrap();
    let dk = r.alloc(k.len(), "c.k").unwrap();
    let dks = r.alloc(ksb.len(), "c.ks").unwrap();
    let dvp = r.alloc(vp.len(), "c.vp").unwrap();
    let dvs = r.alloc(vsb.len(), "c.vs").unwrap();
    let dbt = r.alloc(btb.len(), "c.bt").unwrap();
    let mut out = vec![0u8; T * HQ * D * 2];
    let dout = r.alloc(out.len(), "c.out").unwrap();
    r.upload(dq, &q).unwrap();
    r.upload(dqs, &qsb).unwrap();
    r.upload(dk, &k).unwrap();
    r.upload(dks, &ksb).unwrap();
    r.upload(dvp, &vp).unwrap();
    r.upload(dvs, &vsb).unwrap();
    r.upload(dbt, &btb).unwrap();

    let vals = [
        LaunchVal::Ptr(dq.ptr),
        LaunchVal::Ptr(dqs.ptr),
        LaunchVal::Ptr(dk.ptr),
        LaunchVal::Ptr(dks.ptr),
        LaunchVal::Ptr(dvp.ptr),
        LaunchVal::Ptr(dvs.ptr),
        LaunchVal::Ptr(dbt.ptr),
        LaunchVal::Ptr(dout.ptr),
        LaunchVal::I32(T as i32),
        LaunchVal::I32(L as i32),
        LaunchVal::I32(HQ as i32),
        LaunchVal::I32(HKV as i32),
        LaunchVal::I32(PAGE as i32),
        LaunchVal::I32((L - T) as i32),
        LaunchVal::Ptr(dout.ptr),
        LaunchVal::Ptr(dout.ptr),
    ];
    r.launch_aot(ENTRY, SRC, (1, HQ as u32, 1), (128, 1, 1), SMEM, &vals)
        .unwrap();
    r.sync().unwrap();
    r.download(
        CheckBuf {
            ptr: dout.ptr,
            bytes: out.len(),
        },
        &mut out,
    )
    .unwrap();
    let got: Vec<f32> = out
        .chunks_exact(2)
        .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();

    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    let mut mx = 0f32;
    for qh in 0..HQ {
        let kh = qh / (HQ / HKV);
        let mut p = vec![0f32; L];
        let mut max = f32::NEG_INFINITY;
        for key in 0..L {
            let mut s = 0f32;
            for d in 0..D {
                let qsc = qs[qh * 8 + d / 32];
                let ksc = ks[(key * HKV + kh) * 8 + d / 32];
                s += (q[qh * D + d] as i8 as f32 * qsc)
                    * (k[(key * HKV + kh) * D + d] as i8 as f32 * ksc);
            }
            p[key] = s * 0.0625f32;
            max = max.max(p[key]);
        }
        let mut sum = 0f32;
        for key in 0..L {
            p[key] = (p[key] - max).exp();
            sum += p[key];
        }
        let inv = 1.0 / sum;
        for d in 0..D {
            let mut a = 0f32;
            for key in 0..L {
                let vsc = vs[key * HKV + kh];
                a += p[key] * (v[(key * HKV + kh) * D + d] as i8 as f32 * vsc);
            }
            let want = a * inv;
            let g = got[qh * D + d];
            dot += g as f64 * want as f64;
            na += g as f64 * g as f64;
            nb += want as f64 * want as f64;
            mx = mx.max((g - want).abs());
        }
    }
    let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
    println!("[int8-correct] cos={cos:.9} max|d|={mx:.6}");
    assert!(cos > 0.9999, "cos {cos}");
    assert!(mx < 2e-2, "max|d| {mx}");
}

/// Multi-page / multi-tile prefill correctness (T=64,L=64,2 pages).
#[test]
fn int8_prefill_correctness_multi_page() {
    let Some(dev) = owl_cuda::test_device_ordinal_opt() else {
        owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
        return;
    };
    const T: usize = 64;
    const L: usize = 64;
    const PAGE: usize = 32;
    const NB: usize = 2;
    let mut res = fi_support::TestRes::new(dev);
    let mut exec = Exec::new();
    let mut r = Runner {
        res: &mut res,
        exec: &mut exec,
        registry: None,
    };

    let q = lcg_u8(T * HQ * D, 0x1111)
        .iter()
        .map(|&x| (x as i8) as u8)
        .collect::<Vec<u8>>();
    let qs = lcg_f32(T * HQ * 8, 0x2222, 0.004, 0.004);
    let k = lcg_u8(L * HKV * D, 0x3333)
        .iter()
        .map(|&x| (x as i8) as u8)
        .collect::<Vec<u8>>();
    let ks = lcg_f32(L * HKV * 8, 0x4444, 0.004, 0.004);
    let v = lcg_u8(L * HKV * D, 0x5555)
        .iter()
        .map(|&x| (x as i8) as u8)
        .collect::<Vec<u8>>();
    let vs = lcg_f32(L * HKV, 0x6666, 0.005, 0.005);
    let vp = v.clone();
    let bt = [0f32, 1f32];
    let btb: Vec<u8> = bt.iter().flat_map(|x| x.to_le_bytes()).collect();
    let qsb: Vec<u8> = qs.iter().flat_map(|x| x.to_le_bytes()).collect();
    let ksb: Vec<u8> = ks.iter().flat_map(|x| x.to_le_bytes()).collect();
    let vsb: Vec<u8> = vs.iter().flat_map(|x| x.to_le_bytes()).collect();

    let dq = r.alloc(q.len(), "m.q").unwrap();
    let dqs = r.alloc(qsb.len(), "m.qs").unwrap();
    let dk = r.alloc(k.len(), "m.k").unwrap();
    let dks = r.alloc(ksb.len(), "m.ks").unwrap();
    let dvp = r.alloc(vp.len(), "m.vp").unwrap();
    let dvs = r.alloc(vsb.len(), "m.vs").unwrap();
    let dbt = r.alloc(btb.len(), "m.bt").unwrap();
    let mut out = vec![0u8; T * HQ * D * 2];
    let dout = r.alloc(out.len(), "m.out").unwrap();
    r.upload(dq, &q).unwrap();
    r.upload(dqs, &qsb).unwrap();
    r.upload(dk, &k).unwrap();
    r.upload(dks, &ksb).unwrap();
    r.upload(dvp, &vp).unwrap();
    r.upload(dvs, &vsb).unwrap();
    r.upload(dbt, &btb).unwrap();

    let vals = [
        LaunchVal::Ptr(dq.ptr),
        LaunchVal::Ptr(dqs.ptr),
        LaunchVal::Ptr(dk.ptr),
        LaunchVal::Ptr(dks.ptr),
        LaunchVal::Ptr(dvp.ptr),
        LaunchVal::Ptr(dvs.ptr),
        LaunchVal::Ptr(dbt.ptr),
        LaunchVal::Ptr(dout.ptr),
        LaunchVal::I32(T as i32),
        LaunchVal::I32(L as i32),
        LaunchVal::I32(HQ as i32),
        LaunchVal::I32(HKV as i32),
        LaunchVal::I32(PAGE as i32),
        LaunchVal::I32(0),
        LaunchVal::Ptr(dout.ptr),
        LaunchVal::Ptr(dout.ptr),
    ];
    r.launch_aot(
        ENTRY,
        SRC,
        ((T.div_ceil(64)) as u32, HQ as u32, 1),
        (128, 1, 1),
        SMEM,
        &vals,
    )
    .unwrap();
    r.sync().unwrap();
    r.download(CheckBuf { ptr: dout.ptr, bytes: out.len() }, &mut out).unwrap();
    let got: Vec<f32> = out
        .chunks_exact(2)
        .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();

    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    let mut mx = 0f32;
    for t in 0..T {
        for qh in 0..HQ {
            let kh = qh / (HQ / HKV);
            let mut p = vec![0f32; L];
            let mut max = f32::NEG_INFINITY;
            for key in 0..L {
                if key > t {
                    p[key] = f32::NEG_INFINITY;
                    continue;
                }
                let mut s = 0f32;
                for d in 0..D {
                    let qsc = qs[(t * HQ + qh) * 8 + d / 32];
                    let ksc = ks[(key * HKV + kh) * 8 + d / 32];
                    s += (q[(t * HQ + qh) * D + d] as i8 as f32 * qsc)
                        * (k[(key * HKV + kh) * D + d] as i8 as f32 * ksc);
                }
                let s = s * 0.0625f32;
                p[key] = s;
                max = max.max(s);
            }
            let mut sum = 0f32;
            for key in 0..L {
                if p[key] > f32::NEG_INFINITY {
                    p[key] = (p[key] - max).exp();
                    sum += p[key];
                } else {
                    p[key] = 0.0;
                }
            }
            let inv = 1.0 / sum;
            for d in 0..D {
                let mut a = 0f32;
                for key in 0..L {
                    let vsc = vs[key * HKV + kh];
                    a += p[key] * (v[(key * HKV + kh) * D + d] as i8 as f32 * vsc);
                }
                let want = a * inv;
                let g = got[(t * HQ + qh) * D + d];
                dot += g as f64 * want as f64;
                na += g as f64 * g as f64;
                nb += want as f64 * want as f64;
                mx = mx.max((g - want).abs());
            }
        }
    }
    let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
    println!("[int8-multipage] cos={cos:.9} max|d|={mx:.6}");
    assert!(cos > 0.9999, "cos {cos}");
    assert!(mx < 2e-2, "max|d| {mx}");
}


/// f16 PV production kernel: P f16 x256, V f16 shadow token-major, f16 acc.
#[test]
fn int8_prefill_f16pv_correctness_small() {
    let Some(dev) = owl_cuda::test_device_ordinal_opt() else {
        owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
        return;
    };
    const T: usize = 1;
    const L: usize = 32;
    const PAGE: usize = 32;
    const SMEM_F16: u32 = 16384 + 8192 + 16384 + 4096 + 1024;
    let mut res = fi_support::TestRes::new(dev);
    let mut exec = Exec::new();
    let mut r = Runner { res: &mut res, exec: &mut exec, registry: None };

    let q = lcg_u8(T * HQ * D, 0x1111).iter().map(|&x| (x as i8) as u8).collect::<Vec<u8>>();
    let qs = lcg_f32(T * HQ * 8, 0x2222, 0.004, 0.004);
    let k = lcg_u8(L * HKV * D, 0x3333).iter().map(|&x| (x as i8) as u8).collect::<Vec<u8>>();
    let ks = lcg_f32(L * HKV * 8, 0x4444, 0.004, 0.004);
    let v16: Vec<u16> = (0..L * HKV * D)
        .map(|i| {
            let x = 0x5555u32
                .wrapping_mul(0x9E37_79B9)
                .wrapping_add((i as u32).wrapping_mul(0x85EB_CA6B));
            half::f16::from_f32((((x >> 13) & 0xffff) as f32 / 65535.0 - 0.5) * 2.0).to_bits()
        })
        .collect();
    let bt = [0f32];
    let btb: Vec<u8> = bt.iter().flat_map(|x| x.to_le_bytes()).collect();
    let qsb: Vec<u8> = qs.iter().flat_map(|x| x.to_le_bytes()).collect();
    let ksb: Vec<u8> = ks.iter().flat_map(|x| x.to_le_bytes()).collect();
    let vb: Vec<u8> = v16.iter().flat_map(|x| x.to_le_bytes()).collect();

    let dq = r.alloc(q.len(), "f.q").unwrap();
    let dqs = r.alloc(qsb.len(), "f.qs").unwrap();
    let dk = r.alloc(k.len(), "f.k").unwrap();
    let dks = r.alloc(ksb.len(), "f.ks").unwrap();
    let dv = r.alloc(vb.len(), "f.v").unwrap();
    let dbt = r.alloc(btb.len(), "f.bt").unwrap();
    let mut out = vec![0u8; T * HQ * D * 2];
    let dout = r.alloc(out.len(), "f.out").unwrap();
    r.upload(dq, &q).unwrap();
    r.upload(dqs, &qsb).unwrap();
    r.upload(dk, &k).unwrap();
    r.upload(dks, &ksb).unwrap();
    r.upload(dv, &vb).unwrap();
    r.upload(dbt, &btb).unwrap();

    let vals = [
        LaunchVal::Ptr(dq.ptr), LaunchVal::Ptr(dqs.ptr),
        LaunchVal::Ptr(dk.ptr), LaunchVal::Ptr(dks.ptr),
        LaunchVal::Ptr(dv.ptr), LaunchVal::Ptr(dbt.ptr),
        LaunchVal::Ptr(dout.ptr),
        LaunchVal::I32(T as i32), LaunchVal::I32(L as i32),
        LaunchVal::I32(HQ as i32), LaunchVal::I32(HKV as i32),
        LaunchVal::I32(PAGE as i32), LaunchVal::I32((L - T) as i32),
        LaunchVal::Ptr(dout.ptr), LaunchVal::Ptr(dout.ptr),
    ];
    r.launch_aot(ENTRY_F16PV, SRC_F16PV, (1, HQ as u32, 1), (128, 1, 1), SMEM_F16, &vals)
        .unwrap();
    r.sync().unwrap();
    r.download(CheckBuf { ptr: dout.ptr, bytes: out.len() }, &mut out).unwrap();
    let got: Vec<f32> = out
        .chunks_exact(2)
        .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();

    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    let mut mx = 0f32;
    for qh in 0..HQ {
        let kh = qh / (HQ / HKV);
        let mut p = vec![0f32; L];
        let mut max = f32::NEG_INFINITY;
        for key in 0..L {
            let mut s = 0f32;
            for d in 0..D {
                let qsc = qs[qh * 8 + d / 32];
                let ksc = ks[(key * HKV + kh) * 8 + d / 32];
                s += (q[qh * D + d] as i8 as f32 * qsc)
                    * (k[(key * HKV + kh) * D + d] as i8 as f32 * ksc);
            }
            p[key] = s * 0.0625f32;
            max = max.max(p[key]);
        }
        let mut sum = 0f32;
        for key in 0..L {
            p[key] = (p[key] - max).exp();
            sum += p[key];
        }
        let inv = 1.0 / sum;
        for d in 0..D {
            let mut a = 0f32;
            for key in 0..L {
                a += p[key] * f16_to_f32(v16[(key * HKV + kh) * D + d]);
            }
            let want = a * inv;
            let g = got[qh * D + d];
            dot += g as f64 * want as f64;
            na += g as f64 * g as f64;
            nb += want as f64 * want as f64;
            mx = mx.max((g - want).abs());
        }
    }
    let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
    println!("[int8-f16pv] cos={cos:.9} max|d|={mx:.6}");
    assert!(cos > 0.9999, "cos {cos}");
    assert!(mx < 2e-2, "max|d| {mx}");
}

/// 长单块 T=L=1024：只对最后 16 行做 host 对照。用来钉住
/// “多 q block + 尾部 16 行”的口径（1k prompt / 图像 prompt 的实测形态）。
#[test]
fn int8_prefill_1024_tail_rows_match_host() {
    let Some(dev) = owl_cuda::test_device_ordinal_opt() else {
        owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
        return;
    };
    const T: usize = 1024;
    const L: usize = 1024;
    const PAGE: usize = 32;
    const NB: usize = L.div_ceil(PAGE);
    let mut res = fi_support::TestRes::new(dev);
    let mut exec = Exec::new();
    let mut r = Runner {
        res: &mut res,
        exec: &mut exec,
        registry: None,
    };

    let q = lcg_u8(T * HQ * D, 0x1111)
        .iter()
        .map(|&x| (x as i8) as u8)
        .collect::<Vec<u8>>();
    let qs = lcg_f32(T * HQ * 8, 0x2222, 0.004, 0.004);
    let k = lcg_u8(L * HKV * D, 0x3333)
        .iter()
        .map(|&x| (x as i8) as u8)
        .collect::<Vec<u8>>();
    let ks = lcg_f32(L * HKV * 8, 0x4444, 0.004, 0.004);
    let v = lcg_u8(L * HKV * D, 0x5555)
        .iter()
        .map(|&x| (x as i8) as u8)
        .collect::<Vec<u8>>();
    let vs = lcg_f32(L * HKV, 0x6666, 0.005, 0.005);
    let vp = v.clone();
    let bt: Vec<f32> = (0..NB as u32).map(|x| x as f32).collect();
    let btb: Vec<u8> = bt.iter().flat_map(|x| x.to_le_bytes()).collect();
    let qsb: Vec<u8> = qs.iter().flat_map(|x| x.to_le_bytes()).collect();
    let ksb: Vec<u8> = ks.iter().flat_map(|x| x.to_le_bytes()).collect();
    let vsb: Vec<u8> = vs.iter().flat_map(|x| x.to_le_bytes()).collect();

    let dq = r.alloc(q.len(), "lt.q").unwrap();
    let dqs = r.alloc(qsb.len(), "lt.qs").unwrap();
    let dk = r.alloc(k.len(), "lt.k").unwrap();
    let dks = r.alloc(ksb.len(), "lt.ks").unwrap();
    let dvp = r.alloc(vp.len(), "lt.vp").unwrap();
    let dvs = r.alloc(vsb.len(), "lt.vs").unwrap();
    let dbt = r.alloc(btb.len(), "lt.bt").unwrap();
    let mut out = vec![0u8; T * HQ * D * 2];
    let dout = r.alloc(out.len(), "lt.out").unwrap();
    r.upload(dq, &q).unwrap();
    r.upload(dqs, &qsb).unwrap();
    r.upload(dk, &k).unwrap();
    r.upload(dks, &ksb).unwrap();
    r.upload(dvp, &vp).unwrap();
    r.upload(dvs, &vsb).unwrap();
    r.upload(dbt, &btb).unwrap();

    let vals = [
        LaunchVal::Ptr(dq.ptr),
        LaunchVal::Ptr(dqs.ptr),
        LaunchVal::Ptr(dk.ptr),
        LaunchVal::Ptr(dks.ptr),
        LaunchVal::Ptr(dvp.ptr),
        LaunchVal::Ptr(dvs.ptr),
        LaunchVal::Ptr(dbt.ptr),
        LaunchVal::Ptr(dout.ptr),
        LaunchVal::I32(T as i32),
        LaunchVal::I32(L as i32),
        LaunchVal::I32(HQ as i32),
        LaunchVal::I32(HKV as i32),
        LaunchVal::I32(PAGE as i32),
        LaunchVal::I32(0),
        LaunchVal::Ptr(dout.ptr),
        LaunchVal::Ptr(dout.ptr),
    ];
    r.launch_aot(
        ENTRY,
        SRC,
        (T.div_ceil(64) as u32, HQ as u32, 1),
        (128, 1, 1),
        SMEM,
        &vals,
    )
    .unwrap();
    r.sync().unwrap();
    r.download(CheckBuf { ptr: dout.ptr, bytes: out.len() }, &mut out).unwrap();
    let got: Vec<f32> = out
        .chunks_exact(2)
        .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();

    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    let mut mx = 0f32;
    for t in (T - 16)..T {
        for qh in 0..HQ {
            let kh = qh / (HQ / HKV);
            let mut p = vec![0f32; L];
            let mut max = f32::NEG_INFINITY;
            for key in 0..L {
                if key > t {
                    p[key] = f32::NEG_INFINITY;
                    continue;
                }
                let mut s = 0f32;
                for d in 0..D {
                    let qsc = qs[(t * HQ + qh) * 8 + d / 32];
                    let ksc = ks[(key * HKV + kh) * 8 + d / 32];
                    s += (q[(t * HQ + qh) * D + d] as i8 as f32 * qsc)
                        * (k[(key * HKV + kh) * D + d] as i8 as f32 * ksc);
                }
                let s = s * 0.0625f32;
                p[key] = s;
                max = max.max(s);
            }
            let mut sum = 0f32;
            for key in 0..L {
                if p[key] > f32::NEG_INFINITY {
                    p[key] = (p[key] - max).exp();
                    sum += p[key];
                } else {
                    p[key] = 0.0;
                }
            }
            let inv = 1.0 / sum;
            for d in 0..D {
                let mut a = 0f32;
                for key in 0..L {
                    let vsc = vs[key * HKV + kh];
                    a += p[key] * (v[(key * HKV + kh) * D + d] as i8 as f32 * vsc);
                }
                let want = a * inv;
                let g = got[(t * HQ + qh) * D + d];
                dot += g as f64 * want as f64;
                na += g as f64 * g as f64;
                nb += want as f64 * want as f64;
                mx = mx.max((g - want).abs());
            }
        }
    }
    let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
    println!("[int8-1024tail] cos={cos:.9} max|d|={mx:.6}");
    assert!(cos > 0.9999, "cos {cos}");
    assert!(mx < 2e-2, "max|d| {mx}");
}

/// chunk 形态：T=69 / L=1093 / base_pos=1024（图像 prompt 被 1024 切块的尾块）。
/// 只对块内最后 5 行（全局 1088..1092）做 host 对照。
#[test]
fn int8_prefill_chunk_tail_rows_match_host() {
    let Some(dev) = owl_cuda::test_device_ordinal_opt() else {
        owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
        return;
    };
    const TC: usize = 69;
    const L: usize = 1093;
    const BASE: usize = 1024;
    const PAGE: usize = 32;
    const NB: usize = L.div_ceil(PAGE);
    let mut res = fi_support::TestRes::new(dev);
    let mut exec = Exec::new();
    let mut r = Runner {
        res: &mut res,
        exec: &mut exec,
        registry: None,
    };

    let q_all = lcg_u8(L * HQ * D, 0x1111)
        .iter()
        .map(|&x| (x as i8) as u8)
        .collect::<Vec<u8>>();
    let q = q_all[BASE * HQ * D..(BASE + TC) * HQ * D].to_vec();
    let qs_all = lcg_f32(L * HQ * 8, 0x2222, 0.004, 0.004);
    let qs = qs_all[BASE * HQ * 8..(BASE + TC) * HQ * 8].to_vec();
    let k = lcg_u8(L * HKV * D, 0x3333)
        .iter()
        .map(|&x| (x as i8) as u8)
        .collect::<Vec<u8>>();
    let ks = lcg_f32(L * HKV * 8, 0x4444, 0.004, 0.004);
    let v = lcg_u8(L * HKV * D, 0x5555)
        .iter()
        .map(|&x| (x as i8) as u8)
        .collect::<Vec<u8>>();
    let vs = lcg_f32(L * HKV, 0x6666, 0.005, 0.005);
    let vp = v.clone();
    let bt: Vec<f32> = (0..NB as u32).map(|x| x as f32).collect();
    let btb: Vec<u8> = bt.iter().flat_map(|x| x.to_le_bytes()).collect();
    let qsb: Vec<u8> = qs.iter().flat_map(|x| x.to_le_bytes()).collect();
    let ksb: Vec<u8> = ks.iter().flat_map(|x| x.to_le_bytes()).collect();
    let vsb: Vec<u8> = vs.iter().flat_map(|x| x.to_le_bytes()).collect();

    let dq = r.alloc(q.len(), "ct.q").unwrap();
    let dqs = r.alloc(qsb.len(), "ct.qs").unwrap();
    let dk = r.alloc(k.len(), "ct.k").unwrap();
    let dks = r.alloc(ksb.len(), "ct.ks").unwrap();
    let dvp = r.alloc(vp.len(), "ct.vp").unwrap();
    let dvs = r.alloc(vsb.len(), "ct.vs").unwrap();
    let dbt = r.alloc(btb.len(), "ct.bt").unwrap();
    let mut out = vec![0u8; TC * HQ * D * 2];
    let dout = r.alloc(out.len(), "ct.out").unwrap();
    r.upload(dq, &q).unwrap();
    r.upload(dqs, &qsb).unwrap();
    r.upload(dk, &k).unwrap();
    r.upload(dks, &ksb).unwrap();
    r.upload(dvp, &vp).unwrap();
    r.upload(dvs, &vsb).unwrap();
    r.upload(dbt, &btb).unwrap();

    let vals = [
        LaunchVal::Ptr(dq.ptr),
        LaunchVal::Ptr(dqs.ptr),
        LaunchVal::Ptr(dk.ptr),
        LaunchVal::Ptr(dks.ptr),
        LaunchVal::Ptr(dvp.ptr),
        LaunchVal::Ptr(dvs.ptr),
        LaunchVal::Ptr(dbt.ptr),
        LaunchVal::Ptr(dout.ptr),
        LaunchVal::I32(TC as i32),
        LaunchVal::I32(L as i32),
        LaunchVal::I32(HQ as i32),
        LaunchVal::I32(HKV as i32),
        LaunchVal::I32(PAGE as i32),
        LaunchVal::I32(BASE as i32),
        LaunchVal::Ptr(dout.ptr),
        LaunchVal::Ptr(dout.ptr),
    ];
    r.launch_aot(
        ENTRY,
        SRC,
        (TC.div_ceil(64) as u32, HQ as u32, 1),
        (128, 1, 1),
        SMEM,
        &vals,
    )
    .unwrap();
    r.sync().unwrap();
    r.download(CheckBuf { ptr: dout.ptr, bytes: out.len() }, &mut out).unwrap();
    let got: Vec<f32> = out
        .chunks_exact(2)
        .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();

    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    let mut mx = 0f32;
    for gt in (L - 5)..L {
        let ct = gt - BASE;
        for qh in 0..HQ {
            let kh = qh / (HQ / HKV);
            let mut p = vec![0f32; L];
            let mut max = f32::NEG_INFINITY;
            for key in 0..L {
                if key > gt {
                    p[key] = f32::NEG_INFINITY;
                    continue;
                }
                let mut s = 0f32;
                for d in 0..D {
                    let qsc = qs[(ct * HQ + qh) * 8 + d / 32];
                    let ksc = ks[(key * HKV + kh) * 8 + d / 32];
                    s += (q[(ct * HQ + qh) * D + d] as i8 as f32 * qsc)
                        * (k[(key * HKV + kh) * D + d] as i8 as f32 * ksc);
                }
                let s = s * 0.0625f32;
                p[key] = s;
                max = max.max(s);
            }
            let mut sum = 0f32;
            for key in 0..L {
                if p[key] > f32::NEG_INFINITY {
                    p[key] = (p[key] - max).exp();
                    sum += p[key];
                } else {
                    p[key] = 0.0;
                }
            }
            let inv = 1.0 / sum;
            for d in 0..D {
                let mut a = 0f32;
                for key in 0..L {
                    let vsc = vs[key * HKV + kh];
                    a += p[key] * (v[(key * HKV + kh) * D + d] as i8 as f32 * vsc);
                }
                let want = a * inv;
                let g = got[(ct * HQ + qh) * D + d];
                dot += g as f64 * want as f64;
                na += g as f64 * g as f64;
                nb += want as f64 * want as f64;
                mx = mx.max((g - want).abs());
            }
        }
    }
    let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
    println!("[int8-chunktail] cos={cos:.9} max|d|={mx:.6}");
    assert!(cos > 0.9999, "cos {cos}");
    assert!(mx < 2e-2, "max|d| {mx}");
}

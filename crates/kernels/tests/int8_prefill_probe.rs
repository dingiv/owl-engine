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
const D: usize = 256;
const HQ: usize = 24;
const HKV: usize = 4;
const SMEM: u32 = 16384 + 16384 + 2048 + 128;

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
    let qs = lcg_f32(t * HQ, 0x2345, 0.004, 0.004);
    let k = lcg_u8(l * HKV * D, 0x3456);
    let ks = lcg_f32(l * HKV, 0x4567, 0.004, 0.004);
    // V prepacked [Hkv][D][L/2] b16, random bytes.
    let page = 32usize;
    let vp = lcg_u8(HKV * D * (l / 2) * 2, 0x5678);
    let nblocks = l / page;
    let bt: Vec<f32> = (0..nblocks as u32).map(|x| x as f32).collect();
    let btb: Vec<u8> = bt.iter().flat_map(|x| x.to_le_bytes()).collect();
    let vs = lcg_f32(nblocks * HKV * 4, 0x6789, 0.005, 0.005);
    let mut out = vec![0u8; t * HQ * D * 4];

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
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
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
    let qs = lcg_f32(T * HQ, 0x2222, 0.004, 0.004);
    let k = lcg_u8(L * HKV * D, 0x3333)
        .iter()
        .map(|&x| (x as i8) as u8)
        .collect::<Vec<u8>>();
    let ks = lcg_f32(L * HKV, 0x4444, 0.004, 0.004);
    let v = lcg_u8(L * HKV * D, 0x5555)
        .iter()
        .map(|&x| (x as i8) as u8)
        .collect::<Vec<u8>>();
    let vs = lcg_f32(HKV * 4, 0x6666, 0.005, 0.005); // one page
    let pp = PAGE / 2;
    let mut vp = vec![0u8; HKV * D * pp * 2];
    for h in 0..HKV {
        for d in 0..D {
            for pair in 0..pp {
                let k0 = pair * 2;
                let k1 = k0 + 1;
                let a = v[(k0 * HKV + h) * D + d];
                let b = v[(k1 * HKV + h) * D + d];
                let off = ((h * D + d) * pp + pair) * 2;
                vp[off] = a;
                vp[off + 1] = b;
            }
        }
    }
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
    let mut out = vec![0u8; T * HQ * D * 4];
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
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    let mut mx = 0f32;
    for qh in 0..HQ {
        let kh = qh / (HQ / HKV);
        let qsc = qs[qh];
        let mut p = vec![0f32; L];
        let mut max = f32::NEG_INFINITY;
        for key in 0..L {
            let ksc = ks[key * HKV + kh];
            let mut s = 0f32;
            for d in 0..D {
                s += (q[qh * D + d] as i8 as f32 * qsc)
                    * (k[(key * HKV + kh) * D + d] as i8 as f32 * ksc);
            }
            p[key] = s;
            max = max.max(s);
        }
        let mut sum = 0f32;
        for key in 0..L {
            p[key] = (p[key] - max).exp();
            sum += p[key];
        }
        let inv = 1.0 / sum;
        for d in 0..D {
            let vsc = vs[kh * 4 + d / 64]; // page 0
            let mut a = 0f32;
            for key in 0..L {
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

//! INT8 KV 生产写者三件套:Q per-head 量化 / K-only writer / 整页 V pack。
//! host 独立参考逐字节对拍(不复用核内量化代码)。
#![cfg(feature = "device")]

mod fi_support;

use owl_kernels::bootcheck::{CheckBuf, CheckRunner};
use owl_kernels::contract::LaunchVal;
use owl_kernels::device::Exec;
use owl_kernels::server::selfcheck::Runner;

const D: usize = 256;
const HKV: usize = 2;
const PAGE: usize = 32;
const NB: usize = 2;

fn f32_to_f16(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let man = (bits >> 13) & 0x3FF;
    if exp <= 0 {
        return sign;
    }
    if exp >= 31 {
        return sign | 0x7C00;
    }
    sign | ((exp as u16) << 10) | man as u16
}
fn f32_to_f16_rn(x: f32) -> u16 {
    half::f16::from_f32(x).to_bits()
}
fn f16_to_f32(h: u16) -> f32 {
    let s = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let e = ((h >> 10) & 0x1F) as i32;
    let m = (h & 0x3FF) as f32;
    if e == 0 {
        return s * m * 2f32.powi(-24);
    }
    s * (1.0 + m / 1024.0) * 2f32.powi(e - 15)
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
fn quant_half_away(x: f32, inv: f32, lo: f32) -> i8 {
    let mut q = x * inv;
    q = if q >= 0.0 { q + 0.5 } else { q - 0.5 };
    q = q.clamp(lo, 127.0);
    q as i8
}
fn hb(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[test]
fn qk_write_and_v_pack_match_host() {
    let Some(dev) = owl_cuda::test_device_ordinal_opt() else {
        owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
        return;
    };
    let mut res = fi_support::TestRes::new(dev);
    let mut exec = Exec::new();
    let mut r = Runner {
        res: &mut res,
        exec: &mut exec,
        registry: None,
    };

    // ---- Q: rows = 2 tokens * 2 heads, cols = D ----
    const ROWS: usize = 4;
    let qf = lcg_f32(ROWS * D, 0x1111, -1.0, 2.0);
    let qh: Vec<u16> = qf.iter().map(|&x| f32_to_f16(x)).collect();
    let mut q_want = vec![0i8; ROWS * D];
    let mut qs_want = vec![0f32; ROWS * 8];
    for row in 0..ROWS {
        for g in 0..8 {
            let mut amax = 0f32;
            for d in g * 32..(g + 1) * 32 {
                amax = amax.max(f16_to_f32(qh[row * D + d]).abs());
            }
            let s = (amax / 127.0).max(1e-6);
            let inv = 1.0 / s;
            qs_want[row * 8 + g] = s;
            for d in g * 32..(g + 1) * 32 {
                q_want[row * D + d] = quant_half_away(f16_to_f32(qh[row * D + d]), inv, -127.0);
            }
        }
    }
    let dq = r.alloc(hb(&qh).len(), "p.q").unwrap();
    let dqs = r.alloc(ROWS * 8 * 4, "p.qs").unwrap();
    let dqo = r.alloc(ROWS * D, "p.qo").unwrap();
    r.upload(dq, &hb(&qh)).unwrap();
    let qv = [
        LaunchVal::Ptr(dq.ptr),
        LaunchVal::I32(ROWS as i32),
        LaunchVal::I32(D as i32),
        LaunchVal::Ptr(dqs.ptr),
        LaunchVal::Ptr(dqo.ptr),
    ];
    r.launch_aot(
        "owl_quant_int8_qk_f16",
        "cu/attn/quant_int8_qk.cu",
        (ROWS as u32, 1, 1),
        (256, 1, 1),
        0,
        &qv,
    )
    .unwrap();
    r.sync().unwrap();
    let mut got_q = vec![0u8; ROWS * D];
    let mut got_qs = vec![0u8; ROWS * 8 * 4];
    r.download(
        CheckBuf {
            ptr: dqo.ptr,
            bytes: got_q.len(),
        },
        &mut got_q,
    )
    .unwrap();
    r.download(
        CheckBuf {
            ptr: dqs.ptr,
            bytes: got_qs.len(),
        },
        &mut got_qs,
    )
    .unwrap();
    let got_q: Vec<i8> = got_q.iter().map(|&b| b as i8).collect();
    let got_qs: Vec<f32> = got_qs
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(got_q, q_want, "Q int8 字节");
    for (i, (g, w)) in got_qs.iter().zip(&qs_want).enumerate() {
        assert!((g - w).abs() <= 1e-7, "qscale[{i}] {g} != {w}");
    }

    // ---- K-only writer: T=2, slots [0, 33] ----
    const T: usize = 2;
    let kf = lcg_f32(T * HKV * D, 0x2222, -2.0, 4.0);
    let kh: Vec<u16> = kf.iter().map(|&x| f32_to_f16(x)).collect();
    let slots = [0f32, 33.0];
    let mut k_want = vec![0x7Fi8; NB * PAGE * HKV * D];
    let mut ks_want = vec![0f32; NB * PAGE * HKV * 8];
    for t in 0..T {
        let slot = slots[t] as usize;
        let (blk, off) = (slot / PAGE, slot % PAGE);
        for h in 0..HKV {
            let dst = ((blk * PAGE + off) * HKV + h) * D;
            for g in 0..8 {
                let mut amax = 0f32;
                for d in g * 32..(g + 1) * 32 {
                    amax = amax.max(f16_to_f32(kh[(t * HKV + h) * D + d]).abs());
                }
                let s = (amax / 127.0).max(1e-6);
                let inv = 1.0 / s;
                for d in g * 32..(g + 1) * 32 {
                    k_want[dst + d] =
                        quant_half_away(f16_to_f32(kh[(t * HKV + h) * D + d]), inv, -128.0);
                }
                ks_want[((blk * PAGE + off) * HKV + h) * 8 + g] = s;
            }
        }
    }
    let dk = r.alloc(hb(&kh).len(), "p.k").unwrap();
    let dkc = r.alloc(k_want.len(), "p.kc").unwrap();
    let dks = r.alloc(ks_want.len() * 4, "p.ks").unwrap();
    let dslots = r.alloc(T * 4, "p.slots").unwrap();
    let dout = r.alloc(2, "p.out").unwrap();
    r.upload(dk, &hb(&kh)).unwrap();
    r.upload(dkc, &vec![0x7Fu8; k_want.len()]).unwrap();
    r.upload(dks, &vec![0u8; ks_want.len() * 4]).unwrap();
    r.upload(
        dslots,
        &slots
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<u8>>(),
    )
    .unwrap();
    let kv = [
        LaunchVal::Ptr(dk.ptr),
        LaunchVal::Ptr(dkc.ptr),
        LaunchVal::Ptr(dks.ptr),
        LaunchVal::Ptr(dslots.ptr),
        LaunchVal::I32((HKV * D) as i32),
        LaunchVal::I32(HKV as i32),
        LaunchVal::I32(D as i32),
        LaunchVal::I32(PAGE as i32),
        LaunchVal::I32(T as i32),
        LaunchVal::Ptr(dout.ptr),
    ];
    r.launch_aot(
        "owl_reshape_and_cache_int8_k_only_knhd",
        "cu/attn/reshape_and_cache_int8.cu",
        ((T * HKV + 3) as u32 / 4, 1, 1),
        (128, 1, 1),
        0,
        &kv,
    )
    .unwrap();
    r.sync().unwrap();
    let mut got_k = vec![0u8; k_want.len()];
    let mut got_ks = vec![0u8; ks_want.len() * 4];
    r.download(
        CheckBuf {
            ptr: dkc.ptr,
            bytes: got_k.len(),
        },
        &mut got_k,
    )
    .unwrap();
    r.download(
        CheckBuf {
            ptr: dks.ptr,
            bytes: got_ks.len(),
        },
        &mut got_ks,
    )
    .unwrap();
    let got_k: Vec<i8> = got_k.iter().map(|&b| b as i8).collect();
    if got_k != k_want {
        let i = got_k
            .iter()
            .zip(&k_want)
            .position(|(a, b)| a != b)
            .unwrap_or(0);
        eprintln!("[k-mismatch] i={i} got={} want={}", got_k[i], k_want[i]);
        eprintln!(
            "[k-got]  {:?}",
            &got_k[i.saturating_sub(4)..(i + 12).min(got_k.len())]
        );
        eprintln!(
            "[k-want] {:?}",
            &k_want[i.saturating_sub(4)..(i + 12).min(k_want.len())]
        );
        panic!("K int8 字节");
    }
    let got_ks: Vec<f32> = got_ks
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    for (i, (g, w)) in got_ks.iter().zip(&ks_want).enumerate() {
        assert!((g - w).abs() <= 1e-7, "kscale[{i}] {g} != {w}");
    }

    // ---- V prefill pack: one full page at physical page 1 ----
    const TV: usize = PAGE;
    let vf = lcg_f32(TV * HKV * D, 0x3333, -1.5, 3.0);
    let vh: Vec<u16> = vf.iter().map(|&x| f32_to_f16(x)).collect();
    let vslots: Vec<f32> = (PAGE..2 * PAGE).map(|x| x as f32).collect();
    let pp = PAGE / 2;
    let mut v_want = vec![0x7Fu8; 2 * HKV * D * pp * 2];
    let mut vs_want = vec![0f32; NB * PAGE * HKV];
    for h in 0..HKV {
        for i in 0..TV {
            let mut amax = 0f32;
            for d in 0..D {
                amax = amax.max(f16_to_f32(vh[(i * HKV + h) * D + d]).abs());
            }
            let scale = (amax / 127.0).max(1e-6);
            let inv = 1.0 / scale;
            vs_want[(PAGE + i) * HKV + h] = scale;
            for d in 0..D {
                let a = quant_half_away(f16_to_f32(vh[(i * HKV + h) * D + d]), inv, -127.0) as u8;
                let off = (((1 * HKV + h) * D + d) * pp + (i / 2)) * 2;
                if (i & 1) == 0 {
                    v_want[off] = a;
                } else {
                    v_want[off + 1] = a;
                }
            }
        }
    }
    let dv = r.alloc(hb(&vh).len(), "p.v").unwrap();
    let dvp = r.alloc(v_want.len(), "p.vp").unwrap();
    let dvs = r.alloc(vs_want.len() * 4, "p.vs").unwrap();
    let dvslots = r.alloc(TV * 4, "p.vslots").unwrap();
    r.upload(dv, &hb(&vh)).unwrap();
    r.upload(dvp, &vec![0x7Fu8; v_want.len()]).unwrap();
    r.upload(dvs, &vec![0u8; vs_want.len() * 4]).unwrap();
    r.upload(
        dvslots,
        &vslots
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<u8>>(),
    )
    .unwrap();
    let vv = [
        LaunchVal::Ptr(dv.ptr),
        LaunchVal::Ptr(dvslots.ptr),
        LaunchVal::I32(TV as i32),
        LaunchVal::I32(HKV as i32),
        LaunchVal::I32(D as i32),
        LaunchVal::I32(PAGE as i32),
        LaunchVal::Ptr(dvs.ptr),
        LaunchVal::Ptr(dvp.ptr),
        LaunchVal::Ptr(dout.ptr),
    ];
    r.launch_aot(
        "owl_v_pack_int8_prefill",
        "cu/attn/pack_v_int8_paged.cu",
        (1, HKV as u32, 1),
        (256, 1, 1),
        0,
        &vv,
    )
    .unwrap();
    r.sync().unwrap();
    let mut got_vp = vec![0u8; v_want.len()];
    let mut got_vs = vec![0u8; vs_want.len() * 4];
    r.download(
        CheckBuf {
            ptr: dvp.ptr,
            bytes: got_vp.len(),
        },
        &mut got_vp,
    )
    .unwrap();
    r.download(
        CheckBuf {
            ptr: dvs.ptr,
            bytes: got_vs.len(),
        },
        &mut got_vs,
    )
    .unwrap();
    assert_eq!(got_vp, v_want, "V packed 字节");
    let got_vs: Vec<f32> = got_vs
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    for (i, (g, w)) in got_vs.iter().zip(&vs_want).enumerate() {
        assert!((g - w).abs() <= 1e-7, "vscale[{i}] {g} != {w}");
    }

    println!("[int8-pipeline] Q/K/V writers bit-exact OK");
}

/// INT8 decode:2 页 / 33 keys,对照 host 参考(无 /16)。
#[test]
fn int8_decode_matches_host() {
    let Some(dev) = owl_cuda::test_device_ordinal_opt() else {
        owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
        return;
    };
    const B: usize = 1;
    const HQ: usize = 4;
    const HK: usize = 2;
    const L: usize = 4096;
    const PAGES: usize = 128;
    let mut res = fi_support::TestRes::new(dev);
    let mut exec = Exec::new();
    let mut r = Runner {
        res: &mut res,
        exec: &mut exec,
        registry: None,
    };

    let q: Vec<i8> = (0..B * HQ * D)
        .map(|i| ((i as i32 * 37 % 255) - 127) as i8)
        .collect();
    let qs: Vec<f32> = (0..B * HQ * 8).map(|i| 0.003 + 0.0001 * i as f32).collect();
    let k: Vec<i8> = (0..PAGES * PAGE * HK * D)
        .map(|i| ((i as i32 * 53 % 255) - 127) as i8)
        .collect();
    let ks: Vec<f32> = (0..PAGES * PAGE * HK * 8)
        .map(|i| 0.002 + 0.00002 * i as f32)
        .collect();
    let pp = PAGE / 2;
    let v_int: Vec<i8> = (0..PAGES * HK * D * PAGE)
        .map(|i| ((i as i32 * 29 % 255) - 127) as i8)
        .collect();
    let vs: Vec<f32> = (0..PAGES * PAGE * HK)
        .map(|i| 0.004 + 0.00003 * i as f32)
        .collect();
    let mut vp = vec![0u8; PAGES * HK * D * pp * 2];
    for pg in 0..PAGES {
        for h in 0..HK {
            for d in 0..D {
                for pair in 0..pp {
                    let key0 = pair * 2;
                    let key1 = key0 + 1;
                    let a = v_int[((pg * PAGE + key0) * HK + h) * D + d] as u8;
                    let b = v_int[((pg * PAGE + key1) * HK + h) * D + d] as u8;
                    let off = (((pg * HK + h) * D + d) * pp + pair) * 2;
                    vp[off] = a;
                    vp[off + 1] = b;
                }
            }
        }
    }
    let bt: Vec<f32> = (0..PAGES).map(|x| x as f32).collect();
    let lens = [L as f32];

    let dq = r.alloc(q.len(), "d.q").unwrap();
    let dqs = r.alloc(qs.len() * 4, "d.qs").unwrap();
    let dk = r.alloc(k.len(), "d.k").unwrap();
    let dks = r.alloc(ks.len() * 4, "d.ks").unwrap();
    let dvp = r.alloc(vp.len(), "d.vp").unwrap();
    let dvs = r.alloc(vs.len() * 4, "d.vs").unwrap();
    let dbt = r.alloc(bt.len() * 4, "d.bt").unwrap();
    let dlens = r.alloc(lens.len() * 4, "d.lens").unwrap();
    let dout = r.alloc(B * HQ * D * 4, "d.out").unwrap();
    r.upload(dq, &q.iter().map(|&x| x as u8).collect::<Vec<u8>>())
        .unwrap();
    r.upload(
        dqs,
        &qs.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>(),
    )
    .unwrap();
    r.upload(dk, &k.iter().map(|&x| x as u8).collect::<Vec<u8>>())
        .unwrap();
    r.upload(
        dks,
        &ks.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>(),
    )
    .unwrap();
    r.upload(dvp, &vp).unwrap();
    r.upload(
        dvs,
        &vs.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>(),
    )
    .unwrap();
    r.upload(
        dbt,
        &bt.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>(),
    )
    .unwrap();
    r.upload(
        dlens,
        &lens
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<u8>>(),
    )
    .unwrap();

    let vals = [
        LaunchVal::Ptr(dq.ptr),
        LaunchVal::Ptr(dqs.ptr),
        LaunchVal::Ptr(dk.ptr),
        LaunchVal::Ptr(dks.ptr),
        LaunchVal::Ptr(dvp.ptr),
        LaunchVal::Ptr(dvs.ptr),
        LaunchVal::Ptr(dbt.ptr),
        LaunchVal::Ptr(dlens.ptr),
        LaunchVal::Ptr(dout.ptr),
        LaunchVal::I32(B as i32),
        LaunchVal::I32(HQ as i32),
        LaunchVal::I32(HK as i32),
        LaunchVal::I32(PAGE as i32),
        LaunchVal::I32(PAGES as i32),
        LaunchVal::Ptr(dout.ptr),
        LaunchVal::Ptr(dout.ptr),
    ];
    r.launch_aot(
        "owl_int8_decode_paged",
        "cu/attn/int8_decode.cu",
        (HQ as u32, B as u32, 1),
        (256, 1, 1),
        0,
        &vals,
    )
    .unwrap();
    r.sync().unwrap();
    let mut out = vec![0u8; B * HQ * D * 4];
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
    for b in 0..B {
        for h in 0..HQ {
            let kh = h / (HQ / HK);
            let mut p = vec![0f32; L];
            let mut max = f32::NEG_INFINITY;
            for key in 0..L {
                let pg = key / PAGE;
                let off = key % PAGE;
                let mut s = 0f32;
                for d in 0..D {
                    let ksc = ks[((pg * PAGE + off) * HK + kh) * 8 + d / 32];
                    s += (q[(b * HQ + h) * D + d] as f32 * qs[(b * HQ + h) * 8 + d / 32])
                        * (k[((pg * PAGE + off) * HK + kh) * D + d] as f32 * ksc);
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
                    let pg = key / PAGE;
                    let off = key % PAGE;
                    let vsv = vs[(pg * PAGE + off) * HK + kh];
                    let v = v_int[((pg * PAGE + off) * HK + kh) * D + d] as f32;
                    a += p[key] * v * vsv;
                }
                let want = a * inv;
                let g = got[(b * HQ + h) * D + d];
                dot += g as f64 * want as f64;
                na += g as f64 * g as f64;
                nb += want as f64 * want as f64;
                mx = mx.max((g - want).abs());
            }
        }
    }
    let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
    println!("[int8-decode] cos={cos:.9} max|d|={mx:.6}");
    assert!(cos > 0.9999, "cos {cos}");
    assert!(mx < 2e-2, "max|d| {mx}");

    // ---- split + reduce:同一批输入,4 片 ----
    const NPARTS: usize = 256;
    let dpout = r.alloc(B * HQ * NPARTS * D * 4, "d.pout").unwrap();
    let dpstat = r.alloc(B * HQ * NPARTS * 2 * 4, "d.pstat").unwrap();
    let dout2 = r.alloc(B * HQ * D * 2, "d.out2").unwrap();
    let svals = [
        LaunchVal::Ptr(dq.ptr),
        LaunchVal::Ptr(dqs.ptr),
        LaunchVal::Ptr(dk.ptr),
        LaunchVal::Ptr(dks.ptr),
        LaunchVal::Ptr(dvp.ptr),
        LaunchVal::Ptr(dvs.ptr),
        LaunchVal::Ptr(dbt.ptr),
        LaunchVal::Ptr(dlens.ptr),
        LaunchVal::Ptr(dpout.ptr),
        LaunchVal::Ptr(dpstat.ptr),
        LaunchVal::Ptr(dout.ptr),
        LaunchVal::I32(B as i32),
        LaunchVal::I32(HQ as i32),
        LaunchVal::I32(HK as i32),
        LaunchVal::I32(PAGE as i32),
        LaunchVal::I32(PAGES as i32),
        LaunchVal::I32(NPARTS as i32),
        LaunchVal::Ptr(dout.ptr),
        LaunchVal::Ptr(dout.ptr),
    ];
    r.launch_aot(
        "owl_int8_decode_paged_split",
        "cu/attn/int8_decode.cu",
        (HQ as u32, B as u32, NPARTS as u32),
        (256, 1, 1),
        0,
        &svals,
    )
    .unwrap();
    let rvals = [
        LaunchVal::Ptr(dpout.ptr),
        LaunchVal::Ptr(dpstat.ptr),
        LaunchVal::Ptr(dout.ptr),
        LaunchVal::Ptr(dout2.ptr),
        LaunchVal::I32(B as i32),
        LaunchVal::I32(HQ as i32),
        LaunchVal::I32(NPARTS as i32),
        LaunchVal::I32(D as i32),
    ];
    r.launch_aot(
        "owl_int8_decode_reduce",
        "cu/attn/int8_decode.cu",
        (HQ as u32, B as u32, 1),
        (256, 1, 1),
        0,
        &rvals,
    )
    .unwrap();
    r.sync().unwrap();
    let mut out2 = vec![0u8; B * HQ * D * 2];
    r.download(
        CheckBuf {
            ptr: dout2.ptr,
            bytes: out2.len(),
        },
        &mut out2,
    )
    .unwrap();
    let got2: Vec<f32> = out2
        .chunks_exact(2)
        .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    let mut mx2 = 0f32;
    for (a, b) in got2.iter().zip(&got) {
        mx2 = mx2.max((a - b).abs());
    }
    println!("[int8-decode] split/reduce max|d|={mx2:.6}");
    // reduce 输出 f16:大输出幅度下 4 ULP 可到 ~8e-3;与单块 f32 的绝对差不是核错。
    assert!(mx2 < 2e-2, "split/reduce max|d| {mx2}");

    // ---- GQA + tensor-core split:同一批输入,对照已通过的 split/reduce ----
    let dtc_out = r.alloc(B * HQ * NPARTS * D * 4, "d.tcout").unwrap();
    let dtc_stat = r.alloc(B * HQ * NPARTS * 2 * 4, "d.tcstat").unwrap();
    let dout3 = r.alloc(B * HQ * D * 2, "d.out3").unwrap();
    let tvals = [
        LaunchVal::Ptr(dq.ptr),
        LaunchVal::Ptr(dqs.ptr),
        LaunchVal::Ptr(dk.ptr),
        LaunchVal::Ptr(dks.ptr),
        LaunchVal::Ptr(dvp.ptr),
        LaunchVal::Ptr(dvs.ptr),
        LaunchVal::Ptr(dbt.ptr),
        LaunchVal::Ptr(dlens.ptr),
        LaunchVal::Ptr(dtc_out.ptr),
        LaunchVal::Ptr(dtc_stat.ptr),
        LaunchVal::Ptr(dout.ptr),
        LaunchVal::I32(B as i32),
        LaunchVal::I32(HQ as i32),
        LaunchVal::I32(HK as i32),
        LaunchVal::I32(PAGE as i32),
        LaunchVal::I32(PAGES as i32),
        LaunchVal::I32(NPARTS as i32),
        LaunchVal::Ptr(dout.ptr),
        LaunchVal::Ptr(dout.ptr),
    ];
    r.launch_aot(
        "owl_int8_decode_gqa_tc_split",
        "cu/attn/int8_decode.cu",
        (HK as u32, B as u32, NPARTS as u32),
        (256, 1, 1),
        0,
        &tvals,
    )
    .unwrap();
    let trvals = [
        LaunchVal::Ptr(dtc_out.ptr),
        LaunchVal::Ptr(dtc_stat.ptr),
        LaunchVal::Ptr(dout.ptr),
        LaunchVal::Ptr(dout3.ptr),
        LaunchVal::I32(B as i32),
        LaunchVal::I32(HQ as i32),
        LaunchVal::I32(NPARTS as i32),
        LaunchVal::I32(D as i32),
    ];
    r.launch_aot(
        "owl_int8_decode_reduce",
        "cu/attn/int8_decode.cu",
        (HQ as u32, B as u32, 1),
        (256, 1, 1),
        0,
        &trvals,
    )
    .unwrap();
    r.sync().unwrap();
    let mut out3 = vec![0u8; B * HQ * D * 2];
    r.download(
        CheckBuf {
            ptr: dout3.ptr,
            bytes: out3.len(),
        },
        &mut out3,
    )
    .unwrap();
    let got3: Vec<f32> = out3
        .chunks_exact(2)
        .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    let mut mx3 = 0f32;
    for (a, b) in got3.iter().zip(&got2) {
        mx3 = mx3.max((a - b).abs());
    }
    println!("[int8-decode] tc split max|d|={mx3:.6}");
    // P 8-bit 量化的固有误差（合成随机数据 L=4096 实测 ~5e-3）；
    // 真实模型走 needle/数学质量门，不靠这个阈值卡精度。
    assert!(mx3 < 1e-2, "tc split max|d| {mx3}");
}

/// Fused NormRope→INT8 writers (production Q/K path): host reference.
/// 这两个核是 prefill/decode 的生产写者,之前的 probe 只测了旧
/// `quant_int8_qk` / `reshape_and_cache_int8`,覆盖不到这里的 group scale。
#[test]
fn fused_qk_writers_match_host() {
    let Some(dev) = owl_cuda::test_device_ordinal_opt() else {
        owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
        return;
    };
    const HQ: usize = 2;
    const T: usize = 2;
    const HALF: usize = 64;
    const MAXP: usize = 64;
    let mut res = fi_support::TestRes::new(dev);
    let mut exec = Exec::new();
    let mut r = Runner {
        res: &mut res,
        exec: &mut exec,
        registry: None,
    };

    let xq_f = lcg_f32(T * HQ * D, 0x1111, -1.5, 3.0);
    let xq: Vec<u16> = xq_f.iter().map(|&x| f32_to_f16_rn(x)).collect();
    let xk_f = lcg_f32(T * HKV * D, 0x2222, -1.5, 3.0);
    let xk: Vec<u16> = xk_f.iter().map(|&x| f32_to_f16_rn(x)).collect();
    let wf = lcg_f32(D, 0x3333, -0.5, 1.0);
    let w: Vec<u16> = wf.iter().map(|&x| f32_to_f16_rn(x)).collect();
    let mut cos = vec![0u16; MAXP * HALF];
    let mut sin = vec![0u16; MAXP * HALF];
    for p in 0..MAXP {
        for d in 0..HALF {
            let th = (p as f32 * 0.017 + d as f32 * 0.031).sin();
            cos[p * HALF + d] = f32_to_f16_rn(th.cos());
            sin[p * HALF + d] = f32_to_f16_rn(th.sin());
        }
    }
    let pos: Vec<f32> = (0..T).map(|i| i as f32).collect();
    let eps = 1e-6f32;
    let slots = [0f32, 33.0];

    let wf32: Vec<f32> = w.iter().map(|&h| f16_to_f32(h)).collect();
    let cosf: Vec<f32> = cos.iter().map(|&h| f16_to_f32(h)).collect();
    let sinf: Vec<f32> = sin.iter().map(|&h| f16_to_f32(h)).collect();
    let rope_half = |val: &[f32], p: usize| -> Vec<f32> {
        let base = p * HALF;
        let mut out = vec![0f32; D];
        let mut sum = 0f32;
        for d in 0..D {
            sum += val[d] * val[d];
        }
        let inv = 1.0 / (sum / D as f32 + eps).sqrt();
        let n: Vec<f32> = (0..D).map(|d| val[d] * inv * (1.0 + wf32[d])).collect();
        for d in 0..HALF {
            let cf = cosf[base + d];
            let sf = sinf[base + d];
            out[d] = n[d] * cf - n[d + HALF] * sf;
            out[d + HALF] = n[d + HALF] * cf + n[d] * sf;
        }
        for d in 2 * HALF..D {
            out[d] = n[d];
        }
        for d in 0..D {
            out[d] = f16_to_f32(f32_to_f16_rn(out[d]));
        }
        out
    };

    let mut q_want = vec![0i8; T * HQ * D];
    let mut qs_want = vec![0f32; T * HQ * 8];
    for t in 0..T {
        for h in 0..HQ {
            let vals: Vec<f32> = xq[(t * HQ + h) * D..(t * HQ + h + 1) * D]
                .iter()
                .map(|&x| f16_to_f32(x))
                .collect();
            let out = rope_half(&vals, t);
            for g in 0..8 {
                let mut amax = 0f32;
                for d in g * 32..(g + 1) * 32 {
                    amax = amax.max(out[d].abs());
                }
                let s = (amax / 127.0).max(1e-6);
                qs_want[(t * HQ + h) * 8 + g] = s;
                for d in g * 32..(g + 1) * 32 {
                    q_want[(t * HQ + h) * D + d] = quant_half_away(out[d], 1.0 / s, -127.0);
                }
            }
        }
    }
    let mut k_want = vec![0x7Fi8; NB * PAGE * HKV * D];
    let mut ks_want = vec![0f32; NB * PAGE * HKV * 8];
    for t in 0..T {
        let slot = slots[t] as usize;
        let (blk, off) = (slot / PAGE, slot % PAGE);
        for h in 0..HKV {
            let vals: Vec<f32> = xk[(t * HKV + h) * D..(t * HKV + h + 1) * D]
                .iter()
                .map(|&x| f16_to_f32(x))
                .collect();
            let out = rope_half(&vals, t);
            for g in 0..8 {
                let mut amax = 0f32;
                for d in g * 32..(g + 1) * 32 {
                    amax = amax.max(out[d].abs());
                }
                let s = (amax / 127.0).max(1e-6);
                ks_want[((blk * PAGE + off) * HKV + h) * 8 + g] = s;
                for d in g * 32..(g + 1) * 32 {
                    k_want[((blk * PAGE + off) * HKV + h) * D + d] =
                        quant_half_away(out[d], 1.0 / s, -128.0);
                }
            }
        }
    }

    let dq_x = r.alloc(hb(&xq).len(), "fq.x").unwrap();
    let dk_x = r.alloc(hb(&xk).len(), "fk.x").unwrap();
    let dw = r.alloc(hb(&w).len(), "f.w").unwrap();
    let dcos = r.alloc(hb(&cos).len(), "f.cos").unwrap();
    let dsin = r.alloc(hb(&sin).len(), "f.sin").unwrap();
    let dpos = r
        .alloc(pos.len() * 4, "f.pos")
        .unwrap();
    let dqs = r.alloc(qs_want.len() * 4, "f.qs").unwrap();
    let dqo = r.alloc(q_want.len(), "f.qo").unwrap();
    let dkc = r.alloc(k_want.len(), "f.kc").unwrap();
    let dks = r.alloc(ks_want.len() * 4, "f.ks").unwrap();
    let dslots = r.alloc(T * 4, "f.slots").unwrap();
    let dout = r.alloc(2, "f.out").unwrap();
    r.upload(dq_x, &hb(&xq)).unwrap();
    r.upload(dk_x, &hb(&xk)).unwrap();
    r.upload(dw, &hb(&w)).unwrap();
    r.upload(dcos, &hb(&cos)).unwrap();
    r.upload(dsin, &hb(&sin)).unwrap();
    r.upload(
        dpos,
        &pos.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>(),
    )
    .unwrap();
    r.upload(dqs, &vec![0u8; qs_want.len() * 4]).unwrap();
    r.upload(dqo, &vec![0u8; q_want.len()]).unwrap();
    r.upload(dkc, &vec![0x7Fu8; k_want.len()]).unwrap();
    r.upload(dks, &vec![0u8; ks_want.len() * 4]).unwrap();
    r.upload(
        dslots,
        &slots.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>(),
    )
    .unwrap();

    let qv = [
        LaunchVal::Ptr(dq_x.ptr),
        LaunchVal::Ptr(dw.ptr),
        LaunchVal::Ptr(dcos.ptr),
        LaunchVal::Ptr(dsin.ptr),
        LaunchVal::Ptr(dpos.ptr),
        LaunchVal::F32(eps),
        LaunchVal::U64((HQ * D) as u64),
        LaunchVal::U64(D as u64),
        LaunchVal::U64(HALF as u64),
        LaunchVal::I32(1),
        LaunchVal::Ptr(dqs.ptr),
        LaunchVal::Ptr(dqo.ptr),
    ];
    r.launch_aot(
        "owl_norm_rope_q_int8_f16",
        "cu/attn/norm_rope.cu",
        (T as u32, HQ as u32, 1),
        (D as u32, 1, 1),
        (D * 4) as u32,
        &qv,
    )
    .unwrap();
    let kv = [
        LaunchVal::Ptr(dk_x.ptr),
        LaunchVal::Ptr(dw.ptr),
        LaunchVal::Ptr(dcos.ptr),
        LaunchVal::Ptr(dsin.ptr),
        LaunchVal::Ptr(dpos.ptr),
        LaunchVal::F32(eps),
        LaunchVal::U64((HKV * D) as u64),
        LaunchVal::U64(D as u64),
        LaunchVal::U64(HALF as u64),
        LaunchVal::I32(1),
        LaunchVal::Ptr(dslots.ptr),
        LaunchVal::Ptr(dkc.ptr),
        LaunchVal::Ptr(dks.ptr),
        LaunchVal::I32(HKV as i32),
        LaunchVal::I32(D as i32),
        LaunchVal::I32(PAGE as i32),
        LaunchVal::Ptr(dout.ptr),
    ];
    r.launch_aot(
        "owl_norm_rope_k_int8_write_f16",
        "cu/attn/norm_rope.cu",
        (T as u32, HKV as u32, 1),
        (D as u32, 1, 1),
        (D * 4) as u32,
        &kv,
    )
    .unwrap();
    r.sync().unwrap();

    let mut got_q = vec![0u8; q_want.len()];
    let mut got_qs = vec![0u8; qs_want.len() * 4];
    let mut got_k = vec![0u8; k_want.len()];
    let mut got_ks = vec![0u8; ks_want.len() * 4];
    r.download(CheckBuf { ptr: dqo.ptr, bytes: got_q.len() }, &mut got_q).unwrap();
    r.download(CheckBuf { ptr: dqs.ptr, bytes: got_qs.len() }, &mut got_qs).unwrap();
    r.download(CheckBuf { ptr: dkc.ptr, bytes: got_k.len() }, &mut got_k).unwrap();
    r.download(CheckBuf { ptr: dks.ptr, bytes: got_ks.len() }, &mut got_ks).unwrap();
    let got_q: Vec<i8> = got_q.iter().map(|&b| b as i8).collect();
    let got_k: Vec<i8> = got_k.iter().map(|&b| b as i8).collect();
    let got_qs: Vec<f32> = got_qs.chunks_exact(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();
    let got_ks: Vec<f32> = got_ks.chunks_exact(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();
    if got_q != q_want {
        let i = got_q.iter().zip(&q_want).position(|(a,b)| a != b).unwrap();
        eprintln!("[q-mismatch] i={i} got={} want={}", got_q[i], q_want[i]);
        panic!("Q fused writer wrong");
    }
    for (i,(g,w)) in got_qs.iter().zip(&qs_want).enumerate() {
        assert!((g-w).abs() <= 1e-7, "qscale[{i}] {g} != {w}");
    }
    if got_k != k_want {
        let i = got_k.iter().zip(&k_want).position(|(a,b)| a != b).unwrap();
        eprintln!("[k-mismatch] i={i} got={} want={}", got_k[i], k_want[i]);
        panic!("K fused writer wrong");
    }
    for (i,(g,w)) in got_ks.iter().zip(&ks_want).enumerate() {
        assert!((g-w).abs() <= 1e-7, "kscale[{i}] {g} != {w}");
    }
    println!("[int8-pipeline] fused Q/K writers bit-exact OK");
}

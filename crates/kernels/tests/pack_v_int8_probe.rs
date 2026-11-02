//! INT8 V prepack probe: host bit-exact scales + packed bytes.
#![cfg(feature = "device")]

mod fi_support;

use owl_kernels::bootcheck::{CheckBuf, CheckRunner};
use owl_kernels::contract::LaunchVal;
use owl_kernels::device::Exec;
use owl_kernels::server::selfcheck::Runner;

const T: usize = 5; // odd -> tail pair padding
const HKV: usize = 2;
const D: usize = 256;
const PAGE: usize = 32;

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
fn quant(x: f32, inv: f32) -> i8 {
    let mut q = x * inv;
    q = if q >= 0.0 { q + 0.5 } else { q - 0.5 };
    q = q.clamp(-127.0, 127.0);
    q as i8
}

#[test]
fn pack_v_int8_bit_exact() {
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

    let vals = lcg_f32(T * HKV * D, 0x1234, -2.0, 4.0);
    let bits: Vec<u16> = vals.iter().map(|&x| f32_to_f16(x)).collect();
    let vb: Vec<u8> = bits.iter().flat_map(|h| h.to_le_bytes()).collect();
    let p = PAGE / 2;
    let npages = (T + PAGE - 1) / PAGE;
    let mut want_scale = vec![0f32; HKV * 4];
    let mut want_vp = vec![0u8; npages * HKV * D * p * 2];
    for h in 0..HKV {
        for g in 0..4 {
            let mut amax = 0f32;
            for t in 0..T {
                for d in g * 64..(g + 1) * 64 {
                    amax = amax.max(f16_to_f32(bits[(t * HKV + h) * D + d]).abs());
                }
            }
            let s = (amax / 127.0).max(1e-6);
            want_scale[h * 4 + g] = s;
            let inv = 1.0 / s;
            for d in g * 64..(g + 1) * 64 {
                for pg in 0..npages {
                    for pair in 0..p {
                        let k0 = pg * PAGE + pair * 2;
                        let k1 = k0 + 1;
                        let a = if k0 < T {
                            quant(f16_to_f32(bits[(k0 * HKV + h) * D + d]), inv) as u8
                        } else {
                            0
                        };
                        let b = if k1 < T {
                            quant(f16_to_f32(bits[(k1 * HKV + h) * D + d]), inv) as u8
                        } else {
                            0
                        };
                        let off = (((pg * HKV + h) * D + d) * p + pair) * 2;
                        want_vp[off] = a;
                        want_vp[off + 1] = b;
                    }
                }
            }
        }
    }

    let dv = r.alloc(vb.len(), "v").unwrap();
    let dvs = r.alloc(want_scale.len() * 4, "vs").unwrap();
    let dvp = r.alloc(want_vp.len(), "vp").unwrap();
    r.upload(dv, &vb).unwrap();

    let scale_vals = [
        LaunchVal::Ptr(dv.ptr),
        LaunchVal::I32(T as i32),
        LaunchVal::I32(HKV as i32),
        LaunchVal::I32(D as i32),
        LaunchVal::Ptr(dvs.ptr),
    ];
    r.launch_aot(
        "owl_v_scale_int8",
        "cu/attn/pack_v_int8.cu",
        ((HKV * 4) as u32, 64, 1),
        (256, 1, 1),
        0,
        &scale_vals,
    )
    .unwrap();
    let fin_vals = [LaunchVal::Ptr(dvs.ptr), LaunchVal::I32((HKV * 4) as i32)];
    r.launch_aot(
        "owl_v_scale_finalize",
        "cu/attn/pack_v_int8.cu",
        (1, 1, 1),
        (256, 1, 1),
        0,
        &fin_vals,
    )
    .unwrap();
    r.sync().unwrap();

    let pack_vals = [
        LaunchVal::Ptr(dv.ptr),
        LaunchVal::I32(T as i32),
        LaunchVal::I32(HKV as i32),
        LaunchVal::I32(D as i32),
        LaunchVal::I32(PAGE as i32),
        LaunchVal::Ptr(dvs.ptr),
        LaunchVal::Ptr(dvp.ptr),
    ];
    let total = npages * HKV * D * p;
    r.launch_aot(
        "owl_pack_v_int8",
        "cu/attn/pack_v_int8.cu",
        (((total + 255) / 256) as u32, 1, 1),
        (256, 1, 1),
        0,
        &pack_vals,
    )
    .unwrap();
    r.sync().unwrap();

    let mut got_scale = vec![0u8; want_scale.len() * 4];
    let mut got_vp = vec![0u8; want_vp.len()];
    r.download(
        CheckBuf {
            ptr: dvs.ptr,
            bytes: got_scale.len(),
        },
        &mut got_scale,
    )
    .unwrap();
    r.download(
        CheckBuf {
            ptr: dvp.ptr,
            bytes: got_vp.len(),
        },
        &mut got_vp,
    )
    .unwrap();

    let got_scale: Vec<f32> = got_scale
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    for i in 0..want_scale.len() {
        assert!(
            (got_scale[i] - want_scale[i]).abs() <= 1e-7,
            "scale[{i}] got {} want {}",
            got_scale[i],
            want_scale[i]
        );
    }
    assert_eq!(got_vp, want_vp, "packed V bytes mismatch");
    println!("[pack-v] OK {} pairs, {} scales", HKV * D * p, HKV * 4);
}

/// O(ctx) cost of the two-pass V scale+pack (T=65k, Hkv=4, D=256, page=32).
#[test]
fn pack_v_int8_timing_65k() {
    let Some(dev) = owl_cuda::test_device_ordinal_opt() else {
        owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
        return;
    };
    const TT: usize = 65536;
    let mut res = fi_support::TestRes::new(dev);
    let mut exec = Exec::new();
    let mut r = Runner {
        res: &mut res,
        exec: &mut exec,
        registry: None,
    };
    let v = vec![0u8; TT * 4 * D * 2];
    let npages = TT / PAGE;
    let vp_bytes = npages * 4 * D * (PAGE / 2) * 2;
    let dv = r.alloc(v.len(), "t.v").unwrap();
    let dvs = r.alloc(4 * 4 * 4, "t.vs").unwrap();
    let dvp = r.alloc(vp_bytes, "t.vp").unwrap();
    r.upload(dv, &v).unwrap();
    let sv = [
        LaunchVal::Ptr(dv.ptr),
        LaunchVal::I32(TT as i32),
        LaunchVal::I32(4),
        LaunchVal::I32(D as i32),
        LaunchVal::Ptr(dvs.ptr),
    ];
    let pv = [
        LaunchVal::Ptr(dv.ptr),
        LaunchVal::I32(TT as i32),
        LaunchVal::I32(4),
        LaunchVal::I32(D as i32),
        LaunchVal::I32(PAGE as i32),
        LaunchVal::Ptr(dvs.ptr),
        LaunchVal::Ptr(dvp.ptr),
    ];
    let total = npages * 4 * D * (PAGE / 2);
    for _ in 0..2 {
        r.launch_aot(
            "owl_v_scale_int8",
            "cu/attn/pack_v_int8.cu",
            (16, 64, 1),
            (256, 1, 1),
            0,
            &sv,
        )
        .unwrap();
        let fin = [LaunchVal::Ptr(dvs.ptr), LaunchVal::I32(16)];
        r.launch_aot(
            "owl_v_scale_finalize",
            "cu/attn/pack_v_int8.cu",
            (1, 1, 1),
            (16, 1, 1),
            0,
            &fin,
        )
        .unwrap();
        r.launch_aot(
            "owl_pack_v_int8",
            "cu/attn/pack_v_int8.cu",
            (((total + 255) / 256) as u32, 1, 1),
            (256, 1, 1),
            0,
            &pv,
        )
        .unwrap();
    }
    r.sync().unwrap();
    let t0 = std::time::Instant::now();
    for _ in 0..3 {
        r.launch_aot(
            "owl_v_scale_int8",
            "cu/attn/pack_v_int8.cu",
            (16, 64, 1),
            (256, 1, 1),
            0,
            &sv,
        )
        .unwrap();
        let fin = [LaunchVal::Ptr(dvs.ptr), LaunchVal::I32(16)];
        r.launch_aot(
            "owl_v_scale_finalize",
            "cu/attn/pack_v_int8.cu",
            (1, 1, 1),
            (16, 1, 1),
            0,
            &fin,
        )
        .unwrap();
        r.launch_aot(
            "owl_pack_v_int8",
            "cu/attn/pack_v_int8.cu",
            (((total + 255) / 256) as u32, 1, 1),
            (256, 1, 1),
            0,
            &pv,
        )
        .unwrap();
    }
    r.sync().unwrap();
    println!("[pack-v] T=65k 3x scale+pack: {:?}", t0.elapsed());
}

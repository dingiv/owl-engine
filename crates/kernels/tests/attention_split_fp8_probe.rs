//! split fp8 直读变体证可性探针(2026-10-12 修雷配套;§二十九)。
//!
//! 对照 `chunked_prefill_fp8_probe.rs`(host f32 参考逐式对拍;三门同款)。
//! 背景:split f16 核读 e4m3 字节池 = 字节错位(2048+ 输出垃圾,潜伏雷),
//! fp8kv 变体落地后的数值门。覆盖 **partition 语义**(nparts=1/3:分区在线
//! softmax + (m,l) 中性值 + reduce 合并 == 全局 softmax,浮点差 << 门)。
//! 无 GPU 跳过(OWL_TEST_DEVICE)。

use std::sync::Arc;

use cudarc::driver::{CudaContext, DevicePtr, LaunchConfig, PushKernelArg};

const T: usize = 64;
const SEQ_LEN: usize = 256;
const PAGE: usize = 32;
const NB: usize = SEQ_LEN / PAGE;
const X: usize = 8;
const H: usize = 8;
const HKV: usize = 2;
const HD: usize = 256;

fn half_bits(v: f32) -> u16 {
    half::f16::from_f32(v).to_bits()
}

fn f16_to_f32(b: u16) -> f32 {
    half::f16::from_bits(b).to_f32()
}

// e4m3 粗编解码(位型精确;探针量级 ∈ [±448],NAN 跳过)。
// ⚠️ subnormal = m_int × 2^-9(m 为尾码整数;与 normal 的 (1+m/8) 不同式 ——
// 首版误写 (m/8)×2^-9 差 8 倍,探针三门红于此定谳:核对,host 错)
fn e4m3_decode(code: u8) -> f32 {
    let s = if code & 0x80 != 0 { -1f32 } else { 1f32 };
    let e = ((code >> 3) & 0x0F) as i32;
    let m_int = (code & 0x07) as i32;
    if e == 15 && m_int == 7 {
        return f32::NAN;
    }
    let v = if e == 0 {
        (m_int as f32) * 2f32.powi(-9)
    } else {
        (1.0 + m_int as f32 / 8.0) * 2f32.powi(e - 7)
    };
    s * v
}

fn e4m3_encode(v: f32) -> u8 {
    if v.is_nan() {
        return 0x7F;
    }
    let mut best = 0u8;
    let mut best_err = f32::INFINITY;
    for code in 0..=255u8 {
        if code & 0x7F == 0x7F {
            continue;
        }
        let d = e4m3_decode(code);
        let err = (d - v).abs();
        if err < best_err || (err == best_err && code & 1 == 0 && best & 1 == 1) {
            best_err = err;
            best = code;
        }
    }
    best
}

fn lcg(n: usize, seed: u32) -> Vec<u16> {
    (0..n)
        .map(|i| {
            let x = seed
                .wrapping_mul(0x9E37_79B9)
                .wrapping_add((i as u32).wrapping_mul(0x85EB_CA6B));
            half_bits(((x & 0xFFFF) as f32 / 65535.0 - 0.5) * 4.0)
        })
        .collect()
}

/// host 参考:全局 softmax f32 逐式(kv_fp8 = 输入池是否 e4m3 字节)。
/// classic 布局与核同源:K x=8 转置 [hd/8, page, 8],V 行主 [hd, page]。
/// 数学上 == split(nparts 分区在线 softmax + reduce 合并),浮点差 << 门。
fn host_attn_ref(q: &[u16], kcache: &[u8], vcache: &[u8], kv_fp8: bool) -> Vec<f32> {
    let dec = |pool: &[u8], i: usize| -> f32 {
        if kv_fp8 {
            e4m3_decode(pool[i])
        } else {
            f16_to_f32(pool[2 * i] as u16 | ((pool[2 * i + 1] as u16) << 8))
        }
    };
    let mut out = vec![0f32; T * H * HD];
    let kbs = HKV * HD * PAGE;
    let khs = HD * PAGE;
    let scale = 1.0 / (HD as f32).sqrt();
    let mut qk = [0f32; SEQ_LEN];
    for t in 0..T {
        for h in 0..H {
            let kvh = h / (H / HKV);
            for b in 0..=t {
                let pb = b / PAGE;
                let bb = b % PAGE;
                let mut s = 0f32;
                for d in 0..HD {
                    let gy = d / X;
                    let gx = d % X;
                    let kidx = pb * kbs + kvh * khs + bb * X + gy * (PAGE * X) + gx;
                    s += f16_to_f32(q[(t * H + h) * HD + d]) * dec(kcache, kidx);
                }
                qk[b] = s * scale;
            }
            let m = qk[..=t].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let l: f32 = qk[..=t].iter().map(|&x| (x - m).exp()).sum();
            for d in 0..HD {
                let mut acc = 0f32;
                for b in 0..=t {
                    let pb = b / PAGE;
                    let bb = b % PAGE;
                    let vidx = pb * kbs + kvh * khs + d * PAGE + bb;
                    acc += (qk[b] - m).exp() * dec(vcache, vidx);
                }
                out[(t * H + h) * HD + d] = acc / l;
            }
        }
    }
    out
}

fn run_case(
    stream: &Arc<cudarc::driver::CudaStream>,
    m: &Arc<cudarc::driver::CudaModule>,
    nparts: usize,
    np1_ref: Option<&[f32]>,
) -> Vec<f32> {
    let pool_elems = NB * HKV * HD * PAGE;
    let k_f16 = lcg(pool_elems, 401);
    let v_f16 = lcg(pool_elems, 402);
    let q_f16 = lcg(T * H * HD, 403);
    let k_fp8: Vec<u8> = k_f16.iter().map(|&h| e4m3_encode(f16_to_f32(h))).collect();
    let v_fp8: Vec<u8> = v_f16.iter().map(|&h| e4m3_encode(f16_to_f32(h))).collect();

    let dev_u16 = |v: &[u16]| {
        let mut d = unsafe { stream.alloc::<u16>(v.len()) }.unwrap();
        stream.memcpy_htod(v, &mut d).unwrap();
        d
    };
    let dev_u8 = |v: &[u8]| {
        let mut d = unsafe { stream.alloc::<u8>(v.len()) }.unwrap();
        stream.memcpy_htod(v, &mut d).unwrap();
        d
    };
    let dev_f32 = |v: &[f32]| {
        let mut d = unsafe { stream.alloc::<f32>(v.len()) }.unwrap();
        stream.memcpy_htod(v, &mut d).unwrap();
        d
    };
    let d_q = dev_u16(&q_f16);
    let d_k16 = dev_u16(&k_f16);
    let d_v16 = dev_u16(&v_f16);
    let d_k8 = dev_u8(&k_fp8);
    let d_v8 = dev_u8(&v_fp8);
    let d_bt = dev_f32(&(0..NB).map(|i| i as f32).collect::<Vec<_>>());
    let d_alibi = dev_u16(&[0u16; 16]); // 树序槽(核不解引用)
    let d_scr16 = dev_u16(&vec![0u16; T * H * nparts * HD]);
    let d_scr32 = dev_f32(&vec![0f32; T * H * nparts * 2]);
    let d_out16 = dev_u16(&vec![0u16; T * H * HD]);
    let d_out8 = dev_u16(&vec![0u16; T * H * HD]);
    let d_dummy = dev_u16(&[0u16; 16]);

    let run_pair = |tag: &str, k_name: &str, k_ptr: u64, v_ptr: u64, scr: &cudarc::driver::CudaSlice<u16>, out: &cudarc::driver::CudaSlice<u16>| {
        // ---- K1:split 主核(写 scratch)----
        let f = m
            .load_function(k_name)
            .unwrap_or_else(|e| panic!("[{tag}] load {k_name}: {e:?}"));
        // >48K smem opt-in(与 server 侧 launch.rs 同款属性)
        use cudarc::driver::sys::CUfunction_attribute_enum as Attr;
        f.set_attribute(Attr::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 65536)
            .expect("opt-in 64K smem");
        let (q_p, bt_p, ab_p, dm_p) = (
            d_q.device_ptr(stream).0,
            d_bt.device_ptr(stream).0,
            d_alibi.device_ptr(stream).0,
            d_dummy.device_ptr(stream).0,
        );
        let (s_p, o_p, k_p, v_p) = (
            scr.device_ptr(stream).0,
            out.device_ptr(stream).0,
            k_ptr,
            v_ptr,
        );
        let s32_p = d_scr32.device_ptr(stream).0;
        let (nkv, scale, np_i, t_i, ctx0) =
            (HKV as i32, 1f32 / (HD as f32).sqrt(), nparts as i32, T as i32, 0i32);
        let (kbs, khs, page, hq) =
            ((HKV * HD * PAGE) as i32, (HD * PAGE) as i32, PAGE as i32, H as i32);
        // K1 序:q,k,v,bt,scr_out,scr_stat,alibi,scale,hkv,T,ctx_base,
        //        nparts,kbs,khs,page,hq,dummy(末参 = 契约 4 哑输出)
        let mut b = stream.launch_builder(&f);
        b.arg(&q_p)
            .arg(&k_p)
            .arg(&v_p)
            .arg(&bt_p)
            .arg(&s_p)
            .arg(&s32_p)
            .arg(&ab_p)
            .arg(&scale)
            .arg(&nkv)
            .arg(&t_i)
            .arg(&ctx0)
            .arg(&np_i)
            .arg(&kbs)
            .arg(&khs)
            .arg(&page)
            .arg(&hq)
            .arg(&dm_p);
        let qchunks = (T + 63) / 64;
        let cfg = LaunchConfig {
            grid_dim: ((H / HKV) as u32, HKV as u32, (qchunks * nparts) as u32),
            block_dim: (256, 1, 1),
            shared_mem_bytes: (64 * HD * 2 * 2) as u32,
        };
        if let Err(e) = unsafe { b.launch(cfg) } {
            panic!("[{tag}] K1 launch {k_name}: {e:?}");
        }
        // ---- K2:partition 归并(scratch → out)----
        let f2 = m
            .load_function("owl_prefill_split_reduce_f16_hd256")
            .expect("load reduce");
        let mut b2 = stream.launch_builder(&f2);
        let (hq_i, hd_i) = (H as i32, HD as i32);
        b2.arg(&ab_p)
            .arg(&s_p)
            .arg(&s32_p)
            .arg(&np_i)
            .arg(&hq_i)
            .arg(&hd_i)
            .arg(&t_i)
            .arg(&o_p);
        let cfg2 = LaunchConfig {
            grid_dim: (((T * H + 255) / 256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        if let Err(e) = unsafe { b2.launch(cfg2) } {
            panic!("[{tag}] K2 launch: {e:?}");
        }
        if let Err(e) = stream.synchronize() {
            panic!("[{tag}] sync 后 ILLEGAL: {e:?}");
        }
        println!("[np{nparts} {tag}] K1+K2 launch+sync OK");
    };

    run_pair(
        "f16",
        "owl_prefill_split_f16_hd256",
        d_k16.device_ptr(stream).0,
        d_v16.device_ptr(stream).0,
        &d_scr16,
        &d_out16,
    );
    run_pair(
        "fp8",
        "owl_prefill_split_fp8kv_hd256",
        d_k8.device_ptr(stream).0,
        d_v8.device_ptr(stream).0,
        &d_scr16,
        &d_out8,
    );

    let mut o16 = vec![0u16; T * H * HD];
    stream.memcpy_dtoh(&d_out16, &mut o16).unwrap();
    let mut o8 = vec![0u16; T * H * HD];
    stream.memcpy_dtoh(&d_out8, &mut o8).unwrap();

    let k_f16_bytes: Vec<u8> = k_f16.iter().flat_map(|h| h.to_le_bytes()).collect();
    let v_f16_bytes: Vec<u8> = v_f16.iter().flat_map(|h| h.to_le_bytes()).collect();
    let ref_f16 = host_attn_ref(&q_f16, &k_f16_bytes, &v_f16_bytes, false);
    let ref_fp8 = host_attn_ref(&q_f16, &k_fp8, &v_fp8, true);

    let stat = |a: &[f32], b: &[f32]| -> (f32, f32, usize) {
        let mut mx = 0f32;
        let mut sum = 0f32;
        let mut amx = 0usize;
        for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
            let d = (x - y).abs();
            if d > mx {
                mx = d;
                amx = i;
            }
            sum += d;
        }
        (sum / a.len() as f32, mx, amx)
    };
    let to_f32 = |o: &[u16]| -> Vec<f32> { o.iter().map(|&h| f16_to_f32(h)).collect() };

    // ① gpu_f16 vs host_f16 = 布局/读序 + partition 语义自检(np1 硬门;np>1 诊断)
    let (m16, x16, a16) = stat(&to_f32(&o16), &ref_f16);
    println!("① np{nparts} gpu_f16 vs host_f16 — mean={m16:.6} max={x16:.6} @ {a16}");
    assert!(x16 < 2e-2, "np{nparts} f16 布局/分区自检超门(max {x16})");

    // ② gpu_fp8 vs host_fp8 = 修雷硬门;附误差分布诊断(孤点 vs 系统性)
    let (m8, x8, a8) = stat(&to_f32(&o8), &ref_fp8);
    let (t8, h8, d8) = (a8 / (H * HD), (a8 / HD) % H, a8 % HD);
    println!("② np{nparts} gpu_fp8 vs host_fp8 — mean={m8:.6} max={x8:.6} @ t={t8} h={h8} d={d8}");
    let of = to_f32(&o8);
    let mut hot: Vec<(usize, f32, f32, f32)> = of
        .iter()
        .zip(&ref_fp8)
        .enumerate()
        .map(|(i, (&g, &r))| (i, g, r, (g - r).abs()))
        .filter(|&(_, _, _, e)| e > 4e-3)
        .collect();
    hot.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());
    for &(i, g, r, e) in hot.iter().take(6) {
        let (tt, hh, dd) = (i / (H * HD), (i / HD) % H, i % HD);
        println!("   hot t={tt} h={hh} d={dd}: gpu={g:.6} host={r:.6} err={e:.6}");
    }
    // 诊断:t 分布 + d 直方 + hot 点的 V 字节/f16 原值(锁定错位模式)
    let ts: std::collections::BTreeSet<usize> = hot.iter().map(|&(i, ..)| i / (H * HD)).collect();
    let mut dh: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
    for &(i, ..) in &hot {
        *dh.entry(i % HD).or_insert(0) += 1;
    }
    println!("   ts={:?} d_hist={:?}", &ts, &dh);
    for &(i, g, r, e) in hot.iter().take(4) {
        let (tt, hh, dd) = (i / (H * HD), (i / HD) % H, i % HD);
        let kvh = hh / (H / HKV);
        let vidx = kvh * (HD * PAGE) + dd * PAGE; // phys=0,off=0(gt=0)
        println!(
            "   byte d={dd}: v8[{}]={:#04x} v16_f32={:.6} gpu={g:.6} host={r:.6} err={e:.6}",
            vidx,
            v_fp8[vidx],
            f16_to_f32(v_f16[vidx])
        );
    }
    println!("   hot_count(>4e-3) = {}", hot.len());
    assert!(m8 < 2e-3 && x8 < 1e-2, "np{nparts} split fp8 对 host 参考超门(mean {m8}, max {x8})");

    // ③ f16 vs fp8 = 量化噪声 tripwire
    let (mn, xn, _) = stat(&to_f32(&o16), &to_f32(&o8));
    println!("③ np{nparts} f16 vs fp8 量化噪声 — mean={mn:.5} max={xn:.5}");
    assert!(mn < 5e-2 && xn < 3e-1, "np{nparts} f16/fp8 tripwire(mean {mn}, max {xn})");

    // ④ np>1 vs np1 同核对拍(分区合并数学等价全局;隔离「核分区 bug vs host 错」)
    if let Some(r1) = np1_ref {
        let (m4, x4, a4) = stat(&to_f32(&o8), r1);
        println!("④ np{nparts} gpu_fp8 vs np1 同核 — mean={m4:.6} max={x4:.6} @ {a4}");
        assert!(x4 < 2e-2, "np{nparts} 分区合并对 np1 超门(max {x4})— 核分区路径疑");
        // ⑤ scr_stat 逐分区对拍:定位 m/l 哪一环错(仅 np>1)
        let mut gs = vec![0f32; T * H * nparts * 2];
        stream.memcpy_dtoh(&d_scr32, &mut gs).unwrap();
        let scale = 1.0 / (HD as f32).sqrt();
        let mut bad = 0usize;
        for t in 0..T {
            for h in 0..H {
                let kvh = h / (H / HKV);
                let ctx_end = t + 1;
                let max_ctx = T;
                let p_len = (max_ctx + nparts - 1) / nparts;
                for p in 0..nparts {
                    let ps = p * p_len;
                    let pe = std::cmp::min((p + 1) * p_len, max_ctx);
                    let (mut m, mut l) = (f32::NEG_INFINITY, 0f32);
                    if ps < ctx_end {
                        for b in ps..pe.min(ctx_end) {
                            let pb = b / PAGE;
                            let bb = b % PAGE;
                            let mut s = 0f32;
                            for d in 0..HD {
                                let kidx = pb * (HKV * HD * PAGE) + kvh * (HD * PAGE)
                                    + bb * X + (d / X) * (PAGE * X) + d % X;
                                s += f16_to_f32(q_f16[(t * H + h) * HD + d]) * e4m3_decode(k_fp8[kidx]);
                            }
                            s *= scale;
                            if s > m {
                                l = l * (m - s).exp() + 1.0;
                                m = s;
                            } else {
                                l += (s - m).exp();
                            }
                        }
                    }
                    let bidx = ((t * H + h) * nparts + p) * 2;
                    let (gm, gl) = (gs[bidx], gs[bidx + 1]);
                    let dm = (gm - m).abs();
                    let dl = (gl - l).abs();
                    if dm > 1e-3 || dl > 1e-2 * l.max(1.0) {
                        if bad < 5 {
                            println!("   ⑤ t={t} h={h} p={p}: gpu m={gm:.4} l={gl:.4} | host m={m:.4} l={l:.4}");
                            // 反查:gpu m 对应哪个 K[x]?(锁定错位源)
                            for x in 0..max_ctx {
                                let pb = x / PAGE;
                                let bb = x % PAGE;
                                let mut s = 0f32;
                                for d in 0..HD {
                                    let kidx = pb * (HKV * HD * PAGE) + kvh * (HD * PAGE)
                                        + bb * X + (d / X) * (PAGE * X) + d % X;
                                    s += f16_to_f32(q_f16[(t * H + h) * HD + d]) * e4m3_decode(k_fp8[kidx]);
                                }
                                s *= scale;
                                if (s - gm).abs() < 5e-4 {
                                    println!("      ↳ gpu m ≈ q·K[{x}]·scale");
                                }
                            }
                        }
                        bad += 1;
                    }
                }
            }
        }
        println!("   ⑤ stat mismatch(count, t<8) = {bad}");
    }
    of
}

#[test]
fn split_fp8_matches_f16() {
    if !owl_shared::env_reader::flag("OWL_TEST_DEVICE") {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    }
    let ctx = CudaContext::new(1).expect("ctx");
    ctx.bind_to_thread().expect("bind");
    let stream = ctx.default_stream();
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(
        owl_kernels::sources::attention::PREFILL_SPLIT_F16,
        cudarc::nvrtc::CompileOptions {
            arch: Some("compute_86"),
            include_paths: vec!["/usr/local/cuda/include".into()],
            ..Default::default()
        },
    )
    .expect("nvrtc compile prefill_split");
    let m = ctx.load_module(ptx).expect("module");

    // nparts=1(基本路径)+ nparts=3(分区在线 softmax + 中性值 + 归并);
    // np>1 的 ① 降为诊断打印(host 全局 softmax 参考 vs 分区合并浮点不等价),
    // 判定走 ④ 同核对拍
    let mut np1_out: Option<Vec<f32>> = None;
    for np in [1usize, 2usize, 3usize] {
        let of = run_case(&stream, &m, np, np1_out.as_deref());
        if np == 1 {
            np1_out = Some(of);
        }
    }
    println!("split fp8 探针 PASS(np1/np2/np3 三形状):fp8kv 直读贴 host 参考,分区合并 == 全局 softmax,f16 差 = 量化噪声");
}

//! kv布局统一契约 P2 读核 parity 探针(decode v2 + prefill split)。
//!
//! 门 = **bitwise**:kNHD 变体保持逻辑 d→线程映射不变,仅字节排布不同 →
//! 同一逻辑 KV 内容(两池分别按各自布局写入同一批值)必须产出逐位一致输出。
//! (设计文档 §六.2 预留了容差门;实现保持算术序不变,故直接收位等门。)
//!
//! 覆盖:vllm_paged_attention_v2(f16/fp8,hd256bs32,decode+seq 形状)、
//! owl_prefill_split(f16/fp8kv)。映射 oracle 独立重实现(与 env.rs 互证)。
//! 无 GPU 跳过(OWL_TEST_DEVICE;卡 = ordinal 1)。

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream, DevicePtr, LaunchConfig, PushKernelArg};

const HQ: usize = 8;
const HKV: usize = 2;
const HD: usize = 256;
const PAGE: usize = 32;
const NB: usize = 6; // 池块数
const CTX: usize = 96; // 上下文 token 数(3 块)
fn scale_f32() -> f32 { 1.0 / ((HD as f32).sqrt()) }

fn half_bits(f: f32) -> u16 {
    half::f16::from_f32(f).to_bits()
}

// ---- 独立映射 oracle(经典/统一;V 复用 K 的 knhd 公式)----
fn classic_k_off(slot: usize, h: usize, d: usize) -> usize {
    let (b, p) = (slot / PAGE, slot % PAGE);
    (((b * HKV + h) * (HD / 8) + d / 8) * PAGE + p) * 8 + d % 8
}
fn classic_v_off(slot: usize, h: usize, d: usize) -> usize {
    let (b, p) = (slot / PAGE, slot % PAGE);
    ((b * HKV + h) * HD + d) * PAGE + p
}
fn knhd_off(slot: usize, h: usize, d: usize) -> usize {
    let (b, p) = (slot / PAGE, slot % PAGE);
    ((b * PAGE + p) * HKV + h) * HD + d
}

fn compile(ctx: &Arc<CudaContext>, src: &str, name: &str) -> cudarc::driver::CudaFunction {
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(
        src,
        cudarc::nvrtc::CompileOptions {
            arch: Some("compute_86"),
            include_paths: vec!["/usr/local/cuda/include".into()],
            ..Default::default()
        },
    )
    .unwrap_or_else(|e| panic!("nvrtc({name}): {e:?}"));
    let m = ctx.load_module(ptx).unwrap_or_else(|e| panic!("module({name}): {e:?}"));
    m.load_function(name).unwrap_or_else(|e| panic!("fn({name}): {e:?}"))
}

/// 双池生成:同一逻辑值序列,按两布局落位。gen(slot,h,d) = 值位型。
/// kind = 池语义:K 池 classic 侧走 K 布局,V 池 classic 侧走 **V 布局**
/// (2026-10-12 探针案:V 池误用 K 布局 = classic 核读乱序 V 的假象根因;
/// knhd 侧 K/V 同构,一律 knhd_off)。
#[derive(Clone, Copy, PartialEq)]
enum PoolKind {
    KClassic,
    VClassic,
}
fn fill_pools_u16(kind: PoolKind, gen: impl Fn(usize, usize, usize) -> u16) -> (Vec<u16>, Vec<u16>) {
    let n = NB * PAGE * HKV * HD;
    let mut c = vec![0u16; n];
    let mut k = vec![0u16; n];
    for slot in 0..CTX {
        for h in 0..HKV {
            for d in 0..HD {
                let v = gen(slot, h, d);
                match kind {
                    PoolKind::KClassic => c[classic_k_off(slot, h, d)] = v,
                    PoolKind::VClassic => c[classic_v_off(slot, h, d)] = v,
                }
                k[knhd_off(slot, h, d)] = v;
            }
        }
    }
    (c, k)
}

fn fill_pools_u8(kind: PoolKind, gen: impl Fn(usize, usize, usize) -> u8) -> (Vec<u8>, Vec<u8>) {
    let n = NB * PAGE * HKV * HD;
    let mut c = vec![0u8; n];
    let mut k = vec![0u8; n];
    for slot in 0..CTX {
        for h in 0..HKV {
            for d in 0..HD {
                let v = gen(slot, h, d);
                match kind {
                    PoolKind::KClassic => c[classic_k_off(slot, h, d)] = v,
                    PoolKind::VClassic => c[classic_v_off(slot, h, d)] = v,
                }
                k[knhd_off(slot, h, d)] = v;
            }
        }
    }
    (c, k)
}

#[test]
fn paged_decode_v2_knhd_parity() {
    if !owl_shared::env_reader::flag("OWL_TEST_DEVICE") {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    }
    let ctx = CudaContext::new(1).expect("ctx");
    ctx.bind_to_thread().expect("bind");
    let stream: Arc<CudaStream> = ctx.default_stream();

    let q: Vec<u16> = (0..HQ * HD).map(|i| half_bits(((i * 41) % 173) as f32 / 13.0 - 6.0)).collect();
    let block_tables = [0f32, 1.0, 2.0, 3.0]; // 单序列恒等页表
    let context_lens = [CTX as f32];
    let nparts = 1usize;
    let tmp_elems = HQ * nparts * HD;

    // f16 池:值域 ±4(避 NaN/Inf)
    let (kc_c, kc_n) = fill_pools_u16(PoolKind::KClassic, |s, h, d| {
        half_bits((((s * 37 + h * 11 + d) % 173) as f32) / 20.0 - 4.0)
    });
    let (vc_c, vc_n) = fill_pools_u16(PoolKind::VClassic, |s, h, d| {
        half_bits((((s * 53 + h * 7 + d) % 191) as f32) / 22.0 - 4.0)
    });

    // ---- 设备池字节对账(上载后回读,排除装配错)----
    {
        let mut d_kc = unsafe { stream.alloc::<u16>(kc_c.len()) }.unwrap();
        stream.memcpy_htod(&kc_c, &mut d_kc).unwrap();
        let mut back = vec![0u16; kc_c.len()];
        stream.memcpy_dtoh(&d_kc, &mut back).unwrap();
        let mut mism = 0usize;
        for s in 0..32 {
            for h in 0..HKV {
                for d in 0..HD {
                    let want = half_bits((((s * 37 + h * 11 + d) % 173) as f32) / 20.0 - 4.0);
                    if back[classic_k_off(s, h, d)] != want {
                        mism += 1;
                    }
                }
            }
        }
        eprintln!("[pool] classic 池回读错位 {mism}");
        let mut d_kn = unsafe { stream.alloc::<u16>(kc_n.len()) }.unwrap();
        stream.memcpy_htod(&kc_n, &mut d_kn).unwrap();
        stream.memcpy_dtoh(&d_kn, &mut back).unwrap();
        let mut mism2 = 0usize;
        for s in 0..32 {
            for h in 0..HKV {
                for d in 0..HD {
                    let want = half_bits((((s * 37 + h * 11 + d) % 173) as f32) / 20.0 - 4.0);
                    if back[knhd_off(s, h, d)] != want {
                        mism2 += 1;
                    }
                }
            }
        }
        eprintln!("[pool] knhd 池回读错位 {mism2}");
    }

    let run = |name: &str, kcp: &[u16], vcp: &[u16], kstride: i32| -> Vec<u16> {
        let f = compile(&ctx, owl_kernels::sources::attention::PAGED_ATTENTION_F16, name);
        let mut d_q = unsafe { stream.alloc::<u16>(q.len()) }.unwrap();
        stream.memcpy_htod(&q, &mut d_q).unwrap();
        let mut d_kc = unsafe { stream.alloc::<u16>(kcp.len()) }.unwrap();
        stream.memcpy_htod(kcp, &mut d_kc).unwrap();
        let mut d_vc = unsafe { stream.alloc::<u16>(vcp.len()) }.unwrap();
        stream.memcpy_htod(vcp, &mut d_vc).unwrap();
        let mut d_bt = unsafe { stream.alloc::<f32>(block_tables.len()) }.unwrap();
        stream.memcpy_htod(&block_tables, &mut d_bt).unwrap();
        let mut d_cl = unsafe { stream.alloc::<f32>(context_lens.len()) }.unwrap();
        stream.memcpy_htod(&context_lens, &mut d_cl).unwrap();
        let mut d_es = unsafe { stream.alloc::<f32>(HQ * nparts) }.unwrap();
        let mut d_ml = unsafe { stream.alloc::<f32>(HQ * nparts) }.unwrap();
        let mut d_to = unsafe { stream.alloc::<u16>(tmp_elems) }.unwrap();
        stream.memset_zeros(&mut d_to).unwrap();
        let mut d_alibi = unsafe { stream.alloc::<f32>(1) }.unwrap();
        let (qp, kcp2, vcp2, btp, clp, alp, esp, mlp, top) = (
            d_q.device_ptr(&stream).0,
            d_kc.device_ptr(&stream).0,
            d_vc.device_ptr(&stream).0,
            d_bt.device_ptr(&stream).0,
            d_cl.device_ptr(&stream).0,
            d_alibi.device_ptr(&stream).0,
            d_es.device_ptr(&stream).0,
            d_ml.device_ptr(&stream).0,
            d_to.device_ptr(&stream).0,
        );
        let nkv = HKV as i32;
        let mnb = 4i32;
        let qstride = (HQ * HD) as i32;
        let bstride = (HKV * HD * PAGE) as i32;
        let scale = scale_f32();
        let sscap = 1.0f32;
        let swin = -1i32;
        let ualibi = 0i32;
        let mut b = stream.launch_builder(&f);
        b.arg(&qp)
            .arg(&kcp2)
            .arg(&vcp2)
            .arg(&btp)
            .arg(&clp)
            .arg(&alp)
            .arg(&esp)
            .arg(&mlp)
            .arg(&nkv)
            .arg(&scale)
            .arg(&mnb)
            .arg(&qstride)
            .arg(&bstride)
            .arg(&kstride)
            .arg(&sscap)
            .arg(&swin)
            .arg(&ualibi)
            .arg(&top);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (HQ as u32, 1, nparts as u32),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 2048u32.max((128 / 32 / 2) * HD as u32 * 4),
            })
        }
        .unwrap();
        stream.synchronize().unwrap();
        let mut out = vec![0u16; tmp_elems];
        stream.memcpy_dtoh(&d_to, &mut out).unwrap();
        out
    };

    let out_c = run(
        "vllm_paged_attention_v2_f16_hd256bs32",
        &kc_c,
        &vc_c,
        (HD * PAGE) as i32, // classic head stride
    );
    let out_n = run(
        "vllm_paged_attention_v2_f16_knhd_hd256bs32",
        &kc_n,
        &vc_n,
        HD as i32, // kNHD head stride
    );
    // ---- host 参考(f32 朴素注意力;定位 classic/knhd 谁在说谎)----
    let f32_of = half::f16::from_bits;
    let mut host_ref = vec![0f32; HQ * HD];
    for h in 0..HQ {
        let hk = h * HKV / HQ; // GQA 组映射(h/Hq_per_kv)
        let mut logits = vec![0f32; CTX];
        for s in 0..CTX {
            let mut dot = 0f32;
            for d in 0..HD {
                let qv = f32_of(q[h * HD + d]).to_f32();
                let kv = f32_of(kc_c[classic_k_off(s, hk, d)]).to_f32();
                dot += qv * kv;
            }
            logits[s] = dot * scale_f32();
        }
        if h == 0 { eprintln!("[hlog] rust logits[0..4] = {:?}", &logits[0..4]); }
        let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0f32;
        let w: Vec<f32> = logits.iter().map(|&l| { let e = (l - m).exp(); sum += e; e }).collect();
        for d in 0..HD {
            let mut acc = 0f32;
            for s in 0..CTX {
                acc += w[s] / sum * f32_of(vc_c[classic_v_off(s, hk, d)]).to_f32();
            }
            host_ref[h * HD + d] = acc;
        }
    }
    let near = |a: &u16, b: &f32| (half::f16::from_bits(*a).to_f32() - *b).abs() < 5e-2;
    let c_ok = out_c.iter().zip(host_ref.iter()).filter(|(a, b)| !near(*a, *b)).count();
    let n_ok = out_n.iter().zip(host_ref.iter()).filter(|(a, b)| !near(*a, *b)).count();
    eprintln!("[ref] host 参考比对:classic 离谱 {c_ok}/{} knhd 离谱 {n_ok}(容差 5e-2)", out_c.len());
    eprintln!("[ref] host_ref[0..4] = {:?}", &host_ref[..4]);

    // 竞态判别:classic 连跑两次
    let out_c2 = run(
        "vllm_paged_attention_v2_f16_hd256bs32",
        &kc_c,
        &vc_c,
        (HD * PAGE) as i32,
    );
    eprintln!("[race] classic run1 == run2? {}", out_c == out_c2);
    let out_n2 = run(
        "vllm_paged_attention_v2_f16_knhd_hd256bs32",
        &kc_n,
        &vc_n,
        HD as i32,
    );
    eprintln!("[race] knhd run1 == run2? {}", out_n == out_n2);

    // 判别实验:knhd 入口 × classic 池 × classic stride —— 若与 classic 输出
    // 位等,则模板参未生效(执行了 classic 寻址)
    let out_diag = run(
        "vllm_paged_attention_v2_f16_knhd_hd256bs32",
        &kc_c,
        &vc_c,
        (HD * PAGE) as i32,
    );
    let diag_same = out_diag == out_c;
    eprintln!("[diag] knhd入口×classic池×classic.stride == classic输出? {diag_same}");
    let diff_n = out_diag.iter().zip(out_n.iter()).filter(|(a, b)| a != b).count();
    eprintln!("[diag] knhd入口 classic池 vs knhd池 差异数 {diff_n}/{}", out_c.len());
    let mut c_nonzero = 0usize;
    for (i, (&a, &b)) in out_c.iter().zip(out_n.iter()).enumerate() {
        if i < 4 {
            eprintln!("[diag] out_c[{i}] = {:?} (0x{:04x})  out_n[{i}] = {:?} (0x{:04x})",
                half::f16::from_bits(a), a, half::f16::from_bits(b), b);
        }
        if a != 0 { c_nonzero += 1; }
    }
    let n_nz = out_n.iter().filter(|&&v| v != 0).count();
    eprintln!("[diag] out_c 非零 {c_nonzero}/{}  out_n 非零 {n_nz}", out_c.len());
    for head in 0..2 {
        let bits: Vec<u32> = (0..6).map(|d| out_n[head * HD + d] as u32).collect();
        eprintln!("[diag] out_n head{head} d0..6 bits = {bits:08x?}");
    }
    // 头间是否重复(只算了 head 0 并广播?)
    eprintln!("[diag] head0==head1? {}", out_n[0..HD] == out_n[HD..2 * HD]);

    let bad = out_c.iter().zip(out_n.iter()).filter(|(a, b)| a != b).count();
    assert_eq!(bad, 0, "v2 f16 decode parity:{bad} 处位型不符");
    assert!(out_c.iter().any(|&v| v != 0), "v2 f16 全零输出(装配错)");

    // ---- fp8 变体(e4m3 字节池;避 0x7F/0xFF NaN)----
    let (kc_c8, kc_n8) = fill_pools_u8(PoolKind::KClassic, |s, h, d| (((s * 29 + h * 13 + d) % 120) + 1) as u8);
    let (vc_c8, vc_n8) = fill_pools_u8(PoolKind::VClassic, |s, h, d| (((s * 31 + h * 17 + d) % 120) + 1) as u8);
    let run8 = |name: &str, kcp: &[u8], vcp: &[u8], kstride: i32| -> Vec<u16> {
        let f = compile(&ctx, owl_kernels::sources::attention::PAGED_ATTENTION_F16, name);
        let mut d_q = unsafe { stream.alloc::<u16>(q.len()) }.unwrap();
        stream.memcpy_htod(&q, &mut d_q).unwrap();
        let mut d_kc = unsafe { stream.alloc::<u8>(kcp.len()) }.unwrap();
        stream.memcpy_htod(kcp, &mut d_kc).unwrap();
        let mut d_vc = unsafe { stream.alloc::<u8>(vcp.len()) }.unwrap();
        stream.memcpy_htod(vcp, &mut d_vc).unwrap();
        let mut d_bt = unsafe { stream.alloc::<f32>(block_tables.len()) }.unwrap();
        stream.memcpy_htod(&block_tables, &mut d_bt).unwrap();
        let mut d_cl = unsafe { stream.alloc::<f32>(context_lens.len()) }.unwrap();
        stream.memcpy_htod(&context_lens, &mut d_cl).unwrap();
        let mut d_es = unsafe { stream.alloc::<f32>(HQ * nparts) }.unwrap();
        let mut d_ml = unsafe { stream.alloc::<f32>(HQ * nparts) }.unwrap();
        let mut d_to = unsafe { stream.alloc::<u16>(tmp_elems) }.unwrap();
        stream.memset_zeros(&mut d_to).unwrap();
        let mut d_alibi = unsafe { stream.alloc::<f32>(1) }.unwrap();
        let (qp, kcp2, vcp2, btp, clp, alp, esp, mlp, top) = (
            d_q.device_ptr(&stream).0,
            d_kc.device_ptr(&stream).0,
            d_vc.device_ptr(&stream).0,
            d_bt.device_ptr(&stream).0,
            d_cl.device_ptr(&stream).0,
            d_alibi.device_ptr(&stream).0,
            d_es.device_ptr(&stream).0,
            d_ml.device_ptr(&stream).0,
            d_to.device_ptr(&stream).0,
        );
        let nkv = HKV as i32;
        let mnb = 4i32;
        let qstride = (HQ * HD) as i32;
        let bstride = (HKV * HD * PAGE) as i32;
        let scale = scale_f32();
        let sscap = 1.0f32;
        let swin = -1i32;
        let ualibi = 0i32;
        let mut b = stream.launch_builder(&f);
        b.arg(&qp)
            .arg(&kcp2)
            .arg(&vcp2)
            .arg(&btp)
            .arg(&clp)
            .arg(&alp)
            .arg(&esp)
            .arg(&mlp)
            .arg(&nkv)
            .arg(&scale)
            .arg(&mnb)
            .arg(&qstride)
            .arg(&bstride)
            .arg(&kstride)
            .arg(&sscap)
            .arg(&swin)
            .arg(&ualibi)
            .arg(&top);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (HQ as u32, 1, nparts as u32),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 2048u32.max((128 / 32 / 2) * HD as u32 * 4),
            })
        }
        .unwrap();
        stream.synchronize().unwrap();
        let mut out = vec![0u16; tmp_elems];
        stream.memcpy_dtoh(&d_to, &mut out).unwrap();
        out
    };
    let out_c8 = run8(
        "vllm_paged_attention_v2_fp8_hd256bs32",
        &kc_c8,
        &vc_c8,
        (HD * PAGE) as i32,
    );
    let out_n8 = run8(
        "vllm_paged_attention_v2_fp8_knhd_hd256bs32",
        &kc_n8,
        &vc_n8,
        HD as i32,
    );
    let bad8 = out_c8.iter().zip(out_n8.iter()).filter(|(a, b)| a != b).count();
    assert_eq!(bad8, 0, "v2 fp8 decode parity:{bad8} 处位型不符");
    assert!(out_c8.iter().any(|&v| v != 0), "v2 fp8 全零输出(装配错)");
}

#[test]
fn prefill_split_knhd_parity() {
    if !owl_shared::env_reader::flag("OWL_TEST_DEVICE") {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    }
    let ctx = CudaContext::new(1).expect("ctx");
    ctx.bind_to_thread().expect("bind");
    let stream: Arc<CudaStream> = ctx.default_stream();

    let t_tokens = 8usize;
    let nparts = 1usize;
    let q: Vec<u16> = (0..t_tokens * HQ * HD)
        .map(|i| half_bits(((i * 43) % 181) as f32 / 13.0 - 6.0))
        .collect();
    let block_tables = [0f32, 1.0, 2.0, 3.0];
    let scr_out_elems = t_tokens * HQ * nparts * HD;
    let scr_stat_elems = t_tokens * HQ * nparts * 2;

    let (kc_c, kc_n) = fill_pools_u16(PoolKind::KClassic, |s, h, d| {
        half_bits((((s * 37 + h * 11 + d) % 173) as f32) / 20.0 - 4.0)
    });
    let (vc_c, vc_n) = fill_pools_u16(PoolKind::VClassic, |s, h, d| {
        half_bits((((s * 53 + h * 7 + d) % 191) as f32) / 22.0 - 4.0)
    });

    let run = |name: &str, kcp: &[u16], vcp: &[u16], kstride: i32| -> (Vec<u16>, Vec<f32>) {
        let f = compile(&ctx, owl_kernels::sources::attention::PREFILL_SPLIT_F16, name);
        // >48K smem opt-in(与 server 侧 launch.rs / 既有 split 探针同款)
        f.set_attribute(
            cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            65536,
        )
        .expect("opt-in 64K smem");
        let mut d_q = unsafe { stream.alloc::<u16>(q.len()) }.unwrap();
        stream.memcpy_htod(&q, &mut d_q).unwrap();
        let mut d_kc = unsafe { stream.alloc::<u16>(kcp.len()) }.unwrap();
        stream.memcpy_htod(kcp, &mut d_kc).unwrap();
        let mut d_vc = unsafe { stream.alloc::<u16>(vcp.len()) }.unwrap();
        stream.memcpy_htod(vcp, &mut d_vc).unwrap();
        let mut d_bt = unsafe { stream.alloc::<f32>(block_tables.len()) }.unwrap();
        stream.memcpy_htod(&block_tables, &mut d_bt).unwrap();
        let mut d_so = unsafe { stream.alloc::<u16>(scr_out_elems) }.unwrap();
        let mut d_ss = unsafe { stream.alloc::<f32>(scr_stat_elems) }.unwrap();
        let mut d_alibi = unsafe { stream.alloc::<u16>(1) }.unwrap();
        let mut d_dummy = unsafe { stream.alloc::<u16>(1) }.unwrap();
        let (qp, kcp2, vcp2, btp, sop, ssp, alp, dmp) = (
            d_q.device_ptr(&stream).0,
            d_kc.device_ptr(&stream).0,
            d_vc.device_ptr(&stream).0,
            d_bt.device_ptr(&stream).0,
            d_so.device_ptr(&stream).0,
            d_ss.device_ptr(&stream).0,
            d_alibi.device_ptr(&stream).0,
            d_dummy.device_ptr(&stream).0,
        );
        let scale = scale_f32();
        let (nkv, tt, ctxb, np_) = (HKV as i32, t_tokens as i32, 0i32, nparts as i32);
        let bstride = (HKV * HD * PAGE) as i32;
        let hqi = HQ as i32;
        let pg = PAGE as i32;
        let mut b = stream.launch_builder(&f);
        b.arg(&qp)
            .arg(&kcp2)
            .arg(&vcp2)
            .arg(&btp)
            .arg(&sop)
            .arg(&ssp)
            .arg(&alp)
            .arg(&scale)
            .arg(&nkv)
            .arg(&tt)
            .arg(&ctxb)
            .arg(&np_)
            .arg(&bstride)
            .arg(&kstride)
            .arg(&pg)
            .arg(&hqi)
            .arg(&dmp);
        let qchunks = (t_tokens + 63) / 64;
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: ((HQ / HKV) as u32, HKV as u32, (qchunks * nparts) as u32),
                block_dim: (256, 1, 1),
                shared_mem_bytes: (64 * HD * 2 * 2) as u32,
            })
        }
        .unwrap();
        stream.synchronize().unwrap();
        let mut so = vec![0u16; scr_out_elems];
        let mut ss = vec![0f32; scr_stat_elems];
        stream.memcpy_dtoh(&d_so, &mut so).unwrap();
        stream.memcpy_dtoh(&d_ss, &mut ss).unwrap();
        (so, ss)
    };

    let (so_c, ss_c) = run(
        "owl_prefill_split_f16_hd256",
        &kc_c,
        &vc_c,
        (HD * PAGE) as i32,
    );
    let (so_n, ss_n) = run(
        "owl_prefill_split_f16_knhd_hd256",
        &kc_n,
        &vc_n,
        HD as i32,
    );
    let bad = so_c.iter().zip(so_n.iter()).filter(|(a, b)| a != b).count();
    assert_eq!(bad, 0, "split f16 scr_out parity:{bad} 处位型不符");
    assert_eq!(ss_c, ss_n, "split f16 scr_stat(m,l)不符");
    assert!(so_c.iter().any(|&v| v != 0), "split f16 全零输出(装配错)");

    // ---- fp8kv 变体 ----
    let (kc_c8, kc_n8) = fill_pools_u8(PoolKind::KClassic, |s, h, d| (((s * 29 + h * 13 + d) % 120) + 1) as u8);
    let (vc_c8, vc_n8) = fill_pools_u8(PoolKind::VClassic, |s, h, d| (((s * 31 + h * 17 + d) % 120) + 1) as u8);
    let run8 = |name: &str, kcp: &[u8], vcp: &[u8], kstride: i32| -> (Vec<u16>, Vec<f32>) {
        let f = compile(&ctx, owl_kernels::sources::attention::PREFILL_SPLIT_F16, name);
        // >48K smem opt-in(与 server 侧 launch.rs / 既有 split 探针同款)
        f.set_attribute(
            cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            65536,
        )
        .expect("opt-in 64K smem");
        let mut d_q = unsafe { stream.alloc::<u16>(q.len()) }.unwrap();
        stream.memcpy_htod(&q, &mut d_q).unwrap();
        let mut d_kc = unsafe { stream.alloc::<u8>(kcp.len()) }.unwrap();
        stream.memcpy_htod(kcp, &mut d_kc).unwrap();
        let mut d_vc = unsafe { stream.alloc::<u8>(vcp.len()) }.unwrap();
        stream.memcpy_htod(vcp, &mut d_vc).unwrap();
        let mut d_bt = unsafe { stream.alloc::<f32>(block_tables.len()) }.unwrap();
        stream.memcpy_htod(&block_tables, &mut d_bt).unwrap();
        let mut d_so = unsafe { stream.alloc::<u16>(scr_out_elems) }.unwrap();
        let mut d_ss = unsafe { stream.alloc::<f32>(scr_stat_elems) }.unwrap();
        let mut d_alibi = unsafe { stream.alloc::<u16>(1) }.unwrap();
        let mut d_dummy = unsafe { stream.alloc::<u16>(1) }.unwrap();
        let (qp, kcp2, vcp2, btp, sop, ssp, alp, dmp) = (
            d_q.device_ptr(&stream).0,
            d_kc.device_ptr(&stream).0,
            d_vc.device_ptr(&stream).0,
            d_bt.device_ptr(&stream).0,
            d_so.device_ptr(&stream).0,
            d_ss.device_ptr(&stream).0,
            d_alibi.device_ptr(&stream).0,
            d_dummy.device_ptr(&stream).0,
        );
        let scale = scale_f32();
        let (nkv, tt, ctxb, np_) = (HKV as i32, t_tokens as i32, 0i32, nparts as i32);
        let bstride = (HKV * HD * PAGE) as i32;
        let hqi = HQ as i32;
        let pg = PAGE as i32;
        let mut b = stream.launch_builder(&f);
        b.arg(&qp)
            .arg(&kcp2)
            .arg(&vcp2)
            .arg(&btp)
            .arg(&sop)
            .arg(&ssp)
            .arg(&alp)
            .arg(&scale)
            .arg(&nkv)
            .arg(&tt)
            .arg(&ctxb)
            .arg(&np_)
            .arg(&bstride)
            .arg(&kstride)
            .arg(&pg)
            .arg(&hqi)
            .arg(&dmp);
        let qchunks = (t_tokens + 63) / 64;
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: ((HQ / HKV) as u32, HKV as u32, (qchunks * nparts) as u32),
                block_dim: (256, 1, 1),
                shared_mem_bytes: (64 * HD * 2 * 2) as u32,
            })
        }
        .unwrap();
        stream.synchronize().unwrap();
        let mut so = vec![0u16; scr_out_elems];
        let mut ss = vec![0f32; scr_stat_elems];
        stream.memcpy_dtoh(&d_so, &mut so).unwrap();
        stream.memcpy_dtoh(&d_ss, &mut ss).unwrap();
        (so, ss)
    };
    let (so_c8, ss_c8) = run8(
        "owl_prefill_split_fp8kv_hd256",
        &kc_c8,
        &vc_c8,
        (HD * PAGE) as i32,
    );
    let (so_n8, ss_n8) = run8(
        "owl_prefill_split_fp8kv_knhd_hd256",
        &kc_n8,
        &vc_n8,
        HD as i32,
    );
    let bad8 = so_c8.iter().zip(so_n8.iter()).filter(|(a, b)| a != b).count();
    assert_eq!(bad8, 0, "split fp8 scr_out parity:{bad8} 处位型不符");
    assert_eq!(ss_c8, ss_n8, "split fp8 scr_stat(m,l)不符");
    assert!(so_c8.iter().any(|&v| v != 0), "split fp8 全零输出(装配错)");
}

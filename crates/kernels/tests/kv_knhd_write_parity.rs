//! kv布局统一契约 P1 写核 parity 探针(roadmap.local/kv布局统一契约-立项设计与施工.md §六.1)。
//!
//! 门 = **bitwise**:写核为纯重排(同值同量化式),classic 与 kNHD 两布局
//! 写出的池经 host 独立映射回读必须逐位一致;融合插池的 q_out(数学面)
//! 两变体同式 → 逐位一致。
//!
//! 覆盖:reshape_and_cache(f16/bf16/fp8kv)+ qknorm_rope_kv_insert
//! (f16/fp8kv);slots 含跨页/尾槽/padding(-1)。
//! 映射 oracle = 本文件独立重实现(与 env.rs 单源互证,双实现更强)。
//! 无 GPU 跳过(OWL_TEST_DEVICE;卡 = ordinal 1,3090 空闲卡纪律)。

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream, DevicePtr, LaunchConfig, PushKernelArg};

const HKV: usize = 2;
const HD: usize = 64;
const HQ: usize = 4;
const HALF: usize = HD / 2;
const PAGE: usize = 32;
const NB: usize = 4;
const T: usize = 8;
const X: usize = 8;

// ---- 独立映射 oracle(勿改为引用 env 单源 —— 探针价值在双实现互证)----
fn classic_k_off(slot: usize, h: usize, d: usize, page: usize, x: usize) -> usize {
    let (b, p) = (slot / page, slot % page);
    (((b * HKV + h) * (HD / x) + d / x) * page + p) * x + d % x
}
fn classic_v_off(slot: usize, h: usize, d: usize, page: usize) -> usize {
    let (b, p) = (slot / page, slot % page);
    ((b * HKV + h) * HD + d) * page + p
}
fn knhd_off(slot: usize, h: usize, d: usize, page: usize) -> usize {
    let (b, p) = (slot / page, slot % page);
    ((b * page + p) * HKV + h) * HD + d
}

fn half_bits(f: f32) -> u16 {
    half::f16::from_f32(f).to_bits()
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

fn deterministic_slots() -> [f32; T] {
    // 跨页/尾槽/padding 全覆盖
    [0.0, 31.0, 32.0, 63.0, 64.0, -1.0, 95.0, 33.0]
}

fn deterministic_u16(i: usize, salt: u32) -> u16 {
    half_bits((((i * 37 + salt as usize * 53) % 199) as f32) / 20.0 - 5.0)
}

#[test]
fn reshape_and_cache_knhd_parity() {
    if !owl_shared::env_reader::flag("OWL_TEST_DEVICE") {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    }
    let ctx = CudaContext::new(1).expect("ctx");
    ctx.bind_to_thread().expect("bind");
    let stream = ctx.default_stream();
    let slots = deterministic_slots();
    let pool_elems = NB * HKV * HD * PAGE;

    // ---- f16 变体 ----
    {
        let key: Vec<u16> = (0..T * HKV * HD).map(|i| deterministic_u16(i, 1)).collect();
        let value: Vec<u16> = (0..T * HKV * HD).map(|i| deterministic_u16(i, 2)).collect();
        let run = |name: &str, src: &str| -> (Vec<u16>, Vec<u16>) {
            let f = compile(&ctx, src, name);
            let mut d_k = unsafe { stream.alloc::<u16>(key.len()) }.unwrap();
            stream.memcpy_htod(&key, &mut d_k).unwrap();
            let mut d_v = unsafe { stream.alloc::<u16>(value.len()) }.unwrap();
            stream.memcpy_htod(&value, &mut d_v).unwrap();
            let mut d_kc = unsafe { stream.alloc::<u16>(pool_elems) }.unwrap();
            let mut d_vc = unsafe { stream.alloc::<u16>(pool_elems) }.unwrap();
            stream.memset_zeros(&mut d_kc).unwrap();
            stream.memset_zeros(&mut d_vc).unwrap();
            let mut d_slots = unsafe { stream.alloc::<f32>(slots.len()) }.unwrap();
            stream.memcpy_htod(&slots, &mut d_slots).unwrap();
            let mut d_out = unsafe { stream.alloc::<u16>(1) }.unwrap();
            let (kp, vp, kcp, vcp, slp, op) = (
                d_k.device_ptr(&stream).0,
                d_v.device_ptr(&stream).0,
                d_kc.device_ptr(&stream).0,
                d_vc.device_ptr(&stream).0,
                d_slots.device_ptr(&stream).0,
                d_out.device_ptr(&stream).0,
            );
            let (ks, hs, hst, bst, xt) = (
                (HKV * HD) as i32,
                HKV as i32,
                HD as i32,
                PAGE as i32,
                X as i32,
            );
            let mut b = stream.launch_builder(&f);
            b.arg(&kp)
                .arg(&vp)
                .arg(&kcp)
                .arg(&vcp)
                .arg(&slp)
                .arg(&ks) // key_stride
                .arg(&ks) // value_stride
                .arg(&hs)
                .arg(&hst)
                .arg(&bst)
                .arg(&xt)
                .arg(&op);
            unsafe {
                b.launch(LaunchConfig { grid_dim: (T as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })
            }
            .unwrap();
            stream.synchronize().unwrap();
            let mut kco = vec![0u16; pool_elems];
            let mut vco = vec![0u16; pool_elems];
            stream.memcpy_dtoh(&d_kc, &mut kco).unwrap();
            stream.memcpy_dtoh(&d_vc, &mut vco).unwrap();
            (kco, vco)
        };
        let (kc_c, vc_c) = run(
            "vllm_reshape_and_cache_f16",
            owl_kernels::sources::attention::RESHAPE_AND_CACHE_F16,
        );
        let (kc_n, vc_n) = run(
            "vllm_reshape_and_cache_f16_knhd",
            owl_kernels::sources::attention::RESHAPE_AND_CACHE_F16,
        );
        for &slot in &slots {
            if slot < 0.0 {
                continue;
            }
            let slot = slot as usize;
            for h in 0..HKV {
                for d in 0..HD {
                    let co = classic_k_off(slot, h, d, PAGE, X);
                    let no = knhd_off(slot, h, d, PAGE);
                    assert_eq!(kc_c[co], kc_n[no], "K f16 slot={slot} h={h} d={d}");
                    let cov = classic_v_off(slot, h, d, PAGE);
                    assert_eq!(vc_c[cov], vc_n[no], "V f16 slot={slot} h={h} d={d}");
                }
            }
        }
    }

    // ---- fp8kv 变体(e4m3 字节池)----
    {
        let key: Vec<u16> = (0..T * HKV * HD).map(|i| deterministic_u16(i, 3)).collect();
        let value: Vec<u16> = (0..T * HKV * HD).map(|i| deterministic_u16(i, 4)).collect();
        let run = |name: &str| -> (Vec<u8>, Vec<u8>) {
            let f = compile(
                &ctx,
                owl_kernels::sources::attention::RESHAPE_AND_CACHE_FP8KV,
                name,
            );
            let mut d_k = unsafe { stream.alloc::<u16>(key.len()) }.unwrap();
            stream.memcpy_htod(&key, &mut d_k).unwrap();
            let mut d_v = unsafe { stream.alloc::<u16>(value.len()) }.unwrap();
            stream.memcpy_htod(&value, &mut d_v).unwrap();
            let mut d_kc = unsafe { stream.alloc::<u8>(pool_elems) }.unwrap();
            let mut d_vc = unsafe { stream.alloc::<u8>(pool_elems) }.unwrap();
            stream.memset_zeros(&mut d_kc).unwrap();
            stream.memset_zeros(&mut d_vc).unwrap();
            let mut d_slots = unsafe { stream.alloc::<f32>(slots.len()) }.unwrap();
            stream.memcpy_htod(&slots, &mut d_slots).unwrap();
            let mut d_out = unsafe { stream.alloc::<u16>(1) }.unwrap();
            let (kp, vp, kcp, vcp, slp, op) = (
                d_k.device_ptr(&stream).0,
                d_v.device_ptr(&stream).0,
                d_kc.device_ptr(&stream).0,
                d_vc.device_ptr(&stream).0,
                d_slots.device_ptr(&stream).0,
                d_out.device_ptr(&stream).0,
            );
            let (ks, hs, hst, bst, xt) = (
                (HKV * HD) as i32,
                HKV as i32,
                HD as i32,
                PAGE as i32,
                X as i32,
            );
            let mut b = stream.launch_builder(&f);
            b.arg(&kp)
                .arg(&vp)
                .arg(&kcp)
                .arg(&vcp)
                .arg(&slp)
                .arg(&ks)
                .arg(&ks)
                .arg(&hs)
                .arg(&hst)
                .arg(&bst)
                .arg(&xt)
                .arg(&op);
            unsafe {
                b.launch(LaunchConfig { grid_dim: (T as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })
            }
            .unwrap();
            stream.synchronize().unwrap();
            let mut kco = vec![0u8; pool_elems];
            let mut vco = vec![0u8; pool_elems];
            stream.memcpy_dtoh(&d_kc, &mut kco).unwrap();
            stream.memcpy_dtoh(&d_vc, &mut vco).unwrap();
            (kco, vco)
        };
        let (kc_c, vc_c) = run("owl_reshape_and_cache_fp8kv");
        let (kc_n, vc_n) = run("owl_reshape_and_cache_fp8kv_knhd");
        for &slot in &slots {
            if slot < 0.0 {
                continue;
            }
            let slot = slot as usize;
            for h in 0..HKV {
                for d in 0..HD {
                    let no = knhd_off(slot, h, d, PAGE);
                    assert_eq!(
                        kc_c[classic_k_off(slot, h, d, PAGE, X)],
                        kc_n[no],
                        "K fp8 slot={slot} h={h} d={d}"
                    );
                    assert_eq!(
                        vc_c[classic_v_off(slot, h, d, PAGE)],
                        vc_n[no],
                        "V fp8 slot={slot} h={h} d={d}"
                    );
                }
            }
        }
    }

    // ---- bf16 变体(DFlash2 草稿池;位型 = f32 高 16 位)----
    {
        let key: Vec<u16> = (0..T * HKV * HD)
            .map(|i| ((((i * 37 + 11) % 199) as f32 / 20.0 - 5.0).to_bits() >> 16) as u16)
            .collect();
        let value: Vec<u16> = (0..T * HKV * HD)
            .map(|i| ((((i * 53 + 7) % 199) as f32 / 20.0 - 5.0).to_bits() >> 16) as u16)
            .collect();
        let run = |name: &str| -> (Vec<u16>, Vec<u16>) {
            let f = compile(&ctx, owl_kernels::sources::attention::RESHAPE_AND_CACHE_F16, name);
            let mut d_k = unsafe { stream.alloc::<u16>(key.len()) }.unwrap();
            stream.memcpy_htod(&key, &mut d_k).unwrap();
            let mut d_v = unsafe { stream.alloc::<u16>(value.len()) }.unwrap();
            stream.memcpy_htod(&value, &mut d_v).unwrap();
            let mut d_kc = unsafe { stream.alloc::<u16>(pool_elems) }.unwrap();
            let mut d_vc = unsafe { stream.alloc::<u16>(pool_elems) }.unwrap();
            stream.memset_zeros(&mut d_kc).unwrap();
            stream.memset_zeros(&mut d_vc).unwrap();
            let mut d_slots = unsafe { stream.alloc::<f32>(slots.len()) }.unwrap();
            stream.memcpy_htod(&slots, &mut d_slots).unwrap();
            let mut d_out = unsafe { stream.alloc::<u16>(1) }.unwrap();
            let (kp, vp, kcp, vcp, slp, op) = (
                d_k.device_ptr(&stream).0,
                d_v.device_ptr(&stream).0,
                d_kc.device_ptr(&stream).0,
                d_vc.device_ptr(&stream).0,
                d_slots.device_ptr(&stream).0,
                d_out.device_ptr(&stream).0,
            );
            let (ks, hs, hst, bst, xt) = (
                (HKV * HD) as i32,
                HKV as i32,
                HD as i32,
                PAGE as i32,
                X as i32,
            );
            let mut b = stream.launch_builder(&f);
            b.arg(&kp)
                .arg(&vp)
                .arg(&kcp)
                .arg(&vcp)
                .arg(&slp)
                .arg(&ks)
                .arg(&ks)
                .arg(&hs)
                .arg(&hst)
                .arg(&bst)
                .arg(&xt)
                .arg(&op);
            unsafe {
                b.launch(LaunchConfig { grid_dim: (T as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })
            }
            .unwrap();
            stream.synchronize().unwrap();
            let mut kco = vec![0u16; pool_elems];
            let mut vco = vec![0u16; pool_elems];
            stream.memcpy_dtoh(&d_kc, &mut kco).unwrap();
            stream.memcpy_dtoh(&d_vc, &mut vco).unwrap();
            (kco, vco)
        };
        let (kc_c, vc_c) = run("vllm_reshape_and_cache_bf16");
        let (kc_n, vc_n) = run("vllm_reshape_and_cache_bf16_knhd");
        for &slot in &slots {
            if slot < 0.0 {
                continue;
            }
            let slot = slot as usize;
            for h in 0..HKV {
                for d in 0..HD {
                    let no = knhd_off(slot, h, d, PAGE);
                    assert_eq!(
                        kc_c[classic_k_off(slot, h, d, PAGE, X)],
                        kc_n[no],
                        "K bf16 slot={slot} h={h} d={d}"
                    );
                    assert_eq!(
                        vc_c[classic_v_off(slot, h, d, PAGE)],
                        vc_n[no],
                        "V bf16 slot={slot} h={h} d={d}"
                    );
                }
            }
        }
    }
}


/// 池 dtype I/O 统一面(macro 双型展开用)
trait PoolIo: Copy {
    fn alloc(stream: &Arc<CudaStream>, n: usize) -> cudarc::driver::CudaSlice<Self>;
    fn zero(stream: &Arc<CudaStream>, d: &mut cudarc::driver::CudaSlice<Self>);
    fn read(stream: &Arc<CudaStream>, d: &cudarc::driver::CudaSlice<Self>, out: &mut Vec<Self>);
}
impl PoolIo for u16 {
    fn alloc(stream: &Arc<CudaStream>, n: usize) -> cudarc::driver::CudaSlice<Self> {
        unsafe { stream.alloc::<u16>(n) }.unwrap()
    }
    fn zero(stream: &Arc<CudaStream>, d: &mut cudarc::driver::CudaSlice<Self>) {
        stream.memset_zeros(d).unwrap();
    }
    fn read(stream: &Arc<CudaStream>, d: &cudarc::driver::CudaSlice<Self>, out: &mut Vec<Self>) {
        stream.memcpy_dtoh(d, out).unwrap();
    }
}
impl PoolIo for u8 {
    fn alloc(stream: &Arc<CudaStream>, n: usize) -> cudarc::driver::CudaSlice<Self> {
        unsafe { stream.alloc::<u8>(n) }.unwrap()
    }
    fn zero(stream: &Arc<CudaStream>, d: &mut cudarc::driver::CudaSlice<Self>) {
        stream.memset_zeros(d).unwrap();
    }
    fn read(stream: &Arc<CudaStream>, d: &cudarc::driver::CudaSlice<Self>, out: &mut Vec<Self>) {
        stream.memcpy_dtoh(d, out).unwrap();
    }
}

#[test]
fn fused_insert_knhd_parity() {
    if !owl_shared::env_reader::flag("OWL_TEST_DEVICE") {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    }
    let ctx = CudaContext::new(1).expect("ctx");
    ctx.bind_to_thread().expect("bind");
    let stream = ctx.default_stream();
    let slots = deterministic_slots();
    let pool_elems = NB * HKV * HD * PAGE;
    let max_pos = 8usize;

    // 输入面:f16 位型确定性;q_raw [T, HQ·2HD](per-head [value|gate])
    let q_raw: Vec<u16> = (0..T * HQ * 2 * HD).map(|i| deterministic_u16(i, 5)).collect();
    let k: Vec<u16> = (0..T * HKV * HD).map(|i| deterministic_u16(i, 6)).collect();
    let v: Vec<u16> = (0..T * HKV * HD).map(|i| deterministic_u16(i, 7)).collect();
    let q_w: Vec<u16> = (0..HD).map(|i| half_bits(((i % 13) as f32) / 50.0)).collect();
    let k_w: Vec<u16> = (0..HD).map(|i| half_bits(((i % 7) as f32) / 40.0)).collect();
    let cos_t: Vec<u16> = (0..max_pos * HALF).map(|i| half_bits(((i % 17) as f32 / 17.0) * 2.0 - 1.0)).collect();
    let sin_t: Vec<u16> = (0..max_pos * HALF).map(|i| half_bits(((i % 11) as f32 / 11.0) * 2.0 - 1.0)).collect();
    let pos: Vec<f32> = (0..T).map(|t| (t % max_pos) as f32).collect();
    let eps = 1e-6f32;
    let (hkv_i, half_i, page_i, w_off_i) = (HKV as i32, HALF as i32, PAGE as i32, 1i32);

    macro_rules! run {
        ($name:expr, $src:expr, $pool:ty) => {{
            let f = compile(&ctx, $src, $name);
            let mut d_q = unsafe { stream.alloc::<u16>(q_raw.len()) }.unwrap();
            stream.memcpy_htod(&q_raw, &mut d_q).unwrap();
            let mut d_k = unsafe { stream.alloc::<u16>(k.len()) }.unwrap();
            stream.memcpy_htod(&k, &mut d_k).unwrap();
            let mut d_v = unsafe { stream.alloc::<u16>(v.len()) }.unwrap();
            stream.memcpy_htod(&v, &mut d_v).unwrap();
            let mut d_kc = PoolIo::alloc(&stream, pool_elems);
            let mut d_vc = PoolIo::alloc(&stream, pool_elems);
            PoolIo::zero(&stream, &mut d_kc);
            PoolIo::zero(&stream, &mut d_vc);
            let mut d_slots = unsafe { stream.alloc::<f32>(slots.len()) }.unwrap();
            stream.memcpy_htod(&slots, &mut d_slots).unwrap();
            let mut d_qw = unsafe { stream.alloc::<u16>(q_w.len()) }.unwrap();
            stream.memcpy_htod(&q_w, &mut d_qw).unwrap();
            let mut d_kw = unsafe { stream.alloc::<u16>(k_w.len()) }.unwrap();
            stream.memcpy_htod(&k_w, &mut d_kw).unwrap();
            let mut d_cos = unsafe { stream.alloc::<u16>(cos_t.len()) }.unwrap();
            stream.memcpy_htod(&cos_t, &mut d_cos).unwrap();
            let mut d_sin = unsafe { stream.alloc::<u16>(sin_t.len()) }.unwrap();
            stream.memcpy_htod(&sin_t, &mut d_sin).unwrap();
            let mut d_pos = unsafe { stream.alloc::<f32>(pos.len()) }.unwrap();
            stream.memcpy_htod(&pos, &mut d_pos).unwrap();
            let mut d_qo = unsafe { stream.alloc::<u16>(T * HQ * HD) }.unwrap();
            let (q, kp, vp, kcp, vcp, slp, qw, kw, co, si, po, qo) = (
                d_q.device_ptr(&stream).0,
                d_k.device_ptr(&stream).0,
                d_v.device_ptr(&stream).0,
                d_kc.device_ptr(&stream).0,
                d_vc.device_ptr(&stream).0,
                d_slots.device_ptr(&stream).0,
                d_qw.device_ptr(&stream).0,
                d_kw.device_ptr(&stream).0,
                d_cos.device_ptr(&stream).0,
                d_sin.device_ptr(&stream).0,
                d_pos.device_ptr(&stream).0,
                d_qo.device_ptr(&stream).0,
            );
            let mut b = stream.launch_builder(&f);
            b.arg(&q)
                .arg(&kp)
                .arg(&vp)
                .arg(&kcp)
                .arg(&vcp)
                .arg(&slp)
                .arg(&qw)
                .arg(&kw)
                .arg(&co)
                .arg(&si)
                .arg(&po)
                .arg(&eps)
                .arg(&hkv_i)
                .arg(&half_i)
                .arg(&page_i)
                .arg(&w_off_i)
                .arg(&qo);
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: (T as u32, (HQ + HKV) as u32, 1),
                    block_dim: (HD as u32, 1, 1),
                    shared_mem_bytes: (HD * 4) as u32,
                })
            }
            .unwrap();
            stream.synchronize().unwrap();
            let mut qoo = vec![0u16; T * HQ * HD];
            let mut kco: Vec<$pool> = vec![0 as $pool; pool_elems];
            let mut vco: Vec<$pool> = vec![0 as $pool; pool_elems];
            stream.memcpy_dtoh(&d_qo, &mut qoo).unwrap();
            PoolIo::read(&stream, &d_kc, &mut kco);
            PoolIo::read(&stream, &d_vc, &mut vco);
            (qoo, kco, vco)
        }};
    }

    // ---- f16 池 ----
    let (qo_c, kc_c, vc_c) = run!(
        "owl_qknorm_rope_kv_insert_f16",
        owl_kernels::sources::owl::QKNORM_ROPE_KV_INSERT_F16,
        u16
    );
    let (qo_n, kc_n, vc_n) = run!(
        "owl_qknorm_rope_kv_insert_f16_knhd",
        owl_kernels::sources::owl::QKNORM_ROPE_KV_INSERT_F16,
        u16
    );
    assert_eq!(qo_c, qo_n, "q_out f16 bitwise");
    for &slot in &slots {
        if slot < 0.0 {
            continue;
        }
        let slot = slot as usize;
        for h in 0..HKV {
            for d in 0..HD {
                let no = knhd_off(slot, h, d, PAGE);
                assert_eq!(
                    kc_c[classic_k_off(slot, h, d, PAGE, X)],
                    kc_n[no],
                    "insert K f16 slot={slot} h={h} d={d}"
                );
                assert_eq!(
                    vc_c[classic_v_off(slot, h, d, PAGE)],
                    vc_n[no],
                    "insert V f16 slot={slot} h={h} d={d}"
                );
            }
        }
    }

    // ---- fp8 池(e4m3 字节)----
    let (qo_c, kc_c, vc_c) = run!(
        "owl_qknorm_rope_kv_insert_f16_fp8kv",
        owl_kernels::sources::owl::QKNORM_ROPE_KV_INSERT_F16_FP8KV,
        u8
    );
    let (qo_n, kc_n, vc_n) = run!(
        "owl_qknorm_rope_kv_insert_f16_fp8kv_knhd",
        owl_kernels::sources::owl::QKNORM_ROPE_KV_INSERT_F16_FP8KV,
        u8
    );
    assert_eq!(qo_c, qo_n, "q_out fp8kv bitwise");
    for &slot in &slots {
        if slot < 0.0 {
            continue;
        }
        let slot = slot as usize;
        for h in 0..HKV {
            for d in 0..HD {
                let no = knhd_off(slot, h, d, PAGE);
                assert_eq!(
                    kc_c[classic_k_off(slot, h, d, PAGE, X)],
                    kc_n[no],
                    "insert K fp8 slot={slot} h={h} d={d}"
                );
                assert_eq!(
                    vc_c[classic_v_off(slot, h, d, PAGE)],
                    vc_n[no],
                    "insert V fp8 slot={slot} h={h} d={d}"
                );
            }
        }
    }
}

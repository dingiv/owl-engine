//! GDN 标量门 chunked 前向(lmdeploy pre_sm90 port)—— 算子层金标对拍
//! (2026-10-03)。金标 = FLA 0.6.0 五级阶段张量(testdata/gdn_fla/cases,
//! f32 raw bin);本核单发替代五核流水,只比对两处终端量:
//! - `o`   [T,NV,VD]:FLA chunk_fwd_o 产物(内含 scale);
//! - state [NV,KD,VD]:初态 c1-c3 = 零,c4 = s0;跑后对拍 FLA `final`。
//!
//! 容差律:本核 fp32 全程,FLA 有 bf16 中间量(w/u/h 存储)—— 偏差主体
//! 是 FLA 侧量化误差,沿既有 5e-2 硬门。
//! 无 GPU 跳过(OWL_TEST_DEVICE)。

use cudarc::driver::{CudaContext, DevicePtr, LaunchConfig, PushKernelArg};

const CASES: &[(&str, usize)] = &[
    ("c1_t64", 64),
    ("c2_t128", 128),
    ("c3_t96", 96),
    ("c4_t64_s0", 64),
];

const NK: usize = 16;
const NV: usize = 48;
const KD: usize = 128;
const VD: usize = 128;
const SCALE: f32 = 0.08838834764831845; // 1/sqrt(128)

struct Case {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    g: Vec<f32>,
    beta: Vec<f32>,
    s0: Option<Vec<f32>>,
    o: Vec<f32>,
    final_state: Vec<f32>,
}

fn read_bin(dir: &str, name: &str) -> Vec<f32> {
    let p = format!("{dir}/{name}.bin");
    let bytes = owl_shared::file_loader::read(&p).unwrap_or_else(|e| panic!("{p}: {e}"));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn load_case(dir: &str, tag: &str, _t: usize) -> Case {
    let rd = |n: &str| read_bin(dir, &format!("{tag}.{n}"));
    let has_s0 = owl_shared::file_loader::metadata(format!("{dir}/{tag}.s0.bin")).is_ok();
    Case {
        q: rd("q"),
        k: rd("k"),
        v: rd("v"),
        g: rd("g"),
        beta: rd("beta"),
        s0: has_s0.then(|| rd("s0")),
        o: rd("o"),
        final_state: rd("final"),
    }
}

fn max_dev(a: &[f32], b: &[f32]) -> (usize, f32) {
    let mut worst = 0f32;
    let mut idx = 0;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let d = (x - y).abs();
        if d > worst {
            worst = d;
            idx = i;
        }
    }
    (idx, worst)
}

#[test]
fn gdn_scalar_golden() {
    if !owl_shared::env_reader::flag("OWL_TEST_DEVICE") {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    }
    let ctx = CudaContext::new(1).expect("ctx");
    ctx.bind_to_thread().expect("bind");
    let stream = ctx.default_stream();

    let cubin = owl_kernels::family::gdn_scalar::cubin::CHUNK_SCALAR_F32;
    let m = ctx
        .load_module(cudarc::nvrtc::Ptx::from_binary(cubin.to_vec()))
        .expect("load gdn_chunk_scalar cubin");
    let f = m
        .load_function(owl_kernels::family::gdn_scalar::cubin::KERNEL_F32)
        .expect("fn owl_gdn_chunk_scalar_f32");

    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/gdn_fla/cases");

    for (tag, t) in CASES {
        let case = load_case(dir, tag, *t);
        assert_eq!(case.q.len(), t * NK * KD, "{tag}: q 形状");

        let mut q = stream.alloc_zeros::<f32>(case.q.len()).unwrap();
        stream.memcpy_htod(&case.q, &mut q).unwrap();
        let mut k = stream.alloc_zeros::<f32>(case.k.len()).unwrap();
        stream.memcpy_htod(&case.k, &mut k).unwrap();
        let mut v = stream.alloc_zeros::<f32>(case.v.len()).unwrap();
        stream.memcpy_htod(&case.v, &mut v).unwrap();
        let mut beta = stream.alloc_zeros::<f32>(case.beta.len()).unwrap();
        stream.memcpy_htod(&case.beta, &mut beta).unwrap();
        let mut g = stream.alloc_zeros::<f32>(case.g.len()).unwrap();
        stream.memcpy_htod(&case.g, &mut g).unwrap();
        // state = [NV, KD, VD]:c4 带 s0,其余零初态
        let state_host: Vec<f32> = case.s0.clone().unwrap_or_else(|| vec![0.0; NV * KD * VD]);
        let mut state = stream.alloc_zeros::<f32>(NV * KD * VD).unwrap();
        stream.memcpy_htod(&state_host, &mut state).unwrap();
        let o = stream.alloc_zeros::<f32>(t * NV * VD).unwrap();
        let seq_off: Vec<i32> = vec![0, *t as i32];
        let mut soff = stream.alloc_zeros::<i32>(2).unwrap();
        stream.memcpy_htod(&seq_off, &mut soff).unwrap();

        let q_p = q.device_ptr(&stream).0;
        let k_p = k.device_ptr(&stream).0;
        let v_p = v.device_ptr(&stream).0;
        let beta_p = beta.device_ptr(&stream).0;
        let g_p = g.device_ptr(&stream).0;
        let state_p = state.device_ptr(&stream).0;
        let o_p = o.device_ptr(&stream).0;
        let soff_p = soff.device_ptr(&stream).0;
        let nk = NK as i32;
        let hv = NV as i32;
        let kd = KD as i32;
        let scale = SCALE;

        {
            let mut b = stream.launch_builder(&f);
            b.arg(&o_p)
                .arg(&q_p)
                .arg(&k_p)
                .arg(&v_p)
                .arg(&beta_p)
                .arg(&g_p)
                .arg(&state_p)
                .arg(&soff_p)
                .arg(&nk)
                .arg(&hv)
                .arg(&kd)
                .arg(&scale);
            let cfg = LaunchConfig {
                grid_dim: (1, NV as u32, 1),
                block_dim: (owl_kernels::family::gdn_scalar::cubin::BLOCK, 1, 1),
                shared_mem_bytes: owl_kernels::family::gdn_scalar::cubin::SMEM_D128,
            };
            unsafe { b.launch(cfg) }.expect("owl_gdn_chunk_scalar_f32 launch");
        }
        stream.synchronize().expect("sync");

        let o_host = stream.memcpy_dtov(&o).expect("dtoh o");
        let st_host = stream.memcpy_dtov(&state).expect("dtoh state");
        let (oi, doo) = max_dev(&o_host, &case.o);
        let (si, dst) = max_dev(&st_host, &case.final_state);
        eprintln!("[{tag}] o={doo:.3e}@{oi} state={dst:.3e}@{si}");
        assert!(doo < 5e-2, "{tag}: o 偏差 {doo}");
        assert!(dst < 5e-2, "{tag}: state 偏差 {dst}");
    }
}

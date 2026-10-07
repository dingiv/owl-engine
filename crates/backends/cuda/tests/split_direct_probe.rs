//! 临时探针 2:直调真 split 核(绕开 owl server marshalling)。
//! 用完即删。

use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::CompileOptions;

fn src() -> String {
    let s = include_str!("../../../kernels/cu/attention/prefill_split_f16.cu").to_string();
    match owl_shared::env_reader::str("SPLIT_SKIP").as_deref() {
        Some("qload") => s.replace(
            "    #pragma unroll\n    for (int v = 0; v < VQ; ++v) {\n        const uint4 raw = *reinterpret_cast<const uint4*>(\n            q + q_off + v * VEC);\n        const uint16_t* h = reinterpret_cast<const uint16_t*>(&raw);\n        #pragma unroll\n        for (int e = 0; e < VEC; ++e) qv[v][e] = __half2float(\n            *reinterpret_cast<const __half*>(&h[e]));\n    }",
            "    // q load skipped"),
        _ => s,
    }
}
fn SRC() -> String { src() }

#[test]
fn split_direct_launch() {
    let ctx = CudaContext::new(1).expect("ctx");
    ctx.bind_to_thread().expect("bind");
    let stream = ctx.default_stream();
    let opts = CompileOptions {
        arch: Some("compute_86"),
        include_paths: vec!["/usr/local/cuda/include".to_string()],
        ..Default::default()
    };
    let rtc = cudarc::nvrtc::compile_ptx_with_opts(&SRC(), opts).expect("ptx");
    let ptx_str = rtc.to_src();
    for line in ptx_str.lines() {
        if line.contains(".param") || line.contains(".entry") || line.contains(".local") {
            println!("[ptx] {line}");
        }
    }
    let m = ctx.load_module(rtc).expect("module");
    let f = m.load_function("owl_prefill_split_f16_hd256").expect("func");


    use cudarc::driver::sys::CUfunction_attribute_enum as Attr;
    f.set_attribute(Attr::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 65536)
        .expect("setattr");

    let (t, hq, hkv, hd, nparts) = (8usize, 2usize, 1usize, 256usize, 1usize);
    let mut q = q_dev(&stream, t * hq * hd);
    let mut kc = q_dev(&stream, hkv * (hd / 8) * 512);
    let mut vc = q_dev(&stream, hkv * hd * 32);
    let mut bt = q_dev_f32(&stream, 1);
    let mut scr_out = q_dev(&stream, t * hq * nparts * hd);
    let mut scr_stat = q_dev_f32(&stream, t * hq * nparts * 2);
    let mut dummy = q_dev(&stream, 1);
    let mut dummy2 = q_dev(&stream, 1);

    let scale: f32 = 0.0625;
    let (ihkv, it, icb, inp) = (hkv as i32, t as i32, 0i32, nparts as i32);
    let (kbs, khs, pg, ihq) = ((hkv * hd * 32) as i32, (hd * 32) as i32, 32i32, hq as i32);

    let smem: u32 = owl_shared::env_reader::parse_or("SPLIT_SMEM", 65536);
    let cfg = LaunchConfig {
        grid_dim: ((hq / hkv) as u32, hkv as u32, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: smem,
    };
    let mut b = stream.launch_builder(&f);
    b.arg(&q);
    b.arg(&kc);
    b.arg(&vc);
    b.arg(&bt);
    b.arg(&mut scr_out);
    b.arg(&mut scr_stat);
    b.arg(&dummy); // k0_alibi 槽(不解引用)
    // 尾参 dummy 复用同一 slice:拷贝一份指针别名(探针域可接受)
    b.arg(&scale);
    b.arg(&ihkv);
    b.arg(&it);
    b.arg(&icb);
    b.arg(&inp);
    b.arg(&kbs);
    b.arg(&khs);
    b.arg(&pg);
    b.arg(&ihq);
    b.arg(&mut dummy2); // 尾参输出(契约 4)
    unsafe { b.launch(cfg) }.expect("launch split");

    // reduce 直调
    let f2 = m.load_function("owl_prefill_split_reduce_f16_hd256").expect("func2");
    let mut out = stream.alloc_zeros::<u16>(t * hq * hd).expect("out");
    let (inp, ihq2, ihd, it2) = (nparts as i32, hq as i32, hd as i32, t as i32);
    let cfg2 = LaunchConfig {
        grid_dim: (((t * hq * hd) + 255) as u32 / 256, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b2 = stream.launch_builder(&f2);
    b2.arg(&dummy); // alibi
    b2.arg(&mut scr_out);
    b2.arg(&mut scr_stat);
    b2.arg(&inp);
    b2.arg(&ihq2);
    b2.arg(&ihd);
    b2.arg(&it2);
    b2.arg(&mut out);
    unsafe { b2.launch(cfg2) }.expect("launch reduce");
    stream.synchronize().expect("sync");

    // 数值校验:host softmax 金标(q/k/v = 0 填充 → 全零输出,仅验证管线)
    let mut oh = vec![0u16; t * hq * hd];
    stream.memcpy_dtoh(&out, &mut oh).expect("dtoh out");
    let nz = oh.iter().filter(|&&v| v != 0).count();
    println!("[probe2] 真 split 核直调 ✓ out 非零 {}/{}", nz, oh.len());
    // scr_stat[16..32] 诊断段
    let mut sb = vec![0f32; 32];
    stream.memcpy_dtoh(&scr_stat, &mut sb).expect("dtoh stat");
    println!("[probe2] scr_stat = {sb:?}");
}

fn q_dev(stream: &std::sync::Arc<cudarc::driver::CudaStream>, n: usize) -> cudarc::driver::CudaSlice<u16> {
    stream.alloc_zeros::<u16>(n).expect("alloc u16")
}
fn q_dev_f32(stream: &std::sync::Arc<cudarc::driver::CudaStream>, n: usize) -> cudarc::driver::CudaSlice<f32> {
    stream.alloc_zeros::<f32>(n).expect("alloc f32")
}

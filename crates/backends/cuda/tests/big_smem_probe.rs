//! 临时探针:大动态 smem(64KB)发射在本机 driver(615.71.09 patched)上
//! 是否可用 —— cuLaunchKernel SIGSEGV 归因。用完即删。

use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::CompileOptions;

const SRC: &str = r#"
extern "C" __global__ void smem_touch(unsigned short* out, unsigned int n) {
    extern __shared__ unsigned short smem[];
    for (unsigned int i = threadIdx.x; i < n; i += blockDim.x) smem[i] = (unsigned short)i;
    __syncthreads();
    unsigned short acc = 0;
    for (unsigned int i = threadIdx.x; i < n; i += blockDim.x) acc += smem[n - 1 - i];
    if (threadIdx.x == 0) out[blockIdx.x] = acc;
}
"#;

#[test]
fn big_dynamic_smem_launch() {
    let ctx = CudaContext::new(1).expect("ctx");
    ctx.bind_to_thread().expect("bind");
    let stream = ctx.default_stream();
    let opts = CompileOptions {
        arch: Some("compute_86"),
        ..Default::default()
    };
    let rtc = cudarc::nvrtc::compile_ptx_with_opts(SRC, opts).expect("ptx");
    let m = ctx.load_module(rtc).expect("module");
    let f = m.load_function("smem_touch").expect("func");

    // opt-in:64KB
    use cudarc::driver::sys::CUfunction_attribute_enum as Attr;
    f.set_attribute(Attr::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 65536)
        .expect("setattr");

    let mut out = stream.alloc_zeros::<u16>(4).expect("alloc");
    let n: u32 = 32768; // 64KB
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 65536,
    };
    let mut b = stream.launch_builder(&f);
    b.arg(&mut out);
    b.arg(&n);
    unsafe { b.launch(cfg) }.expect("launch 64KB");
    stream.synchronize().expect("sync");
    println!("[probe] 64KB 动态 smem 发射 ✓");
}

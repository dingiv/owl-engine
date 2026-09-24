//! M-Ⅱ 追凶:TensorParallelRowLinear(out_proj)数值探针。
//! 输入 = HF 层 0 门控 rmsnorm 输出(真值);权重 = 真实 out_proj.weight [1024,2048];
//! host 参考 = numpy 语义 x@W.T。若 owl 输出 ≈ 0 或错位,即 RowLinear 断裂实锤。

use owl_cuda::CudaDevice;
use owl_engine::models::dry_kernels::DryKernels;
use owl_engine::models::layers::distributed::{Comm, TensorParallelRowLinear};
use owl_engine::models::layers::{ctx_scope, VarBuilderX};
use owl_nn::cublas::NnBlas;
use std::sync::Arc;

#[test]
fn outproj_rowlinear_numerics() {
    let dev = Arc::new(CudaDevice::new(owl_cuda::test_device_ordinal(), owl_cuda::TEST_POOL_BYTES).unwrap());
    let scratch = dev.default_pool();
    let wpool = dev.default_pool();
    let ops = owl_nn::OpsCtx::new(&dev).unwrap();
    let blas = NnBlas::new(&dev).unwrap();
    let dry = DryKernels::new(dev.ctx()).unwrap();
    ctx_scope::install(ops, blas, dry, scratch.clone(), wpool.clone(), &dev);

    let st = owl_engine::loader::safetensors::SafeTensorsFile::open(
        "/home/div/Documents/codes/models/Qwen/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors",
    ).unwrap();
    let w = st.tensor_f32("model.language_model.layers.0.linear_attn.out_proj.weight").unwrap();
    assert_eq!(w.len(), 1024 * 2048);

    // 输入:HF 层 0 门控 rmsnorm 输出(真值,由 /tmp/hf_probe.py 抓取)
    // /tmp/hf_l0_norm.npy:HF 层 0 门控 rmsnorm 输出(80,128) f32 LE
    let raw = std::fs::read("/tmp/hf_l0_norm.npy").expect("读 npy");
    // NPY v1:128 字节头
    let hdr = raw.iter().position(|&b| b == b'\n').unwrap();
    let x: Vec<f32> = raw[hdr + 1..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(x.len(), 5 * 2048);

    let vb = VarBuilderX::new(
        &owl_engine::downloader::ModelPaths {
            tokenizer_filename: Default::default(), tokenizer_config_filename: Default::default(),
            config_filename: Default::default(), generation_config_filename: Default::default(),
            filenames: vec![std::path::PathBuf::from("/home/div/Documents/codes/models/Qwen/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors")],
            auxiliary_filenames: vec![], chat_template_filename: None,
        },
        false, owl_nn::Dtype::F32, &dev,
    ).unwrap().with_pool(wpool.clone());

    let vb = vb.pp("model.layers.0.linear_attn.out_proj");
    let row = TensorParallelRowLinear::new_loaded(
        2048, 1024, &vb, owl_engine::models::layers::Shard::default(),
        &None, &None, owl_nn::Dtype::F32, false, Comm::new(),
    ).expect("RowLinear 构造");

    let xt = owl_engine::models::layers::ctor::from_vec(x.clone(), (5usize, 2048), &dev).unwrap();
    let y = row.forward(&xt).expect("RowLinear forward");
    dev.ctx().synchronize().unwrap();
    let mut got = vec![0f32; 5 * 1024];
    unsafe {
        use owl_cuda::ffi::sys;
        sys::cuMemcpyDtoH_v2(
            got.as_mut_ptr() as *mut std::ffi::c_void,
            y.device_ptr() as sys::CUdeviceptr, got.len() * 4,
        ).result().unwrap();
    }

    // numpy 参考:y = x @ W.T(W [1024,2048])
    let mut want = vec![0f32; 5 * 1024];
    for t in 0..5 {
        for o in 0..1024 {
            let mut s = 0f32;
            for i in 0..2048 {
                s += x[t * 2048 + i] * w[o * 2048 + i];
            }
            want[t * 1024 + o] = s;
        }
    }
    let maxd: f32 = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
    let got_max = got.iter().fold(0f32, |m, v| m.max(v.abs()));
    eprintln!("[outproj] owl max|y| = {got_max:.4}, max|owl-numpy| = {maxd:.4e}");
    assert!(got_max > 0.1, "RowLinear 输出幅值异常: {got_max}");
    assert!(maxd < 0.05, "RowLinear 数值发散: max|Δ| = {maxd}");
}

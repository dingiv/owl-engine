
use owl_engine::config::Config;
use owl_engine::models::layers::{ctx_scope, VarBuilderX};
use owl_engine::models::qwen3_5::{DecodeGraphAdapter, Qwen3_5ForCausalLM, InputMetadata};
use owl_engine::graphplan::GraphPlan;
use owl_graph::GraphAllowance;
use owl_engine::models::dry_kernels::DryKernels;
use owl_engine::models::layers::distributed::Comm;
use owl_nn::cublas::NnBlas;
use owl_nn::OpsCtx;
use owl_cuda::CudaDevice;
use owl_iface::{Device as _, MemPhase, PoolConfig, PoolKind};
use std::sync::Arc;

fn tiny_config() -> Config {
    serde_json::from_str(
        r#"{
        "architectures": ["Qwen3_5ForCausalLM"],
        "hidden_size": 128,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "head_dim": 32,
        "num_hidden_layers": 2,
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-5,
        "intermediate_size": 256,
        "vocab_size": 96,
        "rope_theta": 10000.0,
        "hidden_act": "Silu",
        "attn_output_gate": false,
        "tie_word_embeddings": false,
        "extra_config_json": "{\"full_attention_interval\": 1}"
    }"#,
    )
    .expect("tiny config 反序列化")
}

fn dtoh_f32(dev: &CudaDevice, ptr: *mut f32, n: usize) -> Vec<f32> {
    use owl_cuda::ffi::sys;
    dev.ctx().bind_to_thread().unwrap();
    let mut out = vec![0f32; n];
    unsafe {
        sys::cuMemcpyDtoH_v2(
            out.as_mut_ptr() as *mut std::ffi::c_void,
            ptr as sys::CUdeviceptr,
            out.len() * 4,
        )
        .result()
        .unwrap();
    }
    out
}

#[test]
fn dry_run_fake_weights_graph_capture_replay() {
    let dev = CudaDevice::new(owl_cuda::test_device_ordinal()).expect("需要 CUDA 设备");
    let config = tiny_config();
    let vocab = config.vocab_size.unwrap();
    let layers = config.num_hidden_layers;
    let hkv = config.num_key_value_heads;
    let hd = config.head_dim.unwrap();
    let slots_cap = 64usize; // 空跑步数上限的 KV 容量

    // P 阶段池:权重/激活 分池(裁决 5;A1.1 分解,数字 = 空跑量级)
    let wpool = Arc::new(
        dev.create_pool(PoolConfig {
            name: "dry-weights".into(),
            kind: PoolKind::Weights,
            bytes: 64 << 20,
        })
        .unwrap(),
    );
    let spool = Arc::new(
        dev.create_pool(PoolConfig {
            name: "dry-scratch".into(),
            kind: PoolKind::Scratch,
            bytes: 32 << 20,
        })
        .unwrap(),
    );

    // ctx_scope rig(OwlTensor 垫片的执行凭证源;临时脚手架)
    let ops = OpsCtx::new_with_scratch(&dev, 8 << 20).expect("ops");
    let blas = NnBlas::new(&dev).expect("blas");
    let dry = DryKernels::new(dev.ctx()).expect("dry kernels");
    ctx_scope::install(ops, blas, dry, Arc::clone(&spool), Arc::clone(&wpool), &dev);


    // 假权重 VarBuilder + 模型
    let vb = VarBuilderX::from_fake(Arc::clone(&wpool), &dev).expect("fake vb");
    let model = Arc::new(
        Qwen3_5ForCausalLM::new(
            &vb,
            Comm::new(),
            &config,
            owl_nn::Dtype::F32,
            false,
            &dev,
        )
        .expect("模型构造(假权重)"),
    );

    // 每层 KV cache:flat [slots_cap, Hkv*D](naive 核 slot 直排布局)
    let kv_dim = hkv * hd;
    let mut kv_caches: Vec<(owl_nn::DynTensor<owl_cuda::CudaDevice>, owl_nn::DynTensor<owl_cuda::CudaDevice>)> =
        Vec::new();
    for _ in 0..layers {
        let k = owl_nn::TensorPoolOps::from_vec_tensor(wpool.as_ref(), &[slots_cap, kv_dim], vec![0f32; slots_cap * kv_dim]).unwrap();
        let v = owl_nn::TensorPoolOps::from_vec_tensor(wpool.as_ref(), &[slots_cap, kv_dim], vec![0f32; slots_cap * kv_dim]).unwrap();
        kv_caches.push((owl_nn::DynTensor::from_f32(&k), owl_nn::DynTensor::from_f32(&v)));
    }

    let adapter = DecodeGraphAdapter::new(Arc::clone(&model), kv_caches.clone(), vocab, false);

    // ---- warmup(姿势 6):eager decode 形态一次(seq_len=1)----
    {
        let ectx = ctx_scope::eager_ctx(MemPhase::Live).expect("eager ctx");
        let _g = ctx_scope::push(&ectx);
        let frontier = owl_nn::TensorPoolOps::from_vec_tensor(wpool.as_ref(), &[1], vec![11u32]).unwrap();
        let positions = owl_nn::TensorPoolOps::from_vec_tensor(wpool.as_ref(), &[1], vec![0u32]).unwrap();
        let slots = owl_nn::TensorPoolOps::from_vec_tensor(wpool.as_ref(), &[1], vec![0u32]).unwrap();
        let kv_lens = owl_nn::TensorPoolOps::from_vec_tensor(wpool.as_ref(), &[1], vec![1u32]).unwrap();
        let positions_dyn = owl_nn::DynTensor::from_u32(&positions);
        let meta = InputMetadata {
            seqlens: Some(vec![1]),
            is_prefill: false,
            is_mtp_verify: false,
            mamba_slot_mapping: None,
            sequence_ids: Some(vec![0]),
            decode_ptrs: Some(owl_engine::models::layers::vendor::DecodePtrs {
                slots: owl_iface::DevBuf::device_ptr(&slots) as *const i32,
                kv_lens: owl_iface::DevBuf::device_ptr(&kv_lens) as *const i32,
            }),
        };
        let frontier_dyn = owl_nn::DynTensor::from_u32(&frontier);
        model
            .forward(&frontier_dyn, &positions_dyn, Some(&kv_caches), &meta, false)
            .unwrap();
        dev.ctx().synchronize().unwrap();
    }

    // ---- 捕获(档位 [1,2])----
    let adapter_mut = adapter;
    let (plan, outcome) = GraphPlan::capture(
        &dev,
        vec![1, 2],
        2,
        vocab,
        GraphAllowance { decode_bytes: 64 << 20, verify_bytes: 0 },
        |_bs| 4 << 20,
        adapter_mut.clone(),
        owl_cuda::ffi::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
    )
    .expect("捕获");
    match outcome {
        owl_engine::graphplan::CaptureOutcome::Captured { profiles, .. } => {
            assert_eq!(profiles, vec![1, 2], "两档全定影");
        }
        _ => panic!("冒烟不应降级"),
    }

    // ---- 装填 + replay,判别三连 ----
    let load = |plan: &GraphPlan, adapter: &DecodeGraphAdapter, step: usize, tok: &[u32]| {
        // host → bindings(EagerOnly H2D;R1 装填口)
        let b = plan.bindings();
        b.write_frontier_from_host(&dev, tok).unwrap();
        b.write_positions_from_host(&dev, &[step as u32, step as u32]).unwrap();
        b.write_slot_mapping_from_host(&dev, &[step as u32 - 1, step as u32]).unwrap();
        b.write_kv_lens_from_host(&dev, &[step as u32, step as u32 + 1]).unwrap();
        // adapter 装填(bindings DynTensor 克隆 = keepalive)
        adapter.load_inputs(
            b.frontier().clone(),
            b.positions().clone(),
            b.slot_mapping().clone(),
            b.kv_lens().clone(),
        );
    };

    // replay(1):step 0(先读基线:logits_out 清零)
    load(&plan, &adapter_mut, 1, &[11, 22]);
    let lp = plan.bindings().logits_out().device_ptr() as *mut f32;
    // 设备地址 host 不可直解引用:清零走 H2D memcpy(零块)
    {
        let zeros = vec![0.0f32; 2 * vocab];
        use owl_cuda::ffi::sys;
        dev.ctx().bind_to_thread().unwrap();
        unsafe {
            sys::cuMemcpyHtoD_v2(
                lp as sys::CUdeviceptr,
                zeros.as_ptr() as *const std::ffi::c_void,
                zeros.len() * 4,
            )
            .result()
            .unwrap();
        }
    }
    plan.replay(1).unwrap();
    dev.ctx().synchronize().unwrap();
    let logits1 = dtoh_f32(&dev, lp, 2 * vocab);
    assert!(logits1.iter().any(|&v| v != 0.0), "判别①: replay 后 logits 非全零(图执行)");

    // replay(1) 第二次:确定性
    load(&plan, &adapter_mut, 1, &[11, 22]);
    plan.replay(1).unwrap();
    dev.ctx().synchronize().unwrap();
    let logits2 = dtoh_f32(&dev, lp, 2 * vocab);
    assert_eq!(
        logits1.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
        logits2.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
        "判别②: 同输入两次 replay bit 级一致"
    );

    // replay(2):换 frontier → 输出不同(数据流过)
    load(&plan, &adapter_mut, 2, &[77, 88]);
    plan.replay(2).unwrap();
    dev.ctx().synchronize().unwrap();
    let logits3 = dtoh_f32(&dev, lp, 2 * vocab);
    assert_ne!(
        logits1.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
        logits3.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
        "判别③: 换输入输出不同(数据流经全链)"
    );
}

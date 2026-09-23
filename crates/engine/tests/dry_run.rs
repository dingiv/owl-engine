
use owl_cuda::CudaDevice;
use owl_engine::config::Config;
use owl_engine::models::dry_kernels::DryKernels;
use owl_engine::models::layers::distributed::Comm;
use owl_engine::models::layers::{ctx_scope, VarBuilderX};
use owl_nn::cublas::NnBlas;
use owl_engine::models::qwen3_5::InputMetadata;
use owl_engine::models::qwen3_5::Qwen3_5ForCausalLM;
use owl_engine::session::{InputSlot, OutputSlot, Session};
use owl_iface::{Device as _, PoolConfig, PoolKind};
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

/// 假权重 + Session 编排面:空跑判别三连
#[test]
fn session_fake_weights_plan_step_discriminants() {
    let dev = CudaDevice::new(owl_cuda::test_device_ordinal(), owl_cuda::TEST_POOL_BYTES).expect("需要 CUDA 设备");
    let config = tiny_config();
    let vocab = config.vocab_size.unwrap();
    let layers = config.num_hidden_layers;
    let hkv = config.num_key_value_heads;
    let hd = config.head_dim.unwrap();

    // 池:S6 激活 scratch + P 阶段权重(ctx_scope 桥 + from_fake 装载共用)
    let scratch = Arc::new(
        dev.create_pool(PoolConfig {
            name: format!("sess-scratch-{}", std::process::id()),
            kind: PoolKind::Scratch,
            bytes: 64 << 20,
        })
        .unwrap(),
    );
    let wpool = Arc::new(
        dev.create_pool(PoolConfig {
            name: format!("sess-weights-{}", std::process::id()),
            kind: PoolKind::Weights,
            bytes: 256 << 20,
        })
        .unwrap(),
    );

    // ctx_scope 桥安装(层 candle 形态签名 → TLS ctx 消费)
    let ops = owl_nn::OpsCtx::new(&dev).unwrap();
    let blas = NnBlas::new(&dev).unwrap();
    let dry = DryKernels::new(dev.ctx()).unwrap();
    ctx_scope::install(ops, blas, dry, scratch.clone(), wpool.clone(), &dev);

    // 假权重模型
    let vb = VarBuilderX::from_fake(wpool.clone(), &dev).unwrap();
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

    // KV cache:flat [slots_cap, Hkv*D](slot 直排;naive 核布局)
    let slots_cap = 8usize;
    let kv_dim = hkv * hd;
    let kv_caches: Vec<(
        owl_nn::DynTensor<owl_cuda::CudaDevice>,
        owl_nn::DynTensor<owl_cuda::CudaDevice>,
    )> = (0..layers)
        .map(|_| {
            let k = owl_nn::TensorPoolOps::from_vec_tensor(
                wpool.as_ref(),
                &[slots_cap, kv_dim],
                vec![0f32; slots_cap * kv_dim],
            )
            .unwrap();
            let v = owl_nn::TensorPoolOps::from_vec_tensor(
                wpool.as_ref(),
                &[slots_cap, kv_dim],
                vec![0f32; slots_cap * kv_dim],
            )
            .unwrap();
            (owl_nn::DynTensor::from_f32(&k), owl_nn::DynTensor::from_f32(&v))
        })
        .collect();

    // ---- 编排闭包:唯一一份业务逻辑(eager/捕获/回放共享)----
    let forward = move |sc: &owl_engine::session::StepCtx| -> Result<(), owl_engine::Error> {
        let _push = ctx_scope::push(sc.ctx()); // 执行凭证 → 层桥
        let ids = sc.input("frontier")?;
        let pos = sc.input("positions")?;
        let _slots = sc.input("slot_mapping")?;
        let kvlens = sc.input("kv_lens")?;
        let meta = InputMetadata {
            seqlens: Some(vec![1; sc.bs()]), // xinfer decode 语义:每序列 1 token
            is_prefill: false,
            is_mtp_verify: false,
            mamba_slot_mapping: None,
            sequence_ids: Some((0..sc.bs()).collect()),
            decode_ptrs: Some(owl_engine::models::layers::vendor::DecodePtrs {
                slots: ids.device_ptr() as *const i32,
                kv_lens: kvlens.device_ptr() as *const i32,
            }),
        };
        let logits = model.forward(ids, pos, Some(&kv_caches), &meta, false)?;
        sc.output("logits", &logits)
    };

    // ---- Session::plan(warmup + 逐档捕获 + A1.4 预检,全在内部)----
    let (plan, outcome) = Session::plan(
        &dev,
        owl_engine::session::SessionDesc {
            profiles: vec![1, 2],
            inputs: vec![
                InputSlot::u32("frontier", 4).init(vec![11, 0]),
                InputSlot::u32("positions", 4).init(vec![0, 1]),
                InputSlot::u32("slot_mapping", 4).init(vec![0, 1]),
                InputSlot::u32("kv_lens", 4).init(vec![1, 2]),
            ],
            outputs: vec![OutputSlot::f32("logits", &[4, vocab])],
            scratch: Some(scratch),
            per_profile_bytes: |_bs| 4 << 20,
        },
        forward,
    )
    .expect("Session::plan");

    assert!(matches!(
        outcome,
        owl_engine::session::PlanOutcome::Captured { .. }
    ), "冒烟不应降级");
    assert_eq!(plan.profiles(), &[1, 2], "两档全定影");
    assert!(plan.is_captured());

    // ---- 判别①:replay 后 logits 非全零(图执行)----
    plan.write("frontier", &[11]).unwrap();
    plan.write("positions", &[0]).unwrap();
    plan.write("slot_mapping", &[0]).unwrap();
    plan.write("kv_lens", &[1]).unwrap();
    plan.step(&[]).unwrap(); // replay 最小 ≥1 档(输入已经 write 装填)
    dev.ctx().synchronize().unwrap();
    let lp = plan.output_ptr("logits").unwrap() as *mut f32;
    let logits1 = dtoh_f32(&dev, lp, 2 * vocab);
    assert!(
        logits1.iter().any(|&v| v != 0.0),
        "判别①失败:replay 后 logits 全零(图未执行或写错槽)"
    );

    // ---- 判别②:同输入两次 replay bit 级一致(确定性)----
    plan.step(&[]).unwrap();
    dev.ctx().synchronize().unwrap();
    let logits2 = dtoh_f32(&dev, lp, 2 * vocab);
    assert_eq!(
        logits1.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
        logits2.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
        "判别②失败:同输入 replay 不确定"
    );

    // ---- 判别③:换输入输出变化(数据流经全链)----
    plan.write("frontier", &[42]).unwrap();
    plan.step(&[]).unwrap();
    dev.ctx().synchronize().unwrap();
    let logits3 = dtoh_f32(&dev, lp, 2 * vocab);
    assert_ne!(
        logits1.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
        logits3.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
        "判别③失败:换输入输出未变(数据未流过)"
    );
}

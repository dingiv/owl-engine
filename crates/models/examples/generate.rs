//! M-e 端到端(2026-09-26):Qwen3.5-0.8B **文本生成跑通**。
//!
//! 全声明式形态:分词器与生成循环都不在本文件 ——
//! - 分词器:ModelSpec.tokenizer 声明(事实)→ load_tokenizer 装配
//! - 生成:GenSpec 声明 → interpreters::eval_generate(步循环解释器)
//! 本样例只剩:装配声明 → 调解释器 → 解码打印。
//!
//! 性能注:每 token = 整模单树 ~500 次 actor 往返,0.1s 级/token 是 host
//! 轮询主导,非 GPU 瓶颈 —— 提速归 Session/图捕获(M-f)。
//!
//! 用法:
//! ```text
//! OWL_TEST_DEVICE=3 cargo run -q -p owl-models --example generate -- "你好" 24
//! ```

use std::path::PathBuf;
use std::time::Instant;

use owl_models::contract::Dtype;
use owl_models::interpreters::{eval_generate, eval_load, GenSpec, Sampling};
use owl_models::layers::gdn::GdnBuffers;
use owl_models::layers::rope::Rope;
use owl_models::module::KvBuffers;
use owl_models::specs::{load_0_8b, load_tokenizer};
use owl_models::TensorOps;

/// 槽位上限(prompt + 生成的总 token 预算;decode 直排 KV 按槽连续写)
const SLOTS: usize = 64;

fn f32b(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

async fn zero_block(gpu: &mut owl_cuda::GpuClient, n: usize, shape: Vec<usize>) -> TensorOps {
    let b = owl_models::interpreters::eval_ops(
        TensorOps::from_host(Dtype::F32, vec![n], &f32b(&vec![0.0; n])).step(),
        gpu,
    )
    .await
    .expect("zero block");
    TensorOps::of_block(b.id, Dtype::F32, shape)
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let prompt = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "请用一句话介绍你自己。".into());
    let max_new: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(24);
    let ordinal: usize = std::env::var("OWL_TEST_DEVICE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/Qwen3.5-0.8B");
    let tok = load_tokenizer(&dir).expect("tokenizer");
    let ids = tok.encode(&tok.chat_wrap(&prompt));
    println!("[gen] prompt({} tok): {prompt}", ids.len());

    let mut gpu = owl_cuda::GpuClient::spawn(owl_cuda::DeviceSelector::Ordinal(ordinal))
        .expect("gpu server boot(OWL_TEST_DEVICE 指向空闲卡)");
    let t0 = Instant::now();
    let model = load_0_8b(&dir, &mut gpu).await.expect("load_0_8b");
    let rp = Rope::new(262_144, 256, 64, 10_000_000.0).expect("rope");
    eval_load(&rp, &mut gpu, &rp.tables(), &Default::default())
        .await
        .expect("rope 表装载");

    // 常驻状态:6 个 full 层 KV + 18 个 GDN 状态(0 号槽直排;零初始化)
    let mut kvs = Vec::new();
    let mut gdns = Vec::new();
    for _ in 0..6 {
        kvs.push(KvBuffers {
            k_cache: zero_block(&mut gpu, SLOTS * 2 * 256, vec![SLOTS, 2, 256]).await,
            v_cache: zero_block(&mut gpu, SLOTS * 2 * 256, vec![SLOTS, 2, 256]).await,
            slots: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[0.0])),
            kv_lens: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[1.0])),
        });
    }
    for _ in 0..18 {
        gdns.push(GdnBuffers {
            conv_q: zero_block(&mut gpu, SLOTS * 2048 * 3, vec![SLOTS, 2048, 3]).await,
            conv_k: zero_block(&mut gpu, SLOTS * 2048 * 3, vec![SLOTS, 2048, 3]).await,
            conv_v: zero_block(&mut gpu, SLOTS * 2048 * 3, vec![SLOTS, 2048, 3]).await,
            rec: zero_block(&mut gpu, SLOTS * 16 * 128 * 128, vec![SLOTS, 16, 128, 128]).await,
            slots: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[0.0])),
        });
    }
    println!("[gen] 装载完成 {:.1}s", t0.elapsed().as_secs_f32());

    // ── 生成 = 一次解释器调用(循环在 interpreters/generate.rs)──
    let spec = GenSpec {
        max_new,
        eos_ids: tok.eos_ids().to_vec(),
        sampling: Sampling::Greedy,
    };
    let t_gen = Instant::now();
    let out_ids = eval_generate(&model, &mut gpu, &rp, &kvs, &gdns, &ids, &spec)
        .await
        .expect("eval_generate");
    let dt = t_gen.elapsed().as_secs_f32();
    let text = tok.decode(&out_ids);
    println!("{text}");
    println!(
        "[gen] {} tok / {dt:.1}s = {:.2} s/tok(host 轮询主导;提速归 M-f 图)",
        out_ids.len(),
        dt / out_ids.len().max(1) as f32
    );
    gpu.close().await.ok();
}

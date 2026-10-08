//! attention port 家族核测(K0 起;vendor attention.rs rev c0f19f2 复用面)。
//!
//! 布局契约(vLLM classic;K1/K2 paged_attention 同款):
//! - key_cache   `[num_blocks, Hkv, D/x, block_size, x]`(x = 16/sizeof(f16) = 8)
//! - value_cache `[num_blocks, Hkv, D, block_size]`
//! - slot_mapping = **物理槽**(block_idx·block_size + offset;单序列连续分配
//!   时 block_table 恒等 ⇒ 物理槽 = 逻辑 pos);负 = padding 跳写;f32 过线(契约 5);
//! - 发射 grid = (num_tokens,1,1) **显式**(核内 blockIdx.x = token_idx;
//!   哨兵自动 1D 会越界读 slot_mapping,禁用 —— port 头注第 3 条)。

use crate::contract::DeviceClient as _;
use crate::ops::SemanticKernel;
use crate::kernel::kernel_with;
use crate::testkit::{f32b, gpu_client, gpu_enabled};
use crate::tensor::Dtype;
use crate::TensorOps;

/// 物理槽 → cache 下标(vLLM classic 布局 host 参照;与核内逐式同源)
fn host_cache_idx(slot: i64, head: usize, d: usize, hkv: usize, hd: usize, p: usize, x: usize) -> (usize, usize) {
    let block_idx = (slot / p as i64) as usize;
    let block_offset = (slot % p as i64) as usize;
    let key = block_idx * hkv * (hd / x) * p * x
        + head * (hd / x) * p * x
        + (d / x) * p * x
        + block_offset * x
        + (d % x);
    let value = block_idx * hkv * hd * p + head * hd * p + d * p + block_offset;
    (key, value)
}

/// K0 验收:reshape_and_cache f16 —— 恒等分页 + 跳块物理槽 + padding 跳写,
/// 写后读回与 host 参照**逐位一致**(纯拷贝核);空洞槽保持零。
#[tokio::test]
async fn gpu_reshape_and_cache_f16_matches_host() {
    if !gpu_enabled() {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    }
    // 形状:hkv=2, hd=8, block_size=4, num_blocks=3;T=10(含 padding)
    let (hkv, hd, p, nb) = (2usize, 8usize, 4usize, 3usize);
    let x = 16usize / 2; // f16 → 8
    let t_len = 10usize;
    let halfb = |v: &[f32]| -> Vec<u8> {
        v.iter().flat_map(|f| half::f16::from_f32(*f).to_le_bytes()).collect()
    };

    // 输入(确定性;过 f16 量化)
    let key: Vec<f32> = (0..t_len * hkv * hd)
        .map(|i| half::f16::from_f32(((i as f32) * 0.37 - 1.0).sin()).to_f32())
        .collect();
    let value: Vec<f32> = (0..t_len * hkv * hd)
        .map(|i| half::f16::from_f32(((i as f32) * 0.23 + 0.5).cos()).to_f32())
        .collect();
    // 物理槽:token0-3 → 块0(slot0-3);token4 → padding(-1);token5-9 →
    // 块2(slot8-11)+ 块1(slot5)——物理槽 ≠ 逻辑序,块1 留空洞
    let slots: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, -1.0, 8.0, 9.0, 10.0, 11.0, 5.0];

    let mut gpu = gpu_client().await;

    // 设备面:cache 池(zeros 物化取块句柄;写后按块读回)
    let kc_b = crate::interpreters::eval_ops(
        TensorOps::zeros(Dtype::F16, vec![nb, hkv, hd / x, p, x]).step(),
        &mut gpu,
    )
    .await
    .expect("kc 池");
    let vc_b = crate::interpreters::eval_ops(
        TensorOps::zeros(Dtype::F16, vec![nb, hkv, hd, p]).step(),
        &mut gpu,
    )
    .await
    .expect("vc 池");

    // 发射(显式 grid = T;契约 5 槽表 f32)
    let decl = TensorOps::call(SemanticKernel::K0Write).aux(&[t_len])
    .arg(&TensorOps::from_host(Dtype::F16, vec![t_len, hkv, hd], &halfb(&key)))
    .arg(&TensorOps::from_host(Dtype::F16, vec![t_len, hkv, hd], &halfb(&value)))
    .arg(&TensorOps::of_block(kc_b.id, Dtype::F16, vec![nb, hkv, hd / x, p, x]))
    .arg(&TensorOps::of_block(vc_b.id, Dtype::F16, vec![nb, hkv, hd, p]))
    .arg(&TensorOps::from_host(Dtype::F32, vec![t_len], &f32b(&slots)))
    .arg_i32(hkv as i32 * hd as i32) // key_stride = 行密排 = H·D
    .arg_i32(hkv as i32 * hd as i32)
    .arg_i32(hkv as i32)
    .arg_i32(hd as i32)
    .arg_i32(p as i32)
    .arg_i32(x as i32)
    // 末位 T = 哑输出(契约 4;核不写,cache 双池经 ins 就地写);
    // grid 已显式 = (T,1,1),out 元素数无意义
    .with_shape(Dtype::F16, vec![1]);
    crate::interpreters::eval_ops(decl.step(), &mut gpu).await.expect("eval");

    // 读回双池
    let mut read = async |b: &crate::contract::Bytes, shape: Vec<usize>| -> Vec<f32> {
        let leaf = TensorOps::of_block(b.id, Dtype::F16, shape);
        crate::testkit::harvest_f16(&mut gpu, &leaf).await
    };
    let got_k = read(&kc_b, vec![nb, hkv, hd / x, p, x]).await;
    let got_v = read(&vc_b, vec![nb, hkv, hd, p]).await;

    // host 参照:同式散写(padding 跳写;空洞保持 0)
    let total_k = nb * hkv * (hd / x) * p * x;
    let total_v = nb * hkv * hd * p;
    let (mut ref_k, mut ref_v) = (vec![0f32; total_k], vec![0f32; total_v]);
    for (tok, &slot) in slots.iter().enumerate() {
        if slot < 0.0 {
            continue;
        }
        let slot = slot as i64;
        for h in 0..hkv {
            for d in 0..hd {
                let (ki, vi) = host_cache_idx(slot, h, d, hkv, hd, p, x);
                ref_k[ki] = key[tok * hkv * hd + h * hd + d];
                ref_v[vi] = value[tok * hkv * hd + h * hd + d];
            }
        }
    }

    // 逐位对拍(纯拷贝核;量化后的 host 参照同一位型)
    let q = |v: f32| half::f16::from_f32(v).to_f32();
    for i in 0..total_k {
        assert_eq!(got_k[i], q(ref_k[i]), "key[{i}] 位型不符");
    }
    for i in 0..total_v {
        assert_eq!(got_v[i], q(ref_v[i]), "value[{i}] 位型不符");
    }

    // 空洞槽断言:slot4(无 token 写)保持零
    let (ki, vi) = host_cache_idx(4, 1, 3, hkv, hd, p, x);
    assert_eq!(got_k[ki], 0.0, "空洞槽 key 应保持零");
    assert_eq!(got_v[vi], 0.0, "空洞槽 value 应保持零");
    gpu.close().await.expect("关机");
}

// ============================================================================
// K1/K2 真机 smoke(nvrtc 懒编译在首次 launch;classic 布局 + 数值双关)
// ============================================================================

/// 场景装配:单序列 T=24 跨 2 块(恒等块表),K/V 经 K0 核写入池
/// (布局自洽),q = token0 key 行;返回(池句柄, 分页表, 序长表)。
async fn setup_paged_fixture(
    gpu: &mut owl_cuda::GpuClient,
    hq: usize, hkv: usize, hd: usize, p: usize, nb: usize, t_ctx: usize,
) -> (crate::contract::Bytes, crate::contract::Bytes, Vec<f32>, Vec<f32>, Vec<f32>) {
    let x = 8usize;
    let halfb = |v: &[f32]| -> Vec<u8> {
        v.iter().flat_map(|f| half::f16::from_f32(*f).to_le_bytes()).collect()
    };
    let q8 = |v: f32| half::f16::from_f32(v).to_f32();
    let pat = |t: usize, h: usize, d: usize, salt: f32| {
        q8(((t * 7 + h * 31 + d * 13) as f32 * 0.11 + salt).sin() * 0.9)
    };
    let key: Vec<f32> = (0..t_ctx * hkv * hd)
        .map(|i| {
            let (t, r) = (i / (hkv * hd), i % (hkv * hd));
            pat(t, r / hd, r % hd, 0.0)
        })
        .collect();
    let value: Vec<f32> = (0..t_ctx * hkv * hd)
        .map(|i| {
            let (t, r) = (i / (hkv * hd), i % (hkv * hd));
            pat(t, r / hd, r % hd, 3.7)
        })
        .collect();

    let kc_b = crate::interpreters::eval_ops(
        TensorOps::zeros(Dtype::F16, vec![nb, hkv, hd / x, p, x]).step(), gpu)
        .await.expect("kc 池");
    let vc_b = crate::interpreters::eval_ops(
        TensorOps::zeros(Dtype::F16, vec![nb, hkv, hd, p]).step(), gpu)
        .await.expect("vc 池");
    let slots: Vec<f32> = (0..t_ctx).map(|t| t as f32).collect();
    let wr = TensorOps::call(SemanticKernel::K0Write).aux(&[t_ctx])
        .arg(&TensorOps::from_host(Dtype::F16, vec![t_ctx, hkv, hd], &halfb(&key)))
        .arg(&TensorOps::from_host(Dtype::F16, vec![t_ctx, hkv, hd], &halfb(&value)))
        .arg(&TensorOps::of_block(kc_b.id, Dtype::F16, vec![nb, hkv, hd / x, p, x]))
        .arg(&TensorOps::of_block(vc_b.id, Dtype::F16, vec![nb, hkv, hd, p]))
        .arg(&TensorOps::from_host(Dtype::F32, vec![t_ctx], &f32b(&slots)))
        .arg_i32(hkv as i32 * hd as i32)
        .arg_i32(hkv as i32 * hd as i32)
        .arg_i32(hkv as i32)
        .arg_i32(hd as i32)
        .arg_i32(p as i32)
        .arg_i32(x as i32)
        .with_shape(Dtype::F16, vec![1]);
    crate::interpreters::eval_ops(wr.step(), gpu).await.expect("K0 写池");

    let tables = vec![0.0f32, 1.0]; // 恒等块表 [1, 2]
    (kc_b, vc_b, key, value, tables)
}

/// host 参照:token-major 语义注意力(f16 量化值;分页索引由核验)
fn host_attn_ref(
    q_row: &[f32], key: &[f32], value: &[f32], h: usize, kvh_map: &[usize],
    hkv: usize, hd: usize, t_ctx: usize, scale: f32,
) -> Vec<f32> {
    let q8 = |v: f32| half::f16::from_f32(v).to_f32();
    let kvh = kvh_map[h];
    let (mut scores, mut maxs) = (vec![0f32; t_ctx], f32::NEG_INFINITY);
    for s in 0..t_ctx {
        let mut acc = 0f32;
        for d in 0..hd {
            acc += q8(q_row[d]) * q8(key[s * hkv * hd + kvh * hd + d]);
        }
        scores[s] = acc * scale;
        maxs = maxs.max(scores[s]);
    }
    let (mut denom, mut out) = (0f32, vec![0f32; hd]);
    for s in 0..t_ctx {
        scores[s] = (scores[s] - maxs).exp();
        denom += scores[s];
    }
    for d in 0..hd {
        let mut acc = 0f32;
        for s in 0..t_ctx {
            acc += scores[s] / denom * q8(value[s * hkv * hd + kvh * hd + d]);
        }
        out[d] = acc;
    }
    out
}

/// chunked prefill hd256 档 smoke(qwen3.5-0.8B 真实头维;GQA 4:1)
#[tokio::test]
async fn gpu_prefill_paged_attn_f16_hd256_smoke() {
    let t_ctx = 5usize; // 诊断二分:模型内 T≥5 挂
    return gpu_prefill_paged_attn_f16_hd256_impl(t_ctx).await;
}

#[tokio::test]
async fn gpu_prefill_paged_attn_f16_hd256_smoke8() {
    let t_ctx = 8usize;
    return gpu_prefill_paged_attn_f16_hd256_impl(t_ctx).await;
}

async fn gpu_prefill_paged_attn_f16_hd256_impl(t_ctx: usize) {
    if !gpu_enabled() {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    }
    let (hq, hkv, hd, p, nb) = (8usize, 2usize, 256usize, 32usize, 1usize);
    let kvh_map: Vec<usize> = (0..hq).map(|h| h / (hq / hkv)).collect();
    let mut gpu = gpu_client().await;
    let (kc_b, vc_b, key, value, _tables) =
        setup_paged_fixture(&mut gpu, hq, hkv, hd, p, nb, t_ctx).await;

    let mut q: Vec<f32> = Vec::with_capacity(t_ctx * hq * hd);
    for t in 0..t_ctx {
        for _h in 0..hq {
            q.extend(key[t * hkv * hd..t * hkv * hd + hd].iter().cloned());
        }
    }
    let scale = 1.0 / (hd as f32).sqrt();
    let smem = (64 + 2 * hd * p * 2) as u32;
    let q8 = |v: f32| half::f16::from_f32(v).to_f32();
    let qb = |v: &[f32]| v.iter().flat_map(|f| half::f16::from_f32(*f).to_le_bytes()).collect::<Vec<u8>>();

    let decl = TensorOps::call(SemanticKernel::PagedPrefill).aux(&[hd, hkv, hq, t_ctx])
    .arg(&TensorOps::from_host(Dtype::F16, vec![t_ctx, hq, hd], &qb(&q)))
    .arg(&TensorOps::of_block(kc_b.id, Dtype::F16, vec![nb, hkv, hd / 8, p, 8]))
    .arg(&TensorOps::of_block(vc_b.id, Dtype::F16, vec![nb, hkv, hd, p]))
    .arg(&TensorOps::from_host(Dtype::F32, vec![1, 1], &f32b(&[0.0])))
    .arg(&TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[t_ctx as f32])))
    .arg(&TensorOps::from_host(Dtype::F32, vec![2], &f32b(&[0.0, t_ctx as f32])))
    .arg(&TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[0.0])))
    .arg(&TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[0.0])))
    .arg_i32(hkv as i32)
    .arg_f32(scale)
    .arg_i32(1)
    .arg_i32(1)
    .arg_i32(hq as i32)
    .arg_i32(t_ctx as i32)
    .arg_f32(1.0)
    .arg_i32(hq as i32 * hd as i32)
    .arg_i32(-1)
    .arg_i32(nb as i32)
    .arg_i32(hkv as i32 * hd as i32 * p as i32)
    .arg_i32(hd as i32 * p as i32)
    .arg_i32(0)
    .arg_i32(0)
    .with_shape(Dtype::F16, vec![t_ctx, hq, hd]);
    let got = crate::testkit::harvest_f16(&mut gpu, &decl).await;

    for (t, h) in (0..t_ctx).flat_map(|t| (0..hq).map(move |h| (t, h))) {
        let kvh = kvh_map[h];
        let q_row = &q[t * hq * hd + h * hd..t * hq * hd + h * hd + hd];
        let (mut scores, mut maxs) = (vec![0f32; t + 1], f32::NEG_INFINITY);
        for s in 0..=t {
            let mut acc = 0f32;
            for d in 0..hd {
                acc += q8(q_row[d]) * q8(key[s * hkv * hd + kvh * hd + d]);
            }
            scores[s] = acc * scale;
            maxs = maxs.max(scores[s]);
        }
        let (mut denom, mut want) = (0f32, vec![0f32; hd]);
        for s in 0..=t {
            scores[s] = (scores[s] - maxs).exp();
            denom += scores[s];
        }
        for d in 0..hd {
            let mut acc = 0f32;
            for s in 0..=t {
                acc += scores[s] / denom * q8(value[s * hkv * hd + kvh * hd + d]);
            }
            want[d] = acc;
        }
        let got_row = &got[t * hq * hd + h * hd..t * hq * hd + h * hd + hd];
        for d in 0..hd {
            assert!(
                (got_row[d] - want[d]).abs() < 2e-2,
                "prefill256[{t},{h},{d}] got {} want {}",
                got_row[d],
                want[d]
            );
        }
    }
    gpu.close().await.expect("关机");
}

/// K1:v1 核 nvrtc 编译 + 布局 + 数值(2e-2 rel,f16 链现行容差律)
#[tokio::test]
async fn gpu_paged_attention_v1_f16_smoke() {
    if !gpu_enabled() {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    }
    let (hq, hkv, hd, p, nb, t_ctx) = (2usize, 1usize, 128usize, 32usize, 1usize, 24usize);
    let kvh_map: Vec<usize> = (0..hq).map(|h| h / (hq / hkv)).collect();
    let mut gpu = gpu_client().await;
    let (kc_b, vc_b, key, value, tables) =
        setup_paged_fixture(&mut gpu, hq, hkv, hd, p, nb, t_ctx).await;

    // q = token0 key 行(hq 行同值,覆盖 GQA 两条映射)
    let q: Vec<f32> = key[..hd].iter().chain(key[..hd].iter()).cloned().collect();
    let scale = 1.0 / (hd as f32).sqrt();
    let shared = ((t_ctx + p - 1) / p * p * 4).max((4 / 2) * hd * 4) as u32; // NUM_WARPS=4

    let decl = TensorOps::call(SemanticKernel::PagedDecode).aux(&[hd, hq, hkv, nb])
        .arg(&TensorOps::from_host(Dtype::F16, vec![1, hq, hd],
            &q.iter().flat_map(|f| half::f16::from_f32(*f).to_le_bytes()).collect::<Vec<u8>>()))
        .arg(&TensorOps::of_block(kc_b.id, Dtype::F16, vec![nb, hkv, hd / 8, p, 8]))
        .arg(&TensorOps::of_block(vc_b.id, Dtype::F16, vec![nb, hkv, hd, p]))
        .arg(&TensorOps::from_host(Dtype::F32, vec![1, 2], &f32b(&tables)))
        .arg(&TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[t_ctx as f32])))
        .arg(&TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[0.0]))) // alibi 哑块(旗标 0 不解引用)
        .arg_i32(hkv as i32)
        .arg_f32(scale)
        .arg_i32(nb as i32)
        .arg_i32(hq as i32 * hd as i32)              // q_stride
        .arg_i32(hkv as i32 * hd as i32 * p as i32)  // kv_block_stride
        .arg_i32(hd as i32 * p as i32)               // kv_head_stride
        .arg_f32(1.0)                                 // softscapping 直通
        .arg_i32(-1)                                  // sliding_window 关
        .arg_i32(0)                                   // use_alibi 关
        .with_shape(Dtype::F16, vec![1, hq, hd]);     // 末位 T = 新分配输出

    // nvrtc 懒编译在此行发生:编译失败 = 整个扁平化路线回炉
    let got = crate::testkit::harvest_f16(&mut gpu, &decl).await;

    let mut ref_out = Vec::new();
    for h in 0..hq {
        ref_out.extend(host_attn_ref(&q[h * hd..(h + 1) * hd], &key, &value, h, &kvh_map, hkv, hd, t_ctx, scale));
    }
    crate::testkit::assert_close(&got, &ref_out, 2e-2, "paged v1 f16");
    gpu.close().await.expect("关机");
}

/// v1 hd256 档 smoke(qwen3.5-0.8B 真实头维)
#[tokio::test]
async fn gpu_paged_attention_v1_f16_hd256_smoke() {
    if !gpu_enabled() {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    }
    let (hq, hkv, hd, p, nb, t_ctx) = (8usize, 2usize, 256usize, 32usize, 1usize, 8usize);
    let kvh_map: Vec<usize> = (0..hq).map(|h| h / (hq / hkv)).collect();
    let mut gpu = gpu_client().await;
    let (kc_b, vc_b, key, value, tables) =
        setup_paged_fixture(&mut gpu, hq, hkv, hd, p, nb, t_ctx).await;

    let mut q: Vec<f32> = Vec::with_capacity(hq * hd);
    for h in 0..hq {
        let kvh = kvh_map[h];
        q.extend(key[kvh * hd..kvh * hd + hd].iter().cloned());
    }
    let scale = 1.0 / (hd as f32).sqrt();
    let shared = ((t_ctx + p - 1) / p * p * 4).max((4 / 2) * hd * 4) as u32;
    let qb = |v: &[f32]| v.iter().flat_map(|f| half::f16::from_f32(*f).to_le_bytes()).collect::<Vec<u8>>();

    let decl = TensorOps::call(SemanticKernel::PagedDecode).aux(&[hd, hq, hkv, nb])
    .arg(&TensorOps::from_host(Dtype::F16, vec![1, hq, hd], &qb(&q)))
    .arg(&TensorOps::of_block(kc_b.id, Dtype::F16, vec![nb, hkv, hd / 8, p, 8]))
    .arg(&TensorOps::of_block(vc_b.id, Dtype::F16, vec![nb, hkv, hd, p]))
    .arg(&TensorOps::from_host(Dtype::F32, vec![1, 1], &f32b(&[0.0])))
    .arg(&TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[t_ctx as f32])))
    .arg(&TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[0.0])))
    .arg_i32(hkv as i32)
    .arg_f32(scale)
    .arg_i32(nb as i32)
    .arg_i32(hq as i32 * hd as i32)
    .arg_i32(hkv as i32 * hd as i32 * p as i32)
    .arg_i32(hd as i32 * p as i32)
    .arg_f32(1.0)
    .arg_i32(-1)
    .arg_i32(0)
    .with_shape(Dtype::F16, vec![1, hq, hd]);
    let got = crate::testkit::harvest_f16(&mut gpu, &decl).await;

    let mut ref_out = Vec::new();
    for h in 0..hq {
        let kvh = kvh_map[h];
        let q_row = &q[h * hd..h * hd + hd];
        let (mut scores, mut maxs) = (vec![0f32; t_ctx], f32::NEG_INFINITY);
        for s in 0..t_ctx {
            let mut acc = 0f32;
            for d in 0..hd {
                acc += half::f16::from_f32(q_row[d]).to_f32()
                    * half::f16::from_f32(key[s * hkv * hd + kvh * hd + d]).to_f32();
            }
            scores[s] = acc * scale;
            maxs = maxs.max(scores[s]);
        }
        let (mut denom, mut out) = (0f32, vec![0f32; hd]);
        for s in 0..t_ctx {
            scores[s] = (scores[s] - maxs).exp();
            denom += scores[s];
        }
        for d in 0..hd {
            let mut acc = 0f32;
            for s in 0..t_ctx {
                acc += scores[s] / denom * half::f16::from_f32(value[s * hkv * hd + kvh * hd + d]).to_f32();
            }
            out[d] = acc;
        }
        ref_out.extend(out);
    }
    crate::testkit::assert_close(&got, &ref_out, 2e-2, "paged v1 hd256");
    gpu.close().await.expect("关机");
}

/// K2:v2 分片(单 partition)+ reduce 链,与 v1 同参照
#[tokio::test]
async fn gpu_paged_attention_v2_reduce_f16_smoke() {
    if !gpu_enabled() {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    }
    let (hq, hkv, hd, p, nb, t_ctx) = (2usize, 1usize, 128usize, 32usize, 1usize, 24usize);
    let kvh_map: Vec<usize> = (0..hq).map(|h| h / (hq / hkv)).collect();
    let mut gpu = gpu_client().await;
    let (kc_b, vc_b, key, value, tables) =
        setup_paged_fixture(&mut gpu, hq, hkv, hd, p, nb, t_ctx).await;

    let q: Vec<f32> = key[..hd].iter().chain(key[..hd].iter()).cloned().collect();
    let scale = 1.0 / (hd as f32).sqrt();
    let shared = ((t_ctx + p - 1) / p * p * 4).max((4 / 2) * hd * 4) as u32;
    let qb = |v: &[f32]| v.iter().flat_map(|f| half::f16::from_f32(*f).to_le_bytes()).collect::<Vec<u8>>();

    // v2:exp_sums/max_logits 为 ins 就地写,tmp_out = 末位新分配
    let es_b = crate::interpreters::eval_ops(
        TensorOps::zeros(Dtype::F32, vec![1, hq, 1]).step(), &mut gpu).await.expect("es");
    let ml_b = crate::interpreters::eval_ops(
        TensorOps::zeros(Dtype::F32, vec![1, hq, 1]).step(), &mut gpu).await.expect("ml");
    let v2 = TensorOps::call(SemanticKernel::PagedDecodeV2).aux(&[hd, hq, hkv, nb, 1])
        .arg(&TensorOps::from_host(Dtype::F16, vec![1, hq, hd], &qb(&q)))
        .arg(&TensorOps::of_block(kc_b.id, Dtype::F16, vec![nb, hkv, hd / 8, p, 8]))
        .arg(&TensorOps::of_block(vc_b.id, Dtype::F16, vec![nb, hkv, hd, p]))
        .arg(&TensorOps::from_host(Dtype::F32, vec![1, 2], &f32b(&tables)))
        .arg(&TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[t_ctx as f32])))
        .arg(&TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[0.0])))
        .arg(&TensorOps::of_block(es_b.id, Dtype::F32, vec![1, hq, 1]))
        .arg(&TensorOps::of_block(ml_b.id, Dtype::F32, vec![1, hq, 1]))
        .arg_i32(hkv as i32)
        .arg_f32(scale)
        .arg_i32(nb as i32)
        .arg_i32(hq as i32 * hd as i32)
        .arg_i32(hkv as i32 * hd as i32 * p as i32)
        .arg_i32(hd as i32 * p as i32)
        .arg_f32(1.0)
        .arg_i32(-1)
        .arg_i32(0)
        .with_shape(Dtype::F16, vec![1, hq, hd]);
    let tmp = crate::interpreters::eval_ops(v2.step(), &mut gpu).await.expect("v2 eval");

    // reduce:分片归约 → 终出
    let rd = TensorOps::call(SemanticKernel::PagedV2Reduce).aux(&[hd, hq, 1])
        .arg(&TensorOps::of_block(es_b.id, Dtype::F32, vec![1, hq, 1]))
        .arg(&TensorOps::of_block(ml_b.id, Dtype::F32, vec![1, hq, 1]))
        .arg(&TensorOps::of_block(tmp.id, Dtype::F16, vec![1, hq, hd]))
        .arg(&TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[t_ctx as f32])))
        .arg_i32(1) // max_num_partitions
        .with_shape(Dtype::F16, vec![1, hq, hd]);
    let got = crate::testkit::harvest_f16(&mut gpu, &rd).await;

    let mut ref_out = Vec::new();
    for h in 0..hq {
        ref_out.extend(host_attn_ref(&q[h * hd..(h + 1) * hd], &key, &value, h, &kvh_map, hkv, hd, t_ctx, scale));
    }
    crate::testkit::assert_close(&got, &ref_out, 2e-2, "paged v2+reduce f16");
    gpu.close().await.expect("关机");
}

/// prefill 批核真机 smoke:chunked prefill 因果语义(查询 token t 看
/// [0,t])+ paged 池(BLOCK=32 特化,与 decode 的 16 不同);host 参照 =
/// 逐 token 因果注意力。
#[tokio::test]
async fn gpu_prefill_paged_attn_f16_smoke() {
    if !gpu_enabled() {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    }
    let (hq, hkv, hd, p, nb, t_ctx) = (2usize, 1usize, 128usize, 32usize, 1usize, 24usize);
    let kvh_map: Vec<usize> = (0..hq).map(|h| h / (hq / hkv)).collect();
    let mut gpu = gpu_client().await;
    let (kc_b, vc_b, key, value, _tables) =
        setup_paged_fixture(&mut gpu, hq, hkv, hd, p, nb, t_ctx).await;

    // 查询 = 各 token 自己的 key 行(prefill 真实形态:token 打分自己历史)
    let mut q: Vec<f32> = Vec::with_capacity(t_ctx * hq * hd);
    for t in 0..t_ctx {
        for _h in 0..hq {
            q.extend(key[t * hkv * hd..t * hkv * hd + hd].iter().cloned());
        }
    }
    let scale = 1.0 / (hd as f32).sqrt();
    let smem = (64 + 2 * hd * p * 2) as u32; // 头注契约
    let q8 = |v: f32| half::f16::from_f32(v).to_f32();
    let qb = |v: &[f32]| v.iter().flat_map(|f| half::f16::from_f32(*f).to_le_bytes()).collect::<Vec<u8>>();

    let decl = TensorOps::call(SemanticKernel::PagedPrefill).aux(&[hd, hkv, hq, t_ctx])
    .arg(&TensorOps::from_host(Dtype::F16, vec![t_ctx, hq, hd], &qb(&q)))
    .arg(&TensorOps::of_block(kc_b.id, Dtype::F16, vec![nb, hkv, hd / 8, p, 8]))
    .arg(&TensorOps::of_block(vc_b.id, Dtype::F16, vec![nb, hkv, hd, p]))
    .arg(&TensorOps::from_host(Dtype::F32, vec![1, 1], &f32b(&[0.0]))) // 恒等块表
    .arg(&TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[t_ctx as f32])))
    .arg(&TensorOps::from_host(Dtype::F32, vec![2], &f32b(&[0.0, t_ctx as f32]))) // query_start_len
    .arg(&TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[0.0]))) // alibi 哑
    .arg(&TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[0.0]))) // sinks 哑
    .arg_i32(hkv as i32)
    .arg_f32(scale)
    .arg_i32(1)                    // block_table_stride
    .arg_i32(1)                    // num_seqs
    .arg_i32(hq as i32)
    .arg_i32(t_ctx as i32)
    .arg_f32(1.0)                  // softscapping
    .arg_i32(hq as i32 * hd as i32) // o_stride_tokens
    .arg_i32(-1)                   // sliding_window 关
    .arg_i32(nb as i32)            // total_num_blocks
    .arg_i32(hkv as i32 * hd as i32 * p as i32) // kv_block_stride
    .arg_i32(hd as i32 * p as i32) // kv_head_stride
    .arg_i32(0)                    // use_alibi
    .arg_i32(0)                    // use_sinks
    .with_shape(Dtype::F16, vec![t_ctx, hq, hd]);
    let got = crate::testkit::harvest_f16(&mut gpu, &decl).await;

    // host 参照:逐 token 因果(token t 看 [0,t])
    for (t, h) in (0..t_ctx).flat_map(|t| (0..hq).map(move |h| (t, h))) {
        let kvh = kvh_map[h];
        let q_row = &q[t * hq * hd + h * hd..t * hq * hd + h * hd + hd];
        let (mut scores, mut maxs) = (vec![0f32; t + 1], f32::NEG_INFINITY);
        for s in 0..=t {
            let mut acc = 0f32;
            for d in 0..hd {
                acc += q8(q_row[d]) * q8(key[s * hkv * hd + kvh * hd + d]);
            }
            scores[s] = acc * scale;
            maxs = maxs.max(scores[s]);
        }
        let (mut denom, mut want) = (0f32, vec![0f32; hd]);
        for s in 0..=t {
            scores[s] = (scores[s] - maxs).exp();
            denom += scores[s];
        }
        for d in 0..hd {
            let mut acc = 0f32;
            for s in 0..=t {
                acc += scores[s] / denom * q8(value[s * hkv * hd + kvh * hd + d]);
            }
            want[d] = acc;
        }
        let got_row = &got[t * hq * hd + h * hd..t * hq * hd + h * hd + hd];
        for d in 0..hd {
            assert!(
                (got_row[d] - want[d]).abs() < 2e-2,
                "prefill[{t},{h},{d}] got {} want {}",
                got_row[d],
                want[d]
            );
        }
    }
    gpu.close().await.expect("关机");
}

/// 设备 argmax 对拍 host(E3 采样器哨兵):随机 + 平局 + 偏移三用例
#[tokio::test]
async fn gpu_argmax_f32idx_matches_host() {
    if !crate::testkit::gpu_enabled() {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    }
    use crate::ops::argmax_f32idx;
    let mut gpu = crate::testkit::gpu_client().await;

    // 伪随机 + 人造双峰平局(max 出现在 idx 1000 与 7777 同值 → 首见者胜)
    let n = 100_003usize;
    let mut vals: Vec<f32> = (0..n)
        .map(|i| ((i as f64 * 0.618_033_988_7).fract() * 2.0 - 1.0) as f32)
        .collect();
    vals[1000] = 0.987;
    vals[7777] = 0.987; // 平局:首见(1000)胜
    vals[99_999] = 0.5;
    let bytes: Vec<u8> = vals
        .iter()
        .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
        .collect();

    // host 参照按 f16 量化后比对(核读 f16;0.99998 与 0.99987 在 f16
    // 同为 1.0 —— 首见者胜,与 reshape 测试的量化参照同纪律)
    let q = |v: f32| half::f16::from_f32(v).to_f32();
    let host = |vals: &[f32], off: usize| -> usize {
        vals[off..]
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |a, (i, &v)| {
                if q(v) > a.1 { (i, q(v)) } else { a }
            })
            .0
            + off
    };

    // 用例 1:offset = 0
    let decl = argmax_f32idx(&TensorOps::from_host(Dtype::F16, vec![n], &bytes), n, 0);
    let b = crate::interpreters::eval_ops(decl.step(), &mut gpu).await.expect("eval");
    let mut buf = [0u8; 4];
    gpu.dtoh(&b, &mut buf).await.expect("dtoh");
    let got = f32::from_le_bytes(buf) as usize;
    eprintln!("[dbg argmax] got={got} host={} max_host_val={}", host(&vals, 0), vals[host(&vals, 0)]);
    eprintln!("[dbg argmax] vals[got]={:?}", vals.get(got));
    assert_eq!(got, host(&vals, 0), "offset=0 argmax 错位");

    // 用例 2:offset = 1000(末行窄视等价;平局验证首见胜)
    let n2 = n - 1000;
    let decl = argmax_f32idx(&TensorOps::from_host(Dtype::F16, vec![n2], &bytes[2000..]), n2, 0);
    let b = crate::interpreters::eval_ops(decl.step(), &mut gpu).await.expect("eval");
    let mut buf = [0u8; 4];
    gpu.dtoh(&b, &mut buf).await.expect("dtoh");
    let got = f32::from_le_bytes(buf) as usize;
    assert_eq!(
        got,
        host(&vals, 1000) - 1000,
        "切片相对索引 argmax 错位"
    );

    // 用例 3:大 n(词表量级 151_936,越 64K 边界扫查)
    let n3 = 151_936usize;
    let vals3: Vec<f32> = (0..n3)
        .map(|i| ((i as f64 * 0.754_877_666).fract() * 2.0 - 1.0) as f32)
        .collect();
    let bytes3: Vec<u8> = vals3
        .iter()
        .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
        .collect();
    let decl = argmax_f32idx(&TensorOps::from_host(Dtype::F16, vec![n3], &bytes3), n3, 0);
    let b = crate::interpreters::eval_ops(decl.step(), &mut gpu).await.expect("eval");
    let mut buf = [0u8; 4];
    gpu.dtoh(&b, &mut buf).await.expect("dtoh");
    let got = f32::from_le_bytes(buf) as usize;
    assert_eq!(got, host(&vals3, 0), "词表量级 argmax 错位");

    gpu.close().await.expect("关机");
}

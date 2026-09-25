//! attention 端到端样例(M-b):声明式 full-attention 层,GPU server 真发射,
//! host 手写参考全链对拍(Kernel 节点面,CPU face 不参与 —— 见 layers.rs
//! 声明/装载/毒值测试)。
//!
//! 覆盖:qk-norm ×(1+w)(rows=T×H 复用 owl_rmsnorm_f32 w_off)、rope
//! interleaved partial 复用、per-head [value|gate] 切分(owl_narrow_strided
//! ×2)、naive decode attn 连续槽窗(bs=2 异窗位 + 跨步缓存读)、输出门
//! sigmoid × mul、o_proj。
//!
//! 用法:
//! ```text
//! OWL_TEST_DEVICE=3 cargo run -p owl-models --example attention -- gpu
//! ```

use owl_models::contract::DeviceClient;
use owl_models::interpreter::eval_ops;
use owl_models::layers::attention::{Attention, KvBuffers};
use owl_models::layers::rope::Rope;
use owl_models::module::KernelCtx;
use owl_models::tensor::Dtype;
use owl_models::TensorOps;

fn f32b(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

// ============================================================================
// 配置(small-but-faithful:结构同 0.8B,维度缩小)
// ============================================================================

const HQ: usize = 2; // q 头(GQA 2:1)
const HKV: usize = 1; // kv 头
const HD: usize = 8; // head_dim
const HIDDEN: usize = 3;
const ROTARY: usize = 4; // partial 0.5( exercises 直通段)
const MAX_POS: usize = 64;
const THETA: f32 = 10_000.0;
const EPS: f32 = 1e-6;
const MAX_SLOTS: usize = 8;

// ============================================================================
// host 参考实现(对拍锚;与 kernels/cu/text/attention.cu 同一数学,独立副本)
// ============================================================================

struct HostRef {
    wq: Vec<f32>,  // [HQ*HD*2, HIDDEN](safetensors [out,in] 行主序)
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>, // [HIDDEN, HQ*HD]
    qnw: Vec<f32>,
    knw: Vec<f32>,
    kc: Vec<f32>, // [MAX_SLOTS, HKV, HD]
    vc: Vec<f32>,
}

impl HostRef {
    /// y[o] = Σ_i x[i]·w[o·in+i](与 Linear 装载期转置 + owl_matmul 同语义)
    fn lin(&self, x: &[f32], w: &[f32], out_dim: usize) -> Vec<f32> {
        (0..out_dim)
            .map(|o| {
                (0..HIDDEN).map(|i| x[i] * w[o * HIDDEN + i]).sum::<f32>()
            })
            .collect()
    }

    /// 逐行 rmsnorm ×(1+w)(行 = head;与 owl_rmsnorm_f32 w_off=1 同语义)
    fn rms_addone(&self, x: &[f32], gamma: &[f32], cols: usize) -> Vec<f32> {
        x.chunks_exact(cols)
            .flat_map(|row| {
                let ms = row.iter().map(|v| v * v).sum::<f32>() / cols as f32;
                let inv = 1.0 / (ms + EPS).sqrt();
                row.iter().zip(gamma).map(move |(v, g)| v * inv * (g + 1.0))
            })
            .collect()
    }

    /// rope interleaved partial(相邻对 2i/2i+1;旋转维外直通)
    fn rope(&self, x: &[f32], heads: usize, pos: usize) -> Vec<f32> {
        let half = ROTARY / 2;
        let mut out = x.to_vec();
        for h in 0..heads {
            let row = &x[h * HD..(h + 1) * HD];
            for i in 0..half {
                let ang = (pos as f32) * THETA.powf(-(2.0 * i as f32) / ROTARY as f32);
                let (c, s) = (ang.cos(), ang.sin());
                let (a, b) = (row[2 * i], row[2 * i + 1]);
                out[h * HD + 2 * i] = a * c - b * s;
                out[h * HD + 2 * i + 1] = a * s + b * c;
            }
            // 2*half..HD 直通(已由 to_vec 复制)
        }
        out
    }

    /// 一个 decode 步(bs 个 token;先写 cache 后按连续槽窗打分,核同序)
    fn step(&mut self, xs: &[f32], pos: &[f32], slots: &[f32], kv_lens: &[f32]) -> Vec<f32> {
        let bs = xs.len() / HIDDEN;
        let mut out_all = vec![0.0f32; bs * HIDDEN];
        for t in 0..bs {
            let x = &xs[t * HIDDEN..(t + 1) * HIDDEN];
            let q_raw = self.lin(x, &self.wq, HQ * HD * 2);
            let k = self.lin(x, &self.wk, HKV * HD);
            let v = self.lin(x, &self.wv, HKV * HD);

            // per-head [value|gate] 切分
            let mut q = vec![0.0f32; HQ * HD];
            let mut gate = vec![0.0f32; HQ * HD];
            for h in 0..HQ {
                for d in 0..HD {
                    q[h * HD + d] = q_raw[h * HD * 2 + d];
                    gate[h * HD + d] = q_raw[h * HD * 2 + HD + d];
                }
            }

            // qk-norm(per-head 行)→ rope
            let q = self.rms_addone(&q, &self.qnw, HD);
            let k = self.rms_addone(&k, &self.knw, HD);
            let q = self.rope(&q, HQ, pos[t] as usize);
            let k = self.rope(&k, HKV, pos[t] as usize);

            // naive decode attention(连续槽窗 [slot-kv_len+1, slot])
            let slot = slots[t] as i32;
            let kv_len = kv_lens[t] as i32;
            let base = slot - kv_len + 1;
            let mut y = vec![0.0f32; HQ * HD];
            for h in 0..HQ {
                let kvh = h / (HQ / HKV);
                if slot >= 0 {
                    for d in 0..HD {
                        self.kc[slot as usize * HKV * HD + kvh * HD + d] = k[kvh * HD + d];
                        self.vc[slot as usize * HKV * HD + kvh * HD + d] = v[kvh * HD + d];
                    }
                }
                let qs = &q[h * HD..(h + 1) * HD];
                let scale = 1.0 / (HD as f32).sqrt();
                let mut scores = vec![0.0f32; kv_len as usize];
                let mut maxs = f32::MIN;
                for (si, sc) in scores.iter_mut().enumerate() {
                    let row = (base + si as i32) as usize;
                    let off = row * HKV * HD + kvh * HD;
                    let acc: f32 =
                        qs.iter().zip(&self.kc[off..off + HD]).map(|(a, b)| a * b).sum::<f32>() * scale;
                    *sc = acc;
                    maxs = maxs.max(acc);
                }
                let denom: f32 = scores.iter().map(|s| (s - maxs).exp()).sum();
                for (si, sc) in scores.iter().enumerate() {
                    let wgt = (sc - maxs).exp() / denom;
                    let row = (base + si as i32) as usize;
                    let off = row * HKV * HD + kvh * HD;
                    for d in 0..HD {
                        y[h * HD + d] += wgt * self.vc[off + d];
                    }
                }
            }

            // 输出门 + o_proj
            for (yi, g) in y.iter_mut().zip(&gate) {
                *yi *= 1.0 / (1.0 + (-g).exp());
            }
            let o = self.lin(&y, &self.wo, HIDDEN);
            out_all[t * HIDDEN..(t + 1) * HIDDEN].copy_from_slice(&o);
        }
        out_all
    }
}

// ============================================================================
// GPU 管线:两步 decode(bs=2 异窗位 → bs=1 跨步缓存读)
// ============================================================================

#[tokio::main]
async fn main() {
    use owl_cuda::{test_device_ordinal, DeviceSelector, GpuClient};

    // 权重(确定性公式;源布局 [out,in] 行主序)
    let wq: Vec<f32> = (0..HQ * HD * 2 * HIDDEN).map(|i| (i as f32 * 0.061) - 0.35).collect();
    let wk: Vec<f32> = (0..HKV * HD * HIDDEN).map(|i| (i as f32 * 0.043) - 0.22).collect();
    let wv: Vec<f32> = (0..HKV * HD * HIDDEN).map(|i| (i as f32 * 0.083) - 0.18).collect();
    let wo: Vec<f32> = (0..HIDDEN * HQ * HD).map(|i| (i as f32 * 0.037) - 0.31).collect();
    let qnw: Vec<f32> = (0..HD).map(|i| (i as f32 * 0.11) - 0.2).collect();
    let knw: Vec<f32> = (0..HD).map(|i| (i as f32 * 0.07) - 0.1).collect();

    // GPU server face
    let mut client = GpuClient::spawn(DeviceSelector::Ordinal(test_device_ordinal()))
        .expect("gpu server boot");

    // 层容器 + 装载(六槽)
    let attn = Attention::new(HQ, HKV, HD, HIDDEN, EPS);
    let src = std::collections::HashMap::from([
        ("q_proj".to_string(), wq.clone()),
        ("k_proj".to_string(), wk.clone()),
        ("v_proj".to_string(), wv.clone()),
        ("o_proj".to_string(), wo.clone()),
        ("q_norm".to_string(), qnw.clone()),
        ("k_norm".to_string(), knw.clone()),
    ]);
    owl_models::interpreter::eval_load(&attn, &mut client, &src, &Default::default())
        .await
        .expect("eval_load 六槽");

    // rope 表(全局一份;cos/sin 物化)
    let rp = Rope::new(MAX_POS, HD, ROTARY, THETA).expect("rope new");
    owl_models::interpreter::eval_load(&rp, &mut client, &rp.tables(), &Default::default())
        .await
        .expect("rope 表物化");

    // KV 常驻块(Zeros = 清零分配;跨步持久)
    let kc = client.alloc(MAX_SLOTS * HKV * HD * 4).await.expect("alloc kc");
    let vc = client.alloc(MAX_SLOTS * HKV * HD * 4).await.expect("alloc vc");

    // host 参考(独立副本;跨步持缓存)
    let mut host = HostRef {
        wq, wk, wv, wo, qnw, knw,
        kc: vec![0.0; MAX_SLOTS * HKV * HD],
        vc: vec![0.0; MAX_SLOTS * HKV * HD],
    };

    // 两步 decode:
    // A:bs=2,异序列异窗位(seq0 窗 [0,1),seq1 窗 [4,6) —— 基址非零,行 4 为零行)
    // B:seq0 续步,窗 [0,2) 读回 A 步写入的 row 0(跨步缓存读)
    let steps: [(&[f32], &[f32], &[f32], &[f32]); 2] = [
        (
            &[0.5, -0.25, 1.0, -0.6, 0.4, 0.9], // xs(bs=2 × hidden)
            &[11.0, 3.0],                       // pos
            &[0.0, 5.0],                        // slots
            &[1.0, 2.0],                        // kv_lens(含本步)
        ),
        (
            &[0.3, 0.8, -0.5],
            &[12.0],
            &[1.0],
            &[2.0],
        ),
    ];

    for (si, (xs, pos, slots, kv_lens)) in steps.iter().enumerate() {
        let bs = xs.len() / HIDDEN;

        // 声明
        let kv = KvBuffers {
            k_cache: TensorOps::of_block(kc.id, Dtype::F32, vec![MAX_SLOTS, HKV, HD]),
            v_cache: TensorOps::of_block(vc.id, Dtype::F32, vec![MAX_SLOTS, HKV, HD]),
            slots: TensorOps::from_host(Dtype::F32, vec![bs], &f32b(slots)),
            kv_lens: TensorOps::from_host(Dtype::F32, vec![bs], &f32b(kv_lens)),
        };
        let xs_t = TensorOps::from_host(Dtype::F32, vec![bs, HIDDEN], &f32b(xs));
        let pos_t = TensorOps::from_host(Dtype::F32, vec![bs], &f32b(pos));
        let decl = attn.forward(&xs_t, &rp, &pos_t, &kv, &KernelCtx { tokens: bs });

        // 执行 + 收割
        let bytes = eval_ops(decl.step(), &mut client).await.expect("eval");
        let mut buf = vec![0u8; bs * HIDDEN * 4];
        client.dtoh(&bytes, &mut buf).await.expect("dtoh");
        let got: Vec<f32> = buf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // host 参考
        let want = host.step(xs, pos, slots, kv_lens);

        // 对拍
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() < 1e-4,
                "step {si} [{i}] gpu {g} vs host {w}\n  gpu = {got:?}\n  host = {want:?}"
            );
        }
        println!("step {si}: gpu = {got:?}");
    }

    client.close().await.expect("server 关机");
    println!("demo: attention 端到端 ✓(GPU 两步 decode,host 全链参考 allclose 1e-4)");
}

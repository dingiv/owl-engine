//! GDN 端到端样例(M-c):GatedDeltaNet 层,GPU server 真发射,host 全链
//! 手写参考两步 decode 对拍(small-but-faithful:结构同 0.8B,维度缩小)。
//!
//! 覆盖:SplitQkvZa 投影(4 Linear)→ narrow 列切分 → conv 三段独立发射
//! (w_offset 段基址 + state 原地滑窗 + silu)→ q/k per-head l2norm →
//! g/beta 门控(g 新核 / beta 复用 sigmoid)→ delta 单步递推(rec 原地)
//! → norm_act(×w 非零中心 × silu(z))→ out_proj。
//!
//! 用法:
//! ```text
//! OWL_TEST_DEVICE=3 cargo run -q -p owl-models --example gdn
//! ```

use owl_models::contract::DeviceClient;
use owl_models::interpreter::eval_ops;
use owl_models::layers::gdn::{fixture, GatedDeltaNet, GdnBuffers};
use owl_models::module::{ForwardCtx, Module};
use owl_models::tensor::Dtype;
use owl_models::TensorOps;

fn f32b(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

const NK: usize = 2;
const HK_DIM: usize = 4;
const NV: usize = 2;
const HV_DIM: usize = 4;
const HIDDEN: usize = 6;
const MAX_SLOTS: usize = 4;

// ============================================================================
// host 全链参考(独立副本;内维显式传参 —— M-b/M-c 两次踩同一雷的教训)
// ============================================================================

struct HostRef {
    src: std::collections::HashMap<String, Vec<f32>>,
    conv: [Vec<f32>; 3], // q/k/v 段状态 [max_slots, seg, 3]
    rec: Vec<f32>,       // [max_slots, NV, HK_DIM, HV_DIM]
}

impl HostRef {
    fn new(src: std::collections::HashMap<String, Vec<f32>>) -> Self {
        let key_dim = NK * HK_DIM;
        let value_dim = NV * HV_DIM;
        HostRef {
            conv: [vec![0.0; MAX_SLOTS * key_dim * 3], vec![0.0; MAX_SLOTS * key_dim * 3], vec![0.0; MAX_SLOTS * value_dim * 3]],
            rec: vec![0.0; MAX_SLOTS * NV * HK_DIM * HV_DIM],
            src,
        }
    }

    fn lin(&self, x: &[f32], w: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
        (0..out_dim)
            .map(|o| (0..in_dim).map(|i| x[i] * w[o * in_dim + i]).sum::<f32>())
            .collect()
    }

    fn step(&mut self, xs: &[f32], slots: &[f32]) -> Vec<f32> {
        let key_dim = NK * HK_DIM;
        let value_dim = NV * HV_DIM;
        let conv_dim = 2 * key_dim + value_dim;
        let batch = slots.len();
        let s = &self.src;
        let mut out_all = vec![0.0f32; batch * HIDDEN];
        for t in 0..batch {
            let slot = slots[t] as i32;
            let x = &xs[t * HIDDEN..(t + 1) * HIDDEN];
            let qkv = self.lin(x, &s["in_proj_qkv"], conv_dim, HIDDEN);
            let z = self.lin(x, &s["in_proj_z"], value_dim, HIDDEN);
            let b_v = self.lin(x, &s["in_proj_b"], NV, HIDDEN);
            let a_v = self.lin(x, &s["in_proj_a"], NV, HIDDEN);
            let segs = [&qkv[0..key_dim], &qkv[key_dim..2 * key_dim], &qkv[2 * key_dim..]];
            let seg_dim = [key_dim, key_dim, value_dim];
            let seg_off = [0usize, key_dim, 2 * key_dim];
            let mut post = [vec![0.0f32; key_dim], vec![0.0f32; key_dim], vec![0.0f32; value_dim]];
            for (seg, inp) in segs.iter().enumerate() {
                let st = &mut self.conv[seg];
                let dim = seg_dim[seg];
                for ch in 0..dim {
                    let sb = (slot as usize * dim + ch) * 3;
                    let wb = (seg_off[seg] + ch) * 4;
                    let hist = [st[sb], st[sb + 1], st[sb + 2]];
                    let w = &s["conv1d"];
                    let mut sum = inp[ch] * w[wb + 3];
                    for kk in 0..3 {
                        sum += hist[kk] * w[wb + kk];
                    }
                    sum /= 1.0 + (-sum).exp();
                    st[sb] = hist[1];
                    st[sb + 1] = hist[2];
                    st[sb + 2] = inp[ch];
                    post[seg][ch] = sum;
                }
            }
            // q/k l2norm per-head
            let l2 = |row: &[f32]| {
                let ss: f32 = row.iter().map(|v| v * v).sum();
                let inv = 1.0 / (ss.max(0.0) + 1e-6).sqrt();
                row.iter().map(|v| v * inv).collect::<Vec<f32>>()
            };
            let mut qn = vec![0.0f32; key_dim];
            let mut kn = vec![0.0f32; key_dim];
            for h in 0..NK {
                qn[h * HK_DIM..(h + 1) * HK_DIM].copy_from_slice(&l2(&post[0][h * HK_DIM..(h + 1) * HK_DIM]));
                kn[h * HK_DIM..(h + 1) * HK_DIM].copy_from_slice(&l2(&post[1][h * HK_DIM..(h + 1) * HK_DIM]));
            }
            // 门控
            let a_log = &s["A_log"];
            let dtb = &s["dt_bias"];
            let g_v: Vec<f32> = (0..NV)
                .map(|h| {
                    let x2 = a_v[h] + dtb[h];
                    let sp = if x2 < 20.0 { x2.exp().ln_1p() } else { x2 };
                    -a_log[h].exp() * sp
                })
                .collect();
            let beta_v: Vec<f32> = b_v.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect();
            // delta 单步
            let q_scale = 1.0 / (HK_DIM as f32).sqrt();
            let mut y = vec![0.0f32; value_dim];
            for vh in 0..NV {
                let kh = vh / (NV / NK);
                let decay = g_v[vh].exp();
                let sb = (slot as usize * NV + vh) * HK_DIM * HV_DIM;
                let qoff = kh * HK_DIM;
                let voff = vh * HV_DIM;
                let mut kv_mem = vec![0.0f32; HV_DIM];
                for j in 0..HK_DIM {
                    for d in 0..HV_DIM {
                        let idx = sb + j * HV_DIM + d;
                        self.rec[idx] *= decay;
                        kv_mem[d] += self.rec[idx] * kn[qoff + j];
                    }
                }
                for d in 0..HV_DIM {
                    let delta = (post[2][voff + d] - kv_mem[d]) * beta_v[vh];
                    let mut acc = 0.0f32;
                    for j in 0..HK_DIM {
                        let idx = sb + j * HV_DIM + d;
                        self.rec[idx] += kn[qoff + j] * delta;
                        acc += self.rec[idx] * qn[qoff + j] * q_scale;
                    }
                    y[voff + d] = acc;
                }
            }
            // norm_act(per-head ×w × silu(z))+ out_proj
            let gamma = &s["norm"];
            let mut gated = vec![0.0f32; value_dim];
            for vh in 0..NV {
                let base = vh * HV_DIM;
                let ms: f32 = (0..HV_DIM).map(|i| y[base + i] * y[base + i]).sum::<f32>() / HV_DIM as f32;
                let inv = 1.0 / (ms.max(0.0) + 1e-6).sqrt();
                for i in 0..HV_DIM {
                    let zv = z[base + i];
                    gated[base + i] = y[base + i] * inv * gamma[i] * (zv / (1.0 + (-zv).exp()));
                }
            }
            let o = self.lin(&gated, &s["out_proj"], HIDDEN, value_dim);
            out_all[t * HIDDEN..(t + 1) * HIDDEN].copy_from_slice(&o);
        }
        out_all
    }
}

#[tokio::main]
async fn main() {
    use owl_cuda::{test_device_ordinal, DeviceSelector, GpuClient};

    let key_dim = NK * HK_DIM;
    let value_dim = NV * HV_DIM;
    let src = fixture::weights(NK, HK_DIM, NV, HV_DIM, HIDDEN)
        .into_iter()
        .enumerate()
        .map(|(i, (k, n))| (k, fixture::gen(n, 10.0 + i as f32)))
        .collect::<std::collections::HashMap<String, Vec<f32>>>();

    let mut client = GpuClient::spawn(DeviceSelector::Ordinal(test_device_ordinal())).expect("gpu server boot");
    let layer = GatedDeltaNet::new(NK, HK_DIM, NV, HV_DIM, HIDDEN, 1e-6);
    owl_models::interpreter::eval_load(&layer, &mut client, &src, &Default::default())
        .await
        .expect("eval_load 九槽");

    // 常驻状态块(清零分配)
    async fn alloc_block(client: &mut owl_cuda::GpuClient, v: &[f32], shape: Vec<usize>) -> TensorOps {
        let b = eval_ops(TensorOps::from_host(Dtype::F32, vec![v.len()], &f32b(v)).step(), client)
            .await
            .expect("alloc");
        TensorOps::of_block(b.id, Dtype::F32, shape)
    }
    let fill = |n: usize| vec![0.0f32; n];
    let cq = alloc_block(&mut client, &fill(MAX_SLOTS * key_dim * 3), vec![MAX_SLOTS, key_dim, 3]).await;
    let ck = alloc_block(&mut client, &fill(MAX_SLOTS * key_dim * 3), vec![MAX_SLOTS, key_dim, 3]).await;
    let cv = alloc_block(&mut client, &fill(MAX_SLOTS * value_dim * 3), vec![MAX_SLOTS, value_dim, 3]).await;
    let rc = alloc_block(&mut client, &fill(MAX_SLOTS * NV * HK_DIM * HV_DIM), vec![MAX_SLOTS, NV, HK_DIM, HV_DIM]).await;

    // host 参考(独立副本;跨步持缓存)
    let mut host = HostRef::new(src.clone());

    // 两步 decode:A(bs=2,slot 0/2)→ B(bs=1,slot 1 续步)
    let steps: [(&[f32], &[f32]); 2] = [
        (&fixture::gen(2 * HIDDEN, 1.0), &[0.0, 2.0]),
        (&fixture::gen(HIDDEN, 2.0), &[1.0]),
    ];

    for (si, (xs, slots)) in steps.iter().enumerate() {
        let batch = slots.len();
        let gdn_buf = GdnBuffers {
            conv_q: cq.clone(),
            conv_k: ck.clone(),
            conv_v: cv.clone(),
            rec: rc.clone(),
            slots: TensorOps::from_host(Dtype::F32, vec![batch], &f32b(slots)),
        };
        let xs_t = TensorOps::from_host(Dtype::F32, vec![batch, HIDDEN], &f32b(xs));
        let ctx = ForwardCtx::gdn_decode(batch, &gdn_buf);
        let decl = layer.forward(&xs_t, &ctx);

        let n = batch * HIDDEN;
        let bytes = eval_ops(decl.step(), &mut client).await.expect("eval");
        let mut buf = vec![0u8; n * 4];
        client.dtoh(&bytes, &mut buf).await.expect("dtoh");
        let got: Vec<f32> = buf.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

        let want = host.step(xs, slots);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!((g - w).abs() < 1e-4, "step {si} [{i}] gpu {g} vs host {w}");
        }
        println!("step {si}: gpu = {got:?}");
    }

    client.close().await.expect("server 关机");
    println!("demo: gdn 端到端 ✓(GPU 两步 decode,host 全链参考 allclose 1e-4)");
}

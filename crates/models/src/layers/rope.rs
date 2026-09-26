//! Rope:旋转位置编码(rotate-half + partial;Kernel 注册表)。
//!
//! Qwen3.5-0.8B 实测配置:head_dim 256,partial_rotary_factor 0.25 →
//! rotary_dim 64(half 32)。**配对 = rotate-half (i, i+half)** ——
//! 2026-09-26 HF 探针翻案:config `mrope_interleaved` 指三网格(T/H/W)
//! 在频率维交错,非 GPT-J 相邻对;文本路径三网格同 pos 退化后就是普通
//! rotate-half(maxdiff 6e-8 vs 相邻对 2.5)。
//! 文本路径三 section 的 pos 相同(mrope 退化为一维 rope,t [11,11,10]
//! 分段仅在视觉多模态路径有意义 —— 本层只吃一份 pos)。
//!
//! cos/sin 表:new 期纯计算存于层内,经 `tables()` 容器内表源供执行器
//! 取数(不经外部数据源);forward 常驻块引用。
//! kernel 源 = 注册表 `owl_rope_half_partial_f32`(owl-kernels cu/text)。

use crate::contract::ModelError;
use crate::kernel::Kernel;
use crate::kernel;
use crate::module::{Loadable, LoaderCtx, LoaderOps, Weight};
use crate::tensor::Dtype;
use crate::TensorOps;

pub struct Rope {
    /// cos 表(new 期纯计算;经 tables() 源供给执行器)
    cos_data: Vec<f32>,
    /// sin 表
    sin_data: Vec<f32>,
    /// cos 表槽 [max_pos, half](常驻块)
    cos: Weight,
    /// sin 表槽 [max_pos, half]
    sin: Weight,
    head_dim: usize,
    rotary_dim: usize,
}

impl Rope {
    /// `theta`(rope_theta = 1e7)、`rotary_dim` = head_dim × partial(0.25 → 64)。
    /// 配置校验是 new 唯一可败点(结构参数非法 = 容器语义不成立,非数据错)。
    pub fn new(
        max_pos: usize,
        head_dim: usize,
        rotary_dim: usize,
        theta: f32,
    ) -> Result<Rope, ModelError> {
        if rotary_dim % 2 != 0 || rotary_dim > head_dim {
            return Err(ModelError::Msg(format!(
                "Rope::new: rotary_dim {rotary_dim} 非法(须偶数且 ≤ head_dim {head_dim})"
            )));
        }
        let half = rotary_dim / 2;
        // inv_freq[i] = theta^(-2i / rotary_dim);rotate-half:对 (i, i+half) 用同一频率
        // (与 HF compute_default_rope_parameters 同式:arange(0,dim,2)/dim)
        let cos_t: Vec<f32> = (0..max_pos)
            .flat_map(|p| {
                (0..half).map(move |i| {
                    ((p as f32) * theta.powf(-(2.0 * i as f32) / rotary_dim as f32)).cos()
                })
            })
            .collect();
        let sin_t: Vec<f32> = (0..max_pos)
            .flat_map(|p| {
                (0..half).map(move |i| {
                    ((p as f32) * theta.powf(-(2.0 * i as f32) / rotary_dim as f32)).sin()
                })
            })
            .collect();
        Ok(Rope {
            cos_data: cos_t,
            sin_data: sin_t,
            cos: Weight::new("cos_table", vec![max_pos, half]),
            sin: Weight::new("sin_table", vec![max_pos, half]),
            head_dim,
            rotary_dim,
        })
    }

    /// 容器内表源(执行器取数用;借用层内表,生命周期随层)
    pub fn tables(&self) -> crate::module::TableSource<'_> {
        crate::module::TableSource::new(vec![
            ("cos_table", &self.cos_data),
            ("sin_table", &self.sin_data),
        ])
    }

    fn launch_kernel(name: &'static str, tokens: usize) -> Kernel {
        kernel::kernel_with(name, (tokens as u32, 1, 1), (128, 1, 1), 0)
    }

    /// q 旋转:[T, Hq*HD] → [T, Hq*HD](前 rotary_dim 维转,余直通)
    pub fn forward_q(
        &self,
        q: &TensorOps,
        pos: &TensorOps,
        tokens: usize,
        q_heads: usize,
    ) -> TensorOps {
        // 核名/输出 dtype 跟随 x 声明(F5;表 Weight 经 LoaderCtx 同 dtype)
        let dt = q.dtype;
        let name = if dt == Dtype::F16 {
            "owl_rope_half_partial_f16"
        } else {
            "owl_rope_half_partial_f32"
        };
        TensorOps::of(Self::launch_kernel(
            name,
            tokens,
        ))
        .arg(q)
        .arg(&self.cos.decl())
        .arg(&self.sin.decl())
        .arg(pos)
        .arg_usize(q_heads)
        .arg_usize(self.head_dim)
        .arg_usize(self.rotary_dim / 2)
        .with_shape(dt, vec![tokens, q_heads * self.head_dim])
    }

    /// k 旋转:[T, Hkv*HD] → [T, Hkv*HD]
    pub fn forward_k(
        &self,
        k: &TensorOps,
        pos: &TensorOps,
        tokens: usize,
        kv_heads: usize,
    ) -> TensorOps {
        let dt = k.dtype;
        let name = if dt == Dtype::F16 {
            "owl_rope_half_partial_f16"
        } else {
            "owl_rope_half_partial_f32"
        };
        TensorOps::of(Self::launch_kernel(
            name,
            tokens,
        ))
        .arg(k)
        .arg(&self.cos.decl())
        .arg(&self.sin.decl())
        .arg(pos)
        .arg_usize(kv_heads)
        .arg_usize(self.head_dim)
        .arg_usize(self.rotary_dim / 2)
        .with_shape(dt, vec![tokens, kv_heads * self.head_dim])
    }
}

impl Loadable for Rope {
    fn layout(&self, ctx: &LoaderCtx) -> LoaderOps {
        self.cos.layout(ctx).chain(self.sin.layout(ctx))
    }

}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::f32b;
    use crate::tensor::Dtype;

    #[tokio::test]
    async fn declaration_is_wellformed() {
        let (head_dim, rotary_dim, heads) = (8usize, 4usize, 2usize);
        let mut face = owl_cpu::CpuFace::new();
        let rp = Rope::new(64, head_dim, rotary_dim, 10_000.0).expect("new");
        crate::interpreters::eval_load(&rp, &mut face, &rp.tables(), &Default::default())
            .await
            .expect("表物化");

        let pos = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[5.0]));
        let q = TensorOps::from_host(
            Dtype::F32,
            vec![1, heads * head_dim],
            &f32b(&vec![0.3; heads * head_dim]),
        );

        let out = rp.forward_q(&q, &pos, 1, heads);
        assert!(!out.is_poisoned(), "rope 声明不应有毒");
        assert_eq!(out.shape(), &[1, heads * head_dim]);

        assert!(Rope::new(64, head_dim, 7, 10_000.0).is_err(), "非偶 rotary_dim 应拒");
        assert!(Rope::new(64, head_dim, head_dim * 2, 10_000.0).is_err(), "超 head_dim 应拒");
    }

    /// Rope = 位置上下文机制(双输入 + pos),不进 Module(C4 后仍成立;
    /// attention 经 ForwardCtx 持 &Rope 调用其固有 forward_q/k)
    #[test]
    fn stays_outside_module_interface() {
        fn assert_impl<M: crate::module::Module>(_: &M) {}
        assert_impl(&crate::layers::mlp::Mlp::new(2, 4)); // Mlp 在 Module 内
        // Rope 无 forward(&TensorOps, &ForwardCtx) 签名 —— 编译期即证不在 trait 内
        let rp = Rope::new(64, 8, 4, 10_000.0).expect("new");
        let pos = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[5.0]));
        let _ = rp.forward_q(
            &TensorOps::from_host(Dtype::F32, vec![1, 16], &f32b(&vec![0.3; 16])),
            &pos,
            1,
            2,
        );
    }

    /// HF parity(rotate-half partial 金标准 = HF 类直跑;GPU vs HF。
    /// rope kernel 是 Kernel 节点 = GPU-only(CPU face 不注册,
    /// 同 owl_narrow_strided 先例);双门控 OWL_HF_PARITY=1 + OWL_TEST_DEVICE)
    #[tokio::test]
    async fn parity_matches_hf() {
        use crate::testkit::{assert_close, harvest, hf_python, parity_enabled, skip_note, st_read, st_write, tmp_path};
        if !parity_enabled() {
            skip_note();
            return;
        }
        let (tokens, heads, head_dim, rotary_dim) = (3usize, 2usize, 8usize, 4usize);
        let theta = 10_000.0f32;
        let x: Vec<f32> = (0..tokens * heads * head_dim)
            .map(|i| ((i as f32 * 0.37) - 1.1).cos())
            .collect();
        let pos: Vec<f32> = vec![2.0, 7.0, 13.0];

        let inp = tmp_path("rope_in");
        let outp = tmp_path("rope_out");
        st_write(
            &inp,
            &[
                ("x", &x, vec![tokens, heads * head_dim]),
                ("pos", &pos, vec![tokens]),
            ],
        );
        let contract = format!(
            r#"{{"heads": {heads}, "head_dim": {head_dim}, "rotary_dim": {rotary_dim}, "theta": {theta}}}"#
        );
        let manifest = hf_python("rope.py", &[inp.to_str().unwrap(), outp.to_str().unwrap(), &contract]);
        eprintln!("[hf manifest] rope: {manifest}");
        let want = st_read(&outp, "y");

        let x_t = TensorOps::from_host(Dtype::F32, vec![tokens, heads * head_dim], &f32b(&x));
        let pos_t = TensorOps::from_host(Dtype::F32, vec![tokens], &f32b(&pos));

        // GPU 臂 vs HF(kernel = owl_rope_half_partial_f32;CPU face 不执行 Kernel 节点)
        if crate::testkit::gpu_enabled() {
            let mut gpu = crate::testkit::gpu_client().await;
            let rp2 = Rope::new(256, head_dim, rotary_dim, theta).expect("rope new");
            crate::interpreters::eval_load(&rp2, &mut gpu, &rp2.tables(), &Default::default())
                .await
                .expect("eval_load(gpu)");
            let gpu_out = harvest(&mut gpu, &rp2.forward_q(&x_t, &pos_t, tokens, heads)).await;
            assert_close(&gpu_out, &want, 1e-5, "rope-gpu");
            gpu.close().await.expect("server 关机");
        }

        std::fs::remove_file(&inp).ok();
        std::fs::remove_file(&outp).ok();
    }
}
#[cfg(test)]
mod f16_tests {
    use super::*;
    use crate::contract::DeviceClient as _;
    use crate::testkit::{gpu_client, gpu_enabled};
    use crate::tensor::Dtype;

    fn half_le(f: f32) -> [u8; 2] {
        half::f16::from_f32(f).to_le_bytes()
    }

    /// GPU:f16 rope(rotate-half partial)vs host f32(F3;OWL_TEST_DEVICE 门控)
    #[tokio::test]
    async fn gpu_rope_f16_matches_host() {
        if !gpu_enabled() {
            eprintln!("skip: OWL_TEST_DEVICE 未设");
            return;
        }
        let (max_pos, hd, half, heads, tokens) = (64usize, 8usize, 4usize, 2usize, 3usize);
        // 表(host f32 生成 → f16 上载;与 Rope::new 同式)
        let cos: Vec<f32> = (0..max_pos)
            .flat_map(|p| (0..half).map(move |i| (p as f32 * (10000f32).powf(-(2.0 * i as f32) / hd as f32)).cos()))
            .collect();
        let sin: Vec<f32> = (0..max_pos)
            .flat_map(|p| (0..half).map(move |i| (p as f32 * (10000f32).powf(-(2.0 * i as f32) / hd as f32)).sin()))
            .collect();
        let x: Vec<f32> = (0..tokens * heads * hd).map(|i| ((i as f32 * 0.29) - 2.0).sin()).collect();
        let pos = [1.0f32, 4.0f32, 9.0f32];

        let mut gpu = gpu_client().await;
        let sh = crate::contract::Shape::from;
        let dx = gpu.htod(Dtype::F16, &sh(vec![tokens, heads, hd]), &x.iter().flat_map(|f| half_le(*f)).collect::<Vec<u8>>()).await.expect("x");
        let dc = gpu.htod(Dtype::F16, &sh(vec![max_pos, half]), &cos.iter().flat_map(|f| half_le(*f)).collect::<Vec<u8>>()).await.expect("cos");
        let ds = gpu.htod(Dtype::F16, &sh(vec![max_pos, half]), &sin.iter().flat_map(|f| half_le(*f)).collect::<Vec<u8>>()).await.expect("sin");
        let dp = gpu.htod(Dtype::F32, &sh(vec![tokens]), &crate::testkit::f32b(&pos)).await.expect("pos");
        let (x_d, c_d, s_d, p_d) = (
            TensorOps::of_block(dx.id, Dtype::F16, vec![tokens, heads, hd]),
            TensorOps::of_block(dc.id, Dtype::F16, vec![max_pos, half]),
            TensorOps::of_block(ds.id, Dtype::F16, vec![max_pos, half]),
            TensorOps::of_block(dp.id, Dtype::F32, vec![tokens]),
        );
        let decl = TensorOps::of(crate::kernel::kernel_with(
            "owl_rope_half_partial_f16", (tokens as u32, 1, 1), (128, 1, 1), 0,
        ))
        .arg(&x_d).arg(&c_d).arg(&s_d).arg(&p_d)
        .arg_usize(heads).arg_usize(hd).arg_usize(half)
        .with_shape(Dtype::F16, vec![tokens, heads, hd]);
        let out = crate::interpreters::eval_ops(decl.step(), &mut gpu).await.expect("eval");
        let mut buf = vec![0u8; tokens * heads * hd * 2];
        gpu.dtoh(&out, &mut buf).await.expect("dtoh");

        let mut max_diff = 0f32;
        for t in 0..tokens {
            let p = pos[t] as usize;
            for h in 0..heads {
                for i in 0..half {
                    let a = x[(t * heads + h) * hd + i];
                    let b = x[(t * heads + h) * hd + i + half];
                    let cf = cos[p * half + i];
                    let sf = sin[p * half + i];
                    let want0 = a * cf - b * sf;
                    let want1 = b * cf + a * sf;
                    let g0 = half::f16::from_le_bytes([buf[((t * heads + h) * hd + i) * 2], buf[((t * heads + h) * hd + i) * 2 + 1]]).to_f32();
                    let g1 = half::f16::from_le_bytes([buf[((t * heads + h) * hd + i + half) * 2], buf[((t * heads + h) * hd + i + half) * 2 + 1]]).to_f32();
                    max_diff = max_diff.max((g0 - want0).abs()).max((g1 - want1).abs());
                }
                // partial 直通维逐位
                for d in 2 * half..hd {
                    let g = half::f16::from_le_bytes([buf[((t * heads + h) * hd + d) * 2], buf[((t * heads + h) * hd + d) * 2 + 1]]).to_f32();
                    assert_eq!(g, x[(t * heads + h) * hd + d], "直通维 [{t},{h},{d}]");
                }
            }
        }
        eprintln!("[rope f16] max_diff = {max_diff:.5}");
        assert!(max_diff < 2e-2, "旋转维容差 2e-2(f16 表量化),得 {max_diff}");
        gpu.close().await.expect("关机");
    }
}

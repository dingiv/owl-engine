//! Linear:仿射层(容器 + LoaderOps 装载 + Matmul;无 bias —— Qwen3.5
//! 全系 attention/mlp 权重无 bias)。
//!
//! 权重槽 = **原生布局直读**(2026-09-26 F5-2:检查点 [out, in] 行主序
//! 原样装载,forward matmul_nt —— cuBLAS OP_T 免费转置,host 转置路径
//! 全删;旧"装载期转置"是 naive-matmul 时代产物)。

use crate::contract::Dtype;
use crate::layers::narrow_strided;
use crate::module::{ForwardCtx, Loadable, LoaderCtx, LoaderOps, Module, QuantPlan, Weight};
use crate::formats::w4a16::marlin_n_pack;
use crate::TensorOps;

pub struct Linear {
    /// 权重槽 [out, in](检查点原生布局,零转置)
    w: Weight,
    /// E3 量化臂尺寸(声明期常量)
    out_dim: usize,
    in_dim: usize,
    /// 量化组大小(Some = 已启用;None = f16 直读;构造期定形)
    quant_group: Option<usize>,
    /// 量化四件套(qweight marlin-packed / scales / zeros(仅 AWQ) /
    /// workspace;键 = `{w.key}.qweight/.scales/.zeros/.ws`)
    qw: Option<Weight>,
    sc: Option<Weight>,
    zs: Option<Weight>,
    ws: Option<Weight>,
    ctmp: Option<Weight>,
}

impl Linear {
    /// 权重槽叶子声明(刀3a':gdn b/a 直发自研 gemv 需绕过 Linear::forward
    /// 的 cublas 臂;仅 f16 直读态合法 —— 量化态调用方须走 forward)
    pub fn weight_decl(&self) -> TensorOps {
        self.w.decl()
    }

    /// 准备容器(`key` = 数据源槽键;`plan` = 量化计划构造期注入 ——
    /// 零数据零副作用,零突变面)。量化计划下按尺寸门控当场定形:
    /// marlin tile 约束(n%256==0 且 k%128==0,与装载源同源)不满足的
    /// 小线性保持 f16 直读 —— 装载源侧对同款判定输出反量化 `.weight`。
    pub fn new(key: &'static str, out_dim: usize, in_dim: usize, plan: QuantPlan) -> Linear {
        let mut lin = Linear {
            w: Weight::new(key, vec![out_dim, in_dim]),
            out_dim,
            in_dim,
            quant_group: None,
            qw: None,
            sc: None,
            zs: None,
            ws: None,
            ctmp: None,
        };
        let eligible = crate::formats::w4a16::marlin_eligible(out_dim, in_dim);
        match plan {
            QuantPlan::W4A16 if eligible => lin.quantize_g128(),
            QuantPlan::W4A16Awq if eligible => lin.quantize_awq32(),
            _ => {}
        }
        lin
    }

    /// 取出权重槽(单权重基本函数 `load_weight` 的入口;消费层容器)
    pub fn into_weight(self) -> Weight {
        self.w
    }

    /// W4A16 定形(E3;REQ-PRE-01,构造期私有):forward 换 marlin GEMM
    /// (foreign 通道),装载换三件套(qweight marlin-packed i32 / scales
    /// f16 / workspace 零初始化)。
    fn quantize_g128(&mut self) {
        debug_assert!(crate::formats::w4a16::marlin_eligible(self.out_dim, self.in_dim));
        let g = 128usize;
        let n_pack = marlin_n_pack(self.out_dim); // 直通(eligible 保证 n%256==0)
        let key: &'static str = self.w.key();
        self.qw = Some(Weight::new_typed_u32(
            key,
            vec![self.in_dim / 16, n_pack * 16 / 8],
        ));
        self.sc = Some(Weight::new_typed_f16(
            key,
            vec![self.in_dim / g, n_pack],
        ));
        self.ws = Some(Weight::new_typed_u32(
            key,
            vec![crate::ops::marlin_ws_elems(n_pack)],
        ));
        // c_tmp 独立小块(use_fp32_reduce=false 不触碰,但槽位契约要求
        // 独立指针 —— 不可与 ws alias)
        self.ctmp = Some(Weight::new_typed_u32(key, vec![1]));
        self.quant_group = Some(g);
    }

    /// AWQ g32-asym 定形(2026-10-01;构造期私有):与 g128 同 B 打包,
    /// 增 zeros 槽(pack_marlin_z 产物 [in/g, out/8] U32),GEMM 换
    /// kU4(has_zp)内核;组大小 32(group_blocks = 32/16 = 2 幂次合法)。
    fn quantize_awq32(&mut self) {
        debug_assert!(crate::formats::w4a16::marlin_eligible(self.out_dim, self.in_dim));
        let g = 32usize;
        let n_pack = marlin_n_pack(self.out_dim);
        let key: &'static str = self.w.key();
        self.qw = Some(Weight::new_typed_u32(
            key,
            vec![self.in_dim / 16, n_pack * 16 / 8],
        ));
        self.sc = Some(Weight::new_typed_f16(
            key,
            vec![self.in_dim / g, n_pack],
        ));
        // zeros:pack_marlin_z 产出 (groups, n/8) i32 —— 与 scales 同组轴
        self.zs = Some(Weight::new_typed_u32(
            key,
            vec![self.in_dim / g, n_pack / 8],
        ));
        self.ws = Some(Weight::new_typed_u32(key, vec![crate::ops::marlin_ws_elems(n_pack)]));
        self.ctmp = Some(Weight::new_typed_u32(key, vec![1]));
        self.quant_group = Some(g);
    }
}

impl Module for Linear {
    /// f16 臂:y = x @ W^T([.., in] → [.., out];W 原生 [out, in],nt 直读;
    /// 未装载 → 毒值声明)。
    /// W4A16 臂(E3):foreign GEMM_W4A16(6 Block + 4 sz;out 槽 2 ——
    /// sig 逃生舱 O 槽表达),ws 同块双槽(ws + c_tmp,use_fp32_reduce=false
    /// 不触碰 c_tmp)。
    /// AWQ 臂(2026-10-01):同构但换 GEMM_W4A16_AWQ(7 Block = scales 后
    /// 插 zeros 槽;kU4 has_zp 内核,组大小 32)。
    fn forward(&self, xs: &TensorOps, ctx: &ForwardCtx) -> TensorOps {
        let _ = ctx;
        self.forward_inner(xs)
    }
}

impl Linear {
    /// ctx 无关前向(Embedding::lm_head_matmul 复用;14 个调用点零改动)。
    pub(crate) fn forward_inner(&self, xs: &TensorOps) -> TensorOps {
        if let (Some(g), Some(qw), Some(sc), Some(ws), Some(ctmp)) = (
            self.quant_group, &self.qw, &self.sc, &self.ws, &self.ctmp,
        ) {
            let m: usize = xs.shape()[..xs.shape().len() - 1].iter().product();
            // n_pack = 打包档(= out_dim;pad 路线废案)
            let n_pack = marlin_n_pack(self.out_dim);
            // 激活 dtype 决定臂:F16 → GEMM_W4A16;BF16 → GEMM_W4A16_BF16
            // (E5-DF3 同日十二:草稿路径 BF16 化,scales 保持 F16 装载,
            // a/c=BF16 的 marlin 实例;s_type = F16 两族一致)
            let bf16_act = xs.dtype == Dtype::BF16;
            // marlin 臂装配 = crate::ops::§6(2026-10-12 用户律:层零 kernels
            // 知识 —— 名字/签名/AWQ 插槽全在动作表单源)
            let marlin = crate::ops::marlin_node(
                xs,
                &qw.decl(),
                &sc.decl(),
                &ws.decl(),
                &ctmp.decl(),
                self.zs.as_ref().map(|z| z.decl()).as_ref(),
                m,
                self.in_dim,
                n_pack,
                g as usize,
                bf16_act,
            );
            return narrow_strided(
                &marlin,
                m,
                n_pack,
                0,
                self.out_dim,
                vec![m, self.out_dim],
            );
        }
        xs.matmul_nt(&self.w.decl())
    }
}

impl Loadable for Linear {
    /// f16 臂:单 Want(w)。W4A16 臂:三 Want(qweight U32 / scales F16 /
    /// ws U32 零初始化)—— 键 = `{w.key}.qweight/.scales/.ws`,装载源
    /// (w4a16.rs)按同款谓词供给。AWQ 臂增 zeros Want(`{w.key}.zeros`,
    /// pack_marlin_z 产物;源 awq.rs 供给)。
    fn layout(&self, ctx: &LoaderCtx) -> LoaderOps {
        match (self.quant_group, (&self.qw, &self.sc, &self.ws, &self.ctmp)) {
            (Some(_g), (Some(qw), Some(sc), Some(ws), Some(ctmp))) => {
                let base = self.w.key();
                // qweight 键:AWQ + ctx.device_repack → 设备重排
                // (源供原始 packed,DMA 上卡后 GPU 重排到 marlin 布局);
                // 否则经典键(CPU 懒物化)。
                let device = self.zs.is_some() && ctx.device_repack;
                let ops = if device {
                    // 设备重排:want 键 = 原始 packed 源键(源零重排直拷),
                    // 布局带核名与原始形状;DMA 上卡后 GPU 重排到 marlin 布局
                    qw.layout_as_device_rearrange(
                        format!("{base}.packed_raw"),
                        crate::ops::SemanticKernel::CtRepack,
                        self.out_dim,
                        self.in_dim / 8,
                        ctx,
                    )
                } else {
                    qw.layout_as(format!("{base}.qweight"), ctx)
                };
                let ops = ops.chain(sc.layout_as(format!("{base}.scales"), ctx));
                let ops = match (&self.zs, device) {
                    (Some(zs), true) => {
                        // 设备重排:scales/zeros 暂留 CPU 懒物化(数据小;
                        // 浮出为瓶颈再上 gather 核)
                        ops.chain(zs.layout_as(format!("{base}.zeros"), ctx))
                    }
                    (Some(zs), false) => ops.chain(zs.layout_as(format!("{base}.zeros"), ctx)),
                    (None, _) => ops,
                };
                ops.chain(ws.layout_as(format!("{base}.marlin_ws"), ctx))
                    .chain(ctmp.layout_as(format!("{base}.marlin_ctmp"), ctx))
            }
            _ => self.w.layout(ctx),
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::{assert_close, f32b, harvest, hf_python, parity_enabled, skip_note, st_read, st_write, tmp_path, Src};
    use crate::tensor::Dtype;
    use std::collections::HashMap;

    #[tokio::test]
    async fn native_nt_matmul_matches_host() {
        let w = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 原生 [2,3] 行主序(out=2, in=3)
        let x = vec![0.5, -1.0, 2.0];
        let mut want = vec![0.0f32; 2];
        for (o, wo) in want.iter_mut().enumerate() {
            *wo = (0..3).map(|i| x[i] * w[o * 3 + i]).sum();
        }

        let mut face = owl_cpu::CpuFace::new();
        let lin = Linear::new("w", 2, 3, QuantPlan::F16);
        let src = HashMap::from([("w".to_string(), w)]);
        crate::interpreters::eval_load(&lin, &mut face, &src, &Default::default())
            .await
            .expect("eval_load");

        let xs = TensorOps::from_host(Dtype::F32, vec![1, 3], &f32b(&x));
        let got = harvest(&mut face, &lin.forward(&xs, &ForwardCtx::minimal(1))).await;
        assert_eq!(got.len(), 2);
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() < 1e-6, "{g} vs {w}");
        }
    }

    #[tokio::test]
    async fn unloaded_slot_becomes_poison_at_boundary() {
        let mut face = owl_cpu::CpuFace::new();
        let lin = Linear::new("w", 2, 3, QuantPlan::F16);
        let xs = TensorOps::from_host(Dtype::F32, vec![1, 3], &f32b(&[1.0, 2.0, 3.0]));

        let out = lin.forward(&xs, &ForwardCtx::minimal(1));
        assert!(out.is_poisoned(), "未装载槽的声明应立即带毒(随链流动)");
        let err = crate::interpreters::eval_ops(out.step(), &mut face)
            .await
            .unwrap_err();
        assert!(format!("{err:?}").contains("未装载"), "{err:?}");

        let empty: Src = HashMap::new();
        let err = crate::interpreters::eval_load(&lin, &mut face, &empty, &Default::default())
            .await
            .unwrap_err();
        assert!(format!("{err:?}").contains("缺键"), "{err:?}");

        let lin3 = Linear::new("w", 2, 3, QuantPlan::F16);
        let bad_len = HashMap::from([("w".to_string(), vec![1.0; 5])]);
        let err = crate::interpreters::eval_load(&lin3, &mut face, &bad_len, &Default::default())
            .await
            .unwrap_err();
        assert!(format!("{err:?}").contains("元素"), "{err:?}");
    }

    /// HF parity(F.linear = x@W^T;多行 tokens=2;门控 OWL_HF_PARITY=1)
    #[tokio::test]
    async fn parity_matches_hf() {
        if !parity_enabled() {
            skip_note();
            return;
        }
        let (tokens, out_dim, in_dim) = (2usize, 2usize, 3usize);
        let x: Vec<f32> = (0..tokens * in_dim).map(|i| ((i as f32 * 0.53) - 1.2).cos()).collect();
        let w: Vec<f32> = (0..out_dim * in_dim).map(|i| (i as f32 * 0.29) - 0.5).collect();
        let xs = TensorOps::from_host(Dtype::F32, vec![tokens, in_dim], &f32b(&x));
        let ctx = ForwardCtx::minimal(tokens);

        let inp = tmp_path("linear_in");
        let outp = tmp_path("linear_out");
        st_write(&inp, &[("x", &x, vec![tokens, in_dim]), ("w", &w, vec![out_dim, in_dim])]);
        let manifest = hf_python("linear.py", &[
            inp.to_str().unwrap(), outp.to_str().unwrap(),
            &format!(r#"{{"out": {out_dim}, "in": {in_dim}}}"#),
        ]);
        eprintln!("[hf manifest] linear: {manifest}");
        let want = st_read(&outp, "y");

        for (face_tag, make) in [("cpu", None), ("gpu", Some(()))] {
            let (_, out) = if make.is_none() {
                let mut face = owl_cpu::CpuFace::new();
                let lin = Linear::new("w", out_dim, in_dim, QuantPlan::F16);
                let src = HashMap::from([("w".to_string(), w.clone())]);
                crate::interpreters::eval_load(&lin, &mut face, &src, &Default::default())
                    .await.expect("eval_load");
                ("cpu", harvest(&mut face, &lin.forward(&xs, &ctx)).await)
            } else {
                let mut gpu = crate::testkit::gpu_client().await;
                let lin = Linear::new("w", out_dim, in_dim, QuantPlan::F16);
                let src = HashMap::from([("w".to_string(), w.clone())]);
                crate::interpreters::eval_load(&lin, &mut gpu, &src, &Default::default())
                    .await.expect("eval_load");
                let o = harvest(&mut gpu, &lin.forward(&xs, &ctx)).await;
                gpu.close().await.expect("server 关机");
                ("gpu", o)
            };
            assert_close(&out, &want, 1e-5, &format!("linear-{face_tag}"));
        }
        owl_shared::file_loader::remove_file(&inp).ok();
        owl_shared::file_loader::remove_file(&outp).ok();
    }
}

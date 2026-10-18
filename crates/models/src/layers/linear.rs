//! Linear:仿射层(容器 + LoaderOps 装载 + Matmul;无 bias —— Qwen3.5
//! 全系 attention/mlp 权重无 bias)。
//!
//! 权重槽 = **原生布局直读**(2026-09-26 F5-2:检查点 [out, in] 行主序
//! 原样装载,forward matmul_nt —— cuBLAS OP_T 免费转置,host 转置路径
//! 全删;旧"装载期转置"是 naive-matmul 时代产物)。

use crate::contract::Dtype;
use crate::ops::SemanticKernel;
use crate::layers::narrow_strided;
use crate::module::{ForwardCtx, Loadable, LoaderCtx, LoaderOps, Module, QuantPlan, Weight};
use crate::formats::util::marlin_n_pack;
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
    /// **W4A8 全局因子**(f32[1];装载期由该层权重尺度算出 ——
    /// `pack_marlin_s_a8` 第二返回值 `max/4096`)。`Some` = 本层走 a8 臂
    /// (**单一事实源**:因子在则 a8 在)。
    a8_factor: Option<Weight>,
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
            a8_factor: None,
        };
        let eligible = crate::formats::util::marlin_eligible(out_dim, in_dim);
        match plan {
            QuantPlan::W4A16 if eligible => lin.quantize_g128(),
            QuantPlan::W4A16Awq if eligible => lin.quantize_awq32(),
            QuantPlan::W4A8Awq if eligible => lin.quantize_awq_a8(),
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
        // 构造期前置(**响亮**,release 也拦 —— 2026-10-17 审查 §五轮-C2):
        // 唯一调用点(`new` 的 match)已用同一谓词门控,这里是私有函数自己的
        // 不变式;原先 `debug_assert` ⇒ release 静默放行(将来新增调用点漏门 =
        // 按非法形态建 plan)。
        assert!(
            crate::formats::util::marlin_eligible(self.out_dim, self.in_dim),
            "g128: marlin 形态不满足(out {}, in {}),构造点漏了 eligible 门",
            self.out_dim,
            self.in_dim
        );
        let g = 128usize;
        let n_pack = marlin_n_pack(self.out_dim); // 直通(eligible 保证 n%256==0)
        let key: &'static str = self.w.key();
        self.qw = Some(Weight::new_typed_u32(
            key,
            vec![self.in_dim / 16, n_pack * 16 / 8],
        ));
        self.sc = Some(Weight::new_typed_scales(
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
        // 构造期前置(**响亮**,release 也拦 —— 2026-10-17 审查 §五轮-C2):
        // 唯一调用点(`new` 的 match)已用同一谓词门控,这里是私有函数自己的
        // 不变式;原先 `debug_assert` ⇒ release 静默放行(将来新增调用点漏门 =
        // 按非法形态建 plan)。
        assert!(
            crate::formats::util::marlin_eligible(self.out_dim, self.in_dim),
            "awq-g32: marlin 形态不满足(out {}, in {}),构造点漏了 eligible 门",
            self.out_dim,
            self.in_dim
        );
        let g = 32usize;
        let n_pack = marlin_n_pack(self.out_dim);
        let key: &'static str = self.w.key();
        self.qw = Some(Weight::new_typed_u32(
            key,
            vec![self.in_dim / 16, n_pack * 16 / 8],
        ));
        self.sc = Some(Weight::new_typed_scales(
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

    /// **W4A8 × AWQ 定形**(S2;构造期私有):与 `quantize_awq32` **同尺寸**
    /// (B/scales/zp 都在同一套 tile 尺寸下换布局 —— 字节数不变),另加
    /// `a8_factor` f32[1]。
    ///
    /// 容量说明:`a_scales`(per-token 尺度)**不在这里** —— 它是**每轮
    /// 按调用形状**的瞬态(`TensorOps::zeros(F32, [m,1])`,引擎按形状分配),
    /// 故层里零容量常量、零预留(装 8192 行也只是一次 32KB memset)。
    fn quantize_awq_a8(&mut self) {
        // 构造期前置(**响亮**,release 也拦 —— 2026-10-17 审查 §五轮-C2):
        // 唯一调用点(`new` 的 match)已用同一谓词门控,这里是私有函数自己的
        // 不变式;原先 `debug_assert` ⇒ release 静默放行(将来新增调用点漏门 =
        // 按非法形态建 plan)。
        assert!(
            crate::formats::util::marlin_eligible(self.out_dim, self.in_dim),
            "w4a8: marlin 形态不满足(out {}, in {}),构造点漏了 eligible 门",
            self.out_dim,
            self.in_dim
        );
        let g = 32usize;
        let n_pack = marlin_n_pack(self.out_dim);
        let key: &'static str = self.w.key();
        self.qw = Some(Weight::new_typed_u32(
            key,
            vec![self.in_dim / 16, n_pack * 16 / 8],
        ));
        // scales_a8 线 dtype = F16 槽(int16 归一位型;与 a16 同尺寸)
        self.sc = Some(Weight::new_typed_scales(
            key,
            vec![self.in_dim / g, n_pack],
        ));
        self.zs = Some(Weight::new_typed_u32(
            key,
            vec![self.in_dim / g, n_pack / 8],
        ));
        self.ws = Some(Weight::new_typed_u32(key, vec![crate::ops::marlin_ws_elems(n_pack)]));
        self.ctmp = Some(Weight::new_typed_u32(key, vec![1]));
        // 全局因子(装载器填;核侧设备标量块)
        self.a8_factor = Some(Weight::new_typed(key, vec![1], Dtype::F32));
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
        // ---- W4A8 × AWQ 臂(S2):激活量化 → a8 GEMM ----
        //
        // 与 a16 臂**同源不同布局**;两处本仓特有的形态:
        // ① `a_scales` = **每轮瞬态**(`TensorOps::zeros`)—— 引擎按调用形状
        //    分配(m 行 ⇒ m 个 f32),故无容量常量/无预留、无跨调用竞争;
        // ② 全局因子 = **设备标量块**(`a8_factor`,装载器填)—— 量化核直接读它,
        //    零 host 回路零额外发射。
        if let (Some(g), Some(qw), Some(sc), Some(zs), Some(ws), Some(ctmp), Some(fac)) = (
            self.quant_group,
            &self.qw,
            &self.sc,
            &self.zs,
            &self.ws,
            &self.ctmp,
            &self.a8_factor,
        ) {
            let m: usize = xs.shape()[..xs.shape().len() - 1].iter().product();
            let n_pack = marlin_n_pack(self.out_dim);
            let a_scales = TensorOps::zeros(Dtype::F32, vec![m, 1]);
            let q = crate::ops::quant_int8_row(xs, &fac.decl(), &a_scales, m, self.in_dim);
            // 槽序(w4a8 表)= a, b, out, scales, zeros, a_scales, ws, c_tmp,
            // m, k, n, groupsize(out 由装配器按 O 槽插入 —— 声明驱动)
            let gemm = TensorOps::call(SemanticKernel::MarlinW4A8Awq)
                .arg(&q)
                .arg(&qw.decl())
                .arg(&sc.decl())
                .arg(&zs.decl())
                .arg(&a_scales)
                .arg(&ws.decl())
                .arg(&ctmp.decl())
                .arg_usize(m)
                .arg_usize(self.in_dim)
                .arg_usize(n_pack)
                .arg_usize(g)
                .with_shape(Dtype::F16, vec![m, n_pack]);
            return narrow_strided(&gemm, m, n_pack, 0, self.out_dim, vec![m, self.out_dim]);
        }
        if let (Some(g), Some(qw), Some(sc), Some(ws), Some(ctmp)) = (
            self.quant_group, &self.qw, &self.sc, &self.ws, &self.ctmp,
        ) {
            let m: usize = xs.shape()[..xs.shape().len() - 1].iter().product();
            // n_pack = 打包档(= out_dim;pad 路线废案)
            let n_pack = marlin_n_pack(self.out_dim);
            // 激活 dtype 决定臂:F16 → GEMM_W4A16;BF16 → GEMM_W4A16_BF16
            // (E5-DF3 同日十二:草稿路径 BF16 化,scales 保持 F16 装载,
            // a/c=BF16 的 marlin 实例;s_type = F16 两族一致)
            // **臂由装载事实决定**(2026-10-16 定谳):尺度槽的声明 dtype = 装载期
            // 按模型 dtype 钉的(bf16 模型 ⇒ BF16,见 `Weight::new_typed_scales`)。
            // 激活若与它不符(f16 探针 / 混合臂),**显式铸**过去 —— 绝不能落进
            // “另一条臂”按错口径读尺度:C 侧 `s_type = c_type`,实测 bf16 尺度被
            // f16 臂按 f16 读 ⇒ 单块幅度 ~15×、端到端残留 7%(立案 §P)。
            let sc_dt = sc.dtype();
            let xs_aligned = if xs.dtype == sc_dt {
                None
            } else {
                Some(crate::layers::dflash2::ensure_dt(xs, sc_dt))
            };
            let xs: &TensorOps = xs_aligned.as_ref().unwrap_or(xs);
            let bf16_act = xs.dtype == Dtype::BF16;
            // marlin 臂(2026-10-12 三形态归一 Call):语义动作声明,
            // 名字路由归 driver(zs 在场 = AWQ 7 父;dt = BF16 → bf16 臂)
            let marlin = match self.zs.as_ref() {
                Some(z) => TensorOps::call(SemanticKernel::MarlinW4A16)
                    .arg(xs)
                    .arg(&qw.decl())
                    .arg(&sc.decl())
                    .arg(&z.decl())
                    .arg(&ws.decl())
                    .arg(&ctmp.decl()),
                None => TensorOps::call(SemanticKernel::MarlinW4A16)
                    .arg(xs)
                    .arg(&qw.decl())
                    .arg(&sc.decl())
                    .arg(&ws.decl())
                    .arg(&ctmp.decl()),
            };
            let marlin = marlin
                .arg_usize(m)
                .arg_usize(self.in_dim)
                .arg_usize(n_pack)
                .arg_usize(g)
                .with_shape(if bf16_act { Dtype::BF16 } else { Dtype::F16 }, vec![m, n_pack]);
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
        // ---- W4A8 键族(S2):与 a16 键族**并列**,同源不同布局 ----
        if let (Some(_g), Some(qw), Some(sc), Some(zs), Some(ws), Some(ctmp), Some(fac)) = (
            self.quant_group,
            &self.qw,
            &self.sc,
            &self.zs,
            &self.ws,
            &self.ctmp,
            &self.a8_factor,
        ) {
            let base = self.w.key();
            let device = ctx.device_repack;
            let ops = if device {
                // 设备重排(生产默认):源 packed 直拷上卡 → `owl_ct_repack_a8_u32`
                // 出 a8 布局 B(与 a16 共用 `load_direct_device_rearrange` 快路)
                qw.layout_as_device_rearrange(
                    format!("{base}.packed_raw"),
                    crate::ops::SemanticKernel::CtRepackA8,
                    self.out_dim,
                    self.in_dim / 8,
                    ctx,
                )
            } else {
                // CPU 懒物化(a8 键族;含因子)
                qw.layout_as(format!("{base}.qweight_a8"), ctx)
            };
            return ops
                .chain(sc.layout_as(format!("{base}.scales_a8"), ctx))
                .chain(zs.layout_as(format!("{base}.zeros_a8"), ctx))
                .chain(ws.layout_as(format!("{base}.marlin_ws"), ctx))
                .chain(ctmp.layout_as(format!("{base}.marlin_ctmp"), ctx))
                .chain(fac.layout_as(format!("{base}.a8_factor"), ctx));
        }
        match (self.quant_group, (&self.qw, &self.sc, &self.ws, &self.ctmp)) {
            (Some(_g), (Some(qw), Some(sc), Some(ws), Some(ctmp))) => {
                let base = self.w.key();
                // qweight 键:设备重排 → 原始 packed,DMA 上卡后 GPU 重排到
                // marlin 布局(AWQ kU4 与 W4A16 对称共用同核同 B 字节布局;
                // 2026-10-13 放开 zs 限制 —— W4A16 无 zeros want,原条件
                // `zs.is_some()` 把草稿钉在 CPU 两步重排 ≈24.6s);否则
                // 经典键(CPU 懒物化)。
                let device = ctx.device_repack;
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

    /// **W4A8 验收门(S2;GPU + 真检查点)**:同一条真实 AWQ 层,两条 plan
    /// (`W4A16Awq` ↔ `W4A8Awq`)**同输入对拍** —— 走全链:装载(a8 键族 /
    /// 设备重排 a8 核)→ `Linear::forward`(激活 int8 量化 + kS8×kU4 GEMM)→ 回读。
    ///
    /// 判据 = **噪声/信号 RMS**(a8 是另一档量化:per-token int8 激活 + int16
    /// 权重尺度 ⇒ 不逐位;a16 是 f16 激活 + f16 尺度)。阈值 1e-2 远超理论
    /// 偏差(激活 1/(2·127) ≈ 4e-3 + 尺度 1.2e-4),但足以抓住**布局/接线级**
    /// 错误(拿错 B 表、a_scales 没接上、因子没折——量级都在 10⁻¹ 以上)。
    ///
    /// 跑法:`OWL_TEST_DEVICE=1 ckpt:27b-awq=<cyankiwi 27B> cargo test -p owl-models
    ///       --lib gpu_w4a8_matches_w4a16_on_real_layer -- --nocapture`
    #[tokio::test]
    async fn gpu_w4a8_matches_w4a16_on_real_layer() {
        use crate::module::QuantPlan;
        use crate::testkit::{gpu_client, gpu_enabled};
        if !gpu_enabled() {
            eprintln!("skip: OWL_TEST_DEVICE 未设");
            return;
        }
        let Some(dir) = owl_shared::testkit::resource("ckpt:27b-awq") else {
            eprintln!("skip: ckpt:27b-awq 未设(cyankiwi 27B 检查点目录)");
            return;
        };
        // 真层:layer0.mlp.gate_proj(17408×5120)——两档都 marlin_eligible。
        // 局部键 = "gate_proj"(层内惯例);全键由 `map_keys` 在**声明期**改写
        // (与整模路同一条 `KeyConvention` 语义 —— 这里手写前缀即可)。
        let (local, prefix) = ("gate_proj", "model.language_model.layers.0.mlp.");
        let (out_dim, in_dim, m) = (17408usize, 5120usize, 8usize);
        struct Prefixed<'a>(&'a Linear, &'a str);
        impl crate::module::Loadable for Prefixed<'_> {
            fn layout(&self, ctx: &LoaderCtx) -> crate::module::LoaderOps {
                let p = self.1.to_string();
                self.0.layout(ctx).map_keys(move |k| format!("{p}{k}"))
            }
        }
        // 确定性输入(r16 对齐:进 f16 域前先舍入,免"装载域 vs 输入域"两套口径)
        let r16 = |x: f32| half::f16::from_f32(x).to_f32();
        let xs: Vec<f32> = (0..m * in_dim).map(|i| r16(((i as f32) * 0.013).sin() * 0.8)).collect();
        let xb: Vec<u8> = xs.iter().flat_map(|f| half::f16::from_f32(*f).to_le_bytes()).collect();

        let mut got: Vec<Vec<f32>> = Vec::new();
        for plan in [QuantPlan::W4A16Awq, QuantPlan::W4A8Awq] {
            let lin = Linear::new(local, out_dim, in_dim, plan);
            let comp = Prefixed(&lin, prefix);
            let mut gpu = gpu_client().await;
            let opts = crate::formats::load::LoadOpts::new(&gpu, Dtype::F16);
            let manifest = crate::formats::load::load_from_dir(
                std::path::Path::new(&dir),
                &mut gpu,
                &comp,
                opts,
            )
            .await
            .unwrap_or_else(|e| panic!("{plan:?} 装载失败: {e}"));
            let keys: Vec<String> = manifest.entries().iter().map(|e| e.key.clone()).collect();
            eprintln!("[{plan:?}] 装载 {} 键", keys.len());
            if matches!(plan, QuantPlan::W4A8Awq) {
                for suf in [".scales_a8", ".zeros_a8", ".a8_factor"] {
                    assert!(
                        keys.iter().any(|k| k.ends_with(suf)),
                        "a8 档缺键 {suf};实得 {keys:?}"
                    );
                }
            }
            let x = TensorOps::from_host(Dtype::F16, vec![m, in_dim], &xb);
            let decl = lin.forward(&x, &crate::module::ForwardCtx::minimal(m));
            let out = crate::testkit::harvest_f16(&mut gpu, &decl).await;
            gpu.close().await.expect("关机");
            got.push(out);
        }
        // ---- 判据:噪声/信号 RMS(a8 vs a16)----
        let (a16, a8) = (&got[0], &got[1]);
        assert_eq!(a16.len(), a8.len());
        let n = a16.len();
        let rms = (a16.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / n as f64).sqrt();
        let d: f64 = a16
            .iter()
            .zip(a8)
            .map(|(x, y)| ((x - y) as f64).abs())
            .sum::<f64>()
            / n as f64;
        let noise = d / rms.max(1e-9);
        let nan = a8.iter().filter(|v| !v.is_finite()).count();
        eprintln!(
            "[w4a8-ab] 真实层 {prefix}{local} m={m}: 噪声/信号 RMS = {noise:.5}(a16 信号 RMS={rms:.4});a8 非有限 {nan}"
        );
        assert_eq!(nan, 0, "a8 输出含非有限值");
        assert!(
            noise < 1e-2,
            "a8 与 a16 偏差过大({noise:.5})—— 布局/接线级错误距此量级 ≫ 10⁻¹"
        );
    }

    /// **W4A8 接线门(零 GPU)**:plan → 声明面(a8 键族 want)→ 前向面
    /// (量化 → a8 GEMM 的节点链)。这条门锁的是"层把 a8 档接对了",
    /// 数值正确性在 kernels 侧已有(§S1d 对拍 + a8 repack 逐位门)。
    #[test]
    fn w4a8_plan_declares_a8_keys_and_emits_quant_then_gemm() {
        use crate::module::{Loadable, LoaderCtx, QuantPlan};
        // marlin_eligible:out%256==0 && k%128==0
        let lin = Linear::new("l.w", 256, 256, QuantPlan::W4A8Awq);

        // ---- ① 声明面:键族与 dtype(设备重排档 = 生产默认) ----
        let ctx = LoaderCtx { device_repack: true, ..Default::default() };
        let ops = lin.layout(&ctx);
        let wants: Vec<(String, Dtype)> = ops
            .wants()
            .iter()
            .map(|w| (w.key.clone(), w.dtype))
            .collect();
        let has = |suf: &str, dt: Dtype| {
            wants.iter().any(|(k, d)| k.ends_with(suf) && *d == dt)
        };
        assert!(
            has(".packed_raw", Dtype::U32),
            "设备重排档:B 走原始 packed 源键(U32 位型;经 a8 repack 核);实得 {wants:?}"
        );
        assert!(has(".scales_a8", Dtype::F16), "scales_a8 = int16 位型塞 F16 槽");
        assert!(has(".zeros_a8", Dtype::U32), "zeros_a8");
        assert!(has(".a8_factor", Dtype::F32), "全局因子(设备标量块)");
        assert!(!has(".qweight_a8", Dtype::U32), "设备档不得同时要 CPU 版 B(白跑全模重排)");
        // CPU 档换键
        let ctx_cpu = LoaderCtx { device_repack: false, ..Default::default() };
        let cpu_keys: Vec<String> = lin.layout(&ctx_cpu).wants().iter().map(|w| w.key.clone()).collect();
        assert!(cpu_keys.iter().any(|k| k.ends_with(".qweight_a8")), "CPU 档要 qweight_a8");

        // ---- ② 前向面:量化(quant_int8_row) → a8 GEMM,7 父 ----
        let xs = TensorOps::zeros(Dtype::F16, vec![8, 256]);
        let out = lin.forward_inner(&xs);
        // 外层是窄视图(与 a16 臂同款的 out_dim 裁切)
        let gemm = out.parents.first().expect("窄视图的父 = GEMM").clone();
        match &gemm.op {
            crate::ops::Op::Call { op, .. } => assert_eq!(
                op.op_id().0,
                "marlin.w4a8_awq",
                "a8 臂必须发 w4a8 语义动作"
            ),
            other => panic!("期望 Call(MarlinW4A8Awq),实得 {other:?}"),
        }
        assert_eq!(gemm.parents.len(), 7, "a8 槽 7 父:q/b/scales/zeros/a_scales/ws/c_tmp");
        // 第一父 = 量化输出(其 op 必须是量化动作)
        let q = gemm.parents[0].clone();
        match &q.op {
            crate::ops::Op::Call { op, .. } => assert_eq!(op.op_id().0, "elems.quant_int8_row_f16"),
            other => panic!("期望量化 Call,实得 {other:?}"),
        }
        assert_eq!(q.dtype, Dtype::I8, "量化出 int8(有符号)");
        // 量化 3 父:x(f16 激活) / mul(设备标量因子) / scales(每轮瞬态)
        assert_eq!(q.parents.len(), 3, "量化槽 3 父:x / mul / scales");
        assert!(
            matches!(q.parents[2].op, crate::ops::Op::Zeros),
            "a_scales = 引擎按形状分配的瞬态(Op::Zeros);实得 {:?}",
            q.parents[2].op
        );
    }

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

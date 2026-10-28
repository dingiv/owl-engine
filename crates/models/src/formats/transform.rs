//! **目标格式变换面**(`docs/arch/装载管线-源精度x目标格式-设计.md` §3.2/§4)。
//!
//! 装载的三轴正交化之后,这里是第三件:**把「源字节」变成「目标格式的 N 个槽」**。
//! 一个 [`Transform`] 只回答四件事 —— 我是谁([`Transform::id`])、我出几件货
//! ([`Transform::slots`],装载前的暂存账按它求和)、我在哪算([`Transform::place`])、
//! 以及怎么算([`Transform::run`])。
//!
//! **为什么必须有 1:N**:NVFP4 是「一源二槽 + 一个 host 标量」(f16 权重 →
//! `q` + `sf` + `gb`)。旧的 `Layout` 派发是 1:1 模型,表达不了 ⇒ 每加一个
//! 目标格式都要在 `load.rs` 里新开一个"顺手多交付一件"的分支。
//!
//! **范围(今天)**:`Transform` 收的是**真格式变换**(源精度 → 另一种数值格式)。
//! `Direct`/`Transposed`/`DeviceRearrange` 是**搬运/重排**(字节不变或纯换序),
//! 仍走 `load.rs` 的既有三臂 —— 设计 §8 的 P3 才把它们包成同形状
//! (`Copy`/`HostTranspose`/`DeviceOp`),那一步是纯机械收敛,不夹带数值变化。
//!
//! 纪律:**变换只吃 [`decode_to_f32`] 的产物**,不许自己按"源大概是 f16"猜字节;
//! 产物长度/形状由 [`Transform::slots`] 声明,装载面按声明验账(不许句式不一致)。

use crate::contract::{Dtype, ModelError};
use crate::formats::source::{decode_to_f32, SrcDtype};

/// 一个槽的落地声明(**目标侧**形状/dtype 由变换自己算)。
///
/// `name` 是**诊断名**(`q`/`sf`/…),进日志与 manifest;条件语句(账长校验、
/// 显存账)一律按 `dtype × shape` 求,不按名字分支。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlotDecl {
    pub name: &'static str,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
}

impl SlotDecl {
    /// 槽字节数(`numel × 元素宽`)
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.shape.iter().product::<usize>() * self.dtype.size_bytes()
    }
}

/// 变换在**哪里**算(装载面据此选臂:Host = 纯主机 + 一次 htod;Device = 走核)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Place {
    /// 主机(源字节本来就在 host;装载期一次性,无需 stream)
    Host,
    /// 设备(已有设备核;数据大到 host 往返不划算)
    Device,
}

/// 变换的执行语境(装载面填;**只有装载面能造**,变换只读/产物写)。
pub struct TransformCtx<'a> {
    /// 源字节(**原始精度**,未解码;`src_dt` 说它是什么)
    pub src_bytes: &'a [u8],
    /// 源精度事实(来自文件元数据,禁止由 plan 反推)
    pub src_dt: SrcDtype,
    /// 源的行数(目标 N 轴)
    pub rows: usize,
    /// 源的列数(目标 K 轴)
    pub cols: usize,
    /// 产物缓冲(与 [`Transform::slots`] **同序同长**;装载面按声明预分配)
    pub outs: &'a mut [Vec<u8>],
    /// host 标量回填(与槽无关的旁路产物,如 NVFP4 的 `gb`;空 = 本变换不用)
    pub scalars: &'a mut [f32],
}

impl TransformCtx<'_> {
    /// 源元素数(声明口径;与 `src_bytes` 实测对齐由变换自检)
    #[must_use]
    pub const fn elems(&self) -> usize {
        self.rows * self.cols
    }
}

/// 目标格式变换(新增一个目标格式 = 新增一个实现 + 一个 `Layout` 变体)
pub trait Transform {
    /// 稳定标识(日志/manifest/错文)
    fn id(&self) -> &'static str;
    /// 产物声明(**装载前**就能算:暂存账、槽分配、账长门都按它)
    fn slots(&self) -> Result<Vec<SlotDecl>, ModelError>;
    /// 在哪算
    fn place(&self) -> Place;
    /// 执行(输入 `src_bytes`+`src_dt`,输出各槽字节 + 可选 host 标量)
    fn run(&self, cx: &mut TransformCtx<'_>) -> Result<(), ModelError>;
}

// ============================================================================
// §1 NVFP4 权重打包(host;`q` + `sf` 两槽 + `gb` 标量)
// ============================================================================

/// **NVFP4 权重打包**(f16 源 → e2m1 数据 + e4m3/16 块缩放)。
///
/// 走 **Host** 的理由(设计 §4):`owl_nvfp4_pack_b` 本就是纯主机函数,而源字节
/// 流式读文件时天然在 host ⇒ 不新增算子(否则量化器还要再走一遍
/// client/server/ops 三件套)。将来要提速就换一个 `Place::Device` 实现,
/// **接口不变**。
///
/// 数值口径**单一来源**:打包与 per-tensor 全局缩放 `gb` 都由
/// [`owl_kernels::nvfp4::pack_weight_b`] 产出(同一个 `quantize` 里的
/// `amax`),装载面把返回值原样回填进层的 `gb` ⇒ 发射时 `alpha = ga·gb`
/// 两侧同源(混用另一份 amax = 静默错值)。
pub struct NvFp4Pack {
    /// 目标 N(检查点里的 out 维)
    pub rows: usize,
    /// 目标 K(检查点里的 in 维)
    pub cols: usize,
}

impl Transform for NvFp4Pack {
    fn id(&self) -> &'static str {
        "nvfp4.pack_b"
    }

    fn slots(&self) -> Result<Vec<SlotDecl>, ModelError> {
        let (n, k) = (self.rows, self.cols);
        // 形态门(与 `Linear::quantize_nvfp4` 同一谓词;此处再拒一次是**廉价自证**:
        // 谁把不合规形状喂进来,要在装载期就响,而不是等 SF 布局写错位)
        if k % 16 != 0 || n % 8 != 0 {
            return Err(ModelError::Msg(format!(
                "NVFP4: 形状 {n}x{k} 不合规(K 需 16 对齐 = 块缩放粒度,N 需 8 对齐 = epilogue 向量化)"
            )));
        }
        Ok(vec![
            // 数据槽:每字节两个 e2m1(k 偶数在低 nibble)
            SlotDecl { name: "q", dtype: Dtype::U8, shape: vec![n, k / 2] },
            // 缩放槽:CUTLASS SFB 交错布局,字节数 = `N*K/16`(cosize 口径)
            SlotDecl { name: "sf", dtype: Dtype::U8, shape: vec![n * k / 16] },
        ])
    }

    fn place(&self) -> Place {
        Place::Host
    }

    fn run(&self, cx: &mut TransformCtx<'_>) -> Result<(), ModelError> {
        let elems = cx.elems();
        let esz = cx.src_dt.size_bytes();
        let src_elems = cx.src_bytes.len() / esz.max(1);
        if src_elems != elems {
            return Err(ModelError::Msg(format!(
                "NVFP4 打包: 源 {} 个 {} 元素 != 声明 {}x{} = {elems}",
                src_elems,
                cx.src_dt.name(),
                self.rows,
                self.cols
            )));
        }
        let slots = self.slots()?;
        let (q_bytes, sf_bytes) = (slots[0].bytes(), slots[1].bytes());

        // 1) 源 → f16 主缓冲(**按行带解码**,把 f32 中间量钉在 ~32MB:
        //    4B 的最大单键 2.5 亿参数级,整键 f32 副本会平白多占数百 MB)
        let mut b16 = vec![0u8; elems * 2];
        let band = (32usize << 20).checked_div(self.cols.max(1)).unwrap_or(1).max(1);
        let mut f32_band = vec![0f32; band.min(self.rows) * self.cols];
        let rows_per_band = (f32_band.len() / self.cols.max(1)).max(1);
        for r0 in (0..self.rows).step_by(rows_per_band) {
            let r1 = (r0 + rows_per_band).min(self.rows);
            let n_band = (r1 - r0) * self.cols;
            let (s0, s1) = (r0 * self.cols * esz, r1 * self.cols * esz);
            decode_to_f32(&cx.src_bytes[s0..s1], cx.src_dt, &mut f32_band[..n_band])?;
            crate::f16c::f32_slice_to_f16_bytes(
                &f32_band[..n_band],
                &mut b16[r0 * self.cols * 2..r1 * self.cols * 2],
            );
        }

        // 2) 打包(pack_b 的输入契约 = host `__half[N*K]` 行主序)
        let gb = {
            // 两个产物同时要可变借用 ⇒ 先按位拆(`outs` 声明同序同长,
            // 装载面已按 `slots()` 预分配;这里再验一次长度 = 廉价自证)
            let (head, tail) = cx.outs.split_at_mut(1);
            let (q, sf) = (head[0].as_mut_slice(), tail[0].as_mut_slice());
            if q.len() != q_bytes || sf.len() != sf_bytes {
                return Err(ModelError::Msg(format!(
                    "NVFP4 打包: 产物缓冲 q={}/{} sf={}/{} 与声明不符",
                    q.len(),
                    q_bytes,
                    sf.len(),
                    sf_bytes
                )));
            }
            owl_kernels::nvfp4::pack_weight_b(self.rows, self.cols, &b16, q, sf)
                .map_err(|e| ModelError::Msg(format!("NVFP4 打包失败: {e}")))?
        };

        // 3) host 标量回填(`gb` 由装载面写进层)
        if let Some(s) = cx.scalars.first_mut() {
            *s = gb;
        }
        Ok(())
    }
}

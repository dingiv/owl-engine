//! XQA verify / decode runtime(2026-10-13 C3a:**构建期 cubin + Rust host 发射器**)。
//!
//! 形态:两个宏变体资产(`xqa_verify` / `xqa_decode`,build.rs `-DGENERATE_CUBIN`
//! 编出)+ 本模块的 Rust 发射器(替换原 `cu/xqa/owl_xqa_launch.cu` 的 C 发射器)。
//! 编译期常量(ctaTile / smem / rowsPerBlock / tokensPerPage / …)**由内核侧探针核
//! 在装配期报回**(资产自描述,见 `family::xqa` 头注)⇒ Rust 侧零常量复制。
//!
//! 无内部状态:semaphores / scratch 由**调用方**给(层侧持有;图内安全
//! 责任在层侧 —— semaphores 每次发射前必须为零)。本 runtime 只做
//! 「解析 → 块指针解析 → 组装值表 → `Exec::launch` → 回执 out」。
//!
//! 两条臂共用同一套槽序/解析(`parse_verify`;两份实例化签名一致),差别只在
//! 装载的**资产变体**:
//! - `XqaRuntime`       → `xqa_verify`(SPEC_DEC=1 `M_TILESIZE=32`)
//! - `XqaDecodeRuntime` → `xqa_decode`(SPEC_DEC=1 `M_TILESIZE=16`,裸 decode q=1)

use crate::client::xqa::{parse_verify, DECODE_XQA, PREFILL_XQA, VERIFY_XQA};
use crate::contract::{Bytes, Dtype, LaunchMsg, OpError, OpId};
use crate::family::xqa::{
    KVCacheList, XqaArgs, XqaConsts, ASSET_DECODE, ASSET_VERIFY, CONSTS_SLOTS, SYM_CONSTS, SYM_KERNEL,
};
use crate::registry::{FamilyGuard, KernelFamily};
use std::sync::Arc;

/// 臂状态:内核句柄 + 资产自描述常量(装配期一次定)
pub struct XqaRuntime {
    kernel: Arc<cudarc::driver::CudaFunction>,
    consts: XqaConsts,
}

pub struct XqaDecodeRuntime {
    kernel: Arc<cudarc::driver::CudaFunction>,
    consts: XqaConsts,
}

/// 装配公共体:载资产 → 载内核与探针核 → **发一次探针核读常量** → 置 smem 上限。
///
/// 为什么探针核要真发一次:常量住内核侧的编译期表达式里(单 TU cubin 会把设备
/// 变量内部化,`cuModuleGetGlobal` 取不到)。一次 1 线程发射 + 44B D2H = 装配期
/// 一次性成本(~50µs),换 host 侧**零常量复制**。
fn assemble_common(
    res: &mut dyn crate::device::DeviceRes,
    exec: &mut crate::device::Exec,
    variant: &'static str,
    op: OpId,
) -> Result<(Arc<cudarc::driver::CudaFunction>, XqaConsts), OpError> {
    let bytes = crate::family::cubins::xqa_asset(variant).ok_or_else(|| OpError::Asset {
        op: op.0.to_string(),
        detail: format!("XQA 资产 {variant} 不存在(feature `xqa` 未开?build.rs 未编?)"),
    })?;
    let kernel = exec.load_cubin(res, variant, bytes, SYM_KERNEL, op)?;
    let probe = exec.load_cubin(res, variant, bytes, SYM_CONSTS, op)?;

    // ---- 探针核:1 线程 + 44B 输出 ----
    let out = res.alloc(CONSTS_SLOTS * 4, "xqa.consts")?;
    let vals = [crate::contract::LaunchVal::Ptr(out.ptr)];
    exec.launch(res, op, &probe, SYM_CONSTS, (1, 1, 1), (32, 1, 1), 0, &vals)?;
    // ⚠️ **回读前必须同步**:`DeviceRes::download` 走 `cuMemcpyDtoH_v2`(阻塞在
    // **NULL/legacy 默认流**上),与我们的 compute 流**不同流 ⇒ 不构成同步点**
    // ⇒ 竞态读到全零(实测:同一份资产 verify 读到 40、decode 读到 0,间歇性;
    // 装配期布局门当场拦下)。装配期路径非热路径,显式同步最省事。
    // (对照:oracle 面一律 `r.sync()` 后再 `download` —— 同款纪律)
    res.stream()?.synchronize().map_err(|e| OpError::Launch {
        op: op.0.to_string(),
        stage: crate::contract::Stage::Compute,
        detail: format!("xqa consts sync: {e:?}"),
    })?;
    let mut buf = [0u8; CONSTS_SLOTS * 4];
    res.download(out.ptr, &mut buf)?;
    let slots: Vec<u32> = buf
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let consts = XqaConsts::from_slots(&slots)?;
    // 动态 smem 上限(原 `configureKernel()` 的 cudaFuncSetAttribute)
    crate::device::Exec::set_shared_limit(&kernel, consts.smem)?;
    eprintln!(
        "[xqa] 资产自描述 {variant}: smem={} ctaTile=({}, {}) headGrp={} cta=({},{},{}) rowsPerBlock={} multiBlock={} tokensPerPage={} headElems={}",
        consts.smem, consts.cta_tile_x, consts.cta_tile_y, consts.head_grp,
        consts.cta_x, consts.cta_y, consts.cta_z, consts.rows_per_block,
        consts.allow_multi_block, consts.tokens_per_page, consts.valid_elems_per_head
    );
    Ok((kernel, consts))
}

impl XqaRuntime {
    /// **本族身份**(族名 = 线契约名;字面量住 `family::<族>`,零门)
    /// —— 声明面 `ops/<域>::FAMILY` 与执行面 `id()` 引用**同一字面量**。
    pub const ID: OpId = OpId(crate::family::xqa::VERIFY_XQA);
    // 本族算子表:由 `KernelSpec::FAMILY` 派生(单一提升点 = ops::xqa 的 static)
}

/// 发射公共体(两条臂共用;差别只在装载的资产变体)
/// 发射公共体(几何可变面单列:`single_block` = KV-split 关 = prefill 臂契约;
/// 见 [`crate::family::xqa::XqaConsts::launch_geometry_single_block`])
fn emit_geom(
    msg: &LaunchMsg,
    res: &mut dyn crate::device::DeviceRes,
    exec: &mut crate::device::Exec,
    op: &'static str,
    kernel: &Arc<cudarc::driver::CudaFunction>,
    consts: &XqaConsts,
    single_block: bool,
) -> Result<Bytes, OpError> {
    let c = parse_verify(msg)?;
    // 逐参命名解析(失败即报"哪一个参 + 哪个块 id" —— DeadBlock 定谳用)
    let r = |name: &'static str, b: &crate::client::xqa::BlockRef| {
        res.resolve(b.id).map(|p| p + b.byte_offset).map_err(|e| {
            OpError::Contract {
                op: op.into(),
                field: name,
                expect: format!("块 id={} 在本账房可解析", b.id),
                got: format!("{e}"),
            }
        })
    };
    let q = r("q", &c.q)?;
    let kc = r("k_cache", &c.k_cache)?;
    let vc = r("v_cache", &c.v_cache)?;
    let pt = r("page_table", &c.page_table)?;
    let sl = r("seq_lens", &c.seq_lens)?;
    let mk = r("mask", &c.mask)?;
    let sem = r("semaphores", &c.semaphores)?;
    let scratch = r("scratch", &c.scratch)?;
    let out = r("out", &crate::client::xqa::BlockRef { id: c.out.id, byte_offset: 0 })?;
    // F2 入参尺寸自证:scratch 是内核的**持久工作区**(KV-split partial +
    // 行归约),少分配 ⇒ 越界写邻块。精确需求随 geometry 变且住内核
    // `MemSegmenter`(host 推 = 复制内核常量)⇒ 只守下限,公式/拒语住
    // `family::xqa`。
    let have_scr =
        res.resolve_len(c.scratch.id)?.saturating_sub(c.scratch.byte_offset as usize);
    crate::family::xqa::scratch_size_check(have_scr).map_err(|detail| OpError::Contract {
        op: op.to_string(),
        field: "scratch",
        expect: format!("块账长 ≥ {} B(下限)", crate::family::xqa::XQA_SCRATCH_FLOOR_BYTES),
        got: detail,
    })?;

    // 图元(常量全部来自资产自描述;Rust 侧零复制)
    let (grid, block, smem) = if single_block {
        consts.launch_geometry_single_block(c.hkv as u32, c.q_seq_len as u32)
    } else {
        consts.launch_geometry(
            c.sm_count as u32,
            /*batch=*/ 1,
            c.hkv as u32,
            c.max_seq_len as u32,
            c.q_seq_len as u32,
        )
    };
    let args = XqaArgs {
        q_seq_len: c.q_seq_len as u32,
        nb_k_heads: c.hkv as u32,
        q_cu_seq_lens: 0,
        q_scale: c.q_scale,
        q_scale_ptr: 0,
        output: out,
        q,
        mask: mk,
        attention_sinks: 0,
        cache: KVCacheList {
            k_cache: kc,
            v_cache: vc,
            page_list: pt,
            seq_len_list: sl,
            max_nb_pages_per_seq: consts.max_nb_pages_per_seq(c.max_seq_len as u32),
            _pad: 0,
        },
        batch_size: 1,
        kv_scale: 1.0, // fp8 KV 无 per-tensor scale
        kv_scale_ptr: 0,
        stride_page: consts.stride_in_heads(c.stride_page as u64),
        stride_token: consts.stride_in_heads(c.stride_token as u64),
        stride_head: consts.stride_in_heads(c.stride_head as u64),
        semaphores: sem,
        scratch,
    };
    exec.launch(res, OpId(op), kernel, crate::family::xqa::SYM_KERNEL, grid, block, smem, &args.vals(consts.head_grp))?;
    Ok(Bytes::new(c.out.id, msg.out_elems, Dtype::F16, 2))
}

impl KernelFamily for XqaRuntime {
    fn id(&self) -> OpId {
        Self::ID
    }



    /// 装配:载 `xqa_verify` 资产 + 探针核读常量(workspace 仍由调用方给)
    fn assemble(
        res: &mut dyn crate::device::DeviceRes,
        exec: &mut crate::device::Exec,
    ) -> Result<Box<dyn KernelFamily>, OpError> {
        let (kernel, consts) = assemble_common(res, exec, ASSET_VERIFY, OpId(VERIFY_XQA))?;
        Ok(FamilyGuard::boxed(XqaRuntime { kernel, consts }))
    }

    fn run(
        &mut self,
        msg: &LaunchMsg,
        res: &mut dyn crate::device::DeviceRes,
        exec: &mut crate::device::Exec,
    ) -> Result<Bytes, OpError> {
        emit_geom(msg, res, exec, VERIFY_XQA, &self.kernel, &self.consts, false)
    }
}

/// **prefill(chunked q=T)臂 runtime**(自 `feat/prefill-graph` 合入,2026-10-15)。
///
/// 与 [`XqaRuntime`] **同一份资产/同一槽序**(`M_TILESIZE=32` 的 SPEC_DEC 单遍核
/// 在 q=T 时就是 chunked prefill 的因果块),唯一分叉 = **几何强制单块**
/// (`nbSubSeq = 1`):`grid.y = hkv·ceil(q·headGrp/rowsPerBlock)` 已把 SM 喂饱
/// (q=512 ⇒ 384 CTAs),实测再开 KV-split 时间一字不变 ⇒ 本臂结构上**不需要**
/// scratch/semaphores(`isMultiBlock` 恒假;顺带绕开 semaphores 尺寸雷,见
/// `cu/xqa/README.md`)。该不变量**只住这里**(调用方拿不到第二处决定权)。
pub struct XqaPrefillRuntime {
    kernel: Arc<cudarc::driver::CudaFunction>,
    consts: XqaConsts,
}

impl XqaPrefillRuntime {
    /// **本族身份**(族名 = 线契约名;字面量住 `family::xqa`,零门)
    pub const ID: OpId = OpId(crate::family::xqa::PREFILL_XQA);
}

impl KernelFamily for XqaPrefillRuntime {
    fn id(&self) -> OpId {
        Self::ID
    }

    /// 装配:载 **verify 资产**(同一份实例化)
    fn assemble(
        res: &mut dyn crate::device::DeviceRes,
        exec: &mut crate::device::Exec,
    ) -> Result<Box<dyn KernelFamily>, OpError> {
        let (kernel, consts) = assemble_common(res, exec, ASSET_VERIFY, OpId(PREFILL_XQA))?;
        Ok(FamilyGuard::boxed(XqaPrefillRuntime { kernel, consts }))
    }

    fn run(
        &mut self,
        msg: &LaunchMsg,
        res: &mut dyn crate::device::DeviceRes,
        exec: &mut crate::device::Exec,
    ) -> Result<Bytes, OpError> {
        emit_geom(msg, res, exec, PREFILL_XQA, &self.kernel, &self.consts, true)
    }
}

impl XqaDecodeRuntime {
    /// **本族身份**(族名 = 线契约名;字面量住 `family::xqa`,零门)
    /// —— 声明面 `ops::xqa::XqaDecode::FAMILY` 与本处引用**同一字面量**。
    pub const ID: OpId = OpId(crate::family::xqa::DECODE_XQA);
}

impl KernelFamily for XqaDecodeRuntime {
    fn id(&self) -> OpId {
        Self::ID
    }



    fn assemble(
        res: &mut dyn crate::device::DeviceRes,
        exec: &mut crate::device::Exec,
    ) -> Result<Box<dyn KernelFamily>, OpError> {
        let (kernel, consts) = assemble_common(res, exec, ASSET_DECODE, OpId(DECODE_XQA))?;
        Ok(FamilyGuard::boxed(XqaDecodeRuntime { kernel, consts }))
    }

    fn run(
        &mut self,
        msg: &LaunchMsg,
        res: &mut dyn crate::device::DeviceRes,
        exec: &mut crate::device::Exec,
    ) -> Result<Bytes, OpError> {
        emit_geom(msg, res, exec, DECODE_XQA, &self.kernel, &self.consts, false)
    }
}

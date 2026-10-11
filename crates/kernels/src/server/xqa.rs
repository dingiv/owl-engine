//! XQA verify / decode runtime(StaticLib 链路;host 函数发射)。
//!
//! 无内部状态:semaphores / scratch 由**调用方**给(层侧持有;图内安全
//! 责任在层侧 —— semaphores 每次发射前必须为零)。本 runtime 只做
//! 「解析 → 块指针解析 → `owl_xqa_launch*` → 回执 out」。
//!
//! 两条臂共用同一套槽序/解析(`parse_verify`;两份实例化签名一致),
//! 差别只在宿主发射函数:
//! - `XqaRuntime`       → `owl_xqa_launch`        (SPEC_DEC=1 `M_TILESIZE=32`,verify)
//! - `XqaDecodeRuntime` → `owl_xqa_launch_decode` (SPEC_DEC=1 `M_TILESIZE=16`,裸 decode q=1)

use crate::client::xqa::{parse_verify, DECODE_XQA, VERIFY_XQA};
use crate::contract::{Bytes, LaunchMsg, OpError, OpId};
use crate::family::xqa::{owl_xqa_launch, owl_xqa_launch_decode};
use crate::registry::{FamilyGuard, KernelFamily};

pub struct XqaRuntime;
pub struct XqaDecodeRuntime;

impl XqaRuntime {
    /// 本族算子表(**引用 `ops::xqa` 的命名 static** —— 单一提升点)
    const OPS: &'static [&'static crate::spec::KernelMeta] = &[&crate::ops::xqa::XQA_VERIFY];
}
impl XqaDecodeRuntime {
    const OPS: &'static [&'static crate::spec::KernelMeta] = &[&crate::ops::xqa::XQA_DECODE];
}

/// 发射公共体(`decode=false` → verify 实例化,`true` → decode 实例化)。
fn emit(
    msg: &LaunchMsg,
    res: &mut dyn crate::device::DeviceRes,
    op: &'static str,
    decode: bool,
) -> Result<Bytes, OpError> {
    let c = parse_verify(msg)?;
    // 逐参命名解析(失败即报"哪一个参 + 哪个块 id" —— DeadBlock 定谳用)
    let r = |name: &'static str, b: &crate::client::xqa::BlockRef| {
        res.resolve(&Bytes { id: b.id, len: 0 }).map(|p| p + b.byte_offset).map_err(|e| {
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
    let stream = res.stream()?;
    // 公共实参(两实例化签名逐字一致)
    let args = (
        c.sm_count as u32,
        c.hkv as u32,
        0u32, // sliding window 关(本实例化 SLIDING_WINDOW=0)
        c.q_scale,
        out,
        q,
        kc,
        vc,
        pt,
        c.max_seq_len as u32,
        sl,
        1u32,  // batch = 1(owl 单会话)
        1.0f32, // kv_scale(fp8 KV 无 per-tensor scale)
        c.q_seq_len as u32,
        mk,
        sem,
        scratch,
        c.stride_page as u64,
        c.stride_token as u64,
        c.stride_head as u64,
    );
    let rc = unsafe {
        let st = stream.cu_stream() as *mut std::ffi::c_void;
        let f: unsafe extern "C" fn(
            u32, u32, u32, f32,
            *mut std::ffi::c_void, *const std::ffi::c_void,
            *mut std::ffi::c_void, *mut std::ffi::c_void,
            *const i32, u32, *const u32, u32, f32, u32, *const u32,
            *mut u32, *mut std::ffi::c_void,
            u64, u64, u64, *mut std::ffi::c_void,
        ) -> i32 =
            if decode { owl_xqa_launch_decode } else { owl_xqa_launch };
        f(
            args.0,
            args.1,
            args.2,
            args.3,
            args.4 as *mut std::ffi::c_void,
            args.5 as *const std::ffi::c_void,
            args.6 as *mut std::ffi::c_void,
            args.7 as *mut std::ffi::c_void,
            args.8 as *const i32,
            args.9,
            args.10 as *const u32,
            args.11,
            args.12,
            args.13,
            args.14 as *const u32,
            args.15 as *mut u32,
            args.16 as *mut std::ffi::c_void,
            args.17,
            args.18,
            args.19,
            st,
        )
    };
    if rc != 0 {
        return Err(OpError::Launch {
            op: op.into(),
            stage: crate::contract::Stage::Compute,
            detail: format!(
                "{} 参数拒绝 rc={rc}(q_seq_len={} hd={} page={} max_seq={} sm={})",
                if decode { "owl_xqa_launch_decode" } else { "owl_xqa_launch" },
                c.q_seq_len,
                c.hd,
                c.page,
                c.max_seq_len,
                c.sm_count
            ),
        });
    }
    Ok(Bytes { id: c.out.id, len: msg.out_elems })
}

impl KernelFamily for XqaRuntime {
    fn id(&self) -> OpId {
        OpId(VERIFY_XQA)
    }

    fn ops(&self) -> &'static [&'static crate::spec::KernelMeta] {
        Self::OPS
    }

    fn linkage(&self) -> crate::contract::Linkage {
        crate::contract::Linkage::StaticLib
    }

    /// 无装配(workspace 由调用方给 —— 层侧持有,图内安全责任在层侧)
    fn assemble(
        _res: &mut dyn crate::device::DeviceRes,
        _exec: &mut crate::device::Exec,
    ) -> Result<Box<dyn KernelFamily>, OpError> {
        Ok(FamilyGuard::boxed(XqaRuntime))
    }

    fn run(
        &mut self,
        msg: &LaunchMsg,
        res: &mut dyn crate::device::DeviceRes,
        _exec: &mut crate::device::Exec,
    ) -> Result<Bytes, OpError> {
        emit(msg, res, VERIFY_XQA, false)
    }
}

impl KernelFamily for XqaDecodeRuntime {
    fn id(&self) -> OpId {
        OpId(DECODE_XQA)
    }

    fn ops(&self) -> &'static [&'static crate::spec::KernelMeta] {
        Self::OPS
    }

    fn linkage(&self) -> crate::contract::Linkage {
        crate::contract::Linkage::StaticLib
    }

    fn assemble(
        _res: &mut dyn crate::device::DeviceRes,
        _exec: &mut crate::device::Exec,
    ) -> Result<Box<dyn KernelFamily>, OpError> {
        Ok(FamilyGuard::boxed(XqaDecodeRuntime))
    }

    fn run(
        &mut self,
        msg: &LaunchMsg,
        res: &mut dyn crate::device::DeviceRes,
        _exec: &mut crate::device::Exec,
    ) -> Result<Bytes, OpError> {
        emit(msg, res, DECODE_XQA, true)
    }
}

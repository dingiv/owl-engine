//! XQA verify/decode runtime(StaticLib 链路;host 函数发射)。
//!
//! 无内部状态:semaphores / scratch 由**调用方**给(层侧持有;图内安全
//! 责任在层侧 —— semaphores 每次发射前必须为零)。本 runtime 只做
//! 「解析 → 块指针解析 → `owl_xqa_launch` → 回执 out」。

use crate::contract::{Bytes, LaunchMsg, OpError, OpId};
use crate::client::xqa::{parse_verify, VERIFY_XQA};
use crate::family::xqa::owl_xqa_launch;
use crate::registry::KernelSpec;

pub struct XqaRuntime;

impl KernelSpec for XqaRuntime {
    fn id(&self) -> OpId {
        OpId(VERIFY_XQA)
    }

    fn names(&self) -> Vec<&'static str> {
        vec![VERIFY_XQA]
    }

    fn linkage(&self) -> crate::contract::Linkage {
        crate::contract::Linkage::StaticLib
    }

    fn validate(&self) -> Result<(), OpError> {
        Ok(())
    }

    /// 无装配(workspace 由调用方给 —— 层侧持有,图内安全责任在层侧)
    fn init(
        &mut self,
        _res: &mut dyn crate::device::DeviceRes,
        _exec: &mut crate::device::Exec,
    ) -> Result<(), OpError> {
        Ok(())
    }

    fn run(
        &mut self,
        msg: &LaunchMsg,
        res: &mut dyn crate::device::DeviceRes,
        _exec: &mut crate::device::Exec,
    ) -> Result<Bytes, OpError> {
        let c = parse_verify(msg)?;
        // 逐参命名解析(失败即报"哪一个参 + 哪个块 id" —— DeadBlock 定谳用)
        let r = |name: &'static str, b: &crate::client::xqa::BlockRef| {
            res.resolve(&Bytes { id: b.id, len: 0 }).map(|p| p + b.byte_offset).map_err(|e| {
                OpError::Contract {
                    op: VERIFY_XQA.into(),
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
        let rc = unsafe {
            owl_xqa_launch(
                c.sm_count as u32,
                c.hkv as u32,
                0, // sliding window 关(本实例化 SLIDING_WINDOW=0)
                c.q_scale,
                out as *mut std::ffi::c_void,
                q as *const std::ffi::c_void,
                kc as *mut std::ffi::c_void,
                vc as *mut std::ffi::c_void,
                pt as *const i32,
                c.max_seq_len as u32,
                sl as *const u32,
                1, // batch = 1(owl 单会话)
                1.0, // kv_scale(fp8 KV 无 per-tensor scale)
                c.q_seq_len as u32,
                mk as *const u32,
                sem as *mut u32,
                scratch as *mut std::ffi::c_void,
                c.stride_page as u64,
                c.stride_token as u64,
                c.stride_head as u64,
                stream.cu_stream() as *mut std::ffi::c_void,
            )
        };
        if rc != 0 {
            return Err(OpError::Launch {
                op: VERIFY_XQA.into(),
                stage: owl_kernels_contract_stage(),
                detail: format!(
                    "owl_xqa_launch 参数拒绝 rc={rc}(q_seq_len={} hd={} page={} max_seq={} sm={})",
                    c.q_seq_len, c.hd, c.page, c.max_seq_len, c.sm_count
                ),
            });
        }
        Ok(Bytes { id: c.out.id, len: msg.out_elems })
    }
}

/// 失败阶段标注(发射期)
fn owl_kernels_contract_stage() -> crate::contract::Stage {
    crate::contract::Stage::Compute
}

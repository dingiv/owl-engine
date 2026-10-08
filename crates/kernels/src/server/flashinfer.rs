//! FlashInfer paged prefill runtime(StaticLib 链路;FFI plan/run 分离)。
//!
//! init:workspace 预钉(float 64MB + int 32MB;构造期,捕获内首次初始化
//! 事故类结构性消灭)。run:plan(每形状缓存,键 = 形状七元组)→ run
//! (f16/fp8kv 按名路由)。

use crate::contract::{Bytes, LaunchMsg, OpError, OpId};
use crate::device::ScratchBuf;

use crate::client::flashinfer::{parse_prefill, PREFILL_FI, PREFILL_FI_FP8KV};
use crate::family::flashinfer::{
    owl_fi_prefill_plan, owl_fi_prefill_run, owl_fi_prefill_run_fp8kv,
    FI_FLOAT_WS_BYTES, FI_HOST_STAGING_BYTES, FI_INT_WS_BYTES,
};
use crate::registry::KernelSpec;

#[derive(Clone, Copy, PartialEq)]
struct PlanKey {
    total_rows: usize,
    ctx_total: usize,
    t: usize,
    hq: usize,
    hkv: usize,
    hd: usize,
    page: usize,
}

pub struct FiRuntime {
    float_ws: Option<ScratchBuf>,
    int_ws: Option<ScratchBuf>,
    host_staging: Vec<u8>,
    plan: Option<(PlanKey, [i64; 15])>,
}

impl Default for FiRuntime {
    fn default() -> Self {
        Self {
            float_ws: None,
            int_ws: None,
            host_staging: vec![0u8; FI_HOST_STAGING_BYTES],
            plan: None,
        }
    }
}

impl FiRuntime {
    // 名字 = client face re-export(contract::names 单源),零本地字面量
    // —— 2026-10-12 review I-10 案(旧 N_F16/N_FP8KV 已删)
}

impl KernelSpec for FiRuntime {
    fn id(&self) -> OpId {
        OpId(PREFILL_FI)
    }

    fn names(&self) -> Vec<&'static str> {
        vec![PREFILL_FI, PREFILL_FI_FP8KV]
    }

    fn linkage(&self) -> crate::contract::Linkage {
        crate::contract::Linkage::StaticLib
    }

    fn validate(&self) -> Result<(), OpError> { Ok(()) }
    fn init(&mut self, res: &mut dyn crate::device::DeviceRes, _exec: &mut crate::device::Exec) -> Result<(), OpError> {
        if self.float_ws.is_some() {
            return Ok(());
        }
        self.float_ws = Some(res.alloc(FI_FLOAT_WS_BYTES, "fi.float_ws")?);
        self.int_ws = Some(res.alloc(FI_INT_WS_BYTES, "fi.int_ws")?);
        Ok(())
    }

    fn run(&mut self, msg: &LaunchMsg, res: &mut dyn crate::device::DeviceRes, _exec: &mut crate::device::Exec) -> Result<Bytes, OpError> {
        let call = parse_prefill(msg)?;
        let r = |b: &crate::client::flashinfer::BlockRef| {
            res.resolve(&Bytes { id: b.id, len: 0 }).map(|p| p + b.byte_offset)
        };
        let q = r(&call.q)?;
        let kc = r(&call.kc_fi)?;
        let vc = r(&call.vc)?;
        let q_cu = r(&call.q_cu)?;
        let indices = r(&call.indices)?;
        let indptr = r(&call.indptr)?;
        let last_len = r(&call.last_len)?;
        let out_ref = crate::client::flashinfer::BlockRef { id: call.out.id, byte_offset: 0 };
        let out_ptr = r(&out_ref)?;
        // 就地解包( locals 贯穿 plan/run;None = init 未跑,Asset 拒)
        let float_ws = self.float_ws.as_ref().ok_or_else(|| OpError::Asset {
            op: PREFILL_FI.into(),
            detail: "workspace 未装配".into(),
        })?;
        let int_ws = self.int_ws.as_ref().ok_or_else(|| OpError::Asset {
            op: PREFILL_FI.into(),
            detail: "workspace 未装配".into(),
        })?;
        let stream = res.stream()?;

        // plan(host,每形状一次;键 = 形状七元组)
        let key = PlanKey {
            total_rows: call.total_rows,
            ctx_total: call.ctx_total,
            t: call.t,
            hq: call.hq,
            hkv: call.hkv,
            hd: call.hd,
            page: call.page,
        };
        if self.plan.as_ref().map(|(k, _)| *k != key).unwrap_or(true) {
            let qo_indptr = [0i32, call.t as i32];
            let kv_indptr = [0i32, call.ctx_total as i32];
            let (mut plan15, mut cta_tile_q, mut split_kv) = ([0i64; 15], 0i32, 0i32);
            let rc = unsafe {
                owl_fi_prefill_plan(
                    float_ws.ptr as *mut std::ffi::c_void,
                    float_ws.bytes,
                    int_ws.ptr as *mut std::ffi::c_void,
                    int_ws.bytes,
                    self.host_staging.as_mut_ptr() as *mut std::ffi::c_void,
                    self.host_staging.len(),
                    plan15.as_mut_ptr(),
                    &mut cta_tile_q,
                    &mut split_kv,
                    qo_indptr.as_ptr(),
                    kv_indptr.as_ptr(),
                    call.total_rows as i32,
                    1, // batch = 1(owl 单会话)
                    call.hq as i32,
                    call.hkv as i32,
                    call.hd as i32,
                    call.page as i32,
                    stream.cu_stream() as *mut std::ffi::c_void,
                )
            };
            if rc != 0 {
                return Err(OpError::Launch {
                    op: msg.kernel.name.clone(),
                    stage: crate::contract::Stage::Load,
                    detail: format!(
                        "plan err {rc}(hd={} page={} T={} ctx={})",
                        call.hd, call.page, call.t, call.ctx_total
                    ),
                });
            }
            self.plan = Some((key, plan15));
        }
        let plan15 = self.plan.as_ref().map(|(_, p)| *p).expect("plan 缓存");

        let run = if call.fp8kv {
            owl_fi_prefill_run_fp8kv
        } else {
            owl_fi_prefill_run
        };
        let rc = unsafe {
            run(
                q as *const std::ffi::c_void,
                kc as *const std::ffi::c_void,
                vc as *const std::ffi::c_void,
                out_ptr as *mut std::ffi::c_void,
                q_cu as *mut i32,
                indices as *mut i32,
                indptr as *mut i32,
                last_len as *mut i32,
                plan15.as_ptr(),
                int_ws.ptr as *mut std::ffi::c_void,
                int_ws.bytes,
                float_ws.ptr as *mut std::ffi::c_void,
                float_ws.bytes,
                1, // batch
                call.hq as i32,
                call.hkv as i32,
                call.hd as i32,
                call.page as i32,
                call.total_rows as i32,
                call.sm_scale,
                stream.cu_stream() as *mut std::ffi::c_void,
            )
        };
        if rc != 0 {
            return Err(OpError::Launch {
                op: msg.kernel.name.clone(),
                stage: crate::contract::Stage::Compute,
                detail: format!("run err {rc}"),
            });
        }
        Ok(Bytes { id: call.out.id, len: msg.out_elems })
    }
}

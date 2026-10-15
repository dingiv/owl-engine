//! FlashInfer paged prefill runtime(FFI 直调链路;plan/run 分离)。
//!
//! init:workspace 预钉(float 64MB + int 32MB;构造期,捕获内首次初始化
//! 事故类结构性消灭)。run:plan(每形状缓存,键 = 形状七元组)→ run
//! (f16/fp8kv 按名路由)。

use crate::contract::{Bytes, Dtype, LaunchMsg, OpError, OpId};
use crate::device::ScratchBuf;

use crate::client::flashinfer::{parse_prefill, PREFILL_FI};
use crate::family::flashinfer::{
    owl_fi_prefill_plan, owl_fi_prefill_run, owl_fi_prefill_run_fp8kv,
    FI_FLOAT_WS_BYTES, FI_HOST_STAGING_BYTES, FI_INT_WS_BYTES,
};
use crate::registry::{FamilyGuard, KernelFamily};

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

impl FiRuntime {
    /// **本族身份**(族名 = 线契约名;字面量住 `family::flashinfer`,零门)
    pub const ID: OpId = OpId(crate::family::flashinfer::PREFILL_FI);

    // 本族算子表:由 `KernelSpec::FAMILY` 派生(f16 + fp8 两臂)
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


impl KernelFamily for FiRuntime {
    fn id(&self) -> OpId {
        Self::ID
    }

    /// 本族**已建表**算子(fp8 臂;f16 臂未建表 —— 它由族主名承担,
    /// 见 `KernelFamily::names()` 的 id ∪ ops 律)


    /// 构造函数:workspace 预钉(float 64MB + int 32MB;构造期,
    /// 捕获内首次初始化事故类结构性消灭)
    fn assemble(
        res: &mut dyn crate::device::DeviceRes,
        _exec: &mut crate::device::Exec,
    ) -> Result<Box<dyn KernelFamily>, OpError> {
        let me = FiRuntime {
            float_ws: Some(res.alloc(FI_FLOAT_WS_BYTES, "fi.float_ws")?),
            int_ws: Some(res.alloc(FI_INT_WS_BYTES, "fi.int_ws")?),
            host_staging: vec![0u8; FI_HOST_STAGING_BYTES],
            plan: None,
        };
        Ok(FamilyGuard::boxed(me))
    }

    fn run(&mut self, msg: &LaunchMsg, res: &mut dyn crate::device::DeviceRes, _exec: &mut crate::device::Exec) -> Result<Bytes, OpError> {
        let call = parse_prefill(msg)?;
        let r = |b: &crate::client::flashinfer::BlockRef| {
            res.resolve(b.id).map(|p| p + b.byte_offset)
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
        // ⚠️ **plan 必须在捕获窗之外算**(2026-10-14 实测 + 结构性守卫;自
        // `feat/prefill-graph` 合入)。
        //
        // 事实(本机实测,`cudaStreamCaptureModeGlobal`,sm_86):
        // `cudaMemcpyAsync` 的 **H2D/D2H × pinned/pageable 四组合全部可被捕获**
        // (各成 1 个 memcpy 节点,回放 3 次值正确);不可捕获的只有**同步**
        // `cudaMemcpy` 与 `cudaMalloc/Free`。⇒ plan 内部的
        // `cudaMemcpyAsync(int_ws ← page_locked staging)` **会被烘进图**,而
        // `host_staging` 是**可变**缓冲(换形状会重写)—— 一旦 plan 在窗内算,
        // 图里就留下"从可变 host 缓冲拷贝"的节点,回放期取**当时**的内容
        // ⇒ 错调度表/越界,且**不报错**。
        //
        // 现状为何没炸:图捕获前有一次 warmup eager(`GraphPlan::plan`),plan 已
        // 缓存 ⇒ 捕获期不重算。但那是"时序恰好",不是不变量 —— 故此处立硬门:
        // 捕获期若 plan 未命中,**结构化拒**(→ 上层 `Ok(None)` → eager 原路),
        // 不静默、不把错节点烘进图。
        // (更强做法:每形状一份独立 staging 缓冲 ⇒ 节点源恒定;待有需求再上。)
        if self.plan.as_ref().map(|(k, _)| *k != key).unwrap_or(true) {
            if res.capturing().unwrap_or(false) {
                return Err(OpError::Launch {
                    op: msg.kernel.name.clone(),
                    stage: crate::contract::Stage::Load,
                    detail: "FI plan 未缓存而处于捕获窗:plan 内含 H2D(可被捕获),\
                             必须在窗口外算 —— 拒捕回 eager".to_string(),
                });
            }
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
        // 家族面入账(A13 补漏):FlashInfer = **直连 FFI 发射**(plan/run 均不经
        // `Exec::launch`),账上本无痕;ptrs = q/kc/vc/out 四块地址。
        res.note_device("fam_ffi_launch", &msg.kernel.name, 0, &[q, kc, vc, out_ptr]);
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
        Ok(Bytes::new(call.out.id, msg.out_elems, Dtype::F16, 2))
    }
}

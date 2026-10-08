//! Marlin W4A16 runtime(StaticLib 链路;纯 FFI 零句柄,stream/dev 注入)。
//! 三名同构:f16 / AWQ / bf16(名路由 → gemm_v2_raw / gemm_v2_raw_bf16)。

use crate::contract::{Bytes, LaunchMsg, OpError, OpId, Stage};
use crate::device::LaunchVal;
use crate::registry::RunEnv;
use crate::family::marlin;
use crate::client::marlin::{parse_gemm, GEMM_W4A16};
use crate::registry::FamilyRuntime;

/// marlin 家族(f16/AWQ/bf16 三名;一个 runtime 实例按名分派)
#[derive(Default)]
pub struct MarlinRuntime;

impl MarlinRuntime {
    pub const N_F16: &'static str = "marlin_gemm_w4a16";
    pub const N_AWQ: &'static str = "marlin_gemm_w4a16_awq";
    pub const N_BF16: &'static str = "marlin_gemm_w4a16_bf16";
}

impl FamilyRuntime for MarlinRuntime {
    fn id(&self) -> OpId {
        OpId(GEMM_W4A16)
    }

    fn names(&self) -> Vec<&'static str> {
        vec![Self::N_F16, Self::N_AWQ, Self::N_BF16]
    }

    fn linkage(&self) -> crate::contract::Linkage {
        crate::contract::Linkage::StaticLib
    }

    fn init(&mut self, _env: &mut RunEnv) -> Result<(), OpError> {
        Ok(()) // 纯 FFI 零句柄;链接期已解析
    }

    fn run(&mut self, msg: &LaunchMsg, env: &mut RunEnv) -> Result<Bytes, OpError> {
        // AWQ 臂:kU4 has_zp,7 块(独立 FFI);f16/bf16 走 6 块通用臂
        if msg.kernel.name == Self::N_AWQ {
            return self.run_awq(msg, env);
        }
        let call = parse_gemm(msg, msg.kernel.name == marlin::GEMM_W4A16_BF16)?;
        let r = |b: &crate::client::marlin::BlockRef| {
            env.res.resolve(&Bytes { id: b.id, len: 0 }).map(|p| p + b.byte_offset)
        };
        let a = r(&call.a)?;
        let b = r(&call.b)?;
        let out = r(&call.out)?;
        let scales = r(&call.scales)?;
        let ws = r(&call.ws)?;
        let c_tmp = r(&call.c_tmp)?;
        let dev = env.res.device_ordinal()?;
        let stream = env.res.stream()?;
        // 排队即回执(fire-and-forget;marlin host launcher 入 COMPUTE 流)
        let rr = if call.bf16 {
            unsafe {
                marlin::gemm_v2_raw_bf16(
                    a as *const u16, b as *const i32, out as *mut u16,
                    scales as *const u16, c_tmp as *const std::ffi::c_void,
                    call.m as i32, call.n as i32, call.k as i32,
                    ws as *mut i32, call.groupsize, dev,
                    stream.cu_stream() as usize,
                )
            }
            .map_err(|code| {
                format!("err={code}({})", marlin::v2_err_str(code))
            })
        } else {
            unsafe {
                marlin::gemm_v2_raw(
                    a as *const u16, b as *const i32, out as *mut u16,
                    scales as *const u16, c_tmp as *const std::ffi::c_void,
                    call.m as i32, call.n as i32, call.k as i32,
                    ws as *mut i32, call.groupsize, dev,
                    stream.cu_stream() as usize,
                )
            }
            .map_err(|e| format!("err {e}: {}", marlin::v2_err_str(e)))
        };
        rr.map_err(|detail| OpError::Launch {
            op: msg.kernel.name.clone(),
            stage: Stage::Compute,
            detail,
        })?;
        Ok(Bytes { id: call.out.id, len: msg.out_elems })
    }
}

impl MarlinRuntime {
    /// AWQ kU4 臂(7 块;zeros 槽 FFI 实参位 4,ws 位 5,c_tmp 位 6)
    fn run_awq(&mut self, msg: &LaunchMsg, env: &mut RunEnv) -> Result<Bytes, OpError> {
        let call = crate::client::marlin::parse_gemm_awq(msg)?;
        let r = |b: &crate::client::marlin::BlockRef| {
            env.res.resolve(&Bytes { id: b.id, len: 0 }).map(|p| p + b.byte_offset)
        };
        let a = r(&call.a)?;
        let b = r(&call.b)?;
        let out = r(&call.out)?;
        let scales = r(&call.scales)?;
        let zeros = r(call.zeros.as_ref().ok_or_else(|| OpError::Contract {
            op: msg.kernel.name.clone(),
            field: "zeros",
            expect: "AWQ 臂必带 zeros 槽".into(),
            got: "无".into(),
        })?)?;
        let ws = r(&call.ws)?;
        let c_tmp = r(&call.c_tmp)?;
        let dev = env.res.device_ordinal()?;
        let stream = env.res.stream()?;
        let rr = unsafe {
            marlin::gemm_v2_awq_raw(
                a as *const u16, b as *const i32, out as *mut u16,
                scales as *const u16, zeros as *const i32, c_tmp as *const std::ffi::c_void,
                call.m as i32, call.n as i32, call.k as i32,
                ws as *mut i32, call.groupsize, dev,
                stream.cu_stream() as usize,
            )
        };
        rr.map_err(|e| OpError::Launch {
            op: msg.kernel.name.clone(),
            stage: Stage::Compute,
            detail: format!("err {e}: {}", marlin::v2_err_str(e)),
        })?;
        Ok(Bytes { id: call.out.id, len: msg.out_elems })
    }
}


//! Marlin W4A16 runtime(FFI 直调链路;纯 FFI 零句柄,stream/dev 注入)。
//! 三名同构:f16 / AWQ / bf16(名路由 → gemm_v2_raw / gemm_v2_raw_bf16)。

use crate::contract::{Bytes, LaunchMsg, OpError, OpId, Stage};

use crate::client::marlin::{parse_gemm, GEMM_W4A16_AWQ};
use crate::family::marlin;
use crate::registry::{FamilyGuard, KernelFamily};

/// marlin 家族(f16/AWQ/bf16 三名;一个 runtime 实例按名分派。
/// 名字 = 本族身份常量(`family::marlin`;声明面/执行面同源)
/// —— 2026-10-12 review I-10 案)
#[derive(Default)]
pub struct MarlinRuntime;


impl KernelFamily for MarlinRuntime {
    fn id(&self) -> OpId {
        Self::ID
    }



    /// 构造函数:纯 FFI 零句柄;链接期已解析(装配门 = 无)
    fn assemble(
        _res: &mut dyn crate::device::DeviceRes,
        _exec: &mut crate::device::Exec,
    ) -> Result<Box<dyn KernelFamily>, OpError> {
        Ok(FamilyGuard::boxed(MarlinRuntime))
    }

    fn run(&mut self, msg: &LaunchMsg, res: &mut dyn crate::device::DeviceRes, exec: &mut crate::device::Exec) -> Result<Bytes, OpError> {
        // AWQ 臂:kU4 has_zp,7 块(独立 FFI);f16/bf16 走 6 块通用臂
        if msg.kernel.name == GEMM_W4A16_AWQ {
            return self.run_awq(msg, res, exec);
        }
        let call = parse_gemm(msg, msg.kernel.name == marlin::GEMM_W4A16_BF16)?;
        let r = |b: &crate::client::marlin::BlockRef| {
            res.resolve(&Bytes { id: b.id, len: 0 }).map(|p| p + b.byte_offset)
        };
        let a = r(&call.a)?;
        let b = r(&call.b)?;
        let out = r(&call.out)?;
        let scales = r(&call.scales)?;
        let ws = r(&call.ws)?;
        let c_tmp = r(&call.c_tmp)?;
        let dev = res.device_ordinal()?;
        let stream = res.stream()?;
        res.note_device("fam_ffi_launch", &msg.kernel.name, 0, &[a, b, out]);
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
    /// **本族身份**(族名 = 线契约名;字面量住 `family::<族>`,零门)
    /// —— 声明面 `ops/<域>::FAMILY` 与执行面 `id()` 引用**同一字面量**。
    pub const ID: OpId = OpId(crate::family::marlin::GEMM_W4A16);
    /// AWQ kU4 臂(7 块;zeros 槽 FFI 实参位 4,ws 位 5,c_tmp 位 6)
    fn run_awq(&mut self, msg: &LaunchMsg, res: &mut dyn crate::device::DeviceRes, _exec: &mut crate::device::Exec) -> Result<Bytes, OpError> {
        let call = crate::client::marlin::parse_gemm_awq(msg)?;
        let r = |b: &crate::client::marlin::BlockRef| {
            res.resolve(&Bytes { id: b.id, len: 0 }).map(|p| p + b.byte_offset)
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
        let dev = res.device_ordinal()?;
        let stream = res.stream()?;
        // 家族面入账(A13 补漏):**直连 FFI 发射**不计 `Command` ⇒ 账上本无痕。
        // marlin/cublas/FI 三族的 run 都是 FFI 直发(不经 `Exec::launch`),
        // 故各 arm 自报一条;`ptrs` = 本 call 的实参地址(生命周期判据)。
        res.note_device("fam_ffi_launch", &msg.kernel.name, 0, &[a, b, out]);
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


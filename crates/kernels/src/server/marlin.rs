//! Marlin W4A16/W4A8 runtime(FFI 直调链路;纯 FFI 零句柄,stream/dev 注入)。
//! 四名同构:f16 / AWQ / bf16(W4A16)+ W4A8×AWQ(名路由 → gemm_v2_*_raw)。

use crate::contract::{Bytes, Dtype, LaunchMsg, OpError, OpId, Stage};

use crate::client::marlin::{parse_gemm, GEMM_W4A16_AWQ, GEMM_W4A8_AWQ};
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
        // AWQ 臂:kU4 has_zp,7 块(独立 FFI);W4A8×AWQ 臂:再加 a_scales,
        // 8 块(int8 激活;独立 FFI);f16/bf16 走 6 块通用臂
        if msg.kernel.name == GEMM_W4A16_AWQ {
            return self.run_awq(msg, res, exec);
        }
        if msg.kernel.name == GEMM_W4A8_AWQ {
            return self.run_w4a8_awq(msg, res, exec);
        }
        let call = parse_gemm(msg, msg.kernel.name == marlin::GEMM_W4A16_BF16)?;
        let r = |b: &crate::client::marlin::BlockRef| {
            res.resolve(b.id).map(|p| p + b.byte_offset)
        };
        let a = r(&call.a)?;
        let b = r(&call.b)?;
        let out = r(&call.out)?;
        let scales = r(&call.scales)?;
        let ws = r(&call.ws)?;
        let c_tmp = r(&call.c_tmp)?;
        check_pack_sizes(&msg.kernel.name, &call, res)?;
        let dev = res.device_ordinal()?;
        let stream = res.stream()?;
        let (m, n, k) = call.ffi_mnk(); // 槽序 m,k,n → C 形参序 m,n,k(单源)
        res.note_device("fam_ffi_launch", &msg.kernel.name, 0, &[a, b, out]);
        // 排队即回执(fire-and-forget;marlin host launcher 入 COMPUTE 流)
        let rr = if call.bf16 {
            unsafe {
                marlin::gemm_v2_raw_bf16(
                    a as *const u16, b as *const i32, out as *mut u16,
                    scales as *const u16, c_tmp as *const std::ffi::c_void,
                    m, n, k,
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
                    m, n, k,
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
        Ok(Bytes::new(call.out.id, msg.out_elems, if call.bf16 { Dtype::BF16 } else { Dtype::F16 }, 2))
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
            res.resolve(b.id).map(|p| p + b.byte_offset)
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
        check_pack_sizes(&msg.kernel.name, &call, res)?;
        let dev = res.device_ordinal()?;
        let stream = res.stream()?;
        // 家族面入账(A13 补漏):**直连 FFI 发射**不计 `Command` ⇒ 账上本无痕。
        // marlin/cublas/FI 三族的 run 都是 FFI 直发(不经 `Exec::launch`),
        // 故各 arm 自报一条;`ptrs` = 本 call 的实参地址(生命周期判据)。
        res.note_device("fam_ffi_launch", &msg.kernel.name, 0, &[a, b, out]);
        let (m, n, k) = call.ffi_mnk(); // 槽序 m,k,n → C 形参序 m,n,k(单源)
        let rr = unsafe {
            marlin::gemm_v2_awq_raw(
                a as *const u16, b as *const i32, out as *mut u16,
                scales as *const u16, zeros as *const i32, c_tmp as *const std::ffi::c_void,
                m, n, k,
                ws as *mut i32, call.groupsize, dev,
                stream.cu_stream() as usize,
            )
        };
        rr.map_err(|e| OpError::Launch {
            op: msg.kernel.name.clone(),
            stage: Stage::Compute,
            detail: format!("err {e}: {}", marlin::v2_err_str(e)),
        })?;
        Ok(Bytes::new(call.out.id, msg.out_elems, if call.bf16 { Dtype::BF16 } else { Dtype::F16 }, 2))
    }

    /// **W4A8 × AWQ 臂**(7 块 = AWQ 槽表后插 `a_scales`;int8 已量化激活 +
    /// per-token fp32 尺度)。
    ///
    /// 与 AWQ 臂唯一的结构差异 = `c_tmp` **传空指针**:C 侧 `a_is_s8=true`
    /// 强制 `use_fp32_reduce`,c_tmp 由 host 发射器流序自管(槽位仍须在
    /// 线格式里、仍有实体 —— F2 账长门照旧,见 [`check_pack_sizes`])
    /// —— 暗雷面在 C 侧已焊缝(见 `marlin_gemm_v2_awq_w4a8_ffi` 注)。
    fn run_w4a8_awq(&mut self, msg: &LaunchMsg, res: &mut dyn crate::device::DeviceRes, _exec: &mut crate::device::Exec) -> Result<Bytes, OpError> {
        let call = crate::client::marlin::parse_gemm_w4a8_awq(msg)?;
        let r = |b: &crate::client::marlin::BlockRef| {
            res.resolve(b.id).map(|p| p + b.byte_offset)
        };
        let a = r(&call.a)?;
        let b = r(&call.b)?;
        let out = r(&call.out)?;
        let scales = r(&call.scales)?;
        let zeros = r(call.zeros.as_ref().ok_or_else(|| OpError::Contract {
            op: msg.kernel.name.clone(),
            field: "zeros",
            expect: "W4A8×AWQ 臂必带 zeros 槽".into(),
            got: "无".into(),
        })?)?;
        let a_scales = r(call.a_scales.as_ref().ok_or_else(|| OpError::Contract {
            op: msg.kernel.name.clone(),
            field: "a_scales",
            expect: "W4A8×AWQ 臂必带 a_scales 槽".into(),
            got: "无".into(),
        })?)?;
        let ws = r(&call.ws)?;
        let c_tmp = r(&call.c_tmp)?;
        check_pack_sizes(&msg.kernel.name, &call, res)?;
        let dev = res.device_ordinal()?;
        let stream = res.stream()?;
        res.note_device("fam_ffi_launch", &msg.kernel.name, 0, &[a, b, out]);
        let (m, n, k) = call.ffi_mnk(); // 槽序 m,k,n → C 形参序 m,n,k(单源)
        let rr = unsafe {
            marlin::gemm_v2_awq_w4a8_raw(
                a as *const i8, b as *const i32, out as *mut u16,
                scales as *const u16, zeros as *const i32, a_scales as *const f32,
                std::ptr::null(), // c_tmp:host 自管(C 侧 a_is_s8 ⇒ use_fp32_reduce)
                m, n, k,
                ws as *mut i32, call.groupsize, dev,
                stream.cu_stream() as usize,
                false, // w4a8_m8:配置档已退役(2026-10-27;16-row 恒为生产档)
            )
        };
        // c_tmp 槽仍被 F2 门核账长(线格式契约),但**不进 FFI**(上面传 null)
        let _ = c_tmp;
        rr.map_err(|e| OpError::Launch {
            op: msg.kernel.name.clone(),
            stage: Stage::Compute,
            detail: format!("err {e}: {}", marlin::v2_err_str(e)),
        })?;
        // W4A8 当前只接 f16 档(wrapper 硬编 is_bf16=0;bf16 档需另接)
        Ok(Bytes::new(call.out.id, msg.out_elems, Dtype::F16, 2))
    }
}

/// **F2 入参尺寸自证**(两臂共用):FFI 越界**不会当场报错**而是写进邻块
/// (A13 期间 `邻块 clobber` 正是头号嫌疑)⇒ 在直发**前**拿块账长比公式。
/// 公式与拒语原文均住 `family::marlin`(单源;此处只做 `String → OpError` 映射)。
fn check_pack_sizes(
    op: &str,
    call: &crate::client::marlin::MarlinCall,
    res: &dyn crate::device::DeviceRes,
) -> Result<(), OpError> {
    // BlockSlice 兼容:可用字节 = 账长 − 块内偏移
    let have_ws = res.resolve_len(call.ws.id)?.saturating_sub(call.ws.byte_offset as usize);
    marlin::ws_size_check(call.n, have_ws).map_err(|detail| OpError::Contract {
        op: op.to_string(),
        field: "ws",
        expect: format!("块账长 ≥ {} B", marlin::v2_workspace_bytes(call.n)),
        got: detail,
    })?;
    let have_ct = res.resolve_len(call.c_tmp.id)?.saturating_sub(call.c_tmp.byte_offset as usize);
    marlin::c_tmp_size_check(have_ct).map_err(|detail| OpError::Contract {
        op: op.to_string(),
        field: "c_tmp",
        expect: format!("块账长 ≥ {} B", marlin::C_TMP_MIN_BYTES),
        got: detail,
    })?;
    Ok(())
}


#[cfg(all(test, feature = "device"))]
mod f2_gate {
    //! F2 门**接线**证明(负例;零 GPU):ws/c_tmp 块短一截 ⇒ `run` 在
    //! 「解析后、FFI 发射前」结构化拒(field 指名槽位),而不是让 FFI 越界写
    //! 邻块。正例(够长放行)在 `family::marlin::size_checks`(纯函数)。
    use super::*;
    use crate::contract::{Arg, Dtype, KernelSource};
    use crate::device::ScratchBuf;
    use crate::registry::KernelFamily;

    /// 账长替身:只回答 `resolve_len`(本测试的判决输入);其余面一律 Err ——
    /// 走到了就是接线错了(这些面本应在门之后才被调用)。
    struct FixedRes {
        ws_len: usize,
        c_tmp_len: usize,
    }
    impl crate::device::DeviceRes for FixedRes {
        fn context(&self) -> Result<std::sync::Arc<cudarc::driver::CudaContext>, OpError> {
            Err(unexpected("context"))
        }
        fn stream(&self) -> Result<std::sync::Arc<cudarc::driver::CudaStream>, OpError> {
            Err(unexpected("stream"))
        }
        fn resolve(&self, _id: u64) -> Result<u64, OpError> {
            Ok(0x1000) // 占位指针:门只看账长,不看地址
        }
        fn resolve_len(&self, id: u64) -> Result<usize, OpError> {
            Ok(match id {
                6 => self.ws_len,
                7 => self.c_tmp_len,
                _ => 1 << 20,
            })
        }
        fn alloc(&mut self, _b: usize, _t: &str) -> Result<ScratchBuf, OpError> {
            Err(unexpected("alloc"))
        }
        fn capturing(&self) -> Result<bool, OpError> {
            Err(unexpected("capturing"))
        }
        fn record_capture(&mut self, _n: crate::device::LaunchNote) -> Result<(), OpError> {
            Err(unexpected("record_capture"))
        }
        fn upload(&mut self, _d: u64, _s: &[u8]) -> Result<(), OpError> {
            Err(unexpected("upload"))
        }
        fn device_ordinal(&self) -> Result<i32, OpError> {
            Err(unexpected("device_ordinal"))
        }
        fn upload_block(&mut self, _b: &crate::contract::Bytes, _s: &[u8]) -> Result<(), OpError> {
            Err(unexpected("upload_block"))
        }
        fn alloc_block(&mut self, _b: usize, _t: &str) -> Result<crate::contract::Bytes, OpError> {
            Err(unexpected("alloc_block"))
        }
        fn download(&mut self, _s: u64, _d: &mut [u8]) -> Result<(), OpError> {
            Err(unexpected("download"))
        }
    }
    fn unexpected(what: &str) -> OpError {
        OpError::Contract {
            op: "f2.gate".into(),
            field: "unexpected",
            expect: format!("门应在 FFI 前拦下,不该走到 {what}()"),
            got: "走到了".into(),
        }
    }

    /// AWQ 臂 7 块 + 4 sz(n=2560 ⇒ ws 需求 = 2560/64*128*4 = 20480 B)
    fn awq_msg() -> crate::contract::LaunchMsg {
        let blk = |id: u64| Arg::block(id, Dtype::F16, 2);
        crate::contract::LaunchMsg {
            kernel: KernelSource { name: crate::family::marlin::GEMM_W4A16_AWQ.into(), source: String::new() },
            args: vec![
                blk(1), blk(2), blk(3), blk(4), blk(5), blk(6), blk(7),
                Arg::U64(64), Arg::U64(2560), Arg::U64(2560), Arg::U64(32),
            ],
            grid: (0, 0, 0),
            block: (0, 0, 0),
            shared_mem: 0,
            out_elems: 64 * 2560,
        }
    }

    #[test]
    fn short_ws_is_rejected_before_ffi() {
        let need = marlin::v2_workspace_bytes(2560);
        // 够长 ⇒ 门放行(随后撞在替身的 stream() 上 —— 证明拦它的不是门)
        let mut ok = FixedRes { ws_len: need, c_tmp_len: 4 };
        let mut rt = MarlinRuntime;
        let e = rt
            .run(&awq_msg(), &mut ok, &mut crate::device::Exec::new())
            .expect_err("替身无 GPU 面,门后必拒");
        assert!(matches!(&e, OpError::Contract { field: "unexpected", .. }), "门不该拦够长的 ws: {e}");

        // 短 1 B ⇒ 门在 FFI 前拒,且指名槽位与两边尺寸
        let mut short = FixedRes { ws_len: need - 1, c_tmp_len: 4 };
        let e = rt
            .run(&awq_msg(), &mut short, &mut crate::device::Exec::new())
            .expect_err("ws 短一截必拒");
        match &e {
            OpError::Contract { field, expect, got, .. } => {
                assert_eq!(*field, "ws");
                assert!(expect.contains(&need.to_string()), "expect 须带需求字节: {expect}");
                assert!(got.contains("ws 块账长"), "got 须保留 family 拒语: {got}");
            }
            other => panic!("期望结构化 Contract,得 {other:?}"),
        }

        // c_tmp 零容量 ⇒ 同样在 FFI 前拒(公式语义在 family 单源)
        let mut no_ct = FixedRes { ws_len: need, c_tmp_len: 0 };
        let e = rt
            .run(&awq_msg(), &mut no_ct, &mut crate::device::Exec::new())
            .expect_err("c_tmp 零容量必拒");
        assert!(matches!(&e, OpError::Contract { field: "c_tmp", .. }), "{e}");
    }
}

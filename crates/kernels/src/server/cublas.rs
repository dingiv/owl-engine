//! cublas GEMM runtime(FFI 直调链路,链接期符号已解析)。
//!
//! init:OwlCublas 句柄 + 私有工作区预钉(ensure_blas 律:构造期急切,
//! 捕获内首次初始化事故类结构性消灭)。run:解析 → 三指针 → gemm。

use crate::contract::{Bytes, Dtype, LaunchMsg, OpError, OpId};

use crate::client::cublas::{parse_gemm, parse_gemm_dst, GEMM_BF16, GEMM_F16, GEMM_F16_DST};
use crate::registry::{FamilyGuard, KernelFamily};
use crate::family::cublas::OwlCublas;


pub struct CublasRuntime {
    blas: Option<OwlCublas>,
    /// **私有工作区**（C1 刀1.5）：必须由 runtime **持有**到句柄销毁；
    /// 旧实现分配后丢弃 = 悬垂指针（k-probe 失败的真因候选）。
    /// 4 MiB 取代 cuBLAS 内部默认工作区（实测首 gemm 后 ~0.8 GiB 未入账）。
    /// 4MB 工作区(2026-10-22 预绑;只由 cublas 内部按指针消费,host 侧不再读)。
    #[allow(dead_code)]
    ws: Option<crate::device::ScratchBuf>,
}

impl CublasRuntime {
    /// **本族身份**(族名 = 线契约名;字面量住 `family::<族>`,零门)
    /// —— 声明面 `ops/<域>::FAMILY` 与执行面 `id()` 引用**同一字面量**。
    pub const ID: OpId = OpId(crate::family::cublas::GEMM_F16);
    // 本族算子表:由 `KernelSpec::FAMILY` 派生(`KernelFamily::ops()` 缺省实现)
}


impl KernelFamily for CublasRuntime {
    fn id(&self) -> OpId {
        Self::ID
    }



    /// 构造函数:OwlCublas 句柄预钉(ensure_blas 律:构造期急切,
    /// 捕获内首次初始化事故类结构性消灭)
    fn assemble(
        res: &mut dyn crate::device::DeviceRes,
        _exec: &mut crate::device::Exec,
    ) -> Result<Box<dyn KernelFamily>, OpError> {
        let stream = res.stream()?;
        let blas = OwlCublas::new(stream.clone())
            .map_err(|e| OpError::Asset { op: GEMM_F16.into(), detail: format!("cublas handle: {e}") })?;
        // C1 刀1.5 律:**装配期预绑私有工作区**,避免 cuBLAS 首次 gemv splitK
        // 在捕获/首次 gemm 时内部再分配(~0.8 GiB 实测未入账)。工作区由
        // runtime 持有(旧实现分配后丢弃 = 悬垂指针,k-probe 失败真因候选)。
        let ws = res
            .alloc(4 << 20, "cublas.ws")
            .map_err(|e| OpError::Asset { op: GEMM_F16.into(), detail: format!("cublas ws: {e}") })?;
        blas.set_workspace(ws.ptr, ws.bytes)
            .map_err(|e| OpError::Asset { op: GEMM_F16.into(), detail: format!("cublas set ws: {e}") })?;
        Ok(FamilyGuard::boxed(CublasRuntime { blas: Some(blas), ws: Some(ws) }))
    }

    fn run(&mut self, msg: &LaunchMsg, res: &mut dyn crate::device::DeviceRes, _exec: &mut crate::device::Exec) -> Result<Bytes, OpError> {
        // 变体(f16/bf16/**写列窗**)按【核名】路由;线内 nt sz = cublas 转置标志
        if msg.kernel.name == GEMM_F16_DST {
            let call = parse_gemm_dst(msg)?;
            let a = res.resolve(call.a.id)? + call.a.byte_offset;
            let b = res.resolve(call.b.id)? + call.b.byte_offset;
            let out = res.resolve(call.out.id)? + call.out.byte_offset;
            let blas = self.blas.as_ref().ok_or_else(|| OpError::Asset {
                op: GEMM_F16.into(),
                detail: "cublas 句柄未装配".into(),
            })?;
            res.note_device("fam_ffi_launch", &msg.kernel.name, 0, &[a, b, out]);
            blas.gemm_f16_dst(a, b, out, call.m, call.k, call.n, call.nt, call.ldc, call.off)
                .map_err(|e| OpError::Launch {
                    op: msg.kernel.name.clone(),
                    stage: crate::contract::Stage::Compute,
                    detail: format!(
                        "m={} k={} n={} nt={} ldc={} off={}: {e}",
                        call.m, call.k, call.n, call.nt, call.ldc, call.off
                    ),
                })?;
            return Ok(Bytes::new(call.out.id, msg.out_elems, Dtype::F16, 2));
        }
        let bf16 = msg.kernel.name == GEMM_BF16;
        let call = parse_gemm(msg)?;
        let a = res.resolve(call.a.id)? + call.a.byte_offset;
        let b = res.resolve(call.b.id)? + call.b.byte_offset;
        let out = res.resolve(call.out.id)? + call.out.byte_offset;
        let blas = self.blas.as_ref().ok_or_else(|| OpError::Asset {
            op: GEMM_F16.into(),
            detail: "cublas 句柄未装配".into(),
        })?;
        // 家族面入账(A13 补漏):cuBLAS = **直连 FFI 发射**(不经 `Exec::launch`),
        // 账上本无痕;ptrs = A/B/out 三块地址。
        res.note_device("fam_ffi_launch", &msg.kernel.name, 0, &[a, b, out]);
        let gemm = if bf16 {
            blas.gemm_bf16(a, b, out, call.m, call.k, call.n, call.nt)
        } else {
            blas.gemm_f16(a, b, out, call.m, call.k, call.n, call.nt)
        };
        gemm.map_err(|e| OpError::Launch {
            op: msg.kernel.name.clone(),
            stage: crate::contract::Stage::Compute,
            detail: format!("m={} k={} n={} nt={}: {e}", call.m, call.k, call.n, call.nt),
        })?;
        Ok(Bytes::new(call.out.id, msg.out_elems, if bf16 { Dtype::BF16 } else { Dtype::F16 }, 2))
    }
}

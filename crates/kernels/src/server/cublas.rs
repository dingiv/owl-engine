//! cublas GEMM runtime(FFI 直调链路,链接期符号已解析)。
//!
//! init:OwlCublas 句柄 + 私有工作区预钉(ensure_blas 律:构造期急切,
//! 捕获内首次初始化事故类结构性消灭)。run:解析 → 三指针 → gemm。

use crate::contract::{Bytes, Dtype, LaunchMsg, OpError, OpId};

use crate::client::cublas::{parse_gemm, GEMM_BF16, GEMM_F16};
use crate::registry::{FamilyGuard, KernelFamily};
use crate::family::cublas::OwlCublas;


#[derive(Default)]
pub struct CublasRuntime {
    blas: Option<OwlCublas>,
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
        // ⚠️ C1 刀1.5 律(SetWorkspace 预绑)**悬置中**(2026-10-12 review C 案
        // 定形):M4 cutover 时为 bisect k-probe 失败临时禁用,原 4MB ws
        // 分配后丢弃(死账面)—— 现已删除分配,句柄裸奄(默认池行为)。
        // 结案路径:诊断定谳非毒源后,此处补 `res.alloc(4<<20, "cublas.ws")`
        // + `blas.set_workspace(ptr, bytes)`(OwlCublas::set_workspace 已备);
        // 定谳为毒源则本注释升级为永久禁用依据。捕获期 gemv splitK 在
        // 悬置期间会走池分配(MEM_ALLOC/FREE 节点回归)—— 图捕获联调前
        // 必须先结本案。
        Ok(FamilyGuard::boxed(CublasRuntime { blas: Some(blas) }))
    }

    fn run(&mut self, msg: &LaunchMsg, res: &mut dyn crate::device::DeviceRes, _exec: &mut crate::device::Exec) -> Result<Bytes, OpError> {
        // 变体(f16/bf16)按【核名】路由;线内 nt sz = cublas 转置标志
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

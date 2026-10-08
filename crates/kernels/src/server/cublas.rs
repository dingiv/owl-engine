//! cublas GEMM runtime(StaticLib 链路;FFI 直调,链接期符号已解析)。
//!
//! init:OwlCublas 句柄 + 私有工作区预钉(ensure_blas 律:构造期急切,
//! 捕获内首次初始化事故类结构性消灭)。run:解析 → 三指针 → gemm。

use crate::contract::{Bytes, LaunchMsg, OpError, OpId};

use crate::client::cublas::{parse_gemm, GEMM_BF16, GEMM_F16};
use crate::registry::KernelSpec;
use crate::family::cublas::OwlCublas;


pub struct CublasRuntime {
    blas: Option<OwlCublas>,
}

impl Default for CublasRuntime {
    fn default() -> Self {
        Self { blas: None }
    }
}

impl KernelSpec for CublasRuntime {
    fn id(&self) -> OpId {
        OpId(GEMM_F16)
    }

    fn names(&self) -> Vec<&'static str> {
        vec![GEMM_F16, GEMM_BF16]
    }

    fn linkage(&self) -> crate::contract::Linkage {
        crate::contract::Linkage::StaticLib
    }

    fn validate(&self) -> Result<(), OpError> { Ok(()) }
    fn init(&mut self, res: &mut dyn crate::device::DeviceRes, _exec: &mut crate::device::Exec) -> Result<(), OpError> {
        if self.blas.is_some() {
            return Ok(());
        }
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
        self.blas = Some(blas);
        Ok(())
    }

    fn run(&mut self, msg: &LaunchMsg, res: &mut dyn crate::device::DeviceRes, _exec: &mut crate::device::Exec) -> Result<Bytes, OpError> {
        // 变体(f16/bf16)按【核名】路由;线内 nt sz = cublas 转置标志
        let bf16 = msg.kernel.name == GEMM_BF16;
        let call = parse_gemm(msg)?;
        let a = res.resolve(&Bytes { id: call.a.id, len: 0 })? + call.a.byte_offset;
        let b = res.resolve(&Bytes { id: call.b.id, len: 0 })? + call.b.byte_offset;
        let out = res.resolve(&Bytes { id: call.out.id, len: 0 })? + call.out.byte_offset;
        let blas = self.blas.as_ref().ok_or_else(|| OpError::Asset {
            op: GEMM_F16.into(),
            detail: "cublas 句柄未装配".into(),
        })?;
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
        Ok(Bytes { id: call.out.id, len: msg.out_elems })
    }
}

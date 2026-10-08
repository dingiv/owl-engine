//! GDN scalar runtime(单核臂;lmdeploy pre_sm90 port;AOT cubin)。
//!
//! 移植自 handle_gdn_scalar:o_f32 坟场(A1.7)+ soff 不变盒按 T 键
//! (捕获回放读死栈案第二违例的解法)+ cast-out f32→f16。

use crate::contract::{Bytes, InvariantBox, LaunchMsg, OpError, OpId, Stage};
use crate::device::{LaunchVal, ScratchBuf};
use crate::registry::RunEnv;
use crate::family::gdn_scalar::cubin;
use crate::client::gdn_scalar::{GdnScalarCall, GDN_SCALAR};
use crate::registry::FamilyRuntime;
use std::collections::HashMap;
use std::sync::Arc;

const OP: OpId = OpId(GDN_SCALAR);
const ASSET: &str = "gdn_scalar";
const CAST: &str = "owl_cast";

#[derive(Default)]
pub struct GdnScalarRuntime {
    loaded: bool,
    k: Option<Arc<cudarc::driver::CudaFunction>>,
    cast_f32_f16: Option<Arc<cudarc::driver::CudaFunction>>,
    o_f32: Option<ScratchBuf>,
    o_f32_grave: Vec<ScratchBuf>,
    soff: Option<ScratchBuf>,
    soff_boxes: HashMap<usize, InvariantBox>, // T 键不变盒
    t_cap: usize,
}

impl FamilyRuntime for GdnScalarRuntime {
    fn id(&self) -> OpId {
        OP
    }

    fn linkage(&self) -> crate::contract::Linkage {
        crate::contract::Linkage::AotCubin
    }

    fn init(&mut self, env: &mut RunEnv) -> Result<(), OpError> {
        if self.loaded {
            return Ok(());
        }
        self.k = Some(
            env.exec.load_cubin(env.res, ASSET, cubin::CHUNK_SCALAR_F32, cubin::KERNEL_F32, OP)?,
        );
        self.cast_f32_f16 = Some(env.exec.load_nvrtc(
            env.res, CAST,
            crate::sources::attention::CAST,
            "owl_cast_f32_f16", OP,
        )?);
        self.soff = Some(env.res.alloc(2 * 4, "gdn_scalar.soff")?);
        self.loaded = true;
        Ok(())
    }

    fn run(&mut self, msg: &LaunchMsg, env: &mut RunEnv) -> Result<Bytes, OpError> {
        let (call, out) = GdnScalarCall::parse(msg)?;
        // 拆借置顶(res/exec 之后全走局部;env 不再触碰)
        let RunEnv { res, exec } = env;
        let shape = call.shape();
        let t = shape.t as usize;
        let hv = shape.nv as usize;
        let kd = shape.kd as usize;
        let k = self.k.as_ref().ok_or_else(|| OpError::Asset { op: OP.0.to_string(), detail: "未装载".into() })?;
        let cast = self
            .cast_f32_f16
            .as_ref()
            .ok_or_else(|| OpError::Asset { op: OP.0.to_string(), detail: "cast 未装载".into() })?;

        // ── o_f32 容量(扩容 = 坟场,A1.7)──
        if t > self.t_cap {
            let nb = res.alloc(t * hv * kd * 4, "gdn_scalar.o_f32")?;
            if let Some(old) = self.o_f32.take() {
                self.o_f32_grave.push(old);
            }
            self.o_f32 = Some(nb);
            self.t_cap = t;
        }
        let o32 = self.o_f32.as_ref().unwrap().ptr;

        // ── soff:按 T 键不变盒 + upload(捕获窗 async 节点,回放恒读盒)──
        let soff = self.soff.as_ref().unwrap().ptr;
        let box_ = self
            .soff_boxes
            .entry(t)
            .or_insert_with(|| {
                InvariantBox::freeze([0i32, t as i32].iter().flat_map(|v| v.to_le_bytes()).collect())
            });
        res.upload(soff, box_.as_bytes())?;

        // ── 主核发射(内参序 = extern C 声明序;grid (ns, hv))──
        let state_p =
            res.resolve(call.state())? + (call.slot() as u64) * (hv * kd * kd * 4) as u64;
        let q_p = res.resolve(call.q())?;
        let k_p = res.resolve(call.k())?;
        let v_p = res.resolve(call.v())?;
        let beta_p = res.resolve(call.beta())?;
        let g_p = res.resolve(call.g())?;
        let vals: &[crate::device::LaunchVal] = &[
            crate::device::LaunchVal::Ptr(o32),
            crate::device::LaunchVal::Ptr(q_p),
            crate::device::LaunchVal::Ptr(k_p),
            crate::device::LaunchVal::Ptr(v_p),
            crate::device::LaunchVal::Ptr(beta_p),
            crate::device::LaunchVal::Ptr(g_p),
            crate::device::LaunchVal::Ptr(state_p),
            crate::device::LaunchVal::Ptr(soff),
            crate::device::LaunchVal::I32(shape.nk as i32),
            crate::device::LaunchVal::I32(hv as i32),
            crate::device::LaunchVal::I32(kd as i32),
            crate::device::LaunchVal::F32(1.0 / (kd as f32).sqrt()),
        ];
        let grid = (shape.ns, hv as u32, 1);
        let block = (cubin::BLOCK, 1, 1);
        let smem = cubin::SMEM_D128;
        exec.launch(res, OP, k, "gdn_scalar", grid, block, smem, vals)?;

        // ── cast-out:o_f32 → out(f16)──
        let out_p = res.resolve(&out)?;
        let n_out = t * hv * kd;
        exec.launch(
            res, OP, cast, "cast_out", (n_out.div_ceil(256) as u32, 1, 1), (256, 1, 1), 0,
            &[crate::device::LaunchVal::Ptr(o32),
              crate::device::LaunchVal::I32(n_out as i32),
              crate::device::LaunchVal::Ptr(out_p)],
        )?;
        Ok(out)
    }
}

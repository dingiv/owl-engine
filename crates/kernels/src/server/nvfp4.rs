//! NVFP4 块缩放 GEMM 家族(foreign:FFI 直调 `libnvfp4.a`;2026-10-28)。
//!
//! 一次 `nvfp4_gemm` 调用在**设备上**做完五件事(**零 host 工作、零同步**):
//! ```text
//!   dg ← 0 → amax(a) → g = amax/(6·448), alpha = g·gb (→ dg 槽)
//!   → quant(a → aq, asf) → CUTLASS NVFP4 GEMM(alpha_ptr = dg)
//! ```
//!
//! ## 为什么 alpha 能留在设备侧(2026-10-28 进图刀)
//! `gb` 是**权重侧 per-tensor 全局缩放**,装载期 `pack_weight_b` 就算出来了、常驻在层上
//! (host 常量)⇒ `alpha = g·gb` 完全能在收尾核里做。epilogue 用
//! `fusion::LinearCombination`(Args 带 `alpha_ptr` ⇒ `Sm90ScalarBroadcast` 优先取指针)
//! ⇒ GEMM 直接读设备上的 4 字节。旧实现回读 `g` 到 host 再乘——那是**结构性**的
//! host 工作(一次 stream 排空/层),也是本族不能进图的原因。
//!
//! ## 因此本族**可进 CUDA graph**(decode / propose / verify 三张图都能捕)
//! 捕获窗禁的是 host 工作/同步/D2H;本族现在一样都没有(``dg` 由核自己清、alpha 在设备)。
//! 声明面同步:`ops/nvfp4.rs` 的 `CAPTURE = NoHostWork`(宏据此注入 `feat.graph`)。
//!
//! ## 权重打包一次、激活每调用一次
//! `bsf` 只依赖 `(N,K)`(`sm100_blockscaled_layout.hpp::tile_atom_to_shape_SFB`
//! 只取本操作数范围)⇒ 装载期 `pack_b` 一次常驻;激活每次调用重新量化。

use std::collections::HashMap;

use crate::contract::{Bytes, Dtype, LaunchMsg, OpError, OpId};

use crate::client::nvfp4::{parse_gemm, GEMM};
use crate::nvfp4::Nvfp4Plan;
use crate::registry::{FamilyGuard, KernelFamily};

pub struct Nvfp4Runtime {
    /// 形状 → 计划(布局/步长/workspace 都在计划里;按 `(m,k,n)` 缓存)
    plans: HashMap<(usize, usize, usize), Nvfp4Plan>,
    /// 计划声明的 workspace(CUTLASS 实测 0;真的非 0 时才分配并常驻)
    ws: Option<crate::device::ScratchBuf>,
}

impl Nvfp4Runtime {
    pub const ID: OpId = OpId(GEMM);

    /// 保证计划存在(只增;取用走 `plans.get`,避免 `&mut self` 借住整段 run)
    fn ensure_plan(&mut self, m: usize, n: usize, k: usize) -> Result<(), OpError> {
        if !self.plans.contains_key(&(m, k, n)) {
            let plan = Nvfp4Plan::new(m, n, k).ok_or_else(|| OpError::Asset {
                op: GEMM.into(),
                detail: format!(
                    "建 plan 失败 m={m} n={n} k={k}(要求 m>0(无 128 门)、n%8==0、k%16==0,\
                     且 SF 交错布局的仿射模型须与 cute 逐点一致)"
                ),
            })?;
            self.plans.insert((m, k, n), plan);
        }
        Ok(())
    }
}

impl KernelFamily for Nvfp4Runtime {
    fn id(&self) -> OpId {
        Self::ID
    }

    fn assemble(
        _res: &mut dyn crate::device::DeviceRes,
        _exec: &mut crate::device::Exec,
    ) -> Result<Box<dyn KernelFamily>, OpError> {
        // FFI 零句柄:链接期已解析(banner 在 boot 打过 = 链接已验)
        Ok(FamilyGuard::boxed(Nvfp4Runtime { plans: HashMap::new(), ws: None }))
    }

    fn run(
        &mut self,
        msg: &LaunchMsg,
        res: &mut dyn crate::device::DeviceRes,
        _exec: &mut crate::device::Exec,
    ) -> Result<Bytes, OpError> {
        let call = parse_gemm(msg)?;
        self.ensure_plan(call.m, call.n, call.k)?;
        let plan = self.plans.get(&(call.m, call.k, call.n)).expect("ensure_plan 刚插入");
        // **入参尺寸自证**(FFI 越界不报错,只会踩邻块)
        let need = [
            ("aq", call.aq.id, plan.a_q_bytes()),
            ("asf", call.asf.id, plan.sfa_bytes()),
            ("bq", call.bq.id, plan.b_q_bytes()),
            ("bsf", call.bsf.id, plan.sfb_bytes()),
            // `dg` = f32[2]（`g` 与 `alpha`）—— 少一个 f32 就会把 alpha 写到邻块
            ("dg", call.dg.id, 2 * std::mem::size_of::<f32>()),
        ];
        for (name, id, want) in need {
            let got = res.resolve_len(id)?;
            if got < want {
                return Err(OpError::Contract {
                    op: GEMM.into(),
                    field: "block",
                    expect: format!("{name} ≥ {want} 字节"),
                    got: format!("{got} 字节"),
                });
            }
        }
        let a = res.resolve(call.a.id)? + call.a.byte_offset;
        let bq = res.resolve(call.bq.id)? + call.bq.byte_offset;
        let bsf = res.resolve(call.bsf.id)? + call.bsf.byte_offset;
        let out = res.resolve(call.out.id)? + call.out.byte_offset;
        let aq = res.resolve(call.aq.id)? + call.aq.byte_offset;
        let asf = res.resolve(call.asf.id)? + call.asf.byte_offset;
        let dg = res.resolve(call.dg.id)? + call.dg.byte_offset;
        let stream = res.stream()?;
        let raw = stream.cu_stream() as *mut std::ffi::c_void;

        // workspace:计划说 0 就不给(实测 0;真非 0 时分配并常驻 —— 每调用分配会在账上抖)
        let ws_bytes = plan.ws_bytes();
        if ws_bytes > 0 && self.ws.is_none() {
            let b = res.alloc(ws_bytes, "nvfp4.ws")?;
            self.ws = Some(b);
        }
        let ws = self.ws.as_ref().map_or(std::ptr::null_mut(), |b| b.ptr as *mut std::ffi::c_void);

        res.note_device("fam_ffi_launch", &msg.kernel.name, 0, &[a, bq, out]);
        // ① 设备侧量化(三发:清 dg → amax → g/alpha + quantize)
        // `dg` 槽 = **f32[2]**:`[0]` = `g`(量化核读),`[1]` = `alpha = g·gb`
        // (接下来就是 GEMM 的 `alpha_ptr`,传 `dg + 1`)。
        // SAFETY: 指针来自账本块的 `resolve`(上面已按 plan 逐个验账长);
        // 形状 = plan 的形状 = 本次调用的 (m,k)。
        let rc = unsafe {
            plan.quant_act(
                a as *const std::ffi::c_void,
                aq as *mut u8,
                asf as *mut u8,
                dg as *mut f32,
                call.gb,
                raw,
            )
        };
        if rc != 0 {
            return Err(OpError::Launch {
                op: msg.kernel.name.clone(),
                stage: crate::contract::Stage::Compute,
                detail: format!("quant_act rc={rc} m={} k={}", call.m, call.k),
            });
        }
        // ② FP4 GEMM(**alpha 从设备读**、同流序已写定 ⇒ 零同步、可捕获)
        // SAFETY: 同 ①;`alpha_ptr = dg+1`;`out` 是 f16 [m,n],与 plan 的 stride 一致。
        // SAFETY(指针算术):`dg` 指向账本里 **≥ 8 字节**的块(上面 `need` 已验),
        // 故 `+1` 仍在块内。
        let alpha_dev = unsafe { (dg as *const f32).add(1) };
        let rc = unsafe {
            plan.run_dev(
                raw,
                aq as *const u8,
                asf as *const u8,
                alpha_dev,
                bq as *const u8,
                bsf as *const u8,
                out as *mut std::ffi::c_void,
                ws,
            )
        };
        if rc != 0 {
            return Err(OpError::Launch {
                op: msg.kernel.name.clone(),
                stage: crate::contract::Stage::Compute,
                detail: format!(
                    "gemm rc={rc} m={} n={} k={} gb={:e}",
                    call.m, call.n, call.k, call.gb
                ),
            });
        }
        Ok(Bytes::new(call.out.id, msg.out_elems, Dtype::F16, 2))
    }
}

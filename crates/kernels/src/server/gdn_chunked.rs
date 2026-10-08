//! GdnChunkedRuntime —— gdn_chunked 六核编排(服务端面;M3 旁路样板,
//! 2026-10-12)。自 server/foreign.rs::handle_gdn_chunked 全量移植:
//! 发射序/dtype 配方/scratch 几何/meta 表驻留逐行对齐,差异仅在
//! **Result 全链**(零 ack 侧信道)+ **Exec 横切**(捕获守卫/登记一次写成)。
//!
//! 配方(契约单源 = kernels::op::gdn_chunked + gdn_chunked.rs ABI 头):
//! cast-in q/k/v/beta→bf16、g→f32;发射 cumsum→kkt→merge→wu→h→o;
//! cast-out o(bf16)→out(f16)。
//!
//! ⚠️ **g 语义律**(坑册 §17):kkt/h/o 的 "g" 参数全部吃 **cumsum 产物**
//! (gcum);喂 raw gate = exp(±24) 爆炸。
//!
//! ⚠️ **A1.7 坟场律**:scratch 扩容换新块后旧块入坟场永不归还(捕获图
//! 持旧指针;scalar 案同款),直至 runtime 销毁。
//!
//! ⚠️ **ai_buf 字节池注**:merge 核写 bf16 字节、wu 按 bf16 读;历史
//! Rust 视角 f32 仅为字节池标注(新链 ScratchBuf 无 dtype 假装)。

use cudarc::driver::CudaFunction;
use crate::contract::{Bytes, LaunchMsg, OpError, OpId};
use crate::device::{DeviceRes, Exec, LaunchVal, ScratchBuf};

use crate::family::gdn_chunked::cubins;
use crate::family::gdn_chunked::cubins::launch as LC;
use crate::client::gdn_chunked::{GdnChunkedCall, GDN_CHUNKED};
use crate::registry::KernelSpec;
use std::collections::HashMap;
use std::sync::Arc;

const OP: OpId = OpId(GDN_CHUNKED);
const ASSET: &str = "gdn_chunked";
const CAST: &str = "owl_cast";
// 家族几何(HV/KD/VD)随调用 shape 单源(GdnShape);server 零字面量
// —— 旧 HV27/D128 常量声明后未用已删(2026-10-12 review G 案)

/// 私有 scratch(几何 = 生产 realloc 同款;ptr 句柄,账本在 res 侧)
#[derive(Default)]
struct Scratch {
    g_cum: Option<ScratchBuf>,       // t*hv f32
    a_buf: Option<ScratchBuf>,       // t*hv*64 f32
    ai_buf: Option<ScratchBuf>,      // t*hv*64 bf16 字节
    w: Option<ScratchBuf>,           // t*hv*kd bf16
    u: Option<ScratchBuf>,           // t*hv*vd bf16
    h_buf: Option<ScratchBuf>,       // nt*hv*vd*kd bf16
    v_new: Option<ScratchBuf>,       // t*hv*vd bf16
    state_t: Option<ScratchBuf>,     // hv*kd*vd f32(fork [V,K] 布局,首次定维)
    state_out_t: Option<ScratchBuf>, // hv*kd*vd f32
    in_q: Option<ScratchBuf>,        // t*nk*kd bf16
    in_k: Option<ScratchBuf>,        // t*nk*kd bf16
    in_v: Option<ScratchBuf>,        // t*hv*vd bf16
    in_beta: Option<ScratchBuf>,     // t*hv bf16
    in_g: Option<ScratchBuf>,        // t*hv f32
    o_b16: Option<ScratchBuf>,       // t*hv*vd bf16
    meta: HashMap<(usize, usize), ScratchBuf>, // (T,NT) 键 i64 表驻留
    grave: Vec<ScratchBuf>,          // A1.7 坟场
}

/// cast 发射(block 256;handler 同款;自由函数免长借用)
fn cast_exec(
    res: &mut dyn DeviceRes,
    exec: &mut Exec,
    f: &Arc<CudaFunction>,
    src: u64,
    n: usize,
    dst: u64,
) -> Result<(), OpError> {
    exec.launch(
        res, OP, f, "cast", (n.div_ceil(256) as u32, 1, 1), (256, 1, 1), 0,
        &[LaunchVal::Ptr(src), LaunchVal::I32(n as i32), LaunchVal::Ptr(dst)],
    )
}

/// 单 scratch 槽取/扩容(扩容 = 旧代入坟场;容量语义 = 字节)
fn ensure_one(
    cur: &mut Option<ScratchBuf>,
    grave: &mut Vec<ScratchBuf>,
    res: &mut dyn DeviceRes,
    bytes: usize,
    tag: &'static str,
) -> Result<u64, OpError> {
    if cur.as_ref().is_none_or(|b| b.bytes < bytes) {
        if let Some(old) = cur.take() {
            grave.push(old);
        }
        *cur = Some(res.alloc(bytes, tag)?);
    }
    Ok(cur.as_ref().expect("刚就位").ptr)
}

/// 家族 runtime(FamilyRuntime;注册于 backends::ops M4 切换面)
#[derive(Default)]
pub struct GdnChunkedRuntime {
    loaded: bool,
    fns: HashMap<&'static str, Arc<CudaFunction>>,
    scratch: Scratch,
}

impl KernelSpec for GdnChunkedRuntime {
    fn id(&self) -> OpId {
        OP
    }

    fn linkage(&self) -> crate::contract::Linkage {
        crate::contract::Linkage::AotCubin
    }

    /// boot 装配:六 cubin(符号逐一校验 = P4 门)+ cast nvrtc + setattr
    fn validate(&self) -> Result<(), OpError> { Ok(()) }
    fn init(&mut self, res: &mut dyn crate::device::DeviceRes, exec: &mut crate::device::Exec) -> Result<(), OpError> {
        if self.loaded {
            return Ok(());
        }
        let table: &[(&'static str, &[u8], &'static str)] = &[
            ("cumsum", cubins::CUMSUM, "chunk_local_cumsum_scalar_kernel"),
            ("kkt", cubins::KKT, "chunk_scaled_dot_kkt_fwd_kernel"),
            ("merge", cubins::MERGE, "merge_16x16_to_64x64_inverse_kernel"),
            ("wu", cubins::WU, "recompute_w_u_fwd_kernel"),
            ("h", cubins::H, "chunk_gated_delta_rule_fwd_kernel_h_blockdim64"),
            ("o", cubins::O, "chunk_fwd_kernel_o"),
        ];
        for (name, bytes, symbol) in table {
            let f = exec.load_cubin(res, ASSET, bytes, symbol, OP)?;
            self.fns.insert(name, f);
        }
        for sym in [
            "owl_cast_f16_bf16",
            "owl_cast_bf16_f16",
            "owl_cast_f16_f32",
            "owl_state_kv_to_vk",
            "owl_state_vk_to_kv",
        ] {
            let f = exec
                .load_nvrtc(res, CAST, crate::sources::attention::CAST, sym, OP)?;
            self.fns.insert(sym, f);
        }
        for (name, sz) in [
            ("wu", LC::WU_SHARED),
            ("h", LC::H_SHARED),
            ("o", LC::O_SHARED),
        ] {
            Exec::set_shared_limit(self.fns.get(name).expect("刚装载"), sz)?;
        }
        self.loaded = true;
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn run(&mut self, msg: &LaunchMsg, res: &mut dyn crate::device::DeviceRes, exec: &mut crate::device::Exec) -> Result<Bytes, OpError> {
        let (call, out) = GdnChunkedCall::parse(msg)?;
        let shape = call.shape();
        let t = shape.t as usize;
        let hv = shape.nv as usize;
        let nk = shape.nk as usize;
        let kd = shape.kd as usize;
        let vd = kd; // GDN:KD == VD(27B = 128)
        let nt = t.div_ceil(64);

        // 拆借:res / exec / self 字段三方独立可变借
        let Self {
            fns,
            scratch: st,
            loaded: _,
        } = self;
        let f = |name: &str| -> Result<Arc<CudaFunction>, OpError> {
            fns.get(name).cloned().ok_or_else(|| OpError::Asset {
                op: OP.0.to_string(),
                detail: format!("符号 {name} 未装载"),
            })
        };

        // ── scratch 容量(扩容 = 坟场;几何 = 生产 realloc 同款)──
        let g_cum = ensure_one(&mut st.g_cum, &mut st.grave, res, t * hv * 4, "gdn.g_cum")?;
        let a_buf = ensure_one(&mut st.a_buf, &mut st.grave, res, t * hv * 64 * 4, "gdn.a")?;
        let ai_buf = ensure_one(&mut st.ai_buf, &mut st.grave, res, t * hv * 64 * 2, "gdn.ai")?;
        let w = ensure_one(&mut st.w, &mut st.grave, res, t * hv * kd * 2, "gdn.w")?;
        let u = ensure_one(&mut st.u, &mut st.grave, res, t * hv * vd * 2, "gdn.u")?;
        let h_buf =
            ensure_one(&mut st.h_buf, &mut st.grave, res, nt * hv * vd * kd * 2, "gdn.h_buf")?;
        let v_new =
            ensure_one(&mut st.v_new, &mut st.grave, res, t * hv * vd * 2, "gdn.v_new")?;
        // 状态转置缓冲:与其他 scratch 同一 ensure 律(扩容 = 坟场;
        // 2026-10-12 review E 案:原一次性定容不随形状扩,换 hv/kd 即 OOB)
        let state_t =
            ensure_one(&mut st.state_t, &mut st.grave, res, hv * kd * vd * 4, "gdn.state_t")?;
        let state_out =
            ensure_one(&mut st.state_out_t, &mut st.grave, res, hv * kd * vd * 4, "gdn.state_out")?;
        let in_q = ensure_one(&mut st.in_q, &mut st.grave, res, t * nk * kd * 2, "gdn.in_q")?;
        let in_k = ensure_one(&mut st.in_k, &mut st.grave, res, t * nk * kd * 2, "gdn.in_k")?;
        let in_v = ensure_one(&mut st.in_v, &mut st.grave, res, t * hv * vd * 2, "gdn.in_v")?;
        let in_beta =
            ensure_one(&mut st.in_beta, &mut st.grave, res, t * hv * 2, "gdn.in_beta")?;
        let in_g = ensure_one(&mut st.in_g, &mut st.grave, res, t * hv * 4, "gdn.in_g")?;
        let o_b16 =
            ensure_one(&mut st.o_b16, &mut st.grave, res, t * hv * vd * 2, "gdn.o_b16")?;

        // ── 输入解析(块句柄 → 设备指针)──
        let q_p = res.resolve(call.q())?;
        let k_p = res.resolve(call.k())?;
        let v_p = res.resolve(call.v())?;
        let beta_p = res.resolve(call.beta())?;
        let g_p = res.resolve(&call.gate().0)?;
        let state_slot_p = res
            .resolve(&call.state().pool)?
            + (call.state().slot as u64) * (hv * kd * vd * 4) as u64;
        let out_p = res.resolve(&out)?;
        let scale = shape.scale();
        let ti = t as i32;
        let scratch: u64 = 0; // triton 尾参(global/profile scratch)

        // ── cast-in(dtype 统一律:handler 同款;block 256)──
        let f16bf16 = f("owl_cast_f16_bf16")?;
        let bf16f16 = f("owl_cast_bf16_f16")?;
        let f16f32 = f("owl_cast_f16_f32")?;
        cast_exec(res, exec, &f16bf16, q_p, t * nk * kd, in_q)?;
        cast_exec(res, exec, &f16bf16, k_p, t * nk * kd, in_k)?;
        cast_exec(res, exec, &f16bf16, v_p, t * hv * vd, in_v)?;
        cast_exec(res, exec, &f16bf16, beta_p, t * hv, in_beta)?;
        cast_exec(res, exec, &f16f32, g_p, t * hv, in_g)?;

        // ── meta 表驻留((T,NT) 键;handler 同款布局 [cu|coff|idx])──
        let (cu_p, coff_p, idx_p) = {
            let key = (t, nt);
            if !st.meta.contains_key(&key) {
                let mut host: Vec<i64> = Vec::with_capacity(4 + nt * 2);
                host.extend_from_slice(&[0i64, t as i64]);
                host.extend_from_slice(&[0i64, nt as i64]);
                host.extend((0..nt).flat_map(|i| [0i64, i as i64]));
                let nb = res.alloc(host.len() * 8, "gdn.meta")?;
                let bytes: Vec<u8> =
                    host.iter().flat_map(|v| v.to_le_bytes()).collect();
                res.upload(nb.ptr, &bytes)?;
                st.meta.insert(key, nb);
            }
            let base = st.meta.get(&key).unwrap().ptr;
            (base, base + 16, base + 32)
        };

        // ── 状态转置入:池 [HV,K,V] → fork [V,K](handler 同款)──
        let kv_vk = f("owl_state_kv_to_vk")?;
        let vk_kv = f("owl_state_vk_to_kv")?;
        {
            exec.launch(
                res, OP, &kv_vk, "state_kv_to_vk", (vd as u32, hv as u32, 1),
                (kd as u32, 1, 1), 0,
                &[LaunchVal::Ptr(state_slot_p), LaunchVal::Ptr(state_t),
                  LaunchVal::I32(kd as i32), LaunchVal::I32(vd as i32)],
            )?;
        }

        // ── 1. cumsum:raw g(f32 镜像)→ g_cum ──
        let f_cum = f("cumsum")?;
        exec.launch(
            res, OP, &f_cum, "cumsum", (nt as u32, hv as u32, 1),
            (32 * LC::CUMSUM_WARPS, 1, 1), LC::CUMSUM_SHARED,
            &[LaunchVal::Ptr(in_g), LaunchVal::Ptr(g_cum), LaunchVal::Ptr(cu_p),
              LaunchVal::Ptr(idx_p), LaunchVal::I32(ti),
              LaunchVal::Ptr(scratch), LaunchVal::Ptr(scratch)],
        )?;

        // ── 2. kkt:k/beta/g_cum → A(f32)【g 语义律:吃 cumsum 产物】──
        let f_kkt = f("kkt")?;
        exec.launch(
            res, OP, &f_kkt, "kkt", (nt as u32, hv as u32, 1),
            (32 * LC::KKT_WARPS, 1, 1), LC::KKT_SHARED,
            &[LaunchVal::Ptr(in_k), LaunchVal::Ptr(in_beta), LaunchVal::Ptr(g_cum),
              LaunchVal::Ptr(a_buf), LaunchVal::Ptr(cu_p), LaunchVal::Ptr(idx_p),
              LaunchVal::I32(ti), LaunchVal::Ptr(scratch), LaunchVal::Ptr(scratch)],
        )?;

        // ── 3. merge:solve_tril A → Ai(bf16 字节)──
        let f_merge = f("merge")?;
        exec.launch(
            res, OP, &f_merge, "merge", (nt as u32, hv as u32, 1),
            (32 * LC::MERGE_WARPS, 1, 1), LC::MERGE_SHARED,
            &[LaunchVal::Ptr(a_buf), LaunchVal::Ptr(ai_buf), LaunchVal::Ptr(cu_p),
              LaunchVal::Ptr(idx_p), LaunchVal::I32(ti),
              LaunchVal::Ptr(scratch), LaunchVal::Ptr(scratch)],
        )?;

        // ── 4. wu:k/v/beta/Ai/g_cum → w/u(bf16)──
        let f_wu = f("wu")?;
        exec.launch(
            res, OP, &f_wu, "wu", (nt as u32, hv as u32, 1),
            (32 * LC::WU_WARPS, 1, 1), LC::WU_SHARED,
            &[LaunchVal::Ptr(in_k), LaunchVal::Ptr(in_v), LaunchVal::Ptr(in_beta),
              LaunchVal::Ptr(w), LaunchVal::Ptr(u), LaunchVal::Ptr(ai_buf),
              LaunchVal::Ptr(g_cum), LaunchVal::Ptr(cu_p), LaunchVal::Ptr(idx_p),
              LaunchVal::I32(ti), LaunchVal::Ptr(scratch), LaunchVal::Ptr(scratch)],
        )?;

        // ── 5. h(SASS 审计序;grid 二维)──
        let f_h = f("h")?;
        exec.launch(
            res, OP, &f_h, "h", ((vd as u32).div_ceil(LC::H_BV), hv as u32, 1),
            (32 * LC::H_WARPS, 1, 1), LC::H_SHARED,
            &[LaunchVal::Ptr(in_k), LaunchVal::Ptr(u), LaunchVal::Ptr(w),
              LaunchVal::Ptr(v_new), LaunchVal::Ptr(g_cum), LaunchVal::Ptr(h_buf),
              LaunchVal::Ptr(state_t), LaunchVal::Ptr(state_out),
              LaunchVal::Ptr(cu_p), LaunchVal::Ptr(coff_p), LaunchVal::I32(ti),
              LaunchVal::Ptr(scratch), LaunchVal::Ptr(scratch)],
        )?;

        // ── 6. 状态转置回:fork [V,K] → 池 [K,V] ──
        {
            exec.launch(
                res, OP, &vk_kv, "state_vk_to_kv", (vd as u32, hv as u32, 1),
                (kd as u32, 1, 1), 0,
                &[LaunchVal::Ptr(state_out), LaunchVal::Ptr(state_slot_p),
                  LaunchVal::I32(kd as i32), LaunchVal::I32(vd as i32)],
            )?;
        }

        // ── 7. o:q/k/v_new/h_buf/g_cum → o_b16(bf16)──
        let f_o = f("o")?;
        exec.launch(
            res, OP, &f_o, "o", ((vd as u32).div_ceil(LC::O_BV), nt as u32, hv as u32),
            (32 * LC::O_WARPS, 1, 1), LC::O_SHARED,
            &[LaunchVal::Ptr(in_q), LaunchVal::Ptr(in_k), LaunchVal::Ptr(v_new),
              LaunchVal::Ptr(h_buf), LaunchVal::Ptr(g_cum), LaunchVal::Ptr(o_b16),
              LaunchVal::Ptr(cu_p), LaunchVal::Ptr(idx_p), LaunchVal::F32(scale),
              LaunchVal::I32(ti), LaunchVal::Ptr(scratch), LaunchVal::Ptr(scratch)],
        )?;

        // ── 8. cast-out:o(bf16)→ out(f16;层侧纯 f16 契约)──
        cast_exec(res, exec, &bf16f16, o_b16, t * hv * vd, out_p)?;

        Ok(out)
    }
}

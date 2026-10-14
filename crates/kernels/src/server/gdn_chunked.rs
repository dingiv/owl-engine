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
use crate::registry::{FamilyGuard, KernelFamily};
use std::collections::HashMap;
use std::sync::Arc;

const OP: OpId = OpId(GDN_CHUNKED);
const ASSET: &str = "gdn_chunked";
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
    /// 诊断仪(OWL_GDN_DIAG):**逐 call 打印**前 N 次(跨臂对齐比较用;
    /// 按形状去重会看不到同臂逐 chunk 的演化)
    diag_calls: usize,
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

/// 诊断门(2026-10-14 A13 立案;缺省关)。
///
/// ⚠️ **回读纪律(本轮两次踩中)**:本仪表用 `DeviceRes::download`
/// (= `cuMemcpyDtoH`,**NULL 流**语义)—— 与 compute 流**无跨流序**,
/// 不加 `res.stream()?.synchronize()` 会读到**旧值/全零**;即便加了
/// sync,个别 run 仍观察到 stale(疑 host 跑在 GPU 前面)⇒ 判读时
/// **同一量读两次比对**,不一致即弃用该次读数。
///
/// 背景:chunked 臂(六核)在**同臂同 T、数据不同**时呈「末行 logits 塔平」
/// 式的塔陷(same prompt:T=512 塔陷 / T=64 正确),而 varlen 臂恒健康。
/// 本仪表只回答三个是非题(不猜):
///   1. `g_cum` 是**chunk 局部**(每 chunk 从头累计,锯齿)还是**全局**累计?
///   2. `g_cum` 的**域**是 nats 还是 log2(host 双口径重算对拍)?
///   3. 跨 chunk 状态(h0→ht)有没有被 `exp(g_cum_last)` **清零**?
static DIAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
fn diag_on() -> bool {
    // 零依赖律:kernels crate 不得依赖 owl_shared ⇒ 就地读(与 driver.rs
    // 的 OWL_RESOLVE_TRACE 同一先例;诊断面一次性 OnceLock,非热路径)
    *DIAG.get_or_init(|| std::env::var_os("OWL_GDN_DIAG").is_some_and(|v| v != "0"))
}

/// bf16 字节 → f32(零依赖律:本 crate 不带 half;bf16 = f32 高 16 位)
#[inline]
fn bf16_to_f32(lo: u8, hi: u8) -> f32 {
    f32::from_bits((u16::from_le_bytes([lo, hi]) as u32) << 16)
}

/// 诊断:`OWL_GDN_ZERO_SCRATCH=1` ⇒ 每次调用前把全部私有 scratch 清零。
/// 动机:boot warmup 用**未初始化输入**跑过本 handler(实测 raw g=-200/T=512)
/// ⇒ 未写区可能残留 inf/NaN;真实调用若读到该区即被毒(且任何布局扰动都会
/// 改掉重叠关系 = "布局运气"）。
fn zero_scratch_on() -> bool {
    static Z: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *Z.get_or_init(|| std::env::var_os("OWL_GDN_ZERO_SCRATCH").is_some_and(|v| v != "0"))
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// (min, max, 非零个数, sum)
fn stat(v: &[f32]) -> (f32, f32, usize, f64) {
    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;
    let mut nz = 0usize;
    let mut s = 0f64;
    for &x in v {
        if x < mn {
            mn = x;
        }
        if x > mx {
            mx = x;
        }
        if x != 0.0 {
            nz += 1;
        }
        s += x as f64;
    }
    (mn, mx, nz, s)
}

/// 单 scratch 槽取/扩容(扩容 = 旧代入坟场;容量语义 = 字节)
fn ensure_one(
    cur: &mut Option<ScratchBuf>,
    grave: &mut Vec<ScratchBuf>,
    res: &mut dyn DeviceRes,
    bytes: usize,
    tag: &'static str,
) -> Result<u64, OpError> {
    // 诊断(2026-10-14 A13):`OWL_GDN_FRESH_SCRATCH=1` ⇒ 每 call 新分配
    // (alloc_zeros ⇒ 必零),用来判「复用 scratch 里有没有被读到 stale 区」
    static FRESH: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let fresh = *FRESH
        .get_or_init(|| std::env::var_os("OWL_GDN_FRESH_SCRATCH").is_some_and(|v| v != "0"));
    if fresh {
        if let Some(old) = cur.take() {
            grave.push(old);
        }
        *cur = Some(res.alloc(bytes, tag)?);
        return Ok(cur.as_ref().expect("刚就位").ptr);
    }
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
    fns: HashMap<&'static str, Arc<CudaFunction>>,
    scratch: Scratch,
}

impl GdnChunkedRuntime {
    /// **本族身份**(族名 = 线契约名;字面量住 `family::gdn_chunked`,零门)
    /// —— 声明面 `ops::gdn_chunked::*::FAMILY` 与本处引用**同一字面量**。
    pub const ID: OpId = OpId(crate::family::gdn_chunked::GDN_CHUNKED_FWD);
}


impl KernelFamily for GdnChunkedRuntime {
    fn id(&self) -> OpId {
        Self::ID
    }


    /// 构造函数:六 cubin(符号逐一校验 = 装配门)+ cast nvrtc + setattr
    fn assemble(
        res: &mut dyn crate::device::DeviceRes,
        exec: &mut crate::device::Exec,
    ) -> Result<Box<dyn KernelFamily>, OpError> {
        let mut me = GdnChunkedRuntime::default();
        // 资产取用:**构建期 AOT 自编**(缺 = 装配期拒启,见下)
        let table: &[(&'static str, Option<&'static [u8]>, &'static str)] = &[
            ("cumsum", cubins::cumsum(), "chunk_local_cumsum_scalar_kernel"),
            ("kkt", cubins::kkt(), "chunk_scaled_dot_kkt_fwd_kernel"),
            ("merge", cubins::merge(), "merge_16x16_to_64x64_inverse_kernel"),
            ("wu", cubins::wu(), "recompute_w_u_fwd_kernel"),
            ("h", cubins::h(), "chunk_gated_delta_rule_fwd_kernel_h_blockdim64"),
            ("o", cubins::o(), "chunk_fwd_kernel_o"),
        ];
        let missing: Vec<&str> =
            table.iter().filter(|(_, b, _)| b.is_none()).map(|(n, _, _)| *n).collect();
        if !missing.is_empty() {
            return Err(OpError::Asset {
                op: OP.0.to_string(),
                detail: format!(
                    "gdn_chunked 资产缺失 {missing:?}(六核 = {:?})—— 构建期 AOT 未跑?\
                     缺省应自动自编(需 Python+triton,`GDN_TRITON_PY` 可指解释器);\
                     `GDN_TRITON_BUILD=0` 会显式跳过 ⇒ 本族不可用。见 triton/README.md",
                    cubins::NAMES
                ),
            });
        }
        for (name, bytes, symbol) in table {
            let bytes = bytes.expect("上方已查缺失");
            let f = exec.load_cubin(res, ASSET, bytes, symbol, OP)?;
            me.fns.insert(name, f);
        }
        // 构建期资产(源哈希查表;`cu/owl/cast.cu` + `cu/owl/state_transpose.cu`
        // 都已在 include_str! 闭包内)。源按入口名取自算子自托管面(R5);
        // C 轮拆分后 cast 与状态转置**各自成文件** ⇒ 两个源各装一撮符号。
        for (src, syms) in [
            (
                crate::ops::source("owl_cast_f16_bf16"),
                &["owl_cast_f16_bf16", "owl_cast_bf16_f16", "owl_cast_f16_f32"][..],
            ),
            (
                // 家族私有核:不走算子准入面(故没有 `ops::source` 出口),
                // 常量就近住家族模块(crate::family::gdn_chunked::asset)
                crate::family::gdn_chunked::asset::STATE_TRANSPOSE,
                &["owl_state_kv_to_vk", "owl_state_vk_to_kv"][..],
            ),
        ] {
            for sym in syms {
                let f = exec.load_aot(res, src, sym, OP)?;
                me.fns.insert(sym, f);
            }
        }
        for (name, sz) in [
            ("wu", LC::WU_SHARED),
            ("h", LC::H_SHARED),
            ("o", LC::O_SHARED),
        ] {
            Exec::set_shared_limit(me.fns.get(name).expect("刚装载"), sz)?;
        }
        Ok(FamilyGuard::boxed(me))
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

        // ── 诊断:`OWL_GDN_TOUCH=<区名>` = **内容不变**的 dtoh+htod 往返搬运
        // (只为把 host 挡住 ⇒ 分离「stale 内容」与「host-ahead 时序」两假设:
        //  搬运后仍病 = 内容;搬运后变健康 = 时序窗口)──
        if let Ok(spec) = std::env::var("OWL_GDN_TOUCH") {
            let all = spec == "1" || spec == "all";
            let want = |n: &str| all || spec.split(',').any(|x| x.trim() == n);
            let bufs: [(u64, usize); 15] = [
                (g_cum, t * hv * 4),
                (a_buf, t * hv * 64 * 4),
                (ai_buf, t * hv * 64 * 2),
                (w, t * hv * kd * 2),
                (u, t * hv * vd * 2),
                (h_buf, nt * hv * vd * kd * 2),
                (v_new, t * hv * vd * 2),
                (state_t, hv * kd * vd * 4),
                (state_out, hv * kd * vd * 4),
                (in_q, t * nk * kd * 2),
                (in_k, t * nk * kd * 2),
                (in_v, t * hv * vd * 2),
                (in_beta, t * hv * 2),
                (in_g, t * hv * 4),
                (o_b16, t * hv * vd * 2),
            ];
            let names = [
                "g_cum", "a", "ai", "w", "u", "h_buf", "v_new", "state", "state", "in", "in",
                "in", "in", "g_cum", "o",
            ];
            for (i, (ptr, bytes)) in bufs.into_iter().enumerate() {
                if !want(names[i]) {
                    continue;
                }
                let mut buf = vec![0u8; bytes];
                res.download(ptr, &mut buf)?;
                res.upload(ptr, &buf)?;
            }
        }

        // ── 诊断:全 scratch 清零(见 zero_scratch_on 注)──
        if zero_scratch_on() {
            // 区域过滤:OWL_GDN_ZERO_SCRATCH=1|all = 全清;或逗号名单
            // (h_buf/a/ai/w/u/v_new/o/g_cum/state/in*)= 只清该区(分区定位用)
            let spec = std::env::var("OWL_GDN_ZERO_SCRATCH").unwrap_or_default();
            let all = spec == "1" || spec == "all" || spec.is_empty();
            let want = |n: &str| all || spec.split(',').any(|x| x.trim() == n);
            let bufs: [(u64, usize); 15] = [
                (g_cum, t * hv * 4),
                (a_buf, t * hv * 64 * 4),
                (ai_buf, t * hv * 64 * 2),
                (w, t * hv * kd * 2),
                (u, t * hv * vd * 2),
                (h_buf, nt * hv * vd * kd * 2),
                (v_new, t * hv * vd * 2),
                (state_t, hv * kd * vd * 4),
                (state_out, hv * kd * vd * 4),
                (in_q, t * nk * kd * 2),
                (in_k, t * nk * kd * 2),
                (in_v, t * hv * vd * 2),
                (in_beta, t * hv * 2),
                (in_g, t * hv * 4),
                (o_b16, t * hv * vd * 2),
            ];
            let names = [
                "g_cum", "a", "ai", "w", "u", "h_buf", "v_new", "state", "state", "in", "in",
                "in", "in", "g_cum", "o",
            ];
            let mut zbuf: Vec<u8> = Vec::new();
            for (i, (ptr, bytes)) in bufs.into_iter().enumerate() {
                if !want(names[i]) {
                    continue;
                }
                if zbuf.len() < bytes {
                    zbuf.resize(bytes, 0);
                }
                res.upload(ptr, &zbuf[..bytes])?;
            }
        }

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
        let diag = diag_on() && st.diag_calls < 48;
        let diag_tag = if diag {
            st.diag_calls += 1;
            format!(
                "#c{} cap={}",
                st.diag_calls,
                res.capturing().unwrap_or(false)
            )
        } else {
            String::new()
        };

        // ── meta 表驻留((T,NT) 键;handler 同款布局 [cu|coff|idx])──
        let (cu_p, coff_p, idx_p) = {
            let key = (t, nt);
            if let std::collections::hash_map::Entry::Vacant(e) = st.meta.entry(key) {
                let mut host: Vec<i64> = Vec::with_capacity(4 + nt * 2);
                host.extend_from_slice(&[0i64, t as i64]);
                host.extend_from_slice(&[0i64, nt as i64]);
                host.extend((0..nt).flat_map(|i| [0i64, i as i64]));
                let nb = res.alloc(host.len() * 8, "gdn.meta")?;
                let bytes: Vec<u8> =
                    host.iter().flat_map(|v| v.to_le_bytes()).collect();
                res.upload(nb.ptr, &bytes)?;
                e.insert(nb);
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

        // ── 诊断仪(1):g 域/作用域 + 跨 chunk 态是否被清零 ──
        if diag {
            // ⚠️ 回读前必须显式 sync:memcpy_dtoh_sync 在 NULL 流,与 compute
            // 流不同流 ⇒ 不 sync 会读到半写入/全零(本仓 2026-10-13 同病史)
            res.stream()?.synchronize().map_err(|e| OpError::Launch {
                op: OP.0.to_string(), stage: crate::contract::Stage::Load,
                detail: format!("diag sync: {e:?}"),
            })?;
            let mut gb = vec![0u8; t * hv * 4];
            res.download(in_g, &mut gb)?;
            // 判别:in_k(本 call 刚 cast,必非零)与 g_cum 二次回读(睡 50ms)
            let mut kb = vec![0u8; t * nk * kd * 2];
            res.download(in_k, &mut kb)?;
            let knz = kb.iter().filter(|&&b| b != 0).count();
            let mut cb = vec![0u8; t * hv * 4];
            res.download(g_cum, &mut cb)?;
            let cnz1 = cb.iter().filter(|&&b| b != 0).count();
            std::thread::sleep(std::time::Duration::from_millis(50));
            res.download(g_cum, &mut cb)?;
            let cnz2 = cb.iter().filter(|&&b| b != 0).count();
            eprintln!("[gdn-diag] 回读判别: in_k 非零字节={knz}/{} | g_cum 非零字节 立即={cnz1} 睡50ms后={cnz2}",
                kb.len());
            let (gv, cv) = (f32s(&gb), f32s(&cb));
            let (gmn, gmx, gnz, _) = stat(&gv);
            let (cmn, cmx, cnz, _) = stat(&cv);
            eprintln!(
                "[gdn-diag] {diag_tag} T={t} NT={nt} raw g: min={gmn:.4} max={gmx:.4} nz={gnz}/{} | g_cum: min={cmn:.4} max={cmx:.4} nz={cnz}/{}",
                gv.len(), cv.len()
            );
            // 逐 chunk(头 0):首/末值 —— 局部累计 = 每 chunk 首值≈0(锯齿);
            // 全局累计 = 首值单调递减。同时给 host 双口径(local-nats /
            // local-log2)重算,g_cum 与哪个逐元素对得上 = 域定谳
            let h0i = |t_i: usize| t_i * hv; // 头 0 在 [T,HV] 里的下标
            let mut dev_first = Vec::new();
            let mut dev_last = Vec::new();
            for i_t in 0..nt {
                let (s, e) = (i_t * 64, (((i_t + 1) * 64).min(t)).saturating_sub(1));
                dev_first.push(cv[h0i(s)]);
                dev_last.push(cv[h0i(e)]);
            }
            let tail: Vec<String> = dev_first
                .iter()
                .zip(&dev_last)
                .map(|(a, b)| format!("{a:.2}/{b:.2}"))
                .collect();
            eprintln!("[gdn-diag] 头0 各 chunk g_cum 首/末: {}", tail.join(" "));
            // host 重算(头 0):local cumsum(nats)与 local cumsum(log2)
            let mut loc_nats = vec![0f32; t];
            let mut loc_log2 = vec![0f32; t];
            for i_t in 0..nt {
                let mut a = 0f32;
                let mut b = 0f32;
                for k in (i_t * 64)..(((i_t + 1) * 64).min(t)) {
                    a += gv[h0i(k)];
                    b += gv[h0i(k)] * std::f32::consts::LOG2_E;
                    loc_nats[k] = a;
                    loc_log2[k] = b;
                }
            }
            let md = |w: &[f32]| -> f32 {
                (0..t).map(|k| (cv[h0i(k)] - w[k]).abs()).fold(0f32, f32::max)
            };
            eprintln!(
                "[gdn-diag] 与 host 重算 max|Δ|: local-nats={:.3e} local-log2={:.3e}(×ln2 口径差)",
                md(&loc_nats),
                md(&loc_log2)
            );
            let mut h0b = vec![0u8; hv * kd * vd * 4];
            res.download(state_t, &mut h0b)?;
            let (amn, amx, anz, _) = stat(&f32s(&h0b));
            eprintln!("[gdn-diag] 入态 h0([V,K] 转置后): min={amn:.4} max={amx:.4} nz={anz}/{}", hv * kd * vd);
        }

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

        // ── 诊断仪(2):终态 ht 与出块 —— 态被清零/爆掉 = 一眼可见 ──
        if diag {
            res.stream()?.synchronize().map_err(|e| OpError::Launch {
                op: OP.0.to_string(), stage: crate::contract::Stage::Load,
                detail: format!("diag sync: {e:?}"),
            })?;
            // h_buf 逐 chunk block(inter-chunk 态):未写 = 全零(alloc_zeros)
            // ⇒ o 核会读到 0(态贡献消失)= 塔平签名
            let blk = hv * vd * kd * 2;
            let mut hblk = vec![0u8; blk];
            let mut hline: Vec<String> = Vec::with_capacity(nt);
            for i_t in 0..nt.min(12) {
                res.download(h_buf + (i_t * blk) as u64, &mut hblk)?;
                let vs: Vec<f32> = hblk
                    .chunks_exact(2)
                    .map(|c| bf16_to_f32(c[0], c[1]))
                    .collect();
                let (mn, mx, nz, _) = stat(&vs);
                hline.push(format!("#{i_t}:{mx:.2}/{nz}"));
            }
            eprintln!("[gdn-diag] {diag_tag} h_buf 逐 chunk(max|·|/非零数, blk={blk}B): {}", hline.join(" "));
            let mut hb = vec![0u8; hv * kd * vd * 4];
            res.download(state_out, &mut hb)?;
            let (hmn, hmx, hnz, _) = stat(&f32s(&hb));
            let mut ob = vec![0u8; t * hv * vd * 2];
            res.download(o_b16, &mut ob)?;
            let ov: Vec<f32> = ob
                .chunks_exact(2)
                .map(|c| bf16_to_f32(c[0], c[1]))
                .collect();
            let (omn, omx, onz, _) = stat(&ov);
            eprintln!(
                "[gdn-diag] 终态 ht(fork [V,K]): min={hmn:.4} max={hmx:.4} nz={hnz}/{} | o(bf16→f32): min={omn:.4} max={omx:.4} nz={onz}/{}",
                hv * kd * vd,
                ov.len()
            );
        }

        // ── 8. cast-out:o(bf16)→ out(f16;层侧纯 f16 契约)──
        cast_exec(res, exec, &bf16f16, o_b16, t * hv * vd, out_p)?;

        Ok(out)
    }
}

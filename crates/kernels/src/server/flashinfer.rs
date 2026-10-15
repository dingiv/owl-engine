//! FlashInfer paged prefill runtime(FFI 直调链路;plan/run 分离)。
//!
//! init:workspace 预钉(float 64MB + int 32MB;构造期,捕获内首次初始化
//! 事故类结构性消灭)。run:plan(每形状缓存,键 = 形状七元组)→ run
//! (f16/fp8kv 按名路由)。

use crate::contract::{Bytes, Dtype, LaunchMsg, OpError, OpId};
use crate::device::ScratchBuf;

use crate::client::flashinfer::{parse_prefill, PREFILL_FI};
use crate::family::flashinfer::{
    owl_fi_prefill_plan, owl_fi_prefill_run, owl_fi_prefill_run_fp8kv,
    FI_FLOAT_WS_BYTES, FI_HOST_STAGING_BYTES, FI_INT_WS_BYTES,
};
use crate::registry::{FamilyGuard, KernelFamily};

#[derive(Clone, Copy, PartialEq, Debug)]
struct PlanKey {
    total_rows: usize,
    /// **split 关时恒 0**(归一化):源码级 F-5 —— split 关时 plan 的数组
    /// 只由 qo 长度定,ctx 三条链(`kv_len_arr`/`effective_kv_len_arr`/
    /// `kv_chunk_size`)只在 split 分支被消费 ⇒ 键去 ctx 后,**一个桶一份 plan
    /// 跨 ctx 复用**(M2b 的 P-1)。split 开时必须留着(那时 ctx 真参与)。
    ctx_total: usize,
    /// **图档 plan**(`FI_CUDA_GRAPH`;见 common.h `owl_fi_cuda_graph`)
    /// —— 数组与 `padded_batch_size` 都与 eager 档不同,必须分键,否则
    /// 一次 eager 调用会把图档 plan 挤掉(mask 内容随之失配)
    graph: bool,
    /// split-kv 开关(`FI_SPLIT_KV`;与 C++ `owl_fi_split_kv_enabled` 同源)——
    /// 决定 ctx 是否进键(见上),必须显式入键,否则切档后复用错 plan
    split: bool,
    t: usize,
    hq: usize,
    hkv: usize,
    hd: usize,
    page: usize,
}

/// 图档 plan 开关(与 C++ `owl_fi_cuda_graph` 同源:env `FI_CUDA_GRAPH`)。
/// plan 期读一次(非热路径);两处必须一致,否则 PlanKey 与真数组错档。
fn fi_cuda_graph() -> bool {
    std::env::var_os("FI_CUDA_GRAPH").is_some_and(|v| !v.is_empty() && v != "0")
}

/// split-kv 开关(与 C++ `owl_fi_split_kv_enabled` 同源:env `FI_SPLIT_KV`)
fn fi_split_kv() -> bool {
    std::env::var_os("FI_SPLIT_KV").is_some_and(|v| !v.is_empty() && v != "0")
}

/// **plan 键归一化(纯函数;可单测)**:split 关 ⇒ 抹掉 `ctx_total`
/// (源码级 F-5:那三条 ctx 链只在 split 分支被消费)⇒ 一个桶一份 plan 跨 ctx
/// 复用(M2b 的 P-1);split 开 ⇒ ctx 真参与定瓦,必须进键。
fn plan_key(
    total_rows: usize,
    ctx_total: usize,
    t: usize,
    hq: usize,
    hkv: usize,
    hd: usize,
    page: usize,
    graph: bool,
    split: bool,
) -> PlanKey {
    PlanKey {
        total_rows,
        ctx_total: if split { ctx_total } else { 0 },
        t,
        hq,
        hkv,
        hd,
        page,
        graph,
        split,
    }
}

/// 诊断打印开关(与 C++ `owl_fi_debug` 同源:env `FI_DEBUG`)
fn fi_debug() -> bool {
    std::env::var_os("FI_DEBUG").is_some_and(|v| !v.is_empty() && v != "0")
}

impl FiRuntime {
    /// **本族身份**(族名 = 线契约名;字面量住 `family::flashinfer`,零门)
    pub const ID: OpId = OpId(crate::family::flashinfer::PREFILL_FI);

    // 本族算子表:由 `KernelSpec::FAMILY` 派生(f16 + fp8 两臂)
}

pub struct FiRuntime {
    float_ws: Option<ScratchBuf>,
    int_ws: Option<ScratchBuf>,
    host_staging: Vec<u8>,
    /// plan 缓存(**单条**)+ 它的**尾部 CTA 掩码**(P-5)。为什么 mask 必须
    /// 与 plan 同车:MASK 内容 = `i < tiles`,而 `tiles/padded` 都是 plan 的
    /// 函数;若共用一个 mask 缓冲,一次换形的 eager 调用会把上一条 plan 的
    /// 掩码改写 ⇒ **回放旧图时掩码失配 ⇒ 真行漏算**(静默错值)。
    plan: Option<(PlanKey, [i64; 15], ScratchBuf, usize /* tiles */)>,
}

impl Default for FiRuntime {
    fn default() -> Self {
        Self {
            float_ws: None,
            int_ws: None,
            host_staging: vec![0u8; FI_HOST_STAGING_BYTES],
            plan: None,
        }
    }
}


impl KernelFamily for FiRuntime {
    fn id(&self) -> OpId {
        Self::ID
    }

    /// 本族**已建表**算子(fp8 臂;f16 臂未建表 —— 它由族主名承担,
    /// 见 `KernelFamily::names()` 的 id ∪ ops 律)


    /// 构造函数:workspace 预钉(float 64MB + int 32MB;构造期,
    /// 捕获内首次初始化事故类结构性消灭)
    fn assemble(
        res: &mut dyn crate::device::DeviceRes,
        _exec: &mut crate::device::Exec,
    ) -> Result<Box<dyn KernelFamily>, OpError> {
        let me = FiRuntime {
            float_ws: Some(res.alloc(FI_FLOAT_WS_BYTES, "fi.float_ws")?),
            int_ws: Some(res.alloc(FI_INT_WS_BYTES, "fi.int_ws")?),
            host_staging: vec![0u8; FI_HOST_STAGING_BYTES],
            plan: None,
        };
        Ok(FamilyGuard::boxed(me))
    }

    fn run(&mut self, msg: &LaunchMsg, res: &mut dyn crate::device::DeviceRes, _exec: &mut crate::device::Exec) -> Result<Bytes, OpError> {
        let call = parse_prefill(msg)?;
        let r = |b: &crate::client::flashinfer::BlockRef| {
            res.resolve(b.id).map(|p| p + b.byte_offset)
        };
        let q = r(&call.q)?;
        let kc = r(&call.kc_fi)?;
        let vc = r(&call.vc)?;
        let q_cu = r(&call.q_cu)?;
        let indices = r(&call.indices)?;
        let indptr = r(&call.indptr)?;
        let last_len = r(&call.last_len)?;
        let out_ref = crate::client::flashinfer::BlockRef { id: call.out.id, byte_offset: 0 };
        let out_ptr = r(&out_ref)?;
        // 就地解包( locals 贯穿 plan/run;None = init 未跑,Asset 拒)
        let float_ws = self.float_ws.as_ref().ok_or_else(|| OpError::Asset {
            op: PREFILL_FI.into(),
            detail: "workspace 未装配".into(),
        })?;
        let int_ws = self.int_ws.as_ref().ok_or_else(|| OpError::Asset {
            op: PREFILL_FI.into(),
            detail: "workspace 未装配".into(),
        })?;
        let stream = res.stream()?;

        // plan(host,每形状一次;键 = 形状七元组)
        let key = plan_key(
            call.total_rows,
            call.ctx_total,
            call.t,
            call.hq,
            call.hkv,
            call.hd,
            call.page,
            fi_cuda_graph(),
            fi_split_kv(),
        );
        // ⚠️ **plan 必须在捕获窗之外算**(2026-10-14 实测 + 结构性守卫;自
        // `feat/prefill-graph` 合入)。
        //
        // 事实(本机实测,`cudaStreamCaptureModeGlobal`,sm_86):
        // `cudaMemcpyAsync` 的 **H2D/D2H × pinned/pageable 四组合全部可被捕获**
        // (各成 1 个 memcpy 节点,回放 3 次值正确);不可捕获的只有**同步**
        // `cudaMemcpy` 与 `cudaMalloc/Free`。⇒ plan 内部的
        // `cudaMemcpyAsync(int_ws ← page_locked staging)` **会被烘进图**,而
        // `host_staging` 是**可变**缓冲(换形状会重写)—— 一旦 plan 在窗内算,
        // 图里就留下"从可变 host 缓冲拷贝"的节点,回放期取**当时**的内容
        // ⇒ 错调度表/越界,且**不报错**。
        //
        // 现状为何没炸:图捕获前有一次 warmup eager(`GraphPlan::plan`),plan 已
        // 缓存 ⇒ 捕获期不重算。但那是"时序恰好",不是不变量 —— 故此处立硬门:
        // 捕获期若 plan 未命中,**结构化拒**(→ 上层 `Ok(None)` → eager 原路),
        // 不静默、不把错节点烘进图。
        // (更强做法:每形状一份独立 staging 缓冲 ⇒ 节点源恒定;待有需求再上。)
        if self.plan.as_ref().map(|(k, ..)| *k != key).unwrap_or(true) {
            if res.capturing().unwrap_or(false) {
                return Err(OpError::Launch {
                    op: msg.kernel.name.clone(),
                    stage: crate::contract::Stage::Load,
                    detail: "FI plan 未缓存而处于捕获窗:plan 内含 H2D(可被捕获),\
                             必须在窗口外算 —— 拒捕回 eager".to_string(),
                });
            }
            let qo_indptr = [0i32, call.t as i32];
            let kv_indptr = [0i32, call.ctx_total as i32];
            let (mut plan15, mut cta_tile_q, mut split_kv) = ([0i64; 15], 0i32, 0i32);
            let rc = unsafe {
                owl_fi_prefill_plan(
                    float_ws.ptr as *mut std::ffi::c_void,
                    float_ws.bytes,
                    int_ws.ptr as *mut std::ffi::c_void,
                    int_ws.bytes,
                    self.host_staging.as_mut_ptr() as *mut std::ffi::c_void,
                    self.host_staging.len(),
                    plan15.as_mut_ptr(),
                    &mut cta_tile_q,
                    &mut split_kv,
                    qo_indptr.as_ptr(),
                    kv_indptr.as_ptr(),
                    call.total_rows as i32,
                    1, // batch = 1(owl 单会话)
                    call.hq as i32,
                    call.hkv as i32,
                    call.hd as i32,
                    call.page as i32,
                    stream.cu_stream() as *mut std::ffi::c_void,
                )
            };
            if rc != 0 {
                return Err(OpError::Launch {
                    op: msg.kernel.name.clone(),
                    stage: crate::contract::Stage::Load,
                    detail: format!(
                        "plan err {rc}(hd={} page={} T={} ctx={})",
                        call.hd, call.page, call.t, call.ctx_total
                    ),
                });
            }
            // ── **P-5:尾部 CTA 掩码**(2026-10-15;设计 §4.1c F-6/F-7/F-8)
            //
            // 图档 plan 的 `padded_batch_size = max(2·SM/hkv, tiles)` ⇒ 尾部
            // CTA 数可以远超真 tile 数(t_pad=128/3090:41 vs 12)。不掩码时
            // 它们读 plan 表的**残留表项**(行为取决于上一次 plan 写了什么 =
            // 静默);掩码后核侧 `if (!mask[bx]) return;` 立即退。
            //
            // 内容 `i < tiles` 由**本侧**算(唯一事实源 = plan15):
            //   tiles = ceil_div(packed_qo_len, cta_tile_q),packed = t·gqa
            // (`batch=1`;多请求时须逐请求求和 —— 本仓 FI 调用恒 batch=1,
            //  非 1 在此结构化拒,不给"算错了但看着对"的余地)
            let gqa = call.hq / call.hkv.max(1);
            let (padded, cta_tile_q) = (plan15[0].max(0) as usize, plan15[3].max(0) as usize);
            if gqa == 0 || cta_tile_q == 0 {
                return Err(OpError::Contract {
                    op: PREFILL_FI.into(),
                    field: "plan15",
                    expect: "cta_tile_q > 0 且 gqa ≥ 1".into(),
                    got: format!("gqa={gqa} cta_tile_q={cta_tile_q}"),
                });
            }
            let tiles = (call.total_rows * gqa).div_ceil(cta_tile_q).max(1);
            let mask = res.alloc(padded.max(1), "fi.mask")?;
            let mut m = vec![0u8; padded.max(1)];
            for (i, v) in m.iter_mut().enumerate() {
                *v = u8::from(i < tiles);
            }
            res.upload(mask.ptr, &m)?;
            if padded > tiles && fi_debug() {
                eprintln!(
                    "[owl-fi][mask] 图档 plan:T={} ctx={} padded={} tiles={} ⇒ 掩 {} 个尾部 CTA",
                    call.t,
                    call.ctx_total,
                    padded,
                    tiles,
                    padded - tiles
                );
            }
            self.plan = Some((key, plan15, mask, tiles));
        }
        let (plan15, mask_ptr) = {
            let (_, p, m, _) = self.plan.as_ref().expect("plan 缓存");
            (*p, m.ptr)
        };

        let run = if call.fp8kv {
            owl_fi_prefill_run_fp8kv
        } else {
            owl_fi_prefill_run
        };
        // 家族面入账(A13 补漏):FlashInfer = **直连 FFI 发射**(plan/run 均不经
        // `Exec::launch`),账上本无痕;ptrs = q/kc/vc/out 四块地址。
        res.note_device("fam_ffi_launch", &msg.kernel.name, 0, &[q, kc, vc, out_ptr]);
        let rc = unsafe {
            run(
                q as *const std::ffi::c_void,
                kc as *const std::ffi::c_void,
                vc as *const std::ffi::c_void,
                out_ptr as *mut std::ffi::c_void,
                q_cu as *mut i32,
                indices as *mut i32,
                indptr as *mut i32,
                last_len as *mut i32,
                plan15.as_ptr(),
                int_ws.ptr as *mut std::ffi::c_void,
                int_ws.bytes,
                float_ws.ptr as *mut std::ffi::c_void,
                float_ws.bytes,
                1, // batch
                call.hq as i32,
                call.hkv as i32,
                call.hd as i32,
                call.page as i32,
                call.total_rows as i32,
                call.sm_scale,
                mask_ptr as *const std::ffi::c_void,
                stream.cu_stream() as *mut std::ffi::c_void,
            )
        };
        if rc != 0 {
            return Err(OpError::Launch {
                op: msg.kernel.name.clone(),
                stage: crate::contract::Stage::Compute,
                detail: format!("run err {rc}"),
            });
        }
        Ok(Bytes::new(call.out.id, msg.out_elems, Dtype::F16, 2))
    }
}

#[cfg(test)]
mod plan_key_tests {
    use super::plan_key;

    /// **P-1 判据**:split 关 ⇒ ctx 不进键(换 ctx 复用同一份 plan);
    /// split 开 ⇒ ctx 进键(不得跨 ctx 复用)。图档位单列(数组/padded 都不同)。
    #[test]
    fn ctx_enters_key_only_when_split_on() {
        let k = |rows: usize, ctx: usize, graph: bool, split: bool| {
            plan_key(rows, ctx, rows, 24, 4, 256, 32, graph, split)
        };
        let a = k(512, 512, true, false);
        let b = k(512, 2048, true, false);
        assert_eq!(a, b, "split 关:ctx 必须被归一化掉(F-5)");
        assert_eq!(a.ctx_total, 0);

        let c = k(512, 512, true, true);
        let d = k(512, 2048, true, true);
        assert_ne!(c, d, "split 开:ctx 必须分键");

        // 图档位:同形状不同档 ⇒ 不同键(数组与 padded_batch_size 都不同)
        assert_ne!(k(512, 0, false, false), k(512, 0, true, false));
        // 形状位:行数/页 仍分键
        assert_ne!(k(512, 0, true, false), k(511, 0, true, false));
        assert_ne!(
            plan_key(512, 0, 512, 24, 4, 256, 32, true, false),
            plan_key(512, 0, 512, 24, 4, 256, 64, true, false),
            "page 必须分键"
        );
    }
}

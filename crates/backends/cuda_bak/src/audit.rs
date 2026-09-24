//! 哨兵③:driver 自审——捕获图的节点参数对账账本区间(roadmap §四·五)。
//!
//! 与哨兵①(捕获期自动登记)互补:①抓"我们经封装发射的"依赖,③抓
//! **绕过封装的逃逸**——第三方 FFI 私货、参数布局意外。原理:
//!
//! 1. `cuGraphGetNodes` 枚举图节点(含 child graph 递归,cublas 子图);
//! 2. 每个 kernel 节点经 `cuKernelGetParamInfo` 枚举参数布局;
//! 3. 每个 8 字节参数值:命中已登记池区间 → 必须是租约成员(否则
//!    依赖泄漏 = LawViolation);不命中且像设备地址 → 逃逸嫌疑(警告)。
//!
//! 冷路径(每图一次),热路径零参与。

use crate::governor::Governor;
use crate::ffi::sys;
use owl_iface::BackendError;
use std::collections::BTreeSet;
use std::sync::Arc;

/// 审计报告:driver 真相(图实际引用了什么)
#[derive(Debug, Default, Clone)]
pub struct AuditReport {
    /// 全部节点(含非 kernel 族:空节点/事件等)
    pub total_nodes: usize,
    /// kernel 族节点(含 child graph 内)
    pub kernel_nodes: usize,
    /// 无法解析参数的 kernel 节点(kern 句柄为空;M1 不静默放行,单独计数)
    pub opaque_nodes: usize,
    /// 8 字节参数命中已登记池区间(设备指针引用,去重)
    pub ptr_refs: BTreeSet<u64>,
    /// 命中池区间但 ∉ 租约集 = 依赖泄漏(违约证据)
    pub leaked: BTreeSet<u64>,
    /// 像设备地址但未被任何池区间登记 = 逃逸嫌疑(警告,非违约:
    /// 大标量/对齐填充也可能落在该区间)
    pub suspicious: BTreeSet<u64>,
    /// 显式豁免集(P0-2:哨兵③ 拒绝建图的白名单;暂空,按需填)。
    /// 语义:命中 suspicious 但在 allowlist 中的指针不拦截。
    pub allowlist: BTreeSet<u64>,
}

impl AuditReport {
    pub fn is_clean(&self) -> bool {
        self.leaked.is_empty()
    }
}

/// 低于此值的 8 字节参数按标量处理(设备 VA 不会这么低;16MiB 门槛)
const DEVICE_VA_FLOOR: u64 = 16 * 1024 * 1024;
/// 用户态设备 VA 上限(x86-64 48 位)
const DEVICE_VA_CEIL: u64 = 1 << 48;

fn check(st: sys::CUresult, what: &'static str) -> Result<(), BackendError> {
    if st != sys::CUresult::CUDA_SUCCESS {
        return Err(BackendError::Init(format!("哨兵③ {what}: {st:?}")));
    }
    Ok(())
}

/// 审计一张已捕获的 CUgraph。`lease_ids` = 租约缓冲的 token id 集合
/// (区间的 buf_id 命中它才算合法引用)。
pub(crate) fn audit_graph(
    cu_graph: sys::CUgraph,
    gov: &Arc<Governor>,
    lease_ids: &BTreeSet<u64>,
) -> Result<AuditReport, BackendError> {
    let mut rep = AuditReport::default();
    let mut intervals = gov.intervals.lock();
    unsafe {
        enumerate_nodes(cu_graph, &mut rep, &mut intervals, lease_ids)?;
    }
    drop(intervals);
    if !rep.leaked.is_empty() {
        eprintln!(
            "[owl-audit] 哨兵③ 违约细节:泄漏地址 = {:?}, 全部指针引用 = {:?}, 嫌疑 = {:?}",
            rep.leaked, rep.ptr_refs, rep.suspicious
        );
        return Err(BackendError::LawViolation(
            "哨兵③:图内地址命中账本区间但 ∉ 租约集(依赖泄漏;详见日志)",
        ));
    }
    Ok(rep)
}

unsafe fn enumerate_nodes(
    cu_graph: sys::CUgraph,
    rep: &mut AuditReport,
    intervals: &std::collections::BTreeMap<u64, (u64, u64)>,
    lease_ids: &BTreeSet<u64>,
) -> Result<(), BackendError> {
    let mut count: usize = 0;
    check(
        sys::cuGraphGetNodes(cu_graph, std::ptr::null_mut(), &mut count),
        "cuGraphGetNodes(枚举)",
    )?;
    let mut nodes: Vec<sys::CUgraphNode> = vec![std::ptr::null_mut(); count];
    check(
        sys::cuGraphGetNodes(cu_graph, nodes.as_mut_ptr(), &mut count),
        "cuGraphGetNodes(取值)",
    )?;

    for node in nodes.iter() {
        rep.total_nodes += 1;
        // repr(u32) C 枚举:零值 = KERNEL,driver 立即覆写
        let mut ntype: sys::CUgraphNodeType = std::mem::zeroed();
        check(sys::cuGraphNodeGetType(*node, &mut ntype), "cuGraphNodeGetType")?;
        match ntype {
            // child graph(cublas split-K 等自研 AR 同族):递归
            sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_GRAPH => {
                let mut params: sys::CUDA_KERNEL_NODE_PARAMS = std::mem::zeroed();
                check(
                    sys::cuGraphKernelNodeGetParams_v2(*node, &mut params),
                    "cuGraphKernelNodeGetParams_v2(child)",
                )?;
                // child graph 节点的 func 槽位按驱动文档承载子图句柄
                enumerate_nodes(params.func as sys::CUgraph, rep, intervals, lease_ids)?;
            }
            sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL => {
                rep.kernel_nodes += 1;
                audit_kernel_node(*node, rep, intervals, lease_ids)?;
            }
            // memcpy/memset/empty/event 族:一期不解析参数(无指针参数语义),
            // 计数即留痕;出现非预期的 memcpy 节点 = 有人塞了拷贝私货
            _ => {}
        }
    }
    Ok(())
}

unsafe fn audit_kernel_node(
    node: sys::CUgraphNode,
    rep: &mut AuditReport,
    intervals: &std::collections::BTreeMap<u64, (u64, u64)>,
    lease_ids: &BTreeSet<u64>,
) -> Result<(), BackendError> {
    let mut params: sys::CUDA_KERNEL_NODE_PARAMS = std::mem::zeroed();
    check(
        sys::cuGraphKernelNodeGetParams_v2(node, &mut params),
        "cuGraphKernelNodeGetParams_v2",
    )?;
    // kern 句柄:驱动对 nvrtc 路径可能只给 func,经 cuFuncGetKernel 回退
    let mut kern = params.kern;
    if kern.is_null() {
        let r = unsafe { func_to_kernel(params.func, &mut kern) };
        if !r {
            rep.opaque_nodes += 1;
            return Ok(());
        }
    }

    // pass 1:参数个数
    let mut n_params: usize = 0;
    loop {
        let mut off: usize = 0;
        let mut size: usize = 0;
        let r = sys::cuKernelGetParamInfo(kern, n_params, &mut off, &mut size);
        if r != sys::CUresult::CUDA_SUCCESS {
            break;
        }
        n_params += 1;
    }
    // pass 2:参数值对账。参数可能是标量(size≤8)或结构体(size>8,
    // 如 cublas 内核的 176B 参数块):统一按 8 字节对齐扫描取值区,
    // 落进已登记区间的值 = 设备指针引用。
    for i in 0..n_params {
        let mut off: usize = 0;
        let mut size: usize = 0;
        check(
            sys::cuKernelGetParamInfo(kern, i, &mut off, &mut size),
            "cuKernelGetParamInfo",
        )?;
        // kernelParams[i] 指向参数 i 的取值区(裸指针数组,手工偏移)
        let slot = params.kernelParams.add(i);
        if slot.is_null() || size == 0 {
            rep.opaque_nodes += 1;
            continue;
        }
        let base_ptr = slot as *const u8;
        for w in 0..(size / 8) {
            let v = *(base_ptr.add(w * 8) as *const u64);
            // 区间反查:最后一个 base ≤ v 的区间是否覆盖 v(池区间不重叠,
            // 故只需检查最近的 base)
            let hit = intervals
                .range(..=v)
                .next_back()
                .filter(|(base, &(len, _))| v - *base < len)
                .map(|(_, &(_, id))| id);
            match hit {
                Some(id) => {
                    rep.ptr_refs.insert(v);
                    if !lease_ids.contains(&id) {
                        rep.leaked.insert(v);
                    }
                }
                None => {
                    if v >= DEVICE_VA_FLOOR && v < DEVICE_VA_CEIL {
                        rep.suspicious.insert(v);
                    }
                }
            }
        }
    }
    Ok(())
}

/// `cuFuncGetKernel` 原始 FFI(cudarc 0.19 未绑定;与 device.rs P2P 窄口
/// 同一模式:culib 动态加载)。只读查询,零分配。
unsafe fn func_to_kernel(func: sys::CUfunction, out: &mut sys::CUkernel) -> bool {
    use cudarc::driver::sys as raw_sys;
    type F = unsafe extern "C" fn(sys::CUfunction, *mut sys::CUkernel) -> sys::CUresult;
    let f: F = match raw_sys::culib().get(b"cuFuncGetKernel\0") {
        Ok(sym) => *sym,
        Err(_) => return false,
    };
    f(func, out) == sys::CUresult::CUDA_SUCCESS
}

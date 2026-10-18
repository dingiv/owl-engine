//! **VMM 显式授予原语**(TP2 S1)—— peer 内存的底层机制面。
//!
//! 设计出处:`docs/arch/TP2-施工设计.md` §2.2(分配/授予序列)/ §2.3(账本)。
//! 白名单出处:`crate::ffi::sys::vmm`(A4 准入登记在 `ffi.rs`)。
//!
//! # 为什么是 VMM 而不是 `cudaDeviceEnablePeerAccess`(A2.6 修正,硬纪律)
//!
//! 3080 档(BAR1 256 MiB)上 `enable` 触发**进程级急切窗口预建** ⇒ 撞
//! BAR1 预算墙;唯一合法姿势 = 显式授予(`reserve → create → map →
//! setAccess → export/import`)。出处:
//! `docs/llama.cpp/vmm-p2p-3080会战-测试报告.md` 坑 3/7/8、
//! `docs/llama.cpp/v3基线不通-立案排查.md` §10.5。
//! **本文件里没有、也不许出现任何 enable/peer-access 启用调用。**
//!
//! # 纪律
//!
//! 1. **失败即回滚**:构建路径全程走 [`Guard`](RAII)——任一步失败,已完成的
//!    步骤按逆序撤销(unmap → release → addressFree),半成品不留驱动里;
//! 2. **`setAccess` 必须显式调 even for self**(CUDA VMM 语义;坑 9:漏了
//!    它是 `cuMemsetD8 invalid` 的经典现场)—— 本模块把 map 与 setAccess
//!    **拆开**,由调用方显式传权(不隐式代劳);
//! 3. **释放是显式口**(D2 裁决):[`release`] 是唯一正规释放;`Drop` 兜底
//!    走同一序列,错误写 stderr(**不静默**,但 Drop 无法向上传递)。
//!
//! # 13.x 类型形态(唯一构造点)
//!
//! `CUmemLocation` 在 CUDA 13.x 的 bindgen 形态是
//! `{ type_, __bindgen_anon_1: { id } }`(12.x 是直接 `{ type_, id }`)。
//! 本仓构建 = CUDA 13.2(`cudarc` 的 `cuda-13020` feature)⇒ 用 13.x 形态;
//! 升/降 CUDA 大版本时改 [`dev_location`] 一处 + `state.rs` 的粒度查询口径。

use crate::ffi::sys::vmm as ffi;
use crate::ffi::sys::CUdeviceptr;
use owl_iface::contract::ModelError;

/// 粒度对齐(向上取整;`align` 必须是 2 的幂 —— VMM 粒度都是)。
///
/// 溢出 = 响亮拒(不许静默钳位;`bytes` 来自远端凭据,不可信)。
fn round_up(n: u64, align: u64) -> Result<u64, ModelError> {
    debug_assert!(align.is_power_of_two());
    let pad = match n.checked_add(align - 1) {
        Some(v) => v,
        None => {
            return Err(ModelError::Msg(format!(
                "VMM 对齐溢出:bytes={n} align={align}(拒绝,不许钳位)"
            )))
        }
    };
    Ok(pad & !(align - 1))
}

/// `CUresult` 检查(统一错误形状:带操作名 + rc;诊断可定位)。
fn check(rc: ffi::CUresult, what: &str) -> Result<(), ModelError> {
    if rc == ffi::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(ModelError::Msg(format!("VMM {what} 失败: {rc:?}")))
    }
}

/// 设备属性(PINNED + POSIX fd 句柄;peer 授予的固定形态)。
pub(super) fn dev_prop(ordinal: usize) -> ffi::CUmemAllocationProp {
    // zeroed 合法:残余字段全部语义为 0(compressionType 无压缩、
    // win32 元数据空指针在 POSIX 下无关),不是"未初始化读取"。
    let mut prop: ffi::CUmemAllocationProp = unsafe { std::mem::zeroed() };
    prop.type_ = ffi::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location = dev_location(ordinal);
    prop.requestedHandleTypes =
        ffi::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    prop
}

/// 设备位置(CUDA 13.x 形态;**唯一构造点** —— 见模块头)。
fn dev_location(ordinal: usize) -> ffi::CUmemLocation {
    let mut loc: ffi::CUmemLocation = unsafe { std::mem::zeroed() };
    loc.type_ = ffi::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
    loc.__bindgen_anon_1 = ffi::CUmemLocation_st__bindgen_ty_1 { id: ordinal as i32 };
    loc
}

/// 最小分配粒度(设备相关;GA102/PCIe 实测 2 MiB)。
pub(super) fn min_granularity(prop: &ffi::CUmemAllocationProp) -> Result<u64, ModelError> {
    let mut g: usize = 0;
    check(
        unsafe {
            ffi::cuMemGetAllocationGranularity(
                &mut g,
                prop,
                ffi::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_MINIMUM,
            )
        },
        "cuMemGetAllocationGranularity",
    )?;
    if g == 0 {
        // 驱动给了 0 = 异常;0 会让后续除法/对齐全错 —— 响亮拒
        return Err(ModelError::Msg("VMM 粒度 = 0(驱动异常;拒绝继续)".into()));
    }
    Ok(g as u64)
}

/// 一段已映射的 VMM 区域(所有权证;释放走 [`release`])。
pub(super) struct VmmMapping {
    /// 预留的 VA 基址
    pub(super) va: u64,
    /// 映射总字节(**粒度对齐后**;unmap/addressFree 用它,不是请求值)
    pub(super) bytes: usize,
    /// 分配句柄(本端 create 或对端 import 得来;释放时 release)
    pub(super) handle: ffi::CUmemGenericAllocationHandle,
}

/// 构建守卫(RAII):失败路径自动按逆序回滚。
///
/// 成功路径必须 [`disarm`](Guard::disarm) 交出所有权 —— 漏了 disarm 会在
/// Drop 时把刚建好的映射拆掉(编译器不会拦,门会拦:块账里没有它)。
struct Guard {
    va: Option<u64>,
    bytes: usize,
    handle: Option<ffi::CUmemGenericAllocationHandle>,
    mapped: bool,
}

impl Guard {
    fn new(bytes: usize) -> Self {
        Self { va: None, bytes, handle: None, mapped: false }
    }

    fn reserve(&mut self, gran: u64) -> Result<(), ModelError> {
        let mut va: CUdeviceptr = 0;
        check(
            unsafe {
                ffi::cuMemAddressReserve(
                    &mut va,
                    self.bytes,
                    gran as usize,
                    0, // addr = 0:交给驱动挑(高位 VA 路由缺陷已由驱动 fork 处理)
                    0, // flags 保留位
                )
            },
            "cuMemAddressReserve",
        )?;
        self.va = Some(va);
        Ok(())
    }

    fn create(&mut self, prop: &ffi::CUmemAllocationProp) -> Result<(), ModelError> {
        let mut h: ffi::CUmemGenericAllocationHandle = 0;
        check(
            unsafe { ffi::cuMemCreate(&mut h, self.bytes, prop, 0) },
            "cuMemCreate",
        )?;
        self.handle = Some(h);
        Ok(())
    }

    fn import(&mut self, fd: i32) -> Result<(), ModelError> {
        let mut h: ffi::CUmemGenericAllocationHandle = 0;
        check(
            unsafe {
                ffi::cuMemImportFromShareableHandle(
                    &mut h,
                    // osHandle 是 `*mut c_void`;POSIX 语义 = fd 整数塞指针宽
                    // (官方示例同款:reinterpret 而非解引用)
                    fd as *mut std::ffi::c_void,
                    ffi::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                )
            },
            "cuMemImportFromShareableHandle",
        )?;
        self.handle = Some(h);
        Ok(())
    }

    fn map(&mut self) -> Result<(), ModelError> {
        let (va, handle) = match (self.va, self.handle) {
            (Some(v), Some(h)) => (v, h),
            _ => {
                return Err(ModelError::Msg(
                    "VMM map 前置缺失(reserve/handle 未完成;编程错误)".into(),
                ))
            }
        };
        check(unsafe { ffi::cuMemMap(va, self.bytes, 0, handle, 0) }, "cuMemMap")?;
        self.mapped = true;
        Ok(())
    }

    /// 交出所有权(不再回滚)。
    fn disarm(mut self) -> VmmMapping {
        let va = self.va.take().expect("VMM disarm:va 缺失(编程错误)");
        let handle = self.handle.take().expect("VMM disarm:handle 缺失(编程错误)");
        let m = VmmMapping { va, bytes: self.bytes, handle };
        // 显式说明:所有权已交出,Drop 不得再动
        std::mem::forget(self);
        m
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        // 这是**失败路径**(成功路径一律 disarm):只做逆序撤销。
        // 撤销失败无法向上传递(无返回值口)⇒ 写 stderr 保底可见
        // (与 release 的 Err 面同一序列;此处失败不覆盖主错误)。
        if self.mapped {
            if let Some(va) = self.va {
                if let Err(e) = check(unsafe { ffi::cuMemUnmap(va, self.bytes) }, "cuMemUnmap(回滚)") {
                    eprintln!("[vmm] 回滚失败(泄漏风险,记录在案): {e}");
                }
            }
        }
        if let Some(h) = self.handle {
            if let Err(e) = check(unsafe { ffi::cuMemRelease(h) }, "cuMemRelease(回滚)") {
                eprintln!("[vmm] 回滚失败(泄漏风险,记录在案): {e}");
            }
        }
        if let Some(va) = self.va {
            if let Err(e) = check(unsafe { ffi::cuMemAddressFree(va, self.bytes) }, "cuMemAddressFree(回滚)")
            {
                eprintln!("[vmm] 回滚失败(泄漏风险,记录在案): {e}");
            }
        }
    }
}

/// 本端分配:`reserve → create → map`(不含 setAccess/export —— 调用方显式做)。
///
/// 返回 `(映射, 粒度)`。`bytes` = 请求字节数(函数内部向上取整到粒度)。
pub(super) fn alloc_mapped(
    ordinal: usize,
    bytes: u64,
) -> Result<(VmmMapping, u64), ModelError> {
    if bytes == 0 {
        return Err(ModelError::Msg("VMM alloc(bytes=0;拒绝)".into()));
    }
    let prop = dev_prop(ordinal);
    let gran = min_granularity(&prop)?;
    let aligned = round_up(bytes, gran)?;
    if aligned > usize::MAX as u64 {
        return Err(ModelError::Msg(format!(
            "VMM alloc(bytes={bytes} → 对齐 {aligned})超出 usize(拒绝)"
        )));
    }
    let mut g = Guard::new(aligned as usize);
    g.reserve(gran)?;
    g.create(&prop)?;
    g.map()?;
    Ok((g.disarm(), gran))
}

/// 对端导入:`import → reserve → map`(size 必须 = 分配侧 create 的大小)。
///
/// `mapped_bytes` 来自对端凭据(已对齐);`gran` 由本端查询(对齐校验用)。
pub(super) fn import_mapped(
    ordinal: usize,
    fd: i32,
    mapped_bytes: u64,
) -> Result<(VmmMapping, u64), ModelError> {
    if mapped_bytes == 0 {
        return Err(ModelError::Msg("VMM import(bytes=0;拒绝)".into()));
    }
    let prop = dev_prop(ordinal);
    let gran = min_granularity(&prop)?;
    if mapped_bytes % gran != 0 {
        return Err(ModelError::Msg(format!(
            "VMM import:对端映射大小 {mapped_bytes} 不是本端粒度 {gran} 的倍数(拒绝)"
        )));
    }
    if mapped_bytes > usize::MAX as u64 {
        return Err(ModelError::Msg("VMM import:size 超出 usize(拒绝)".into()));
    }
    let mut g = Guard::new(mapped_bytes as usize);
    g.import(fd)?;
    g.reserve(gran)?;
    g.map()?;
    Ok((g.disarm(), gran))
}

/// 授予访问权(**map 之后必须显式调** —— CUDA VMM 语义,坑 9)。
///
/// 只授本端设备(`self`)的权限:`read_write=true` → RW(分配侧),`false` → R
/// (导入侧/别名)。**不授 peer** —— 对端读的是它自己的 VA,不经本端 VA。
pub(super) fn set_access_self(
    m: &VmmMapping,
    ordinal: usize,
    read_write: bool,
) -> Result<(), ModelError> {
    let desc = ffi::CUmemAccessDesc {
        location: dev_location(ordinal),
        flags: if read_write {
            ffi::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        } else {
            ffi::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READ
        },
    };
    check(
        unsafe { ffi::cuMemSetAccess(m.va, m.bytes, &desc, 1) },
        "cuMemSetAccess(self)",
    )
}

/// 导出可共享句柄(POSIX fd)。
///
/// ⚠️ **所有权**:返回的新 fd 归**分配侧**;导入侧(同进程)拿到的是同一张
/// fd 表项,**绝不可 close**(否则分配侧 release 时双 close 事故)。
/// 跨进程形态才需要 SCM_RIGHTS + 各自副本 —— 本仓 TP2 = 单进程双 actor,
/// 不进那条路。
pub(super) fn export_fd(m: &VmmMapping) -> Result<i32, ModelError> {
    let mut out: i32 = -1;
    check(
        unsafe {
            ffi::cuMemExportToShareableHandle(
                &mut out as *mut i32 as *mut std::ffi::c_void,
                m.handle,
                ffi::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                0,
            )
        },
        "cuMemExportToShareableHandle",
    )?;
    if out < 0 {
        return Err(ModelError::Msg(format!(
            "VMM 导出的 fd 非法({out};驱动异常 —— 拒绝把它传给对端)"
        )));
    }
    Ok(out)
}

/// 关闭 fd(接管所有权;**只允许分配侧** —— 见 [`export_fd`] 所有权注)。
///
/// 用 `File::from_raw_fd` 接管 ⇒ drop 即 `close(2)`,零 libc 依赖。
pub(super) fn close_fd(fd: i32) {
    use std::os::fd::FromRawFd;
    unsafe { drop(std::fs::File::from_raw_fd(fd)) };
}

/// 显式释放:`unmap → release → addressFree`(三步全走,错误聚合上报)。
///
/// 三步各自都可能失败;**不短路**(一个失败不该把后两步也漏掉 —— 漏了
/// 泄漏更多)。返回聚合错误(全成功才 `Ok`)。
pub(super) fn release(m: VmmMapping) -> Result<(), ModelError> {
    let mut errs: Vec<String> = Vec::new();
    if let Err(e) = check(unsafe { ffi::cuMemUnmap(m.va, m.bytes) }, "cuMemUnmap") {
        errs.push(e.to_string());
    }
    if let Err(e) = check(unsafe { ffi::cuMemRelease(m.handle) }, "cuMemRelease") {
        errs.push(e.to_string());
    }
    if let Err(e) = check(unsafe { ffi::cuMemAddressFree(m.va, m.bytes) }, "cuMemAddressFree") {
        errs.push(e.to_string());
    }
    if errs.is_empty() {
        Ok(())
    } else {
        Err(ModelError::Msg(format!(
            "VMM 释放存在失败步骤(泄漏风险;逐条):{}",
            errs.join(" ‖ ")
        )))
    }
}

/// 主错误 + 回滚(释放失败聚合进错误串;主错误在前)。
///
/// 用途:分配序列的中间步骤(如 `setAccess`/`export`)失败时,把已建好的
/// 映射收回去 —— 两个错误都不丢(AGENTS:不许吞错)。
pub(super) fn fail_with_rollback(m: VmmMapping, primary: ModelError) -> ModelError {
    match release(m) {
        Ok(()) => primary,
        Err(e) => ModelError::Msg(format!("回滚也失败 → 泄漏风险:{primary} ‖ {e}")),
    }
}

#[cfg(test)]
mod tests {
    use super::round_up;

    #[test]
    fn round_up_exact_and_partial() {
        assert_eq!(round_up(0, 1024).unwrap(), 0);
        assert_eq!(round_up(1, 1024).unwrap(), 1024);
        assert_eq!(round_up(1024, 1024).unwrap(), 1024, "恰好对齐不回退");
        assert_eq!(round_up(1025, 1024).unwrap(), 2048);
        // 64MiB + 64B(header)→ 仍是 64MiB 的倍数情形(2MiB 粒度下不变)
        assert_eq!(round_up((64 << 20) + 64, 2 << 20).unwrap(), 66 << 20);
    }

    #[test]
    fn round_up_overflow_is_loud() {
        // u64::MAX 附近的溢出必须响亮拒,不许钳位
        assert!(round_up(u64::MAX, 1 << 20).is_err());
        assert!(round_up(u64::MAX - 3, 4096).is_err());
    }
}

//! 装配期自检执行面(设计 §5.5):把 `DeviceRes × Exec` 桥到零门的
//! [`CheckRunner`],并在 boot 期逐算子跑 oracle。
//!
//! 为什么需要适配器:oracle 住 `ops/`(零 feature 门,与算子声明同居),
//! 而设备面是 `feature = "device"` 门控的 —— 适配器是两者唯一的接触点。

use crate::contract::{OpError, OpId, LaunchVal, Stage};
use crate::device::{DeviceRes, Exec};
use crate::bootcheck::{CheckBuf, CheckRunner, CheckReport};
use crate::spec::{KernelMeta};

/// 自检运行时(设备侧实现)
///
/// `registry`:**必须**在注册表建成之后注入(foreign 算子自检要经它发射 ——
/// 所以自检调用点从 `init_all` 内部搬到了 `boot_with` 的构造后,见 registry)。
pub struct Runner<'a> {
    pub res: &'a mut dyn DeviceRes,
    pub exec: &'a mut Exec,
    pub registry: Option<&'a mut crate::registry::OpRegistry>,
}

impl CheckRunner for Runner<'_> {
    fn alloc(&mut self, bytes: usize, tag: &'static str) -> Result<CheckBuf, OpError> {
        let b = self.res.alloc(bytes, tag)?;
        Ok(CheckBuf { ptr: b.ptr, bytes: b.bytes })
    }

    fn alloc_block(&mut self, bytes: usize, tag: &'static str) -> Result<crate::contract::Bytes, OpError> {
        self.res.alloc_block(bytes, tag)
    }

    fn upload_block(&mut self, b: &crate::contract::Bytes, src: &[u8]) -> Result<(), OpError> {
        self.res.upload_block(b, src)
    }

    fn download_block(&mut self, b: &crate::contract::Bytes, dst: &mut [u8]) -> Result<(), OpError> {
        let p = self.res.resolve(b.id)?;
        self.res.download(p, dst)
    }

    fn dispatch(
        &mut self,
        msg: crate::contract::LaunchMsg,
    ) -> Result<crate::contract::Bytes, OpError> {
        let Some(reg) = self.registry.as_mut() else {
            return Err(OpError::Contract {
                op: msg.kernel.name.clone(),
                field: "dispatch",
                expect: "注册表在场(自检运行面在注册表建成之后注入)".into(),
                got: "无注册表 runner(仅 launch_aot 路径)".into(),
            });
        };
        reg.execute(&msg, self.res, self.exec)
    }

    fn upload(&mut self, dst: CheckBuf, src: &[u8]) -> Result<(), OpError> {
        self.res.upload(dst.ptr, src)
    }

    fn download(&mut self, src: CheckBuf, dst: &mut [u8]) -> Result<(), OpError> {
        self.res.download(src.ptr, dst)
    }

    fn launch_aot(
        &mut self,
        name: &'static str,
        source: &'static str,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem: u32,
        vals: &[LaunchVal],
    ) -> Result<(), OpError> {
        let f = self.exec.load_aot(self.res, source, name, OpId(name))?;
        self.exec.launch(self.res, OpId(name), &f, name, grid, block, shared_mem, vals)
    }

    fn sync(&mut self) -> Result<(), OpError> {
        let stream = self.res.stream()?;
        stream
            .synchronize()
            .map_err(|e| OpError::Launch { op: "selfcheck".into(), stage: Stage::Compute, detail: format!("sync: {e:?}") })
    }
}

/// **装配期自检**(逐算子 oracle;任一失败 = Err ⇒ 拒启)。
///
/// 返回通过的项数;每项打一行报告(`max|err|` + 毒值律结论)。
pub fn run_selfchecks(
    metas: impl Iterator<Item = &'static KernelMeta>,
    res: &mut dyn DeviceRes,
    exec: &mut Exec,
    registry: &mut crate::registry::OpRegistry,
) -> Result<usize, OpError> {
    let mut n = 0usize;
    for m in metas {
        let Some(c) = m.selfcheck else { continue };
        // 先打名再跑:失败/挂起时能立刻定位是哪一个(2026-10-13 实测教训 ——
        // 生产 boot 里自检失败时,日志只有"族被 drop"的兜底告警)。
        // 2026-10-21:走统一输出口(INFO 级;默认级别 warn 不出,排障用
        // `--log-level info` 或 tag 白名单 `kernels`)。
        crate::klog_info!("selfcheck", "{} …({})", m.entry, c.name());
        let mut r = Runner { res, exec, registry: Some(&mut *registry) };
        let rep: CheckReport = c.run(&mut r as &mut dyn CheckRunner).map_err(|e| {
            OpError::Asset {
                op: m.entry.to_string(),
                detail: format!("装配期自检 `{}` 失败:{e}", c.name()),
            }
        })?;
        let poison = match rep.poison_untouched {
            Some(k) => format!("毒值律 ✓(未写字节 {k})"),
            None => "毒值律 —(非池算子)".to_string(),
        };
        crate::klog_info!(
            "selfcheck",
            "{} ✓ {} max|err|={:.3e} ≤ {:.1e} {poison}",
            m.entry, c.name(), rep.worst_abs_err, rep.tolerance
        );
        n += 1;
    }
    Ok(n)
}

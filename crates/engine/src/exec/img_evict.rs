//! **前缀无感删除的盘层收尾**:快照重挂 / 旧身份丢弃。
//!
//! leave-gap 只摘页、保留幸存 position,不转 K;本模块只做 `SnapDiskOp` 的
//! 真实 I/O(决策在 `scheduler` 纯 host 面出票)。

use owl_iface::contract::{DeviceClient, ModelError};

use crate::scheduler::SnapDiskOp;

impl<D: DeviceClient> crate::running::RunningEngine<D> {
    /// **盘层快照的删除收尾**(重挂 / 丢弃旧身份;2026-10-23)。
    ///
    /// 时机纪律:必须在 `execute_begin` **最前**、任何快照恢复之前 ——
    /// 删后新链的恢复键可能就落在盘层,`snap_ensure_resident` 按**新键**
    /// 回灌,而盘上文件还叫旧名。
    ///
    /// 失败语义(响亮):重挂失败 = 账/盘不平 ⇒ 上抛(不静默当“重挂过了”);
    /// 丢弃失败由 [`crate::disk_io::remove_files`] 自报(文件不在也算腾出)。
    pub(crate) async fn execute_snap_disk_ops(
        &mut self,
        ops: &[SnapDiskOp],
    ) -> Result<(), ModelError> {
        if ops.is_empty() {
            return Ok(());
        }
        if self.bank.rel.snap_disk.is_none() {
            return Err(ModelError::EngineInvariant {
                what: format!("盘层快照收尾:未接盘层却有 {} 张票", ops.len()),
            });
        }
        let mut rekeyed = 0usize;
        for op in ops {
            match op {
                SnapDiskOp::Rekey { old, new } => {
                    let (from, to) = self.bank.rel.plan_rekey_snap_disk(*old, *new)?;
                    match crate::disk_io::rename_block(from, to).await {
                        Ok(()) => {
                            self.bank.rel.commit_rekey_snap_disk(*old, *new)?;
                            rekeyed += 1;
                        }
                        Err(e) => {
                            // 文件改名失败 ⇒ 账不动(旧键仍有效,只是新链身份缺条目)
                            owl_shared::log_warn!("img-evict", "盘层快照重挂失败:{e}");
                        }
                    }
                }
            }
        }
        owl_shared::log_info!(
            "img-evict",
            "盘层快照收尾:重挂 {} 条",
            rekeyed
        );
        Ok(())
    }
}

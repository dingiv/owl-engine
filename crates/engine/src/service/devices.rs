//! **设备束解析（引擎 init 的一部分）** —— 2026-11-15 P1 自 `apps/server` 下沉。
//!
//! 为什么归引擎：选卡 = **设备所有权**（谁开 CUDA context、谁持卡锁），
//! 不是 HTTP 服务面的策略。`apps/server/src/config.rs` 里的同名实现只服务
//! 一个入口；测试/CLI/将来的 headless 入口都得复制一遍。
//!
//! 语义（原样搬运，零行为变化）：
//! - `mode = single` ⇒ 一条 `device`（空 = 自动挑一张空闲卡）；
//! - `mode = tensor` ⇒ 两条（空 = 自动挑两张）；
//! - 卡位只用 **BDF / UUID / 名字子串**（`DeviceSpec::parse` 拒 CUDA 序号）；
//! - 候选过滤：卡锁（owl 实例）→ `nvidia-smi` 实占内存占比阈值；
//! - 列表去重消费：`["3090","3090"]` = 两张不同的 3090。

use owl_shared::config::{ParallelCfg, ParallelMode};
use owl_shared::device::{parse_nvidia_smi_mem, DeviceRow, DeviceSpec};

/// 本进程持有的设备束（rank 序 = 列表序）。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Devices {
    /// 引擎 ordinal（交 `cudaSetDevice`）
    ordinals: Vec<usize>,
    /// 设备 UUID（卡锁 + 日志身份；与 `ordinals` 同长同序）
    uuids: Vec<String>,
}

impl Devices {
    /// 构造即校验（空/长度不齐 = 拒；字段私有 ⇒ 唯一入口）。
    pub fn new(ordinals: Vec<usize>, uuids: Vec<String>) -> Result<Self, String> {
        if ordinals.is_empty() {
            return Err("设备束为空(至少要一张卡)".into());
        }
        if ordinals.len() != uuids.len() {
            return Err(format!(
                "设备束长度不齐:ordinals={} uuids={}(内部不变量)",
                ordinals.len(),
                uuids.len()
            ));
        }
        Ok(Self { ordinals, uuids })
    }

    /// rank 数（单卡 = 1；TP v1 恒 2）
    pub fn world(&self) -> usize {
        self.ordinals.len()
    }
    /// 全部 rank 的 ordinal（rank 序 = 列表序）
    pub fn ordinals(&self) -> &[usize] {
        &self.ordinals
    }
    /// 全部 rank 的 UUID（与 [`Self::ordinals`] 同长同序）
    pub fn uuids(&self) -> &[String] {
        &self.uuids
    }
    /// rank0（单卡档即唯一那张）的引擎 ordinal
    pub fn device(&self) -> usize {
        self.ordinals[0]
    }
}

/// **选卡总入口**（唯一真源 = `[parallel]`）。
pub fn resolve_devices(par: &ParallelCfg, min_free_ratio: f64) -> Result<Devices, String> {
    let specs = par
        .device
        .iter()
        .map(|s| DeviceSpec::parse(s))
        .collect::<Result<Vec<_>, String>>()?;
    match par.mode {
        ParallelMode::Single => {
            let picked = pick_distinct(&specs, 1, min_free_ratio)?;
            eprintln!(
                "[boot] 选卡(mode=single, device={:?}):ordinal {} [{}|{}] {}",
                par.device, picked[0].ordinal, picked[0].bdf, picked[0].name, picked[0].uuid
            );
            Devices::new(vec![picked[0].ordinal], vec![picked[0].uuid.clone()])
        }
        ParallelMode::Tensor => {
            let picked = pick_distinct(&specs, 2, min_free_ratio)?;
            eprintln!(
                "[boot] 选卡(TP 档, device={:?}):rank0 = ordinal {} → {};rank1 = ordinal {} → {}(rank 序 = 列表序;logger 前 ⇒ 裸输出)",
                par.device, picked[0].ordinal, picked[0].uuid, picked[1].ordinal, picked[1].uuid
            );
            Devices::new(
                picked.iter().map(|r| r.ordinal).collect(),
                picked.iter().map(|r| r.uuid.clone()).collect(),
            )
        }
    }
}

/// 按序取 N 张互不相同的卡（去重消费；`specs` 空 = 自动挑空闲卡）。
fn pick_distinct(
    specs: &[DeviceSpec],
    n: usize,
    min_free_ratio: f64,
) -> Result<Vec<DeviceRow>, String> {
    let rows = owl_cuda::ffi::device_rows()?;
    let live = owl_shared::gpu_lock::live_locks();
    let mem = vram_free_ratio_map();
    let wildcard = DeviceSpec::default();
    let mut used: Vec<String> = Vec::new();
    let mut out: Vec<DeviceRow> = Vec::new();
    for i in 0..n {
        let spec = specs.get(i).unwrap_or(&wildcard);
        let hit: Vec<&DeviceRow> = rows
            .iter()
            .filter(|r| spec.matches(r) && !used.contains(&r.uuid))
            .collect();
        if hit.is_empty() {
            let table: Vec<String> = rows
                .iter()
                .map(|r| format!("{}[{}|{}|{}]", r.ordinal, r.bdf, r.name, r.uuid))
                .collect();
            return Err(format!(
                "[parallel] device 第 {i} 项 = {} 未命中任何**未占用**设备(设备表={table:?})",
                spec.describe()
            ));
        }
        // ① 卡锁（owl 实例）
        let free: Vec<&DeviceRow> = hit
            .iter()
            .copied()
            .filter(|r| !live.iter().any(|l| l.uuid == r.uuid))
            .collect();
        // ② 实占内存过滤（锁只认 owl；别的进程也要避让）
        let free = match &mem {
            Some(m) => {
                let mut keep = Vec::new();
                let mut skipped: Vec<String> = Vec::new();
                for r in free {
                    match m.get(&r.uuid) {
                        Some(ratio) if *ratio < min_free_ratio => {
                            skipped.push(format!("{} {} free={:.0}%", r.uuid, r.bdf, ratio * 100.0));
                        }
                        _ => keep.push(r),
                    }
                }
                if !skipped.is_empty() {
                    eprintln!(
                        "[boot] 选卡:按实占内存跳过 {} 台(阈值 {:.0}%):{skipped:?}",
                        skipped.len(),
                        min_free_ratio * 100.0
                    );
                }
                keep
            }
            None => {
                if i == 0 {
                    eprintln!(
                        "[boot] 选卡:nvidia-smi 内存查询不可用 ⇒ 跳过实占内存过滤(只按卡锁避让)"
                    );
                }
                free
            }
        };
        let picked = free.into_iter().next().ok_or_else(|| {
            let holders: Vec<String> = live
                .iter()
                .map(|l| format!("{} pid={} port={}", l.uuid, l.pid, l.port))
                .collect();
            format!(
                "[parallel] device 第 {i} 项 = {} 的候选全部被占(spec 命中 {} 台;持有者={holders:?})",
                spec.describe(),
                hit.len()
            )
        })?;
        used.push(picked.uuid.clone());
        out.push(picked.clone());
    }
    Ok(out)
}

/// `nvidia-smi` 的每卡空闲占比（UUID → free/total）。不可用 = `None`（只按卡锁避让）。
fn vram_free_ratio_map() -> Option<std::collections::HashMap<String, f64>> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=uuid,memory.free,memory.total", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let map = parse_nvidia_smi_mem(&String::from_utf8_lossy(&out.stdout));
    if map.is_empty() {
        None
    } else {
        Some(map)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 构造门：空 / 长度不齐 = 拒
    #[test]
    fn devices_constructor_rejects_bad_shape() {
        assert!(Devices::new(vec![], vec![]).is_err());
        assert!(Devices::new(vec![0], vec![]).is_err());
        assert!(Devices::new(vec![0], vec!["u".into()]).is_ok());
        let d = Devices::new(vec![3, 4], vec!["a".into(), "b".into()]).expect("两卡");
        assert_eq!(d.world(), 2);
        assert_eq!(d.device(), 3);
        assert_eq!(d.uuids(), ["a", "b"]);
    }
}

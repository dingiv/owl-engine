//! 设备 ⇄ 宿主搬运原语（2026-11-18 S1，自 `owl-store::moves` 迁入）。
//!
//! **store 只描述搬运**：`owl_store::BlockLeg` 的 `OutPlan` / `InPlan` /
//! `ReadPlan` 给出 `SwapPiece`、槽台与块身份；真正的 `dtoh_at` /
//! `upload_pinned_on` 在本模块执行。调用方拿到执行结果后再回 store
//! `commit_*` / `abort_*`（见 `docs/devlog/dev/engine-models-store-boundary-handoff-2026-11-18.md`）。
//!
//! 两条唯一的搬运口（全部是自由函数 —— `AGENTS.md` 的 I/O 顶层函数纪律）：
//!
//! ```text
//! download_span(face, dev, off, dst)   设备块 --dtoh--> 宿主/临时缓冲
//! upload_span(face, src, dev, off)     宿主缓冲 --pinned 暂存--> 设备块
//! offload_to_host(face, dev, len, host)  整块设备 → HostBlock
//! onload_from_host(face, host, dev, len) 整块 HostBlock → 设备
//! ```
//!
//! 形态律（沿用 store 版）：
//! - **宿主内容不 pin**：`dtoh` 直接读进 pageable 缓冲；回灌时经一次性
//!   pinned 暂存租约（`alloc_pinned` → 拷贝 → `upload_pinned_on`），租约所有权
//!   移入设备 server，流完成回调归还其池；
//! - **搬运不在捕获窗内**：调用点必须在段边界；本模块不做时序判断；
//! - **失败即响亮**：任一原语失败原样上抛（带块号/字节），调用方据此回滚整单。
//!
//! S1 迁移说明：生产搬运当前主要走 `face.swap`（批量、带槽台）；本文件是
//! 逐片/整块原语的唯一落点，先迁入 engine，待后续逐片路径接线。
#![cfg_attr(not(test), allow(dead_code))]

use owl_iface::contract::{Bytes, DeviceClient, ModelError, UpStream};
use owl_store::HostBlock;

type Result<T> = std::result::Result<T, ModelError>;

fn map_err(e: ModelError, what: &str, bytes: usize) -> ModelError {
    ModelError::Msg(format!("{what}({bytes} 字节):{e}"))
}

/// **设备 → 宿主（带偏移）** —— 状态存储“搬出去”的基本原语。
///
/// `dev_off_bytes` = 逻辑块在该设备缓冲内的页偏移（`槽 × 每页字节`）；
/// 长度 = `dst.len()`（显式入参，坑册 §42：句柄自述长度不是真源）。
/// 后端面走 [`DeviceClient::dtoh_at`]（缺实现 = 响亮拒，不静默少搬）。
pub(crate) async fn download_span<D: DeviceClient>(
    face: &mut D,
    dev: &Bytes,
    dev_off_bytes: usize,
    dst: &mut [u8],
) -> Result<()> {
    let n = dst.len();
    face.dtoh_at(dev, dev_off_bytes, dst)
        .await
        .map_err(|e| map_err(e, "搬出 dtoh_at 失败", n))?;
    Ok(())
}

/// **宿主 → 设备（带偏移）** —— pinned 一次性租约作 DMA 源，所有权移入 server。
pub(crate) async fn upload_span<D: DeviceClient>(
    face: &mut D,
    src: &[u8],
    dev: &Bytes,
    dev_off_bytes: usize,
) -> Result<()> {
    let n = src.len();
    let mut lease = face
        .alloc_pinned(n, "engine:io-moves-staging")
        .await
        .map_err(|e| map_err(e, "搬入 alloc_pinned 失败", n))?;
    lease.slice_bytes_mut()[..n].copy_from_slice(src);
    // **落 COMPUTE 流**：搬入的内容就是随后计算要读的状态 —— 同流 FIFO 即
    // 结构性有序（放 H2D 流 = 与消费核无跨流序；2026-10-17 与参数面/swap_in
    // 同律归位。推导见 common-dev-pitfall §51 同族）。
    face.upload_pinned_on(UpStream::Compute, lease, dev, dev_off_bytes, n)
        .await
        .map_err(|e| map_err(e, "搬入 upload_pinned 失败", n))?;
    Ok(())
}

/// **设备 → 宿主（整块）** —— [`download_span`] 在偏移 0 的糖。
///
/// 长度须逐字节相等（宿主块大小 = 逻辑块字节）—— 不符 = 响亮拒（不许部分搬）。
pub(crate) async fn offload_to_host<D: DeviceClient>(
    face: &mut D,
    dev: &Bytes,
    len_bytes: usize,
    host: &mut HostBlock,
) -> Result<()> {
    let n = len_bytes;
    if n != host.len() {
        return Err(ModelError::Msg(format!(
            "卸载长度不符:设备块 {n} 字节 ≠ 宿主块 {} 字节",
            host.len()
        )));
    }
    download_span(face, dev, 0, host.as_mut_slice()).await
}

/// **宿主 → 设备（整块）** —— [`upload_span`] 在偏移 0 的糖。
///
/// 暂存租约的所有权随 `upload_pinned_on` 移入 server，由流完成回调归还其池
/// （即“回执 = 提交成功”，真落地由调用方在段边界后的栅栏兜底）。
pub(crate) async fn onload_from_host<D: DeviceClient>(
    face: &mut D,
    host: &HostBlock,
    dev: &Bytes,
    len_bytes: usize,
) -> Result<()> {
    let n = len_bytes;
    if n != host.len() {
        return Err(ModelError::Msg(format!(
            "回灌长度不符:宿主块 {} 字节 ≠ 设备块 {n} 字节",
            host.len()
        )));
    }
    upload_span(face, host.as_slice(), dev, 0).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use owl_cpu::CpuFace;
    use owl_iface::contract::{DeviceClient, Dtype};
    use owl_store::host::{HostBlock, HostTier};
    use owl_store::id::ChainHash;

    /// 造一个设备块并写入可辨识图样（全程走契约面：alloc → pinned 租约 → upload_pinned）。
    /// ⚠️ CPU 面值块恒 f32 ⇒ `elems` 按 f32 记账（块字节 = elems × 4）。
    async fn device_block(face: &mut CpuFace, len_bytes: usize) -> (Bytes, usize) {
        let elems = len_bytes / 4;
        let b = face.alloc(Dtype::F32, elems, "t").await.expect("alloc");
        let n = len_bytes;
        let pat: Vec<u8> = (0..n).map(|i| (i % 251) as u8).collect();
        let mut lease = face.alloc_pinned(n, "t").await.expect("pinned 租约");
        lease.slice_bytes_mut()[..n].copy_from_slice(&pat);
        face.upload_pinned(lease, &b, 0, n)
            .await
            .expect("upload_pinned");
        face.sync().await.expect("sync");
        (b, n)
    }

    #[tokio::test]
    async fn round_trip_device_host_device_is_byte_exact() {
        let mut face = CpuFace::default();
        let (dev, n) = device_block(&mut face, 8 << 10).await; // 8KiB 块

        let mut tier = HostTier::new(n, 1 << 20).expect("宿主层");
        let mut buf = tier.acquire().expect("宿主缓冲");
        offload_to_host(&mut face, &dev, n, &mut buf)
            .await
            .expect("卸载");

        // 设备侧清零（模拟“已被驱逐/复用”）⇒ 回灌后必须逐字节复原
        face.memset_zero(&dev, n).await.expect("清零");
        let mut after_zero = vec![0u8; n];
        face.dtoh(&dev, &mut after_zero).await.expect("回读");
        assert!(after_zero.iter().all(|b| *b == 0), "清零未生效,门失效");

        onload_from_host(&mut face, &buf, &dev, n)
            .await
            .expect("回灌");
        // upload_pinned 回执 = 提交成功；真落地靠栅栏兜底（契约面语义）⇒ 这里 sync
        face.sync().await.expect("sync");
        let mut back = vec![0u8; n];
        face.dtoh(&dev, &mut back).await.expect("回读");
        assert_eq!(back, buf.as_slice(), "宿主→设备回灌必须逐字节一致");
        assert_eq!(
            back,
            (0..n).map(|i| (i % 251) as u8).collect::<Vec<_>>(),
            "图样"
        );
        tier.put(ChainHash::new(0), buf).expect("驻留");
    }

    #[tokio::test]
    async fn length_mismatch_is_rejected_before_touching_device() {
        let mut face = CpuFace::default();
        let (dev, n) = device_block(&mut face, 1024).await;
        let mut buf = HostBlock::zeroed(128); // 故意不符
        let err = offload_to_host(&mut face, &dev, n, &mut buf)
            .await
            .expect_err("必须拒");
        assert!(matches!(err, ModelError::Msg(_)), "得 {err:?}");
        let err = onload_from_host(&mut face, &buf, &dev, n)
            .await
            .expect_err("必须拒");
        assert!(matches!(err, ModelError::Msg(_)), "得 {err:?}");
    }

    #[tokio::test]
    async fn device_side_failure_propagates_loudly() {
        let mut face = CpuFace::default();
        let ghost = Bytes::new(9999, 1024, Dtype::F32, 1); // 未登记的块
        let mut buf = HostBlock::zeroed(4096);
        let err = offload_to_host(&mut face, &ghost, 4096, &mut buf)
            .await
            .expect_err("必须上抛");
        assert!(matches!(err, ModelError::Msg(_)), "得 {err:?}");
    }
}

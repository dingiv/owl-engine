//! 搬运闭环:设备 ⇄ 宿主(设计卷 §3.3 L2 的"搬家"腿)。
//!
//! **两条唯一的搬运口**(全部写成自由函数 —— `arch.md` 律⑦;store 本体不藏 I/O):
//!
//! ```text
//! offload_to_host(face, dev, buf)   设备块 --dtoh--> 宿主块(pageable;D2H 读语义)
//! onload_from_host(face, buf, dev)  宿主块 --pinned 暂存--> 设备块(upload_pinned)
//! ```
//!
//! 形态律(vLLM 同款):
//! - **宿主内容不 pin**:`dtoh` 直接读进 pageable 缓冲;回灌时经**一次性 pinned
//!   暂存租约**(`alloc_pinned` → 拷贝 → `upload_pinned`,租约所有权移入 server,
//!   流完成回调归还其池);
//! - **搬运不在捕获窗内**:调用点必须在段边界(设计卷 §3.4);本模块不做时序判断,
//!   只做搬运 —— 时序是 engine 的责任(接口形状上不含"随时搬"的能力);
//! - **失败即响亮**:任一原语失败原样上抛(带块号/字节),调用方据此回滚整单。

use owl_iface::contract::{Bytes, DeviceClient, Dtype, ModelError};

use crate::error::StoreError;
use crate::host::HostBlock;

/// 块句柄的**自述**字节数(元素口径 × dtype 宽)。
///
/// ⚠️ **不是块字节的真源**(2026-10-16 实测):CPU 面值块恒 f32
/// (`CpuFace::alloc` 回 `Bytes{dtype: F32}`),`htod` 更回 `len = 0`
/// ⇒ 各 face 的元素口径约定不同。**块字节的真源 = 调用方的块几何常量**
/// (store 用 `KvStore::block_bytes()`),搬运口一律**显式收 `len_bytes`**。
pub fn bytes_of(b: &Bytes) -> usize {
    b.len.saturating_mul(dtype_bytes(b.dtype))
}

fn dtype_bytes(d: Dtype) -> usize {
    match d {
        Dtype::F16 | Dtype::BF16 => 2,
        Dtype::F32 | Dtype::U32 => 4,
        // 8 位族(§S 2026-10-17 dtype 分离):fp8×2 与 int8 各自点名,
        // 与 `U8`(不透明字节)同宽 —— 本函数只用于块句柄**自述宽**
        // (真源仍是调用方几何),故三臂同值为 1。
        Dtype::U8 | Dtype::F8E4M3 | Dtype::F8E5M2 | Dtype::I8 => 1,
    }
}

fn map_err(e: ModelError, what: &str, bytes: usize) -> StoreError {
    StoreError::Invariant(format!("{what}({bytes} 字节):{e}"))
}

/// **设备 → 宿主(带偏移)** —— 状态存储"搬出去"的基本原语。
///
/// `dev_off_bytes` = 逻辑块在该设备缓冲内的页偏移(`槽 × 每页字节`);
/// 长度 = `dst.len()`(显式入参,坑册 §42:句柄自述长度不是真源)。
/// 后端面走 [`DeviceClient::dtoh_at`](缺实现 = 响亮拒,不静默少搬)。
pub async fn download_span<D: DeviceClient>(
    face: &mut D,
    dev: &Bytes,
    dev_off_bytes: usize,
    dst: &mut [u8],
) -> Result<(), StoreError> {
    let n = dst.len();
    face.dtoh_at(dev, dev_off_bytes, dst)
        .await
        .map_err(|e| map_err(e, "搬出 dtoh_at 失败", n))?;
    Ok(())
}

/// **宿主 → 设备(带偏移)** —— pinned 一次性租约作 DMA 源,所有权移入 server。
pub async fn upload_span<D: DeviceClient>(
    face: &mut D,
    src: &[u8],
    dev: &Bytes,
    dev_off_bytes: usize,
) -> Result<(), StoreError> {
    let n = src.len();
    let mut lease = face
        .alloc_pinned(n, "store:moves-staging")
        .await
        .map_err(|e| map_err(e, "搬入 alloc_pinned 失败", n))?;
    lease.slice_bytes_mut()[..n].copy_from_slice(src);
    face.upload_pinned(lease, dev, dev_off_bytes, n)
        .await
        .map_err(|e| map_err(e, "搬入 upload_pinned 失败", n))?;
    Ok(())
}

/// **设备 → 宿主(整块)** —— [`offload_to_host`] = `download_span(…, 0, …)` 的糖。
///
/// 长度须逐字节相等(宿主块大小 = 逻辑块字节)—— 不符 = 响亮拒(不许部分搬)。
pub async fn offload_to_host<D: DeviceClient>(
    face: &mut D,
    dev: &Bytes,
    len_bytes: usize,
    host: &mut HostBlock,
) -> Result<(), StoreError> {
    let n = len_bytes;
    if n != host.len() {
        return Err(StoreError::Invariant(format!(
            "卸载长度不符:设备块 {n} 字节 ≠ 宿主块 {} 字节",
            host.len()
        )));
    }
    download_span(face, dev, 0, host.as_mut_slice()).await
}

/// **宿主 → 设备(整块)** —— [`onload_from_host`] = `upload_span(…, 0)` 的糖。
///
/// 暂存租约的所有权随 `upload_pinned` 移入 server,由流完成回调归还其池
/// (即"回执 = 提交成功",真落地由调用方在段边界后的栅栏兜底)。
pub async fn onload_from_host<D: DeviceClient>(
    face: &mut D,
    host: &HostBlock,
    dev: &Bytes,
    len_bytes: usize,
) -> Result<(), StoreError> {
    let n = len_bytes;
    if n != host.len() {
        return Err(StoreError::Invariant(format!(
            "回灌长度不符:宿主块 {} 字节 ≠ 设备块 {n} 字节",
            host.len()
        )));
    }
    upload_span(face, host.as_slice(), dev, 0).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::{HostBlock, HostTier};
    use crate::id::ChainHash;
    use owl_cpu::CpuFace;

    /// 造一个设备块并写入可辨识图样(全程走契约面:alloc → pinned 租约 → upload_pinned)。
    /// ⚠️ CPU 面值块恒 f32 ⇒ `elems` 按 f32 记账(块字节 = elems × 4)。
    async fn device_block(face: &mut CpuFace, len_bytes: usize) -> (Bytes, usize) {
        let elems = len_bytes / 4;
        let b = face.alloc(Dtype::F32, elems, "t").await.expect("alloc");
        let n = len_bytes;
        let pat: Vec<u8> = (0..n).map(|i| (i % 251) as u8).collect();
        let mut lease = face.alloc_pinned(n, "t").await.expect("pinned 租约");
        lease.slice_bytes_mut()[..n].copy_from_slice(&pat);
        face.upload_pinned(lease, &b, 0, n).await.expect("upload_pinned");
        face.sync().await.expect("sync");
        (b, n)
    }

    #[tokio::test]
    async fn round_trip_device_host_device_is_byte_exact() {
        let mut face = CpuFace::default();
        let (dev, n) = device_block(&mut face, 8 << 10).await; // 8KiB 块

        let mut tier = HostTier::new(n, 1 << 20).expect("宿主层");
        let mut buf = tier.acquire().expect("宿主缓冲");
        offload_to_host(&mut face, &dev, n, &mut buf).await.expect("卸载");

        // 设备侧清零(模拟"已被驱逐/复用")⇒ 回灌后必须逐字节复原
        face.memset_zero(&dev, n).await.expect("清零");
        let mut after_zero = vec![0u8; n];
        face.dtoh(&dev, &mut after_zero).await.expect("回读");
        assert!(after_zero.iter().all(|b| *b == 0), "清零未生效,门失效");

        onload_from_host(&mut face, &buf, &dev, n).await.expect("回灌");
        // upload_pinned 回执 = 提交成功;真落地靠栅栏兜底(契约面语义)⇒ 这里 sync
        face.sync().await.expect("sync");
        let mut back = vec![0u8; n];
        face.dtoh(&dev, &mut back).await.expect("回读");
        assert_eq!(back, buf.as_slice(), "宿主→设备回灌必须逐字节一致");
        assert_eq!(back, (0..n).map(|i| (i % 251) as u8).collect::<Vec<_>>(), "图样");
        tier.put(ChainHash::new(0), buf).expect("驻留");
    }

    #[tokio::test]
    async fn length_mismatch_is_rejected_before_touching_device() {
        let mut face = CpuFace::default();
        let (dev, n) = device_block(&mut face, 1024).await;
        let mut buf = HostBlock::zeroed(128); // 故意不符
        let err = offload_to_host(&mut face, &dev, n, &mut buf).await.expect_err("必须拒");
        assert!(matches!(err, StoreError::Invariant(_)), "得 {err:?}");
        let err = onload_from_host(&mut face, &buf, &dev, n).await.expect_err("必须拒");
        assert!(matches!(err, StoreError::Invariant(_)), "得 {err:?}");
    }

    #[tokio::test]
    async fn device_side_failure_propagates_loudly() {
        let mut face = CpuFace::default();
        let ghost = Bytes::new(9999, 1024, Dtype::F32, 1); // 未登记的块
        let mut buf = HostBlock::zeroed(4096);
        let err = offload_to_host(&mut face, &ghost, 4096, &mut buf).await.expect_err("必须上抛");
        assert!(matches!(err, StoreError::Invariant(_)), "得 {err:?}");
    }
}

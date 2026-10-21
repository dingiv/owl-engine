//! **快照下沉经济账探针**(2026-10-19;§9.1(6)b 立案的定点取证)。
//!
//! 背景:8417 tok 冷 prompt ⇒ 14 个前缀边界 ⇒ 12 次状态腿快照下沉(每次 159 MiB)
//! ⇒ `prefill.bnd` avg 71.6ms / p99 212ms ≈ 该请求 TTFT 16%。连发实测给出
//! **边际成本 ≈ 100 ms/条**(0.75 s/GB ⇒ 1.6 GB/s 级)。
//!
//! 两个候选病因(本探针分辨):
//!  ① **逐片 DMA 开销**:状态腿一条快照 = 4 tensor × N 层 ≈ **192 片**(每片 ~0.8 MiB)
//!     ⇒ 192 次 `memcpy_dtoh_async`(每次 driver 调用开销 O(0.5ms))≈ 100ms;
//!  ② **pinned → pageable 宿主拷贝**:finish 回调里 `conn.put` 一次 159 MiB
//!     拷进宿主层的 pageable `Vec<u8>`(≈1.6 GB/s?)≈ 100ms。
//!
//! 判据:同字节量下**只变片数**(1 / 4 / 192 片)⇒ 若耗时随片数线性涨 = ①;
//! 若三种片数耗时接近、而纯 host memcpy(pinned→pageable,159 MiB)≈ 同样量级 = ②。
//!
//! 跑法:
//! ```bash
//! CUDA_VISIBLE_DEVICES=GPU-<3090Ti> OWL_TEST_DEVICE=<BDF> \
//!   cargo test -p owl-cuda --test swap_econ_probe -- --ignored --nocapture
//! ```

use std::any::Any;
use std::time::Instant;

use owl_cuda::{
    test_device_ordinal, ConnSlot, Connector, DeviceClient as _, DeviceSelector, Dtype, GpuClient,
    GpuServer, ModelError, SwapDir, SwapPiece,
};

/// 一条快照的字节量(27B 实测:`4 槽 × 159252480B/条`)
const SNAP_BYTES: usize = 159_252_480;
const N_PIECES_REAL: usize = 192; // 4 tensor × 48 GDN 层(实测片数)

/// **丢弃型**连接器:只走 DMA/staging 路,不做宿主拷贝 ⇒ 拆出"纯搬运"那半。
struct NullConn {
    seen: usize,
}

impl Connector for NullConn {
    fn put(&mut self, _slot: ConnSlot, _off: usize, bytes: &[u8]) -> Result<(), ModelError> {
        self.seen += bytes.len();
        Ok(())
    }
    fn get(&self, _slot: ConnSlot, _off: usize, out: &mut [u8]) -> Result<(), ModelError> {
        out.fill(0);
        Ok(())
    }
    fn tier_name(&self) -> &'static str {
        "null(probe)"
    }
    fn unbox(self: Box<Self>) -> Box<dyn Any + Send> {
        self
    }
}

/// 宿主层同款:pageable RAM 槽台。`huge: true` = 带大页提示(与 `HostBlock::zeroed` 同形)。
struct VecConn {
    slots: Vec<Vec<u8>>,
}

fn host_block(len: usize, huge: bool) -> Vec<u8> {
    let mut v = vec![0u8; len];
    if huge {
        #[cfg(target_os = "linux")]
        unsafe {
            const HP: usize = 2 << 20;
            let base = v.as_mut_ptr() as usize;
            let lo = (base + HP - 1) & !(HP - 1);
            let hi = (base + len) & !(HP - 1);
            if hi > lo {
                let _ = libc::madvise(lo as *mut libc::c_void, hi - lo, libc::MADV_HUGEPAGE);
            }
        }
    }
    v
}

impl Connector for VecConn {
    fn put(&mut self, slot: ConnSlot, off: usize, bytes: &[u8]) -> Result<(), ModelError> {
        let s = self
            .slots
            .get_mut(slot.0 as usize)
            .ok_or_else(|| ModelError::Msg(format!("VecConn: 槽 {} 不在场", slot.0)))?;
        s[off..off + bytes.len()].copy_from_slice(bytes);
        Ok(())
    }
    fn get(&self, slot: ConnSlot, off: usize, out: &mut [u8]) -> Result<(), ModelError> {
        let s = self
            .slots
            .get(slot.0 as usize)
            .ok_or_else(|| ModelError::Msg(format!("VecConn: 槽 {} 不在场", slot.0)))?;
        out.copy_from_slice(&s[off..off + out.len()]);
        Ok(())
    }
    fn tier_name(&self) -> &'static str {
        "ram(probe)"
    }
    fn unbox(self: Box<Self>) -> Box<dyn Any + Send> {
        self
    }
}

fn boot() -> GpuClient {
    let (tx, rx) = std::sync::mpsc::channel::<owl_cuda::Command>();
    let server = GpuServer::new(rx, DeviceSelector::Ordinal(test_device_ordinal()), None);
    let client = GpuClient::new(tx);
    std::thread::Builder::new()
        .spawn(move || server.run().expect("server run"))
        .expect("spawn");
    client
}

/// 把 `total` 字节切成 `n` 片(片等长;末片取余)的 swap 片表。
fn pieces(block: u64, total: usize, n: usize) -> Vec<SwapPiece> {
    let per = total / n;
    let mut out = Vec::with_capacity(n);
    let mut dev = 0usize;
    let mut conn = 0usize;
    for i in 0..n {
        let len = if i + 1 == n { total - dev } else { per };
        out.push(SwapPiece { block, dev_off: dev, conn_off: conn, len, slot: ConnSlot(0) });
        dev += len;
        conn += len;
    }
    out
}

async fn swap_out_once(client: &mut GpuClient, block: u64, total: usize, n: usize, host: Vec<u8>) -> (f64, Vec<u8>) {
    let conn = VecConn { slots: vec![host] };
    let t0 = Instant::now();
    let conn = client
        .swap(SwapDir::Out, &pieces(block, total, n), Box::new(conn))
        .await
        .expect("swap out");
    let dt = t0.elapsed().as_secs_f64() * 1e3;
    let conn = *conn.unbox().downcast::<VecConn>().expect("原箱奉还");
    (dt, conn.slots[0].clone())
}

#[ignore = "bench: 快照下沉经济账(片数 × 宿主拷贝)"]
#[tokio::test]
async fn swap_econ() {
    if owl_cuda::test_device_bdf().is_none() {
        owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
        return;
    }
    let mut client = boot();
    client.sync().await.expect("sync");
    let blk = client
        .alloc(Dtype::U8, SNAP_BYTES, "econ")
        .await
        .expect("alloc 159MiB");
    // 先写一遍(经 pinned 租约,顺带预热 staging 池)
    for i in 0..(SNAP_BYTES / (1 << 20)) {
        let mut lease = client.alloc_pinned(1 << 20, "econ:warm").await.expect("pinned");
        lease.slice_bytes_mut()[..1 << 20].fill((i % 251) as u8);
        client
            .upload_pinned(lease, &blk, i << 20, 1 << 20)
            .await
            .expect("upload");
    }
    client.sync().await.expect("sync");

    // ① 片数扫描(每档 2 次:第一次含宿主 Vec 首触,第二次为稳态)
    for n in [1usize, 4, 24, N_PIECES_REAL] {
        for round in 0..2 {
            let host = vec![0u8; SNAP_BYTES];
            let (dt, _) = swap_out_once(&mut client, blk.id, SNAP_BYTES, n, host).await;
            eprintln!(
                "[swap-econ] 多片 Out {n:3} 片({:.1} MiB/片)round{round} = {dt:8.1} ms ⇒ {:.2} GB/s",
                SNAP_BYTES as f64 / n as f64 / 1048576.0,
                SNAP_BYTES as f64 / 1e9 / (dt / 1e3)
            );
        }
    }

    // ①b 拆解:192 片 → **丢弃型**连接器(只走 DMA + pinned staging,无宿主拷贝)
    for round in 0..2 {
        let conn = NullConn { seen: 0 };
        let t0 = Instant::now();
        let conn = client
            .swap(SwapDir::Out, &pieces(blk.id, SNAP_BYTES, N_PIECES_REAL), Box::new(conn))
            .await
            .expect("swap out(null)");
        let dt = t0.elapsed().as_secs_f64() * 1e3;
        let conn = *conn.unbox().downcast::<NullConn>().expect("原箱");
        eprintln!(
            "[swap-econ] 192 片 Out → NullConn(无宿主拷贝)round{round} = {dt:8.1} ms(搬了 {} B)",
            conn.seen
        );
    }
    // ①c 对照:同 192 片 → THP 宿主块(与生产 HostBlock 同形)
    for round in 0..2 {
        let (dt, _) = swap_out_once(&mut client, blk.id, SNAP_BYTES, N_PIECES_REAL, host_block(SNAP_BYTES, true)).await;
        eprintln!("[swap-econ] 192 片 Out → 宿主块(THP 冷写)round{round} = {dt:8.1} ms");
    }

    // ② 纯 host 拷贝:pageable → pageable(基线)vs 一份"pinned 源 → pageable 目的"
    let src = vec![7u8; SNAP_BYTES];
    for round in 0..2 {
        let mut dst = vec![0u8; SNAP_BYTES];
        let t0 = Instant::now();
        dst.copy_from_slice(&src[..]);
        let dt = t0.elapsed().as_secs_f64() * 1e3;
        eprintln!(
            "[swap-econ] host memcpy pageable→pageable 159 MiB round{round} = {dt:8.1} ms ⇒ {:.2} GB/s(冷目的)",
            SNAP_BYTES as f64 / 1e9 / (dt / 1e3)
        );
        let t0 = Instant::now();
        dst.copy_from_slice(&src[..]);
        let dt = t0.elapsed().as_secs_f64() * 1e3;
        eprintln!(
            "[swap-econ] host memcpy 同缓冲复拷(warm)                  = {dt:8.1} ms ⇒ {:.2} GB/s",
            SNAP_BYTES as f64 / 1e9 / (dt / 1e3)
        );
    }
}

/// **大页提示 A/B**(2026-10-19):冷写 159 MiB 的缺页成本 —— 4 KiB 页 × 4 万次
/// vs `MADV_HUGEPAGE` 的 2 MiB 页 × 76 次(内核 THP = `[madvise]` ✓)。
#[ignore = "bench: 冷写缺页 A/B(4KB vs THP)"]
#[tokio::test]
async fn host_cold_write_hugepage_ab() {
    for tag in ["4KB(基线)", "MADV_HUGEPAGE"] {
        for round in 0..2 {
            let mut dst = vec![0u8; SNAP_BYTES];
            if tag.starts_with("MADV") {
                #[cfg(target_os = "linux")]
                unsafe {
                    // 2MB 对齐的内区间(整页覆盖;边缘一段按 4KB 处理)
                    const HP: usize = 2 << 20;
                    let base = dst.as_mut_ptr() as usize;
                    let lo = (base + HP - 1) & !(HP - 1);
                    let hi = (base + dst.len()) & !(HP - 1);
                    let r = libc::madvise(lo as *mut libc::c_void, hi - lo, libc::MADV_HUGEPAGE);
                    if r != 0 {
                        eprintln!(
                            "[swap-econ] madvise 失败: {:?}(base=0x{base:x} lo=0x{lo:x} len={})",
                            std::io::Error::last_os_error(),
                            hi - lo
                        );
                    }
                }
            }
            let t0 = Instant::now();
            for chunk in dst.chunks_mut(1 << 20) {
                chunk.fill(0xA5);
            }
            let dt = t0.elapsed().as_secs_f64() * 1e3;
            eprintln!(
                "[swap-econ] 冷写 159 MiB {tag:14} round{round} = {dt:7.1} ms ⇒ {:.2} GB/s",
                SNAP_BYTES as f64 / 1e9 / (dt / 1e3)
            );
        }
    }
}

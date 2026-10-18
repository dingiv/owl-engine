//! **`swap` 一条命令门**(状态存储的交换面,2026-10-16;设计卷 §15)。
//!
//! 判据:
//! 1. `Out`(设备 → 连接器):N 片一次提交、一次 ack **即全部落定** ——
//!    回执后逐片比对必须逐字节一致(`dev_off` 与 `conn_off` **各说各话**:
//!    设备侧分层、连接器侧连续);
//! 2. `In`(连接器 → 设备):回执后再从设备读回,必须逐字节一致;
//! 3. 越界/空表 = **响亮拒**(不静默少搬);
//! 4. 回执**原箱奉还**(`unbox` 取回自己的连接器类型)。
//!
//! 跑法:`CUDA_VISIBLE_DEVICES=GPU-<3080> OWL_TEST_DEVICE=0 \
//!        cargo test -p owl-cuda --test swap_probe -- --nocapture`

use std::any::Any;

use owl_cuda::{
    test_device_ordinal, ConnSlot, Connector, DeviceClient as _, DeviceSelector, Dtype, GpuClient,
    GpuServer, ModelError, SwapDir, SwapPiece,
};

const BLOCK: usize = 4096; // 设备块字节

fn pattern(n: usize, seed: u8) -> Vec<u8> {
    (0..n).map(|i| seed.wrapping_add((i % 251) as u8)).collect()
}

/// 期望 swap 失败(`Ok` 侧 `Box<dyn Connector>` 不实现 `Debug` ⇒ 不能用 `expect_err`)
macro_rules! must_err {
    ($fut:expr) => {
        match $fut.await {
            Ok(_) => panic!("应响亮拒,却成功了"),
            Err(e) => e,
        }
    };
}

/// 测试连接器:RAM 槽台(每槽 = 一块逻辑块的连续字节)。
struct VecConn {
    slots: Vec<Vec<u8>>,
}

impl Connector for VecConn {
    fn put(&mut self, slot: ConnSlot, off: usize, bytes: &[u8]) -> Result<(), ModelError> {
        let n = self.slots.len();
        let s = self
            .slots
            .get_mut(slot.0 as usize)
            .ok_or_else(|| ModelError::Msg(format!("VecConn: 槽 {} ≥ {n}", slot.0)))?;
        let end = off.checked_add(bytes.len())
            .ok_or_else(|| ModelError::Msg("VecConn: 偏移溢出".into()))?;
        if end > s.len() {
            return Err(ModelError::Msg(format!("VecConn: 槽 {} 越界 {end} > {}", slot.0, s.len())));
        }
        s[off..end].copy_from_slice(bytes);
        Ok(())
    }

    fn get(&self, slot: ConnSlot, off: usize, out: &mut [u8]) -> Result<(), ModelError> {
        let n = self.slots.len();
        let s = self
            .slots
            .get(slot.0 as usize)
            .ok_or_else(|| ModelError::Msg(format!("VecConn: 槽 {} ≥ {n}", slot.0)))?;
        let end = off.checked_add(out.len())
            .ok_or_else(|| ModelError::Msg("VecConn: 偏移溢出".into()))?;
        if end > s.len() {
            return Err(ModelError::Msg(format!("VecConn: 槽 {} 越界 {end} > {}", slot.0, s.len())));
        }
        out.copy_from_slice(&s[off..end]);
        Ok(())
    }

    fn tier_name(&self) -> &'static str {
        "test-ram"
    }

    fn unbox(self: Box<Self>) -> Box<dyn Any + Send> {
        self
    }
}

/// 起 server + client(与 dtoh_at_probe 同脚手架)。
fn boot() -> GpuClient {
    let (tx, rx) = std::sync::mpsc::channel::<owl_cuda::Command>();
    let server = GpuServer::new(rx, DeviceSelector::Ordinal(test_device_ordinal()), None);
    let client = GpuClient::new(tx);
    std::thread::Builder::new()
        .spawn(move || server.run().expect("server run"))
        .expect("spawn server");
    client
}

/// **换出 → 换入往返**(两个设备块:源块 → 连接器 → 目标块)。
#[tokio::test]
async fn swap_out_then_in_is_byte_exact() {
    if owl_cuda::test_device_bdf().is_none() {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    }
    let mut client = boot();
    client.sync().await.expect("sync");

    let src = client.alloc(Dtype::U32, BLOCK / 4, "probe").await.expect("src");
    let dst = client.alloc(Dtype::U32, BLOCK / 4, "probe").await.expect("dst");
    // 源块:三段图样(带偏移写入)
    let a = pattern(1024, 1);
    let b = pattern(1024, 2);
    let c = pattern(2048, 3);
    for (data, off) in [(&a, 0usize), (&b, 1024), (&c, 2048)] {
        let mut lease = client.alloc_pinned(data.len(), "probe:swap").await.expect("pinned");
        lease.slice_bytes_mut()[..data.len()].copy_from_slice(data);
        client.upload_pinned(lease, &src, off, data.len()).await.expect("upload");
    }
    client.sync().await.expect("sync");

    // ── Out:三片 → 槽 0(**片序与槽序故意不同**,验证 `dev_off`/`conn_off` 各归其位)
    //  片 0: dev[2048..4096)(2048B,长片) → conn[2048..4096)
    //  片 1: dev[0..1024)               → conn[0..1024)
    //  片 2: dev[1024..2048)            → conn[1024..2048)
    let pieces_out = vec![
        SwapPiece { block: src.id, dev_off: 2048, conn_off: 2048, len: 2048, slot: ConnSlot(0) },
        SwapPiece { block: src.id, dev_off: 0, conn_off: 0, len: 1024, slot: ConnSlot(0) },
        SwapPiece { block: src.id, dev_off: 1024, conn_off: 1024, len: 1024, slot: ConnSlot(0) },
    ];
    let conn = VecConn { slots: vec![vec![0u8; BLOCK]] };
    let conn = client.swap(SwapDir::Out, &pieces_out, Box::new(conn)).await.expect("swap out");
    // 回执 = 原箱奉还(unbox 取回具体类型)
    let conn = *conn.unbox().downcast::<VecConn>().expect("原箱奉还的必须是 VecConn");
    let got = &conn.slots[0];
    assert_eq!(&got[0..1024], &a[..], "片 1(dev[0..1024))落 conn[0..1024)");
    assert_eq!(&got[1024..2048], &b[..], "片 2(dev[1024..2048))落 conn[1024..2048)");
    assert_eq!(&got[2048..4096], &c[..], "片 0(dev[2048..4096),2048B)落 conn[2048..4096)");

    // ── In:槽 0 → 目标块(片序再换一次:先尾后头)
    //  片 0: conn[2048..4096) → dst[2048..4096)
    //  片 1: conn[0..1024)    → dst[0..1024)
    //  片 2: conn[1024..2048) → dst[1024..2048)
    let pieces_in = vec![
        SwapPiece { block: dst.id, dev_off: 2048, conn_off: 2048, len: 2048, slot: ConnSlot(0) },
        SwapPiece { block: dst.id, dev_off: 0, conn_off: 0, len: 1024, slot: ConnSlot(0) },
        SwapPiece { block: dst.id, dev_off: 1024, conn_off: 1024, len: 1024, slot: ConnSlot(0) },
    ];
    let conn2 = VecConn { slots: vec![got.clone()] };
    let conn2 = client.swap(SwapDir::In, &pieces_in, Box::new(conn2)).await.expect("swap in");
    let conn2 = *conn2.unbox().downcast::<VecConn>().expect("In 也原箱奉还");
    assert_eq!(conn2.slots[0], got[..], "In 不该改连接器内容(只是读源)");
    client.sync().await.expect("sync");

    let mut back = vec![0u8; BLOCK];
    client.dtoh(&dst, &mut back).await.expect("dtoh");
    // 逐段拼接后必须与源块**整体**一致
    let mut want = Vec::with_capacity(BLOCK);
    want.extend_from_slice(&a);
    want.extend_from_slice(&b);
    want.extend_from_slice(&c);
    assert_eq!(back, want, "换出→换入往返必须逐字节一致");
}

/// **越界 / 空表 = 响亮拒**(不许静默少搬;回执必达)。
#[tokio::test]
async fn swap_rejects_bounds_and_empty_loudly() {
    if owl_cuda::test_device_bdf().is_none() {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    }
    let mut client = boot();
    client.sync().await.expect("sync");
    let dev = client.alloc(Dtype::U32, BLOCK / 4, "probe").await.expect("dev");

    // ① 设备侧越界
    let bad = vec![SwapPiece { block: dev.id, dev_off: BLOCK - 8, conn_off: 0, len: 16, slot: ConnSlot(0) }];
    let conn = VecConn { slots: vec![vec![0u8; BLOCK]] };
    let err = must_err!(client.swap(SwapDir::Out, &bad, Box::new(conn)));
    eprintln!("[swap 门] 设备越界拒原文: {err}");

    // ② 连接器侧越界(put 自己拒 —— 契约要求响亮)
    let ok = vec![SwapPiece { block: dev.id, dev_off: 0, conn_off: BLOCK - 8, len: 16, slot: ConnSlot(0) }];
    let conn = VecConn { slots: vec![vec![0u8; BLOCK]] };
    let err = must_err!(client.swap(SwapDir::Out, &ok, Box::new(conn)));
    eprintln!("[swap 门] 连接器越界拒原文: {err}");

    // ③ 空片段表
    let conn = VecConn { slots: vec![] };
    let err = must_err!(client.swap(SwapDir::Out, &[], Box::new(conn)));
    eprintln!("[swap 门] 空表拒原文: {err}");
    assert!(format!("{err}").contains("空片段表"), "错误须自带定位:{err}");
}

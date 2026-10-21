//! TP2 S2 冒烟:AR 两核(`ar_stage`/`ar_reduce`)在真机上的最小实证(自环)。
//!
//! 判据(单卡自环;C3 的双 rank 对拍门 G1-G3 另立):
//! 1. stage → reduce 两步序列**不死锁**、header 游标推进(1 → 2);
//! 2. payload 数值 = 2× 原值(自环 allreduce 的数学期望;连跑两次 = 4×);
//! 3. **超时负例**:对端 produced 永不推进 ⇒ 5 s 内 `err=2` 响亮落账(D4),
//!    且 payload **保持原样**(不静默出结果)。
//!
//! 运行:`OWL_TEST_DEVICE=<空闲卡> cargo test -p owl-kernels --features device
//!       --test comm_ar_smoke -- --nocapture`

use cudarc::driver::{CudaContext, CudaFunction, DevicePtr, LaunchConfig, PushKernelArg};
use owl_kernels::family::comm::asset;
use std::sync::Arc;

const HDR: usize = 64;
const SLOT_BYTES: u64 = 4096;
const N_SLOTS: u32 = 2;
const MAGIC: u32 = 0x4F57_4C41; // "OWLA"
const BLOCK: u32 = 256;
const N: usize = 1024;

fn dev_ordinal() -> Option<usize> {
    owl_shared::env_reader::parse("OWL_TEST_DEVICE")
}

/// 资产 → 模块 → 符号(与生产 `Exec::load_aot` 同一条资产链)
fn load_fn(ctx: &Arc<CudaContext>, src_id: &str, symbol: &str) -> CudaFunction {
    let a = owl_kernels::family::cubins::asset(src_id)
        .unwrap_or_else(|| panic!("{src_id} 应有构建期 fatbin 资产"));
    let m = ctx
        .load_module(cudarc::nvrtc::Ptx::from_binary(a.bytes.to_vec()))
        .unwrap_or_else(|e| panic!("load_module {src_id}: {e:?}"));
    m.load_function(symbol).unwrap_or_else(|e| panic!("符号 {symbol}: {e:?}"))
}

/// staging header(64 B;布局与 `cu/comm/ar_stage.cu` 头注一致)
fn hdr_bytes(slot_bytes: u32, n_slots: u32, produced: u64, consumed: u64) -> [u8; HDR] {
    let mut b = [0u8; HDR];
    b[0..8].copy_from_slice(&produced.to_le_bytes());
    b[8..16].copy_from_slice(&consumed.to_le_bytes());
    b[16..20].copy_from_slice(&slot_bytes.to_le_bytes());
    b[20..24].copy_from_slice(&n_slots.to_le_bytes());
    b[24..28].copy_from_slice(&MAGIC.to_le_bytes());
    b
}

fn hdr_u64(h: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(h[off..off + 8].try_into().expect("8B"))
}
fn hdr_u32(h: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(h[off..off + 4].try_into().expect("4B"))
}

/// 6 槽发射(stage/reduce 同型)
#[allow(clippy::too_many_arguments)]
unsafe fn launch6(
    stream: &Arc<cudarc::driver::CudaStream>,
    f: &CudaFunction,
    a: u64,
    own: u64,
    peer: u64,
    bytes: u64,
    slot_bytes: u64,
    slab_off: u64,
    tag: &str,
) {
    stream
        .launch_builder(f)
        .arg(&a)
        .arg(&own)
        .arg(&peer)
        .arg(&bytes)
        .arg(&slot_bytes)
        .arg(&slab_off)
        .launch(LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        })
        .unwrap_or_else(|e| panic!("{tag} launch: {e:?}"));
}

#[test]
fn comm_ar_smoke() {
    let Some(ord) = dev_ordinal() else {
        owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设(需一张空闲卡)");
        return;
    };
    let ctx = CudaContext::new(ord).expect("建上下文");
    let stream = ctx.default_stream();
    let f_stage = load_fn(&ctx, asset::STAGE_SRC, asset::STAGE_F32);
    let f_reduce = load_fn(&ctx, asset::REDUCE_SRC, asset::REDUCE_F32);

    let total = HDR + (SLOT_BYTES * N_SLOTS as u64) as usize;

    // ── 正例:自环两跑(游标 1 → 2;payload 2× → 4×)────────────────────
    let mut hst = vec![0u8; total];
    hst[..HDR].copy_from_slice(&hdr_bytes(SLOT_BYTES as u32, N_SLOTS, 0, 0));
    let staging = stream.clone_htod(&hst).expect("htod staging");
    let xs: Vec<f32> = (0..N).map(|i| (i as f32) * 0.5 - 3.0).collect();
    let payload = stream.clone_htod(&xs).expect("htod payload");
    let bytes = (N * 4) as u64;
    let slab_off = 0u64;

    for round in 1..=2u64 {
        let (p_local, _) = payload.device_ptr(&stream);
        let (p_own, _) = staging.device_ptr(&stream);
        let (p_peer, _) = staging.device_ptr(&stream); // 自环:own == peer
        unsafe {
            launch6(&stream, &f_stage, p_local, p_own, p_peer, bytes, SLOT_BYTES, slab_off, "ar_stage");
            launch6(&stream, &f_reduce, p_local, p_own, p_peer, bytes, SLOT_BYTES, slab_off, "ar_reduce");
        }
        stream.synchronize().expect("sync");
        let h = stream.clone_dtoh(&staging).expect("dtoh staging");
        assert_eq!(hdr_u64(&h, 0), round, "produced 游标(round {round})");
        assert_eq!(hdr_u64(&h, 8), round, "consumed 游标(round {round})");
        assert_eq!(hdr_u32(&h, 28), 0, "err 应为 0(round {round})");
        let got: Vec<f32> = stream.clone_dtoh(&payload).expect("dtoh payload");
        let scale = 1u32 << round; // 2× → 4×
        for (i, (g, x)) in got.iter().zip(xs.iter()).enumerate() {
            let want = x * scale as f32;
            assert!(
                (g - want).abs() <= 1e-3 * want.abs().max(1.0),
                "round {round}: payload[{i}] = {g},期望 {want}(自环 = {scale}×)"
            );
        }
    }

    // ── 负例:超时路径(peer.produced 恒 0 ⇒ reduce 等 consumed=0 → 超时)──
    let mut hst2 = vec![0u8; total];
    hst2[..HDR].copy_from_slice(&hdr_bytes(SLOT_BYTES as u32, N_SLOTS, 0, 0));
    let dead = stream.clone_htod(&hst2).expect("htod dead staging");
    let before: Vec<f32> = stream.clone_dtoh(&payload).expect("dtoh before");
    let (p_dp, _) = payload.device_ptr(&stream);
    let (p_dead, _) = dead.device_ptr(&stream);
    let t0 = std::time::Instant::now();
    unsafe {
        // reduce(out=payload, own=dead(cons=0), peer=dead(prod=0))⇒ 等 prod>0 超时
        launch6(&stream, &f_reduce, p_dp, p_dead, p_dead, bytes, SLOT_BYTES, slab_off, "ar_reduce(timeout)");
    }
    stream.synchronize().expect("sync");
    let waited = t0.elapsed();
    let h2 = stream.clone_dtoh(&dead).expect("dtoh dead");
    assert_eq!(hdr_u32(&h2, 28), 2, "超时必须置 err=2(reduce 数据超时)");
    assert_eq!(hdr_u64(&h2, 8), 0, "超时不得发布 consumed(不静默)");
    let after: Vec<f32> = stream.clone_dtoh(&payload).expect("dtoh after");
    assert_eq!(before, after, "超时路径不得改 payload(全有或全无)");
    assert!(waited.as_secs() < 30, "超时应 ~5s 内返回,实测 {waited:?}");

    eprintln!(
        "comm_ar_smoke 全绿:自环 2 轮(游标 1→2,payload 2×/4×);超时负例 err=2({:?})",
        waited
    );
}

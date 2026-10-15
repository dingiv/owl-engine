//! **事件记账底噪**探针(`docs/arch/三流事件序-设计与迁移.md` §5「性能」判据)。
//!
//! E1 起每条上 GPU 的命令末尾多一次 `cuEventRecord`;判据 = **个位数 µs**
//! —— 若超标,E2 就得改"批量记账"(一条命令一次 record 已是理论下界)。
//!
//! 本测只量**原语本身**:裸 context + 一条流 + 池化事件轮转(与 server
//! 同姿势:`CU_EVENT_DISABLE_TIMING` + 8 事件轮转,不复用挂在等待中的事件)。
//!
//! 运行:`OWL_TEST_DEVICE=<空闲卡> cargo test -p owl-cuda --test event_record_cost`
//! 门:单次 record 的 p50 **< 10µs**(实测远超则响亮失败 —— 门不静默放宽)。

use owl_cuda::ffi;
use owl_cuda::test_device_ordinal;

/// 轮转记录 N 次,返回 (p50_ns, p99_ns, 单次均值_ns)
fn record_cost(n: usize, pool: usize) -> (u64, u64, u64) {
    let ctx = ffi::CudaContext::new(test_device_ordinal()).expect("开设备");
    ctx.bind_to_thread().expect("bind");
    let stream = ctx.new_stream().expect("建流");
    let mut events = Vec::with_capacity(pool);
    for i in 0..pool {
        events.push(
            ffi::event_create(ffi::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING)
                .unwrap_or_else(|e| panic!("建事件({i}): {e:?}")),
        );
    }
    let mut samples = Vec::with_capacity(n);
    for i in 0..n {
        let ev = events[i % pool];
        let t0 = std::time::Instant::now();
        unsafe { ffi::event_record(ev, stream.cu_stream()) }.expect("record");
        samples.push(t0.elapsed().as_nanos() as u64);
    }
    // 排空一次(把提交队列落定,顺带确认 record 本身不阻塞)
    stream.synchronize().expect("流排空");
    for ev in events {
        unsafe { let _ = ffi::event_destroy(ev); }
    }
    let total: u64 = samples.iter().sum();
    samples.sort_unstable();
    (
        samples[samples.len() / 2],
        samples[(samples.len() * 99) / 100],
        total / samples.len() as u64,
    )
}

#[test]
fn event_record_is_sub_10us() {
    if !owl_shared::env_reader::flag("OWL_TEST_DEVICE") {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    }
    let (p50, p99, mean) = record_cost(2000, 8);
    eprintln!("[ev-cost] cuEventRecord 单次: p50={p50}ns p99={p99}ns mean={mean}ns(池 8 轮转)");
    assert!(
        p50 < 10_000,
        "cuEventRecord p50 = {p50}ns ≥ 10µs —— E2 必须改批量记账(E1 判据见设计文档 §5)"
    );
}

/// **跨流等待的提交开销**(E2 判据):A 流录事件 → B 流 `waitEvent` N 次。
/// 这是 E3 要用来替换「host 全排空(≈1.9ms/键)」的那一刀的代价上界;
/// 判据同前:**个位数 µs**(若超标,E3 的收益就要重算)。
#[test]
fn stream_wait_event_is_sub_10us() {
    if !owl_shared::env_reader::flag("OWL_TEST_DEVICE") {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    }
    let ctx = ffi::CudaContext::new(test_device_ordinal()).expect("开设备");
    ctx.bind_to_thread().expect("bind");
    let a = ctx.new_stream().expect("流 A(H2D 角色)");
    let b = ctx.new_stream().expect("流 B(COMPUTE 角色)");
    let mut pool = Vec::new();
    for _ in 0..8 {
        pool.push(
            ffi::event_create(ffi::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING).expect("建事件"),
        );
    }
    let mut samples = Vec::with_capacity(2000);
    for i in 0..2000usize {
        let ev = pool[i % pool.len()];
        unsafe { ffi::event_record(ev, a.cu_stream()) }.expect("record(A)");
        let t0 = std::time::Instant::now();
        unsafe { ffi::stream_wait_event(b.cu_stream(), ev, ffi::sys::CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT) }
            .expect("wait(B)");
        samples.push(t0.elapsed().as_nanos() as u64);
    }
    let _ = b.synchronize();
    let _ = a.synchronize();
    for ev in pool {
        unsafe { let _ = ffi::event_destroy(ev); }
    }
    samples.sort_unstable();
    let p50 = samples[samples.len() / 2];
    let p99 = samples[(samples.len() * 99) / 100];
    let mean = samples.iter().sum::<u64>() / samples.len() as u64;
    eprintln!("[ev-cost] cuStreamWaitEvent 提交: p50={p50}ns p99={p99}ns mean={mean}ns(A 录→B 等, 池 8)");
    assert!(
        p50 < 10_000,
        "cuStreamWaitEvent p50 = {p50}ns ≥ 10µs —— E3 的收益须重算(设计 §5)"
    );
}

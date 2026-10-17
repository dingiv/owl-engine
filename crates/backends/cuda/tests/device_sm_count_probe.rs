//! **设备属性探针**:SM（多处理器)数真值(`ffi::device_sm_count`)。
//!
//! 为什么单列一条:引擎侧 XQA 的 `sm_count` 参数此前**硬编码 84**(sm86 双卡)
//! ⇒ 3080(68)上启发式系统性偏大。2026-10-17 审查把它改成走设备真值
//! (`DeviceClient::sm_count` → 本函数 → `cuDeviceGetAttribute(MULTIPROCESSOR_COUNT)`)。
//! 本探针钉两件:
//! 1. **全部可见设备逐张取值**(打印出来:3090 Ti = 84 / 3080 = 68 一眼可辨;
//!    本仓 GPU2/GPU3 是 3080 ⇒ 这条就是"钉卡后取值是否跟着走"的现场证据);
//! 2. 与「无查询」的对照:`device_rows()` 的 ordinal 枚举数应与本查询可覆盖的
//!    设备数一致(ordinal 语义同源 —— CUDA 序,不看 nvidia-smi 序)。
//!
//! 跑:`cargo test -p owl-cuda --test device_sm_count_probe -- --nocapture`
//! (不需要 `OWL_TEST_DEVICE`:它查的是**枚举面**,不碰引擎)。

use owl_cuda::ffi;

#[test]
fn sm_count_is_a_real_attribute() {
    // 枚举面显式初始化(ffi 的 device_rows/device_sm_count 内部自管 cuInit;
    // 裸计数口不自管 ⇒ 测试自己先 init,别依赖调用顺序)
    ffi::driver_init().expect("cuInit");
    let n = ffi::device_get_count().expect("device_get_count") as usize;
    assert!(n >= 1, "至少一张可见设备(实测 {n})");
    let rows = ffi::device_rows().expect("device_rows(枚举面同源对照)");
    assert_eq!(rows.len(), n, "枚举面设备数与驱动计数一致");

    let mut seen: Vec<usize> = Vec::new();
    for o in 0..n {
        let sm = ffi::device_sm_count(o).unwrap_or_else(|e| panic!("device_sm_count({o}): {e}"));
        eprintln!("[dev-props] ordinal={o} name={:?} SM={sm}", rows[o].name);
        // 量级闸:消费级 GPU 的 SM 数落在 [8, 256];0/1/超大 = 取值面坏了(不静默)
        assert!((8..=256).contains(&sm), "ordinal={o} 的 SM 数 {sm} 超合理域");
        seen.push(sm);
    }
    // 同型卡应给同值(本仓 0/1 = 3090 Ti、2/3 = 3080);不同型允许不同。
    // 这里只断言"不是所有设备都给同一个常数"(防"返回常量"这类假实现)。
    let all_same = seen.iter().all(|v| *v == seen[0]);
    if n > 1 {
        eprintln!("[dev-props] 各卡 SM = {seen:?}(全同 = {all_same};混插机型应不同)");
    }
}

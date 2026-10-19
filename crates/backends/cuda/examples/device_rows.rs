//! **设备表打印**(排障小工具):CUDA ordinal ↔ 名字 / BDF / UUID。
//!
//! 用途:确认本 boot 的 CUDA 序号映射(顺序受 `CUDA_DEVICE_ORDER` 影响,
//! 默认 `FASTEST_FIRST` ⇒ 与 `nvidia-smi` 的 index **不一致**)。
//!
//! ```bash
//! cargo run -p owl-cuda --example device_rows
//! ```
fn main() {
    match owl_cuda::ffi::device_rows() {
        Ok(rows) => {
            for r in rows {
                println!("ordinal={} name={:?} bdf={} uuid={}", r.ordinal, r.name, r.bdf, r.uuid);
            }
        }
        Err(e) => {
            eprintln!("设备枚举失败: {e}");
            std::process::exit(1);
        }
    }
}

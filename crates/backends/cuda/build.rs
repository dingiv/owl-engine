//! 构建脚本(owl-cuda)—— **只为 CUPTI 归因 shim 服务**(feature `cupti`)。
//!
//! 默认(feature 关)本脚本**什么都不做**:不编 shim、不引 libcupti、
//! 不加任何链接参数 ⇒ 生产二进制与之前逐字节等价(A4 依赖纪律)。
//!
//! 为什么用 C shim 而不是纯 Rust FFI:CUPTI 的 activity 明文结构**逐 CUDA
//! 版本演进**(13.2 = `CUpti_ActivityKernel11`),Rust 侧手抄 ABI = 静默错值
//! 陷阱;shim 在编译期吃真头文件,只把压好的 plain struct 交给 Rust。
//! 与 `owl-kernels` 的 marlin/flashinfer shim 同一形态。

use std::path::PathBuf;
use std::process::Command;

fn main() {

    // FIXME: 这个是做什么的? 不允许依赖环境变量 CARGO_FEATURE_CUPTI
    println!("cargo:rerun-if-changed=native/cupti_shim.c");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");

    // feature 关(缺省)= 零动作:不进链接面
    if std::env::var("CARGO_FEATURE_CUPTI").is_err() {
        return;
    }

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    let cuda_home =
        std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".to_string());
    let inc = PathBuf::from(&cuda_home).join("include");
    let lib = PathBuf::from(&cuda_home).join("lib64");

    let obj = out_dir.join("cupti_shim.o");
    let ar = out_dir.join("libowl_cupti_shim.a");
    let src = PathBuf::from("native/cupti_shim.c");

    // 1) 编译 shim(C;不吃 nvcc)
    let st = Command::new("cc")
        .args(["-O2", "-fPIC", "-Wall", "-Wno-unused-parameter"])
        .arg(format!("-I{}", inc.display()))
        .arg("-c")
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .status()
        .expect("cupti shim: cc 启动失败");
    assert!(st.success(), "cupti shim: cc 编译失败(检查 CUDA_HOME/include/cupti_activity.h)");

    // 2) 归档
    let _ = std::fs::remove_file(&ar);
    let st = Command::new("ar")
        .arg("rcs")
        .arg(&ar)
        .arg(&obj)
        .status()
        .expect("cupti shim: ar 启动失败");
    assert!(st.success(), "cupti shim: ar 失败");

    // 3) 链接面:静态 shim + 动态 libcupti(+ rpath,免 LD_LIBRARY_PATH)
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-lib=static=owl_cupti_shim");
    println!("cargo:rustc-link-lib=dylib=cupti");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());
    println!("cargo:warning=owl-cuda: CUPTI 归因 shim 已编入(feature cupti;{} 头)", inc.display());
}

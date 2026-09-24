//! 构建期预编译:cu/ops.cu → PTX(nvcc)。
//!
//! 产物经 OUT_DIR/ops.ptx 被 include_str! 内嵌进二进制,运行时零编译
//! (驱动 load_module 时对该设备 JIT)。
//!
//! 环境变量:
//! - `OWL_CUDA_ARCH`:目标虚拟架构(默认 compute_86;PTX 前向兼容,
//!   高卡由驱动 JIT;换卡/换架构 = 改此值重编)。
//! - `OWL_NVCC`:nvcc 可执行文件(默认 "nvcc")。

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=cu/ops.cu");
    println!("cargo:rerun-if-env-changed=OWL_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=OWL_NVCC");

    let arch = env::var("OWL_CUDA_ARCH").unwrap_or_else(|_| "compute_86".into());
    let nvcc = env::var("OWL_NVCC").unwrap_or_else(|_| "nvcc".into());

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let ptx = out_dir.join("ops.ptx");

    let src = PathBuf::from("cu/ops.cu");
    println!("cargo:rerun-if-changed={}", src.display());

    let status = Command::new(&nvcc)
        .args([
            format!("--gpu-architecture={arch}"),
            "--ptx".into(),
            src.display().to_string(),
            "-o".into(),
            ptx.display().to_string(),
        ])
        .status()
        .unwrap_or_else(|e| panic!("build.rs: 无法启动 {nvcc:?}(需要 CUDA toolkit 在 PATH): {e}"));

    if !status.success() {
        panic!("build.rs: nvcc 预编译 cu/ops.cu 失败(arch={arch})");
    }
    println!("cargo:rustc-env=OWL_OPS_PTX_PATH={}", ptx.display());
}

//! 构建期预编译:cu/ops.cu → PTX(nvcc);marlin W4A16 → .a(nvcc,feature "marlin")。
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
    // PTX 预编面仅 cuda feature 需要;marlin 段仅 marlin feature 需要
    // (纯源码之家消费方零 nvcc 依赖)
    if std::env::var_os("CARGO_FEATURE_MARLIN").is_some() {
        build_marlin();
    }
    let has_cuda = std::env::var_os("CARGO_FEATURE_CUDA").is_some();
    if !has_cuda {
        return;
    }
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


fn build_marlin() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    // ------------------------------------------------------------------
    // marlin W4A16(f16 激活 × u4 权重 × f16 scales;foreign-kernel 通道)
    // 预编 .a 入库,日常构建零 nvcc;FORCE_BUILD 时三源 nvcc(sm86)。
    // 旗子纪律(照 xinfer marlin-ffi 先例,禁 fast_math;vLLM≥12.8 的
    // -static-global-template-stub=false 必带,否则跨 TU 拒链)。
    // ------------------------------------------------------------------
    if std::env::var_os("CARGO_FEATURE_MARLIN").is_some() {
        println!("cargo:rerun-if-changed=cu/marlin/prebuilt/libmarlin.a");
        println!("cargo:rerun-if-env-changed=MARLIN_FORCE_BUILD");
        println!("cargo:rerun-if-env-changed=MARLIN_CUDA_ARCH");
        println!("cargo:rerun-if-env-changed=MARLIN_NVCC");
        let marlin_prebuilt = PathBuf::from("cu/marlin/prebuilt/libmarlin.a");
        let force = env::var("MARLIN_FORCE_BUILD").ok().as_deref() == Some("1");
        let obj_dir = out_dir.join("marlin");
        std::fs::create_dir_all(&obj_dir).unwrap();
        if marlin_prebuilt.exists() && !force {
            let dest = out_dir.join("libmarlin.a");
            std::fs::copy(&marlin_prebuilt, &dest).expect("copy prebuilt libmarlin.a");
            println!("cargo:warning=owl-kernels: using prebuilt {}", marlin_prebuilt.display());
        } else {
            let arch = env::var("MARLIN_CUDA_ARCH").unwrap_or_else(|_| "86".into());
            let nvcc_m = env::var("MARLIN_NVCC").unwrap_or_else(|_| "nvcc".into());
            let base = PathBuf::from("cu/marlin");
            let sources = [
                base.join("marlin_host.cu"),
                base.join("vllm_marlin/sm80_kernel_float16_u4_float16.cu"),   // AWQ 非对称(kU4)
                base.join("vllm_marlin/sm80_kernel_float16_u4b8_float16.cu"), // GPTQ 对称(kU4B8)
            ];
            let includes = [base.join("vllm_marlin"), base.join("vllm_marlin/shim")];
            let mut objs = Vec::new();
            for src in &sources {
                println!("cargo:rerun-if-changed={}", src.display());
                let obj = obj_dir.join(src.file_name().unwrap()).with_extension("o");
                let mut args = vec![
                    "-c".to_string(),
                    src.to_str().unwrap().to_string(),
                    "-o".to_string(),
                    obj.to_str().unwrap().to_string(),
                ];
                for inc in &includes {
                    args.push("-I".into());
                    args.push(inc.to_str().unwrap().to_string());
                }
                args.extend([
                    format!("-arch=compute_{arch}"),
                    format!("-code=sm_{arch}"),
                    "-O3".into(),
                    "-std=c++17".into(),
                    "-Xcompiler".into(),
                    "-fPIC".into(),
                    // 禁 -fvisibility=hidden:__global__ 模板实例跨 TU 拒链(xinfer 坑)
                    "--expt-relaxed-constexpr".into(),
                    "-static-global-template-stub=false".into(),
                    "-Xcudafe".into(),
                    "--diag_suppress=20236".into(),
                ]);
                let status = Command::new(&nvcc_m)
                    .args(&args)
                    .status()
                    .unwrap_or_else(|e| panic!("marlin: failed to spawn {nvcc_m}: {e}"));
                assert!(status.success(), "marlin: nvcc failed for {}", src.display());
                objs.push(obj);
            }
            let lib = out_dir.join("libmarlin.a");
            let ar = env::var("AR").unwrap_or_else(|_| "ar".into());
            let mut cmd = Command::new(&ar);
            cmd.arg("rcs").arg(lib.to_str().unwrap());
            for o in &objs {
                cmd.arg(o.to_str().unwrap());
            }
            let status = cmd.status().expect("marlin: failed to spawn ar");
            assert!(status.success(), "marlin: ar failed");
            println!("cargo:warning=owl-kernels: built libmarlin.a (sm_{arch}) from source");
        }
        let cuda_lib = env::var("CUDA_HOME")
            .map(|h| PathBuf::from(h).join("lib64"))
            .unwrap_or_else(|_| PathBuf::from("/usr/local/cuda/lib64"));
        println!("cargo:rustc-link-search=native={}", out_dir.display());
        println!("cargo:rustc-link-search=native={}", cuda_lib.display());
        println!("cargo:rustc-link-lib=static=marlin");
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rustc-link-lib=dylib=stdc++");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", cuda_lib.display());
    }
}

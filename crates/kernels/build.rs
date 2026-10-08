//! 构建期预编译:marlin W4A16 → .a(nvcc,feature "marlin")。
//! (旧 cu/ops.cu → PTX 预编链随 cuda_ops 退役删除,2026-10-12;
//!  现役内核编译 = server nvrtc 懒编译 + AOT cubin 资产,零 build.rs PTX 面。)

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // 仅 marlin feature 需要 nvcc(纯源码之家消费方零 nvcc 依赖)
    if std::env::var_os("CARGO_FEATURE_MARLIN").is_none() {
        return;
    }
    build_marlin();
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
        // .h 追踪(E5-DF3 同日十二:selector 改动不触发重编 → host .o 落后
        // 一代 → 新旧分支混链 undefined)
        for h in [
            "cu/marlin/vllm_marlin/kernel.h",
            "cu/marlin/vllm_marlin/kernel_selector.h",
            "cu/marlin/vllm_marlin/kernel_selector_full.upstream.h",
            "cu/marlin/vllm_marlin/marlin_template.h",
            "cu/marlin/vllm_marlin/marlin_dtypes.cuh",
            "cu/marlin/vllm_marlin/marlin_mma.h",
            "cu/marlin/vllm_marlin/dequant.h",
        ] {
            println!("cargo:rerun-if-changed={}", h);
        }
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
                base.join("vllm_marlin/sm80_kernel_bfloat16_u4b8_bfloat16.cu"), // BF16 激活变体(E5-DF3 草稿 BF16 化)
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

    // ------------------------------------------------------------------
    // FlashInfer prefill(预编 .a 入库;foreign-kernel 通道;E1.5 工作包)
    // FLASHINFER_FORCE_BUILD=1 重编:nvcc -arch=sm_86 -I<flashinfer>/include
    // (fork 检出:~/.cudaforge/git/checkouts/flashinfer-0f06c2305a276bcb;
    //  裁剪面见 cu/flashinfer/owl_fi_attention.cu 头注)
    // ------------------------------------------------------------------
    if std::env::var_os("CARGO_FEATURE_FLASHINFER").is_some() {
        println!("cargo:rerun-if-changed=cu/flashinfer/prebuilt/libowl_flashinfer.a");
        println!("cargo:rerun-if-env-changed=FLASHINFER_FORCE_BUILD");
        let fi_prebuilt = PathBuf::from("cu/flashinfer/prebuilt/libowl_flashinfer.a");
        let force = env::var("FLASHINFER_FORCE_BUILD").ok().as_deref() == Some("1");
        let fi_dir = out_dir.join("flashinfer");
        std::fs::create_dir_all(&fi_dir).unwrap();
        if fi_prebuilt.exists() && !force {
            let dest = fi_dir.join("libowl_flashinfer.a");
            std::fs::copy(&fi_prebuilt, &dest).expect("copy prebuilt libowl_flashinfer.a");
            println!("cargo:warning=owl-kernels: using prebuilt {}", fi_prebuilt.display());
        } else {
            let fi_include = env::var("FLASHINFER_INCLUDE").unwrap_or_else(|_| {
                "/home/div/.cudaforge/git/checkouts/flashinfer-0f06c2305a276bcb/include".into()
            });
            let nvcc_m = env::var("FLASHINFER_NVCC").unwrap_or_else(|_| "nvcc".into());
            let src = PathBuf::from("cu/flashinfer/owl_fi_attention.cu");
            let obj = fi_dir.join("owl_fi_attention.o");
            let status = std::process::Command::new(nvcc_m)
                .args(["-O3", "-std=c++17", "-arch=sm_86"])
                .arg(format!("-I{}", fi_include))
                .arg("-c").arg(&src).arg("-o").arg(&obj)
                .status()
                .expect("flashinfer: failed to spawn nvcc");
            assert!(status.success(), "flashinfer: nvcc failed");
            let ar = env::var("AR").unwrap_or_else(|_| "ar".into());
            let status = std::process::Command::new(ar)
                .args(["rcs"]).arg(fi_dir.join("libowl_flashinfer.a")).arg(&obj)
                .status()
                .expect("flashinfer: failed to spawn ar");
            assert!(status.success(), "flashinfer: ar failed");
            println!("cargo:warning=owl-kernels: built libowl_flashinfer.a from source");
        }
        println!("cargo:rustc-link-search=native={}", fi_dir.display());
        println!("cargo:rustc-link-lib=static=owl_flashinfer");
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rustc-link-lib=dylib=stdc++");
        let cuda_lib = env::var("CUDA_HOME")
            .map(|h| PathBuf::from(h).join("lib64"))
            .unwrap_or_else(|_| PathBuf::from("/usr/local/cuda/lib64"));
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", cuda_lib.display());
    }
}

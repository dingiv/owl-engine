//! C1 spike 探针:**构建期 fatbin 资产**载入 + 发射 + host 参考对拍。
//!
//! 目的(设计 `docs/arch/kernel-准入门槛-施工设计.md` §15.3/§15.4):
//! 在把 native 全量从 NVRTC 迁到 cubin 之前,先用**最小可判据**证明
//! "构建期 `nvcc -fatbin` 产物能被 driver 载入、符号能解析、发射出正确数值"
//! 这条链端到端成立(含 `include_bytes!` 从 `$OUT_DIR` 取资产)。
//!
//! 判据三条(任一失败即机制不成立):
//! 1. **载入**:`cuModuleLoadData` 吃 fatbin(多代际容器,非纯 cubin)不报错;
//! 2. **符号**:每个入口在**自己 `SOURCE`** 的资产里可见,且**边界独占**
//!    (§15.3 审计①:dtype 变体随源码走,build.rs 只需参数化 **arch**;
//!    C 轮拆分后本判据升级为“文件边界 = 算子族边界”);
//! 3. **数值**:`owl_add_f16` 与 host 参考**逐元素逐位一致**(f16 加法在
//!    两端都是 IEEE half 运算,无容差余地),且与 NVRTC 源码通道**逐位等价**
//!    (迁移期"换载体不换数值"硬门)。
//!
//! f16 缓冲在测试里按 `u16` 位型走(`half::f16` 未实现 cudarc 的 `DeviceRepr`,
//! 且 f16 位型拷贝本就无必要做类型化 —— 位型走私在测试域是显式的)。
//!
//! 运行纪律:`OWL_TEST_DEVICE` 钉卡(缺省跳过)+ `--test-threads=1`。

use cudarc::driver::{CudaContext, CudaFunction, DevicePtr, LaunchConfig, PushKernelArg};
use half::f16;

/// 构建期资产 —— 经**源哈希**查表(与生产路径同一入口,非测试专用通道)
fn add_asset() -> &'static owl_kernels::family::cubins::CubinAsset {
    let src = owl_kernels::spec::meta_of::<owl_kernels::ops::elems::AddF16>().src;
    owl_kernels::family::cubins::asset(src)
        .expect("cu/elems/add.cu 应有构建期 fatbin 资产(build.rs 未编?)")
}

const N: usize = 1024;
const BLOCK: u32 = 256;

fn dev_ordinal() -> Option<usize> {
    owl_shared::env_reader::parse("OWL_TEST_DEVICE")
}

fn bits(v: &[f16]) -> Vec<u16> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// 发射 owl_add_f16(n=1024,N 个元素),回读位型
fn run_add(
    ctx: &std::sync::Arc<CudaContext>,
    f: &CudaFunction,
    a: &[f16],
    b: &[f16],
    tag: &str,
) -> Vec<f16> {
    let stream = ctx.default_stream();
    let da = stream.clone_htod(&bits(a)).expect("htod a");
    let db = stream.clone_htod(&bits(b)).expect("htod b");
    let dout = stream.alloc_zeros::<u16>(N).expect("alloc out");
    let (pa, _) = da.device_ptr(&stream);
    let (pb, _) = db.device_ptr(&stream);
    let (po, _) = dout.device_ptr(&stream);
    let n = N as u64;
    // SAFETY: 参数槽位与 `owl_add_f16` 的实例行签名严格对位
    // (a,b,out 三个 `const __half*` → u64 位型;n → size_t),块长/网格覆盖 N。
    unsafe {
        stream
            .launch_builder(f)
            .arg(&pa)
            .arg(&pb)
            .arg(&po)
            .arg(&n)
            .launch(LaunchConfig {
                grid_dim: ((N as u32).div_ceil(BLOCK), 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
            .unwrap_or_else(|e| panic!("{tag} launch: {e:?}"));
    }
    stream.synchronize().expect("sync");
    let out: Vec<u16> = stream.clone_dtoh(&dout).expect("dtoh");
    out.into_iter().map(f16::from_bits).collect()
}

fn inputs() -> (Vec<f16>, Vec<f16>) {
    // 输入域:覆盖正负、±0、次正规、f16 极值(防"全正小数"式假绿)
    let a: Vec<f16> = (0..N)
        .map(|i| match i % 7 {
            0 => f16::from_f32(1.0 + i as f32),
            1 => f16::from_f32(-(i as f32) * 0.5),
            2 => f16::from_f32(0.0),
            3 => f16::from_f32(-0.0),
            4 => f16::MIN_POSITIVE,
            5 => f16::from_f32(65504.0),
            _ => f16::from_f32((i as f32) * 1e-3),
        })
        .collect();
    let b: Vec<f16> = (0..N).map(|i| f16::from_f32((i % 13) as f32 - 6.5)).collect();
    (a, b)
}

/// fatbin 载入 + **文件边界 = 算子族边界**(判据 1+2;**C 轮改造**)
///
/// C 轮把 `ops_pair.cu` 按**算子族**拆开后,旧判据“一个文件里全变体一次全出”
/// 反了过来:**每个文件只应装本族的入口**。本门因此改成两条 ——
/// 1. 每个已建表入口,在它自己 `SOURCE` 的资产里**必须**可见;
/// 2. 同一入口在**别的**资产的模块里**必须不可见**(边界证明:拆干净了 ——
///    既没把邻族的核顺手带过去,也没出现“同一核两处各留一份”)。
///
/// 两组判据的清单由 [`owl_kernels::ops::all`] 按源分组**自动生成** ——
/// 加表/加文件即自动纳入,零第二份清单。
#[test]
fn fatbin_loads_and_exposes_all_variants() {
    let Some(ord) = dev_ordinal() else {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    };
    use std::collections::BTreeMap;
    let ctx = CudaContext::new(ord).expect("context");

    // 源文本 → 该源的全部已建表入口(每源一个模块)
    let mut by_src: BTreeMap<&'static str, Vec<&'static str>> = BTreeMap::new();
    for m in owl_kernels::ops::all() {
        if m.src.is_empty() {
            continue; // foreign(FFI):无 .cu 源,另由 face 常量对拍
        }
        by_src.entry(m.src).or_default().push(m.entry);
    }
    let all_names: Vec<&'static str> = by_src.values().flatten().copied().collect();

    let mut mods = Vec::new();
    for (src, names) in &by_src {
        let asset = owl_kernels::family::cubins::asset(src)
            .unwrap_or_else(|| panic!("源无 AOT 资产(build.rs 未编?):首入口 {}", names[0]));
        let module = ctx
            .load_module(cudarc::nvrtc::Ptx::from_binary(asset.bytes.to_vec()))
            .unwrap_or_else(|e| panic!("fatbin 载入失败({}):{e:?}", asset.module));
        mods.push((asset.module, module, names));
    }

    // 判据 1:本族符号可见
    for (asset, module, names) in &mods {
        for n in names.iter() {
            module
                .load_function(n)
                .unwrap_or_else(|e| panic!("{n} 不在自己的资产 {asset} 里: {e:?}"));
        }
    }
    // 判据 2:跨文件边界 —— 邻族入口不可见
    for (asset, module, names) in &mods {
        for n in &all_names {
            if names.contains(n) {
                continue;
            }
            assert!(
                module.load_function(n).is_err(),
                "{asset} 里竟然有邻族入口 {n} —— 拆分没拆干净(同一核两处各留一份?)"
            );
        }
    }
    eprintln!(
        "C1 判据 1+2 绿:{} 个资产载入成功,{} 个入口各归其位(跨文件边界逐一无泄漏)",
        mods.len(),
        all_names.len()
    );
}

/// 发射 + 数值(判据 3a):`owl_add_f16` 对 host 参考逐位一致
#[test]
fn fatbin_launch_matches_host_reference() {
    let Some(ord) = dev_ordinal() else {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    };
    let ctx = CudaContext::new(ord).expect("context");
    let asset = add_asset();
    let module = ctx
        .load_module(cudarc::nvrtc::Ptx::from_binary(asset.bytes.to_vec()))
        .expect("fatbin 载入");
    let f = module.load_function("owl_add_f16").expect("owl_add_f16");
    let (a, b) = inputs();
    let want: Vec<f16> = a.iter().zip(&b).map(|(x, y)| *x + *y).collect();
    let got = run_add(&ctx, &f, &a, &b, "fatbin");

    let mut bad = 0usize;
    let mut first: Option<(usize, f16, f16)> = None;
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        if g != w {
            bad += 1;
            if first.is_none() {
                first = Some((i, *g, *w));
            }
        }
    }
    assert_eq!(
        bad, 0,
        "owl_add_f16 fatbin 通道与 host 参考不一致: {bad}/{N} 处不同,首个 {first:?}"
    );
    eprintln!("C1 判据 3a 绿:owl_add_f16 × {N} 逐位一致(fatbin → driver → SASS)");
}

/// 判据 3b:同一核走 **NVRTC 源码**与走 **fatbin 资产**结果逐位一致
/// (证明"换载体不换数值"—— 迁移期等价性硬门)。
#[test]
fn nvrtc_and_fatbin_agree_bitwise() {
    let Some(ord) = dev_ordinal() else {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    };
    let ctx = CudaContext::new(ord).expect("context");
    // NVRTC 对照臂:源文本按资产 id 从磁盘取(仅测试面;生产路径零源文本)
    let src_id = owl_kernels::spec::meta_of::<owl_kernels::ops::elems::AddF16>().src;
    let src = owl_kernels::list::cu_text(src_id);
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(
        &src,
        cudarc::nvrtc::CompileOptions {
            include_paths: vec!["/usr/local/cuda/include".into()],
            ..Default::default()
        },
    )
    .expect("nvrtc 编译(判据:源在树 ⇒ 任何平台可重编)");
    let m_nvrtc = ctx.load_module(ptx).expect("nvrtc module");
    let f_nvrtc = m_nvrtc.load_function("owl_add_f16").expect("nvrtc fn");
    let m_fat = ctx
        .load_module(cudarc::nvrtc::Ptx::from_binary(add_asset().bytes.to_vec()))
        .expect("fatbin module");
    let f_fat = m_fat.load_function("owl_add_f16").expect("fatbin fn");

    let (a, b) = inputs();
    let out_nvrtc = run_add(&ctx, &f_nvrtc, &a, &b, "nvrtc");
    let out_fat = run_add(&ctx, &f_fat, &a, &b, "fatbin");
    let diff = out_nvrtc
        .iter()
        .zip(&out_fat)
        .filter(|(x, y)| x != y)
        .count();
    assert_eq!(diff, 0, "两载体不一致: {diff}/{N}(迁移期等价性硬门破了)");
    eprintln!("C1 判据 3b 绿:nvrtc 与 fatbin 两条载体逐位一致");
}

// ============================================================================
// C2 等价性(AOT 面):**同源不同编译器**的输出逐位对拍
// ============================================================================
//
// 迁渡判据:fatbin(nvcc -O3 + ptxas)与 NVRTC(→ PTX → 驱动 JIT)是**两条
// 独立代码生成链**;源码同一份(`__CUDACC_RTC__` 分臂保证 NVRTC 侧预处理结果
// 与改动前逐字相同),但 SASS 不保证逐位一致。故对**有精度风险的核**(smem
// 归约 / 分区归约 / 平局语义)做逐位对拍:一致 ⇒ AOT 面继承既有全部
// 正确性证据(金标/e2e)。
//
// 覆盖:三个被 C2 改动的源文件各一(smem 归约 / 平局 / 分区注意力链)。

use cudarc::driver::CudaModule;
use std::sync::Arc;

/// 双载体模块:nvrtc(源)与 fatbin(资产)
fn dual_modules(ctx: &Arc<CudaContext>, src_id: &str, source: &str) -> (Arc<CudaModule>, Arc<CudaModule>) {
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(
        source,
        cudarc::nvrtc::CompileOptions {
            arch: Some("compute_86"),
            include_paths: vec!["/usr/local/cuda/include".into()],
            ..Default::default()
        },
    )
    .expect("nvrtc");
    let a = ctx.load_module(ptx).expect("nvrtc module");
    let asset = owl_kernels::family::cubins::asset(src_id).expect("源应有 AOT 资产");
    let b = ctx
        .load_module(cudarc::nvrtc::Ptx::from_binary(asset.bytes.to_vec()))
        .expect("fatbin module");
    (a, b)
}

fn cmp_bitwise(tag: &str, a: &[u16], b: &[u16]) {
    let bad = a.iter().zip(b).filter(|(x, y)| x != y).count();
    let first = a.iter().zip(b).position(|(x, y)| x != y);
    assert_eq!(bad, 0, "{tag}: nvrtc vs fatbin 输出不一致 {bad}/{} 处,首个 {first:?}", a.len());
    eprintln!("[aot-equiv] {tag} 绿({} 元素逐位一致)", a.len());
}

fn lcg_f16(n: usize, seed: u32) -> Vec<u16> {
    (0..n)
        .map(|i| {
            let x = seed
                .wrapping_mul(0x9E37_79B9)
                .wrapping_add((i as u32).wrapping_mul(0x85EB_CA6B));
            let f = ((x & 0xFFFF) as f32 / 65535.0 - 0.5) * 4.0;
            f16::from_f32(f).to_bits()
        })
        .collect()
}

/// smem 归约 + 分支(`owl_rmsnorm_f16`;w_off 两臂各测)
#[test]
fn aot_equiv_rmsnorm_f16() {
    let Some(ord) = dev_ordinal() else {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    };
    let ctx = CudaContext::new(ord).expect("ctx");
    let stream = ctx.default_stream();
    let src_id = owl_kernels::spec::meta_of::<owl_kernels::ops::ln::RmsnormF16>().src;
    let src = owl_kernels::list::cu_text(src_id);
    let (mn, mf) = dual_modules(&ctx, src_id, &src);

    const ROWS: usize = 8;
    const N: usize = 1024;
    let x = lcg_f16(ROWS * N, 7);
    let alpha = lcg_f16(N, 13);

    let run = |m: &Arc<CudaModule>, w_off: i32| -> Vec<u16> {
        let f = m.load_function("owl_rmsnorm_f16").expect("fn");
        let dx = stream.clone_htod(&x).unwrap();
        let da = stream.clone_htod(&alpha).unwrap();
        let dout = stream.alloc_zeros::<u16>(ROWS * N).unwrap();
        let (px, _) = dx.device_ptr(&stream);
        let (pa, _) = da.device_ptr(&stream);
        let (po, _) = dout.device_ptr(&stream);
        let (nn, eps) = (N as i32, 1e-6f32);
        unsafe {
            stream
                .launch_builder(&f)
                .arg(&px)
                .arg(&pa)
                .arg(&po)
                .arg(&nn)
                .arg(&eps)
                .arg(&w_off)
                .launch(LaunchConfig {
                    grid_dim: (ROWS as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 256 * 4,
                })
                .expect("launch");
        }
        stream.synchronize().unwrap();
        stream.clone_dtoh(&dout).unwrap()
    };
    for w_off in [0i32, 1i32] {
        let a = run(&mn, w_off);
        let b = run(&mf, w_off);
        cmp_bitwise(&format!("owl_rmsnorm_f16(w_off={w_off})"), &a, &b);
    }
}

/// 平局语义(`owl_argmax_f32idx_f16`;该源文件为 C2 改动之一)
#[test]
fn aot_equiv_argmax_ties() {
    let Some(ord) = dev_ordinal() else {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    };
    let ctx = CudaContext::new(ord).expect("ctx");
    let stream = ctx.default_stream();
    let src_id = owl_kernels::spec::meta_of::<owl_kernels::ops::elems::ArgmaxF32IdxF16>().src;
    let src = owl_kernels::list::cu_text(src_id);
    let (mn, mf) = dual_modules(&ctx, src_id, &src);

    // 精心构造平局:多个相同最大值(取**最小索引**) + 全负 + 首位最大
    let mut x = lcg_f16(4096, 5);
    x[100] = f16::from_f32(7.5).to_bits();
    x[200] = f16::from_f32(7.5).to_bits();
    x[300] = f16::from_f32(7.5).to_bits();
    x[0] = f16::from_f32(9.0).to_bits();
    let run = |m: &Arc<CudaModule>| -> Vec<u16> {
        let f = m.load_function("owl_argmax_f32idx_f16").expect("fn");
        let dx = stream.clone_htod(&x).unwrap();
        let dout = stream.alloc_zeros::<f32>(1).unwrap();
        let (px, _) = dx.device_ptr(&stream);
        let (po, _) = dout.device_ptr(&stream);
        let (n, off) = (x.len() as i32, 0i32);
        unsafe {
            stream
                .launch_builder(&f)
                .arg(&px)
                .arg(&n)
                .arg(&off)
                .arg(&po)
                .launch(LaunchConfig { grid_dim: (1, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })
                .expect("launch");
        }
        stream.synchronize().unwrap();
        let out: Vec<f32> = stream.clone_dtoh(&dout).unwrap();
        out.iter().map(|v| v.to_bits() as u16).collect()
    };
    let a = run(&mn);
    let b = run(&mf);
    cmp_bitwise("owl_argmax_f32idx_f16(平局域)", &a, &b);
}

/// 分区注意力链(`vllm_paged_attention_v2_f16_hd256bs32` + reduce;该源文件为
/// C2 改动之一 —— 内层核限定符 `__global__` → `__device__` 在 nvcc 面)
#[test]
fn aot_equiv_paged_attention_v2_chain() {
    let Some(ord) = dev_ordinal() else {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    };
    let ctx = CudaContext::new(ord).expect("ctx");
    let stream = ctx.default_stream();
    let src_id = owl_kernels::spec::meta_of::<owl_kernels::ops::attn::PagedV2F16Hd256Bs32>().src;
    let src = owl_kernels::list::cu_text(src_id);
    let (mn, mf) = dual_modules(&ctx, src_id, &src);

    const HD: usize = 256;
    const PAGE: usize = 32;
    const X: usize = 8;
    const HEADS: usize = 8;
    const KV_HEADS: usize = 2;
    const NB: usize = 64;
    const THREADS: usize = 128;
    const SEQ: usize = 600;
    let nparts = SEQ.div_ceil(512); // 2
    let bps = SEQ.div_ceil(PAGE) + 2; // 21
    let head_stride = KV_HEADS * (HD / X) * PAGE * X; // 16384
    let k = lcg_f16(NB * head_stride, 11);
    let v = lcg_f16(NB * head_stride, 23);
    let q = lcg_f16(HEADS * HD, 37);
    let mut bt = vec![0f32; bps];
    for (b, e) in bt.iter_mut().enumerate() {
        *e = b as f32;
    }
    let lens = vec![SEQ as f32];
    let alibi = vec![0f32];

    let run = |m: &Arc<CudaModule>| -> Vec<u16> {
        let kf = m.load_function("vllm_paged_attention_v2_f16_hd256bs32").expect("v2 fn");
        let rf = m.load_function("vllm_paged_attention_v2_reduce_f16_hd256").expect("reduce fn");
        let dq = stream.clone_htod(&q).unwrap();
        let dk = stream.clone_htod(&k).unwrap();
        let dv = stream.clone_htod(&v).unwrap();
        let dbt = stream.clone_htod(&bt).unwrap();
        let dl = stream.clone_htod(&lens).unwrap();
        let dab = stream.clone_htod(&alibi).unwrap();
        let d_es = stream.alloc_zeros::<f32>(HEADS * nparts).unwrap();
        let d_ml = stream.alloc_zeros::<f32>(HEADS * nparts).unwrap();
        let d_tmp = stream.alloc_zeros::<u16>(HEADS * nparts * HD).unwrap();
        let d_out = stream.alloc_zeros::<u16>(HEADS * HD).unwrap();
        let (pq, _) = dq.device_ptr(&stream);
        let (pk, _) = dk.device_ptr(&stream);
        let (pv, _) = dv.device_ptr(&stream);
        let (pbt, _) = dbt.device_ptr(&stream);
        let (pl, _) = dl.device_ptr(&stream);
        let (pab, _) = dab.device_ptr(&stream);
        let (pes, _) = d_es.device_ptr(&stream);
        let (pml, _) = d_ml.device_ptr(&stream);
        let (ptp, _) = d_tmp.device_ptr(&stream);
        let (po, _) = d_out.device_ptr(&stream);
        let (nkv, scale, mnbp) = (KV_HEADS as i32, 1f32 / (HD as f32).sqrt(), bps as i32);
        let (qst, bst, hst) = ((HEADS * HD) as i32, head_stride as i32, (HD / X * PAGE * X) as i32);
        let (ss, sw, ua) = (1f32, 0i32, 0i32);
        unsafe {
            stream
                .launch_builder(&kf)
                .arg(&pq).arg(&pk).arg(&pv).arg(&pbt).arg(&pl).arg(&pab)
                .arg(&pes).arg(&pml)
                .arg(&nkv).arg(&scale).arg(&mnbp).arg(&qst).arg(&bst).arg(&hst)
                .arg(&ss).arg(&sw).arg(&ua).arg(&ptp)
                .launch(LaunchConfig {
                    grid_dim: (HEADS as u32, 1, nparts as u32),
                    block_dim: (THREADS as u32, 1, 1),
                    shared_mem_bytes: (SEQ.div_ceil(PAGE) * PAGE * 4).max(2048) as u32,
                })
                .expect("v2 launch");
            let np = nparts as i32;
            stream
                .launch_builder(&rf)
                .arg(&pes).arg(&pml).arg(&ptp).arg(&pl).arg(&np).arg(&po)
                .launch(LaunchConfig {
                    grid_dim: (HEADS as u32, 1, 1),
                    block_dim: (THREADS as u32, 1, 1),
                    shared_mem_bytes: (2 * nparts * 4) as u32,
                })
                .expect("reduce launch");
        }
        stream.synchronize().unwrap();
        stream.clone_dtoh(&d_out).unwrap()
    };
    let a = run(&mn);
    let b = run(&mf);
    cmp_bitwise("paged_attention_v2_f16_hd256bs32 + reduce", &a, &b);
}

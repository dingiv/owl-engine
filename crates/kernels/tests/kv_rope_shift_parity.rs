//! **图像窗口旋转**的池内 K Δ-rope 核 —— 设备 ⇔ host 逐位对拍
//! (image-rotation, 2026-10-20;设计卷 §5.3/§8 G1a 的机械面)。
//!
//! 门(三条):
//! 1. `Δ = 0` ⇒ 池**逐字节不变**(恒等旋转;表恰为 (1, 0));
//! 2. 任意 Δ ⇒ 与 host 独立重实现一致(映射 oracle 在本文件独立写 ——
//!    与 `kv_knhd_write_parity` 同款:双实现互证比"引用同一份"更强)。
//!    **判据 = 配对量级 ≤2 ulp**:设备侧 `a*c − b*s` 被 nvcc 收缩成 FMA(默认
//!    `-fmad=true`)、host 是两次独立舍入 ⇒ 末位可差 1;抵销情形必须按配对
//!    尺度取 ulp(按输出取会爆到 500× —— 实测);
//! 3. `+Δ` 再 `−Δ` ⇒ 回到起点(配对量级 ≤8 ulp;池内其余页一字未动)。
//!
//! 布局 = `[nb, hkv, hd/x, page, x]`(`StatePool::k_shape_kv`,kNHD 分页);
//! 形状取 0.8B/27B 两档(hd 256 / x 8 / page 32 / half 32)。
//!
//! 无 GPU 跳过(`#[owl_test(gpu)]`);卡 = ordinal 1(3090 空闲卡纪律)。

use owl_macros::owl_test;
use std::sync::Arc;

use cudarc::driver::{CudaContext, DevicePtr, LaunchConfig, PushKernelArg};

const HKV: usize = 2;
const HD: usize = 256;
const PAGE: usize = 32;
const X: usize = 8;
const HALF: usize = 32; // rotary_dim 64 / 2
const NB: usize = 6; // 池页数(远大于被测页列表,便于验"别页未动")

fn half_bits(f: f32) -> u16 {
    half::f16::from_f32(f).to_bits()
}

fn h2f(bits: u16) -> f32 {
    f32::from(half::f16::from_bits(bits))
}

fn compile(ctx: &Arc<CudaContext>, src: &str, name: &str) -> cudarc::driver::CudaFunction {
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(
        src,
        cudarc::nvrtc::CompileOptions {
            arch: Some("compute_86"),
            include_paths: vec!["/usr/local/cuda/include".into()],
            ..Default::default()
        },
    )
    .unwrap_or_else(|e| panic!("nvrtc({name}): {e:?}"));
    let m = ctx.load_module(ptx).unwrap_or_else(|e| panic!("module({name}): {e:?}"));
    m.load_function(name).unwrap_or_else(|e| panic!("fn({name}): {e:?}"))
}

/// **独立映射 oracle**(勿改成引用 models 侧实现 —— 探针价值在双实现互证)
fn k_off(blk: usize, h: usize, d: usize) -> usize {
    (blk * HKV + h) * HD * PAGE + (d / X) * (PAGE * X) + (d % X)
}

/// **host 参考**:对 `blocks` 列出的页做 rotate-half(`Δ` 单行表广播)
fn host_shift(k: &mut [u16], cos1: &[f32], sin1: &[f32], blocks: &[u32]) {
    for &b in blocks {
        let b = b as usize;
        for h in 0..HKV {
            for t in 0..PAGE {
                for d in 0..HALF {
                    // 低半 d / 高半 d+HALF;两者在池里的 token 位置相同
                    let lo = k_off(b, h, d) + t * X;
                    let hi = k_off(b, h, d + HALF) + t * X;
                    let c = cos1[d];
                    let s = sin1[d];
                    let a = h2f(k[lo]);
                    let bb = h2f(k[hi]);
                    k[lo] = half_bits(a * c - bb * s);
                    k[hi] = half_bits(bb * c + a * s);
                }
            }
        }
    }
}

/// 一张表(单行 Δ):`cos(Δ·θ_j)`,θ_j = theta^(-2j/rotary_dim)
fn delta_row(delta: i64) -> (Vec<f32>, Vec<f32>) {
    let rotary_dim = 2 * HALF;
    let theta = 1e7f32;
    let inv: Vec<f32> = (0..HALF)
        .map(|j| theta.powf(-(2.0 * j as f32) / rotary_dim as f32))
        .collect();
    let c = inv.iter().map(|f| (*f * delta as f32).cos()).collect();
    let s = inv.iter().map(|f| (*f * delta as f32).sin()).collect();
    (c, s)
}

/// 确定性初值(±5 量级,含跨零;保证有可观测的抵销)
fn pool_init(n: usize) -> Vec<u16> {
    (0..n)
        .map(|i| half_bits((((i * 37 + 11) % 401) as f32) / 40.0 - 5.0))
        .collect()
}

/// **一次上传 → N 次发射 → 回读**(N 次同一 Δ;叠转门的执行面)。
///
/// 抽成模块级是为了让"单次对拍"与"叠转积累"共用同一条发射路径 ——
/// 两处各写一份核参就是两套规范。
#[allow(clippy::too_many_arguments)]
fn run_shift_n(
    stream: &Arc<cudarc::driver::CudaStream>,
    f: &cudarc::driver::CudaFunction,
    base: &[u16],
    pool_elems: usize,
    cos1: &[f32],
    sin1: &[f32],
    blocks: &[u32],
    rounds: usize,
) -> Vec<u16> {
    let mut d_k = unsafe { stream.alloc::<u16>(pool_elems) }.unwrap();
    stream.memcpy_htod(base, &mut d_k).unwrap();
    let mut d_c = unsafe { stream.alloc::<u16>(HALF) }.unwrap();
    let mut d_s = unsafe { stream.alloc::<u16>(HALF) }.unwrap();
    let cb: Vec<u16> = cos1.iter().map(|v| half_bits(*v)).collect();
    let sb: Vec<u16> = sin1.iter().map(|v| half_bits(*v)).collect();
    stream.memcpy_htod(&cb, &mut d_c).unwrap();
    stream.memcpy_htod(&sb, &mut d_s).unwrap();
    let mut d_b = unsafe { stream.alloc::<f32>(blocks.len()) }.unwrap();
    let bl: Vec<f32> = blocks.iter().map(|b| *b as f32).collect();
    stream.memcpy_htod(&bl, &mut d_b).unwrap();
    let (kp, cp, sp, bp) = (
        d_k.device_ptr(stream).0,
        d_c.device_ptr(stream).0,
        d_s.device_ptr(stream).0,
        d_b.device_ptr(stream).0,
    );
    let (hkv, hd, page, x, half) =
        ((HKV) as u64, (HD) as u64, (PAGE) as u64, (X) as u64, (HALF) as u64);
    let units = blocks.len() * HKV * (HALF / X);
    for _ in 0..rounds {
        let mut b = stream.launch_builder(f);
        b.arg(&cp)
            .arg(&sp)
            .arg(&bp)
            .arg(&hkv)
            .arg(&hd)
            .arg(&page)
            .arg(&x)
            .arg(&half)
            .arg(&kp);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (units as u32, 1, 1),
                block_dim: ((PAGE * X) as u32, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .unwrap();
    }
    stream.synchronize().unwrap();
    let mut out = vec![0u16; pool_elems];
    stream.memcpy_dtoh(&d_k, &mut out).unwrap();
    out
}

/// **叠转累积误差门**(`image_evict_delta_limit` 的事实依据,2026-10-23)。
///
/// 同一页连续 `N` 次 `R(−S)` 与一次 `R(−N·S)` 数学等价 ⇒ 差 = **纯 f16 累积
/// 舍入**。判据:配对量级 ≤ `N` ulp(每轮 ≤1 ulp 的上界);并打印实测系数,
/// 供"一条链允许被重转多少次"定值。实测参考见
/// `docs/devlog/memory/prefix-deletion.md` §5quinquies。
#[owl_test(gpu)]
fn kv_rope_k_shift_drift_under_repeats() {
    let ctx = CudaContext::new(1).expect("ctx");
    ctx.bind_to_thread().expect("bind");
    let stream = ctx.default_stream();
    let f = compile(
        &ctx,
        owl_kernels::cu_sig::cu_text(owl_kernels::ops::source("owl_kv_rope_k_shift_f16")).as_str(),
        "owl_kv_rope_k_shift_f16",
    );
    let pool_elems = NB * HKV * HD * PAGE;
    let base = pool_init(pool_elems);
    let blocks: [u32; 3] = [2, 0, 4];
    let s_step = 40i64; // 与引擎门同量级的平移
    let dg_per_half = HALF / X;
    for &rounds in &[4usize, 16, 32, 64] {
        let (c1, s1) = delta_row(s_step);
        let many = run_shift_n(&stream, &f, &base, pool_elems, &c1, &s1, &blocks, rounds);
        let (cn, sn) = delta_row(s_step * rounds as i64);
        let once = run_shift_n(&stream, &f, &base, pool_elems, &cn, &sn, &blocks, 1);
        let mut worst_ulp = 0f32;
        let mut bad = 0usize;
        for &b in &blocks {
            let b = b as usize;
            for h in 0..HKV {
                for dg in 0..dg_per_half {
                    for t in 0..PAGE {
                        for d in 0..X {
                            let lo = k_off(b, h, dg * X + d) + t * X;
                            let hi = k_off(b, h, dg * X + d + HALF) + t * X;
                            let scale = [many[lo], many[hi], once[lo], once[hi]]
                                .iter()
                                .map(|v| h2f(*v).abs())
                                .fold(0f32, f32::max)
                                .max(1e-3);
                            let ulp = 2f32.powi(-10) * scale;
                            for i in [lo, hi] {
                                let e = (h2f(many[i]) - h2f(once[i])).abs();
                                worst_ulp = worst_ulp.max(e / ulp);
                                if e > rounds as f32 * ulp {
                                    bad += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(bad, 0, "N={rounds}: 叠转误差超过 N ulp 上界({bad} 个元素)");
        eprintln!(
            "[drift] N={rounds:>3} · 最差 {:.2} ulp(= {:.2} ulp/轮)· 判据 ≤ {rounds} ulp",
            worst_ulp,
            worst_ulp / rounds as f32
        );
    }
}

#[owl_test(gpu)]
fn kv_rope_k_shift_matches_host() {
    let ctx = CudaContext::new(1).expect("ctx");
    ctx.bind_to_thread().expect("bind");
    let stream = ctx.default_stream();
    let f = compile(
        &ctx,
        owl_kernels::cu_sig::cu_text(owl_kernels::ops::source("owl_kv_rope_k_shift_f16")).as_str(),
        "owl_kv_rope_k_shift_f16",
    );
    let pool_elems = NB * HKV * HD * PAGE;
    let base = pool_init(pool_elems);
    // 页列表:跨页/非序(2,0,4)—— 逻辑序无关是契约的一部分
    let blocks: [u32; 3] = [2, 0, 4];

    // `run` = 一次完整"上传池 → 发射 → 回读"
    let run = |cos1: &[f32], sin1: &[f32], blocks: &[u32]| -> Vec<u16> {
        run_shift_n(&stream, &f, &base, pool_elems, cos1, sin1, blocks, 1)
    };

    // ① Δ = 0 ⇒ 逐字节空操作
    {
        let (c, s) = delta_row(0);
        let got = run(&c, &s, &blocks);
        assert_eq!(got, base, "Δ=0 必须逐字节不变(恒等旋转)");
    }

    // ② 任意 Δ ⇒ 与 host 逐位一致
    for &delta in &[1i64, 977, -333, 4096] {
        let (c, s) = delta_row(delta);
        let got = run(&c, &s, &blocks);
        let mut want = base.clone();
        host_shift(&mut want, &c, &s, &blocks);
        // **判据 = 配对量级的 ≤2 ulp**(不是逐位)。
        //
        // 两个来源,都不是核错:
        // ① nvcc 把 `a*c − b*s` **收缩成 FMA**(默认 `-fmad=true`),host 是两次独立
        //    舍入 ⇒ 末位差 1;
        // ② 抵销情形(`a*c ≈ b*s`)下输出相对**输入**极小 ⇒ 按输出取 ulp 会爆,
        //    必须按**配对量级**(该 (token, d/d+half) 对的四个值)取尺度。
        // `Δ=0` 仍是**逐字节**恒等(上一条门;`a·1 − b·0` 任何收缩下都精确)。
        let dg_per_half = HALF / X;
        let mut bad = 0usize;
        let mut worst = 0f32;
        for &b in &blocks {
            let b = b as usize;
            for h in 0..HKV {
                for dg in 0..dg_per_half {
                    for t in 0..PAGE {
                        for d in 0..X {
                            let lo = k_off(b, h, dg * X + d) + t * X;
                            let hi = k_off(b, h, dg * X + d + HALF) + t * X;
                            let scale = [got[lo], got[hi], want[lo], want[hi]]
                                .iter()
                                .map(|v| h2f(*v).abs())
                                .fold(0f32, f32::max)
                                .max(1e-3);
                            let tol = 2.0 * 2f32.powi(-10) * scale;
                            for i in [lo, hi] {
                                let e = (h2f(got[i]) - h2f(want[i])).abs();
                                worst = worst.max(e / tol);
                                if e > tol {
                                    bad += 1;
                                    if bad < 8 {
                                        eprintln!(
                                            "[diff] Δ={delta} idx={i} dev={:#06x} host={:#06x} ({:.2} ulp-pair)",
                                            got[i], want[i], e / tol
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(
            worst <= 1.0,
            "Δ={delta}: {bad} 个元素越 2 ulp(配对量级;最差 {worst:.2})—— 不是收缩误差,是真错值"
        );
        // 未列出页必须一字未动
        for p in 0..NB {
            if blocks.contains(&(p as u32)) {
                continue;
            }
            let lo = p * HKV * HD * PAGE;
            let hi = lo + HKV * HD * PAGE;
            assert_eq!(got[lo..hi], base[lo..hi], "页 {p} 不在 blocks 里却被改");
        }
    }

    // ③ +Δ 再 −Δ ⇒ 回到起点(逐位;f16 旋转对在配对量级内可逆)
    {
        let (cp, sp) = delta_row(977);
        let (cm, sm) = delta_row(-977);
        let mut d_k = unsafe { stream.alloc::<u16>(pool_elems) }.unwrap();
        stream.memcpy_htod(&base, &mut d_k).unwrap();
        let launch = |c: &[f32], s: &[f32]| {
            let cb: Vec<u16> = c.iter().map(|v| half_bits(*v)).collect();
            let sb: Vec<u16> = s.iter().map(|v| half_bits(*v)).collect();
            let mut d_c = unsafe { stream.alloc::<u16>(HALF) }.unwrap();
            let mut d_s = unsafe { stream.alloc::<u16>(HALF) }.unwrap();
            stream.memcpy_htod(&cb, &mut d_c).unwrap();
            stream.memcpy_htod(&sb, &mut d_s).unwrap();
            let mut d_b = unsafe { stream.alloc::<f32>(blocks.len()) }.unwrap();
            let bl: Vec<f32> = blocks.iter().map(|b| *b as f32).collect();
            stream.memcpy_htod(&bl, &mut d_b).unwrap();
            let (kp, cp2, sp2, bp) = (
                d_k.device_ptr(&stream).0,
                d_c.device_ptr(&stream).0,
                d_s.device_ptr(&stream).0,
                d_b.device_ptr(&stream).0,
            );
            let (hkv, hd, page, x, half) =
                ((HKV) as u64, (HD) as u64, (PAGE) as u64, (X) as u64, (HALF) as u64);
            let units = blocks.len() * HKV * (HALF / X);
            let mut b = stream.launch_builder(&f);
            b.arg(&cp2)
                .arg(&sp2)
                .arg(&bp)
                .arg(&hkv)
                .arg(&hd)
                .arg(&page)
                .arg(&x)
                .arg(&half)
                .arg(&kp);
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: (units as u32, 1, 1),
                    block_dim: ((PAGE * X) as u32, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .unwrap();
            stream.synchronize().unwrap();
        };
        launch(&cp, &sp);
        launch(&cm, &sm);
        let mut out = vec![0u16; pool_elems];
        stream.memcpy_dtoh(&d_k, &mut out).unwrap();
        // 只比对被测页(其余页不动是上一条门的事);尺度 = **配对量级**
        // (f16 旋转对在大角度下会抵销 ⇒ 小值元素的相对误差可以很大,
        //  但绝对误差仍受配对 |a|,|b| 的 ulp 支配 —— f16 存储的固有代价)
        let dg_per_half = HALF / X;
        let mut worst = 0f32;
        for &b in &blocks {
            let b = b as usize;
            for h in 0..HKV {
                for dg in 0..dg_per_half {
                    for t in 0..PAGE {
                        for d in 0..X {
                            let lo = k_off(b, h, dg * X + d) + t * X;
                            let hi = k_off(b, h, dg * X + d + HALF) + t * X;
                            let scale = [out[lo], out[hi], base[lo], base[hi]]
                                .iter()
                                .map(|v| h2f(*v).abs())
                                .fold(0f32, f32::max)
                                .max(1e-3);
                            let tol = 8.0 * 2f32.powi(-10) * scale;
                            for i in [lo, hi] {
                                worst = worst.max((h2f(out[i]) - h2f(base[i])).abs() / tol);
                            }
                        }
                    }
                }
            }
        }
        assert!(worst <= 1.0, "R(Δ)∘R(−Δ) 往返 {worst:.3}× 超 8 ulp(配对量级)");
    }
}

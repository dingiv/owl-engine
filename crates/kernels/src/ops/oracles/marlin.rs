//! `marlin_gemm_w4a16_awq` 装配期自检(**生产发射量第一**:spec 档 2822 发/轮)。
//!
//! 语义(kU4 非对称 AWQ):`out[m, n] = Σ_k a[m,k]·(q[n,k] − zp[n, k/g])·s[n, k/g]`,
//! f16 入出、f32 累加;权重以 marlin 交错布局打包(`pack_marlin_b/s/z`)。
//!
//! **参考复用自家 packer**(布局知识只住 packer),参考数学 = 原始 q/s/zp 上的
//! 显式反量化(与 `marlin_parity.rs::host_dequant_gemm_awq` 同一套独立实现 ——
//! 该实现曾与 vLLM golden 对拍过)。
//!
//! 门控:oracle 住 `#[cfg(feature = "marlin")]`(要用 packer),`SELFCHECK`
//! 在 `ops/marlin.rs` 用一对互斥 cfg 常量挂 —— 表(零门)与 oracle(门内)
//! 的接触点只有这一个。

use crate::contract::{Arg, KernelSource, LaunchMsg, OpError};
use crate::family::marlin::{c_tmp_float_len_v2, repack, v2_workspace_len};
use crate::bootcheck::{BootCheck, CheckReport, CheckRunner};

use super::{f16_bits, f16_to_f32};

// 自检几何(**必须满足 marlin 打包约束** `k%128==0 && n%256==0` ——
// 2026-10-13 实测:小形状会被 packer 直接 assert 掉):
// m=8(W4A8 m8 形状;m8 tile 由 `[kernel] marlin_w4a8_m8 = true` 显式开)、n=256、k=128、g=32 ⇒ 4 组
const M: usize = 8;
const N: usize = 256;
/// ⚠️ **K 的用途不只是形状**:内核的 `slice_count = prob_k / thread_k`
/// (k 分片)× 多 warp 归约路径 —— `thread_k ∈ {64,128}` ⇒ **K=128 时
/// slice_count=1,归约路径根本没被跑到**(2026-10-16 实测定谳:先前的
/// oracle 几何恰好避开了整条 `global_reduce_fp32` 路;a16 生产臂同病)。
/// K=512 ⇒ slice_count = 4…8 ⇒ 归约路径进自检。
/// (成本:参考实现 4×256×512 = 524k MAC —— 仍远低于任何可感量级。)
const K: usize = 512;
const G: usize = 32;
const TOL: f32 = 2e-2; // 相对(噪声/信号 RMS 口径同 parity 门)

pub struct Case {
    /// [m, k] f16 位型(激活)
    pub a: Vec<u16>,
    /// [n, k] 的 int4 值(0..15;逐元素一字节)
    pub q: Vec<u8>,
    /// [n, groups] 尺度
    pub s: Vec<f32>,
    /// [n, groups] 零偏(0..15)
    pub zp: Vec<u8>,
}

pub fn case() -> Case {
    let groups = K / G;
    // 确定性伪随机(与 parity 用例同族口径;幅值小 ⇒ f16 域信噪比可判)
    let a_f: Vec<f32> = (0..M * K).map(|i| ((i as f32 * 0.31) - 4.0).sin() * 0.5).collect();
    Case {
        a: a_f.iter().map(|f| f16_bits(*f)).collect(),
        q: (0..N * K).map(|i| ((i * 7 + 3) % 16) as u8).collect(),
        s: (0..N * groups).map(|i| 0.02 + ((i * 13) % 13) as f32 * 0.004).collect(),
        zp: (0..N * groups).map(|i| ((i * 5 + 2) % 16) as u8).collect(),
    }
}

/// **独立 host 参考**(显式反量化 + f32 累加)。
///
/// `symmetric: true` = **故意做错**(忽略零偏,按 zp≡8 对称域算)——
/// 敏感性负例用;`transposed: true` = 把权重当 [k, n] 读(布局错)。
pub fn reference(c: &Case, symmetric: bool, transposed: bool) -> Vec<f32> {
    let groups = K / G;
    let mut out = vec![0f32; M * N];
    for r in 0..M {
        for nn in 0..N {
            let mut acc = 0f32;
            for kk in 0..K {
                let (qi, si, zi) = if transposed {
                    // 故意错:按 [k, n] 读
                    (kk * N + nn, (kk / G) * N + nn, (kk / G) * N + nn)
                } else {
                    (nn * K + kk, nn * groups + kk / G, nn * groups + kk / G)
                };
                let q = c.q[qi % c.q.len()] as f32;
                let z = if symmetric { 8.0 } else { c.zp[zi % c.zp.len()] as f32 };
                let s = c.s[si % c.s.len()];
                acc += f16_to_f32(c.a[r * K + kk]) * (q - z) * s;
            }
            out[r * N + nn] = acc;
        }
    }
    out
}

/// 打包并发射(经注册表 dispatch),返回原始输出(f16 位型)
pub fn run_case(r: &mut dyn CheckRunner, c: &Case) -> Result<Vec<u16>, OpError> {
    let b_packed = repack::pack_marlin_b(&c.q, K, N);
    let s_packed = repack::pack_marlin_s(&c.s, N, K / G);
    let z_packed = repack::pack_marlin_z(&c.zp, N, K / G);
    let ws_words = v2_workspace_len(N).max(N / 128 * 16);
    // c_tmp:parity 用例取 1 word(内核只把它当 FFI 透传/规约暂存)
    let c_tmp_words = c_tmp_float_len_v2(M, 1).max(1);

    let da = r.alloc_block(M * K * 2, "selfcheck.marlin.a")?;
    let db = r.alloc_block(b_packed.len() * 4, "selfcheck.marlin.b")?;
    let dout = r.alloc_block(M * N * 2, "selfcheck.marlin.out")?;
    let ds = r.alloc_block(s_packed.len() * 2, "selfcheck.marlin.s")?;
    let dz = r.alloc_block(z_packed.len() * 4, "selfcheck.marlin.z")?;
    let dws = r.alloc_block(ws_words * 4, "selfcheck.marlin.ws")?;
    let dct = r.alloc_block(c_tmp_words * 4, "selfcheck.marlin.ctmp")?;
    let le16 = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let le32 = |v: &[i32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    r.upload_block(&da, &le16(&c.a))?;
    r.upload_block(&db, &le32(&b_packed))?;
    r.upload_block(&ds, &le16(&s_packed))?;
    r.upload_block(&dz, &le32(&z_packed))?;
    // ⚠️ **workspace 必须零初始化**(线契约):`slice_count>1` 时内核用
    // workspace 里的 locks 做跨块同步 —— 喂 cudaMalloc 垃圾 = 自旋死等
    // (实测:GPU 100% util 但功耗只有 ~1/3,低功耗自旋的典型相;
    // K=128 时 slice_count=1 走不到这一段,故长期未暴露)。
    r.upload_block(&dws, &vec![0u8; ws_words * 4])?;

    // 线槽序(AWQ 臂;权威 = ops::marlin 表 / client::marlin::SIG_AWQ):
    // [a, b, out(O), scales, zeros, ws, c_tmp, m, k, n, groupsize]
    let msg = LaunchMsg {
        kernel: KernelSource {
            name: crate::family::marlin::GEMM_W4A16_AWQ.to_string(),
            source: String::new(),
        },
        args: vec![
            Arg::block(da.id, da.dtype, da.rank),
            Arg::block(db.id, db.dtype, db.rank),
            Arg::block(dout.id, dout.dtype, dout.rank),
            Arg::block(ds.id, ds.dtype, ds.rank),
            Arg::block(dz.id, dz.dtype, dz.rank),
            Arg::block(dws.id, dws.dtype, dws.rank),
            Arg::block(dct.id, dct.dtype, dct.rank),
            Arg::U64(M as u64),
            Arg::U64(K as u64),
            Arg::U64(N as u64),
            Arg::U64(G as u64),
        ],
        grid: (0, 0, 0),
        block: (0, 0, 0),
        shared_mem: 0,
        out_elems: M * N,
    };
    r.dispatch(msg)?;
    r.sync()?;
    let mut raw = vec![0u8; M * N * 2];
    r.download_block(&dout, &mut raw)?;
    Ok(raw.chunks_exact(2).map(|x| u16::from_le_bytes([x[0], x[1]])).collect())
}

/// (噪声/信号 RMS, max|err|, 首个越界)——与 parity 门同口径。
///
/// `floor_frac` = 每元素容差的**绝对地板**(以信号 RMS 为单位):
/// 容差 = `TOL·max(|wantᵢ|, floor_frac·rms)`;`floor_frac <= 0` = 旧行为
/// (地板 = 1e-3 常量)。
///
/// 为何需要地板:输出是 f16,且某些算子(尤其 W4A8 的 int16 尺度路径)
/// 的误差是**绝对**量级的(≈ Σ|项|·2⁻¹³)—— 近零输出上相对误差天然放大,
/// 只看 `|wantᵢ|` 会报假红(2026-10-16 W4A8 首跑实测)。
pub fn noise_metrics_floor(
    got: &[u16],
    want: &[f32],
    floor_frac: f32,
) -> (f64, f32, Option<(usize, f32, f32)>) {
    let n = got.len().min(want.len());
    if n == 0 {
        return (0.0, 0.0, None);
    }
    let sig: f64 = want.iter().take(n).map(|w| (*w as f64) * (*w as f64)).sum();
    let rms = (sig / n as f64).sqrt().max(1e-6);
    let floor = if floor_frac > 0.0 { (rms as f32) * floor_frac } else { 1e-3 };
    let (mut sum_abs, mut max_abs) = (0f64, 0f32);
    let mut first = None;
    for i in 0..n {
        let g = f16_to_f32(got[i]);
        let d = (g - want[i]).abs();
        sum_abs += d as f64;
        max_abs = max_abs.max(d);
        if first.is_none() && d > TOL * want[i].abs().max(floor) {
            first = Some((i, g, want[i]));
        }
    }
    (sum_abs / n as f64 / rms, max_abs, first)
}


pub struct MarlinAwqOracle;

impl BootCheck for MarlinAwqOracle {
    fn name(&self) -> &'static str {
        "host 显式反量化参考(q−zp)·s + 自家 packer(经注册表 dispatch)"
    }

    fn run(&self, r: &mut dyn CheckRunner) -> Result<CheckReport, OpError> {
        let c = case();
        let got = run_case(r, &c)?;
        let want = reference(&c, false, false);
        let (noise, max_abs, first_bad) = noise_metrics_floor(&got, &want, TOL_FLOOR);
        if let Some((i, g, w)) = first_bad {
            return Err(OpError::Contract {
                op: crate::family::marlin::GEMM_W4A16_AWQ.into(),
                field: "out",
                expect: format!(
                    "逐元素相对偏差 ≤ {TOL:.1e}(地板 = {TOL_FLOOR}×信号 RMS;host 显式反量化参考)"
                ),
                got: format!(
                    "首越界 idx={i} gpu={g} host={w}(噪声/信号 RMS={noise:.4}, max|err|={max_abs:.4})"
                ),
            });
        }
        if noise > TOL as f64 {
            return Err(OpError::Contract {
                op: crate::family::marlin::GEMM_W4A16_AWQ.into(),
                field: "out",
                expect: format!("噪声/信号 RMS ≤ {TOL:.1e}"),
                got: format!("{noise:.4}(max|err|={max_abs:.4})"),
            });
        }
        Ok(CheckReport::numeric(crate::family::marlin::GEMM_W4A16_AWQ, max_abs, TOL))
    }
}

// ============================================================================
// W4A8 × AWQ(§S1d 对拍;2026-10-16)
// ============================================================================
//
// 语义(int8 激活 × kU4 零偏):`out[m,n] = a_s[m] · Σ_k a_i8[m,k]·(q[n,k]−zp[n,k/g])·s[n,k/g]`。
//
// 与 a16 臂的**三处布局差异**(全部要在本自检里被验到,否则等于没接):
// 1. `B` 走 `pack_marlin_b_a8`(tile 32×32,与 a16 的 16×64+1024-PIPE 不同);
// 2. `scales` 走 `pack_marlin_s_a8`(perm_single + **int16 归一** + 交出全局因子);
// 3. `zp` 走 `pack_marlin_z_a8`(**无 n-interleave**)。
//
// 关键不变量:归一化(÷max×4096)与全局因子(max/4096)必须**相互抵消**
// ⇒ 本自检的 host 参考用**原始尺度**,不模仿内核的整数中间态 —— 因子没折进去
// 或折反了,输出会差 4096/max 倍,当场红。

/// W4A8 口径的噪声阈值(设计卷 §6 S1d 的判据线:`mean_rel < 2e-3`)
pub const TOL_A8: f32 = 2e-3;
/// **每元素容差的绝对地板**(以信号 RMS 为单位;见 [`noise_metrics_floor`])。
///
/// 两个臂都需要它,且理由不同:
/// - a16(AWQ):`slice_count>1` 时部分和经 **f16** 写穿 C 再相加(上游设计),
///   近零输出上绝对误差被放大(实测 K=512 时 d=4.2e-4 vs |want|=0.0155);
/// - a8:f32 部分和(c_tmp 自管)但 int16 尺度路径本身是绝对量级误差。
pub const TOL_FLOOR: f32 = 0.5;

/// W4A8 用例(int8 激活 + per-token 尺度;几何与 AWQ 用例同承)
///
/// 数据命意:(q−zp) ∈ [1,15]、a ∈ [20,119] 全**同号** —— 输出无相消,
/// 相对口径才有意义。
///
/// ⚠️**非退化律**(两次踩坑定下):zp 必须同时依赖 **组内位置(n%8)** 与
/// **组序号(n/8)** —— a8↔a16 的 zp 布局差 = “8-列组内置换 + 跨组错位”
/// (探针实测:注入列 37 在 a16 布局下被内核读作 21,而 `37%8 == 21%8`)
/// ⇒ 只依赖 `n%8` 的数据让该差异**逐字节隐形**(2026-10-16 首版假绿)。
/// scales 同理(perm 轴周期必须 > 32)。
pub struct CaseA8 {
    /// [m, k] int8 激活
    pub a: Vec<i8>,
    /// [m] per-token 激活尺度(逐行互异 ⇒ 行映射错也拦得住)
    pub a_scale: Vec<f32>,
    /// [n, k] 的 int4 值(0..15;逐元素一字节)
    pub q: Vec<u8>,
    /// [n, groups] 真尺度
    pub s: Vec<f32>,
    /// [n, groups] 零偏(0..15)
    pub zp: Vec<u8>,
}

/// 负例开关(证明判据能红;2026-10-13 P5 同律)
#[derive(Clone, Copy, Default)]
pub struct A8Neg {
    /// **不折全局因子**(a_scales 只给 per-token 尺度)—— 内核 int16 尺度路径
    /// 与调用方因子的配合关系被打破
    pub no_factor: bool,
    /// **zp 全零**(忽略零偏;本用例 zp∈[0,7]、q∈[8,15] ⇒ 必差)
    pub zp_zero: bool,
    /// **zp 用 a16 布局**(带 n-interleave)—— 真门(纠正过一次假绿:见
    /// [`case_a8`] 的非退化律)
    pub zp_a16_layout: bool,
    /// **B 用 a16 布局**(tile 16×64 + 1024-PIPE)而非 a8 的 32×32
    pub b_a16_layout: bool,
}

pub fn case_a8() -> CaseA8 {
    let groups = K / G;
    let mut q = vec![0u8; N * K];
    let mut s = vec![0f32; N * groups];
    let mut zp = vec![0u8; N * groups];
    for nn in 0..N {
        for kk in 0..K {
            q[nn * K + kk] = ((kk * 7 + nn * 3) % 8 + 8) as u8;
        }
        for g in 0..groups {
            s[nn * groups + g] = 0.02 + ((nn * 37 + g * 11) % 53) as f32 * 0.0008;
            // 组内位置 + 组序号 + 组号 三者都进 ⇒ 任何“组内置换/跨组错位”都现形
            zp[nn * groups + g] = (((nn % 8) + ((nn / 8) % 8) + g) % 8) as u8;
        }
    }
    CaseA8 {
        a: (0..M * K).map(|i| ((i * 13 + 7) % 100 + 20) as i8).collect(),
        a_scale: (0..M).map(|m| 0.02 + m as f32 * 0.0031).collect(),
        q,
        s,
        zp,
    }
}

/// **独立 host 参考**(显式反量化 + f32 累加;用**原始**尺度 —— 不看内核中间态)
pub fn reference_a8(c: &CaseA8) -> Vec<f32> {
    let groups = K / G;
    let mut out = vec![0f32; M * N];
    for r in 0..M {
        for nn in 0..N {
            let mut acc = 0f32;
            for kk in 0..K {
                let q = c.q[nn * K + kk] as f32;
                let z = c.zp[nn * groups + kk / G] as f32;
                acc += c.a[r * K + kk] as f32 * (q - z) * c.s[nn * groups + kk / G];
            }
            out[r * N + nn] = acc * c.a_scale[r];
        }
    }
    out
}

/// 打包并发射(经注册表 dispatch → `MarlinRuntime::run_w4a8_awq`),返回原始输出
pub fn run_case_a8(r: &mut dyn CheckRunner, c: &CaseA8, neg: A8Neg) -> Result<Vec<u16>, OpError> {
    run_raw(
        r,
        c,
        &c.zp,
        !neg.zp_a16_layout,
        !neg.b_a16_layout,
        !neg.no_factor,
        neg.zp_zero,
    )
}

/// **细粒度发射口**(判别探针用):z_data = 实际喂入的 zp 值;
/// `z_a8` = zp 用 a8 布局(否则 a16);`b_a8` = B 用 a8 布局;
/// `fold` = a_scales 折全局因子;`zp_zero` 优先于 z_data(全零)。
pub fn run_raw(
    r: &mut dyn CheckRunner,
    c: &CaseA8,
    z_data: &[u8],
    z_a8: bool,
    b_a8: bool,
    fold: bool,
    zp_zero: bool,
) -> Result<Vec<u16>, OpError> {
    let groups = K / G;
    let b_packed = if b_a8 {
        repack::pack_marlin_b_a8(&c.q, K, N)
    } else {
        repack::pack_marlin_b(&c.q, K, N)
    };
    let (s_bits, factor) = repack::pack_marlin_s_a8(&c.s, N, groups);
    let z_packed = if zp_zero {
        vec![0i32; groups * (N / 8)]
    } else if z_a8 {
        repack::pack_marlin_z_a8(z_data, N, groups)
    } else {
        repack::pack_marlin_z(z_data, N, groups)
    };
    let ws_words = v2_workspace_len(N).max(N / 128 * 16);
    // c_tmp 槽位要有实体(线格式契约),但 W4A8 内核**不读它**(host 自管)
    let c_tmp_words = c_tmp_float_len_v2(M, 1).max(1);
    // 全局因子折进 a_scales(装载期契约;负例 no_factor 故意不折)
    let a_scales: Vec<f32> = c
        .a_scale
        .iter()
        .map(|s| if fold { *s * factor } else { *s })
        .collect();

    let da = r.alloc_block(M * K, "selfcheck.marlin.a8.a")?;
    let das = r.alloc_block(M * 4, "selfcheck.marlin.a8.a_s")?;
    let db = r.alloc_block(b_packed.len() * 4, "selfcheck.marlin.a8.b")?;
    let dout = r.alloc_block(M * N * 2, "selfcheck.marlin.a8.out")?;
    let ds = r.alloc_block(s_bits.len() * 2, "selfcheck.marlin.a8.s")?;
    let dz = r.alloc_block(z_packed.len() * 4, "selfcheck.marlin.a8.z")?;
    let dws = r.alloc_block(ws_words * 4, "selfcheck.marlin.a8.ws")?;
    let dct = r.alloc_block(c_tmp_words * 4, "selfcheck.marlin.a8.ctmp")?;
    let le16 = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let le32 = |v: &[i32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    r.upload_block(&da, &c.a.iter().map(|v| *v as u8).collect::<Vec<u8>>())?;
    r.upload_block(&das, &a_scales.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
    r.upload_block(&db, &le32(&b_packed))?;
    r.upload_block(&ds, &le16(&s_bits))?;
    r.upload_block(&dz, &le32(&z_packed))?;
    // workspace 零初始化(同 a16 臂注:reduce 路径的 locks 依赖它)
    r.upload_block(&dws, &vec![0u8; ws_words * 4])?;

    // 线槽序(权威 = ops::marlin 表 / client::marlin::PARSE_SLOTS_W4A8):
    // [a, b, out(O), scales, zeros, a_scales, ws, c_tmp, m, k, n, groupsize]
    let msg = LaunchMsg {
        kernel: KernelSource {
            name: crate::family::marlin::GEMM_W4A8_AWQ.to_string(),
            source: String::new(),
        },
        args: vec![
            Arg::block(da.id, da.dtype, da.rank),
            Arg::block(db.id, db.dtype, db.rank),
            Arg::block(dout.id, dout.dtype, dout.rank),
            Arg::block(ds.id, ds.dtype, ds.rank),
            Arg::block(dz.id, dz.dtype, dz.rank),
            Arg::block(das.id, das.dtype, das.rank),
            Arg::block(dws.id, dws.dtype, dws.rank),
            Arg::block(dct.id, dct.dtype, dct.rank),
            Arg::U64(M as u64),
            Arg::U64(K as u64),
            Arg::U64(N as u64),
            Arg::U64(G as u64),
        ],
        grid: (0, 0, 0),
        block: (0, 0, 0),
        shared_mem: 0,
        out_elems: M * N,
    };
    r.dispatch(msg)?;
    r.sync()?;
    let mut raw = vec![0u8; M * N * 2];
    r.download_block(&dout, &mut raw)?;
    Ok(raw.chunks_exact(2).map(|x| u16::from_le_bytes([x[0], x[1]])).collect())
}

/// **zp 布局判别探针**(2026-10-16):zp 只在一列置值 ⇒ 输出差落在哪一列
/// 就是“内核把该 nibble 当成哪一列”。返回各列的平均 |Δ|(f16→f32)。
///
/// 用途:直接回答“a8 与 a16 的 zp 布局到底差在哪里 / 内核按哪张表读”
/// —— 不再靠读模板反推。
pub fn zp_column_response(
    r: &mut dyn CheckRunner,
    c: &CaseA8,
    n0: usize,
    g0: usize,
    z_a8: bool,
) -> Result<Vec<f32>, OpError> {
    let groups = K / G;
    let base = run_raw(r, c, &c.zp, z_a8, true, true, true)?; // zp 全零基底
    let mut z = vec![0u8; N * groups];
    z[n0 * groups + g0] = 8;
    let hit = run_raw(r, c, &z, z_a8, true, true, false)?;
    let mut resp = vec![0f32; N];
    for mm in 0..M {
        for nn in 0..N {
            let d = (f16_to_f32(hit[mm * N + nn]) - f16_to_f32(base[mm * N + nn])).abs();
            resp[nn] += d / M as f32;
        }
    }
    Ok(resp)
}

/// W4A8 × AWQ 装配期自检(a8 三件套 + int8 激活路;判据 = 设计卷 §6 S1d)
pub struct MarlinW4a8AwqOracle;

impl BootCheck for MarlinW4a8AwqOracle {
    fn name(&self) -> &'static str {
        "host 显式反量化参考 a_s·(q−zp)·s + a8 三件套 packer(经注册表 dispatch)"
    }

    fn run(&self, r: &mut dyn CheckRunner) -> Result<CheckReport, OpError> {
        let c = case_a8();
        let got = run_case_a8(r, &c, A8Neg::default())?;
        let want = reference_a8(&c);
        let (noise, max_abs, first_bad) = noise_metrics_floor(&got, &want, TOL_FLOOR);
        if let Some((i, g, w)) = first_bad {
            return Err(OpError::Contract {
                op: crate::family::marlin::GEMM_W4A8_AWQ.into(),
                field: "out",
                expect: format!(
                    "逐元素相对偏差 ≤ {TOL_A8:.1e}(地板 = {TOL_FLOOR}×信号 RMS;host 显式反量化参考)"
                ),
                got: format!(
                    "首越界 idx={i} gpu={g} host={w}(噪声/信号 RMS={noise:.4}, max|err|={max_abs:.4})"
                ),
            });
        }
        if noise > TOL_A8 as f64 {
            return Err(OpError::Contract {
                op: crate::family::marlin::GEMM_W4A8_AWQ.into(),
                field: "out",
                expect: format!("噪声/信号 RMS ≤ {TOL_A8:.1e}"),
                got: format!("{noise:.4}(max|err|={max_abs:.4})"),
            });
        }
        Ok(CheckReport::numeric(
            crate::family::marlin::GEMM_W4A8_AWQ,
            max_abs,
            TOL_A8,
        ))
    }
}

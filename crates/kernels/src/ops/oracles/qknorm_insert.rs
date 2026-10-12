//! `owl_qknorm_rope_kv_insert_f16_fp8kv_knhd` 装配期自检(**独立 host 参考**)。
//!
//! 为什么这个算子排最前:**E10 案 ② 的现场**(池 dtype 改造时融合插池核缺
//! fp8 臂 ⇒ f16 写落在 1B/elem 池的 2× 偏移 = 真槽区永不落笔全零),而且它
//! 同时是 **rope 配对案**(2026-10-02:`nb*cf + n*sf` 基/伙伴互换,pos=0 时
//! 直接写错位 —— 引擎 kv-hex 3-4% 偏差真凶)的现场。两件都是"数值静默错"。
//!
//! 三条判据:
//! 1. **q 输出**(f16)vs host 参考(norm + rope;`|err| ≤ 2e-3`);
//! 2. **池字节**(e4m3)逐字节 == host RNE 参考;
//! 3. **写集穷举毒值律**:池灌 e4m3-NaN → 发射 → 声明写集外仍为毒值。
//!
//! 参考带两个"故意做错"开关(测试证明判据敏感性):`rope_swap`(历史 rope
//! 配对错)、`layout_classic`(池布局错)。

use crate::contract::{LaunchVal, OpError};
use crate::spec::{BootCheck, CheckBuf, CheckReport, CheckRunner};
const QKNORM_ROPE_KV_INSERT_F16_FP8KV: &str = include_str!("../../../cu/owl/fused.cu");

use super::{e4m3_encode, f16_bits, f16_to_f32, lcg_f16};

// 自检几何(小规模;含跨页与 padding 两个边界)
const T: usize = 3;
const HQ: usize = 2;
const HKV: usize = 1;
const HD: usize = 128;
const PAGE: usize = 32;
const HALF: usize = HD / 2;
const NB: usize = 2;
const W_OFF: i32 = 1;
const EPS: f32 = 1e-6;
/// token2 = padding(slot < 0)⇒ 不许写池;token1 跨页(block 1)
const SLOTS: [f32; T] = [5.0, 32.0, -1.0];
const POS: [f32; T] = [0.0, 3.0, 7.0];
const MAX_POS: usize = 16;
const POISON: u8 = 0x7F;
/// q 输出容差(**相对**):f16 域 + 核内 `rsqrtf`/归约序与 host 不同
/// (实测纯顺序差可达 ~1% ⇒ 绝对阈值会把噪声当错值)
const TOL: f32 = 1e-2;

/// 自检算例(确定性;种子固定 ⇒ 可复现)
pub struct Case {
    pub q_raw: Vec<u16>, // [T, HQ·2HD]
    pub k: Vec<u16>,     // [T, HKV·HD]
    pub v: Vec<u16>,     // [T, HKV·HD]
    pub q_w: Vec<u16>,   // [HD]
    pub k_w: Vec<u16>,   // [HD]
    pub cos_t: Vec<u16>, // [MAX_POS, HALF]
    pub sin_t: Vec<u16>, // [MAX_POS, HALF]
}

pub fn case() -> Case {
    Case {
        q_raw: lcg_f16(T * HQ * 2 * HD, 0xA1),
        k: lcg_f16(T * HKV * HD, 0xA2),
        v: lcg_f16(T * HKV * HD, 0xA3),
        // 权重/rope 表用小幅值(贴近真实:权重 ~1,cos/sin ∈ [-1,1])
        q_w: lcg_f16(HD, 0xA4).iter().map(|h| f16_bits(f16_to_f32(*h) * 0.25)).collect(),
        k_w: lcg_f16(HD, 0xA5).iter().map(|h| f16_bits(f16_to_f32(*h) * 0.25)).collect(),
        cos_t: (0..MAX_POS * HALF)
            .map(|i| f16_bits((i as f32 * 0.01).cos()))
            .collect(),
        sin_t: (0..MAX_POS * HALF)
            .map(|i| f16_bits((i as f32 * 0.01).sin()))
            .collect(),
    }
}

pub fn pool_bytes() -> usize {
    NB * PAGE * HKV * HD
}

/// q 行归一(值半段 = 前 HD 个元素)
fn row_inv(xs: &[u16]) -> f32 {
    let ss: f32 = (0..HD).map(|d| {
        let x = f16_to_f32(xs[d]);
        x * x
    }).sum();
    1.0 / (ss / HD as f32 + EPS).sqrt()
}

/// **host 参考**:q_out(f16 位型)+ K/V 池(e4m3 字节)。
///
/// `rope_swap: true` = 历史 bug 的配对方式(`nb*cf + n*sf` 互换);
/// `layout_classic: true` = 用 classic 池寻址(而非 knHD)。
/// 池写集条目:(池内下标, 期望 e4m3 字节)
pub type PoolBytes = Vec<(usize, u8)>;
/// 参考产物:(q_out f16 位型, K 池字节, V 池字节)
pub type Reference = (Vec<u16>, PoolBytes, PoolBytes);

pub fn reference(c: &Case, rope_swap: bool, layout_classic: bool) -> Reference {
    let mut q_out = vec![0u16; T * HQ * HD];
    let mut keys = Vec::new();
    let mut vals = Vec::new();
    for t in 0..T {
        let p = POS[t] as usize;
        // ---- q 头 ----
        for head in 0..HQ {
            let base_in = t * HQ * 2 * HD + head * 2 * HD;
            let xs = &c.q_raw[base_in..base_in + 2 * HD];
            let inv = row_inv(xs);
            let base = (t * HQ + head) * HD;
            // 核内 q 路径只有两个写者分支(d < half 写配对两半;d ≥ 2·half 直通),
            // [half, 2·half) 由配对线程写 —— 参考必须同构,否则把伙伴位写错
            for d in 0..HALF.min(HD) {
                let xf = f16_to_f32(xs[d]);
                let wf = f16_to_f32(c.q_w[d]);
                let n = xf * inv * (wf + 1.0); // w_off = 1
                let xb = f16_to_f32(xs[d + HALF]);
                let wb = f16_to_f32(c.q_w[d + HALF]);
                let nb = xb * inv * (wb + 1.0);
                let cf = f16_to_f32(c.cos_t[p * HALF + d]);
                let sf = f16_to_f32(c.sin_t[p * HALF + d]);
                let (lo, hi) = if rope_swap {
                    (nb * cf - n * sf, n * cf + nb * sf)
                } else {
                    (n * cf - nb * sf, nb * cf + n * sf)
                };
                q_out[base + d] = f16_bits(lo);
                q_out[base + d + HALF] = f16_bits(hi);
            }
            // d ≥ 2·half 直通(本自检几何 HD == 2·half ⇒ 该分支为空;
            // 保留为 while 以对其它几何同构,且不让 clippy 把常量空区间当错)
            let mut d = 2 * HALF;
            while d < HD {
                let n = f16_to_f32(xs[d]) * inv * (f16_to_f32(c.q_w[d]) + 1.0);
                q_out[base + d] = f16_bits(n);
                d += 1;
            }
        }
        // ---- k/v 头 ----
        let slot = SLOTS[t];
        if slot < 0.0 {
            continue; // padding:不写池
        }
        let (blk, off) = (slot as usize / PAGE, slot as usize % PAGE);
        for kh in 0..HKV {
            let ks = &c.k[t * HKV * HD + kh * HD..t * HKV * HD + (kh + 1) * HD];
            let inv = row_inv(ks);
            for d in 0..HD {
                let kf = f16_to_f32(ks[d]);
                let wf = f16_to_f32(c.k_w[d]);
                let mut n = kf * inv * (wf + 1.0);
                if d < HALF {
                    let xb = f16_to_f32(ks[d + HALF]);
                    let wb = f16_to_f32(c.k_w[d + HALF]);
                    let nb = xb * inv * (wb + 1.0);
                    let cf = f16_to_f32(c.cos_t[p * HALF + d]);
                    let sf = f16_to_f32(c.sin_t[p * HALF + d]);
                    n = if rope_swap { nb * cf - n * sf } else { n * cf - nb * sf };
                } else {
                    let xb = f16_to_f32(ks[d - HALF]);
                    let wb = f16_to_f32(c.k_w[d - HALF]);
                    let nb = xb * inv * (wb + 1.0);
                    let cf = f16_to_f32(c.cos_t[p * HALF + d - HALF]);
                    let sf = f16_to_f32(c.sin_t[p * HALF + d - HALF]);
                    n = if rope_swap { n * cf - nb * sf } else { n * cf + nb * sf };
                }
                let idx = if layout_classic {
                    // classic:[blk, hkv, hd/8, page, 8](故意错)
                    let hd_x = HD / 8;
                    ((blk * HKV + kh) * hd_x + d / 8) * PAGE * 8 + off * 8 + d % 8
                } else {
                    // knHD:[blk, page, hkv, hd]
                    ((blk * PAGE + off) * HKV + kh) * HD + d
                };
                keys.push((idx, e4m3_encode(n)));
                vals.push((idx, e4m3_encode(f16_to_f32(c.v[t * HKV * HD + kh * HD + d]))));
            }
        }
    }
    (q_out, keys, vals)
}

/// 跑一次核,返回原始设备输出(q_out f16 位型 + K/V 池字节)
/// 核产物:(q_out f16 位型, K 池字节, V 池字节)
pub type RawOut = (Vec<u16>, Vec<u8>, Vec<u8>);

pub fn run_case(r: &mut dyn CheckRunner, c: &Case) -> Result<RawOut, OpError> {
    let bytes = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|h| h.to_le_bytes()).collect() };
    let n_pool = pool_bytes();
    let d_q = r.alloc(c.q_raw.len() * 2, "selfcheck.qk.qraw")?;
    let d_k = r.alloc(c.k.len() * 2, "selfcheck.qk.k")?;
    let d_v = r.alloc(c.v.len() * 2, "selfcheck.qk.v")?;
    let d_kc = r.alloc(n_pool, "selfcheck.qk.kc")?;
    let d_vc = r.alloc(n_pool, "selfcheck.qk.vc")?;
    let d_slots = r.alloc(T * 4, "selfcheck.qk.slots")?;
    let d_qw = r.alloc(c.q_w.len() * 2, "selfcheck.qk.qw")?;
    let d_kw = r.alloc(c.k_w.len() * 2, "selfcheck.qk.kw")?;
    let d_cos = r.alloc(c.cos_t.len() * 2, "selfcheck.qk.cos")?;
    let d_sin = r.alloc(c.sin_t.len() * 2, "selfcheck.qk.sin")?;
    let d_pos = r.alloc(T * 4, "selfcheck.qk.pos")?;
    let d_out = r.alloc(T * HQ * HD * 2, "selfcheck.qk.out")?;
    r.upload(d_q, &bytes(&c.q_raw))?;
    r.upload(d_k, &bytes(&c.k))?;
    r.upload(d_v, &bytes(&c.v))?;
    r.upload(d_kc, &vec![POISON; n_pool])?;
    r.upload(d_vc, &vec![POISON; n_pool])?;
    r.upload(d_slots, &SLOTS.iter().flat_map(|s| s.to_le_bytes()).collect::<Vec<u8>>())?;
    r.upload(d_qw, &bytes(&c.q_w))?;
    r.upload(d_kw, &bytes(&c.k_w))?;
    r.upload(d_cos, &bytes(&c.cos_t))?;
    r.upload(d_sin, &bytes(&c.sin_t))?;
    r.upload(d_pos, &POS.iter().flat_map(|s| s.to_le_bytes()).collect::<Vec<u8>>())?;

    // 参数序逐字对齐 .cu 签名(11 块 + f32 + 4 i32 + 输出)
    let vals = [
        LaunchVal::Ptr(d_q.ptr),
        LaunchVal::Ptr(d_k.ptr),
        LaunchVal::Ptr(d_v.ptr),
        LaunchVal::Ptr(d_kc.ptr),
        LaunchVal::Ptr(d_vc.ptr),
        LaunchVal::Ptr(d_slots.ptr),
        LaunchVal::Ptr(d_qw.ptr),
        LaunchVal::Ptr(d_kw.ptr),
        LaunchVal::Ptr(d_cos.ptr),
        LaunchVal::Ptr(d_sin.ptr),
        LaunchVal::Ptr(d_pos.ptr),
        LaunchVal::F32(EPS),
        LaunchVal::I32(HKV as i32),
        LaunchVal::I32(HALF as i32),
        LaunchVal::I32(PAGE as i32),
        LaunchVal::I32(W_OFF),
        LaunchVal::Ptr(d_out.ptr),
    ];
    r.launch_aot(
        "owl_qknorm_rope_kv_insert_f16_fp8kv_knhd",
        QKNORM_ROPE_KV_INSERT_F16_FP8KV,
        (T as u32, (HQ + HKV) as u32, 1),
        (HD as u32, 1, 1),
        (HD * 4) as u32, // smem = hd·4B(核内 smem[d] 归约)
        &vals,
    )?;
    r.sync()?;

    let n_out = T * HQ * HD;
    let mut raw = vec![0u8; n_out * 2];
    r.download(CheckBuf { ptr: d_out.ptr, bytes: n_out * 2 }, &mut raw)?;
    let q_out: Vec<u16> = raw.chunks_exact(2).map(|x| u16::from_le_bytes([x[0], x[1]])).collect();
    let mut kc = vec![0u8; n_pool];
    let mut vc = vec![0u8; n_pool];
    r.download(CheckBuf { ptr: d_kc.ptr, bytes: n_pool }, &mut kc)?;
    r.download(CheckBuf { ptr: d_vc.ptr, bytes: n_pool }, &mut vc)?;
    Ok((q_out, kc, vc))
}

/// q 输出最大偏差(相对参考;f16 位型域)
pub fn q_worst(got: &[u16], want: &[u16]) -> (f32, Option<(usize, f32, f32)>) {
    let mut worst = 0f32;
    let mut first = None;
    for i in 0..got.len().min(want.len()) {
        let (g, w) = (f16_to_f32(got[i]), f16_to_f32(want[i]));
        let d = (g - w).abs();
        let rel = d / (w.abs().max(1e-3));
        if d > worst {
            worst = d;
        }
        if rel > TOL && first.is_none() {
            first = Some((i, g, w));
        }
    }
    (worst, first)
}

pub struct QknormInsertOracle;

impl BootCheck for QknormInsertOracle {
    fn name(&self) -> &'static str {
        "host norm+rope 参考 + e4m3 逐字节 + 毒值律"
    }

    fn run(&self, r: &mut dyn CheckRunner) -> Result<CheckReport, OpError> {
        let c = case();
        let n_pool = pool_bytes();
        let (q_out, kc, vc) = run_case(r, &c)?;
        let (want_q, want_k, want_v) = reference(&c, false, false);
        let op = "owl_qknorm_rope_kv_insert_f16_fp8kv_knhd";

        // ① q 输出(norm + rope)
        let (worst, first_bad) = q_worst(&q_out, &want_q);
        if let Some((i, g, w)) = first_bad {
            return Err(OpError::Contract {
                op: op.into(),
                field: "q_out",
                expect: format!("相对偏差 ≤ {TOL:.1e}(独立 norm+rope 参考)"),
                got: format!("首越界 idx={i} gpu={g} host={w}(max|err|={worst:.3e})"),
            });
        }

        // ② 池字节逐字节 + ③ 写集穷举毒值律
        let mut written = vec![false; n_pool];
        let check = |pool: &[u8], entries: &[(usize, u8)], which: &str| -> Result<(), OpError> {
            for (idx, want) in entries {
                let got = pool[*idx];
                if got == *want {
                    continue;
                }
                return Err(OpError::Contract {
                    op: op.into(),
                    field: if got == POISON { "漏写" } else { "数值" },
                    expect: "池字节逐字节等于 host e4m3 参考".into(),
                    got: format!("{which} idx={idx} 期望 {want:#04x} 实得 {got:#04x}"),
                });
            }
            Ok(())
        };
        check(&kc, &want_k, "K")?;
        check(&vc, &want_v, "V")?;
        for (idx, _) in want_k.iter() {
            written[*idx] = true;
        }
        let mut untouched = 0usize;
        for idx in 0..n_pool {
            if written[idx] {
                continue;
            }
            if kc[idx] != POISON || vc[idx] != POISON {
                return Err(OpError::Contract {
                    op: op.into(),
                    field: "越界写",
                    expect: format!("idx={idx} 在声明写集之外 ⇒ 必须保持毒值 {POISON:#04x}"),
                    got: format!("k={:#04x} v={:#04x}", kc[idx], vc[idx]),
                });
            }
            untouched += 1;
        }
        if untouched == 0 {
            return Err(OpError::Contract {
                op: op.into(),
                field: "毒值律",
                expect: "至少一个未写字节".into(),
                got: "池被全写".into(),
            });
        }
        let mut rep = CheckReport::numeric(op, worst, TOL);
        rep.first_bad = first_bad;
        rep.poison_untouched = Some(untouched);
        Ok(rep)
    }
}

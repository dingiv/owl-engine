//! `owl_naive_attn_nc_bf16_v2` 装配期自检(**独立 host 参考**)。
//!
//! 语义(`crates/kernels/cu/owl/dflash2.cu` 的 `owl_naive_attn_nc_bf16_v2`):
//! - 查询 = 自块 T 行,**非因果**:每行看 `[srow_begin, kv_len)`;
//! - `prefix = kv_len − gridDim.x`(自块 = 最后 T 行);
//! - `srow_begin = max(0, prefix − 2048)`(**滑窗**;自块恒全可见);
//! - 前缀行从池读(classic 布局),自块行从 `k_self/v_self` 直读;
//! - **单头全局 softmax**(非 online 语义)。
//!
//! 判据:GPU vs host 参考(上面这套语义**手写**),`max|err| ≤ 2e-2`(bf16 域)。
//! 几何取 `prefix = 2560 > 2048` ⇒ **滑窗真的咬掉 512 行**(③ 重写核的风险面
//! 正在这里;只多 8 行时差值落在 bf16 噪声内,判据区分不出来),同时把
//! HQ/HKV 缩小到 4/1 让 host 参考成本落在几十毫秒级。

use crate::contract::{LaunchVal, OpError};
use crate::spec::{BootCheck, CheckBuf, CheckReport, CheckRunner};
const DFLASH2_F16: &str = "cu/owl/naive_attn_nc.cu";

use super::{bf16_to_f32, lcg_bf16};

/// 自检几何(滑窗生效:prefix > 2048)
const T: usize = 8;
const HQ: usize = 4;
const HKV: usize = 1;
const HD: usize = 128;
const PAGE: usize = 32;
const X: usize = 8; // bf16:16B/2B
/// 前缀长 = 2048(窗) + 512 ⇒ **滑窗真的咬掉 512 行**(仅 8 行时差值落在
/// bf16 噪声内,判据区分不出来 —— 2026-10-13 实测:8.8e-3 vs 噪声 3.9e-3)
const PREFIX: usize = 2560;
const TOL: f32 = 2e-2;

/// 池元素下标(**classic x-block 布局**,逐字手写自 `.cu` 寻址式)
fn k_idx(kvh: usize, d: usize, page_off: usize) -> usize {
    let hd_x = HD / X;
    kvh * hd_x * PAGE * X + (d / X) * PAGE * X + page_off * X + (d % X)
}
fn v_idx(kvh: usize, d: usize, page_off: usize) -> usize {
    kvh * HD * PAGE + d * PAGE + page_off
}

/// host 参考:f32 全量 softmax(单头全局;滑窗 + 自块)。
///
/// `window: false` = **故意关掉滑窗** —— 测试用它证明判据对滑窗敏感
/// (③ 重写核的风险面正在滑窗路径)。
pub fn host_ref(
    q: &[u16],
    k_self: &[u16],
    v_self: &[u16],
    kc: &[u16],
    vc: &[u16],
    kv_len: usize,
    window: bool,
) -> Vec<f32> {
    let prefix = kv_len - T;
    let srow_begin = if window { prefix.saturating_sub(2048) } else { 0 };
    let scale = 1.0 / (HD as f32).sqrt();
    let mut out = vec![0f32; T * HQ * HD];
    for t in 0..T {
        for h in 0..HQ {
            let kvh = h / (HQ / HKV);
            let mut qk = Vec::with_capacity(kv_len - srow_begin);
            for srow in srow_begin..kv_len {
                let mut acc = 0f32;
                for d in 0..HD {
                    let qd = bf16_to_f32(q[(t * HQ + h) * HD + d]);
                    let kv = if srow < prefix {
                        let base = (srow / PAGE) * HKV * HD * PAGE;
                        kc[k_idx(kvh, d, srow % PAGE) + base]
                    } else {
                        k_self[((srow - prefix) * HKV + kvh) * HD + d]
                    };
                    acc += qd * bf16_to_f32(kv);
                }
                qk.push(acc * scale);
            }
            let m = qk.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let l: f32 = qk.iter().map(|&x| (x - m).exp()).sum();
            for d in 0..HD {
                let mut acc = 0f32;
                for (i, srow) in (srow_begin..kv_len).enumerate() {
                    let v = if srow < prefix {
                        let base = (srow / PAGE) * HKV * HD * PAGE;
                        vc[v_idx(kvh, d, srow % PAGE) + base]
                    } else {
                        v_self[((srow - prefix) * HKV + kvh) * HD + d]
                    };
                    acc += (qk[i] - m).exp() * bf16_to_f32(v);
                }
                out[(t * HQ + h) * HD + d] = acc / l;
            }
        }
    }
    out
}

/// 自检几何常量(测试与 oracle 共用)
pub struct Case {
    pub q: Vec<u16>,
    pub k_self: Vec<u16>,
    pub v_self: Vec<u16>,
    pub kc: Vec<u16>,
    pub vc: Vec<u16>,
    pub kv_len: usize,
}

/// 构造固定自检算例(确定性 LCG;种子固定 ⇒ 可复现)
pub fn case() -> Case {
    let kv_len = PREFIX + T;
    let nb = kv_len.div_ceil(PAGE);
    let pool = nb * HKV * HD * PAGE;
    Case {
        q: lcg_bf16(T * HQ * HD, 0x1111),
        k_self: lcg_bf16(T * HKV * HD, 0x2222),
        v_self: lcg_bf16(T * HKV * HD, 0x3333),
        kc: lcg_bf16(pool, 0x4444),
        vc: lcg_bf16(pool, 0x5555),
        kv_len,
    }
}

/// 跑一次核,返回**原始设备输出**(bf16 → f32);比较交给调用方
/// (oracle 与负例测试共用同一发射路径 ⇒ 判据敏感性可被证明)
pub fn run_case(r: &mut dyn CheckRunner, c: &Case) -> Result<Vec<f32>, OpError> {
    let bytes = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|h| h.to_le_bytes()).collect() };
    let d_q = r.alloc(c.q.len() * 2, "selfcheck.nc.q")?;
    let d_ks = r.alloc(c.k_self.len() * 2, "selfcheck.nc.ks")?;
    let d_vs = r.alloc(c.v_self.len() * 2, "selfcheck.nc.vs")?;
    let d_kc = r.alloc(c.kc.len() * 2, "selfcheck.nc.kc")?;
    let d_vc = r.alloc(c.vc.len() * 2, "selfcheck.nc.vc")?;
    let d_len = r.alloc(4, "selfcheck.nc.len")?;
    let d_out = r.alloc(T * HQ * HD * 2, "selfcheck.nc.out")?;
    r.upload(d_q, &bytes(&c.q))?;
    r.upload(d_ks, &bytes(&c.k_self))?;
    r.upload(d_vs, &bytes(&c.v_self))?;
    r.upload(d_kc, &bytes(&c.kc))?;
    r.upload(d_vc, &bytes(&c.vc))?;
    r.upload(d_len, &(c.kv_len as f32).to_le_bytes())?;

    // 参数序逐字对齐 .cu 签名:
    // [q, k_self, v_self, kc, vc, kv_len_p, q_heads, kv_heads, d_dim, page, x, out]
    let vals = [
        LaunchVal::Ptr(d_q.ptr),
        LaunchVal::Ptr(d_ks.ptr),
        LaunchVal::Ptr(d_vs.ptr),
        LaunchVal::Ptr(d_kc.ptr),
        LaunchVal::Ptr(d_vc.ptr),
        LaunchVal::Ptr(d_len.ptr),
        LaunchVal::I32(HQ as i32),
        LaunchVal::I32(HKV as i32),
        LaunchVal::I32(HD as i32),
        LaunchVal::I32(PAGE as i32),
        LaunchVal::I32(X as i32),
        LaunchVal::Ptr(d_out.ptr),
    ];
    r.launch_aot(
        "owl_naive_attn_nc_bf16_v2",
        DFLASH2_F16,
        (T as u32, HQ as u32, 1),
        (HD as u32, 1, 1),
        0,
        &vals,
    )?;
    r.sync()?;
    let n = T * HQ * HD;
    let mut raw = vec![0u8; n * 2];
    r.download(CheckBuf { ptr: d_out.ptr, bytes: n * 2 }, &mut raw)?;
    Ok(raw
        .chunks_exact(2)
        .map(|x| bf16_to_f32(u16::from_le_bytes([x[0], x[1]])))
        .collect())
}

/// 逐元素最大偏差 + 首个越界元素
pub fn worst_err(got: &[f32], want: &[f32]) -> (f32, Option<(usize, f32, f32)>) {
    let mut worst = 0f32;
    let mut first = None;
    for i in 0..got.len().min(want.len()) {
        let d = (got[i] - want[i]).abs();
        if d > worst {
            worst = d;
        }
        if d > TOL && first.is_none() {
            first = Some((i, got[i], want[i]));
        }
    }
    (worst, first)
}

pub struct NcBf16V2Oracle;

impl BootCheck for NcBf16V2Oracle {
    fn name(&self) -> &'static str {
        "host 全量 softmax 参考(滑窗 + 自块)"
    }

    fn run(&self, r: &mut dyn CheckRunner) -> Result<CheckReport, OpError> {
        let c = case();
        let got = run_case(r, &c)?;
        let want = host_ref(&c.q, &c.k_self, &c.v_self, &c.kc, &c.vc, c.kv_len, true);
        let (worst, first_bad) = worst_err(&got, &want);
        if let Some((i, g, w)) = first_bad {
            return Err(OpError::Contract {
                op: "owl_naive_attn_nc_bf16_v2".into(),
                field: "out",
                expect: format!("|gpu − host| ≤ {TOL:.1e}(独立 host 全量 softmax 参考)"),
                got: format!("首越界 idx={i} gpu={g} host={w}(max|err|={worst:.3e})"),
            });
        }
        let mut rep = CheckReport::numeric("owl_naive_attn_nc_bf16_v2", worst, TOL);
        rep.first_bad = first_bad;
        Ok(rep)
    }
}

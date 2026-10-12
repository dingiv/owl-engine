//! `owl_reshape_and_cache_fp8kv_knhd` 装配期自检。
//!
//! 两条判据(**数值 + 写集穷举**,设计 §5.4/§5.5):
//! 1. **数值**:池字节逐字节 == host e4m3 RNE 参考(与核内 `__nv_fp8_e4m3` 同式);
//! 2. **毒值律**:池先灌 e4m3-NaN 码(`0x7F`),发射后**声明写集之外的字节
//!    必须仍是毒值** —— 越界写当场红,漏写则数值判据红。
//!
//! 这正是 E10「池 dtype 改造漏写者 ⇒ 槽区永不落笔全零」的机器面:
//! 漏写 = 毒值残留,越写 = 毒值被覆盖。

use crate::contract::{LaunchVal, OpError};
use crate::spec::{BootCheck, CheckBuf, CheckReport, CheckRunner};
const RESHAPE_AND_CACHE_FP8KV: &str = include_str!("../../../cu/attention/reshape_and_cache_fp8kv.cu");

use super::{e4m3_encode, f16_to_f32, lcg_f16};

/// 自检几何(小规模,秒级;含跨页 + padding 两个边界)
const HKV: usize = 2;
const HD: usize = 256;
const BS: usize = 32;
const NB: usize = 4;
const T: usize = 4;
/// token3 = padding(slot < 0)⇒ 不许写任何字节
const SLOTS: [f32; T] = [0.0, 31.0, 32.0, -1.0];
/// 毒值 = e4m3 NaN 码
const POISON: u8 = 0x7F;
/// classic 布局专参(仅负例参考用;knHD 寻址与 x 无关)
const X: usize = 8;
const TOL: f32 = 0.0; // 字节级判据:必须逐字节相等

/// 自检算例(确定性;种子固定 ⇒ 可复现)
pub struct Case {
    pub key: Vec<u16>,
    pub value: Vec<u16>,
}

/// 构造固定算例
pub fn case() -> Case {
    Case { key: lcg_f16(T * HKV * HD, 0x1234), value: lcg_f16(T * HKV * HD, 0x9ABC) }
}

/// 池字节数
pub fn pool_bytes() -> usize {
    NB * HKV * HD * BS
}

/// 跑一次核,返回**原始池字节**(毒值灌池 → 发射 → 回读);
/// 比较交给调用方(oracle 与负例测试共用同一发射路径)
pub fn run_case(r: &mut dyn CheckRunner, c: &Case) -> Result<(Vec<u8>, Vec<u8>), OpError> {
    let n = pool_bytes();
    let d_key = r.alloc(c.key.len() * 2, "selfcheck.k0.key")?;
    let d_val = r.alloc(c.value.len() * 2, "selfcheck.k0.val")?;
    let d_kc = r.alloc(n, "selfcheck.k0.kc")?;
    let d_vc = r.alloc(n, "selfcheck.k0.vc")?;
    let d_slots = r.alloc(T * 4, "selfcheck.k0.slots")?;
    let d_out = r.alloc(2, "selfcheck.k0.out")?;
    let hb = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|h| h.to_le_bytes()).collect() };
    r.upload(d_key, &hb(&c.key))?;
    r.upload(d_val, &hb(&c.value))?;
    r.upload(d_kc, &vec![POISON; n])?;
    r.upload(d_vc, &vec![POISON; n])?;
    r.upload(d_slots, &SLOTS.iter().flat_map(|s| s.to_le_bytes()).collect::<Vec<u8>>())?;

    let vals = [
        LaunchVal::Ptr(d_key.ptr),
        LaunchVal::Ptr(d_val.ptr),
        LaunchVal::Ptr(d_kc.ptr),
        LaunchVal::Ptr(d_vc.ptr),
        LaunchVal::Ptr(d_slots.ptr),
        LaunchVal::I32((HKV * HD) as i32),
        LaunchVal::I32((HKV * HD) as i32),
        LaunchVal::I32(HKV as i32),
        LaunchVal::I32(HD as i32),
        LaunchVal::I32(BS as i32),
        LaunchVal::I32(8),
        LaunchVal::Ptr(d_out.ptr),
    ];
    r.launch_aot(
        "owl_reshape_and_cache_fp8kv_knhd",
        RESHAPE_AND_CACHE_FP8KV,
        (T as u32, 1, 1),
        (256, 1, 1),
        0,
        &vals,
    )?;
    r.sync()?;
    let mut kc = vec![0u8; n];
    let mut vc = vec![0u8; n];
    r.download(CheckBuf { ptr: d_kc.ptr, bytes: n }, &mut kc)?;
    r.download(CheckBuf { ptr: d_vc.ptr, bytes: n }, &mut vc)?;
    Ok((kc, vc))
}

/// 声明写集(knHD 寻址展开):返回 `(下标, 期望 K 字节, 期望 V 字节)`
/// —— 参考 = host e4m3 RNE。`layout_classic: true` = **故意用错布局**
/// (测试用它证明判据对寻址敏感)。
pub fn reference(c: &Case, layout_classic: bool) -> Vec<(usize, u8, u8)> {
    let mut v = Vec::new();
    for (t, slot) in SLOTS.iter().enumerate().take(T) {
        if *slot < 0.0 {
            continue;
        }
        let slot = *slot as usize;
        let (blk, off) = (slot / BS, slot % BS);
        for h in 0..HKV {
            for d in 0..HD {
                let src = t * HKV * HD + h * HD + d;
                let idx = if layout_classic {
                    // classic x-block(故意错):[blk, h, d/X, off, X]
                    let hd_x = HD / X;
                    blk * HKV * hd_x * BS * X + h * hd_x * BS * X + (d / X) * BS * X + off * X + (d % X)
                } else {
                    // knHD:[blk, off, h, d]
                    blk * HKV * HD * BS + off * HKV * HD + h * HD + d
                };
                v.push((
                    idx,
                    e4m3_encode(f16_to_f32(c.key[src])),
                    e4m3_encode(f16_to_f32(c.value[src])),
                ));
            }
        }
    }
    v
}

pub struct K0Fp8KnhdOracle;

impl BootCheck for K0Fp8KnhdOracle {
    fn name(&self) -> &'static str {
        "e4m3 逐字节参考 + 写集穷举毒值律"
    }

    fn run(&self, r: &mut dyn CheckRunner) -> Result<CheckReport, OpError> {
        let c = case();
        let n = pool_bytes();
        let (kc, vc) = run_case(r, &c)?;
        let want = reference(&c, false);
        let mut written = vec![false; n];
        for (idx, wk, wv) in &want {
            written[*idx] = true;
            for (got, w, tag) in [(kc[*idx], *wk, "K"), (vc[*idx], *wv, "V")] {
                if got == w {
                    continue;
                }
                let field = if got == POISON { "漏写" } else { "数值" };
                return Err(OpError::Contract {
                    op: "owl_reshape_and_cache_fp8kv_knhd".into(),
                    field,
                    expect: "池字节逐字节等于 host e4m3 参考".into(),
                    got: format!("{tag} idx={idx} 期望 {w:#04x} 实得 {got:#04x}"),
                });
            }
        }
        // 毒值律:声明写集之外的字节必须仍是毒值(越界写当场红)
        let mut untouched = 0usize;
        for idx in 0..n {
            if written[idx] {
                continue;
            }
            if kc[idx] != POISON || vc[idx] != POISON {
                return Err(OpError::Contract {
                    op: "owl_reshape_and_cache_fp8kv_knhd".into(),
                    field: "越界写",
                    expect: format!("idx={idx} 在声明写集之外 ⇒ 必须保持毒值 {POISON:#04x}"),
                    got: format!("k={:#04x} v={:#04x}", kc[idx], vc[idx]),
                });
            }
            untouched += 1;
        }
        if untouched == 0 {
            return Err(OpError::Contract {
                op: "owl_reshape_and_cache_fp8kv_knhd".into(),
                field: "毒值律",
                expect: "至少一个未写字节".into(),
                got: "池被全写".into(),
            });
        }
        let mut rep = CheckReport::numeric("owl_reshape_and_cache_fp8kv_knhd", 0.0, TOL);
        rep.poison_untouched = Some(untouched);
        Ok(rep)
    }
}

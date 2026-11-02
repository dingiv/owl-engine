//! `owl_reshape_and_cache_int8_knhd` 装配期自检。
//!
//! 判据(与 fp8 K0 oracle 同纪律):
//! 1. **数值**:int8 池字节 + fp32 scale 位型逐字节 == host 参考;
//! 2. **毒值律**:池先灌毒值,声明写集之外的字节/scale 必须原样保留。
//!
//! host 参考独立实现量化约定(pth/vLLM int8_per_token_head writer 的镜像),
//! 不复用被测核代码。

use crate::bootcheck::{BootCheck, CheckBuf, CheckReport, CheckRunner};
use crate::contract::{LaunchVal, OpError};

use super::{f16_to_f32, lcg_f16};

const SRC: &str = "cu/attn/reshape_and_cache_int8.cu";
const ENTRY: &str = "owl_reshape_and_cache_int8_knhd";

const HKV: usize = 2;
const HD: usize = 256;
const BS: usize = 32;
const NB: usize = 4;
const T: usize = 4;
/// token3 = padding(slot < 0)⇒ 不许写任何字节
const SLOTS: [f32; T] = [0.0, 31.0, 32.0, -1.0];
const POISON_I8: i8 = 0x7F;
const POISON_F32: f32 = f32::NAN;

pub struct Case {
    pub key: Vec<u16>,
    pub value: Vec<u16>,
}

pub fn case() -> Case {
    Case {
        key: lcg_f16(T * HKV * HD, 0x1234),
        value: lcg_f16(T * HKV * HD, 0x9ABC),
    }
}

pub fn pool_elems() -> usize {
    NB * BS * HKV * HD
}

pub fn scale_elems() -> usize {
    NB * BS * HKV
}

fn quantize_row(row: &[u16]) -> (Vec<i8>, f32) {
    let x: Vec<f32> = row.iter().map(|&h| f16_to_f32(h)).collect();
    let absmax = x.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let scale = (absmax / 127.0f32).max(1e-6f32);
    let inv = 1.0f32 / scale;
    let q = x
        .iter()
        .map(|&v| {
            let mut q = v * inv;
            q = if q >= 0.0 { q + 0.5 } else { q - 0.5 };
            q = q.clamp(-128.0, 127.0);
            q as i8
        })
        .collect();
    (q, scale)
}

/// 独立 host 参考:返回 (K 池, V 池, K scale, V scale),未写位置保持毒值。
pub fn reference(c: &Case) -> (Vec<i8>, Vec<i8>, Vec<f32>, Vec<f32>) {
    let mut kc = vec![POISON_I8; pool_elems()];
    let mut vc = vec![POISON_I8; pool_elems()];
    let mut ks = vec![POISON_F32; scale_elems()];
    let mut vs = vec![POISON_F32; scale_elems()];

    for (t, slot) in SLOTS.iter().enumerate() {
        if *slot < 0.0 {
            continue;
        }
        let slot = *slot as usize;
        let (blk, off) = (slot / BS, slot % BS);
        for h in 0..HKV {
            let row = t * HKV * HD + h * HD;
            let (kq, k_scale) = quantize_row(&c.key[row..row + HD]);
            let (vq, v_scale) = quantize_row(&c.value[row..row + HD]);
            let dst = (blk * BS + off) * HKV * HD + h * HD;
            kc[dst..dst + HD].copy_from_slice(&kq);
            vc[dst..dst + HD].copy_from_slice(&vq);
            let sidx = (blk * BS + off) * HKV + h;
            ks[sidx] = k_scale;
            vs[sidx] = v_scale;
        }
    }
    (kc, vc, ks, vs)
}

fn bytes_i8(v: &[i8]) -> Vec<u8> {
    v.iter().map(|&x| x as u8).collect()
}

fn bytes_f32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect()
}

/// 跑一次核,返回原始池字节与 scale 字节(毒值灌池 → 发射 → 回读)。
pub fn run_case(
    r: &mut dyn CheckRunner,
    c: &Case,
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>), OpError> {
    let n = pool_elems();
    let sn = scale_elems();
    let d_key = r.alloc(c.key.len() * 2, "selfcheck.int8.key")?;
    let d_val = r.alloc(c.value.len() * 2, "selfcheck.int8.val")?;
    let d_kc = r.alloc(n, "selfcheck.int8.kc")?;
    let d_vc = r.alloc(n, "selfcheck.int8.vc")?;
    let d_ks = r.alloc(sn * 4, "selfcheck.int8.ks")?;
    let d_vs = r.alloc(sn * 4, "selfcheck.int8.vs")?;
    let d_slots = r.alloc(T * 4, "selfcheck.int8.slots")?;
    let d_out = r.alloc(2, "selfcheck.int8.out")?;

    let hb = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|h| h.to_le_bytes()).collect() };
    r.upload(d_key, &hb(&c.key))?;
    r.upload(d_val, &hb(&c.value))?;
    r.upload(d_kc, &vec![POISON_I8 as u8; n])?;
    r.upload(d_vc, &vec![POISON_I8 as u8; n])?;
    r.upload(d_ks, &bytes_f32(&vec![POISON_F32; sn]))?;
    r.upload(d_vs, &bytes_f32(&vec![POISON_F32; sn]))?;
    r.upload(
        d_slots,
        &SLOTS
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect::<Vec<u8>>(),
    )?;

    let vals = [
        LaunchVal::Ptr(d_key.ptr),
        LaunchVal::Ptr(d_val.ptr),
        LaunchVal::Ptr(d_kc.ptr),
        LaunchVal::Ptr(d_vc.ptr),
        LaunchVal::Ptr(d_ks.ptr),
        LaunchVal::Ptr(d_vs.ptr),
        LaunchVal::Ptr(d_slots.ptr),
        LaunchVal::I32((HKV * HD) as i32),
        LaunchVal::I32((HKV * HD) as i32),
        LaunchVal::I32(HKV as i32),
        LaunchVal::I32(HD as i32),
        LaunchVal::I32(BS as i32),
        LaunchVal::I32(T as i32),
        LaunchVal::Ptr(d_out.ptr),
    ];
    let warps = T * HKV;
    let grid = ((warps + 3) / 4) as u32;
    r.launch_aot(ENTRY, SRC, (grid, 1, 1), (128, 1, 1), 0, &vals)?;
    r.sync()?;

    let mut kc = vec![0u8; n];
    let mut vc = vec![0u8; n];
    let mut ks = vec![0u8; sn * 4];
    let mut vs = vec![0u8; sn * 4];
    r.download(
        CheckBuf {
            ptr: d_kc.ptr,
            bytes: n,
        },
        &mut kc,
    )?;
    r.download(
        CheckBuf {
            ptr: d_vc.ptr,
            bytes: n,
        },
        &mut vc,
    )?;
    r.download(
        CheckBuf {
            ptr: d_ks.ptr,
            bytes: sn * 4,
        },
        &mut ks,
    )?;
    r.download(
        CheckBuf {
            ptr: d_vs.ptr,
            bytes: sn * 4,
        },
        &mut vs,
    )?;
    Ok((kc, vc, ks, vs))
}

pub struct K0Int8KnhdOracle;

impl BootCheck for K0Int8KnhdOracle {
    fn name(&self) -> &'static str {
        "int8 池字节/scale 位型逐字节参考 + 写集穷举毒值律"
    }

    fn run(&self, r: &mut dyn CheckRunner) -> Result<CheckReport, OpError> {
        let c = case();
        let (kc, vc, ks, vs) = run_case(r, &c)?;
        let (wk, wv, wks, wvs) = reference(&c);
        let wk = bytes_i8(&wk);
        let wv = bytes_i8(&wv);
        let wks = bytes_f32(&wks);
        let wvs = bytes_f32(&wvs);

        for (got, want, tag) in [
            (&kc, &wk, "K"),
            (&vc, &wv, "V"),
            (&ks, &wks, "k_scale"),
            (&vs, &wvs, "v_scale"),
        ] {
            if got != want {
                let idx = got
                    .iter()
                    .zip(want.iter())
                    .position(|(a, b)| a != b)
                    .unwrap_or(0);
                return Err(OpError::Contract {
                    op: ENTRY.into(),
                    field: "数值",
                    expect: format!("{tag} 逐字节等于 host 参考"),
                    got: format!(
                        "首差 idx={idx} 期望 {:#04x} 实得 {:#04x}",
                        want[idx], got[idx]
                    ),
                });
            }
        }

        // 毒值律:参考里仍是毒值的位置 = 未写集;数值门已保证实得 == 参考,
        // 这里只需统计未写集非空(全写 = 写集声明过宽)。
        let mut untouched = 0usize;
        for &w in &wk {
            if w == POISON_I8 as u8 {
                untouched += 1;
            }
        }
        for &w in &wv {
            if w == POISON_I8 as u8 {
                untouched += 1;
            }
        }
        for chunk in wks.chunks_exact(4) {
            if u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) == POISON_F32.to_bits()
            {
                untouched += 1;
            }
        }
        for chunk in wvs.chunks_exact(4) {
            if u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) == POISON_F32.to_bits()
            {
                untouched += 1;
            }
        }
        if untouched == 0 {
            return Err(OpError::Contract {
                op: ENTRY.into(),
                field: "毒值律",
                expect: "至少一个未写字节".into(),
                got: "池被全写".into(),
            });
        }

        let mut rep = CheckReport::numeric(ENTRY, 0.0, 0.0);
        rep.poison_untouched = Some(untouched);
        Ok(rep)
    }
}

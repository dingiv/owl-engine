//! `owl_quant_int8_row_{f16,bf16}` 对拍门(S1c;2026-10-16)。
//!
//! W4A8 的**喂料手**:激活 → per-token int8 + 尺度。判据 = host 独立参考
//! **逐元素对拍**(口径与核内逐点对齐):
//! - 行 absmax → `scale = absmax/127 · mul`;
//! - `q = 饱和(round-to-nearest-*even*)(x · 127/absmax)`(`cvt.rni.sat.s8`);
//! - 零行 ⇒ `scale = 0` 且整行 `q = 0`;
//! - `mul`(权重侧全局因子)**只进 scales 出口**,与量化无关
//!   ⇒ 不变量:同输入换 `mul`,q 逐字节不变、scales 线性跟随。
//!
//! 用例行(每行一个刁钻面,见 [`build_rows`]):伪随机常规 / 全零 / 大动态范围 /
//! **舍入平局**(absmax=127 ⇒ inv=1,值取 k+0.5;`round-half-away` 会在这里红)/
//! 极小白量级 / 交替正负 / 稀疏单峰 / 全等。
//!
//! 跑法:`OWL_TEST_DEVICE=<3080 ordinal> cargo test -p owl-kernels \
//!       --features device --test quant_int8_row_probe -- --nocapture`
#![cfg(feature = "device")]

mod fi_support;
use fi_support::TestRes;

use owl_kernels::bootcheck::CheckRunner;
use owl_kernels::contract::{Dtype, LaunchVal};
use owl_kernels::device::Exec;
use owl_kernels::ops::oracles::{bf16_to_f32, f16_bits, f16_to_f32, f32_to_bf16, lcg_f16};
use owl_kernels::server::selfcheck::Runner;

const SRC: &str = "cu/elems/quant_int8_row.cu";
const BLOCK: u32 = 256;
const ROWS: usize = 8;
/// 两组 cols:非 256 倍数(核内 strided 兜底)与对齐档
const COLS_CASES: [usize; 2] = [300, 512];

/// 用例行(逐行一个刁钻面)
fn build_rows(cols: usize) -> Vec<f32> {
    let mut x = vec![0f32; ROWS * cols];
    for r in 0..ROWS {
        let row = &mut x[r * cols..(r + 1) * cols];
        match r {
            // 0 伪随机常规(与仓内其它 oracle 同源口径,量级 ±4)
            0 => {
                let bits = lcg_f16(cols, 0x51C);
                for (c, b) in row.iter_mut().enumerate() {
                    *b = f16_to_f32(bits[c]);
                }
            }
            // 1 全零行(⇒ scale 0、整行 0)
            1 => {}
            // 2 大动态范围:单峰 2048 + 小台阶
            2 => {
                row[cols / 3] = 2048.0;
                for (c, v) in row.iter_mut().enumerate() {
                    *v = ((c % 7) as f32 - 3.0) * 0.25;
                }
                row[cols / 3] = 2048.0;
            }
            // 3 **舍入平局**(absmax=127 ⇒ inv=1.0 ⇒ q = round_ties_even(x))
            3 => {
                row[0] = 127.0;
                for (c, v) in row.iter_mut().enumerate().skip(1) {
                    let k = (c % 8) as f32 + 0.5; // 0.5 … 7.5
                    *v = if c % 2 == 0 { k } else { -k };
                }
            }
            // 4 极小白量级(2^-9 台阶,在 f16 域可精确表示)
            4 => {
                for (c, v) in row.iter_mut().enumerate() {
                    *v = ((c % 5) as f32 - 2.0) * (1.0 / 512.0);
                }
            }
            // 5 交替正负(检验"分段归约"没把符号弄丢)
            5 => {
                for (c, v) in row.iter_mut().enumerate() {
                    let mag = ((c % 9) as f32) * 0.125;
                    *v = if c % 2 == 0 { mag } else { -mag };
                }
            }
            // 6 稀疏单峰
            6 => {
                row[cols - 1] = 96.0;
            }
            // 7 全等(⇒ 每个元素都量化到 ±127)
            _ => {
                for v in row.iter_mut() {
                    *v = 2.0;
                }
            }
        }
    }
    x
}

/// host 独立参考(f32 域;round-to-nearest-**even** = `cvt.rni` 口径,
/// 饱和由 Rust 浮点→整型的饱和 cast 提供)
fn host_ref(x: &[f32], rows: usize, cols: usize, mul: f32) -> (Vec<i8>, Vec<f32>) {
    let mut q = vec![0i8; rows * cols];
    let mut s = vec![0f32; rows];
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let absmax = row.iter().fold(0f32, |m, v| m.max(v.abs()));
        s[r] = absmax / 127.0 * mul;
        let inv = if absmax == 0.0 { 0.0 } else { 127.0 / absmax };
        for (c, v) in row.iter().enumerate() {
            q[r * cols + c] = (v * inv).round_ties_even() as i8;
        }
    }
    (q, s)
}

fn with_runner<R>(f: impl FnOnce(&mut dyn CheckRunner) -> R) -> Option<R> {
    let dev = owl_shared::env_reader::parse::<usize>("OWL_TEST_DEVICE")?;
    let mut res = TestRes::new(dev);
    let mut exec = Exec::new();
    // 自检面:纯 native 发射(`launch_aot`),注册表只服务家族臂 ⇒ None
    let mut r = Runner { res: &mut res, exec: &mut exec, registry: None };
    Some(f(&mut r))
}

/// 发射一趟(几何与 `driver::elems::quant_int8_row` 单源:grid = rows,block = 256)
fn run(dt: Dtype, x_bits: &[u8], rows: usize, cols: usize, mul: f32) -> Option<(Vec<i8>, Vec<f32>)> {
    with_runner(|r| {
        let x = r.alloc(x_bits.len(), "s1c.x").expect("alloc x");
        r.upload(x, x_bits).expect("upload x");
        let s = r.alloc(rows * 4, "s1c.scales").expect("alloc scales");
        let m = r.alloc(4, "s1c.mul").expect("alloc mul");
        r.upload(m, &mul.to_le_bytes()).expect("upload mul");
        let q = r.alloc(rows * cols, "s1c.q").expect("alloc q");
        let name: &'static str = match dt {
            Dtype::F16 => "owl_quant_int8_row_f16",
            Dtype::BF16 => "owl_quant_int8_row_bf16",
            other => panic!("无 {other:?} 变体"),
        };
        r.launch_aot(
            name,
            SRC,
            (rows as u32, 1, 1),
            (BLOCK, 1, 1),
            0,
            &[
                LaunchVal::Ptr(x.ptr),
                LaunchVal::I32(rows as i32),
                LaunchVal::I32(cols as i32),
                LaunchVal::Ptr(m.ptr), // 设备标量块(float* mul)
                LaunchVal::Ptr(s.ptr),
                LaunchVal::Ptr(q.ptr),
            ],
        )
        .expect("quant_int8_row launch");
        r.sync().expect("sync");
        let mut qh = vec![0u8; rows * cols];
        r.download(q, &mut qh).expect("dl q");
        let mut sh = vec![0u8; rows * 4];
        r.download(s, &mut sh).expect("dl s");
        (
            qh.iter().map(|b| *b as i8).collect::<Vec<i8>>(),
            sh.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<f32>>(),
        )
    })
}

fn cmp_i8(got: &[i8], want: &[i8], cols: usize, what: &str) {
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(
            g, w,
            "{what}: 首处不符 idx={i}(row {}, col {})gpu={g} host={w}",
            i / cols,
            i % cols
        );
    }
}

/// f16 臂:f16 域精确用例(值先过 f16 舍入,参考用**舍入后**的值)
#[test]
fn quant_int8_row_f16_matches_host() {
    let Some(dev_ok) = with_runner(|_| true) else {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    };
    assert!(dev_ok);
    for cols in COLS_CASES {
        let x_f32 = build_rows(cols);
        let bits: Vec<u16> = x_f32.iter().map(|v| f16_bits(*v)).collect();
        let x_f16: Vec<f32> = bits.iter().map(|b| f16_to_f32(*b)).collect();
        let raw: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
        let (q, s) = run(Dtype::F16, &raw, ROWS, cols, 1.0).expect("run");
        let (wq, ws) = host_ref(&x_f16, ROWS, cols, 1.0);
        cmp_i8(&q, &wq, cols, &format!("f16 cols={cols}"));
        for r in 0..ROWS {
            assert!(
                (s[r] - ws[r]).abs() <= ws[r].abs().max(1e-12) * 1e-6,
                "f16 cols={cols} row={r}: scales gpu={} host={}",
                s[r],
                ws[r]
            );
        }
        eprintln!(
            "[quant-int8] f16 cols={cols} ✓ q 逐元素一致 / scales 逐行一致(absmax 最大 {:.3})",
            ws.iter().cloned().fold(0f32, f32::max)
        );
    }
}

/// bf16 臂(同判据;值先过 bf16 舍入)
#[test]
fn quant_int8_row_bf16_matches_host() {
    let Some(res) = with_runner(|_| ()) else {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    };
    let () = res;
    let cols = COLS_CASES[0];
    let x_f32 = build_rows(cols);
    let bits: Vec<u16> = x_f32.iter().map(|v| f32_to_bf16(*v)).collect();
    let x_bf: Vec<f32> = bits.iter().map(|b| bf16_to_f32(*b)).collect();
    let raw: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
    let (q, s) = run(Dtype::BF16, &raw, ROWS, cols, 1.0).expect("run");
    let (wq, ws) = host_ref(&x_bf, ROWS, cols, 1.0);
    cmp_i8(&q, &wq, cols, "bf16");
    for r in 0..ROWS {
        assert!(
            (s[r] - ws[r]).abs() <= ws[r].abs().max(1e-12) * 1e-6,
            "bf16 row={r}: scales gpu={} host={}",
            s[r],
            ws[r]
        );
    }
    eprintln!("[quant-int8] bf16 cols={cols} ✓ q 逐元素一致 / scales 逐行一致");
}

/// **不变量**:`mul`(权重侧全局因子)只进 scales 出口 —— q 逐字节不变、
/// scales 线性跟随。这条一旦破,W4A8 的量化与因子折叠就会互相污染。
#[test]
fn mul_only_scales_q_invariant() {
    let Some(res) = with_runner(|_| ()) else {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    };
    let () = res;
    let cols = COLS_CASES[0];
    let x_f32 = build_rows(cols);
    let bits: Vec<u16> = x_f32.iter().map(|v| f16_bits(*v)).collect();
    let raw: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
    let (q1, s1) = run(Dtype::F16, &raw, ROWS, cols, 1.0).expect("run mul=1");
    let (q4, s4) = run(Dtype::F16, &raw, ROWS, cols, 0.25).expect("run mul=0.25");
    assert_eq!(q1, q4, "q 必须与 mul 无关(因子只进 scales 出口)");
    for r in 0..ROWS {
        assert!(
            (s4[r] - s1[r] * 0.25).abs() <= s1[r].abs().max(1e-12) * 1e-6,
            "row={r}: scales 未线性跟随 mul(gpu {} vs 1.0 档 {})",
            s4[r],
            s1[r]
        );
    }
    eprintln!("[quant-int8] mul 不变量 ✓ q 逐字节不变 / scales ×0.25 线性跟随");
}

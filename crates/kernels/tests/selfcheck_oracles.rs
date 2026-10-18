//! 装配期 oracle 门(2026-10-13 P5):**正例 + 判据敏感性负例**。
//!
//! 正例 = oracle 自己跑一遍(与生产 boot 同一条路径:真 DeviceRes × Exec)。
//! 负例 = 把参考**故意做错**(NC 关掉滑窗 / K0 换成 classic 寻址),证明
//! 判据真的能红 —— 否则"自检通过"可能只是判据太松(门变摆设)。
//!
//! 运行:
//! ```bash
//! OWL_TEST_DEVICE=1 cargo test -p owl-kernels --features device,cublas,marlin,flashinfer,xqa \
//!   --test selfcheck_oracles -- --test-threads=1 --nocapture
//! ```
#![cfg(feature = "device")]

use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use owl_kernels::contract::{Dtype, OpError};
use owl_kernels::device::{DeviceRes, Exec, LaunchNote, ScratchBuf};
use owl_kernels::ops::oracles::{
    cublas_gemm as cb, k0_fp8_knhd as k0, nc_bf16_v2 as nc, qknorm_insert as qk,
};
#[cfg(feature = "marlin")]
use owl_kernels::ops::oracles::marlin as ml;
use owl_kernels::server::selfcheck::Runner;
use owl_kernels::bootcheck::{BootCheck, CheckRunner};

/// 测试用资源面(最小实现:分配/上传/回读 —— oracle 的全部需求)
struct TestRes {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    blocks: HashMap<u64, CudaSlice<u8>>,
    next: u64,
}

impl TestRes {
    fn new() -> Option<Self> {
        let ordinal = owl_cuda::test_device_ordinal_opt()?;
        let ctx = CudaContext::new(ordinal).expect("ctx");
        ctx.bind_to_thread().expect("bind");
        let stream = ctx.default_stream();
        Some(Self { ctx, stream, blocks: HashMap::new(), next: 1 })
    }
}

impl DeviceRes for TestRes {
    fn context(&self) -> Result<Arc<CudaContext>, OpError> {
        Ok(self.ctx.clone())
    }
    fn stream(&self) -> Result<Arc<CudaStream>, OpError> {
        Ok(self.stream.clone())
    }
    fn resolve(&self, id: u64) -> Result<u64, OpError> {
        self.blocks
            .get(&id)
            .map(|s| {
                use cudarc::driver::DevicePtr;
                s.device_ptr(&self.stream).0
            })
            .ok_or_else(|| OpError::Contract {
                op: "test".into(),
                field: "resolve",
                expect: "块在册".into(),
                got: format!("id={}", id),
            })
    }
    fn resolve_len(&self, id: u64) -> Result<usize, OpError> {
        self.blocks.get(&id).map(|s| s.len()).ok_or_else(|| OpError::Contract {
            op: "test".into(),
            field: "resolve_len",
            expect: "块在册".into(),
            got: format!("id={}", id),
        })
    }
    fn alloc(&mut self, bytes: usize, _tag: &str) -> Result<ScratchBuf, OpError> {
        let s = unsafe { self.stream.alloc::<u8>(bytes.max(1)) }.map_err(|e| OpError::Contract {
            op: "test".into(),
            field: "alloc",
            expect: "设备分配".into(),
            got: format!("{e:?}"),
        })?;
        let id = self.next;
        self.next += 1;
        let ptr = {
            use cudarc::driver::DevicePtr;
            s.device_ptr(&self.stream).0
        };
        self.blocks.insert(id, s);
        Ok(ScratchBuf { ptr, bytes })
    }
    fn capturing(&self) -> Result<bool, OpError> {
        Ok(false)
    }
    fn record_capture(&mut self, _n: LaunchNote) -> Result<(), OpError> {
        Ok(())
    }
    fn upload(&mut self, dst: u64, src: &[u8]) -> Result<(), OpError> {
        // ⚠️ 不许 clone CudaSlice(深拷贝 ⇒ 写进副本,设备缓冲永远是零 ——
        // 2026-10-13 实测:那会让 oracle 拿到全零输入而"假绿")
        use cudarc::driver::DevicePtr;
        let stream = self.stream.clone();
        let id = self
            .blocks
            .iter()
            .find(|(_, s)| s.device_ptr(&stream).0 == dst)
            .map(|(id, _)| *id)
            .ok_or_else(|| OpError::Contract {
                op: "test".into(),
                field: "upload",
                expect: "ptr 在册".into(),
                got: format!("{dst:#x}"),
            })?;
        let s = self.blocks.get_mut(&id).expect("刚找到");
        let n = src.len().min(s.len());
        stream.memcpy_htod(&src[..n], s).map_err(|e| OpError::Contract {
            op: "test".into(),
            field: "upload",
            expect: "htod".into(),
            got: format!("{e:?}"),
        })
    }
    fn upload_block(
        &mut self,
        b: &owl_kernels::contract::Bytes,
        src: &[u8],
    ) -> Result<(), owl_kernels::contract::OpError> {
        // 测试面:按块 id 直查(账房即 self.blocks)
        let stream = self.stream.clone();
        let s = self.blocks.get_mut(&b.id).ok_or_else(|| {
            owl_kernels::contract::OpError::Contract {
                op: "test".into(),
                field: "upload_block",
                expect: "块在册".into(),
                got: format!("id={}", b.id),
            }
        })?;
        stream.memcpy_htod(src, s).map_err(|e| owl_kernels::contract::OpError::Contract {
            op: "test".into(),
            field: "upload_block",
            expect: "htod".into(),
            got: format!("{e:?}"),
        })
    }

    fn alloc_block(
        &mut self,
        bytes: usize,
        _tag: &str,
    ) -> Result<owl_kernels::contract::Bytes, OpError> {
        let s = unsafe { self.stream.alloc::<u8>(bytes.max(1)) }.map_err(|e| OpError::Contract {
            op: "test".into(),
            field: "alloc_block",
            expect: "设备分配".into(),
            got: format!("{e:?}"),
        })?;
        let id = self.next;
        self.next += 1;
        self.blocks.insert(id, s);
        Ok(owl_kernels::contract::Bytes::new(id, bytes, Dtype::U8, 0))
    }
    fn download(&mut self, src: u64, dst: &mut [u8]) -> Result<(), OpError> {
        use cudarc::driver::DevicePtr;
        let stream = self.stream.clone();
        let id = self
            .blocks
            .iter()
            .find(|(_, s)| s.device_ptr(&stream).0 == src)
            .map(|(id, _)| *id)
            .ok_or_else(|| OpError::Contract {
                op: "test".into(),
                field: "download",
                expect: "ptr 在册".into(),
                got: format!("{src:#x}"),
            })?;
        let s = self.blocks.get(&id).expect("刚找到");
        let mut tmp = vec![0u8; s.len()];
        stream.memcpy_dtoh(s, &mut tmp).map_err(|e| OpError::Contract {
            op: "test".into(),
            field: "download",
            expect: "dtoh".into(),
            got: format!("{e:?}"),
        })?;
        dst.copy_from_slice(&tmp[..dst.len()]);
        Ok(())
    }
    fn device_ordinal(&self) -> Result<i32, OpError> {
        Ok(0)
    }
}

fn with_runner<R>(f: impl FnOnce(&mut dyn CheckRunner) -> R) -> Option<R> {
    let mut res = TestRes::new()?;
    let mut exec = Exec::new();
    // 自检面(仅 native oracle:launch_aot 路径;foreign 需 registry 注入)
    let mut r = Runner { res: &mut res, exec: &mut exec, registry: None };
    Some(f(&mut r))
}

/// foreign oracle 面:先 boot 真注册表(全族装配),再跑 oracle(dispatch 经它)
fn with_registry_runner<R>(f: impl FnOnce(&mut dyn CheckRunner) -> R) -> Option<R> {
    let mut res = TestRes::new()?;
    let mut exec = Exec::new();
    let mut reg = owl_kernels::registry::OpRegistry::init_all(
        owl_kernels::server::family_factories(),
        &mut res,
        &mut exec,
    )
    .expect("注册表装配");
    let out = {
        let mut r = Runner { res: &mut res, exec: &mut exec, registry: Some(&mut reg) };
        f(&mut r)
    };
    let _ = reg.close();
    Some(out)
}

/// **foreign oracle**(cublas GEMM):alloc_block + dispatch + download_block 三件
#[test]
fn foreign_oracle_cublas_gemm_passes() {
    let Some(got) = with_registry_runner(|r| cb::run_case(r, &cb::case())) else {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    };
    let got = got.expect("cublas dispatch");
    let c = cb::case();
    let want = cb::reference(&c, false);
    let (worst, bad) = cb::rel_worst(&got, &want);
    println!("[oracle] cublas GEMM ✓ max|err|={worst:.3e}(相对阈值 {})", 1e-2);
    assert!(bad.is_none(), "cublas 输出越界:{bad:?}");
}

/// **foreign oracle**(marlin W4A16 AWQ):自家 packer 造输入 + dispatch +
/// 独立 host 反量化参考
#[cfg(feature = "marlin")]
#[test]
fn foreign_oracle_marlin_awq_passes() {
    let Some(got) = with_registry_runner(|r| ml::run_case(r, &ml::case())) else {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    };
    let got = got.expect("marlin dispatch");
    let c = ml::case();
    let want = ml::reference(&c, false, false);
    let (noise, max_abs, bad) = ml::noise_metrics_floor(&got, &want, ml::TOL_FLOOR);
    println!("[oracle] marlin AWQ ✓ 噪声/信号 RMS={noise:.5} max|err|={max_abs:.5}");
    assert!(bad.is_none(), "marlin 输出越界:{bad:?}");
    assert!(noise < 2e-2, "marlin 噪声比 {noise}");
}

/// **负例 ⑤**(判据敏感性):忽略零偏(zp≡8 对称域)与布局错各一
#[cfg(feature = "marlin")]
#[test]
fn marlin_oracle_is_sensitive() {
    let Some(got) = with_registry_runner(|r| ml::run_case(r, &ml::case())) else {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    };
    let got = got.expect("marlin dispatch");
    let c = ml::case();
    let (n_right, _, bad_right) =
        ml::noise_metrics_floor(&got, &ml::reference(&c, false, false), ml::TOL_FLOOR);
    let (n_sym, _, _) = ml::noise_metrics_floor(&got, &ml::reference(&c, true, false), ml::TOL_FLOOR);
    let (n_t, _, _) = ml::noise_metrics_floor(&got, &ml::reference(&c, false, true), ml::TOL_FLOOR);
    println!(
        "[oracle-neg] marlin: 正参考 RMS={n_right:.5} / 忽略零偏 {n_sym:.5} / 布局错 {n_t:.5}"
    );
    assert!(bad_right.is_none(), "正参考不该越界:{bad_right:?}");
    assert!(n_sym > 2e-2, "忽略零偏的参考竟也通过({n_sym:.5})—— 判据对 zp 不敏感!");
    assert!(n_t > 2e-2, "布局错的参考竟也通过({n_t:.5})—— 判据对布局不敏感!");
}

/// **W4A8 × AWQ 对拍**(S1d;2026-10-16):a8 三件套 packer(B 32×32 tile /
/// scales perm_single+int16 归一 / zp 无 interleave)与 int8 激活 st 路径
/// 一次性验完 —— host 参考用**原始尺度**,故全局因子折错也当场红。
#[cfg(feature = "marlin")]
#[test]
fn foreign_oracle_marlin_w4a8_passes() {
    let Some(got) = with_registry_runner(|r| ml::run_case_a8(r, &ml::case_a8(), ml::A8Neg::default()))
    else {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    };
    let got = got.expect("marlin w4a8 dispatch");
    let c = ml::case_a8();
    let want = ml::reference_a8(&c);
    let (noise, max_abs, bad) = ml::noise_metrics_floor(&got, &want, ml::TOL_FLOOR);
    println!("[oracle] marlin W4A8×AWQ ✓ 噪声/信号 RMS={noise:.5} max|err|={max_abs:.5}");
    assert!(bad.is_none(), "marlin W4A8 输出越界:{bad:?}");
    assert!(noise < ml::TOL_A8 as f64, "marlin W4A8 噪声比 {noise}");
}

/// **W4A8 判据敏感性**(两条都证明判据真能红):
/// ① 不折全局因子 ② zp 用 a16 布局(带 n-interleave)
#[cfg(feature = "marlin")]
#[test]
fn marlin_w4a8_oracle_is_sensitive() {
    let Some(got) = with_registry_runner(|r| ml::run_case_a8(r, &ml::case_a8(), ml::A8Neg::default()))
    else {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    };
    let got = got.expect("marlin w4a8 dispatch");
    let c = ml::case_a8();
    let (n_right, _, bad_right) = ml::noise_metrics_floor(&got, &ml::reference_a8(&c), ml::TOL_FLOOR);
    // ① 不折因子:同一次核输出 vs 同参考 —— 重跑一次宿主编的负例
    let no_factor = with_registry_runner(|r| {
        ml::run_case_a8(r, &ml::case_a8(), ml::A8Neg { no_factor: true, ..Default::default() })
    })
    .expect("runner")
    .expect("no_factor dispatch");
    let (n_nofac, _, _) = ml::noise_metrics_floor(&no_factor, &ml::reference_a8(&c), ml::TOL_FLOOR);
    // ② zp 全零(忽略零偏)
    let zp0 = with_registry_runner(|r| {
        ml::run_case_a8(r, &ml::case_a8(), ml::A8Neg { zp_zero: true, ..Default::default() })
    })
    .expect("runner")
    .expect("zp_zero dispatch");
    let (n_zp0, _, _) = ml::noise_metrics_floor(&zp0, &ml::reference_a8(&c), ml::TOL_FLOOR);
    // ③ B 用 a16 布局
    let b16 = with_registry_runner(|r| {
        ml::run_case_a8(r, &ml::case_a8(), ml::A8Neg { b_a16_layout: true, ..Default::default() })
    })
    .expect("runner")
    .expect("b_a16 dispatch");
    let (n_b16, _, _) = ml::noise_metrics_floor(&b16, &ml::reference_a8(&c), ml::TOL_FLOOR);
    // ④ zp 用 a16 布局(8-列组内置换 + 跨组错位)
    let zpa16 = with_registry_runner(|r| {
        ml::run_case_a8(r, &ml::case_a8(), ml::A8Neg { zp_a16_layout: true, ..Default::default() })
    })
    .expect("runner")
    .expect("zp_a16 dispatch");
    let (n_zpa16, _, _) = ml::noise_metrics_floor(&zpa16, &ml::reference_a8(&c), ml::TOL_FLOOR);
    println!(
        "[oracle-neg] marlin W4A8: 正参考 RMS={n_right:.5} / 不折因子 {n_nofac:.5} / zp 全零 {n_zp0:.5} / B-a16 布局 {n_b16:.5} / zp-a16 布局 {n_zpa16:.5}"
    );
    assert!(bad_right.is_none(), "正参考不该越界:{bad_right:?}");
    assert!(
        n_nofac > ml::TOL_A8 as f64,
        "不折全局因子的输出竟也通过({n_nofac:.5})—— 判据对 int16 尺度路径不敏感!"
    );
    assert!(
        n_zp0 > ml::TOL_A8 as f64,
        "zp 全零竟也通过({n_zp0:.5})—— 判据对 zp 不敏感!"
    );
    assert!(
        n_b16 > ml::TOL_A8 as f64,
        "B 用 a16 布局竟也通过({n_b16:.5})—— 判据对 B 布局不敏感!"
    );
    assert!(
        n_zpa16 > ml::TOL_A8 as f64,
        "zp 用 a16 布局竟也通过({n_zpa16:.5})—— 判据对 zp 布局不敏感(数据退化?)!"
    );
}

/// **负例 ④**(判据敏感性):参考按 `a·b`(布局错)⇒ 必须越界
#[test]
fn cublas_oracle_is_sensitive_to_layout() {
    let Some(got) = with_registry_runner(|r| cb::run_case(r, &cb::case())) else {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    };
    let got = got.expect("cublas dispatch");
    let c = cb::case();
    let (w_right, bad_right) = cb::rel_worst(&got, &cb::reference(&c, false));
    let (w_wrong, _) = cb::rel_worst(&got, &cb::reference(&c, true));
    println!("[oracle-neg] cublas: 正参考 max|err|={w_right:.3e} / 布局错参考 {w_wrong:.3e}");
    assert!(bad_right.is_none(), "正参考不该越界:{bad_right:?}");
    assert!(w_wrong > 1.0, "布局错参考竟也通过({w_wrong:.3e})—— 判据对 GEMM 布局不敏感!");
}

/// **正例**:两个 oracle 在生产 boot 的同一条路径上通过
#[test]
fn oracles_pass_on_real_device() {
    let Some(out) = with_runner(|r| {
        let a = k0::K0Fp8KnhdOracle.run(r);
        let b = nc::NcBf16V2Oracle.run(r);
        let c = qk::QknormInsertOracle.run(r);
        (a, b, c)
    }) else {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    };
    let (k0r, ncr, qkr) = out;
    let k0r = k0r.unwrap_or_else(|e| panic!("K0 oracle 失败: {e}"));
    let ncr = ncr.unwrap_or_else(|e| panic!("NC oracle 失败: {e}"));
    let qkr = qkr.unwrap_or_else(|e| panic!("融合插池 oracle 失败: {e}"));
    println!(
        "[oracle] K0 fp8 knHD ✓ 毒值未写字节 {:?}",
        k0r.poison_untouched
    );
    println!("[oracle] NC v2 ✓ max|err|={:.3e} ≤ {:.1e}", ncr.worst_abs_err, ncr.tolerance);
    println!(
        "[oracle] 融合插池 ✓ max|err|={:.3e} ≤ {:.1e} 毒值未写字节 {:?}",
        qkr.worst_abs_err, qkr.tolerance, qkr.poison_untouched
    );
}

/// **负例 ③**(判据敏感性):融合插池参考换成**历史 rope 配对错**
/// (`nb*cf + n*sf` 互换,2026-10-02 案)⇒ q 输出必须越界;
/// 换成 classic 池寻址 ⇒ 池字节必须对不上
#[test]
fn qknorm_insert_oracle_is_sensitive() {
    let Some(res) = with_runner(|r| qk::run_case(r, &qk::case())) else {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    };
    let (q_out, kc, _vc) = res.expect("融合插池 run");
    let c = qk::case();
    let (want_q, _want_k, _) = qk::reference(&c, false, false);
    let (want_q_swap, _, _) = qk::reference(&c, true, false);
    let (_, want_k_classic, _) = qk::reference(&c, false, true);

    let (w_right, bad_right) = qk::q_worst(&q_out, &want_q);
    let (w_swap, _) = qk::q_worst(&q_out, &want_q_swap);
    let bad_classic = want_k_classic.iter().filter(|(i, w)| kc[*i] != *w).count();
    println!(
        "[oracle-neg] 融合插池: 正参考 q max|err|={w_right:.3e} / rope 互换参考 {w_swap:.3e} / classic 池寻址不符 {bad_classic}"
    );
    assert!(bad_right.is_none(), "正参考不该越界:{bad_right:?}");
    assert!(w_swap > 2e-3, "rope 互换的参考竟也通过({w_swap:.3e})—— 判据对 rope 配对不敏感!");
    assert!(bad_classic > 0, "classic 池寻址竟也全对 —— 判据对池寻址不敏感!");
}

/// **负例 ①**(判据敏感性):NC 参考**关掉滑窗** ⇒ 必须超出容差
/// (证明判据对滑窗路径敏感 —— ③ 重写核的风险面)
#[test]
fn nc_oracle_is_sensitive_to_window() {
    let Some(got) = with_runner(|r| nc::run_case(r, &nc::case())) else {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    };
    let got = got.expect("NC run");
    let c = nc::case();
    let right = nc::host_ref(&c.q, &c.k_self, &c.v_self, &c.kc, &c.vc, c.kv_len, true);
    let wrong = nc::host_ref(&c.q, &c.k_self, &c.v_self, &c.kc, &c.vc, c.kv_len, false);
    let (w_right, bad_right) = nc::worst_err(&got, &right);
    let (w_wrong, _) = nc::worst_err(&got, &wrong);
    println!("[oracle-neg] NC: 正参考 max|err|={w_right:.3e} / 关滑窗参考 max|err|={w_wrong:.3e}");
    assert!(bad_right.is_none(), "正参考不该越界:{bad_right:?}");
    assert!(w_wrong > 2e-2, "关掉滑窗的参考竟也通过(max|err|={w_wrong:.3e})—— 判据对滑窗不敏感!");
}

/// **负例 ②**:K0 参考**换成 classic 寻址** ⇒ 字节必须对不上
/// (证明"逐字节 + 寻址"判据不是摆设)
#[test]
fn k0_oracle_is_sensitive_to_layout() {
    let Some(pools) = with_runner(|r| k0::run_case(r, &k0::case())) else {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    };
    let (kc, vc) = pools.expect("K0 run");
    let c = k0::case();
    let right = k0::reference(&c, false);
    let wrong = k0::reference(&c, true);
    let hits = |r: &[(usize, u8, u8)]| -> usize {
        r.iter().filter(|(i, wk, wv)| kc[*i] != *wk || vc[*i] != *wv).count()
    };
    let (bad_right, bad_wrong) = (hits(&right), hits(&wrong));
    println!("[oracle-neg] K0: 正参考不符 {bad_right} / classic 参考不符 {bad_wrong}");
    assert_eq!(bad_right, 0, "正参考不该有不符字节");
    assert!(bad_wrong > 0, "classic 参考竟也全对 —— 判据对寻址不敏感!");
}

/// **zp 布局判别探针**(诊断门;读数已写进设计卷 §14.3)
///
/// 判据:zp 只在一列置值 ⇒ 输出响应的列 = “内核把该 nibble 当成哪一列”。
/// 实测(2026-10-16):a8 布局 **8/8 自命中**;a16 布局 3/8 自命中,且所有错位都满足
/// **`hit % 8 == n0 % 8`**(组内位置保持、组序号 `n/8` 错位)。
#[cfg(feature = "marlin")]
#[test]
#[ignore = "诊断探针(非门);`cargo test -- --ignored --nocapture` 手动跑"]
fn probe_zp_hit_column() {
    let c = ml::case_a8();
    let g0 = 1usize;
    let cols = [8usize, 21, 37, 64, 100, 128, 200, 255];
    let Some(rows) = with_registry_runner(|r| {
        let mut v = Vec::new();
        for n0 in cols {
            let a8 = ml::zp_column_response(r, &c, n0, g0, true).expect("a8 zp 探针");
            let a16 = ml::zp_column_response(r, &c, n0, g0, false).expect("a16 zp 探针");
            let peak = |resp: &[f32]| {
                let mut best = (0usize, 0f32);
                for (i, v) in resp.iter().enumerate() {
                    if *v > best.1 {
                        best = (i, *v);
                    }
                }
                best
            };
            v.push((n0, peak(&a8), peak(&a16)));
        }
        v
    }) else {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    };
    let (mut a8_ok, mut a16_ok, mut same_mod8) = (0, 0, 0);
    for (n0, (c8, d8), (c16, d16)) in rows.iter().copied() {
        println!(
            "[zp-probe] n0={n0:3} | a8 → 命中 {c8:3} Δ={d8:7.3} | a16 → 命中 {c16:3} Δ={d16:7.3} | n0%8={} 命中%8={} n0/8={} 命中/8={}",
            n0 % 8, c16 % 8, n0 / 8, c16 / 8
        );
        if c8 == n0 { a8_ok += 1; }
        if c16 == n0 { a16_ok += 1; }
        if c16 % 8 == n0 % 8 { same_mod8 += 1; }
    }
    println!("[zp-probe] 自命中:a8 {a8_ok}/8,a16 {a16_ok}/8;错位保持 n%8:{same_mod8}/8");
    assert_eq!(a8_ok, 8, "a8 布局应每列都自命中(我们的 packer 必须与内核读法一致)");
    assert!(a16_ok < 8, "a16 布局并非处处等价(本行红 = 两条 zp 布局其实是同一张)");
    assert_eq!(
        same_mod8, 8,
        "a16 错位应恒保持 n%8(只动组序号)—— 这正是“zp 只依赖 n%8”的数据会让差异隐形的原因"
    );
}

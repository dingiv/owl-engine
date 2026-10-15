//! feature 门:本测试依赖 device(Exec/DeviceRes);无 feature 时整文件置空。
#![cfg(feature = "device")]

//! gdn_chunked **新链 E2E**:客户端面(face builder)→ 线格式 →
//! GdnChunkedRuntime(服务端面)→ 金标对拍。
//!
//! 与 tests/gdn_chunked_golden.rs(手搓发射链)不同,本测试跑的是
//! **生产 runtime 本体**(operator-contract 设计 §4.10:金标即生产镜像,
//! handler/golden 平行实现的漂移面结构性消失)。旁路战略验证件:
//! 新链全绿 + 恒等后,M4 才切 server 分派(旧 foreign.rs 全程不动)。
//!
//! 运行纪律:OWL_TEST_DEVICE + `--test-threads=1`(坑册 §16)。

use cudarc::driver::{CudaContext, CudaSlice, DevicePtr, PushKernelArg};
use owl_kernels::server::gdn_chunked::GdnChunkedRuntime;
use owl_kernels::contract::{Bytes, Dtype, LaunchMsg};
use owl_kernels::device::{DeviceRes, Exec, ScratchBuf};
use owl_kernels::client::gdn_chunked::GdnChunkedCall;
use owl_kernels::registry::KernelFamily;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

// ── 测试资源面:CudaContext 直挂(生产 server 的同构最小实现)──
struct TestRes {
    dev_ordinal: i32,
    ctx: Arc<CudaContext>,
    stream: Arc<cudarc::driver::CudaStream>,
    blocks: HashMap<u64, CudaSlice<u8>>,
    tags: HashMap<&'static str, (u64, usize)>, // tag → (id, bytes)
    next_id: u64,
    notes: std::cell::RefCell<Vec<String>>,
}

impl TestRes {
    fn new(dev: usize) -> Self {
        let ctx = CudaContext::new(dev).expect("ctx");
        ctx.bind_to_thread().expect("bind");
        let stream = ctx.default_stream();
        Self { dev_ordinal: dev as i32, ctx, stream, blocks: HashMap::new(), tags: HashMap::new(), next_id: 1, notes: RefCell::default() }
    }

    fn alloc_id(&mut self, bytes: usize) -> (u64, CudaSlice<u8>) {
        let id = self.next_id;
        self.next_id += 1;
        let d = self.stream.alloc_zeros::<u8>(bytes).expect("alloc");
        (id, d)
    }

    fn upload_new(&mut self, bytes: &[u8]) -> Bytes {
        let (id, mut d) = self.alloc_id(bytes.len());
        self.stream.memcpy_htod(bytes, &mut d).expect("htod");
        self.blocks.insert(id, d);
        Bytes::new(id, 0, Dtype::U8, 0) // 测试替身:只按 id 解析
    }

    fn slice_of(&self, id: u64) -> &CudaSlice<u8> {
        self.blocks.get(&id).expect("块在册")
    }

    #[allow(dead_code)] // 对拍辅助面(当前 case 未用,保留供扩展)
    fn take_bytes_f32(&self, id: u64) -> Vec<f32> {
        let s = self.slice_of(id);
        let host = self.stream.clone_dtoh(s).expect("dtoh");
        host.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// bf16 字节块 → f32(位型展宽)
    fn take_bytes_bf16(&self, id: u64) -> Vec<f32> {
        let s = self.slice_of(id);
        let host = self.stream.clone_dtoh(s).expect("dtoh");
        host.chunks_exact(2)
            .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect()
    }
}

impl DeviceRes for TestRes {
    fn context(&self) -> Result<Arc<CudaContext>, owl_kernels::contract::OpError> {
        Ok(self.ctx.clone())
    }
    fn stream(&self) -> Result<Arc<cudarc::driver::CudaStream>, owl_kernels::contract::OpError> {
        Ok(self.stream.clone())
    }
    fn resolve(&self, id: u64) -> Result<u64, owl_kernels::contract::OpError> {
        self.blocks.get(&id).map(|s| s.device_ptr(&self.stream).0).ok_or_else(|| {
            let mut keys: Vec<u64> = self.blocks.keys().copied().collect();
            keys.sort();
            eprintln!("[resolve miss] id={id} 在册 keys={keys:?}");
            owl_kernels::contract::OpError::Contract {
                op: "test".into(),
                field: "resolve",
                expect: "块在册".into(),
                got: id.to_string(),
            }
        })
    }
    fn alloc(&mut self, bytes: usize, tag: &'static str) -> Result<ScratchBuf, owl_kernels::contract::OpError> {
        let (id, d) = self.alloc_id(bytes);
        let ptr = d.device_ptr(&self.stream).0;
        self.blocks.insert(id, d);
        self.tags.insert(tag, (id, bytes));
        Ok(ScratchBuf { ptr, bytes })
    }

    fn capturing(&self) -> Result<bool, owl_kernels::contract::OpError> {
        Ok(false)
    }
    fn resolve_len(&self, id: u64) -> Result<usize, owl_kernels::contract::OpError> {
        self.blocks.get(&id).map(|s| s.len()).ok_or_else(|| {
            owl_kernels::contract::OpError::Contract {
                op: "test".into(),
                field: "resolve_len",
                expect: "块在册".into(),
                got: id.to_string(),
            }
        })
    }
    fn record_capture(&mut self, note: owl_kernels::device::LaunchNote) -> Result<(), owl_kernels::contract::OpError> {
        self.notes.borrow_mut().push(format!("{}/{}", note.op, note.kernel));
        Ok(())
    }
    fn device_ordinal(&self) -> Result<i32, owl_kernels::contract::OpError> {
        Ok(self.dev_ordinal)
    }
    fn upload(&mut self, dst: u64, src: &[u8]) -> Result<(), owl_kernels::contract::OpError> {
        let target = self
            .blocks
            .iter()
            .find(|(_, s)| s.device_ptr(&self.stream).0 == dst)
            .map(|(id, _)| *id);
        if let Some(id) = target {
            let view = self.blocks.get_mut(&id).expect("在册");
            self.stream.memcpy_htod(src, view).expect("htod");
            return Ok(());
        }
        Err(owl_kernels::contract::OpError::Contract {
            op: "test".into(), field: "upload", expect: "指针在册".into(), got: format!("{dst:#x}"),
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
        tag: &'static str,
    ) -> Result<owl_kernels::contract::Bytes, owl_kernels::contract::OpError> {
        // 测试面:走裸分配 + 在册登记(resolve 可解析)
        let s = unsafe { self.stream.alloc::<u8>(bytes.max(1)) }
            .map_err(|e| owl_kernels::contract::OpError::Contract {
                op: "test".into(),
                field: "alloc_block",
                expect: "-".into(),
                got: format!("{e:?}"),
            })?;
        let id = self.next_id;
        self.next_id += 1;
        self.blocks.insert(id, s);
        let _ = tag;
        Ok(owl_kernels::contract::Bytes::new(id, bytes, Dtype::U8, 0))
    }
    fn download(
        &mut self,
        _src: u64,
        _dst: &mut [u8],
    ) -> Result<(), owl_kernels::contract::OpError> {
        Err(owl_kernels::contract::OpError::Contract {
            op: "test".into(),
            field: "download",
            expect: "-".into(),
            got: "-".into(),
        })
    }

}

// ── 金标读取 + host 铸造(与金标测试同款)──

const NK: usize = 16;
const NV: usize = 48;
const KD: usize = 128;
const VD: usize = 128;

fn read_bin(dir: &str, name: &str) -> Vec<f32> {
    let p = format!("{dir}/{name}.bin");
    let bytes = std::fs::read(&p).unwrap_or_else(|e| panic!("{p}: {e}"));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[allow(dead_code)] // 对拍辅助面(当前 case 未用,保留)
fn f32_to_bf16(src: &[f32]) -> Vec<u8> {
    src.iter()
        .flat_map(|f| half::bf16::from_f32(*f).to_le_bytes())
        .collect()
}

fn f32_to_f16_bytes(src: &[f32]) -> Vec<u8> {
    src.iter()
        .flat_map(|f| half::f16::from_f32(*f).to_le_bytes())
        .collect()
}

/// [HV, X, Y] → [HV, Y, X](金标代际布局换算)
#[allow(dead_code)] // 对拍辅助面(当前 case 未用,保留)
fn transpose_kv(src: &[f32], hv: usize, x: usize, y: usize) -> Vec<f32> {
    let mut out = vec![0f32; src.len()];
    for hi in 0..hv {
        for xi in 0..x {
            for yi in 0..y {
                out[hi * x * y + yi * x + xi] = src[hi * x * y + xi * y + yi];
            }
        }
    }
    out
}

fn transpose_kv_nt(src: &[f32], nt: usize, hv: usize, x: usize, y: usize) -> Vec<f32> {
    let mut out = vec![0f32; src.len()];
    for ti in 0..nt {
        let base = ti * hv * x * y;
        for hi in 0..hv {
            for xi in 0..x {
                for yi in 0..y {
                    out[base + hi * x * y + yi * x + xi] = src[base + hi * x * y + xi * y + yi];
                }
            }
        }
    }
    out
}

fn max_dev(got: &[f32], want: &[f32]) -> f32 {
    got.iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

    /// 诊断:scratch 槽位读回(f32 视角)+ 统计
    fn scratch_stats(res: &TestRes, tag: &str, n: usize) -> (usize, usize, f32, f32) {
        let (id, bytes) = res.tags.get(tag).copied().unwrap_or((0, 0));
        let n = n.min(bytes / 4);
        if id == 0 { return (0, 0, 0.0, 0.0); }
        let s = res.blocks.get(&id).expect("在册");
        let host = res.stream.clone_dtoh(s).expect("dtoh");
        let mut nan = 0; let mut inf = 0;
        let mut mn = f32::INFINITY; let mut mx = f32::NEG_INFINITY;
        for c in host.chunks_exact(4).take(n) {
            let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            if v.is_nan() { nan += 1; } else if v.is_infinite() { inf += 1; }
            else { mn = mn.min(v); mx = mx.max(v); }
        }
        (nan, inf, mn, mx)
    }

    /// 诊断:scratch 槽位读回(bf16 视角)
    fn scratch_stats_bf16(res: &TestRes, tag: &str, n: usize) -> (usize, usize, f32, f32) {
        let (id, bytes) = res.tags.get(tag).copied().unwrap_or((0, 0));
        let n = n.min(bytes / 2);
        if id == 0 { return (0, 0, 0.0, 0.0); }
        let s = res.blocks.get(&id).expect("在册");
        let host = res.stream.clone_dtoh(s).expect("dtoh");
        let mut nan = 0; let mut inf = 0;
        let mut mn = f32::INFINITY; let mut mx = f32::NEG_INFINITY;
        for c in host.chunks_exact(2).take(n) {
            let v = half::bf16::from_le_bytes([c[0], c[1]]).to_f32();
            if v.is_nan() { nan += 1; } else if v.is_infinite() { inf += 1; }
            else { mn = mn.min(v); mx = mx.max(v); }
        }
        (nan, inf, mn, mx)
    }

#[test]
#[allow(clippy::too_many_lines)]
fn gdn_chunked_new_chain_matches_golden() {
    if !std::env::var_os("OWL_TEST_DEVICE").is_some() {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    }
    let dev: usize = std::env::var("OWL_TEST_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let mut res = TestRes::new(dev);
    let mut exec = Exec::new();
    let mut rt = <GdnChunkedRuntime as KernelFamily>::assemble(&mut res, &mut exec).expect("runtime assemble");

    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/gdn_fla/cases");
    // (tag, T, tight)
    let cases: &[(&str, usize, bool)] = &[
        ("c1_t64", 64, false),
        ("c2_t128", 128, false),
        ("c5_t128_deep", 128, true),
    ];

    let rcp_ln2 = std::f32::consts::LOG2_E;
    for (tag, t, tight) in cases {
        let nt = t.div_ceil(64);
        let q = read_bin(dir, &format!("{tag}.q"));
        let k = read_bin(dir, &format!("{tag}.k"));
        let v = read_bin(dir, &format!("{tag}.v"));
        let beta = read_bin(dir, &format!("{tag}.beta"));
        let g = read_bin(dir, &format!("{tag}.g"));
        let s0 = std::fs::metadata(format!("{dir}/{tag}.s0.bin")).is_ok();
        let golden_o = read_bin(dir, &format!("{tag}.o"));

        // ── face builder(host 铸造:bf16×4 + f16 gate;引擎配方)──
        let shape = owl_kernels::client::gdn_chunked::GdnShape {
            t: *t as u32, nv: NV as u32, nk: NK as u32, kd: KD as u32,
        };
        // ⚠️ 引擎输入契约:层侧 q/k/v/beta = **f16**(runtime cast-in 再铸
        // bf16/f32);喂 bf16 = 位型错读(±1.453 = 0.1 的 bf16 位当 f16)
        let q_b = res.upload_new(&f32_to_f16_bytes(&q));
        let k_b = res.upload_new(&f32_to_f16_bytes(&k));
        let v_b = res.upload_new(&f32_to_f16_bytes(&v));
        let beta_b = res.upload_new(&f32_to_f16_bytes(&beta));
        let g_b = res.upload_new(&f32_to_f16_bytes(&g));
        // state:**池布局 [K,V] 原始字节** —— runtime 的 trans-in(kv_to_vk)
        // 自己做 [K,V]→[V,K](引擎池同款);测试预转置 = 转两次(深谷案首证)
        let s0v = if s0 {
            read_bin(dir, &format!("{tag}.s0"))
        } else {
            vec![0f32; NV * KD * VD]
        };
        let mut s0_bytes = Vec::with_capacity(s0v.len() * 4);
        for f in &s0v { s0_bytes.extend_from_slice(&f.to_le_bytes()); }
        let state_b = res.upload_new(&s0_bytes);
        let (out_id, out_slice) = res.alloc_id(t * NV * VD * 2);
        res.blocks.insert(out_id, out_slice); // 直取路径必须自己落表(upload 路径内含)
        let out_b = Bytes::new(out_id, t * NV * VD, Dtype::F16, 3); // [t, NV, VD] f16

        let call = GdnChunkedCall::builder(shape)
            .q(&q_b).unwrap()
            .k(&k_b).unwrap()
            .v(&v_b).unwrap()
            .beta(&beta_b).unwrap()
            .gate_raw(&g_b).unwrap()
            .state(&state_b, 0).unwrap()
            .build().unwrap();
        let msg: LaunchMsg = call.to_launch(out_b.clone());

        // ── 新链执行(run;Result 全链)──
        rt.run(&msg, &mut res, &mut exec).expect("runtime run");
        res.stream.synchronize().expect("sync");

        // ── 对拍(o;宽/紧分级同金标测试)──
        {
            let s = res.slice_of(out_id);
            let host = res.stream.clone_dtoh(s).expect("dtoh");
            let f16s: Vec<u16> = host.chunks_exact(2).take(8).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            let f16s_all: Vec<u16> = host.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            let nz = f16s_all.iter().filter(|v| **v != 0).count();
            eprintln!("[{tag}] out_id={out_id} 前8 f16 位型={f16s:x?} 非零总数={nz}/{}", f16s_all.len());
        }
        let (oid, _ob) = res.tags.get("gdn.o_b16").copied().expect("o_b16 tag");
        let got = res.take_bytes_bf16(oid);
        let (atol, rtol) = if *tight { (1e-4f32, 1e-3f32) } else { (5e-2f32, 6e-2f32) };
        let (mut nan_g, mut inf_g, mut nan_w) = (0usize, 0usize, 0usize);
        let (mut gmax, mut wmax) = (0f32, 0f32);
        for (a, b) in got.iter().zip(&golden_o) {
            if a.is_nan() { nan_g += 1; }
            if a.is_infinite() { inf_g += 1; }
            if b.is_nan() { nan_w += 1; }
            if !a.is_nan() { gmax = gmax.max(a.abs()); }
            wmax = wmax.max(b.abs());
        }
        let mut worst = 0f32;
        for (a, b) in got.iter().zip(&golden_o) {
            if a.is_nan() || a.is_infinite() { worst = f32::INFINITY; break; }
            let n = (a - b).abs() / (atol + rtol * b.abs());
            worst = worst.max(n);
        }
        eprintln!("[{tag}] 新链 o 对拍: ×{worst:.2} gotNaN={nan_g} gotInf={inf_g} wantNaN={nan_w} gotMax={gmax:.4} wantMax={wmax:.4}(nt={nt})");
        // ── 阶段验尸(scratch 逐槽统计;f32 槽 vs bf16 槽)──
        let (tn, ti_, tmn, tmx) = scratch_stats(&res,"gdn.in_g", t * NV);
        eprintln!("  in_g(f32): nan={tn} inf={ti_} [{tmn:.3},{tmx:.3}](期望 ≈ g 域)");
        let (tn, ti_, tmn, tmx) = scratch_stats(&res,"gdn.g_cum", t * NV);
        eprintln!("  g_cum(f32): nan={tn} inf={ti_} [{tmn:.3},{tmx:.3}](期望全负)");
        let (tn, ti_, tmn, tmx) = scratch_stats_bf16(&res,"gdn.in_q", t * NK * KD);
        eprintln!("  in_q(bf16): nan={tn} inf={ti_} [{tmn:.3},{tmx:.3}](期望 q 域)");
        let (tn, ti_, tmn, tmx) = scratch_stats_bf16(&res,"gdn.w", t * NV * KD);
        eprintln!("  w(bf16): nan={tn} inf={ti_} [{tmn:.3},{tmx:.3}]");
        let (tn, ti_, tmn, tmx) = scratch_stats_bf16(&res,"gdn.o_b16", t * NV * VD);
        eprintln!("  o_b16(bf16): nan={tn} inf={ti_} [{tmn:.3},{tmx:.3}]");
        let (tn, ti_, tmn, tmx) = scratch_stats(&res, "gdn.a_buf", t * NV * 64);
        eprintln!("  A(f32): nan={tn} inf={ti_} [{tmn:.3},{tmx:.3}](期望 |A|≲0.5)");
        let (tn, ti_, tmn, tmx) = scratch_stats_bf16(&res, "gdn.ai_buf", t * NV * 64);
        eprintln!("  Ai(bf16): nan={tn} inf={ti_} [{tmn:.3},{tmx:.3}](期望 |Ai|≲0.5)");
        let (tn, ti_, tmn, tmx) = scratch_stats_bf16(&res, "gdn.in_k", t * NK * KD);
        eprintln!("  in_k(bf16): nan={tn} inf={ti_} [{tmn:.3},{tmx:.3}]");
        let (tn, ti_, tmn, tmx) = scratch_stats_bf16(&res, "gdn.in_beta", t * NV);
        eprintln!("  in_beta(bf16): nan={tn} inf={ti_} [{tmn:.3},{tmx:.3}](期望 (0,1))");
        assert!(worst <= 1.0, "{tag}: 新链 o 超容差 ×{worst:.2}");
        let _ = (max_dev, transpose_kv_nt, take_f32_unused, rcp_ln2, &golden_o);
    }
}

fn take_f32_unused() {}

/// **NT 分解恒等门(2026-10-14 A13 立案;补金标 NT≤2 的覆盖缺口)**
///
/// 判据:同一输入,T=512(NT=8)**一次调用** ≡ **8 次 T=64 链式调用**
/// (chunked delta rule 对 BT 边界可分解:态在池块里原地交接;金标
/// `gdn_chunked_golden` 只到 T=128 = NT≤2,NT≥3 从未被任何门覆盖)。
///
/// 生产症状 = 这条恒等在 NT≥3 处破:引擎里 `prefill_chunk=512` 塌陷、
/// `=64/128/192/256` 正常(同一 chunked 臂),varlen 臂恒正常。
/// 本门只跑算子链(1 秒,零模型装载)⇒ 破/不破一眼可判,且能逐阶段定位
/// (h_buf / v_new / w / o 全在 tags 表里可取)。
#[test]
#[allow(clippy::too_many_lines)]
fn gdn_chunked_nt_decomposition_identity() {
    if std::env::var_os("OWL_TEST_DEVICE").is_none() {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    }
    let dev: usize = std::env::var("OWL_TEST_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let mut res = TestRes::new(dev);
    let mut exec = Exec::new();
    let mut rt = <GdnChunkedRuntime as KernelFamily>::assemble(&mut res, &mut exec).expect("runtime assemble");

    const T_BIG: usize = 512;
    const T_SM: usize = 64;
    const NT: usize = T_BIG / T_SM;

    // ── 合成输入(确定性 LCG;生产量级 + 深谷 g 域)──
    let lcg = |seed: u32, i: usize| -> f32 {
        let x = seed
            .wrapping_mul(0x9E37_79B9)
            .wrapping_add((i as u32).wrapping_mul(0x85EB_CA6B));
        ((x >> 8) as f32) / 16777216.0
    };
    let g_depth = std::env::var("OWL_NT_G_DEPTH").ok().and_then(|v| v.parse().ok()).unwrap_or(8.0f32);
    let n_qk = T_BIG * NK * KD;
    let n_v = T_BIG * NV * VD;
    let n_h = T_BIG * NV;
    let mut q = vec![0f32; n_qk];
    let mut k = vec![0f32; n_qk];
    let mut v = vec![0f32; n_v];
    let mut beta = vec![0f32; n_h];
    let mut g = vec![0f32; n_h];
    for i in 0..n_qk {
        q[i] = (lcg(11, i) - 0.5) * 0.2;
        k[i] = (lcg(23, i) - 0.5) * 0.2;
    }
    for i in 0..n_v {
        v[i] = (lcg(37, i) - 0.5) * 0.2;
    }
    for i in 0..n_h {
        beta[i] = 0.1 + 0.8 * lcg(53, i);
        g[i] = -(0.01 + g_depth * lcg(71, i));
    }
    // **生产量级对齐**(2026-10-14):层侧喂进 chunked 臂的是 **l2norm 后**的
    // q_n/k_n(|·|=1)+ conv 后 v_c,状态量级 ~30;合成输入若不做这一步,
    // 恒等门会「因为量级太小而空转」(首版实测 |o|max=1e-23)
    let normalize_rows = |xs: &mut Vec<f32>, rows: usize, heads: usize, d: usize| {
        for r in 0..rows {
            for h in 0..heads {
                let base = r * heads * d + h * d;
                let n: f32 = xs[base..base + d].iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
                for i in base..base + d {
                    xs[i] /= n;
                }
            }
        }
    };
    normalize_rows(&mut q, T_BIG, NK, KD);
    normalize_rows(&mut k, T_BIG, NK, KD);
    let v_scale: f32 = std::env::var("OWL_NT_VSCALE").ok().and_then(|x| x.parse().ok()).unwrap_or(2.0);
    for x in v.iter_mut() {
        *x *= v_scale;
    }
    let to_f16 = |xs: &[f32]| -> Vec<u8> {
        xs.iter().flat_map(|x| half::f16::from_f32(*x).to_le_bytes()).collect()
    };

    // ── 输入块(整块 + 8 段切片各一套)──
    let mut run_arm = |rt: &mut Box<dyn KernelFamily>, res: &mut TestRes, exec: &mut Exec,
                       t: usize, slice: Option<usize>| -> (Vec<f32>, Vec<f32>) {
        // slice = Some(i) → 取第 i 段 T_SM 行;None → 整块 T_BIG 行
        let (off_qk, rows_qk, off_v, rows_v, off_h, rows_h) = match slice {
            None => (0, t, 0, t, 0, t),
            Some(i) => (
                i * T_SM * NK * KD, t,
                i * T_SM * NV * VD, t,
                i * T_SM * NV, t,
            ),
        };
        let cut = |xs: &[f32], off: usize, rows: usize, per_row: usize| -> Vec<u8> {
            to_f16(&xs[off..off + rows * per_row])
        };
        let q_b = res.upload_new(&cut(&q, off_qk, rows_qk, NK * KD));
        let k_b = res.upload_new(&cut(&k, off_qk, rows_qk, NK * KD));
        let v_b = res.upload_new(&cut(&v, off_v, rows_v, NV * VD));
        let beta_b = res.upload_new(&cut(&beta, off_h, rows_h, NV));
        let g_b = res.upload_new(&cut(&g, off_h, rows_h, NV));
        let state_b = res.upload_new(&vec![0u8; NV * KD * VD * 4]); // 每臂独立起点(零态)
        let (out_id, out_slice) = res.alloc_id(t * NV * VD * 2);
        res.blocks.insert(out_id, out_slice);
        let out_b = Bytes::new(out_id, t * NV * VD, Dtype::F16, 3); // [t, NV, VD] f16
        let shape = owl_kernels::client::gdn_chunked::GdnShape {
            t: t as u32, nv: NV as u32, nk: NK as u32, kd: KD as u32,
        };
        let call = GdnChunkedCall::builder(shape)
            .q(&q_b).unwrap().k(&k_b).unwrap().v(&v_b).unwrap()
            .beta(&beta_b).unwrap().gate_raw(&g_b).unwrap()
            .state(&state_b, 0).unwrap()
            .build().unwrap();
        let msg: LaunchMsg = call.to_launch(out_b.clone());
        rt.run(&msg, res, exec).expect("runtime run");
        res.stream.synchronize().expect("sync");
        let o = res.take_bytes_bf16(out_id);
        // 态回读(池布局 [K,V] f32;handler 已转置写回同块)
        let st = res.take_bytes_f32(state_b.id);
        (o, st)
    };

    // arm A:一次 T=512(NT=8)
    let (o_a, st_a) = run_arm(&mut rt, &mut res, &mut exec, T_BIG, None);
    let st_a_snapshot = st_a.clone();
    let (w_a, vn_a) = (res.tags.get("gdn.w").copied(), res.tags.get("gdn.v_new").copied());

    // arm B:8 次 T=64(NT=1),**态链式**(每次调用写回同一 state 块 ⇒ 直接复用)
    // ⚠️ 为让链式生效,必须用**同一个 state 块**:见下面的循环版本
    let mut o_b = Vec::with_capacity(T_BIG * NV * VD);
    let mut st_b = vec![0f32; NV * KD * VD];
    {
        // 重建输入(与 arm A 同源切片),但 state 块跨调用共享
        let qs: Vec<Bytes> = (0..NT).map(|i| res.upload_new(&to_f16(&q[i * T_SM * NK * KD..(i + 1) * T_SM * NK * KD]))).collect();
        let ks: Vec<Bytes> = (0..NT).map(|i| res.upload_new(&to_f16(&k[i * T_SM * NK * KD..(i + 1) * T_SM * NK * KD]))).collect();
        let vs: Vec<Bytes> = (0..NT).map(|i| res.upload_new(&to_f16(&v[i * T_SM * NV * VD..(i + 1) * T_SM * NV * VD]))).collect();
        let bs: Vec<Bytes> = (0..NT).map(|i| res.upload_new(&to_f16(&beta[i * T_SM * NV..(i + 1) * T_SM * NV]))).collect();
        let gs: Vec<Bytes> = (0..NT).map(|i| res.upload_new(&to_f16(&g[i * T_SM * NV..(i + 1) * T_SM * NV]))).collect();
        let state_b = res.upload_new(&vec![0u8; NV * KD * VD * 4]);
        let shape = owl_kernels::client::gdn_chunked::GdnShape {
            t: T_SM as u32, nv: NV as u32, nk: NK as u32, kd: KD as u32,
        };
        for i in 0..NT {
            let (out_id, out_slice) = res.alloc_id(T_SM * NV * VD * 2);
            res.blocks.insert(out_id, out_slice);
            let out_b = Bytes::new(out_id, T_SM * NV * VD, Dtype::F16, 3); // [T_SM, NV, VD] f16
            let call = GdnChunkedCall::builder(shape)
                .q(&qs[i]).unwrap().k(&ks[i]).unwrap().v(&vs[i]).unwrap()
                .beta(&bs[i]).unwrap().gate_raw(&gs[i]).unwrap()
                .state(&state_b, 0).unwrap()
                .build().unwrap();
            let msg: LaunchMsg = call.to_launch(out_b.clone());
            rt.run(&msg, &mut res, &mut exec).expect("runtime run");
            res.stream.synchronize().expect("sync");
            o_b.extend(res.take_bytes_bf16(out_id));
        }
        st_b = res.take_bytes_f32(state_b.id);
    }

    // ── 对拍 ──
    let (mut worst_o, mut at_o) = (0f32, 0usize);
    for (i, (a, b)) in o_a.iter().zip(&o_b).enumerate() {
        let d = (a - b).abs();
        if d > worst_o { worst_o = d; at_o = i; }
    }
    let (mut worst_s, mut at_s) = (0f32, 0usize);
    for (i, (a, b)) in st_a_snapshot.iter().zip(&st_b).enumerate() {
        let d = (a - b).abs();
        if d > worst_s { worst_s = d; at_s = i; }
    }
    let omax = o_a.iter().fold(0f32, |m, x| m.max(x.abs()));
    eprintln!(
        "[nt-ident] T={T_BIG}(NT={NT}) g_depth={g_depth}\n  \
         o: max|Δ|={worst_o:.3e} @{at_o} (|o|max={omax:.3e})\n  \
         state: max|Δ|={worst_s:.3e} @{at_s} (|st|max={:.3e})",
        st_a_snapshot.iter().fold(0f32, |m, x| m.max(x.abs()))
    );
    if let Some((id, n)) = w_a { let _ = n; let _ = id; }
    if let Some((id, n)) = vn_a { let _ = n; let _ = id; }
    let tol = std::env::var("OWL_NT_TOL").ok().and_then(|v| v.parse().ok()).unwrap_or(2e-2f32);
    assert!(
        worst_o <= tol * omax.max(1.0),
        "NT 分解恒等门破:o max|Δ|={worst_o:.3e} > tol={tol}(|o|max={omax:.3e})—— NT≥3 的链内多 chunk 路径错(A13)"
    );
}

/// **函数级双臂对拍(2026-10-14 A13):生产 handler 六核链 vs varlen 递推核**
///
/// 为什么(用户点破):两臂**数学等价**(同一个 gated delta rule),差别只在
/// 数值路径(链:bf16 中间张量 + chunk 总量一次 `exp`;varlen:f16 进/f32 累加 +
/// 逐 token `expf(g)`)。引擎里 chunk=512 时 chunked 塌陷、varlen 健康 ⇒ 若两臂
/// 在**同一输入**下函数级就分歧,即拿到最小触发器;若一致,则病灶在引擎侧
/// (输入/scratch/生命周期),不在链。
///
/// 口径(三处必须对齐,否则假红):
///   * 态:池布局 `[slot,NV,K,V]` f32 **同字节直喂两臂**(链内部自转 [V,K]);
///   * 输入:q/k **l2norm 后**(|·|=1,层侧真实口径)+ v 生产量级 + f16 位型;
///   * 判据:两臂精度路径不同 ⇒ **不逐位**,按相对信号 RMS + max|Δ| 报点。
#[test]
#[allow(clippy::too_many_lines)]
fn gdn_chunked_matches_varlen_fn_level() {
    if std::env::var_os("OWL_TEST_DEVICE").is_none() {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    }
    let dev: usize = std::env::var("OWL_TEST_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let mut res = TestRes::new(dev);
    let mut exec = Exec::new();
    let mut rt = <GdnChunkedRuntime as KernelFamily>::assemble(&mut res, &mut exec).expect("runtime assemble");

    // varlen 核(nvrtc;与 gdn_big_t_arms_parity 同款发射)
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(
        owl_kernels::cu_sig::cu_text(owl_kernels::ops::source("owl_gdn_recurrence_varlen_gqa_f16")).as_str(),
        cudarc::nvrtc::CompileOptions { arch: Some("compute_86"), include_paths: vec!["/usr/local/cuda/include".into()], ..Default::default() },
    ).expect("nvrtc gdn_recurrence");
    let m_var = res.ctx.load_module(ptx).expect("module");
    let f_var = m_var.load_function("owl_gdn_recurrence_varlen_gqa_f16").expect("fn varlen");
    let scale: f32 = 1.0 / (KD as f32).sqrt();

    let lcg = |seed: u32, i: usize| -> f32 {
        let x = seed.wrapping_mul(0x9E37_79B9).wrapping_add((i as u32).wrapping_mul(0x85EB_CA6B));
        ((x >> 8) as f32) / 16777216.0
    };
    let g_depth: f32 = std::env::var("OWL_XARM_G").ok().and_then(|v| v.parse().ok()).unwrap_or(9.0);
    let cases: Vec<usize> = std::env::var("OWL_XARM_T")
        .ok()
        .and_then(|v| v.split(',').map(|s| s.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![64, 128, 256, 512]);

    for t in cases {
        // ── 生产量级输入(q/k 逐头 l2norm)──
        let mut q: Vec<f32> = (0..t * NK * KD).map(|i| (lcg(11, i) - 0.5) * 0.2).collect();
        let mut k: Vec<f32> = (0..t * NK * KD).map(|i| (lcg(23, i) - 0.5) * 0.2).collect();
        let v: Vec<f32> = (0..t * NV * VD).map(|i| (lcg(37, i) - 0.5) * 2.0).collect();
        let beta: Vec<f32> = (0..t * NV).map(|i| 0.1 + 0.8 * lcg(53, i)).collect();
        let g: Vec<f32> = (0..t * NV).map(|i| -(0.01 + g_depth * lcg(71, i))).collect();
        for (xs, heads) in [(&mut q, NK), (&mut k, NK)] {
            for r in 0..t {
                for h in 0..heads {
                    let b = r * heads * KD + h * KD;
                    let n: f32 = xs[b..b + KD].iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
                    for i in b..b + KD {
                        xs[i] /= n;
                    }
                }
            }
        }
        let f16b = |xs: &[f32]| -> Vec<u8> { xs.iter().flat_map(|x| half::f16::from_f32(*x).to_le_bytes()).collect() };
        let (qb, kb, vb, bb, gb) = (f16b(&q), f16b(&k), f16b(&v), f16b(&beta), f16b(&g));

        // 初态(生产里第二块起是非零态;两种都测)
        let st0: Vec<f32> = match std::env::var("OWL_XARM_ST0").as_deref() {
            Ok("rand") => (0..NV * KD * VD).map(|i| (lcg(97, i) - 0.5) * 2.0).collect(),
            _ => vec![0f32; NV * KD * VD],
        };
        let st0_bytes: Vec<u8> = st0.iter().flat_map(|x| x.to_le_bytes()).collect();

        // ── 臂 A:生产 handler 六核链 ──
        let q_b = res.upload_new(&qb);
        let k_b = res.upload_new(&kb);
        let v_b = res.upload_new(&vb);
        let beta_b = res.upload_new(&bb);
        let g_b = res.upload_new(&gb);
        let state_b = res.upload_new(&st0_bytes);
        let (out_id, out_slice) = res.alloc_id(t * NV * VD * 2);
        res.blocks.insert(out_id, out_slice);
        let out_b = Bytes::new(out_id, t * NV * VD, Dtype::F16, 3); // [t, NV, VD] f16
        let shape = owl_kernels::client::gdn_chunked::GdnShape { t: t as u32, nv: NV as u32, nk: NK as u32, kd: KD as u32 };
        let call = GdnChunkedCall::builder(shape)
            .q(&q_b).unwrap().k(&k_b).unwrap().v(&v_b).unwrap()
            .beta(&beta_b).unwrap().gate_raw(&g_b).unwrap()
            .state(&state_b, 0).unwrap()
            .build().unwrap();
        let msg: LaunchMsg = call.to_launch(out_b.clone());
        rt.run(&msg, &mut res, &mut exec).expect("chain run");
        res.stream.synchronize().expect("sync");
        let o_chain: Vec<f32> = {
            let host = res.stream.clone_dtoh(res.slice_of(out_id)).expect("dtoh");
            host.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect()
        };
        let st_chain = res.take_bytes_f32(state_b.id);

        // ── 臂 B:varlen(同输入/同初态字节)──
        let (mut d_q, mut d_k, mut d_v, mut d_b, mut d_g) = (
            unsafe { res.stream.alloc::<u8>(qb.len()) }.unwrap(),
            unsafe { res.stream.alloc::<u8>(kb.len()) }.unwrap(),
            unsafe { res.stream.alloc::<u8>(vb.len()) }.unwrap(),
            unsafe { res.stream.alloc::<u8>(bb.len()) }.unwrap(),
            unsafe { res.stream.alloc::<u8>(gb.len()) }.unwrap(),
        );
        res.stream.memcpy_htod(&qb, &mut d_q).unwrap();
        res.stream.memcpy_htod(&kb, &mut d_k).unwrap();
        res.stream.memcpy_htod(&vb, &mut d_v).unwrap();
        res.stream.memcpy_htod(&bb, &mut d_b).unwrap();
        res.stream.memcpy_htod(&gb, &mut d_g).unwrap();
        let mut d_st = unsafe { res.stream.alloc::<f32>(NV * KD * VD) }.unwrap();
        res.stream.memcpy_htod(&st0, &mut d_st).unwrap();
        let d_slots = res.stream.alloc_zeros::<f32>(1).unwrap();
        let cu_host = [0.0f32, t as f32];
        let mut d_cu = res.stream.alloc_zeros::<f32>(2).unwrap();
        res.stream.memcpy_htod(&cu_host, &mut d_cu).unwrap();
        let d_out = unsafe { res.stream.alloc::<u16>(t * NV * VD) }.unwrap();
        {
            let (qp, kp, vp, gp, bp) = (
                d_q.device_ptr(&res.stream).0, d_k.device_ptr(&res.stream).0,
                d_v.device_ptr(&res.stream).0, d_g.device_ptr(&res.stream).0,
                d_b.device_ptr(&res.stream).0,
            );
            let (stp, slp, cup, op) = (
                d_st.device_ptr(&res.stream).0, d_slots.device_ptr(&res.stream).0,
                d_cu.device_ptr(&res.stream).0, d_out.device_ptr(&res.stream).0,
            );
            let (batch_u, nv_u, nk_u, kd_u, vd_u) = (1u64, NV as u64, NK as u64, KD as u64, VD as u64);
            let mut b = res.stream.launch_builder(&f_var);
            b.arg(&qp).arg(&kp).arg(&vp).arg(&gp).arg(&bp)
                .arg(&stp).arg(&slp).arg(&cup)
                .arg(&batch_u).arg(&nv_u).arg(&nk_u).arg(&kd_u).arg(&vd_u)
                .arg(&scale).arg(&op);
            unsafe {
                b.launch(cudarc::driver::LaunchConfig {
                    grid_dim: (VD.div_ceil(8) as u32, NV as u32, 1),
                    block_dim: (32, 8, 1),
                    shared_mem_bytes: ((4 * KD + 4) * 4) as u32,
                })
            }
            .expect("varlen launch");
        }
        res.stream.synchronize().expect("sync");
        let mut o_var16 = vec![0u16; t * NV * VD];
        res.stream.memcpy_dtoh(&d_out, &mut o_var16).unwrap();
        let o_var: Vec<f32> = o_var16.iter().map(|&x| half::f16::from_bits(x).to_f32()).collect();
        let mut st_var = vec![0f32; NV * KD * VD];
        res.stream.memcpy_dtoh(&d_st, &mut st_var).unwrap();

        // ── 对拍(相对信号 RMS + max|Δ|)──
        let dev_of = |a: &[f32], b: &[f32]| -> (f64, f32, f32) {
            let (mut sa, mut sq, mut mx) = (0f64, 0f64, 0f32);
            for (x, y) in a.iter().zip(b) {
                sa += (*x as f64 - *y as f64).abs();
                sq += (*y as f64) * (*y as f64);
                mx = mx.max((*x - *y).abs());
            }
            let rms = (sq / a.len() as f64).sqrt().max(1e-9);
            (sa / a.len() as f64 / rms, mx, rms as f32)
        };
        let (ro, mo, rmso) = dev_of(&o_chain, &o_var);
        let (rs, ms, rmss) = dev_of(&st_chain, &st_var);
        eprintln!(
            "[x-arm] T={t:3} NT={:2} g_depth={g_depth} st0={} | o: 噪声/RMS={ro:.4} max|Δ|={mo:.3e}(信号RMS={rmso:.3e}) | 终态: 噪声/RMS={rs:.4} max|Δ|={ms:.3e}(RMS={rmss:.3e})",
            t.div_ceil(64),
            if st0.iter().any(|x| *x != 0.0) { "rand" } else { "zero" },
        );
        let tol: f64 = std::env::var("OWL_XARM_TOL").ok().and_then(|v| v.parse().ok()).unwrap_or(5e-2);
        assert!(ro <= tol && rs <= tol, "[x-arm] T={t} 两臂函数级分歧:o 噪声/RMS={ro:.4} 终态={rs:.4} > tol={tol}");
    }
}

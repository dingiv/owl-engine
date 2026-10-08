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

use cudarc::driver::{CudaContext, CudaSlice, DevicePtr};
use owl_kernels::server::gdn_chunked::GdnChunkedRuntime;
use owl_kernels::contract::{Bytes, LaunchMsg, OpId};
use owl_kernels::device::{DeviceRes, Exec, LaunchVal, ScratchBuf};
use owl_kernels::client::gdn_chunked::GdnChunkedCall;
use owl_kernels::registry::{FamilyRuntime, RunEnv};
use std::cell::RefCell;
use std::collections::HashMap;
use cudarc::driver::DevicePtrMut;
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
        let d = unsafe { self.stream.alloc_zeros::<u8>(bytes) }.expect("alloc");
        (id, d)
    }

    fn upload_new(&mut self, bytes: &[u8]) -> Bytes {
        let (id, mut d) = self.alloc_id(bytes.len());
        self.stream.memcpy_htod(bytes, &mut d).expect("htod");
        self.blocks.insert(id, d);
        Bytes { id, len: 0 }
    }

    fn slice_of(&self, id: u64) -> &CudaSlice<u8> {
        self.blocks.get(&id).expect("块在册")
    }

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
    fn resolve(&self, b: &Bytes) -> Result<u64, owl_kernels::contract::OpError> {
        self.blocks.get(&b.id).map(|s| s.device_ptr(&self.stream).0).ok_or_else(|| {
            let mut keys: Vec<u64> = self.blocks.keys().copied().collect();
            keys.sort();
            eprintln!("[resolve miss] id={} 在册 keys={keys:?} len={}", b.id, b.len);
            owl_kernels::contract::OpError::Contract {
                op: "test".into(),
                field: "resolve",
                expect: "块在册".into(),
                got: b.id.to_string(),
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
            let mut view = self.blocks.get_mut(&id).expect("在册");
            self.stream.memcpy_htod(src, view).expect("htod");
            return Ok(());
        }
        Err(owl_kernels::contract::OpError::Contract {
            op: "test".into(), field: "upload", expect: "指针在册".into(), got: format!("{dst:#x}"),
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
    let mut rt = GdnChunkedRuntime::default();

    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/gdn_fla/cases");
    // (tag, T, tight)
    let cases: &[(&str, usize, bool)] = &[
        ("c1_t64", 64, false),
        ("c2_t128", 128, false),
        ("c5_t128_deep", 128, true),
    ];

    let rcp_ln2 = 1.442_695_f32;
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
        let out_b = Bytes { id: out_id, len: t * NV * VD };

        let call = GdnChunkedCall::builder(shape)
            .q(&q_b).unwrap()
            .k(&k_b).unwrap()
            .v(&v_b).unwrap()
            .beta(&beta_b).unwrap()
            .gate_raw(&g_b).unwrap()
            .state(&state_b, 0).unwrap()
            .build().unwrap();
        let msg: LaunchMsg = call.to_launch(out_b.clone());

        // ── 新链执行(init + run;Result 全链)──
        {
            let mut env = RunEnv::new(&mut res, &mut exec);
            rt.init(&mut env).expect("runtime init");
            rt.run(&msg, &mut env).expect("runtime run");
        }
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

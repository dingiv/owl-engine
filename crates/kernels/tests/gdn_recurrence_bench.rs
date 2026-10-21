//! **GDN 递推核台**(`owl_gdn_recurrence_varlen_gqa_f16`)—— P1.4 的判据台。
//!
//! # 为什么要有它
//!
//! P0.1 逐核账(W4A8,TP2,8k prompt)钉出 prefill 的**第三大项**:
//! `owl_gdn_recurrence_varlen_gqa_f16` = **1.33 s / 19%**(768 发 × 1730 µs)。
//! 账面上它只有 ~0.5 TFLOP 的算术量 ⇒ **1.7 µs/token 全是延迟暴露**:
//! 核内 `for t in 0..seq_len` 是**串行链**,每 token 只有 ~40 个 FMA + 两次 warp 归约,
//! 而 q/k 预取(→ shared)与 `v` 标量装载都**跨不过 `__syncthreads()`** ⇒
//! 每 token 至少吃掉 1–2 次 DRAM 往返(~600–900 cycle)。
//!
//! 判据(改之前立、改之后复):
//! 1. **数值逐位**:本台把 `out` + `state` 落盘(`OWL_GDN_REF_DUMP=1`),改后同形再跑
//!    **逐字节比对** —— 纯调度/预取改动**不得**改变算术序(否则是另一件事);
//! 2. **时延**:同形状 K 次发射的 µs/发(单卡,`T` 扫 128/512/980 = 生产 chunk);
//! 3. **规模律**:µs/token 应随 T 近似**常数**(串行链的签名);若改后显著下降 ⇒
//!    预取真的把延迟藏住了。
//!
//! 跑法(单卡;`OWL_TEST_DEVICE` = CUDA ordinal):
//!
//! ```bash
//! OWL_TEST_DEVICE=2 cargo test -p owl-kernels --features device \
//!   --test gdn_recurrence_bench -- --nocapture --test-threads=1
//! # 参考落盘(改核之前跑一次)
//! OWL_GDN_REF_DUMP=1 OWL_TEST_DEVICE=2 cargo test -p owl-kernels --features device \
//!   --test gdn_recurrence_bench -- --nocapture --test-threads=1
//! ```
#![cfg(feature = "device")]

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr};
use half::f16;
use owl_kernels::contract::{Bytes, Dtype, LaunchVal, OpError, OpId};
use owl_kernels::device::{DevPtr, DeviceRes, Exec, LaunchNote, ScratchBuf};

// ============================================================================
// §1 测试资源面(最小实现;与 ln_elems_bench.rs 同款 —— 勿改语义)
// ============================================================================


struct TestRes {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    blocks: HashMap<u64, CudaSlice<u8>>,
    next: u64,
}

impl TestRes {
    fn new() -> Option<Self> {
        let ordinal = owl_shared::env_reader::parse::<usize>("OWL_TEST_DEVICE")?;
        let ctx = CudaContext::new(ordinal).expect("ctx");
        ctx.bind_to_thread().expect("bind");
        let stream = ctx.default_stream();
        Some(Self { ctx, stream, blocks: HashMap::new(), next: 1 })
    }
    fn blk(&mut self, nbytes: usize, tag: &str) -> u64 {
        match self.alloc_block(nbytes, tag) {
            Ok(b) => b.id,
            Err(e) => panic!("alloc_block({tag}): {e:?}"),
        }
    }
    fn up(&mut self, id: u64, src: &[u8]) {
        let b = Bytes::new(id, src.len(), Dtype::U8, 0);
        match self.upload_block(&b, src) {
            Ok(()) => {}
            Err(e) => panic!("upload_block(id={id}): {e:?}"),
        }
    }
    /// 块内**指定字节偏移**写入(表类块只初始化要读的行)
    #[allow(dead_code)] // bench 复用样板:部分 runner 不走上传口
    fn up_at(&mut self, id: u64, off: usize, src: &[u8]) {
        let stream = self.stream.clone();
        let mut tmp = unsafe { stream.alloc::<u8>(src.len()) }.expect("tmp alloc");
        stream.memcpy_htod(src, &mut tmp).expect("htod tmp");
        let big = self.blocks.get_mut(&id).expect("块在册");
        let mut sub = big.slice_mut(off..off + src.len());
        stream.memcpy_dtod(&tmp, &mut sub).expect("d2d");
    }
    fn down(&mut self, id: u64, dst: &mut [u8]) {
        let p = match self.resolve(id) {
            Ok(p) => p,
            Err(e) => panic!("resolve(id={id}): {e:?}"),
        };
        match self.download(p, dst) {
            Ok(()) => {}
            Err(e) => panic!("download(id={id}): {e:?}"),
        }
    }
    fn ptr(&self, id: u64) -> u64 {
        self.resolve(id).unwrap_or_else(|e| panic!("resolve(id={id}): {e:?}"))
    }
    fn sync(&self) {
        self.stream.synchronize().expect("sync");
    }
}

impl DeviceRes for TestRes {
    fn context(&self) -> Result<Arc<CudaContext>, OpError> {
        Ok(self.ctx.clone())
    }
    fn stream(&self) -> Result<Arc<CudaStream>, OpError> {
        Ok(self.stream.clone())
    }
    fn resolve(&self, id: u64) -> Result<DevPtr, OpError> {
        self.blocks.get(&id).map(|s| s.device_ptr(&self.stream).0).ok_or_else(|| OpError::Contract {
            op: "bench".into(),
            field: "resolve",
            expect: "块在册".into(),
            got: format!("id={id}"),
        })
    }
    fn resolve_len(&self, id: u64) -> Result<usize, OpError> {
        self.blocks.get(&id).map(|s| s.len()).ok_or_else(|| OpError::Contract {
            op: "bench".into(),
            field: "resolve_len",
            expect: "块在册".into(),
            got: format!("id={id}"),
        })
    }
    fn alloc(&mut self, bytes: usize, _tag: &str) -> Result<ScratchBuf, OpError> {
        let s = unsafe { self.stream.alloc::<u8>(bytes.max(1)) }.map_err(|e| OpError::Contract {
            op: "bench".into(),
            field: "alloc",
            expect: "设备分配".into(),
            got: format!("{e:?}"),
        })?;
        let ptr = s.device_ptr(&self.stream).0;
        Ok(ScratchBuf { ptr, bytes })
    }
    fn capturing(&self) -> Result<bool, OpError> {
        Ok(false)
    }
    fn record_capture(&mut self, _n: LaunchNote) -> Result<(), OpError> {
        Ok(())
    }
    fn upload(&mut self, _dst: DevPtr, _src: &[u8]) -> Result<(), OpError> {
        Err(OpError::Contract {
            op: "bench".into(),
            field: "upload",
            expect: "走 upload_block".into(),
            got: "裸指针上传未实现".into(),
        })
    }
    fn device_ordinal(&self) -> Result<i32, OpError> {
        Ok(self.ctx.ordinal() as i32)
    }
    fn alloc_block(&mut self, bytes: usize, _tag: &str) -> Result<Bytes, OpError> {
        let s = unsafe { self.stream.alloc::<u8>(bytes.max(1)) }.map_err(|e| OpError::Contract {
            op: "bench".into(),
            field: "alloc_block",
            expect: "设备分配".into(),
            got: format!("{e:?}"),
        })?;
        let id = self.next;
        self.next += 1;
        self.blocks.insert(id, s);
        Ok(Bytes::new(id, bytes, Dtype::U8, 0))
    }
    fn upload_block(&mut self, b: &Bytes, src: &[u8]) -> Result<(), OpError> {
        let s = self.blocks.get_mut(&b.id).ok_or_else(|| OpError::Contract {
            op: "bench".into(),
            field: "upload_block",
            expect: "块在册".into(),
            got: format!("id={}", b.id),
        })?;
        let n = src.len().min(s.len());
        self.stream.memcpy_htod(&src[..n], &mut *s).map_err(|e| OpError::Contract {
            op: "bench".into(),
            field: "upload_block",
            expect: "htod".into(),
            got: format!("{e:?}"),
        })
    }
    fn download(&mut self, src: DevPtr, dst: &mut [u8]) -> Result<(), OpError> {
        let id = self
            .blocks
            .iter()
            .find(|(_, s)| s.device_ptr(&self.stream).0 == src)
            .map(|(id, _)| *id)
            .ok_or_else(|| OpError::Contract {
                op: "bench".into(),
                field: "download",
                expect: "ptr 在册".into(),
                got: format!("{src:#x}"),
            })?;
        let s = self.blocks.get(&id).expect("刚找到");
        let mut tmp = vec![0u8; s.len()];
        self.stream.memcpy_dtoh(s, &mut tmp).map_err(|e| OpError::Contract {
            op: "bench".into(),
            field: "download",
            expect: "dtoh".into(),
            got: format!("{e:?}"),
        })?;
        dst.copy_from_slice(&tmp[..dst.len()]);
        Ok(())
    }
}

// ============================================================================
// §1 输入(27B 生产形状:nv=48 / nk=16 / kd=vd=128)
// ============================================================================

const NK: usize = 16;
#[allow(dead_code)]
const NV: usize = 48;
const KD: usize = 128;
const VD: usize = 128;

/// **生产形状 = TP 每 rank 的 GDN 头数**(2026-10-19 现场钉:生产 nsys 里
/// `gdn_recurrence` 的 grid = `(16, 24, 1)` ⇒ `nv = 24`、`nk = 8`;
/// 全量档(nv=48/nk=16)只在 TP1 出现)。台子默认跟生产走,可用 env 覆盖。
fn shape_env() -> (usize, usize) {
    let e = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    (e("OWL_GDN_NV", 24), e("OWL_GDN_NK", 8))
}

fn as_bytes_f16(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn as_bytes_f32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// 确定性 [0,1)
fn rnd(seed: u32, i: usize) -> f32 {
    let x = seed
        .wrapping_mul(0x9E37_79B9)
        .wrapping_add((i as u32).wrapping_mul(0x85EB_CA6B));
    ((x >> 8) & 0x00FF_FFFF) as f32 / 16777216.0
}

fn h(x: f32) -> u16 {
    f16::from_f32(x).to_bits()
}

struct In {
    q: Vec<u16>,
    k: Vec<u16>,
    v: Vec<u16>,
    g: Vec<u16>,
    beta: Vec<u16>,
    state: Vec<f32>,
}

/// 生产量级:q/k/v = 投影输出小量级;beta ∈ (0,1);g = **raw log 门** ∈ [-0.5,-0.01]
/// (核内 `expf(g)`;深谷档见 `gdn_big_t_arms_parity`)。
fn gen(t: usize, nv: usize, nk: usize) -> In {
    let n_qk = t * nk * KD;
    let n_v = t * nv * VD;
    In {
        q: (0..n_qk).map(|i| h((rnd(0x51, i) - 0.5) * 0.1)).collect(),
        k: (0..n_qk).map(|i| h((rnd(0x52, i) - 0.5) * 0.1)).collect(),
        v: (0..n_v).map(|i| h((rnd(0x53, i) - 0.5) * 0.1)).collect(),
        g: (0..t * nv).map(|i| h(-0.01 - rnd(0x54, i) * 0.49)).collect(),
        beta: (0..t * nv).map(|i| h(0.1 + rnd(0x55, i) * 0.8)).collect(),
        state: (0..nv * KD * VD).map(|i| (rnd(0x56, i) - 0.5) * 0.02).collect(),
    }
}

fn fire(
    res: &mut TestRes,
    exec: &mut Exec,
    f: &CudaFunction,
    entry: &'static str,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    smem: u32,
    vals: &[LaunchVal],
) {
    if let Err(e) = exec.launch(res, OpId(owl_kernels::spec::NATIVE), f, entry, grid, block, smem, vals)
    {
        panic!("launch({entry}) grid={grid:?} block={block:?}: {e:?}");
    }
}

/// SM 当前时钟(判读用 —— **空载 210 MHz / 满载 2.1 GHz,差 10×**,
/// 不预热就量的读数会系统性偏低 2× 以上;2026-10-19 首轮踩过)。
fn sm_clock_mhz(ordinal: usize) -> String {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=clocks.sm","--format=csv,noheader,nounits","-i", &ordinal.to_string()])
        .output();
    match out {
        Ok(o) => String::from_utf8_lossy(&o.stdout).trim().to_string(),
        Err(_) => "?".into(),
    }
}

/// 预热:持续发射到 `ms` 毫秒,让 boost 时钟落定(否则量的是 ramp 段)。
#[allow(clippy::too_many_arguments)]
fn spin_up(
    res: &mut TestRes,
    exec: &mut Exec,
    f: &CudaFunction,
    entry: &'static str,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    smem: u32,
    vals: &[LaunchVal],
    ms: u64,
) {
    let t0 = Instant::now();
    while t0.elapsed().as_millis() < ms as u128 {
        fire(res, exec, f, entry, grid, block, smem, vals);
    }
    res.sync();
}

/// K × M 发射的 µs/发(M=1 = host 提交地板;M=8 = 设备读数)
fn time_us(
    res: &mut TestRes,
    exec: &mut Exec,
    f: &CudaFunction,
    entry: &'static str,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    smem: u32,
    vals: &[LaunchVal],
    iters: usize,
    m: usize,
) -> f64 {
    for _ in 0..2 {
        fire(res, exec, f, entry, grid, block, smem, vals);
    }
    res.sync();
    let t0 = Instant::now();
    for _ in 0..iters {
        for _ in 0..m {
            fire(res, exec, f, entry, grid, block, smem, vals);
        }
    }
    res.sync();
    t0.elapsed().as_secs_f64() * 1e6 / (iters * m) as f64
}

// ============================================================================
// §2 台子
// ============================================================================

fn bench_one(res: &mut TestRes, entry: &'static str, src: &'static str, t: usize, nv: usize, nk: usize) {
    let mut exec = Exec::new();
    let f = exec.load_aot(res, src, entry, OpId(owl_kernels::spec::NATIVE)).expect("资产");

    let inp = gen(t, nv, nk);
    let bq = res.blk(t * nk * KD * 2, "q");
    let bk = res.blk(t * nk * KD * 2, "k");
    let bv = res.blk(t * nv * VD * 2, "v");
    let bg = res.blk(t * nv * 2, "g");
    let bb = res.blk(t * nv * 2, "beta");
    let bs = res.blk(nv * KD * VD * 4, "state");
    let bslot = res.blk(4, "slots");
    let bcu = res.blk(8, "cu_seqlens");
    let bo = res.blk(t * nv * VD * 2, "out");
    res.up(bq, &as_bytes_f16(&inp.q));
    res.up(bk, &as_bytes_f16(&inp.k));
    res.up(bv, &as_bytes_f16(&inp.v));
    res.up(bg, &as_bytes_f16(&inp.g));
    res.up(bb, &as_bytes_f16(&inp.beta));
    res.up(bs, &as_bytes_f32(&inp.state));
    res.up(bslot, &as_bytes_f32(&[0.0f32]));
    res.up(bcu, &as_bytes_f32(&[0.0f32, t as f32]));
    res.sync();

    let (pq, pk, pv, pg, pb, ps, psl, pcu, po) = (
        res.ptr(bq),
        res.ptr(bk),
        res.ptr(bv),
        res.ptr(bg),
        res.ptr(bb),
        res.ptr(bs),
        res.ptr(bslot),
        res.ptr(bcu),
        res.ptr(bo),
    );

    let prod = owl_kernels::driver::gdn::recurrence_varlen_gqa(
        owl_kernels::driver::DType::F16,
        VD,
        nv,
        KD,
        1,
    );
    let grid = prod.shape.grid;
    let block = prod.shape.block;
    let smem = prod.shape.smem;

    let vals = [
        LaunchVal::Ptr(pq),
        LaunchVal::Ptr(pk),
        LaunchVal::Ptr(pv),
        LaunchVal::Ptr(pg),
        LaunchVal::Ptr(pb),
        LaunchVal::Ptr(ps),
        LaunchVal::Ptr(psl),
        LaunchVal::Ptr(pcu),
        LaunchVal::U64(1),   // batch
        LaunchVal::U64(nv as u64),
        LaunchVal::U64(nk as u64),
        LaunchVal::U64(KD as u64),
        LaunchVal::U64(VD as u64),
        LaunchVal::F32(1.0 / (KD as f32).sqrt()),
        LaunchVal::Ptr(po),
    ];

    // ── 落盘 / 比对(逐位)──
    let reff = format!("/tmp/gdn-rec-ref-nv{nv}-T{t}.bin");
    let mut out = vec![0u8; t * nv * VD * 2];
    fire(res, &mut exec, &f, entry, grid, block, smem, &vals);
    res.sync();
    res.down(bo, &mut out);
    let mut st = vec![0u8; nv * KD * VD * 4];
    res.down(bs, &mut st);
    let mut blob = out.clone();
    blob.extend_from_slice(&st);
    if std::env::var("OWL_GDN_REF_DUMP").is_ok() {
        std::fs::write(&reff, &blob).expect("落盘参考");
        println!("  [ref] 已落盘 {reff}({} B)", blob.len());
    } else if let Ok(want) = std::fs::read(&reff) {
        let same = want.len() == blob.len() && want == blob;
        let diff = if same {
            "逐字节相同".to_string()
        } else {
            let n = want.len().min(blob.len());
            let at = (0..n).find(|i| want[*i] != blob[*i]).unwrap_or(n);
            format!("✗ 首个不同字节 @{at}(want {} got {})", want[at], blob.get(at).copied().unwrap_or(0))
        };
        println!("  [ref] vs {reff}: {diff}");
    } else {
        println!("  [ref] 无参考文件({reff});数值判据跳过");
    }

    // 复位 state(核是 in/out)
    res.up(bs, &as_bytes_f32(&inp.state));
    res.sync();

    let ord: usize = std::env::var("OWL_TEST_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let warm_ms: u64 = std::env::var("OWL_GDN_WARM_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(2500);
    spin_up(res, &mut exec, &f, entry, grid, block, smem, &vals, warm_ms);
    let c0 = sm_clock_mhz(ord);
    let us1 = time_us(res, &mut exec, &f, entry, grid, block, smem, &vals, 20, 1);
    let us8 = time_us(res, &mut exec, &f, entry, grid, block, smem, &vals, 4, 8);
    let c1 = sm_clock_mhz(ord);
    println!(
        "  nv={nv:<3} nk={nk:<3} T={t:<5} grid={grid:?} block={block:?}  M=1 {us1:>9.1} µs · M=8 {us8:>9.1} µs \
         ⇒ {:>6.2} µs/token  (sm {c0}→{c1} MHz)",
        us8 / t as f64
    );
}

#[test]
fn gdn_recurrence_bench() {
    let Some(mut res) = TestRes::new() else {
        eprintln!("未设 OWL_TEST_DEVICE ⇒ skip");
        return;
    };
    let entry = "owl_gdn_recurrence_varlen_gqa_f16";
    let src = <owl_kernels::ops::gdn::RecurrenceVarlenGqaF16 as owl_kernels::spec::KernelSpec>::SRC;
    let (nv, nk) = shape_env();
    println!("\n═══ {entry}  nv={nv} nk={nk} kd={KD} vd={VD}(生产 = TP 每 rank nv=24/nk=8)");
    for t in [128usize, 512, 1024] {
        bench_one(&mut res, entry, src, t, nv, nk);
    }
}

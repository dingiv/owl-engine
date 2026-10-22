//! **int8 权重 GEMV 台**(`owl_gemv_i8w_f16`)—— P1.5-a 的判据台。
//!
//! # 判据(两条,缺一不算完)
//!
//! 1. **带宽**:`rows=124160`(词表切后每 rank 的行数)· `cols=5120` ⇒ 权重 **0.635 GB**;
//!    目标 **≥ 700 GB/s**(= 0.9 ms;现役 bf16 gemv 是 724 GB/s / 1.76 ms);
//! 2. **数值**:与 **host f32 参考**逐元素比(不是与另一个核比)——
//!    相对误差以 `max|ref|` 为尺度,容差 = int8 量化自身的下界(`~2/127·√d` 量级,
//!    本核只做"同一份量化权重的点积",误差不应超过量化误差本身)。
//!
//! 跑法:
//! ```bash
//! OWL_TEST_DEVICE=01:00.0 cargo test -p owl-kernels --features device \
//!   --test gemv_i8w_bench -- --nocapture --test-threads=1
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
        // 测试域唯一门控 = OWL_TEST_DEVICE(BDF);经 owl-cuda 统一解析,
        // 不再直吃 CUDA ordinal(FASTEST_FIRST 序会漂)。
        let ordinal = owl_cuda::test_device_ordinal_opt()?;
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
// §2 台
// ============================================================================

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
    for _ in 0..3 {
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

/// 确定性 [0,1)
fn rnd(seed: u32, i: usize) -> f32 {
    let x = seed
        .wrapping_mul(0x9E37_79B9)
        .wrapping_add((i as u32).wrapping_mul(0x85EB_CA6B));
    ((x >> 8) & 0x00FF_FFFF) as f32 / 16777216.0
}

/// 一档(rows × cols)跑完即报:µs/发、GB/s、与 host f32 参考的相对误差。
fn bench_one(res: &mut TestRes, entry: &'static str, src: &'static str, rows: usize, cols: usize) {
    let mut exec = Exec::new();
    let f = exec.load_aot(res, src, entry, OpId(owl_kernels::spec::NATIVE)).expect("资产");

    // 权重:~N(0, 0.02) 截断后 per-row 对称量化(与装载器同一式)
    let mut wq = vec![0i8; rows * cols];
    let mut sc = vec![0f32; rows];
    let mut wf = vec![0f32; rows * cols];   // host 参考用(反量化后的权重)
    for r in 0..rows {
        let mut mx = 0.0f32;
        let mut row = vec![0f32; cols];
        for k in 0..cols {
            // 轻尾分布:多数小值 + 少量大值(贴近真实词表行)
            let u = rnd(0x71u32.wrapping_add(r as u32), r * cols + k);
            let v = (u - 0.5) * 0.08 * if u > 0.99 { 4.0 } else { 1.0 };
            row[k] = v;
            mx = mx.max(v.abs());
        }
        let s = if mx > 0.0 { mx / 127.0 } else { 1.0 };
        sc[r] = s;
        for k in 0..cols {
            let q = (row[k] / s).round().clamp(-127.0, 127.0);
            wq[r * cols + k] = q as i8;
            wf[r * cols + k] = q * s;
        }
    }
    let x: Vec<u16> = (0..cols).map(|k| f16::from_f32((rnd(0x72, k) - 0.5) * 0.1).to_bits()).collect();

    // host f32 参考(逐行,顺序累加 —— 与核的 fp32 累加同域,容差只看量化误差)
    let xf: Vec<f32> = x.iter().map(|b| f16::from_bits(*b).to_f32()).collect();
    let mut want = vec![0f32; rows];
    for r in 0..rows {
        let z: f64 = (0..cols).map(|k| wf[r * cols + k] as f64 * xf[k] as f64).sum();
        want[r] = z as f32;
    }

    let bw = res.blk(rows * cols, "w");
    let bs = res.blk(rows * 4, "scale");
    let bx = res.blk(cols * 2, "x");
    let by = res.blk(rows * 2, "y");
    res.up(bw, &wq.iter().map(|v| *v as u8).collect::<Vec<u8>>());
    res.up(bs, &sc.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
    res.up(bx, &x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
    let (pw, ps, px, py) = (res.ptr(bw), res.ptr(bs), res.ptr(bx), res.ptr(by));

    let prod = owl_kernels::driver::elems::gemv_i8w(rows, cols, 1);
    let (grid, block, smem) = (prod.shape.grid, prod.shape.block, prod.shape.smem);
    let vals = [
        LaunchVal::Ptr(pw),
        LaunchVal::Ptr(ps),
        LaunchVal::Ptr(px),
        LaunchVal::I32(rows as i32),
        LaunchVal::I32(cols as i32),
        LaunchVal::Ptr(py),
    ];

    fire(res, &mut exec, &f, entry, grid, block, smem, &vals);
    res.sync();
    let mut got = vec![0u8; rows * 2];
    res.down(by, &mut got);
    let scale = want.iter().fold(0.0f32, |a, b| a.max(b.abs())).max(1e-6);
    let mut worst = 0.0f32;
    let mut at = 0usize;
    for (i, b) in got.chunks_exact(2).enumerate() {
        let g = f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32();
        let e = (g - want[i]).abs() / scale;
        if e > worst {
            worst = e;
            at = i;
        }
    }

    let us1 = time_us(res, &mut exec, &f, entry, grid, block, smem, &vals, 20, 1);
    let us8 = time_us(res, &mut exec, &f, entry, grid, block, smem, &vals, 4, 8);
    let bytes = (rows * cols + rows * 4 + cols * 2 + rows * 2) as f64;
    let gbs8 = bytes / (us8 * 1e-6) / 1e9;
    let gbs1 = bytes / (us1 * 1e-6) / 1e9;
    println!(
        "  rows={rows:<7} cols={cols:<5} grid={grid:?}  M=1 {us1:>8.1} µs({gbs1:>6.1} GB/s) · \
         8发/同grid {us8:>8.1} µs({gbs8:>6.1} GB/s)  数值 max|Δ|/max = {worst:.2e} @{at}"
    );
}

/// **多 token 快路 A/B**:同一份权重/输入,旧核(grid.y=tokens,每 token 重读权重)
/// vs 新核(权重一次,TOK_TILE=8)对比;输出逐位一致 + 吞吐。
fn bench_mt(res: &mut TestRes, rows: usize, cols: usize, tokens: usize) {
    let mut exec = Exec::new();
    let src = <owl_kernels::ops::elems::GemvI8wF16 as owl_kernels::spec::KernelSpec>::SRC;
    let f_old = exec
        .load_aot(res, src, "owl_gemv_i8w_f16", OpId(owl_kernels::spec::NATIVE))
        .expect("old 资产");
    let prod = owl_kernels::driver::elems::gemv_i8w(rows, cols, tokens);
    let f_new = exec
        .load_aot(res, src, prod.name, OpId(owl_kernels::spec::NATIVE))
        .expect("mt 资产");

    // 同一份 per-row 对称 int8 权重
    let mut wq = vec![0i8; rows * cols];
    let mut sc = vec![0f32; rows];
    for r in 0..rows {
        let mut mx = 0.0f32;
        let mut row = vec![0f32; cols];
        for k in 0..cols {
            let u = rnd(0x71u32.wrapping_add(r as u32), r * cols + k);
            let v = (u - 0.5) * 0.08 * if u > 0.99 { 4.0 } else { 1.0 };
            row[k] = v;
            mx = mx.max(v.abs());
        }
        let s = if mx > 0.0 { mx / 127.0 } else { 1.0 };
        sc[r] = s;
        for k in 0..cols {
            wq[r * cols + k] = (row[k] / s).round().clamp(-127.0, 127.0) as i8;
        }
    }
    let x: Vec<u16> = (0..tokens * cols)
        .map(|i| f16::from_f32((rnd(0x72u32.wrapping_add(i as u32), i) - 0.5) * 0.1).to_bits())
        .collect();

    let bw = res.blk(rows * cols, "w");
    let bs = res.blk(rows * 4, "scale");
    let bx = res.blk(tokens * cols * 2, "x");
    let byo = res.blk(tokens * rows * 2, "y_old");
    let byn = res.blk(tokens * rows * 2, "y_new");
    res.up(bw, &wq.iter().map(|v| *v as u8).collect::<Vec<u8>>());
    res.up(bs, &sc.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
    res.up(bx, &x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
    let (pw, ps, px, pyo, pyn) = (res.ptr(bw), res.ptr(bs), res.ptr(bx), res.ptr(byo), res.ptr(byn));

    // 旧核:grid.y = tokens,每个 token 各读一遍权重
    let old_grid = (rows.div_ceil(8) as u32, tokens as u32, 1);
    let old_block = (256, 1, 1);
    let old_smem = (cols * 2) as u32;
    let old_vals = [
        LaunchVal::Ptr(pw),
        LaunchVal::Ptr(ps),
        LaunchVal::Ptr(px),
        LaunchVal::I32(rows as i32),
        LaunchVal::I32(cols as i32),
        LaunchVal::Ptr(pyo),
    ];
    fire(res, &mut exec, &f_old, "owl_gemv_i8w_f16", old_grid, old_block, old_smem, &old_vals);

    // 新核:driver 在 tokens=7 时选 mt7,其余 >1 选 mt8
    let (new_grid, new_block, new_smem) = (prod.shape.grid, prod.shape.block, prod.shape.smem);
    let new_vals = [
        LaunchVal::Ptr(pw),
        LaunchVal::Ptr(ps),
        LaunchVal::Ptr(px),
        LaunchVal::I32(rows as i32),
        LaunchVal::I32(cols as i32),
        LaunchVal::I32(tokens as i32),
        LaunchVal::Ptr(pyn),
    ];
    fire(res, &mut exec, &f_new, prod.name, new_grid, new_block, new_smem, &new_vals);
    res.sync();

    let mut yo = vec![0u8; tokens * rows * 2];
    let mut yn = vec![0u8; tokens * rows * 2];
    res.down(byo, &mut yo);
    res.down(byn, &mut yn);
    let mut max_abs = 0.0f32;
    let mut max_ref = 0.0f32;
    let mut diff = 0usize;
    for v in 0..tokens * rows {
        let o = f16::from_bits(u16::from_le_bytes([yo[v*2], yo[v*2+1]])).to_f32();
        let n = f16::from_bits(u16::from_le_bytes([yn[v*2], yn[v*2+1]])).to_f32();
        if yo[v*2] != yn[v*2] || yo[v*2+1] != yn[v*2+1] { diff += 1; }
        max_abs = max_abs.max((o - n).abs());
        max_ref = max_ref.max(o.abs());
    }
    assert!(
        max_abs <= (max_ref * 1e-3).max(1e-6),
        "多 token 快路与旧核偏差过大:max|Δ|={max_abs} / max|ref|={max_ref}(diff {diff})"
    );
    let us_old = time_us(res, &mut exec, &f_old, "owl_gemv_i8w_f16", old_grid, old_block, old_smem, &old_vals, 10, 1);
    let us_new = time_us(res, &mut exec, &f_new, prod.name, new_grid, new_block, new_smem, &new_vals, 30, 1);
    let wbytes = (rows * cols) as f64;
    println!(
        "  rows={rows:<7} cols={cols:<5} tokens={tokens}  old {us_old:>8.1} µs({:>6.1} GB/s) · \
         mt {us_new:>8.1} µs({:>6.1} GB/s) · speedup {:.2}×",
        wbytes / (us_old * 1e-6) / 1e9,
        wbytes / (us_new * 1e-6) / 1e9,
        us_old / us_new
    );
}

#[test]
fn gemv_i8w_bench() {
    let Some(mut res) = TestRes::new() else {
        eprintln!("未设 OWL_TEST_DEVICE ⇒ skip");
        return;
    };
    let entry = "owl_gemv_i8w_f16";
    let src = <owl_kernels::ops::elems::GemvI8wF16 as owl_kernels::spec::KernelSpec>::SRC;
    println!("\n═══ {entry}(m=1;int8 权重 + per-row f32 尺度)");
    // 生产:词表切后每 rank 124160 行 × 5120(0.635 GB);另给两档看规模律
    for (rows, cols) in [(124160usize, 5120usize), (32768, 5120), (124160, 2048)] {
        bench_one(&mut res, entry, src, rows, cols);
    }
}

#[test]
fn gemv_i8w_mt_bench() {
    let Some(mut res) = TestRes::new() else {
        eprintln!("未设 OWL_TEST_DEVICE(BDF)⇒ skip");
        return;
    };
    let rows = owl_shared::env_reader::parse_or("OWL_MT_ROWS", 124160usize);
    let tokens = owl_shared::env_reader::parse_or("OWL_MT_TOKENS", 8usize);
    println!("\n═══ 多 token 快路(tokens={tokens};旧核 vs 权重只读一遍;rows={rows})");
    bench_mt(&mut res, rows, 5120, tokens);
}

/// **W8A8 head A/B**:per-token int8 激活量化 + dp4a 多头 GEMV。
/// - 小形状(rows<=1024)与 host 整数点积对拍;
/// - 生产形状只量时间(host 全量点积 5B 次,不跑)。
fn bench_a8(res: &mut TestRes, rows: usize, cols: usize, tokens: usize) {
    let mut exec = Exec::new();
    let fscale = exec
        .load_aot(res, "cu/elems/quant_i8_act.cu", "owl_quant_i8_act_scale_f16", OpId(owl_kernels::spec::NATIVE))
        .expect("a8 scale 资产");
    let fq = exec
        .load_aot(res, "cu/elems/quant_i8_act.cu", "owl_quant_i8_act_q_f16", OpId(owl_kernels::spec::NATIVE))
        .expect("a8 q 资产");
    let prod = owl_kernels::driver::elems::gemv_i8w_a8_f16(rows, cols, tokens);
    let fg = exec
        .load_aot(res, "cu/elems/gemv_i8w_a8.cu", prod.name, OpId(owl_kernels::spec::NATIVE))
        .expect("a8 gemv 资产");

    // 权重:与 W8A16 台同分布
    let mut wq = vec![0i8; rows * cols];
    let mut sw = vec![0f32; rows];
    for r in 0..rows {
        let mut mx = 0.0f32;
        let mut row = vec![0f32; cols];
        for k in 0..cols {
            let u = rnd(0x71u32.wrapping_add(r as u32), r * cols + k);
            let v = (u - 0.5) * 0.08 * if u > 0.99 { 4.0 } else { 1.0 };
            row[k] = v;
            mx = mx.max(v.abs());
        }
        let s = if mx > 0.0 { mx / 127.0 } else { 1.0 };
        sw[r] = s;
        for k in 0..cols {
            wq[r * cols + k] = (row[k] / s).round_ties_even().clamp(-127.0, 127.0) as i8;
        }
    }
    // 激活 + host 量化参考
    let x: Vec<u16> = (0..tokens * cols)
        .map(|i| f16::from_f32((rnd(0x72u32.wrapping_add(i as u32), i) - 0.5) * 0.1).to_bits())
        .collect();
    let xf: Vec<f32> = x.iter().map(|b| f16::from_bits(*b).to_f32()).collect();
    let mut qh = vec![0i8; tokens * cols];
    let mut sx = vec![0f32; tokens];
    for t in 0..tokens {
        let mut mx = 0.0f32;
        for k in 0..cols {
            mx = mx.max(xf[t * cols + k].abs());
        }
        let s = if mx > 0.0 { mx / 127.0 } else { 1.0 };
        sx[t] = s;
        for k in 0..cols {
            qh[t * cols + k] = (xf[t * cols + k] / s).round_ties_even().clamp(-127.0, 127.0) as i8;
        }
    }

    let bw = res.blk(rows * cols, "w");
    let bsw = res.blk(rows * 4, "sw");
    let bx = res.blk(tokens * cols * 2, "x");
    let bq = res.blk(tokens * cols, "q");
    let bsx = res.blk(tokens * 4, "sx");
    let by = res.blk(tokens * rows * 2, "y");
    res.up(bw, &wq.iter().map(|v| *v as u8).collect::<Vec<u8>>());
    res.up(bsw, &sw.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
    res.up(bx, &x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
    let (pw, psw, px, pq, psx, py) = (
        res.ptr(bw), res.ptr(bsw), res.ptr(bx), res.ptr(bq), res.ptr(bsx), res.ptr(by),
    );

    let scale_vals = [
        LaunchVal::Ptr(px),
        LaunchVal::I32(tokens as i32),
        LaunchVal::I32(cols as i32),
        LaunchVal::Ptr(psx),
    ];
    let q_vals = [
        LaunchVal::Ptr(px),
        LaunchVal::Ptr(psx),
        LaunchVal::I32(tokens as i32),
        LaunchVal::I32(cols as i32),
        LaunchVal::Ptr(pq),
    ];
    let g_vals = [
        LaunchVal::Ptr(pw),
        LaunchVal::Ptr(psw),
        LaunchVal::Ptr(pq),
        LaunchVal::Ptr(psx),
        LaunchVal::I32(rows as i32),
        LaunchVal::I32(cols as i32),
        LaunchVal::I32(tokens as i32),
        LaunchVal::Ptr(py),
    ];
    let scale_grid = (tokens as u32, 1, 1);
    let q_grid = (tokens as u32, 1, 1);
    fire(res, &mut exec, &fscale, "owl_quant_i8_act_scale_f16", scale_grid, (256,1,1), 0, &scale_vals);
    fire(res, &mut exec, &fq, "owl_quant_i8_act_q_f16", q_grid, (256,1,1), 0, &q_vals);
    fire(res, &mut exec, &fg, prod.name, prod.shape.grid, prod.shape.block, prod.shape.smem, &g_vals);
    res.sync();

    // 小形状:与 host 整数点积对拍(权重/激活都是同一份 int8)
    if rows <= 1024 {
        let mut got = vec![0u8; tokens * rows * 2];
        res.down(by, &mut got);
        let mut max_abs = 0.0f32;
        let mut max_ref = 0.0f32;
        for t in 0..tokens {
            for r in 0..rows {
                let mut acc: i64 = 0;
                for k in 0..cols {
                    acc += (wq[r * cols + k] as i64) * (qh[t * cols + k] as i64);
                }
                let want = acc as f32 * (sw[r] * sx[t]);
                let got = f16::from_bits(u16::from_le_bytes([got[(t * rows + r) * 2], got[(t * rows + r) * 2 + 1]])).to_f32();
                max_abs = max_abs.max((got - want).abs());
                max_ref = max_ref.max(want.abs());
            }
        }
        assert!(max_abs <= (max_ref * 1e-3).max(1e-5), "W8A8 host 对拍:max|Δ|={max_abs}/max|ref|={max_ref}");
        eprintln!("  W8A8 host 对拍 rows={rows} cols={cols}: max|Δ|/max|ref| = {:.2e}", max_abs / max_ref.max(1e-6));
    }

    let us_scale = time_us(res, &mut exec, &fscale, "owl_quant_i8_act_scale_f16", scale_grid, (256,1,1), 0, &scale_vals, 20, 1);
    let us_q = time_us(res, &mut exec, &fq, "owl_quant_i8_act_q_f16", q_grid, (256,1,1), 0, &q_vals, 20, 1);
    let us_g = time_us(res, &mut exec, &fg, prod.name, prod.shape.grid, prod.shape.block, prod.shape.smem, &g_vals, 20, 1);
    println!(
        "  W8A8 rows={rows:<7} cols={cols:<5} tokens={tokens}  scale {us_scale:>7.1} µs · q {us_q:>7.1} µs · gemv {us_g:>7.1} µs · total {:.1} µs",
        us_scale + us_q + us_g
    );
}

#[test]
fn gemv_i8w_a8_bench() {
    let Some(mut res) = TestRes::new() else { return; };
    println!("\n═══ W8A8 head(per-token int8 激活 + dp4a)");
    bench_a8(&mut res, 64, 128, 8); // host 对拍
    let rows = owl_shared::env_reader::parse_or("OWL_A8_ROWS", 124160usize);
    let tokens = owl_shared::env_reader::parse_or("OWL_A8_TOKENS", 7usize);
    bench_a8(&mut res, rows, 5120, tokens);
}

/// **W8A8 int8 Tensor Core 台**(`mma.m16n8k32.s8s8s32`)。
///
/// 判据:
/// 1. 小形状与 host 整数点积**逐位**相同(f16 输出 bit 比,不是容差);
/// 2. 生产形状只量时间(0.635 GB 权重;目标 ≥ dp4a 的 477 GB/s)。
fn bench_tc(res: &mut TestRes, rows: usize, cols: usize, tokens: usize, parity: bool) {
    let mut exec = Exec::new();
    let prod = owl_kernels::driver::elems::gemm_i8w_a8_tc_f16(rows, cols, tokens);
    let fg = exec
        .load_aot(
            res,
            "cu/elems/gemm_i8w_a8_tc.cu",
            prod.name,
            OpId(owl_kernels::spec::NATIVE),
        )
        .expect("tc 资产");

    let mut wq = vec![0i8; rows * cols];
    let mut sw = vec![0f32; rows];
    for r in 0..rows {
        let mut mx = 0.0f32;
        let mut row = vec![0f32; cols];
        for k in 0..cols {
            let u = rnd(0x81u32.wrapping_add(r as u32), r * cols + k);
            let v = (u - 0.5) * 0.08 * if u > 0.99 { 4.0 } else { 1.0 };
            row[k] = v;
            mx = mx.max(v.abs());
        }
        let s = if mx > 0.0 { mx / 127.0 } else { 1.0 };
        sw[r] = s;
        for k in 0..cols {
            wq[r * cols + k] = (row[k] / s).round_ties_even().clamp(-127.0, 127.0) as i8;
        }
    }
    let x: Vec<u16> = (0..tokens * cols)
        .map(|i| f16::from_f32((rnd(0x82u32.wrapping_add(i as u32), i) - 0.5) * 0.1).to_bits())
        .collect();
    let xf: Vec<f32> = x.iter().map(|b| f16::from_bits(*b).to_f32()).collect();
    let mut qh = vec![0i8; tokens * cols];
    let mut sx = vec![0f32; tokens];
    for t in 0..tokens {
        let mut mx = 0.0f32;
        for k in 0..cols {
            mx = mx.max(xf[t * cols + k].abs());
        }
        let s = if mx > 0.0 { mx / 127.0 } else { 1.0 };
        sx[t] = s;
        for k in 0..cols {
            qh[t * cols + k] = (xf[t * cols + k] / s).round_ties_even().clamp(-127.0, 127.0) as i8;
        }
    }

    let bw = res.blk(rows * cols, "tc.w");
    let bsw = res.blk(rows * 4, "tc.sw");
    let bq = res.blk(tokens * cols, "tc.q");
    let bsx = res.blk(tokens * 4, "tc.sx");
    let by = res.blk(tokens * rows * 2, "tc.y");
    res.up(bw, &wq.iter().map(|v| *v as u8).collect::<Vec<u8>>());
    res.up(bsw, &sw.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
    res.up(bq, &qh.iter().map(|v| *v as u8).collect::<Vec<u8>>());
    res.up(bsx, &sx.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
    let (pw, psw, pq, psx, py) = (
        res.ptr(bw), res.ptr(bsw), res.ptr(bq), res.ptr(bsx), res.ptr(by),
    );
    let vals = [
        LaunchVal::Ptr(pw),
        LaunchVal::Ptr(psw),
        LaunchVal::Ptr(pq),
        LaunchVal::Ptr(psx),
        LaunchVal::I32(rows as i32),
        LaunchVal::I32(cols as i32),
        LaunchVal::I32(tokens as i32),
        LaunchVal::Ptr(py),
    ];
    fire(
        res,
        &mut exec,
        &fg,
        prod.name,
        prod.shape.grid,
        prod.shape.block,
        prod.shape.smem,
        &vals,
    );
    res.sync();

    if parity {
        let mut got = vec![0u8; tokens * rows * 2];
        res.down(by, &mut got);
        let mut bad = 0usize;
        for t in 0..tokens {
            for r in 0..rows {
                let mut acc: i64 = 0;
                for k in 0..cols {
                    acc += (wq[r * cols + k] as i64) * (qh[t * cols + k] as i64);
                }
                let want = f16::from_f32(acc as f32 * (sw[r] * sx[t])).to_bits();
                let hit = u16::from_le_bytes([got[(t * rows + r) * 2], got[(t * rows + r) * 2 + 1]]);
                if want != hit {
                    if bad < 5 {
                        eprintln!(
                            "  TC bad t{t} r{r}: want {want:04x} got {hit:04x}"
                        );
                    }
                    bad += 1;
                }
            }
        }
        assert_eq!(bad, 0, "TC 与 host 整数点积不逐位一致({bad} 项)");
        eprintln!("  TC host 对拍 rows={rows} cols={cols} tokens={tokens}: 逐位一致 ✓");
    }

    let us = time_us(
        res,
        &mut exec,
        &fg,
        prod.name,
        prod.shape.grid,
        prod.shape.block,
        prod.shape.smem,
        &vals,
        30,
        1,
    );
    let wbytes = (rows * cols) as f64;
    println!(
        "  TC rows={rows:<7} cols={cols:<5} tokens={tokens}  {us:>7.1} µs({:>6.1} GB/s)",
        wbytes / (us * 1e-6) / 1e9
    );
}

#[test]
fn gemm_i8w_a8_tc_bench() {
    let Some(mut res) = TestRes::new() else {
        eprintln!("未设 OWL_TEST_DEVICE(BDF)⇒ skip");
        return;
    };
    println!("\n═══ W8A8 int8 Tensor Core(mma.m16n8k32.s8s8s32)");
    bench_tc(&mut res, 64, 512, 8, true);
    let rows = owl_shared::env_reader::parse_or("OWL_TC_ROWS", 124160usize);
    let tokens = owl_shared::env_reader::parse_or("OWL_TC_TOKENS", 8usize);
    bench_tc(&mut res, rows, 5120, tokens, false);
}

/// **W8A16 f16 Tensor Core 台**：int8 权重 HBM 读 + 寄存器反量化 + f16 MMA。
///
/// 判据：小形状与 host 参考一致；生产形状量时间（权重 0.635 GB）。
/// 这是“学 Marlin W4A16”的同款流水，只是权重 8-bit、激活仍 f16。
fn bench_a16_tc(res: &mut TestRes, rows: usize, cols: usize, tokens: usize, parity: bool) {
    let mut exec = Exec::new();
    let prod = owl_kernels::driver::elems::gemm_i8w_a16_tc_f16(rows, cols, tokens);
    let fg = exec
        .load_aot(
            res,
            "cu/elems/gemm_i8w_a16_tc.cu",
            prod.name,
            OpId(owl_kernels::spec::NATIVE),
        )
        .expect("a16 tc 资产");

    let mut wq = vec![0i8; rows * cols];
    let mut sw = vec![0f32; rows];
    for r in 0..rows {
        let mut mx = 0.0f32;
        let mut row = vec![0f32; cols];
        for k in 0..cols {
            let u = rnd(0x91u32.wrapping_add(r as u32), r * cols + k);
            let v = (u - 0.5) * 0.08 * if u > 0.99 { 4.0 } else { 1.0 };
            row[k] = v;
            mx = mx.max(v.abs());
        }
        let s = if mx > 0.0 { mx / 127.0 } else { 1.0 };
        sw[r] = s;
        for k in 0..cols {
            wq[r * cols + k] = (row[k] / s).round_ties_even().clamp(-127.0, 127.0) as i8;
        }
    }
    let x: Vec<u16> = (0..tokens * cols)
        .map(|i| f16::from_f32((rnd(0x92u32.wrapping_add(i as u32), i) - 0.5) * 0.1).to_bits())
        .collect();

    let bw = res.blk(rows * cols, "a16.w");
    let bsw = res.blk(rows * 4, "a16.sw");
    let bx = res.blk(tokens * cols * 2, "a16.x");
    let by = res.blk(tokens * rows * 2, "a16.y");
    res.up(bw, &wq.iter().map(|v| *v as u8).collect::<Vec<u8>>());
    res.up(bsw, &sw.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
    res.up(bx, &x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
    let (pw, psw, px, py) = (res.ptr(bw), res.ptr(bsw), res.ptr(bx), res.ptr(by));
    let vals = [
        LaunchVal::Ptr(pw),
        LaunchVal::Ptr(psw),
        LaunchVal::Ptr(px),
        LaunchVal::I32(rows as i32),
        LaunchVal::I32(cols as i32),
        LaunchVal::I32(tokens as i32),
        LaunchVal::Ptr(py),
    ];
    fire(
        res,
        &mut exec,
        &fg,
        prod.name,
        prod.shape.grid,
        prod.shape.block,
        prod.shape.smem,
        &vals,
    );
    res.sync();

    if parity {
        let mut got = vec![0u8; tokens * rows * 2];
        res.down(by, &mut got);
        let mut max_abs = 0.0f32;
        let mut max_ref = 0.0f32;
        for t in 0..tokens {
            for r in 0..rows {
                let mut acc: f32 = 0.0;
                for k in 0..cols {
                    let xv = f16::from_bits(x[t * cols + k]).to_f32();
                    acc += xv * (wq[r * cols + k] as f32);
                }
                let want = acc * sw[r];
                let got = f16::from_bits(u16::from_le_bytes([got[(t * rows + r) * 2], got[(t * rows + r) * 2 + 1]])).to_f32();
                max_abs = max_abs.max((got - want).abs());
                max_ref = max_ref.max(want.abs());
            }
        }
        assert!(
            max_abs <= (max_ref * 1e-3).max(1e-5),
            "W8A16 TC host 对拍: max|Δ|={max_abs}/max|ref|={max_ref}"
        );
        eprintln!(
            "  A16TC host 对拍 rows={rows} cols={cols} tokens={tokens}: max|Δ|/max|ref| = {:.2e}",
            max_abs / max_ref.max(1e-6)
        );
    }

    let us = time_us(
        res,
        &mut exec,
        &fg,
        prod.name,
        prod.shape.grid,
        prod.shape.block,
        prod.shape.smem,
        &vals,
        30,
        1,
    );
    let wbytes = (rows * cols) as f64;
    println!(
        "  A16TC rows={rows:<7} cols={cols:<5} tokens={tokens}  {us:>7.1} µs({:>6.1} GB/s)",
        wbytes / (us * 1e-6) / 1e9
    );
}

#[test]
fn gemm_i8w_a16_tc_bench() {
    let Some(mut res) = TestRes::new() else {
        eprintln!("未设 OWL_TEST_DEVICE(BDF)⇒ skip");
        return;
    };
    println!("\n═══ W8A16 f16 Tensor Core(int8 权重 + 寄存器反量化 + f16 MMA)");
    bench_a16_tc(&mut res, 64, 512, 8, true);
    let rows = owl_shared::env_reader::parse_or("OWL_A16TC_ROWS", 124160usize);
    let tokens = owl_shared::env_reader::parse_or("OWL_A16TC_TOKENS", 8usize);
    bench_a16_tc(&mut res, rows, 5120, tokens, false);
}

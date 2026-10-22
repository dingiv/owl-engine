//! **小核延迟台**(norm / fused_add_rmsnorm / embed / argmax)—— §58 刀③④⑤ 的判据台。
//!
//! # 为什么要有它
//!
//! §58 的 nsys 逐核账钉出三笔"不随 rank 缩"的固定成本:128 个 `gridX=1` 的 norm
//! 核(~11–15µs/核)、单线程 `owl_embed_f16`(288µs)、`owl_argmax`(27µs)。
//! 它们都在 decode 关键路径上、每步都发。改它们必须**两条判据同时成立**:
//!
//! 1. **数值**(律㉗):每条臂的输出 vs **独立 host 参考**(不共用核内公式);
//! 2. **时延**:同一发射 K 次(不插同步)× 内层 M 次 —— **M=1 与 M=32 两档并列**:
//!    两档接近 ⇒ 量的是 host 提交地板(读数无效);随 M 明显下探 ⇒ 设备受限,可信。
//!
//! 发射走**生产同一条路**(`Exec::load_aot` 资产装载 + `Exec::launch`;几何默认取
//! `driver::*` 的生产函数,标 `*`)⇒ 台子 = 同款核、同款几何。
//! `TestRes` 与 `selfcheck_oracles.rs` 同款(最小资源面;唯一规范在 `device::DeviceRes`)。
//!
//! # 跑法(单卡;`OWL_TEST_DEVICE` = CUDA ordinal)
//!
//! ```bash
//! OWL_TEST_DEVICE=2 cargo test -p owl-kernels --features device \
//!   --test ln_elems_bench -- --nocapture --test-threads=1
//! ```
#![cfg(feature = "device")]

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr};
use owl_kernels::contract::{Bytes, Dtype, LaunchVal, OpError};
use owl_kernels::device::{DevPtr, DeviceRes, Exec, LaunchNote, ScratchBuf};
use owl_kernels::driver::{DType, ROW_REDUCE_BLOCK, elems as geom_e, ln as geom_ln, ops as geom};
use owl_kernels::ops::oracles::{f16_bits, f16_to_f32, lcg_f16};
use owl_kernels::contract::OpId;

// ============================================================================
// §0 host 侧小工具
// ============================================================================

/// f32 → f16 位型(**RNE**,与核内 `__float2half` 同律;`oracles::f16_bits` 是截断,
/// 不能用于"原地写 f16 再读"的参考)
fn f32_to_f16_rne(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let raw_exp = ((bits >> 23) & 0xFF) as i32;
    let mut man = (bits >> 13) & 0x3FF;
    let mut exp = raw_exp - 127 + 15;
    if raw_exp == 0xFF {
        return sign | 0x7C00;
    }
    if exp >= 31 {
        return sign | 0x7C00;
    }
    if exp <= 0 {
        return sign; // 测试输入恒在正规域(避免次正规路径引入第二套语义)
    }
    let rem = bits & 0x1FFF;
    if rem > 0x1000 || (rem == 0x1000 && (man & 1) == 1) {
        man += 1;
        if man > 0x3FF {
            man = 0;
            exp += 1;
            if exp >= 31 {
                return sign | 0x7C00;
            }
        }
    }
    sign | ((exp as u16) << 10) | man as u16
}

fn as_bytes_f16(v: &[u16]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2) }
}
fn as_bytes_f32(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
fn f16_from_bytes(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}
fn f32_row(h: &[u16]) -> Vec<f32> {
    h.iter().map(|x| f16_to_f32(*x)).collect()
}

// ============================================================================
// §1 测试资源面(最小实现;与 selfcheck_oracles.rs 同款)
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
// §2 量法
// ============================================================================

/// 发射一次(**不校验返回值 —— 发射失败必须响亮**:`expect`)
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

/// K × M 发射的 µs/发(M=1 = host 提交地板;M=32 = 设备读数)
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

/// 相对误差(以 max|ref| 为尺度)与最坏位置。
///
/// ⚠️ **位型解码器必须跟 dtype 走**(f16 / bf16 两套):拿错解码器会把"位型"
/// 当值读 ⇒ 误差 0.5~0.8 量级的假红(2026-10-19 现场:bf16 臂先红了一次,
/// 查下去是台子把 bf16 位型按 f16 解 —— 判据自己的 bug)。
fn rel_worst_with(got: &[u16], want: &[f32], dec: fn(u16) -> f32) -> (f32, usize) {
    let scale = want.iter().fold(0.0f32, |a, b| a.max(b.abs())).max(1e-6);
    let mut worst = 0.0f32;
    let mut at = 0usize;
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let e = (dec(*g) - w).abs() / scale;
        if e > worst {
            worst = e;
            at = i;
        }
    }
    (worst, at)
}

/// f16 臂的简写
fn rel_worst(got: &[u16], want: &[f32]) -> (f32, usize) {
    rel_worst_with(got, want, f16_to_f32)
}

const TOL: f32 = 1e-2; // f16 域 + 归约序差异(与 qknorm oracle 同口径)

fn verdict_ok(worst: f32, at: usize) -> String {
    if worst > TOL { format!("✗{worst:.2e}@{at}") } else { format!("✓ {worst:.1e}") }
}

// ============================================================================
// §3 rmsnorm f16(decode 形状:1 行 × hidden 5120)
// ============================================================================

fn rmsnorm_bench(res: &mut TestRes, n: usize, rows: usize) {
    let entry = "owl_rmsnorm_f16";
    let src = <owl_kernels::ops::ln::RmsnormF16 as owl_kernels::spec::KernelSpec>::SRC;
    let mut exec = Exec::new();
    let f = exec.load_aot(res, src, entry, OpId(owl_kernels::spec::NATIVE)).expect("资产");

    // 生产几何(单源:driver;改档即改这里)
    let prod = geom::rmsnorm(DType::F16, &[vec![rows, n], vec![n]]);
    println!("\n═══ {entry}  rows={rows} n={n}");
    println!("    生产几何(driver): grid={:?} block={:?} smem={}", prod.shape.grid, prod.shape.block, prod.shape.smem);

    let x_h = lcg_f16(rows * n, 0xC1);
    let w_h: Vec<u16> = lcg_f16(n, 0xC2).iter().map(|h| f16_bits(f16_to_f32(*h) * 0.25 + 1.0)).collect();
    let eps = 1e-6f32;
    let w_off = 0i32;

    let (bx, bw) = (res.blk(rows * n * 2, "x"), res.blk(n * 2, "w"));
    res.up(bx, as_bytes_f16(&x_h));
    res.up(bw, as_bytes_f16(&w_h));
    let (px, pw) = (res.ptr(bx), res.ptr(bw));

    // host 参考(f64 累加;独立实现)
    let mut want = vec![0.0f32; rows * n];
    for r in 0..rows {
        let row = f32_row(&x_h[r * n..(r + 1) * n]);
        let ss: f64 = row.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let inv = 1.0f32 / ((ss / n as f64) as f32 + eps).sqrt();
        for i in 0..n {
            let a = f16_to_f32(w_h[i]);
            want[r * n + i] = row[i] * inv * (if w_off != 0 { a + 1.0 } else { a });
        }
    }

    println!("  {:>22} {:>10} {:>10}  {}", "臂(block)", "M=1 µs", "M=32 µs", "数值");
    for blk in [256u32, 512, 1024] {
        let bo = res.blk(rows * n * 2, "out");
        let po = res.ptr(bo);
        let grid = prod.shape.grid;
        let smem = 4 * blk;
        let vals = [
            LaunchVal::Ptr(px),
            LaunchVal::Ptr(pw),
            LaunchVal::Ptr(po),
            LaunchVal::I32(n as i32),
            LaunchVal::F32(eps),
            LaunchVal::I32(w_off),
        ];
        let block = (blk, 1, 1);
        fire(res, &mut exec, &f, entry, grid, block, smem, &vals);
        res.sync();
        let mut got = vec![0u8; rows * n * 2];
        res.down(bo, &mut got);
        let (worst, at) = rel_worst(&f16_from_bytes(&got), &want);
        let us1 = time_us(res, &mut exec, &f, entry, grid, block, smem, &vals, 200, 1);
        let us32 = time_us(res, &mut exec, &f, entry, grid, block, smem, &vals, 10, 32);
        let mark = if block == prod.shape.block { "*" } else { " " };
        println!("  {mark}{:>21} {us1:>10.2} {us32:>10.2}  {}", format!("block={blk}"), verdict_ok(worst, at));
    }
}

// ============================================================================
// §4 fused_add_rmsnorm f16(同形状;residual 原地语义)
// ============================================================================

fn fused_add_rmsnorm_bench(res: &mut TestRes, n: usize, rows: usize) {
    let entry = "owl_fused_add_rmsnorm_f16";
    let src = <owl_kernels::ops::ln::FusedAddRmsnormF16 as owl_kernels::spec::KernelSpec>::SRC;
    let mut exec = Exec::new();
    let f = exec.load_aot(res, src, entry, OpId(owl_kernels::spec::NATIVE)).expect("资产");

    let prod = geom_ln::fused_add_rmsnorm(DType::F16, rows, n);
    println!("\n═══ {entry}  rows={rows} n={n}");
    println!("    生产几何(driver): grid={:?} block={:?} smem={}", prod.shape.grid, prod.shape.block, prod.shape.smem);

    let m_h = lcg_f16(rows * n, 0xD1);
    let r_h = lcg_f16(rows * n, 0xD2);
    let w_h: Vec<u16> = lcg_f16(n, 0xD3).iter().map(|h| f16_bits(f16_to_f32(*h) * 0.25 + 1.0)).collect();
    let eps = 1e-6f32;
    let w_off = 0i32;

    let (bm, br, bw) = (res.blk(rows * n * 2, "mixed"), res.blk(rows * n * 2, "res"), res.blk(n * 2, "w"));
    res.up(bm, as_bytes_f16(&m_h));
    res.up(br, as_bytes_f16(&r_h));
    res.up(bw, as_bytes_f16(&w_h));
    let (pm, pr, pw) = (res.ptr(bm), res.ptr(br), res.ptr(bw));

    // 参考:核内**未舍入**的 v = m + r 进归约(与被替核逐式一致),
    // residual **落的是舍入后的 f16**;输出用 residual 的舍入值×inv×w。
    let mut want_r = vec![0u16; rows * n];
    let mut want = vec![0.0f32; rows * n];
    for r in 0..rows {
        let mut acc = vec![0.0f32; n];
        for i in 0..n {
            let v = f16_to_f32(m_h[r * n + i]) + f16_to_f32(r_h[r * n + i]);
            let h = f32_to_f16_rne(v);
            want_r[r * n + i] = h;
            acc[i] = v;
        }
        let ss: f64 = acc.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let inv = 1.0f32 / ((ss / n as f64) as f32 + eps).sqrt();
        for i in 0..n {
            let a = f16_to_f32(w_h[i]);
            want[r * n + i] = f16_to_f32(want_r[r * n + i]) * inv * (if w_off != 0 { a + 1.0 } else { a });
        }
    }

    println!("  {:>22} {:>10} {:>10} {:>14} {:>10}", "臂(block)", "M=1 µs", "M=32 µs", "out 数值", "residual");
    for blk in [256u32, 512, 1024] {
        let bo = res.blk(rows * n * 2, "out");
        let po = res.ptr(bo);
        let grid = prod.shape.grid;
        let smem = 4 * blk;
        let vals = [
            LaunchVal::Ptr(pm),
            LaunchVal::Ptr(pr),
            LaunchVal::Ptr(pw),
            LaunchVal::F32(eps),
            LaunchVal::U64(n as u64),
            LaunchVal::I32(w_off),
            LaunchVal::Ptr(po),
        ];
        let block = (blk, 1, 1);
        res.up(br, as_bytes_f16(&r_h)); // 原地核 ⇒ 每条臂复位
        fire(res, &mut exec, &f, entry, grid, block, smem, &vals);
        res.sync();
        let mut got = vec![0u8; rows * n * 2];
        res.down(bo, &mut got);
        let (worst, at) = rel_worst(&f16_from_bytes(&got), &want);
        let mut gr = vec![0u8; rows * n * 2];
        res.down(br, &mut gr);
        let nbad = f16_from_bytes(&gr).iter().zip(want_r.iter()).filter(|(g, w)| g != w).count();
        let us1 = time_us(res, &mut exec, &f, entry, grid, block, smem, &vals, 200, 1);
        let us32 = time_us(res, &mut exec, &f, entry, grid, block, smem, &vals, 10, 32);
        let mark = if block == prod.shape.block { "*" } else { " " };
        println!(
            "  {mark}{:>21} {us1:>10.2} {us32:>10.2} {:>14} {:>10}",
            format!("block={blk}"),
            verdict_ok(worst, at),
            if nbad == 0 { "✓ 逐位".to_string() } else { format!("✗{nbad}") }
        );
    }
}

// ============================================================================
// §5 embed f16(decode:1 token;表 = 真 vocab)
// ============================================================================

fn embed_bench(res: &mut TestRes, vocab: usize, d_dim: usize, tokens: usize) {
    let entry = "owl_embed_f16";
    let src = <owl_kernels::ops::elems::EmbedF16 as owl_kernels::spec::KernelSpec>::SRC;
    let mut exec = Exec::new();
    let f = exec.load_aot(res, src, entry, OpId(owl_kernels::spec::NATIVE)).expect("资产");

    let prod = geom_e::embed(DType::F16, tokens);
    println!("\n═══ {entry}  vocab={vocab} d={d_dim} tokens={tokens}(表 {:.2} GiB)",
        (vocab * d_dim * 2) as f64 / (1u64 << 30) as f64);
    println!("    生产几何(driver): grid={:?} block={:?}", prod.shape.grid, prod.shape.block);

    let bw = res.blk(vocab * d_dim * 2, "w");
    let pw = res.ptr(bw);
    let ids_u: Vec<u32> = (0..tokens).map(|i| ((i * 7919 + 13) % vocab) as u32).collect();
    let ids_f: Vec<f32> = ids_u.iter().map(|&i| i as f32).collect();
    let mut want = vec![0u16; tokens * d_dim];
    for t in 0..tokens {
        let row = lcg_f16(d_dim, 0xE1 + t as u32);
        res.up_at(bw, ids_u[t] as usize * d_dim * 2, as_bytes_f16(&row));
        want[t * d_dim..(t + 1) * d_dim].copy_from_slice(&row);
    }
    let bi = res.blk(tokens * 4, "ids");
    res.up(bi, as_bytes_f32(&ids_f));
    let pi = res.ptr(bi);

    println!("  {:>22} {:>10} {:>10}  {}", "臂(block)", "M=1 µs", "M=32 µs", "数值");
    for blk in [1u32, 64, 128, 256, 512, 1024] {
        let bo = res.blk(tokens * d_dim * 2, "out");
        let po = res.ptr(bo);
        let grid = prod.shape.grid;
        let block = (blk, 1, 1);
        let vals = [
            LaunchVal::Ptr(pw),
            LaunchVal::Ptr(pi),
            LaunchVal::U64(d_dim as u64),
            LaunchVal::Ptr(po),
        ];
        fire(res, &mut exec, &f, entry, grid, block, 0, &vals);
        res.sync();
        let mut got = vec![0u8; tokens * d_dim * 2];
        res.down(bo, &mut got);
        let nbad = f16_from_bytes(&got).iter().zip(want.iter()).filter(|(g, w)| g != w).count();
        let us1 = time_us(res, &mut exec, &f, entry, grid, block, 0, &vals, 200, 1);
        let us32 = time_us(res, &mut exec, &f, entry, grid, block, 0, &vals, 10, 32);
        let mark = if block == prod.shape.block { "*" } else { " " };
        println!(
            "  {mark}{:>21} {us1:>10.2} {us32:>10.2}  {}",
            format!("block={blk}"),
            if nbad == 0 { "✓ 逐位".to_string() } else { format!("✗{nbad}") }
        );
    }
}

// ============================================================================
// §6 argmax f16(n = 词表)
// ============================================================================

fn argmax_bench(res: &mut TestRes, n: usize) {
    let entry = "owl_argmax_f32idx_f16";
    let src = <owl_kernels::ops::elems::ArgmaxF32IdxF16 as owl_kernels::spec::KernelSpec>::SRC;
    let mut exec = Exec::new();
    let f = exec.load_aot(res, src, entry, OpId(owl_kernels::spec::NATIVE)).expect("资产");

    let prod = geom::argmax();
    println!("\n═══ {entry}  n={n}");
    println!("    生产几何(driver): grid={:?} block={:?}", prod.shape.grid, prod.shape.block);

    let mut x_h = lcg_f16(n, 0xF1);
    // ⚠️ 期望索引必须**从上传的输入算**(不能拿"我设的"当答案):
    // `lcg_f16` 会产出多个**恰好相等**的最大值(幅值上限恰为 4.0),而本核的
    // 平局律 = **取最小索引** —— 拿手填的索引当判据会误报(2026-10-19 现场:
    // 台子先报了 ✗,查下去是**判据错**而不是核错)。
    x_h[n / 3] = f16_bits(4.0);
    let want_idx = x_h
        .iter()
        .enumerate()
        .fold((0usize, f16_to_f32(x_h[0])), |a, (i, h)| {
            let v = f16_to_f32(*h);
            if v > a.1 { (i, v) } else { a }
        })
        .0;
    let bx = res.blk(n * 2, "x");
    res.up(bx, as_bytes_f16(&x_h));
    let px = res.ptr(bx);
    let bo = res.blk(4, "out");
    let po = res.ptr(bo);

    println!("  {:>22} {:>10} {:>10}  {}", "臂(block)", "M=1 µs", "M=32 µs", "命中");
    // ⚠️ 只测试 ≤ 256 的块:本核 `__shared__` 静态数组按 `ARGMAX_BLOCK=256` 定长,
    // 块 > 256 = **越界写**(实测 block=1024 触发 `CUDA_ERROR_ILLEGAL_ADDRESS`)。
    // ⇒ "加块"不是本核的可选刀(要改必须同改 ARGMAX_BLOCK/改动态 smem);
    // 本核价值仅 27µs/步(0.1%),暂不动。
    for blk in [128u32, 256] {
        let grid = prod.shape.grid;
        let block = (blk, 1, 1);
        let vals = [
            LaunchVal::Ptr(px),
            LaunchVal::I32(n as i32),
            LaunchVal::I32(0),
            LaunchVal::Ptr(po),
        ];
        fire(res, &mut exec, &f, entry, grid, block, 0, &vals);
        res.sync();
        let mut got = [0u8; 4];
        res.down(bo, &mut got);
        let idx = f32::from_le_bytes(got) as i64;
        let us1 = time_us(res, &mut exec, &f, entry, grid, block, 0, &vals, 200, 1);
        let us32 = time_us(res, &mut exec, &f, entry, grid, block, 0, &vals, 10, 32);
        let mark = if block == prod.shape.block { "*" } else { " " };
        println!(
            "  {mark}{:>21} {us1:>10.2} {us32:>10.2}  {}",
            format!("block={blk}"),
            if idx == want_idx as i64 { format!("✓ (期望 {want_idx})") } else { format!("✗{idx} ≠ {want_idx}") }
        );
    }
}

// ============================================================================
// §6b bf16 臂数值门(与 f16 同款改动;bf16 只在 DFlash2 草稿层走 ⇒ 必须**单独**验:
//     spec-decode 档不在本轮 tp 门覆盖面内)
// ============================================================================

/// bf16 侧参考用 RNE(与核内 `__float2bfloat16` 同律)
fn f32_to_bf16_rne(v: f32) -> u16 {
    let bits = v.to_bits();
    let round = ((bits >> 16) & 1) + 0x7FFF;
    (bits.wrapping_add(round) >> 16) as u16
}
fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

fn bf16_parity(res: &mut TestRes, n: usize, rows: usize) {
    use owl_kernels::ops::oracles::lcg_bf16;
    println!("\n═══ bf16 臂(block={ROW_REDUCE_BLOCK};与 f16 同几何同核体)\n");
    let (eps, w_off) = (1e-6f32, 0i32);

    // ── rmsnorm_bf16 ──
    {
        let entry = "owl_rmsnorm_bf16";
        let src = <owl_kernels::ops::ln::RmsnormBf16 as owl_kernels::spec::KernelSpec>::SRC;
        let mut exec = Exec::new();
        let f = exec.load_aot(res, src, entry, OpId(owl_kernels::spec::NATIVE)).expect("资产");
        let prod = geom::rmsnorm(DType::BF16, &[vec![rows, n], vec![n]]);
        let x_h = lcg_bf16(rows * n, 0x51);
        let w_h: Vec<u16> =
            lcg_bf16(n, 0x52).iter().map(|h| f32_to_bf16_rne(bf16_to_f32(*h) * 0.25 + 1.0)).collect();
        let (bx, bw, bo) = (res.blk(rows * n * 2, "x"), res.blk(n * 2, "w"), res.blk(rows * n * 2, "o"));
        res.up(bx, as_bytes_f16(&x_h));
        res.up(bw, as_bytes_f16(&w_h));
        let (px, pw, po) = (res.ptr(bx), res.ptr(bw), res.ptr(bo));
        let mut want = vec![0.0f32; rows * n];
        for r in 0..rows {
            let row: Vec<f32> = x_h[r * n..(r + 1) * n].iter().map(|h| bf16_to_f32(*h)).collect();
            let ss: f64 = row.iter().map(|v| (*v as f64) * (*v as f64)).sum();
            let inv = 1.0f32 / ((ss / n as f64) as f32 + eps).sqrt();
            for i in 0..n {
                want[r * n + i] = row[i] * inv * bf16_to_f32(w_h[i]);
            }
        }
        let vals = [
            LaunchVal::Ptr(px),
            LaunchVal::Ptr(pw),
            LaunchVal::Ptr(po),
            LaunchVal::I32(n as i32),
            LaunchVal::F32(eps),
            LaunchVal::I32(w_off),
        ];
        fire(res, &mut exec, &f, entry, prod.shape.grid, prod.shape.block, prod.shape.smem, &vals);
        res.sync();
        let mut got = vec![0u8; rows * n * 2];
        res.down(bo, &mut got);
        let (worst, at) = rel_worst_with(&f16_from_bytes(&got), &want, bf16_to_f32);
        println!("  {entry:<32} grid={:?} block={:?} smem={} → {}", prod.shape.grid, prod.shape.block, prod.shape.smem, verdict_ok(worst, at));
    }

    // ── fused_add_rmsnorm_bf16(含 residual 逐位)──
    {
        let entry = "owl_fused_add_rmsnorm_bf16";
        let src =
            <owl_kernels::ops::ln::FusedAddRmsnormBf16 as owl_kernels::spec::KernelSpec>::SRC;
        let mut exec = Exec::new();
        let f = exec.load_aot(res, src, entry, OpId(owl_kernels::spec::NATIVE)).expect("资产");
        let prod = geom_ln::fused_add_rmsnorm(DType::BF16, rows, n);
        let m_h = lcg_bf16(rows * n, 0x61);
        let r_h = lcg_bf16(rows * n, 0x62);
        let w_h: Vec<u16> =
            lcg_bf16(n, 0x63).iter().map(|h| f32_to_bf16_rne(bf16_to_f32(*h) * 0.25 + 1.0)).collect();
        let (bm, br, bw, bo) = (
            res.blk(rows * n * 2, "m"),
            res.blk(rows * n * 2, "r"),
            res.blk(n * 2, "w"),
            res.blk(rows * n * 2, "o"),
        );
        res.up(bm, as_bytes_f16(&m_h));
        res.up(br, as_bytes_f16(&r_h));
        res.up(bw, as_bytes_f16(&w_h));
        let (pm, pr, pw, po) = (res.ptr(bm), res.ptr(br), res.ptr(bw), res.ptr(bo));
        let mut want_r = vec![0u16; rows * n];
        let mut want = vec![0.0f32; rows * n];
        for r in 0..rows {
            let mut acc = vec![0.0f32; n];
            for i in 0..n {
                let v = bf16_to_f32(m_h[r * n + i]) + bf16_to_f32(r_h[r * n + i]);
                let h = f32_to_bf16_rne(v);
                want_r[r * n + i] = h;
                acc[i] = v;
            }
            let ss: f64 = acc.iter().map(|v| (*v as f64) * (*v as f64)).sum();
            let inv = 1.0f32 / ((ss / n as f64) as f32 + eps).sqrt();
            for i in 0..n {
                want[r * n + i] = bf16_to_f32(want_r[r * n + i]) * inv * bf16_to_f32(w_h[i]);
            }
        }
        let vals = [
            LaunchVal::Ptr(pm),
            LaunchVal::Ptr(pr),
            LaunchVal::Ptr(pw),
            LaunchVal::F32(eps),
            LaunchVal::U64(n as u64),
            LaunchVal::I32(w_off),
            LaunchVal::Ptr(po),
        ];
        fire(res, &mut exec, &f, entry, prod.shape.grid, prod.shape.block, prod.shape.smem, &vals);
        res.sync();
        let mut got = vec![0u8; rows * n * 2];
        res.down(bo, &mut got);
        let (worst, at) = rel_worst_with(&f16_from_bytes(&got), &want, bf16_to_f32);
        let mut gr = vec![0u8; rows * n * 2];
        res.down(br, &mut gr);
        let nbad = f16_from_bytes(&gr).iter().zip(want_r.iter()).filter(|(g, w)| g != w).count();
        println!(
            "  {entry:<32} grid={:?} block={:?} smem={} → {} · residual {}",
            prod.shape.grid,
            prod.shape.block,
            prod.shape.smem,
            verdict_ok(worst, at),
            if nbad == 0 { "✓ 逐位".to_string() } else { format!("✗{nbad}") }
        );
    }
}

// ============================================================================
// §5b embed int8(P8;q/s → f16,host f32 参考)
// ============================================================================

fn embed_i8_parity(res: &mut TestRes) {
    let entry = "owl_embed_i8_f16";
    let src = <owl_kernels::ops::elems::EmbedI8F16 as owl_kernels::spec::KernelSpec>::SRC;
    let mut exec = Exec::new();
    let f = exec.load_aot(res, src, entry, OpId(owl_kernels::spec::NATIVE)).expect("资产");

    let (vocab, d_dim, tokens) = (64usize, 512usize, 3usize);
    // int8 位型(含负值);per-row scale 正数
    let qb: Vec<u8> = (0..vocab * d_dim)
        .map(|i| ((((i * 37 + 11) % 255) as i32) - 127) as i8 as u8)
        .collect();
    let scales: Vec<f32> = (0..vocab).map(|i| 0.001 + i as f32 * 0.0007).collect();
    let ids: Vec<f32> = (0..tokens).map(|t| ((t * 17 + 3) % vocab) as f32).collect();

    let bq = res.blk(qb.len(), "q");
    res.up(bq, &qb);
    let bs = res.blk(scales.len() * 4, "s");
    res.up(bs, as_bytes_f32(&scales));
    let bi = res.blk(tokens * 4, "ids");
    res.up(bi, as_bytes_f32(&ids));
    let bo = res.blk(tokens * d_dim * 2, "out");

    let geom = geom_e::embed_i8(tokens);
    let vals = [
        LaunchVal::Ptr(res.ptr(bq)),
        LaunchVal::Ptr(res.ptr(bs)),
        LaunchVal::Ptr(res.ptr(bi)),
        LaunchVal::U64(d_dim as u64),
        LaunchVal::Ptr(res.ptr(bo)),
    ];
    fire(res, &mut exec, &f, entry, geom.shape.grid, geom.shape.block, 0, &vals);
    res.sync();
    let mut got = vec![0u8; tokens * d_dim * 2];
    res.down(bo, &mut got);
    let got = f16_from_bytes(&got);

    let mut bad = 0usize;
    let mut worst = 0f32;
    for t in 0..tokens {
        let id = ids[t] as usize;
        for i in 0..d_dim {
            let qi = qb[id * d_dim + i] as i8 as f32;
            let want = half::f16::from_f32(qi * scales[id]);
            let g = half::f16::from_bits(got[t * d_dim + i]);
            if g.to_bits() != want.to_bits() {
                bad += 1;
                worst = worst.max((g.to_f32() - want.to_f32()).abs());
            }
        }
    }
    println!(
        "  {entry:<32} grid={:?} block={:?} → {} (bad={bad}, worst={worst:e})",
        geom.shape.grid, geom.shape.block, if bad == 0 { "✓ 逐位" } else { "✗" }
    );
    assert_eq!(bad, 0, "embed_i8 与 host f32 参考不一致(bad={bad}, worst={worst:e})");
}

#[test]
fn embed_i8_parity_gate() {
    let Some(mut res) = TestRes::new() else {
        owl_shared::testkit::report::skip("需 OWL_TEST_DEVICE=<CUDA ordinal>");
        return;
    };
    embed_i8_parity(&mut res);
}

// ============================================================================
// §7 台子
// ============================================================================

#[test]
fn small_kernel_bench_and_parity() {
    let Some(mut res) = TestRes::new() else {
        owl_shared::testkit::report::skip("需 OWL_TEST_DEVICE=<CUDA ordinal>");
        return;
    };
    println!("[bench] CUDA ordinal = {}(`*` = 生产几何)", res.ctx.ordinal());
    rmsnorm_bench(&mut res, 5120, 1);
    fused_add_rmsnorm_bench(&mut res, 5120, 1);
    bf16_parity(&mut res, 5120, 1);
    embed_bench(&mut res, 248320, 5120, 1);
    argmax_bench(&mut res, 248320);
    println!("\n[bench] 判据:数值 ≤ {TOL:.0e} / 逐位相等 / argmax 命中;时延看 M=32 档");
}

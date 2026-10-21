//! **装载期 int8 量化台**(`owl_quant_i8_row_w_{scale,q}_bf16`)—— P1.5-a 判据台。
//!
//! # 判据(唯一一条,硬)
//!
//! **与 host 参考逐元素相同**(不是容差):同一份 bf16 权重,设备两发 vs 主机
//! `rtn_i8_row_sym`(`s = max|w|/127` → `q = RN-even(w/s)` 截 ±127)必须
//! `scale` 位型相同、`q` 字节全等。任何差异都意味着"装载期与投影期两套语义"。
//!
//! ⚠️ 两个已知的"对不上"来源(本台子就是为抓它们):
//! 1. `x * (1/s)` vs `x / s` —— 舍入分家;两侧统一用**除法**;
//! 2. `round()`(half-away)vs `__float2int_rn`(RN-even) —— 两侧统一 **RN-even**。
//!
//! 跑法:`OWL_TEST_DEVICE=2 cargo test -p owl-kernels --features device \
//!   --test quant_i8_row_w_bench -- --nocapture --test-threads=1`
#![cfg(feature = "device")]

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr};
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
        panic!("launch({entry}): {e:?}");
    }
}

fn rnd(seed: u32, i: usize) -> f32 {
    let x = seed
        .wrapping_mul(0x9E37_79B9)
        .wrapping_add((i as u32).wrapping_mul(0x85EB_CA6B));
    ((x >> 8) & 0x00FF_FFFF) as f32 / 16777216.0
}

/// **host 参考**(与核逐式同源 —— 见头注的两条纪律)
fn rtn_i8_row_sym(w: &[f32], rows: usize, cols: usize) -> (Vec<f32>, Vec<i8>) {
    let mut sc = vec![0f32; rows];
    let mut q = vec![0i8; rows * cols];
    for r in 0..rows {
        let row = &w[r * cols..(r + 1) * cols];
        let amax = row.iter().fold(0f32, |a, v| a.max(v.abs()));
        let s = if amax > 0.0 { amax / 127.0 } else { 1.0 };
        sc[r] = s;
        for (i, v) in row.iter().enumerate() {
            // RN-even + 截断(与 __float2int_rn 同律)
            let qv = (v / s).round_ties_even().clamp(-127.0, 127.0);
            q[r * cols + i] = qv as i8;
        }
    }
    (sc, q)
}

fn run_case(res: &mut TestRes, rows: usize, cols: usize) {
    // 轻尾分布(贴近词表行:多数小值、少量大值)—— bf16 位型先量化成 bf16 再回 f32,
    // 保证"host 看到的值"与设备读到的**是同一批数"
    let mut wb = vec![0u16; rows * cols];
    let mut wf = vec![0f32; rows * cols];
    for i in 0..rows * cols {
        let u = rnd(0x91, i);
        let v = (u - 0.5) * 0.08 * if u > 0.995 { 5.0 } else { 1.0 };
        // bf16 位型 = f32 高 16 位(截断);host 与设备看到的是**同一批数**(判据前提)
        let b16 = (v.to_bits() >> 16) as u16;
        wb[i] = b16;
        wf[i] = f32::from_bits((b16 as u32) << 16);
    }
    // 全零行 / 常量行(边界:amax=0 ⇒ s=1)
    for k in 0..cols {
        wb[k] = 0;
        wf[k] = 0.0;
        let q16 = (0.25f32.to_bits() >> 16) as u16;
        wb[cols + k] = q16;
        wf[cols + k] = f32::from_bits((q16 as u32) << 16);
    }
    let (want_sc, want_q) = rtn_i8_row_sym(&wf, rows, cols);

    let bw = res.blk(rows * cols * 2, "w");
    let bs = res.blk(rows * 4, "scale");
    let bq = res.blk(rows * cols, "q");
    res.up(bw, &wb.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
    let (pw, ps, pq) = (res.ptr(bw), res.ptr(bs), res.ptr(bq));

    let mut exec = Exec::new();
    let src = "cu/load/quant_i8_row_w.cu";
    let f_scale = exec.load_aot(res, src, "owl_quant_i8_row_w_scale_bf16", OpId(owl_kernels::spec::NATIVE)).expect("资产");
    let f_q = exec.load_aot(res, src, "owl_quant_i8_row_w_q_bf16", OpId(owl_kernels::spec::NATIVE)).expect("资产");
    let g = owl_kernels::driver::elems::quant_i8_row_w_grid(rows);
    let (grid, block, smem) = (g.grid, g.block, g.smem);
    let v_scale = [LaunchVal::Ptr(pw), LaunchVal::I32(rows as i32), LaunchVal::I32(cols as i32), LaunchVal::Ptr(ps)];
    let v_q = [
        LaunchVal::Ptr(pw),
        LaunchVal::Ptr(ps),
        LaunchVal::I32(rows as i32),
        LaunchVal::I32(cols as i32),
        LaunchVal::Ptr(pq),
    ];

    fire(res, &mut exec, &f_scale, "owl_quant_i8_row_w_scale_bf16", grid, block, smem, &v_scale);
    fire(res, &mut exec, &f_q, "owl_quant_i8_row_w_q_bf16", grid, block, smem, &v_q);
    res.sync();

    let mut got_sc = vec![0u8; rows * 4];
    let mut got_q = vec![0u8; rows * cols];
    res.down(bs, &mut got_sc);
    res.down(bq, &mut got_q);
    // scale:位型比对(f32 全等)
    let mut bad_sc = 0usize;
    for r in 0..rows {
        let g = f32::from_le_bytes(got_sc[r * 4..r * 4 + 4].try_into().unwrap());
        if g.to_bits() != want_sc[r].to_bits() {
            bad_sc += 1;
        }
    }
    // q:字节比对
    let bad_q = got_q
        .iter()
        .zip(want_q.iter())
        .filter(|(g, w)| **g as i8 != **w)
        .count();

    // 计时(两发合计;装载期一次)
    let t0 = Instant::now();
    for _ in 0..5 {
        fire(res, &mut exec, &f_scale, "owl_quant_i8_row_w_scale_bf16", grid, block, smem, &v_scale);
        fire(res, &mut exec, &f_q, "owl_quant_i8_row_w_q_bf16", grid, block, smem, &v_q);
    }
    res.sync();
    let us = t0.elapsed().as_secs_f64() * 1e6 / 5.0;
    let bytes = (rows * cols * 2 + rows * cols + rows * 4) as f64;
    println!(
        "  rows={rows:<7} cols={cols:<5}  两发合计 {us:>8.1} µs({:>6.1} GB/s)  \
         scale 不同 {bad_sc} / {rows} · q 不同 {bad_q} / {}  {}",
        bytes / (us * 1e-6) / 1e9,
        rows * cols,
        if bad_sc == 0 && bad_q == 0 { "⇒ 逐元素相同 ✓" } else { "⇒ ✗ 与 host RTN 不符" }
    );
    assert_eq!((bad_sc, bad_q), (0, 0), "装载期量化与 host 参考不符(见头注两条纪律)");
}

#[test]
fn quant_i8_row_w_bench() {
    let Some(mut res) = TestRes::new() else {
        eprintln!("未设 OWL_TEST_DEVICE ⇒ skip");
        return;
    };
    println!("\n═══ owl_quant_i8_row_w_scale/q_bf16(per-row 对称 int8;host RTN 逐元素对拍)");
    // 生产:词表切后 124160 行 × 5120;另给小档看规模律
    for (rows, cols) in [(124160usize, 5120usize), (8192, 5120), (124160, 1024)] {
        run_case(&mut res, rows, cols);
    }
}

//! feature 门:本测试依赖 device(Exec/DeviceRes);无 feature 时整文件置空。
#![cfg(feature = "device")]

//! cublas runtime 隔离测试(M4 cutover 验收;已知矩阵双路对拍)。
//! 路径:CublasRuntime::run(TestRes) vs host 朴素 matmul。

use owl_kernels::server::cublas::CublasRuntime;
use owl_kernels::contract::{Bytes, LaunchMsg, OpError};
use owl_kernels::device::{DeviceRes, Exec, LaunchVal, ScratchBuf};
use owl_kernels::registry::{FamilyRuntime, RunEnv};
use std::collections::HashMap;
use std::sync::Arc;
use cudarc::driver::{CudaSlice, DevicePtr};

struct TestRes {
    ctx: Arc<cudarc::driver::CudaContext>,
    stream: Arc<cudarc::driver::CudaStream>,
    blocks: HashMap<u64, CudaSlice<u8>>,
    next_id: u64,
}

impl TestRes {
    fn new(dev: usize) -> Self {
        let ctx = cudarc::driver::CudaContext::new(dev).expect("ctx");
        ctx.bind_to_thread().expect("bind");
        let stream = ctx.default_stream();
        Self { ctx, stream, blocks: HashMap::new(), next_id: 1 }
    }
    fn upload(&mut self, bytes: &[u8]) -> Bytes {
        let id = self.next_id;
        self.next_id += 1;
        let mut d = self.stream.alloc_zeros::<u8>(bytes.len()).expect("alloc");
        self.stream.memcpy_htod(bytes, &mut d).expect("htod");
        self.blocks.insert(id, d);
        Bytes { id, len: 0 }
    }
}

impl DeviceRes for TestRes {
    fn context(&self) -> Result<Arc<cudarc::driver::CudaContext>, OpError> {
        Ok(self.ctx.clone())
    }
    fn stream(&self) -> Result<Arc<cudarc::driver::CudaStream>, OpError> {
        Ok(self.stream.clone())
    }
    fn resolve(&self, b: &Bytes) -> Result<u64, OpError> {
        self.blocks
            .get(&b.id)
            .map(|s| s.device_ptr(&self.stream).0)
            .ok_or_else(|| OpError::Contract { op: "t".into(), field: "resolve", expect: "在册".into(), got: b.id.to_string() })
    }
    fn alloc(&mut self, bytes: usize, _tag: &'static str) -> Result<ScratchBuf, OpError> {
        let d = self.stream.alloc_zeros::<u8>(bytes).expect("alloc");
        let ptr = d.device_ptr(&self.stream).0;
        self.blocks.insert(self.next_id, d);
        self.next_id += 1;
        Ok(ScratchBuf { ptr, bytes })
    }
    fn capturing(&self) -> Result<bool, OpError> {
        Ok(false)
    }
    fn record_capture(&mut self, _note: owl_kernels::device::LaunchNote) -> Result<(), OpError> {
        Ok(())
    }
    fn upload(&mut self, dst: u64, src: &[u8]) -> Result<(), OpError> {
        use cudarc::driver::DevicePtr;
        let target = self
            .blocks
            .iter()
            .find(|(_, s)| s.device_ptr(&self.stream).0 == dst)
            .map(|(id, _)| *id);
        if let Some(id) = target {
            let mut view = self.blocks.get_mut(&id).expect("在册");
            cudarc::driver::CudaStream::memcpy_htod(&self.stream, src, view)
                .expect("htod");
            return Ok(());
        }
        Err(OpError::Contract { op: "t".into(), field: "upload", expect: "指针在册".into(), got: format!("{dst:#x}") })
    }
    fn device_ordinal(&self) -> Result<i32, OpError> {
        Ok(0)
    }
}

#[test]
fn cublas_runtime_matches_host_matmul() {
    if std::env::var_os("OWL_TEST_DEVICE").is_none() {
        eprintln!("skip");
        return;
    }
    let dev: usize = std::env::var("OWL_TEST_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let mut res = TestRes::new(dev);
    let mut exec = Exec::new();
    let mut rt = CublasRuntime::default();
    {
        let mut env = RunEnv::new(&mut res, &mut exec);
        rt.init(&mut env).expect("init");
    }

    // C[2,3] = A[2,6] × W[3,6]^T(nt=true;owl Linear 形态)
    let (t, hidden, n_out) = (2usize, 6usize, 3usize);
    let a: Vec<f32> = (0..t * hidden).map(|i| ((i as f32 + 1.0) * 0.21).sin() * 0.5).collect();
    let w: Vec<f32> = (0..n_out * hidden).map(|i| ((i as f32 + 4.0) * 0.17).sin() * 0.7).collect();
    let mut want = vec![0f32; t * n_out];
    for i in 0..t {
        for j in 0..n_out {
            let mut acc = 0f32;
            for kk in 0..hidden {
                acc += a[i * hidden + kk] * w[j * hidden + kk];
            }
            want[i * n_out + j] = half::f16::from_f32(acc).to_f32();
        }
    }
    let f16b = |v: &[f32]| -> Vec<u8> {
        v.iter().flat_map(|f| half::f16::from_f32(*f).to_le_bytes()).collect()
    };

    let a_b = res.upload(&f16b(&a));
    let w_b = res.upload(&f16b(&w));
    let (out_id, _out) = {
        let d = res.stream.alloc_zeros::<u8>(t * n_out * 2).expect("out");
        let ptr = d.device_ptr(&res.stream).0;
        res.blocks.insert(res.next_id, d);
        let id = res.next_id;
        res.next_id += 1;
        (id, ptr)
    };
    let out_b = Bytes { id: out_id, len: t * n_out };

    let scale = (1.0f64 / hidden as f64).to_bits() as u64;
    let msg = LaunchMsg {
        kernel: owl_kernels::contract::KernelSpec {
            name: owl_kernels::client::cublas::GEMM_F16.to_string(),
            source: String::new(),
        },
        args: vec![
            owl_kernels::contract::Arg::Block { id: a_b.id },
            owl_kernels::contract::Arg::Block { id: w_b.id },
            owl_kernels::contract::Arg::Block { id: out_b.id },
            owl_kernels::contract::Arg::U64(n_out as u64), // m = n_out(权重行)
            owl_kernels::contract::Arg::U64(hidden as u64), // k
            owl_kernels::contract::Arg::U64(t as u64), // n = tokens
            owl_kernels::contract::Arg::U64(1),        // nt = true
        ],
        grid: (0, 0, 0),
        block: (0, 0, 0),
        shared_mem: 0,
        out_elems: t * n_out,
    };
    let _ = scale;
    {
        let mut env = RunEnv::new(&mut res, &mut exec);
        rt.run(&msg, &mut env).expect("cublas run");
    }
    res.stream.synchronize().expect("sync");

    for (id, s) in &res.blocks {
        use cudarc::driver::DevicePtr;
        eprintln!("[test ptr] id={id} ptr={:#x} bytes={}", s.device_ptr(&res.stream).0, s.len());
    }
    // 输入完整性对照(upload 后 vs run 后)
    let a_after = res.take_bytes_f16_test(a_b.id);
    let w_after = res.take_bytes_f16_test(w_b.id);
    eprintln!("a after-run = {a_after:?}(期望 = a 原值)");
    eprintln!("w after-run = {w_after:?}(期望 = w 原值)");
    let got = res.take_bytes_f16_test(out_id);
    // 对照臂:直接 OwlCublas::gemm_f16(旧 handler 同款调用)同缓冲重跑
    {
        let stream = res.stream.clone();
        let blas = owl_kernels::family::cublas::OwlCublas::new(stream.clone()).expect("direct blas");
        let a_p = res.resolve(&a_b).expect("a");
        let b_p = res.resolve(&w_b).expect("b");
        let o_p = res.resolve(&out_b).expect("o");
        blas.gemm_f16(a_p, b_p, o_p, n_out, hidden, t, true).expect("direct gemm");
        stream.synchronize().expect("sync");
    }
    let got_direct = res.take_bytes_f16_test(out_id);
    eprintln!("runtime got={got:?}");
    eprintln!("direct  got={got_direct:?}");
    // runtime 与 direct 互证(同缓冲同参 → 应逐位一致)
    for (a, b) in got.iter().zip(&got_direct) {
        assert!((a - b).abs() < 1e-6, "runtime vs direct 分歧: {a} vs {b}");
    }
    let mut worst = 0f32;
    for (g, w) in got.iter().zip(&want) {
        worst = worst.max((g - w).abs());
    }
    eprintln!("[cublas 隔离] max|Δ|={worst:.4e}(want={want:?} got={got:?})");
    assert!(worst < 5e-2, "cublas runtime 对拍超差 {worst}");
}

impl TestRes {
    pub fn take_bytes_f16_test(&self, id: u64) -> Vec<f32> {
        let s = self.blocks.get(&id).expect("在册");
        let host = self.stream.clone_dtoh(s).expect("dtoh");
        host.chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect()
    }
}
